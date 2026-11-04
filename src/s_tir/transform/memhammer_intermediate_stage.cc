/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/support/gpu_thread_tag.h>

#include "memhammer_rewrite_rule.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

Stmt CopyLoopChain(const std::vector<const ForObj*> loops, const Stmt& inner_body, int ith = -1,
                   ffi::Optional<For>* ith_loop = nullptr) {
  Stmt ret = inner_body;
  for (int i = static_cast<int>(loops.size() - 1); i >= 0; i--) {
    ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*loops[i]);
    new_loop->body = ret;
    ret = For(new_loop);
    if (ith == i) {
      *ith_loop = ret.as_or_throw<For>();
    }
  }
  return ret;
}

/*!
 * \brief lift all the thread binding loops
 * \param stmt the top loop
 * \return a pair. The first is the transformed stmt.
 *         The second is the lowest thread binding loop.
 */
std::pair<Stmt, ffi::Optional<For>> LiftThreadBindingLoops(Stmt stmt) {
  std::vector<const ForObj*> normal_loops;
  std::vector<const ForObj*> thread_binding_loops;
  Stmt body = stmt;
  while (const ForObj* loop = body.as<ForObj>()) {
    if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
      thread_binding_loops.push_back(loop);
    } else {
      normal_loops.push_back(loop);
    }
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  body = CopyLoopChain(normal_loops, body);
  ffi::Optional<For> compute_location;
  body = CopyLoopChain(thread_binding_loops, body,
                       static_cast<int>(thread_binding_loops.size()) - 1, &compute_location);

  return std::make_pair(body, compute_location);
}

/*!
 * \brief Analyze the access pattern for tensor rank promotion.
 * Rank promotion is a transformation that reshapes the tensor
 * but doesn't change its underlying data layout.
 * After the reshape, we expect that all dimensions of the access indices
 * will be in the form of floormod(floordiv(x, a), b).
 * Rank promotion removes strided access, thus enabling further tensor compacting
 */
class IndexPatternFinder : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  IndexPatternFinder(const ffi::Map<Var, Range>& var_range, ffi::Array<PrimExpr>* resulting_index)
      : var_range_(var_range), resulting_index_(resulting_index) {}
  struct Operator {
    enum class OpKind { Mul, FloorDiv, FloorMod };
    OpKind kind;
    int64_t operand;
  };

  /*!
   * \brief Calculate the new tensor shape after rank promotion.
   * For each dimension of original shape, it will be compacted.
   * \param indices The access indices of the tensor
   * \param var_range The iter range of the vars in the indices
   * \param rewrite_indices The access indices after rank promotion
   * \return The new tensor shape after rank promotion.
   */
  static ffi::Array<PrimExpr> getRankPromotedShape(ffi::Array<PrimExpr> indices,
                                                   const ffi::Map<Var, Range>& var_range,
                                                   ffi::Array<PrimExpr>* rewrite_indices) {
    ffi::Map<Var, sym::IntSet> var_dom = sym::AsIntSet(var_range);
    ffi::Array<PrimExpr> new_shape;
    for (const PrimExpr& expr : indices) {
      ffi::Array<PrimExpr> indices_dim;
      auto extractor = ffi::make_object<IndexPatternFinder>(var_range, &indices_dim);
      extractor->Visit(expr);
      if (!extractor->success_) {
        return {};
      }
      ffi::Array<PrimExpr> access_shape = extractor->access_shape_;
      PrimExpr product_shape = 1;
      for (PrimExpr e : access_shape) {
        product_shape *= e;
      }
      new_shape.push_back(product_shape);
      PrimExpr flatten_index = 0;
      for (int i = 0; i < static_cast<int>(access_shape.size()); i++) {
        flatten_index = flatten_index * access_shape[i] + indices_dim[i];
      }
      rewrite_indices->push_back(flatten_index);
    }
    return new_shape;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
    if (!success_) {
      return std::nullopt;
    }
    if (ffi::Optional<Range> range = var_range_.Get(ffi::GetRef<Var>(op))) {
      PrimExpr index = ffi::GetRef<Var>(op).as_or_throw<PrimExpr>();
      int64_t max = static_cast<int64_t>(range.value()->extent.as<IntImmObj>()->value);
      int64_t extent = max;
      for (int i = static_cast<int>(operator_stack.size()) - 1; i >= 0; i--) {
        Operator o = operator_stack[i];
        switch (o.kind) {
          case Operator::OpKind::Mul:
            max *= o.operand;
            index = index * IntImm::Int32(o.operand);
            break;
          case Operator::OpKind::FloorDiv:
            if (max % o.operand != 0 && o.operand % max != 0) {
              success_ = false;
              return std::nullopt;
            }
            max = max / o.operand;
            if (extent > max) {
              extent = std::max(static_cast<int64_t>(1), max);
            }
            if (max % extent != 0) {
              success_ = false;
              return std::nullopt;
            }
            index = floordiv(index, IntImm::Int32(o.operand));
            break;
          case Operator::OpKind::FloorMod:
            int64_t step = max / extent;
            if (step % o.operand != 0 && o.operand % step != 0) {
              success_ = false;
              return std::nullopt;
            }
            if (step % o.operand == 0) {
              extent = 1;
              max = 0;
            } else {
              extent = std::max(static_cast<int64_t>(1), std::min(extent, o.operand / step));
              max = extent * step;
            }
            index = floormod(index, IntImm::Int32(o.operand));
        }
      }
      if (extent > 1) {
        TVM_FFI_ICHECK(max % extent == 0);
        access_shape_.push_back(IntImm::Int32(extent));
        resulting_index_->push_back(floordiv(index, max / extent));
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const FloorDivObj* op) final {
    int64_t b = static_cast<int64_t>(op->b.as<IntImmObj>()->value);
    operator_stack.push_back(Operator{Operator::OpKind::FloorDiv, b});
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    operator_stack.pop_back();
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const FloorModObj* op) final {
    int64_t b = static_cast<int64_t>(op->b.as<IntImmObj>()->value);
    operator_stack.push_back(Operator{Operator::OpKind::FloorMod, b});
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    operator_stack.pop_back();
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const MulObj* op) final {
    int64_t b = static_cast<int64_t>(op->b.as<IntImmObj>()->value);
    operator_stack.push_back(Operator{Operator::OpKind::Mul, b});
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    operator_stack.pop_back();
    return std::nullopt;
  }

  ffi::Map<Var, Range> var_range_;
  ffi::Array<PrimExpr> access_shape_;
  ffi::Array<PrimExpr>* resulting_index_;
  std::vector<Operator> operator_stack;
  bool success_ = true;
};

class TensorLoadReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  TensorLoadReplacer(const TensorVar& tgt_tensor, const TensorLoad& new_tensor_load)
      : tgt_tensor_(tgt_tensor), new_tensor_load_(new_tensor_load) {}

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) {
    if (op->source.as_or_throw<tvm::tirx::TensorVar>().same_as(tgt_tensor_)) {
      return new_tensor_load_;
    }
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }

 private:
  TensorVar tgt_tensor_;
  TensorLoad new_tensor_load_;
};

/*!
 * \brief Insert a cache stage to the compute location
 * \param stmt the stmt
 * \param is_write_cache whether to write a read cache or write cache
 * \param storage_scope the storage scope of the new cache
 * \param compute_location the compute location.
 * \param outer_loops the outer loops of this stmt
 * \param alloc_tensor the new cache block
 * \return a pair. The first is the stmt after transformation.
 *         The second is the SeqStmt that contains 2 stages (one original and another inserted).
 */
std::pair<Stmt, SeqStmt> InsertCacheStage(Stmt stmt, bool is_write_cache, ffi::String storage_scope,
                                          ffi::Optional<For> compute_location,
                                          const ffi::Array<For>& outer_loops,
                                          TensorVar* alloc_tensor) {
  static const auto scope_wmma_accumulator = runtime::MemoryScope::Get("wmma.accumulator");
  static const auto scope_mma_matrixC = runtime::MemoryScope::Get("mma.matrixC");

  Stmt body = stmt;
  std::vector<const ForObj*> loops;
  std::vector<const ForObj*> loops_under_compute_location;
  std::vector<const ForObj*> relaxed_thread_loops;
  bool need_relax = !compute_location.has_value();
  ffi::Map<Var, Range> var_range;
  PrimExpr vector_bytes = -1;
  // Step 1. Perform rank promotion on the tensor access, turning a strided-changing dimension into
  // several contiguous-changing dimensions
  // Step 1.1 collect loop var range for rank promotion
  while (const ForObj* loop = body.as<ForObj>()) {
    if (need_relax) {
      var_range.Set(loop->var, Range(loop->begin, loop->end));
      loops_under_compute_location.push_back(loop);
    } else {
      loops.push_back(loop);
    }
    if (compute_location.has_value() && loop == compute_location.value().get()) {
      need_relax = true;
    }
    if (loop->kind == ForKind::kVectorized) {
      vector_bytes = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
    }
    body = loop->body->size() == 1 ? loop->body->seq[0] : loop->body;
  }
  ffi::Optional<PrimExpr> predicate;
  if (const auto* op = body.as<IfObj>()) {
    // the predicate is generated by coalescing
    predicate = op->condition;
    TVM_FFI_ICHECK_EQ(op->then_case->size(), 1);
    body = op->then_case->seq[0];
  }
  const auto memory_scope = runtime::MemoryScope::Get(storage_scope);
  for (const For& loop : outer_loops) {
    if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
      const ffi::String thread_tag = tvm::s_tir::GetThreadBinding(loop).value();
      auto thread_scope = ParseThreadBindingTag(thread_tag);
      if (CanRelaxStorageUnderThread(memory_scope, thread_scope)) {
        if (is_write_cache && thread_scope.dim_index == 0) {
          // writing C_reindex_m16n8k8_matrixC_shared_dyn is warp execution
          continue;
        }
        var_range.Set(loop->var, Range(loop->begin, loop->end));
        relaxed_thread_loops.push_back(loop.get());
      }
    }
  }

  sym::Analyzer analyzer;
  const TensorLoadObj* target_tensor_load = nullptr;
  if (is_write_cache) {
    auto walk_fn = [&](const TensorLoad& tensor_load) -> ffi::Expected<ffi::WalkResult> {
      if (tensor_load->source.as_or_throw<tvm::tirx::TensorVar>().scope() ==
              scope_wmma_accumulator ||
          tensor_load->source.as_or_throw<tvm::tirx::TensorVar>().scope() == scope_mma_matrixC) {
        if (target_tensor_load == nullptr) {
          target_tensor_load = tensor_load.get();
        } else {
          TVM_FFI_ICHECK(target_tensor_load->source.as_or_throw<tvm::tirx::TensorVar>().same_as(
              tensor_load->source.as_or_throw<tvm::tirx::TensorVar>()))
              << "More than one target tensor found";
          TVM_FFI_ICHECK(target_tensor_load->indices.size() == tensor_load->indices.size());
          for (size_t i = 0; i < target_tensor_load->indices.size(); i++) {
            TVM_FFI_ICHECK(
                analyzer->CanProveEqual(target_tensor_load->indices[i], tensor_load->indices[i]));
          }
        }
      }
      return ffi::WalkResult::Advance();
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(stmt, walk_fn);
    TVM_FFI_ICHECK(target_tensor_load);
  }

  const TensorStoreObj* tensor_store = TVM_TYPE_AS(body, TensorStoreObj);
  ffi::Array<PrimExpr> cache_indices;
  ffi::Array<PrimExpr> new_shape;
  bool use_rank_promotion = false;
  if (!is_write_cache && tensor_store->value.as<TensorLoadObj>()) {
    ffi::Array<PrimExpr> indices =
        is_write_cache ? tensor_store->indices : tensor_store->value.as<TensorLoadObj>()->indices;
    new_shape = IndexPatternFinder::getRankPromotedShape(indices, var_range, &cache_indices);
    // write cache disabled for now
    // rank promotion for write cache cannot guarantee the shape fits wmma.accumulator
    if (!new_shape.empty()) {
      use_rank_promotion = true;
    }
  }
  ffi::Array<PrimVar> new_loop_vars;
  ffi::Map<Var, PrimExpr> subst_map;
  if (!use_rank_promotion) {
    cache_indices.clear();
    for (const ForObj* loop : relaxed_thread_loops) {
      new_shape.push_back(Range::GetExtentFromBeginEnd(loop->begin, loop->end));
    }
    for (const ForObj* loop : loops_under_compute_location) {
      new_shape.push_back(Range::GetExtentFromBeginEnd(loop->begin, loop->end));
    }
  }

  for (int i = 0; i < static_cast<int>(relaxed_thread_loops.size()); i++) {
    const ForObj* loop = relaxed_thread_loops[i];
    PrimVar new_loop_var = loop->var.CopyWithSuffix("_cache");
    new_loop_vars.push_back(new_loop_var);
    subst_map.Set(loop->var, new_loop_var);
    if (!use_rank_promotion) {
      cache_indices.push_back(loop->var);
    }
  }
  for (int i = 0; i < static_cast<int>(loops_under_compute_location.size()); i++) {
    const ForObj* loop = loops_under_compute_location[i];
    PrimVar new_loop_var = loop->var.CopyWithSuffix("_cache");
    new_loop_vars.push_back(new_loop_var);
    subst_map.Set(loop->var, new_loop_var);
    if (!use_rank_promotion) {
      cache_indices.push_back(loop->var);
    }
  }
  auto map_var = [&subst_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = subst_map.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  ffi::Array<PrimExpr> subst_indices;
  ffi::Array<PrimExpr> subst_cache_indices;
  if (is_write_cache) {
    for (PrimExpr e : tensor_store->indices) {
      subst_indices.push_back(
          ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, map_var).as_or_throw<PrimExpr>());
    }
  }
  for (PrimExpr e : cache_indices) {
    subst_cache_indices.push_back(
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(e, map_var).as_or_throw<PrimExpr>());
  }

  TensorVar new_tensor{ffi::UnsafeInit{}};
  if (is_write_cache) {
    // this is needed for global <- cast(load(wmma))
    // shared stage should have the same dtype as wmma
    new_tensor = WithScope(target_tensor_load->source.as_or_throw<tvm::tirx::TensorVar>(),
                           runtime::MemoryScope::Get(storage_scope));
  } else {
    new_tensor = WithScope(tensor_store->dest.as_or_throw<TensorVar>(),
                           runtime::MemoryScope::Get(storage_scope));
  }
  ffi::ObjectPtr<TensorTypeObj> tensor_type =
      ffi::make_object<TensorTypeObj>(*new_tensor.ty().get());
  tensor_type->shape = new_shape;
  new_tensor = TensorVar(new_tensor->name, TensorType(std::move(tensor_type)), new_tensor->loc);
  *alloc_tensor = new_tensor;

  Stmt generate_body{ffi::UnsafeInit{}};
  if (is_write_cache) {
    // copy from wmma to new cache tensor
    TensorLoad new_tensor_load = TensorLoad(new_tensor, cache_indices);
    generate_body =
        ffi::make_object<TensorLoadReplacer>(
            target_tensor_load->source.as_or_throw<tvm::tirx::TensorVar>(), new_tensor_load)
            ->Mutate(ffi::GetRef<Stmt>(tensor_store))
            .ValueOrUnchanged(ffi::GetRef<Stmt>(tensor_store));
    generate_body =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(generate_body, map_var).as_or_throw<Stmt>();
  } else {
    generate_body =
        TensorStore(new_tensor, subst_cache_indices,
                    ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(tensor_store->value, map_var)
                        .as_or_throw<PrimExpr>());
  }

  if (predicate.has_value()) {
    // generated by coalescing
    TVM_FFI_ICHECK_EQ(loops_under_compute_location.size(), 2);
    PrimExpr subst_value = 0;
    PrimExpr subst_predicate =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(predicate.value(), map_var)
            .as_or_throw<PrimExpr>();
    generate_body = If(subst_predicate, generate_body);
  }

  for (int i = static_cast<int>(loops_under_compute_location.size()) - 1; i >= 0; i--) {
    const ForObj* orig_loop = loops_under_compute_location[i];
    ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*orig_loop);
    new_loop->var = new_loop_vars[i + relaxed_thread_loops.size()];
    new_loop->body = generate_body;
    generate_body = For(new_loop);
  }
  for (int i = static_cast<int>(relaxed_thread_loops.size()) - 1; i >= 0; i--) {
    const ForObj* orig_loop = relaxed_thread_loops[i];
    ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*orig_loop);
    new_loop->var = new_loop_vars[i];
    new_loop->body = generate_body;
    new_loop->kind = ForKind::kDefault;
    new_loop->annotations = {};
    generate_body = For(new_loop);
  }
  Stmt rewrite_body{ffi::UnsafeInit{}};
  if (is_write_cache) {
    TensorLoad new_tensor_load = TensorLoad(new_tensor, cache_indices);
    rewrite_body =
        TensorStore(new_tensor, cache_indices, ffi::GetRef<TensorLoad>(target_tensor_load));
  } else {
    rewrite_body = TensorStore(tensor_store->dest.as_or_throw<TensorVar>(), tensor_store->indices,
                               TensorLoad(new_tensor, cache_indices));
  }
  if (predicate.has_value()) {
    rewrite_body = If(predicate.value(), rewrite_body);
  }
  for (int i = static_cast<int>(loops_under_compute_location.size()) - 1; i >= 0; i--) {
    const ForObj* orig_loop = loops_under_compute_location[i];
    ffi::ObjectPtr<ForObj> new_loop = ffi::make_object<ForObj>(*orig_loop);
    new_loop->body = rewrite_body;
    rewrite_body = For(new_loop);
  }
  SeqStmt insert_location{ffi::UnsafeInit{}};
  if (is_write_cache) {
    generate_body = insert_location = SeqStmt({rewrite_body, generate_body});
  } else {
    generate_body = insert_location = SeqStmt({generate_body, rewrite_body});
  }
  generate_body = CopyLoopChain(loops, generate_body);
  return std::make_pair(generate_body, insert_location);
}

Stmt CreateLocalStage::Rewrite(const Stmt& stmt, const ConstraintSet& constraints,
                               OutputSet* output) const {
  auto [body, compute_location] = LiftThreadBindingLoops(stmt);
  TensorVar cache_tensor{ffi::UnsafeInit{}};
  Stmt after_caching = InsertCacheStage(body, false, "local", compute_location,
                                        constraints.outer_loops, &cache_tensor)
                           .first;
  output->alloc_tensor.push_back(cache_tensor);
  return after_caching;
}

}  // namespace s_tir
}  // namespace tvm
