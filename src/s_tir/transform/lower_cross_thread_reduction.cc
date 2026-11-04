/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file lower_cross_thread_reduction.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/support/gpu_thread_tag.h>
#include <tvm/sym/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/type.h>

#include "../schedule/analysis.h"
#include "../schedule/utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;

using namespace tvm::tirx;
using support::GPUThreadTag;

// Implement a hash and equality function for GPUThreadTag so that
// GPUThreadTag can serve as map key class
struct GPUThreadTagHash {
  size_t operator()(const GPUThreadTag& scope) const {
    return static_cast<size_t>(scope.rank * 30 + scope.dim_index);
  }
};

struct GPUThreadTagEqual {
  bool operator()(const GPUThreadTag& a, const GPUThreadTag& b) const {
    return a.rank == b.rank && a.dim_index == b.dim_index;
  }
};

/*!
 * \brief Checks if a loop is bound to threadIdx.x/y/z
 * \brief loop The loop to be checked
 * \return True if the loop is bound to threadIdx.x/y/z
 */
bool IsBoundToThreadIdx(const ForObj* loop) {
  if (!tvm::s_tir::GetThreadBinding(loop).has_value()) {
    return false;
  }
  support::GPUThreadTag scope = ParseThreadBindingTag(tvm::s_tir::GetThreadBinding(loop).value());
  return scope.rank == 1 && scope.dim_index >= 0;
}

/*!
 * \brief Check the dominant property of a block:
 * the block is the only writer of its output, dominating the reader of its output tensors
 * \param scope_block The scope block of the block to be checked
 * \param block The block whose dominant property is to be checked
 * \return A boolean indicating if the block is a dominant block
 */
bool IsDominantBlock(const SBlock& scope_block, const SBlock& block) {
  // Step 1. Count the number of writers for each tensor written by the scope block.
  std::unordered_map<const VarObj*, int> tensor_writer_cnt;
  auto walk_fn = [&tensor_writer_cnt](const SBlock& block) -> ffi::Expected<ffi::WalkResult> {
    for (const TensorRegion& tensor_region : block->writes) {
      ++tensor_writer_cnt[tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().get()];
    }
    return ffi::WalkResult::Skip();
  };
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(scope_block->body, walk_fn);
  // Step 2. Check whether `block` is the only writer of its outputs.
  for (const TensorRegion& tensor_region : block->writes) {
    TVM_FFI_ICHECK(
        tensor_writer_cnt.count(tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().get()));
    if (tensor_writer_cnt[tensor_region->source.as_or_throw<tvm::tirx::TensorVar>().get()] != 1) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Check whether the input block is a reduction block.
 * \param realize The block to be checked
 * \param loop_range_map The mapping from the loop variables outside the input block to their ranges
 * \param scope_block The scope block of the input block
 * \param analyzer The analyzer
 * \return A boolean indicating whether the input block is a reduction block.
 * \note A similar check has been implemented in "src/s_tir/schedule/analysis.h", but that check is
 * based on `tirx.Schedule`. Here we have no schedule information, and thus we must implement the
 * check again.
 */
bool IsReductionBlock(const SBlockRealize& realize, const ffi::Map<Var, Range>& loop_range_map,
                      const SBlock& scope_block, sym::AnalyzerObj* analyzer) {
  const auto* block = realize->block.as<SBlockObj>();
  // Cond 1. The block has the `init` statement.
  if (!block->init.has_value()) {
    return false;
  }
  // Cond 2. All the block bindings are quasi-affine expressions.
  if (!IsAffineBinding(realize, loop_range_map, analyzer)) {
    return false;
  }
  // Cond 3. All block vars are either data parallel block vars or reduction block vars. Meanwhile,
  // we collect all the reduction block vars.
  if (!ContainsOnlyDataParAndReductionBlockIter(block->iter_vars)) {
    return false;
  }
  // Cond 4. Dominant: the block is the only writer of its output, dominating the reader of its
  // output tensors.
  if (!IsDominantBlock(scope_block, ffi::GetRef<SBlock>(block))) {
    return false;
  }
  // Cond 5. The reduction block vars are not used to index the output tensors.
  return ReductionIterNotIndexOutputTensor(ffi::GetRef<SBlock>(block));
}

/*!
 * \brief Create intermediate tensors according to the input tensors and tensor kind
 * \param reduction_tensors The old reduction tensors which provide the tensor names and data types
 * \param is_cross_thread_tensor A boolean indicating whether to create tensors for the cross-thread
 * computation results or not, which is used for determine the tensor name prefix
 * \return The created tensors
 */
ffi::Array<TensorVar> MakeScratchpads(const ffi::Array<TensorVar>& reduction_tensors,
                                      bool is_cross_thread_tensor) {
  static const auto scope_local = runtime::MemoryScope::Get("local");
  ffi::Array<TensorVar> new_tensors;
  new_tensors.reserve(reduction_tensors.size());
  for (const TensorVar& tensor : reduction_tensors) {
    ffi::String name = is_cross_thread_tensor ? "cross" : "in";
    name = name + "_thread_" + tensor->name;
    new_tensors.push_back(
        TensorVar(name, TensorType(/*shape=*/{IntImm::Int32(1)}, /*dtype=*/tensor.dtype(),
                                   scope_local, /*data_alignment=*/0,
                                   tvm::tirx::TileLayout::DefaultLayout({IntImm::Int32(1)}))));
  }
  return new_tensors;
}

/*!
 * \brief Substitute given source tensors with given target tensors respectively in the input
 * statement
 */
class TensorReplacer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Run(ffi::Array<TensorVar> src_tensors, ffi::Array<TensorVar> tgt_tensors, Stmt stmt) {
    ffi::Map<TensorVar, TensorVar> tensor_map;
    TVM_FFI_ICHECK_EQ(src_tensors.size(), tgt_tensors.size());
    int n_tensors = src_tensors.size();
    for (int i = 0; i < n_tensors; ++i) {
      tensor_map.Set(src_tensors[i], tgt_tensors[i]);
    }
    return ffi::make_object<TensorReplacer>(tensor_map)
        ->Mutate(stmt, InplaceMode::kAllow)
        .ValueOrUnchanged(std::move(stmt));
  }

  explicit TensorReplacer(ffi::Map<TensorVar, TensorVar> tensor_map) {
    for (const auto& [tensor, replacement] : tensor_map) VarRemapSet(tensor, replacement);
  }

 private:
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* load, InplaceMode inplace_mode) final {
    auto replacement = VarRemapGet(load->source).as<TensorVar>();
    return replacement ? TensorLoad(replacement.value(), {0}) : ffi::GetRef<TensorLoad>(load);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* store, InplaceMode inplace_mode) final {
    if (auto replacement = VarRemapGet(store->dest.as_or_throw<TensorVar>()).as<TensorVar>()) {
      PrimExpr value = StmtExprMutator::Mutate(ffi::AnyView(store->value), inplace_mode)
                           .ValueOrUnchanged(store->value)
                           .as_or_throw<PrimExpr>();
      return TensorStore(replacement.value(), {0}, std::move(value));
    } else {
      return StmtExprMutator::Mutate_(store, inplace_mode);
    }
  }
};

/*!
 * \brief Substitute a given source block with a given target block, or remove the source block
 * branch from the AST if the target block is undefined
 */
class InThreadReducerMaker : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) override {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    return StmtExprMutator::Mutate(value, inplace_mode);
  }

  /*!
   * \brief Visitor class to collect all reduction block variables under a loop.
   */
  class UnderLoopReductionBlockVarCollector : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;
    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }
    /*!
     * \brief Check if the given statement has any reduction blocks.
     * \param stmt The statement to check.
     * \return True if the statement has reduction blocks, false otherwise.
     */
    static bool CheckHasReductionBlocks(const Stmt& stmt) {
      auto collector = ffi::make_object<UnderLoopReductionBlockVarCollector>();
      collector->Visit(stmt);
      return collector->reduction_block_vars_.size() > 0;
    }

   private:
    ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
      ffi::Array<IterVar> iter_vars = block->iter_vars;
      for (const IterVar& iter_var : block->iter_vars) {
        if (iter_var->iter_type == kCommReduce) {
          reduction_block_vars_.push_back(iter_var);
        }
      }
      return StmtExprVisitor::Visit_(block);
    }

    /*! \brief the map from thread tag to its extent */
    ffi::Array<IterVar> reduction_block_vars_;
  };

  static ffi::Optional<Stmt> Make(const SBlockRealizeObj* src_realize,
                                  ffi::Optional<SBlockRealize> tgt_realize, Stmt stmt) {
    return ffi::make_object<InThreadReducerMaker>(src_realize, std::move(tgt_realize))
        ->Rewrite(std::move(stmt));
  }

  explicit InThreadReducerMaker(const SBlockRealizeObj* src_realize,
                                ffi::Optional<SBlockRealize> tgt_realize)
      : src_realize_(src_realize), tgt_realize_(tgt_realize) {}

 private:
  // Statement removal is carried separately from the mutator's required Stmt result.
  // Each enclosing loop or sequence consumes the Optional before constructing its body.
  ffi::Optional<Stmt> Rewrite(Stmt stmt) {
    removed_ = false;
    Stmt result = Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt).as_or_throw<Stmt>();
    if (removed_) {
      return std::nullopt;
    }
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* realize, InplaceMode inplace_mode) final {
    if (realize == src_realize_) {
      if (tgt_realize_.has_value()) {
        return tgt_realize_.value();
      }
      removed_ = true;
    }
    return ffi::Unchanged();
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    auto body = Rewrite(loop->body);
    if (!body.has_value()) {
      removed_ = true;
      return ffi::Unchanged();
    }
    For res = ffi::GetRef<For>(loop);
    if (!body.value().same_as(loop->body)) {
      res.CopyOnWrite()->body = body.value();
    }
    if (tvm::s_tir::GetThreadBinding(res).has_value() &&
        UnderLoopReductionBlockVarCollector::CheckHasReductionBlocks(res)) {
      return res->body;
    }
    return res;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj* seq, InplaceMode inplace_mode) final {
    ffi::Array<Stmt> stmts;
    stmts.reserve(seq->size());
    for (const Stmt& stmt : seq->seq) {
      if (auto result = Rewrite(stmt)) {
        stmts.push_back(result.value());
      }
    }
    removed_ = stmts.empty();
    if (removed_) {
      return ffi::Unchanged();
    }
    return SeqStmt(stmts);
  }

  bool removed_ = false;

  const SBlockRealizeObj* src_realize_;
  ffi::Optional<SBlockRealize> tgt_realize_;
};

/*!
 * \brief Create the lowered allreduce block transformed from the input reduction block
 * \param realize The block-realize which contains the old reduction block
 * \param it_tensors The tensors to store in-thread reduction results
 * \param ct_tensors The tensors to store cross-thread reduction results
 * \param wb_tensors The tensors to store the final reduction results
 * \param old_wb_indices The indices used to access the write-back tensors when storing the final
 * reduction results into the write-back tensors
 * \param reducer The reduction function
 * \param combiner_rhs The RHS values of the combiner
 * \param reduction_loops The reduction loops
 */
Stmt TransformReductionBlock(const SBlockRealizeObj* realize,                         //
                             const ffi::Optional<ffi::Array<TensorVar>>& it_tensors,  //
                             const ffi::Array<TensorVar>& ct_tensors,                 //
                             const ffi::Array<TensorVar>& wb_tensors,                 //
                             const ffi::Array<PrimExpr>& old_wb_indices,              //
                             const te::CommReducer& reducer,                          //
                             const ffi::Array<PrimExpr>& combiner_rhs,                //
                             const std::vector<const ForObj*>& reduction_loops) {
  static const auto scope_local = runtime::MemoryScope::Get("local");

  int n_tensors = wb_tensors.size();
  const SBlockObj* block = realize->block.get();

  auto f_create_tensor_regions = [](ffi::Array<TensorVar> tensors) {
    ffi::Array<TensorRegion> regions;
    regions.reserve(tensors.size());
    for (const TensorVar& tensor : tensors) {
      regions.push_back(TensorRegion(tensor, {Range::FromMinExtent(0, 1)}, TensorRegionType()));
    }
    return regions;
  };

  ffi::Array<TensorRegion> ct_tensor_regions = f_create_tensor_regions(ct_tensors);
  ffi::Optional<ffi::Array<TensorRegion>> it_tensor_regions = std::nullopt;
  if (it_tensors.has_value()) {
    it_tensor_regions = f_create_tensor_regions(it_tensors.value());
  }
  // In total, the block is transformed into at most 4 statements
  // - Stmt 1: initialize the tensor for in-thread reduction
  // - Stmt 2: do in-thread reduction
  // - Stmt 3: do cross-thread reduction
  // - Stmt 4: write cross-thread reduction result to the original tensor
  ffi::Array<Stmt> stmts;
  stmts.reserve(4);
  // Stmt 1: initialize the tensor for in-thread reduction
  if (it_tensors.has_value()) {
    ffi::Array<Stmt> inits;
    inits.reserve(n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
      inits.push_back(
          TensorStore(it_tensors.value()[i], {IntImm::Int32(0)}, reducer->identity_element[i]));
    }
    stmts.push_back(SBlockRealize(/*iter_values=*/{},
                                  /*predicate=*/IntImm::Bool(true),
                                  /*block=*/
                                  SBlock(/*iter_vars=*/{},
                                         /*reads=*/{},
                                         /*writes=*/it_tensor_regions.value(),
                                         /*name_hint=*/block->name_hint + "_in_thread_init",
                                         /*body=*/SeqStmt(inits))));
  }
  // Stmt 2: do in-thread reduction
  {
    ffi::Optional<SBlockRealize> new_realize = std::nullopt;
    // If need to generate in-thread reduction,
    // then replace `wb_tensors` with `it_tensors` accordingly in given BlockRealize
    // otherwise, directly remove given BlockRealize
    if (it_tensors.has_value()) {
      ffi::ObjectPtr<SBlockObj> new_block = ffi::make_object<SBlockObj>(*block);
      new_block->writes = it_tensor_regions.value();
      new_block->name_hint = new_block->name_hint + "_in_thread";
      new_block->body =
          TensorReplacer::Run(wb_tensors, it_tensors.value(), std::move(new_block->body));
      new_block->init = std::nullopt;
      ffi::ObjectPtr<SBlockRealizeObj> n = ffi::make_object<SBlockRealizeObj>(*realize);
      n->block = SBlock(new_block);
      new_realize = SBlockRealize(n);
    }
    For loop = ffi::GetRef<For>(reduction_loops[0]);
    if (ffi::Optional<Stmt> stmt =
            InThreadReducerMaker::Make(realize, new_realize, std::move(loop))) {
      stmts.push_back(stmt.value());
    }
  }
  // Stmt 3: do cross-thread reduction
  {
    // Step 3.1. Carry the reducer directly as typed staging operands.
    ffi::Array<Var> combine_vars;
    for (const Var& var : reducer->lhs) combine_vars.push_back(var);
    for (const Var& var : reducer->rhs) combine_vars.push_back(var);
    LambdaExpr combine(combine_vars, tvm::Tuple(reducer->result));
    ffi::Array<PrimExpr> values;
    if (it_tensors.has_value()) {
      for (int i = 0; i < n_tensors; ++i) {
        values.push_back(TensorLoad(it_tensors.value()[i], {IntImm::Int32(0)}));
      }
    } else {
      values = combiner_rhs;
    }
    ffi::Array<PrimExpr> destinations;
    for (int i = 0; i < n_tensors; ++i) {
      destinations.push_back(TensorLoad(ct_tensors[i], {0}));
    }
    ffi::Array<PrimExpr> thread_axes;
    for (const ForObj* reduction_loop : reduction_loops) {
      if (tvm::s_tir::GetThreadBinding(reduction_loop).has_value()) {
        thread_axes.push_back(reduction_loop->var);
      }
    }
    ffi::Array<Expr> parameters{combine,
                                tvm::Tuple(reducer->identity_element),
                                tvm::Tuple(values),
                                IntImm::Bool(true),
                                tvm::Tuple(destinations),
                                tvm::Tuple(thread_axes)};
    // Step 3.2. Create the block and the block-realize.
    ffi::Array<IterVar> iter_vars;
    ffi::Array<PrimExpr> bindings;
    ffi::Array<TensorRegion> reads;
    if (it_tensors.has_value()) {
      iter_vars = ffi::Array<IterVar>{};
      bindings = ffi::Array<PrimExpr>{};
      reads = it_tensor_regions.value();
    } else {
      iter_vars = block->iter_vars;
      bindings = realize->iter_values;
      reads = block->reads;
    }
    stmts.push_back(SBlockRealize(
        /*iter_values=*/std::move(bindings),
        /*predicate=*/IntImm::Bool(true),
        /*block=*/
        SBlock(/*iter_vars=*/std::move(iter_vars),
               /*reads=*/std::move(reads),
               /*writes=*/ct_tensor_regions,
               /*name_hint=*/block->name_hint + "_cross_thread",
               /*body=*/
               Evaluate(Call(/*dtype=*/PrimType::Void(),
                             /*op=*/tirx::gpu_thread_allreduce_op(),
                             /*args=*/std::move(parameters))
                            .as_or_throw<PrimExpr>()))));
  }
  // Stmt 4: write cross-thread reduction result to the original tensor
  {
    TVM_FFI_ICHECK_EQ(block->iter_vars.size(), realize->iter_values.size());
    int n_iter = static_cast<int>(block->iter_vars.size());
    ffi::Array<IterVar> iter_vars;
    ffi::Array<PrimExpr> bindings;
    ffi::Map<Var, Var> var_map;
    iter_vars.reserve(n_iter);
    bindings.reserve(n_iter);
    for (int i = 0; i < n_iter; ++i) {
      const IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->iter_values[i];
      if (iter_var->iter_type != kCommReduce) {
        IterVar new_iter_var{ffi::UnsafeInit{}};
        {
          ffi::ObjectPtr<IterVarObj> n = ffi::make_object<IterVarObj>(*iter_var.get());
          Var v(iter_var->var->name, iter_var->var->ty, iter_var->var->loc);
          n->var = v.as_or_throw<PrimVar>();
          new_iter_var = IterVar(n);
        }
        iter_vars.push_back(new_iter_var);
        bindings.push_back(binding);
        var_map.Set(iter_var->var, new_iter_var->var);
      }
    }
    ffi::Array<Stmt> wb_updates;
    ffi::Array<TensorRegion> wb_regions;
    wb_updates.reserve(n_tensors);
    wb_regions.reserve(n_tensors);
    int n_dim = static_cast<int>(old_wb_indices.size());
    auto map_var = [&var_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = var_map.Get(var)) {
        return ffi::Any((*std::move(repl)).as_or_throw<PrimExpr>());
      }
      return ffi::Unchanged();
    };
    ffi::Array<Range> region = block->writes[0]->region.Map([&map_var](const Range& range) {
      PrimExpr min = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->begin, map_var)
                         .as_or_throw<PrimExpr>();
      PrimExpr extent = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(range->extent, map_var)
                            .as_or_throw<PrimExpr>();
      return Range::FromMinExtent(min, extent);
    });
    ffi::Array<PrimExpr> wb_indices;
    wb_indices.reserve(n_dim);
    for (int d = 0; d < n_dim; ++d) {
      wb_indices.push_back(ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(old_wb_indices[d], map_var)
                               .as_or_throw<PrimExpr>());
    }
    for (int i = 0; i < n_tensors; ++i) {
      wb_updates.push_back(
          TensorStore(wb_tensors[i], wb_indices, TensorLoad(ct_tensors[i], {IntImm::Int32(0)})));
      wb_regions.push_back(TensorRegion(wb_tensors[i], region, TensorRegionType()));
    }

    // Construct the predicate of the write-back block. It is the conjunction of
    // - each predicate clause of the original block which contains spatial loop var, and
    // - `t == 0` for each reduction thread dim when the write-back tensor is not local.
    PrimExpr wb_predicate = IntImm::Bool(true);
    std::unordered_set<const VarObj*> reduction_loop_vars;
    reduction_loop_vars.reserve(reduction_loops.size());
    for (const ForObj* reduction_loop : reduction_loops) {
      reduction_loop_vars.insert(reduction_loop->var.get());
    }
    std::unordered_set<const ffi::Object*> visited_predicate_nodes;
    auto walk_fn = [&wb_predicate, &reduction_loop_vars, &visited_predicate_nodes](
                       const And& and_expr) -> ffi::Expected<ffi::WalkResult> {
      if (!visited_predicate_nodes.insert(and_expr.get()).second) {
        return ffi::WalkResult::Advance();
      }
      ffi::Array<PrimExpr> sub_exprs = {and_expr->a, and_expr->b};
      for (PrimExpr sub_expr : sub_exprs) {
        if (sub_expr->IsInstance<AndObj>()) {
          continue;
        }
        bool is_reduction = [sub_expr, &reduction_loop_vars]() {
          ffi::Array<Var> vars = UndefinedVars(sub_expr);
          for (Var var : vars) {
            if (reduction_loop_vars.find(var.get()) != reduction_loop_vars.end()) {
              return true;
            }
          }
          return false;
        }();
        if (!is_reduction) {
          wb_predicate = wb_predicate && sub_expr;
        }
      }
      return ffi::WalkResult::Advance();
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(realize->predicate, walk_fn);
    if (wb_tensors[0].scope() != scope_local) {
      for (const ForObj* loop : reduction_loops) {
        if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
          wb_predicate =
              wb_predicate && (static_cast<PrimExpr>(loop->var) == IntImm(loop->var.ty(), 0));
        }
      }
    }

    stmts.push_back(SBlockRealize(
        /*iter_values=*/std::move(bindings),
        /*predicate=*/wb_predicate,
        /*block=*/
        SBlock(/*iter_vars=*/std::move(iter_vars),
               /*reads=*/std::move(ct_tensor_regions),
               /*writes=*/std::move(wb_regions),
               /*name_hint=*/block->name_hint + "_write_back",
               /*body=*/SeqStmt(wb_updates))));
  }
  // Final step: Wrap all the above four statements with the reduction loops bound to threadIdx
  Stmt new_stmt = SeqStmt(std::move(stmts));
  for (auto rit = reduction_loops.rbegin(); rit != reduction_loops.rend(); ++rit) {
    const ForObj* loop = *rit;
    if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
      ffi::ObjectPtr<ForObj> n = ffi::make_object<ForObj>(*loop);
      n->body = std::move(new_stmt);
      new_stmt = For(n);
    }
  }
  return new_stmt;
}

/*!
 * \brief Detect cross-thread reduction pattern and then transform
 */
class CrossThreadReductionTransformer : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

 private:
  // Check if the input block needs cross-thread reduction.
  std::vector<const ForObj*> NeedCrossThreadReduction(const SBlockRealizeObj* realize) {
    // Step 0. If the block is the root block, just return.
    if (block_stack_.empty()) {
      return {};
    }

    // Step 1. If the block is not a reduction block, cross-thread reduction is not needed.
    if (!IsReductionBlock(ffi::GetRef<SBlockRealize>(realize), loop_range_map_,
                          ffi::GetRef<SBlock>(block_stack_.back()), analyzer_.get())) {
      return {};
    }

    // Step 2. Collect all the vars that appear in the bindings of reduction block iters.
    std::unordered_set<const VarObj*> reduction_vars;
    GetVarsTouchedByBlockIters(ffi::GetRef<SBlockRealize>(realize), nullptr, &reduction_vars);

    // Step 3. Collect the loops whose loop vars appear in the bindings of reduction block iters.
    // We call these loops "reduction-related".
    // Step 4. See whether at least one reduction-related loop is bound to thread axis in GPU - if
    // so, cross-thread reduction is needed. If none of the reduction-related loops is bound to
    // thread axis, cross-thread reduction is not needed for the input block.
    bool need = false;
    std::vector<const ForObj*> reduction_loops;
    for (const ForObj* loop : loop_stack_) {
      if (reduction_vars.count(loop->var.get())) {
        // Step 3. Collect the loop.
        reduction_loops.push_back(loop);
        // Step 4. See whether the loop is bound to some thread axis.
        if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
          need = true;
        }
      }
    }
    return need ? reduction_loops : std::vector<const ForObj*>{};
  }

  // Check if the input block needs thread broadcast rewrite.
  // One block needs broadcast rewrite when
  // 1. it consumes a tensor produced by cross-thread reduction under
  // the same kernel (i.e., same group of blockIdx),
  // 2. it writes to non-local memory,
  // 3. at least one of the reduction thread vars of the cross-thread reduction
  // is free to this block (i.e., not bound to the block).
  std::vector<std::pair<GPUThreadTag, Range>> NeedCrossThreadBroadcast(
      const SBlockRealizeObj* realize) {
    static const auto scope_local = runtime::MemoryScope::Get("local");

    SBlock block = realize->block;

    // If the block writes to local memory, no rewrite is needed.
    for (TensorRegion write_region : block->writes) {
      if (write_region->source.as_or_throw<tvm::tirx::TensorVar>().scope() == scope_local) {
        return {};
      }
    }

    // Find out the reduction threads for the read-tensors which are produced by
    // cross-thread reduction.
    std::unordered_map<GPUThreadTag, Range, GPUThreadTagHash, GPUThreadTagEqual> thread2range;
    for (TensorRegion read_region : block->reads) {
      auto tensor_it =
          crt_tensor2threads_.find(read_region->source.as_or_throw<tvm::tirx::TensorVar>().get());
      if (tensor_it == crt_tensor2threads_.end()) {
        continue;
      }
      for (auto [scope, range] : tensor_it->second) {
        thread2range.insert_or_assign(scope, range);
      }
    }

    // Erase those threads which are not free to this block.
    for (const ForObj* loop : loop_stack_) {
      if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
        GPUThreadTag scope = ParseThreadBindingTag(tvm::s_tir::GetThreadBinding(loop).value());
        thread2range.erase(scope);
      }
    }
    std::vector<std::pair<GPUThreadTag, Range>> unbound_thread2range_list;
    for (auto [scope, range] : thread2range) {
      unbound_thread2range_list.emplace_back(scope, range);
    }
    return unbound_thread2range_list;
  }

  /*!
   * \brief Given that the input block needs cross-thread reduction, check if cross-thread reduction
   * can be applied to the block (i.e., the block satisfies all necessary conditions of cross-thread
   * reduction)
   * \param block The block to be checked
   * \param reduction_loops The reduction loops above the block
   * \return A tuple consisting of five elements:
   *  - an integer which indicates the number of reduction loops that are bound to thread axes,
   *  - the detected commutative reducer of the reduction,
   *  - the reduction tensors which store the reduction results,
   *  - the RHS values of the reduction updates,
   *  - the indices which is used to access the reduction tensors when storing the reduction results
   */
  std::tuple<int, te::CommReducer, ffi::Array<TensorVar>, ffi::Array<PrimExpr>,
             ffi::Array<PrimExpr>>
  CheckCanApplyCrossThreadReduction(const SBlockObj* block,
                                    const std::vector<const ForObj*>& reduction_loops) const {
    static const auto scope_local = runtime::MemoryScope::Get("local");

    // Condition 1. All the reduction-related loops should be the deepest among all statements
    // outside the block (ignoring SeqStmt here).
    int n_deepest_reduction_loops = 0;
    for (auto rit = statement_stack_.rbegin() + 1; rit != statement_stack_.rend(); ++rit) {
      const StmtObj* stmt = *rit;
      if ((*rit)->IsInstance<SeqStmtObj>()) {
        // Skip SeqStmt.
        continue;
      }
      if (std::find(reduction_loops.begin(), reduction_loops.end(),
                    reinterpret_cast<const ForObj*>(stmt)) == reduction_loops.end()) {
        break;
      }
      ++n_deepest_reduction_loops;
    }
    TVM_FFI_CHECK_EQ(n_deepest_reduction_loops, reduction_loops.size(), ValueError)
        << "Cross-thread reduction requires all the reduction-related loops to be the "
           "deepest among all statements outside the desired block. However, block "
        << block->name_hint
        << " needs cross-thread reduction, while the reduction-related loops outside of it are not "
           "the deepest statements, which violates the condition.";

    // Condition 2. All the reduction-related loops that are bound to thread axes should only be
    // bound to `threadIdx.x/y/z`.
    int n_bound_reduction_loops = 0;
    for (const ForObj* reduction_loop : reduction_loops) {
      if (tvm::s_tir::GetThreadBinding(reduction_loop).has_value()) {
        ++n_bound_reduction_loops;
        TVM_FFI_CHECK(IsBoundToThreadIdx(reduction_loop), ValueError)
            << "Cross-thread reduction requires all the reduction-related loops that "
               "are bound to GPU thread axes to only be bound `threadIdx.x/y/z`. However, loop "
            << reduction_loop->var->name << " violates the condition.";
      }
    }

    // Condition 3. Get the identity values of the block init and the TensorStore block combiner
    // updates of the reduction. Extract the commutative reducer, combiner lhs and combiner rhs from
    // the reduction identities and the reduction combiner.
    ffi::Array<PrimExpr> init_values;
    ffi::Array<TensorStore> updates;
    te::CommReducer reducer{ffi::UnsafeInit{}};
    ffi::Array<PrimExpr> combiner_lhs;
    ffi::Array<PrimExpr> combiner_rhs;
    std::tie(init_values, updates) =
        GetInitValuesAndUpdatesFromReductionBlock(std::nullopt, ffi::GetRef<SBlock>(block));
    std::tie(reducer, combiner_lhs, combiner_rhs) =
        GetReducerAndCombinerLhsRhs(std::nullopt, init_values, updates);

    // Condition 4. All reduction tensors should be all local or all non-local.
    int is_local_tensor = -1;
    ffi::Array<TensorVar> reduction_tensors;
    reduction_tensors.reserve(updates.size());
    for (const TensorStore& tensor_store : updates) {
      reduction_tensors.push_back(tensor_store->dest.as_or_throw<TensorVar>());
      if (tensor_store->dest.as_or_throw<TensorVar>().scope() == scope_local) {
        TVM_FFI_CHECK_NE(is_local_tensor, 0, ValueError)
            << "Cross-thread reduction requires all reduction tensors to be all "
               "local or all non-local. However, here some tensor is local while some tensor is "
               "shared or global.";
        is_local_tensor = 1;
      } else {
        TVM_FFI_CHECK_NE(is_local_tensor, 1, ValueError)
            << "Cross-thread reduction requires all reduction tensors to be all "
               "local or all non-local. However, here some tensor is local while some tensor is "
               "shared or global.";
        is_local_tensor = 0;
      }
    }

    // Condition 5. The block should be the last block under the first reduction-related loop.
    bool visit = false;
    auto walk_fn = [block, &visit](const SBlockRealize& realize) -> ffi::Expected<ffi::WalkResult> {
      TVM_FFI_CHECK(!visit, ValueError)
          << "Cross-thread reduction cannot be applied when the reduction "
             "block isn't the last block under its first reduction-related loop";
      if (realize->block.get() == block) {
        visit = true;
      }
      return ffi::WalkResult::Skip();
    };
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(ffi::GetRef<For>(reduction_loops[0]), walk_fn);
    return std::make_tuple(n_bound_reduction_loops,       //
                           std::move(reducer),            //
                           std::move(reduction_tensors),  //
                           std::move(combiner_rhs),       //
                           updates[0]->indices);
  }

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value, InplaceMode inplace_mode) final {
    if (value.as<ExprObj>()) return ffi::Unchanged();
    const auto* stmt = value.as<StmtObj>();
    if (!stmt) return StmtExprMutator::Mutate(value, inplace_mode);
    statement_stack_.push_back(stmt);
    auto result = StmtExprMutator::Mutate(value, inplace_mode);
    statement_stack_.pop_back();
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* loop, InplaceMode inplace_mode) final {
    loop_stack_.push_back(loop);
    loop_range_map_.Set(loop->var, Range(loop->begin, loop->end));

    // Collect loop-thread information:
    // - when encountering a threadIdx loop, we keep note of its domain and
    // the "loop var -> thread scope" relation, in order to collect all existing
    // threads within a thread block.
    // - we are careful about thread block boundary for safety.
    bool is_block_idx = false;
    bool is_thread_idx = false;
    if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
      GPUThreadTag scope = ParseThreadBindingTag(tvm::s_tir::GetThreadBinding(loop).value());
      if (scope.rank == 1 && scope.dim_index >= 0) {
        is_thread_idx = true;
        ++thread_idx_depth;
      } else if (scope.rank == 0) {
        is_block_idx = true;
        ++block_idx_depth;
      }
    }

    Stmt result =
        StmtExprMutator::Mutate_(loop, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(loop));
    loop_stack_.pop_back();
    loop_range_map_.erase(loop->var);
    if (is_thread_idx) {
      --thread_idx_depth;
    }
    if (is_block_idx) {
      --block_idx_depth;
    }
    if (is_block_idx || (is_thread_idx && thread_idx_depth == 0 && block_idx_depth == 0)) {
      crt_tensor2threads_.clear();
    }

    // Replace `result` with the pre-stored result if `loop` appears as a key in `loop2new_stmt_`.
    auto it = loop2new_stmt_.find(loop);
    if (it != loop2new_stmt_.end()) {
      return it->second;
    } else {
      return result;
    }
  }

  UnchangedOr<Stmt> Mutate_(const SBlockObj* block, InplaceMode inplace_mode) final {
    ffi::Map<Var, Range> old_loop_range_map;
    block_stack_.push_back(block);
    std::swap(old_loop_range_map, loop_range_map_);
    SBlock new_block = StmtExprMutator::Mutate_(block, inplace_mode)
                           .ValueOrUnchanged(ffi::GetRef<Stmt>(block))
                           .as_or_throw<SBlock>();
    block_stack_.pop_back();
    std::swap(old_loop_range_map, loop_range_map_);

    // Insert the new allocated tensors into the block's `alloc_tensors` field.
    auto it = block2new_tensors_.find(block);
    if (it != block2new_tensors_.end()) {
      SBlockObj* p_new_block = new_block.CopyOnWrite();
      for (const TensorVar& new_tensor : it->second) {
        if (new_tensor.defined()) {
          p_new_block->alloc_tensors.push_back(new_tensor);
        }
      }
    }
    return new_block;
  }

  void MakeCrossThreadReduction(const SBlockRealizeObj* realize,
                                const std::vector<const ForObj*> reduction_loops) {
    const SBlockObj* block = realize->block.get();

    // Step 1. Check whether cross-thread reduction can be applied. If no, throw an exception on
    // which condition the block violates.
    int n_bound_reduction_loops = 0;
    te::CommReducer reducer{ffi::UnsafeInit{}};
    ffi::Array<TensorVar> reduction_tensors;
    ffi::Array<PrimExpr> combiner_rhs;
    ffi::Array<PrimExpr> wb_indices;
    std::tie(n_bound_reduction_loops, reducer, reduction_tensors, combiner_rhs, wb_indices) =
        CheckCanApplyCrossThreadReduction(block, reduction_loops);
    // Step 2. Before doing the cross-thread reduction, in-thread reduction is needed when
    //  - not all the reduction-related loops are bound to thread axes, or
    //  - the block-realize has a non-constant-true predicate.
    bool need_in_thread_reduction =
        n_bound_reduction_loops < static_cast<int>(reduction_loops.size()) ||
        !IsOne(realize->predicate);
    // Step 3. Create intermediate tensors, storing them in `ct_tensors` and
    // `it_tensors`. Let the scope block allocate these new tensors.
    ffi::Array<TensorVar>& new_tensors = block2new_tensors_[block_stack_.back()];
    ffi::Array<TensorVar> ct_tensors =
        MakeScratchpads(reduction_tensors, /*is_cross_thread_tensor=*/true);
    new_tensors.insert(new_tensors.end(), ct_tensors.begin(), ct_tensors.end());
    ffi::Optional<ffi::Array<TensorVar>> it_tensors = std::nullopt;
    if (need_in_thread_reduction) {
      it_tensors = MakeScratchpads(reduction_tensors, /*is_cross_thread_tensor=*/false);
      new_tensors.insert(new_tensors.end(), it_tensors.value().begin(), it_tensors.value().end());
    }
    // Step 4. Transform.
    loop2new_stmt_.insert_or_assign(
        reduction_loops[0],
        TransformReductionBlock(realize, it_tensors, ct_tensors, reduction_tensors, wb_indices,
                                reducer, combiner_rhs, reduction_loops));

    // Step 5. Record the reduction thread dims for the write-back tensors.
    // The information is used for consumer block broadcasting detection.
    std::vector<std::pair<GPUThreadTag, Range>> reduction_threads;
    reduction_threads.reserve(reduction_loops.size());
    for (const ForObj* loop : reduction_loops) {
      if (tvm::s_tir::GetThreadBinding(loop).has_value()) {
        reduction_threads.emplace_back(
            ParseThreadBindingTag(tvm::s_tir::GetThreadBinding(loop).value()),
            Range(loop->begin, loop->end));
      }
    }
    for (const TensorVar& reduction_tensor : reduction_tensors) {
      crt_tensor2threads_[reduction_tensor.get()] = reduction_threads;
    }
  }

  Stmt MakeCrossThreadBroadcast(
      const SBlockRealizeObj* realize,
      const std::vector<std::pair<GPUThreadTag, Range>>& unbound_thread2range) {
    // Step 1. Generate loop var for each unbound thread.
    // Update the block predicate with clauses of `thread_var == min`.
    PrimExpr predicate = realize->predicate;
    ffi::Array<Var> loop_vars;
    loop_vars.reserve(unbound_thread2range.size());
    for (auto [scope, range] : unbound_thread2range) {
      std::string dim_index(1, static_cast<char>(scope.dim_index + 'x'));
      Var loop_var("t" + dim_index, range->begin.ty());
      loop_vars.push_back(loop_var);
      predicate = (loop_var.as_or_throw<PrimExpr>() == range->begin) && predicate;
    }

    // Step 2. Update the BlockRealize with the new predicate.
    ffi::ObjectPtr<SBlockRealizeObj> p_realize = ffi::make_object<SBlockRealizeObj>(*realize);
    p_realize->predicate = std::move(predicate);

    // Step 3. Wrap the updated BlockRealize with the new loops.
    Stmt body(p_realize);
    for (int i = 0; i < static_cast<int>(unbound_thread2range.size()); ++i) {
      std::string dim_index(1, static_cast<char>(unbound_thread2range[i].first.dim_index + 'x'));
      body = For(loop_vars[i].as_or_throw<PrimVar>(), unbound_thread2range[i].second->begin,
                 unbound_thread2range[i].second->begin + unbound_thread2range[i].second->extent,
                 IntImm(loop_vars[i].as_or_throw<PrimVar>().ty(), 1), ForKind::kParallel, body,
                 {{tvm::s_tir::attr::kThreadBinding.name, ffi::String("threadIdx." + dim_index)}});
    }
    return body;
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* realize, InplaceMode inplace_mode) final {
    // Part 1. Check if the block needs cross-thread reduction rewrite.
    std::vector<const ForObj*> reduction_loops = NeedCrossThreadReduction(realize);
    if (!reduction_loops.empty()) {
      // Keep the original statement during traversal.  The complete transformed
      // subtree replaces it when returning to the first reduction-related loop.
      has_cross_thread_reduction_ = true;
      MakeCrossThreadReduction(realize, reduction_loops);
      return ffi::Unchanged();
    }

    if (!has_cross_thread_reduction_) {
      return StmtExprMutator::Mutate_(realize, inplace_mode);
    }

    // Part 2. Check if the block needs all-thread broadcasting rewrite.
    // We only check this when cross-thread reduction was detected.
    std::vector<std::pair<GPUThreadTag, Range>> unbound_thread2range =
        NeedCrossThreadBroadcast(realize);
    if (!unbound_thread2range.empty()) {
      return MakeCrossThreadBroadcast(realize, unbound_thread2range);
    }

    return StmtExprMutator::Mutate_(realize, inplace_mode);
  }

  bool has_cross_thread_reduction_ = false;
  std::vector<const StmtObj*> statement_stack_;
  std::vector<const ForObj*> loop_stack_;
  std::vector<const SBlockObj*> block_stack_;
  std::unordered_map<const SBlockObj*, ffi::Array<TensorVar>> block2new_tensors_;
  std::unordered_map<const ForObj*, Stmt> loop2new_stmt_;
  ffi::Map<Var, Range> loop_range_map_;
  sym::Analyzer analyzer_;

  int block_idx_depth = 0;
  int thread_idx_depth = 0;
  std::unordered_map<const VarObj*, std::vector<std::pair<GPUThreadTag, Range>>>
      crt_tensor2threads_;
};

namespace transform {

Pass LowerCrossThreadReduction() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    FunctionObj* fptr = f.CopyOnWrite();
    fptr->body = ffi::make_object<CrossThreadReductionTransformer>()
                     ->Mutate(fptr->body)
                     .ValueOrUnchanged(fptr->body);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerCrossThreadReduction");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerCrossThreadReduction", LowerCrossThreadReduction);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
