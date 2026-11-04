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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*!
 * \brief Check whether the loop has any annotation
 * \param sref The sref of loop
 * \return Whether the loop has any annotation
 */
inline bool HasAnnOrBinding(const ForObj* loop) {
  return tvm::s_tir::GetThreadBinding(loop).has_value() || !loop->annotations.empty();
}

/*! \brief The visitor for extracting the stride of a var in a PrimExpr. */
class StrideExtractor : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  /*!
   * \brief Extracting the stride of a var in a PrimExpr.
   *        e.g the stride of `x` in `(x * 2 + 1) * 3 + 1` is 6
   * \param expr The given PrimExpr.
   * \param var The target var.
   * \return The stride of the var.
   */
  static int64_t Extract(const PrimExpr& expr, const Var& var) {
    auto extractor = ffi::make_object<StrideExtractor>(var);
    extractor->Visit(expr);
    return extractor->strides_[expr.get()];
  }

  explicit StrideExtractor(const Var& var) : var_(var) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const MulObj* node) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(node));

    if (const auto* a = node->a.as<IntImmObj>()) {
      if (strides_.count(node->b.get())) {
        strides_[node] = static_cast<int64_t>(strides_[node->b.get()] * a->value);
      }
    } else if (const auto* b = node->b.as<IntImmObj>()) {
      if (strides_.count(node->a.get())) {
        strides_[node] = static_cast<int64_t>(strides_[node->a.get()] * b->value);
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const AddObj* node) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(node));
    int64_t stride_a, stride_b;
    if (strides_.count(node->a.get())) {
      stride_a = strides_[node->a.get()];
    } else {
      stride_a = INT64_MAX;
    }
    if (strides_.count(node->b.get())) {
      stride_b = strides_[node->b.get()];
    } else {
      stride_b = INT64_MAX;
    }
    if (stride_a != INT64_MAX || stride_b != INT64_MAX) {
      strides_[node] = std::min(stride_a, stride_b);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* node) final {
    if (node == var_.get()) {
      strides_[node] = 1;
    }
    return std::nullopt;
  }

  const Var& var_;
  std::unordered_map<const ExprObj*, int64_t> strides_;
};

struct ParsedAnnotation {
  int max_parallel_extent;
  int max_vectorize_extent;
  int unroll_explicit;
  int unroll_implicit;
  int num_parallel_loops;
  int num_vectorize_loops;
};

bool ParseAnnotation(const SBlock& block, ParsedAnnotation* parsed) {
  bool found = false;
  *parsed = ParsedAnnotation{-1, -1, -1, -1, -1, -1};
  for (const auto& ann : block->annotations) {
    if (ann.first == tvm::s_tir::attr::kMetaScheduleParallel.name) {
      found = true;
      if (auto opt_int_imm = ann.second.try_cast<IntImm>()) {
        parsed->max_parallel_extent = (*opt_int_imm)->value.as<int>().value();
      }
    } else if (ann.first == tvm::s_tir::attr::kMetaScheduleVectorize.name) {
      found = true;
      if (auto opt_int_imm = ann.second.try_cast<IntImm>()) {
        parsed->max_vectorize_extent = (*opt_int_imm)->value.as<int>().value();
      }
    } else if (ann.first == tvm::s_tir::attr::kMetaScheduleUnrollExplicit.name) {
      found = true;
      if (auto opt_int_imm = ann.second.try_cast<IntImm>()) {
        parsed->unroll_explicit = (*opt_int_imm)->value.as<int>().value();
      }
    } else if (ann.first == tvm::s_tir::attr::kMetaScheduleUnrollImplicit.name) {
      found = true;
      if (auto opt_int_imm = ann.second.try_cast<IntImm>()) {
        parsed->unroll_implicit = (*opt_int_imm)->value.as<int>().value();
      }
    }
  }
  return found;
}

void RemoveParsedAnn(const Schedule& sch, const SBlockRV& block_rv,
                     const ParsedAnnotation& parsed) {
  if (parsed.max_parallel_extent != -1) {
    sch->Unannotate(block_rv, tvm::s_tir::attr::kMetaScheduleParallel.name);
  }
  if (parsed.max_vectorize_extent != -1) {
    sch->Unannotate(block_rv, tvm::s_tir::attr::kMetaScheduleVectorize.name);
  }
  if (parsed.unroll_explicit != -1) {
    sch->Unannotate(block_rv, tvm::s_tir::attr::kMetaScheduleUnrollExplicit.name);
  }
  if (parsed.unroll_implicit != -1) {
    sch->Unannotate(block_rv, tvm::s_tir::attr::kMetaScheduleUnrollImplicit.name);
  }
}

int CalculateNumRewritableLoops(const ffi::Array<StmtSRef>& loop_srefs,
                                const std::vector<int>& loop_types) {
  int rw_loops_num = 0;
  TVM_FFI_ICHECK_EQ(loop_srefs.size(), loop_types.size());
  for (size_t i = 0; i < loop_srefs.size(); ++i) {
    const StmtSRef& loop_sref = loop_srefs[i];
    const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
    if (HasAnnOrBinding(loop)) {
      continue;
    }
    // Cannot vectorize reduce axis
    if (loop_types[i] != IterVarType::kDataPar) {
      continue;
    }
    // Cannot fuse with a loop with multiple children
    if (!IsSingleStmt(loop->body)) {
      continue;
    }
    // Check if the loop extent is valid
    PrimExpr extent_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
    if (const auto* extent = extent_expr.as<IntImmObj>();
        !extent || !extent->value.as<int64_t>().has_value()) {
      continue;
    }
    ++rw_loops_num;
  }
  return rw_loops_num;
}

void AdjustParallelVectorize(const Schedule& sch, const SBlockRV& block_rv,
                             const ffi::Array<LoopRV>& loop_rvs, ParsedAnnotation* parsed) {
  StmtSRef block_sref = sch->GetSRef(block_rv);
  if (parsed->max_parallel_extent == -1 && parsed->max_vectorize_extent == -1) {
    return;
  }
  const int n_loops = loop_rvs.size();
  if (n_loops == 0) {
    parsed->max_parallel_extent = -1;
    parsed->max_vectorize_extent = -1;
    return;
  }
  // Extract loop_srefs, and calculate the iterator types
  ffi::Array<StmtSRef> loop_srefs;
  std::vector<int> loop_types;
  {
    loop_srefs.reserve(n_loops);
    loop_types.reserve(n_loops);
    for (const LoopRV& loop_rv : loop_rvs) {
      loop_srefs.push_back(sch->GetSRef(loop_rv));
      loop_types.push_back(GetLoopIterType(loop_srefs.back()));
    }
  }
  // check the maximal number of axes that are vectorizable (contiguous memory access)
  SBlockRealize realize = GetSBlockRealize(sch->state(), block_sref);
  ffi::Array<TensorRegion> tensor_access(realize->block->reads);
  tensor_access.insert(tensor_access.end(), realize->block->writes.begin(),
                       realize->block->writes.end());
  std::unordered_map<const VarObj*, PrimExpr> binding_map;
  for (size_t i = 0; i < realize->iter_values.size(); i++) {
    binding_map.insert_or_assign(realize->block->iter_vars[i]->var.get(), realize->iter_values[i]);
  }
  auto f_substitute = [&binding_map](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto it = binding_map.find(var.get()); it != binding_map.end()) {
      return ffi::Any(it->second);
    }
    return ffi::Unchanged();
  };
  int max_fusible = INT32_MAX;
  // for each block read/write, get the strides of the loop vars and find the fusible
  // (vectorizable) axes
  for (const TensorRegion& access : tensor_access) {
    int fusible = 0;
    bool can_analyze_contiguous_access = true;
    std::vector<int64_t> strides;
    // get strides for each loop var
    for (const StmtSRef& loop_sref : loop_srefs) {
      int64_t stride = 0, tensor_stride = 1;
      const auto* var = loop_sref->StmtAs<ForObj>();
      sym::Analyzer analyzer;
      for (int i = access->region.size() - 1; i >= 0; i--) {
        PrimExpr idx = analyzer->Simplify(
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(access->region[i]->begin, f_substitute)
                .as_or_throw<PrimExpr>());
        int64_t coef = StrideExtractor::Extract(idx, var->var);
        if (coef != 0) {
          stride = coef * tensor_stride;
          break;
        }
        const auto* shape =
            access->source.as_or_throw<tvm::tirx::TensorVar>().shape()[i].as<IntImmObj>();
        if (shape == nullptr) {
          can_analyze_contiguous_access = false;
          break;
        }
        tensor_stride = static_cast<int64_t>(tensor_stride * shape->value);
      }
      if (!can_analyze_contiguous_access) {
        break;
      }
      strides.push_back(stride);
    }
    if (!can_analyze_contiguous_access) {
      max_fusible = 0;
      break;
    }
    int prev_used_iter = -1;
    // check the number of fusible loops
    for (int i = strides.size() - 1; i >= 0; i--) {
      if (strides[i] == 0) {
        // not used in the tensor access, safe to fuse
        fusible++;
        continue;
      } else if (prev_used_iter == -1) {
        // the stride of last axis is not 1 means the memory access is not contiguous
        if (strides[i] != 1 && fusible != 0) {
          break;
        }
        fusible++;
        prev_used_iter = i;
      } else {
        // contiguous memory access
        PrimExpr prev_used_iter_extent_imm_expr =
            Range::GetExtentFromBeginEnd(TVM_SREF_TO_FOR(loop_srefs[prev_used_iter])->begin,
                                         TVM_SREF_TO_FOR(loop_srefs[prev_used_iter])->end);
        const auto* prev_used_iter_extent_imm = prev_used_iter_extent_imm_expr.as<IntImmObj>();
        auto prev_used_iter_extent = prev_used_iter_extent_imm
                                         ? prev_used_iter_extent_imm->value.as<int64_t>()
                                         : std::nullopt;
        if (!prev_used_iter_extent.has_value()) {
          break;
        }
        if (strides[i] == strides[prev_used_iter] * (*prev_used_iter_extent)) {
          fusible++;
          prev_used_iter = i;
        } else {
          break;
        }
      }
    }
    max_fusible = std::min(max_fusible, fusible);
  }

  // Calculate how many loops are rewritable, i.e. valid for vectorization and parallelization.
  int max_rw_loops = CalculateNumRewritableLoops(loop_srefs, loop_types);

  // Calculate the parallelize extent
  if (parsed->max_parallel_extent != -1) {
    int max_extent = parsed->max_parallel_extent;
    int& num_fusible = parsed->num_parallel_loops = 0;
    int64_t prod_extent = 1;
    for (int i = 0; i < n_loops && loop_types[i] == IterVarType::kDataPar; ++i) {
      const StmtSRef& loop_sref = loop_srefs[i];
      const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
      if (HasAnnOrBinding(loop)) {
        break;
      }
      // Check if the loop extent is valid
      PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
      const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
      auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
      if (!extent.has_value()) {
        break;
      }
      // Then we can fuse it in
      ++num_fusible;
      // Check if we need to break
      prod_extent *= *extent;
      if (prod_extent > max_extent || !IsSingleStmt(loop->body)) {
        break;
      }
    }
    if (prod_extent == 1) {
      num_fusible = -1;
    }
  }
  // Calculate the vectorize extent
  if (parsed->max_vectorize_extent != -1) {
    int max_extent = parsed->max_vectorize_extent;
    int& num_fusible = parsed->num_vectorize_loops = 0;
    int64_t prod_extent = 1;
    for (int i = n_loops - 1;
         i >= 0 && loop_types[i] == IterVarType::kDataPar && num_fusible < max_fusible; --i) {
      const StmtSRef& loop_sref = loop_srefs[i];
      const ForObj* loop = TVM_SREF_TO_FOR(loop_sref);
      if (HasAnnOrBinding(loop)) {
        break;
      }
      // Cannot vectorize reduce axis
      if (GetLoopIterType(loop_sref) != IterVarType::kDataPar) {
        break;
      }
      // Cannot fuse with a loop with multiple children
      if (!IsSingleStmt(loop->body)) {
        break;
      }
      // Check if the loop extent is valid
      PrimExpr extent_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
      const auto* extent_imm = extent_imm_expr.as<IntImmObj>();
      auto extent = extent_imm ? extent_imm->value.as<int64_t>() : std::nullopt;
      if (!extent.has_value()) {
        break;
      }
      // Check if the extent is still in a good range
      prod_extent *= *extent;
      if (prod_extent > max_extent) {
        break;
      }
      ++num_fusible;
    }
    if (prod_extent == 1) {
      num_fusible = -1;
    }
  }

  if (parsed->num_parallel_loops != -1 && parsed->num_vectorize_loops != -1) {
    if (max_rw_loops == n_loops && max_fusible == n_loops) {
      // All loops can be fused, parallelized and vectorized
      parsed->num_parallel_loops = n_loops;
      parsed->num_vectorize_loops = n_loops;
    } else {
      // Prefer num_vectorize to num_parallel
      parsed->num_parallel_loops =
          std::min(parsed->num_parallel_loops, n_loops - parsed->num_vectorize_loops);
    }
  }
}

bool FindAnnotatedRootBlock(const Schedule& sch, ParsedAnnotation* parsed, SBlockRV* root_rv) {
  IRModule mod = sch->mod();
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* function = base_func.as<FunctionObj>()) {
      const SBlockRealizeObj* block_realize =
          function->body.has_value() && function->body.value()->size() == 1
              ? function->body.value()->seq[0].as<SBlockRealizeObj>()
              : nullptr;
      if (block_realize != nullptr) {
        SBlock block = block_realize->block;
        if (ParseAnnotation(block, parsed)) {
          *root_rv = sch->GetSBlock(block->name_hint, g_var->name_hint);
          RemoveParsedAnn(sch, *root_rv, *parsed);
          return true;
        }
      }
    }
  }
  return false;
}

void RewriteFuseSplitParallelVectorize(const Schedule& sch, ffi::Array<LoopRV>* loop_rvs,
                                       int vec_len) {
  size_t n_loops = loop_rvs->size();
  LoopRV fused = sch->Fuse({loop_rvs->begin(), loop_rvs->end()});
  ffi::Array<LoopRV> split = sch->Split(fused, {std::nullopt, IntImm::Int32(vec_len)});
  TVM_FFI_ICHECK_EQ(split.size(), 2);
  const LoopRV& outer = split[0];
  const LoopRV& inner = split[1];
  sch->Parallel(outer);
  sch->Vectorize(inner);
  for (size_t i = 0; i < n_loops - 1; ++i) {
    loop_rvs->Set(i, outer);
  }
  loop_rvs->Set(n_loops - 1, inner);
}

void RewriteParallel(const Schedule& sch, size_t n, ffi::Array<LoopRV>* loop_rvs) {
  TVM_FFI_ICHECK_LE(n, loop_rvs->size());
  LoopRV fused = sch->Fuse({loop_rvs->begin(), loop_rvs->begin() + n});
  sch->Parallel(fused);
  for (size_t i = 0; i < n; ++i) {
    loop_rvs->Set(i, fused);
  }
}

void RewriteVectorize(const Schedule& sch, size_t n, ffi::Array<LoopRV>* loop_rvs) {
  size_t n_loops = loop_rvs->size();
  TVM_FFI_ICHECK_LE(n, n_loops);
  LoopRV fused = sch->Fuse({loop_rvs->end() - n, loop_rvs->end()});
  sch->Vectorize(fused);
  for (size_t i = n_loops - n; i < n_loops; ++i) {
    loop_rvs->Set(i, fused);
  }
}

void RewriteUnroll(const Schedule& sch, int unroll_explicit, int max_step, const SBlockRV& block,
                   const LoopRV& loop) {
  // Do not unroll for pure spatial block.
  if (max_step <= 0 || IsSpatial(sch->GetSRef(block))) {
    return;
  }

  sch->Annotate(loop, tvm::tirx::attr::kAutoUnrollMaxStep.name, max_step);
  sch->Annotate(loop, tvm::tirx::attr::kUnrollExplicit.name, unroll_explicit);
}

}  // namespace s_tir

namespace s_tir {
namespace meta_schedule {

using s_tir::Schedule;

class RewriteParallelVectorizeUnrollObj : public PostprocObj {
 public:
  void InitializeWithTuneContext(const TuneContext& context) final {}

  bool Apply(const Schedule& sch) final {
    s_tir::ParsedAnnotation parsed_root;
    s_tir::SBlockRV root_rv{ffi::UnsafeInit()};
    while (s_tir::FindAnnotatedRootBlock(sch, &parsed_root, &root_rv)) {
      for (s_tir::SBlockRV block_rv : sch->GetChildBlocks(root_rv)) {
        ffi::Array<s_tir::LoopRV> loop_rvs = sch->GetLoops(block_rv);
        if (loop_rvs.empty()) {
          continue;
        }
        s_tir::ParsedAnnotation parsed = parsed_root;
        s_tir::AdjustParallelVectorize(sch, block_rv, loop_rvs, &parsed);
        const int loops_num = loop_rvs.size();
        try {
          if (parsed.num_parallel_loops == loops_num && parsed.num_vectorize_loops == loops_num) {
            // Fuse, split, vectorize and parallelize
            s_tir::RewriteFuseSplitParallelVectorize(sch, &loop_rvs, parsed.max_vectorize_extent);
          } else {
            // Parallel
            if (parsed.num_parallel_loops > 0) {
              s_tir::RewriteParallel(sch, parsed.num_parallel_loops, &loop_rvs);
            }
            // Vectorize
            if (parsed.num_vectorize_loops > 0) {
              s_tir::RewriteVectorize(sch, parsed.num_vectorize_loops, &loop_rvs);
            }
          }
          // AutoUnroll
          if (parsed.unroll_explicit != -1 || parsed.unroll_implicit != -1) {
            TVM_FFI_ICHECK(parsed.unroll_explicit == -1 || parsed.unroll_implicit == -1);
            int unroll_explicit = parsed.unroll_explicit != -1;
            int max_step = parsed.unroll_explicit + parsed.unroll_implicit + 1;
            s_tir::RewriteUnroll(sch, unroll_explicit, max_step, block_rv, loop_rvs[0]);
          }
        } catch (const ffi::Error& e) {
          if (s_tir::GetScheduleErrorContext(e) == nullptr) {
            throw;
          }
          DLOG(WARNING) << "Failed to apply parallelization/vectorization: " << e.what();
          return false;
        }
      }
    }
    return true;
  }

  Postproc Clone() const {
    ffi::ObjectPtr<RewriteParallelVectorizeUnrollObj> n =
        ffi::make_object<RewriteParallelVectorizeUnrollObj>(*this);
    return Postproc(ffi::UnsafeInit{}, n);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RewriteParallelVectorizeUnrollObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RewriteParallelVectorizeUnroll",
                                    RewriteParallelVectorizeUnrollObj, PostprocObj);
};

Postproc Postproc::RewriteParallelVectorizeUnroll() {
  ffi::ObjectPtr<RewriteParallelVectorizeUnrollObj> n =
      ffi::make_object<RewriteParallelVectorizeUnrollObj>();
  return Postproc(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RewriteParallelVectorizeUnrollObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocRewriteParallelVectorizeUnroll",
                        Postproc::RewriteParallelVectorizeUnroll);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
