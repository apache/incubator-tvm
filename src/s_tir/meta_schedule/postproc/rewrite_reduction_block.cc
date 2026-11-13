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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/support/gpu_thread_tag.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*! \brief The visitor that finds all the reduction block to be decomposed */
struct ReductionBlockFinder : public StmtExprVisitor {
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  /*! \brief Find all the reduction blocks that should be decomposed */
  static std::vector<std::pair<StmtSRef, ffi::String>> Find(const ScheduleState& self) {
    std::vector<std::pair<StmtSRef, ffi::String>> results;
    for (const auto& kv : self->mod->functions) {
      GlobalVar g_var = kv.first;
      BaseFunc base_func = kv.second;
      if (const auto* function = base_func.as<FunctionObj>()) {
        auto finder = ffi::make_object<ReductionBlockFinder>();
        finder->Visit(function->body);
        for (const SBlockObj* block : finder->results_) {
          results.emplace_back(self->stmt2ref.at(block), g_var->name_hint);
        }
      }
    }
    return results;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final {
    support::GPUThreadTag thread_scope = GetThreadBindingTag(loop);
    if (IsThreadIdx(thread_scope) || IsBlockIdx(thread_scope)) {
      thread_bound_loop_vars_.insert(loop->var.get());
    }
    return StmtExprVisitor::Visit_(loop);
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeObj* realize) final {
    if (realize->block->init.has_value() && AllReductionIterVarAreUnbound(realize)) {
      results_.push_back(realize->block.get());
    }
    return StmtExprVisitor::Visit_(realize);
  }

  bool AllReductionIterVarAreUnbound(const SBlockRealizeObj* realize) const {
    if (thread_bound_loop_vars_.empty()) {
      return true;
    }
    auto f_find = [this](const VarObj* var) -> bool { return thread_bound_loop_vars_.count(var); };
    auto walkfn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
      return f_find(var.get()) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                               : ffi::WalkResult::Advance();
    };
    const SBlockObj* block = realize->block.get();
    TVM_FFI_ICHECK_EQ(block->iter_vars.size(), realize->iter_values.size());
    int n = block->iter_vars.size();
    for (int i = 0; i < n; ++i) {
      IterVar iter_var = block->iter_vars[i];
      PrimExpr binding = realize->iter_values[i];
      if (iter_var->iter_type == s_tir::kCommReduce) {
        if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(binding, walkfn).has_value()) {
          return false;
        }
      }
    }
    return true;
  }

  /*! \brief The results of the collection */
  std::vector<const SBlockObj*> results_;
  /*! \brief Loop variables that are bound to threads */
  std::unordered_set<const VarObj*> thread_bound_loop_vars_;
};

/*!
 * \brief Find the innermost loop that the `init` of the input block could be decomposed to
 * \param block_sref The StmtSRef of the block to be decomposed
 * \return The index of the innermost loop where the `init` of the input block could be decomposed,
 * or -1 if the `init` does not need to be decomposed.
 */
int FindDecomposePoint(const StmtSRef& block_sref) {
  ffi::Array<StmtSRef> loop_srefs = GetLoops(block_sref);
  int n = loop_srefs.size();
  for (int i = 0; i < n; ++i) {
    if (GetLoopIterType(loop_srefs[i]) != IterVarType::kDataPar) {
      return i;
    }
  }
  return -1;
}

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief Rewrite reduction block by moving the init block out */
class RewriteReductionBlockObj : public PostprocObj {
 public:
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RewriteReductionBlockObj>();
  }

  // Inherited from PostprocObj
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocObj
  bool Apply(const s_tir::Schedule& sch) final;

  Postproc Clone() const {
    ffi::ObjectPtr<RewriteReductionBlockObj> n = ffi::make_object<RewriteReductionBlockObj>(*this);
    return Postproc(ffi::UnsafeInit{}, n);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RewriteReductionBlock",
                                    RewriteReductionBlockObj, PostprocObj);
};

bool RewriteReductionBlockObj::Apply(const s_tir::Schedule& sch) {
  for (;;) {
    std::vector<std::pair<tirx::StmtSRef, ffi::String>> results =
        s_tir::ReductionBlockFinder::Find(sch->state());
    int rewritten = 0;
    for (const auto& kv : results) {
      const tirx::StmtSRef& block_sref = kv.first;
      const ffi::String& global_var_name = kv.second;
      int decompose_point = s_tir::FindDecomposePoint(block_sref);
      if (decompose_point == -1) {
        continue;
      }
      s_tir::SBlockRV block_rv = GetRVFromSRef(sch, block_sref, global_var_name);
      ffi::Array<s_tir::LoopRV> loop_rvs = sch->GetLoops(block_rv);
      s_tir::SBlockRV init_block_rv = sch->DecomposeReduction(block_rv, loop_rvs[decompose_point]);

      // Rewrite auto tensorization related annotations
      if (s_tir::GetAnn<ffi::String>(block_sref, tvm::s_tir::attr::kMetaScheduleAutoTensorize.name)
              .has_value()) {
        // Remove tensorization annotation as it shouldn't be propagated to the init block.
        sch->Unannotate(init_block_rv, tvm::s_tir::attr::kMetaScheduleAutoTensorize.name);
        ffi::Optional<ffi::String> tensorize_init = s_tir::GetAnn<ffi::String>(
            block_sref, tvm::s_tir::attr::kMetaScheduleAutoTensorizeInit.name);
        // The annotation of tensorization of the init statement should be moved to the init block
        // after 'DecomposeReduction'.
        // Annotate to hint `RewriteTensorize` postprocessor even if tensorize_init is std::nullopt.
        sch->Annotate(init_block_rv, tvm::s_tir::attr::kMetaScheduleAutoTensorize.name,
                      tensorize_init.value_or(""));
        if (tensorize_init.has_value()) {
          sch->Unannotate(block_rv, tvm::s_tir::attr::kMetaScheduleAutoTensorizeInit.name);
          sch->Unannotate(init_block_rv, tvm::s_tir::attr::kMetaScheduleAutoTensorizeInit.name);
        }
      }
      ++rewritten;
    }
    if (rewritten == 0) {
      break;
    }
  }
  return true;
}

Postproc Postproc::RewriteReductionBlock() {
  ffi::ObjectPtr<RewriteReductionBlockObj> n = ffi::make_object<RewriteReductionBlockObj>();
  return Postproc(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RewriteReductionBlockObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocRewriteReductionBlock",
                        Postproc::RewriteReductionBlock);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
