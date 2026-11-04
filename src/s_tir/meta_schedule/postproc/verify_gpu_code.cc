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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/transform.h>
#include <tvm/support/gpu_thread_tag.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/transform.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

class ThreadExtentChecker : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  static bool Check(const Stmt& stmt, int thread_warp_size) {
    try {
      TVM_FFI_ICHECK(thread_warp_size > 0);
      auto checker = ffi::make_object<ThreadExtentChecker>(thread_warp_size);
      checker->Visit(stmt);
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

  explicit ThreadExtentChecker(int thread_warp_size) : thread_warp_size_(thread_warp_size) {}

 private:
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) {
    support::GPUThreadTag thread_scope = GetThreadBindingTag(loop);
    if (IsThreadIdx(thread_scope)) {
      PrimExpr p_ext_imm_expr = Range::GetExtentFromBeginEnd(loop->begin, loop->end);
      const auto* p_ext_imm = p_ext_imm_expr.as<IntImmObj>();
      if (auto p_ext = p_ext_imm ? p_ext_imm->value.as<int64_t>() : std::nullopt;
          p_ext.has_value()) {
        int64_t ext = *p_ext;
        if (thread_scope.dim_index == 0) {
          std::swap(thread_idx_x, ext);
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
          std::swap(thread_idx_x, ext);
        } else if (thread_scope.dim_index == 1) {
          std::swap(thread_idx_y, ext);
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
          std::swap(thread_idx_y, ext);
        } else if (thread_scope.dim_index == 2) {
          std::swap(thread_idx_z, ext);
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
          std::swap(thread_idx_z, ext);
        } else {
          TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
        }
        return std::nullopt;
      } else {
        throw std::runtime_error("Dynamic thread extent");
      }
    }
    return StmtExprVisitor::Visit_(loop);
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) {
    int old_thread_idx_x = thread_idx_x;
    if (block->annotations.count(tvm::s_tir::attr::kWarpExecution.name)) {
      thread_idx_x = thread_warp_size_;
    }
    if (ffi::Optional<int64_t> low_inclusive =
            GetAnn<int64_t>(block, tvm::s_tir::attr::kMetaScheduleThreadExtentLowInclusive.name)) {
      if (ffi::Optional<int64_t> high_inclusive = GetAnn<int64_t>(
              block, tvm::s_tir::attr::kMetaScheduleThreadExtentHighInclusive.name)) {
        int64_t low = low_inclusive.value();
        int64_t high = high_inclusive.value();
        int64_t thread_extent_product = thread_idx_x * thread_idx_y * thread_idx_z;
        if (!(low <= thread_extent_product && thread_extent_product <= high)) {
          throw std::runtime_error("Thread extent");
        }
      }
    }
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(block));
    thread_idx_x = old_thread_idx_x;
    return std::nullopt;
  }

  int64_t thread_idx_x = 1;
  int64_t thread_idx_y = 1;
  int64_t thread_idx_z = 1;
  int thread_warp_size_ = -1;
};

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief Extract attribute from a target. */
IntImm Extract(const Target& target, const char* name) {
  TVM_FFI_ICHECK(target.defined());
  if (ffi::Optional<int64_t> v = target->GetAttr<int64_t>(name)) {
    return IntImm::Int64(v.value());
  }
  TVM_FFI_THROW(AttributedError) << "\"" << name << "\" is not defined in the target";
  throw;
}

/*! \brief Verify the correctness of the generated GPU code. */
class VerifyGPUCodeObj : public PostprocObj {
 public:
  ffi::Optional<Target> target_;
  ffi::Map<ffi::String, PrimExpr> target_constraints_{ffi::UnsafeInit()};
  int thread_warp_size_ = -1;

  void InitializeWithTuneContext(const TuneContext& context) final {
    TVM_FFI_ICHECK(context->target.has_value());
    this->target_ = context->target.value();
    this->target_constraints_ = ffi::Map<ffi::String, PrimExpr>{
        {"max_shared_memory_per_block",
         Extract(this->target_.value(), "max_shared_memory_per_block")},
        {"max_threads_per_block", Extract(this->target_.value(), "max_threads_per_block")},
        {"max_vthread", IntImm::Int32(8)},
        {"max_vector_bytes", IntImm::Int32(16)},
    };
    thread_warp_size_ = Extract(this->target_.value(), "thread_warp_size")->value.as<int>().value();
  }

  bool Verify(const IRModule& mod) const {
    for (const auto& kv : mod->functions) {
      if (auto function = kv.second.as<tirx::Function>()) {
        if (!s_tir::VerifyGPUCode(function.value(), this->target_constraints_)) {
          return false;
        }
      }
    }
    return true;
  }

  bool Apply(const s_tir::Schedule& sch) final {
    IRModule mod = sch->mod();
    for (const auto& kv : mod->functions) {
      const GlobalVar& g_var = kv.first;
      const BaseFunc& base_func = kv.second;
      if (const auto* function = base_func.as<tirx::FunctionObj>()) {
        if (function->body.has_value() &&
            !s_tir::ThreadExtentChecker::Check(function->body.value(), thread_warp_size_)) {
          return false;
        }
        IRModule lowered{ffi::UnsafeInit()};
        try {
          auto pass_list = ffi::Array<tvm::transform::Pass>();
          // Phase 1
          pass_list.push_back(s_tir::transform::LowerCrossThreadReduction());
          pass_list.push_back(s_tir::transform::LowerInitBlock());
          pass_list.push_back(s_tir::transform::PlanAndUpdateTensorAllocationLocation());
          pass_list.push_back(s_tir::transform::ConvertBlocksToOpaque());
          pass_list.push_back(s_tir::transform::LiftThreadBinding());
          pass_list.push_back(s_tir::transform::ManifestSharedMemoryLocalStage());
          pass_list.push_back(s_tir::transform::CompactTensorAllocation());
          pass_list.push_back(s_tir::transform::StmtSimplify());
          pass_list.push_back(s_tir::transform::LowerAutoCopy());
          pass_list.push_back(s_tir::transform::LowerMatchTensor());
          pass_list.push_back(s_tir::transform::InjectSoftwarePipeline());
          pass_list.push_back(s_tir::transform::LowerOpaqueBlock());
          pass_list.push_back(s_tir::transform::LowerThreadBinding());
          pass_list.push_back(tirx::transform::FlattenTensor());
          pass_list.push_back(tirx::transform::BF16ComputeLegalize());
          pass_list.push_back(tirx::transform::NarrowDataType(32));
          pass_list.push_back(tirx::transform::StmtSimplify());
          // Phase 2
          pass_list.push_back(tirx::transform::VectorizeLoop(true));
          pass_list.push_back(s_tir::transform::InjectVirtualThread());
          pass_list.push_back(tirx::transform::StorageRewrite());
          pass_list.push_back(s_tir::transform::MergeSharedMemoryAllocations());
          pass_list.push_back(tirx::transform::LowerIntrin());
          // Convert Function to IRModule
          tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();
          tirx::Function f = WithAttr(ffi::GetRef<tirx::Function>(function),
                                      tvm::attr::kGlobalSymbol.name, ffi::String(g_var->name_hint));
          f = WithAttr(f, tvm::attr::kTarget.name,
                       this->target_.value());  // Required for LowerIntrin
          bool noalias = pass_ctx->GetConfig<bool>(tvm::tirx::attr::kNoAlias.name).value_or(true);
          if (noalias) {
            f = WithAttr(std::move(f), tvm::tirx::attr::kNoAlias.name, true);
          }
          IRModule mod =
              IRModule(ffi::Map<GlobalVar, BaseFunc>({{GlobalVar(g_var->name_hint), f}}));
          lowered = tvm::transform::Sequential(pass_list)(std::move(mod));
        } catch (const std::exception&) {
          return false;
        }
        if (!Verify(lowered)) {
          return false;
        }
      }
    }
    return true;
  }

  Postproc Clone() const {
    ffi::ObjectPtr<VerifyGPUCodeObj> n = ffi::make_object<VerifyGPUCodeObj>(*this);
    n->target_constraints_ = this->target_constraints_;
    return Postproc(ffi::UnsafeInit{}, n);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<VerifyGPUCodeObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.VerifyGPUCode", VerifyGPUCodeObj,
                                    PostprocObj);
};

Postproc Postproc::VerifyGPUCode() {
  ffi::ObjectPtr<VerifyGPUCodeObj> n = ffi::make_object<VerifyGPUCodeObj>();
  return Postproc(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  VerifyGPUCodeObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocVerifyGPUCode", Postproc::VerifyGPUCode);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
