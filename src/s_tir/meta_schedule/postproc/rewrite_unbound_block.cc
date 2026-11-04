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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/meta_schedule/schedule/cuda/thread_bind.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/support/gpu_thread_tag.h>

#include "../utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

/*! \brief Find all the blocks that are not bound */
class UnboundBlockFinder : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }

  static std::vector<std::pair<StmtSRef, ffi::String>> Find(const ScheduleState& self) {
    auto finder = ffi::make_object<UnboundBlockFinder>(self);
    for (const auto& kv : self->mod->functions) {
      GlobalVar g_var = kv.first;
      BaseFunc base_func = kv.second;
      if (const auto* function = base_func.as<FunctionObj>()) {
        finder->global_var_name_ = g_var->name_hint;
        TVM_FFI_ICHECK(function->body.has_value());
        TVM_FFI_ICHECK_EQ(function->body.value()->size(), 1);
        finder->Visit(function->body.value()->seq[0].as_or_throw<SBlockRealize>()->block->body);
      }
    }
    return std::move(finder->blocks_);
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* loop) final {
    support::GPUThreadTag thread_scope = GetThreadBindingTag(loop);
    if (IsBlockIdx(thread_scope)) {
      ++n_block_idx_;
    } else if (IsThreadIdx(thread_scope)) {
      ++n_thread_idx_;
    }
    if (n_block_idx_ == 0 || n_thread_idx_ == 0) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(loop));
    }
    if (IsBlockIdx(thread_scope)) {
      --n_block_idx_;
    } else if (IsThreadIdx(thread_scope)) {
      --n_thread_idx_;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockObj* block) final {
    blocks_.emplace_back(self_->stmt2ref.at(block), global_var_name_);
    return std::nullopt;
  }

 public:
  explicit UnboundBlockFinder(const ScheduleState& self)
      : self_{self}, blocks_{}, n_block_idx_{0}, n_thread_idx_{0} {}

 private:
  /*! \brief The schedule state */
  const ScheduleState& self_;
  /*! \brief The list of unbound blocks */
  std::vector<std::pair<StmtSRef, ffi::String>> blocks_;
  /*!  \brief The number of blockIdx above the current stmt */
  int n_block_idx_;
  /*!  \brief The number of threadIdx above the current stmt */
  int n_thread_idx_;
  /*! \brief The name of the global var */
  ffi::String global_var_name_;
};

}  // namespace s_tir
}  // namespace tvm

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief Add thread binding to unbound blocks */
class RewriteUnboundBlockObj : public PostprocObj {
 public:
  // Inherited from PostprocObj
  void InitializeWithTuneContext(const TuneContext& context) final {
    TVM_FFI_CHECK(context->target.has_value(), ValueError) << "target is not defined";
    ffi::Optional<int64_t> max_threads_per_block =
        context->target.value()->GetAttr<int64_t>("max_threads_per_block");
    TVM_FFI_CHECK(max_threads_per_block.has_value(), ValueError)
        << "missing attribute `max_threads_per_block` in the target";
    this->max_threads_per_block_ = max_threads_per_block.value();
  }

  // Inherited from PostprocObj
  bool Apply(const s_tir::Schedule& sch) final;

  Postproc Clone() const {
    ffi::ObjectPtr<RewriteUnboundBlockObj> n = ffi::make_object<RewriteUnboundBlockObj>(*this);
    return Postproc(ffi::UnsafeInit{}, n);
  }

 public:
  /*! \brief The max number of threads per block from Target */
  int max_threads_per_block_ = -1;
  /*! \brief The max number of threadblocks in the CUDA device */
  int max_threadblocks_ = -1;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RewriteUnboundBlockObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RewriteUnboundBlock",
                                    RewriteUnboundBlockObj, PostprocObj);
};

bool RewriteUnboundBlockObj::Apply(const s_tir::Schedule& sch) {
  using s_tir::ExprRV;
  using s_tir::LoopRV;
  using s_tir::SBlockRV;
  using s_tir::Schedule;
  TVM_FFI_ICHECK_NE(this->max_threads_per_block_, -1);
  auto get_factor = [t = this->max_threads_per_block_](int max_extent) -> ExprRV {
    return IntImm::Int32(std::min(t, max_extent));
  };
  std::vector<std::pair<tirx::StmtSRef, ffi::String>> unbound_blocks =
      s_tir::UnboundBlockFinder::Find(sch->state());
  for (const auto& kv : unbound_blocks) {
    tirx::StmtSRef block_sref = kv.first;
    ffi::String global_var_name = kv.second;
    SBlockRV block_rv = GetRVFromSRef(sch, block_sref, global_var_name);
    BindBlockThreadIdx(sch, block_rv, max_threadblocks_, max_threads_per_block_, get_factor);
  }
  return true;
}

Postproc Postproc::RewriteUnboundBlock(int max_threadblocks) {
  ffi::ObjectPtr<RewriteUnboundBlockObj> n = ffi::make_object<RewriteUnboundBlockObj>();
  n->max_threadblocks_ = max_threadblocks;
  n->max_threads_per_block_ = -1;
  return Postproc(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { RewriteUnboundBlockObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.PostprocRewriteUnboundBlock",
                        Postproc::RewriteUnboundBlock);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
