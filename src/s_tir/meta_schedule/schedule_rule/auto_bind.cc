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

#include <algorithm>
#include <limits>

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class AutoBindObj : public ScheduleRuleObj {
 public:
  // Inherited from ScheduleRuleObj
  void InitializeWithTuneContext(const TuneContext& context) final {
    TVM_FFI_CHECK(context->target.has_value(), ValueError) << "target is not defined";
    ffi::Optional<int64_t> max_threads_per_block =
        context->target.value()->GetAttr<int64_t>("max_threads_per_block");
    TVM_FFI_CHECK(max_threads_per_block.has_value(), ValueError)
        << "missing attribute `max_threads_per_block` in the target";
    this->max_threads_per_block_ = max_threads_per_block.value();
  }

  // Inherited from ScheduleRuleObj
  ffi::Array<s_tir::Schedule> Apply(const s_tir::Schedule& sch,
                                    const s_tir::SBlockRV& block_rv) final;

  // Inherited from ScheduleRuleObj
  ScheduleRule Clone() const final {
    ffi::ObjectPtr<AutoBindObj> n = ffi::make_object<AutoBindObj>(*this);
    return ScheduleRule(ffi::UnsafeInit{}, n);
  }

 public:
  /*! \brief The max number of threads per block from Target */
  int64_t max_threads_per_block_ = -1;
  /*! \brief The max number of threadblocks in the CUDA device */
  int64_t max_threadblocks_ = -1;
  /*! \brief thread_extents Candidates of thread axis extent. */
  ffi::Array<int64_t> thread_extents_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AutoBindObj>();
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.AutoBind", AutoBindObj, ScheduleRuleObj);
};

ffi::Array<s_tir::Schedule> AutoBindObj::Apply(const s_tir::Schedule& sch,
                                               const s_tir::SBlockRV& block_rv) {
  TVM_FFI_ICHECK_NE(this->max_threads_per_block_, -1);
  auto get_factor = MakeFactorSampler(sch, this->thread_extents_);
  BindBlockThreadIdx(sch, block_rv, max_threadblocks_, max_threads_per_block_, get_factor);
  return {sch};
}

ScheduleRule ScheduleRule::AutoBind(int max_threadblocks, ffi::Array<int64_t> thread_extents,
                                    int max_threads_per_block) {
  ffi::ObjectPtr<AutoBindObj> n = ffi::make_object<AutoBindObj>();
  n->max_threadblocks_ = max_threadblocks;
  n->max_threads_per_block_ = max_threads_per_block;
  n->thread_extents_ = std::move(thread_extents);
  return ScheduleRule(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() { AutoBindObj::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.meta_schedule.ScheduleRuleAutoBind", ScheduleRule::AutoBind);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
