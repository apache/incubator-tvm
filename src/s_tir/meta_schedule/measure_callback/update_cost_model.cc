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

#include "../utils.h"

namespace tvm {
namespace s_tir {
namespace meta_schedule {

class UpdateCostModelObj : public MeasureCallbackObj {
 public:
  void Apply(const TaskScheduler& task_scheduler, int task_id,
             const ffi::Array<MeasureCandidate>& measure_candidates,
             const ffi::Array<BuilderResult>& builder_results,
             const ffi::Array<RunnerResult>& runner_results) final {
    auto _ = Profiler::TimedScope("MeasureCallback/UpdateCostModel");
    const TaskRecord& task = task_scheduler->tasks_[task_id];
    if (!task_scheduler->cost_model_.has_value()) {
      return;
    }
    CostModel cost_model = task_scheduler->cost_model_.value();
    TVM_FFI_ICHECK(task->measure_candidates.has_value())
        << "Task's measure candidates must be present!";
    TVM_FFI_ICHECK_EQ(measure_candidates.size(), builder_results.size());
    TVM_FFI_ICHECK_EQ(runner_results.size(), builder_results.size());
    int n = builder_results.size();
    ffi::Array<MeasureCandidate> pruned_candidate;
    ffi::Array<RunnerResult> pruned_runner_result;
    pruned_candidate.reserve(n);
    pruned_runner_result.reserve(n);
    for (int i = 0; i < n; i++) {
      if (!builder_results[i]->error_msg.has_value() &&  //
          (runner_results[i]->error_msg.has_value() ||   //
           (runner_results[i]->run_secs.has_value() &&
            Sum(runner_results[i]->run_secs.value()) > 0))) {
        pruned_candidate.push_back(measure_candidates[i]);
        pruned_runner_result.push_back(runner_results[i]);
      }
    }
    cost_model->Update(task->ctx, pruned_candidate, pruned_runner_result);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<UpdateCostModelObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.UpdateCostModel", UpdateCostModelObj,
                                    MeasureCallbackObj);
};

MeasureCallback MeasureCallback::UpdateCostModel() {
  ffi::ObjectPtr<UpdateCostModelObj> n = ffi::make_object<UpdateCostModelObj>();
  return MeasureCallback(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  UpdateCostModelObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.MeasureCallbackUpdateCostModel",
                        MeasureCallback::UpdateCostModel);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
