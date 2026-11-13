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

class AddToDatabaseObj : public MeasureCallbackObj {
 public:
  void Apply(const TaskScheduler& task_scheduler, int task_id,
             const ffi::Array<MeasureCandidate>& measure_candidates,
             const ffi::Array<BuilderResult>& builder_results,
             const ffi::Array<RunnerResult>& runner_results) final {
    if (!task_scheduler->database_.has_value()) {
      return;
    }
    auto _ = Profiler::TimedScope("MeasureCallback/AddToDatabase");
    TuneContext task = task_scheduler->tasks_[task_id]->ctx;
    Database database = task_scheduler->database_.value();
    Workload workload = database->CommitWorkload(task->mod.value());
    Target target = task->target.value();
    TVM_FFI_ICHECK_EQ(runner_results.size(), measure_candidates.size());
    int n = runner_results.size();
    for (int i = 0; i < n; ++i) {
      RunnerResult result = runner_results[i];
      MeasureCandidate candidate = measure_candidates[i];
      ffi::Array<FloatImm> run_secs;
      if (result->run_secs.has_value()) {
        run_secs = result->run_secs.value();
      } else {
        run_secs = ffi::Array<FloatImm>{FloatImm(PrimType::Float(32), 1e10)};
      }
      database->CommitTuningRecord(TuningRecord(
          /*trace=*/candidate->sch->trace().value(),
          /*workload=*/workload,
          /*run_secs=*/run_secs,
          /*target=*/target,
          /*args_info=*/candidate->args_info));
    }
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AddToDatabaseObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.AddToDatabase", AddToDatabaseObj,
                                    MeasureCallbackObj);
};

MeasureCallback MeasureCallback::AddToDatabase() {
  ffi::ObjectPtr<AddToDatabaseObj> n = ffi::make_object<AddToDatabaseObj>();
  return MeasureCallback(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AddToDatabaseObj>();
  refl::GlobalDef().def("s_tir.meta_schedule.MeasureCallbackAddToDatabase",
                        MeasureCallback::AddToDatabase);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
