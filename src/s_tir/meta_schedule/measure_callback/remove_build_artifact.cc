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

class RemoveBuildArtifactObj : public MeasureCallbackObj {
 public:
  void Apply(const TaskScheduler& task_scheduler, int task_id,
             const ffi::Array<MeasureCandidate>& measure_candidates,
             const ffi::Array<BuilderResult>& builder_results,
             const ffi::Array<RunnerResult>& runner_results) final {
    static auto f_rm =
        tvm::ffi::Function::GetGlobalRequired("s_tir.meta_schedule.remove_build_dir");
    auto _ = Profiler::TimedScope("MeasureCallback/RemoveBuildArtifact");
    for (const BuilderResult& build_result : builder_results) {
      if (ffi::Optional<ffi::String> path = build_result->artifact_path) {
        f_rm(path.value());
      }
    }
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RemoveBuildArtifactObj>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.RemoveBuildArtifact",
                                    RemoveBuildArtifactObj, MeasureCallbackObj);
};

MeasureCallback MeasureCallback::RemoveBuildArtifact() {
  ffi::ObjectPtr<RemoveBuildArtifactObj> n = ffi::make_object<RemoveBuildArtifactObj>();
  return MeasureCallback(ffi::UnsafeInit{}, n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  RemoveBuildArtifactObj::RegisterReflection();
  refl::GlobalDef().def("s_tir.meta_schedule.MeasureCallbackRemoveBuildArtifact",
                        MeasureCallback::RemoveBuildArtifact);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
