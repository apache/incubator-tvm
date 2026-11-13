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

void PyMeasureCallbackObj::Apply(const TaskScheduler& task_scheduler,                     //
                                 int task_id,                                             //
                                 const ffi::Array<MeasureCandidate>& measure_candidates,  //
                                 const ffi::Array<BuilderResult>& builds,                 //
                                 const ffi::Array<RunnerResult>& results) {
  TVM_FFI_ICHECK(f_apply != nullptr) << "PyMeasureCallback's Apply method not implemented!";
  auto _ = Profiler::TimedScope("MeasureCallback/PyMeasureCallback");
  return f_apply(task_scheduler, task_id, measure_candidates, builds, results);
}

MeasureCallback MeasureCallback::PyMeasureCallback(PyMeasureCallbackObj::FApply f_apply) {
  ffi::ObjectPtr<PyMeasureCallbackObj> n = ffi::make_object<PyMeasureCallbackObj>();
  n->f_apply = std::move(f_apply);
  return MeasureCallback(ffi::UnsafeInit{}, n);
}

ffi::Array<MeasureCallback, void> MeasureCallback::Default() {
  return {
      MeasureCallback::AddToDatabase(),
      MeasureCallback::RemoveBuildArtifact(),
      MeasureCallback::UpdateCostModel(),
  };
}

// Pattern A (RM): auto-default repr from reflection.

TVM_FFI_STATIC_INIT_BLOCK() {
  MeasureCallbackObj::RegisterReflection();
  PyMeasureCallbackObj::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.meta_schedule.MeasureCallbackApply", &MeasureCallbackObj::Apply)
      .def("s_tir.meta_schedule.MeasureCallbackPyMeasureCallback",
           MeasureCallback::PyMeasureCallback)
      .def("s_tir.meta_schedule.MeasureCallbackDefault", MeasureCallback::Default);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
