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

void PyCostModelObj::Load(const ffi::String& path) {
  TVM_FFI_ICHECK(f_load != nullptr) << "PyCostModel's Load method not implemented!";
  f_load(path);
}

void PyCostModelObj::Save(const ffi::String& path) {
  TVM_FFI_ICHECK(f_save != nullptr) << "PyCostModel's Save method not implemented!";
  f_save(path);
}

void PyCostModelObj::Update(const TuneContext& context,
                            const ffi::Array<MeasureCandidate>& candidates,
                            const ffi::Array<RunnerResult>& results) {
  TVM_FFI_ICHECK(f_update != nullptr) << "PyCostModel's Update method not implemented!";
  f_update(context, candidates, results);
}

std::vector<double> PyCostModelObj::Predict(const TuneContext& context,
                                            const ffi::Array<MeasureCandidate>& candidates) {
  TVM_FFI_ICHECK(f_predict != nullptr) << "PyCostModel's Predict method not implemented!";
  std::vector<double> result(candidates.size(), 0.0);
  f_predict(context, candidates, result.data());
  return result;
}

CostModel CostModel::PyCostModel(PyCostModelObj::FLoad f_load,      //
                                 PyCostModelObj::FSave f_save,      //
                                 PyCostModelObj::FUpdate f_update,  //
                                 PyCostModelObj::FPredict f_predict) {
  ffi::ObjectPtr<PyCostModelObj> n = ffi::make_object<PyCostModelObj>();
  n->f_load = std::move(f_load);
  n->f_save = std::move(f_save);
  n->f_update = std::move(f_update);
  n->f_predict = std::move(f_predict);
  return CostModel(ffi::UnsafeInit{}, n);
}

// Pattern A (RM): auto-default repr from reflection.
// Ensure the type index is allocated for Python registration to work.

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Trigger type index allocation for types that Python @register_object needs to find.
  refl::ObjectDef<CostModelObj>();
  refl::ObjectDef<PyCostModelObj>();
  refl::GlobalDef()
      .def_method("s_tir.meta_schedule.CostModelLoad", &CostModelObj::Load)
      .def_method("s_tir.meta_schedule.CostModelSave", &CostModelObj::Save)
      .def_method("s_tir.meta_schedule.CostModelUpdate", &CostModelObj::Update)
      .def("s_tir.meta_schedule.CostModelPredict",
           [](CostModel model,                          //
              const TuneContext& context,               //
              ffi::Array<MeasureCandidate> candidates,  //
              void* p_addr) -> void {
             std::vector<double> result = model->Predict(context, candidates);
             std::copy(result.begin(), result.end(), static_cast<double*>(p_addr));
           })
      .def("s_tir.meta_schedule.CostModelPyCostModel", CostModel::PyCostModel);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
