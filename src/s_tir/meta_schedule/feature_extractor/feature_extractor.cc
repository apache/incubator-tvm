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

ffi::Array<tvm::ffi::Tensor> PyFeatureExtractorObj::ExtractFrom(
    const TuneContext& context, const ffi::Array<MeasureCandidate>& candidates) {
  TVM_FFI_ICHECK(f_extract_from != nullptr)
      << "PyFeatureExtractor's ExtractFrom method not implemented!";
  return f_extract_from(context, candidates);
}

FeatureExtractor FeatureExtractor::PyFeatureExtractor(
    PyFeatureExtractorObj::FExtractFrom f_extract_from) {
  ffi::ObjectPtr<PyFeatureExtractorObj> n = ffi::make_object<PyFeatureExtractorObj>();
  n->f_extract_from = std::move(f_extract_from);
  return FeatureExtractor(ffi::UnsafeInit{}, n);
}

// Pattern A (RM): auto-default repr from reflection.

TVM_FFI_STATIC_INIT_BLOCK() {
  FeatureExtractorObj::RegisterReflection();
  PyFeatureExtractorObj::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_method("s_tir.meta_schedule.FeatureExtractorExtractFrom",
                  &FeatureExtractorObj::ExtractFrom)
      .def("s_tir.meta_schedule.FeatureExtractorPyFeatureExtractor",
           FeatureExtractor::PyFeatureExtractor);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
