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

/******** Constructors ********/

BuilderInput::BuilderInput(IRModule mod, Target target,
                           ffi::Optional<ffi::Map<ffi::String, ffi::Tensor>> params) {
  if (params) {
    for (const auto& entry : params.value()) {
      TVM_FFI_ICHECK(entry.second.defined());
    }
  }
  ffi::ObjectPtr<BuilderInputObj> n = ffi::make_object<BuilderInputObj>(target);
  n->mod = std::move(mod);
  n->params = std::move(params);
  data_ = std::move(n);
}

BuilderResult::BuilderResult(ffi::Optional<ffi::String> artifact_path,
                             ffi::Optional<ffi::String> error_msg) {
  ffi::ObjectPtr<BuilderResultObj> n = ffi::make_object<BuilderResultObj>();
  n->artifact_path = std::move(artifact_path);
  n->error_msg = std::move(error_msg);
  data_ = std::move(n);
}

Builder Builder::PyBuilder(BuilderObj::FBuild f_build) {
  ffi::ObjectPtr<PyBuilderObj> n = ffi::make_object<PyBuilderObj>();
  n->f_build = std::move(f_build);
  return Builder(std::move(n));
}

/******** FFI ********/

TVM_FFI_STATIC_INIT_BLOCK() {
  BuilderInputObj::RegisterReflection();
  BuilderResultObj::RegisterReflection();
  PyBuilderObj::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.meta_schedule.BuilderInput",
           [](IRModule mod, Target target, ffi::Optional<ffi::Map<ffi::String, ffi::Tensor>> params)
               -> BuilderInput { return BuilderInput(mod, target, params); })
      .def("s_tir.meta_schedule.BuilderResult",
           [](ffi::Optional<ffi::String> artifact_path, ffi::Optional<ffi::String> error_msg)
               -> BuilderResult { return BuilderResult(artifact_path, error_msg); })
      .def_method("s_tir.meta_schedule.BuilderBuild", &BuilderObj::Build)
      .def("s_tir.meta_schedule.BuilderPyBuilder", Builder::PyBuilder);
}

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm
