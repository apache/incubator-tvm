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

/*!
 * \file src/s_tir/tensor_intrin.cc
 * \brief Tensor intrinsic registry and construction.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/tensor_intrin.h>

namespace tvm {
namespace s_tir {
using tirx::Function;
using tirx::TensorTypeObj;

TVM_FFI_STATIC_INIT_BLOCK() { TensorIntrinObj::RegisterReflection(); }

class TensorIntrinManager {
 public:
  ffi::Map<ffi::String, TensorIntrin> reg;

  static TensorIntrinManager* Global() {
    static TensorIntrinManager* inst = new TensorIntrinManager();
    return inst;
  }
};

TensorIntrin::TensorIntrin(Function desc, Function impl) {
  // Check the number of func var is equal
  TVM_FFI_CHECK_EQ(desc->params.size(), impl->params.size(), ValueError)
      << "The number of parameters of the description and the implementation of the "
         "tensor intrinsic doesn't match.";
  auto is_handle = [](const Var& param) {
    return param->ty.as<PtrTypeObj>() != nullptr || param->ty.as<TensorTypeObj>() != nullptr;
  };
  for (size_t i = 0; i < desc->params.size(); i++) {
    TVM_FFI_CHECK(is_handle(desc->params[i]), ValueError)
        << "Parameters of the description of the "
           "tensor intrinsic should be handle only.";
    TVM_FFI_CHECK(is_handle(impl->params[i]), ValueError)
        << "Parameters of the implementation of "
           "the tensor intrinsic should be handle only.";
  }
  ffi::ObjectPtr<TensorIntrinObj> n =
      ffi::make_object<TensorIntrinObj>(std::move(desc), std::move(impl));
  data_ = std::move(n);
}

void TensorIntrin::Register(ffi::String name, TensorIntrin intrin, bool override) {
  TensorIntrinManager* manager = TensorIntrinManager::Global();
  if (!override) {
    TVM_FFI_CHECK_EQ(manager->reg.count(name), 0, ValueError)
        << "TensorIntrin '" << name << "' has already been registered";
  }
  manager->reg.Set(name, intrin);
}

ffi::Optional<TensorIntrin> TensorIntrin::Get(ffi::String name, bool allow_missing) {
  const TensorIntrinManager* manager = TensorIntrinManager::Global();
  auto it = manager->reg.find(name);
  if (it == manager->reg.end()) {
    if (allow_missing) {
      return std::nullopt;
    } else {
      TVM_FFI_THROW(ValueError) << "TensorIntrin '" << name << "' is not registered";
    }
  }
  return (*it).second;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.TensorIntrin",
           [](Function desc_func, Function intrin_func) {
             return TensorIntrin(desc_func, intrin_func);
           })
      .def("s_tir.TensorIntrinRegister", TensorIntrin::Register)
      .def("s_tir.TensorIntrinGet", TensorIntrin::Get);
}

}  // namespace s_tir
}  // namespace tvm
