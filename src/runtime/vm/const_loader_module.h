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
 * \file const_loader_module.h
 * \brief Defines an interface to use the ConstLoaderModule.
 */

#ifndef TVM_RUNTIME_VM_CONST_LOADER_MODULE_H_
#define TVM_RUNTIME_VM_CONST_LOADER_MODULE_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/tensor.h>

namespace tvm {
namespace runtime {

/*!
 * \brief Create a ConstLoader module object.
 *
 * \param const_var_tensor Maps const var name to Tensor containing data for the var.
 * \param const_vars_by_symbol Maps the name of a module init function to a list of names of
 * const vars whose data will be passed to that init function.
 *
 * \return The created ConstLoaderModule.
 *
 * Dispatches through the FFI registry ("ffi.Module.create.const_loader").
 * The creator is registered by the VM runtime.
 */
inline ffi::Module ConstLoaderModuleCreate(
    const ffi::Map<ffi::String, ffi::Tensor>& const_var_tensor,
    const ffi::Map<ffi::String, ffi::Array<ffi::String>>& const_vars_by_symbol) {
  static const auto fcreate = ffi::Function::GetGlobal("ffi.Module.create.const_loader");
  TVM_FFI_CHECK(fcreate.has_value(), RuntimeError)
      << "ffi.Module.create.const_loader is not registered by the VM runtime. "
      << "Ensure the runtime includes VM support.";
  return (*fcreate)(const_var_tensor, const_vars_by_symbol).cast<ffi::Module>();
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_CONST_LOADER_MODULE_H_
