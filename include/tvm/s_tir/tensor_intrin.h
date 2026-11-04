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
 * \file tvm/s_tir/tensor_intrin.h
 * \brief Tensor intrinsics for schedulable TIR.
 */
#ifndef TVM_S_TIR_TENSOR_INTRIN_H_
#define TVM_S_TIR_TENSOR_INTRIN_H_

#include <tvm/base.h>
#include <tvm/tirx/function.h>

namespace tvm {
namespace s_tir {

/*!
 * \brief Tensor intrinsics for tensorization
 */
class TensorIntrinObj : public ffi::Object {
 public:
  explicit TensorIntrinObj(ffi::UnsafeInit tag) : desc(tag), impl(tag) {}

  TensorIntrinObj(tirx::Function desc, tirx::Function impl)
      : desc(std::move(desc)), impl(std::move(impl)) {}

  /*! \brief The function to describe the computation. */
  tirx::Function desc;
  /*! \brief The function of the implementation for the execution. */
  tirx::Function impl;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorIntrinObj>()
        .def_ro("desc", &TensorIntrinObj::desc)
        .def_ro("impl", &TensorIntrinObj::impl);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.TensorIntrin", TensorIntrinObj, ffi::Object);
};

/*!
 * \brief Managed reference to TensorIntrinObj.
 */
class TensorIntrin : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param desc The function to describe the computation.
   * \param impl The function of the implementation for the execution.
   */
  TVM_DLL explicit TensorIntrin(tirx::Function desc, tirx::Function impl);

  /*!
   * \brief Create and register a TensorIntrin. After registration, the TensorIntrin can be looked
   * up with its name.
   * \param name The name of the TensorIntrin to register
   * \param intrin The TensorIntrin to register.
   * \param override Whether override existing intrinsic.
   * \throws This method throws an exception if the TensorIntrin with the specified name already
   *         exists.
   */
  TVM_DLL static void Register(ffi::String name, TensorIntrin intrin, bool override = false);

  /*!
   * \brief Look up TensorIntrin by name. Raises an exception if not found.
   * \param name The name of the TensorIntrin.
   * \param allow_missing Whether to allow missing tensor intrin. If false, an exception is raised
   *    if the tensor intrin is not found.
   * \return The TensorIntrin with the specified name.
   * \throws This method throws an exception if the TensorIntrin does not exist and allow_missing is
   * false.
   */
  TVM_DLL static ffi::Optional<TensorIntrin> Get(ffi::String name, bool allow_missing = false);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorIntrin, ffi::ObjectRef, TensorIntrinObj);
};

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_TENSOR_INTRIN_H_
