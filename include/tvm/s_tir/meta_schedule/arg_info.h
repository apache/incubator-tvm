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
#ifndef TVM_S_TIR_META_SCHEDULE_ARG_INFO_H_
#define TVM_S_TIR_META_SCHEDULE_ARG_INFO_H_

#include <tvm/base.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/function.h>

namespace tvm {
namespace s_tir {
namespace meta_schedule {

/*! \brief The argument information. */
class ArgInfoObj : public ffi::Object {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO("s_tir.meta_schedule.ArgInfo", ArgInfoObj, ffi::Object);

 public:
  /*! \brief Default destructor. */
  virtual ~ArgInfoObj() = default;
  /*! \brief Converts the ArgInfo to its corresponding JSON representation. */
  virtual ffi::ObjectRef AsJSON() const = 0;
};

/*!
 * \brief Managed reference to ArgInfoObj
 * \sa ArgInfoObj
 */
class ArgInfo : public ffi::ObjectRef {
 public:
  /*!
   * \brief Parse the argument information from a JSON object.
   * \param json_obj The json object to parse.
   * \return The argument information parsed.
   */
  TVM_DLL static ArgInfo FromJSON(const ffi::ObjectRef& json_obj);
  /*!
   * \brief Extract a list of the argument information from Function.
   * \param func The Function to get argument information from.
   * \return An array of the argument information derived.
   */
  TVM_DLL static ffi::Array<ArgInfo, void> FromFunction(const tirx::Function& func);
  /*!
   * \brief Extract a list of the argument information from the entry func of an IRModule
   * \param mod The IRModule to extract argument information from.
   * \param remove_preproc Whether to remove the preprocessing blocks.
   * \return An array of the argument information derived.
   */
  TVM_DLL static ffi::Array<ArgInfo, void> FromEntryFunc(const IRModule& mod, bool remove_preproc);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ArgInfo, ffi::ObjectRef, ArgInfoObj);

 protected:
  ArgInfo() = default;
};

/*! \brief The tensor argument information. */
class TensorInfoObj : public ArgInfoObj {
 public:
  /*! \brief The data type of the tensor. */
  DLDataType dtype;
  /*! \brief The shape of the tensor. */
  ffi::Shape shape;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorInfoObj>()
        .def_ro("dtype", &TensorInfoObj::dtype)
        .def_ro("shape", &TensorInfoObj::shape);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("s_tir.meta_schedule.TensorInfo", TensorInfoObj, ArgInfoObj);

 public:
  ffi::ObjectRef AsJSON() const;
};

/*!
 * \brief Managed reference to TensorInfoObj
 * \sa TensorInfoObj
 */
class TensorInfo : public ArgInfo {
 public:
  /*!
   * \brief Constructor of TensorInfo.
   * \param dtype The data type of the tensor argument.
   * \param shape The shape tuple of the tensor argument.
   */
  TVM_DLL explicit TensorInfo(DLDataType dtype, ffi::Shape shape);
  /*!
   * \brief Parse the argument information from a JSON object.
   * \param json_obj The json object to parse.
   * \return The argument information parsed.
   */
  TVM_DLL static TensorInfo FromJSON(const ffi::ObjectRef& json_obj);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorInfo, ArgInfo, TensorInfoObj);
};

}  // namespace meta_schedule
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_META_SCHEDULE_ARG_INFO_H_
