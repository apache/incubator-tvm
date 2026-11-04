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
 * \file tvm/relax/op/create.h
 * \brief Attributes for tensor creation operators.
 */
#ifndef TVM_RELAX_OP_CREATE_H_
#define TVM_RELAX_OP_CREATE_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes used in full/full_like, ones/ones_like, and zeros/zeros_like operators */
struct InitAttrs : public AttrsObj {
  ffi::Optional<DLDataType> dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.InitAttrs", InitAttrs, AttrsObj);
};  // struct InitAttrs

/*! \brief Attributes used in tril and triu operator */
struct TriluAttrs : public AttrsObj {
  int k;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.TriluAttrs", TriluAttrs, AttrsObj);
};  // struct TriluAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_CREATE_H_
