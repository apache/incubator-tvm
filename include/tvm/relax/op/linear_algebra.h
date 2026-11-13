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
 * \file tvm/relax/op/linear_algebra.h
 * \brief Attributes for linear algebra operators.
 */
#ifndef TVM_RELAX_OP_LINEAR_ALGEBRA_H_
#define TVM_RELAX_OP_LINEAR_ALGEBRA_H_

#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

/*! \brief Attributes for matmul operator */
struct MatmulAttrs : public AttrsObj {
  ffi::Optional<DLDataType> out_dtype;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.MatmulAttrs", MatmulAttrs, AttrsObj);
};  // struct MatmulAttrs

/*! \brief Attributes used in einsum operator */
struct EinsumAttrs : public AttrsObj {
  ffi::String subscripts;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.attrs.EinsumAttrs", EinsumAttrs, AttrsObj);
};  // struct EinsumAttrs

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_LINEAR_ALGEBRA_H_
