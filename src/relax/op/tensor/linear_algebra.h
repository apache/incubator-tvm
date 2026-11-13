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
 * \file linear_algebra.h
 * \brief The functions to make Relax linear algebra operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_LINEAR_ALGEBRA_H_
#define TVM_RELAX_OP_TENSOR_LINEAR_ALGEBRA_H_

#include <tvm/relax/op/linear_algebra.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief General matrix multiplication of two tensors.
 * The semantics and output shape deduction rule is specified as
 * https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html.
 * \param x1 The first input tensor.
 * \param x2 The second input tensor.
 * \param out_dtype The data type of the matmul result.
 * When it is not specified, the output dtype will be the same as input dtype.
 * \return The computed result.
 */
Expr matmul(Expr x1, Expr x2, ffi::Optional<DLDataType> out_dtype);

/*!
 * \brief Einstein summation on the operands.
 * \param operands The input tensors.
 * \param subscripts The einsum expression string.
 * \return The computed result.
 */
Expr einsum(Expr operands, ffi::String subscripts);

/*!
 * \brief Compute the outer product of two input expressions.
 * \param x1 The first input expression.
 * \param x2 The second input expression.
 * \return The resulting expression representing the outer product.
 */
Expr outer(Expr x1, Expr x2);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_LINEAR_ALGEBRA_H_
