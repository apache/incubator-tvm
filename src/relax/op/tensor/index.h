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
 * \file index.h
 * \brief The functions to make Relax tensor indexing operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_INDEX_H_
#define TVM_RELAX_OP_TENSOR_INDEX_H_

#include <tvm/relax/op/index.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Take elements from a tensor along an axis.
 * \param x The source tensor.
 * \param indices The indices of the values to extract.
 * It is required to be a one-dimensional tensor which has integer dtype.
 * \param axis The axis over which to select values.
 * If it is `std::nullopt`, the input tensor is required to be one-dimensional.
 * \param mode The mode for handling out-of-bounds indices.
 * \return The taken result.
 */
Expr take(Expr x, Expr indices, ffi::Optional<int64_t> axis, ffi::String mode = "fast");

/*!
 * \brief Strided slice of a tensor.
 * \param x The source tensor to be sliced.
 * \param axes Axes along which slicing is applied.
 * \param begin The indices to begin with in the slicing, inclusive.
 * \param end The indices indicating end of the slice, exclusive.
 * \param strides Specifies the stride values, it can be negative in that case,
 * the input tensor will be reversed in that particular axis.
 * If it is `std::nullopt`, it by default is an list of ones of the same length as `axes`.
 * \param assume_inbound Whether to assume the indices are in bound.
 * \return The sliced result
 */
Expr strided_slice(Expr x, Expr axes, Expr begin, Expr end,
                   ffi::Optional<Expr> strides = std::nullopt, bool assume_inbound = false);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_INDEX_H_
