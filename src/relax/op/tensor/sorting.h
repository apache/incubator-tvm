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
 * \file sorting.h
 * \brief The functions to make Relax tensor sorting operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_SORTING_H_
#define TVM_RELAX_OP_TENSOR_SORTING_H_

#include <tvm/relax/op/sorting.h>

#include <algorithm>
#include <utility>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Reverses the order of elements along given axis.
 * \param data The input tensor.
 * \param axis The axis to sort on.
 * \param descending Whether to sort in descending order.
 * \return The computed result.
 */
Expr sort(Expr data, int axis, bool descending);

/*!
 * \brief Performs sorting along the given axis and returns an array of indices.
 * \param data The input tensor.
 * \param axis The axis to sort on.
 * \param descending Whether to sort in descending order.
 * \param dtype The data type of the output indices.
 * \return The computed result.
 */
Expr argsort(Expr data, int axis, bool descending, ffi::Optional<DLDataType> dtype);

/*!
 * \brief Get the top k elements in an input tensor along the given axis.
 * \param data The input tensor.
 * \param k Number of top elements.
 * \param axis The axis to sort on.
 * \param ret_type The return type, can be set to one of [both, values, indices].
 * \param largest Whether to return largest or smallest elements.
 * \param dtype The data type of the indices output.
 * \return The computed result.
 */
Expr topk(Expr data, int k, int axis, ffi::String ret_type, bool largest,
          ffi::Optional<DLDataType> dtype);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_SORTING_H_
