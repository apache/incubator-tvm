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
 * \file pad_utils.h
 * \brief Padding helpers
 */
#ifndef TVM_TOPI_DETAIL_PAD_UTILS_H_
#define TVM_TOPI_DETAIL_PAD_UTILS_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/te/operation.h>

#include <vector>

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief Get padding size for each side given padding height and width
 *
 * \param pad_h The amount to pad each of the top and bottom sides
 * \param pad_w The amount to pad each of the left and right sides
 *
 * \return An array of 4 elements, representing padding sizes for
 * each individual side. The array is in the order { top, left, bottom, right }
 */
inline ffi::Array<PrimExpr> GetPadTuple(PrimExpr pad_h, PrimExpr pad_w) {
  pad_h *= 2;
  pad_w *= 2;

  auto pad_top = indexdiv(pad_h + 1, 2);
  auto pad_left = indexdiv(pad_w + 1, 2);

  return {pad_top, pad_left, pad_h - pad_top, pad_w - pad_left};
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_PAD_UTILS_H_
