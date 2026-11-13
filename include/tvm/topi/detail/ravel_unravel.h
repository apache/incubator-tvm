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
 * \file ravel_unravel.h
 * \brief Index ravel and unraval operations
 */
#ifndef TVM_TOPI_DETAIL_RAVEL_UNRAVEL_H_
#define TVM_TOPI_DETAIL_RAVEL_UNRAVEL_H_

#include <tvm/te/operation.h>

#include <vector>

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief Flatten the indices to 1D
 *
 * \param indices The input coordinates
 * \param shape Shape of the tensor
 *
 * \return The index after flattening
 */
inline PrimExpr RavelIndex(ffi::Array<PrimExpr> indices, ffi::Array<PrimExpr> shape) {
  TVM_FFI_ICHECK_EQ(indices.size(), shape.size()) << "indices and shape must have equal size";
  if (indices.size() == 0U) {
    return 0;
  }
  PrimExpr idx = indices[0];
  for (size_t i = 1; i < indices.size(); ++i) {
    idx = idx * shape[i] + indices[i];
  }
  return idx;
}

/*!
 * \brief Convert flattened index to coordinate array
 *
 * \param idx The 1D index
 * \param shape Shape of the tensor
 *
 * \return The coordinate corresponding to the 1D index
 */
inline ffi::Array<PrimExpr> UnravelIndex(PrimExpr idx, ffi::Array<PrimExpr> shape) {
  std::vector<PrimExpr> indices;

  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    indices.push_back(indexmod(idx, shape[i]));
    idx = indexdiv(idx, shape[i]);
  }
  std::reverse(indices.begin(), indices.end());
  return indices;
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_RAVEL_UNRAVEL_H_
