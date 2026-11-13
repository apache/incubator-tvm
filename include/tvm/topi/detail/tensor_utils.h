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
 * \file tensor_utils.h
 * \brief Utility functions for handling tensor
 */
#ifndef TVM_TOPI_DETAIL_TENSOR_UTILS_H_
#define TVM_TOPI_DETAIL_TENSOR_UTILS_H_

#include <tvm/te/operation.h>

#include <vector>
namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief Check whether input shape has dimension of size 0;
 *
 * \param x Input shape
 *
 * \return True if the input shape is empty.
 */
inline bool is_empty_shape(const ffi::Array<PrimExpr>& x) {
  bool is_empty = false;
  for (const auto& dim : x) {
    if (auto int_dim = dim.as<IntImmObj>()) {
      if (int_dim->value == 0) {
        is_empty = true;
        break;
      }
    }
  }
  return is_empty;
}

/*!
 * \brief Sample a point in a tensor using bilinear interpolation.
 *
 * \param input The input tensor.
 * \param indices The index of the target point, which can be fractional
 * \param max_y The maximum of y dimension
 * \param max_x The maximum of x dimension
 *
 * \return The interpolated value in the given index.
 */
inline PrimExpr bilinear_sample_nchw(const Tensor& input, const ffi::Array<PrimExpr>& indices,
                                     const PrimExpr max_y, const PrimExpr max_x) {
  auto batch_id = indices[0];
  auto channel_id = indices[1];
  auto in_y = indices[2];
  auto in_x = indices[3];

  auto y_low = tvm::prim::cast(tvm::prim::floor(in_y), PrimType::Int(32));
  auto y_high = y_low + 1;

  auto x_low = tvm::prim::cast(tvm::prim::floor(in_x), PrimType::Int(32));
  auto x_high = x_low + 1;

  auto wy_h = in_y - y_low;
  auto wx_h = in_x - x_low;
  auto wy_l = 1 - wy_h;
  auto wx_l = 1 - wx_h;

  PrimExpr val = 0;
  std::vector<std::vector<PrimExpr>> wx_xp{{wx_l, x_low}, {wx_h, x_high}};
  std::vector<std::vector<PrimExpr>> wy_yp{{wy_l, y_low}, {wy_h, y_high}};
  for (auto wx_xp_ele : wx_xp) {
    for (auto wy_yp_ele : wy_yp) {
      auto wx = wx_xp_ele[0];
      auto xp = wx_xp_ele[1];
      auto wy = wy_yp_ele[0];
      auto yp = wy_yp_ele[1];
      val += tvm::if_then_else(0 <= yp && yp <= max_y && 0 <= xp && xp <= max_x,
                               wx * wy * input(batch_id, channel_id, yp, xp), 0);
    }
  }
  return val;
}

/*!
 * \brief Sample a point in a tensor using bilinear interpolation.
 *
 * \param input The input tensor.
 * \param indices The index of the target point, which can be fractional
 * \param max_y The maximum of y dimension
 * \param max_x The maximum of x dimension
 *
 * \return The interpolated value in the given index.
 */
inline PrimExpr bilinear_sample_nhwc(const Tensor& input, const ffi::Array<PrimExpr>& indices,
                                     const PrimExpr max_y, const PrimExpr max_x) {
  auto batch_id = indices[0];
  auto channel_id = indices[3];
  auto in_y = indices[1];
  auto in_x = indices[2];

  auto y_low = tvm::prim::cast(tvm::prim::floor(in_y), PrimType::Int(32));
  auto y_high = y_low + 1;

  auto x_low = tvm::prim::cast(tvm::prim::floor(in_x), PrimType::Int(32));
  auto x_high = x_low + 1;

  auto wy_h = in_y - y_low;
  auto wx_h = in_x - x_low;
  auto wy_l = 1 - wy_h;
  auto wx_l = 1 - wx_h;

  PrimExpr val = 0;
  std::vector<std::vector<PrimExpr>> wx_xp{{wx_l, x_low}, {wx_h, x_high}};
  std::vector<std::vector<PrimExpr>> wy_yp{{wy_l, y_low}, {wy_h, y_high}};
  for (auto wx_xp_ele : wx_xp) {
    for (auto wy_yp_ele : wy_yp) {
      auto wx = wx_xp_ele[0];
      auto xp = wx_xp_ele[1];
      auto wy = wy_yp_ele[0];
      auto yp = wy_yp_ele[1];
      val += tvm::if_then_else(0 <= yp && yp <= max_y && 0 <= xp && xp <= max_x,
                               wx * wy * input(batch_id, yp, xp, channel_id), 0);
    }
  }
  return val;
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_TENSOR_UTILS_H_
