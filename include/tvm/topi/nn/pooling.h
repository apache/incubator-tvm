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
 * \brief Pooling op constructions
 * \file nn/pooling.h
 */
#ifndef TVM_TOPI_NN_POOLING_H_
#define TVM_TOPI_NN_POOLING_H_

#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/topi/detail/pad_utils.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <string>
#include <vector>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*! \brief Pooling type */
enum PoolType : int {
  kAvgPool,
  kMaxPool,
};

inline Tensor pool_grad_impl(const Tensor& out_grad, const Tensor& x,
                             const ffi::Array<PrimExpr>& kernel_size,
                             const ffi::Array<PrimExpr>& stride_size,
                             const ffi::Array<PrimExpr>& padding_size, PoolType pool_type,
                             bool ceil_mode, const size_t height_axis, const size_t width_axis,
                             bool count_include_pad) {
  using namespace tvm::prim;
  TVM_FFI_ICHECK(out_grad->shape.size() >= 2) << "Pooling grad output must >= 2-D (H, W)";
  TVM_FFI_ICHECK(x->shape.size() >= 2) << "Pooling input must >= 2-D (H, W)";
  TVM_FFI_ICHECK_EQ(kernel_size.size(), 2) << "Pooling kernel_size must have 2 elements";
  TVM_FFI_ICHECK_EQ(stride_size.size(), 2) << "Pooling stride_size must have 2 elements";
  TVM_FFI_ICHECK_EQ(padding_size.size(), 4) << "Pooling padding_size must have 4 elements";

  auto kernel_height = kernel_size[0];
  auto kernel_width = kernel_size[1];
  auto stride_height = stride_size[0];
  auto stride_width = stride_size[1];

  auto height = x->shape[height_axis];
  auto width = x->shape[width_axis];

  auto pad_top = padding_size[0];
  auto pad_left = padding_size[1];
  auto pad_bottom = padding_size[2];
  auto pad_right = padding_size[3];

  if (ceil_mode) {
    // Additional padding to ensure we do ceil instead of floor when
    // dividing by stride.
    pad_bottom += stride_height - 1;
    pad_right += stride_width - 1;
  }

  ffi::Array<PrimExpr> pad_before(std::vector<PrimExpr>(x->shape.size(), 0));
  pad_before.Set(height_axis, pad_top);
  pad_before.Set(width_axis, pad_left);

  ffi::Array<PrimExpr> pad_after(std::vector<PrimExpr>(x->shape.size(), 0));
  pad_after.Set(height_axis, pad_bottom);
  pad_after.Set(width_axis, pad_right);
  sym::Analyzer analyzer;
  auto out_height =
      analyzer->Simplify((height - kernel_height + pad_top + pad_bottom) / stride_height + 1);
  auto out_width =
      analyzer->Simplify((width - kernel_width + pad_left + pad_right) / stride_width + 1);

  auto dheight = tvm::te::reduce_axis(Range(0, kernel_height), "dh");
  auto dwidth = tvm::te::reduce_axis(Range(0, kernel_width), "dw");

  ffi::Array<PrimExpr> data_shape = x->shape;
  ffi::Array<PrimExpr> out_shape = data_shape;
  out_shape.Set(height_axis, out_height);
  out_shape.Set(width_axis, out_width);

  const auto* padding_h0 = pad_top.as<IntImmObj>();
  const auto* padding_w0 = pad_left.as<IntImmObj>();
  const auto* padding_h1 = pad_bottom.as<IntImmObj>();
  const auto* padding_w1 = pad_right.as<IntImmObj>();
  const bool do_pad =
      ((padding_h0 && padding_h0->value != 0) || (padding_w0 && padding_w0->value != 0)) ||
      ((padding_h1 && padding_h1->value != 0) || (padding_w1 && padding_w1->value != 0));

  if (pool_type == kMaxPool) {
    ffi::Array<PrimExpr> ravel_shape{data_shape.begin(), data_shape.end()};
    ravel_shape.Set(height_axis, ravel_shape[height_axis] + pad_top + pad_bottom);
    ravel_shape.Set(width_axis, ravel_shape[width_axis] + pad_left + pad_right);

    auto windowh =
        tvm::te::reduce_axis(Range(0, (kernel_height + stride_height - 1) / stride_height), "wh");
    auto windoww =
        tvm::te::reduce_axis(Range(0, (kernel_width + stride_width - 1) / stride_width), "ww");

    auto argmax = MakeArgmaxReducer();
    auto pad_x =
        do_pad ? pad(x, pad_before, pad_after, tvm::prim::min_value(PrimType(x->dtype)), "pad_temp")
               : x;

    auto mp_argmax = tvm::te::compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& inds) {
          ffi::Array<PrimExpr> window_inds =
              inds.Map([](const PrimVar& var) { return var.as_or_throw<PrimExpr>(); });
          window_inds.Set(height_axis, inds[height_axis] * stride_height + dheight);
          window_inds.Set(width_axis, inds[width_axis] * stride_width + dwidth);
          auto idx = detail::RavelIndex(window_inds, ravel_shape);
          return argmax({idx, pad_x(window_inds)}, {dheight, dwidth}, nullptr);
        },
        "maxpool_grad_argmax", kCommReduceIdx);

    auto mp_inds = mp_argmax[0];

    return tvm::te::compute(
        data_shape,
        [&](const ffi::Array<PrimVar>& inds) {
          ffi::Array<PrimExpr> pad_inds =
              inds.Map([](const PrimVar& var) { return var.as_or_throw<PrimExpr>(); });
          pad_inds.Set(height_axis, pad_inds[height_axis] + pad_top);
          pad_inds.Set(width_axis, pad_inds[width_axis] + pad_left);
          auto idx = detail::RavelIndex(pad_inds, ravel_shape);

          ffi::Array<PrimExpr> out_idx =
              inds.Map([](const PrimVar& var) { return var.as_or_throw<PrimExpr>(); });
          out_idx.Set(height_axis, (inds[height_axis] + pad_top) / stride_height - windowh);
          out_idx.Set(width_axis, (inds[width_axis] + pad_left) / stride_width - windoww);

          PrimExpr out_idx_lower_h = prim::Select(
              pad_inds[height_axis] < kernel_height, IntImm(pad_inds[height_axis].ty(), 0),
              (pad_inds[height_axis] - kernel_height) / stride_height + 1);
          PrimExpr out_idx_lower_w = prim::Select(
              pad_inds[width_axis] < kernel_width, IntImm(pad_inds[width_axis].ty(), 0),
              (pad_inds[width_axis] - kernel_width) / stride_width + 1);

          return tvm::prim::sum(
              tvm::if_then_else(prim::And(prim::And(out_idx[height_axis] >= out_idx_lower_h,
                                                    out_idx[width_axis] >= out_idx_lower_w),
                                          mp_inds(out_idx) == idx),
                                out_grad(out_idx), MakeConst(PrimType(x->dtype), 0)),
              {windowh, windoww});
        },
        "T_pool_grad", "pool_grad_max");
  } else if (pool_type == kAvgPool) {
    auto windowh =
        tvm::te::reduce_axis(Range(0, (kernel_height + stride_height - 1) / stride_height), "wh");
    auto windoww =
        tvm::te::reduce_axis(Range(0, (kernel_width + stride_width - 1) / stride_width), "ww");
    return tvm::te::compute(
        data_shape,
        [&](const ffi::Array<PrimVar>& inds) {
          PrimExpr pad_h_idx = inds[height_axis] + pad_top;
          PrimExpr pad_w_idx = inds[width_axis] + pad_left;

          // output indices whose pooling windows cover current input element (can be out-of-bound)
          ffi::Array<PrimExpr> out_idx =
              inds.Map([](const PrimVar& var) { return var.as_or_throw<PrimExpr>(); });
          out_idx.Set(height_axis, (pad_h_idx / stride_height - windowh));
          out_idx.Set(width_axis, (pad_w_idx / stride_width - windoww));

          PrimExpr out_idx_lower_h =
              prim::Select(pad_h_idx < kernel_height, IntImm(pad_h_idx.ty(), 0),
                           (pad_h_idx - kernel_height) / stride_height + 1);
          PrimExpr out_idx_lower_w =
              prim::Select(pad_w_idx < kernel_width, IntImm(pad_w_idx.ty(), 0),
                           (pad_w_idx - kernel_width) / stride_width + 1);

          PrimExpr divide_factor = kernel_height * kernel_width;  // number of pooled elements
          if (!count_include_pad) {
            PrimExpr h_start = out_idx[height_axis] * stride_height - pad_top;
            PrimExpr w_start = out_idx[width_axis] * stride_width - pad_left;

            PrimExpr h_end = min(h_start + kernel_height, height);
            PrimExpr w_end = min(w_start + kernel_width, width);
            h_start = max(h_start, IntImm(h_start.ty(), 0));
            w_start = max(w_start, IntImm(w_start.ty(), 0));
            divide_factor = max((h_end - h_start) * (w_end - w_start), IntImm(h_end.ty(), 1));
          }
          return tvm::prim::sum(
              tvm::if_then_else(prim::And(prim::And(out_idx[height_axis] >= out_idx_lower_h,
                                                    out_idx[height_axis] < out_height),
                                          prim::And(out_idx[width_axis] >= out_idx_lower_w,
                                                    out_idx[width_axis] < out_width)),
                                out_grad(out_idx) / divide_factor,
                                MakeConst(PrimType(out_grad->dtype), 0)),
              {windowh, windoww});
        },
        "T_pool_grad", "pool_grad_avg");
  } else {
    TVM_FFI_THROW(ValueError) << "Unrecognized pool_type: " << pool_type;
  }
}

/*!
 * \brief Find index of Depth, Height or Width dimension in a layout string.
 *
 * \param layout The layout string
 * \param depth_axis set as the index of depth ('D') if not nullptr.
 * \param height_axis set as the index of height ('H') if not nullptr.
 * \param width_axis set as the index of width ('W') if not nullptr.
 *
 * \return true if the layout is valid (i.e., no tiling on D, H or W dimensions, no duplicates and
 * if the requested dimensions are found), otherwise false.
 */
inline bool find_depth_height_width(const std::string& layout, int* depth_axis, int* height_axis,
                                    int* width_axis) {
  if (depth_axis) *depth_axis = -1;
  if (height_axis) *height_axis = -1;
  if (width_axis) *width_axis = -1;
  int curr_idx = 0;
  for (size_t i = 0; i < layout.size(); ++i) {
    if ((layout[i] >= 'A' && layout[i] <= 'Z') || (layout[i] >= 'a' && layout[i] <= 'z')) {
      if (layout[i] == 'D' && depth_axis) {
        if (*depth_axis != -1) return false;
        *depth_axis = curr_idx;
      } else if (layout[i] == 'H' && height_axis) {
        if (*height_axis != -1) return false;
        *height_axis = curr_idx;
      } else if (layout[i] == 'W' && width_axis) {
        if (*width_axis != -1) return false;
        *width_axis = curr_idx;
      } else if (layout[i] == 'd' || layout[i] == 'h' || layout[i] == 'w') {
        // do not support split on height, width or depth, e.g., NCHW16w
        return false;
      }
      ++curr_idx;
    }
  }
  if ((depth_axis && *depth_axis == -1) || (height_axis && *height_axis == -1) ||
      (width_axis && *width_axis == -1))
    return false;
  return true;
}

inline bool find_height_width(const std::string& layout, int* height_axis, int* width_axis) {
  return find_depth_height_width(layout, /*depth_axis=*/nullptr, height_axis, width_axis);
}

inline bool find_width(const std::string& layout, int* width_axis) {
  return find_depth_height_width(layout, /*depth_axis=*/nullptr, /*height_axis=*/nullptr,
                                 width_axis);
}

/*!
 * \brief Calculate gradient of pooling on height and width dimension of data.
 *        It decides the height and width dimension according to the layout string,
 *        in which 'W' and 'H' means width and height respectively.
 *        Width and height dimension cannot be split.
 *        For example, NCHW, NCHW16c, etc. are valid for pool,
 *        while NCHW16w, NCHW16h are not.
 *        See \a layout for more information of the layout string convention.
 * \param out_grad The output gradient tensor.
 * \param x The input tensor.
 * \param kernel_size Vector of two ints: {kernel_height, kernel_width}
 * \param stride_size Vector of two ints: {stride_height, stride_width}
 * \param padding_size Vector of two ints: {padding_height, padding_width}
 * \param pool_type The type of pooling operator
 * \param ceil_mode Whether to use ceil when calculating the output size
 * \param layout The input layout. Pooling supports any layout as long as 'H' and 'W' appear.
 *        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
 *        where upper case indicates a dimension and
 *        the corresponding lower case (with factor size) indicates the split dimension.
 *        For example, NCHW16c can describe a 5-D tensor of
 *        [batch_size, channel, height, width, channel_block].
 *        (in which factor size `16` will not be used in pooling but for other operators,
 *        it can be used to decide the output shape).
 *        Since pooling does not care about the factor size of dimensions
 *        other than `H` and `W`, one can pass `NCHWc` as well.
 * \param  count_include_pad Whether include padding in the calculation when pool_type is 'avg'
 *
 *
 * \return The output tensor in the same layout
 */
inline Tensor pool_grad(const Tensor& out_grad, const Tensor& x,
                        const ffi::Array<PrimExpr>& kernel_size,
                        const ffi::Array<PrimExpr>& stride_size,
                        const ffi::Array<PrimExpr>& padding_size, PoolType pool_type,
                        bool ceil_mode, const std::string& layout = "NCHW",
                        bool count_include_pad = true) {
  int height_axis = -1, width_axis = -1;
  TVM_FFI_ICHECK(find_height_width(layout, &height_axis, &width_axis))
      << "Unsupported layout " << layout;
  return pool_grad_impl(out_grad, x, kernel_size, stride_size, padding_size, pool_type, ceil_mode,
                        height_axis, width_axis, count_include_pad);
}

inline PrimExpr start_index(const PrimVar& out_index, const PrimExpr& odim, const PrimExpr& idim) {
  return indexdiv(out_index * idim, odim);
}

inline PrimExpr end_index(const PrimVar& out_index, const PrimExpr& odim, const PrimExpr& idim) {
  PrimExpr tmp = indexdiv((out_index + 1) * idim, odim);
  return tvm::prim::Select(indexmod((out_index + 1) * idim, odim) == 0, tmp, tmp + 1);
}

/*!
 * \brief Perform adaptive pooling on N dimensional data
 *
 * \param x The input tensor
 * \param output_size int vector of size in each dimension
 * \param pool_type The type of pooling operator
 * \param axes indices of each dimension
 *
 * \return The output tensor in same layout order
 */
inline Tensor adaptive_pool_impl(const Tensor& x, const ffi::Array<PrimExpr>& output_size,
                                 PoolType pool_type, const std::vector<int>& axes) {
  const auto n_dim = output_size.size();
  TVM_FFI_ICHECK_EQ(axes.size(), n_dim) << "The number of axes not equal to the in/out dimension";

  ffi::Array<PrimExpr> data_shape = x->shape;
  ffi::Array<PrimExpr> out_shape = data_shape;
  ffi::Array<PrimExpr> in_size, out_size;
  for (size_t i = 0; i < n_dim; ++i) {
    in_size.push_back(data_shape[axes[i]]);
    out_size.push_back(output_size[i]);
    out_shape.Set(axes[i], out_size[i]);
  }

  auto get_iter_vars = [=](const ffi::Array<PrimVar>& output, bool reduce_indices) {
    ffi::Array<PrimExpr> indices;
    for (size_t i = 0; i < output.size(); ++i) indices.push_back(output[i]);
    ffi::Array<s_tir::IterVar> reduce_axes;
    for (size_t i = 0; i < n_dim; ++i) {
      auto i_start = start_index(output[axes[i]], out_size[i], in_size[i]);
      auto i_end = end_index(output[axes[i]], out_size[i], in_size[i]);
      auto rv_name = "rv" + std::to_string(i);
      auto rv_axis = tvm::te::reduce_axis(Range(0, i_end - i_start), rv_name);
      reduce_axes.push_back(rv_axis);
      if (reduce_indices) {
        indices.Set(axes[i], i_start + rv_axis);
      }
    }
    return std::make_tuple(indices, reduce_axes);
  };

  ffi::Map<ffi::String, ffi::Any> attrs;
  if (pool_type == kMaxPool) {
    attrs.Set(tvm::s_tir::attr::kScheduleRule.name,
              tvm::ffi::String("meta_schedule.adaptive_pool_max"));
    return tvm::te::compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& output) {
          ffi::Array<PrimExpr> indices;
          ffi::Array<s_tir::IterVar> reduce_axes;
          std::tie(indices, reduce_axes) = get_iter_vars(output, true);
          return tvm::max(x(indices), reduce_axes);  // NOLINT(*)
        },
        "adaptive_pool_max", "adaptive_pool_max", attrs);
  } else if (pool_type == kAvgPool) {
    attrs.Set(tvm::s_tir::attr::kScheduleRule.name,
              tvm::ffi::String("meta_schedule.adaptive_pool_avg"));
    auto pool_sum = tvm::te::compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& output) {
          ffi::Array<PrimExpr> indices;
          ffi::Array<s_tir::IterVar> reduce_axes;
          std::tie(indices, reduce_axes) = get_iter_vars(output, true);
          return tvm::prim::sum(x(indices), reduce_axes);
        },
        "adaptive_pool_sum", "adaptive_pool_sum");

    return tvm::te::compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& output) {
          ffi::Array<PrimExpr> indices;
          ffi::Array<s_tir::IterVar> reduce_axes;
          std::tie(indices, reduce_axes) = get_iter_vars(output, false);

          PrimExpr divide_factor = tvm::prim::cast(1, PrimType(x->dtype));
          for (size_t i = 0; i < n_dim; ++i) {
            divide_factor *=
                tvm::prim::cast(reduce_axes[i]->dom.value()->extent, PrimType::Int(32));
          }

          return div(pool_sum(indices), divide_factor);
        },
        "adaptive_pool_avg", kElementWise, attrs);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

/*!
 * \brief Adaptively perform pooling on height and width dimension of data.
 *        The pooling kernel and stride sizes are automatically chosen for desired output sizes.
 *        It decides the height and width dimension according to the layout string,
 *        in which 'W' and 'H' means width and height respectively.
 *        Width and height dimension cannot be split.
 *        For example, NCHW, NCHW16c, etc. are valid for pool,
 *        while NCHW16w, NCHW16h are not.
 *        See \a layout for more information of the layout string convention.
 *
 * \param x The input tensor
 * \param output_size Vector of two ints: {output_height, output_width}
 * \param pool_type The type of pooling operator
 * \param layout The input layout. Pooling supports any layout as long as 'H' and 'W' appear.
 *        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
 *        where upper case indicates a dimension and
 *        the corresponding lower case (with factor size) indicates the split dimension.
 *        For example, NCHW16c can describe a 5-D tensor of
 *        [batch_size, channel, height, width, channel_block].
 *        (in which factor size `16` will not be used in pooling but for other operators,
 *        it can be used to decide the output shape).
 *        Since pooling does not care about the factor size of dimensions
 *        other than `H` and `W`, one can pass `NCHWc` as well.
 *
 * \return The output tensor in same layout order
 */
inline Tensor adaptive_pool(const Tensor& x, const ffi::Array<PrimExpr>& output_size,
                            PoolType pool_type, const std::string& layout = "NCHW") {
  int height_axis = -1, width_axis = -1;
  TVM_FFI_ICHECK(find_height_width(layout, &height_axis, &width_axis))
      << "Unsupported layout " << layout;
  return adaptive_pool_impl(x, output_size, pool_type, {height_axis, width_axis});
}

/*!
 * \brief Adaptively perform pooling on three dimensional data.
 *        See the two dimensional version above for details.
 * \param x The input tensor
 * \param output_size Vector of three ints: {output_depth, output_height, output_width}
 * \param pool_type The type of pooling operator
 * \param layout The input layout. The default is "NCDHW".
 */
inline Tensor adaptive_pool3d(const Tensor& x, const ffi::Array<PrimExpr>& output_size,
                              PoolType pool_type, const std::string& layout = "NCDHW") {
  int depth_axis = -1, height_axis = -1, width_axis = -1;
  TVM_FFI_ICHECK(find_depth_height_width(layout, &depth_axis, &height_axis, &width_axis))
      << "Unsupported layout " << layout;
  return adaptive_pool_impl(x, output_size, pool_type, {depth_axis, height_axis, width_axis});
}

/*!
 * \brief Adaptively perform pooling on one dimensional data.
 *        See the two dimensional version above for details.
 * \param x The input tensor
 * \param output_size Vector of one int: {output_width}
 * \param pool_type The type of pooling operator
 * \param layout The input layout. The default is "NCW".
 */
inline Tensor adaptive_pool1d(const Tensor& x, const ffi::Array<PrimExpr>& output_size,
                              PoolType pool_type, const std::string& layout = "NCW") {
  int width_axis = -1;
  TVM_FFI_ICHECK(find_width(layout, &width_axis)) << "Unsupported layout " << layout;
  return adaptive_pool_impl(x, output_size, pool_type, {width_axis});
}

/*!
 * \brief Perform global pooling on height and width dimension of data.
 *        It decides the height and width dimension according to the layout string,
 *        in which 'W' and 'H' means width and height respectively.
 *        Width and height dimension cannot be split.
 *        For example, NCHW, NCHW16c, ... are valid for global_pool,
 *        while NCHW16w, NCHW16h are not.
 *        See \a layout for more information of the layout string convention.
 *
 * \param x The input tensor represent as layout
 * \param pool_type The type of pooling operator
 * \param layout The input layout. global-pooling supports any layout as long as 'H' and 'W' appear.
 *        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
 *        where upper case indicates a dimension and
 *        the corresponding lower case (with factor size) indicates the sub-dimension.
 *        For example, `NCHW16c` can describe a 5-D tensor of
 *        [batch_size, channel, height, width, channel_block].
 *        (in which factor size `16` will not be used in pooling but for other operators,
 *        it can be used to decide the output shape).
 *        Since pooling does not care about the factor size of
 *        dimensions other than `H` and `W`, one can pass `NCHWc` as well.
 *
 * \return The output tensor in same layout with height and width dimension size of 1.
 *         e.g., for NCHW, the output shape will be [batch, channel, 1, 1]
 */
inline Tensor global_pool(const Tensor& x, PoolType pool_type, const std::string& layout = "NCHW") {
  return adaptive_pool(x, ffi::Array<PrimExpr>{1, 1}, pool_type, layout);
}

/*!
 * \brief Perform pooling on N-dimension of data.
 *
 * \param x The input tensor
 * \param kernel_size Vector of N ints
 * \param stride_size Vector of N ints
 * \param dilation_size Vector of N ints
 * \param padding_size Vector of N*2 ints [head_pad_d1, head_pad_d2, ...,
 *        head_pad_dN, tail_pad_d1, tail_pad_d2, ..., tail_pad_dN]
 * \param pool_type The type of pooling operator
 * \param ceil_mode Whether to use ceil when calculating the output size
 * \param axis Vector of indices for the N dimensions
 * \param count_include_pad Whether include padding in the calculation
 *
 * \return The output tensor in same layout order
 */
inline Tensor pool_impl_nd(const Tensor& x, const ffi::Array<PrimExpr>& kernel_size,
                           const ffi::Array<PrimExpr>& stride_size,
                           const ffi::Array<PrimExpr>& dilation_size,
                           const ffi::Array<PrimExpr>& padding_size, PoolType pool_type,
                           bool ceil_mode, const std::vector<int>& axis, bool count_include_pad) {
  using namespace tvm::prim;
  int k_size = kernel_size.size();
  int x_size = x->shape.size();
  TVM_FFI_ICHECK_EQ(stride_size.size(), k_size)
      << "Pooling stride_size must have same elements as kernel";
  TVM_FFI_ICHECK_EQ(padding_size.size(), k_size * 2)
      << "Pooling padding_size must has double elements of"
         " kernel";
  TVM_FFI_ICHECK_EQ(axis.size(), k_size) << "axis must have same elements as kernel";

  ffi::Array<IterVar> daxis;
  std::vector<PrimExpr> kernel(kernel_size.begin(), kernel_size.end());
  std::vector<PrimExpr> stride(stride_size.begin(), stride_size.end());
  std::vector<PrimExpr> dilation(dilation_size.begin(), dilation_size.end());
  std::vector<PrimExpr> pad_head(padding_size.begin(), padding_size.begin() + k_size);
  std::vector<PrimExpr> pad_tail(padding_size.begin() + k_size, padding_size.end());
  std::vector<PrimExpr> offset(k_size, 0);
  ffi::Array<PrimExpr> pad_before(std::vector<PrimExpr>(x_size, 0));
  ffi::Array<PrimExpr> pad_after(std::vector<PrimExpr>(x_size, 0));
  ffi::Array<PrimExpr> data_shape = x->shape;
  ffi::Array<PrimExpr> out_shape = data_shape;

  bool do_pad = false;
  for (int i = 0; i < k_size; i++) {
    int ii = axis[i];

    if (ceil_mode) {
      // The offset[i] is an additional padding to ensure we do ceil instead of floor when
      // dividing by stride.
      // In the case of ceil_mode=True and count_include_pad=True,
      // in order to obtain the correct boundary,
      // we also need to use the offset[i] to eliminate this extra padding.
      offset[i] = stride[i] - 1;
      pad_tail[i] += offset[i];
    }

    const auto* padding0 = pad_head[i].as<IntImmObj>();
    const auto* padding1 = pad_tail[i].as<IntImmObj>();
    do_pad = do_pad || (padding0 && padding0->value != 0) || (padding1 && padding1->value != 0);

    daxis.push_back(tvm::te::reduce_axis(Range(0, kernel[i]), "rv" + std::to_string(i)));

    pad_before.Set(ii, pad_head[i]);
    pad_after.Set(ii, pad_tail[i]);

    sym::Analyzer analyzer;

    PrimExpr numerator =
        data_shape[ii] - (kernel[i] - 1) * dilation[i] - 1 + pad_head[i] + pad_tail[i];
    auto raw_out = indexdiv(numerator, stride[i]) + 1;
    if (ceil_mode) {
      // In the case of ceil_mode=True, we need to check if the last pooling window is valid.
      // If not, we skip the last window as it would start in the bottom padded region,
      // we need to minus 1 to get the correct output shape.
      auto invalid_last = (raw_out - 1) * stride[i] >= data_shape[ii] + pad_head[i];
      auto out_dim = analyzer->Simplify(if_then_else(invalid_last, raw_out - 1, raw_out));
      out_shape.Set(ii, out_dim);
    } else {
      auto out_dim = analyzer->Simplify(raw_out);
      out_shape.Set(ii, out_dim);
    }
  }

  ffi::Map<ffi::String, ffi::Any> attrs;
  if (pool_type == kMaxPool) {
    auto temp =
        do_pad ? pad(x, pad_before, pad_after, tvm::prim::min_value(PrimType(x->dtype)), "pad_temp")
               : x;
    attrs.Set(tvm::s_tir::attr::kScheduleRule.name, tvm::ffi::String("meta_schedule.pool_max"));
    return tvm::te::compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& output) {
          ffi::Array<PrimExpr> indices;
          for (const PrimVar& var : output) indices.push_back(var);

          for (int i = 0; i < k_size; i++) {
            int ii = axis[i];
            indices.Set(ii, output[ii] * stride[i] + daxis[i] * dilation[i]);
          }
          return tvm::max(temp(indices), daxis);
        },
        "pool_max", "pool_max", attrs);
  } else if (pool_type == kAvgPool) {
    attrs.Set(tvm::s_tir::attr::kScheduleRule.name, tvm::ffi::String("meta_schedule.pool_avg"));
    // Pad the inputs
    auto temp = do_pad ? pad(x, pad_before, pad_after, 0, "pad_temp") : x;

    // TVM compute for summing the pooling window.
    auto pool_sum = tvm::te::compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& output) {
          ffi::Array<PrimExpr> indices;
          for (const PrimVar& var : output) indices.push_back(var);

          for (int i = 0; i < k_size; i++) {
            int ii = axis[i];
            indices.Set(ii, output[ii] * stride[i] + daxis[i] * dilation[i]);
          }
          return tvm::prim::sum(temp(indices), daxis);
        },
        "pool_sum", "pool_sum");

    // TVM compute for dividing the reduced window sum by kernel size.
    return tvm::te::compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& output) {
          ffi::Array<PrimExpr> indices;
          for (const PrimVar& var : output) indices.push_back(var);
          if (count_include_pad) {
            auto num_el = IntImm::Int32(1);
            for (int i = 0; i < k_size; i++) {
              int ii = axis[i];
              PrimExpr start = output[ii] * stride[i] - pad_head[i];
              // When computing the output shape in ceil_mode,
              // we have added the extra padding of offset[i],
              // so now in order to calculate the correct boundary ,
              // we need to substract the offset[i].
              PrimExpr end = start + (kernel[i] - 1) * dilation[i];
              end = min(end, data_shape[ii] + pad_tail[i] - 1 - offset[i]);
              num_el *= (end - start) / dilation[i] + 1;
            }
            return div(pool_sum(indices), num_el);
          } else {
            auto num_el = IntImm::Int32(1);
            for (int i = 0; i < k_size; i++) {
              int ii = axis[i];

              // Let start and end contain the first and last index of our Tensor
              // along the relevant dimension we use in our calculation.
              // Assume indices -1, -2 represent the padding before (tail) and
              // len(arr), len(arr) + 1 represent the padding after (head).
              PrimExpr start = output[ii] * stride[i] - pad_head[i];
              PrimExpr end = start + (kernel[i] - 1) * dilation[i];

              // if start < 0, e.g. we start on a tail padded number this will be a positive
              // number that represents the number of steps along the dilated kernel to reach a
              // non-padded value. Otherwise this should be 0.
              PrimExpr jumps_to_non_pad = (dilation[i] - 1 - start) / dilation[i];
              jumps_to_non_pad = max(jumps_to_non_pad, IntImm(jumps_to_non_pad.ty(), 0));

              end = min(end, data_shape[ii] - 1);
              num_el *= (end - (start + dilation[i] * jumps_to_non_pad)) / dilation[i] + 1;
            }

            PrimExpr divide_factor = max(num_el, IntImm::Int32(1));
            return div(pool_sum(indices), divide_factor);
          }
        },
        "pool_avg", kElementWise, attrs);
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
    return x;
  }
}

/*!
 * \brief Perform pooling on the width dimension of data.
 *        Width axis is determined by the layout string
 *        in which 'W' means width.
 *        Width dimension cannot be split.
 *        For example, NCW, NCW16c, etc. are valid for pool,
 *        while NCW16w is not.
 *        See \a layout for more information of the layout string convention.
 * \param x The input tensor.
 * \param kernel_size Vector of one int: {kernel_width}
 * \param stride_size Vector of one int: {stride_width}
 * \param dilation_size Vector of one int: {dilation_width}
 * \param padding_size Vector of two ints: {head_pad_width, tail_pad_width}
 * \param pool_type The type of pooling operator
 * \param ceil_mode Whether to use ceil when calculating the output size
 * \param layout The input layout. Pooling supports any layout as long as 'W' appears.
 *        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
 *        where upper case indicates a dimension and
 *        the corresponding lower case (with factor size) indicates the split dimension.
 *        For example, NCW16c can describe a 4-D tensor of
 *        [batch_size, channel, width, channel_block].
 *        (in which factor size `16` will not be used in pooling but for other operators,
 *        it can be used to decide the output shape).
 *        Since pooling does not care about the factor size of dimensions
 *        other than `W`, one can pass `NCWc` as well.
 * \param  count_include_pad Whether include padding in the calculation when pool_type is 'avg'
 *
 *
 * \return The output tensor in the same layout
 */
inline Tensor pool1d(const Tensor& x, const ffi::Array<PrimExpr>& kernel_size,
                     const ffi::Array<PrimExpr>& stride_size,
                     const ffi::Array<PrimExpr>& dilation_size,
                     const ffi::Array<PrimExpr>& padding_size, PoolType pool_type, bool ceil_mode,
                     const std::string& layout = "NCW", bool count_include_pad = true) {
  int width_axis = -1;
  TVM_FFI_ICHECK(find_width(layout, &width_axis)) << "Unsupported layout " << layout;
  std::vector<int> axis = {width_axis};
  return pool_impl_nd(x, kernel_size, stride_size, dilation_size, padding_size, pool_type,
                      ceil_mode, axis, count_include_pad);
}

/*!
 * \brief Perform pooling on height and width dimension of data.
 *        It decides the height and width dimension according to the layout string,
 *        in which 'W' and 'H' means width and height respectively.
 *        Width and height dimension cannot be split.
 *        For example, NCHW, NCHW16c, etc. are valid for pool,
 *        while NCHW16w, NCHW16h are not.
 *        See \a layout for more information of the layout string convention.
 * \param x The input tensor.
 * \param kernel_size Vector of two ints: {kernel_height, kernel_width}
 * \param stride_size Vector of two ints: {stride_height, stride_width}
 * \param dilation_size Vector of two ints: {dilation_height, dilation_width}
 * \param padding_size Vector of two ints: {padding_height, padding_width}
 * \param pool_type The type of pooling operator
 * \param ceil_mode Whether to use ceil when calculating the output size
 * \param layout The input layout. Pooling supports any layout as long as 'H' and 'W' appear.
 *        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
 *        where upper case indicates a dimension and
 *        the corresponding lower case (with factor size) indicates the split dimension.
 *        For example, NCHW16c can describe a 5-D tensor of
 *        [batch_size, channel, height, width, channel_block].
 *        (in which factor size `16` will not be used in pooling but for other operators,
 *        it can be used to decide the output shape).
 *        Since pooling does not care about the factor size of dimensions
 *        other than `H` and `W`, one can pass `NCHWc` as well.
 * \param  count_include_pad Whether include padding in the calculation when pool_type is 'avg'
 *
 *
 * \return The output tensor in the same layout
 */
inline Tensor pool2d(const Tensor& x, const ffi::Array<PrimExpr>& kernel_size,
                     const ffi::Array<PrimExpr>& stride_size,
                     const ffi::Array<PrimExpr>& dilation_size,
                     const ffi::Array<PrimExpr>& padding_size, PoolType pool_type, bool ceil_mode,
                     const std::string& layout = "NCHW", bool count_include_pad = true) {
  int height_axis = -1, width_axis = -1;
  TVM_FFI_ICHECK(find_height_width(layout, &height_axis, &width_axis))
      << "Unsupported layout " << layout;
  std::vector<int> axis = {height_axis, width_axis};
  return pool_impl_nd(x, kernel_size, stride_size, dilation_size, padding_size, pool_type,
                      ceil_mode, axis, count_include_pad);
}

/*!
 * \brief Perform pooling on depth, height and width dimension of data.
 *        It decides the depth, height and width dimension according to the layout string,
 *        in which 'D', 'W' and 'H' means depth, width and height respectively.
 *        Depth, Width and height dimension cannot be split.
 *        For example, NCDHW, NCDHW16c, etc. are valid for pool,
 *        while NCDHW16d, NCDHW16w or NCDHW16h are not.
 *        See \a layout for more information of the layout string convention.
 * \param x The input tensor.
 * \param kernel_size Vector of three ints: {kernel_depth, kernel_height, kernel_width}
 * \param stride_size Vector of three ints: {stride_depth, stride_height, stride_width}
 * \param dilation_size Vector of three ints: {dilation_depth, dilation_height, dilation_width}
 * \param padding_size Vector of six ints: {head_pad_depth, head_pad_height, head_pad_width,
 *        tail_pad_depth, tail_pad_height, tail_pad_width}
 * \param pool_type The type of pooling operator
 * \param ceil_mode Whether to use ceil when calculating the output size
 * \param layout The input layout. Pooling supports any layout as long as 'D', 'H' and 'W' appear.
 *        The layout is supposed to be composed of upper cases, lower cases and (optional) numbers,
 *        where upper case indicates a dimension and
 *        the corresponding lower case (with factor size) indicates the split dimension.
 *        For example, NCDHW16c can describe a 6-D tensor of
 *        [batch_size, channel, depth, height, width, channel_block].
 *        (in which factor size `16` will not be used in pooling but for other operators,
 *        it can be used to decide the output shape).
 *        Since pooling does not care about the factor size of dimensions
 *        other than `D`, `H` and `W`, one can pass `NCDHWc` as well.
 * \param  count_include_pad Whether include padding in the calculation when pool_type is 'avg'
 *
 *
 * \return The output tensor in the same layout
 */
inline Tensor pool3d(const Tensor& x, const ffi::Array<PrimExpr>& kernel_size,
                     const ffi::Array<PrimExpr>& stride_size,
                     const ffi::Array<PrimExpr>& dilation_size,
                     const ffi::Array<PrimExpr>& padding_size, PoolType pool_type, bool ceil_mode,
                     const std::string& layout = "NCDHW", bool count_include_pad = true) {
  int depth_axis = -1, height_axis = -1, width_axis = -1;
  TVM_FFI_ICHECK(find_depth_height_width(layout, &depth_axis, &height_axis, &width_axis))
      << "Unsupported layout " << layout;
  std::vector<int> axis = {depth_axis, height_axis, width_axis};
  return pool_impl_nd(x, kernel_size, stride_size, dilation_size, padding_size, pool_type,
                      ceil_mode, axis, count_include_pad);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_POOLING_H_
