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
 * \brief layer normalization op constructions
 * \file nn/layer_norm.h
 */
#ifndef TVM_TOPI_NN_LAYER_NORM_H_
#define TVM_TOPI_NN_LAYER_NORM_H_

#include <tvm/te/operation.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Layer normalization.
 * \param data N-D tensor with shape [d_0, d_1, ..., d_{N-1}]
 * \param gamma K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where K == len(axis) and
 *              d_{axis_k} == r_k
 * \param beta Optional, K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where
 *             d_{axis_k} == r_k
 * \param axis The axis to normalize over.
 * \param epsilon The epsilon value to avoid division by zero.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 * \return The normalized tensor, with the same shape as data.
 */
inline Tensor layer_norm(const Tensor& data, const Tensor& gamma, const ffi::Optional<Tensor>& beta,
                         const ffi::Array<int64_t>& axis, double epsilon,
                         std::string name = "T_layer_norm", std::string tag = kInjective) {
  using namespace tvm::prim;
  const auto& data_type = data->dtype;
  const auto& gamma_type = gamma->dtype;
  const auto& beta_type = beta.has_value() ? beta.value()->dtype : data_type;
  TVM_FFI_ICHECK(data_type == gamma_type && data_type == beta_type)
      << "layer_norm: data, gamma and beta must have the same type";
  TVM_FFI_ICHECK(data_type == PrimType::Float(32) || data_type == PrimType::Float(16))
      << "layer_norm: only support float32 and float16 for now";
  bool is_float16 = data_type == PrimType::Float(16);
  // Two-pass algorithm for improved numerical stability:
  //   pass1: mean = E[x]
  //   pass2: var = E[(x - mean)^2]
  auto ndim = data->shape.size();
  TVM_FFI_ICHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
  auto real_axis = GetRealAxis(static_cast<int>(ndim), axis);
  auto reduce_axes = MakeReduceAxes(real_axis, data);
  auto target_shape =
      MakeReduceTargetShape(real_axis, data, /*keepdims=*/false, /*atleast1d=*/false);
  PrimType f32_ty = PrimType::Float(32);

  auto make_eval_range = [&real_axis, &reduce_axes,
                          ndim](const ffi::Array<PrimVar>& non_reduce_indices) {
    ffi::Array<PrimExpr> eval_range;
    int arg_counter = 0;
    int red_counter = 0;

    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        // real_axis contains i
        eval_range.push_back(reduce_axes[red_counter]);
        red_counter++;
      } else {
        eval_range.push_back(non_reduce_indices[arg_counter]);
        arg_counter++;
      }
    }
    return eval_range;
  };

  Tensor temp_sum = te::compute(
      target_shape,
      [is_float16, &data, &reduce_axes, &make_eval_range,
       f32_ty](const ffi::Array<PrimVar>& indices) {
        auto eval_range = make_eval_range(indices);
        PrimExpr x = data(eval_range);
        if (is_float16) {
          x = prim::Cast(f32_ty, x);
        }
        return prim::sum(x, reduce_axes);
      },
      data->op.value()->name + "_sum", kCommReduce);

  PrimType reduce_dtype = is_float16 ? PrimType::Float(32) : PrimType(data->dtype);
  PrimExpr reduce_extent = MakeConst(reduce_dtype, 1);
  for (int i : real_axis) {
    reduce_extent *= data->shape[i];
  }
  Tensor temp_mean = te::compute(
      target_shape,
      [&temp_sum, &reduce_extent](const ffi::Array<PrimVar>& indices) {
        return temp_sum(indices) / reduce_extent;
      },
      data->op.value()->name + "_mean", kInjective);

  Tensor temp_var_sum = te::compute(
      target_shape,
      [is_float16, &data, &reduce_axes, &make_eval_range, &temp_mean,
       f32_ty](const ffi::Array<PrimVar>& indices) {
        auto eval_range = make_eval_range(indices);
        PrimExpr x = data(eval_range);
        if (is_float16) {
          x = prim::Cast(f32_ty, x);
        }
        PrimExpr diff = x - temp_mean(indices);
        return prim::sum(diff * diff, reduce_axes);
      },
      data->op.value()->name + "_var_sum", kCommReduce);

  auto layer_norm_func = [&](const ffi::Array<PrimVar>& indices) {
    ffi::Array<PrimVar> reduce_indices, non_reduce_indices;
    for (int i = 0, n = static_cast<int>(indices.size()); i < n; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        reduce_indices.push_back(indices[i]);
      } else {
        non_reduce_indices.push_back(indices[i]);
      }
    }
    auto mean = temp_mean(non_reduce_indices);
    auto var = temp_var_sum(non_reduce_indices) / reduce_extent;
    auto layer_norm = (data(indices) - mean) * prim::rsqrt(var + MakeConst(var.ty(), epsilon));
    if (is_float16) {
      layer_norm = prim::Cast(PrimType::Float(16), layer_norm);
    }
    layer_norm = topi::multiply(layer_norm, gamma(reduce_indices));
    if (beta.has_value()) {
      layer_norm = topi::add(layer_norm, beta.value()(reduce_indices));
    }
    return layer_norm;
  };
  return te::compute(data->shape, layer_norm_func, name, tag);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_NN_LAYER_NORM_H_
