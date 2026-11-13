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
 * \brief instance normalization op constructions
 * \file nn/instance_norm.h
 */
#ifndef TVM_TOPI_NN_INSTANCE_NORM_H_
#define TVM_TOPI_NN_INSTANCE_NORM_H_

#include <tvm/te/operation.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Instance normalization.
 * \param data N-D tensor with shape [d_0, d_1, ..., d_{N-1}]
 * \param gamma K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where K == len(axis) and
 *              d_{axis_k} == r_k
 * \param beta Optional, K-D tensor with shape [r_0, r_1, ..., r_{K-1}] where
 *             d_{axis_k} == r_k
 * \param channel_axis The axis of the channel dimension
 * \param axis The axis to normalize over (the axis along which mean and variance are
 * computed).
 * \param epsilon The epsilon value to avoid division by zero.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 * \return The normalized tensor, with the same shape as data.
 */
inline Tensor instance_norm(const Tensor& data, const Tensor& gamma,
                            const ffi::Optional<Tensor>& beta, int channel_axis,
                            const ffi::Array<int64_t>& axis, double epsilon,
                            std::string name = "T_instance_norm", std::string tag = kInjective) {
  using namespace tvm::prim;
  const auto& data_type = data->dtype;
  const auto& gamma_type = gamma->dtype;
  const auto& beta_type = beta.has_value() ? beta.value()->dtype : data_type;
  TVM_FFI_ICHECK(data_type == gamma_type && data_type == beta_type)
      << "instance_norm: data, gamma and beta must have the same type";
  TVM_FFI_ICHECK(data_type == PrimType::Float(32) || data_type == PrimType::Float(16))
      << "instance_norm: only support float32 and float16 for now";
  bool is_float16 = data_type == PrimType::Float(16);
  // sum x and x^2
  auto ndim = data->shape.size();
  TVM_FFI_ICHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
  auto real_axis = GetRealAxis(static_cast<int>(ndim), axis);
  auto reduce_axes = MakeReduceAxes(real_axis, data);
  auto target_shape =
      MakeReduceTargetShape(real_axis, data, /*keepdims=*/false, /*atleast1d=*/true);
  auto func = MakeTupleSumReducer();
  PrimType f32_ty = PrimType::Float(32);

  auto compute = [ndim, is_float16, &real_axis, &reduce_axes, &func, &data,
                  f32_ty](const ffi::Array<PrimVar>& indices) {
    ffi::Array<PrimExpr> eval_range;
    int arg_counter = 0;
    int red_counter = 0;

    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        // real_axis contains i
        eval_range.push_back(reduce_axes[red_counter]);
        red_counter++;
      } else {
        eval_range.push_back(indices[arg_counter]);
        arg_counter++;
      }
    }
    auto square = [is_float16, f32_ty](const PrimExpr& x) {
      if (is_float16) {
        return prim::Cast(f32_ty, x) * prim::Cast(f32_ty, x);
      }
      return x * x;
    };
    if (is_float16) {
      return func({prim::Cast(f32_ty, data(eval_range)), square(data(eval_range))}, reduce_axes,
                  nullptr);
    } else {
      return func({data(eval_range), square(data(eval_range))}, reduce_axes, nullptr);
    }
  };

  auto temp_x_x2 =
      tvm::te::compute(target_shape, compute, data->op.value()->name + "_red_temp", kCommReduce);

  auto temp_x = temp_x_x2[0];
  auto temp_x2 = temp_x_x2[1];

  auto reduce_extent = MakeConst(PrimType(data->dtype), 1);
  for (int i : real_axis) {
    reduce_extent *= data->shape[i];
  }
  auto instance_norm_func = [&](const ffi::Array<PrimVar>& indices) {
    ffi::Array<PrimVar> reduce_indices, non_reduce_indices;

    for (int i = 0, n = static_cast<int>(indices.size()); i < n; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        reduce_indices.push_back(indices[i]);
      } else {
        non_reduce_indices.push_back(indices[i]);
      }
    }
    PrimVar channel = indices[channel_axis];
    auto mean = temp_x(non_reduce_indices) / reduce_extent;
    auto var = temp_x2(non_reduce_indices) / reduce_extent - mean * mean;
    auto instance_norm =
        (data(indices) - mean) * tvm::prim::rsqrt(var + MakeConst(var.ty(), epsilon));
    if (is_float16) {
      instance_norm = prim::Cast(PrimType::Float(16), instance_norm);
    }
    instance_norm = topi::multiply(instance_norm, gamma(channel));
    if (beta.has_value()) {
      instance_norm = topi::add(instance_norm, beta.value()(channel));
    }
    return instance_norm;
  };
  return tvm::te::compute(data->shape, instance_norm_func, name, tag);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_NN_INSTANCE_NORM_H_
