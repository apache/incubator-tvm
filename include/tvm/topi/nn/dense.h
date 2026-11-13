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
 * \brief Dense op constructions
 * \file nn/dense.h
 */
#ifndef TVM_TOPI_NN_DENSE_H_
#define TVM_TOPI_NN_DENSE_H_

#include <tvm/te/operation.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Creates an operation that calculates data * weight^T + bias
 *
 * \param data Tensor with shape [batch, in_dim]
 * \param weight Tensor with shape [out_dim, in_dim]
 * \param bias Tensor with shape [out_dim]. Optional; to omit bias, pass std::nullopt
 * \param out_dtype Output data type. Used for mixed precision.
 *
 * \return Tensor with shape [batch, out_dim]
 */
inline tvm::te::Tensor dense(const tvm::te::Tensor& data, const tvm::te::Tensor& weight,
                             const ffi::Optional<tvm::te::Tensor>& bias,
                             const PrimType& out_dtype) {
  TVM_FFI_ICHECK_EQ(data->shape.size(), 2) << "dense requires 2-D data";
  TVM_FFI_ICHECK_EQ(weight->shape.size(), 2) << "dense requires 2-D weight";
  if (bias.has_value()) {
    TVM_FFI_ICHECK_EQ(bias.value()->shape.size(), 1) << "dense requires 1-D bias";
  }

  auto batch = data->shape[0];
  auto in_dim = data->shape[1];
  auto out_dim = weight->shape[0];

  auto k = tvm::te::reduce_axis(Range(0, in_dim), "k");
  auto matmul = tvm::te::compute(
      {batch, out_dim},
      [&](PrimVar i, PrimVar j) {
        return tvm::prim::sum(
            tvm::prim::cast(data(i, k), out_dtype) * tvm::prim::cast(weight(j, k), out_dtype), {k});
      },
      "tensor", "dense");

  if (bias.has_value()) {
    matmul = tvm::te::compute(
        {batch, out_dim},
        [&](PrimVar i, PrimVar j) {
          return matmul(i, j) + tvm::prim::cast(bias.value()(j), out_dtype);
        },
        "tensor", kBroadcast);
  }

  return matmul;
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_DENSE_H_
