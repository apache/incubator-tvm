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
 * \brief Binary op constructions
 * \file nn/bnn.h
 */
#ifndef TVM_TOPI_NN_BNN_H_
#define TVM_TOPI_NN_BNN_H_

#include <tvm/sym/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/tags.h>

#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Binarization and bit-packing along a certain axis.
 *
 * \param data N-D tensor, can be any layout
 * \param axis The axis along which to do binarization and bit-packing. This axis
 * must have a size equal to an integer multiple of 32.
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return Output tensor with dtype uint32
 */
inline tvm::te::Tensor binarize_pack(const tvm::te::Tensor& data, int axis,
                                     std::string name = "PackedInput",
                                     std::string tag = "binarize_pack") {
  auto ishape = data->shape;
  TVM_FFI_ICHECK_EQ(GetConstInt(ishape[axis]) % 32, 0)
      << "binarize_pack: axis size must be a multiple of 32";

  sym::Analyzer analyzer;
  auto n = ishape.size();
  ffi::Array<PrimExpr> oshape;
  for (size_t i = 0; i < n; ++i) {
    oshape.push_back(i == static_cast<size_t>(axis) ? analyzer->Simplify(indexdiv(ishape[i], 32))
                                                    : ishape[i]);
  }

  return tvm::te::compute(
      oshape,
      [&](const ffi::Array<PrimVar>& indices) {
        ffi::Array<PrimExpr> start_idx;
        for (size_t i = 0; i < n; ++i) {
          start_idx.push_back(i == static_cast<size_t>(axis) ? indices[i] * 32
                                                             : static_cast<PrimExpr>(indices[i]));
        }
        PrimExpr packed = IntImm(PrimType::UInt(32), 0);
        for (size_t j = 0; j < 32; ++j) {
          ffi::Array<PrimExpr> idx;
          for (size_t i = 0; i < n; ++i) {
            idx.push_back(i == static_cast<size_t>(axis) ? start_idx[i] + static_cast<int>(j)
                                                         : start_idx[i]);
          }
          auto sign = tvm::prim::cast(data(idx) >= 0, PrimType::UInt(32));
          packed = (packed | sign);
          if (j == 31) {
            return packed;
          }
          packed = packed << 1;
        }
        return packed;  // never reached, but suppress compiler warning
      },
      name, tag);
}

/*!
 * \brief Binary matrix multiplication using xor and bit-count
 *
 * \param data Tensor with shape [batch, in_dim], dtype is uint32
 * \param weight Tensor with shape [out_dim, in_dim], dtype is uint32
 *
 * \return Tensor with shape [batch, out_dim], dtype is float32
 */
inline tvm::te::Tensor binary_dense(const tvm::te::Tensor& data, const tvm::te::Tensor& weight) {
  using namespace tvm::prim;
  TVM_FFI_ICHECK_EQ(data->shape.size(), 2) << "binary_dense requires 2-D data";
  TVM_FFI_ICHECK_EQ(weight->shape.size(), 2) << "binary_dense requires 2-D weight";
  TVM_FFI_ICHECK_EQ(data->dtype, PrimType::UInt(32)) << "binary_dense requires uint32 data";
  TVM_FFI_ICHECK_EQ(weight->dtype, PrimType::UInt(32)) << "binary_dense requires uint32 weight";

  auto batch = data->shape[0];
  auto in_dim = data->shape[1];
  auto out_dim = weight->shape[0];

  auto k = tvm::te::reduce_axis(Range(0, in_dim), "k");
  auto matmul = tvm::te::compute(
      {batch, out_dim},
      [&](PrimVar i, PrimVar j) {
        return tvm::prim::sum(popcount(data(i, k) ^ weight(j, k)), {k});
      },
      "tensor", "binary_dense");

  return tvm::te::compute(
      {batch, out_dim}, [&](PrimVar i, PrimVar j) { return 32 * in_dim - 2.0f * matmul(i, j); },
      "tensor", kElementWise);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_BNN_H_
