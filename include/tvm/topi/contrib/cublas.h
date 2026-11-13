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
 * \brief External function interface to cuBLAS libraries
 * \file cublas.h
 */
#ifndef TVM_TOPI_CONTRIB_CUBLAS_H_
#define TVM_TOPI_CONTRIB_CUBLAS_H_

#include <tvm/te/operation.h>
#include <tvm/topi/detail/extern.h>

namespace tvm {
namespace topi {
namespace contrib {

using namespace tvm::te;
using namespace topi::detail;
/*!
 * \brief Create an op that multiplies lhs and rhs with cuBLAS
 *
 * \param lhs The left matrix operand
 * \param rhs The right matrix operand
 * \param transa Whether to transpose lhs
 * \param transb Whether to transpose rhs
 *
 * \return The output tensor
 */
inline Tensor cublas_matmul(const Tensor& lhs, const Tensor& rhs, bool transa, bool transb) {
  auto n = transa ? lhs->shape[1] : lhs->shape[0];
  auto m = transb ? rhs->shape[0] : rhs->shape[1];

  return make_extern(
      {{n, m}}, {lhs->GetDataType()}, {lhs, rhs},
      [&](ffi::Array<TensorVar> ins, ffi::Array<TensorVar> outs) {
        return call_packed({StringImm("tvm.contrib.cublas.matmul"), pack_buffer(ins[0]),
                            pack_buffer(ins[1]), pack_buffer(outs[0]), IntImm::Int32(transa),
                            IntImm::Int32(transb)});
      },
      "C", "", {})[0];
}

/*!
 * \brief Create an op that multiplies batch matrices
 *        lhs and rhs with cuBLAS
 *
 * \param lhs The left matrix operand
 * \param rhs The right matrix operand
 * \param transa Whether to transpose lhs
 * \param transb Whether to transpose rhs
 *
 * \return The output tensor
 */
inline Tensor cublas_batch_matmul(const Tensor& lhs, const Tensor& rhs, bool transa, bool transb) {
  auto b = lhs->shape[0];
  auto n = transa ? lhs->shape[2] : lhs->shape[1];
  auto m = transb ? rhs->shape[1] : rhs->shape[2];

  return make_extern(
      {{b, n, m}}, {lhs->GetDataType()}, {lhs, rhs},
      [&](ffi::Array<TensorVar> ins, ffi::Array<TensorVar> outs) {
        return call_packed({StringImm("tvm.contrib.cublas.batch_matmul"), pack_buffer(ins[0]),
                            pack_buffer(ins[1]), pack_buffer(outs[0]), IntImm::Int32(transa),
                            IntImm::Int32(transb)});
      },
      "C", "", {})[0];
}

}  // namespace contrib
}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_CONTRIB_CUBLAS_H_
