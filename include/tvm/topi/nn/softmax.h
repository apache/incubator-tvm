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
 * \brief Softmax op constructions
 * \file nn/softmax.h
 */
#ifndef TVM_TOPI_NN_SOFTMAX_H_
#define TVM_TOPI_NN_SOFTMAX_H_

#include <tvm/te/operation.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <string>

namespace tvm {
namespace topi {
namespace nn {

using namespace tvm::te;

/*!
 * \brief Softmax activation
 *
 * \param x The input tensor. Can be any dimension
 * \param axis The channel axis along which softmax is performed
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the softmax operation
 */
inline Tensor softmax(const Tensor& x, int axis = -1, std::string name = "tensor",
                      std::string tag = "softmax_output") {
  using namespace tvm::prim;
  auto input_shape = x->shape;
  auto ndim = input_shape.size();
  if (axis < 0) {
    axis = ndim + axis;
  }
  TVM_FFI_ICHECK_LT(axis, ndim) << "axis parameter should be less than input dim";

  auto k1 = tvm::te::reduce_axis(Range(0, input_shape[axis]), "k1");
  auto k2 = tvm::te::reduce_axis(Range(0, input_shape[axis]), "k2");
  auto reduced_shape = MakeReduceTargetShape({axis}, x, false, false);

  tvm::ffi::Map<ffi::String, ffi::Any> attrs;
  attrs.Set(tvm::topi::attr::kAxis, IntImm::Int32(axis));

  auto insert_reduce_index = [axis, ndim](const ffi::Array<PrimVar>& indices,
                                          const IterVar& reduce_index) {
    ffi::Array<PrimExpr> eval_range;
    int arg_counter = 0;
    for (size_t i = 0; i < ndim; ++i) {
      if (static_cast<int>(i) == axis) {
        eval_range.push_back(reduce_index);
      } else {
        eval_range.push_back(indices[arg_counter++]);
      }
    }
    return eval_range;
  };

  auto get_non_reduce_indices = [axis, ndim](const ffi::Array<PrimVar>& indices) {
    ffi::Array<PrimExpr> non_reduce_indices;
    for (size_t i = 0; i < ndim; ++i) {
      if (static_cast<int>(i) != axis) non_reduce_indices.push_back(indices[i]);
    }
    return non_reduce_indices;
  };

  auto _compute_max = [&](const ffi::Array<PrimVar>& indices) {
    auto eval_range = insert_reduce_index(indices, k1);
    return topi::MaxOp(x(eval_range), {k1});
  };

  auto _compute_exp = [&](const Tensor& max_elem, const ffi::Array<PrimVar>& indices) {
    auto non_reduce_indices = get_non_reduce_indices(indices);
    return tvm::prim::exp(x(indices) - max_elem(non_reduce_indices));
  };

  auto _compute_expsum = [&](const Tensor& exp, const ffi::Array<PrimVar>& indices) {
    auto eval_range = insert_reduce_index(indices, k2);
    return tvm::prim::sum(exp(eval_range), {k2});
  };

  auto _normalize = [&](const Tensor& exp, const Tensor& expsum,
                        const ffi::Array<PrimVar>& indices) {
    auto non_reduce_indices = get_non_reduce_indices(indices);
    return exp(indices) / expsum(non_reduce_indices);
  };

  auto max_elem = tvm::te::compute(reduced_shape, _compute_max);
  auto exp = tvm::te::compute(input_shape, [&](const ffi::Array<PrimVar>& indices) {
    return _compute_exp(max_elem, indices);
  });
  auto expsum = tvm::te::compute(reduced_shape, [&](const ffi::Array<PrimVar>& indices) {
    return _compute_expsum(exp, indices);
  });
  return tvm::te::compute(
      input_shape,
      [&](const ffi::Array<PrimVar>& indices) { return _normalize(exp, expsum, indices); }, name,
      tag, attrs);
}

/*!
 * \brief Log softmax activation
 *
 * \param x The input tensor. 2-D where log softmax is performed along the second dimension
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the log softmax operation
 */
inline Tensor log_softmax(const Tensor& x, std::string name = "tensor",
                          std::string tag = "log_softmax_output") {
  TVM_FFI_ICHECK_EQ(x->shape.size(), 2) << "Log softmax requires 2-D input";

  PrimExpr m = x->shape[0];
  PrimExpr n = x->shape[1];

  auto k = tvm::te::reduce_axis(Range(0, n), "k");
  auto max_elem =
      tvm::te::compute({m}, [&](PrimVar i) { return tvm::max(x(i, k), ffi::Array<IterVar>{k}); });
  k = tvm::te::reduce_axis(Range(0, n), "k");

  auto expsum = tvm::te::compute(
      {m}, [&](PrimVar i) { return tvm::prim::sum(tvm::prim::exp(x(i, k) - max_elem(i)), {k}); });

  return tvm::te::compute(
      x->shape,
      [&](PrimVar i, PrimVar j) { return x(i, j) - max_elem(i) - tvm::prim::log(expsum(i)); }, name,
      tag);
}

}  // namespace nn
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_NN_SOFTMAX_H_
