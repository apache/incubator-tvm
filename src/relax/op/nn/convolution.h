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
 * \file convolution.h
 * \brief The functions to make Relax neural network convolution operator calls.
 */

#ifndef TVM_RELAX_OP_NN_CONVOLUTION_H_
#define TVM_RELAX_OP_NN_CONVOLUTION_H_

#include <tvm/relax/op/nn.h>

#include <string>
#include <utility>

#include "../op_common.h"

namespace tvm {
namespace relax {

template <typename T>
inline Expr MakeConv(Expr data, Expr weight, ffi::Array<int64_t> strides,
                     ffi::Array<int64_t> padding, ffi::Array<int64_t> dilation, int groups,
                     ffi::String data_layout, ffi::String kernel_layout, ffi::String out_layout,
                     ffi::Optional<DLDataType> out_dtype, std::string op_name) {
  auto attrs = ffi::make_object<T>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = out_dtype;
  const Op op = Op::Get(op_name);
  return Call(Type::Missing(), op, {data, weight}, Attrs(attrs), {});
}

/*! \brief 1D convolution */
Expr conv1d(Expr data, Expr weight, ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
            ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
            ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
            ffi::Optional<DLDataType> out_dtype);

/*! \brief 2D convolution */
Expr conv2d(Expr data, Expr weight, ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
            ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
            ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
            ffi::Optional<DLDataType> out_dtype);

/*! \brief 3D convolution */
Expr conv3d(Expr data, Expr weight, ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
            ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
            ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
            ffi::Optional<DLDataType> out_dtype);

/*!
 * \brief One dimensional transposed convolution operator.
 *
 * This operator is intended to be the backward operator of conv1d. It can be used to calculate the
 * gradient of the result of conv1d w.r.t. the input of conv1d.
 */
Expr conv1d_transpose(Expr data, Expr weight, ffi::Array<int64_t> strides,
                      ffi::Array<int64_t> padding, ffi::Array<int64_t> output_padding,
                      ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
                      ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
                      ffi::Optional<DLDataType> out_dtype);

/*!
 * \brief Two dimensional transposed convolution operator.
 *
 * This operator is intended to be the backward operator of conv2d. It can be used to calculate the
 * gradient of the result of conv2d w.r.t. the input of conv2d.
 */
Expr conv2d_transpose(Expr data, Expr weight, ffi::Array<int64_t> strides,
                      ffi::Array<int64_t> padding, ffi::Array<int64_t> output_padding,
                      ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
                      ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
                      ffi::Optional<DLDataType> out_dtype);

/*!
 * \brief Three dimensional transposed convolution operator.
 *
 * This operator is intended to be the backward operator of conv3d. It can be used to calculate the
 * gradient of the result of conv3d w.r.t. the input of conv3d.
 */
Expr conv3d_transpose(Expr data, Expr weight, ffi::Array<int64_t> strides,
                      ffi::Array<int64_t> padding, ffi::Array<int64_t> output_padding,
                      ffi::Array<int64_t> dilation, int groups, ffi::String data_layout,
                      ffi::String kernel_layout, ffi::Optional<ffi::String> out_layout,
                      ffi::Optional<DLDataType> out_dtype);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_NN_CONVOLUTION_H_
