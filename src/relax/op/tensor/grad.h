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
 * \file grad.h
 * \brief The functions to make Relax gradient operators.
 */
#ifndef TVM_RELAX_OP_TENSOR_GRAD_H_
#define TVM_RELAX_OP_TENSOR_GRAD_H_

#include <tvm/relax/op/index.h>
#include <tvm/relax/op/nn.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief No gradient dummy operator.
 * \param input The corresponding input tensor.
 * \return The no-gradient representation w.r.t. input.
 */
Expr no_grad(Expr input);

/*! \brief Backward operator of relax.nll_loss. All parameters except output_grad is the same as
 * relax.nll_loss. Returns the gradient w.r.t. predictions. */
Expr nll_loss_backward(Expr output_grad, Expr predictions, Expr targets,
                       ffi::Optional<Expr> weights, ffi::String reduction, int ignore_index);

/*! \brief Backward operator of relax.max_pool2d. All parameters except output_grad is the same as
 * relax.max_pool2d. Returns the gradient w.r.t. data. */
Expr max_pool2d_backward(Expr output_grad, Expr data, ffi::Array<int64_t> pool_size,
                         ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                         ffi::Array<int64_t> dilation, bool ceil_mode, bool count_include_pad,
                         ffi::String layout, ffi::Optional<ffi::String> out_layout);

/*! \brief Backward operator of relax.avg_pool2d. All parameters except output_grad is the same as
 * relax.avg_pool2d. Returns the gradient w.r.t. data. */
Expr avg_pool2d_backward(Expr output_grad, Expr data, ffi::Array<int64_t> pool_size,
                         ffi::Array<int64_t> strides, ffi::Array<int64_t> padding,
                         ffi::Array<int64_t> dilation, bool ceil_mode, bool count_include_pad,
                         ffi::String layout, ffi::Optional<ffi::String> out_layout);

/*! \brief Backward operator of relax.take. All parameters except output_grad is the same as
 * relax.take. Returns the gradient w.r.t. data. */
Expr take_backward(Expr output_grad, Expr x, Expr indices, ffi::Optional<int64_t> axis);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_GRAD_H_
