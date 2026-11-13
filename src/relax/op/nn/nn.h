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
 * \file nn.h
 * \brief The functions to make Relax neural network operator calls.
 */

#ifndef TVM_RELAX_OP_NN_NN_H_
#define TVM_RELAX_OP_NN_NN_H_

#include <tvm/relax/op/nn.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief Rectified linear unit. */
Expr relu(Expr data);

/*! \brief Leaky rectified linear unit. */
Expr leakyrelu(Expr data, double alpha);

/*! \brief Gaussian Error Linear Units function. */
Expr gelu(Expr data);

/*! \brief Gaussian Error Linear Units function approximated by tanh. */
Expr gelu_tanh(Expr data);

/*! \brief Parametric Rectified Linear Unit function.*/
Expr prelu(Expr data, Expr alpha, int axis);

/*! \brief Scaled Exponential Linear Unit function. */
Expr selu(Expr data);

/*! \brief Sigmoid Linear Unit function. */
Expr silu(Expr data);

/*! \brief Softmax function. */
Expr softmax(Expr data, int axis);

/*! \brief Softplus function. */
Expr softplus(Expr data, double beta, double threshold);

/*! \brief LogSoftmax function. */
Expr log_softmax(Expr data, int axis);

/*! \brief Pixel Shuffle function. */
Expr pixel_shuffle(Expr data, int upscale_factor);

/*! \brief Compute batch normalization. */
Expr batch_norm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,  //
                int axis, double epsilon, bool center, bool scale, double momentum, bool training);

/*! \brief Compute layer normalization. */
Expr layer_norm(Expr data, Expr gamma, Expr beta, ffi::Array<int64_t> axes, double epsilon,
                bool center, bool scale);

/*! \brief Compute group normalization. */
Expr group_norm(Expr data, Expr gamma, Expr beta, int num_groups, int channel_axis,
                ffi::Array<int64_t> axes, double epsilon, bool center, bool scale);

/*! \brief Compute instance normalization. */
Expr instance_norm(Expr data, Expr gamma, Expr beta, int channel_axis, ffi::Array<int64_t> axes,
                   double epsilon, bool center, bool scale);

/*! \brief Compute root mean square normalization. */
Expr rms_norm(Expr data, Expr weight, ffi::Array<int64_t> axes, double epsilon);

/*!
 * \brief Applies the dropout operation to the input tensor.
 * \param data The input data to the operator.
 * \param rate The probability for an element to be reset to 0.
 * \return A Tuple of two tensors.
 * The first one is the original tensor and the second one is a
 * mask tensor (1.0 where element not dropped, 0.0 where dropped)
 */
Expr dropout(Expr data, double rate);

/*! \brief CrossEntropy with logits. */
Expr cross_entropy_with_logits(Expr predictions, Expr labels);

/*! \brief Negative log likelihood loss. */
Expr nll_loss(Expr predictions, Expr targets, ffi::Optional<Expr> weights, ffi::String reduction,
              int ignore_index);

/*! \brief Batch flatten: flatten all dimensions except the first (batch) dimension. */
Expr batch_flatten(Expr data);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_NN_NN_H_
