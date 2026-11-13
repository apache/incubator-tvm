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
 * \file sampling.h
 * \brief The functions to make Relax tensor sampling operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_SAMPLING_H_
#define TVM_RELAX_OP_TENSOR_SAMPLING_H_

#include <tvm/relax/op/sampling.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Returns a tensor where each row contains the index sampled from the multinomial
 *        probability distribution located in the corresponding row of tensor prob.
 * \param prob A 2-D tensor of shape (batch, vocab_size) representing probability distributions.
 *        Each row is a distribution across vocabulary for a batch, where:
 *        Values range from [0, 1], indicating the probability of each vocabulary item.
 *        The sum of values in each row is 1, forming a valid distribution.
 * \param uniform_sample A 2-D tensor with the shape (n, 1). Values range from 0 to 1, indicating
 *        probabilities sampled uniformly.
 * \param sample_indices The 2-D tensor with the shape [n, 1], which indicates the specific
 *        probability distribution to sample from. The value of sample_indices[i]
 *        determines that the ith token should be sampled from the sample_indices[i]th
 *        probability distribution. For instance, if there are 3 distinct probability
 *        distributions and the requirement is to sample 2, 3, and 4 tokens from each,
 *        then sample_indices would be [0, 0, 1, 1, 1, 2, 2, 2, 2].
 * \param dtype The data type of the output tensor.
 * \return The sampled result.
 */
Expr multinomial_from_uniform(Expr prob, Expr uniform_sample, Expr sample_indices,
                              DLDataType dtype);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_SAMPLING_H_
