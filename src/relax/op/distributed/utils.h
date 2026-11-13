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
 * \file utils.h
 * \brief The util function for dtensor infer type
 */

#ifndef TVM_RELAX_OP_DISTRIBUTED_UTILS_H_
#define TVM_RELAX_OP_DISTRIBUTED_UTILS_H_

#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/op_attr.h>
#include <tvm/sym/analyzer.h>

#include "../op_common.h"

namespace tvm {
namespace relax {
namespace distributed {

/*!
 * \brief Get the dtensor type of the operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The dtensor type of each input.
 * \note This function require every input tensor to be DTensor.
 */
ffi::Array<distributed::DTensorType> GetInputDTensorType(const Call& call, const BlockBuilder& ctx);

/*!
 * \brief Perform a local sharding spec propagation to infer the output dtensor
          type or tuple of dtensor type.
 *
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param output_ty The original output type
 * \param f_build_graph The function to build axis graph
 * \return The inferred output type
 */
Type InferShardingSpec(const Call& call, const BlockBuilder& ctx, const Type& orig_output_ty,
                       distributed::FBuildAxisGraph f_build_graph);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_DISTRIBUTED_UTILS_H_
