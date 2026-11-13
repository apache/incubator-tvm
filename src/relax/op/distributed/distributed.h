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
 * \file distributed.h
 * \brief The functions to make Relax redistribute operator calls.
 */
#ifndef TVM_RELAX_OP_DISTRIBUTED_DISTRIBUTED_H_
#define TVM_RELAX_OP_DISTRIBUTED_DISTRIBUTED_H_

#include <tvm/relax/op/distributed.h>

#include "../op_common.h"
#include "utils.h"

namespace tvm {
namespace relax {

/*!
 * \brief Annotate sharding plan for tensor.
 * \param input The input tensor.
 * \param device_mesh The device mesh of the sharding plan.
 * \param placement The placement of the sharding plan.
 * \return The tensor unmodified.
 */
Expr annotate_sharding(Expr input, distributed::DeviceMesh device_mesh,
                       distributed::Placement placement);

/*!
 * \brief Redistribute tensor.
 * \param input The input tensor.
 * \param device_mesh The device mesh after redistribution.
 * \param placement The placement after redistribution.
 * \return The result.
 */
Expr redistribute(Expr input, distributed::DeviceMesh device_mesh,
                  distributed::Placement placement);

/*!
 * \brief slice tensor into several parts along one axis,
          and each worker takes one part.
          Assumes input is already broadcasted.
          This is a specialized version of redistribute op.
 * \param input The input tensor.
 * \param num_workers The number of workers.
 * \param axis The tensor axis to slice.
 * \return The result.
 */
Expr redistribute_replica_to_shard(Expr input, int num_workers, int axis);
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_DISTRIBUTED_DISTRIBUTED_H_
