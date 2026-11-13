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
 * \file ccl.h
 * \brief The functions to make Relax ccl operator calls.
 */

#ifndef TVM_RELAX_OP_CCL_CCL_H_
#define TVM_RELAX_OP_CCL_CCL_H_

#include <tvm/relax/op/ccl.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief AllReduce. */
Expr allreduce(Expr data, ffi::String op_type, bool in_group);

/*! \brief AllGather. */
Expr allgather(Expr data, int num_workers, bool in_group);

/*! \brief Broadcast data from worker-0 to all other workers. */
Expr broadcast_from_worker0(Expr data);

/*! \brief Perform a scatter operation from worker-0, chunking the given buffer into equal parts. */
Expr scatter_from_worker0(Expr data, int num_workers, int axis);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_CCL_CCL_H_
