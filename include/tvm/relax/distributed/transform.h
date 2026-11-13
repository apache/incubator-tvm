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
 * \file tvm/relax/distributed/transform.h
 * \brief Relax distributed specific transformation passes.
 */
#ifndef TVM_RELAX_DISTRIBUTED_TRANSFORM_H_
#define TVM_RELAX_DISTRIBUTED_TRANSFORM_H_

#include <tvm/base.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>

namespace tvm {
namespace relax {
namespace distributed {
namespace transform {

using Pass = tvm::transform::Pass;
using PassInfo = tvm::transform::PassInfo;
using PassContext = tvm::transform::PassContext;
using Function = tvm::relax::Function;
using DataflowBlock = tvm::relax::DataflowBlock;
using tvm::transform::CreateModulePass;

/*!
 * \brief Propagate sharding information.
 *
 * \return The Pass.
 */
TVM_DLL Pass PropagateSharding();

/*!
 * \brief Lower global view TensorIR into local view.
 *
 * \return The Pass.
 */
TVM_DLL Pass LowerGlobalViewToLocalView();

/*!
 * \brief Legalize redistribute op to ccl op.
 *
 * \return The Pass.
 */
TVM_DLL Pass LegalizeRedistribute();

/*!
 * \brief Lower DistIR to Relax
 *
 * \return The Pass.
 */
TVM_DLL Pass LowerDistIR();
}  // namespace transform
}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_TRANSFORM_H_
