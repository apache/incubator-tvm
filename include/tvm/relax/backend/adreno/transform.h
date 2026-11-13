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
 * \file tvm/relax/backend/adreno/transform.h
 * \brief Adreno GPU specific transformation passes.
 */
#ifndef TVM_RELAX_BACKEND_ADRENO_TRANSFORM_H_
#define TVM_RELAX_BACKEND_ADRENO_TRANSFORM_H_

#include <tvm/base.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
namespace tvm {
namespace relax {
namespace backend {
namespace adreno {
namespace transform {

using Pass = tvm::transform::Pass;
using PassInfo = tvm::transform::PassInfo;
using PassContext = tvm::transform::PassContext;
using Function = tvm::relax::Function;
using DataflowBlock = tvm::relax::DataflowBlock;
using tvm::relax::transform::CreateFunctionPass;
using tvm::transform::CreateModulePass;

/*!
 * \brief This pass is designed to annotate the memory scope information via VDevice attribute.
 * This pass need operator attrbutes which in general vanish aftre legalization.
 * FuseOps and FuseTIR are modified to pass on the operator specific attributes and also
 * op_pattern details as part of the tirx::Function. This pass is Adreno specific and annotates each
 * BindingVar with appropriate HintInDevice. RealizeVDevice pass followed by handles these hints.
 * Followed by this pass we also invoke SpecializeFunctionBasedOnCallSite which updates the
 * var_tensor_map based on this new VDevice information.
 */
TVM_DLL Pass AnnotateCustomMemoryScope(Target target);

/*
 * \brief This is a texture specific pass that can optimize unnecessary to_device copies.
 * Like texture_scope -> ToVDevice -> global scope. In this case the producer can directly
 * store into global scope avoiding unnecessary device copy.
 */
TVM_DLL Pass FoldVDeviceScopeChange();

}  // namespace transform
}  // namespace adreno
}  // namespace backend
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_ADRENO_TRANSFORM_H_
