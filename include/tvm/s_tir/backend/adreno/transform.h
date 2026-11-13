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
 * \file tvm/s_tir/backend/adreno/transform.h
 * \brief S-TIR specific Adreno GPU transformation passes.
 */
#ifndef TVM_S_TIR_BACKEND_ADRENO_TRANSFORM_H_
#define TVM_S_TIR_BACKEND_ADRENO_TRANSFORM_H_

#include <tvm/base.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/s_tir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>

#include <string>
#include <vector>

namespace tvm {
namespace s_tir {
namespace backend {
namespace adreno {
namespace transform {

using tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;

/*!
 * \brief Texture flattening pass.
 * \return The pass.
 */
TVM_DLL Pass TextureFlatten();

/*!
 * \brief Inject Texture Allocation intrinsic.
 * \return The pass.
 */
TVM_DLL Pass InjectTextureAlloc();

}  // namespace transform
}  // namespace adreno
}  // namespace backend
}  // namespace s_tir
}  // namespace tvm

#endif  // TVM_S_TIR_BACKEND_ADRENO_TRANSFORM_H_
