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
 * \file tvm/relax/backend.h
 * \brief Relax backend specific transformation passes.
 */
#ifndef TVM_RELAX_BACKEND_H_
#define TVM_RELAX_BACKEND_H_

#include <tvm/base.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {
namespace transform {

/*!
 * \brief Perform builtin lowering to map most of the op to VM builtin functions.
 *
 * \return The Pass.
 */
TVM_DLL Pass LowerRuntimeBuiltin();

/*!
 * \brief Lower the shape expression in relax to VM shape heap and TIR functions.
 *
 * \return The Pass.
 */
TVM_DLL Pass VMShapeLower();

}  // namespace transform
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_BACKEND_H_
