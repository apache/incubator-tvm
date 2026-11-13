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

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/relax/op_attr.h>

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeCallTIR(const Call& call, const BlockBuilder& ctx) {
  if (call->ty_args.size() != 1) {
    TVM_FFI_VISIT_THROW(InternalError, call) << "ty_args should have exact 1 output type.";
  }
  TVM_FFI_ICHECK(call->args[0]->IsInstance<GlobalVarObj>())
      << "call_tir expects the first argument to be a GlobalVar referring to a TIR tirx::Function. "
      << "However, gets " << call->args[0];
  return call->ty_args[0];
}

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("relax.call_tir")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeCallTIR);
  // clang-format on
}

Type InferDistTypeStopLiftParams(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "stop_lift_params should have exact 1 arg.";
  }
  return call->args[0]->ty.as_or_throw<Type>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.builtin.stop_lift_params")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder,
                InferDistTypeStopLiftParams);
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
