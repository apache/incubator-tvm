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
#include "tvm/relax/op/ccl.h"

#include <tvm/relax/block_builder.h>
#include <tvm/relax/op_attr.h>

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeAllReduce(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TVM_FFI_ICHECK(input_dtensor_tys.size() == 1);
  DTensorType input_dtensor_ty = input_dtensor_tys[0];
  TensorType tensor_ty = input_dtensor_ty->tensor_ty;
  DeviceMesh device_mesh = input_dtensor_ty->device_mesh;
  // FIXME: this is a hack where there's only 1d mesh
  return DTensorType(tensor_ty, device_mesh,
                     Placement::FromText(std::string(device_mesh->shape.size(), 'R')));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("relax.ccl.allreduce")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeAllReduce);
  // clang-format on
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
