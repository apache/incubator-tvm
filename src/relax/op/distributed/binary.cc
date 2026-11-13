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

#include "binary.h"

#include <tvm/relax/op_attr.h>

namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferDistTypeBroadcast(
      call, ctx, [](const Call& call, const BlockBuilder& ctx, const Type& lhs, const Type& rhs) {
        return InferBinaryArithOpOutDtype(call, ctx, lhs, rhs);
      });
}

Type InferDistTypeBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferDistTypeBroadcast(
      call, ctx,
      [](const Call& call, const BlockBuilder& ctx, const TensorType& x1_ty,
         const TensorType& x2_ty) { return PrimType::Bool(); });
}

/***************** Arithmetic operators *****************/

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("relax.add")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.divide")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.floor_divide")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.multiply")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.power")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.subtract")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.mod")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.floor_mod")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  /***************** Comparison operators *****************/

  OpDef("relax.equal")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastCMP);

  OpDef("relax.greater")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastCMP);

  OpDef("relax.greater_equal")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastCMP);

  OpDef("relax.less")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastCMP);

  OpDef("relax.less_equal")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastCMP);

  OpDef("relax.not_equal")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastCMP);

  /***************** Min/Max operators *****************/

  OpDef("relax.minimum")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.maximum")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  /***************** Logical operators *****************/

  OpDef("relax.logical_and")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.logical_or")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.logical_xor")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  /***************** Bitwise operators *****************/

  OpDef("relax.bitwise_and")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.bitwise_or")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.bitwise_xor")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.left_shift")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);

  OpDef("relax.right_shift")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeBroadcastArith);
  // clang-format on
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
