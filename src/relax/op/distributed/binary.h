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
 * \file binary.h
 * \brief The functions to infer type for distributed binary operator
 */

#ifndef TVM_RELAX_OP_DISTRIBUTED_BINARY_H_
#define TVM_RELAX_OP_DISTRIBUTED_BINARY_H_

#include <tvm/ffi/extra/visit_error_context.h>

#include <algorithm>

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

template <typename FType>
Type InferDistTypeBroadcast(const Call& call, const BlockBuilder& ctx, FType f_compute_out_dtype) {
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TensorType x1_ty = input_dtensor_tys[0]->tensor_ty;
  TensorType x2_ty = input_dtensor_tys[1]->tensor_ty;

  // Dtype
  ffi::Optional<PrimType> output_dtype = f_compute_out_dtype(call, ctx, x1_ty, x2_ty);

  // ndims
  TVM_FFI_ICHECK(!x1_ty->IsUnknownNdim() && !x2_ty->IsUnknownNdim())
      << "Unknown ndim is not supported for distributed operators.";
  int output_ndim = std::max(x1_ty->ndim, x2_ty->ndim);

  const auto* x1_shape = x1_ty->shape.as<ShapeExprObj>();
  const auto* x2_shape = x2_ty->shape.as<ShapeExprObj>();
  Type output_tensor_ty = Type::Missing();
  // Shapes and ndims
  if (x1_shape && x2_shape) {
    // If all inputs have shapes, directly infer shapes
    ffi::Optional<ffi::Array<PrimExpr>> output_shape =
        InferBinaryBroadcastShape(call, ctx, x1_shape->values, x2_shape->values);
    if (!output_shape.has_value()) {
      output_tensor_ty = TensorType(output_dtype, /*ndim=*/output_ndim);
    } else {
      TVM_FFI_ICHECK_EQ(static_cast<int>(output_shape.value().size()), output_ndim);
      output_tensor_ty = TensorType(ShapeExpr(output_shape.value()), output_dtype);
    }
  } else {
    TVM_FFI_VISIT_THROW(InternalError, call) << "Cannot infer shape for binary broadcast operator.";
  }
  return InferShardingSpec(call, ctx, output_tensor_ty, distributed::BuildAxisGraphBinary);
}

Type InferDistTypeBroadcastArith(const Call& call, const BlockBuilder& ctx);

Type InferDistTypeBroadcastCMP(const Call& call, const BlockBuilder& ctx);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_DISTRIBUTED_BINARY_H_
