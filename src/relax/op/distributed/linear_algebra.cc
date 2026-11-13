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

#include "linear_algebra.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/relax/op_attr.h>

#include <algorithm>
namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeMatmul(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TensorType x1_ty = input_dtensor_tys[0]->tensor_ty;
  TensorType x2_ty = input_dtensor_tys[1]->tensor_ty;

  const auto* attrs = call->attrs.as<MatmulAttrs>();
  ffi::Optional<PrimType> out_dtype = attrs->out_dtype.has_value()
                                          ? PrimType(attrs->out_dtype.value())
                                          : InferBinaryArithOpOutDtype(call, ctx, x1_ty, x2_ty);

  if (x1_ty->IsUnknownNdim() || x2_ty->IsUnknownNdim()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul requires both inputs to have known ndim. However, "
        << (x1_ty->IsUnknownNdim() ? "x1" : "x2") << " has unknown ndim.";
  }

  int x1_ndim = x1_ty->ndim;
  int x2_ndim = x2_ty->ndim;
  if (x1_ndim == 0 || x2_ndim == 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul requires both inputs to have at least 1 dimension. However, "
        << (x1_ndim == 0 ? "x1" : "x2") << " is a 0-rank tensor.";
  }

  int x1_prepended = 0;
  int x2_appended = 0;
  if (x1_ndim == 1) {
    x1_ndim = 2;
    x1_prepended = 1;
  }
  if (x2_ndim == 1) {
    x2_ndim = 2;
    x2_appended = 1;
  }
  int output_ndim = std::max(x1_ndim, x2_ndim) - x1_prepended - x2_appended;

  const auto* x1_shape = x1_ty->shape.as<ShapeExprObj>();
  const auto* x2_shape = x2_ty->shape.as<ShapeExprObj>();
  if (x1_shape == nullptr || x2_shape == nullptr) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "input of distributed operator must have shape";
  }

  ffi::Array<PrimExpr> x1_shape_prefix{x1_shape->values.begin(),
                                       x1_shape->values.end() - 2 + x1_prepended};
  ffi::Array<PrimExpr> x2_shape_prefix{x2_shape->values.begin(),
                                       x2_shape->values.end() - 2 + x2_appended};
  ffi::Optional<ffi::Array<PrimExpr>> output_shape_prefix =
      InferBinaryBroadcastShape(call, ctx, x1_shape_prefix, x2_shape_prefix);
  TVM_FFI_ICHECK(output_shape_prefix.has_value()) << "Failed to infer output shape of Matmul";
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr x1_reduction_length = x1_shape->values[x1_ty->ndim - 1];
  PrimExpr x2_reduction_length = x2_shape->values[x2_ndim - 2];
  if (analyzer->CanProve(x1_reduction_length != x2_reduction_length)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul requires the reduction length of x1 and x2 to be equal. However, "
           "the reduction lengths of x1 and x2 are "
        << x1_reduction_length << " and " << x2_reduction_length << " respectively.";
  }

  ffi::Array<PrimExpr> output_shape = output_shape_prefix.value();
  if (!x1_prepended) {
    output_shape.push_back(x1_shape->values[x1_ndim - 2]);
  }
  if (!x2_appended) {
    output_shape.push_back(x2_shape->values[x2_ndim - 1]);
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(output_shape.size()), output_ndim);
  TensorType output_tensor_ty(ShapeExpr(output_shape), out_dtype);
  return InferShardingSpec(call, ctx, output_tensor_ty, distributed::BuildAxisGraphMatmul);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("relax.matmul")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeMatmul);
  // clang-format on
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
