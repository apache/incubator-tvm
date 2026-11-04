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

#include "manipulate.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/relax/op_attr.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypePermuteDims(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TensorType data_ty = input_dtensor_tys[0]->tensor_ty;

  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();

  // Todo(relax-team): revisit here for better check on if the input tensor has
  // ndim same as the number of input axes.
  if (!attrs->axes.has_value() && data_ty->IsUnknownNdim()) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Input of distributed operator must have known ndim";
  }

  if (attrs->axes.has_value()) {
    int n_axis = attrs->axes.value().size();
    if (!data_ty->IsUnknownNdim() && n_axis != data_ty->ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "PermuteDims expects the number of input axes to equal the ndim of the "
             "input tensor. However, the tensor ndim is "
          << data_ty->ndim << " while the given number of axes is " << n_axis;
    }
  }

  std::vector<int> axes;
  if (attrs->axes.has_value()) {
    axes = NormalizeAxes(call, ctx, data_ty->ndim, attrs->axes.value());
  } else {
    // Construct the reverse permutation via std::iota
    axes.resize(data_ty->ndim);
    std::iota(axes.rbegin(), axes.rend(), 0);
  }
  if (IsIdentityPermutation(axes)) {
    return input_dtensor_tys[0];
  }

  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Input of distributed operator must have known shape";
  }
  std::vector<PrimExpr> new_shape;
  new_shape.reserve(data_ty->ndim);
  for (int i = 0; i < data_ty->ndim; ++i) {
    new_shape.push_back(data_shape->values[axes[i]]);
  }
  TensorType output_tensor_ty(ShapeExpr(new_shape), data_ty->dtype);
  return InferShardingSpec(call, ctx, output_tensor_ty, distributed::BuildAxisGraphPermuteDims);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("relax.permute_dims")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypePermuteDims);
  // clang-format on
}

Type InferDistTypeReshape(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Reshape op should take 2 arguments";
  }
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TensorType data_ty = input_dtensor_tys[0]->tensor_ty;

  const auto* new_shape_ty = GetTypeAs<ShapeTypeObj>(call->args[1]);
  if (!data_ty.defined()) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Reshape requires the input data to be Tensor. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (new_shape_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Reshape requires the input new shape to be Shape. However, the given one is "
        << call->args[1]->ty->GetTypeKey();
  }

  ffi::Optional<ffi::Array<PrimExpr>> old_shape_values;
  if (data_ty->shape.has_value()) {
    const auto* old_shape_ty = GetTypeAs<ShapeTypeObj>(data_ty->shape.value());
    TVM_FFI_ICHECK_NOTNULL(old_shape_ty);
    old_shape_values = old_shape_ty->values;
  }

  if (new_shape_ty->values.has_value() && old_shape_values.has_value()) {
    PrimExpr new_shape_prod = ComputeShapeProduct(new_shape_ty->values.value());
    PrimExpr old_shape_prod = ComputeShapeProduct(old_shape_values.value());
    if (ctx->GetAnalyzer()->CanProve(old_shape_prod != new_shape_prod)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "Reshape expects the new shape to be convertible from the old shape. "
             "However, the old shape is "
          << data_ty->shape << ", with product " << old_shape_prod << ", while the new shape is "
          << call->args[1] << ", with product " << new_shape_prod;
    }
  }
  Expr target_shape = call->args[1];
  Type output_tensor_ty = Type::Missing();
  // If shape values are defined, use them
  if (target_shape->IsInstance<VarObj>() && new_shape_ty->values.has_value()) {
    output_tensor_ty = TensorType(ShapeExpr(new_shape_ty->values.value()), data_ty->dtype);
  } else {
    output_tensor_ty = TensorType(target_shape, data_ty->dtype);
  }
  return InferShardingSpec(call, ctx, output_tensor_ty, distributed::BuildAxisGraphReshape);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("relax.reshape")
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeReshape);
  // clang-format on
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
