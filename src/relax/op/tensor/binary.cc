/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include "binary.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>

#include <algorithm>

namespace tvm {
namespace relax {
using namespace tvm::prim;

template <typename FType>
Type InferTypeBroadcast(const Call& call, const BlockBuilder& ctx, FType f_compute_out_dtype) {
  CheckNumArguments(call, ctx);

  auto lhs_ty = GetType(call->args[0]);
  auto rhs_ty = GetType(call->args[1]);

  TVM_FFI_CHECK(lhs_ty.as<PrimTypeObj>() || lhs_ty.as<TensorTypeObj>(), TypeError)
      << "Arguments to binary operators must be either R.Tensor or R.Prim types, "
      << "but " << call->op << " has LHS type " << lhs_ty;
  TVM_FFI_CHECK(rhs_ty.as<PrimTypeObj>() || rhs_ty.as<TensorTypeObj>(), TypeError)
      << "Arguments to binary operators must be either R.Tensor or R.Prim types, "
      << "but " << call->op << " has RHS type " << rhs_ty;

  // Dtype
  ffi::Optional<PrimType> output_dtype = f_compute_out_dtype(call, ctx, lhs_ty, rhs_ty);

  if (lhs_ty.as<PrimTypeObj>() && rhs_ty.as<PrimTypeObj>()) {
    TVM_FFI_ICHECK(output_dtype.has_value());
    return output_dtype.value();
  }

  // VDevice
  ffi::Optional<VDevice> vdevice = InferBinaryArithOpOutVDevice(call, ctx, lhs_ty, rhs_ty);

  auto get_ndim = [&](const Type& ty) -> int {
    if (ty.as<PrimTypeObj>()) {
      return 1;
    } else if (const auto* tensor = ty.as<TensorTypeObj>()) {
      return tensor->ndim;
    } else {
      return kUnknownNDim;
    }
  };

  // ndims
  int output_ndim = [&]() {
    int lhs_ndim = get_ndim(lhs_ty);
    int rhs_ndim = get_ndim(rhs_ty);
    if (lhs_ndim == kUnknownNDim || rhs_ndim == kUnknownNDim) {
      return kUnknownNDim;
    } else {
      return std::max(lhs_ndim, rhs_ndim);
    }
  }();

  // Shapes

  auto get_shape = [](const Type& ty) -> ffi::Optional<ffi::Array<PrimExpr>> {
    if (ty.as<PrimTypeObj>()) {
      return ffi::Array<PrimExpr>{IntImm::Int64(1)};
    } else if (const auto* tensor = ty.as<TensorTypeObj>()) {
      return tensor->GetShape();
    } else {
      return std::nullopt;
    }
  };

  // If both inputs have a known shape, directly infer the shape of
  // the output.
  auto lhs_shape = get_shape(lhs_ty);
  auto rhs_shape = get_shape(rhs_ty);
  if (lhs_shape && rhs_shape) {
    ffi::Optional<ffi::Array<PrimExpr>> output_shape =
        InferBinaryBroadcastShape(call, ctx, lhs_shape.value(), rhs_shape.value());
    if (output_shape.has_value()) {
      TVM_FFI_ICHECK_EQ(static_cast<int>(output_shape.value().size()), output_ndim);
      return TensorType(ShapeExpr(output_shape.value()), output_dtype, vdevice);
    }
  }

  auto get_shape_expr = [](const Type& ty) -> ffi::Optional<Expr> {
    if (const auto* tensor = ty.as<TensorTypeObj>()) {
      return tensor->shape;
    } else {
      return std::nullopt;
    }
  };

  // If the input shape is unknown, but both inputs have the same
  // `ShapeType`variable for their shape, then propagate that
  // variable to the output.
  auto lhs_shape_expr = get_shape_expr(lhs_ty);
  auto rhs_shape_expr = get_shape_expr(rhs_ty);
  if (lhs_shape_expr.has_value() && lhs_shape_expr.same_as(rhs_shape_expr)) {
    return TensorType(lhs_shape_expr.value(), output_dtype, vdevice);
  }

  // If neither of those cases holds, then fall back to an unknown
  // shape with `output_ndim` dimensionality.
  return TensorType(output_dtype, output_ndim, vdevice);
}

Type InferTypeBroadcastArith(const Call& call, const BlockBuilder& ctx) {
  return InferTypeBroadcast(
      call, ctx, [](const Call& call, const BlockBuilder& ctx, const Type& lhs, const Type& rhs) {
        return InferBinaryArithOpOutDtype(call, ctx, lhs, rhs);
      });
}

Type InferTypeBroadcastCMP(const Call& call, const BlockBuilder& ctx) {
  return InferTypeBroadcast(call, ctx,
                            [](const Call& call, const BlockBuilder& ctx, const Type& lhs_ty,
                               const Type& rhs_ty) { return PrimType::Bool(); });
}

InferLayoutOutput InferLayoutBinaryEwise(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  LayoutDecision layout1 = GetLayoutDecision(var_layout_map, call->args[0]);
  LayoutDecision layout2 = GetLayoutDecision(var_layout_map, call->args[1]);

  auto* x1_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  auto* x2_ty = GetTypeAs<TensorTypeObj>(call->args[1]);

  TVM_FFI_ICHECK(!x1_ty->IsUnknownNdim() && !x2_ty->IsUnknownNdim())
      << "Unknown dim tensors should not be handled by this function";

  ffi::Optional<ShapeExpr> shape1 = x1_ty->shape.as<ShapeExpr>();
  ffi::Optional<ShapeExpr> shape2 = x2_ty->shape.as<ShapeExpr>();
  // Lets handle sub indexing as long as primal dims are matching
  if (((layout1->layout.has_value() ? layout1->layout.value().ndim() : 0) !=
       (layout1->layout.has_value() ? layout1->layout.value().ndim_primal() : 0)) ||
      ((layout2->layout.has_value() ? layout2->layout.value().ndim() : 0) !=
       (layout2->layout.has_value() ? layout2->layout.value().ndim_primal() : 0))) {
    if ((layout1->layout.has_value() ? layout1->layout.value().ndim_primal() : 0) ==
        (layout2->layout.has_value() ? layout2->layout.value().ndim_primal() : 0)) {
      if (((layout1->layout.has_value() ? layout1->layout.value().ndim() : 0) >=
           (layout2->layout.has_value() ? layout2->layout.value().ndim() : 0)) &&
          shape2.has_value()) {
        if (CanProveLayoutTransform(InitialLayout(shape2.value()->values.size()), layout1->layout,
                                    shape2.value()->values)) {
          return InferLayoutOutput({layout1, layout1}, {layout1}, call->attrs);
        }
      } else if (shape1.has_value()) {
        if (CanProveLayoutTransform(InitialLayout(shape1.value()->values.size()), layout2->layout,
                                    shape1.value()->values)) {
          return InferLayoutOutput({layout2, layout2}, {layout2}, call->attrs);
        }
      }
    }
  }

  if (x1_ty->ndim <= x2_ty->ndim) {
    if (x1_ty->ndim == 0) {
      LayoutDecision out_layout = layout2;
      return InferLayoutOutput({LayoutDecision(std::nullopt), layout2}, {out_layout}, call->attrs);
    }
    LayoutDecision out_layout = FollowDecision(layout1, x2_ty->ndim);
    return InferLayoutOutput({layout1, out_layout}, {out_layout}, call->attrs);
  } else {
    if (x2_ty->ndim == 0) {
      LayoutDecision out_layout = layout1;
      return InferLayoutOutput({layout1, LayoutDecision(std::nullopt)}, {out_layout}, call->attrs);
    }
    LayoutDecision out_layout = FollowDecision(layout2, x1_ty->ndim);

    return InferLayoutOutput({out_layout, layout2}, {out_layout}, call->attrs);
  }
}

/***************** Arithmetic operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(add);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(divide);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(floor_divide);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(log_add_exp);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(multiply);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(power);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(atan2);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(subtract);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(mod);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(floor_mod);

/***************** Comparison operators *****************/

RELAX_REGISTER_CMP_OP_AND_IMPL(equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(greater);
RELAX_REGISTER_CMP_OP_AND_IMPL(greater_equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(less);
RELAX_REGISTER_CMP_OP_AND_IMPL(less_equal);
RELAX_REGISTER_CMP_OP_AND_IMPL(not_equal);

/***************** Min/Max operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(minimum);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(maximum);

/***************** Logical operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(logical_and);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(logical_or);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(logical_xor);

/***************** Bitwise operators *****************/

RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(bitwise_and);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(bitwise_or);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(bitwise_xor);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(left_shift);
RELAX_REGISTER_BINARY_BROADCAST_OP_AND_IMPL(right_shift);

}  // namespace relax
}  // namespace tvm
