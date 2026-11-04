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
 * \file search.cc
 * \brief Searching operators.
 */

#include "search.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <algorithm>
#include <utility>

namespace tvm {
namespace relax {

void ArgmaxArgminAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ArgmaxArgminAttrs>()
      .def_ro("axis", &ArgmaxArgminAttrs::axis,
              "The axis along which to perform the argmin/argmax.",
              refl::DefaultValue(ffi::Optional<int64_t>{}))
      .def_ro("keepdims", &ArgmaxArgminAttrs::keepdims,
              "If this is set to `True`, the reduced axis is left in the result as dimension "
              "with size "
              "one.",
              refl::DefaultValue(false));
}

void BucketizeAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<BucketizeAttrs>()
      .def_ro("out_int32", &BucketizeAttrs::out_int32,
              "Indicate the output datatype, int32 if True, int64 otherwise.",
              refl::DefaultValue(false))
      .def_ro("right", &BucketizeAttrs::right, "Determines the behavior for values in boundaries",
              refl::DefaultValue(false));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ArgmaxArgminAttrs::RegisterReflection();
  BucketizeAttrs::RegisterReflection();
}

/* relax.bucketize */

Expr bucketize(Expr input_tensor, Expr boundaries, bool out_int32, bool right) {
  auto attrs = ffi::make_object<BucketizeAttrs>();
  attrs->out_int32 = std::move(out_int32);
  attrs->right = std::move(right);
  static const Op op = Op::Get("relax.bucketize");
  return Call(Type::Missing(), op, {std::move(input_tensor), std::move(boundaries)}, Attrs(attrs),
              {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.bucketize", bucketize);
}

Type InferTypeBucketize(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  ffi::Array<TensorType> input_ty = GetInputTensorType(call);
  TensorType input_tensor_info = input_ty[0];
  TensorType boundaries_info = input_ty[1];

  if (!boundaries_info->IsUnknownNdim() && boundaries_info->ndim != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Bucketize requires boundary to be 1-D array but got " << boundaries_info->ndim;
  }

  auto attrs = call->attrs.as<BucketizeAttrs>();
  PrimType out_dtype = PrimType::Int(64);
  if (attrs->out_int32) {
    out_dtype = PrimType::Int(32);
  }

  const auto* data_shape = input_tensor_info->shape.as<ShapeExprObj>();
  if (data_shape) {
    return TensorType(ShapeExpr(data_shape->values), out_dtype, input_tensor_info->vdevice);
  }
  return TensorType(out_dtype, input_tensor_info->ndim, input_tensor_info->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.bucketize")
      .signature(
          sig::arg("input_tensor", " N-D tensor or a Scalar containing the search value(s)."),
          sig::arg(
              "boundaries",
              "1-D tensor, must contain a strictly increasing sequence, or the return value is "
              "undefined."),
          sig::call_attrs<BucketizeAttrs>())
      .set_native_fn_attr<&InferTypeBucketize>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.where */
Expr where(Expr condition, Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.where");
  return Call(Type::Missing(), op, {std::move(condition), std::move(x1), std::move(x2)},
              std::nullopt, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.where", where);
}

Type InferTypeWhere(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType cond_ty = input_ty[0];
  TensorType x1_ty = input_ty[1];
  TensorType x2_ty = input_ty[2];

  ffi::Optional<VDevice> vdev;
  for (int i = 0; i < 3; ++i) {
    if (input_ty[i]->vdevice.has_value()) {
      if (!vdev.has_value()) {
        vdev = input_ty[i]->vdevice.value();
      } else if (input_ty[i]->vdevice.value()->target.has_value()) {
        // mismatch
        if (input_ty[i]->vdevice.value() != vdev.value()) {
          vdev = std::nullopt;
          break;
        }
      }
    }
  }

  // Where condition validation only checks the boolean element kind; lanes are irrelevant here.
  if (!cond_ty->IsUnknownDtype() && !cond_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLBool)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Where requires the input condition tensor to have boolean dtype. However, "
           "the given condition dtype is "
        << cond_ty->dtype;
  }
  ffi::Optional<PrimType> output_dtype = InferBinaryArithOpOutDtype(call, ctx, x1_ty, x2_ty);

  int output_ndim;
  if (cond_ty->IsUnknownNdim() || x1_ty->IsUnknownNdim() || x2_ty->IsUnknownNdim()) {
    output_ndim = kUnknownNDim;
  } else {
    output_ndim = std::max(cond_ty->ndim, std::max(x1_ty->ndim, x2_ty->ndim));
  }

  const auto* cond_shape = cond_ty->shape.as<ShapeExprObj>();
  const auto* x1_shape = x1_ty->shape.as<ShapeExprObj>();
  const auto* x2_shape = x2_ty->shape.as<ShapeExprObj>();
  if (cond_shape && x1_shape && x2_shape) {
    // Step 1. Compute the broadcasted shape of x1's and x2's
    ffi::Optional<ffi::Array<PrimExpr>> broadcasted_shape =
        InferBinaryBroadcastShape(call, ctx, x1_shape->values, x2_shape->values);
    if (!broadcasted_shape.has_value()) {
      if (vdev.has_value()) {
        return TensorType(output_dtype, output_ndim, vdev);
      }
      return TensorType(output_dtype, output_ndim);
    }
    // Step 2. Compute the broadcasted shape of cond's and the previous broadcasted shape.
    broadcasted_shape =
        InferBinaryBroadcastShape(call, ctx, cond_shape->values, broadcasted_shape.value());
    if (!broadcasted_shape.has_value()) {
      if (vdev.has_value()) {
        return TensorType(output_dtype, output_ndim, vdev);
      }
      return TensorType(output_dtype, output_ndim);
    }
    TVM_FFI_ICHECK_EQ(static_cast<int>(broadcasted_shape.value().size()), output_ndim);
    if (vdev.has_value()) {
      return TensorType(ShapeExpr(broadcasted_shape.value()), output_dtype, vdev);
    }
    return TensorType(ShapeExpr(broadcasted_shape.value()), output_dtype);
  } else if (cond_ty->shape.has_value() &&            //
             x1_ty->shape.has_value() &&              //
             x2_ty->shape.has_value() &&              //
             cond_ty->shape.same_as(x1_ty->shape) &&  //
             cond_ty->shape.same_as(x2_ty->shape)) {
    if (vdev.has_value()) {
      return TensorType(cond_ty->shape.value(), output_dtype, vdev);
    }
    return TensorType(cond_ty->shape.value(), output_dtype);
  } else {
    if (vdev.has_value()) {
      return TensorType(output_dtype, output_ndim, vdev);
    }
    return TensorType(output_dtype, output_ndim);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.where")
      .signature(sig::arg("condition", "When True, yield `x1`; otherwise, yield `x2`."),
                 sig::arg("x1", "The first input tensor."),
                 sig::arg("x2", "The second input tensor."))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeWhere)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.argmax & relax.argmin */

Type InferTypeArgmaxArgmin(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<ArgmaxArgminAttrs>();

  int axis = -1;
  if (!data_ty->IsUnknownNdim() && attrs->axis.has_value()) {
    axis = NormalizeAxis(call, data_ty->ndim, attrs->axis.value());
  }

  int out_ndim;
  if (attrs->keepdims) {
    out_ndim = data_ty->ndim;
  } else if (!attrs->axis.has_value()) {
    out_ndim = 0;
  } else if (data_ty->IsUnknownNdim()) {
    out_ndim = kUnknownNDim;
  } else {
    out_ndim = data_ty->ndim - 1;
    TVM_FFI_ICHECK_GE(out_ndim, 0);
  }

  PrimType out_dtype = PrimType::Int(64);
  // The inference rule for reduction operator output shapes:
  // - axes is None, keepdims is false -> return the zero-rank shape;
  // - axes is None, keepdims is true -> return the shape whose ndim is the same as input and every
  // value is 1.
  // - axes is not None, keepdims is false -> the returned shape does not contain the input axes.
  // - axes is not None, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axes
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();
  if (data_shape == nullptr) {
    if (!attrs->axis.has_value() && attrs->keepdims && out_ndim != kUnknownNDim) {
      return TensorType(ShapeExpr(ffi::Array<PrimExpr>(out_ndim, IntImm(out_dtype, /*value=*/1))),
                        out_dtype, data_ty->vdevice);
    } else {
      return out_ndim == 0
                 ? TensorType(ShapeExpr(ffi::Array<PrimExpr>()), out_dtype, data_ty->vdevice)
                 : TensorType(out_dtype, out_ndim, data_ty->vdevice);
    }
  }

  if (data_ty->ndim > 0) {
    out_dtype = data_shape->values[0].ty();
  }

  ffi::Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_ty->ndim; ++i) {
    if (attrs->axis.has_value() && i != axis) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm(out_dtype, /*value=*/1));
    }
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);
  return TensorType(ShapeExpr(out_shape), out_dtype, data_ty->vdevice);
}

Expr argmax(Expr x, ffi::Optional<int64_t> axis, bool keepdims) {
  ffi::ObjectPtr<ArgmaxArgminAttrs> attrs = ffi::make_object<ArgmaxArgminAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = std::move(keepdims);
  static const Op op = Op::Get("relax.argmax");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs(attrs));
}

Expr argmin(Expr x, ffi::Optional<int64_t> axis, bool keepdims) {
  ffi::ObjectPtr<ArgmaxArgminAttrs> attrs = ffi::make_object<ArgmaxArgminAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = std::move(keepdims);
  static const Op op = Op::Get("relax.argmin");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs(attrs));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("relax.op.argmax", argmax);

  OpDef("relax.argmax")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<ArgmaxArgminAttrs>())
      .set_native_fn_attr<&InferTypeArgmaxArgmin>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
};

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("relax.op.argmin", argmin);

  OpDef("relax.argmin")
      .signature(sig::arg("x", "The input data tensor"), sig::call_attrs<ArgmaxArgminAttrs>())
      .set_native_fn_attr<&InferTypeArgmaxArgmin>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
};

}  // namespace relax
}  // namespace tvm
