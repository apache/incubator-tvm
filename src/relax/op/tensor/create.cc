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
 * \file create.cc
 * \brief Creation operators.
 */

#include "create.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/sym/analyzer.h>

#include <string>
#include <utility>

#include "tvm/relax/expr.h"

namespace tvm {
namespace relax {

void InitAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<InitAttrs>().def_ro("dtype", &InitAttrs::dtype,
                                      "The data type of the created tensor.",
                                      refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void TriluAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<TriluAttrs>().def_ro(
      "k", &TriluAttrs::k,
      "The number of diagonals above or below the main diagonal to exclude or include.");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  InitAttrs::RegisterReflection();
  TriluAttrs::RegisterReflection();
}

/* Initialization operators */

/* relax.full */
// Prefer shape arrays to the generic Expr conversion of sequences to tuples.
Expr full(ffi::Variant<ffi::Array<PrimExpr>, Expr> shape, Expr fill_value,
          ffi::Optional<DLDataType> dtype) {
  Expr shape_in_expr = [&]() -> Expr {
    if (const auto* expr = shape.as<ExprObj>()) {
      return ffi::GetRef<Expr>(expr);
    }
    return ShapeExpr(shape.get<ffi::Array<PrimExpr>>());
  }();

  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op op = Op::Get("relax.full");
  return Call(Type::Missing(), op, {std::move(shape_in_expr), std::move(fill_value)}, Attrs(attrs),
              {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.full", full);
}

Type InferTypeFull(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Full op should have 2 arguments";
  }
  const auto* shape_ty = GetTypeAs<ShapeTypeObj>(call->args[0]);
  const auto* fill_value_ty = GetTypeAs<TensorTypeObj>(call->args[1]);
  if (shape_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Full requires the input shape to be a Shape. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  if (fill_value_ty == nullptr || fill_value_ty->ndim != 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Full requires the input fill value to be zero rank Tensor. However, the given one is "
        << call->args[1]->ty;
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  ffi::Optional<PrimType> out_dtype = attrs->dtype.has_value()
                                          ? ffi::Optional<PrimType>(PrimType(attrs->dtype.value()))
                                          : fill_value_ty->dtype;
  return TensorType(/*shape=*/call->args[0], out_dtype, fill_value_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.full")
      .signature(sig::arg("shape", "The shape of the created tensor."),
                 sig::arg("fill_value", "The scalar tensor, denoting the value to fill."),
                 sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeFull>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kDataDependent, true)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.full_like */
Expr full_like(Expr x, Expr fill_value, ffi::Optional<DLDataType> dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op op = Op::Get("relax.full_like");
  return Call(Type::Missing(), op, {std::move(x), std::move(fill_value)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.full_like", full_like);
}

Type InferTypeFullLike(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  ffi::Array<TensorType> input_ty = GetInputTensorType(call);
  TensorType data_ty = input_ty[0];
  TensorType fill_value_ty = input_ty[1];
  if (fill_value_ty->ndim != 0) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "FullLike requires the input fill value to be zero "
                                             "rank Tensor. However, the given one has ndim"
                                          << fill_value_ty->ndim;
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  if (!attrs->dtype.has_value()) {
    return data_ty;
  } else {
    auto output_ty = ffi::make_object<TensorTypeObj>(*data_ty.get());
    output_ty->dtype = PrimType(attrs->dtype.value());
    return TensorType(output_ty);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.full_like")
      .signature(sig::arg("x", "The input tensor."),
                 sig::arg("fill_value", "The scalar value to fill."), sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeFullLike>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

// Structure info inference for ones and zeros
Type InferTypeOnesZeros(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Ones/Zeros should have 1 argument";
  }

  const auto* shape_ty = GetTypeAs<ShapeTypeObj>(call->args[0]);
  if (shape_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Ones/Zeros requires the input shape to be a Shape. However, the given one is "
        << call->args[0]->ty->GetTypeKey();
  }
  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs->dtype.has_value());
  return TensorType(/*shape=*/call->args[0], PrimType(attrs->dtype.value()));
}

// Structure info inference for ones_like and zeros_like
Type InferTypeOnesLikeZerosLike(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<InitAttrs>();
  if (!attrs->dtype.has_value()) {
    return data_ty;
  } else {
    auto output_ty = ffi::make_object<TensorTypeObj>(*data_ty.get());
    output_ty->dtype = PrimType(attrs->dtype.value());
    return TensorType(output_ty);
  }
}

/* relax.ones & relax.ones_like */
Expr ones(Expr shape, DLDataType dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op op = Op::Get("relax.ones");
  return Call(Type::Missing(), op, {std::move(shape)}, Attrs(attrs), {});
}

Expr ones_like(Expr x, ffi::Optional<DLDataType> dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op op = Op::Get("relax.ones_like");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.ones", ones).def("relax.op.ones_like", ones_like);

  OpDef("relax.ones")
      .signature(sig::arg("shape", "The shape of the created tensor."),
                 sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeOnesZeros>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  OpDef("relax.ones_like")
      .signature(sig::arg("x", "The input tensor."), sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeOnesLikeZerosLike>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.zeros & relax.zeros_like */
Expr zeros(Expr shape, DLDataType dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;

  static const Op op = Op::Get("relax.zeros");
  return Call(Type::Missing(), op, {std::move(shape)}, Attrs(attrs), {});
}

Expr zeros_like(Expr x, ffi::Optional<DLDataType> dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op op = Op::Get("relax.zeros_like");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.zeros", zeros).def("relax.op.zeros_like", zeros_like);

  OpDef("relax.zeros")
      .signature(sig::arg("shape", "The shape of the created tensor."),
                 sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeOnesZeros>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  OpDef("relax.zeros_like")
      .signature(sig::arg("x", "The input tensor."), sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeOnesLikeZerosLike>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.eye & relax.eye_like */
Expr eye(PrimExpr n, PrimExpr m, PrimExpr k, DLDataType dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op op = Op::Get("relax.eye");
  return Call(Type::Missing(), op, {std::move(n), std::move(m), std::move(k)}, Attrs(attrs), {});
}

Expr eye_like(Expr x, PrimExpr k, ffi::Optional<DLDataType> dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op op = Op::Get("relax.eye_like");
  return Call(Type::Missing(), op, {std::move(x), std::move(k)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.eye", eye).def("relax.op.eye_like", eye_like);
}

Type InferTypeEye(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Eye op should have 3 arguments: n, m, and k, but got "
                                          << call->args.size() << " arguments";
  }

  auto get_prim_value = [](const Expr& expr, std::string key) {
    auto prim_value = expr.as<PrimExpr>();
    if (!prim_value) {
      TVM_FFI_VISIT_THROW(TypeError, expr)
          << "Eye expects the `" << key << "` to be a PrimExpr, but got " << expr->GetTypeKey();
    }
    return prim_value.value();
  };

  PrimExpr n = get_prim_value(call->args[0], "n");
  PrimExpr m = get_prim_value(call->args[1], "m");

  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs->dtype.has_value());
  DLDataType dtype = attrs->dtype.value();
  return TensorType(ShapeExpr({n, m}), PrimType(dtype));
}

Type InferTypeEyeLike(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Eye_like op should have 2 arguments: x and k, but got " << call->args.size()
        << " arguments";
  }

  const auto* x_ty = GetTypeAs<TensorTypeObj>(call->args[0]);
  if (x_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Eye_like expects the input `x` to be a Tensor, but got "
        << call->args[0]->ty->GetTypeKey();
  }
  if (x_ty->ndim != 2 && x_ty->ndim != kUnknownNDim) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Eye_like expects the input tensor to be 2-dimensional, but got " << x_ty->ndim
        << " dimensions";
  }

  const auto* attrs = call->attrs.as<InitAttrs>();
  ffi::Optional<PrimType> out_dtype = attrs->dtype.has_value()
                                          ? ffi::Optional<PrimType>(PrimType(attrs->dtype.value()))
                                          : x_ty->dtype;

  return TensorType(x_ty->shape.value(), out_dtype, x_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.eye")
      .signature(sig::arg("n", "Number of rows in the output."),
                 sig::arg("m", "Number of columns in the output."),
                 sig::arg("k", "Index of the diagonal."), sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeEye>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  OpDef("relax.eye_like")
      .signature(sig::arg("x", "The input tensor."), sig::arg("k", "Index of the diagonal."),
                 sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeEyeLike>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.arange */
Expr arange(PrimExpr start, PrimExpr stop, PrimExpr step, DLDataType dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op op = Op::Get("relax.arange");
  return Call(Type::Missing(), op, {std::move(start), std::move(stop), std::move(step)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.arange", arange);
}

Type InferTypeArange(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 3) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Arange should have 3 arguments, which are `start`, `end` and `step`, but got "
        << call->args.size() << " arguments";
  }
  // TODO(Siyuan): Support indirect prim_values
  auto get_prim_value = [](const Expr& expr, std::string key) {
    auto prim_value = expr.as<PrimExpr>();
    if (!prim_value) {
      TVM_FFI_VISIT_THROW(TypeError, expr)
          << "Arange expects the `" << key << "` to be a PrimExpr, but got " << expr->GetTypeKey();
    }
    return prim_value.value();
  };
  PrimExpr start = get_prim_value(call->args[0], "start");
  PrimExpr end = get_prim_value(call->args[1], "end");
  PrimExpr step = get_prim_value(call->args[2], "step");
  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs->dtype.has_value());
  DLDataType dtype = attrs->dtype.value();
  PrimExpr num_elem = [&]() -> PrimExpr {
    if (start.ty().code() == DLDataTypeCode::kDLInt && end.ty().code() == DLDataTypeCode::kDLInt &&
        step.ty().code() == DLDataTypeCode::kDLInt) {
      return tvm::floordiv((end - start + step - 1), step);
    } else {
      return tvm::prim::cast(
          tvm::prim::ceil(tvm::prim::cast(end - start, tvm::PrimType::Float(32)) / step),
          tvm::PrimType::Int(64));
    }
  }();
  sym::Analyzer analyzer;
  num_elem = analyzer->Simplify(num_elem);
  return TensorType(ShapeExpr({num_elem}), PrimType(dtype));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.arange")
      .signature(sig::arg("start", "The starting value for the set of points."),
                 sig::arg("end", "The ending value for the set of points."),
                 sig::arg("step", "The gap between each pair of adjacent points."),
                 sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeArange>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.hamming_window */
Expr hamming_window(PrimExpr window_size, PrimExpr periodic, PrimExpr alpha, PrimExpr beta,
                    DLDataType dtype) {
  ffi::ObjectPtr<InitAttrs> attrs = ffi::make_object<InitAttrs>();
  attrs->dtype = dtype;
  static const Op op = Op::Get("relax.hamming_window");
  return Call(Type::Missing(), op,
              {std::move(window_size), std::move(periodic), std::move(alpha), std::move(beta)},
              Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.hamming_window", hamming_window);
}

Type InferTypeHammingWindow(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  const auto* attrs = call->attrs.as<InitAttrs>();
  TVM_FFI_ICHECK(attrs->dtype.has_value());
  DLDataType dtype = attrs->dtype.value();
  if (dtype.code == DLDataTypeCode::kDLInt || dtype.code == DLDataTypeCode::kDLUInt) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Hamming Window expects the datatype to be float but got " << dtype;
  }
  auto get_prim_value = [](const Expr& expr, std::string key) {
    auto prim_value = expr.as<PrimExpr>();
    if (!prim_value) {
      TVM_FFI_VISIT_THROW(TypeError, expr) << "Hamming_window expects the `" << key
                                           << "` to be a PrimExpr, but got " << expr->GetTypeKey();
    }
    return prim_value.value();
  };
  PrimExpr window_size = get_prim_value(call->args[0], "window_size");

  sym::Analyzer analyzer;
  if (analyzer->CanProveLess(window_size, 1)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Hamming_window expects the window_size must be greater than zero but got "
        << window_size;
  }
  window_size = analyzer->Simplify(window_size);
  return TensorType(ShapeExpr({window_size}), PrimType(dtype));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.hamming_window")
      .signature(
          sig::arg<IntExpr>("window_size", "The size of the window"),
          sig::arg("periodic",
                   "If True, returns a window to be used as periodic function. If False, return a "
                   "symmetric window"),
          sig::arg("alpha", "The coefficient alpha"), sig::arg("beta", "The coefficient beta"),
          sig::call_attrs<InitAttrs>())
      .set_native_fn_attr<&InferTypeHammingWindow>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.tril & relax.triu */

Expr tril(Expr x, Expr k) {
  static const Op op = Op::Get("relax.tril");
  return Call(Type::Missing(), op, {x, k});
}

Expr triu(Expr x, Expr k) {
  static const Op op = Op::Get("relax.triu");
  return Call(Type::Missing(), op, {x, k});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.op.tril", static_cast<Expr (*)(Expr, Expr)>(tril))
      .def("relax.op.triu", static_cast<Expr (*)(Expr, Expr)>(triu));
}

Type InferTypeTrilTriu(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  auto [data_ty, offset] = GetArgType<TensorType, PrimType>(call);

  if (!data_ty->IsUnknownNdim() && data_ty->ndim < 2) {
    TVM_FFI_VISIT_THROW(ValueError, call) << call->op
                                          << " requires the input tensor to have at least two "
                                             "dimensions. However, the given input has "
                                          << data_ty->ndim << " dimension(s).";
  }
  return data_ty;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.tril")
      .signature(sig::arg("x", "The input tensor."),
                 sig::arg<IntExpr>("k", "The offset of the diagonal."))
      .set_native_fn_attr<&InferTypeTrilTriu>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  OpDef("relax.triu")
      .signature(sig::arg("x", "The input tensor."),
                 sig::arg<IntExpr>("k", "The offset of the diagonal."))
      .set_native_fn_attr<&InferTypeTrilTriu>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
