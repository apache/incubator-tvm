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
 * \file unary.cc
 * \brief Relax unary arithmetic operators.
 */

#include "unary.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <utility>

namespace tvm {
namespace relax {

Type InferTypeUnaryCheck(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  return InferTypeUnary<false>(call, [](const TensorType& input_ty) { return PrimType::Bool(); });
}

/***************** Arithmetic operators *****************/

Expr abs(Expr x) {
  static const Op op = Op::Get("relax.abs");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr acos(Expr x) {
  static const Op op = Op::Get("relax.acos");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr acosh(Expr x) {
  static const Op op = Op::Get("relax.acosh");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr asin(Expr x) {
  static const Op op = Op::Get("relax.asin");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr asinh(Expr x) {
  static const Op op = Op::Get("relax.asinh");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr atan(Expr x) {
  static const Op op = Op::Get("relax.atan");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr atanh(Expr x) {
  static const Op op = Op::Get("relax.atanh");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr bitwise_not(Expr x) {
  static const Op op = Op::Get("relax.bitwise_not");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr ceil(Expr x) {
  static const Op op = Op::Get("relax.ceil");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr cos(Expr x) {
  static const Op op = Op::Get("relax.cos");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr cosh(Expr x) {
  static const Op op = Op::Get("relax.cosh");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr exp(Expr x) {
  static const Op op = Op::Get("relax.exp");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr floor(Expr x) {
  static const Op op = Op::Get("relax.floor");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr log(Expr x) {
  static const Op op = Op::Get("relax.log");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr logical_not(Expr x) {
  static const Op op = Op::Get("relax.logical_not");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr negative(Expr x) {
  static const Op op = Op::Get("relax.negative");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr round(Expr x) {
  static const Op op = Op::Get("relax.round");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr rsqrt(Expr x) {
  static const Op op = Op::Get("relax.rsqrt");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr sigmoid(Expr x) {
  static const Op op = Op::Get("relax.sigmoid");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr sign(Expr x) {
  static const Op op = Op::Get("relax.sign");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr sin(Expr x) {
  static const Op op = Op::Get("relax.sin");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr sinh(Expr x) {
  static const Op op = Op::Get("relax.sinh");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr square(Expr x) {
  static const Op op = Op::Get("relax.square");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr sqrt(Expr x) {
  static const Op op = Op::Get("relax.sqrt");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr tan(Expr x) {
  static const Op op = Op::Get("relax.tan");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr tanh(Expr x) {
  static const Op op = Op::Get("relax.tanh");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr trunc(Expr x) {
  static const Op op = Op::Get("relax.trunc");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr erf(Expr x) {
  static const Op op = Op::Get("relax.erf");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr isfinite(Expr x) {
  static const Op op = Op::Get("relax.isfinite");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr isinf(Expr x) {
  static const Op op = Op::Get("relax.isinf");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

Expr isnan(Expr x) {
  static const Op op = Op::Get("relax.isnan");
  return Call(Type::Missing(), op, {std::move(x)}, std::nullopt, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("relax.op.abs", abs);

  OpDef("relax.abs")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.acos", acos);

  OpDef("relax.acos")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.acosh", acosh);

  OpDef("relax.acosh")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.asin", asin);

  OpDef("relax.asin")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.asinh", asinh);

  OpDef("relax.asinh")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.atan", atan);

  OpDef("relax.atan")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.atanh", atanh);

  OpDef("relax.atanh")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.bitwise_not", bitwise_not);

  OpDef("relax.bitwise_not")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.ceil", ceil);

  OpDef("relax.ceil")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.cos", cos);

  OpDef("relax.cos")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.cosh", cosh);

  OpDef("relax.cosh")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.exp", exp);

  OpDef("relax.exp")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.floor", floor);

  OpDef("relax.floor")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.log", log);

  OpDef("relax.log")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.logical_not", logical_not);

  OpDef("relax.logical_not")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.negative", negative);

  OpDef("relax.negative")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.round", round);

  OpDef("relax.round")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.rsqrt", rsqrt);

  OpDef("relax.rsqrt")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.sigmoid", sigmoid);

  OpDef("relax.sigmoid")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.sign", sign);

  OpDef("relax.sign")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.sin", sin);

  OpDef("relax.sin")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.sinh", sinh);

  OpDef("relax.sinh")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.square", square);

  OpDef("relax.square")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.sqrt", sqrt);

  OpDef("relax.sqrt")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.tan", tan);

  OpDef("relax.tan")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.tanh", tanh);

  OpDef("relax.tanh")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.trunc", trunc);

  OpDef("relax.trunc")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<false>>(tvm::op_attr::kInferType);

  tvm::ffi::reflection::GlobalDef().def("relax.op.erf", erf);

  OpDef("relax.erf")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryArithContextFree<true>>(tvm::op_attr::kInferType);

  // relax.clip

  OpDef("relax.clip")
      .signature(sig::arg("x", "The input tensor."),
                 sig::arg("min", "The lower-bound of the range to be clipped to"),
                 sig::arg("max", "The upper-bound of the range to be clipped to"))
      .set_native_fn_attr<&ReturnTypeFromArgContextFree<0>>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

Expr clip(Expr x, Expr min, Expr max) {
  TVM_FFI_ICHECK(min.as<PrimExpr>())
      << "The argument `min` of relax.clip is expected to be a PrimExpr, but got "
      << min->GetTypeKey();
  TVM_FFI_ICHECK(max.as<PrimExpr>())
      << "The argument `max` of relax.clip is expected to be a PrimExpr, but got "
      << max->GetTypeKey();
  static const Op op = Op::Get("relax.clip");
  return Call(Type::Missing(), op, {std::move(x), std::move(min), std::move(max)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.clip", clip);

  /***************** Check operators *****************/

  tvm::ffi::reflection::GlobalDef().def("relax.op.isfinite", isfinite);

  OpDef("relax.isfinite")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryCheck>(
          tvm::op_attr::kInferType);  // require_float_dtype=false
                                      // for check op

  tvm::ffi::reflection::GlobalDef().def("relax.op.isinf", isinf);

  OpDef("relax.isinf")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryCheck>(
          tvm::op_attr::kInferType);  // require_float_dtype=false
                                      // for check op

  tvm::ffi::reflection::GlobalDef().def("relax.op.isnan", isnan);

  OpDef("relax.isnan")
      .signature(
          sig::arg("x", "The input tensor."),
          sig::var_ty_args("out_type", "Optional output tensor type carrying the virtual device."))
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_native_fn_attr<&InferTypeUnaryCheck>(
          tvm::op_attr::kInferType);  // require_float_dtype=false
                                      // for check op
}

}  // namespace relax
}  // namespace tvm
