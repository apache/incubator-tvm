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
 * \file datatype.cc
 * \brief Datatype operators.
 */

#include "datatype.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <utility>

namespace tvm {
namespace relax {

void AstypeAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AstypeAttrs>().def_ro("dtype", &AstypeAttrs::dtype, "Target data type");
}

void WrapParamAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<WrapParamAttrs>().def_ro("dtype", &WrapParamAttrs::dtype, "Target data type",
                                           refl::DefaultValue((DLDataType{kDLFloat, 32, 1})));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  AstypeAttrs::RegisterReflection();
  WrapParamAttrs::RegisterReflection();
}

/* relax.astype */

Expr astype(Expr x, DLDataType dtype) {
  ffi::ObjectPtr<AstypeAttrs> attrs = ffi::make_object<AstypeAttrs>();
  attrs->dtype = dtype;

  static const Op op = Op::Get("relax.astype");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.astype", astype);
}

Type InferTypeAstype(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<AstypeAttrs>();
  ffi::ObjectPtr<TensorTypeObj> new_ty = ffi::make_object<TensorTypeObj>(*ty.get());
  new_ty->dtype = PrimType(attrs->dtype);
  return TensorType(new_ty);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.astype")
      .signature(sig::arg("x", "The input tensor"), sig::call_attrs<AstypeAttrs>())
      .set_native_fn_attr<&InferTypeAstype>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kFollow)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.wrap_param */

Expr MakeWrapParam(Expr data, DLDataType dtype) {
  ffi::ObjectPtr<WrapParamAttrs> attrs = ffi::make_object<WrapParamAttrs>();
  attrs->dtype = dtype;

  static const Op op = Op::Get("relax.wrap_param");
  return Call(Type::Missing(), op, {std::move(data)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.wrap_param", MakeWrapParam);
}

Type InferTypeWrapParam(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType ty = GetUnaryInputTensorType(call);
  const auto* attrs = call->attrs.as<WrapParamAttrs>();
  ffi::ObjectPtr<TensorTypeObj> new_ty = ffi::make_object<TensorTypeObj>(*ty.get());
  new_ty->dtype = PrimType(attrs->dtype);
  return TensorType(new_ty);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.wrap_param")
      .signature(sig::arg("data", "The input tensor"), sig::call_attrs<WrapParamAttrs>())
      .set_native_fn_attr<&InferTypeWrapParam>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
