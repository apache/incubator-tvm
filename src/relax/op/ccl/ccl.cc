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

#include "ccl.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/op_attr.h>

#include <utility>

namespace tvm {
namespace relax {
using namespace tvm::prim;

/* relax.ccl.allreduce */

void AllReduceAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AllReduceAttrs>()
      .def_ro("op_type", &AllReduceAttrs::op_type,
              "The type of reduction operation to be applied to the input data. Now only sum is "
              "supported.",
              refl::DefaultValue(ffi::String("sum")))
      .def_ro("in_group", &AllReduceAttrs::in_group,
              "Whether the reduction operation performs in group or globally or in group as "
              "default.",
              refl::DefaultValue(true));
}

void AllGatherAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AllGatherAttrs>()
      .def_ro("num_workers", &AllGatherAttrs::num_workers,
              "The number of workers, also the number of parts the given buffer should be "
              "chunked into.")
      .def_ro("in_group", &AllGatherAttrs::in_group,
              "Whether the allgather operation performs in group or globally or in group as "
              "default.",
              refl::DefaultValue(true));
}

void ScatterCollectiveAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<ScatterCollectiveAttrs>()
      .def_ro("num_workers", &ScatterCollectiveAttrs::num_workers,
              "The number of workers, also the number of parts the given buffer should be "
              "chunked into.")
      .def_ro("axis", &ScatterCollectiveAttrs::axis,
              "The axis of the tensor to be scattered. The tensor will be chunked along "
              "this axis.",
              refl::DefaultValue(0));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  AllReduceAttrs::RegisterReflection();
  AllGatherAttrs::RegisterReflection();
  ScatterCollectiveAttrs::RegisterReflection();
}

Expr allreduce(Expr x, ffi::String op_type, bool in_group) {
  ffi::ObjectPtr<AllReduceAttrs> attrs = ffi::make_object<AllReduceAttrs>();
  attrs->op_type = std::move(op_type);
  attrs->in_group = std::move(in_group);

  static const Op op = Op::Get("relax.ccl.allreduce");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.ccl.allreduce", allreduce);
}

Type InferTypeAllReduce(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType input_ty = GetUnaryInputTensorType(call);
  return input_ty;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.ccl.allreduce")
      .signature(sig::arg("x", "Input to which allreduce will be applied."),
                 sig::call_attrs<AllReduceAttrs>())
      .set_native_fn_attr<&InferTypeAllReduce>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.ccl.allgather */

Expr allgather(Expr x, int num_workers, bool in_group) {
  ffi::ObjectPtr<AllGatherAttrs> attrs = ffi::make_object<AllGatherAttrs>();
  attrs->num_workers = std::move(num_workers);
  attrs->in_group = std::move(in_group);

  static const Op op = Op::Get("relax.ccl.allgather");
  return Call(Type::Missing(), op, {std::move(x)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.ccl.allgather", allgather);
}

Type InferTypeAllGather(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType input_ty = GetUnaryInputTensorType(call);

  const auto* attrs = call->attrs.as<AllGatherAttrs>();
  int num_workers = attrs->num_workers;

  ffi::Optional<PrimType> output_dtype = input_ty->dtype;
  auto input_shape = input_ty->GetShape();
  if (!input_shape.has_value()) {
    return input_ty;
  }
  ffi::Array<PrimExpr> output_shape = input_shape.value();
  output_shape.Set(0, floor(output_shape[0] * num_workers));
  return TensorType(ShapeExpr(output_shape), output_dtype, input_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.ccl.allgather")
      .signature(sig::arg("x", "Input to which allgather will be applied."),
                 sig::call_attrs<AllGatherAttrs>())
      .set_native_fn_attr<&InferTypeAllGather>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.ccl.broadcast_from_worker0 */
Expr broadcast_from_worker0(Expr x) {
  static const Op op = Op::Get("relax.ccl.broadcast_from_worker0");
  return Call(Type::Missing(), op, {std::move(x)}, {}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.ccl.broadcast_from_worker0", broadcast_from_worker0);
}

Type InferTypeBroadcastFromZero(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType input_ty = GetUnaryInputTensorType(call);
  return input_ty;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.ccl.broadcast_from_worker0")
      .signature(sig::arg("x", "Input to be broadcast."))
      .set_native_fn_attr<&InferTypeBroadcastFromZero>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kInferLayout, InferLayoutUnaryEwise)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.ccl.scatter_from_worker0 */

Expr scatter_from_worker0(Expr data, int num_workers, int axis) {
  ffi::ObjectPtr<ScatterCollectiveAttrs> attrs = ffi::make_object<ScatterCollectiveAttrs>();
  attrs->num_workers = std::move(num_workers);
  attrs->axis = std::move(axis);
  static const Op op = Op::Get("relax.ccl.scatter_from_worker0");

  return Call(Type::Missing(), op, {std::move(data)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.ccl.scatter_from_worker0", scatter_from_worker0);
}

Type InferTypeScatter(const Call& call, const BlockBuilder& ctx) {
  TensorType input_ty = GetUnaryInputTensorType(call, ctx);
  ffi::Optional<PrimType> output_dtype = input_ty->dtype;

  const auto* attrs = call->attrs.as<ScatterCollectiveAttrs>();
  int num_workers = attrs->num_workers;

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  auto input_shape = input_ty->GetShape();
  TVM_FFI_ICHECK(input_shape.has_value())
      << "input tensor of scatter_from_worker0 should have defined shape.";

  if (analyzer->CanProve(floormod(input_shape.value()[attrs->axis], PrimExpr(num_workers)) != 0)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "scatter_from_worker0 expects the size of axis " << attrs->axis
        << " of input tensor to be divisible by the num_workers. However, axis " << attrs->axis
        << " of input tensor is " << input_shape.value() << " while num_workers is " << num_workers;
  }

  ffi::Array<PrimExpr> output_shape = input_shape.value();
  output_shape.Set(attrs->axis, div(output_shape[attrs->axis], num_workers));
  return TensorType(ShapeExpr(output_shape), output_dtype, input_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.ccl.scatter_from_worker0")
      .signature(
          sig::arg(
              "x",
              "The buffer to be divided into equal parts and sent to each worker accordingly."),
          sig::call_attrs<ScatterCollectiveAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeScatter)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
