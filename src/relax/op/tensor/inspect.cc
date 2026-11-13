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
 * \file inspect.cc
 * \brief Operators to access runtime DLTensor parameters
 */

#include "inspect.h"

#include <tvm/ffi/cast.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/s_tir/function.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>

#include <tuple>

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace inspect {

TensorType GetTensorArgInfo(const Call& call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 1, TypeError)
      << "Operator " << call->op << " expects one argument, "
      << "but received " << call->args.size() << " arguments: " << call->args;

  const auto& arg = call->args[0];
  auto ty = GetType(arg);

  auto tensor_ty = ty.as<TensorType>();
  TVM_FFI_CHECK(tensor_ty, TypeError) << "Operator " << call->op << " expects a tensor argument, "
                                      << "but argument " << arg << " has type " << ty;

  return tensor_ty.value();
}

std::tuple<TensorType, ffi::Optional<int64_t>> GetTensorArgInfoWithIndex(const Call& call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 2, TypeError)
      << "Operator " << call->op << " expects two arguments, "
      << "but received " << call->args.size() << " arguments: " << call->args;
  const auto& arg = call->args[0];
  const auto& axis = call->args[1];

  auto tensor_ty = arg->ty.as<TensorTypeObj>();
  TVM_FFI_CHECK(tensor_ty, TypeError)
      << "Operator " << call->op << " expects arguments (tensor, axis), "
      << "but the first argument " << arg << " in expression " << call << " has type " << arg->ty;

  auto axis_ty = axis->ty.as<PrimTypeObj>();
  TVM_FFI_CHECK(axis_ty, TypeError)
      << "Operator " << call->op << " expects arguments (tensor, axis), "
      << "but the second argument " << arg << " in expression " << call << " has type " << axis->ty;

  ffi::Optional<int64_t> int_imm_axis = std::nullopt;
  if (auto prim_value = axis.as<PrimExpr>()) {
    if (const auto* int_imm = prim_value->as<IntImmObj>()) {
      int_imm_axis = static_cast<int64_t>(int_imm->value);
    }
  }

  if (int_imm_axis) {
    TVM_FFI_ICHECK_GE(int_imm_axis.value(), 0);
  }
  if (int_imm_axis && !tensor_ty->IsUnknownNdim()) {
    TVM_FFI_CHECK_LT(int_imm_axis.value(), tensor_ty->ndim, ValueError)
        << "Expression " << call << " attempts to access " << arg << ".shape["
        << int_imm_axis.value() << "]"
        << ", but " << arg << ".shape only has " << tensor_ty->ndim << " elements";
  }

  return {ffi::GetRef<TensorType>(tensor_ty), int_imm_axis};
}

tirx::Function GetDLTensorField(tirx::TVMStructFieldKind field, PrimType field_ty) {
  tvm::Var dlpack_handle("dlpack_handle", PtrType::VoidPtrType());

  tvm::Var value("value", field_ty);

  tvm::SeqStmt body(
      {tvm::Bind(value, tvm::Call(field_ty, tirx::abi_field_get_op(),
                                  {dlpack_handle, IntImm::Int32(0), IntImm::Int32(field)})
                            .as_or_throw<PrimExpr>()),
       tvm::Return(value)});

  DictAttrs attrs(
      {{tvm::s_tir::attr::kIsScheduled.name, true}, {tvm::tirx::attr::kIsHostFunc.name, true}});

  tirx::Function func(ffi::Array<tvm::Var>{dlpack_handle}, body, field_ty, attrs);

  return func;
}

Expr NormalizeToKnownPrimExpr(const BlockBuilder&, Call call) { return call; }

//// relax.tensor_dtype_code

Expr tensor_dtype_code(Expr expr) {
  static const Op op = Op::Get("relax.inspect.tensor_dtype_code");
  return Call(Type::Missing(), op, {expr});
}

Type InferTypeTensorDtypeCode(const Call& call, const BlockBuilder&) { return PrimType::UInt(8); }

Expr LegalizeTensorDtypeCode(const BlockBuilder& bb, const Call& call) {
  PrimType field_ty = call->ty.as_or_throw<tvm::PrimType>();

  Expr arg = call->args[0];
  tirx::Function getter = GetDLTensorField(tirx::TVMStructFieldKind::kDLTensorTypeCode, field_ty);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_code");
  return Call(Type::Missing(), Op::Get("relax.call_tir_packed"), {gvar_getter, Tuple({arg})});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.inspect.tensor_dtype_code")
      .signature(sig::arg("tensor", "The tensor to be inspected"))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTensorDtypeCode)
      .set_attr(tvm::relax::op_attr::kLegalize, LegalizeTensorDtypeCode)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kNormalize, NormalizeToKnownPrimExpr)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

//// relax.tensor_dtype_bits

Expr tensor_dtype_bits(Expr expr) {
  static const Op op = Op::Get("relax.inspect.tensor_dtype_bits");
  return Call(Type::Missing(), op, {expr});
}

Type InferTypeTensorDtypeBits(const Call& call, const BlockBuilder&) { return PrimType::UInt(8); }

Expr LegalizeTensorDtypeBits(const BlockBuilder& bb, const Call& call) {
  PrimType field_ty = call->ty.as_or_throw<tvm::PrimType>();

  Expr arg = call->args[0];
  tirx::Function getter = GetDLTensorField(tirx::TVMStructFieldKind::kDLTensorTypeBits, field_ty);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_bits");
  return Call(Type::Missing(), Op::Get("relax.call_tir_packed"), {gvar_getter, Tuple({arg})});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.inspect.tensor_dtype_bits")
      .signature(sig::arg("tensor", "The tensor to be inspected"))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTensorDtypeBits)
      .set_attr(tvm::relax::op_attr::kLegalize, LegalizeTensorDtypeBits)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kNormalize, NormalizeToKnownPrimExpr)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

//// relax.tensor_dtype_lanes

Expr tensor_dtype_lanes(Expr expr) {
  static const Op op = Op::Get("relax.inspect.tensor_dtype_lanes");
  return Call(Type::Missing(), op, {expr});
}

Type InferTypeTensorDtypeLanes(const Call& call, const BlockBuilder&) { return PrimType::UInt(16); }

Expr LegalizeTensorDtypeLanes(const BlockBuilder& bb, const Call& call) {
  PrimType field_ty = call->ty.as_or_throw<tvm::PrimType>();

  Expr arg = call->args[0];
  tirx::Function getter = GetDLTensorField(tirx::TVMStructFieldKind::kDLTensorTypeLanes, field_ty);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_dtype_lanes");
  return Call(Type::Missing(), Op::Get("relax.call_tir_packed"), {gvar_getter, Tuple({arg})});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.inspect.tensor_dtype_lanes")
      .signature(sig::arg("tensor", "The tensor to be inspected"))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTensorDtypeLanes)
      .set_attr(tvm::relax::op_attr::kLegalize, LegalizeTensorDtypeLanes)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kNormalize, NormalizeToKnownPrimExpr)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

//// relax.tensor_ndim

Expr tensor_ndim(Expr expr) {
  static const Op op = Op::Get("relax.inspect.tensor_ndim");
  return Call(Type::Missing(), op, {expr});
}

Type InferTypeTensorNDim(const Call& call, const BlockBuilder&) { return PrimType::Int(32); }

Expr LegalizeTensorNDim(const BlockBuilder& bb, const Call& call) {
  PrimType field_ty = call->ty.as_or_throw<tvm::PrimType>();

  Expr arg = call->args[0];
  tirx::Function getter = GetDLTensorField(tirx::TVMStructFieldKind::kDLTensorNDim, field_ty);

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_ndim");
  return Call(Type::Missing(), Op::Get("relax.call_tir_packed"), {gvar_getter, Tuple({arg})});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.inspect.tensor_ndim")
      .signature(sig::arg("tensor", "The tensor to be inspected"))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTensorNDim)
      .set_attr(tvm::relax::op_attr::kLegalize, LegalizeTensorNDim)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kNormalize, NormalizeToKnownPrimExpr)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

//// relax.tensor_shape_i

Expr tensor_shape_i(Expr expr) {
  static const Op op = Op::Get("relax.inspect.tensor_shape_i");
  return Call(Type::Missing(), op, {expr});
}

Type InferTypeTensorShape(const Call& call, const BlockBuilder&) {
  auto dlpack_type = PrimType::Int(64);

  auto [tensor_ty, int_imm_axis] = GetTensorArgInfoWithIndex(call);

  auto tensor_shape = tensor_ty->GetShape();

  if (int_imm_axis && tensor_shape.has_value()) {
    return tensor_shape.value()[int_imm_axis.value()].ty();
  } else {
    return dlpack_type;
  }
}

Expr LegalizeTensorShape(const BlockBuilder& bb, const Call& call) {
  PrimType field_ty = call->ty.as_or_throw<tvm::PrimType>();

  tirx::Function getter = [&]() -> tirx::Function {
    tvm::Var dlpack_handle("dlpack_handle", PtrType::VoidPtrType());
    tvm::Var axis("axis", PrimType::Int(64));

    tvm::Var ndim("ndim", PrimType::Int(32));

    tirx::TensorVar shape_buffer =
        tirx::TensorVar("shape", tirx::TensorType({ndim.as_or_throw<PrimExpr>()}, field_ty));

    tvm::Var extent("extent", field_ty);

    tvm::SeqStmt body(
        {tvm::AssertStmt(0 <= axis.as_or_throw<PrimExpr>(), StringImm("RuntimeError"),
                         {StringImm("Specified axis may not be negative")}),
         tvm::Bind(ndim, tvm::Call(ndim->ty.as_or_throw<PrimType>(), tirx::abi_field_get_op(),
                                   {dlpack_handle, IntImm::Int32(0),
                                    IntImm::Int32(tirx::TVMStructFieldKind::kDLTensorNDim)})
                             .as_or_throw<PrimExpr>()),
         tvm::AssertStmt(
             axis.as_or_throw<PrimExpr>() <
                 tvm::prim::cast(ndim.as_or_throw<PrimExpr>(), axis->ty.as_or_throw<PrimType>()),
             StringImm("RuntimeError"),
             {StringImm("Specified axis may not be larger than the tensor's dimensionality")}),
         tvm::Bind(shape_buffer,
                   tvm::Call(shape_buffer.ty(), tvm::tirx::decl_tensor_op(),
                             {tvm::Call(shape_buffer.ty()->DataPtrType(), tirx::abi_field_get_op(),
                                        {dlpack_handle, IntImm::Int32(0),
                                         IntImm::Int32(tirx::TVMStructFieldKind::kDLTensorShape)})},
                             {}, {shape_buffer.ty()})),
         tvm::Bind(extent, TensorLoad(shape_buffer, {axis.as_or_throw<PrimExpr>()})),
         tvm::Return(extent)});

    DictAttrs attrs(
        {{tvm::s_tir::attr::kIsScheduled.name, true}, {tvm::tirx::attr::kIsHostFunc.name, true}});

    tirx::Function func({dlpack_handle, axis}, body, field_ty, attrs);

    return func;
  }();

  GlobalVar gvar_getter = bb->AddFunction(getter, "_get_tensor_shape_i");
  return Call(Type::Missing(), Op::Get("relax.call_tir_packed"), {gvar_getter, Tuple(call->args)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.inspect.tensor_shape_i")
      .signature(sig::arg("tensor", "The tensor to be inspected"),
                 sig::arg<IntExpr>("axis", "The axis whose extent should be returned"))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTensorShape)
      .set_attr(tvm::relax::op_attr::kLegalize, LegalizeTensorShape)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kNormalize, NormalizeToKnownPrimExpr)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

//// relax.tensor_stride_i

Expr tensor_stride_i(Expr expr) {
  static const Op op = Op::Get("relax.inspect.tensor_stride_i");
  return Call(Type::Missing(), op, {expr});
}

Type InferTypeTensorStride(const Call& call, const BlockBuilder&) {
  auto dlpack_type = PrimType::Int(64);

  auto [tensor_ty, int_imm_axis] = GetTensorArgInfoWithIndex(call);

  auto opt_tensor_shape = tensor_ty->GetShape();

  if (int_imm_axis && opt_tensor_shape.has_value()) {
    // As of 2024-03-14, Relax does not have an explicit
    // representation for striding in `TensorType`.  The
    // `FLegalize` function for most operators is implemented in terms
    // of `topi`, and is then converted from TE to `tirx::Function`
    // using `tvm::tirx::CreateFunction`.  The `te::Tensor` is
    // converted to a `tirx::TensorVar` in `RewriteStageToBlock`, and uses
    // the default empty list for the strides.  The empty strides
    // represent a compact data array.
    //
    // Therefore, while Relax does not explicitly represent the
    // striding of a tensor, it implicitly requires compact striding
    // for any legalizable Tensor.
    auto tensor_shape = opt_tensor_shape.value();
    PrimExpr stride = IntImm::Int64(1);
    for (size_t axis = int_imm_axis.value() + 1; axis < tensor_shape.size(); axis++) {
      stride = stride * tensor_shape[axis];
    }
    return stride.ty();
  } else {
    return dlpack_type;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.inspect.tensor_stride_i")
      .signature(sig::arg("tensor", "The tensor to be inspected"),
                 sig::arg<IntExpr>("axis", "The axis whose extent should be returned"))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTensorStride)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kNormalize, NormalizeToKnownPrimExpr)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

//// relax.tensor_byte_offset

Expr tensor_byte_offset(Expr expr) {
  static const Op op = Op::Get("relax.inspect.tensor_byte_offset");
  return Call(Type::Missing(), op, {expr});
}

Type InferTypeTensorByteOffset(const Call& call, const BlockBuilder&) {
  auto dlpack_type = PrimType::UInt(64);

  auto tensor_ty = GetTensorArgInfo(call);

  auto opt_tensor_shape = tensor_ty->GetShape();
  if (opt_tensor_shape.has_value()) {
    // Relax implicitly requires that the byte offset is zero for any
    // legalizable tensor.  See InferTypeTensorStride for full
    // explanation.
    return dlpack_type;
  } else {
    return dlpack_type;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.inspect.tensor_byte_offset")
      .signature(sig::arg("tensor", "The tensor to be inspected"))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTensorByteOffset)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kNormalize, NormalizeToKnownPrimExpr)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

//// relax.tensor_elem_offset

Expr tensor_elem_offset(Expr expr) {
  static const Op op = Op::Get("relax.inspect.tensor_elem_offset");
  return Call(Type::Missing(), op, {expr});
}

Type InferTypeTensorElemOffset(const Call& call, const BlockBuilder&) {
  auto dlpack_type = PrimType::UInt(64);

  auto tensor_ty = GetTensorArgInfo(call);

  auto opt_tensor_shape = tensor_ty->GetShape();
  if (opt_tensor_shape.has_value()) {
    // Relax implicitly requires that the element offset is zero for
    // any legalizable tensor.  See InferTypeTensorStride for
    // full explanation.
    return dlpack_type;
  } else {
    return dlpack_type;
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.inspect.tensor_elem_offset")
      .signature(sig::arg("tensor", "The tensor to be inspected"))
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeTensorElemOffset)
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kNormalize, NormalizeToKnownPrimExpr)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace inspect
}  // namespace relax
}  // namespace tvm
