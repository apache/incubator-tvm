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
 * \file view.cc
 * \brief Operator to view an existing tensor.
 */

#include "view.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

namespace tvm {
namespace relax {
using namespace tvm::prim;

/* relax.op.memory.view */
Expr view(Expr x, ffi::Optional<Expr> shape, ffi::Optional<Expr> dtype,
          ffi::Optional<Expr> relative_byte_offset) {
  Tuple void_expr(ffi::Array<Expr>{});

  static const Op op = Op::Get("relax.memory.view");
  return Call(Type::Missing(), op,
              {
                  x,
                  shape.value_or(void_expr),
                  dtype.value_or(void_expr),
                  relative_byte_offset.value_or(void_expr),
              });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.memory.view", view);
}

Type InferTypeView(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 4) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Operator " << call->op << " should receive 4 arguments, "
        << "but received " << call->args;
  }
  Expr arg_data = call->args[0];
  Expr arg_shape = call->args[1];
  Expr arg_dtype = call->args[2];
  Expr arg_relative_byte_offset = call->args[3];

  TensorType data_ty = [&]() -> TensorType {
    Type ty = GetType(arg_data);
    if (auto opt = ty.as<TensorType>()) {
      return opt.value();
    } else {
      TVM_FFI_THROW(TypeError) << "Operator " << call->op
                               << " expects first argument to be a tensor, "
                               << "but received " << arg_data << " with type " << ty;
    }
  }();
  auto view_shape_ty = [&]() -> const ShapeTypeObj* {
    Type ty = GetType(arg_shape);
    if (HasVoidType(arg_shape)) {
      // No shape change is applied.  The input tensor's shape is
      // kept as-is.
      return nullptr;
    } else if (auto ptr = ty.as<ShapeTypeObj>()) {
      // The `R.view` operation returns a different shape.
      return ptr;
    } else {
      TVM_FFI_THROW(TypeError) << "Operator " << call->op
                               << " expects second argument to be a ShapeExpr, "
                               << "or a void-type (empty relax tuple), "
                               << "but received " << arg_shape << " with type " << ty;
    }
  }();

  bool view_dtype_arg_present = false;
  auto view_dtype = [&]() -> ffi::Optional<PrimType> {
    Type ty = GetType(arg_dtype);

    if (HasVoidType(arg_dtype)) {
      // No datatype change is applied.  The input tensor's dtype is
      // kept as-is.
      return std::nullopt;
    }

    Expr arg_value = arg_dtype;
    while (auto arg_var = arg_value.as<Var>()) {
      if (auto bound_value = ctx->LookupBinding(arg_var.value())) {
        arg_value = bound_value.value();
      } else {
        break;
      }
    }

    static const Op null_value_op = Op::Get("relax.null_value");
    if (const CallObj* call_node = arg_value.as<CallObj>()) {
      if (call_node->op.same_as(null_value_op)) {
        // No datatype change is applied.  This is the non-void
        // representation used for missing optional dtype arguments.
        return std::nullopt;
      }
    }

    view_dtype_arg_present = true;

    // In general, Type inference should only depend on the
    // Type of the arguments, and not on the arguments
    // themselves.  However, dtype DataTypeImm uses
    // `AnyType`, so we need to inspect the argument itself
    // in this case.
    if (auto dtype_imm = arg_value.as<DataTypeImmObj>()) {
      // We know the datatype for the view.
      TVM_FFI_CHECK(!PrimType(dtype_imm->value).IsVoid(), TypeError)
          << "Operator " << call->op
          << " expects the dtype argument to be a concrete tensor datatype, "
          << "but received void.  Use relax.null_value() if no dtype change is requested.";
      return PrimType(dtype_imm->value);
    } else if (ty.as<AnyTypeObj>()) {
      // The view changes the datatype, but we don't know what it is
      // being changed into.
      return std::nullopt;
    } else {
      TVM_FFI_THROW(TypeError) << "Operator " << call->op
                               << " expects the dtype argument to be a dtype DataTypeImm, "
                               << "but received " << arg_dtype << " with type " << ty;
    }
  }();

  auto view_relative_byte_offset = [&]() -> ffi::Optional<PrimExpr> {
    Type ty = GetType(arg_relative_byte_offset);

    if (HasVoidType(arg_relative_byte_offset)) {
      // No byte offset is specified, so no change is applied.
      return IntImm::Int64(0);
    } else if (auto prim_ty = ty.as<PrimTypeObj>()) {
      TVM_FFI_CHECK_EQ(prim_ty->dtype, (DLDataType{kDLInt, 64, 1}), TypeError)
          << "Operator " << call->op
          << " expects the relative_byte_offset to be a 64-bit integer, but received "
          << arg_relative_byte_offset << ", which has type " << ty;
      if (arg_relative_byte_offset.as<DataflowVarObj>()) {
        // A primitive DataflowVar is a runtime value, not a symbolic value
        // that can be analyzed.
        return std::nullopt;
      } else if (auto prim_value = arg_relative_byte_offset.as<PrimExpr>()) {
        // An offset of known value is applied.  The known value may
        // be dynamic.
        return prim_value.value();
      } else {
        // An offset of unknown value is applied.
        return std::nullopt;
      }
    } else {
      TVM_FFI_THROW(TypeError) << "Operator " << call->op
                               << " expects the relative_byte_offset argument "
                               << "to be a Relax PrimExpr.  "
                               << "However, expression " << call
                               << " provides relative_byte_offset of " << arg_relative_byte_offset
                               << ", which has type " << ty;
    }
  }();

  ffi::Optional<ffi::Array<PrimExpr>> input_shape = data_ty->GetShape();

  ffi::Optional<ffi::Array<PrimExpr>> output_shape = std::nullopt;
  int output_ndim = kUnknownNDim;
  if (view_shape_ty && view_shape_ty->values.has_value()) {
    output_shape = view_shape_ty->values.value();
  } else if (view_shape_ty) {
    output_ndim = view_shape_ty->ndim;
  } else if (input_shape) {
    output_shape = input_shape;
  } else {
    output_ndim = data_ty->ndim;
  }

  ffi::Optional<PrimType> output_dtype = view_dtype_arg_present ? view_dtype : data_ty->dtype;

  // Helper function returns the number of bytes per vectorized element.
  auto get_size_bytes = [](DLDataType dtype) -> ffi::Optional<IntImm> {
    PrimType ty(dtype);
    if (ty.IsVoid() || ty.IsScalableVector()) {
      return std::nullopt;
    } else {
      return IntImm::Int64(static_cast<int64_t>(ty.StorageBytes()));
    }
  };

  // Helper function, returns the number of elements in an array,
  // given the shape of that array.
  auto get_num_elements =
      [&ctx](const ffi::Optional<ffi::Array<PrimExpr>>& shape) -> ffi::Optional<PrimExpr> {
    if (!shape.has_value()) {
      return std::nullopt;
    }

    PrimExpr num_elements = IntImm::Int32(1);
    for (const auto& dim : shape.value()) {
      num_elements *= dim;
    }
    return ctx->GetAnalyzer()->Simplify(num_elements);
  };

  ffi::Optional<PrimExpr> input_nelements = get_num_elements(input_shape);
  ffi::Optional<PrimExpr> output_nelements = get_num_elements(output_shape);

  ffi::Optional<IntImm> input_element_size =
      data_ty->dtype.has_value() ? get_size_bytes(data_ty->dtype.value()->dtype) : std::nullopt;
  ffi::Optional<IntImm> output_element_size =
      output_dtype.has_value() ? get_size_bytes(output_dtype.value()->dtype) : std::nullopt;

  if (input_nelements && output_nelements && input_element_size && output_element_size &&
      view_relative_byte_offset) {
    // The shapes and dtype of input and output are known.  We know
    // the byte_offset that is applied, and can verify that the view
    // does not overrun the bounds of the original array.

    PrimExpr input_nbytes = input_nelements.value() * input_element_size.value();
    PrimExpr output_nbytes = output_nelements.value() * output_element_size.value();
    PrimExpr view_end = output_nbytes + view_relative_byte_offset.value();

    if (ctx->GetAnalyzer()->CanProve(output_nbytes + view_relative_byte_offset.value() >
                                     input_nbytes)) {
      TVM_FFI_THROW(ValueError)
          << "Views into an array must not exceed the bounds of the array being viewed.  "
          << "However, expression " << call << " attempted to create view of type "
          << TensorType(ShapeExpr(output_shape.value()), output_dtype)
          << " with relative byte offset " << view_relative_byte_offset
          << ", viewing into the array " << arg_data << " of type " << data_ty << ".  "
          << "The end of the view would occur at byte " << view_end
          << ", relative to the start of array " << arg_data << ", but " << arg_data << " is only "
          << input_nbytes << " long.";
    }

  } else if (input_nelements && output_nelements && input_element_size && output_element_size) {
    // The shapes and dtype of input and output are known.  However,
    // we don't know if the `byte_offset` is being adjusted.  We can
    // still check validate using the size of the view.  If the view
    // is larger than the original array, then it would overrun its
    // bounds regardless of the `relative_byte_offset` being applied.

    PrimExpr input_nbytes = input_nelements.value() * input_element_size.value();
    PrimExpr output_nbytes = output_nelements.value() * output_element_size.value();

    if (ctx->GetAnalyzer()->CanProve(output_nbytes > input_nbytes)) {
      TVM_FFI_THROW(ValueError)
          << "Views into an array must not exceed the bounds of the array being viewed.  "
          << "However, expression " << call << " attempted to create view of type "
          << TensorType(ShapeExpr(output_shape.value()), output_dtype)
          << " from input array of type " << data_ty << ".  "
          << "This view would increase the size from " << output_nbytes << " bytes to "
          << output_nbytes << " bytes.";
    }

  } else if (input_element_size && output_element_size && !view_shape_ty) {
    // The output view has a known dtype, which is different from the
    // known dtype of the input array.  Because the view's shape is
    // the same as the original array, when counted in number of
    // elements, an increase to the per-element size would cause the
    // view to be larger than the original array.

    TVM_FFI_CHECK_GE(input_element_size.value()->value, output_element_size.value()->value,
                     ValueError)
        << "Operator " << call->op
        << " may not produce a view that exceeds the bounds of the original array.  "
        << "In expression " << call << " the data type is changed from " << data_ty->dtype << " to "
        << view_dtype.value() << ", increasing the size per element from " << input_element_size
        << " bytes to " << output_element_size << " bytes.  "
        << "Consider providing a new shape for the R.view.";
  } else if (input_nelements && output_nelements && !view_dtype_arg_present) {
    // The shape is being updated, while keeping the datatype the
    // same.  Even though we don't know the size of each element, we
    // know it must be the same for the input and output arrays.  An
    // increase to the number of elements would cause the view to be
    // larger than the original array, regardless of the size of each
    // individual element.

    if (ctx->GetAnalyzer()->CanProve(output_nelements.value() > input_nelements.value())) {
      TVM_FFI_THROW(ValueError)
          << "Views into an array must not exceed the bounds of the array being viewed.  "
          << "However, expression " << call << " attempted to view array " << arg_data
          << " (shape = " << input_shape << ", " << input_nelements << " elements) as shape "
          << output_shape << " with " << output_nelements << " elements.";
    }
  } else if (view_relative_byte_offset && !view_shape_ty && !view_dtype) {
    // The byte_offset is being updated, but neither the shape nor the
    // dtype is changing.  Any non-zero offset will cause the view to
    // overrun the bounds of the original array.
    if (ctx->GetAnalyzer()->CanProve(view_relative_byte_offset.value() > 0)) {
      TVM_FFI_THROW(ValueError)
          << "Views into an array must not exceed the bounds of the array being viewed.  "
          << "However, expression " << call << " attempted to offset the view by "
          << view_relative_byte_offset << " bytes, "
          << "without reducing either the number of elements in the view "
          << "or the size of each element.";
    }
  }

  if (output_shape.has_value()) {
    return TensorType(ShapeExpr(output_shape.value()), output_dtype, data_ty->vdevice);
  } else {
    return TensorType(output_dtype, output_ndim, data_ty->vdevice);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tvm.relax.type.infer_view_ty", InferTypeView);
}

Expr LowerBuiltinView(const BlockBuilder& bb, const Call& call) {
  Expr data = call->args[0];
  Expr shape = call->args[1];
  Expr dtype = call->args[2];
  Expr relative_byte_offset = call->args[3];

  auto is_null_value = [&bb](Expr arg) {
    while (auto arg_var = arg.as<Var>()) {
      if (auto bound_value = bb->LookupBinding(arg_var.value())) {
        arg = bound_value.value();
      } else {
        break;
      }
    }

    static const Op null_value_op = Op::Get("relax.null_value");
    if (const CallObj* call_node = arg.as<CallObj>()) {
      return call_node->op.same_as(null_value_op);
    }
    return false;
  };

  bool dtype_is_unspecified = HasVoidType(dtype) || is_null_value(dtype);

  if (HasVoidType(shape) && dtype_is_unspecified && HasVoidType(relative_byte_offset)) {
    // Special-case, no change is required by the view.
    return data;
  }

  // Prior to legalization, it is useful to use void-type argument to
  // specify "no change".  This allows for better shape inference when
  // a pass updates the input `data` tensor.  However, when we
  // legalize the `R.view`, we must provide an explicit parameters.

  if (HasVoidType(shape)) {
    auto data_shape = data->ty.as<TensorType>().value()->GetShape();
    TVM_FFI_ICHECK(data_shape.has_value())
        << "Legalization of " << call->op
        << " requires that either the output shape be explicitly specified, "
        << "or the input shape is known.  "
        << "However, in expression " << call << ", no output shape is specified, "
        << "and the input " << data << " of type " << data->ty << " has unknown shape.";
    shape = ShapeExpr(data_shape.value());
  }

  if (dtype_is_unspecified) {
    auto data_tensor_ty = data->ty.as<TensorType>().value();
    TVM_FFI_ICHECK(!data_tensor_ty->IsUnknownDtype())
        << "Legalization of " << call->op
        << " requires that either the output dtype be explicitly specified, "
        << "or the input dtype is known.  "
        << "However, in expression " << call << ", no output dtype is specified, "
        << "and the input " << data << " of type " << data->ty << " has unknown dtype.";
    dtype = DataTypeImm(data_tensor_ty->dtype.value()->dtype);
  }

  if (HasVoidType(relative_byte_offset)) {
    relative_byte_offset = IntImm::Int64(0);
  }

  TypeDeriveFunc infer_ty_env_func(EnvFunc::Get("tvm.relax.type.infer_view_ty"));
  auto runtime_view_ty = FuncType::OpaqueFunc(infer_ty_env_func, true);

  ExternFunc runtime_view_func("runtime.TensorCreateView", runtime_view_ty);

  return Call(Type::Missing(), runtime_view_func, {data, shape, dtype, relative_byte_offset});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.memory.view")
      .signature(sig::arg("x", "The input tensor."), sig::arg("shape", "The view's shape."),
                 sig::arg("dtype", "The view's data type."),
                 sig::arg("relative_byte_offset",
                          "The view's byte offset, relative to the input tensor's byte offset."))
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeView)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_attr(tvm::relax::op_attr::kLowerBuiltin, LowerBuiltinView);
}

Expr ensure_zero_offset(const Expr& x) {
  static const Op op = Op::Get("relax.memory.ensure_zero_offset");
  return Call(Type::Missing(), op, {x});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.memory.ensure_zero_offset", ensure_zero_offset);
}

Type InferTypeEnsureZeroOffset(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Operator " << call->op << " should receive 1 argument, "
        << "but received " << call->args;
  }
  return GetType(call->args[0]);
}

Expr LowerBuiltinEnsureZeroOffset(const BlockBuilder& bb, const Call& call) {
  const ExternFunc builtin_ensure_zero_offset_{"vm.builtin.ensure_zero_offset"};
  return Call(Type::Missing(), builtin_ensure_zero_offset_, call->args, std::nullopt,
              {GetType(call)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.memory.ensure_zero_offset")
      .signature(sig::arg("x", "The input tensor."))
      .set_attr(tvm::relax::op_attr::kRequiresArgumentShapes, false)
      .set_native_fn_attr<&InferTypeEnsureZeroOffset>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true)
      .set_attr(tvm::relax::op_attr::kLowerBuiltin, LowerBuiltinEnsureZeroOffset);
}

}  // namespace relax
}  // namespace tvm
