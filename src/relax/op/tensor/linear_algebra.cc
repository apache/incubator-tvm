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
 * \file linear_algebra.cc
 * \brief Linear algebra operators.
 */

#include "linear_algebra.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/topi/einsum.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

void MatmulAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<MatmulAttrs>().def_ro("out_dtype", &MatmulAttrs::out_dtype,
                                        "The data type of the output tensor",
                                        refl::DefaultValue(ffi::Optional<DLDataType>{}));
}

void EinsumAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<EinsumAttrs>().def_ro("subscripts", &EinsumAttrs::subscripts,
                                        "The einsum expression string");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  MatmulAttrs::RegisterReflection();
  EinsumAttrs::RegisterReflection();
}

/* relax.matmul */

Expr matmul(Expr x1, Expr x2, ffi::Optional<DLDataType> out_dtype) {
  ffi::ObjectPtr<MatmulAttrs> attrs = ffi::make_object<MatmulAttrs>();
  attrs->out_dtype = out_dtype;

  static const Op op = Op::Get("relax.matmul");
  return Call(Type::Missing(), op, {std::move(x1), std::move(x2)}, Attrs(attrs), {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.matmul", matmul);
}

Type InferTypeMatmul(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  Expr lhs = call->args[0];
  Expr rhs = call->args[1];
  TensorType x1_ty = input_ty[0];
  TensorType x2_ty = input_ty[1];

  ffi::Optional<VDevice> vdev;
  if (x1_ty->vdevice.has_value() && x2_ty->vdevice.has_value()) {
    if (x1_ty->vdevice.value() == x2_ty->vdevice.value()) {
      vdev = x1_ty->vdevice.value();
    }
  } else if (x1_ty->vdevice.has_value()) {
    vdev = x1_ty->vdevice.value();
  } else if (x2_ty->vdevice.has_value()) {
    vdev = x2_ty->vdevice.value();
  }

  const auto* attrs = call->attrs.as<MatmulAttrs>();
  ffi::Optional<PrimType> out_dtype = attrs->out_dtype.has_value()
                                          ? PrimType(attrs->out_dtype.value())
                                          : InferBinaryArithOpOutDtype(call, ctx, x1_ty, x2_ty);

  if (x1_ty->IsUnknownNdim() || x2_ty->IsUnknownNdim()) {
    if (vdev.has_value()) {
      return TensorType(out_dtype, kUnknownNDim, vdev);
    }
    return TensorType(out_dtype, kUnknownNDim);
  }
  int x1_ndim = x1_ty->ndim;
  int x2_ndim = x2_ty->ndim;
  if (x1_ndim == 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul operands must not be scalar.  "
        << "However, the expression " << call << " has a LHS of " << lhs << " with type " << x1_ty
        << ", which is scalar (zero-dimensional) tensor.";
  }
  if (x2_ndim == 0) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul operands must not be scalar.  "
        << "However, the expression " << call << " has a RHS of " << rhs << " with type " << x2_ty
        << ", which is scalar (zero-dimensional) tensor.";
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
    if (vdev.has_value()) {
      return TensorType(out_dtype, output_ndim, vdev);
    }
    return TensorType(out_dtype, output_ndim);
  }

  ffi::Array<PrimExpr> x1_shape_prefix{x1_shape->values.begin(),
                                       x1_shape->values.end() - 2 + x1_prepended};
  ffi::Array<PrimExpr> x2_shape_prefix{x2_shape->values.begin(),
                                       x2_shape->values.end() - 2 + x2_appended};
  ffi::Optional<ffi::Array<PrimExpr>> output_shape_prefix =
      InferBinaryBroadcastShape(call, ctx, x1_shape_prefix, x2_shape_prefix);
  if (!output_shape_prefix.has_value()) {
    if (vdev.has_value()) {
      return TensorType(out_dtype, output_ndim, vdev);
    }
    return TensorType(out_dtype, output_ndim);
  }

  sym::Analyzer analyzer = ctx->GetAnalyzer();
  PrimExpr x1_reduction_length = x1_shape->values[x1_ty->ndim - 1];
  PrimExpr x2_reduction_length = x2_shape->values[x2_ndim - 2];
  if (analyzer->CanProve(x1_reduction_length != x2_reduction_length)) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Matmul requires the reduction length of the operands to be equal.  "
        << "However, the LHS " << lhs << " has shape " << x1_ty->shape << ", while the RHS " << rhs
        << " has shape " << x2_ty->shape << ".  The reduction dimensions of " << x1_reduction_length
        << " and " << x2_reduction_length << " are not equal.";
  }

  ffi::Array<PrimExpr> output_shape = output_shape_prefix.value();
  if (!x1_prepended) {
    output_shape.push_back(x1_shape->values[x1_ndim - 2]);
  }
  if (!x2_appended) {
    output_shape.push_back(x2_shape->values[x2_ndim - 1]);
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(output_shape.size()), output_ndim);
  if (vdev.has_value()) {
    return TensorType(ShapeExpr(output_shape), out_dtype, vdev);
  }
  return TensorType(ShapeExpr(output_shape), out_dtype);
}

Call InferMixedPrecisionMatmul(const Call& call, DLDataType out_dtype) {
  return matmul(call->args[0], call->args[1], out_dtype).as_or_throw<Call>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.matmul")
      .signature(sig::arg("x1", "The first input tensor."),
                 sig::arg("x2", "The second input tensor."), sig::call_attrs<MatmulAttrs>())
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeMatmul)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionMatmul)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.einsum */

Expr einsum(Expr operands, ffi::String subscripts) {
  ffi::ObjectPtr<EinsumAttrs> attrs = ffi::make_object<EinsumAttrs>();
  attrs->subscripts = std::move(subscripts);

  static const Op op = Op::Get("relax.einsum");
  return Call(Type::Missing(), op, {std::move(operands)}, Attrs{attrs}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.einsum", einsum);
}

Type InferTypeEinsum(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (call->args.size() != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Einsum op should take 1 argument";
  }
  ffi::Array<TensorType> operands_tensor_ty = GetTensorTypeFromTuple(call, call->args[0]);
  if (operands_tensor_ty.empty()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Einsum op expects at least one tensor in the input Tuple. However, the "
           "given input Tuple is empty.";
  }

  const auto* attrs = call->attrs.as<EinsumAttrs>();

  bool vdevice_unknown = false;
  ffi::Optional<VDevice> vdev;
  for (TensorType ty : operands_tensor_ty) {
    if (!vdevice_unknown) {
      if (ty->vdevice.has_value()) {
        if (!vdev.has_value()) {
          vdev = ty->vdevice.value();
        } else if (ty->vdevice.value()->target.has_value()) {
          // mismatch
          if (ty->vdevice.value() != vdev.value()) {
            vdevice_unknown = true;
          }
        }
      }
    }
  }

  ffi::String subscripts = attrs->subscripts;

  ffi::Optional<PrimType> operand_ty = operands_tensor_ty[0]->dtype;
  std::vector<ffi::Array<PrimExpr>> input_shapes;
  input_shapes.reserve(operands_tensor_ty.size());

  for (TensorType tensor_ty : operands_tensor_ty) {
    // Check the input tuple consists of tensors with same dtype
    if (tensor_ty->dtype != operand_ty) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << "Einsum expects all input tensors to have the same dtype. However, the "
             "input contains tensors with dtype "
          << operand_ty << " and " << tensor_ty->dtype;
    }

    // Get input shapes
    const auto* shape_expr = tensor_ty->shape.as<ShapeExprObj>();
    if (shape_expr != nullptr) {
      input_shapes.push_back(shape_expr->values);
    } else {
      if (!vdevice_unknown) {
        return TensorType(operand_ty, tensor_ty->ndim, vdev);
      }
      return TensorType(operand_ty, tensor_ty->ndim);
    }
  }
  // Calculate output shape using InferEinsumShape in topi
  ffi::Array<PrimExpr> oshape = topi::InferEinsumShape(subscripts, input_shapes);

  if (!vdevice_unknown) {
    return TensorType(ShapeExpr(oshape), operand_ty, vdev);
  }
  return TensorType(ShapeExpr(oshape), operand_ty);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.einsum")
      .signature(sig::arg("operands", "The input tensors."), sig::call_attrs<EinsumAttrs>())
      .set_native_fn_attr<&InferTypeEinsum>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.outer */

Expr outer(Expr x1, Expr x2) {
  static const Op op = Op::Get("relax.outer");
  return Call(Type::Missing(), op, {std::move(x1), std::move(x2)}, {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.outer", outer);
}

Type InferTypeOuter(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  auto input_ty = GetInputTensorType(call);
  auto x1_ty = input_ty[0];
  auto x2_ty = input_ty[1];

  // Ensure both inputs are 1D tensors
  if (x1_ty->ndim != 1 || x2_ty->ndim != 1) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "torch.outer requires both inputs to be 1D tensors.";
  }

  // Determine output shape
  auto x1_shape = x1_ty->shape.as<ShapeExprObj>();
  auto x2_shape = x2_ty->shape.as<ShapeExprObj>();
  if (!x1_shape || !x2_shape) {
    return TensorType(x1_ty->dtype, 2);
  }
  ffi::Array<PrimExpr> output_shape = {x1_shape->values[0], x2_shape->values[0]};
  return TensorType(ShapeExpr(output_shape), x1_ty->dtype);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.outer")
      .signature(sig::arg("x1", "The first input tensor."),
                 sig::arg("x2", "The second input tensor."))
      .set_native_fn_attr<&InferTypeOuter>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
