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
 * \file set.cc
 * \brief Relax set operators.
 */

#include "set.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.unique */

Expr unique(Expr x, PrimExpr sorted, PrimExpr return_index, PrimExpr return_inverse,
            PrimExpr return_counts, ffi::Optional<PrimExpr> axis) {
  static const Op op = Op::Get("relax.unique");
  if (!axis) {
    return Call(Type::Missing(), op,
                {std::move(x), sorted, return_index, return_inverse, return_counts});
  } else {
    PrimExpr pv_axis = axis.value();
    return Call(Type::Missing(), op,
                {std::move(x), sorted, return_index, return_inverse, return_counts, pv_axis});
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.unique", unique);
}

Type InferTypeUnique(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = call->args[0]->ty.as_or_throw<TensorType>();
  ffi::Optional<PrimExpr> axis;
  if (call->args.size() == 6) {
    if (auto prim_value = call->args[5].as<PrimExpr>()) {
      axis = prim_value.value();
    }
  }
  if (!data_ty->IsUnknownNdim() && axis.has_value()) {
    // Normalize the axis for sanity check purpose.
    if (const auto* axis_int = axis.as<IntImmObj>()) {
      NormalizeAxis(call, data_ty->ndim, axis_int->value.as<int>().value());
    }
  }
  TVM_FFI_ICHECK(call->args[2].as<PrimExpr>());
  TVM_FFI_ICHECK(call->args[3].as<PrimExpr>());
  TVM_FFI_ICHECK(call->args[4].as<PrimExpr>());

  PrimExpr return_index = call->args[2].as_or_throw<PrimExpr>();
  PrimExpr return_inverse = call->args[3].as_or_throw<PrimExpr>();
  PrimExpr return_counts = call->args[4].as_or_throw<PrimExpr>();

  auto f_convert_to_int64 = [](const PrimExpr& value) {
    TVM_FFI_ICHECK(value->IsInstance<IntImmObj>())
        << value << " expects to be IntImm, but gets " << value->GetTypeKey();
    const auto* val_node = value.as<IntImmObj>();
    auto val_imm = ffi::GetRef<IntImm>(val_node);
    return static_cast<int64_t>(val_imm->value);
  };

  int64_t n_int_return = f_convert_to_int64(return_index) + f_convert_to_int64(return_inverse) +
                         f_convert_to_int64(return_counts);

  std::vector<Type> output_ty;
  output_ty.reserve(1 + n_int_return);

  // unique values
  if (data_ty->ndim == 0) {
    output_ty.push_back(
        TensorType(ShapeExpr({IntImm::Int64(/*value=*/1)}), data_ty->dtype, data_ty->vdevice));
  } else if (axis.has_value()) {
    output_ty.push_back(TensorType(data_ty->dtype, data_ty->ndim, data_ty->vdevice));
  } else {
    output_ty.push_back(TensorType(data_ty->dtype, /*ndim=*/1, data_ty->vdevice));
  }

  // index, inverse_indices, and counts
  // index: always 1D
  if (f_convert_to_int64(return_index)) {
    if (data_ty->ndim == 0) {
      output_ty.push_back(
          TensorType(ShapeExpr({IntImm::Int64(/*value=*/1)}), PrimType::Int(64), data_ty->vdevice));
    } else {
      output_ty.push_back(TensorType(PrimType::Int(64), /*ndim=*/1, data_ty->vdevice));
    }
  }

  // inverse_indices: always 1D per ONNX spec
  if (f_convert_to_int64(return_inverse)) {
    if (data_ty->ndim == 0) {
      output_ty.push_back(
          TensorType(ShapeExpr({IntImm::Int64(/*value=*/1)}), PrimType::Int(64), data_ty->vdevice));
    } else {
      output_ty.push_back(TensorType(PrimType::Int(64), /*ndim=*/1, data_ty->vdevice));
    }
  }

  // counts: always 1D
  if (f_convert_to_int64(return_counts)) {
    if (data_ty->ndim == 0) {
      output_ty.push_back(
          TensorType(ShapeExpr({IntImm::Int64(/*value=*/1)}), PrimType::Int(64), data_ty->vdevice));
    } else {
      output_ty.push_back(TensorType(PrimType::Int(64), /*ndim=*/1, data_ty->vdevice));
    }
  }

  if (output_ty.size() == 1) {
    return output_ty[0];
  } else {
    return TupleType(output_ty);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.unique",
        "Optional axis: The dimension to apply unique. If it is std::nullopt, the unique values of "
        "the flattened input are returned.")
      .signature(
          sig::arg("x", "The input tensor"),
          sig::arg(
              "sorted",
              "Whether to sort the unique elements in ascending order before returning as output."),
          sig::arg("return_index",
                   "Whether to return an additional tensor with indices for where elements in the "
                   "unique "
                   "tensor come from the original input."),
          sig::arg("return_inverse",
                   "Whether to return an additional tensor with indices for where elements in the "
                   "original input ended up in the returned unique list."),
          sig::arg("return_counts",
                   "Whether to return an additional tensor with counts of each unique elements"),
          sig::var_args("args"))
      .set_native_fn_attr<&InferTypeUnique>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kCallPacked, "relax.run.unique")
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

/* relax.nonzero */
Expr nonzero(Expr x) {
  static const Op op = Op::Get("relax.nonzero");
  return Call(Type::Missing(), op, {std::move(x)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.nonzero", nonzero);
}

Type InferTypeNonzero(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  TensorType data_ty = GetInputTensorType(call, 0);
  return TensorType(PrimType::Int(64), 2, data_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nonzero")
      .signature(sig::arg("x", "The input tensor"))
      .set_native_fn_attr<&InferTypeNonzero>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kCallPacked, "relax.run.nonzero")
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
