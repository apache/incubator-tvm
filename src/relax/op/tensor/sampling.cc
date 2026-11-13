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
 * \file sampling.cc
 * \brief sampling operators.
 */

#include "sampling.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/op_attr.h>

#include <utility>

namespace tvm {
namespace relax {

void MultinomialFromUniformAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<MultinomialFromUniformAttrs>().def_ro(
      "dtype", &MultinomialFromUniformAttrs::dtype, "Data type of the output indices.",
      refl::DefaultValue((DLDataType{kDLInt, 64, 1})));
}

TVM_FFI_STATIC_INIT_BLOCK() { MultinomialFromUniformAttrs::RegisterReflection(); }

/* relax.multinomial_from_uniform */

Expr multinomial_from_uniform(Expr prob, Expr uniform_sample, Expr sample_indices,
                              DLDataType dtype) {
  ffi::ObjectPtr<MultinomialFromUniformAttrs> attrs =
      ffi::make_object<MultinomialFromUniformAttrs>();
  attrs->dtype = dtype;

  static const Op op = Op::Get("relax.multinomial_from_uniform");
  return Call(Type::Missing(), op,
              {std::move(prob), std::move(uniform_sample), std::move(sample_indices)}, Attrs(attrs),
              {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.op.multinomial_from_uniform", multinomial_from_uniform);
}

Type InferTypeMultinomialFromUniform(const CallObj* call_node) {
  const Call call = ffi::GetRef<Call>(call_node);
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  CheckNumArguments(call);
  TensorType prob_ty = GetInputTensorType(call, 0);
  TensorType uniform_sample_ty = GetInputTensorType(call, 1);
  TensorType sample_indices_ty = GetInputTensorType(call, 2);
  const auto* attrs = call->attrs.as<MultinomialFromUniformAttrs>();

  // Only the element kind matters here; shape inference does not depend on vector lanes.
  if (!prob_ty->IsUnknownDtype() &&
      !prob_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Multinomial_from_uniform op requires the input prob to have float dtype. "
           "However, the given prob dtype is "
        << prob_ty->dtype;
  }
  // Only the element kind matters here; shape inference does not depend on vector lanes.
  if (!uniform_sample_ty->IsUnknownDtype() &&
      !uniform_sample_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLFloat,
                                                    DLDataTypeCode::kDLBfloat)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Multinomial_from_uniform op requires the input uniform_sample to have float "
           "dtype. However, the given uniform_sample dtype is "
        << uniform_sample_ty->dtype;
  }
  // Only the element kind matters here; shape inference does not depend on vector lanes.
  if (!sample_indices_ty->IsUnknownDtype() &&
      !sample_indices_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLInt)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Multinomial from uniform op requires the input sample_indices to have int "
           "dtype. However, the given sample_indices dtype is "
        << sample_indices_ty->dtype;
  }
  if (prob_ty->IsUnknownNdim() || uniform_sample_ty->IsUnknownNdim() ||
      sample_indices_ty->IsUnknownNdim()) {
    return TensorType(PrimType(attrs->dtype), kUnknownNDim, prob_ty->vdevice);
  }
  if (prob_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Multinomial_from_uniform op requires the input prob to be a 2D tensor. "
           "However, the given prob tensor has ndim "
        << prob_ty->ndim;
  }
  if (uniform_sample_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Multinomial_from_uniform op requires the input uniform_sample to be a 2D "
           "tensor. However, the given uniform_sample tensor has ndim "
        << uniform_sample_ty->ndim;
  }
  if (sample_indices_ty->ndim != 2) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Multinomial_from_uniform op requires the input sample_indices to be a 2D "
           "tensor. However, the given sample_indices tensor has ndim "
        << sample_indices_ty->ndim;
  }

  // Expected to be `(batch, vocab_size)`
  const auto* prob_shape = prob_ty->shape.as<ShapeExprObj>();
  // Expected to be `(n, 1)`
  const auto* uniform_sample_shape = uniform_sample_ty->shape.as<ShapeExprObj>();
  // Expected to be `(n, 1)`
  const auto* sample_indices_shape = sample_indices_ty->shape.as<ShapeExprObj>();
  // The output shape is expected to be `(n, 1)`

  if (prob_shape == nullptr || uniform_sample_shape == nullptr || sample_indices_shape == nullptr) {
    return TensorType(PrimType(attrs->dtype), 2, prob_ty->vdevice);
  }

  PrimExpr batch = prob_shape->values[0];
  PrimExpr n = uniform_sample_shape->values[0];
  sym::Analyzer ana;
  if (!ana->CanProveEqual(n, sample_indices_shape->values[0])) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Multinomial_from_uniform op requires the input uniform_sample and "
           "sample_indices to have the same batch size. "
           "However, the given uniform_sample tensor has batch size `"
        << n << "` and the given sample_indices tensor has batch size `"
        << sample_indices_shape->values[0] << "`";
  }
  if (!tvm::prim::IsOne(uniform_sample_shape->values[1]) ||
      !tvm::prim::IsOne(sample_indices_shape->values[1])) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Multinomial_from_uniform op requires the input uniform_sample and "
           "sample_indices to be 2D tensors with the second dimension being 1. "
           "However, the given uniform_sample tensor has shape "
        << uniform_sample_ty->shape << " and the given sample_indices tensor has shape "
        << sample_indices_ty->shape;
  }
  return TensorType(ShapeExpr({n, 1}), PrimType(attrs->dtype), prob_ty->vdevice);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.multinomial_from_uniform")
      .signature(sig::arg("prob", "The probability tensor."),
                 sig::arg("uniform_sample", "The uniform sample tensor."),
                 sig::arg("sample_indices", "The sample indices tensor."),
                 sig::call_attrs<MultinomialFromUniformAttrs>())
      .set_native_fn_attr<&InferTypeMultinomialFromUniform>(tvm::op_attr::kInferType)
      .set_attr(tvm::relax::op_attr::kPurity, true);
}

}  // namespace relax
}  // namespace tvm
