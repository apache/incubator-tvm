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

#include "attention.h"

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/op_attr.h>

#include <utility>

namespace tvm {
namespace relax {

/* relax.nn.attention */

Expr attention(Expr query, Expr key, Expr value, ffi::Optional<Expr> bias,
               ffi::Optional<FloatImm> scale, ffi::Optional<ffi::String> causal_mask,
               ffi::Optional<IntImm> window_size) {
  ffi::ObjectPtr<AttentionAttrs> attrs = ffi::make_object<AttentionAttrs>();
  attrs->scale = scale;
  attrs->causal_mask = causal_mask;
  attrs->window_size = window_size;

  if (bias) {
    return Call(Type::Missing(), Op::Get("relax.nn.attention_bias"),
                {std::move(query), std::move(key), std::move(value), bias.value()}, Attrs(attrs),
                {});
  }
  return Call(Type::Missing(), Op::Get("relax.nn.attention"),
              {std::move(query), std::move(key), std::move(value)}, Attrs(attrs), {});
}

Expr attention_var_len(Expr query, Expr key, Expr value, Expr seqstart_q, Expr seqstart_k,
                       Expr max_seqlen_q, Expr max_seqlen_k, ffi::Optional<FloatImm> scale,
                       ffi::Optional<ffi::String> causal_mask, ffi::Optional<IntImm> window_size) {
  ffi::ObjectPtr<AttentionAttrs> attrs = ffi::make_object<AttentionAttrs>();
  attrs->scale = scale;
  attrs->causal_mask = causal_mask;
  attrs->window_size = window_size;

  return Call(Type::Missing(), Op::Get("relax.nn.attention_var_len"),
              {query, key, value, seqstart_q, seqstart_k, max_seqlen_q, max_seqlen_k}, Attrs(attrs),
              {});
}

void AttentionAttrs::RegisterReflection() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AttentionAttrs>()
      .def_ro(
          "scale", &AttentionAttrs::scale,
          "The custom scale applied before the softmax. The default value is 1 / sqrt(head_dim).",
          refl::DefaultValue(ffi::Optional<FloatImm>{}))
      .def_ro("causal_mask", &AttentionAttrs::causal_mask,
              "The type of the causal mask, i.e. 'TopLeft' and 'BottomRight'.",
              refl::DefaultValue(ffi::Optional<ffi::String>{}))
      .def_ro("window_size", &AttentionAttrs::window_size,
              "The size of the window for sliding-window attention.",
              refl::DefaultValue(ffi::Optional<IntImm>{}));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.op.nn.attention", attention)
      .def("relax.op.nn.attention_var_len", attention_var_len);
}

Type InferTypeAttention(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<TensorType> input_ty = GetInputTensorType(call, ctx);
  TensorType q_ty = input_ty[0];
  TensorType k_ty = input_ty[1];
  TensorType v_ty = input_ty[2];
  auto diag_dim = [&](TensorType ty, ffi::String name) {
    if (ty->ndim != 4) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The " << name << " should have 4 dimension, namely "
          << "[batch size, sequence length, number of heads, dimension of heads].";
    }
  };
  diag_dim(q_ty, "query");
  diag_dim(k_ty, "key");
  diag_dim(v_ty, "value");
  const ShapeExprObj* q_shape = q_ty->shape.as<ShapeExprObj>();
  const ShapeExprObj* k_shape = k_ty->shape.as<ShapeExprObj>();
  const ShapeExprObj* v_shape = v_ty->shape.as<ShapeExprObj>();
  PrimExpr num_batches = q_shape->values[0];
  PrimExpr num_queries = q_shape->values[1];
  PrimExpr num_heads = q_shape->values[2];
  PrimExpr head_dim = q_shape->values[3];
  PrimExpr num_keys = k_shape->values[1];
  PrimExpr head_dim_value = v_shape->values[3];
  sym::Analyzer analyzer = ctx->GetAnalyzer();
  auto diag_equal = [&](PrimExpr v1, PrimExpr v2, ffi::String m1, ffi::String m2, ffi::String dim) {
    if (analyzer->CanProve(v1 != v2)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The " << m1 << " " << dim << " and the " << m2 << " " << dim
          << " should be the same. However, the " << dim << " of " << m1 << " is " << v1
          << " while the " << dim << " of " << m2 << " is " << v2;
    }
  };
  auto multiple_of = [&](PrimExpr v1, PrimExpr v2, ffi::String m1, ffi::String m2,
                         ffi::String dim) {
    if (analyzer->CanProve(indexmod(v1, v2) != 0)) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The " << m1 << " " << dim << " should be a multiple of " << m2 << " " << dim
          << ". However, the " << dim << " of " << m1 << " is " << v1 << " while the " << dim
          << " of " << m2 << " is " << v2;
    }
  };

  diag_equal(num_batches, k_shape->values[0], "query", "key", "batch size");
  diag_equal(num_batches, v_shape->values[0], "query", "value", "batch size");
  multiple_of(num_heads, k_shape->values[2], "query", "key", "number of heads");
  multiple_of(num_heads, v_shape->values[2], "query", "value", "number of heads");
  diag_equal(num_keys, v_shape->values[1], "key", "value", "sequence length");
  diag_equal(head_dim, k_shape->values[3], "query", "key", "dimension of heads");

  if (input_ty.size() == 4) {
    TensorType bias_ty = input_ty[3];
    const ShapeExprObj* bias_shape = bias_ty->shape.as<ShapeExprObj>();
    if (bias_ty->ndim != 4) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "The bias should have 4 dimensions."
          << "However, the bias input has " << bias_ty->ndim << " dimensions.";
    }
    auto diag_equal_or_broadcast = [&](PrimExpr v1, PrimExpr v2, ffi::String m1, ffi::String m2,
                                       ffi::String dim) {
      if (analyzer->CanProve(v1 != v2) && !tvm::prim::IsOne(v2)) {
        TVM_FFI_VISIT_THROW(ValueError, call)
            << "The " << m1 << " " << dim << " and the " << m2 << " " << dim
            << " should be the same or broadcastable. However, the " << dim << " of " << m1
            << " is " << v1 << " while the " << dim << " of " << m2 << " is " << v2;
      }
    };
    diag_equal_or_broadcast(num_batches, bias_shape->values[0], "query", "bias", "batch size");
    diag_equal_or_broadcast(num_heads, bias_shape->values[1], "query", "bias", "number of heads");
    diag_equal_or_broadcast(num_queries, bias_shape->values[2], "query", "bias", "sequence length");
    diag_equal(num_keys, bias_shape->values[3], "key", "bias", "sequence length");
  }

  ffi::Array<PrimExpr> output_shape = {num_batches, num_queries, num_heads, head_dim_value};
  return TensorType(ShapeExpr(output_shape), q_ty->dtype, q_ty->vdevice);
}

Call InferMixedPrecisionAttention(const Call& call, DLDataType out_dtype) {
  return attention(call->args[0], call->args[1], call->args[2], std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt)
      .as_or_throw<Call>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.nn.attention")
      .signature(sig::arg("query", "The input queries tensor."),
                 sig::arg("key", "The input keys tensor."),
                 sig::arg("value", "The input values tensor."), sig::call_attrs<AttentionAttrs>())
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionAttention)
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeAttention)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  OpDef("relax.nn.attention_bias")
      .signature(sig::arg("query", "The input queries tensor."),
                 sig::arg("key", "The input keys tensor."),
                 sig::arg("value", "The input values tensor."),
                 sig::arg("bias", "The input bias tensor."), sig::call_attrs<AttentionAttrs>())
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionAttention)
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeAttention)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  OpDef("relax.nn.attention_var_len")
      .signature(sig::arg("query", "The input queries tensor."),
                 sig::arg("key", "The input keys tensor."),
                 sig::arg("value", "The input values tensor."),
                 sig::arg("seqstart_q", "The cumsum of query sequence lengths, prepended with 0."),
                 sig::arg("seqstart_k", "The cumsum of key sequence lengths, prepended with 0."),
                 sig::arg("max_seqlen_q", "The maximum query sequence length in the batch."),
                 sig::arg("max_seqlen_k", "The maximum key sequence length in the batch."),
                 sig::call_attrs<AttentionAttrs>())
      .set_attr(tvm::relax::op_attr::kMixedPrecisionPolicy, MixedPrecisionPolicyKind::kAlways)
      .set_attr(tvm::relax::op_attr::kInferMixedPrecision, InferMixedPrecisionAttention)
      .set_attr(tvm::relax::op_attr::kInferTypeWithBuilder, InferTypeAttention)
      .set_attr(tvm::relax::op_attr::kPurity, true);

  AttentionAttrs::RegisterReflection();
}

}  // namespace relax
}  // namespace tvm
