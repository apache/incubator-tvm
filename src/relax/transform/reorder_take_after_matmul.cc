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
 * \file tvm/relax/transform/expand_matmul_of_sum.cc
 * \brief Expand `matmul(x, A+B)` to `matmul(x, A) + matmul(x,B)`
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/linear_algebra.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <unordered_set>
#include <vector>

#include "../op/tensor/index.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

namespace {
std::tuple<DFPattern, ffi::TypedFunction<Expr(Expr, ffi::Map<DFPattern, Expr>)>> CreatePatterns() {
  auto pat_lhs = WildcardPattern();

  auto pat_weights = WildcardPattern();
  auto pat_indices = WildcardPattern();
  auto pat_rhs = IsOp("relax.take")(pat_weights, pat_indices);

  auto pat_matmul = IsOp("relax.matmul")(pat_lhs, pat_rhs);

  auto rewriter = [=](Expr expr, ffi::Map<DFPattern, Expr> matches) -> Expr {
    auto out_dtype = expr.as<CallNode>()->attrs.as<MatmulAttrs>()->out_dtype;
    auto lhs = matches[pat_lhs];
    auto weights = matches[pat_weights];
    auto indices = matches[pat_indices];

    const auto* take_call = matches[pat_rhs].as<CallObj>();
    TVM_FFI_CHECK(take_call, InternalError) << "Match of relax.take operator should produce Call, "
                                            << "but instead produces " << matches[pat_rhs]
                                            << " with type " << matches[pat_rhs]->GetTypeKey();
    const auto* attrs = take_call->attrs.as<TakeAttrs>();
    TVM_FFI_CHECK(attrs, InternalError)
        << "Attributes for relax.take operator should be TakeAttrs, "
        << "but were instead " << take_call->attrs << " with type " << take_call->GetTypeKey();

    const auto* lhs_ty = lhs->ty.as<TensorTypeObj>();
    if (!lhs_ty) return expr;

    const auto* weights_ty = weights->ty.as<TensorTypeObj>();
    if (!weights_ty) return expr;

    const auto* indices_ty = indices->ty.as<TensorTypeObj>();
    if (!indices_ty) return expr;

    const auto* matmul_ty = expr->ty.as<TensorTypeObj>();
    if (!matmul_ty) return expr;

    if (!attrs->axis.has_value()) return expr;
    auto axis = attrs->axis.value();

    if (lhs_ty->IsUnknownNdim() || indices_ty->IsUnknownNdim() || matmul_ty->IsUnknownNdim() ||
        weights_ty->IsUnknownNdim())
      return expr;

    if (indices_ty->ndim == 1 && axis + 1 == weights_ty->ndim) {
      // Simpler case.  The activations may have batch dimensions, but
      // the weights do not.

      // lhs.shape = [*batch, infeatures]
      // weights.shape = [infeatures, table_size]
      // indices.shape = [outfeatures]

      // out_table.shape = [*batch, table_size]
      auto out_table = matmul(lhs, weights, out_dtype);
      // new_output.shape = [*batch, outfeatures]
      auto new_output = take(out_table, indices, matmul_ty->ndim - 1, attrs->mode);

      return new_output;
    } else if (lhs_ty->ndim == 3 && weights_ty->ndim == 3 && indices_ty->ndim == 1 && axis == 0 &&
               weights_ty->GetShape().has_value() && lhs_ty->GetShape().has_value()) {
      // More complicated case, used for batched LoRA.  The conditions
      // on the argument dimensions can probably be relaxed, but would
      // probably need to remove the use of the einsum operator.

      auto lhs_shape = lhs_ty->GetShape().value();
      auto weight_shape = weights_ty->GetShape().value();

      // lhs.shape = [batch1, batch2, infeatures]
      // weights.shape = [table_size, infeatures, outfeatures]
      // indices.shape = [batch1]

      // reordered_weight.shape = [infeatures, table_size, outfeatures]
      auto reordered_weight = permute_dims(weights, ffi::Array<int64_t>{1, 0, 2});
      // fused_weight.shape = [infeatures, table_size * outfeatures]
      auto fused_weight = reshape(reordered_weight,
                                  ShapeExpr({weight_shape[1], weight_shape[0] * weight_shape[2]}));
      // fused_output.shape = [batch1, batch2, table_size * outfeatures]
      auto fused_output = matmul(lhs, fused_weight, out_dtype);
      // indexed_output.shape = [batch1, batch2, table_size, outfeatures]
      auto indexed_output = reshape(
          fused_output, ShapeExpr({lhs_shape[0], lhs_shape[1], weight_shape[0], weight_shape[2]}));

      // TODO(Lunderberg): Find a better way to express these last two
      // steps.  For an output at [i,j,k], the value is
      // `indexed_output[i, j, indices[i], k]`, but there doesn't seem
      // to be a good way to express that in relax.  It could be
      // written using `call_te`, but that would prevent later
      // optimizations from recognizing the high-level relax
      // operations.

      // duplicated_output.shape = [batch1, batch2, batch1, outfeatures]
      auto duplicated_output = take(indexed_output, indices, 2, attrs->mode);
      // new_output.shape = [batch1, batch2, outfeatures]
      auto new_output = einsum(Tuple({duplicated_output}), "ijik->ijk");

      return new_output;
    } else {
      return expr;
    }
  };

  return {pat_matmul, rewriter};
}

}  // namespace

namespace transform {
Pass ReorderTakeAfterMatmul() {
  auto pass_func = [=](Function func, IRModule mod, PassContext pc) {
    auto [pattern, rewriter] = CreatePatterns();
    return RewriteCall(pattern, rewriter, func);
  };
  return CreateFunctionPass(pass_func, 1, "ReorderTakeAfterMatmul");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ReorderTakeAfterMatmul", ReorderTakeAfterMatmul);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
