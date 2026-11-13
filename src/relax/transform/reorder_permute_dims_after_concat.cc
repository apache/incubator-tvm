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
 * \file tvm/relax/transform/reorder_permute_dims_after_concat.cc
 * \brief Reorder concat(permute_dims(A), permute_dims(B)) into permute_dims(concat(A,B))
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_set>
#include <vector>

#include "../op/tensor/index.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

namespace {
std::tuple<DFPattern, ffi::TypedFunction<Expr(Expr, ffi::Map<DFPattern, Expr>)>> CreatePatterns() {
  // TODO(Lunderberg): Allow pattern-matching to handle a flexible
  // number of arguments, each of which matches the same type of
  // pattern.
  //
  // Because we instantiate one DFPattern for each value in
  // `min_concat <= i <= max_concat`, we don't want to set
  // `max_concat` to an extremely high value.  The current value of 12
  // was chosen to be significantly higher than the highest value
  // required so far (3, for query/key/value in attention layers), but
  // not so high that it requires an excessive number of `DFPattern`.
  //
  // This value is deliberately *NOT* exposed, as `max_concat` may be
  // increased at any point that it is required, and other use cases
  // should not depend on its value.  If there is a use case that
  // requires more matmuls to be handled, and pattern-matching does
  // not yet support a flexible number of `Tuple` elements,
  // `max_concat` should be increased.
  size_t min_concat = 2;
  size_t max_concat = 12;

  std::vector<DFPattern> pat_args;
  std::vector<DFPattern> pat_permute_dims;
  for (size_t i = 0; i < max_concat; i++) {
    auto arg = WildcardPattern();
    pat_args.push_back(arg);
    pat_permute_dims.push_back(IsOp("relax.permute_dims")(arg));
  }

  auto make_pattern_with_num_concat = [&](size_t num_concat) -> DFPattern {
    TVM_FFI_ICHECK_LT(num_concat, pat_permute_dims.size());
    auto concat_tuple = TuplePattern(
        ffi::Array<DFPattern>(pat_permute_dims.begin(), pat_permute_dims.begin() + num_concat));
    return IsOp("relax.concat")(concat_tuple);
  };

  DFPattern pat_concat = make_pattern_with_num_concat(min_concat);
  for (size_t i = min_concat + 1; i < max_concat; i++) {
    pat_concat = pat_concat | make_pattern_with_num_concat(i);
  }

  auto get_permute_dims_optional_axes = [](const Expr& expr) -> ffi::Optional<ffi::Array<int64_t>> {
    auto call = expr.as<CallObj>();
    TVM_FFI_ICHECK(call);
    auto attrs = call->attrs.as<PermuteDimsAttrs>();
    TVM_FFI_ICHECK(attrs);

    return attrs->axes;
  };

  auto try_get_permute_dims_axes =
      [get_permute_dims_optional_axes](const Expr& expr) -> ffi::Optional<ffi::Array<int64_t>> {
    if (auto opt_axes = get_permute_dims_optional_axes(expr)) {
      return opt_axes;
    }

    auto call = expr.as_or_throw<Call>();
    ffi::Array<int64_t> permutation;
    auto arg_ty = call->args[0]->ty.as<TensorTypeObj>();
    TVM_FFI_ICHECK(arg_ty) << "Expected permute_dims to have a single tensor argument, "
                           << "but argument " << call->args[0] << " has type " << call->args[0]->ty;

    // An implicit permutation depends on the input rank, so it cannot be materialized here.
    if (arg_ty->IsUnknownNdim()) {
      return std::nullopt;
    }

    size_t ndim = arg_ty->ndim;
    for (size_t i = 0; i < ndim; i++) {
      permutation.push_back(static_cast<int64_t>(ndim - i - 1));
    }
    return permutation;
  };

  auto try_get_compatible_permute_dims_axes =
      [try_get_permute_dims_axes](
          const ffi::Array<Expr>& permute_dims) -> ffi::Optional<ffi::Array<int64_t>> {
    auto opt_first_axes = try_get_permute_dims_axes(permute_dims[0]);
    if (!opt_first_axes) {
      return std::nullopt;
    }
    const auto& first_axes = opt_first_axes.value();

    for (size_t i_arg = 1; i_arg < permute_dims.size(); i_arg++) {
      auto opt_i_axes = try_get_permute_dims_axes(permute_dims[i_arg]);
      if (!opt_i_axes) {
        return std::nullopt;
      }
      const auto& i_axes = opt_i_axes.value();
      if (i_axes.size() != first_axes.size()) {
        return std::nullopt;
      }
      for (size_t i_axis = 0; i_axis < first_axes.size(); i_axis++) {
        if (i_axes[i_axis] != first_axes[i_axis]) {
          return std::nullopt;
        }
      }
    }
    return opt_first_axes;
  };

  auto rewriter = [=](Expr expr, ffi::Map<DFPattern, Expr> matches) -> Expr {
    ffi::Array<Expr> args;
    ffi::Array<Expr> all_permute_dims;
    for (size_t i = 0; i < max_concat; i++) {
      if (auto permute_dim_expr = matches.Get(pat_permute_dims[i])) {
        all_permute_dims.push_back(permute_dim_expr.value());
        args.push_back(matches[pat_args[i]]);
      }
    }

    TVM_FFI_CHECK_GE(all_permute_dims.size(), min_concat, InternalError)
        << "Pattern match should return at least " << min_concat << " items, but only found "
        << all_permute_dims.size() << ": " << all_permute_dims;

    auto opt_permute_dims_axes = try_get_compatible_permute_dims_axes(all_permute_dims);
    if (!opt_permute_dims_axes) {
      return expr;
    }
    const auto& permute_dims_axes = opt_permute_dims_axes.value();

    ffi::Optional<ffi::Array<int64_t>> permute_axes =
        get_permute_dims_optional_axes(all_permute_dims[0]);

    Call concat_call = matches[pat_concat].as_or_throw<Call>();
    auto concat_attrs = concat_call->attrs.as<ConcatAttrs>();
    TVM_FFI_ICHECK(concat_attrs);

    int64_t old_concat_axis = concat_attrs->axis.value_or(0);
    int64_t ndim = static_cast<int64_t>(permute_dims_axes.size());
    if (old_concat_axis < 0) {
      old_concat_axis += ndim;
    }
    TVM_FFI_ICHECK_GE(old_concat_axis, 0)
        << "concat axis " << old_concat_axis << " out of range for " << ndim << "-D input";
    TVM_FFI_ICHECK_LT(old_concat_axis, ndim)
        << "concat axis " << old_concat_axis << " out of range for " << ndim << "-D input";

    int64_t new_concat_axis = permute_dims_axes[static_cast<size_t>(old_concat_axis)];

    auto new_concat = concat(Tuple(args), new_concat_axis);
    auto new_permute_dims = permute_dims(new_concat, permute_axes);

    return new_permute_dims;
  };

  return {pat_concat, rewriter};
}

}  // namespace

namespace transform {
Pass ReorderPermuteDimsAfterConcat() {
  auto pass_func = [=](Function func, IRModule mod, PassContext pc) {
    auto [pattern, rewriter] = CreatePatterns();
    return RewriteCall(pattern, rewriter, func);
  };
  return CreateFunctionPass(pass_func, 1, "ReorderPermuteDimsAfterConcat");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ReorderPermuteDimsAfterConcat",
                        ReorderPermuteDimsAfterConcat);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
