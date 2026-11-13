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
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_set>
#include <vector>

#include "../op/tensor/binary.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

namespace {
std::tuple<DFPattern, ffi::TypedFunction<Expr(Expr, ffi::Map<DFPattern, Expr>)>> CreatePatterns(
    const Function& func) {
  auto compile_time_arr = ComputableAtCompileTime(func);
  std::unordered_set<Var> compile_time_lookup(compile_time_arr.begin(), compile_time_arr.end());

  auto pat_lhs = WildcardPattern();

  auto pat_rhs_a = WildcardPattern();
  auto pat_rhs_b = WildcardPattern();
  auto pat_rhs_sum = IsOp("relax.add")(pat_rhs_a, pat_rhs_b);

  auto pat_rhs_permute_dims = IsOp("relax.permute_dims")(pat_rhs_sum);

  auto pat_rhs = pat_rhs_sum | pat_rhs_permute_dims;

  auto pat_matmul = IsOp("relax.matmul")(pat_lhs, pat_rhs);

  auto rewriter = [=](Expr expr, ffi::Map<DFPattern, Expr> matches) -> Expr {
    auto lhs = matches[pat_lhs];
    auto rhs_a = matches[pat_rhs_a];
    auto rhs_b = matches[pat_rhs_b];

    // Suppress the rewrite if `(A+B)` can be computed at
    // compile-time.
    auto is_compile_time = [&compile_time_lookup](Expr arg) -> bool {
      if (auto as_var = arg.as<Var>()) {
        return compile_time_lookup.count(as_var.value());
      } else {
        return false;
      }
    };

    if (is_compile_time(rhs_a) && is_compile_time(rhs_b)) {
      return expr;
    }

    if (matches.count(pat_rhs_permute_dims)) {
      auto call_permute = matches[pat_rhs_permute_dims].as_or_throw<Call>();
      auto attrs = call_permute->attrs.as<PermuteDimsAttrs>();
      TVM_FFI_ICHECK(attrs) << "Operator permute_dims should have PermuteDimsAttrs, "
                            << "but " << call_permute << " has attributes " << call_permute->attrs;
      auto axes = attrs->axes;

      rhs_a = permute_dims(rhs_a, axes);
      rhs_b = permute_dims(rhs_b, axes);
    }

    return add(matmul(lhs, rhs_a, std::nullopt), matmul(lhs, rhs_b, std::nullopt));
  };

  return {pat_matmul, rewriter};
}

}  // namespace

namespace transform {
Pass ExpandMatmulOfSum() {
  auto pass_func = [=](Function func, IRModule mod, PassContext pc) {
    auto [pattern, rewriter] = CreatePatterns(func);
    return RewriteCall(pattern, rewriter, func);
  };
  return CreateFunctionPass(pass_func, 1, "ExpandMatmulOfSum");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ExpandMatmulOfSum", ExpandMatmulOfSum);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
