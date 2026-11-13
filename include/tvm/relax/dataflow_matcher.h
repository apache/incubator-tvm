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
 * \file tvm/relax/dataflow_matcher.h
 * \brief A pattern matcher for matching dataflow properties.
 */
#ifndef TVM_RELAX_DATAFLOW_MATCHER_H_
#define TVM_RELAX_DATAFLOW_MATCHER_H_

#include <tvm/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>
#include <tvm/relax/dataflow_pattern.h>

#include <memory>

namespace tvm {
namespace relax {

/**
 * \brief Determine if a pattern matches an expression.
 * \note The behavior of MatchExpr is to match a relax.Expr (`expr`) syntactically through
 * one given pattern (`pattern`).
 *
 * \param pattern The pattern to match
 * \param expr The expression to match
 * \param bindings The mapping from relax.Var to relax.Expr
 * \return true if matched
 * \return false if unmatched
 */
bool MatchExpr(DFPattern pattern, Expr expr,
               ffi::Optional<ffi::Map<Var, Expr>> bindings = std::nullopt);

/* \brief Similar to above, but return pairs of a matching pattern and an expression.  */
ffi::Optional<ffi::Map<DFPattern, Expr>> ExtractMatchedExpr(
    DFPattern pattern, Expr expr, ffi::Optional<ffi::Map<Var, Expr>> bindings = std::nullopt);

/**
 * \brief Match a sub-graph in a DataflowBlock with a graph of patterns and return the mapping.
 * \param ctx The graph-wise patterns.
 * \param dfb The function to match.
 * \return Matched patterns and corresponding bound variables
 */
TVM_DLL ffi::Optional<ffi::Map<DFPattern, Var>> MatchGraph(const PatternContext& ctx,
                                                           const DataflowBlock& dfb);

/**
 * \brief Rewrite a function with the given pattern and the rewriter function.
 * \param ctx The pattern constraint context under which rewriting takes place.
 * \param rewriter The function to be called on a successful matching for rewriting.
    Given the map of patterns and corresponding variables (bound variables or parameters),
    it should return a map that specifies new values for matched bound variables.
 * \param f The function to rewrite
 * \return The rewritten or the input function, depending on the pattern matching result.
 */
TVM_DLL Function RewriteBindings(
    const PatternContext& ctx,
    ffi::TypedFunction<ffi::Map<Var, Expr>(ffi::Map<DFPattern, Var>, ffi::Map<Var, Expr>)> rewriter,
    Function f);

/**
 * \brief Rewrite a function with the given pattern and the rewriter function.
 *
 * Pattern match and replace at an expression level.  This level of
 * granularity does not allow simultaneous replacement cannot be
 * performed.  In addition, removal of bindings cannot be performed
 * explicitly, and is only done implicitly through RemoveAllUnused.
 * See also `RewriteBindings`, which performs replacement on a
 * block-level, and does not have these restrictions.
 *
 * \param pattern The pattern to be replaced
 *
 * \param rewriter The function to be called on a successful pattern
 *  matching.  Given the matched expression and a map of sub-matches,
 *  it should return the replacement expression.  If the expression
 *  doesn't require updating (e.g. replacement required checks beyond
 *  those expressed in the pattern), it should return the expression
 *  unmodified.
 *
 * \param func The function to rewrite
 *
 * \return The updated function, if any updates were applied.
 */
TVM_DLL Function RewriteCall(const DFPattern& pattern,
                             ffi::TypedFunction<Expr(Expr, ffi::Map<DFPattern, Expr>)> rewriter,
                             Function func);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DATAFLOW_MATCHER_H_
