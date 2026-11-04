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
 * \file tvm/relax/analysis.h
 * \brief The set of Relax specific analysis on IR.
 */
#ifndef TVM_RELAX_ANALYSIS_H_
#define TVM_RELAX_ANALYSIS_H_

#include <tvm/base.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/index_map.h>

#include <functional>
#include <set>
#include <utility>

namespace tvm {
namespace relax {
//-----------------------------------
// Shape expression analysis
//----------------------------------
/*!
 * \brief Can prove the two symbolic shape arrays equals to each other.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana The analyzer used for integer analysis.
 * \return The prove result.
 *
 * \note This function does best effort prove, which means
 *       if result is false, there is still possibility that
 *       two shapes equals to each other during runtime.
 */
TVM_DLL bool CanProveShapeEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs,
                                const sym::Analyzer& ana);

/*!
 * \brief Can prove the two symbolic shape expressions equals to each other.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana The analyzer used for integer analysis.
 *
 * \note This function does best effort prove, which means
 *       if result is false, there is still possibility that
 *       two shapes equals to each other during runtime.
 */
TVM_DLL bool CanProveShapeEqual(const Expr& lhs, const Expr& rhs, const sym::Analyzer& ana);

//-----------------------------------
// Foundational Type analysis
//-----------------------------------
/*!
 * \brief Get the corresponding static type from a given type.
 * \param info The type.
 * \return the corresponding static type.
 */
TVM_DLL Type GetStaticType(const Type& info);

/*!
 * \brief Get the corresponding type from static type.
 * \param type The input type
 * \return the corresponding type.
 */
TVM_DLL Type TypeFromStaticType(const Type& type);

/*!
 * \return Derive the call's ret value type from inputs.
 * \param finfo The function type.
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 * \return The derived type of the call.
 * \note  call->op field is ignored during derivation and we only rely on information
 *        presented by func_ty.
 */
TVM_DLL Type DeriveCallRetType(const FuncType& finfo, const Call& call, const BlockBuilder& ctx);
/*!
 * \brief Derive the call's ret value type using a caller-provided analyzer.
 * \param finfo The function type.
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 * \param ana Context analyzer to prove symbolic expression equality.
 * \return The derived type of the call.
 */
TVM_DLL Type DeriveCallRetType(const FuncType& finfo, const Call& call, const BlockBuilder& ctx,
                               const sym::Analyzer& ana);

/*!
 * \brief Erase the info to a corresponding more coarse grained
 *        type that is still well-defined(with all the vars in scope).
 *
 * When we are returning a Type to another scope,
 * it is important to remember that Type may carry
 * dependencies on var that is not defined the other scope.
 *
 * In such cases, it is important to call EraseToWellDefined to get
 * another Type that **only** contains the vars that are defined
 * in the target scope.
 *
 * For example, consider the following function
 *
 * \code
 *
 * @R.function
 * def f(x: R.Tensor[(n, m)]):
 *     k = tirx.Var("k", "int64")
 *     v0 = opaque_fn(x)
 *     v1 = match_cast(v0, R.Tensor[(n, k)])
 *     v2 : R.Tensor[(n + 1, k + 2)] = pad(v1)
 *     return v2
 *
 * \endcode
 *
 * In the above code, the return value y have shape `(n + 1, k + 2)`,
 * However, at the level of function signature, only n, m are defined,
 * k is undefined here.
 *
 * When we call EraseToWellDefined(R.Tensor[(n + 1, k + 2)], var_map={n: n, m: m}),
 * we will obtain R.Tensor(ndim=2), which is an erased info that does not depend
 * on k(which is undefined from parameter signature).
 *
 * However, if we call EraseToWellDefined(R.Tensor[(n + 1, m)], var_map={n: n, m: m}),
 * Then the return value will be R.Tensor[(n + 1, m)], because both n and m are defined.
 *
 * We can also make these var map to return a different expression.
 * For example, EraseToWellDefined(R.Tensor[(n + 1, m)], var_map={n: 2, m: m})
 * will give us R.Tensor[(3, m)], where n get replaced by 2.
 *
 * Use this function in the following scenarios:
 * - Decide the ty of expr with sub-scopes, such as If, SeqExpr
 * - Decide the deduced return ty of a function that can be fully decided by params.
 *
 * \param info The type.
 * \param f_var_map callback function to specify
 *        whether a Var is defined in the target scope and the value it maps to,
 *        return nullopt if var is undefined.
 *
 * \return the corresponding erased type.
 */
TVM_DLL Type EraseToWellDefined(
    const Type& info, std::function<ffi::Optional<Expr>(const Var& var)> f_var_map = nullptr);
/*!
 * \brief EraseToWellDefined overload using a caller-provided analyzer.
 * \param info The type.
 * \param f_var_map callback function to specify
 *        whether a Var is defined in the target scope and the value it maps to,
 *        return nullopt if var is undefined.
 * \param ana Context analyzer to prove symbolic expression equality.
 * \return the corresponding erased type.
 */
TVM_DLL Type EraseToWellDefined(const Type& info,
                                std::function<ffi::Optional<Expr>(const Var& var)> f_var_map,
                                const sym::Analyzer& ana);

/*!
 * \brief EraseToWellDefined variant with map.
 * \param info The type.
 * \param var_map map to specify
 *        whether a Var is defined in the target scope and the value it maps to,
 *        return nullopt if var is undefined.
 *
 * \return the corresponding erased type.
 */
TVM_DLL Type EraseToWellDefined(const Type& info, ffi::Map<Var, Expr> var_map);
/*!
 * \brief EraseToWellDefined map overload using a caller-provided analyzer.
 * \param info The type.
 * \param var_map map to specify
 *        whether a Var is defined in the target scope and the value it maps to,
 *        return nullopt if var is undefined.
 * \param ana Context analyzer to prove symbolic expression equality.
 * \return the corresponding erased type.
 */
TVM_DLL Type EraseToWellDefined(const Type& info, ffi::Map<Var, Expr> var_map,
                                const sym::Analyzer& ana);

/*!
 * \brief Fine grained result of base check.
 *
 * This analysis comes with different levels of checking failures
 * that can help to customize the compilation decisions.
 *
 * For a given pair of lhs_ty, rhs_ty. We adopt
 * the following terminology:
 * - LSet = {value | value matches lhs_ty}
 * - RSet = {value | value matches rhs_ty}
 *
 * See the definition of each level below.
 */
enum class BaseCheckResult {
  /*!
   * \brief The two value sets have no intersection at all: Interset(LSet, RSet) = empty
   */
  kFailL0 = 0,
  /*!
   * \brief LSet is not superset of RSet by only looking at static information.
   *
   * \note This level will trigger static type checking error when lhs is param and rhs is arg.
   */
  kFailL1 = 1,
  /*!
   * \brief WLSet is not superset of RSet because of mismatch in value information.
   *
   * L1-level mismatches in params of FuncType is categorized as
   * If lhs is FuncType, then L1-level mismatch in its params
   * is categorized as L2-level mismatch for lhs.
   *
   * Design considerations for functions:
   * - (a) We want to be able to erase type/value in function signature
   *       when we unify function type and preserve simpler representations.
   * - (b) We automatically insert match_cast at function boundary, so
   *       we can erase (int)->int argument as (object)->int.
   *       The input shape/type mismatch will be detected by runtime checks at function boundary.
   *       This behavior is also consistent with the ffi::Function behavior.
   *
   * \note This level means there is no problem about static known information.
   *       It is OK for the checker to do best effort and return this value.
   */
  kFailL2 = 2,
  /*! \brief LSet is superset of RSet. */
  kPass = 3
};

/*!
 * \brief Run a base check to see if base subsumes derived.
 *
 * This function returns fine-grained base-check result on reasons of failure.
 *
 * \param base The base type.
 * \param derived The derived type.
 * \return Whether the relation holds.
 *
 * \sa BaseCheckResult
 */
TVM_DLL BaseCheckResult TypeBaseCheck(const Type& base, const Type& derived);
/*!
 * \brief Run a base check using a caller-provided analyzer.
 * \param base The base type.
 * \param derived The derived type.
 * \param ana Context analyzer to prove symbolic expression equality.
 * \return Whether the relation holds.
 *
 * \sa BaseCheckResult
 */
TVM_DLL BaseCheckResult TypeBaseCheck(const Type& base, const Type& derived,
                                      const sym::Analyzer& ana);

/*!
 * \brief Check the relation of two type to see if one subsumes another one.
 *
 * \param base The base type.
 * \param derived The derived type.
 * \return Whether the relation holds.
 */
TVM_DLL bool IsBaseOf(const Type& base, const Type& derived);
/*!
 * \brief Check whether one type subsumes another using a caller-provided analyzer.
 * \param base The base type.
 * \param derived The derived type.
 * \param ana Context analyzer to prove symbolic expression equality.
 * \return Whether the relation holds.
 */
TVM_DLL bool IsBaseOf(const Type& base, const Type& derived, const sym::Analyzer& ana);

/*!
 * \brief Return the condition for which base is a superset of derived
 *
 * This function returns finer-grained conditions for kFailL2 than TypeBaseCheck
 *
 * If the returned expression is true, or simplifies to true, then
 * base is a superset of derived.  If the returned expression is
 * false, or simplifies to false, then base is not a superset of
 * derived.
 *
 * If the returned expression is neither true nor false, it is an
 * expression in terms of the symbolic variables available in `base`
 * and `derived`.
 *
 * \param base The base type.
 * \param derived The derived type.
 * \return Whether base is a base of derived.
 *
 * \sa BaseCheckResult
 */
TVM_DLL PrimExpr TypeBaseCheckPrecondition(const Type& base, const Type& derived);

/*!
 * \brief Unify the two type to their least common ancestor.
 *
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \return The unified information.
 */
TVM_DLL Type TypeLCA(const Type& lhs, const Type& rhs);
/*!
 * \brief Unify two types using a caller-provided analyzer.
 * \param lhs The left operand.
 * \param rhs The right operand.
 * \param ana Context analyzer to prove symbolic expression equality.
 * \return The unified information.
 */
TVM_DLL Type TypeLCA(const Type& lhs, const Type& rhs, const sym::Analyzer& ana);

/*!
 * \brief Get the TIR variables that appear in the input type.
 * The returned list is deduplicated - each TIR variable will appear at most once.
 * \param ty The type object to be analyzed.
 * \return The list of TIR variables that appear in the input type.
 */
TVM_DLL ffi::Array<tvm::Var> TIRVarsInType(const Type& ty);

/*!
 * \brief Get the TIR variables that appear in the input type.
 *
 * Returns all symbolic variables that are definable based on, and
 * used within, the Type.
 *
 * \param ty The type object to be analyzed.
 *
 * \return A tuple of (definable,used) TIR variables.  Both lists are
 *   deduplicated, each TIR variable will appear at most once, and in
 *   order of occurrence.
 */
TVM_DLL ffi::Array<tvm::Var> DefinableTIRVarsInType(const Type& ty);

/*! \brief Collect expressions whose usage requires them to be non-negative
 *
 * Any PrimExpr that is used as a tensor shape, or as an element in a
 * ShapeExpr, may not be negative.  This utility function can be used
 * to generate assertions prior to calling a kernel, or to provide
 * assumptions within a kernel that may be useful for simplification.
 *
 * \param ty The type to be analyzed
 *
 * \return A list of non-negative expressions.
 */
TVM_DLL ffi::Array<PrimExpr> CollectNonNegativeExpressions(const Type& ty);

/*!
 * \brief Get the TIR variables that defined in the input function.
 * The returned list is deduplicated - each TIR variable will appear at most once.
 * \param expr The relax expression (e.g. a Function) to be analyzed.
 * \return The list of TIR variables that are defined in the input function.
 */
TVM_DLL ffi::Array<tvm::Var> DefinedSymbolicVars(const Expr& expr);

/*!
 * \brief Get the TIR variables that are used but not defined in the input function.
 * The returned list is deduplicated - each TIR variable will appear at most once.
 * \param expr The relax expression (e.g. a Function) to be analyzed.
 * \return The list of TIR variables that are used but not defined in the input function.
 */
TVM_DLL ffi::Array<tvm::Var> FreeSymbolicVars(const Expr& expr);
//-----------------------------------
// General IR analysis
//-----------------------------------
/*!
 * \brief Get all bound variables from expression expr.
 *
 * Bound variables are all variables that are declared in the expr.
 * They only have meaning inside that expr, and can only be used in it.
 *
 * \param expr the expression.
 *
 * \return List of bound vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::ffi::Array<Var> BoundVars(const Expr& expr);

/*!
 * \brief Get free type parameters from expression expr.
 *
 * Free variables are variables that are not bound by a
 * varbinding or a function parameter in the context.
 *
 * \param expr the expression.
 *
 * \return List of free vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::ffi::Array<Var> FreeVars(const Expr& expr);

/*!
 * \brief Get all variables from expression expr.
 *
 * \param expr the expression.
 *
 * \return List of all vars, in the PostDFS order in the expression.
 */
TVM_DLL tvm::ffi::Array<Var> AllVars(const Expr& expr);

/*!
 * \brief Get all global variables from expression expr.
 *
 * AllVars is a superset of BoundVars and FreeVars.
 * The union of BoundVars and FreeVars is Allvars.
 *
 * \param expr the expression.
 *
 * \return List of all global variables, in the PostDFS order in the expression.
 */
TVM_DLL tvm::ffi::Array<GlobalVar> AllGlobalVars(const Expr& expr);

/*!
 * \brief Find all sets of recursive or mutually recursive functions in the module.
 *
 * Two or more functions are mutually recursive if there is some cycle of references
 * among them. For example, if there are two functions A and B, they are
 * mutually recursive if A calls B and B calls A. Another case would be with
 * three functions A, B, and C, where A calls B, B calls C, and C calls A.
 *
 * (Note that functions do not have to call each other to reference each other.
 * For example, if a function returns another function, that is still a reference
 * that could potentially be recursive, even without a call.)
 *
 * If a function is simply recursive and not mutually recursive with any other,
 * it will be reported as a group by itself.
 *
 * \param m The module
 *
 * \return List of all groups of mutually recursive functions.
 *     Each member of the result is a list of functions in the module
 *     that are all mutually recursive.
 *     If a function is simply recursive and not mutually recursive with any other,
 *     then it will be listed as a group by itself.
 */
TVM_DLL tvm::ffi::Array<tvm::ffi::Array<GlobalVar>> DetectRecursion(const IRModule& m);

/*!
 * \brief Analyze var -> value mapping from VarBindings.
 *
 * \param m The IRModule to check.
 * \return Var -> Value (Expr)
 */
TVM_DLL ffi::Map<Var, Expr> AnalyzeVar2Value(const IRModule& m);

/*!
 * \brief Analyze var -> value mapping from VarBindings.
 *
 * \param expr The expression to check.
 * \return Var -> Value (Expr)
 */
TVM_DLL ffi::Map<Var, Expr> AnalyzeVar2Value(const Expr& expr);

/*!
 * \brief Analyze var -> value mapping from VarBindings.
 *
 * \param dfb The dataflow block to check.
 * \return Var -> Value (Expr)
 */
TVM_DLL ffi::Map<Var, Expr> AnalyzeVar2Value(const DataflowBlock& dfb);

/*!
 * \brief Return a mapping from variable name to its Bindings.
 *
 * \param fn The function to be analyzed.
 * \return A mapping from variable name to its Bindings.
 */
TVM_DLL ffi::Map<ffi::String, ffi::Array<Binding>> NameToBinding(const Function& fn);

/*!
 * \brief Get the use-def chain of variables inside a dataflow block.
 *
 * \param dfb The dataflow block to be analyzed.
 * \return A map mapping variable definitions to a set of uses.
 */
TVM_DLL ffi::Map<Var, ffi::Array<Var>> DataflowBlockUseDef(const DataflowBlock& dfb);

/*!
 * \brief Get the use-def chain of variables inside a function.
 *
 * \param expr The expression to be analyzed.
 *
 * \return A tuple of variable usage and variable outputs.  The first
 * element is a map from variable definitions to the set of downstream
 * users of that definition.  The second element is a list of
 * variables whose usage occurs outside of any variable binding,
 * typically the output body of a relax::Function or a relax::SeqExpr.
 */
std::pair<ffi::Map<Var, ffi::Array<Var>>, ffi::Array<Var>> FunctionUseDef(const Expr& expr);

/*! \brief A utility struct returned by CollectVarUsage
 */
struct VarUsageInfo {
  /* \brief A map from variables to the bound expression.
   *
   * This is equivalent to the output of AnalyzeVar2Value
   */
  ffi::Map<Var, Expr> bound_values;

  /* \brief The map from variables to downstream usages of the variable
   *
   * This is equivalent to the first output of FunctionUseDef.
   */
  ffi::Map<Var, ffi::Array<Var>> downstream_usage;

  /* \brief A list of variables produced as output
   *
   * This is equivalent to the second output of FunctionUseDef
   */
  ffi::Array<Var> outputs;
};

/*! \brief Collect variable bindings and usage
 *
 * This function is equivalent to calling both FunctionUseDef and
 * AnalyzeVar2Value, but requires only a single traversal of the
 * expression.
 *
 * \param expr The expression to analyze
 *
 * \return The collected information
 */
VarUsageInfo CollectVarUsage(const Expr& expr);

/*!
 * \brief Get the used variables in an expression.
 *
 * This function collects all variables that are referenced within the given expression.
 *
 * \param expr The expression to analyze
 *
 * \return A set of variable nodes that are used in the expression
 */
TVM_DLL std::set<const VarObj*> GetUsedVars(const Expr& expr);

/*!
 * \brief Remove unused statements inside DataflowBlocks.
 *
 * \param expr The expression (typically a relax::Function) from which
 * to remove unused statements.
 *
 * \return The updated function with no unused statements in DataflowBlock.
 */
TVM_DLL Expr RemoveAllUnused(Expr expr);

/*!
 * \brief Annotate Op Pattern Kind for tirx::Function, which is used in relax FuseOps.
 *
 * \param func The tirx::Function to be analyzed.
 * \return The Op Pattern Kind.
 *
 * \note This analysis applies on TIR function but is primarily used by relax passes.
 *       As a result we place it under the relax namespace.
 */
TVM_DLL OpPatternKind AnalyzeOpPatternKind(const tirx::Function& func);

/*!
 * \brief Check if the given tirx::Function is essentially doing a reshape operation.
 * The reshape operation also includes expand_dims, squeeze, flatten, etc.
 * \details Here the allowed reshape pattern is: for example, assume the operation is
 *  `B[l_0, l_1, ..., l_b] = A[r_0, r_1, ..., r_a]`, we check if we can prove that the flattened
 * index of l_0, ..., l_b under tensor B equals to the flattened index of r_0, ..., r_a under
 * tensor A.
 * \param func The function to be examined.
 * \return A boolean indicating if the given tirx::Function is doing a reshape.
 * \note According to the description above, the returned result can only be false-negative and
 * cannot be false-positive, since whenever we cannot prove the equality, we return false. This
 * property guarantees the safety of this function.
 */
TVM_DLL bool HasReshapePattern(const tirx::Function& func);

/*!
 * \brief Check if the given expression (likely a function body) contains any impure calls.
 * \param expr The expression to be examined. If expr is a function, we check the body.
 * \param own_name (Optional.) If we are checking a recursive function body,
 *   the caller can pass the function's name so recursive calls
 *   can be ignored in the check (must be a Var or GlobalVar).
 * \return The impure expression, if one exists within the given
 *   expression.  Otherwise, std::nullopt.
 * \note Relies on Type annotations, so ensure that the module has been normalized first.
 *   Also, an impure call in a *nested* function does *not* mean that the outer expression contains
 *   an impure call--it only does if the nested function is *later called*.
 */
TVM_DLL ffi::Optional<Expr> FindImpureCall(
    const Expr& expr, const ffi::Optional<Expr>& own_name = ffi::Optional<Expr>(std::nullopt));

/*!
 * \brief Check if the given expression (likely a function body) contains any impure calls.
 * \param expr The expression to be examined. If expr is a function, we check the body.
 * \param own_name (Optional.) If we are checking a recursive function body,
 *   the caller can pass the function's name so recursive calls
 *   can be ignored in the check (must be a Var or GlobalVar).
 * \return A boolean indicating if the expression contains any impure calls.
 * \note Relies on Type annotations, so ensure that the module has been normalized first.
 *   Also, an impure call in a *nested* function does *not* mean that the outer expression contains
 *   an impure call--it only does if the nested function is *later called*.
 */
TVM_DLL bool ContainsImpureCall(
    const Expr& expr, const ffi::Optional<Expr>& own_name = ffi::Optional<Expr>(std::nullopt));

/*!
 * \brief Check if an IRModule or Function is well-formed.
 *
 * Throws an ffi::Error on the first well-formedness violation. The error is
 * seeded with the offending node so a pass runner can resolve a precise access
 * path.
 *
 * \param obj The IRModule or relax::Function to check.
 * \param check_ty If true, verify that every Expr has ty populated.
 * \note By default the type information is always checked. It is only in test cases
 * where `check_ty` might be false, so that other well-formed requirements
 * will be well tested and will not be blocked by not having type information.
 */
TVM_DLL void CheckWellFormed(ffi::Variant<IRModule, Function> obj, bool check_ty = true);

/*!
 * \brief Using the layout transforms on the outputs, suggest layout transformation on the blocks
 * and tensors for the tirx::Function.
 *
 * \param fn The tirx::Function to be analyzed.
 * \param write_tensor_transformations Array of IndexMap transformations on tirx::Function outputs.
 * \return Suggested transforms per block in `fn`. For each block the returned value is a map
 * from the object (block or tensor) to it's index map transformation.
 */

TVM_DLL ffi::Map<s_tir::SBlock, ffi::Map<ffi::ObjectRef, tirx::IndexMap>> SuggestLayoutTransforms(
    const Function& fn, ffi::Array<tirx::IndexMap> write_tensor_transformations);

/* \brief Collect variables whose value can be computed at compile-time
 *
 * If a function has the `kNumInput` attribute, then the first
 * `kNumInput` parameters are provided at run-time, while all
 * remaining parameters may be known at compile-time.  This utility
 * collects all variable bindings that only depend, directly or
 * indirectly, on the parameters known at compile-time.
 *
 * \param func The relax::Function to analyze
 *
 * \return The set of variables that can be computed at compile-time,
 * in order of their occurrence within the function.
 */
TVM_DLL ffi::Array<Var> ComputableAtCompileTime(const Function& func);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_H_
