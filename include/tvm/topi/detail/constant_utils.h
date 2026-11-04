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
 * \file constant_utils.h
 * \brief Utility functions for handling constants in TVM expressions
 */
#ifndef TVM_TOPI_DETAIL_CONSTANT_UTILS_H_
#define TVM_TOPI_DETAIL_CONSTANT_UTILS_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/te/operation.h>

#include <string>
#include <vector>

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief Test whether the given Expr is a constant integer
 *
 * \param expr the Expr to query
 *
 * \return true if the given expr is a constant int or uint, false otherwise.
 */
inline bool IsConstInt(PrimExpr expr) { return expr->IsInstance<tvm::IntImmObj>(); }

/*!
 * \brief Test whether the given Array has every element as constant integer.
 * Undefined elements are also treat as constants.
 *
 * \param array the array to query
 *
 * \return true if every element in array is constant int or uint, false otherwise.
 */
inline bool IsConstIntArray(ffi::Array<PrimExpr> array) {
  bool IsConstInt = true;
  for (auto const& elem : array) {
    IsConstInt &= !elem.defined() || elem->IsInstance<tvm::IntImmObj>();
  }
  return IsConstInt;
}

/*!
 * \brief Get the value of the given constant integer expression. An error
 * is logged if the given expression is not a constant integer.
 *
 * \param expr The expression to get the value of
 *
 * \return The integer value.
 */
inline int64_t GetConstInt(PrimExpr expr) {
  if (expr->IsInstance<tvm::IntImmObj>()) {
    return static_cast<int64_t>(expr.as<tvm::IntImmObj>()->value);
  }
  LOG(ERROR) << "expr must be a constant integer";
  return -1;
}

/*!
 * \brief Get the value of all the constant integer expressions in the given array
 *
 * \param exprs The array of expressions to get the values of
 * \param var_name The name to be used when logging an error in the event that any
 * of the expressions are not constant integers.
 *
 * \return A vector of the integer values
 */
inline std::vector<int> GetConstIntValues(ffi::Array<PrimExpr> exprs, const std::string& var_name) {
  std::vector<int> result;
  if (!exprs.defined()) return result;
  for (auto expr : exprs) {
    TVM_FFI_ICHECK(IsConstInt(expr))
        << "All elements of " << var_name << " must be constant integers";
    result.push_back(GetConstInt(expr));
  }
  return result;
}

/*!
 * \brief Get the value of all the constant integer expressions in the given array
 *
 * \param exprs The array of expressions to get the values of
 * \param var_name The name to be used when logging an error in the event that any
 * of the expressions are not constant integers.
 *
 * \return A vector of the int64_t values
 */
inline std::vector<int64_t> GetConstInt64Values(ffi::Array<PrimExpr> exprs,
                                                const std::string& var_name) {
  std::vector<int64_t> result;
  if (!exprs.defined()) return result;
  for (auto expr : exprs) {
    TVM_FFI_ICHECK(IsConstInt(expr))
        << "All elements of " << var_name << " must be constant integers";
    result.push_back(GetConstInt(expr));
  }
  return result;
}

/*!
 * \brief Check whether the two expressions are equal or not, if not simplify the expressions and
 * check again
 * \note This is stronger equality check than tvm::tirx::Equal
 * \param lhs First expression
 * \param rhs Second expression
 * \return result True if both expressions are equal, else false
 */
inline bool EqualCheck(PrimExpr lhs, PrimExpr rhs) {
  tvm::prim::ExprDeepEqual expr_equal;
  bool result = expr_equal(lhs, rhs);
  if (!result) {
    PrimExpr t = tvm::sym::Analyzer()->Simplify(lhs - rhs);
    if (const IntImmObj* i = t.as<IntImmObj>()) {
      result = i->value == 0;
    }
  }
  return result;
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_CONSTANT_UTILS_H_
