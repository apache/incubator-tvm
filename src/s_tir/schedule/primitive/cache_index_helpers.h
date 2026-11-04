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
 * \file cache_index_helpers.h
 * \brief Analysis tools and utility functions used by the cache_index primitive,
 *        extracted from common_subexpr_elim_tools.
 */

#ifndef TVM_S_TIR_SCHEDULE_PRIMITIVE_CACHE_INDEX_HELPERS_H_
#define TVM_S_TIR_SCHEDULE_PRIMITIVE_CACHE_INDEX_HELPERS_H_

#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/stmt_functor.h>  // For the class s_tir::StmtExprVisitor
#include <tvm/support/ordered_map.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>
#include <utility>  // For pairs datatype
#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief A computation table is a hashtable which associates to each expression being computed
          a number (which is the number of time that it is computed)
          It is important to note that the hash used is a ffi::StructuralHash (and not an
 ffi::ObjectPtrHash) as we need to hash similarly deeply equal terms. The comparison used is
 prim::ExprDeepEqual, which is stricter than ffi::StructuralEqual (as it does not do variables
 remapping), so it is compatible with ffi::StructuralHash (intended to be used with
 ffi::StructuralEqual).
 */
using ComputationTable =
    support::OrderedMap<PrimExpr, size_t, ffi::StructuralHash, prim::ExprDeepEqual>;

/*!
 * \brief A cache of computations is made of a pair of two hashtables, which respectively associate
          to each statement or expression of the program its computation table. Its purpose is
          to avoid the CSE pass from recomputing repeatedly the same tables of computations.
 */
struct ComputationCache {
  // Part of the cache for statements
  // It maps each known statement to its computation table
  std::unordered_map<Stmt, ComputationTable, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      cache_stmt_table_computations_;

  // Part of the cache for expressions
  // It maps each known expression to its computation table
  std::unordered_map<PrimExpr, ComputationTable, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      cache_expr_table_computations_;
};

/*!
 * \brief Visitor which returns in a hashtable the (syntatic) computations done by an expression
          or by a statement.
 * \note Computations here are considered syntactically, meaning that semantically equivalent
          computations that are not syntactically the same are not merged together.
 */
class ComputationsDoneBy : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;

  // Toplevel (static) methods
  static ComputationTable GetComputationsDoneBy(
      const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);
  static ComputationTable GetComputationsDoneBy(
      const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);

  // Constructor
  ComputationsDoneBy(std::function<bool(const PrimExpr&)> is_eligible_computation,
                     std::function<bool(const PrimExpr&)> can_contain_computations);

 protected:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView expr) override;

  ffi::Optional<VisitInterrupt> Visit_(const IfObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) override;
  ffi::Optional<VisitInterrupt> Visit_(const WhileObj* op) override;

 private:
  static ComputationTable ComputationsDoneByChildrenOf(
      const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);
  static ComputationTable ComputationsDoneByChildrenOf(
      const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);

  // The predicate used for knowing which computations are eligible
  std::function<bool(const PrimExpr&)> is_eligible_computation_;
  // The predicate used for knowing in which nodes we can search for eligible computations
  std::function<bool(const PrimExpr&)> can_contain_computations_;
  // The object being constructed and "returned" by the Visit() method
  ComputationTable table_of_computations_;
  // Cache for preventing to compute repeatedly the computations done by the same stmt or expr
  static ComputationCache cache_;
};

/*!
 * \brief Visitor that computes the *direct* subexpressions of a given expression.
 * \note Returns only the direct subexpressions of the given expressions, not all the subexprs.
          So for instance, for (A+(B+C)) it will return A and (B+C) if they are eligible,
          but not B and C.
 */
class DirectSubexpr : public s_tir::StmtExprVisitor {
 public:
  using s_tir::StmtExprVisitor::Visit_;

  // Toplevel (static) function
  static std::vector<PrimExpr> GetDirectSubexpressions(
      const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
      std::function<bool(const PrimExpr&)> can_contain_computations);

  // Constructor
  DirectSubexpr(std::function<bool(const PrimExpr&)> is_eligible_computation,
                std::function<bool(const PrimExpr&)> can_contain_computations);

 protected:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView expr) override;

 private:
  // The predicate used for knowing which computations are eligible
  std::function<bool(const PrimExpr&)> is_eligible_computation_;
  // The predicate used for knowing in which nodes we can search for eligible subexpressions
  std::function<bool(const PrimExpr&)> can_contain_computations_;

  // We haven't entered the Visit() method yet
  bool entered_ = false;
  // The vector of direct subexpressions that we are building
  std::vector<PrimExpr> direct_subexpr_;
};

bool EqualTerms(const PrimExpr& a, const PrimExpr& b);
// Used for deciding the (decidable) equivalence relation
PrimExpr NormalizeTerm(const PrimExpr& expr, bool do_normalization);
// The equivalence relation, which is the syntactical equality when `identify_equiv_terms` is false
bool EquivalentTerms(const PrimExpr& a, const PrimExpr& b, bool identify_equiv_terms);
std::vector<std::pair<PrimExpr, size_t>> SyntacticToSemanticComputations(
    const ComputationTable& table, bool identify_equiv_terms);

void InsertElemToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>* sorted_vec,
                                            const std::pair<PrimExpr, size_t>& pair);

void InsertVectorToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>* sorted_vec,
                                              const std::vector<PrimExpr>& vec_to_add,
                                              bool identify_equiv_terms, size_t increase_count = 1);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_S_TIR_SCHEDULE_PRIMITIVE_CACHE_INDEX_HELPERS_H_
