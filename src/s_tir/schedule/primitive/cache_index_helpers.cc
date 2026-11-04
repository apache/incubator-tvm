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
 * \file cache_index_helpers.cc
 * \brief Implementation of analysis tools and utility functions used by the cache_index
 *        primitive, extracted from common_subexpr_elim_tools.
 */

#include "cache_index_helpers.h"

#include <tvm/ffi/cast.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/sym/analyzer.h>  // For the sym::Analyzer::Simplify() method simplifying terms
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt.h>

#include <algorithm>      // For std::find_if
#include <unordered_map>  // For the hashtable datatype
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

// cache_ is a static variable of the class ComputationsDoneBy, and C++ requires to define here
// such static attribute, otherwise it causes a linking error.
ComputationCache ComputationsDoneBy::cache_;

/* ********************************** Class ComputationsDoneBy **********************************
*********************************************************************************************** */

/*!
 * \brief Does the union of two tables of computations.
 * \param table_main Pointer to one of the two tables. The union will be written into it.
 * \param table_aux The other table, which won't change.
 * \note Does it directly in the first argument A for efficiency, as the union of A and B
 *       necessarily gives something which contains A, so we avoid its copy.
 */
void UnionOfComputationTables(ComputationTable* table_main, const ComputationTable& table_aux) {
  if (table_main == nullptr) {
    return;
  }
  // Adds each element of the second table to the first one
  for (const auto& current : table_aux) {
    (*table_main)[current.first] += current.second;
  }
}

/*!
 * \brief Does the union of three tables of computations.
 */
ComputationTable UnionOfComputationTables(const ComputationTable& table1,
                                          const ComputationTable& table2,
                                          const ComputationTable& table3) {
  ComputationTable result = table1;  // Copy needed as the union of 2 writes into its first arg
  UnionOfComputationTables(&result, table2);
  UnionOfComputationTables(&result, table3);

  return result;
}

/*!
 * \brief Does the intersection of two tables of computations.
 */
ComputationTable IntersectComputationTables(const ComputationTable& table1,
                                            const ComputationTable& table2) {
  ComputationTable result;
  for (const auto& current : table1) {
    auto it = table2.find(current.first);
    if (it != table2.end()) {
      result[current.first] = current.second + it->second;
    }
  }
  return result;
}

/*!
 * \brief Does the intersection of three tables of computations.
 */
ComputationTable IntersectComputationTables(const ComputationTable& table1,
                                            const ComputationTable& table2,
                                            const ComputationTable& table3) {
  ComputationTable result = IntersectComputationTables(table1, table2);
  result = IntersectComputationTables(result, table3);
  return result;
}

/*!
 * \brief Recompute the number of times that each computation in table_main is seen in the tables
          contained by the vector of tables vecTables.
 */
void RecomputeNbTimesSeen(ComputationTable* table_main,
                          const std::vector<const ComputationTable*>& vec_tables) {
  if (table_main == nullptr) {
    return;
  }
  for (auto& current_elem : *table_main) {
    current_elem.second = 0;
    for (auto current_table : vec_tables) {
      auto it = current_table->find(current_elem.first);
      if (it != current_table->end()) {
        current_elem.second += it->second;
      }
    }
  }
}

/*!
 * \brief Builds a table for a node that has three children.
 */
ComputationTable BuildTableForThreeChildrenNode(const ComputationTable& table_child1,
                                                const ComputationTable& table_child2,
                                                const ComputationTable& table_child3) {
  ComputationTable result;
  ComputationTable child2_inter_child3 = IntersectComputationTables(table_child2, table_child3);
  ComputationTable child1_inter_child2 = IntersectComputationTables(table_child1, table_child2);
  ComputationTable child1_inter_child3 = IntersectComputationTables(table_child1, table_child3);

  result = UnionOfComputationTables(child2_inter_child3, child1_inter_child2, child1_inter_child3);

  std::vector<const ComputationTable*> vec_tables = {&table_child1, &table_child2, &table_child3};
  RecomputeNbTimesSeen(&result, vec_tables);

  return result;
}

/*!
 * \brief Toplevel (static) method for a PrimExpr
 */
ComputationTable ComputationsDoneBy::GetComputationsDoneBy(
    const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  if (expr.as<IntImmObj>() != nullptr || expr.as<FloatImmObj>() != nullptr ||
      expr.as<StringImmObj>() != nullptr || expr.as<PrimVar>()) {
    return {};
  }

  auto it_table_expr = cache_.cache_expr_table_computations_.find(expr);
  if (it_table_expr != cache_.cache_expr_table_computations_.end()) {
    return it_table_expr->second;
  }

  auto computations_done_by =
      ffi::make_object<ComputationsDoneBy>(is_eligible_computation, can_contain_computations);
  computations_done_by->Visit(expr);
  cache_.cache_expr_table_computations_[expr] = computations_done_by->table_of_computations_;

  return computations_done_by->table_of_computations_;
}

/*!
 * \brief Toplevel (static) method for a Stmt
 */
ComputationTable ComputationsDoneBy::GetComputationsDoneBy(
    const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  auto it_table_stmt = cache_.cache_stmt_table_computations_.find(stmt);
  if (it_table_stmt != cache_.cache_stmt_table_computations_.end()) {
    return it_table_stmt->second;
  }

  auto computations_done_by =
      ffi::make_object<ComputationsDoneBy>(is_eligible_computation, can_contain_computations);
  computations_done_by->Visit(stmt);
  cache_.cache_stmt_table_computations_[stmt] = computations_done_by->table_of_computations_;

  return computations_done_by->table_of_computations_;
}

/*!
 * \brief Constructor of ComputationsDoneBy.
 */
ComputationsDoneBy::ComputationsDoneBy(
    std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations)
    : is_eligible_computation_(is_eligible_computation),
      can_contain_computations_(can_contain_computations) {}

/*!
 * \brief Dispatch expressions and statements through one intercepted entry
 */
ffi::Optional<VisitInterrupt> ComputationsDoneBy::Visit(ffi::AnyView expr_value) {
  if (auto stmt = expr_value.as<Stmt>()) {
    auto it_table_stmt = cache_.cache_stmt_table_computations_.find(stmt.value());
    if (it_table_stmt != cache_.cache_stmt_table_computations_.end()) {
      UnionOfComputationTables(&table_of_computations_, it_table_stmt->second);
      return std::nullopt;
    }

    ComputationTable temp = ComputationsDoneByChildrenOf(stmt.value(), is_eligible_computation_,
                                                         can_contain_computations_);
    UnionOfComputationTables(&table_of_computations_, temp);
    return std::nullopt;
  }

  auto opt_expr = expr_value.as<PrimExpr>();
  if (!opt_expr) {
    return s_tir::StmtExprVisitor::Visit(expr_value);
  }
  PrimExpr expr = opt_expr.value();
  if (expr.as<IntImmObj>() != nullptr || expr.as<FloatImmObj>() != nullptr ||
      expr.as<StringImmObj>() != nullptr || expr.as<PrimVar>()) {
    return std::nullopt;
  }

  auto it_table_expr = cache_.cache_expr_table_computations_.find(expr);
  if (it_table_expr != cache_.cache_expr_table_computations_.end()) {
    UnionOfComputationTables(&table_of_computations_, it_table_expr->second);
    return std::nullopt;
  }

  if (is_eligible_computation_(expr)) {
    table_of_computations_[expr]++;
    return std::nullopt;
  }

  if (can_contain_computations_(expr)) {
    ComputationTable temp =
        ComputationsDoneByChildrenOf(expr, is_eligible_computation_, can_contain_computations_);
    UnionOfComputationTables(&table_of_computations_, temp);
    return std::nullopt;
  }
  return std::nullopt;
}

/*!
 * \brief The method which overrides the specific treatment for an IfObj
 */
ffi::Optional<VisitInterrupt> ComputationsDoneBy::Visit_(const IfObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->condition));
  ComputationTable computations_done_by_cond = table_of_computations_;
  table_of_computations_.clear();

  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->then_case));
  ComputationTable computations_done_by_then = table_of_computations_;
  table_of_computations_.clear();

  ComputationTable computations_done_by_else;
  if (op->else_case) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->else_case.value()));
    computations_done_by_else = table_of_computations_;
    table_of_computations_.clear();
  }

  table_of_computations_ = BuildTableForThreeChildrenNode(
      computations_done_by_cond, computations_done_by_then, computations_done_by_else);

  Stmt ref_to_op = ffi::GetRef<Stmt>(op);
  cache_.cache_stmt_table_computations_[ref_to_op] = table_of_computations_;
  return std::nullopt;
}

/*!
 * \brief The method which overrides the specific treatment for a ForObj
 */
ffi::Optional<VisitInterrupt> ComputationsDoneBy::Visit_(const ForObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->begin));
  ComputationTable computations_done_by_min = table_of_computations_;
  table_of_computations_.clear();

  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->end));
  ComputationTable computations_done_by_extent = table_of_computations_;
  table_of_computations_.clear();

  ComputationTable computations_done_by_body;
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->body));
  computations_done_by_body = table_of_computations_;
  table_of_computations_.clear();

  table_of_computations_ = BuildTableForThreeChildrenNode(
      computations_done_by_min, computations_done_by_extent, computations_done_by_body);

  Stmt ref_to_op = ffi::GetRef<Stmt>(op);
  cache_.cache_stmt_table_computations_[ref_to_op] = table_of_computations_;
  return std::nullopt;
}

/*!
 * \brief The method which overrides the specific treatment for a WhileObj
 */
ffi::Optional<VisitInterrupt> ComputationsDoneBy::Visit_(const WhileObj* op) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->condition));
  ComputationTable computations_done_by_condition = table_of_computations_;
  table_of_computations_.clear();

  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->body));
  ComputationTable computations_done_by_body = table_of_computations_;
  table_of_computations_.clear();

  table_of_computations_ =
      IntersectComputationTables(computations_done_by_condition, computations_done_by_body);

  Stmt ref_to_op = ffi::GetRef<Stmt>(op);
  cache_.cache_stmt_table_computations_[ref_to_op] = table_of_computations_;
  return std::nullopt;
}

/*!
 * \brief Static method that returns the computations done by the children of an expression.
 */
ComputationTable ComputationsDoneBy::ComputationsDoneByChildrenOf(
    const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  auto computations_done_by =
      ffi::make_object<ComputationsDoneBy>(is_eligible_computation, can_contain_computations);
  computations_done_by->s_tir::StmtExprVisitor::Visit(expr);
  cache_.cache_expr_table_computations_[expr] = computations_done_by->table_of_computations_;

  return computations_done_by->table_of_computations_;
}

/*!
 * \brief Static method that returns the computations done by the children of a statement.
 */
ComputationTable ComputationsDoneBy::ComputationsDoneByChildrenOf(
    const Stmt& stmt, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  auto computations_done_by =
      ffi::make_object<ComputationsDoneBy>(is_eligible_computation, can_contain_computations);
  computations_done_by->s_tir::StmtExprVisitor::Visit(stmt);
  cache_.cache_stmt_table_computations_[stmt] = computations_done_by->table_of_computations_;

  return computations_done_by->table_of_computations_;
}

/* *********************************** Class DirectSubexpr **************************************
*********************************************************************************************** */

/*!
 * \brief Toplevel (static) function that returns the direct subexpressions of a given expression
 */
std::vector<PrimExpr> DirectSubexpr::GetDirectSubexpressions(
    const PrimExpr& expr, std::function<bool(const PrimExpr&)> is_eligible_computation,
    std::function<bool(const PrimExpr&)> can_contain_computations) {
  auto direct_subexpr =
      ffi::make_object<DirectSubexpr>(is_eligible_computation, can_contain_computations);
  direct_subexpr->Visit(expr);

  return direct_subexpr->direct_subexpr_;
}

/*!
 * \brief Constructor of DirectSubexpr.
 */
DirectSubexpr::DirectSubexpr(std::function<bool(const PrimExpr&)> is_eligible_computation,
                             std::function<bool(const PrimExpr&)> can_contain_computations)
    : is_eligible_computation_(is_eligible_computation),
      can_contain_computations_(can_contain_computations) {}

/*!
 * \brief Find direct eligible subexpressions through the generic entry
 */
ffi::Optional<VisitInterrupt> DirectSubexpr::Visit(ffi::AnyView expr_value) {
  auto opt_expr = expr_value.as<PrimExpr>();
  if (!opt_expr) return s_tir::StmtExprVisitor::Visit(expr_value);
  PrimExpr expr = opt_expr.value();
  if (entered_) {
    if (is_eligible_computation_(expr)) {
      direct_subexpr_.push_back(expr);
      return std::nullopt;
    } else {
      if (can_contain_computations_(expr)) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit(expr));
      }
      return std::nullopt;
    }
  }

  if (can_contain_computations_(expr)) {
    entered_ = true;
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(s_tir::StmtExprVisitor::Visit(expr));
  }
  return std::nullopt;
}

/* ********************************** Utility functions *********************************
*********************************************************************************************** */

/*!
 * \brief Decides if two terms are equal syntactically
 */
bool EqualTerms(const PrimExpr& a, const PrimExpr& b) {
  prim::ExprDeepEqual deep_equal_;
  return deep_equal_(a, b);
}

/*!
 * \brief Normalization function of a term, use to decide the equivalence relation of interest
 */
PrimExpr NormalizeTerm(const PrimExpr& expr, bool do_normalization) {
  if (do_normalization) {
    sym::Analyzer analyzer;
    return analyzer->Simplify(expr);
  } else {
    return expr;
  }
}

/*!
 * \brief Decides if two terms are equivalent semantically
 */
bool EquivalentTerms(const PrimExpr& a, const PrimExpr& b, bool identify_equiv_terms) {
  return EqualTerms(NormalizeTerm(a, identify_equiv_terms), NormalizeTerm(b, identify_equiv_terms));
}

/*!
 * \brief Transforms a hashtable of syntactic computations into a vector or pairs
          (expression, counter) where equivalent computations are merged and their counters added.
 */
std::vector<std::pair<PrimExpr, size_t>> SyntacticToSemanticComputations(
    const ComputationTable& table, bool identify_equiv_terms) {
  std::vector<std::pair<PrimExpr, size_t>> result;

  if (!identify_equiv_terms) {
    result.reserve(table.size());
    for (const auto& elem : table) {
      result.push_back(elem);
    }

    return result;
  }

  support::OrderedMap<PrimExpr, std::pair<PrimExpr, size_t>, ffi::StructuralHash,
                      prim::ExprDeepEqual>
      norm_table;

  norm_table.reserve(table.size());

  for (const auto& elem : table) {
    PrimExpr norm_elem = NormalizeTerm(elem.first, identify_equiv_terms);
    auto it_found = norm_table.find(norm_elem);
    if (it_found == norm_table.end()) {
      norm_table.insert(norm_elem, elem);
    } else {
      it_found->second.second += elem.second;
    }
  }

  result.reserve(norm_table.size());

  for (const auto& kv : norm_table) {
    result.push_back(kv.second);
  }

  return result;
}

/*!
 * \brief Inserts a pair (expr,nb) to a sorted vector of such pairs (sorted by decreasing
          size of expressions) and maintain the vector sorted while doing so.
 */
void InsertElemToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>* sorted_vec,
                                            const std::pair<PrimExpr, size_t>& pair) {
  if (sorted_vec == nullptr) {
    return;
  }
  auto insertion_point = std::lower_bound(
      sorted_vec->begin(), sorted_vec->end(), pair,
      [](const std::pair<PrimExpr, size_t>& left, const std::pair<PrimExpr, size_t>& right) {
        return (CalculateExprComplexity(left.first) >= CalculateExprComplexity(right.first));
      });
  sorted_vec->insert(insertion_point, pair);
}

/*!
 * \brief Inserts a vector of expressions into a sorted vector of computations (sorted by
          decreasing size of the expression) and maintain the vector sorted while doing so.
 */
void InsertVectorToSortedSemanticComputations(std::vector<std::pair<PrimExpr, size_t>>* sorted_vec,
                                              const std::vector<PrimExpr>& vec_to_add,
                                              bool identify_equiv_terms, size_t increase_count) {
  if (sorted_vec == nullptr) {
    return;
  }
  for (auto elem_to_add : vec_to_add) {
    auto it_found =
        std::find_if(sorted_vec->begin(), sorted_vec->end(),
                     [elem_to_add, identify_equiv_terms](std::pair<PrimExpr, size_t> elem) {
                       return EquivalentTerms(elem.first, elem_to_add, identify_equiv_terms);
                     });

    if (it_found != sorted_vec->end()) {
      it_found->second += increase_count;
    } else {
      InsertElemToSortedSemanticComputations(sorted_vec, {elem_to_add, increase_count});
    }
  }
}

}  // namespace tirx
}  // namespace tvm
