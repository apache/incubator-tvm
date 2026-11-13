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
 * \file s_tir/analysis/conditional_bounds.cc
 * \brief Scoped conditional bounds and private inequality support for S-TIR.
 */
#include "conditional_bounds.h"

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

namespace tvm {
namespace s_tir {
using namespace tvm::prim;

using namespace tvm::tirx;
using sym::Analyzer;
using sym::AnalyzerObj;
using sym::EvalSet;
using sym::IntSet;

namespace {
// The solver's intermediate representations remain local to this analysis.
struct IntGroupBounds {
  PrimExpr coef;
  ffi::Array<PrimExpr> lower;
  ffi::Array<PrimExpr> equal;
  ffi::Array<PrimExpr> upper;

  IntGroupBounds(PrimExpr coef, ffi::Array<PrimExpr> lower, ffi::Array<PrimExpr> equal,
                 ffi::Array<PrimExpr> upper)
      : coef(std::move(coef)),
        lower(std::move(lower)),
        equal(std::move(equal)),
        upper(std::move(upper)) {
    TVM_FFI_ICHECK(this->coef.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))
        << "Coefficient in IntGroupBounds must be integers";
  }

  ffi::Optional<Range> FindBestRange(const ffi::Map<Var, Range>& vranges_addl) const;
  IntGroupBounds operator+(const Range& range);
};

struct IntConstraints {
  ffi::Array<PrimVar> variables;
  ffi::Map<Var, Range> ranges;
  ffi::Array<PrimExpr> relations;

  IntConstraints(ffi::Array<PrimVar> variables, ffi::Map<Var, Range> ranges,
                 ffi::Array<PrimExpr> relations);
};

using GroupedBounds =
    std::unordered_map<Var, IntGroupBounds, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using PartialSolvedInequalities = std::pair<GroupedBounds, ffi::Array<PrimExpr>>;

// Rewrite, canonicalize, then rewrite to factor multipliers out.
constexpr int kSimplifyRewriteCanonicalRewrite = 3;

IntConstraints::IntConstraints(ffi::Array<PrimVar> variables, ffi::Map<Var, Range> ranges,
                               ffi::Array<PrimExpr> relations) {
  if (!variables.defined()) {
    variables = ffi::Array<PrimVar>();
  }
  if (!ranges.defined()) {
    ranges = ffi::Map<Var, Range>();
  }
  TVM_FFI_ICHECK(relations.defined());
  for (const PrimVar& var : variables) {
    TVM_FFI_CHECK(var.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt), TypeError)
        << "Variables in IntConstraints must be integers";
  }
  this->variables = std::move(variables);
  this->ranges = std::move(ranges);
  this->relations = std::move(relations);
}

ffi::Array<PrimExpr> AsConditions(const ffi::Array<PrimVar>& variables, const GroupedBounds& bounds,
                                  const ffi::Array<PrimExpr>& relations) {
  ffi::Array<PrimExpr> res;
  // use variables to keep the order of iteration
  // so as to get rid of any non-determinism.
  TVM_FFI_ICHECK_EQ(variables.size(), bounds.size());
  for (const auto v : variables) {
    TVM_FFI_ICHECK(bounds.count(v));
    const auto& bnds = bounds.at(v);
    PrimExpr lhs = bnds.coef * v.as_or_throw<PrimExpr>();
    for (const PrimExpr& rhs : bnds.equal) {
      res.push_back(lhs == rhs);
    }
    for (const PrimExpr& rhs : bnds.lower) {
      res.push_back(lhs >= rhs);
    }
    for (const PrimExpr& rhs : bnds.upper) {
      res.push_back(lhs <= rhs);
    }
  }
  for (const PrimExpr& e : relations) {
    res.push_back(e);
  }
  return res;
}

IntGroupBounds IntGroupBounds::operator+(const Range& r) {
  Analyzer analyzer;
  ffi::Array<PrimExpr> equal;
  ffi::Array<PrimExpr> lower;
  ffi::Array<PrimExpr> upper;
  const PrimExpr& coef = this->coef;
  if (tvm::prim::IsOne(r->extent)) {
    equal.push_back(analyzer->Simplify(r->begin * coef));
  } else {
    lower.push_back(analyzer->Simplify(r->begin * coef));
    upper.push_back(analyzer->Simplify((r->begin + r->extent - 1) * coef));
  }
  for (const auto& eq : this->equal) equal.push_back(eq);
  for (const auto& lb : this->lower) lower.push_back(lb);
  for (const auto& ub : this->upper) upper.push_back(ub);
  return IntGroupBounds(coef, lower, equal, upper);
}

ffi::Optional<Range> IntGroupBounds::FindBestRange(const ffi::Map<Var, Range>& vranges_addl) const {
  Analyzer analyzer;
  {
    sym::ConstraintContext context;
    for (const auto& [var, range] : vranges_addl) {
      context->Append(sym::RangeConstraint(var, range));
    }
    analyzer->SyncConstraintContext(context);
  }

  ffi::Map<Var, IntSet> var_intsets;
  for (auto kv : vranges_addl) {
    var_intsets.Set(kv.first, IntSet::FromRange(kv.second));
  }

  const ffi::Array<PrimExpr>& equal = this->equal;
  const PrimExpr& coef = this->coef;

  std::vector<PrimExpr> lowers(equal.begin(), equal.end());
  std::vector<PrimExpr> uppers(equal.begin(), equal.end());
  for (const auto& expr : this->lower) {
    lowers.push_back(expr);
  }
  for (const auto& expr : this->upper) {
    uppers.push_back(expr);
  }

  if (lowers.size() == 1 && uppers.size() == 1 && tvm::prim::IsOne(coef)) {
    return Range(analyzer->Simplify(lowers[0]), analyzer->Simplify(uppers[0] + 1));
  }

  // Here we will try all pairs of lower and upper bounds and find the best pair, that is, the
  // pair with the minimal difference between the upper and the lower.
  // Note that the bounds are for v, not for v*coef

  // The lower bound of the best pair so far
  ffi::Optional<PrimExpr> best_lower;
  // The difference between the upper and the lower of the best pair, maybe overapproximation
  ffi::Optional<PrimExpr> best_diff_over;

  for (const PrimExpr& low : lowers) {
    for (const PrimExpr& upp : uppers) {
      // Since diff may depend on some other variables, we compute its overapproximation
      ffi::Optional<PrimExpr> diff_over;
      PrimExpr diff_1 = analyzer->Simplify(floordiv(upp - low, coef), 3);
      IntSet diff_set1 = EvalSet(diff_1, var_intsets);
      if (diff_set1.HasUpperBound()) {
        diff_over = analyzer->Simplify(diff_set1.max(), 3);
      }

      // low is the lower bound for v*coef, but we need the lower bound for v.
      // We use rounding-up division to compute it. Since we want to use a single formula
      PrimExpr low_divided = analyzer->Simplify(floordiv(low + coef - 1, coef), 3);

      // Compute another difference which may be more precise (or not).
      PrimExpr diff_2 = analyzer->Simplify(floordiv(upp, coef) - low_divided, 3);
      IntSet diff_set2 = EvalSet(diff_2, var_intsets);
      if (diff_set2.HasUpperBound()) {
        PrimExpr diff_over_2 = analyzer->Simplify(diff_set2.max(), 3);
        diff_over = diff_over.has_value() ? (analyzer->CanProve(diff_over_2 - diff_over.value() < 0)
                                                 ? diff_over_2
                                                 : diff_over.value())
                                          : diff_over_2;
      }

      // If it is provable that the new one is strictly better than the current best one,
      // then replace it. Note that we are biased towards earlier pairs which should be simpler.
      if (diff_over.has_value() &&
          (!best_diff_over.has_value() ||
           analyzer->CanProve(diff_over.value() - best_diff_over.value() < 0))) {
        best_lower = low_divided;
        best_diff_over = diff_over.value();
      }
    }
  }

  if (!best_lower.has_value()) {
    TVM_FFI_ICHECK(!best_diff_over.has_value());
    return std::nullopt;
  }
  return Range::FromMinExtent(best_lower.value(), analyzer->Simplify(best_diff_over.value() + 1));
}

struct ExprLess {
  bool operator()(const PrimExpr& l, const PrimExpr& r) const {
    return CalculateExprComplexity(l) < CalculateExprComplexity(r);
  }
};

/*!
 * \brief normalize to the form `expr <= 0`
 */
class NormalizeComparisons : public tvm::ExprMutator {
 public:
  UnchangedOr<PrimExpr> Mutate_(const prim::EQObj* op, InplaceMode inplace_mode) override {
    return Make<prim::EQ>(op->a, op->b);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::NEObj* op, InplaceMode inplace_mode) override {
    return Make<prim::NE>(op->a, op->b);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LTObj* op, InplaceMode inplace_mode) override {
    return Make<prim::LT>(op->a, op->b);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LEObj* op, InplaceMode inplace_mode) override {
    return Make<prim::LE>(op->a, op->b);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::GTObj* op, InplaceMode inplace_mode) override {
    return Make<prim::LT>(op->b, op->a);
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::GEObj* op, InplaceMode inplace_mode) override {
    return Make<prim::LE>(op->b, op->a);
  }

 private:
  template <class T>
  PrimExpr Make(const PrimExpr& a, const PrimExpr& b) {
    // rewrite LT to LE for ints
    PrimType a_ty = a.ty();
    if (std::is_same<T, prim::LT>::value &&
        (a_ty.code() == DLDataTypeCode::kDLInt || a_ty.code() == DLDataTypeCode::kDLUInt)) {
      return prim::LE(analyzer_->Simplify(a - b + 1), IntImm(a.ty(), 0));
    }
    return T(analyzer_->Simplify(a - b), IntImm(a.ty(), 0));
  }
  sym::Analyzer analyzer_;
};

void AddInequality(std::vector<PrimExpr>* inequality_set, const PrimExpr& new_ineq,
                   AnalyzerObj* analyzer) {
  if (analyzer->CanProve(new_ineq) ||
      std::find_if(inequality_set->begin(), inequality_set->end(), [&](const PrimExpr& e) {
        return ffi::StructuralEqual()(e, new_ineq);
      }) != inequality_set->end()) {
    // redundant: follows from the vranges
    // or has already been added
    return;
  }
  if (const prim::LEObj* new_le = new_ineq.as<prim::LEObj>()) {
    for (auto iter = inequality_set->begin(); iter != inequality_set->end();) {
      const prim::LEObj* le = iter->as<prim::LEObj>();
      if (le && analyzer->CanProve(new_le->a - le->a <= 0)) {
        return;
      } else if (le && analyzer->CanProve(le->a - new_le->a <= 0)) {
        iter = inequality_set->erase(iter);
      } else {
        ++iter;
      }
    }
  }

  inequality_set->push_back(new_ineq);
}

void ClassifyByPolarity(const PrimVar& var, const std::vector<PrimExpr>& current_ineq_set,
                        std::vector<PrimExpr>* next_ineq_set, std::vector<PrimExpr>* rest,
                        std::vector<std::pair<int64_t, PrimExpr>>* coef_pos,
                        std::vector<std::pair<int64_t, PrimExpr>>* coef_neg,
                        AnalyzerObj* analyzer) {
  // Take formulas from current_ineq_set and classify them according to polarity wrt var
  // and store to coef_pos and coef_neg respectively.
  for (const PrimExpr& ineq : current_ineq_set) {
    if (const prim::LEObj* le = ineq.as<prim::LEObj>()) {
      ffi::Array<PrimExpr> coef = tirx::DetectLinearEquation(le->a, {var});
      const auto* imm = !coef.empty() ? coef[0].as<IntImmObj>() : nullptr;
      if (auto value = imm ? imm->value.as<int64_t>() : std::nullopt; value.has_value()) {
        int64_t coef0 = *value;
        if (coef0 == 0) {
          // zero polarity, straight to next_ineq_set
          AddInequality(next_ineq_set, ineq, analyzer);
        } else if (coef0 > 0) {
          coef_pos->push_back({coef0, coef[1]});
        } else if (coef0 < 0) {
          coef_neg->push_back({coef0, coef[1]});
        }
        continue;
      }
    } else if (const prim::EQObj* eq = ineq.as<prim::EQObj>()) {
      ffi::Array<PrimExpr> coef = tirx::DetectLinearEquation(eq->a, {var});
      const auto* imm = !coef.empty() ? coef[0].as<IntImmObj>() : nullptr;
      if (auto value = imm ? imm->value.as<int64_t>() : std::nullopt; value.has_value()) {
        int64_t coef0 = *value;
        if (coef0 == 0) {
          // zero polarity, straight to next_ineq_set
          AddInequality(next_ineq_set, ineq, analyzer);
        } else if (coef0 > 0) {
          // Equalities may be considered as pairs of two inequalities
          coef_pos->push_back({coef0, coef[1]});
          coef_neg->push_back({-coef0, -coef[1]});
        } else if (coef0 < 0) {
          coef_pos->push_back({-coef0, -coef[1]});
          coef_neg->push_back({coef0, coef[1]});
        }
        continue;
      }
    }

    // if nothing worked, put it in rest
    rest->push_back(ineq);
  }
}

void MoveEquality(std::vector<PrimExpr>* upper_bounds, std::vector<PrimExpr>* lower_bounds,
                  std::vector<PrimExpr>* equalities) {
  // those exist in both upper & lower bounds will be moved to equalities
  for (auto ub = upper_bounds->begin(); ub != upper_bounds->end();) {
    auto lb = std::find_if(lower_bounds->begin(), lower_bounds->end(),
                           [&](const PrimExpr& e) { return ffi::StructuralEqual()(e, *ub); });
    if (lb != lower_bounds->end()) {
      equalities->push_back(*lb);
      lower_bounds->erase(lb);
      ub = upper_bounds->erase(ub);
    } else {
      ++ub;
    }
  }
}

PartialSolvedInequalities SolveLinearInequalities(const IntConstraints& system_to_solve) {
  sym::Analyzer analyzer;
  {
    sym::ConstraintContext context;
    for (const auto& [var, range] : system_to_solve.ranges) {
      context->Append(sym::RangeConstraint(var, range));
    }
    analyzer->SyncConstraintContext(context);
  }

  // The algorithm consists in doing the following things for each variable v
  // - Take formulas from `current_ineq_set_to_solve` and
  //   classify them according to polarity wrt v.
  // - Combine each formula of positive polarity (wrt v)
  //   with each formula of negative polarity.
  // - Put the resulting combinations into `next_ineq_set_to_solve`
  //   along with unclassifiable formulas.
  // - Replace `current_ineq_set_to_solve` with `next_ineq_set_to_solve`
  //   and move to the next variable.

  // normalized inequality
  std::vector<PrimExpr> current_ineq_set_to_solve;
  std::vector<PrimExpr> next_ineq_set_to_solve;
  // A vector of pairs (c, e), c > 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, PrimExpr>> coef_pos;
  // A vector of pairs (c, e), c < 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, PrimExpr>> coef_neg;

  // formulas we don't know what to do with
  std::vector<PrimExpr> rest;

  // Simplify each inequality into the form `expr <= 0` and add to current formulas
  auto normalizer = ffi::make_object<NormalizeComparisons>();
  for (const PrimExpr& ineq : system_to_solve.relations) {
    PrimExpr simplified = analyzer->Simplify(ineq, kSimplifyRewriteCanonicalRewrite);
    PrimExpr normalized = normalizer->Mutate(simplified).ValueOrUnchanged(simplified);
    AddInequality(&current_ineq_set_to_solve, normalized, analyzer.get());
  }

  GroupedBounds res_bounds;
  for (const PrimVar& v : system_to_solve.variables) {
    TVM_FFI_ICHECK(!res_bounds.count(v))
        << "Variable " << v
        << " appears more than one time in the `variables` which might be a bug";

    next_ineq_set_to_solve.clear();
    coef_pos.clear();
    coef_neg.clear();

    // Add bounds from vranges
    if (system_to_solve.ranges.count(v)) {
      const Range& range = system_to_solve.ranges[v];
      PrimExpr range_lbound = analyzer->Simplify(range->begin, kSimplifyRewriteCanonicalRewrite);
      PrimExpr range_ubound =
          analyzer->Simplify(range->begin + range->extent - 1, kSimplifyRewriteCanonicalRewrite);
      coef_neg.push_back({-1, range_lbound});
      coef_pos.push_back({1, -range_ubound});
    }

    ClassifyByPolarity(v, current_ineq_set_to_solve, &next_ineq_set_to_solve, &rest, &coef_pos,
                       &coef_neg, analyzer.get());

    // Combine each positive inequality with each negative one (by adding them together)
    for (const auto& pos : coef_pos) {
      for (const auto& neg : coef_neg) {
        auto first_gcd = std::gcd(pos.first, -neg.first);
        PrimType v_ty = v.ty();
        PrimExpr c_pos = prim::MakeConst(v_ty, neg.first / first_gcd);
        PrimExpr c_neg = IntImm(v_ty, pos.first / first_gcd);
        // eliminate the current variable
        PrimExpr new_lhs = c_neg * neg.second - c_pos * pos.second;
        PrimExpr new_ineq = prim::LE(new_lhs, IntImm(pos.second.ty(), 0));
        // we need rewrite_simplify -> canonical_simplify -> rewrite_simplify
        // to help simplify things like (((y + 10) - (-1*(y - 20))) <= 0) => y - 5 <= 0
        // with steps = 2 it's (y*2) - 10 <= 0
        new_ineq = analyzer->Simplify(new_ineq, kSimplifyRewriteCanonicalRewrite);
        new_ineq = normalizer->Mutate(new_ineq).ValueOrUnchanged(new_ineq);
        AddInequality(&next_ineq_set_to_solve, new_ineq, analyzer.get());
      }
    }

    // Now we have to generate resulting (in)equalities for the variable v

    // Find the common denominator in a sense
    // We will generate formulas of the form coef_lcm*v <= bound
    int64_t coef_lcm = 1;
    for (const auto& pos : coef_pos) {
      coef_lcm = std::lcm(coef_lcm, pos.first);
    }
    for (const auto& neg : coef_neg) {
      coef_lcm = std::lcm(coef_lcm, -neg.first);
    }

    // The resulting lower and upper bounds
    std::vector<PrimExpr> upper_bounds;
    std::vector<PrimExpr> lower_bounds;
    upper_bounds.reserve(coef_pos.size());
    lower_bounds.reserve(coef_neg.size());

    for (const auto& pos : coef_pos) {
      PrimExpr bound = prim::MakeConst(v.ty(), -coef_lcm / pos.first) * pos.second;
      bound = analyzer->Simplify(bound, kSimplifyRewriteCanonicalRewrite);
      // Don't add if any of the existing bounds is better
      if (std::any_of(upper_bounds.begin(), upper_bounds.end(),
                      [&bound, &analyzer](const PrimExpr& o) {
                        return analyzer->CanProve(o - bound <= 0);
                      })) {
        continue;
      }
      // Erase all worse bounds
      for (auto iter = upper_bounds.begin(); iter != upper_bounds.end();) {
        if (analyzer->CanProve(*iter - bound >= 0)) {
          iter = upper_bounds.erase(iter);
        } else {
          ++iter;
        }
      }
      // Add the upper bound
      upper_bounds.push_back(bound);
    }
    for (const auto& neg : coef_neg) {
      PrimExpr bound = prim::MakeConst(v.ty(), -coef_lcm / neg.first) * neg.second;
      bound = analyzer->Simplify(bound, kSimplifyRewriteCanonicalRewrite);
      // Don't add if any of the existing bounds is better
      if (std::any_of(lower_bounds.begin(), lower_bounds.end(),
                      [&bound, &analyzer](const PrimExpr& o) {
                        return analyzer->CanProve(o - bound >= 0);
                      })) {
        continue;
      }
      // Erase all worse bounds
      for (auto iter = lower_bounds.begin(); iter != lower_bounds.end();) {
        if (analyzer->CanProve(*iter - bound <= 0)) {
          iter = lower_bounds.erase(iter);
        } else {
          ++iter;
        }
      }
      // Add the lower bound
      lower_bounds.push_back(bound);
    }

    std::vector<PrimExpr> equal;
    equal.reserve(std::min(upper_bounds.size(), lower_bounds.size()));
    MoveEquality(&upper_bounds, &lower_bounds, &equal);
    std::vector<PrimExpr> equal_list(equal.begin(), equal.end());
    std::sort(equal_list.begin(), equal_list.end(), ExprLess());

    // Write it to the result.
    IntGroupBounds bnds(prim::MakeConst(v->ty.as_or_throw<PrimType>(), coef_lcm),
                        ffi::Array<PrimExpr>(lower_bounds.begin(), lower_bounds.end()),
                        ffi::Array<PrimExpr>(equal_list.begin(), equal_list.end()),
                        ffi::Array<PrimExpr>(upper_bounds.begin(), upper_bounds.end()));
    res_bounds.emplace(v, bnds);

    std::swap(current_ineq_set_to_solve, next_ineq_set_to_solve);
  }

  // Everything that is left goes to res.relations
  ffi::Array<PrimExpr> other_conditions;
  for (const PrimExpr& e : current_ineq_set_to_solve) {
    PrimExpr e_simp = analyzer->Simplify(e, kSimplifyRewriteCanonicalRewrite);
    if (IsConstInt(e_simp, 0)) {
      // contradiction detected
      other_conditions = {IntImm::Bool(false)};
      break;
    } else if (IsConstInt(e_simp, 1)) {
      continue;
    } else {
      other_conditions.push_back(e_simp);
    }
  }

  for (const PrimExpr& e : rest) {
    other_conditions.push_back(e);
  }

  return {res_bounds, other_conditions};
}

#ifdef _MSC_VER
#pragma optimize("g", off)
#endif
IntConstraints SolveInequalitiesToRange(const IntConstraints& inequalities) {
  // Resulting ranges will contain ranges for the new variables and for the variables that are
  // not in the inequalities.variables but are in inequalities.ranges
  ffi::Map<Var, Range> res_ranges;
  // we get a set of equality, lower, upper bound of each variable.
  auto solved_system = SolveLinearInequalities(inequalities);

  GroupedBounds solved_bounds = solved_system.first;
  ffi::Array<PrimExpr> solved_other_relations = solved_system.second;

  ffi::Array<PrimExpr> res_relations;

  // this keeps being updated during determining the range of each variable.
  ffi::Map<Var, Range> vranges;
  for (std::pair<Var, Range> vr : inequalities.ranges) {
    vranges.Set(vr.first, vr.second);
  }

  // We process variables in the reverse direction to start with the most independent one.
  // This order is needed to compute new ranges.
  for (auto it = inequalities.variables.rbegin(); it != inequalities.variables.rend(); ++it) {
    sym::Analyzer analyzer;
    {
      sym::ConstraintContext context;
      for (const auto& [var, range] : vranges) {
        context->Append(sym::RangeConstraint(var, range));
      }
      analyzer->SyncConstraintContext(context);
    }

    const PrimVar& var = *it;
    TVM_FFI_ICHECK(solved_bounds.count(var));
    auto bnd = solved_bounds.at(var);
    if (IsOne(bnd.coef) && !bnd.equal.empty()) {
      // There is an equation of the form `v == expr`, so this variable can be completely removed.
      // Note that we use the 0-th expression because they are ordered by complexity,
      // so it must be the simplest one.
      // The MSVC compiler optimization must be disabled for the expression `bnd.equal[0]` which
      // triggers an internal compiler error.
      Range best_range(bnd.equal[0],
                       analyzer->Simplify(bnd.equal[0] + 1, kSimplifyRewriteCanonicalRewrite));
      res_ranges.Set(var, best_range);
      vranges.Set(var, best_range);
    } else {
      if (vranges.count(var) > 0) {
        bnd = bnd + vranges[var];
      }

      auto best_range = bnd.FindBestRange(vranges);

      if (best_range.has_value()) {
        if (analyzer->CanProveGreaterEqual(-best_range.value()->extent, 0)) {
          // range.extent <= 0 implies the input inequality system is unsolvable
          return IntConstraints(/*variables=*/{}, /*ranges=*/{},
                                /*relations=*/{IntImm::Bool(false)});
        }
        res_ranges.Set(var, best_range.value());
        vranges.Set(var, best_range.value());
      }
    }
  }

  // Add the original conditions to the resulting conditions
  sym::Analyzer analyzer;
  {
    sym::ConstraintContext context;
    for (const auto& [var, range] : vranges) {
      context->Append(sym::RangeConstraint(var, range));
    }
    analyzer->SyncConstraintContext(context);
  }
  for (const PrimExpr& old_cond :
       AsConditions(inequalities.variables, solved_bounds, solved_other_relations)) {
    if (!analyzer->CanProve(old_cond)) {
      // those not represented in vranges (res_ranges)
      res_relations.push_back(old_cond);
    }
  }

  IntConstraints system(inequalities.variables, res_ranges, res_relations);

  return system;
}
#ifdef _MSC_VER
#pragma optimize("g", on)
#endif

}  // namespace

ffi::Optional<ffi::Map<Var, Range>> ConditionalBoundsContext::TrySolveCondition() {
  // extract equations and related vars from condition expression.
  // currently only extract simple integral equations which could be solvable.
  sym::Analyzer analyzer;
  PrimExpr condition = analyzer->Simplify(condition_);
  if (IsConstInt(condition)) {
    return std::nullopt;
  }
  ffi::Array<PrimExpr> equations;
  ffi::Array<PrimVar> vars;
  std::function<void(const PrimExpr&)> fvisit = [&equations, &vars, &fvisit](const PrimExpr& e) {
    if (e->IsInstance<prim::GEObj>() || e->IsInstance<prim::GTObj>() ||
        e->IsInstance<prim::LEObj>() || e->IsInstance<prim::LTObj>() ||
        e->IsInstance<prim::EQObj>() || e->IsInstance<prim::NEObj>()) {
      bool is_simple = true;
      std::vector<PrimVar> cand_vars;
      auto walk_fn = [&cand_vars, &is_simple,
                      &e](const PrimExpr& obj) -> ffi::Expected<ffi::WalkResult> {
        if (obj.same_as(e)) {
          return ffi::WalkResult::Advance();
        } else if (const VarObj* var = obj.as<VarObj>()) {
          PrimType var_ty = var->ty.as_or_throw<PrimType>();
          if (var_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
            cand_vars.push_back(ffi::GetRef<Var>(var).as_or_throw<PrimVar>());
          }
        } else {
          is_simple &= obj->IsInstance<prim::AddObj>() || obj->IsInstance<prim::SubObj>() ||
                       obj->IsInstance<prim::MulObj>() || obj->IsInstance<prim::FloorDivObj>() ||
                       obj->IsInstance<prim::FloorModObj>() || obj->IsInstance<IntImmObj>();
        }
        return ffi::WalkResult::Advance();
      };
      ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(e, walk_fn);
      if (is_simple && !cand_vars.empty()) {
        for (const PrimVar& new_var : cand_vars) {
          if (!std::any_of(vars.begin(), vars.end(),
                           [&new_var](const PrimVar& v) { return v.same_as(new_var); })) {
            vars.push_back(new_var);
          }
        }
        equations.push_back(e.as_or_throw<PrimExpr>());
      }
    } else if (e->IsInstance<prim::AndObj>()) {
      prim::And op = e.as_or_throw<prim::And>();
      fvisit(op->a);
      fvisit(op->b);
    } else if (e->IsInstance<CallObj>()) {
      Call op = e.as_or_throw<Call>();
      if (op->op.same_as(prim::likely_op())) {
        fvisit(op->args[0].as_or_throw<PrimExpr>());
      }
    }
  };
  fvisit(condition);
  if (equations.empty() || vars.empty()) {
    return std::nullopt;
  }
  // build dom ranges for related vars
  ffi::Map<Var, Range> ranges;
  for (const Var& v : vars) {
    sym::IntSet dom{ffi::UnsafeInit{}};
    auto relax_it = relax_map_->find(v);
    if (relax_it != relax_map_->end()) {
      dom = (*relax_it).second;
    } else {
      auto hint_it = hint_map_->find(v);
      if (hint_it != hint_map_->end()) {
        dom = (*hint_it).second;
      }
    }
    if (dom.defined()) {
      ranges.Set(v, Range::FromMinExtent(dom.min(), analyzer->Simplify(dom.max() - dom.min() + 1)));
    }
  }
  // solve constraints
  IntConstraints constraint(vars, ranges, equations);
  IntConstraints result = SolveInequalitiesToRange(constraint);
  if (!result.relations.empty()) {
    return std::nullopt;
  }
  return result.ranges;
}

ConditionalBoundsContext::ConditionalBoundsContext(const PrimExpr& condition,
                                                   ffi::Map<Var, sym::IntSet>* relax_map,
                                                   ffi::Map<Var, sym::IntSet>* hint_map,
                                                   std::vector<PrimExpr>* pending_conditions)
    : condition_(condition),
      relax_map_(relax_map),
      hint_map_(hint_map),
      pending_conditions_(pending_conditions),
      origin_pending_conditions_num_(pending_conditions->size()) {}

void ConditionalBoundsContext::EnterWithScope() {
  ffi::Optional<ffi::Map<Var, Range>> constraints = TrySolveCondition();
  if (!constraints.has_value()) {
    // fail to process the condition, add to unresolved
    pending_conditions_->push_back(condition_);
    return;
  }
  // update solved var ranges
  for (const auto& kv : constraints.value()) {
    const Var& var = kv.first;
    sym::IntSet new_dom = sym::IntSet::FromRange(kv.second);
    auto relax_it = relax_map_->find(var);
    if (relax_it != relax_map_->end()) {
      // this is a bound for relaxed var
      origin_map_.Set(var, (*relax_it).second);
      relax_map_->Set(var, sym::Intersect({(*relax_it).second, new_dom}));
    } else {
      // this is a bound for free var
      auto hint_it = hint_map_->find(var);
      if (hint_it != hint_map_->end()) {
        origin_map_.Set(var, (*hint_it).second);
        hint_map_->Set(var, sym::Intersect({(*hint_it).second, new_dom}));
      } else {
        origin_map_.Set(var, sym::IntSet::Nothing());
        hint_map_->Set(var, new_dom);
      }
    }
  }
}

void ConditionalBoundsContext::ExitWithScope() {
  pending_conditions_->erase(pending_conditions_->begin() + origin_pending_conditions_num_,
                             pending_conditions_->end());
  for (const auto& p : origin_map_) {
    const Var& var = p.first;
    auto relax_it = relax_map_->find(var);
    if (relax_it != relax_map_->end()) {
      // recover bound for relaxed var
      relax_map_->Set(var, p.second);
    } else {
      // recover bound for free var
      auto hint_it = hint_map_->find(var);
      TVM_FFI_ICHECK(hint_it != hint_map_->end());
      if (p.second.IsNothing()) {
        hint_map_->erase(var);
      } else {
        hint_map_->Set(var, p.second);
      }
    }
  }
}

}  // namespace s_tir
}  // namespace tvm
