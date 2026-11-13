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
 * \file src/relax/ir/dataflow_expr_rewriter.cc
 * \brief A transform to match a Relax Expr and rewrite
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>

#include <algorithm>

#include "../transform/utils.h"
#include "dataflow_matcher.h"
#include "dataflow_rewriter.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace {
class GlobalVarReplacer : public ExprMutator {
 public:
  explicit GlobalVarReplacer(ffi::Map<GlobalVar, GlobalVar> gvar_map) : gvar_map_(gvar_map) {}

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const GlobalVarObj* op) override {
    auto gvar = ffi::GetRef<GlobalVar>(op);
    if (auto opt = gvar_map_.Get(gvar)) {
      gvar = opt.value();
    }
    return gvar;
  }

 private:
  ffi::Map<GlobalVar, GlobalVar> gvar_map_;
};

ffi::Array<Binding> TopologicalSort(const ffi::Array<Binding>& bindings) {
  std::unordered_set<Var> remaining_bindings;
  for (const auto& binding : bindings) {
    remaining_bindings.insert(binding->var);
  }

  // Utility structure used to track bindings that are moved later in
  // the list.
  struct DelayedBinding {
    Binding binding;
    std::unordered_set<Var> unmet_requirements;
    bool emitted;
  };
  std::vector<DelayedBinding> delayed_bindings;
  ffi::Array<Binding> sorted_bindings;

  // Utility function to append the
  auto push_sorted_binding = [&](Binding binding) {
    sorted_bindings.push_back(binding);
    remaining_bindings.erase(binding->var);
    for (auto& delayed_binding : delayed_bindings) {
      delayed_binding.unmet_requirements.erase(binding->var);
    }
  };

  bool required_sorting = false;
  for (const auto& binding : bindings) {
    // Collect any variables used by this binding, but are emitted by
    // a later binding.
    std::unordered_set<Var> unmet_requirements;
    for (auto free_var : FreeVars(GetBoundValue(binding))) {
      if (remaining_bindings.count(free_var)) {
        unmet_requirements.insert(free_var);
      }
    }

    if (unmet_requirements.empty()) {
      push_sorted_binding(binding);
    } else {
      required_sorting = true;
      delayed_bindings.push_back(DelayedBinding{binding, unmet_requirements, false});
    }

    bool requires_delayed_binding_check = true;
    while (requires_delayed_binding_check) {
      requires_delayed_binding_check = false;
      for (auto& delayed_binding : delayed_bindings) {
        if (!delayed_binding.emitted && delayed_binding.unmet_requirements.empty()) {
          // If we find a delayed binding that can be emitted, mark it
          // as emitted and push to the sorted list.  This may
          delayed_binding.emitted = true;
          requires_delayed_binding_check = true;
          push_sorted_binding(delayed_binding.binding);

          // The break is not necessary for a topological sort, but is
          // necessary to minimize the amount of re-ordering that is
          // performed.  With this break, the next binding is always
          // the earliest binding that is legal to emit at this point.
          break;
        }
      }
    }

    // Remove any delayed bindings that have been emitted, now that we
    // are done iterating over the delayed bindings.
    delayed_bindings.erase(
        std::remove_if(delayed_bindings.begin(), delayed_bindings.end(),
                       [](const auto& delayed_binding) { return delayed_binding.emitted; }),
        delayed_bindings.end());
  }

  // All bindings should be emitted by this point.  If any remain,
  // then there exists a circular dependency somewhere in the
  // remaining bindings.
  TVM_FFI_CHECK(delayed_bindings.empty(), ValueError) << "Bindings contain circular dependency";

  if (required_sorting) {
    return sorted_bindings;
  } else {
    return bindings;
  }
}
}  // namespace

void RewriteSpec::Append(RewriteSpec other) {
  if (variable_rewrites.empty()) {
    *this = std::move(other);
    return;
  }
  if (other.variable_rewrites.empty()) {
    return;
  }

  UniqueNameSupply gvar_name_supply("");
  for (const auto& [gvar, func] : new_subroutines) {
    gvar_name_supply->ReserveName(gvar->name_hint);
  }

  ffi::Map<GlobalVar, GlobalVar> gvar_rewrites;
  for (auto [gvar, func] : other.new_subroutines) {
    if (auto it = new_subroutines.find(gvar); it != new_subroutines.end()) {
      // The two rewrites provide the same GlobalVar.
      // (e.g. Multiple rewrites of the same pattern.)  Ensure that
      // they are referring to the same underlying BaseFunc.
      TVM_FFI_ICHECK(func.same_as((*it).second));
    } else if (auto new_name = gvar_name_supply->FreshName(gvar->name_hint);
               new_name != gvar->name_hint) {
      // The two rewrites provide distinct GlobalVar subroutines,
      // but with conflicting names.  Because an IRModule must have
      // enough names for each GlobalVar, even if they are not
      // publicly exposed, one of the GlobalVars must be replaced.
      // Replacing the GlobalVar here, when the conflict is first
      // identified, minimizes the size of the `relax::Expr` that
      // must be updated with `GlobalVarReplacer`.
      GlobalVar new_gvar = gvar;
      new_gvar.CopyOnWrite()->name_hint = new_name;
      gvar_rewrites.Set(gvar, new_gvar);
      new_subroutines.Set(new_gvar, func);
    } else {
      new_subroutines.Set(gvar, func);
    }
  }

  for (auto [var, expr] : other.variable_rewrites) {
    if (gvar_rewrites.size()) {
      expr = GlobalVarReplacer(gvar_rewrites)(expr);
    }
    variable_rewrites.Set(var, expr);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.dpl.PatternMatchingRewriterFromPattern",
           [](DFPattern pattern,
              ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func) {
             return PatternMatchingRewriter::FromPattern(pattern, func);
           })
      .def("relax.dpl.PatternMatchingRewriterFromModule",
           [](IRModule mod) { return PatternMatchingRewriter::FromModule(mod); })
      .def("relax.dpl.PatternMatchingRewriterApply",
           [](PatternMatchingRewriter rewriter,
              ffi::Variant<Expr, IRModule> obj) -> ffi::Variant<Expr, IRModule> {
             if (auto expr = obj.as<Expr>()) {
               return rewriter(expr.value());
             } else if (auto mod = obj.as<IRModule>()) {
               return rewriter(mod.value());
             } else {
               TVM_FFI_THROW(InternalError)
                   << "Unreachable: object does not contain either variant type";
             }
           });
}

RewriteSpec ExprPatternRewriterObj::RewriteBindings(const ffi::Array<Binding>& bindings) const {
  ffi::Map<Var, Expr> variable_rewrites;
  ffi::Map<Var, Expr> binding_lookup;
  for (const auto& binding : bindings) {
    auto bound_value = GetBoundValue(binding);
    if (auto new_expr = RewriteExpr(bound_value, binding_lookup)) {
      variable_rewrites.Set(binding->var, new_expr.value());
    } else {
      binding_lookup.Set(binding->var, bound_value);
    }
  }
  if (variable_rewrites.size()) {
    return RewriteSpec{variable_rewrites, new_subroutines};
  } else {
    return RewriteSpec();
  }
}

ffi::Optional<Expr> ExprPatternRewriterObj::RewriteExpr(const Expr& expr,
                                                        const ffi::Map<Var, Expr>& bindings) const {
  if (auto opt_matches = ExtractMatchedExpr(pattern, expr, bindings)) {
    auto matches = opt_matches.value();
    if (additional_bindings) {
      // Append any additional matches that from the unwrapped
      // `OrPattern`.  When matching against `pat = pat_lhs |
      // pat_rhs`, we call `ExtractMatchedExpr` on `pat_lhs` and
      // `pat_rhs` separately.  The top-level `pat` is never seen by
      // `ExtractMatchedExpr`, and must be re-added afterward.
      auto matched_expr = DFPatternMatcher::UnwrapBindings(expr, bindings);
      for (const auto& pat : additional_bindings.value()) {
        matches.Set(pat, matched_expr);
      }
    }

    ffi::Optional<Expr> rewritten_expr = func(expr, matches);
    if (rewritten_expr.has_value() && !rewritten_expr.same_as(expr)) {
      return rewritten_expr.value();
    }
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.dpl.PatternRewriter",
      [](DFPattern pattern,
         ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func) {
        return ExprPatternRewriter(pattern, func);
      });
}

ExprPatternRewriter::ExprPatternRewriter(
    DFPattern pattern,
    ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
    ffi::Optional<ffi::Array<DFPattern>> additional_bindings,
    ffi::Map<GlobalVar, BaseFunc> new_subroutines)
    : PatternMatchingRewriter(ffi::UnsafeInit{}) {
  auto node = ffi::make_object<ExprPatternRewriterObj>(pattern);
  node->func = std::move(func);
  node->additional_bindings = std::move(additional_bindings);
  node->new_subroutines = std::move(new_subroutines);
  data_ = std::move(node);
}

RewriteSpec OrRewriterObj::RewriteBindings(const ffi::Array<Binding>& bindings) const {
  auto lhs_match = lhs->RewriteBindings(bindings);
  if (!lhs_match) {
    // If no rewrites found on LHS, RHS is allowed to modify any
    // variable binding.
    return rhs->RewriteBindings(bindings);
  }

  // The LHS matched some subset of the bindings.  These
  // replacements may not be normalized expressions, so the RHS may
  // only replace variable bindings that haven't been modified by
  // the LHS.  Variable replacements from the RHS may still occur,
  // but will need to wait for the next round of
  // iterate-until-converged.
  ffi::Array<Binding> remaining_bindings;
  for (const auto& binding : bindings) {
    if (!lhs_match.variable_rewrites.count(binding->var)) {
      remaining_bindings.push_back(binding);
    }
  }

  if (remaining_bindings.empty()) {
    // Early bail-out, the RHS has no bindings available to rewrite.
    return lhs_match;
  }

  lhs_match.Append(rhs->RewriteBindings(remaining_bindings));
  return lhs_match;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dpl.OrRewriter",
                        [](PatternMatchingRewriter lhs, PatternMatchingRewriter rhs) {
                          return OrRewriter(lhs, rhs);
                        });
}

OrRewriter::OrRewriter(PatternMatchingRewriter lhs, PatternMatchingRewriter rhs)
    : PatternMatchingRewriter(ffi::UnsafeInit{}) {
  auto node = ffi::make_object<OrRewriterObj>(std::move(lhs), std::move(rhs));
  data_ = std::move(node);
}

RewriteSpec TupleRewriterObj::RewriteBindings(const ffi::Array<Binding>& bindings) const {
  TVM_FFI_ICHECK_LE(patterns.size(), 3)
      << "For performance reasons, "
      << "matching of implicit tuple patterns is currently limited"
      << " to tuples with 3 elements or fewer.";
  ffi::Map<Var, Expr> variable_rewrites = GenerateVariableRewrites(bindings);

  if (variable_rewrites.size()) {
    return RewriteSpec{variable_rewrites, new_subroutines};
  } else {
    return RewriteSpec();
  }
}

ffi::Map<Var, Expr> TupleRewriterObj::GenerateVariableRewrites(
    const ffi::Array<Binding>& bindings) const {
  ffi::Map<Var, Expr> rewrites;

  ffi::Map<Var, Expr> binding_lookup;

  std::vector<VarInfo> info_vec;

  std::unordered_map<Var, size_t> binding_index_lookup;

  // Initialize a vector of indices, each of which corresponds to a
  // potential match for a tuple element.
  //
  // \param tuple_index_of_current_expr The index for the most recent
  // binding.
  //
  // \param indices An output vector, into which indices will be
  // generated.
  //
  // \returns bool True if the indices could be initialized to a
  // potential match.  False, otherwise.
  auto initialize_indices = [&](size_t tuple_index_of_current_expr,
                                std::vector<size_t>& indices) -> bool {
    if (!info_vec.back().matches[tuple_index_of_current_expr]) {
      return false;
    }

    indices = std::vector<size_t>(patterns.size(), info_vec.size());

    indices[tuple_index_of_current_expr] = info_vec.size() - 1;

    for (size_t i_rev = 0; i_rev < indices.size(); i_rev++) {
      size_t i = indices.size() - i_rev - 1;
      if (indices[i] == info_vec.size() - 1) {
        continue;
      }

      auto binding_index = [&]() -> std::optional<size_t> {
        if (indices[i] == info_vec.size() - 1) {
          return info_vec.size() - 1;
        }

        for (size_t j_rev = 1; j_rev < info_vec.size(); j_rev++) {
          size_t j = info_vec.size() - j_rev - 1;
          if (info_vec[j].matches[i] && !info_vec[j].used &&
              (j + 1 >= indices.size() ||
               std::all_of(indices.begin() + (j + 1), indices.end(),
                           [j](size_t prev_binding_index) { return j != prev_binding_index; }))) {
            return j;
          }
        }

        return std::nullopt;
      }();

      if (binding_index.has_value()) {
        indices[i] = binding_index.value();
      } else {
        return false;
      }
    }

    return true;
  };

  auto decrement_indices = [&](std::vector<size_t>& indices) -> bool {
    TVM_FFI_ICHECK_EQ(indices.size(), patterns.size());

    // Step 1, find the first index that can be decremented, while
    // still generating a valid set of indices.
    size_t i_forward;
    for (i_forward = 0; i_forward < indices.size(); i_forward++) {
      if (indices[i_forward] == info_vec.size() - 1) {
        continue;
      }

      bool found_valid = false;
      size_t& index = indices[i_forward];
      while (index) {
        index--;
        if (info_vec[index].matches[i_forward] && !info_vec[index].used &&
            std::all_of(
                indices.begin() + (i_forward + 1), indices.end(),
                [index](size_t later_binding_index) { return index != later_binding_index; })) {
          found_valid = true;
          break;
        }
      }
      if (found_valid) {
        break;
      }
    }

    // Step 2, if we reached the end, then all indices were
    // decremented to zero without finding anything.  Return false to
    // indicate that we've reached the end.
    if (i_forward == indices.size()) {
      return false;
    }

    // Step 3, refill all indices that were decremented to zero before from 0 to
    for (size_t i = 0; i < i_forward; i++) {
      size_t i_backward = i_forward - (i + 1);
      if (indices[i_backward] == info_vec.size() - 1) {
        continue;
      }

      auto binding_index = [&]() -> std::optional<size_t> {
        for (size_t j_rev = 1; j_rev < info_vec.size(); j_rev++) {
          size_t j = info_vec.size() - j_rev - 1;
          if (info_vec[j].matches[i_backward] && !info_vec[j].used &&
              std::all_of(indices.begin() + (j + 1), indices.end(),
                          [j](size_t prev_binding_index) { return j != prev_binding_index; })) {
            return j;
          }
        }

        return std::nullopt;
      }();

      if (binding_index.has_value()) {
        indices[i_backward] = binding_index.value();
      } else {
        return false;
      }
    }

    return true;
  };

  for (size_t i_binding = 0; i_binding < bindings.size(); i_binding++) {
    const auto& binding = bindings[i_binding];

    auto expr = GetBoundValue(binding);

    binding_index_lookup[binding->var] = i_binding;

    info_vec.push_back(VarInfo{
        binding->var,
        expr,
        patterns.Map(
            [&](const DFPattern& pat) { return ExtractMatchedExpr(pat, expr, binding_lookup); }),
        std::unordered_set<Var>(),
        false,
    });

    auto new_match = [&]() -> std::optional<std::pair<std::vector<size_t>, std::vector<Expr>>> {
      std::vector<size_t> indices;
      for (size_t i = 0; i < patterns.size(); i++) {
        if (initialize_indices(patterns.size() - i - 1, indices)) {
          do {
            if (auto match = TryMatchByBindingIndex(info_vec, indices)) {
              return std::pair{indices, match.value()};
            }
          } while (decrement_indices(indices));
        }
      }
      return std::nullopt;
    }();

    if (new_match) {
      const auto& [indices, exprs] = new_match.value();
      TVM_FFI_ICHECK_EQ(indices.size(), exprs.size());
      for (size_t i = 0; i < indices.size(); i++) {
        TVM_FFI_ICHECK_LT(indices[i], info_vec.size());
        auto& info = info_vec[indices[i]];

        TVM_FFI_CHECK(!info.used, InternalError)
            << "Produced multiple replacements for variable " << info.var;

        rewrites.Set(info.var, exprs[i]);
        binding_lookup.erase(info.var);
        info.used = true;
      }
    } else {
      binding_lookup.Set(binding->var, expr);
    }

    for (const auto& prev_var : FreeVars(expr)) {
      if (auto it = binding_index_lookup.find(prev_var); it != binding_index_lookup.end()) {
        info_vec[it->second].downstream_usage.insert(binding->var);
      }
    }
  }

  return rewrites;
}

std::optional<std::vector<Expr>> TupleRewriterObj::TryMatchByBindingIndex(
    const std::vector<VarInfo>& info_vec, const std::vector<size_t>& indices) const {
  TVM_FFI_ICHECK_GE(indices.size(), 1);

  TVM_FFI_ICHECK_EQ(indices.size(), patterns.size());
  for (size_t i = 0; i < indices.size(); i++) {
    const auto& info = info_vec[indices[i]];
    if (info.used || !info.matches[i]) {
      return std::nullopt;
    }
  }

  ffi::Map<DFPattern, Expr> merged_matches = info_vec[indices[0]].matches[0].value();
  for (size_t i = 1; i < indices.size(); i++) {
    for (const auto& [pat, expr] : info_vec[indices[i]].matches[i].value()) {
      if (auto it = merged_matches.find(pat); it != merged_matches.end()) {
        if (!ffi::StructuralEqual()(expr, (*it).second)) {
          return std::nullopt;
        }
      } else {
        merged_matches.Set(pat, expr);
      }
    }
  }

  bool tuple_element_is_already_used_outside_of_matched_tuple = [&]() -> bool {
    std::unordered_set<Var> matched_vars;
    for (const auto& [pat, expr] : merged_matches) {
      if (auto opt = expr.as<Var>()) {
        matched_vars.insert(opt.value());
      }
    }

    for (size_t index : indices) {
      const auto& downstream_of_rewritten_var = info_vec[index].downstream_usage;

      for (const auto& uses_matched_var : downstream_of_rewritten_var) {
        if (!matched_vars.count(uses_matched_var)) {
          return true;
        }
      }
    }

    return false;
  }();
  if (tuple_element_is_already_used_outside_of_matched_tuple) {
    return std::nullopt;
  }

  auto full_tuple = [&]() -> relax::Expr {
    ffi::Array<Expr> fields;
    for (size_t index : indices) {
      fields.push_back(info_vec[index].expr);
    }
    return relax::Tuple(fields);
  }();

  auto opt_rewritten = func(full_tuple, merged_matches);
  if (!opt_rewritten) {
    return std::nullopt;
  }
  auto rewritten = opt_rewritten.value();

  if (rewritten.same_as(full_tuple)) {
    return std::nullopt;
  }

  std::vector<Expr> rewrites;
  if (auto inline_tuple = rewritten.as<TupleObj>()) {
    const auto& fields = inline_tuple->fields;
    TVM_FFI_ICHECK_EQ(fields.size(), indices.size())
        << "Expected to receive " << indices.size() << " values to replace TuplePattern with "
        << indices.size() << " fields, but received " << fields.size() << " values";
    rewrites = {fields.begin(), fields.end()};
  } else {
    for (size_t i = 0; i < indices.size(); i++) {
      rewrites.push_back(TupleGetItem(rewritten, i));
    }
  }
  return rewrites;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.dpl.TupleRewriter",
      [](ffi::Array<DFPattern> patterns,
         ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func) {
        return TupleRewriter(patterns, func);
      });
}

TupleRewriter::TupleRewriter(
    ffi::Array<DFPattern> patterns,
    ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
    ffi::Optional<ffi::Array<DFPattern>> additional_bindings,
    ffi::Map<GlobalVar, BaseFunc> new_subroutines)
    : PatternMatchingRewriter(ffi::UnsafeInit{}) {
  auto node = ffi::make_object<TupleRewriterObj>();
  node->patterns = std::move(patterns);
  node->func = std::move(func);
  node->additional_bindings = std::move(additional_bindings);
  node->new_subroutines = std::move(new_subroutines);
  data_ = std::move(node);
}

PatternMatchingRewriter PatternMatchingRewriter::FromPattern(
    DFPattern pattern,
    ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
    ffi::Optional<ffi::Array<DFPattern>> additional_bindings,
    ffi::Map<GlobalVar, BaseFunc> new_subroutines) {
  if (auto or_pattern = pattern.as<OrPatternObj>()) {
    auto new_additional_bindings = additional_bindings.value_or({});
    new_additional_bindings.push_back(pattern);
    return OrRewriter(PatternMatchingRewriter::FromPattern(
                          or_pattern->left, func, new_additional_bindings, new_subroutines),
                      PatternMatchingRewriter::FromPattern(
                          or_pattern->right, func, new_additional_bindings, new_subroutines));
  } else if (auto tuple_pattern = pattern.as<TuplePatternObj>()) {
    auto new_additional_bindings = additional_bindings.value_or({});
    new_additional_bindings.push_back(pattern);
    // If the Tuple appears as a Relax binding, apply it first.  As a
    // fallback, also check for implicit tuples.
    return OrRewriter(
        ExprPatternRewriter(pattern, func, additional_bindings, new_subroutines),
        TupleRewriter(tuple_pattern->fields, func, new_additional_bindings, new_subroutines));
  } else {
    return ExprPatternRewriter(pattern, func, additional_bindings, new_subroutines);
  }
}

PatternMatchingRewriter PatternMatchingRewriter::FromModule(IRModule mod) {
  Function func_pattern = [&]() {
    TVM_FFI_CHECK(mod->ContainGlobalVar("pattern"), KeyError)
        << "Expected module to contain 'pattern', "
        << "a Relax function defining the pattern to be matched, "
        << "but the module did not contain a 'pattern' function.";
    auto base_func = mod->Lookup("pattern");
    TVM_FFI_CHECK(base_func->IsInstance<FunctionObj>(), TypeError)
        << "Expected module to contain 'pattern', "
        << "a Relax function defining the pattern to be matched, "
        << "but the 'pattern' function was of type " << base_func->GetTypeKey() << ".";
    return base_func.as_or_throw<Function>();
  }();
  Function func_replacement = [&]() {
    TVM_FFI_CHECK(mod->ContainGlobalVar("replacement"), KeyError)
        << "Expected module to contain 'replacement', "
        << "a Relax function defining the replacement to be matched, "
        << "but the module did not contain a 'replacement' function.";
    auto base_func = mod->Lookup("replacement");
    TVM_FFI_CHECK(base_func->IsInstance<FunctionObj>(), TypeError)
        << "Expected module to contain 'replacement', "
        << "a Relax function defining the replacement to be made on a successful match, "
        << "but the 'replacement' function was of type " << base_func->GetTypeKey() << ".";
    return base_func.as_or_throw<Function>();
  }();

  ffi::Map<GlobalVar, BaseFunc> new_subroutines;
  for (const auto& [gvar, func] : mod->functions) {
    if (gvar->name_hint != "pattern" && gvar->name_hint != "replacement") {
      bool is_public = func->GetAttr(tvm::attr::kGlobalSymbol).has_value();
      TVM_FFI_CHECK(!is_public, ValueError)
          << "Expected module to have no publicly-exposed functions "
          << "other than 'pattern' and 'replacement'.  "
          << "However, function '" << gvar->name_hint << "' of type " << func->GetTypeKey()
          << " is publicly exposed.";
      new_subroutines.Set(gvar, func);
    }
  }

  auto ty_pattern = GetType(func_pattern);
  auto ty_replacement = GetType(func_replacement);
  TVM_FFI_CHECK(ffi::StructuralEqual()(ty_pattern, ty_replacement), ValueError)
      << "The pattern and replacement must have the same signature, "
      << "but the pattern has type " << ty_pattern << ", while the replacement has type "
      << ty_replacement;

  ffi::Array<DFPattern> param_wildcards;
  ffi::Map<Var, DFPattern> pattern_lookup;
  for (const auto& param : func_pattern->params) {
    WildcardPattern wildcard;
    param_wildcards.push_back(wildcard);
    pattern_lookup.Set(param, TypePattern(wildcard, GetType(param)));
  }

  std::function<DFPattern(Expr)> make_pattern = [&](Expr expr) -> DFPattern {
    if (auto var = expr.as<Var>()) {
      return pattern_lookup[var.value()];

    } else if (auto call = expr.as<CallObj>()) {
      auto op = make_pattern(call->op);
      auto args = call->args.Map(make_pattern);
      return CallPattern(op, args);

    } else if (auto tuple = expr.as<TupleObj>()) {
      auto fields = tuple->fields.Map(make_pattern);
      return TuplePattern(fields);

    } else if (auto tuple_get_item = expr.as<TupleGetItemObj>()) {
      auto tuple = make_pattern(tuple_get_item->tuple);
      return TupleGetItemPattern(tuple, tuple_get_item->index);

    } else if (auto op = expr.as<Op>()) {
      return ExprPattern(op.value());

    } else if (auto func = expr.as<ExternFuncObj>()) {
      return ExternFuncPattern(func->global_symbol);

    } else if (auto prim = expr.as<PrimExpr>()) {
      return TypePattern(WildcardPattern(), prim.value().ty());

    } else {
      TVM_FFI_THROW(TypeError) << "Cannot convert Relax expression of type " << expr->GetTypeKey()
                               << " into pattern-matching rule.";
    }
  };

  for (const auto& block : func_pattern->body->blocks) {
    for (const auto& binding : block->bindings) {
      auto value_pattern = make_pattern(GetBoundValue(binding));
      if (auto match_cast = binding.as<MatchCastObj>()) {
        value_pattern = TypePattern(value_pattern, match_cast->ty);
      }
      pattern_lookup.Set(binding->var, value_pattern);
    }
  }

  DFPattern top_pattern = make_pattern(func_pattern->body->body);

  ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> rewriter_func =
      [param_wildcards = std::move(param_wildcards),
       orig_func_replacement = std::move(func_replacement)](
          Expr expr, ffi::Map<DFPattern, Expr> matches) -> ffi::Optional<Expr> {
    auto func_replacement = CopyWithNewVars(orig_func_replacement);

    ffi::Array<BindingBlock> new_blocks;

    ffi::Array<Binding> wildcard_bindings;
    TVM_FFI_ICHECK_EQ(param_wildcards.size(), func_replacement->params.size());
    for (size_t i = 0; i < param_wildcards.size(); i++) {
      Expr matched_expr = matches[param_wildcards[i]];

      // Introduce an intermediate variable, to ensure that the
      // MatchCast's target will be a Var, even for expressions that
      // wouldn't normally be normalized into a variable.
      Var intermediate_var("intermediate_var", GetType(matched_expr));
      wildcard_bindings.push_back(VarBinding(intermediate_var, matched_expr));
      wildcard_bindings.push_back(MatchCast(func_replacement->params[i], intermediate_var,
                                            GetType(func_replacement->params[i])));
    }

    new_blocks.push_back(DataflowBlock(wildcard_bindings));

    for (const auto& block : func_replacement->body->blocks) {
      new_blocks.push_back(block);
    }

    return SeqExpr(new_blocks, func_replacement->body->body);
  };

  return PatternMatchingRewriter::FromPattern(top_pattern, rewriter_func, std::nullopt,
                                              new_subroutines);
}

ffi::Optional<ffi::Map<DFPattern, Expr>> ExtractMatchedExpr(
    DFPattern pattern, Expr expr, ffi::Optional<ffi::Map<Var, Expr>> bindings_opt) {
  auto bindings = bindings_opt.value_or({});
  DFPatternMatcher matcher(bindings);

  if (!matcher.Match(pattern, expr)) {
    return std::nullopt;
  }

  return matcher.GetMemo();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dpl.extract_matched_expr", ExtractMatchedExpr);
}

bool MatchExpr(DFPattern pattern, Expr expr, ffi::Optional<ffi::Map<Var, Expr>> bindings_opt) {
  return static_cast<bool>(ExtractMatchedExpr(pattern, expr, bindings_opt));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dpl.match_expr", MatchExpr);
}

/*!
 * \brief Apply pattern matching to each expression, replacing
 * matches with the output of a user-provided rewriter function.
 */
class PatternMatchingMutator : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  explicit PatternMatchingMutator(const PatternMatchingRewriterObj* rewriter)
      : rewriter_(rewriter) {}

  ffi::Map<GlobalVar, BaseFunc> GetNewSubroutines() const { return new_subroutines_; }

  Expr VisitExpr_(const SeqExprObj* seq) override {
    SeqExpr prev = ExprMutator::VisitExpr_(seq).as_or_throw<SeqExpr>();

    ffi::StructuralEqual struct_equal;

    while (auto opt = TryRewriteSeqExpr(prev)) {
      SeqExpr next = builder_->Normalize(opt.value()).as_or_throw<SeqExpr>();
      if (struct_equal(prev, next)) {
        break;
      }

      // Canonicalization may result in two previously-different
      // expressions being recognized as identical.  Elimination of
      // common subexpressions may result in trival var-to-var
      // bindings that can be canonicalized.  Therefore, iterate the
      // simplification steps until converged.
      while (true) {
        auto start_of_loop = next;
        next = CanonicalizeBindings(next).as_or_throw<SeqExpr>();
        next = EliminateCommonSubexpr(next).as_or_throw<SeqExpr>();
        next = RemoveAllUnused(next).as_or_throw<SeqExpr>();
        if (struct_equal(start_of_loop, next)) {
          break;
        }
      }

      if (struct_equal(prev, next)) {
        break;
      }

      prev = next;
    }

    return prev;
  }

  ffi::Optional<SeqExpr> TryRewriteSeqExpr(const SeqExpr& seq) {
    ffi::Array<BindingBlock> old_blocks = seq->blocks;

    // If the SeqExpr's output is not a variable, treat it as if it
    // were the last variable binding of the last block.  This
    // simplifies the special handling of the SeqExpr's body.
    ffi::Optional<Var> dummy_output_var = std::nullopt;
    if (!seq->body->IsInstance<VarObj>()) {
      dummy_output_var = Var("dummy_output_var", GetType(seq->body));
      VarBinding dummy_binding(dummy_output_var.value(), seq->body);

      auto last_block = [&]() {
        if (seq->blocks.size()) {
          auto last_block = old_blocks.back();
          old_blocks.pop_back();
          return last_block;
        } else {
          return BindingBlock(ffi::Array<Binding>{});
        }
      }();

      last_block.CopyOnWrite()->bindings.push_back(dummy_binding);
      old_blocks.push_back(last_block);
    }

    auto rewrite_block = [&](ffi::Array<Binding> orig_bindings) -> ffi::Array<Binding> {
      auto rewrites = rewriter_->RewriteBindings(orig_bindings);
      if (!rewrites) return orig_bindings;

      for (auto [gvar, func] : rewrites.new_subroutines) {
        new_subroutines_.Set(gvar, func);
      }

      auto bindings = orig_bindings.Map([&](Binding binding) -> Binding {
        if (auto new_expr = rewrites.variable_rewrites.Get(binding->var)) {
          if (auto match_cast = binding.as<MatchCastObj>()) {
            return MatchCast(binding->var, new_expr.value(), match_cast->ty);
          } else {
            return VarBinding(binding->var, new_expr.value());
          }
        } else {
          return binding;
        }
      });

      if (bindings.same_as(orig_bindings)) {
        return orig_bindings;
      }

      // The rewriter may have introduced additional dependencies
      // between computations.  Since pattern-matching only occurs
      // within blocks that may be re-ordered, these can be resolved
      // by performing a topological sort.
      bindings = TopologicalSort(bindings);

      return bindings;
    };

    // Utility function to return the rewrites that should be applied
    // to a given block.
    auto get_rewrites = [&](BindingBlock block) -> ffi::Array<Binding> {
      if (block.as<DataflowBlockObj>()) {
        // Early return for DataflowBlock.  Since neither control flow
        // nor impure functions are allowed within the dataflow block,
        // all bindings may be considered at the same time.
        return rewrite_block(block->bindings);
      }

      RewriteSpec rewrites;

      ffi::Array<Binding> collected_bindings;
      ffi::Array<Binding> finalized_bindings;

      auto handle_collected_rewrites = [&]() {
        if (collected_bindings.size()) {
          auto bindings = rewrite_block(collected_bindings);
          if (finalized_bindings.empty()) {
            finalized_bindings = bindings;
          } else {
            for (const auto& binding : bindings) {
              finalized_bindings.push_back(binding);
            }
          }
          collected_bindings.clear();
        }
      };

      for (const auto& binding : block->bindings) {
        auto value = GetBoundValue(binding);
        bool is_dataflow = (!value.as<IfExprObj>()) &&
                           (!(value.as<CallObj>() && IsImpureCall(value.as_or_throw<Call>())));
        if (is_dataflow) {
          // This binding satisfies the dataflow constraints.
          collected_bindings.push_back(binding);
        } else {
          // This binding does not satisfy the dataflow constraints.
          // Any operations prior to this binding should be checked
          // for pattern-match replacements.
          handle_collected_rewrites();
          finalized_bindings.push_back(binding);
        }
      }

      // Check for rewrites in dataflow operations after the last
      // non-dataflow segment.
      handle_collected_rewrites();

      return finalized_bindings;
    };

    // Utility function, check for and apply rewrites to a single
    // block.
    auto visit_block = [&](BindingBlock old_block) -> BindingBlock {
      auto new_bindings = get_rewrites(old_block);
      if (new_bindings.same_as(old_block->bindings)) {
        return old_block;
      }

      if (old_block.as<DataflowBlockObj>()) {
        builder_->BeginDataflowBlock();
      } else {
        builder_->BeginBindingBlock();
      }

      for (const auto& binding : new_bindings) {
        auto value = builder_->Normalize(GetBoundValue(binding));

        if (binding.as<VarBindingObj>()) {
          builder_->EmitNormalized(VarBinding(binding->var, value));
        } else if (auto match_cast = binding.as<MatchCastObj>()) {
          builder_->EmitNormalized(MatchCast(binding->var, value, match_cast->ty));
        } else {
          TVM_FFI_THROW(InternalError) << "Binding must be either VarBinding or MatchCast";
        }
      }
      return builder_->EndBlock();
    };

    auto new_blocks = old_blocks.Map(visit_block);
    if (old_blocks.same_as(new_blocks)) {
      return std::nullopt;
    }

    // Restore the body of the SeqExpr, if needed.
    auto new_body = [&]() -> Expr {
      if (dummy_output_var) {
        auto last_block = new_blocks.back();
        new_blocks.pop_back();

        auto last_binding = last_block->bindings.back();
        last_block.CopyOnWrite()->bindings.pop_back();
        TVM_FFI_ICHECK(dummy_output_var.same_as(last_binding->var));

        if (last_block->bindings.size()) {
          new_blocks.push_back(last_block);
        }

        return GetBoundValue(last_binding);
      } else {
        return seq->body;
      }
    }();

    return SeqExpr(new_blocks, new_body);
  }

 private:
  const PatternMatchingRewriterObj* rewriter_;
  ffi::Map<GlobalVar, BaseFunc> new_subroutines_;
};

Expr PatternMatchingRewriter::operator()(Expr expr) {
  PatternMatchingMutator mutator(get());
  auto new_expr = mutator(expr);
  auto new_subroutines = mutator.GetNewSubroutines();
  TVM_FFI_ICHECK_EQ(new_subroutines.size(), 0)
      << "If PatternMatchingRewriter provides subroutines, "
      << "then it must be applied to an entire IRModule.  "
      << "However, PatternMatchingRewriter produced subroutines "
      << [&]() -> ffi::Array<GlobalVar> {
    std::vector<GlobalVar> vec;
    for (const auto& [gvar, func] : new_subroutines) {
      vec.push_back(gvar);
    }
    std::sort(vec.begin(), vec.end(),
              [](const GlobalVar& a, const GlobalVar& b) { return a->name_hint < b->name_hint; });
    return vec;
  }() << "when applied to "
      << "Relax expression of type " << expr->GetTypeKey();
  return new_expr;
}

IRModule PatternMatchingRewriterObj::operator()(IRModule mod,
                                                const tvm::transform::PassContext& pass_ctx) const {
  PatternMatchingMutator mutator(this);

  IRModule updates;
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto func = base_func.as<Function>()) {
      auto rewritten = mutator(func.value()).as_or_throw<Function>();
      if (!rewritten.same_as(base_func)) {
        updates->Add(gvar, rewritten);
      }
    }
  }

  if (updates->functions.size()) {
    auto write_ptr = mod.CopyOnWrite();
    write_ptr->Update(updates);
    write_ptr->Update(IRModule(mutator.GetNewSubroutines()));
  }

  return mod;
}
tvm::transform::PassInfo PatternMatchingRewriterObj::Info() const {
  return tvm::transform::PassInfo(0, "PatternMatchingRewriter");
}

Function RewriteCall(const DFPattern& pat,
                     ffi::TypedFunction<Expr(Expr, ffi::Map<DFPattern, Expr>)> rewriter,
                     Function func) {
  return PatternMatchingRewriter::FromPattern(pat, rewriter)(func).as_or_throw<Function>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dpl.rewrite_call", RewriteCall);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  PatternMatchingRewriterObj::RegisterReflection();
  ExprPatternRewriterObj::RegisterReflection();
  OrRewriterObj::RegisterReflection();
  TupleRewriterObj::RegisterReflection();
}

}  // namespace relax
}  // namespace tvm
