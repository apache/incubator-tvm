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
 * \file src/relax/ir/dataflow_block_rewriter.cc
 * \brief A transform to match a Relax DataflowBlock and rewrite
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/sym/analyzer.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dataflow_matcher.h"
#include "dataflow_rewriter.h"

namespace tvm {
namespace relax {

class MatcherUseDefAnalysis : public relax::ExprVisitor {
 public:
  std::vector<const VarObj*> vars;
  std::map<const VarObj*, std::vector<const VarObj*>> def2use;
  // caller -> callee table.
  std::map<const VarObj*, std::vector<const VarObj*>> caller2callees;

  const VarObj* cur_user_ = nullptr;

  void VisitBinding_(const VarBindingObj* binding) override {
    // init
    cur_user_ = binding->var.get();
    this->VisitVarDef(binding->var);
    this->VisitExpr(binding->value);
    cur_user_ = nullptr;
  }

  void VisitExpr_(const VarObj* op) override {
    if (nullptr == cur_user_) return;

    auto check_and_push = [](std::vector<const VarObj*>& vec, const VarObj* var) {
      if (std::find(vec.begin(), vec.end(), var) == vec.end()) {
        vec.push_back(var);
      }
    };

    check_and_push(def2use[op], cur_user_);
    check_and_push(vars, op);

    caller2callees[cur_user_].push_back(op);
  }
};

struct PNode {
  const DFPatternObj* ptr;
  std::vector<std::pair<PNode*, const std::vector<PairCons>&>> children;
  std::vector<std::pair<PNode*, const std::vector<PairCons>&>> parents;
};

struct RNode {
  const VarObj* ptr;
  std::vector<RNode*> children;
  std::vector<RNode*> parents;
};

struct MatchState {
  void add(const PNode* p, const RNode* r) {
    match_p_r[p] = r;
    match_r_p[r] = p;
  }

  void add(const DFConstraintObj* constraint) { validated_constraints_.insert(constraint); }

  void add(MatchState&& other) {
    match_p_r.merge(std::move(other.match_p_r));
    match_r_p.merge(std::move(other.match_r_p));
    validated_constraints_.merge(other.validated_constraints_);
  }

  const VarObj* matched(const PNode* p) const {
    if (auto it = match_p_r.find(p); it != match_p_r.end()) {
      return it->second->ptr;
    }
    return nullptr;
  }

  const DFPatternObj* matched(const RNode* r) const {
    if (auto it = match_r_p.find(r); it != match_r_p.end()) {
      return it->second->ptr;
    }
    return nullptr;
  }

  const VarObj* matched(const PNode& p) const { return matched(&p); }
  const DFPatternObj* matched(const RNode& r) const { return matched(&r); }

  bool is_validated(const DFConstraintObj* constraint) const {
    return validated_constraints_.count(constraint);
  }

 private:
  std::unordered_map<const PNode*, const RNode*> match_p_r;
  std::unordered_map<const RNode*, const PNode*> match_r_p;
  std::unordered_set<const DFConstraintObj*> validated_constraints_;
};

/**
 * \brief This method try to match a real node and a pattern node along with its neighbors.
 */
static std::optional<MatchState> TryMatch(const PNode& p, const RNode& r,
                                          const MatchState& current_match, DFPatternMatcher* m,
                                          const MatcherUseDefAnalysis& ud_analysis) {
  if (!m->Match(ffi::GetRef<DFPattern>(p.ptr), ffi::GetRef<Var>(r.ptr))) return std::nullopt;

  MatchState new_match;

  new_match.add(&p, &r);

  // forward matching;
  for (const auto& [pchild, constraints] : p.children) {
    bool any_cons_sat = false;
    for (const auto& rchild : r.children) {
      if (new_match.matched(rchild)) {
        // The child variable is already matched to other child pattern in a previous iteration.
        continue;
      }
      if (auto v = current_match.matched(pchild); v && v != rchild->ptr) {
        // The child pattern is already matched to other variable in a earlier call to TryMatch.
        continue;
      }

      const auto& uses = ud_analysis.def2use.at(r.ptr);

      // check edge constraints.
      bool all_cons_pass = true;
      for (const auto& cons : constraints) {
        if (cons.type == PairCons::kOnlyUsedBy && uses.size() != 1) {
          all_cons_pass = false;
          break;
        }

        if (cons.index != -1) {
          const auto& callees = ud_analysis.caller2callees.at(rchild->ptr);
          if (callees.size() <= static_cast<size_t>(cons.index) || callees[cons.index] != r.ptr) {
            all_cons_pass = false;
            break;
          }
        }
      }
      if (!all_cons_pass || new_match.matched(pchild)) continue;
      any_cons_sat = true;

      if (auto match_rec = TryMatch(*pchild, *rchild, current_match, m, ud_analysis)) {
        new_match.add(pchild, rchild);
        new_match.add(std::move(*match_rec));
      }
    }
    if (!new_match.matched(pchild) || !any_cons_sat) return std::nullopt;
  }

  return new_match;
}

static std::optional<MatchState> TryValidate(
    const MatchState& current_match,
    const std::unordered_map<const DFPatternObj*, PNode>& pattern2node,
    const std::vector<DFConstraint>& validation_constraints, sym::AnalyzerObj* analyzer) {
  MatchState new_match;

  std::function<ffi::Optional<Var>(const DFPatternObj*)> query_match_state =
      [&pattern2node, &current_match](const DFPatternObj* pattern) -> ffi::Optional<Var> {
    auto it = pattern2node.find(pattern);
    TVM_FFI_ICHECK(it != pattern2node.end())
        << "DFConstraint attempted to access DFPattern " << ffi::GetRef<DFPattern>(pattern)
        << ", which does not appear in the PatternContext";
    const auto& p_node = it->second;
    if (auto ptr = current_match.matched(p_node)) {
      return ffi::GetRef<Var>(ptr);
    } else {
      return std::nullopt;
    }
  };

  for (const auto& constraint : validation_constraints) {
    if (!current_match.is_validated(constraint.get())) {
      auto [necessary_condition, is_sufficient] = constraint->AsCondition(query_match_state);

      necessary_condition = analyzer->Simplify(necessary_condition);
      const auto* known = necessary_condition.as<IntImmObj>();

      if (known && known->value != 0 && is_sufficient) {
        // The condition passes, and the expression provided is both
        // necessary and sufficient for the constraint to pass.  Mark
        // the constraint as passing, to avoid re-checking it unless
        // we backtrack.
        new_match.add(constraint.get());
      } else if (known && known->value == 0) {
        // The condition fails.  Even if additional information would
        // be required to pass a constraint, it may bail out early as
        // a failure (e.g. shape mismatch in the first two items out
        // of N shapes that must all match).
        return std::nullopt;
      } else if (is_sufficient) {
        // The condition depends on dynamic parameters.  In the
        // future, this may be exposed to the user as a condition for
        // optimization, or can be combined with the conditions
        // provided from other constraints.
        return std::nullopt;
      }
    }
  }

  return new_match;
}

static std::optional<MatchState> MatchTree(
    const MatchState& current_match, size_t current_root_idx,
    const std::unordered_map<const DFPatternObj*, PNode>& pattern2node,
    const std::unordered_map<const VarObj*, RNode>& var2node, DFPatternMatcher* matcher,
    const std::vector<DFPattern>& roots, const std::vector<DFConstraint>& validation_constraints,
    const MatcherUseDefAnalysis& ud_analysis, sym::AnalyzerObj* analyzer) {
  auto get_next_root = [&](size_t root_idx) -> const PNode* {
    // Look for the next unmatched root node.
    for (; root_idx < roots.size(); ++root_idx) {
      const auto& root = pattern2node.at(roots[root_idx].get());
      if (!current_match.matched(root)) {
        return &root;
      }
    }
    return nullptr;
  };

  const auto root = get_next_root(current_root_idx);

  if (!root) {
    // All root nodes have been matched
    return current_match;
  }

  MatchState new_match = current_match;

  for (const auto& var : ud_analysis.vars) {
    const RNode& r_node = var2node.at(var);
    if (new_match.matched(r_node)) continue;
    if (auto match = TryMatch(*root, r_node, new_match, matcher, ud_analysis)) {
      // Recursively try to match the next subtree.
      new_match.add(std::move(*match));
      if (auto validation =
              TryValidate(new_match, pattern2node, validation_constraints, analyzer)) {
        new_match.add(std::move(*validation));
        if (auto match_rec =
                MatchTree(new_match, current_root_idx + 1, pattern2node, var2node, matcher, roots,
                          validation_constraints, ud_analysis, analyzer)) {
          new_match.add(std::move(*match_rec));
          return new_match;
        }
      }
      // Recursive matching has failed, backtrack.
      new_match = current_match;
      continue;
    }
  }

  return std::nullopt;
}

ffi::Optional<ffi::Map<DFPattern, Var>> MatchGraph(const PatternContext& ctx,
                                                   const ffi::Array<Binding>& binding_arr,
                                                   const ffi::Map<Var, Expr>& bindings) {
  // TODO(@ganler): Handle non-may external use.
  TVM_FFI_ICHECK(ctx->allow_extern_use == PatternContextObj::kMay) << "Only kMay is supported yet.";
  DFPatternMatcher matcher(bindings);

  MatcherUseDefAnalysis ud_analysis;
  for (const auto& binding : binding_arr) {
    ud_analysis.VisitBinding(binding);
  }

  // First construct a graph of PNode and RNode.
  std::unordered_map<const VarObj*, RNode> var2node;
  var2node.reserve(bindings.size());

  for (const VarObj* cur_var : ud_analysis.vars) {
    const auto& uses = ud_analysis.def2use.at(cur_var);
    RNode& cur_node = var2node[cur_var];
    cur_node.ptr = cur_var;
    for (const VarObj* use : uses) {
      auto& use_node = var2node[use];
      use_node.ptr = use;
      cur_node.children.push_back(&use_node);
      use_node.parents.push_back(&cur_node);
    }
  }

  std::unordered_map<const DFPatternObj*, PNode> pattern2node;
  pattern2node.reserve(ctx->edge_constraints.size());

  for (const auto& def_pattern : ctx->src_ordered) {
    PNode& def_node = pattern2node[def_pattern.get()];
    const auto& uses = ctx->edge_constraints.at(def_pattern);
    def_node.ptr = def_pattern.get();
    def_node.children.reserve(uses.size());
    for (const auto& [use_pattern, cons] : uses) {
      PNode& use_node = pattern2node[use_pattern.get()];
      use_node.ptr = use_pattern.get();
      use_node.parents.emplace_back(&def_node, std::ref(cons));
      def_node.children.emplace_back(&use_node, std::ref(cons));
    }
  }

  std::vector<DFPattern> roots;
  for (const auto& pat : ctx->src_ordered) {
    if (pattern2node[pat.get()].parents.empty()) {
      roots.push_back(pat);
    }
  }

  if (roots.empty()) {
    return std::nullopt;
  }

  sym::Analyzer analyzer;
  auto match = MatchTree({}, 0, pattern2node, var2node, &matcher, roots,
                         ctx->validation_constraints, ud_analysis, analyzer.get());
  if (!match) {
    return std::nullopt;
  }

  ffi::Map<DFPattern, Var> ret;
  for (const auto& [pat, p_node] : pattern2node) {
    TVM_FFI_ICHECK(match->matched(p_node));
    ret.Set(ffi::GetRef<DFPattern>(pat), ffi::GetRef<Var>(match->matched(p_node)));
  }
  return ret;
}

ffi::Optional<ffi::Map<DFPattern, Var>> MatchGraph(const PatternContext& ctx,
                                                   const DataflowBlock& dfb) {
  return MatchGraph(ctx, dfb->bindings, AnalyzeVar2Value(dfb));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.dpl.match_dfb",
      [](const PatternContext& ctx, const DataflowBlock& dfb) { return MatchGraph(ctx, dfb); });
}

class PatternContextRewriterObj : public PatternMatchingRewriterObj {
 public:
  PatternContext pattern;
  ffi::TypedFunction<ffi::Map<Var, Expr>(ffi::Map<DFPattern, Var>, ffi::Map<Var, Expr>)>
      rewriter_func;

  RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PatternContextRewriterObj>()
        .def_ro("pattern", &PatternContextRewriterObj::pattern)
        .def_ro("rewriter_func", &PatternContextRewriterObj::rewriter_func);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.dpl.PatternContextRewriter", PatternContextRewriterObj,
                                    PatternMatchingRewriterObj);

 private:
  ffi::Optional<ffi::Map<Var, Expr>> MatchBindings(const ffi::Array<Binding>& bindings) const {
    ffi::Map<Var, Expr> var_lookup;
    for (const auto& binding : bindings) {
      var_lookup.Set(binding->var, GetBoundValue(binding));
    }

    if (auto matches = MatchGraph(pattern, bindings, var_lookup)) {
      ffi::Map<Var, Expr> replacements = rewriter_func(matches.value(), var_lookup);
      if (replacements.size()) {
        return replacements;
      }
    }

    return std::nullopt;
  }
};

class PatternContextRewriter : public PatternMatchingRewriter {
 public:
  PatternContextRewriter(
      PatternContext pattern,
      ffi::TypedFunction<ffi::Map<Var, Expr>(ffi::Map<DFPattern, Var>, ffi::Map<Var, Expr>)>
          rewriter_func);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PatternContextRewriter, PatternMatchingRewriter,
                                                PatternContextRewriterObj);
};

RewriteSpec PatternContextRewriterObj::RewriteBindings(const ffi::Array<Binding>& bindings) const {
  std::vector<Binding> remaining_bindings{bindings.begin(), bindings.end()};

  ffi::Map<Var, Expr> variable_rewrites;
  while (auto opt = MatchBindings(remaining_bindings)) {
    auto new_rewrites = opt.value();
    remaining_bindings.erase(std::remove_if(remaining_bindings.begin(), remaining_bindings.end(),
                                            [&new_rewrites](const Binding& binding) {
                                              return new_rewrites.count(binding->var);
                                            }),
                             remaining_bindings.end());
    for (const auto& [var, expr] : new_rewrites) {
      variable_rewrites.Set(var, expr);
    }
  }

  return RewriteSpec{variable_rewrites, {}};
}

PatternContextRewriter::PatternContextRewriter(
    PatternContext pattern,
    ffi::TypedFunction<ffi::Map<Var, Expr>(ffi::Map<DFPattern, Var>, ffi::Map<Var, Expr>)>
        rewriter_func)
    : PatternMatchingRewriter(ffi::UnsafeInit{}) {
  auto node = ffi::make_object<PatternContextRewriterObj>();
  node->pattern = std::move(pattern);
  node->rewriter_func = std::move(rewriter_func);
  data_ = std::move(node);
}

Function RewriteBindings(
    const PatternContext& ctx,
    ffi::TypedFunction<ffi::Map<Var, Expr>(ffi::Map<DFPattern, Var>, ffi::Map<Var, Expr>)> rewriter,
    Function func) {
  // return BlockPatternRewriter::Run(ctx, rewriter, func);
  return PatternContextRewriter(ctx, rewriter)(func).as_or_throw<Function>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dpl.rewrite_bindings", RewriteBindings);
}

TVM_FFI_STATIC_INIT_BLOCK() { PatternContextRewriterObj::RegisterReflection(); }

}  // namespace relax
}  // namespace tvm
