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
 * \file src/relax/ir/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relax.
 */

#include "dataflow_matcher.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../transform/utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

using tvm::sym::Analyzer;
using tvm::sym::AnalyzerObj;

/*!
 * \brief Match the attributes of an object.
 * \param attrs The attributes of the object.
 * \param attributes The attributes to match.
 * \return True if the attributes match, false otherwise.
 */
bool MatchAttrs(const Any& attrs, const ffi::Map<ffi::String, ffi::Any>& attributes) {
  // TODO(tqchen): consider lift to common utils
  if (auto* dict_attrs = attrs.as<DictAttrsObj>()) {
    for (auto kv : attributes) {
      auto attr_name = kv.first;
      auto attr_value = kv.second;
      if (dict_attrs->dict.count(attr_name)) {
        if (!ffi::StructuralEqual()(attr_value, dict_attrs->dict[attr_name])) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  } else {
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(attrs.type_index());
    // use new reflection mechanism
    TVM_FFI_ICHECK(type_info->metadata != nullptr)
        << "Type " << attrs.GetTypeKey() << " do not have reflection metadata";
    size_t match_count = 0;
    bool success = true;
    const ffi::Object* obj = attrs.cast<const ffi::Object*>();
    ffi::reflection::ForEachFieldInfoWithEarlyStop(
        type_info, [&](const TVMFFIFieldInfo* field_info) {
          ffi::String field_name(field_info->name);
          if (attributes.count(field_name)) {
            ffi::reflection::FieldGetter field_getter(field_info);
            ffi::Any field_value = field_getter(obj);
            if (!ffi::StructuralEqual()(attributes[field_name], field_value)) {
              success = false;
              return true;
            }
            match_count++;
          }
          return false;
        });
    return success && match_count == attributes.size();
  }
}

// Pattern Matcher
bool DFPatternMatcher::Match(const DFPattern& pattern, const Expr& expr) {
  memo_.clear();
  matched_nodes_.clear();
  return VisitDFPattern(pattern, expr);
}

Expr DFPatternMatcher::UnwrapBindings(Expr expr, const ffi::Map<Var, Expr>& var2val) {
  auto unwrap = [&](Expr expr) -> ffi::Optional<Expr> {
    // Unwrap variables into the value to which they are bound.
    if (var2val.size()) {
      if (const VarObj* var = expr.as<VarObj>()) {
        if (auto may = var2val.Get(ffi::GetRef<Var>(var))) {
          return may.value();
        }
      }
    }

    // Unwrap SeqExpr with no bindings.  These can occur due to Relax
    // IR constraints for the bodies of Function and IfExpr nodes.
    if (auto seq = expr.as<SeqExprObj>()) {
      if (seq->blocks.empty()) {
        return seq->body;
      }
    }

    return std::nullopt;
  };

  while (auto unwrapped = unwrap(expr)) {
    expr = unwrapped.value();
  }

  return expr;
}

void DFPatternMatcher::ClearMap(size_t watermark) {
  for (size_t i = watermark; i < matched_nodes_.size(); ++i) {
    memo_.erase(matched_nodes_[i]);
  }
  matched_nodes_.erase(matched_nodes_.begin() + watermark, matched_nodes_.end());
}

bool DFPatternMatcher::VisitDFPattern(const DFPattern& pattern, const Expr& expr0) {
  TVM_FFI_ICHECK(pattern.defined()) << "Null pattern found when matching against " << expr0;

  auto expr = UnwrapBindings(expr0, var2val_);
  if (memoize_ && memo_.count(pattern)) {
    return expr.same_as(memo_.at(pattern));
  } else {
    PrimExpr cached_condition = symbolic_expr_condition_;
    size_t watermark = matched_nodes_.size();
    bool out = DFPatternFunctor::VisitDFPattern(pattern, expr);
    if (out) {
      memo_.insert_or_assign(pattern, expr);
      matched_nodes_.push_back(pattern);
    } else {
      ClearMap(watermark);
      symbolic_expr_condition_ = cached_condition;
    }
    return out;
  }
}

bool DFPatternMatcher::VisitDFPattern_(const OrPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  return VisitDFPattern(op->left, expr) || VisitDFPattern(op->right, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const AndPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  return VisitDFPattern(op->left, expr) && VisitDFPattern(op->right, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const NotPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  return !VisitDFPattern(op->reject, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const AttrPatternObj* attr_pattern, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  bool matches = VisitDFPattern(attr_pattern->pattern, expr);
  if (!matches) return matches;
  VLOG(1) << "considering AttrPatternObj at:\n" << expr;
  auto attributes = attr_pattern->attrs.as<DictAttrsObj>()->dict;
  if (const auto* op_node = expr.as<OpObj>()) {
    Op op = ffi::GetRef<Op>(op_node);
    for (auto kv : attributes) {
      auto attr_name = kv.first;
      auto attr_value = kv.second;
      if (Op::HasAttrMap(attr_name)) {
        auto op_map = Op::GetAttrMap<ffi::Any>(attr_name);
        if (op_map.count(op)) {
          matches &= ffi::StructuralEqual()(attr_value, op_map[op]);
        } else {
          matches = false;
        }
      } else {
        matches = false;
      }
    }
  } else if (auto* op = expr.as<CallObj>()) {
    matches &= MatchAttrs(op->attrs, attributes);
  } else if (auto* op = expr.as<FunctionObj>()) {
    matches = true;
    for (auto kv : attributes) {
      if (matches && op->attrs->dict.count(kv.first)) {
        matches &= ffi::StructuralEqual()(kv.second, op->attrs->dict[kv.first]);
      } else {
        matches = false;
        break;
      }
    }
  } else {
    matches = false;
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const CallPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  // utilities
  auto get_op_node = [](const CallPatternObj* op) -> const tvm::OpObj* {
    if (op) {
      if (auto* expr_pattern = op->op.as<ExprPatternObj>()) {
        return expr_pattern->expr.as<OpObj>();
      }
    }
    return nullptr;
  };
  auto is_pattern_op = [&get_op_node](const CallPatternObj* op, std::string op_type) {
    if (const auto* op_node = get_op_node(op)) {
      if (op_node->name == op_type) {
        return true;
      }
    }
    return false;
  };
  auto is_expr_op = [](const Expr& expr, std::string op_type) {
    if (const auto* call_node = expr.as<CallObj>()) {
      if (const auto* op_node = call_node->op.as<OpObj>()) {
        if (op_node->name == op_type) {
          return true;
        }
      }
    }
    return false;
  };

  // logic
  auto watermark = matched_nodes_.size();
  if (const auto* call_node = expr.as<CallObj>()) {
    auto matches_op = VisitDFPattern(op->op, call_node->op);
    if (matches_op) {
      auto watermark2 = matched_nodes_.size();

      auto match_args = [this, &watermark2](const ffi::Array<DFPattern>& pattern_args,
                                            auto expr_begin, auto expr_end) {
        bool matches = true;
        auto pattern_it = pattern_args.begin();
        auto expr_it = expr_begin;
        if (pattern_args.defined()) {
          while (matches && pattern_it != pattern_args.end())
            matches &= VisitDFPattern(*(pattern_it++), *(expr_it++));
        }
        if (!matches) ClearMap(watermark2);
        return matches;
      };

      const size_t n_arg_pattern = op->args.size();
      const size_t n_arg_expr = call_node->args.size();
      // if allow variable args, #pattern must >= #expr.
      if (op->varg_default_wildcard && n_arg_expr < n_arg_pattern) return false;
      // if variable args are not allowed, #pattern must == #expr.
      if (!op->varg_default_wildcard && n_arg_expr != n_arg_pattern) return false;

      // Standard case
      if (match_args(op->args, call_node->args.begin(), call_node->args.end())) return true;

      // Commutative Matching.
      if (const OpObj* op_node = call_node->op.as<OpObj>()) {
        if ((op_node->name == "relax.add") || (op_node->name == "relax.multiply")) {
          if (match_args(op->args, call_node->args.rbegin(), call_node->args.rend())) {
            return true;
          }
        }
      }
    } else {
      ClearMap(watermark);
      // associate divide/multiply
      if (is_pattern_op(op, "relax.divide")) {
        if (const auto* arg_node = op->args[0].as<CallPatternObj>()) {
          if (is_pattern_op(arg_node, "relax.multiply") && is_expr_op(expr, "relax.multiply") &&
              (is_expr_op(call_node->args[0], "relax.divide") ||
               is_expr_op(call_node->args[1], "relax.divide"))) {
            bool out = false;
            for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
              auto div = CallPattern(op->op, {arg_node->args[arg_id], op->args[1]});
              auto mul = CallPattern(arg_node->op, {arg_node->args[(arg_id + 1) % 2], div});
              out = VisitDFPattern(mul, expr);
              if (out) {
                return true;
              } else {
                ClearMap(watermark);
              }
            }
            return out;
          }
        }
      }
      if (is_pattern_op(op, "relax.multiply")) {
        // associate multiply/divide
        for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
          if (auto* arg_node = op->args[arg_id].as<CallPatternObj>()) {
            if (is_pattern_op(arg_node, "relax.divide") && is_expr_op(expr, "relax.divide") &&
                (is_expr_op(call_node->args[0], "relax.multiply") ||
                 is_expr_op(call_node->args[1], "relax.multiply"))) {
              auto mul = CallPattern(op->op, {arg_node->args[0], op->args[(arg_id + 1) % 2]});
              auto div = CallPattern(arg_node->op, {mul, arg_node->args[1]});
              return VisitDFPattern(div, expr);
            }
          }
        }
      }
    }
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const ExprPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  return ffi::StructuralEqual()(op->expr, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const FunctionPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  bool matches = false;
  if (const auto* func = expr.as<FunctionObj>()) {
    matches = true;
    if (op->params.defined()) {
      size_t i = 0;
      if (op->params.size() == func->params.size()) {
        while (matches && i < op->params.size()) {
          matches &= VisitDFPattern(op->params[i], func->params[i]);
          ++i;
        }
      } else {
        matches = false;
      }
    }
    if (matches) {
      matches &= VisitDFPattern(op->body, func->body);
    }
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const TupleGetItemPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  if (const auto* tuple_get_item_node = expr.as<TupleGetItemObj>()) {
    return (op->index == -1 || op->index == tuple_get_item_node->index) &&
           VisitDFPattern(op->tuple, tuple_get_item_node->tuple);
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const TuplePatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  bool matches = false;
  if (const auto* tuple_node = expr.as<TupleObj>()) {
    matches = true;
    if (op->fields.size() == tuple_node->fields.size()) {
      size_t i = 0;
      while (matches && i < op->fields.size()) {
        matches &= VisitDFPattern(op->fields[i], tuple_node->fields[i]);
        ++i;
      }
    } else {
      matches = false;
    }
  }
  return matches;
}

bool DFPatternMatcher::TryUnorderedMatch(size_t idx, const tvm::ffi::Array<DFPattern> patterns,
                                         const tvm::ffi::Array<Expr> fields,
                                         std::vector<int8_t>& match_cache,
                                         std::vector<bool>& matched) {
  if (idx >= patterns.size()) return true;
  constexpr int8_t kUnknown = -1;
  auto this_pattern = patterns[idx];
  for (size_t i = 0; i < fields.size(); ++i) {
    if (matched[i]) continue;
    const size_t table_idx = idx * fields.size() + i;
    match_cache[table_idx] =
        kUnknown ? VisitDFPattern(this_pattern, fields[i]) : match_cache[table_idx];
    if (match_cache[table_idx]) {
      // continue to match the rest;
      matched[i] = true;
      if (TryUnorderedMatch(idx + 1, patterns, fields, match_cache, matched)) return true;
      matched[i] = false;
    }
  }

  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const UnorderedTuplePatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);

  if (const auto* tuple_node = expr.as<TupleObj>()) {
    if (op->fields.size() == tuple_node->fields.size()) {
      constexpr int8_t kUnknown = -1;
      TVM_FFI_ICHECK_LE(op->fields.size(), std::numeric_limits<uint8_t>::max())
          << "Too many fields!";
      // dynamic programming.
      std::vector<int8_t> match_cache(op->fields.size() * op->fields.size(), kUnknown);
      std::vector<bool> field_match_bitmap(op->fields.size(), false);
      return TryUnorderedMatch(0, op->fields, tuple_node->fields, match_cache, field_match_bitmap);
    }
  }
  return false;
}

// Check accumulated shape equalities without changing the constraints retained for backtracking.
// Bindings live only in this check, and use variable identity rather than printed names.
static bool ShapeConstraintsConsistent(const PrimExpr& condition) {
  std::vector<PrimExpr> constraints{condition};
  for (size_t i = 0; i < constraints.size();) {
    if (const auto* conjunction = constraints[i].as<prim::AndObj>()) {
      PrimExpr lhs = conjunction->a;
      PrimExpr rhs = conjunction->b;
      constraints[i] = std::move(lhs);
      constraints.push_back(std::move(rhs));
    } else {
      ++i;
    }
  }

  Analyzer analyzer;
  // Follow every possible local alias before binding, including chains longer than
  // the analyzer's default two simplification passes.
  int simplify_steps = 2 * static_cast<int>(constraints.size());
  // Each successful binding eliminates a variable from an equality. Revisit earlier
  // equalities that may become directly bindable after a later constraint is processed.
  for (size_t iteration = 0; iteration < constraints.size(); ++iteration) {
    bool changed = false;
    for (const PrimExpr& constraint : constraints) {
      PrimExpr simplified = analyzer->Simplify(constraint, simplify_steps);
      if (prim::IsZero(simplified)) return false;
      if (const auto* equality = simplified.as<prim::EQObj>()) {
        PrimExpr lhs = equality->a;
        PrimExpr rhs = equality->b;
        if (!lhs.as<tvm::VarObj>()) std::swap(lhs, rhs);
        if (const auto* var = lhs.as<tvm::VarObj>();
            var && SideEffect(rhs) <= CallEffectKind::kPure) {
          auto variables = tvm::UndefinedVars(rhs);
          if (std::none_of(variables.begin(), variables.end(),
                           [var](const tvm::Var& used) { return used.get() == var; })) {
            auto context = analyzer->GetConstraintContext();
            context->Append(sym::BindingConstraint(ffi::GetRef<tvm::Var>(var), rhs));
            analyzer->SyncConstraintContext(context);
            changed = true;
          }
        }
      }
    }
    if (!changed) break;
  }
  // Composite dimensions may not admit a variable binding. Check each remaining
  // equality against its peers, so repeated expressions also detect contradictions.
  for (PrimExpr& constraint : constraints) {
    constraint = analyzer->Simplify(constraint, simplify_steps);
    if (prim::IsZero(constraint)) return false;
  }
  for (size_t i = 0; i < constraints.size(); ++i) {
    if (prim::IsOne(constraints[i])) continue;
    PrimExpr others = IntImm::Bool(true);
    for (size_t j = 0; j < constraints.size(); ++j) {
      if (i != j) others = others && constraints[j];
    }
    auto saved_context = analyzer->GetConstraintContext();
    auto context = saved_context->Snapshot();
    context->Append(sym::PredicateConstraint(others));
    analyzer->SyncConstraintContext(context);
    bool inconsistent = prim::IsZero(analyzer->Simplify(constraints[i], simplify_steps));
    analyzer->SyncConstraintContext(saved_context);
    if (inconsistent) return false;
  }
  return true;
}

bool DFPatternMatcher::VisitDFPattern_(const TypePatternObj* op, const Expr& expr0) {
  if (!VisitDFPattern(op->pattern, expr0)) {
    return false;
  }

  auto expr = UnwrapBindings(expr0, var2val_);
  auto expr_ty = GetType(expr);

  PrimExpr new_constraint = TypeBaseCheckPrecondition(op->ty, expr_ty);
  if (auto* as_int = new_constraint.as<IntImmObj>()) {
    return static_cast<bool>(as_int->value);
  }

  symbolic_expr_condition_ = analyzer_->Simplify(symbolic_expr_condition_ && new_constraint);

  if (auto* as_int = symbolic_expr_condition_.as<IntImmObj>()) {
    return static_cast<bool>(as_int->value);
  } else {
    return ShapeConstraintsConsistent(symbolic_expr_condition_);
  }
}

static bool ShapeEqual(AnalyzerObj* analyzer, const ffi::Array<PrimExpr>& lhs,
                       const ffi::Array<PrimExpr>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i)
    if (!tvm::prim::IsOne(analyzer->Simplify(lhs[i] == rhs[i]))) return false;
  return true;
}

bool DFPatternMatcher::VisitDFPattern_(const ShapePatternObj* op, const Expr& expr) {
  // no need to jump, as var.shape == value.shape
  if (const auto* tinfo = GetTypeAs<TensorTypeObj>(expr)) {
    if (const ShapeExprObj* shape_expr = tinfo->shape.as<ShapeExprObj>()) {
      return ShapeEqual(analyzer_.get(), op->shape, shape_expr->values) &&
             VisitDFPattern(op->pattern, expr);
    }
  }
  return false;
}

std::tuple<PrimExpr, bool> SameShapeConstraintObj::AsCondition(
    std::function<ffi::Optional<Var>(const DFPatternObj*)> match_state) const {
  ffi::Optional<ffi::Array<PrimExpr>> expected_shape;
  bool all_shapes_defined = true;

  // The expression that must be true in order
  PrimExpr all_dimensions_equal = IntImm::Bool(true);

  for (const auto& arg : args) {
    if (auto opt_var = match_state(arg.get())) {
      auto var = opt_var.value();
      auto opt_var_shape = [&]() -> ffi::Optional<ffi::Array<PrimExpr>> {
        auto ty = GetType(var);
        if (auto tensor = ty.as<TensorTypeObj>()) {
          return tensor->GetShape();
        } else if (auto shape_expr = ty.as<ShapeTypeObj>()) {
          return shape_expr->values;
        } else {
          return std::nullopt;
        }
      }();

      if (!opt_var_shape.has_value()) {
        // The pattern has matched to something without a shape.
        // Therefore, it cannot have the same shape as something else.
        return {PrimExpr(IntImm::Bool(false)), true};
      }
      auto var_shape = opt_var_shape.value();

      if (expected_shape.has_value()) {
        auto prev_shape = expected_shape.value();
        if (prev_shape.size() == var_shape.size()) {
          // The dimensionalities match, so build up the expression
          // that must be true for the shapes to be equivalent.
          for (size_t i = 0; i < prev_shape.size(); i++) {
            all_dimensions_equal = all_dimensions_equal && (var_shape[i] == prev_shape[i]);
          }

        } else {
          // The shapes have different dimensionality.  No need to
          // perform potentially-expensive simplifications, because
          // the dimensions do not match.
          return {PrimExpr(IntImm::Bool(false)), true};
        }

      } else {
        // This is the first pattern with a known match.  Store the
        // shape so it can be compared against later shapes.
        expected_shape = var_shape;
      }

    } else {
      // Missing an argument, so the constraint will either return
      // std::nullopt or false at this point.  However, delay the return of
      // std::nullopt until the end of the function, because we'd rather
      // return "false" if it possible to do so.
      all_shapes_defined = false;
    }
  }

  return {all_dimensions_equal, all_shapes_defined};
}

bool DFPatternMatcher::VisitDFPattern_(const PrimArrPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  if (const ShapeExprObj* shape_expr = expr.as<ShapeExprObj>())
    return ShapeEqual(analyzer_.get(), op->fields, shape_expr->values);
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const DataTypePatternObj* op, const Expr& expr) {
  // no need to jump, as var.dtype == value.dtype
  auto expr_ty = expr.as<ExprObj>()->ty;
  if (const TensorTypeObj* tensor_ty = expr_ty.as<TensorTypeObj>()) {
    return op->dtype == tensor_ty->dtype.value()->dtype && VisitDFPattern(op->pattern, expr);
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const VarPatternObj* op, const Expr& expr) {
  // We don't jump for var pattern, as there's no need to access its value to judge it.
  if (const auto* var_node = expr.as<VarObj>()) {
    // "" means any name.
    return "" == op->name_hint() || op->name_hint() == var_node->name;
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const ExternFuncPatternObj* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  if (const auto* extern_fn = expr.as<ExternFuncObj>()) {
    return "" == op->global_symbol() || op->global_symbol() == extern_fn->global_symbol;
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const ConstantPatternObj* op, const Expr& expr0) {
  // constants can be binded to relax.Var as well.
  auto expr = UnwrapBindings(expr0, var2val_);
  auto* constant = expr.as<GenericConstObj>();
  return constant && constant->value.as<ffi::TensorObj>();
}

bool DFPatternMatcher::VisitDFPattern_(const DataflowVarPatternObj* op, const Expr& expr) {
  // DataflowVar is inherented from Var, so dispatch it to VarPattern.
  return expr->IsInstance<DataflowVarObj>() &&
         VisitDFPattern_(static_cast<const VarPatternObj*>(op), expr);
}

bool DFPatternMatcher::VisitDFPattern_(const GlobalVarPatternObj* op, const Expr& expr) {
  // GlobalVarPattern is not inherited from Var, so we need to handle it separately.
  if (const auto* var_node = expr.as<GlobalVarObj>()) {
    std::string pat = std::string(op->name_hint());
    std::string var_name = std::string(var_node->name_hint);
    return pat.empty() || var_name.find(pat) != std::string::npos;
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const WildcardPatternObj* op, const Expr& expr) {
  return true;
}

}  // namespace relax
}  // namespace tvm
