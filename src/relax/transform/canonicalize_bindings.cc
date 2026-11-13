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
 * \file src/relax/transform/canonicalize_bindings.cc
 * \brief Pass for simplifying modules by folding var bindings and match shape nodes.
 *        May include other forms of simplification in the future.
 *        Ideally should be used before constant folding and eliminating unused bindings.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relax/utils.h>
#include <tvm/tirx/stmt_functor.h>

#include "../op/op_common.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace {

class SymbolicVarCanonicalizer : public ExprMutator {
  class RuntimePrimVarCollector;

 public:
  static Expr Apply(Expr expr);

  PrimExpr VisitTypePrimExprField(const PrimExpr& expr) final {
    if (!canonicalize_shape_values_) return expr;
    return CanonicalizeShapeValue(expr);
  }

  Expr VisitExpr_(const CallObj* op) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
    static const Op& call_tir_with_grad_op = Op::Get("relax.call_tir_with_grad");
    bool is_call_tir = op->op.same_as(call_tir_op) || op->op.same_as(call_tir_inplace_op) ||
                       op->op.same_as(call_tir_with_grad_op);
    if (!is_call_tir || op->args.size() != 2 || op->ty_args.size() != 1) {
      return ExprMutator::VisitExpr_(op);
    }
    // The out_ty of a call_tir is checked against the argument types. Canonicalizing the
    // two independently can leave them mentioning different variables for the same value,
    // so the parts of the canonical out_ty that no longer follow from the arguments are
    // replaced by what the arguments imply.
    Expr new_op = this->VisitExpr(op->op);
    ffi::Array<Expr> new_args =
        op->args.Map([this](const Expr& arg) { return this->VisitExpr(arg); });
    Type out_ty = this->VisitExprDepTypeField(op->ty_args[0]);
    ffi::Optional<ffi::Array<int64_t>> inplace_indices;
    if (const auto* attrs = op->attrs.as<CallTIRInplaceAttrs>()) {
      inplace_indices = attrs->inplace_indices;
    }
    auto implied = InferCallTIROutputTypeFromArguments(GetType(new_args[0]), GetType(new_args[1]),
                                                       inplace_indices);
    if (implied.has_value()) {
      out_ty = ReconcileOutType(implied.value(), out_ty);
    }
    bool unchanged =
        new_op.same_as(op->op) && new_args.same_as(op->args) && out_ty.same_as(op->ty_args[0]);
    if (unchanged) {
      return ffi::GetRef<Expr>(op);
    }
    return Call(Type::Missing(), new_op, new_args, op->attrs, {out_ty}, op->loc);
  }

  /*!
   * \brief Keep each output of out_ty that follows from the implied type. For a tensor that
   * does not, take the implied shape and keep the dtype and vdevice, since the tirx::Function
   * signature carries no vdevice and erases the shape of an output with a dimension the
   * arguments do not determine.
   */
  static Type ReconcileOutType(const Type& implied, const Type& out_ty) {
    if (IsBaseOf(implied, out_ty)) {
      return out_ty;
    }
    const auto* implied_tuple = implied.as<TupleTypeObj>();
    const auto* out_tuple = out_ty.as<TupleTypeObj>();
    if (implied_tuple && out_tuple && implied_tuple->fields.size() == out_tuple->fields.size()) {
      ffi::Array<Type> fields;
      for (size_t i = 0; i < out_tuple->fields.size(); ++i) {
        fields.push_back(ReconcileOutType(implied_tuple->fields[i], out_tuple->fields[i]));
      }
      return TupleType(fields, out_tuple->loc);
    }
    const auto* implied_tensor = implied.as<TensorTypeObj>();
    const auto* out_tensor = out_ty.as<TensorTypeObj>();
    if (implied_tensor && out_tensor && implied_tensor->shape.has_value()) {
      return TensorType(implied_tensor->shape.value(), out_tensor->dtype, out_tensor->vdevice,
                        out_tensor->loc);
    }
    return implied;
  }

  Expr VisitExpr_(const ShapeExprObj* op) final {
    if (!canonicalize_shape_values_) return ffi::GetRef<Expr>(op);
    ffi::Array<PrimExpr> values =
        op->values.Map([this](const PrimExpr& value) { return CanonicalizeShapeValue(value); });
    if (values.same_as(op->values)) {
      return ffi::GetRef<Expr>(op);
    }
    return ShapeExpr(values, op->loc);
  }

  Expr VisitExpr_(const FunctionObj* func) override {
    auto cached = known_values_;
    auto cached_runtime_uses = runtime_prim_var_uses_;
    runtime_prim_var_uses_ = RuntimePrimVarCollector::Collect(func->body);
    auto output = ExprMutator::VisitExpr_(func);
    known_values_ = cached;
    runtime_prim_var_uses_ = cached_runtime_uses;
    return output;
  }

  void VisitBinding_(const MatchCastObj* binding) override {
    auto tir_var_map =
        InferSymbolicVarMap({{binding->var, binding->value}}, builder_->GetAnalyzer());
    bool has_runtime_use = false;
    for (const auto& [var, value] : tir_var_map) {
      if (var.same_as(binding->var)) continue;
      auto tir_var = var.as<PrimVar>();
      if (!tir_var) continue;
      has_runtime_use = has_runtime_use || runtime_prim_var_uses_.count(*tir_var);
      PrimExpr prim_expr = value.as_or_throw<PrimExpr>();
      if (auto it = known_values_.find(tir_var.value()); it != known_values_.end()) {
        TVM_FFI_CHECK(!builder_->GetAnalyzer()->CanProve(it->second.expr != prim_expr), ValueError)
            << "MatchCast statements must be consistent.  "
            << "However, the definition of Relax variable " << it->second.source->var
            << " implies that TIR variable " << tir_var.value() << " is " << it->second.expr
            << ", while the later definition of Relax variable " << binding->var
            << " instead implies that TIR variable " << tir_var.value() << " is " << prim_expr;
      } else {
        known_values_.insert_or_assign(tir_var.value(),
                                       KnownValue{prim_expr, ffi::GetRef<MatchCast>(binding)});
      }
    }
    // A MatchCast that defines a runtime-used symbolic variable cannot be folded away.
    // Preserve its symbolic pattern while still canonicalizing downstream shape annotations.
    bool cached = canonicalize_shape_values_;
    canonicalize_shape_values_ = !has_runtime_use;
    ExprMutator::VisitBinding_(binding);
    canonicalize_shape_values_ = cached;
  }

  Expr VisitExpr_(const IfExprObj* op) override {
    Expr guard = this->VisitExpr(op->cond);

    auto cached = known_values_;
    Expr true_b = this->VisitWithInnerScope(op->true_branch);
    known_values_ = cached;
    Expr false_b = this->VisitWithInnerScope(op->false_branch);
    known_values_ = cached;

    if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
        op->false_branch.same_as(false_b)) {
      return ffi::GetRef<Expr>(op);
    }

    // The two branches may have had different TIR variables inlined.
    // For example, one branch has a dynamic implementation and
    // produces `R.Tensor([M,N])`, while the other branch checks if
    // `N==16` and produces `R.Tensor([M,16])`.  After the branch, the
    // output is `R.Tensor([M,N])`.  However, the `GetStructLCA` would
    // correctly return `R.Tensor(ndim=2)`, removing all shape
    // information.
    //
    // Since we know the Type prior to replacing TIR variables,
    // this pass can provide a better Type than the generic
    // handling in ExprMutator, by restoring the symbolic variables
    // within each branch.
    auto new_ty = VisitExprDepTypeField(op->ty.as_or_throw<Type>());

    ffi::StructuralEqual struct_equal;
    if (!struct_equal(new_ty, GetType(true_b))) {
      auto output_var = Var("then_branch_with_dyn", new_ty);

      true_b = SeqExpr({BindingBlock({
                           MatchCast(output_var, true_b, new_ty),
                       })},
                       output_var);
    }

    if (!struct_equal(new_ty, GetType(false_b))) {
      auto output_var = Var("else_branch_with_dyn", new_ty);

      false_b = SeqExpr({BindingBlock({
                            MatchCast(output_var, false_b, new_ty),
                        })},
                        output_var);
    }

    return IfExpr(guard, true_b, false_b, op->loc);
  }

 private:
  class RuntimePrimVarCollector : public ExprVisitor {
   public:
    static std::unordered_set<tvm::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> Collect(
        const Expr& expr) {
      RuntimePrimVarCollector collector;
      collector.VisitExpr(expr);
      return collector.uses_;
    }

    void VisitExprDepTypeField(const Type&) final {}

    void VisitExpr_(const ShapeExprObj*) final {}

    void VisitExpr_(const VarObj* op) final {
      Var var = ffi::GetRef<Var>(op);
      if (auto prim_var = var.as<PrimVar>()) {
        uses_.insert(*prim_var);
      }
    }

   private:
    std::unordered_set<tvm::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> uses_;
  };

  PrimExpr CanonicalizeShapeValue(const PrimExpr& expr) {
    auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      auto prim_var = var.as<PrimVar>();
      if (!prim_var) return ffi::Unchanged();
      auto it = known_values_.find(*prim_var);
      if (it == known_values_.end()) return ffi::Unchanged();
      return ffi::Any(it->second.expr);
    };
    PrimExpr output =
        ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(expr, f_substitute).as_or_throw<PrimExpr>();
    return output.same_as(expr) ? expr : builder_->GetAnalyzer()->Simplify(output);
  }

  struct KnownValue {
    PrimExpr expr;
    MatchCast source;
  };

  std::unordered_map<tvm::Var, KnownValue, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> known_values_;
  std::unordered_set<tvm::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> runtime_prim_var_uses_;
  bool canonicalize_shape_values_ = true;
};

Expr SymbolicVarCanonicalizer::Apply(Expr expr) {
  SymbolicVarCanonicalizer mutator;
  if (!expr->IsInstance<FunctionObj>()) {
    // Dataflow rewriting canonicalizes a SeqExpr directly, without entering
    // VisitExpr_(FunctionObj), so initialize the same runtime-use context.
    mutator.runtime_prim_var_uses_ = RuntimePrimVarCollector::Collect(expr);
  }
  return mutator(std::move(expr));
}

struct CanonicalizationPlan {
  ffi::Map<Var, Var> replace_usage;
  ffi::Map<Var, Var> replace_binding;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> bindings_to_remove;
  ffi::Map<Var, GenericConst> inline_constant;
};

/*! \brief Utility class to identify usage location
 *
 * Canonicalization of a variable binding may require information from
 * later in the function.  For example, replacing `dataflow_x = expr`
 * with `var_x = expr` to avoid a trivial binding of `var_x =
 * dataflow_x` later in the function.  This utility examines a relax
 * expression, and plans the changes to be made in a mutation pass.
 */
class CanonicalizePlanner : public ExprVisitor {
 public:
  static CanonicalizationPlan Collect(const Expr& expr) {
    CanonicalizePlanner visitor;
    visitor.VisitExpr(expr);

    CanonicalizationPlan plan;

    // If a Var has been defined inside a DataflowBlock, is only used
    // within a DataflowBlock, and is not already handled by removal
    // of trivial bindings, then we can replace it with a DataflowVar.
    for (auto var : visitor.defined_inside_dataflow_) {
      if (!var.as<DataflowVarObj>() && !visitor.used_outside_home_dataflow_.count(var)) {
        DataflowVar new_var(var->name, GetType(var));

        plan.replace_binding.Set(var, new_var);
        plan.replace_usage.Set(var, new_var);
      }
    }

    for (const auto& [var, constant] : visitor.known_bound_to_constant_) {
      plan.inline_constant.Set(var, constant);
    }

    for (const auto& binding_iter : visitor.trivial_bindings_) {
      Var bound_var = binding_iter.first;
      Var bound_to = binding_iter.second;

      while (auto opt = visitor.trivial_bindings_.Get(bound_to)) {
        // This may be a trivial binding into a trivial binding.  In
        // that case, unwrap the bindings until we find the earliest
        // non-trivial binding.
        bound_to = opt.value();
      }
      while (auto opt = plan.replace_binding.Get(bound_to)) {
        // The variable we are binding to may have already been
        // replaced, if it fell into Case 4 (Var = DataflowVar).  In
        // that case, we check against its replacement instead.
        bound_to = opt.value();
      }

      if (bound_var.as<DataflowVarObj>() || !bound_to.as<DataflowVarObj>() ||
          !visitor.used_outside_home_dataflow_.count(bound_var)) {
        // Case 1: Var = Var
        // Case 2: DataflowVar = Var
        // Case 3: DataflowVar = DataflowVar
        // Case 4a: Var = DataflowVar, where the Var is not used
        //          outside the DataflowBlock containing the binding
        //
        // For these four cases, the trivial binding can be unwrapped,
        // using the bound variable directly at the point of use.
        plan.replace_usage.Set(bound_var, bound_to);
        plan.bindings_to_remove.insert(bound_var);
      } else {
        // Case 4b: Var = DataflowVar, where the Var is used somewhere
        //          outside the DataflowBlock containing the binding
        //
        // Replacing a Var with a DataflowVar could result in illegal
        // use of a DataflowVar outside of a DataflowBlock.  Instead,
        // we replace in the opposite direction, replacing the binding
        // of the DataflowVar with a binding of the Var.
        plan.replace_binding.Set(bound_to, bound_var);
        plan.replace_usage.Set(bound_to, bound_var);
        plan.bindings_to_remove.insert(bound_var);
      }
    }

    return plan;
  }

 private:
  void VisitExpr_(const FunctionObj* func) override {
    // for functions, treat any free vars as used outside their home DF block
    auto cache = current_block_;
    current_block_ = ffi::Optional<BindingBlock>();
    auto free_vars = FreeVars(ffi::GetRef<Function>(func));
    for (auto var : free_vars) {
      used_outside_home_dataflow_.insert(var);
    }
    ExprVisitor::VisitExpr_(func);
    current_block_ = cache;
  }

  void VisitExpr_(const SeqExprObj* seq) override {
    // need to reset current_block_ for nested seq exprs (such as in If nodes)
    auto cache = current_block_;
    current_block_ = ffi::Optional<BindingBlock>();
    ExprVisitor::VisitExpr_(seq);
    current_block_ = cache;
  }

  void VisitBindingBlock_(const BindingBlockObj* block) override {
    TVM_FFI_ICHECK(!current_block_.has_value()) << "Forgetting to unset current block";
    current_block_ = ffi::GetRef<BindingBlock>(block);
    ExprVisitor::VisitBindingBlock_(block);
    current_block_ = ffi::Optional<BindingBlock>();
  }

  void VisitBindingBlock_(const DataflowBlockObj* block) override {
    TVM_FFI_ICHECK(!current_block_.has_value()) << "Forgetting to unset current block";
    current_block_ = ffi::GetRef<DataflowBlock>(block);
    ExprVisitor::VisitBindingBlock_(block);
    current_block_ = ffi::Optional<BindingBlock>();
  }

  ffi::Optional<Expr> UnwrapKnownValue(Expr expr) {
    // If the expression is a variable, then it can be unwrapped into
    // its known value.
    auto unwrap_var = [this](Expr expr) -> Expr {
      if (auto var = expr.as<Var>()) {
        if (auto opt = known_bindings_.Get(var.value())) {
          return opt.value();
        }
      }
      return expr;
    };

    auto recursively_unwrap_var = [&unwrap_var](Expr expr) -> Expr {
      while (true) {
        auto new_expr = unwrap_var(expr);
        if (new_expr.same_as(expr)) {
          return expr;
        } else {
          expr = new_expr;
        }
      }
    };

    // If the expression is a TupleGetItem, which accesses a field of
    // a known tuple, then it can be unwrapped into a direct access of
    // that field.
    if (auto tuple_get_item = expr.as<TupleGetItemObj>()) {
      Expr tuple = recursively_unwrap_var(tuple_get_item->tuple);
      if (auto ptr = tuple.as<TupleObj>()) {
        return ptr->fields[tuple_get_item->index];
      }
    }

    // If the expression is a Tuple, and each element is
    // `TupleGetItem(earlier_tuple, i)`, then this is just a copy of
    // `earlier_tuple`.
    auto earlier_tuple = [&]() -> ffi::Optional<Expr> {
      auto expr_tuple = expr.as<TupleObj>();
      if (!expr_tuple) {
        return std::nullopt;
      }

      if (expr_tuple->fields.empty()) {
        return std::nullopt;
      }

      auto first_element = recursively_unwrap_var(expr_tuple->fields[0]).as<TupleGetItemObj>();
      if (!first_element) {
        return std::nullopt;
      }

      auto earlier_tuple_size =
          GetType(first_element->tuple).as_or_throw<TupleType>()->fields.size();
      if (earlier_tuple_size != expr_tuple->fields.size()) {
        return std::nullopt;
      }

      Expr earlier_tuple = recursively_unwrap_var(first_element->tuple);

      for (size_t i = 0; i < expr_tuple->fields.size(); i++) {
        auto element = recursively_unwrap_var(expr_tuple->fields[i]).as<TupleGetItemObj>();
        if (!element) {
          return std::nullopt;
        }
        if (static_cast<size_t>(element->index) != i) {
          return std::nullopt;
        }

        auto source_of_element = recursively_unwrap_var(element->tuple);

        if (!earlier_tuple.same_as(source_of_element)) {
          return std::nullopt;
        }
      }

      return earlier_tuple;
    }();
    if (earlier_tuple) {
      return earlier_tuple.value();
    }

    return std::nullopt;
  }

  void VisitBinding(const Binding& binding) override {
    bool has_same_ty = [&]() {
      if (binding.as<VarBindingObj>()) {
        return true;
      } else if (auto match_cast = binding.as<MatchCastObj>()) {
        return ffi::StructuralEqual()(GetType(binding->var), GetType(match_cast->value));
      } else {
        TVM_FFI_THROW(InternalError) << "Invalid binding type: " << binding->GetTypeKey();
      }
    }();

    Expr value = GetBoundValue(binding);

    if (auto unwrapped = UnwrapKnownValue(value)) {
      value = unwrapped.value();
    }

    if (auto parent = value.as<Var>(); parent && has_same_ty) {
      trivial_bindings_.Set(binding->var, parent.value());
    }

    if (auto constant = value.as<GenericConst>()) {
      known_bound_to_constant_.Set(binding->var, constant.value());
    }

    known_bindings_.Set(binding->var, value);
    def_blocks_.Set(binding->var, current_block_.value());

    ExprVisitor::VisitBinding(binding);
  }

  void VisitVarDef(const Var& var) override {
    if (inside_dataflow()) {
      defined_inside_dataflow_.insert(var);
    }
  }

  void VisitExpr_(const VarObj* var) override {
    auto var_ref = ffi::GetRef<Var>(var);
    // if a var is used in a dataflow block but *not* the one
    // where it was defined, it also needs to be exposed, so also we treat that as
    // used outside of a dataflow block
    if (!inside_dataflow() || (def_blocks_.count(var_ref) &&
                               (current_block_.has_value() &&
                                !current_block_.value().same_as(def_blocks_.at(var_ref))))) {
      used_outside_home_dataflow_.insert(ffi::GetRef<Var>(var));
    }
  }

  inline bool inside_dataflow() {
    return current_block_.has_value() && current_block_.value().as<DataflowBlockObj>();
  }

  ffi::Optional<BindingBlock> current_block_;
  ffi::Map<Var, BindingBlock> def_blocks_;

  ffi::Map<Var, Var> trivial_bindings_;
  ffi::Map<Var, Expr> known_bindings_;
  ffi::Map<Var, GenericConst> known_bound_to_constant_;
  std::unordered_set<Var> defined_inside_dataflow_;
  // Set of vars either used outside a dataflow block altogether or outside their
  // home dataflow block (the one where they were defined)
  std::unordered_set<Var> used_outside_home_dataflow_;
};

/*! \brief The mutator class to apply a CanonicalizationPlan */
class BindingCanonicalizer : public ExprMutator {
 public:
  static Expr Apply(Expr expr) {
    auto used_outside_home_dataflow = CanonicalizePlanner::Collect(expr);
    BindingCanonicalizer mutator(std::move(used_outside_home_dataflow));
    return mutator.VisitExpr(expr);
  }

 private:
  explicit BindingCanonicalizer(CanonicalizationPlan plan) : plan_(plan) {}

  void VisitBinding(const Binding& binding) override {
    if (!plan_.bindings_to_remove.count(binding->var)) {
      ExprMutator::VisitBinding(binding);
    }
  }

  Var VisitVarDef(const Var& var) override {
    Var new_var = var;
    while (auto opt = plan_.replace_binding.Get(new_var)) {
      new_var = opt.value();
    }

    return ExprMutator::VisitVarDef(new_var);
  }

  Expr VisitExpr_(const VarObj* var) override {
    Var new_var = ffi::GetRef<Var>(var);
    while (auto opt = plan_.replace_usage.Get(new_var)) {
      new_var = opt.value();
    }
    if (auto opt = plan_.inline_constant.Get(new_var)) {
      return VisitExpr(opt.value());
    }

    return ExprMutator::VisitExpr_(new_var.get());
  }

  // Special case: for dataflow blocks, we will check for dataflow vars that solely exist
  // to be bound to the output. In this case, we will get rid of those bindings and
  // use the dataflow var's definition directly
  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block) override {
    auto new_block = ExprMutator::VisitBindingBlock_(block).as_or_throw<DataflowBlock>();
    std::unordered_set<DataflowVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> disqualified_set;
    std::unordered_set<DataflowVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> output_vars;

    std::unordered_map<DataflowVar, Expr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> candidates;
    for (int i = new_block->bindings.size() - 1; i >= 0; i--) {
      auto binding = new_block->bindings[i];
      auto var = binding->var;
      auto value = GetBoundValue(binding);

      if (var->IsInstance<DataflowVarObj>()) {
        auto df_var = var.as_or_throw<DataflowVar>();

        // disqualify any vars that appear in the RHS
        // (for a function literal, consider only free vars)
        ffi::Array<Var> rhs_vars;
        if (!value->IsInstance<FunctionObj>()) {
          rhs_vars = FreeVars(value);
        } else {
          rhs_vars = AllVars(value);
        }

        for (auto rhs_var : rhs_vars) {
          if (rhs_var->IsInstance<DataflowVarObj>()) {
            disqualified_set.insert(rhs_var.as_or_throw<DataflowVar>());
          }
        }

        // if the current var is an output and has not been disqualified,
        // then include it in the candidate map
        if (!disqualified_set.count(df_var) && output_vars.count(df_var)) {
          candidates.insert_or_assign(df_var, value);
        }
      } else {
        // The LHS is an output binding.
        // We are looking for cases where the RHS is a single dataflow var;
        // disqualify if the RHS is not a single dataflow var
        // or if the var has been output before
        if (const auto* rhs_var = value.as<DataflowVarObj>()) {
          if (output_vars.count(ffi::GetRef<DataflowVar>(rhs_var))) {
            disqualified_set.insert(ffi::GetRef<DataflowVar>(rhs_var));
          }
          output_vars.insert(ffi::GetRef<DataflowVar>(rhs_var));
        } else {
          ffi::Array<Var> disqualified;
          // for function literal, consider only free vars
          if (value->IsInstance<FunctionObj>()) {
            disqualified = FreeVars(value);
          } else {
            disqualified = AllVars(value);
          }

          for (auto rhs_var : disqualified) {
            if (rhs_var->IsInstance<DataflowVarObj>()) {
              disqualified_set.insert(rhs_var.as_or_throw<DataflowVar>());
            }
          }
        }
      }
    }

    // second pass: for each binding where the LHS is a candidate, remove the binding.
    // If the RHS is a candidate, replace it with the definition
    ffi::Array<Binding> new_bindings;
    bool changed = false;
    for (auto binding : new_block->bindings) {
      if (binding->var->IsInstance<DataflowVarObj>() &&
          candidates.count(binding->var.as_or_throw<DataflowVar>())) {
        changed = true;
        continue;
      } else if (!binding->var->IsInstance<DataflowVarObj>() &&
                 GetBoundValue(binding)->IsInstance<DataflowVarObj>() &&
                 candidates.count(GetBoundValue(binding).as_or_throw<DataflowVar>())) {
        changed = true;
        if (auto* match_binding = binding.as<MatchCastObj>()) {
          auto new_binding = MatchCast(
              binding->var, candidates.at(match_binding->value.as_or_throw<DataflowVar>()),
              match_binding->ty);
          new_bindings.push_back(new_binding);
        } else if (auto* var_binding = binding.as<VarBindingObj>()) {
          auto new_binding = VarBinding(
              binding->var, candidates.at(var_binding->value.as_or_throw<DataflowVar>()));
          new_bindings.push_back(new_binding);
        } else {
          TVM_FFI_ICHECK(false) << "Invalid binding";  // never happens
        }
      } else {
        new_bindings.push_back(binding);
      }
    }

    if (!changed) {
      return new_block;
    }
    return DataflowBlock(new_bindings);
  }

 private:
  CanonicalizationPlan plan_;
};
}  // namespace

Expr CanonicalizeTIRVariables(Expr expr) {
  return SymbolicVarCanonicalizer::Apply(std::move(expr));
}

Expr CanonicalizeRelaxBindings(Expr expr) { return BindingCanonicalizer::Apply(std::move(expr)); }

Expr CanonicalizeBindings(Expr expr) {
  expr = CanonicalizeTIRVariables(std::move(expr));
  expr = CanonicalizeRelaxBindings(std::move(expr));
  return expr;
}

namespace transform {

Pass CanonicalizeTIRVariables() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return CanonicalizeTIRVariables(f).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, 1, "CanonicalizeTIRVariables");
}

Pass CanonicalizeRelaxBindings() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return CanonicalizeBindings(f).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, 1, "CanonicalizeRelaxBindings");
}

Pass CanonicalizeBindings() {
  return tvm::transform::Sequential(
      {
          CanonicalizeTIRVariables(),
          CanonicalizeRelaxBindings(),
      },
      "CanonicalizeBindings");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.CanonicalizeBindings", CanonicalizeBindings);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
