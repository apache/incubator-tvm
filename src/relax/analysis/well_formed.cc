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
 * \file relax/analysis/well_formed.cc
 * \brief Check if the IRModule is well-formed.
 *
 * This pass is supposed to be applied to normalized Relax AST.
 * If it's malformed, an ffi::Error is thrown on the first violation, seeded
 * with the offending node so the caller can resolve a precise access path.
 * This pass will check:
 *    1. Each Expr should have `ty` field already populated, when
 *      `check_ty` is true.
 *    2. GlobalVars are defined before use. And all GlobalVars have different names.
 *    3. When a Function has a corresponding GlobalVar and a `global_symbol`
 *       attribute, the name of the GlobalVar must equal the value of the
 *       `global_symbol` attribute value.
 *    4. Any variable cannot used as different function parameters in the same IRModule.
 *    5. Vars, including those introduced in definitional type positions, are defined before use.
 *    6. Vars are defined exactly once.
 *    7. DataflowVars cannot be defined inside BindingBlock.
 *    8. Vars defined in IfExprObj, except the return Var, are invisible
 *       out of the If body.(May change for new AST designs)
 *    9. SeqExpr only serves as function body, or in the true and
 *       false branches in IfExprObj.
 *    10. The IR is in ANF:
 *       (a) Expressions cannot contain nested complex expressions.
 *           Here are the expressions that may be nested inside other expressions:
 *           Var, DataflowVar, GlobalVar, GenericConst, ShapeExpr,
 *           Op, Tuple (we call these "leaf" expressions).
 *       (b) The right-hand side of a binding may contain a non-leaf expression
 *           (where all expressions nested in it are leaf expressions),
 *           other than SeqExprs (see rule 9)
 *       (c) Exceptions: The body of a Function node and the true branch
 *           and false branch of If nodes *must* be SeqExprs.
 *       (d) Places where non-leaf expressions cannot appear:
 *           * The tuple_value field of TupleGetItem nodes
 *           * The cond field of If nodes
 *           * The op or args fields of Call nodes
 *           * Inside the fields of Tuple nodes
 *    11. Expr always has ty (with the exception of Op).
 *    12. DataflowBlocks may not contain If nodes.
 *    13. DataflowBlocks may not contain calls to impure functions or operators
 *        (only checked if check_ty is true).
 *    14. If a function has is_pure set to true and the kForcePure attribute is not set,
 *        the body may not contain any impure call (only checked if check_ty is true).
 *    15. If the kForcePure attribute is set for a function,
 *        that function's is_pure field must be true.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/type_functor.h>
#include <tvm/relax/utils.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace relax {
using namespace tvm::prim;

// TODO(relax-team): Consider further refactor using
// Scope Frame to store manage the var context.
//
/*! \brief Helper to implement well formed check.*/
class WellFormedChecker : public relax::ExprVisitor, public relax::TypeVisitor {
 public:
  // Throws ffi::Error on the first well-formedness violation, seeded with the
  // offending node so the caller can resolve an access path. Returns normally
  // when the object is well-formed.
  static void Check(ffi::Variant<IRModule, Function> obj, bool check_ty) {
    WellFormedChecker well_formed_checker = WellFormedChecker(obj.as<IRModule>(), check_ty);

    if (const auto* mod = obj.as<IRModuleObj>()) {
      for (const auto& it : mod->functions) {
        // visit relax.Function
        if (auto* n = it.second.as<FunctionObj>()) {
          Function func = ffi::GetRef<Function>(n);
          well_formed_checker.func_name_map_[n] = it.first->name_hint;
          well_formed_checker.CheckGlobalVarAndGsymbolConsistency(it.first, func);
          well_formed_checker.VisitExpr(func);
        }
      }
    } else if (const auto* func = obj.as<FunctionObj>()) {
      well_formed_checker.VisitExpr(ffi::GetRef<Expr>(func));
    } else {
      TVM_FFI_THROW(InternalError) << "Unreachable, "
                                   << "variant did not contain any of the allowed types";
    }
  }

 private:
  WellFormedChecker(ffi::Optional<IRModule> mod, bool check_ty)
      : mod_(std::move(mod)), check_ty(check_ty), cur_visited_func_(nullptr) {}

  using relax::ExprVisitor::VisitExpr_;

  // Possible mode of visitor
  enum class VisitMode {
    /*!
     * \brief Check all vars are well-defined
     */
    kDefault,
    /*!
     * \brief Match define the vars on first occurance.
     * Do not check the well-defined property of composite expr.
     */
    kMatchVarDef
  };

  class PrimitiveExprChecker : public tirx::StmtExprVisitor {
   public:
    explicit PrimitiveExprChecker(WellFormedChecker* parent)
        : tirx::StmtExprVisitor([] {
            static const VTable table = [] {
              VTable table;
              PrimitiveExprChecker::InitVTable(&table);
              table.Finalize();
              return table;
            }();
            return &table;
          }()),
          parent_(parent) {}

   private:
    ffi::Optional<VisitInterrupt> Visit_(const tvm::VarObj* op) final {
      parent_->VisitExpr(ffi::GetRef<Expr>(op));
      return std::nullopt;
    }

    static void InitVTable(VTable* table) {
      tirx::StmtExprVisitor::InitVTable(table);
      SetDispatch<tirx::StmtExprVisitor, DataflowVarObj>(table);
    }

    WellFormedChecker* parent_{nullptr};
  };

  /*! \brief Get the name of a function for use in error messages. */
  std::string FuncName(const FunctionObj* func) const {
    auto it = func_name_map_.find(func);
    if (it != func_name_map_.end()) {
      return "\"" + it->second + "\"";
    }
    return "(anonymous function)";
  }

  void CheckGlobalVarAndGsymbolConsistency(GlobalVar var, Function func) {
    // the uniqueness of all global vars are ensured by IRModule->global_var_map_, so do not need
    // to check again

    // check name in global var and gsymbol
    ffi::Optional<ffi::String> gsymbol = func->GetAttr(tvm::attr::kGlobalSymbol);
    if (gsymbol.has_value() && gsymbol != var->name_hint) {
      TVM_FFI_VISIT_THROW(ValueError, func->loc)
          << "Name in GlobalVar is not equal to name in gsymbol: " << var
          << " != " << gsymbol.value();
    }
  }

  void VisitExpr(const Expr& expr) final {
    if (!expr.as<OpObj>() && expr->ty.as<MissingType>().has_value()) {
      TVM_FFI_VISIT_THROW(TypeError, expr) << "The ty of Expr " << expr << " is missing.";
    }
    relax::ExprVisitor::VisitExpr(expr);
  }

  void VisitExpr_(const GlobalVarObj* op) final {
    GlobalVar var = ffi::GetRef<GlobalVar>(op);
    if (mod_.has_value()) {
      if (!(mod_.value()->ContainGlobalVar(var->name_hint) &&
            mod_.value()->GetGlobalVar(var->name_hint).same_as(var))) {
        TVM_FFI_VISIT_THROW(ValueError, var)
            << "GlobalVar " << ffi::GetRef<Expr>(op) << " is not defined.";
      }
    }

    if (!op->ty.as<MissingType>().has_value()) {
      if (op->ty.as<tvm::FuncTypeObj>()) {
        // A native callee owns its parameter symbols and tensor contracts.
        // Explicit call bridges validate their Relax-facing use separately.
        return;
      }
      if (!op->ty->IsInstance<FuncTypeObj>()) {
        TVM_FFI_VISIT_THROW(TypeError, var)
            << "The ty of GlobalVar " << ffi::GetRef<Expr>(op) << " must be either FuncType.";
      }
    }

    CheckType(op);
  }

  void VisitExpr_(const TupleObj* op) final {
    TVM_FFI_VISIT_BEGIN();
    for (size_t i = 0; i < op->fields.size(); i++) {
      Expr expr = op->fields[i];
      if (IsLeafOrTuple(expr)) {
        this->VisitExpr(expr);
      } else {
        TVM_FFI_VISIT_THROW(ValueError, expr)
            << "Tuple is not in ANF form, field " << i << " gets " << expr->GetTypeKey();
      }
    }

    CheckType(op);
    TVM_FFI_VISIT_END(ffi::GetRef<Expr>(op));
  }

  void VisitExpr_(const TupleGetItemObj* op) final {
    if (IsLeafOrTuple(op->tuple)) {
      this->VisitExpr(op->tuple);
    } else {
      TVM_FFI_VISIT_THROW(TypeError, ffi::GetRef<Expr>(op))
          << "The tuple value in a TupleGetItem node must be a leaf expression.";
    }
    CheckType(op);
  }

  void VisitExpr_(const VarObj* op) final {
    Var var = ffi::GetRef<Var>(op);
    if (var_set_.count(var) == 0 && recur_vars_.count(var) == 0) {
      TVM_FFI_VISIT_THROW(ValueError, var) << "Var " << ffi::GetRef<Expr>(op) << " is not defined.";
    }
    CheckType(op);
  }

  void VisitExpr_(const DataflowVarObj* op) final {
    DataflowVar var = ffi::GetRef<DataflowVar>(op);
    if (!is_dataflow_) {
      TVM_FFI_VISIT_THROW(ValueError, var)
          << "DataflowVar " << ffi::GetRef<Expr>(op) << " is used outside DataflowBlock.";
    }
    if (dataflow_var_set_.count(var) == 0) {
      TVM_FFI_VISIT_THROW(ValueError, var)
          << "DataflowVar " << ffi::GetRef<Expr>(op) << " is not defined.";
    }
    CheckType(op);
  }

  void VisitExpr_(const FunctionObj* op) final {
    TVM_FFI_VISIT_BEGIN();
    // set current visited function.
    // for nested functions, we only set the outermost function.
    if (cur_visited_func_ == nullptr) {
      cur_visited_func_ = op;
    }

    // save the var_set_ for local function
    auto prev_var_set = var_set_;
    auto prev_dataflow_var_set = dataflow_var_set_;
    bool old_dataflow_state = is_dataflow_;
    is_dataflow_ = false;

    // ensure the purity attributes are valid
    if (op->GetAttr(tvm::relax::attr::kForcePure).value_or(false) && !op->is_pure) {
      TVM_FFI_VISIT_THROW(ValueError, op->loc)
          << "Function " << ffi::GetRef<Expr>(op) << " has true for "
          << tvm::relax::attr::kForcePure.name << " but false for is_pure; "
          << tvm::relax::attr::kForcePure.name << " should be true only if is_pure is also true.";
    }

    // Bare symbolic occurrences in parameter types are signature definitions,
    // including when a later parameter type defines a symbol used by an
    // earlier compound expression.
    WithMode(VisitMode::kMatchVarDef, [&]() {
      for (Var param : op->params) {
        relax::TypeVisitor::VisitType(GetType(param));
      }
    });
    for (Var param : op->params) {
      if (!prev_var_set.count(param)) {
        var_set_.erase(param);
      }
    }

    // Primitive parameters may supply dimensions anywhere in the signature.
    // Register each once so duplicate parameters remain an error.
    for (Var param : op->params) {
      if (GetType(param).as<PrimTypeObj>()) {
        RegisterVarDefinition(param);
      }
    }

    // Other runtime parameters retain source-order definition scope.
    for (Var param : op->params) {
      if (auto* dataflow_var = param.as<DataflowVarObj>()) {
        TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<DataflowVar>(dataflow_var))
            << "DataflowVar " << param << " is defined outside DataflowBlock.";
      }
      if (!GetType(param).as<PrimTypeObj>()) {
        RegisterVarDefinition(param);
      }

      auto it = param_var_func_map_.find(param);
      if (it != param_var_func_map_.end() && it->second != cur_visited_func_) {
        TVM_FFI_VISIT_THROW(ValueError, param->loc)
            << "Relax variable " << param << " is used as a parameter in both function "
            << FuncName(it->second) << " and function " << FuncName(cur_visited_func_) << ".";
      }
      param_var_func_map_.insert({param, cur_visited_func_});
      CheckType(param.get());
    }
    // check function ret_ty
    if (!op->ret_ty.as<MissingType>().has_value()) {
      this->VisitType(op->ret_ty);
    } else {
      TVM_FFI_VISIT_THROW(TypeError, ffi::GetRef<Expr>(op)) << "Function must have defined ret_ty";
    }

    // if we are not forcing purity and the function is annotated as pure, it must not contain an
    // impure call
    if (check_ty && !op->GetAttr(tvm::relax::attr::kForcePure).value_or(false) && op->is_pure) {
      if (auto impure = FindImpureCall(op->body)) {
        TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<Expr>(op))
            << "Function " << op << " is annotated as pure but contains an impure call: " << impure
            << ".  Please set " << tvm::relax::attr::kForcePure.name << " to true "
            << "or use a pure operator variant (e.g., call_pure_packed) "
            << "if it is necessary to override this judgment.";
      }
    }

    this->VisitSeqExpr(op->body.get());

    is_dataflow_ = old_dataflow_state;
    dataflow_var_set_ = prev_dataflow_var_set;
    var_set_ = prev_var_set;

    if (cur_visited_func_ == op) {
      cur_visited_func_ = nullptr;
    }
    TVM_FFI_VISIT_END(ffi::GetRef<Expr>(op));
  }

  void VisitExpr_(const CallObj* call) final {
    TVM_FFI_VISIT_BEGIN();
    TVM_FFI_CHECK(call->op.as<OpObj>() || !call->op->ty.as<tvm::FuncTypeObj>(), TypeError)
        << "Ordinary Relax calls cannot invoke a native TIRx function; "
        << "use R.call_tir or R.call_tir_packed";
    if (IsLeafOrTuple(call->op)) {
      const FunctionObj* prev_visited_func = cur_visited_func_;
      cur_visited_func_ = nullptr;  // do not attribute the callee to its caller
      this->VisitExpr(call->op);
      cur_visited_func_ = prev_visited_func;
    } else {
      TVM_FFI_VISIT_THROW(TypeError, ffi::GetRef<Call>(call))
          << "The called expression must be a leaf expression";
    }
    for (size_t i = 0; i < call->args.size(); i++) {
      Expr arg = call->args[i];
      if (IsLeafOrTuple(arg)) {
        this->VisitExpr(arg);
      } else {
        TVM_FFI_VISIT_THROW(ValueError, arg->loc)
            << "Call is not in ANF form, arg " << i << " gets " << arg->GetTypeKey();
      }
    }

    for (const Type& ty_arg : call->ty_args) {
      this->VisitType(ty_arg);
    }

    CheckType(call);
    if (is_dataflow_ && check_ty) {
      if (auto impure = FindImpureCall(ffi::GetRef<Call>(call))) {
        TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<Call>(call))
            << "Impure function call " << impure << " occurs within a dataflow block.";
      }
    }

    // If the operation has defined a custom normalization function
    // using the FNormalize attribute, the call node must be normalized in order to be well-formed.
    // If we apply the FNormalize and it produces any change, modified the expression, re-visit in
    // case it produced a nested expression.

    if (auto func_normalize = op_map_normalize_.get(call->op, nullptr); func_normalize != nullptr) {
      auto dummy_builder = tvm::relax::BlockBuilder::Create(mod_);
      Call before_normalize = ffi::GetRef<Call>(call);
      ffi::Optional<Expr> after_normalize = std::nullopt;
      try {
        after_normalize = func_normalize(dummy_builder, before_normalize);
      } catch (std::exception& err) {
        TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<Call>(call))
            << "If an operator defines an operator-specific normalization function (FNormalize), "
            << "calls to that operator must be normalized with it.  "
            << "However, normalization of " << before_normalize << " resulted in the error: \n"
            << err.what();
      }
      if (after_normalize && !after_normalize.same_as(before_normalize)) {
        TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<Call>(call))
            << "If an operator defines an operator-specific normalization function (FNormalize), "
            << "calls to that operator must be normalized with it.  "
            << "However, normalization of " << before_normalize << " resulted in "
            << after_normalize;
      }
    }

    if (auto op = call->op.as<Op>()) {
      try {
        op.value().Validate(call);
      } catch (std::exception& err) {
        TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<Call>(call))
            << "Operator-specific validation for " << call->op << " identified error: \n"
            << err.what();
      }
    }

    bool has_infer_type = true;
    if (auto op = call->op.as<Op>()) {
      has_infer_type =
          op_map_infer_type_.count(op.value()) || op_map_infer_type_with_builder_.count(op.value());
    }
    if (check_ty && !call->ty.as<MissingType>().has_value() &&
        (!call->ty.as<PrimTypeObj>() || has_infer_type)) {
      // The `InferType` method isn't currently exposed by the
      // Normalizer, and can only be called indirectly by normalizing
      // an expression that does not yet have `Type`.
      auto dummy_builder = tvm::relax::BlockBuilder::Create(mod_);
      Call copied(Type::Missing(), call->op, call->args, call->attrs, call->ty_args);
      ffi::Optional<Expr> normalized = std::nullopt;
      try {
        normalized = dummy_builder->Normalize(copied);
      } catch (std::exception& err) {
        TVM_FFI_VISIT_THROW(TypeError, ffi::GetRef<Call>(call))
            << "Each Relax expression must be able to have its Type inferred.  "
            << "However, inferring the type of expression " << ffi::GetRef<Call>(call)
            << " resulted in the error: \n"
            << err.what();
      }
      if (normalized.has_value()) {
        auto inferred_ty = GetType(normalized.value());
        auto current_ty = call->ty.as_or_throw<Type>();

        // An error should be raised if the annotated Type is
        // provably incorrect.  This check is done using
        // `TypeBaseCheck(...) < kFailL1`, because `kFailL1`
        // represents cases that are neither provably correct nor
        // provably incorrect.  If this check were replaced with
        // `!IsBaseOf(...)`, cases that are correct but not provably
        // so would raise an exception.
        //
        // For example, if a dynamic size in the inferred Type
        // is equivalent to the expression used in the annotated
        // Type, but the TIR simplifications are not sufficient
        // to prove that the two expressions are equivalent, we should
        // not raise an error.
        if (TypeBaseCheck(current_ty, inferred_ty) < BaseCheckResult::kFailL1) {
          TVM_FFI_VISIT_THROW(TypeError, ffi::GetRef<Expr>(call))
              << "All information in Type annotations must be correct.  "
              << "However, while the expression " << ffi::GetRef<Call>(call) << " is annotated as "
              << current_ty << ", the expression outputs " << inferred_ty;
        }
      }
    }
    TVM_FFI_VISIT_END(ffi::GetRef<Call>(call));
  }

  void VisitExpr_(const IfExprObj* op) final {
    TVM_FFI_VISIT_BEGIN();
    if (is_dataflow_) {
      TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<Expr>(op))
          << "IfExpr nodes are not allowed to appear in dataflow blocks.";
    }
    if (IsLeafOrTuple(op->cond)) {
      this->VisitExpr(op->cond);
    } else {
      TVM_FFI_VISIT_THROW(TypeError, ffi::GetRef<Expr>(op))
          << "The condition for an if node must be a leaf expression.";
    }

    std::unordered_set<Var> previous_var_set = var_set_;
    this->VisitSeqExpr(op->true_branch.get());
    var_set_ = previous_var_set;
    this->VisitSeqExpr(op->false_branch.get());
    var_set_ = previous_var_set;

    CheckType(op);
    TVM_FFI_VISIT_END(ffi::GetRef<Expr>(op));
  }

  void VisitExpr_(const ShapeExprObj* op) final {
    for (PrimExpr expr : op->values) {
      // check if the symbolic vars in the expr are defined, e.g, 2 * m
      VisitPrimitiveExpr(expr);
      if (expr.ty().code() != DLDataTypeCode::kDLInt) {
        TVM_FFI_VISIT_THROW(TypeError, expr)
            << "Shape expressions must be of integer type, but got " << expr.ty()->dtype;
      }
    }
    CheckType(op);
  }

  void VisitExpr_(const SeqExprObj* op) final {
    TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<Expr>(op))
        << "SeqExpr only serves as the function body in FunctionObj, "
           "or the true/false branch body in IfExprObj.";
  }

  void VisitSeqExpr(const SeqExprObj* op) {
    TVM_FFI_VISIT_BEGIN();
    // a special call only if SeqExpr is the function body
    // in FunctionObj or the true/false branch body in IfExprObj
    for (BindingBlock block : op->blocks) {
      this->VisitBindingBlock(block);
    }
    if (!IsLeafOrTuple(op->body)) {
      TVM_FFI_VISIT_THROW(TypeError, ffi::GetRef<Expr>(op))
          << "SeqExpr bodies must be leaf expressions.";
    }
    this->VisitExpr(op->body);
    CheckType(op);
    TVM_FFI_VISIT_END(ffi::GetRef<Expr>(op));
  }

  void VisitBinding_(const VarBindingObj* binding) final {
    bool is_lambda = false;
    if (binding->value->IsInstance<FunctionObj>()) {
      is_lambda = true;
      recur_vars_.insert(binding->var);
    }
    if (binding->value->IsInstance<tirx::FunctionObj>()) {
      TVM_FFI_VISIT_THROW(ValueError, binding->value)
          << "Inline tirx::Function is disallowed in Relax IR.";
    } else {
      this->VisitExpr(binding->value);
    }

    this->VisitVarDef(binding->var);

    if (check_ty && !binding->var->ty.as<MissingType>().has_value() &&
        !binding->value->ty.as<MissingType>().has_value()) {
      auto expr_ty = GetType(binding->value);
      auto var_ty = GetType(binding->var);
      if (!IsBaseOf(var_ty, expr_ty)) {
        TVM_FFI_VISIT_THROW(TypeError, binding->var)
            << "Expression of type " << expr_ty << " cannot be assigned to a variable of type "
            << var_ty;
      }
    }

    if (is_lambda) {
      recur_vars_.erase(binding->var);
    }
  }

  void VisitBinding_(const MatchCastObj* binding) final {
    this->VisitExpr(binding->value);
    // define the vars
    WithMode(VisitMode::kMatchVarDef, [&]() { this->VisitType(binding->ty); });

    this->VisitType(binding->ty);
    this->VisitVarDef(binding->var);
  }

  void VisitBindingBlock_(const DataflowBlockObj* block) final {
    bool old_is_dataflow_ = is_dataflow_;
    is_dataflow_ = true;
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    is_dataflow_ = old_is_dataflow_;
    dataflow_var_set_.clear();
  }

  void VisitVarDef_(const DataflowVarObj* var) final {
    if (!is_dataflow_) {
      TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<DataflowVar>(var))
          << "DataflowVar " << var << " is defined outside DataflowBlock.";
    }
    DataflowVar lv = ffi::GetRef<DataflowVar>(var);
    if (dataflow_var_set_.count(lv) == 1) {
      TVM_FFI_VISIT_THROW(ValueError, lv) << "DataflowVar " << lv << " is defined more than once.";
    }
    // register DataflowVar
    dataflow_var_set_.insert(lv);
    CheckType(var);
  }

  void VisitVarDef_(const VarObj* var) final {
    Var gv = ffi::GetRef<Var>(var);
    RegisterVarDefinition(gv);
    CheckType(var);
  }

  void RegisterVarDefinition(const Var& var) {
    if (var_set_.count(var) == 1) {
      TVM_FFI_VISIT_THROW(ValueError, var) << "Var " << var << " is defined more than once.";
    }
    var_set_.insert(var);
  }

  void VisitType_(const FuncTypeObj* op) final {
    auto prev_var_set = var_set_;
    const FunctionObj* prev_visited_func = cur_visited_func_;
    cur_visited_func_ = nullptr;
    if (op->params.has_value()) {
      WithMode(VisitMode::kMatchVarDef, [&]() {
        TVM_FFI_ICHECK(mode_ == VisitMode::kMatchVarDef);
        for (Type param : op->params.value()) {
          this->VisitType(param);
        }
      });
    }
    this->VisitType(op->ret);
    cur_visited_func_ = prev_visited_func;
    var_set_ = prev_var_set;
  }

  void VisitTypeExprField(const Expr& expr) final {
    if (auto* op = expr.as<DataflowVarObj>()) {
      TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<DataflowVar>(op))
          << "DataflowVar cannot be used as a dependent type expression.";
    }
    if (mode_ == VisitMode::kMatchVarDef) {
      // Populate a canonical Var definition on first occurrence.
      if (auto* op = expr.as<VarObj>()) {
        MarkTypeVarDefinition(ffi::GetRef<Var>(op));
      }
      if (auto* shape = expr.as<relax::ShapeExprObj>()) {
        for (auto val : shape->values) {
          this->VisitTypeExprField(val);
        }
      }
    } else {
      relax::ExprVisitor::VisitExpr(expr);
    }
  }

  void VisitTypeExprField(const PrimExpr& expr) final {
    if (auto* op = expr.as<DataflowVarObj>()) {
      TVM_FFI_VISIT_THROW(ValueError, ffi::GetRef<DataflowVar>(op))
          << "DataflowVar cannot be used as a dependent type expression.";
    }
    if (mode_ == VisitMode::kMatchVarDef) {
      // A primitive Var is the same canonical Var as a Relax binding.  A
      // first occurrence in a definitional type position enters the one
      // lexical definition set used by all Var occurrences.
      if (auto* op = expr.as<VarObj>()) {
        MarkTypeVarDefinition(ffi::GetRef<Var>(op));
      }
    } else {
      VisitPrimitiveExpr(expr);
    }
  }

  void VisitPrimitiveExpr(const PrimExpr& expr) {
    ffi::make_object<PrimitiveExprChecker>(this)->Visit(expr);
  }

  void MarkTypeVarDefinition(const Var& var) {
    var_set_.insert(var);
    if (cur_visited_func_ == nullptr) {
      return;
    }
    auto it = param_var_func_map_.find(var);
    if (it != param_var_func_map_.end() && it->second != cur_visited_func_) {
      TVM_FFI_VISIT_THROW(ValueError, var->loc)
          << "Var " << var << " is defined in both function " << FuncName(it->second)
          << " and function " << FuncName(cur_visited_func_) << ".";
    }
    param_var_func_map_.insert({var, cur_visited_func_});
  }

  void CheckType(const ExprObj* op) {
    if (!check_ty) {
      return;
    }

    if (auto* ty = op->ty.as<TypeObj>()) {
      this->VisitType(ffi::GetRef<Type>(ty));
    } else {
      TVM_FFI_VISIT_THROW(TypeError, ffi::GetRef<Expr>(op))
          << "Expr must have ty populated. "
          << " Expr.type_key=" << op->GetTypeKey();
    }
  }

  // Run callback with mode.
  template <typename FType>
  void WithMode(VisitMode mode, FType callback) {
    std::swap(mode_, mode);
    callback();
    std::swap(mode_, mode);
  }

  ffi::Optional<IRModule> mod_;
  const bool check_ty;
  bool is_dataflow_;
  // Current visited function.
  const FunctionObj* cur_visited_func_;
  // Map from function pointer to its global name (for error messages).
  std::unordered_map<const FunctionObj*, std::string> func_name_map_;
  // Current visit mode.
  VisitMode mode_ = VisitMode::kDefault;
  // set of context variables.
  std::unordered_set<Var> var_set_;
  std::unordered_set<Var> recur_vars_;
  std::unordered_set<DataflowVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> dataflow_var_set_;
  std::unordered_map<Var, const FunctionObj*> param_var_func_map_;

  decltype(Op::GetAttrMap(tvm::relax::op_attr::kNormalize)) op_map_normalize_ =
      Op::GetAttrMap(tvm::relax::op_attr::kNormalize);
  decltype(Op::GetAttrMap(tvm::op_attr::kInferType)) op_map_infer_type_ =
      Op::GetAttrMap(tvm::op_attr::kInferType);
  decltype(Op::GetAttrMap(
      tvm::relax::op_attr::kInferTypeWithBuilder)) op_map_infer_type_with_builder_ =
      Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder);
};

void CheckWellFormed(ffi::Variant<IRModule, Function> obj, bool check_ty) {
  WellFormedChecker::Check(obj, check_ty);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("relax.analysis.check_well_formed", CheckWellFormed);
}

}  // namespace relax
}  // namespace tvm
