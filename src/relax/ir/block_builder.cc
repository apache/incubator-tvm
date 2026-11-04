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
 * \file src/relax/block_builder.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_functor.h>
#include <tvm/relax/utils.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Block builder have three categories of logics that are interdependent with each other.
//
// The logics are somewhat interdependent with each other.
// To help us implement a block builder in two parts:
//
// - BlockBuilderImpl: implements ctx and scope management, with no normalization.
// - BlockBuilderImplWithNormalize: subclasses BlockBuilderImpl and implements normalization.
//
// The final blockbuilder create will be backed by BlockBuilderWithNormalize

namespace tvm {
namespace relax {
using namespace tvm::prim;

//---------------------------------------
// ctx and scope management.
//---------------------------------------
class BlockBuilderImpl : public BlockBuilderObj {
 public:
  explicit BlockBuilderImpl(IRModule context_mod) : context_mod_(std::move(context_mod)) {}

  ~BlockBuilderImpl() {
    if (!block_stack_.empty()) {
      LOG(WARNING) << "BlockBuilder destroyed with remaining blocks!";
    }
  }

  //-------------------------------
  // Global Context management
  //-------------------------------
  UniqueNameSupply name_supply() final { return name_supply_; }

  IRModule GetContextIRModule() const final { return context_mod_; }

  IRModule Finalize() final { return transform::NormalizeGlobalVar()(context_mod_); }

  GlobalVar AddFunction(const BaseFunc& func, ffi::String func_name_hint) final {
    LazyInitCtxFuncDedupMap();
    auto it = ctx_func_dedup_map_->find(func);
    if (it == ctx_func_dedup_map_->end()) {
      context_mod_.CopyOnWrite();

      ffi::String func_name = GetUniqueName(func_name_hint);
      while (context_mod_->ContainGlobalVar(func_name)) {
        func_name = GetUniqueName(func_name_hint);
      }
      GlobalVar gvar(func_name);

      Type finfo = Type::Missing();
      if (!func->ty.as<MissingType>().has_value()) {
        finfo = GetType(func);
      } else {
        TVM_FFI_THROW(RuntimeError) << "Expect ty field to be populated";
      }
      UpdateType(gvar, finfo);

      context_mod_->Add(gvar, func);

      (*ctx_func_dedup_map_)[func].insert(gvar);
      return gvar;
    } else {
      TVM_FFI_ICHECK(it->second.size())
          << "Values contained in de-duplication map must be non-empty sets, "
          << "but found an empty set for function of type " << func->GetTypeKey();
      // To provide deterministic results, return the GlobalVar that
      // comes first in lexicographic order.
      return *std::min_element(
          it->second.begin(), it->second.end(),
          [](const GlobalVar& a, const GlobalVar& b) { return a->name_hint < b->name_hint; });
    }
  }

  void UpdateFunction(const GlobalVar& gv, BaseFunc function) final {
    context_mod_.CopyOnWrite();

    // Remove function from the de-duplication map.
    if (ctx_func_dedup_map_ != nullptr) {
      auto it = context_mod_->functions.find(gv);
      if (it != context_mod_->functions.end()) {
        BaseFunc old_func = (*it).second;
        auto ptr = ctx_func_dedup_map_->find(old_func);
        TVM_FFI_ICHECK(ptr != ctx_func_dedup_map_->end())
            << "BlockBuilder::UpdateFunction is updating " << gv
            << ", which appears in the BlockBuilder's context_mod_, "
            << "but does not appear in the de-duplication map";
        TVM_FFI_ICHECK(ptr->second.count(gv))
            << "BlockBuilder::UpdateFunction is updating " << gv
            << ", but the de-duplication map for the previous value of this function "
            << "does not include " << gv;
        ptr->second.erase(gv);
        if (ptr->second.empty()) {
          ctx_func_dedup_map_->erase(ptr);
        }
      }
    }

    context_mod_->Update(gv, function);

    // add new dedup map item.
    if (ctx_func_dedup_map_ != nullptr) {
      (*ctx_func_dedup_map_)[function].insert(gv);
    }
  }

  //-------------------------------
  // Scope management
  //-------------------------------
  ffi::Optional<Expr> LookupBinding(const Var& var) final {
    auto it = binding_table_.find(var);
    if (it == binding_table_.end()) return std::nullopt;
    return it->second;
  }

  void BeginDataflowBlock() final { block_stack_.emplace_back(BindingBlockFrame{{}, true}); }

  void BeginBindingBlock() final { block_stack_.emplace_back(BindingBlockFrame{{}, false}); }

  void BeginScope(ffi::Optional<ffi::Array<Var>> params) final {
    // The current implementation handles the collection of shape var
    // defined in parameter type annotations. The implementation
    // is correct (since we will simply erase all relax Vars in EraseToWellDefined),
    // but can be further improved.
    //
    // TODO(relax-team): Add support for relax Var in type annotations.

    scope_stack_.emplace_back(ScopeFrame());
    if (params.has_value()) {
      for (const auto& param : params.value()) {
        AddDefinitionToScope(param);
      }
    }
  }

  void BeginInnerScope() final {
    if (scope_stack_.size()) {
      scope_stack_.emplace_back(scope_stack_.back());
    } else {
      scope_stack_.emplace_back(ScopeFrame());
    }
  }

  void AddDefinitionToScope(Var var) final {
    if (scope_stack_.empty()) {
      return;
    }

    auto& shape_var_map = CurrentScopeFrame()->shape_var_map;

    // The current implementation handles the collection of shape var
    // defined in parameter type annotations. The implementation
    // is correct (since we will simply erase all relax Vars in EraseToWellDefined),
    // but can be further improved.
    ffi::Map<PrimVar, PrimExpr> var_map = TypeVarCollector::Collect(GetType(var));
    for (const auto& kv : var_map) {
      const PrimVar& shape_var = kv.first;
      const PrimExpr& shape_expr = kv.second;
      auto it = shape_var_map.find(shape_var);
      if (it == shape_var_map.end()) {
        shape_var_map.Set(shape_var, shape_expr);
        // Expose the shape variable as non-negative, for purposes
        // of shape inference.  In many cases, knowning that the
        // shape variable is non-negative allows for simpler
        // expressions for dynamic shapes.
        auto context = analyzer_->GetConstraintContext();
        context->Append(sym::PredicateConstraint(shape_var >= 0));
        analyzer_->SyncConstraintContext(context);
      } else {
        const PrimExpr& old_shape_expr = (*it).second;
        TVM_FFI_ICHECK(old_shape_expr.same_as(shape_expr) ||
                       analyzer_->CanProveEqual(old_shape_expr, shape_expr))
            << "Inconsistent shape var " << shape_var << " in scope: " << old_shape_expr << " vs "
            << shape_expr;
      }
    }
  }

  void EndScope() final { scope_stack_.pop_back(); }

  BindingBlock EndBlock() final {
    BindingBlockFrame* cur_frame = CurrentBindingBlockFrame();
    BindingBlock ret = cur_frame->is_dataflow ? DataflowBlock(cur_frame->bindings)
                                              : BindingBlock(cur_frame->bindings);
    block_stack_.pop_back();
    return ret;
  }

  bool CurrentBlockIsDataFlow() final { return CurrentBindingBlockFrame()->is_dataflow; }

  Var Emit(Expr expr, ffi::String name_hint) final {
    return this->Emit(expr, CurrentBindingBlockFrame()->is_dataflow, name_hint);
  }

  Var EmitMatchCast(Expr value, Type ty, ffi::String name_hint) final {
    value = this->Normalize(value);

    TVM_FFI_ICHECK(TypeBaseCheck(GetType(value), ty) != BaseCheckResult::kFailL0)
        << "It is impossible to match cast any value into the target ty. "
           "But got value type: "
        << GetType(value) << ", given type: " << ty;

    // NOTE: do match cast checking later in a pass.
    BindingBlockFrame* cur_frame = CurrentBindingBlockFrame();
    Var var = CreateVar(cur_frame->is_dataflow, name_hint);
    UpdateType(var, ty);

    MatchCast match_cast(var, value, ty);
    cur_frame->bindings.push_back(match_cast);
    // NOTE match shape do not follow simple binding rule
    // as a result should not appear in binding table.

    AddDefinitionToScope(var);
    return var;
  }

  Var EmitOutput(Expr output, ffi::String name_hint) final {
    BindingBlockFrame* cur_frame = CurrentBindingBlockFrame();

    TVM_FFI_ICHECK(cur_frame->is_dataflow) << "EmitOutput has to be called inside dataflow block.";

    return Emit(output, false, name_hint);
  }

  void EmitNormalized(Binding binding) final {
    BindingBlockFrame* cur_frame = CurrentBindingBlockFrame();

    if (const auto* var_binding = binding.as<VarBindingObj>()) {
      if (!cur_frame->is_dataflow) {
        TVM_FFI_ICHECK(!var_binding->var.as<DataflowVarObj>())
            << "Cannot emit dataflow var in non-dataflow block";
      }
      // normalized check
      TVM_FFI_ICHECK(!var_binding->var->ty.as<MissingType>().has_value());
      TVM_FFI_ICHECK(!var_binding->value->ty.as<MissingType>().has_value());
      cur_frame->bindings.push_back(binding);
      binding_table_.insert_or_assign(var_binding->var, var_binding->value);
    } else if (const auto* match_cast = binding.as<MatchCastObj>()) {
      if (!cur_frame->is_dataflow) {
        TVM_FFI_ICHECK(!match_cast->var.as<DataflowVarObj>())
            << "Cannot emit dataflow var in non-dataflow block";
      }
      // normalized check
      TVM_FFI_ICHECK(!match_cast->var->ty.as<MissingType>().has_value());
      TVM_FFI_ICHECK(!match_cast->value->ty.as<MissingType>().has_value());
      // NOTE match shape do not follow simple binding rule
      // as a result should not appear in binding table.
      cur_frame->bindings.push_back(binding);
      AddDefinitionToScope(match_cast->var);
    } else {
      TVM_FFI_THROW(InternalError) << "Unsupported binding type: " << binding->GetTypeKey();
    }
  }

  sym::Analyzer GetAnalyzer() final { return analyzer_; }

 protected:
  /*!
   * \brief A representation of a block frame.
   *
   * A block frame is a record containing the bindings needed
   * to build a binding block, and a boolean to indicate if the
   * block being built is a DataflowBlock or not.
   */
  struct BindingBlockFrame {
    /*!
     * \brief List of bindings
     */
    ffi::Array<Binding> bindings;
    /*! \brief Whether current block is dataflow block. */
    bool is_dataflow;
    /*!
     * \brief Binding map used by normalizer.
     *
     * \note The normalizer only caches reuse in the current block scope
     *       and will not cache bindings from parent scope.
     */
    std::unordered_map<Expr, Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> normalize_binding_map;
  };
  /*!
   * \brief A representation of a scope frame.
   *
   * A scope frame records tracks the context of current scope.
   */
  struct ScopeFrame {
    // NOTE: for simplicity, only tracks symbolic var for now
    // the scope is only used for erasure, so less information means
    // more conservative analysis.
    // Consider impl alternative: merge with block frame if we have more frame kinds.
    //
    // TODO(relax-team) tracks the var defined also through match-cast.
    /*! \brief set of defined symbolic vars, value as themself. */
    ffi::Map<PrimVar, PrimExpr> shape_var_map;
  };

  /*! \brief A stack to store block frames. */
  std::vector<BindingBlockFrame> block_stack_;

  /*! \brief A stack to store scope frames. */
  std::vector<ScopeFrame> scope_stack_;

  /*! \brief A binding table that maps var to value. */
  std::unordered_map<Var, Expr, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> binding_table_;

  /*! \brief A unique name supply to get unique names for IR construction. */
  UniqueNameSupply name_supply_;

  /*! \brief The IRModule being built by the BlockBuilder. */
  IRModule context_mod_;

  /*! \brief Internal analzyer */
  sym::Analyzer analyzer_;

  /*!
   * \return The current frame.
   * \note Never hold the value of current frame between Normalize
   *       or other scope calls this value can change if the block stack get updated,
   *       then the block frame is no longer valid.
   */
  BindingBlockFrame* CurrentBindingBlockFrame() {
    TVM_FFI_ICHECK(!block_stack_.empty()) << "no block is being built";
    return &block_stack_.back();
  }

  /*!
   * \return The current scope frame.
   * \note only use this value
   */
  ScopeFrame* CurrentScopeFrame() {
    TVM_FFI_ICHECK(!scope_stack_.empty()) << "no scope is being opened";
    return &scope_stack_.back();
  }

  /*!
   * \brief Emits an Expr, and returns the variable it is bound to.
   * \param expr The Expr to be emitted.
   * \param is_dataflow Is the bound variable a DataflowVar or not(i.e. Var).
   * \param name_hint Name hint for the bound variable.
   * \note This Emit function normalizes the \p expr,
   *       and performs shape/type deductions by calling Normalize.
   * \return The new variable that \p expr is bound to.
   */
  Var Emit(Expr expr, bool is_dataflow, ffi::String name_hint) {
    expr = this->Normalize(expr);

    Var var = CreateVar(is_dataflow, name_hint);

    // set the values
    UpdateType(var, expr->ty.as_or_throw<Type>());

    CurrentBindingBlockFrame()->bindings.push_back(VarBinding(var, expr));

    // update the binding table
    binding_table_.insert_or_assign(var, expr);

    return var;
  }

  /*!
   * \brief Create var for bindings
   * \param is_dataflow Is the bound variable a DataflowVar or not(i.e. Var).
   * \param name_hint Name hint for the bound variable.
   * \return The created var.
   */
  Var CreateVar(bool is_dataflow, ffi::String name_hint) {
    if (name_hint.empty()) {
      name_hint = is_dataflow ? "lv" : "gv";
    }
    name_hint = GetUniqueName(name_hint);
    return is_dataflow ? DataflowVar(name_hint, /*ty_annotation=*/std::nullopt)
                       : Var(name_hint, Type::Missing());
  }

 private:
  std::string GetUniqueName(const std::string& prefix) {
    return name_supply_->FreshName(prefix, /*add_prefix*/ false, /*add_underscore*/ false);
  }

  /*! \brief A custom structural hashing that ignores Tensor raw data. */
  class StructuralHashIgnoreNDarray {
   public:
    uint64_t operator()(const ffi::ObjectRef& key) const {
      return ffi::StructuralHash::Hash(key, /*map_free_vars=*/false,
                                       /*skip_tensor_content=*/true);
    }
  };

  /*!
   * \brief Structural equality that DOES compare tensor (constant) data content. The hash above
   * intentionally ignores tensor content for speed, but the equality must stay exact: otherwise two
   * grouped functions that differ only in their bound constants (e.g. two conv layers with
   * different weights) would be incorrectly treated as duplicates and merged.
   */
  class StructuralEqualConsiderNDarray {
   public:
    bool operator()(const ffi::ObjectRef& lhs, const ffi::ObjectRef& rhs) const {
      return ffi::StructuralEqual::Equal(lhs, rhs, /*map_free_vars=*/false,
                                         /*skip_tensor_content=*/false);
    }
  };

  /*!
   * \brief A hashmap to store the mapping of Relax functions and TIR Functions
   * in context_mod to their GlobalVar to avoid generating duplicated functions.
   * We use a custom hash to avoid hashing constants that may be bound to each BaseFunc.
   */
  std::unique_ptr<std::unordered_map<
      BaseFunc, std::unordered_set<GlobalVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>,
      StructuralHashIgnoreNDarray, StructuralEqualConsiderNDarray>>
      ctx_func_dedup_map_ = nullptr;

  /*!
   * \brief lazily initialize function dedeup map.
   */
  void LazyInitCtxFuncDedupMap() {
    if (ctx_func_dedup_map_ != nullptr) return;
    ctx_func_dedup_map_ = std::make_unique<std::unordered_map<
        BaseFunc, std::unordered_set<GlobalVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>,
        StructuralHashIgnoreNDarray, StructuralEqualConsiderNDarray>>();
    for (const auto& kv : context_mod_->functions) {
      const GlobalVar gv = kv.first;
      const BaseFunc func = kv.second;
      (*ctx_func_dedup_map_)[func].insert(gv);
    }
  }

  // Collect all the variables that a parameter var can define.
  // The collector is used to making sure that we record the
  // shape vars as defined when calling BeginScope(params)
  class TypeVarCollector : public TypeVisitor {
   public:
    static ffi::Map<PrimVar, PrimExpr> Collect(const Type& ty) {
      TypeVarCollector collector;
      collector(ty);
      return collector.shape_var_map_;
    }

   private:
    void VisitType_(const TensorTypeObj* op) final {
      if (const auto* shape_expr = op->shape.as<ShapeExprObj>()) {
        for (const PrimExpr& s : shape_expr->values) {
          // Only collect single var defined shape. Ignore something like `R.Tensor((m + 1, n + 1))
          if (auto var = s.as<PrimVar>()) {
            shape_var_map_.Set(var.value(), s);
          }
        }
      }
    }

    void VisitType_(const ShapeTypeObj* op) final {
      for (const PrimExpr& s : op->values.value_or(ffi::Array<PrimExpr>())) {
        // Only collect single var defined shape. Ignore something like `R.Shape((m + 1, n + 1))
        if (auto var = s.as<PrimVar>()) {
          shape_var_map_.Set(var.value(), s);
        }
      }
    }

   private:
    ffi::Map<PrimVar, PrimExpr> shape_var_map_;
  };
};

//---------------------------------------
// Normalization
//---------------------------------------
#define RELAX_EXPR_NORMALIZER_LEAF(OP) \
  Expr VisitExpr_(const OP* op) final { return ffi::GetRef<Expr>(op); }

// TODO(relax-team): Check normalize logic after type.

// Normalizer on type:
//
// We take benefit of the following invariants(that are checked in constructor):
// - If an expr appears in Type, then it is already normalized.
//   As a result, we do not need to peek into Type in Normalization.
// - GenericConst, ShapeExpr, already have their Type populated in constructing time.
class Normalizer : public BlockBuilderImpl, private ExprFunctor<Expr(const Expr&)> {
 public:
  explicit Normalizer(IRModule context_mod) : BlockBuilderImpl(context_mod) {}
  explicit Normalizer(IRModule context_mod,
                      BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript)
      : BlockBuilderImpl(context_mod), apply_f_normalize_(false) {}

  Expr Normalize(const Expr& expr) final {
    Expr normalized = this->VisitExpr(expr);
    // Invariant:
    // After Normalize: an Expr always have
    // ty (with the exception of Op).
    if (!normalized->IsInstance<OpObj>()) {
      TVM_FFI_ICHECK(!normalized->ty.as<MissingType>().has_value())
          << "The ty of an Expr except OpObj after "
             "normalization must not be missing. However, this Expr does not have ty: "
          << normalized;
    }

    return normalized;
  }

  /*!
   * \brief Normalize Argument values to call and other IR sub-fields.
   * \param arg The argument.
   * \return The normalized value.
   *
   * \note This function create a new binding for non-leaf expressions except for tuple.
   */
  Expr NormalizeArgument(const Expr& arg) final {
    if (!block_stack_.empty()) {
      // cache lookup
      BindingBlockFrame* cur_frame = CurrentBindingBlockFrame();
      auto it = cur_frame->normalize_binding_map.find(arg);
      if (it != cur_frame->normalize_binding_map.end()) {
        return it->second;
      }
    }
    // skip visit expr's cache, normalize arg
    Expr post = ExprFunctor::VisitExpr(arg);

    if (!IsLeafOrTuple(arg)) {
      TVM_FFI_ICHECK(!block_stack_.empty()) << "Cannot normalize non-leaf without a scope";
      Var var = this->Emit(post, "");
      // NOTE: current frame addr can change due to underlying vector
      // re-allocation, redo lookup
      CurrentBindingBlockFrame()->normalize_binding_map.insert_or_assign(arg, var);
      return var;
    } else {
      return post;
    }
  }

  RELAX_EXPR_NORMALIZER_LEAF(ExternFuncObj);
  RELAX_EXPR_NORMALIZER_LEAF(GlobalVarObj);
  RELAX_EXPR_NORMALIZER_LEAF(OpObj);
  RELAX_EXPR_NORMALIZER_LEAF(GenericConstObj);
  RELAX_EXPR_NORMALIZER_LEAF(ShapeExprObj);
  RELAX_EXPR_NORMALIZER_LEAF(StringImmObj);
  RELAX_EXPR_NORMALIZER_LEAF(DataTypeImmObj);

  Expr VisitExprDefault_(const ffi::Object* op) final {
    Expr expr = ffi::GetRef<Expr>(static_cast<const ExprObj*>(op));
    if (expr.as<PrimExpr>() || expr.as<OpaqueExpr>()) return expr;
    return ExprFunctor::VisitExprDefault_(op);
  }

  template <typename T>
  Expr VisitVar_(const typename T::ContainerType* var) {
    // Parameters and free-vars must be present with type
    // Other vars must have already been normalized through binding
    TVM_FFI_ICHECK(!var->ty.template as<MissingType>().has_value())
        << "Var " << var->name << " does not have type.";
    return ffi::GetRef<Var>(var);
  }

  Expr VisitExpr_(const VarObj* var_ptr) final {
    auto var = VisitVar_<Var>(var_ptr);
    if (HasVoidType(var)) {
      return VisitExpr(Tuple(ffi::Array<Expr>{}));
    } else {
      return var;
    }
  }

  Expr VisitExpr_(const DataflowVarObj* var) final { return VisitVar_<DataflowVar>(var); }

  Expr VisitExpr(const Expr& expr) final {
    // lookup normalize map
    if (!block_stack_.empty()) {
      BindingBlockFrame* cur_frame = CurrentBindingBlockFrame();
      auto it = cur_frame->normalize_binding_map.find(expr);
      if (it != cur_frame->normalize_binding_map.end()) {
        return it->second;
      }
    }
    return ExprFunctor::VisitExpr(expr);
  }

  Expr VisitExpr_(const TupleObj* op) final {
    bool unchanged = true;
    ffi::Array<Expr> new_fields;

    for (const Expr& field : op->fields) {
      Expr new_field = this->NormalizeArgument(field);
      new_fields.push_back(new_field);
      unchanged &= new_field.same_as(field);
    }

    Tuple tuple = unchanged ? ffi::GetRef<Tuple>(op) : Tuple(new_fields, op->loc);
    // Update tuple fields.
    if (tuple->ty.as<MissingType>().has_value()) {
      ffi::Array<Type> tuple_ty;
      for (Expr field : tuple->fields) {
        tuple_ty.push_back(GetType(field));
      }
      UpdateType(tuple, TupleType(tuple_ty, op->loc));
    }
    return tuple;
  }

  Expr VisitExpr_(const FunctionObj* op) final {
    Expr new_body = this->VisitWithNewScope(op->body, op->params);

    if (new_body.same_as(op->body)) {
      return ffi::GetRef<Function>(op);
    } else {
      return Function(op->params, new_body, op->ret_ty, op->is_pure, op->attrs);
    }
  }

  Expr VisitExpr_(const CallObj* op) final {
    TVM_FFI_CHECK(op->op.as<OpObj>() || !op->op->ty.as<tvm::FuncTypeObj>(), TypeError)
        << "Ordinary Relax calls cannot invoke a native TIRx function; "
        << "use R.call_tir for destination passing or R.call_tir_packed for a direct result";
    Expr new_op = this->NormalizeArgument(op->op);

    ffi::Array<Expr> new_args =
        op->args.Map([this](const Expr& arg) { return NormalizeArgument(arg); });

    Call call = [&]() -> Call {
      if (new_op.same_as(op->op) && new_args.same_as(op->args)) {
        return ffi::GetRef<Call>(op);
      } else {
        return Call(Type::Missing(), new_op, new_args, op->attrs, op->ty_args);
      }
    }();

    if (call->ty.as<MissingType>().has_value()) {
      auto inferred_ty = InferType(call);
      UpdateType(call, inferred_ty);
    }

    // If the operation has defined a custom normalization
    // function using the FNormalize attribute, apply it.  If the
    // normalization modified the expression, re-visit in case it
    // produced a nested expression.
    if (apply_f_normalize_) {
      if (auto func_normalize = op_map_normalize_.get(op->op, nullptr); func_normalize != nullptr) {
        Expr normalized = func_normalize(ffi::GetRef<BlockBuilder>(this), call);
        if (!normalized.same_as(call)) {
          return VisitExpr(normalized);
        }
      }
    }

    // TVMScript uses a builder with FNormalize disabled so it can construct
    // intentionally ill-formed IR for a later check_well_formed call.
    if (apply_f_normalize_) {
      if (auto op = call->op.as<Op>()) op.value().Validate(call.get());
    }
    return call;
  }

  Expr VisitExpr_(const SeqExprObj* op) final {
    bool unchanged = true;
    ffi::Array<BindingBlock> new_blocks;
    for (BindingBlock block : op->blocks) {
      BindingBlock new_block = this->VisitBindingBlock(block);
      new_blocks.push_back(new_block);
      unchanged &= new_block.same_as(block);
    }

    // Because the input may not be normalized, the SeqExpr may occur
    // nested within another SeqExpr.  In that case, we want to use
    // whatever binding-block type the parent uses, so that we any
    // bindings collected into the prologue will be compatible with
    // the parent block.
    if (block_stack_.size() && CurrentBlockIsDataFlow()) {
      this->BeginDataflowBlock();
    } else {
      this->BeginBindingBlock();
    }

    // the body may not be a leaf expression, so check for that
    Expr new_body = this->NormalizeArgument(op->body);
    unchanged &= new_body.same_as(op->body);
    BindingBlock prologue = this->EndBlock();

    if (!prologue->bindings.empty()) {
      new_blocks.push_back(prologue);
      unchanged = false;
    }

    // Combine nearby blocks if possible
    ffi::Array<BindingBlock> normalized_blocks = NormalizeBlocks(new_blocks);
    unchanged &= normalized_blocks.same_as(new_blocks);

    SeqExpr seq_expr = [&]() -> SeqExpr {
      if (unchanged) {
        return ffi::GetRef<SeqExpr>(op);
      } else {
        return SeqExpr(normalized_blocks, new_body, op->loc);
      }
    }();

    // only do shape/type inference if the SeqExpr does not have shape/type
    if (seq_expr->ty.as<MissingType>().has_value()) {
      UpdateType(seq_expr, EraseToWellDefinedInScope(GetType(seq_expr->body)));
    }
    return seq_expr;
  }

  Expr VisitExpr_(const IfExprObj* op) final {
    Expr new_cond = this->NormalizeArgument(op->cond);
    Expr new_true = this->VisitWithNewScope(op->true_branch);
    Expr new_false = this->VisitWithNewScope(op->false_branch);

    IfExpr if_node = [&]() -> IfExpr {
      if (new_cond.same_as(op->cond) && new_true.same_as(op->true_branch) &&
          new_false.same_as(op->false_branch)) {
        return ffi::GetRef<IfExpr>(op);
      } else {
        return IfExpr(new_cond, new_true, new_false, op->loc);
      }
    }();
    if (if_node->ty.as<MissingType>().has_value()) {
      auto true_info = EraseToWellDefinedInScope(GetType(new_true));
      auto false_info = EraseToWellDefinedInScope(GetType(new_false));
      UpdateType(if_node, TypeLCA(true_info, false_info));
    }
    return if_node;
  }

  Expr VisitExpr_(const TupleGetItemObj* op) final {
    Expr new_tuple = this->NormalizeArgument(op->tuple);

    TupleGetItem node = new_tuple.same_as(op->tuple) ? ffi::GetRef<TupleGetItem>(op)
                                                     : TupleGetItem(new_tuple, op->index);

    if (node->ty.as<MissingType>().has_value()) {
      auto opt = MatchType<TupleType>(node->tuple);
      TVM_FFI_ICHECK(opt) << "The type of Tuple must be TupleType, "
                          << "but expression " << node->tuple << " has type " << node->tuple->ty;
      UpdateType(node, opt.value()->fields[node->index]);
    }

    return node;
  }

  Binding VisitBinding(const Binding& binding) {
    if (auto* var_binding = binding.as<VarBindingObj>()) {
      return this->VisitVarBinding(ffi::GetRef<VarBinding>(var_binding));
    } else {
      auto* match_cast = binding.as<MatchCastObj>();
      TVM_FFI_ICHECK(match_cast) << "Unsupported binding type: " << binding->GetTypeKey();
      return this->VisitMatchCast(ffi::GetRef<MatchCast>(match_cast));
    }
  }

  VarBinding VisitVarBinding(VarBinding binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (!new_value.same_as(binding->value)) {
      binding = VarBinding(binding->var, new_value, binding->loc);
    }
    if (binding->var->ty.as<MissingType>().has_value()) {
      UpdateType(binding->var, GetType(new_value));
    }
    return binding;
  }

  MatchCast VisitMatchCast(MatchCast binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (!new_value.same_as(binding->value)) {
      binding = MatchCast(binding->var, new_value, binding->ty, binding->loc);
    }
    if (binding->var->ty.as<MissingType>().has_value()) {
      UpdateType(binding->var, binding->ty);
    }
    return binding;
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) {
    if (block.as<DataflowBlockObj>()) {
      this->BeginDataflowBlock();
    } else {
      this->BeginBindingBlock();
    }

    bool unchanged = true;
    for (const Binding& binding : block->bindings) {
      Binding new_binding = this->VisitBinding(binding);
      unchanged &= new_binding.same_as(binding);

      this->EmitNormalized(new_binding);
    }
    BindingBlock new_block = this->EndBlock();
    unchanged &= new_block->bindings.size() == block->bindings.size();
    if (unchanged) {
      return block;
    }
    return new_block;
  }

 private:
  // Helper function to infer the type of a Call.
  Type InferType(const Call& call) {
    if (auto* op_ptr = call->op.as<OpObj>()) {
      // Case 1: the op field is a primitive op, look up FInferTypeWithBuilder attribute
      Op op = ffi::GetRef<Op>(op_ptr);
      bool is_dist_op = false;
      for (const auto& arg : call->args) {
        if (arg->ty.as<distributed::DTensorTypeObj>()) {
          is_dist_op = true;
          break;
        }
      }
      if (is_dist_op) {
        for (const auto& arg : call->args) {
          TVM_FFI_ICHECK(!arg->ty.as<TensorTypeObj>())
              << "Distributed operator must take DTensor instead of Tensor as input";
        }
        TVM_FFI_ICHECK(op_map_dist_infer_ty.count(op))
            << " Cannot find the dist.FInferTypeWithBuilder attribute registered to op: "
            << op->name;
        return op_map_dist_infer_ty[op](call, ffi::GetRef<BlockBuilder>(this));
      }
      bool has_context_free = op_map_context_free_infer_ty.count(op);
      bool has_contextual = op_map_infer_ty.count(op);
      TVM_FFI_ICHECK(!(has_context_free && has_contextual))
          << "Operator " << op->name << " registers both FInferType and FInferTypeWithBuilder";
      if (has_context_free) {
        return Call::ReinferType(call.get());
      }
      TVM_FFI_ICHECK(has_contextual)
          << " Cannot find the FInferType or FInferTypeWithBuilder attribute registered to op: "
          << op->name;
      return op_map_infer_ty[op](call, ffi::GetRef<BlockBuilder>(this));
    } else {
      // derive using function parameters
      TVM_FFI_ICHECK(!call->op->ty.as<MissingType>().has_value());
      auto opt = MatchType<FuncType>(call->op);
      TVM_FFI_ICHECK(opt) << "Call->op must contains a function type";
      FuncType finfo = opt.value();
      return DeriveCallRetType(finfo, call, ffi::GetRef<BlockBuilder>(this), analyzer_);
    }
  }

  // erase to well defined within current scope.
  Type EraseToWellDefinedInScope(Type info) {
    if (scope_stack_.empty()) {
      // If no scopes are active, then this fragment does not require
      // any normalization.
      return info;
    }
    auto* curr_scope = CurrentScopeFrame();
    auto f_var_map = [curr_scope](const Var& var) -> ffi::Optional<Expr> {
      auto prim_var = var.as<PrimVar>();
      if (!prim_var) return std::nullopt;
      auto it = curr_scope->shape_var_map.find(prim_var.value());
      if (it != curr_scope->shape_var_map.end()) return (*it).second;
      return std::nullopt;
    };
    return EraseToWellDefined(info, f_var_map);
  }

  Expr VisitWithNewScope(const Expr& expr, ffi::Optional<ffi::Array<Var>> params = std::nullopt) {
    if (params.has_value()) {
      this->BeginScope(params.value());
    } else {
      this->BeginInnerScope();
    }

    Expr ret = [&]() -> Expr {
      // SeqExpr do not need to prepare for normalization.
      if (expr.as<SeqExprObj>()) {
        return this->VisitExpr(expr);
      } else {
        this->BeginBindingBlock();
        Expr post = this->NormalizeArgument(expr);
        BindingBlock prologue = this->EndBlock();
        // "New scopes" (function bodies, if/else clauses) must be wrapped in seq exprs.
        // Don't wrap if it's already a seq and there are no bindings to add
        if (post.as<SeqExprObj>() && prologue->bindings.empty()) {
          return post;
        }
        ffi::Array<BindingBlock> bindings;
        if (!prologue->bindings.empty()) {
          bindings.push_back(prologue);
        }

        SeqExpr seq(bindings, post);
        UpdateType(seq, EraseToWellDefinedInScope(GetType(seq->body)));

        return seq;
      }
    }();
    this->EndScope();
    return ret;
  }

  ffi::Array<BindingBlock> FlattenBlocks(const ffi::Array<BindingBlock>& blocks) {
    // If there is a binding that is a seq expr, split the current block,
    // add the nested blocks prior to the seq expr, and bind the seq expr body
    // to the var
    ffi::Array<BindingBlock> ret;
    bool changed = false;
    for (const BindingBlock& block : blocks) {
      bool is_dataflow = block->IsInstance<DataflowBlockObj>();
      ffi::Array<Binding> current;
      for (const Binding& binding : block->bindings) {
        Expr value = [&]() -> Expr {
          if (const auto* var_binding = binding.as<VarBindingObj>()) {
            return var_binding->value;
          } else if (const auto* match_cast = binding.as<MatchCastObj>()) {
            return match_cast->value;
          } else {
            TVM_FFI_THROW(InternalError) << "Unknown binding type: " << binding->GetTypeKey();
            throw;
          }
        }();
        // if we encounter a nested seq, we have to flatten it:
        //   1. Append the binding block we've accumulated so far
        //   2. Reset the current block
        //   3. Append the inner blocks
        //   4. Add a binding of the current var to the seq expr's body to the current block
        // then continue
        if (auto seq = value.as<SeqExprObj>()) {
          changed = true;
          ret.push_back(is_dataflow ? DataflowBlock(current) : BindingBlock(current));
          current = {};
          // We do not need to flatten recursively because the normalizer will have normalized
          // and thus flattened the inner SeqExprs already
          for (const BindingBlock& block : seq->blocks) {
            if (is_dataflow && !block->IsInstance<DataflowBlockObj>()) {
              // A DataflowBlock occurring within a non-DataflowBlock
              // usually is an error, resulting from return of a
              // `BindingBlock`.  However, it may still be well-formed
              // if there are no relax::DataflowVar instances used by
              // the non-DataflowBlock.  This would result in multiple
              // dataflow sections, split by non-dataflow portions,
              // but would still be valid.
              //
              // Since the most common occurrence is due to mis-use,
              // explicitly check for it here rather than waiting for a
              // WellFormed check later on.

              auto free_vars = FreeVars(SeqExpr({block}, Tuple(ffi::Array<Expr>{})));
              ffi::Array<DataflowVar> free_dataflow_vars;
              for (const auto& var : free_vars) {
                if (auto opt = var.as<DataflowVar>()) {
                  free_dataflow_vars.push_back(opt.value());
                }
              }

              if (free_dataflow_vars.size()) {
                TVM_FFI_THROW(InternalError)
                    << "Malformed AST: "
                    << "A DataflowVar may only be used within a DataflowBlock.  "
                    << "The variable " << binding->var << " is defined within a DataflowBlock, "
                    << "but is bound to a SeqExpr that contains non-dataflow BindingBlocks.  "
                    << "These non-dataflow BindingBlocks use the DataflowVars "
                    << free_dataflow_vars << ", which is invalid.";
              }
            }
            ret.push_back(block);
          }

          if (const auto* var_binding = binding.as<VarBindingObj>()) {
            current.push_back(VarBinding(var_binding->var, seq->body));
          } else if (const auto* match_cast = binding.as<MatchCastObj>()) {
            current.push_back(MatchCast(match_cast->var, seq->body, match_cast->ty));
          } else {
            TVM_FFI_THROW(InternalError) << "Unknown binding type: " << binding->GetTypeKey();
            throw;
          }
        } else {
          current.push_back(binding);
        }
      }
      ret.push_back(is_dataflow ? DataflowBlock(current) : BindingBlock(current));
    }
    return changed ? ret : blocks;
  }

  ffi::Array<BindingBlock> NormalizeBlocks(const ffi::Array<BindingBlock>& blocks) {
    bool changed = false;
    ffi::Array<BindingBlock> ret;
    auto flattened = FlattenBlocks(blocks);
    if (!flattened.same_as(blocks)) {
      changed = true;
    }
    for (const BindingBlock& block : flattened) {
      if (block->bindings.empty()) {
        // Case 1. Skip empty blocks
        changed = true;
      } else if (!ret.empty() && ret.back()->type_index() == block->type_index()) {
        // Case 2. Merge with previous block if possible
        BindingBlock merged(ffi::UnsafeInit{});
        // NOTE: should check DataflowBlockObj first.
        if (const auto* dataflow_block = ret.back().as<DataflowBlockObj>()) {
          auto n = ffi::make_object<DataflowBlockObj>(*dataflow_block);
          n->bindings.insert(n->bindings.end(), block->bindings.begin(), block->bindings.end());
          merged = DataflowBlock(ffi::UnsafeInit{}, std::move(n));
        } else if (const auto* binding_block = ret.back().as<BindingBlockObj>()) {
          auto n = ffi::make_object<BindingBlockObj>(*binding_block);
          n->bindings.insert(n->bindings.end(), block->bindings.begin(), block->bindings.end());
          merged = BindingBlock(ffi::UnsafeInit{}, std::move(n));
        } else {
          TVM_FFI_THROW(InternalError) << "Unknown block type: " << ret.back()->GetTypeKey();
        }
        ret.pop_back();
        ret.push_back(merged);
        changed = true;
      } else {
        // Case 3. Add to the result
        ret.push_back(block);
      }
    }
    return changed ? ret : blocks;
  }

  /*! \brief Operator type inference map. */
  decltype(Op::GetAttrMap(tvm::op_attr::kInferType)) op_map_context_free_infer_ty =
      Op::GetAttrMap(tvm::op_attr::kInferType);
  decltype(Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder)) op_map_infer_ty =
      Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder);
  decltype(Op::GetAttrMap(
      tvm::relax::distributed::op_attr::kInferTypeWithBuilder)) op_map_dist_infer_ty =
      Op::GetAttrMap(tvm::relax::distributed::op_attr::kInferTypeWithBuilder);
  /*! \brief Operator normalization function */
  decltype(Op::GetAttrMap(tvm::relax::op_attr::kNormalize)) op_map_normalize_ =
      Op::GetAttrMap(tvm::relax::op_attr::kNormalize);

  /*! \brief Whether the FNormalize function should be applied */
  bool apply_f_normalize_{true};
};

BlockBuilder BlockBuilder::Create(ffi::Optional<IRModule> mod) {
  ffi::ObjectPtr<BlockBuilderObj> n = ffi::make_object<Normalizer>(mod.value_or(IRModule()));
  return BlockBuilder(ffi::UnsafeInit{}, std::move(n));
}

BlockBuilder BlockBuilder::Create(ffi::Optional<IRModule> mod,
                                  BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript) {
  ffi::ObjectPtr<BlockBuilderObj> n = ffi::make_object<Normalizer>(
      mod.value_or(IRModule()), BlockBuilder::DisableOperatorSpecificNormalizationForTVMScript());
  return BlockBuilder(ffi::UnsafeInit{}, std::move(n));
}

//---------------------------------------
// User facing function registration.
//---------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<BlockBuilderObj>();
  refl::GlobalDef()
      .def("relax.BlockBuilderCreate",
           [](ffi::Optional<IRModule> mod) { return BlockBuilder::Create(mod); })
      .def_method("relax.BlockBuilderBeginDataflowBlock", &BlockBuilderObj::BeginDataflowBlock)
      .def_method("relax.BlockBuilderBeginBindingBlock", &BlockBuilderObj::BeginBindingBlock)
      .def_method("relax.BlockBuilderEndBlock", &BlockBuilderObj::EndBlock)
      .def_method("relax.BlockBuilderNormalize", &BlockBuilderObj::Normalize)
      .def("relax.BlockBuilderEmit",
           [](BlockBuilder builder, Expr expr, ffi::String name_hint) {
             return builder->Emit(expr, name_hint);
           })
      .def("relax.BlockBuilderEmitMatchCast",
           [](BlockBuilder builder, Expr value, Type ty, ffi::String name_hint) {
             return builder->EmitMatchCast(value, ty, name_hint);
           })
      .def("relax.BlockBuilderEmitOutput",
           [](BlockBuilder builder, const Expr& output, ffi::String name_hint) {
             return builder->EmitOutput(output, name_hint);
           })
      .def("relax.BlockBuilderEmitNormalized",
           [](BlockBuilder builder, Binding binding) { return builder->EmitNormalized(binding); })
      .def("relax.BlockBuilderGetUniqueName",
           [](BlockBuilder builder, ffi::String name_hint) {
             return builder->name_supply()->FreshName(name_hint, /*add_prefix*/ false,
                                                      /*add_underscore*/ false);
           })
      .def_method("relax.BlockBuilderAddFunction", &BlockBuilderObj::AddFunction)
      .def_method("relax.BlockBuilderUpdateFunction", &BlockBuilderObj::UpdateFunction)
      .def_method("relax.BlockBuilderGetContextIRModule", &BlockBuilderObj::GetContextIRModule)
      .def_method("relax.BlockBuilderFinalize", &BlockBuilderObj::Finalize)
      .def_method("relax.BlockBuilderCurrentBlockIsDataFlow",
                  &BlockBuilderObj::CurrentBlockIsDataFlow)
      .def_method("relax.BlockBuilderLookupBinding", &BlockBuilderObj::LookupBinding)
      .def_method("relax.BlockBuilderBeginScope", &BlockBuilderObj::BeginScope)
      .def_method("relax.BlockBuilderEndScope", &BlockBuilderObj::EndScope);
}
}  // namespace relax
}  // namespace tvm
