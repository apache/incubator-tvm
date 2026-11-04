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
 * \file src/relax/backend/vm/vm_shape_lower.cc
 * \brief Lower the function boundary type checks and symbolic shape computations.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_functor.h>
#include <tvm/runtime/vm/builtin.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_set>

namespace tvm {
namespace relax {
using namespace tvm::prim;

/*! \brief A slot used in PrimExpr lowering. */
struct PrimExprSlot {
  PrimExprSlot(PrimExpr expr, int index) : expr(std::move(expr)), index(index) {}
  /*! \brief The existing */
  PrimExpr expr;
  /*! \brief The slot index */
  int index;
  // The following three members are auxiliary data
  // to help shape rewriting.
  /*!
   * \brief List of slots whose PrimExpr uses this PrimExpr.
   * \note Users won't be empty only if PrimExpr is a Var and it does not include itself.
   */
  std::vector<PrimExprSlot*> user_slots;
  /*!
   * \brief Number of outstanding vars that are not defined in this PrimExpr.
   * \note This is a helper counter used in analysis to perform computations.
   */
  int outstanding_defs = 0;
  /*! \brief Whether we have computed the value. */
  bool value_computed = false;
};

/*!
 * \brief Helper dats structure to collect pairs of match shapes
 *        in a recursive matching process.
 */
struct MatchShapeTodoItem {
  Expr input;
  ffi::Array<PrimExpr> pattern;
  ffi::String err_ctx;
};

/*! \brief Slot map used for shape lowering. */
using PrimExprSlotMap =
    std::unordered_map<PrimExpr, PrimExprSlot*, ffi::StructuralHash, prim::ExprDeepEqual>;

using LiveVarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

static bool IsRelaxOwnedCall(const CallObj* call) {
  auto op = call->op.as<Op>();
  if (!op) return true;
  static auto infer_type_map = Op::GetAttrMap(tvm::op_attr::kInferType);
  static auto infer_type_with_builder_map =
      Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder);
  static auto legalize_map = Op::GetAttrMap(tvm::relax::op_attr::kLegalize);
  return infer_type_map.count(op.value()) || infer_type_with_builder_map.count(op.value()) ||
         legalize_map.count(op.value());
}

// Collector to collect PrimExprSlotMap
class PrimExprSlotCollector : public ExprVisitor, public TypeVisitor {
 public:
  // collect the PrimExpr slot for a given function
  static void Collect(Function func, std::vector<std::unique_ptr<PrimExprSlot>>* slot_vec,
                      PrimExprSlotMap* slot_map) {
    PrimExprSlotCollector collector;
    collector.slot_vec_ = slot_vec;
    collector.slot_map_ = slot_map;
    VarUsageInfo usage = CollectVarUsage(func);
    collector.bound_values_ = &usage.bound_values;
    for (const Var& output : usage.outputs) collector.MarkLive(output);
    for (const Var& param : func->params) collector.VisitType(GetType(param));
    collector.VisitType(func->ret_ty);
    collector.VisitExpr(func->body);
    collector.collect_slots_ = true;
    for (const Var& param : func->params) {
      collector.VisitType(GetType(param));
    }
    collector.VisitExpr(func->body);
    collector.VisitType(func->ret_ty);
  }

 private:
  void VisitBinding_(const VarBindingObj* op) final {
    if (!collect_slots_) {
      VisitVarDef(op->var);
      return;
    }
    bool collect_scalar = collect_scalar_;
    collect_scalar_ = live_vars_.count(op->var);
    ExprVisitor::VisitExpr(op->value);
    collect_scalar_ = collect_scalar;
    VisitVarDef(op->var);
  }

  void MarkLive(const PrimExpr& expr) {
    for (const Var& var : tvm::UndefinedVars(expr)) MarkLive(var);
  }

  void MarkLive(const Var& var) {
    if (!live_vars_.insert(var).second) return;
    if (auto it = bound_values_->find(var); it != bound_values_->end()) {
      for (const VarObj* used : GetUsedVars((*it).second)) MarkLive(ffi::GetRef<Var>(used));
    }
  }

  void VisitExpr_(const VarObj* op) final {
    Var var = ffi::GetRef<Var>(op);
    if (collect_scalar_ && !var.as<DataflowVarObj>()) {
      if (auto prim_var = var.as<PrimVar>();
          prim_var && prim_var.value().ty()->dtype == DLDataType{kDLInt, 64, 1}) {
        HandlePrimExpr(prim_var.value());
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ShapeExprObj* op) final {
    for (const PrimExpr& value : op->values) HandlePrimExpr(value);
  }

  void HandlePrimExpr(const PrimExpr& expr) {
    collect_slots_ ? CollectPrimExprSlot(expr) : MarkLive(expr);
  }

  void CollectPrimExprSlot(const PrimExpr& expr) {
    if (expr->IsInstance<IntImmObj>()) return;
    if (const auto* call = expr.as<CallObj>()) {
      TVM_FFI_CHECK(!IsRelaxOwnedCall(call), ValueError)
          << "VM shape expressions cannot compile a Relax-owned Call: " << ffi::GetRef<Call>(call);
    }
    if (slot_map_->count(expr) == 0) {
      auto slot = std::make_unique<PrimExprSlot>(expr, static_cast<int>(slot_vec_->size()));
      slot_map_->emplace(expr, slot.get());
      slot_vec_->emplace_back(std::move(slot));
    }
    for (tvm::Var var : tvm::UndefinedVars(expr)) {
      if (!var.same_as(expr)) {
        CollectPrimExprSlot(var.as_or_throw<PrimExpr>());
      }
    }
  }

  void VisitBinding_(const MatchCastObj* op) final {
    // Visit the match cast type so we can define
    // the symbolic variables here.
    this->VisitType(op->ty);
  }

  void VisitExpr_(const FunctionObj* op) final {
    // Do not recurse into function node as it is self-contained
  }

  void VisitType_(const FuncTypeObj* op) final {
    // Do not recurse into function type as it is self-contained
  }

  void VisitTypePrimExprField(const PrimExpr& expr) final { HandlePrimExpr(expr); }

  void VisitTypeExprField(const PrimExpr& expr) final { HandlePrimExpr(expr); }

  void VisitTypeExprField(const Expr& expr) final { ExprVisitor::VisitExpr(expr); }

  std::vector<std::unique_ptr<PrimExprSlot>>* slot_vec_;
  PrimExprSlotMap* slot_map_;
  const ffi::Map<Var, Expr>* bound_values_;
  LiveVarSet live_vars_;
  bool collect_slots_{false};
  bool collect_scalar_{true};
};

/*!
 * \brief Main logic to transform the shape lowered functions
 *
 * Consider the following input:
 *
 * \code
 *
 *  def f(x: R.Tuple(R.Tensor([m, n+1]), R.Tensor([n, 2])) -> R.Tensor:
 *     return x
 *
 * \endcode
 *
 * Overall flow of the algorithm:
 * - Preprocess: PrimExprSlot collection, we scan the function and allocate PrimExprSlot
 *   for each PrimExpr. In the above example, the result mapping from the slot index
 *   to expr would be {0:m, 1: n+1: 2: n}. Note that "n+1" also get a slot.
 *   PrimExprSlot also comes with auxiliary fields that track whether its value
 *   can be readily computed.
 *
 * Steps at each matching point:
 * - Step 0: We call CheckMatchCast,
 *   which will recursively unpack the Type, and generate static information checks.
 *   Note that this step only generates functions for checking types and ndim info, but not
 *   the symbolic shape variables. The symbolic shape-matching results will be returned as
 *   vector<MatchShapeTodoItem>. This is because symbolic shape matching may not be completed
 *   in a single round. Importantly, CheckMatchCast also deals with tuple unpacking.
 *
 * - Step 1: We then call RunMatch to generate the statements for matching symbolic shapes.
 *   In the above example, the first round will store the value of m, n to their corresponding
 *   slot. RunMatch may return outstanding items. In the above example x.shape[1] == n+1 cannot
 *   be checked in the first round. RunMatch will populate new vars(this case n, m), these vars
 *   are added to a ready queue (ready_vars_)
 *
 * - Step 2: We EmitOutstandingPrimExprCompute to check if ready_vars will trigger new values
 *   to be computed. We eagerly compute all the outstanding values. The trigger is done through
 *   a ref counter which decreases when each outstanding def is satisfied.
 *   This step can also generate additional TIR functions to carry out shape computations.
 *
 * - Step 3: RunMatch again for given outstanding match todos. This time all invariants
 *   should be checked.
 *
 * The above step would populate each slot(which is backed by an element in shape_heap).
 * Each time we find a symbolic shape tuple, we call MakeShape for given slot indices
 * in the shape_heap.
 *
 *
 * Key functions in the flow:
 * - PrimExprSlotCollector: preprocessing and collecting the slots
 * - CheckMatchCast: recursively structinfo unpacking, generate checks and match items.
 * - RunMatch: generate symbolic shape matches
 * - EmitOutstandingPrimExprCompute: tracks the variables to be computed and emit shape computation
 * - VisitExpr_(ShapeExprObj*): makes symbolic shape tuple.
 *
 * The checks and symbolic shape all maps to runtime builtin functions. Please checkout
 * runtime/vm/builtin.cc for their definitions.
 *
 * Shape computation are lowered to host-side TIR functions that load var from slot
 * and store computed results into the slot. For a given slot map: {0:m, 1: n+1: 2: n}
 * It will create the shape_func below that loads data from H[2](n's slot) run compute
 * and store back to H[1](n+1's slot).
 *
 * \code
 *
 * @T.function
 * def shape_func(H: T.Tensor([3], "int64")):
 *     H[1] = H[2] + 1
 *
 * \endcode
 *
 * The current implementation will batch all shape computations at each match point.
 * For example, all the expressions that depend on n, m will be computed in a single
 * shape_func at the function boundary. If there are follow-up match_cast points,
 * that defines new variable, then we might we will generate new shape functions
 * to compute expressions that depend on these variables.
 */
class VMShapeLowerMutator
    : public ExprMutator,
      public TypeFunctor<void(const Type&, Expr, bool, bool, const ffi::String&,
                              std::vector<MatchShapeTodoItem>*)> {
 public:
  static IRModule Lower(IRModule mod, bool emit_err_ctx) {
    VMShapeLowerMutator mutator(mod, emit_err_ctx);

    for (auto& kv : mod->functions) {
      if (auto* func = kv.second.as<FunctionObj>()) {
        Function updated_func = mutator.Rewrite(kv.first, ffi::GetRef<Function>(func));
        mutator.builder_->UpdateFunction(kv.first, updated_func);
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit VMShapeLowerMutator(IRModule mod, bool emit_err_ctx)
      : ExprMutator(mod), emit_err_ctx_(emit_err_ctx) {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const VarObj* op) final {
    Var var = ffi::GetRef<Var>(op);
    if (!var.as<DataflowVarObj>()) {
      if (auto prim_var = var.as<PrimVar>(); prim_var && slot_map_.count(*prim_var)) {
        return RewritePrimValue(*prim_var);
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const CallObj* op) final {
    if (auto prim_expr = ffi::GetRef<Call>(op).as<PrimExpr>();
        prim_expr && slot_map_.count(prim_expr.value()) && !IsRelaxOwnedCall(op)) {
      return RewritePrimValue(prim_expr.value());
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExprFallback_(const ExprObj* op) final {
    if (auto prim_expr = ffi::GetRef<Expr>(op).as<PrimExpr>();
        prim_expr && slot_map_.count(prim_expr.value())) {
      return RewritePrimValue(prim_expr.value());
    }
    return ExprMutator::VisitExprFallback_(op);
  }

  void VisitBinding_(const VarBindingObj* binding) final {
    if (binding->var.as<DataflowVarObj>()) {
      ExprMutator::VisitBinding_(binding);
      return;
    }
    ExprMutator::VisitBinding_(binding);
    PrimExprSlot* result_slot = GetPrimValueSlot(binding->var);
    if (result_slot != nullptr) {
      EmitRuntimePrimValueMatch(binding->var, result_slot, "");
    }
  }

  // Unit rewrite function per function.
  Function Rewrite(GlobalVar gvar, Function func) {
    // prepare mapping and heap var
    slot_vec_.clear();
    slot_map_.clear();
    current_gvar_ = gvar;
    PrimExprSlotCollector::Collect(func, &slot_vec_, &slot_map_);
    heap_size_ = IntImm(tvm::PrimType(ShapeDType()), static_cast<int64_t>(slot_vec_.size()));
    VarBinding shape_heap_binding = this->AllocShapeHeapBinding(heap_size_);
    shape_heap_ = shape_heap_binding->var;

    // prepare slot information
    this->PopulateSlotInfo();

    ffi::Array<BindingBlock> blocks;

    builder_->BeginScope(func->params);

    {
      // Check the parameter section.
      builder_->BeginBindingBlock();
      this->builder_->EmitNormalized(shape_heap_binding);
      std::vector<MatchShapeTodoItem> match_todos;
      size_t num_input = func->params.size();
      if (auto opt_num_input = func->attrs.GetAttr(tvm::relax::attr::kNumInput)) {
        // If the function has the attribute 'num_input', do shape checking on for the real inputs
        // and skip weights.
        num_input = static_cast<size_t>(opt_num_input.value());
      }
      for (size_t i = 0; i < func->params.size(); ++i) {
        Type ty = GetType(func->params[i]);
        std::ostringstream err_ctx;
        err_ctx << "ErrorContext(fn=" << gvar->name_hint << ", loc=param[" << i
                << "], param=" << func->params[i]->name << ", annotation=" << ty << ") ";
        this->CheckMatchCast(ty, func->params[i], true, i >= num_input, err_ctx.str(),
                             &match_todos);
        if (PrimExprSlot* slot = GetPrimValueSlot(func->params[i])) {
          match_todos.push_back(MatchShapeTodoItem{func->params[i], {slot->expr}, err_ctx.str()});
        }
      }
      // insert heap generation logic.
      match_todos = this->RunMatch(match_todos, false);
      this->EmitOutstandingPrimExprCompute();
      this->RunMatch(match_todos, true);

      BindingBlock pre_block = builder_->EndBlock();
      blocks.push_back(pre_block);
    }

    // new body.
    auto body_seq = this->VisitWithNewScope(func->body, func->params).as_or_throw<SeqExpr>();
    blocks.insert(blocks.end(), body_seq->blocks.begin(), body_seq->blocks.end());

    {
      // Insert the return value check
      builder_->BeginBindingBlock();
      std::ostringstream err_ctx;
      err_ctx << "ErrorContext(fn=" << gvar->name_hint
              << ", loc=return, annotation=" << func->ret_ty << ") ";
      std::vector<MatchShapeTodoItem> match_todos;
      // NOTE: the return value's shape computation must already be defined.
      this->CheckMatchCast(func->ret_ty, body_seq->body, false, false, err_ctx.str(), &match_todos);
      // NOTE: the return value's shape computation must already be defined.
      this->RunMatch(match_todos, true);
      BindingBlock post_block = builder_->EndBlock();
      blocks.push_back(post_block);
    }

    auto new_body = builder_->Normalize(SeqExpr(blocks, body_seq->body));

    current_gvar_ = std::nullopt;

    // create a new function
    return Function(func->params, new_body, func->ret_ty, func->is_pure, func->attrs);
  }

  //-------------------------------------------------------
  // PrimExpr slot handling
  //-------------------------------------------------------
  static DLDataType ShapeDType() { return DLDataType{kDLInt, 64, 1}; }

  PrimExprSlot* GetPrimValueSlot(const Var& var) const {
    if (var.as<DataflowVarObj>()) return nullptr;
    auto prim_var = var.as<PrimVar>();
    if (!prim_var) return nullptr;
    auto it = slot_map_.find(PrimExpr(*prim_var));
    return it == slot_map_.end() ? nullptr : it->second;
  }

  /*! \brief Match one runtime register against its demanded leaf slot. */
  void EmitRuntimePrimValueMatch(const Var& var, PrimExprSlot* slot, const ffi::String& err_ctx) {
    Var runtime_var = var;
    if (auto it = var_remap_.find(var); it != var_remap_.end()) {
      runtime_var = it->second;
    }

    auto [code, rvalue] = MakeMatchArgs(slot->expr, false);
    ffi::Array<Expr> args = {runtime_var, shape_heap_, IntImm::Int64(static_cast<int>(code)),
                             rvalue, GetErrContext(err_ctx)};
    builder_->Emit(Call(Type::Missing(), builtin_match_prim_value_, args, std::nullopt, {void_ty_}),
                   "_");
    this->EmitOutstandingPrimExprCompute();
  }

  /*! \brief populate additional information in the slot. */
  void PopulateSlotInfo() {
    for (auto& kv : slot_map_) {
      auto* slot = kv.second;
      if (!slot->expr.as<tvm::VarObj>()) {
        ffi::Array<tvm::Var> dep_vars = tvm::UndefinedVars(slot->expr);
        for (auto var : dep_vars) {
          auto it = slot_map_.find(var.as_or_throw<PrimExpr>());
          TVM_FFI_ICHECK(it != slot_map_.end())
              << "Var " << var << "is not defined in the function but is referenced by "
              << slot->expr;
          auto* var_slot = it->second;
          // populate the use slot.
          var_slot->user_slots.push_back(slot);
        }
        // set outstanding defs.
        slot->outstanding_defs += static_cast<int>(dep_vars.size());
      }
    }
  }
  //-------------------------------------------------------
  // Helper functions
  //-------------------------------------------------------
  StringImm GetErrContext(ffi::String err_ctx) const {
    return emit_err_ctx_ ? StringImm(err_ctx) : StringImm("");
  }

  VarBinding AllocShapeHeapBinding(IntImm heap_size) {
    if (heap_size->value > 0) {
      TensorType heap_ty(PrimType(ShapeDType()), 1);
      Var var("shape_heap", heap_ty);
      // set up the builtin func.
      Call call(Type::Missing(), call_builtin_with_ctx_op_,
                {builtin_alloc_shape_heap_, Tuple({PrimExpr(heap_size)})}, std::nullopt, {heap_ty});
      UpdateType(call, heap_ty);
      return VarBinding(var, call);
    } else {
      Var var("shape_heap", AnyType());
      Call call(Type::Missing(), null_value_op_, {});
      UpdateType(call, AnyType());
      return VarBinding(var, call);
    }
  }

  //-------------------------------------------------------
  // Expr mutation overloading.
  //-------------------------------------------------------
  Expr VisitExpr_(const FunctionObj* op) final {
    TVM_FFI_THROW(InternalError) << "VMShapeLower do not work for local functions, make sure "
                                 << " to run it after LambdaLift";
    return ffi::GetRef<Expr>(op);
  }

  std::pair<Expr, Expr> MakeSymbolicShapeArg(const PrimExpr& expr) {
    using runtime::vm::MakeShapeCode;

    if (auto* int_expr = expr.as<IntImmObj>()) {
      return {IntImm::Int64(static_cast<int>(MakeShapeCode::kUseImm)),
              IntImm::Int64(int_expr->value)};
    } else {
      auto it = slot_map_.find(expr);
      TVM_FFI_ICHECK(it != slot_map_.end());
      auto* slot = it->second;
      TVM_FFI_ICHECK(slot->value_computed)
          << "PrimExpr " << expr << " in function " << current_gvar_ << " has not been computed";
      return {IntImm::Int64(static_cast<int>(MakeShapeCode::kLoadShape)),
              IntImm::Int64(slot->index)};
    }
  }

  Expr RewritePrimValue(const PrimExpr& value) {
    using runtime::vm::MakeShapeCode;
    // Constant shape can be preserved.
    bool is_const_value = value->IsInstance<IntImmObj>() || value->IsInstance<FloatImmObj>();
    if (is_const_value) {
      return value;
    }

    ffi::Array<Expr> args = {shape_heap_};
    auto [code, value_or_index] = MakeSymbolicShapeArg(value);
    args.push_back(code);
    args.push_back(value_or_index);

    // make_shape(heap, n, c[0], r[0], c[1], r[1] ..., c[n], r[n])
    Call call(Type::Missing(), builtin_make_prim_value_, args, std::nullopt, {value.ty()});
    return call;
  }

  Expr VisitExpr_(const ShapeExprObj* op) final {
    using runtime::vm::MakeShapeCode;
    // Constant shape can be preserved.
    bool is_const_shape = std::all_of(op->values.begin(), op->values.end(),
                                      [](const PrimExpr& e) { return e->IsInstance<IntImmObj>(); });
    if (is_const_shape) {
      return ffi::GetRef<Expr>(op);
    }

    ffi::Array<Expr> args = {shape_heap_, IntImm::Int64(static_cast<int64_t>(op->values.size()))};
    for (PrimExpr expr : op->values) {
      auto [code, value_or_index] = MakeSymbolicShapeArg(expr);
      args.push_back(code);
      args.push_back(value_or_index);
    }

    // make_shape(heap, n, c[0], r[0], c[1], r[1] ..., c[n], r[n])
    Call call(Type::Missing(), builtin_make_shape_, args, std::nullopt,
              {ShapeType(static_cast<int>(op->values.size()))});
    return call;
  }

  void VisitBinding_(const MatchCastObj* binding) final {
    std::ostringstream err_ctx;
    err_ctx << "ErrorContext(match_cast, ty=" << binding->ty << ") ";
    Expr value = this->VisitExpr(binding->value);
    std::vector<MatchShapeTodoItem> match_todos;
    // always_check=false
    this->CheckMatchCast(binding->ty, value, false, false, err_ctx.str(), &match_todos);

    match_todos = this->RunMatch(match_todos, false);
    this->EmitOutstandingPrimExprCompute();
    this->RunMatch(match_todos, true);

    // These checks are emitted as extra, in codegen
    // match-cast is simply ignored and treated as a normal binding.
    ExprMutator::VisitBinding_(binding);
    if (PrimExprSlot* slot = GetPrimValueSlot(binding->var)) {
      EmitRuntimePrimValueMatch(binding->var, slot, err_ctx.str());
    }
  }

  // Do not override shape in type fields
  // We only override the shape that are already part of the normal function values
  // If future passes lift those values out into the values,
  // then codegen may not be able to handle symbolic values.
  // Place this pass as last pass before codegen.
  Type VisitExprDepTypeField(const Type& ty) final { return ty; }

  /* \brief Internal utility function used for RunMatch()
   *
   * \param expr The expression to be matched
   *
   * \param require_value_computed Whether we require all expr to be computed.
   *
   * \return The MatchShapeCode, and a relax expression specifying the
   *    argument used by that MatchShapeCode.
   */
  std::pair<runtime::vm::MatchShapeCode, Expr> MakeMatchArgs(const PrimExpr& expr,
                                                             bool require_value_computed) {
    using runtime::vm::MatchShapeCode;

    if (auto* int_expr = expr.as<IntImmObj>()) {
      return {MatchShapeCode::kAssertEqualToImm, IntImm::Int64(int_expr->value)};
    }

    auto it = slot_map_.find(expr);
    TVM_FFI_ICHECK(it != slot_map_.end());
    auto* slot = it->second;
    if (slot->value_computed) {
      return {MatchShapeCode::kAssertEqualToLoad, IntImm::Int64(slot->index)};
    }

    // the value is not yet computed
    TVM_FFI_ICHECK(!require_value_computed) << "PrimExpr " << expr << " is not computed";
    if (expr.as<tvm::VarObj>()) {
      // It is a var we will populate it in this round.

      slot->value_computed = true;
      ready_vars_.push_back(slot);

      return {MatchShapeCode::kStoreToHeap, IntImm::Int64(slot->index)};
    }

    // otherwise, we skip and mark it as outstanding
    return {MatchShapeCode::kNoOp, IntImm::Int64(0)};
  }

  //-------------------------------------------------------
  // Shape computations.
  //-------------------------------------------------------
  /*!
   * \brief Execute the match todo items.
   *
   * This function can populate vars in the match items when seeing it for the first time.
   * These new vars will be added to this->ready_vars_.
   *
   * If an item contains PrimExpr that are yet to be computed (but may be computable through
   * vars defined in this round), it will be returned to the caller.
   *
   * The caller should call EmitOutstandingPrimExprCompute, then call RunMatch again.
   *
   * \param match_todos The list of match items to be executed.
   * \param require_value_computed Whether we require all expr to be computed.
   * \return List of outstanding items that contains value that are yet to be computed.
   */
  std::vector<MatchShapeTodoItem> RunMatch(const std::vector<MatchShapeTodoItem>& match_todos,
                                           bool require_value_computed) {
    std::vector<MatchShapeTodoItem> outstanding_todos;

    using runtime::vm::MatchShapeCode;
    for (const MatchShapeTodoItem& item : match_todos) {
      bool all_nop = true;
      bool any_nop = false;

      ffi::Array<Expr> args = {item.input, shape_heap_};

      Expr match_op = item.input->ty.as<PrimTypeObj>() ? Expr(builtin_match_prim_value_)
                                                       : Expr(builtin_match_shape_);
      if (item.input->ty.as<PrimTypeObj>()) {
        TVM_FFI_ICHECK_EQ(item.pattern.size(), 1);
      } else {
        args.push_back(IntImm::Int64(item.pattern.size()));
      }

      for (PrimExpr expr : item.pattern) {
        auto [code, rvalue] = MakeMatchArgs(expr, require_value_computed);
        all_nop = all_nop && code == MatchShapeCode::kNoOp;
        any_nop = any_nop || code == MatchShapeCode::kNoOp;
        args.push_back(IntImm::Int64(static_cast<int>(code)));
        args.push_back(rvalue);
      }
      if (any_nop) {
        outstanding_todos.push_back(item);
      }
      args.push_back(GetErrContext(item.err_ctx));
      if (!all_nop) {
        Call call(Type::Missing(), match_op, args, std::nullopt, {void_ty_});
        builder_->Emit(call, "_");
      }
    }
    return outstanding_todos;
  }

  /*!
   * \brief Compute a list of prim expr that now be computed
   *        for given ready vars.
   */
  std::vector<PrimExprSlot*> GetReadyPrimExprSlots() {
    std::vector<PrimExprSlot*> to_compute;
    for (PrimExprSlot* slot : ready_vars_) {
      for (PrimExprSlot* user : slot->user_slots) {
        TVM_FFI_ICHECK_GT(user->outstanding_defs, 0);
        user->outstanding_defs -= 1;
        if (user->outstanding_defs == 0) {
          to_compute.push_back(user);
        }
      }
    }
    ready_vars_.clear();
    return to_compute;
  }

  /*!
   * \brief Check the dependent expressions of ready_vars_,
   *
   * If there are outstanding PrimExpr that can now be computed
   * we generate a tirx::Function that compute the extra shape values
   *
   * We will then clear the ready_vars.
   *
   * \return Number of PrimExpr computed.
   */
  size_t EmitOutstandingPrimExprCompute() {
    std::vector<PrimExprSlot*> to_compute = GetReadyPrimExprSlots();
    if (to_compute.size() == 0) return 0;
    TVM_FFI_ICHECK_GT(heap_size_->value, 0);
    // construct a tirx::Function that compute the shape.
    ffi::Array<PrimExpr> buffer_shape{heap_size_};
    tirx::TensorVar buffer =
        tirx::TensorVar("H", tirx::TensorType(buffer_shape, PrimType(ShapeDType())));

    ffi::Map<tvm::Var, PrimExpr> var_map;
    for (const auto& [expr, slot] : slot_map_) {
      if (auto var = expr.as<tvm::Var>()) {
        var_map.Set(var.value(),
                    TensorLoad(buffer, {IntImm(tvm::PrimType(ShapeDType()), slot->index)}));
      }
    }
    auto f_substitute =
        [&var_map](const tvm::Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = var_map.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };

    ffi::Array<tvm::Stmt> seq;
    for (PrimExprSlot* slot : to_compute) {
      TVM_FFI_ICHECK(!slot->value_computed);
      slot->value_computed = true;
      PrimExpr value = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(slot->expr, f_substitute)
                           .as_or_throw<PrimExpr>();
      seq.push_back(
          tvm::TensorStore(buffer, {IntImm(tvm::PrimType(ShapeDType()), slot->index)}, value));
    }

    tvm::SeqStmt body(seq);
    ffi::Array<tvm::Var> params{buffer};
    Type ret_type = VoidType();

    // TODO(relax-team): Consider attach the target attribute to
    // the shape_func to indicate that this is a host function
    // This could require us to attach target to the relax function here.
    tirx::Function shape_func(params, body, ret_type);
    shape_func =
        WithAttr(std::move(shape_func), tvm::attr::kScriptNamespace.name, ffi::String("s_tir"));
    if (!shape_func->attrs.GetAttr<tvm::Target>(tvm::attr::kTarget.name).has_value()) {
      // kTarget and kIsHostFunc are mutually exclusive
      shape_func =
          WithAttr<tirx::Function>(std::move(shape_func), tvm::tirx::attr::kIsHostFunc.name, true);
    }
    GlobalVar shape_func_var = builder_->AddFunction(shape_func, "shape_func");
    builder_->Emit(Call(Type::Missing(), Op::Get("relax.call_tir_packed"),
                        {shape_func_var, Tuple({shape_heap_})}),
                   "_");
    return to_compute.size();
  }
  //-------------------------------------------------------
  // Type value match logic
  //
  // CheckMatchCast is the only function needed by
  // other code sections
  //-------------------------------------------------------
  /*!
   * \brief Insert runtime check of the match cast condition(value, ty).
   *
   * \param ty The type to be matched.
   * \param value The input value.
   * \param always_check Whether we insert runtime check even if we can prove
   *        that value's type already satisfies the condition.
   *        This option is necessary for argument checking per our calling convention.
   * \param dynamic_only Whether we only check values with dynamic shapes.
   * \param err_ctx Extra error context to bring more informative error reporting.
   * \param match_todos List of match shape todo items collected when recursively
   *                    visit the match cast.
   */
  void CheckMatchCast(const Type& ty, Expr value, bool always_check, bool dynamic_only,
                      const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) {
    return this->VisitType(ty, value, always_check, dynamic_only, err_ctx, match_todos);
  }

  void VisitType(const Type& ty, Expr value, bool always_check, bool dynamic_only,
                 const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) final {
    // short-cut, if the type already satisfies the
    // constraint during match cast, we can skip matching
    if (!always_check && IsBaseOf(ty, GetType(value))) return;
    return TypeFunctor::VisitType(ty, value, always_check, dynamic_only, err_ctx, match_todos);
  }

  void VisitType_(const AnyTypeObj* op, Expr value, bool always_check, bool dynamic_only,
                  const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) final {}

  void VisitType_(const StringTypeObj* op, Expr value, bool always_check, bool dynamic_only,
                  const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) final {
    if (always_check || !IsBaseOf(StringType(), GetType(value))) {
      builder_->Emit(Call(Type::Missing(), ExternFunc("vm.builtin.check_string_info"),
                          {value, GetErrContext(err_ctx)}, std::nullopt, {void_ty_}),
                     "_");
    }
  }

  void VisitType_(const PrimTypeObj* op, Expr value, bool always_check, bool dynamic_only,
                  const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) final {
    // emit runtime check of shape
    if (always_check || !IsBaseOf(PrimType(op->dtype), GetType(value))) {
      // check_shape_info(value, ndim, err_ctx)
      Call call(Type::Missing(), builtin_check_prim_value_info_,
                {value, DataTypeImm(op->dtype), GetErrContext(err_ctx)}, std::nullopt, {void_ty_});
      builder_->Emit(call, "_");
    }
  }

  void VisitType_(const ShapeTypeObj* op, Expr value, bool always_check, bool dynamic_only,
                  const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) final {
    // emit runtime check of shape
    if (always_check || !IsBaseOf(ShapeType(op->ndim), GetType(value))) {
      // check_shape_info(value, ndim, err_ctx)
      Call call(Type::Missing(), builtin_check_shape_info_,
                {value, IntImm::Int64(op->ndim), GetErrContext(err_ctx)}, std::nullopt, {void_ty_});
      builder_->Emit(call, "_");
    }
    if (op->values.has_value()) {
      match_todos->push_back(MatchShapeTodoItem{value, op->values.value(), err_ctx});
    }
  }

  void VisitType_(const TensorTypeObj* op, Expr value, bool always_check, bool dynamic_only,
                  const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) final {
    // emit runtime check of shape
    auto* shape_expr = op->shape.as<ShapeExprObj>();
    if (dynamic_only && std::all_of(shape_expr->values.begin(), shape_expr->values.end(),
                                    [](const PrimExpr& e) { return e->IsInstance<IntImmObj>(); })) {
      // if we only check dynamic shapes, and the shape is static, we can skip.
      return;
    }
    if (always_check || !IsBaseOf(TensorType(op->dtype, op->ndim), GetType(value))) {
      // check_tensor_info(value, ndim, dtype, err_ctx)
      Expr dtype_arg = op->IsUnknownDtype() ? Expr(Call(Type::Missing(), null_value_op_, {}))
                                            : Expr(DataTypeImm(op->dtype.value()->dtype));
      Call call(Type::Missing(), builtin_check_tensor_info_,
                {value, IntImm::Int64(op->ndim), dtype_arg, GetErrContext(err_ctx)}, std::nullopt,
                {void_ty_});
      builder_->Emit(call, "_");
    }

    if (shape_expr != nullptr) {
      match_todos->push_back(MatchShapeTodoItem{value, shape_expr->values, err_ctx});
    } else if (op->shape.as<VarObj>()) {
      // NOTE: This part of the logic is left empty for future support as it is less common.
      // Future implementors: we can emit a binding here and assert here.
      TVM_FFI_THROW(InternalError)
          << "Cannot handle Tensor shape pattern where a var appears multiple times";
    } else {
      TVM_FFI_ICHECK(!op->shape.has_value()) << "Can only handle tensor shape pattern var";
    }
  }

  // Internal helper function to make tuple get item.
  // This function will try to simplify constant tuples
  // the return value **always** have type.
  Expr MakeTupleGetItem(Expr value, int64_t index) {
    if (auto* tuple_expr = value.as<TupleObj>()) {
      return tuple_expr->fields[index];
    } else if (GetTypeAs<TupleTypeObj>(value)) {
      // value is tuple type, it is OK to run tuple get item.
      return TupleGetItem(value, index);
    } else {
      // call runtime tuple get item, and return a object.
      Call call(Type::Missing(), builtin_tuple_getitem_, {value, IntImm::Int64(index)},
                std::nullopt, {object_ty_});
      UpdateType(call, ObjectType());
      return call;
    }
  }

  void VisitType_(const TupleTypeObj* op, Expr value, bool always_check, bool dynamic_only,
                  const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) final {
    auto* value_tinfo = GetTypeAs<TupleTypeObj>(value);
    if (value_tinfo) {
      TVM_FFI_CHECK_EQ(value_tinfo->fields.size(), op->fields.size(), TypeError)
          << err_ctx << " during match-cast we find tuple size mismatch";
    }
    if (always_check || !value_tinfo) {
      // check_tuple_info(value, tuple_size)
      Call call(
          Type::Missing(), builtin_check_tuple_info_,
          {value, IntImm::Int64(static_cast<int64_t>(op->fields.size())), GetErrContext(err_ctx)},
          std::nullopt, {void_ty_});
      builder_->Emit(call, "_");
    }
    // recursively visit each sub-field and run matching
    for (size_t i = 0; i < op->fields.size(); ++i) {
      this->VisitType(op->fields[i], MakeTupleGetItem(value, i), always_check, dynamic_only,
                      err_ctx, match_todos);
    }
  }

  void VisitType_(const FuncTypeObj* op, Expr value, bool always_check, bool dynamic_only,
                  const ffi::String& err_ctx, std::vector<MatchShapeTodoItem>* match_todos) final {
    // we only check function is callable.
    if (!always_check && MatchType<FuncType>(value)) return;
    // check_func_info(value, err_ctx)
    Call call(Type::Missing(), builtin_check_func_info_, {value, GetErrContext(err_ctx)},
              std::nullopt, {void_ty_});
    builder_->Emit(call, "_");
  }

  //-------------------------------------------------------
  // Private member fields.
  //-------------------------------------------------------
  /*! \brief whether to emit error context, can be turned off for testing purposes. */
  bool emit_err_ctx_{true};
  /*! \brief heap ptr to store the PrimExpr slots. */
  Var shape_heap_{ffi::UnsafeInit{}};
  /*! \brief heap size. */
  IntImm heap_size_{ffi::UnsafeInit{}};
  /*! \brief index => slot. */
  std::vector<std::unique_ptr<PrimExprSlot>> slot_vec_;
  /*! \brief Expr => slot. */
  PrimExprSlotMap slot_map_;
  ffi::Optional<GlobalVar> current_gvar_ = std::nullopt;
  /*!
   * \brief List of vars that are being defined but
   * have not go through outstanding shape compute check.
   */
  std::vector<PrimExprSlot*> ready_vars_;
  // call builtin cop
  const Op call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const Op null_value_op_ = Op::Get("relax.null_value");
  // common type
  const Type object_ty_ = AnyType();
  const Type void_ty_ = TupleType(ffi::Array<Type>({}));
  // check function
  const ExternFunc builtin_alloc_shape_heap_{"vm.builtin.alloc_shape_heap"};
  const ExternFunc builtin_match_shape_{"vm.builtin.match_shape"};
  const ExternFunc builtin_make_shape_{"vm.builtin.make_shape"};
  const ExternFunc builtin_check_shape_info_{"vm.builtin.check_shape_info"};
  const ExternFunc builtin_match_prim_value_{"vm.builtin.match_prim_value"};
  const ExternFunc builtin_make_prim_value_{"vm.builtin.make_prim_value"};
  const ExternFunc builtin_check_prim_value_info_{"vm.builtin.check_prim_value_info"};
  const ExternFunc builtin_check_tensor_info_{"vm.builtin.check_tensor_info"};
  const ExternFunc builtin_check_tuple_info_{"vm.builtin.check_tuple_info"};
  const ExternFunc builtin_check_func_info_{"vm.builtin.check_func_info"};
  const ExternFunc builtin_tuple_getitem_{"vm.builtin.tuple_getitem"};
};

namespace transform {

Pass VMShapeLower(bool emit_err_ctx) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return VMShapeLowerMutator::Lower(mod, emit_err_ctx);
  };
  return CreateModulePass(pass_func, 0, "VMShapeLower");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.VMShapeLower",
                        [](bool emit_err_ctx) { return VMShapeLower(emit_err_ctx); });
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
