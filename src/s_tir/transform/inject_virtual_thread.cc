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
 * \file inject_virtual_thread.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>

#include <unordered_set>

#include "ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

namespace {

ffi::Optional<Var> GetBufferDataVar(const ffi::Any& data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto* call = data.as<CallObj>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) && call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

}  // namespace

// If expression is touched by var.
class ExprTouched final : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  explicit ExprTouched(const std::unordered_set<const VarObj*>& touched) : touched_var_(touched) {}

  void Reset(bool check_write) {
    expr_touched_ = false;
    used_vars_.clear();
    write_vars_.clear();
    check_write_ = check_write;
  }

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView n) final {
    // early stopping
    if (expr_touched_ && !check_write_) return std::nullopt;
    return StmtExprVisitor::Visit(n);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    HandleUseVar(op->source.as_or_throw<tvm::tirx::TensorVar>().get());
    return StmtExprVisitor::Visit_(op);
  }
  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
    HandleUseVar(op);
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    if (op->op.same_as(tirx::masked_load_op()) || op->op.same_as(tirx::masked_store_op())) {
      bool is_load = op->op.same_as(tirx::masked_load_op());
      const VarObj* tensor = op->args[0].as_or_throw<Var>().get();
      if (is_load) {
        HandleUseVar(tensor);
      } else {
        HandleWriteVar(tensor);
      }
      for (size_t i = 1; i < op->args.size(); ++i) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->args[i]));
      }
    } else if (op->op.same_as(tirx::tensor_data_ptr_op())) {
      const VarObj* tensor = op->args[0].as_or_throw<Var>().get();
      HandleUseVar(tensor);
      // An escaping address carries no direction information.  Conservatively
      // connect it to the other operands of an opaque statement so virtual
      // threads never share storage that the statement may write.
      if (check_write_) HandleWriteVar(tensor);
    } else if (op->op.same_as(tirx::address_of_op()) && op->args[0].as<TensorLoadObj>()) {
      const auto* load = op->args[0].as<TensorLoadObj>();
      const VarObj* tensor = load->source.as_or_throw<TensorVar>().get();
      HandleUseVar(tensor);
      if (check_write_) HandleWriteVar(tensor);
      for (const PrimExpr& index : load->indices) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
      }
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    }
    return std::nullopt;
  }
  void HandleUseVar(const VarObj* var) {
    auto it = touched_var_.find(var);
    if (it != touched_var_.end()) {
      expr_touched_ = true;
    }
    // rember the used vars
    // in case the var get touched later in a loop.
    if (!expr_touched_) {
      used_vars_.push_back(var);
    }
  }
  void HandleWriteVar(const VarObj* var) { write_vars_.push_back(var); }
  // the fields.
  bool expr_touched_{false};
  std::vector<const VarObj*> used_vars_;
  std::vector<const VarObj*> write_vars_;
  const std::unordered_set<const VarObj*>& touched_var_;
  bool check_write_{false};
};

// Analyze if the buffers are invariant to value of var
class VarTouchedAnalysis : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
    if (value.as<ExprObj>()) return std::nullopt;
    return StmtExprVisitor::Visit(value);
  }
  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return DispatchAllocTensor(op, call);
    }
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::decl_tensor_op()))
      return StmtExprVisitor::Visit_(op);
    expr_touched_->Reset(false);
    expr_touched_->Visit(op->value);
    Record(op->var.get(), *expr_touched_);
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    expr_touched_->Reset(false);
    expr_touched_->Visit(op->value);
    for (const auto& index : op->indices) {
      expr_touched_->Visit(index);
    }
    Record(op->dest.as_or_throw<TensorVar>().get(), *expr_touched_);
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
    expr_touched_->Reset(false);
    expr_touched_->Visit(op->begin);
    expr_touched_->Visit(op->end);
    expr_touched_->Visit(op->step);
    Record(op->var.get(), *expr_touched_);
    return this->Visit(op->body);
  }
  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtObj* op) final {
    if (!op->op.same_as(tirx::launch_thread_op())) has_opaque_region_ = true;
    expr_touched_->Reset(false);
    for (const Expr& arg : op->args) expr_touched_->Visit(arg);
    for (const Var& var : op->body_params) Record(var.get(), *expr_touched_);
    return Visit(op->body);
  }

  bool has_opaque_region_{false};

  // external function call
  ffi::Optional<VisitInterrupt> Visit_(const EvaluateObj* op) final {
    expr_touched_->Reset(true);
    expr_touched_->Visit(op->value);
    for (const VarObj* var : expr_touched_->write_vars_) {
      Record(var, *expr_touched_);
    }
    return std::nullopt;
  }
  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindObj* op, const CallObj* call) {
    expr_touched_->Reset(false);
    tvm::Tuple shape = tvm::Tuple(call->ty_args[0].as_or_throw<tirx::TensorType>()->shape);
    for (const Expr& extent : shape->fields) {
      expr_touched_->Visit(extent);
    }
    Record(op->var.as_or_throw<TensorVar>().get(), *expr_touched_);
    return StmtExprVisitor::Visit_(op);
  }
  void Record(const VarObj* var, const ExprTouched& tc) {
    if (touched_var_.count(var)) return;
    if (tc.expr_touched_) {
      touched_var_.insert(var);
    } else {
      for (const VarObj* r : tc.used_vars_) {
        if (r != var) {
          affect_[r].push_back(var);
        }
      }
    }
  }

  std::unordered_set<const VarObj*> TouchedVar(const Stmt& stmt, const VarObj* var) {
    touched_var_.insert(var);
    this->Visit(stmt);
    // do a DFS to push affect around dependency.
    std::vector<const VarObj*> pending(touched_var_.begin(), touched_var_.end());
    while (!pending.empty()) {
      const VarObj* v = pending.back();
      pending.pop_back();
      for (const VarObj* r : affect_[v]) {
        if (!touched_var_.count(r)) {
          touched_var_.insert(r);
          pending.push_back(r);
        }
      }
    }
    return std::move(touched_var_);
  }

 private:
  // Whether variable is touched by the thread variable.
  std::unordered_set<const VarObj*> touched_var_;
  ffi::ObjectPtr<ExprTouched> expr_touched_ = ffi::make_object<ExprTouched>(touched_var_);
  // x -> all the buffers x read from
  std::unordered_map<const VarObj*, std::vector<const VarObj*>> affect_;
};

// Inject virtual thread loop
// rewrite the buffer access pattern when necessary.
class VTInjector : public s_tir::StmtExprMutatorWithConstraintContext {
 public:
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate;
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate_;

  // constructor
  VTInjector(sym::AnalyzerObj* analyzer, Var var, int num_threads,
             const std::unordered_set<const VarObj*>& touched_var, bool allow_share)
      : StmtExprMutatorWithConstraintContext(analyzer->GetConstraintContext()),
        var_(var),
        num_threads_(num_threads),
        touched_var_(touched_var),
        allow_share_(allow_share) {}
  // Inject VTLoop when needed.
  UnchangedOr<ffi::Any> Mutate(ffi::AnyView value,
                               InplaceMode inplace_mode = InplaceMode::kDisallow) final {
    if (!value.as<StmtObj>())
      return StmtExprMutatorWithConstraintContext::Mutate(value, inplace_mode);
    TVM_FFI_ICHECK(!visit_touched_var_);
    auto result = StmtExprMutatorWithConstraintContext::Mutate(value, inplace_mode);
    if (visit_touched_var_ || trigger_base_inject_) {
      if (!vt_loop_injected_) {
        Stmt stmt = std::move(result).ValueOrUnchanged(value).as_or_throw<Stmt>();
        return ffi::Any(InjectVTLoop(stmt, false));
      }
      visit_touched_var_ = false;
      trigger_base_inject_ = false;
    }
    return result;
  }
  // Variable
  UnchangedOr<Expr> Mutate_(const TensorRegionObj* op, InplaceMode inplace_mode) final {
    if (!op->source.as<TensorVar>()) {
      return StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode);
    }
    auto region = Mutate(op->region).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
    if (region.UnchangedOrSameAs(op->region)) return ffi::Unchanged();
    TensorRegion node = ffi::GetRef<TensorRegion>(op);
    node.CopyOnWrite()->region = std::move(region).ValueUnchecked();
    return node;
  }

  UnchangedOr<Expr> Mutate_(const VarObj* op, InplaceMode inplace_mode) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      TVM_FFI_ICHECK(!alloc_remap_.count(op))
          << "TensorVar address may get rewritten in virtual thread";
      if (touched_var_.count(op)) {
        visit_touched_var_ = true;
      }
    }
    return StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode);
  }
  PrimExpr RewriteIndex(PrimExpr index, PrimExpr alloc_extent) {
    return GetSyncedAnalyzer()->Simplify(index + var_.as_or_throw<PrimExpr>() * alloc_extent);
  }
  // Expression.
  UnchangedOr<Expr> Mutate_(const CallObj* op, InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::masked_load_op()) || op->op.same_as(tirx::masked_store_op())) {
      bool is_load = op->op.same_as(tirx::masked_load_op());
      TensorVar tensor = op->args[0].as_or_throw<TensorVar>();
      ffi::Optional<PrimExpr> value;
      if (!is_load)
        value = Mutate(op->args[1]).ValueOrUnchanged(op->args[1]).as_or_throw<PrimExpr>();
      ffi::Array<PrimExpr> indices;
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        indices.push_back(
            Mutate(op->args[i]).ValueOrUnchanged(op->args[i]).as_or_throw<PrimExpr>());
      }
      PrimExpr predicate = Mutate(op->args[op->args.size() - 1])
                               .ValueOrUnchanged(op->args[op->args.size() - 1])
                               .as_or_throw<PrimExpr>();
      if (is_load) {
        TensorLoad access = VisitTensorAccess(TensorLoad(tensor, indices, op->loc));
        ffi::Array<Expr> args{access->source.as_or_throw<tvm::tirx::TensorVar>()};
        for (const PrimExpr& index : access->indices) args.push_back(index);
        args.push_back(predicate);
        return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
      }
      TensorStore access = VisitTensorAccess(TensorStore(tensor, indices, value.value(), op->loc));
      ffi::Array<Expr> args{access->dest.as_or_throw<TensorVar>(), access->value};
      for (const PrimExpr& index : access->indices) args.push_back(index);
      args.push_back(predicate);
      return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
    } else if (op->op.same_as(tirx::tensor_data_ptr_op())) {
      auto tensor_var = GetBufferDataVar(ffi::GetRef<Call>(op)).value();
      auto it = alloc_remap_.find(tensor_var.get());
      if (it == alloc_remap_.end()) {
        return StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode);
      }
      visit_touched_var_ = true;
      TensorVar tensor = tensor_var.as_or_throw<TensorVar>();
      PrimType dtype = tensor.dtype();
      int bytes = (dtype.bits() * dtype.lanes() + 7) / 8;
      return Call(op->ty, tirx::ptr_byte_offset_op(),
                  {GetRemappedTensor(tensor, it->second).data(),
                   RewriteIndex(PrimExpr(0), it->second) * bytes},
                  {}, {}, op->loc);
    } else {
      return StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode);
    }
  }
  UnchangedOr<Stmt> Mutate_(const EvaluateObj* op, InplaceMode inplace_mode) final {
    trigger_base_inject_ = !allow_share_;
    return StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode);
  }
  // TensorLoad
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj* op, InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node = ffi::GetRef<TensorLoad>(op);
    if (!indices.UnchangedOrSameAs(op->indices)) {
      node.CopyOnWrite()->indices = std::move(indices).ValueUnchecked();
    }
    return VisitTensorAccess(std::move(node));
  }
  // TensorStore
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    auto value = Mutate(op->value);
    auto indices = Mutate(op->indices).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore node = ffi::GetRef<TensorStore>(op);
    if (!value.UnchangedOrSameAs(op->value) || !indices.UnchangedOrSameAs(op->indices)) {
      auto* n = node.CopyOnWrite();
      n->value = std::move(value).ValueOrUnchanged(op->value);
      n->indices = std::move(indices).ValueOrUnchanged(op->indices);
    }
    trigger_base_inject_ = !allow_share_;
    return VisitTensorAccess(std::move(node));
  }

  TensorStore VisitTensorAccess(TensorStore node) {
    TensorVar tensor = node->dest.as_or_throw<TensorVar>();
    if (touched_var_.count(tensor.get())) {
      visit_touched_var_ = true;
    }

    auto it = alloc_remap_.find(tensor.get());
    if (it != alloc_remap_.end()) {
      TVM_FFI_ICHECK_EQ(node->indices.size(), 1)
          << "InjectVirtualThread expects rewritten allocations to be flat memory.";
      auto writer = node.CopyOnWrite();
      writer->dest = GetRemappedTensor(tensor, it->second);
      writer->indices = {RewriteIndex(node->indices[0], it->second)};
    }

    return node;
  }

  TensorLoad VisitTensorAccess(TensorLoad node) {
    TensorVar tensor = node->source.as_or_throw<tvm::tirx::TensorVar>();
    if (touched_var_.count(tensor.get())) {
      visit_touched_var_ = true;
    }
    auto it = alloc_remap_.find(tensor.get());
    if (it == alloc_remap_.end()) return node;
    TVM_FFI_ICHECK_EQ(node->indices.size(), 1)
        << "InjectVirtualThread expects rewritten allocations to be flat memory.";
    auto* writer = node.CopyOnWrite();
    writer->source = GetRemappedTensor(tensor, it->second);
    writer->indices = {RewriteIndex(node->indices[0], it->second)};
    return node;
  }

  TensorVar GetRemappedTensor(TensorVar tensor, PrimExpr alloc_extent) {
    if (auto replacement = VarRemapGetExpected(tensor).value().as<TensorVar>())
      return replacement.value();
    TensorVar original = tensor;

    TVM_FFI_ICHECK_EQ(tensor.shape().size(), 1)
        << "Expected tensors being rewritten to already be flattened.";
    auto writer = ffi::make_object<TensorTypeObj>(*tensor.ty().get());
    writer->shape = {tensor.shape()[0] * alloc_extent};
    tensor = TensorVar(tensor->name, TensorType(std::move(writer)), tensor->loc);

    VarRemapSetExpected(original, tensor).value();
    return tensor;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final {
    // A launch extent is evaluated before entering its body. If it depends on
    // the virtual index, put the virtual loop outside the entire launch.
    auto args =
        this->Mutate(op->args, InplaceMode::kDisallow).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(ffi::GetRef<Stmt>(op), true);
    }
    visit_touched_var_ = false;
    auto body = this->Mutate(op->body, inplace_mode);
    if (args.UnchangedOrSameAs(op->args) && body.UnchangedOrSameAs(op->body)) {
      return ffi::Unchanged();
    }
    return RegionStmt(op->op, std::move(args).ValueOrUnchanged(op->args), op->body_params,
                      op->attrs, std::move(body).ValueOrUnchanged(op->body), op->result_vars,
                      op->loc);
  }

  // Bind
  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      return Mutate_AllocTensor(op, call, inplace_mode);
    }
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::decl_tensor_op()))
      return StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode);
    auto value_result = this->Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    Expr value = std::move(value_result).ValueOrUnchanged(op->value);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(ffi::GetRef<Stmt>(op), true);
    }
    visit_touched_var_ = false;
    if (value_unchanged) {
      return ffi::Unchanged();
    } else {
      return Bind(op->var, value);
    }
  }
  // For
  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(IsZero(op->begin));
    auto extent_result = this->Mutate(op->end, inplace_mode);
    bool extent_unchanged = extent_result.UnchangedOrSameAs(op->end);
    PrimExpr extent = std::move(extent_result).ValueOrUnchanged(op->end);
    if (visit_touched_var_ && !vt_loop_injected_) {
      Stmt stmt = InjectVTLoop(ffi::GetRef<Stmt>(op), true);
      ++max_loop_depth_;
      return stmt;
    }
    visit_touched_var_ = false;
    auto body_result = this->Mutate(op->body, inplace_mode);
    bool body_unchanged = body_result.UnchangedOrSameAs(op->body);
    Stmt body = std::move(body_result).ValueOrUnchanged(op->body);
    ++max_loop_depth_;
    if (extent_unchanged && body_unchanged) {
      return ffi::Unchanged();
    } else {
      if (inplace_mode == InplaceMode::kAllow) {
        auto* writable = const_cast<ForObj*>(op);
        writable->end = std::move(extent);
        writable->body = std::move(body);
        return ffi::Unchanged();
      } else {
        auto copy = ffi::make_object<ForObj>(*op);
        copy->end = std::move(extent);
        copy->body = std::move(body);
        return For(std::move(copy));
      }
    }
  }
  // If
  UnchangedOr<Stmt> Mutate_(const IfObj* op, InplaceMode inplace_mode) final {
    auto condition_result = this->Mutate(op->condition, inplace_mode);
    bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
    PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(ffi::GetRef<Stmt>(op), true);
    }
    visit_touched_var_ = false;
    TVM_FFI_ICHECK_EQ(max_loop_depth_, 0);
    auto then_case_result = this->Mutate(op->then_case, inplace_mode);
    bool then_case_unchanged = then_case_result.UnchangedOrSameAs(op->then_case);
    Stmt then_case = std::move(then_case_result).ValueOrUnchanged(op->then_case);
    ffi::Optional<SeqStmt> else_case = std::nullopt;
    if (op->else_case) {
      int temp = max_loop_depth_;
      max_loop_depth_ = 0;
      else_case =
          this->Mutate(op->else_case.value(), inplace_mode).ValueOrUnchanged(op->else_case.value());
      max_loop_depth_ = std::max(temp, max_loop_depth_);
    }
    if (condition_unchanged && then_case_unchanged && else_case.same_as(op->else_case)) {
      return ffi::Unchanged();
    } else {
      return If(condition, then_case, else_case);
    }
  }

  // While
  UnchangedOr<Stmt> Mutate_(const WhileObj* op, InplaceMode inplace_mode) final {
    // TODO(masahi): What should we do for While nodes?
    TVM_FFI_THROW(InternalError) << "WhileObj in InjectVirtualThread not supported yet";
    TVM_FFI_UNREACHABLE();
  }

  // Seq
  // When a Bind child triggers VT injection, we group the Bind together with
  // all remaining siblings (which may reference the bound variable) and wrap
  // them as a single unit in the VT loop.  This preserves the semantics that
  // With flat Bind (no body), a Bind whose value touches vt_var must be
  // grouped with all remaining siblings and wrapped in a VT loop together.
  // This preserves the scoping that was implicit when Bind carried a body.
  UnchangedOr<Stmt> Mutate_(const SeqStmtObj* op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK_EQ(max_loop_depth_, 0);
    ffi::Array<Stmt> new_seq;
    bool changed = false;
    for (size_t i = 0; i < op->seq.size(); ++i) {
      int temp = max_loop_depth_;
      max_loop_depth_ = 0;
      // For Bind children, pre-check if the value touches vt_var before
      // visiting.  If so, group the Bind with all remaining siblings and
      // wrap the group with InjectVTLoop.
      if (const auto* bind = op->seq[i].as<BindObj>(); bind && !vt_loop_injected_) {
        // Visit just the value expression to probe for vt_var dependency.
        TVM_FFI_ICHECK(!visit_touched_var_);
        // This is a probe: the original Bind is revisited when forming the group.
        this->Mutate(bind->value, InplaceMode::kDisallow);
        if (visit_touched_var_) {
          // Reset flag (InjectVTLoop will handle it).
          visit_touched_var_ = false;
          // Gather the original Bind + all remaining original siblings.
          ffi::Array<Stmt> group;
          for (size_t j = i; j < op->seq.size(); ++j) {
            group.push_back(op->seq[j]);
          }
          Stmt grouped = SeqStmt(group, op->loc);
          // before_mutation=true: InjectVTLoop will re-visit the entire group
          // with vt_loop_injected_=true, properly substituting vt_var.
          Stmt wrapped = InjectVTLoop(grouped, true);
          new_seq.push_back(wrapped);
          max_loop_depth_ = std::max(max_loop_depth_, temp);
          changed = true;
          // All remaining siblings consumed — exit loop.
          break;
        }
        // Value did not touch vt_var.  Reset and visit the Bind normally.
        visit_touched_var_ = false;
      }
      // Non-Bind child or Bind that does not touch vt_var: visit normally.
      auto child_result = this->Mutate(op->seq[i]);
      bool child_unchanged = child_result.UnchangedOrSameAs(op->seq[i]);
      Stmt child = std::move(child_result).ValueOrUnchanged(op->seq[i]).as_or_throw<Stmt>();
      max_loop_depth_ = std::max(max_loop_depth_, temp);
      if (!child_unchanged) changed = true;
      new_seq.push_back(child);
    }
    if (!changed) return ffi::Unchanged();
    return SeqStmt(new_seq, op->loc);
  }
  // Allocate
  // AllocTensor
  UnchangedOr<Stmt> Mutate_AllocTensor(const BindObj* op, const CallObj* call,
                                       InplaceMode inplace_mode) {
    tvm::Tuple shape = tvm::Tuple(call->ty_args[0].as_or_throw<tirx::TensorType>()->shape);
    ffi::Array<PrimExpr> original_shape = shape->fields.as_or_throw<ffi::Array<PrimExpr>>();
    ffi::Array<PrimExpr> new_shape = original_shape.Map([this](const PrimExpr& s) {
      // Keep the retained allocation and its tensor type unchanged.
      return Mutate(s, InplaceMode::kDisallow).ValueOrUnchanged(s);
    });

    ffi::Array<Expr> args = call->args;
    if (!args.empty()) {
      args.Set(0, Mutate(args[0], InplaceMode::kDisallow).ValueOrUnchanged(args[0]));
    }
    if (visit_touched_var_ && !vt_loop_injected_) {
      return InjectVTLoop(ffi::GetRef<Stmt>(op), true);
    }

    visit_touched_var_ = false;

    if (touched_var_.count(op->var.as_or_throw<TensorVar>().get()) || !allow_share_) {
      TVM_FFI_ICHECK_EQ(new_shape.size(), 1)
          << "InjectVirtualThread expects rewritten allocations to be flat memory.";
      PrimExpr stride = new_shape[0];
      new_shape = {stride * num_threads_};
      alloc_remap_.insert_or_assign(op->var.as_or_throw<TensorVar>().get(), stride);
    }

    if (new_shape.same_as(original_shape) && args.same_as(call->args)) {
      return ffi::Unchanged();
    } else {
      auto type = ffi::make_object<TensorTypeObj>(*op->var.as_or_throw<TensorVar>().ty().get());
      type->shape = new_shape;
      TensorVar new_tensor =
          TensorVar(op->var.as_or_throw<TensorVar>()->name, TensorType(std::move(type)),
                    op->var.as_or_throw<TensorVar>()->loc);
      VarRemapSetExpected(op->var.as_or_throw<TensorVar>(), new_tensor).value();

      return Bind(new_tensor,
                  Call(new_tensor.ty(), tirx::alloc_tensor_op(), args, call->attrs,
                       {new_tensor.ty()}, call->loc),
                  op->loc);
    }
  }

  // inject vthread loop
  Stmt InjectVTLoop(Stmt stmt, bool before_mutation) {
    TVM_FFI_ICHECK(!vt_loop_injected_);
    // reset the flags
    visit_touched_var_ = false;
    trigger_base_inject_ = false;
    vt_loop_injected_ = true;
    if (before_mutation) {
      stmt = this->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
    }
    // reset the flags after processing.
    vt_loop_injected_ = false;
    visit_touched_var_ = false;
    // only unroll if number of vthreads are small
    if (max_loop_depth_ == 0 && num_threads_ < 16) {
      // do unrolling if it is inside innermost content.
      ffi::Array<Stmt> seq;
      for (int i = 0; i < num_threads_; ++i) {
        PrimType var_ty = var_->ty.as_or_throw<PrimType>();
        auto f_substitute = [this, i,
                             var_ty](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
          if (var.same_as(var_)) return ffi::Any(IntImm(var_ty, i));
          return ffi::Unchanged();
        };
        seq.push_back(
            ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(stmt, f_substitute).as_or_throw<Stmt>());
      }
      return SeqStmt(seq);
    } else {
      // insert a for loop
      Var idx(var_->name + ".s", var_->ty);
      auto f_substitute = [this,
                           &idx](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
        if (var.same_as(var_)) return ffi::Any(idx.as_or_throw<PrimExpr>());
        return ffi::Unchanged();
      };
      stmt = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(stmt, f_substitute).as_or_throw<Stmt>();
      PrimType idx_dtype = idx->ty.as_or_throw<PrimType>();
      return For(idx.as_or_throw<PrimVar>(), IntImm(idx_dtype, 0),
                 prim::MakeConst(idx_dtype, num_threads_),
                 IntImm(idx.as_or_throw<PrimVar>().ty(), 1), ForKind::kDefault, stmt);
    }
  }

 private:
  // vthread variable
  Var var_;
  // the threads/lanes
  int num_threads_;
  // whether the loop is already injected.
  bool vt_loop_injected_{false};
  // whether current expression get touched.
  bool visit_touched_var_{false};
  // Trigger base stmt
  bool trigger_base_inject_{false};
  // the counter of loops in after mutation.
  int max_loop_depth_{0};
  // The variables that get touched.
  const std::unordered_set<const VarObj*>& touched_var_;
  // Whether allow shareding.
  bool allow_share_;
  /* \brief The allocations that get touched -> extent
   *
   * Maps from the buffer_var of an allocate node to the original
   * extent of the allocation.  Used when rewriting the indices of
   * TensorLoad/TensorStore.
   */
  std::unordered_map<const VarObj*, PrimExpr> alloc_remap_;
  /*! \brief Map of tensors that are modified.
   *
   * Tensors allocated or written to within the virtual thread loop
   * must have one copy per virtual thread.  This is done by enlarging
   * the allocated buffer size, then modifying the indices at which
   * each virtual thread accesses the buffer.
   */
};

class VirtualThreadInjector : public s_tir::StmtExprMutatorWithConstraintContext {
 public:
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate;
  using s_tir::StmtExprMutatorWithConstraintContext::Mutate_;

  using StmtExprMutatorWithConstraintContext::StmtExprMutatorWithConstraintContext;

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj* op, InplaceMode inplace_mode) final {
    Stmt stmt = StmtExprMutatorWithConstraintContext::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    op = stmt.as<RegionStmtObj>();
    if (op && op->op.same_as(tirx::launch_thread_op()) &&
        std::string(op->args[0].as_or_throw<StringImm>()->value).rfind("vthread", 0) == 0) {
      PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
      const auto* extent = op->args[1].as_or_throw<PrimExpr>().as<IntImmObj>();
      TVM_FFI_ICHECK(extent) << "Virtual thread extent must be a constant integer";
      int nthread = extent->value.as<int>().value();
      TVM_FFI_ICHECK_GT(nthread, 0) << "Virtual thread extent must be positive";
      auto vs = ffi::make_object<VarTouchedAnalysis>();
      auto touched = vs->TouchedVar(op->body, var.get());
      if (vs->has_opaque_region_) {
        // Keep unknown operations and their result definitions inside one
        // lexical loop; they have no sharing or distribution semantics.
        return For(var, IntImm(var.ty(), 0), op->args[1].as_or_throw<PrimExpr>(),
                   IntImm(var.ty(), 1), ForKind::kDefault, op->body);
      }
      auto injector = ffi::make_object<VTInjector>(GetSyncedAnalyzer(), var, nthread, touched,
                                                   /*allow_share=*/true);
      return injector->Mutate(op->body).ValueOrUnchanged(op->body);
    } else {
      return stmt;
    }
  }
};

namespace transform {

Pass InjectVirtualThread() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();

    sym::Analyzer analyzer;

    n->body = ffi::make_object<VirtualThreadInjector>(analyzer->GetConstraintContext())
                  ->Mutate(n->body.value(), InplaceMode::kAllow)
                  .ValueOrUnchanged(std::move(n->body).value());
    n->body = s_tir::ConvertSSA(std::move(n->body).value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.InjectVirtualThread");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.InjectVirtualThread", InjectVirtualThread);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
