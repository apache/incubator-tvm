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
 * \file stmt_simplify.cc
 * \brief Statement simplifier based on analyzer
 */

#include "stmt_simplify.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/stmt_functor_with_constraint_context.h>
#include <tvm/s_tir/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace s_tir {
using namespace tirx;
using namespace tvm::prim;

namespace {

class StmtSimplifier : public StmtExprMutatorWithConstraintContext {
 public:
  using StmtExprMutatorWithConstraintContext::Mutate;
  using StmtExprMutatorWithConstraintContext::Mutate_;
  explicit StmtSimplifier(const sym::Analyzer& analyzer)
      : StmtExprMutatorWithConstraintContext(analyzer->GetConstraintContext()) {}

  Function Run(Function func);

  UnchangedOr<Stmt> Mutate_(const EvaluateObj* op, InplaceMode inplace_mode) final {
    Stmt stmt = Parent::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    if (const auto* evaluate = stmt.as<EvaluateObj>()) {
      if (const auto* call = evaluate->value.as<CallObj>();
          call && call->op.same_as(s_tir::async_wait()) &&
          GetSyncedAnalyzer()->CanProve(call->args[1].as_or_throw<PrimExpr>() < 0)) {
        // Unrolling may make a wait count negative; such a wait is a no-op.
        return Evaluate(0);
      }
    }
    return stmt;
  }

 protected:
  using Parent = StmtExprMutatorWithConstraintContext;

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) final;

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final;

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) override;

  UnchangedOr<Stmt> Mutate_(const IfObj* op, InplaceMode inplace_mode) override;

  // eliminate useless stores
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) override;

 private:
  bool ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs);

  /* \brief Internal utility for checking conditionals
   *
   * Substitutes any known Bind values and then simplifies with the analyzer.
   */
  ffi::Optional<bool> ProveCondition(PrimExpr condition);

  // Pure Bind values kept for substitution into assert conditions.
  // Grows monotonically under SSA — no scope-based cleanup required.
  ffi::Map<Var, PrimExpr> non_inlined_bindings_;
};

Function StmtSimplifier::Run(Function func) {
  AddParamConstraints(func);
  auto* n = func.CopyOnWrite();
  // Shared string literals no longer enter the primitive analyzer.  Inline their
  // SSA bindings so existing literal arguments remain constants during lowering.
  ffi::Map<Var, StringImm> string_bindings;
  n->body = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                std::move(n->body),
                [&](const Bind& bind) -> ffi::UnchangedOr<ffi::Any> {
                  if (!bind.defined()) return ffi::Unchanged();
                  if (auto value = bind->value.as<StringImm>()) {
                    string_bindings.Set(bind->var, *value);
                    return ffi::Any(Evaluate(0));
                  }
                  return ffi::Unchanged();
                },
                [&](const Var& var) -> ffi::UnchangedOr<ffi::Any> {
                  if (auto value = string_bindings.Get(var)) return ffi::Any(*value);
                  return ffi::Unchanged();
                })
                .as_or_throw<ffi::Optional<SeqStmt>>();
  n->body = Mutate(n->body, InplaceMode::kAllow)
                .as_or_throw<UnchangedOr<ffi::Optional<SeqStmt>>>()
                .ValueOrUnchanged(n->body);
  return func;
}

UnchangedOr<ffi::Any> StmtSimplifier::Mutate(ffi::AnyView input, InplaceMode inplace_mode) {
  if (input.as<TensorType>()) {
    return ffi::Unchanged();
  }
  if (def_region_kind() != kTVMFFIDefRegionKindNone && input.as<Var>()) {
    return Parent::Mutate(input, inplace_mode);
  }
  if (auto call = input.as<Call>(); call && call.value()->op.same_as(assume_always_op())) {
    return Parent::Mutate(input, inplace_mode);
  }
  if (auto expr = input.as<PrimExpr>()) {
    PrimExpr simplified = GetSyncedAnalyzer()->Simplify(*expr);
    if (simplified.same_as(*expr)) return ffi::Unchanged();
    return simplified;
  }
  return Parent::Mutate(input, inplace_mode);
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const ForObj* op, InplaceMode inplace_mode) {
  return Parent::Mutate_(op, inplace_mode);
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const BindObj* op, InplaceMode inplace_mode) {
  if (const auto* call = op->value.as<CallObj>()) {
    // Preserve tensor type metadata; only declaration data is simplified.
    if (call->op.same_as(tirx::alloc_tensor_op())) return ffi::Unchanged();
    if (call->op.same_as(tirx::decl_tensor_op())) {
      // The Call and its arguments may be shared even when the Bind is writable.
      auto data = this->Mutate(call->args[0], InplaceMode::kDisallow);
      if (data.UnchangedOrSameAs(call->args[0])) return ffi::Unchanged();
      auto value = ffi::make_object<CallObj>(*call);
      value->args.Set(0, std::move(data).ValueUnchecked());
      if (inplace_mode == InplaceMode::kAllow) {
        const_cast<BindObj*>(op)->value = Expr(std::move(value));
        return ffi::Unchanged();
      }
      auto copy = ffi::make_object<BindObj>(*op);
      copy->value = Expr(std::move(value));
      return Stmt(std::move(copy));
    }
  }
  auto prim_value = op->value.as<PrimExpr>();
  if (!prim_value) {
    return Parent::Mutate_(op, inplace_mode);
  }
  PrimExpr value =
      this->Mutate(prim_value.value(), inplace_mode).ValueOrUnchanged(prim_value.value());
  // Keep the statement: the sequence collector records its transformed
  // binding for following siblings after this callback returns.
  if (SideEffect(value) <= CallEffectKind::kPure) {
    // Record the binding so we can substitute it into assert conditions
    // (see Mutate_(const AssertStmtObj*, InplaceMode)).  Under SSA each var is
    // bound exactly once, so the map grows monotonically without key
    // conflicts.  No scope-based cleanup is needed because vars bound
    // in inner scopes are only referenced within those scopes; stale
    // entries are harmless and never consulted again.
    non_inlined_bindings_.Set(op->var, value);
  }

  if (value.same_as(op->value)) {
    return ffi::Unchanged();
  } else {
    if (inplace_mode == InplaceMode::kAllow) {
      auto* n = const_cast<BindObj*>(op);
      n->value = std::move(value);
      return ffi::Unchanged();
    }
    auto n = ffi::make_object<BindObj>(*op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const IfObj* op, InplaceMode inplace_mode) {
  if (ffi::Optional<bool> cond = ProveCondition(op->condition)) {
    if (cond.value()) {
      return this->Mutate(op->then_case, inplace_mode).ValueOrUnchanged(op->then_case);
    } else if (op->else_case) {
      return this->Mutate(op->else_case.value(), inplace_mode)
          .ValueOrUnchanged(op->else_case.value());
    } else {
      return Evaluate(0);
    }
  } else {
    return Parent::Mutate_(op, inplace_mode);
  }
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) {
  TensorStore store = Parent::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                          .as_or_throw<TensorStore>();
  if (const TensorLoadObj* load = store->value.as<TensorLoadObj>()) {
    TensorVar tensor = load->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor.same_as(store->dest.as_or_throw<TensorVar>()) &&
        ArrayDeepEqual(load->indices, store->indices)) {
      return Evaluate(0);
    }
  }
  return store;
}

bool StmtSimplifier::ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs,
                                    const ffi::Array<PrimExpr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); i++) {
    if (!prim::ExprDeepEqual()(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

ffi::Optional<bool> StmtSimplifier::ProveCondition(PrimExpr condition) {
  auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = non_inlined_bindings_.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  condition = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(condition, f_substitute)
                  .as_or_throw<PrimExpr>();
  condition = GetSyncedAnalyzer()->Simplify(condition);
  if (const auto* as_int = condition.as<IntImmObj>()) {
    return as_int->value != 0;
  } else {
    return std::nullopt;
  }
}

}  // namespace

Function StmtSimplify(Function func, const sym::Analyzer& analyzer) {
  return ffi::make_object<StmtSimplifier>(analyzer)->Run(std::move(func));
}

namespace transform {

Pass StmtSimplify() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    sym::Analyzer analyzer;
    return s_tir::StmtSimplify(std::move(f), analyzer);
  };
  return tirx::transform::CreateFunctionPass(pass_func, 0, "s_tir.StmtSimplify");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.StmtSimplify", StmtSimplify);
}

}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
