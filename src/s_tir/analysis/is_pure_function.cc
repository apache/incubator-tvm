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
 * \file is_pure_function.cc
 * \brief Function purity analysis
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

namespace {
class PurityChecker : public StmtExprVisitor {
  friend class tvm::ObjectVisitor;
  using Result = ffi::Optional<VisitInterrupt>;

 public:
  static bool Check(const Function& func, bool assert_on_error) {
    auto visitor = ffi::make_object<PurityChecker>(assert_on_error);
    visitor->Visit(func);
    return visitor->is_pure_;
  }

  explicit PurityChecker(bool assert_on_error)
      : StmtExprVisitor(GlobalVTable()), assert_on_error_(assert_on_error) {}

 private:
  using StmtExprVisitor::Visit_;
  Result Visit(ffi::AnyView value) override {
    bool is_root = !entered_;
    entered_ = true;
    if (auto function = value.as<Function>(); function && !is_root) {
      TVM_FFI_VISIT_THROW(TypeError, function.value())
          << "Unsupported non-primitive TIR expression " << function.value()->GetTypeKey();
    }
    if (auto stmt = value.as<Stmt>(); stmt && !GlobalVTable()->CanDispatch(value)) {
      TVM_FFI_VISIT_THROW(InternalError, stmt.value())
          << "Do not have a default for " << stmt.value()->GetTypeKey();
    }
    if (auto expr = value.as<Expr>(); expr && !GlobalVTable()->CanDispatch(value)) {
      if (value.as<OpaqueExprObj>() || value.as<VarObj>()) return std::nullopt;
      TVM_FFI_VISIT_THROW(TypeError, expr.value())
          << "Unsupported non-primitive TIR expression " << expr.value()->GetTypeKey();
    }
    return StmtExprVisitor::Visit(value);
  }

  Result Visit_(const BindObj* op) final {
    if (const auto* call = op->value.as<CallObj>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      internal_allocations_.insert(op->var);
      allocation_calls_.insert(call);
    }
    this->Visit(op->value);
    return this->Visit(op->var->ty);
  }

  Result Visit_(const TensorStoreObj* op) override {
    this->Visit(op->value);
    this->Visit(op->indices);

    if (!internal_allocations_.count(op->dest.as_or_throw<TensorVar>())) {
      is_pure_ = false;
      if (assert_on_error_) {
        TVM_FFI_THROW(AssertionError)
            << "Pure functions must not write to tensors, "
            << ", but function contains store to " << op->dest.as_or_throw<TensorVar>()
            << op->indices << " of value " << op->value;
      }
    }
    return std::nullopt;
  }

  Result Visit_(const CallObj* call) override {
    if (call->op.as<GlobalVar>() || call->op.as<OpaqueExprObj>()) this->Visit(call->op);
    this->Visit(call->args);
    if (allocation_calls_.count(call)) return std::nullopt;

    static auto op_call_effect = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
    CallEffectKind effect = [&]() {
      if (auto opt = call->op.as<Op>()) {
        return op_call_effect[opt.value()];
      } else {
        return CallEffectKind::kOpaque;
      }
    }();

    if (effect == CallEffectKind::kUpdateState || effect == CallEffectKind::kOpaque) {
      is_pure_ = false;
      if (assert_on_error_) {
        TVM_FFI_THROW(AssertionError)
            << "Pure functions must not contain calls to impure operators, "
            << "but " << ffi::GetRef<Call>(call) << " calls operator " << call->op
            << ", which has side effect " << effect;
      }
    }
    return std::nullopt;
  }

  // Match the analysis' traversal policy without introducing binder state.
  Result Visit_(const FunctionObj* op) {
    for (const auto& param : op->params) {
      if (!param->ty.as<TensorTypeObj>()) this->Visit(param->ty);
    }
    for (const auto& param : op->params) {
      if (param->ty.as<TensorTypeObj>()) this->Visit(param->ty);
    }
    return this->Visit(op->body);
  }

  Result Visit_(const LambdaExprObj* op) {
    this->Visit(op->ty);
    for (const auto& var : op->vars) this->Visit(var->ty);
    return this->Visit(op->body);
  }

  Result Visit_(const RegionStmtObj* op) override {
    this->Visit(op->args);
    this->Visit(op->attrs);
    for (const auto& param : op->body_params) this->Visit(param->ty);
    this->Visit(op->body);
    for (const auto& var : op->result_vars) this->Visit(var->ty);
    return std::nullopt;
  }

  Result Visit_(const ForObj* op) override {
    this->Visit(op->begin);
    this->Visit(op->end);
    this->Visit(op->step);
    this->Visit(op->var->ty);
    return this->Visit(op->body);
  }

  Result Visit_(const prim::LetObj* op) override {
    this->Visit(op->value);
    this->Visit(op->var->ty);
    return this->Visit(op->body);
  }

  Result Visit_(const TensorLoadObj* op) override { return this->Visit(op->indices); }

  Result Visit_(const TensorRegionObj* op) override {
    if (!op->source.as<TensorVar>()) this->Visit(op->source);
    return this->Visit(op->region);
  }

  Result Visit_(const GenericConstObj* op) override { return std::nullopt; }

  Result Visit_(const AssertStmtObj* op) override {
    this->Visit(op->condition);
    this->Visit(op->error_kind);
    return this->Visit(op->message_parts);
  }

  Result Visit_(const prim::RampObj* op) override {
    this->Visit(op->base);
    this->Visit(op->stride);
    return this->Visit(op->lanes);
  }

  Result Visit_(const prim::BroadcastObj* op) override {
    this->Visit(op->value);
    return this->Visit(op->lanes);
  }

  Result Visit_(const SBlockObj* op) override {
    for (const auto& axis : op->iter_vars) {
      this->Visit(axis->dom);
      this->Visit(axis->var.as_or_throw<Var>()->ty);
    }
    for (const auto& tensor : op->alloc_tensors) this->Visit(tensor->ty);
    for (const auto& match : op->match_tensors) {
      this->Visit(match->source);
      this->Visit(match->tensor->ty);
    }
    this->Visit(op->reads);
    this->Visit(op->writes);
    this->Visit(op->init);
    return this->Visit(op->body);
  }

  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      StmtExprVisitor::InitVTable(&table);
      SetDispatch<PurityChecker, FunctionObj>(&table);
      SetDispatch<PurityChecker, LambdaExprObj>(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }

  bool entered_{false};
  bool assert_on_error_{false};
  bool is_pure_{true};
  std::unordered_set<Var> internal_allocations_;
  std::unordered_set<const CallObj*> allocation_calls_;
};
}  // namespace

bool IsPureFunction(const Function& func, bool assert_on_error) {
  return PurityChecker::Check(func, assert_on_error);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.analysis.is_pure_function", IsPureFunction);
}

}  // namespace s_tir
}  // namespace tvm
