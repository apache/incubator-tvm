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
 * \file remove_store_undef.cc
 * \brief Remove stores of tirx::undef_op
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

struct UndefInfo {
  std::unordered_set<const TensorStoreObj*> undef_stores;
  std::unordered_set<const VarObj*> undef_bind_vars;
};

class StoreUndefLocator : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  static UndefInfo Locate(Stmt stmt) {
    auto locator = ffi::make_object<StoreUndefLocator>();
    locator->Visit(std::move(stmt));
    return {locator->undef_stores_, locator->var_bindings_with_undef_};
  }

  StoreUndefLocator() = default;

 private:
  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreObj* op) final {
    // Check the value for undef.
    bool stash_undef = false;
    std::swap(has_undef_, stash_undef);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->value));
    std::swap(has_undef_, stash_undef);
    if (stash_undef) {
      auto value = op->value.as<PrimExpr>();
      TVM_FFI_ICHECK(value && SideEffect(value.value()) <= CallEffectKind::kReadState)
          << "Error: T.undef() used in TensorStore expressions "
          << "must not have other side effects";
      undef_stores_.insert(op);
    }

    // Check indices for undef.  Undef in tensor indices is always an
    // error (there is no valid lowering).  With flat Bind, we must
    // check indices eagerly because the Bind node is a sibling rather
    // than an ancestor and may be removed before post-validation.
    bool idx_undef = false;
    std::swap(has_undef_, idx_undef);
    for (const auto& idx : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(idx));
    }
    std::swap(has_undef_, idx_undef);
    TVM_FFI_ICHECK(!idx_undef) << "Error: T.undef() may not be used in tensor indices";
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadObj* op) final {
    // Check indices for undef.  Undef in tensor indices is always an error.
    bool idx_undef = false;
    std::swap(has_undef_, idx_undef);
    for (const auto& idx : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(idx));
    }
    std::swap(has_undef_, idx_undef);
    TVM_FFI_ICHECK(!idx_undef) << "Error: T.undef() may not be used in tensor indices";
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindObj* op) final {
    bool stash_undef = false;
    std::swap(has_undef_, stash_undef);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(op->value));
    std::swap(has_undef_, stash_undef);
    if (stash_undef) {
      auto value = op->value.as<PrimExpr>();
      TVM_FFI_ICHECK(value && SideEffect(value.value()) <= CallEffectKind::kReadState)
          << "Error: T.undef() used in Let expressions "
          << "must not have other side effects";
      var_bindings_with_undef_.insert(op->var.get());
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarObj* op) final {
    if (var_bindings_with_undef_.count(op)) {
      has_undef_ = true;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    if (op->op.same_as(tirx::undef_op())) {
      has_undef_ = true;
    }
    return StmtExprVisitor::Visit_(op);
  }

  bool has_undef_{false};

  std::unordered_set<const VarObj*> var_bindings_with_undef_;
  std::unordered_set<const TensorStoreObj*> undef_stores_;
};

// Remove TensorStores whose value depends on T.undef, and also
// remove Bind nodes whose value contains undef.  Undef in tensor
// indices is already caught eagerly in the locator phase.
class StoreUndefRemover : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Apply(Stmt stmt) {
    auto info = StoreUndefLocator::Locate(stmt);
    auto mutator = ffi::make_object<StoreUndefRemover>(info);
    return mutator->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(std::move(stmt));
  }

 private:
  using Parent = StmtExprMutator;

 public:
  explicit StoreUndefRemover(const UndefInfo& info)
      : stores_to_remove_(info.undef_stores), bind_vars_to_remove_(info.undef_bind_vars) {}

 private:
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) final {
    if (stores_to_remove_.count(op)) {
      return Evaluate(0);
    } else {
      return Parent::Mutate_(op, inplace_mode);
    }
  }

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) final {
    if (bind_vars_to_remove_.count(op->var.get())) {
      return Evaluate(0);
    } else {
      return Parent::Mutate_(op, inplace_mode);
    }
  }

  const std::unordered_set<const TensorStoreObj*>& stores_to_remove_;
  const std::unordered_set<const VarObj*>& bind_vars_to_remove_;
};

// Check that no tirx::undef_op() remains in the IR.
class ContainsUndefChecker : public StmtExprVisitor {
 public:
  using StmtExprVisitor::Visit_;
  static bool Check(const Stmt& stmt) {
    auto checker = ffi::make_object<ContainsUndefChecker>();
    checker->Visit(stmt);
    return checker->contains_undef;
  }

 private:
  ffi::Optional<VisitInterrupt> Visit_(const CallObj* op) final {
    if (op->op.same_as(tirx::undef_op())) {
      contains_undef = true;
    }
    return StmtExprVisitor::Visit_(op);
  }

  bool contains_undef{false};
};

namespace transform {
Pass RemoveStoreUndefInternal() {
  auto pass_func = [](Function f, IRModule m, tvm::transform::PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto* n = f.CopyOnWrite();
    n->body = StoreUndefRemover::Apply(std::move(n->body).value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.RemoveStoreUndefInternal");
}

Pass ValidateAllUndefRemoved() {
  auto pass_func = [](Function f, IRModule m, tvm::transform::PassContext ctx) {
    if (!f->body.has_value()) return f;
    bool contains_undef = ContainsUndefChecker::Check(f->body.value());
    TVM_FFI_ICHECK(!contains_undef)
        << "Expected removal of TensorStore containing tirx::undef_op() "
        << "to remove all instances of tirx::undef_op().  "
        << "Instead, result was"
        << "\n"
        << f;
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.ValidateAllUndefRemoved");
}

Pass RemoveStoreUndef() {
  return tvm::transform::Sequential(
      {RemoveStoreUndefInternal(), tirx::transform::RemoveNoOp(), ValidateAllUndefRemoved()},
      "s_tir.RemoveStoreUndef");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.RemoveStoreUndef", RemoveStoreUndef);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
