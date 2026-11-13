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
 *
 * \file analysis.cc
 *
 * \brief Analysis functions for Relax.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {

template <typename T>
struct InsertionSet {
  std::unordered_set<T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> set;
  std::vector<T> data;
  void Insert(const T& t) {
    if (set.count(t) == 0) {
      set.insert(t);
      data.push_back(t);
    }
  }
};

class VarVisitor : protected ExprVisitor {
 public:
  ffi::Array<Var> Free(const Expr& expr) {
    this->VisitExpr(expr);
    ffi::Array<Var> ret;
    for (const auto& v : vars_.data) {
      if (bound_vars_.set.count(v) == 0) {
        ret.push_back(v);
      }
    }
    return ret;
  }

  ffi::Array<Var> Collect() {
    ffi::Array<Var> ret;
    for (const auto& v : bound_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  ffi::Array<Var> Bound(const Expr& expr) {
    this->VisitExpr(expr);
    return Collect();
  }

  ffi::Array<Var> All(const Expr& expr) {
    this->VisitExpr(expr);
    ffi::Array<Var> ret;
    for (const auto& v : vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  ffi::Array<GlobalVar> AllGlobalVars(const Expr& expr) {
    this->VisitExpr(expr);
    ffi::Array<GlobalVar> ret;
    for (const auto& v : global_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  void MarkBounded(const Var& v) {
    bound_vars_.Insert(v);
    vars_.Insert(v);
  }

  void MarkTypeDefinitionsBound(const Type& type) {
    for (const Var& var : DefinableTIRVarsInType(type)) {
      MarkBounded(var);
    }
  }

  void VisitExprDepTypeField(const Type&) final {}

  void VisitExpr_(const VarObj* var) final { vars_.Insert(ffi::GetRef<Var>(var)); }

  void VisitExpr_(const FunctionObj* op) final {
    for (const auto& param : op->params) {
      MarkBounded(param);
      MarkTypeDefinitionsBound(GetType(param));
    }
    VisitExpr(op->body);
  }

  void VisitExpr_(const GlobalVarObj* op) final { global_vars_.Insert(ffi::GetRef<GlobalVar>(op)); }

  void VisitExpr_(const CallObj* call_node) final {
    VisitLoc(call_node->loc);
    VisitExpr(call_node->op);

    for (Type ty_arg : call_node->ty_args) {
      VisitExprDepTypeField(ty_arg);
    }

    for (Expr arg : call_node->args) {
      VisitExpr(arg);
    }
  }

  void VisitBinding_(const VarBindingObj* binding) final {
    MarkBounded(binding->var);
    VisitExpr(binding->value);
    VisitVarDef(binding->var);
  }

  void VisitBinding_(const MatchCastObj* binding) final {
    MarkBounded(binding->var);
    MarkTypeDefinitionsBound(binding->ty);
    ExprVisitor::VisitBinding_(binding);
  }

 private:
  InsertionSet<Var> vars_;
  InsertionSet<Var> bound_vars_;
  InsertionSet<GlobalVar> global_vars_;
};

tvm::ffi::Array<Var> FreeVars(const Expr& expr) { return VarVisitor().Free(expr); }

tvm::ffi::Array<Var> BoundVars(const Expr& expr) { return VarVisitor().Bound(expr); }

tvm::ffi::Array<Var> AllVars(const Expr& expr) { return VarVisitor().All(expr); }

tvm::ffi::Array<GlobalVar> AllGlobalVars(const Expr& expr) {
  return VarVisitor().AllGlobalVars(expr);
}

ffi::Optional<Expr> FindImpureCall(const Expr& expr, const ffi::Optional<Expr>& own_name) {
  class ImpureCallChecker : public ExprVisitor {
   public:
    static ffi::Optional<Expr> Check(const Expr& expr, const ffi::Optional<Expr>& own_name) {
      ImpureCallChecker visitor(own_name);
      visitor.VisitExpr(expr);
      return visitor.impure_expr_;
    }

   private:
    explicit ImpureCallChecker(const ffi::Optional<Expr>& own_name) : own_name_(own_name) {}

    void VisitExpr(const Expr& expr) override {
      // Early bail-out if we found an impure expression
      if (!impure_expr_) {
        ExprVisitor::VisitExpr(expr);
      }
    }

    void VisitExpr_(const FunctionObj* func) override {
      // we don't visit inner functions because an impure call in an inner function
      // does *not* mean the outer function contains an impure call
    }

    void VisitExprDepTypeField(const Type&) final {
      // Purity is a property of runtime value computation.  Dependent type
      // expressions may contain TIRX calls, but those calls are not executed
      // as part of the Relax expression being checked.
    }

    void VisitExpr_(const CallObj* call) override {
      // ignore recursive calls if we find one
      bool is_recursive = (own_name_ && own_name_.value().same_as(call->op));
      auto expr = ffi::GetRef<Call>(call);
      if (!is_recursive && IsImpureCall(expr)) {
        impure_expr_ = expr;
      } else {
        ExprVisitor::VisitExpr_(call);
      }
    }

   private:
    const ffi::Optional<Expr>& own_name_;
    ffi::Optional<Expr> impure_expr_ = std::nullopt;
  };

  if (own_name) {
    TVM_FFI_ICHECK(own_name.value().as<VarObj>() || own_name.value().as<GlobalVarObj>())
        << "Must pass a Var or GlobalVar for own_name";
  }

  Expr to_check = expr;
  if (auto func = to_check.as<FunctionObj>()) {
    to_check = func->body;
  }
  return ImpureCallChecker::Check(to_check, own_name);
}

bool ContainsImpureCall(const Expr& expr, const ffi::Optional<Expr>& own_name) {
  return FindImpureCall(expr, own_name).has_value();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.analysis.free_vars", FreeVars)
      .def("relax.analysis.bound_vars", BoundVars)
      .def("relax.analysis.all_vars", AllVars)
      .def("relax.analysis.all_global_vars", AllGlobalVars)
      .def("relax.analysis.contains_impure_call", ContainsImpureCall);
}

}  // namespace relax
}  // namespace tvm
