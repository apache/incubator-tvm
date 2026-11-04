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
 * \file src/relax/analysis/udchain.cc
 * \brief Implementation of use-def analysis.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../support/ordered_set.h"

namespace tvm {
namespace relax {

class UDChain : relax::ExprVisitor {
 public:
  static VarUsageInfo Collect(const Expr& expr) {
    UDChain visitor;
    visitor.VisitExpr(expr);

    ffi::Array<Var> output(visitor.outputs.begin(), visitor.outputs.end());

    ffi::Map<Var, ffi::Array<Var>> use_def;
    for (const auto& [var, usage] : visitor.usage_map) {
      use_def.Set(var, ffi::Array<Var>(usage.begin(), usage.end()));
    }

    return VarUsageInfo{visitor.bound_values, use_def, output};
  }

 private:
  ffi::Map<Var, Expr> bound_values;
  std::unordered_set<Var> forward_declarations;
  std::unordered_map<Var, support::OrderedSet<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>>
      usage_map;
  support::OrderedSet<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> outputs;

  ffi::Optional<Var> cur_user_;

  void VisitBinding_(const VarBindingObj* binding) override {
    TVM_FFI_ICHECK(!bound_values.count(binding->var))
        << "Variable " << binding->var << " was defined multiple times";
    bound_values.Set(binding->var, binding->value);

    auto cache = cur_user_;
    cur_user_ = binding->var;
    ExprVisitor::VisitBinding_(binding);
    cur_user_ = cache;
  }

  void VisitBinding_(const VarBindingObj* binding, const FunctionObj* func) override {
    // A local Relax function may be recursively defined.  References to
    // `binding->var` that appear within `func` are valid.
    DefineVar(binding->var);
    forward_declarations.insert(binding->var);
    ExprVisitor::VisitBinding_(binding, func);
  }

  void VisitVarDef(const Var& var) override {
    if (forward_declarations.count(var)) {
      forward_declarations.erase(var);
    } else {
      DefineVar(var);
    }
  }
  void VisitExpr_(const VarObj* op) override {
    auto var = ffi::GetRef<Var>(op);

    if (cur_user_) {
      usage_map[var].insert(cur_user_.value());
    } else {
      outputs.insert(var);
    }
  }

  void VisitExpr_(const FunctionObj* op) override {
    cur_user_ = std::nullopt;
    ExprVisitor::VisitExpr_(op);
  }

  void DefineVar(const Var& var) {
    TVM_FFI_ICHECK(!usage_map.count(var))
        << "Variable " << var << " was used before its definition";
    usage_map[var] = {};
  }
};

std::pair<ffi::Map<Var, ffi::Array<Var>>, ffi::Array<Var>> FunctionUseDef(const Expr& fn) {
  auto usage = UDChain::Collect(fn);
  return {usage.downstream_usage, usage.outputs};
}

ffi::Map<Var, ffi::Array<Var>> DataflowBlockUseDef(const DataflowBlock& dfb) {
  auto usage = UDChain::Collect(SeqExpr({dfb}, Tuple(ffi::Array<Expr>())));
  return usage.downstream_usage;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.analysis.udchain", DataflowBlockUseDef)
      .def("relax.analysis.used_vars", [](const Expr& expr) {
        auto used_vars = GetUsedVars(expr);
        ffi::Array<Var> result;
        for (const VarObj* var_node : used_vars) {
          result.push_back(ffi::GetRef<Var>(var_node));
        }
        return result;
      });
}

VarUsageInfo CollectVarUsage(const Expr& expr) { return UDChain::Collect(expr); }

std::set<const VarObj*> GetUsedVars(const Expr& expr) {
  class UsedVars : public ExprVisitor {
   public:
    std::set<const VarObj*> used_vars;
    void VisitExprDepTypeField(const Type&) final {}
    void VisitExpr_(const VarObj* op) override { used_vars.insert(op); }
  } visitor;
  visitor.VisitExpr(expr);
  return std::move(visitor.used_vars);
}

}  // namespace relax
}  // namespace tvm
