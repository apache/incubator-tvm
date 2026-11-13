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
 * \file computable_at_compile_time.cc
 *
 * \brief Utilities for identifying potentially compile-time variables
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include "../../support/ordered_set.h"

namespace tvm {
namespace relax {

namespace {
class CompileTimeCollector : ExprVisitor {
 public:
  static ffi::Array<Var> Collect(const Function& func) {
    CompileTimeCollector visitor;
    visitor(func);
    return ffi::Array<Var>(visitor.computable_vars_.begin(), visitor.computable_vars_.end());
  }

 private:
  void VisitExpr_(const FunctionObj* func) override {
    if (auto opt_num_input = func->attrs.GetAttr(tvm::relax::attr::kNumInput)) {
      size_t num_input = opt_num_input.value();
      for (size_t i = num_input; i < func->params.size(); i++) {
        MarkAsKnown(func->params[i]);
      }
    }

    ExprVisitor::VisitExpr_(func);
  }

  void VisitBinding(const Binding& binding) override {
    Expr value = GetBoundValue(binding);
    bool can_compute_at_compile_time = [&]() {
      for (const Var& var : FreeVars(value)) {
        if (!known_vars_.count(var)) {
          return false;
        }
      }

      return true;
    }();

    if (can_compute_at_compile_time) {
      MarkAsKnown(binding->var);
    }

    ExprVisitor::VisitBinding(binding);
  }

  void MarkAsKnown(const Var& var) {
    known_vars_.insert(var);
    computable_vars_.insert(var);
    for (const auto& tir_var : DefinableTIRVarsInType(GetType(var))) {
      known_vars_.insert(tir_var);
    }
  }

  support::OrderedSet<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> known_vars_;
  support::OrderedSet<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> computable_vars_;
};
}  // namespace

ffi::Array<Var> ComputableAtCompileTime(const Function& func) {
  return CompileTimeCollector::Collect(func);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.computable_at_compile_time", ComputableAtCompileTime);
}

}  // namespace relax
}  // namespace tvm
