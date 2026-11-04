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
 * \file tvm/relax/transform/update_param_type.cc
 * \brief Mutate IRModule to accept new parameters
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

namespace {
class ParamTypeMutator : public ExprMutator {
 public:
  explicit ParamTypeMutator(ffi::TypedFunction<ffi::Optional<Type>(Var)> ty_func)
      : ty_func_(ty_func) {}

  using ExprMutator::VisitExpr_;
  using ExprMutator::VisitVarDef_;

  Expr VisitExpr_(const FunctionObj* op) override {
    auto func = ffi::GetRef<Function>(op);

    auto params = op->params.Map([this](Var param) {
      if (auto new_ty = ty_func_(param)) {
        auto new_param = WithType(param, new_ty.value());
        var_remap_.insert_or_assign(param, new_param);
        return new_param;
      } else {
        return param;
      }
    });

    if (!params.same_as(func->params)) {
      func.CopyOnWrite()->params = params;
    }
    return ExprMutator::VisitExpr_(func.get());
  }

  ffi::TypedFunction<ffi::Optional<Type>(Var)> ty_func_;
};
}  // namespace

namespace transform {
Pass UpdateParamType(ffi::TypedFunction<ffi::Optional<Type>(Var)> ty_func) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    ParamTypeMutator mutator(ty_func);

    std::unordered_set<GlobalVar> to_remove;
    std::unordered_map<GlobalVar, Function> to_add;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto updated = mutator(func.value()).as_or_throw<Function>();
        if (!updated.same_as(base_func)) {
          GlobalVar new_gvar(gvar->name_hint);
          UpdateType(new_gvar, GetType(updated));
          to_add.insert({new_gvar, updated});
          to_remove.insert(gvar);
        }
      }
    }

    if (to_remove.size() || to_add.size()) {
      auto write_ptr = mod.CopyOnWrite();

      for (const auto& gvar : to_remove) {
        write_ptr->Remove(gvar);
      }
      for (const auto& [gvar, func] : to_add) {
        write_ptr->Add(gvar, func);
      }
    }

    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 1, "UpdateParamType");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.UpdateParamType", UpdateParamType);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
