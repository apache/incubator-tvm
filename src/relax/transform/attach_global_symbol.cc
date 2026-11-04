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
 * \file src/relax/transform/attach_global_symbol.cc
 * \brief Attach global_symbol to Relax functions and TIR Functions for codegen.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <vector>

namespace tvm {
namespace relax {
namespace transform {

namespace {

// File-local mutator: replace GlobalVar references inside a relax::Function.
struct RelaxGvarMutator : ExprMutator {
  ffi::Map<GlobalVar, GlobalVar> replacements;
  explicit RelaxGvarMutator(ffi::Map<GlobalVar, GlobalVar> replacements)
      : replacements(replacements) {}

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const GlobalVarObj* node) override {
    auto gvar = ffi::GetRef<GlobalVar>(node);
    return replacements.Get(gvar).value_or(gvar);
  }
};

// File-local mutator: replace GlobalVar references inside a tirx::Function.
struct TirxGvarMutator : tirx::StmtExprMutator {
  ffi::Map<GlobalVar, GlobalVar> replacements;
  explicit TirxGvarMutator(ffi::Map<GlobalVar, GlobalVar> replacements)
      : replacements(replacements) {}

  using tirx::StmtExprMutator::Mutate_;
  UnchangedOr<Expr> Mutate_(const CallObj* node, InplaceMode inplace_mode) override {
    auto call = tirx::StmtExprMutator::Mutate_(node, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(node))
                    .as_or_throw<tvm::Call>();
    if (auto old_gvar = call->op.as<GlobalVar>()) {
      if (auto new_gvar = replacements.Get(old_gvar.value())) {
        call.CopyOnWrite()->op = new_gvar.value();
      }
    }
    return call.as_or_throw<PrimExpr>();
  }
};

// Replace GlobalVar references across all functions in the module.
// Direct dispatch on function type — no ObjectFunctor indirection needed
// since this file already includes the relax + tirx headers.
IRModule ReplaceGlobalVarsInModule(IRModule mod, ffi::Map<GlobalVar, GlobalVar> replacements) {
  if (replacements.empty()) {
    return mod;
  }

  std::vector<GlobalVar> to_remove;
  IRModule updates;

  for (const auto& [old_gvar, old_func] : mod->functions) {
    auto new_gvar = replacements.Get(old_gvar).value_or(old_gvar);
    BaseFunc new_func = old_func;

    if (auto* function_node = old_func.as<tirx::FunctionObj>()) {
      auto func = ffi::GetRef<tirx::Function>(function_node);
      auto mutator = ffi::make_object<TirxGvarMutator>(replacements);
      auto new_body =
          mutator->Mutate(func->body, InplaceMode::kDisallow).ValueOrUnchanged(func->body);
      if (!new_body.same_as(func->body)) {
        func.CopyOnWrite()->body = new_body;
      }
      // Update kGlobalSymbol if the function is externally exposed and being renamed.
      if (func->GetAttr(tvm::attr::kGlobalSymbol)) {
        if (new_gvar->name_hint != old_gvar->name_hint) {
          func = WithAttr(func, tvm::attr::kGlobalSymbol.name, new_gvar->name_hint);
        }
      }
      new_func = func;
    } else if (auto* relax_func_node = old_func.as<FunctionObj>()) {
      RelaxGvarMutator mutator(replacements);
      auto new_relax_func = mutator(ffi::GetRef<Function>(relax_func_node)).as_or_throw<Function>();
      // Update kGlobalSymbol if the function is externally exposed and being renamed.
      if (new_relax_func->GetAttr(tvm::attr::kGlobalSymbol)) {
        if (new_gvar->name_hint != old_gvar->name_hint) {
          new_relax_func =
              WithAttr(new_relax_func, tvm::attr::kGlobalSymbol.name, new_gvar->name_hint);
        }
      }
      new_func = new_relax_func;
    } else if (old_func.as<ExternFuncObj>()) {
      // ExternFunc: no internal GlobalVar references to update.
      new_func = old_func;
    } else {
      new_func = old_func;
    }

    if (!new_gvar.same_as(old_gvar)) {
      to_remove.push_back(old_gvar);
    }
    if (!old_gvar.same_as(new_gvar) || !old_func.same_as(new_func)) {
      updates->Add(new_gvar, new_func);
    }
  }

  if (to_remove.size() || updates->functions.size()) {
    auto write_ptr = mod.CopyOnWrite();
    for (const auto& old_gvar : to_remove) {
      write_ptr->Remove(old_gvar);
    }
    write_ptr->Update(updates);
  }
  return mod;
}

}  // namespace

Pass AttachGlobalSymbol() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    ffi::String c_prefix = mod->GetAttr(tvm::attr::kSystemLibPrefix).value_or("");
    IRModule updates;
    ffi::Map<GlobalVar, GlobalVar> gvar_updates;

    for (const auto& [gvar, func] : mod->functions) {
      ffi::Optional<ffi::String> old_name = func->GetAttr(tvm::attr::kGlobalSymbol);

      // TODO(tvm-team): re-enable once fix relax integration part
      // if (old_name) continue;

      ffi::Optional<ffi::String> new_name;
      BaseFunc new_func = func;

      if (auto* function = func.as<tirx::FunctionObj>()) {
        new_name = c_prefix + gvar->name_hint;
        new_func = WithAttr(ffi::GetRef<tirx::Function>(function), tvm::attr::kGlobalSymbol.name,
                            new_name);
      } else if (auto* relax_func = func.as<FunctionObj>()) {
        new_name = gvar->name_hint;
        new_func =
            WithAttr(ffi::GetRef<Function>(relax_func), tvm::attr::kGlobalSymbol.name, new_name);
      }

      if (new_name.has_value() && (!old_name.has_value() || old_name.value() != new_name.value())) {
        updates->Add(gvar, new_func);
        if (new_name.value() != gvar->name_hint) {
          GlobalVar new_gvar(new_name.value());
          if (auto ty = gvar->ty.as<Type>()) {
            UpdateType(new_gvar, ty.value());
          }

          gvar_updates.Set(gvar, new_gvar);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);

      if (gvar_updates.size()) {
        mod = ReplaceGlobalVarsInModule(mod, gvar_updates);
      }
    }
    return mod;
  };
  return CreateModulePass(pass_func, 0, "AttachGlobalSymbol");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AttachGlobalSymbol", AttachGlobalSymbol);
}
}  // namespace transform
}  // namespace relax
}  // namespace tvm
