/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace relax {

Function FunctionBindSymbolicVars(
    Function func, ffi::Map<ffi::Variant<PrimVar, ffi::String>, PrimExpr> obj_remap) {
  // Early bail-out if no updates need to be made.
  if (obj_remap.empty()) {
    return func;
  }

  ffi::Array<tvm::Var> old_symbolic_vars = DefinedSymbolicVars(func);

  // Map from string to the variable(s) with that name.
  std::unordered_map<std::string, ffi::Array<tvm::Var>> string_lookup;
  std::unordered_set<const tvm::VarObj*> symbolic_var_set;
  for (const auto& var : old_symbolic_vars) {
    string_lookup[var->name].push_back(var);
    symbolic_var_set.insert(var.get());
  }

  // Replacement map to be used when rewriting the function.
  ffi::Map<Var, Expr> var_remap;
  for (const auto& [key, replacement] : obj_remap) {
    if (auto opt = key.as<ffi::String>()) {
      ffi::String string_key = opt.value();
      auto it = string_lookup.find(string_key);
      TVM_FFI_ICHECK(it != string_lookup.end())
          << "Function does not use symbolic var with name \"" << string_key << "\".  "
          << "Function has symbolic variables " << old_symbolic_vars;

      TVM_FFI_ICHECK_EQ(it->second.size(), 1)
          << "Function contains multiple symbolic variables with name \"" << string_key << "\".  "
          << "The TIR variables " << it->second << " are all named \"" << string_key << "\"";
      auto var = it->second[0];

      TVM_FFI_ICHECK(!var_remap.count(var))
          << "Remap of variable " << var << " was defined multiple times";
      var_remap.Set(var, replacement);
    } else if (auto opt = key.as<PrimVar>()) {
      auto var = opt.value();

      TVM_FFI_ICHECK(!var_remap.count(var))
          << "Remap of variable " << var << " was defined multiple times";
      TVM_FFI_ICHECK(symbolic_var_set.count(var.get()))
          << "Function does not use variable " << var << " as a symbolic variable.  "
          << "Function has symbolic variables " << old_symbolic_vars;
      var_remap.Set(var, replacement);
    } else {
      TVM_FFI_THROW(InternalError)
          << "Expected symbolic variable to be a tvm::Var or a string name, "
          << "but " << key << " was of type " << key.GetTypeKey();
    }
  }

  auto new_func = Bind(func, var_remap).as_or_throw<Function>();

  auto free_symbolic_vars = FreeSymbolicVars(new_func);

  TVM_FFI_ICHECK(free_symbolic_vars.empty())
      << "Resulting function should not have any undefined symbolic variables, "
      << "but TIR variables " << free_symbolic_vars << " were undefined.";

  return new_func;
}

namespace {
IRModule ModuleBindSymbolicVars(
    IRModule mod, ffi::Map<ffi::Variant<PrimVar, ffi::String>, PrimExpr> binding_map) {
  std::unordered_set<ffi::Any, ffi::AnyHash, ffi::AnyEqual> used;
  IRModule updates;
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto opt = base_func.as<Function>()) {
      auto func = opt.value();

      // Collect bindings that are used by this function.
      auto func_binding_map = [&]() -> ffi::Map<ffi::Variant<PrimVar, ffi::String>, PrimExpr> {
        std::unordered_set<std::string> var_names;
        std::unordered_set<const tvm::VarObj*> vars;
        for (const auto& var : DefinedSymbolicVars(func)) {
          var_names.insert(var->name);
          vars.insert(var.get());
        }

        ffi::Map<ffi::Variant<PrimVar, ffi::String>, PrimExpr> out;
        for (const auto& [key, replacement] : binding_map) {
          bool used_by_function = false;
          if (auto opt = key.as<ffi::String>()) {
            used_by_function = var_names.count(opt.value());
          } else if (auto var = key.as<PrimVar>()) {
            used_by_function = vars.count(var.value().get());
          } else {
            TVM_FFI_THROW(InternalError)
                << "Expected symbolic variable to be a tvm::Var "
                << "or a string name, but " << key << " was of type " << key.GetTypeKey();
          }
          if (used_by_function) {
            used.insert(key);
            out.Set(key, replacement);
          }
        }
        return out;
      }();
      func = FunctionBindSymbolicVars(func, func_binding_map);

      if (!func.same_as(base_func)) {
        updates->Add(gvar, func);
      }
    }
  }

  ffi::Array<ffi::Any> unused;
  for (const auto& [key, replacement] : binding_map) {
    if (!used.count(key)) {
      unused.push_back(key);
    }
  }
  TVM_FFI_ICHECK_EQ(unused.size(), 0) << "Binding map contains keys " << unused
                                      << ", which did not correspond to any symbolic variables "
                                      << "in the module.";

  if (updates->functions.size()) {
    mod.CopyOnWrite()->Update(updates);
  }
  return mod;
}
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.FunctionBindSymbolicVars", FunctionBindSymbolicVars);
}

namespace transform {

Pass BindSymbolicVars(ffi::Map<ffi::Variant<PrimVar, ffi::String>, PrimExpr> binding_map,
                      ffi::Optional<ffi::String> func_name) {
  auto pass_func = [=](IRModule mod, PassContext context) -> IRModule {
    if (func_name) {
      auto gvar = mod->GetGlobalVar(func_name.value());
      auto func = mod->Lookup(gvar).as_or_throw<Function>();
      auto new_func = FunctionBindSymbolicVars(func, binding_map);
      if (!func.same_as(new_func)) {
        mod.CopyOnWrite()->Update(gvar, new_func);
      }
    } else {
      mod = ModuleBindSymbolicVars(mod, binding_map);
    }
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 1, "relax.BindSymbolicVars");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.BindSymbolicVars", BindSymbolicVars);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
