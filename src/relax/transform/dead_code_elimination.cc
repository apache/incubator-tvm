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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/dead_code_elimination.cc
 * \brief Dead code elimination pass.
 * \sa tvm/relax/ir/binding_rewrite.cc
 *
 * Currently it removes:
 *   1. Unused local VarBindings
 *      (those where the bound var is unused and no impure operation is used).
 *   2. Unused Relax functions in the module.
 *      We detect the call chain from the entry function, and remove all unused functions.
 *
 * Any binding blocks that are left empty will be removed by the normalizer.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_set>

#include "utils.h"

namespace tvm {
namespace relax {

namespace {

struct RelaxCalleeCollector : relax::ExprVisitor {
  std::vector<GlobalVar>* callees;
  explicit RelaxCalleeCollector(std::vector<GlobalVar>* out) : callees(out) {}
  void VisitExpr_(const GlobalVarObj* node) final {
    callees->push_back(ffi::GetRef<GlobalVar>(node));
  }
};

struct TIRxCalleeCollector : tirx::StmtExprVisitor {
  std::vector<GlobalVar>* callees;
  explicit TIRxCalleeCollector(std::vector<GlobalVar>* out) : callees(out) {}
  using tirx::StmtExprVisitor::Visit_;
  ffi::Optional<VisitInterrupt> Visit_(const CallObj* node) final {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(tirx::StmtExprVisitor::Visit_(node));
    if (auto opt_gvar = node->op.as<GlobalVar>()) {
      callees->push_back(opt_gvar.value());
    }
    return std::nullopt;
  }
};

// Collect the GlobalVars directly called by `func`. Dedups while
// preserving first-encounter order (same semantics the old
// support::OrderedSet path provided).
ffi::Array<GlobalVar> CollectCallees(const BaseFunc& func) {
  std::vector<GlobalVar> raw;
  if (auto opt = func.as<relax::Function>()) {
    RelaxCalleeCollector visitor(&raw);
    visitor(opt.value());
  } else if (func.as<relax::ExternFunc>()) {
    // no callees
  } else if (auto opt = func.as<tirx::Function>()) {
    ffi::make_object<TIRxCalleeCollector>(&raw)->Visit(opt.value()->body);
  }
  // dedup preserving order
  ffi::Array<GlobalVar> result;
  std::unordered_set<GlobalVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> seen;
  for (const auto& gv : raw) {
    if (seen.insert(gv).second) result.push_back(gv);
  }
  return result;
}

ffi::Map<GlobalVar, ffi::Array<GlobalVar>> CollectCallMap(const IRModule& mod) {
  ffi::Map<GlobalVar, ffi::Array<GlobalVar>> call_map;
  for (const auto& [gvar, base_func] : mod->functions) {
    call_map.Set(gvar, CollectCallees(base_func));
  }
  return call_map;
}

}  // namespace

IRModule RemoveUnusedFunctions(IRModule mod, const std::unordered_set<GlobalVar>& entry_funcs) {
  auto call_map = CollectCallMap(mod);

  std::unordered_set<GlobalVar> reachable = entry_funcs;
  std::vector<GlobalVar> to_visit(entry_funcs.begin(), entry_funcs.end());
  bool all_callees_in_module = true;

  while (to_visit.size()) {
    GlobalVar visiting = to_visit.back();
    to_visit.pop_back();

    if (auto it = call_map.find(visiting); it != call_map.end()) {
      for (GlobalVar callee : (*it).second) {
        if (!reachable.count(callee)) {
          reachable.insert(callee);
          to_visit.push_back(callee);
        }
      }
    } else {
      all_callees_in_module = false;
    }
  }

  if (!all_callees_in_module) {
    return mod;
  }

  std::vector<GlobalVar> to_remove;
  for (const auto& [gvar, func] : mod->functions) {
    // The tracer contains all user-provided entry functions, all
    // externally-callable functions, and anything that is directly or
    // indirectly accessible from an entry function.
    if (!reachable.count(gvar)) {
      to_remove.push_back(gvar);
    }
  }

  if (to_remove.size()) {
    auto write_ptr = mod.CopyOnWrite();
    for (const auto& gvar : to_remove) {
      write_ptr->Remove(gvar);
    }
  }

  return mod;
}

IRModule DeadCodeElimination(const IRModule& arg_mod,
                             ffi::Array<ffi::String> entry_function_names) {
  IRModule mod = arg_mod;

  // S0: Make a list of all user-specified entry functions and
  // externally-visible entry functions.
  std::unordered_set<GlobalVar> entry_functions;
  for (const auto& name : entry_function_names) {
    entry_functions.insert(mod->GetGlobalVar(name));
  }
  for (const auto& gv : mod->GetGlobalVars()) {
    const auto& func = mod->Lookup(gv);
    if (func.as<ExternFuncObj>() || func->GetLinkageType() == LinkageType::kExternal) {
      entry_functions.insert(gv);
    }
  }

  // S1: remove unused functions to reduce the number of functions to be analyzed.
  mod = RemoveUnusedFunctions(mod, entry_functions);

  // S2: remove unused variables in each function.
  {
    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<Function>()) {
        auto new_func = RemoveAllUnused(opt.value()).as_or_throw<Function>();
        if (!new_func.same_as(base_func)) {
          updates->Add(gvar, new_func);
        }
      }
    }
    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
  }

  // S3: remove unused functions again as some callers may be removed in S2.
  mod = RemoveUnusedFunctions(mod, entry_functions);

  return mod;
}

namespace transform {

Pass DeadCodeElimination(ffi::Array<ffi::String> entry_functions) {
  auto pass_func = [=](IRModule m, PassContext pc) {
    return relax::DeadCodeElimination(m, entry_functions);
  };
  return CreateModulePass(pass_func, 1, "DeadCodeElimination");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.DeadCodeElimination", DeadCodeElimination);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
