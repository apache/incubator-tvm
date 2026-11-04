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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <utility>
#include <vector>

#include "../../support/ordered_set.h"
#include "utils.h"

namespace tvm {
namespace relax {

namespace {

class FunctionInliner : public ExprMutator {
 public:
  explicit FunctionInliner(
      const ffi::Map<ffi::Variant<ffi::String, GlobalVar>, Function>& replacements)
      : replacements_(replacements) {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const FunctionObj* op) override {
    auto node = ExprMutator::VisitExpr_(op);
    if (node.get() != op) {
      node = CanonicalizeBindings(node);
      node = RemoveAllUnused(node);
    }
    return node;
  }

  Expr VisitExpr_(const CallObj* op) override {
    auto node = ExprMutator::VisitExpr_(op).as_or_throw<Call>();

    if (auto opt = node->op.as<GlobalVar>()) {
      auto gvar = opt.value();
      if (auto opt = GetFunction(gvar)) {
        auto callee = opt.value();
        TVM_FFI_ICHECK_EQ(callee->params.size(), node->args.size())
            << "Attempted to inline call to " << gvar << ", which accepts " << callee->params.size()
            << " parameters.  "
            << "However, it was called with " << node->args.size() << " arguments in expression "
            << node;

        Expr inlined = InlinedCall(callee, node->args);

        TVM_FFI_ICHECK(!inline_stack_.count(gvar))
            << "Relax function inlining does not support recursive functions.  "
            << "However, recursive function " << gvar << " was requested to be inlined.";

        inline_stack_.insert(gvar);
        inlined = VisitExpr(std::move(inlined));
        inline_stack_.erase(gvar);

        return inlined;
      }
    }

    return node;
  }

 private:
  ffi::Optional<Function> GetFunction(const GlobalVar& gvar) const {
    if (auto opt = replacements_.Get(gvar)) {
      return opt;
    } else if (auto opt = replacements_.Get(gvar->name_hint)) {
      return opt;
    } else {
      return std::nullopt;
    }
  }

  Expr InlinedCall(Function func, const ffi::Array<Expr>& args) const {
    // Ensures that the inlined instance does not have duplicate usage
    // with other inlined copies, or with the original callee.
    func = CopyWithNewVars(std::move(func));

    ffi::Array<Binding> param_bindings;

    ffi::Map<Var, Expr> param_map;
    for (size_t i = 0; i < args.size(); i++) {
      // Option 1: Use tvm::relax::Bind to substitute arguments into
      // the body.  If the arguments contain DataflowVar instances,
      // but the subroutine does not use DataflowBlock, this would
      // result in invalid AST.
      //
      // Option 2: Define a VarBinding `param[i] = args[i]` for each
      // parameter, then rely on CanonicalizeBindings to replace with
      // DataflowVar where possible.  This would solve the invalid use
      // of DataflowVar, but wouldn't handle symbolic variables.  If
      // the subroutine has symbolic variables defined by its
      // arguments, the VarBinding would leave them undefined.
      //
      // Option 3: Define a MatchCast `param[i] = args[i]` for each
      // parameter, followed by CanonicalizeBindings.  This is the
      // first option that would result in well-formed AST, but it
      // wouldn't be optimal.  Symbolic variables would have two
      // copies, one from the initial definition, and one
      // from the MatchCast inlined portion.
      //
      // Option 4: Define a VarBinding `param[i] = args[i]`, with
      // CanonicalizeBindings to handle conversion of Var to
      // DataflowVar, and tvm::relax::Bind to handle substitution of
      // symbolic variables.  This would result in a well-formed Relax
      // function, with no duplicate definitions of symbolic
      // variables.
      //
      // This implementation uses Option 4.

      Var param_var(func->params[i]->name, args[i]->ty);
      param_bindings.push_back(VarBinding(param_var, args[i]));
      param_map.Set(func->params[i], param_var);
    }

    DataflowBlock binding_block(param_bindings);
    Expr body = Bind(func, param_map).as<FunctionObj>()->body;

    return SeqExpr({binding_block}, body);
  }

  const ffi::Map<ffi::Variant<ffi::String, GlobalVar>, Function>& replacements_;
  std::unordered_set<GlobalVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> inline_stack_;
};
}  // namespace

/*!
 * \brief Bind params to function by using name
 * \param func Relax function
 * \param params params dict
 * \return Function
 */
Function FunctionInlineFunctions(
    Function func, const ffi::Map<ffi::Variant<ffi::String, GlobalVar>, Function>& replacements) {
  for (const auto& [key, func] : replacements) {
    if (auto ptr = key.as<GlobalVarObj>()) {
      TVM_FFI_CHECK(!replacements.count(ptr->name_hint), ValueError)
          << "Map of functions to inline must be unambiguous.  "
          << "However, the map provided contains both the GlobalVar " << key << " and the string \'"
          << ptr->name_hint << "'";
    }
  }

  FunctionInliner mutator(replacements);
  return mutator(std::move(func)).as_or_throw<Function>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.FunctionInlineFunctions", FunctionInlineFunctions);
}

namespace transform {

Pass InlinePrivateFunctions() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    ffi::Map<ffi::Variant<ffi::String, GlobalVar>, Function> replacements;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<relax::Function>()) {
        auto func = opt.value();
        bool is_private = !func->GetAttr(tvm::attr::kGlobalSymbol).has_value();
        if (is_private) {
          replacements.Set(gvar, func);
        }
      }
    }

    if (replacements.empty()) {
      // Early bail-out if there are no private functions.
      return mod;
    }

    for (const auto& recursive_set : DetectRecursion(mod)) {
      for (const auto& recursive_func : recursive_set) {
        replacements.erase(recursive_func);
      }
    }

    if (replacements.empty()) {
      // Early bail-out if all private functions are recursive.
      return mod;
    }

    IRModule updates;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (!replacements.count(gvar)) {
        if (auto opt = base_func.as<relax::Function>()) {
          auto func = FunctionInlineFunctions(opt.value(), replacements);
          if (!base_func.same_as(func)) {
            updates->Add(gvar, func);
          }
        }
      }
    }

    std::vector<GlobalVar> funcs_to_remove;
    for (const auto& [gvar, base_func] : mod->functions) {
      if (replacements.count(gvar)) {
        funcs_to_remove.push_back(gvar);
      }
    }

    auto write_ptr = mod.CopyOnWrite();
    for (const GlobalVar& gvar : funcs_to_remove) {
      write_ptr->Remove(gvar);
    }
    write_ptr->Update(updates);
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "InlinePrivateFunctions");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.InlinePrivateFunctions", InlinePrivateFunctions);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
