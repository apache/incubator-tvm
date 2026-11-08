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
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <algorithm>
#include <tuple>

#include "global_var_reference.h"

namespace tvm {
namespace relax {

namespace {

template <typename T, typename U>
using PMap = std::unordered_map<T, U, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

ffi::Optional<Function> ExpandParams(Function func) {
  bool is_exposed = func->attrs.GetAttr(tvm::attr::kGlobalSymbol).has_value();
  if (is_exposed) return std::nullopt;

  bool has_tuple_param =
      std::any_of(func->params.begin(), func->params.end(),
                  [](const Var& param) -> bool { return param->ty.as<TupleTypeObj>(); });

  if (!has_tuple_param) return std::nullopt;

  ffi::Array<Var> params;
  ffi::Array<Binding> bindings;

  std::function<void(const Var&)> expand_param = [&](const Var& param) {
    if (auto ty = param->ty.as<TupleTypeObj>()) {
      ffi::Array<Expr> internal_tuple;
      for (size_t i = 0; i < ty->fields.size(); i++) {
        auto name =
            static_cast<const std::stringstream&>(std::stringstream() << param->name << "_" << i)
                .str();
        Var new_param(name, ty->fields[i]);
        internal_tuple.push_back(new_param);
        expand_param(new_param);
      }
      bindings.push_back(VarBinding(param, Tuple(internal_tuple)));
    } else {
      params.push_back(param);
    }
  };

  for (const auto& param : func->params) {
    expand_param(param);
  }

  FuncType new_ty(params.Map([](const auto& var) { return GetType(var); }), func->ret_ty,
                  func->ty.as_or_throw<FuncType>()->purity);

  auto write_ptr = func.CopyOnWrite();
  write_ptr->params = params;
  write_ptr->body = SeqExpr({BindingBlock(bindings)}, func->body);
  write_ptr->ty = new_ty;

  return func;
}

class TupleExpander : public ExprMutator {
 public:
  explicit TupleExpander(PMap<GlobalVar, GlobalVar> callees) : replacements_(callees) {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* op) override {
    auto node = ExprMutator::VisitExpr_(op).as_or_throw<Call>();

    if (auto gvar = node->op.as<GlobalVar>()) {
      if (auto it = replacements_.find(gvar.value()); it != replacements_.end()) {
        ffi::Array<Expr> new_args;

        std::function<void(const Expr&)> expand_arg = [&](const Expr& arg) {
          if (auto ty = arg->ty.as<TupleTypeObj>()) {
            for (size_t i = 0; i < ty->fields.size(); i++) {
              expand_arg(TupleGetItem(arg, i));
            }
          } else {
            new_args.push_back(arg);
          }
        };

        for (const auto& arg : node->args) {
          expand_arg(arg);
        }

        auto write_ptr = node.CopyOnWrite();
        write_ptr->op = it->second;
        write_ptr->args = new_args;
      }
    }

    return node;
  }

  PMap<GlobalVar, GlobalVar> replacements_;
};

}  // namespace

namespace transform {

Pass ExpandTupleArguments() {
  auto pass_func = [=](IRModule mod, PassContext pc) -> IRModule {
    auto value_references = GlobalVarValueCollector::Collect(mod);
    PMap<GlobalVar, GlobalVar> gvar_replacements;

    {
      PMap<GlobalVar, Function> new_callees;

      for (const auto& [gvar, base_func] : mod->functions) {
        if (value_references.count(gvar)) continue;
        if (auto func = base_func.as<Function>()) {
          if (auto opt = ExpandParams(func.value())) {
            auto new_func = opt.value();
            GlobalVar new_gvar(gvar->name_hint);
            new_gvar->ty = new_func->ty;
            gvar_replacements.insert_or_assign(gvar, new_gvar);
            new_callees.insert_or_assign(new_gvar, new_func);
          }
        }
      }

      if (gvar_replacements.empty()) {
        return mod;
      }
      auto write_ptr = mod.CopyOnWrite();
      for (auto [old_gvar, new_gvar] : gvar_replacements) {
        write_ptr->Remove(old_gvar);
        write_ptr->Add(new_gvar, new_callees.at(new_gvar));
      }
    }

    TupleExpander mutator(std::move(gvar_replacements));

    IRModule caller_updates;

    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        auto mutated = mutator.VisitExpr(func.value()).as_or_throw<Function>();
        if (!mutated.same_as(base_func)) {
          caller_updates->Add(gvar, mutated);
        }
      }
    }

    if (caller_updates->functions.size()) {
      mod.CopyOnWrite()->Update(caller_updates);
    }
    return mod;
  };
  auto inner_pass = CreateModulePass(pass_func, 0, "ExpandTupleArgumentsInner");

  return tvm::transform::Sequential(
      {
          inner_pass,
          CanonicalizeBindings(),
          DeadCodeElimination({}),
      },
      "ExpandTupleArguments");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.ExpandTupleArguments", ExpandTupleArguments);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
