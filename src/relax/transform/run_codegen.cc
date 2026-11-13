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
 * \file tvm/relax/transform/run_codegen.cc
 * \brief Run codegen for annotated relax functions.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>

#include "../../support/ordered_set.h"
#include "utils.h"

namespace tvm {
namespace relax {

class CodeGenRunner : ExprMutator {
 public:
  using OptionMap = ffi::Map<ffi::String, ffi::Any>;

  explicit CodeGenRunner(IRModule mod) : ExprMutator(mod) {}

  IRModule Run(ffi::Optional<ffi::Map<ffi::String, OptionMap>> target_options,
               ffi::Array<ffi::String> entry_function_names) {
    IRModule mod = builder_->GetContextIRModule();

    support::OrderedSet<GlobalVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> entry_functions;
    // Any user-provided functions are treated as entry functions.
    for (const auto& name : entry_function_names) {
      entry_functions.insert(mod->GetGlobalVar(name));
    }

    // In addtion, any externally-exposed function that does not
    // belong to a specific codegen may be an entry function.  These
    // are added in alphabetical order, to ensure consistent order of
    // evaluation for debug/test purposes.
    {
      std::vector<GlobalVar> attr_entry_functions;
      for (const auto& [gv, func] : mod->functions) {
        if (func->GetLinkageType() == LinkageType::kExternal &&
            !func->GetAttr(tvm::relax::attr::kCodegen) && func->IsInstance<relax::FunctionObj>()) {
          attr_entry_functions.push_back(gv);
        }
      }
      std::sort(attr_entry_functions.begin(), attr_entry_functions.end(),
                [](const auto& gvar_a, const auto& gvar_b) {
                  return gvar_a->name_hint > gvar_b->name_hint;
                });
      for (const auto& gvar : attr_entry_functions) {
        entry_functions.insert(gvar);
      }
    }

    for (const auto& gvar : entry_functions) {
      builder_->UpdateFunction(gvar, VisitExpr(mod->Lookup(gvar)).as_or_throw<BaseFunc>());
    }

    auto ext_mods = InvokeCodegen(mod, target_options.value_or({}));
    auto out_mod = builder_->GetContextIRModule();

    if (ext_mods.size()) {
      if (auto opt_old_ext_mods = mod->GetAttr(tvm::attr::kExternalMods)) {
        auto old_ext_mods = opt_old_ext_mods.value();
        ext_mods.insert(ext_mods.begin(), old_ext_mods.begin(), old_ext_mods.end());
      }
      out_mod = WithAttr(out_mod, tvm::attr::kExternalMods.name, std::move(ext_mods));
    }

    if (constant_names.size()) {
      // Some backends (e.g. TensorRT) expect constants to be passed when they are instantiated
      ffi::Map<ffi::String, ffi::Tensor> constants;
      for (const auto& [constant, name] : constant_names) {
        TVM_FFI_ICHECK(!constants.count(name)) << "More than one constant with the name " << name;
        constants.Set(name, constant->value.cast<ffi::Tensor>());
      }
      out_mod =
          WithAttr(out_mod, tvm::relax::attr::kConstNameToConstant.name, std::move(constants));
    }

    // TODO(@tvm-team): Implicit pass dependency. Revisit when we have a better way to handle this.
    return DeadCodeElimination(out_mod, entry_function_names);
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* call_node) override {
    auto call = ExprMutator::VisitExpr_(call_node).as_or_throw<Call>();
    if (auto const* gvar_node = call_node->op.as<GlobalVarObj>()) {
      const GlobalVar gvar = ffi::GetRef<GlobalVar>(gvar_node);

      auto create_call_dps_packed = [call_node, this](Expr extern_func, Type ret_ty) {
        ffi::Array<Expr> new_args({extern_func});
        new_args.push_back(Tuple(call_node->args.Map([this](Expr arg) { return VisitExpr(arg); })));

        static const Op call_op = Op::Get("relax.call_dps_packed");

        return Call(Type::Missing(), call_op, new_args, std::nullopt, {ret_ty});
      };

      auto ret_ty = GetType(call);
      if (auto it = extern_funcs_.find(gvar_node); it != extern_funcs_.end()) {
        return create_call_dps_packed(it->second, ret_ty);
      } else if (auto opt_func = builder_->GetContextIRModule()->Lookup(gvar).as<Function>()) {
        // TODO(@sunggg): Is there any better way to get this func?
        Function func = opt_func.value();
        Expr new_func = VisitExpr(func);

        if (new_func->IsInstance<ExternFuncObj>()) {
          extern_funcs_.insert_or_assign(gvar_node, new_func);
          // Remove the global symbol and codegen attributes from the function so that it can be
          // removed the module.
          func = WithoutAttr(std::move(func), tvm::attr::kGlobalSymbol.name);
          func = WithoutAttr(std::move(func), tvm::relax::attr::kCodegen.name);
          builder_->UpdateFunction(gvar, func);
          return create_call_dps_packed(new_func, ret_ty);
        }
      }
    }
    ffi::Array<Expr> new_args;
    for (const auto& arg : call_node->args) {
      new_args.push_back(VisitExpr(arg));
    }

    Type ret_ty = Type::Missing();
    if (call_node->ty.as<PrimTypeObj>()) {
      if (auto op = call_node->op.as<Op>()) {
        static auto infer_type_map = Op::GetAttrMap(tvm::op_attr::kInferType);
        static auto infer_type_with_builder_map =
            Op::GetAttrMap(tvm::relax::op_attr::kInferTypeWithBuilder);
        if (!infer_type_map.count(op.value()) && !infer_type_with_builder_map.count(op.value())) {
          ret_ty = call_node->ty.as_or_throw<Type>();
        }
      }
    }
    return Call(ret_ty, call_node->op, new_args, call_node->attrs, call_node->ty_args,
                call_node->loc);
  }

  Expr VisitExpr_(const FunctionObj* func_node) override {
    Function func = ffi::GetRef<Function>(func_node);
    auto opt_codegen = func->GetAttr(tvm::relax::attr::kCodegen);
    if (opt_codegen) {
      auto ext_symbol = GetExtSymbol(func);
      size_t count = 0;
      PostOrderVisit(func->body, [=, this, &count](Expr e) {
        if (auto constant = e.as<GenericConst>();
            constant && constant.value()->value.as<ffi::TensorObj>()) {
          // Make sure to pick a unique name
          auto name = ext_symbol + "_" + opt_codegen.value() + "_const_" + std::to_string(count++);
          constant_names.Set(constant.value(), name);
        }
      });
      return ExternFunc(GetExtSymbol(func));
    } else {
      return ExprMutator::VisitExpr_(func_node);
    }
  }

 private:
  ffi::Array<ffi::Module> InvokeCodegen(IRModule mod,
                                        ffi::Map<ffi::String, OptionMap> target_options) {
    std::unordered_map<std::string, ffi::Array<Function>> target_functions;

    for (const auto& entry : mod->functions) {
      if (entry.second->IsInstance<tirx::FunctionObj>()) {
        continue;
      }
      PostOrderVisit(entry.second, [&target_functions](Expr e) {
        if (e->IsInstance<FunctionObj>()) {
          auto f = e.as_or_throw<Function>();
          if (auto target_opt = f->GetAttr(tvm::relax::attr::kCodegen)) {
            ffi::String target = target_opt.value();
            target_functions[target].push_back(f);
          }
        }
      });
    }

    ffi::Array<ffi::Module> ext_mods;

    for (const auto& [target, functions] : target_functions) {
      OptionMap options = target_options.Get(target).value_or(OptionMap());
      // Start the codegen process.
      // Get the codegen with its ffi key.
      ffi::String codegen_name = "relax.ext." + target;
      const auto codegen = tvm::ffi::Function::GetGlobal(codegen_name);
      TVM_FFI_ICHECK(codegen.has_value()) << "Codegen is not found: " << codegen_name << "\n";

      ffi::Array<ffi::Module> compiled_functions =
          (*codegen)(functions, options, constant_names).cast<ffi::Array<ffi::Module>>();
      ext_mods.insert(ext_mods.end(), compiled_functions.begin(), compiled_functions.end());
    }

    return ext_mods;
  }

  /*! \brief The names of all constants in the original module. */
  ffi::Map<GenericConst, ffi::String> constant_names;
  /*! \brief Extern funcs for each global variable.  */
  std::unordered_map<const GlobalVarObj*, Expr> extern_funcs_;
};

}  // namespace relax

namespace transform {
Pass RunCodegen(
    ffi::Optional<ffi::Map<ffi::String, ffi::Map<ffi::String, ffi::Any>>> target_options,
    ffi::Array<ffi::String> entry_functions) {
  auto pass_func = [=](IRModule m, PassContext pc) {
    return relax::CodeGenRunner(m).Run(target_options, entry_functions);
  };
  return CreateModulePass(pass_func, 0, "RunCodegen");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.RunCodegen", RunCodegen);
}

}  // namespace transform
}  // namespace tvm
