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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>

#include <utility>

namespace tvm {
namespace relax {

ffi::Map<Var, Expr> NormalizeBindings(const Function& func,
                                      const ffi::Map<Any, ffi::ObjectRef>& untyped_params) {
  TVM_FFI_ICHECK(func.defined());
  TVM_FFI_ICHECK(untyped_params.defined());

  // Map from string to the variable(s) with that name.
  std::unordered_map<std::string, ffi::Array<tvm::Var>> string_lookup;
  std::unordered_set<const tvm::VarObj*> var_set;
  for (const auto& param : func->params) {
    string_lookup[param->name].push_back(param);
    var_set.insert(param.get());
  }

  ffi::Map<tvm::Var, relax::Expr> relax_var_remap;

  auto normalize_key = [&](ffi::Any obj) -> tvm::Var {
    if (auto opt_str = obj.as<ffi::String>()) {
      std::string str = opt_str.value();
      auto it = string_lookup.find(str);
      TVM_FFI_ICHECK(it != string_lookup.end())
          << "Function does not have parameter with name \"" << str << "\".  "
          << "Function parameters are named "
          << func->params.Map([](const auto& param) { return param->name; });
      TVM_FFI_ICHECK_EQ(it->second.size(), 1)
          << "Function contains multiple parameters with name \"" << str << "\".  "
          << "The Relax variables " << it->second << " are all named \"" << str << "\"";
      auto var = it->second[0];
      TVM_FFI_ICHECK(!relax_var_remap.count(var))
          << "Remap of variable " << var << " was defined multiple times";

      return var;
    } else if (auto opt_var = obj.as<tvm::Var>()) {
      auto var = opt_var.value();
      TVM_FFI_ICHECK(!relax_var_remap.count(var))
          << "Remap of variable " << var << " was defined multiple times";
      TVM_FFI_ICHECK(var_set.count(var.get()))
          << "Function does not use Relax variable " << var << " as a parameter.  "
          << "Function parameters are " << func->params;
      return var;
    } else {
      TVM_FFI_THROW(InternalError)
          << "Expected bound parameter to be a tvm::Var, "
          << " or a string that uniquely identifies a tvm::Var param within the function.  "
          << "However, received object " << obj << " of type " << obj.GetTypeKey();
    }
  };
  auto normalize_value = [&](ffi::Any obj) -> relax::Expr {
    if (auto opt = obj.as<relax::Expr>()) {
      return opt.value();
    } else if (auto tensor = obj.as<ffi::Tensor>(); tensor && tensor->defined()) {
      return MakeTensorConst(tensor.value());
    } else {
      TVM_FFI_THROW(InternalError)
          << "Cannot coerce object of type " << obj.GetTypeKey() << " into relax expression";
    }
  };

  for (const auto& [key, value] : untyped_params) {
    relax_var_remap.Set(normalize_key(key), normalize_value(value));
  }

  sym::Analyzer analyzer;
  return InferSymbolicVarMap(relax_var_remap, analyzer);
}

/*!
 * \brief Bind params to function by using name
 * \param func Relax function
 * \param params params dict
 * \return Function
 */
Function FunctionBindParams(Function func, const ffi::Map<Any, ffi::ObjectRef>& untyped_params) {
  ffi::Map<Var, Expr> bind_dict = NormalizeBindings(func, untyped_params);
  Expr bound_expr = Bind(func, bind_dict);
  return bound_expr.as_or_throw<Function>();
}

/*!
 * \brief Bind params to a specific function in a module
 * \param m The module
 * \param func_name The name of the specific function
 * \param param The param dict
 * \return The module after binding params.
 */
IRModule BindParam(IRModule m, ffi::String func_name, ffi::Map<Any, ffi::ObjectRef> bind_params) {
  IRModuleObj* new_module = m.CopyOnWrite();
  ffi::Map<GlobalVar, BaseFunc> functions = m->functions;
  for (const auto& func_pr : functions) {
    if (const auto* relax_f = func_pr.second.as<FunctionObj>()) {
      if (relax_f->GetLinkageType() == LinkageType::kExternal) {
        // Use global_symbol if it's external linkage
        ffi::Optional<ffi::String> gsymbol = relax_f->GetAttr(tvm::attr::kGlobalSymbol);
        if (gsymbol.has_value() && gsymbol.value() == func_name) {
          Function f_after_bind = FunctionBindParams(ffi::GetRef<Function>(relax_f), bind_params);
          new_module->Update(func_pr.first, f_after_bind);
        }
      } else {
        // Use global var's name_hint if it's internal linkage
        if (func_pr.first->name_hint == func_name) {
          Function f_after_bind = FunctionBindParams(ffi::GetRef<Function>(relax_f), bind_params);
          new_module->Update(func_pr.first, f_after_bind);
        }
      }
    }
  }
  return ffi::GetRef<IRModule>(new_module);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.FunctionBindParams", FunctionBindParams);
}

namespace transform {

Pass BindParams(ffi::String func_name, ffi::Map<Any, ffi::ObjectRef> params) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return BindParam(std::move(mod), func_name, params);
  };
  return CreateModulePass(pass_func, 0, "BindParams");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.BindParams", BindParams);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
