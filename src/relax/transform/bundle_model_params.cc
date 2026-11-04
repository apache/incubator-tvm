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
 * \file tvm/relax/transform/bundle_model_params.cc
 * \brief Lift local functions into global functions.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <unordered_set>

#include "utils.h"

namespace tvm {
namespace relax {

class ModelParamBundler : public ExprMutator {
 public:
  explicit ModelParamBundler(ffi::Optional<ffi::String> param_tuple_name)
      : param_tuple_name_(param_tuple_name) {}

  Expr VisitExpr_(const FunctionObj* op) override {
    Function func = ffi::GetRef<Function>(op);
    auto opt_num_input = func->attrs.GetAttr(tvm::relax::attr::kNumInput);
    if (!opt_num_input) return func;
    auto signed_num_input = opt_num_input.value();

    TVM_FFI_ICHECK_GE(signed_num_input, 0);
    TVM_FFI_ICHECK_LE(signed_num_input, func->params.size())
        << "Function was declared to have " << signed_num_input << " runtime inputs, "
        << "but only has " << func->params.size() << " parameters total.";
    size_t num_input = signed_num_input;

    ffi::Array<Var> params;
    for (size_t i = 0; i < num_input; i++) {
      params.push_back(func->params[i]);
    }

    std::unordered_set<const VarObj*> signature_vars;
    for (const tvm::Var& var : DefinableTIRVarsInType(TupleType(params.Map(GetType)))) {
      signature_vars.insert(var.get());
    }

    std::unordered_set<const VarObj*> bundled_prim_params;
    ffi::Array<Var> bundled_prim_params_in_order;
    for (size_t i = num_input; i < func->params.size(); i++) {
      if (func->params[i].as<PrimVar>()) {
        bundled_prim_params.insert(func->params[i].get());
        bundled_prim_params_in_order.push_back(func->params[i]);
      }
    }

    auto erase_removed_prim_params = [&](const Type& type) {
      return EraseToWellDefined(type, [&](const Var& var) -> ffi::Optional<Expr> {
        if (bundled_prim_params.count(var.get()) && !signature_vars.count(var.get())) {
          return std::nullopt;
        }
        return var;
      });
    };

    ffi::Array<Type> param_tuple;
    for (size_t i = num_input; i < func->params.size(); i++) {
      Type field_type = erase_removed_prim_params(GetType(func->params[i]));
      param_tuple.push_back(field_type);
      var_to_field_type_.Set(func->params[i], field_type);
    }

    Var var_param_tuple(param_tuple_name_.value_or("model_params"), TupleType(param_tuple));
    params.push_back(var_param_tuple);

    for (size_t i = num_input; i < func->params.size(); i++) {
      var_to_expr_.Set(func->params[i], TupleGetItem(var_param_tuple, i - num_input));
    }

    Type ret_ty = erase_removed_prim_params(func->ret_ty);
    bool previous_rewrite_model_params = rewrite_model_params_;
    ffi::Array<Var> previous_pending_prim_params = std::move(pending_prim_params_);
    rewrite_model_params_ = true;
    pending_prim_params_ = std::move(bundled_prim_params_in_order);
    Expr body = VisitWithNewScope(func->body, params);
    rewrite_model_params_ = previous_rewrite_model_params;
    pending_prim_params_ = std::move(previous_pending_prim_params);
    return Function(params, body, ret_ty, func->is_pure, func->attrs);
  }

  Expr VisitExpr_(const SeqExprObj* op) override {
    if (pending_prim_params_.empty()) {
      return ExprMutator::VisitExpr_(op);
    }

    ffi::Array<Var> prim_params = std::move(pending_prim_params_);
    pending_prim_params_.clear();

    builder_->BeginBindingBlock();
    for (const Var& var : prim_params) {
      auto it = var_to_expr_.find(var);
      TVM_FFI_ICHECK(it != var_to_expr_.end());
      var_remap_.insert_or_assign(var, builder_->Emit((*it).second, var->name));
    }
    BindingBlock prologue = builder_->EndBlock();

    SeqExpr body = ExprMutator::VisitExpr_(op).as_or_throw<SeqExpr>();
    ffi::Array<BindingBlock> blocks;
    blocks.push_back(prologue);
    for (const BindingBlock& block : body->blocks) {
      blocks.push_back(block);
    }
    return SeqExpr(blocks, body->body);
  }

  Expr VisitExpr_(const VarObj* op) override {
    auto var = ffi::GetRef<Var>(op);
    if (!rewrite_model_params_) {
      return ExprMutator::VisitExpr_(op);
    }
    if (auto it = var_to_expr_.find(var); it != var_to_expr_.end()) {
      bool is_prim_param = var.as<PrimVar>().has_value();
      if (is_prim_param) {
        if (auto cached = var_remap_.find(var); cached != var_remap_.end()) {
          return cached->second;
        }
        TVM_FFI_THROW(InternalError)
            << "Bundled primitive parameters must be materialized in the function prologue";
      }
      auto field_type = var_to_field_type_.find(var);
      TVM_FFI_ICHECK(field_type != var_to_field_type_.end());
      Type rebound_type = VisitExprDepTypeField(GetType(var));
      Var replacement = (*field_type).second.same_as(rebound_type)
                            ? builder_->Emit((*it).second, op->name)
                            : builder_->EmitMatchCast((*it).second, rebound_type, op->name);
      return replacement;
    }
    return ExprMutator::VisitExpr_(op);
  }

 private:
  ffi::Optional<ffi::String> param_tuple_name_;
  ffi::Map<Var, Expr> var_to_expr_;
  ffi::Map<Var, Type> var_to_field_type_;
  ffi::Array<Var> pending_prim_params_;
  bool rewrite_model_params_{false};
};

Function BundleModelParams(const Function& func, ffi::Optional<ffi::String> param_tuple_name) {
  ModelParamBundler mutator(param_tuple_name);
  return mutator(func).as_or_throw<Function>();
}

namespace transform {
Pass BundleModelParams(ffi::Optional<ffi::String> param_tuple_name) {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    IRModule updates;

    ModelParamBundler mutator(param_tuple_name);

    for (const auto& [gvar, func] : mod->functions) {
      if (auto opt = func.as<relax::Function>()) {
        auto new_func = mutator(opt.value()).as_or_throw<relax::Function>();
        if (!new_func.same_as(func)) {
          updates->Add(gvar, new_func);
        }
      }
    }

    if (updates->functions.size()) {
      mod.CopyOnWrite()->Update(updates);
    }
    return mod;
  };
  return CreateModulePass(pass_func, 1, "BundleModelParams");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.BundleModelParams", BundleModelParams);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
