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
 * \file src/relax/transform/allocate_workspace.cc
 * \brief Allocate a workspace and append it to the arguments of external functions, to
 * satisfy their temporary storage requirement.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/unique_name_supply.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../op/op_common.h"

namespace tvm {
namespace relax {

class ExternFunctionRewriter : ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  ExternFunctionRewriter(IRModule mod, size_t max_workspace_size)
      : ExprMutator(mod), max_workspace_size_(max_workspace_size) {}

  std::unordered_map<const GlobalVarObj*, Function> Run() {
    std::unordered_map<const GlobalVarObj*, Function> ret;
    for (const auto& [gvar, f] : builder_->GetContextIRModule()->functions) {
      if (f->GetAttr(tvm::relax::attr::kWorkspaceSize)) {
        ret.insert_or_assign(gvar.get(), VisitExpr(f).as_or_throw<Function>());
      }
    }
    return ret;
  }

  Expr VisitExpr_(const FunctionObj* func_node) override {
    if (!func_node->GetAttr(tvm::relax::attr::kCodegen) &&
        !func_node->GetAttr(tvm::relax::attr::kComposite)) {
      return ExprMutator::VisitExpr_(func_node);
    }
    if (auto workspace = func_node->GetAttr(tvm::relax::attr::kWorkspaceSize)) {
      // Append the workspace parameter to this function.
      ffi::Array<Var> new_params = func_node->params;

      auto ty = TensorType(ShapeExpr({IntImm::Int32(max_workspace_size_)}), PrimType::UInt(8));
      Var workspace_param(name_sup_->FreshName("workspace"), ty);

      if (func_node->GetAttr(tvm::relax::attr::kCodegen)) {
        workspace_var_param_ = workspace_param;
      }

      new_params.push_back(workspace_param);
      auto new_attrs = func_node->attrs;
      new_attrs.CopyOnWrite()->dict.erase(tvm::relax::attr::kWorkspaceSize.name);
      return Function(new_params, VisitExpr(func_node->body), func_node->ret_ty, func_node->is_pure,
                      new_attrs);
    }
    return ExprMutator::VisitExpr_(func_node);
  }

  Expr VisitExpr_(const CallObj* call_node) override {
    auto new_op = VisitExpr(call_node->op);
    if (auto var = new_op.as<Var>()) {
      if (auto callee = builder_->LookupBinding(var.value());
          callee && callee.value()->IsInstance<FunctionObj>() &&
          callee.value().as_or_throw<Function>()->GetAttr<ffi::String>(
              tvm::relax::attr::kComposite.name)) {
        // Append the workspace argument to this call. The callee should have been updated to accept
        // a workspace as the last parameter.
        auto new_args = call_node->args;
        TVM_FFI_ICHECK(workspace_var_param_.has_value());
        new_args.push_back(workspace_var_param_.value());
        return Call(Type::Missing(), new_op, new_args, call_node->attrs, call_node->ty_args,
                    call_node->loc);
      }
    }
    return ExprMutator::VisitExpr_(call_node);
  }

 private:
  UniqueNameSupply name_sup_;
  /*! \brief A variable that represents the workspace parameter passed from main. */
  ffi::Optional<Var> workspace_var_param_;
  size_t max_workspace_size_ = 0;
};

class WorkspaceProvider : ExprMutator {
 public:
  explicit WorkspaceProvider(IRModule mod) : ExprMutator(mod), mod_(mod) {}
  using ExprMutator::VisitBindingBlock_;
  using ExprMutator::VisitExpr_;

  IRModule Run() {
    for (const auto& [gvar, f] : mod_->functions) {
      if (auto workspace = f->GetAttr(tvm::relax::attr::kWorkspaceSize)) {
        max_workspace_size_ = std::max<size_t>(max_workspace_size_, workspace.value());
      }
    }

    if (max_workspace_size_ == 0) {
      return mod_;
    }

    auto new_funcs = relax::ExternFunctionRewriter(mod_, max_workspace_size_).Run();

    for (const auto& [gvar, f] : new_funcs) {
      auto new_gvar = builder_->AddFunction(f, gvar->name_hint);
      // This is only required since the well-formed check requires kGlobalSymbol to be the same
      // as the actual name of the global variable.
      builder_->UpdateFunction(new_gvar,
                               WithAttr(f, tvm::attr::kGlobalSymbol.name, new_gvar->name_hint));
      gvar_map_.insert_or_assign(gvar, new_gvar);
      new_gvars_.insert(new_gvar);
      builder_->GetContextIRModule()->Remove(ffi::GetRef<GlobalVar>(gvar));
    }

    for (const auto& [gvar, f] : mod_->functions) {
      workspace_var_main_ = std::nullopt;
      if (!f->IsInstance<relax::FunctionObj>() || f->GetAttr(tvm::relax::attr::kCodegen) ||
          f->GetAttr(tvm::relax::attr::kComposite)) {
        continue;
      }
      auto func = mod_->Lookup(gvar).as_or_throw<Function>();
      auto new_func =
          Function(func->params, VisitExpr(func->body), func->ret_ty, func->is_pure, func->attrs);
      builder_->UpdateFunction(gvar, new_func);
    }
    return builder_->GetContextIRModule();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block_node) final {
    builder_->BeginDataflowBlock();
    if (!workspace_var_main_.has_value()) {
      auto shape = ShapeExpr({IntImm::Int32(max_workspace_size_)});
      auto ty = DataTypeImm((DLDataType{kDLUInt, 8, 1}));
      auto workspace = MakeAllocTensor(shape, ty, IntImm::Int64(0));
      workspace_var_main_ = builder_->Emit(workspace, "workspace_main");
    }
    for (const auto& binding : block_node->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  Expr VisitExpr_(const GlobalVarObj* gvar_node) override {
    if (gvar_map_.count(gvar_node)) {
      return gvar_map_.at(gvar_node);
    }
    return ExprMutator::VisitExpr_(gvar_node);
  }

  Expr VisitExpr_(const CallObj* call_node) override {
    auto new_op = VisitExpr(call_node->op);

    if (auto gv = new_op.as<GlobalVar>()) {
      if (new_gvars_.count(gv.value())) {
        auto new_args = call_node->args;
        TVM_FFI_ICHECK(workspace_var_main_.has_value());
        new_args.push_back(workspace_var_main_.value());
        return Call(Type::Missing(), new_op, new_args, call_node->attrs, call_node->ty_args,
                    call_node->loc);
      }
    }

    return ExprMutator::VisitExpr_(call_node);
  }

 private:
  IRModule mod_;
  /*! \brief A variable that represents the workspace created at the beginning of main. */
  ffi::Optional<Var> workspace_var_main_;
  size_t max_workspace_size_ = 0;
  /*! \brief A map from old global variables representing a function with workspace requirement to
   * the new ones that are transformed to take an additional workspace parameter. This is only
   * needed since the type of the global variables changes between transformation. */
  std::unordered_map<const GlobalVarObj*, GlobalVar> gvar_map_;
  std::unordered_set<GlobalVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> new_gvars_;
};

}  // namespace relax

namespace transform {

Pass AllocateWorkspace() {
  auto pass_func = [=](IRModule m, PassContext pc) { return relax::WorkspaceProvider(m).Run(); };

  return CreateModulePass(pass_func, 0, "AllocateWorkspace");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AllocateWorkspace", AllocateWorkspace);
}

}  // namespace transform
}  // namespace tvm
