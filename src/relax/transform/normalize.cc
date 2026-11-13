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
 * \file tvm/relax/transform/normalize.cc
 * \brief Pass for transforming Relax IR to normal form, i.e., the expressions are normalized(no
 * nesting and hence the AST is in ANF), and all ty of expressions are
 * available.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>

namespace tvm {
namespace relax {

// TODO(@altanh): LCA binding lifting
class NormalizeMutator : public ExprMutatorBase {
 public:
  NormalizeMutator() : builder_(BlockBuilder::Create(std::nullopt)) {}

  Expr VisitExpr(const Expr& expr) override {
    return builder_->Normalize(ExprMutatorBase::VisitExpr(expr));
  }

  Expr VisitExpr_(const FunctionObj* op) final {
    Expr body = this->VisitWithNewScope(op->body, op->params);

    if (body.same_as(op->body)) {
      return ffi::GetRef<Expr>(op);
    } else {
      return Function(op->params, body, op->ret_ty, op->is_pure, op->attrs);
    }
  }

  Expr VisitExpr_(const IfExprObj* op) final {
    Expr guard = this->VisitExpr(op->cond);
    Expr true_b = this->VisitWithNewScope(op->true_branch);
    Expr false_b = this->VisitWithNewScope(op->false_branch);
    if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
        op->false_branch.same_as(false_b)) {
      return ffi::GetRef<Expr>(op);
    } else {
      return IfExpr(guard, true_b, false_b, op->loc);
    }
  }

  Expr VisitWithNewScope(const Expr& expr, ffi::Optional<ffi::Array<Var>> params = std::nullopt) {
    builder_->BeginBindingBlock();
    if (params.has_value()) {
      builder_->BeginScope(params);
    } else {
      builder_->BeginInnerScope();
    }
    Expr ret = this->VisitExpr(expr);
    BindingBlock prologue = builder_->EndBlock();
    if (!prologue->bindings.empty()) {
      ret = SeqExpr({prologue}, ret);
    }
    builder_->EndScope();
    return ret;
  }

  Expr VisitExpr_(const SeqExprObj* op) final {
    bool all_blocks_unchanged = true;
    ffi::Array<BindingBlock> blocks;
    for (auto block : op->blocks) {
      BindingBlock new_block = this->VisitBindingBlock(block);
      if (!new_block->bindings.empty()) {
        blocks.push_back(new_block);
      }
      all_blocks_unchanged &= block.same_as(new_block);
    }

    builder_->BeginBindingBlock();
    Expr body = this->VisitExpr(op->body);
    BindingBlock prologue = builder_->EndBlock();
    if (!prologue->bindings.empty()) {
      blocks.push_back(prologue);
      all_blocks_unchanged = false;
    }

    if (all_blocks_unchanged && body.same_as(op->body)) {
      return ffi::GetRef<Expr>(op);
    } else {
      return SeqExpr(blocks, body);
    }
  }

  BindingBlock VisitBindingBlock(const BindingBlock& block) final {
    BindingBlock ret(ffi::UnsafeInit{});
    if (const auto* node = block.as<DataflowBlockObj>()) {
      ret = VisitBindingBlock_(node);
    } else if (const auto* node = block.as<BindingBlockObj>()) {
      ret = VisitBindingBlock_(node);
    } else {
      TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
    }
    return ret;
  }

  BindingBlock VisitBindingBlock_(const BindingBlockObj* block) {
    builder_->BeginBindingBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block) {
    builder_->BeginDataflowBlock();
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    return builder_->EndBlock();
  }

  void VisitBinding(const Binding& binding) {
    if (const auto* node = binding.as<VarBindingObj>()) {
      VisitBinding_(node);
    } else if (const auto* node = binding.as<MatchCastObj>()) {
      VisitBinding_(node);
    } else {
      TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
    }
  }

  void VisitBinding_(const VarBindingObj* binding) {
    Expr new_value = this->VisitExpr(binding->value);
    if (binding->var->ty.as<MissingType>().has_value()) {
      UpdateType(binding->var, GetType(new_value));
    }

    if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(ffi::GetRef<VarBinding>(binding));
    } else {
      builder_->EmitNormalized(VarBinding(binding->var, new_value));
    }
  }

  void VisitBinding_(const MatchCastObj* binding) {
    Expr new_value = this->VisitExpr(binding->value);

    if (new_value.same_as(binding->value)) {
      builder_->EmitNormalized(ffi::GetRef<MatchCast>(binding));
    } else {
      builder_->EmitNormalized(
          MatchCast(binding->var, builder_->NormalizeArgument(new_value), binding->ty));
    }
  }

 private:
  /*! \brief Internal block builder to emit bindings during rewriting. */
  BlockBuilder builder_;
};  // namespace relax

Expr Normalize(const Expr& e) { return NormalizeMutator().VisitExpr(e); }

class GlobalVarNormalizer : private ExprMutator {
 public:
  static IRModule Normalize(const IRModule& m) {
    GlobalVarNormalizer renamer(m);
    return renamer.RenameModule();
  }

 private:
  explicit GlobalVarNormalizer(const IRModule& m) : ExprMutator(), module_(m) {}

  using ExprMutator::VisitExpr_;

  IRModule RenameModule() {
    if (!NeedRename()) {
      return std::move(module_);
    }

    // Step 1. Add public functions (functions with global_symbol attributes)
    AddPublicFunctions();

    // Step 2. Rename private functions
    AddPrivateFunctions();

    // Step 3. Substitute global vars in functions
    for (auto [gvar, func] : module_->functions) {
      if (!func->IsInstance<FunctionObj>()) {
        continue;
      }
      auto new_func = this->VisitExpr(func).as_or_throw<BaseFunc>();
      builder_->UpdateFunction(gvar_map_[gvar], new_func);
    }

    // Step 4. Update the original module (because we do not want to copy all metadata to the new
    // module)
    auto after_module = builder_->GetContextIRModule();
    auto module_node = module_.CopyOnWrite();
    module_node->functions = after_module->functions;
    module_node->global_var_map_ = after_module->global_var_map_;
    return std::move(module_);
  }

  /*! \brief Check if any function needs to be renamed. */
  bool NeedRename() {
    for (const auto& [gvar, func] : module_->functions) {
      auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
      if (global_symbol && global_symbol.value() != gvar->name_hint) {
        return true;
      }
    }
    return false;
  }

  /*! \brief Add public functions to the builder, and update the name supplier. */
  void AddPublicFunctions() {
    for (const auto& [gvar, func] : module_->functions) {
      auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
      if (!global_symbol) {
        continue;
      }

      auto global_symbol_value = global_symbol.value();
      TVM_FFI_ICHECK(!name_supply_->ContainsName(global_symbol_value))
          << "IRModule contains duplicate global symbol: " << global_symbol_value;
      name_supply_->ReserveName(global_symbol_value);
      auto new_gvar = builder_->AddFunction(func, global_symbol_value);
      gvar_map_.Set(gvar, new_gvar);
    }
  }

  /*!
   * \brief Add private functions to the builder with names provided by name supplier. Renaming may
   * happen if the name of any function conflicts with the name of a public function.
   */
  void AddPrivateFunctions() {
    for (auto [gvar, func] : module_->functions) {
      auto global_symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
      if (global_symbol) {
        continue;
      }

      auto new_name = name_supply_->FreshName(gvar->name_hint, false, false);
      auto new_gvar = builder_->AddFunction(func, new_name);
      gvar_map_.Set(gvar, new_gvar);
    }
  }

  Expr VisitExpr_(const GlobalVarObj* op) final {
    TVM_FFI_ICHECK(gvar_map_.count(ffi::GetRef<GlobalVar>(op)));
    return gvar_map_[ffi::GetRef<GlobalVar>(op)];
  }

  IRModule module_;
  UniqueNameSupply name_supply_;
  ffi::Map<GlobalVar, GlobalVar> gvar_map_;
};

namespace transform {

Pass Normalize() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return Normalize(f).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, 1, "Normalize");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.Normalize", Normalize);
}

Pass NormalizeGlobalVar() {
  auto pass_func = [=](IRModule mod, PassContext pc) {
    return GlobalVarNormalizer::Normalize(mod);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"NormalizeGlobalVar");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.NormalizeGlobalVar", NormalizeGlobalVar);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
