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
 * \file src/relax/ir/binding_rewrite.cc
 * \brief Implementation of binding rewriters.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/binding_rewrite.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>

#include <functional>
#include <iterator>
#include <set>
#include <vector>

#include "../../support/ordered_set.h"

namespace tvm {
namespace relax {

struct VarIdentityLess {
  bool operator()(const Var& lhs, const Var& rhs) const { return lhs.get() < rhs.get(); }
};

using VarIdentitySet = std::set<Var, VarIdentityLess>;

TVM_FFI_STATIC_INIT_BLOCK() { DataflowBlockRewriteObj::RegisterReflection(); }

DataflowBlockRewrite::DataflowBlockRewrite(DataflowBlock dfb, Function root_fn) {
  auto n = ffi::make_object<DataflowBlockRewriteObj>(dfb);
  n->root_fn_ = root_fn;
  n->original_fn_ptr_ = root_fn.get();
  auto p = FunctionUseDef(root_fn);
  n->to_users_ = std::move(p.first);
  n->fn_outputs_ = std::move(p.second);
  n->name_supply_ = UniqueNameSupply(n->to_users_.begin(), n->to_users_.end(),
                                     [](const auto& p) { return p.first->name; });

  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.DataflowBlockRewrite", [](DataflowBlock dfb, Function root_fn) {
    return DataflowBlockRewrite(dfb, root_fn);
  });
}

void DataflowBlockRewriteObj::ReplaceAllUses(Var old_var, Var new_var) {
  class ReplaceAllUsePass : public ExprMutator {
    Var old_var, new_var;
    const DataflowBlockObj* const to_catch;

   public:
    ffi::Optional<DataflowBlock> caught;

    ReplaceAllUsePass(Var old_var, Var new_var, const DataflowBlockObj* to_catch)
        : old_var(old_var), new_var(new_var), to_catch(to_catch) {}

    using ExprMutator::VisitExpr_;

    Expr VisitExpr_(const VarObj* op) override {
      return (op == old_var.get()) ? new_var : ffi::GetRef<Expr>(op);
    }

    BindingBlock VisitBindingBlock_(const DataflowBlockObj* op) override {
      BindingBlock res = ExprMutator::VisitBindingBlock_(op);
      if (op == to_catch) caught = res.as_or_throw<DataflowBlock>();
      return res;
    }
  };

  TVM_FFI_ICHECK(to_users_.find(old_var) != to_users_.end()) << "Cannot find " << old_var;
  TVM_FFI_ICHECK(to_users_.find(new_var) != to_users_.end()) << "Cannot find " << new_var;
  if (old_var.same_as(new_var)) return;

  // replace uses inside the DataflowBlock.
  ReplaceAllUsePass replacer(old_var, new_var, dfb_.get());
  if (root_fn_) {
    root_fn_ = replacer.VisitExpr(root_fn_.value()).as_or_throw<Function>();
    dfb_ = replacer.caught.value();
  } else {
    dfb_ = replacer.VisitBindingBlock(dfb_).as_or_throw<DataflowBlock>();
  }

  // update udchain
  // old_var -> old_var users | changed to {}
  // new_var -> {?}           | changed to old_var users
  auto new_var_uses = to_users_[new_var];
  for (Var user : to_users_[old_var]) {
    if (new_var_uses.end() ==
        std::find_if(new_var_uses.begin(), new_var_uses.end(),
                     [&](const Var& candidate) { return candidate.same_as(user); })) {
      new_var_uses.push_back(user);
    }
  }

  to_users_.Set(new_var, new_var_uses);
  to_users_.Set(old_var, {});

  auto it_old_output =
      std::find_if(fn_outputs_.begin(), fn_outputs_.end(),
                   [&](const Var& candidate) { return candidate.same_as(old_var); });
  if (it_old_output != fn_outputs_.end()) {
    fn_outputs_.Set(std::distance(fn_outputs_.begin(), it_old_output), new_var);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dfb_rewrite_replace_all_uses",
                        [](DataflowBlockRewrite rwt, Var old_var, Var new_var) {
                          rwt->ReplaceAllUses(old_var, new_var);
                        });
}

class UpdateDFB : public ExprMutator {
 private:
  DataflowBlock old_dfb, new_dfb;

 public:
  UpdateDFB(DataflowBlock old_dfb, DataflowBlock new_dfb)
      : old_dfb(std::move(old_dfb)), new_dfb(std::move(new_dfb)) {}

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* op) override {
    return old_dfb.get() == op ? new_dfb : old_dfb;
  }
};

void DataflowBlockRewriteObj::Add(Binding binding) {
  auto [var, val] = [binding] {
    if (auto vb = binding.as<VarBindingObj>()) {
      return std::make_pair(vb->var, vb->value);
    } else if (auto mc = binding.as<MatchCastObj>()) {
      return std::make_pair(mc->var, mc->value);
    }
    TVM_FFI_THROW(InternalError) << "Unsupported binding type";
    throw;
  }();

  TVM_FFI_ICHECK(0 == to_users_.count(var)) << var << " has been defined so cannot be added.";

  // Add this VarBinding statement after the definition of uses.
  auto used_vars = GetUsedVars(val);

  size_t line_last_req_def = 0;
  for (size_t i = 0; i < dfb_->bindings.size(); ++i) {
    auto line = dfb_->bindings[i];
    if (used_vars.find(line->var.get()) != used_vars.cend()) line_last_req_def = i;
  }

  auto old_dfb = dfb_;

  dfb_.CopyOnWrite()->bindings.insert(dfb_->bindings.begin() + 1 + line_last_req_def, binding);

  if (root_fn_) {
    auto updater = UpdateDFB(old_dfb, dfb_);
    root_fn_ = updater.VisitExpr(root_fn_.value()).as_or_throw<Function>();
  }

  for (const VarObj* v : used_vars) {
    auto var = ffi::GetRef<Var>(v);
    if (auto users = to_users_.Get(var)) {
      users.value().push_back(var);
    }
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.dfb_rewrite_add_binding",
           [](DataflowBlockRewrite rwt, Binding vb) { rwt->Add(vb); })
      .def("relax.dfb_rewrite_add",
           [](DataflowBlockRewrite rwt, Expr expr, ffi::Optional<ffi::String> name, bool is_dfvar) {
             if (name.has_value()) {
               rwt->Add(name.value(), expr, is_dfvar);
             } else {
               rwt->Add(expr, is_dfvar);
             }
           });
}

VarIdentitySet GetUnusedVars(ffi::Map<Var, ffi::Array<Var>> users_map, ffi::Array<Var> fn_outputs) {
  std::vector<Var> unused;

  // iterative dataflow algorithm.
  size_t prev_size;
  do {
    prev_size = unused.size();

    std::vector<Var> used;
    used.reserve(users_map.size());
    for (const auto& [def, users] : users_map) {
      // var -> [users...]
      // var is unused iff
      //   user -> empty
      //   var is not output var
      if (users.empty() &&  // def is not used by fn outputs.
          std::find_if(fn_outputs.begin(), fn_outputs.end(), [&](const Var& output) {
            return output.same_as(def);
          }) == fn_outputs.end()) {
        unused.push_back(def);
      } else {
        used.push_back(def);
      }
    }

    for (size_t i = prev_size; i < unused.size(); ++i) {
      users_map.erase(unused[i]);
      // remove def site.
      for (const auto& used_var : used) {
        TVM_FFI_ICHECK(users_map.count(used_var));
        ffi::Array<Var> var_users = users_map[used_var];
        // remove the unused var from the use site.
        if (auto it = std::find_if(var_users.begin(), var_users.end(),
                                   [&](const Var& user) { return user.same_as(unused[i]); });
            it != var_users.end()) {
          var_users.erase(it);
          users_map.Set(used_var, std::move(var_users));
        }
      }
    }
  } while (prev_size != unused.size());  // changed? => continue.

  return VarIdentitySet(unused.begin(), unused.end());
}

class RemoveUnusedVars : public ExprMutator {
 public:
  VarIdentitySet unused_vars;
  ffi::Optional<DataflowBlock> caught_rewrite = std::nullopt;

  explicit RemoveUnusedVars(VarIdentitySet unused_vars) : unused_vars(std::move(unused_vars)) {}

  RemoveUnusedVars(ffi::Map<Var, ffi::Array<Var>> users, ffi::Array<Var> fn_outputs)
      : RemoveUnusedVars(GetUnusedVars(users, fn_outputs)) {}

  void VisitBinding_(const VarBindingObj* binding) override {
    bool can_remove = unused_vars.count(binding->var) &&
                      (in_dataflow_block_ || !ContainsImpureCall(binding->value));
    if (!can_remove) {
      ExprMutator::VisitBinding_(binding);
    }
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockObj* block) override {
    bool capture_output = caught_rewrite && block == caught_rewrite.value().get();

    bool cache = in_dataflow_block_;
    in_dataflow_block_ = true;
    BindingBlock output = ExprMutator::VisitBindingBlock_(block);
    in_dataflow_block_ = cache;

    if (capture_output) {
      caught_rewrite = output.as_or_throw<DataflowBlock>();
    }

    return output;
  }

 private:
  bool in_dataflow_block_{false};
};

void DataflowBlockRewriteObj::RemoveUnused(Var unused, bool allow_undef) {
  // first need to check if this var is used.
  if (to_users_.count(unused) == 0) {  // no def.
    if (allow_undef) return;
    TVM_FFI_THROW(InternalError)
        << unused << " undefined. Set allow_undef=True to allow 'removing' undefined var";
  }

  TVM_FFI_ICHECK(to_users_[unused].empty())
      << unused << " is used by " << to_users_[unused].size() << " vars";

  auto old_dfb = dfb_;

  RemoveUnusedVars remover({unused});
  dfb_ = remover.VisitBindingBlock(old_dfb).as_or_throw<DataflowBlock>();

  if (root_fn_) {
    auto updater = UpdateDFB(old_dfb, dfb_);
    root_fn_ = updater.VisitExpr(root_fn_.value()).as_or_throw<Function>();
  }

  to_users_.erase(unused);  // update use-def chain.
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dfb_rewrite_remove_unused",
                        [](DataflowBlockRewrite rwt, Var unused, bool allow_undef) {
                          rwt->RemoveUnused(unused, allow_undef);
                        });
}

void DataflowBlockRewriteObj::RemoveAllUnused() {
  RemoveUnusedVars remover(to_users_, fn_outputs_);
  remover.caught_rewrite = dfb_;

  if (root_fn_) {
    // this could also clean unused variables in other DataflowBlock.
    root_fn_ = remover.VisitExpr(root_fn_.value()).as_or_throw<Function>();
    // DataflowBlock could be None.
    dfb_ = remover.caught_rewrite.value();
  } else {
    dfb_ = remover.VisitBindingBlock(dfb_).as_or_throw<DataflowBlock>();
  }

  // clean up use-def chain.
  for (const auto& unused : remover.unused_vars) to_users_.erase(unused);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.dfb_rewrite_remove_all_unused",
                        [](DataflowBlockRewrite rwt) { rwt->RemoveAllUnused(); });
}

Expr RemoveAllUnused(Expr expr) {
  auto var_usage = CollectVarUsage(expr);

  // For the purpose of
  support::OrderedSet<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> externally_exposed(
      var_usage.outputs.begin(), var_usage.outputs.end());
  for (const auto& [var, expr] : var_usage.bound_values) {
    if (ContainsImpureCall(expr)) {
      externally_exposed.insert(var);
    }
  }

  RemoveUnusedVars remover(var_usage.downstream_usage,
                           ffi::Array<Var>(externally_exposed.begin(), externally_exposed.end()));
  return remover.VisitExpr(std::move(expr));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.analysis.remove_all_unused", RemoveAllUnused);
}

IRModule DataflowBlockRewriteObj::MutateIRModule(IRModule irmod) {
  BlockBuilder builder = BlockBuilder::Create(irmod);

  for (auto& [gvar, fn] : irmod->functions) {
    if (root_fn_ && original_fn_ptr_ == fn.get()) {
      builder->UpdateFunction(gvar, root_fn_.value());
      break;
    }
  }

  return builder->GetContextIRModule();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.dfb_rewrite_mutate_irmodule",
      [](DataflowBlockRewrite rwt, IRModule irmod) { return rwt->MutateIRModule(irmod); });
}

}  // namespace relax
}  // namespace tvm
