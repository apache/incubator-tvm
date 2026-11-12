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
 * \file src/relax/transform/merge_composite_functions.cc
 * \brief Group one or multiple composite functions created by FuseOpsByPattern into a new
 * function.
 *
 * The new function will be annotated with kCodegen and kGlobalSymbol attributes, and it is
 * intented to be offloaded to an external backend.
 *
 * A group for one composite function can be merged into another group for one of its arguments,
 * which we call the parent group for brevity, if the following conditions are met:
 * - The argument is the result of calling a composite function offloaded to the same backend
 * - Merging into the parent group would not create a cyclic dependency with other parent groups
 *
 * For example, in the subgraph below the bottom group cannot be merged into the left parent group,
 * since the right parent group for X depends on an output from the left parent group.
 *
 *  O = Offloaded to A
 *  X = Offloaded to B
 *
 * Correct partitioning:
 *
 *     O         O
 *    / \       /	            \
 *   O   X --> O    +     +    X
 *    \ /             \ /
 *     O               O
 *
 * The algorithm proceeds by assigning a group to each subexpression in the function according to
 * its dataflow. On encountering a call node whose callee is a composite function, we check the
 * two conditions above to see if we can merge this call node into one of its parent groups, and
 * if we can merge some of its parent groups.
 *
 * To detect cyclic dependencies between groups, we propagate dependency relations, both direct
 * and indirect ones, as we flow through the function. The propagation of indirect dependencies
 * is important since the dependency relation is transitive.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/type.h>
#include <tvm/tirx/function.h>

#include "../../support/arena.h"
#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace {

using Group = GraphPartitioner::Group;

/*! \brief Assign group to each subexpression in a function according to its
 * dataflow, and returns a mapping from a subexpression to its group. */
class CompositeGroupsBuilder : public MemoizedExprTranslator<Group*> {
 public:
  using GroupMap = std::unordered_map<const ffi::Object*, Group*>;
  using MemoizedExprTranslator<Group*>::VisitExpr_;

  CompositeGroupsBuilder(IRModule mod, support::Arena* arena) : mod_(mod), arena_(arena) {}

  GroupMap Run(Function func) {
    var_usage_ = CollectVarUsage(func);
    for (const auto& [var, value] : var_usage_.bound_values) {
      value_to_bound_vars_[value.get()].push_back(var);
    }

    for (const auto& param : func->params) {
      memo_[param] = arena_->make<Group>();
    }

    PostOrderVisit(func, [this](const Expr& expr) {
      if (expr->IsInstance<GenericConstObj>() || expr->IsInstance<ShapeExprObj>() ||
          (!expr->IsInstance<CallObj>() && !expr->IsInstance<VarObj>() && expr.as<PrimExpr>())) {
        memo_[expr] = arena_->make<Group>();
      }
    });

    VisitExpr(func->body);

    GroupMap group_map;
    for (const auto& [expr, group] : memo_) {
      group_map[expr.get()] = group->FindRoot();
    }

    return group_map;
  }

  Group* VisitBinding(const Binding& binding) {
    if (const auto* node = binding.as<VarBindingObj>()) {
      return VisitBinding_(node);
    } else {
      TVM_FFI_THROW(TypeError) << "Invalid type: " << binding->GetTypeKey();
    }
  }

  void VisitBindingBlock_(const BindingBlockObj* block) {
    for (Binding binding : block->bindings) {
      VisitBinding(binding);
    }
  }

  void VisitBindingBlock_(const DataflowBlockObj* block) {
    for (Binding binding : block->bindings) {
      VisitBinding(binding);
    }
  }

  void VisitBindingBlock(const BindingBlock& block) {
    if (const auto* node = block.as<DataflowBlockObj>()) {
      VisitBindingBlock_(node);
    } else if (const auto* node = block.as<BindingBlockObj>()) {
      VisitBindingBlock_(node);
    } else {
      TVM_FFI_THROW(TypeError) << "Invalid type: " << block->GetTypeKey();
    }
  }

  Group* VisitExpr_(const SeqExprObj* op) {
    for (BindingBlock block : op->blocks) {
      VisitBindingBlock(block);
    }
    return VisitExpr(op->body);
  }

  Group* VisitExpr_(const CallObj* call) {
    for (const Expr& arg : call->args) {
      EnsureVisited(arg);
    }
    std::vector<Group*> groups_to_merge = GetGroupsToMerge(call);
    Group* group;

    if (groups_to_merge.size() == 0) {
      // Create new group if there is nothing to merge with
      group = CreateNewGroup(call);
    } else {
      auto it = groups_to_merge.cbegin();
      // Assign the first mergable group to current node
      // to reduce the number of groups created
      group = *it++;
      group->num_nodes += 1;

      // Merge all groups
      for (; it != groups_to_merge.cend(); ++it) {
        MergeGroup(*it, group);
      }
    }

    UpdateGroupDependencies(group, call->args);
    return group;
  }

  Group* VisitExpr_(const TupleObj* tuple) {
    Expr tuple_expr = ffi::GetRef<Tuple>(tuple);
    if (!IsFlatTensorTuple(tuple_expr)) return arena_->make<Group>();

    for (const Expr& field : tuple->fields) {
      EnsureVisited(field);
    }
    if (HasOnlyTupleGetItemUsers(tuple_expr)) {
      Group* tuple_group = nullptr;
      for (const Expr& field : tuple->fields) {
        auto it = memo_.find(field);
        if (it == memo_.end()) {
          tuple_group = nullptr;
          break;
        }
        Group* field_group = it->second->FindRoot();
        if (tuple_group == nullptr) {
          tuple_group = field_group;
        } else if (tuple_group != field_group) {
          tuple_group = nullptr;
          break;
        }
      }
      if (tuple_group != nullptr && CanAbsorbTupleNodes(tuple_group)) {
        tuple_group->num_nodes += 1;
        return tuple_group;
      }
    }

    Group* group = arena_->make<Group>();
    UpdateGroupDependencies(group, tuple->fields);
    return group;
  }

  Group* VisitExpr_(const TupleGetItemObj* tuple_get_item) {
    if (!IsFlatTensorTuple(tuple_get_item->tuple)) return arena_->make<Group>();

    EnsureVisited(tuple_get_item->tuple);
    auto it = memo_.find(tuple_get_item->tuple);
    if (it != memo_.end() && HasOnlyTupleGetItemUsers(tuple_get_item->tuple)) {
      Group* tuple_group = it->second->FindRoot();
      if (CanAbsorbTupleNodes(tuple_group)) {
        tuple_group->num_nodes += 1;
        return tuple_group;
      }
    }

    Group* group = arena_->make<Group>();
    UpdateGroupDependencies(group, {tuple_get_item->tuple});
    return group;
  }

 private:
  void EnsureVisited(const Expr& expr) {
    if (!expr.as<GlobalVarObj>() && !memo_.count(expr)) {
      VisitExpr(expr);
    }
  }

  ffi::Optional<ffi::String> GetCodegenName(const Expr& callee) {
    auto const* gvar = callee.as<GlobalVarObj>();
    if (!gvar) {
      return std::nullopt;
    }

    auto composite_name_opt =
        mod_->Lookup(ffi::GetRef<GlobalVar>(gvar))->GetAttr(tvm::relax::attr::kComposite);
    if (!composite_name_opt) {
      return std::nullopt;
    }

    return relax::GetCodegenName(composite_name_opt.value());
  }

  ffi::Optional<ffi::String> GetCodegenName(Group* group) {
    if (auto opt_str = group->attrs.Get(tvm::relax::attr::kCodegen.name)) {
      return opt_str.value().as_or_throw<ffi::String>();
    }
    return std::nullopt;
  }

  bool CanAbsorbTupleNodes(Group* group) { return GetCodegenName(group->FindRoot()).has_value(); }

  bool IsFlatTensorTuple(const Expr& expr) {
    const auto* tuple_type = GetType(expr).as<TupleTypeObj>();
    if (tuple_type == nullptr || tuple_type->fields.empty()) return false;
    return std::all_of(tuple_type->fields.begin(), tuple_type->fields.end(),
                       [](const Type& field) { return field->IsInstance<TensorTypeObj>(); });
  }

  bool HasOnlyTupleGetItemUsers(const Expr& tuple_expr) {
    if (auto it = tuple_get_item_only_usage_.find(tuple_expr.get());
        it != tuple_get_item_only_usage_.end()) {
      return it->second;
    }

    bool result = ComputeHasOnlyTupleGetItemUsers(tuple_expr);
    tuple_get_item_only_usage_[tuple_expr.get()] = result;
    return result;
  }

  bool ComputeHasOnlyTupleGetItemUsers(const Expr& tuple_expr) {
    ffi::Optional<Var> tuple_var;
    if (const auto* var = tuple_expr.as<VarObj>()) {
      tuple_var = ffi::GetRef<Var>(var);
    } else if (auto it = value_to_bound_vars_.find(tuple_expr.get());
               it != value_to_bound_vars_.end() && it->second.size() == 1) {
      tuple_var = it->second[0];
    }
    if (!tuple_var.has_value()) return false;

    // Follow aliases in both directions.  Checking only tuple_var would allow an alias to be used
    // exclusively by TupleGetItem while the original tuple still escapes from the external region.
    std::vector<Var> pending{tuple_var.value()};
    std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> aliases;
    bool has_tuple_get_item = false;
    while (!pending.empty()) {
      Var current = pending.back();
      pending.pop_back();
      if (!aliases.insert(current).second) continue;

      if (std::any_of(var_usage_.outputs.begin(), var_usage_.outputs.end(),
                      [&](const Var& output) { return output.same_as(current); })) {
        return false;
      }

      // Walk toward the original tuple value when current is itself an alias.
      if (auto binding_it = var_usage_.bound_values.find(current);
          binding_it != var_usage_.bound_values.end()) {
        if (const auto* source = (*binding_it).second.as<VarObj>()) {
          pending.push_back(ffi::GetRef<Var>(source));
        }
      }

      auto uses_it = var_usage_.downstream_usage.find(current);
      if (uses_it == var_usage_.downstream_usage.end()) continue;
      for (const Var& user : (*uses_it).second) {
        auto binding_it = var_usage_.bound_values.find(user);
        if (binding_it == var_usage_.bound_values.end()) return false;
        const Expr& bound_value = (*binding_it).second;
        if (const auto* tuple_get_item = bound_value.as<TupleGetItemObj>()) {
          if (!tuple_get_item->tuple.same_as(current)) return false;
          has_tuple_get_item = true;
        } else if (const auto* source = bound_value.as<VarObj>()) {
          if (!ffi::GetRef<Var>(source).same_as(current)) return false;
          pending.push_back(user);
        } else {
          return false;
        }
      }
    }
    return has_tuple_get_item;
  }

  Group* CreateNewGroup(const CallObj* call) {
    Group* group = arena_->make<Group>();
    if (ffi::Optional<ffi::String> codegen_name = GetCodegenName(call->op)) {
      group->attrs.Set(tvm::relax::attr::kCodegen.name, codegen_name.value());
    }
    return group;
  }

  void MergeGroup(Group* from, Group* to) {
    TVM_FFI_ICHECK_EQ(GetCodegenName(from), GetCodegenName(to));

    Group* from_root = from->FindRoot();
    Group* to_root = to->FindRoot();
    if (from_root == to_root) {
      return;
    }

    from_root->parent = to_root;
    to_root->num_nodes += from_root->num_nodes;

    // Update the group_deps_, maintaining the invariant that
    // all groups in the map are root groups.
    group_deps_[to_root].merge(group_deps_[from_root]);
    group_deps_.erase(from_root);
    for (auto& it : group_deps_) {
      if (it.second.count(from_root)) {
        it.second.erase(from_root);
        it.second.insert(to_root);
      }
    }
  }

  std::unordered_set<Group*> GetParentGroupDependencies(const ffi::Array<Expr>& args) {
    // Collect groups that parent groups depend on
    std::unordered_set<Group*> dependencies;

    for (const auto& arg : args) {
      if (arg.as<GlobalVarObj>()) continue;
      for (auto dep : group_deps_[memo_[arg]->FindRoot()]) {
        dependencies.insert(dep);
      }
    }

    return dependencies;
  }

  void UpdateGroupDependencies(Group* group, const ffi::Array<Expr>& args) {
    Group* group_root = group->FindRoot();

    std::function<void(Expr)> visit_expr = [&](Expr expr) {
      if (expr.as<GlobalVarObj>()) return;
      if (auto tuple = expr.as<TupleObj>()) {
        for (const auto& field : tuple->fields) {
          visit_expr(field);
        }
        return;
      }

      TVM_FFI_ICHECK(memo_.count(expr))
          << "Could not find memo-ized group for expression of type " << expr->GetTypeKey();
      auto arg_group_root = memo_[expr]->FindRoot();

      if (arg_group_root == group_root) {
        // If arg and the current node are in the same group,
        // there is nothing to update.
        return;
      }

      // Add the group of arg as dependency
      group_deps_[group_root].insert(arg_group_root);
      // Propagate dependencies of arg
      for (auto dep : group_deps_[arg_group_root]) {
        group_deps_[group_root].insert(dep);
      }
    };

    for (const auto& arg : args) {
      visit_expr(arg);
    }
  }

  std::vector<Group*> GetGroupsToMerge(const CallObj* call) {
    ffi::Optional<ffi::String> codegen_name = GetCodegenName(call->op);
    if (!codegen_name.has_value()) {
      return {};
    }

    std::vector<Group*> groups_to_merge;
    std::unordered_set<Group*> parent_dependencies = GetParentGroupDependencies(call->args);

    for (const auto& arg : call->args) {
      if (arg.as<GlobalVarObj>()) continue;
      auto arg_group = memo_[arg];
      ffi::Optional<ffi::String> arg_codegen_name = GetCodegenName(arg_group);
      if (arg_codegen_name == codegen_name && !parent_dependencies.count(arg_group->FindRoot())) {
        // If there is a parent group with the same target, which none of the parent dependency
        // groups depends on, merging "this" call node into the parent group will not form a cyclic
        // dependency.
        groups_to_merge.push_back(arg_group);
      }
    }

    return groups_to_merge;
  }

  IRModule mod_;
  support::Arena* arena_;
  VarUsageInfo var_usage_;
  std::unordered_map<const ffi::Object*, std::vector<Var>> value_to_bound_vars_;
  std::unordered_map<const ffi::Object*, bool> tuple_get_item_only_usage_;
  // Map from group to its dependencies. All groups in this map, whether it's
  // the key or in value, should be root node (that is, group->parent == nullptr).
  std::unordered_map<Group*, std::unordered_set<Group*>> group_deps_;
};

/*! \brief Inline definitions of composite functions at the global level into their call sites.
  This is necessary to make functions created by MergeCompositeFunctions self-contained - each
  external backend compiler does not need to refer to the original containing module.
 */
class CompositeInliner : public ExprMutator {
 public:
  explicit CompositeInliner(IRModule mod) : ExprMutator(mod), mod_(mod) {}
  using ExprMutator::VisitExpr_;

  Function Run(Function func) {
    inlined_functions_ = ffi::Map<Function, Function>();
    auto new_body = VisitExpr(ToNonDataflow(func->body));
    auto new_func =
        Function(func->params, new_body, func->ret_ty, func->is_pure, func->attrs, func->loc);
    return new_func;
  }

  Expr VisitExpr_(const CallObj* call) {
    if (call->op->IsInstance<GlobalVarObj>()) {
      auto gvar = call->op.as_or_throw<GlobalVar>();
      auto func = mod_->Lookup(gvar).as_or_throw<Function>();
      if (func->GetAttr(tvm::relax::attr::kComposite)) {
        if (!inlined_functions_.count(func)) {
          auto new_func = CopyWithNewVars(func);
          new_func = WithoutAttr(new_func, tvm::relax::attr::kPrimitive.name);
          inlined_functions_.Set(func, new_func);
        }
        return Call(Type::Missing(), inlined_functions_[func], call->args);
      }
    }

    return ExprMutator::VisitExpr_(call);
  }

 private:
  IRModule mod_;
  ffi::Map<Function, Function> inlined_functions_;
};

/*!
 * \brief Wrap each created composite function with another function, whose body consists
 * only of a call to the composite function, and annotate the outer function with kCodegen
 * and kGlobalSymbol attributes.
 */
class CompositeFunctionAnnotator : public ExprMutator {
 public:
  explicit CompositeFunctionAnnotator(IRModule mod, IRModule new_mod)
      : ExprMutator(new_mod), mod_(new_mod), inliner(mod) {
    mod_.CopyOnWrite();
  }
  using ExprMutator::VisitExpr_;

  IRModule update() {
    auto gvar = mod_->GetGlobalVar("main");
    auto func = mod_->Lookup(gvar).as_or_throw<Function>();
    builder_->UpdateFunction(gvar, VisitExpr(func).as_or_throw<Function>());
    return builder_->GetContextIRModule();
  }

  Expr VisitExpr_(const CallObj* call) {
    if (call->op->IsInstance<GlobalVarObj>()) {
      GlobalVar cur_var = call->op.as_or_throw<GlobalVar>();
      auto func = mod_->Lookup(cur_var).as_or_throw<Function>();
      if (auto codegen_name = func->GetAttr(tvm::relax::attr::kCodegen)) {
        GlobalVar new_var = [&]() -> GlobalVar {
          if (var_map_.count(cur_var) > 0) {
            // if we visited before, we don't need to create the new function,
            // use the one we stored.
            return var_map_.at(cur_var);
          } else {
            // if it is first time, create the new function with a new name.
            // remove old function from the irmoulde under construction.
            auto old_var = builder_->GetContextIRModule()->GetGlobalVar(cur_var->name_hint);
            builder_->GetContextIRModule()->Remove(old_var);

            // rename the function.
            ffi::String new_func_name = cur_var->name_hint + "_" + codegen_name.value();
            Function new_func = inliner.Run(func.as_or_throw<Function>());
            new_func = WithAttr(new_func, tvm::attr::kGlobalSymbol.name, new_func_name);
            new_func = WithoutAttr(std::move(new_func), tvm::relax::attr::kPrimitive.name);
            // add a function with a new name.
            auto new_var = builder_->AddFunction(new_func, new_func_name);
            var_map_.insert_or_assign(cur_var, new_var);
            return new_var;
          }
        }();
        // we call new var instead of the old one.
        // we don't have to update args since we are just updating the function to call,
        // without any change in the arguments.
        return Call(Type::Missing(), new_var, call->args);
      }
    }
    return ffi::GetRef<Call>(call);
  }

 private:
  IRModule mod_;
  CompositeInliner inliner;
  std::unordered_map<GlobalVar, GlobalVar> var_map_;
};

}  // namespace

IRModule MergeCompositeFunctions(IRModule mod) {
  auto gvar = mod->GetGlobalVar("main");
  auto func = mod->Lookup(gvar).as_or_throw<Function>();
  support::Arena arena;
  auto group_map = CompositeGroupsBuilder(mod, &arena).Run(func);
  // The group map only covers main. Sibling Relax functions are not in it, and
  // visiting them makes OperatorFusor abort in GetGroupFromVar.
  auto new_mod = MakeGroupedFunctions(mod, group_map, /*lift_constants*/ true,
                                      /*entry_function_names*/ {"main"});
  new_mod = CompositeFunctionAnnotator(mod, new_mod).update();

  // TODO(@tvm-team): Implicit pass dependency. Revisit when we have a better way to handle this.
  return DeadCodeElimination(new_mod, {"main"});
}

namespace transform {

Pass MergeCompositeFunctions() {
  auto pass_func =  //
      [=](IRModule mod, PassContext pc) { return relax::MergeCompositeFunctions(mod); };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"MergeCompositeFunctions");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.MergeCompositeFunctions", MergeCompositeFunctions);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
