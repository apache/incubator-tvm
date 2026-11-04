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
 * \file src/relax/transform/dataflow_inplace.cc
 * \brief Pass that converts eligible operator calls in dataflow blocks
 *   into in-place versions.
 */

#include <tvm/base.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op/op.h>
#include <tvm/relax/op_attr.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>
#include <tvm/tirx/stmt_functor.h>

#include "utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

// Ops that may return a tensor sharing storage with the first argument.
// These ops has been verified to share storage with the first argument in
// tests/python/relax/test_dataflow_inplace.py.
bool IsViewMemoryOp(const OpObj* op_node) {
  // TODO: Consider to add more ops that may return a tensor sharing storage with
  // the first argument in the future.
  static const std::unordered_set<std::string> kViewOps = {
      "relax.expand_dims", "relax.squeeze",
      "relax.reshape",     "relax.permute_dims",
      "relax.flatten",     "relax.nn.batch_flatten",
      "relax.memory.view", "relax.memory.ensure_zero_offset",
  };
  return kViewOps.count(op_node->name);
}

// Look up alias ids for a call argument (only Var args are expected in dataflow blocks).
std::unordered_set<int> GetVarAliasSetFromExpr(
    const Expr& arg, const std::unordered_map<Var, std::unordered_set<int>>& alias_sets) {
  if (auto* var_node = arg.as<VarObj>()) {
    Var var = ffi::GetRef<Var>(var_node);
    if (!alias_sets.count(var)) {
      return {-1};
    }
    return alias_sets.at(var);
  }
  return {-1};
}

// In-place on arg `candidate` is invalid if another distinct operand may alias the same
// storage (e.g. two expand_dims views of x bound to different vars). Reject on any shared
// alias id; -1 in the other operand's set does not skip checking other ids. Same var twice
// (e.g. add(z, z)) is allowed.
bool InplaceArgDisjointFromOtherCallArgs(
    const CallObj* call_node, int candidate,
    const std::unordered_map<Var, std::unordered_set<int>>& alias_sets) {
  const auto* cand_var_node = call_node->args[candidate].as<VarObj>();
  if (!cand_var_node) {
    return false;
  }
  auto cand_set = GetVarAliasSetFromExpr(call_node->args[candidate], alias_sets);
  if (cand_set.count(-1)) {
    return false;
  }
  for (size_t j = 0; j < call_node->args.size(); j++) {
    if (static_cast<int>(j) == candidate) {
      continue;
    }
    const Expr& other_arg = call_node->args[j];
    if (other_arg.same_as(call_node->args[candidate])) {
      continue;
    }
    auto other_set = GetVarAliasSetFromExpr(other_arg, alias_sets);
    for (int alias_idx : other_set) {
      if (cand_set.count(alias_idx)) {
        return false;
      }
    }
  }
  return true;
}

// Perform liveness analysis on a dataflow block, returning a map of vars to
// pairs of indices (the liveness interval, from the starting index to the end index).
// A starting index of -1 means the var is defined before the block starts and an end index
// of block->bindings.size() (one past the last index) means it is live after the block ends.
std::unordered_map<Var, std::pair<int, int>> AnalyzeLiveness(const DataflowBlock& block) {
  std::unordered_map<Var, std::pair<int, int>> ret;
  for (int i = block->bindings.size() - 1; i >= 0; i--) {
    Binding b = block->bindings[i];
    Var defined_var = b->var;
    Expr value = GetBoundValue(b);
    ffi::Array<Var> used_vars;
    // for a function literal, we consider only the free vars
    // (those captured from the outer scope)
    if (value.as<FunctionObj>()) {
      used_vars = FreeVars(value);
    } else if (value.as<TupleGetItemObj>()) {
      // Special case: we do not consider a tuple index to be a "use."
      // This is a bit of a hack but allows us to do operations that
      // create tuples to be done in-place (otherwise, any index of the tuple
      // would be considered a use and so the tuple would be live later).
      // Hence we keep the array empty.
    } else {
      used_vars = AllVars(value);
    }

    for (auto var : used_vars) {
      int range_end = i;
      // if the var is not a dataflow var, then it is live
      // after the block (we are not checking later blocks)
      if (!var.as<DataflowVarObj>()) {
        range_end = block->bindings.size();
      }
      if (!ret.count(var)) {
        ret[var] = {-1, range_end};
      }
    }

    if (!ret.count(defined_var)) {
      // if it's an output, then it lives past the end of the block
      if (!defined_var.as<DataflowVarObj>()) {
        ret[defined_var] = {i, block->bindings.size()};
      } else {
        // otherwise, it's live only here
        ret[defined_var] = {i, i};
      }
    } else {
      // this means the var is used later but we encountered its definition now
      auto last_range = ret[defined_var];
      TVM_FFI_ICHECK_EQ(last_range.first, -1);
      std::pair<int, int> new_range = {i, last_range.second};
      ret[defined_var] = new_range;
    }
  }
  return ret;
}

class AliasAnalyzer {
 public:
  AliasAnalyzer() : alias_map_(), tuple_map_(), mem_idx_(0) {}

  // The analysis returns a map of vars to memory locations that it *could* map to
  // (any unique allocation = one memory location), plus a map of memory locations
  // that correspond to tuples (this maps to sets of memory locations for each tuple element).
  // Note: inputs are values that should be assumed not to be aliased and are therefore
  // (in the case of in-place ops) safe to overwrite. This may not be true of function args.
  std::pair<std::unordered_map<Var, std::unordered_set<int>>,
            std::unordered_map<int, std::vector<std::unordered_set<int>>>>
  Analyze(const DataflowBlock& block, const ffi::Array<Var>& inputs) {
    for (auto input : inputs) {
      int curr_idx = get_fresh_idx();
      alias_map_[input] = {curr_idx};
      if (auto* tup_info = GetTypeAs<TupleTypeObj>(input)) {
        InsertFreshTuple(curr_idx, tup_info);
      }
    }

    for (const Binding& binding : block->bindings) {
      Var current_var = binding->var;
      Expr value = GetBoundValue(binding);
      alias_map_[current_var] = GetAliasSet(value, current_var);
    }

    return {alias_map_, tuple_map_};
  }

 private:
  int get_fresh_idx() {
    int ret = mem_idx_;
    mem_idx_++;
    return ret;
  }

  // Fresh tuple = each element is assumed to be a unique allocation
  void InsertFreshTuple(int tup_idx, const TupleTypeObj* tup_info) {
    std::vector<std::unordered_set<int>> tuple_set;
    for (int i = 0; i < static_cast<int>(tup_info->fields.size()); i++) {
      int curr_field = get_fresh_idx();
      tuple_set.push_back({curr_field});
      if (auto* nested_tup_info = tup_info->fields[i].as<TupleTypeObj>()) {
        InsertFreshTuple(curr_field, nested_tup_info);
      }
    }
    tuple_map_[tup_idx] = tuple_set;
  }

  // given a tuple index, add the given memory location indices to each component's
  // alias set
  void UpdateTupleComponents(int tup_idx, const std::unordered_set<int>& insert_idxs) {
    if (tuple_map_.count(tup_idx)) {
      auto tuple_comps = tuple_map_[tup_idx];
      for (size_t i = 0; i < tuple_comps.size(); i++) {
        auto comp_set = tuple_comps[i];

        // if a member is a tuple, update its components as well
        for (int member : comp_set) {
          if (tuple_map_.count(member)) {
            UpdateTupleComponents(member, insert_idxs);
          }
        }

        // update after iterating to avoid iterating over the inserted elements
        tuple_map_[tup_idx][i].insert(insert_idxs.begin(), insert_idxs.end());
      }
    }
  }

  // capture the given index and also its tuple components (including recursively)
  // if they exist
  void AddCapturedIndices(std::unordered_set<int>* captured_set, int idx) {
    captured_set->insert(idx);
    if (tuple_map_.count(idx)) {
      for (auto comp_set : tuple_map_[idx]) {
        for (auto tup_comp_idx : comp_set) {
          AddCapturedIndices(captured_set, tup_comp_idx);
        }
      }
    }
  }

  // Conservative extremely pessimistic assumption:
  // assume that the result of a non-op call can be aliased to any argument
  // or that it could be a newly allocated value.
  // For tuples, assume all members are aliased. Yeah, it's bad.
  // (Skip first arg is for handling call_pure_packed, where the first arg is an ExternFunc that we
  // should ignore)
  std::unordered_set<int> HandleMysteryCall(const CallObj* call_node, const Var& bound_var,
                                            bool skip_first_arg = false) {
    // the result may or may not be newly allocated
    std::unordered_set<int> ret;
    int res_idx = get_fresh_idx();
    // the result may be a tuple
    if (auto* tup_info_node = GetTypeAs<TupleTypeObj>(bound_var)) {
      InsertFreshTuple(res_idx, tup_info_node);
    }
    AddCapturedIndices(&ret, res_idx);

    for (size_t i = (skip_first_arg) ? 1 : 0; i < call_node->args.size(); i++) {
      auto arg = call_node->args[i];
      auto arg_alias_set = GetAliasSet(arg, bound_var);
      for (int alias_idx : arg_alias_set) {
        AddCapturedIndices(&ret, alias_idx);
      }
    }
    // if the result is a tuple, the components can also potentially be aliased to any arg
    // or, in fact, to each other
    UpdateTupleComponents(res_idx, ret);
    return ret;
  }

  // given the expression value, return the set of memory locations corresponding to it
  // (the var the expression is being bound to is needed for type)
  std::unordered_set<int> GetAliasSet(const Expr& value, const Var& bound_var) {
    std::unordered_set<int> ret;

    // cases for value:
    // constant: it's a fresh index
    // var: look up in alias map (-1 if not present)
    // op call: assume it's fresh (may need to make list of exceptions)
    // tuple: fresh entry in tuple index, recurse to determine indices for values
    // function/packed call: chaos reigns, alias with any other argument
    //   (if tuple is passed, assume also aliased with all members of the tuple)
    // tuple index: -1 if tuple is not in tuple map, otherwise look up corresponding entry
    // function constant: give them a fresh index (TODO: we can handle in more detail if this is a
    // case we need to support) prim value: fresh index if node: should not happen inside dataflow
    // block
    if (value.as<GenericConstObj>() || value.as<FunctionObj>()) {
      // TODO(@slyubomirsky): We will probably want special handling for closures
      ret.insert(get_fresh_idx());
    } else if (auto* target_var_node = value.as<VarObj>()) {
      auto target_var = ffi::GetRef<Var>(target_var_node);
      if (alias_map_.count(target_var)) {
        ret.insert(alias_map_[target_var].begin(), alias_map_[target_var].end());
      } else {
        ret.insert(-1);
      }
    } else if (value.as<PrimExpr>()) {
      ret.insert(get_fresh_idx());
    } else if (auto* target_tuple = value.as<TupleObj>()) {
      // fresh idx but we update the tuple map
      int tup_idx = get_fresh_idx();
      ret.insert(tup_idx);
      std::vector<std::unordered_set<int>> new_tuple_map;
      for (auto field : target_tuple->fields) {
        new_tuple_map.push_back(GetAliasSet(field, bound_var));
      }
      tuple_map_[tup_idx] = new_tuple_map;
    } else if (auto* target_tgi = value.as<TupleGetItemObj>()) {
      std::unordered_set<int> tuple_set = GetAliasSet(target_tgi->tuple, bound_var);
      // if -1 is a member of the tuple set, then we have to assume the result is -1
      if (tuple_set.count(-1)) {
        ret.insert(-1);
      } else {
        // otherwise, consider all members that are tuples of appropriate size and index into them
        // (this is safe because the type system will ensure we're not indexing into a tuple
        // of the wrong size)
        for (int member : tuple_set) {
          if (tuple_map_.count(member) &&
              static_cast<int>(tuple_map_[member].size()) > target_tgi->index) {
            auto member_set = tuple_map_[member][target_tgi->index];
            ret.insert(member_set.begin(), member_set.end());
          }
        }
      }
    } else if (auto* call_node = value.as<CallObj>()) {
      if (auto* op_node = call_node->op.as<OpObj>()) {
        // call_pure_packed: treat as non-op call
        if (op_node->name == "relax.call_pure_packed") {
          return HandleMysteryCall(call_node, bound_var, true);
        } else if (op_node->name == "relax.call_tir") {
          // call_tir: can potentially return a tuple
          if (auto* tuple_ty = call_node->ty_args[0].as<TupleTypeObj>()) {
            int tup_idx = get_fresh_idx();
            ret.insert(tup_idx);
            InsertFreshTuple(tup_idx, tuple_ty);
          } else {
            ret.insert(get_fresh_idx());
          }
        } else if (IsViewMemoryOp(op_node) && !call_node->args.empty()) {
          // View-like ops may share storage with their input (and with other views of it).
          return GetAliasSet(call_node->args[0], bound_var);
        } else {
          // We are assuming most op calls return fresh values.
          // We may have to track more exceptions

          // If the returned value is a tuple, we'll assume it's a fresh tuple
          // (there may be exceptions to this too)
          if (auto* tup_info = GetTypeAs<TupleTypeObj>(bound_var)) {
            int tup_idx = get_fresh_idx();
            ret.insert(tup_idx);
            InsertFreshTuple(tup_idx, tup_info);
            return ret;
          }
          ret.insert(get_fresh_idx());
        }
      } else {
        // assume any non-op call can be extremely dangerous and do anything
        return HandleMysteryCall(call_node, bound_var);
      }
    }

    return ret;
  }

  std::unordered_map<Var, std::unordered_set<int>> alias_map_;
  std::unordered_map<int, std::vector<std::unordered_set<int>>> tuple_map_;
  int mem_idx_;
};

// given a shape, return the number of elements corresponding to it (product of elements)
PrimExpr NumElements(const ShapeExpr& shape) {
  PrimExpr ret = IntImm::Int64(1);
  for (auto dim : shape->values) {
    ret *= dim;
  }
  return ret;
}

// Given the type of the result, return any type nested in it
// that is eleigible to be used for in-place computations (tensors are eligible
// only if all their dimensions are integer constants, tuples are eligible if
// all members are eligible though we can consider only individual members separately)
std::unordered_set<Type, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> GatherCandidateType(
    const Type& result_ty) {
  if (auto* tensor_info = result_ty.as<TensorTypeObj>()) {
    // don't consider void dtype (don't know the size at compile time)
    if (tensor_info->IsUnknownDtype()) {
      return {};
    }
    // don't consider cases where we don't know the shape at compile time
    // (we will use the analyzer to do best-effort analysis where there are vars)
    if (tensor_info->shape.as<ShapeExprObj>()) {
      return {ffi::GetRef<TensorType>(tensor_info)};
    } else {
      return {};
    }
  } else if (auto* tuple_info = result_ty.as<TupleTypeObj>()) {
    // we can see if the whole tuple matches or go for any of the components
    std::unordered_set<Type, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> ret;
    for (auto field : tuple_info->fields) {
      auto field_candidates = GatherCandidateType(field);
      ret.insert(field_candidates.begin(), field_candidates.end());
    }
    // at least one field should be eligible to be done in-place
    if (!ret.empty()) {
      ret.insert(ffi::GetRef<Type>(tuple_info));
    }
    return ret;
  } else {
    // don't consider any other types
    return {};
  }
}

// Given the two type, return a pair of bools where the first element is true if
// the two type have the same number of elements and dtype and the second element is true
// if the shapes match _exactly_. Performs this check recursively and ensures the
// stated condition is true for all tensor members of the type (return false
// if a single pair of corresponding tensors does not meet the condition).
std::pair<bool, bool> SizeMatches(const Type& target_info, const Type& arg_info,
                                  const BlockBuilder& ctx) {
  if (target_info.as<TensorTypeObj>() && arg_info.as<TensorTypeObj>()) {
    auto target_tensor = target_info.as_or_throw<TensorType>();
    auto arg_tensor = arg_info.as_or_throw<TensorType>();
    if (target_tensor->shape.has_value() && target_tensor->shape.as<ShapeExprObj>() &&
        arg_tensor->shape.has_value() && arg_tensor->shape.as<ShapeExprObj>()) {
      if (target_tensor->dtype != arg_tensor->dtype) {
        return {false, false};
      }
      auto target_shape = target_tensor->shape.value().as_or_throw<ShapeExpr>();
      auto arg_shape = arg_tensor->shape.value().as_or_throw<ShapeExpr>();
      PrimExpr target_size = NumElements(target_shape);
      PrimExpr arg_size = NumElements(arg_shape);
      if (!ctx->GetAnalyzer()->CanProve(arg_size >= target_size)) {
        return {false, false};
      }
      // exact match: number of dims and each dim matches
      if (target_shape->values.size() == arg_shape->values.size()) {
        for (size_t i = 0; i < target_shape->values.size(); i++) {
          if (!ctx->GetAnalyzer()->CanProveEqual(target_shape->values[i], arg_shape->values[i])) {
            return {true, false};
          }
        }
        return {true, true};
      }
      return {true, false};
    } else {
      return {false, false};
    }
  } else if (target_info.as<TupleTypeObj>() && arg_info.as<TupleTypeObj>()) {
    auto target_tup = target_info.as_or_throw<TupleType>();
    auto arg_tup = arg_info.as_or_throw<TupleType>();
    if (target_tup->fields.size() != arg_tup->fields.size()) {
      return {false, false};
    }
    bool all_exact = true;
    for (size_t i = 0; i < target_tup->fields.size(); i++) {
      // if members aren't either tuples or tensors, simply skip them,
      // since they don't matter for in-place computations
      if (!(target_tup->fields[i].as<TensorTypeObj>() ||
            target_tup->fields[i].as<TupleTypeObj>()) &&
          !(arg_tup->fields[i].as<TensorTypeObj>() || arg_tup->fields[i].as<TupleTypeObj>())) {
        continue;
      }
      auto [field_size_match, field_exact_match] =
          SizeMatches(target_tup->fields[i], arg_tup->fields[i], ctx);
      if (!field_size_match) {
        return {false, false};
      }
      all_exact = all_exact && field_exact_match;
    }
    return {true, all_exact};
  } else {
    return {false, false};
  }
}

// Given an alias index, check if it's a tuple and gather the sets of aliases for the tuple
// members if so (apply recursively if any of those members are tuples).
// Return false if the alias set contains -1, meaning a reference to an unknown or
// possibly dangerous value (no checking we can do for that).
bool GatherSetsToCheckForLiveness(
    const std::unordered_map<Var, std::unordered_set<int>>& alias_sets,
    const std::unordered_map<int, std::vector<std::unordered_set<int>>>& tuple_map,
    std::vector<std::unordered_set<int>>* sets_to_check, int alias_idx) {
  if (tuple_map.count(alias_idx)) {
    for (auto member_set : tuple_map.at(alias_idx)) {
      // contains -1 -> unknown and dangerous, we can short-circuit
      if (member_set.count(-1)) {
        return false;
      }
      sets_to_check->push_back(member_set);

      // if a member can be a tuple, check it recursively
      for (int member : member_set) {
        if (tuple_map.count(member)) {
          if (!GatherSetsToCheckForLiveness(alias_sets, tuple_map, sets_to_check, member)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// Check that the target is not live past the index and that no alias of it is live past the
// binding index (if the target is a tuple, check the conditions recursively for the members)
bool InplaceConditionsMet(
    const std::unordered_map<Var, std::pair<int, int>>& live_ranges,
    const std::unordered_map<Var, std::unordered_set<int>>& alias_sets,
    const std::unordered_map<int, std::vector<std::unordered_set<int>>>& tuple_map,
    const std::unordered_set<Var>& currently_live, const Expr& target, int binding_idx) {
  if (auto* var_node = target.as<VarObj>()) {
    auto current_var = ffi::GetRef<Var>(var_node);
    // if the var is live past this point, we can't use it for in-place computations anyway
    if (live_ranges.count(current_var)) {
      auto live_range = live_ranges.at(current_var);
      if (live_range.second > binding_idx) {
        return false;
      }
    }

    // no entry for the current var -> it must be something external and we have to assume the worst
    if (!alias_sets.count(current_var)) {
      return false;
    }
    auto alias_set = alias_sets.at(current_var);
    // -1 -> an external value and we must assume the worst
    if (alias_set.count(-1)) {
      return false;
    }
    std::vector<std::unordered_set<int>> sets_to_check = {alias_set};
    std::unordered_set<int> indices_checked;
    // If a possible alias is a tuple, we will also check for aliases of the members
    // (possibly recursively)
    for (int alias_idx : alias_set) {
      if (!GatherSetsToCheckForLiveness(alias_sets, tuple_map, &sets_to_check, alias_idx)) {
        return false;
      }
    }

    for (Var other_var : currently_live) {
      if (other_var.same_as(target)) {
        continue;
      }
      // not represented = spooky unknown value that should be modeled by -1
      if (!alias_sets.count(other_var) || !live_ranges.count(other_var)) {
        continue;
      }
      // var is not live past this point => don't need to worry
      if (live_ranges.at(other_var).second <= binding_idx) {
        continue;
      }
      auto other_alias_set = alias_sets.at(other_var);
      for (int alias_idx : other_alias_set) {
        for (auto check_set : sets_to_check) {
          if (check_set.count(alias_idx)) {
            return false;
          }
        }
      }
    }
    return true;
  } else if (auto* tup_node = target.as<TupleObj>()) {
    for (auto field : tup_node->fields) {
      if (!InplaceConditionsMet(live_ranges, alias_sets, tuple_map, currently_live, field,
                                binding_idx)) {
        return false;
      }
    }
    return true;
  } else {
    return true;
  }
}

// this is obviously not a complete list
static std::unordered_set<std::string> SUPPORTED_OPS = {"relax.add",      "relax.subtract",
                                                        "relax.multiply", "relax.divide",
                                                        "relax.nn.silu",  "relax.nn.relu"};
bool OpSupportsInplace(const Op& op) { return SUPPORTED_OPS.count(op->name); }

/*! \brief Corresponds to a binding where at least one argument meets the conditions to be
 *  made in-place. Contains the binding index and indices of the applicable arguments
 */
class InplaceOpportunityObj : public ffi::Object {
 public:
  int64_t binding_idx;
  ffi::Array<int64_t> arg_idxs;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<InplaceOpportunityObj>()
        .def_ro("binding_idx", &InplaceOpportunityObj::binding_idx)
        .def_ro("arg_idxs", &InplaceOpportunityObj::arg_idxs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("relax.transform.InplaceOpportunity", InplaceOpportunityObj,
                              ffi::Object);
};

TVM_FFI_STATIC_INIT_BLOCK() { InplaceOpportunityObj::RegisterReflection(); }

class InplaceOpportunity : public ffi::ObjectRef {
 public:
  TVM_DLL InplaceOpportunity(int64_t binding_idx, const ffi::Array<int64_t>& arg_idxs) {
    auto node = ffi::make_object<InplaceOpportunityObj>();
    node->binding_idx = binding_idx;
    node->arg_idxs = arg_idxs;
    data_ = std::move(node);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(InplaceOpportunity, ffi::ObjectRef,
                                                InplaceOpportunityObj);
};

// Check for in-place eligibility:
//  1. see if there's an arg big enough to hold the result
//  2. see if the arg is live past the call
//  3. see if the arg has an alias that's live past the call
// If the conditions are met, record the index of that binding.
// Returns two lists of lists:
// 1. A list of bindings where at least one argument meets the in-place conditions and the *size*
//    matches the size of the result.
// 2. A list of bindings where at least one argument meets the in-place conditions
//    and *exactly* matches the shape of the result.
// For both lists, each element is a list of ints of the following format:
//   The first element is the index of the *binding* in the block.
//   All remaining elements are the indices of *eligible arguments* in that call.
std::pair<std::vector<InplaceOpportunity>, std::vector<InplaceOpportunity>>
FindInplaceOpportunities(const DataflowBlock& block, const ffi::Array<Var>& inputs,
                         const BlockBuilder& ctx) {
  auto live_ranges = AnalyzeLiveness(block);
  AliasAnalyzer analyzer;
  auto alias_info = analyzer.Analyze(block, inputs);
  auto alias_sets = alias_info.first;
  auto tuple_map = alias_info.second;

  std::vector<InplaceOpportunity> size_match_list;
  std::vector<InplaceOpportunity> exact_match_list;

  // sort the live ranges by starting index
  std::vector<Var> live_order;
  for (auto kv : live_ranges) {
    live_order.push_back(kv.first);
  }
  std::sort(live_order.begin(), live_order.end(),
            [&live_ranges](const Var& var1, const Var& var2) -> bool {
              return live_ranges[var1].first < live_ranges[var2].first;
            });

  std::unordered_set<Var> currently_live;
  int last_live = 0;

  for (size_t i = 0; i < block->bindings.size(); i++) {
    // include all vars that are currently live
    for (int j = last_live; j < static_cast<int>(live_order.size()); j++) {
      auto live_var = live_order[j];
      auto live_range = live_ranges[live_var];
      if (live_range.first > static_cast<int>(i)) {
        break;
      }
      currently_live.insert(live_var);
      last_live++;
    }
    // remove vars whose range has come to an end
    // (keep a separate set to avoid changing the set while iterating on it)
    std::unordered_set<Var> remove;
    for (auto var : currently_live) {
      auto live_range = live_ranges[var];
      if (live_range.second < static_cast<int>(i)) {
        remove.insert(var);
      }
    }
    for (auto var : remove) {
      currently_live.erase(var);
    }

    // if we reach a binding check the conditions
    Binding b = block->bindings[i];
    Var defined_var = b->var;
    Expr value = GetBoundValue(b);

    if (auto* call_node = value.as<CallObj>()) {
      if (auto* op_node = call_node->op.as<OpObj>()) {
        if (!OpSupportsInplace(ffi::GetRef<Op>(op_node))) {
          continue;
        }

        std::unordered_set<int> candidates;
        std::unordered_set<int> exact_match_candidates;

        auto target_ty = GatherCandidateType(GetType(defined_var));
        // can't be done in-place, ignore
        if (target_ty.empty()) {
          continue;
        }

        // Check that at least one argument matches size with the result
        for (size_t j = 0; j < call_node->args.size(); j++) {
          auto arg = call_node->args[j];
          for (auto target : target_ty) {
            auto [matches_size, matches_exactly] = SizeMatches(target, GetType(arg), ctx);
            if (matches_size) {
              candidates.insert(static_cast<int>(j));
              if (matches_exactly) {
                exact_match_candidates.insert(static_cast<int>(j));
              }
            }
          }
        }
        if (candidates.empty()) {
          continue;
        }

        // Make sure at least one candidate is not live past this point and does not have an alias
        // live past this point
        std::unordered_set<int> remove_candidates;
        for (auto candidate : candidates) {
          if (!InplaceConditionsMet(live_ranges, alias_sets, tuple_map, currently_live,
                                    call_node->args[candidate], i) ||
              !InplaceArgDisjointFromOtherCallArgs(call_node, candidate, alias_sets)) {
            remove_candidates.insert(candidate);
          }
        }
        // (remove now to avoid modifying the list as we iterate on it)
        for (auto candidate : remove_candidates) {
          candidates.erase(candidate);
        }

        // if we have a candidate, then this can be made in-place. Report the appropriate candidates
        if (candidates.empty()) {
          continue;
        }

        // produce a list of candidates for this index
        ffi::Array<int64_t> size_candidate_list;
        for (auto candidate : candidates) {
          size_candidate_list.push_back(static_cast<int64_t>(candidate));
        }
        size_match_list.push_back(InplaceOpportunity(static_cast<int64_t>(i), size_candidate_list));

        // also gather up the exact match candidates if there are any
        ffi::Array<int64_t> exact_candidate_list;
        for (auto candidate : candidates) {
          if (!exact_match_candidates.count(candidate)) {
            continue;
          }
          exact_candidate_list.push_back(static_cast<int64_t>(candidate));
        }
        if (exact_candidate_list.empty()) {
          continue;
        }
        exact_match_list.push_back(
            InplaceOpportunity(static_cast<int64_t>(i), exact_candidate_list));
      }
    }
  }

  return {size_match_list, exact_match_list};
}

// Replace tensors in a tirx::Function according to the mapping.
tvm::Stmt RemapTensors(const tvm::Stmt& stmt,
                       const ffi::Map<tirx::TensorVar, tirx::TensorVar>& tensor_map) {
  class TensorMapper : public tirx::StmtExprMutator {
   public:
    explicit TensorMapper(const ffi::Map<tirx::TensorVar, tirx::TensorVar>& tensor_map) {
      for (const auto& [source, target] : tensor_map) {
        VarRemapSet(source, target);
      }
    }
  };

  return ffi::make_object<TensorMapper>(tensor_map)->Mutate(stmt).ValueOrUnchanged(stmt);
}

class ModuleInplaceTransformer : public ExprMutator {
 public:
  explicit ModuleInplaceTransformer(const IRModule& mod) : mod_(mod) {
    builder_ = BlockBuilder::Create(mod);
  }

  IRModule Transform() {
    // visit every Relax function in the module
    for (auto kv : mod_->functions) {
      if (auto* func_node = kv.second.as<FunctionObj>()) {
        auto gv = kv.first;
        auto func_params = func_node->params;
        auto function = VisitExpr(ffi::GetRef<Function>(func_node)).as_or_throw<Function>();
        builder_->UpdateFunction(gv, function);
      }
    }

    auto ret = builder_->GetContextIRModule();
    // clean up to avoid polluting the IRModule
    for (auto gv : legalizers_added) {
      ret->Remove(gv);
    }
    return ret;
  }

  Expr VisitExpr_(const FunctionObj* op) override {
    auto old_func_params = func_params;
    func_params = op->params;
    auto ret = ExprMutator::VisitExpr_(op);
    func_params = old_func_params;
    return ret;
  }

  // the only case we will override: we will visit all binding blocks
  // and replace any valid calls in them
  BindingBlock VisitBindingBlock_(const DataflowBlockObj* op) override {
    auto block = ffi::GetRef<DataflowBlock>(op);
    auto old_idxs = inplace_idxs;

    // For now, only handle exact match cases.
    // Note: Not passing any input values for now, as we can't make any assumptions
    // about them.
    auto matches_found = FindInplaceOpportunities(block, {}, builder_);
    ffi::Map<Binding, ffi::Array<int64_t>> new_idxs;
    for (auto match : matches_found.second) {
      new_idxs.Set(block->bindings[match->binding_idx], match->arg_idxs);
    }

    inplace_idxs = new_idxs;
    auto ret = ExprMutator::VisitBindingBlock_(op);
    inplace_idxs = old_idxs;
    return ret;
  }

  Expr ReplaceBoundCall(const Binding& binding) {
    // can just pick the first index arbitrarily (only using one output for now too)
    // now replace the binding appropriately
    auto arg_idxs = inplace_idxs.at(binding);
    auto target = GetBoundValue(binding).as_or_throw<Call>();
    auto new_call = CreateInplaceCall(target, {arg_idxs[0]});
    return builder_->Normalize(new_call);
  }

  void VisitBinding_(const VarBindingObj* binding) override {
    auto binding_ref = ffi::GetRef<VarBinding>(binding);
    if (!inplace_idxs.count(binding_ref)) {
      ExprMutator::VisitBinding_(binding);
      return;
    }
    Expr new_value = ReplaceBoundCall(binding_ref);
    builder_->EmitNormalized(VarBinding(binding->var, new_value, binding->loc));
  }

  void VisitBinding_(const MatchCastObj* binding) override {
    auto binding_ref = ffi::GetRef<MatchCast>(binding);
    if (!inplace_idxs.count(binding_ref)) {
      ExprMutator::VisitBinding_(binding);
      return;
    }
    Expr new_value = ReplaceBoundCall(binding_ref);
    builder_->EmitNormalized(MatchCast(binding->var, new_value, binding->ty, binding->loc));
  }

  // Given the call and indices of arguments that could be done in-place,
  // replace the call with a call to an in-place tirx::Function.
  // (Made public for testing.)
  Call CreateInplaceCall(const Call& call, const ffi::Array<int64_t>& inplace_indices) {
    static const auto& legalize_map = Op::GetAttrMap(tvm::relax::op_attr::kLegalize);
    static const auto call_tir_inplace_op = Op::Get("relax.call_tir_inplace");

    auto op = call->op.as_or_throw<Op>();
    auto legalized_call = legalize_map[op](builder_, call).as_or_throw<Call>();
    auto* legalized_call_cow = legalized_call.CopyOnWrite();

    // The legalized call should be call_tir. We will replace it with call_tir_inplace
    // and replace the called tirx::Function with an inplace version
    auto legal_op = legalized_call->args[0].as_or_throw<GlobalVar>();
    legalizers_added.push_back(legal_op);
    auto inline_legal_op_name = legal_op->name_hint + "_inplace";

    auto mod = builder_->GetContextIRModule();
    auto old_function = mod->Lookup(legal_op).as_or_throw<tirx::Function>();

    tvm::Stmt new_body = old_function->body.value();

    size_t num_outs = inplace_indices.size();
    size_t num_params = old_function->params.size();

    // the replacement we must make:
    // 1. For each output var, replace its corresponding tensors with the corresponding inplace
    // index
    //    var's tensors
    // 2. For each output var, replace its instances with the corresponding inplace index var
    // 3. Do the same for the *tensor vars* corresponding to the output vars
    // 4. Remove the output vars from the param list
    ffi::Map<tirx::TensorVar, tirx::TensorVar> tensor_subst_map;
    ffi::Map<tvm::Var, tvm::Var> var_subst_map;
    for (size_t i = 0; i < num_outs; i++) {
      // we will substitute output i with the corresponding param indicated by inplace indices
      auto output_var = old_function->params[num_params - num_outs + i];
      auto inplace_var = old_function->params[inplace_indices[i]];
      var_subst_map.Set(output_var, inplace_var);

      // also do the same with the tensor vars
      auto output_tensor = output_var.as_or_throw<tirx::TensorVar>();
      auto inplace_tensor = inplace_var.as_or_throw<tirx::TensorVar>();
      var_subst_map.Set(output_tensor, inplace_tensor);
      tensor_subst_map.Set(output_tensor, inplace_tensor);
    }

    // apply substitutions
    new_body = RemapTensors(new_body, tensor_subst_map);
    auto f_substitute =
        [&var_subst_map](const tvm::Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
      if (auto repl = var_subst_map.Get(var)) return ffi::Any(*std::move(repl));
      return ffi::Unchanged();
    };
    new_body = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(new_body), f_substitute)
                   .as_or_throw<tvm::Stmt>();

    // now get rid of the last num_outputs arguments
    // (couldn't do earlier or else it would have thrown off the indexing)
    ffi::Array<tvm::Var> new_params(old_function->params.begin(),
                                    old_function->params.begin() + (num_params - num_outs));

    tirx::Function new_function(new_params, tvm::SeqStmt(new_body), old_function->ret_type,
                                old_function->attrs, old_function->loc);

    // note: this might be a good time to get rid of the old legalized function, but we don't do it
    // now because later ops might need the same one. Instead, we will clean up at the end
    auto new_gv = builder_->AddFunction(new_function, inline_legal_op_name);

    // update the call (change the op, update the argument, change the attrs)
    legalized_call_cow->op = call_tir_inplace_op;

    ffi::Array<Expr> new_args(legalized_call->args.begin(), legalized_call->args.end());
    new_args.Set(0, new_gv);
    legalized_call_cow->args = new_args;

    ffi::ObjectPtr<CallTIRInplaceAttrs> attrs = ffi::make_object<CallTIRInplaceAttrs>();
    attrs->inplace_indices = inplace_indices;
    legalized_call_cow->attrs = Attrs(attrs);

    return legalized_call;
  }

  // Made public for testing.
  IRModule CurrentMod() { return builder_->GetContextIRModule(); }

 private:
  const IRModule& mod_;
  // Keep track of legalizers we add so we can clean up at the end.
  ffi::Array<GlobalVar> legalizers_added;
  // The current function's params will be treated as non-aliased
  // (we are assuming good behavior on the user's part).
  ffi::Array<Var> func_params;
  // map of eligible bindings to indices of arguments that can be used as the in-place target
  ffi::Map<Binding, ffi::Array<int64_t>> inplace_idxs;
};

namespace transform {

ffi::Map<Var, ffi::Array<int64_t>> DataflowLivenessAnalysis(const DataflowBlock& block) {
  auto liveness_ranges = AnalyzeLiveness(block);
  ffi::Map<Var, ffi::Array<int64_t>> ret;
  for (auto kv : liveness_ranges) {
    ret.Set(kv.first, {kv.second.first, kv.second.second});
  }
  return ret;
}

ffi::Array<ffi::ObjectRef> DataflowAliasAnalysis(const DataflowBlock& block,
                                                 ffi::Array<Var> inputs) {
  AliasAnalyzer analyzer;
  auto res = analyzer.Analyze(block, inputs);
  auto alias_sets = res.first;
  auto tuple_map = res.second;
  ffi::Map<Var, ffi::Array<int64_t>> new_alias_sets;
  ffi::Map<IntImm, ffi::Array<ffi::Array<int64_t>>> new_tuple_map;
  for (auto kv : alias_sets) {
    ffi::Array<int64_t> aliases;
    for (auto alias : kv.second) {
      aliases.push_back(alias);
    }
    new_alias_sets.Set(kv.first, aliases);
  }
  for (auto kv : tuple_map) {
    ffi::Array<ffi::Array<int64_t>> elem_aliases;
    for (auto alias_set : kv.second) {
      ffi::Array<int64_t> dim_aliases;
      for (auto alias : alias_set) {
        dim_aliases.push_back(alias);
      }
      elem_aliases.push_back(dim_aliases);
    }
    new_tuple_map.Set(IntImm::Int32(kv.first), elem_aliases);
  }
  return {new_alias_sets, new_tuple_map};
}

// this would be preferable to do as a dataflow block pass,
// but the transformation adds new Functions, so it affects the module
tvm::transform::Pass DataflowUseInplaceCalls() {
  return tvm::transform::CreateModulePass(
      [](const IRModule& mod, const PassContext& ctx) -> IRModule {
        ModuleInplaceTransformer transformer(mod);
        return transformer.Transform();
      },
      0, "DataflowInsertInPlaceCalls");
}

ffi::Array<ffi::Array<InplaceOpportunity>> DataflowInplaceAnalysis(const DataflowBlock& block,
                                                                   const ffi::Array<Var>& inputs,
                                                                   const IRModule& mod) {
  auto index_lists = relax::FindInplaceOpportunities(block, inputs, BlockBuilder::Create(mod));
  return {ffi::Array<InplaceOpportunity>(index_lists.first.begin(), index_lists.first.end()),
          ffi::Array<InplaceOpportunity>(index_lists.second.begin(), index_lists.second.end())};
}

// these are exposed only for testing
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.testing.transform.DataflowLivenessAnalysis", DataflowLivenessAnalysis)
      .def("relax.testing.transform.DataflowAliasAnalysis", DataflowAliasAnalysis)
      .def("relax.testing.transform.DataflowInplaceAnalysis", DataflowInplaceAnalysis)
      .def("relax.testing.transform.SingleInplaceCall",
           [](const IRModule& mod, const Call& call,
              const ffi::Array<int64_t>& inplace_indices) -> ffi::Array<ffi::ObjectRef> {
             ModuleInplaceTransformer transformer(mod);
             auto ret_call = transformer.CreateInplaceCall(call, inplace_indices);
             return ffi::Array<ffi::ObjectRef>{ret_call, transformer.CurrentMod()};
           });
}

// actually exposed
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.DataflowUseInplaceCalls", DataflowUseInplaceCalls);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
