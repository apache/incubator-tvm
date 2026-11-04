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
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relax/utils.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace {

template <typename T>
using PSet = std::unordered_set<T, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

template <typename T, typename U>
using PMap = std::unordered_map<T, U, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

class PartialTupleUsageCollector : ExprVisitor {
 public:
  static PMap<GlobalVar, std::vector<bool>> Collect(const IRModule& mod) {
    PMap<GlobalVar, size_t> num_outputs;

    for (const auto& [gvar, base_func] : mod->functions) {
      bool is_exposed = base_func->attrs.GetAttr(tvm::attr::kGlobalSymbol).has_value();

      if (!is_exposed) {
        if (auto relax_func = base_func.as<FunctionObj>()) {
          if (auto out_tuple = relax_func->ret_ty.as<TupleTypeObj>()) {
            num_outputs[gvar] = out_tuple->fields.size();
          }
        }
      }
    }

    if (num_outputs.empty()) {
      // Early bail-out if the module has no private functions that
      // return tuples.
      return {};
    }

    PartialTupleUsageCollector visitor(std::move(num_outputs));
    for (const auto& [gvar, base_func] : mod->functions) {
      if (auto func = base_func.as<Function>()) {
        visitor.VisitExpr(func.value());
      }
    }

    PMap<GlobalVar, std::vector<bool>> to_update;
    for (const auto& [gvar, mask] : visitor.output_usage_mask_) {
      bool has_unused_output =
          std::any_of(mask.begin(), mask.end(), [](const bool is_used) { return !is_used; });
      if (has_unused_output) {
        to_update[gvar] = mask;
      }
    }

    return to_update;
  }

 private:
  explicit PartialTupleUsageCollector(PMap<GlobalVar, size_t> num_outputs) {
    for (const auto& [gvar, num_output] : num_outputs) {
      output_usage_mask_[gvar] = std::vector<bool>(num_output, false);
    }
  }

  void VisitBinding(const Binding& binding) override {
    ExprVisitor::VisitBinding(binding);
    known_bindings_.Set(binding->var, GetBoundValue(binding));
  }

  void VisitExpr_(const TupleGetItemObj* op) override {
    if (auto* usage_mask_ptr = GetCalleeUsageMask(op->tuple)) {
      auto& used_indices = *usage_mask_ptr;

      TVM_FFI_CHECK_GE(op->index, 0, IndexError)
          << "Indices for TupleGetItem must be non-negative, "
          << "but expression " << ffi::GetRef<Expr>(op) << " uses a tuple index of " << op->index;
      size_t index = op->index;

      TVM_FFI_CHECK_LT(index, used_indices.size(), IndexError)
          << "Indices for TupleGetItem must be less than the size of the tuple, "
          << "but expression " << ffi::GetRef<Expr>(op) << " uses a tuple index of " << op->index
          << " for a tuple of size " << used_indices.size();
      used_indices[index] = true;
    }
  }

  void VisitExpr_(const VarObj* op) override {
    if (auto* usage_mask_ptr = GetCalleeUsageMask(ffi::GetRef<Var>(op))) {
      auto& usage_mask = *usage_mask_ptr;
      for (size_t i = 0; i < usage_mask.size(); i++) {
        usage_mask[i] = true;
      }
    }
  }

  std::vector<bool>* GetCalleeUsageMask(Expr expr) {
    if (!expr->ty.as<TupleTypeObj>()) {
      return nullptr;
    }

    expr = UnwrapBindings(expr);
    if (auto call = expr.as<CallObj>()) {
      if (auto callee = call->op.as<GlobalVar>()) {
        if (auto it = output_usage_mask_.find(callee.value()); it != output_usage_mask_.end()) {
          return &it->second;
        }
      }
    }

    return nullptr;
  }

  Expr UnwrapBindings(Expr expr) const {
    auto get_bound_value = [&](const Expr& expr) -> ffi::Optional<Expr> {
      if (auto var = expr.as<Var>()) {
        if (auto known_binding = known_bindings_.Get(var.value())) {
          return known_binding.value();
        }
      }
      return std::nullopt;
    };

    while (auto unwrapped = get_bound_value(expr)) {
      expr = unwrapped.value();
    }
    return expr;
  }

  ffi::Map<Var, Expr> known_bindings_;
  PMap<GlobalVar, std::vector<bool>> output_usage_mask_;
};

Function UpdateCallee(Function func, const std::vector<bool>& usage_mask) {
  auto old_func_ty = func->ty.as<FuncTypeObj>();

  auto old_ret_ty = func->ret_ty.as<TupleTypeObj>();
  TVM_FFI_ICHECK(old_ret_ty) << "All functions returning non-tuple outputs "
                             << "should have been pruned already by PartialTupleUsageCollector";

  ffi::Array<Expr> outputs;

  // This helper variable will be removed by the post-proc of
  // CanonicalizeBindings and DeadCodeElimination.
  Var previous_outputs("previous_outputs", func->ret_ty);

  for (size_t i = 0; i < usage_mask.size(); i++) {
    if (usage_mask[i]) {
      outputs.push_back(TupleGetItem(previous_outputs, i));
    }
  }

  Expr new_output = outputs.size() == 1 ? outputs[0] : Tuple(outputs);
  Type new_return_ty = outputs.size() == 1 ? GetType(outputs[0]) : TupleType(outputs.Map(GetType));

  VarBinding binding(previous_outputs, func->body);
  BindingBlock binding_block({binding});
  SeqExpr new_body({binding_block}, new_output);

  auto old_ty = func->ty.as_or_throw<FuncType>();
  FuncType new_ty(old_func_ty->params.value(), new_return_ty, old_func_ty->purity);

  auto write_ptr = func.CopyOnWrite();
  write_ptr->ty = new_ty;
  write_ptr->body = new_body;

  return func;
}

class CallSiteMutator : public ExprMutator {
 public:
  explicit CallSiteMutator(PMap<GlobalVar, std::function<Expr(Call)>> callsite_updaters)
      : callsite_updaters_(callsite_updaters) {}

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallObj* op) override {
    auto node = ExprMutator::VisitExpr_(op).as_or_throw<Call>();

    if (auto gvar = node->op.as<GlobalVar>()) {
      if (auto it = callsite_updaters_.find(gvar.value()); it != callsite_updaters_.end()) {
        return it->second(node);
      }
    }

    return node;
  }

  PMap<GlobalVar, std::function<Expr(Call)>> callsite_updaters_;
};

}  // namespace

namespace transform {

Pass RemoveUnusedOutputs() {
  auto pass_func = [=](IRModule mod, PassContext pc) -> IRModule {
    auto usage = PartialTupleUsageCollector::Collect(mod);

    if (usage.empty()) {
      // Early bail-out if there are no updates to make.
      return mod;
    }

    PMap<GlobalVar, std::function<Expr(Call)>> callsite_updaters;

    {
      IRModule new_callees;

      for (const auto& [gvar, base_func] : mod->functions) {
        if (auto func = base_func.as<Function>()) {
          if (auto it = usage.find(gvar); it != usage.end()) {
            const auto& usage_mask = it->second;
            auto new_func = UpdateCallee(func.value(), usage_mask);

            GlobalVar new_gvar(gvar->name_hint);
            new_gvar->ty = new_func->ty;
            new_callees->Add(new_gvar, new_func);

            callsite_updaters[gvar] = [old_gvar = gvar, new_gvar, usage_mask](Call call) -> Expr {
              TVM_FFI_CHECK(call->op.same_as(old_gvar), InternalError)
                  << "Updater should be applied to " << old_gvar << ", but was applied to "
                  << call->op;

              auto old_call_ty = call->ty.as<TupleTypeObj>();
              TVM_FFI_CHECK(old_call_ty, InternalError)
                  << "Updater should be applied to Call producing an output tuple, "
                  << "but " << call << " has type " << call->ty;
              TVM_FFI_ICHECK_EQ(usage_mask.size(), old_call_ty->fields.size())
                  << "Function " << call->op << " produces " << usage_mask.size() << " outputs, "
                  << "but " << call << " was used in a context expecting "
                  << old_call_ty->fields.size() << " outputs.";

              Call new_call(Type::Missing(), new_gvar, call->args);

              int num_outputs_used = 0;
              for (bool used : usage_mask) {
                num_outputs_used += used;
              }

              ffi::Array<Expr> new_results;
              int new_result_index = 0;
              for (size_t i = 0; i < usage_mask.size(); i++) {
                if (usage_mask[i]) {
                  // This element of the old output tuple was used.  We replace
                  // it either with access into the new output tuple, if callee
                  // still produces multiple outputs, or with the output
                  // itself, if the callee has been reduced to producing a
                  // single output.
                  auto replacement = [&]() -> Expr {
                    if (num_outputs_used == 1) {
                      return new_call;
                    } else {
                      return TupleGetItem(new_call, new_result_index);
                    }
                  }();
                  new_results.push_back(replacement);
                  new_result_index++;
                } else {
                  // This element of the tuple was unused in the old output,
                  // and is no longer generated from the modified callee.  We
                  // could remember the index mapping and re-index any access
                  // into the old tuple, but it's simpler to just let
                  // CanonicalizeBindings and DCE handle it.
                  new_results.push_back(PrimExpr(FloatImm(tvm::PrimType::Float(64), std::nan(""))));
                }
              }

              return Tuple(new_results);
            };
          }
        }
      }

      auto write_ptr = mod.CopyOnWrite();
      for (const auto& [gvar, callee] : new_callees->functions) {
        write_ptr->Remove(write_ptr->GetGlobalVar(gvar->name_hint));
        write_ptr->Add(gvar, callee);
      }
    }

    CallSiteMutator mutator(std::move(callsite_updaters));

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
  auto inner_pass = CreateModulePass(pass_func, 0, "RemoveUnusedOutputsInner");
  return tvm::transform::Sequential(
      {
          inner_pass,
          CanonicalizeBindings(),
          DeadCodeElimination({}),
      },
      "RemoveUnusedOutputs");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.RemoveUnusedOutputs", RemoveUnusedOutputs);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
