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
 * \file src/relax/transform/kill_after_last_use.cc
 * \brief Kill storage/tensor objects after last use, if not already killed
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/transform.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/stmt_functor.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace relax {

class UnusedTrivialBindingRemover : public ExprMutator {
 public:
  static Expr Apply(Expr expr) {
    struct UsedCollector : ExprVisitor {
      void VisitExpr_(const VarObj* val) override { used.insert(val); }
      void VisitExpr_(const DataflowVarObj* val) override {
        VisitExpr_(static_cast<const VarObj*>(val));
      }

      void VisitBinding_(const VarBindingObj* binding, const VarObj* val) override {
        has_trivial_binding.insert(binding->var.get());
        ExprVisitor::VisitBinding_(binding, val);
      }
      void VisitBinding_(const MatchCastObj* binding) override {
        if (binding->value.as<VarObj>() &&
            ffi::StructuralEqual()(GetType(binding->var), GetType(binding->value))) {
          has_trivial_binding.insert(binding->var.get());
        }
        ExprVisitor::VisitBinding_(binding);
      }
      void VisitBinding_(const VarBindingObj* binding, const DataflowVarObj* val) override {
        VisitBinding_(binding, static_cast<const VarObj*>(val));
      }

      std::unordered_set<const VarObj*> used;
      std::unordered_set<const VarObj*> has_trivial_binding;
    };

    UsedCollector collector;
    collector(expr);

    auto to_remove = std::move(collector.has_trivial_binding);
    for (const auto& used : collector.used) {
      to_remove.erase(used);
    }

    UnusedTrivialBindingRemover remover(to_remove);
    return remover(expr);
  }

 private:
  explicit UnusedTrivialBindingRemover(std::unordered_set<const VarObj*> to_remove)
      : to_remove_(std::move(to_remove)) {}

  void VisitBinding(const Binding& binding) override {
    if (!to_remove_.count(binding->var.get())) {
      ExprMutator::VisitBinding(binding);
    }
  }

  std::unordered_set<const VarObj*> to_remove_;
};

class CollectLastUsage : public ExprVisitor {
 public:
  struct LastUsage {
    std::vector<const VarObj*> tensors;
    std::vector<const VarObj*> storage;
    std::vector<const VarObj*> objects;
  };
  using Result = std::unordered_map<const VarObj*, LastUsage>;

  static Result Collect(const Expr& expr) {
    CollectLastUsage visitor;
    visitor(expr);

    Result output;
    for (const auto* var : visitor.binding_order_) {
      if (auto it = visitor.last_usage_of_.find(var); it != visitor.last_usage_of_.end()) {
        const auto* last_usage_point = it->second;
        bool is_output = last_usage_point == nullptr;
        bool already_killed = visitor.killed_objects_.count(var);

        // Currently, the VM requires that objects to be killed
        // only exist in VM registers. This requires KillAfterLastUse
        // to have more knowledge about the VM implementation than
        // should exist at this stage of lowering. In the future,
        // this may be handled more easily at the CodeGenVM level.
        //
        // Variables bound to `relax.null_value` are excluded for the
        // same reason as constants: CodeGenVM special-cases `null_value` to bypass register
        // allocation, so such a variable is never a valid target for R.vm.kill_object. A new
        // codegen special case should be reflected here as well.
        bool stored_in_vm_register =
            !(visitor.constant_tensors_.count(var) || visitor.null_value_objects_.count(var) ||
              var->ty.as<FuncTypeObj>() || var->ty.as<ShapeTypeObj>() || var->ty.as<PrimTypeObj>());

        if (!is_output && !already_killed) {
          if (visitor.storage_objects_.count(var)) {
            output[last_usage_point].storage.push_back(var);
          } else if (var->ty.as<TensorTypeObj>() && stored_in_vm_register) {
            output[last_usage_point].tensors.push_back(var);
          } else if (stored_in_vm_register) {
            output[last_usage_point].objects.push_back(var);
          }
        }
      }
    }

    return output;
  }

  void VisitBinding(const Binding& binding) override {
    auto cache = current_binding_;
    current_binding_ = binding->var.get();
    binding_order_.push_back(current_binding_);
    ExprVisitor::VisitBinding(binding);
    current_binding_ = cache;
  }

  void VisitExpr_(const VarObj* op) override {
    ExprVisitor::VisitExpr_(op);
    // Overwrite any previous usage, such that after the visitor
    // completes, last_usage_of_ contains the last usage point.  If
    // this occurs in an output, then current_binding_ will be
    // nullptr.
    last_usage_of_[op] = current_binding_;
  }

  void VisitBinding_(const VarBindingObj* binding, const CallObj* val) override {
    static const Op vm_alloc_storage = Op::Get("relax.vm.alloc_storage");
    static const Op mem_alloc_storage = Op::Get("relax.memory.alloc_storage");
    static const Op null_value_op = Op::Get("relax.null_value");

    static const Op mem_kill_tensor = Op::Get("relax.memory.kill_tensor");
    static const Op mem_kill_storage = Op::Get("relax.memory.kill_storage");
    static const Op vm_kill_object = Op::Get("relax.vm.kill_object");

    if (val->op.same_as(vm_alloc_storage) || val->op.same_as(mem_alloc_storage)) {
      storage_objects_.insert(binding->var.get());
    } else if (val->op.same_as(null_value_op)) {
      null_value_objects_.insert(binding->var.get());
    } else if (val->op.same_as(mem_kill_tensor) || val->op.same_as(mem_kill_storage) ||
               val->op.same_as(vm_kill_object)) {
      TVM_FFI_ICHECK_EQ(val->args.size(), 1)
          << "Operator " << val->op << " should have one argument, "
          << "but instead found " << val->args.size() << " arguments: " << val->args;
      auto killed_object = val->args[0].as<VarObj>();
      TVM_FFI_ICHECK(killed_object)
          << "Internal error: non-normalized expression " << ffi::GetRef<Call>(val);
      killed_objects_.insert(killed_object);
    } else {
      // Only recursively visit if it isn't one of the special cases.
      ExprVisitor::VisitBinding_(binding, val);
    }
  }

  void VisitBinding_(const VarBindingObj* binding, const GenericConstObj* val) override {
    if (val->value.as<ffi::TensorObj>()) constant_tensors_.insert(binding->var.get());
  }

 private:
  // The current binding being visited, or nullptr if no binding is
  // being visited.
  const VarObj* current_binding_{nullptr};

  // Order of bindings, to ensure consistent order of destruction, in
  // case a Binding is the last usage for more than one variable.
  std::vector<const VarObj*> binding_order_;

  // Map from a variable to the last variable binding that makes use
  // of it.
  std::unordered_map<const VarObj*, const VarObj*> last_usage_of_;

  // Storage objects, eligible for R.vm.kill_object.  This cannot be
  // determined solely from the Type, because the
  // `R.*.alloc_storage` operators return AnyType
  std::unordered_set<const VarObj*> storage_objects_;

  // Constants, which do not have a VM register, and may *not* have
  // R.builtin.kill_tensor called on them.
  std::unordered_set<const VarObj*> constant_tensors_;

  // Variables bound to `relax.null_value`, which do not occupy a VM
  // register in CodeGenVM, and therefore must
  // never be passed to R.vm.kill_object.
  std::unordered_set<const VarObj*> null_value_objects_;

  // Set of objects that already have a call node to kill them.  Should not have a duplicate
  std::unordered_set<const VarObj*> killed_objects_;

  // Trivial var-to-var bindings.
  std::unordered_map<const VarObj*, const VarObj*> trivial_bindings_;
};

class KillInserter : public ExprMutator {
 private:
  Expr VisitExpr_(const FunctionObj* op) override {
    last_usage_ = CollectLastUsage::Collect(ffi::GetRef<Expr>(op));
    auto mutated = ExprMutator::VisitExpr_(op);
    last_usage_.clear();
    return mutated;
  }

  Expr VisitExpr_(const SeqExprObj* op) override {
    last_usage_ = CollectLastUsage::Collect(ffi::GetRef<Expr>(op));
    auto mutated = ExprMutator::VisitExpr_(op);
    last_usage_.clear();
    return mutated;
  }

  void VisitBinding(const Binding& binding) override {
    ExprMutator::VisitBinding(binding);
    if (auto it = last_usage_.find(binding->var.get()); it != last_usage_.end()) {
      static const Op mem_kill_tensor = Op::Get("relax.memory.kill_tensor");
      for (const auto& tensor_obj : it->second.tensors) {
        builder_->Emit(Call(Type::Missing(), mem_kill_tensor, {ffi::GetRef<Expr>(tensor_obj)}),
                       /*name_hint=*/"_");
      }

      static const Op mem_kill_storage = Op::Get("relax.memory.kill_storage");
      for (const VarObj* storage_obj : it->second.storage) {
        builder_->Emit(Call(Type::Missing(), mem_kill_storage, {ffi::GetRef<Expr>(storage_obj)}),
                       /*name_hint=*/"_");
      }

      static const Op vm_kill_object = Op::Get("relax.vm.kill_object");
      for (const VarObj* obj : it->second.objects) {
        builder_->Emit(Call(Type::Missing(), vm_kill_object, {ffi::GetRef<Expr>(obj)}),
                       /*name_hint=*/"_");
      }
    }
  }

  CollectLastUsage::Result last_usage_;
};

Expr KillAfterLastUse(Expr expr) {
  expr = CanonicalizeBindings(expr);
  expr = UnusedTrivialBindingRemover::Apply(expr);

  KillInserter mutator;
  return mutator(expr);
}

namespace transform {

Pass KillAfterLastUse() {
  auto pass_func = [=](Function func, IRModule m, PassContext pc) {
    return relax::KillAfterLastUse(std::move(func)).as_or_throw<Function>();
  };
  return CreateFunctionPass(pass_func, /*opt_level=*/0, "KillAfterLastUse");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.KillAfterLastUse", KillAfterLastUse);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
