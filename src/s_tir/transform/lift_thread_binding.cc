/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file convert_block_to_opaque.cc
 * \brief Convert the blocks to opaque blocks which do not have block vars.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

std::pair<std::unordered_map<Stmt, std::vector<std::pair<IterVar, ffi::Map<ffi::String, ffi::Any>>>,
                             ffi::ObjectPtrHash, ffi::ObjectPtrEqual>,
          ffi::Map<Var, Var>>
FindLoopLCA(const Stmt& root) {
  class LCAFinder : public StmtExprVisitor {
   public:
    using StmtExprVisitor::Visit_;
    ffi::Optional<VisitInterrupt> Visit(ffi::AnyView value) override {
      if (value.as<ExprObj>()) return std::nullopt;
      return StmtExprVisitor::Visit(value);
    }
    ffi::Optional<VisitInterrupt> Visit_(const ForObj* op) final {
      stack.push_back(ffi::GetRef<Stmt>(op));
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
      if (tvm::s_tir::GetThreadBinding(op).has_value()) {
        UpdateLCA(op);
      }
      stack.pop_back();
      return std::nullopt;
    }

    void UpdateLCA(const ForObj* loop) {
      std::string thread_tag = tvm::s_tir::GetThreadBinding(loop).value();
      {
        ffi::Map<ffi::String, ffi::Any>* tgt = &annotations[thread_tag];
        for (const auto& kv : loop->annotations) {
          tgt->Set(kv.first, kv.second);
        }
      }
      auto it = iters.find(thread_tag);
      if (it == iters.end()) {
        IterVar iter_var(Range(loop->begin, loop->end),  //
                         loop->var
                             .as_or_throw<Var>()        //
                             .CopyWithName(thread_tag)  //
                             .as_or_throw<PrimVar>(),   //
                         kThreadIndex,                  //
                         thread_tag);
        iters.emplace(thread_tag, iter_var);
        lca[thread_tag] = stack;
        var_subst.Set(loop->var, iter_var->var);
        return;
      }
      var_subst.Set(loop->var, it->second->var);
      std::vector<Stmt>& path = lca[thread_tag];
      uint32_t i = 0;
      for (; i < stack.size() && i < path.size(); ++i) {
        if (!stack[i].same_as(path[i])) {
          break;
        }
      }
      path.erase(path.begin() + i, path.end());
    }

    std::unordered_map<std::string, std::vector<Stmt>> lca;
    std::unordered_map<std::string, IterVar> iters;
    std::unordered_map<std::string, ffi::Map<ffi::String, ffi::Any>> annotations;
    ffi::Map<Var, Var> var_subst;
    std::vector<Stmt> stack;
  };
  auto finder = ffi::make_object<LCAFinder>();
  finder->Visit(root);
  std::unordered_map<Stmt, std::vector<std::pair<IterVar, ffi::Map<ffi::String, ffi::Any>>>,
                     ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      result;
  std::vector<std::string> sorted_thread_tags;
  for (const auto& kv : finder->lca) {
    sorted_thread_tags.push_back(kv.first);
  }
  std::sort(sorted_thread_tags.begin(), sorted_thread_tags.end(),
            [](const std::string& lhs, const std::string& rhs) { return lhs.size() > rhs.size(); });
  for (const auto& thread_tag : sorted_thread_tags) {
    Stmt lca = finder->lca[thread_tag].back();
    const IterVar& iter = finder->iters.at(thread_tag);
    const ffi::Map<ffi::String, ffi::Any>& annotations = finder->annotations[thread_tag];
    result[lca].emplace_back(iter, annotations);
  }
  return {result, finder->var_subst};
}

/*!
 * \brief Substitute expr via BlockRealize value bindings and convert each block into opaque
 *        blocks.
 */
class ThreadBindingLifter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const ForObj* _op, InplaceMode inplace_mode) final {
    For op = ffi::GetRef<For>(_op);
    bool is_kernel_root = false;
    if (tvm::s_tir::GetThreadBinding(op).has_value()) {
      if (iter_lca.empty()) {
        is_kernel_root = true;
        SetKernelRoot(_op);
      }
    }
    For new_op = StmtExprMutator::Mutate_(_op, InplaceMode::kDisallow)
                     .ValueOrUnchanged(ffi::GetRef<Stmt>(_op))
                     .as_or_throw<For>();
    Stmt body = std::move(new_op.CopyOnWrite()->body);
    if (auto it = iter_lca.find(op); it != iter_lca.end()) {
      for (const auto& [iter_var, annotation] : it->second) {
        auto annotations = annotation;
        annotations.Set(tvm::s_tir::attr::kThreadBinding.name, iter_var->thread_tag);
        body = For(iter_var->var, iter_var->dom.value()->begin,
                   iter_var->dom.value()->begin + iter_var->dom.value()->extent,
                   IntImm(iter_var->var.ty(), 1), ForKind::kParallel, std::move(body),
                   std::move(annotations));
      }
    }
    if (is_kernel_root) {
      iter_lca.clear();
    }
    if (tvm::s_tir::GetThreadBinding(op).has_value()) {
      return body;
    } else {
      new_op.CopyOnWrite()->body = std::move(body);
      return new_op;
    }
  }

  void SetKernelRoot(const ForObj* op) {
    auto result = FindLoopLCA(ffi::GetRef<Stmt>(op));
    this->iter_lca = std::move(result.first);
    for (const auto& [var, replacement] : result.second) VarRemapSet(var, replacement);
  }

  std::unordered_map<Stmt, std::vector<std::pair<IterVar, ffi::Map<ffi::String, ffi::Any>>>,
                     ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      iter_lca;
};

namespace transform {

Pass LiftThreadBinding() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    FunctionObj* fptr = f.CopyOnWrite();
    fptr->body = ffi::make_object<ThreadBindingLifter>()
                     ->Mutate(fptr->body, InplaceMode::kAllow)
                     .ValueOrUnchanged(std::move(fptr->body));
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LiftThreadBinding");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LiftThreadBinding", LiftThreadBinding);
}
}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
