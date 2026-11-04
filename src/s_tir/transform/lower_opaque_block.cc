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
 * \file lower_opaque_block.cc
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>

#include <optional>

#include "ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/*!
 * \brief Remove SBlock to ensure that the TIR can not be scheduled again.
 */
class OpaqueBlockLower : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Rewrite(Stmt body) {
    auto lower = ffi::make_object<OpaqueBlockLower>();
    lower->storage_align_ = CollectStorageAlignAnnotation(body);
    return lower->Mutate(body, InplaceMode::kAllow).ValueOrUnchanged(std::move(body));
  }

 private:
  struct UnrollPolicy {
    std::optional<ffi::Any> auto_unroll_max_step;
    std::optional<ffi::Any> unroll_explicit;
  };

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) final {
    // We have convert blocks into opaque blocks in previous passes.
    TVM_FFI_ICHECK(op->iter_values.empty())
        << "Non-opaque blocks are not allowed in FlattenTensor. Please "
           "call pass ConvertBlocksToOpaque before.";
    // Block policy belongs to the loops inside the block after opaque lowering.
    auto parent_policy = unroll_policy_.Current();
    SBlock new_block = unroll_policy_.WithNewScope([&]() {
      unroll_policy_.Current() = parent_policy;
      UpdateUnrollPolicy(op->block->annotations);
      return this->Mutate(op->block, inplace_mode)
          .ValueOrUnchanged(op->block)
          .as_or_throw<SBlock>();
    });
    PrimExpr predicate = this->Mutate(op->predicate, inplace_mode).ValueOrUnchanged(op->predicate);
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = new_block->body;
    if (!IsOne(predicate)) {
      body = If(predicate, std::move(body));
    }
    // Step 3. Handle allocations in reverse order
    ffi::Map<Var, ffi::Array<PrimExpr>> addresses;
    if (auto value = new_block->annotations.Get(s_tir::attr::kTensorAllocatedAddr.name)) {
      for (const auto& entry : value.value().cast<TensorAllocatedAddresses>()) {
        addresses.Set(entry.get<0>(), entry.get<1>());
      }
    }
    for (size_t i = new_block->alloc_tensors.size(); i > 0; --i) {
      const TensorVar& tensor = new_block->alloc_tensors[i - 1];
      ffi::Map<ffi::String, ffi::Any> allocate_annotations;
      auto it = storage_align_.find(tensor);
      if (it != storage_align_.end()) {
        StorageAlignAnnotation allocate_aligns;
        for (auto tuple : it->second) {
          tuple.Set<0>(-1);
          allocate_aligns.push_back(tuple);
        }
        allocate_annotations.Set(s_tir::attr::kTensorDimAlign.name, allocate_aligns);
      }
      ffi::Array<Expr> args;
      if (auto address = addresses.Get(tensor)) args.push_back(tvm::Tuple(address.value()));
      body = SeqStmt({Bind(tensor, Call(tensor.ty(), tirx::alloc_tensor_op(), args,
                                        DictAttrs(allocate_annotations), {tensor.ty()})),
                      std::move(body)});
    }
    return body;
  }

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->Mutate(op->begin, inplace_mode).ValueOrUnchanged(op->begin);
    PrimExpr end = this->Mutate(op->end, inplace_mode).ValueOrUnchanged(op->end);
    PrimExpr extent = Range::GetExtentFromBeginEnd(min, end);
    PrimExpr step = this->Mutate(op->step, inplace_mode).ValueOrUnchanged(op->step);
    auto previous_remap = VarRemapGet(op->var);
    if (!tvm::s_tir::GetThreadBinding(op).has_value() && IsOne(extent) && op->annotations.empty()) {
      // handling unit loop
      VarRemapSet(op->var, prim::cast(min, op->var.ty()));
    }

    // Keep policy on surviving descendants when this owner is lowered away.
    auto parent_policy = unroll_policy_.Current();
    ffi::Map<ffi::String, ffi::Any> new_annotations;
    auto annotations = op->annotations;
    Stmt body = unroll_policy_.WithNewScope([&]() {
      unroll_policy_.Current() = parent_policy;
      UpdateUnrollPolicy(op->annotations);
      const auto& policy = unroll_policy_.Current();
      if (policy.auto_unroll_max_step.has_value()) {
        annotations.Set(tvm::tirx::attr::kAutoUnrollMaxStep.name,
                        policy.auto_unroll_max_step.value());
      }
      if (policy.unroll_explicit.has_value()) {
        annotations.Set(tvm::tirx::attr::kUnrollExplicit.name, policy.unroll_explicit.value());
      }
      // Rewrite annotations before visiting body-local definitions.
      new_annotations = HandleAnnotations(annotations);
      return this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    });
    VarRemapSet(op->var, previous_remap);

    // Step 2. Keep thread-binding loops until LowerThreadBinding.
    if (!tvm::s_tir::GetThreadBinding(op).has_value() && IsOne(extent) && op->annotations.empty() &&
        !op->annotations.count(tvm::s_tir::attr::kIrregularLoopMark.name)) {
      return body;
    }
    return For(op->var, std::move(min), std::move(end), std::move(step), op->kind, std::move(body),
               std::move(new_annotations), op->loc);
  }

  void UpdateUnrollPolicy(const ffi::Map<ffi::String, ffi::Any>& annotations) {
    auto& policy = unroll_policy_.Current();
    if (auto value = annotations.Get(tvm::tirx::attr::kAutoUnrollMaxStep.name);
        value.has_value() && value.value() != nullptr) {
      policy.auto_unroll_max_step = value.value();
    }
    if (auto value = annotations.Get(tvm::tirx::attr::kUnrollExplicit.name);
        value.has_value() && value.value() != nullptr) {
      policy.unroll_explicit = value.value();
    }
  }

  // Effective policy is materialized on each surviving loop, preserving nested overrides.
  ScopeStack<UnrollPolicy> unroll_policy_;

  // Preserve loop annotations while remapping enclosing bindings.
  ffi::Map<ffi::String, ffi::Any> HandleAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations) {
    ffi::Map<ffi::String, ffi::Any> preserved;
    for (const auto& [key, value] : annotations) {
      preserved.Set(key, this->Mutate(value, InplaceMode::kDisallow).ValueOrUnchanged(value));
    }
    return preserved;
  }

  /*! \brief The map from tensor var to its storage alignment information. */
  std::unordered_map<Var, StorageAlignAnnotation> storage_align_;
};

namespace transform {

Pass LowerOpaqueBlock() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto fptr = f.CopyOnWrite();
    fptr->body = OpaqueBlockLower::Rewrite(std::move(fptr->body).value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "s_tir.LowerOpaqueBlock");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerOpaqueBlock", LowerOpaqueBlock);
}
}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
