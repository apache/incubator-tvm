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
 * \file stmt_functor.cc
 * \brief Native traversal of schedulable TIR nodes.
 */
#include <tvm/s_tir/stmt_functor.h>

#include <utility>
#include <vector>

namespace tvm {
namespace s_tir {

using namespace tirx;

void StmtExprVisitor::InitVTable(VTable* vtable) {
  tirx::StmtExprVisitor::InitVTable(vtable);
  SetDispatch<StmtExprVisitor, SBlockObj>(vtable);
  SetDispatch<StmtExprVisitor, SBlockRealizeObj>(vtable);
}

void StmtExprMutator::InitVTable(VTable* vtable) {
  tirx::StmtExprMutator::InitVTable(vtable);
  SetDispatch<StmtExprMutator, SBlockObj>(vtable);
  SetDispatch<StmtExprMutator, SBlockRealizeObj>(vtable);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SBlockObj* op) {
  return VisitBlock(this, op);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::VisitBlock(ffi::StructuralVisitorObj* visitor,
                                                          const SBlockObj* op) {
  for (const IterVar& iter_var : op->iter_vars) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
        visitor->VisitExpected(iter_var->dom.value()->begin).value());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(
        visitor->VisitExpected(iter_var->dom.value()->extent).value());
  }
  for (const TensorVar& tensor : op->alloc_tensors) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(tensor).value(); }));
  }
  // Define match-tensor targets before visiting reads/writes that may use them.
  // This differs from the old TIRX native order (reads/writes before matches)
  // and agrees with structural traversal's definition-before-use contract.
  for (const MatchTensorRegion& match_tensor_region : op->match_tensors) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
        kTVMFFIDefRegionKindSimple,
        [&]() { return visitor->VisitExpected(match_tensor_region->tensor).value(); }));
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(match_tensor_region->source).value());
  }
  if (auto addresses = op->annotations.Get(attr::kTensorAllocatedAddr.name)) {
    // The owner reference identifies the allocation; it is not a data access.
    for (const auto& entry : addresses.value().cast<TensorAllocatedAddresses>()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(entry.get<1>()).value());
    }
  }
  for (const TensorRegion& region : op->reads) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(region).value());
  }
  for (const TensorRegion& region : op->writes) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(region).value());
  }
  if (op->init.has_value()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(op->init.value()).value());
  }
  return visitor->VisitExpected(op->body).value();
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::Visit_(const SBlockRealizeObj* op) {
  return VisitBlockRealize(this, op);
}

ffi::Optional<VisitInterrupt> StmtExprVisitor::VisitBlockRealize(ffi::StructuralVisitorObj* visitor,
                                                                 const SBlockRealizeObj* op) {
  for (const auto& child : op->iter_values) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(child).value());
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(op->predicate).value());
  return visitor->VisitExpected(op->block).value();
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SBlockObj* op, InplaceMode inplace_mode) {
  return MutateBlock(this, op, inplace_mode);
}

UnchangedOr<Stmt> StmtExprMutator::MutateBlock(ffi::StructuralMutatorObj* mutator,
                                               const SBlockObj* op, InplaceMode inplace_mode) {
  // SBlock iteration variables keep their binders; only their domains are expressions here.
  const auto* iters = op->iter_vars.GetArrayObj();
  InplaceMode iter_mode = iters->unique() ? inplace_mode : InplaceMode::kDisallow;
  std::vector<std::pair<size_t, IterVar>> replacements;
  for (size_t i = 0; i < iters->size(); ++i) {
    const auto* iter = (*iters)[i].as<IterVarObj>();
    if (!iter->dom.has_value()) continue;
    InplaceMode domain_mode = iter->unique() ? iter_mode : InplaceMode::kDisallow;
    auto domain = mutator->MutateExpected(iter->dom.value(), domain_mode)
                      .value()
                      .as_or_throw<UnchangedOr<Range>>();
    if (domain.UnchangedOrSameAs(iter->dom.value())) continue;
    if (domain_mode == InplaceMode::kAllow) {
      const_cast<IterVarObj*>(iter)->dom = std::move(domain).ValueUnchecked();
    } else {
      auto updated = ffi::make_object<IterVarObj>(*iter);
      updated->dom = std::move(domain).ValueUnchecked();
      replacements.emplace_back(i, IterVar(std::move(updated)));
    }
  }
  UnchangedOr<ffi::Array<IterVar>> iter_vars = ffi::Unchanged();
  if (!replacements.empty()) {
    if (iter_mode == InplaceMode::kAllow) {
      for (auto& [i, iter] : replacements) {
        const_cast<ffi::ArrayObj*>(iters)->SetItem(i, std::move(iter));
      }
    } else {
      ffi::Array<IterVar> updated = op->iter_vars;
      for (auto& [i, iter] : replacements) updated.Set(i, std::move(iter));
      iter_vars = std::move(updated);
    }
  }
  auto alloc_tensors =
      mutator
          ->WithDefRegionKind(
              kTVMFFIDefRegionKindSimple,
              [&] { return mutator->MutateExpected(op->alloc_tensors, inplace_mode).value(); })
          .as_or_throw<UnchangedOr<ffi::Array<TensorVar>>>();
  auto match_tensors = mutator->MutateExpected(op->match_tensors, inplace_mode)
                           .value()
                           .as_or_throw<UnchangedOr<ffi::Array<MatchTensorRegion>>>();
  auto annotations = op->annotations;
  if (auto addresses = annotations.Get(attr::kTensorAllocatedAddr.name)) {
    auto updated = mutator->MutateExpected(addresses.value(), InplaceMode::kDisallow).value();
    if (!updated.IsUnchanged()) {
      annotations.Set(attr::kTensorAllocatedAddr.name, std::move(updated).ValueUnchecked());
    }
  }
  auto reads = mutator->MutateExpected(op->reads, inplace_mode)
                   .value()
                   .as_or_throw<UnchangedOr<ffi::Array<TensorRegion>>>();
  auto writes = mutator->MutateExpected(op->writes, inplace_mode)
                    .value()
                    .as_or_throw<UnchangedOr<ffi::Array<TensorRegion>>>();
  auto init = mutator->MutateExpected(op->init, inplace_mode)
                  .value()
                  .as_or_throw<UnchangedOr<ffi::Optional<SeqStmt>>>();
  auto body =
      mutator->MutateExpected(op->body, inplace_mode).value().as_or_throw<UnchangedOr<SeqStmt>>();
  if (iter_vars.UnchangedOrSameAs(op->iter_vars) &&
      alloc_tensors.UnchangedOrSameAs(op->alloc_tensors) && reads.UnchangedOrSameAs(op->reads) &&
      writes.UnchangedOrSameAs(op->writes) && match_tensors.UnchangedOrSameAs(op->match_tensors) &&
      init.UnchangedOrSameAs(op->init) && body.UnchangedOrSameAs(op->body) &&
      annotations.same_as(op->annotations))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<SBlockObj*>(op);
    writable->annotations = std::move(annotations);
    if (!iter_vars.IsUnchanged()) writable->iter_vars = std::move(iter_vars).ValueUnchecked();
    if (!alloc_tensors.IsUnchanged())
      writable->alloc_tensors = std::move(alloc_tensors).ValueUnchecked();
    if (!reads.IsUnchanged()) writable->reads = std::move(reads).ValueUnchecked();
    if (!writes.IsUnchanged()) writable->writes = std::move(writes).ValueUnchecked();
    if (!match_tensors.IsUnchanged())
      writable->match_tensors = std::move(match_tensors).ValueUnchecked();
    if (!init.IsUnchanged()) writable->init = std::move(init).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<SBlockObj>(*op);
  copy->annotations = std::move(annotations);
  if (!iter_vars.IsUnchanged()) copy->iter_vars = std::move(iter_vars).ValueUnchecked();
  if (!alloc_tensors.IsUnchanged()) copy->alloc_tensors = std::move(alloc_tensors).ValueUnchecked();
  if (!reads.IsUnchanged()) copy->reads = std::move(reads).ValueUnchecked();
  if (!writes.IsUnchanged()) copy->writes = std::move(writes).ValueUnchecked();
  if (!match_tensors.IsUnchanged()) copy->match_tensors = std::move(match_tensors).ValueUnchecked();
  if (!init.IsUnchanged()) copy->init = std::move(init).ValueUnchecked();
  if (!body.IsUnchanged()) copy->body = std::move(body).ValueUnchecked();
  return Stmt(std::move(copy));
}

UnchangedOr<Stmt> StmtExprMutator::Mutate_(const SBlockRealizeObj* op, InplaceMode inplace_mode) {
  return MutateBlockRealize(this, op, inplace_mode);
}

UnchangedOr<Stmt> StmtExprMutator::MutateBlockRealize(ffi::StructuralMutatorObj* mutator,
                                                      const SBlockRealizeObj* op,
                                                      InplaceMode inplace_mode) {
  auto iter_values = mutator->MutateExpected(op->iter_values, inplace_mode)
                         .value()
                         .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto predicate = mutator->MutateExpected(op->predicate, inplace_mode)
                       .value()
                       .as_or_throw<UnchangedOr<PrimExpr>>();
  auto block =
      mutator->MutateExpected(op->block, inplace_mode).value().as_or_throw<UnchangedOr<SBlock>>();
  if (iter_values.UnchangedOrSameAs(op->iter_values) &&
      predicate.UnchangedOrSameAs(op->predicate) && block.UnchangedOrSameAs(op->block))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<SBlockRealizeObj*>(op);
    if (!iter_values.IsUnchanged()) writable->iter_values = std::move(iter_values).ValueUnchecked();
    if (!predicate.IsUnchanged()) writable->predicate = std::move(predicate).ValueUnchecked();
    if (!block.IsUnchanged()) writable->block = std::move(block).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<SBlockRealizeObj>(*op);
  if (!iter_values.IsUnchanged()) copy->iter_values = std::move(iter_values).ValueUnchecked();
  if (!predicate.IsUnchanged()) copy->predicate = std::move(predicate).ValueUnchecked();
  if (!block.IsUnchanged()) copy->block = std::move(block).ValueUnchecked();
  return Stmt(std::move(copy));
}

}  // namespace s_tir
}  // namespace tvm
