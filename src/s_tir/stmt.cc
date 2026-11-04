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
 * \file tvm/s_tir/stmt.cc
 * \brief Schedulable block definitions and structural traversal.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/attr.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;
using namespace tvm::prim;

const Op& async_copy_scope() {
  static const Op op = Op::Get("s_tir.async_copy_scope");
  return op;
}

const Op& async_commit() {
  static const Op op = Op::Get("s_tir.async_commit");
  return op;
}

const Op& async_wait() {
  static const Op op = Op::Get("s_tir.async_wait");
  return op;
}

const Op& manual_sync() {
  static const Op op = Op::Get("s_tir.manual_sync");
  return op;
}

static ffi::Array<Var> RegionNoBodyParams(const CallObj*) { return {}; }

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("s_tir.async_copy_scope", "Mark eligible copies for asynchronous lowering.")
      .signature()
      .set_native_fn_attr<&RegionNoBodyParams>(tvm::op_attr::kRegionGetBodyParams);
  OpDef("s_tir.manual_sync", "Use explicitly authored synchronization within the body.")
      .signature()
      .set_native_fn_attr<&RegionNoBodyParams>(tvm::op_attr::kRegionGetBodyParams);
  OpDef("s_tir.async_commit", "Commit asynchronous copies to a queue.")
      .signature(sig::arg<IntImm>("queue_id"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kUpdateState);
  OpDef("s_tir.async_wait", "Wait for committed asynchronous copies.")
      .signature(sig::arg<IntImm>("queue_id"), sig::arg<PrimExpr>("inflight_count"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kUpdateState);
}

namespace {

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> MatchTensorRegionVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const MatchTensorRegionObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchTensorRegionObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindPattern, [&]() { return visitor->VisitExpected(self->tensor); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->source));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> MatchTensorRegionMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const MatchTensorRegionObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchTensorRegionObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<TensorVar>, mapped_tensor,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->tensor);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<TensorRegion>, mapped_source,
                                    mutator->MutateExpected(self->source));
  if (mapped_tensor.UnchangedOrSameAs(self->tensor) &&
      mapped_source.UnchangedOrSameAs(self->source)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<MatchTensorRegionObj> copy = ffi::make_object<MatchTensorRegionObj>(*self);
  copy->tensor = std::move(mapped_tensor).ValueOrUnchanged(std::move(copy->tensor));
  copy->source = std::move(mapped_source).ValueOrUnchanged(std::move(copy->source));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> MatchTensorRegionMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  MatchTensorRegionObj* self = const_cast<MatchTensorRegionObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const MatchTensorRegionObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<TensorVar>, mapped_tensor,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->tensor,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<TensorRegion>, mapped_source,
      mutator->MutateExpected(self->source, ffi::InplaceMode::kAllow));
  if (mapped_tensor.UnchangedOrSameAs(self->tensor) &&
      mapped_source.UnchangedOrSameAs(self->source)) {
    return ffi::Unchanged();
  }
  if (!mapped_tensor.IsUnchanged()) self->tensor = std::move(mapped_tensor).ValueUnchecked();
  if (!mapped_source.IsUnchanged()) self->source = std::move(mapped_source).ValueUnchecked();
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> SBlockVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // Establish allocation and match-tensor definitions before their region uses.
  // Whole iterators and annotations remain part of structural traversal.
  // skips: name_hint
  const SBlockObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->iter_vars));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->alloc_tensors); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->match_tensors));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->reads));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->writes));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->annotations));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->init));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SBlockMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // Establish allocation and match-tensor definitions before their region uses.
  // Whole iterators and annotations remain part of structural traversal.
  // skips: name_hint
  const SBlockObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<IterVar>>, mapped_iter_vars,
                                    mutator->MutateExpected(self->iter_vars));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<TensorVar>>, mapped_alloc_tensors,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->alloc_tensors);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<MatchTensorRegion>>,
                                    mapped_match_tensors,
                                    mutator->MutateExpected(self->match_tensors));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<TensorRegion>>, mapped_reads,
                                    mutator->MutateExpected(self->reads));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<TensorRegion>>, mapped_writes,
                                    mutator->MutateExpected(self->writes));
  using AnnotationMap = ffi::Map<ffi::String, ffi::Any>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<AnnotationMap>, mapped_annotations,
                                    mutator->MutateExpected(self->annotations));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_init,
                                    mutator->MutateExpected(self->init));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_iter_vars.UnchangedOrSameAs(self->iter_vars) &&
      mapped_reads.UnchangedOrSameAs(self->reads) &&
      mapped_writes.UnchangedOrSameAs(self->writes) &&
      mapped_alloc_tensors.UnchangedOrSameAs(self->alloc_tensors) &&
      mapped_match_tensors.UnchangedOrSameAs(self->match_tensors) &&
      mapped_annotations.UnchangedOrSameAs(self->annotations) &&
      (mapped_init.IsUnchanged() || ffi::AnyView(mapped_init).same_as(self->init)) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<SBlockObj> copy = ffi::make_object<SBlockObj>(*self);
  copy->iter_vars = std::move(mapped_iter_vars).ValueOrUnchanged(std::move(copy->iter_vars));
  copy->reads = std::move(mapped_reads).ValueOrUnchanged(std::move(copy->reads));
  copy->writes = std::move(mapped_writes).ValueOrUnchanged(std::move(copy->writes));
  copy->alloc_tensors =
      std::move(mapped_alloc_tensors).ValueOrUnchanged(std::move(copy->alloc_tensors));
  copy->match_tensors =
      std::move(mapped_match_tensors).ValueOrUnchanged(std::move(copy->match_tensors));
  copy->annotations = std::move(mapped_annotations).ValueOrUnchanged(std::move(copy->annotations));
  if (!mapped_init.IsUnchanged()) {
    auto replacement = std::move(mapped_init).ValueUnchecked();
    copy->init = replacement.has_value()
                     ? ffi::Optional<SeqStmt>(SeqStmt(std::move(replacement).value()))
                     : std::nullopt;
  }
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SBlockMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // Establish allocation and match-tensor definitions before their region uses.
  // Whole iterators and annotations remain part of structural traversal.
  // skips: name_hint
  SBlockObj* self = const_cast<SBlockObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<IterVar>>, mapped_iter_vars,
      mutator->MutateExpected(self->iter_vars, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<TensorVar>>, mapped_alloc_tensors,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->alloc_tensors,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<MatchTensorRegion>>, mapped_match_tensors,
      mutator->MutateExpected(self->match_tensors, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<TensorRegion>>, mapped_reads,
                                    mutator->MutateExpected(self->reads, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<TensorRegion>>, mapped_writes,
      mutator->MutateExpected(self->writes, ffi::InplaceMode::kAllow));
  using AnnotationMap = ffi::Map<ffi::String, ffi::Any>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<AnnotationMap>, mapped_annotations,
      mutator->MutateExpected(self->annotations, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_init,
                                    mutator->MutateExpected(self->init, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  if (mapped_iter_vars.UnchangedOrSameAs(self->iter_vars) &&
      mapped_reads.UnchangedOrSameAs(self->reads) &&
      mapped_writes.UnchangedOrSameAs(self->writes) &&
      mapped_alloc_tensors.UnchangedOrSameAs(self->alloc_tensors) &&
      mapped_match_tensors.UnchangedOrSameAs(self->match_tensors) &&
      mapped_annotations.UnchangedOrSameAs(self->annotations) &&
      (mapped_init.IsUnchanged() || ffi::AnyView(mapped_init).same_as(self->init)) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  if (!mapped_iter_vars.IsUnchanged())
    self->iter_vars = std::move(mapped_iter_vars).ValueUnchecked();
  if (!mapped_reads.IsUnchanged()) self->reads = std::move(mapped_reads).ValueUnchecked();
  if (!mapped_writes.IsUnchanged()) self->writes = std::move(mapped_writes).ValueUnchecked();
  if (!mapped_alloc_tensors.IsUnchanged()) {
    self->alloc_tensors = std::move(mapped_alloc_tensors).ValueUnchecked();
  }
  if (!mapped_match_tensors.IsUnchanged()) {
    self->match_tensors = std::move(mapped_match_tensors).ValueUnchecked();
  }
  if (!mapped_annotations.IsUnchanged())
    self->annotations = std::move(mapped_annotations).ValueUnchecked();
  if (!mapped_init.IsUnchanged()) {
    auto replacement = std::move(mapped_init).ValueUnchecked();
    self->init = replacement.has_value()
                     ? ffi::Optional<SeqStmt>(SeqStmt(std::move(replacement).value()))
                     : std::nullopt;
  }
  if (!mapped_body.IsUnchanged()) self->body = SeqStmt(std::move(mapped_body).ValueUnchecked());
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> SBlockRealizeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SBlockRealizeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->iter_values));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->predicate));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->block));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SBlockRealizeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SBlockRealizeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_iter_values,
                                    mutator->MutateExpected(self->iter_values));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_predicate,
                                    mutator->MutateExpected(self->predicate));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SBlock>, mapped_block,
                                    mutator->MutateExpected(self->block));
  if (mapped_iter_values.UnchangedOrSameAs(self->iter_values) &&
      mapped_predicate.UnchangedOrSameAs(self->predicate) &&
      mapped_block.UnchangedOrSameAs(self->block)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<SBlockRealizeObj> copy = ffi::make_object<SBlockRealizeObj>(*self);
  copy->iter_values = std::move(mapped_iter_values).ValueOrUnchanged(std::move(copy->iter_values));
  copy->predicate = std::move(mapped_predicate).ValueOrUnchanged(std::move(copy->predicate));
  copy->block = std::move(mapped_block).ValueOrUnchanged(std::move(copy->block));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SBlockRealizeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  SBlockRealizeObj* self = const_cast<SBlockRealizeObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SBlockRealizeObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_iter_values,
      mutator->MutateExpected(self->iter_values, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<PrimExpr>, mapped_predicate,
      mutator->MutateExpected(self->predicate, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SBlock>, mapped_block,
                                    mutator->MutateExpected(self->block, ffi::InplaceMode::kAllow));
  if (mapped_iter_values.UnchangedOrSameAs(self->iter_values) &&
      mapped_predicate.UnchangedOrSameAs(self->predicate) &&
      mapped_block.UnchangedOrSameAs(self->block)) {
    return ffi::Unchanged();
  }
  if (!mapped_iter_values.IsUnchanged())
    self->iter_values = std::move(mapped_iter_values).ValueUnchecked();
  if (!mapped_predicate.IsUnchanged())
    self->predicate = std::move(mapped_predicate).ValueUnchecked();
  if (!mapped_block.IsUnchanged()) self->block = std::move(mapped_block).ValueUnchecked();
  return ffi::Unchanged();
}

}  // namespace

// MatchTensorRegion
MatchTensorRegion::MatchTensorRegion(TensorVar tensor, TensorRegion source) {
  const TensorVar& source_tensor = source->source.as_or_throw<TensorVar>();
  TVM_FFI_ICHECK_EQ(source_tensor.shape().size(), source->region.size())
      << "MatchTensorRegion source must match its tensor rank";
  sym::Analyzer analyzer;
  // Check scope and dtype
  TVM_FFI_ICHECK_EQ(tensor.scope(), source_tensor.scope())
      << "MatchTensor " << tensor << " scope mismatch:" << tensor.scope() << " vs. "
      << source_tensor.scope();
  TVM_FFI_ICHECK_EQ(tensor.dtype(), source_tensor.dtype())
      << "MatchTensor " << tensor << " data type mismatch:" << tensor.dtype() << " vs. "
      << source_tensor.dtype();

  // Check data_alignment
  TVM_FFI_ICHECK(source_tensor.ty()->data_alignment % tensor.ty()->data_alignment == 0)
      << "Trying to match tensor to another one with lower alignment requirement "
      << " required alignment=" << tensor.ty()->data_alignment
      << ", provided alignment=" << source_tensor.ty()->data_alignment;

  // Validate shape
  TVM_FFI_ICHECK(source->region.size() >= tensor.shape().size())
      << "Dimension of source ffi::Array<Range> expected to be larger or equal than target tensor "
         "shape, but "
         "got "
      << source->region.size() << " vs. " << tensor.shape().size();
  size_t offset = source->region.size() - tensor.shape().size();
  for (size_t i = 0; i < offset; ++i) {
    TVM_FFI_ICHECK(analyzer->CanProve(source->region[i]->extent == 1))
        << "The higher dimension should be 1, but got " << source->region[i]->extent << ".";
  }
  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    const Range& source_range = source->region[i + offset];
    const PrimExpr& tensor_shape = tensor.shape()[i];
    if (!tensor_shape.as<PrimVar>()) {
      TVM_FFI_ICHECK(analyzer->CanProve(source_range->extent == tensor_shape))
          << "The dimension mismatched between source region and target tensor shape, got "
          << source_range->extent << " vs. " << tensor_shape << ".";
    }
  }
  // Note that we do not check elem_offset and strides in this function
  ffi::ObjectPtr<MatchTensorRegionObj> node =
      ffi::make_object<MatchTensorRegionObj>(std::move(tensor), std::move(source));
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MatchTensorRegionObj::RegisterReflection();
  refl::TypeAttrDef<MatchTensorRegionObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&MatchTensorRegionVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&MatchTensorRegionMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&MatchTensorRegionMaybeInplaceMutate>());

  refl::GlobalDef().def("s_tir.MatchTensorRegion", [](TensorVar tensor, TensorRegion source) {
    return MatchTensorRegion(tensor, source);
  });
}

// Block
SBlock::SBlock(ffi::Array<IterVar> iter_vars, ffi::Array<TensorRegion> reads,
               ffi::Array<TensorRegion> writes, ffi::String name_hint, SeqStmt body,
               ffi::Optional<SeqStmt> init, ffi::Array<TensorVar> alloc_tensors,
               ffi::Array<MatchTensorRegion> match_tensors, ffi::Map<ffi::String, Any> annotations,
               Location loc)
    : Stmt(ffi::UnsafeInit{}) {
  for (const auto& regions : {reads, writes}) {
    for (const TensorRegion& region : regions) {
      const auto tensor = region->source.as_or_throw<TensorVar>();
      TVM_FFI_ICHECK_EQ(tensor.shape().size(), region->region.size())
          << "SBlock region must match its tensor rank";
    }
  }
  ffi::ObjectPtr<SBlockObj> node = ffi::make_object<SBlockObj>(std::move(body));
  node->iter_vars = std::move(iter_vars);
  node->reads = std::move(reads);
  node->writes = std::move(writes);
  node->name_hint = std::move(name_hint);
  node->init = std::move(init);
  node->alloc_tensors = std::move(alloc_tensors);
  node->match_tensors = std::move(match_tensors);
  node->annotations = std::move(annotations);
  node->loc = loc;
  data_ = std::move(node);
}

SBlock::SBlock(ffi::String name_hint, SeqStmt body, ffi::Array<TensorVar> alloc_tensors,
               Location loc)
    : Stmt(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<SBlockObj> node = ffi::make_object<SBlockObj>(std::move(body));
  node->iter_vars = {};
  node->reads = {};
  node->writes = {};
  node->name_hint = std::move(name_hint);
  node->init = std::nullopt;
  node->alloc_tensors = std::move(alloc_tensors);
  node->match_tensors = {};
  node->annotations = {};
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SBlockObj::RegisterReflection();
  refl::TypeAttrDef<SBlockObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&SBlockVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&SBlockMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&SBlockMaybeInplaceMutate>());

  refl::GlobalDef().def("s_tir.SBlock",
                        [](ffi::Array<IterVar> iter_vars, ffi::Array<TensorRegion> reads,
                           ffi::Array<TensorRegion> writes, ffi::String name_hint, SeqStmt body,
                           ffi::Optional<SeqStmt> init, ffi::Array<TensorVar> alloc_tensors,
                           ffi::Array<MatchTensorRegion> match_tensors,
                           ffi::Map<ffi::String, Any> annotations, Location loc) {
                          return SBlock(iter_vars, reads, writes, name_hint, body, init,
                                        alloc_tensors, match_tensors, annotations, loc);
                        });
}

// BlockRealize
SBlockRealize::SBlockRealize(ffi::Array<PrimExpr> values, PrimExpr predicate, SBlock block,
                             Location loc)
    : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK_EQ(block->iter_vars.size(), values.size(), ValueError)
      << "BlockRealize needs to have the same number of iter_vars and binding values";
  PrimType predicate_ty = predicate.ty();
  TVM_FFI_CHECK(predicate_ty.MatchesCode(DLDataTypeCode::kDLBool), TypeError)
      << "Expect Block.predicate to be a bool expression";
  ffi::ObjectPtr<SBlockRealizeObj> node =
      ffi::make_object<SBlockRealizeObj>(std::move(predicate), std::move(block));
  node->iter_values = std::move(values);
  node->loc = loc;
  data_ = std::move(node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SBlockRealizeObj::RegisterReflection();
  refl::TypeAttrDef<SBlockRealizeObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&SBlockRealizeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&SBlockRealizeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&SBlockRealizeMaybeInplaceMutate>());

  refl::GlobalDef().def("s_tir.SBlockRealize", [](ffi::Array<PrimExpr> iter_values,
                                                  PrimExpr predicate, SBlock block, Location loc) {
    return SBlockRealize(iter_values, predicate, block, loc);
  });
}

}  // namespace s_tir
}  // namespace tvm
