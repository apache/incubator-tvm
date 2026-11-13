/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>

#include <unordered_set>

namespace tvm {
namespace relax {

namespace {

// Traverses only ExprObj::ty.  The per-node payload is an intentional constant leaf:
// ExternFuncObj::global_symbol is scalar and BaseFuncObj::attrs is metadata.
template <typename TNode>
TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TypeOnlyExprVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TNode* self = ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  return std::nullopt;
}

template <typename TNode>
TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TypeOnlyExprMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TNode* self = ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  if (mapped_ty.UnchangedOrSameAs(self->ty)) return ffi::Unchanged();
  ffi::ObjectPtr<TNode> copy = ffi::make_object<TNode>(*self);
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  return ffi::Any(std::move(copy));
}

template <typename TNode>
TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TypeOnlyExprMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  TNode* self = const_cast<TNode*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) {
    self->ty = std::move(mapped_ty).ValueUnchecked();
  }
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> IfExprVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const IfExprObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfExprObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->cond));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->true_branch));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->false_branch));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfExprMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const IfExprObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfExprObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_cond,
                                    mutator->MutateExpected(self->cond));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_true_branch,
                                    mutator->MutateExpected(self->true_branch));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_false_branch,
                                    mutator->MutateExpected(self->false_branch));
  if (mapped_ty.UnchangedOrSameAs(self->ty) && mapped_cond.UnchangedOrSameAs(self->cond) &&
      mapped_true_branch.UnchangedOrSameAs(self->true_branch) &&
      mapped_false_branch.UnchangedOrSameAs(self->false_branch)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<IfExprObj> copy = ffi::make_object<IfExprObj>(*self);
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->cond = std::move(mapped_cond).ValueOrUnchanged(std::move(copy->cond));
  copy->true_branch = std::move(mapped_true_branch).ValueOrUnchanged(std::move(copy->true_branch));
  copy->false_branch =
      std::move(mapped_false_branch).ValueOrUnchanged(std::move(copy->false_branch));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IfExprMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  IfExprObj* self = const_cast<IfExprObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IfExprObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_cond,
                                    mutator->MutateExpected(self->cond, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<SeqExpr>, mapped_true_branch,
      mutator->MutateExpected(self->true_branch, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<SeqExpr>, mapped_false_branch,
      mutator->MutateExpected(self->false_branch, ffi::InplaceMode::kAllow));
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) self->ty = std::move(mapped_ty).ValueUnchecked();
  if (!mapped_cond.UnchangedOrSameAs(self->cond)) {
    self->cond = std::move(mapped_cond).ValueUnchecked();
  }
  if (!mapped_true_branch.UnchangedOrSameAs(self->true_branch)) {
    self->true_branch = std::move(mapped_true_branch).ValueUnchecked();
  }
  if (!mapped_false_branch.UnchangedOrSameAs(self->false_branch)) {
    self->false_branch = std::move(mapped_false_branch).ValueUnchecked();
  }
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> ShapeExprVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const ShapeExprObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeExprObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->values));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ShapeExprMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const ShapeExprObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeExprObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_values,
                                    mutator->MutateExpected(self->values));
  if (mapped_ty.UnchangedOrSameAs(self->ty) && mapped_values.UnchangedOrSameAs(self->values)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<ShapeExprObj> copy = ffi::make_object<ShapeExprObj>(*self);
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->values = std::move(mapped_values).ValueOrUnchanged(std::move(copy->values));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ShapeExprMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  ShapeExprObj* self = const_cast<ShapeExprObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeExprObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<PrimExpr>>, mapped_values,
      mutator->MutateExpected(self->values, ffi::InplaceMode::kAllow));
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) self->ty = std::move(mapped_ty).ValueUnchecked();
  if (!mapped_values.UnchangedOrSameAs(self->values)) {
    self->values = std::move(mapped_values).ValueUnchecked();
  }
  return ffi::Unchanged();
}

// Hooks do not inherit, so DataflowVar must mirror the base VarObj remap, PrimType-skip, and
// Simple-to-None definition-region protocol.  Keep this hook triple in lockstep with VarObj.
TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> DataflowVarVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const DataflowVarObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DataflowVarObj>(value);
  if (!self->ty.as<PrimTypeObj>()) {
    if (visitor->def_region_kind() == kTVMFFIDefRegionKindSimple) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
          kTVMFFIDefRegionKindNone, [&]() { return visitor->VisitExpected(self->ty); }));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
    }
  }
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> DataflowVarMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const DataflowVarObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DataflowVarObj>(value);
  ffi::Expected<ffi::Any> remap_result = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::detail::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    return std::move(remap_result);
  }
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindNone) {
    return ffi::Unchanged();
  }
  ffi::UnchangedOr<ffi::Any> result = ffi::Unchanged();
  ffi::Any mapped_value;
  if (!self->ty.as<PrimTypeObj>()) {
    ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
        mutator->def_region_kind() == kTVMFFIDefRegionKindSimple
            ? mutator->WithDefRegionKind(kTVMFFIDefRegionKindNone,
                                         [&]() { return mutator->MutateExpected(self->ty); })
            : mutator->MutateExpected(self->ty);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                      std::move(mapped_ty_result));
    if (!mapped_ty.UnchangedOrSameAs(self->ty)) {
      ffi::ObjectPtr<DataflowVarObj> copy = ffi::make_object<DataflowVarObj>(*self);
      copy->ty = std::move(mapped_ty).ValueUnchecked();
      mapped_value = ffi::Any(std::move(copy));
      result = mapped_value;
    }
  }
  if (!result.IsUnchanged() || mutator->def_region_kind() == kTVMFFIDefRegionKindPattern) {
    ffi::AnyView value_to_store = result.IsUnchanged() ? value : ffi::AnyView(mapped_value);
    auto set_result = mutator->VarRemapSetExpected(value, value_to_store);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return ffi::Unexpected(std::move(set_result).error());
    }
  }
  return std::move(result);
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> DataflowVarMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  DataflowVarObj* self = const_cast<DataflowVarObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DataflowVarObj>(value));
  ffi::Expected<ffi::Any> remap_result = mutator->VarRemapGetExpected(value);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::detail::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    return std::move(remap_result);
  }
  if (mutator->def_region_kind() == kTVMFFIDefRegionKindNone) {
    return ffi::Unchanged();
  }
  ffi::UnchangedOr<ffi::Any> result = ffi::Unchanged();
  ffi::Any mapped_value;
  if (!self->ty.as<PrimTypeObj>()) {
    ffi::Expected<ffi::UnchangedOr<ffi::Any>> mapped_ty_result =
        mutator->def_region_kind() == kTVMFFIDefRegionKindSimple
            ? mutator->WithDefRegionKind(
                  kTVMFFIDefRegionKindNone,
                  [&]() { return mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow); })
            : mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                      std::move(mapped_ty_result));
    if (!mapped_ty.UnchangedOrSameAs(self->ty)) {
      self->ty = std::move(mapped_ty).ValueUnchecked();
      mapped_value = ffi::Any(self);
      result = mapped_value;
    }
  }
  if (!result.IsUnchanged() || mutator->def_region_kind() == kTVMFFIDefRegionKindPattern) {
    ffi::AnyView value_to_store = result.IsUnchanged() ? value : ffi::AnyView(mapped_value);
    auto set_result = mutator->VarRemapSetExpected(value, value_to_store);
    if (TVM_FFI_PREDICT_FALSE(set_result.is_err())) {
      return ffi::Unexpected(std::move(set_result).error());
    }
  }
  return std::move(result);
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> SeqExprVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const SeqExprObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqExprObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->blocks));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SeqExprMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const SeqExprObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqExprObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<BindingBlock>>, mapped_blocks,
                                    mutator->MutateExpected(self->blocks));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_blocks.UnchangedOrSameAs(self->blocks) && mapped_ty.UnchangedOrSameAs(self->ty) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<SeqExprObj> copy = ffi::make_object<SeqExprObj>(*self);
  copy->blocks = std::move(mapped_blocks).ValueOrUnchanged(std::move(copy->blocks));
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> SeqExprMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  SeqExprObj* self = const_cast<SeqExprObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const SeqExprObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Array<BindingBlock>>, mapped_blocks,
      mutator->MutateExpected(self->blocks, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Expr>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  if (!mapped_blocks.UnchangedOrSameAs(self->blocks)) {
    self->blocks = std::move(mapped_blocks).ValueUnchecked();
  }
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) self->ty = std::move(mapped_ty).ValueUnchecked();
  if (!mapped_body.UnchangedOrSameAs(self->body)) {
    self->body = std::move(mapped_body).ValueUnchecked();
  }
  return ffi::Unchanged();
}

// Parameters precede the reflected ty field in this hook triple so their pattern region establishes
// the remap before the derived function type can refer to those symbolic definitions.
TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> FunctionVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), is_pure (scalar)
  const FunctionObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindPattern, [&]() { return visitor->VisitExpected(self->params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ret_ty));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FunctionMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), is_pure (scalar)
  const FunctionObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->params);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_body,
                                    mutator->MutateExpected(self->body));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret_ty,
                                    mutator->MutateExpected(self->ret_ty));
  if (mapped_params.UnchangedOrSameAs(self->params) && mapped_ty.UnchangedOrSameAs(self->ty) &&
      mapped_body.UnchangedOrSameAs(self->body) && mapped_ret_ty.UnchangedOrSameAs(self->ret_ty)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<FunctionObj> copy = ffi::make_object<FunctionObj>(*self);
  copy->params = std::move(mapped_params).ValueOrUnchanged(std::move(copy->params));
  copy->ty = std::move(mapped_ty).ValueOrUnchanged(std::move(copy->ty));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  copy->ret_ty = std::move(mapped_ret_ty).ValueOrUnchanged(std::move(copy->ret_ty));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FunctionMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), is_pure (scalar)
  FunctionObj* self = const_cast<FunctionObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->params,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                    mutator->MutateExpected(self->ty, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SeqExpr>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Type>, mapped_ret_ty,
      mutator->MutateExpected(self->ret_ty, ffi::InplaceMode::kAllow));
  if (!mapped_params.UnchangedOrSameAs(self->params)) {
    self->params = std::move(mapped_params).ValueUnchecked();
  }
  if (!mapped_ty.UnchangedOrSameAs(self->ty)) self->ty = std::move(mapped_ty).ValueUnchecked();
  if (!mapped_body.UnchangedOrSameAs(self->body)) {
    self->body = std::move(mapped_body).ValueUnchecked();
  }
  if (!mapped_ret_ty.UnchangedOrSameAs(self->ret_ty)) {
    self->ret_ty = std::move(mapped_ret_ty).ValueUnchecked();
  }
  return ffi::Unchanged();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { BindingObj::RegisterReflection(); }

IfExpr::IfExpr(Expr cond, Expr true_branch, Expr false_branch, Location loc)
    : Expr(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<IfExprObj> n =
      ffi::make_object<IfExprObj>(std::move(cond), std::move(true_branch), std::move(false_branch));
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  IfExprObj::RegisterReflection();
  refl::TypeAttrDef<IfExprObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&IfExprVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&IfExprMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&IfExprMaybeInplaceMutate>());

  refl::GlobalDef().def("relax.IfExpr",
                        [](Expr cond, Expr true_branch, Expr false_branch, Location loc) {
                          return IfExpr(cond, true_branch, false_branch, loc);
                        });
}

ShapeExpr::ShapeExpr(ffi::Array<PrimExpr> values, Location loc) : Expr(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ShapeExprObj> n = ffi::make_object<ShapeExprObj>();

  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmObj>()) {
      return tvm::prim::cast(value, PrimType::Int(64));
    }
    TVM_FFI_ICHECK(value.ty().MatchesElementType(DLDataTypeCode::kDLInt, 64))
        << "the value in ShapeType can only have dtype of int64";
    return value;
  });
  n->loc = loc;
  n->ty = ShapeType(values, loc);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ShapeExprObj::RegisterReflection();
  refl::TypeAttrDef<ShapeExprObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&ShapeExprVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&ShapeExprMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&ShapeExprMaybeInplaceMutate>());

  refl::GlobalDef().def("relax.ShapeExpr", [](ffi::Array<PrimExpr> values, Location loc) {
    return ShapeExpr(values, loc);
  });
}

DataflowVar::DataflowVar(ffi::String name, ffi::Optional<Type> ty_annotation, Location loc)
    : Var(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<DataflowVarObj> n = ffi::make_object<DataflowVarObj>();
  n->name = std::move(name);
  if (ty_annotation.has_value()) {
    n->ty = ty_annotation.value();
  }
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DataflowVarObj::RegisterReflection();
  refl::TypeAttrDef<DataflowVarObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&DataflowVarVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&DataflowVarMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&DataflowVarMaybeInplaceMutate>());

  refl::GlobalDef().def("relax.DataflowVar",
                        [](ffi::String name, ffi::Optional<Type> ty_annotation, Location loc) {
                          return DataflowVar(name, ty_annotation, loc);
                        });
}

GenericConst MakeTensorConst(ffi::Tensor data, ffi::Optional<Type> ty_annotation, Location loc) {
  TVM_FFI_ICHECK(data.defined());
  if (ty_annotation.has_value()) {
    return GenericConst(ty_annotation.value(), std::move(data), std::move(loc));
  }
  ffi::Array<PrimExpr> shape;
  for (int64_t dim : data.shape()) {
    shape.push_back(IntImm::Int64(dim));
  }
  TensorType ty(ShapeExpr(shape), PrimType(data.dtype()), std::nullopt, loc);
  return GenericConst(std::move(ty), std::move(data), std::move(loc));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("relax.MakeTensorConst", MakeTensorConst);
}

MatchCast::MatchCast(Var var, Expr value, Type ty, Location loc) {
  TVM_FFI_ICHECK(var.defined()) << "MatchCast requires var to be defined";
  ffi::ObjectPtr<MatchCastObj> n = ffi::make_object<MatchCastObj>(std::move(var), std::move(value));
  n->ty = std::move(ty);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MatchCastObj::RegisterReflection();

  refl::GlobalDef().def("relax.MatchCast", [](Var var, Expr value, Type ty, Location loc) {
    return MatchCast(var, value, ty, loc);
  });
}

VarBinding::VarBinding(Var var, Expr value, Location loc) {
  ffi::ObjectPtr<VarBindingObj> n =
      ffi::make_object<VarBindingObj>(std::move(var), std::move(value));
  n->loc = loc;
  data_ = std::move(n);
}

bool VarBindingObj::SEqual(const VarBindingObj* other,
                           ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
  if (value->IsInstance<FunctionObj>()) {
    // Recursive function definitions may reference the bound variable
    // within the value being bound.  In these cases, the
    // var comparison must occur first to define the var, to ensure it is
    // defined at point of use.
    return equal(var, other->var, true, "var") && equal(value, other->value, false, "value");
  } else {
    // In all other cases, visit the bound value before the variable
    // it is bound to, in order to provide better error messages.
    return equal(value, other->value, false, "value") && equal(var, other->var, true, "var");
  }
}

int64_t VarBindingObj::SHash(int64_t init_hash,
                             ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
  int64_t hash_value = init_hash;
  if (value->IsInstance<FunctionObj>()) {
    hash_value = hash(var, hash_value, true);
    hash_value = hash(value, hash_value, false);
  } else {
    hash_value = hash(value, hash_value, false);
    hash_value = hash(var, hash_value, true);
  }
  return hash_value;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  VarBindingObj::RegisterReflection();

  refl::GlobalDef().def("relax.VarBinding", [](Var var, Expr value, Location loc) {
    return VarBinding(var, value, loc);
  });
}

BindingBlock::BindingBlock(ffi::Array<Binding> bindings, Location loc) {
  ffi::ObjectPtr<BindingBlockObj> n = ffi::make_object<BindingBlockObj>();
  n->bindings = std::move(bindings);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BindingBlockObj::RegisterReflection();

  refl::GlobalDef().def("relax.BindingBlock", [](ffi::Array<Binding> bindings, Location loc) {
    return BindingBlock(bindings, loc);
  });
}

BindingBlockObj* BindingBlock::CopyOnWrite() {
  // The `TVM_DEFINE_OBJECT_REF_COW_METHOD` cannot be used for
  // BindingBlock, because it is the base class for `DataflowBlock`.
  // If the `TVM_DEFINE_OBJECT_REF_COW_METHOD` were used, the
  // automatic implementation would erroneously convert from a
  // `DataflowBlock` to a `BindingBlock`.
  TVM_FFI_ICHECK(data_ != nullptr);
  if (!data_.unique()) {
    ffi::ObjectPtr<BindingBlockObj> node;
    if (auto dataflow_block = as<DataflowBlockObj>()) {
      node = ffi::make_object<DataflowBlockObj>(*dataflow_block);
    } else {
      node = ffi::make_object<BindingBlockObj>(*(operator->()));
    }
    ffi::ObjectPtr<ffi::Object>(std::move(node)).swap(data_);
  }
  return static_cast<BindingBlockObj*>(data_.get());
}

DataflowBlock::DataflowBlock(ffi::Array<Binding> bindings, Location loc)
    : BindingBlock(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<DataflowBlockObj> n = ffi::make_object<DataflowBlockObj>();
  n->bindings = std::move(bindings);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DataflowBlockObj::RegisterReflection();

  refl::GlobalDef().def("relax.DataflowBlock", [](ffi::Array<Binding> bindings, Location loc) {
    return DataflowBlock(bindings, loc);
  });
}

SeqExpr::SeqExpr(Expr body) : Expr(ffi::UnsafeInit{}) {
  if (auto seq = body.as<SeqExpr>()) {
    *this = seq.value();
  } else {
    *this = SeqExpr(ffi::Array<BindingBlock>{}, body);
  }
}

SeqExpr::SeqExpr(ffi::Array<BindingBlock> blocks, Expr body, Location loc)
    : Expr(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<SeqExprObj> n = ffi::make_object<SeqExprObj>(std::move(body));
  n->blocks = std::move(blocks);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  SeqExprObj::RegisterReflection();
  refl::TypeAttrDef<SeqExprObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&SeqExprVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&SeqExprMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&SeqExprMaybeInplaceMutate>());

  refl::GlobalDef().def("relax.SeqExpr", [](ffi::Array<BindingBlock> blocks, Expr body,
                                            Location loc) { return SeqExpr(blocks, body, loc); });
}

Function::Function(ffi::Array<Var> params, Expr body, ffi::Optional<Type> ret_ty, bool is_pure,
                   DictAttrs attrs, Location loc)
    : BaseFunc(ffi::UnsafeInit{}) {
  // Set the function type.
  // For function, we take a conservative approach and require the function type
  // to be known at construction time.
  ffi::Array<Type> param_ty;

  for (const Var& param : params) {
    TVM_FFI_ICHECK(!param->ty.as<MissingType>().has_value())
        << "relax.Function requires params to contain ty";
    param_ty.push_back(GetType(param));
  }

  ffi::Optional<Type> body_ty;

  if (!body->ty.as<MissingType>().has_value()) {
    body_ty = GetType(body);
  }

  TVM_FFI_ICHECK(body_ty.has_value() || ret_ty.has_value())
      << "Function must be constructed with either "
      << "an explicit type for the return type, "
      << "or a normalized body with type.";

  // Use the body's type if there is no explicit return type,
  // or if the body may provide a more granular return type.
  bool use_body_ty =
      !ret_ty.has_value() || (body_ty && ret_ty && IsBaseOf(ret_ty.value(), body_ty.value()));

  if (use_body_ty) {
    // MatchCast nodes within the body may introduce new symbolic
    // variables.  These are in-scope for the function body, but not
    // for the function's return type.  When hoisting the body's type
    // to the function return type, symbolic variables may only be
    // used if they were defined by the function's parameters.
    auto f_var_map = [&] {
      auto tir_vars = DefinableTIRVarsInType(TupleType(params.Map(GetType)));
      std::unordered_set<tvm::Var> lookup(tir_vars.begin(), tir_vars.end());
      return [lookup = std::move(lookup)](const Var& var) -> ffi::Optional<Expr> {
        if (auto prim_var = var.as<PrimVar>(); prim_var && lookup.count(prim_var.value())) {
          return prim_var.value().as_or_throw<PrimExpr>();
        }
        return std::nullopt;
      };
    }();
    ret_ty = EraseToWellDefined(body_ty.value(), f_var_map);
  }

  FuncType func_ty(param_ty, ret_ty.value(), is_pure);

  // set the fields
  ffi::ObjectPtr<FunctionObj> n = ffi::make_object<FunctionObj>(std::move(body));
  n->params = std::move(params);
  n->ret_ty = ret_ty.value();
  n->is_pure = is_pure;
  n->ty = std::move(func_ty);
  n->attrs = std::move(attrs);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FunctionObj::RegisterReflection();
  refl::TypeAttrDef<FunctionObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&FunctionVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&FunctionMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&FunctionMaybeInplaceMutate>());

  refl::GlobalDef().def("relax.Function",
                        [](ffi::Array<Var> params, Expr body, ffi::Optional<Type> ret_ty,
                           bool is_pure, DictAttrs attrs, Location loc) {
                          return Function(params, body, ret_ty, is_pure, attrs, loc);
                        });
}

Function Function::CreateEmpty(ffi::Array<Var> params, Type ret_ty, bool is_pure, DictAttrs attrs,
                               Location loc) {
  ffi::Array<Type> param_ty;
  for (const Var& param : params) {
    TVM_FFI_ICHECK(!param->ty.as<MissingType>().has_value())
        << "relax.Function requires params to contain ty.";
    param_ty.push_back(GetType(param));
  }

  FuncType finfo(param_ty, ret_ty, is_pure);

  // A dummy body, to ensure that the empty function is still well-formed.
  Expr body = [&]() -> Expr {
    Var output("output", ret_ty);
    Call expr(Type::Missing(), ExternFunc("_dummy_function", FuncType({}, ret_ty)), {});

    return SeqExpr({BindingBlock({VarBinding(output, expr)})}, output);
  }();

  // set the fields
  ffi::ObjectPtr<FunctionObj> n = ffi::make_object<FunctionObj>(std::move(body));
  n->params = std::move(params);
  n->is_pure = is_pure;
  n->ty = std::move(finfo);
  n->ret_ty = std::move(ret_ty);
  n->attrs = std::move(attrs);
  n->loc = loc;
  return Function(std::move(n));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.FunctionCreateEmpty",
      [](ffi::Array<Var> params, Type ret_ty, bool is_pure, DictAttrs attrs, Location loc) {
        return Function::CreateEmpty(params, ret_ty, is_pure, attrs, loc);
      });
}

// Special opaque derivation function for ExternFunc
// Take look at ty_args to figure out the return Type.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  auto infer_by_ty_args = [](const Call& call, const BlockBuilder& ctx) -> Type {
    TVM_FFI_ICHECK(call->ty_args.defined()) << "ty_args field of CallObj should always be defined";
    if (call->ty_args.empty()) {
      return AnyType();
    } else if (call->ty_args.size() == 1) {
      return call->ty_args[0];
    } else {
      return TupleType(call->ty_args);
    }
  };
  refl::GlobalDef().def("tvm.relax.type.infer_by_ty_args", infer_by_ty_args);
}

// Get the derive function.
FuncType GetExternFuncType() {
  EnvFunc fn = EnvFunc::Get("tvm.relax.type.infer_by_ty_args");
  TypeDeriveFunc derive(fn);
  return FuncType::OpaqueFunc(derive);
}

ExternFunc::ExternFunc(ffi::String global_symbol, Location loc)
    : ExternFunc(global_symbol, GetExternFuncType(), loc) {}

ExternFunc::ExternFunc(ffi::String global_symbol, Type ty, Location loc)
    : BaseFunc(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(ty.as<FuncTypeObj>())
      << "ExternFunc must have FuncType, "
      << "but declaration of '" << global_symbol << "' received " << ty;

  ffi::ObjectPtr<ExternFuncObj> n = ffi::make_object<ExternFuncObj>();
  n->global_symbol = std::move(global_symbol);
  n->loc = loc;
  n->ty = ty;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ExternFuncObj::RegisterReflection();
  refl::TypeAttrDef<ExternFuncObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&TypeOnlyExprVisit<ExternFuncObj>>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TypeOnlyExprMutate<ExternFuncObj>>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TypeOnlyExprMaybeInplaceMutate<ExternFuncObj>>());

  refl::GlobalDef().def("relax.ExternFunc",
                        [](ffi::String global_symbol, ffi::Optional<Type> ty, Location loc) {
                          if (ty.has_value()) {
                            return ExternFunc(global_symbol, ty.value(), loc);
                          } else {
                            return ExternFunc(global_symbol, loc);
                          }
                        });
}

Expr GetShapeOf(const Expr& expr) {
  // default case, to be normalized.
  TVM_FFI_ICHECK(!expr->ty.as<MissingType>().has_value())
      << "GetShapeOf can only be applied to normalized expr";
  auto* tinfo = GetTypeAs<TensorTypeObj>(expr);

  TVM_FFI_ICHECK(tinfo != nullptr) << "ShapeOf can only be applied to expr with TensorType";
  if (tinfo->shape.has_value()) return tinfo->shape.value();

  static const Op op = Op::Get("relax.shape_of");
  // default case, call shape of, eagerly normalize the expr.
  Call call_shape_of(Type::Missing(), op, {expr}, {}, {});
  UpdateType(call_shape_of, ShapeType(tinfo->ndim));
  return call_shape_of;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.GetShapeOf", [](const Expr& expr) { return GetShapeOf(expr); })
      .def("relax.FuncWithAttr",
           [](BaseFunc func, ffi::String key, ffi::ObjectRef value) -> ffi::Optional<Function> {
             if (func->IsInstance<relax::FunctionObj>()) {
               return WithAttr(std::move(func).as_or_throw<relax::Function>(), key, value);
             }
             return std::nullopt;
           })
      .def("relax.FuncWithAttrs",
           [](BaseFunc func, ffi::Map<ffi::String, ffi::Any> attr_map) -> ffi::Optional<Function> {
             if (func->IsInstance<relax::FunctionObj>()) {
               return WithAttrs(std::move(func).as_or_throw<relax::Function>(), attr_map);
             }
             return std::nullopt;
           })
      .def("relax.FuncWithoutAttr", [](BaseFunc func, ffi::String key) -> ffi::Optional<Function> {
        if (func->IsInstance<relax::FunctionObj>()) {
          return WithoutAttr(std::move(func).as_or_throw<relax::Function>(), key);
        }
        return std::nullopt;
      });
}

}  // namespace relax
}  // namespace tvm
