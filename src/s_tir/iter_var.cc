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
 * \file iter_var.cc
 * \brief Iteration-variable definitions.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/iter_var.h>

#include <utility>

namespace tvm {
namespace s_tir {

namespace {

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> IterVarVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: iter_type, thread_tag
  const IterVarObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IterVarObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->dom));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->var); }));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IterVarMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: iter_type, thread_tag
  const IterVarObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IterVarObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Range>>, mapped_dom,
                                    mutator->MutateExpected(self->dom));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimVar>, mapped_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var);
                                    }));
  if (mapped_dom.UnchangedOrSameAs(self->dom) && mapped_var.UnchangedOrSameAs(self->var)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<IterVarObj> copy = ffi::make_object<IterVarObj>(*self);
  copy->dom = std::move(mapped_dom).ValueOrUnchanged(std::move(copy->dom));
  copy->var = std::move(mapped_var).ValueOrUnchanged(std::move(copy->var));
  copy->ty = copy->var.ty();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IterVarMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: iter_type, thread_tag
  IterVarObj* self = const_cast<IterVarObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IterVarObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Range>>, mapped_dom,
                                    mutator->MutateExpected(self->dom, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimVar>, mapped_var,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                      return mutator->MutateExpected(self->var,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  self->dom = std::move(mapped_dom).ValueOrUnchanged(std::move(self->dom));
  self->var = std::move(mapped_var).ValueOrUnchanged(std::move(self->var));
  self->ty = self->var.ty();
  return ffi::Unchanged();
}

}  // namespace

// IterVar
IterVar::IterVar(ffi::Optional<Range> dom, PrimVar var, IterVarType t, ffi::String thread_tag,
                 Location loc)
    : PrimExpr(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<IterVarObj> n = ffi::make_object<IterVarObj>(var);
  if (dom.has_value() && dom.value()->extent.defined()) {
    PrimType extent_ty = dom.value()->extent.ty();
    PrimType var_ty = var.ty();
    TVM_FFI_ICHECK(extent_ty.code() == DLDataTypeCode::kDLInt)
        << "The dtype of the domain of an IterVar must be an integer type. However, the domain's "
           "dtype is "
        << extent_ty->dtype;
    TVM_FFI_ICHECK(extent_ty == var_ty)
        << "The dtype of the extent of an IterVar (" << extent_ty->dtype
        << ") must match its associated Var's dtype (" << var_ty->dtype << ")";
  }
  n->dom = dom;
  n->iter_type = t;
  n->thread_tag = thread_tag;
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  IterVarObj::RegisterReflection();
  refl::TypeAttrDef<IterVarObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&IterVarVisit>())
      .attr(ffi::type_attr::kStructuralMutate, ffi::FStructuralMutate::FromNative<&IterVarMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&IterVarMaybeInplaceMutate>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.IterVar", [](ffi::Optional<Range> dom, PrimVar var, int iter_type,
                                            ffi::String thread_tag, Location loc) {
    return IterVar(dom, var, static_cast<IterVarType>(iter_type), thread_tag, loc);
  });
}

}  // namespace s_tir
}  // namespace tvm
