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
 * \file src/relax/distributed/type.cc
 * \brief Relax DTensor type.
 */

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/distributed/type.h>
namespace tvm {
namespace relax {
namespace distributed {

namespace {

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> DTensorTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const DTensorTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DTensorTypeObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->device_mesh));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->placement));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->tensor_ty));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> DTensorTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const DTensorTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DTensorTypeObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<DeviceMesh>, mapped_device_mesh,
                                    mutator->MutateExpected(self->device_mesh));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Placement>, mapped_placement,
                                    mutator->MutateExpected(self->placement));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<TensorType>, mapped_tensor_ty,
                                    mutator->MutateExpected(self->tensor_ty));
  if (mapped_device_mesh.UnchangedOrSameAs(self->device_mesh) &&
      mapped_placement.UnchangedOrSameAs(self->placement) &&
      mapped_tensor_ty.UnchangedOrSameAs(self->tensor_ty)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<DTensorTypeObj> copy = ffi::make_object<DTensorTypeObj>(*self);
  copy->device_mesh = std::move(mapped_device_mesh).ValueOrUnchanged(std::move(copy->device_mesh));
  copy->placement = std::move(mapped_placement).ValueOrUnchanged(std::move(copy->placement));
  copy->tensor_ty = std::move(mapped_tensor_ty).ValueOrUnchanged(std::move(copy->tensor_ty));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> DTensorTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  DTensorTypeObj* self = const_cast<DTensorTypeObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const DTensorTypeObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<DeviceMesh>, mapped_device_mesh,
      mutator->MutateExpected(self->device_mesh, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Placement>, mapped_placement,
      mutator->MutateExpected(self->placement, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<TensorType>, mapped_tensor_ty,
      mutator->MutateExpected(self->tensor_ty, ffi::InplaceMode::kAllow));
  if (!mapped_device_mesh.UnchangedOrSameAs(self->device_mesh)) {
    self->device_mesh = std::move(mapped_device_mesh).ValueUnchecked();
  }
  if (!mapped_placement.UnchangedOrSameAs(self->placement)) {
    self->placement = std::move(mapped_placement).ValueUnchecked();
  }
  if (!mapped_tensor_ty.UnchangedOrSameAs(self->tensor_ty)) {
    self->tensor_ty = std::move(mapped_tensor_ty).ValueUnchecked();
  }
  return ffi::Unchanged();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  PlacementObj::RegisterReflection();
  PlacementSpecObj::RegisterReflection();
}

PlacementSpec PlacementSpec::Sharding(int axis) {
  ffi::ObjectPtr<PlacementSpecObj> n = ffi::make_object<PlacementSpecObj>();
  n->axis = axis;
  n->kind = PlacementSpecKind::kSharding;
  return PlacementSpec(ffi::UnsafeInit{}, std::move(n));
}

PlacementSpec PlacementSpec::Replica() {
  ffi::ObjectPtr<PlacementSpecObj> n = ffi::make_object<PlacementSpecObj>();
  n->axis = -1;
  n->kind = PlacementSpecKind::kReplica;
  return PlacementSpec(ffi::UnsafeInit{}, std::move(n));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.distributed.Sharding", [](int axis) { return PlacementSpec::Sharding(axis); })
      .def("relax.distributed.Replica", []() { return PlacementSpec::Replica(); });
}

ffi::String PlacementObj::ToString() const {
  std::stringstream ss;
  for (size_t i = 0; i < dim_specs.size(); ++i) {
    if (i != 0) {
      ss << ", ";
    }
    if (dim_specs[i]->kind == PlacementSpecKind::kReplica) {
      ss << "R";
    } else {
      ss << "S[" << dim_specs[i]->axis << "]";
    }
  }
  return ss.str();
}

Placement::Placement(ffi::Array<PlacementSpec> dim_specs) {
  ffi::ObjectPtr<PlacementObj> n = ffi::make_object<PlacementObj>();
  n->dim_specs = std::move(dim_specs);
  data_ = std::move(n);
}

Placement Placement::FromText(ffi::String text_repr) {
  ffi::Array<PlacementSpec> dim_specs;
  std::stringstream ss(text_repr);
  while (true) {
    char indicator = 0;
    ss >> indicator;
    if (ss.eof()) {
      break;
    }
    if (indicator == 'R') {
      dim_specs.push_back(PlacementSpec::Replica());
    } else if (indicator == 'S') {
      char lbracket;
      ss >> lbracket;
      TVM_FFI_ICHECK_EQ(lbracket, '[');
      std::string substr;
      getline(ss, substr, ']');
      std::stringstream ss2(substr);
      int dim;
      ss2 >> dim;
      dim_specs.push_back(PlacementSpec::Sharding(dim));
      TVM_FFI_ICHECK(ss2.eof()) << "Invalid placement text repr";
    } else if (indicator == ',') {
      continue;
    } else if (indicator == ' ') {
      continue;
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid placement text repr";
    }
  }
  return Placement(dim_specs);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.distributed.PlacementFromText", Placement::FromText)
      .def("relax.distributed.Placement",
           [](ffi::Array<PlacementSpec> dim_specs) { return Placement(dim_specs); });
}

// DTensor
DTensorType::DTensorType(TensorType tensor_ty, DeviceMesh device_mesh, Placement placement,
                         Location loc)
    : Type(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK(device_mesh.defined(), ValueError) << "device_mesh must be defined";
  TVM_FFI_CHECK(placement.defined(), ValueError) << "placement must be defined";
  TVM_FFI_CHECK_EQ(device_mesh->shape.size(), placement->dim_specs.size(), ValueError)
      << "The device mesh and placement must have the same dimension size";
  for (auto spec : placement->dim_specs) {
    TVM_FFI_CHECK(spec.defined(), ValueError) << "placement specs must be defined";
    if (spec->kind == PlacementSpecKind::kReplica) continue;
    TVM_FFI_CHECK_LT(spec->axis, tensor_ty->ndim, ValueError)
        << "Sharding dimension should be smaller than tensor ndim";
  }
  ffi::ObjectPtr<DTensorTypeObj> n = ffi::make_object<DTensorTypeObj>(
      std::move(tensor_ty), std::move(device_mesh), std::move(placement));
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  DTensorTypeObj::RegisterReflection();
  refl::TypeAttrDef<DTensorTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&DTensorTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&DTensorTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&DTensorTypeMaybeInplaceMutate>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.distributed.DTensorType",
      [](TensorType tensor_ty, DeviceMesh device_mesh, Placement placement, Location loc) {
        return DTensorType(tensor_ty, device_mesh, placement, loc);
      });
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
