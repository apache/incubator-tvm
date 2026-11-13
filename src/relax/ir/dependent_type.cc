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
 * \file src/relax/ir/dependent_type.cc
 * \brief Relax type nodes.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_functor.h>

namespace tvm {
namespace relax {
using namespace tvm::prim;

namespace {

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> ShapeTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  const ShapeTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeTypeObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->values));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ShapeTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  const ShapeTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeTypeObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<ffi::Array<PrimExpr>>>,
                                    mapped_values, mutator->MutateExpected(self->values));
  if (mapped_values.UnchangedOrSameAs(self->values)) return ffi::Unchanged();
  ffi::ObjectPtr<ShapeTypeObj> copy = ffi::make_object<ShapeTypeObj>(*self);
  copy->values = std::move(mapped_values).ValueOrUnchanged(std::move(copy->values));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> ShapeTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  ShapeTypeObj* self = const_cast<ShapeTypeObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeTypeObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<ffi::Array<PrimExpr>>>, mapped_values,
      mutator->MutateExpected(self->values, ffi::InplaceMode::kAllow));
  if (!mapped_values.UnchangedOrSameAs(self->values)) {
    self->values = std::move(mapped_values).ValueUnchecked();
  }
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TensorTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  const TensorTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->shape));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->dtype));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->vdevice));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  const TensorTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Expr>>, mapped_shape,
                                    mutator->MutateExpected(self->shape));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimType>>, mapped_dtype,
                                    mutator->MutateExpected(self->dtype));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<VDevice>>, mapped_vdevice,
                                    mutator->MutateExpected(self->vdevice));
  if (mapped_shape.UnchangedOrSameAs(self->shape) && mapped_dtype.UnchangedOrSameAs(self->dtype) &&
      mapped_vdevice.UnchangedOrSameAs(self->vdevice)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<TensorTypeObj> copy = ffi::make_object<TensorTypeObj>(*self);
  copy->shape = std::move(mapped_shape).ValueOrUnchanged(std::move(copy->shape));
  copy->dtype = std::move(mapped_dtype).ValueOrUnchanged(std::move(copy->dtype));
  copy->vdevice = std::move(mapped_vdevice).ValueOrUnchanged(std::move(copy->vdevice));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TensorTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  TensorTypeObj* self = const_cast<TensorTypeObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Expr>>, mapped_shape,
                                    mutator->MutateExpected(self->shape, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimType>>, mapped_dtype,
                                    mutator->MutateExpected(self->dtype, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<VDevice>>, mapped_vdevice,
      mutator->MutateExpected(self->vdevice, ffi::InplaceMode::kAllow));
  if (!mapped_shape.UnchangedOrSameAs(self->shape)) {
    self->shape = std::move(mapped_shape).ValueUnchecked();
  }
  if (!mapped_dtype.UnchangedOrSameAs(self->dtype)) {
    self->dtype = std::move(mapped_dtype).ValueUnchecked();
  }
  if (!mapped_vdevice.UnchangedOrSameAs(self->vdevice)) {
    self->vdevice = std::move(mapped_vdevice).ValueUnchecked();
  }
  return ffi::Unchanged();
}

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> FuncTypeVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: derive_func (environment-backed callable metadata), purity (scalar)
  const FuncTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeObj>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindPattern, [&]() { return visitor->VisitExpected(self->params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ret));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FuncTypeMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: derive_func (environment-backed callable metadata), purity (scalar)
  const FuncTypeObj* self =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeObj>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<ffi::Array<Type>>>, mapped_params,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern,
                                 [&]() { return mutator->MutateExpected(self->params); }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret,
                                    mutator->MutateExpected(self->ret));
  if (mapped_params.UnchangedOrSameAs(self->params) && mapped_ret.UnchangedOrSameAs(self->ret)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<FuncTypeObj> copy = ffi::make_object<FuncTypeObj>(*self);
  copy->params = std::move(mapped_params).ValueOrUnchanged(std::move(copy->params));
  copy->ret = std::move(mapped_ret).ValueOrUnchanged(std::move(copy->ret));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FuncTypeMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: derive_func (environment-backed callable metadata), purity (scalar)
  FuncTypeObj* self = const_cast<FuncTypeObj*>(
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeObj>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<ffi::Array<Type>>>, mapped_params,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
        return mutator->MutateExpected(self->params, ffi::InplaceMode::kAllow);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret,
                                    mutator->MutateExpected(self->ret, ffi::InplaceMode::kAllow));
  if (!mapped_params.UnchangedOrSameAs(self->params)) {
    self->params = std::move(mapped_params).ValueUnchecked();
  }
  if (!mapped_ret.UnchangedOrSameAs(self->ret)) {
    self->ret = std::move(mapped_ret).ValueUnchecked();
  }
  return ffi::Unchanged();
}

}  // namespace

// Shape
ShapeType::ShapeType(ffi::Array<PrimExpr> values, Location loc) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ShapeTypeObj> n = ffi::make_object<ShapeTypeObj>();
  n->ndim = static_cast<int>(values.size());
  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmObj>()) {
      return tvm::prim::cast(value, PrimType::Int(64));
    }
    TVM_FFI_ICHECK(value.ty().MatchesElementType(DLDataTypeCode::kDLInt, 64))
        << "the value in ShapeType can only have dtype of int64";
    return value;
  });
  n->loc = loc;
  data_ = std::move(n);
}

ShapeType::ShapeType(int ndim, Location loc) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ShapeTypeObj> n = ffi::make_object<ShapeTypeObj>();
  TVM_FFI_ICHECK(ndim >= -1) << "ndim of ShapeType must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  ShapeTypeObj::RegisterReflection();
  refl::TypeAttrDef<ShapeTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&ShapeTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&ShapeTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&ShapeTypeMaybeInplaceMutate>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.ShapeType", [](ffi::Optional<ffi::Array<PrimExpr>> values, int ndim, Location loc) {
        if (values.has_value()) {
          TVM_FFI_CHECK_EQ(ndim, kUnknownNDim, ValueError) << "Cannot both specify values and ndim";
          return ShapeType(values.value(), loc);
        } else {
          return ShapeType(ndim, loc);
        }
      });
}

// Tensor
TensorType::TensorType(Expr shape, ffi::Optional<PrimType> dtype, ffi::Optional<VDevice> vdevice,
                       Location loc)
    : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TensorTypeObj> n = ffi::make_object<TensorTypeObj>();
  // assign ndim before move
  TVM_FFI_ICHECK(shape.defined()) << "Must provide a shape in this constructor";
  ffi::Optional<ShapeType> shape_ty = MatchType<ShapeType>(shape);
  TVM_FFI_ICHECK(shape_ty) << "We expect shape to contain pre-set shape type";
  TVM_FFI_ICHECK(shape->IsInstance<ShapeExprObj>() || shape->IsInstance<VarObj>())
      << "We require shape to be normalized when constructing TensorType";
  n->ndim = shape_ty.value()->ndim;
  // assign rest of the fields.
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->vdevice = vdevice;
  n->loc = loc;
  data_ = std::move(n);
}

TensorType::TensorType(ffi::Optional<PrimType> dtype, int ndim, ffi::Optional<VDevice> vdevice,
                       Location loc)
    : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TensorTypeObj> n = ffi::make_object<TensorTypeObj>();
  TVM_FFI_ICHECK(ndim >= -1) << "ndim of TensorType must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->dtype = dtype;
  n->vdevice = vdevice;
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorTypeObj::RegisterReflection();
  refl::TypeAttrDef<TensorTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&TensorTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TensorTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TensorTypeMaybeInplaceMutate>())
      .attr(tvm::type_attr::kExprSubscriptRealizeByTy,
            [](Expr value, SubscriptSlice slice, Location loc) -> ffi::ObjectRef {
              TVM_FFI_CHECK_EQ(slice.size(), 1, IndexError)
                  << "A Relax expression requires exactly one index";
              auto index = slice[0].as<PrimExpr>();
              TVM_FFI_CHECK(index.has_value(), TypeError)
                  << "A Relax expression requires a point index";
              const auto* imm = index.value().as<IntImmObj>();
              TVM_FFI_CHECK(imm != nullptr, TypeError)
                  << "A Relax expression requires a constant integer index";
              return TupleGetItem(value, imm->value.as<int>().value(), loc);
            });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.TensorType", [](ffi::Optional<Expr> shape, ffi::Optional<PrimType> dtype, int ndim,
                             ffi::Optional<VDevice> vdevice, Location loc) {
        if (shape.has_value()) {
          TVM_FFI_CHECK_EQ(ndim, kUnknownNDim, ValueError) << "Cannot both specify shape and ndim";
          return TensorType(shape.value(), dtype, vdevice, loc);
        } else {
          return TensorType(dtype, ndim, vdevice, loc);
        }
      });
}

// Func
FuncType::FuncType(ffi::Array<Type> params, Type ret, bool purity, Location loc)
    : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<FuncTypeObj> n = ffi::make_object<FuncTypeObj>();
  n->params = std::move(params);
  n->ret = std::move(ret);
  n->purity = std::move(purity);
  n->loc = loc;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FuncTypeObj::RegisterReflection();
  refl::TypeAttrDef<FuncTypeObj>()
      .attr(ffi::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&FuncTypeVisit>())
      .attr(ffi::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&FuncTypeMutate>())
      .attr(ffi::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&FuncTypeMaybeInplaceMutate>());

  refl::GlobalDef().def("relax.FuncType",
                        [](ffi::Array<Type> params, Type ret, bool purity, Location loc) {
                          return FuncType(params, ret, purity, loc);
                        });
}

FuncType FuncType::OpaqueFunc(TypeDeriveFunc derive_func, bool purity, Location loc) {
  ffi::ObjectPtr<FuncTypeObj> n = ffi::make_object<FuncTypeObj>();
  n->derive_func = std::move(derive_func);
  n->ret = AnyType();
  n->purity = std::move(purity);
  n->loc = loc;
  return FuncType(n);
}

FuncType FuncType::OpaqueFunc(Type ret, bool purity, Location loc) {
  ffi::ObjectPtr<FuncTypeObj> n = ffi::make_object<FuncTypeObj>();
  n->ret = std::move(ret);
  n->purity = std::move(purity);
  n->loc = loc;
  return FuncType(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.FuncTypeOpaqueFunc", [](ffi::Optional<Type> ret,
                                                       ffi::Optional<TypeDeriveFunc> derive_func,
                                                       bool purity, Location loc) {
    if (derive_func.has_value()) {
      TVM_FFI_CHECK(!ret.has_value(), ValueError) << "Cannot specify both ret and derive_func";
      return FuncType::OpaqueFunc(derive_func.value(), purity, loc);
    } else {
      return FuncType::OpaqueFunc(ret.value_or(AnyType()), purity, loc);
    }
  });
}

// Helper functions
void UpdateType(Expr expr, Type ty) {
  TVM_FFI_ICHECK(expr->ty.as<MissingType>().has_value())
      << "To ensure idempotency, "
      << "the expression passed to UpdateType "
      << "must not have any prior type.  "
      << "However, expression " << expr << " has type " << expr->ty
      << ", which cannot be overwritten with " << ty;
  expr->ty = ty;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.UpdateType", [](Expr expr, Type ty) { UpdateType(expr, ty); })
      .def("relax.ExprType", [](Expr expr) { return GetType(expr); });
}

}  // namespace relax
}  // namespace tvm
