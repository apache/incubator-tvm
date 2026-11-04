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
 * \file type_functor.cc
 * \brief Implementations of Relax type functors.
 */
#include <tvm/ffi/cast.h>
#include <tvm/relax/type_functor.h>

namespace tvm {
namespace relax {

void TypeVisitor::VisitType_(const AnyTypeObj* op) {}

void TypeVisitor::VisitType_(const PrimTypeObj* op) {}

void TypeVisitor::VisitType_(const StringTypeObj* op) {}

void TypeVisitor::VisitType_(const ShapeTypeObj* op) {
  if (op->values.has_value()) {
    for (PrimExpr value : op->values.value()) {
      this->VisitTypeExprField(value);
    }
  }
}

void TypeVisitor::VisitType_(const TensorTypeObj* op) {
  if (op->shape.has_value()) {
    this->VisitTypeExprField(op->shape.value());
  }
}

void TypeVisitor::VisitType_(const distributed::DTensorTypeObj* op) {
  this->VisitType(op->tensor_ty);
}

void TypeVisitor::VisitType_(const TupleTypeObj* op) {
  for (Type field : op->fields) {
    this->VisitType(field);
  }
}

void TypeVisitor::VisitType_(const tvm::FuncTypeObj* op) {
  // Native callbacks may appear inside Relax tuples. Their signature symbols
  // and tensor contracts belong to the native function, not the Relax scope.
}

void TypeVisitor::VisitType_(const FuncTypeObj* op) {
  if (op->params.has_value()) {
    for (Type param : op->params.value()) {
      this->VisitType(param);
    }
  }
  this->VisitType(op->ret);
}

Type TypeMutator::VisitType_(const AnyTypeObj* op) { return ffi::GetRef<Type>(op); }

Type TypeMutator::VisitType_(const PrimTypeObj* op) { return ffi::GetRef<Type>(op); }

Type TypeMutator::VisitType_(const StringTypeObj* op) { return ffi::GetRef<Type>(op); }

Type TypeMutator::VisitType_(const ShapeTypeObj* op) {
  if (!op->values.has_value()) {
    return ffi::GetRef<Type>(op);
  }

  // if no changes are made the original array will be returned.
  ffi::Optional<ffi::Array<PrimExpr>> values = op->values.value().Map(
      [this](const PrimExpr& expr) { return this->VisitTypeExprField(expr); });

  if (values.same_as(op->values)) {
    return ffi::GetRef<Type>(op);
  } else {
    return ShapeType(values.value(), op->loc);
  }
}

Type TypeMutator::VisitType_(const TensorTypeObj* op) {
  if (!op->shape.has_value()) {
    return ffi::GetRef<Type>(op);
  }

  ffi::Optional<Expr> shape = this->VisitTypeExprField(op->shape.value());
  ffi::Optional<VDevice> vdev = op->vdevice;

  if (shape.same_as(op->shape)) {
    return ffi::GetRef<Type>(op);
  } else {
    return TensorType(shape.value(), op->dtype, vdev, op->loc);
  }
}

Type TypeMutator::VisitType_(const distributed::DTensorTypeObj* op) {
  TensorType tensor_ty = this->VisitType(op->tensor_ty).as_or_throw<TensorType>();
  return distributed::DTensorType(tensor_ty, op->device_mesh, op->placement);
}

Type TypeMutator::VisitType_(const TupleTypeObj* op) {
  ffi::Array<Type> fields = op->fields.Map([this](const Type& ty) { return this->VisitType(ty); });

  if (fields.same_as(op->fields)) {
    return ffi::GetRef<Type>(op);
  } else {
    return TupleType(fields, op->loc);
  }
}

Type TypeMutator::VisitType_(const tvm::FuncTypeObj* op) {
  // Relax type rewrites preserve the native callback's signature.
  return ffi::GetRef<Type>(op);
}

Type TypeMutator::VisitType_(const FuncTypeObj* op) {
  ffi::Optional<ffi::Array<Type>> params;

  if (op->params.has_value()) {
    params = op->params.value().Map([this](const Type& ty) { return this->VisitType(ty); });
  }

  Type ret = this->VisitType(op->ret);

  if (params.same_as(op->params) && ret.same_as(op->ret)) {
    return ffi::GetRef<Type>(op);
  } else {
    TVM_FFI_ICHECK(ret.defined()) << "FuncType must contain ret";
    if (params.has_value()) {
      return FuncType(params.value(), ret, op->purity, op->loc);
    } else {
      return FuncType::OpaqueFunc(ret, op->purity, op->loc);
    }
  }
}

}  // namespace relax
}  // namespace tvm
