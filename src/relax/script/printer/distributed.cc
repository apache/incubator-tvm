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
#include <tvm/relax/distributed/type.h>
#include <tvm/script/printer/doc_translator.h>

#include <optional>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> PlacementDocTranslate(DocTranslatorObj*, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* placement = ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<
      const relax::distributed::PlacementObj>(input);
  return LiteralDoc::Str(placement->ToString(), std::nullopt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::distributed::PlacementObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&PlacementDocTranslate>());
}

ffi::Optional<ExprDoc> DTensorTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* ty = ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<
      const relax::distributed::DTensorTypeObj>(input);
  ffi::Array<ExprDoc> args;
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  const relax::TensorType& tensor = ty->tensor_ty;
  bool require_keywords = !tensor->shape.has_value();
  if (tensor->shape.has_value()) {
    if (auto shape = tensor->shape.value().as<relax::ShapeExprObj>()) {
      ffi::Array<ExprDoc> dimensions;
      for (const PrimExpr& dim : shape->values) dimensions.push_back(RelaxShapeDim(d, dim));
      args.push_back(TupleDoc(dimensions));
    } else {
      args.push_back(d->Translate(tensor->shape.value()).value());
    }
  }
  if (!tensor->IsUnknownDtype()) {
    ExprDoc dtype = LiteralDoc::DataType(tensor->dtype.value()->dtype, std::nullopt);
    if (require_keywords) {
      keys.push_back("dtype");
      values.push_back(dtype);
    } else {
      args.push_back(dtype);
    }
  } else {
    require_keywords = true;
  }
  ExprDoc mesh = d->Translate(ty->device_mesh).value();
  if (auto selector = GlobalInfoSelector(d, ty->device_mesh)) {
    mesh = LiteralDoc::Str(selector.value(), std::nullopt);
  }
  ExprDoc placement = d->Translate(ty->placement).value();
  if (require_keywords) {
    keys.push_back("device_mesh");
    values.push_back(mesh);
    keys.push_back("placement");
    values.push_back(placement);
  } else {
    args.push_back(mesh);
    args.push_back(placement);
  }
  if (!tensor->shape.has_value() && !tensor->IsUnknownNdim()) {
    keys.push_back("ndim");
    values.push_back(LiteralDoc::Int(tensor->ndim, std::nullopt));
  }
  return NamespaceDoc("relax")->Attr("DTensor")->Call(args, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::distributed::DTensorTypeObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&DTensorTypeDocTranslate>());
}

ffi::Optional<ExprDoc> DeviceMeshDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* mesh = ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<
      const relax::distributed::DeviceMeshObj>(input);
  ffi::Array<ExprDoc> dimensions;
  for (int64_t value : mesh->shape) {
    dimensions.push_back(LiteralDoc::Int(value, std::nullopt));
  }
  ExprDoc devices = LiteralDoc::None(std::nullopt);
  if (mesh->device_range.has_value()) {
    devices = d->Translate(mesh->device_range.value()).value();
  } else {
    ffi::Array<ExprDoc> ids;
    for (int64_t value : mesh->device_ids) ids.push_back(LiteralDoc::Int(value, std::nullopt));
    devices = ListDoc(ids);
  }
  return NamespaceDoc("relax")->Attr("device_mesh")->Call({TupleDoc(dimensions), devices});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::distributed::DeviceMeshObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&DeviceMeshDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
