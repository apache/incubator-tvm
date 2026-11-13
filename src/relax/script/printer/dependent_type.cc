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

#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/support.h>
#include <tvm/tirx/type.h>

#include <optional>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> ShapeTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* ty =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::ShapeTypeObj>(input);
  if (ty->values.has_value()) {
    ffi::Array<ExprDoc> values;
    for (const PrimExpr& value : ty->values.value()) values.push_back(RelaxShapeDim(d, value));
    return NamespaceDoc("relax")->Attr("Shape")->Call({ListDoc(values)});
  }
  return NamespaceDoc("relax")->Attr("Shape")->Call({}, {"ndim"},
                                                    {LiteralDoc::Int(ty->ndim, std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::ShapeTypeObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&ShapeTypeDocTranslate>());
}

ffi::Optional<ExprDoc> TensorTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* ty =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::TensorTypeObj>(input);
  ffi::Array<ExprDoc> args;
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (ty->shape.has_value()) {
    if (auto shape = ty->shape.value().as<relax::ShapeExprObj>()) {
      ffi::Array<ExprDoc> dimensions;
      for (const PrimExpr& dim : shape->values) dimensions.push_back(RelaxShapeDim(d, dim));
      args.push_back(TupleDoc(dimensions));
    } else {
      args.push_back(d->Translate(ty->shape.value()).value());
    }
  }
  if (!ty->IsUnknownDtype()) {
    keys.push_back("dtype");
    values.push_back(LiteralDoc::DataType(ty->dtype.value()->dtype, std::nullopt));
  }
  if (!ty->shape.has_value() && !ty->IsUnknownNdim()) {
    keys.push_back("ndim");
    values.push_back(LiteralDoc::Int(ty->ndim, std::nullopt));
  }
  if (ty->vdevice.has_value()) {
    keys.push_back("vdevice");
    if (auto selector = GlobalInfoSelector(d, ty->vdevice.value())) {
      values.push_back(LiteralDoc::Str(selector.value(), std::nullopt));
    } else {
      values.push_back(d->Translate(ty->vdevice.value()).value());
    }
  }
  return NamespaceDoc("relax")->Attr("Tensor")->Call(args, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::TensorTypeObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&TensorTypeDocTranslate>());
}

ffi::Optional<ExprDoc> RelaxFuncTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object*) {
  const auto* ty =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::FuncTypeObj>(input);
  if (!ty->params.has_value()) {
    ffi::Array<ffi::String> keys;
    ffi::Array<ExprDoc> values;
    if (!ty->ret.as<AnyTypeObj>()) {
      keys.push_back("ret");
      values.push_back(d->Translate(ty->ret).value());
    }
    if (ty->purity) {
      keys.push_back("purity");
      values.push_back(LiteralDoc::Boolean(true, std::nullopt));
    }
    if (ty->derive_func.has_value()) {
      keys.push_back("derive_func");
      values.push_back(LiteralDoc::Str(ty->derive_func.value()->name, std::nullopt));
    }
    ExprDoc doc = NamespaceDoc("relax")->Attr("Callable");
    return doc->Call({}, keys, values);
  }
  ffi::Array<ExprDoc> params;
  for (const Type& param : ty->params.value()) params.push_back(d->Translate(param).value());
  return NamespaceDoc("relax")
      ->Attr("Callable")
      ->Call({TupleDoc(params), d->Translate(ty->ret).value(),
              LiteralDoc::Boolean(ty->purity, std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::FuncTypeObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&RelaxFuncTypeDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
