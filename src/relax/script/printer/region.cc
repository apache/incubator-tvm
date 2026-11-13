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
#include <tvm/tirx/type.h>

#include <optional>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<Var> BindingVar(const ffi::Object* destination) {
  if (!destination) return std::nullopt;
  TVM_FFI_CHECK(destination->IsInstance<VarObj>(), TypeError)
      << "printer binding destination must be a Var";
  return ffi::GetRef<Var>(static_cast<const VarObj*>(destination));
}

}  // namespace

ffi::Array<StmtDoc> RelaxSeqBody(DocTranslatorObj* d, const relax::SeqExprObj* seq,
                                 ffi::Optional<IdDoc> destination,
                                 ffi::Optional<ExprDoc> annotation,
                                 const ffi::Object* destination_object) {
  auto docs = d->WithDocScope([&]() {
    ffi::Optional<ExprDoc> value =
        d->Translate(ffi::GetRef<relax::SeqExpr>(seq), BindingVar(destination_object));
    if (destination.has_value()) {
      if (value.has_value()) {
        d->Emit(AssignDoc(destination.value(), value.value(), annotation),
                ffi::GetRef<ffi::ObjectRef>(seq));
      } else {
        TVM_FFI_CHECK(destination_object != nullptr, ValueError)
            << "printer Relax SeqExpr completed a binding without a destination";
      }
    } else {
      TVM_FFI_CHECK(value.has_value(), ValueError)
          << "printer Relax SeqExpr needs a value without a destination";
      d->Emit(ExprStmtDoc(value.value()), ffi::GetRef<ffi::ObjectRef>(seq));
    }
  });
  return ToStmtDocArray(docs);
}

namespace {

ffi::Optional<ExprDoc> SeqExprDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object* destination) {
  const auto* seq =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::SeqExprObj>(input);
  for (const relax::BindingBlock& block : seq->blocks) d->Translate(block);
  if (seq->blocks.empty()) {
    if (auto var = seq->body.as<Var>()) {
      d->VarGetOrAllocId(var.value(), false);
      if (d->GetImplicitDefs().count(var.value()) && !var.value()->ty.as<PrimTypeObj>()) {
        IdDoc id = VarDoc(d, var.value());
        d->Emit(AssignDoc(id, std::nullopt, d->Translate(var.value()->ty).value()),
                ffi::GetRef<ffi::ObjectRef>(var.value().get()));
      }
    }
  }
  return d->Translate(seq->body, BindingVar(destination));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::SeqExprObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&SeqExprDocTranslate>());
}

ffi::Optional<ExprDoc> BindingBlockDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object* destination) {
  const auto* block =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::BindingBlockObj>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  for (const relax::Binding& binding : block->bindings) d->Translate(binding);
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::BindingBlockObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&BindingBlockDocTranslate>());
}

ffi::Optional<ExprDoc> DataflowBlockDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object* destination) {
  const auto* block =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::DataflowBlockObj>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  auto docs = d->WithDocScope([&]() {
    ffi::Array<ExprDoc> outputs;
    for (const relax::Binding& binding : block->bindings) {
      d->Translate(binding);
      if (!binding->var.as<relax::DataflowVarObj>()) {
        outputs.push_back(d->Translate(binding->var).value());
      }
    }
    d->Emit(ExprStmtDoc(NamespaceDoc("relax")->Attr("output")->Call(outputs)),
            ffi::GetRef<ffi::ObjectRef>(block));
  });
  d->Emit(ScopeDoc(std::nullopt, NamespaceDoc("relax")->Attr("dataflow")->Call({}),
                   ToStmtDocArray(docs)),
          ffi::GetRef<ffi::ObjectRef>(block));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::DataflowBlockObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&DataflowBlockDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
