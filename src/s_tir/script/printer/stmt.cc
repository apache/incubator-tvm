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
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/support.h>
#include <tvm/tirx/type.h>

namespace tvm {
namespace script {
namespace printer {
namespace details {
namespace {

CallDoc TensorHelperCall(DocTranslatorObj* d, const Type& type, const ffi::String& helper) {
  CallDoc rhs = d->Translate(type).value().as_or_throw<CallDoc>();
  TVM_FFI_CHECK(rhs->callee.as_or_throw<AttrAccessDoc>()->name == "Tensor", TypeError)
      << "Ts." << helper << " cannot reconstruct this nonrepresentable TensorType";
  // S-TIR's allocation and matching helpers retain their public scope/align names.
  // Translate the canonical Type fields at this sugar boundary only.
  for (size_t i = 0; i < rhs->kwargs_keys.size(); ++i) {
    if (rhs->kwargs_keys[i] == "data_alignment") rhs->kwargs_keys.Set(i, "align");
  }
  rhs->callee = NamespaceDoc("s_tir")->Attr(helper);
  return rhs;
}

void PrintMatchTensor(DocTranslatorObj* d, const s_tir::MatchTensorRegionObj* match,
                      const ffi::Array<PrimExpr>& allocated_addr = {}) {
  ExprDoc source = d->Translate(match->source).value();
  CallDoc rhs = TensorHelperCall(d, match->tensor->ty, "match_tensor");
  rhs->args.insert(rhs->args.begin(), source);
  if (!allocated_addr.empty()) {
    rhs->kwargs_keys.push_back("allocated_addr");
    rhs->kwargs_values.push_back(d->Translate(allocated_addr).value());
  }
  IdDoc lhs = VarDoc(d, match->tensor);
  d->Emit(AssignDoc(lhs, rhs, std::nullopt), ffi::GetRef<ffi::ObjectRef>(match));
}

ffi::Array<StmtDoc> SBlockBody(DocTranslatorObj* d, const s_tir::SBlockObj* block,
                               const s_tir::SBlockRealizeObj* realize) {
  auto docs = d->WithDocScope([&]() {
    static const auto scope_shared = runtime::MemoryScope::Get("shared");
    static const auto scope_shared_dyn = runtime::MemoryScope::Get("shared.dyn");
    static const auto scope_local = runtime::MemoryScope::Get("local");

    TVM_FFI_CHECK(!realize || realize->iter_values.size() == block->iter_vars.size(), TypeError)
        << "SBlockRealize binding count must match its iter vars";
    ffi::Array<AssignDoc> axes;
    for (size_t i = 0; i < block->iter_vars.size(); ++i) {
      const s_tir::IterVar& iter = block->iter_vars[i];
      ffi::String kind;
      switch (iter->iter_type) {
        case s_tir::IterVarType::kDataPar:
          kind = "spatial";
          break;
        case s_tir::IterVarType::kCommReduce:
          kind = "reduce";
          break;
        case s_tir::IterVarType::kOrdered:
          kind = "scan";
          break;
        case s_tir::IterVarType::kOpaque:
          kind = "opaque";
          break;
        default:
          TVM_FFI_THROW(TypeError) << "printer unsupported SBlock iter var kind";
      }
      ExprDoc domain = prim::IsZero(iter->dom.value()->begin) &&
                               iter->dom.value()->begin.ty() == iter->dom.value()->extent.ty()
                           ? d->Translate(iter->dom.value()->extent).value()
                           : NamespaceDoc("ir")
                                 ->Attr("Range")
                                 ->Attr("from_min_extent")
                                 ->Call({d->Translate(iter->dom.value()->begin).value(),
                                         d->Translate(iter->dom.value()->extent).value()});
      ffi::Array<ExprDoc> args = {domain};
      if (realize) args.push_back(d->Translate(realize->iter_values[i]).value());
      IdDoc lhs = VarDoc(d, iter->var);
      ffi::Array<ffi::String> keys;
      ffi::Array<ExprDoc> values;
      if (iter->var.ty() != PrimType::Int(32)) {
        keys.push_back("dtype");
        values.push_back(
            LiteralDoc::Str(ffi::DLDataTypeToString(iter->var.ty()->dtype), std::nullopt));
      }
      axes.push_back(
          AssignDoc(lhs, NamespaceDoc("s_tir")->Attr("axis")->Attr(kind)->Call(args, keys, values),
                    std::nullopt));
    }
    for (size_t i = 0; i < axes.size(); ++i) d->Emit(axes[i], block->iter_vars[i]);
    if (realize && !tvm::prim::IsOne(realize->predicate)) {
      d->Emit(ExprStmtDoc(NamespaceDoc("s_tir")->Attr("where")->Call(
                  {d->Translate(realize->predicate).value()})),
              realize->predicate);
    }
    ffi::Array<ExprDoc> reads;
    for (const TensorRegion& region : block->reads) reads.push_back(d->Translate(region).value());
    d->Emit(ExprStmtDoc(NamespaceDoc("s_tir")->Attr("reads")->Call(reads)), block->reads);
    ffi::Array<ExprDoc> writes;
    for (const TensorRegion& region : block->writes) writes.push_back(d->Translate(region).value());
    d->Emit(ExprStmtDoc(NamespaceDoc("s_tir")->Attr("writes")->Call(writes)), block->writes);
    auto annotations = block->annotations;
    ffi::Map<Var, ffi::Array<PrimExpr>> addresses;
    if (auto value = annotations.Get(s_tir::attr::kTensorAllocatedAddr.name)) {
      for (const auto& entry : value.value().cast<s_tir::TensorAllocatedAddresses>()) {
        addresses.Set(entry.get<0>(), entry.get<1>());
      }
      annotations.erase(s_tir::attr::kTensorAllocatedAddr.name);
    }
    if (!annotations.empty()) {
      d->Emit(ExprStmtDoc(NamespaceDoc("s_tir")
                              ->Attr("sblock_attr")
                              ->Call({d->Translate(annotations).value()})),
              block->annotations);
    }
    for (const tirx::TensorVar& tensor : block->alloc_tensors) {
      CallDoc rhs = TensorHelperCall(d, tensor->ty, "sblock_alloc_tensor");
      const auto* tensor_type = tensor->ty.as<tirx::TensorTypeObj>();
      auto allocated_addr = addresses.Get(tensor).value_or(ffi::Array<PrimExpr>{});
      TVM_FFI_CHECK(
          allocated_addr.empty() ||
              (tensor_type->scope != runtime::MemoryScope::Global() &&
               tensor_type->scope != scope_shared && tensor_type->scope != scope_shared_dyn &&
               tensor_type->scope != scope_local),
          TypeError)
          << "Ts.sblock_alloc_tensor does not accept allocated_addr in " << tensor_type->scope;
      if (!allocated_addr.empty()) {
        rhs->kwargs_keys.push_back("allocated_addr");
        rhs->kwargs_values.push_back(d->Translate(allocated_addr).value());
      }
      IdDoc lhs = VarDoc(d, tensor);
      d->Emit(AssignDoc(lhs, rhs, std::nullopt), ffi::GetRef<ffi::ObjectRef>(tensor.get()));
    }
    for (const s_tir::MatchTensorRegion& match : block->match_tensors) {
      PrintMatchTensor(d, match.get(),
                       addresses.Get(match->tensor).value_or(ffi::Array<PrimExpr>{}));
    }
    if (block->init.has_value()) {
      d->Emit(
          ScopeDoc(std::nullopt, NamespaceDoc("s_tir")->Attr("init")->Call({}),
                   ToStmtDocArray(d->WithDocScope([&]() { d->Translate(block->init.value()); }))),
          block->init.value());
    }
    d->Translate(block->body);
  });
  return ToStmtDocArray(docs);
}

ffi::Optional<ExprDoc> SBlockRealizeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object* destination) {
  const auto* realize =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const s_tir::SBlockRealizeObj>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  const s_tir::SBlockObj* block = realize->block.get();
  d->Emit(ScopeDoc(std::nullopt,
                   NamespaceDoc("s_tir")->Attr("sblock")->Call(
                       {LiteralDoc::Str(block->name_hint, std::nullopt)}),
                   SBlockBody(d, block, realize)),
          ffi::GetRef<ffi::ObjectRef>(realize));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<s_tir::SBlockRealizeObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&SBlockRealizeDocTranslate>());
}

ffi::Optional<ExprDoc> SBlockDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object* destination) {
  const auto* block =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const s_tir::SBlockObj>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  d->Emit(ScopeDoc(std::nullopt,
                   NamespaceDoc("s_tir")->Attr("sblock")->Call(
                       {LiteralDoc::Str(block->name_hint, std::nullopt)}, {"no_realize"},
                       {LiteralDoc::Boolean(true, std::nullopt)}),
                   SBlockBody(d, block, nullptr)),
          ffi::GetRef<ffi::ObjectRef>(block));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<s_tir::SBlockObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&SBlockDocTranslate>());
}

ffi::Optional<ExprDoc> MatchTensorRegionDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                     const ffi::Object* destination) {
  const auto* match =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const s_tir::MatchTensorRegionObj>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  PrintMatchTensor(d, match);
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<s_tir::MatchTensorRegionObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&MatchTensorRegionDocTranslate>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterScriptRepr<s_tir::MatchTensorRegionObj>();
  RegisterScriptRepr<s_tir::SBlockObj>();
  RegisterScriptRepr<s_tir::SBlockRealizeObj>();
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
