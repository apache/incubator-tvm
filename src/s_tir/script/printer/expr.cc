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
#include <tvm/ir/op.h>
#include <tvm/s_tir/iter_var.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/support.h>

namespace tvm {
namespace script {
namespace printer {
namespace details {

TVM_FFI_STATIC_INIT_BLOCK() { RegisterScriptRepr<s_tir::IterVarObj>(); }
namespace {

ffi::Optional<ExprDoc> IterVarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object*) {
  const auto* iter =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const s_tir::IterVarObj>(input);
  ExprDoc domain = d->Translate(iter->dom).value();
  if (iter->dom.has_value()) {
    domain.as_or_throw<CallDoc>()->callee = NamespaceDoc("tirx")->Attr("Range");
  }
  return NamespaceDoc("s_tir")
      ->Attr("iter_var")
      ->Call({d->Translate(iter->var).value(), domain,
              LiteralDoc::Str(s_tir::IterVarType2String(iter->iter_type), std::nullopt),
              LiteralDoc::Str(iter->thread_tag, std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<s_tir::IterVarObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&IterVarDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
