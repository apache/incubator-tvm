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
#include <tvm/te/operation.h>

namespace tvm::script::printer::details {
TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterScriptRepr<te::CommReducerObj>();
  RegisterScriptRepr<te::ReduceObj>();
}
namespace {
ffi::Optional<ExprDoc> CommReducerDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* reducer =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const te::CommReducerObj>(input);
  TVM_FFI_CHECK(reducer->lhs.size() == reducer->rhs.size(), TypeError)
      << "printer comm reducer arguments must be paired";
  ffi::Array<IdDoc> args;
  ffi::Array<ExprDoc> result;
  {
    for (const PrimVar& var : reducer->lhs) args.push_back(VarDoc(d, var));
    for (const PrimVar& var : reducer->rhs) args.push_back(VarDoc(d, var));
    for (const PrimExpr& value : reducer->result) result.push_back(d->Translate(value).value());
  }
  ffi::Array<ExprDoc> identity;
  for (const PrimExpr& value : reducer->identity_element)
    identity.push_back(d->Translate(value).value());
  ExprDoc result_doc = result.size() == 1 ? result[0] : ExprDoc(TupleDoc(result));
  return NamespaceDoc("s_tir")
      ->Attr("comm_reducer")
      ->Call({LambdaDoc(args, result_doc), ListDoc(identity)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<te::CommReducerObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&CommReducerDocTranslate>());
}

ffi::Optional<ExprDoc> ReduceDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object*) {
  const auto* node =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const te::ReduceObj>(input);
  // Reduction axes bind the source and predicate. Their declarations must use
  // the same identifiers as the IterVars, without becoming external captures.
  for (const s_tir::IterVar& axis : node->axis) {
    d->VarGetOrAllocId(axis->var, false);
    if (d->GetImplicitDefs().count(axis->var)) {
      IdDoc id = VarDoc(d, axis->var);
      ExprDoc value = NamespaceDoc("ir")->Attr("dynamic")->Call(
          {LiteralDoc::Str(axis->var->name, std::nullopt)}, {"dtype"},
          {LiteralDoc::DataType(axis->var.ty()->dtype, std::nullopt)});
      d->Emit(AssignDoc(id, value, std::nullopt), axis->var);
    }
  }
  ExprDoc combiner = d->Translate(node->combiner).value();
  ExprDoc source = d->Translate(node->source).value();
  ExprDoc axis = d->Translate(node->axis).value();
  ExprDoc condition = d->Translate(node->condition).value();
  ExprDoc value_index = LiteralDoc::Int(node->value_index, std::nullopt);
  ExprDoc init = d->Translate(node->init).value();
  d->RecordOrigin(source, node->source);
  d->RecordOrigin(axis, node->axis);
  d->RecordOrigin(init, node->init);
  return NamespaceDoc("s_tir")->Attr("Reduce")->Call(
      {combiner, source, axis, condition, value_index, init});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<te::ReduceObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&ReduceDocTranslate>());
}

}  // namespace
}  // namespace tvm::script::printer::details
