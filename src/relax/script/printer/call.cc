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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/op.h>
#include <tvm/relax/expr.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/support.h>

#include <optional>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {
namespace {

// These semantic wrappers convert bare Python scalar tuple fields to 64-bit
// values. Use raw construction when ordinary tuple syntax would change the IR.
bool InlineTupleNeedsRawCall(DocTranslatorObj* d, const Expr& expr) {
  if (const auto* tuple = expr.as<TupleObj>()) {
    for (const Expr& field : tuple->fields) {
      if (InlineTupleNeedsRawCall(d, field)) return true;
    }
  } else if (const auto* imm = expr.as<IntImmObj>()) {
    DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
    auto implicit =
        ffi::StringToDLDataType(d->GetExtraConfig<ffi::String>("ir.int_dtype", "int32"));
    return dtype == implicit && dtype != ffi::StringToDLDataType("int64");
  } else if (const auto* imm = expr.as<FloatImmObj>()) {
    DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
    auto implicit =
        ffi::StringToDLDataType(d->GetExtraConfig<ffi::String>("ir.float_dtype", "void"));
    return dtype == implicit && dtype != ffi::StringToDLDataType("float64");
  }
  return false;
}

ffi::Optional<ExprDoc> InlineTupleCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                   const ffi::Object*) {
  const auto* call =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(input);
  if (call->args.size() > 1 && InlineTupleNeedsRawCall(d, call->args[1]))
    return RawCallDocTranslate(d, call);
  if (auto doc = StandardCallDocTranslate(d, call)) return doc;
  return RawCallDocTranslate(d, call);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"relax.call_builtin_with_ctx", "relax.make_closure",
                           "relax.invoke_closure", "relax.invoke_pure_closure"}) {
    OpDef(name).set_native_fn_attr<&InlineTupleCallDocTranslate>(
        tvm::script::printer::op_attr::kDocTranslateOpCall);
  }
}

// Script logical operators also serve as primitive aliases. A Relax Call
// containing a primitive operand must not take that alternate construction path.
ffi::Optional<ExprDoc> LogicalCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* call =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(input);
  for (const Expr& arg : call->args) {
    if (arg->ty.as<PrimType>()) return RawCallDocTranslate(d, call);
  }
  if (auto doc = StandardCallDocTranslate(d, call)) return doc;
  return RawCallDocTranslate(d, call);
}

// VM allocation wrappers convert bare integer offsets/device indices to int64.
// Keep explicit construction when the renderer would hide a different width.
ffi::Optional<ExprDoc> VMAllocCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* call =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(input);
  auto needs_raw = [&](size_t index) {
    if (index >= call->args.size()) return false;
    const auto* imm = call->args[index].as<IntImmObj>();
    if (!imm) return false;
    DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
    auto implicit =
        ffi::StringToDLDataType(d->GetExtraConfig<ffi::String>("ir.int_dtype", "int32"));
    return dtype == implicit && dtype != ffi::StringToDLDataType("int64") &&
           dtype != ffi::StringToDLDataType("bool");
  };
  if (needs_raw(1) || needs_raw(4)) return RawCallDocTranslate(d, call);
  if (auto doc = StandardCallDocTranslate(d, call)) return doc;
  return RawCallDocTranslate(d, call);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"relax.logical_and", "relax.logical_or", "relax.logical_not"}) {
    OpDef(name).set_native_fn_attr<&LogicalCallDocTranslate>(
        tvm::script::printer::op_attr::kDocTranslateOpCall);
  }
  for (const char* name : {"relax.vm.alloc_storage", "relax.vm.alloc_tensor"}) {
    OpDef(name).set_native_fn_attr<&VMAllocCallDocTranslate>(
        tvm::script::printer::op_attr::kDocTranslateOpCall);
  }
}

ffi::Optional<ExprDoc> CallDPSPackedDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object*) {
  const auto* call =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallObj>(input);
  if (call->attrs.has_value() || call->args.size() != 2 || !call->args[1].as<TupleObj>() ||
      call->ty_args.size() != 1 || InlineTupleNeedsRawCall(d, call->args[1])) {
    return RawCallDocTranslate(d, call);
  }
  ExprDoc callee = d->Translate(call->args[0]).value();
  if (const auto* packed = call->args[0].as<relax::ExternFuncObj>()) {
    if (!ffi::StructuralEqual()(packed->ty, relax::ExternFunc(packed->global_symbol)->ty)) {
      return RawCallDocTranslate(d, call);
    }
    callee = LiteralDoc::Str(packed->global_symbol, std::nullopt);
  } else if (call->args[0].as<StringImmObj>()) {
    // A bare Python string now constructs ExternFunc, so retain this explicit IR value.
    callee = d->WithExtraStateScope<bool>("ir.explicit_value_type", true,
                                          [&] { return d->Translate(call->args[0]).value(); });
  }
  d->RecordOrigin(callee, call->args[0]);
  ExprDoc args = d->Translate(call->args[1]).value();
  ffi::Array<ffi::String> keys = {"ty_args"};
  ffi::Array<ExprDoc> values = {ListDoc({d->Translate(call->ty_args[0]).value()})};
  return NamespaceDoc("relax")->Attr("call_dps_packed")->Call({callee, args}, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_dps_packed")
      .set_native_fn_attr<&CallDPSPackedDocTranslate>(
          tvm::script::printer::op_attr::kDocTranslateOpCall);
}

}  // namespace
}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
