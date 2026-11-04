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
#include <tvm/ir/module.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/target/target.h>

#include <optional>
#include <string>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

// Selectors are derived from the module's forward registry, never cached as
// object-to-string state. Definitions and standalone objects keep constructors.
ffi::Optional<ffi::String> GlobalInfoSelector(DocTranslatorObj* d, const GlobalInfo& info) {
  auto module = d->GetOrCreateExtraState<ffi::Optional<IRModule>>("ir.module");
  if (!module.has_value()) return std::nullopt;
  const auto& infos = module.value()->global_infos;
  if (auto query = info.as<relax::VDevice>()) {
    if (!query.value()->target.has_value()) return std::nullopt;
    if (auto devices = infos.Get("vdevice")) {
      ffi::String kind = query.value()->target.value()->kind->name;
      size_t index = 0;
      for (const GlobalInfo& entry : devices.value()) {
        if (auto device = entry.as<relax::VDevice>();
            device && device.value()->target.has_value() &&
            device.value()->target.value()->kind->name == kind) {
          if (entry.same_as(info)) {
            return ffi::String(std::string(kind) + ":" + std::to_string(index));
          }
          ++index;
        }
      }
    }
  }
  if (auto meshes = infos.Get("mesh")) {
    size_t index = 0;
    for (const GlobalInfo& entry : meshes.value()) {
      if (entry.same_as(info)) return ffi::String("mesh[" + std::to_string(index) + "]");
      ++index;
    }
  }
  return std::nullopt;
}

namespace {

ffi::Optional<ExprDoc> VDeviceDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object*) {
  const auto* device =
      ffi::detail::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::VDeviceObj>(input);
  return NamespaceDoc("relax")->Attr("vdevice")->Call(
      {d->Translate(device->target).value()}, {"vdevice_id", "memory_scope"},
      {LiteralDoc::Int(device->vdevice_id, std::nullopt),
       LiteralDoc::Str(device->memory_scope.name(), std::nullopt)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::VDeviceObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&VDeviceDocTranslate>());
}

ffi::Optional<ExprDoc> DummyGlobalInfoDocTranslate(DocTranslatorObj*, ffi::AnyView,
                                                   const ffi::Object*) {
  return NamespaceDoc("relax")->Attr("dummy_global_info")->Call({});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::DummyGlobalInfoObj>().attr(
      tvm::script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&DummyGlobalInfoDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
