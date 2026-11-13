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
#ifndef TVM_S_TIR_MEMORY_SCOPE_H_
#define TVM_S_TIR_MEMORY_SCOPE_H_

#include <tvm/runtime/memory_scope.h>

namespace tvm {
namespace s_tir {

// Backend scopes may be registered after a compiler pass is initialized.  Look
// up optional identities without caching absence or registering unknown names.
inline ffi::Optional<runtime::MemoryScope> GetRegisteredMemoryScope(const ffi::String& name) {
  static ffi::reflection::TypeAttrColumn states(ffi::type_attr::kEnumState);
  auto state = states[runtime::MemoryScopeObj::RuntimeTypeIndex()].cast<ffi::EnumState>();
  if (auto entry = state->indexes.Get(name)) return entry->as_or_throw<runtime::MemoryScope>();
  return std::nullopt;
}

inline bool IsGlobalMemoryScope(const runtime::MemoryScope& scope) {
  if (scope == runtime::MemoryScope::Global()) return true;
  static const ffi::String names[] = {
      "global.ddr",        "global.vtcm",         "global.workspace",    "global.texture",
      "global.texture-1d", "global.texture-nchw", "global.texture-nhwc", "global.texture-weight"};
  for (const auto& name : names) {
    if (auto registered = GetRegisteredMemoryScope(name)) {
      if (scope == *registered) return true;
    }
  }
  return false;
}

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_MEMORY_SCOPE_H_
