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

#include "./pattern_registry.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace relax {
namespace backend {
static std::vector<FusionPattern>* GetRegistryTable() {
  static std::vector<FusionPattern> table;
  return &table;
}

void RegisterPatterns(ffi::Array<FusionPattern> entries) {
  auto* table = GetRegistryTable();
  for (const auto& entry : entries) {
    table->push_back(entry);
  }
}

void RemovePatterns(ffi::Array<ffi::String> names) {
  std::unordered_set<ffi::String> name_set{names.begin(), names.end()};

  auto* table = GetRegistryTable();
  table->erase(
      std::remove_if(table->begin(), table->end(),
                     [&](const FusionPattern& entry) { return name_set.count(entry->name) > 0; }),
      table->end());
}

ffi::Array<FusionPattern> GetPatternsWithPrefix(const ffi::String& prefix) {
  auto* table = GetRegistryTable();
  ffi::Array<FusionPattern> result;
  for (auto it = table->rbegin(); it != table->rend(); ++it) {
    if ((*it)->name.starts_with(prefix.data())) {
      result.push_back(*it);
    }
  }
  return result;
}

ffi::Optional<FusionPattern> GetPattern(const ffi::String& pattern_name) {
  auto* table = GetRegistryTable();
  for (auto it = table->rbegin(); it != table->rend(); ++it) {
    if ((*it)->name == pattern_name) {
      return *it;
    }
  }
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.backend.RegisterPatterns", RegisterPatterns)
      .def("relax.backend.RemovePatterns", RemovePatterns)
      .def("relax.backend.GetPatternsWithPrefix", GetPatternsWithPrefix)
      .def("relax.backend.GetPattern", GetPattern);
}

}  // namespace backend
}  // namespace relax
}  // namespace tvm
