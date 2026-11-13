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

#include <gtest/gtest.h>
#include <tvm/runtime/memory_scope.h>

#include "../../../src/s_tir/schedule/utils.h"

namespace tvm {
namespace s_tir {

TEST(MemoryScopePolicy, GlobalSpacesRelaxAcrossBlocksAndThreads) {
  auto block = ParseThreadBindingTag("blockIdx.x");
  auto thread = ParseThreadBindingTag("threadIdx.y");
  for (const char* name : {"global", "global.ddr", "global.vtcm", "global.workspace",
                           "global.texture", "global.texture-1d", "global.texture-nchw",
                           "global.texture-nhwc", "global.texture-weight"}) {
    SCOPED_TRACE(name);
    if (auto scope = GetRegisteredMemoryScope(name)) {
      EXPECT_TRUE(CanRelaxStorageUnderThread(*scope, block));
      EXPECT_TRUE(CanRelaxStorageUnderThread(*scope, thread));
    }
  }
  // The legacy standalone texture scope is distinct from global texture spaces.
  if (auto texture = GetRegisteredMemoryScope("texture")) {
    EXPECT_FALSE(CanRelaxStorageUnderThread(*texture, block));
    EXPECT_FALSE(CanRelaxStorageUnderThread(*texture, thread));
  }
}

TEST(MemoryScopePolicy, OptionalLookupObservesLaterRegistration) {
  const ffi::String name = "test.optional_scope_registration";
  EXPECT_FALSE(GetRegisteredMemoryScope(name).has_value());
  auto scope = runtime::MemoryScope::Register(name);
  auto found = GetRegisteredMemoryScope(name);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(*found, scope);
}

TEST(MemoryScopePolicy, SharedSpacesRelaxWithinBlock) {
  for (const char* name : {"shared", "shared.dyn"}) {
    SCOPED_TRACE(name);
    auto scope = runtime::MemoryScope::Get(name);
    EXPECT_FALSE(CanRelaxStorageUnderThread(scope, ParseThreadBindingTag("blockIdx.x")));
    EXPECT_TRUE(CanRelaxStorageUnderThread(scope, ParseThreadBindingTag("threadIdx.x")));
    EXPECT_TRUE(CanRelaxStorageUnderThread(scope, ParseThreadBindingTag("threadIdx.y")));
    EXPECT_TRUE(CanRelaxStorageUnderThread(scope, ParseThreadBindingTag("vthread")));
  }
}

TEST(MemoryScopePolicy, WarpRelaxesOnlyThreadX) {
  auto warp = runtime::MemoryScope::Get("warp");
  EXPECT_TRUE(CanRelaxStorageUnderThread(warp, ParseThreadBindingTag("threadIdx.x")));
  for (const char* binding : {"blockIdx.x", "threadIdx.y", "threadIdx.z", "vthread"}) {
    SCOPED_TRACE(binding);
    EXPECT_FALSE(CanRelaxStorageUnderThread(warp, ParseThreadBindingTag(binding)));
  }
}

TEST(MemoryScopePolicy, RegistrationDoesNotInferThreadVisibility) {
  static const auto extension = runtime::MemoryScope::Register("global.test_thread_visibility");
  for (const auto& scope : {runtime::MemoryScope::Get("local"), extension}) {
    SCOPED_TRACE(scope.name());
    EXPECT_FALSE(CanRelaxStorageUnderThread(scope, ParseThreadBindingTag("blockIdx.x")));
    EXPECT_FALSE(CanRelaxStorageUnderThread(scope, ParseThreadBindingTag("threadIdx.x")));
  }
}

}  // namespace s_tir
}  // namespace tvm
