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

#include "../../../src/support/arena.h"

#include <gtest/gtest.h>

#include <memory>

namespace tvm {
namespace test {

namespace {
// A non-trivially-destructible type: destructing it has an observable
// side effect (incrementing a counter), unlike a plain-data struct.
struct DtorCounter {
  explicit DtorCounter(int* counter) : counter(counter) {}
  ~DtorCounter() { (*counter)++; }
  int* counter;
};
}  // namespace

TEST(ArenaTests, MakeRunsDestructorOnFreeAll) {
  int destroyed = 0;
  {
    support::Arena arena;
    for (int i = 0; i < 8; ++i) {
      arena.make<DtorCounter>(&destroyed);
    }
    EXPECT_EQ(destroyed, 0);
    // FreeAll() may be called directly (not only via ~Arena()), e.g. by
    // MinRPCServer. It must run pending destructors itself rather than
    // relying on the caller to do so, or on ~Arena() running afterwards.
    arena.FreeAll();
    EXPECT_EQ(destroyed, 8);
  }
  // ~Arena() must not re-run the same destructors after an explicit
  // FreeAll(), and must not touch the now-freed ArenaDeleter bookkeeping.
  EXPECT_EQ(destroyed, 8);
}

TEST(ArenaTests, MakeRunsDestructorOnRecycleAll) {
  int destroyed = 0;
  support::Arena arena;
  arena.make<DtorCounter>(&destroyed);
  arena.RecycleAll();
  EXPECT_EQ(destroyed, 1);

  arena.make<DtorCounter>(&destroyed);
  EXPECT_EQ(destroyed, 1);
}

TEST(ArenaTests, TrivialTypeUnaffected) {
  support::Arena arena;
  int* x = arena.make<int>(42);
  EXPECT_EQ(*x, 42);
  // No crash/UB freeing an arena that only ever held trivially
  // destructible objects.
  arena.FreeAll();
}

namespace {
struct MoveOnlyHolder {
  explicit MoveOnlyHolder(std::unique_ptr<int> value) : value(std::move(value)) {}
  std::unique_ptr<int> value;
};
}  // namespace

TEST(ArenaTests, MakeAcceptsMoveOnlyArgument) {
  support::Arena arena;
  // Regression test: make<T>(std::make_unique<...>(...)) must not be ambiguous with std::forward
  // via ADL
  auto* holder = arena.make<MoveOnlyHolder>(std::make_unique<int>(7));
  EXPECT_EQ(*holder->value, 7);
}

}  // namespace test
}  // namespace tvm
