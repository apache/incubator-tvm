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

#include "../../../src/runtime/vm/attn_utils.h"

namespace tvm {
namespace runtime {
namespace vm {
namespace {

HostMemoryVector IntVector(std::initializer_list<int32_t> values) {
  HostMemoryVector result(values.size(), DLDataType{kDLInt, 32, 1}, Device{kDLCPU, 0});
  for (int32_t value : values) result.push_back(value);
  return result;
}

std::vector<int32_t> ReadInts(const ffi::Tensor& tensor) {
  std::vector<int32_t> result(tensor.shape().Product());
  tvm::runtime::CopyToBytes(tensor, result.data(), result.size() * sizeof(int32_t));
  return result;
}

TEST(PagedKVCacheAuxData, GlobalAndSlidingMetadataRemainIndependent) {
  // Exercise the manager used by WebGPU and Metal on CPU. The normal CPU
  // cache uses CachedPagedKVCacheAuxDataManager and would miss this aliasing.
  Device cpu{kDLCPU, 0};
  PlainPagedKVCacheAuxDataManager manager(2, 128, 512, DLDataType{kDLInt, 32, 1}, cpu, cpu, nullptr,
                                          true);
  auto full_indptr = IntVector({0, 4});
  auto sliding_indptr = IntVector({0, 2});
  auto full_indices = IntVector({10, 11, 12, 13});
  auto sliding_indices = IntVector({12, 13});
  auto last_page = IntVector({7});
  auto full_offset = IntVector({0});
  auto sliding_offset = IntVector({5});
  auto sink = IntVector({0});
  for (int iteration = 0; iteration < 2; ++iteration) {
    manager.ResetAttnAuxDataCopy();
    for (int depth = 0; depth < kPagedKVCacheMaxBlockDepth; ++depth) {
      auto full_ip = manager.CopyPageIndptrOnDepthAsync(&full_indptr, depth);
      auto full_ix = manager.CopyPageIndicesOnDepthAsync(&full_indices, depth);
      auto full_len = manager.CopyLengthInfoOnDepthAsync(&last_page, &full_offset, &sink, depth);
      auto sliding_ip = manager.CopyPageIndptrOnDepthAsync(&sliding_indptr, depth, true);
      auto sliding_ix = manager.CopyPageIndicesOnDepthAsync(&sliding_indices, depth, true);
      auto sliding_len =
          manager.CopyLengthInfoOnDepthAsync(&last_page, &sliding_offset, &sink, depth, true);
      manager.CommitAttnAuxDataCopy();
      EXPECT_EQ(ReadInts(full_ip), (std::vector<int32_t>{0, 4}));
      EXPECT_EQ(ReadInts(full_ix), (std::vector<int32_t>{10, 11, 12, 13}));
      EXPECT_EQ(ReadInts(full_len), (std::vector<int32_t>{7, 0, 0}));
      EXPECT_EQ(ReadInts(sliding_ip), (std::vector<int32_t>{0, 2}));
      EXPECT_EQ(ReadInts(sliding_ix), (std::vector<int32_t>{12, 13}));
      EXPECT_EQ(ReadInts(sliding_len), (std::vector<int32_t>{7, 5, 0}));
    }
  }
}

}  // namespace
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
