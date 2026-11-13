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
#include <tvm/runtime/tensor.h>

#include <cstring>

#include "../../../src/runtime/extra/disco/protocol.h"

namespace tvm {
namespace runtime {

TEST(DiscoDebugObject, TensorPreservesBase64Payload) {
  const char* expected[] = {
      "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAEIAQABAAAAAAAAAAEAAAAAAAAAAA==1",
      "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAEIAQACAAAAAAAAAAIAAAAAAAAAAIA=1",
      "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAAEIAQADAAAAAAAAAAMAAAAAAAAAAID/1",
  };
  const uint8_t data[] = {0, 128, 255};
  for (int64_t size = 1; size <= 3; ++size) {
    auto tensor = Empty(ffi::Shape({size}), DLDataType{kDLUInt, 8, 1}, Device{kDLCPU, 0});
    std::memcpy(tensor.GetDLTensorPtr()->data, data, size);
    auto object = ffi::make_object<DiscoDebugObject>();
    object->data = tensor;
    std::string encoded = object->SaveToStr();
    EXPECT_EQ(encoded, expected[size - 1]);
    auto loaded = DiscoDebugObject::LoadFromStr(encoded)->data.cast<ffi::Tensor>();
    EXPECT_EQ(loaded.dtype(), tensor.dtype());
    ASSERT_EQ(loaded.shape()[0], size);
    EXPECT_EQ(std::memcmp(loaded.GetDLTensorPtr()->data, data, size), 0);
  }
}

}  // namespace runtime
}  // namespace tvm
