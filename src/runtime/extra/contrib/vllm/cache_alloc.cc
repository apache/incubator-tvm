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
#include <cuda_runtime.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

namespace tvm {
namespace runtime {
namespace vllm {

ffi::Array<ffi::Tensor> AllocateKVCache(int head_size, int num_layers, int num_heads,
                                        int block_size, int num_blocks) {
  ffi::Array<ffi::Tensor> cache;
  int element_size = 2;
  int vec_size = 16 / element_size;

  int device_id;
  cudaGetDevice(&device_id);

  DLDevice dev{DLDeviceType::kDLCUDA, device_id};

  for (int i = 0; i < num_layers; ++i) {
    ffi::Tensor key_blocks =
        tvm::runtime::Empty({num_blocks, num_heads, head_size / vec_size, block_size, vec_size},
                            DLDataType{kDLFloat, 16, 1}, dev);
    ffi::Tensor value_blocks = tvm::runtime::Empty({num_blocks, num_heads, head_size, block_size},
                                                   DLDataType{kDLFloat, 16, 1}, dev);
    cache.push_back(key_blocks);
    cache.push_back(value_blocks);
  }

  return cache;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tvm.contrib.vllm.allocate_kv_cache", AllocateKVCache);
}

}  // namespace vllm
}  // namespace runtime
}  // namespace tvm
