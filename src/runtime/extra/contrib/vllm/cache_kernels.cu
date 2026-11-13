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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include <algorithm>
#include <cassert>
#include <vector>

namespace vllm {

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ key_cache,      // [num_blocks, num_heads, head_size/x, block_size, x]
    scalar_t* __restrict__ value_cache,    // [num_blocks, num_heads, head_size, block_size]
    const int* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads, const int head_size,
    const int block_size, const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int src_key_idx = token_idx * key_stride + i;
    const int src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x +
                            head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
                            block_offset * x + x_offset;
    const int tgt_value_idx = block_idx * num_heads * head_size * block_size +
                              head_idx * head_size * block_size + head_offset * block_size +
                              block_offset;
    key_cache[tgt_key_idx] = __ldg(&key[src_key_idx]);
    value_cache[tgt_value_idx] = __ldg(&value[src_value_idx]);
  }
}

template <typename scalar_t>
__global__ void reconstruct_from_cache_kernel(
    const scalar_t* __restrict__ key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ value_cache,  // [num_blocks, num_heads, head_size, block_size]
    const int* __restrict__ slot_mapping,      // [num_tokens]
    scalar_t* __restrict__ key,                // [num_tokens, num_heads, head_size]
    scalar_t* __restrict__ value,              // [num_tokens, num_heads, head_size]
    const int key_stride, const int value_stride, const int num_heads, const int head_size,
    const int block_size, const int x) {
  const int token_idx = blockIdx.x;
  const int slot_idx = slot_mapping[token_idx];
  const int block_idx = slot_idx / block_size;
  const int block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int tgt_key_idx = token_idx * key_stride + i;
    const int tgt_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const int src_key_idx = block_idx * num_heads * (head_size / x) * block_size * x +
                            head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
                            block_offset * x + x_offset;
    const int src_value_idx = block_idx * num_heads * head_size * block_size +
                              head_idx * head_size * block_size + head_offset * block_size +
                              block_offset;

    key[tgt_key_idx] = __ldg(&key_cache[src_key_idx]);
    value[tgt_value_idx] = __ldg(&value_cache[src_value_idx]);
  }
}

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs, int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

}  // namespace vllm

namespace tvm {
namespace runtime {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tvm.contrib.vllm.reshape_and_cache",
           [](ffi::Tensor key, ffi::Tensor value, ffi::Tensor key_cache, ffi::Tensor value_cache,
              ffi::Tensor slot_mapping) {
             TVM_FFI_ICHECK(key.defined());
             TVM_FFI_ICHECK(key.IsContiguous());
             TVM_FFI_ICHECK(key.numel() == 0 || key.IsAligned(kAllocAlignment));
             TVM_FFI_ICHECK(value.defined());
             TVM_FFI_ICHECK(value.IsContiguous());
             TVM_FFI_ICHECK(value.numel() == 0 || value.IsAligned(kAllocAlignment));
             TVM_FFI_ICHECK(key_cache.defined());
             TVM_FFI_ICHECK(key_cache.IsContiguous());
             TVM_FFI_ICHECK(key_cache.numel() == 0 || key_cache.IsAligned(kAllocAlignment));
             TVM_FFI_ICHECK(value_cache.defined());
             TVM_FFI_ICHECK(value_cache.IsContiguous());
             TVM_FFI_ICHECK(value_cache.numel() == 0 || value_cache.IsAligned(kAllocAlignment));
             TVM_FFI_ICHECK(slot_mapping.defined());
             TVM_FFI_ICHECK(slot_mapping.IsContiguous());
             TVM_FFI_ICHECK(slot_mapping.numel() == 0 || slot_mapping.IsAligned(kAllocAlignment));
             int num_tokens = key.shape()[0];
             int num_heads = key.shape()[1];
             int head_size = key.shape()[2];
             int block_size = key_cache.shape()[3];
             int vec_size = key_cache.shape()[4];

             int key_stride = key.shape()[1] * key.shape()[2];
             int value_stride = value.shape()[1] * value.shape()[2];

             dim3 grid(num_tokens);
             dim3 block(std::min(num_heads * head_size, 512));

             using scalar_t = uint16_t;
             vllm::reshape_and_cache_kernel<scalar_t>
                 <<<grid, block>>>(static_cast<const scalar_t*>(key.data_ptr()),
                                   static_cast<const scalar_t*>(value.data_ptr()),
                                   static_cast<scalar_t*>(key_cache.data_ptr()),
                                   static_cast<scalar_t*>(value_cache.data_ptr()),
                                   static_cast<const int*>(slot_mapping.data_ptr()), key_stride,
                                   value_stride, num_heads, head_size, block_size, vec_size);

             return ffi::Array<ffi::Tensor>{key_cache, value_cache};
           })
      .def("tvm.contrib.vllm.reconstruct_from_cache",
           [](ffi::Tensor key_cache, ffi::Tensor value_cache, ffi::Tensor slot_mapping) {
             TVM_FFI_ICHECK(key_cache.defined());
             TVM_FFI_ICHECK(key_cache.IsContiguous());
             TVM_FFI_ICHECK(key_cache.numel() == 0 || key_cache.IsAligned(kAllocAlignment));
             TVM_FFI_ICHECK(value_cache.defined());
             TVM_FFI_ICHECK(value_cache.IsContiguous());
             TVM_FFI_ICHECK(value_cache.numel() == 0 || value_cache.IsAligned(kAllocAlignment));
             TVM_FFI_ICHECK(slot_mapping.defined());
             TVM_FFI_ICHECK(slot_mapping.IsContiguous());
             TVM_FFI_ICHECK(slot_mapping.numel() == 0 || slot_mapping.IsAligned(kAllocAlignment));
             int num_tokens = slot_mapping.shape()[0];
             int num_heads = value_cache.shape()[1];
             int head_size = value_cache.shape()[2];
             int block_size = value_cache.shape()[3];
             int vec_size = key_cache.shape()[4];

             DLDevice dev = key_cache.device();
             auto key =
                 tvm::runtime::Empty({num_tokens, num_heads, head_size}, key_cache.dtype(), dev);
             auto value =
                 tvm::runtime::Empty({num_tokens, num_heads, head_size}, key_cache.dtype(), dev);

             int key_stride = key.shape()[1] * key.shape()[2];
             int value_stride = value.shape()[1] * value.shape()[2];

             dim3 grid(num_tokens);
             dim3 block(std::min(num_heads * head_size, 512));

             using scalar_t = uint16_t;
             vllm::reconstruct_from_cache_kernel<scalar_t><<<grid, block>>>(
                 static_cast<const scalar_t*>(key_cache.data_ptr()),
                 static_cast<const scalar_t*>(value_cache.data_ptr()),
                 static_cast<const int*>(slot_mapping.data_ptr()),
                 static_cast<scalar_t*>(key.data_ptr()), static_cast<scalar_t*>(value.data_ptr()),
                 key_stride, value_stride, num_heads, head_size, block_size, vec_size);

             return ffi::Array<ffi::Tensor>{key, value};
           })
      .def("tvm.contrib.vllm.copy_blocks", [](ffi::Array<ffi::Tensor> key_value_caches,
                                              ffi::Tensor block_mapping) {
        TVM_FFI_ICHECK(block_mapping.defined());
        TVM_FFI_ICHECK(block_mapping.IsContiguous());
        TVM_FFI_ICHECK(block_mapping.numel() == 0 || block_mapping.IsAligned(kAllocAlignment));
        for (const auto& tensor : key_value_caches) {
          TVM_FFI_ICHECK(tensor.defined());
          TVM_FFI_ICHECK(tensor.IsContiguous());
          TVM_FFI_ICHECK(tensor.numel() == 0 || tensor.IsAligned(kAllocAlignment));
        }
        auto num_layers = key_value_caches.size() / 2;
        auto num_pairs = block_mapping.shape()[0] / 2;

        if (num_layers == 0) {
          return;
        }

        std::vector<int64_t> key_cache_ptrs(num_layers);
        std::vector<int64_t> value_cache_ptrs(num_layers);
        for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
          key_cache_ptrs[layer_idx] =
              reinterpret_cast<int64_t>(key_value_caches[2 * layer_idx].data_ptr());
          value_cache_ptrs[layer_idx] =
              reinterpret_cast<int64_t>(key_value_caches[2 * layer_idx + 1].data_ptr());
        }

        ffi::Tensor key_cache =
            key_value_caches[1];  // [num_blocks, num_heads, head_size, block_size]
        DLDevice dev = key_cache.device();

        ffi::Tensor key_cache_ptrs_gpu =
            tvm::runtime::Empty({static_cast<int>(num_layers)}, DLDataType{kDLInt, 64, 1}, dev);
        ffi::Tensor value_cache_ptrs_gpu =
            tvm::runtime::Empty({static_cast<int>(num_layers)}, DLDataType{kDLInt, 64, 1}, dev);
        tvm::runtime::CopyFromBytes(key_cache_ptrs_gpu, key_cache_ptrs.data(),
                                    sizeof(int64_t) * key_cache_ptrs.size());
        tvm::runtime::CopyFromBytes(value_cache_ptrs_gpu, value_cache_ptrs.data(),
                                    sizeof(int64_t) * value_cache_ptrs.size());

        ffi::Tensor block_mapping_gpu =
            tvm::runtime::Empty(block_mapping.shape(), DLDataType{kDLInt, 64, 1}, dev);
        tvm::runtime::CopyFromBytes(block_mapping_gpu, block_mapping.data_ptr(),
                                    sizeof(int64_t) * block_mapping.shape()[0]);

        const int numel_per_block =
            key_cache.shape()[1] * key_cache.shape()[2] * key_cache.shape()[3];
        dim3 grid(num_layers, num_pairs);
        dim3 block(std::min(1024, numel_per_block));

        using scalar_t = uint16_t;
        vllm::copy_blocks_kernel<scalar_t>
            <<<grid, block>>>(static_cast<int64_t*>(key_cache_ptrs_gpu.data_ptr()),
                              static_cast<int64_t*>(value_cache_ptrs_gpu.data_ptr()),
                              static_cast<int64_t*>(block_mapping_gpu.data_ptr()), numel_per_block);
      });
}

}  // namespace runtime
}  // namespace tvm
