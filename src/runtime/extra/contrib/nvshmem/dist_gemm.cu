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
#include <nvshmem.h>
#include <nvshmemx.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>

namespace tvm {
namespace runtime {

void* get_pointer(ffi::Tensor data, ffi::Shape index) {
  TVM_FFI_ICHECK(data.defined());
  TVM_FFI_ICHECK(data.IsContiguous()) << "data is not contiguous";
  char* ptr = reinterpret_cast<char*>(data.data_ptr()) + data.byte_offset();
  int64_t offset = 0;
  // stride may be null, use shape instead
  for (int i = 0; i < static_cast<int>(index.size()); i++) {
    offset *= data.shape()[i];
    offset += index[i];
  }
  return static_cast<void*>(ptr + offset * ffi::GetDataSize(1, data.dtype()));
}

void cuStreamWaitValue64Wrapper(TVMStreamHandle strm, void* addr, uint64_t expected) {
  cuStreamWaitValue64(CUstream(strm), reinterpret_cast<CUdeviceptr>(addr), expected,
                      CU_STREAM_WAIT_VALUE_EQ);
}

void cuStreamWriteValue64Wrapper(TVMStreamHandle strm, void* addr, uint64_t value, int dst_device) {
  int my_rank = nvshmem_my_pe();
  void* remote_addr = my_rank == dst_device ? addr : nvshmem_ptr(addr, dst_device);
  cuStreamWriteValue64(CUstream(strm), reinterpret_cast<CUdeviceptr>(remote_addr), value,
                       CU_STREAM_WRITE_VALUE_DEFAULT);
}

void copy_to_peer(void* dst, int dst_device, void* src, size_t size, TVMStreamHandle stream) {
  int my_rank = nvshmem_my_pe();
  void* remote_dst = my_rank == dst_device ? dst : nvshmem_ptr(dst, dst_device);
  cudaMemcpyAsync(remote_dst, src, size, cudaMemcpyDefault, CUstream(stream));
}

Device current_cuda_device() {
  int device_id;
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
  return Device{DLDeviceType::kDLCUDA, device_id};
}

TVMStreamHandle stream_create() {
  Device device = current_cuda_device();
  return DeviceAPI::Get(device)->CreateStream(device);
}

void stream_sync(TVMStreamHandle from_stream, TVMStreamHandle to_stream) {
  Device device = current_cuda_device();
  DeviceAPI::Get(device)->SyncStreamFromTo(device, from_stream, to_stream);
}

TVMStreamHandle stream_from_int(int64_t stream) {
  return reinterpret_cast<TVMStreamHandle>(stream);
}

void set_streaming_policy(int64_t stream, void* ptr, size_t size) {
  cudaStream_t strm = static_cast<cudaStream_t>(stream_from_int(stream));
  struct cudaAccessPolicyWindow accessPolicyWindow = {ptr, size, 0.0, cudaAccessPropertyStreaming,
                                                      cudaAccessPropertyStreaming};
  cudaStreamAttrValue streamAttrValue;
  streamAttrValue.accessPolicyWindow = accessPolicyWindow;
  cudaStreamSetAttribute(strm, cudaStreamAttributeAccessPolicyWindow, &streamAttrValue);
}

void transfer_to_peers_reduce_scatter(ffi::Tensor semaphore, ffi::Tensor gemm_out,
                                      ffi::Tensor staging_buffer, int64_t stream_handle, int32_t M,
                                      int32_t N, int32_t BLK_M, int32_t BLK_N, int32_t WORLD_SIZE) {
  TVM_FFI_ICHECK(semaphore.defined());
  TVM_FFI_ICHECK(gemm_out.defined());
  TVM_FFI_ICHECK(staging_buffer.defined());
  TVMStreamHandle stream = stream_from_int(stream_handle);
  int my_rank = nvshmem_my_pe();
  int LOCAL_M = M / WORLD_SIZE;
  for (int i = 0; i < WORLD_SIZE; i++) {
    int to_rank = (my_rank + i + 1) % WORLD_SIZE;
    if (to_rank != my_rank) {
      cuStreamWaitValue64Wrapper(stream, get_pointer(semaphore, ffi::Shape{to_rank}),
                                 LOCAL_M / BLK_M * N / BLK_N);
      copy_to_peer(get_pointer(staging_buffer, ffi::Shape{my_rank, 0, 0}), to_rank,
                   get_pointer(gemm_out, ffi::Shape{to_rank * LOCAL_M, 0}), LOCAL_M * N * 2,
                   stream);
    } else {
      int device_id;
      TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
      TVMStreamHandle main_stream = TVMFFIEnvGetStream(kDLCUDA, device_id);
      copy_to_peer(get_pointer(staging_buffer, ffi::Shape{my_rank, 0, 0}), to_rank,
                   get_pointer(gemm_out, ffi::Shape{to_rank * LOCAL_M, 0}), LOCAL_M * N * 2,
                   main_stream);
    }
  }
}

void transfer_to_peers_all_gather(ffi::Tensor semaphore, ffi::Tensor A, ffi::Tensor ag_out,
                                  int64_t stream_handle, int32_t M, int32_t K, int32_t WORLD_SIZE) {
  TVM_FFI_ICHECK(semaphore.defined());
  TVM_FFI_ICHECK(A.defined());
  TVM_FFI_ICHECK(ag_out.defined());
  TVMStreamHandle stream = stream_from_int(stream_handle);
  int my_rank = nvshmem_my_pe();
  int LOCAL_M = M / WORLD_SIZE;
  for (int i = 0; i < WORLD_SIZE; i++) {
    int to_rank = (my_rank + WORLD_SIZE - i - 1) % WORLD_SIZE;
    if (to_rank != my_rank) {
      copy_to_peer(get_pointer(ag_out, ffi::Shape{my_rank * LOCAL_M, 0}), to_rank,
                   get_pointer(A, ffi::Shape{0, 0}), LOCAL_M * K * 2, stream);
      cuStreamWriteValue64Wrapper(stream, get_pointer(semaphore, ffi::Shape{my_rank}), 1, to_rank);
    }
  }
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.disco.copy_to_peer", copy_to_peer)
      .def("runtime.disco.cu_stream_wait_value64", cuStreamWaitValue64Wrapper)
      .def("runtime.disco.stream_create", stream_create)
      .def("runtime.disco.stream_sync", stream_sync)
      .def("runtime.disco.transfer_to_peers_reduce_scatter", transfer_to_peers_reduce_scatter)
      .def("runtime.disco.transfer_to_peers_all_gather", transfer_to_peers_all_gather)
      .def("runtime.disco.set_streaming_policy", [](int64_t stream, ffi::Tensor ptr, size_t size) {
        TVM_FFI_ICHECK(ptr.defined());
        set_streaming_policy(stream, ptr.data_ptr(), size);
      });
}

}  // namespace runtime
}  // namespace tvm
