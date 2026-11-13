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
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/disco/cuda_ipc_memory.h>
#include <tvm/runtime/tensor.h>

#include <unordered_map>

#include "../../../../../3rdparty/tensorrt_llm/custom_allreduce_kernels.h"
#include "../nccl/nccl_context.h"

namespace tvm {
namespace runtime {
namespace cuda_ipc {

using tensorrt_llm::MAX_ALL_REDUCE_BLOCKS;
using tensorrt_llm::MAX_RANKS_PER_NODE;

/*!
 * \brief All-gather the IPC memory handles across all distributed workers.
 * On each worker, we copy the IPC handle to GPU memory. And nccl AllGather
 * is reused to all-gather the handles. Finally the all-gathered handles
 * on each worker are copied from GPU to CPU.
 */
std::vector<cudaIpcMemHandle_t> AllGatherIPCHandles(nccl::CCLThreadLocalContext* ctx,
                                                    cudaIpcMemHandle_t local_handle) {
  void *d_src, *d_dst;
  TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&d_src, CUDA_IPC_HANDLE_SIZE));
  TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&d_dst, CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers));
  TVM_FFI_CHECK_CUDA_ERROR(
      cudaMemcpy(d_src, &local_handle, CUDA_IPC_HANDLE_SIZE, cudaMemcpyHostToDevice));
  NCCL_CALL(ncclAllGather(d_src, d_dst, CUDA_IPC_HANDLE_SIZE, ncclChar, ctx->global_comm,
                          /*stream=*/nullptr));
  std::vector<char> serial_handles(CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers, 0);
  TVM_FFI_CHECK_CUDA_ERROR(cudaMemcpy(serial_handles.data(), d_dst,
                                      CUDA_IPC_HANDLE_SIZE * ctx->worker->num_workers,
                                      cudaMemcpyDefault));
  std::vector<cudaIpcMemHandle_t> handles(ctx->worker->num_workers);
  for (int i = 0; i < ctx->worker->num_workers; ++i) {
    memcpy(handles[i].reserved, &serial_handles[i * CUDA_IPC_HANDLE_SIZE], CUDA_IPC_HANDLE_SIZE);
  }
  TVM_FFI_CHECK_CUDA_ERROR(cudaFree(d_src));
  TVM_FFI_CHECK_CUDA_ERROR(cudaFree(d_dst));
  return handles;
}

/*!
 * \brief The memory allocator of CUDAIPCMemory.
 * Overriding TensorMemoryPool for efficient memory management.
 */
class CUDAIPCMemoryPool final : public TensorMemoryPool {
 public:
  explicit CUDAIPCMemoryPool() = default;

  ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                    ffi::Optional<MemoryScope> mem_scope = std::nullopt) final {
    static const auto scope = MemoryScope::Get("ipc_memory");
    TVM_FFI_CHECK(!mem_scope.has_value() || mem_scope.value() == scope, ValueError)
        << "CUDA IPC memory requires the ipc_memory scope";
    return TensorMemoryPool::Empty(shape, dtype, device);
  }

  CUDAIPCMemory GetIPCMemoryFromDevicePtr(void* ptr) const {
    auto it = ipc_memory_map_.find(ptr);
    TVM_FFI_ICHECK(it != ipc_memory_map_.end())
        << "The given pointer's CUDAIPCMemory object does not exist. Please use global function "
           "\"cuda_ipc.alloc_storage\" to allocate the CUDAIPCMemory object first.";
    return it->second;
  }

  /*! \brief Return the global CUDAIPCMemory singleton allocator. */
  static CUDAIPCMemoryPool* Global() {
    static auto* pool =
        new ffi::ObjectPtr<CUDAIPCMemoryPool>(ffi::make_object<CUDAIPCMemoryPool>());
    return pool->get();
  }

 private:
  ffi::Tensor AllocBacking(size_t size, size_t alignment, DLDevice dev) final {
    auto [data_ptr, data_comm_ptrs] = AllocIPCMemory(
        dev, size, alignment, DLDataType{kDLUInt, 8, 1}, /*reset_memory_to_zero=*/false);
    int barrier_ptr_size = sizeof(uint32_t) * (MAX_ALL_REDUCE_BLOCKS + 2) * MAX_RANKS_PER_NODE;
    auto [barrier_in_ptr, barrier_in_comm_ptrs] =
        AllocIPCMemory(dev, barrier_ptr_size, alignment, DLDataType{kDLUInt, 32, 1},
                       /*reset_memory_to_zero=*/true);
    auto [barrier_out_ptr, barrier_out_comm_ptrs] =
        AllocIPCMemory(dev, barrier_ptr_size, alignment, DLDataType{kDLUInt, 32, 1},
                       /*reset_memory_to_zero=*/true);

    // Create the CUDAIPCMemory object.
    ffi::ObjectPtr<CUDAIPCMemoryObj> ipc_memory = ffi::make_object<CUDAIPCMemoryObj>();
    nccl::CCLThreadLocalContext* nccl_ctx = nccl::CCLThreadLocalContext::Get();
    ipc_memory->remote_data = data_comm_ptrs;
    ipc_memory->barrier_in = barrier_in_comm_ptrs;
    ipc_memory->barrier_out = barrier_out_comm_ptrs;
    ipc_memory->barrier_flag = 1;
    ipc_memory->num_workers = nccl_ctx->worker->num_workers;
    ipc_memory->worker_id = nccl_ctx->worker->worker_id;
    CUDAIPCMemory memory(ffi::UnsafeInit{}, std::move(ipc_memory));
    ipc_memory_map_.insert_or_assign(data_ptr, memory);
    struct Backing {
      CUDAIPCMemory memory;
      void AllocData(DLTensor* tensor) { tensor->data = memory->remote_data[memory->worker_id]; }
      void FreeData(DLTensor* tensor) {
        ffi::CUDADeviceGuard device_guard(tensor->device.device_id);
        FreeIPCMemory(memory->remote_data, memory->worker_id);
        FreeIPCMemory(memory->barrier_in, memory->worker_id);
        FreeIPCMemory(memory->barrier_out, memory->worker_id);
        // The original global lookup map has process lifetime, independently of
        // the thread-local NCCL context used to exchange the handles.
        Global()->ipc_memory_map_.erase(tensor->data);
      }
    };
    int64_t extent = static_cast<int64_t>(size);
    return ffi::Tensor::FromNDAlloc(Backing{std::move(memory)}, ffi::ShapeView(&extent, 1),
                                    DLDataType{kDLUInt, 8, 1}, dev);
  }

  /*!
   * \brief Allocate CUDA memory with the required size, alignment and dtype,
   * and return the IPC memory data pointers.
   * \returns The local data pointer of the allocated CUDA memory,
   * and a list of pointers that contains the CUDA IPC memory pointer
   * of the allocated memory on each worker.
   * For the i-th pointer, if i is the worker id of the given device,
   * then the returned i-th pointer points to the local CUDA memory,
   * or otherwise it is an IPC memory pointer.
   * \details This function first allocates local memory on every worker,
   * and creates an IPC memory pointer for the local memory.
   * Then it uses nccl all-gather to synchronize the IPC memory pointers
   * across all workers, so that every worker know each other's IPC memory
   * pointer.
   */
  std::pair<void*, std::vector<void*>> AllocIPCMemory(Device dev, size_t size, size_t alignment,
                                                      DLDataType type_hint,
                                                      bool reset_memory_to_zero) {
    // Alloc local buffer
    TVM_FFI_ICHECK(dev.device_type == kDLCUDA);
    void* ptr;
    TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
    TVM_FFI_CHECK_CUDA_ERROR(cudaMalloc(&ptr, size));
    // Reset allocated memory to zero when required.
    // We explicitly synchronize after memset, to make sure memset finishes
    // before using all-gather to exchange IPC handles.
    // This is important to ensure the memory reset get ordered
    // before any other peers read the memory.
    if (reset_memory_to_zero) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaMemset(ptr, 0, size));
      TVM_FFI_CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    // Create ipc handle
    cudaIpcMemHandle_t local_handle;
    TVM_FFI_CHECK_CUDA_ERROR(cudaIpcGetMemHandle(&local_handle, ptr));
    // All-gather IPC handles.
    nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
    std::vector<cudaIpcMemHandle_t> handles = AllGatherIPCHandles(ctx, local_handle);
    // Collect the all-gather results.
    std::vector<void*> comm_ptrs(ctx->worker->num_workers);
    for (size_t node_id = 0; node_id < handles.size(); ++node_id) {
      if (static_cast<int>(node_id) == ctx->worker->worker_id) {
        comm_ptrs[node_id] = ptr;
      } else {
        uint8_t* foreign_buffer;
        TVM_FFI_CHECK_CUDA_ERROR(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&foreign_buffer),
                                                      handles[node_id],
                                                      cudaIpcMemLazyEnablePeerAccess));
        comm_ptrs[node_id] = foreign_buffer;
      }
    }
    return std::make_pair(ptr, comm_ptrs);
  }

  /*! \brief Free the IPC memory pointers. */
  static void FreeIPCMemory(std::vector<void*> comm_ptrs, int worker_id) {
    for (int i = 0; i < static_cast<int>(comm_ptrs.size()); ++i) {
      if (i != worker_id) {
        // Free ipc handle.
        TVM_FFI_CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(comm_ptrs[i]));
      } else {
        // Free local buffer.
        TVM_FFI_CHECK_CUDA_ERROR(cudaFree(comm_ptrs[i]));
      }
    }
  }

  /*! \brief The mapping from local CUDA memory pointer to its allocated CUDAIPCMemory object. */
  std::unordered_map<void*, CUDAIPCMemory> ipc_memory_map_;
};

/*!
 * \brief Allocate owning CUDA IPC backing for VM tensors.
 */
ffi::Tensor IPCAllocTensor(ffi::Shape buffer_shape, DLDataType dtype_hint) {
  nccl::CCLThreadLocalContext* nccl_ctx = nccl::CCLThreadLocalContext::Get();
  Device device{DLDeviceType::kDLCUDA, nccl_ctx->device_id};
  static const auto scope = MemoryScope::Get("ipc_memory");
  return CUDAIPCMemoryPool::Global()->Empty(buffer_shape, dtype_hint, device, scope);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  MemoryScope::Global();
  refl::EnumDef<MemoryScopeObj>("ipc_memory")
      .set_attr(attr::kMemoryAllocProperty,
                static_cast<int>(MemoryAllocProperty::kSeparateAllocations));
  refl::GlobalDef()
      .def("runtime.disco.cuda_ipc.empty", IPCAllocTensor)
      .def("runtime.disco.cuda_ipc.alloc_storage", IPCAllocTensor)
      .def("runtime.disco.cuda_ipc.cuda_ipc_memory_allocator_clear",
           []() { CUDAIPCMemoryPool::Global()->Clear(); });
}

/******************** CUDAIPCMemoryObj ********************/

// Direct to CUDAIPCMemoryPool::Global.
TensorMemoryPool* CUDAIPCMemory::GlobalMemoryPool() { return CUDAIPCMemoryPool::Global(); }

// Direct to CUDAIPCMemoryPool::GlobalGetIPCMemoryFromDevicePtr.
CUDAIPCMemory CUDAIPCMemory::GetIPCMemoryFromDevicePtr(void* ptr) {
  return CUDAIPCMemoryPool::Global()->GetIPCMemoryFromDevicePtr(ptr);
}

}  // namespace cuda_ipc
}  // namespace runtime
}  // namespace tvm
