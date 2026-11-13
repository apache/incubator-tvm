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
#include <tvm/runtime/disco/cuda_ipc_memory.h>

#include "../../../../../3rdparty/tensorrt_llm/custom_allreduce_kernels.h"
#include "../nccl/nccl_context.h"

namespace tvm {
namespace runtime {
namespace nccl {
namespace cuda_ipc {

using tvm::runtime::cuda_ipc::CUDAIPCMemory;

/*! \brief Compute the size (i.e., number of elements) of the input tensor. */
inline int64_t TensorSize(const DLTensor* tensor) {
  int64_t size = 1;
  for (int i = tensor->ndim - 1; i >= 0; --i) {
    if (tensor->strides) {
      TVM_FFI_ICHECK_EQ(tensor->strides[i], size);
    }
    size *= tensor->shape[i];
  }
  return size;
}

/*! \brief Check if customized all-reduce kernels can be applied. */
inline bool CanApplyCustomAllReduce(int64_t num_elements, DLDataType dtype) {
  // The customized all-reduce kernel has the following requirement(s).
  return num_elements % (16 / ((dtype.bits * dtype.lanes + 7) / 8)) == 0;
}

/*! \brief Check if the two-shot customized all-reduce kernel can be applied. */
inline bool CanApplyTwoShotAllReduce(int64_t num_elements, DLDataType dtype, int num_workers) {
  // The two-shot customized all-reduce kernel has the following requirement(s).
  return (num_elements / num_workers) % (16 / ((dtype.bits * dtype.lanes + 7) / 8)) == 0;
}

/*!
 * \brief Customized all-reduce kernel backed by CUDA IPC memory.
 * \param send The input tensor of all-reduce.
 * \param strategy The all-reduce strategy. See AllReduceStrategyType for detail.
 * \param recv The output tensor of all-reduce.
 */
void CustomAllReduce(DLTensor* send, int strategy, DLTensor* recv) {
  int64_t num_elements = TensorSize(send);
  nccl::CCLThreadLocalContext* ctx = nccl::CCLThreadLocalContext::Get();
  TVM_FFI_ICHECK_EQ(ctx->worker->num_groups, 1)
      << "Custom AllReduce for multiple group is not yet implemented.";

  tensorrt_llm::AllReduceStrategyType strategy_ =
      static_cast<tensorrt_llm::AllReduceStrategyType>(strategy);
  if (strategy_ == tensorrt_llm::AllReduceStrategyType::AUTO) {
    strategy_ = tensorrt_llm::SelectImplementation(
        num_elements * ((send->dtype.bits * send->dtype.lanes + 7) / 8), ctx->worker->num_workers);
  }

  if (strategy_ == tensorrt_llm::AllReduceStrategyType::RING ||
      !CanApplyCustomAllReduce(num_elements, send->dtype)) {
    // Dispatch to nccl AllReduce if the customized all-reduce cannot apply.
    deviceStream_t stream = ctx->GetDefaultStream();
    NCCL_CALL(ncclAllReduce(send->data, recv->data, num_elements,
                            /*datatype=*/nccl::AsNCCLDataType(send->dtype),
                            /*op=*/ncclSum, ctx->global_comm, stream));
    return;
  }

  // Initialize the all-reduce kernel arguments.
  tensorrt_llm::AllReduceParams params;
  params.ranks_per_node = ctx->worker->num_workers;
  params.rank = ctx->worker->worker_id;
  params.local_rank = ctx->worker->worker_id;
  CUDAIPCMemory ipc_memory = CUDAIPCMemory::GetIPCMemoryFromDevicePtr(send->data);
  params.barrier_flag = ipc_memory->barrier_flag++;
  for (int i = 0; i < ctx->worker->num_workers; ++i) {
    params.peer_comm_buffer_ptrs[i] = ipc_memory->remote_data[i];
  }
  for (int i = 0; i < ctx->worker->num_workers; ++i) {
    params.peer_barrier_ptrs_in[i] = reinterpret_cast<uint32_t*>(ipc_memory->barrier_in[i]);
  }
  for (int i = 0; i < ctx->worker->num_workers; ++i) {
    params.peer_barrier_ptrs_out[i] = reinterpret_cast<uint32_t*>(ipc_memory->barrier_out[i]);
  }

  if (!CanApplyTwoShotAllReduce(num_elements, send->dtype, ctx->worker->num_workers)) {
    // Two-shot all-reduce does not support this case.
    // So we fallback to the one-shot strategy.
    strategy_ = tensorrt_llm::AllReduceStrategyType::ONESHOT;
  }

  tensorrt_llm::customAllReduce(params, recv->data, num_elements, send->dtype, strategy_,
                                ctx->GetDefaultStream());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.disco.cuda_ipc.custom_allreduce", CustomAllReduce);
}

}  // namespace cuda_ipc
}  // namespace nccl
}  // namespace runtime
}  // namespace tvm
