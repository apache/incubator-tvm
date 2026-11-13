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
#ifndef TVM_RUNTIME_DISCO_BUILTIN_H_
#define TVM_RUNTIME_DISCO_BUILTIN_H_

#include <tvm/ffi/dtype.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/runtime/extra_base.h>
#include <tvm/runtime/tensor.h>

#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Possible kinds of reduction operations.
 */
enum class ReduceKind : int32_t {
  kSum = 0,
  kProd = 1,
  kMin = 2,
  kMax = 3,
  kAvg = 4,
};

/*! \brief Converts `ReduceKind` to string */
inline std::string ReduceKind2String(ReduceKind kind) {
  switch (kind) {
    case ReduceKind::kSum:
      return "kSum";
    case ReduceKind::kProd:
      return "kProd";
    case ReduceKind::kMin:
      return "kMin";
    case ReduceKind::kMax:
      return "kMax";
    case ReduceKind::kAvg:
      return "kAvg";
  }
  TVM_FFI_THROW(ValueError) << "Unknown ReduceKind: " << static_cast<int>(kind);
}

/*!
 * \brief Load a runtime Module, then create and initialize a RelaxVM
 * \param path The path to the runtime Module (a DSO file) to be loaded
 * \param device The default device used to initialize the RelaxVM
 * \return The RelaxVM as a runtime Module
 */
TVM_RUNTIME_EXTRA_DLL ffi::Module LoadVMModule(std::string path, ffi::Optional<Device> device);
/*!
 * \brief Create an uninitialized empty Tensor
 * \param shape The shape of the Tensor
 * \param dtype The dtype of the Tensor
 * \param device The device the Tensor is created on. If None, use the thread local default device
 * \return The Tensor created
 */
TVM_RUNTIME_EXTRA_DLL ffi::Tensor DiscoEmptyTensor(ffi::Shape shape, DLDataType dtype,
                                                   ffi::Optional<Device> device);
/*!
 * \brief Perform an allreduce operation using the underlying communication library
 * \param send The array send to perform allreduce on
 * \param reduce_kind The kind of reduction operation (e.g. sum, avg, min, max)
 * \param in_group Whether the allreduce operation performs globally or in group as default.
 * \param recv The array receives the outcome of allreduce
 */
TVM_RUNTIME_EXTRA_DLL void AllReduce(ffi::Tensor send, ReduceKind reduce_kind, bool in_group,
                                     ffi::Tensor recv);
/*!
 * \brief Perform an allgather operation using the underlying communication library
 * \param send The array send to perform allgather on
 * \param in_group Whether the allgather operation performs globally or in group as default.
 * \param recv The array receives the outcome of allgather
 */
TVM_RUNTIME_EXTRA_DLL void AllGather(ffi::Tensor send, bool in_group, ffi::Tensor recv);
/*!
 * \brief Perform a broadcast operation from worker-0
 * \param send The buffer to be broadcasted
 * \param in_group Whether the broadcast operation performs globally or in group as default.
 * \param recv The buffer receives the broadcasted array
 */
TVM_RUNTIME_EXTRA_DLL void BroadcastFromWorker0(ffi::Tensor send, bool in_group, ffi::Tensor recv);
/*!
 * \brief Perform a scatter operation from worker-0, chunking the given buffer into equal parts.
 * \param send For worker-0, it must be provided, and otherwise, the buffer must be None.
 * The buffer will be divided into equal parts and sent to each worker accordingly.
 * \param in_group Whether the scatter operation performs globally or in group as default.
 * \param recv The receiving buffer, which must not be None.
 */
TVM_RUNTIME_EXTRA_DLL void ScatterFromWorker0(ffi::Optional<ffi::Tensor> send, bool in_group,
                                              ffi::Tensor recv);
/*!
 * \brief Perform a gather operation to worker-0.
 * \param send The sending buffer, which must not be None.
 * \param in_group Whether the gather operation performs globally or in group as default.
 * \param recv For worker-0, it must be provided, and otherwise, the buffer must be None. The
 * receiving buffer will be divided into equal parts and receive from each worker accordingly.
 */
TVM_RUNTIME_EXTRA_DLL void GatherToWorker0(ffi::Tensor send, bool in_group,
                                           ffi::Optional<ffi::Tensor> recv);
/*!
 * \brief Receive a buffer from worker-0. No-op if the current worker is worker-0.
 * \param buffer The buffer to be received
 */
TVM_RUNTIME_EXTRA_DLL void RecvFromWorker0(ffi::Tensor buffer);
/*!
 * \brief Send a buffer to the corresponding worker in the next group.
 * An error is thrown if the worker is already in the last group.
 * \param buffer The sending buffer.
 */
TVM_RUNTIME_EXTRA_DLL void SendToNextGroup(ffi::Tensor buffer);
/*!
 * \brief Receive a buffer from the corresponding worker in the previous group.
 * An error is thrown if the worker is already in the first group.
 * \param buffer The receiving buffer.
 */
TVM_RUNTIME_EXTRA_DLL void RecvFromPrevGroup(ffi::Tensor buffer);
/*!
 * \brief Send a buffer to the target receiver worker (globally across all groups).
 * \param buffer The sending buffer.
 * \param receiver_id The global receiver worker id.
 */
TVM_RUNTIME_EXTRA_DLL void SendToWorker(ffi::Tensor buffer, int receiver_id);
/*!
 * \brief Receive a buffer from the target sender worker (globally across all groups).
 * \param buffer The receiving buffer.
 * \param sender_id The global sender worker id.
 */
TVM_RUNTIME_EXTRA_DLL void RecvFromWorker(ffi::Tensor buffer, int sender_id);
/*! \brief Get the local worker id */
TVM_RUNTIME_EXTRA_DLL int WorkerId();
/*!
 * \brief Called by the worker thread. Waiting until the worker completes all its tasks.
 * As a specific example, on a CUDA worker, it blocks until all kernels are launched and
 * cudaStreamSynchronize is complete.
 */
TVM_RUNTIME_EXTRA_DLL void SyncWorker();

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_BUILTIN_H_
