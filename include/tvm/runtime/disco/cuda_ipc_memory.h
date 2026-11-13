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

#ifndef TVM_RUNTIME_DISCO_CUDA_IPC_MEMORY_H_
#define TVM_RUNTIME_DISCO_CUDA_IPC_MEMORY_H_

#include <tvm/runtime/base.h>
#include <tvm/runtime/extra_base.h>
#include <tvm/runtime/tensor.h>

#include <vector>

namespace tvm {
namespace runtime {
namespace cuda_ipc {

/*!
 * \brief The CUDA IPC (interprocess communication) memory object,
 * which internally contains data pointers to CUDA IPC memory.
 * It is be useful for efficient all-reduce implementation.
 * \note Right now the class members are closely tied with customized
 * all-reduce kernel. They may also be extended for other uses in
 * the future.
 */
class CUDAIPCMemoryObj : public ffi::Object {
 public:
  /*! \brief The number of GPU workers. */
  int num_workers;
  /*! \brief The worker id corresponding to this IPC memory object. */
  int worker_id;
  /*!
   * \brief The data pointers of all all-reduce inputs.
   * It has "num_workers" pointers. The i-th pointer is the data pointer on worker i.
   * If "i != worker_id", the pointer is an IPC data pointer.
   * Otherwise, the pointer is a local CUDA data pointer.
   */
  std::vector<void*> remote_data;

  // We introduce the barrier helper data below per CUDAIPCMemory object
  // so that they can be used by custom collective operations and allow
  // fine-grained synchronization on each buffer. These barriers have
  // low overhead, and can potentially enable concurrent execution of
  // kernels in future.
  /*!
   * \brief The pointers to input barrier signals of all workers for all-reduce.
   * It has "num_workers" pointers, and the pointer arrangement is the same as "remote_data".
   */
  std::vector<void*> barrier_in;
  /*!
   * \brief The pointers to output barrier signals of all workers for all-reduce.
   * It has "num_workers" pointers, and the pointer arrangement is the same as "remote_data".
   */
  std::vector<void*> barrier_out;
  /*! \brief The integer buffer flag for all-reduce. */
  int barrier_flag;

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("tvm.runtime.disco.cuda_ipc_memory", CUDAIPCMemoryObj, ffi::Object);
};

/*!
 * \brief Managed reference to CUDAIPCMemoryObj.
 * \sa CUDAIPCMemory
 */
class CUDAIPCMemory : public ffi::ObjectRef {
 public:
  /*! \brief Get the global singleton CUDAIPCMemory pool. */
  TVM_RUNTIME_EXTRA_DLL static TensorMemoryPool* GlobalMemoryPool();
  /*!
   * \brief Given a local CUDA data pointer, return the CUDAIPCMemory object of the pointer.
   * \note The pointer's CUDAIPCMemory is expected to have been allocated
   * through global function "cuda_ipc.alloc_storage". Or otherwise this
   * function will raise exception.
   */
  TVM_RUNTIME_EXTRA_DLL static CUDAIPCMemory GetIPCMemoryFromDevicePtr(void* ptr);

  explicit CUDAIPCMemory(ffi::UnsafeInit tag, ffi::ObjectPtr<CUDAIPCMemoryObj> n)
      : ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CUDAIPCMemory, ffi::ObjectRef, CUDAIPCMemoryObj);
};

}  // namespace cuda_ipc
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_CUDA_IPC_MEMORY_H_
