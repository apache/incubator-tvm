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

#ifndef TVM_RUNTIME_DISCO_NCCL_NCCL_CONTEXT_H_
#define TVM_RUNTIME_DISCO_NCCL_NCCL_CONTEXT_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/disco/session.h>

#include "../../../../support/process_id.h"
#include "../utils.h"

/* `TVM_NCCL_RCCL_SWITCH` is set to 0 for NCCL, 1 for RCCL */
#ifndef TVM_NCCL_RCCL_SWITCH
#define TVM_NCCL_RCCL_SWITCH 0
#endif
#if TVM_NCCL_RCCL_SWITCH == 0
#include <nccl.h>
#include <tvm/ffi/extra/cuda/base.h>
#else
#include <rccl/rccl.h>

#include "../../rocm_utils.h"
#endif

namespace tvm {
namespace runtime {
namespace nccl {

#define NCCL_CALL(cmd)                                                                        \
  do {                                                                                        \
    auto r = (cmd);                                                                           \
    if (r != ncclSuccess) {                                                                   \
      TVM_FFI_THROW(InternalError) << TVM_DISCO_CCL_NAME "Errror: " << ncclGetErrorString(r); \
    }                                                                                         \
  } while (0)

#if TVM_NCCL_RCCL_SWITCH == 0

#define TVM_DISCO_DEVICE_NAME "cuda"
#define TVM_DISCO_CCL_NAME "nccl"

using deviceStream_t = cudaStream_t;
const constexpr DLDeviceType TVM_DISCO_DEVICE_TYPE = DLDeviceType::kDLCUDA;
inline void SetDevice(int device_id) { TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(device_id)); }
inline void StreamSynchronize(deviceStream_t stream) {
  TVM_FFI_CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
}
inline void StreamCreate(deviceStream_t* stream) {
  TVM_FFI_CHECK_CUDA_ERROR(cudaStreamCreate(stream));
}
inline void StreamDestroy(deviceStream_t stream) {
  TVM_FFI_CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

#else

#define TVM_DISCO_DEVICE_NAME "rocm"
#define TVM_DISCO_CCL_NAME "rccl"

using deviceStream_t = hipStream_t;
const constexpr DLDeviceType TVM_DISCO_DEVICE_TYPE = DLDeviceType::kDLROCM;
inline void SetDevice(int device_id) { ROCM_CALL(hipSetDevice(device_id)); }
inline void StreamSynchronize(deviceStream_t stream) { ROCM_CALL(hipStreamSynchronize(stream)); }
inline void StreamCreate(deviceStream_t* stream) { ROCM_CALL(hipStreamCreate(stream)); }
inline void StreamDestroy(deviceStream_t stream) { ROCM_CALL(hipStreamDestroy(stream)); }

#endif

/*! \brief Convert DLPack dtype to ncclDataType. */
inline ncclDataType_t AsNCCLDataType(DLDataType dtype) {
  if (dtype == DLDataType{kDLInt, 8, 1}) {
    return ncclInt8;
  }
  if (dtype == DLDataType{kDLUInt, 8, 1} || dtype == DLDataType{kDLFloat8_e4m3fn, 8, 1} ||
      dtype == DLDataType{kDLFloat8_e5m2, 8, 1}) {
    // For float8 data type, pretend to be uint8 in nccl.
    // And will throw error when allreduce, as it makes no sense in this case.
    return ncclUint8;
  }
  if (dtype == DLDataType{kDLInt, 32, 1}) {
    return ncclInt32;
  }
  if (dtype == DLDataType{kDLUInt, 32, 1}) {
    return ncclUint32;
  }
  if (dtype == DLDataType{kDLInt, 64, 1}) {
    return ncclInt64;
  }
  if (dtype == DLDataType{kDLUInt, 64, 1}) {
    return ncclUint64;
  }
  if (dtype == DLDataType{kDLFloat, 16, 1}) {
    return ncclFloat16;
  }
  if (dtype == DLDataType{kDLFloat, 32, 1}) {
    return ncclFloat32;
  }
  if (dtype == DLDataType{kDLFloat, 64, 1}) {
    return ncclFloat64;
  }
  if (dtype == DLDataType{kDLBfloat, 16, 1}) {
    return ncclBfloat16;
  }
  TVM_FFI_THROW(ValueError) << "Unsupported data type " << dtype;
  throw;
}

struct CCLThreadLocalContext {
  DiscoWorker* worker = nullptr;
  int device_id;
  deviceStream_t default_stream = nullptr;
  ncclComm_t global_comm = nullptr;
  ncclComm_t group_comm = nullptr;

  ~CCLThreadLocalContext() { Clear(); }

  void Clear() {
    if (group_comm) {
      NCCL_CALL(ncclCommDestroy(group_comm));
      if (global_comm == group_comm) {
        global_comm = nullptr;
      }
      group_comm = nullptr;
    }
    if (global_comm) {
      NCCL_CALL(ncclCommDestroy(global_comm));
      global_comm = nullptr;
    }
    if (default_stream) {
      StreamDestroy(default_stream);
      default_stream = nullptr;
    }
    worker = nullptr;
  }

  deviceStream_t GetDefaultStream() {
    const auto func = tvm::ffi::Function::GetGlobal("runtime.get_" TVM_DISCO_DEVICE_NAME "_stream");
    TVM_FFI_ICHECK(func.has_value());
    deviceStream_t stream = static_cast<deviceStream_t>((*func)().cast<void*>());
    return stream == nullptr ? default_stream : stream;
  }

  static CCLThreadLocalContext* Get();
};

}  // namespace nccl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DISCO_NCCL_NCCL_CONTEXT_H_
