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

#include <cuda_fp16.h>
#include <float.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include "../cublas/cublas_utils.h"
#include "gemm_runner.cuh"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

struct KernelTraitsM64 {
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _8, _1>;
};

namespace tvm {
namespace runtime {

template <typename ElementA, typename ElementB, typename ElementC>
void tvm_cutlass_fp8_gemm(ffi::Tensor x, ffi::Tensor weight, ffi::Tensor workspace,
                          ffi::Tensor alpha, ffi::Tensor out) {
  TVM_FFI_ICHECK(x.defined());
  TVM_FFI_ICHECK(x.IsContiguous());
  TVM_FFI_ICHECK(x.numel() == 0 || x.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(weight.defined());
  TVM_FFI_ICHECK(weight.IsContiguous());
  TVM_FFI_ICHECK(weight.numel() == 0 || weight.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(workspace.defined());
  TVM_FFI_ICHECK(workspace.IsContiguous());
  TVM_FFI_ICHECK(workspace.numel() == 0 || workspace.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(alpha.defined());
  TVM_FFI_ICHECK(alpha.IsContiguous());
  TVM_FFI_ICHECK(alpha.numel() == 0 || alpha.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(out.defined());
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK(out.numel() == 0 || out.IsAligned(kAllocAlignment));
  // Workspace is used for storing device-side gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));

  TVM_FFI_ICHECK_GE(x.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 2);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 1);
  TVM_FFI_ICHECK_GE(out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(alpha.dtype().code, kDLFloat);
  TVM_FFI_ICHECK_EQ(alpha.dtype().bits, 32);
  TVM_FFI_ICHECK_EQ(alpha.ndim(), 1);
  TVM_FFI_ICHECK_EQ(alpha.shape()[0], 1);
  int64_t m = 1;
  for (int i = 0; i < x.ndim() - 1; ++i) {
    m *= x.shape()[i];
  }
  int64_t n = weight.shape()[0];
  TVM_FFI_ICHECK_EQ(x.shape()[x.ndim() - 1], weight.shape()[1])
      << "Only col-major weight is supported now.";
  int64_t k = x.shape()[x.ndim() - 1];
  const float* beta = nullptr;
  if (m <= 64) {
    cutlass_gemm<KernelTraitsM64>(static_cast<ElementA*>(x.data_ptr()),
                                  static_cast<ElementB*>(weight.data_ptr()),
                                  static_cast<uint8_t*>(workspace.data_ptr()), workspace.shape()[0],
                                  m, n, k, static_cast<float*>(alpha.data_ptr()), beta,
                                  static_cast<ElementC*>(out.data_ptr()), stream);
  } else {
    tvm::contrib::CuBlasLtThreadEntry* cublas_entry =
        tvm::contrib::CuBlasLtThreadEntry::ThreadLocal(x.device());
    tvm::contrib::CallCublasLt(
        cublas_entry->handle, stream, cublas_entry->matmul_pref_desc, x.GetDLTensorPtr(),
        weight.GetDLTensorPtr(), nullptr, alpha.GetDLTensorPtr(), nullptr, out.GetDLTensorPtr(),
        /*transa=*/false, /*transb=*/true, cublas_entry->workspace_ptr,
        cublas_entry->workspace_size, CUBLASLT_EPILOGUE_DEFAULT, std::nullopt);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("cutlass.gemm_e5m2_e5m2_fp16",
           tvm_cutlass_fp8_gemm<cutlass::float_e5m2_t, cutlass::float_e5m2_t, cutlass::half_t>)
      .def("cutlass.gemm_e5m2_e4m3_fp16",
           tvm_cutlass_fp8_gemm<cutlass::float_e5m2_t, cutlass::float_e4m3_t, cutlass::half_t>)
      .def("cutlass.gemm_e4m3_e4m3_fp16",
           tvm_cutlass_fp8_gemm<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t>);
}

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
