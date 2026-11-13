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

#include "fp8_groupwise_scaled_group_gemm_runner_sm100.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace tvm {
namespace runtime {

void tvm_fp8_groupwise_scaled_group_gemm_sm100(ffi::Tensor a, ffi::Tensor b, ffi::Tensor scales_a,
                                               ffi::Tensor scales_b, ffi::Tensor indptr,
                                               ffi::Tensor workspace, int64_t block_size_0,
                                               int64_t block_size_1, ffi::Tensor out) {
  TVM_FFI_ICHECK(a.defined());
  TVM_FFI_ICHECK(a.IsContiguous());
  TVM_FFI_ICHECK(a.numel() == 0 || a.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(b.defined());
  TVM_FFI_ICHECK(b.IsContiguous());
  TVM_FFI_ICHECK(b.numel() == 0 || b.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(scales_a.defined());
  TVM_FFI_ICHECK(scales_a.IsContiguous());
  TVM_FFI_ICHECK(scales_a.numel() == 0 || scales_a.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(scales_b.defined());
  TVM_FFI_ICHECK(scales_b.IsContiguous());
  TVM_FFI_ICHECK(scales_b.numel() == 0 || scales_b.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(indptr.defined());
  TVM_FFI_ICHECK(indptr.IsContiguous());
  TVM_FFI_ICHECK(indptr.numel() == 0 || indptr.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(workspace.defined());
  TVM_FFI_ICHECK(workspace.IsContiguous());
  TVM_FFI_ICHECK(workspace.numel() == 0 || workspace.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(out.defined());
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK(out.numel() == 0 || out.IsAligned(kAllocAlignment));
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommended size is 4MB.
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, a.device().device_id));
  TVM_FFI_ICHECK_EQ(a.ndim(), 2);
  TVM_FFI_ICHECK_EQ(b.ndim(), 3);
  TVM_FFI_ICHECK_EQ(indptr.ndim(), 1);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  int num_groups = b.shape()[0];
  int n = b.shape()[1];
  int k = b.shape()[2];

  TVM_FFI_ICHECK_EQ(scales_a.ndim(), a.ndim());
  TVM_FFI_ICHECK_EQ(scales_b.ndim(), b.ndim());
  // scales_a is row-major of (m, k / block_size)
  TVM_FFI_ICHECK_EQ((k + block_size_1 - 1) / block_size_1, scales_a.shape()[1]);
  TVM_FFI_ICHECK_EQ(scales_a.shape()[0], a.shape()[0]);
  // scales_b is col-major of (k / block_size, n / block_size)
  TVM_FFI_ICHECK_EQ(scales_b.shape()[0], num_groups);
  TVM_FFI_ICHECK_EQ((n + block_size_0 - 1) / block_size_0, scales_b.shape()[1]);
  TVM_FFI_ICHECK_EQ((k + block_size_1 - 1) / block_size_1, scales_b.shape()[2]);

  TVM_FFI_ICHECK_EQ(a.dtype(), (DLDataType{kDLFloat8_e4m3fn, 8, 1}));
  TVM_FFI_ICHECK_EQ(b.dtype(), (DLDataType{kDLFloat8_e4m3fn, 8, 1}));
  TVM_FFI_ICHECK_EQ(scales_a.dtype(), (DLDataType{kDLFloat, 32, 1}));
  TVM_FFI_ICHECK_EQ(scales_b.dtype(), (DLDataType{kDLFloat, 32, 1}));
  TVM_FFI_ICHECK_EQ(indptr.dtype(), (DLDataType{kDLInt, 64, 1}));
  TVM_FFI_ICHECK_EQ(workspace.dtype(), (DLDataType{kDLUInt, 8, 1}));

  if (out.dtype() == DLDataType{kDLFloat, 16, 1}) {
    using Dtype = cutlass::half_t;
    cutlass_fp8_groupwise_scaled_group_gemm_sm100<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                  Dtype, float>(
        static_cast<cutlass::float_e4m3_t*>(a.data_ptr()),
        static_cast<cutlass::float_e4m3_t*>(b.data_ptr()), static_cast<float*>(scales_a.data_ptr()),
        static_cast<float*>(scales_b.data_ptr()), static_cast<int64_t*>(indptr.data_ptr()),
        static_cast<uint8_t*>(workspace.data_ptr()), workspace.shape()[0], n, k, num_groups,
        static_cast<Dtype*>(out.data_ptr()), stream);
  } else if (out.dtype() == DLDataType{kDLBfloat, 16, 1}) {
    using Dtype = cutlass::bfloat16_t;
    cutlass_fp8_groupwise_scaled_group_gemm_sm100<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                                  Dtype, float>(
        static_cast<cutlass::float_e4m3_t*>(a.data_ptr()),
        static_cast<cutlass::float_e4m3_t*>(b.data_ptr()), static_cast<float*>(scales_a.data_ptr()),
        static_cast<float*>(scales_b.data_ptr()), static_cast<int64_t*>(indptr.data_ptr()),
        static_cast<uint8_t*>(workspace.data_ptr()), workspace.shape()[0], n, k, num_groups,
        static_cast<Dtype*>(out.data_ptr()), stream);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("cutlass.groupwise_scaled_group_gemm_e4m3fn_e4m3fn",
                        tvm_fp8_groupwise_scaled_group_gemm_sm100);
}

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
