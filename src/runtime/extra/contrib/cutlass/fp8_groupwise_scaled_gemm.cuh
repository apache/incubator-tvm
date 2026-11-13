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
#include <tvm/runtime/logging.h>
#include <tvm/runtime/tensor.h>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

namespace tvm {
namespace runtime {

template <int Arch, typename TileShape, typename ClusterShape, typename ElementA, typename ElementB,
          typename ElementC, typename ElementBlockScale>
struct CutlassFP8GroupwiseGemm;

template <int Arch, typename TileShape, typename ClusterShape>
void tvm_cutlass_fp8_groupwise_scaled_gemm_impl(ffi::Tensor a, ffi::Tensor b, ffi::Tensor scales_a,
                                                ffi::Tensor scales_b, ffi::Tensor workspace,
                                                int64_t block_size_0, int64_t block_size_1,
                                                ffi::Tensor out) {
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
  TVM_FFI_ICHECK(workspace.defined());
  TVM_FFI_ICHECK(workspace.IsContiguous());
  TVM_FFI_ICHECK(workspace.numel() == 0 || workspace.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(out.defined());
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK(out.numel() == 0 || out.IsAligned(kAllocAlignment));
  // Workspace is used for storing device-side gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, a.device().device_id));

  TVM_FFI_ICHECK_GE(a.ndim(), 2);
  TVM_FFI_ICHECK_EQ(scales_a.ndim(), a.ndim());
  TVM_FFI_ICHECK_EQ(b.ndim(), 2);
  TVM_FFI_ICHECK_EQ(scales_b.ndim(), 2);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), a.ndim());
  int64_t m = 1;
  for (int64_t i = 0; i < a.ndim() - 1; ++i) {
    m *= a.shape()[i];
  }
  int64_t n = b.shape()[0];
  TVM_FFI_ICHECK_EQ(a.shape()[a.ndim() - 1], b.shape()[1]) << "Only col-major B is supported now.";
  int64_t k = a.shape()[a.ndim() - 1];

  // scales_a is col-major of (*a_shape[:-1], k / block_size)
  TVM_FFI_ICHECK_EQ(scales_a.shape()[0] * block_size_1, k);
  for (int64_t i = 1; i < scales_a.ndim(); ++i) {
    TVM_FFI_ICHECK_EQ(scales_a.shape()[i], a.shape()[i - 1]);
  }
  // scales_b is col-major of (k / block_size, n / block_size)
  TVM_FFI_ICHECK_EQ((n + block_size_0 - 1) / block_size_0, scales_b.shape()[0]);
  TVM_FFI_ICHECK_EQ(scales_b.shape()[1] * block_size_1, k);

  TVM_FFI_ICHECK_EQ(a.dtype(), (DLDataType{kDLFloat8_e4m3fn, 8, 1}));
  TVM_FFI_ICHECK_EQ(b.dtype(), (DLDataType{kDLFloat8_e4m3fn, 8, 1}));
  TVM_FFI_ICHECK_EQ(scales_a.dtype(), (DLDataType{kDLFloat, 32, 1}));
  TVM_FFI_ICHECK_EQ(scales_b.dtype(), (DLDataType{kDLFloat, 32, 1}));
  TVM_FFI_ICHECK_EQ(workspace.dtype(), (DLDataType{kDLUInt, 8, 1}));
  int64_t workspace_nbytes =
      workspace.shape()[0] * ((workspace.dtype().bits * workspace.dtype().lanes + 7) / 8);

  if (out.dtype() == DLDataType{kDLFloat, 16, 1}) {
    CutlassFP8GroupwiseGemm<Arch, TileShape, ClusterShape, cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, cutlass::half_t,
                            float>::run(static_cast<cutlass::float_e4m3_t*>(a.data_ptr()),
                                        static_cast<cutlass::float_e4m3_t*>(b.data_ptr()),
                                        static_cast<float*>(scales_a.data_ptr()),
                                        static_cast<float*>(scales_b.data_ptr()),
                                        static_cast<cutlass::half_t*>(out.data_ptr()),
                                        static_cast<uint8_t*>(workspace.data_ptr()),
                                        workspace_nbytes, m, n, k, 1, stream);
  } else if (out.dtype() == DLDataType{kDLBfloat, 16, 1}) {
    CutlassFP8GroupwiseGemm<Arch, TileShape, ClusterShape, cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, cutlass::bfloat16_t,
                            float>::run(static_cast<cutlass::float_e4m3_t*>(a.data_ptr()),
                                        static_cast<cutlass::float_e4m3_t*>(b.data_ptr()),
                                        static_cast<float*>(scales_a.data_ptr()),
                                        static_cast<float*>(scales_b.data_ptr()),
                                        static_cast<cutlass::bfloat16_t*>(out.data_ptr()),
                                        static_cast<uint8_t*>(workspace.data_ptr()),
                                        workspace_nbytes, m, n, k, 1, stream);
  } else {
    LOG(FATAL) << "Unsupported output dtype: " << out.dtype();
  }
}

template <int Arch, typename TileShape, typename ClusterShape>
void tvm_cutlass_fp8_groupwise_scaled_bmm_impl(ffi::Tensor a, ffi::Tensor b, ffi::Tensor scales_a,
                                               ffi::Tensor scales_b, ffi::Tensor workspace,
                                               int64_t block_size_0, int64_t block_size_1,
                                               ffi::Tensor out) {
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
  TVM_FFI_ICHECK(workspace.defined());
  TVM_FFI_ICHECK(workspace.IsContiguous());
  TVM_FFI_ICHECK(workspace.numel() == 0 || workspace.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(out.defined());
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK(out.numel() == 0 || out.IsAligned(kAllocAlignment));
  // Workspace is used for storing device-side gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, a.device().device_id));

  TVM_FFI_ICHECK_EQ(a.ndim(), 3);
  TVM_FFI_ICHECK_EQ(scales_a.ndim(), 3);
  TVM_FFI_ICHECK_EQ(b.ndim(), 3);
  TVM_FFI_ICHECK_EQ(scales_b.ndim(), 3);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), 3);
  int64_t batch_size = a.shape()[0];
  int64_t m = a.shape()[1];
  int64_t n = b.shape()[1];
  TVM_FFI_ICHECK_EQ(a.shape()[2], b.shape()[2]) << "Only col-major B is supported now.";
  int64_t k = a.shape()[2];
  TVM_FFI_ICHECK_EQ(b.shape()[0], batch_size);
  TVM_FFI_ICHECK_EQ(scales_a.shape()[0], batch_size);
  TVM_FFI_ICHECK_EQ(scales_b.shape()[0], batch_size);
  TVM_FFI_ICHECK_EQ(out.shape()[0], batch_size);

  // scales_a is col-major of (batch_size, m, k / block_size)
  TVM_FFI_ICHECK_EQ(scales_a.shape()[1] * block_size_1, k);
  TVM_FFI_ICHECK_EQ(scales_a.shape()[2], m);
  // scales_b is col-major of (k / block_size, n / block_size)
  TVM_FFI_ICHECK_EQ(scales_b.shape()[1] * block_size_0, n);
  TVM_FFI_ICHECK_EQ(scales_b.shape()[2] * block_size_1, k);

  TVM_FFI_ICHECK_EQ(a.dtype(), (DLDataType{kDLFloat8_e4m3fn, 8, 1}));
  TVM_FFI_ICHECK_EQ(b.dtype(), (DLDataType{kDLFloat8_e4m3fn, 8, 1}));
  TVM_FFI_ICHECK_EQ(scales_a.dtype(), (DLDataType{kDLFloat, 32, 1}));
  TVM_FFI_ICHECK_EQ(scales_b.dtype(), (DLDataType{kDLFloat, 32, 1}));
  TVM_FFI_ICHECK_EQ(workspace.dtype(), (DLDataType{kDLUInt, 8, 1}));
  int64_t workspace_nbytes =
      workspace.shape()[0] * ((workspace.dtype().bits * workspace.dtype().lanes + 7) / 8);

  if (out.dtype() == DLDataType{kDLFloat, 16, 1}) {
    CutlassFP8GroupwiseGemm<Arch, TileShape, ClusterShape, cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, cutlass::half_t,
                            float>::run(static_cast<cutlass::float_e4m3_t*>(a.data_ptr()),
                                        static_cast<cutlass::float_e4m3_t*>(b.data_ptr()),
                                        static_cast<float*>(scales_a.data_ptr()),
                                        static_cast<float*>(scales_b.data_ptr()),
                                        static_cast<cutlass::half_t*>(out.data_ptr()),
                                        static_cast<uint8_t*>(workspace.data_ptr()),
                                        workspace_nbytes, m, n, k, batch_size, stream);
  } else if (out.dtype() == DLDataType{kDLBfloat, 16, 1}) {
    CutlassFP8GroupwiseGemm<Arch, TileShape, ClusterShape, cutlass::float_e4m3_t,
                            cutlass::float_e4m3_t, cutlass::bfloat16_t,
                            float>::run(static_cast<cutlass::float_e4m3_t*>(a.data_ptr()),
                                        static_cast<cutlass::float_e4m3_t*>(b.data_ptr()),
                                        static_cast<float*>(scales_a.data_ptr()),
                                        static_cast<float*>(scales_b.data_ptr()),
                                        static_cast<cutlass::bfloat16_t*>(out.data_ptr()),
                                        static_cast<uint8_t*>(workspace.data_ptr()),
                                        workspace_nbytes, m, n, k, batch_size, stream);
  } else {
    LOG(FATAL) << "Unsupported output dtype: " << out.dtype();
  }
}

}  // namespace runtime
}  // namespace tvm
