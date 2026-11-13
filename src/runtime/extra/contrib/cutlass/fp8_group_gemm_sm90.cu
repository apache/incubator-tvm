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

#include "fp16_group_gemm_runner_sm90.cuh"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <>
struct KernelTraits<cutlass::float_e4m3_t> {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
  using TileShape = Shape<_128, _256, _64>;  // Threadblock-level tile size
  using ClusterShape = Shape<_2, _2, _1>;    // Shape of the threadblocks in a cluster
};

template <>
struct KernelTraits<cutlass::float_e5m2_t> : KernelTraits<cutlass::float_e4m3_t> {};

namespace tvm {
namespace runtime {

template <typename ElementA, typename ElementB, typename ElementC>
void tvm_cutlass_fp8_group_gemm(ffi::Tensor x, ffi::Tensor weight, ffi::Tensor indptr,
                                ffi::Tensor workspace, ffi::Tensor alpha, ffi::Tensor out) {
  TVM_FFI_ICHECK(x.defined());
  TVM_FFI_ICHECK(x.IsContiguous());
  TVM_FFI_ICHECK(x.numel() == 0 || x.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(weight.defined());
  TVM_FFI_ICHECK(weight.IsContiguous());
  TVM_FFI_ICHECK(weight.numel() == 0 || weight.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(indptr.defined());
  TVM_FFI_ICHECK(indptr.IsContiguous());
  TVM_FFI_ICHECK(indptr.numel() == 0 || indptr.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(workspace.defined());
  TVM_FFI_ICHECK(workspace.IsContiguous());
  TVM_FFI_ICHECK(workspace.numel() == 0 || workspace.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(alpha.defined());
  TVM_FFI_ICHECK(alpha.IsContiguous());
  TVM_FFI_ICHECK(alpha.numel() == 0 || alpha.IsAligned(kAllocAlignment));
  TVM_FFI_ICHECK(out.defined());
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK(out.numel() == 0 || out.IsAligned(kAllocAlignment));
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  TVM_FFI_ICHECK_EQ(x.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 3);
  TVM_FFI_ICHECK_EQ(indptr.ndim(), 1);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(alpha.dtype().code, kDLFloat);
  TVM_FFI_ICHECK_EQ(alpha.dtype().bits, 32);
  TVM_FFI_ICHECK_EQ(alpha.ndim(), 1);
  TVM_FFI_ICHECK_EQ(alpha.shape()[0], 1);
  int num_groups = weight.shape()[0];
  int n = weight.shape()[1];
  int k = x.shape()[1];
  const float* beta = nullptr;
  cutlass_group_gemm_sm90(
      static_cast<ElementA*>(x.data_ptr()), static_cast<ElementB*>(weight.data_ptr()),
      static_cast<int64_t*>(indptr.data_ptr()), static_cast<uint8_t*>(workspace.data_ptr()),
      workspace.shape()[0], n, k, num_groups, static_cast<float*>(alpha.data_ptr()), beta,
      static_cast<ElementC*>(out.data_ptr()), stream);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def(
          "cutlass.group_gemm_e5m2_e5m2_fp16",
          tvm_cutlass_fp8_group_gemm<cutlass::float_e5m2_t, cutlass::float_e5m2_t, cutlass::half_t>)
      .def(
          "cutlass.group_gemm_e5m2_e4m3_fp16",
          tvm_cutlass_fp8_group_gemm<cutlass::float_e5m2_t, cutlass::float_e4m3_t, cutlass::half_t>)
      .def("cutlass.group_gemm_e4m3_e4m3_fp16",
           tvm_cutlass_fp8_group_gemm<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                      cutlass::half_t>);
}

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
