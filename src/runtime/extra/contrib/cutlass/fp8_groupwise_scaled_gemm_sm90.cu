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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/tensor.h>

#include "../cublas/cublas_utils.h"
#include "fp8_groupwise_scaled_gemm.cuh"
#include "fp8_groupwise_scaled_gemm_runner_sm90.cuh"

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

namespace tvm {
namespace runtime {

template <typename TileShape, typename ClusterShape, typename ElementA, typename ElementB,
          typename ElementC, typename ElementBlockScale>
struct CutlassFP8GroupwiseGemm<90, TileShape, ClusterShape, ElementA, ElementB, ElementC,
                               ElementBlockScale> {
  static void run(ElementA* a, ElementB* b, ElementBlockScale* scales_a,
                  ElementBlockScale* scales_b, ElementC* out, uint8_t* workspace,
                  int64_t workspace_size, int64_t m, int64_t n, int64_t k, int64_t l,
                  cudaStream_t stream) {
    cutlass_fp8_groupwise_scaled_mm_sm90<TileShape, ClusterShape, ElementA, ElementB, ElementC,
                                         ElementBlockScale>(
        a, b, scales_a, scales_b, out, workspace, workspace_size, m, n, k, l, stream);
  }
};

void tvm_cutlass_fp8_groupwise_scaled_gemm_sm90(ffi::Tensor a, ffi::Tensor b, ffi::Tensor scales_a,
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
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  tvm_cutlass_fp8_groupwise_scaled_gemm_impl<90, TileShape, ClusterShape>(
      a, b, scales_a, scales_b, workspace, block_size_0, block_size_1, out);
}

void tvm_cutlass_fp8_groupwise_scaled_bmm_sm90(ffi::Tensor a, ffi::Tensor b, ffi::Tensor scales_a,
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
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  tvm_cutlass_fp8_groupwise_scaled_bmm_impl<90, TileShape, ClusterShape>(
      a, b, scales_a, scales_b, workspace, block_size_0, block_size_1, out);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn",
           tvm_cutlass_fp8_groupwise_scaled_gemm_sm90)
      .def("cutlass.groupwise_scaled_bmm_e4m3fn_e4m3fn", tvm_cutlass_fp8_groupwise_scaled_bmm_sm90);
}

}  // namespace runtime
}  // namespace tvm

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
