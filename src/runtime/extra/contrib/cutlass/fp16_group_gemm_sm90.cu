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

#include "fp16_group_gemm.cuh"
#include "fp16_group_gemm_runner_sm90.cuh"

namespace tvm {
namespace runtime {

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

template <typename ElementA, typename ElementB, typename ElementC>
struct CutlassGroupGemm<90, ElementA, ElementB, ElementC> {
  static void run(ElementA* A, ElementB* B, int64_t* indptr, uint8_t* workspace, int workspace_size,
                  int N, int K, int num_groups, float alpha, float beta, ElementC* C,
                  cudaStream_t stream) {
    cutlass_group_gemm_sm90<ElementA, ElementB, ElementC>(A, B, indptr, workspace, workspace_size,
                                                          N, K, num_groups, alpha, beta, C, stream);
  }
};

void tvm_cutlass_group_gemm_sm90(ffi::Tensor x, ffi::Tensor weight, ffi::Tensor indptr,
                                 ffi::Tensor workspace, ffi::Tensor out) {
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
  TVM_FFI_ICHECK(out.defined());
  TVM_FFI_ICHECK(out.IsContiguous());
  TVM_FFI_ICHECK(out.numel() == 0 || out.IsAligned(kAllocAlignment));
  tvm_cutlass_group_gemm_impl<90>(x, weight, indptr, workspace, out);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("cutlass.group_gemm", tvm_cutlass_group_gemm_sm90);
}

#endif  // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

}  // namespace runtime
}  // namespace tvm
