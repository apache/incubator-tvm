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
#include <tvm/runtime/tensor.h>

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"

namespace tvm {
namespace runtime {

template <int Arch, typename ElementA, typename ElementB, typename ElementC>
struct CutlassGroupGemm;

template <int Arch>
void tvm_cutlass_group_gemm_impl(ffi::Tensor x, ffi::Tensor weight, ffi::Tensor indptr,
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
  // Workspace is used for storing device-side group gemm arguments and cutlass internal workspace.
  // Recommened size is 4MB.
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, x.device().device_id));
  TVM_FFI_ICHECK_EQ(x.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 3);
  TVM_FFI_ICHECK_EQ(indptr.ndim(), 1);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 1);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  int num_groups = weight.shape()[0];
  int n = weight.shape()[1];
  int k = weight.shape()[2];
  float alpha = 1.0f;
  float beta = 0.0f;

  if (x.dtype() == DLDataType{kDLFloat, 16, 1}) {
    TVM_FFI_ICHECK((weight.dtype() == DLDataType{kDLFloat, 16, 1}));
    TVM_FFI_ICHECK((out.dtype() == DLDataType{kDLFloat, 16, 1}));
    using Dtype = cutlass::half_t;
    CutlassGroupGemm<Arch, Dtype, Dtype, Dtype>::run(
        static_cast<Dtype*>(x.data_ptr()), static_cast<Dtype*>(weight.data_ptr()),
        static_cast<int64_t*>(indptr.data_ptr()), static_cast<uint8_t*>(workspace.data_ptr()),
        workspace.shape()[0], n, k, num_groups, alpha, beta, static_cast<Dtype*>(out.data_ptr()),
        stream);
  } else if (x.dtype() == DLDataType{kDLBfloat, 16, 1}) {
    TVM_FFI_ICHECK((weight.dtype() == DLDataType{kDLBfloat, 16, 1}));
    TVM_FFI_ICHECK((out.dtype() == DLDataType{kDLBfloat, 16, 1}));
    using Dtype = cutlass::bfloat16_t;
    CutlassGroupGemm<Arch, Dtype, Dtype, Dtype>::run(
        static_cast<Dtype*>(x.data_ptr()), static_cast<Dtype*>(weight.data_ptr()),
        static_cast<int64_t*>(indptr.data_ptr()), static_cast<uint8_t*>(workspace.data_ptr()),
        workspace.shape()[0], n, k, num_groups, alpha, beta, static_cast<Dtype*>(out.data_ptr()),
        stream);
  }
}

}  // namespace runtime
}  // namespace tvm
