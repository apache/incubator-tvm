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

/*!
 * \file src/runtime/contrib/dnnl/dnnl_kernel.h
 * \brief Use external dnnl library kernels.
 */

#ifndef TVM_RUNTIME_CONTRIB_DNNL_DNNL_KERNEL_H_
#define TVM_RUNTIME_CONTRIB_DNNL_DNNL_KERNEL_H_

#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/extra_base.h>

#include <vector>

#include "dnnl.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace dnnl;

extern "C" TVM_RUNTIME_EXTRA_DLL void dnnl_conv2d(float* data, float* weights, float* out, int p_N_,
                                                  int p_C_, int p_H_, int p_W_, int p_O_, int p_G_,
                                                  int p_Ph0_, int p_Pw0_, int p_Ph1_, int p_Pw1_,
                                                  int p_Kh_, int p_Kw_, int p_Sh_, int p_Sw_);

extern "C" TVM_RUNTIME_EXTRA_DLL void dnnl_fused_conv2d_relu(float* data, float* weights,
                                                             float* out, int p_N_, int p_C_,
                                                             int p_H_, int p_W_, int p_O_, int p_G_,
                                                             int p_Ph0_, int p_Pw0_, int p_Ph1_,
                                                             int p_Pw1_, int p_Kh_, int p_Kw_,
                                                             int p_Sh_, int p_Sw_);

extern "C" TVM_RUNTIME_EXTRA_DLL void dnnl_fused_conv2d_bias_relu(
    float* data, float* weights, float* bias, float* out, int p_N_, int p_C_, int p_H_, int p_W_,
    int p_O_, int p_G_, int p_Ph0_, int p_Pw0_, int p_Ph1_, int p_Pw1_, int p_Kh_, int p_Kw_,
    int p_Sh_, int p_Sw_);

extern "C" TVM_RUNTIME_EXTRA_DLL void dnnl_dense(float* data, float* weight, float* out, int p_B_,
                                                 int p_I_, int p_O_);

extern "C" TVM_RUNTIME_EXTRA_DLL void dnnl_relu(float* data, float* out,
                                                std::vector<int64_t> shape);

extern "C" TVM_RUNTIME_EXTRA_DLL void dnnl_bn(float* data, float* gamma, float* beta, float* mean,
                                              float* variance, float* out, float* new_mean,
                                              float* new_variance, int p_n_, int p_c_, int p_h_,
                                              int p_w_, int p_e_);

extern "C" TVM_RUNTIME_EXTRA_DLL void dnnl_binary_op(float* data, float* weight, float* out,
                                                     int binary_algo, std::vector<int64_t> shape);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_DNNL_DNNL_KERNEL_H_
