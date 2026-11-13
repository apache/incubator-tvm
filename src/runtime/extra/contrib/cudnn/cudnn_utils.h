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
 * \file Use external cudnn utils function
 */

#ifndef TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_
#define TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_

#include <cudnn.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/runtime/device_api.h>

#include <string>

namespace tvm {
namespace contrib {

#define CUDNN_CALL(func)                                                               \
  {                                                                                    \
    cudnnStatus_t e = (func);                                                          \
    TVM_FFI_ICHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  }

/*! breif Convert DLTensor type to CuDNN type */
struct CuDNNDataType {
  static cudnnDataType_t DLTypeToCuDNNType(const DLDataType& dtype);
  template <int v>
  static const void* GetConst(cudnnDataType_t type);
};  // struct CuDNNDataType

inline void GetStride(int nbdim, const int* dims, int* strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >= 0; --i) {
    mul *= dims[i];
    strides[i] = mul;
  }
}

inline void GetCudnnStride(int nbdim, const int* dims, int* strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >= 0; --i) {
    strides[i] = mul;
    mul *= dims[i];
  }
}

struct ConvEntry {
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionMode_t mode{CUDNN_CROSS_CORRELATION};
  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  cudnnTensorDescriptor_t input_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnActivationDescriptor_t activation_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  // cudnnMathType_t math_type;
  Device device;
  runtime::DeviceAPI* cuda_api;
  ffi::Tensor workspace_owner;
  void* workspace{nullptr};
  size_t workspace_size{0};
  ConvEntry();
  ~ConvEntry();
  void UpdateWorkspace(const size_t wsize);
  void CleanWorkspace();
};  // ConvThreadEntry

struct SoftmaxEntry {
  cudnnSoftmaxMode_t mode;
  cudnnDataType_t data_type;
  cudnnTensorDescriptor_t shape_desc;
  SoftmaxEntry();
  ~SoftmaxEntry();
};  // SoftmaxEntry

struct CuDNNThreadEntry {
  CuDNNThreadEntry();
  ~CuDNNThreadEntry();

  bool exists() const { return handle; }

  cudnnHandle_t handle{nullptr};
  ConvEntry conv_entry;
  SoftmaxEntry softmax_entry;
  runtime::DeviceAPI* cuda_api{nullptr};
  static CuDNNThreadEntry* ThreadLocal(Device curr_device, bool check_exists = true);
};  // CuDNNThreadEntry

void SetConvDescriptors(CuDNNThreadEntry* entry_ptr, int format, int dims, int groups,
                        const int pad[], const int stride[], const int dilation[], int64_t x_dim[],
                        int64_t w_dim[], int64_t y_dim[], DLDataType data_dtype,
                        const std::string& conv_dtype);

void FindAlgo(int format, int dims, int groups, const int pad[], const int stride[],
              const int dilation[], const int x_dim[], const int w_dim[], const int y_dim[],
              const std::string& data_dtype, const std::string& conv_dtype, bool verbose,
              ffi::Any* ret);

void ConvolutionForward(int mode, int format, int algo, int dims, int groups, const int pad[],
                        const int stride[], const int dilation[], const DLTensor* x,
                        const DLTensor* w, const DLTensor* y, const std::string& conv_dtype);

void ConvolutionBiasActivationForward(int mode, int format, int algo, int dims, int groups, int act,
                                      double coef, const int pad[], const int stride[],
                                      const int dilation[], const DLTensor* x, const DLTensor* w,
                                      const DLTensor* y, const DLTensor* bias,
                                      const std::string& conv_dtype);

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_
