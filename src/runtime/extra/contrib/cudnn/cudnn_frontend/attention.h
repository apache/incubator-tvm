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
 * \file src/runtime/contrib/cudnn/cudnn_frontend/attention.h
 * \brief cuDNN scale dot product attention implementation
 */

#ifndef TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_FRONTEND_ATTENTION_H_
#define TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_FRONTEND_ATTENTION_H_

#include <cudnn_frontend.h>
#include <tvm/ffi/function.h>

#include <memory>
#include <string>

#define CUDNN_FRONTEND_CALL(func)                             \
  do {                                                        \
    auto status = (func);                                     \
    TVM_FFI_ICHECK(status.is_good()) << status.get_message(); \
  } while (0)

namespace tvm {
namespace contrib {

class CuDNNSDPARunnerObj : public tvm::ffi::Object {
 public:
  CuDNNSDPARunnerObj() {}

  ~CuDNNSDPARunnerObj() {}

  static constexpr const char* _type_key = "contrib.cudnn.SDPARunner";

  void Init(int64_t batch, int64_t seq_len, int64_t num_heads, int64_t num_kv_heads,
            int64_t head_size, int64_t head_size_v, double scale, const DLDataType& data_type,
            const std::string& layout);

  void Run(const DLTensor* qkv, DLTensor* workspace, DLTensor* out);

  static constexpr int kTensorIDQ = 0;
  static constexpr int kTensorIDK = 1;
  static constexpr int kTensorIDV = 2;
  static constexpr int kTensorIDOut = 4;

 private:
  std::unique_ptr<cudnn_frontend::graph::Graph> graph_{nullptr};
  int64_t offset_q_{0};
  int64_t offset_k_{0};
  int64_t offset_v_{0};
};

class CuDNNSDPARunner : public tvm::ffi::ObjectRef {
 public:
  static CuDNNSDPARunner Create() {
    auto n = ffi::make_object<CuDNNSDPARunnerObj>();
    return CuDNNSDPARunner(ffi::UnsafeInit{}, n);
  }

  explicit CuDNNSDPARunner(ffi::UnsafeInit tag, ffi::ObjectPtr<CuDNNSDPARunnerObj> n)
      : tvm::ffi::ObjectRef(tag) {
    data_ = std::move(n);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(CuDNNSDPARunner, tvm::ffi::ObjectRef,
                                                CuDNNSDPARunnerObj);
};

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_FRONTEND_ATTENTION_H_
