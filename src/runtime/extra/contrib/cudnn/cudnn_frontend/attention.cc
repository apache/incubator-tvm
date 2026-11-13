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
 * \file src/runtime/contrib/cudnn/cudnn_frontend/attention.cc
 * \brief cuDNN scale dot product attention implementation
 */

#include "./attention.h"

#include <tvm/ffi/extra/cuda/base.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/device_api.h>

#include "../cudnn_utils.h"

namespace tvm {
namespace contrib {

void CuDNNSDPARunnerObj::Init(int64_t batch, int64_t seq_len, int64_t num_heads,
                              int64_t num_kv_heads, int64_t head_size, int64_t head_size_v,
                              double scale, const DLDataType& data_type,
                              const std::string& layout) {
  graph_ = std::make_unique<cudnn_frontend::graph::Graph>();

  TVM_FFI_ICHECK(data_type.code == DLDataTypeCode::kDLFloat && data_type.bits == 16)
      << "Only float16 is supported";

  graph_->set_io_data_type(cudnn_frontend::DataType_t::HALF)
      .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

  auto q_desc = cudnn_frontend::graph::Tensor_attributes().set_name("Q").set_uid(kTensorIDQ);
  auto k_desc = cudnn_frontend::graph::Tensor_attributes().set_name("K").set_uid(kTensorIDK);
  auto v_desc = cudnn_frontend::graph::Tensor_attributes().set_name("V").set_uid(kTensorIDV);
  auto o_desc = cudnn_frontend::graph::Tensor_attributes().set_name("Out").set_uid(kTensorIDOut);

  std::vector<int64_t> q_stride, k_stride, v_stride,
      o_stride;  // stride in the order of (batch, num_heads, seq_len, head_size)

  if (layout == "BS3NH") {
    int64_t stride_H = 1;
    int64_t q_stride_N = head_size;
    int64_t k_stride_N = head_size;
    int64_t v_stride_N = head_size_v;
    int64_t stride_S =
        num_heads * q_stride_N + num_kv_heads * k_stride_N + num_kv_heads * v_stride_N;
    int64_t stride_B = stride_S * seq_len;
    q_stride = {stride_B, q_stride_N, stride_S, stride_H};
    k_stride = {stride_B, k_stride_N, stride_S, stride_H};
    v_stride = {stride_B, v_stride_N, stride_S, stride_H};
    o_stride = {seq_len * num_heads * head_size_v, head_size_v, num_heads * head_size_v, 1};
    offset_k_ = num_heads * head_size;
    offset_v_ = offset_k_ + num_kv_heads * head_size;
  } else if (layout == "SBN3H") {
    TVM_FFI_ICHECK_EQ(num_kv_heads, num_heads);
    int64_t stride_H = 1;
    int64_t stride_N = head_size + head_size + head_size_v;
    int64_t stride_B = num_heads * stride_N;
    int64_t stride_S = stride_B * batch;
    q_stride = k_stride = v_stride = {stride_B, stride_N, stride_S, stride_H};
    o_stride = {num_heads * head_size_v, head_size_v, num_heads * head_size_v * batch, 1};
    offset_k_ = head_size;
    offset_v_ = offset_k_ * 2;
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported layout: " << layout;
  }

  q_desc = q_desc.set_dim({batch, num_heads, seq_len, head_size}).set_stride(q_stride);
  k_desc = k_desc.set_dim({batch, num_kv_heads, seq_len, head_size}).set_stride(k_stride);
  v_desc = v_desc.set_dim({batch, num_kv_heads, seq_len, head_size_v}).set_stride(v_stride);
  auto sdpa_options = cudnn_frontend::graph::SDPA_attributes()
                          .set_name("flash_attention")
                          .set_is_inference(true)
                          .set_alibi_mask(false)
                          .set_causal_mask(false)
                          .set_attn_scale(scale);

  auto q = graph_->tensor(q_desc);
  auto k = graph_->tensor(k_desc);
  auto v = graph_->tensor(v_desc);
  auto [o, stats] = graph_->sdpa(q, k, v, sdpa_options);
  TVM_FFI_ICHECK(stats == nullptr);
  o->set_output(true).set_dim({batch, num_heads, seq_len, head_size_v}).set_stride(o_stride);
  int device_id;
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetDevice(&device_id));
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal(DLDevice{kDLCUDA, device_id});
  CUDNN_FRONTEND_CALL(graph_->build(entry_ptr->handle, {cudnn_frontend::HeurMode_t::A}));
}

void CuDNNSDPARunnerObj::Run(const DLTensor* qkv, DLTensor* workspace, DLTensor* out) {
  auto* qkv_base = reinterpret_cast<uint8_t*>(qkv->data) + qkv->byte_offset;
  auto* q_ptr = reinterpret_cast<uint16_t*>(qkv_base) + offset_q_;
  auto* k_ptr = reinterpret_cast<uint16_t*>(qkv_base) + offset_k_;
  auto* v_ptr = reinterpret_cast<uint16_t*>(qkv_base) + offset_v_;
  auto* out_ptr = reinterpret_cast<uint8_t*>(out->data) + out->byte_offset;

  size_t workspace_size = graph_->get_workspace_size();
  TVM_FFI_ICHECK_LE(workspace_size, workspace->shape[0]) << "Workspace size too small";
  std::unordered_map<cudnn_frontend::graph::Tensor_attributes::uid_t, void*> inputs = {
      {kTensorIDQ, q_ptr}, {kTensorIDK, k_ptr}, {kTensorIDV, v_ptr}, {kTensorIDOut, out_ptr}};

  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal(qkv->device);
  CUDNN_FRONTEND_CALL(graph_->execute(entry_ptr->handle, inputs, workspace->data));
}

}  // namespace contrib
}  // namespace tvm
