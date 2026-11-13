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
 * \brief Tag definitions
 * \file tags.h
 */
#ifndef TVM_TOPI_TAGS_H_
#define TVM_TOPI_TAGS_H_

#include <string>

namespace tvm {
namespace topi {

constexpr auto kElementWise = "elemwise";
constexpr auto kInjective = "injective";
constexpr auto kCommReduce = "comm_reduce";
constexpr auto kCommReduceIdx = "comm_reduce_idx";
constexpr auto kBroadcast = "broadcast";
constexpr auto kMatMul = "matmul";
constexpr auto kConv2dNCHW = "conv2d_nchw";
constexpr auto kConv2dHWCN = "conv2d_hwcn";
constexpr auto kDepthwiseConv2dNCHW = "depthwise_conv2d_nchw";
constexpr auto kDepthwiseConv2dNHWC = "depthwise_conv2d_nhwc";
constexpr auto kDepthwiseConv2dBackInputNHWC = "depthwise_conv2d_back_input_nhwc";
constexpr auto kDepthwiseConv2dBackWeightNHWC = "depthwise_conv2d_back_weight_nhwc";
constexpr auto kEinsum = "einsum";
constexpr auto kGroupConv2d = "group_conv2d";

inline bool is_broadcast(std::string tag) {
  return tag.rfind(kElementWise, 0) == 0 || tag.rfind(kBroadcast, 0) == 0;
}

inline bool is_injective(std::string tag) {
  return tag.rfind(kElementWise, 0) == 0 || tag.rfind(kBroadcast, 0) == 0 ||
         tag.rfind(kInjective, 0) == 0;
}

namespace attr {
constexpr const char* kSrcLayout = "src_layout";
constexpr const char* kDstLayout = "dst_layout";
constexpr const char* kInputShape = "input_shape";
constexpr const char* kLayoutFreePlaceholders = "layout_free_placeholders";
constexpr const char* kConstMatrix = "const_matrix";
constexpr const char* kAutoSchedulerSimplifyConstTensorIndices =
    "auto_scheduler_simplify_const_tensor_indices";
constexpr const char* kWorkload = "workload";
constexpr const char* kAxis = "axis";
}  // namespace attr

}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_TAGS_H_
