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
 * \file runtime/contrib/tensorrt/utils.h
 * \brief Helper functions used by TensorRTBuilder or TensorRTOpConverters.
 */

#ifndef TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_UTILS_H_
#define TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_UTILS_H_

#include <string>
#include <vector>

#include "NvInfer.h"

// This integration targets the TensorRT 10 API. TensorRT 10 removed a large set of APIs the
// pre-TRT10 code relied on (implicit batch, binding indices, addConvolution/addPooling/addPadding,
// IFullyConnectedLayer, IBuilder::setMaxBatchSize, IBuilderConfig::setMaxWorkspaceSize,
// IExecutionContext::execute, obj->destroy(), ...). Emit a clear error instead of a flood of
// "has no member" diagnostics on older releases.
#if !defined(NV_TENSORRT_MAJOR) || NV_TENSORRT_MAJOR < 10
#error \
    "TVM's TensorRT runtime requires TensorRT 10.0 or newer (or set TVM_USE_TENSORRT_RUNTIME=OFF)."
#endif

// There is a conflict between cpplint and clang-format-10.
// clang-format off
#define TRT_VERSION_GE(major, minor, patch)                                                    \
  ((NV_TENSORRT_MAJOR > major) || (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR > minor) || \
  (NV_TENSORRT_MAJOR == major && NV_TENSORRT_MINOR == minor && NV_TENSORRT_PATCH >= patch))
// clang-format on

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief Helper function to convert a vector-like container to TRT Dims.
 * \param vec A container supporting size() and operator[] (e.g. std::vector or ffi::Array).
 * \return TRT Dims.
 */
template <typename Container>
inline nvinfer1::Dims VectorToTrtDims(const Container& vec) {
  nvinfer1::Dims dims;
  // Dims(nbDims=0, d[0]=1) is used to represent a scalar in TRT.
  dims.d[0] = 1;
  dims.nbDims = static_cast<int32_t>(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    dims.d[i] = static_cast<int64_t>(vec[i]);
  }
  return dims;
}

/*!
 * \brief Helper function to convert TRT Dims to vector.
 * \param vec TRT Dims.
 * \return Vector.
 */
inline std::vector<int> TrtDimsToVector(const nvinfer1::Dims& dims) {
  return std::vector<int>(dims.d, dims.d + dims.nbDims);
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TENSORRT_TENSORRT_UTILS_H_
