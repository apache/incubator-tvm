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
#ifndef TVM_S_TIR_FUNCTION_H_
#define TVM_S_TIR_FUNCTION_H_

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/accessor.h>

namespace tvm {
namespace s_tir {
namespace attr {
/*!
 * \brief Mark the function as scheduled, so the default schedule will pass will skip it.
 *
 * Type: bool
 */
inline constexpr ffi::reflection::AttrKey<bool> kIsScheduled{"tirx.is_scheduled"};

/*! \brief Mark the tensors which is const access and can be transformed layout. */
inline constexpr ffi::reflection::AttrKey<ffi::Array<int64_t>> kLayoutFreeTensors{
    "layout_free_tensors"};
/*!
 * \brief Marks the layout transforms to be used for a tensor.
 *
 * Only applies to a tensor-like input, as it should be made part of the
 * Function attributes for TIR.
 */
constexpr const char* kLayoutTransforms = "layout_transforms";
inline constexpr ffi::reflection::AttrKey<int64_t> kEstimatedFlops{"estimated_flops"};

}  // namespace attr
}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_FUNCTION_H_
