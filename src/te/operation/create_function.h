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

#ifndef TVM_TE_OPERATION_CREATE_FUNCTION_H_
#define TVM_TE_OPERATION_CREATE_FUNCTION_H_

#include <tvm/ffi/container/array.h>
#include <tvm/te/tensor.h>
#include <tvm/tirx/function.h>

#include <optional>

namespace tvm {
namespace tirx {

/*! \brief Use Tensor Expression to create a schedulable TensorIR func. */
Function CreateFunction(const ffi::Array<te::Tensor>& arg_list,
                        std::optional<PrimType> index_dtype_override = std::nullopt);

/*! \brief Use Tensor Expression to create a schedulable TensorIR func. */
Function CreateFunction(const ffi::Array<ffi::ObjectRef>& arg_list,
                        std::optional<PrimType> index_dtype_override);

}  // namespace tirx
}  // namespace tvm

#endif  // TVM_TE_OPERATION_CREATE_FUNCTION_H_
