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
 * \file s_tir/tensor_region.h
 * \brief Tensor region construction shared by S-TIR passes.
 */
#ifndef TVM_S_TIR_TENSOR_REGION_H_
#define TVM_S_TIR_TENSOR_REGION_H_

#include <tvm/tirx/expr.h>

namespace tvm::s_tir {

/*! \brief Select the entire tensor. */
TensorRegion FullTensorRegion(tirx::TensorVar tensor);

/*! \brief Construct unit or vector-lane ranges from point indices. */
TensorRegion TensorRegionFromPoint(tirx::TensorVar tensor, ffi::Array<PrimExpr> indices);

}  // namespace tvm::s_tir

#endif  // TVM_S_TIR_TENSOR_REGION_H_
