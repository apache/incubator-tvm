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
 * \file s_tir/tensor_region.cc
 * \brief Tensor region construction shared by S-TIR passes.
 */
#include "tensor_region.h"

#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>

namespace tvm::s_tir {
using namespace tvm::prim;

TensorRegion FullTensorRegion(tirx::TensorVar tensor) {
  ffi::Array<Range> region;
  for (PrimExpr extent : tensor.shape()) {
    region.push_back(Range::FromMinExtent(0, extent));
  }
  return TensorRegion(tensor, region, TensorRegionType());
}

TensorRegion TensorRegionFromPoint(tirx::TensorVar tensor, ffi::Array<PrimExpr> indices) {
  TVM_FFI_ICHECK_EQ(tensor.shape().size(), indices.size())
      << "Tensor rank and region dimension mismatch";
  ffi::Array<Range> region;
  for (const PrimExpr& index : indices) {
    if (const prim::RampObj* ramp_index = index.as<prim::RampObj>()) {
      region.push_back(
          Range::FromMinExtent(ramp_index->base, ramp_index->stride * ramp_index->lanes));
    } else {
      region.push_back(Range::FromMinExtent(index, MakeConst(index.ty(), 1)));
    }
  }
  return TensorRegion(tensor, region, TensorRegionType());
}

}  // namespace tvm::s_tir
