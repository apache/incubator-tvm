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

#include "statistical.h"

#include <tvm/ffi/extra/visit_error_context.h>

#include <vector>
namespace tvm {
namespace relax {
namespace distributed {

Type InferDistTypeStatistical(const Call& call, const BlockBuilder& ctx) {
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TensorType data_ty = input_dtensor_tys[0]->tensor_ty;

  const auto* attrs = call->attrs.as<StatisticalAttrs>();

  std::vector<int> axes;
  if (!data_ty->IsUnknownNdim() && attrs->axis.has_value()) {
    axes = NormalizeAxes(call, ctx, data_ty->ndim, attrs->axis.value());
  }

  int out_ndim = 0;
  if (attrs->keepdims) {
    out_ndim = data_ty->ndim;
  } else if (!attrs->axis.has_value()) {
    out_ndim = 0;
  } else if (data_ty->IsUnknownNdim()) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Input of distributed operator must be known ndim";
  } else {
    out_ndim = data_ty->ndim - axes.size();
    TVM_FFI_ICHECK_GE(out_ndim, 0);
  }

  // The inference rule for reduction operator output shapes:
  // - axes is None, keepdims is false -> return the zero-rank shape;
  // - axes is None, keepdims is true -> return the shape whose ndim is the same as input and every
  // value is 1.
  // - axes is not None, keepdims is false -> the returned shape does not contain the input axes.
  // - axes is not None, keepdims is true -> the returned shape has value 1 at the positions of the
  // input axes
  const auto* data_shape = data_ty->shape.as<ShapeExprObj>();

  if (data_shape == nullptr) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Input of distributed operator must be known shape";
  }
  ffi::Array<PrimExpr> out_shape;
  out_shape.reserve(out_ndim);
  for (int i = 0; i < data_ty->ndim; ++i) {
    if (attrs->axis.has_value() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
      out_shape.push_back(data_shape->values[i]);
    } else if (attrs->keepdims) {
      out_shape.push_back(IntImm::Int64(/*value=*/1));
    }
  }
  TVM_FFI_ICHECK_EQ(static_cast<int>(out_shape.size()), out_ndim);
  TensorType output_tensor_ty = TensorType(ShapeExpr(out_shape), data_ty->dtype);

  return InferShardingSpec(call, ctx, output_tensor_ty, distributed::BuildAxisGraphReduce);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  RELAX_REGISTER_STATISTICAL_DIST_INFER_TYPE(max);
  RELAX_REGISTER_STATISTICAL_DIST_INFER_TYPE(mean);
  RELAX_REGISTER_STATISTICAL_DIST_INFER_TYPE(min);
  RELAX_REGISTER_STATISTICAL_DIST_INFER_TYPE(prod);
  RELAX_REGISTER_STATISTICAL_DIST_INFER_TYPE(std);
  RELAX_REGISTER_STATISTICAL_DIST_INFER_TYPE(sum);
  RELAX_REGISTER_STATISTICAL_DIST_INFER_TYPE(variance);
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
