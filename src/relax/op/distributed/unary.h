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
 * \file unary.h
 * \brief The functions to infer type for distributed unary operator
 */

#ifndef TVM_RELAX_OP_DISTRIBUTED_UNARY_H_
#define TVM_RELAX_OP_DISTRIBUTED_UNARY_H_

#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/relax/op_attr.h>

#include "utils.h"

namespace tvm {
namespace relax {
namespace distributed {

template <bool require_float_dtype, typename FType>
Type InferDistTypeUnary(const Call& call, const BlockBuilder& ctx, FType f_compute_out_dtype) {
  ffi::Array<distributed::DTensorType> input_dtensor_tys = GetInputDTensorType(call, ctx);
  TVM_FFI_ICHECK(input_dtensor_tys.size() == 1);
  distributed::DTensorType input_dtensor_ty = input_dtensor_tys[0];
  TensorType input_tensor_ty = input_dtensor_ty->tensor_ty;

  // Unary op validation preserves the old float-kind check; lanes do not affect this policy.
  if (require_float_dtype && !input_tensor_ty->IsUnknownDtype() &&
      !input_tensor_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLFloat)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << call->op
        << " requires the input tensor to have float dtype. However, the given input dtype is "
        << input_tensor_ty->dtype;
  }
  auto output_ty = ffi::make_object<TensorTypeObj>(*input_tensor_ty.get());
  ffi::Optional<PrimType> computed_dtype = f_compute_out_dtype(input_tensor_ty);
  output_ty->dtype = computed_dtype;
  TensorType out_tensor_ty(output_ty);
  return distributed::DTensorType(out_tensor_ty, input_dtensor_ty->device_mesh,
                                  input_dtensor_ty->placement);
}

template <bool require_float_dtype>
Type InferDistTypeUnaryArith(const Call& call, const BlockBuilder& ctx) {
  return InferDistTypeUnary<require_float_dtype>(
      call, ctx, [](const TensorType& input_ty) { return input_ty->dtype; });
}

Type InferDistTypeUnaryCheck(const Call& call, const BlockBuilder& ctx);

// clang-format off
#define RELAX_REGISTER_UNARY_ARITH_DIST_INFER_TYPE(OpName, RequireFloatDtype) \
  OpDef("relax." #OpName)                                                     \
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeUnaryArith<RequireFloatDtype>)

#define RELAX_REGISTER_UNARY_CHECK_DIST_INFER_TYPE(OpName) \
  OpDef("relax." #OpName)                              \
      .set_attr(tvm::relax::distributed::op_attr::kInferTypeWithBuilder, InferDistTypeUnaryCheck)
// clang-format on

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_DISTRIBUTED_UNARY_H_
