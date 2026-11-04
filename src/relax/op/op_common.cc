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

#include "op_common.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>

#include <algorithm>
#include <sstream>

namespace tvm {
namespace relax {
using namespace tvm::prim;

ffi::Array<Expr> GetCallArgs(const Call& call) {
  static const Op call_tir_op = Op::Get("relax.call_tir");
  ffi::Array<Expr> args;
  if (call->op.same_as(call_tir_op)) {
    args = call->args[1].as_or_throw<Tuple>()->fields;
  } else {
    args = call->args;
  }
  return args;
}

void CheckNumArguments(const Call& call) {
  Op op = call->op.as_or_throw<Op>();
  int expected_input = op->args_info.size();
  if (op->var_args_info.has_value() ? call->args.size() < op->args_info.size()
                                    : call->args.size() != op->args_info.size()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "Operator " << op << " expects " << (op->var_args_info.has_value() ? "at least " : "")
        << expected_input << " arguments"
        << ", but was called with " << call->args.size() << " arguments";
  }
}

TensorType GetInputTensorType(const Call& call, size_t i_arg) {
  Op op = call->op.as_or_throw<Op>();

  TVM_FFI_ICHECK(op->var_args_info.has_value() ? call->args.size() >= op->args_info.size()
                                               : call->args.size() == op->args_info.size())
      << "Failure caught by this check "
      << "should have previously been caught by `CheckNumArguments`";
  TVM_FFI_ICHECK_LT(i_arg, call->args.size());

  auto arg = call->args[i_arg];
  auto ty = GetType(arg);

  if (auto tensor_ty = ty.as<TensorType>()) {
    return tensor_ty.value();
  } else {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Operator " << op << " requires argument " << i_arg
        << (i_arg < op->args_info.size() ? " (" + std::string(op->args_info[i_arg]->name) + ")"
                                         : "")
        << " to be a tensor.  "
        << "However, the argument " << arg << " is instead of type " << ty;
    TVM_FFI_UNREACHABLE();
  }
}

ffi::Array<TensorType> GetInputTensorType(const Call& call) {
  CheckNumArguments(call);

  Op op = call->op.as_or_throw<Op>();
  ffi::Array<TensorType> input_tensor_ty;
  for (size_t i = 0; i < call->args.size(); ++i) {
    input_tensor_ty.push_back(GetInputTensorType(call, i));
  }
  return input_tensor_ty;
}

ffi::Array<TensorType> GetTensorTypeFromTuple(const Call& call, const Expr& tup) {
  const auto* tuple_ty = GetTypeAs<TupleTypeObj>(tup);
  if (tuple_ty == nullptr) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << call->op << " expects the input to be a Tuple of Tensors. However, the given input is "
        << tup->ty->GetTypeKey();
  }

  ffi::Array<TensorType> tensor_ty;
  tensor_ty.reserve(tuple_ty->fields.size());
  for (Type field_ty : tuple_ty->fields) {
    const auto* field_tensor_ty = field_ty.as<TensorTypeObj>();
    if (field_tensor_ty == nullptr) {
      TVM_FFI_VISIT_THROW(TypeError, call)
          << call->op << " expects the input to be a Tuple of Tensors. However, the given input is "
          << tup->ty;
    }
    tensor_ty.push_back(ffi::GetRef<TensorType>(field_tensor_ty));
  }
  return tensor_ty;
}

BinaryBroadcastShapeInferResult InferBinaryBroadcastShape(sym::AnalyzerObj* analyzer,
                                                          const ffi::Array<PrimExpr>& x1_shape,
                                                          const ffi::Array<PrimExpr>& x2_shape) {
  BinaryBroadcastShapeInferResult result;
  int x1_ndim = x1_shape.size();
  int x2_ndim = x2_shape.size();
  int max_ndim = std::max(x1_ndim, x2_ndim);

  std::vector<PrimExpr> output_shape;
  output_shape.reserve(max_ndim);

  int i = 1;
  for (; i <= std::min(x1_ndim, x2_ndim); ++i) {
    const PrimExpr& dim0 = x1_shape[x1_ndim - i];
    const PrimExpr& dim1 = x2_shape[x2_ndim - i];
    const auto* int_dim0 = dim0.as<IntImmObj>();
    const auto* int_dim1 = dim1.as<IntImmObj>();
    if (int_dim0 != nullptr && int_dim0->value == 1) {
      output_shape.push_back(dim1);
    } else if (int_dim1 != nullptr && int_dim1->value == 1) {
      output_shape.push_back(dim0);
    } else if (analyzer->CanProveEqual(dim0, dim1)) {
      output_shape.push_back(dim0);
    } else if (int_dim0 && int_dim1 && int_dim0->value != int_dim1->value) {
      result.status = BinaryBroadcastShapeInferResult::Status::kConflict;
      result.message = [&]() {
        std::ostringstream os;
        os << "the first input shape at dim " << x1_ndim - i << " is " << dim0
           << " and the second input shape at dim " << x2_ndim - i << " is " << dim1
           << ", which are not broadcastable.";
        return ffi::String(os.str());
      }();
      return result;
    } else {
      result.status = BinaryBroadcastShapeInferResult::Status::kUnknown;
      return result;
    }
  }
  auto& longer_shape = (x1_ndim > x2_ndim) ? x1_shape : x2_shape;
  for (; i <= max_ndim; ++i) {
    output_shape.push_back(longer_shape[max_ndim - i]);
  }
  result.status = BinaryBroadcastShapeInferResult::Status::kSuccess;
  result.shape = ffi::Array<PrimExpr>(output_shape.rbegin(), output_shape.rend());
  return result;
}

ffi::Optional<ffi::Array<PrimExpr>> InferBinaryBroadcastShape(
    const Call& call, const BlockBuilder& ctx, const ffi::Array<PrimExpr>& x1_shape,
    const ffi::Array<PrimExpr>& x2_shape) {
  auto infer_result = InferBinaryBroadcastShape(ctx->GetAnalyzer().get(), x1_shape, x2_shape);
  if (infer_result.status == BinaryBroadcastShapeInferResult::Status::kConflict) {
    TVM_FFI_ICHECK(infer_result.message.has_value());
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "In " << call->op << ", " << infer_result.message.value();
  } else if (infer_result.status == BinaryBroadcastShapeInferResult::Status::kSuccess) {
    TVM_FFI_ICHECK(infer_result.shape.has_value());
    return infer_result.shape.value();
  } else {
    // Unknown status, use simple fallback when shape mismatch.
    return std::nullopt;
  }
  TVM_FFI_UNREACHABLE();
}

std::vector<int> NormalizeAxes(const Call& call, int ndim, const ffi::Array<int64_t>& axes) {
  TVM_FFI_ICHECK_NE(ndim, kUnknownNDim) << "The ndim is required to be known for this function.";
  std::vector<bool> appeared_dims_set;
  std::vector<int> axes_non_neg;
  appeared_dims_set.resize(ndim, /*value=*/false);
  axes_non_neg.reserve(axes.size());
  for (int64_t axis : axes) {
    int _axis = static_cast<int>(axis);
    if (_axis < -ndim || _axis >= ndim) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "In " << call->op << ", the input axis " << _axis
          << " is out of range. The input tensor has " << ndim
          << " dimensions, so axis should be in range [" << -ndim << ", " << ndim << ").";
    } else if (_axis < 0) {
      _axis = ndim + _axis;
    }

    if (appeared_dims_set[_axis]) {
      TVM_FFI_VISIT_THROW(ValueError, call)
          << "In " << call->op
          << ", the input axes is required to be non-repetitive. However, there are "
             "multiple given axes referring to axis "
          << _axis;
    }
    appeared_dims_set[_axis] = true;
    axes_non_neg.push_back(_axis);
  }
  return axes_non_neg;
}

InferLayoutOutput InferLayoutUnaryEwise(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map) {
  TVM_FFI_ICHECK(NoDesiredLayout(call, desired_layouts));
  LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  return InferLayoutOutput({layout}, {layout}, call->attrs);
}

bool CanProveLayoutTransform(const ffi::Optional<SLayout>& input_layout,
                             const ffi::Optional<SLayout>& desired_layout,
                             ffi::Array<PrimExpr> shape) {
  if (!input_layout.has_value() || !desired_layout.has_value()) return false;
  bool can_prove = true;
  try {
    tirx::SBijectiveLayout todesired(input_layout.value(), desired_layout.value());
    ffi::Array<PrimExpr> desired_shape = todesired.ForwardShape(shape);
    ffi::Array<PrimExpr> back_shape = todesired.BackwardShape(desired_shape);
    sym::Analyzer analyzer;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (tvm::prim::IsConstInt(shape[i])) {
        if (!analyzer->CanProveEqual(shape[i], back_shape[i])) {
          can_prove = false;
          break;
        }
      }
    }
  } catch (std::exception& err) {
    return false;
  }
  return can_prove;
}

}  // namespace relax
}  // namespace tvm
