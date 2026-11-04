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
 * \file op_common.h
 * \brief A set of utilities and common functionality
 * for Relax ops.
 */
#ifndef TVM_RELAX_OP_OP_COMMON_H_
#define TVM_RELAX_OP_OP_COMMON_H_

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/op_attr.h>
#include <tvm/s_tir/data_layout.h>
#include <tvm/sym/analyzer.h>

#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../transform/infer_amp_utils.h"
#include "../transform/infer_layout_utils.h"

namespace tvm {
namespace relax {
using namespace tvm::prim;

/************ Op input type getter ************/

/*!
 * \brief Check that the operator has
 *
 * Verify that the number of arguments matches the expected number for
 * the operator.
 *
 * \param call The context Call to the operator.
 *
 * \param ctx The error reporting context.
 */
void CheckNumArguments(const Call& call);
inline void CheckNumArguments(const Call& call, const BlockBuilder&) { CheckNumArguments(call); }

/*!
 * \brief Get the tensor type of the operator input.
 * \param call The context Call to the operator.
 * \param i_arg The index of the argument to check
 * \param ctx The error reporting context.
 * \return The tensor type of the argument
 */
TensorType GetInputTensorType(const Call& call, size_t i_arg);
inline TensorType GetInputTensorType(const Call& call, size_t i_arg, const BlockBuilder&) {
  return GetInputTensorType(call, i_arg);
}

/*!
 * \brief Get the tensor type of the operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor type of each input.
 * \note This function require every input to be Tensor. The number of call arguments is required
 * to match the number of inputs of the op being called.
 */
ffi::Array<TensorType> GetInputTensorType(const Call& call);
inline ffi::Array<TensorType> GetInputTensorType(const Call& call, const BlockBuilder&) {
  return GetInputTensorType(call);
}

/*!
 * \brief Get the tensor type of the unary operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor type of the unary operator input.
 * \throw Throw exception if the number of input is not one, or the type of the input is not
 * a tensor type.
 */
inline TensorType GetUnaryInputTensorType(const Call& call) { return GetInputTensorType(call)[0]; }

// Tensor-dependent context-free rules cannot infer unresolved nested expressions or
// distributed placements. The Relax builder normalizes those inputs first.
inline bool RequiresTensorInputNormalization(const Call& call) {
  CheckNumArguments(call);
  for (const Expr& arg : call->args) {
    if (arg->ty.as<MissingTypeObj>() || arg->ty.as<distributed::DTensorTypeObj>()) return true;
  }
  return false;
}
inline TensorType GetUnaryInputTensorType(const Call& call, const BlockBuilder&) {
  return GetUnaryInputTensorType(call);
}

/*!
 * \brief Get the tensor type of tuple input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param tup The input tuple.
 * \return The tensor types of tuple input.
 * \throw Throw exception if input expression is not a tuple.
 */
ffi::Array<TensorType> GetTensorTypeFromTuple(const Call& call, const Expr& tup);
inline ffi::Array<TensorType> GetTensorTypeFromTuple(const Call& call, const BlockBuilder&,
                                                     const Expr& tup) {
  return GetTensorTypeFromTuple(call, tup);
}

namespace detail {
/*! \brief Implementation helper for GetArgType */
template <typename ArgType>
ArgType GetArgTypeByIndex(const Call& call, const Op& op, size_t index) {
  if (call->args[index]->ty.as<MissingType>().has_value()) {
    TVM_FFI_VISIT_THROW(InternalError, call)
        << op << " op should have arguments with defined Type.  "
        << "However, args[" << index << "] has undefined type.";
  }

  auto ty = GetType(call->args[index]);
  auto typed_ty = ty.as<ArgType>();

  if (!typed_ty.has_value()) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << op << " requires that args[" << index << "] be a " << ArgType::ContainerType::_type_key
        << ", but was instead " << ty << " of type " << ty->GetTypeKey();
  }

  return typed_ty.value();
}

/*! \brief Implementation helper for GetArgType */
template <typename... ArgTypes, size_t... Indices>
std::tuple<ArgTypes...> GetArgTypeHelper(const Call& call, const Op& op,
                                         std::index_sequence<Indices...>) {
  return std::tuple<ArgTypes...>{GetArgTypeByIndex<ArgTypes>(call, op, Indices)...};
}
}  // namespace detail

/*!
 * \brief Get all argument types as expected types.
 *
 * \tparam ArgTypes The expected types of arguments, in the order they appear.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The argument types.
 * \throw Throw exception if input expression is not a tuple.
 */
template <typename... ArgTypes>
std::tuple<ArgTypes...> GetArgType(const Call& call) {
  Op op = call->op.as_or_throw<Op>();
  size_t n_input = op->args_info.size();

  // Unfortunately, because signature registrations in OpDef
  // occur during initialization of globals and are
  // not available at compile-time, this cannot be a static_assert.
  TVM_FFI_ICHECK(op->var_args_info.has_value() ? sizeof...(ArgTypes) >= n_input
                                               : sizeof...(ArgTypes) == n_input)
      << "Internal error: " << op << " op defines " << n_input << " arguments in its OpDef() call, "
      << "but GetArgType was given " << sizeof...(ArgTypes) << " template arguments.";

  CheckNumArguments(call);
  TVM_FFI_ICHECK_GE(call->args.size(), sizeof...(ArgTypes));
  return detail::GetArgTypeHelper<ArgTypes...>(call, op,
                                               std::make_index_sequence<sizeof...(ArgTypes)>());
}

template <typename... ArgTypes>
std::tuple<ArgTypes...> GetArgType(const Call& call, const BlockBuilder&) {
  return GetArgType<ArgTypes...>(call);
}

/************ Op registration macro ************/

/*!
 * \brief Infer the type for unary elementwise ops.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param f_compute_out_dtype The function to compute the output dtype, with
 * signature PrimType f_compute_out_dtype(const TensorType& input_ty).
 * \tparam require_float_dtype whether this op requires the input dtype to be float
 * \tparam Ftype the type of f_compute_out_dtype
 * \return The inferred type.
 */
template <bool require_float_dtype, typename FType>
inline Type InferTypeUnary(const Call& call, FType f_compute_out_dtype) {
  if (RequiresTensorInputNormalization(call)) return Type::Missing();
  TensorType input_ty = GetUnaryInputTensorType(call);
  if (require_float_dtype && !input_ty->IsUnknownDtype() &&
      !input_ty->dtype.value().MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << call->op
        << " requires the input tensor to have float dtype. However, the given input dtype is "
        << input_ty->dtype.value();
  }
  ffi::Optional<PrimType> computed_dtype = f_compute_out_dtype(input_ty);
  bool same_dtype = computed_dtype.has_value() == input_ty->dtype.has_value() &&
                    (!computed_dtype.has_value() ||
                     computed_dtype.value()->dtype == input_ty->dtype.value()->dtype);
  TensorType inferred = input_ty;
  if (!call->ty_args.empty() || !same_dtype) {
    auto output_ty = ffi::make_object<TensorTypeObj>(*input_ty.get());
    output_ty->dtype = computed_dtype;
    if (call->ty_args.size() > 0) {
      auto defined_ty = call->ty_args[0].as<TensorTypeObj>();
      TVM_FFI_ICHECK(defined_ty);
      auto shape = output_ty->GetShape();
      TVM_FFI_ICHECK(shape.has_value());
      TVM_FFI_ICHECK(defined_ty->vdevice.has_value());
      inferred =
          TensorType(ShapeExpr(shape.value()), output_ty->dtype, defined_ty->vdevice.value());
    } else {
      inferred = TensorType(output_ty);
    }
  }
  if (const auto* old = call->ty.as<TensorTypeObj>()) {
    auto same_optional_ref = [](const auto& lhs, const auto& rhs) {
      return lhs.has_value() == rhs.has_value() &&
             (!lhs.has_value() || lhs.value().same_as(rhs.value()));
    };
    bool same_output_dtype =
        old->dtype.has_value() == inferred->dtype.has_value() &&
        (!old->dtype.has_value() || old->dtype.value()->dtype == inferred->dtype.value()->dtype);
    if (old->ndim == inferred->ndim && same_output_dtype &&
        same_optional_ref(old->shape, inferred->shape) &&
        same_optional_ref(old->vdevice, inferred->vdevice)) {
      return call->ty;
    }
  }
  return inferred;
}

template <bool require_float_dtype, typename FType>
inline Type InferTypeUnary(const Call& call, const BlockBuilder&, FType f_compute_out_dtype) {
  return InferTypeUnary<require_float_dtype>(call, f_compute_out_dtype);
}

/*!
 * \brief Derive the output type of a call_tir from the callee signature and the
 * argument types, or nullopt when it cannot be derived.
 */
ffi::Optional<Type> InferCallTIROutputTypeFromArguments(
    Type func_ty, Type arg_ty, ffi::Optional<ffi::Array<int64_t>> opt_inplace_indices);

/*!
 * \brief Infer the type by returning the type of the input argument.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \tparam arg_index The index of the argument to infer the output dtype from.
 * \return The inferred type.
 */
template <int arg_index>
Type ReturnTypeFromArg(const Call& call, const BlockBuilder& ctx) {
  Op op = call->op.as_or_throw<Op>();
  CheckNumArguments(call, ctx);
  int n_input = call->args.size();
  if (arg_index >= n_input) {
    TVM_FFI_VISIT_THROW(IndexError, call)
        << op << " op has only " << n_input << "arguments, but try to get the arg with index "
        << arg_index;
  }
  return GetType(call->args[arg_index]);
}

template <int arg_index>
Type ReturnTypeFromArgContextFree(const CallObj* call_node) {
  Call call = ffi::GetRef<Call>(call_node);
  Op op = call->op.as_or_throw<Op>();
  CheckNumArguments(call);
  int n_input = call->args.size();
  if (arg_index >= n_input) {
    TVM_FFI_VISIT_THROW(IndexError, call)
        << op << " op has only " << n_input << "arguments, but try to get the arg with index "
        << arg_index;
  }
  return call->args[arg_index]->ty;
}

/*!
 * \brief Infer the type for unary arithmetic elementwise ops. It's also
 * used in some NN operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \tparam require_float_dtype whether this op requires the input dtype to be float
 * \return The inferred type.
 */
template <bool require_float_dtype>
Type InferTypeUnaryArith(const Call& call, const BlockBuilder& ctx) {
  return InferTypeUnary<require_float_dtype>(
      call, ctx, [](const TensorType& input_ty) { return input_ty->dtype; });
}

template <bool require_float_dtype>
Type InferTypeUnaryArithContextFree(const CallObj* call_node) {
  Call call = ffi::GetRef<Call>(call_node);
  return InferTypeUnary<require_float_dtype>(
      call, [](const TensorType& input_ty) { return input_ty->dtype; });
}

/*!
 * \brief SLayout infer util for unary elementwise ops. It will simply take the layout of the input.
 * \param call The context Call to the operator.
 * \param desired_layouts The desired layouts of certain ops.
 * \param var_layout_map The layout of vars.
 * \return The inferred layout result.
 */
InferLayoutOutput InferLayoutUnaryEwise(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map);

/*!
 * \brief Get the element dtype from Type
 *
 * \param ty The Type to expect
 * \return The inferred element dtype.
 * \throw Throw exception if the Type doesn't have an element type.
 */
inline std::optional<PrimType> GetElementDType(const Type& ty) {
  if (const auto* prim = ty.as<PrimTypeObj>()) {
    return ffi::GetRef<PrimType>(prim);
  } else if (const auto* tensor = ty.as<TensorTypeObj>()) {
    if (tensor->dtype.has_value()) {
      return tensor->dtype.value();
    } else {
      return std::nullopt;
    }
  } else {
    return std::nullopt;
    TVM_FFI_THROW(TypeError) << "Only PrimType and TensorType "
                             << "have an associated data type.  "
                             << "Cannot determine element type of " << ty;
  }
}

/*!
 * \brief Infer the output datatype for binary arithmetic operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param lhs_ty The type of the first operand
 * \param rhs_ty The type of the second operand
 * \return The inferred output dtype.
 * \throw Throw exception if the dtype of two input TensorType don’t match
 */
inline ffi::Optional<PrimType> InferBinaryArithOpOutDtype(const Call& call, const Type& lhs_ty,
                                                          const Type& rhs_ty) {
  // Formatting the full Call in an inference error would invoke the printer
  // and reenter the same inference hook before the exception is thrown.
  auto opt_lhs_dtype = GetElementDType(lhs_ty);
  if (!opt_lhs_dtype) {
    if (const auto* lhs_tensor = lhs_ty.as<TensorTypeObj>()) {
      if (lhs_tensor->IsUnknownDtype()) return std::nullopt;
    }
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Binary operators must have the same datatype for both operands.  "
        << "However, " << call->op << " has LHS type " << lhs_ty << ".   This is of type "
        << lhs_ty->GetTypeKey() << ", which does not have a datatype.";
  }
  auto lhs_dtype = opt_lhs_dtype.value();

  auto opt_rhs_dtype = GetElementDType(rhs_ty);
  if (!opt_rhs_dtype) {
    if (const auto* rhs_tensor = rhs_ty.as<TensorTypeObj>()) {
      if (rhs_tensor->IsUnknownDtype()) return std::nullopt;
    }
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Binary operators must have the same datatype for both operands.  "
        << "However, " << call->op << " has RHS type " << rhs_ty << ".   This is of type "
        << rhs_ty->GetTypeKey() << ", which does not have a datatype.";
  }
  auto rhs_dtype = opt_rhs_dtype.value();

  if (lhs_dtype != rhs_dtype && !lhs_dtype.MatchesCode(DLDataTypeCode::kDLBool) &&
      !rhs_dtype.MatchesCode(DLDataTypeCode::kDLBool)) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Binary operators must have the same datatype for both operands.  "
        << "However, " << call->op << " uses datatype " << lhs_dtype << " on the LHS (Type of "
        << lhs_ty << "), and datatype " << rhs_dtype << " on the RHS (Type of " << rhs_ty << ").";
  }
  return lhs_dtype;
}

inline ffi::Optional<PrimType> InferBinaryArithOpOutDtype(const Call& call, const BlockBuilder&,
                                                          const Type& lhs_ty, const Type& rhs_ty) {
  return InferBinaryArithOpOutDtype(call, lhs_ty, rhs_ty);
}

/*!
 * \brief Infer the output virtual device for binary arithmetic operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param lhs_ty The type of the first operand
 * \param rhs_ty The type of the second operand
 * \return The inferred output vdevice.
 * \throw Throw exception if the vdevice of two input TensorType don’t match
 */
inline ffi::Optional<VDevice> InferBinaryArithOpOutVDevice(const Call& call, const Type& lhs_ty,
                                                           const Type& rhs_ty) {
  auto get_vdevice = [&](const Type& ty) -> ffi::Optional<VDevice> {
    if (const auto* tensor = ty.as<TensorTypeObj>()) {
      return tensor->vdevice;
    } else {
      return std::nullopt;
    }
  };

  /*
   * This is the case where the output VDevice defined by a customization pass.
   * Like targets that supports mixed VDevices (like differed by memory_scope for Adreno)
   * and have specialized derivation for output VDevice.
   */
  if (call->ty_args.size() > 0) {
    return get_vdevice(call->ty_args[0]);
  }

  auto lhs_vdevice = get_vdevice(lhs_ty);
  auto rhs_vdevice = get_vdevice(rhs_ty);

  if (!lhs_vdevice.has_value() || !lhs_vdevice.value()->target.has_value()) {
    return rhs_vdevice;
  }
  if (!rhs_vdevice.has_value() || !rhs_vdevice.value()->target.has_value()) {
    return lhs_vdevice;
  }

  if (lhs_vdevice.value() != rhs_vdevice.value()) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Binary operators with Tensor arguments "
                                          << "must have the same VDevice for both operands.  "
                                          << "However, " << call << " has a LHS on VDevice "
                                          << lhs_vdevice << " and a RHS on VDevice " << rhs_vdevice;
  }
  return lhs_vdevice;
}

inline ffi::Optional<VDevice> InferBinaryArithOpOutVDevice(const Call& call, const BlockBuilder&,
                                                           const Type& lhs_ty, const Type& rhs_ty) {
  return InferBinaryArithOpOutVDevice(call, lhs_ty, rhs_ty);
}

/*! \brief Result of binary broadcast shape inference without diagnostic context. */
struct BinaryBroadcastShapeInferResult {
  enum class Status {
    /*! \brief Broadcast output shape is known. */
    kSuccess,
    /*! \brief Shapes may be broadcastable but cannot be proved symbolically. */
    kUnknown,
    /*! \brief Concrete shapes are not broadcastable. */
    kConflict,
  };

  /*! \brief Inference status. */
  Status status = Status::kUnknown;
  /*! \brief Broadcasted shape if status is kSuccess. */
  ffi::Optional<ffi::Array<PrimExpr>> shape;
  /*! \brief Human-readable conflict description if status is kConflict. */
  ffi::Optional<ffi::String> message;
};

/*!
 * \brief Infer the output shape for binary broadcast operators.
 * \param analyzer The arithmetic analyzer used to prove shape equality.
 * \param x1_shape The shape of the first operand.
 * \param x2_shape The shape of the second operand.
 * \return Inference status and broadcasted shape, or a conflict message.
 */
BinaryBroadcastShapeInferResult InferBinaryBroadcastShape(sym::AnalyzerObj* analyzer,
                                                          const ffi::Array<PrimExpr>& x1_shape,
                                                          const ffi::Array<PrimExpr>& x2_shape);

/*!
 * \brief Infer the output shape for binary broadcast operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param x1_shape The shape of the first operand.
 * \param x2_shape The shape of the second operand.
 * \return The inferred output shape after broadcasting. Or `std::nullopt` if the output shape
 * cannot be determined due to symbolic broadcast.
 */
ffi::Optional<ffi::Array<PrimExpr>> InferBinaryBroadcastShape(const Call& call,
                                                              const BlockBuilder& ctx,
                                                              const ffi::Array<PrimExpr>& x1_shape,
                                                              const ffi::Array<PrimExpr>& x2_shape);

/*!
 * \brief Convert all axes to non-negative indices, and meanwhile check if the given array of axes
 * are all in range and non-repetitive with regards to the given ndim.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param ndim The ndim constraint, which is required to be known already.
 * \param axes The axis indices to be checked
 * \return The input axes in non-negative indexing.
 * \throw Throw exception if there exists out-of-range axis index or repetitive indices.
 */
std::vector<int> NormalizeAxes(const Call& call, int ndim, const ffi::Array<int64_t>& axes);
inline std::vector<int> NormalizeAxes(const Call& call, const BlockBuilder&, int ndim,
                                      const ffi::Array<int64_t>& axes) {
  return NormalizeAxes(call, ndim, axes);
}

/*!
 * \brief Convert the given axis to non-negative index. Meanwhile check if the axis is in range
 * with regards to the given ndim.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param ndim The ndim constraint.
 * \param axis The axis index to be checked
 * \return The input axis in non-negative indexing.
 * \throw Throw exception the given axis is out-of-range.
 */
inline int NormalizeAxis(const Call& call, int ndim, int axis) {
  return NormalizeAxes(call, ndim, {axis})[0];
}
inline int NormalizeAxis(const Call& call, const BlockBuilder&, int ndim, int axis) {
  return NormalizeAxis(call, ndim, axis);
}

/*!
 * \brief Compute the product of all the given shape values.
 * \param shape_values The given shape values.
 * \return The product of all the given shape values.
 */
PrimExpr ComputeShapeProduct(const ffi::Array<PrimExpr>& shape_values);

/*!
 * \brief Check if the given permutation is identity permutation.
 * \param permutation The given permutation.
 * \return Whether the given permutation is identity permutation.
 */
bool IsIdentityPermutation(const std::vector<int>& permutation);

/*!
 * \brief Convert an array of integers to int64 dtype.
 * \param int_imms The input IntImms to be converted.
 * \return The conversion result, where every IntImm has dtype int64
 */
inline ffi::Array<IntImm> ConvertIntImmToInt64(const ffi::Array<IntImm>& int_imms) {
  return int_imms.Map(
      [](const IntImm& i) { return cast(i, PrimType::Int(64)).as_or_throw<IntImm>(); });
}

/************ Utilities for NN operators ************/

/*!
 * \brief Complete the padding to a 2-length array.
 * - If the padding length is 1, the same padding is used on all left/right sides
 * - If the padding length is 2, padding is in the order of (left, right)
 * \param padding The given padding to be completed
 * \return The completed padding.
 * \throws Throws error if the input padding length is neither 1 or 2.
 */
inline ffi::Array<int64_t> GetCompletePadding1D(ffi::Array<int64_t> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0]};
  } else if (padding.size() == 2) {
    return padding;
  }
  TVM_FFI_THROW(InternalError)
      << "The input padding length is expected to be either 1 or 2. However, the given "
         "padding is "
      << padding;
  throw;
}

/*!
 * \brief Complete the padding to a 4-length array.
 * - If the padding length is 1, the same padding is used on all top/left/bottom/right sides
 * - If the padding length is 2, top/bottom sides use padding[0] and left/right use padding[1]
 * - If the padding length is 4, padding is in the order of (top, left, bottom, right)
 * \param padding The given padding to be completed
 * \return The completed padding.
 * \throws Throws error if the input padding length is neither 1, 2 or 4.
 */
inline ffi::Array<int64_t> GetCompletePadding2D(ffi::Array<int64_t> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0], padding[0], padding[0]};
  } else if (padding.size() == 2) {
    return {padding[0], padding[1], padding[0], padding[1]};
  } else if (padding.size() == 4) {
    return padding;
  }
  TVM_FFI_THROW(InternalError)
      << "The input padding length is expected to be either 1, 2 or 4. However, the given "
         "padding is "
      << padding;
  throw;
}

/*!
 * \brief Complete the padding to a 6-length array.
 * - If the padding length is 1, the same padding is used on all front/top/left/back/bottom/right
 * sides
 * - If the padding length is 3, front/back sides use padding[0], top/bottom sides use padding[1]
 * and left/right use padding[2]
 * - If the padding length is 6, padding is in the order of (front, top, left, back, bottom, right)
 * \param padding The given padding to be completed
 * \return The completed padding.
 * \throws Throws error if the input padding length is neither 1, 3 or 6.
 */
inline ffi::Array<int64_t> GetCompletePadding3D(ffi::Array<int64_t> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0], padding[0], padding[0], padding[0], padding[0]};
  } else if (padding.size() == 3) {
    return {padding[0], padding[1], padding[2], padding[0], padding[1], padding[2]};
  } else if (padding.size() == 6) {
    return padding;
  }
  TVM_FFI_THROW(InternalError)
      << "The input padding length is expected to be either 1, 3 or 6. However, the given "
         "padding is "
      << padding;
  throw;
}

/*!
 * \brief Check if the given tensor layout can be converted to the given target layout.
 * If convertible, return the tensor layout and the bijective conversion in tirx::SLayout and
 * tirx::SBijectiveLayout accordingly.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param tensor_layout The tensor layout to be checked
 * \param tgt_layout The target layout to be matched
 * \param tensor_name The name of the input tensor
 * \return The tensor layout and the bijective conversion in tirx::SLayout and
 * tirx::SBijectiveLayout accordingly.
 */
inline std::pair<tirx::SLayout, tirx::SBijectiveLayout> CheckTensorLayout(
    const Call& call, const ffi::String& tensor_layout, const ffi::String& tgt_layout,
    const ffi::String& tensor_name) {
  tvm::PrimType i64_ty = tvm::PrimType::Int(64);
  tirx::SLayout _tensor_layout(tensor_layout, i64_ty);
  auto tensor2tgt =
      tirx::SBijectiveLayout::Create(_tensor_layout, tirx::SLayout(tgt_layout, i64_ty));
  if (!tensor2tgt.has_value()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << call->op << " requires the given " << tensor_name << " layout to be convertible from "
        << tgt_layout << " layout. However, the given layout " << tensor_layout
        << " is not convertible.";
  }
  return {_tensor_layout, tensor2tgt.value()};
}

inline std::pair<tirx::SLayout, tirx::SBijectiveLayout> CheckTensorLayout(
    const Call& call, const BlockBuilder&, const ffi::String& tensor_layout,
    const ffi::String& tgt_layout, const ffi::String& tensor_name) {
  return CheckTensorLayout(call, tensor_layout, tgt_layout, tensor_name);
}

/*!
 * \brief Check if the given tensor type has expected ndim per the given layout (or the ndim
 * is unknown), and try to cast the shape to ShapeExpr.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param ty The input tensor type to be checked.
 * \param layout The layout that the given tensor is expected to have.
 * \return The shape of the input tensor in ShapeExpr, or `std::nullopt` if the shape is unknown.
 */
inline ffi::Optional<ShapeExpr> CheckNdimPerLayoutAndGetShape(const Call& call,
                                                              const TensorType& ty,
                                                              const tirx::SLayout& layout) {
  if (!ty->IsUnknownNdim() && ty->ndim != static_cast<int>(layout.ndim())) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "In " << call->op << ", layout " << layout << " requires the input to be "
        << layout.ndim() << "-dim tensor. However, the given input has ndim " << ty->ndim;
  }
  if (const auto* shape_expr = ty->shape.as<ShapeExprObj>()) {
    return ffi::GetRef<ShapeExpr>(shape_expr);
  }
  return std::nullopt;
}

inline ffi::Optional<ShapeExpr> CheckNdimPerLayoutAndGetShape(const Call& call, const BlockBuilder&,
                                                              const TensorType& ty,
                                                              const tirx::SLayout& layout) {
  return CheckNdimPerLayoutAndGetShape(call, ty, layout);
}

Expr MakeVMAllocStorage(Expr size, PrimExpr runtime_device_index, DataTypeImm dtype,
                        StringImm storage_scope = StringImm("global"));
Expr MakeVMAllocTensor(Expr storage, PrimExpr offset, Expr shape, DataTypeImm dtype,
                       PrimExpr runtime_device_index);

Expr MakeAllocTensor(Expr shape, DataTypeImm dtype, PrimExpr runtime_device_index,
                     StringImm storage_scope = StringImm("global"));

/**
 * \brief Return the argument of the call.
 *        Note: If this is a call_tir, return the arguments passed to the TIR func
 *
 * \param call The call node
 * \return The arguments of the call
 */
ffi::Array<Expr> GetCallArgs(const Call& call);

/**
 * \brief Checks the given shape can be proved from the source layout to dst layout
 * \param input_layout is the layout of given shape
 * \param desired_layout is the target layout the shape to be transformed
 * \param shape array
 * \return true or false depending on the compatibility
 */
bool CanProveLayoutTransform(const ffi::Optional<SLayout>& input_layout,
                             const ffi::Optional<SLayout>& desired_layout,
                             ffi::Array<PrimExpr> shape);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_OP_COMMON_H_
