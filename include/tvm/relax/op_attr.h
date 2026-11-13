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
 * \file tvm/relax/op_attr.h
 * \brief Attribute keys and callback contracts for Relax operators.
 */
#ifndef TVM_RELAX_OP_ATTR_H_
#define TVM_RELAX_OP_ATTR_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/type.h>
#include <tvm/te/tensor.h>

namespace tvm {
namespace relax {

class LayoutDecision;
class InferLayoutOutput;

enum OpPatternKind {
  // Elementwise operation
  kElemWise = 0,
  // Broadcasting operator, can always map output axis to the input in order.
  // for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
  // Note that the axis need to be in order so transpose is not a bcast operator.
  kBroadcast = 1,
  // Injective operator, can always injectively map output axis to a single input axis.
  // All injective operator can still be safely fused to injective and reduction.
  kInjective = 2,
  // Communicative reduction operator.
  kCommReduce = 3,
  // Complex operation, can still fuse elemwise operations into its output.
  // but cannot chain another complex op
  kOutEWiseFusable = 4,
  // The pattern for tuple nodes. Can fuse into subsequent injective ops,
  // but treated specially
  kTuple = 7,
  // Opaque operation, cannot fuse anything.
  kOpaque = 8
};

namespace op_attr {
inline constexpr ffi::reflection::AttrKey<bool> kPurity{"FPurity"};

/*!
 * \brief Packed function implementation for operators. The relax operator will be lowered to
 * this packed function call during codegen.
 */
inline constexpr ffi::reflection::AttrKey<ffi::String> kCallPacked{"FCallPacked"};
/*!
 * \brief The function type of a normalization function.
 *
 * A normalization function is used when a `relax::Call` may be
 * expressed in multiple syntactically valid and semantically
 * equivalent forms, to normalize to a single representation.
 *
 * Note: `FNormalize` is applied for each expression as part of the
 *    `relax::BlockBuilder`. Operator-specific validation belongs in
 *    `OpDef::set_validator` and runs after normalization.
 *
 * \param bb The BlockBuilder context.
 *
 * \param call The call to be normalized.  It is provided by-value, to
 * avoid copies for the common case where the call is already normalized.
 */
inline constexpr ffi::reflection::AttrKey<
    ffi::TypedFunction<Expr(const BlockBuilder& bb, Call call)>>
    kNormalize{"FNormalize"};
/*! \brief The function type of a legalization function.
 *
 * A legalization function is used to replace a `relax::Call` with
 * more concrete implementations.  For example, the operation
 * `relax.op.add` may be replaced with a call to a TIR function
 * implementing addition of two tensors.
 *
 * The purpose of `FLegalize` is to remove calls to the operator while
 * lowering.  Therefore, unlike `FNormalize`, the resulting expression
 * may *not* contain the original operator.
 *
 * \param bb The BlockBuilder context.
 * \param call The call to be legalized.
 */
inline constexpr ffi::reflection::AttrKey<
    ffi::TypedFunction<Expr(const BlockBuilder& bb, const Call& call)>>
    kLegalize{"FLegalize"};
/*! \brief The function type of a function to lower the runtime builtin.
 *
 * A builtin function may be lowered to a lowered form in `LowerRuntimeBuiltin`.
 *
 * \param bb The BlockBuilder context.
 * \param call The call to be lowered.
 */
inline constexpr ffi::reflection::AttrKey<
    ffi::TypedFunction<Expr(const BlockBuilder& bb, const Call& call)>>
    kLowerBuiltin{"FLowerBuiltin"};
/*!
 * \brief Gradient for a specific op.
 *
 * \param orig_var the original var corresponding to orig_call.
 * \param orig_call the original Call(op) expr.
 * \param output_grad the gradient of the Expr.
 * \param ctx the current block builder context.
 * \return the gradient for each parameter.
 */
inline constexpr ffi::reflection::AttrKey<ffi::TypedFunction<tvm::ffi::Array<Expr>(
    const Var& orig_var, const Call& orig_call, const Var& output_grad, const BlockBuilder& ctx)>>
    kPrimalGradient{"FPrimalGradient"};
/*!
 * \brief Infer output type given the call
 *
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 */
inline constexpr ffi::reflection::AttrKey<
    ffi::TypedFunction<Type(const Call& call, const BlockBuilder& ctx)>>
    kInferTypeWithBuilder{"relax.FInferTypeWithBuilder"};
inline constexpr ffi::reflection::AttrKey<bool> kDataDependent{"FDataDependent"};
inline constexpr ffi::reflection::AttrKey<bool> kRequiresArgumentShapes{"RequiresArgumentShapes"};
inline constexpr ffi::reflection::AttrKey<bool> kAllocator{"TAllocator"};
inline constexpr ffi::reflection::AttrKey<
    ffi::TypedFunction<Call(const Call& call_node, DLDataType out_dtype)>>
    kInferMixedPrecision{"FInferMixedPrecision"};
/*! \brief the operator pattern */
inline constexpr ffi::reflection::AttrKey<int> kMixedPrecisionPolicy{"TMixedPrecisionPolicy"};
/*!
 * \brief SLayout conversion interface.
 * \param call The call node.
 * \param desired_layouts The desired layouts of the operator.
 * \param var_layout_map The layout of the variables.
 */
inline constexpr ffi::reflection::AttrKey<ffi::TypedFunction<InferLayoutOutput(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const ffi::Map<Var, NestedMsg<LayoutDecision>>& var_layout_map)>>
    kInferLayout{"FRelaxInferLayout"};
}  // namespace op_attr

// Compatibility names; new callers can deduce the value type from the key.
using FCallPacked = decltype(op_attr::kCallPacked)::ValueType;
using FInferTypeWithBuilder = decltype(op_attr::kInferTypeWithBuilder)::ValueType;
using FNormalize = decltype(op_attr::kNormalize)::ValueType;
using FLegalize = decltype(op_attr::kLegalize)::ValueType;
using FLowerBuiltin = decltype(op_attr::kLowerBuiltin)::ValueType;
using FPrimalGradient = decltype(op_attr::kPrimalGradient)::ValueType;

namespace distributed {
namespace op_attr {
inline constexpr ffi::reflection::AttrKey<FInferTypeWithBuilder> kInferTypeWithBuilder{
    "relax.dist.FInferTypeWithBuilder"};
}  // namespace op_attr
}  // namespace distributed

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_ATTR_H_
