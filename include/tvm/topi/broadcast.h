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
 * \brief Broadcast op constructions
 * \file topi/broadcast.h
 */
#ifndef TVM_TOPI_BROADCAST_H_
#define TVM_TOPI_BROADCAST_H_

#include <tvm/tirx/op/math.h>
#include <tvm/topi/detail/broadcast.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <string>

namespace tvm {
namespace topi {
/*!
 * \brief Creates an operation that broadcasts a tensor into a compatible
 * shape according to numpy's rules
 *
 * \param t The input tensor
 * \param output_shape The target output shape, must be compatible
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is a broadcast operation
 */
inline tvm::te::Tensor broadcast_to(const tvm::te::Tensor& t,
                                    const tvm::ffi::Array<tvm::PrimExpr>& output_shape,
                                    std::string name = "T_broadcast_to",
                                    std::string tag = kBroadcast) {
  TVM_FFI_ICHECK_GE(output_shape.size(), t->shape.size())
      << "Not a broadcast, output dimensionality smaller than input.\noutput: " << output_shape
      << "\nvs\ninput: " << t;
  auto bh = detail::BroadcastShape(output_shape, t->shape);
  TVM_FFI_ICHECK_EQ(output_shape.size(), bh.common_shape.size());
  ffi::Array<PrimExpr> oshape;
  for (size_t i = 0; i < output_shape.size(); ++i) {
    if (output_shape[i].as<IntImmObj>() == nullptr) {
      oshape.push_back(output_shape[i]);
    } else {
      TVM_FFI_ICHECK(topi::detail::EqualCheck(output_shape[i], bh.common_shape[i]));
      oshape.push_back(bh.common_shape[i]);
    }
  }
  auto l = [&](tvm::ffi::Array<tvm::PrimVar> ovars) {
    return t(detail::InputIndexFromBroadcast(ovars, t, bh.vars2, bh.all_vars));
  };
  return tvm::te::compute(oshape, l, name, tag);
}

#define TOPI_DEFINE_BCAST_OP(Name, ComputeRule)                                                 \
  inline tvm::PrimExpr Name(const tvm::PrimExpr& a, const tvm::PrimExpr& b) { ComputeRule; }    \
  inline tvm::te::Tensor Name(const tvm::te::Tensor& A, const tvm::te::Tensor& B,               \
                              std::string name = "T_" #Name, std::string tag = kBroadcast) {    \
    auto l = [](tvm::PrimExpr a, tvm::PrimExpr b) { ComputeRule; };                             \
    return detail::WithBroadcast(l, A, B, name, tag);                                           \
  }                                                                                             \
  inline tvm::te::Tensor Name(const tvm::te::Tensor& A, const tvm::PrimExpr& B,                 \
                              std::string name = "T_" #Name, std::string tag = kElementWise) {  \
    auto l = [](tvm::PrimExpr a, tvm::PrimExpr b) { ComputeRule; };                             \
    return tvm::te::compute(                                                                    \
        A->shape, [&](const ::tvm::ffi::Array<::tvm::PrimVar>& i) { return l(A(i), B); }, name, \
        tag);                                                                                   \
  }                                                                                             \
  inline tvm::te::Tensor Name(const tvm::PrimExpr& A, const tvm::te::Tensor& B,                 \
                              std::string name = "T_" #Name, std::string tag = kElementWise) {  \
    auto l = [&](tvm::PrimExpr a, tvm::PrimExpr b) { ComputeRule; };                            \
    return tvm::te::compute(                                                                    \
        B->shape, [&](const ::tvm::ffi::Array<::tvm::PrimVar>& i) { return l(A, B(i)); }, name, \
        tag);                                                                                   \
  }

#define TOPI_DEFINE_OP_OVERLOAD(Name, OpName)                                       \
  inline tvm::te::Tensor Name(const tvm::te::Tensor& A, const tvm::te::Tensor& B) { \
    return topi::OpName(A, B);                                                      \
  }                                                                                 \
  inline tvm::te::Tensor Name(const tvm::PrimExpr& A, const tvm::te::Tensor& B) {   \
    return topi::OpName(A, B);                                                      \
  }                                                                                 \
  inline tvm::te::Tensor Name(const tvm::te::Tensor& A, const tvm::PrimExpr& B) {   \
    return topi::OpName(A, B);                                                      \
  }

/*!
 * \fn logical_and
 * \brief Compute A && B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(logical_and, { return a && b; });
TOPI_DEFINE_OP_OVERLOAD(operator&&, logical_and);

/*!
 * \fn logical_or
 * \brief Compute A || B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(logical_or, { return a || b; });
TOPI_DEFINE_OP_OVERLOAD(operator||, logical_or);

/*!
 * \fn logical_xor
 * \brief Compute A ^ B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(logical_xor, { return a ^ b; });

/*!
 * \fn bitwise_and
 * \brief Compute A & B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(bitwise_and, { return a & b; });
TOPI_DEFINE_OP_OVERLOAD(operator&, bitwise_and);

/*!
 * \fn bitwise_or
 * \brief Compute A | B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(bitwise_or, { return a | b; });
TOPI_DEFINE_OP_OVERLOAD(operator|, bitwise_or);

/*!
 * \fn bitwise_xor
 * \brief Compute A ^ B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(bitwise_xor, { return a ^ b; });
TOPI_DEFINE_OP_OVERLOAD(operator^, bitwise_xor);

/*!
 * \fn add
 * \brief Compute A + B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(add, { return a + b; });
TOPI_DEFINE_OP_OVERLOAD(operator+, add);

/*!
 * \fn subtract
 * \brief Compute A - B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(subtract, { return a - b; });
TOPI_DEFINE_OP_OVERLOAD(operator-, subtract);

/*!
 * \fn multiply
 * \brief Compute A * B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(multiply, { return a * b; });
TOPI_DEFINE_OP_OVERLOAD(operator*, multiply);

/*!
 * \fn divide
 * \brief Compute A / B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(divide, { return div(a, b); });

/*!
 * \fn floor divide
 * \brief Compute prim::floor(A / B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(floor_divide, {
  PrimType a_ty = a.ty();
  if (a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return floordiv(a, b);
  } else {
    return prim::floor(div(a, b));
  }
});

/*!
 * \fn log_add_exp
 * \brief Compute log(exp(A) + exp(B)) with auto-broadcasting.
 *
 * This operation is useful for numerically stable log-sum-exp computations,
 * which frequently appear in probabilistic and statistical models.
 *
 * \param A The first input tensor, or Expr.
 * \param B The second input tensor, or Expr.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return The computed log-sum-exp result.
 */
TOPI_DEFINE_BCAST_OP(log_add_exp, { return tirx::logaddexp(a, b); });

/*!
 * \fn trunc divide
 * \brief Compute prim::trunc(A / B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(trunc_divide, {
  PrimType a_ty = a.ty();
  if (a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return truncdiv(a, b);
  } else {
    return prim::trunc(div(a, b));
  }
});

/*!
 * \fn mod
 * \brief Compute A % B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(mod, { return truncmod(a, b); });

/*!
 * \fn floor mod
 * \brief Compute A - floor_div(A, B) * B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(floor_mod, {
  PrimType a_ty = a.ty();
  if (a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return floormod(a, b);
  } else {
    return a - floor_divide(a, b) * b;
  }
});

/*!
 * \fn trunc mod
 * \brief Compute A - trunc_div(A, B) * B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(trunc_mod, {
  PrimType a_ty = a.ty();
  if (a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return truncmod(a, b);
  } else {
    return a - trunc_divide(a, b) * b;
  }
});

/*!
 * \fn maximum
 * \brief Compute maximum(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(maximum, { return tvm::max(a, b); });

/*!
 * \fn minimum
 * \brief Compute minimum(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(minimum, { return tvm::min(a, b); });

/*!
 * \fn power
 * \brief Compute power(A, B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(power, { return tvm::prim::pow(a, b); });

/*!
 * \fn atan2
 * \brief Compute atan2(y, x) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr (y-coordinates).
 * \param B The second tensor, or Expr (x-coordinates).
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(atan2, { return tvm::prim::atan2(a, b); });

/*!
 * \fn left_shift
 * \brief Compute A << B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(left_shift, { return a << b; });
TOPI_DEFINE_OP_OVERLOAD(operator<<, left_shift);

/*!
 * \fn right_shift
 * \brief Compute A >> B with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(right_shift, { return a >> b; });
TOPI_DEFINE_OP_OVERLOAD(operator>>, right_shift);

/*!
 * \fn greater
 * \brief Compute (A > B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(greater, { return (a > b); });

/*!
 * \fn less
 * \brief Compute (A < B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(less, { return (a < b); });

/*!
 * \fn equal
 * \brief Compute (A == B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(equal, { return (a == b); });

/*!
 * \fn not_equal
 * \brief Compute (A != B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(not_equal, { return (a != b); });

/*!
 * \fn greater_equal
 * \brief Compute (A >= B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(greater_equal, { return (a >= b); });

/*!
 * \fn less_equal
 * \brief Compute (A <= B) with auto-broadcasting.
 *
 * \param A The first tensor, or Expr
 * \param B The second tensor, or Expr
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return The result.
 */
TOPI_DEFINE_BCAST_OP(less_equal, { return (a <= b); });

}  // namespace topi
}  // namespace tvm

#endif  // TVM_TOPI_BROADCAST_H_
