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
 * \file elemwise.h
 * \brief Elementwise op constructions
 */
#ifndef TVM_TOPI_ELEMWISE_H_
#define TVM_TOPI_ELEMWISE_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/math.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <string>

#include "broadcast.h"

namespace tvm {
namespace topi {
using namespace tvm::te;

// Unary intrinsic operators
#define TOPI_DECLARE_UNARY_OP(OpName, ExprFunc)                                             \
  inline Tensor OpName(const Tensor& x, std::string name = "T_" #OpName,                    \
                       std::string tag = kElementWise) {                                    \
    return compute(                                                                         \
        x->shape, [&](const ffi::Array<PrimVar>& i) { return ExprFunc(x(i)); }, name, tag); \
  }

TOPI_DECLARE_UNARY_OP(exp, ::tvm::prim::exp);
TOPI_DECLARE_UNARY_OP(erf, ::tvm::prim::erf);
TOPI_DECLARE_UNARY_OP(sigmoid, ::tvm::prim::sigmoid);
TOPI_DECLARE_UNARY_OP(sqrt, ::tvm::prim::sqrt);
TOPI_DECLARE_UNARY_OP(log, ::tvm::prim::log);
TOPI_DECLARE_UNARY_OP(log2, ::tvm::prim::log2);
TOPI_DECLARE_UNARY_OP(log10, ::tvm::prim::log10);
TOPI_DECLARE_UNARY_OP(floor, ::tvm::prim::floor);
TOPI_DECLARE_UNARY_OP(ceil, ::tvm::prim::ceil);
TOPI_DECLARE_UNARY_OP(round, ::tvm::prim::round);
TOPI_DECLARE_UNARY_OP(trunc, ::tvm::prim::trunc);
TOPI_DECLARE_UNARY_OP(abs, ::tvm::prim::abs);
TOPI_DECLARE_UNARY_OP(cos, ::tvm::prim::cos);
TOPI_DECLARE_UNARY_OP(cosh, ::tvm::prim::cosh);
TOPI_DECLARE_UNARY_OP(tan, ::tvm::prim::tan);
TOPI_DECLARE_UNARY_OP(sin, ::tvm::prim::sin);
TOPI_DECLARE_UNARY_OP(sinh, ::tvm::prim::sinh);
TOPI_DECLARE_UNARY_OP(acos, ::tvm::prim::acos);
TOPI_DECLARE_UNARY_OP(acosh, ::tvm::prim::acosh);
TOPI_DECLARE_UNARY_OP(asin, ::tvm::prim::asin);
TOPI_DECLARE_UNARY_OP(asinh, ::tvm::prim::asinh);
TOPI_DECLARE_UNARY_OP(atan, ::tvm::prim::atan);
TOPI_DECLARE_UNARY_OP(atanh, ::tvm::prim::atanh);
TOPI_DECLARE_UNARY_OP(isnan, ::tvm::prim::isnan);
TOPI_DECLARE_UNARY_OP(tanh, ::tvm::prim::tanh);
TOPI_DECLARE_UNARY_OP(isfinite, ::tvm::prim::isfinite);
TOPI_DECLARE_UNARY_OP(isinf, ::tvm::prim::isinf);

/*!
 * \brief Fast_tanh_float implementation from Eigen
 * https://github.com/eigenteam/eigen-git-mirror/blob/master/Eigen/src/Core/MathFunctionsImpl.h#L26
 */
inline Tensor fast_tanh_float(const Tensor& in, std::string name, std::string tag) {
  using namespace tvm::prim;
  // Clamp the inputs to the range [-9, 9] since anything outside
  // this range is +/-1.0f in single-precision.
  PrimType input_type = in->GetDataType();
  auto x = maximum(MakeConst(input_type, -9.0), minimum(MakeConst(input_type, 9.0), in));

  // The monomial coefficients of the numerator polynomial (odd).
  auto alpha_1 = MakeConst(input_type, 4.89352455891786e-03);
  auto alpha_3 = MakeConst(input_type, 6.37261928875436e-04);
  auto alpha_5 = MakeConst(input_type, 1.48572235717979e-05);
  auto alpha_7 = MakeConst(input_type, 5.12229709037114e-08);
  auto alpha_9 = MakeConst(input_type, -8.60467152213735e-11);
  auto alpha_11 = MakeConst(input_type, 2.00018790482477e-13);
  auto alpha_13 = MakeConst(input_type, -2.76076847742355e-16);

  // The monomial coefficients of the denominator polynomial (even).
  auto beta_0 = MakeConst(input_type, 4.89352518554385e-03);
  auto beta_2 = MakeConst(input_type, 2.26843463243900e-03);
  auto beta_4 = MakeConst(input_type, 1.18534705686654e-04);
  auto beta_6 = MakeConst(input_type, 1.19825839466702e-06);

  return compute(
      x->shape,
      [&](const ffi::Array<PrimVar>& i) {
        auto x2 = x(i) * x(i);
        auto p = x2 * alpha_13 + alpha_11;
        p = x2 * p + alpha_9;
        p = x2 * p + alpha_7;
        p = x2 * p + alpha_5;
        p = x2 * p + alpha_3;
        p = x2 * p + alpha_1;
        p = x(i) * p;

        auto q = x2 * beta_6 + beta_4;
        q = x2 * q + beta_2;
        q = x2 * q + beta_0;
        return p / q;
      },
      name, tag);
}

/*!
 * \brief Creates an operation that returns hyperbolic tanh of a given tensor
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is tanh
 */
inline Tensor fast_tanh(const Tensor& x, std::string name = "T_fast_tanh",
                        std::string tag = kElementWise) {
  if (x->GetDataType().MatchesElementType(DLDataTypeCode::kDLFloat, 32)) {
    // invoke fast_tanh_float implementation
    return fast_tanh_float(x, name, tag);
  } else {
    // fallback to default implementation
    return compute(
        x->shape, [&](const ffi::Array<PrimVar>& i) { return ::tvm::prim::tanh(x(i)); }, name, tag);
  }
}

/*!
 * \brief Creates an operation that returns identity of a given tensor
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the identity operation
 */
inline Tensor identity(const Tensor& x, std::string name = "T_identity",
                       std::string tag = kElementWise) {
  return compute(x->shape, [&](const ffi::Array<PrimVar>& i) { return x(i); }, name, tag);
}

/*!
 * \brief Creates an operation that returns the negation of a given tensor
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the negation operation
 */
inline Tensor negative(const Tensor& x, std::string name = "T_negative",
                       std::string tag = kElementWise) {
  return compute(x->shape, [&](const ffi::Array<PrimVar>& i) { return -x(i); }, name, tag);
}

/*!
 * \brief Creates an operation that returns the logical NOT of a given tensor
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the logical NOT operation
 */
inline Tensor logical_not(const Tensor& x, std::string name = "T_logical_not",
                          std::string tag = kElementWise) {
  return compute(x->shape, [&](const ffi::Array<PrimVar>& i) { return !x(i); }, name, tag);
}

/*!
 * \brief Creates an operation that returns the bitwise NOT of a given tensor
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the bitwise NOT operation
 */
inline Tensor bitwise_not(const Tensor& x, std::string name = "T_bitwise_not",
                          std::string tag = kElementWise) {
  return compute(x->shape, [&](const ffi::Array<PrimVar>& i) { return ~x(i); }, name, tag);
}

/*!
 * \brief Returns the sign of the tensor
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the sign
 */
inline Tensor sign(const Tensor& x, std::string name = "T_sign", std::string tag = kElementWise) {
  using namespace tvm::prim;
  return compute(
      x->shape,
      [&](const ffi::Array<PrimVar>& i) {
        PrimType x_type(x->GetDataType());
        PrimExpr zero = MakeConst(x_type, 0);
        PrimExpr one = MakeConst(x_type, 1);
        PrimExpr minus_one = MakeConst(x_type, -1);
        auto s1 = tvm::prim::Select((x(i) < zero), minus_one, zero);
        auto s2 = tvm::prim::Select((x(i) > zero), one, s1);
        return s2;
      },
      name, tag);
}

/*!
 * \brief Creates an operation that returns rsqrt of a given tensor
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the rsqrt operation
 */
inline Tensor rsqrt(const Tensor& x, std::string name = "tensor", std::string tag = kElementWise) {
  using namespace tvm::prim;
  return compute(
      x->shape,
      [&](const ffi::Array<PrimVar>& i) {
        PrimExpr one = MakeConst(x->GetDataType(), 1);
        return one / tvm::prim::sqrt(x(i));
      },
      name, tag);
}

/*!
 * \brief Creates an operation that clips each element of a tensor to
 * the interval [a_min, a_max]
 *
 * \param x The input tensor
 * \param a_min The inclusive lower bound of the interval
 * \param a_max The inclusive upper bound of the interval
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the clip operation
 */
inline Tensor clip(const Tensor& x, const PrimExpr& a_min, const PrimExpr& a_max,
                   std::string name = "T_clip", std::string tag = kElementWise) {
  return compute(
      x->shape,
      [&](const ffi::Array<PrimVar>& i) {
        PrimType x_type(x->GetDataType());
        auto min_val = tvm::prim::cast(a_min, x_type);
        auto max_val = tvm::prim::cast(a_max, x_type);
        return tvm::max(tvm::min(x(i), max_val), min_val);  // NOLINT(*)
      },
      name, tag);
}

/*!
 * \brief Cast each element of x to the given type. If expr is
 * scalar and type is a corresponding vector type, a
 * Broadcast is generated, otherwise a Cast is generated.
 *
 * \param x The input tensor
 * \param type The type to cast to
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the cast operation
 */
inline Tensor cast(const Tensor& x, PrimType type, std::string name, std::string tag);

inline Tensor cast(const Tensor& x, DLDataType type, std::string name = "T_cast",
                   std::string tag = kElementWise) {
  using namespace tvm::prim;
  return cast(x, PrimType(type), std::move(name), std::move(tag));
}

inline Tensor cast(const Tensor& x, PrimType type, std::string name = "T_cast",
                   std::string tag = kElementWise) {
  return compute(
      x->shape,
      [&](const ffi::Array<PrimVar>& i) -> PrimExpr {
        auto expr = x(i);
        PrimType expr_ty = expr.ty();
        if (expr_ty.MatchesElementType(type.code(), type.bits())) {
          if (expr_ty.lanes() == type.lanes()) {
            return expr;
          } else if (expr_ty.lanes() == 1 && type.IsFixedLengthVector()) {
            return tvm::prim::Broadcast(expr, type.lanes());
          }
        }

        return tvm::prim::cast(x(i), type);
      },
      name, tag);
}

/*!
 * \brief Reinterpret each element of x to the given type.

 * \param x The input tensor
 * \param type The type to cast to
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the reinterpret operation
 */
inline Tensor reinterpret(const Tensor& x, PrimType type, std::string name, std::string tag);

inline Tensor reinterpret(const Tensor& x, DLDataType type, std::string name = "tensor",
                          std::string tag = kElementWise) {
  using namespace tvm::prim;
  return reinterpret(x, PrimType(type), std::move(name), std::move(tag));
}

inline Tensor reinterpret(const Tensor& x, PrimType type, std::string name = "tensor",
                          std::string tag = kElementWise) {
  return compute(
      x->shape, [&](const ffi::Array<PrimVar>& i) { return tirx::reinterpret(type, x(i)); }, name,
      tag);
}

/*!
 * \brief Creates an operation that sum each element of a tensor
 *
 * \param xs The input tensor array
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the sum operation
 */
inline Tensor elemwise_sum(const ffi::Array<Tensor>& xs, std::string name = "T_elemwise_sum",
                           std::string tag = kElementWise) {
  TVM_FFI_ICHECK_GT(xs.size(), 0) << "elemwise sum must have at least one input tensor.";
  return compute(
      xs[0]->shape,
      [&](const ffi::Array<PrimVar>& i) {
        auto sum_expr = xs[0](i);
        for (size_t j = 1; j < xs.size(); j++) {
          sum_expr = sum_expr + xs[j](i);
        }
        return sum_expr;
      },
      name, tag);
}

/*!
 * \brief Creates an operation that fill a tensor with fill_value
 *
 * \param shape The shape of a tensor
 * \param dtype The Type of fill_value
 * \param fill_value The value to be filled
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the full operation
 */
inline Tensor full(const ffi::Array<PrimExpr>& shape, PrimType dtype, const PrimExpr fill_value,
                   std::string name, std::string tag);

inline Tensor full(const ffi::Array<PrimExpr>& shape, DLDataType dtype, const PrimExpr fill_value,
                   std::string name = "T_full", std::string tag = kElementWise) {
  return full(shape, PrimType(dtype), fill_value, std::move(name), std::move(tag));
}

inline Tensor full(const ffi::Array<PrimExpr>& shape, PrimType dtype, const PrimExpr fill_value,
                   std::string name = "T_full", std::string tag = kElementWise) {
  PrimExpr ev = prim::cast(fill_value, dtype);
  if (!ev.defined()) {
    LOG(ERROR) << "Can't cast fill_value to " << dtype;
  }
  return compute(shape, [&](const ffi::Array<PrimVar>& i) { return ev; }, name, tag);
}

/*!
 * \brief Creates an operation that construct a tensor with same shape as input tensor,
 * then fill a tensor with fill_value
 *
 * \param x The input tensor
 * \param fill_value The value to be filled
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op memeber is the full_like operation
 */
inline Tensor full_like(const Tensor& x, const PrimExpr fill_value,
                        std::string name = "T_full_like", std::string tag = kElementWise) {
  PrimExpr ev = prim::cast(fill_value, x->GetDataType());
  return compute(x->shape, [&](const ffi::Array<PrimVar>& i) { return ev; }, name, tag);
}

/*!
 * \brief Fast exponential function implementation
 *
 * \param _x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is exponent operation
 *
 * \note Function computes:
 * log2(e^x) = x * log2(e) * log2(2) =>
 * log2(e^x) = log2(2^(x*log2(e))) =>
 * e^x = 2^(x*log2(e))
 * Splitting power x*log2(e) into integer and fractional parts:
 * e^(n+f) = e^n * e^f
 * n = floor(x*log2(e) + 1/2)
 * f = x - n * ln(2)
 * exp(x) = 2^n * exp(y)
 * Approximation for fractional part:
 * y = exp(f) = 1 + 2 * P(x**2)/(Q(x**2) - P(x**2))
 */
inline Tensor fast_exp_float32(const Tensor& _x, std::string name, std::string tag) {
  PrimType f32_ty = PrimType::Float(32);
  auto x_hi = FloatImm(f32_ty, 88.3762626647950f);
  auto x_lo = FloatImm(f32_ty, -88.3762626647949f);
  auto log2e = FloatImm(f32_ty, 1.44269504088896341f);
  auto ln2 = FloatImm(f32_ty, 0.6931471805599453f);
  PrimExpr p[6] = {FloatImm(f32_ty, 1.9875691500E-4f), FloatImm(f32_ty, 1.3981999507E-3f),
                   FloatImm(f32_ty, 8.3334519073E-3f), FloatImm(f32_ty, 4.1665795894E-2f),
                   FloatImm(f32_ty, 1.6666665459E-1f), FloatImm(f32_ty, 5.0000001201E-1f)};
  auto one = FloatImm(f32_ty, 1.0f);
  auto one_half = FloatImm(f32_ty, 0.5f);
  auto b = FloatImm(f32_ty, 127.0f);

  return compute(
      _x->shape,
      [&](const ffi::Array<PrimVar>& i) {
        // clamp x
        auto x = ::tvm::max(::tvm::min(_x(i), x_hi), x_lo);
        // integer part
        auto n = ::tvm::prim::floor(x * log2e + one_half);
        // fractional part
        auto f = x - n * ln2;
        auto y =
            (((((p[0] * f + p[1]) * f + p[2]) * f + p[3]) * f + p[4]) * f + p[5]) * f * f + f + one;
        // Return 2^m * exp(r).
        auto ef = tvm::tirx::reinterpret(PrimType::Float(32),
                                         ::tvm::prim::cast(n + b, PrimType::Int(32)) << 23);
        return ::tvm::max(ef * y, _x(i));  // NOLINT(*)
      },
      name, tag);
}

/*!
 * \brief Fast exponential function implementation
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is exponent operation
 *
 */
inline Tensor fast_exp(const Tensor& x, std::string name = "T_fast_exp",
                       std::string tag = kElementWise) {
  if (x->GetDataType().MatchesElementType(DLDataTypeCode::kDLFloat, 32)) {
    auto ret = fast_exp_float32(x, name, tag);
    return ret;
  } else {
    return compute(
        x->shape, [&](const ffi::Array<PrimVar>& i) { return ::tvm::prim::exp(x(i)); }, name, tag);
  }
}

/*!
 * \brief Fast_erf_float expression from Eigen
 */
inline Tensor fast_erf_float32(const Tensor& data, std::string name, std::string tag) {
  using namespace tvm::prim;
  return compute(
      data->shape,
      [&](const ffi::Array<PrimVar>& i) { return tirx::fast_erf_float_expr(data(i), 32); }, name,
      tag);
}

/*!
 * \brief Fast_erf_float expression from Eigen for float16.
 */
inline Tensor fast_erf_float16(const Tensor& data, std::string name, std::string tag) {
  using namespace tvm::prim;
  return compute(
      data->shape,
      [&](const ffi::Array<PrimVar>& i) { return tirx::fast_erf_float_expr(data(i), 16); }, name,
      tag);
}

/*!
 * \brief Fast erf implementation
 *
 * \param x The input tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is erf operation
 */
inline Tensor fast_erf(const Tensor& x, std::string name = "T_fast_erf",
                       std::string tag = kElementWise) {
  PrimType x_type(x->GetDataType());
  if (x_type.MatchesElementType(DLDataTypeCode::kDLFloat, 32)) {
    auto ret = fast_erf_float32(x, name, tag);
    return ret;
  } else if (x_type.MatchesElementType(DLDataTypeCode::kDLFloat, 16)) {
    auto ret = fast_erf_float16(x, name, tag);
    return ret;
  } else {
    return topi::erf(x);
  }
}

}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_ELEMWISE_H_
