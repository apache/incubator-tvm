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
 * \file topi/reduction.h
 * \brief Reduction op constructors
 */
#ifndef TVM_TOPI_REDUCTION_H_
#define TVM_TOPI_REDUCTION_H_

#include <tvm/te/operation.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/detail/ravel_unravel.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/tags.h>
#include <tvm/topi/transform.h>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace tvm {
namespace topi {
using namespace tvm::te;

/*! \brief The operation to use for CommReduce */
using FReduce = std::function<PrimExpr(PrimExpr source, const ffi::Array<IterVar>& axis,
                                       ffi::Array<PrimExpr> init, Location loc)>;

/*! \brief The operation to use for CommReduceIdx */
using FCommReduce = std::function<ffi::Array<PrimExpr>(
    ffi::Array<PrimExpr> exprs, const ffi::Array<IterVar>& axis, PrimExpr* condition)>;

/*!
 * \brief Convert a reduction axis which could be empty or have negative
 * elements into a real axis with valid dimension indices.
 *
 * \param ndim Number of dimensions in the target.
 * \param axis The axis parameter.
 *
 * \return A sorted array of valid dimension indices, with no duplicates.
 * If the input axis is None, the result will be an axis including all dimensions.
 * If any input element is negative, it will be treated as an offset from the
 * last dimension (same as python indexing rules).
 */
inline std::vector<int> GetRealAxis(int ndim, const ffi::Optional<ffi::Array<int64_t>>& axis) {
  std::vector<int> real_axis;
  if (!axis.has_value()) {
    for (int i = 0; i < ndim; ++i) {
      real_axis.push_back(i);
    }
  } else {
    // Use a set so duplicates are removed and the dims are sorted
    for (int64_t elem : axis.value()) {
      int64_t val = elem;
      if (val < 0) {
        val += ndim;
      }
      TVM_FFI_ICHECK_LT(val, ndim) << " exceeds the maximum dimension " << ndim;
      TVM_FFI_ICHECK_GE(val, 0);
      real_axis.push_back(static_cast<int>(val));
    }
    std::sort(real_axis.begin(), real_axis.end());
    real_axis.resize(std::unique(real_axis.begin(), real_axis.end()) - real_axis.begin());
  }
  return real_axis;
}

/*! \brief Enumerate the axes for a reduce op */
inline ffi::Array<IterVar> MakeReduceAxes(const std::vector<int>& real_axis, const Tensor& data) {
  ffi::Array<IterVar> reduce_axes;
  for (auto i : real_axis) {
    std::string name = "k" + std::to_string(i);
    reduce_axes.push_back(tvm::te::reduce_axis(Range(0, data->shape[i]), name));
  }
  return reduce_axes;
}

/*! \brief Calculate the target shape for a reduce op */
inline ffi::Array<PrimExpr> MakeReduceTargetShape(const std::vector<int>& real_axis,
                                                  const Tensor& data, bool keepdims,
                                                  bool atleast1d) {
  auto ndim = data->shape.size();
  ffi::Array<PrimExpr> target_shape;
  if (keepdims) {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        // real_axis contains i
        target_shape.push_back(1);
      } else {
        target_shape.push_back(data->shape[i]);
      }
    }
  } else {
    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) == real_axis.end()) {
        // real_axis does not contain i
        target_shape.push_back(data->shape[i]);
      }
    }
  }
  if (target_shape.size() == 0 && atleast1d) {
    target_shape.push_back(1);
  }
  return target_shape;
}

/*!
 * \brief Create a reduction operation.
 *
 * \param data The input tensor.
 * \param func The reduction function eg. tvm::prim::sum
 * \param target_shape The output Tensor shape.
 * \param reduce_axes The real axes along which the reduction is performed.
 * \param squeeze_axes The real axes to squeeze. Unsqueezed, reduced axes will
 *                     have shape 1 in the output tensor.
 * \param loc The location of this reducer in the source.
 *
 * \return The result tensor.
 */
inline Tensor DoCommReduce(const Tensor& data, FReduce func,
                           const ffi::Array<PrimExpr>& target_shape,
                           const std::vector<int>& reduce_axes,
                           const std::vector<int>& squeeze_axes, Location loc = UnknownLoc()) {
  auto r_axes = MakeReduceAxes(reduce_axes, data);
  auto compute = [&](const ffi::Array<PrimVar>& indices) {
    ffi::Array<PrimExpr> eval_range;
    ffi::Array<PrimVar> eval_indices;
    int arg_counter = 0;
    int red_counter = 0;

    for (size_t i = 0; i < data->shape.size(); ++i) {
      bool squeeze_i = std::find(squeeze_axes.begin(), squeeze_axes.end(), i) != squeeze_axes.end();
      if (std::find(reduce_axes.begin(), reduce_axes.end(), i) != reduce_axes.end()) {
        // real_axis contains i
        eval_range.push_back(r_axes[red_counter]);
        eval_indices.push_back(r_axes[red_counter]->var);
        red_counter++;
        arg_counter += !squeeze_i;
        continue;
      }
      eval_range.push_back(indices[arg_counter]);
      arg_counter++;
    }

    return func(data(eval_range), r_axes, {}, loc);
  };

  return tvm::te::compute(target_shape, compute, data->op.value()->name + "_red", kCommReduce);
}

/*!
 * \brief Create a reduction operation.
 *
 * \param data The input tensor.
 * \param axis The axes along which the reduction is performed.
 * \param func The reduction function eg. tvm::prim::sum
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 *
 * \return The result tensor.
 */
inline Tensor CommReduce(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                         FReduce func, bool keepdims, bool atleast1d) {
  auto ndim = data->shape.size();
  if (ndim == 0) {
    auto identity = topi::identity(data, data->op.value()->name + "_red", kCommReduce);
    return atleast1d ? topi::expand_dims(identity, 0, 1) : identity;
  }
  auto real_axis = GetRealAxis(static_cast<int>(ndim), axis);
  auto target_shape = MakeReduceTargetShape(real_axis, data, keepdims, atleast1d);
  return DoCommReduce(data, func, target_shape, real_axis,
                      keepdims ? std::vector<int>() : real_axis);
}

/*!
 * \brief Create an index reduction operation.
 *
 * \param data The input tensor.
 * \param axis The axes along which the reduction is performed.
 * \param func The reduction function
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 *
 * \return The result tensor.
 */
inline Tensor CommReduceIdx(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                            FCommReduce func, bool keepdims, bool atleast1d) {
  auto ndim = data->shape.size();
  TVM_FFI_ICHECK_NE(ndim, 0) << "Cannot reduce a 0 dim Tensor";
  auto real_axis = GetRealAxis(static_cast<int>(ndim), axis);
  auto reduce_axes = MakeReduceAxes(real_axis, data);
  auto target_shape = MakeReduceTargetShape(real_axis, data, keepdims, atleast1d);

  auto compute = [ndim, keepdims, &real_axis, &reduce_axes, &func,
                  &data](const ffi::Array<PrimVar>& indices) {
    ffi::Array<PrimExpr> eval_range;
    ffi::Array<PrimExpr> eval_indices;
    int arg_counter = 0;
    int red_counter = 0;

    for (size_t i = 0; i < ndim; ++i) {
      if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
        // real_axis contains i
        eval_range.push_back(reduce_axes[red_counter]);
        eval_indices.push_back(reduce_axes[red_counter]->var);
        red_counter++;
      } else {
        if (!keepdims) {
          eval_range.push_back(indices[arg_counter]);
          arg_counter++;
        } else {
          eval_range.push_back(indices[i]);
        }
      }
    }

    ffi::Array<PrimExpr> ravel_shape;
    for (auto i : real_axis) {
      ravel_shape.push_back(data->shape[i]);
    }
    auto idx = detail::RavelIndex(eval_indices, ravel_shape);
    return func({idx, data(eval_range)}, reduce_axes, nullptr);
  };

  auto temp_idx_val =
      tvm::te::compute(target_shape, compute, data->op.value()->name + "_red_temp", kCommReduceIdx);
  auto temp_idx = temp_idx_val[0];
  auto temp_val = temp_idx_val[1];
  return tvm::te::compute(
      target_shape, [&temp_idx](const ffi::Array<PrimVar>& indices) { return temp_idx(indices); },
      data->op.value()->name + "_red", kCommReduceIdx);
}

/*! \brief A combiner function for a reduction */
using FCombine =
    std::function<ffi::Array<PrimExpr>(ffi::Array<PrimVar> lhs, ffi::Array<PrimVar> rhs)>;

/*! \brief An initializer function for a reduction */
using FIdentity = std::function<ffi::Array<PrimExpr>(std::vector<PrimType> types)>;

/*!
 * \brief Create a commutative reducer for a reduction
 *
 * \param fcombine A function to combine exprs
 * \param fidentity A function to initialize elements
 * \param name The name of the operation
 *
 * \return A reducer function which creates a reduce expression over an axis.
 */
inline FCommReduce MakeCommReducer(FCombine fcombine, FIdentity fidentity,
                                   std::string name = "reduce") {
  return [fcombine, fidentity, name](ffi::Array<PrimExpr> exprs, const ffi::Array<IterVar>& axis,
                                     PrimExpr* condition) {
    ffi::Array<PrimVar> lhs, rhs;
    ffi::Array<PrimVar> callback_lhs, callback_rhs;
    std::vector<PrimType> dtypes;

    for (size_t i = 0; i < exprs.size(); ++i) {
      PrimType dtype = exprs[i].ty();
      dtypes.push_back(dtype);
      PrimVar lhs_var(name + "_lhs_" + std::to_string(i), dtype);
      PrimVar rhs_var(name + "_rhs_" + std::to_string(i), dtype);
      lhs.push_back(lhs_var);
      rhs.push_back(rhs_var);
      callback_lhs.push_back(lhs_var);
      callback_rhs.push_back(rhs_var);
    }

    auto result = fcombine(callback_lhs, callback_rhs);
    auto id_elem = fidentity(dtypes);
    auto cond = condition != nullptr ? *condition : IntImm::Bool(true);

    auto combiner = tvm::te::CommReducer(lhs, rhs, result, id_elem);
    ffi::Array<PrimExpr> outputs;
    for (size_t i = 0; i < exprs.size(); ++i) {
      outputs.push_back(tvm::te::Reduce(combiner, exprs, axis, cond, static_cast<int>(i), {}));
    }
    return outputs;
  };
}

/*! \brief Wrap tvm::min to ensure we get the correct overload */
inline PrimExpr MinOp(PrimExpr source, ffi::Array<IterVar> axis, ffi::Array<PrimExpr> init = {},
                      Location loc = UnknownLoc()) {
  return tvm::min(source, axis, init, loc);
}

/*! \brief Wrap tvm::max to ensure we get the correct overload */
inline PrimExpr MaxOp(PrimExpr source, ffi::Array<IterVar> axis, ffi::Array<PrimExpr> init = {},
                      Location loc = UnknownLoc()) {
  return tvm::max(source, axis, init, loc);  // NOLINT(*)
}

/*! \brief Wrap tvm::prim::prod to ensure we get the correct overload */
inline PrimExpr ProdOp(PrimExpr source, ffi::Array<IterVar> axis, ffi::Array<PrimExpr> init = {},
                       Location loc = UnknownLoc()) {
  return tvm::prim::prod(source, axis, init, loc);  // NOLINT(*)
}

/*!
 * \brief Creates an operation that sums array elements over a given axis
 *
 * \param data The input tensor
 * \param axis The axis to sum over. If axis is empty, the operation will
 * sum over all elements of the array.
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 *
 * \return A Tensor whose op member is the sum operation
 */
inline Tensor sum(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                  bool keepdims = false, bool atleast1d = false) {
  // Reduction dispatch only depends on boolean element kind; lane encoding is irrelevant here.
  if (data->dtype.code() == DLDataTypeCode::kDLBool) {
    return CommReduce(data, axis, tvm::prim::any, keepdims, atleast1d);
  } else {
    return CommReduce(data, axis, tvm::prim::sum, keepdims, atleast1d);
  }
}

inline Tensor collapse_sum(const Tensor& data, ffi::Array<PrimExpr> target_shape) {
  const auto& ishape = data->shape;
  const auto& oshape = target_shape;
  int isize = data->shape.size();
  int osize = target_shape.size();

  TVM_FFI_ICHECK_GE(isize, osize)
      << "Invalid collapse: input dimensionality smaller than output dimensionality.\ninput shape: "
      << data->shape << "\nvs\noutput shape: " << target_shape;

  std::vector<int> reduce_axes;
  std::vector<int> squeeze_axes;
  tvm::PrimExpr one(1);

  for (int i_ax = isize - 1, o_ax = osize - 1; i_ax >= 0; --i_ax) {
    if (o_ax >= 0 && topi::detail::EqualCheck(ishape[i_ax], oshape[o_ax])) {
      --o_ax;
      continue;
    }
    reduce_axes.push_back(i_ax);
    if (o_ax < 0) {  // squeeze o_ax if was added during expansion
      squeeze_axes.push_back(i_ax);
    } else if (topi::detail::EqualCheck(one, oshape[o_ax])) {
      --o_ax;
    }
  }

  if (reduce_axes.size() == 0) return topi::identity(data, "tensor", kCommReduce);

  std::reverse(reduce_axes.begin(), reduce_axes.end());
  std::reverse(squeeze_axes.begin(), squeeze_axes.end());
  return DoCommReduce(data, tvm::prim::sum, target_shape, reduce_axes, squeeze_axes);
}

/*!
 * \brief Creates an operation that computes the logical AND of elements
 * over a given axis
 *
 * \param data The input boolean tensor
 * \param axis The axes to reduce. If axis is empty, the operation will
 * perform logical AND over all elements of the array.
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 *
 * \return A Tensor whose op member is the all operation
 */
inline Tensor all(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                  bool keepdims = false, bool atleast1d = false) {
  return CommReduce(data, axis, tvm::prim::all, keepdims, atleast1d);
}

/*!
 * \brief Creates an operation that computes the logical OR of elements
 * over a given axis
 *
 * \param data The input boolean tensor
 * \param axis The axes to reduce. If axis is empty, the operation will
 * perform logical OR over all elements of the array.
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 *
 * \return A Tensor whose op member is the all operation
 */
inline Tensor any(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                  bool keepdims = false, bool atleast1d = false) {
  return CommReduce(data, axis, tvm::prim::any, keepdims, atleast1d);
}

/*!
 * \brief Creates an operation that finds the minimum of elements over
 * a given axis.
 *
 * \param data The input tensor
 * \param axis The axis to find the minimum over. If axis is empty, the
 * operation will find the minimum over all elements of the array.
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 *
 * \return A Tensor whose op member is the min operation
 */
inline Tensor min(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                  bool keepdims = false, bool atleast1d = false) {
  return CommReduce(data, axis, MinOp, keepdims, atleast1d);
}

/*!
 * \brief Creates an operation that finds the maximum of elements over
 * a given axis.
 *
 * \param data The input tensor
 * \param axis The axis to find the maximum over. If axis is empty, the
 * operation will find the maximum over all elements of the array.
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 *
 * \return A Tensor whose op member is the max operation
 */
inline Tensor max(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                  bool keepdims = false, bool atleast1d = false) {
  return CommReduce(data, axis, MaxOp, keepdims, atleast1d);
}

inline FCommReduce MakeArgminReducer(bool select_last_index = false) {
  // Create a Commutative Reducer with a comparison operation, and method to get the initial value.
  auto fcombine = [=](ffi::Array<PrimVar> lhs, ffi::Array<PrimVar> rhs) {
    ffi::Array<PrimExpr> result;

    // Casting to avoid operator ambiguity
    PrimExpr lhs_idx = static_cast<PrimExpr>(lhs[0]);
    PrimExpr rhs_idx = static_cast<PrimExpr>(rhs[0]);
    PrimExpr lhs_val = static_cast<PrimExpr>(lhs[1]);
    PrimExpr rhs_val = static_cast<PrimExpr>(rhs[1]);

    // These variables compare the actual values of the array
    auto is_smaller = lhs_val < rhs_val;
    auto is_same = lhs_val == rhs_val;

    // This checks if the indices are correct for the reduction. E.g. for select_last_index
    // it gives precedence for later indices of the same element and precedence for sooner
    // indices if not select_last_index;
    PrimExpr proper_index = select_last_index ? lhs_idx > rhs_idx : lhs_idx < rhs_idx;

    PrimExpr update_index = is_smaller || (is_same && proper_index);
    result.push_back(tvm::prim::Select(update_index, lhs[0], rhs[0]));  // idx
    result.push_back(tvm::prim::Select(is_smaller, lhs[1], rhs[1]));    // val
    return result;
  };
  auto fidentity = [&](std::vector<PrimType> types) {
    ffi::Array<PrimExpr> result;
    result.push_back(tvm::prim::MakeConst(types[0], -1));  // idx
    result.push_back(tvm::prim::max_value(types[1]));      // val
    return result;
  };
  return MakeCommReducer(fcombine, fidentity, "argmin");
}

/*!
 * \brief Creates an operation that finds the indices of the minimum
 * values over a given axis.
 *
 * \param data The input tensor
 * \param axis The axis along which the argmin is performed. If axis is empty,
 * the operation will find the minimum index over all elements of the array.
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 * \param select_last_index Whether to select the last index if the minimum element
 * appears multiple times, else select the first index.
 *
 * \return A Tensor whose op member is the argmin operation
 */
inline Tensor argmin(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                     bool keepdims = false, bool atleast1d = false,
                     bool select_last_index = false) {
  auto reducer = MakeArgminReducer(select_last_index);
  return CommReduceIdx(data, axis, reducer, keepdims, atleast1d);
}

inline FCommReduce MakeArgmaxReducer(bool select_last_index = false) {
  // Create a Commutative Reducer with a comparison operation, and method to get the initial value.
  auto fcombine = [=](ffi::Array<PrimVar> lhs, ffi::Array<PrimVar> rhs) {
    ffi::Array<PrimExpr> result;

    // Casting to avoid operator ambiguity
    PrimExpr lhs_idx = static_cast<PrimExpr>(lhs[0]);
    PrimExpr rhs_idx = static_cast<PrimExpr>(rhs[0]);
    PrimExpr lhs_val = static_cast<PrimExpr>(lhs[1]);
    PrimExpr rhs_val = static_cast<PrimExpr>(rhs[1]);

    // These variables compare the actual values of the array
    auto is_bigger = lhs_val > rhs_val;
    auto is_same = lhs_val == rhs_val;

    // This checks if the indices are correct for the reduction. E.g. for select_last_index
    // it gives precedence for later indices of the same element and precedence for sooner
    // indices if not select_last_index;
    PrimExpr proper_index = select_last_index ? lhs_idx > rhs_idx : lhs_idx < rhs_idx;

    PrimExpr update_index = is_bigger || (is_same && proper_index);
    result.push_back(tvm::prim::Select(update_index, lhs[0], rhs[0]));  // idx
    result.push_back(tvm::prim::Select(is_bigger, lhs[1], rhs[1]));     // val
    return result;
  };
  auto fidentity = [&](std::vector<PrimType> types) {
    ffi::Array<PrimExpr> result;
    result.push_back(tvm::prim::MakeConst(types[0], -1));  // idx
    result.push_back(tvm::prim::min_value(types[1]));      // val
    return result;
  };
  return MakeCommReducer(fcombine, fidentity, "argmax");
}

/*!
 * \brief Creates an operation that finds the indices of the maximum
 * values over a given axis.
 *
 * \param data The input tensor
 * \param axis The axis along which the argmax is performed. If axis is empty,
 * the operation will find the maximum index over all elements of the array.
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 * \param select_last_index Whether to select the last index if the maximum element
 * appears multiple times, else select the first index.
 * \return A Tensor whose op member is the argmax operation
 */
inline Tensor argmax(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                     bool keepdims = false, bool atleast1d = false,
                     bool select_last_index = false) {
  auto reducer = MakeArgmaxReducer(select_last_index);
  return CommReduceIdx(data, axis, reducer, keepdims, atleast1d);
}

/*!
 * \brief Creates product operation over given axis.
 *
 * \param data The input tensor
 * \param axis The axis to do product over. If axis is empty, the
 * operation will do the product over all elements of the array.
 * \param keepdims If this is set to true, the axes which are reduced are
 * left in the result as dimensions with size one. This enables the result
 * to broadcast correctly against the input array.
 * \param atleast1d Whether the output need to be atleast1d.
 *
 * \return A Tensor whose op member is the prod operation
 */
inline Tensor prod(const Tensor& data, const ffi::Optional<ffi::Array<int64_t>>& axis,
                   bool keepdims = false, bool atleast1d = false) {
  return CommReduce(data, axis, ProdOp, keepdims, atleast1d);
}

/*!
 * \brief Create communitive reducer summing over tuples
 */
inline FCommReduce MakeTupleSumReducer() {
  auto fcombine = [](ffi::Array<PrimVar> lhs, ffi::Array<PrimVar> rhs) {
    ffi::Array<PrimExpr> result;
    TVM_FFI_ICHECK_EQ(lhs.size(), rhs.size());
    result.reserve(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
      result.push_back(lhs[i] + rhs[i]);
    }
    return result;
  };
  auto fidentity = [](std::vector<PrimType> types) {
    ffi::Array<PrimExpr> result;
    for (size_t i = 0; i < types.size(); ++i) {
      result.push_back(tvm::prim::MakeConst(types[i], 0));
    }
    return result;
  };
  return MakeCommReducer(fcombine, fidentity, "tuple_sum");
}

}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_REDUCTION_H_
