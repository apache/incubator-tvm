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
 * \file topi/transform.h
 * \brief Transform op constructors
 */
#ifndef TVM_TOPI_TRANSFORM_H_
#define TVM_TOPI_TRANSFORM_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/s_tir/data_layout.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/sym/analyzer.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/index_map.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/detail/broadcast.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/detail/ravel_unravel.h>
#include <tvm/topi/detail/strided_slice.h>
#include <tvm/topi/detail/tensor_utils.h>
#include <tvm/topi/tags.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tvm/ffi/dtype.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace topi {
using namespace tvm::te;
using namespace topi::detail;

/*!
 * \brief Creates an operation to slide a window over the input x.
 *
 * \param x The input tensor.
 * \param axis What axis the window begins sliding over. Window will be slid
 * over this axis and all following axes. The axis value determines the window
 * shape (and thus, the number of strides): window shape and strides must both
 * be of length `data.ndim-axis`.
 * \param window_shape The window shape to form over the input. Window shape
 * must be of length `data.ndim-axis`.
 * \param strides How to stride the window along each dimension. Strides must be
 * of length `data.ndim-axis`.
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the sliding_window operation
 */
inline Tensor sliding_window(const Tensor& x, int axis, ffi::Array<int64_t> window_shape,
                             ffi::Array<int64_t> strides, std::string name = "T_sliding_window",
                             std::string tag = "") {
  TVM_FFI_ICHECK_GE(axis, 0);
  auto _axis = size_t(axis);
  TVM_FFI_ICHECK_LT(_axis, x->shape.size()) << "axis must be a valid dimension index of x.";
  TVM_FFI_ICHECK_EQ(x->shape.size() - _axis, window_shape.size())
      << "There must be a window shape for every dimension of x "
      << "over which we are sliding the window.";
  TVM_FFI_ICHECK_EQ(strides.size(), window_shape.size())
      << "Windows and strides should be the same length.";

  // Compute the new shape.
  ffi::Array<PrimExpr> new_shape;
  // Dimensions up until `axis` remain the same.
  for (size_t i = 0; i < _axis; ++i) {
    new_shape.push_back(x->shape[i]);
  }

  // New dimensions which result from sliding the window in each dimension. One new dimension per
  // window dimension.
  for (size_t i = 0; i < window_shape.size(); ++i) {
    // Length of the shape along this dimension.
    auto dim_len = x->shape[_axis + i];
    // Length of the window along this dimension.
    PrimExpr window_len = IntImm::Int64(window_shape[i]);
    // Strides along this dimension.
    PrimExpr stride = IntImm::Int64(strides[i]);

    new_shape.push_back(floordiv(dim_len - (window_len - 1) + stride - 1, stride));
  }

  // Dimensions comprising the window.
  for (size_t i = 0; i < window_shape.size(); ++i) {
    new_shape.push_back(IntImm::Int64(window_shape[i]));
  }

  TVM_FFI_ICHECK(new_shape.size() == _axis + 2 * window_shape.size());

  return compute(
      new_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        // The index at which to index the old tensor x.
        ffi::Array<PrimExpr> idx;

        // Dimensions up until `axis` remain the same.
        for (size_t i = 0; i < _axis; ++i) {
          idx.push_back(indices[i]);
        }

        for (size_t i = 0; i < window_shape.size(); ++i) {
          // Which window in this dimension we are indexing.
          auto window_idx = indices[_axis + i];
          // Which index within the window we are indexing.
          auto idx_within_window = indices[_axis + window_shape.size() + i];
          // Stride value for this dimension.
          PrimExpr stride = IntImm::Int64(strides[i]);

          idx.push_back(window_idx * stride + idx_within_window);
        }

        TVM_FFI_ICHECK(idx.size() == x->shape.size());

        return x(idx);
      },
      name, tag);
}

/*!
 * \brief Creates an operation to insert new dimensions of length 1
 *
 * \param x The input tensor
 * \param axis The index of the first new dimension (allows negative
 * indices as offsets from the last dimension)
 * \param num_newaxis The number of new dimensions to insert
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the dim expansion operation
 */
inline Tensor expand_dims(const Tensor& x, int axis, int num_newaxis = 1,
                          std::string name = "T_expand_dims", std::string tag = kBroadcast) {
  int ndim = static_cast<int>(x->shape.size());
  TVM_FFI_ICHECK(-ndim - 1 <= axis && axis <= ndim)
      << "expand_dims only accepts `axis` in [-data.ndim - 1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  TVM_FFI_ICHECK(num_newaxis >= 0) << "expand_dims only accepts `num_newaxis >= 0`"
                                   << ", but got num_newaxis = " << num_newaxis;
  if (axis < 0) {
    // Calculate offset from last dimension
    axis = ndim + axis + 1;
  }
  ffi::Array<PrimExpr> new_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    new_shape.push_back(x->shape[i]);
  }
  for (size_t i = 0; i < static_cast<size_t>(num_newaxis); ++i) {
    new_shape.push_back(1);
  }
  for (size_t i = axis; i < x->shape.size(); ++i) {
    new_shape.push_back(x->shape[i]);
  }

  return compute(
      new_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        ffi::Array<PrimExpr> idx;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
          idx.push_back(indices[i]);
        }
        for (size_t i = axis + num_newaxis; i < indices.size(); ++i) {
          idx.push_back(indices[i]);
        }
        return x(idx);
      },
      name, tag);
}

/*!
 * \brief Permute the dimensions of an array
 *
 * \param x The input tensor
 * \param opt_axes The indices of the permutation. If this is empty,
 * the dimensions will be reversed.
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the transpose operation
 */
inline Tensor transpose(const Tensor& x, ffi::Optional<ffi::Array<int64_t>> opt_axes,
                        std::string name = "T_transpose", std::string tag = kInjective) {
  ffi::Array<int64_t> axes = opt_axes.value_or({});
  if (axes.size() == 0) {
    for (int i = static_cast<int>(x->shape.size()) - 1; i >= 0; --i) {
      axes.push_back(i);
    }
  }

  ffi::Array<PrimExpr> new_shape;
  for (size_t i = 0; i < axes.size(); ++i) {
    int axis = static_cast<int>(axes[i]);
    int new_axis = axis;
    if (axis < 0) {
      new_axis = static_cast<int>(x->shape.size()) + axis;
      axes.Set(i, new_axis);
    }
    TVM_FFI_ICHECK((new_axis >= 0) && (new_axis < static_cast<int>(x->shape.size())))
        << "axis=" << axis << " is invalid for the " << static_cast<int>(x->shape.size())
        << "-dimensional input tensor";

    for (size_t j = 0; j < axes.size(); ++j) {
      if (i != j) {
        TVM_FFI_ICHECK(new_axis != static_cast<int>(axes[j])) << "repeated axis in transpose";
      }
    }
    new_shape.push_back(x->shape[new_axis]);
  }

  return compute(
      new_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        std::vector<PrimExpr> idx;
        for (size_t i = 0; i < axes.size(); ++i) {
          idx.push_back(1);
        }
        for (size_t i = 0; i < axes.size(); ++i) {
          int axis = static_cast<int>(axes[i]);
          idx[axis] = indices[i];
        }
        return x(idx);
      },
      name, tag);
}

/*!
 * \brief Reverse the tensor for variable length slices.
 * Input is first sliced along batch axis and then elements are reversed along seq axis.
 *
 * \param x The input tensor
 * \param seq_lengths A 1D Tensor with length x.dims[batch_axis]. Pass std::nullopt to omit it.
 * If not defined batch axis is ignored and tensor is reversed along seq_axis.
 * \param seq_axis The axis along which the elements will be reveresed
 * \param batch_axis The axis along which the tensor will be sliced
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the reverse_sequence operation
 */
inline Tensor reverse_sequence(const Tensor& x, const ffi::Optional<Tensor>& seq_lengths,
                               int seq_axis = 1, int batch_axis = 0,
                               std::string name = "T_reverse_sequence",
                               std::string tag = kInjective) {
  size_t src_tensor_dim = x->shape.size();
  int seq_axis_inp = seq_axis;

  if (seq_lengths.has_value()) {
    size_t seq_lengths_dim = seq_lengths.value()->shape.size();
    int batch_axis_inp = batch_axis;
    if (batch_axis < 0) {
      batch_axis = static_cast<int>(x->shape.size()) + batch_axis;
    }

    TVM_FFI_ICHECK(seq_lengths_dim == 1) << "seq_lengths should be 1D vector";

    TVM_FFI_ICHECK(GetConstInt(seq_lengths.value()->shape[0]) == GetConstInt(x->shape[batch_axis]))
        << "For reverse_sequnece seq_lengths size should match with dimension of batch axis"
        << ", but got dimension of batch_axis = " << GetConstInt(x->shape[batch_axis])
        << ", and seq_length size = " << GetConstInt(seq_lengths.value()->shape[0]);

    TVM_FFI_ICHECK((0 <= batch_axis) && (batch_axis < static_cast<int>(x->shape.size())))
        << "batch_axis=" << batch_axis_inp << " is invalid for the "
        << static_cast<int>(x->shape.size()) << "-dimensional input tensor";
  }

  if (seq_axis < 0) {
    seq_axis = static_cast<int>(x->shape.size()) + seq_axis;
  }
  TVM_FFI_ICHECK((0 <= seq_axis) && (seq_axis < static_cast<int>(x->shape.size())))
      << "seq_axis=" << seq_axis_inp << " is invalid for the " << static_cast<int>(x->shape.size())
      << "-dimensional input tensor";

  auto func = [&](const ffi::Array<PrimVar>& indices) {
    ffi::Array<PrimExpr> real_indices;
    for (size_t i = 0; i < src_tensor_dim; ++i) {
      if (i == static_cast<size_t>(seq_axis)) {
        if (seq_lengths.has_value()) {
          auto len = seq_lengths.value()(indices[batch_axis]);
          auto idx = if_then_else(
              len <= 1 || len <= indices[i], indices[i],
              if_then_else(len > x->shape[i], x->shape[i] - 1 - indices[i], len - 1 - indices[i]));
          real_indices.push_back(idx);
        } else {
          real_indices.push_back(x->shape[i] - 1 - indices[i]);
        }
      } else {
        real_indices.push_back(indices[i]);
      }
    }
    return x(real_indices);
  };

  return compute(x->shape, func, name, tag);
}

/*!
 * \brief Reshape a tensor
 *
 * \param x The input tensor
 * \param newshape The new shape
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the reshape operation
 */
inline Tensor reshape(const Tensor& x, ffi::Array<PrimExpr> newshape,
                      std::string name = "T_reshape", std::string tag = kInjective) {
  auto x_shape = x->shape;
  ffi::Array<PrimExpr> target_shape;

  for (const auto& ele : newshape) {
    target_shape.push_back(ele);
  }

  // If either the input shape or the target shape contains a zero, return an empty tensor.
  if (is_empty_shape(target_shape) || is_empty_shape(x->shape)) {
    return compute(
        target_shape,
        [&](const ffi::Array<PrimVar>& indices) { return tvm::prim::cast(0, PrimType(x->dtype)); },
        name, tag);
  } else {
    return compute(
        target_shape,
        [&](const ffi::Array<PrimVar>& indices) {
          ffi::Array<PrimExpr> prim_indices =
              indices.Map([](const PrimVar& var) { return var.as_or_throw<PrimExpr>(); });
          return x(UnravelIndex(RavelIndex(prim_indices, target_shape), x_shape));
        },
        name, tag);
  }
}

/*!
 * \brief Converts a flat index or array of flat indices into a tuple of coordinate arrays
 *
 * \param x The input tensor having indices.
 * \param shape The shape tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor of coordinate arrays.
 */

inline Tensor unravel_index(const Tensor& x, const Tensor& shape, std::string name = "T_unravel",
                            std::string tag = kInjective) {
  auto x_shape = x->shape;
  auto shape_shape = shape->shape;

  ffi::Array<PrimExpr> oshape;
  oshape.push_back(shape_shape[0]);
  if (x_shape.size() != 0) {
    oshape.push_back(x_shape[0]);
  }

  auto func = [&](const ffi::Array<PrimVar>& indices) {
    auto i = indices[0];
    std::vector<PrimExpr> indices_divs;
    PrimExpr ret = 0;
    PrimExpr cur_val = 0;
    PrimExpr index_val = 0;

    if (x_shape.size() != 0) {
      index_val = x[indices[1]];
    } else {
      index_val = x();
    }
    indices_divs.push_back(index_val);
    for (int v = GetConstInt(shape_shape[0]) - 1; v >= 0; --v) {
      ret = tvm::if_then_else(i == v, indexmod(indices_divs.back(), shape[v]), ret);
      cur_val = indexdiv(indices_divs.back(), shape[v]);
      indices_divs.push_back(cur_val);
    }
    return ret;
  };

  return compute(oshape, func, name, tag);
}

/*!
 * \brief Remove size 1 dimensions from the shape of a tensor.
 * The removed dimensions must have a constant size of 1.
 *
 * \param x The input tensor
 * \param opt_axes Indices of the dimensions to remove. If this is None,
 * all entries with a constant size of 1 will be removed.
 * \param atleast1d Whether the output need to be atleast1d.
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the squeeze operation
 */
inline Tensor squeeze(const Tensor& x, ffi::Optional<ffi::Array<int64_t>> opt_axes,
                      bool atleast1d = false, std::string name = "T_squeeze",
                      std::string tag = kInjective) {
  auto ndim = x->shape.size();
  std::vector<int> axis_val;
  if (!opt_axes.has_value()) {
    for (size_t i = 0; i < ndim; ++i) {
      if (IsConstInt(x->shape[i]) && x->shape[i].as_or_throw<IntImm>()->value == 1) {
        axis_val.push_back(static_cast<int>(i));
      }
    }
  } else {
    ffi::Array<int64_t> axis = *std::move(opt_axes);
    for (size_t i = 0; i < axis.size(); ++i) {
      int64_t val = axis[i];
      if (val < 0) {
        val += static_cast<int>(x->shape.size());
      }
      // If a dimension is not 1, silently skip it (no-op).
      bool is_const = IsConstInt(x->shape[val]);
      if ((is_const && x->shape[val].as_or_throw<IntImm>()->value == 1) || !is_const) {
        axis_val.push_back(val);
      }
    }
  }

  std::unordered_set<int> axis_set(axis_val.begin(), axis_val.end());

  ffi::Array<PrimExpr> out_shape;
  for (size_t i = 0; i < ndim; ++i) {
    if (axis_set.count(static_cast<int>(i)) == 0) {
      out_shape.push_back(x->shape[i]);
    }
  }
  if (out_shape.size() == 0 && atleast1d) {
    out_shape.push_back(1);
  }

  return compute(
      out_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        ffi::Array<PrimExpr> real_indices;
        int flag = 0;
        for (size_t i = 0; i < ndim; ++i) {
          if (axis_set.count(static_cast<int>(i)) == 0) {
            real_indices.push_back(indices[i - flag]);
          } else {
            real_indices.push_back(0);
            flag += 1;
          }
        }
        return x(real_indices);
      },
      name, tag);
}

/*!
 * \brief Join a sequence of tensors along an existing axis
 *
 * \param inputs The input tensors
 * \param axis The axis along which the tensors will be joined
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the concatenate operation
 */
inline Tensor concatenate(const ffi::Array<Tensor>& inputs, int axis = 0,
                          std::string name = "T_concat", std::string tag = kInjective) {
  int ndim = static_cast<int>(inputs[0]->shape.size());
  TVM_FFI_ICHECK(-ndim <= axis && axis < ndim)
      << "concatenate only accepts `axis` in [-ndim, ndim)"
      << ", but got axis = " << axis << ", and ndim = " << ndim;
  if (axis < 0) {
    axis += ndim;
  }
  TVM_FFI_ICHECK_LT(axis, inputs[0]->shape.size()) << "axis out of bounds";

  ffi::Array<PrimExpr> axis_sizes;
  for (auto t : inputs) {
    axis_sizes.push_back(t->shape[axis]);
  }
  sym::Analyzer analyzer;
  PrimExpr join_size = axis_sizes[0];
  for (size_t i = 1; i < axis_sizes.size(); ++i) {
    join_size += axis_sizes[i];
  }
  join_size = analyzer->Simplify(join_size);
  ffi::Array<PrimExpr> out_shape;
  for (size_t i = 0; i < inputs[0]->shape.size(); ++i) {
    out_shape.push_back(i == static_cast<size_t>(axis) ? join_size : inputs[0]->shape[i]);
  }

  return compute(
      out_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        auto ret = inputs[0](indices);
        PrimExpr ind = indices[axis].as_or_throw<PrimExpr>();
        for (size_t i = 0; i < inputs.size() - 1; ++i) {
          ind -= axis_sizes[i];

          ffi::Array<PrimExpr> idx;
          for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
            idx.push_back(indices[i]);
          }
          idx.push_back(ind);
          for (size_t i = axis + 1; i < indices.size(); ++i) {
            idx.push_back(indices[i]);
          }

          ret = tvm::if_then_else(ind >= 0, inputs[i + 1](idx), ret);
        }
        return ret;
      },
      name, tag);
}

/*!
 * \brief Join a sequence of tensors along a new axis.
 *
 * \param inputs The input tensors
 * \param axis The axis along which the tensors will be stacked
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the stack operation
 */
inline Tensor stack(const ffi::Array<Tensor>& inputs, int axis = 0, std::string name = "T_stack",
                    std::string tag = kInjective) {
  int ndim = static_cast<int>(inputs[0]->shape.size());
  TVM_FFI_ICHECK(-ndim - 1 <= axis && axis <= ndim)
      << "stack only accepts `axis` in [-ndim, ndim)"
      << ", but got axis = " << axis << ", and ndim = " << ndim;
  if (axis < 0) {
    axis += ndim + 1;
  }
  TVM_FFI_ICHECK_LT(axis, inputs[0]->shape.size() + 1) << "axis out of bounds";

  const int stack_size = static_cast<int>(inputs.size());
  ffi::Array<PrimExpr> out_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) out_shape.push_back(inputs[0]->shape[i]);
  out_shape.push_back(stack_size);
  for (size_t i = static_cast<size_t>(axis); i < static_cast<size_t>(ndim); ++i)
    out_shape.push_back(inputs[0]->shape[i]);

  return compute(
      out_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        ffi::Array<PrimExpr> idx;
        for (size_t i = 0; i < indices.size(); ++i)
          if (i != static_cast<size_t>(axis)) idx.push_back(indices[i]);
        auto ind = indices[axis];
        auto ret = inputs[0](idx);
        for (int i = 0; i < static_cast<int>(inputs.size() - 1); ++i) {
          ret = tvm::if_then_else(ind == i + 1, inputs[i + 1](idx), ret);
        }
        return ret;
      },
      name, tag);
}

/*!
 * \brief Split a tensor into multiple sub-tensors
 *
 * \param x The input tensor
 * \param split_indices The indices to split the input at. This must be in ascending
 * order.
 * \param axis The axis to split along.
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the split operation
 */
inline ffi::Array<Tensor> split_indices_array(const Tensor& x, ffi::Array<PrimExpr> split_indices,
                                              int axis, std::string name = "T_split",
                                              std::string tag = kInjective) {
  if (axis < 0) {
    axis += static_cast<int>(x->shape.size());
  }
  TVM_FFI_ICHECK_LT(axis, x->shape.size()) << "axis out of bounds";

  auto src_axis_size = x->shape[axis];
  std::vector<PrimExpr> begin_ids;
  begin_ids.push_back(0);

  for (auto idx : split_indices) {
    auto idx_node = idx.as<IntImmObj>();
    auto back_node = begin_ids.back().as<IntImmObj>();
    if (idx_node && back_node) {
      TVM_FFI_ICHECK_GT(idx_node->value, back_node->value) << "split_indices must be sorted";
    }
    begin_ids.push_back(idx);
  }

  ffi::Array<ffi::Array<PrimExpr>> out_shapes;
  for (size_t i = 0; i < begin_ids.size(); ++i) {
    PrimExpr out_axis_size =
        (i == begin_ids.size() - 1 ? src_axis_size : begin_ids[i + 1]) - begin_ids[i];

    ffi::Array<PrimExpr> shape;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
      shape.push_back(x->shape[i]);
    }
    shape.push_back(out_axis_size);
    for (size_t i = axis + 1; i < x->shape.size(); ++i) {
      shape.push_back(x->shape[i]);
    }

    out_shapes.push_back(shape);
  }

  ffi::Array<Tensor> result;
  for (size_t i = 0; i < begin_ids.size(); ++i) {
    result.push_back(compute(
        out_shapes[i],
        [&](const ffi::Array<PrimVar>& indices) {
          auto begin = begin_ids[i];
          ffi::Array<PrimExpr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(indices[j]);
          }
          real_indices.push_back(indices[axis] + begin);
          for (size_t j = axis + 1; j < indices.size(); ++j) {
            real_indices.push_back(indices[j]);
          }

          return x(real_indices);
        },
        name, tag));
  }

  return result;
}

inline PrimExpr DynamicCanonicalizeIndex(PrimExpr index, PrimExpr extent, PrimExpr stride) {
  if (index.same_as(extent)) {
    return tvm::if_then_else(stride < 0, extent - 1, extent);
  }

  PrimExpr begin_range = tvm::if_then_else(stride < 0, -1, 0);
  PrimExpr end_range = tvm::if_then_else(stride < 0, extent - 1, extent);

  if (!(index->IsInstance<tvm::IntImmObj>() && index.as_or_throw<IntImm>()->value >= 0)) {
    index = tvm::if_then_else(index < 0, index + extent, index);
  }

  return tvm::min(tvm::max(index, begin_range), end_range);
}

inline int64_t StaticCanonicalizeIndex(int64_t index, int64_t extent, int64_t stride) {
  int64_t begin_range = stride < 0 ? -1 : 0;
  int64_t end_range = stride < 0 ? extent - 1 : extent;
  if (index < 0) {
    index += extent;
  }
  return std::min(std::max(index, begin_range), end_range);
}

inline PrimExpr CanonicalizeIndex(PrimExpr index, PrimExpr extent, PrimExpr stride) {
  if (index->IsInstance<tvm::IntImmObj>() && extent->IsInstance<tvm::IntImmObj>() &&
      stride->IsInstance<tvm::IntImmObj>()) {
    return tvm::IntImm(
        tvm::PrimType::Int(64),
        StaticCanonicalizeIndex(GetConstInt(index), GetConstInt(extent), GetConstInt(stride)));
  }
  return DynamicCanonicalizeIndex(index, extent, stride);
}

inline PrimExpr GetLength(PrimExpr begin, PrimExpr end, PrimExpr stride, PrimExpr extent,
                          bool assume_inbound = true) {
  if (assume_inbound) {
    return ceildiv(end - begin, stride);
  } else {
    begin = CanonicalizeIndex(begin, extent, stride);
    end = CanonicalizeIndex(end, extent, stride);
    return tvm::if_then_else(stride < 0, ceildiv(begin - end, -stride),
                             ceildiv(end - begin, stride));
  }
}

/*!
 * \brief strided_slice of a tensor where begin/end/stride can be mixed static and dynamic
 *
 * \param x The input tensor
 * \param begin The indices to begin with in the slicing
 * \param end Indices indicating end of the slice
 * \param strides Specifies the stride values, it can be negative
 * in that case, the input tensor will be reversed in that particular axis
 * \param axes Specifies which axes will be updated.
 * \param assume_inbound Specifies if all indices are assumed to be inbound
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the dynamic_strided_slice operation
 */
inline te::Tensor dynamic_strided_slice_with_axes(
    const te::Tensor& x, const ffi::Array<PrimExpr>& begin, const ffi::Array<PrimExpr>& end,
    const ffi::Array<PrimExpr>& strides, const ffi::Array<int64_t>& axes,
    bool assume_inbound = true, std::string name = "T_dynamic_strided_slice_with_axes",
    std::string tag = kInjective) {
  const size_t src_tensor_dim = x->shape.size();
  TVM_FFI_ICHECK_EQ(begin.size(), end.size());
  TVM_FFI_ICHECK_EQ(begin.size(), strides.size());
  TVM_FFI_ICHECK_EQ(begin.size(), axes.size());
  TVM_FFI_ICHECK_LE(begin.size(), src_tensor_dim);

  for (const auto& axis_imm : axes) {
    int axis = static_cast<int>(axis_imm);
    TVM_FFI_ICHECK_LT(axis, src_tensor_dim);
  }

  sym::Analyzer analyzer;

  ffi::Array<PrimExpr> out_shape = x->shape;
  for (size_t i = 0; i < begin.size(); i++) {
    int axis = static_cast<int>(axes[i]);
    PrimExpr new_shape = analyzer->Simplify(
        GetLength(begin[i], end[i], strides[i], out_shape[axis], assume_inbound));
    out_shape.Set(axis, new_shape);
  }

  return te::compute(
      out_shape,
      [&](const ffi::Array<tvm::PrimVar>& indices) {
        ffi::Array<PrimExpr> real_indices =
            indices.Map([](const auto& var) -> PrimExpr { return var; });

        for (size_t i = 0; i < begin.size(); i++) {
          int axis = static_cast<int>(axes[i]);
          PrimExpr begin_index = begin[i];
          if (!assume_inbound) {
            begin_index = CanonicalizeIndex(begin_index, x->shape[axis], strides[i]);
          }
          PrimExpr new_index = indices[axis] * strides[i] + begin_index;
          real_indices.Set(axis, new_index);
        }

        return x(real_indices);
      },
      name, tag);
}

/*!
 * \brief strided_slice of a tensor where begin/end/stride can be mixed static and dynamic
 *
 * \param x The input tensor
 * \param begin The indices to begin with in the slicing
 * \param end Indices indicating end of the slice
 * \param strides Specifies the stride values, it can be negative
 * in that case, the input tensor will be reversed in that particular axis
 * \param assume_inbound Specifies if all indices are assumed to be inbound
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the dynamic_strided_slice operation
 */
inline Tensor dynamic_strided_slice(const Tensor& x, const ffi::Array<PrimExpr>& begin,
                                    const ffi::Array<PrimExpr>& end,
                                    const ffi::Array<PrimExpr>& strides, bool assume_inbound = true,
                                    std::string name = "T_dynamic_strided_slice",
                                    std::string tag = kInjective) {
  const size_t src_tensor_dim = x->shape.size();
  TVM_FFI_ICHECK_LE(begin.size(), src_tensor_dim);
  TVM_FFI_ICHECK_LE(end.size(), src_tensor_dim);
  TVM_FFI_ICHECK_LE(strides.size(), src_tensor_dim);
  TVM_FFI_ICHECK_EQ(begin.size(), end.size());
  TVM_FFI_ICHECK_EQ(begin.size(), strides.size());

  const size_t num_slice_axes = begin.size();
  ffi::Array<PrimExpr> out_shape;

  sym::Analyzer analyzer;
  for (size_t i = 0; i < num_slice_axes; ++i) {
    // Dynamic scalar tensor loads cannot be simplified while inferring shape.
    if (!te::IsTensorLoad(begin[i]) && !te::IsTensorLoad(end[i]) && !te::IsTensorLoad(strides[i])) {
      out_shape.push_back(
          analyzer->Simplify(GetLength(begin[i], end[i], strides[i], x->shape[i], assume_inbound)));
    } else {
      out_shape.push_back(tvm::PrimVar("dim"));
    }
  }

  for (size_t i = num_slice_axes; i < src_tensor_dim; ++i) {
    out_shape.push_back(x->shape[i]);
  }

  return te::compute(
      out_shape,
      [&](const ffi::Array<tvm::PrimVar>& indices) {
        ffi::Array<PrimExpr> real_indices;
        for (size_t i = 0; i < num_slice_axes; ++i) {
          PrimExpr begin_index = tvm::min(begin[i], x->shape[i] - 1);
          if (!assume_inbound) {
            begin_index = CanonicalizeIndex(begin[i], x->shape[i], strides[i]);
          }
          real_indices.push_back(indices[i] * strides[i] + begin_index);
        }
        // keep input dim
        for (size_t i = num_slice_axes; i < src_tensor_dim; ++i) {
          real_indices.push_back(indices[i]);
        }
        return x(real_indices);
      },
      name, tag);
}

/*!
 * \brief strided_slice of a tensor with dynamic begin/end/stride
 *
 * \param x The input tensor
 * \param begin The indices to begin with in the slicing
 * \param end Indices indicating end of the slice
 * \param strides Specifies the stride values, it can be negative
 * in that case, the input tensor will be reversed in that particular axis
 * \param assume_inbound Specifies if all indices are assumed to be inbound
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the dynamic_strided_slice operation
 */
inline te::Tensor dynamic_strided_slice(const te::Tensor& x, const te::Tensor& begin,
                                        const te::Tensor& end, const te::Tensor& strides,
                                        bool assume_inbound = true,
                                        std::string name = "T_strided_slice_dynamic",
                                        std::string tag = topi::kInjective) {
  PrimType index_ty = begin->shape[0].ty();
  const int64_t num_dynamic_axes = static_cast<int64_t>(begin->shape[0].as<IntImmObj>()->value);
  TVM_FFI_ICHECK_EQ(end->shape[0].as<IntImmObj>()->value, num_dynamic_axes);
  TVM_FFI_ICHECK_EQ(strides->shape[0].as<IntImmObj>()->value, num_dynamic_axes);

  ffi::Array<PrimExpr> begin_expr, end_expr, strides_expr;
  for (int64_t i = 0; i < num_dynamic_axes; ++i) {
    auto ind = IntImm(index_ty, i);
    begin_expr.push_back(begin(ind));
    end_expr.push_back(end(ind));
    strides_expr.push_back(strides(ind));
  }
  return dynamic_strided_slice(x, begin_expr, end_expr, strides_expr, assume_inbound, name, tag);
}

/*!
 * \brief Calculate the output shape of strided_slice, the entry point for Relax type relation
 *
 * \param ishape The input tensor shape
 * \param begin The indices to begin with in the slicing
 * \param end Indices indicating end of the slice
 * \param strides Specifies the stride values, it can be negative
 * in that case, the input tensor will be reversed in that particular axis
 * \param axes Axes along which slicing is applied. When it is specified, the length of begin, end,
 * strides, and axes argument must be equal
 * \param slice_mode Specifies the slice mode
 *
 * \return The output shape of strided_slice using the arguments above
 */
inline ffi::Array<PrimExpr> StridedSliceOutputShape(const ffi::Array<PrimExpr>& ishape,
                                                    const ffi::Array<ffi::Optional<IntImm>>& begin,
                                                    const ffi::Array<ffi::Optional<IntImm>>& end,
                                                    const ffi::Array<IntImm>& strides,
                                                    const ffi::Array<int64_t>& axes,
                                                    const std::string& slice_mode) {
  TVM_FFI_ICHECK(axes.size() == begin.size() && axes.size() == end.size() &&
                 axes.size() == strides.size());
  std::vector<int64_t> begin_vec, end_vec, strides_vec;
  std::tie(begin_vec, end_vec, strides_vec) = ConvertToVec(begin, end, strides, slice_mode);
  PrimType index_ty =
      (begin.size() > 0 && begin[0].has_value()) ? begin[0].value().ty() : PrimType::Int(64);
  auto begin_canonicalized =
      StridedSliceCanonicalizeBegin(ishape, begin_vec, strides_vec, axes, index_ty, slice_mode);
  return StridedSliceOutputShape(ishape, begin_vec, end_vec, strides_vec, axes, slice_mode,
                                 begin_canonicalized, true);
}

/*!
 * \brief strided_slice of a tensor
 *
 * \param x The input tensor
 * \param begin The indices to begin with in the slicing
 * \param end Indices indicating end of the slice
 * \param strides Specifies the stride values, it can be negative
 * in that case, the input tensor will be reversed in that particular axis
 * \param axes Axes along which slicing is applied. When it is specified, the length of begin, end,
 * strides, and axes argument must be equal
 * \param slice_mode Specifies the slice mode
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the sstrided_slice operation
 */
inline Tensor strided_slice_with_axes(
    const Tensor& x, const ffi::Array<ffi::Optional<IntImm>>& begin,
    const ffi::Array<ffi::Optional<IntImm>>& end, const ffi::Array<IntImm>& strides,
    const ffi::Array<int64_t>& axes, std::string slice_mode = "end",
    std::string name = "T_strided_slice_with_axes", std::string tag = kInjective) {
  const int64_t src_tensor_dim = static_cast<int64_t>(x->shape.size());
  TVM_FFI_ICHECK(static_cast<int64_t>(axes.size()) <= src_tensor_dim);
  TVM_FFI_ICHECK(axes.size() == begin.size() && axes.size() == end.size() &&
                 axes.size() == strides.size());

  // Normalize negative axes
  ffi::Array<int64_t> normalized_axes;
  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    if (axis < 0) {
      axis += src_tensor_dim;
    }
    TVM_FFI_ICHECK(axis >= 0 && axis < src_tensor_dim)
        << "Axis " << axes[i] << " is out of bounds for tensor with " << src_tensor_dim
        << " dimensions";
    normalized_axes.push_back(axis);
  }

  std::vector<int64_t> begin_vec, end_vec, strides_vec;
  std::tie(begin_vec, end_vec, strides_vec) = ConvertToVec(begin, end, strides, slice_mode);

  PrimType index_ty =
      (begin.size() > 0 && begin[0].has_value()) ? begin[0].value().ty() : PrimType::Int(64);
  auto begin_expr = StridedSliceCanonicalizeBegin(x->shape, begin_vec, strides_vec, normalized_axes,
                                                  index_ty, slice_mode);
  auto out_shape = StridedSliceOutputShape(x->shape, begin_vec, end_vec, strides_vec,
                                           normalized_axes, slice_mode, begin_expr);

  return te::compute(
      out_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        ffi::Array<PrimExpr> real_indices;
        for (size_t i = 0; i < out_shape.size(); ++i) real_indices.push_back(indices[i]);
        for (size_t i = 0; i < normalized_axes.size(); ++i) {
          int64_t ax = normalized_axes[i];
          auto stride = IntImm(strides[i]->ty.as_or_throw<PrimType>(), strides_vec[i]);
          PrimExpr ind = indices[ax] * stride + begin_expr[i];
          real_indices.Set(ax, ind);
        }
        return x(real_indices);
      },
      name, tag);
}

/*!
 * \brief strided_slice of a tensor
 *
 * \param x The input tensor
 * \param begin The indices to begin with in the slicing
 * \param end Indices indicating end of the slice
 * \param strides Specifies the stride values, it can be negative
 * in that case, the input tensor will be reversed in that particular axis
 * \param slice_mode Specifies the slice mode
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the strided_slice operation
 */
inline Tensor strided_slice(const Tensor& x, const ffi::Array<ffi::Optional<IntImm>>& begin,
                            const ffi::Array<ffi::Optional<IntImm>>& end,
                            const ffi::Array<IntImm>& strides, std::string slice_mode = "end",
                            std::string name = "T_strided_slice", std::string tag = kInjective) {
  using namespace tvm::prim;
  size_t src_tensor_dim = static_cast<size_t>(x->shape.size());
  ffi::Array<int64_t> axes;
  for (size_t i = 0; i < src_tensor_dim; ++i) axes.push_back(i);
  ffi::Array<ffi::Optional<IntImm>> begin_full(begin);
  ffi::Array<ffi::Optional<IntImm>> end_full(end);
  ffi::Array<IntImm> strides_full(strides);

  PrimType index_ty =
      (begin.size() > 0 && begin[0].has_value()) ? begin[0].value().ty() : PrimType::Int(64);
  const IntImm one = IntImm(index_ty, 1);
  const IntImm zero = IntImm(index_ty, 0);
  const IntImm max_range = max_value(index_ty).as_or_throw<IntImm>();

  for (size_t i = strides.size(); i < src_tensor_dim; ++i) {
    strides_full.push_back(one);
  }
  for (size_t i = begin.size(); i < src_tensor_dim; ++i) {
    begin_full.push_back(strides_full[i]->value > 0 ? zero : max_range);
  }
  for (size_t i = end.size(); i < src_tensor_dim; ++i) {
    end_full.push_back(strides_full[i]->value < 0 ? zero : max_range);
  }

  return strided_slice_with_axes(x, begin_full, end_full, strides_full, axes, slice_mode, name,
                                 tag);
}

/*!
 * \brief Split a tensor into a number of sub-tensors
 *
 * \param x The input tensor
 * \param num_sections The number of sections to split the tensor into.
 * this must be an integer factor of the size of the axis being split.
 * \param axis The axis to split along.
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the split operation
 */
inline ffi::Array<Tensor> split_n_sections(const Tensor& x, int num_sections, int axis,
                                           std::string name = "T_split_sections",
                                           std::string tag = kInjective) {
  if (axis < 0) {
    axis += static_cast<int>(x->shape.size());
  }
  TVM_FFI_ICHECK_LT(axis, x->shape.size()) << "axis out of bounds";

  auto src_axis_size = x->shape[axis];

  TVM_FFI_ICHECK_GT(num_sections, 0) << "Slice count must be > 0";

  ffi::Array<PrimExpr> split_indices;
  auto seg_size = indexdiv(src_axis_size + num_sections - 1, num_sections);
  for (int i = 0; i < num_sections; ++i) {
    // region at index 0 is added by split()
    if (i != 0) {
      split_indices.push_back(seg_size * i);
    }
  }

  return split_indices_array(x, split_indices, axis, name, tag);
}

/*!
 * \brief Take elements from an flattened input array when axis is None.
 *
 * \param a The source array.
 * \param indices The indices of the values to extract.
 * \param batch_dims The number of batch dimensions.
 * \param mode The mode of the operation.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return A Tensor whose op member is the take operation
 */
inline Tensor take(const Tensor& a, const Tensor& indices, int batch_dims,
                   std::string mode = "fast", std::string name = "T_take",
                   std::string tag = kInjective) {
  ffi::Array<PrimExpr> a_shape = a->shape;
  ffi::Array<PrimExpr> out_shape = indices->shape;
  PrimExpr a_size = 1;
  for (size_t i = 0; i < a_shape.size(); ++i) {
    a_size = a_size * a_shape[i];
  }

  if (mode == "clip") {
    return compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& out_index) {
          auto idx = tvm::min(tvm::max(0, indices(out_index)), a_size - 1);
          return a(UnravelIndex(idx, a_shape));
        },
        name, tag);
  } else if (mode == "fast") {
    LOG(WARNING) << "Fast mode segfaults when there are out-of-bounds indices. "
                    "Make sure input indices are in bound";
    return compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& out_index) {
          return a(UnravelIndex(indices(out_index), a_shape));
        },
        name, tag);
  } else if (mode == "nan") {
    return compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& out_index) {
          auto idx = tvm::if_then_else(
              indices(out_index) < 0 || indices(out_index) >= a_size,
              tvm::FloatImm(tvm::PrimType(a->dtype), std::numeric_limits<float>::quiet_NaN()),
              indices(out_index));
          return a(UnravelIndex(idx, a_shape));
        },
        name, tag);
  } else {  // mode == "wrap"
    return compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& out_index) {
          auto idx = truncmod(truncmod(indices(out_index), a_size) + a_size, a_size);
          return a(UnravelIndex(idx, a_shape));
        },
        name, tag);
  }
}

/*!
 * \brief Mask the out-of-boundary elements of each sequence.
 *
 * \param data The source array.
 * \param valid_length The real length of each sequence.
 * \param mask_value The masking value.
 * \param axis The axis of the temporal dimension of the sequence
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return A Tensor whose op member is the sequence_mask operation
 */
inline Tensor sequence_mask(const Tensor& data, const Tensor& valid_length, double mask_value,
                            int axis, std::string name = "T_sequence_mask",
                            std::string tag = kInjective) {
  TVM_FFI_ICHECK(axis == 0 || axis == 1) << "axis must be either 0 or 1";
  TVM_FFI_ICHECK_EQ(valid_length->shape.size(), 1)
      << "valid_length must have ndim=1, i.e., (batch_size,).";
  auto length_dim = data->shape[axis];
  auto batch_dim = data->shape[1 - axis];
  ffi::Array<PrimExpr> out_shape = data->shape;
  Tensor out = compute(
      out_shape,
      [&](const ffi::Array<PrimVar>& out_index) {
        ffi::Array<PrimExpr> len_index;
        auto tid = out_index[axis];
        auto bid = out_index[1 - axis];
        len_index.push_back(bid);
        PrimExpr ret = tvm::if_then_else(
            tvm::prim::cast(tid, PrimType(valid_length->dtype)) >= valid_length(len_index),
            tvm::prim::MakeConst(PrimType(data->dtype), mask_value), data(out_index));
        return ret;
      },
      name, tag);
  return out;
}

/*!
 * \brief Take elements from an array along an axis.
 *
 * \param a The source array.
 * \param indices The indices of the values to extract.
 * \param batch_dims The number of batch dimensions. By default is 0.
 * \param axis The axis over which to select values. By default,
 * the flattened input array is used.
 * \param mode The mode for handling out of bound indices.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return A Tensor whose op member is the take operation
 */
inline Tensor take(const Tensor& a, ffi::Variant<Tensor, PrimExpr> indices, int batch_dims,
                   int axis, std::string mode = "fast", std::string name = "T_take",
                   std::string tag = kInjective) {
  if (axis < 0) {
    axis += static_cast<int>(a->shape.size());
  }
  TVM_FFI_ICHECK_GE(axis, 0) << "axis out of bounds";
  TVM_FFI_ICHECK_LT(axis, a->shape.size()) << "axis out of bounds";
  auto axis_dim = a->shape[axis];
  auto indices_shape = [&]() -> ffi::Array<PrimExpr> {
    if (auto tensor = indices.as<TensorObj>()) {
      return tensor->shape;
    } else {
      return {};
    }
  }();

  int indices_len = static_cast<int>(indices_shape.size());

  int batch_dims_ = batch_dims;
  if (batch_dims_ != 0) {
    TVM_FFI_ICHECK_GE(batch_dims_, -indices_len) << "batch_dims out of bounds";
    TVM_FFI_ICHECK_LE(batch_dims_, indices_len) << "batch_dims out of bounds";

    if (batch_dims_ < 0) {
      batch_dims_ = indices_len + batch_dims_;
    }

    TVM_FFI_ICHECK_LT(batch_dims_, a->shape.size()) << "batch_dims out of bounds";
    TVM_FFI_ICHECK_LE(batch_dims_, axis) << "batch_dims must be less than or equal to axis";
    for (int i = 0; i < batch_dims_; ++i) {
      auto addr1 = a->shape[i];
      auto addr2 = indices_shape[i];
      auto v1 = static_cast<IntImm*>(&addr1)->get()->value;
      auto v2 = static_cast<IntImm*>(&addr2)->get()->value;
      TVM_FFI_ICHECK_EQ(v1, v2) << "a.shape[" << i << "] should be equal to indices.shape[" << i
                                << "]";
    }
  }

  // The result shape is a.shape[:axis] + indices.shape[batch_dims:] +
  // a.shape[axis + 1:].

  ffi::Array<PrimExpr> out_shape;
  for (int i = 0; i < batch_dims_; ++i) {
    out_shape.push_back(a->shape[i]);
  }
  for (int i = batch_dims_; i < axis; ++i) {
    out_shape.push_back(a->shape[i]);
  }
  for (int i = batch_dims_; i < indices_len; ++i) {
    out_shape.push_back(indices_shape[i]);
  }
  for (size_t i = axis + 1; i < a->shape.size(); ++i) {
    out_shape.push_back(a->shape[i]);
  }

  auto get_index = [&](const ffi::Array<PrimExpr>& indices_position) -> PrimExpr {
    if (auto tensor = indices.as<Tensor>()) {
      return tensor.value()(indices_position);
    }
    auto prim = indices.as<PrimExpr>();
    TVM_FFI_ICHECK(prim.has_value()) << "Variant did not contain either allowed type";
    TVM_FFI_ICHECK_EQ(indices_position.size(), 0);
    return prim.value();
  };

  if (mode == "clip") {
    if (batch_dims_ == 0) {
      return compute(
          out_shape,
          [&](const ffi::Array<PrimVar>& out_index) {
            ffi::Array<PrimExpr> indices_position;
            for (size_t j = axis; j < static_cast<size_t>(axis + indices_len); ++j) {
              indices_position.push_back(out_index[j]);
            }
            ffi::Array<PrimExpr> real_indices;
            for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
              real_indices.push_back(out_index[j]);
            }
            auto idx = tvm::min(tvm::max(0, get_index(indices_position)), axis_dim - 1);
            real_indices.push_back(idx);
            for (size_t j = axis + indices_len; j < out_index.size(); ++j) {
              real_indices.push_back(out_index[j]);
            }
            return a(real_indices);
          },
          name, tag);
    } else {
      return compute(
          out_shape,
          [&](const ffi::Array<PrimVar>& out_index) {
            ffi::Array<PrimExpr> indices_position;
            for (size_t j = 0; j < static_cast<size_t>(batch_dims_); ++j) {
              indices_position.push_back(out_index[j]);
            }
            for (size_t j = axis; j < static_cast<size_t>(axis + indices_len - batch_dims_); ++j) {
              indices_position.push_back(out_index[j]);
            }
            ffi::Array<PrimExpr> real_indices;
            for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
              real_indices.push_back(out_index[j]);
            }
            auto idx = tvm::min(tvm::max(0, get_index(indices_position)), axis_dim - 1);
            real_indices.push_back(idx);
            for (size_t j = axis + indices_len - batch_dims_; j < out_index.size(); ++j) {
              real_indices.push_back(out_index[j]);
            }
            return a(real_indices);
          },
          name, tag);
    }
  } else if (mode == "fast") {
    LOG(WARNING) << "Fast mode segfaults when there are out-of-bounds indices. "
                    "Make sure input indices are in bound";
    return compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& out_index) {
          ffi::Array<PrimExpr> indices_position;
          for (size_t j = axis; j < static_cast<size_t>(axis + indices_len); ++j) {
            indices_position.push_back(out_index[j]);
          }
          ffi::Array<PrimExpr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(out_index[j]);
          }
          real_indices.push_back(get_index(indices_position));
          for (size_t j = axis + indices_len; j < out_index.size(); ++j) {
            real_indices.push_back(out_index[j]);
          }
          return a(real_indices);
        },
        name, tag);
  } else if (mode == "nan") {
    return compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& out_index) {
          ffi::Array<PrimExpr> indices_position;
          for (size_t j = axis; j < static_cast<size_t>(axis + indices_len); ++j) {
            indices_position.push_back(out_index[j]);
          }
          ffi::Array<PrimExpr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(out_index[j]);
          }
          PrimExpr idx = get_index(indices_position);
          real_indices.push_back(idx);
          for (size_t j = axis + indices_len; j < out_index.size(); ++j) {
            real_indices.push_back(out_index[j]);
          }
          PrimExpr in_bounds = idx >= 0 && idx < axis_dim;
          return tvm::if_then_else(
              in_bounds, a(real_indices),
              tvm::prim::MakeConst(PrimType(a->dtype), std::numeric_limits<float>::quiet_NaN()));
        },
        name, tag);
  } else {  // mode == "wrap"
    return compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& out_index) {
          ffi::Array<PrimExpr> indices_position;
          for (size_t j = axis; j < static_cast<size_t>(axis + indices_len); ++j) {
            indices_position.push_back(out_index[j]);
          }
          ffi::Array<PrimExpr> real_indices;
          for (size_t j = 0; j < static_cast<size_t>(axis); ++j) {
            real_indices.push_back(out_index[j]);
          }
          auto idx = truncmod(truncmod(get_index(indices_position), axis_dim) + axis_dim, axis_dim);
          real_indices.push_back(idx);
          for (size_t j = axis + indices_len; j < out_index.size(); ++j) {
            real_indices.push_back(out_index[j]);
          }
          return a(real_indices);
        },
        name, tag);
  }
}

/*!
 * \brief Return the elements, either from x or y, depending on the condition.
 *
 * \param condition The condition array.
 * \param x First array to be selected.
 * \param y Second array to be selected.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return A Tensor selected from x or y depending on condition.
 */
inline Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y,
                    std::string name = "T_where", std::string tag = kBroadcast) {
  TVM_FFI_ICHECK_EQ(x->dtype, y->dtype)
      << "x and y must have the same dtype: " << x->dtype << " vs " << y->dtype;
  auto get_out_shape = [&]() {
    auto bh1 = detail::BroadcastShape(x->shape, y->shape);
    ffi::Array<PrimExpr> common_shape1(bh1.common_shape.begin(), bh1.common_shape.end());
    auto bh2 = detail::BroadcastShape(condition->shape, common_shape1);
    ffi::Array<PrimExpr> common_shape2(bh2.common_shape.begin(), bh2.common_shape.end());
    return common_shape2;
  };

  auto oshape = get_out_shape();

  auto c_bh = detail::BroadcastShape(condition->shape, oshape);
  auto x_bh = detail::BroadcastShape(x->shape, oshape);
  auto y_bh = detail::BroadcastShape(y->shape, oshape);

  auto select = [&](tvm::ffi::Array<tvm::PrimVar> ovars) {
    auto c = condition(InputIndexFromBroadcast(ovars, condition, c_bh.vars1, c_bh.all_vars));
    auto true_val = x(InputIndexFromBroadcast(ovars, x, x_bh.vars1, x_bh.all_vars));
    auto false_val = y(InputIndexFromBroadcast(ovars, y, y_bh.vars1, y_bh.all_vars));
    return tvm::prim::Select(c != 0, true_val, false_val);
  };

  return compute(oshape, select, name, tag);
}

/*!
 * \brief Creates an operation to repeat elements of an array
 *
 * \param x The input tensor
 * \param repeats The number of repetitions for each element
 * \param axis The axis along which to repeat values (allows
 * negative indices as offsets from the last dimension)
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the repeat operation
 */
inline Tensor repeat(const Tensor& x, int repeats, int axis, std::string name = "T_repeat",
                     std::string tag = kBroadcast) {
  int ndim = static_cast<int>(x->shape.size());
  TVM_FFI_ICHECK(-ndim - 1 <= axis && axis <= ndim)
      << "repeat only accepts `axis` in [-data.ndim - 1, data.ndim]"
      << ", but got axis = " << axis << ", and data.ndim = " << ndim;
  TVM_FFI_ICHECK(repeats >= 1) << "repeat only accepts `repeats >= 1`"
                               << ", but got repeats = " << repeats;
  if (axis < 0) {
    // Calculate offset from last dimension
    axis += ndim;
  }
  ffi::Array<PrimExpr> new_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
    new_shape.push_back(x->shape[i]);
  }
  new_shape.push_back(repeats * x->shape[axis]);
  for (size_t i = axis + 1; i < x->shape.size(); ++i) {
    new_shape.push_back(x->shape[i]);
  }

  return compute(
      new_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        ffi::Array<PrimExpr> idx;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
          idx.push_back(indices[i]);
        }
        idx.push_back(indexdiv(indices[axis], repeats));
        for (size_t i = axis + 1; i < indices.size(); ++i) {
          idx.push_back(indices[i]);
        }
        return x(idx);
      },
      name, tag);
}

/*!
 * \brief Creates an operation to tile elements of an array
 *
 * \param x The input tensor
 * \param reps The number of times for repeating the tensor
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the tile operation
 */
inline Tensor tile(const Tensor& x, ffi::Array<int64_t> reps, std::string name = "T_tile",
                   std::string tag = kBroadcast) {
  size_t ndim = x->shape.size();
  size_t rdim = reps.size();
  size_t tdim = (ndim > rdim) ? ndim : rdim;
  ffi::Array<PrimExpr> data_shape;
  ffi::Array<PrimExpr> reps_shape;
  ffi::Array<PrimExpr> new_shape;
  if (ndim == rdim) {
    for (size_t i = 0; i < ndim; ++i) {
      data_shape.push_back(x->shape[i]);
      reps_shape.push_back(IntImm::Int64(reps[i]));
    }
  } else if (ndim > rdim) {
    for (size_t i = 0; i < ndim; ++i) data_shape.push_back(x->shape[i]);
    for (size_t i = 0; i < (ndim - rdim); ++i) reps_shape.push_back(1);
    for (size_t i = 0; i < rdim; ++i) reps_shape.push_back(IntImm::Int64(reps[i]));
  } else {
    for (size_t i = 0; i < (rdim - ndim); ++i) data_shape.push_back(1);
    for (size_t i = 0; i < ndim; ++i) data_shape.push_back(x->shape[i]);
    for (size_t i = 0; i < rdim; ++i) reps_shape.push_back(IntImm::Int64(reps[i]));
  }
  for (size_t i = 0; i < tdim; ++i) new_shape.push_back(data_shape[i] * reps_shape[i]);

  if (is_empty_shape(new_shape)) {
    return compute(
        new_shape,
        [&](const ffi::Array<PrimVar>& indices) { return tvm::prim::cast(0, PrimType(x->dtype)); },
        name, tag);
  } else {
    return compute(
        new_shape,
        [&](const ffi::Array<PrimVar>& indices) {
          ffi::Array<PrimExpr> idx;
          if (ndim >= rdim) {
            for (size_t i = 0; i < ndim; ++i) idx.push_back(indexmod(indices[i], x->shape[i]));
          } else {
            for (size_t i = 0; i < ndim; ++i)
              idx.push_back(indexmod(indices[rdim - ndim + i], x->shape[i]));
          }
          return x(idx);
        },
        name, tag);
  }
}

/*!
 * \brief Creates an operation to tile elements of an array
 *
 * \param x The input tensor
 * \param new_shape The shape of the output after tiling
 * \param rdim The rank of the reps, provided by caller
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the tile operation
 */
inline Tensor dyn_tile(const Tensor& x, ffi::Array<PrimExpr> new_shape, size_t rdim,
                       std::string name = "T_tile", std::string tag = kBroadcast) {
  size_t ndim = x->shape.size();
  if (is_empty_shape(new_shape)) {
    return compute(
        new_shape,
        [&](const ffi::Array<PrimVar>& indices) { return tvm::prim::cast(0, PrimType(x->dtype)); },
        name, tag);
  } else {
    return compute(
        new_shape,
        [&](const ffi::Array<PrimVar>& indices) {
          ffi::Array<PrimExpr> idx;
          if (ndim >= rdim) {
            for (size_t i = 0; i < ndim; ++i) {
              idx.push_back(indexmod(indices[i], x->shape[i]));
            }
          } else {
            for (size_t i = 0; i < ndim; ++i) {
              idx.push_back(indexmod(indices[rdim - ndim + i], x->shape[i]));
            }
          }
          return x(idx);
        },
        name, tag);
  }
}

/*!
 * \brief Gather values along given axis from given indices.
 *
 * \param data The input data to the operator.
 * \param axis The axis along which to index.
 * \param indices The indices of values to gather.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return A Tensor whose op member is the gather operation
 */
inline Tensor gather(const Tensor& data, int axis, const Tensor& indices,
                     std::string name = "T_gather", std::string tag = kInjective) {
  size_t ndim_d = data->shape.size();
  size_t ndim_i = indices->shape.size();
  TVM_FFI_ICHECK_GE(ndim_d, 1) << "Cannot gather from a scalar.";
  TVM_FFI_ICHECK_EQ(ndim_d, ndim_i);
  if (axis < 0) {
    axis += ndim_d;
  }
  TVM_FFI_ICHECK_GE(axis, 0);
  TVM_FFI_ICHECK_LT(axis, ndim_d);
  if (indices->shape[axis].as<IntImmObj>()) {
    size_t indices_dim_i = static_cast<size_t>(GetConstInt(indices->shape[axis]));
    TVM_FFI_ICHECK_GE(indices_dim_i, 1);
  }
  // Index tensors are validated by integer element kind; vector lane encoding is irrelevant here.
  PrimType indices_ty = indices->dtype;
  TVM_FFI_ICHECK(indices_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt));

  ffi::Array<PrimExpr> out_shape;
  for (size_t i = 0; i < ndim_i; ++i) {
    out_shape.push_back(indices->shape[i]);
  }

  return compute(
      out_shape,
      [&](const ffi::Array<PrimVar>& out_index) {
        ffi::Array<PrimExpr> indices_position;
        for (size_t i = 0; i < ndim_i; ++i) {
          indices_position.push_back(out_index[i]);
        }
        ffi::Array<PrimExpr> real_indices;
        for (size_t i = 0; i < ndim_i; ++i) {
          if (i == static_cast<size_t>(axis)) {
            real_indices.push_back(indices(indices_position));
          } else {
            real_indices.push_back(indices_position[i]);
          }
        }
        return data(real_indices);
      },
      name, tag);
}

/*!
 * \brief Gather elements from a n-dimension array.
 *
 * \param data The source array.
 * \param indices The indices of the values to extract.
 * \param batch_dims The number of batch dimensions.
 * \param name The name of the operation.
 * \param tag The tag to mark the operation.
 *
 * \return A Tensor whose op member is the gather_nd operation
 */
inline Tensor gather_nd(const Tensor& data, const Tensor& indices, int batch_dims = 0,
                        std::string name = "T_gather_nd", std::string tag = kInjective) {
  size_t ndim_d = data->shape.size();
  size_t ndim_i = indices->shape.size();
  TVM_FFI_ICHECK_GE(ndim_i, 1) << "indices tensor must have at least 1 dimensions";
  size_t indices_dim0 = static_cast<size_t>(GetConstInt(indices->shape[0]));
  TVM_FFI_ICHECK_LE(indices_dim0, ndim_d) << "dim 0 of indices tensor must be no more "
                                          << "than dimensions of data tensor";
  ffi::Array<PrimExpr> out_shape;
  for (size_t i = 1; i < ndim_i; ++i) {
    out_shape.push_back(indices->shape[i]);
  }
  for (size_t i = indices_dim0 + batch_dims; i < ndim_d; ++i) {
    out_shape.push_back(data->shape[i]);
  }
  return compute(
      out_shape,
      [&](const ffi::Array<PrimVar>& out_index) {
        ffi::Array<PrimExpr> indices_position;
        indices_position.push_back(0);
        for (size_t i = 0; i < ndim_i - 1; ++i) {
          indices_position.push_back(out_index[i]);
        }
        ffi::Array<PrimExpr> real_indices;
        for (size_t i = 0; i < static_cast<size_t>(batch_dims); ++i) {
          real_indices.push_back(out_index[i]);
        }
        for (size_t i = 0; i < indices_dim0; ++i) {
          indices_position.Set(0, IntImm::Int32(i));
          // Index tensors are validated by integer element kind; vector lane encoding is
          // irrelevant for choosing whether an index cast is needed.
          PrimType indices_ty = indices->dtype;
          if (indices_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
            real_indices.push_back(indices(indices_position));
          } else {
            real_indices.push_back(
                tvm::prim::cast(indices(indices_position), tvm::PrimType::Int(32)));
          }
        }
        if (real_indices.size() == ndim_d) {
          return data(real_indices);
        }
        for (size_t i = ndim_i - 1; i < out_index.size(); ++i) {
          real_indices.push_back(out_index[i]);
        }
        return data(real_indices);
      },
      name, tag);
}

/*!
 * \brief Creates an operation that calculates a matrix multiplication
 *  (row-major notation):
 *      A(i, k) * B(k, j), if trans_a == trans_b
 *          the usual transposed combinations, otherwise
 *
 * \param A The matrix A
 * \param B The matrix B
 * \param trans_a Is A's layout transposed?
 * \param trans_b Is B's layout transposed?
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the matmul operation
 */
inline tvm::te::Tensor matmul(const tvm::te::Tensor& A, const tvm::te::Tensor& B,
                              bool trans_a = false, bool trans_b = false,
                              std::string name = "T_matmul", std::string tag = kMatMul) {
  tvm::ffi::Array<tvm::PrimExpr> output_shape{A->shape[trans_a ? 1 : 0], B->shape[trans_b ? 0 : 1]};
  auto k = tvm::te::reduce_axis(tvm::Range{0, A->shape[trans_a ? 0 : 1]}, "k");
  auto l = [&](tvm::PrimVar i, tvm::PrimVar j) {
    return tvm::prim::sum((trans_a ? A[k][i] : A[i][k]) * (trans_b ? B[j][k] : B[k][j]), {k});
  };
  return tvm::te::compute(output_shape, l, name, tag);
}

/*!
 * \brief A generalization of matrix multiplication to tensors.
 *
 * \param A The tensor A
 * \param B The tensor B
 * \param axes The number of the dimensions to reduce over
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor computing the result
 */
inline Tensor tensordot(const Tensor& A, const tvm::te::Tensor& B, int axes = 2,
                        std::string name = "T_tensordot", std::string tag = kMatMul) {
  using namespace tvm::prim;
  TVM_FFI_ICHECK_GE(A->shape.size(), axes);
  TVM_FFI_ICHECK_GE(B->shape.size(), axes);

  ffi::Array<PrimExpr> output_shape(A->shape.begin(), A->shape.end() + (-axes));
  for (auto it = B->shape.begin() + axes; it != B->shape.end(); ++it) output_shape.push_back(*it);

  ffi::Array<IterVar> iter_vars;
  for (int i = 0; i < axes; ++i)
    iter_vars.push_back(reduce_axis(Range(0, B->shape[i]), "k" + std::to_string(i)));

  auto func = [&A, &B, &iter_vars, axes](const ffi::Array<PrimVar>& input_indices) {
    ffi::Array<PrimExpr> A_indices;
    for (auto it = input_indices.begin(); it != input_indices.begin() + (A->shape.size() - axes);
         ++it) {
      A_indices.push_back((*it).as_or_throw<PrimExpr>());
    }
    for (auto& v : iter_vars) A_indices.push_back(v);

    ffi::Array<PrimExpr> B_indices;
    for (auto& v : iter_vars) B_indices.push_back(v);

    auto it = input_indices.begin() + (A->shape.size() - axes);
    for (; it != input_indices.end(); ++it) {
      B_indices.push_back((*it).as_or_throw<PrimExpr>());
    }

    // Some passes don't like reductions with empty axis, so avoid it here
    if (iter_vars.empty()) {
      return A(A_indices) * B(B_indices);
    } else {
      return sum(A(A_indices) * B(B_indices), iter_vars);
    }
  };

  return compute(output_shape, func, name, tag);
}

/*!
 * \brief A generalization of matrix multiplication to tensors.
 *
 * \param A The tensor A
 * \param B The tensor B
 * \param A_axes The indices of the dimensions of tensor A to reduce over
 * \param B_axes The indices of the dimensions of tensor B to reduce over
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor computing the result
 */
inline Tensor tensordot(const Tensor& A, const tvm::te::Tensor& B, ffi::Array<PrimExpr> A_axes,
                        ffi::Array<PrimExpr> B_axes, std::string name = "T_tensordot",
                        std::string tag = kMatMul) {
  using namespace tvm::prim;
  TVM_FFI_ICHECK_EQ(A_axes.size(), B_axes.size());

  auto A_axes_val = GetConstIntValues(A_axes, "A_axes");
  auto B_axes_val = GetConstIntValues(B_axes, "B_axes");

  ffi::Array<PrimExpr> output_shape;
  for (unsigned i = 0; i < A->shape.size(); ++i)
    if (std::find(A_axes_val.begin(), A_axes_val.end(), i) == A_axes_val.end())
      output_shape.push_back(A->shape[i]);
  for (unsigned i = 0; i < B->shape.size(); ++i)
    if (std::find(B_axes_val.begin(), B_axes_val.end(), i) == B_axes_val.end())
      output_shape.push_back(B->shape[i]);

  ffi::Array<IterVar> iter_vars;
  for (unsigned i = 0; i < B_axes_val.size(); ++i)
    iter_vars.push_back(reduce_axis(Range(0, B->shape[B_axes_val[i]]), "k" + std::to_string(i)));

  auto func = [&A, &B, &iter_vars, A_axes_val,
               B_axes_val](const ffi::Array<PrimVar>& input_indices) {
    int idx_input = 0;
    ffi::Array<PrimExpr> A_indices;
    for (unsigned i = 0; i < A->shape.size(); ++i) {
      auto axes_pos = std::find(A_axes_val.begin(), A_axes_val.end(), i);
      if (axes_pos == A_axes_val.end()) {
        A_indices.push_back(input_indices[idx_input++]);
      } else {
        A_indices.push_back(iter_vars[axes_pos - A_axes_val.begin()]);
      }
    }

    ffi::Array<PrimExpr> B_indices;
    for (unsigned i = 0; i < B->shape.size(); ++i) {
      auto axes_pos = std::find(B_axes_val.begin(), B_axes_val.end(), i);
      if (axes_pos == B_axes_val.end()) {
        B_indices.push_back(input_indices[idx_input++]);
      } else {
        B_indices.push_back(iter_vars[axes_pos - B_axes_val.begin()]);
      }
    }
    return sum(A(A_indices) * B(B_indices), iter_vars);
  };
  return compute(output_shape, func, name, tag);
}

inline Tensor arange(const PrimExpr& start, const PrimExpr& stop, const PrimExpr& step,
                     PrimType dtype, std::string name = "T_arange", std::string tag = kInjective) {
  sym::Analyzer analyzer;
  PrimType start_ty = start.ty();
  PrimType stop_ty = stop.ty();
  PrimType step_ty = step.ty();
  bool is_all_int = start_ty.code() == DLDataTypeCode::kDLInt &&
                    stop_ty.code() == DLDataTypeCode::kDLInt &&
                    step_ty.code() == DLDataTypeCode::kDLInt;
  PrimExpr num_elem = [&]() -> PrimExpr {
    if (is_all_int && analyzer->CanProveGreaterEqual(step, 1)) {
      // fast path for integer arange when step is positive
      return tvm::floordiv((stop - start + step - 1), step);
    } else if (is_all_int && analyzer->CanProveLess(step, 0)) {
      // fast path for integer arange when step is negative
      return tvm::floordiv((start - stop - step - 1), -step);
    } else {
      // fallback path for non-integer or step of unknown sign
      return tvm::prim::cast(
          tvm::prim::ceil(tvm::prim::cast(stop - start, tvm::PrimType::Float(32)) / step),
          DefaultIndexType());
    }
  }();
  num_elem = analyzer->Simplify(num_elem);

  return compute(
      {num_elem},
      [&](const ffi::Array<PrimVar>& indices) {
        return tvm::prim::cast(start + step * indices[0], dtype);
      },
      name, tag);
}

/*!
 * \brief Produce grids by expanding input over dimensions defined by other inputs
 *
 * \param inputs The input tensors
 * \param indexing The indexing mode, either "xy" or "ij"
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 *
 * \return A Tensor whose op member is the meshgrid operation
 */
inline ffi::Array<Tensor> meshgrid(const ffi::Array<Tensor>& inputs, const std::string& indexing,
                                   std::string name = "T_meshgrid", std::string tag = kInjective) {
  const bool cartesian_indexing = indexing == "xy" && inputs.size() >= 2;
  ffi::Array<PrimExpr> out_shape;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const int src_index = (cartesian_indexing && i < 2) ? 1 - i : i;
    out_shape.push_back(inputs[src_index]->shape.size() == 0 ? 1 : inputs[src_index]->shape[0]);
  }
  ffi::Array<Tensor> result;
  for (size_t i = 0; i < inputs.size(); ++i) {
    result.push_back(compute(
        out_shape,
        [&](const ffi::Array<PrimVar>& indices) {
          const int src_index = (cartesian_indexing && i < 2) ? 1 - i : i;
          auto ndim = inputs[i]->GetShape().size();
          ffi::Array<PrimExpr> real_indices = {};
          if (ndim > 0) {
            real_indices = {indices[src_index]};
          }
          return inputs[i](real_indices);
        },
        name, tag));
  }
  return result;
}

/*!
 * \brief Transform the layout according to \p src_layout and \p dst_layout
 * \param src the source input.
 * \param src_layout the source layout.
 * \param dst_layout the destination layout.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \param schedule_rule name of specialized schedule rule to use.
 * \return A tensor with shape in \p dst_layout
 */
inline Tensor layout_transform(const Tensor& src, const std::string& src_layout,
                               const std::string& dst_layout,
                               const std::string schedule_rule = "None",
                               const std::string name = "T_layout_trans",
                               const std::string tag = kInjective) {
  SLayout src_layout_struct(src_layout);
  SLayout dst_layout_struct(dst_layout);

  if (src_layout_struct.Equals(dst_layout_struct)) {
    return src;
  }

  TVM_FFI_ICHECK(src_layout_struct.defined() && dst_layout_struct.defined())
      << "cannot convert from/to undefined layout";

  auto layout_converter = tirx::SBijectiveLayout::Create(src_layout_struct, dst_layout_struct);
  TVM_FFI_ICHECK(layout_converter.has_value())
      << "cannot convert from " << src_layout << " to " << dst_layout;

  ffi::Array<PrimExpr> dst_shape = layout_converter.value().ForwardShape(src->shape);

  ffi::Map<ffi::String, ffi::Any> attrs = {
      {tvm::s_tir::attr::kScheduleRule.name, ffi::String(schedule_rule)},
      // Information about layouts needed for the schedule rule
      {tvm::topi::attr::kSrcLayout, ffi::String(src_layout)},
      {tvm::topi::attr::kDstLayout, ffi::String(dst_layout)},
      {tvm::topi::attr::kInputShape, src->shape}};

  return compute(
      dst_shape,
      [&](const ffi::Array<PrimVar>& dst_indices) {
        ffi::Array<PrimExpr> dst_indices_expr =
            dst_indices.Map([](const PrimVar& var) { return var.as_or_throw<PrimExpr>(); });
        ffi::Array<PrimExpr> src_indices = layout_converter.value().BackwardIndex(dst_indices_expr);
        PrimExpr in_range = PrimExpr(1) > PrimExpr(0);  // init with dtype=bool and value=true
        for (size_t i = 0; i < src.ndim(); ++i) {
          in_range = in_range && (src_indices[i] < src->shape[i]);
        }
        return if_then_else(in_range, src(src_indices),
                            tvm::prim::cast(PrimExpr(0), PrimType(src->dtype)));
      },
      name, tag, attrs);
}

/*! \brief Utility function for auto_scheduler_layout_transform */
inline void parse_auto_scheduler_layout(const ffi::String& layout, ffi::Array<PrimExpr>* shape,
                                        std::vector<std::string>* axes) {
  int32_t factor = 0;
  std::string axis = "";
  for (char c : std::string(layout)) {
    if (c >= 'A' && c <= 'z') {
      axis += c;
      if (factor != 0) {
        shape->push_back(factor);
        factor = 0;
      }
    } else if (c >= '0' && c <= '9') {
      factor = factor * 10 + c - '0';
      if (!axis.empty()) {
        axes->push_back(axis);
        axis = "";
      }
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid layout " << layout;
    }
  }
  if (!axis.empty()) {
    axes->push_back(axis);
  }
}

/*!
 * \brief Transform the auto-scheduler generated layout according to
 *        \p src_layout and \p dst_layout
 * \param src the source input.
 * \param src_layout the source layout.
 * \param dst_layout the destination layout.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return A tensor with shape in \p dst_layout
 */
inline Tensor auto_scheduler_layout_transform(
    const Tensor& src, const ffi::String& src_layout, const ffi::String& dst_layout,
    const ffi::String name = "T_auto_scheduler_layout_trans", const ffi::String tag = kInjective) {
  ffi::Array<PrimExpr> src_shape;
  std::vector<std::string> src_axes;
  ffi::Array<PrimExpr> dst_shape;
  std::vector<std::string> dst_axes;

  parse_auto_scheduler_layout(src_layout, &src_shape, &src_axes);
  parse_auto_scheduler_layout(dst_layout, &dst_shape, &dst_axes);
  return compute(
      dst_shape,
      [&](const ffi::Array<PrimVar>& dst_indices) {
        ffi::Array<PrimExpr> dst_indices_expr =
            dst_indices.Map([](const PrimVar& var) { return var.as_or_throw<PrimExpr>(); });
        ffi::Array<PrimExpr> src_indices;
        for (const std::string& src_axis : src_axes) {
          PrimExpr src_index = 0;
          TVM_FFI_ICHECK_EQ(dst_indices_expr.size(), dst_axes.size());
          for (size_t i = 0; i < dst_axes.size(); ++i) {
            if (dst_axes[i] == src_axis) {
              src_index = src_index * dst_shape[i] + dst_indices_expr[i];
            }
          }
          src_indices.push_back(src_index);
        }
        return src(src_indices);
      },
      name, tag);
}

/*!
 * \brief Transform the meta-schedule generated layout according to TIR's IndexMap
 * \param src the source input.
 * \param index_map The TIR IndexMap
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return A tensor. The layout transformation method
 * \note Example:
 *
 * For the indexing pattern below:
 *
 *  for i in range(32):
 *    for j in range(64):
 *      load A[
 *        i / 16 *  4 + j / 16,
 *        i % 16 * 16 + j % 16,
 *      ]
 *
 *  The corresponding indexing pattern in TIR is:
 *
 *    A[i, j] => A'[i / 4, j / 16, i % 4, j % 16]
 *
 *  which converts the pattern to:
 *
 *  for i in range(32):
 *    for j in range(64):
 *      load A'[
 *        i / 16 + j / 64,
 *        i % 16,
 *        j % 64 / 16,
 *        j % 16,
 *      ]
 *
 *  In this case, the transformation pattern is:
 *    A'[a, b, c, d] = A[a * 4 + c, b * 16 + d]
 */
inline Tensor meta_schedule_layout_transform(
    const Tensor& src, const tirx::IndexMap& index_map,
    const ffi::String name = "T_meta_schedule_layout_trans", const ffi::String tag = kInjective) {
  sym::Analyzer analyzer;
  ffi::Array<Range> iter_domain;
  iter_domain.reserve(src->shape.size());
  for (const PrimExpr& e : src->shape) {
    iter_domain.push_back(Range::FromMinExtent(IntImm(e.ty(), 0), e));
  }
  ffi::Array<PrimExpr> post_transform_shape = index_map->MapShape(src->shape, analyzer);
  return compute(
      post_transform_shape,
      [src, inv = index_map.Inverse(iter_domain, analyzer),
       &analyzer](const ffi::Array<PrimVar>& indices) -> PrimExpr {
        ffi::Array<PrimExpr> prim_indices =
            indices.Map([](const PrimVar& var) { return var.as_or_throw<PrimExpr>(); });
        return src(inv->MapIndices(prim_indices, analyzer));
      },
      name, tag);
}

/*!
 * \brief Get the shape of input tensor.
 * \param src the input tensor.
 * \param dtype the type of the elements in the tensor.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return Tensor of input shape.
 */
inline Tensor shape(const Tensor& src, PrimType dtype, const std::string name = "T_shape",
                    const std::string tag = kInjective) {
  int ndim = static_cast<int>(src->shape.size());
  ffi::Array<PrimExpr> out_shape{ndim};
  return compute(
      out_shape,
      [&](const ffi::Array<PrimVar>& indices) {
        auto idx = indices[0];
        PrimExpr ret = 0;
        for (int i = 0; i < ndim; ++i) {
          ret = tvm::if_then_else(idx == i, src->shape[i], ret);
        }
        return tvm::prim::cast(ret, dtype);
      },
      name, tag);
}

inline Tensor shape(const Tensor& src, DLDataType dtype, const std::string name = "T_shape",
                    const std::string tag = kInjective) {
  return shape(src, PrimType(dtype), name, tag);
}

/*!
 * \brief Get the size of input tensor.
 * \param src the input tensor.
 * \param dtype the type of the elements in the tensor.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return Tensor of input shape.
 */
inline te::Tensor tensor_size(const te::Tensor& src, PrimType dtype,
                              const std::string& name = "tensor_size",
                              const std::string& tag = kInjective) {
  int ndim = static_cast<int>(src->shape.size());
  ffi::Array<PrimExpr> out_tensor_size = {};
  return compute(
      out_tensor_size,
      [&](const ffi::Array<PrimVar>& indices) {
        PrimExpr ret = 1;
        for (int i = 0; i < ndim; ++i) {
          ret *= src->shape[i];
        }
        return tvm::prim::cast(ret, dtype);
      },
      name, tag);
}

inline te::Tensor tensor_size(const te::Tensor& src, DLDataType dtype,
                              const std::string& name = "tensor_size",
                              const std::string& tag = kInjective) {
  return tensor_size(src, PrimType(dtype), name, tag);
}

/*!
 * \brief Returns a one-hot tensor where the locations repsented by indices take value on_value,
    other locations take value off_value.
 * \param indices locations to set to on_value.
 * \param on_value value that locations represented by indices take on.
 * \param off_value value that other locations take on.
 * \param depth depth of the one-hot dimension.
 * \param axis axis to fill.
 * \param dtype data type of the output tensor.
 * \param oshape shape of the output tensor.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return one-hot tensor.
 */
inline Tensor one_hot(const Tensor& indices, const PrimExpr on_value, const PrimExpr off_value,
                      int depth, int axis, PrimType dtype,
                      ffi::Array<PrimExpr> oshape = ffi::Array<PrimExpr>(),
                      const std::string name = "T_one_hot", const std::string tag = kInjective) {
  int true_axis = (axis == -1) ? indices->shape.size() : axis;
  if (oshape.size() == 0) {
    int ndim = indices->shape.size() + 1;
    int indices_index = 0;
    for (int i = 0; i < ndim; i++) {
      if (i == true_axis) {
        oshape.push_back(IntImm::Int32(depth));
      } else {
        oshape.push_back(indices->shape[indices_index++]);
      }
    }
  }

  PrimExpr on_value_cast = prim::cast(on_value, dtype);
  PrimExpr off_value_cast = prim::cast(off_value, dtype);
  return compute(
      oshape,
      [&](const ffi::Array<PrimVar>& iter_vars) {
        ffi::Array<PrimVar> indices_indices;
        for (size_t i = 0; i < iter_vars.size(); i++) {
          if (static_cast<int>(i) == true_axis) {
            continue;
          }

          indices_indices.push_back(iter_vars[i]);
        }

        auto idx = iter_vars[true_axis];
        return prim::Select(indices(indices_indices) == idx.as_or_throw<PrimExpr>(), on_value_cast,
                            off_value_cast);
      },
      name, tag);
}

inline Tensor one_hot(const Tensor& indices, const PrimExpr on_value, const PrimExpr off_value,
                      int depth, int axis, DLDataType dtype,
                      ffi::Array<PrimExpr> oshape = ffi::Array<PrimExpr>(),
                      const std::string name = "T_one_hot", const std::string tag = kInjective) {
  return one_hot(indices, on_value, off_value, depth, axis, PrimType(dtype), std::move(oshape),
                 name, tag);
}

/*!
 * \brief Get a dense tensor.
 * \param sparse_indices sparse_indices[i] contains sparse_values[i] will be placed.
 * \param output_shape is the shape of the dense output tensor .
 * \param sparse_values is a 0-D or 1-D tensor. Values for each row of sparse_indices.
 * \param default_value is a 0-D tensor. Defaults to zero.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return Tensor of output_shape.
 */
inline Tensor sparse_to_dense(const Tensor& sparse_indices,
                              const ffi::Array<PrimExpr>& output_shape, const Tensor& sparse_values,
                              const PrimExpr& default_value,
                              const std::string name = "T_sparse_to_dense",
                              const std::string tag = kInjective) {
  // Sparse indices are validated by signed integer element kind; lane encoding is irrelevant here.
  TVM_FFI_ICHECK_EQ(sparse_indices->dtype.code(), DLDataTypeCode::kDLInt)
      << "sparse_indices only accepts integer values";
  TVM_FFI_ICHECK_LE(sparse_indices->shape.size(), 3)
      << "sparse_indices tensor should be 0D, 1D, or 2D only";
  TVM_FFI_ICHECK_LE(sparse_values->shape.size(), 2)
      << "sparse_values tensor should be 0D or 1D only";

  const auto rank_sparse_indices = static_cast<int>(sparse_indices->shape.size());
  ffi::Array<PrimExpr> oshape;
  for (auto l : output_shape) {
    oshape.push_back(l);
  }
  return compute(
      oshape,
      [&](const ffi::Array<PrimVar>& indices) {
        PrimExpr ret = default_value;
        if (0 == rank_sparse_indices) {
          ret = if_then_else(indices[0].as_or_throw<PrimExpr>() == sparse_indices(),
                             sparse_values(), ret);
        } else if (1 == rank_sparse_indices) {
          for (int j = 0; j < GetConstInt(sparse_indices->shape[0]); j++) {
            ret = if_then_else(indices[0].as_or_throw<PrimExpr>() == sparse_indices[j],
                               sparse_values[j], ret);
          }
        } else {
          for (int j = 0; j < GetConstInt(sparse_indices->shape[0]); j++) {
            PrimExpr aggregate_condition = IntImm::Bool(true);
            for (int k = 0; k < GetConstInt(sparse_indices->shape[1]); k++) {
              PrimExpr comparision = indices[k].as_or_throw<PrimExpr>() == sparse_indices[j][k];
              aggregate_condition = aggregate_condition && comparision;
            }
            ret = if_then_else(aggregate_condition, sparse_values[j], ret);
          }
        }
        return ret;
      },
      name, tag);
}

/*!
 * \brief Returns a tensor with the diagonal of input tensor replaced with the provided diagonals.
 * \param input input tensor.
 * \param diagonal values to be filled in the diagonals.
 * \param k1 lower limit (included) of the range of diagonals.
 * \param k2 upper limit (included) of the range of diagonals.
 * \param super_diag_right_align bool, true iff super-diagonal is right aligned (left-padded).
 * \param sub_diag_right_align bool, true iff sub-diagonal is right aligned (left-padded).
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return new tensor with given diagonal values.
 */
inline Tensor matrix_set_diag(const Tensor& input, const Tensor& diagonal, int k1, int k2,
                              bool super_diag_right_align, bool sub_diag_right_align,
                              const std::string name = "T_matrix_set_diag",
                              const std::string tag = kInjective) {
  size_t ndim = input->shape.size() - 1;

  bool only_one_diagonal = k1 == k2;

  return compute(
      input->shape,
      [&](const ffi::Array<PrimVar>& iter_vars) {
        auto get_diag = [&]() {
          ffi::Array<PrimExpr> diagonal_indices;
          PrimExpr k = only_one_diagonal ? PrimExpr(k1) : iter_vars[ndim] - iter_vars[ndim - 1];
          PrimExpr offset = 0;
          for (size_t i = 0; i < ndim - 1; i++) {
            diagonal_indices.push_back(iter_vars[i]);
          }
          if (!only_one_diagonal) {
            // Determining which diagonal/sub-diagonal/super-diagonal it is
            diagonal_indices.push_back(k2 - k);

            // Calculating the offset in diagonal tensor for this diagonal
            auto get_offset = [&](PrimExpr M, PrimExpr N) {
              // offset = max_diagonal_length - diagonal_length
              return diagonal->shape[diagonal->shape.size() - 1] - if_then_else(M < N, M, N);
            };
            offset = if_then_else(
                k >= 0,
                super_diag_right_align ? get_offset(input->shape[ndim] - k, input->shape[ndim - 1])
                                       : 0,
                sub_diag_right_align ? get_offset(input->shape[ndim], input->shape[ndim - 1] + k)
                                     : 0);
          }
          diagonal_indices.push_back(if_then_else(k >= 0, iter_vars[ndim - 1], iter_vars[ndim]) +
                                     offset);
          return diagonal(diagonal_indices);
        };
        return if_then_else((PrimExpr)iter_vars[ndim] - iter_vars[ndim - 1] >= k1,
                            if_then_else((PrimExpr)iter_vars[ndim] - iter_vars[ndim - 1] <= k2,
                                         get_diag(), input(iter_vars)),
                            input(iter_vars));
      },
      name, tag);
}

/*!
 * \brief Numpy style advanced indexing with tensor.
 * \param data is input data.
 * \param indices is list of indexing tensors.
 * \param name output tensor name.
 * \param tag output tensor tag.
 * \return Output tensor.
 */
inline Tensor adv_index(const Tensor& data, const ffi::Array<Tensor>& indices,
                        const std::string name = "advanced_index",
                        const std::string tag = kInjective) {
  TVM_FFI_ICHECK_LE(indices.size(), data->shape.size()) << "too many indices for data!";
  ffi::Array<PrimExpr> oshape;
  ffi::Array<PrimExpr> broadcast_shape;
  ffi::Array<Tensor> bindices;

  broadcast_shape = indices[0]->shape;
  for (size_t i = 1; i < indices.size(); ++i) {
    auto bh = detail::BroadcastShape(broadcast_shape, indices[i]->shape);
    broadcast_shape = ffi::Array<PrimExpr>(bh.common_shape.begin(), bh.common_shape.end());
  }
  if (indices.size() == 1) {
    // quick path
    bindices = indices;
  } else {
    // Do broadcast for indices
    for (size_t i = 0; i < indices.size(); ++i) {
      bindices.push_back(broadcast_to(indices[i], broadcast_shape));
    }
  }

  for (const auto& dim : broadcast_shape) {
    oshape.push_back(dim);
  }
  for (size_t i = indices.size(); i < data->shape.size(); ++i) {
    oshape.push_back(data->shape[i]);
  }

  return compute(
      oshape,
      [&](const ffi::Array<PrimVar>& iter_var) {
        ffi::Array<PrimExpr> tensor_indices;
        for (size_t i = 0; i < broadcast_shape.size(); ++i) {
          tensor_indices.push_back(iter_var[i]);
        }
        ffi::Array<PrimExpr> real_indices;
        for (size_t i = 0; i < bindices.size(); ++i) {
          real_indices.push_back(bindices[i](tensor_indices));
        }
        for (size_t i = broadcast_shape.size(); i < iter_var.size(); ++i) {
          real_indices.push_back(iter_var[i]);
        }

        return data(real_indices);
      },
      name, tag);
}

namespace relax {
// relax dynamic slice
inline te::Tensor dynamic_strided_slice(const te::Tensor& x, const te::Tensor& begin,
                                        const te::Tensor& end, const te::Tensor& strides,
                                        ffi::Array<PrimExpr> output_shape,
                                        std::string name = "T_strided_slice_dynamic",
                                        std::string tag = kInjective) {
  const size_t num_dynamic_axes = x.ndim();
  TVM_FFI_ICHECK_EQ(begin.ndim(), 1);
  TVM_FFI_ICHECK_EQ(end.ndim(), 1);
  TVM_FFI_ICHECK_EQ(strides.ndim(), 1);
  const auto* len_begin = begin->shape[0].as<IntImmObj>();
  const auto* len_end = end->shape[0].as<IntImmObj>();
  const auto* len_strides = strides->shape[0].as<IntImmObj>();
  TVM_FFI_ICHECK(len_begin);
  TVM_FFI_ICHECK(len_end);
  TVM_FFI_ICHECK(len_strides);
  TVM_FFI_ICHECK_EQ(len_begin->value, num_dynamic_axes);
  TVM_FFI_ICHECK_EQ(len_end->value, num_dynamic_axes);
  TVM_FFI_ICHECK_EQ(len_strides->value, num_dynamic_axes);

  return te::compute(
      output_shape,
      [&](const ffi::Array<tvm::PrimVar>& indices) {
        ffi::Array<PrimExpr> real_indices;
        for (size_t i = 0; i < num_dynamic_axes; ++i) {
          auto ind = IntImm::Int64(i);
          PrimExpr begin_index = CanonicalizeIndex(begin(ind), x->shape[i], strides(ind));
          real_indices.push_back(indices[i] * strides(ind) + begin_index);
        }
        return x(real_indices);
      },
      name, tag);
}

}  // namespace relax

}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_TRANSFORM_H_
