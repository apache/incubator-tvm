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
 * \file manipulate.h
 * \brief The functions to make Relax tensor manipulation operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_MANIPULATE_H_
#define TVM_RELAX_OP_TENSOR_MANIPULATE_H_

#include <tvm/ffi/container/variant.h>
#include <tvm/relax/op/manipulate.h>

#include "../op_common.h"
#include "tvm/relax/expr.h"

namespace tvm {
namespace relax {

/*! \brief Broadcasts a tensor to a specified shape. */
Expr broadcast_to(Expr x, Expr shape);

/*!
 * \brief Concatenate the input tensors along the given axis.
 * \param tensors An Expr in Tuple type, containing the tensors to be concatenated,
 * or a list of tensors
 * \param axis The axis along which the tensors are concatenated.
 * If it is `std::nullopt`, the input tensor is required to be flattened before concatenation.
 * \return The concatenated tensor.
 */
Expr concat(Expr tensors, ffi::Optional<int64_t> axis);

/*!
 * \brief Insert new axes at the positions given by `axis`.
 * \param x The input data to the operator.
 * \param axis The axes at which the input array are expanded.
 * \return The transformed result.
 */
Expr expand_dims(Expr x, ffi::Array<int64_t> axis);

/*!
 * \brief Flatten all the tensor dimensions into one.
 * \param x The input data to the operator.
 * \return The flattened result.
 */
Expr flatten(Expr x);

/*!
 * \brief Transform layout of a tensor.
 * \param x The input data to the operator.
 * \param index_map The transformation to apply.
 * \param pad_value The value used for padding if the transformation results in implicit padding. If
 * not specified, any value can be used.
 * \return The transformed result.
 */
Expr layout_transform(Expr x, tirx::IndexMap index_map, ffi::Optional<PrimExpr> pad_value);

/*!
 * \brief Permutes the dimensions of an array.
 * \param x The input data to the operator.
 * \param axes The target axes order, reverse order if not specified.
 * \return The transposed result.
 */
Expr permute_dims(Expr x, ffi::Optional<ffi::Array<int64_t>> axes);

/*!
 * \brief Reshape the input array, supporting `-1` inference in the new
 * shape when the new shape is given as an Array of PrimExpr.
 * \param x The input data to the operator.
 * \param shape The new shape. Should be compatible with the original shape.
 * It is required to be either an Array of PrimExpr, or a Shape in Relax
 * \return The reshaped result.
 */
Expr reshape(Expr x, ffi::Variant<ffi::Array<PrimExpr>, Expr> shape);

/*!
 * \brief Split input tensor along axis by sections or indices.
 * - If indices_or_sections is an integer, the input will be divided equally
 * along given axis (if possible). Last section will be smaller if the tensor
 * size along the given dimension is not divisible by the integer.
 * - If indices_or_sections is a tuple of mixture of int or PrimExpr,
 * the entries indicate the indices where along axis the array is split.
 * \param x The tensor to be split.
 * \param indices_or_sections Indices or sections to split into.
 * It is required to be an Array of PrimExpr or an integer.
 * \param axis The axis over which to split.
 * \return The computed result.
 */
Expr split(Expr x, ffi::Variant<IntImm, ffi::Array<IntImm>> indices_or_sections, int axis);

/*!
 * \brief Squeeze axes in the array.
 * \param x The input data to the operator.
 * \param axis The set of axes to remove.
 * If it is `std::nullopt`, remove all axis of dimensions 1.
 * If any specified axis has dimension that does not equal 1, it is an error.
 * \return The squeezed result.
 */
Expr squeeze(Expr x, ffi::Optional<ffi::Array<int64_t>> axis);
/*!
 * \brief Stack tensors along the specified axis.
 * \param tensors The input tensors to be stacked.
 * \param axis The axis along which the tensors will be stacked.
 * \return The stacked result.
 */
Expr stack(Expr tensors, ffi::Optional<int64_t> axis);
/*!
 * \brief Return a summation of data to the shape of collapse_target.
 * For details, please see the operator `relax.collapse_sum_to`.
 * \param data The input tensor.
 * \param collapse_target The tensor whose shape is the shape to collapse to.
 * \return The result tensor after summation.
 */
Expr collapse_sum_like(Expr data, Expr collapse_target);

/*!
 * \brief Return a summation of data to the given shape.
 * collapse_sum_to is intended as the backward operator of broadcast_to and
 * other broadcast operators in the automatic differentiation process.
 * We expect that data is the result of broadcasting some tensor of the given shape in some
 * broadcast operation. Thus the given shape and data.shape must follow broadcast rules.
 * \param data The input tensor.
 * \param shape The shape to collapse to.
 * \return The result tensor of the given shape after summation.
 */
Expr collapse_sum_to(Expr data, Expr shape);

/*!
 * \brief Repeats elements of an array.
 * \param data The input tensor.
 * \param repeats The number of repetitions.
 * \param axis The axis along which to repeat values. The negative numbers are interpreted counting
 * from the backward. By default, use the flattened input array, and return a flat output array.
 * \return The computed result.
 */
Expr repeat(Expr data, int repeats, ffi::Optional<int64_t> axis = std::nullopt);

/*!
 * \brief Construct an array by repeating data the number of times given by reps.
 *
 * If reps has length l, and data has dimension d, the result will have dimension of max(l, d).
 *
 * If d < l, data is promoted to be l-dimensional by prepending new axes. So a shape (3,) Tensor is
 * promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not
 * the desired behavior, promote data to d-dimensions manually before calling this function.
 *
 * If d > l, reps is promoted to length d by pre-pending 1's to it. Thus for a data of shape
 * (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2).
 * \param data The input tensor.
 * \param repeats The number of repetitions of data along each axis.
 * \return The computed result.
 */
Expr tile(Expr data, ffi::Array<int64_t> repeats);

/*!
 * \brief Reverses the order of elements along given axis.
 * \param data The input tensor.
 * \param axis The axis to flip on
 * \return The computed result.
 */
Expr flip(Expr data, int64_t axis);

/*!
 * \brief Reverses variable length slices along seq_axis.
 * \param data The input tensor.
 * \param seq_lengths A 1-D tensor containing sequence lengths for each batch.
 * \param seq_axis The axis along which to reverse.
 * \param batch_axis The axis that indexes the batch.
 * \return The computed result.
 */
Expr reverse_sequence(Expr data, Expr seq_lengths, int64_t seq_axis, int64_t batch_axis);

/*!
 * \brief Gather elements from a tensor using indices.
 * \param data The input tensor.
 * \param indices The indices tensor, must have integer type.
 * \param axis The axis along which to index. Default is 0.
 * \return The computed result.
 *
 * \note The shape of indices must match the shape of data, except at dimension axis
 *       where it must just be not null. The output will have the same shape as indices.
 */
Expr gather_elements(Expr data, Expr indices, int axis = 0);

/*!
 * \brief Gather values from a tensor using N-dimensional indices.
 * \param data The input tensor.
 * \param indices The indices tensor, must have integer type.
 * \param batch_dims The number of batch dimensions. Default is 0.
 * \return The computed result.
 *
 * \note For batch_dims > 0, the first batch_dims dimensions of data and indices must be equal.
 *       The last dimension of indices indicates the depth of each index vector.
 *       The output shape is batch_dims + indices.shape[:-1] + data.shape[batch_dims +
 * indices.shape[-1]:]
 */
Expr gather_nd(Expr data, Expr indices, int batch_dims = 0);

/*!
 * \brief NumPy/PyTorch‑style advanced indexing with tensors.
 * \param data The input tensor.
 * \param indices  A Tuple expression (or list) containing the index tensors.
 * \return The indexed tensor.
 *
 * \note When all shapes are static, Relax checks that the index shapes are
 *       broadcast-compatible. Bounds checking of the values in indices is
 *       deferred to runtime.
 */
Expr index_tensor(Expr data, Expr indices);

/*!
 * \brief Put values into an array according to indices.
 * \param data The input tensor to be modified.
 * \param indices The index positions where values should be placed.
 *                This should be a tuple of 1D tensors (one for each dimension).
 * \param values The values to place at the specified indices.
 * \param accumulate Whether to accumulate (add) values rather than replace.
 *                  If true, equivalent to tensor[indices] += values.
 *                  If false, equivalent to tensor[indices] = values.
 * \return The computed result with values placed at specified indices.
 */
Expr index_put(Expr data, Expr indices, Expr values, bool accumulate = false);

/*!
 * \brief Generate coordinate grids from input 1D tensors.
 * \param tensors A tuple of 1D tensors representing coordinate vectors.
 * \param indexing Indexing mode, either "ij" (matrix indexing) or "xy" (Cartesian indexing).
 * \return A tuple of tensors representing the coordinate grids.
 */
Expr meshgrid(Expr tensors, ffi::Optional<ffi::String> indexing = ffi::String("ij"));

/*!
 * \brief Scatter updates into an array according to indices.
 * \param data The input tensor.
 * \param indices The index positions to update in `data`.
 * \param updates The values to replace to.
 * \param axis The axis along which to scatter the elements.
 * \param reduction The reduction mode of the scatter elements,
 * either "update", "add", "mul", "mean", "max" or "min".
 * \return The computed result.
 */
Expr scatter_elements(Expr data, Expr indices, Expr updates, int axis, ffi::String reduction);

/*!
 * \brief Scatter updates into an array according to indices.
 * \param data The input tensor to be updated.
 * \param indices The index positions to update in `data`.
 * \param updates The values to replace to.
 * \param reduction The reduction mode of the scatter operation.
 *        Supported modes are:
 *        - "update": Replace the values at the indices with the update values.
 *        - "add": Add the update values to the existing values at the indices.
 *        - "mul": Multiply the existing values at the indices by the update values.
 *        - "max": Take the maximum of the existing value and the update value at each index.
 *        - "min": Take the minimum of the existing value and the update value at each index.
 * \return The computed result tensor with the same shape as `data`.
 *
 * \note The shape of `indices` defines the shape of the scattered tensor.
 *       The last dimension of `indices` corresponds to the depth of each index vector.
 *       The shape of `updates` must match the shape of `indices` except for the last dimension,
 *       which must match the slice shape at each index.
 */
Expr scatter_nd(Expr data, Expr indices, Expr updates, ffi::String reduction);

/*!
 * \brief Embeds the values of the src tensor into input at the given dimension.
 * \param input The input tensor to be updated.
 * \param src The tensor to embed into input.
 * \param dim The dimension to insert the slice into.
 * \param start The start index of where to insert the slice.
 * \param end The end index of where to insert the slice.
 * \param step The how many elements to skip in
 * \return  The computed result tensor with the same shape as `data`.
 */
Expr slice_scatter(Expr input, Expr src, int axis, PrimExpr start, PrimExpr end, PrimExpr step);

/*!
 * \brief Returns a one-hot tensor.
 * \param indices The indices to set to `on_value`.
 * \param on_value The value to fill at `indices`.
 * \param off_value The value to fill at other locations.
 * \param depth The depth of the one hot dimension.
 * \param axis The axis to fill.
 * \return The computed result.
 */
Expr one_hot(Expr indices, PrimExpr on_value, PrimExpr off_value, int depth, int axis);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_MANIPULATE_H_
