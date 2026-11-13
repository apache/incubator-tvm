# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""ScatterElements operator"""

from tvm import te
from tvm_tirx import tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as T

from . import utils
from .math import cast


def scatter_elements(data, indices, updates, axis=0, reduction="update"):
    """Scatter elements from updates to corresponding indices of copied data.

    Data, indices, updates and output have the same shape.
    Indices can not have duplicates (if idx1 != idx2, then indices[idx1] != indices[idx2])
    if reduction == "update".

    .. code-block::

        output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0
        output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1

    where the update function f is determined by the reduction.
    Five types of the function are supported: "update", "add", "mul", "min" and "max" (see below)

    Parameters
    ----------
    data : tvm.te.Tensor
        The source array.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    updates : tvm.te.Tensor
        The updates to apply at the Indices

    axis : optional, int
        The axis to scatter on. It is zero by default.

    reduction : optional, string
        The update mode for the algorithm, either "update", "add", "mul", "min" or "max"
        If update, the update values will replace the input data
        If add, the update values will be added to the input data
        If mul, the input data will be multiplied on the update values
        If mean, the input data will be mean between the update values and the input data
        If min, there is choice of minimal between the update values and the input data
        If max, there is choice of maximal between the update values and the input data
        It is "update" by default

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if not isinstance(axis, int):
        axis = utils.get_const_int(axis)

    # Prepare ranges and strides
    shape = data.shape
    if axis < 0:
        axis = len(shape) + axis
    axis_range = cast(shape[axis], indices.dtype)

    full_range = 1
    after_axis_range = 1
    for i, value in enumerate(shape, 0):
        full_range *= value
        if i > axis:
            after_axis_range *= value
    before_axis_stride = axis_range * after_axis_range

    ind_shape = indices.shape
    ind_axis_range = ind_shape[axis]

    ind_before_axis_range = 1
    ind_after_axis_range = 1
    for i, value in enumerate(ind_shape, 0):
        if i < axis:
            ind_before_axis_range *= value
        elif i > axis:
            ind_after_axis_range *= value
    ind_before_axis_stride = ind_axis_range * ind_after_axis_range

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr, reduce_func):
        # pylint: disable=invalid-name
        data = data_ptr
        indices = indices_ptr
        updates = updates_ptr
        out = out_ptr

        # Copy initial input data to output
        with IRBuilder() as ib:
            with T.parallel(0, full_range) as i:
                T.tensor_store(out, T.tensor_indices(out, i), data[T.tensor_indices(data, i)])

            with T.parallel(0, ind_before_axis_range * ind_after_axis_range) as fused:
                i = fused // ind_after_axis_range
                j = fused % ind_after_axis_range
                pre_index1 = i * ind_before_axis_stride + j
                pre_index2 = i * before_axis_stride + j
                with T.serial(0, ind_axis_range) as k:
                    # Offset along indices or updates
                    index1 = pre_index1 + k * ind_after_axis_range
                    # Get index and shift to positive side if need
                    k_new = indices[T.tensor_indices(indices, index1)]
                    shifted_index = k_new + (k_new < 0) * axis_range
                    # Offset along data
                    index2 = pre_index2 + shifted_index * after_axis_range
                    reduce_func(out, index2, updates[T.tensor_indices(updates, index1)])

            return ib.get()

    def update_func(dst_ptr, dst_index, update):
        T.tensor_store(dst_ptr, T.tensor_indices(dst_ptr, dst_index), update)

    def add_func(dst_ptr, dst_index, update):
        T.tensor_store(
            dst_ptr,
            T.tensor_indices(dst_ptr, dst_index),
            dst_ptr[T.tensor_indices(dst_ptr, dst_index)] + (update),
        )

    def mul_func(dst_ptr, dst_index, update):
        T.tensor_store(
            dst_ptr,
            T.tensor_indices(dst_ptr, dst_index),
            dst_ptr[T.tensor_indices(dst_ptr, dst_index)] * (update),
        )

    def mean_func(dst_ptr, dst_index, update):
        T.tensor_store(
            dst_ptr,
            T.tensor_indices(dst_ptr, dst_index),
            (dst_ptr[T.tensor_indices(dst_ptr, dst_index)] + update) / 2,
        )

    def min_func(dst_ptr, dst_index, update):
        T.tensor_store(
            dst_ptr,
            T.tensor_indices(dst_ptr, dst_index),
            tirx.min(dst_ptr[T.tensor_indices(dst_ptr, dst_index)], update),
        )

    def max_func(dst_ptr, dst_index, update):
        T.tensor_store(
            dst_ptr,
            T.tensor_indices(dst_ptr, dst_index),
            tirx.max(dst_ptr[T.tensor_indices(dst_ptr, dst_index)], update),
        )

    reduce_func = None
    if reduction == "update":
        reduce_func = update_func
    elif reduction == "add":
        reduce_func = add_func
    elif reduction == "mul":
        reduce_func = mul_func
    elif reduction == "mean":
        reduce_func = mean_func
    elif reduction == "min":
        reduce_func = min_func
    elif reduction == "max":
        reduce_func = max_func
    else:
        raise NotImplementedError(
            "scatter_elements reduction not in [update, add, mul, mean, min, max]:", reduction
        )

    out_buf = tirx.Var("out_buf", tirx.TensorType(data.shape, data.dtype, layout=None))
    return te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0], reduce_func),
        dtype=data.dtype,
        out_tensors=[out_buf],
        name="scatter_elements.generic",
        tag="scatter_elements.generic",
    )
