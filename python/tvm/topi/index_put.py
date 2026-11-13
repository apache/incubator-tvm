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
# pylint: disable=invalid-name
"""IndexPut operator"""

from tvm import te
from tvm_tirx import tirx
from tvm_tirx.script.ir_builder import IRBuilder
from tvm_tirx.tirx.script import ir_builder as T

from . import utils


def index_put(data, indices, values, accumulate=False):
    """Put values into an array according to indices.

    Parameters
    ----------
    data : tvm.te.Tensor
        The source array to be modified.

    indices : Tuple[tvm.te.Tensor]
        Tuple of index tensors (can be multi-dimensional) specifying positions.
        Index tensors are broadcast together following NumPy broadcasting rules.

    values : tvm.te.Tensor
        The values to place at the specified indices.

    accumulate : bool, optional
        Whether to accumulate (add) values rather than replace.
        If True, performs tensor[indices] += values
        If False, performs tensor[indices] = values
        Default is False.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if not isinstance(indices, list | tuple):
        indices = [indices]

    # Check indices match data dimensions
    if len(indices) != len(data.shape):
        raise ValueError(
            f"Number of index tensors ({len(indices)}) must match "
            f"data dimensions ({len(data.shape)})"
        )

    # Prepare ranges and strides
    shape = data.shape
    full_range = 1
    for dim in shape:
        full_range *= dim

    index_shapes = [idx.shape for idx in indices]
    broadcast_ndim = max(len(s) for s in index_shapes)
    broadcast_shape = []

    for i in range(broadcast_ndim):
        max_dim = 1
        for idx_shape in index_shapes:
            # Right-align shapes
            dim_idx = len(idx_shape) - broadcast_ndim + i
            if dim_idx >= 0:
                dim_size = idx_shape[dim_idx]
                if not utils.equal_const_int(dim_size, 1):
                    if utils.equal_const_int(max_dim, 1):
                        max_dim = dim_size
                    elif not utils.equal_const_int(dim_size, max_dim):
                        raise ValueError(f"Cannot broadcast index shapes: {index_shapes}")
        broadcast_shape.append(max_dim)

    # Compute total number of elements after broadcasting
    index_len = 1
    for dim in broadcast_shape:
        index_len *= dim

    def gen_ir(data_ptr, index_ptrs, values_ptr, out_ptr, reduce_func):
        data = data_ptr
        indices = index_ptrs
        values = values_ptr
        out = out_ptr

        with IRBuilder() as ib:
            with T.parallel(0, full_range) as i:
                T.tensor_store(out, T.tensor_indices(out, i), data[T.tensor_indices(data, i)])

            with T.parallel(0, index_len) as k:
                # Decompose k into multi-dimensional broadcast index
                k_temp = k
                broadcast_indices = []
                for i in range(broadcast_ndim - 1, -1, -1):
                    broadcast_indices.insert(0, k_temp % broadcast_shape[i])
                    k_temp = k_temp // broadcast_shape[i]

                flat_index = 0
                stride = 1
                for dim in range(len(shape) - 1, -1, -1):
                    # Get the index for this dimension using broadcasting
                    idx_shape = index_shapes[dim]
                    idx_ndim = len(idx_shape)

                    # Compute the linear index into this index tensor
                    idx_offset = 0
                    idx_stride = 1
                    for i in range(broadcast_ndim - 1, -1, -1):
                        # Right-align the index shape with broadcast shape
                        dim_idx = idx_ndim - broadcast_ndim + i
                        if dim_idx >= 0:
                            dim_size = idx_shape[dim_idx]
                            # Use broadcasting: if size is 1, use index 0
                            # otherwise use broadcast_indices[i]
                            if utils.equal_const_int(dim_size, 1):
                                idx_in_dim = 0
                            else:
                                idx_in_dim = broadcast_indices[i]
                            idx_offset += idx_in_dim * idx_stride
                            idx_stride *= dim_size

                    idx_val = indices[dim][T.tensor_indices(indices[dim], (idx_offset))]
                    shifted_idx = idx_val + (idx_val < 0) * shape[dim]
                    flat_index += shifted_idx * stride
                    stride *= shape[dim]

                reduce_func(out, flat_index, values[T.tensor_indices(values, k)])

            return ib.get()

    def update_func(dst_ptr, dst_index, update):
        T.tensor_store(dst_ptr, T.tensor_indices(dst_ptr, dst_index), update)

    def add_func(dst_ptr, dst_index, update):
        T.tensor_store(
            dst_ptr,
            T.tensor_indices(dst_ptr, dst_index),
            dst_ptr[T.tensor_indices(dst_ptr, dst_index)] + (update),
        )

    reduce_func = add_func if accumulate else update_func

    # Prepare input buffers
    in_tensors = [data]
    in_tensors.extend(indices)
    in_tensors.append(values)

    out_buf = tirx.Var("out_buf", tirx.TensorType(data.shape, data.dtype, layout=None))
    return te.extern(
        [data.shape],
        in_tensors,
        lambda ins, outs: gen_ir(ins[0], ins[1:-1], ins[-1], outs[0], reduce_func),
        dtype=data.dtype,
        out_tensors=[out_buf],
        name="index_put.generic",
        tag="index_put.generic",
    )
