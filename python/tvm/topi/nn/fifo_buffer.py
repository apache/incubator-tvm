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
# ruff: noqa: E741

"""FIFO buffer op"""

import tvm
import tvm_tirx
from tvm import te

from .. import tag
from ..transform import concatenate, strided_slice


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",fifo_buffer")
def fifo_buffer(data, buffer, axis):
    """
    FIFO buffer to enable computation reuse in CNNs with sliding indow input

    Compute equivalent of

    .. code-block:: python

        concat(buffer, data, axis=axis)
        .slice_axis(axis=axis,
                    begin=data.shape[axis],
                    end=data.shape[axis]+buffer.shape[axis])

    Useful for

    * Encoding explicit re-use of computation in convolution ops operated on a sliding window input
    * Implementing a FIFO queue to cache intermediate results, e.g. as in Fast WaveNet.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data
    buffer : tvm.te.Tensor
        Previous value of the FIFO buffer
    axis : int
        Specify which axis should be used for buffering

    Returns
    -------
    result : tvm.te.Tensor
        Updated value for the buffer
    """
    assert len(data.shape) == len(buffer.shape), (
        f"buffer and data must have same number of dimensions, "
        f"buffer.shape = {buffer.shape}, data.shape = {data.shape}"
    )
    assert len(buffer.shape) >= 1, "Zero-dimension tensor not supported"
    assert 0 <= axis < len(buffer.shape), "buffer axis out of range"
    for i in range(len(data.shape)):
        if i == axis:
            assert int(str(data.shape[i])) <= int(str(buffer.shape[i]))
        else:
            assert int(str(data.shape[i])) == int(str(buffer.shape[i]))

    buflen = buffer.shape[axis]
    data_size = data.shape[axis]

    # Explicitly write out formula up to 4D, and then use concat+slice combo for 5D and higher
    if len(buffer.shape) == 1:
        return te.compute(
            buffer.shape,
            lambda i: tvm_tirx.tirx.if_then_else(
                i < buflen - data_size, buffer[i + data_size], data[i - buflen + data_size]
            ),
            name="new_buffer",
        )
    if len(buffer.shape) == 2:
        if axis == 0:
            return te.compute(
                buffer.shape,
                lambda i, j: tvm_tirx.tirx.if_then_else(
                    i < buflen - data_size,
                    buffer[i + data_size, j],
                    data[i - buflen + data_size, j],
                ),
                name="new_buffer",
            )
        if axis == 1:
            return te.compute(
                buffer.shape,
                lambda i, j: tvm_tirx.tirx.if_then_else(
                    j < buflen - data_size,
                    buffer[i, j + data_size],
                    data[i, j - buflen + data_size],
                ),
                name="new_buffer",
            )
        assert False, f"Invalid value for axis; it should be at most {len(buffer.shape)}"
    elif len(buffer.shape) == 3:
        if axis == 0:
            return te.compute(
                buffer.shape,
                lambda i, j, k: tvm_tirx.tirx.if_then_else(
                    i < buflen - data_size,
                    buffer[i + data_size, j, k],
                    data[i - buflen + data_size, j, k],
                ),
                name="new_buffer",
            )
        if axis == 1:
            return te.compute(
                buffer.shape,
                lambda i, j, k: tvm_tirx.tirx.if_then_else(
                    j < buflen - data_size,
                    buffer[i, j + data_size, k],
                    data[i, j - buflen + data_size, k],
                ),
                name="new_buffer",
            )
        if axis == 2:
            return te.compute(
                buffer.shape,
                lambda i, j, k: tvm_tirx.tirx.if_then_else(
                    k < buflen - data_size,
                    buffer[i, j, k + data_size],
                    data[i, j, k - buflen + data_size],
                ),
                name="new_buffer",
            )
        assert False, f"Invalid value for axis; it should be at most {len(buffer.shape)}"
    elif len(buffer.shape) == 4:
        if axis == 0:
            return te.compute(
                buffer.shape,
                lambda i, j, k, l: tvm_tirx.tirx.if_then_else(
                    i < buflen - data_size,
                    buffer[i + data_size, j, k, l],
                    data[i - buflen + data_size, j, k, l],
                ),
                name="new_buffer",
            )
        if axis == 1:
            return te.compute(
                buffer.shape,
                lambda i, j, k, l: tvm_tirx.tirx.if_then_else(
                    j < buflen - data_size,
                    buffer[i, j + data_size, k, l],
                    data[i, j - buflen + data_size, k, l],
                ),
                name="new_buffer",
            )
        if axis == 2:
            return te.compute(
                buffer.shape,
                lambda i, j, k, l: tvm_tirx.tirx.if_then_else(
                    k < buflen - data_size,
                    buffer[i, j, k + data_size, l],
                    data[i, j, k - buflen + data_size, l],
                ),
                name="new_buffer",
            )
        if axis == 3:
            return te.compute(
                buffer.shape,
                lambda i, j, k, l: tvm_tirx.tirx.if_then_else(
                    l < buflen - data_size,
                    buffer[i, j, k, l + data_size],
                    data[i, j, k, l - buflen + data_size],
                ),
                name="new_buffer",
            )
        assert False, f"Invalid value for axis; it should be at most {len(buffer.shape)}"
    else:
        # Implement FIFO buffer as combination of concat and slice
        begin = [0] * len(buffer.shape)
        begin[axis] = data.shape[axis]
        end = list(buffer.shape[:])
        end[axis] += data.shape[axis]
        return strided_slice(concatenate((buffer, data), axis=axis), begin=begin, end=end)
    return None
