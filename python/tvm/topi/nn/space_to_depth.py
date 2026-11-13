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
"""TVM operator space_to_depth compute."""

import tvm_tirx
from tvm import te

from .. import tag


def space_to_depth(data, block_size, layout="NCHW"):
    """Perform space to depth transformation on the data

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D tensor in either NCHW or NHWC layout.

    block_size : int
        Size of blocks to decompose into channel dimension.

    layout : string
        Either NCHW or NHWC, indicating data layout.

    Returns
    -------
    output : tvm.te.Tensor
        Output of shape [N, C * block_size**2, H / block_size, W / block_size]
    """

    if layout == "NCHW":
        in_n, in_c, in_h, in_w = data.shape
        output_shape = [
            in_n,
            in_c * block_size * block_size,
            tvm_tirx.tirx.truncdiv(in_h, block_size),
            tvm_tirx.tirx.truncdiv(in_w, block_size),
        ]
    elif layout == "NHWC":
        in_n, in_h, in_w, in_c = data.shape
        output_shape = [
            in_n,
            tvm_tirx.tirx.truncdiv(in_h, block_size),
            tvm_tirx.tirx.truncdiv(in_w, block_size),
            in_c * block_size * block_size,
        ]
    else:
        raise ValueError("Only NCHW and NHWC layouts are currently supported.")

    def _get_indices(*indices):
        if layout == "NCHW":
            n, c, y, x = indices
        elif layout == "NHWC":
            n, y, x, c = indices
        return n, c, y, x

    def _get_pixel(n, c, y, x):
        block_offset = tvm_tirx.tirx.truncdiv(c, in_c)
        channel_idx = tvm_tirx.tirx.truncmod(c, in_c)
        x_idx = tvm_tirx.tirx.truncmod(block_offset, block_size)
        y_idx = tvm_tirx.tirx.truncdiv(block_offset, block_size)

        if layout == "NCHW":
            output = data(n, channel_idx, y_idx + (y * block_size), x_idx + (x * block_size))
        else:
            output = data(n, y_idx + (y * block_size), x_idx + (x * block_size), channel_idx)
        return output

    def _compute(*indices):
        n, c, y, x = _get_indices(*indices)
        return _get_pixel(n, c, y, x)

    return te.compute(output_shape, _compute, name="space_to_depth", tag=tag.INJECTIVE)
