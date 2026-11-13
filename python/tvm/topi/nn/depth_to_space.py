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
"""TVM operator depth_to_space compute."""

import tvm_tirx
from tvm import te

from .. import tag


def depth_to_space(data, block_size, layout="NCHW", mode="DCR"):
    """Perform depth to space transformation on the data

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D tensor in either NCHW or NHWC layout.

    block_size : int
        Size of blocks to compose from channel dimension.

    layout : string
        Either NCHW or NHWC, indicating data layout.

    mode : string
        Either DCR or CDR, indicates how channels should be accessed.
        In DCR, channels are interwoven in the Tensorflow style while
        in CDR channels are accessed sequentially as in Pytorch.

    Returns
    -------
    output : tvm.te.Tensor
        Output of shape [N, C / block_size**2, H * block_size, W * block_size]
    """
    if layout == "NCHW":
        in_n, in_c, in_h, in_w = data.shape
        channel_factor = tvm_tirx.tirx.truncdiv(in_c, (block_size * block_size))
        output_shape = [in_n, channel_factor, in_h * block_size, in_w * block_size]
    elif layout == "NHWC":
        in_n, in_h, in_w, in_c = data.shape
        channel_factor = tvm_tirx.tirx.truncdiv(in_c, (block_size * block_size))
        output_shape = [in_n, in_h * block_size, in_w * block_size, channel_factor]
    else:
        raise ValueError("Only NCHW and NHWC layouts are currently supported.")

    def _get_indices(*indices):
        if layout == "NCHW":
            n, c, y, x = indices
        elif layout == "NHWC":
            n, y, x, c = indices
        return n, c, y, x

    def _get_pixel(n, c, y, x):
        block_x = tvm_tirx.tirx.truncdiv(x, block_size)
        block_y = tvm_tirx.tirx.truncdiv(y, block_size)
        idx_x = tvm_tirx.tirx.truncmod(x, block_size)
        idx_y = tvm_tirx.tirx.truncmod(y, block_size)
        if mode == "DCR":
            channel_idx = channel_factor * ((block_size * idx_y) + idx_x) + c
        else:
            channel_idx = (c * block_size * block_size) + ((block_size * idx_y) + idx_x)

        if layout == "NCHW":
            output = data(n, channel_idx, block_y, block_x)
        else:
            output = data(n, block_y, block_x, channel_idx)
        return output

    def _compute(*indices):
        n, c, y, x = _get_indices(*indices)
        return _get_pixel(n, c, y, x)

    return te.compute(output_shape, _compute, name="depth_to_space", tag=tag.INJECTIVE)
