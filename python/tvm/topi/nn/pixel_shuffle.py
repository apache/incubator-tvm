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
# ruff: noqa: RUF005
"""TVM operator pixel shuffle compute."""

import tvm
import tvm_tirx


def pixel_shuffle(data, upscale_factor, name="PixelShuffle"):
    """PixelShuffle operator that rearranges elements in a tensor of shape
    [..., C * r * r, H, W] to [..., C, H * r, W * r].

    Parameters
    ----------
    data : tvm.te.Tensor
        N-D input tensor with at least 3 dimensions. Channel must be at index -3.

    upscale_factor : int
        The upscale factor (r).

    name : str
        Name of the output tensor.

    Returns
    -------
    output : tvm.te.Tensor
        Pixel shuffled tensor with shape [..., C, H*r, W*r]
    """
    assert isinstance(upscale_factor, int) and upscale_factor > 0
    ndim = len(data.shape)
    assert ndim >= 3, "Input must be at least 3D"

    upscale_factor_const = tvm_tirx.tirx.const(upscale_factor, "int32")
    c_in, h_in, w_in = data.shape[-3], data.shape[-2], data.shape[-1]

    c_out = tvm_tirx.tirx.floordiv(c_in, upscale_factor_const * upscale_factor_const)
    h_out = h_in * upscale_factor_const
    w_out = w_in * upscale_factor_const

    out_shape = list(data.shape[:-3]) + [c_out, h_out, w_out]

    def _compute(*indices):
        batch_indices = indices[:-3]
        c_out_idx, h_out_idx, w_out_idx = indices[-3], indices[-2], indices[-1]

        h_idx = tvm_tirx.tirx.floordiv(h_out_idx, upscale_factor_const)
        h_offset = h_out_idx % upscale_factor_const

        w_idx = tvm_tirx.tirx.floordiv(w_out_idx, upscale_factor_const)
        w_offset = w_out_idx % upscale_factor_const

        c_in_idx = (
            (c_out_idx * upscale_factor_const * upscale_factor_const)
            + (h_offset * upscale_factor_const)
            + w_offset
        )

        index_tuple = batch_indices + (c_in_idx, h_idx, w_idx)
        return data[index_tuple]

    return tvm.te.compute(out_shape, _compute, name=name)
