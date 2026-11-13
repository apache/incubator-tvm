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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Deformable Conv2D operators"""

import tvm_tirx
from tvm import te

from ..cpp.utils import bilinear_sample_nchw, bilinear_sample_nhwc
from ..utils import get_const_tuple
from .utils import get_pad_tuple


def deformable_conv2d_nchw(
    data, offset, kernel, strides, padding, dilation, deformable_groups, groups, out_dtype
):
    """Deformable conv2D operator in NCHW layout.

    The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    offset : tvm.te.Tensor
        4-D with shape [batch, deformable_groups * filter_height * filter_width * 2,
        out_height, out_width].

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    deformable_groups : int
        number of deformable groups

    groups : int
        number of groups

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = data.dtype

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = get_const_tuple(data.shape)
    out_channel, channel, kernel_h, kernel_w = get_const_tuple(kernel.shape)
    _, _, out_height, out_width = get_const_tuple(offset.shape)
    assert in_channel % deformable_groups == 0, "Input cahnnels must divide deformable group size"
    assert groups == 1, "deformable_conv2d_nchw does not support groups > 1"

    ic_per_dgroup = channel // deformable_groups

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, _, _ = get_pad_tuple(padding, (dilated_kernel_h, dilated_kernel_w))
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    zero = tvm_tirx.tirx.const(0.0, data.dtype)

    def _bilinear(n, c, h, w):
        outside = tvm_tirx.tirx.any(h < 0, w < 0, h >= in_height, w >= in_width)
        val = bilinear_sample_nchw(data, (n, c, h, w), in_height - 1, in_width - 1)
        return tvm_tirx.tirx.if_then_else(outside, zero, val)

    data_deform = te.compute(
        (batch, in_channel, kernel_h, kernel_w, out_height, out_width),
        lambda n, c, kh, kw, y, x: _bilinear(
            n,
            c,
            y * stride_h
            - pad_top
            + kh * dilation_h
            + offset[
                n, c // ic_per_dgroup * (kernel_w * kernel_h * 2) + (kh * kernel_w + kw) * 2, y, x
            ],
            x * stride_w
            - pad_left
            + kw * dilation_w
            + offset[
                n,
                c // ic_per_dgroup * (kernel_w * kernel_h * 2) + (kh * kernel_w + kw) * 2 + 1,
                y,
                x,
            ],
        ),
        tag="data_deform",
    )
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda n, f, y, x: te.sum(
            data_deform[n, rc, ry, rx, y, x].astype(out_dtype)
            * kernel[f, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx],
        ),
        tag="deformable_conv2d_nchw",
    )


def deformable_conv2d_nhwc(
    data, offset, kernel, strides, padding, dilation, deformable_groups, groups, out_dtype
):
    """Deformable conv2D operator in NHWC layout.

    The deformable convolution operation is described in https://arxiv.org/abs/1703.06211

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    offset : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width,
                        deformable_groups * filter_height * filter_width * 2].

    kernel : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    deformable_groups : int
        number of deformable groups

    groups : int
        number of groups

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    if out_dtype is None:
        out_dtype = data.dtype

    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = get_const_tuple(data.shape)
    kernel_h, kernel_w, channel, out_channel = get_const_tuple(kernel.shape)
    _, out_height, out_width, _ = get_const_tuple(offset.shape)
    assert in_channel % deformable_groups == 0, "Input cahnnels must divide deformable group size"
    assert groups == 1, "deformable_conv2d_nchw does not support groups > 1"

    ic_per_dgroup = channel // deformable_groups

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, _, _ = get_pad_tuple(padding, (dilated_kernel_h, dilated_kernel_w))
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    zero = tvm_tirx.tirx.const(0.0, data.dtype)

    def _bilinear(n, h, w, c):
        outside = tvm_tirx.tirx.any(h < 0, w < 0, h >= in_height, w >= in_width)
        val = bilinear_sample_nhwc(data, (n, h, w, c), in_height - 1, in_width - 1)
        return tvm_tirx.tirx.if_then_else(outside, zero, val)

    data_deform = te.compute(
        (batch, kernel_h, kernel_w, in_channel, out_height, out_width),
        lambda n, kh, kw, c, y, x: _bilinear(
            n,
            y * stride_h
            - pad_top
            + kh * dilation_h
            + offset[
                n, y, x, c // ic_per_dgroup * (kernel_w * kernel_h * 2) + (kh * kernel_w + kw) * 2
            ],
            x * stride_w
            - pad_left
            + kw * dilation_w
            + offset[
                n,
                y,
                x,
                c // ic_per_dgroup * (kernel_w * kernel_h * 2) + (kh * kernel_w + kw) * 2 + 1,
            ],
            c,
        ),
        tag="data_deform",
    )
    return te.compute(
        (batch, out_height, out_width, out_channel),
        lambda n, y, x, f: te.sum(
            data_deform[n, ry, rx, rc, y, x].astype(out_dtype)
            * kernel[ry, rx, rc, f].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        tag="deformable_conv2d_nhwc",
    )
