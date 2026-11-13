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
"""External function interface to BLAS libraries."""

import tvm_tirx
from tvm import te

from ..topi.nn.utils import get_pad_tuple


def matmul(lhs, rhs, transa=False, transb=False, **kwargs):
    """Create an extern op that compute matrix mult of A and rhs with CrhsLAS
    This function serves as an example on how to call external libraries.

    Parameters
    ----------
    lhs: Tensor
        The left matrix operand
    rhs: Tensor
        The right matrix operand
    transa: bool
        Whether transpose lhs
    transb: bool
        Whether transpose rhs

    Returns
    -------
    C: Tensor
        The result tensor.
    """
    n = lhs.shape[1] if transa else lhs.shape[0]
    m = rhs.shape[0] if transb else rhs.shape[1]
    return te.extern(
        (n, m),
        [lhs, rhs],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.dnnl.matmul", ins[0], ins[1], outs[0], transa, transb
        ),
        name="C",
        **kwargs,
    )


def dnnl_conv2d(
    src,
    weights,
    stride,
    padding,
    dilation,
    groups,
    channel_last=False,
    out_dtype="float32",
    **kwargs,
):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    src : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    weights : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups: str
        input data layout: NCHW or NHWC

    channel_last: bool
        chose if input/output data format is in channel_last format(NHWC) or
        in plain format(NCHW)

    out_dtype: str
        output datatype: now only support float32

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """

    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    pre_cast = src.dtype == "float32"
    post_cast = out_dtype == "float32"

    if channel_last:
        batch, in_height, in_width, _ = src.shape
        kernel_h, kernel_w, _, num_filter = weights.shape
    else:
        batch, _, in_height, in_width = src.shape
        num_filter, _, kernel_h, kernel_w = weights.shape

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_height = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1

    if channel_last:
        out_shape = (batch, out_height, out_width, out_channel)
    else:
        out_shape = (batch, out_channel, out_height, out_width)

    return te.extern(
        out_shape,
        [src, weights],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.dnnl.conv2d",
            ins[0],
            ins[1],
            outs[0],
            pad_top,
            pad_down,
            pad_left,
            pad_right,
            stride[0],
            stride[1],
            groups,
            channel_last,
            pre_cast,
            post_cast,
        ),
        name="C",
        dtype=out_dtype,
        **kwargs,
    )
