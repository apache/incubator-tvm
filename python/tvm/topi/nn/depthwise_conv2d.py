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
# pylint: disable=invalid-name, unused-variable, too-many-locals, unused-argument
# ruff: noqa: F841
"""Depthwise convolution operators"""

from collections import namedtuple

import numpy as np

import tvm_tirx
from tvm import te

from ..utils import get_const_tuple, simplify
from .dilate import dilate
from .pad import pad
from .utils import get_pad_tuple

# workload description of depthwise-conv2d
Workload = namedtuple(
    "Workload",
    [
        "in_dtype",
        "out_dtype",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "kernel_h",
        "kernel_w",
        "padt",
        "padl",
        "padb",
        "padr",
        "dilation_h",
        "dilation_w",
        "stride_h",
        "stride_w",
    ],
)


def _get_workload(data, kernel, stride, padding, dilation, out_dtype, data_layout="NCHW"):
    """Get the workload structure for a depthwise conv2d.

    Input data and filter should use NCHW layout.
    """
    if data_layout == "NCHW":
        _, in_channel, height, width = get_const_tuple(data.shape)
        filter_channel, channel_multiplier, kh, kw = get_const_tuple(kernel.shape)
    elif data_layout == "NHWC":
        _, height, width, in_channel = get_const_tuple(data.shape)
        kh, kw, filter_channel, channel_multiplier = get_const_tuple(kernel.shape)
    elif data_layout == "NCHWc":
        _, in_channel_chunk, height, width, in_channel_block = get_const_tuple(data.shape)
        in_channel = in_channel_chunk * in_channel_block
        (filter_channel_chunk, cm_chunk, kh, kw, cm_block, filter_channel_block) = get_const_tuple(
            kernel.shape
        )
        filter_channel = filter_channel_chunk * filter_channel_block
        channel_multiplier = cm_chunk * cm_block

        assert in_channel_block == filter_channel_block, (
            f"Incorrect dimensions, data has block size {in_channel_block}, but filter has "
            f"block size {filter_channel_block}"
        )

    else:
        raise ValueError(f"Data layout {data_layout} not supported")

    assert in_channel == filter_channel, (
        f"Incorrect dimensions, data has {in_channel} channels but filter expects "
        f"{filter_channel} channels"
    )

    out_channel = filter_channel * channel_multiplier
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, tuple | list) else (dilation, dilation)
    )
    if isinstance(stride, tuple | list):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    assert (data.dtype == kernel.dtype) or (data.dtype == "uint8" and kernel.dtype == "int8"), (
        f"Do not support inputs with different data types now. {data.dtype} vs. {kernel.dtype}"
    )
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    pt, pl, pb, pr = get_pad_tuple(padding, (dilated_kernel_h, dilated_kernel_w))
    return Workload(
        data.dtype,
        out_dtype,
        height,
        width,
        in_channel,
        out_channel,
        kh,
        kw,
        pt,
        pl,
        pb,
        pr,
        dilation_h,
        dilation_w,
        HSTR,
        WSTR,
    )


def depthwise_conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Depthwise convolution nchw forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [in_channel, channel_multiplier, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        The spatial stride, or (stride_height, stride_width).

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    # shape of dilated kernel
    filter_channel, channel_multiplier, filter_height, filter_width = Filter.shape

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = simplify(in_channel * channel_multiplier)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # padding stage
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    # depthconv stage
    idxdiv = tvm_tirx.tirx.indexdiv
    idxmod = tvm_tirx.tirx.indexmod
    di = te.reduce_axis((0, filter_height), name="di")
    dj = te.reduce_axis((0, filter_width), name="dj")
    Output = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda b, c, i, j: te.sum(
            (
                PaddedInput[
                    b,
                    idxdiv(c, channel_multiplier),
                    i * stride_h + di * dilation_h,
                    j * stride_w + dj * dilation_w,
                ].astype(out_dtype)
                * Filter[
                    idxdiv(c, channel_multiplier), idxmod(c, channel_multiplier), di, dj
                ].astype(out_dtype)
            ),
            axis=[di, dj],
        ),
        name="DepthwiseConv2d",
        tag="depthwise_conv2d_nchw",
    )
    return Output


def depthwise_conv2d_nhwc(
    Input, Filter, stride, padding, dilation, kernel_layout="HWOI", out_dtype=None
):
    """Depthwise convolution nhwc forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape

    # shape of dilated kernel
    if kernel_layout == "HWIO":
        filter_height, filter_width, channel_multiplier, filter_channel = Filter.shape
        kernel_permutation = [0, 1, 3, 2]
    else:
        filter_height, filter_width, filter_channel, channel_multiplier = Filter.shape
        kernel_permutation = [0, 1, 2, 3]

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = simplify(in_channel * channel_multiplier)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # padding stage
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    # depthconv stage
    idxdiv = tvm_tirx.tirx.indexdiv
    idxmod = tvm_tirx.tirx.indexmod

    di = te.reduce_axis((0, filter_height), name="di")
    dj = te.reduce_axis((0, filter_width), name="dj")
    Output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda b, i, j, c: te.sum(
            (
                PaddedInput[
                    b,
                    i * stride_h + di * dilation_h,
                    j * stride_w + dj * dilation_w,
                    idxdiv(c, channel_multiplier),
                ].astype(out_dtype)
                * Filter[
                    tuple(
                        np.array(
                            [di, dj, idxdiv(c, channel_multiplier), idxmod(c, channel_multiplier)]
                        )[kernel_permutation]
                    )
                ].astype(out_dtype)
            ),
            axis=[di, dj],
        ),
        name="DepthwiseConv2d",
        tag="depthwise_conv2d_nhwc",
    )
    return Output


def depthwise_conv2d_backward_input_nhwc(Filter, Out_grad, oshape, ishape, stride, padding):
    """Depthwise convolution nhwc backward wrt input operator.

    Parameters
    ----------
    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]

    Out_grad : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    """
    batch, in_h, in_w, in_c = ishape
    _, out_h, out_w, out_c = oshape
    filter_h, filter_w, _, channel_multiplier = Filter.shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    dilated_out_grad = dilate(Out_grad, [1, stride_h, stride_w, 1], name="dilated_out_grad")

    # padding params in forward propagation
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    # padding params in backward propagation
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = (filter_h - 1 - fpad_bottom) + (stride_h - 1)
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = (filter_w - 1 - fpad_right) + (stride_w - 1)

    padded_out_grad = pad(
        dilated_out_grad,
        [0, bpad_top, bpad_left, 0],
        [0, bpad_bottom, bpad_right, 0],
        name="padded_out_grad",
    )

    dh = te.reduce_axis((0, filter_h), name="dh")
    dw = te.reduce_axis((0, filter_w), name="dw")
    dc = te.reduce_axis((0, channel_multiplier), name="dc")

    In_grad = te.compute(
        (batch, in_h, in_w, in_c),
        lambda b, h, w, c: te.sum(
            padded_out_grad[b, h + dh, w + dw, c * channel_multiplier + dc]
            * Filter[filter_h - 1 - dh, filter_w - 1 - dw, c, dc],
            axis=[dh, dw, dc],
        ),
        tag="depthwise_conv2d_backward_input_nhwc",
    )

    return In_grad


def depthwise_conv2d_backward_weight_nhwc(Input, Out_grad, oshape, fshape, stride, padding):
    """Depthwise convolution nhwc backward wrt weight operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Out_grad : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, channel_multiplier]
    """
    batch, out_h, out_w, out_c = oshape
    filter_h, filter_w, _, channel_multiplier = fshape
    in_c = Input.shape[3].value
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (filter_h, filter_w))

    padded_in = pad(
        Input, [0, pad_top, pad_left, 0], [0, pad_bottom, pad_right, 0], name="padded_in"
    )

    dh = te.reduce_axis((0, Out_grad.shape[1].value), name="dh")
    dw = te.reduce_axis((0, Out_grad.shape[2].value), name="dw")
    db = te.reduce_axis((0, batch), name="db")
    idxdiv = tvm_tirx.tirx.indexdiv
    idxmod = tvm_tirx.tirx.indexmod

    Weight_grad = te.compute(
        (filter_h, filter_w, in_c, channel_multiplier),
        lambda fh, fw, c, m: te.sum(
            Out_grad[db, dh, dw, c * channel_multiplier + idxmod(m, channel_multiplier)]
            * padded_in[db, fh + dh * stride_h, fw + dw * stride_w, c],
            axis=[db, dh, dw],
        ),
        tag="depthwise_conv2d_backward_weight_nhwc",
    )

    return Weight_grad


def depthwise_conv2d_NCHWc(
    Input, Filter, stride, padding, dilation, layout, out_layout, out_dtype=None
):
    """Depthwise convolution NCHW[x]c forward operator.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    Filter : tvm.te.Tensor
        6-D with shape [out_channel_chunk, 1, filter_height, filter_width, 1, out_channel_block]
        In NCHWc depthwise convolution,
        we group kernel's in_channel and channel_multiplier together then do the tiling.

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
         dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    raise ValueError("missing register for topi.nn.depthwise_conv2d_NCHWc")
