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
# pylint: disable=unused-argument, redefined-builtin
"""Bitserial Conv2D operators"""

import tvm_tirx
from tvm import te

from ..utils import get_const_tuple
from .bitserial_util import bitpack
from .pad import pad
from .utils import get_pad_tuple


def bitserial_conv2d_nchw(
    data,
    kernel,
    stride,
    padding,
    activation_bits,
    weight_bits,
    pack_dtype="uint32",
    out_dtype="int16",
    unipolar=True,
):
    """Bitserial Conv2D operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two or four ints
        padding size, [pad_height, pad_width], [pad_top, pad_left, pad_down, pad_right]

    activation_bits: int
        number of bits used for activations/input elements

    weight_bits: int
        number of bits used for weight elements

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type

    unipolar: bool
        if binarization style is in unipolar 1/0 format, instead of bipolar -1/+1 format

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(data, activation_bits, pack_axis=1, bit_axis=2, pack_type=pack_dtype)
    if len(kernel.shape) == 4:
        Filter_q = bitpack(kernel, weight_bits, pack_axis=1, bit_axis=4, pack_type=pack_dtype)
    else:
        Filter_q = kernel
    batch, in_channel, activation_bits, in_height, in_width = Input_q.shape
    num_filter, _, kernel_h, kernel_w, weight_bits = Filter_q.shape

    if isinstance(padding, int) or (isinstance(padding, tuple | list) and len(padding) == 2):
        TPAD, LPAD, DPAD, RPAD = get_pad_tuple(padding, kernel)
    else:
        TPAD, LPAD, DPAD, RPAD = padding
    pad_before = [0, 0, 0, TPAD, LPAD]
    pad_after = [0, 0, 0, DPAD, RPAD]

    PadInput_q = pad(Input_q, pad_before, pad_after, name="pad_temp")
    # compute the output shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    out_channel = num_filter
    out_height = (in_height - kernel_h + TPAD + DPAD) // stride_h + 1
    out_width = (in_width - kernel_w + LPAD + RPAD) // stride_w + 1

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    b1 = te.reduce_axis((0, activation_bits), name="b1")
    b2 = te.reduce_axis((0, weight_bits), name="b2")

    if unipolar:

        def _conv(nn, ff, yy, xx):
            b1b2 = (b1 + b2).astype(out_dtype)
            return te.sum(
                (
                    (
                        tvm_tirx.tirx.popcount(
                            PadInput_q[nn, rc, b1, yy * stride_h + ry, xx * stride_w + rx]
                            & Filter_q[ff, rc, ry, rx, b2]
                        )
                        - tvm_tirx.tirx.popcount(
                            PadInput_q[nn, rc, b1, yy * stride_h + ry, xx * stride_w + rx]
                            & ~Filter_q[ff, rc, ry, rx, b2]
                        )
                    )
                    << (b1b2)
                ).astype(out_dtype),
                axis=[rc, ry, rx, b2, b1],
            ).astype(out_dtype)

    else:

        def _conv(nn, ff, yy, xx):
            b1b2 = (b1 + b2).astype(out_dtype)
            return te.sum(
                (
                    tvm_tirx.tirx.popcount(
                        PadInput_q[nn, rc, b1, yy * stride_h + ry, xx * stride_w + rx]
                        & Filter_q[ff, rc, ry, rx, b2]
                    )
                    << (b1b2)
                ).astype(out_dtype),
                axis=[rc, ry, rx, b2, b1],
            ).astype(out_dtype)

    return te.compute(
        (batch, out_channel, out_height, out_width),
        _conv,
        name="Conv2dOutput",
        tag="bitserial_conv2d_nchw",
    )


def bitserial_conv2d_nhwc(
    data,
    kernel,
    stride,
    padding,
    activation_bits,
    weight_bits,
    pack_dtype="uint32",
    out_dtype="int16",
    unipolar=True,
):
    """Bitserial Conv2D operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    kernel : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two or four ints
        padding size, [pad_height, pad_width], [pad_top, pad_left, pad_down, pad_right]

    activation_bits: int
        number of bits used for activations/input elements

    weight_bits: int
        number of bits used for weight elements

    out_dtype: str
        return type of convolution

    pack_dtype: str
        bit packing type

    unipolar: bool
        if binarization style is in unipolar 1/0 format, instead of bipolar -1/+1 format

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    Input_q = bitpack(data, activation_bits, pack_axis=3, bit_axis=4, pack_type=pack_dtype)
    if len(kernel.shape) == 4:
        Filter_q = bitpack(kernel, weight_bits, pack_axis=2, bit_axis=4, pack_type=pack_dtype)
        kernel_h, kernel_w, _, num_filter, _ = get_const_tuple(Filter_q.shape)
    else:
        Filter_q = kernel
        kernel_h, kernel_w, _, _, num_filter = get_const_tuple(Filter_q.shape)
    batch, in_height, in_width, in_channel_q, _ = get_const_tuple(Input_q.shape)

    if isinstance(padding, int) or (isinstance(padding, tuple | list) and len(padding) == 2):
        TPAD, LPAD, DPAD, RPAD = get_pad_tuple(padding, kernel)
    else:
        TPAD, LPAD, DPAD, RPAD = padding
    pad_before = [0, TPAD, LPAD, 0, 0]
    pad_after = [0, DPAD, RPAD, 0, 0]

    # compute the output shape
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    out_channel = num_filter
    out_height = (in_height - kernel_h + TPAD + DPAD) // stride_h + 1
    out_width = (in_width - kernel_w + LPAD + RPAD) // stride_w + 1
    PadInput_q = pad(Input_q, pad_before, pad_after, name="PaddedInput")

    rc = te.reduce_axis((0, in_channel_q), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    b1 = te.reduce_axis((0, activation_bits), name="b1")
    b2 = te.reduce_axis((0, weight_bits), name="b2")

    if unipolar:

        def _conv(nn, yy, xx, ff):
            b1b2 = (b1 + b2).astype(out_dtype)
            return te.sum(
                (
                    (
                        tvm_tirx.tirx.popcount(
                            PadInput_q[nn, yy * stride_h + ry, xx * stride_w + rx, rc, b1]
                            & Filter_q[ry, rx, rc, ff, b2]
                        )
                        - tvm_tirx.tirx.popcount(
                            PadInput_q[nn, yy * stride_h + ry, xx * stride_w + rx, rc, b1]
                            & ~Filter_q[ry, rx, rc, ff, b2]
                        )
                    )
                    << b1b2
                ).astype(out_dtype),
                axis=[rc, ry, rx, b2, b1],
            )

    else:

        def _conv(nn, yy, xx, ff):
            b1b2 = (b1 + b2).astype(out_dtype)
            return te.sum(
                (
                    tvm_tirx.tirx.popcount(
                        PadInput_q[nn, yy * stride_h + ry, xx * stride_w + rx, rc, b1]
                        & Filter_q[ry, rx, rc, ff, b2]
                    )
                    << b1b2
                ).astype(out_dtype),
                axis=[rc, ry, rx, b2, b1],
            )

    conv = te.compute(
        (batch, out_height, out_width, out_channel),
        _conv,
        name="Conv2dOutput",
        tag="bitserial_conv2d_nhwc",
    )

    return conv
