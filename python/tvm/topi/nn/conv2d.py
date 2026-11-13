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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
# ruff: noqa: F841, RUF005
"""Conv2D operators"""

import re
from collections import namedtuple
from collections.abc import Sequence

import numpy as np

import tvm_tirx
from tvm import te

from ..utils import get_const_int, get_const_tuple, simplify, tag
from .pad import pad
from .utils import get_pad_tuple, get_pad_tuple_generic
from .winograd_util import winograd_transform_matrices

# workload description of conv2d
Workload = namedtuple(
    "Workload",
    [
        "in_dtype",
        "out_dtype",
        "height",
        "width",
        "in_filter",
        "groups",
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


def conv2d(
    input, filter, strides, padding, dilation, data_layout="NCHW", kernel_layout="", out_dtype=None
):
    """Conv2D operator.

    Parameters
    ----------
    input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] in data_layout

    filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] in kernel_layout

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    data_layout : str
        layout of data

    kernel_layout : Optional[str]
        layout of kernel. If unspecified, use default layout inferred from data_layout. "OIHW" if
        data_layout == "NCHW", "HWIO" if data_layout == "NHWC".

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    # search platform specific declaration first
    # default declaration
    return conv(input, filter, strides, padding, dilation, 1, data_layout, kernel_layout, out_dtype)


def _get_workload(data, kernel, stride, padding, dilation, out_dtype, data_layout="NCHW"):
    """Get the workload structure."""
    if data_layout == "NCHW":
        _, CI, IH, IW = get_const_tuple(data.shape)
    elif data_layout == "NHWC":
        _, IH, IW, CI = get_const_tuple(data.shape)
    elif data_layout == "HWCN":
        IH, IW, CI, _ = get_const_tuple(data.shape)
    else:
        raise ValueError(f"not support this layout {data_layout} yet")

    if data_layout == "NCHW":
        CO, CIG, KH, KW = get_const_tuple(kernel.shape)
    else:
        KH, KW, CIG, CO = get_const_tuple(kernel.shape)

    dilation_h, dilation_w = (
        dilation if isinstance(dilation, tuple | list) else (dilation, dilation)
    )
    pt, pl, pb, pr = get_pad_tuple(
        padding,
        (get_const_int((KH - 1) * dilation_h + 1), get_const_int((KW - 1) * dilation_w + 1)),
    )
    GRPS = CI // CIG
    if isinstance(stride, tuple | list):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    assert (data.dtype == kernel.dtype) or (data.dtype == "uint8" and kernel.dtype == "int8"), (
        f"Do not support inputs with different data types now. {data.dtype} vs. {kernel.dtype}"
    )
    return Workload(
        data.dtype,
        out_dtype,
        IH,
        IW,
        CI,
        GRPS,
        CO,
        KH,
        KW,
        pt,
        pl,
        pb,
        pr,
        dilation_h,
        dilation_w,
        HSTR,
        WSTR,
    )


def conv2d_nchw(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return conv(Input, Filter, stride, padding, dilation, 1, "NCHW", "OIHW", out_dtype=out_dtype)


def conv2d_hwcn(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in HWCN layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [in_height, in_width, in_channel, batch]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [out_height, out_width, out_channel, batch]
    """
    return conv(Input, Filter, stride, padding, dilation, 1, "HWCN", "HWIO", out_dtype=out_dtype)


def conv2d_nhwc(
    Input,
    Filter,
    stride,
    padding,
    dilation,
    out_dtype="float32",
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """Convolution operator in NHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str = "float32",
        The type of output tensor

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the input tensor.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    return conv(
        Input,
        Filter,
        stride,
        padding,
        dilation,
        1,
        "NHWC",
        "HWIO",
        out_dtype,
        auto_scheduler_rewritten_layout,
        meta_schedule_original_shape,
        auto_scheduler_should_rewrite_layout=True,
    )


def conv2d_NCHWc(data, kernel, stride, padding, dilation, layout, out_layout, out_dtype="float32"):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """

    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, tuple | list) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, tuple | list) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, kernel_ic_bn, oc_bn = get_const_tuple(
        kernel.shape
    )
    num_filter = oc_chunk * oc_bn
    groups = in_channel // (ic_chunk_group * kernel_ic_bn)

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    idxdiv = tvm_tirx.tirx.indexdiv
    idxmod = tvm_tirx.tirx.indexmod

    if groups == 1:
        ic = te.reduce_axis((0, in_channel), name="ic")
        return te.compute(
            oshape,
            lambda n, oc_chunk, oh, ow, oc_block: te.sum(
                data_pad[
                    n,
                    idxdiv(ic, ic_bn),
                    oh * HSTR + kh * dilation_h,
                    ow * WSTR + kw * dilation_w,
                    idxmod(ic, ic_bn),
                ].astype(out_dtype)
                * kernel[oc_chunk, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block].astype(
                    out_dtype
                ),
                axis=[ic, kh, kw],
            ),
            name="conv2d_NCHWc",
            tag="conv2d_NCHWc",
        )
    ic = te.reduce_axis((0, in_channel // groups), name="ic")
    return te.compute(
        oshape,
        lambda n, occ, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                (occ // (oc_chunk // groups)) * (ic_chunk // groups) + idxdiv(ic, ic_bn),
                oh * HSTR + kh * dilation_h,
                ow * WSTR + kw * dilation_w,
                idxmod(ic, ic_bn),
            ].astype(out_dtype)
            * kernel[occ, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block].astype(out_dtype),
            axis=[ic, kh, kw],
        ),
        name="conv2d_NCHWc",
        tag="conv2d_NCHWc",
    )


def conv2d_NCHWc_OIHWo(
    data: te.Tensor, kernel, stride, padding, dilation, layout, out_layout, out_dtype="float32"
):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        6-D with shape ``[num_filter_chunk, in_channel_chunk, filter_height,
        filter_width, num_filter_block]``.

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """

    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, tuple | list) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, tuple | list) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    kernel_shape = get_const_tuple(kernel.shape)
    if len(kernel_shape) == 6:  # OIHW4i4o
        oc_chunk, ic_chunk_group, kernel_height, kernel_width, kernel_ic_bn, oc_bn = kernel_shape
        groups = in_channel // (ic_chunk_group * kernel_ic_bn)
    else:  # OIHW4o
        oc_chunk, ic, kernel_height, kernel_width, oc_bn = kernel_shape
        groups = in_channel // ic

    num_filter = oc_chunk * oc_bn
    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="conv2d_data_pad")
    else:
        data_pad = data

    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    idxdiv = tvm_tirx.tirx.indexdiv
    idxmod = tvm_tirx.tirx.indexmod

    def compute_conv2d(*args):
        n, occ, oh, ow, ocb = args
        ic = te.reduce_axis((0, in_channel // groups), name="ic")
        if groups == 1:
            data_pad_ = data_pad[
                n,
                idxdiv(ic, ic_bn),
                oh * HSTR + kh * dilation_h,
                ow * WSTR + kw * dilation_w,
                idxmod(ic, ic_bn),
            ]
        else:
            data_pad_ = data_pad[
                n,
                (occ // (oc_chunk // groups)) * (ic_chunk // groups) + idxdiv(ic, ic_bn),
                oh * HSTR + kh * dilation_h,
                ow * WSTR + kw * dilation_w,
                idxmod(ic, ic_bn),
            ]
        if len(kernel_shape) == 5:
            kernel_ = kernel[occ, ic, kh, kw, ocb]
        else:
            kernel_ = kernel[occ, idxdiv(ic, oc_bn), kh, kw, idxmod(ic, oc_bn), ocb]

        if out_dtype is not None:
            data_pad_ = data_pad_.astype(out_dtype)
            kernel_ = kernel_.astype(out_dtype)

        return te.sum(
            data_pad_ * kernel_,
            axis=[ic, kh, kw],
        )

    return te.compute(
        oshape,
        lambda *indices: compute_conv2d(*indices),  # pylint: disable=W0108
        name="conv2d_NCHWc_OIHWo",
        tag="conv2d_NCHWc_OIHWo",
    )


def conv2d_NCHWc_int8(
    data, kernel, stride, padding, dilation, layout, out_layout, out_dtype="int32", n_elems=4
):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        7-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width, in_channel_block/4,
        num_filter_block, 4]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    n_elems : int
        numer of int8 elements accumulated

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """

    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, tuple | list) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, tuple | list) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)[
        :6
    ]
    groups = ic_chunk // ic_chunk_group

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    if groups == 1:
        ic_outer = te.reduce_axis((0, in_channel // ic_bn), name="ic_outer")
        ic_f_inner = te.reduce_axis((0, ic_bn // n_elems), name="ic_f_inner")
        ic_s_inner = te.reduce_axis((0, n_elems), name="ic_s_inner")
        return te.compute(
            oshape,
            lambda n, oc_chunk, oh, ow, oc_block: te.sum(
                data_pad[
                    n,
                    ic_outer,
                    oh * HSTR + kh * dilation_h,
                    ow * WSTR + kw * dilation_w,
                    ic_f_inner * n_elems + ic_s_inner,
                ].astype(out_dtype)
                * kernel[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner].astype(
                    out_dtype
                ),
                axis=[kh, kw, ic_outer, ic_f_inner, ic_s_inner],
            ),
            name="conv2d_NCHWc_int8",
            tag="conv2d_NCHWc_int8",
            attrs={"schedule_rule": "conv2d_NCHWc_int8"},
        )
    # for int8 group conv support
    ic_chunk = in_channel // ic_bn
    ic_outer = te.reduce_axis((0, ic_chunk // groups), name="ic_outer")
    ic_f_inner = te.reduce_axis((0, ic_bn // n_elems), name="ic_f_inner")
    ic_s_inner = te.reduce_axis((0, n_elems), name="ic_s_inner")
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    return te.compute(
        oshape,
        lambda n, occ, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                (occ * oc_bn // (oc_chunk * oc_bn // groups)) * (ic_chunk // groups) + ic_outer,
                oh * HSTR + kh,
                ow * WSTR + kw,
                ic_f_inner * n_elems + ic_s_inner,
            ].astype(out_dtype)
            * kernel[occ, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner].astype(out_dtype),
            axis=[kh, kw, ic_outer, ic_f_inner, ic_s_inner],
        ),
        name="conv2d_NCHWc_int8",
        tag="conv2d_NCHWc_int8",
        attrs={"schedule_rule": "conv2d_NCHWc_int8"},
    )


def conv2d_winograd_weight_transform(kernel, tile_size):
    """Weight transformation for winograd

    Parameters
    ----------
    kernel: Tensor
        The raw kernel tensor with layout "NCHW".
    tile_size: int
        Tile size of winograd transform. e.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [alpha, alpha, CO, CI]
    """
    shape = get_const_tuple(kernel.shape)
    assert shape[2] == shape[3], "Only support NxN kernel"

    K = shape[3]
    r = tile_size + K - 1
    shape = (r, r) + shape[:2]

    _, _, G = winograd_transform_matrices(tile_size, K, kernel.dtype)

    r_kh = te.reduce_axis((0, K), name="r_kh")
    r_kw = te.reduce_axis((0, K), name="r_kw")
    return te.compute(
        shape,
        lambda eps, nu, co, ci: te.sum(
            kernel[co][ci][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
        ),
        name="transform_weight",
    )


def group_conv2d_nchw(Input, Filter, stride, padding, dilation, groups, out_dtype=None):
    """Group convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel // groups, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return conv(
        Input, Filter, stride, padding, dilation, groups, "NCHW", "OIHW", out_dtype=out_dtype
    )


def conv(
    inp: te.Tensor,
    filt: te.Tensor,
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    groups: int,
    data_layout: str,
    kernel_layout: str = "",
    out_dtype: str | None = None,
    auto_scheduler_rewritten_layout: str | None = None,
    meta_schedule_original_shape=None,
    auto_scheduler_should_rewrite_layout: bool = False,
):
    """Convolution operator in NCHW or NHWC layout.

    Supports 1D, 2D, 3D, ... and grouping.

    Parameters
    ----------
    inp : tvm.te.Tensor
        N-D with shape [batch, in_channel, in_height, in_width, ...] in `data_layout`

    filt : tvm.te.Tensor
        N-D with shape [num_filter, in_channel // groups, filter_height, filter_width, ...] in
        `kernel_layout`

    stride : int or a list/tuple of dim ints
        (where dim=2 for NCHW, dim=1 for NCH, etc.)
        Stride size, or [stride_height, stride_width, ...]

    padding : int or a list/tuple of dim or 2*dim ints
        (where dim=2 for NCHW, dim=1 for NCH, etc.)
        padding size, or
        [pad_height, pad_width, ...] for dim ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2*dim ints

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    data_layout : str
        Layout of the input. N indicates batch dimension, C indicates
        channels, any other character indicates HW (or H or HWD for 1D and 3D).

    kernel_layout: Optional[str]
        Layout of the filter. I indicates input channels, O indicates output channels,
        any other character indicates HW dimension of the filter (or H or HWD for 1D and 3D).
        If kernel_layout is empty, use data_layout to infer the default kernel_layout. Default
        kernel_layout is OIHW for NCHW data layout, HWIO for NHWC data layout.

    out_dtype : str
        Elements are converted to this type before elementwise multiplication
        and summation.

    auto_scheduler_rewritten_layout: str
        Layout from autoscheduler's layout rewritting.

    meta_schedule_original_shape : Optional[List[Expr]]
        The original shape of the input tensor.

    auto_scheduler_should_rewrite_layout : bool
        Should auto scheduler be allowed to rewrite the layout of the filter
        tensor. Defaults to false. This can cause errors if used with grouped
        convs.

    Returns
    -------
    Output : tvm.te.Tensor
        N-D with shape [batch, out_channel, out_height, out_width, ...] in `data_layout`
    """
    dim = len(inp.shape) - 2
    if out_dtype is None:
        out_dtype = inp.dtype
    assert isinstance(stride, int) or len(stride) == dim
    assert isinstance(dilation, int) or len(dilation) == dim
    if isinstance(stride, int):
        strides = [stride for _ in range(dim)]
    else:
        strides = stride

    if isinstance(dilation, int):
        dilations = [dilation for _ in range(dim)]
    else:
        dilations = list(dilation)

    # transform from data_layout to NCHW
    data_permutation_to = [data_layout.find("N"), data_layout.find("C")] + [
        x.span()[0] for x in re.finditer("[^NC]", data_layout)
    ]
    # transform from NCHW to data_layout
    data_permutation_from = np.argsort(data_permutation_to)
    # transform from CHW to data_layout
    data_permutation_from_reductions = data_permutation_from[1:].copy()
    data_permutation_from_reductions[
        data_permutation_from_reductions > data_permutation_from[0]
    ] -= 1

    if kernel_layout == "":
        # kernel permutation, if C appears before HW then num_filter is first, otherwise it is last
        # tkonolige: I don't really understand kernel ordering for NHWC, it seems
        # like num_filters should match the N dimension
        if data_layout.find("C") < re.search("[^NC]", data_layout).span()[0]:
            kernel_permutation_to = [0, 1] + list(range(2, dim + 2))
        else:
            kernel_permutation_to = [dim + 1, dim] + list(range(dim))
    else:
        # transform from kernel_layout to OIHW
        kernel_permutation_to = [kernel_layout.find("O"), kernel_layout.find("I")] + [
            x.span()[0] for x in re.finditer("[^OI]", kernel_layout)
        ]
    # transform from OIHW to kernel_layout
    kernel_permutation_from = np.argsort(kernel_permutation_to)

    if meta_schedule_original_shape:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")
    batch, in_channel, *dimensions = np.array(get_const_tuple(inp.shape))[
        data_permutation_to
    ].tolist()
    num_filter, _, *kernel_dimensions = np.array(get_const_tuple(filt.shape))[
        kernel_permutation_to
    ].tolist()

    # Autoscheduler may have messed with the input layout, so we extract the
    # dimensions that it gives us
    if auto_scheduler_rewritten_layout:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    if isinstance(in_channel, int):
        assert in_channel % groups == 0, "input channels must divide group size"
    if isinstance(num_filter, int):
        assert num_filter % groups == 0, "output channels must divide group size"

    dilated_kernel_dimensions = [(k - 1) * dil + 1 for k, dil in zip(kernel_dimensions, dilations)]
    pad_begin, pad_end = get_pad_tuple_generic(padding, dilated_kernel_dimensions)
    # compute the output shape
    out_channel = num_filter
    out_dimensions = [
        simplify((d - (k - 1) * dil - 1 + pb + pe) // stride + 1)
        for d, k, dil, pb, pe, stride in zip(
            dimensions, kernel_dimensions, dilations, pad_begin, pad_end, strides
        )
    ]
    for out_dim in out_dimensions:
        if isinstance(out_dim, int) and out_dim <= 0:
            raise ValueError(
                f"Invalid conv parameters: lead to negative output shape {out_dimensions}. "
            )
    # compute graph
    pad_before = list(np.array([0, 0] + pad_begin)[data_permutation_from])
    pad_after = list(np.array([0, 0] + pad_end)[data_permutation_from])
    temp = pad(inp, pad_before, pad_after, name="pad_temp")
    rc = te.reduce_axis((0, in_channel // groups), name="rc")
    rs = [te.reduce_axis((0, k), name=f"r{i}") for i, k in zip(["y", "x", "z"], kernel_dimensions)]

    def compute(*args):
        nn, ff, *dim_indices = list(np.array(args)[data_permutation_to])

        if groups == 1:
            simplified_channel_index = rc
        else:
            simplified_channel_index = ff // (num_filter // groups) * (in_channel // groups) + rc

        return te.sum(
            temp.__getitem__(
                tuple(
                    np.array(
                        [nn, simplified_channel_index]
                        + [
                            di * stride + r * dil
                            for di, stride, r, dil in zip(dim_indices, strides, rs, dilations)
                        ]
                    )[data_permutation_from]
                )
            ).astype(out_dtype)
            * filt.__getitem__(tuple(np.array([ff, rc] + rs)[kernel_permutation_from])).astype(
                out_dtype
            ),
            # Schedules depend on reduction axes being in the same order as the
            # layout, so we reorder here.
            axis=np.array([rc, *rs])[data_permutation_from_reductions].tolist(),
        )

    out = te.compute(
        list(np.array([batch, out_channel] + out_dimensions)[data_permutation_from]),
        compute,
        # tag is expected to be lowercase
        tag=f"{'group_' if groups > 1 else ''}conv{dim}d_{data_layout.lower()}",
        name=f"{'group_' if groups > 1 else ''}conv{dim}d_{data_layout.lower()}",
        attrs={"layout_free_placeholders": [filt]} if auto_scheduler_should_rewrite_layout else {},
        varargs_names=list(np.array(["nn", "ff", "yy", "xx", "zz"])[data_permutation_from]),
    )
    # if we used autoscheduler's changed layout we need to rewrite the ordering
    # of the output dimensions
    if auto_scheduler_rewritten_layout:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")
    return out


def group_conv2d_nhwc(Input, Filter, stride, padding, dilation, groups, out_dtype=None):
    """Group convolution operator in NHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel, ...]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel // groups, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation : int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    groups : int
        number of groups

    out_dtype : str
        The output type. This is used for mixed precision.

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    return conv(
        Input, Filter, stride, padding, dilation, groups, "NHWC", "HWIO", out_dtype=out_dtype
    )


def unpack_NCHWc_to_nchw(packed_out, out_dtype):
    """Unpack conv2d_NCHWc output from layout NCHWc to NCHW

    Parameters
    ----------
    packed_out : tvm.te.Tensor
        The output tensor of conv2d_NCHWc.

    out_dtype : str
        The output dtype.

    Returns
    -------
    unpacked_out : tvm.te.Tensor
        The unpacked output tensor in NCHW layout.
    """
    n, oc_chunk, oh, ow, oc_bn = get_const_tuple(packed_out.shape)

    idxmod = tvm_tirx.tirx.indexmod
    idxdiv = tvm_tirx.tirx.indexdiv

    oshape = (n, oc_chunk * oc_bn, oh, ow)
    unpacked_out = te.compute(
        oshape,
        lambda n, c, h, w: packed_out[n, idxdiv(c, oc_bn), h, w, idxmod(c, oc_bn)].astype(
            out_dtype
        ),
        name="output_unpack",
        tag=tag.INJECTIVE + ",unpack_nchwc",
    )
    return unpacked_out


def conv2d_winograd_nhwc(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    pre_computed=False,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """Conv2D Winograd in NHWC layout.
    This is a clean version to be used by the auto-scheduler for both CPU and GPU.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    weight : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype : str, optional
        Specifies the output data type.
    pre_computed: bool
        Whether the kernel is precomputed
    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.
    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the input tensor.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    tile_size = 4
    return _conv2d_winograd_nhwc_impl(
        data,
        weight,
        strides,
        padding,
        dilation,
        out_dtype,
        tile_size,
        pre_computed=pre_computed,
        write_cache_level=2,
        auto_scheduler_rewritten_layout=auto_scheduler_rewritten_layout,
        meta_schedule_original_shape=meta_schedule_original_shape,
    )


def conv2d_winograd_nchw(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    pre_computed=False,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """Conv2D Winograd in NCHW layout.
    This is a clean version to be used by the auto-scheduler for both CPU and GPU.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]
    weight : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype : str, optional
        Specifies the output data type.
    pre_computed: bool
        Whether the kernel is precomputed
    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.
    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the input tensor.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    tile_size = 4
    return _conv2d_winograd_nchw_impl(
        data,
        weight,
        strides,
        padding,
        dilation,
        out_dtype,
        tile_size,
        pre_computed,
        auto_scheduler_rewritten_layout,
        meta_schedule_original_shape,
    )


def _conv2d_winograd_nhwc_impl(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    tile_size,
    pre_computed=False,
    write_cache_level=None,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """Conv2D Winograd implementation in NHWC layout.
    This is a clean version to be used by the auto-scheduler for both CPU and GPU.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    weight : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype : str, optional
        Specifies the output data type.
    tile_size : int
        The size of the tile to use for the Winograd filter
    pre_computed: bool = False
        Whether the kernel is precomputed
    write_cache_level: Optional[int] = None
        The cache level to write to in multi-level tiling rule in MetaSchedule.
    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.
    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the input tensor.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    N, H, W, CI = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    if meta_schedule_original_shape:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
    if not pre_computed:
        KH, KW, CI, CO = get_const_tuple(weight.shape)
    else:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    pad_t, pad_l, pad_b, pad_r = get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides
    assert HSTR == 1 and WSTR == 1 and KH == 3 and KW == 3

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (H + pad_t + pad_b - KH) // HSTR + 1
    W = (W + pad_l + pad_r - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW

    pad_extra = (nW - 1) * m + alpha - (H + pad_t + pad_b)
    data_pad = pad(
        data,
        (0, pad_t, pad_l, 0),
        (0, pad_b + pad_extra, pad_r + pad_extra, 0),
        name="data_pad",
        attrs={"schedule_rule": "None"},
    )

    if not pre_computed:
        r_kh = te.reduce_axis((0, KH), name="r_kh")
        r_kw = te.reduce_axis((0, KW), name="r_kw")
        kernel_pack = te.compute(
            (alpha, alpha, CO, CI),
            lambda eps, nu, co, ci: te.sum(
                weight[r_kh, r_kw, ci, co] * G[eps, r_kh] * G[nu, r_kw], axis=[r_kh, r_kw]
            ),
            name="kernel_pack",
        )
        bgemm_attrs = {}
    else:
        kernel_pack = weight
        bgemm_attrs = {"layout_free_placeholders": [kernel_pack]}
    if write_cache_level is not None:
        if not isinstance(write_cache_level, int):
            bgemm_attrs["meta_schedule.write_cache_level"] = write_cache_level
        else:
            bgemm_attrs["meta_schedule.write_cache_level"] = [write_cache_level]

    # pack data tile
    input_tile = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: data_pad[
            p // (nH * nW), ((p // nW) % nH) * m + eps, (p % nW) * m + nu, ci
        ],
        name="input_tile",
        attrs={"schedule_rule": "None"},
    )

    # transform data
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    data_pack = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: te.sum(
            input_tile[r_a, r_b, p, ci] * B[r_a, eps] * B[r_b, nu], axis=[r_a, r_b]
        ),
        name="data_pack",
        attrs={
            "auto_scheduler_simplify_const_tensor_indices": ["eps", "nu", "r_a", "r_b"],
            "schedule_rule": "conv2d_nhwc_winograd_data_pack",
        },
    )

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    bgemm = te.compute(
        (alpha, alpha, P, CO),
        lambda eps, nu, p, co: te.sum(
            data_pack[eps, nu, p, ci] * kernel_pack[eps, nu, co, ci], axis=[ci]
        ),
        name="bgemm",
        attrs=bgemm_attrs,
    )

    if auto_scheduler_rewritten_layout:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    # inverse transform

    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    inverse = te.compute(
        (m, m, P, CO),
        lambda vh, vw, p, co: te.sum(
            bgemm[r_a, r_b, p, co] * A[r_a, vh] * A[r_b, vw], axis=[r_a, r_b]
        ),
        name="inverse",
        attrs={
            "auto_scheduler_simplify_const_tensor_indices": ["vh", "vw", "r_a", "r_b"],
            "schedule_rule": "conv2d_nhwc_winograd_inverse",
        },
    )

    # output
    output = te.compute(
        (N, H, W, CO),
        lambda n, h, w, co: inverse[h % m, w % m, n * nH * nW + (h // m) * nW + (w // m), co],
        name="conv2d_winograd",
    )

    return output


def _conv2d_winograd_nchw_impl(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    tile_size,
    pre_computed=False,
    write_cache_level=None,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """
    write_cache_level: Optional[int] = None
        The cache level to write to in multi-level tiling rule in MetaSchedule.
    """
    del auto_scheduler_rewritten_layout

    N, CI, H, W = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    if meta_schedule_original_shape:
        raise RuntimeError("LEGACY-FLOW triggered, to be removed")

    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides

    if not pre_computed:  # kernel tensor is raw tensor, do strict check
        CO, CI, KH, KW = get_const_tuple(weight.shape)
        alpha = KW + tile_size - 1
        assert HSTR == 1 and WSTR == 1 and KH == KW
    else:
        alpha, _, CI, CO = get_const_tuple(weight.shape)
        KH = KW = alpha + 1 - tile_size
        assert HSTR == 1 and WSTR == 1 and dilation_h == 1 and dilation_w == 1

    pad_t, pad_l, pad_b, pad_r = get_pad_tuple(padding, (KH, KW))
    assert HSTR == 1 and WSTR == 1 and KH == 3 and KW == 3

    pt, pl, pb, pr = get_pad_tuple(padding, (KH, KW))
    data_pad = pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")

    r = KW
    m = tile_size
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (H + pt + pb - KH) // HSTR + 1
    W = (W + pl + pr - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m

    P = N * nH * nW if isinstance(N, int) else nH * nW

    # transform kernel
    if not pre_computed:
        r_kh = te.reduce_axis((0, KH), name="r_kh")
        r_kw = te.reduce_axis((0, KW), name="r_kw")
        kernel_pack = te.compute(
            (alpha, alpha, CI, CO),
            lambda eps, nu, ci, co: te.sum(
                weight[co, ci, r_kh, r_kw] * G[eps, r_kh] * G[nu, r_kw], axis=[r_kh, r_kw]
            ),
            name="kernel_pack",
        )
        bgemm_attrs = {}
    else:
        kernel_pack = weight
        bgemm_attrs = {"layout_free_placeholders": [kernel_pack]}
    if write_cache_level is not None:
        if not isinstance(write_cache_level, int):
            bgemm_attrs["meta_schedule.write_cache_level"] = write_cache_level
        else:
            bgemm_attrs["meta_schedule.write_cache_level"] = [write_cache_level]

    # pack data tile
    input_tile = te.compute(
        (CI, P, alpha, alpha),
        lambda ci, p, eps, nu: data_pad[
            p // (nH * nW), ci, ((p // nW) % nH) * m + eps, (p % nW) * m + nu
        ],
        name="input_tile",
        attrs={"schedule_rule": "None"},
    )

    # transform data
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    data_pack = te.compute(
        (alpha, alpha, CI, P),
        lambda eps, nu, ci, p: te.sum(
            input_tile[ci, p, r_a, r_b] * B[r_a, eps] * B[r_b, nu], axis=[r_a, r_b]
        ),
        name="data_pack",
        attrs={"schedule_rule": "conv2d_nchw_winograd_data_pack"},
    )

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    bgemm = te.compute(
        (alpha, alpha, CO, P),
        lambda eps, nu, co, p: te.sum(
            data_pack[eps, nu, ci, p] * kernel_pack[eps, nu, ci, co], axis=[ci]
        ),
        name="bgemm",
        attrs=bgemm_attrs,
    )

    # inverse transform
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    inverse = te.compute(
        (CO, P, m, m),
        lambda co, p, vh, vw: te.sum(
            bgemm[r_a, r_b, co, p] * A[r_a, vh] * A[r_b, vw], axis=[r_a, r_b]
        ),
        name="inverse",
        attrs={"schedule_rule": "conv2d_nchw_winograd_inverse"},
    )

    # output
    output = te.compute(
        (N, CO, H, W),
        lambda n, co, h, w: inverse[co, n * nH * nW + (h // m) * nW + (w // m), h % m, w % m],
        name="conv2d_winograd",
    )

    return output


def conv2d_winograd_nhwc_without_weight_transform(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """Conv2D Winograd without layout transform in NHWC layout.
    This is a clean version to be used by the auto-scheduler for both CPU and GPU.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    weight : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype : str, optional
        Specifies the output data type.
    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.
    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the input tensor.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """

    return conv2d_winograd_nhwc(
        data,
        weight,
        strides,
        padding,
        dilation,
        out_dtype,
        pre_computed=True,
        auto_scheduler_rewritten_layout=auto_scheduler_rewritten_layout,
        meta_schedule_original_shape=meta_schedule_original_shape,
    )


def conv2d_winograd_nchw_without_weight_transform(
    data,
    weight,
    strides,
    padding,
    dilation,
    out_dtype,
    auto_scheduler_rewritten_layout="",
    meta_schedule_original_shape=None,
):
    """Conv2D Winograd without layout transform in NCHW layout.
    This is a clean version to be used by meta-schedule for both CPU and GPU.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]
    weight : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]
    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]
    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]
    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]
    out_dtype : str, optional
        Specifies the output data type.
    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.
    meta_schedule_original_shape: Optional[List[Expr]] = None
        The original shape of the input tensor.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    return conv2d_winograd_nchw(
        data,
        weight,
        strides,
        padding,
        dilation,
        out_dtype,
        pre_computed=True,
        auto_scheduler_rewritten_layout=auto_scheduler_rewritten_layout,
        meta_schedule_original_shape=meta_schedule_original_shape,
    )
