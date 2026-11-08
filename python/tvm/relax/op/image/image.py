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
"""Image operators."""

from typing import cast

import tvm_ffi

from tvm_tirx.ir import Attrs, is_prim_expr
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir import PrimType as _PrimType
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ...expr import Expr, ShapeExpr

PrimExprLike = int | Expr
SizeLike = PrimExprLike | tuple[PrimExprLike, ...]


@tvm_ffi.register_object("relax.attrs.Resize2DAttrs")
class Resize2DAttrs(Attrs):
    """Attributes used in image resize2d operator"""


def resize2d(
    data: Expr,
    size: SizeLike,
    roi: float | tuple[float] | None = None,
    layout: str = "NCHW",
    method: str = "linear",
    coordinate_transformation_mode: str = "half_pixel",
    rounding_method: str = "round",
    cubic_alpha: float = -0.75,
    cubic_exclude: int = 0,
    extrapolation_value: float = 0.0,
    out_dtype: str | tvm_ffi.dtype | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Image resize2d operator.

    This operator takes data as input and does 2D scaling to the given scale factor.
    In the default case, where the data_layout is `NCHW`
    with data of shape (n, c, h, w)
    out will have a shape (n, c, size[0], size[1])

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("linear", "nearest_neighbor", "cubic")

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    size: SizeLike
        The out size to which the image will be resized.
        If specified as a list, it is required to have length either 1 or 2.
        If specified as an Expr, it is required to have ndim 2.

    roi: Optional[Union[float, Tuple[float]]]
        The region of interest for cropping the input image. Expected to be of
        size 4, and format [start_h, start_w, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    layout : str
        Layout of the input.

    method : str
        Scale method to used [nearest_neighbor, linear, cubic].

    coordinate_transformation_mode : str
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor. Definitions can be found
        in topi/image/resize.py.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: str
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    cubic_alpha: float
        Spline Coefficient for bicubic interpolation

    cubic_exclude: int
        Flag to exclude exterior of the image during bicubic interpolation

    extrapolation_value: float
        Fill value to use when roi is outside of the image

    out_dtype : Optional[str | tvm_ffi.dtype]
        The dtype of the output tensor.
        It it is not specified, the output will have the same dtype as input if not specified.

    Returns
    -------
    result: relax.Expr
        The resized result.
    """
    if roi is None:
        roi = (0.0, 0.0, 0.0, 0.0)  # type: ignore
    elif isinstance(roi, float):
        roi = (roi, roi, roi, roi)  # type: ignore
    elif isinstance(roi, tuple | list):
        roi = tuple(val if isinstance(val, float) else float(val) for val in roi)
    else:
        raise NotImplementedError(f"Unsupported roi type {type(roi)}")

    if isinstance(size, int) or is_prim_expr(size):
        size = (size, size)
    if isinstance(size, tuple | list):
        if len(size) == 1:
            size = ShapeExpr([size[0], size[0]])
        else:
            size = ShapeExpr(size)

    return _Call(
        "relax.image.resize2d",
        [data, size],
        attrs=_make_attrs(
            "relax.attrs.Resize2DAttrs",
            roi=roi,
            layout=layout,
            method=method,
            coordinate_transformation_mode=coordinate_transformation_mode,
            rounding_method=rounding_method,
            cubic_alpha=cubic_alpha,
            cubic_exclude=cubic_exclude,
            extrapolation_value=extrapolation_value,
            out_dtype=(out_dtype.dtype if isinstance(out_dtype, _PrimType) else out_dtype),
        ),
        loc=loc,
        ty=ty,
    )


def resize3d(
    data: Expr,
    size: SizeLike,
    roi: float | tuple[float] | None = None,
    layout: str = "NCDHW",
    method: str = "linear",
    coordinate_transformation_mode: str = "half_pixel",
    rounding_method: str = "",
    cubic_alpha: float = -0.75,
    cubic_exclude: int = 0,
    extrapolation_value: float = 0.0,
    out_dtype: str | tvm_ffi.dtype | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Image resize3d operator.

    This operator takes data as input and does 3D scaling to the given output size.
    In the default case, where data layout is `NCDHW`
    with data of shape (n, c, d, h, w),
    the output has shape (n, c, size[0], size[1], size[2]).
    """
    if roi is None:
        roi = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
    elif isinstance(roi, float):
        roi = (roi, roi, roi, roi, roi, roi)  # type: ignore
    elif isinstance(roi, tuple | list):
        roi = tuple(val if isinstance(val, float) else float(val) for val in roi)
    else:
        raise NotImplementedError(f"Unsupported roi type {type(roi)}")

    if isinstance(size, int) or is_prim_expr(size):
        size = (size, size, size)
    if isinstance(size, tuple | list):
        if len(size) == 1:
            size = ShapeExpr([size[0], size[0], size[0]])
        else:
            size = ShapeExpr(size)

    return _Call(
        "relax.image.resize3d",
        [data, size],
        attrs=_make_attrs(
            "relax.attrs.Resize3DAttrs",
            roi=roi,
            layout=layout,
            method=method,
            coordinate_transformation_mode=coordinate_transformation_mode,
            rounding_method=rounding_method,
            cubic_alpha=cubic_alpha,
            cubic_exclude=cubic_exclude,
            extrapolation_value=extrapolation_value,
            out_dtype=(out_dtype.dtype if isinstance(out_dtype, _PrimType) else out_dtype),
        ),
        loc=loc,
        ty=ty,
    )


def grid_sample(
    data: Expr,
    grid: Expr,
    method: str = "bilinear",
    layout: str = "NCHW",
    padding_mode: str = "zeros",
    align_corners: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Applies grid sampling to input feature map.

    Given data and grid, the output is computed by sampling from data using
    the grid coordinates.

    Parameters
    ----------
    data : relax.Expr
        The input data tensor with shape [N, C, H, W] for NCHW layout.

    grid : relax.Expr
        The grid tensor with shape [N, H_out, W_out, 2]. The values are normalized
        to [-1, 1], where (-1, -1) is the top-left corner and (1, 1) is the bottom-right.

    method : str
        Interpolation method. Can be 'nearest', 'bilinear', or 'bicubic'.

    layout : str
        Layout of the input data. Default is 'NCHW'.

    padding_mode : str
        Padding mode for outside grid values. Can be 'zeros', 'border', or 'reflection'.

    align_corners : bool
        If True, the corner pixels of the input and output tensors are aligned.

    Returns
    -------
    result : relax.Expr
        The sampled output tensor with shape [N, C, H_out, W_out].
    """
    return _Call(
        "relax.image.grid_sample",
        [data, grid],
        attrs=_make_attrs(
            "relax.attrs.GridSampleAttrs",
            method=method,
            layout=layout,
            padding_mode=padding_mode,
            align_corners=align_corners,
        ),
        loc=loc,
        ty=ty,
    )


def affine_grid(
    data: Expr,
    size: SizeLike,
    align_corners: bool = True,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Generate a 2D or 3D sampling grid using an affine transformation matrix.

    This operation is described in https://arxiv.org/pdf/1506.02025.pdf.
    It generates a uniform sampling grid within the target shape, normalizes it
    to [-1, 1], and applies the provided affine transformation.

    Parameters
    ----------
    data : relax.Expr
        The input affine matrix tensor with shape [batch, 2, 3] for 2D or
        [batch, 3, 4] for 3D.

    size : SizeLike
        The target output spatial shape, (H, W) for 2D or (D, H, W) for 3D. If a
        single integer or PrimExpr is provided, it is interpreted as a square 2D
        output shape (size, size).

    align_corners : bool
        If True, normalized grid coordinates map to corner pixels; if False, to
        pixel centers (the PyTorch / ONNX default).

    Returns
    -------
    result : relax.Expr
        The output grid tensor with shape [batch, 2, H, W] for 2D or
        [batch, 3, D, H, W] for 3D.
    """
    if isinstance(size, int) or is_prim_expr(size):
        size = (size, size)
    if isinstance(size, tuple | list):
        size = ShapeExpr(size)

    return cast(
        Expr,
        _Call(
            "relax.image.affine_grid",
            [data, size],
            attrs=_make_attrs("relax.attrs.AffineGridAttrs", align_corners=align_corners),
            loc=loc,
            ty=ty,
        ),
    )


@tvm_ffi.register_object("relax.attrs.Dilation2DAttrs")
class Dilation2DAttrs(Attrs):
    """Attributes for image.dilation2d"""


def dilation2d(
    input: Expr,
    filter: Expr,
    strides: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int, int, int] | str = 0,
    dilations: int | tuple[int, int] = (1, 1),
    data_layout: str = "NCHW",
    out_dtype: str | tvm_ffi.dtype | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    r"""Morphological Dilation 2D operator.

    This operator computes the morphological dilation of the input image
    with the given filter (structuring element). For each position, the
    output is the maximum of `input + filter` over the filter window.

    .. math::

        output[n, c, h, w] = \max_{f_h, f_w}
            \left( input[n, c, h' + d_h \cdot f_h, w' + d_w \cdot f_w]
                 + filter[c, f_h, f_w] \right)

    where :math:`h', w'` are the top-left positions of the filter window
    and :math:`d_h, d_w` are the dilation factors.

    Parameters
    ----------
    input : relax.Expr
        4-D input tensor. With NCHW layout: [batch, in_channel, in_height, in_width].
        With NHWC layout: [batch, in_height, in_width, in_channel].

    filter : relax.Expr
        3-D filter tensor with shape [in_channel, filter_height, filter_width].

    strides : Union[int, Tuple[int, int]]
        Stride size for the filter window, or [stride_h, stride_w].

    padding : Union[int, Tuple[int, ...]]
        Padding size. Can be an int for symmetric padding,
        [pad_top, pad_left, pad_bottom, pad_right], or 2-tuple
        [pad_h, pad_w] which is expanded to 4-tuple.

    dilations : Union[int, Tuple[int, int]]
        Dilation size for the filter, or [dilation_h, dilation_w].

    data_layout : str
        Layout of the input data. Supported: "NCHW", "NHWC".

    out_dtype : Optional[str | tvm_ffi.dtype]
        Specifies the output data type. If None, uses the input dtype.

    Returns
    -------
    result : relax.Expr
        The dilated output tensor.
    """
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilations, int):
        dilations = (dilations, dilations)

    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif isinstance(padding, tuple | list):
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        elif len(padding) == 4:
            padding = tuple(padding)
        else:
            raise ValueError(f"padding must be int, 2-tuple, or 4-tuple, got {len(padding)}-tuple")
    else:
        raise TypeError(f"padding must be int, tuple, or list, got {type(padding)}")

    return _Call(
        "relax.image.dilation2d",
        [input, filter],
        attrs=_make_attrs(
            "relax.attrs.Dilation2DAttrs",
            strides=strides,
            padding=padding,
            dilations=dilations,
            data_layout=data_layout,
            out_dtype=(out_dtype.dtype if isinstance(out_dtype, _PrimType) else out_dtype),
        ),
        ty=ty,
        loc=loc,
    )  # type: ignore
