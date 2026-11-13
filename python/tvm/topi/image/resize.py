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
# ruff: noqa: E741, F821
"""TVM operator input resize compute."""

import tvm
import tvm_tirx
from tvm import te
from tvm.topi.utils import nchw_pack_layout, nchw_xc_layout

from .. import tag


def can_convert_multiply_to_intdiv(origin_size, scaled_size):
    """Check whether can convert multiplication to division"""
    # Only support IntImm type
    if not isinstance(scaled_size, tvm_tirx.ir.IntImm):
        return False

    div = scaled_size / origin_size.astype("float")
    if div.value % 1 != 0:
        return False
    epsilon = 1e-5
    check = 1 / (epsilon * origin_size + epsilon)
    if div > check:
        return False
    return True


def get_1d_indices(indices, layout="NCW"):
    """Get 1d indices"""
    (cc, inum, ic) = (0, 0, 0)
    if layout == "NWC":
        n, x, c = indices
        cc = None
    elif layout == "NCW":
        n, c, x = indices
        cc = None
    elif ncw_pack_layout(layout):
        n, c, x, inum, ic = indices
    else:
        # else must be NCHWxc
        assert ncw_xc_layout(layout)
        n, c, x, cc = indices

    return n, c, x, cc, inum, ic


def get_2d_indices(indices, layout="NCHW"):
    """Get 2d indices"""
    (cc, inum, ic) = (0, 0, 0)
    if layout == "NHWC":
        n, y, x, c = indices
        cc = None
    elif layout == "NCHW":
        n, c, y, x = indices
        cc = None
    elif nchw_pack_layout(layout):
        n, c, y, x, inum, ic = indices
    else:
        # else must be NCHWxc
        assert nchw_xc_layout(layout)
        n, c, y, x, cc = indices

    return n, c, y, x, cc, inum, ic


def get_3d_indices(indices, layout="NCDHW"):
    """Get 3d indices"""
    if layout == "NDHWC":
        n, z, y, x, c = indices
        cc = None
    elif layout == "NCDHW":
        n, c, z, y, x = indices
        cc = None
    else:
        n, c, z, y, x, cc = indices

    return n, c, z, y, x, cc


def get_1d_pixel(data, layout, image_width, n, c, x, cc, ib, ic):
    """Get 1d pixel"""
    x = tvm.te.max(tvm.te.min(x, image_width - 1), 0)
    if layout == "NWC":
        return data(n, x, c).astype("float")
    if layout == "NCW":
        return data(n, c, x).astype("float")
    if ncw_pack_layout(layout):
        return data(n, c, x, ib, ic).astype("float")

    # else must be NCHWxc
    assert ncw_xc_layout(layout)
    return data(n, c, x, cc).astype("float")


def get_2d_pixel(data, layout, image_height, image_width, n, c, y, x, cc, ib, ic):
    """Get 2d pixel"""
    y = tvm.te.max(tvm.te.min(y, image_height - 1), 0)
    x = tvm.te.max(tvm.te.min(x, image_width - 1), 0)
    if layout == "NHWC":
        return data(n, y, x, c).astype("float")
    if layout == "NCHW":
        return data(n, c, y, x).astype("float")
    if nchw_pack_layout(layout):
        return data(n, c, y, x, ib, ic).astype("float")

    # else must be NCHWxc
    assert nchw_xc_layout(layout)
    return data(n, c, y, x, cc).astype("float")


def get_3d_pixel(data, layout, image_depth, image_height, image_width, n, c, z, y, x, cc):
    """Get 3d pixel"""
    z = tvm.te.max(tvm.te.min(z, image_depth - 1), 0)
    y = tvm.te.max(tvm.te.min(y, image_height - 1), 0)
    x = tvm.te.max(tvm.te.min(x, image_width - 1), 0)
    if layout == "NDHWC":
        return data(n, z, y, x, c).astype("float")
    if layout == "NCDHW":
        return data(n, c, z, y, x).astype("float")
    # else must be NCDHWxc
    return data(n, c, z, y, x, cc).astype("float")


def get_inx(
    x,
    image_width,
    target_width,
    coordinate_transformation_mode,
    start_x=0,
    end_x=-1,
    use_int_div=False,
):
    """Infer input x from output x with various coordinate transformation methods"""
    scale_x = te.div(image_width.astype("float"), target_width.astype("float"))
    if coordinate_transformation_mode == "half_pixel":
        in_x = (x + 0.5) * scale_x - 0.5
    elif coordinate_transformation_mode == "align_corners":
        in_x = (image_width - 1).astype("float") / (target_width - 1) * x
    elif coordinate_transformation_mode == "asymmetric":
        if use_int_div:
            in_x = te.div(x, te.div(target_width, image_width))
        else:
            in_x = scale_x * x
    elif coordinate_transformation_mode == "pytorch_half_pixel":
        in_x = te.if_then_else(target_width > 1, (x + 0.5) * scale_x - 0.5, 0.0)
    elif coordinate_transformation_mode == "tf_half_pixel_for_nn":
        in_x = (x + 0.5) * scale_x
    elif coordinate_transformation_mode == "tf_crop_and_resize":
        in_x = te.if_then_else(
            target_width > 1,
            start_x * (image_width - 1)
            + x * (end_x - start_x) * (image_width - 1).astype("float") / (target_width - 1),
            0.5 * (start_x + end_x) * (image_width - 1),
        )
    else:
        raise ValueError(
            f"Unsupported coordinate_transformation_mode: {coordinate_transformation_mode}"
        )
    return in_x


def get_closest_index(in_x, rounding_method, boxes, use_int_div=False):
    """get the closest index to a value based on a certain rounding method"""
    if use_int_div:
        closest_x_index = in_x.astype("int32")
        return closest_x_index

    if rounding_method == "round" or boxes is not None:
        closest_x_index = te.round(in_x).astype("int32")
    elif rounding_method == "round_prefer_floor":
        closest_x_index = te.ceil(in_x - 0.5).astype("int32")
    elif rounding_method == "round_prefer_ceil":
        closest_x_index = te.floor(in_x + 0.5).astype("int32")
    elif rounding_method == "floor":
        # Add epsilon to floor to prevent gpu rounding errors.
        epsilon = 1e-5
        closest_x_index = te.floor(in_x + epsilon).astype("int32")
    elif rounding_method == "ceil":
        # Subract epsilon from ceil to prevent gpu rounding errors.
        epsilon = 1e-5
        closest_x_index = te.ceil(in_x - epsilon).astype("int32")
    else:
        raise ValueError(f"Unknown rounding method: {rounding_method}")
    return closest_x_index


def _lerp(A, B, t):
    """Perform Linear interpolation in 1D"""
    return A * (1.0 - t) + B * t


def _cubic_spline_weights(t, alpha):
    """create cubic spline weights in 1D"""
    t2 = t * t
    t3 = t * t * t
    w1 = alpha * (t3 - 2 * t2 + t)
    w2 = (alpha + 2) * t3 - (3 + alpha) * t2 + 1
    w3 = -(alpha + 2) * t3 + (3 + 2 * alpha) * t2 - alpha * t
    w4 = -alpha * t3 + alpha * t2
    return [w1, w2, w3, w4]


def _cubic_kernel(inputs, w):
    """perform cubic interpolation in 1D"""
    return sum([a_i * w_i for a_i, w_i in zip(inputs, w)])


def _resize_1d(
    indices,
    data,
    roi,
    image_width,
    target_width,
    boxes=None,
    box_indices=None,
    method=None,
    extrapolation_value=0.0,
    layout="NCW",
    coordinate_transformation_mode="align_corners",
    rounding_method="",
    alpha=-0.5,
    exclude_outside=0,
    out_dtype=None,
):
    """Perform resize operation on the data with selected method and options.

    Parameters
    ----------
    indices : tuple
        The indices of input data

    data : tvm.te.Tensor
        inputs is a 3-D tensor with shape
        [batch, channel, in_width]
        or  [batch, in_width, channel]

    roi: Tuple of Float or Expr
        The region of interest for cropping the input image. Expected to be of
        size 2, and format [start_w, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    image_width : integer
        Input image width

    target_width : integer
        The target resized image width

    boxes : tvm.te.Tensor, optional
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    box_indices : tvm.te.Tensor, optional
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    layout: string, optional
        "NCW", "NWC", or "NCWc".

    method: string, optional
        method of interpolation ("nearest", "linear", "bicubic")

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: string, optional
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    alpha: float, optional
        Bicubic spline coefficient

    exclude_outside: bool, optional:
        Exclude values outside the image fdor bicubic interpolation

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : out_dtype
        The computed result with type out_dtype
    """

    def _cast_output(value, data_dtype="float32", out_dtype=None):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data_dtype
        return value.astype(dtype)

    n, c, x, cc, inum, ic = get_1d_indices(indices, layout)
    box_idx = box_indices(n) if box_indices is not None else n
    if boxes is not None:
        # TODO(mbrookhart): Find an example of this
        raise NotImplementedError("resize1d with image boxes not yet implemented")
    in_x = get_inx(x, image_width, target_width, coordinate_transformation_mode, roi[0], roi[1])

    if method == "nearest_neighbor":
        if rounding_method == "":
            if coordinate_transformation_mode == "align_corners":
                rounding_method = "round"
            else:
                rounding_method = "floor"

        closest_x_index = get_closest_index(in_x, rounding_method, boxes)

        value = get_1d_pixel(data, layout, image_width, box_idx, c, closest_x_index, cc, inum, ic)
    elif method == "linear":
        x_int = te.floor(in_x).astype("int32")

        x_lerp = in_x - x_int

        p = [0 for i in range(2)]
        for i in range(2):
            p[i] = get_1d_pixel(data, layout, image_width, box_idx, c, x_int + i, cc, inum, ic)

        value = _lerp(*p, x_lerp)

    elif method == "cubic":
        xint = te.floor(in_x).astype("int32")
        xfract = in_x - te.floor(in_x)

        # Get the surrounding values
        p = [0 for i in range(4)]
        for i in range(4):
            p[i] = get_1d_pixel(data, layout, image_width, box_idx, c, xint + i - 1, cc, inum, ic)

        wx = _cubic_spline_weights(xfract, alpha)
        if exclude_outside:
            for i in range(4):
                wx[i] = te.if_then_else(
                    te.any(xint - 1 + i < 0, xint + i > image_width), 0.0, wx[i]
                )
            sum_wx = sum(wx)
            wx = [w / sum_wx for w in wx]
        value = _cubic_kernel(p, wx)

    else:
        raise ValueError("Unknown resize method:", method)

    if coordinate_transformation_mode == "tf_crop_and_resize":
        # use extrapolation_value if in_x is out of boundary
        value = tvm_tirx.tirx.if_then_else(
            in_x < 0,
            extrapolation_value,
            tvm_tirx.tirx.if_then_else(in_x > image_width - 1, extrapolation_value, value),
        )
    return _cast_output(value, data.dtype, out_dtype=out_dtype)


def resize1d(
    data,
    roi,
    size,
    layout="NCW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    bicubic_alpha=-0.75,
    bicubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
    output_shape=None,
):
    """Perform resize operation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        inputs is a 3-D tensor with shape
        [batch, channel in_width]
        or  [batch in_width, channel]

    roi: Tuple of Float or Expr
        The region of interest for cropping the input image. Expected to be of
        size 2, and format [start_w, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    size: Tuple
        Output resolution scale to

    layout: string, optional
        "NCW", "NWC", or "NCWc".

    coordinate_transformation_mode: string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.
        Available options are "half_pixel", "align_corners" and "asymmetric".

    method: string, optional
        method of interpolation ("nearest", "linear", "bicubic")

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method:
        Method for rounding coordinate locations

    bicubic_alpha: float, optional
        Bicubic spline coefficient

    bicubic_exclude: bool, optional:
        Exclude values outside the image fdor bicubic interpolation

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    output_shape: tvm_ffi.Array, optional
        Shape to return. If left None will be inferred
        (If shape is determined dynamically, pass out_dtype.shape as output_shape)

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, chananel, in_width*scale]
        or [batch, in_width*scale, channel]
        or 5-D with shape [batch, channel-major, in_width*scale, channel-minor]
    """
    method = method.lower()
    if layout == "NWC":
        in_n, in_w, in_c = data.shape
        if output_shape is None:
            output_shape = [in_n, size[0], in_c]
    elif layout == "NCW":
        in_n, in_c, in_w = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0]]
    elif ncw_pack_layout(layout):  # for NCWinic
        in_n, in_c, in_w, in_inum, in_ic = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], in_inum, in_ic]
    elif ncw_xc_layout(layout):  # for NCWxc
        in_n, in_c, in_w, in_cc = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], in_cc]
    else:
        raise ValueError(f"{layout} layout is not supported.")

    if isinstance(size, tuple):
        size = list(size)

    for i in range(1):
        if isinstance(size[i], int):
            size[i] = tvm_tirx.ir.IntImm("int32", size[i])

    def compute_func(*indices):
        return _resize_1d(
            indices,
            data,
            roi,
            in_w,
            size[0],
            method=method,
            layout=layout,
            coordinate_transformation_mode=coordinate_transformation_mode,
            rounding_method=rounding_method,
            alpha=bicubic_alpha,
            exclude_outside=bicubic_exclude,
            extrapolation_value=extrapolation_value,
            out_dtype=out_dtype,
        )

    return te.compute(output_shape, compute_func, name="resize", tag=tag.INJECTIVE)


def _resize_2d(
    indices,
    data,
    roi,
    image_height,
    image_width,
    target_height,
    target_width,
    boxes=None,
    box_indices=None,
    method=None,
    extrapolation_value=0.0,
    layout="NCHW",
    coordinate_transformation_mode="align_corners",
    rounding_method="",
    alpha=-0.5,
    exclude_outside=0,
    out_dtype=None,
):
    """Perform resize operation on the data with selected method and options.

    Parameters
    ----------
    indices : tuple
        The indices of input data

    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    roi: Tuple of Float or Expr
        The region of interest for cropping the input image. Expected to be of
        size 4, and format [start_h, start_w, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    image_height : integer
        Input image height

    image_width : integer
        Input image width

    target_height : integer
        The target resized image height

    target_width : integer
        The target resized image width

    boxes : tvm.te.Tensor, optional
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    method: string, optional
        method of interpolation ("nearest", "linear", "bicubic")

    box_indices : tvm.te.Tensor, optional
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: string, optional
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    alpha: float, optional
        Bicubic spline coefficient

    exclude_outside: bool, optional:
        Exclude values outside the image fdor bicubic interpolation

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : out_dtype
        The computed result with type out_dtype
    """

    def _cast_output(value, data_dtype="float32", out_dtype=None):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data_dtype
        return value.astype(dtype)

    height_use_int_div = False
    width_use_int_div = False
    if method == "nearest_neighbor" and coordinate_transformation_mode == "asymmetric":
        if rounding_method == "floor" or rounding_method == "":
            height_use_int_div = can_convert_multiply_to_intdiv(image_height, target_height)
            width_use_int_div = can_convert_multiply_to_intdiv(image_width, target_width)

    n, c, y, x, cc, inum, ic = get_2d_indices(indices, layout)
    box_idx = box_indices(n) if box_indices is not None else n
    if boxes is not None:
        y1, x1 = boxes(n, 0), boxes(n, 1)
        y2, x2 = boxes(n, 2), boxes(n, 3)

        in_h = (image_height - 1) * (y2 - y1)
        in_w = (image_width - 1) * (x2 - x1)
        h_scale = in_h.astype("float") / (target_height - 1)
        w_scale = in_w.astype("float") / (target_width - 1)

        in_y = y1 * (image_height - 1) + h_scale * y
        in_x = x1 * (image_width - 1) + w_scale * x
    else:
        in_x = get_inx(
            x,
            image_width,
            target_width,
            coordinate_transformation_mode,
            roi[1],
            roi[3],
            width_use_int_div,
        )
        in_y = get_inx(
            y,
            image_height,
            target_height,
            coordinate_transformation_mode,
            roi[0],
            roi[2],
            height_use_int_div,
        )

    if method == "nearest_neighbor":
        if rounding_method == "":
            if coordinate_transformation_mode == "align_corners":
                rounding_method = "round"
            else:
                rounding_method = "floor"

        closest_x_index = get_closest_index(in_x, rounding_method, boxes, width_use_int_div)
        closest_y_index = get_closest_index(in_y, rounding_method, boxes, height_use_int_div)

        value = get_2d_pixel(
            data,
            layout,
            image_height,
            image_width,
            box_idx,
            c,
            closest_y_index,
            closest_x_index,
            cc,
            inum,
            ic,
        )
    elif method == "linear":
        y_int = te.floor(in_y).astype("int32")
        x_int = te.floor(in_x).astype("int32")

        y_lerp = in_y - y_int
        x_lerp = in_x - x_int

        p = [[0 for i in range(2)] for j in range(2)]
        for j in range(2):
            for i in range(2):
                p[j][i] = get_2d_pixel(
                    data,
                    layout,
                    image_height,
                    image_width,
                    box_idx,
                    c,
                    y_int + j,
                    x_int + i,
                    cc,
                    inum,
                    ic,
                )

        top = _lerp(*p[0], x_lerp)
        bottom = _lerp(*p[1], x_lerp)
        value = _lerp(top, bottom, y_lerp)

    elif method == "cubic":
        xint = te.floor(in_x).astype("int32")
        xfract = in_x - te.floor(in_x)

        yint = te.floor(in_y).astype("int32")
        yfract = in_y - te.floor(in_y)

        # Get the surrounding values
        p = [[0 for i in range(4)] for j in range(4)]
        for j in range(4):
            for i in range(4):
                p[j][i] = get_2d_pixel(
                    data,
                    layout,
                    image_height,
                    image_width,
                    box_idx,
                    c,
                    yint + j - 1,
                    xint + i - 1,
                    cc,
                    inum,
                    ic,
                )

        wx = _cubic_spline_weights(xfract, alpha)
        wy = _cubic_spline_weights(yfract, alpha)
        if exclude_outside:
            for i in range(4):
                wx[i] = te.if_then_else(
                    te.any(xint - 1 + i < 0, xint + i > image_width), 0.0, wx[i]
                )
                wy[i] = te.if_then_else(
                    te.any(yint - 1 + i < 0, yint + i > image_height), 0.0, wy[i]
                )
            sum_wx = sum(wx)
            sum_wy = sum(wy)
            wx = [w / sum_wx for w in wx]
            wy = [w / sum_wy for w in wy]
        col0 = _cubic_kernel(p[0], wx)
        col1 = _cubic_kernel(p[1], wx)
        col2 = _cubic_kernel(p[2], wx)
        col3 = _cubic_kernel(p[3], wx)
        value = _cubic_kernel([col0, col1, col2, col3], wy)

    else:
        raise ValueError("Unknown resize method:", method)

    if coordinate_transformation_mode == "tf_crop_and_resize":
        out = tvm_tirx.tirx.if_then_else(
            in_y < 0,
            extrapolation_value,
            tvm_tirx.tirx.if_then_else(in_y > image_height - 1, extrapolation_value, value),
        )
        # use extrapolation_value if in_x is out of boundary
        value = tvm_tirx.tirx.if_then_else(
            in_x < 0,
            extrapolation_value,
            tvm_tirx.tirx.if_then_else(in_x > image_width - 1, extrapolation_value, out),
        )
    return _cast_output(value, data.dtype, out_dtype=out_dtype)


def resize2d(
    data,
    roi,
    size,
    layout="NCHW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    bicubic_alpha=-0.75,
    bicubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
    output_shape=None,
):
    """Perform resize operation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    roi: Tuple of Float or Expr
        The region of interest for cropping the input image. Expected to be of
        size 4, and format [start_h, start_w, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    size: Tuple
        Output resolution scale to

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    method: string, optional
        method of interpolation ("nearest", "linear", "bicubic")

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method:
        Method for rounding coordinate locations

    bicubic_alpha: float, optional
        Bicubic spline coefficient

    bicubic_exclude: bool, optional:
        Exclude values outside the image fdor bicubic interpolation

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    output_shape: tvm_ffi.Array, optional
        Shape to return. If left None will be inferred
        (If shape is determined dynamically, pass out_dtype.shape as output_shape)

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
        or 5-D with shape [batch, channel-major, in_height*scale, in_width*scale, channel-minor]
    """
    method = method.lower()
    if layout == "NHWC":
        in_n, in_h, in_w, in_c = data.shape
        if output_shape is None:
            output_shape = [in_n, size[0], size[1], in_c]
    elif layout == "NCHW":
        in_n, in_c, in_h, in_w = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1]]
    elif nchw_pack_layout(layout):  # for NCHWinic
        in_n, in_c, in_h, in_w, in_inum, in_ic = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1], in_inum, in_ic]
    elif nchw_xc_layout(layout):  # for NCHWxc
        in_n, in_c, in_h, in_w, in_cc = data.shape
        if output_shape is None:
            output_shape = [in_n, in_c, size[0], size[1], in_cc]
    else:
        raise ValueError(f"{layout} layout is not supported.")

    if isinstance(size, tuple):
        size = list(size)

    for i in range(2):
        if isinstance(size[i], int):
            size[i] = tvm_tirx.ir.IntImm("int32", size[i])

    def compute_func(*indices):
        return _resize_2d(
            indices,
            data,
            roi,
            in_h,
            in_w,
            size[0],
            size[1],
            method=method,
            layout=layout,
            coordinate_transformation_mode=coordinate_transformation_mode,
            rounding_method=rounding_method,
            alpha=bicubic_alpha,
            exclude_outside=bicubic_exclude,
            extrapolation_value=extrapolation_value,
            out_dtype=out_dtype,
        )

    return te.compute(output_shape, compute_func, name="resize", tag=tag.INJECTIVE)


def crop_and_resize(
    data,
    boxes,
    box_indices,
    crop_size,
    layout="NCHW",
    method="bilinear",
    extrapolation_value=None,
    out_dtype=None,
):
    """Perform crop and resize operation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    boxes : tvm.te.Tensor
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    box_indices : tvm.te.Tensor
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    crop_size : Tuple
        The target size of each box.

    layout : string, optional
        "NCHW", "NHWC"

    method : {"bilinear", "nearest_neighbor"}
        Method to be used for resizing.

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    out_dtype : string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [num_boxes, channel, crop_height, crop_width]
        or [num_boxes, crop_height, crop_width, channel]
    """
    method = method.lower()
    target_h = crop_size[0]
    target_w = crop_size[1]
    if layout == "NHWC":
        output_shape = [box_indices.shape[0], crop_size[0], crop_size[1], data.shape[3]]
        image_h = data.shape[1].astype("int32")
        image_w = data.shape[2].astype("int32")
    elif layout == "NCHW":
        output_shape = [box_indices.shape[0], data.shape[1], crop_size[0], crop_size[1]]
        image_h = data.shape[2].astype("int32")
        image_w = data.shape[3].astype("int32")
    elif layout.startswith("NCHW"):  # for NCHWxc
        output_shape = [
            box_indices.shape[0],
            data.shape[1],
            crop_size[0],
            crop_size[1],
            data.shape[4],
        ]
        image_h = data.shape[2].astype("int32")
        image_w = data.shape[3].astype("int32")
    else:
        raise ValueError(f"{layout} layout is not supported.")
    if method == "bilinear":
        method = "linear"

    def compute_func(*indices):
        return _resize_2d(
            indices,
            data,
            [0.0] * 4,
            image_h,
            image_w,
            target_h,
            target_w,
            boxes,
            box_indices,
            method=method,
            extrapolation_value=extrapolation_value,
            layout=layout,
            coordinate_transformation_mode="tf_crop_and_resize",
            out_dtype=out_dtype,
        )

    return te.compute(output_shape, compute_func, name="crop_and_resize", tag=tag.INJECTIVE)


def _resize_3d(
    indices,
    data,
    roi,
    image_depth,
    image_height,
    image_width,
    target_depth,
    target_height,
    target_width,
    boxes=None,
    box_indices=None,
    method=None,
    extrapolation_value=0.0,
    layout="NCHW",
    coordinate_transformation_mode="align_corners",
    rounding_method="",
    alpha=-0.5,
    exclude_outside=0,
    out_dtype=None,
):
    """Perform resize operation on the data with selected method and options.

    Parameters
    ----------
    indices : tuple
        The indices of input data

    data : tvm.te.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    roi: Tuple of Float or Expr
        The region of interest for cropping the input image. Expected to be of
        size 6, and format [start_d, start_h, start_w, end_d, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    image_depth : integer
        Input image depth

    image_height : integer
        Input image height

    image_width : integer
        Input image width

    target_depth : integer
        The target resized image depth

    target_height : integer
        The target resized image height

    target_width : integer
        The target resized image width

    boxes : tvm.te.Tensor, optional
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    box_indices : tvm.te.Tensor, optional
        A 1-D tensor of shape [num_boxes], box_indices[i] specifies the data that
        the i-th box refers to.

    method: string, optional
        method of interpolation ("nearest", "linear", "bicubic")

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: string, optional
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    alpha: float, optional
        Bicubic spline coefficient

    exclude_oiutside: bool, optional:
        Exclude values outside the image fdor bicubic interpolation

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : out_dtype
        The computed result with type out_dtype
    """

    def _cast_output(value, data_dtype="float32", out_dtype=None):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data_dtype
        return value.astype(dtype)

    n, c, z, y, x, cc = get_3d_indices(indices, layout)
    box_idx = box_indices(n) if box_indices is not None else n
    if boxes is not None:
        # TODO(mbrookhart): Find an example of this
        raise NotImplementedError("resize1d with image boxes not yet implemented")
    in_z = get_inx(z, image_depth, target_depth, coordinate_transformation_mode, roi[2], roi[5])
    in_y = get_inx(y, image_height, target_height, coordinate_transformation_mode, roi[1], roi[4])
    in_x = get_inx(x, image_width, target_width, coordinate_transformation_mode, roi[0], roi[3])

    if method == "nearest_neighbor":
        if rounding_method == "":
            if coordinate_transformation_mode == "align_corners":
                rounding_method = "round"
            else:
                rounding_method = "floor"

        closest_z_index = get_closest_index(in_z, rounding_method, boxes)
        closest_y_index = get_closest_index(in_y, rounding_method, boxes)
        closest_x_index = get_closest_index(in_x, rounding_method, boxes)

        value = get_3d_pixel(
            data,
            layout,
            image_depth,
            image_height,
            image_width,
            box_idx,
            c,
            closest_z_index,
            closest_y_index,
            closest_x_index,
            cc,
        )
    elif method == "linear":
        z_int = te.floor(in_z).astype("int32")
        y_int = te.floor(in_y).astype("int32")
        x_int = te.floor(in_x).astype("int32")

        z_lerp = in_z - z_int
        y_lerp = in_y - y_int
        x_lerp = in_x - x_int

        p = [[[0 for i in range(2)] for j in range(2)] for k in range(2)]
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    p[k][j][i] = get_3d_pixel(
                        data,
                        layout,
                        image_depth,
                        image_height,
                        image_width,
                        box_idx,
                        c,
                        z_int + k,
                        y_int + j,
                        x_int + i,
                        cc,
                    )
        l = [[0 for i in range(2)] for j in range(2)]
        for j in range(2):
            for i in range(2):
                l[j][i] = _lerp(*p[j][i], x_lerp)

        top = _lerp(*l[0], y_lerp)
        bottom = _lerp(*l[1], y_lerp)
        value = _lerp(top, bottom, z_lerp)

    elif method == "cubic":
        zint = te.floor(in_z).astype("int32")
        zfract = in_z - te.floor(in_z)

        yint = te.floor(in_y).astype("int32")
        yfract = in_y - te.floor(in_y)

        xint = te.floor(in_x).astype("int32")
        xfract = in_x - te.floor(in_x)

        # Get the surrounding values
        p = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
        for k in range(4):
            for j in range(4):
                for i in range(4):
                    p[k][j][i] = get_3d_pixel(
                        data,
                        layout,
                        image_depth,
                        image_height,
                        image_width,
                        box_idx,
                        c,
                        zint + k - 1,
                        yint + j - 1,
                        xint + i - 1,
                        cc,
                    )

            wz = _cubic_spline_weights(zfract, alpha)
            wy = _cubic_spline_weights(yfract, alpha)
            wx = _cubic_spline_weights(xfract, alpha)
            if exclude_outside:
                for i in range(4):
                    wz[i] = te.if_then_else(
                        te.any(xint - 1 + i < 0, xint + i > image_height), 0.0, wx[i]
                    )
                    wy[i] = te.if_then_else(
                        te.any(yint - 1 + i < 0, yint + i > image_height), 0.0, wy[i]
                    )
                    wx[i] = te.if_then_else(
                        te.any(xint - 1 + i < 0, xint + i > image_width), 0.0, wx[i]
                    )
                sum_wz = sum(wz)
                sum_wy = sum(wy)
                sum_wx = sum(wx)
                wz = [w / sum_wz for w in wz]
                wy = [w / sum_wy for w in wy]
                wx = [w / sum_wx for w in wx]

            l = [[0 for i in range(4)] for j in range(4)]
            for j in range(4):
                for i in range(4):
                    l[j][i] = _cubic_kernel(p[j][i], wx)
            col0 = _cubic_kernel(l[0], wy)
            col1 = _cubic_kernel(l[1], wy)
            col2 = _cubic_kernel(l[2], wy)
            col3 = _cubic_kernel(l[3], wy)
            value = _cubic_kernel([col0, col1, col2, col3], wz)

    else:
        raise ValueError("Unknown resize method:", method)

    if coordinate_transformation_mode == "tf_crop_and_resize":
        out = tvm_tirx.tirx.if_then_else(
            in_z < 0,
            extrapolation_value,
            tvm_tirx.tirx.if_then_else(in_z > image_depth - 1, extrapolation_value, value),
        )
        out = tvm_tirx.tirx.if_then_else(
            in_y < 0,
            extrapolation_value,
            tvm_tirx.tirx.if_then_else(in_y > image_height - 1, extrapolation_value, value),
        )
        # use extrapolation_value if in_x is out of boundary
        value = tvm_tirx.tirx.if_then_else(
            in_x < 0,
            extrapolation_value,
            tvm_tirx.tirx.if_then_else(in_x > image_width - 1, extrapolation_value, out),
        )
    return _cast_output(value, data.dtype, out_dtype=out_dtype)


def resize3d(
    data,
    roi,
    size,
    layout="NCDHW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    bicubic_alpha=-0.75,
    bicubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
    output_shape=None,
):
    """Perform resize operation on the data.

    Parameters
    ----------
    data : tvm.te.Tensor
        inputs is a 5-D tensor with shape
        [batch, channel, in_depth, in_height, in_width]
        or  [batch, in_depth, in_height, in_width, channel]

    roi: Tuple of Float or Expr
        The region of interest for cropping the input image. Expected to be of
        size 6, and format [start_d, start_h, start_w, end_d, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    size: Tuple
        Output resolution scale to

    layout: string, optional
        "NCDHW", "NDHWC", or "NCDHWc".

    method: string, optional
        method of interpolation ("nearest", "linear", "bicubic")

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method:
        Method for rounding coordinate locations

    bicubic_alpha: float, optional
        Bicubic spline coefficient

    bicubic_exclude: bool, optional:
        Exclude values outside the image fdor bicubic interpolation

    extrapolation_value: float, optional
        Value used for extrapolation, when applicable.

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    output_shape: tvm_ffi.Array, optional
        Shape to return. If left None will be inferred
        (If shape is determined dynamically, pass out_dtype.shape as output_shape)

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, channel, in_depth*scale, in_height*scale, in_width*scale]
        or [batch, in_depth*scale, in_height*scale, in_width*scale, channel]
        or 5-D with shape
        [batch, channel-major, in_depth*scale, in_height*scale, in_width*scale, channel-minor]
    """

    method = method.lower()
    if layout == "NDHWC":
        in_n, in_d, in_h, in_w, in_c = data.shape
        output_shape = [in_n, size[0], size[1], size[2], in_c]
    elif layout == "NCDHW":
        in_n, in_c, in_d, in_h, in_w = data.shape
        output_shape = [in_n, in_c, size[0], size[1], size[2]]
    # Otherwise layout must be NCHWxc
    else:
        in_n, in_c, in_d, in_h, in_w, in_cc = data.shape
        output_shape = [in_n, in_c, size[0], size[1], size[2], in_cc]

    if isinstance(size, tuple):
        size = list(size)

    for i in range(3):
        if isinstance(size[i], int):
            size[i] = tvm_tirx.ir.IntImm("int32", size[i])

    def compute_func(*indices):
        return _resize_3d(
            indices,
            data,
            roi,
            in_d,
            in_h,
            in_w,
            size[0],
            size[1],
            size[2],
            method=method,
            layout=layout,
            coordinate_transformation_mode=coordinate_transformation_mode,
            rounding_method=rounding_method,
            alpha=bicubic_alpha,
            exclude_outside=bicubic_exclude,
            extrapolation_value=extrapolation_value,
            out_dtype=out_dtype,
        )

    return te.compute(output_shape, compute_func, name="resize", tag=tag.INJECTIVE)
