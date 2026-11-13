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
"""affine_grid and grid_sample operator"""

from tvm import te
from tvm_tirx import tirx


def affine_grid(data, target_shape, align_corners=True):
    """affine_grid operator that generates a 2D or 3D sampling grid.

    This operation is described in https://arxiv.org/pdf/1506.02025.pdf. It generates a uniform
    sampling grid within the target shape and normalizes it to [-1, 1]. The provided affine
    transformation is then applied on the sampling grid.

    Parameters
    ----------
    data : tvm.Tensor
        3-D with shape [batch, 2, 3] for 2D or [batch, 3, 4] for 3D. The affine matrix.

    target_shape: list/tuple of int
        Specifies the output spatial shape (H, W) for 2D or (D, H, W) for 3D.

    align_corners : bool
        If True, normalized coordinates map to corner pixels; if False, to pixel centers
        (the PyTorch / ONNX default).

    Returns
    -------
    Output : tvm.Tensor
        [batch, 2, H, W] for 2D or [batch, 3, D, H, W] for 3D.
    """
    assert len(target_shape) in (2, 3)
    if align_corners:
        assert all(s > 1 for s in target_shape), (
            "target spatial dims should be greater than 1 when align_corners is True"
        )

    dtype = data.dtype
    if align_corners:
        starts = [tirx.const(-1.0, dtype=dtype) for _ in target_shape]
        steps = [tirx.const((2.0 - 1e-7) / (s - 1), dtype=dtype) for s in target_shape]
    else:
        # Pixel centers: coordinate i maps to (2 * i + 1) / size - 1.
        starts = [tirx.const(-1.0 + 1.0 / s, dtype=dtype) for s in target_shape]
        steps = [tirx.const(2.0 / s, dtype=dtype) for s in target_shape]

    ndim = len(target_shape)

    def _compute(n, dim, *coords):
        # coords are ordered slowest-to-fastest (e.g. (k, i, j)); the affine matrix
        # columns are fastest-to-slowest (x, y, z), so index it in reverse.
        val = data[n, dim, ndim]  # translation column
        for r in range(ndim):
            coord = starts[r] + coords[r] * steps[r]
            val += data[n, dim, ndim - 1 - r] * coord
        return val

    oshape = (data.shape[0], ndim, *target_shape)
    return te.compute(oshape, _compute, tag="affine_grid")


def _grid_sample_2d(
    data, grid, method="bilinear", layout="NCHW", padding_mode="zeros", align_corners=True
):
    """Applies bilinear/nearest/bicubic sampling to input feature map.

    Given :math:`data` and :math:`grid` assuming NCHW layout, then the output is computed by

    .. math::

        x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
        y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
        output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})

    :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and
    :math:`G()` denotes the interpolation method.

    The out-boundary points will be padded with zeros if padding_mode is "zeros", or
    border pixel value if padding_mode is "border", or
    inner pixel value if padding_mode is "reflection".

    The left-top corner (-1, -1) and right-bottom corner (1, 1) in grid will be map to
    (0, 0) and (h - 1, w - 1) of data if align_corners is "True", or
    (-0.5, -0.5) and (h - 0.5, w - 0.5) of data if align_corners is "False".

    The shape of the output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).

    The operator assumes that :math:`grid` has been normalized to [-1, 1].

    grid_sample often cooperates with affine_grid which generates sampling grids for grid_sample.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    grid : tvm.Tensor
        4-D with shape [batch, 2, out_height, out_width]

    method : str
        The interpolation method "nearest", "bilinear", "bicubic" are supported.

    layout : str
        The layout of input data and the output.

    padding_mode : str
        The padding mode for outside grid values, "zeros", "border", "reflection" are supported.

    align_corners: bool
        Geometrically, we consider the pixels of the input as squares rather than points.
        If set to "True", the extrema ("-1" and "1") are considered as referring
        to the center points of the input corner pixels. If set to "False", they
        are instead considered as referring to the corner points of the input corner
        pixels, making the sampling more resolution agnostic.

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, in_channel, out_height, out_width]
    """

    assert method in ("bilinear", "nearest", "bicubic"), f"{method} is not supported"
    assert padding_mode in ("zeros", "border", "reflection"), f"{padding_mode} is not supported"
    assert layout == "NCHW", f"{layout} is not supported"

    batch, in_channel, in_height, in_width = data.shape
    out_height, out_width = grid.shape[2:]

    def _get_pixel_value(n, c, h, w):
        return te.if_then_else(
            te.all(h >= 0, w >= 0, h < in_height, w < in_width),
            data[n, c, h, w],
            tirx.const(0.0, dtype=data.dtype),
        )

    def _unnormalize(h, w):
        if align_corners:
            y = (h + 1) * (in_height - 1) / 2
            x = (w + 1) * (in_width - 1) / 2
        else:
            y = -0.5 + (h + 1) * in_height / 2
            x = -0.5 + (w + 1) * in_width / 2
        return (y, x)

    def _clip_coordinates(x, size):
        return te.min(te.max(x, 0), size - 1)

    def _compute_source_index(n, h, w):
        y = grid[n, 1, h, w]
        x = grid[n, 0, h, w]
        y, x = _unnormalize(y, x)

        if padding_mode == "reflection":
            y = _reflect_coordinates(y, in_height)
            x = _reflect_coordinates(x, in_width)
            y = _clip_coordinates(y, in_height)
            x = _clip_coordinates(x, in_width)
        elif padding_mode == "border":
            y = _clip_coordinates(y, in_height)
            x = _clip_coordinates(x, in_width)

        return (y, x)

    def _reflect_coordinates(x, size):
        def __refelection(x, size, corner_start):
            def __reflect(index, size, corner_start):
                index_align_corner = te.abs(corner_start - index)
                size_times = te.truncdiv(index_align_corner.astype("int32"), size).astype("int32")
                t = tirx.Mod(size_times, 2)
                extra = index_align_corner - size_times * size
                return tirx.if_then_else(
                    tirx.EQ(t, 0), extra + corner_start, size - extra + corner_start
                )

            return tirx.if_then_else(
                tirx.all(x >= corner_start, x <= size + corner_start),
                x,
                __reflect(x, size, corner_start),
            )

        if align_corners:
            new_x = __refelection(x, size - 1, 0)
        else:
            new_x = __refelection(x, size, -0.5)
        return new_x

    def _bilinear_sample(n, c, h, w):
        y, x = _compute_source_index(n, h, w)
        y0 = te.floor(y).astype("int32")
        x0 = te.floor(x).astype("int32")
        y1 = y0 + tirx.const(1, "int32")
        x1 = x0 + tirx.const(1, "int32")

        return (
            _get_pixel_value(n, c, y0, x0) * (1.0 - (y - y0)) * (1.0 - (x - x0))
            + _get_pixel_value(n, c, y0, x1) * (1.0 - (y - y0)) * (x - x0)
            + _get_pixel_value(n, c, y1, x0) * (y - y0) * (1.0 - (x - x0))
            + _get_pixel_value(n, c, y1, x1) * (y - y0) * (x - x0)
        )

    def _nearest_sample(n, c, h, w):
        y, x = _compute_source_index(n, h, w)
        y_new = te.nearbyint(y).astype("int32")
        x_new = te.nearbyint(x).astype("int32")

        return _get_pixel_value(n, c, y_new, x_new)

    def _bicubic_sample(n, c, h, w):
        A = -0.75  # -0.75 is used in pytorch, it maybe different in other frameworks

        def cubic_weight_1(fraction):
            return ((A + 2) * fraction - (A + 3)) * fraction * fraction + 1

        def cubic_weight_2(fraction):
            return ((A * fraction - 5 * A) * fraction + 8 * A) * fraction - 4 * A

        def cubic_interp_1d(pixel_0, pixel_1, pixel_2, pixel_3, fraction):
            weights = [0] * 4
            weights[0] = cubic_weight_2(fraction + 1)
            weights[1] = cubic_weight_1(fraction)
            weights[2] = cubic_weight_1(1 - fraction)
            weights[3] = cubic_weight_2(2 - fraction)
            return (
                pixel_0 * weights[0]
                + pixel_1 * weights[1]
                + pixel_2 * weights[2]
                + pixel_3 * weights[3]
            )

        y = grid[n, 1, h, w]
        x = grid[n, 0, h, w]
        y, x = _unnormalize(y, x)
        y_floor = te.floor(y).astype("int32")
        x_floor = te.floor(x).astype("int32")
        y_fraction = y - y_floor
        x_fraction = x - x_floor

        coefficients = [0] * 4

        for i in range(4):
            y_ = y_floor - 1 + i
            x_0 = x_floor - 1
            x_1 = x_floor + 0
            x_2 = x_floor + 1
            x_3 = x_floor + 2

            if padding_mode == "border":
                y_ = _clip_coordinates(y_, in_height).astype("int32")
                x_0 = _clip_coordinates(x_0, in_width).astype("int32")
                x_1 = _clip_coordinates(x_1, in_width).astype("int32")
                x_2 = _clip_coordinates(x_2, in_width).astype("int32")
                x_3 = _clip_coordinates(x_3, in_width).astype("int32")

            elif padding_mode == "reflection":
                y_ = _reflect_coordinates(y_, in_height)
                x_0 = _reflect_coordinates(x_0, in_width)
                x_1 = _reflect_coordinates(x_1, in_width)
                x_2 = _reflect_coordinates(x_2, in_width)
                x_3 = _reflect_coordinates(x_3, in_width)

                y_ = _clip_coordinates(y_, in_height).astype("int32")
                x_0 = _clip_coordinates(x_0, in_width).astype("int32")
                x_1 = _clip_coordinates(x_1, in_width).astype("int32")
                x_2 = _clip_coordinates(x_2, in_width).astype("int32")
                x_3 = _clip_coordinates(x_3, in_width).astype("int32")

            coefficients[i] = cubic_interp_1d(
                _get_pixel_value(n, c, y_, x_0),
                _get_pixel_value(n, c, y_, x_1),
                _get_pixel_value(n, c, y_, x_2),
                _get_pixel_value(n, c, y_, x_3),
                x_fraction,
            )

        return cubic_interp_1d(
            coefficients[0], coefficients[1], coefficients[2], coefficients[3], y_fraction
        )

    if method == "bilinear":
        interpolation = _bilinear_sample
    elif method == "nearest":
        interpolation = _nearest_sample
    else:  # method == "bicubic"
        interpolation = _bicubic_sample

    return te.compute((batch, in_channel, out_height, out_width), interpolation, tag="grid_sample")


def _grid_sample_3d(
    data, grid, method="bilinear", layout="NCDHW", padding_mode="zeros", align_corners=True
):
    """Applies bilinear/nearest sampling to input feature map.

    Given :math:`data` and :math:`grid` assuming NCDHW layout, then the output is computed by

    .. math::

        x_{src} = grid[batch, 0, z_{dst}, y_{dst}, x_{dst}] \\
        y_{src} = grid[batch, 1, z_{dst}, y_{dst}, x_{dst}] \\
        z_{src} = grid[batch, 2, z_{dst}, y_{dst}, x_{dst}] \\
        output[batch, channel, z_{src}, y_{dst}, x_{dst}]
        = G(data[batch, channel, z_{src}, y_{src}, x_{src})

    :math:`x_{dst}`, :math:`y_{dst}`, :math:`z_{dst}` enumerate all spatial locations
    in :math:`output`, and :math:`G()` denotes the interpolation method.

    The out-boundary points will be padded with zeros if padding_mode is "zeros", or
    border pixel value if padding_mode is "border", or
    inner pixel value if padding_mode is "reflection".

    The left-top corner (-1, -1, -1) and right-bottom corner (1, 1, 1) in grid will be map to
    (0, 0, 0) and (d - 1, h - 1, w - 1) of data if align_corners is "True", or
    (-0.5, -0.5, -0.5) and (d - 0.5, h - 0.5, w - 0.5) of data if align_corners is "False".

    The shape of the output will be
    (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3], grid.shape[4]).

    The operator assumes that :math:`grid` has been normalized to [-1, 1].

    grid_sample often cooperates with affine_grid which generates sampling grids for grid_sample.

    Parameters
    ----------
    data : tvm.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    grid : tvm.Tensor
        5-D with shape [batch, 3, out_depth, out_height, out_width]

    method : str
        The interpolation method "nearest", "bilinear"("trilinear") are supported.

    layout : str
        The layout of input data and the output.

    padding_mode : str
        The padding mode for outside grid values, "zeros", "border", "reflection" are supported.

    align_corners: bool
        Geometrically, we consider the pixels of the input as squares rather than points.
        If set to "True", the extrema ("-1" and "1") are considered as referring
        to the center points of the input corner pixels. If set to "False", they
        are instead considered as referring to the corner points of the input corner
        pixels, making the sampling more resolution agnostic.

    Returns
    -------
    Output : tvm.Tensor
        5-D with shape [batch, in_channel, out_depth, out_height, out_width]
    """

    assert method in ("bilinear", "nearest"), f"{method} is not supported"
    assert padding_mode in ("zeros", "border", "reflection"), f"{padding_mode} is not supported"
    assert layout == "NCDHW", f"{layout} is not supported"

    batch, in_channel, in_depth, in_height, in_width = data.shape
    out_depth, out_height, out_width = grid.shape[2:]

    def _get_pixel_value(n, c, d, h, w):
        return te.if_then_else(
            te.all(d >= 0, h >= 0, w >= 0, d < in_depth, h < in_height, w < in_width),
            data[n, c, d, h, w],
            tirx.const(0.0, dtype=data.dtype),
        )

    def _compute_source_index(n, d, h, w):
        z = grid[n, 2, d, h, w]
        y = grid[n, 1, d, h, w]
        x = grid[n, 0, d, h, w]

        if align_corners:
            z = (z + 1) * (in_depth - 1) / 2
            y = (y + 1) * (in_height - 1) / 2
            x = (x + 1) * (in_width - 1) / 2
        else:
            z = -0.5 + (z + 1) * in_depth / 2
            y = -0.5 + (y + 1) * in_height / 2
            x = -0.5 + (x + 1) * in_width / 2

        if padding_mode == "reflection":
            z = _reflect_coordinates(z, in_depth)
            y = _reflect_coordinates(y, in_height)
            x = _reflect_coordinates(x, in_width)
            z = _clip_coordinates(z, in_depth)
            y = _clip_coordinates(y, in_height)
            x = _clip_coordinates(x, in_width)
        elif padding_mode == "border":
            z = _clip_coordinates(z, in_depth)
            y = _clip_coordinates(y, in_height)
            x = _clip_coordinates(x, in_width)

        return (z, y, x)

    def _clip_coordinates(x, size):
        return te.min(te.max(x, 0), size - 1)

    def _reflect_coordinates(x, size):
        def __refelection(x, size, corner_start):
            def __reflect(index, size, corner_start):
                index_align_corner = te.abs(corner_start - index)
                size_times = te.truncdiv(index_align_corner.astype("int32"), size).astype("int32")
                t = tirx.Mod(size_times, 2)
                extra = index_align_corner - size_times * size
                return tirx.if_then_else(
                    tirx.EQ(t, 0), extra + corner_start, size - extra + corner_start
                )

            return tirx.if_then_else(
                tirx.all(x >= corner_start, x <= size + corner_start),
                x,
                __reflect(x, size, corner_start),
            )

        if align_corners:
            return __refelection(x, size - 1, 0)
        return __refelection(x, size, -0.5)

    def _trilinear_sample(n, c, d, h, w):
        z, y, x = _compute_source_index(n, d, h, w)
        z0 = te.floor(z).astype("int32")
        y0 = te.floor(y).astype("int32")
        x0 = te.floor(x).astype("int32")
        z1 = z0 + tirx.const(1, "int32")
        y1 = y0 + tirx.const(1, "int32")
        x1 = x0 + tirx.const(1, "int32")

        return (
            _get_pixel_value(n, c, z0, y0, x0) * (1 - (x - x0)) * (1 - (y - y0)) * (1 - (z - z0))
            + _get_pixel_value(n, c, z0, y0, x1) * (x - x0) * (1 - (y - y0)) * (1 - (z - z0))
            + _get_pixel_value(n, c, z1, y1, x0) * (1 - (x - x0)) * (y - y0) * (z - z0)
            + _get_pixel_value(n, c, z1, y1, x1) * (x - x0) * (y - y0) * (z - z0)
            + _get_pixel_value(n, c, z0, y1, x0) * (1 - (x - x0)) * (y - y0) * (1 - (z - z0))
            + _get_pixel_value(n, c, z1, y0, x1) * (x - x0) * (1 - (y - y0)) * (z - z0)
            + _get_pixel_value(n, c, z1, y0, x0) * (1 - (x - x0)) * (1 - (y - y0)) * (z - z0)
            + _get_pixel_value(n, c, z0, y1, x1) * (x - x0) * (y - y0) * (1 - (z - z0))
        )

    def _nearest_sample(n, c, d, h, w):
        z, y, x = _compute_source_index(n, d, h, w)
        z_new = te.nearbyint(z).astype("int32")
        y_new = te.nearbyint(y).astype("int32")
        x_new = te.nearbyint(x).astype("int32")
        return _get_pixel_value(n, c, z_new, y_new, x_new)

    if method == "bilinear":
        interpolation = _trilinear_sample
    else:  # method == "nearest"
        interpolation = _nearest_sample

    return te.compute(
        (batch, in_channel, out_depth, out_height, out_width), interpolation, tag="grid_sample"
    )


def grid_sample(
    data, grid, method="bilinear", layout="NCHW", padding_mode="zeros", align_corners=True
):
    """Applies grid sampling to input feature map.

    Given :math:`data` and :math:`grid`, then for 4-D the output is computed by

    .. math::

        x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
        y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
        output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src}])

    :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and
    :math:`G()` denotes the interpolation function.

    The out-boundary points will be padded with zeros if padding_mode is "zeros", or
    border pixel value if padding_mode is "border", or
    inner pixel value if padding_mode is "reflection".

    The left-top corner (-1, -1) and right-bottom corner (1, 1) in grid will be map to
    (0, 0) and (h - 1, w - 1) of data if align_corners is "True", or
    (-0.5, -0.5) and (h - 0.5, w - 0.5) of data if align_corners is "False".

    The shape of the output will be
    4-D (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]), or
    5-D (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3], grid.shape[4]).

    The operator assumes that :math:`grid` has been normalized to [-1, 1].

    grid_sample often cooperates with affine_grid which generates sampling grids for grid_sample.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width], or
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    grid : tvm.Tensor
        4-D with shape [batch, 2, out_height, out_width], or
        5-D with shape [batch, 3, out_depth, out_height, out_width]

    method : str
        The interpolation method, 4-D "nearest", "bilinear", "bicubic" and
        5-D "nearest", "bilinear"("trilinear") are supported.

    layout : str
        The layout of input data and the output.

    padding_mode : str
        The padding mode for outside grid values, "zeros", "border", "reflection" are supported.

    align_corners: bool
        Geometrically, we consider the pixels of the input as squares rather than points.
        If set to "True", the extrema ("-1" and "1") are considered as referring
        to the center points of the input corner pixels. If set to "False", they
        are instead considered as referring to the corner points of the input corner
        pixels, making the sampling more resolution agnostic.

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, in_channel, out_height, out_width], or
        5-D with shape [batch, in_channel, out_depth, out_height, out_width]
    """

    if len(layout) == 4:
        compute = _grid_sample_2d
    elif len(layout) == 5:
        compute = _grid_sample_3d
    else:
        msg = f"layout {layout} is not supported"
        raise ValueError(msg)

    return compute(data, grid, method, layout, padding_mode, align_corners)
