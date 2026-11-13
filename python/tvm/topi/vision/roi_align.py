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
"""ROI Align operator"""

import tvm_tirx
from tvm import te

from ..cpp.utils import bilinear_sample_nchw, bilinear_sample_nhwc


def _sample_common(
    i,
    c,
    ph,
    pw,
    rois,
    pooled_size_h,
    pooled_size_w,
    spatial_scale,
    sample_ratio,
    aligned,
    dtype,
    avg_mode,
    bilinear_func,
):
    roi = rois[i]
    batch_index = roi[0].astype("int32")
    roi_start_w = roi[1] * spatial_scale
    roi_start_h = roi[2] * spatial_scale
    roi_end_w = roi[3] * spatial_scale
    roi_end_h = roi[4] * spatial_scale

    if aligned:
        roi_h = roi_end_h - roi_start_h
        roi_w = roi_end_w - roi_start_w
    else:
        roi_h = te.max(roi_end_h - roi_start_h, tvm_tirx.tirx.const(1.0, dtype))
        roi_w = te.max(roi_end_w - roi_start_w, tvm_tirx.tirx.const(1.0, dtype))

    pooled_size_h_const = tvm_tirx.tirx.const(pooled_size_h, dtype)
    pooled_size_w_const = tvm_tirx.tirx.const(pooled_size_w, dtype)
    bin_h = roi_h / pooled_size_h_const
    bin_w = roi_w / pooled_size_w_const

    if sample_ratio > 0:
        roi_bin_grid_h = tvm_tirx.tirx.const(sample_ratio, "int32")
        roi_bin_grid_w = tvm_tirx.tirx.const(sample_ratio, "int32")
    else:
        roi_bin_grid_h = te.ceil(roi_h / pooled_size_h_const).astype("int32")
        roi_bin_grid_w = te.ceil(roi_w / pooled_size_w_const).astype("int32")

    count = roi_bin_grid_h * roi_bin_grid_w
    rh = te.reduce_axis((0, roi_bin_grid_h), name="rh")
    rw = te.reduce_axis((0, roi_bin_grid_w), name="rw")
    roi_start_h = roi_start_h + tvm_tirx.tirx.Cast(dtype, ph) * bin_h
    roi_start_w = roi_start_w + tvm_tirx.tirx.Cast(dtype, pw) * bin_w

    def sample_value(rh_idx, rw_idx):
        return bilinear_func(
            batch_index,
            c,
            roi_start_h
            + (tvm_tirx.tirx.Cast(dtype, rh_idx) + tvm_tirx.tirx.const(0.5, dtype))
            * bin_h
            / tvm_tirx.tirx.Cast(dtype, roi_bin_grid_h),
            roi_start_w
            + (tvm_tirx.tirx.Cast(dtype, rw_idx) + tvm_tirx.tirx.const(0.5, dtype))
            * bin_w
            / tvm_tirx.tirx.Cast(dtype, roi_bin_grid_w),
        )

    if avg_mode:
        return te.sum(
            sample_value(rh, rw) / tvm_tirx.tirx.Cast(dtype, count),
            axis=[rh, rw],
        )
    return te.max(sample_value(rh, rw), axis=[rh, rw])


def roi_align_nchw(data, rois, pooled_size, spatial_scale, mode, sample_ratio=-1, aligned=False):
    """ROI align operator in NCHW layout."""
    avg_mode = mode in (b"avg", "avg", 0)
    max_mode = mode in (b"max", "max", 1)
    assert avg_mode or max_mode, "Mode must be avg or max. Please pass in a valid mode."

    _, channel, height, width = data.shape
    num_roi, _ = rois.shape
    dtype = rois.dtype

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    height_f = tvm_tirx.tirx.Cast(dtype, height)
    width_f = tvm_tirx.tirx.Cast(dtype, width)
    zero = tvm_tirx.tirx.const(0.0, data.dtype)

    def _bilinear(n, c, y, x):
        outside = tvm_tirx.tirx.any(y < -1.0, x < -1.0, y > height_f, x > width_f)
        y = te.min(te.max(y, 0.0), tvm_tirx.tirx.Cast(dtype, height - 1))
        x = te.min(te.max(x, 0.0), tvm_tirx.tirx.Cast(dtype, width - 1))
        val = bilinear_sample_nchw(data, (n, c, y, x), height - 1, width - 1)
        return tvm_tirx.tirx.if_then_else(outside, zero, val)

    return te.compute(
        (num_roi, channel, pooled_size_h, pooled_size_w),
        lambda i, c, ph, pw: _sample_common(
            i,
            c,
            ph,
            pw,
            rois,
            pooled_size_h,
            pooled_size_w,
            spatial_scale,
            sample_ratio,
            aligned,
            dtype,
            avg_mode,
            _bilinear,
        ),
        tag="pool,roi_align_nchw",
    )


def roi_align_nhwc(data, rois, pooled_size, spatial_scale, mode, sample_ratio=-1, aligned=False):
    """ROI align operator in NHWC layout."""
    avg_mode = mode in (b"avg", "avg", 0)
    max_mode = mode in (b"max", "max", 1)
    assert avg_mode or max_mode, "Mode must be avg or max. Please pass in a valid mode."

    _, height, width, channel = data.shape
    num_roi, _ = rois.shape
    dtype = rois.dtype

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    height_f = tvm_tirx.tirx.Cast(dtype, height)
    width_f = tvm_tirx.tirx.Cast(dtype, width)
    zero = tvm_tirx.tirx.const(0.0, data.dtype)

    def _bilinear(n, c, y, x):
        outside = tvm_tirx.tirx.any(y < -1.0, x < -1.0, y > height_f, x > width_f)
        y = te.min(te.max(y, 0.0), tvm_tirx.tirx.Cast(dtype, height - 1))
        x = te.min(te.max(x, 0.0), tvm_tirx.tirx.Cast(dtype, width - 1))
        val = bilinear_sample_nhwc(data, (n, y, x, c), height - 1, width - 1)
        return tvm_tirx.tirx.if_then_else(outside, zero, val)

    return te.compute(
        (num_roi, pooled_size_h, pooled_size_w, channel),
        lambda i, ph, pw, c: _sample_common(
            i,
            c,
            ph,
            pw,
            rois,
            pooled_size_h,
            pooled_size_w,
            spatial_scale,
            sample_ratio,
            aligned,
            dtype,
            avg_mode,
            _bilinear,
        ),
        tag="pool,roi_align_nhwc",
    )


def roi_align(
    data,
    rois,
    pooled_size,
    spatial_scale,
    mode="avg",
    sample_ratio=-1,
    aligned=False,
    layout="NCHW",
):
    """ROI align operator."""
    if layout == "NCHW":
        return roi_align_nchw(data, rois, pooled_size, spatial_scale, mode, sample_ratio, aligned)
    if layout == "NHWC":
        return roi_align_nhwc(data, rois, pooled_size, spatial_scale, mode, sample_ratio, aligned)
    raise ValueError(f"Unsupported layout for roi_align: {layout}")
