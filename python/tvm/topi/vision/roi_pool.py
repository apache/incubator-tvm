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
"""ROI Pool operator"""

import tvm_tirx
from tvm import te


def roi_pool_nchw(data, rois, pooled_size, spatial_scale):
    """ROI pool operator in NCHW layout."""
    _, channel, height, width = data.shape
    num_roi, _ = rois.shape

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    zero = tvm_tirx.tirx.const(0.0, data.dtype)
    roi_dtype = rois.dtype

    neg_inf = tvm_tirx.tirx.const(float("-inf"), data.dtype)

    def _round_away(x):
        # ONNX MaxRoiPool spec uses ties-away-from-zero rounding for coordinate
        # mapping (matching std::round semantics in the reference implementation).
        # Use floor(x + 0.5) to be explicit and independent of tir.round semantics.
        half = tvm_tirx.tirx.const(0.5, roi_dtype)
        return te.floor(x + half)

    def _bin_bounds(i, ph, pw):
        roi = rois[i]
        roi_start_w = _round_away(roi[1] * spatial_scale).astype("int32")
        roi_start_h = _round_away(roi[2] * spatial_scale).astype("int32")
        roi_end_w = _round_away(roi[3] * spatial_scale).astype("int32")
        roi_end_h = _round_away(roi[4] * spatial_scale).astype("int32")

        roi_h = te.max(roi_end_h - roi_start_h + 1, tvm_tirx.tirx.const(1, "int32"))
        roi_w = te.max(roi_end_w - roi_start_w + 1, tvm_tirx.tirx.const(1, "int32"))

        bin_h = tvm_tirx.tirx.Cast(roi_dtype, roi_h) / tvm_tirx.tirx.const(
            float(pooled_size_h), roi_dtype
        )
        bin_w = tvm_tirx.tirx.Cast(roi_dtype, roi_w) / tvm_tirx.tirx.const(
            float(pooled_size_w), roi_dtype
        )

        hstart = te.floor(tvm_tirx.tirx.Cast(roi_dtype, ph) * bin_h).astype("int32")
        wstart = te.floor(tvm_tirx.tirx.Cast(roi_dtype, pw) * bin_w).astype("int32")
        hend = te.ceil(tvm_tirx.tirx.Cast(roi_dtype, ph + 1) * bin_h).astype("int32")
        wend = te.ceil(tvm_tirx.tirx.Cast(roi_dtype, pw + 1) * bin_w).astype("int32")

        hstart = te.min(te.max(hstart + roi_start_h, 0), height)
        hend = te.min(te.max(hend + roi_start_h, 0), height)
        wstart = te.min(te.max(wstart + roi_start_w, 0), width)
        wend = te.min(te.max(wend + roi_start_w, 0), width)
        return hstart, hend, wstart, wend

    def _sample(i, c, ph, pw):
        roi = rois[i]
        batch_index = roi[0].astype("int32")
        hstart, hend, wstart, wend = _bin_bounds(i, ph, pw)
        valid = tvm_tirx.tirx.all(hstart <= rh, rh < hend, wstart <= rw, rw < wend)
        return tvm_tirx.tirx.if_then_else(valid, data[batch_index, c, rh, rw], neg_inf)

    def _is_empty(i, ph, pw):
        hstart, hend, wstart, wend = _bin_bounds(i, ph, pw)
        return tvm_tirx.tirx.any(hend <= hstart, wend <= wstart)

    rh = te.reduce_axis((0, height), name="rh")
    rw = te.reduce_axis((0, width), name="rw")
    pooled = te.compute(
        (num_roi, channel, pooled_size_h, pooled_size_w),
        lambda i, c, ph, pw: te.max(_sample(i, c, ph, pw), axis=[rh, rw]),
        tag="pool,roi_pool_nchw",
    )

    return te.compute(
        (num_roi, channel, pooled_size_h, pooled_size_w),
        lambda i, c, ph, pw: tvm_tirx.tirx.if_then_else(
            _is_empty(i, ph, pw), zero, pooled[i, c, ph, pw]
        ),
    )


def roi_pool(data, rois, pooled_size, spatial_scale, layout="NCHW"):
    """ROI pool operator."""
    if layout == "NCHW":
        return roi_pool_nchw(data, rois, pooled_size, spatial_scale)
    raise ValueError(f"Unsupported layout for roi_pool: {layout}")
