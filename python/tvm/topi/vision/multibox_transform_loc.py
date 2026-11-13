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
"""Multibox location transform (SSD / TFLite DetectionPostProcess decode)."""

import tvm_tirx
from tvm import te, topi


def multibox_transform_loc(
    cls_pred,
    loc_pred,
    anchor,
    variances,
    clip=False,
    threshold=0.0,
    keep_background=True,
    apply_softmax=True,
):
    """TFLite ``DecodeCenterSizeBoxes``-style decode + score post-process.

    Inputs must match Relax op contracts: ``cls_pred [B,C,N]``, ``loc_pred [B,4*N]``,
    ``anchor [1,N,4]`` ltrb; per-anchor loc order ``(x,y,w,h)`` after yxhw→xywh reorder.

    Parameters
    ----------
    cls_pred : te.Tensor
        ``[B, C, N]`` logits or scores.
    loc_pred : te.Tensor
        ``[B, 4*N]`` encodings ``(x,y,w,h)`` per anchor.
    anchor : te.Tensor
        ``[1, N, 4]`` ``(left, top, right, bottom)``.
    variances : tuple of 4 float
        ``(x,y,w,h)`` = ``1/x_scale, 1/y_scale, 1/w_scale, 1/h_scale`` (TFLite).
    clip : bool
        Clip ``ymin,xmin,ymax,xmax`` to ``[0,1]``.
    threshold : float
        ``scores *= (scores >= threshold)`` after optional softmax.
    keep_background : bool
        If False: ``scores[:,0,:] = 0``.
    apply_softmax : bool
        Apply softmax across classes before thresholding.

    Returns
    -------
    boxes : te.Tensor
        ``[B, N, 4]`` as ``(ymin,xmin,ymax,xmax)``.
    scores : te.Tensor
        ``[B, C, N]`` scores after optional softmax, threshold, and background masking.
    """
    dtype = cls_pred.dtype
    B = cls_pred.shape[0]
    num_anchors = cls_pred.shape[2]
    loc_reshaped = topi.reshape(loc_pred, [B, num_anchors, 4])

    vx = tvm_tirx.tirx.const(float(variances[0]), dtype)
    vy = tvm_tirx.tirx.const(float(variances[1]), dtype)
    vw = tvm_tirx.tirx.const(float(variances[2]), dtype)
    vh = tvm_tirx.tirx.const(float(variances[3]), dtype)
    half = tvm_tirx.tirx.const(0.5, dtype)
    zero = tvm_tirx.tirx.const(0.0, dtype)
    one = tvm_tirx.tirx.const(1.0, dtype)
    th = tvm_tirx.tirx.const(float(threshold), dtype)

    def decode_bbox(b, a, k):
        left = anchor[0, a, 0]
        top = anchor[0, a, 1]
        right = anchor[0, a, 2]
        bottom = anchor[0, a, 3]
        ay = (top + bottom) * half
        ax = (left + right) * half
        ah = bottom - top
        aw = right - left
        ex = loc_reshaped[b, a, 0]
        ey = loc_reshaped[b, a, 1]
        ew = loc_reshaped[b, a, 2]
        eh = loc_reshaped[b, a, 3]
        ycenter = ey * vy * ah + ay
        xcenter = ex * vx * aw + ax
        half_h = half * te.exp(eh * vh) * ah
        half_w = half * te.exp(ew * vw) * aw
        ymin = ycenter - half_h
        xmin = xcenter - half_w
        ymax = ycenter + half_h
        xmax = xcenter + half_w
        if clip:
            ymin = te.max(zero, te.min(one, ymin))
            xmin = te.max(zero, te.min(one, xmin))
            ymax = te.max(zero, te.min(one, ymax))
            xmax = te.max(zero, te.min(one, xmax))
        return tvm_tirx.tirx.Select(
            k == 0,
            ymin,
            tvm_tirx.tirx.Select(k == 1, xmin, tvm_tirx.tirx.Select(k == 2, ymax, xmax)),
        )

    boxes = te.compute((B, num_anchors, 4), decode_bbox, name="multibox_boxes")

    scores = topi.nn.softmax(cls_pred, axis=1) if apply_softmax else cls_pred
    mask = topi.cast(topi.greater_equal(scores, th), dtype)
    scores = scores * mask
    if not keep_background:

        def zero_bg(b, c, n):
            s = scores[b, c, n]
            return te.if_then_else(c == 0, zero, s)

        scores = te.compute(scores.shape, zero_bg, name="multibox_scores_bg")

    return [boxes, scores]
