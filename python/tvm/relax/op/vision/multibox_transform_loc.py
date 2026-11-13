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
"""Multibox location transform for object detection."""

import tvm_ffi

from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC as _UNKNOWN_LOC
from tvm_tirx.ir.location import Location as _Location


@tvm_ffi.register_object("relax.attrs.MultiboxTransformLocAttrs")
class MultiboxTransformLocAttrs(Attrs):
    """Attributes for vision.multibox_transform_loc"""


def multibox_transform_loc(
    cls_pred,
    loc_pred,
    anchor,
    clip=False,
    threshold=0.0,
    variances=(1.0, 1.0, 1.0, 1.0),
    keep_background=True,
    apply_softmax=True,
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
):
    """SSD / TFLite-style decode: priors + offsets → boxes; prepare class scores.

    Box decode follows TFLite ``DecodeCenterSizeBoxes``; expected tensor layout matches
    ``tflite_frontend.convert_detection_postprocess`` (loc reorder yxhw→xywh, anchor ltrb).

    Parameters
    ----------
    cls_pred : relax.Expr
        ``[B, C, N]`` class logits or scores.
    loc_pred : relax.Expr
        ``[B, 4*N]`` per-anchor encodings as ``(x,y,w,h)`` after reorder (see above).
    anchor : relax.Expr
        ``[1, N, 4]`` priors: ``(left, top, right, bottom)``.
    clip : bool
        If True, clip ``ymin,xmin,ymax,xmax`` to ``[0, 1]``.
    threshold : float
        After softmax, multiply scores by mask ``(score >= threshold)``.
    variances : tuple of 4 floats
        ``(x,y,w,h)`` = TFLite ``1/x_scale, 1/y_scale, 1/w_scale, 1/h_scale``.
        Use magnitudes consistent with the model: very large ``w``/``h`` entries scale the
        encoded height/width terms inside ``exp(...)`` and can overflow in float32/float16.
    keep_background : bool
        If False, set output scores at class index 0 to zero.
    apply_softmax : bool
        If True, apply softmax over the class axis before thresholding.

    Returns
    -------
    result : relax.Expr
        Tuple ``(boxes, scores)``: ``boxes`` is ``[B, N, 4]`` as ``(ymin,xmin,ymax,xmax)``;
        ``scores`` is ``[B, C, N]`` softmax, post-processed like the implementation.

    Notes
    -----
    **Shape/dtype (checked in ``FInferType`` when static):**

    - ``cls_pred``: 3-D; ``loc_pred``: 2-D; ``anchor``: 3-D.
    - ``cls_pred``, ``loc_pred``, ``anchor`` dtypes must match.
    - ``N = cls_pred.shape[2]``; ``loc_pred.shape[1] == 4*N``; ``anchor.shape == [1,N,4]``.
    - ``loc_pred.shape[1]`` must be divisible by 4.
    - ``cls_pred.shape[0]`` must equal ``loc_pred.shape[0]`` (batch).

    If ``cls_pred`` has **unknown** shape, inference only returns generic rank-3 tensor
    type for the two outputs; it does **not** verify ``4*N`` vs ``loc_pred`` or
    ``anchor.shape[1]`` vs ``N``, because ``N`` is not available statically. Other checks
    (ranks, dtypes, ``loc_pred.shape[1] % 4 == 0`` when known, batch match when both batch
    axes are known, etc.) still run where applicable.
    """
    return _Call(
        "relax.vision.multibox_transform_loc",
        [cls_pred, loc_pred, anchor],
        attrs=_make_attrs(
            "relax.attrs.MultiboxTransformLocAttrs",
            clip=clip,
            threshold=threshold,
            variances=variances,
            keep_background=keep_background,
            apply_softmax=apply_softmax,
        ),
        loc=loc,
        ty=ty,
    )
