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
"""Non-maximum suppression operators."""

import tvm_ffi

from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC as _UNKNOWN_LOC
from tvm_tirx.ir.location import Location as _Location


@tvm_ffi.register_object("relax.attrs.AllClassNonMaximumSuppressionAttrs")
class AllClassNonMaximumSuppressionAttrs(Attrs):
    """Attributes for vision.all_class_non_max_suppression"""


def all_class_non_max_suppression(
    boxes,
    scores,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    output_format="onnx",
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
):
    """Non-maximum suppression operator for object detection, corresponding to ONNX
    NonMaxSuppression and TensorFlow combined_non_max_suppression.
    NMS is performed for each class separately.

    Parameters
    ----------
    boxes : relax.Expr
        3-D tensor with shape (batch_size, num_boxes, 4)
    scores: relax.Expr
        3-D tensor with shape (batch_size, num_classes, num_boxes)
    max_output_boxes_per_class : relax.Expr
        The maxinum number of output selected boxes per class
    iou_threshold : relax.Expr
        IoU test threshold
    score_threshold : relax.Expr
        Score threshold to filter out low score boxes early
    output_format : str, optional
        "onnx" or "tensorflow", see below.

    Returns
    -------
    out : relax.Expr
        If `output_format` is "onnx", the output is two tensors. The first is `indices` of size
        `(batch_size * num_class* num_boxes , 3)` and the second is a scalar tensor
        `num_total_detection` of shape `(1,)` representing the total number of selected
        boxes. The three values in `indices` encode batch, class, and box indices.
        Rows of `indices` are ordered such that selected boxes from batch 0, class 0 come
        first, in descending of scores, followed by boxes from batch 0, class 1 etc.
        The output uses dynamic_strided_slice to trim to only valid detections,
        so the first tensor has shape (num_total_detection, 3) containing only valid rows.

        If `output_format` is "tensorflow", the output is three tensors, the first
        is `indices` of size `(batch_size, num_class * num_boxes , 2)`, the second is `scores` of
        size `(batch_size, num_class * num_boxes)`, and the third is `num_total_detection` of size
        `(batch_size,)` representing the total number of selected boxes per batch. The two values
        in `indices` encode class and box indices. Of num_class * num_boxes boxes in `indices` at
        batch b, only the first `num_total_detection[b]` entries are valid. The second axis of
        `indices` and `scores` are sorted within each class by box scores, but not across classes.
        So the box indices and scores for the class 0 come first in a sorted order, followed by
        the class 1 etc.
    """
    return _Call(
        "relax.vision.all_class_non_max_suppression",
        [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
        attrs=_make_attrs(
            "relax.attrs.AllClassNonMaximumSuppressionAttrs", output_format=output_format
        ),
        loc=loc,
        ty=ty,
    )


@tvm_ffi.register_object("relax.attrs.GetValidCountsAttrs")
class GetValidCountsAttrs(Attrs):
    """Attributes for vision.get_valid_counts"""


def get_valid_counts(
    data,
    score_threshold=0,
    id_index=0,
    score_index=1,
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : relax.Expr
        3-D tensor with shape [batch_size, num_anchors, elem_length].

    score_threshold : float, optional
        Lower limit of score for valid bounding boxes.

    id_index : int, optional
        Index of the class categories. Set to ``-1`` to disable the class-id check.

    score_index : int, optional
        Index of the scores/confidence of boxes.

    Returns
    -------
    out : relax.Expr
        A tuple ``(valid_count, out_tensor, out_indices)`` where ``valid_count``
        has shape ``[batch_size]``, ``out_tensor`` has shape
        ``[batch_size, num_anchors, elem_length]``, and ``out_indices`` has shape
        ``[batch_size, num_anchors]``.
    """
    return _Call(
        "relax.vision.get_valid_counts",
        [data],
        attrs=_make_attrs(
            "relax.attrs.GetValidCountsAttrs",
            score_threshold=score_threshold,
            id_index=id_index,
            score_index=score_index,
        ),
        loc=loc,
        ty=ty,
    )


@tvm_ffi.register_object("relax.attrs.NonMaximumSuppressionAttrs")
class NonMaximumSuppressionAttrs(Attrs):
    """Attributes for vision.non_max_suppression"""


def non_max_suppression(
    data,
    valid_count,
    indices,
    max_output_size=-1,
    iou_threshold=0.5,
    force_suppress=False,
    top_k=-1,
    coord_start=2,
    score_index=1,
    id_index=0,
    return_indices=True,
    invalid_to_bottom=False,
    soft_nms_sigma=0.0,
    score_threshold=0.0,
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : relax.Expr
        3-D tensor with shape [batch_size, num_anchors, elem_length].

    valid_count : relax.Expr
        1-D tensor for valid number of boxes.

    indices : relax.Expr
        2-D tensor with shape [batch_size, num_anchors].

    max_output_size : int, optional
        Max number of output valid boxes, -1 for no limit.

    iou_threshold : float, optional
        Non-maximum suppression IoU threshold.

    force_suppress : bool, optional
        Whether to suppress all detections regardless of class_id. When
        ``id_index`` is ``-1``, all valid boxes are treated as belonging to the
        same class, so this flag has the same effect as ``True``.

    top_k : int, optional
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : int, optional
        Start index of the consecutive 4 coordinates.

    score_index : int, optional
        Index of the scores/confidence of boxes.

    id_index : int, optional
        Index of the class categories. Set to ``-1`` to suppress boxes across
        all classes.

    return_indices : bool, optional
        Whether to return box indices in input data.

    invalid_to_bottom : bool, optional
        Whether to move valid bounding boxes to the top of the returned tensor.
        This option only affects the ``return_indices=False`` path.

    soft_nms_sigma : float, optional
        Sigma for soft-NMS Gaussian penalty. When ``0.0`` (default), standard
        hard NMS is used. Positive values decay overlapping box scores instead
        of suppressing them outright.

    score_threshold : float, optional
        Post-decay minimum score for a box to remain eligible during soft-NMS.
        Only used when ``soft_nms_sigma > 0``. This is distinct from
        ``get_valid_counts.score_threshold``, which filters boxes before NMS.
        Defaults to ``0.0``.

    Returns
    -------
    out : relax.Expr
        The return tuple shape depends on ``soft_nms_sigma``.
        If ``return_indices`` is ``True`` and ``soft_nms_sigma`` is ``0.0``,
        returns a 2-tuple ``(box_indices, valid_box_count)`` with shapes
        ``[batch_size, num_anchors]`` and ``[batch_size, 1]``.
        If ``return_indices`` is ``True`` and ``soft_nms_sigma > 0``,
        returns a 3-tuple ``(out_data, box_indices, valid_box_count)`` where
        decayed ``out_data`` is prepended and has the same shape as the input
        data.
        Otherwise returns the modified data tensor.
    """
    return _Call(
        "relax.vision.non_max_suppression",
        [data, valid_count, indices],
        attrs=_make_attrs(
            "relax.attrs.NonMaximumSuppressionAttrs",
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            force_suppress=force_suppress,
            top_k=top_k,
            coord_start=coord_start,
            score_index=score_index,
            id_index=id_index,
            return_indices=return_indices,
            invalid_to_bottom=invalid_to_bottom,
            soft_nms_sigma=soft_nms_sigma,
            score_threshold=score_threshold,
        ),
        loc=loc,
        ty=ty,
    )
