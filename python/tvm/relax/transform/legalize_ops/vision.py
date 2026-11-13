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
"""Default legalization function for vision network related operators."""

import tvm_tirx
from tvm import relax, te, topi
from tvm_tirx import tirx
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr, TupleGetItem
from .common import register_legalize


@register_legalize("relax.vision.all_class_non_max_suppression")
def _all_class_non_max_suppression(block_builder: BlockBuilder, call: Call) -> Expr:
    """Legalize all_class_non_max_suppression with dynamic output trimming.

    This implementation uses dynamic_strided_slice to trim the NMS output to only
    contain valid detections, improving memory efficiency and ONNX compatibility.

    Returns
    -------
    result : Expr
        The legalized NMS result.

        - For ONNX output format, returns a tuple of
          `(trimmed_indices, num_total_detections)`, where `trimmed_indices`
          contains only valid detection indices.
        - For TensorFlow output format, returns the TOPI result directly to
          preserve the `(selected_indices, selected_scores, num_detections)`
          layout expected by the Relax op.
    """
    boxes = call.args[0]
    scores = call.args[1]
    max_output_boxes_per_class = call.args[2]
    iou_threshold = call.args[3]
    score_threshold = call.args[4]
    output_format = call.attrs.output_format

    scores_shape = scores.ty.shape
    if len(scores_shape) == 3:
        _, _, num_boxes = scores_shape
    elif len(scores_shape) == 2:
        _, num_boxes = scores_shape
    else:
        raise ValueError(f"Unexpected scores shape: {scores_shape}")

    if isinstance(max_output_boxes_per_class, tvm_tirx.ir.GenericConst):
        max_boxes_val = int(max_output_boxes_per_class.value.numpy())
    else:
        max_boxes_val = int(num_boxes)

    # Get NMS result with fixed shape from TOPI
    nms_result = block_builder.call_te(
        topi.vision.all_class_non_max_suppression,
        boxes,
        scores,
        max_boxes_val,
        iou_threshold,
        score_threshold,
        output_format,
    )

    if output_format == "tensorflow":
        return nms_result

    selected_indices = block_builder.emit(TupleGetItem(nms_result, 0))
    num_total_detections = block_builder.emit(TupleGetItem(nms_result, 1))

    # Build slicing parameters using TE to avoid high-level Relax ops during legalization
    def build_begin():
        return te.compute((2,), lambda i: tirx.const(0, "int64"), name="begin")

    def build_strides():
        return te.compute((2,), lambda i: tirx.const(1, "int64"), name="strides")

    def build_end(count_tensor):
        # end = [count_tensor[0], 3]
        def compute_end(i):
            return tirx.if_then_else(
                i == 0,
                tirx.Cast("int64", count_tensor[0]),
                tirx.const(3, "int64"),
            )

        return te.compute((2,), compute_end, name="end")

    begin = block_builder.call_te(build_begin)
    strides = block_builder.call_te(build_strides)
    end = block_builder.call_te(build_end, num_total_detections)

    # Apply dynamic strided slice to trim to valid detections only
    trimmed_indices = block_builder.emit(
        relax.op.dynamic_strided_slice(selected_indices, begin, end, strides)
    )

    # Return trimmed indices along with num_total_detections for compatibility
    return relax.Tuple([trimmed_indices, num_total_detections])


@register_legalize("relax.vision.roi_align")
def _roi_align(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.vision.roi_align,
        call.args[0],
        call.args[1],
        pooled_size=call.attrs.pooled_size,
        spatial_scale=call.attrs.spatial_scale,
        mode=call.attrs.mode,
        sample_ratio=call.attrs.sample_ratio,
        aligned=call.attrs.aligned,
        layout=call.attrs.layout,
    )


@register_legalize("relax.vision.get_valid_counts")
def _get_valid_counts(block_builder: BlockBuilder, call: Call) -> Expr:
    return block_builder.call_te(
        topi.vision.get_valid_counts,
        call.args[0],
        score_threshold=call.attrs.score_threshold,
        id_index=call.attrs.id_index,
        score_index=call.attrs.score_index,
    )


@register_legalize("relax.vision.non_max_suppression")
def _non_max_suppression(block_builder: BlockBuilder, call: Call) -> Expr:
    return block_builder.call_te(
        topi.vision.non_max_suppression,
        call.args[0],
        call.args[1],
        call.args[2],
        max_output_size=call.attrs.max_output_size,
        iou_threshold=call.attrs.iou_threshold,
        force_suppress=call.attrs.force_suppress,
        top_k=call.attrs.top_k,
        coord_start=call.attrs.coord_start,
        score_index=call.attrs.score_index,
        id_index=call.attrs.id_index,
        return_indices=call.attrs.return_indices,
        invalid_to_bottom=call.attrs.invalid_to_bottom,
        soft_nms_sigma=call.attrs.soft_nms_sigma,
        score_threshold=call.attrs.score_threshold,
    )


@register_legalize("relax.vision.roi_pool")
def _roi_pool(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.vision.roi_pool,
        call.args[0],
        call.args[1],
        pooled_size=call.attrs.pooled_size,
        spatial_scale=call.attrs.spatial_scale,
        layout=call.attrs.layout,
    )


@register_legalize("relax.vision.multibox_transform_loc")
def _multibox_transform_loc(bb: BlockBuilder, call: Call) -> Expr:
    variances = tuple(float(x) for x in call.attrs.variances)

    def _te(cls_pred, loc_pred, anchor):
        return topi.vision.multibox_transform_loc(
            cls_pred,
            loc_pred,
            anchor,
            variances,
            clip=call.attrs.clip,
            threshold=call.attrs.threshold,
            keep_background=call.attrs.keep_background,
            apply_softmax=call.attrs.apply_softmax,
        )

    return bb.call_te(
        _te,
        call.args[0],
        call.args[1],
        call.args[2],
        function_name_hint="multibox_transform_loc",
    )
