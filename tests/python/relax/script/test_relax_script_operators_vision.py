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
"""Relax script operators vision."""

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx import IRModule


def _check(
    parsed: relax.Function | IRModule,
    expect: relax.Function | IRModule | None,
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm_tirx.script.from_source(
        test, extra_vars={"I": tvm_tirx.script.ir, "R": tvm.relax.script, "T": tvm_tirx.script.tirx}
    )
    tvm_tirx.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm_tirx.ir.assert_structural_equal(parsed, expect)


def test_all_class_non_max_suppression():
    @R.function
    def foo(
        boxes: R.Tensor((10, 5, 4), "float32"),
        scores: R.Tensor((10, 8, 5), "float32"),
        max_output_boxes_per_class: R.Tensor((), "int64"),
        iou_threshold: R.Tensor((), "float32"),
        score_threshold: R.Tensor((), "float32"),
    ) -> R.Tuple(R.Tensor((400, 3), "int64"), R.Tensor((1,), "int64")):
        gv: R.Tuple(R.Tensor((400, 3), "int64"), R.Tensor((1,), "int64")) = (
            R.vision.all_class_non_max_suppression(
                boxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                "onnx",
            )
        )
        return gv

    boxes = relax.Var("boxes", R.Tensor((10, 5, 4), "float32"))
    scores = relax.Var("scores", R.Tensor((10, 8, 5), "float32"))
    max_output_boxes_per_class = relax.Var("max_output_boxes_per_class", R.Tensor((), "int64"))
    iou_threshold = relax.Var("iou_threshold", R.Tensor((), "float32"))
    score_threshold = relax.Var("score_threshold", R.Tensor((), "float32"))

    bb = relax.BlockBuilder()
    with bb.function(
        "foo", [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    ):
        gv = bb.emit(
            relax.op.vision.all_class_non_max_suppression(
                boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_get_valid_counts():
    @R.function
    def foo(
        data: R.Tensor((10, 5, 6), "float32"),
    ) -> R.Tuple(
        R.Tensor((10,), "int32"),
        R.Tensor((10, 5, 6), "float32"),
        R.Tensor((10, 5), "int32"),
    ):
        gv: R.Tuple(
            R.Tensor((10,), "int32"),
            R.Tensor((10, 5, 6), "float32"),
            R.Tensor((10, 5), "int32"),
        ) = R.vision.get_valid_counts(data, score_threshold=0.5, id_index=0, score_index=1)
        return gv

    data = relax.Var("data", R.Tensor((10, 5, 6), "float32"))

    bb = relax.BlockBuilder()
    with bb.function("foo", [data]):
        gv = bb.emit(
            relax.op.vision.get_valid_counts(data, score_threshold=0.5, id_index=0, score_index=1)
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_non_max_suppression_return_indices():
    @R.function
    def foo(
        data: R.Tensor((2, 5, 6), "float32"),
        valid_count: R.Tensor((2,), "int32"),
        indices: R.Tensor((2, 5), "int32"),
    ) -> R.Tuple(R.Tensor((2, 5), "int32"), R.Tensor((2, 1), "int32")):
        gv: R.Tuple(R.Tensor((2, 5), "int32"), R.Tensor((2, 1), "int32")) = (
            R.vision.non_max_suppression(
                data,
                valid_count,
                indices,
                max_output_size=-1,
                iou_threshold=0.5,
                force_suppress=False,
                top_k=3,
                coord_start=2,
                score_index=1,
                id_index=0,
                return_indices=True,
                invalid_to_bottom=False,
                soft_nms_sigma=0.0,
                score_threshold=0.0,
            )
        )
        return gv

    data = relax.Var("data", R.Tensor((2, 5, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices = relax.Var("indices", R.Tensor((2, 5), "int32"))

    bb = relax.BlockBuilder()
    with bb.function("foo", [data, valid_count, indices]):
        gv = bb.emit(
            relax.op.vision.non_max_suppression(
                data,
                valid_count,
                indices,
                max_output_size=-1,
                iou_threshold=0.5,
                force_suppress=False,
                top_k=3,
                coord_start=2,
                score_index=1,
                id_index=0,
                return_indices=True,
                invalid_to_bottom=False,
                soft_nms_sigma=0.0,
                score_threshold=0.0,
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_non_max_suppression_return_indices_soft_nms():
    @R.function
    def foo(
        data: R.Tensor((2, 5, 6), "float32"),
        valid_count: R.Tensor((2,), "int32"),
        indices: R.Tensor((2, 5), "int32"),
    ) -> R.Tuple(
        R.Tensor((2, 5, 6), "float32"),
        R.Tensor((2, 5), "int32"),
        R.Tensor((2, 1), "int32"),
    ):
        gv: R.Tuple(
            R.Tensor((2, 5, 6), "float32"),
            R.Tensor((2, 5), "int32"),
            R.Tensor((2, 1), "int32"),
        ) = R.vision.non_max_suppression(
            data,
            valid_count,
            indices,
            max_output_size=-1,
            iou_threshold=0.5,
            force_suppress=False,
            top_k=3,
            coord_start=2,
            score_index=1,
            id_index=0,
            return_indices=True,
            invalid_to_bottom=False,
            soft_nms_sigma=0.5,
            score_threshold=0.0,
        )
        return gv

    data = relax.Var("data", R.Tensor((2, 5, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices = relax.Var("indices", R.Tensor((2, 5), "int32"))

    bb = relax.BlockBuilder()
    with bb.function("foo", [data, valid_count, indices]):
        gv = bb.emit(
            relax.op.vision.non_max_suppression(
                data,
                valid_count,
                indices,
                max_output_size=-1,
                iou_threshold=0.5,
                force_suppress=False,
                top_k=3,
                coord_start=2,
                score_index=1,
                id_index=0,
                return_indices=True,
                invalid_to_bottom=False,
                soft_nms_sigma=0.5,
                score_threshold=0.0,
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_non_max_suppression_return_data():
    @R.function
    def foo(
        data: R.Tensor((2, 5, 6), "float32"),
        valid_count: R.Tensor((2,), "int32"),
        indices: R.Tensor((2, 5), "int32"),
    ) -> R.Tensor((2, 5, 6), "float32"):
        gv: R.Tensor((2, 5, 6), "float32") = R.vision.non_max_suppression(
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
            return_indices=False,
            invalid_to_bottom=True,
            soft_nms_sigma=0.0,
            score_threshold=0.0,
        )
        return gv

    data = relax.Var("data", R.Tensor((2, 5, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices = relax.Var("indices", R.Tensor((2, 5), "int32"))

    bb = relax.BlockBuilder()
    with bb.function("foo", [data, valid_count, indices]):
        gv = bb.emit(
            relax.op.vision.non_max_suppression(
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
                return_indices=False,
                invalid_to_bottom=True,
                soft_nms_sigma=0.0,
                score_threshold=0.0,
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_multibox_transform_loc():
    @R.function
    def foo(
        cls: R.Tensor((1, 3, 5), "float32"),
        loc: R.Tensor((1, 20), "float32"),
        anc: R.Tensor((1, 5, 4), "float32"),
    ) -> R.Tuple(R.Tensor((1, 5, 4), "float32"), R.Tensor((1, 3, 5), "float32")):
        gv: R.Tuple(R.Tensor((1, 5, 4), "float32"), R.Tensor((1, 3, 5), "float32")) = (
            R.vision.multibox_transform_loc(
                cls,
                loc,
                anc,
                clip=False,
                threshold=0.0,
                variances=(1.0, 1.0, 1.0, 1.0),
                keep_background=True,
                apply_softmax=True,
            )
        )
        return gv

    cls = relax.Var("cls", R.Tensor((1, 3, 5), "float32"))
    loc = relax.Var("loc", R.Tensor((1, 20), "float32"))
    anc = relax.Var("anc", R.Tensor((1, 5, 4), "float32"))

    bb = relax.BlockBuilder()
    with bb.function("foo", [cls, loc, anc]):
        gv = bb.emit(
            relax.op.vision.multibox_transform_loc(
                cls,
                loc,
                anc,
                clip=False,
                threshold=0.0,
                variances=(1.0, 1.0, 1.0, 1.0),
                keep_background=True,
                apply_softmax=True,
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_roi_align():
    @R.function
    def foo(
        x: R.Tensor((1, 2, 8, 8), "float32"),
        rois: R.Tensor((2, 5), "float32"),
    ) -> R.Tensor((2, 2, 3, 3), "float32"):
        gv: R.Tensor((2, 2, 3, 3), "float32") = R.vision.roi_align(
            x,
            rois,
            pooled_size=(3, 3),
            spatial_scale=1.0,
            sample_ratio=2,
            layout="NCHW",
            mode="avg",
        )
        return gv

    x = relax.Var("x", R.Tensor((1, 2, 8, 8), "float32"))
    rois = relax.Var("rois", R.Tensor((2, 5), "float32"))

    bb = relax.BlockBuilder()
    with bb.function("foo", [x, rois]):
        gv = bb.emit(
            relax.op.vision.roi_align(
                x, rois, (3, 3), 1.0, sample_ratio=2, layout="NCHW", mode="avg"
            )
        )
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


if __name__ == "__main__":
    tvm_tirx.testing.main()
