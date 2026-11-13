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
# pylint: disable=missing-function-docstring,missing-module-docstring
# ruff: noqa: E501, F401
import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


def test_annotate_read_tensor_access():
    @Ts.function
    def before(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    @Ts.function
    def expected(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi - 1 : vi - 1 + 2, vj - 1 : vj - 1 + 2])
                Ts.writes(B[vi, vj])
                Ts.sblock_attr({"explicit_read_region": [0]})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    block = sch.get_sblock("B")
    sch.annotate_tensor_access(
        block, 0, "read", lambda vi, vj: ((vi - 1, vi + 1), (vj - 1, vj + 1))
    )
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_annotate_write_tensor_access():
    @Ts.function
    def before(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    @Ts.function
    def expected(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi, vj])
                Ts.writes(B[vi : vi + 2, vj : vj + 2])
                Ts.sblock_attr({"explicit_write_region": [0]})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    block = sch.get_sblock("B")
    sch.annotate_tensor_access(block, 0, "write", lambda vi, vj: ((vi, vi + 2), (vj, vj + 2)))
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_annotate_tensor_access_for_resize():
    # fmt: off
    @Ts.function
    def resize_before(x: T.Tensor((1, 1, 32, 32), "float16"), resize: T.Tensor((1, 1, 16, 16), "float16")):
        for i0, i1, i2, i3 in T.grid(1, 1, 16, 16):
            with Ts.sblock("resize"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(x[v_i0, v_i1, 0:32, 0:32])
                Ts.writes(resize[v_i0, v_i1, v_i2, v_i3])
                resize[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T.Cast("float32", x[v_i0, v_i1, T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i2) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0), T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i3) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0)]))

    @Ts.function
    def resize_expected(x: T.Tensor((1, 1, 32, 32), "float16"), resize: T.Tensor((1, 1, 16, 16), "float16")):
        for i0, i1, i2, i3 in T.grid(1, 1, 16, 16):
            with Ts.sblock("resize"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(x[v_i0, v_i1, v_i2 * 2 - 3:v_i2 * 2 + 3, v_i3 * 2 - 3:v_i3 * 2 + 3])
                Ts.writes(resize[v_i0, v_i1, v_i2, v_i3])
                Ts.sblock_attr({"explicit_read_region": [0]})
                resize[v_i0, v_i1, v_i2, v_i3] = T.Cast("float16", T.Cast("float32", x[v_i0, v_i1, T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i2) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0), T.max(T.min(T.Cast("int32", T.floor((T.Cast("float32", v_i3) + T.float32(0.5)) * T.float32(2) - T.float32(0.5) + T.float32(1.0000000000000001e-05))), 31), 0)]))
    # fmt: on
    sch = tvm.s_tir.Schedule(resize_before, debug_mask="all")
    block = sch.get_sblock("resize")
    sch.annotate_tensor_access(
        block,
        0,
        "read",
        gen_new_ranges=lambda v_i0, v_i1, v_i2, v_i3: [
            v_i0,
            v_i1,
            (v_i2 * 2 - 3, v_i2 * 2 + 3),
            (v_i3 * 2 - 3, v_i3 * 2 + 3),
        ],
    )
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], resize_expected)
    verify_trace_roundtrip(sch=sch, mod=resize_before)


def test_annotate_tensor_access_read_and_write():
    @Ts.function
    def before(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi, vj])
                Ts.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(B[vi, vj])
                Ts.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    @Ts.function
    def expected(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi - 1 : vi + 2, vj - 1 : vj + 2])
                Ts.writes(B[vi : vi + 2, vj : vj + 2])
                Ts.sblock_attr({"explicit_read_region": [0], "explicit_write_region": [0]})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(B[vi, vj])
                Ts.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    block = sch.get_sblock("B")

    sch.annotate_tensor_access(
        block, 0, "read", lambda vi, vj: ((vi - 1, vi + 2), (vj - 1, vj + 2))
    )

    sch.annotate_tensor_access(block, 0, "write", lambda vi, vj: ((vi, vi + 2), (vj, vj + 2)))

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_double_annotate_tensor_access_read():
    @Ts.function
    def before(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi, vj])
                Ts.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(B[vi, vj])
                Ts.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    @Ts.function
    def expected(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi - 2 : vi + 3, vj - 2 : vj + 3])
                Ts.writes(B[vi, vj])
                Ts.sblock_attr({"explicit_read_region": [0]})
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(B[vi, vj])
                Ts.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    block = sch.get_sblock("B")

    sch.annotate_tensor_access(
        block, 0, "read", lambda vi, vj: ((vi - 1, vi + 2), (vj - 1, vj + 2))
    )

    sch.annotate_tensor_access(
        block, 0, "read", lambda vi, vj: ((vi - 2, vi + 3), (vj - 2, vj + 3))
    )

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], expected)
    verify_trace_roundtrip(sch=sch, mod=before)


def test_annotate_tensor_access_with_compute_at_for_resize():
    # fmt: off
    @Ts.function
    def before(x: T.Tensor((1, 3, 200, 200), "float32"), y: T.Tensor((1, 3, 100, 100), "float32")):
        x_global = Ts.sblock_alloc_tensor([1, 3, 200, 200], dtype="float32")
        for ax0, ax1, ax2, ax3 in T.grid(1, 3, 200, 200):
            with Ts.sblock("cache"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
        for i0, i1, i2, i3 in T.grid(1, 3, 100, 100):
            with Ts.sblock("resize"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                y[v_i0, v_i1, v_i2, v_i3] = x_global[v_i0, v_i1, T.Cast("int32", T.floor(v_i2 * 2 + 0.5)), T.Cast("int32", T.floor(v_i3 * 2 + 0.5))]

    @Ts.function
    def after(x: T.Tensor((1, 3, 200, 200), "float32"), y: T.Tensor((1, 3, 100, 100), "float32")):
        x_global = Ts.sblock_alloc_tensor((1, 3, 200, 200))
        for i0, i1, i2_0, i3_0 in T.grid(1, 3, 10, 10):
            for ax0, ax1 in T.grid(24, 24):
                with Ts.sblock("cache"):
                    v0 = Ts.axis.spatial(1, 0)
                    v1 = Ts.axis.spatial(3, i1)
                    v2 = Ts.axis.spatial(200, i2_0 * 20 - 3 + ax0)
                    v3 = Ts.axis.spatial(200, i3_0 * 20 - 3 + ax1)
                    Ts.where(3 <= i2_0 * 20 + ax0 and i2_0 * 20 + ax0 < 203 and 3 <= i3_0 * 20 + ax1 and i3_0 * 20 + ax1 < 203)
                    Ts.reads(x[v0, v1, v2, v3])
                    Ts.writes(x_global[v0, v1, v2, v3])
                    x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
            for i2_1, i3_1 in T.grid(10, 10):
                with Ts.sblock("resize"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    v_i2 = Ts.axis.spatial(100, i2_0 * 10 + i2_1)
                    v_i3 = Ts.axis.spatial(100, i3_0 * 10 + i3_1)
                    Ts.reads(x_global[v_i0, v_i1, v_i2 * 2 - 3:v_i2 * 2 - 3 + 6, v_i3 * 2 - 3:v_i3 * 2 - 3 + 6])
                    Ts.writes(y[v_i0, v_i1, v_i2, v_i3])
                    Ts.sblock_attr({"explicit_read_region": [0]})
                    y[v_i0, v_i1, v_i2, v_i3] = x_global[v_i0, v_i1, T.Cast("int32", T.floor(T.Cast("float32", v_i2 * 2) + T.float32(0.5))), T.Cast("int32", T.floor(T.Cast("float32", v_i3 * 2) + T.float32(0.5)))]

    @Ts.function
    def after_without_annotate_tensor_access(x: T.Tensor((1, 3, 200, 200), "float32"), y: T.Tensor((1, 3, 100, 100), "float32")):
        x_global = Ts.sblock_alloc_tensor((1, 3, 200, 200))
        for i0, i1, i2_0, i3_0 in T.grid(1, 3, 10, 10):
            for ax0, ax1 in T.grid(200, 200):
                with Ts.sblock("cache"):
                    v0 = Ts.axis.spatial(1, 0)
                    v1, v2, v3 = Ts.axis.remap("SSS", [i1, ax0, ax1])
                    Ts.reads(x[v0, v1, v2, v3])
                    Ts.writes(x_global[v0, v1, v2, v3])
                    x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
            for i2_1, i3_1 in T.grid(10, 10):
                with Ts.sblock("resize"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    v_i2 = Ts.axis.spatial(100, i2_0 * 10 + i2_1)
                    v_i3 = Ts.axis.spatial(100, i3_0 * 10 + i3_1)
                    Ts.reads(x_global[v_i0, v_i1, 0:200, 0:200])
                    Ts.writes(y[v_i0, v_i1, v_i2, v_i3])
                    y[v_i0, v_i1, v_i2, v_i3] = x_global[v_i0, v_i1, T.Cast("int32", T.floor(T.Cast("float32", v_i2 * 2) + T.float32(0.5))), T.Cast("int32", T.floor(T.Cast("float32", v_i3 * 2) + T.float32(0.5)))]
    # fmt: on

    # Schedule with annotate_tensor_access
    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    block = sch.get_sblock("resize")
    cache_block = sch.get_sblock("cache")

    # Annotate tensor access
    sch.annotate_tensor_access(
        block,
        0,
        "read",
        lambda vn, vc, vh, vw: (vn, vc, (vh * 2 - 3, vh * 2 + 3), (vw * 2 - 3, vw * 2 + 3)),
    )

    h, w = sch.get_loops(block)[-2:]
    ho, hi = sch.split(h, factors=[10, 10])
    wo, wi = sch.split(w, factors=[10, 10])
    sch.reorder(ho, wo, hi, wi)
    sch.compute_at(cache_block, wo)

    assert_structural_equal_ignore_global_symbol(sch.mod["main"], after)
    verify_trace_roundtrip(sch=sch, mod=before)

    # Schedule without annotate_tensor_access
    sch_without_annotate = tvm.s_tir.Schedule(before, debug_mask="all")
    block_without_annotate = sch_without_annotate.get_sblock("resize")
    cache_block_without_annotate = sch_without_annotate.get_sblock("cache")

    h, w = sch_without_annotate.get_loops(block_without_annotate)[-2:]
    ho, hi = sch_without_annotate.split(h, factors=[10, 10])
    wo, wi = sch_without_annotate.split(w, factors=[10, 10])
    sch_without_annotate.reorder(ho, wo, hi, wi)
    sch_without_annotate.compute_at(cache_block_without_annotate, wo)

    assert_structural_equal_ignore_global_symbol(
        sch_without_annotate.mod["main"], after_without_annotate_tensor_access
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
