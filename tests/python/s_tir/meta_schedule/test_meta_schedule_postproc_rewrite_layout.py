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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# ruff: noqa: E501
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

tvm_tirx.runtime.MemoryScope.register("global2")


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> ms.TuneContext:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[
                ms.postproc.RewriteLayout(),
            ],
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


def _apply_rewrite_layout(mod):
    """Apply the RewriteLayout postproc transformation."""
    target = Target("cuda", host="llvm")
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[
                ms.postproc.RewriteLayout(),
            ],
            mutator_probs={},
        ),
        task_name="test",
    )
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    if not ctx.space_generator.postprocs[0].apply(sch):
        raise RuntimeError("RewriteLayout postproc failed")
    return sch.mod


def test_tir_matmul():
    """Main functionality test

    A new block should be inserted to transform the layout, with the
    compute block operating on the temporary transformed tensor.
    """

    @Ts.function(private=True)
    def before(
        A: T.Tensor((16, 16), "float32"),
        B: T.Tensor((16, 16), "float32"),
        C: T.Tensor((16, 16), "float32"),
    ) -> None:
        T.func_attr({"layout_free_tensors": [1]})
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with Ts.sblock("matmul"):
                vi = Ts.axis.S(16, i0 * 4 + i1)
                vj = Ts.axis.S(16, j)
                vk = Ts.axis.R(16, k0 * 4 + k1)
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @Ts.function(private=True)
    def expected(
        A: T.Tensor((16, 16), "float32"),
        B: T.Tensor((16, 16), "float32"),
        C: T.Tensor((16, 16), "float32"),
    ) -> None:
        T.func_attr({"layout_free_tensors": [1]})
        B_reindex = Ts.sblock_alloc_tensor([16, 4, 4], dtype="float32")
        for ax0, ax1 in T.grid(16, 16):
            with Ts.sblock("layout_rewrite"):
                i0, i1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc": True})
                B_reindex[i1, i0 // 4, i0 % 4] = B[i0, i1]
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with Ts.sblock("matmul"):
                vi = Ts.axis.spatial(16, i0 * 4 + i1)
                vj = Ts.axis.spatial(16, j)
                vk = Ts.axis.reduce(16, k0 * 4 + k1)
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B_reindex[vj, vk // 4, vk % 4]

    mod = tvm_tirx.IRModule.from_expr(before)
    mod = _apply_rewrite_layout(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_rewritten_tensors_must_occur_within_block():
    """Tensors must occur within a Block"""

    @Ts.function(private=True)
    def before(
        A: T.Tensor((16, 16), "float32"),
    ) -> None:
        T.func_attr({"layout_free_tensors": [0]})
        for i, j in T.grid(16, 16):
            T.evaluate(A[i, j])

    mod = tvm_tirx.IRModule.from_expr(before)
    with pytest.raises(RuntimeError):
        _apply_rewrite_layout(mod)


def test_extent_one():
    """Tensors with dimensions of extent 1 can be transformed

    Regression test for a previous bug, in which the removal of
    trivial variables resulted in an error in `IndexMap::Inverse`.
    """

    @Ts.function(private=True)
    def before(
        A: T.Tensor((16, 1), "float32"),
    ) -> None:
        T.func_attr({"layout_free_tensors": [0]})
        for i, j in T.grid(16, 1):
            with Ts.sblock("block"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                T.evaluate(A[vi, vj])

    @Ts.function(private=True)
    def expected(A: T.Tensor((16, 1), "float32")):
        T.func_attr({"layout_free_tensors": [0]})

        A_global = Ts.sblock_alloc_tensor([16], dtype="float32")
        for ax0, ax1 in T.grid(16, 1):
            with Ts.sblock("A_global"):
                v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc": True})
                A_global[v0] = A[v0, v1]

        for i, j in T.grid(16, 1):
            with Ts.sblock("block"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                T.evaluate(A_global[vi])

    mod = tvm_tirx.IRModule.from_expr(before)
    mod = _apply_rewrite_layout(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


@Ts.function
def tir_matmul(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
) -> None:
    T.func_attr({"layout_free_tensors": [1]})
    for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
        with Ts.sblock("matmul"):
            vi = Ts.axis.S(16, i0 * 4 + i1)
            vj = Ts.axis.S(16, j)
            vk = Ts.axis.R(16, k0 * 4 + k1)
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@Ts.function
def rewritten_tir_matmul(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
) -> None:
    T.func_attr({"layout_free_tensors": [1]})
    B_reindex = Ts.sblock_alloc_tensor([16, 4, 4], dtype="float32")
    for ax0, ax1 in T.grid(16, 16):
        with Ts.sblock("layout_rewrite"):
            i0, i1 = Ts.axis.remap("SS", [ax0, ax1])
            Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc": True})
            B_reindex[i1, i0 // 4, i0 % 4] = B[i0, i1]
    for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
        with Ts.sblock("matmul"):
            vi = Ts.axis.spatial(16, i0 * 4 + i1)
            vj = Ts.axis.spatial(16, j)
            vk = Ts.axis.reduce(16, k0 * 4 + k1)
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B_reindex[vj, vk // 4, vk % 4]


def test_layout_rewrite():
    target = _target()
    ctx = _create_context(tir_matmul, target)
    sch = tvm.s_tir.Schedule(tir_matmul, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], rewritten_tir_matmul)


# fmt: off
@tvm_tirx.script.ir_module
class Conv2dCacheRead:
    @Ts.function
    def main(p0: T.Tensor((1, 56, 56, 64), "float32"), p1: T.Tensor((3, 3, 64, 64), "float32"), conv2d_nhwc: T.Tensor((1, 56, 56, 64), "float32")):
        T.func_attr({"layout_free_tensors": [1], "tirx.noalias": True, "global_symbol": "main"})
        pad_temp = Ts.sblock_alloc_tensor([1, 58, 58, 64], dtype="float32")
        conv2d_nhwc_global = Ts.sblock_alloc_tensor([1, 56, 56, 64], dtype="float32")
        pad_temp_global = Ts.sblock_alloc_tensor([1, 58, 58, 64], dtype="float32")
        p1_global = Ts.sblock_alloc_tensor([3, 3, 64, 64], dtype="float32")
        for i0_0_i1_0_i2_0_fused in T.parallel(4, annotations={"tirx.auto_unroll_max_step":16, "tirx.unroll_explicit":1}):
            for ax0, ax1, ax2 in T.grid(1, 30, 30):
                for ax3_fused in T.vectorized(64):
                    with Ts.sblock("pad_temp"):
                        i0 = Ts.axis.spatial(1, ax0)
                        i1 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused // 2 * 28 + ax1)
                        i2 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused % 2 * 28 + ax2)
                        i3 = Ts.axis.spatial(64, ax3_fused)
                        Ts.reads(p0[i0, i1 - 1, i2 - 1, i3])
                        Ts.writes(pad_temp[i0, i1, i2, i3])
                        pad_temp[i0, i1, i2, i3] = T.if_then_else(1 <= i1 and i1 < 57 and 1 <= i2 and i2 < 57, p0[i0, i1 - 1, i2 - 1, i3], T.float32(0))
            for i3_0 in T.serial(16):
                for ax0_ax1_ax2_ax3_fused in T.serial(57600):
                    with Ts.sblock("pad_temp_global"):
                        v0 = Ts.axis.spatial(1, 0)
                        v1 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused // 2 * 28 + ax0_ax1_ax2_ax3_fused // 1920)
                        v2 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused % 2 * 28 + ax0_ax1_ax2_ax3_fused % 1920 // 64)
                        v3 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 64)
                        Ts.reads(pad_temp[v0, v1, v2, v3])
                        Ts.writes(pad_temp_global[v0, v1, v2, v3])
                        pad_temp_global[v0, v1, v2, v3] = pad_temp[v0, v1, v2, v3]
                for ax0_ax1_ax2_ax3_fused in T.serial(2304):
                    with Ts.sblock("p1_global"):
                        v0 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused // 768)
                        v1 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 768 // 256)
                        v2 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 256 // 4)
                        v3 = Ts.axis.spatial(64, i3_0 * 4 + ax0_ax1_ax2_ax3_fused % 4)
                        Ts.reads(p1[v0, v1, v2, v3])
                        Ts.writes(p1_global[v0, v1, v2, v3])
                        p1_global[v0, v1, v2, v3] = p1[v0, v1, v2, v3]
                for i0_1, i1_1, i2_1, i3_1 in T.grid(1, 7, 2, 1):
                    for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i0_3_init, i1_3_init, i2_3_init in T.grid(1, 1, 14, 2, 1, 4, 1):
                        for i3_3_fused_init in T.vectorized(2):
                            with Ts.sblock("conv2d_nhwc_init"):
                                nn = Ts.axis.spatial(1, i0_1 + i0_2_init + i0_3_init)
                                yy = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + i1_2_init * 4 + i1_3_init)
                                xx = Ts.axis.spatial(56, i2_3_init + i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + i2_2_init)
                                ff = Ts.axis.spatial(64, i3_0 * 4 + i3_1 * 4 + i3_2_init * 2 + i3_3_fused_init)
                                Ts.reads()
                                Ts.writes(conv2d_nhwc_global[nn, yy, xx, ff])
                                Ts.sblock_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                                conv2d_nhwc_global[nn, yy, xx, ff] = T.float32(0)
                    for i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3 in T.grid(1, 1, 2, 1, 1, 14, 2, 3, 3, 32, 1, 4, 1):
                        for i3_3_fused in T.vectorized(2):
                            with Ts.sblock("conv2d_nhwc_update"):
                                nn = Ts.axis.spatial(1, i0_1 + i0_2 + i0_3)
                                yy = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + i1_2 * 4 + i1_3)
                                xx = Ts.axis.spatial(56, i2_3 + i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + i2_2)
                                ff = Ts.axis.spatial(64, i3_0 * 4 + i3_1 * 4 + i3_2 * 2 + i3_3_fused)
                                ry = Ts.axis.reduce(3, i4_0 * 3 + i4_1)
                                rx = Ts.axis.reduce(3, i5_0 * 3 + i5_1)
                                rc = Ts.axis.reduce(64, i6_0 * 32 + i6_1)
                                Ts.reads(conv2d_nhwc_global[nn, yy, xx, ff], pad_temp_global[nn, yy + ry, xx + rx, rc], p1_global[ry, rx, rc, ff])
                                Ts.writes(conv2d_nhwc_global[nn, yy, xx, ff])
                                Ts.sblock_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                                conv2d_nhwc_global[nn, yy, xx, ff] = conv2d_nhwc_global[nn, yy, xx, ff] + pad_temp_global[nn, yy + ry, xx + rx, rc] * p1_global[ry, rx, rc, ff]
                    for ax0, ax1, ax2 in T.grid(1, 4, 14):
                        for ax3_fused in T.vectorized(4):
                            with Ts.sblock("conv2d_nhwc_global"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + ax1)
                                v2 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + ax2)
                                v3 = Ts.axis.spatial(64, i3_0 * 4 + ax3_fused)
                                Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                                Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]


@tvm_tirx.script.ir_module
class Conv2dCacheReadRewritten:
    @Ts.function
    def main(p0: T.Tensor((1, 56, 56, 64), "float32"), p1: T.Tensor((3, 3, 64, 64), "float32"), conv2d_nhwc: T.Tensor((1, 56, 56, 64), "float32")):
        T.func_attr({"layout_free_tensors": [1], "tirx.noalias": True, "global_symbol": "main"})
        pad_temp = Ts.sblock_alloc_tensor([1, 58, 58, 64], dtype="float32")
        conv2d_nhwc_global = Ts.sblock_alloc_tensor([1, 56, 56, 64], dtype="float32")
        pad_temp_global = Ts.sblock_alloc_tensor([1, 58, 58, 64], dtype="float32")
        p1_global = Ts.sblock_alloc_tensor([16, 2, 2, 3, 3, 32, 2], dtype="float32")
        p1_global_1 = Ts.sblock_alloc_tensor([16, 2, 2, 3, 3, 32, 2], dtype="float32")
        for ax0, ax1, ax2, ax3 in T.grid(3, 3, 64, 64):
            with Ts.sblock("p1_global"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(p1[v0, v1, v2, v3])
                Ts.writes(p1_global_1[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2])
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc":True})
                p1_global_1[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2] = p1[v0, v1, v2, v3]
        for i0_0_i1_0_i2_0_fused in T.parallel(4, annotations={"tirx.auto_unroll_max_step":16, "tirx.unroll_explicit":1}):
            for ax0, ax1, ax2 in T.grid(1, 30, 30):
                for ax3_fused in T.vectorized(64):
                    with Ts.sblock("pad_temp"):
                        i0 = Ts.axis.spatial(1, ax0)
                        i1 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused // 2 * 28 + ax1)
                        i2 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused % 2 * 28 + ax2)
                        i3 = Ts.axis.spatial(64, ax3_fused)
                        Ts.reads(p0[i0, i1 - 1, i2 - 1, i3])
                        Ts.writes(pad_temp[i0, i1, i2, i3])
                        pad_temp[i0, i1, i2, i3] = T.if_then_else(1 <= i1 and i1 < 57 and 1 <= i2 and i2 < 57, p0[i0, i1 - 1, i2 - 1, i3], T.float32(0))
            for i3_0 in T.serial(16):
                for ax0_ax1_ax2_ax3_fused in T.serial(57600):
                    with Ts.sblock("pad_temp_global"):
                        v0 = Ts.axis.spatial(1, 0)
                        v1 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused // 2 * 28 + ax0_ax1_ax2_ax3_fused // 1920)
                        v2 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused % 2 * 28 + ax0_ax1_ax2_ax3_fused % 1920 // 64)
                        v3 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 64)
                        Ts.reads(pad_temp[v0, v1, v2, v3])
                        Ts.writes(pad_temp_global[v0, v1, v2, v3])
                        pad_temp_global[v0, v1, v2, v3] = pad_temp[v0, v1, v2, v3]
                for ax0_ax1_ax2_ax3_fused in T.serial(2304):
                    with Ts.sblock("p1_global"):
                        v0 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused // 768)
                        v1 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 768 // 256)
                        v2 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 256 // 4)
                        v3 = Ts.axis.spatial(64, i3_0 * 4 + ax0_ax1_ax2_ax3_fused % 4)
                        Ts.reads(p1_global_1[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2])
                        Ts.writes(p1_global[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2])
                        p1_global[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2] = p1_global_1[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2]
                for i0_1, i1_1, i2_1, i3_1 in T.grid(1, 7, 2, 1):
                    for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i0_3_init, i1_3_init, i2_3_init in T.grid(1, 1, 14, 2, 1, 4, 1):
                        for i3_3_fused_init in T.vectorized(2):
                            with Ts.sblock("conv2d_nhwc_init"):
                                nn = Ts.axis.spatial(1, i0_1 + i0_2_init + i0_3_init)
                                yy = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + i1_2_init * 4 + i1_3_init)
                                xx = Ts.axis.spatial(56, i2_3_init + i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + i2_2_init)
                                ff = Ts.axis.spatial(64, i3_0 * 4 + i3_1 * 4 + i3_2_init * 2 + i3_3_fused_init)
                                Ts.reads()
                                Ts.writes(conv2d_nhwc_global[nn, yy, xx, ff])
                                Ts.sblock_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                                conv2d_nhwc_global[nn, yy, xx, ff] = T.float32(0)
                    for i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3 in T.grid(1, 1, 2, 1, 1, 14, 2, 3, 3, 32, 1, 4, 1):
                        for i3_3_fused in T.vectorized(2):
                            with Ts.sblock("conv2d_nhwc_update"):
                                nn = Ts.axis.spatial(1, i0_1 + i0_2 + i0_3)
                                yy = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + i1_2 * 4 + i1_3)
                                xx = Ts.axis.spatial(56, i2_3 + i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + i2_2)
                                ff = Ts.axis.spatial(64, i3_0 * 4 + i3_1 * 4 + i3_2 * 2 + i3_3_fused)
                                ry = Ts.axis.reduce(3, i4_0 * 3 + i4_1)
                                rx = Ts.axis.reduce(3, i5_0 * 3 + i5_1)
                                rc = Ts.axis.reduce(64, i6_0 * 32 + i6_1)
                                Ts.reads(conv2d_nhwc_global[nn, yy, xx, ff], pad_temp_global[nn, yy + ry, xx + rx, rc], p1_global[ff // 4, rc // 32, ff % 4 // 2, ry, rx, rc % 32, ff % 2])
                                Ts.writes(conv2d_nhwc_global[nn, yy, xx, ff])
                                Ts.sblock_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                                conv2d_nhwc_global[nn, yy, xx, ff] = conv2d_nhwc_global[nn, yy, xx, ff] + pad_temp_global[nn, yy + ry, xx + rx, rc] * p1_global[ff // 4, rc // 32, ff % 4 // 2, ry, rx, rc % 32, ff % 2]
                    for ax0, ax1, ax2 in T.grid(1, 4, 14):
                        for ax3_fused in T.vectorized(4):
                            with Ts.sblock("conv2d_nhwc_global"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + ax1)
                                v2 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + ax2)
                                v3 = Ts.axis.spatial(64, i3_0 * 4 + ax3_fused)
                                Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                                Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]


@tvm_tirx.script.ir_module
class Conv2dCacheReadMultipleRewritten:
    @Ts.function
    def main(p0: T.Tensor((1, 56, 56, 64), "float32"), p1: T.Tensor((3, 3, 64, 64), "float32"), conv2d_nhwc: T.Tensor((1, 56, 56, 64), "float32")):
        T.func_attr({"layout_free_tensors": [1], "tirx.noalias": True, "global_symbol": "main"})
        pad_temp = Ts.sblock_alloc_tensor([1, 58, 58, 64], dtype="float32")
        conv2d_nhwc_global = Ts.sblock_alloc_tensor([1, 56, 56, 64], dtype="float32")
        pad_temp_global = Ts.sblock_alloc_tensor([1, 58, 58, 64], dtype="float32")
        p1_global = Ts.sblock_alloc_tensor([16, 2, 2, 3, 3, 32, 2], dtype="float32")
        p1_global2 = Ts.sblock_alloc_tensor([16, 2, 2, 3, 3, 32, 2], dtype="float32", scope="global2")
        p1_global_1 = Ts.sblock_alloc_tensor([16, 2, 2, 3, 3, 32, 2], dtype="float32")
        for ax0, ax1, ax2, ax3 in T.grid(3, 3, 64, 64):
            with Ts.sblock("p1_global"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(p1[v0, v1, v2, v3])
                Ts.writes(p1_global_1[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2])
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc":True})
                p1_global_1[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2] = p1[v0, v1, v2, v3]
        for ax0, ax1, ax2, ax3 in T.grid(3, 3, 64, 64):
            with Ts.sblock("p1_global2"):
                v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(p1_global_1[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2])
                Ts.writes(p1_global2[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2])
                p1_global2[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2] = p1_global_1[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2]
        for i0_0_i1_0_i2_0_fused in T.parallel(4, annotations={"tirx.auto_unroll_max_step":16, "tirx.unroll_explicit":1}):
            for ax0, ax1, ax2 in T.grid(1, 30, 30):
                for ax3_fused in T.vectorized(64):
                    with Ts.sblock("pad_temp"):
                        i0 = Ts.axis.spatial(1, ax0)
                        i1 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused // 2 * 28 + ax1)
                        i2 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused % 2 * 28 + ax2)
                        i3 = Ts.axis.spatial(64, ax3_fused)
                        Ts.reads(p0[i0, i1 - 1, i2 - 1, i3])
                        Ts.writes(pad_temp[i0, i1, i2, i3])
                        pad_temp[i0, i1, i2, i3] = T.if_then_else(1 <= i1 and i1 < 57 and 1 <= i2 and i2 < 57, p0[i0, i1 - 1, i2 - 1, i3], T.float32(0))
            for i3_0 in T.serial(16):
                for ax0_ax1_ax2_ax3_fused in T.serial(57600):
                    with Ts.sblock("pad_temp_global"):
                        v0 = Ts.axis.spatial(1, 0)
                        v1 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused // 2 * 28 + ax0_ax1_ax2_ax3_fused // 1920)
                        v2 = Ts.axis.spatial(58, i0_0_i1_0_i2_0_fused % 2 * 28 + ax0_ax1_ax2_ax3_fused % 1920 // 64)
                        v3 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 64)
                        Ts.reads(pad_temp[v0, v1, v2, v3])
                        Ts.writes(pad_temp_global[v0, v1, v2, v3])
                        pad_temp_global[v0, v1, v2, v3] = pad_temp[v0, v1, v2, v3]
                for ax0_ax1_ax2_ax3_fused in T.serial(2304):
                    with Ts.sblock("p1_global"):
                        v0 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused // 768)
                        v1 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 768 // 256)
                        v2 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 256 // 4)
                        v3 = Ts.axis.spatial(64, i3_0 * 4 + ax0_ax1_ax2_ax3_fused % 4)
                        Ts.reads(p1_global2[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2])
                        Ts.writes(p1_global[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2])
                        p1_global[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2] = p1_global2[v3 // 4, v2 // 32, v3 % 4 // 2, v0, v1, v2 % 32, v3 % 2]
                for i0_1, i1_1, i2_1, i3_1 in T.grid(1, 7, 2, 1):
                    for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i0_3_init, i1_3_init, i2_3_init in T.grid(1, 1, 14, 2, 1, 4, 1):
                        for i3_3_fused_init in T.vectorized(2):
                            with Ts.sblock("conv2d_nhwc_init"):
                                nn = Ts.axis.spatial(1, i0_1 + i0_2_init + i0_3_init)
                                yy = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + i1_2_init * 4 + i1_3_init)
                                xx = Ts.axis.spatial(56, i2_3_init + i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + i2_2_init)
                                ff = Ts.axis.spatial(64, i3_0 * 4 + i3_1 * 4 + i3_2_init * 2 + i3_3_fused_init)
                                Ts.reads()
                                Ts.writes(conv2d_nhwc_global[nn, yy, xx, ff])
                                Ts.sblock_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                                conv2d_nhwc_global[nn, yy, xx, ff] = T.float32(0)
                    for i4_0, i5_0, i6_0, i0_2, i1_2, i2_2, i3_2, i4_1, i5_1, i6_1, i0_3, i1_3, i2_3 in T.grid(1, 1, 2, 1, 1, 14, 2, 3, 3, 32, 1, 4, 1):
                        for i3_3_fused in T.vectorized(2):
                            with Ts.sblock("conv2d_nhwc_update"):
                                nn = Ts.axis.spatial(1, i0_1 + i0_2 + i0_3)
                                yy = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + i1_2 * 4 + i1_3)
                                xx = Ts.axis.spatial(56, i2_3 + i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + i2_2)
                                ff = Ts.axis.spatial(64, i3_0 * 4 + i3_1 * 4 + i3_2 * 2 + i3_3_fused)
                                ry = Ts.axis.reduce(3, i4_0 * 3 + i4_1)
                                rx = Ts.axis.reduce(3, i5_0 * 3 + i5_1)
                                rc = Ts.axis.reduce(64, i6_0 * 32 + i6_1)
                                Ts.reads(conv2d_nhwc_global[nn, yy, xx, ff], pad_temp_global[nn, yy + ry, xx + rx, rc], p1_global[ff // 4, rc // 32, ff % 4 // 2, ry, rx, rc % 32, ff % 2])
                                Ts.writes(conv2d_nhwc_global[nn, yy, xx, ff])
                                Ts.sblock_attr({"meta_schedule.tiling_structure":"SSRSRS"})
                                conv2d_nhwc_global[nn, yy, xx, ff] = conv2d_nhwc_global[nn, yy, xx, ff] + pad_temp_global[nn, yy + ry, xx + rx, rc] * p1_global[ff // 4, rc // 32, ff % 4 // 2, ry, rx, rc % 32, ff % 2]
                    for ax0, ax1, ax2 in T.grid(1, 4, 14):
                        for ax3_fused in T.vectorized(4):
                            with Ts.sblock("conv2d_nhwc_global"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused // 2 * 28 + i1_1 * 4 + ax1)
                                v2 = Ts.axis.spatial(56, i0_0_i1_0_i2_0_fused % 2 * 28 + i2_1 * 14 + ax2)
                                v3 = Ts.axis.spatial(64, i3_0 * 4 + ax3_fused)
                                Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                                Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]

# fmt: on


def test_layout_rewrite_cache_read():
    target = Target("llvm")
    ctx = _create_context(Conv2dCacheRead, target)
    sch = tvm.s_tir.Schedule(Conv2dCacheRead, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm_tirx.ir.assert_structural_equal(sch.mod, Conv2dCacheReadRewritten)


def test_layout_rewrite_cache_read_multiple():
    target = Target("llvm")
    ctx = _create_context(Conv2dCacheRead, target)
    sch = tvm.s_tir.Schedule(Conv2dCacheRead, debug_mask="all")
    sch.cache_read(sch.get_sblock("p1_global"), 0, "global2")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm_tirx.ir.assert_structural_equal(sch.mod, Conv2dCacheReadMultipleRewritten)


def test_layout_rewrite_int64_index():
    @Ts.function(private=True)
    def before(
        p0: T.Tensor((T.int64(12), T.int64(197), T.int64(64)), "int8"),
        p1: T.Tensor((T.int64(12), T.int64(197), T.int64(64)), "int8"),
        T_batch_matmul_NT: T.Tensor((T.int64(12), T.int64(197), T.int64(197)), "int32"),
    ):
        T.func_attr({"layout_free_tensors": [1], "tirx.noalias": True})
        for b_0_i_0_fused in T.parallel(T.int64(394)):
            for j_0 in T.serial(T.int64(1)):
                for b_1, i_1, j_1 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                    for b_2_init, i_2_init, j_2_init, b_3_init, i_3_init, j_3_init in T.grid(
                        T.int64(6), T.int64(1), T.int64(197), T.int64(1), T.int64(1), T.int64(1)
                    ):
                        with Ts.sblock("T_batch_matmul_NT_init"):
                            v_b = Ts.axis.spatial(
                                T.int64(12),
                                b_3_init
                                + b_0_i_0_fused // T.int64(197) * T.int64(6)
                                + b_1 * T.int64(6)
                                + b_2_init,
                            )
                            v_i = Ts.axis.spatial(
                                T.int64(197),
                                b_0_i_0_fused % T.int64(197) + i_1 + i_2_init + i_3_init,
                            )
                            v_j = Ts.axis.spatial(
                                T.int64(197),
                                j_3_init + j_0 * T.int64(197) + j_1 * T.int64(197) + j_2_init,
                            )
                            T_batch_matmul_NT[v_b, v_i, v_j] = 0
                    for k_0, b_2, i_2, j_2, k_1, b_3, i_3, j_3 in T.grid(
                        T.int64(64),
                        T.int64(6),
                        T.int64(1),
                        T.int64(197),
                        T.int64(1),
                        T.int64(1),
                        T.int64(1),
                        T.int64(1),
                    ):
                        with Ts.sblock("T_batch_matmul_NT_update"):
                            v_b = Ts.axis.spatial(
                                T.int64(12),
                                b_3
                                + b_0_i_0_fused // T.int64(197) * T.int64(6)
                                + b_1 * T.int64(6)
                                + b_2,
                            )
                            v_i = Ts.axis.spatial(
                                T.int64(197), b_0_i_0_fused % T.int64(197) + i_1 + i_2 + i_3
                            )
                            v_j = Ts.axis.spatial(
                                T.int64(197), j_3 + j_0 * T.int64(197) + j_1 * T.int64(197) + j_2
                            )
                            v_k = Ts.axis.reduce(T.int64(64), k_0 + k_1)
                            T_batch_matmul_NT[v_b, v_i, v_j] = T_batch_matmul_NT[
                                v_b, v_i, v_j
                            ] + T.Cast("int32", p0[v_b, v_i, v_k]) * T.Cast(
                                "int32", p1[v_b, v_j, v_k]
                            )

    @Ts.function(private=True)
    def expected(
        p0: T.Tensor((T.int64(12), T.int64(197), T.int64(64)), "int8"),
        p1: T.Tensor((T.int64(12), T.int64(197), T.int64(64)), "int8"),
        T_batch_matmul_NT: T.Tensor((T.int64(12), T.int64(197), T.int64(197)), "int32"),
    ):
        T.func_attr({"tirx.noalias": True, "layout_free_tensors": [1]})
        p1_global = Ts.sblock_alloc_tensor(
            [T.int64(2), T.int64(64), T.int64(6), T.int64(197)], dtype="int8"
        )
        for ax0, ax1, ax2 in T.grid(T.int64(12), T.int64(197), T.int64(64)):
            with Ts.sblock("p1_global"):
                v0, v1, v2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(p1[v0, v1, v2])
                Ts.writes(p1_global[v0 // T.int64(6), v2, v0 % T.int64(6), v1])
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc": True})
                p1_global[v0 // T.int64(6), v2, v0 % T.int64(6), v1] = p1[v0, v1, v2]
        for b_0_i_0_fused in T.parallel(T.int64(394)):
            for j_0, b_1, i_1, j_1 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                for b_2_init, i_2_init, j_2_init, b_3_init, i_3_init, j_3_init in T.grid(
                    T.int64(6), T.int64(1), T.int64(197), T.int64(1), T.int64(1), T.int64(1)
                ):
                    with Ts.sblock("T_batch_matmul_NT_init"):
                        v_b = Ts.axis.spatial(
                            T.int64(12),
                            b_3_init
                            + b_0_i_0_fused // T.int64(197) * T.int64(6)
                            + b_1 * T.int64(6)
                            + b_2_init,
                        )
                        v_i = Ts.axis.spatial(
                            T.int64(197), b_0_i_0_fused % T.int64(197) + i_1 + i_2_init + i_3_init
                        )
                        v_j = Ts.axis.spatial(
                            T.int64(197),
                            j_3_init + j_0 * T.int64(197) + j_1 * T.int64(197) + j_2_init,
                        )
                        T_batch_matmul_NT[v_b, v_i, v_j] = 0
                for k_0, b_2, i_2, j_2, k_1, b_3, i_3, j_3 in T.grid(
                    T.int64(64),
                    T.int64(6),
                    T.int64(1),
                    T.int64(197),
                    T.int64(1),
                    T.int64(1),
                    T.int64(1),
                    T.int64(1),
                ):
                    with Ts.sblock("T_batch_matmul_NT_update"):
                        v_b = Ts.axis.spatial(
                            T.int64(12),
                            b_3
                            + b_0_i_0_fused // T.int64(197) * T.int64(6)
                            + b_1 * T.int64(6)
                            + b_2,
                        )
                        v_i = Ts.axis.spatial(
                            T.int64(197), b_0_i_0_fused % T.int64(197) + i_1 + i_2 + i_3
                        )
                        v_j = Ts.axis.spatial(
                            T.int64(197), j_3 + j_0 * T.int64(197) + j_1 * T.int64(197) + j_2
                        )
                        v_k = Ts.axis.reduce(T.int64(64), k_0 + k_1)
                        T_batch_matmul_NT[v_b, v_i, v_j] = T_batch_matmul_NT[
                            v_b, v_i, v_j
                        ] + T.Cast("int32", p0[v_b, v_i, v_k]) * T.Cast(
                            "int32", p1_global[v_b // T.int64(6), v_k, v_b % T.int64(6), v_j]
                        )

    mod = tvm_tirx.IRModule.from_expr(before)
    mod = _apply_rewrite_layout(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
