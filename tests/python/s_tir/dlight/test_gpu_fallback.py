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
# pylint: disable=missing-docstring
# ruff: noqa: E501, E741, F841
import pytest

import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def test_fallback():
    @I.ir_module
    class Before:
        @Ts.function
        def main(
            A: T.Tensor((1, 32, 1, 128), "float16"),
            C: T.Tensor((1, 1, 4096), "float16"),
        ):
            B = Ts.sblock_alloc_tensor((1, 1, 32, 128), "float16")
            for i, j, k, l in T.grid(1, 1, 32, 128):
                with Ts.sblock("T_transpose"):
                    vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk, vl] = A[vi, vk, vj, vl]
            for i, j, k in T.grid(1, 1, 4096):
                with Ts.sblock("T_reshape"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = B[0, 0, vk % 4096 // 128, vk % 128]

    @I.ir_module
    class After:
        @Ts.function
        def main(
            A: T.Tensor((1, 32, 1, 128), "float16"),
            C: T.Tensor((1, 1, 4096), "float16"),
        ):
            T.func_attr({"tirx.is_scheduled": True})
            for ax0_fused_0 in T.parallel(4, annotations={"thread_binding": "blockIdx.x"}):
                for ax0_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_reshape"):
                        v0 = Ts.axis.spatial(4096, ax0_fused_0 * 1024 + ax0_fused_1)
                        Ts.reads(A[0, v0 // 128, 0, v0 % 128])
                        Ts.writes(C[0, 0, v0])
                        C[0, 0, v0] = A[0, v0 // 128, 0, v0 % 128]

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.Fallback(), target=target
    )(Before)
    assert_structural_equal(mod, After)


def test_fallback_skips_zero_extent_spatial():
    @I.ir_module
    class Module:
        @Ts.function
        def main(A: T.Tensor((4, 0), "float32"), B: T.Tensor((4, 0), "float32")):
            for i, j in T.grid(4, 0):
                with Ts.sblock("copy"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.Fallback(), target=Target("nvidia/geforce-rtx-3090-ti")
    )(Module)
    assert_structural_equal(mod, Module)


def test_fallback_reduction():
    @I.ir_module
    class Module:
        @Ts.function
        def main(A: T.Tensor((1, 6144), "float32"), B: T.Tensor((1,), "float32")):
            for ax0, ax1 in T.grid(1, 6144):
                with Ts.sblock("block"):
                    v0 = Ts.axis.spatial(1, ax0)
                    v1 = Ts.axis.reduce(6144, ax1)
                    Ts.reads(A[v0, v1])
                    Ts.writes(B[v0])
                    with Ts.init():
                        B[v0] = T.float32(0)
                    B[v0] = B[v0] + T.Cast("float32", A[v0, v1])

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((1, 6144), "float32"), B: T.Tensor((1,), "float32")):
            T.func_attr({"tirx.is_scheduled": True})
            for ax0_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for ax0_fused_1 in T.parallel(
                    T.int64(1024), annotations={"thread_binding": "threadIdx.x"}
                ):
                    with Ts.sblock("block_init"):
                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                        Ts.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(1))
                        Ts.reads()
                        Ts.writes(B[0])
                        B[0] = T.float32(0)
                    for ax1 in range(6144):
                        with Ts.sblock("block_update"):
                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                            v1 = Ts.axis.reduce(6144, ax1)
                            Ts.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(1))
                            Ts.reads(B[0], A[0, v1])
                            Ts.writes(B[0])
                            B[0] = B[0] + T.Cast("float32", A[0, v1])

    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.Fallback(), target=Target("apple/m1-gpu")
    )(Module)
    assert_structural_equal(mod, Expected)


def test_fallback_irregular_spatial():
    nhead = T.dynamic("nhead", "int32")
    nlayer = T.dynamic("nlayer", "int32")
    seqlen = T.dynamic("seqlen", "int32")
    npage = T.dynamic("npage", "int32")
    page_size = T.dynamic("page_size", "int32")
    num_total_pages = T.dynamic("num_total_pages", "int32")
    num_total_seqs_plus_1 = T.dynamic("num_total_seqs_plus_1", "int32")

    @Ts.function(private=True)
    def func(
        pages: T.Tensor((num_total_pages, nlayer, nhead, page_size), "float16"),
        page_table_indptr: T.Tensor((num_total_seqs_plus_1,), "int32"),
        page_table_values: T.Tensor((npage,), "int32"),
        values: T.Tensor((nlayer, nhead, seqlen), "float16"),
        seq_id: T.int32,
    ):
        for l, h, pos in T.grid(nlayer, nhead, seqlen):
            with Ts.sblock("block"):
                vl, vh, vp = Ts.axis.remap("SSS", [l, h, pos])
                values[vl, vh, vp] = pages[
                    page_table_values[page_table_indptr[seq_id] + T.floordiv(vp, page_size)],
                    vl,
                    vh,
                    T.floormod(vp, page_size),
                ]

    # fmt: off
    nhead = T.dynamic("nhead", "int32")
    nlayer = T.dynamic("nlayer", "int32")
    seqlen = T.dynamic("seqlen", "int32")
    npage = T.dynamic("npage", "int32")
    page_size = T.dynamic("page_size", "int32")
    num_total_pages = T.dynamic("num_total_pages", "int32")
    num_total_seqs_plus_1 = T.dynamic("num_total_seqs_plus_1", "int32")

    @Ts.function(private=True)
    def expected(pages: T.Tensor((num_total_pages, nlayer, nhead, page_size), 'float16'), page_table_indptr: T.Tensor((num_total_seqs_plus_1,), 'int32'), page_table_values: T.Tensor((npage,), 'int32'), values: T.Tensor((nlayer, nhead, seqlen), 'float16'), seq_id: T.int32):
        T.func_attr({"tirx.is_scheduled": True})

        for ax0_ax1_ax2_fused_0 in T.parallel((nlayer * nhead * seqlen + 1023) // 1024, annotations={"thread_binding": "blockIdx.x"}):
            for ax0_ax1_ax2_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("block"):
                    v0 = Ts.axis.spatial(nlayer, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) // (nhead * seqlen))
                    v1 = Ts.axis.spatial(nhead, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % (nhead * seqlen) // seqlen)
                    v2 = Ts.axis.spatial(seqlen, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % seqlen)
                    Ts.where(ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 < nlayer * nhead * seqlen)
                    Ts.reads(pages[page_table_values[page_table_indptr[seq_id] + v2 // page_size], v0, v1, v2 % page_size], page_table_values[page_table_indptr[seq_id] + v2 // page_size], page_table_indptr[seq_id])
                    Ts.writes(values[v0, v1, v2])
                    values[v0, v1, v2] = pages[page_table_values[page_table_indptr[seq_id] + v2 // page_size], v0, v1, v2 % page_size]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = tvm_tirx.IRModule({"main": func})
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.Fallback(), target=target
    )(mod)
    assert_structural_equal(mod["main"], expected)


def test_gpu_fallback_ignores_non_gpu_functions():
    @I.ir_module
    class Before:
        # This function has no "target" attribute, and is scheduled
        # using the target supplied to ApplyDefaultSchedule.
        @Ts.function
        def gpu_func(
            A: T.Tensor((1, 32, 1, 128), "float16"),
            C: T.Tensor((1, 1, 4096), "float16"),
        ):
            B = Ts.sblock_alloc_tensor((1, 1, 32, 128), "float16")
            for i, j, k, l in T.grid(1, 1, 32, 128):
                with Ts.sblock("T_transpose"):
                    vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk, vl] = A[vi, vk, vj, vl]
            for i, j, k in T.grid(1, 1, 4096):
                with Ts.sblock("T_reshape"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = B[0, 0, vk % 4096 // 128, vk % 128]

        # This function is identical, except that it is explicitly
        # annotated with the "target" attribute, and is scheduled
        # based on the annotation's target.
        @Ts.function
        def cpu_func(
            A: T.Tensor((1, 32, 1, 128), "float16"),
            C: T.Tensor((1, 1, 4096), "float16"),
        ):
            T.func_attr({"target": T.target("llvm")})
            B = Ts.sblock_alloc_tensor((1, 1, 32, 128), "float16")
            for i, j, k, l in T.grid(1, 1, 32, 128):
                with Ts.sblock("T_transpose"):
                    vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk, vl] = A[vi, vk, vj, vl]
            for i, j, k in T.grid(1, 1, 4096):
                with Ts.sblock("T_reshape"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = B[0, 0, vk % 4096 // 128, vk % 128]

    @I.ir_module
    class After:
        @Ts.function
        def gpu_func(
            A: T.Tensor((1, 32, 1, 128), "float16"),
            C: T.Tensor((1, 1, 4096), "float16"),
        ):
            T.func_attr({"tirx.is_scheduled": True})
            for ax0_fused_0 in T.parallel(4, annotations={"thread_binding": "blockIdx.x"}):
                for ax0_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_reshape"):
                        v0 = Ts.axis.spatial(4096, ax0_fused_0 * 1024 + ax0_fused_1)
                        Ts.reads(A[0, v0 // 128, 0, v0 % 128])
                        Ts.writes(C[0, 0, v0])
                        C[0, 0, v0] = A[0, v0 // 128, 0, v0 % 128]

        @Ts.function
        def cpu_func(
            A: T.Tensor((1, 32, 1, 128), "float16"),
            C: T.Tensor((1, 1, 4096), "float16"),
        ):
            T.func_attr({"target": T.target("llvm")})
            B = Ts.sblock_alloc_tensor((1, 1, 32, 128), "float16")
            for i, j, k, l in T.grid(1, 1, 32, 128):
                with Ts.sblock("T_transpose"):
                    vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
                    B[vi, vj, vk, vl] = A[vi, vk, vj, vl]
            for i, j, k in T.grid(1, 1, 4096):
                with Ts.sblock("T_reshape"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = B[0, 0, vk % 4096 // 128, vk % 128]

    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.Fallback(), target=Target("cuda")
    )(Before)
    assert_structural_equal(mod, After)


def test_schedule_error_propagates_from_rule():
    # ScheduleError indicates a broken rule and must propagate.
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
            for i in range(128):
                with Ts.sblock("copy"):
                    vi = Ts.axis.remap("S", [i])
                    Ts.reads(A[vi])
                    Ts.writes(C[vi])
                    C[vi] = A[vi] * T.float32(2)

    class BrokenRule(dl.base.ScheduleRule):
        def apply(self, func, target, tunable):
            sch = s_tir.Schedule(func)
            sch.get_sblock("no_such_block")
            return sch

    with pytest.raises(s_tir.ScheduleError):
        dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
            BrokenRule(), dl.gpu.Fallback(), target=Target("nvidia/geforce-rtx-3090-ti")
        )(Before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
