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
# ruff: noqa: F401, F841
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import verify_trace_roundtrip
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@Ts.function
def single_elementwise(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

# fmt: on
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def test_blockize_outer():
    @Ts.function
    def after_blockize_outer(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
    ) -> None:
        with Ts.sblock("blockized_B"):
            vio = Ts.axis.spatial(1, 0)
            vjo = Ts.axis.spatial(1, 0)
            for i, j in T.grid(128, 128):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] * 2.0

    func = single_elementwise
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    x, _ = s.get_loops(s.get_sblock("B"))
    s.blockize(x)
    tvm_tirx.ir.assert_structural_equal(
        s.mod["main"], after_blockize_outer.with_attr("global_symbol", "single_elementwise")
    )
    verify_trace_roundtrip(sch=s, mod=func)


def test_blockize_inner():
    @Ts.function
    def after_blockize_inner(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
    ) -> None:
        for i in T.serial(128):
            with Ts.sblock("blockized_B"):
                vi = Ts.axis.spatial(128, i)
                vjo = Ts.axis.spatial(1, 0)
                for j in T.serial(128):
                    with Ts.sblock("B"):
                        vj = Ts.axis.remap("S", [j])
                        B[vi, vj] = A[vi, vj] * 2.0

    func = single_elementwise
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    _, y = s.get_loops(s.get_sblock("B"))
    s.blockize(y)
    tvm_tirx.ir.assert_structural_equal(
        s.mod["main"], after_blockize_inner.with_attr("global_symbol", "single_elementwise")
    )
    verify_trace_roundtrip(sch=s, mod=func)


def test_two_elementwise_blockize_reverse_compute_at():
    @Ts.function
    def before_blockize_rca(
        A: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ) -> None:
        B = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i, j in T.grid(8, 8):
            with Ts.sblock("B_o"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                Ts.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with Ts.sblock("B"):
                        vi_i, vj_i = Ts.axis.remap("SS", [i_1, j_1])
                        Ts.reads(A[vi * 16 + vi_i, vj * 16 + vj_i])
                        Ts.writes(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        B[vi * 16 + vi_i, vj * 16 + vj_i] = A[vi * 16 + vi_i, vj * 16 + vj_i] * 2.0
            for ax0, ax1 in T.grid(16, 16):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(128, i * 16 + ax0)
                    vj = Ts.axis.spatial(128, j * 16 + ax1)
                    Ts.reads(B[vi, vj])
                    Ts.writes(C[vi, vj])
                    C[vi, vj] = B[vi, vj] + 1.0

    @Ts.function
    def after_blockize_rca(
        A: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ) -> None:
        B = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i, j in T.grid(8, 8):
            with Ts.sblock("B_o"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                Ts.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with Ts.sblock("B"):
                        vi_i, vj_i = Ts.axis.remap("SS", [i_1, j_1])
                        Ts.reads(A[vi * 16 + vi_i, vj * 16 + vj_i])
                        Ts.writes(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        B[vi * 16 + vi_i, vj * 16 + vj_i] = A[vi * 16 + vi_i, vj * 16 + vj_i] * 2.0
            with Ts.sblock("C_o"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                Ts.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                for ax0, ax1 in T.grid(16, 16):
                    with Ts.sblock("C"):
                        vi_i, vj_i = Ts.axis.remap("SS", [ax0, ax1])
                        Ts.reads(B[vi * 16 + vi_i, vj * 16 + vj_i])
                        Ts.writes(C[vi * 16 + vi_i, vj * 16 + vj_i])
                        C[vi * 16 + vi_i, vj * 16 + vj_i] = B[vi * 16 + vi_i, vj * 16 + vj_i] + 1.0

    func = before_blockize_rca
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    _, _, x, _ = s.get_loops(s.get_sblock("C"))
    s.blockize(x)
    tvm_tirx.ir.assert_structural_equal(
        s.mod["main"], after_blockize_rca.with_attr("global_symbol", "before_blockize_rca")
    )
    verify_trace_roundtrip(sch=s, mod=func)


def test_two_elementwise_blockize_compute_at():
    @Ts.function
    def before_blockize_compute_at(
        A: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ) -> None:
        # body
        # with Ts.sblock("root")
        B = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i_0, j_0 in T.grid(8, 8):
            for ax0, ax1 in T.grid(16, 16):
                with Ts.sblock("B"):
                    vi = Ts.axis.spatial(128, i_0 * 16 + ax0)
                    vj = Ts.axis.spatial(128, j_0 * 16 + ax1)
                    Ts.reads(A[vi, vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * 2.0
            with Ts.sblock("C_o"):
                vi_o, vj_o = Ts.axis.remap("SS", [i_0, j_0])
                Ts.reads(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                Ts.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with Ts.sblock("C"):
                        vi_i, vj_i = Ts.axis.remap("SS", [i_1, j_1])
                        Ts.reads(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        Ts.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + 1.0
                        )

    @Ts.function
    def after_blockize_compute_at(
        A: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ) -> None:
        B = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i_0, j_0 in T.grid(8, 8):
            with Ts.sblock("B_o"):
                vi_o, vj_o = Ts.axis.remap("SS", [i_0, j_0])
                Ts.reads(A[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                Ts.writes(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for ax0, ax1 in T.grid(16, 16):
                    with Ts.sblock("B"):
                        vi_i, vj_i = Ts.axis.remap("SS", [ax0, ax1])
                        Ts.reads(A[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        Ts.writes(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            A[vi_o * 16 + vi_i, vj_o * 16 + vj_i] * 2.0
                        )
            with Ts.sblock("C_o"):
                vi_o, vj_o = Ts.axis.remap("SS", [i_0, j_0])
                Ts.reads(B[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                Ts.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with Ts.sblock("C"):
                        vi_i, vj_i = Ts.axis.remap("SS", [i_1, j_1])
                        Ts.reads(B[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        Ts.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = (
                            B[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + 1.0
                        )

    func = before_blockize_compute_at
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    _, _, x, _ = s.get_loops(s.get_sblock("B"))
    s.blockize(x)
    tvm_tirx.ir.assert_structural_equal(
        s.mod["main"],
        after_blockize_compute_at.with_attr("global_symbol", "before_blockize_compute_at"),
    )
    verify_trace_roundtrip(sch=s, mod=func)


def test_blockize_init_loops():
    @Ts.function
    def rowsum(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128,), "float32")) -> None:
        for k, i in T.grid(128, 128):
            with Ts.sblock("B"):
                vk, vi = Ts.axis.remap("RS", [k, i])
                with Ts.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]

    @Ts.function
    def after_rowsum_blockize(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128,), "float32"),
    ) -> None:
        with Ts.sblock("blockized_B"):
            vko = Ts.axis.R(1, 0)
            vio = Ts.axis.S(1, 0)
            with Ts.init():
                for i1 in T.serial(0, 128):
                    with Ts.sblock("B_init"):
                        vi_init = Ts.axis.S(128, i1)
                        B[vi_init] = T.float32(0)
            for i0, i1_1 in T.grid(128, 128):
                with Ts.sblock("B"):
                    vk, vi = Ts.axis.remap("RS", [i0, i1_1])
                    B[vi] = B[vi] + A[vi, vk]

    s = tvm.s_tir.Schedule(rowsum, debug_mask="all")
    k, _ = s.get_loops(s.get_sblock("B"))
    s.blockize(k)
    tvm_tirx.ir.assert_structural_equal(
        s.mod["main"], after_rowsum_blockize.with_attr("global_symbol", "rowsum")
    )
    verify_trace_roundtrip(sch=s, mod=rowsum)


@pytest.mark.parametrize("preserve_unit_iters", [True, False])
def test_blockize_outer_int64_shape(preserve_unit_iters):
    @Ts.function
    def single_elementwise_int64(
        A: T.Tensor((T.int64(16), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(16), T.int64(128)), "float32"),
    ) -> None:
        for i0, j0, i1, j1 in T.grid(T.int64(1), T.int64(8), T.int64(16), T.int64(16)):
            with Ts.sblock("B"):
                vi = Ts.axis.S(T.int64(16), i0 * T.int64(16) + i1)
                vj = Ts.axis.S(T.int64(128), j0 * T.int64(16) + j1)
                B[vi, vj] = A[vi, vj] + 1.0

    @Ts.function
    def after_single_elementwise_int64_blockize(
        A: T.Tensor((T.int64(16), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(16), T.int64(128)), "float32"),
    ) -> None:
        for i0, j0 in T.grid(T.int64(1), T.int64(8)):
            with Ts.sblock("B_o"):
                vi_o = Ts.axis.spatial(T.int64(1), T.int64(0))
                vj_o = Ts.axis.spatial(T.int64(8), j0)
                for i1, j1 in T.grid(T.int64(16), T.int64(16)):
                    with Ts.sblock("B"):
                        vi_i, vj_i = Ts.axis.remap("SS", [i1, j1])
                        B[vi_i, vj_o * T.int64(16) + vj_i] = A[
                            vi_i, vj_o * T.int64(16) + vj_i
                        ] + T.float32(1)

    @Ts.function
    def after_single_elementwise_int64_blockize_preserve_unit_iters(
        A: T.Tensor((T.int64(16), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(16), T.int64(128)), "float32"),
    ) -> None:
        for i0, j0 in T.grid(T.int64(1), T.int64(8)):
            with Ts.sblock("B_o"):
                vi_o = Ts.axis.spatial(T.int64(1), i0)
                vj_o = Ts.axis.spatial(T.int64(8), j0)
                for i1, j1 in T.grid(T.int64(16), T.int64(16)):
                    with Ts.sblock("B"):
                        vi_i, vj_i = Ts.axis.remap("SS", [i1, j1])
                        B[vi_i, vj_o * T.int64(16) + vj_i] = A[
                            vi_i, vj_o * T.int64(16) + vj_i
                        ] + T.float32(1)

    s = tvm.s_tir.Schedule(single_elementwise_int64, debug_mask="all")
    _, _, i1, _ = s.get_loops(s.get_sblock("B"))
    s.blockize(i1, preserve_unit_iters=preserve_unit_iters)
    expected = (
        after_single_elementwise_int64_blockize_preserve_unit_iters
        if preserve_unit_iters
        else after_single_elementwise_int64_blockize
    )
    tvm_tirx.ir.assert_structural_equal(
        s.mod["main"], expected.with_attr("global_symbol", "single_elementwise_int64")
    )
    verify_trace_roundtrip(sch=s, mod=single_elementwise_int64)


def test_blockize_blocks():
    @Ts.function
    def blocks_func(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")) -> None:
        for m in T.serial(6):
            for i, j in T.grid(3, 1):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A[vi, vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * 2.0

            for i, j in T.grid(128, 64):
                with Ts.sblock("C"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A[vi, vj + 64])
                    Ts.writes(B[vi, vj + 64])
                    B[vi, vj + 64] = A[vi, vj + 64] * 3.0

    @Ts.function
    def after_blocks_blockize(
        A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")
    ) -> None:
        for m in range(6):
            with Ts.sblock("outer_B_C_"):
                vi_o = Ts.axis.spatial(1, 0)
                vj_o = Ts.axis.spatial(1, 0)
                Ts.reads(A[0:128, 0:128])
                Ts.writes(B[0:128, 0:128])
                for i, j in T.grid(3, 1):
                    with Ts.sblock("B"):
                        vi_i = Ts.axis.spatial(3, i)
                        Ts.reads(A[vi_i, 0])
                        Ts.writes(B[vi_i, 0])
                        B[vi_i, 0] = A[vi_i, 0] * T.float32(2)
                for i, j in T.grid(128, 64):
                    with Ts.sblock("C"):
                        vi_i, vj_i = Ts.axis.remap("SS", [i, j])
                        Ts.reads(A[vi_i, vj_i + 64])
                        Ts.writes(B[vi_i, vj_i + 64])
                        B[vi_i, vj_i + 64] = A[vi_i, vj_i + 64] * T.float32(3)

    s = tvm.s_tir.Schedule(blocks_func, debug_mask="all")
    blocks = [s.get_sblock("B"), s.get_sblock("C")]
    s.blockize(blocks, preserve_unit_iters=False)
    expected = after_blocks_blockize
    tvm_tirx.ir.assert_structural_equal(
        s.mod["main"], expected.with_attr("global_symbol", "blocks_func")
    )
    verify_trace_roundtrip(sch=s, mod=blocks_func)


if __name__ == "__main__":
    tvm_tirx.testing.main()
