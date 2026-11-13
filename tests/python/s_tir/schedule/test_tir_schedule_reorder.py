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
# ruff: noqa: E741, F401
import sys

import pytest

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

# pylint: disable=no-member,invalid-name,unused-variable


@Ts.function
def elementwise(
    A: T.Tensor((128, 128, 128, 128), "float32"), B: T.Tensor((128, 128, 128, 128), "float32")
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.function
def elementwise_not_affine(
    A: T.Tensor((128, 128, 128, 128), "float32"), B: T.Tensor((128, 128, 128, 128), "float32")
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 8):
        with Ts.sblock("B"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            vl = Ts.axis.S(128, l * 16)
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.function
def elementwise_dependent_loop(
    A: T.Tensor((128, 128, 128, 128), "float32"), B: T.Tensor((128, 128, 128, 128), "float32")
) -> None:
    for i in T.serial(0, 128):
        for j, k, l in T.grid(128, i, 128):
            with Ts.sblock("B"):
                vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
                B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.function
def elementwise_predicate(
    A: T.Tensor((128, 128, 128, 128), "float32"), B: T.Tensor((128, 128, 128, 128), "float32")
) -> None:
    for i, j, k, l in T.grid(128, 128, 128, 128):
        with Ts.sblock("B"):
            Ts.where(i * 2097152 + j * 16384 + k * 128 + l < 100)
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.function
def elementwise_non_single_branch(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    C = Ts.sblock_alloc_tensor((128, 128, 128))

    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128):
            with Ts.sblock("C"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                C[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for k in T.serial(0, 128):
            with Ts.sblock("B"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                B[vi, vj, vk] = C[vi, vj, vk] * 2.0


@Ts.function
def elementwise_with_loops_not_same_scope(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("A"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            for k in T.serial(0, 128):
                with Ts.sblock("B"):
                    vk = Ts.axis.S(128, k)
                    Ts.reads([A[vi, vj, vk]])
                    Ts.writes([B[vi, vj, vk]])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_with_wrong_block_var_type(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            vk = Ts.axis.scan(128, k)
            Ts.reads([A[vi, vj, vk]])
            Ts.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_reordered(
    A: T.Tensor((128, 128, 128, 128), "float32"), B: T.Tensor((128, 128, 128, 128), "float32")
) -> None:
    for l, j, k, i in T.grid(128, 128, 128, 128):
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.function
def elementwise_reordered2(
    A: T.Tensor((128, 128, 128, 128), "float32"), B: T.Tensor((128, 128, 128, 128), "float32")
) -> None:
    for k, j, i, l in T.grid(128, 128, 128, 128):
        with Ts.sblock("B"):
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.function
def elementwise_reordered_with_predicate(
    A: T.Tensor((128, 128, 128, 128), "float32"), B: T.Tensor((128, 128, 128, 128), "float32")
) -> None:
    for l, j, k, i in T.grid(128, 128, 128, 128):
        with Ts.sblock("B"):
            Ts.where(i * 2097152 + j * 16384 + k * 128 + l < 100)
            vi, vj, vk, vl = Ts.axis.remap("SSSS", [i, j, k, l])
            B[vi, vj, vk, vl] = A[vi, vj, vk, vl] * 2.0


@Ts.function
def opaque_access(A: T.Tensor([16, 16], "float32"), B: T.Tensor([16, 16], "float32")) -> None:
    for i, j in T.grid(16, 16):
        with Ts.sblock("A"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads([])
            Ts.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for i, j in T.grid(16, 16):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads([])
            Ts.writes([B[0:16, 0:16]])
            T.evaluate(T.gpu_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj))


@Ts.function
def opaque_access_reorder(
    A: T.Tensor([16, 16], "float32"), B: T.Tensor([16, 16], "float32")
) -> None:
    for j, i in T.grid(16, 16):
        with Ts.sblock("A"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads([])
            Ts.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for j, i in T.grid(16, 16):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads([])
            Ts.writes([B[0:16, 0:16]])
            T.evaluate(T.gpu_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj))


# pylint: enable=no-member,invalid-name,unused-variable


def test_reorder():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k, l = sch.get_loops(block_b)
    sch.reorder(l, i)
    assert_structural_equal_ignore_global_symbol(elementwise_reordered, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reorder2():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k, l = sch.get_loops(block_b)
    sch.reorder(k, i, l)
    assert_structural_equal_ignore_global_symbol(elementwise_reordered2, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reorder_with_opaque_access():
    sch = tvm.s_tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_sblock("A")
    i, j = sch.get_loops(block_a)
    sch.reorder(j, i)
    block_b = sch.get_sblock("B")
    i, j = sch.get_loops(block_b)
    sch.reorder(j, i)
    assert_structural_equal_ignore_global_symbol(opaque_access_reorder, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_reorder_overlapped_access():
    @Ts.function
    def overlapped_access(A: T.Tensor((14, 4), "float32"), B: T.Tensor((14, 4), "float32")):
        # example to write first axis multiple times
        for v0, v1, v2 in T.grid(6, 4, 4):
            with Ts.sblock("block"):
                i = Ts.axis.spatial(14, v0 * 2 + v1)
                j = Ts.axis.spatial(4, v2)
                B[i, j] = A[i, j] + 1.0

    @Ts.function
    def overlapped_access_reorder(A: T.Tensor((14, 4), "float32"), B: T.Tensor((14, 4), "float32")):
        # example to write first axis multiple times
        for v0, v2, v1 in T.grid(6, 4, 4):
            with Ts.sblock("block"):
                i = Ts.axis.spatial(14, v0 * 2 + v1)
                j = Ts.axis.spatial(4, v2)
                B[i, j] = A[i, j] + 1.0

    sch = tvm.s_tir.Schedule(overlapped_access, debug_mask="all")
    v0, v1, v2 = sch.get_loops(sch.get_sblock("block"))
    sch.reorder(v0, v2, v1)
    assert_structural_equal_ignore_global_symbol(overlapped_access_reorder, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=overlapped_access)


def test_reorder_with_partial_affineness():
    @Ts.function
    def non_affine_func(A: T.Tensor((14, 4), "float32"), B: T.Tensor((14, 4), "float32")):
        for v0, v1, v2 in T.grid(6, 4, 4):
            with Ts.sblock("block"):
                i = Ts.axis.spatial(14, v0 * v0 + v1)
                j = Ts.axis.spatial(4, v2)
                B[i, j] = A[i, j] + 1.0

    @Ts.function
    def non_affine_func_reorder(A: T.Tensor((14, 4), "float32"), B: T.Tensor((14, 4), "float32")):
        for v0, v2, v1 in T.grid(6, 4, 4):
            with Ts.sblock("block"):
                i = Ts.axis.spatial(14, v0 * v0 + v1)
                j = Ts.axis.spatial(4, v2)
                B[i, j] = A[i, j] + 1.0

    sch = tvm.s_tir.Schedule(non_affine_func, debug_mask="all")
    v0, v1, v2 = sch.get_loops(sch.get_sblock("block"))
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(v0, v2, v1)

    sch.reorder(v2, v1)
    assert_structural_equal_ignore_global_symbol(non_affine_func_reorder, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=non_affine_func)


def test_reorder_with_cascade_tiled_ops():
    @Ts.function
    def cascade_pool_ops(
        x: T.Tensor((1, 16, 112, 112), "float32"), y2: T.Tensor((1, 16, 108, 108), "float32")
    ) -> None:
        y1 = Ts.sblock_alloc_tensor([1, 16, 110, 110], dtype="float32")
        for n, c, h, w, kh, kw in T.grid(1, 16, 110, 110, 3, 3):
            with Ts.sblock("pool_0"):
                ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [n, c, h, w, kh, kw])
                with Ts.init():
                    y1[ax0, ax1, ax2, ax3] = 0.0
                y1[ax0, ax1, ax2, ax3] = y1[ax0, ax1, ax2, ax3] + x[ax0, ax1, ax2 + rv0, ax3 + rv1]
        for n, c, h, w, kh, kw in T.grid(1, 16, 108, 108, 3, 3):
            with Ts.sblock("pool_1"):
                ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [n, c, h, w, kh, kw])
                with Ts.init():
                    y2[ax0, ax1, ax2, ax3] = 0.0
                y2[ax0, ax1, ax2, ax3] = y2[ax0, ax1, ax2, ax3] + y1[ax0, ax1, ax2 + rv0, ax3 + rv1]

    @Ts.function
    def cascade_pool_ops_tile_reordered(
        x: T.Tensor((1, 16, 112, 112), "float32"), y2: T.Tensor((1, 16, 108, 108), "float32")
    ) -> None:
        y1 = Ts.sblock_alloc_tensor([1, 16, 110, 110], dtype="float32")
        for n, c, h_o in T.grid(1, 16, 27):
            for w, h_i, kh, kw in T.grid(110, 6, 3, 3):
                with Ts.sblock("pool_0"):
                    ax0 = Ts.axis.spatial(1, 0)
                    ax1 = Ts.axis.spatial(16, c)
                    ax2 = Ts.axis.spatial(110, h_o * 4 + h_i)
                    ax3, rv0, rv1 = Ts.axis.remap("SRR", [w, kh, kw])
                    with Ts.init():
                        y1[ax0, ax1, ax2, ax3] = 0.0
                    y1[ax0, ax1, ax2, ax3] = (
                        y1[ax0, ax1, ax2, ax3] + x[ax0, ax1, ax2 + rv0, ax3 + rv1]
                    )
            for h_i, w, kh, kw in T.grid(4, 108, 3, 3):
                with Ts.sblock("pool_1"):
                    ax0 = Ts.axis.spatial(1, n)
                    ax1 = Ts.axis.spatial(16, c)
                    ax2 = Ts.axis.spatial(108, h_o * 4 + h_i)
                    ax3, rv0, rv1 = Ts.axis.remap("SRR", [w, kh, kw])
                    with Ts.init():
                        y2[ax0, ax1, ax2, ax3] = 0.0
                    y2[ax0, ax1, ax2, ax3] = (
                        y2[ax0, ax1, ax2, ax3] + y1[ax0, ax1, ax2 + rv0, ax3 + rv1]
                    )

    sch = tvm.s_tir.schedule.Schedule(cascade_pool_ops)
    pool_0 = sch.get_sblock("pool_0")
    pool_1 = sch.get_sblock("pool_1")
    _, _, h, w, _, _ = sch.get_loops(pool_1)
    ho, _ = sch.split(h, factors=[None, 4])
    sch.compute_at(pool_0, ho)
    _, _, _, h_i, w, _, _ = sch.get_loops(pool_0)
    sch.reorder(w, h_i)
    assert_structural_equal_ignore_global_symbol(
        cascade_pool_ops_tile_reordered, sch.mod["main"], True
    )
    verify_trace_roundtrip(sch=sch, mod=cascade_pool_ops)


def test_reorder_with_predicate():
    sch = tvm.s_tir.Schedule(elementwise_predicate, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(l, i)


def test_reorder_fail_with_multi_appearance_loops():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(k, i, i)


def test_reorder_fail_with_non_single_branch_loop():
    sch = tvm.s_tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(k, i)
    sch = tvm.s_tir.Schedule(elementwise_non_single_branch, debug_mask="all")
    block_b = sch.get_sblock("B")
    block_c = sch.get_sblock("C")
    i, j, k1 = sch.get_loops(block_b)
    _, _, k2 = sch.get_loops(block_c)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(k1, i, k2)


def test_reorder_fail_with_loops_not_under_same_scope():
    sch = tvm.s_tir.Schedule(elementwise_with_loops_not_same_scope, debug_mask="all")
    block_b = sch.get_sblock("B")
    block_a = sch.get_sblock("A")
    i, j = sch.get_loops(block_a)
    k = sch.get_loops(block_b)[0]
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(k, i)


def test_reorder_fail_with_wrong_block_var_type():
    sch = tvm.s_tir.Schedule(elementwise_with_wrong_block_var_type, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(k, i)


def test_reorder_fail_with_dependent_loops():
    sch = tvm.s_tir.Schedule(elementwise_dependent_loop, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(l, i)


def test_reorder_fail_not_affine_bindings():
    sch = tvm.s_tir.Schedule(elementwise_not_affine, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k, l = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder(l, i)


if __name__ == "__main__":
    tvm_tirx.testing.main()
