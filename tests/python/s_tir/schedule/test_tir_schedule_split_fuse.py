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
from __future__ import annotations

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import te
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm_tirx import tirx
from tvm_tirx.ir import IntImm
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable


@Ts.function
def elementwise(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("B"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_dependent_loops(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for i in T.serial(0, 128):
        for j, k in T.grid(i, 128):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i)
                vj = Ts.axis.S(i, j)
                vk = Ts.axis.S(128, k)
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_symbolic(
    A: T.Tensor((128, 128, n), "float32"),  # noqa: F821
    B: T.Tensor((128, 128, n), "float32"),  # noqa: F821
    n: T.int32,
) -> None:
    for i, j, k in T.grid(128, 128, n):
        with Ts.sblock("B"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_symbolic_fused(
    A: T.Tensor((128, 128, n), "float32"),  # noqa: F821
    B: T.Tensor((128, 128, n), "float32"),  # noqa: F821
    n: T.int32,
) -> None:
    for i_j_k_fused in T.serial(0, (n * 16384)):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, T.floordiv(i_j_k_fused, n * 128))
            vj = Ts.axis.S(128, T.floordiv(T.floormod(i_j_k_fused, n * 128), n))
            vk = Ts.axis.S(n, T.floormod(i_j_k_fused, n))
            Ts.reads([A[vi, vj, vk]])
            Ts.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_symbolic_split(
    A: T.Tensor((128, 128, n), "float32"),  # noqa: F821
    B: T.Tensor((128, 128, n), "float32"),  # noqa: F821
    n: T.int32,
) -> None:
    for i, j, k0, k1 in T.grid(128, 128, 10, T.floordiv((n + 9), 10)):
        with Ts.sblock("B"):
            Ts.where(((k0 * T.floordiv((n + 9), 10)) + k1) < n)
            vi, vj = Ts.axis.remap("SS", [i, j])
            vk = Ts.axis.S(n, k0 * T.floordiv(n + 9, 10) + k1)
            Ts.reads([A[vi, vj, vk]])
            Ts.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_with_seq(
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
def elementwise_with_anno(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        for k in T.serial(0, 128, annotations={"useless_annotation": True}):
            with Ts.sblock("B"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                Ts.reads([A[vi, vj, vk]])
                Ts.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_with_thread_binding(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        for k in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("B"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                Ts.reads([A[vi, vj, vk]])
                Ts.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_with_starting_point(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        for k in T.serial(10, 128):
            with Ts.sblock("B"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                Ts.reads([A[vi, vj, vk]])
                Ts.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_with_opaque_block(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("opaque"):
            Ts.reads([A[i, j, k]])
            Ts.writes([B[i, j, k]])
            with Ts.sblock("B"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                Ts.reads([A[vi, vj, vk]])
                Ts.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_fused(
    A: T.Tensor((128, 128, 128), "float32"), B: T.Tensor((128, 128, 128), "float32")
) -> None:
    for fused in T.serial(0, 2097152):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, T.floordiv(fused, 16384))
            vj = Ts.axis.S(128, T.floordiv(T.floormod(fused, 16384), 128))
            vk = Ts.axis.S(128, T.floormod(fused, 128))
            Ts.reads([A[vi, vj, vk]])
            Ts.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_split_case0(
    A: T.Tensor([128, 128, 128], "float32"), B: T.Tensor([128, 128, 128], "float32")
) -> None:
    for i1, i2, i3, j1, j2, k1, k2 in T.grid(2, 1, 64, 4, 32, 16, 8):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, i1 * 64 + i2 * 64 + i3)
            vj = Ts.axis.S(128, j1 * 32 + j2)
            vk = Ts.axis.S(128, k1 * 8 + k2)
            Ts.reads([A[vi, vj, vk]])
            Ts.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_split_case1(
    A: T.Tensor([128, 128, 128], "float32"), B: T.Tensor([128, 128, 128], "float32")
) -> None:
    for i1, i2, i3, j1, j2, j3, k1, k2, k3 in T.grid(2, 1, 64, 2, 1, 64, 2, 1, 64):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, i1 * 64 + i2 * 64 + i3)
            vj = Ts.axis.S(128, j1 * 64 + j2 * 64 + j3)
            vk = Ts.axis.S(128, k1 * 64 + k2 * 64 + k3)
            Ts.reads([A[vi, vj, vk]])
            Ts.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_split_with_predicate(
    A: T.Tensor([128, 128, 128], "float32"), B: T.Tensor([128, 128, 128], "float32")
) -> None:
    for i0, i1, i2, j0, j1, k0, k1 in T.grid(1000, 2, 3, 1, 129, 3, 43):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, i0 * 6 + i1 * 3 + i2)
            vj = Ts.axis.S(128, j0 * 129 + j1)
            vk = Ts.axis.S(128, k0 * 43 + k1)
            Ts.where((i0 * 2 + i1) * 3 + i2 < 128 and j0 * 129 + j1 < 128 and k0 * 43 + k1 < 128)
            Ts.reads([A[vi, vj, vk]])
            Ts.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_fuse_with_opaque_block(
    A: T.Tensor([128, 128, 128], "float32"), B: T.Tensor([128, 128, 128], "float32")
) -> None:
    for i_j_k_fused in T.serial(0, 2097152):
        with Ts.sblock("opaque"):
            Ts.reads(
                [
                    A[
                        T.floordiv(i_j_k_fused, 16384),
                        T.floordiv(T.floormod(i_j_k_fused, 16384), 128),
                        T.floormod(i_j_k_fused, 128),
                    ]
                ]
            )
            Ts.writes(
                [
                    B[
                        T.floordiv(i_j_k_fused, 16384),
                        T.floordiv(T.floormod(i_j_k_fused, 16384), 128),
                        T.floormod(i_j_k_fused, 128),
                    ]
                ]
            )
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, T.floordiv(i_j_k_fused, 16384))
                vj = Ts.axis.S(128, T.floordiv(T.floormod(i_j_k_fused, 16384), 128))
                vk = Ts.axis.S(128, T.floormod(i_j_k_fused, 128))
                Ts.reads([A[vi, vj, vk]])
                Ts.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def elementwise_split_with_opaque_block(
    A: T.Tensor([128, 128, 128], "float32"), B: T.Tensor([128, 128, 128], "float32")
) -> None:
    for i0, i1, j, k in T.grid(8, 16, 128, 128):
        with Ts.sblock("opaque"):
            Ts.reads([A[i0 * 16 + i1, j, k]])
            Ts.writes([B[i0 * 16 + i1, j, k]])
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i0 * 16 + i1)
                vj, vk = Ts.axis.remap("SS", [j, k])
                Ts.reads([A[vi, vj, vk]])
                Ts.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0


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
def opaque_access_fused(A: T.Tensor([16, 16], "float32"), B: T.Tensor([16, 16], "float32")) -> None:
    for i_j_fused in T.serial(0, 256):
        with Ts.sblock("A"):
            vi = Ts.axis.S(16, T.floordiv(i_j_fused, 16))
            vj = Ts.axis.S(16, T.floormod(i_j_fused, 16))
            Ts.reads([])
            Ts.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for i_j_fused in T.serial(0, 256):
        with Ts.sblock("B"):
            vi = Ts.axis.S(16, T.floordiv(i_j_fused, 16))
            vj = Ts.axis.S(16, T.floormod(i_j_fused, 16))
            Ts.reads([])
            Ts.writes([B[0:16, 0:16]])
            T.evaluate(T.gpu_fill_fragment(B.data, 16, 16, 16, 0, ((vi * 16) + vj)))


@Ts.function
def opaque_access_split(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")) -> None:
    for i, j0, j1 in T.grid(16, 4, 4):
        with Ts.sblock("A"):
            vi = Ts.axis.S(16, i)
            vj = Ts.axis.S(16, j0 * 4 + j1)
            Ts.reads([])
            Ts.writes([A[0:16, 0:16]])
            A[vi, vj] = 1
    for i, j0, j1 in T.grid(16, 4, 4):
        with Ts.sblock("B"):
            vi = Ts.axis.S(16, i)
            vj = Ts.axis.S(16, j0 * 4 + j1)
            Ts.reads([])
            Ts.writes([B[0:16, 0:16]])
            T.evaluate(T.gpu_fill_fragment(B.data, 16, 16, 16, 0, ((vi * 16) + vj)))


@Ts.function
def elementwise_not_affine(
    A: T.Tensor((127, 128), "float32"), B: T.Tensor((127, 128), "float32")
) -> None:
    for i in T.serial(0, 4):
        for j, k in T.grid(T.min(31, 126 - i * 32) + 1, 128):
            with Ts.sblock("B"):
                vi = Ts.axis.S(127, i * 32 + j)
                vj = Ts.axis.S(128, k)
                B[vi, vj] = A[vi, vj]


@Ts.function
def elementwise_not_affine_fused(
    A: T.Tensor([127, 128], "float32"), B: T.Tensor([127, 128], "float32")
) -> None:
    for i in T.grid(4):
        for j_k_fused in T.serial(0, T.min(31, 126 - i * 32) * 128 + 128):
            with Ts.sblock("B"):
                vi = Ts.axis.S(
                    127,
                    i * 32 + T.floordiv(j_k_fused, 128),
                )
                vj = Ts.axis.S(128, T.floormod(j_k_fused, 128))
                Ts.reads([A[vi, vj]])
                Ts.writes([B[vi, vj]])
                B[vi, vj] = A[vi, vj]


# pylint: enable=no-member,invalid-name,unused-variable


def test_fuse():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    sch.fuse(i, j, k)
    assert_structural_equal_ignore_global_symbol(elementwise_fused, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


@pytest.mark.parametrize("disable_predication", [True, False])
def test_split(disable_predication):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[2, 1, 64], disable_predication=disable_predication)
    sch.split(j, factors=[4, 32], disable_predication=disable_predication)
    sch.split(k, factors=[16, 8], disable_predication=disable_predication)
    assert_structural_equal_ignore_global_symbol(elementwise_split_case0, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_split_with_inferred_factor():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[None, 1, 64])
    sch.split(j, factors=[2, None, 64])
    sch.split(k, factors=[2, 1, None])
    assert_structural_equal_ignore_global_symbol(elementwise_split_case1, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_split_with_dynamic_inferred_factor():
    N = T.dynamic("N", "int32")
    M = T.dynamic("M", "int32")

    @Ts.function
    def before(A: T.Tensor((N, 128, M), "float32"), B: T.Tensor((N, 128, M), "float32")) -> None:
        for i, j, k in T.grid(N, 128, M):
            with Ts.sblock("B"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0

    N = T.dynamic("N", "int32")
    M = T.dynamic("M", "int32")

    @Ts.function
    def expected(A: T.Tensor((N, 128, M), "float32"), B: T.Tensor((N, 128, M), "float32")) -> None:
        for i_0, i_1, j_0, j_1, k_0, k_1 in T.grid((N + 15) // 16, 16, 4, 32, 16, (M + 15) // 16):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(N, i_0 * 16 + i_1)
                vj = Ts.axis.spatial(128, j_0 * 32 + j_1)
                vk = Ts.axis.spatial(M, k_0 * ((M + 15) // 16) + k_1)
                Ts.where(i_0 * 16 + i_1 < N and k_0 * ((M + 15) // 16) + k_1 < M)
                B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2.0)

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[None, 16])
    sch.split(j, factors=[4, 32])
    sch.split(k, factors=[16, None])
    assert_structural_equal_ignore_global_symbol(expected, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=before)


def test_split_with_predicate():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[1000, 2, 3])
    sch.split(j, factors=[None, 129])
    sch.split(k, factors=[3, None])
    assert_structural_equal_ignore_global_symbol(elementwise_split_with_predicate, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_fuse_fail_not_only_child():
    sch = tvm.s_tir.Schedule(elementwise_with_seq, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.fuse(j, k)


def test_fuse_split_fail_with_annotation():
    sch = tvm.s_tir.Schedule(elementwise_with_anno, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_split_fail_not_start_with_zero():
    sch = tvm.s_tir.Schedule(elementwise_with_anno, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_with_opaque_block():
    sch = tvm.s_tir.Schedule(elementwise_with_opaque_block, debug_mask="all")
    block_opaque = sch.get_sblock("opaque")
    i, j, k = sch.get_loops(block_opaque)
    sch.fuse(i, j, k)
    assert_structural_equal_ignore_global_symbol(
        elementwise_fuse_with_opaque_block, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_with_opaque_block)


def test_fuse_with_opaque_access():
    sch = tvm.s_tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_sblock("A")
    i, j = sch.get_loops(block_a)
    sch.fuse(i, j)
    block_b = sch.get_sblock("B")
    i, j = sch.get_loops(block_b)
    sch.fuse(i, j)
    assert_structural_equal_ignore_global_symbol(opaque_access_fused, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_split_with_opaque_block():
    sch = tvm.s_tir.Schedule(elementwise_with_opaque_block, debug_mask="all")
    block_opaque = sch.get_sblock("opaque")
    i, _, _ = sch.get_loops(block_opaque)
    sch.split(i, factors=[None, 16])
    assert_structural_equal_ignore_global_symbol(
        elementwise_split_with_opaque_block, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_with_opaque_block)


def test_split_with_opaque_access():
    sch = tvm.s_tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_sblock("A")
    _, j = sch.get_loops(block_a)
    sch.split(j, factors=[None, 4])
    block_b = sch.get_sblock("B")
    _, j = sch.get_loops(block_b)
    sch.split(j, factors=[None, 4])
    assert_structural_equal_ignore_global_symbol(opaque_access_split, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_split_with_non_positive_factors():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.split(i, factors=[-2, -64])
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.split(j, factors=[0, None])
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.split(k, factors=[None, -16])


def test_fuse_split_fail_with_thread_binding():
    sch = tvm.s_tir.Schedule(elementwise_with_thread_binding, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.split(k, factors=[None, 10])


def test_fuse_symbolic():
    sch = tvm.s_tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    sch.fuse(i, j, k)
    assert_structural_equal_ignore_global_symbol(elementwise_symbolic_fused, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_symbolic)


def test_split_symbolic():
    sch = tvm.s_tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, _, k = sch.get_loops(block_b)
    sch.split(k, factors=[10, None])
    assert_structural_equal_ignore_global_symbol(elementwise_symbolic_split, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_symbolic)


def test_fuse_fail_with_dependent_loops():
    sch = tvm.s_tir.Schedule(elementwise_dependent_loops, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, _ = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.fuse(i, j)


def test_fuse_not_affine():
    sch = tvm.s_tir.Schedule(elementwise_not_affine, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, j, k = sch.get_loops(block_b)
    sch.fuse(j, k)
    assert_structural_equal_ignore_global_symbol(elementwise_not_affine_fused, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_not_affine)


def test_add_unit_loop_above_block():
    @Ts.function
    def zero_dim(
        A: T.Tensor((), "int32"),
        B: T.Tensor((), "int32"),
        C: T.Tensor((), "int32"),
    ) -> None:
        with Ts.sblock("C"):
            vi = Ts.axis.spatial(1, 0)
            C[()] = A[()] + B[()]

    @Ts.function
    def zero_dim_added(
        A: T.Tensor((), "int32"),
        B: T.Tensor((), "int32"),
        C: T.Tensor((), "int32"),
    ) -> None:
        for u in range(1):
            with Ts.sblock("C"):
                vi = Ts.axis.spatial(1, 0)
                C[()] = A[()] + B[()]

    sch = tvm.s_tir.Schedule(zero_dim, debug_mask="all")
    block = sch.get_sblock("C")
    sch.add_unit_loop(block)
    assert_structural_equal_ignore_global_symbol(zero_dim_added, sch.mod["main"])


def test_add_unit_loop_above_loop():
    @Ts.function
    def zero_dim(
        A: T.Tensor((), "int32"),
        B: T.Tensor((), "int32"),
        C: T.Tensor((), "int32"),
    ) -> None:
        for u in range(1):
            with Ts.sblock("C"):
                vi = Ts.axis.spatial(1, 0)
                C[()] = A[()] + B[()]

    @Ts.function
    def zero_dim_added(
        A: T.Tensor((), "int32"),
        B: T.Tensor((), "int32"),
        C: T.Tensor((), "int32"),
    ) -> None:
        for u1, u2 in T.grid(1, 1):
            with Ts.sblock("C"):
                vi = Ts.axis.spatial(1, 0)
                C[()] = A[()] + B[()]

    sch = tvm.s_tir.Schedule(zero_dim, debug_mask="all")
    block = sch.get_sblock("C")
    (loop,) = sch.get_loops(block)
    sch.add_unit_loop(loop)
    assert_structural_equal_ignore_global_symbol(zero_dim_added, sch.mod["main"])


@pytest.mark.skip("Pending fix in affine analysis")
def test_fuse_int64():
    def _create_function():
        n = te.const(16, "int32")
        m = te.const(32, "int64")
        A = te.placeholder((n, m), name="A", dtype="int32")
        B = te.compute((n, m), lambda i, j: A[i, j] + 1, name="B")
        return te.create_function([A, B])

    mod = _create_function()
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    i, j = sch.get_loops(sch.get_sblock("B"))
    sch.fuse(i, j)
    verify_trace_roundtrip(sch=sch, mod=mod)


def test_split_int64_extent_with_mixed_factors():
    def _create_function():
        m = te.const(384, "int64")
        A = te.placeholder((m,), name="A", dtype="float32")
        B = te.compute((m,), lambda i: A[i] + 1, name="B")
        return te.create_function([A, B])

    mod = _create_function()
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    (i,) = sch.get_loops(sch.get_sblock("B"))
    sch.split(
        i,
        factors=[
            te.const(1, "int64"),
            te.const(512, "int32"),
        ],
    )


def test_split_int64_extent_with_int32_factors():
    def _create_function():
        m = te.const(12, "int64")
        A = te.placeholder((m,), name="A", dtype="float32")
        B = te.compute((m,), lambda i: A[i] + 1, name="B")
        return te.create_function([A, B])

    mod = _create_function()
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    (i,) = sch.get_loops(sch.get_sblock("B"))
    sch.split(
        i,
        factors=[
            te.const(1, "int32"),
            te.const(1, "int32"),
            te.const(3, "int32"),
            te.const(1, "int32"),
            te.const(4, "int32"),
        ],
    )


def test_split_int64_factors():
    sch = tvm.s_tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, _, k = sch.get_loops(block_b)
    sch.split(k, factors=[IntImm(ty="int64", value=10), None])
    assert_structural_equal_ignore_global_symbol(elementwise_symbolic_split, sch.mod["main"])


def test_unsupported_target_scalable_split():
    @Ts.function
    def before(A: T.Tensor((128,), "float32")):
        T.func_attr({"global_symbol": "my_module", "tirx.noalias": True})
        for i in T.serial(128):
            with Ts.sblock("A"):
                v_i = Ts.axis.remap("S", [i])
                A[v_i] = 1.0

    sch = tvm.s_tir.Schedule(before)
    (a,) = sch.get_loops("A")

    err_msg = "The product of factors is not larger than or equal to the extent of loop ir.For#0"
    with pytest.raises(tvm.s_tir.schedule.ScheduleError, match=err_msg):
        sch.split(a, factors=[T.ceildiv(128, 4 * T.vscale()), 4 * T.vscale()])


def test_fused_symbolic_2D_tiling():
    @Ts.function
    def before(
        A: T.Tensor((M, N), "float32"),  # noqa: F821
        B: T.Tensor((M, N), "float32"),  # noqa: F821
        M: T.int32,
        N: T.int32,
    ) -> None:
        for i, j in T.grid(M, N):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0

    @Ts.function
    def expected(
        A: T.Tensor((M, N), "float32"),  # noqa: F821
        B: T.Tensor((M, N), "float32"),  # noqa: F821
        M: T.int32,
        N: T.int32,
    ) -> None:
        for i_0_j_0_fused, i_1, j_1 in T.grid(((M + 63) // 64) * ((N + 15) // 16), 64, 16):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(M, i_0_j_0_fused // ((N + 15) // 16) * 64 + i_1)
                vj = Ts.axis.spatial(N, i_0_j_0_fused % ((N + 15) // 16) * 16 + j_1)
                Ts.where(
                    i_0_j_0_fused // ((N + 15) // 16) * 64 + i_1 < M
                    and i_0_j_0_fused % ((N + 15) // 16) * 16 + j_1 < N
                )
                B[vi, vj] = A[vi, vj] * T.float32(2.0)

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j = sch.get_loops(block_b)
    i0, i1 = sch.split(i, factors=[None, 64])
    j0, j1 = sch.split(j, factors=[None, 16])
    sch.reorder(i0, j0, i1, j1)
    sch.fuse(i0, j0)
    assert_structural_equal_ignore_global_symbol(expected, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
