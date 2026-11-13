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
def element_wise(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


@Ts.function
def element_wise_parallelized(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")
) -> None:
    for i0 in T.parallel(0, 128):
        for i1 in T.serial(0, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i0, i1])
                B[vi, vj] = A[vi, vj] * 2.0


@Ts.function
def element_wise_i_bound(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")
) -> None:
    for i0 in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
        for i1 in T.serial(0, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i0, i1])
                B[vi, vj] = A[vi, vj] * 2.0


@Ts.function
def element_wise_compute_at_split(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i in T.serial(0, 128):
        for j0 in T.serial(0, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j0])
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o, j1i in T.grid(32, 4):
            with Ts.sblock("C"):
                vi = Ts.axis.S(128, i)
                vj = Ts.axis.S(128, j1o * 4 + j1i)
                C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def element_wise_compute_at_split_vectorized(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i in T.serial(0, 128):
        for j0 in T.serial(0, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j0])
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o in T.serial(0, 32):
            for j1i in T.vectorized(0, 4):
                with Ts.sblock("C"):
                    vi = Ts.axis.S(128, i)
                    vj = Ts.axis.S(128, j1o * 4 + j1i)
                    C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def element_wise_split_predicate(
    A: T.Tensor([128, 128], "float32"), B: T.Tensor([128, 128], "float32")
) -> None:
    for i, j_0, j_1 in T.grid(128, 13, 10):
        with Ts.sblock("B"):
            Ts.where(j_0 * 10 + j_1 < 128)
            vi = Ts.axis.S(128, i)
            vj = Ts.axis.S(128, j_0 * 10 + j_1)
            B[vi, vj] = A[vi, vj] * 2.0


@Ts.function
def element_wise_split_predicate_parallelized(
    A: T.Tensor([128, 128], "float32"), B: T.Tensor([128, 128], "float32")
) -> None:
    for i in T.serial(0, 128):
        for j_0 in T.parallel(0, 13):
            for j_1 in T.serial(0, 10):
                with Ts.sblock("B"):
                    Ts.where(j_0 * 10 + j_1 < 128)
                    vi = Ts.axis.S(128, i)
                    vj = Ts.axis.S(128, j_0 * 10 + j_1)
                    B[vi, vj] = A[vi, vj] * 2.0


@Ts.function
def element_wise_split_predicate_vectorized(
    A: T.Tensor([128, 128], "float32"), B: T.Tensor([128, 128], "float32")
) -> None:
    for i in T.vectorized(0, 128):
        for j_0, j_1 in T.grid(13, 10):
            with Ts.sblock("B"):
                Ts.where(j_0 * 10 + j_1 < 128)
                vi = Ts.axis.S(128, i)
                vj = Ts.axis.S(128, j_0 * 10 + j_1)
                B[vi, vj] = A[vi, vj] * 2.0


@Ts.function
def element_wise_compute_at_split_j0_j1o_bound(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i in T.serial(0, 128):
        for j0 in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j0])
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
            for j1i in T.serial(0, 4):
                with Ts.sblock("C"):
                    vi = Ts.axis.S(128, i)
                    vj = Ts.axis.S(128, j1o * 4 + j1i)
                    C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def matmul(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def rowsum(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128,), "float32")) -> None:
    for i, k in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vk = Ts.axis.remap("SR", [i, k])
            with Ts.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@Ts.function
def rowsum_unrolled(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128,), "float32")) -> None:
    for i0 in T.unroll(0, 128):
        for i1 in T.serial(0, 128):
            with Ts.sblock("B"):
                vi, vk = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@Ts.function
def rowsum_not_quasi_affine(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128,), "float32")
) -> None:
    for i, k in T.grid(128, 16):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, i)
            vk = Ts.axis.R(128, T.floordiv(k * k, 2))
            with Ts.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@Ts.function
def rowsum_not_compact_data_flow(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128,), "float32")
) -> None:
    for i, k in T.grid(128, 16):
        with Ts.sblock("B"):
            vi, vk = Ts.axis.remap("SR", [i, k])
            with Ts.init():
                B[vk] = 0.0
            B[vk] = B[vk] + A[vi, vk]


@Ts.function
def rowsum_cross_thread_reduction(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128,), "float32")
) -> None:
    for i0 in T.serial(0, 128):
        for i1 in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("B"):
                vi, vk = Ts.axis.remap("SR", [i0, i1])
                with Ts.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@Ts.function
def opaque_block(A: T.Tensor((16,), "float32")) -> None:
    for i in T.serial(0, 15):
        with Ts.sblock("opaque"):
            A[i + 1] = A[i + 1] + A[i]


@Ts.function
def block_inside_init(
    A: T.Tensor([128, 128, 128], dtype="float32"), B: T.Tensor([128, 128], dtype="float32")
) -> None:
    for i in T.serial(0, 128):
        with Ts.sblock("outer"):
            vi = Ts.axis.S(128, i)
            with Ts.init():
                for j in T.serial(0, 128):
                    with Ts.sblock("init"):
                        vj = Ts.axis.S(128, j)
                        B[vi, vj] = 0.0
            for k in T.serial(0, 128):
                for j in T.serial(0, 128):
                    with Ts.sblock("inner"):
                        vj, vk = Ts.axis.remap("SR", [j, k])
                        B[vi, vj] = B[vi, vj] + A[vi, vj, vk]


@Ts.function
def thread_bound_block_inside_init(
    A: T.Tensor([128, 128, 128], dtype="float32"), B: T.Tensor([128, 128], dtype="float32")
) -> None:
    for i in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
        with Ts.sblock("outer"):
            vi = Ts.axis.S(128, i)
            with Ts.init():
                for j in T.serial(0, 128):
                    with Ts.sblock("init"):
                        vj = Ts.axis.S(128, j)
                        B[vi, vj] = 0.0
            for k in T.serial(0, 128):
                for j in T.serial(0, 128):
                    with Ts.sblock("inner"):
                        vj, vk = Ts.axis.remap("SR", [j, k])
                        B[vi, vj] = B[vi, vj] + A[vi, vj, vk]


@Ts.function
def decomposed_gemm(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
):
    local = Ts.sblock_alloc_tensor((16, 16), "float32")
    for i, j in T.grid(4, 4):
        for ii, jj in T.grid(4, 4):
            with Ts.sblock("init"):
                vi = Ts.axis.S(16, i * 4 + ii)
                vj = Ts.axis.S(16, j * 4 + jj)
                local[vi, vj] = 0
        for k, ii, jj in T.grid(16, 4, 4):
            with Ts.sblock("update"):
                vi = Ts.axis.S(16, i * 4 + ii)
                vj = Ts.axis.S(16, j * 4 + jj)
                vk = Ts.axis.R(16, k)
                local[vi, vj] += A[vi, vk] * B[vj, vk]
        for ii, jj in T.grid(4, 4):
            with Ts.sblock("C"):
                vi = Ts.axis.S(16, i * 4 + ii)
                vj = Ts.axis.S(16, j * 4 + jj)
                C[vi, vj] = local[vi, vj]


@Ts.function
def decomposed_gemm_after_vectorize(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
):
    local = Ts.sblock_alloc_tensor((16, 16), "float32")
    for i, j in T.grid(4, 4):
        for ii, jj in T.grid(4, 4):
            with Ts.sblock("init"):
                vi = Ts.axis.S(16, i * 4 + ii)
                vj = Ts.axis.S(16, j * 4 + jj)
                local[vi, vj] = 0
        for k, ii, jj in T.grid(16, 4, 4):
            with Ts.sblock("update"):
                vi = Ts.axis.S(16, i * 4 + ii)
                vj = Ts.axis.S(16, j * 4 + jj)
                vk = Ts.axis.R(16, k)
                local[vi, vj] += A[vi, vk] * B[vj, vk]
        for ii in range(4):
            for jj in T.vectorized(4):
                with Ts.sblock("C"):
                    vi = Ts.axis.S(16, i * 4 + ii)
                    vj = Ts.axis.S(16, j * 4 + jj)
                    C[vi, vj] = local[vi, vj]


@Ts.function
def nested_block_bind(
    A: T.Tensor((16, 16, 16, 16), "float32"), B: T.Tensor((16, 16, 16), "float32")
):
    for i, j in T.grid(16, 16):
        with Ts.sblock("outer"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            for k, l in T.grid(16, 16):
                with Ts.sblock("inner"):
                    vk, vl = Ts.axis.remap("SR", [k, l])
                    with Ts.init():
                        B[vi, vj, vk] = 0.0
                    B[vi, vj, vk] = B[vi, vj, vk] + A[vi, vj, vk, vl]


@Ts.function
def thread_bound_nested_block(
    A: T.Tensor((16, 16, 16, 16), "float32"), B: T.Tensor((16, 16, 16), "float32")
) -> None:
    for i in T.serial(16):
        for j in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
            with Ts.sblock("outer"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                for k in T.serial(16):
                    for l in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("inner"):
                            vk, vl = Ts.axis.remap("SR", [k, l])
                            with Ts.init():
                                B[vi, vj, vk] = T.float32(0)
                            B[vi, vj, vk] = B[vi, vj, vk] + A[vi, vj, vk, vl]


@Ts.function
def nested_block_bind_after_cache_read(
    A: T.Tensor((16, 16), "float32"), B: T.Tensor((16,), "float32")
) -> None:
    for i in T.serial(16):
        with Ts.sblock("outer"):
            vi = Ts.axis.spatial(16, i)
            A_shared = Ts.sblock_alloc_tensor([1, 16], dtype="float32", scope="shared")
            for ax0, ax1 in T.grid(1, 16):
                with Ts.sblock("A_shared"):
                    v0 = Ts.axis.spatial(16, vi + ax0)
                    v1 = Ts.axis.spatial(16, ax1)
                    A_shared[v0, v1] = A[v0, v1]
            for j in T.serial(16):
                with Ts.sblock("inner"):
                    vj = Ts.axis.reduce(16, j)
                    with Ts.init():
                        B[vi] = T.float32(0)
                    B[vi] = B[vi] + A_shared[vi, vj]


@Ts.function
def thread_bound_nested_block_after_cache_read(
    A: T.Tensor((16, 16), "float32"), B: T.Tensor((16,), "float32")
) -> None:
    for i in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
        with Ts.sblock("outer"):
            vi = Ts.axis.spatial(16, i)
            A_shared = Ts.sblock_alloc_tensor([1, 16], dtype="float32", scope="shared")
            for ax0, ax1 in T.grid(1, 16):
                with Ts.sblock("A_shared"):
                    v0 = Ts.axis.spatial(16, vi + ax0)
                    v1 = Ts.axis.spatial(16, ax1)
                    A_shared[v0, v1] = A[v0, v1]
            for j in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("inner"):
                    vj = Ts.axis.reduce(16, j)
                    with Ts.init():
                        B[vi] = T.float32(0)
                    B[vi] = B[vi] + A_shared[vi, vj]


@Ts.function
def decomposed_gemm_parallelize_init(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
) -> None:
    local = Ts.sblock_alloc_tensor([16, 16], dtype="float32")
    for i, j in T.grid(4, 4):
        for ii in T.serial(4):
            for jj in T.vectorized(4):
                with Ts.sblock("init"):
                    vi = Ts.axis.spatial(16, i * 4 + ii)
                    vj = Ts.axis.spatial(16, j * 4 + jj)
                    Ts.reads()
                    Ts.writes(local[vi, vj])
                    local[vi, vj] = 0
        for k, ii, jj in T.grid(16, 4, 4):
            with Ts.sblock("update"):
                vi = Ts.axis.spatial(16, i * 4 + ii)
                vj = Ts.axis.spatial(16, j * 4 + jj)
                vk = Ts.axis.reduce(16, k)
                Ts.reads(local[vi, vj], A[vi, vk], B[vj, vk])
                Ts.writes(local[vi, vj])
                local[vi, vj] = local[vi, vj] + A[vi, vk] * B[vj, vk]
        for ii, jj in T.grid(4, 4):
            with Ts.sblock("C"):
                vi = Ts.axis.spatial(16, i * 4 + ii)
                vj = Ts.axis.spatial(16, j * 4 + jj)
                Ts.reads(local[vi, vj])
                Ts.writes(C[vi, vj])
                C[vi, vj] = local[vi, vj]


@Ts.function
def scatter_compute(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
    for i in T.grid(8):
        with Ts.sblock("first_half"):
            vi = Ts.axis.spatial(16, 8 + i)
            B[vi] = A[vi - 8]

    for i in T.grid(8):
        with Ts.sblock("last_half"):
            vi = Ts.axis.spatial(16, i)
            B[vi] = A[vi + 8]


@Ts.function
def scatter_compute_parallelize(
    A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")
) -> None:
    # body
    # with Ts.sblock("root")
    for i in T.parallel(8):
        with Ts.sblock("first_half"):
            vi = Ts.axis.spatial(16, 8 + i)
            Ts.reads(A[vi - 8])
            Ts.writes(B[vi])
            B[vi] = A[vi - 8]
    for i in T.parallel(8):
        with Ts.sblock("last_half"):
            vi = Ts.axis.spatial(16, i)
            Ts.reads(A[vi + 8])
            Ts.writes(B[vi])
            B[vi] = A[vi + 8]


# pylint: enable=no-member,invalid-name,unused-variable


def test_parallel():
    s = tvm.s_tir.Schedule(element_wise, debug_mask="all")
    i, _ = s.get_loops(s.get_sblock("B"))
    s.parallel(i)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], element_wise_parallelized)
    verify_trace_roundtrip(s, mod=element_wise)


def test_parallel_predicate():
    s = tvm.s_tir.Schedule(element_wise_split_predicate, debug_mask="all")
    _, j, _ = s.get_loops(s.get_sblock("B"))
    s.parallel(j)
    assert_structural_equal_ignore_global_symbol(
        s.mod["main"], element_wise_split_predicate_parallelized
    )
    verify_trace_roundtrip(s, mod=element_wise_split_predicate)


def test_parallel_reduction_block_iter():
    s = tvm.s_tir.Schedule(matmul, debug_mask="all")
    _, _, k = s.get_loops(s.get_sblock("C"))
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.parallel(k)


def test_parallel_not_quasi_affine():
    s = tvm.s_tir.Schedule(rowsum_not_quasi_affine, debug_mask="all")
    i, _ = s.get_loops(s.get_sblock("B"))
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.parallel(i)


def test_parallel_not_compact_data_flow():
    s = tvm.s_tir.Schedule(rowsum_not_compact_data_flow, debug_mask="all")
    i, _ = s.get_loops(s.get_sblock("B"))
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.parallel(i)


def test_vectorize():
    s = tvm.s_tir.Schedule(element_wise_compute_at_split, debug_mask="all")
    _, _, j1i = s.get_loops(s.get_sblock("C"))
    s.vectorize(j1i)
    assert_structural_equal_ignore_global_symbol(
        s.mod["main"], element_wise_compute_at_split_vectorized
    )
    verify_trace_roundtrip(s, mod=element_wise_compute_at_split)


def test_vectorize_predicate():
    s = tvm.s_tir.Schedule(element_wise_split_predicate, debug_mask="all")
    i, _, _ = s.get_loops(s.get_sblock("B"))
    s.vectorize(i)
    assert_structural_equal_ignore_global_symbol(
        s.mod["main"], element_wise_split_predicate_vectorized
    )
    verify_trace_roundtrip(s, mod=element_wise_split_predicate)


def test_vectorize_opaque_block():
    s = tvm.s_tir.Schedule(opaque_block, debug_mask="all")
    (i,) = s.get_loops(s.get_sblock("opaque"))
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.vectorize(i)


def test_unroll():
    s = tvm.s_tir.Schedule(rowsum, debug_mask="all")
    i, _ = s.get_loops(s.get_sblock("B"))
    s.unroll(i)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], rowsum_unrolled)
    verify_trace_roundtrip(s, mod=rowsum)


def test_unroll_after_bind():
    s = tvm.s_tir.Schedule(rowsum, debug_mask="all")
    i, _ = s.get_loops(s.get_sblock("B"))
    s.bind(i, "blockIdx.x")
    s.unroll(i)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], rowsum_unrolled)
    verify_trace_roundtrip(s, mod=rowsum)


def test_bind1():
    s = tvm.s_tir.Schedule(element_wise, debug_mask="all")
    i, _ = s.get_loops(s.get_sblock("B"))
    s.bind(i, "threadIdx.x")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], element_wise_i_bound)
    verify_trace_roundtrip(s, mod=element_wise)


def test_bind2():
    s = tvm.s_tir.Schedule(element_wise_compute_at_split, debug_mask="all")
    _, j0 = s.get_loops(s.get_sblock("B"))
    _, j1o, _ = s.get_loops(s.get_sblock("C"))
    s.bind(j0, "threadIdx.x")
    s.bind(j1o, "threadIdx.x")
    assert_structural_equal_ignore_global_symbol(
        s.mod["main"], element_wise_compute_at_split_j0_j1o_bound
    )
    verify_trace_roundtrip(s, mod=element_wise_compute_at_split)


def test_bind_cross_thread_reduction():
    s = tvm.s_tir.Schedule(rowsum, debug_mask="all")
    _, k = s.get_loops(s.get_sblock("B"))
    s.bind(k, "threadIdx.x")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], rowsum_cross_thread_reduction)
    verify_trace_roundtrip(s, mod=rowsum)


def test_bind_not_cross_thread_reduction():
    s = tvm.s_tir.Schedule(rowsum, debug_mask="all")
    _, k = s.get_loops(s.get_sblock("B"))
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.bind(k, "blockIdx.x")


def test_bind_after_bind():
    s = tvm.s_tir.Schedule(element_wise, debug_mask="all")
    i, _ = s.get_loops(s.get_sblock("B"))
    s.bind(i, "blockIdx.x")
    s.bind(i, "threadIdx.x")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], element_wise_i_bound)
    verify_trace_roundtrip(s, mod=element_wise)


def test_block_inside_init():
    s = tvm.s_tir.Schedule(block_inside_init, debug_mask="all")
    (i,) = s.get_loops(s.get_sblock("outer"))
    s.bind(i, "threadIdx.x")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], thread_bound_block_inside_init)
    verify_trace_roundtrip(s, mod=block_inside_init)


def test_vectorize_after_decompose():
    s = tvm.s_tir.Schedule(decomposed_gemm, debug_mask="all")
    jj = s.get_loops(s.get_sblock("C"))[-1]
    s.vectorize(jj)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], decomposed_gemm_after_vectorize)
    verify_trace_roundtrip(s, mod=decomposed_gemm)


def test_nested_block_bind():
    s = tvm.s_tir.Schedule(nested_block_bind)
    block_outer = s.get_sblock("outer")
    block_inner = s.get_sblock("inner")
    _, j = s.get_loops(block_outer)
    _, l = s.get_loops(block_inner)
    s.bind(l, "threadIdx.x")
    s.bind(j, "blockIdx.x")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], thread_bound_nested_block)
    verify_trace_roundtrip(s, mod=nested_block_bind)


def test_nexted_block_bind_after_cache_read():
    s = tvm.s_tir.Schedule(nested_block_bind_after_cache_read)
    block_outer = s.get_sblock("outer")
    block_inner = s.get_sblock("inner")
    (i,) = s.get_loops(block_outer)
    (j,) = s.get_loops(block_inner)
    s.bind(i, "blockIdx.x")
    s.bind(j, "threadIdx.x")
    assert_structural_equal_ignore_global_symbol(
        s.mod["main"], thread_bound_nested_block_after_cache_read
    )
    verify_trace_roundtrip(s, mod=nested_block_bind_after_cache_read)


def test_vectorize_init():
    s = tvm.s_tir.Schedule(decomposed_gemm, debug_mask="all")
    init_blk = s.get_sblock("init")
    upd_blk = s.get_sblock("update")
    _, _, ii_0, jj_0 = s.get_loops(init_blk)
    _, _, k_1, ii_1, jj_1 = s.get_loops(upd_blk)
    s.vectorize(jj_0)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], decomposed_gemm_parallelize_init)
    verify_trace_roundtrip(s, mod=decomposed_gemm)


def test_scatter_parallelize():
    s = tvm.s_tir.Schedule(scatter_compute, debug_mask="all")
    first = s.get_sblock("first_half")
    last = s.get_sblock("last_half")
    (i_0,) = s.get_loops(first)
    (i_1,) = s.get_loops(last)
    s.parallel(i_0)
    s.parallel(i_1)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], scatter_compute_parallelize)
    verify_trace_roundtrip(s, mod=scatter_compute)


def test_bind_thread_iter_var_dtype():
    @Ts.function(private=True)
    def before(
        A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0

    @Ts.function(private=True)
    def expected(
        A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i0 in T.parallel(T.int64(128), annotations={"thread_binding": "threadIdx.x"}):
            # Use T.serial with explicit int64 min so the inner sblock iter_var dom
            # is all-int64 (matches what `s.bind` emits; `range(T.int64(128))` parses
            # min as int32 even when extent is int64).
            for i1 in T.serial(T.int64(0), T.int64(128)):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i0, i1])
                    B[vi, vj] = A[vi, vj] * 2.0

    s = tvm.s_tir.Schedule(before, debug_mask="all")
    i, _ = s.get_loops(s.get_sblock("B"))
    s.bind(i, "threadIdx.x")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], expected)
    verify_trace_roundtrip(s, mod=before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
