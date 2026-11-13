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
# ruff: noqa: F401
import sys

import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm_tirx import tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@Ts.function
def rowsum_blockized(A: T.Tensor([32, 4, 128], "float32"), B: T.Tensor([32, 4], "float32")) -> None:
    for i0, i2_0 in T.grid(32, 16):
        with Ts.sblock("blockized_B"):
            io, ko = Ts.axis.remap("SR", [i0, i2_0])
            with Ts.init():
                for i1 in T.serial(0, 4):
                    with Ts.sblock("B_init"):
                        ii_init = Ts.axis.S(4, i1)
                        B[io, ii_init] = 0.0
            for i1_1, i2_1 in T.grid(4, 8):
                with Ts.sblock("B"):
                    ii = Ts.axis.S(4, i1_1)
                    k = Ts.axis.R(128, ko * 8 + i2_1)
                    B[io, ii] = B[io, ii] + A[io, ii, k]


@Ts.function
def matmul(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def matmul_decompose0(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("init"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = 0.0

    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def matmul_decompose1(
    A: T.Tensor([32, 4, 128], "float32", align=64),
    B: T.Tensor([32, 4], "float32", align=64),
) -> None:
    for i0 in T.serial(0, 32):
        with Ts.sblock("blockized_B_init"):
            io = Ts.axis.S(32, i0)
            for i1 in T.serial(0, 4):
                with Ts.sblock("B_init"):
                    ii = Ts.axis.S(4, i1)
                    B[io, ii] = T.float32(0)
    for i0, i2_o in T.grid(32, 16):
        with Ts.sblock("blockized_B_update"):
            io, ko = Ts.axis.remap("SR", [i0, i2_o])
            for i1, i2_i in T.grid(4, 8):
                with Ts.sblock("B"):
                    ii = Ts.axis.S(4, i1)
                    k = Ts.axis.R(128, ko * 8 + i2_i)
                    B[io, ii] = B[io, ii] + A[io, ii, k]


@Ts.function
def matmul_decompose2(
    A: T.Tensor([128, 128], "float32", align=64),
    B: T.Tensor([128, 128], "float32", align=64),
    C: T.Tensor([128, 128], "float32", align=64),
) -> None:
    for i0, i1 in T.grid(128, 128):
        with Ts.sblock("update_init"):
            vi_init, vj_init = Ts.axis.remap("SS", [i0, i1])
            C[vi_init, vj_init] = T.float32(0)
        for i2 in T.serial(0, 128):
            with Ts.sblock("update_update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i0, i1, i2])
                C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@Ts.function
def matmul_decompose_fail3(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, k, j in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def matmul_decompose4(
    A: T.Tensor([128, 128], "float32", align=64),
    B: T.Tensor([128, 128], "float32", align=64),
    C: T.Tensor([128, 128], "float32", align=64),
) -> None:
    # body
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        for i0_0 in T.serial(0, 16):
            for i0_1_init, i1_init in T.grid(8, 128):
                with Ts.sblock("update_init"):
                    vi_init = Ts.axis.S(128, i0_0 * 8 + i0_1_init)
                    vj_init = Ts.axis.S(128, i1_init)
                    C[vi_init, vj_init] = T.float32(0)
            for i0_1, i1, i2_0, i2_1 in T.grid(8, 128, 19, 7):
                with Ts.sblock("update_update"):
                    Ts.where(((i2_0 * 7) + i2_1) < 128)
                    vi = Ts.axis.S(128, i0_0 * 8 + i0_1)
                    vj = Ts.axis.S(128, i1)
                    vk = Ts.axis.R(128, i2_0 * 7 + i2_1)
                    C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@Ts.function
def matmul_with_annotation(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            Ts.sblock_attr({"test_annotation": 1})
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def matmul_decompose_with_annotation(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("init"):
            Ts.sblock_attr({"test_annotation": 1})
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = 0.0

    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            Ts.sblock_attr({"test_annotation": 1})
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def colsum_with_vectorization(
    A: T.Tensor([128, 32], dtype="float32"), B: T.Tensor([32], dtype="float32")
) -> None:
    for k in T.serial(0, 128):
        for i in T.vectorized(0, 32):
            with Ts.sblock("B"):
                vk, vi = Ts.axis.remap("RS", [k, i])
                with Ts.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vk, vi]


@Ts.function
def colsum_decompose_with_vectorization(
    A: T.Tensor([128, 32], dtype="float32"), B: T.Tensor([32], dtype="float32")
) -> None:
    for i in T.vectorized(0, 32):
        with Ts.sblock("B_init"):
            vi = Ts.axis.S(32, i)
            B[vi] = T.float32(0)
    for k in T.serial(0, 128):
        for i in T.vectorized(0, 32):
            with Ts.sblock("B"):
                vk, vi = Ts.axis.remap("RS", [k, i])
                B[vi] = B[vi] + A[vk, vi]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


def test_reduction_decompose0(use_block_name):
    s = tvm.s_tir.Schedule(matmul, debug_mask="all")
    C = "update" if use_block_name else s.get_sblock("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, i)
    assert_structural_equal_ignore_global_symbol(matmul_decompose0, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


def test_reduction_decompose1(use_block_name):
    s = tvm.s_tir.Schedule(rowsum_blockized, debug_mask="all")
    blockized_B = "blockized_B" if use_block_name else s.get_sblock("blockized_B")
    io, ko = s.get_loops(blockized_B)
    s.decompose_reduction(blockized_B, io)
    assert_structural_equal_ignore_global_symbol(matmul_decompose1, s.mod["main"])
    verify_trace_roundtrip(s, mod=rowsum_blockized)


def test_reduction_decompose2():
    s = tvm.s_tir.Schedule(matmul, debug_mask="all")
    C = s.get_sblock("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, k)
    assert_structural_equal_ignore_global_symbol(matmul_decompose2, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


def test_reduction_decompose3():
    s = tvm.s_tir.Schedule(matmul_decompose_fail3, debug_mask="all")
    C = s.get_sblock("update")
    i, j, k = s.get_loops(C)
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.decompose_reduction(C, k)


def test_reduction_decompose4():
    s = tvm.s_tir.Schedule(matmul, debug_mask="all")
    C = s.get_sblock("update")
    i, j, k = s.get_loops(C)
    io, ii = s.split(i, factors=[16, 8])
    ko, ki = s.split(k, factors=[19, 7])
    s.decompose_reduction(C, ii)
    assert_structural_equal_ignore_global_symbol(matmul_decompose4, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


def test_reduction_decompose_with_annotation():
    s = tvm.s_tir.Schedule(matmul_with_annotation, debug_mask="all")
    C = s.get_sblock("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, i)
    assert_structural_equal_ignore_global_symbol(matmul_decompose_with_annotation, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul_with_annotation)


def test_reduction_decompose_with_different_for_kind():
    s = tvm.s_tir.Schedule(colsum_with_vectorization, debug_mask="all")
    B = s.get_sblock("B")
    k, _ = s.get_loops(B)
    B_init = s.decompose_reduction(B, k)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], colsum_decompose_with_vectorization)
    assert s.get(B).same_as(s.get(s.get_sblock("B_update")))
    assert s.get(B_init).same_as(s.get(s.get_sblock("B_init")))
    verify_trace_roundtrip(s, mod=colsum_with_vectorization)


def test_decompose_reduction_ref_hash_check():
    mod = tvm_tirx.IRModule.from_expr(matmul.with_attr("global_symbol", "main"))
    mod_bak = mod
    hash_before = tvm_ffi.structural_hash(mod_bak)
    s = tvm.s_tir.Schedule(mod["main"], debug_mask="all")
    C = s.get_sblock("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, k)
    hash_after = tvm_ffi.structural_hash(mod_bak)
    assert hash_before == hash_after


def test_decompose_reduction_nested_block():
    @Ts.function
    def nested_block(A: T.Tensor((1, 64), "float32"), B: T.Tensor((1,), "float32")):
        for i, ko in T.grid(1, 2):
            with Ts.sblock("outer"):
                vi, vko = Ts.axis.remap("SR", [i, ko])
                C = Ts.sblock_alloc_tensor((32,), dtype="float32")
                with Ts.init():
                    B[vi] = T.float32(0)
                for ki in T.serial(32):
                    with Ts.sblock("inner_1"):
                        vki = Ts.axis.remap("S", [ki])
                        C[vki] = A[vi, vko * 32 + vki]
                for ki in T.serial(32):
                    with Ts.sblock("inner_2"):
                        vki = Ts.axis.remap("R", [ki])
                        B[vi] += C[vki]

    @Ts.function
    def decomposed_nested_block(A: T.Tensor((1, 64), "float32"), B: T.Tensor((1,), "float32")):
        for i in range(1):
            with Ts.sblock("outer_init"):
                vi = Ts.axis.spatial(1, i)
                Ts.reads()
                Ts.writes(B[vi])
                B[vi] = T.float32(0)
            for ko in range(2):
                with Ts.sblock("outer_update"):
                    vi, vko = Ts.axis.remap("SR", [i, ko])
                    Ts.reads(B[vi], A[vi, vko * 32 : vko * 32 + 32])
                    Ts.writes(B[vi])
                    C = Ts.sblock_alloc_tensor((32,))
                    for ki in range(32):
                        with Ts.sblock("inner_1"):
                            vki = Ts.axis.spatial(32, ki)
                            Ts.reads(A[vi, vko * 32 + vki])
                            Ts.writes(C[vki])
                            C[vki] = A[vi, vko * 32 + vki]
                    for ki in range(32):
                        with Ts.sblock("inner_2"):
                            vki = Ts.axis.reduce(32, ki)
                            Ts.reads(B[vi], C[vki])
                            Ts.writes(B[vi])
                            B[vi] = B[vi] + C[vki]

    sch = tvm.s_tir.Schedule(nested_block, debug_mask="all")
    outer = sch.get_sblock("outer")
    i, ko = sch.get_loops(outer)
    sch.decompose_reduction(outer, ko)

    assert_structural_equal_ignore_global_symbol(decomposed_nested_block, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=nested_block)


def test_decompose_reduction_with_thread_binding():
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((32, 16), "float32"), B: T.Tensor((32,), "float32")):
            for t in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
                for r in T.serial(16):
                    with Ts.sblock("B"):
                        vi, vr = Ts.axis.remap("SR", [t, r])
                        with Ts.init():
                            B[vi] = T.float32(0)
                        B[vi] += A[vi, vr]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((32, 16), "float32"), B: T.Tensor((32,), "float32")):
            for t_init in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("B_init"):
                    vi = Ts.axis.remap("S", [t_init])
                    B[vi] = T.float32(0)
            for t in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
                for r in T.serial(16):
                    with Ts.sblock("B"):
                        vi, vr = Ts.axis.remap("SR", [t, r])
                        B[vi] += A[vi, vr]

    sch = tvm.s_tir.Schedule(Before)
    t, _ = sch.get_loops("B")
    sch.decompose_reduction("B", t)
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_decompose_reduction_preserves_general_spatial_predicates():
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((8, 8), "float32"), B: T.Tensor((8,), "float32")):
            for i, k in T.grid(10, 10):
                with Ts.sblock("B"):
                    Ts.where(1 <= i and i < 9 and 1 <= k and k < 9)
                    vi = Ts.axis.spatial(8, i - 1)
                    vk = Ts.axis.reduce(8, k - 1)
                    with Ts.init():
                        B[vi] = T.float32(0)
                    B[vi] += A[vi, vk]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((8, 8), "float32"), B: T.Tensor((8,), "float32")):
            for i_init in range(10):
                with Ts.sblock("B_init"):
                    Ts.where(1 <= i_init and i_init < 9)
                    vi = Ts.axis.spatial(8, i_init - 1)
                    B[vi] = T.float32(0)
            for i, k in T.grid(10, 10):
                with Ts.sblock("B_update"):
                    Ts.where(1 <= i and i < 9 and 1 <= k and k < 9)
                    vi = Ts.axis.spatial(8, i - 1)
                    vk = Ts.axis.reduce(8, k - 1)
                    B[vi] += A[vi, vk]

    sch = tvm.s_tir.Schedule(Before)
    i, _ = sch.get_loops("B")
    sch.decompose_reduction("B", i)
    tvm_tirx.ir.assert_structural_equal(sch.mod, Expected)


def test_decompose_reduction_drops_mixed_rfactor_bound():
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((20,), "float32"), B: T.Tensor((), "float32")):
            for k in range(20):
                with Ts.sblock("B"):
                    vk = Ts.axis.reduce(20, k)
                    with Ts.init():
                        B[()] = T.float32(0)
                    B[()] += A[vk]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((20,), "float32"), B: T.Tensor((), "float32")):
            B_rf = Ts.sblock_alloc_tensor((16,))
            for k_1_init in range(16):
                with Ts.sblock("B_rf_init"):
                    vk_1 = Ts.axis.spatial(16, k_1_init)
                    B_rf[vk_1] = T.float32(0)
            for k_0, k_1 in T.grid(2, 16):
                with Ts.sblock("B_rf_update"):
                    vk_1, vk_0 = Ts.axis.remap("SR", [k_1, k_0])
                    Ts.where(k_0 * 16 + k_1 < 20)
                    B_rf[vk_1] += A[vk_0 * 16 + vk_1]
            for k_1 in range(16):
                with Ts.sblock("B"):
                    vk_1 = Ts.axis.reduce(16, k_1)
                    with Ts.init():
                        B[()] = T.float32(0)
                    B[()] += B_rf[vk_1]

    sch = tvm.s_tir.Schedule(Before)
    (k,) = sch.get_loops("B")
    _, k_1 = sch.split(k, factors=[None, 16])
    rf = sch.rfactor(k_1, 0)
    k_0, _ = sch.get_loops(rf)
    sch.decompose_reduction(rf, k_0)
    tvm_tirx.ir.assert_structural_equal(sch.mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
