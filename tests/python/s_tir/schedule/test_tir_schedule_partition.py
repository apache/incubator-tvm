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
def elementwise_partition_with_opaque_block(
    A: T.Tensor([128, 128, 128], "float32"), B: T.Tensor([128, 128, 128], "float32")
) -> None:
    with Ts.sblock("root"):
        Ts.reads()
        Ts.writes()
        with Ts.sblock("opaque_i_common"):
            Ts.reads()
            Ts.writes()
            with Ts.sblock("opaque_i0_partition"):
                Ts.reads()
                Ts.writes()
                for i0, j, k in T.grid(112, 128, 128):
                    with Ts.sblock("opaque_i0"):
                        Ts.reads(A[i0, j, k])
                        Ts.writes(B[i0, j, k])
                        with Ts.sblock("B_i0"):
                            vi, vj, vk = Ts.axis.remap("SSS", [i0, j, k])
                            Ts.reads(A[0:112, 0:128, 0:128])
                            Ts.writes(B[0:112, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with Ts.sblock("opaque_i1_partition"):
                Ts.reads()
                Ts.writes()
                for i1 in range(112, 128):
                    for j, k in T.grid(128, 128):
                        with Ts.sblock("opaque_i1"):
                            Ts.reads(A[i1, j, k])
                            Ts.writes(B[i1, j, k])
                            with Ts.sblock("B_i1"):
                                vi, vj, vk = Ts.axis.remap("SSS", [i1, j, k])
                                Ts.reads(A[112:128, 0:128, 0:128])
                                Ts.writes(B[112:128, 0:128, 0:128])
                                B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)


@Ts.function
def elementwise_loop_partition_case0(
    A: T.Tensor([128, 128, 128], "float32"), B: T.Tensor([128, 128, 128], "float32")
) -> None:
    with Ts.sblock("root"):
        Ts.reads()
        Ts.writes()
        with Ts.sblock("B_i_common"):
            Ts.reads()
            Ts.writes()
            with Ts.sblock("B_i0_partition"):
                Ts.reads()
                Ts.writes()
                for i0 in range(2):
                    with Ts.sblock("B_i0_j_common"):
                        Ts.reads()
                        Ts.writes()
                        with Ts.sblock("B_i0_j0_partition"):
                            Ts.reads()
                            Ts.writes()
                            for j0, k in T.grid(4, 128):
                                with Ts.sblock("B_i0_j0"):
                                    vi, vj, vk = Ts.axis.remap("SSS", [i0, j0, k])
                                    Ts.reads(A[0:2, 0:4, 0:128])
                                    Ts.writes(B[0:2, 0:4, 0:128])
                                    B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
                        with Ts.sblock("B_i0_j1_partition"):
                            Ts.reads()
                            Ts.writes()
                            for j1 in range(4, 36):
                                for k in range(128):
                                    with Ts.sblock("B_i0_j1"):
                                        vi, vj, vk = Ts.axis.remap("SSS", [i0, j1, k])
                                        Ts.reads(A[0:2, 4:36, 0:128])
                                        Ts.writes(B[0:2, 4:36, 0:128])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
                        with Ts.sblock("B_i0_j2_partition"):
                            Ts.reads()
                            Ts.writes()
                            for j2 in range(36, 128):
                                for k in range(128):
                                    with Ts.sblock("B_i0_j2"):
                                        vi, vj, vk = Ts.axis.remap("SSS", [i0, j2, k])
                                        Ts.reads(A[0:2, 36:128, 0:128])
                                        Ts.writes(B[0:2, 36:128, 0:128])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with Ts.sblock("B_i1_partition"):
                Ts.reads()
                Ts.writes()
                for i1 in range(2, 3):
                    for j, k in T.grid(128, 128):
                        with Ts.sblock("B_i1"):
                            vi, vj, vk = Ts.axis.remap("SSS", [i1, j, k])
                            Ts.reads(A[2, 0:128, 0:128])
                            Ts.writes(B[2, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with Ts.sblock("B_i2_partition"):
                Ts.reads()
                Ts.writes()
                for i2 in range(3, 67):
                    for j, k in T.grid(128, 128):
                        with Ts.sblock("B_i2"):
                            vi, vj, vk = Ts.axis.remap("SSS", [i2, j, k])
                            Ts.reads(A[3:67, 0:128, 0:128])
                            Ts.writes(B[3:67, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with Ts.sblock("B_i3_partition"):
                Ts.reads()
                Ts.writes()
                for i3 in range(67, 128):
                    for j, k in T.grid(128, 128):
                        with Ts.sblock("B_i3"):
                            vi, vj, vk = Ts.axis.remap("SSS", [i3, j, k])
                            Ts.reads(A[67:128, 0:128, 0:128])
                            Ts.writes(B[67:128, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)


@Ts.function
def elementwise_loop_partition_case1(
    A: T.Tensor([128, 128, 128], "float32"), B: T.Tensor([128, 128, 128], "float32")
) -> None:
    with Ts.sblock("root"):
        Ts.reads()
        Ts.writes()
        with Ts.sblock("B_i_common"):
            Ts.reads()
            Ts.writes()
            with Ts.sblock("B_i0_partition"):
                Ts.reads()
                Ts.writes()
                for i0, j, k in T.grid(63, 128, 128):
                    with Ts.sblock("B_i0"):
                        vi, vj, vk = Ts.axis.remap("SSS", [i0, j, k])
                        Ts.reads(A[0:63, 0:128, 0:128])
                        Ts.writes(B[0:63, 0:128, 0:128])
                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with Ts.sblock("B_i1_partition"):
                Ts.reads()
                Ts.writes()
                for i1 in range(63, 64):
                    for j in range(128):
                        with Ts.sblock("B_i1_k_common"):
                            Ts.reads()
                            Ts.writes()
                            with Ts.sblock("B_i1_k0_partition"):
                                Ts.reads()
                                Ts.writes()
                                for k0 in range(1):
                                    with Ts.sblock("B_i1_k0"):
                                        vi, vj, vk = Ts.axis.remap("SSS", [i1, j, k0])
                                        Ts.reads(A[63, 0:128, 0])
                                        Ts.writes(B[63, 0:128, 0])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
                            with Ts.sblock("B_i1_k1_partition"):
                                Ts.reads()
                                Ts.writes()
                                for k1 in range(1, 65):
                                    with Ts.sblock("B_i1_k1"):
                                        vi, vj, vk = Ts.axis.remap("SSS", [i1, j, k1])
                                        Ts.reads(A[63, 0:128, 1:65])
                                        Ts.writes(B[63, 0:128, 1:65])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
                            with Ts.sblock("B_i1_k2_partition"):
                                Ts.reads()
                                Ts.writes()
                                for k2 in range(65, 128):
                                    with Ts.sblock("B_i1_k2"):
                                        vi, vj, vk = Ts.axis.remap("SSS", [i1, j, k2])
                                        Ts.reads(A[63, 0:128, 65:128])
                                        Ts.writes(B[63, 0:128, 65:128])
                                        B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)
            with Ts.sblock("B_i2_partition"):
                Ts.reads()
                Ts.writes()
                for i2 in range(64, 128):
                    for j, k in T.grid(128, 128):
                        with Ts.sblock("B_i2"):
                            vi, vj, vk = Ts.axis.remap("SSS", [i2, j, k])
                            Ts.reads(A[64:128, 0:128, 0:128])
                            Ts.writes(B[64:128, 0:128, 0:128])
                            B[vi, vj, vk] = A[vi, vj, vk] * T.float32(2)


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
def opaque_access_loop_partition(
    A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")
) -> None:
    for i in range(16):
        with Ts.sblock("A_j_common"):
            Ts.reads()
            Ts.writes()
            with Ts.sblock("A_j0_partition"):
                Ts.reads()
                Ts.writes()
                for j0 in range(12):
                    with Ts.sblock("A_j0"):
                        vi, vj = Ts.axis.remap("SS", [i, j0])
                        Ts.reads()
                        Ts.writes(A[0:16, 0:12])
                        A[vi, vj] = T.float32(1)
            with Ts.sblock("A_j1_partition"):
                Ts.reads()
                Ts.writes()
                for j1 in range(12, 16):
                    with Ts.sblock("A_j1"):
                        vi, vj = Ts.axis.remap("SS", [i, j1])
                        Ts.reads()
                        Ts.writes(A[0:16, 12:16])
                        A[vi, vj] = T.float32(1)
    for i in range(16):
        with Ts.sblock("B_j_common"):
            Ts.reads()
            Ts.writes()
            with Ts.sblock("B_j0_partition"):
                Ts.reads()
                Ts.writes()
                for j0 in range(12):
                    with Ts.sblock("B_j0"):
                        vi, vj = Ts.axis.remap("SS", [i, j0])
                        Ts.reads()
                        Ts.writes(B[0:16, 0:16])
                        T.gpu_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj)
            with Ts.sblock("B_j1_partition"):
                Ts.reads()
                Ts.writes()
                for j1 in range(12, 16):
                    with Ts.sblock("B_j1"):
                        vi, vj = Ts.axis.remap("SS", [i, j1])
                        Ts.reads()
                        Ts.writes(B[0:16, 0:16])
                        T.gpu_fill_fragment(B.data, 16, 16, 16, 0, vi * 16 + vj)


# pylint: enable=no-member,invalid-name,unused-variable


def test_loop_partition():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    sch.loop_partition(i, factors=[2, 1, 64])

    block_b_partition = sch.get_sblock("B_i0")
    i, j, k = sch.get_loops(block_b_partition)
    loops = sch.loop_partition(j, factors=[4, 32])

    assert_structural_equal_ignore_global_symbol(elementwise_loop_partition_case0, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_partition_with_inferred_factor():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    sch.loop_partition(i, factors=[None, 1, 64])

    block_b_partition = sch.get_sblock("B_i1")
    i, j, k = sch.get_loops(block_b_partition)
    sch.loop_partition(k, factors=[1, 64, None])

    assert_structural_equal_ignore_global_symbol(elementwise_loop_partition_case1, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_partition_with_opaque_block():
    sch = tvm.s_tir.Schedule(elementwise_with_opaque_block, debug_mask="all")
    block_opaque = sch.get_sblock("opaque")
    i, _, _ = sch.get_loops(block_opaque)
    sch.loop_partition(i, factors=[None, 16])
    assert_structural_equal_ignore_global_symbol(
        elementwise_partition_with_opaque_block, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_with_opaque_block)


def test_partition_with_opaque_access():
    sch = tvm.s_tir.Schedule(opaque_access, debug_mask="all")
    block_a = sch.get_sblock("A")
    _, j = sch.get_loops(block_a)
    sch.loop_partition(j, factors=[None, 4])
    block_b = sch.get_sblock("B")
    _, j = sch.get_loops(block_b)
    sch.loop_partition(j, factors=[None, 4])
    assert_structural_equal_ignore_global_symbol(opaque_access_loop_partition, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_partition_int64_extent_with_mixed_factors():
    def _create_function():
        m = te.const(384, "int64")
        A = te.placeholder((m,), name="A", dtype="float32")
        B = te.compute((m,), lambda i: A[i] + 1, name="B")
        return te.create_function([A, B])

    mod = _create_function()
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    (i,) = sch.get_loops(sch.get_sblock("B"))
    sch.loop_partition(
        i,
        factors=[
            te.const(1, "int64"),
            te.const(51, "int32"),
        ],
    )


def test_partition_fail_symbolic():
    sch = tvm.s_tir.Schedule(elementwise_symbolic, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, _, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.loop_partition(k, factors=[10, None])


def test_partition_fail_out_of_bound():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.loop_partition(i, factors=[1000, 2, 3])


def test_partition_with_non_positive_factors():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.loop_partition(i, factors=[-2, -64])
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.loop_partition(j, factors=[0, None])
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.loop_partition(k, factors=[None, -16])


def test_partition_fail_with_annotation():
    sch = tvm.s_tir.Schedule(elementwise_with_anno, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.loop_partition(k, factors=[None, 10])


def test_partition_fail_with_thread_binding():
    sch = tvm.s_tir.Schedule(elementwise_with_thread_binding, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.loop_partition(k, factors=[None, 10])


if __name__ == "__main__":
    tvm_tirx.testing.main()
