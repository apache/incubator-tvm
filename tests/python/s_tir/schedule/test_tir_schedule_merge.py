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
    A: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
    D: T.Tensor((64, 64), "float32"),
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
        with Ts.sblock("C"):
            vi = Ts.axis.spatial(128, i_0 * 16 + i_1)
            vj = Ts.axis.spatial(128, j_0 * 16 + j_1)
            Ts.reads(B[vi, vj])
            Ts.writes(C[vi, vj])
            C[vi, vj] = B[vi, vj] + T.float32(1)
    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 8, 8):
        with Ts.sblock("D"):
            vi = Ts.axis.spatial(64, i_0 * 8 + i_1)
            vj = Ts.axis.spatial(64, j_0 * 8 + j_1)
            Ts.reads(B[vi, vj])
            Ts.writes(D[vi, vj])
            D[vi, vj] = B[vi, vj] + T.float32(2)


@Ts.function
def elementwise_merged(
    A: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
    D: T.Tensor((64, 64), "float32"),
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i_0_m in range(8):
        for j_0, i_1, j_1 in T.grid(8, 16, 16):
            with Ts.sblock("C"):
                vi = Ts.axis.spatial(128, i_0_m * 16 + i_1)
                vj = Ts.axis.spatial(128, j_0 * 16 + j_1)
                Ts.reads(B[vi, vj])
                Ts.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + T.float32(1)
        for j_0, i_1, j_1 in T.grid(8, 8, 8):
            with Ts.sblock("D"):
                vi = Ts.axis.spatial(64, i_0_m * 8 + i_1)
                vj = Ts.axis.spatial(64, j_0 * 8 + j_1)
                Ts.reads(B[vi, vj])
                Ts.writes(D[vi, vj])
                D[vi, vj] = B[vi, vj] + T.float32(2)


@Ts.function
def elementwise_merged2(
    A: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
    D: T.Tensor((64, 64), "float32"),
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i_0_m, j_0_m in T.grid(8, 8):
        for i_1, j_1 in T.grid(16, 16):
            with Ts.sblock("C"):
                vi = Ts.axis.spatial(128, i_0_m * 16 + i_1)
                vj = Ts.axis.spatial(128, j_0_m * 16 + j_1)
                Ts.reads(B[vi, vj])
                Ts.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + T.float32(1)
        for i_1, j_1 in T.grid(8, 8):
            with Ts.sblock("D"):
                vi = Ts.axis.spatial(64, i_0_m * 8 + i_1)
                vj = Ts.axis.spatial(64, j_0_m * 8 + j_1)
                Ts.reads(B[vi, vj])
                Ts.writes(D[vi, vj])
                D[vi, vj] = B[vi, vj] + T.float32(2)


def test_merge():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_c = sch.get_sblock("C")
    block_d = sch.get_sblock("D")
    i = sch.get_loops(block_c)[0]
    j = sch.get_loops(block_d)[0]
    sch.merge(i, j)
    assert_structural_equal_ignore_global_symbol(elementwise_merged, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_merge2():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_c = sch.get_sblock("C")
    block_d = sch.get_sblock("D")
    i = sch.get_loops(block_c)[1]
    j = sch.get_loops(block_d)[1]
    sch.merge(i, j)
    assert_structural_equal_ignore_global_symbol(elementwise_merged2, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_merge_fail_not_only_child():
    @Ts.function
    def elementwise_with_seq(
        A: T.Tensor((128, 128, 128), "float32"), C: T.Tensor((128, 128, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor((128, 128, 128))
        D = Ts.sblock_alloc_tensor((128, 128, 128))
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with Ts.sblock("D"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    D[vi, vj, vk] = A[vi, vj, vk] * 2.0
            for k in T.serial(0, 128):
                with Ts.sblock("B"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = B[vi, vj, vk] * 2.0

    sch = tvm.s_tir.Schedule(elementwise_with_seq, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, _, b = sch.get_loops(block_b)
    block_c = sch.get_sblock("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.merge(b, c)


def test_merge_fail_not_start_with_zero():
    @Ts.function
    def elementwise_loops_not_start_with_zero(
        A: T.Tensor((128, 128, 128), "float32"), C: T.Tensor((128, 128, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor((128, 128, 128))
        for i, j in T.grid(128, 128):
            for k in T.serial(1, 128):
                with Ts.sblock("B"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = A[vi, vj, vk] * 2.0

    sch = tvm.s_tir.Schedule(elementwise_loops_not_start_with_zero, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, _, b = sch.get_loops(block_b)
    block_c = sch.get_sblock("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.merge(b, c)


def test_merge_fail_not_same_extent():
    @Ts.function
    def elementwise_loops_not_same_extent(
        A: T.Tensor((128, 128, 128), "float32"), C: T.Tensor((128, 128, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor((64, 128, 128))
        for i, j in T.grid(64, 128):
            for k in T.serial(0, 128):
                with Ts.sblock("B"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = A[vi, vj, vk] * 2.0

    sch = tvm.s_tir.Schedule(elementwise_loops_not_same_extent, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, _, b = sch.get_loops(block_b)
    block_c = sch.get_sblock("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.merge(b, c)


def test_merge_fail_not_same_level():
    @Ts.function
    def elementwise_not_same_level(
        A: T.Tensor((128, 128, 128), "float32"), C: T.Tensor((128, 128, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor((128, 128, 128))
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with Ts.sblock("B"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = A[vi, vj, vk] * 2.0

    sch = tvm.s_tir.Schedule(elementwise_not_same_level, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, b, _ = sch.get_loops(block_b)
    block_c = sch.get_sblock("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.merge(b, c)


def test_merge_fail_with_different_scope():
    @Ts.function
    def elementwise_with_different_scope(
        A: T.Tensor((128, 128, 128), "float32"), C: T.Tensor((128, 128, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor((128, 128, 128))
        with Ts.sblock("A"):
            for i, j in T.grid(128, 128):
                for k in T.serial(0, 128):
                    with Ts.sblock("B"):
                        vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                        B[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for i, j in T.grid(128, 128):
            for k in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    C[vi, vj, vk] = A[vi, vj, vk] * 2.0

    sch = tvm.s_tir.Schedule(elementwise_with_different_scope, debug_mask="all")
    block_b = sch.get_sblock("B")
    _, _, b = sch.get_loops(block_b)
    block_c = sch.get_sblock("C")
    _, _, c = sch.get_loops(block_c)
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.merge(b, c)


if __name__ == "__main__":
    tvm_tirx.testing.main()
