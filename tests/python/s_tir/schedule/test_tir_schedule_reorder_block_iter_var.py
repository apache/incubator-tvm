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
# ruff: noqa: F401

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import verify_trace_roundtrip
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


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
def matmul_after_reorder_block_iter_var(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
):
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("C"):
            vk, vj, vi = Ts.axis.remap("RSS", [k, j, i])
            Ts.reads(A[vi, vk], B[vj, vk])
            Ts.writes(C[vi, vj])
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_reorder_block_iter_var():
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all")
    C = sch.get_sblock("C")
    sch.reorder_block_iter_var(C, [2, 1, 0])
    tvm_tirx.ir.assert_structural_equal(
        matmul_after_reorder_block_iter_var.with_attr("global_symbol", "matmul"), sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=matmul)


def test_reorder_block_iter_var_fail_not_full():
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all")
    C = sch.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder_block_iter_var(C, [2, 1])


def test_reorder_block_iter_var_fail_not_within_bound():
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all")
    C = sch.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder_block_iter_var(C, [-1, 3, 2])


def test_reorder_block_iter_var_fail_not_unique():
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all")
    C = sch.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reorder_block_iter_var(C, [0, 0, 2])


if __name__ == "__main__":
    tvm_tirx.testing.main()
