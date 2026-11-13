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
from __future__ import annotations

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable


@Ts.function
def matmul(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("init"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = T.float32(0)
        for k in range(128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def two_kernels(
    A: T.Tensor((1, seq_len * 8), "int32"),  # noqa: F821
    B: T.Tensor((1, seq_len * 8), "int32", align=8),  # noqa: F821
    seq_len: T.int32,
):
    T.func_attr({"tirx.noalias": True})

    with Ts.sblock("exclusive_scan"):
        Ts.reads()
        Ts.writes()
        s8: T.int32 = seq_len * 8
        if s8 == 0:
            blockIdx_x = T.launch_thread("blockIdx.x", 1)
        else:
            with T.launch_thread("threadIdx.x", 1024) as threadIdx_x:
                blockIdx_x = T.launch_thread("blockIdx.x", T.ceildiv(s8, 1024))
                i: T.int32 = blockIdx_x * 1024 + threadIdx_x
                if i < s8:
                    B[i // s8, i % s8] = A[i // s8, i % s8]


# pylint: enable=no-member,invalid-name,unused-variable


def test_tir_schedule_error_detail():
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all", error_render_level="detail")
    with pytest.raises(tvm.s_tir.ScheduleError) as excinfo:
        sch.get_sblock("wrong_name")
    (msg,) = excinfo.value.args
    assert "Cannot find a block with the name: wrong_name" in msg


def test_tir_schedule_error_fast():
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all", error_render_level="fast")
    with pytest.raises(tvm.s_tir.ScheduleError) as excinfo:
        sch.get_sblock("wrong_name")
    (msg,) = excinfo.value.args
    assert "Cannot find a block with the specified name" in msg


def test_tir_schedule_error_none():
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all", error_render_level="none")
    with pytest.raises(tvm.s_tir.ScheduleError) as excinfo:
        sch.get_sblock("wrong_name")
    (msg,) = excinfo.value.args
    assert "(not rendered)" in msg


def test_tir_schedule_attribute_error():
    sch = tvm.s_tir.Schedule(matmul)
    with pytest.raises(AttributeError):
        sch.non_existent_field()


def test_tir_schedule_two_kernels():
    s_tir.Schedule(two_kernels)


if __name__ == "__main__":
    tvm_tirx.testing.main()
