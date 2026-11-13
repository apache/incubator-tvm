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
# ruff: noqa: E501, F401

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

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg

@Ts.function
def element_wise(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((128, 128), dtype="float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def element_wise_set_dtype(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
    B = Ts.sblock_alloc_tensor((128, 128), "float16")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(B[vi, vj])
            B[vi, vj] = T.cast(A[vi, vj] * 2.0, ty="float16")
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B[vi, vj])
            Ts.writes(C[vi, vj])
            C[vi, vj] = T.cast(B[vi, vj], ty="float32") + 1.0

element_wise_subregion_match_b_subregion0_offset = T.dynamic("element_wise_subregion_match_b_subregion0_offset", "int32")
element_wise_subregion_match_b_subregion1_offset = T.dynamic("element_wise_subregion_match_b_subregion1_offset", "int32")

@Ts.function
def element_wise_subregion_match(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((128, 128), dtype="float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_subregion0 = Ts.match_tensor(B[vi, vj], [], layout=T.TileLayout(T.S[[]] + (element_wise_subregion_match_b_subregion0_offset)))
            B_subregion0[()] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_subregion1 = Ts.match_tensor(B[vi, vj], [], layout=T.TileLayout(T.S[[]] + (element_wise_subregion_match_b_subregion1_offset)))
            C[vi, vj] = B_subregion1[()] + 1.0

element_wise_subregion_match_set_dtype_b_subregion0_offset = T.dynamic("element_wise_subregion_match_set_dtype_b_subregion0_offset", "int32")
element_wise_subregion_match_set_dtype_b_subregion1_offset = T.dynamic("element_wise_subregion_match_set_dtype_b_subregion1_offset", "int32")

@Ts.function
def element_wise_subregion_match_set_dtype(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((128, 128), "float16")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(B[vi, vj])
            B_subregion0 = Ts.match_tensor(B[vi, vj], (), "float16", layout=T.TileLayout(T.S[()] + (element_wise_subregion_match_set_dtype_b_subregion0_offset)))
            B_subregion0[()] = T.cast(A[vi, vj] * 2.0, ty="float16")
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B[vi, vj])
            Ts.writes(C[vi, vj])
            B_subregion1 = Ts.match_tensor(B[vi, vj], (), "float16", layout=T.TileLayout(T.S[()] + (element_wise_subregion_match_set_dtype_b_subregion1_offset)))
            C[vi, vj] = T.cast(B_subregion1[()], ty="float32") + 1.0


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


def test_set_dtype(use_block_name):
    func = element_wise
    sch = tvm.s_tir.Schedule(func, debug_mask="all")
    sch.unsafe_set_dtype("B" if use_block_name else sch.get_sblock("B"), 0, "float16")
    assert_structural_equal_ignore_global_symbol(element_wise_set_dtype, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func)

def test_set_dtype_fail_on_output_tensor(use_block_name):
    func = element_wise
    sch = tvm.s_tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.unsafe_set_dtype('C' if use_block_name else sch.get_sblock("C"), 0, "float16")

def test_set_dtype_fail_on_index_out_of_bound():
    func = element_wise
    sch = tvm.s_tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.unsafe_set_dtype(sch.get_sblock("B"), 1, "float64")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.unsafe_set_dtype(sch.get_sblock("B"), -1, "float64")

def test_set_dtype_subregion():
    func = element_wise_subregion_match
    sch = tvm.s_tir.Schedule(func, debug_mask='all')
    sch.unsafe_set_dtype(sch.get_sblock("B"), 0, "float16")
    assert_structural_equal_ignore_global_symbol(element_wise_subregion_match_set_dtype, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func)

if __name__ == "__main__":
    tvm_tirx.testing.main()
