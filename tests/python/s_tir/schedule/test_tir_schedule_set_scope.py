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
def element_wise_set_scope(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B_shared = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="shared")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_shared[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B_shared[vi, vj] + T.float32(1)

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

element_wise_subregion_match_set_scope_b_subregion0_shared_offset = T.dynamic("element_wise_subregion_match_set_scope_b_subregion0_shared_offset", "int32")
element_wise_subregion_match_set_scope_b_subregion1_shared_offset = T.dynamic("element_wise_subregion_match_set_scope_b_subregion1_shared_offset", "int32")

@Ts.function
def element_wise_subregion_match_set_scope(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B_shared = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="shared")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_subregion0_shared = Ts.match_tensor(B_shared[vi, vj], [], dtype="float32", scope="shared", layout=T.TileLayout(T.S[[]] + (element_wise_subregion_match_set_scope_b_subregion0_shared_offset)))
            B_subregion0_shared[()] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_subregion1_shared = Ts.match_tensor(B_shared[vi, vj], [], dtype="float32", scope="shared", layout=T.TileLayout(T.S[[]] + (element_wise_subregion_match_set_scope_b_subregion1_shared_offset)))
            C[vi, vj] = B_subregion1_shared[()] + T.float32(1)

# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


@pytest.fixture(
    params=[pytest.param(False, id="tensor_index"), pytest.param(True, id="tensor_name")],
    scope="session",
)
def use_tensor_name(request):
    return request.param


def test_set_scope(use_block_name, use_tensor_name):
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    tensor_name = s.get(s.get_sblock("B")).writes[0].source.name
    s.set_scope(
        "B" if use_block_name else s.get_sblock("B"),
        tensor_name if use_tensor_name else 0,
        "shared",
    )
    assert_structural_equal_ignore_global_symbol(element_wise_set_scope, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)

def test_set_scope_fail_on_output_tensor(use_block_name, use_tensor_name):
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.set_scope('C' if use_block_name else s.get_sblock("C"), 'C' if use_tensor_name else 0, "shared")

def test_set_scope_fail_on_index_out_of_bound():
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.set_scope(s.get_sblock("B"), 1, "shared")
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.set_scope(s.get_sblock("B"), -1, "shared")

def test_set_scope_fail_on_invalid_scope():
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.set_scope(s.get_sblock("B"), 0, "test_scope")

def test_set_scope_subregion():
    func = element_wise_subregion_match
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    s.set_scope(s.get_sblock("B"), 0, "shared")
    assert_structural_equal_ignore_global_symbol(element_wise_subregion_match_set_scope, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)

@pytest.mark.parametrize("scope_name", ["shared", "local"])
@pytest.mark.parametrize("method", ["set_scope", "cache_read", "cache_write"])
def test_memory_scope_enum_and_trace(scope_name, method):
    enum_schedule = tvm.s_tir.Schedule(element_wise, debug_mask="all")
    string_schedule = tvm.s_tir.Schedule(element_wise, debug_mask="all")
    getattr(enum_schedule, method)("B", 0, tvm_tirx.runtime.MemoryScope(scope_name))
    getattr(string_schedule, method)("B", 0, scope_name)
    tvm_tirx.ir.assert_structural_equal(enum_schedule.mod, string_schedule.mod)
    assert enum_schedule.trace.as_python() == string_schedule.trace.as_python()
    verify_trace_roundtrip(sch=enum_schedule, mod=element_wise)


def test_thread_sync_accepts_memory_scope():
    mod = tvm_tirx.IRModule({"main": element_wise})
    enum_result = tvm.s_tir.transform.ThreadSync(tvm_tirx.runtime.MemoryScope("shared"))(mod)
    string_result = tvm.s_tir.transform.ThreadSync("shared")(mod)
    tvm_tirx.ir.assert_structural_equal(enum_result, string_result)


if __name__ == "__main__":
    tvm_tirx.testing.main()
