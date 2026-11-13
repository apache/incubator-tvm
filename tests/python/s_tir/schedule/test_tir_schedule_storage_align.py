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

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name

@Ts.function
def element_wise(
    A: T.Tensor([128, 128], "float32",  align=64),
    C: T.Tensor([128, 128], "float32",  align=64),
) -> None:

    # body
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        B = Ts.sblock_alloc_tensor([128, 128],  align=64)
        for i0 in T.serial(0, 128):
            for ax1 in T.serial(0, 128):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i0, ax1])
                    Ts.reads([A[vi, vj]])
                    Ts.writes([B[vi, vj]])
                    B[vi, vj] = A[vi, vj] * T.float32(2)
            for i1 in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi_1, vj_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads([B[vi_1, vj_1]])
                    Ts.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = B[vi_1, vj_1] + T.float32(1)

@Ts.function
def element_wise_storage_align(
    A: T.Tensor([128, 128], "float32",  align=64),
    C: T.Tensor([128, 128], "float32",  align=64),
) -> None:

    # body
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        B = Ts.sblock_alloc_tensor([128, 128],  align=64)
        for i0 in T.serial(0, 128):
            for ax1 in T.serial(0, 128):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i0, ax1])
                    Ts.reads([A[vi, vj]])
                    Ts.writes([B[vi, vj]])
                    Ts.sblock_attr({"tensor_dim_align": [[0, 0, 128, 127]]})
                    B[vi, vj] = A[vi, vj] * T.float32(2)
            for i1 in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi_1, vj_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads([B[vi_1, vj_1]])
                    Ts.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = B[vi_1, vj_1] + T.float32(1)

@Ts.function
def element_wise_invalid_annotation(
    A: T.Tensor([128, 128], "float32",  align=64),
    C: T.Tensor([128, 128], "float32",  align=64),
) -> None:

    # body
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        B = Ts.sblock_alloc_tensor([128, 128],  align=64)
        for i0 in T.serial(0, 128):
            for ax1 in T.serial(0, 128):
                with Ts.sblock("B"):
                    Ts.sblock_attr({"tensor_dim_align": [0]})
                    vi, vj = Ts.axis.remap("SS", [i0, ax1])
                    Ts.reads([A[vi, vj]])
                    Ts.writes([B[vi, vj]])
                    B[vi, vj] = A[vi, vj] * T.float32(2)
            for i1 in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi_1, vj_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads([B[vi_1, vj_1]])
                    Ts.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = B[vi_1, vj_1] + T.float32(1)


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


def test_storage_align(use_block_name):
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    B = 'B' if use_block_name else s.get_sblock("B")
    s.storage_align(B, 0, axis=0, factor=128, offset=127)
    assert_structural_equal_ignore_global_symbol(element_wise_storage_align, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)

def test_storage_align_update():
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    B = s.get_sblock("B")
    s.storage_align(B, 0, axis=0, factor=128, offset=0)
    s.storage_align(B, 0, axis=0, factor=128, offset=127)
    assert_structural_equal_ignore_global_symbol(element_wise_storage_align, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)

def test_storage_align_invalid_factor1():
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    B = s.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.storage_align(B, 0, axis=0, factor=0, offset=127)

def test_storage_align_invalid_factor2():
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    B = s.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.storage_align(B, 0, axis=0, factor=-1, offset=127)

def test_storage_align_invalid_tensor():
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    C = s.get_sblock("C")
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.storage_align(C, 0, axis=0, factor=128, offset=127)

def test_storage_align_invalid_tensor_index():
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    B = s.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.storage_align(B, 2, axis=0, factor=128, offset=127)

def test_storage_align_invalid_axis():
    func = element_wise
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    B = s.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.storage_align(B, 0, axis=2, factor=128, offset=127)

def test_storage_align_invalid_annotation():
    func = element_wise_invalid_annotation
    s = tvm.s_tir.Schedule(func, debug_mask='all')
    B = s.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        s.storage_align(B, 0, axis=2, factor=128, offset=127)

if __name__ == "__main__":
    tvm_tirx.testing.main()
