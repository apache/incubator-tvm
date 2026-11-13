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


@Ts.function
def indirect_mem_access(
    A: T.Tensor([128], dtype="float32"),
    IA: T.Tensor([10], dtype="int32"),
    B: T.Tensor([128], dtype="float32"),
    IB: T.Tensor([10], dtype="int32"),
) -> None:
    for i in range(10):
        with Ts.sblock("B"):
            vi = Ts.axis.spatial(10, i)
            Ts.reads(A[IA[vi]], IA[vi])
            Ts.writes(B[IB[vi]], IB[vi])
            B[IB[vi]] = A[IA[vi]]


@Ts.function
def indirect_mem_access_hide_ia(
    A: T.Tensor([128], dtype="float32"),
    IA: T.Tensor([10], dtype="int32"),
    B: T.Tensor([128], dtype="float32"),
    IB: T.Tensor([10], dtype="int32"),
) -> None:
    for i in range(10):
        with Ts.sblock("B"):
            vi = Ts.axis.spatial(10, i)
            Ts.reads(A[IA[vi]])
            Ts.writes(B[IB[vi]], IB[vi])
            B[IB[vi]] = A[IA[vi]]


@Ts.function
def indirect_mem_access_hide_ib(
    A: T.Tensor([128], dtype="float32"),
    IA: T.Tensor([10], dtype="int32"),
    B: T.Tensor([128], dtype="float32"),
    IB: T.Tensor([10], dtype="int32"),
) -> None:
    for i in range(10):
        with Ts.sblock("B"):
            vi = Ts.axis.spatial(10, i)
            Ts.reads(A[IA[vi]], IA[vi])
            Ts.writes(B[IB[vi]])
            B[IB[vi]] = A[IA[vi]]


def test_hide_tensor_access_read():
    sch = tvm.s_tir.Schedule(indirect_mem_access, debug_mask="all")
    block_b = sch.get_sblock("B")
    sch.unsafe_hide_tensor_access(block_b, "read", [1])
    assert_structural_equal_ignore_global_symbol(indirect_mem_access_hide_ia, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=indirect_mem_access)


def test_hide_tensor_access_write():
    sch = tvm.s_tir.Schedule(indirect_mem_access, debug_mask="all")
    block_b = sch.get_sblock("B")
    sch.unsafe_hide_tensor_access(block_b, "write", [1])
    assert_structural_equal_ignore_global_symbol(indirect_mem_access_hide_ib, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=indirect_mem_access)


def test_hide_tensor_access_fail_tensor_type():
    sch = tvm.s_tir.Schedule(indirect_mem_access, debug_mask="all")
    block_b = sch.get_sblock("B")
    with pytest.raises(RuntimeError):
        sch.unsafe_hide_tensor_access(block_b, "opaque", [0])


def test_hide_tensor_access_fail_tensor_index():
    sch = tvm.s_tir.Schedule(indirect_mem_access, debug_mask="all")
    block_b = sch.get_sblock("B")
    with pytest.raises(RuntimeError):
        sch.unsafe_hide_tensor_access(block_b, "read", [2])


if __name__ == "__main__":
    tvm_tirx.testing.main()
