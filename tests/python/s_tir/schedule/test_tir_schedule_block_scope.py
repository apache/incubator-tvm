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
from tvm_ffi import structural_walk

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule import DepKind
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable


@Ts.function
def elementwise(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


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
        for k in range(0, 128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def war_dependency(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


# pylint: enable=no-member,invalid-name,unused-variable

# pylint: disable=invalid-name


def _get_sblock(s: s_tir.ScheduleState, name_hint: str) -> s_tir.StmtSRef:
    result = None

    def f_visit(node):
        nonlocal result
        if isinstance(node, tvm.s_tir.SBlock) and node.name_hint == name_hint:
            result = node

    func = s.mod["main"]
    structural_walk(func.body, f_visit, order="post")
    assert result is not None and isinstance(result, tvm.s_tir.SBlock)
    return s.get_sref(result)


def test_elementwise_dependency():
    s = tvm.s_tir.ScheduleState(elementwise, debug_mask="all")
    root = _get_sblock(s, "root")
    block_b = _get_sblock(s, "B")
    block_c = _get_sblock(s, "C")
    # Check get_deps_by_src
    (dep,) = s.get_sblock_scope(root).get_deps_by_src(block_b)
    assert dep.src.same_as(block_b)
    assert dep.dst.same_as(block_c)
    assert dep.kind == DepKind.RAW
    # Check get_deps_by_dst
    (dep,) = s.get_sblock_scope(root).get_deps_by_dst(block_c)
    assert dep.src.same_as(block_b)
    assert dep.dst.same_as(block_c)
    assert dep.kind == DepKind.RAW


def test_matmul_dependency():
    s = tvm.s_tir.ScheduleState(matmul, debug_mask="all")
    root = _get_sblock(s, "root")
    init = _get_sblock(s, "init")
    update = _get_sblock(s, "update")
    # Check get_deps_by_src
    p0, p1 = s.get_sblock_scope(root).get_deps_by_src(init)
    assert p0.src.same_as(init)
    assert p0.dst.same_as(update)
    assert p1.src.same_as(init)
    assert p1.dst.same_as(update)
    assert (p0.kind == DepKind.RAW and p1.kind == DepKind.WAW) or (
        p0.kind == DepKind.WAW and p1.kind == DepKind.RAW
    )
    # Check get_deps_by_dst
    p0, p1 = s.get_sblock_scope(root).get_deps_by_dst(update)
    assert p0.src.same_as(init)
    assert p0.dst.same_as(update)
    assert p1.src.same_as(init)
    assert p1.dst.same_as(update)
    assert (p0.kind == DepKind.RAW and p1.kind == DepKind.WAW) or (
        p0.kind == DepKind.WAW and p1.kind == DepKind.RAW
    )


def test_war_dependency():
    s = tvm.s_tir.ScheduleState(war_dependency, debug_mask="all")
    root = _get_sblock(s, "root")
    block_c = _get_sblock(s, "C")
    block_b = _get_sblock(s, "B")
    # Check get_deps_by_src
    (dep,) = s.get_sblock_scope(root).get_deps_by_src(block_c)
    assert dep.src.same_as(block_c)
    assert dep.dst.same_as(block_b)
    assert dep.kind == DepKind.WAR
    # Check get_deps_by_dst
    (dep,) = s.get_sblock_scope(root).get_deps_by_dst(block_b)
    assert dep.src.same_as(block_c)
    assert dep.dst.same_as(block_b)
    assert dep.kind == DepKind.WAR


if __name__ == "__main__":
    tvm_tirx.testing.main()
