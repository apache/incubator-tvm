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
import gc
import sys

import pytest
from tvm_ffi import structural_walk

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import SBlockDependenceInfo
from tvm.s_tir import script as Ts
from tvm.s_tir.sblock_scope import DepKind
from tvm_tirx import tirx
from tvm_tirx.ir import IRModule
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import Function

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
    for i, j in T.grid(128, 128):
        with Ts.sblock("D"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


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


# pylint: enable=no-member,invalid-name,unused-variable


def get_sblocks(func: Function):
    blocks = {}

    def update_blocks(node):
        if isinstance(node, tvm.s_tir.SBlock):
            blocks[node.name_hint] = node

    # post_order_visit(func.body, lambda node: blocks[node.name_hint] = node if isinstance(node, tvm.s_tir.SBlock) else None)
    structural_walk(func.body, update_blocks, order="post")
    return blocks


def _verify_dependence(dependence_info, src_block, dst_block, kind):
    src_sref = dependence_info.get_sref(src_block)
    dst_sref = dependence_info.get_sref(dst_block)
    scope = dependence_info.get_sblock_scope(src_sref.parent)

    def _find_dependence(deps):
        for dep in deps:
            if dep.src == src_sref and dep.dst == dst_sref and dep.kind == kind:
                return dep
        return None

    def _get_dependency_kind_name(dep_kind):
        if isinstance(dep_kind, int):
            dep_kind = DepKind(dep_kind)
        return dep_kind.name

    # Check dependences by src
    deps_by_src = scope.get_deps_by_src(src_sref)
    dependence = _find_dependence(deps_by_src)
    assert dependence, (
        f"Expected a dependency with src block {src_block.name_hint} and dst block {dst_block.name_hint} of kind {kind.name}"
    )

    # Check dependences by dst
    deps_by_dst = scope.get_deps_by_dst(dst_sref)
    dependence = _find_dependence(deps_by_dst)
    assert dependence, (
        f"Expected a dependency with src block {src_block.name_hint} and dst block {dst_block.name_hint}"
    )


def test_RAW_dependences():
    func = elementwise
    dependence_info = SBlockDependenceInfo(func)
    blocks = get_sblocks(func)
    _verify_dependence(dependence_info, blocks["B"], blocks["C"], DepKind.RAW)


def test_WAR_dependences():
    func = war_dependency
    dependence_info = SBlockDependenceInfo(func)
    blocks = get_sblocks(func)
    _verify_dependence(dependence_info, blocks["C"], blocks["B"], DepKind.WAR)


def test_RAW_and_WAW_dependences():
    func = matmul
    dependence_info = SBlockDependenceInfo(func)
    blocks = get_sblocks(func)
    _verify_dependence(dependence_info, blocks["init"], blocks["update"], DepKind.RAW)
    _verify_dependence(dependence_info, blocks["init"], blocks["update"], DepKind.WAW)


if __name__ == "__main__":
    tvm_tirx.testing.main()
