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
import sys

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
from tvm_tirx.ir import IRModule, Range
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
            C[vi, vj] = 0.0
        for k in range(0, 128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def matmul_relu(
    A: T.Tensor((1024, 1024), "float32"),
    B: T.Tensor((1024, 1024), "float32"),
    D: T.Tensor((1024, 1024), "float32"),
) -> None:
    C = Ts.sblock_alloc_tensor((1024, 1024))

    for i, j, k in T.grid(1024, 1024, 1024):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(1024, 1024):
        with Ts.sblock("relu"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = T.max(C[vi, vj], 0.0)


@Ts.function
def matmul_relu_ann1(
    A: T.Tensor((1024, 1024), "float32"),
    B: T.Tensor((1024, 1024), "float32"),
    D: T.Tensor((1024, 1024), "float32"),
) -> None:
    C = Ts.sblock_alloc_tensor((1024, 1024))

    for i in T.serial(0, 1024, annotations={"test1": "aaa", "test4": {"arr": [0, 0], "key": 3}}):
        for j in T.serial(0, 1024, annotations={"test2": 612, "test3": ["aa", 1]}):
            for k in T.serial(0, 1024):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(1024, 1024):
        with Ts.sblock("relu"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = T.max(C[vi, vj], 0.0)


@Ts.function
def matmul_relu_ann2(
    A: T.Tensor((1024, 1024), "float32"),
    B: T.Tensor((1024, 1024), "float32"),
    D: T.Tensor((1024, 1024), "float32"),
) -> None:
    C = Ts.sblock_alloc_tensor((1024, 1024))

    for i, j, k in T.grid(1024, 1024, 1024):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            Ts.sblock_attr({"test1": "aaa", "test4": {"arr": [0, 0], "key": 3}})
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(1024, 1024):
        with Ts.sblock("relu"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.sblock_attr({"test2": 0.22, "test3": ["aa", 1]})
            D[vi, vj] = T.max(C[vi, vj], 0.0)


@tvm_tirx.script.ir_module
class ModuleWithMultipleFuncs:
    @Ts.function
    def vector_add(
        A: T.Tensor(128, "float32"),
        B: T.Tensor(128, "float32"),
    ) -> None:
        for i in range(128):
            with Ts.sblock("init"):
                vi = Ts.axis.remap("S", [i])
                B[vi] = A[vi]

    @Ts.function
    def vector_add_2(
        A: T.Tensor(128, "float32"),
        B: T.Tensor(128, "float32"),
    ) -> None:
        for i in range(128):
            with Ts.sblock("init"):
                vi = Ts.axis.remap("S", [i])
                B[vi] = A[vi]


@Ts.function
def tuple_reduction(data: T.Tensor((4, 32), "float32"), T_add: T.Tensor((4,), "float32")) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    # body
    with Ts.sblock("root"):
        Ts.reads()
        Ts.writes()
        data_red_temp_v0 = Ts.sblock_alloc_tensor([4], dtype="float32")
        data_red_temp_v1 = Ts.sblock_alloc_tensor([4], dtype="float32")
        for i0, i1 in T.grid(4, 32):
            with Ts.sblock("data_red_temp"):
                ax0, k1 = Ts.axis.remap("SR", [i0, i1])
                Ts.reads(data[ax0, k1])
                Ts.writes(data_red_temp_v0[ax0], data_red_temp_v1[ax0])
                with Ts.init():
                    data_red_temp_v0[ax0] = T.float32(0)
                    data_red_temp_v1[ax0] = T.float32(0)
                v_data_red_temp_v0: T.let[T.float32] = data_red_temp_v0[ax0] + data[ax0, k1]
                v_data_red_temp_v1: T.let[T.float32] = (
                    data_red_temp_v1[ax0] + data[ax0, k1] * data[ax0, k1]
                )
                data_red_temp_v0[ax0] = v_data_red_temp_v0
                data_red_temp_v1[ax0] = v_data_red_temp_v1
        for i0 in range(4):
            with Ts.sblock("T_add"):
                ax0 = Ts.axis.remap("S", [i0])
                Ts.reads(data_red_temp_v0[ax0], data_red_temp_v1[ax0])
                Ts.writes(T_add[ax0])
                T_add[ax0] = data_red_temp_v0[ax0] + data_red_temp_v1[ax0]


# pylint: enable=no-member,invalid-name,unused-variable


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


def test_tir_schedule_creation():
    # Tests:
    # - Schedule.__init__ for Function and IRModule
    # - Schedule.mod
    # - Schedule.state
    sch_1 = tvm.s_tir.Schedule(matmul, debug_mask="all")
    sch_2 = tvm.s_tir.Schedule(IRModule({"main": matmul}), debug_mask="all")
    assert sch_1.mod["main"].same_as(sch_2.mod["main"])
    assert sch_1.state.mod["main"].same_as(sch_2.state.mod["main"])


def test_tir_schedule_get_sblock():
    # Tests:
    # - Schedule.get_sblock
    # - Schedule.get_sref
    # - Schedule.get
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all")
    block_rv = sch.get_sblock(name="update")
    block_sref = sch.get_sref(block_rv)
    block = sch.get(block_rv)
    assert block.name_hint == "update"
    assert block_sref.stmt.same_as(block)
    assert sch.state.get_sref(block).same_as(block_sref)
    assert block.same_as(matmul.body[0].block.body[0].body[0].body[1].body[0].block)


def test_tir_schedule_work_on():
    sch = tvm.s_tir.Schedule(ModuleWithMultipleFuncs, debug_mask="all")
    with pytest.raises(ValueError, match="does not know which function to be working on"):
        sch.get_sblock(name="init")
    sch.work_on(func_name="vector_add")
    sch.get_sblock(name="init")
    assert sch.func_working_on == sch.mod.get_global_var("vector_add")


def test_tir_schedule_get_loops(use_block_name):
    # Tests:
    # - Schedule.get_loops
    # - Schedule.get
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all")
    block = "update" if use_block_name else sch.get_sblock(name="update")
    i, j, k = sch.get_loops(block)
    assert sch.get(i).var.name == "i"
    assert sch.get(j).var.name == "j"
    assert sch.get(k).var.name == "k"


def test_tir_schedule_copy_1(use_block_name):
    # Tests:
    # - Schedule.copy
    sch_1 = tvm.s_tir.Schedule(matmul, debug_mask="all")
    block_rv = sch_1.get_sblock(name="update")
    i, j, k = sch_1.get_loops(block="update" if use_block_name else block_rv)
    assert sch_1.get(i).var.name == "i"
    assert sch_1.get(j).var.name == "j"
    assert sch_1.get(k).var.name == "k"

    sch_2 = sch_1.copy()
    assert sch_2.get(block_rv).name_hint == "update"
    assert sch_2.get(i).var.name == "i"
    assert sch_2.get(j).var.name == "j"
    assert sch_2.get(k).var.name == "k"


def test_tir_schedule_copy_2():
    sch = tvm.s_tir.Schedule(mod=matmul, debug_mask="all")
    i, j, k = sch.get_loops(sch.get_sblock("update"))
    sch_copy = sch.copy()
    assert not sch.get_sref(i).same_as(sch_copy.get_sref(i))
    assert not sch.get_sref(j).same_as(sch_copy.get_sref(j))
    assert not sch.get_sref(k).same_as(sch_copy.get_sref(k))
    assert sch.get_sref(i).stmt.same_as(sch_copy.get_sref(i).stmt)
    assert sch.get_sref(j).stmt.same_as(sch_copy.get_sref(j).stmt)
    assert sch.get_sref(k).stmt.same_as(sch_copy.get_sref(k).stmt)
    i_0, i_1 = sch.split(i, factors=[None, 64])
    j_0, j_1 = sch_copy.split(j, factors=[None, 32])

    assert (
        Range.get_extent_from_begin_end(sch.get_sref(i_0).stmt.begin, sch.get_sref(i_0).stmt.end)
        == 2
    )
    assert (
        Range.get_extent_from_begin_end(sch.get_sref(i_1).stmt.begin, sch.get_sref(i_1).stmt.end)
        == 64
    )
    with pytest.raises(IndexError):
        sch_copy.get_sref(i_0)
    with pytest.raises(IndexError):
        sch_copy.get_sref(i_1)

    with pytest.raises(IndexError):
        sch.get_sref(j_0)
    with pytest.raises(IndexError):
        sch.get_sref(j_1)
    assert (
        Range.get_extent_from_begin_end(
            sch_copy.get_sref(j_0).stmt.begin, sch_copy.get_sref(j_0).stmt.end
        )
        == 4
    )
    assert (
        Range.get_extent_from_begin_end(
            sch_copy.get_sref(j_1).stmt.begin, sch_copy.get_sref(j_1).stmt.end
        )
        == 32
    )
    verify_trace_roundtrip(sch, mod=matmul)
    verify_trace_roundtrip(sch_copy, mod=matmul)


def test_tir_schedule_remove_rv():
    # Tests:
    # - Schedule.remove_rv
    sch = tvm.s_tir.Schedule(matmul, debug_mask="all")
    block_rv = sch.get_sblock(name="update")
    assert sch.get(block_rv).name_hint == "update"
    sch.remove_rv(block_rv)
    with pytest.raises(IndexError):
        sch.get(block_rv)


def test_get_child_blocks():
    s = tvm.s_tir.Schedule(matmul, debug_mask="all")
    init = s.get_sblock("init")
    update = s.get_sblock("update")
    # loop
    blocks = s.get_child_blocks(s.get_loops(init)[0])
    assert len(blocks) == 2
    assert s.get(init) == s.get(blocks[0])
    assert s.get(update) == s.get(blocks[1])
    # block
    root = s.get_sblock("root")
    blocks = s.get_child_blocks(root)
    assert len(blocks) == 2
    assert s.get(init) == s.get(blocks[0])
    assert s.get(update) == s.get(blocks[1])


def test_get_producers(use_block_name):
    sch = tvm.s_tir.Schedule(mod=matmul_relu, debug_mask="all")
    block = "relu" if use_block_name else sch.get_sblock("relu")
    (producer,) = sch.get_producers(block)
    tvm_tirx.ir.assert_structural_equal(
        sch.get_sref(producer).stmt,
        sch.get_sref(sch.get_sblock("matmul")).stmt,
    )
    verify_trace_roundtrip(sch, mod=matmul_relu)


def test_get_producers_multiple_tensor_depdencies(use_block_name):
    sch = tvm.s_tir.Schedule(mod=tuple_reduction, debug_mask="all")
    block = "T_add" if use_block_name else sch.get_sblock("T_add")
    (producer,) = sch.get_producers(block)
    tvm_tirx.ir.assert_structural_equal(
        sch.get_sref(producer).stmt,
        sch.get_sref(sch.get_sblock("data_red_temp")).stmt,
    )


def test_get_consumers(use_block_name):
    sch = tvm.s_tir.Schedule(mod=matmul_relu, debug_mask="all")
    block = "matmul" if use_block_name else sch.get_sblock("matmul")
    (consumer,) = sch.get_consumers(block)
    tvm_tirx.ir.assert_structural_equal(
        sch.get_sref(consumer).stmt,
        sch.get_sref(sch.get_sblock("relu")).stmt,
    )
    verify_trace_roundtrip(sch, mod=matmul_relu)


def test_get_consumers_multiple_tensor_depdencies(use_block_name):
    sch = tvm.s_tir.Schedule(mod=tuple_reduction, debug_mask="all")
    block = "data_red_temp" if use_block_name else sch.get_sblock("data_red_temp")
    (consumer,) = sch.get_consumers(block)
    tvm_tirx.ir.assert_structural_equal(
        sch.get_sref(consumer).stmt,
        sch.get_sref(sch.get_sblock("T_add")).stmt,
    )


def test_annotate_unannotate_loop():
    sch = tvm.s_tir.Schedule(mod=matmul_relu, debug_mask="all")
    matmul = sch.get_sblock("matmul")
    relu = sch.get_sblock("relu")
    sch.annotate(sch.get_loops(matmul)[0], "test1", "aaa")
    sch.annotate(sch.get_loops(matmul)[1], "test2", 612)
    sch.annotate(sch.get_loops(matmul)[1], "test3", ["aa", 1])
    sch.annotate(sch.get_loops(matmul)[0], "test4", {"arr": [0, 0], "key": 3})
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_relu_ann1)
    verify_trace_roundtrip(sch=sch, mod=matmul_relu)
    sch.unannotate(sch.get_loops(matmul)[0], "test1")
    sch.unannotate(sch.get_loops(matmul)[1], "test2")
    sch.unannotate(sch.get_loops(matmul)[1], "test3")
    sch.unannotate(sch.get_loops(matmul)[0], "test4")
    verify_trace_roundtrip(sch=sch, mod=matmul_relu)


def test_annotate_unannotate_block():
    sch = tvm.s_tir.Schedule(mod=matmul_relu, debug_mask="all")
    matmul = sch.get_sblock("matmul")
    relu = sch.get_sblock("relu")
    sch.annotate(matmul, "test1", "aaa")
    sch.annotate(relu, "test2", 0.22)
    sch.annotate(relu, "test3", ["aa", 1])
    sch.annotate(matmul, "test4", {"arr": [0, 0], "key": 3})
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_relu_ann2)
    verify_trace_roundtrip(sch=sch, mod=matmul_relu)
    sch.unannotate(matmul, "test1")
    sch.unannotate(relu, "test2")
    sch.unannotate(relu, "test3")
    sch.unannotate(matmul, "test4")
    verify_trace_roundtrip(sch=sch, mod=matmul_relu)


def test_get_output_blocks_single_output():
    sch = tvm.s_tir.Schedule(mod=matmul_relu, debug_mask="all")
    output_blocks = sch.get_output_blocks("root")
    assert len(output_blocks) == 1, "Unexpected number of blocks when 1 was expected"
    block = sch.get(output_blocks[0])
    assert block.name_hint == "relu"
    relu_block = sch.get_sblock("relu")
    assert sch.get(relu_block).same_as(block)


def test_get_output_blocks_multiple_outputs():
    sch = tvm.s_tir.Schedule(mod=matmul, debug_mask="all")
    output_blocks = sch.get_output_blocks("root")
    assert len(output_blocks) == 2, "Unexpected number of blocks when 2 were expected"
    block_1 = sch.get(output_blocks[0])
    assert block_1.name_hint == "init"
    block_2 = sch.get(output_blocks[1])
    assert block_2.name_hint == "update"
    init_block = sch.get_sblock("init")
    assert sch.get(init_block).same_as(block_1)
    update_block = sch.get_sblock("update")
    assert sch.get(update_block).same_as(block_2)


def test_get_output_blocks_nested():
    @Ts.function
    def blockized(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
    ) -> None:
        with Ts.sblock("blockized_B"):
            vio = Ts.axis.spatial(1, 0)
            vjo = Ts.axis.spatial(1, 0)
            for i, j in T.grid(128, 128):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] * 2.0

    sch = tvm.s_tir.Schedule(mod=blockized, debug_mask="all")
    output_blocks = sch.get_output_blocks("root")
    assert len(output_blocks) == 2, "Unexpected number of blocks when 2 were expected"
    block_1 = sch.get(output_blocks[0])
    assert block_1.name_hint == "blockized_B"
    block_2 = sch.get(output_blocks[1])
    assert block_2.name_hint == "B"
    blockized_block = sch.get_sblock("blockized_B")
    assert sch.get(blockized_block).same_as(block_1)
    b_block = sch.get_sblock("B")
    assert sch.get(b_block).same_as(block_2)

    sch = tvm.s_tir.Schedule(mod=blockized, debug_mask="all")
    output_blocks = sch.get_output_blocks("blockized_B")
    assert len(output_blocks) == 1, "Unexpected number of blocks when 1 were expected"
    block = sch.get(output_blocks[0])
    assert block.name_hint == "B"
    b_block = sch.get_sblock("B")
    assert sch.get(b_block).same_as(block)


if __name__ == "__main__":
    tvm_tirx.testing.main()
