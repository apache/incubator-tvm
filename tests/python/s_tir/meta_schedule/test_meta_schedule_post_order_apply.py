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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# ruff: noqa: F401

import math
import sys

import pytest
import tvm_ffi
from tvm_ffi import register_global_func

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import te
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule import TuneContext
from tvm.s_tir.meta_schedule.schedule_rule import PyScheduleRule
from tvm.s_tir.meta_schedule.space_generator import PostOrderApply
from tvm.s_tir.schedule import SBlockRV, Schedule
from tvm.s_tir.utils import derived_object
from tvm_tirx.ir.module import IRModule
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,
# fmt: off

def get_matmul_packed(m, n, k, lhs_type="int8", rhs_dtype="int8", acc_dtype="int32"):
    X = te.placeholder((m, k), name="X", dtype=lhs_type)
    W = te.placeholder((n, k), name="W", dtype=rhs_dtype)

    ak = te.reduce_axis((0, k), name="k")
    matmul = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype(acc_dtype) * W[j, ak].astype(acc_dtype),
            axis=ak,
        ),
        name="compute",
    )
    return te.create_function([X, W, matmul])

@tvm_tirx.script.ir_module
class Matmul:
    @Ts.function
    def main(
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        C: T.Tensor((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main"})

        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

@tvm_tirx.script.ir_module
class DuplicateMatmul:
    @Ts.function
    def main(
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        C: T.Tensor((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main"})

        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

@tvm_tirx.script.ir_module
class TrinityMatmul:
    @Ts.function
    def main(A: T.Tensor((1024, 1024), 'float32'), D: T.Tensor((1024, 1024), 'float32')) -> None:
        T.func_attr({"global_symbol": "main"})

        B = Ts.sblock_alloc_tensor((1024, 1024), "float32")
        C = Ts.sblock_alloc_tensor((1024, 1024), "float32")

        for i, j in T.grid(1024, 1024):
            with Ts.sblock("A"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(1024, 1024):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 3.0
        for i, j in T.grid(1024, 1024):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                D[vi, vj] = C[vi, vj] * 5.0

@tvm_tirx.script.ir_module
class TrinityMatmulProcessedForReference:
    @Ts.function
    def main(
        A: T.Tensor([1024, 1024], dtype="float32"), D: T.Tensor([1024, 1024], dtype="float32")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main"})

        # body
        # with tirx.block("root")
        B = Ts.sblock_alloc_tensor([1024, 1024], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1 in T.grid(16, 64, 64, 16):
            with Ts.sblock("A"):
                vi = Ts.axis.S(1024, i0_0 * 64 + i0_1)
                vj = Ts.axis.S(1024, i1_0 * 16 + i1_1)
                Ts.reads([A[vi, vj]])
                Ts.writes([B[vi, vj]])
                B[vi, vj] = A[vi, vj] * T.float32(2)
        for i0_0, i1_0, i0_1, i1_1 in T.grid(16, 64, 64, 16):
            with Ts.sblock("C"):
                vi = Ts.axis.S(1024, i0_0 * 64 + i0_1)
                vj = Ts.axis.S(1024, i1_0 * 16 + i1_1)
                Ts.reads([B[vi, vj]])
                Ts.writes([D[vi, vj]])
                D[vi, vj] = (B[vi, vj] + T.float32(3)) * T.float32(5)

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _is_root(sch: Schedule, block: SBlockRV) -> bool:
    return sch.get_sref(block).parent is None


def _check_correct(schedule: Schedule):
    trace = schedule.trace
    for inst in trace.decisions:
        assert math.prod(trace.decisions[inst]) == 1024


@derived_object
class WowSoFancyScheduleRule(PyScheduleRule):
    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: SBlockRV) -> list[Schedule]:
        if _is_root(sch, block):
            return [sch]
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[2, 4, 64, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[4, 64, 2, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        return [new_sch]


@derived_object
class DoubleScheduleRule(PyScheduleRule):
    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: SBlockRV) -> list[Schedule]:
        if _is_root(sch, block):
            return [sch]
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[4, 64, 2, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[2, 4, 64, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        result = [new_sch]
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[4, 64, 2, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[2, 4, 64, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        result.append(new_sch)
        return result


@derived_object
class TrinityDoubleRule(PyScheduleRule):
    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: SBlockRV) -> list[Schedule]:
        if _is_root(sch, block):
            return [sch]
        new_sch = sch.copy()
        i, j = new_sch.get_loops(block=block)
        i_0, i_1 = new_sch.split(loop=i, factors=[16, 64])
        j_0, j_1 = new_sch.split(loop=j, factors=[64, 16])
        new_sch.reorder(i_0, j_0, i_1, j_1)
        result = [new_sch]
        new_sch = sch.copy()
        i, j = new_sch.get_loops(block=block)
        i_0, i_1 = new_sch.split(loop=i, factors=[2, 512])
        j_0, j_1 = new_sch.split(loop=j, factors=[2, 512])
        new_sch.reorder(i_0, j_0, i_1, j_1)
        result.append(new_sch)
        return result


@derived_object
class ReorderScheduleRule(PyScheduleRule):
    def _initialize_with_tune_context(self, context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: SBlockRV) -> list[Schedule]:
        if _is_root(sch, block):
            return [sch]
        new_sch = sch.copy()
        i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = new_sch.get_loops(block=block)
        new_sch.reorder(i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3, i_0, j_0)
        result = [new_sch]
        new_sch = sch.copy()
        i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = new_sch.get_loops(block=block)
        new_sch.reorder(i_1, j_3, i_0, j_0, j_1, k_0, i_2, j_2, k_1, i_3)
        result.append(new_sch)
        return result


def test_meta_schedule_post_order_apply():
    mod = Matmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Test Task",
        space_generator=PostOrderApply(
            sch_rules=[WowSoFancyScheduleRule()],
            postprocs=[],
            mutator_probs={},
        ),
    )
    post_order_apply = context.space_generator
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 1
    assert not tvm_ffi.structural_equal(schs[0].mod, mod)
    _check_correct(schs[0])


def test_meta_schedule_post_order_apply_double():
    mod = Matmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Double Rules Task",
        space_generator=PostOrderApply(
            sch_rules=[DoubleScheduleRule()],
            postprocs=[],
            mutator_probs={},
        ),
    )
    post_order_apply = context.space_generator
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 2
    for sch in schs:
        assert not tvm_ffi.structural_equal(sch.mod, mod)
        _check_correct(sch)


def test_meta_schedule_post_order_apply_multiple():
    mod = Matmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Double Rules Task",
        space_generator=PostOrderApply(
            sch_rules=[DoubleScheduleRule(), ReorderScheduleRule()],
            postprocs=[],
            mutator_probs={},
        ),
    )
    post_order_apply = context.space_generator
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 4
    for sch in schs:
        assert not tvm_ffi.structural_equal(sch.mod, mod)
        _check_correct(sch)


def test_meta_schedule_post_order_apply_duplicate_matmul():
    mod = DuplicateMatmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Duplicate Matmul Task",
        space_generator=PostOrderApply(
            sch_rules=[WowSoFancyScheduleRule()],
            postprocs=[],
            mutator_probs={},
        ),
    )
    post_order_apply = context.space_generator
    with pytest.raises(
        RuntimeError,
        match=r".*Duplicated block name matmul in function main not supported!",
    ):
        post_order_apply.generate_design_space(mod)


def test_meta_schedule_post_order_apply_remove_block():
    @derived_object
    class RemoveBlock(PyScheduleRule):
        def _initialize_with_tune_context(self, context: "TuneContext") -> None:
            pass

        def apply(self, sch: Schedule, block: SBlockRV) -> list[Schedule]:
            if _is_root(sch, block):
                return [sch]
            sch = sch.copy()
            if sch.get(block).name_hint == "B":
                sch.compute_inline(block)
            return [sch]

    def correct_trace(a, b, c, d):
        return "\n".join(
            [
                "# from tvm import s_tir",
                "def apply_trace(sch: s_tir.Schedule) -> None:",
                '  b0 = sch.get_sblock(name="A", func_name="main")',
                '  b1 = sch.get_sblock(name="B", func_name="main")',
                '  b2 = sch.get_sblock(name="C", func_name="main")',
                "  sch.compute_inline(block=b1)",
                "  l3, l4 = sch.get_loops(block=b2)",
                "  l5, l6 = sch.split(loop=l3, factors="
                + str(a)
                + ", preserve_unit_iters=True, disable_predication=False)",
                "  l7, l8 = sch.split(loop=l4, factors="
                + str(b)
                + ", preserve_unit_iters=True, disable_predication=False)",
                "  sch.reorder(l5, l7, l6, l8)",
                "  l9, l10 = sch.get_loops(block=b0)",
                "  l11, l12 = sch.split(loop=l9, factors="
                + str(c)
                + ", preserve_unit_iters=True, disable_predication=False)",
                "  l13, l14 = sch.split(loop=l10, factors="
                + str(d)
                + ", preserve_unit_iters=True, disable_predication=False)",
                "  sch.reorder(l11, l13, l12, l14)",
            ]
        )

    mod = TrinityMatmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Remove Block Task",
        space_generator=PostOrderApply(
            sch_rules=[RemoveBlock(), TrinityDoubleRule()],
            postprocs=[],
            mutator_probs={},
        ),
    )
    post_order_apply = context.space_generator
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 4
    for sch in schs:
        with pytest.raises(
            tvm.s_tir.schedule.schedule.ScheduleError,
            match="ScheduleError: An error occurred in the schedule primitive 'get-block'.",
        ):
            sch.get_sblock("B", "main")
        sch_trace = sch.trace.simplified(True)
        assert (
            str(sch_trace) == correct_trace([16, 64], [64, 16], [2, 512], [2, 512])
            or str(sch_trace) == correct_trace([2, 512], [2, 512], [2, 512], [2, 512])
            or str(sch_trace) == correct_trace([16, 64], [64, 16], [16, 64], [64, 16])
            or str(sch_trace) == correct_trace([2, 512], [2, 512], [16, 64], [64, 16])
        )


def test_target_sblocks_search_space():
    # Test that specific blocks of trinity matmul can be targeted.
    def filter_fn(block, target_names) -> bool:
        return block.name_hint in target_names

    def _get_sch(filter_fn):
        mod = TrinityMatmul
        context = TuneContext(
            mod=mod,
            target=Target("llvm"),
            task_name="Custom Search Space Task",
            space_generator=PostOrderApply(
                f_block_filter=filter_fn,
                sch_rules=[TrinityDoubleRule()],
                postprocs=[],
                mutator_probs={},
            ),
        )
        post_order_apply = context.space_generator
        schs = post_order_apply.generate_design_space(mod)
        return schs

    # Start by checking that by default each block has a space generated.
    schs = _get_sch(None)
    assert len(schs) == 8

    # Next check that we can target a specific block and only get its' revelant schedules.
    schs = _get_sch(lambda block: filter_fn(block, ["B"]))
    assert len(schs) == 2

    ## Check that extracting two blocks works.
    schs = _get_sch(lambda block: filter_fn(block, ["A", "C"]))
    assert len(schs) == 4

    ## Finally check that all blocks can be extracted by name.
    schs = _get_sch(lambda block: filter_fn(block, ["A", "B", "C"]))
    assert len(schs) == 8


@pytest.mark.parametrize(
    "target,mod,expected_intr",
    [
        (
            Target(
                {
                    "kind": "llvm",
                    "device": "arm_cpu",
                    "mtriple": "aarch64-linux-gnu",
                    "mattr": ["+neon"],
                    "num-cores": 2,
                }
            ),
            IRModule({"main": get_matmul_packed(128, 128, 128, "int8", "int8", "int32")}),
            "dot_4x4_i8i8s32_neon",
        ),
        (
            Target(
                {
                    "kind": "llvm",
                    "device": "arm_cpu",
                    "mtriple": "aarch64-linux-gnu",
                    "mattr": ["+neon", "+v8.2a", "+dotprod"],
                    "num-cores": 2,
                }
            ),
            IRModule({"main": get_matmul_packed(128, 128, 128, "int8", "int8", "int32")}),
            "dot_4x4_i8i8s32_sdot",
        ),
        (
            Target(
                {
                    "kind": "llvm",
                    "device": "arm_cpu",
                    "mtriple": "aarch64-linux-gnu",
                    "mattr": ["+neon", "+v8.2a", "+dotprod"],
                    "num-cores": 2,
                }
            ),
            IRModule({"main": get_matmul_packed(128, 128, 128, "uint8", "uint8", "uint32")}),
            "dot_4x4_u8u8u32_udot",
        ),
        (
            Target(
                {
                    "kind": "llvm",
                    "device": "arm_cpu",
                    "mtriple": "aarch64-linux-gnu",
                    "mattr": ["+neon", "+v8.2a", "+dotprod"],
                    "num-cores": 2,
                }
            ),
            IRModule({"main": get_matmul_packed(128, 128, 128, "uint8", "uint8", "int32")}),
            "dot_4x4_u8u8i32_hdot",
        ),
    ],
)
def test_meta_schedule_post_order_apply_arm_intrin(target, mod, expected_intr):
    context = TuneContext(
        mod=mod,
        target=target,
        task_name="Arm Intrinsic Task",
        space_generator=PostOrderApply(),  # Triggers default generator
        rand_state=1,  # Change it while all tests are not passing
    )
    post_order_apply = context.space_generator
    schs = post_order_apply.generate_design_space(mod)

    assert len(schs) != 0

    for sch in schs:
        sch.enter_postproc()

        for proc in context.space_generator.postprocs:
            proc.apply(sch)

    assert any(["call_llvm_pure_intrin" in sch.mod.script() for sch in schs])
    assert any([expected_intr in str(sch.trace) for sch in schs])


def test_meta_schedule_derived_object():
    @derived_object
    class RemoveBlock(PyScheduleRule):
        @classmethod
        def class_construct(cls):
            return cls()

        @staticmethod
        def static_construct():
            return RemoveBlock()

    inst_by_init = RemoveBlock()
    assert isinstance(inst_by_init, RemoveBlock)

    inst_by_classmethod = RemoveBlock.class_construct()
    assert isinstance(inst_by_classmethod, RemoveBlock)

    inst_by_staticmethod = RemoveBlock.static_construct()
    assert isinstance(inst_by_staticmethod, RemoveBlock)


if __name__ == "__main__":
    tvm_tirx.testing.main()
