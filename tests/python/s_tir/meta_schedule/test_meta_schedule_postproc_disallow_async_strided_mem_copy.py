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

import tvm
import tvm_tirx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def _target() -> Target:
    return Target("qcom/hexagon-v68", host="llvm")


def _create_context(mod, target) -> ms.TuneContext:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[
                ms.postproc.DisallowAsyncStridedMemCopy(),
            ],
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

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

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def test_postproc_disallow_async_strided_mem_copy_allows():
    mod = Matmul
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")

    matmul_block = sch.get_sblock("matmul")

    loops = sch.get_loops(matmul_block)
    cache_read = sch.cache_read(matmul_block, 0, "global.vtcm")

    sch.compute_at(cache_read, loops[1])

    sch.annotate(loops[1], "software_pipeline_stage", [0, 1])
    sch.annotate(loops[1], "software_pipeline_order", [0, 1])
    sch.annotate(loops[1], "software_pipeline_async_stages", [0])

    ctx = _create_context(sch.mod, target=_target())
    sch.mod.show()
    assert ctx.space_generator.postprocs[0].apply(sch)


def test_postproc_disallow_async_strided_mem_copy_disallows():
    mod = Matmul
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")

    matmul_block = sch.get_sblock("matmul")

    loops = sch.get_loops(matmul_block)
    # Make it a strided mem copy.
    cache_read = sch.cache_read(matmul_block, 1, "global.vtcm")

    sch.compute_at(cache_read, loops[1])
    sch.annotate(loops[1], "software_pipeline_stage", [0, 1])
    sch.annotate(loops[1], "software_pipeline_order", [0, 1])
    sch.annotate(loops[1], "software_pipeline_async_stages", [0])

    sch.mod.show()
    ctx = _create_context(sch.mod, target=_target())
    assert not ctx.space_generator.postprocs[0].apply(sch)


if __name__ == "__main__":
    test_postproc_disallow_async_strided_mem_copy_allows()
    test_postproc_disallow_async_strided_mem_copy_disallows()
