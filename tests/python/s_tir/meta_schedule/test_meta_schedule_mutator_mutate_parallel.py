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

from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

# pylint: disable=invalid-name, no-member


@Ts.function
def matmul(
    A: T.Tensor([512, 512], "float32"),
    B: T.Tensor([512, 512], "float32"),
    C: T.Tensor([512, 512], "float32"),
) -> None:
    for i, j, k in T.grid(512, 512, 512):  # type: ignore
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])  # type: ignore
            with Ts.init():
                C[vi, vj] = 0.0  # type: ignore
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


# pylint: enable=invalid-name, no-member


def _sch(decisions: list[list[int]], ann_val: int) -> Schedule:
    sch = Schedule(matmul, debug_mask="all")
    # pylint: disable=invalid-name
    d0, d1, d2 = decisions
    b0 = sch.get_sblock(name="C", func_name="main")
    root = sch.get_sblock(name="root", func_name="main")
    sch.get_consumers(block=b0)
    b1 = sch.cache_write(block=b0, write_tensor_index=0, storage_scope="global")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6, v7, v8 = sch.sample_perfect_tile(
        loop=l2,
        n=4,
        max_innermost_factor=64,
        decision=d0,
    )
    l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])
    v13, v14, v15, v16 = sch.sample_perfect_tile(
        loop=l3,
        n=4,
        max_innermost_factor=64,
        decision=d1,
    )
    l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])
    v21, v22 = sch.sample_perfect_tile(
        loop=l4,
        n=2,
        max_innermost_factor=64,
        decision=d2,
    )
    l23, l24 = sch.split(loop=l4, factors=[v21, v22])
    sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
    sch.reverse_compute_at(block=b1, loop=l18, preserve_unit_loops=True)
    sch.annotate(block_or_loop=root, ann_key="meta_schedule.parallel", ann_val=ann_val)
    # pylint: enable=invalid-name
    return sch


def _make_mutator(target: Target, max_jobs_per_core: int) -> ms.Mutator:
    ctx = ms.TuneContext(
        mod=matmul,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[],
            mutator_probs={ms.mutator.MutateParallel(max_jobs_per_core): 1.0},
        ),
    )
    return next(iter(ctx.space_generator.mutator_probs.keys()))


def test_mutate_parallel_matmul():
    mutator = _make_mutator(
        target=Target({"kind": "llvm", "num-cores": 16}),
        max_jobs_per_core=256,
    )
    sch = _sch(
        decisions=[
            [4, 32, 4, 1],
            [8, 4, 8, 2],
            [512, 1],
        ],
        ann_val=64,
    )
    results = set()
    for _ in range(100):
        trace = mutator.apply(sch.trace)
        ann_val = int(trace.insts[-1].inputs[1])
        results.add(ann_val)
        if len(results) == 3:
            break
    assert len(results) == 3
    assert results == {4, 32, 4096}


if __name__ == """__main__""":
    test_mutate_parallel_matmul()
