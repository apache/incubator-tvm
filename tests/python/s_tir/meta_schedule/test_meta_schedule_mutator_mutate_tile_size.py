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
import operator
from functools import reduce

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


def _sch(decisions: list[list[int]]) -> Schedule:
    sch = Schedule(matmul, debug_mask="all")
    # pylint: disable=invalid-name
    (d0,) = decisions
    b0 = sch.get_sblock(name="C", func_name="main")
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
    l17, l18, l19, l20 = sch.split(loop=l3, factors=[8, 4, 8, 2])
    l23, l24 = sch.split(loop=l4, factors=[512, 1])
    sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
    sch.reverse_compute_at(block=b1, loop=l18, preserve_unit_loops=True)
    # pylint: enable=invalid-name
    return sch


def _make_mutator(target: Target) -> ms.Mutator:
    ctx = ms.TuneContext(
        mod=matmul,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[],
            mutator_probs={ms.mutator.MutateTileSize(): 1.0},
        ),
    )
    return next(iter(ctx.space_generator.mutator_probs.keys()))


def test_mutate_tile_size_matmul():
    mutator = _make_mutator(
        target=Target({"kind": "llvm", "num-cores": 16}),
    )
    results = {}
    sch = _sch(decisions=[[4, 32, 4, 1]])
    for _ in range(1000):
        trace = mutator.apply(sch.trace)
        assert trace.insts[4].kind.name == "SamplePerfectTile"
        decision = trace.decisions[trace.insts[4]]
        decision = [int(x) for x in decision]
        results[str(decision)] = decision
        assert reduce(operator.mul, decision, 1) == 512
    assert len(results) > 15


def test_mutate_sample_categorical_single_candidate():
    mutator = _make_mutator(
        target=Target({"kind": "llvm", "num-cores": 16}),
    )
    sch = Schedule(matmul, debug_mask="all")
    sch.sample_categorical(candidates=[1], probs=[1.0], decision=0)

    # The mutator finds the SampleCategorical has only one candidate, and thus skips it.
    trace = mutator.apply(sch.trace)
    assert trace is None


if __name__ == "__main__":
    test_mutate_tile_size_matmul()
    test_mutate_sample_categorical_single_candidate()
