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
def element_wise(
    A: T.Tensor([512, 512], dtype="float32"), B: T.Tensor([512, 512], dtype="float32")
) -> None:
    for i, j in T.grid(512, 512):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] + 1.0


# pylint: enable=invalid-name, no-member


def _sch() -> Schedule:
    sch = Schedule(element_wise, debug_mask="all")
    # pylint: disable=invalid-name
    b0 = sch.get_sblock(name="C", func_name="main")
    l1, l2 = sch.get_loops(block=b0)
    l3 = sch.fuse(l1, l2)
    v4 = sch.sample_categorical(
        candidates=[32, 64, 128, 256, 512, 1024],
        probs=[
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
            0.16666666666666666,
        ],
        decision=3,
    )
    l5, l6 = sch.split(loop=l3, factors=[None, v4])
    sch.bind(loop=l5, thread_axis="blockIdx.x")
    sch.bind(loop=l6, thread_axis="threadIdx.x")
    # pylint: enable=invalid-name
    return sch


def _make_mutator(target: Target) -> ms.Mutator:
    ctx = ms.TuneContext(
        mod=element_wise,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[],
            mutator_probs={ms.mutator.MutateThreadBinding(): 1.0},
        ),
    )
    return next(iter(ctx.space_generator.mutator_probs.keys()))


def test_mutate_thread_binding():
    mutator = _make_mutator(target=Target("cuda"))
    sch = _sch()
    results = set()
    for _ in range(100):
        trace = mutator.apply(sch.trace)
        decision = trace.decisions[trace.insts[-4]]
        results.add(decision)
        if len(results) == 5:
            break
    assert len(results) == 5
    assert results == {0, 1, 2, 4, 5}


if __name__ == "__main__":
    test_mutate_thread_binding()
