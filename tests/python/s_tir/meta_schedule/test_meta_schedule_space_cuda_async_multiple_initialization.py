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
"""Tests for MetaSchedule search space on CUDA"""

# isort: off
from typing import Literal

# isort: on
from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.meta_schedule.testing.space_generation import get_rules
from tvm.s_tir.meta_schedule.testing.te_workload import create_te_workload
from tvm_tirx.ir import IRModule
from tvm_tirx.target import Target


def generate_design_space(
    kind: Literal["llvm", "cuda", "hexagon"],
    mod: IRModule,
    target: Target,
    types: type | tuple[type, ...],
    sch_rules: list[ms.ScheduleRule] | None = None,
    initialize_time: int = 1,
) -> list[Schedule]:
    if sch_rules is None:
        sch_rules = get_rules(kind, types)
    else:
        assert types is None
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=sch_rules,
            postprocs=[],
            mutator_probs={},
        ),
        task_name="test",
    )
    # each time cloning will trigger one more initialization
    for _ in range(initialize_time - 1):
        ctx = ctx.clone()
    return ctx.generate_design_space()


def _target():
    return Target("nvidia/geforce-rtx-3070")


def _design_space(mod):
    return generate_design_space(
        kind="cuda",
        mod=mod,
        target=_target(),
        types=ms.ScheduleRule,
        initialize_time=100,
    )


def test_c2d():
    mod = create_te_workload("C2D", 0)
    actual = _design_space(mod)
    assert len(actual) == 3


def test_gmm():
    mod = create_te_workload("GMM", 0)
    actual = _design_space(mod)
    assert len(actual) == 3


if __name__ == "__main__":
    test_c2d()
    test_gmm()
