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

# isort: off
from typing import Literal

# isort: on

import tvm_ffi

from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir.schedule import Trace
from tvm.s_tir.schedule.testing import verify_trace_roundtrip
from tvm_tirx.ir import IRModule
from tvm_tirx.target import Target


def get_rules(
    kind: Literal["llvm", "cuda", "hexagon"],
    types: type | tuple[type, ...],
) -> list[ms.ScheduleRule]:
    """Get default schedule rules"""
    rules = ms.ScheduleRule.create(kind)
    return [rule for rule in rules if isinstance(rule, types)]


def structural_equal_no_gs(mod1: IRModule, mod2: IRModule) -> bool:
    """
    Checks structural equality but ignores global symbols
    """

    # for every function in the modules, remove global symbols from the attrs and then compare
    def remove_global_symbols(mod: IRModule) -> IRModule:
        stripped_mod = IRModule()
        for global_var in mod.get_global_vars():
            func = mod[global_var]
            stripped_mod[global_var] = func.without_attr("global_symbol")
        return stripped_mod

    return tvm_ffi.structural_equal(remove_global_symbols(mod1), remove_global_symbols(mod2))


def generate_design_space(
    kind: Literal["llvm", "cuda", "hexagon"],
    mod: IRModule,
    target: Target,
    types: type | tuple[type, ...],
    sch_rules: list[ms.ScheduleRule] | None = None,
) -> list[Schedule]:
    if sch_rules is None:
        sch_rules = get_rules(kind, types)
    else:
        assert types is None
    return ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=sch_rules,
            postprocs=[],
            mutator_probs={},
        ),
        task_name="test",
    ).generate_design_space()


def _find_match_sketch_id(
    mod: IRModule,
    sketches: list[Schedule],
    expected_mod: IRModule,
    expected_decision: list[tuple[str, list[int]]],
    *,
    debug_mask="all",
) -> int | None:
    for sketch_id, sketch in enumerate(sketches):
        i = 0
        new_decisions = {}
        for inst in sketch.trace.insts:
            if not inst.kind.name.startswith("Sample"):
                continue
            assert i < len(expected_decision)
            if inst.kind.name == expected_decision[i][0]:
                new_decisions[inst] = expected_decision[i][1]
                i += 1
        if len(new_decisions) != len(expected_decision):
            continue
        sch = Schedule(mod, debug_mask=debug_mask)
        Trace(
            insts=sketch.trace.insts,
            decisions=new_decisions,
        ).apply_to_schedule(sch, remove_postproc=True)
        if structural_equal_no_gs(sch.mod, expected_mod):
            verify_trace_roundtrip(sch=sch, mod=mod, debug_mask=debug_mask, text_format="json")
            return sketch_id
    return None


def check_sketches(
    mod: IRModule,
    sketches: list[Schedule],
    expected_mods: list[IRModule],
    expected_decisions: list[list[tuple[str, list[int]]]],
    *,
    debug_mask="all",
):
    assert len(expected_mods) == len(expected_decisions)
    assert len(sketches) == len(expected_mods)
    expected_mods = [
        IRModule({"main": m}) if not isinstance(m, IRModule) else m for m in expected_mods
    ]
    sketches = list(sketches)
    for expected_id, (expected_mod, expected_decision) in enumerate(
        zip(expected_mods, expected_decisions)
    ):
        sketch_id = _find_match_sketch_id(
            mod,
            sketches,
            expected_mod,
            expected_decision,
            debug_mask=debug_mask,
        )
        if sketch_id is None:
            raise AssertionError(
                f"Expected sketch #{expected_id} doesn't exist in the generated sketches."
            )
        sketches.pop(sketch_id)


def print_sketches(sketches: list[Schedule]):
    for i, sch in enumerate(sketches):
        print(f"###### {i}")
        sch.mod.show()
        for inst in sch.trace.insts:
            if inst in sch.trace.decisions:
                print(f'("{inst.kind.name}", {sch.trace.decisions[inst]}),')
