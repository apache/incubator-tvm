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
"""
Apply ScheduleRules onto an IRModule to generate default schedules without tuning,
or a space for MetaSchedule tuning
"""

from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.ir import IRModule
from tvm_tirx.ir.transform import PassContext, module_pass
from tvm_tirx.target import Target

from .schedule_rule import ScheduleRule


def _is_scheduled(func: tirx.Function) -> bool:
    if not isinstance(func, tirx.Function):
        return False
    if "tirx.is_scheduled" not in func.attrs:
        return False
    return func.attrs["tirx.is_scheduled"] == 1


def _get_target(func: tirx.Function, default_target: Target | None = None) -> Target:
    target = func.attrs.get("target")
    if target is None:
        if default_target is None:
            raise ValueError("A target must be supplied or attached to the function")
        return default_target
    else:
        return target


@module_pass(opt_level=0, name="ApplyDefaultSchedule")
class ApplyDefaultSchedule:  # pylint: disable=too-few-public-methods
    """A IRModule pass that applies a list of ScheduleRules to all Functions in the module."""

    def __init__(self, *rules: ScheduleRule, target: Target | None = None):
        """Construct a new ApplyDefaultSchedule pass.

        Parameters
        ----------
        *rules : ScheduleRule
            The ScheduleRules to apply to all Functions in the module.
        target : Target or None
            Default compilation target for functions without a target attribute.
        """
        self.rules = list(rules)
        self.target = Target(target) if target is not None else None

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        updated_functions = {}
        for g_var, func in mod.functions_items():
            if isinstance(func, tirx.Function) and not _is_scheduled(func):
                target = _get_target(func, self.target)

                sch = _apply_rules(func, target, self.rules, tunable=False)
                if sch is not None:
                    assert len(sch) == 1
                    updated_functions[g_var] = (
                        sch[0].mod["main"].with_attr("tirx.is_scheduled", True)
                    )
        for g_var, func in updated_functions.items():
            mod[g_var] = func
        return mod


def _apply_rules(
    func: tirx.Function,
    target: Target,
    rules: list[ScheduleRule],
    tunable: bool,
) -> list[s_tir.Schedule] | None:
    for rule in rules:
        space = rule.apply(func, target, tunable)
        if space is None:
            continue
        if isinstance(space, s_tir.Schedule):
            space = [space]
        return space
    return None
