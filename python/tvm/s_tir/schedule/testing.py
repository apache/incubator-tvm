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
# pylint: disable=dangerous-default-value
"""Testing utilities for the TensorIR schedule API"""

from collections.abc import Sequence
from typing import Any

import tvm_tirx
from tvm.s_tir.schedule import Schedule, Trace
from tvm_tirx.ir import IRModule, assert_structural_equal
from tvm_tirx.tirx import Function


def assert_structural_equal_ignore_global_symbol(
    func1: Function,
    func2: Function,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Asserts that Functions func1 and func2 are structurally equal, setting both
    their global symbol attributes to main so that the global symbol
    will not be a point of comparison.
    """
    assert_structural_equal(
        func1.with_attr("global_symbol", "main"),
        func2.with_attr("global_symbol", "main"),
        *args,
        **kwargs,
    )


def verify_trace_roundtrip(
    sch: Schedule,
    mod: Function | IRModule,
    *,
    debug_mask: str | int = "all",
    text_format: str | Sequence[str] = ["python", "json"],
) -> Schedule:
    """Serialize a traced schedule to JSON, then replay the JSON trace by applying to
    a fresh new schedule, verifying the reproducibility of scheduling.

    Parameters
    ----------
    sch : s_tir.Schedule
        The traced TensorIR schedule to be verified
    mod : Union[Function, IRModule]
        The IRModule or Function to construct the fresh new schedule
    debug_mask : Union[str, int]
        Do extra correctness checking after the class creation and each time
        after calling the Replace method.
        Possible choices of `debug_mask`:
        1) "all" - Turn on all the checks
        2) "none" - Turn off all the checks
        3) An integer - Turn on checks according to the bitmasks provided in ScheduleDebugMask
    text_format: Union[str, Sequence[str]]
        The text format or formats whose round-trip behavior should be
        validated.  If a single string, validate round-trips through
    """
    from tvm_tirx.script import tirx as T  # pylint: disable=import-outside-toplevel

    if not isinstance(text_format, str):
        for opt in text_format:
            new_sch = verify_trace_roundtrip(sch, mod, debug_mask=debug_mask, text_format=opt)
        return new_sch

    trace = sch.trace
    assert trace is not None

    # Step 1. Perform a round-trip through the text-format
    new_sch = Schedule(mod=mod, debug_mask=debug_mask)
    if text_format == "json":
        json_obj = trace.as_json()
        Trace.apply_json_to_schedule(json_obj=json_obj, sch=new_sch)
    elif text_format == "python":
        py_trace = "\n".join(trace.as_python())
        vars_dict = {"T": T}
        vars_dict.update(tvm_tirx.tirx.__dict__)
        exec(py_trace, vars_dict, {"sch": new_sch})  # pylint: disable=exec-used
    else:
        assert text_format in ("json", "python"), f"Unknown text format: {text_format}"

    # Step 2. Verify that the round-trip produced the same scheduling
    assert_structural_equal(new_sch.mod, sch.mod)

    # Step 3. Check the consistency of the text format between the old and new traces
    py_repr = "\n".join(trace.as_python())
    new_py_repr = "\n".join(new_sch.trace.as_python())
    assert py_repr == new_py_repr

    # Step 4. Return the new schedule in case it could be useful
    return new_sch
