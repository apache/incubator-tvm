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
"""An execution trace of a scheduling program"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import tvm_ffi
from tvm_ffi import Array, Map
from tvm_ffi import register_object as _register_object

from tvm_tirx.ir import FloatImm, IntImm, save_json
from tvm_tirx.tirx.index_map import IndexMap

from . import _ffi_api
from .instruction import ATTR_TYPE, INPUT_RV_TYPE, Instruction

if TYPE_CHECKING:
    from .schedule import Schedule


DECISION_TYPE = Any
JSON_TYPE = Any


def _json_from_tvm(obj):
    if obj is None:
        return None
    elif isinstance(obj, bool | int | float | str):
        return obj
    elif isinstance(obj, Array):
        return [_json_from_tvm(i) for i in obj]
    elif isinstance(obj, Map):
        return {_json_from_tvm(k): _json_from_tvm(v) for k, v in obj.items()}
    elif isinstance(obj, str):
        return str(obj)
    elif isinstance(obj, IntImm | FloatImm):
        return obj
    elif isinstance(obj, IndexMap):
        return save_json(obj)
    else:
        raise TypeError("Not supported type: " + str(type(obj)))


@_register_object("s_tir.Trace")
class Trace(tvm_ffi.Object):
    """An execution trace of a scheduling program.

    A trace has two parts:
    1) The instructions invoked so far
    2) The random decisions made upon those instructions, if any

    A trace can be serialized to:
    1) Roundtrippable JSON format: can be saved to file and loaded back
    2) Python syntax: allows users to copy-paste the trace to reproduce the scheduling process

    A trace can be applied to a TensorIR schedule by re-applying all its instructions possibly with
    their decisions accordingly. Re-sampling is invoked if a sampling instruction doesn't have its
    corresponding decision; Otherwise the existing decision will be reused accordingly.

    Attributes
    ----------
    insts : List[Instruction]
        The instructions invoked so far in the program execution
    decisions : Dict[Instruction, DECISION_TYPE]
        The random decisions made upon those instructions
    """

    insts: list[Instruction]
    decisions: dict[Instruction, DECISION_TYPE]

    def __init__(
        self,
        insts: list[Instruction],
        decisions: dict[Instruction, DECISION_TYPE],
    ) -> None:
        """Constructor

        Parameters
        ----------
        insts : List[Instruction]
            The instructions invoked so far in the program execution
        decisions : Dict[Instruction, DECISION_TYPE]
            The random decisions made upon those instructions
        """
        self.__init_handle_by_constructor__(
            _ffi_api.Trace,  # type: ignore # pylint: disable=no-member
            insts,
            decisions,
        )

    def get_decision(self, inst: Instruction) -> DECISION_TYPE | None:
        """Retrieve the decision made on a specific instruction

        Parameters
        ----------
        insts : Instruction
            The instruction whose decision is to be retrieved

        Returns
        -------
        decision : Optional[DECISION_TYPE]
            The corresponding decision; None if there is no decision made on the instruction
        """
        return _ffi_api.TraceGetDecision(self, inst)  # type: ignore # pylint: disable=no-member

    def append(
        self,
        inst: Instruction,
        decision: DECISION_TYPE | None = None,
    ) -> None:
        """Append a new instruction to the trace

        Parameters
        ----------
        insts : Instruction
            The new instruction to be appended
        decision : Optional[DECISION_TYPE] = None
            The random decision made on this instruction
        """
        _ffi_api.TraceAppend(self, inst, decision)  # type: ignore # pylint: disable=no-member

    def pop(self) -> Instruction | None:
        """Remove the last instruction, along with the decision made on that instruction, if any

        Returns
        -------
        popped_inst : Instruction
            Returns the instruction removed; std::nullopt if the trace is empty
        """
        return _ffi_api.TracePop(self)  # type: ignore # pylint: disable=no-member

    def apply_to_schedule(
        self,
        sch: "Schedule",
        remove_postproc: bool,
        decision_provider: (
            Callable[
                [Instruction, list[INPUT_RV_TYPE], list[ATTR_TYPE], DECISION_TYPE], DECISION_TYPE
            ]
            | None
        ) = None,
    ) -> None:
        """Apply the trace to a TensorIR schedule

        Parameters
        ----------
        sch : Schedule
            The schedule to be applied onto
        remove_postproc : bool
            If postprocessing instructions are removed
        decision_provider: Optional[Callable] = None
            A callback that allows users to mutate decisions on the fly when applying instructions.
            The signature of the callback is:
            - The 1st argument: The instruction
            - The 2nd argument: The input random variables
            - The 3rd argument: The attributes
            - The 4th argument: The decision
            - Return: A new decision
        """
        _ffi_api.TraceApplyToSchedule(  # type: ignore # pylint: disable=no-member
            self,
            sch,
            remove_postproc,
            decision_provider,
        )

    def as_json(self, remove_postproc: bool = False) -> JSON_TYPE:
        """Serialize the trace as a JSON-style object

        Parameters
        ----------
        remove_postproc : bool = False
            If postprocessing instructions are removed

        Returns
        -------
        json: JSON_TYPE
            The JSON-style object
        """
        obj = _ffi_api.TraceAsJSON(self, remove_postproc)  # type: ignore # pylint: disable=no-member
        return _json_from_tvm(obj)

    def as_python(self, remove_postproc: bool = False) -> list[str]:
        """Serialize the trace as a sequence of python statements

        Parameters
        ----------
        remove_postproc : bool = False
            If postprocessing instructions are removed

        Returns
        -------
        py_stmts: List[str]
            A sequence of python statements
        """
        return _ffi_api.TraceAsPython(self, remove_postproc)  # type: ignore # pylint: disable=no-member

    def with_decision(
        self,
        inst: Instruction,
        decision: DECISION_TYPE,
        remove_postproc: bool,
    ) -> "Trace":
        """Create a new trace with an instruction whose decision is changed,
        assuming this instruction exists in the resulting trace

        Parameters
        ----------
        inst : Instruction
            The instruction whose decision is to be changed
        decision : DECISION_TYPE
            The decision to be changed to
        remove_postproc : bool
            If postprocessing instructions are removed

        Returns
        -------
        trace: Trace
            The new trace with the decision changed
        """
        return _ffi_api.TraceWithDecision(  # type: ignore # pylint: disable=no-member
            self,
            inst,
            decision,
            remove_postproc,
        )

    def simplified(self, remove_postproc: bool) -> "Trace":
        """Simplify the trace with dead-code elimination

        Parameters
        ----------
        remove_postproc : bool
            If postprocessing instructions are removed

        Returns
        -------
        trace: Trace
            A simplified trace
        """
        return _ffi_api.TraceSimplified(self, remove_postproc)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def apply_json_to_schedule(json_obj: JSON_TYPE, sch: "Schedule") -> None:
        """Apply a JSON-serialized trace to a TensorIR schedule

        Parameters
        ----------
        json_obj : JSON_TYPE
            The JSON-serialized trace
        sch : Schedule
            The TensorIR schedule
        """
        _ffi_api.TraceApplyJSONToSchedule(json_obj, sch)  # type: ignore # pylint: disable=no-member

    def show(self, style: str | None = None) -> None:
        """A sugar for print highlighted TVM script.

        Parameters
        ----------
        style : str, optional

            Pygmentize printing style, auto-detected if None.  See
            `tvm_tirx.script.printer.highlight.cprint` for more details.

        """
        from tvm_tirx.script.printer.highlight import (  # pylint: disable=import-outside-toplevel
            cprint,
        )

        cprint(str(self), style=style)
