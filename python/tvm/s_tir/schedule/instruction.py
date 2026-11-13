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
"""Schedule instructions each corresponds to a schedule primitive"""

from typing import TYPE_CHECKING, Any

import tvm_ffi
from tvm_ffi import register_object as _register_object

from . import _ffi_api

if TYPE_CHECKING:
    from .schedule import RAND_VAR_TYPE

    INPUT_RV_TYPE = RAND_VAR_TYPE | float | int | str | None  # pylint: disable=invalid-name
    OUTPUT_RV_TYPE = RAND_VAR_TYPE  # pylint: disable=invalid-name
    ATTR_TYPE = Any
else:
    INPUT_RV_TYPE = OUTPUT_RV_TYPE = ATTR_TYPE = Any


@_register_object("s_tir.InstructionKind")
class InstructionKind(tvm_ffi.Object):
    """Kind of an instruction, e.g. Split, Reorder, etc.
    Besides the name, every kind of instruction has its own properties, including:
    1) A boolean indicating if the instruction is pure, i.e. change nothing in the schedule state
    2) A functor that applies the instruction to a TensorIR schedule
    3) A functor that converts the instruction to a statement in python syntax
    4) A functor that serialize its attributes to JSON
    5) A functor that deserialize its attributes from JSON

    Unlike `tvm_tirx.ir.op`, `InstructionKind` doesn't support unstructured properties,
    mainly because there is no such usecase yet to add any other property.

    Attributes
    ----------
    name : str
        The name of a kind of instructions

    Note
    ----
    The functor properties are not exposed on python side at the moment
    """

    name: str

    @property
    def is_pure(self) -> bool:
        """Indicates if the instruction is pure, i.e. removing it alone doesn't mutate the schedule
        state. For example, the instruction `GetSBlock` is pure because it changes
        nothing, while `ComputeInline` is not because removing it leads to a different resulting
        schedule.

        Returns
        -------
        pure : bool
            The boolean flag indicating if the instruction is pure
        """
        return bool(self._is_pure)

    @staticmethod
    def get(name: str) -> "InstructionKind":
        """Retrieve an InstructionKind using its name

        Parameters
        ----------
        name : str
            The registered name of the InstructionKind

        Returns
        -------
        kind : InstructionKind
            The InstructionKind retrieved
        """
        return _ffi_api.InstructionKindGet(name)  # type: ignore # pylint: disable=no-member


@_register_object("s_tir.Instruction")
class Instruction(tvm_ffi.Object):
    """Schedule instructions each corresponds to a schedule primitive

    Attributes
    ----------
    kind : InstructionKind
        The kind of the instruction
    inputs : List[INPUT_RV_TYPE]
        The input random variables of the instruction,
        and the type of each element can be one of the following:
        - SBlockRV
        - LoopRV
        - ExprRV
        - float
        - int
        - str
        - None
    attrs : List[ATTR_TYPE]
        The attributes of the instruction. Similar to attributes of an operator,
        attributes of an instruction are arbitrary constant metadata required by the instructions.
        For example, the name of the block to be retrieved in `GetSBlock`.
    outputs : List[OUTPUT_RV_TYPE]
        The output random variables of the instruction,
        and the type of each element can be one of the following:
        - SBlockRV
        - LoopRV
        - ExprRV, atomic variables only, won't be constants or composite Expr
    """

    kind: InstructionKind
    inputs: list[INPUT_RV_TYPE]
    attrs: list[ATTR_TYPE]
    outputs: list[OUTPUT_RV_TYPE]

    def __init__(
        self,
        kind: InstructionKind,
        inputs: list[INPUT_RV_TYPE],
        attrs: list[ATTR_TYPE],
        outputs: list[OUTPUT_RV_TYPE],
    ) -> None:
        """Constructor

        Parameters
        ----------
        kind : InstructionKind
            The kind of the instruction
        inputs : List[INPUT_RV_TYPE]
            The input random variables of the instruction,
            and the type of each element can be one of the following:
            - SBlockRV
            - LoopRV
            - ExprRV
            - float
            - int
            - str
            - None
        attrs : List[ATTR_TYPE]
            The attributes of the instruction. Similar to attributes of an operator,
            attributes of an instruction are arbitrary constant metadata required by the
            instructions. For example, the name of the block to be retrieved in `GetSBlock`.
        outputs : List[OUTPUT_RV_TYPE]
            The output random variables of the instruction,
            and the type of each element can be one of the following:
            - SBlockRV
            - LoopRV
            - ExprRV, atomic variables only, won't be constants or composite Expr
        """
        self.__init_handle_by_constructor__(
            _ffi_api.Instruction,  # type: ignore # pylint: disable=no-member
            kind,
            inputs,
            attrs,
            outputs,
        )
