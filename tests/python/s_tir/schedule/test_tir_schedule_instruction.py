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
# pylint: disable=missing-function-docstring,missing-module-docstring
# mypy: ignore-errors
# ruff: noqa: F401
import sys

import pytest

import tvm
import tvm_tirx.testing
from tvm.s_tir.schedule import Instruction, InstructionKind, LoopRV, SBlockRV


def test_inst_kind_get():
    kind = InstructionKind.get("EnterPostproc")
    assert not kind.is_pure
    assert kind.name == "EnterPostproc"


def test_inst_construct_1():
    block = SBlockRV()
    loop0 = LoopRV()
    loop1 = LoopRV()
    inst = Instruction(
        kind=InstructionKind.get("GetLoops"),
        inputs=[block],
        attrs=[],
        outputs=[loop0, loop1],
    )
    assert str(inst) == "_, _ = sch.get_loops(block=_)"
    assert len(inst.inputs) == 1
    assert len(inst.attrs) == 0
    assert len(inst.outputs) == 2
    assert inst.kind.same_as(InstructionKind.get("GetLoops"))
    assert inst.inputs[0].same_as(block)
    assert inst.outputs[0].same_as(loop0)
    assert inst.outputs[1].same_as(loop1)


def test_inst_construct_2():
    block = SBlockRV()
    inst = Instruction(
        kind=InstructionKind.get("ComputeInline"),
        inputs=[block],
        attrs=[],
        outputs=[],
    )
    assert str(inst) == "sch.compute_inline(block=_)"
    assert len(inst.inputs) == 1
    assert len(inst.attrs) == 0
    assert len(inst.outputs) == 0
    assert inst.kind.same_as(InstructionKind.get("ComputeInline"))
    assert inst.inputs[0].same_as(block)


if __name__ == "__main__":
    tvm_tirx.testing.main()
