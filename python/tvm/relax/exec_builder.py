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
# pylint: disable=invalid-name
# ruff: noqa: RUF012
"""A builder to build Relax VM executable."""

from enum import IntEnum

import tvm_ffi
from tvm_ffi import Shape

import tvm_tirx

from . import _ffi_api
from .vm_build import VMExecutable


class SpecialReg(IntEnum):
    """Magic numbers that represent special registers in vm."""

    VOID_ARG = (1 << 54) + 0
    VM_STATE = (1 << 54) + 1


class VMFuncKind(IntEnum):
    """VM function kind code."""

    PACKED_FUNC = 0
    VM_FUNC = 1


class VMFuncScope:
    """An object corresponds to each VM function, working as a context manager."""

    stack: list["VMFuncScope"] = []

    def __init__(self, exit_callback):
        self.exit_callback = exit_callback

    def __enter__(self):
        VMFuncScope.stack.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        VMFuncScope.stack.pop()
        self.exit_callback()


@tvm_ffi.register_object("relax.ExecBuilder")
class ExecBuilder(tvm_ffi.core.Object):
    """A builder to emit instructions and build executable for the virtual machine."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ExecBuilderCreate)  # type: ignore

    def r(self, idx: int) -> int:
        """set instruction's argument as a register."""
        return _ffi_api.ExecBuilderR(self, idx)  # type: ignore

    def imm(self, value: int) -> int:
        """set instruction's argument as an immediate."""
        return _ffi_api.ExecBuilderImm(self, value)  # type: ignore

    def c(self, idx: int) -> int:
        """set instruction's argument as a constant."""
        return _ffi_api.ExecBuilderC(self, idx)  # type: ignore

    def f(self, name: str) -> int:
        """set instruction's argument as a function."""
        return _ffi_api.ExecBuilderF(self, name)  # type: ignore

    def void_arg(self) -> int:
        return self.r(SpecialReg.VOID_ARG)

    def vm_state(self) -> int:
        return self.r(SpecialReg.VM_STATE)

    def declare_function(self, func_name: str, kind: VMFuncKind = VMFuncKind.PACKED_FUNC) -> None:
        """Declare a function"""
        _ffi_api.ExecBuilderDeclareFunction(self, func_name, kind)  # type: ignore

    def function(
        self, func_name: str, num_inputs: int | None = 0, param_names: list[str] | None = None
    ) -> VMFuncScope:
        """annotate a VM function."""
        _ffi_api.ExecBuilderEmitFunction(self, func_name, num_inputs, param_names)  # type: ignore
        return VMFuncScope(lambda: _ffi_api.ExecBuilderEndFunction(self, func_name))  # type: ignore

    def _check_scope(self) -> None:
        if len(VMFuncScope.stack) == 0:
            raise ValueError("emit should happen in a function scope")

    def convert_constant(self, const: object) -> int:
        return _ffi_api.ExecBuilderConvertConstant(self, const)  # type: ignore

    def emit_call(
        self,
        name: str,
        args: list[tvm_tirx.runtime.Tensor | tvm_ffi.dtype] | None = None,
        dst: int | None = None,
    ) -> None:
        """emit a call instruction which calls a packed function."""
        self._check_scope()
        if dst is None:
            dst = SpecialReg.VOID_ARG
        args_ = []
        if args is not None:
            for arg in args:
                if isinstance(arg, tuple):
                    shape_tuple = Shape(arg)
                    new_arg = self.convert_constant(shape_tuple)
                    args_.append(new_arg)
                elif isinstance(arg, tvm_tirx.runtime.Tensor | tvm_ffi.dtype | Shape):
                    new_arg = self.convert_constant(arg)
                    args_.append(new_arg)
                else:
                    args_.append(arg)
        _ffi_api.ExecBuilderEmitCall(self, name, args_, dst)  # type: ignore

    def emit_ret(self, result: int) -> None:
        """emit a return instruction"""
        self._check_scope()
        _ffi_api.ExecBuilderEmitRet(self, result)  # type: ignore

    def emit_goto(self, pc_offset):
        """emit a goto instruction"""
        self._check_scope()
        _ffi_api.ExecBuilderEmitGoto(self, pc_offset)  # type: ignore

    def emit_if(self, cond, false_offset):
        """emit an if instruction"""
        self._check_scope()
        _ffi_api.ExecBuilderEmitIf(self, cond, false_offset)  # type: ignore

    def get(self) -> VMExecutable:
        """return the executable"""
        return VMExecutable(_ffi_api.ExecBuilderGet(self))  # type: ignore
