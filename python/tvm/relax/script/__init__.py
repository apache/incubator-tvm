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
"""Public canonical TVMScript dialect namespace."""

from __future__ import annotations

import ast as _ast
import importlib as _importlib
import inspect as _inspect
import sys as _sys
from collections.abc import Callable as _Callable
from typing import TYPE_CHECKING
from typing import Any as _Any

if TYPE_CHECKING:
    from tvm.relax.base_py_module import BasePyModule
    from tvm_tirx.ir import IRModule
    from tvm_tirx.runtime import Device
    from tvm_tirx.target import Target


_initialized = False
_initializing = False


def _initialize() -> None:
    global _initialized, _initializing
    if _initialized or _initializing:
        return
    _initializing = True
    try:
        from tvm_tirx.script.parser import entry, register_namespace

        from . import ir_builder as builder

        globals().update(
            (name, value) for name, value in vars(builder).items() if not name.startswith("_")
        )
        globals()["__tvm_value_if__"] = builder.__tvm_value_if__
        globals()["function"] = entry.make_decorator(builder, namespace_path="relax.function")
        globals()["macro"] = entry.make_macro_decorator(
            builder, namespace_path="relax.macro", preserve_return=True
        )
        namespace = _sys.modules[__name__]
        register_namespace("relax", namespace)
        globals()["__all__"] = sorted(name for name in globals() if not name.startswith("_"))
        _initialized = True
    finally:
        _initializing = False


def __getattr__(name: str) -> _Any:
    if name == "ir_builder":
        return _importlib.import_module(__name__ + ".ir_builder")
    if name.startswith("_") and name != "__all__":
        raise AttributeError(name)
    _initialize()
    if name in globals():
        return globals()[name]
    return getattr(_importlib.import_module("tvm.relax.script.ir_builder"), name)


class _PyModuleFactory:
    """Instantiate an executable module without sharing its runtime registry."""

    def __init__(self, module: IRModule, original_class: type) -> None:
        self.ir_module = module
        self.original_class = original_class
        self.__name__ = original_class.__name__

    def __call__(self, device: Device | None = None, target: Target | None = None) -> BasePyModule:
        from tvm.relax.base_py_module import BasePyModule
        from tvm_tirx import cpu, ir

        source = self.ir_module
        instance_module = ir.IRModule(
            source.functions, attrs=source.attrs, global_infos=source.global_infos
        )
        instance = BasePyModule(instance_module, device or cpu(0), target)
        for name, function in source.__pyfuncs__.items():
            instance.add_python_function(name, function)
        return instance

    def __getattr__(self, name: str) -> _Any:
        return getattr(self.ir_module, name)


def py_module(
    module: type | None = None, **options: _Any
) -> IRModule | _PyModuleFactory | _Callable[[type], IRModule | _PyModuleFactory]:
    """Parse a class and attach executable Relax Python-function metadata.

    Parameters
    ----------
    module : type, optional
        Class containing registered script functions and ``I.pyfunc`` members.
        None, the default, returns a decorator. Python bodies are retained as
        their original callables and are not executed during construction.
    **options : _Any
        Options forwarded once to shared ``parse``, including
        ``check_well_formed`` (True by default) and ``track_loc`` (True).

    Returns
    -------
    IRModule or callable
        An IRModule with original Python callables in ``__pyfuncs__`` and
        corresponding Relax ExternFunc metadata. For a BasePyModule subclass,
        return a factory accepting optional device and target, creating a fresh
        runtime instance on each call. With no class, return the decorator.

    Raises
    ------
    TypeError
        If the decorated value is not a class, or an unsupported parser option
        is supplied.
    SyntaxError
        If the source violates a parser syntax restriction. Construction and
        enabled validation propagate their original exceptions unchanged.
    OSError
        If source inspection cannot recover a Python function's definition.

    Notes
    -----
    The decorator captures the original class-definition scope and passes it
    explicitly to shared parsing. It retains no Python frame or scope snapshot.
    ExternFunc metadata uses each original function's source coordinates when
    loc tracking is enabled. Runtime compilation/registration is deferred to
    BasePyModule construction; plain IRModule construction registers no runtime
    functions. Ordinary shared ``I.ir_module`` never performs these Relax steps.

    .. code:: python

        # Source
        @R.py_module
        class Module:
            @I.pyfunc
            def twice(value):
                return value * 2

        # Relax adaptation after one shared parse
        assert Module.__pyfuncs__["twice"](3) == 6
        assert isinstance(Module["twice"], relax.ExternFunc)
    """

    def apply(source_class: type) -> IRModule | _PyModuleFactory:
        from tvm import relax
        from tvm.relax.base_py_module import BasePyModule
        from tvm_tirx import ir
        from tvm_tirx.script.parser.entry import parse
        from tvm_tirx.script.parser.inspect_source import acquire_source, capture_definition_scope

        if not _inspect.isclass(source_class):
            raise TypeError(f"Expect a class, but got: {source_class}")
        frame = _inspect.currentframe().f_back
        try:
            if frame.f_code is py_module.__code__:
                frame = frame.f_back
            definition_scope = capture_definition_scope(frame)
            definition_source = (frame.f_code.co_filename, frame.f_lineno)
        finally:
            del frame
        result = parse(
            source_class,
            definition_scope=definition_scope,
            _definition_source=definition_source,
            **options,
        )
        result.__pyfuncs__ = getattr(result, "__pyfuncs__", {})
        for name, function in result.__pyfuncs__.items():
            tree, filename, _ = acquire_source(function)
            node = tree.body[-1]
            loc = ir.UnknownLoc()
            if options.get("track_loc", True):
                loc = ir.SourceLoc(
                    ir.SourceName(filename),
                    node.lineno,
                    node.col_offset + 1,
                    node.end_lineno,
                    node.end_col_offset + 1,
                )
            result[name] = relax.ExternFunc(name, loc=loc).with_attrs(
                {
                    "is_pyfunc": True,
                    "function_type": "python",
                    "python_function_name": name,
                    "python_source": _ast.unparse(node),
                    "python_packed_func": function,
                }
            )
        result.__name__ = source_class.__name__
        if issubclass(source_class, BasePyModule):
            return _PyModuleFactory(result, source_class)
        return result

    return apply(module) if module is not None else apply
