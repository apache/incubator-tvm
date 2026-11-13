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
# pylint: disable=invalid-name, no-member
"""VM build logics"""

import tvm_ffi

import tvm_tirx
from tvm import relax
from tvm_tirx.codegen import Executable
from tvm_tirx.ir.module import IRModule
from tvm_tirx.tirx.function import Function

from . import _ffi_api


class VMExecutable(Executable):
    """The virtual machine executable object emitted by the VM compiler or the ExecBuilder."""

    def __init__(self, mod: tvm_ffi.Module):
        super().__init__(mod)
        self._stats = self.mod["stats"]
        self._as_text = self.mod["as_text"]
        self._as_python = self.mod["as_python"]

    def stats(self) -> str:
        """print the detailed statistics of the executable."""
        return self._stats()

    def as_text(self) -> str:
        """print the instructions as text format."""
        return self._as_text()

    def as_python(self) -> str:
        """print the instructions as python program."""
        return self._as_python()


def _vmcodegen(
    builder: "relax.ExecBuilder",
    mod: tvm_tirx.IRModule,
) -> tvm_tirx.IRModule:
    """Running VM codegen.

    Parameters
    ----------
    builder: relax.ExecBuilder
        ExecBuilder to collect the vm executable.

    mod: IRModule
        The input IRModule to be built.

    Return
    ------
    leftover: IRModule
        Left over IRModule that may contain extra functions.
    """

    return _ffi_api.VMCodeGen(builder, mod)  # type: ignore


def _auto_attach_system_lib_prefix(
    tir_mod: tvm_tirx.IRModule,
    target: tvm_tirx.target.Target | None = None,
    system_lib: bool | None = None,
):
    """Automatically detect system lib req and attach prefix attr"""
    if target is not None:
        host = target if target.host is None else target.host
        if system_lib is None:
            system_lib = False
            if "wasm" in host.attrs.get("mtriple", ""):
                system_lib = True

    if system_lib:
        if tir_mod.get_attr("system_lib_prefix") is None:
            return tir_mod.with_attr("system_lib_prefix", "")
    return tir_mod


def _is_device_module(mod: tvm_ffi.Module) -> bool:
    return mod.kind in ["cuda", "opencl", "metal", "hip", "vulkan", "webgpu"]


def _vmlink(
    builder: "relax.ExecBuilder",
    target: str | tvm_tirx.target.Target | None,
    tir_mod: tvm_tirx.IRModule | None = None,
    tir_pipeline: str | tvm_tirx.ir.transform.Pass | None = "default",
    ext_libs: list[tvm_ffi.Module] | None = None,
    params: dict[str, list] | None = None,
    *,
    system_lib: bool | None = None,
):
    """
    Internal codegen function to make executable.

    This function is only used for unit-testing purpoes.

    Use build instead.

    Parameters
    ----------
    builder: relax.ExecBuilder
        Builder used to collect executables.

    target : Optional[Union[str, tvm_tirx.target.Target]]
        A build target which can have optional host side compilation target.
        If the target is not specified, the target in the vdevice list will be used.
        For multi-target compilation, the vdevice should be annotated.

    tir_mod: IRModule
        The input TIR IRModule to be linked together.

    ext_libs:  List[tvm_ffi.Module]
        List of compiled external modules.

    params: Optional[Dict[str, list]]
        Extra parameter mappings.

    Returns
    -------
    ex: tvm.relax.Executable
        An executable that can be loaded by virtual machine.
    """
    if isinstance(target, str):
        target = tvm_tirx.target.Target(target)
    if params is None:
        params = {}
    if ext_libs is None:
        ext_libs = []
    lib = None
    relax_ext_libs = []
    tir_ext_libs = []
    if tir_mod is not None and len(tir_mod.get_global_vars()) > 0:
        tir_mod = _auto_attach_system_lib_prefix(tir_mod, target, system_lib)
        lib = tvm_tirx.tirx.compile(tir_mod, target=target, pipeline=tir_pipeline).mod
    for ext_mod in ext_libs:
        if _is_device_module(ext_mod):
            tir_ext_libs.append(ext_mod)
        else:
            relax_ext_libs.append(ext_mod)
    if lib is not None:
        for mod in tir_ext_libs:
            lib.import_module(mod)
    elif len(tir_ext_libs) > 0:
        print("Warning: No TIR module is found, but external modules for TIR are provided.")
    lib = _ffi_api.VMLink(builder, target, lib, relax_ext_libs, params)  # type: ignore
    return VMExecutable(lib)


def build(
    mod: tvm_tirx.IRModule,
    target: str | tvm_tirx.target.Target | None = None,
    params: dict[str, list] | None = None,
    relax_pipeline: None | str | tvm_tirx.ir.transform.Pass = "default",
    tir_pipeline: None | str | tvm_tirx.ir.transform.Pass = "default",
    *,
    system_lib: bool | None = None,
) -> Executable:
    """
    Build an IRModule to VM executable.

    Parameters
    ----------
    mod: IRModule
        The input IRModule to be built.

    target : Optional[Union[str, tvm_tirx.target.Target]]
        A build target which can have optional host side compilation target.

        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        to setup the dimensions and parameters correctly.
        host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a c backend is used.

    params: Optional[Dict[str, list]]
        Parameters for the input IRModule that will be bound.

    relax_pipeline : str = "default"
        The Relax compilation pipeline to use.

    tir_pipelinie : str = "default"
        The TIR compilation pipeline to use.

    system_lib: Optional[bool]
        Whether to build system lib that is being packed statically and
        auto registers generated functions to the system.
        By default auto detects based on the target.

    Returns
    -------
    ex: tvm.relax.Executable
        An executable that can be loaded by virtual machine.

    Example
    -------

    .. code-block:: python

        class InputModule:
            @R.function
            def foo(x: Tensor((3, 4), "float32"), y: Tensor((3, 4), "float32")):
                z = R.add(x, y)
                return z

        mod = InputModule
        target = tvm_tirx.target.Target("llvm", host="llvm")
        ex = tvm.compile(mod, target)
    """

    def _extract_attrs(mod: tvm_tirx.IRModule):
        attrs = dict(mod.attrs) if mod.attrs else {}
        ext_libs = attrs.get("external_mods", [])
        constants = attrs.get("const_name_to_constant", {})
        return ext_libs, constants

    from tvm_tirx.backend.config import prepare_target

    target = prepare_target(target, mod)
    if isinstance(target, str):
        target = tvm_tirx.target.Target(target)
    if not params:
        params = {}

    if relax_pipeline is not None:
        if isinstance(relax_pipeline, str):
            # For GPU targets, prefer the target-specific pipeline which
            # includes DLight scheduling. Without it, TIR functions generated
            # from ops like Clip/ReLU6 lack thread bindings and fail
            # VerifyMemory. CPU targets continue to use the generic pipeline
            # since the CPU-specific pipeline applies fusion passes that can
            # incorrectly remove call_pure_packed calls whose results are
            # unused but whose side effects are relied upon.
            _is_gpu = target is not None and "gpu" in target.keys
            if relax_pipeline == "default" and _is_gpu:
                try:
                    relax_pipeline = relax.get_default_pipeline(target)
                except (ValueError, AttributeError):
                    relax_pipeline = relax.get_pipeline(
                        relax_pipeline,
                        **({"target": target} if relax_pipeline in ("default", "zero") else {}),
                    )
            else:
                relax_pipeline = relax.get_pipeline(
                    relax_pipeline,
                    **({"target": target} if relax_pipeline in ("default", "zero") else {}),
                )
        mod = relax_pipeline(mod)

    ext_libs, constants = _extract_attrs(mod)
    params.update(dict(constants))
    builder = relax.ExecBuilder()
    mod = _vmcodegen(builder, mod)
    return _vmlink(
        builder=builder,
        target=target,
        tir_mod=_filter_tir(mod),
        tir_pipeline=tir_pipeline,
        ext_libs=ext_libs,
        params=params,
        system_lib=system_lib,
    )


def _filter_tir(mod: tvm_tirx.IRModule) -> tvm_tirx.IRModule | None:
    tir_mod = {gvar: func for gvar, func in mod.functions.items() if isinstance(func, Function)}

    if tir_mod:
        return IRModule(tir_mod, attrs=mod.attrs)
    else:
        return None
