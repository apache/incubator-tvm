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
"""The build utils in python."""

from collections.abc import Callable

import tvm
import tvm_tirx
from tvm_tirx.codegen import Executable
from tvm_tirx.ir.module import IRModule
from tvm_tirx.target import Target
from tvm_tirx.tirx import Function


def _contains_relax(mod: Function | IRModule) -> bool:
    if isinstance(mod, Function):
        return False
    if isinstance(mod, IRModule):
        return any(isinstance(func, tvm.relax.Function) for _, func in mod.functions_items())

    raise ValueError(f"Function input must be a Function or IRModule, but got {type(mod)}")


def compile(  # pylint: disable=redefined-builtin
    mod: Function | IRModule,
    target: Target | None = None,
    *,
    relax_pipeline: tvm_tirx.ir.transform.Pass | Callable | str | None = "default",
    tir_pipeline: tvm_tirx.ir.transform.Pass | Callable | str | None = "default",
) -> Executable:
    """
    Compile an IRModule to a runtime executable.

    This function serves as a unified entry point for compiling both TIR and Relax modules.
    It automatically detects the module type and routes to the appropriate build function.

    Parameters
    ----------
    mod : Union[Function, IRModule]
        The input module to be compiled. Can be a Function or an IRModule containing
        TIR or Relax functions.
    target : Optional[Target]
        The target platform and its backend compilation configuration.
    relax_pipeline : Optional[Union[tvm_tirx.ir.transform.Pass, Callable, str]]
        The compilation pipeline to use for Relax functions.
        Only used if the module contains Relax functions.
    tir_pipeline : Optional[Union[tvm_tirx.ir.transform.Pass, Callable, str]]
        The compilation pipeline to use for TIR functions.

    Returns
    -------
    Executable
        A runtime executable that can be loaded and executed.
    """
    # TODO(tvm-team): combine two path into unified one
    if _contains_relax(mod):
        return tvm.relax.build(
            mod,
            target,
            relax_pipeline=relax_pipeline,
            tir_pipeline=tir_pipeline,
        )
    return tvm_tirx.tirx.compile(mod, target, pipeline=tir_pipeline)
