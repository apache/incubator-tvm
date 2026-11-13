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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name
"""S-TIR construction and validation over shared TIRx syntax."""

from tvm_tirx import ir as _ir
from tvm_tirx.script.ir_builder import base as _base
from tvm_tirx.tirx.script.ir_builder.parser_protocol import arg_ as _shared_arg
from tvm_tirx.tirx.script.ir_builder.parser_protocol import bind_ as _shared_bind

from . import _ffi_api
from .frame import BlockInitFrame, SBlockFrame
from .frame import SBlockFrame as _SBlockFrame
from .ir import _get_sblock_name_suffix

# --------------------------------------
# Function
# --------------------------------------


def function(is_private=False, persistent=False, *, private=None):
    """Create an S-TIR function frame."""
    return _ffi_api.Function(is_private if private is None else private, persistent, False)


def function_(*, private=False, persistent=False, decl=False, loc=None):
    """Enter an S-TIR declaration or definition frame."""
    native = _ffi_api.Function(private, persistent, decl)
    return _base.at_(loc, native)


def arg_(name, annotation, *, loc=None):
    """Preserve parameter annotations, including explicitly constructed layouts.

    Tensor annotations are evaluated inside the function frame, which already
    supplies the S-TIR default layout when the constructor omits it.
    """
    return _shared_arg(name, annotation, loc=loc)


def check_well_formed_(function):
    """Validate a completed S-TIR function."""
    from tvm.s_tir import analysis

    try:
        analysis.check_well_formed(_ir.IRModule.from_expr(function))
    except Exception as error:
        raise ValueError(
            "Program is not well-formed. If this is deliberate, set "
            f"check_well_formed=False in the top-level decorator.\n{error}"
        ) from error


# --------------------------------------
# Bindings
# --------------------------------------


def bind_(value=_base.MISSING, **options):
    """Block frames cannot introduce an as-target value."""
    if options.get("frame_value") and isinstance(value, _SBlockFrame):
        raise TypeError("A block does not introduce an as-target value")
    return _shared_bind(value, **options)


# --------------------------------------
# Special
# --------------------------------------
# S-TIR inherits syntax marker and declaration policies from TIRx.


# --------------------------------------
# Control
# --------------------------------------


def sblock(name: str = "", no_realize: bool = False, exec_scope: str = "") -> SBlockFrame:
    """The sblock declaration statement.

    Parameters
    ----------
    name : str
        The name of the sblock.

    no_realize : bool
        The flag whether to construct SBlockRealize or SBlock.

    exec_scope : str
        The execution scope of the block.

    Returns
    -------
    res : SBlockFrame
        The SBlockFrame.
    """
    block_suffix = _get_sblock_name_suffix()
    if block_suffix and name:
        name = name + block_suffix
    return _ffi_api.Block(name, no_realize, exec_scope)  # type: ignore[attr-defined] # pylint: disable=no-member


def init() -> BlockInitFrame:
    """The block initialization statement.

    Returns
    -------
    res : BlockInitFrame
        The BlockInitFrame.
    """
    return _ffi_api.Init()  # type: ignore[attr-defined] # pylint: disable=no-member


# --------------------------------------
# Operators
# --------------------------------------
# Operator hooks are inherited from TIRx.

__all__ = ["arg_", "bind_", "check_well_formed_", "function", "function_", "init", "sblock"]
