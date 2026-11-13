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
"""Relax implementation of the shared source-to-builder protocol.

Hooks normalize syntax operands and construct native builder frames and IR directly.
For example, generated ``X.if_(condition)`` creates the native conditional frame;
``X.then_()`` and ``X.else_()`` enter its branches. See the corresponding shared
``tvm_tirx.script.ir_builder.parser_protocol`` hooks for operand and loc contracts.
"""

from __future__ import annotations

import builtins as _python
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

import tvm_ffi
import tvm_ffi as _ffi

from tvm import relax as _relax
from tvm.relax import Call, Expr, Var, VarBinding
from tvm.relax.type import Type
from tvm.relax.utils import gen_call_tir_inputs
from tvm_tirx import ir as _ir
from tvm_tirx.ir.location import UnknownLoc
from tvm_tirx.script.ir_builder import base as _base
from tvm_tirx.script.ir_builder.base import IRBuilder as _IRBuilder
from tvm_tirx.script.ir_builder.parser_protocol import decl_function

from . import _ffi_api
from . import frame as _frame
from . import ir as _native
from . import op as _op
from .ir import resolve_global_info_

_Loc = _base.LocationEntry | _ir.Location | None


# --------------------------------------
# Function
# --------------------------------------


def function(is_pure: bool = True, is_private: bool = False) -> _frame.FunctionFrame:
    """Start a function frame.

    Parameters
    ----------
    is_pure: bool
        Whether the function is annotated as pure.

    is_private : bool
        Whether the function is annotated as private.

    Returns
    -------
    frame: FunctionFrame
        The constructed function frame.
    """
    return function_(pure=is_pure, private=is_private)


def function_(
    pure: bool = True,
    private: bool = False,
    *,
    decl: bool = False,
    local: bool = False,
    reference: _ir.Var | None = None,
    loc: _Loc = None,
) -> _frame.FunctionFrame:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.function_`.

    Public pure/private options pass to native purity and visibility controls.
    Local function bodies require their previously declared reference.
    """
    if decl:
        return _base.at_(loc, _ffi_api.Function(pure, private, True, local))
    if local:
        if reference is None:
            raise ValueError("A local function requires its declared reference")
        return _base.at_(loc, _ffi_api.LocalFunction(pure, reference))
    return _base.at_(loc, _ffi_api.Function(pure, private, False, False))


def arg_(name: str, ty: Any, *, loc: _Loc = None) -> _ir.Var:
    """Declare a Relax parameter using the shared argument-hook contract.

    Parameters
    ----------
    name : str
        Source parameter name.
    ty : Type, Var, or callable
        Parameter annotation; corresponds to ``annotation`` in the shared
        :func:`tvm_tirx.script.ir_builder.parser_protocol.arg_` contract. Primitive
        annotations create a fresh parameter; an existing variable retains
        its identity.
    loc : LocationEntry, Location or None, optional
        Source location attached to the constructed IR; None leaves it unspecified.

    Returns
    -------
    result : Var
        The parameter registered in the current function frame.
    """
    if not isinstance(ty, _ir.Var):
        ty = _native._type(ty)
    if isinstance(ty, _ir.Var):
        return _ffi_api.ArgVar(name, ty)
    return _base.at_(loc, _ffi_api.Arg(name, _native._type(ty)))


def func_name_(name: str) -> None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.func_name_`."""
    return _ffi_api.FuncName(name)


def func_attr(attrs: dict[str, tvm_ffi.Object]) -> None:
    """Specify the attrs of the last function frame.

    Parameters
    ----------
    attrs: Dict[str, Object]
        The function attrs.
    """
    return _ffi_api.FuncAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def func_ret_type_(annotation: Any, *, loc: _Loc = None) -> None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.func_ret_type_`."""
    return _ffi_api.FuncRetType(_native._type(_base._return_annotation(annotation)))


def check_well_formed_(function: _relax.Function) -> None:
    """Validate a completed Relax function.

    Parameters
    ----------
    function : tvm.relax.Function
        Completed function to validate, without changing its IR.

    Raises
    ------
    ValueError
        If the function fails language variant validation.

    """
    from tvm import s_tir

    message = (
        "Program is not well-formed. If this is deliberate, set "
        "check_well_formed=False in the top-level decorator."
    )
    try:
        _relax.analysis.check_well_formed(function)
        s_tir.analysis.check_well_formed(_ir.IRModule.from_expr(function))
    except Exception as error:
        raise ValueError(f"{message}\n{error}") from error


# --------------------------------------
# Bindings
# --------------------------------------


def call_global_var_(function: _ir.GlobalVar, args: Sequence[Any]) -> _ir.Expr:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.call_global_var_`."""
    return _relax.Call(function, [_relax.utils.convert_to_expr(value) for value in args])


def dataflow(*, loc=None):
    """Create a dataflow context with explicit finalized exports.

    Parameters
    ----------
    loc : LocationEntry, Location or None, optional
        Source location attached to the constructed IR.

    Returns
    -------
    res : frame.BindingBlockFrame
        The constructed frame, retaining source metadata.
    """
    return _base.at_(loc, _ffi_api.Dataflow())


def output(*vars: tuple[Var]) -> None:
    """Expose the dataflow block output variables as global ones.

    Parameters
    ----------
    vars: Tuple[Var]
        The output variables of a dataflow block.
    """
    return _ffi_api.DataflowBlockOutput(vars)  # type: ignore[attr-defined] # pylint: disable=no-member


def seq_expr() -> _frame.SeqExprFrame:  # pylint: disable=invalid-name
    """Create a SeqExpr frame.

    Returns
    -------
    res : _frame.SeqExprFrame
        The result SeqExprFrame
    """
    return _ffi_api.SeqExpr()  # type: ignore[attr-defined] # pylint: disable=no-member


def bind_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    loc: _Loc = None,
    value_loc: _Loc = None,
    name_loc: _Loc = None,
    frame_value: bool = False,
) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.bind_`.

    Relax values emit a binding or MatchCast; value_loc belongs to the actual
    RHS and loc to the binding target. Metadata passes through unchanged.
    """
    if isinstance(loc, _base.LocationEntry):
        loc = loc.loc
    name_loc = loc if name_loc is None else name_loc
    if isinstance(name_loc, _base.LocationEntry):
        name_loc = name_loc.loc
    if frame_value:
        if isinstance(value, _python.list | _python.tuple | tvm_ffi.Array):
            for index, item in enumerate(value):
                bind_(
                    item,
                    name=None if name is None else f"{name}_{index}",
                    loc=loc,
                    name_loc=name_loc,
                    frame_value=True,
                )
        elif isinstance(value, _ir.Var):
            if name is not None:
                _IRBuilder.name(name, value)
            _base.at_(name_loc if name_loc is not None else loc, value)
        return value
    if value is _base.MISSING:
        raise ValueError("Relax bindings require an initializer")
    ty = None if ty is None else _native._type(ty)
    if isinstance(value, _base.AlreadyEmitted):
        return _base.at_(value_loc, value)
    value = _native._value(value, ty)
    if isinstance(value, _relax.MatchCast):
        _base.at_(value_loc, value.value)
        if ty is not None and not _ffi.structural_equal(ty, value.ty):
            raise TypeError("The binding annotation differs from the match-cast type")
        result = _ffi_api.EmitMatchCastWithLoc(
            value.value,
            value.ty,
            UnknownLoc() if name_loc is None else name_loc,
            UnknownLoc() if loc is None else loc,
        )
    elif isinstance(value, _relax.Expr):
        _base.at_(value_loc, value)
        result = _ffi_api.EmitWithLoc(
            value,
            ty,
            UnknownLoc() if name_loc is None else name_loc,
            UnknownLoc() if loc is None else loc,
        )
    else:
        return value
    if name is not None:
        _IRBuilder.name(name, result)
    return _base.at_(name_loc if name_loc is not None else loc, result)


def decl_mutable_cell_(
    value: Any = _base.MISSING,
    *,
    ty: Any = None,
    name: str | None = None,
    loc: _Loc = None,
    name_loc: _Loc = None,
) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.decl_mutable_cell_`.

    Relax has immutable bindings and rejects mutable storage declarations.
    """
    raise TypeError("Relax does not support mutable local storage")


def set_mutable_cell_(target: Any, value: Any, *, loc: _Loc = None) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.set_mutable_cell_`.

    Relax has immutable bindings and rejects mutable storage updates.
    """
    raise TypeError("Relax does not support mutable local storage")


def unpack_(value: Any) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.unpack_`."""
    if isinstance(value, _relax.Tuple):
        return _python.tuple(value.fields)
    if isinstance(value, _relax.Expr) and isinstance(value.ty, _ir.TupleType):
        return _python.tuple(_relax.TupleGetItem(value, i) for i in range(len(value.ty.fields)))
    return value


def emit(value: Expr, annotate_ty: Type | None = None) -> Var:
    """Emit a binding to the last binding block frame.
    Parameters
    ----------
    value: Expr
        The right side value of the bindings to be emitted.

    annotate_ty: Optional[Type]
        The optional type annotation for the emitted value.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.Emit(value, annotate_ty)  # type: ignore[attr-defined] # pylint: disable=no-member


def emit_te(func: Callable, *args: Any, **kwargs: Any) -> Call:
    """Emit a call node according to the te function.
    This function converts arguments from relax expression to te tensor,
    The callback func should return a te tensor or a list of te tensors.

    Parameters
    ----------
    func : Callable
        A function that returns a te tensor or a list of te tensors.

    args : Any, optional
        arguments passed to the function.

    kwargs : Any, optional
        The keyword arguments passed to the function.
        Note that the following keyword args are reserved:

            - 'function_name_hint' for passing name hint to the Function
                that gets generated.
            - 'function_attrs' is reserved for passing func attributes to
                be added to the Function that gets created.

    Returns
    -------
    call : Call
        A newly created call that calls into a tirx function.
    """
    function_name_hint = kwargs.pop("function_name_hint", None)
    tir_func, call_args, out_ty = gen_call_tir_inputs(func, *args, **kwargs)
    if not function_name_hint:
        function_name_hint = func.__name__
    gvar = decl_function(function_name_hint, tir_func)  # type: ignore
    return _op.call_tir(gvar, call_args, ty_args=[out_ty])


def emit_match_cast(value: Expr, ty: Type) -> Var:
    """Emit a match_cast binding to the last binding block frame.
    Parameters
    ----------
    value: Expr
        The value of the MatchCast to be emitted.
    ty: Type
        The ty of the MatchCast to be emitted.

    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.EmitMatchCast(value, ty)  # type: ignore


def emit_var_binding(value: VarBinding) -> Var:
    """Emit a binding to the last binding block frame.
    Parameters
    ----------
    value: VarBinding
        The binding to be emitted.
    Returns
    -------
    var: Var
        The left side var of the emitted binding.
    """
    return _ffi_api.EmitVarBinding(value)  # type: ignore


def emit_(value: Any, *, loc: _Loc = None) -> None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.emit_`.

    Only void expressions may be discarded. Receipts and None emit nothing;
    non-void expressions raise ValueError and unsupported host values raise TypeError.
    """
    if isinstance(value, _base.AlreadyEmitted):
        _base.at_(loc, value)
        return None
    if value is None:
        return
    if isinstance(value, _python.tuple) and not value:
        value = _relax.Tuple([])
    if not isinstance(value, _relax.Expr):
        raise TypeError(f"Unsupported expression statement value: {type(value).__name__}")
    result = bind_(_base.at_(loc, value), name="_", loc=loc)
    if not isinstance(result.ty, _ir.TupleType) or len(result.ty.fields) != 0:
        raise ValueError(
            "Non-void expressions must be bound to a variable; "
            f"expression of type {result.ty} was used as a statement"
        )


def setitem_(target: Any, key: Any, value: Any, *, loc: _Loc = None) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.setitem_`."""
    raise TypeError("Relax does not support indexed assignment")


def setattr_(target: Any, name: str, value: Any, *, loc: _Loc = None) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.setattr_`."""
    raise TypeError("Relax does not support attribute assignment")


# --------------------------------------
# Special protocol remarks
# --------------------------------------
# Syntax policies use canonical registered namespace paths. Entry factories own
# function/helper classification; source-call contexts preserve provenance while
# bind_ owns binding result attribution.


# --------------------------------------
# Control
# --------------------------------------


def if_(condition: Any, *, loc: _Loc = None) -> _frame.IfFrame:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.if_`."""
    if not isinstance(condition, _relax.Expr):
        condition = _relax.prim_value(condition)
    return _base.at_(loc, _ffi_api.If(condition))


def then_(*, loc: _Loc = None) -> _frame.ThenFrame:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.then_`."""
    return _base.at_(loc, _ffi_api.Then())


def else_(*, loc: _Loc = None) -> _frame.ElseFrame:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.else_`."""
    return _base.at_(loc, _ffi_api.Else())


def for_(iterable: Any, *, names: str | Sequence[str] | None = None, loc: _Loc = None) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.for_`.

    Relax rejects source loops; use supported functional control flow.
    """
    raise TypeError("Relax does not support imperative for loops")


def while_(condition: Any, *, loc: _Loc = None) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.while_`."""
    raise TypeError("Relax does not support imperative while loops")


def range_(*args: Any, annotations: dict[str, Any] | None = None) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.range_`."""
    raise TypeError("Relax does not support imperative for loops")


def break_(*, loc: _Loc = None) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.break_`.

    Relax rejects loop-control statements.
    """
    raise TypeError("Relax does not support break")


def continue_(*, loc: _Loc = None) -> NoReturn:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.continue_`.

    Relax rejects loop-control statements.
    """
    raise TypeError("Relax does not support continue")


def func_ret_value(value: Expr) -> None:
    """Specify the return value of the last function frame.

    Parameters
    ----------
    value: Expr
        The function return value.
    """
    return _ffi_api.FuncRetValue(value)  # type: ignore[attr-defined] # pylint: disable=no-member


def return_(value: Any = None, *, loc: _Loc = None) -> None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.return_`."""
    if value is None:
        value = _relax.Tuple([])
    # Normalization may emit bindings, but an existing result keeps its own loc.
    _base.with_at_group_(loc, lambda: _ffi_api.FuncRetValue(_native._value(value)))


def assert_(
    condition: Any,
    message: str | tuple[str, Sequence[Any]] | Sequence[Any] = "",
    *,
    loc: _Loc = None,
) -> None:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.assert_`."""
    if not isinstance(message, _python.str):
        raise TypeError("An assertion message must be construction-time text")
    emit_(_base.at_(loc, _op.assert_op(condition, message)), loc=loc)


# --------------------------------------
# Operators
# --------------------------------------

if_then_else_ = _op.if_then_else_
and_ = _op.and_
or_ = _op.or_
not_ = _op.not_
lt_ = _op.lt_
le_ = _op.le_
gt_ = _op.gt_
ge_ = _op.ge_
eq_ = _op.eq_
ne_ = _op.ne_


supports_mutable_declarations = False
__tvm_value_if__ = True

__all__ = [
    "and_",
    "arg_",
    "assert_",
    "bind_",
    "break_",
    "call_global_var_",
    "check_well_formed_",
    "continue_",
    "dataflow",
    "decl_mutable_cell_",
    "else_",
    "emit",
    "emit_",
    "emit_match_cast",
    "emit_te",
    "emit_var_binding",
    "eq_",
    "for_",
    "func_attr",
    "func_name_",
    "func_ret_type_",
    "func_ret_value",
    "function",
    "function_",
    "ge_",
    "gt_",
    "if_",
    "if_then_else_",
    "le_",
    "lt_",
    "ne_",
    "not_",
    "or_",
    "output",
    "range_",
    "resolve_global_info_",
    "return_",
    "seq_expr",
    "set_mutable_cell_",
    "setattr_",
    "setitem_",
    "then_",
    "unpack_",
    "while_",
]
