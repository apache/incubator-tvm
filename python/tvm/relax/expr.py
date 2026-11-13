# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ruff: noqa: F401
"""The expression nodes of Relax."""

import typing
from collections.abc import Callable, Mapping
from numbers import Integral, Number, Real
from typing import Any, Optional, Union

import numpy as _np  # type: ignore
import tvm_ffi
from tvm_ffi.core import String

import tvm
import tvm.relax
import tvm_tirx.ir
import tvm_tirx.runtime
from tvm_tirx.ir import BaseFunc, Location, Node, Scriptable
from tvm_tirx.ir.expr import _CallableExprWithOp
from tvm_tirx.ir.location import UNKNOWN_LOC, UnknownLoc

from . import _ffi_api

# It is a workaround for mypy: https://github.com/python/mypy/issues/7866#issuecomment-549454370
# This feature is not supported until python 3.10:
# https://docs.python.org/3.10/whatsnew/3.10.html#pep-613-typealias
Expr = tvm_tirx.ir.Expr
Type = tvm_tirx.ir.Type  # pylint: disable=invalid-name
GlobalVar = tvm_tirx.ir.GlobalVar


def prim_value(value: Expr | int | float, dtype: str | None = None) -> Expr:
    """Convert a Python scalar or primitive expression to ``Expr``.

    Parameters
    ----------
    value : Expr | int | float
        The value to convert.

    dtype : Optional[str]
        The dtype to use when converting Python numeric values.

    Returns
    -------
    result : Expr
        The converted primitive expression. Existing primitive expression inputs are
        returned unchanged, even when ``dtype`` is supplied.

    Notes
    -----
    Existing primitive expressions can be used directly in TVMScript. Python
    scalars are context dependent: an unannotated Relax binding or return
    converts them to tensor constants, while this helper constructs primitive
    expressions. ``dtype`` selects the type of Python numeric inputs; omitting
    it uses bool, int64, or float64. The script alias ``R.prim_value`` retains
    these programmatic conversion semantics.
    """
    if tvm_tirx.ir.is_prim_expr(value):
        return value
    if isinstance(value, bool | _np.bool_):
        return tvm_tirx.ir.IntImm(dtype or "bool", int(value))
    if isinstance(value, Integral):
        return tvm_tirx.ir.IntImm(dtype or "int64", int(value))
    if isinstance(value, Real):
        return tvm_tirx.ir.FloatImm(dtype or "float64", float(value))
    tvm_value = tvm_ffi.convert(value)
    if tvm_tirx.ir.is_prim_expr(tvm_value):
        return tvm_value
    raise TypeError(f"Cannot convert {value} with type {type(value)} to `Expr`")


def _relax_type_is_base_of(self: Type, derived: Type) -> bool:
    """Check if this Relax type is a base of another Relax type."""

    return _ffi_api.TypeIsBaseOf(self, derived)  # type: ignore


Type.is_base_of = _relax_type_is_base_of  # type: ignore[attr-defined]


# will be registered afterwards in python/tvm/relax/op/init.py


def _is_tensor_or_missing_type(ty: Type) -> bool:
    return isinstance(ty, tvm.relax.TensorType | tvm_tirx.ir.MissingType)


def _binary_op_helper(lhs: Expr, rhs: Expr, op: Callable):
    if not isinstance(lhs, Expr):  # type: ignore
        raise ValueError("lhs must be Expr")
    if not _is_tensor_or_missing_type(lhs.ty):
        return NotImplemented
    if isinstance(rhs, Expr):  # type: ignore
        return op(lhs, rhs)
    elif isinstance(rhs, Number):
        raise TypeError(f"Please convert {rhs} with `const` first")
    else:
        raise TypeError(f"type {type(rhs)} not supported")


def _binary_rhs_helper(rhs: Expr):
    if isinstance(rhs, Number):
        raise TypeError(f"Please convert {rhs} with `const` first")
    raise TypeError(f"type {type(rhs)} not supported")


@tvm_ffi.register_object("relax.expr.IfExpr")
class IfExpr(_CallableExprWithOp):
    """A conditional expression in Relax.

    Parameters
    ----------
    cond: Expr
        The condition.

    true_branch: Expr
        The expression evaluated when condition is true.

    false_branch: Expr
        The expression evaluated when condition is false.

    loc: Location
        Location that points to original source code
    """

    cond: Expr
    true_branch: Expr
    false_branch: Expr
    loc: Location

    def __bool__(self) -> bool:
        return True

    def __init__(
        self, cond: Expr, true_branch: Expr, false_branch: Expr, loc: Location = UNKNOWN_LOC
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.IfExpr,
            cond,
            true_branch,
            false_branch,
            loc,  # type: ignore
        )


# Compatibility aliases. Tuple expressions are owned by the common IR.
Tuple = tvm_tirx.ir.Tuple
TupleGetItem = tvm_tirx.ir.TupleGetItem


@tvm_ffi.register_object("relax.expr.ShapeExpr")
class ShapeExpr(_CallableExprWithOp):
    """A shape expression which allows users to construct a shape containing Expr.

    Parameters
    ----------
    values: Union[List[Expr], typing.Tuple[Expr, ...], tvm_ffi.Array]
        The values of the shape expression.

    loc: Location
        Location that points to original source code
    """

    values: list[Expr]
    loc: Location

    def __init__(
        self,
        values: list[Expr] | tuple[Expr, ...] | tvm_ffi.Array,
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        self.__init_handle_by_constructor__(_ffi_api.ShapeExpr, values, loc)  # type: ignore

    def __getitem__(self, index):
        if index >= len(self) or index < -len(self):
            raise IndexError("ShapeExpr index out of range")
        return self.values[index]

    def __len__(self):
        return len(self.values)

    def __bool__(self) -> bool:
        return len(self) != 0


def make_shape(shape: list[Any] | tuple[Any, ...]) -> ShapeExpr:
    if isinstance(shape, list | tuple):
        return ShapeExpr(shape)
    raise TypeError(
        "make_shape expects a list or tuple of shape values, "
        f"but received type {type(shape).__name__}"
    )


# Ordinary Relax bindings use the canonical IR Var directly.
Var = tvm_tirx.ir.Var


@tvm_ffi.register_object("relax.expr.DataflowVar")
class DataflowVar(Var):
    """A sub-type of the variable node used to mark dataflow variables from
    normal visible "function local" bindings.


    Parameters
    ----------
    name: str
        The name of the variable.

    ty: Optional[Type]
        The type annotation of the variable.

    loc: Location
        Location that points to original source code
    """

    name: str
    loc: Location

    def __init__(
        self,
        name: str | None = None,
        ty: Type | None = None,
        loc: Location = UNKNOWN_LOC,
        *,
        name_hint: str | None = None,
    ) -> None:
        # pylint: disable=super-init-not-called
        if name is None:
            name = name_hint
        elif name_hint is not None:
            raise TypeError("Specify either name or name_hint, not both")
        if not isinstance(name, str):
            raise TypeError("name must be a str")

        if ty is not None:
            ty = tvm_ffi.convert(ty)
            if not isinstance(ty, Type):
                raise TypeError(
                    "ty needs to be an instance of Type. "
                    "If you attempt to pass in shape, "
                    "use relax.TensorType(shape, dtype)."
                )

        self.__init_handle_by_constructor__(_ffi_api.DataflowVar, name, ty, loc)  # type: ignore


@tvm_ffi.register_object("relax.expr.Binding")
class Binding(Node, Scriptable):
    """The base class of a binding in Relax."""

    var: Var
    loc: Location


@tvm_ffi.register_object("relax.expr.MatchCast")
class MatchCast(Binding):
    """Runtime-match the value to the type.

    This operation does runtime check, populates the un-defined symbolic shape vars
    and vars in ty in the first occurrence, and insert equality assertions in
    other cases.

    Parameters
    ----------
    var: Var
        The return variable that the match cast bind to.

    value: Expr
        The input value expression.

    ty: tvm.relax.Type
        The type to match cast to.
    """

    ty: Type
    value: Expr
    loc: Location

    def __init__(self, var: Var, value: Expr, ty: Type, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MatchCast,
            var,
            value,
            ty,
            loc,  # type: ignore
        )


@tvm_ffi.register_object("relax.expr.VarBinding")
class VarBinding(Binding):
    """Variable binding, bind he variable of the lhs with the rhs.

    Parameters
    ----------
    var: Var
        The return variable that the match cast bind to.

    value: Expr
        The input value expression.

    """

    var: Var
    value: Expr
    loc: Location

    def __init__(self, var: Var, value: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.VarBinding, var, value, loc)  # type: ignore


@tvm_ffi.register_object("relax.expr.BindingBlock")
class BindingBlock(Node, Scriptable):
    """base class of binding block, bindings inside can be impure
    (with side effect or control flow)"""

    bindings: list[Binding]
    loc: Location

    def __init__(self, bindings: list[Binding], loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.BindingBlock, bindings, loc)  # type: ignore


@tvm_ffi.register_object("relax.expr.DataflowBlock")
class DataflowBlock(BindingBlock):
    """dataflow block, bindings inside are pure (no side effect and no control flow)"""

    bindings: list[Binding]
    loc: Location

    def __init__(self, bindings: list[Binding], loc: Location = UNKNOWN_LOC) -> None:
        # pylint: disable=super-init-not-called
        self.__init_handle_by_constructor__(_ffi_api.DataflowBlock, bindings, loc)  # type: ignore


@tvm_ffi.register_object("relax.expr.SeqExpr")
class SeqExpr(_CallableExprWithOp):
    """A sequence of binding blocks followed by an expression."""

    blocks: list[BindingBlock]
    body: Expr
    loc: Location

    def __bool__(self) -> bool:
        return True

    def __init__(self, blocks: list[BindingBlock], body: Expr, loc: Location = UNKNOWN_LOC) -> None:
        self.__init_handle_by_constructor__(_ffi_api.SeqExpr, blocks, body, loc)  # type: ignore


@tvm_ffi.register_object("relax.expr.Function")
class Function(BaseFunc, Scriptable):
    """A Relax function."""

    params: list[Var]
    body: Expr
    ret_ty: Type
    is_pure: bool
    attrs: tvm_tirx.ir.DictAttrs
    loc: Location

    def __init__(
        self,
        params: list[Var],
        body: Expr,
        ret_ty: Type | None = None,
        is_pure: bool | None = True,
        attrs: tvm_tirx.ir.DictAttrs | None = None,
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        if attrs is None:
            attrs = tvm_tirx.ir.DictAttrs({})
        self.__init_handle_by_constructor__(
            _ffi_api.Function,
            params,
            body,
            ret_ty,
            is_pure,
            attrs,
            loc,
        )  # type: ignore

    @staticmethod
    def create_empty(
        params: list[Var],
        ret_ty: Type,
        is_pure: bool | None = True,
        attrs: tvm_tirx.ir.DictAttrs | None = None,
        loc: Location = UNKNOWN_LOC,
    ):
        """Construct a relax.Function but without body"""
        if attrs is None:
            attrs = tvm_tirx.ir.DictAttrs({})
        return _ffi_api.FunctionCreateEmpty(params, ret_ty, is_pure, attrs, loc)  # type: ignore

    def __call__(self, *args):
        """Invoke the global function.

        Parameters
        ----------
        args: List[relax.Expr]
            Arguments.
        """
        return tvm_tirx.ir.Call(self, args, None, None)

    def bind_symbolic_vars(self, binding_map: Mapping[str | tvm_tirx.tirx.Var, Expr]) -> "Function":
        """Return a new function with updated symbolic variable

        Parameters
        ----------
        binding_map: Mapping[str | tvm_tirx.tirx.Var, Expr]

            The mapping of values to be replaced.  Keys may be either
            a `tirx.Var` or a string name of the variable.  If the
            variables are referred to by name, the name must uniquely
            identify a symbolic variable in the function.

        Returns
        -------
        func: Function

            The updated function
        """

        # Relax uses int64 for symbolic variables, but the FFI
        # converts python integers into int32.
        binding_map = {
            key: tvm_tirx.tirx.const(value, "int64") if isinstance(value, int) else value
            for key, value in binding_map.items()
        }

        return _ffi_api.FunctionBindSymbolicVars(self, binding_map)  # type: ignore

    def bind_params(
        self,
        binding_map: Mapping[
            str | Var,
            int | float | Expr | tvm_tirx.runtime.Tensor | _np.ndarray,
        ],
    ) -> "Function":
        """Return a new function with updated symbolic variable

        Parameters
        ----------
        binding_map: Mapping[
                str | Var,
                int | float | Expr | tvm_tirx.runtime.Tensor | _np.ndarray,
        ]

            The mapping of values to be replaced.

            Keys may be either a `relax.Var` or a string name of the
            Relax variable.  If the variables are referred to by name,
            the name must uniquely identify a parameter in the
            function.

            Values must be a relax expression, or a value that is
            convertible into a relax expression.  The value must be
            compatible with the variable being replaced.

        Returns
        -------
        func: Function

            The updated function
        """

        def _normalize_value(value):
            # Conversions that must occur prior to the FFI
            # conversions.
            if isinstance(value, int):
                # Relax uses int64 for symbolic variables, but the FFI
                # converts python integers into int32.
                return tvm_tirx.tirx.const(value, "int64")
            elif isinstance(value, _np.ndarray | tvm_tirx.runtime.Tensor):
                return tvm.relax.const(value)
            else:
                return value

        binding_map = {key: _normalize_value(value) for key, value in binding_map.items()}

        return _ffi_api.FunctionBindParams(self, binding_map)  # type: ignore

    def inline_functions(
        self, function_map: Mapping[str | tvm_tirx.ir.GlobalVar, "Function"]
    ) -> "Function":
        return _ffi_api.FunctionInlineFunctions(self, function_map)  # type: ignore


@tvm_ffi.register_object("relax.expr.ExternFunc")
class ExternFunc(BaseFunc):
    """extern function, which represents a PackedFunc."""

    global_symbol: String
    loc: Location

    def __init__(
        self,
        global_symbol: String,
        ty: Type | None = None,
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ExternFunc,
            global_symbol,
            ty,
            loc,  # type: ignore
        )


def extern(name: str, ty: Type | None = None, loc: Location = UNKNOWN_LOC):
    """Create extern function."""
    return ExternFunc(name, ty, loc)


def const(
    value: bool | int | float | _np.ndarray | tvm_tirx.runtime.Tensor, dtype: str | None = None
) -> tvm_tirx.ir.GenericConst:
    """Create a constant value.

    Parameters
    ----------
    value: bool | int | float | numpy.ndarray | tvm_tirx.runtime.Tensor
        The constant value.

    dtype: Optional[str]
        The data type of the resulting constant.

    Note
    ----
    When dtype is None, we use the following rule:

    - int maps to "int32"
    - float maps to "float32"
    - bool maps to "bool"
    - other using the same default rule as numpy.
    """
    # Needed for bf16 and fp8 support (does not come with numpy)
    import ml_dtypes  # pylint: disable=unused-import,import-outside-toplevel

    if isinstance(dtype, tvm_tirx.ir.PrimType):
        dtype = dtype.dtype

    if isinstance(value, Number | (bool | list)):
        value = _np.array(value, dtype=dtype)

    if not dtype:
        # when dtype is None: int maps to "int32", float maps to "float32"
        dtype = {  # type: ignore
            _np.dtype("int64"): _np.int32,  # type: ignore
            _np.dtype("float64"): _np.float32,  # type: ignore
        }.get(
            value.dtype,
            None,  # type: ignore
        )

    if isinstance(value, _np.ndarray | _np.generic):
        if dtype is not None:
            value = value.astype(dtype)
        value = tvm_tirx.runtime.tensor(value)

    if not isinstance(value, tvm_tirx.runtime.Tensor):
        raise ValueError("value has to be scalar or Tensor")

    return _ffi_api.MakeTensorConst(value, None, UNKNOWN_LOC)


@tvm_ffi.register_object("relax.TEPlaceholderOp")
class TEPlaceholderOp(tvm.te.tensor.Operation):
    """The placeholder op that represents a relax expression."""


def te_tensor(
    value: Expr,
    tir_var_map: dict[tvm_tirx.ir.Var, tvm_tirx.tirx.Expr],
    name: str = "rxplaceholder",
):
    """Create a TE tensor from a Relax expression, with primitive variables
    in the tensor shape substituted by the given mapping.

    Parameters
    ----------
    value : Expr
        The relax expression, which is required to have TensorType.

    tir_var_map : Dict[tvm_tirx.ir.Var, tvm_tirx.tirx.Expr]
        The exact-identity mapping used to refresh canonical primitive Vars
        that appear in the input shape.

    name : str
        The name of the created tensor.
    """
    return _ffi_api.TETensor(value, tir_var_map, name)  # type: ignore


def get_shape_of(expr: Expr) -> Expr:
    """Get shape of expr.

    Parameters
    ----------
    expr: Expr
        The input expr.

    Returns
    -------
    shape: Expr
        The shape expression

    Note
    ----
    This function requires expr to be normalized.
    The function will report an error if expr's Type is not TensorType.
    It will try to return symbolic function when possible. If the tensor do not
    have a compile-time symbolic shape, the function will then choose to return
    `Call(relax.op.shape_of, [expr])`.
    """
    return _ffi_api.GetShapeOf(expr)  # type: ignore


def _update_type(expr: Expr, ty: Type | None) -> None:
    _ffi_api.UpdateType(expr, ty)  # type: ignore
