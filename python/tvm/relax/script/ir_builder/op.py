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
"""Relax expression operators and explicit call construction."""

from __future__ import annotations

import builtins
import numbers as _numbers
from types import SimpleNamespace as _Namespace
from typing import Any

import tvm_tirx
from tvm import relax as _relax
from tvm.relax import Call, Expr, ExternFunc
from tvm.relax.global_info import VDevice
from tvm.relax.op import (
    ArgmaxArgminAttrs,
    ArgsortAttrs,
    AstypeAttrs,
    CallInplacePackedAttrs,
    CallTIRInplaceAttrs,
    CallTIRWithGradAttrs,
    ConcatAttrs,
    EinsumAttrs,
    ExpandDimsAttrs,
    FlipAttrs,
    GatherElementsAttrs,
    GatherNDAttrs,
    HintOnDeviceAttrs,
    IndexPutAttrs,
    InitAttrs,
    LayoutTransformAttrs,
    MatmulAttrs,
    MeshgridAttrs,
    MultinomialFromUniformAttrs,
    OneHotAttrs,
    PermuteDimsAttrs,
    QuantizeAttrs,
    RepeatAttrs,
    ReverseSequenceAttrs,
    ScanopAttrs,
    ScatterElementsAttrs,
    ScatterNDAttrs,
    SliceScatterAttrs,
    SortAttrs,
    SplitAttrs,
    SqueezeAttrs,
    StackAttrs,
    StatisticalAttrs,
    StridedSliceAttrs,
    TakeAttrs,
    TileAttrs,
    TopKAttrs,
    ToVDeviceAttrs,
    TriluAttrs,
    WrapParamAttrs,
    abs,
    acos,
    acosh,
    add,
    arange,
    argmax,
    argmin,
    argsort,
    asin,
    asinh,
    assert_op,
    astype,
    atan,
    atan2,
    atanh,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    broadcast_to,
    bucketize,
    builtin,
    call_builtin_with_ctx,
    call_dps_packed,
    call_inplace_packed,
    call_pure_packed,
    call_tir,
    call_tir_inplace,
    call_tir_packed,
    call_tir_with_grad,
    ccl,
    ceil,
    clip,
    collapse_sum_like,
    collapse_sum_to,
    concat,
    cos,
    cosh,
    cumprod,
    cumsum,
    dequantize,
    divide,
    dynamic_strided_slice,
    einsum,
    equal,
    erf,
    ewise_fma,
    exp,
    expand_dims,
    eye,
    eye_like,
    flatten,
    flip,
    floor,
    floor_divide,
    floor_mod,
    full,
    full_like,
    gather_elements,
    gather_nd,
    grad,
    greater,
    greater_equal,
    hamming_window,
    hint_on_device,
    image,
    index_put,
    index_tensor,
    invoke_closure,
    invoke_pure_closure,
    isfinite,
    isinf,
    isnan,
    layout_transform,
    left_shift,
    less,
    less_equal,
    linear,
    log,
    log_add_exp,
    logical_xor,
    make_closure,
    matmul,
    max,
    maximum,
    mean,
    median,
    memory,
    meshgrid,
    min,
    minimum,
    mod,
    multinomial_from_uniform,
    multiply,
    negative,
    nn,
    nonzero,
    not_equal,
    null_value,
    one_hot,
    ones,
    ones_like,
    outer,
    permute_dims,
    power,
    print,
    prod,
    quantize,
    repeat,
    reshape,
    reverse_sequence,
    right_shift,
    round,
    rsqrt,
    scatter_elements,
    scatter_nd,
    searchsorted,
    shape_of,
    shape_to_tensor,
    sigmoid,
    sign,
    sin,
    sinh,
    size,
    slice_scatter,
    sort,
    split,
    sqrt,
    square,
    squeeze,
    stack,
    std,
    strided_slice,
    subtract,
    sum,
    take,
    tan,
    tanh,
    tensor_to_shape,
    tile,
    topk,
    tril,
    triu,
    trunc,
    unique,
    variance,
    vision,
    vm,
    where,
    wrap_param,
    zeros,
    zeros_like,
)
from tvm.relax.op import call_py_func as call_py_func
from tvm.relax.op import logical_and as _logical_and
from tvm.relax.op import logical_not as _logical_not
from tvm.relax.op import logical_or as _logical_or
from tvm.relax.op import to_vdevice as _to_vdevice
from tvm.relax.op.builtin import stop_lift_params
from tvm.relax.type import Type
from tvm.relax.utils import convert_to_expr
from tvm_tirx import ir as _ir
from tvm_tirx import tirx as _tir
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.ir.prim import _ffi_api as _prim_ffi
from tvm_tirx.script.ir_builder import base as _base

from . import distributed as dist  # noqa: F401
from .ir import _value, lookup_vdevice

py_print = builtins.print
py_tuple = tuple
py_str = str
_Loc = _base.LocationEntry | _ir.Location | None


def to_vdevice(
    data: Expr,
    dst_vdevice: py_str | VDevice = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Copy data to the destination device.

    Parameters
    ----------
    data : Expr
        The tensor to be copied.

    dst_vdevice : Union[py_str, VDevice]
        The destination device where the data is copied to.

    Returns
    -------
    result : Expr
        The copied result.
    """
    if isinstance(dst_vdevice, py_str):
        if ":" in dst_vdevice:
            split_vdev = dst_vdevice.split(":")
            dst_vdevice = lookup_vdevice(split_vdev[0], int(split_vdev[1]))
        else:
            dst_vdevice = lookup_vdevice(dst_vdevice, 0)

    return _to_vdevice(data, dst_vdevice, ty=ty, loc=loc)


def call_packed(
    func: py_str,
    *args: Expr,
    ty_args: Type | list[Type] | None = None,
    **kwargs: Any,
) -> Call:
    """Create a relax Call, which calls a packed function.
    Parameters
    ----------
    func: str
        The name of extern function.
    *args : Expr
        The arguments.
    ty_args: Optional[Union[Type, List[Type]]]
        The list of type information arguments.
    kwargs: Expr
        The keyword arguments.

    Returns
    -------
    call: Call
        The created Relax Call
    """
    op = ExternFunc(func)
    args = py_tuple(convert_to_expr(a) for a in args)
    if ty_args is None:
        ty_args = []
    if isinstance(ty_args, py_tuple):  # type: ignore
        ty_args = list(ty_args)
    elif not isinstance(ty_args, list):
        ty_args = [ty_args]

    ty_args = [ty() if callable(ty) and not isinstance(ty, _ir.Type) else ty for ty in ty_args]

    is_default = False
    if "attrs_type_key" in kwargs:
        attrs_type_key = kwargs["attrs_type_key"]
        kwargs.pop("attrs_type_key")
    else:
        attrs_type_key = "ir.DictAttrs"
        is_default = True
    attrs = None
    if kwargs or not is_default:
        attrs = tvm_tirx.ir.attrs.make_node(attrs_type_key, **kwargs)

    return Call(op, args, attrs=attrs, ty_args=ty_args)


def emit_with_type(
    op: str,
    args: Expr,
    ty_args: Type | list[Type] | None = None,
) -> Call:
    """Create a Relax Call with type arguments.
    Parameters
    ----------
    op: Expr
        The relax op for which type args are to be appended
    args : Expr
        The arguments.
    ty_args: Optional[Union[Type, List[Type]]]
        The list of type arguments.

    Returns
    -------
    call: Call
        The created Relax Call
    """
    return Call(op, args, ty_args=ty_args)


def _logical_pair(lhs, rhs, operation, primitive, python_operation):
    if not isinstance(lhs, _ir.Expr) and not isinstance(rhs, _ir.Expr):
        return python_operation(lhs, rhs)
    if _ir.is_prim_expr(lhs) or _ir.is_prim_expr(rhs):
        return primitive(lhs, rhs)
    return operation(_value(lhs), _value(rhs))


def logical_and(*values, ty_args=None, ty=None, loc=None):
    """Construct conjunction of host, primitive, or tensor values.

    Parameters
    ----------
    values : Expr or Python value
        One or more eagerly evaluated operands. Host pairs follow Python
        logical operations; primitive pairs use primitive IR and tensor pairs
        use the corresponding Relax operation.

    Returns
    -------
    result : Expr or Python value
        The conjunction reduced from left to right.

    Notes
    -----
    All arguments are evaluated before this call; it does not provide Python
    short-circuit evaluation of the argument expressions.
    """
    # Omitted metadata retains host/primitive dispatch; explicit metadata selects Relax.
    if any(field is not None for field in (ty_args, ty, loc)):
        return _logical_and(
            *values, ty_args=ty_args, ty=ty, loc=UNKNOWN_LOC if loc is None else loc
        )
    if not values:
        raise TypeError("logical_and requires at least one operand")
    result = values[0]
    for value in values[1:]:
        result = _logical_pair(result, value, _logical_and, _tir.And, lambda a, b: a and b)
    return result


def logical_or(*values, ty_args=None, ty=None, loc=None):
    """Construct disjunction of host, primitive, or tensor values.

    Parameters
    ----------
    values : Expr or Python value
        One or more eagerly evaluated operands. Host pairs follow Python
        logical operations; primitive pairs use primitive IR and tensor pairs
        use the corresponding Relax operation.

    Returns
    -------
    result : Expr or Python value
        The disjunction reduced from left to right.

    Notes
    -----
    All arguments are evaluated before this call; it does not provide Python
    short-circuit evaluation of the argument expressions.
    """
    # Omitted metadata retains host/primitive dispatch; explicit metadata selects Relax.
    if any(field is not None for field in (ty_args, ty, loc)):
        return _logical_or(*values, ty_args=ty_args, ty=ty, loc=UNKNOWN_LOC if loc is None else loc)
    if not values:
        raise TypeError("logical_or requires at least one operand")
    result = values[0]
    for value in values[1:]:
        result = _logical_pair(result, value, _logical_or, _tir.Or, lambda a, b: a or b)
    return result


def logical_not(value, *, ty_args=None, ty=None, loc=None):
    """Negate a host, primitive, or tensor condition.

    Parameters
    ----------
    value : Expr or Python value
        Operand to negate. Primitive and tensor expressions use their IR
        logical operation; host values use Python truth testing.

    Returns
    -------
    result : Expr or bool
        The logical negation without testing an IR expression as a Python bool.
    """
    # Omitted metadata retains host/primitive dispatch; explicit metadata selects Relax.
    if any(field is not None for field in (ty_args, ty, loc)):
        return _logical_not(value, ty_args=ty_args, ty=ty, loc=UNKNOWN_LOC if loc is None else loc)
    if _ir.is_prim_expr(value):
        return _tir.Not(value)
    if isinstance(value, _ir.Expr):
        return _logical_not(value)
    return not value


def select(condition, true_value, false_value):
    """Select between already-constructed values.

    Parameters
    ----------
    condition : Expr or Python value
        Primitive condition, tensor condition, or host truth value.
    true_value : Expr or Python value
        Value selected when the condition is true.
    false_value : Expr or Python value
        Value selected when the condition is false.

    Returns
    -------
    result : Expr or Python value
        A primitive Select, Relax elementwise where, or the selected host value.

    Notes
    -----
    Both value arguments are evaluated before this call. Host selection returns
    the selected object; tensor selection converts Python numbers and tuples.
    """
    if _ir.is_prim_expr(condition):
        return _tir.Select(condition, true_value, false_value)
    if isinstance(condition, _ir.Expr):
        return _relax.op.where(condition, _value(true_value), _value(false_value))
    return true_value if condition else false_value


def if_then_else_(condition: Any, true_value: Any, false_value: Any) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.if_then_else_`."""
    if hasattr(condition, "__tvm_ffi_object__"):
        condition = condition.__tvm_ffi_object__()
    if not isinstance(condition, _ir.Expr):
        return true_value if condition else false_value
    true_value = (
        true_value.__tvm_ffi_object__() if hasattr(true_value, "__tvm_ffi_object__") else true_value
    )
    false_value = (
        false_value.__tvm_ffi_object__()
        if hasattr(false_value, "__tvm_ffi_object__")
        else false_value
    )
    if _ir.is_prim_expr(condition) and all(
        _ir.is_prim_expr(value)
        if isinstance(value, _ir.Expr)
        else isinstance(value, _numbers.Number)
        for value in (true_value, false_value)
    ):
        return _tir.if_then_else(condition, true_value, false_value)
    return _relax.IfExpr(condition, _value(true_value), _value(false_value))


def and_(*values: Any) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.and_`."""
    return logical_and(*values)


def or_(*values: Any) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.or_`."""
    return logical_or(*values)


def not_(value: Any) -> Any:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.not_`."""
    return logical_not(value)


def lt_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.lt_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(loc, _relax.op.less(lhs, rhs))
    loc = UNKNOWN_LOC if loc is None else loc
    return _prim_ffi._OpLT(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def le_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.le_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(loc, _relax.op.less_equal(lhs, rhs))
    loc = UNKNOWN_LOC if loc is None else loc
    return _prim_ffi._OpLE(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def gt_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.gt_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(loc, _relax.op.greater(lhs, rhs))
    loc = UNKNOWN_LOC if loc is None else loc
    return _prim_ffi._OpGT(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def ge_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.ge_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(loc, _relax.op.greater_equal(lhs, rhs))
    loc = UNKNOWN_LOC if loc is None else loc
    return _prim_ffi._OpGE(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def eq_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.eq_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(loc, _relax.op.equal(lhs, rhs))
    loc = UNKNOWN_LOC if loc is None else loc
    return _prim_ffi._OpEQ(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


def ne_(lhs: Any, rhs: Any, *, loc: _Loc = None) -> _ir.Expr:
    """Implements :func:`tvm_tirx.script.ir_builder.parser_protocol.ne_`."""
    if any(isinstance(value, _ir.Expr) and not _ir.is_prim_expr(value) for value in (lhs, rhs)):
        lhs = _relax.const(lhs) if isinstance(lhs, _numbers.Number) else lhs
        rhs = _relax.const(rhs) if isinstance(rhs, _numbers.Number) else rhs
        return _base.at_(loc, _relax.op.not_equal(lhs, rhs))
    loc = UNKNOWN_LOC if loc is None else loc
    return _prim_ffi._OpNE(lhs, rhs, loc.loc if isinstance(loc, _base.LocationEntry) else loc)


inspect = _Namespace()

__all__ = [
    "ArgmaxArgminAttrs",
    "ArgsortAttrs",
    "AstypeAttrs",
    "CallInplacePackedAttrs",
    "CallTIRInplaceAttrs",
    "CallTIRWithGradAttrs",
    "ConcatAttrs",
    "EinsumAttrs",
    "ExpandDimsAttrs",
    "FlipAttrs",
    "GatherElementsAttrs",
    "GatherNDAttrs",
    "HintOnDeviceAttrs",
    "IndexPutAttrs",
    "InitAttrs",
    "LayoutTransformAttrs",
    "MatmulAttrs",
    "MeshgridAttrs",
    "MultinomialFromUniformAttrs",
    "OneHotAttrs",
    "PermuteDimsAttrs",
    "QuantizeAttrs",
    "RepeatAttrs",
    "ReverseSequenceAttrs",
    "ScanopAttrs",
    "ScatterElementsAttrs",
    "ScatterNDAttrs",
    "SliceScatterAttrs",
    "SortAttrs",
    "SplitAttrs",
    "SqueezeAttrs",
    "StackAttrs",
    "StatisticalAttrs",
    "StridedSliceAttrs",
    "TakeAttrs",
    "TileAttrs",
    "ToVDeviceAttrs",
    "TopKAttrs",
    "TriluAttrs",
    "WrapParamAttrs",
    "abs",
    "acos",
    "acosh",
    "add",
    "and_",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asin",
    "asinh",
    "assert_op",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "broadcast_to",
    "bucketize",
    "builtin",
    "call_builtin_with_ctx",
    "call_dps_packed",
    "call_inplace_packed",
    "call_packed",
    "call_pure_packed",
    "call_py_func",
    "call_tir",
    "call_tir_inplace",
    "call_tir_packed",
    "call_tir_with_grad",
    "ccl",
    "ceil",
    "clip",
    "collapse_sum_like",
    "collapse_sum_to",
    "concat",
    "cos",
    "cosh",
    "cumprod",
    "cumsum",
    "dequantize",
    "divide",
    "dynamic_strided_slice",
    "einsum",
    "emit_with_type",
    "eq_",
    "equal",
    "erf",
    "ewise_fma",
    "exp",
    "expand_dims",
    "eye",
    "eye_like",
    "flatten",
    "flip",
    "floor",
    "floor_divide",
    "floor_mod",
    "full",
    "full_like",
    "gather_elements",
    "gather_nd",
    "ge_",
    "grad",
    "greater",
    "greater_equal",
    "gt_",
    "hamming_window",
    "hint_on_device",
    "if_then_else_",
    "image",
    "index_put",
    "index_tensor",
    "inspect",
    "invoke_closure",
    "invoke_pure_closure",
    "isfinite",
    "isinf",
    "isnan",
    "layout_transform",
    "le_",
    "left_shift",
    "less",
    "less_equal",
    "linear",
    "log",
    "log_add_exp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lt_",
    "make_closure",
    "matmul",
    "max",
    "maximum",
    "mean",
    "median",
    "memory",
    "meshgrid",
    "min",
    "minimum",
    "mod",
    "multinomial_from_uniform",
    "multiply",
    "ne_",
    "negative",
    "nn",
    "nonzero",
    "not_",
    "not_equal",
    "null_value",
    "one_hot",
    "ones",
    "ones_like",
    "or_",
    "outer",
    "permute_dims",
    "power",
    "print",
    "prod",
    "quantize",
    "repeat",
    "reshape",
    "reverse_sequence",
    "right_shift",
    "round",
    "rsqrt",
    "scatter_elements",
    "scatter_nd",
    "searchsorted",
    "select",
    "shape_of",
    "shape_to_tensor",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "size",
    "slice_scatter",
    "sort",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "stop_lift_params",
    "strided_slice",
    "subtract",
    "sum",
    "take",
    "tan",
    "tanh",
    "tensor_to_shape",
    "tile",
    "to_vdevice",
    "topk",
    "tril",
    "triu",
    "trunc",
    "unique",
    "variance",
    "vision",
    "vm",
    "where",
    "wrap_param",
    "zeros",
    "zeros_like",
]
