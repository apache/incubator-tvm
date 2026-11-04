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
# ruff: noqa: E731
"""Default legalization function for creation operators."""

import numpy as np

import tvm_tirx
from tvm import te, topi
from tvm_tirx import tirx
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr, ShapeExpr, const
from ...type import ShapeType
from .common import LegalizeFunc, _try_convert_to_scalar_const, register_legalize


def _full(is_like: bool, fill_value: float | None, function_name: str) -> LegalizeFunc:
    def full_call_te(bb: BlockBuilder, call: Call) -> Expr:
        _fill_value = (
            _try_convert_to_scalar_const(call.args[1]) if fill_value is None else fill_value
        )
        shape = call.args[0].ty.shape if is_like else call.args[0]

        if isinstance(shape, ShapeExpr):
            output_shape = shape.values
        else:
            assert isinstance(shape.ty, ShapeType)
            assert shape.ty.ndim >= 0

            shape = bb.emit(shape)
            output_shape = [tirx.Var(f"s{i}", "int64") for i in range(shape.ty.ndim)]
            bb.match_cast(shape, ShapeType(output_shape))

        return bb.call_te(
            topi.full,
            output_shape,
            call.ty.dtype,
            _fill_value,
            function_name_hint=function_name,
        )

    return full_call_te


def _tril_triu(is_upper: bool, function_name: str) -> LegalizeFunc:
    def tril_triu_call_te(bb: BlockBuilder, call: Call) -> Expr:
        data, k = call.args
        return bb.call_te(
            topi.trilu,
            data,
            k,
            upper=is_upper,
            function_name_hint=function_name,
        )

    return tril_triu_call_te


register_legalize("relax.full", _full(is_like=False, fill_value=None, function_name="full"))
register_legalize("relax.full_like", _full(is_like=True, fill_value=None, function_name="full"))
register_legalize("relax.ones", _full(is_like=False, fill_value=1.0, function_name="ones"))
register_legalize("relax.ones_like", _full(is_like=True, fill_value=1.0, function_name="ones"))
register_legalize("relax.zeros", _full(is_like=False, fill_value=0.0, function_name="zeros"))
register_legalize("relax.zeros_like", _full(is_like=True, fill_value=0.0, function_name="zeros"))
register_legalize("relax.tril", _tril_triu(is_upper=False, function_name="tril"))
register_legalize("relax.triu", _tril_triu(is_upper=True, function_name="triu"))


def _eye(is_like: bool, function_name: str) -> LegalizeFunc:
    def eye_call_te(bb: BlockBuilder, call: Call) -> Expr:
        _convert_to_scalar_const = lambda x: _try_convert_to_scalar_const(x, python_native=True)
        if is_like:
            x = call.args[0]
            k = _convert_to_scalar_const(call.args[1]) if len(call.args) > 1 else 0
            n, m = x.ty.shape
            dtype = x.ty.dtype
        else:
            n = _convert_to_scalar_const(call.args[0])
            m = _convert_to_scalar_const(call.args[1]) if len(call.args) > 1 else n
            k = _convert_to_scalar_const(call.args[2]) if len(call.args) > 2 else 0
            dtype = call.attrs.dtype

        return bb.call_te(
            topi.eye,
            n,
            m,
            k,
            dtype,
            function_name_hint=function_name,
        )

    return eye_call_te


register_legalize("relax.eye", _eye(is_like=False, function_name="eye"))
register_legalize("relax.eye_like", _eye(is_like=True, function_name="eye_like"))


@register_legalize("relax.arange")
def _arange(bb: BlockBuilder, call: Call) -> Expr:
    assert len(call.args) == 3
    assert all(tvm_tirx.ir.is_prim_expr(x) for x in call.args)
    start, end, step = call.args
    dtype = call.attrs.dtype

    def is_const_scalar(x: tirx.Expr):
        return isinstance(x, tvm_tirx.ir.IntImm | tvm_tirx.ir.FloatImm)

    if all([is_const_scalar(x) for x in call.args]):
        return const(np.arange(start.value, end.value, step.value, dtype=dtype), dtype=dtype)
    else:
        return bb.call_te(topi.arange, start, end, step, dtype)


@register_legalize("relax.shape_to_tensor")
def _shape_to_tensor(bb: BlockBuilder, call: Call) -> Expr:
    shape = call.args[0]
    values = shape.values if isinstance(shape, ShapeExpr) else shape.ty.values
    if values is None:
        return call
    values = list(values)
    n = len(values)
    symbolic = [v for v in values if not isinstance(v, tvm_tirx.ir.IntImm)]

    def te_shape_to_tensor(*sym):
        sym = list(sym)
        resolved = [v if isinstance(v, tvm_tirx.ir.IntImm) else sym.pop(0) for v in values]

        def fcompute(i):
            result = tirx.const(0, "int64")
            for idx in range(n - 1, -1, -1):
                result = tirx.if_then_else(i == idx, tirx.Cast("int64", resolved[idx]), result)
            return result

        return te.compute((n,), fcompute, name="shape_to_tensor")

    return bb.call_te(te_shape_to_tensor, *symbolic, function_name_hint="shape_to_tensor")


@register_legalize("relax.hamming_window")
def _hamming_window(bb: BlockBuilder, call: Call) -> Expr:
    assert len(call.args) == 4
    dtype = call.attrs.dtype
    window_size = call.args[0]
    periodic = call.args[1]
    alpha = call.args[2]
    beta = call.args[3]
    return bb.call_te(topi.hamming_window, window_size, periodic, alpha, beta, dtype)
