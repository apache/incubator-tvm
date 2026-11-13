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
# ruff: noqa: RUF005
"""Default legalization function for ccl operators."""

from tvm import topi
from tvm.relax import ExternFunc as _ExternFunc
from tvm_tirx import sym, tirx
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr, ShapeExpr
from ...op import call_dps_packed
from ...type import ShapeType, TensorType
from .common import register_legalize


@register_legalize("relax.ccl.allreduce")
def _allreduce(_bb: BlockBuilder, call: Call) -> Expr:
    op_type_str = call.attrs.op_type
    op_type_map = {
        "sum": 0,
        "prod": 1,
        "min": 2,
        "max": 3,
        "avg": 4,
    }
    if op_type_str not in op_type_map:
        raise ValueError(
            f"Unsupported reduction operation: {op_type_str}. "
            f"Supported operations are {op_type_map.keys()}."
        )
    return call_dps_packed(
        _ExternFunc("runtime.disco.allreduce"),
        [call.args[0], ShapeExpr([op_type_map[op_type_str]]), call.attrs.in_group],
        ty_args=[call.args[0].ty],
    )


@register_legalize("relax.ccl.allgather")
def _allgather(_bb: BlockBuilder, call: Call) -> Expr:
    output_shape = []
    arg_ty = call.args[0].ty
    assert isinstance(arg_ty, TensorType), "The input type of allgather should be TensorType."
    assert isinstance(arg_ty.shape.ty, ShapeType)
    arg_shape = arg_ty.shape.ty
    for i, shape_value in enumerate(arg_shape.values):
        if i == 0:
            output_shape.append(shape_value * call.attrs.num_workers)
        else:
            output_shape.append(shape_value)
    return call_dps_packed(
        _ExternFunc("runtime.disco.allgather"),
        [call.args[0], call.attrs.in_group],
        ty_args=[
            TensorType(
                shape=output_shape,
                dtype=arg_ty.dtype,
                vdevice=arg_ty.vdevice,
            )
        ],
    )


@register_legalize("relax.ccl.broadcast_from_worker0")
def _broadcast_from_worker0(_bb: BlockBuilder, call: Call) -> Expr:
    return call_dps_packed(
        _ExternFunc("runtime.disco.broadcast_from_worker0"),
        [call.args[0], False],
        ty_args=[call.args[0].ty],
    )


# Since collective communication ops are performed on contiguous memory,
# we need to reshape and transpose the input tensor to make sharding dimension in the highest order
def _transpose_for_ccl(_bb: BlockBuilder, expr: Expr, axis: int, num_workers: int):
    assert isinstance(expr.ty, TensorType), "The input type should be TensorType."
    assert isinstance(expr.ty.shape.ty, ShapeType)
    arg_shape = expr.ty.shape.ty
    new_shape = []
    for i, shape_value in enumerate(arg_shape.values):
        if i == axis:
            modulo = sym.Analyzer().simplify(shape_value % num_workers)
            assert modulo == 0, (
                f"scatter_from_worker0 expects the size of axis {axis} of input tensor "
                "to be divisible by num_workers. However, the axis 0 of input tensor "
                f"is {shape_value} while num_workers is {num_workers}"
            )
            new_shape.append(num_workers)
            new_shape.append(tirx.div(shape_value, num_workers))
        else:
            new_shape.append(shape_value)
    reshape_var = _bb.emit_te(topi.reshape, expr, new_shape)
    if axis == 0:
        return reshape_var
    permute_order = [axis] + list(range(axis)) + list(range(axis + 1, len(new_shape)))
    transpose_var = _bb.emit_te(topi.transpose, reshape_var, permute_order)
    return transpose_var


@register_legalize("relax.ccl.scatter_from_worker0")
def _scatter_from_worker0(_bb: BlockBuilder, call: Call) -> Expr:
    transpose_var = _transpose_for_ccl(_bb, call.args[0], call.attrs.axis, call.attrs.num_workers)
    output_shape = transpose_var.ty.shape.ty.values
    output_shape = output_shape[1:]
    return call_dps_packed(
        _ExternFunc("runtime.disco.scatter_from_worker0"),
        [transpose_var, False],
        ty_args=[
            TensorType(
                shape=output_shape,
                dtype=call.args[0].ty.dtype,
                vdevice=call.args[0].ty.vdevice,
            )
        ],
    )
