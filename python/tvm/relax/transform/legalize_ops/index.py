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
"""Default legalization function for index operators."""

import tvm_tirx
from tvm import te, topi
from tvm_tirx import tirx
from tvm_tirx.ir import Call, PrimType

from ...block_builder import BlockBuilder
from ...expr import Expr, Tuple
from ...op import tensor_to_shape
from ...type import ShapeType
from .common import register_legalize


@register_legalize("relax.take")
def _take(bb: BlockBuilder, call: Call) -> Expr:
    # Currently "fast" is the default mode, which leads to segmentation faults
    # when there are out-of-bounds indices.
    return bb.call_te(topi.take, call.args[0], call.args[1], call.attrs.axis, mode=call.attrs.mode)


@register_legalize("relax.strided_slice")
def _strided_slice(bb: BlockBuilder, call: Call) -> Expr:
    def _relax_tuple_to_tir(relax_tuple):
        if isinstance(relax_tuple, Tuple):
            output = []
            for field in relax_tuple.fields:
                assert tvm_tirx.ir.is_prim_expr(field)
                output.append(field)
            return output

        output = []
        for field in relax_tuple.ty.fields:
            assert isinstance(field, PrimType)
            return None
        return output

    if len(call.args) == 4:
        data, axes, begin, end = call.args
        strides = [tvm_tirx.ir.IntImm("int64", 1)] * len(axes.ty.fields)
    elif len(call.args) == 5:
        data, axes, begin, end, strides = call.args
        strides = _relax_tuple_to_tir(strides)
    else:
        raise ValueError(
            f"Expression {call} provides {len(call.args)} arguments, "
            f"but {call.op} requires either 4 or 5 arguments."
        )

    axes = _relax_tuple_to_tir(axes)
    begin = _relax_tuple_to_tir(begin)
    end = _relax_tuple_to_tir(end)

    return bb.call_te(
        topi.strided_slice,
        data,
        begin,
        end,
        strides,
        axes,
        slice_mode="end",
        assume_inbound=call.attrs.assume_inbound,
    )


@register_legalize("relax.dynamic_strided_slice")
def _dynamic_strided_slice(bb: BlockBuilder, call: Call) -> Expr:
    assert len(call.args) == 4
    data, begin, end, strides = call.args

    # 1. Insert shape function
    def shape_func(data, begin, end, strides):
        def _compute(i):
            def canonicalize_index(index, extent, negative_stride):
                begin_range = tirx.const(-1 if negative_stride else 0, "int64")
                end_range = extent - 1 if negative_stride else extent
                index = tirx.Select(index < 0, index + extent, index)
                return tirx.Min(tirx.Max(index, begin_range), end_range)

            def get_length(begin, end, strides, length):
                # Construct each branch with its known stride sign. Tensor reads are
                # mutable, so the analyzer cannot assume repeated reads agree.
                begin_neg = canonicalize_index(begin, length, True)
                end_neg = canonicalize_index(end, length, True)
                begin_pos = canonicalize_index(begin, length, False)
                end_pos = canonicalize_index(end, length, False)
                len1 = tirx.ceildiv(begin_neg - end_neg, -strides)
                len2 = tirx.ceildiv(end_pos - begin_pos, strides)
                return tirx.Select(strides < 0, len1, len2)

            length = tirx.const(-1, "int64")
            for idx in range(data.ndim):
                length = tirx.Select(i == tirx.const(idx, "int64"), data.shape[idx], length)

            return get_length(begin[i], end[i], strides[i], length)

        return te.compute((begin.shape[0],), _compute, name="T_shape_func_strided_slice_dynamic")

    output_shape = bb.normalize(
        bb.call_te(
            shape_func,
            data,
            begin,
            end,
            strides,
        )
    )

    # 2. Convert tensor to shape and match cast with new symbolic vars
    ndim = int(output_shape.ty.shape[0])
    output_shape = bb.emit(tensor_to_shape(output_shape))
    output_shape_vars = [tirx.Var("s", "int64") for i in range(ndim)]
    bb.match_cast(output_shape, ShapeType(output_shape_vars))

    # 3. Pass the output shape vars to TOPI
    return bb.call_te(
        topi.dynamic_strided_slice,
        call.args[0],
        call.args[1],
        call.args[2],
        call.args[3],
        output_shape=output_shape_vars,
    )
