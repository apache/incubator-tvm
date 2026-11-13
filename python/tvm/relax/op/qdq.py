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
"""Relax quantize/dequantize operators"""

import tvm_ffi

from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir import PrimType as _PrimType
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr


@tvm_ffi.register_object("relax.attrs.QuantizeAttrs")
class QuantizeAttrs(Attrs):
    """Attributes used in quantize/dequantize operators"""


def quantize(
    data: Expr,
    scale: Expr,
    zero_point: Expr,
    axis: int = -1,
    out_dtype: str = "int8",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    r"""Quantize op
    This operator takes input and produces quantized output. The input tensor can be of any shape.
    The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/scale) + zero_point), out_dtype::min, out_dtype::max)

    Parameters
    ----------
    data : tvm.relax.Expr
        The input tensor to be quantized.

    scale : tvm.relax.Expr
        The output scale.

    zero_point : tvm.relax.Expr
        The output zero_point.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    out_dtype : str, optional
        The data type of the output tensor.

    Returns
    -------
    result : tvm.relax.Expr
        The computed result.
    """

    return _Call(
        "relax.quantize",
        [data, scale, zero_point],
        attrs=_make_attrs(
            "relax.attrs.QuantizeAttrs",
            axis=axis,
            out_dtype=(out_dtype.dtype if isinstance(out_dtype, _PrimType) else out_dtype),
        ),
        loc=loc,
        ty=ty,
    )


def dequantize(
    data: Expr,
    scale: Expr,
    zero_point: Expr,
    axis: int = -1,
    out_dtype: str = "float32",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    r"""Dequantize op
    This operator takes input and produces dequantized output. The input tensor can be of any shape.
    The output shape is the same as input shape.

    output = clamp(scale * (input_tensor - zero_point), out_dtype::min, out_dtype::max)

    Parameters
    ----------
    data : tvm.relax.Expr
        The input tensor to be dequantized.

    scale : tvm.relax.Expr
        The input scale.

    zero_point : tvm.relax.Expr
        The input zero_point.

    axis : int
        The channel axis for dequantization. Default value is -1 which corresponds to the last axis.

    out_dtype : str, optional
        The data type of the output tensor.

    Returns
    -------
    result : tvm.relax.Expr
        The computed result.
    """

    return _Call(
        "relax.dequantize",
        [data, scale, zero_point],
        attrs=_make_attrs(
            "relax.attrs.QuantizeAttrs",
            axis=axis,
            out_dtype=(out_dtype.dtype if isinstance(out_dtype, _PrimType) else out_dtype),
        ),
        loc=loc,
        ty=ty,
    )
