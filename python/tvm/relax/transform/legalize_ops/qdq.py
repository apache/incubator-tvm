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
"""Default legalization function for quantize/dequantize operators."""

import tvm_tirx
from tvm import te
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr
from .common import _try_convert_to_scalar_const, register_legalize


def clip_cast(val, dtype):
    const_min = tvm_tirx.tirx.min_value(ty=dtype)
    const_max = tvm_tirx.tirx.max_value(ty=dtype)
    return te.max(te.min(val, const_max), const_min).astype(dtype)


def is_const_scalar(x):
    return isinstance(x, tvm_tirx.ir.IntImm | tvm_tirx.ir.FloatImm)


def _is_singleton_qparam(qparam: te.Tensor) -> bool:
    """Return True if qparam is a tensor with all dimensions equal to 1."""
    if not isinstance(qparam, te.Tensor):
        return False
    if len(qparam.shape) == 0:
        return True
    for dim in qparam.shape:
        if not isinstance(dim, tvm_tirx.ir.IntImm) or dim.value != 1:
            return False
    return True


@register_legalize("relax.quantize")
def _quantize(bb: BlockBuilder, call: Call) -> Expr:
    """
    Lower relax.quantize into the sequence of simple operations.
    Quantization formula is defined as: out = clip(round(input / scale) + zp, min_val, max_val)
    """
    axis = call.attrs.axis
    out_dtype = call.attrs.out_dtype

    def te_quantize(
        data: te.Tensor,
        scale: te.Tensor | tvm_tirx.ir.IntImm | tvm_tirx.ir.FloatImm,
        zp: te.Tensor | tvm_tirx.ir.IntImm | tvm_tirx.ir.FloatImm,
    ):
        scale_singleton = _is_singleton_qparam(scale) if isinstance(scale, te.Tensor) else False
        zp_singleton = _is_singleton_qparam(zp) if isinstance(zp, te.Tensor) else False

        def quantize_compute(*indices):
            if is_const_scalar(scale):
                scale_value = scale
            elif scale_singleton:
                scale_value = scale[(0,) * len(scale.shape)]
            else:
                scale_value = scale[indices[axis]]

            if is_const_scalar(zp):
                zp_value = zp
            elif zp_singleton:
                zp_value = zp[(0,) * len(zp.shape)]
            else:
                zp_value = zp[indices[axis]]
            scaled = data[indices] / scale_value
            round_val = (te.round(scaled) if "int" in out_dtype else scaled) + zp_value
            return clip_cast(round_val, out_dtype)

        output_shape = data.shape
        return te.compute(output_shape, quantize_compute, name="quantized")

    return bb.call_te(
        te_quantize,
        call.args[0],
        _try_convert_to_scalar_const(call.args[1]),
        _try_convert_to_scalar_const(call.args[2]),
        function_name_hint="quantize",
    )


@register_legalize("relax.dequantize")
def _dequantize(bb: BlockBuilder, call: Call) -> Expr:
    """
    Lower relax.dequantize into the sequence of simple operations.
    Dequantization formula is defined as: out = scale * (input - zp)
    Compute datatype: float32

    Example of lowering:

        dtype = ["int32"|"float32"]

        qnn.dequantize(data, scale, zp, "float32") -->
            sub = subtract(cast(data, dtype), zp)
            out = multiply(cast(sub, "float32"), scale)

        qnn.dequantize(data, scale, zp, "float16") -->
            sub = subtract(cast(data, dtype), zp)
            mul = multiply(cast(sub, "float32"), cast(scale, "float32"))
            clipped_out = clip(mul, float32(-65504.0), float32(65504.0))
            out = cast(clipped_out, "float16")
    """
    axis = call.attrs.axis
    out_dtype = call.attrs.out_dtype

    def te_dequantize(
        data: te.Tensor,
        scale: te.Tensor | tvm_tirx.ir.IntImm | tvm_tirx.ir.FloatImm,
        zp: te.Tensor | tvm_tirx.ir.IntImm | tvm_tirx.ir.FloatImm,
    ):
        scale_singleton = _is_singleton_qparam(scale) if isinstance(scale, te.Tensor) else False
        zp_singleton = _is_singleton_qparam(zp) if isinstance(zp, te.Tensor) else False

        def dequantize_compute(*indices):
            if is_const_scalar(scale):
                scale_value = scale
            elif scale_singleton:
                scale_value = scale[(0,) * len(scale.shape)]
            else:
                scale_value = scale[indices[axis]]

            if is_const_scalar(zp):
                zp_value = zp
            elif zp_singleton:
                zp_value = zp[(0,) * len(zp.shape)]
            else:
                zp_value = zp[indices[axis]]
            dtype = "float32" if data.dtype.dtype.is_float else "int32"
            sub = data[indices].astype(dtype) - zp_value
            out = sub * scale_value.astype("float32")
            if out_dtype == "float32":
                return out
            return clip_cast(out, out_dtype)

        output_shape = data.shape
        return te.compute(output_shape, dequantize_compute, name="dequantized")

    return bb.call_te(
        te_dequantize,
        call.args[0],
        _try_convert_to_scalar_const(call.args[1]),
        _try_convert_to_scalar_const(call.args[2]),
        function_name_hint="dequantize",
    )
