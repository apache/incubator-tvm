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
"""Elementwise operators"""

# pylint: disable=redefined-builtin,unused-argument
import tvm
import tvm_tirx
from tvm import te
from tvm_tirx import DataTypeCode
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from . import cpp, tag


def _require_float_tensor(op_name, x):
    if not x.dtype.matches_code(DataTypeCode.FLOAT, DataTypeCode.BFLOAT):
        raise TypeError(f"topi.{op_name} only supports floating-point inputs, but got {x.dtype}")
    return x


def _is_integer_tensor(x):
    return x.dtype.matches_code(DataTypeCode.INT, DataTypeCode.UINT)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def identity(x):
    """Take identity of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return te.compute(x.shape, lambda *i: x(*i))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def negative(x):
    """Take negation of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    # pylint: disable=unnecessary-lambda
    return te.compute(x.shape, lambda *i: -x(*i))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def exp(x):
    """Take exponential of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.exp(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def erf(x):
    """Take gauss error function of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.erf(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def tanh(x):
    """Take hyperbolic tanh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.tanh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def tan(x):
    """Take tan of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.tan(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def cos(x):
    """Take cos of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.cos(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def cosh(x):
    """Take cosh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.cosh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def sin(x):
    """Take sin of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.sin(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def sinh(x):
    """Take sinh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.sinh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def acos(x):
    """Take arc cos of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    x = _require_float_tensor("acos", x)
    return te.compute(x.shape, lambda *i: te.acos(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def acosh(x):
    """Take arc cosh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    x = _require_float_tensor("acosh", x)
    return te.compute(x.shape, lambda *i: te.acosh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def asin(x):
    """Take arc sin of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    x = _require_float_tensor("asin", x)
    return te.compute(x.shape, lambda *i: te.asin(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def asinh(x):
    """Take arc sinh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    x = _require_float_tensor("asinh", x)
    return te.compute(x.shape, lambda *i: te.asinh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def atan(x):
    """Take atan of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.atan(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def atanh(x):
    """Take atanh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    x = _require_float_tensor("atanh", x)
    return te.compute(x.shape, lambda *i: te.atanh(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def floor(x):
    """Take floor of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.floor(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def ceil(x):
    """Take ceil of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.ceil(x(*i)))


def sign(x):
    """Returns -1, 0, 1 based on sign of x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.sign(x)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def trunc(x):
    """Take truncated value of the input of x, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.trunc(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def abs(x):
    """Take absolute value of the input of x, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.abs(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def isnan(x):
    """Check if value of x is NaN, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.isnan(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def isfinite(x):
    """Check if value of x is finite, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.isfinite(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def isinf(x):
    """Check if value of x is infinite, element-wise.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.isinf(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def round(x):
    """Round elements of x to nearest integer using ties-to-even (banker's rounding).

    Ties are broken by rounding to the nearest even integer, matching the ONNX Round
    specification and IEEE 754 default rounding mode.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.nearbyint(x(*i)))


def log(x):
    """Take logarithm of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if x.dtype.matches_code(DataTypeCode.INT):
        x = te.compute(x.shape, lambda *i: x(*i).astype("float32"))
    return te.compute(x.shape, lambda *i: te.log(x(*i)), tag=tag.ELEMWISE)


def log2(x):
    """Take logarithm to the base 2 of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if x.dtype.matches_code(DataTypeCode.INT):
        x = te.compute(x.shape, lambda *i: x(*i).astype("float32"))
    return te.compute(x.shape, lambda *i: te.log2(x(*i)), tag=tag.ELEMWISE)


def log10(x):
    """Take logarithm to the base 10 of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if x.dtype.matches_code(DataTypeCode.INT):
        x = te.compute(x.shape, lambda *i: x(*i).astype("float32"))
    return te.compute(x.shape, lambda *i: te.log10(x(*i)), tag=tag.ELEMWISE)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def sqrt(x):
    """Take square root of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if x.dtype.matches_code(DataTypeCode.INT):
        x = te.compute(x.shape, lambda *i: x(*i).astype("float32"))
    return te.compute(x.shape, lambda *i: te.sqrt(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def rsqrt(x):
    """Take inverse square root of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if x.dtype.matches_code(DataTypeCode.INT):
        x = te.compute(x.shape, lambda *i: x(*i).astype("float32"))
    return te.compute(x.shape, lambda *i: te.rsqrt(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def sigmoid(x):
    """Take sigmoid tanh of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: te.sigmoid(x(*i)))


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def left_shift(x, n):
    """Take n bits left shift of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: x(*i) << n)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def right_shift(x, n):
    """Take n bits right shift of input x.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    n : int
        Number of bits.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return te.compute(x.shape, lambda *i: x(*i) >> n)


@tvm.te.tag_scope(tag=tag.ELEMWISE)
def clip(x, a_min, a_max):
    """Clip (limit) the values in an array. Given an interval, values
    outside the interval are clipped to the interval edges.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    a_min : tvm_tirx.tirx.Expr
        Minimum value.
    a_max : tvm_tirx.tirx.Expr
        Maximum value.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """

    def _compute(*indices):
        value = x(*indices)
        const_min = (
            tvm_tirx.tirx.Cast(value.ty, a_min)
            if tvm_tirx.ir.is_prim_expr(a_min)
            else tvm_tirx.tirx.const(a_min, value.ty)
        )
        const_max = (
            tvm_tirx.tirx.Cast(value.ty, a_max)
            if tvm_tirx.ir.is_prim_expr(a_max)
            else tvm_tirx.tirx.const(a_max, value.ty)
        )
        return tvm.te.max(tvm.te.min(value, const_max), const_min)

    return te.compute(x.shape, _compute)


def cast(x, dtype, loc: Location = UNKNOWN_LOC):
    """Cast input to specified data type.

    Parameters
    ----------
    x : tvm.te.Tensor or Expr
        Input argument.

    dtype : str
        Data type.

    loc : Location, optional
        The location of the cast in the source.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if isinstance(x, te.tensor.Tensor):
        return te.compute(x.shape, lambda *i: x(*i).astype(dtype), tag=tag.ELEMWISE)
    # pylint: disable=import-outside-toplevel
    from tvm_tirx.ir.prim import _ffi_api

    return _ffi_api._cast(x, dtype, loc)


def reinterpret(x, dtype):
    """Reinterpret input to specified data type.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    dtype : str
        Data type.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.reinterpret(x, dtype)


def fast_exp(x):
    """Take exponential of input x using fast_exp implementation

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if _is_integer_tensor(x):
        x = cast(x, "float32")
    return cpp.fast_exp(x, x.dtype, tag.ELEMWISE)


def fast_tanh(x):
    """Take hyperbolic tangent of input x using fast_tanh implementation

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if _is_integer_tensor(x):
        x = cast(x, "float32")
    return cpp.fast_tanh(x, x.dtype, tag.ELEMWISE)


def fast_erf(x):
    """Take gauss error function of input x using fast_erf implementation.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.fast_erf(x, x.dtype, tag.ELEMWISE)


def ceil_log2(x, *, target=None):
    """Compute integer ceil log2 with a special code path for vulkan
    SPIR-V does not support log2 on fp64. Instead, we compute integer ceil_log2 via clz
    intrinsic when the target is vulkan.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    if not tvm_tirx.ir.is_prim_expr(x):
        x = tvm_tirx.ir.const(x)

    if x.ty.matches_code(DataTypeCode.FLOAT, DataTypeCode.BFLOAT):
        return tvm_tirx.tirx.ceil(tvm_tirx.tirx.log2(x))

    if target is not None:
        target_name = target.kind.name
        if "vulkan" in target_name:
            clz = tvm_tirx.ir.prim.clz(x)
            bits = x.ty.dtype.bits
            res = tvm_tirx.tirx.if_then_else(x & (x - 1) == 0, bits - clz - 1, bits - clz)
            if res.ty != x.ty:
                return cast(res, x.ty)
            return res

        if "adreno" in str(target.attrs.get("device", "")) or target_name in [
            "metal",
            "rocm",
            "webgpu",
        ]:
            return cast(tvm_tirx.tirx.ceil(tvm_tirx.tirx.log2(cast(x, "float32"))), x.ty)

    return cast(tvm_tirx.tirx.ceil(tvm_tirx.tirx.log2(cast(x, "float64"))), x.ty)
