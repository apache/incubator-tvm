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
# pylint: disable=redefined-builtin, invalid-name
"""Relax unary arithmetic operators."""

from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr
from ..utils import convert_to_expr

###################### Arithmetic operators ######################


def abs(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise absolute value of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.abs", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def acos(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise arc cos of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.acos", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def acosh(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise arc cosh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.acosh", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def asin(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise arc sin of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.asin", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def asinh(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise arc sinh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.asinh", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def atan(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise arc tan of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.atan", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def atanh(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise arc tanh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.atanh", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def bitwise_not(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute bitwise NOT of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.bitwise_not", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def ceil(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Take ceil of input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.ceil", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def cos(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise cos of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.cos", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def cosh(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise cosh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.cosh", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def exp(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise exp of data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.exp", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def floor(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Take floor of input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.floor", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def log(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise natural logarithm of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.log", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def logical_not(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute logical NOT of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.logical_not", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def negative(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise negative of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result
    """
    return _Call("relax.negative", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def round(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Rounds each element of the input data to nearest integer.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.round", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def rsqrt(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise reciprocal square root of the input data.

    .. math::

      1/sqrt(x)

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.rsqrt", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def sigmoid(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise sigmoid of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.sigmoid", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def sign(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Returns an indication of the sign of a number for each element of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.sign", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def sin(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise sin of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.sin", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def sinh(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise sinh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.sinh", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def square(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Squares each element of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.square", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def sqrt(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise square root of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.sqrt", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def tan(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise tan of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.tan", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def tanh(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Compute element-wise tanh of the input data.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.

    Note
    ----
    The input tensor is required to have float dtype
    """
    return _Call("relax.tanh", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def trunc(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Take trunc of input data.
    Parameters
    ----------
    x : relax.Expr
        The input data
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.trunc", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def clip(x: Expr, min: Expr, max: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Clips tensor values to a specified min and max.

    Parameters
    ----------
    x : relax.Expr
        The input data

    min : relax.Expr
        The minimum value

    max : relax.Expr
        The maximum value

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    min = convert_to_expr(min)
    max = convert_to_expr(max)
    return _Call("relax.clip", [x, min, max], loc=loc, ty=ty)  # type: ignore


def erf(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Computes the error function of the input.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        Computed error function for each element.
    """
    return _Call("relax.erf", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


###################### Check operators ######################


def isfinite(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Check if input value is finite.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.isfinite", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def isinf(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Check if input value is infinite.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.isinf", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def isnan(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Check if input value is Nan.

    Parameters
    ----------
    x : relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.isnan", [x], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore
