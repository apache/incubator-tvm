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
"""Relax binary arithmetic and comparison operators."""

from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr

###################### Arithmetic operators ######################


def add(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Addition with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.

    Returns
    -------
    result : Expr
        The computed result.

    Examples
    --------
    .. code:: python

      bb = relax.BlockBuilder()
      a = relax.Var("a", relax.TensorType(shape=(2, 3), dtype="float32"))
      b = relax.Var("b", relax.TensorType(shape=(2, 1), dtype="float32"))
      c = bb.normalize(relax.op.add(a, b))  # c has TensorType(shape=(2, 3), dtype="float32")
    """
    return _Call("relax.add", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def divide(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Division with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.divide", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def floor_divide(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Floor division with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.floor_divide", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def log_add_exp(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """
    Compute the log of the sum of exponentials of the inputs, element-wise.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.

    Returns
    -------
    Expr
        The element-wise log-sum-exp of `x1` and `x2`.
    """
    return _Call("relax.log_add_exp", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


def multiply(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Multiplication with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.

    Returns
    -------
    result : Expr
        The computed result.
    """
    return _Call("relax.multiply", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def power(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC):
    """Power with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.power", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def atan2(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Atan2 with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor (y-coordinates).
    x2 : relax.Expr
        The second input tensor (x-coordinates).

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.atan2", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def subtract(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Subtraction with numpy-style broadcasting.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.subtract", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def mod(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Modulo with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.
    """
    return _Call("relax.mod", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def floor_mod(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Floor modulo with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.
    """
    return _Call("relax.floor_mod", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


###################### Comparison operators ######################


def equal(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Broadcasted element-wise test for (lhs == rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.equal", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def greater(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Broadcasted element-wise test for (lhs > rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.greater", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def greater_equal(
    x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Broadcasted element-wise test for (lhs >= rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.greater_equal", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def less(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Broadcasted element-wise test for (lhs < rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.less", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def less_equal(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Broadcasted element-wise test for (lhs <= rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.less_equal", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def not_equal(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Broadcasted element-wise test for (lhs != rhs).

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.not_equal", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore


def maximum(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Element-wise maximum

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.maximum", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


def minimum(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Element-wise minimum

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.minimum", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


###################### Logical operators ######################


def logical_and(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Logical AND
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.logical_and", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


def logical_or(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Logical OR
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.logical_or", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


def logical_xor(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Logical XOR
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.logical_xor", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


###################### Bitwise operators ######################


def bitwise_and(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Bitwise AND
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.bitwise_and", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


def bitwise_or(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Bitwise OR
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.bitwise_or", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


def bitwise_xor(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Bitwise XOR
    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.
    x2 : relax.Expr
        The second input tensor.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.bitwise_xor", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


def left_shift(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Bitwise Shift Left
    Parameters
    ----------
    x1 : relax.Expr
        The input tensor to be shifted.
    x2 : relax.Expr
        The number of positions to shift.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.left_shift", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)


def right_shift(x1: Expr, x2: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Bitwise Shift Right
    Parameters
    ----------
    x1 : relax.Expr
        The input tensor to be shifted.
    x2 : relax.Expr
        The number of positions to shift.
    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.right_shift", [x1, x2], ty_args=ty_args, loc=loc, ty=ty)
