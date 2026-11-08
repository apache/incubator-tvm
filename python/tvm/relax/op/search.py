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
"""Search operators."""

import tvm_ffi

from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr


def where(condition: Expr, x1: Expr, x2: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Selecting elements from either the input tensors depending on the value of the
    condition.

    For a given position, return the corresponding value in `x1` if `condition` is True,
    and return the corresponding value in `x2` otherwise.

    Parameters
    ----------
    condition : relax.Expr
        When True, yield `x1`; otherwise, yield `x2`.
        Must be broadcasting compatible with `x1` and `x2`.
        Must have boolean dtype.

    x1 : relax.Expr
        The first input tensor.
        Must be broadcasting compatible with `condition` and `x2`.

    x2 : relax.Expr
        The second input tensor.
        Must be broadcasting compatible with `condition` and `x1`.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _Call("relax.where", [condition, x1, x2], loc=loc, ty=ty)  # type: ignore


@tvm_ffi.register_object("relax.attrs.ArgmaxArgminAttrs")
class ArgmaxArgminAttrs(Attrs):
    """Attributes for argmax/argmin operator"""


def argmax(
    x: Expr,
    axis: int | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the argmax of tensor elements over given axis.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[int]
        Axis along which an argmax operation is performed.
        The default, axis=None, will compute the argmax of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axis being reduced is left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call(
        "relax.argmax",
        [x],
        attrs=_make_attrs("relax.attrs.ArgmaxArgminAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def argmin(
    x: Expr,
    axis: int | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the argmin of tensor elements over given axis.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[int]
        Axis along which an argmin operation is performed.
        The default, axis=None, will compute the argmin of all elements in the
        input tensor. Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axis being reduced is left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call(
        "relax.argmin",
        [x],
        attrs=_make_attrs("relax.attrs.ArgmaxArgminAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def bucketize(
    input_tensor,
    boundaries,
    out_int32=False,
    right=False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    """Returns the indices of the buckets to which each value in the input belongs.

    Parameters
    ----------
    input_tensor : relax.Expr
        N-D tensor containing the search values.

    boundaries : relax.Expr
        1-D tensor, must contain a strictly increasing sequence, or the return value is undefined.

    out_int32 : Optional[bool]
        Indicate the output data type. int32 if True, int64 otherwise. Default=False

    right : Optional[bool]
        Determines the behavior for values in boundaries. Similar to torch.bucketize

    Returns
    -------
    result : relax.Expr
        The computed result with same shape as input_tensor.
    """
    return _Call(
        "relax.bucketize",
        [input_tensor, boundaries],
        attrs=_make_attrs("relax.attrs.BucketizeAttrs", out_int32=out_int32, right=right),
        loc=loc,
        ty=ty,
    )


def searchsorted(
    sorted_sequence,
    values,
    out_int32=False,
    right=False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    """Find indices where elements should be inserted to maintain order.
    Similar to torch.searchsorted.

    Parameters
    ----------
    sorted_sequence : relax.Expr
        N-D or 1-D tensor, containing monotonically increasing sequence on the innermost dimension.

    values : relax.Expr
        N-D tensor containing the search values.

    out_int32 : Optional[bool]
        Indicate the output data type. int32 if True, int64 otherwise. Default=False.

    right : Optional[bool]
        Controls which index is returned if a value lands exactly on one of sorted values.
        Default=False.

    Returns
    -------
    result : relax.Expr
        The computed result with same shape as values.
    """
    return _Call(
        "relax.searchsorted",
        [sorted_sequence, values],
        attrs=_make_attrs("relax.attrs.SearchSortedAttrs", out_int32=out_int32, right=right),
        ty=ty,
        loc=loc,
    )
