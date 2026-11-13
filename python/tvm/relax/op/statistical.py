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
# pylint: disable=redefined-builtin
"""Statistical operators."""

import tvm_ffi

from tvm_tirx.ir import Attrs, PrimType
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr


def _raw_dtype(dtype):
    return dtype.dtype if isinstance(dtype, PrimType) else dtype


@tvm_ffi.register_object("relax.attrs.StatisticalAttrs")
class StatisticalAttrs(Attrs):
    """Attributes used in statistical operator"""


def max(
    x: Expr,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the max of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a max operation is performed.
        The default, axis=None, will compute the max of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.max",
        [x],
        attrs=_make_attrs("relax.attrs.StatisticalAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def mean(
    x: Expr,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the mean of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a mean operation is performed.
        The default, axis=None, will compute the mean of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.mean",
        [x],
        attrs=_make_attrs("relax.attrs.StatisticalAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def min(
    x: Expr,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the min of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a min operation is performed.
        The default, axis=None, will compute the min of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.min",
        [x],
        attrs=_make_attrs("relax.attrs.StatisticalAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def prod(
    x: Expr,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the product of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a product is performed.
        The default, axis=None, will compute the product of all elements of the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.prod",
        [x],
        attrs=_make_attrs("relax.attrs.StatisticalAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def std(
    x: Expr,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the standard deviation of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a standard deviation is performed.
        The default, axis=None, will compute the std of all elements of the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.std",
        [x],
        attrs=_make_attrs("relax.attrs.StatisticalAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def sum(
    x: Expr,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the sum of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a sum is performed.
        The default, axis=None, will sum all of the elements of the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.sum",
        [x],
        attrs=_make_attrs("relax.attrs.StatisticalAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.ScanopAttrs")
class ScanopAttrs(Attrs):
    """Attributes for scan operators"""


def cumprod(
    data: Expr,
    axis: int | None = None,
    dtype: str | tvm_ffi.dtype | None = None,
    exclusive: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    """Numpy style cumprod op. Return the cumulative product of the elements along
    a given axis.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis : Optional[int]
        Axis along which the cumulative product is computed. The default (None) is to compute
        the cumprod over the flattened array.

    dtype : Optional[Union[str, tvm_ffi.dtype]]
        Type of the returned array and of the accumulator in which the elements are computed.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : bool
        If false (default), all elements are included in the product.  If
        true, the first element is excluded from the product.

    Returns
    -------
    result : relax.Expr
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.

    Examples
    --------
    .. code-block:: python

        a = [[1, 2, 3], [4, 5, 6]]

        cumprod(a)  # if axis is not provided, cumprod is done over the flattened input.
        -> [ 1,  2,  6, 24, 120, 720]

        cumprod(a, dtype="float32")
        -> [  1.,  2.,  6., 24., 120., 720.]

        cumprod(a, axis=0)  # multiply over rows for each of the 3 columns
        -> [[1, 2, 3],
            [4, 10, 18]]

        cumprod(a, axis=1)
        -> [[ 1,  2,  6],
            [ 4,  20, 120]]

        a = [1, 1, 1, 0, 1, 1, 0]  # a is a boolean array
        cumprod(a, dtype=int32)  # dtype should be provided to get the expected results
        -> [1, 1, 1, 0, 0, 0, 0]
    """
    if exclusive is None:
        exclusive = False

    return _Call(
        "relax.cumprod",
        [data],
        attrs=_make_attrs(
            "relax.attrs.ScanopAttrs", axis=axis, dtype=_raw_dtype(dtype), exclusive=exclusive
        ),
        loc=loc,
        ty=ty,
    )  # type: ignore


def cumsum(
    data: Expr,
    axis: int | None = None,
    dtype: str | tvm_ffi.dtype | None = None,
    exclusive: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    """Numpy style cumsum op. Return the cumulative inclusive sum of the elements along
    a given axis.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis : Optional[int]
        Axis along which the cumulative sum is computed. The default (None) is to compute
        the cumsum over the flattened array.

    dtype : Optional[Union[str, tvm_ffi.dtype]]
        Type of the returned array and of the accumulator in which the elements are summed.
        If dtype is not specified, it defaults to the dtype of data.

    exclusive : bool
        If false (default), all elements are included in the sum.  If
        true, the first element is excluded from the sum.

    Returns
    -------
    result : relax.Expr
        The result has the same size as data, and the same shape as data if axis is not None.
        If axis is None, the result is a 1-d array.

    Examples
    --------
    .. code-block:: python

        a = [[1, 2, 3], [4, 5, 6]]

        cumsum(a)  # if axis is not provided, cumsum is done over the flattened input.
        -> [ 1,  3,  6, 10, 15, 21]

        cumsum(a, dtype="float32")
        -> [  1.,   3.,   6.,  10.,  15.,  21.]

        cumsum(a, axis=0)  # sum over rows for each of the 3 columns
        -> [[1, 2, 3],
            [5, 7, 9]]

        cumsum(a, axis=1)
        -> [[ 1,  3,  6],
            [ 4,  9, 15]]

        a = [1, 0, 1, 0, 1, 1, 0]  # a is a boolean array
        cumsum(a, dtype=int32)  # dtype should be provided to get the expected results
        -> [1, 1, 2, 2, 3, 4, 4]
    """
    if exclusive is None:
        exclusive = False

    return _Call(
        "relax.cumsum",
        [data],
        attrs=_make_attrs(
            "relax.attrs.ScanopAttrs", axis=axis, dtype=_raw_dtype(dtype), exclusive=exclusive
        ),
        loc=loc,
        ty=ty,
    )  # type: ignore


def variance(
    x: Expr,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the variance of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis or axes along which a variance operation is performed.
        The default, axis=None, will compute the variance of all elements in the input tensor.
        Negative indexing is supported.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.variance",
        [x],
        attrs=_make_attrs("relax.attrs.StatisticalAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def median(
    x: Expr,
    axis: int | list[int] | None = None,
    keepdims: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Computes the median of tensor elements over given axes.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor

    axis : Optional[Union[int, List[int]]]
        Axis along which the median is computed. The default (None) is to compute
        the median of the entire flattened tensor.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input tensor.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.median",
        [x],
        attrs=_make_attrs("relax.attrs.StatisticalAttrs", axis=axis, keepdims=keepdims),
        loc=loc,
        ty=ty,
    )  # type: ignore
