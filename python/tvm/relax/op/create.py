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
"""Creation operators."""

import tvm_ffi

from tvm_tirx import DataTypeCode
from tvm_tirx.ir import Attrs, PrimType, is_prim_expr
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr, ShapeExpr, prim_value

PrimExprLike = int | Expr


def _raw_dtype(dtype):
    return dtype.dtype if isinstance(dtype, PrimType) else dtype


def _normalize_shape(shape):
    if isinstance(shape, tuple | list):
        return ShapeExpr(shape)
    if not isinstance(shape, Expr) or is_prim_expr(shape):
        raise TypeError("shape must be a tuple/list or a Relax shape expression")
    return shape


@tvm_ffi.register_object("relax.attrs.InitAttrs")
class InitAttrs(Attrs):
    """Attributes used in full/full_like, ones/ones_like, and zeros/zeros_like operator"""


def full(
    shape: tuple[PrimExprLike] | Expr,
    fill_value: Expr,
    dtype: str | tvm_ffi.dtype | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Fill array with scalar value.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    fill_value : relax.Expr
        The value to fill. Must be a scalar tensor.

    dtype : Optional[str | tvm_ffi.dtype]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of fill_value.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _normalize_shape(shape)
    return _Call(
        "relax.full",
        [shape, fill_value],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def full_like(
    x: Expr,
    fill_value: Expr,
    dtype: str | tvm_ffi.dtype | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Construct a tensor such that
    - its shape is the same as the input data tensor's shape,
    - its value is filled with the input scalar fill value.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    fill_value : relax.Expr
        The value to fill. Must be a scalar tensor.

    dtype : Optional[str | tvm_ffi.dtype]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _Call(
        "relax.full_like",
        [x, fill_value],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def ones(
    shape: tuple[PrimExprLike] | Expr,
    dtype: str | tvm_ffi.dtype,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Construct a tensor of all ones, with the input shape and dtype.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : str | tvm_ffi.dtype
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _normalize_shape(shape)
    return _Call(
        "relax.ones",
        [shape],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def ones_like(
    x: Expr, dtype: str | tvm_ffi.dtype | None = None, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Construct a tensor with all ones, with shape of the input tensor shape.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    dtype : Optional[str | tvm_ffi.dtype]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _Call(
        "relax.ones_like",
        [x],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def zeros(
    shape: tuple[PrimExprLike] | Expr,
    dtype: str | tvm_ffi.dtype,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Construct a tensor of all zeros, with the input shape and dtype.

    Parameters
    ----------
    shape : Union[Tuple[PrimExprLike], Expr]
        The shape of the created tensor.

    dtype : str | tvm_ffi.dtype
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    shape = _normalize_shape(shape)
    return _Call(
        "relax.zeros",
        [shape],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def zeros_like(
    x: Expr, dtype: str | tvm_ffi.dtype | None = None, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Construct a tensor with all zeros, with shape of the input tensor shape.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    dtype : Optional[str | tvm_ffi.dtype]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    return _Call(
        "relax.zeros_like",
        [x],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def eye(
    n: PrimExprLike,
    m: PrimExprLike | None = None,
    k: PrimExprLike = 0,
    dtype: str | tvm_ffi.dtype = "float32",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Construct a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : PrimExprLike
        Number of rows in the output.

    m : Optional[PrimExprLike]
        Number of columns in the output. If None, defaults to n.

    k : PrimExprLike
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.

    dtype : str | tvm_ffi.dtype
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    m = n if m is None else m
    n = prim_value(n)
    m = prim_value(m)
    k = prim_value(k)
    return _Call(
        "relax.eye",
        [n, m, k],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def eye_like(
    x: Expr,
    k: PrimExprLike = 0,
    dtype: str | tvm_ffi.dtype | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Return a 2-D tensor with ones on the diagonal and zeros elsewhere,
    with the same shape as the input tensor.

    Parameters
    ----------
    x : relax.Expr
        The input tensor, which provides the shape, and dtype
        when the `dtype` field is not specified.

    k : PrimExprLike
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.

    dtype : Optional[str | tvm_ffi.dtype]
        The data type of the created tensor.
        If dtype is not given, it will by default use the dtype of the input tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    k = prim_value(k)
    return _Call(
        "relax.eye_like",
        [x, k],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def arange(
    start: PrimExprLike,
    end: PrimExprLike | None = None,
    step: PrimExprLike = 1,
    dtype: str | tvm_ffi.dtype | PrimType | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Construct a tensor with evenly spaced elements.

    Parameters
    ----------
    start : PrimExprLike
        The start of the interval.

    end : Optional[PrimExprLike]
        The end of the interval. If not given, it will be set to start,
        and start will be set to 0.

    step : PrimExprLike
        The step size.

    dtype : Optional[str | tvm_ffi.dtype | PrimType]
        The data type of the created tensor.

    Returns
    -------
    result : relax.Expr
        The result tensor.
    """
    if end is None:
        end = start
        start = 0

    def is_int(expr):
        if isinstance(expr, int):
            return True
        if is_prim_expr(expr):
            return expr.ty.matches_code(DataTypeCode.INT)
        return False

    if dtype is None:
        args = (start, end, step)
        integer_args = all(is_int(arg) for arg in args)
        dtype = "int64" if integer_args else "float32"

    start = prim_value(start)
    end = prim_value(end)
    step = prim_value(step)
    return _Call(
        "relax.arange",
        [start, end, step],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


def hamming_window(
    window_size, periodic, alpha, beta, dtype, *, ty=None, loc: Location = UNKNOWN_LOC
):
    """Hamming window function.

    Parameters
    ----------
    window_size : Expr
        The size of returned window.

    periodic : Expr
        If True, returns a window to be used as periodic function.
        If False, return a symmetric window.

    alpha : Expr
        The co-efficient alpha.

    beta : Expr
        The co-efficient beta.

    Returns
    -------
    ret : relax.Expr
        The result tensor.
    """
    if not is_prim_expr(window_size):
        window_size = prim_value(window_size)
    if not is_prim_expr(periodic):
        periodic = prim_value(periodic)
    if not is_prim_expr(alpha):
        alpha = prim_value(alpha)
    if not is_prim_expr(beta):
        beta = prim_value(beta)

    return _Call(
        "relax.hamming_window",
        [window_size, periodic, alpha, beta],
        attrs=_make_attrs("relax.attrs.InitAttrs", dtype=dtype),
        loc=loc,
        ty=ty,
    )


@tvm_ffi.register_object("relax.attrs.TriluAttrs")
class TriluAttrs(Attrs):
    """Attributes used in tril and triu operator"""


def tril(x: Expr, k: int | Expr = 0, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Return the lower triangular part of a matrix or a batch of matrices.

    Parameters
    ----------
    x : relax.Expr
        The tensor that tril will be applied to.
        It is required to have at least two dimensions.

    k : int
        The index indicating the diagonal above which to zero elements.
        If k = 0, the diagonal is the main diagonal.
        If k < 0, the diagonal is below the main diagonal.
        If k > 0, the diagonal is above the main diagonal.

    Returns
    -------
    ret : relax.Expr
        The result tensor.
    """
    if not is_prim_expr(k):
        k = prim_value(k)

    return _Call("relax.tril", [x, k], loc=loc, ty=ty)  # type: ignore


def triu(x: Expr, k: int | Expr = 0, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Return the upper triangular part of a matrix or a batch of matrices.

    Parameters
    ----------
    x : relax.Expr
        The tensor that triu will be applied to.
        It is required to have at least two dimensions.

    k : int
        The index indicating the diagonal below which to zero elements.
        If k = 0, the diagonal is the main diagonal.
        If k < 0, the diagonal is below the main diagonal.
        If k > 0, the diagonal is above the main diagonal.

    Returns
    -------
    ret : relax.Expr
        The result tensor.
    """
    if not is_prim_expr(k):
        k = prim_value(k)

    return _Call("relax.triu", [x, k], loc=loc, ty=ty)  # type: ignore
