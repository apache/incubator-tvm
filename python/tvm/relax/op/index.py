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
"""Indexing operators."""

import tvm_ffi

from tvm.relax.expr import Tuple as _Tuple
from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr
from ..utils import convert_to_expr

PrimExprLike = int | Expr


@tvm_ffi.register_object("relax.attrs.TakeAttrs")
class TakeAttrs(Attrs):
    """Attributes used in take operator"""


def take(
    x: Expr,
    indices: Expr,
    axis: int | None = None,
    mode: str = "fast",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Take elements from a tensor along an axis.
    Its semantic is mostly similar to `numpy.take`
    (https://numpy.org/doc/stable/reference/generated/numpy.take.html),
    which can cover `torch.take` (https://pytorch.org/docs/stable/generated/torch.take.html) and
    `onnx.gather` (https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13).

    Parameters
    ----------
    x : relax.Expr
        The source tensor.

    indices : relax.Expr
        The indices of the values to extract.

    axis : Optional[int]
        The axis over which to select values.
        If it is none, the input tensor is required to be one-dimensional.

    mode : str
        Specifies how out-of-bounds indices will behave.
        - fast (default): extra indices lead to seg fault (user must make sure indices are in-bound)
        - nan: produce NaNs for out-of-bounds indices
        - wrap: wrap around the indices
        - clip: clip to the range
    Returns
    -------
    ret : relax.Expr
        The taken result.
    """
    return _Call(
        "relax.take",
        [x, indices],
        attrs=_make_attrs("relax.attrs.TakeAttrs", axis=axis, mode=mode),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.StridedSliceAttrs")
class StridedSliceAttrs(Attrs):
    """Attributes used in strided_slice operator"""


def strided_slice(
    x: Expr,
    axes: Expr,
    begin: Expr,
    end: Expr,
    strides: Expr | None = None,
    assume_inbound: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Strided slice of a tensor.

    Parameters
    ----------
    x : relax.Expr
        The source tensor to be sliced.

    axes : List[int]
        Axes along which slicing is applied.

    begin : List[PrimExprLike]
        The indices to begin with in the slicing, inclusive.

    end : List[PrimExprLike]
        The indices indicating end of the slice, exclusive.

    strides : Optional[List[PrimExprLike]]
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.
        If not specified, it by default is an list of ones of the same length as `axes`.

    assume_inbound : bool
        Whether to assume the indices are in bound. If it is set to false,
        out of bound indices will be clipped to the bound.
    Returns
    -------
    ret : relax.Expr
        The sliced result.

    Note
    ----
    strided_slice require the input `begin`, `end` and `strides` to have the
    same length as `axes`.
    """
    axes = convert_to_expr(axes)
    begin = convert_to_expr(begin)
    end = convert_to_expr(end)
    if strides is not None:
        strides = convert_to_expr(strides)
    lengths = [
        len(value.fields) for value in (axes, begin, end, strides) if isinstance(value, _Tuple)
    ]
    if lengths and any(length != lengths[0] for length in lengths):
        raise RuntimeError("axes, begin, end, and strides must have the same length")
    return _Call(
        "relax.strided_slice",
        [x, axes, begin, end, *([] if strides is None else [strides])],
        attrs=_make_attrs("relax.attrs.StridedSliceAttrs", assume_inbound=assume_inbound),
        loc=loc,
        ty=ty,
    )  # type: ignore


def dynamic_strided_slice(
    x: Expr, begin: Expr, end: Expr, strides: Expr, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Dynamic strided slice of a tensor. `begin`, `end`, `strides` can be computed at runtime.

    Parameters
    ----------
    x : Expr
        The source tensor to be sliced.

    begin : Expr
        The indices to begin with in the slicing, inclusive.

    end : Expr
        The indices indicating end of the slice, exclusive.

    strides : Expr
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.
        If not specified, it by default is an list of ones of the same length as `axes`.

    Returns
    -------
    ret : relax.Expr
        The sliced result.

    Note
    ----
    dyn_strided_slice require the input `begin`, `end` and `strides` to have the
    same length as rank of `data` tensor.
    """
    return _Call(
        "relax.dynamic_strided_slice",
        [x, begin, end, strides],
        loc=loc,
        ty=ty,
    )  # type: ignore
