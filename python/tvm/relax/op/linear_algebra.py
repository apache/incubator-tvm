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
"""Relax linear algebra operators"""

import tvm_ffi

from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir import PrimType as _PrimType
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr
from ..expr import Tuple as RxTuple
from .manipulate import permute_dims


@tvm_ffi.register_object("relax.attrs.MatmulAttrs")
class MatmulAttrs(Attrs):
    """Attributes for matmul operator"""


def matmul(
    x1: Expr,
    x2: Expr,
    out_dtype: str | tvm_ffi.dtype | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """General matrix multiplication of two tensors, with broadcasting on batched dimensions.

    The semantics and output shape deduction rule is specified as
    https://data-apis.org/array-api/latest/API_specification/generated/array_api.matmul.html.

    Parameters
    ----------
    x1 : relax.Expr
        The first input tensor.

    x2 : relax.Expr
        The second input tensor.

    out_dtype: Optional[Union[str, tvm_ffi.dtype]]
        The data type of the matmul result.
        When it is not specified, the output dtype will be the same as input dtype.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call(
        "relax.matmul",
        [x1, x2],
        attrs=_make_attrs(
            "relax.attrs.MatmulAttrs",
            out_dtype=(out_dtype.dtype if isinstance(out_dtype, _PrimType) else out_dtype),
        ),
        loc=loc,
        ty=ty,
    )  # type: ignore


def linear(
    data: Expr,
    weight: Expr,
    bias: Expr | None = None,
    out_dtype: str | tvm_ffi.dtype | None = None,
) -> Expr:
    """Applies a linear transformation to the incoming data: y = xA^T + b

    Parameters
    ----------
    data : relax.Expr
        The input data.

    weight : relax.Expr
        The weight tensor.

    bias : Optional[Expr]
        The bias tensor.

    out_dtype: Optional[Union[str, tvm_ffi.dtype]]
        The data type of the matmul result.
        When it is not specified, the output dtype will be the same as input dtype.

    Notes
    -----
    Relax does not regard the Linear Op as a primitive Op,
    while combine the transpose, matmul and add op to implement it.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """

    # Since weight can be 1D or 2D, we use `axes=None` to support both cases.
    x = matmul(data, permute_dims(weight, axes=None), out_dtype=out_dtype)
    return x + bias if bias is not None else x


@tvm_ffi.register_object("relax.attrs.EinsumAttrs")
class EinsumAttrs(Attrs):
    """Attributes for einsum operator"""


def einsum(operands, subscripts, *, ty=None, loc: Location = UNKNOWN_LOC):
    """Evaluates the Einstein summation convention on data

    Parameters
    ----------
    operands : Union(List[relax.Expr], Tuple[relax.Expr])
        A list of expression.

    subscripts : str
        The einsum expression string.

    Returns
    -------
    result : relax.Expr
        The output from the einsum op.
    """
    if isinstance(operands, list | tuple):
        operands = RxTuple(operands)

    return _Call(
        "relax.einsum",
        [operands],
        attrs=_make_attrs("relax.attrs.EinsumAttrs", subscripts=subscripts),
        loc=loc,
        ty=ty,
    )  # type: ignore


def outer(x1: Expr, x2: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """
    Computes the outer product of two input expressions.

    Parameters
    ----------
    x1 : relax.Expr
        The first input expression.

    x2 : relax.Expr
        The second input expression.

    Notes
    -----
    This operation computes the outer product between two expressions,
    resulting in a tensor where each element is the product of elements
    from `x1` and `x2`. It is commonly used in tensor and matrix operations
    to expand lower-dimensional inputs into higher-dimensional representations.

    Returns
    -------
    result : relax.Expr
        The resulting expression representing the outer product.
    """
    return _Call("relax.outer", [x1, x2], loc=loc, ty=ty)
