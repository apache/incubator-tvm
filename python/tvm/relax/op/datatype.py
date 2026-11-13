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
"""Datatype operators."""

import tvm_ffi

from tvm_tirx.ir import Attrs, PrimType
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr


def _raw_dtype(dtype):
    return dtype.dtype if isinstance(dtype, PrimType) else dtype


@tvm_ffi.register_object("relax.attrs.AstypeAttrs")
class AstypeAttrs(Attrs):
    """Attributes used in astype operator"""


def astype(
    x: Expr, dtype: str | tvm_ffi.dtype | PrimType, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Cast input tensor to the given data type.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    dtype: Union[str, tvm_ffi.dtype]
        The target data type

    Returns
    -------
    result : relax.Expr
        The casted result.
    """
    return _Call(
        "relax.astype",
        [x],
        attrs=_make_attrs("relax.attrs.AstypeAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.WrapParamAttrs")
class WrapParamAttrs(Attrs):
    """Attributes used in wrap_param operator"""


def wrap_param(
    data: Expr,
    dtype: str | tvm_ffi.dtype | PrimType = "float32",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Cast input tensor which is model param to data type if the dtype of the input data is not
    the same as the given dtype.
    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.
    dtype : Union[str, tvm_ffi.dtype]
        The target data type
    Returns
    -------
    result : relax.Expr
        The casted result.
    """
    return _Call(
        "relax.wrap_param",
        [data],
        attrs=_make_attrs("relax.attrs.WrapParamAttrs", dtype=_raw_dtype(dtype)),
        loc=loc,
        ty=ty,
    )  # type: ignore
