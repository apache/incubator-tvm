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
"""Common functionality for legalization."""

from collections.abc import Callable

import tvm_tirx
from tvm import te
from tvm_tirx.ir import Call, FloatImm, GenericConst, IntImm
from tvm_tirx.runtime import DataTypeCode

from ...block_builder import BlockBuilder
from ...expr import Expr

##################### Types #####################


# The function type of a TE function, which accepts TE Tensors and
# other attributes, and returns the output TE Tensor.
TEFunc = Callable[..., te.Tensor]

# The function type of a legalization function, which takes a
# BlockBuilder and the Call to be legalized, and outputs the legalization
# result Expr.
LegalizeFunc = Callable[[BlockBuilder, Call], Expr]


def _is_relax_expr(expr: object) -> bool:
    return isinstance(expr, Expr) and not tvm_tirx.ir.is_prim_expr(expr)


def _try_convert_to_scalar_const(
    expr: Expr, python_native: bool = False
) -> Expr | FloatImm | IntImm | bool | float | int:
    """Check if the input Expr is a scalar constant.
    If it is, return its plain value with the same data type or in native python type.
    If it is not, return the input expr.

    Note that if the python_native flag is True, the returned value will be in native python type,
    this might cause loss of data type for example, a float16 constant will be converted to float32
    and a int64 constant will be converted to int32.

    Parameters
    ----------
    expr : Expr
        The expr to be checked and converted.

    Returns
    -------
    ret : Union[Expr, FloatImm, IntImm, bool, float, int]
        Return a FloatImm or IntImm if the given expr is a scalar integer or float constant, and the
        python native flag is False. Or return the plain value of the constant in native python type
        if the python native flag is True.
        Or return the input itself if it is not a scalar constant.
    """
    if (
        isinstance(expr, GenericConst)
        and isinstance(expr.value, tvm_tirx.runtime.Tensor)
        and expr.ty.ndim == 0
    ):
        # get the value of the scalar constant
        value = expr.value.numpy()[()].item()
        dtype = expr.ty.dtype
        dtype_str = str(dtype.dtype)
        if python_native:
            return value
        # preserve the data type of the constant
        if dtype.matches_code(DataTypeCode.FLOAT, DataTypeCode.BFLOAT):
            return tvm_tirx.ir.FloatImm(dtype_str, value)
        elif dtype.matches_code(DataTypeCode.INT, DataTypeCode.UINT, DataTypeCode.BOOL):
            return tvm_tirx.ir.IntImm(dtype_str, value)
    return expr


def _call_topi_without_attr(te_func: TEFunc, function_name: str | None = None) -> LegalizeFunc:
    """A common wrapper util for the ops who has no attributes and whose
    legalization is simply passing its arguments to some TE function.

    Parameters
    ----------
    te_func : TEFunc
        The input TE function which is to be converted to Function.

    function_name : Optional[str]
        The name of the generated Function.
        If it is not specified, the name of `te_func` will be used by default.

    Returns
    -------
    func : LegalizeFunc
        The legalization wrapper function, which wraps the input TE function.
    """
    if function_name is None:
        function_name = te_func.__name__
    return lambda bb, call: bb.call_te(te_func, *call.args, function_name_hint=function_name)


##################### Decorators #####################

_LEGALIZE_ATTR_NAME = "FLegalize"


def register_legalize(op_name: str, legal_func: LegalizeFunc = None):
    """Register legal transformation function for a Relax op.

    Parameters
    ----------
    op_name : str
        The name of the operator

    legal_func: function (bb: BlockBuilder, call: Call) -> new_expr: Expr
        The function for transforming an expr to another expr.
    """
    return tvm_tirx.ir.register_op_attr(op_name, _LEGALIZE_ATTR_NAME, legal_func)
