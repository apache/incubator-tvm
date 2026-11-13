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
"""Register TOPI implementations for TE tensor overload hooks."""

from tvm import te
from tvm.te import _te_tensor_overload as _overload
from tvm_tirx.ir import _prim_expr_overload as _expr
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.runtime import DataTypeCode

from . import broadcast as _broadcast
from . import math as _math


def _is_integer(value):
    if isinstance(value, te.Tensor | te.TensorSlice):
        return value.dtype.matches_code(DataTypeCode.INT)
    return _expr._dtype_is_int(value)


def _binary(op, reflected=False, check_integer=False):
    def implementation(lhs, rhs):
        if not isinstance(lhs, te.Tensor) and not isinstance(rhs, te.Tensor):
            return NotImplemented
        if reflected:
            lhs, rhs = rhs, lhs
        if check_integer and _is_integer(lhs) and _is_integer(rhs):
            raise _expr.div_ambiguity_error()
        return op(lhs, rhs)

    return implementation


_overload.__add__ = _binary(_broadcast.add)
_overload.__radd__ = _binary(_broadcast.add, reflected=True)
_overload.__sub__ = _binary(_broadcast.subtract)
_overload.__rsub__ = _binary(_broadcast.subtract, reflected=True)
_overload.__mul__ = _binary(_broadcast.multiply)
_overload.__rmul__ = _binary(_broadcast.multiply, reflected=True)
_overload.__div__ = _binary(_broadcast.divide, check_integer=True)
_overload.__rdiv__ = _binary(_broadcast.divide, reflected=True, check_integer=True)
_overload.__truediv__ = _overload.__div__
_overload.__rtruediv__ = _overload.__rdiv__


def _astype(value, dtype, loc: Location = UNKNOWN_LOC):
    if not isinstance(value, te.Tensor):
        return NotImplemented
    return _math.cast(value, dtype, loc)


_overload.astype = _astype
