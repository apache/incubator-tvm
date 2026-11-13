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
"""Relax ternary arithmetic operators."""

from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr


def ewise_fma(
    x1: Expr, x2: Expr, x3: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Elementwise fused multiply-add operator
    Returns elementwise result of :math:`x1 * x2 + x3`

    Parameters
    ----------
    x1 : relax.Expr
        The left hand operand of the multiplication

    x2 : relax.Expr
        The right hand operand of the multiplication

    x3 : relax.Expr
        The operand of the addition

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    return _Call("relax.ewise_fma", [x1, x2, x3], ty_args=ty_args, loc=loc, ty=ty)  # type: ignore
