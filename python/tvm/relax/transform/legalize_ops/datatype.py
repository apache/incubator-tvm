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
"""Default legalization function for datatype operators."""

from tvm import relax, topi
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr
from .common import _is_relax_expr, _try_convert_to_scalar_const, register_legalize


@register_legalize("relax.astype")
def _astype(bb: BlockBuilder, call: Call) -> Expr:
    arg = _try_convert_to_scalar_const(call.args[0], python_native=True)
    if _is_relax_expr(arg):
        return bb.call_te(topi.cast, arg, call.attrs.dtype)
    else:
        return relax.const(arg, call.attrs.dtype)
