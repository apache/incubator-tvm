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
# pylint: disable=invalid-name,unused-argument
"""Default legalization function for unary operators."""

from tvm import te, topi
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr
from .common import _call_topi_without_attr, register_legalize

# To avoid conflict of IRModule function name and libc function name, we add
# "tir_" as the prefix of the generated Function name.
register_legalize("relax.abs", _call_topi_without_attr(topi.abs, "tir_abs"))
register_legalize("relax.acos", _call_topi_without_attr(topi.acos, "tir_acos"))
register_legalize("relax.acosh", _call_topi_without_attr(topi.acosh, "tir_acosh"))
register_legalize("relax.asin", _call_topi_without_attr(topi.asin, "tir_asin"))
register_legalize("relax.asinh", _call_topi_without_attr(topi.asinh, "tir_asinh"))
register_legalize("relax.atan", _call_topi_without_attr(topi.atan, "tir_atan"))
register_legalize("relax.atanh", _call_topi_without_attr(topi.atanh, "tir_atanh"))
register_legalize("relax.bitwise_not", _call_topi_without_attr(topi.bitwise_not, "tir_bitwise_not"))
register_legalize("relax.ceil", _call_topi_without_attr(topi.ceil, "tir_ceil"))
register_legalize("relax.cos", _call_topi_without_attr(topi.cos, "tir_cos"))
register_legalize("relax.cosh", _call_topi_without_attr(topi.cosh, "tir_cosh"))
register_legalize("relax.exp", _call_topi_without_attr(topi.exp, "tir_exp"))
register_legalize("relax.floor", _call_topi_without_attr(topi.floor, "tir_floor"))
register_legalize("relax.isfinite", _call_topi_without_attr(topi.isfinite, "tir_isfinite"))
register_legalize("relax.isinf", _call_topi_without_attr(topi.isinf, "tir_isinf"))
register_legalize("relax.isnan", _call_topi_without_attr(topi.isnan, "tir_isnan"))
register_legalize("relax.log", _call_topi_without_attr(topi.log, "tir_log"))
register_legalize("relax.logical_not", _call_topi_without_attr(topi.logical_not, "tir_logical_not"))
register_legalize("relax.negative", _call_topi_without_attr(topi.negative, "tir_negative"))
register_legalize("relax.round", _call_topi_without_attr(topi.round, "tir_round"))
register_legalize("relax.rsqrt", _call_topi_without_attr(topi.rsqrt, "tir_rsqrt"))
register_legalize("relax.sigmoid", _call_topi_without_attr(topi.sigmoid, "tir_sigmoid"))
register_legalize("relax.sign", _call_topi_without_attr(topi.sign, "tir_sign"))
register_legalize("relax.sin", _call_topi_without_attr(topi.sin, "tir_sin"))
register_legalize("relax.sinh", _call_topi_without_attr(topi.sinh, "tir_sinh"))
register_legalize("relax.square", _call_topi_without_attr(lambda x: x * x, "tir_square"))
register_legalize("relax.sqrt", _call_topi_without_attr(topi.sqrt, "tir_sqrt"))
register_legalize("relax.tan", _call_topi_without_attr(topi.tan, "tir_tan"))
register_legalize("relax.tanh", _call_topi_without_attr(topi.tanh, "tir_tanh"))
register_legalize("relax.trunc", _call_topi_without_attr(topi.trunc, "tir_trunc"))
register_legalize("relax.clip", _call_topi_without_attr(topi.clip, "tir_clip"))


@register_legalize("relax.erf")
def _erf(bb: BlockBuilder, call: Call) -> Expr:
    def te_erf(x: te.Tensor):
        dtype = x.dtype
        if dtype == "float16":
            erf = topi.math.cast(topi.erf(topi.math.cast(x, "float32")), "float16")
        else:
            erf = topi.erf(x)
        return erf

    return bb.call_te(te_erf, call.args[0], function_name_hint="erf")
