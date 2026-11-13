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
# pylint: disable=invalid-name, unused-argument, redefined-argument-from-local
"""Relax Use Fast Math pass."""

import tvm
import tvm_tirx
from tvm import topi
from tvm.relax import Call, Expr, PyExprMutator, expr_functor
from tvm_tirx.ir.module import IRModule


@expr_functor.mutator
class FastMathCodeGenerator(PyExprMutator):
    """
    Converts the expensive non linear functions to their fast but approximate counterparts.

    Parameters
    ----------
    mod: IRModule
        The module to be transformed
    """

    def __init__(self, mod):
        super().__init__(mod)

    def visit_call_(self, call: Call) -> Expr:
        if call.op.name == "relax.nn.softmax":
            return self.builder_.call_te(topi.nn.fast_softmax, call.args[0], call.attrs.axis)
        if call.op.name == "relax.exp":
            return self.builder_.call_te(topi.fast_exp, call.args[0])
        if call.op.name == "relax.erf":
            return self.builder_.call_te(topi.fast_erf, call.args[0])
        if call.op.name == "relax.tanh":
            return self.builder_.call_te(topi.fast_tanh, call.args[0])

        return super().visit_call_(call)


@tvm_tirx.ir.transform.module_pass(opt_level=0, name="FastMathTransform")
class FastMathTransform:
    """
    Pass to convert the expensive non linear functions to their fast but approximate counterparts.
    """

    def transform_module(self, mod: IRModule, ctx: tvm_tirx.ir.transform.PassContext) -> IRModule:
        fast_math_codegen = FastMathCodeGenerator(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, tvm.relax.Function):
                func = fast_math_codegen.visit_expr(func)
                fast_math_codegen.builder_.update_func(gv, func)
        return fast_math_codegen.builder_.get()
