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
# pylint: disable=missing-docstring, invalid-name, unnecessary-comprehension, unused-argument

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.contrib.hexagon import hexagon_unary_ops


def op_replace(call_node, func) -> bool:
    if not isinstance(call_node, relax.Call):
        return False
    call_tir_op = tvm_tirx.ir.Op.get("relax.call_tir")
    if call_node.op != call_tir_op:
        return False
    ops = [
        "qnn.tanh",
        "qnn.sqrt",
        "qnn.rsqrt",
        "qnn.exp",
        "qnn.erf",
        "qnn.sigmoid",
        "qnn.hardswish",
        "qnn.log",
        "qnn.abs",
    ]
    if func.attrs["op_attrs"]["op_name"] in ops:
        return True
    return False


@relax.expr_functor.mutator
class Tanh2TakeReplace(tvm.relax.PyExprMutator):
    def __init__(self, mod: tvm_tirx.IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod

    def transform(self) -> tvm_tirx.IRModule:
        # Iterate over all the nodes to check for the node replaceable
        for global_var, func in self.mod_.functions.items():
            # Skip non-relax functions
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.normalize(updated_func)
            self.builder_.update_func(global_var, updated_func)
        # At the end of the transformation we return the updated IRModule from the BlockBuilder.
        return self.builder_.get()

    def visit_call_(self, call_node: relax.Call) -> relax.Call:
        call_tir_op = tvm_tirx.ir.Op.get("relax.call_tir")
        if call_node.op != call_tir_op:
            return call_node

        var = call_node.args[0]
        func = self.mod_[var]

        if call_node.args[1][0].ty.dtype == "uint8":
            if op_replace(call_node, func):
                inp, inp_scale, inp_zp, out_scale, out_zp = [x for x in call_node.args[1]]
                # LUT node creation
                LUT = hexagon_unary_ops.LUT_generation(
                    inp_scale, inp_zp, out_scale, out_zp, call_node.args[0].name_hint
                )
                # Take operation node creation
                take_func = hexagon_unary_ops.generate_take_function(inp, call_node.ty)
                take_func = take_func.without_attr("global_symbol")
                take_func_gv = self.builder_.add_func(take_func, "take")
                take_node = relax.call_tir(
                    take_func_gv,
                    relax.expr.Tuple(
                        [call_node.args[1][0], relax.const(tvm_tirx.runtime.tensor(LUT))]
                    ),
                    ty_args=[call_node.ty],
                )
                return take_node
        return call_node


@tvm_tirx.ir.transform.module_pass(opt_level=2, name="replace_tanh_take")
class PassReplaceWithTakeOpFunctions:
    def transform_module(self, mod, ctx):
        return Tanh2TakeReplace(mod).transform()
