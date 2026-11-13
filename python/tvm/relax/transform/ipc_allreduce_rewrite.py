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
"""Rewrite all-reduce operation to customized all-reduce impl with IPC memory.
The pass is written in Python for experiment, fast development.
"""

import tvm_ffi

import tvm_tirx
from tvm import relax
from tvm.relax.expr import Expr, Var
from tvm.relax.expr_functor import PyExprMutator, PyExprVisitor, mutator, visitor
from tvm_tirx.ir.module import IRModule


@tvm_tirx.ir.transform.module_pass(opt_level=0, name="IPCAllReduceRewrite")
class IPCAllReduceRewrite:
    """Rewrite all-reduce operation to customized all-reduce impl with IPC memory."""

    def __init__(self, allreduce_strategy: int) -> None:
        """Constructor

        Parameters
        ----------
        allreduce_strategy : int
            The all-reduce strategy. Only "1" and "2" are supported.
            "1" stands for one-shot, and "2" stands for two-shot.
        """
        self.allreduce_strategy = allreduce_strategy

    def transform_module(self, mod: IRModule, _ctx: tvm_tirx.ir.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        fcustom_allreduce = tvm_ffi.get_global_func(
            "runtime.disco.cuda_ipc.custom_allreduce", allow_missing=True
        )
        if fcustom_allreduce is None:
            # Customized allreduce is not available.
            return mod

        binding_replacement_map = _Visitor(self.allreduce_strategy).visit(mod)
        return _Rewriter(mod, binding_replacement_map).transform()


@visitor
class _Visitor(PyExprVisitor):  # pylint: disable=abstract-method
    def __init__(self, allreduce_strategy: int) -> None:
        self.allreduce_strategy = allreduce_strategy
        self.alloc_map: dict[Var, relax.Call] = {}
        self.binding_replacement_map: dict[relax.Expr, relax.Expr] = {}
        self.builtin_alloc_tensor_op = tvm_tirx.ir.Op.get("relax.builtin.alloc_tensor")
        self.reshape_op = tvm_tirx.ir.Op.get("relax.reshape")

    def visit(self, mod: IRModule) -> dict[relax.Expr, relax.Expr]:
        """Entry point"""
        for _, func in mod.functions_items():
            if isinstance(func, relax.Function):
                self.alloc_map.clear()
                self.visit_expr(func)
        return self.binding_replacement_map

    def visit_var_binding_(self, binding: relax.VarBinding):
        super().visit_var_binding_(binding)
        if (
            isinstance(binding.value, relax.Call)
            and binding.value.op == self.builtin_alloc_tensor_op
        ):
            self.alloc_map[binding.var] = binding.value
        elif isinstance(binding.value, relax.Var) and binding.value in self.alloc_map:
            self.alloc_map[binding.var] = self.alloc_map[binding.value]
        elif (
            isinstance(binding.value, relax.Call)
            and binding.value.op == self.reshape_op
            and binding.value.args[0] in self.alloc_map
        ):
            self.alloc_map[binding.var] = self.alloc_map[binding.value.args[0]]

    def visit_call_(self, call: relax.Call) -> None:  # pylint: disable=arguments-renamed
        if (
            not isinstance(call.op, relax.ExternFunc)
            or call.op.global_symbol != "runtime.disco.allreduce"
            or call.args[1].values[0] != 0
        ):
            # Return if the call is not a summation all-reduce.
            return

        assert len(call.args) == 4
        allreduce_input, _strategy, _ingroup, allreduce_output = call.args
        alloc_tensor = self.alloc_map.get(allreduce_input, None)
        if alloc_tensor is None or alloc_tensor.args[3].value != "global":
            # Return if the allocation of all-reduce input is not recorded,
            # or the scope of the allocation is not global.
            return

        # Set the scope of the alloc_tensor to IPC memory.
        alloc_tensor = self.alloc_map[allreduce_input]
        self.binding_replacement_map[alloc_tensor] = relax.op.builtin.alloc_tensor(
            alloc_tensor.args[0],
            alloc_tensor.args[1],
            alloc_tensor.args[2],
            tvm_tirx.ir.StringImm("ipc_memory"),
        )

        self.binding_replacement_map[call] = relax.Call(
            relax.ExternFunc("runtime.disco.cuda_ipc.custom_allreduce"),
            # The "cuda_ipc.custom_allreduce" implementation does not
            # yet support num_groups>1, and therefore does not use the
            # `in_group` argument.
            [allreduce_input, relax.prim_value(self.allreduce_strategy), allreduce_output],
        )


@mutator
class _Rewriter(PyExprMutator):
    """Rewrite the IRModule according to the binding replacement provided by the visitor."""

    def __init__(
        self, mod: IRModule, binding_replacement_map: dict[relax.Expr, relax.Expr]
    ) -> None:
        super().__init__(mod)
        self.mod = mod
        self.binding_replacement_map = binding_replacement_map

    def transform(self) -> IRModule:
        """Entry point"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                updated_func = self.visit_expr(func)
                self.builder_.update_func(g_var, updated_func)
        return self.builder_.get()

    def visit_call_(self, call: relax.Call) -> Expr:  # pylint: disable=arguments-renamed
        return (
            super().visit_call_(self.binding_replacement_map[call])
            if call in self.binding_replacement_map
            else super().visit_call_(call)
        )
