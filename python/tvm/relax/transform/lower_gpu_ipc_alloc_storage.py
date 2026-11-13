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
"""Lower the storage/tensor allocation on IPC memory.
The pass is written in Python for experiment, fast development.
"""

import tvm_tirx
from tvm import relax
from tvm.relax.expr import Expr
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm_tirx.ir.module import IRModule


@tvm_tirx.ir.transform.module_pass(opt_level=0, name="LowerGPUIPCAllocStorage")
class LowerGPUIPCAllocStorage:
    """Lower the storage/tensor allocation on IPC memory."""

    def transform_module(self, mod: IRModule, _ctx: tvm_tirx.ir.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        return _Rewriter(mod).transform()


@mutator
class _Rewriter(PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod = mod
        self.memory_alloc_storage_op = tvm_tirx.ir.Op.get("relax.memory.alloc_storage")
        self.memory_alloc_tensor_op = tvm_tirx.ir.Op.get("relax.memory.alloc_tensor")
        self.builtin_alloc_tensor_op = tvm_tirx.ir.Op.get("relax.builtin.alloc_tensor")

    def transform(self) -> IRModule:
        """Entry point"""
        for g_var, func in self.mod.functions_items():
            if isinstance(func, relax.Function):
                updated_func = self.visit_expr(func)
                self.builder_.update_func(g_var, updated_func)
        return self.builder_.get()

    def visit_call_(self, call: relax.Call) -> Expr:  # pylint: disable=arguments-renamed
        if call.op == self.memory_alloc_storage_op and call.args[2].value == "ipc_memory":
            return self.rewrite_alloc_storage(call)
        elif call.op == self.builtin_alloc_tensor_op and call.args[3].value == "ipc_memory":
            return self.rewrite_alloc_tensor(call)
        else:
            return call

    def rewrite_alloc_storage(self, call: relax.Call) -> relax.Call:
        shape = call.args[0]
        dtype = call.args[3]
        return relax.Call(
            relax.ExternFunc("runtime.disco.cuda_ipc.alloc_storage"),
            args=[shape, dtype],
            ty_args=[call.ty],
        )

    def rewrite_alloc_tensor(self, call: relax.Call) -> relax.Call:
        shape = call.args[0]
        dtype = call.args[1]
        ipc_alloc_storage = relax.Call(
            relax.ExternFunc("runtime.disco.cuda_ipc.alloc_storage"),
            args=[shape, dtype],
            ty_args=[relax.AnyType()],
        )
        return relax.Call(
            self.memory_alloc_tensor_op,
            args=[ipc_alloc_storage, call.args[2], shape, dtype, relax.prim_value(0)],
            ty_args=call.ty_args,
        )
