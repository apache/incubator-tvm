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
# pylint: disable=abstract-method,unused-argument
# pylint: disable=missing-function-docstring,missing-module-docstring

import tvm_ffi

import tvm_tirx
from tvm_tirx.ir import Call, Op
from tvm_tirx.ir.module import IRModule

from ..expr import Expr, Function, ShapeExpr
from ..expr_functor import PyExprVisitor, visitor


def estimate_memory_usage(mod: IRModule | Function) -> str:
    """Analysis function that estimates the memory usage of Relax functions
    in an IRModule. The estimation includes the total memory size needed to
    be allocated before and after memory planning.

    The result might be over-estimated, as the estimation is static, which
    does not consider control flows (such as "if" and cross-function calls).
    It simply accumulates the size of every alloc_tensor and alloc_storage.

    This analysis function is used to demonstrate the effect of memory
    planning.

    Parameters
    ----------
    mod : Union[IRModule, Function]
        The input IRModule whose functions inside are to be analyzed.
        If the input is a Function, we will wrap it with a IRModule, with
        the function named "main".

    Returns
    -------
    est : str
        The estimation information, in the form of a string.

    Notes
    -----
    We regards "relax.memory.alloc_tensor/storage" as the results produced by memory planning.
    """

    @visitor
    class MemoryEstimator(PyExprVisitor):
        """The IR visitor which estimates the memory usage of each Relax function.

        Attributes
        ----------
        total_alloc_tensor_mem : int
            The total memory size of alloc_tensor, in bytes.

        total_const_size_tensor_num : int
            The number of constant-size tensors.

        total_dyn_size_tensor_num : int
            The number of dynamic-size tensors.

        planned_alloc_mem : int
            The total memory size of memory.alloc_storage after memory planning, in bytes.

        planned_mem_num : int
            The number of memory.alloc_storages.
        """

        total_alloc_tensor_mem: int
        total_const_size_tensor_num: int
        total_dyn_size_tensor_num: int
        planned_alloc_mem: int
        planned_mem_num: int
        builtin_alloc_tensor_op = Op.get("relax.builtin.alloc_tensor")
        memory_alloc_tensor_op = Op.get("relax.memory.alloc_tensor")
        memory_alloc_storage_op = Op.get("relax.memory.alloc_storage")

        def estimate(self, mod: IRModule) -> str:
            estimation: str = ""
            for global_var, func in mod.functions_items():
                if not isinstance(func, Function):
                    continue

                self.cleanup()
                self.visit_expr(func)
                estimation += self.generate_est_string(global_var.name_hint) + "\n"

            if estimation != "":
                estimation = "Memory usage estimation:\n" + estimation
            return estimation

        def cleanup(self) -> None:
            self.total_alloc_tensor_mem = 0
            self.total_const_size_tensor_num = 0
            self.total_dyn_size_tensor_num = 0
            self.planned_alloc_mem = 0
            self.planned_mem_num = 0

        def visit_call_(self, call: Call) -> None:  # pylint: disable=arguments-differ
            if call.op == self.builtin_alloc_tensor_op:
                self.accumulate_builtin_tensor_alloc(
                    shape=call.args[0], dtype_str=call.args[1].value
                )
            elif call.op == self.memory_alloc_tensor_op:
                self.accumulate_tensor_alloc(shape=call.args[2], dtype_str=call.args[3].value)
            elif call.op == self.memory_alloc_storage_op:
                self.accumulate_storage_alloc(size=call.args[0])

        def calculate_size(self, shape: Expr, dtype_str: str) -> int:
            if not isinstance(shape, ShapeExpr):
                raise TypeError(
                    "The shape of relax.builtin.alloc_tensor and "
                    "relax.memory.alloc_tensor is expected to be ShapeExpr"
                )
            size: int = 1
            for dim_len in shape.values:
                if not isinstance(dim_len, tvm_tirx.ir.IntImm):
                    self.total_dyn_size_tensor_num += 1
                    return -1
                size *= dim_len.value
            dtype = tvm_ffi.dtype(dtype_str)
            return size * ((dtype.bits + 7) // 8) * dtype.lanes

        def accumulate_builtin_tensor_alloc(self, shape: Expr, dtype_str: str) -> None:
            size = self.calculate_size(shape, dtype_str)
            if size == -1:
                return
            self.total_const_size_tensor_num += 1
            self.total_alloc_tensor_mem += size
            self.planned_mem_num += 1
            self.planned_alloc_mem += size

        def accumulate_tensor_alloc(self, shape: Expr, dtype_str: str) -> None:
            size = self.calculate_size(shape, dtype_str)
            if size == -1:
                return
            self.total_const_size_tensor_num += 1
            self.total_alloc_tensor_mem += size

        def accumulate_storage_alloc(self, size: Expr) -> None:
            if not isinstance(size, ShapeExpr):
                raise TypeError(
                    "The size of relax.memory.alloc_storage is expected to be ShapeExpr"
                )

            self.planned_mem_num += 1
            self.planned_alloc_mem += size.values[0].value

        def generate_est_string(self, func_name: str) -> str:
            est = (
                f" * Without memory planning, there are {self.total_const_size_tensor_num} "
                "constant-size memory allocation(s) with total size "
                "{0:.4} GB".format(self.total_alloc_tensor_mem / 2**30)
            )
            if self.total_dyn_size_tensor_num > 0:
                est += f", and {self.total_dyn_size_tensor_num} dynamic-size allocation(s)"
            est += (
                f".\n * With memory planning, there are {self.planned_mem_num} constant-size "
                "memory allocation(s) with total size "
                "{0:.4} GB.\n".format(self.planned_alloc_mem / 2**30)
            )
            if self.total_alloc_tensor_mem != 0:
                est += (
                    " * Memory planning reduces constant memory size to "
                    f"{self.planned_alloc_mem / self.total_alloc_tensor_mem:.1%}."
                )
            return "- Function " + func_name + ":\n" + est

    if isinstance(mod, Function):
        mod = tvm_tirx.IRModule({tvm_tirx.ir.GlobalVar("foo"): mod})

    return MemoryEstimator().estimate(mod)
