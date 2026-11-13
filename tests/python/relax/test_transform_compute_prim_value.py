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

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_prim_value_in_assert_condition():
    N = T.dynamic("N", "int64")

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(A: R.Tensor([N])):
            _ = R.assert_op(N % 16 == 0)
            return A

    N = T.dynamic("N", "int64")

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(A: R.Tensor([N])):
            condition: T.bool = R.call_pure_packed(
                tvm_tirx.ir.Op.get("relax.call_tir_packed"),
                Expected.compute_symbolic_expr,
                (R.prim_value(N),),
            )
            _ = R.assert_op(condition)
            return A

        @Ts.function(private=True)
        def compute_symbolic_expr(N: T.int64) -> T.bool:
            T.func_attr({"tirx.is_host_func": True})
            return N % 16 == 0

    After = tvm.relax.transform.ComputePrimValue()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_prim_value_in_branch_condition():
    N = T.dynamic("N", "int64")

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(A: R.Tensor([N])):
            if R.prim_value(N % 16 == 0):
                out = R.call_packed("fast_vectorized_impl", A, ty_args=[A.ty])
            else:
                out = R.call_packed("slow_non_vectorized_impl", A, ty_args=[A.ty])
            return out

    N = T.dynamic("N", "int64")

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(A: R.Tensor([N])):
            condition: T.bool = R.call_pure_packed(
                tvm_tirx.ir.Op.get("relax.call_tir_packed"),
                Expected.compute_symbolic_expr,
                (R.prim_value(N),),
            )
            if condition:
                out = R.call_packed("fast_vectorized_impl", A, ty_args=[A.ty])
            else:
                out = R.call_packed("slow_non_vectorized_impl", A, ty_args=[A.ty])
            return out

        @Ts.function(private=True)
        def compute_symbolic_expr(N: T.int64) -> T.bool:
            T.func_attr({"tirx.is_host_func": True})
            return N % 16 == 0

    After = tvm.relax.transform.ComputePrimValue()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
