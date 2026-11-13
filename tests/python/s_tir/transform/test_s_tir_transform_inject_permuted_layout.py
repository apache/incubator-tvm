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
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize("columns", [32, 64])
@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("annotation", [1, "g2s_A"])
def test_permute_shared_tensor_accesses(columns, scope, annotation):
    @Ts.function
    def before(A: T.Tensor((8, columns), "float16"), B: T.Tensor((8, columns), "float16")):
        shared = Ts.sblock_alloc_tensor((8, columns), "float16", scope=scope)
        for i, j in T.grid(8, columns):
            with Ts.sblock("load"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.sblock_attr({"permuted_layout": annotation})
                shared[vi, vj] = A[vi, vj]
        for i, j in T.grid(8, columns):
            with Ts.sblock("store"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.sblock_attr({"permuted_layout": annotation})
                B[vi, vj] = shared[vi, vj]

    def permuted_column(i, j):
        return T.bitwise_xor(j // 8, i // 2 if columns == 32 else i) * 8 + j % 8

    @Ts.function
    def expected(A: T.Tensor((8, columns), "float16"), B: T.Tensor((8, columns), "float16")):
        shared = Ts.sblock_alloc_tensor((8, columns), "float16", scope=scope)
        for i, j in T.grid(8, columns):
            with Ts.sblock("load"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi, vj])
                Ts.writes(shared[vi, vj])
                shared[vi, permuted_column(vi, vj)] = A[vi, vj]
        for i, j in T.grid(8, columns):
            with Ts.sblock("store"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(shared[vi, vj])
                Ts.writes(B[vi, vj])
                B[vi, vj] = shared[vi, permuted_column(vi, vj)]

    result = tvm.s_tir.transform.InjectPermutedLayout()(tvm_tirx.IRModule.from_expr(before))[
        "before"
    ]
    tvm_tirx.ir.assert_structural_equal(
        result.without_attr("global_symbol"), expected.without_attr("global_symbol")
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
