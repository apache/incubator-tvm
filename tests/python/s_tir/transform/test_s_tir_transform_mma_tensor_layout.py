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

import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize("scope,shape", [("mma.matrixA", (32, 8)), ("mma.matrixB", (8, 32))])
@pytest.mark.parametrize("access_kind", ["load", "store"])
def test_explicit_matrix_ab_access_is_rejected(scope, shape, access_kind):
    tensor = tirx.Var("tensor", tirx.TensorType(shape, "float32", scope=scope))
    if access_kind == "load":
        body = tvm_tirx.ir.Evaluate(tvm_tirx.ir.TensorLoad(tensor, [0, 0]))
    else:
        body = tvm_tirx.ir.TensorStore(tensor, [0, 0], 0.0)
    block = s_tir.SBlock([], [], [], "root", body, alloc_tensors=[tensor])
    func = tirx.Function([], s_tir.SBlockRealize([], True, block))

    with pytest.raises(RuntimeError, match=f"{scope}.*explicit"):
        s_tir.transform.TransformMmaTensorLayout()(tvm_tirx.IRModule.from_expr(func))


def test_matrix_c_load_store_lower_without_intrinsic_catalogue():
    @Ts.function
    def before(A: T.Tensor((32, 16), "float32"), B: T.Tensor((32, 16), "float32")):
        fragment = Ts.sblock_alloc_tensor((32, 16), "float32", scope="mma.matrixC")
        for i, j in T.grid(32, 16):
            fragment[i, j] = A[i, j]
            B[i, j] = fragment[i, j]

    @Ts.function
    def expected(A: T.Tensor((32, 16), "float32"), B: T.Tensor((32, 16), "float32")):
        fragment = Ts.sblock_alloc_tensor((2, 2, 2, 2), "float32", scope="local")
        for i, j in T.grid(32, 16):
            fragment[i // 16, j // 8, i % 16 // 8, j % 2] = A[i, j]
            B[i, j] = fragment[i // 16, j // 8, i % 16 // 8, j % 2]

    result = s_tir.transform.TransformMmaTensorLayout()(tvm_tirx.IRModule.from_expr(before))[
        "before"
    ]
    tvm_tirx.ir.assert_structural_equal(
        result.without_attr("global_symbol"), expected.without_attr("global_symbol")
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
