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
# ruff: noqa: F401

import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_basic():
    m_tir_matmul = T.dynamic("m", "int64")
    n_tir_matmul = T.dynamic("n", "int64")
    k_tir_matmul = T.dynamic("k", "int64")
    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    k_main = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def tir_matmul(
            A: T.Tensor((m_tir_matmul, n_tir_matmul), "float32"),
            B: T.Tensor((n_tir_matmul, k_tir_matmul), "float32"),
            C: T.Tensor((m_tir_matmul, k_tir_matmul), "float32"),
        ) -> None:
            for i, j, k_tir_matmul_index in T.grid(m_tir_matmul, k_tir_matmul, n_tir_matmul):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k_tir_matmul_index])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function(private=True)
        def main(
            x: R.Tensor((m_main, n_main), "float32"), w: R.Tensor((n_main, k_main), "float32")
        ) -> R.Tensor:
            gv0 = R.call_tir(
                Before.tir_matmul, (x, w), ty_args=[R.Tensor((m_main, k_main), dtype="float32")]
            )
            return gv0

    m_tir_matmul = T.dynamic("m", "int64")
    n_tir_matmul = T.dynamic("n", "int64")
    k_tir_matmul = T.dynamic("k", "int64")
    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    k_main = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def tir_matmul(
            A: T.Tensor((m_tir_matmul, n_tir_matmul), "float32"),
            B: T.Tensor((n_tir_matmul, k_tir_matmul), "float32"),
            C: T.Tensor((m_tir_matmul, k_tir_matmul), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})

            for i, j, k_tir_matmul_index in T.grid(m_tir_matmul, k_tir_matmul, n_tir_matmul):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k_tir_matmul_index])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            x: R.Tensor((m_main, n_main), "float32"), w: R.Tensor((n_main, k_main), "float32")
        ) -> R.Tensor:
            gv0 = R.call_tir(
                Expected.tir_matmul, (x, w), ty_args=[R.Tensor((m_main, k_main), dtype="float32")]
            )
            return gv0

    before = Before
    expected = Expected
    after = relax.transform.AttachGlobalSymbol()(before)
    assert_structural_equal(after, expected)


def test_system_lib_prefix():
    @tvm_tirx.script.ir_module
    class Before:
        I.module_attrs({"system_lib_prefix": "hello_"})

        @Ts.function(private=True)
        def tir_zeros(x: T.Tensor((2), "float32")) -> None:
            x[0] = T.float32(0)

        @R.function(private=True)
        def main() -> R.Tensor:
            gv0 = R.call_tir(Before.tir_zeros, (), ty_args=[R.Tensor((2,), dtype="float32")])
            return gv0

    @tvm_tirx.script.ir_module
    class Expected:
        I.module_attrs({"system_lib_prefix": "hello_"})

        @Ts.function
        def hello_tir_zeros(x: T.Tensor((2), "float32")) -> None:
            T.func_attr({"global_symbol": "hello_tir_zeros"})
            x[0] = T.float32(0)

        @R.function
        def main() -> R.Tensor:
            gv0 = R.call_tir(
                Expected.hello_tir_zeros, (), ty_args=[R.Tensor((2,), dtype="float32")]
            )
            return gv0

    before = Before
    after = relax.transform.AttachGlobalSymbol()(before)
    assert_structural_equal(after, Expected)


if __name__ == "__main__":
    pytest.main([__file__])
