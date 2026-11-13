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
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


def test_canonicalize_loop():
    @Ts.function
    def before(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        T.func_attr({"global_symbol": "main"})
        for i in range(1, 128, 5):
            B[i] = A[i] + 1.0

    @Ts.function
    def expected(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 26):
            B[i * 5 + 1] = A[i * 5 + 1] + 1.0

    mod = tvm_tirx.IRModule.from_expr(before)
    mod = s_tir.transform.CanonicalizeLoop()(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_canonicalize_nested_loop():
    @Ts.function
    def before(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        T.func_attr({"global_symbol": "main"})
        for i in range(1, 128, 5):
            for j in range(2, 128, 3):
                B[i, j] = A[i, j] + 1.0

    @Ts.function
    def expected(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 26):
            for j in T.serial(0, 42):
                B[i * 5 + 1, j * 3 + 2] = A[i * 5 + 1, j * 3 + 2] + 1.0

    mod = tvm_tirx.IRModule.from_expr(before)
    mod = s_tir.transform.CanonicalizeLoop()(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_canonicalize_negative_step():
    @Ts.function
    def before(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 127, step=-3):
            B[i] = A[i] + 1.0

    mod = tvm_tirx.IRModule.from_expr(before)
    with pytest.raises(RuntimeError):
        mod = s_tir.transform.CanonicalizeLoop()(mod)


def test_canonicalize_dynamic_step():
    """Currently we report error for dynamic step since we could not prove it is positive"""

    @Ts.function
    def before(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32"), step: T.int32):
        T.func_attr({"global_symbol": "main"})
        for i in T.serial(0, 128, step=step):
            B[i] = A[i] + 1.0

    mod = tvm_tirx.IRModule.from_expr(before)
    with pytest.raises(RuntimeError):
        mod = s_tir.transform.CanonicalizeLoop()(mod)


if __name__ == "__main__":
    tvm_tirx.testing.main()
