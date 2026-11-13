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

from collections.abc import Callable

import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import topi
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _test_static_shape(name: str, relax_op: Callable, te_func: Callable, dtype: str):
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), dtype)):
            nonlocal dtype
            gv = relax_op(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype)):
            nonlocal dtype
            gv = R.emit_te(te_func, x, function_name_hint=f"tir_{name}")
            return gv

    mod = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def _test_symbolic_shape(name: str, relax_op: Callable, te_func: Callable, dtype: str):
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((m, n), dtype)):
            nonlocal dtype
            gv = relax_op(x)
            return gv

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m, n), dtype)):
            nonlocal dtype
            gv = R.emit_te(te_func, x, function_name_hint=f"tir_{name}")
            return gv

    mod = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


@pytest.mark.parametrize(
    "name, relax_op, te_func, dtype",
    [
        ("abs", R.abs, topi.abs, "float32"),
        ("acos", R.acos, topi.acos, "float32"),
        ("acosh", R.acosh, topi.acosh, "float32"),
        ("asin", R.asin, topi.asin, "float32"),
        ("asinh", R.asinh, topi.asinh, "float32"),
        ("atan", R.atan, topi.atan, "float32"),
        ("atanh", R.atanh, topi.atanh, "float32"),
        ("ceil", R.ceil, topi.ceil, "float32"),
        ("ceil", R.ceil, topi.identity, "int32"),
        ("cos", R.cos, topi.cos, "float32"),
        ("cosh", R.cosh, topi.cosh, "float32"),
        ("exp", R.exp, topi.exp, "float32"),
        ("floor", R.floor, topi.floor, "float32"),
        ("floor", R.floor, topi.identity, "int32"),
        ("isfinite", R.isfinite, topi.isfinite, "float32"),
        ("isinf", R.isinf, topi.isinf, "float32"),
        ("isnan", R.isnan, topi.isnan, "float32"),
        ("log", R.log, topi.log, "float32"),
        ("negative", R.negative, topi.negative, "float32"),
        ("round", R.round, topi.round, "float32"),
        ("round", R.round, topi.identity, "int32"),
        ("rsqrt", R.rsqrt, topi.rsqrt, "float32"),
        ("sigmoid", R.sigmoid, topi.sigmoid, "float32"),
        ("sign", R.sign, topi.sign, "float32"),
        ("sign", R.sign, topi.sign, "int32"),
        ("sin", R.sin, topi.sin, "float32"),
        ("sinh", R.sinh, topi.sinh, "float32"),
        ("sqrt", R.sqrt, topi.sqrt, "float32"),
        ("square", R.square, lambda x: topi.multiply(x, x), "float32"),
        ("tan", R.tan, topi.tan, "float32"),
        ("tanh", R.tanh, topi.tanh, "float32"),
        ("clip", lambda x: R.clip(x, 5, 8), lambda x: topi.clip(x, 5, 8), "float32"),
    ],
)
def test_unary_ops(name: str, relax_op: Callable, te_func: Callable, dtype: str):
    _test_static_shape(name, relax_op, te_func, dtype)
    _test_symbolic_shape(name, relax_op, te_func, dtype)


if __name__ == "__main__":
    tvm_tirx.testing.main()
