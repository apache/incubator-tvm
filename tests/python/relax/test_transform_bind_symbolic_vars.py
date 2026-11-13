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
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_bind_tensors():
    """Symbolic variables may occur in Tensor shapes"""

    batch = T.dynamic("batch", "int64")
    n = T.dynamic("n", "int64")
    k = T.dynamic("k", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((batch, m), dtype="float32"),
            w0: R.Tensor((m, n), dtype="float32"),
            w1: R.Tensor((k, 10), dtype="float32"),
        ) -> R.Tensor((batch, k), dtype="float32"):
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    _ExternFunc("test0"), (x, w0), ty_args=[R.Tensor((batch, n), dtype="float32")]
                )
                out = R.call_dps_packed(
                    _ExternFunc("test1"), (lv0, w1), ty_args=[R.Tensor((batch, k), dtype="float32")]
                )
                R.output(out)
            return out

    symvar_map = {"batch": 1, "k": 3}
    target_func_name = "main"
    After = relax.transform.BindSymbolicVars(symvar_map, target_func_name)(Before)

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, m), dtype="float32"),
            w0: R.Tensor((m, n), dtype="float32"),
            w1: R.Tensor((3, 10), dtype="float32"),
        ) -> R.Tensor((1, 3), dtype="float32"):
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    _ExternFunc("test0"), (x, w0), ty_args=[R.Tensor((1, n), dtype="float32")]
                )
                out = R.call_dps_packed(
                    _ExternFunc("test1"), (lv0, w1), ty_args=[R.Tensor((1, 3), dtype="float32")]
                )
                R.output(out)
            return out

    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_bind_shape():
    """Symbolic variables may occur in ShapeExpr"""

    batch = T.dynamic("batch", "int64")
    n = T.dynamic("n", "int64")
    k = T.dynamic("k", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Shape((batch, m)),
            w0: R.Shape((m, n)),
            w1: R.Shape((k, 10)),
        ) -> R.Shape((batch, k)):
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    _ExternFunc("test0"), (x, w0), ty_args=[R.Tensor((batch, n))]
                )
                out = R.call_dps_packed(
                    _ExternFunc("test1"), (lv0, w1), ty_args=[R.Tensor((batch, k))]
                )
                R.output(out)
            return out

    symvar_map = {"batch": 1, "k": 3}
    target_func_name = "main"
    After = relax.transform.BindSymbolicVars(symvar_map, target_func_name)(Before)

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Shape([1, m]), w0: R.Shape([m, n]), w1: R.Shape([3, 10])) -> R.Shape([1, 3]):
            with R.dataflow():
                lv0 = R.call_dps_packed(_ExternFunc("test0"), (x, w0), ty_args=[R.Tensor((1, n))])
                out = R.call_dps_packed(_ExternFunc("test1"), (lv0, w1), ty_args=[R.Tensor((1, 3))])
                R.output(out)
            return out

    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_arith():
    """Symbolic shapes may use TIR arithmetic expressions"""

    batch = T.dynamic("batch", "int64")
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    k = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((batch, m - 1), dtype="float32"),
            w0: R.Tensor((m, n), dtype="float32"),
            w1: R.Tensor((k, 10), dtype="float32"),
        ) -> R.Tensor((batch, k * m), dtype="float32"):
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    _ExternFunc("test0"),
                    (x, w0),
                    ty_args=[R.Tensor((batch, m + n), dtype="float32")],
                )
                out = R.call_dps_packed(
                    _ExternFunc("test1"),
                    (lv0, w1),
                    ty_args=[R.Tensor((batch, k + n), dtype="float32")],
                )
                R.output(out)
            return out

    symvar_map = {"batch": 1, "k": 2, "m": 3}
    target_func_name = "main"
    After = relax.transform.BindSymbolicVars(symvar_map, target_func_name)(Before)

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 2), dtype="float32"),
            w0: R.Tensor((3, n), dtype="float32"),
            w1: R.Tensor((2, 10), dtype="float32"),
        ) -> R.Tensor((1, 6), dtype="float32"):
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    _ExternFunc("test0"), (x, w0), ty_args=[R.Tensor((1, n + 3), dtype="float32")]
                )
                out = R.call_dps_packed(
                    _ExternFunc("test1"), (lv0, w1), ty_args=[R.Tensor((1, n + 2), dtype="float32")]
                )
                R.output(out)
            return out

    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_bind_multiple_variables_by_name():
    """String names may be used to replace across multiple functions"""

    m_main_1 = T.dynamic("m", "int64")
    n_main_1 = T.dynamic("n", "int64")
    m_main_2 = T.dynamic("m", "int64")
    n_main_2 = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main_1(x: R.Tensor((m_main_1, n_main_1), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor((m_main_2, n_main_2), dtype="float32")):
            return x

    m_main_1 = T.dynamic("m", "int64")
    m_main_2 = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main_1(x: R.Tensor((m_main_1, 16), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor((m_main_2, 16), dtype="float32")):
            return x

    After = relax.transform.BindSymbolicVars({"n": 16})(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_bind_single_variable_by_identity():
    """TIR variables may be used to replace a specific var"""

    m_main_1 = T.dynamic("m", "int64")
    n_main_1 = T.dynamic("n", "int64")
    m_main_2 = T.dynamic("m", "int64")
    n_main_2 = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main_1(x: R.Tensor((m_main_1, n_main_1), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor((m_main_2, n_main_2), dtype="float32")):
            return x

    m_main_1 = T.dynamic("m", "int64")
    m_main_2 = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main_1(x: R.Tensor((m_main_1, 16), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor((m_main_2, n), dtype="float32")):
            return x

    main_1_n = Before["main_1"].params[0].ty.shape[1]
    After = relax.transform.BindSymbolicVars({main_1_n: 16})(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_bind_single_variable_by_function_name():
    """Variable name and function name may be used to replace a specific var"""

    m_main_1 = T.dynamic("m", "int64")
    n_main_1 = T.dynamic("n", "int64")
    m_main_2 = T.dynamic("m", "int64")
    n_main_2 = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main_1(x: R.Tensor((m_main_1, n_main_1), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor((m_main_2, n_main_2), dtype="float32")):
            return x

    m_main_1 = T.dynamic("m", "int64")
    m_main_2 = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main_1(x: R.Tensor((m_main_1, 16), dtype="float32")):
            return x

        @R.function
        def main_2(x: R.Tensor((m_main_2, n), dtype="float32")):
            return x

    After = relax.transform.BindSymbolicVars({"n": 16}, "main_1")(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_error_for_unused_replacement():
    """Each replacement must be used"""

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((m, n), dtype="float32")):
            return x

    with pytest.raises(RuntimeError):
        relax.transform.BindSymbolicVars({"non_existing_var_name": 16})(Before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
