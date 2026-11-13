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
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_remove_unused_relax_parameter():
    """A relax parameter may be removed

    This is only allowed for internal function calls, where all
    callsites can be updated.  For externally-exposed functions, the
    signature may not be modified.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return Before.func(A, B)

        @R.function(private=True)
        def func(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            return A

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return Expected.func(A)

        @R.function(private=True)
        def func(A: R.Tensor) -> R.Tensor:
            return A

    After = tvm.relax.transform.RemoveUnusedParameters()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_replace_symbolic_variables():
    """If a parameter is only required for its symbolic variables, provide them directly

    The relax parameter `A` isn't used by the subroutine.  However,
    its shape defines the symbolic variables `m` and `n`.  When
    removing the `R.Tensor` argument, we may need to provide
    additional parameters to define the symbolic variables.

    A `PrimType` carries only a dtype and defines no TIR var.  Each free
    symbolic variable is therefore promoted through a 1-D `R.Shape`
    parameter, which actually *defines* the variable.
    """

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_func = T.dynamic("m", "int64")
    n_func = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([m_main, n_main], "float32")) -> R.Tensor([m_main, n_main], "float32"):
            return Before.func(A)

        @R.function(private=True)
        def func(A: R.Tensor([m_func, n_func], "float32")) -> R.Tensor([m_func, n_func], "float32"):
            return R.zeros(R.shape([m_func, n_func]), dtype="float32")

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_func = T.dynamic("m", "int64")
    n_func = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([m_main, n_main], "float32")) -> R.Tensor([m_main, n_main], "float32"):
            out: R.Tensor([m_main, n_main], "float32") = Expected.func(
                R.shape([n_main]), R.shape([m_main])
            )
            return out

        @R.function(private=True)
        def func(param_n: R.Shape([n_func]), param_m: R.Shape([m_func])) -> R.Tensor(
            [m_func, n_func], "float32"
        ):
            return R.zeros(R.shape([m_func, n_func]), dtype="float32")

    After = tvm.relax.transform.RemoveUnusedParameters()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_no_extra_symbolic_variables():
    """Don't add symbolic variables if they can be inferred.

    Even though some cases require adding new parameters to provide
    symbolic variables, not every symbolic variable requires a
    distinct parameter.
    """

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_func = T.dynamic("m", "int64")
    n_func = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([m_main, n_main], "float32")) -> R.Tensor([m_main, n_main], "float32"):
            return Before.func(A)

        @R.function(private=True)
        def func(A: R.Tensor([m_func, n_func], "float32")) -> R.Tensor([m_func, n_func], "float32"):
            zeros = R.zeros(R.shape([m_func, n_func]), dtype="float32")
            out = R.add(A, zeros)
            return out

    Expected = Before

    After = tvm.relax.transform.RemoveUnusedParameters()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_remove_extra_prim_parameters():
    """Remove unused scalar parameters.

    The tensor parameter already defines the symbolic dimensions, while the
    dtype-only scalar parameters are unused by the private function.
    """

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_func = T.dynamic("m", "int64")
    n_func = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([m_main, n_main], "float32")) -> R.Tensor([m_main, n_main], "float32"):
            return Before.func(A, R.prim_value(m_main), R.prim_value(n_main))

        @R.function(private=True)
        def func(
            A: R.Tensor([m_func, n_func], "float32"),
            _m: T.int64,
            _n: T.int64,
        ) -> R.Tensor([m_func, n_func], "float32"):
            zeros = R.zeros(R.shape([m_func, n_func]), dtype="float32")
            out = R.add(A, zeros)
            return out

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_func = T.dynamic("m", "int64")
    n_func = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([m_main, n_main], "float32")) -> R.Tensor([m_main, n_main], "float32"):
            return Expected.func(A)

        @R.function(private=True)
        def func(A: R.Tensor([m_func, n_func], "float32")) -> R.Tensor([m_func, n_func], "float32"):
            zeros = R.zeros(R.shape([m_func, n_func]), dtype="float32")
            out = R.add(A, zeros)
            return out

    After = tvm.relax.transform.RemoveUnusedParameters()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_remove_extra_shape_variables():
    """Remove parameters that only serve to define existing symbolic variables

    If a `R.Shape` parameter provides a definition of a symbolic
    variable, but that symbolic variable can be determined from a
    different parameter, then the `R.Shape` parameter can be removed.
    """

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_func = T.dynamic("m", "int64")
    n_func = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([m_main, n_main], "float32")) -> R.Tensor([m_main, n_main], "float32"):
            return Before.func(A, R.shape([m_main, n_main]))

        @R.function(private=True)
        def func(
            A: R.Tensor([m_func, n_func], "float32"),
            _: R.Shape([m_func, n_func]),
        ) -> R.Tensor([m_func, n_func], "float32"):
            zeros = R.zeros(R.shape([m_func, n_func]), dtype="float32")
            out = R.add(A, zeros)
            return out

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_func = T.dynamic("m", "int64")
    n_func = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([m_main, n_main], "float32")) -> R.Tensor([m_main, n_main], "float32"):
            return Expected.func(A)

        @R.function(private=True)
        def func(A: R.Tensor([m_func, n_func], "float32")) -> R.Tensor([m_func, n_func], "float32"):
            zeros = R.zeros(R.shape([m_func, n_func]), dtype="float32")
            out = R.add(A, zeros)
            return out

    After = tvm.relax.transform.RemoveUnusedParameters()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
