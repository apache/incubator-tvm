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
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize("key_type", [tvm_tirx.ir.GlobalVar, str])
def test_inline_simple(key_type):
    """Simple case of inlining

    Inlining can be done either by providing a string name or a
    GlobalVar.
    """

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            B = A * A
            C = Before.subroutine(B)
            D = C + C
            return D

        @R.function(private=True)
        def subroutine(B: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            C = R.concat([B, B], axis=1)
            return C

    @R.function(private=True)
    def expected(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
        B = A * A
        C = R.concat([B, B], axis=1)
        D = C + C
        return D

    gvar = Before.get_global_var("subroutine")
    if key_type is tvm_tirx.ir.GlobalVar:
        key = gvar
    elif key_type is str:
        key = gvar.name_hint
    else:
        raise TypeError(f"Unknown key_type: {key_type}")

    after = Before["main"].inline_functions({key: Before[gvar]})

    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_ambiguous_function_name():
    """Raise an error on ambiguous inputs

    For convenience, the function being replaced can be specified
    either as a string, or as a GlobalVar.  However, all replacements
    must be unambiguous.
    """

    @R.function
    def func():
        return ()

    gvar = tvm_tirx.ir.GlobalVar("name")

    with pytest.raises(ValueError):
        func.inline_functions({gvar: func, "name": func})


def test_inline_dataflow_block():
    """Functions may be inlined within a dataflow block"""

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            with R.dataflow():
                B = A * A
                C = Before.subroutine(B)
                D = C + C
                R.output(D)
            return D

        @R.function(private=True)
        def subroutine(B: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            with R.dataflow():
                C = R.concat([B, B], axis=1)
                R.output(C)
            return C

    @R.function(private=True)
    def expected(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
        with R.dataflow():
            B = A * A
            C = R.concat([B, B], axis=1)
            D = C + C
            R.output(D)
        return D

    after = Before["main"].inline_functions({"subroutine": Before["subroutine"]})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_inline_non_dataflow_block_into_dataflow_block():
    """Function inlining may not produce invalid Relax IR

    A subroutine call may appear within a DataflowBlock, even if the
    subroutine does not itself use a DataflowBlock.  In this case, to
    avoid inserting a non-dataflow block in the middle of a set of
    dataflow bindings, the DataflowBlock in the caller must be split
    up.
    """

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            with R.dataflow():
                B = A * A
                C = Before.subroutine(B)
                D = C + C
                R.output(D)
            return D

        @R.function(private=True)
        def subroutine(B: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            C = R.concat([B, B], axis=1)
            return C

    @R.function(private=True)
    def expected(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
        # DataflowBlock before subroutine
        with R.dataflow():
            B = A * A
            R.output(B)

        # BindingBlock from the inlined subroutine.  Because B is used
        # here, outside of a DataflowBlock, this requires it to be
        # updated from a DataflowVar to a normal Var.
        C = R.concat([B, B], axis=1)

        # Resuming the DataflowBlock after the inlined subroutine
        with R.dataflow():
            D = C + C
            R.output(D)
        return D

    after = Before["main"].inline_functions({"subroutine": Before["subroutine"]})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_subroutine_with_symbolic_vars():
    """Inlined subroutines should use the caller's symbolic variables

    Before inlining, the subroutine and the caller have distinct
    `tirx::Var` for each symbolic variables.  After inlining, only the
    caller's `tirx::Var` symbolic variables should remain.
    """

    n_main = T.dynamic("n", "int64")
    n_subroutine = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor([n_main, 16], "int32")) -> R.Tensor([n_main, 32], "int32"):
            B = A * A
            C = Before.subroutine(B)
            D = C + C
            return D

        @R.function(private=True)
        def subroutine(B: R.Tensor([n_subroutine, 16], "int32")) -> R.Tensor(
            [n_subroutine, 32], "int32"
        ):
            C = R.concat([B, B], axis=1)
            return C

    n = T.dynamic("n", "int64")

    @R.function(private=True)
    def expected(A: R.Tensor([n, 16], "int32")) -> R.Tensor([n, 32], "int32"):
        B = A * A
        C = R.concat([B, B], axis=1)
        D = C + C
        return D

    after = Before["main"].inline_functions({"subroutine": Before["subroutine"]})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_subroutine_with_symbolic_vars_and_static_argument():
    """Inlined subroutines should use the caller's static shape

    Before inlining, the subroutine has symbolic variables, and the
    caller have static shape.  After inlining, no symbolic variables
    should remain.
    """

    n = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            B = A * A
            C = Before.subroutine(B)
            D = C + C
            return D

        @R.function(private=True)
        def subroutine(B: R.Tensor([n, 16], "int32")) -> R.Tensor([n, 32], "int32"):
            C = R.concat([B, B], axis=1)
            return C

    @R.function(private=True)
    def expected(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
        B = A * A
        C = R.concat([B, B], axis=1)
        D = C + C
        return D

    after = Before["main"].inline_functions({"subroutine": Before["subroutine"]})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_inline_multiple_instances():
    """A subroutine may be inlined multiple times

    When inlining, SSA should still be respected.
    """

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor):
            B = Before.subroutine(A)
            C = Before.subroutine(B)
            return C

        @R.function(private=True)
        def subroutine(A0: R.Tensor) -> R.Tensor:
            A1 = A0 * A0
            A2 = A1 + A1
            return A2

    @R.function(private=True)
    def expected(A: R.Tensor):
        # First call
        B = A * A
        C = B + B
        # Second call
        D = C * C
        E = D + D
        return E

    after = Before["main"].inline_functions({"subroutine": Before["subroutine"]})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_inline_multiple_instances_with_distinct_static_shapes():
    """A subroutine may be inlined multiple times

    When inlining, each instance of the inlined function may have a
    different value for the symbolic variables it uses.
    """

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor([16, 16]), B: R.Tensor([32, 32])):
            A_out: R.Tensor([16, 16]) = Before.subroutine(A)
            B_out: R.Tensor([32, 32]) = Before.subroutine(B)
            return (A_out, B_out)

        @R.function(private=True)
        def subroutine(Input: R.Tensor([n, m])) -> R.Tensor([n, m]):
            Output = Input + Input
            return Output

    @R.function(private=True)
    def expected(A: R.Tensor([16, 16]), B: R.Tensor([32, 32])):
        A_out: R.Tensor([16, 16]) = A + A
        B_out: R.Tensor([32, 32]) = B + B
        return (A_out, B_out)

    after = Before["main"].inline_functions({"subroutine": Before["subroutine"]})
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_inline_nested_subroutine_calls():
    """A private function may itself require inlining"""

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            B = A * A
            D = Before.subroutine(B)
            E = D + D
            return E

        @R.function(private=True)
        def subroutine(B: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            C = R.concat([B, B], axis=1)
            D = Before.subsubroutine(C)
            return D

        @R.function(private=True)
        def subsubroutine(C: R.Tensor([16, 32], "int32")) -> R.Tensor([16, 32], "int32"):
            D = C * C * C
            return D

    @R.function(private=True)
    def expected(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
        B = A * A
        C = R.concat([B, B], axis=1)
        D = C * C * C
        E = D + D
        return E

    after = Before["main"].inline_functions(
        {
            "subroutine": Before["subroutine"],
            "subsubroutine": Before["subsubroutine"],
        }
    )
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_error_when_inlining_recursive_function():
    """Inlining a recursive function call should raise an error"""

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main():
            B = Before.subroutine()
            return B

        @R.function(private=True)
        def subroutine() -> R.Tensor([], "int64"):
            R.func_attr({"relax.force_pure": True})
            cond = R.call_packed("dummy_function", ty_args=R.Tensor([], "bool"))
            if cond:
                Out = Before.subroutine()
            else:
                Out = R.const(0, "int64")

            return Out

    with pytest.raises(Exception):
        Before["main"].inline_functions({"subroutine": Before["subroutine"]})


def test_error_when_inlining_mutually_recursive_functions():
    """Inlining a recursive function call should raise an error"""

    @I.ir_module
    class Before:
        @R.function(private=True)
        def main():
            B = Before.subroutine_a()
            return B

        @R.function(private=True)
        def subroutine_a() -> R.Tensor([], "int64"):
            R.func_attr({"relax.force_pure": True})
            cond = R.call_packed("dummy_function", ty_args=R.Tensor([], "bool"))
            if cond:
                Out = Before.subroutine_b()
            else:
                Out = R.const(0, "int64")

            return Out

        @R.function(private=True)
        def subroutine_b() -> R.Tensor([], "int64"):
            R.func_attr({"relax.force_pure": True})
            cond = R.call_packed("dummy_function", ty_args=R.Tensor([], "bool"))
            if cond:
                Out = Before.subroutine_a()
            else:
                Out = R.const(0, "int64")

            return Out

    with pytest.raises(Exception):
        Before["main"].inline_functions(
            {
                "subroutine_a": Before["subroutine_a"],
                "subroutine_b": Before["subroutine_b"],
            }
        )


if __name__ == "__main__":
    tvm_tirx.testing.main()
