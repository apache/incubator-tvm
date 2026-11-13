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
# ruff: noqa: F841

import tvm_ffi

import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax import transform
from tvm.s_tir import script as Ts
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _check_equal(x, y):
    tvm_tirx.ir.assert_structural_equal(x, y)
    tvm_tirx.ir.assert_structural_equal(y, x)

    xhash = tvm_ffi.structural_hash(x, map_free_vars=True)
    yhash = tvm_ffi.structural_hash(y, map_free_vars=True)
    assert xhash == yhash


def _check_save_roundtrip(x):
    y = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(x))
    _check_equal(x, y)


def test_basic():
    """Functions can be listed from local bindings to the IRModule"""

    # the target IRModule
    @I.ir_module
    class Expected:
        @R.function(private=True)
        def main_inner(
            x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
            return s

        @R.function
        def main(x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")) -> R.Tensor(
            (10, 5), "float32"
        ):
            gv1: R.Tensor((10, 5), "float32") = Expected.main_inner(x1, y1)
            return gv1

    @I.ir_module
    class Before:
        @R.function
        def main(x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")) -> R.Tensor(
            (10, 5), "float32"
        ):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

    before = Before
    expected = Expected
    # Perform Lambda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_input_module_is_unmodified():
    """The input module may not be modified

    If the output requires new Type, it must create a new relax
    variable.  It must not update the type of an existing relax
    variable, as that variable may be used by another IRModule.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor(
            (2, 3), "float32"
        ):
            @R.function
            def outer_func(c1: R.Tensor((2, 3), "float32")) -> R.Callable(
                (R.Tensor((2, 3), "float32"),), R.Tensor((2, 3), "float32")
            ):
                @R.function
                def inner_func(x1: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
                    s: R.Tensor((2, 3), "float32") = R.add(x1, c1)
                    return s

                return inner_func

            in_call = outer_func(x)
            res = in_call(y)
            return res

    before = Before
    copy_of_before = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(before))

    transform.LambdaLift()(before)

    tvm_tirx.ir.assert_structural_equal(before, copy_of_before)


def test_closure():
    """Lifting functions may require producing closures"""

    # the expected IRModule
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor(
            (2, 3), "float32"
        ):
            in_call = Expected.main_outer_func(x)
            res = R.invoke_pure_closure(in_call, (y,), ty_args=(R.Tensor((2, 3), dtype="float32")))
            return res

        @R.function(private=True)
        def main_inner_func(x1: R.Tensor((2, 3), "float32"), c1: R.Tensor((2, 3), "float32")):
            r_1: R.Tensor((2, 3), "float32") = R.add(x1, c1)
            return r_1

        @R.function(private=True)
        def main_outer_func(y: R.Tensor((2, 3), "float32")) -> R.Any:
            inner_func = R.make_closure(Expected.main_inner_func, (y,))
            return inner_func

    # IRModule to perform Lambda Lifting
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor(
            (2, 3), "float32"
        ):
            @R.function
            def outer_func(c1: R.Tensor((2, 3), "float32")) -> R.Callable(
                (R.Tensor((2, 3), "float32"),), R.Tensor((2, 3), "float32")
            ):
                @R.function
                def inner_func(x1: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
                    s: R.Tensor((2, 3), "float32") = R.add(x1, c1)
                    return s

                return inner_func

            in_call = outer_func(x)
            res = in_call(y)
            return res

    before = Before
    after = transform.LambdaLift()(before)
    expected = Expected
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_recursive():
    """The lifted function may be recursively defined"""

    # the expected IRModule
    @I.ir_module
    class Expected:
        @R.function(private=True)
        def main_while_loop(
            i: R.Tensor((), "int32"), s: R.Tensor((2, 3), "float32"), x: R.Tensor((2, 3), "float32")
        ) -> R.Tensor((2, 3), "float32"):
            cond: R.Tensor((), "bool") = R.call_pure_packed(
                "test.vm.less", i, R.const(10), ty_args=(R.Tensor((), dtype="bool"))
            )
            c: R.Tensor((), "int32") = R.const(1, dtype="int32")
            if cond:
                new_i: R.Tensor((), "int32") = R.add(i, c)
                new_s: R.Tensor((2, 3), "float32") = R.add(s, x)
                new_r = Expected.main_while_loop(new_i, new_s, x)
                r = new_r
            else:
                r = s
            return r

        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), dtype="float32"):
            while_loop = R.make_closure(Expected.main_while_loop, (x,))
            gv: R.Tensor((2, 3), dtype="float32") = R.invoke_pure_closure(
                while_loop,
                (R.const(0), x),
                ty_args=(R.Tensor((2, 3), dtype="float32")),
            )
            return gv

    # the IRModule to apply lambda lifting
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor:
            @R.function
            def while_loop(i: R.Tensor((), "int32"), s: R.Tensor((2, 3), "float32")) -> R.Tensor(
                (2, 3), "float32"
            ):
                cond: R.Tensor((), "bool") = R.call_pure_packed(
                    "test.vm.less", i, R.const(10), ty_args=(R.Tensor((), dtype="bool"))
                )
                c: R.Tensor((), "int32") = R.const(1, dtype="int32")
                if cond:
                    new_i: R.Tensor((), "int32") = R.add(i, c)
                    new_s: R.Tensor((2, 3), "float32") = R.add(s, x)
                    r: R.Tensor((2, 3), "float32") = while_loop(new_i, new_s)
                else:
                    r: R.Tensor((2, 3), "float32") = s
                return r

            gv: R.Tensor((2, 3), "float32") = while_loop(R.const(0), x)
            return gv

    before = Before
    expected = Expected
    # check well-formness of recursive call
    relax.analysis.check_well_formed(before)

    # Perform Lambda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2

    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_multi_func():
    """Lifting may be required for multiple top-level functions

    De-duplication of GlobalVar names at the IRModule is done by
    appending the name of the function from which they were lifted.
    """

    # expected IRModule
    @I.ir_module
    class Expected:
        @R.function
        def glob_func_1(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv1: R.Tensor((10, 5), "float32") = Expected.glob_func_1_inner(x1, y1)
            return gv1

        @R.function
        def glob_func_2(
            x11: R.Tensor((10, 5), "float32"), y11: R.Tensor((10, 5), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            gv11: R.Tensor((10, 5), "float32") = Expected.glob_func_2_inner(x11, y11)
            return gv11

        @R.function(private=True)
        def glob_func_1_inner(
            x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
            return s

        @R.function(private=True)
        def glob_func_2_inner(
            x21: R.Tensor((10, 5), "float32"), y21: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s1: R.Tensor((10, 5), "float32") = R.add(x21, y21)
            return s1

    # the IRModule to apply lambda lifting
    @I.ir_module
    class Before:
        @R.function
        def glob_func_1(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

        @R.function
        def glob_func_2(
            x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

    before = Before
    expected = Expected
    # Perform Lambda Lifting
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 4
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_no_local_func():
    @I.ir_module
    class Before:
        @Ts.function
        def sub(
            A: T.Tensor((16, 16), "float32"),
            B: T.Tensor((16, 16), "float32"),
            C: T.Tensor((16, 16), "float32"),
        ) -> None:
            for i, j in T.grid(16, 16):
                with Ts.sblock("sub"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    C[vi, vj] = A[vi, vj] - B[vi, vj]

        @R.function
        def before(c0: R.Tensor((16, 16), "float32"), x: R.Tensor(dtype="float32", ndim=2)):
            s = R.call_tir(Before.sub, (c0, x), ty_args=[R.Tensor((16, 16), dtype="float32")])
            return s

    before = Before
    # Perform lambda lifting
    after = transform.LambdaLift()(before)
    # No local functions are lifted
    assert_structural_equal(after, before, map_free_vars=True)
    _check_save_roundtrip(after)


def test_impure_function():
    @I.ir_module
    class Expected:
        @R.function(pure=False, private=True)
        def main_inner() -> R.Tuple:
            y = R.print("Wow!")
            return y

        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            gv1 = Expected.main_inner()
            return x

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            @R.function(pure=False)
            def inner() -> R.Tuple:
                y = R.print("Wow!")
                return y

            gv1 = inner()
            return x

    before = Before
    expected = Expected
    after = transform.LambdaLift()(before)
    assert len(after.functions) == 2
    assert_structural_equal(after, expected, map_free_vars=True)
    _check_save_roundtrip(after)


def test_lambda_function_with_same_name_as_global():
    """Lifted lambda names may not conflict with previous names

    Like `test_basic`, but the module has an existing function
    `main_inner`, which has the same name as the LambdaLift's first
    choice of name for the hoisted function.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")) -> R.Tensor(
            (10, 5), "float32"
        ):
            @R.function
            def inner(
                x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
            ) -> R.Tensor((10, 5), "float32"):
                s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
                return s

            gv1: R.Tensor((10, 5), "float32") = inner(x1, y1)
            return gv1

        @R.function
        def main_inner():
            return ()

    @I.ir_module
    class Expected:
        @R.function
        def main(x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")) -> R.Tensor(
            (10, 5), "float32"
        ):
            gv1: R.Tensor((10, 5), "float32") = Expected.main_inner_0(x1, y1)
            return gv1

        @R.function(private=True)
        def main_inner_0(
            x2: R.Tensor((10, 5), "float32"), y2: R.Tensor((10, 5), "float32")
        ) -> R.Tensor((10, 5), "float32"):
            s: R.Tensor((10, 5), "float32") = R.add(x2, y2)
            return s

        @R.function
        def main_inner():
            return ()

    after = transform.LambdaLift()(Before)
    assert_structural_equal(Expected, after)


def test_symbolic_variable_defined_by_inner_func():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")) -> R.Tensor(
            (10, 5), "float32"
        ):
            @R.function
            def inner(x2: R.Tensor((n, m), "float32"), y2: R.Tensor((n, m), "float32")):
                sum_inner = R.add(x2, y2)
                return sum_inner

            sum_main = inner(x1, y1)
            return sum_main

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x1: R.Tensor((10, 5), "float32"), y1: R.Tensor((10, 5), "float32")) -> R.Tensor(
            (10, 5), "float32"
        ):
            sum_main = Expected.main_inner(x1, y1)
            return sum_main

        @R.function(private=True)
        def main_inner(
            x2: R.Tensor((n, m), "float32"), y2: R.Tensor((n, m), "float32")
        ) -> R.Tensor((n, m), "float32"):
            sum_inner = R.add(x2, y2)
            return sum_inner

    After = transform.LambdaLift()(Before)
    assert_structural_equal(Expected, After)


def test_runtime_symbolic_variable_defined_by_inner_func():
    n_from_param = T.dynamic("n", "int64")
    n_from_match_cast = T.dynamic("n", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((4,), "float32")):
            @R.function
            def from_param(y: R.Tensor((n_from_param,), "float32")):
                z = R.ones((n_from_param,), "float32")
                return z

            @R.function
            def from_match_cast(y: R.Tensor(ndim=1, dtype="float32")):
                y2 = R.match_cast(y, R.Tensor((n_from_match_cast,), "float32"))
                z = R.ones((n_from_match_cast,), "float32")
                return z

            a = from_param(x)
            b = from_match_cast(x)
            return a, b

    After = transform.LambdaLift()(Before)

    relax.analysis.check_well_formed(After)
    assert "R.make_closure" not in After.script()


def test_symbolic_variable_defined_by_outer_func():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x1: R.Tensor((n, m), "float32"), y1: R.Tensor((n, m), "float32")) -> R.Tensor(
            (n, m), "float32"
        ):
            @R.function
            def inner(x2: R.Tensor((n, m), "float32"), y2: R.Tensor((n, m), "float32")):
                sum_inner = R.add(x2, y2)
                return sum_inner

            sum_main = inner(x1, y1)
            return sum_main

    n_main = T.dynamic("n", "int64")
    m_main = T.dynamic("m", "int64")
    n_main_inner = T.dynamic("n", "int64")
    m_main_inner = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x1: R.Tensor((n_main, m_main), "float32"), y1: R.Tensor((n_main, m_main), "float32")
        ) -> R.Tensor((n_main, m_main), "float32"):
            sum_main = Expected.main_inner(x1, y1)
            return sum_main

        @R.function(private=True)
        def main_inner(
            x2: R.Tensor((n_main_inner, m_main_inner), "float32"),
            y2: R.Tensor((n_main_inner, m_main_inner), "float32"),
        ) -> R.Tensor((n_main_inner, m_main_inner), "float32"):
            sum_inner = R.add(x2, y2)
            return sum_inner

    After = transform.LambdaLift()(Before)
    assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
