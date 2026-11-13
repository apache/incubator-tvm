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
"""Test eliminate common subexpr pass"""

import numpy as np

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax.transform import EliminateCommonSubexpr
from tvm.s_tir import script as Ts
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def verify(input, expected, call_only=False):
    tvm_tirx.ir.assert_structural_equal(EliminateCommonSubexpr(call_only)(input), expected)


def test_simple():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv0 = R.add(x, y)
                lv1 = R.add(x, y)
                gv = R.multiply(lv0, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv0 = R.add(x, y)
                lv1 = lv0
                gv = R.multiply(lv0, lv0)
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_constants():
    @I.ir_module
    class Before:
        @R.function
        def foo() -> R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((2, 2), dtype="int32")):
            with R.dataflow():
                # we are not going to bind the constant 1 to a var
                lv0 = R.add(R.const(1, dtype="int32"), R.const(1, dtype="int32"))
                # we expect to bind the repeated large constants
                lv1 = R.add(
                    R.const(tvm_tirx.runtime.tensor(np.zeros((2, 2), dtype="int32"))),
                    R.const(tvm_tirx.runtime.tensor(np.zeros((2, 2), dtype="int32"))),
                )
                gv = (lv0, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo() -> R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((2, 2), dtype="int32")):
            with R.dataflow():
                lv0 = R.add(R.const(1, dtype="int32"), R.const(1, dtype="int32"))
                lv1 = R.add(
                    R.const(tvm_tirx.runtime.tensor(np.zeros((2, 2), dtype="int32"))),
                    R.const(tvm_tirx.runtime.tensor(np.zeros((2, 2), dtype="int32"))),
                )
                gv = (lv0, lv1)
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_repeated_inner_tuples():
    """CSE is only applied at variable bindings

    To remain consistent with the behavior of the normalizer, tuples
    are kept as-is, even if they contain repeated sub-tuples.
    """

    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                # repeated units: (x, x), (x, (x, x)), ((x, x), (x, (x, x)))
                tup = (((x, x), (x, (x, x))), ((x, x), (x, (x, x))), (x, (x, x)))
                gv = tup[0][0][1]
                R.output(gv)
            return gv

    Expected = Before

    verify(Before, Expected)


def test_inner_function():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                # we are going to do CSE inside the local function
                @R.function
                def bar(y: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
                    with R.dataflow():
                        # writing this out in ANF to illustrate why CSE behaves as it does
                        # result of ANF transforming R.add(R.add(y, y), R.add(y, y))
                        lv0 = R.add(y, y)
                        lv1 = R.add(y, y)
                        lv2 = R.add(lv0, lv1)
                        gv = lv2
                        R.output(gv)
                    return R.add(gv, gv)

                # also making the ANF explicit to better illustrate the result of CSE
                # result of ANF transforming R.add(R.add(bar(x), bar(x)), R.add(bar(x), bar(x)))
                lv0 = bar(x)
                lv1 = bar(x)
                lv2 = R.add(lv0, lv1)
                lv3 = bar(x)
                lv4 = bar(x)
                lv5 = R.add(lv3, lv4)
                lv6 = R.add(lv2, lv5)
                gv = lv6
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            with R.dataflow():

                @R.function
                def bar(y: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
                    with R.dataflow():
                        lv0 = R.add(y, y)
                        lv1 = lv0
                        lv2 = R.add(lv0, lv0)
                        gv = lv2
                        R.output(gv)
                    return R.add(gv, gv)

                # can further clean this up
                # using canonicalize bindings, eliminate unused bindings, and CSE again
                lv0 = bar(x)
                lv1 = lv0
                lv2 = R.add(lv0, lv0)
                lv3 = lv0
                lv4 = lv0
                lv5 = lv2
                lv6 = R.add(lv2, lv2)
                gv = lv6
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_call_only():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((160,), dtype="float32")):
            with R.dataflow():
                lv1 = R.arange(R.prim_value(0), R.prim_value(160), R.prim_value(1), dtype="float32")
                lv2 = R.arange(R.prim_value(0), R.prim_value(160), R.prim_value(1), dtype="float32")
                lv3 = R.add(x, lv1)
                out = R.add(lv3, lv2)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((160,), dtype="float32")) -> R.Tensor((160,), dtype="float32"):
            with R.dataflow():
                lv1 = R.arange(R.prim_value(0), R.prim_value(160), R.prim_value(1), dtype="float32")
                lv2 = lv1
                lv3 = R.add(x, lv1)
                out = R.add(lv3, lv1)
                R.output(out)
            return out

    verify(Before, Expected, call_only=True)


def test_cse_outside_dataflow():
    # same example as previously but it will work without a dataflow wrapper
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            lv0 = R.add(x, y)
            lv1 = R.add(x, y)
            gv = R.multiply(lv0, lv1)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            lv0 = R.add(x, y)
            lv1 = lv0
            gv = R.multiply(lv0, lv0)
            return gv

    verify(Before, Expected)


def test_no_cse_across_dataflow():
    # same example as previously but it will work without a dataflow wrapper
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv0 = R.add(x, y)
                lv1 = R.add(x, y)
                gv1 = R.multiply(lv0, lv1)
                R.output(gv1)

            _ = R.print("Prevent dataflow block merging")

            with R.dataflow():
                lv2 = R.add(x, y)
                lv3 = R.add(x, y)
                gv2 = R.multiply(lv2, lv3)
                R.output(gv2)

            gv3 = R.add(x, y)
            gv4 = R.add(x, y)
            gv5 = R.multiply(gv3, gv4)

            output = R.add(R.add(gv1, gv2), gv5)
            return output

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                # The R.add(x,y) may be de-duplicated within a dataflow block
                lv0 = R.add(x, y)
                lv1 = lv0
                gv1 = R.multiply(lv0, lv0)
                R.output(gv1)

            _ = R.print("Prevent dataflow block merging")

            with R.dataflow():
                # However, the later dataflow block may not be
                # de-duplicated using variables in the earlier block.
                lv2 = R.add(x, y)
                lv3 = lv2
                gv2 = R.multiply(lv2, lv2)
                R.output(gv2)

            # And while non-dataflow bindings can be de-duplicated,
            # they cannot be de-duplicated using bindings that were
            # valid in either of the earlier dataflow blocks.
            gv3 = R.add(x, y)
            gv4 = gv3
            gv5 = R.multiply(gv3, gv3)

            output = R.add(R.add(gv1, gv2), gv5)
            return output

    verify(Before, Expected)


def test_no_replacement_across_dataflow_boundary():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                A = R.add(x, y)
                # B has the same value as A, and so instances of B can be replaced with A.
                B = R.add(x, y)
                C = R.multiply(A, B)

                # However, B is exposed for use outside of the
                # DataflowBlock, while A is not.  Therefore, any
                # additional uses of `B` must NOT be replaced with
                # A.
                R.output(B, C)

            # In addition, because `A` is only valid within the
            # dataflow block, the `R.add(x,y)` cannot be de-duplicated
            # as another usage of `A`.
            D = R.add(x, y)
            return (B, C, D)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                A = R.add(x, y)
                B = A
                C = R.multiply(A, A)
                R.output(B, C)

            D = B
            return (B, C, B)

    verify(Before, Expected)


def test_do_not_eliminate_impure():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            # it's a repeated subexpression but it would be wrong to deduplicate it
            p1 = R.print("Message")
            p2 = R.print("Message")
            a1 = R.assert_op(R.const(False), "Always fails")
            lv0 = R.add(x, y)
            lv1 = R.add(x, y)
            gv = R.multiply(lv0, lv1)
            a2 = R.assert_op(R.const(False), "Always fails")
            return gv

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            p1 = R.print("Message")
            p2 = R.print("Message")
            a1 = R.assert_op(R.const(False), "Always fails")
            lv0 = R.add(x, y)
            lv1 = lv0
            gv = R.multiply(lv0, lv0)
            a2 = R.assert_op(R.const(False), "Always fails")
            return gv

    verify(Before, Expected)


def test_do_not_eliminate_shape_expr():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            x = R.reshape(x, [6])
            y = R.reshape(y, [6])
            z = R.add(x, y)
            return z

    Expected = Before

    verify(Before, Expected)


def test_do_not_eliminate_extern_func():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def foo(x: R.Tensor((2, 3), dtype="float32")):
            y = R.call_packed("extern_func_name", x, ty_args=R.Tensor([2, 3]))
            z = R.call_packed("extern_func_name", y, ty_args=R.Tensor([2, 3]))
            return z

    Expected = Before

    verify(Before, Expected)


def test_call_tir_tuple_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16, 16], "int32"), B: R.Tensor([16, 16], "int32")):
            cls = Before
            Prod = R.call_tir(cls.product, [A, B], ty_args=[R.Tensor([16, 16], "int32")])
            Sum = R.call_tir(cls.sum, [A, B], ty_args=[R.Tensor([16, 16], "int32")])
            return (Prod, Sum)

        @Ts.function(private=True)
        def product(
            A: T.Tensor([16, 16], "int32"),
            B: T.Tensor([16, 16], "int32"),
            C: T.Tensor([16, 16], "int32"),
        ):
            for (*iters,) in T.grid(*A.shape):
                with Ts.sblock("compute"):
                    i, j = Ts.axis.remap("SS", iters)
                    C[i, j] = A[i, j] * B[i, j]

        @Ts.function(private=True)
        def sum(
            A: T.Tensor([16, 16], "int32"),
            B: T.Tensor([16, 16], "int32"),
            C: T.Tensor([16, 16], "int32"),
        ):
            for (*iters,) in T.grid(*A.shape):
                with Ts.sblock("compute"):
                    i, j = Ts.axis.remap("SS", iters)
                    C[i, j] = A[i, j] + B[i, j]

    Expected = Before

    # If EliminateCommonSubexpr produces unnormalized expressions,
    # normalization of those expressions may produce additional
    # variables bindings.  This test case should be agnostic to those
    # additional bindings, so DCE is applied after CSE.
    After = tvm_tirx.ir.transform.Sequential(
        [
            EliminateCommonSubexpr(),
            tvm.relax.transform.DeadCodeElimination(),
        ]
    )(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_do_not_eliminate_dtype():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def foo() -> R.Tensor((32, 64), "int32"):
            obj: R.Any = R.vm.alloc_storage(R.shape([24576]), runtime_device_index=0, dtype="uint8")
            a: R.Tensor([32, 64], dtype="int32") = R.vm.alloc_tensor(
                obj, offset=0, shape=R.shape([32, 64]), dtype="int32"
            )
            ret_val: R.Tensor([32, 64], dtype="int32") = R.builtin.alloc_tensor(
                R.shape([32, 64]), R.dtype("int32"), R.prim_value(0)
            )
            _t1: R.Tuple = R.vm.kill_object(a)
            _t3: R.Tuple = R.vm.kill_object(obj)
            lv: R.Tensor([32, 64], dtype="int32") = ret_val
            return lv

    Expected = Before

    verify(Before, Expected)


def test_match_cast():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                A1 = R.add(x, y)
                B1 = R.match_cast(A1, R.Tensor([2, 3], "float32"))

                A2 = R.add(x, y)
                B2 = R.match_cast(A2, R.Tensor([2, 3], "float32"))

                gv = R.multiply(B1, B2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                A1 = R.add(x, y)
                B1 = R.match_cast(A1, R.Tensor([2, 3], "float32"))

                A2 = A1
                B2 = B1
                gv = R.multiply(B1, B1)
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_match_cast_with_symbolic_vars():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")
    p = T.dynamic("p", "int64")
    q = T.dynamic("q", "int64")

    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor(dtype="float32"), y: R.Tensor(dtype="float32")):
            with R.dataflow():
                A1 = R.add(x, y)

                B1 = R.match_cast(A1, R.Tensor([n, m], "float32"))

                A2 = R.add(x, y)
                B2 = R.match_cast(A2, R.Tensor([p, q], "float32"))

                gv = R.multiply(B1, B2)
                R.output(gv)
            return gv

    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")
    p = T.dynamic("p", "int64")
    q = T.dynamic("q", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor(dtype="float32"), y: R.Tensor(dtype="float32")):
            with R.dataflow():
                A1 = R.add(x, y)
                B1 = R.match_cast(A1, R.Tensor([n, m], "float32"))

                A2 = A1
                B2 = R.match_cast(A1, R.Tensor([p, q], "float32"))

                gv = R.multiply(B1, B2)
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_replace_binding_within_branch_with_duplicate_before_branch():
    """Bindings before a branch may be used within the branch"""

    @I.ir_module
    class Before:
        @R.function
        def foo(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
            condition: T.bool,
        ):
            A = R.add(x, y)
            if condition:
                B = R.add(x, y)
                C = R.multiply(x, B)
                D = R.multiply(A, C)
            else:
                B = R.add(x, y)
                C = R.multiply(y, B)
                D = R.multiply(A, C)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def foo(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
            condition: T.bool,
        ):
            A = R.add(x, y)
            if condition:
                B = A
                C = R.multiply(x, A)
                D = R.multiply(A, C)
            else:
                B = A
                C = R.multiply(y, A)
                D = R.multiply(A, C)
            return D

    verify(Before, Expected)


def test_keep_duplicate_across_if_and_then():
    """Bindings in `if` are not valid within `else`"""

    @I.ir_module
    class Before:
        @R.function
        def foo(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
            condition: T.bool,
        ):
            if condition:
                A = R.add(x, y)
                B = R.multiply(x, A)
            else:
                A = R.add(x, y)
                B = R.multiply(y, A)
            return B

    Expected = Before

    verify(Before, Expected)


def test_keep_duplicate_after_branch():
    """Only the final binding is valid after a if/else branch"""

    @I.ir_module
    class Before:
        @R.function
        def foo(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
            condition: T.bool,
        ):
            if condition:
                A = R.add(x, y)
                B = R.multiply(x, A)
            else:
                A = R.add(x, y)
                B = R.multiply(y, A)

            C = R.add(x, y)
            D = R.multiply(B, C)
            return D

    Expected = Before

    verify(Before, Expected)


def test_keep_alloc_tensor():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32")):
            tmp_buf1 = R.builtin.alloc_tensor(R.shape([64]), R.dtype("int32"), R.prim_value(0))
            tmp_buf2 = R.builtin.alloc_tensor(R.shape([64]), R.dtype("int32"), R.prim_value(0))
            out = R.add(tmp_buf1, tmp_buf2)
            return out

    Expected = Before

    verify(Before, Expected)


def test_keep_alloc_storage():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32")):
            tmp_storage1 = R.vm.alloc_storage(R.shape([64]), runtime_device_index=0, dtype="uint8")
            tmp_buf1 = R.vm.alloc_tensor(tmp_storage1, offset=0, shape=R.shape([64]), dtype="int32")
            tmp_storage2 = R.vm.alloc_storage(R.shape([64]), runtime_device_index=0, dtype="uint8")
            tmp_buf2 = R.vm.alloc_tensor(tmp_storage2, offset=0, shape=R.shape([64]), dtype="int32")
            out = R.add(tmp_buf1, tmp_buf2)
            return out

    Expected = Before

    verify(Before, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
