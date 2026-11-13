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

import re

import pytest

import tvm_tirx
from tvm import relax
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import tirx as T
from tvm_tirx.script.parser import relax as R


def test_copy_with_new_vars():
    @R.function
    def before(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
        gv = R.add(x, y)
        return gv

    after = relax.utils.copy_with_new_vars(before)
    assert_structural_equal(after, before)

    assert len(after.params) == len(before.params)
    for before_var, after_var in zip(before.params, after.params):
        assert before_var != after_var


def test_copy_with_new_vars_copied_symbolic_vars():
    m = T.dynamic("m", "int64")

    @R.function
    def before(x: R.Tensor((m,), "float32"), y: R.Tensor((m,), "float32")):
        gv = R.add(x, y)
        return gv

    after = relax.utils.copy_with_new_vars(before)
    assert_structural_equal(after, before)

    assert len(after.params) == len(before.params)
    for before_var, after_var in zip(before.params, after.params):
        assert before_var != after_var
        assert before_var.ty.shape[0] != after_var.ty.shape[0]


def test_copy_with_new_vars_on_ir_module():
    @tvm_tirx.script.ir_module
    class Actual:
        @R.function
        def func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            gv = R.add(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            gv = R.add(x, y)
            return gv

        @R.function
        def func_copied(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            gv = R.add(x, y)
            return gv

    Actual["func_copied"] = relax.utils.copy_with_new_vars(Actual["func"]).with_attr(
        "global_symbol", "func_copied"
    )

    # Assertion will fail if the f_copied contains the same VarObj that's used in
    # the original function, due to var mapping during structural equal.
    assert_structural_equal(Actual, Expected)


def test_copy_with_new_vars_on_ir_module_nested_function():
    @tvm_tirx.script.ir_module
    class Actual:
        @R.function
        def func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            @R.function
            def inner(x: R.Tensor((3,), "float32")) -> R.Tensor((3,), dtype="float32"):
                gv = R.add(x, x)
                return gv

            gv = R.add(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            @R.function
            def inner(x: R.Tensor((3,), "float32")) -> R.Tensor((3,), dtype="float32"):
                gv = R.add(x, x)
                return gv

            gv = R.add(x, y)
            return gv

        @R.function
        def func_copied(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")):
            @R.function
            def inner(x: R.Tensor((3,), "float32")) -> R.Tensor((3,), dtype="float32"):
                gv = R.add(x, x)
                return gv

            gv = R.add(x, y)
            return gv

    Actual["func_copied"] = relax.utils.copy_with_new_vars(Actual["func"]).with_attr(
        "global_symbol", "func_copied"
    )

    assert_structural_equal(Actual, Expected)


def test_assert_structural_equal_in_seqexpr():
    """The first mismatch is correctly identified."""

    @R.function(private=True)
    def func_1(A: R.Tensor([16, 16], "float32")):
        B = R.concat([A, A])
        return B

    @R.function(private=True)
    def func_2(A: R.Tensor([16, 16], "float32")):
        B = R.add(A, A)
        C = R.add(B, B)
        return B

    with pytest.raises(
        ValueError,
        match=re.escape("<root>.ty.ret.shape.ty.values[0].value"),
    ):
        assert_structural_equal(func_1, func_2)


def test_structural_equal_of_call_nodes():
    """relax.Call must be compared by structural equality, not reference"""

    # Three identical calls to relax.op.zeros
    calls_to_op_zero = [relax.op.zeros([16], "int32") for _ in range(3)]

    @R.function(private=True)
    def uses_same_object_twice():
        A = calls_to_op_zero[0]
        B = calls_to_op_zero[0]
        C = R.add(A, B)
        return C

    @R.function(private=True)
    def uses_two_different_objects():
        A = calls_to_op_zero[1]
        B = calls_to_op_zero[2]
        C = R.add(A, B)
        return C

    tvm_tirx.ir.assert_structural_equal(uses_same_object_twice, uses_two_different_objects)


def test_structural_equal_with_recursive_lambda_function():
    """A recursive lambda function may be checked for structural equality

    Recursive function definitions may reference the bound variable
    within the value being bound.  In these cases, the `DefEqual(var,
    other->var)` must occur first, to ensure it is defined at point of
    use.

    In all other cases, checking for structural equality of the bound
    value prior to the variable provides a better error message.
    """

    def define_function():
        @R.function
        def func(n: T.int64):
            @R.function
            def recursive_lambda(i_arg: T.int64) -> T.int64:
                condition = R.equal(i_arg, R.prim_value(0))
                if condition:
                    output = R.prim_value(0)
                else:
                    next_i = R.subtract(i_arg, R.prim_value(1))
                    remainder = recursive_lambda(next_i)
                    output = R.add(i_arg, remainder)
                return output

            return recursive_lambda(n)

        return func

    func_1 = define_function()
    func_2 = define_function()

    tvm_tirx.ir.assert_structural_equal(func_1, func_2)


def test_structural_equal_with_distinct_recursive_lambda_function():
    """A recursive lambda function may be checked for structural equality

    Like `test_structural_equal_with_recursive_lambda_function`, but
    comparing between two distinct functions.
    """

    @R.function(private=True)
    def func_a(n: T.int64):
        @R.function
        def recursive_lambda(i_arg: T.int64) -> T.int64:
            condition = R.equal(i_arg, R.prim_value(0))
            if condition:
                output = R.prim_value(0)
                #                          ^
                # The first mismatch is here
            else:
                next_i = R.subtract(i_arg, R.prim_value(1))
                remainder = recursive_lambda(next_i)
                output = R.add(i_arg, remainder)
            return output

        return recursive_lambda(n)

    @R.function(private=True)
    def func_b(n: T.int64):
        @R.function
        def recursive_lambda(i_arg: T.int64) -> T.int64:
            condition = R.equal(i_arg, R.prim_value(0))
            if condition:
                output = R.prim_value(1)
                #                          ^
                # The first mismatch is here
            else:
                next_i = R.subtract(i_arg, R.prim_value(1))
                remainder = recursive_lambda(next_i)
                output = R.multiply(i_arg, remainder)
            return output

        return recursive_lambda(n)

    # The path to the first mismatch, which should appear within the
    # error message.
    mismatch_path = [
        "<root>",
        "body",
        "blocks[0]",
        "bindings[0]",
        "value",
        "body",
        "blocks[0]",
        "bindings[1]",
        "value",
        "true_branch",
        "body",
        "value",
    ]

    with pytest.raises(ValueError, match=re.escape(".".join(mismatch_path))):
        tvm_tirx.ir.assert_structural_equal(func_a, func_b)


if __name__ == "__main__":
    pytest.main([__file__])
