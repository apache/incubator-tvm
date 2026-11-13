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

import re

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.ir import StringImm, TensorRegion
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

identify_memcpy = tvm.s_tir.analysis._ffi_api._identify_memcpy


def _check_memcpy_results(func, expected):
    """Check that identify_memcpy returns the expected results."""
    results = identify_memcpy(func.body)

    if isinstance(expected, str) or (
        isinstance(expected, tuple) and isinstance(expected[0], TensorRegion)
    ):
        expected = [expected]

    assert len(expected) == len(results)
    for exp, result in zip(expected, results):
        if isinstance(exp, str):
            assert isinstance(result, StringImm)
            assert re.search(exp, result.value)
        else:
            tvm_tirx.ir.assert_structural_equal(result, exp)


def test_1d():
    """Simplest test case"""

    @Ts.function
    def func(A: T.Tensor(1024, "float32"), B: T.Tensor(1024, "float32")):
        for i in T.serial(1024):
            B[i] = A[i]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    expected = (A[0:1024], B[0:1024])
    _check_memcpy_results(func, expected)


def test_1d_compute():
    """Like test_1d, but a computation prevents this being a memcpy"""

    @Ts.function
    def func(A: T.Tensor(1024, "float32"), B: T.Tensor(1024, "float32")):
        for i in T.serial(1024):
            B[i] = A[i] + 1.0

    expected = "Expected TensorStore's value to be TensorLoad"
    _check_memcpy_results(func, expected)


def test_1d_conditional():
    """Like test_1d, but a conditionals prevents this being a memcpy"""

    @Ts.function
    def func(A: T.Tensor(1024, "float32"), B: T.Tensor(1024, "float32")):
        for i in T.serial(1024):
            if i < 1024:
                B[i] = A[i]

    expected = "Expected innermost loop to have TensorStore body"
    _check_memcpy_results(func, expected)


def test_1d_strided_input():
    """Like test_1d, but strided input prevents this being a memcpy"""

    @Ts.function
    def func(A: T.Tensor(2048, "float32"), B: T.Tensor(1024, "float32")):
        for i in T.serial(1024):
            B[i] = A[i * 2]

    expected = "Mismatch between loop iterations (.*) and number of src indices"
    _check_memcpy_results(func, expected)


def test_1d_strided_output():
    """Like test_1d, but strided output prevents this being a memcpy"""

    @Ts.function
    def func(A: T.Tensor(1024, "float32"), B: T.Tensor(2048, "float32")):
        for i in T.serial(1024):
            B[i * 2] = A[i]

    expected = "Mismatch between loop iterations (.*) and number of dst indices"
    _check_memcpy_results(func, expected)


def test_1d_input_2d_output_fused_loop():
    """Like test_1d, but the output is written as a 2-d tensor"""

    @Ts.function
    def func(A: T.Tensor(1024, "float32"), B: T.Tensor((32, 32), "float32")):
        for i in T.serial(1024):
            B[i // 32, i % 32] = A[i]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    expected = (A[0:1024], B[0:32, 0:32])
    _check_memcpy_results(func, expected)


def test_2d_input_1d_output_fused_loop():
    """Like test_1d, but the input is written as a 2-d tensor"""

    @Ts.function
    def func(A: T.Tensor((32, 32), "float32"), B: T.Tensor(1024, "float32")):
        for i in T.serial(1024):
            B[i] = A[i // 32, i % 32]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    expected = (A[0:32, 0:32], B[0:1024])
    _check_memcpy_results(func, expected)


def test_1d_input_1d_output_nested_loop():
    """Like test_1d, but the iterator is written as a nested loop

    In test cases with more than one loop, each loop is checked to see
    if could be written as a memcpy.  The C++ utility function
    operates on individual loops, but for unit testing in Python, it
    is more convenient to return the results for all loops.
    """

    @Ts.function
    def func(A: T.Tensor(1024, "float32"), B: T.Tensor(1024, "float32")):
        for i, j in T.grid(32, 32):
            B[i * 32 + j] = A[i * 32 + j]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    i = func.body[0].var
    expected = [
        (A[0:1024], B[0:1024]),
        (A[i * 32 : i * 32 + 32], B[i * 32 : i * 32 + 32]),
    ]
    _check_memcpy_results(func, expected)


def test_1d_input_1d_output_nested_loop_equivalent_expressions():
    """Like test_1d_input_1d_output_nested_loop, but with equivalent indices

    If the expressions are not identical, the loops may still be
    recognizable as a memcpy, so long as the expressions are
    equivalent.
    """

    @Ts.function
    def func(A: T.Tensor(1024, "float32"), B: T.Tensor(1024, "float32")):
        for i, j in T.grid(32, 32):
            B[i * 32 + j] = A[j + i * 32]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    i = func.body[0].var
    expected = [
        (A[0:1024], B[0:1024]),
        (A[i * 32 : i * 32 + 32], B[i * 32 : i * 32 + 32]),
    ]
    _check_memcpy_results(func, expected)


def test_1d_input_2d_output_nested_loop():
    """Like test_1d_input_1d_output_nested_loop, but with a 2-d output tensor"""

    @Ts.function
    def func(A: T.Tensor(1024, "float32"), B: T.Tensor((32, 32), "float32")):
        for i, j in T.grid(32, 32):
            B[i, j] = A[i * 32 + j]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    i = func.body[0].var
    expected = [
        (A[0:1024], B[0:32, 0:32]),
        (A[i * 32 : i * 32 + 32], B[i, 0:32]),
    ]
    _check_memcpy_results(func, expected)


def test_2d_input_1d_output_nested_loop():
    """Like test_1d_input_1d_output_nested_loop, but with a 2-d input tensor"""

    @Ts.function
    def func(A: T.Tensor((32, 32), "float32"), B: T.Tensor(1024, "float32")):
        for i, j in T.grid(32, 32):
            B[i * 32 + j] = A[i, j]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    i = func.body[0].var
    expected = [
        (A[0:32, 0:32], B[0:1024]),
        (A[i, 0:32], B[i * 32 : i * 32 + 32]),
    ]
    _check_memcpy_results(func, expected)


def test_2d_input_2d_output_nested_loop():
    """Like test_1d_input_1d_output_nested_loop, but with 2-d input/output tensors"""

    @Ts.function
    def func(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32")):
        for i, j in T.grid(32, 32):
            B[i, j] = A[i, j]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    i = func.body[0].var
    expected = [
        (A[0:32, 0:32], B[0:32, 0:32]),
        (A[i, 0:32], B[i, 0:32]),
    ]
    _check_memcpy_results(func, expected)


def test_2d_input_2d_output_transpose_output():
    """test_2d_input_2d_output_nested_loop, but with a transposed output

    This is not recognized as a memcpy, because it results in a transpose.
    """

    @Ts.function
    def func(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32")):
        for i, j in T.grid(32, 32):
            B[j, i] = A[i, j]

    expected = [
        "different source",
        "Mismatch .* number of dst indices touched",
    ]
    _check_memcpy_results(func, expected)


def test_2d_input_2d_output_transpose_input():
    """test_2d_input_2d_output_nested_loop, but with a transposed input

    This is not recognized as a memcpy, because it results in a transpose.
    """

    @Ts.function
    def func(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32")):
        for i, j in T.grid(32, 32):
            B[i, j] = A[j, i]

    expected = [
        "different source",
        "Mismatch .* number of src indices touched",
    ]
    _check_memcpy_results(func, expected)


def test_2d_input_2d_output_transpose_both():
    """test_2d_input_2d_output_nested_loop, but with a transposed input

    The inner loop is not recognized as a memcpy, because it has
    strided access of both the input and output tensors.  However, the
    outer loop is still recognized as a memcpy, because the full
    region has been copied over, even though it occurs out of order.
    """

    @Ts.function
    def func(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32")):
        for i, j in T.grid(32, 32):
            B[j, i] = A[j, i]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    expected = [
        (A[0:32, 0:32], B[0:32, 0:32]),
        "Mismatch .* number of src indices touched",
    ]
    _check_memcpy_results(func, expected)


def test_cache_read():
    """Like test_2d_input_2d_output_nested_loop, but with a 1-d

    The inner loop is a memcpy of a single row at a time.  This
    pattern would appear when B is a read cache of A.
    """

    @Ts.function
    def func(A: T.Tensor((32, 32), "float32"), B: T.Tensor(32, "float32")):
        for i, j in T.grid(32, 32):
            B[j] = A[i, j]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    i = func.body[0].var
    expected = [
        "does not form a bijective transform",
        (A[i, 0:32], B[0:32]),
    ]
    _check_memcpy_results(func, expected)


def test_cache_write():
    """Like test_2d_input_2d_output_nested_loop, but with a 1-d

    The inner loop is a memcpy of a single row at a time.  This
    pattern would appear when A is a write cache of B.
    """

    @Ts.function
    def func(A: T.Tensor(32, "float32"), B: T.Tensor((32, 32), "float32")):
        for i, j in T.grid(32, 32):
            B[i, j] = A[j]

    A, B = [param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)]
    i = func.body[0].var
    expected = [
        "does not form a bijective transform",
        (A[0:32], B[i, 0:32]),
    ]
    _check_memcpy_results(func, expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
