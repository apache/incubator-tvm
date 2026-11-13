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

import sys

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import te
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T

# TODO(csullivan): Additional tests cases needed:
# - Function with 1 arg, inplace update
# - Function with tensor that uses custom storage_scope


@Ts.function
def func_1(A: T.Tensor((16,), "float32"), C: T.Tensor((1,), "float32")):
    for i in T.serial(
        0,
        16,
    ):
        with Ts.sblock():
            B = Ts.sblock_alloc_tensor((1,), dtype="float32")
            with Ts.sblock():
                B[0] = A[i] * T.float32(2)
            with Ts.sblock():
                C[0] = C[0] + A[i] + B[0] + T.float32(1)
                A[i] = B[0] + T.float32(1)


def verify_func_1(module):
    a_np = np.random.randint(low=-128, high=127, size=(16,)).astype(np.float32)
    c_np = np.zeros((1,), dtype=np.float32)
    a = tvm_tirx.runtime.tensor(a_np, device=tvm_tirx.cpu(0))
    c = tvm_tirx.runtime.tensor(c_np, device=tvm_tirx.cpu(0))

    module(a, c)
    tvm_tirx.testing.assert_allclose(c_np + np.sum(3 * a_np + 1), c.numpy(), rtol=1e-4)
    # also test in place update
    tvm_tirx.testing.assert_allclose(a_np * 2 + 1, a.numpy(), rtol=1e-4)


@Ts.function
def func_2(
    C: T.Tensor((1,), "float32"), A: T.Tensor((16,), "float32"), D: T.Tensor((2,), "float32")
):
    for i in T.serial(
        0,
        16,
    ):
        with Ts.sblock():
            B = Ts.sblock_alloc_tensor((1,), dtype="float32")
            with Ts.sblock():
                B[0] = A[i] * T.float32(2)
            with Ts.sblock():
                C[0] = C[0] + A[i] + B[0] + T.float32(1) + D[0]
                A[i] = B[0] + T.float32(1) + D[1]


def verify_func_2(module):
    a_np = np.random.randint(low=-128, high=127, size=(16,)).astype(np.float32)
    d_np = np.random.randint(low=-128, high=127, size=(2,)).astype(np.float32)
    c_np = np.zeros((1,), dtype=np.float32)
    a = tvm_tirx.runtime.tensor(a_np, device=tvm_tirx.cpu(0))
    d = tvm_tirx.runtime.tensor(d_np, device=tvm_tirx.cpu(0))
    c = tvm_tirx.runtime.tensor(c_np, device=tvm_tirx.cpu(0))

    module(c, a, d)
    tvm_tirx.testing.assert_allclose(c_np + np.sum(3 * a_np + 1 + d_np[0]), c.numpy(), rtol=1e-4)
    tvm_tirx.testing.assert_allclose(a_np * 2 + 1 + d_np[1], a.numpy(), rtol=1e-4)


@Ts.function
def func_3(
    C: T.Tensor((1,), "float32"),
    A: T.Tensor((16,), "float32"),
    D: T.Tensor((2,), "float32"),
    E: T.Tensor((16,), "float32"),
    F: T.Tensor((16,), "float32"),
):
    for i in T.serial(
        0,
        16,
    ):
        with Ts.sblock():
            B = Ts.sblock_alloc_tensor((1,), dtype="float32")
            with Ts.sblock():
                B[0] = A[i] * T.float32(2)
            with Ts.sblock():
                E[i] = A[i]
                F[i] = E[i] + 1.0
                C[0] = C[0] + A[i] + B[0] + T.float32(1) + D[0]
                A[i] = B[0] + T.float32(1) + D[1]


def verify_func_3(module):
    a_np = np.random.randint(low=-128, high=127, size=(16,)).astype(np.float32)
    d_np = np.random.randint(low=-128, high=127, size=(2,)).astype(np.float32)
    c_np = np.zeros((1,), dtype=np.float32)
    e_np = np.zeros((16,), dtype=np.float32)
    f_np = np.zeros((16,), dtype=np.float32)
    a = tvm_tirx.runtime.tensor(a_np, device=tvm_tirx.cpu(0))
    d = tvm_tirx.runtime.tensor(d_np, device=tvm_tirx.cpu(0))
    c = tvm_tirx.runtime.tensor(c_np, device=tvm_tirx.cpu(0))
    e = tvm_tirx.runtime.tensor(e_np, device=tvm_tirx.cpu(0))
    f = tvm_tirx.runtime.tensor(f_np, device=tvm_tirx.cpu(0))

    module(c, a, d, e, f)
    tvm_tirx.testing.assert_allclose(c_np + np.sum(3 * a_np + 1 + d_np[0]), c.numpy(), rtol=1e-4)
    tvm_tirx.testing.assert_allclose(a_np * 2 + 1 + d_np[1], a.numpy(), rtol=1e-4)
    tvm_tirx.testing.assert_allclose(a_np, e.numpy(), rtol=1e-4)
    tvm_tirx.testing.assert_allclose(a_np + 1, f.numpy(), rtol=1e-4)


@Ts.function
def func_4(
    C: T.Tensor((1,), "float32"),
    A: T.Tensor((16,), "float32"),
    F: T.Tensor((16,), "float32"),
    D: T.Tensor((2,), "float32"),
    E: T.Tensor((16,), "float32"),
):
    for i in T.serial(
        0,
        16,
    ):
        with Ts.sblock():
            B = Ts.sblock_alloc_tensor((1,), dtype="float32")
            with Ts.sblock():
                B[0] = A[i] * T.float32(2)
            with Ts.sblock():
                E[i] = A[i]
                F[i] = E[i] + 1.0
                C[0] = C[0] + A[i] + B[0] + T.float32(1) + D[0]
                A[i] = B[0] + T.float32(1) + D[1]


def verify_func_4(module):
    a_np = np.random.randint(low=-128, high=127, size=(16,)).astype(np.float32)
    d_np = np.random.randint(low=-128, high=127, size=(2,)).astype(np.float32)
    c_np = np.zeros((1,), dtype=np.float32)
    e_np = np.zeros((16,), dtype=np.float32)
    f_np = np.zeros((16,), dtype=np.float32)
    a = tvm_tirx.runtime.tensor(a_np, device=tvm_tirx.cpu(0))
    d = tvm_tirx.runtime.tensor(d_np, device=tvm_tirx.cpu(0))
    c = tvm_tirx.runtime.tensor(c_np, device=tvm_tirx.cpu(0))
    e = tvm_tirx.runtime.tensor(e_np, device=tvm_tirx.cpu(0))
    f = tvm_tirx.runtime.tensor(f_np, device=tvm_tirx.cpu(0))

    module(c, a, f, d, e)
    tvm_tirx.testing.assert_allclose(c_np + np.sum(3 * a_np + 1 + d_np[0]), c.numpy(), rtol=1e-4)
    tvm_tirx.testing.assert_allclose(a_np * 2 + 1 + d_np[1], a.numpy(), rtol=1e-4)
    tvm_tirx.testing.assert_allclose(a_np, e.numpy(), rtol=1e-4)
    tvm_tirx.testing.assert_allclose(a_np + 1, f.numpy(), rtol=1e-4)


_function_cases = [
    [func_1, ("A"), verify_func_1],
    [func_2, ("C", "D"), verify_func_2],
    [func_3, ("C", "A", "D", "E"), verify_func_3],
    [func_4, ("C", "A", "D", "E"), verify_func_4],
]


class TestFunctions:
    @pytest.mark.parametrize("func,verify", [(case[0], case[2]) for case in _function_cases])
    def test_function_call(self, func, verify):
        target = tvm_tirx.target.Target("llvm")
        func = tvm.compile(func, target=target)
        verify(func)

    @pytest.mark.parametrize("func,params,verify", _function_cases)
    def test_te_extern_call(self, func, params, verify):
        ir_mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
        function = ir_mod["main"]

        buf_name_map = {
            param.name: param for param in func.params if tvm_tirx.tirx.is_tensor_var(param)
        }
        input_tensors = [te.placeholder(buf_name_map[name].shape) for name in params]
        output = te.extern_function(input_tensors, function)
        rt_function = te.create_function(tensors_from_extern_op(output, function))

        target = tvm_tirx.target.Target("llvm")
        func = tvm.compile(rt_function, target=target)
        verify(func)


def tensors_from_extern_op(extern, func):
    if isinstance(extern, list):
        output_tensors = extern
    else:
        output_tensors = [extern]
    output_tir_tensors = []
    input_tir_tensors = []
    input_tensors = []
    for ext in output_tensors:
        output_tir_tensors.extend(ext.op.output_placeholders)
        input_tir_tensors.extend(ext.op.input_placeholders)
        input_tensors.extend(ext.op.input_tensors)
    input_binds = dict(zip(input_tir_tensors, input_tensors))
    output_binds = dict(zip(output_tir_tensors, output_tensors))
    tir_to_te_tensor = {**input_binds, **output_binds}
    ordered_tensors = []
    for var in func.params:
        if tvm_tirx.tirx.is_tensor_var(var):
            ordered_tensors.append(tir_to_te_tensor[var])
    return ordered_tensors


if __name__ == "__main__":
    tvm_tirx.testing.main()
