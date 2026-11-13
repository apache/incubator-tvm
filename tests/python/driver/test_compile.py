#!/usr/bin/env python3
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
import platform

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax, te
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.codegen import Executable
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_compile_tir():
    """Test tvm.compile with TIR input."""
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    func = te.create_function([A, B, C])

    # Test compile with Function
    exec_prim = tvm.compile(func)
    assert isinstance(exec_prim, Executable)

    # Test compile with IRModule containing Function
    mod = tvm_tirx.IRModule.from_expr(func)
    exec_mod = tvm.compile(mod)
    assert isinstance(exec_mod, Executable)

    # Verify the compiled module works
    dev = tvm_tirx.cpu(0)
    a_np = np.random.uniform(size=10).astype(np.float32)
    b_np = np.random.uniform(size=10).astype(np.float32)
    a = tvm_tirx.runtime.tensor(a_np, dev)
    b = tvm_tirx.runtime.tensor(b_np, dev)
    c = tvm_tirx.runtime.tensor(np.zeros(10, dtype=np.float32), dev)

    exec_prim(a, b, c)
    tvm_tirx.testing.assert_allclose(c.numpy(), a_np + b_np)
    exec_mod(a, b, c)
    tvm_tirx.testing.assert_allclose(c.numpy(), a_np + b_np)


def test_compile_relax():
    """Test tvm.compile with Relax input."""

    # Define a simple Relax program
    @I.ir_module
    class MyModule:
        @R.function
        def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")) -> R.Tensor:
            z = R.add(x, y)
            return z

    # Test compile with Relax module
    target = tvm_tirx.target.Target("llvm")
    exec_relax = tvm.compile(MyModule, target)
    assert isinstance(exec_relax, Executable)

    # Verify the compiled module works
    dev = tvm_tirx.cpu(0)
    x_np = np.random.uniform(size=(3, 4)).astype(np.float32)
    y_np = np.random.uniform(size=(3, 4)).astype(np.float32)
    x = tvm_tirx.runtime.tensor(x_np, dev)
    y = tvm_tirx.runtime.tensor(y_np, dev)

    vm = relax.VirtualMachine(exec_relax, dev)
    z = vm["main"](x, y)
    tvm_tirx.testing.assert_allclose(z.numpy(), x_np + y_np)


@pytest.mark.skipif("32bit" in platform.architecture()[0], reason="skipping test for i386.")
def test_compile_mixed_module():
    @tvm_tirx.script.ir_module
    class MyModule:
        @Ts.function
        def add_one(X: T.Tensor((4,), "float32"), Y: T.Tensor((4,), "float32")):
            for i in range(4):
                Y[i] = X[i] + 1

        @R.function
        def main(x: R.Tensor((4,), "float32")):
            cls = MyModule
            with R.dataflow():
                y = R.call_tir(cls.add_one, [x], ty_args=[R.Tensor((4,), "float32")])
                return y

    # Test with custom pipeline
    target = tvm_tirx.target.Target("c")
    ex = tvm.compile(MyModule, target)
    assert isinstance(ex, Executable)

    dev = tvm_tirx.cpu(0)
    x = tvm_tirx.runtime.tensor(np.array([1, 2, 3, 4], dtype=np.float32), dev)
    y = tvm_tirx.runtime.tensor(np.zeros(4, dtype=np.float32), dev)
    # For tirx function, we can directly call the function
    ex["add_one"](x, y)
    tvm_tirx.testing.assert_allclose(y.numpy(), x.numpy() + 1)
    # For relax function, we need to use the vm to call the function
    vm = relax.VirtualMachine(ex, dev)
    z = vm["main"](x)
    tvm_tirx.testing.assert_allclose(z.numpy(), x.numpy() + 1)


if __name__ == "__main__":
    tvm_tirx.testing.main()
