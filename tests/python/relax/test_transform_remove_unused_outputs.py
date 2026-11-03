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

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_simple():
    @I.ir_module
    class Before:
        @R.function
        def main():
            args = Before.func()
            return args[0]

        @R.function(private=True)
        def func() -> R.Tuple([R.Tensor, R.Tensor]):
            A = R.zeros([16, 16], "int32")
            B = R.ones([16, 16], "int32")
            return (A, B)

    @I.ir_module
    class Expected:
        @R.function
        def main():
            A = Expected.func()
            return A

        @R.function(private=True)
        def func() -> R.Tensor([16, 16], "int32"):
            A = R.zeros([16, 16], "int32")
            return A

    After = tvm.relax.transform.RemoveUnusedOutputs()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_use_multiple_outputs():
    @I.ir_module
    class Before:
        @R.function
        def main():
            args = Before.func()
            return (args[0], args[2])

        @R.function(private=True)
        def func() -> R.Tuple([R.Tensor, R.Tensor, R.Tensor]):
            A = R.zeros([16, 16], "int32")
            B = R.ones([16, 16], "int32")
            C = R.zeros([32, 32], "int32")
            return (A, B, C)

    @I.ir_module
    class Expected:
        @R.function
        def main():
            args = Expected.func()
            return (args[0], args[1])

        @R.function(private=True)
        def func() -> R.Tuple([R.Tensor([16, 16], "int32"), R.Tensor([32, 32], "int32")]):
            A = R.zeros([16, 16], "int32")
            C = R.zeros([32, 32], "int32")
            return (A, C)

    After = tvm.relax.transform.RemoveUnusedOutputs()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_multiple_call_sites():
    @I.ir_module
    class Before:
        @R.function
        def main_a():
            args = Before.func()
            return args[0]

        @R.function
        def main_b():
            args = Before.func()
            return args[2]

        @R.function(private=True)
        def func() -> R.Tuple([R.Tensor, R.Tensor, R.Tensor]):
            A = R.zeros([16, 16], "int32")
            B = R.ones([16, 16], "int32")
            C = R.zeros([32, 32], "int32")
            return (A, B, C)

    @I.ir_module
    class Expected:
        @R.function
        def main_a():
            args = Expected.func()
            return args[0]

        @R.function
        def main_b():
            args = Expected.func()
            return args[1]

        @R.function(private=True)
        def func() -> R.Tuple([R.Tensor([16, 16], "int32"), R.Tensor([32, 32], "int32")]):
            A = R.zeros([16, 16], "int32")
            C = R.zeros([32, 32], "int32")
            return (A, C)

    After = tvm.relax.transform.RemoveUnusedOutputs()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_return_tuple():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16, 16], "int32")):
            B = R.add(A, A)
            out_tuple = Before.func(B)
            return out_tuple

        @R.function(private=True)
        def func(B: R.Tensor([16, 16], "int32")) -> R.Tuple(
            R.Tensor([16, 16], "int32"), R.Tensor([16, 16], "int32")
        ):
            C = R.multiply(B, B)
            D = R.add(B, B)
            return (C, D)

    Expected = Before

    After = tvm.relax.transform.RemoveUnusedOutputs()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_unused_impure_call():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor((2,), "float32")):
            _ = Before.effect(x)
            return x

        @R.function(private=True, pure=False)
        def effect(x: R.Tensor((2,), "float32")) -> R.Tuple(
            R.Tensor((2,), "float32"), R.Tensor((2,), "float32")
        ):
            R.print(format="observable effect")
            unused = R.add(x, x)
            return (x, unused)

    After = tvm.relax.transform.RemoveUnusedOutputs()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Before)


def test_unused_pure_call():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((2,), "float32")):
            _ = Before.func(x)
            return x

        @R.function(private=True)
        def func(x: R.Tensor((2,), "float32")) -> R.Tuple(
            R.Tensor((2,), "float32"), R.Tensor((2,), "float32")
        ):
            unused = R.add(x, x)
            return (x, unused)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2,), "float32")):
            return x

    After = tvm.relax.transform.RemoveUnusedOutputs()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("num_calls", [1, 2])
@pytest.mark.parametrize("use_output", [False, True])
def test_impure_call_execution(num_calls, use_output, capsys):
    @I.ir_module
    class Module:
        @R.function(private=True, pure=False)
        def effect(x: R.Tensor((2,), "float32")):
            R.print(format="unused_outputs_effect")
            return (x, x)

    bb = tvm.relax.BlockBuilder(Module)
    x = tvm.relax.Var("x", tvm.relax.TensorType((2,), "float32"))
    with bb.function("main", [x], pure=False):
        for _ in range(num_calls):
            outputs = bb.emit(tvm.relax.Call(Module.get_global_var("effect"), [x]))
        result = bb.emit(tvm.relax.TupleGetItem(outputs, 0)) if use_output else x
        bb.emit_func_output(result)

    Before = bb.get()
    After = tvm.relax.transform.RemoveUnusedOutputs()(Before)
    values = np.array([1, 3], dtype="float32")
    for mod in [Before, After]:
        executable = tvm.compile(mod, target="c")
        vm = tvm.relax.VirtualMachine(executable, tvm_tirx.cpu())
        capsys.readouterr()
        result = vm["main"](tvm_tirx.runtime.tensor(values))
        np.testing.assert_array_equal(result.numpy(), values)
        assert capsys.readouterr().out.count("unused_outputs_effect") == num_calls

    return_type = After["effect"].ty.ret
    if use_output:
        assert isinstance(return_type, tvm.relax.TensorType)
    else:
        assert isinstance(return_type, tvm_tirx.ir.TupleType)
        assert len(return_type.fields) == 2


if __name__ == "__main__":
    tvm_tirx.testing.main()
