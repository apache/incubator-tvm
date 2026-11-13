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


if __name__ == "__main__":
    tvm_tirx.testing.main()
