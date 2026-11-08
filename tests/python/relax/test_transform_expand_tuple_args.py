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
        def main(A: R.Tensor, B: R.Tensor):
            return Before.func((A, B))

        @R.function(private=True)
        def func(args: R.Tuple([R.Tensor, R.Tensor])) -> R.Tensor:
            return args[0]

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return Expected.func(A, B)

        @R.function(private=True)
        def func(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            return A

    After = tvm.relax.transform.ExpandTupleArguments()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_nested():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, B: R.Tensor, C: R.Tensor, D: R.Tensor) -> R.Tensor:
            return Before.func(((A, B), (C, D)))

        @R.function(private=True)
        def func(
            args: R.Tuple(
                [
                    R.Tuple([R.Tensor, R.Tensor]),
                    R.Tuple([R.Tensor, R.Tensor]),
                ]
            ),
        ) -> R.Tensor:
            return args[0][1]

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, B: R.Tensor, C: R.Tensor, D: R.Tensor) -> R.Tensor:
            return Expected.func(A, B, C, D)

        @R.function(private=True)
        def func(A: R.Tensor, B: R.Tensor, C: R.Tensor, D: R.Tensor) -> R.Tensor:
            return B

    After = tvm.relax.transform.ExpandTupleArguments()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_preserve_closure_callee():
    @I.ir_module
    class Before:
        @R.function(private=True)
        def pair_add(pair: R.Tuple(R.Tensor((2,), "float32"), R.Tensor((2,), "float32"))):
            return R.add(pair[0], pair[1])

        @R.function
        def main(x: R.Tensor((2,), "float32"), y: R.Tensor((2,), "float32")):
            closure = R.make_closure(Before.pair_add, ())
            return R.invoke_pure_closure(closure, ((x, y),), ty_args=R.Tensor((2,), "float32"))

    assert tvm.relax.analysis.check_well_formed(Before, check_ty=True)
    after = tvm.relax.transform.ExpandTupleArguments()(Before)
    assert tvm.relax.analysis.check_well_formed(after, check_ty=True)
    tvm_tirx.ir.assert_structural_equal(after, Before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
