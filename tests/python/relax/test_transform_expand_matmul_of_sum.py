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

import inspect

import pytest

import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I


class Base:
    def test_compare(self):
        transform = relax.transform.ExpandMatmulOfSum()

        if inspect.isclass(self.Expected) and issubclass(self.Expected, Exception):
            with pytest.raises(self.Expected):
                transform(self.Before)
        else:
            after = transform(self.Before)
            tvm_tirx.ir.assert_structural_equal(self.Expected, after)


class TestSimple(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([16, 32], "float32"),
            B: R.Tensor([16, 32], "float32"),
        ) -> R.Tensor([32], "float32"):
            weight = R.add(A, B)
            out = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([16, 32], "float32"),
            B: R.Tensor([16, 32], "float32"),
        ) -> R.Tensor([32], "float32"):
            lhs = R.matmul(x, A)
            rhs = R.matmul(x, B)
            out = R.add(lhs, rhs)
            return out


class TestNoExpansionOfCompileTimeAddition(Base):
    """Do not expand compile-time parameters

    This expansion is primarily to prepare the function for a later
    use of `CombineParallelMatmul`.  If the addition can be performed
    at compile-time, this is preferable.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([16, 32], "float32"),
            B: R.Tensor([16, 32], "float32"),
        ) -> R.Tensor([32], "float32"):
            R.func_attr({"num_input": 1})
            weight = R.add(A, B)
            out = R.matmul(x, weight)
            return out

    Expected = Before


class TestExpansionOfRuntimeAddition(Base):
    """Expand runtime addition

    This expansion is primarily to prepare the function for a later
    use of `CombineParallelMatmul`.  The expansion to `x*A + x*B`
    should occur iff `A+B` is not computable at compile-time.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([16, 32], "float32"),
            B: R.Tensor([16, 32], "float32"),
        ) -> R.Tensor([32], "float32"):
            R.func_attr({"num_input": 2})
            weight = R.add(A, B)
            out = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([16, 32], "float32"),
            B: R.Tensor([16, 32], "float32"),
        ) -> R.Tensor([32], "float32"):
            R.func_attr({"num_input": 2})
            lhs = R.matmul(x, A)
            rhs = R.matmul(x, B)
            out = R.add(lhs, rhs)
            return out


class TestRHSPermuteDims(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([32, 16], "float32"),
            B: R.Tensor([32, 16], "float32"),
        ) -> R.Tensor([32], "float32"):
            linear_weight = R.add(A, B)
            matmul_weight = R.permute_dims(linear_weight)
            out = R.matmul(x, matmul_weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([32, 16], "float32"),
            B: R.Tensor([32, 16], "float32"),
        ) -> R.Tensor([32], "float32"):
            A_transpose = R.permute_dims(A)
            lhs = R.matmul(x, A_transpose)
            B_transpose = R.permute_dims(B)
            rhs = R.matmul(x, B_transpose)
            out = R.add(lhs, rhs)
            return out


if __name__ == "__main__":
    tvm_tirx.testing.main()
