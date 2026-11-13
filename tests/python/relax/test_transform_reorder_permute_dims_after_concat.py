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
        transform = relax.transform.ReorderPermuteDimsAfterConcat()

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
            x: R.Tensor([1, 32], "float32"),
            linear_weight_A: R.Tensor([128, 32], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                matmul_weight_A = R.permute_dims(linear_weight_A)
                matmul_weight_B = R.permute_dims(linear_weight_B)
                matmul_weight = R.concat([matmul_weight_A, matmul_weight_B], axis=1)
                out = R.matmul(x, matmul_weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1, 32], "float32"),
            linear_weight_A: R.Tensor([128, 32], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                linear_weight = R.concat([linear_weight_A, linear_weight_B], axis=0)
                matmul_weight = R.permute_dims(linear_weight)
                out = R.matmul(x, matmul_weight)
                R.output(out)
            return out


class TestDoNotRewriteUnknownRank(Base):
    """Do not rewrite implicit permutations when their rank is unknown."""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(dtype="float32"),
            y: R.Tensor(dtype="float32"),
        ):
            with R.dataflow():
                x_t = R.permute_dims(x)
                y_t = R.permute_dims(y)
                out = R.concat([x_t, y_t], axis=0)
                R.output(out)
            return out

    Expected = Before


class TestCombineExplicitAndImplicitAxes(Base):
    """Check for explicit axes to be permuted

    If `R.permute_dims` has no axes specified, it reverses the order
    of all axes.  For a 2-d argument, `R.permute_dims(arg)` and
    `R.permute_dims(arg, [1,0])` are equivalent, and should be
    able to be combinable.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1, 32], "float32"),
            linear_weight_A: R.Tensor([128, 32], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                matmul_weight_A = R.permute_dims(linear_weight_A)
                matmul_weight_B = R.permute_dims(linear_weight_B, axes=[1, 0])
                matmul_weight = R.concat([matmul_weight_A, matmul_weight_B], axis=1)
                out = R.matmul(x, matmul_weight)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1, 32], "float32"),
            linear_weight_A: R.Tensor([128, 32], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                linear_weight = R.concat([linear_weight_A, linear_weight_B], axis=0)
                matmul_weight = R.permute_dims(linear_weight)
                out = R.matmul(x, matmul_weight)
                R.output(out)
            return out


class TestDoNotCombineIncompatibleAxes(Base):
    """No change should be made for incompatible permutations

    The different `R.permute_dims` must each perform the same
    permutation for the reordering to be valid.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1, 32], "float32"),
            weight_A: R.Tensor([32, 128], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                matmul_weight_A = R.permute_dims(weight_A, axes=[0, 1])
                matmul_weight_B = R.permute_dims(linear_weight_B, axes=[1, 0])
                matmul_weight = R.concat([matmul_weight_A, matmul_weight_B], axis=1)
                out = R.matmul(x, matmul_weight)
                R.output(out)
            return out

    Expected = Before


class TestCheckForRewriteAfterIncompatibleChange(Base):
    """Check all R.permute_dims options, not just the first

    Complex conditionals may be implemented in the rewriter, rather
    than the pattern match.  In these cases, the rewriter may return
    the matched expression unmodified.  However, this prevents the
    pattern-matcher from checking later instances of the match.

    By moving the complex conditional to a `ConstrainedPattern`, the
    pattern-matcher can check against all possible matches.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1, 32], "float32"),
            weight_A: R.Tensor([32, 128], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
            linear_weight_C: R.Tensor([128, 32], "float32"),
            linear_weight_D: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                matmul_weight_A = R.permute_dims(weight_A, axes=[0, 1])
                matmul_weight_B = R.permute_dims(linear_weight_B, axes=[1, 0])
                matmul_weight_AB = R.concat([matmul_weight_A, matmul_weight_B], axis=1)
                out_AB = R.matmul(x, matmul_weight_AB)

                matmul_weight_C = R.permute_dims(linear_weight_C)
                matmul_weight_D = R.permute_dims(linear_weight_D)
                matmul_weight_CD = R.concat([matmul_weight_C, matmul_weight_D], axis=1)
                out_CD = R.matmul(x, matmul_weight_CD)

                out = (out_AB, out_CD)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1, 32], "float32"),
            weight_A: R.Tensor([32, 128], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
            linear_weight_C: R.Tensor([128, 32], "float32"),
            linear_weight_D: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                matmul_weight_A = R.permute_dims(weight_A, axes=[0, 1])
                matmul_weight_B = R.permute_dims(linear_weight_B, axes=[1, 0])
                matmul_weight_AB = R.concat([matmul_weight_A, matmul_weight_B], axis=1)
                out_AB = R.matmul(x, matmul_weight_AB)

                linear_weight_CD = R.concat([linear_weight_C, linear_weight_D], axis=0)
                matmul_weight_CD = R.permute_dims(linear_weight_CD)
                out_CD = R.matmul(x, matmul_weight_CD)

                out = (out_AB, out_CD)
                R.output(out)
            return out


class TestCheckForRewriteBeforeIncompatibleChange(Base):
    """Check all R.permute_dims options, not just the first

    Complex conditionals may be implemented in the rewriter, rather
    than the pattern match.  In these cases, the rewriter may return
    the matched expression unmodified.  However, this prevents the
    pattern-matcher from checking later instances of the match.

    By moving the complex conditional to a `ConstrainedPattern`, the
    pattern-matcher can check against all possible matches.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1, 32], "float32"),
            weight_A: R.Tensor([32, 128], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
            linear_weight_C: R.Tensor([128, 32], "float32"),
            linear_weight_D: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                matmul_weight_C = R.permute_dims(linear_weight_C)
                matmul_weight_D = R.permute_dims(linear_weight_D)
                matmul_weight_CD = R.concat([matmul_weight_C, matmul_weight_D], axis=1)
                out_CD = R.matmul(x, matmul_weight_CD)

                matmul_weight_A = R.permute_dims(weight_A, axes=[0, 1])
                matmul_weight_B = R.permute_dims(linear_weight_B, axes=[1, 0])
                matmul_weight_AB = R.concat([matmul_weight_A, matmul_weight_B], axis=1)
                out_AB = R.matmul(x, matmul_weight_AB)

                out = (out_AB, out_CD)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1, 32], "float32"),
            weight_A: R.Tensor([32, 128], "float32"),
            linear_weight_B: R.Tensor([128, 32], "float32"),
            linear_weight_C: R.Tensor([128, 32], "float32"),
            linear_weight_D: R.Tensor([128, 32], "float32"),
        ):
            with R.dataflow():
                linear_weight_CD = R.concat([linear_weight_C, linear_weight_D], axis=0)
                matmul_weight_CD = R.permute_dims(linear_weight_CD)
                out_CD = R.matmul(x, matmul_weight_CD)

                matmul_weight_A = R.permute_dims(weight_A, axes=[0, 1])
                matmul_weight_B = R.permute_dims(linear_weight_B, axes=[1, 0])
                matmul_weight_AB = R.concat([matmul_weight_A, matmul_weight_B], axis=1)
                out_AB = R.matmul(x, matmul_weight_AB)

                out = (out_AB, out_CD)
                R.output(out)
            return out


class TestNegativeConcatAxis(Base):
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1, 4, 8, 8], "float32"),
            y: R.Tensor([1, 4, 8, 8], "float32"),
        ):
            with R.dataflow():
                xt = R.permute_dims(x, axes=[0, 2, 3, 1])
                yt = R.permute_dims(y, axes=[0, 2, 3, 1])
                out = R.concat([xt, yt], axis=-1)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1, 4, 8, 8], "float32"),
            y: R.Tensor([1, 4, 8, 8], "float32"),
        ):
            with R.dataflow():
                merged = R.concat([x, y], axis=1)
                out = R.permute_dims(merged, axes=[0, 2, 3, 1])
                R.output(out)
            return out


if __name__ == "__main__":
    tvm_tirx.testing.main()
