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
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I


# functions that will not change
def test_trivial():
    @I.ir_module
    class Before:
        # already a DF block
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            with R.dataflow():
                x = R.add(A, B)
                y = R.multiply(x, A)
                z = R.add(x, y)
                q = R.multiply(y, z)
                p = R.add(z, q)
                R.output(p)
            return p

        # too small
        @R.function
        def func(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            x = R.add(A, B)
            return x

        # too few pure ops between non-dataflow ops
        @R.function(pure=False)
        def func2(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            _ = R.print("Hi there!")
            y = R.add(A, B)
            _ = R.print("Sum: {}", y)
            x = R.multiply(y, y)
            if R.const(False):
                _ = R.print("True branch")
                q = R.add(x, y)
                _ = R.print("Value of q: {}", q)
                w = q
            else:
                _ = R.print("False branch")
                q = R.subtract(x, y)
                _ = R.print("Value of q: {}", q)
                w = q
            p = R.multiply(w, w)
            return p

    Expected = Before

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_basic():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            w = R.multiply(z, y)
            v = R.add(w, x)
            return v

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)
            return v

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_multiple_blocks():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            w = R.multiply(z, y)
            v = R.add(w, x)
            _ = R.print("Hi mom!")
            a = R.multiply(v, v)
            b = R.add(a, a)
            c = R.subtract(b, a)
            d = R.add(c, c)
            return d

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)
            _ = R.print("Hi mom!")
            with R.dataflow():
                a = R.multiply(v, v)
                b = R.add(a, a)
                c = R.subtract(b, a)
                d = R.add(c, c)
                R.output(d)
            return d

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_extract_inside_branches():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            w = R.multiply(z, y)
            v = R.add(w, x)
            if R.const(True):
                q = R.multiply(v, v)
                a = R.add(q, q)
                b = R.multiply(a, a)
            else:
                q = R.add(v, v)
                a = R.multiply(q, q)
                b = R.add(a, a)
            c = R.multiply(b, b)
            d = R.add(c, c)
            e = R.multiply(d, d)
            return e

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)

            if R.const(True):
                with R.dataflow():
                    q = R.multiply(v, v)
                    a = R.add(q, q)
                    b = R.multiply(a, a)
                    R.output(b)
                # weird but the parser requires this construct
                c = b
            else:
                with R.dataflow():
                    q = R.add(v, v)
                    a = R.multiply(q, q)
                    b = R.add(a, a)
                    R.output(b)
                c = b
            with R.dataflow():
                d = R.multiply(c, c)
                e = R.add(d, d)
                f = R.multiply(e, e)
                R.output(f)
            return f

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_treat_non_call_as_pure():
    @I.ir_module
    class Before:
        @R.function
        def tuples_and_const(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            t1 = (x, y, x)
            t2 = (y, y, x)
            c = R.const([1, 2, 3], dtype="int32")
            return c

        @R.function
        def shapes() -> R.Shape:
            s1 = R.shape((1, 2, 3))
            s2 = R.shape((4, 5, 6))
            s3 = R.shape((7, 8, 9))
            return s3

        @R.function
        def prim_values():
            x = R.prim_value(1)
            y = R.prim_value(2)
            z = R.prim_value(3)
            return z

        @R.function
        def main(t: R.Tuple(R.Tensor, R.Tensor)) -> R.Tensor:
            x = t[0]
            y = t[1]
            z = R.add(x, y)
            w = R.multiply(z, z)
            return w

    @I.ir_module
    class Expected:
        @R.function
        def tuples_and_const(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                t1 = (x, y, x)
                t2 = (y, y, x)
                c = R.const([1, 2, 3], dtype="int32")
                R.output(c)
            return R.const([1, 2, 3], dtype="int32")

        @R.function
        def shapes() -> R.Shape:
            with R.dataflow():
                s1 = R.shape((1, 2, 3))
                s2 = R.shape((4, 5, 6))
                s3 = R.shape((7, 8, 9))
                R.output(s3)
            return s3

        @R.function
        def prim_values():
            with R.dataflow():
                x = R.prim_value(1)
                y = R.prim_value(2)
                z = R.prim_value(3)
                R.output(z)
            return z

        @R.function
        def main(t: R.Tuple(R.Tensor, R.Tensor)) -> R.Tensor:
            with R.dataflow():
                x = t[0]
                y = t[1]
                z = R.add(x, y)
                w = R.multiply(z, z)
                R.output(w)
            return w

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_impure_inner_function():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            @R.function(pure=False)
            def inner_func(x: R.Tensor, y: R.Tensor) -> R.Tensor:
                z = R.add(x, y)
                w = R.multiply(x, z)
                v = R.add(y, w)
                _ = R.print("oops")
                a = R.multiply(v, v)
                b = R.add(a, a)
                c = R.multiply(a, b)
                return c

            z = R.add(x, y)
            w = R.multiply(z, z)
            v = R.divide(w, z)
            q = inner_func(w, v)
            a = R.multiply(q, q)
            b = R.add(a, a)
            c = R.multiply(b, a)
            return c

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():

                @R.function(pure=False)
                def inner_func(x: R.Tensor, y: R.Tensor) -> R.Tensor:
                    with R.dataflow():
                        z = R.add(x, y)
                        w = R.multiply(x, z)
                        v = R.add(y, w)
                        R.output(v)
                    _ = R.print("oops")
                    with R.dataflow():
                        a = R.multiply(v, v)
                        b = R.add(a, a)
                        c = R.multiply(a, b)
                        R.output(c)
                    return c

                z = R.add(x, y)
                w = R.multiply(z, z)
                v = R.divide(w, z)
                R.output(inner_func, v, w)
            q = inner_func(w, v)
            with R.dataflow():
                a = R.multiply(q, q)
                b = R.add(a, a)
                c = R.multiply(b, a)
                R.output(c)
            return c

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_pure_inner_function():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            @R.function
            def inner_func(x: R.Tensor, y: R.Tensor) -> R.Tensor:
                z = R.add(x, y)
                w = R.multiply(x, z)
                v = R.add(y, w)
                return v

            z = R.add(x, y)
            w = R.multiply(z, z)
            v = R.divide(w, z)
            q = inner_func(w, v)
            return q

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():

                @R.function
                def inner_func(x: R.Tensor, y: R.Tensor) -> R.Tensor:
                    with R.dataflow():
                        z = R.add(x, y)
                        w = R.multiply(x, z)
                        v = R.add(y, w)
                        R.output(v)
                    return v

                z = R.add(x, y)
                w = R.multiply(z, z)
                v = R.divide(w, z)
                q = inner_func(w, v)
                R.output(q)
            return q

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_impure_external_function():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def extra(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            q = R.matmul(z, x)
            w = R.nn.relu(q)
            _ = R.print("Whoa")
            return w

        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            w = R.multiply(z, z)
            q = Before.extra(z, w)
            return q

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def extra(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                q = R.matmul(z, x)
                w = R.nn.relu(q)
                R.output(w)
            _ = R.print("Whoa")
            return w

        @R.function(pure=False)
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, z)
                R.output(z, w)
            q = Expected.extra(z, w)
            return q

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_pure_external_function():
    @I.ir_module
    class Before:
        @R.function
        def extra(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            q = R.matmul(z, x)
            w = R.nn.relu(q)
            return w

        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            z = R.add(x, y)
            w = R.multiply(z, z)
            q = Before.extra(z, w)
            return q

    @I.ir_module
    class Expected:
        @R.function
        def extra(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                q = R.matmul(z, x)
                w = R.nn.relu(q)
                R.output(w)
            return w

        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, z)
                q = Expected.extra(z, w)
                R.output(q)
            return q

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_merge_with_preceding_dataflow_block():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                R.output(w)

            # The single binding of `v = R.add` would normally not be
            # enough to make a dataflow block, as `1 < min_size == 2`.
            v = R.add(w, x)
            return v

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                # However, it occurs just after an existing dataflow
                # block, and can be merged into it.
                v = R.add(w, x)
                R.output(v)
            return v

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_merge_with_next_dataflow_block():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            # The single binding of `z = R.add` would normally not be
            # enough to make a dataflow block, as `1 < min_size == 2`.
            z = R.add(x, y)

            # However, it occurs just before an existing dataflow
            # block, and can be merged into it.

            with R.dataflow():
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)
            return v

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor) -> R.Tensor:
            with R.dataflow():
                z = R.add(x, y)
                w = R.multiply(z, y)
                v = R.add(w, x)
                R.output(v)
            return v

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_preserve_existing_dataflow_blocks_at_beginning():
    """Preserve existing DataflowBlocks

    This is a regression test.  In previous implementations, a
    DataflowBlock in the input, without enough bindings to become a
    new dataflow block, could be accidentally ommitted.

    This test is identical to
    `TestPreserveExistingDataflowBlocksAtEnd`, except that the
    existing dataflow block is at the beginning of the function.

    """

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(A0: R.Tensor, B0: R.Tensor):
            # This DataflowBlock is below the minimum size for a new
            # block, but already exists in the input IRModule.
            with R.dataflow():
                A1 = R.add(A0, A0)
                R.output(A1)

            R.print("impure_function")

            # This sequence is large enough that it may be converted
            # to a DataflowBlock.
            B1 = R.add(B0, B0)
            B2 = R.add(B1, B1)
            B3 = R.add(B2, B2)

            return (A1, B3)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(A0: R.Tensor, B0: R.Tensor):
            # This dataflow block should be preserved in the output.
            with R.dataflow():
                A1 = R.add(A0, A0)
                R.output(A1)

            R.print("impure_function")

            with R.dataflow():
                B1 = R.add(B0, B0)
                B2 = R.add(B1, B1)
                B3 = R.add(B2, B2)
                R.output(B3)

            return (A1, B3)

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_preserve_existing_dataflow_blocks_at_end():
    """Preserve existing DataflowBlocks

    This is a regression test.  In previous implementations, a
    DataflowBlock in the input, without enough bindings to become a
    new dataflow block, could be accidentally ommitted.

    This test is identical to
    `TestPreserveExistingDataflowBlocksAtBeginning`, except that the
    existing dataflow block is at the end of the function.

    """

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(A0: R.Tensor, B0: R.Tensor):
            # This sequence is large enough that it may be converted
            # to a DataflowBlock.
            B1 = R.add(B0, B0)
            B2 = R.add(B1, B1)
            B3 = R.add(B2, B2)

            R.print("impure_function")

            # This DataflowBlock is below the minimum size for a new
            # block, but already exists in the input IRModule.
            with R.dataflow():
                A1 = R.add(A0, A0)
                R.output(A1)

            return (A1, B3)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(A0: R.Tensor, B0: R.Tensor):
            with R.dataflow():
                B1 = R.add(B0, B0)
                B2 = R.add(B1, B1)
                B3 = R.add(B2, B2)
                R.output(B3)

            R.print("impure_function")

            # This dataflow block should be preserved in the output.
            with R.dataflow():
                A1 = R.add(A0, A0)
                R.output(A1)

            return (A1, B3)

    After = relax.transform.ConvertToDataflow()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
