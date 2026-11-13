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

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm_tirx.script import ir as I


def test_depth_first_from_inputs():
    """Sort DataflowBlock bindings with DFS, starting from inputs

    Starting with the inputs to the DataflowBlock, sort the variable
    bindings according to their occurrence in a depth-first search.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                B2 = R.add(A, R.const(2))
                C1 = R.add(A, B1)
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-inputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_depth_first_from_input_with_constant():
    """Topological sort must produce legal ordering.

    Here, both `C1` and `C2` use the input tensor `A`.  However, they
    also use the tensors `B1` and `B2`.  The bindings for `C1` and
    `C2` may not be emitted until after all their inputs have been
    emitted.

    In addition, the bindings `B1` and `B2` do not require any of the
    function inputs to compute.  If the DFS only used the function
    parameters as the initial search nodes, it would fail to output
    these variable bindings.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.const(1)
                B2 = R.const(2)
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D2 = R.add(A, C2)
                D1 = R.add(A, C1)
                E = R.add(D1, D2)
                R.output(E)
            return E

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.const(1)
                C1 = R.add(A, B1)
                D1 = R.add(A, C1)
                B2 = R.const(2)
                C2 = R.add(A, B2)
                D2 = R.add(A, C2)
                E = R.add(D1, D2)
                R.output(E)
            return E

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-inputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_depth_first_from_input_with_multiple_inputs():
    """Use parameter order for deterministic sort

    Here, both `C1` and `C2` use the input tensor `A`, as well as
    input tensors `B1` and `B2`, respectively.  Since `B1` appears
    before `B2`, `C1` should be sorted before `C2`.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, B1: R.Tensor, B2: R.Tensor):
            with R.dataflow():
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D2 = R.add(A, C2)
                D1 = R.add(A, C1)
                E = R.add(D1, D2)
                R.output(E)
            return E

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, B1: R.Tensor, B2: R.Tensor):
            with R.dataflow():
                C1 = R.add(A, B1)
                D1 = R.add(A, C1)
                C2 = R.add(A, B2)
                D2 = R.add(A, C2)
                E = R.add(D1, D2)
                R.output(E)
            return E

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-inputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_depth_first_break_ties_by_existing_order():
    """If DFS is ambiguous, provide deterministic output

    Here, both `B1` and `B2` use the input tensor `A`.  Since there
    are no other inputs for `B1` or `B2`, they remain in the same
    relative order as the input function, and `B1` is emitted before
    `B2`.  The DFS then continues, placing `C1` immediately after
    `B1`.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-inputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_depth_first_from_output():
    """Sort DataflowBlock bindings with DFS, starting from outputs

    Starting with the outputs to the DataflowBlock, sort the variable
    bindings according to their occurrence in a depth-first search.

    Like `TestDepthFirstFromInputs`, but perform the search starting
    at the output.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-outputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_depth_first_from_output_tuple_with_binding():
    """A dataflow block may produce multiple outputs

    If a dataflow block produces multiple outputs, the result should
    be sorted according to the order in which the outputs are used.
    Here, `C1` is used before `C2`, so the expressions required to
    compute `C1` are moved before the expressions required to compute
    `C2`.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                R.output(C1, C2)
            gv = (C1, C2)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                R.output(C1, C2)
            gv = (C1, C2)
            return gv

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-outputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_depth_first_from_output_tuple_without_binding():
    """A dataflow block may produce multiple outputs

    Like `TestDepthFirstFromOutputTupleWithBinding`, but the
    DataflowBlock's outputs are not used as part of a variable
    binding.  Because in-line tuples are not normalized to variable
    bindings, this case must be handled explicitly.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                R.output(C1, C2)
            return (C1, C2)

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                R.output(C1, C2)
            return (C1, C2)

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-outputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_depth_first_from_output_with_unused_variables():
    """Sort DataflowBlock bindings with DFS, starting from outputs

    The variables `D1` and `D2` are unused, but must still appear
    within the output DataflowBlock.

    This is analogous to `TestDepthFirstFromInputWithConstant`.
    Similar to how a DFS starting from the function inputs can
    accidentally skip expressions with no inputs, a DFS starting from
    the function outputs can accidentally skip expressions that do not
    contribute to the output.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D1 = R.add(A, C1)
                D2 = R.add(A, C2)
                E = R.add(C1, C2)
                R.output(E)
            return E

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                D1 = R.add(A, C1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D2 = R.add(A, C2)
                E = R.add(C1, C2)
                R.output(E)
            return E

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-outputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_depth_first_from_input_with_unused_parameters():
    """Sort DataflowBlock bindings with DFS, starting from inputs

    Functions may accept parameters that are not used.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, Unused: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                B2 = R.add(A, R.const(2))
                C1 = R.add(A, B1)
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, Unused: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    After = tvm.relax.transform.TopologicalSort(
        order="depth-first",
        direction="from-inputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_breadth_first():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(2))
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(1))
                B2 = R.add(A, R.const(2))
                C1 = R.add(A, B1)
                C2 = R.add(A, B2)
                D = R.add(C1, C2)
                R.output(D)
            return D

    After = tvm.relax.transform.TopologicalSort(
        order="breadth-first",
        direction="from-inputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_breadth_first_break_ties_by_existing_order():
    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B1 = R.add(A, R.const(2))
                C1 = R.add(A, B1)
                B2 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                D = R.add(C2, C1)
                R.output(D)
            return D

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            with R.dataflow():
                B2 = R.add(A, R.const(2))
                B1 = R.add(A, R.const(1))
                C2 = R.add(A, B2)
                C1 = R.add(A, B1)
                D = R.add(C1, C2)
                R.output(D)
            return D

    After = tvm.relax.transform.TopologicalSort(
        order="breadth-first",
        direction="from-inputs",
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
