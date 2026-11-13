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

"""
Test relax transform - Eliminate redundant reshape operations
"""

import tvm
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.transform import DeadCodeElimination, RemoveRedundantReshape
from tvm_tirx.script import ir as I


def _run_pass_compare_output(Before, Expected):
    fused_mod = RemoveRedundantReshape()(Before)
    fused_mod = DeadCodeElimination()(fused_mod)
    tvm_tirx.ir.assert_structural_equal(Expected, fused_mod)


def test_remove_redundant_reshape_pass_one_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((1, 1001, 1, 1), dtype="float16")) -> R.Tensor(
            (1, 1001), dtype="float16"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 1001), dtype="float16") = R.reshape(x, R.shape([1, 1001]))
                lv1: R.Tensor((1, 1001), dtype="float16") = R.reshape(lv, R.shape([1, 1001]))
                gv: R.Tensor((1, 1001), dtype="float16") = R.reshape(lv1, R.shape([1, 1001]))
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 1001, 1, 1), dtype="float16")) -> R.Tensor(
            (1, 1001), dtype="float16"
        ):
            with R.dataflow():
                gv: R.Tensor((1, 1001), dtype="float16") = R.reshape(x, R.shape([1, 1001]))
                R.output(gv)
            return gv

    _run_pass_compare_output(Before, Expected)


def test_remove_redundant_reshape_pass_two_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((1, 1001, 1, 1), dtype="float16")) -> R.Tensor(
            (1, 1001), dtype="float16"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 1001, 1), dtype="float16") = R.reshape(x, R.shape([1, 1001, 1]))
                lv1: R.Tensor((1, 1001), dtype="float16") = R.reshape(lv, R.shape([1, 1001]))
                R.output(lv1)
            return lv1

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 1001, 1, 1), dtype="float16")) -> R.Tensor(
            (1, 1001), dtype="float16"
        ):
            with R.dataflow():
                lv1: R.Tensor((1, 1001), dtype="float16") = R.reshape(x, R.shape([1, 1001]))
                R.output(lv1)
            return lv1

    _run_pass_compare_output(Before, Expected)


def test_remove_redundant_reshape_pass_three_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((1, 1001, 1, 1), dtype="float16")) -> R.Tensor(
            (1, 1001, 1, 1), dtype="float16"
        ):
            with R.dataflow():
                lv: R.Tensor((1, 1001, 1, 1), dtype="float16") = R.reshape(
                    x, R.shape([1, 1001, 1, 1])
                )
                R.output(lv)
            return lv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 1001, 1, 1), dtype="float16")) -> R.Tensor(
            (1, 1001, 1, 1), dtype="float16"
        ):
            return x

    _run_pass_compare_output(Before, Expected)


def _return_shape(mod):
    ret = mod["main"].ret_ty
    field = ret.fields[0] if hasattr(ret, "fields") else ret
    return [int(dim) for dim in field.shape]


def test_remove_redundant_reshape_pass_keeps_zero_sized_chain():
    # A literal 0 in a reshape target means "copy the corresponding input dimension",
    # and relax.op.reshape resolves it against the input it is handed. Combining these
    # two calls re-reads the zeros against x and asks for (0, 3, 5) instead of the
    # (0, 0, 0) the pair produces, so the pair has to survive the pass.
    #
    # Built with the block builder rather than TVMScript because reshape resolves its
    # target at construction: the printed R.shape([0, 0, 5]) is the resolved shape, and
    # parsing it back would resolve those zeros a second time.
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([0, 3, 5], "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv = bb.emit(relax.op.reshape(x, [0, -1, 5]))
            gv = bb.emit_output(relax.op.reshape(lv, [0, 0, -1]))
        bb.emit_func_output(gv)
    before = bb.get()
    assert _return_shape(before) == [0, 0, 0]

    after = DeadCodeElimination()(RemoveRedundantReshape()(before))
    assert _return_shape(after) == [0, 0, 0], (
        "combining the reshapes re-read the literal zeros against x and changed the shape"
    )


def test_remove_redundant_reshape_pass_still_combines_without_zero_dims():
    # The guard above must not stop the pass doing its job on an ordinary chain.
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([1, 1001, 1, 1], "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv = bb.emit(relax.op.reshape(x, [1, 1001, 1]))
            gv = bb.emit_output(relax.op.reshape(lv, [1, 1001]))
        bb.emit_func_output(gv)
    after = DeadCodeElimination()(RemoveRedundantReshape()(bb.get()))
    assert _return_shape(after) == [1, 1001]
    reshapes = [
        binding
        for block in after["main"].body.blocks
        for binding in block.bindings
        if isinstance(binding.value, tvm.relax.Call)
    ]
    assert len(reshapes) == 1, "the chain without zeros should still collapse to one reshape"


if __name__ == "__main__":
    tvm_tirx.testing.main()
