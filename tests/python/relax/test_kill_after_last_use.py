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

import pytest

import tvm
import tvm.relax
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.relax.transform import KillAfterLastUse
from tvm_tirx.script import ir as I
from tvm_tirx.testing import env


def test_basic():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor([16, 32], "float32")):
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            y = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            _dummy = R.call_packed("add_tensors", [x, y], ty_args=(R.Tuple,))
            z = R.add(x, y)
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor([16, 32], "float32")):
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            y = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            _ = R.memory.kill_storage(storage)
            _dummy = R.call_packed("add_tensors", [x, y], ty_args=(R.Tuple,))
            z = R.add(x, y)
            _ = R.memory.kill_tensor(y)
            return z

    After = KillAfterLastUse()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_track_usage_across_trivial_rebindings():
    """To work around VM de-duplication of register usage"""

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = x
            z = R.add(y, R.const(1, "float32"))
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            z = R.add(x, R.const(1, "float32"))
            _ = R.memory.kill_tensor(x)
            return z

    After = KillAfterLastUse()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_track_usage_across_trivial_rebindings_in_match_cast():
    """To work around VM de-duplication of register usage"""

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = R.match_cast(x, R.Tensor([16, 32]))
            z = R.add(y, R.const(1, "float32"))
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(w: R.Tensor([16, 32], "float32")):
            x = R.add(w, R.const(1, "float32"))
            y = R.match_cast(x, R.Tensor([16, 32]))
            _ = R.memory.kill_tensor(x)
            z = R.add(y, R.const(1, "float32"))
            _ = R.memory.kill_tensor(y)
            return z

    After = KillAfterLastUse()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_no_kill_for_null_value():
    """R.null_value() must never be targeted by R.vm.kill_object

    A variable bound to `R.null_value()` is never assigned a real VM
    register by CodeGenVM, which special-cases `null_value` to a sentinel value.
    KillAfterLastUse must therefore never insert `R.vm.kill_object`
    for such a variable, even though its type is `R.Any` (the same
    type used by legitimate killable objectws such as VM storage).
    """

    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main(x: R.Tensor([16, 32], "float32")):
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            y = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            shape_heap: R.Any = R.null_value()
            _dummy = R.call_packed("use_shape_heap", [shape_heap], ty_args=(R.Tuple,))
            z = R.add(x, y)
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor([16, 32], "float32")):
            storage = R.memory.alloc_storage(R.shape([2048]), 0, "global", "uint8")
            y = R.memory.alloc_tensor(storage, 0, R.shape([16, 32]), "float32")
            _ = R.memory.kill_storage(storage)
            shape_heap: R.Any = R.null_value()
            _dummy = R.call_packed("use_shape_heap", [shape_heap], ty_args=(R.Tuple,))
            z = R.add(x, y)
            _ = R.memory.kill_tensor(y)
            return z

    After = KillAfterLastUse()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def _assert_no_kill_of_null_value(func: tvm.relax.Function):
    """Assert no R.vm.kill_object call in `func` targets a null_value()-bound var

    An `R.null_value()`-bound variable never occupies a VM register in
    CodeGenVM, so passing one to
    R.vm.kill_object is always invalid. Checking this structurally
    (rather than only checking that relax.build succeeds) ensures the
    test fails if a future change merely makes codegen tolerant of the
    invalid kill, instead of preventing KillAfterLastUse from
    inserting it in the first place.
    """
    null_value_op = tvm_tirx.ir.Op.get("relax.null_value")
    kill_object_op = tvm_tirx.ir.Op.get("relax.vm.kill_object")

    null_value_vars = []
    killed_args = []

    body = func.body
    assert isinstance(body, tvm.relax.SeqExpr)
    for block in body.blocks:
        for binding in block.bindings:
            value = binding.value
            if not isinstance(value, tvm.relax.Call):
                continue
            if value.op.same_as(null_value_op):
                null_value_vars.append(binding.var)
            elif value.op.same_as(kill_object_op):
                killed_args.append(value.args[0])

    for killed in killed_args:
        for null_value_var in null_value_vars:
            assert not killed.same_as(null_value_var), (
                f"R.vm.kill_object was called on a variable bound to R.null_value(): {killed}"
            )


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_reapply_after_default_pipeline_builds_successfully():
    """KillAfterLastUse may be re-applied to an already-lowered module

    Applying KillAfterLastUse a second time, on top of the output of
    `relax.get_default_pipeline` (which itself ends with a
    KillAfterLastUse application, after VMShapeLower has introduced a
    `shape_heap: R.Any = R.null_value()` binding), must not insert an
    invalid `R.vm.kill_object(shape_heap)`, and the resulting module
    must still build successfully using bytecode.
    """

    @I.ir_module
    class Mod:
        @R.function
        def main(x: R.Tensor((1, 4), "float32")) -> R.Tensor((1, 4), "float32"):
            R.func_attr({"global_symbol": "main", "num_input": 1})
            y: R.Tensor((1, 4), "float32") = R.add(x, R.const(1.0, "float32"))
            z: R.Tensor((1, 4), "float32") = R.add(y, R.const(2.0, "float32"))
            return z

    target = tvm_tirx.target.Target("llvm")
    preoptimized = tvm.relax.get_default_pipeline(target)(Mod)
    second_kill = KillAfterLastUse()(preoptimized)

    # Prove the invalid kill is gone, not merely that codegen tolerates it.
    _assert_no_kill_of_null_value(second_kill["main"])

    tvm.relax.build(second_kill, target=target, relax_pipeline="zero")


if __name__ == "__main__":
    tvm_tirx.testing.main()
