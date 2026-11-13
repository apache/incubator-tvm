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

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.relax.analysis import name_to_binding
from tvm.relax.binding_rewrite import DataflowBlockRewrite
from tvm.relax.expr import DataflowVar, Var


@tvm_tirx.script.ir_module
class Identity:
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = x
            R.output(lv0)
        return lv0


def assert_immutability(rwt, original_dfb, original_root_fn):
    assert rwt.mutated_dfb() != original_dfb
    assert rwt.mutated_root_fn() != original_root_fn
    assert rwt.mutated_root_fn().body.blocks[0] != original_dfb
    assert rwt.mutated_root_fn().body.blocks[0] == rwt.mutated_dfb()


def test_null_construct():
    root_fn = Identity["main"]
    dfb = root_fn.body.blocks[0]

    DataflowBlockRewrite(dfb, root_fn)


def test_simple_add():
    root_fn = Identity["main"]
    dfb = root_fn.body.blocks[0]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.add(name="tmp", expr=Identity["main"].params[0], is_dfvar=True)

    assert_immutability(rwt, dfb, root_fn)

    # check "tmp" added
    assert "tmp" in name_to_binding(rwt.mutated_root_fn())

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                tmp: R.Tensor((32, 32), "float32") = x
                R.output(lv0)
            return lv0

    tvm_tirx.ir.assert_structural_equal(rwt.mutated_root_fn(), GroundTruth["main"])


def test_simple_auto_add_var():
    root_fn = Identity["main"]
    dfb = root_fn.body.blocks[0]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.add(root_fn.params[0], is_dfvar=False)

    assert isinstance(rwt.mutated_dfb().bindings[-1].var, Var)

    assert_immutability(rwt, dfb, root_fn)


def test_simple_auto_add_dfvar():
    root_fn = Identity["main"]
    dfb = root_fn.body.blocks[0]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.add(root_fn.params[0], is_dfvar=True)

    assert isinstance(rwt.mutated_dfb().bindings[-1].var, DataflowVar)

    # immutatbility
    assert_immutability(rwt, dfb, root_fn)


def test_simple_remove_unused():
    @tvm_tirx.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused = lv0
                R.output(lv0)
            return lv0

    root_fn = IdentityUnused["main"]
    dfb = root_fn.body.blocks[0]

    n2binding = name_to_binding(IdentityUnused["main"])

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_unused(n2binding["unused"][0].var)

    assert_immutability(rwt, dfb, root_fn)

    # check "unused" removed
    assert "unused" not in name_to_binding(rwt.mutated_root_fn())

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    tvm_tirx.ir.assert_structural_equal(rwt.mutated_root_fn(), GroundTruth["main"])


def test_remove_unused_undef():
    root_fn = Identity["main"]
    dfb = root_fn.body.blocks[0]

    with pytest.raises(RuntimeError):
        rwt = DataflowBlockRewrite(dfb, root_fn)
        rwt.remove_unused(Var("whatever"))

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_unused(Var("whatever"), allow_undef=True)

    assert root_fn == rwt.mutated_root_fn()


def test_simple_rm_all_unused():
    @tvm_tirx.script.ir_module
    class IdentityUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = lv0
                unused1 = lv0
                R.output(lv0)
            return lv0

    root_fn = IdentityUnused["main"]
    dfb = root_fn.body.blocks[0]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_all_unused()

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    tvm_tirx.ir.assert_structural_equal(rwt.mutated_root_fn(), GroundTruth["main"])


@tvm_tirx.script.ir_module
class DeadDFBlock:
    @R.function
    def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
        with R.dataflow():
            lv0 = x
            R.output(lv0)
        return x


def test_empty_dfb_after_removal():
    root_fn = DeadDFBlock["main"]
    dfb = root_fn.body.blocks[0]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_unused(DeadDFBlock["main"].body.blocks[0].bindings[0].var)

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
            return x

    tvm_tirx.ir.assert_structural_equal(rwt.mutated_root_fn(), GroundTruth["main"])


def test_empty_dfb_after_all_removal():
    dfb = DeadDFBlock["main"].body.blocks[0]
    root_fn = DeadDFBlock["main"]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_all_unused()

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
            return x

    tvm_tirx.ir.assert_structural_equal(rwt.mutated_root_fn(), GroundTruth["main"])


def test_chained_rm_all_unused():
    @tvm_tirx.script.ir_module
    class IdentityChainedUnused:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                unused0 = R.call_dps_packed(
                    _ExternFunc("my_sigmoid"), (x,), ty_args=[R.Tensor((32, 32), dtype="float32")]
                )
                unused1 = R.call_dps_packed(
                    _ExternFunc("my_sigmoid"),
                    (unused0,),
                    ty_args=[R.Tensor((32, 32), dtype="float32")],
                )
                R.output(lv0)
            return lv0

    root_fn = IdentityChainedUnused["main"]
    dfb = root_fn.body.blocks[0]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.remove_all_unused()

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    tvm_tirx.ir.assert_structural_equal(rwt.mutated_root_fn(), GroundTruth["main"])


def test_simple_replace_all_uses():
    @tvm_tirx.script.ir_module
    class Lv0To1:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
            #   lv0 => lv1
            #  /   \
            # lv2  lv3
            #  \   /
            #   lv4
            with R.dataflow():
                lv0: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                    _ExternFunc("my_relu"), (x,), ty_args=[R.Tensor((32, 32), dtype="float32")]
                )
                lv1: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                    _ExternFunc("my_sigmoid"), (x,), ty_args=[R.Tensor((32, 32), dtype="float32")]
                )
                lv2: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                    _ExternFunc("my_add"), (x, lv0), ty_args=[R.Tensor((32, 32), dtype="float32")]
                )
                lv3: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                    _ExternFunc("my_mul"), (x, lv0), ty_args=[R.Tensor((32, 32), dtype="float32")]
                )
                lv4: R.Tensor((32, 32), "float32") = R.call_dps_packed(
                    _ExternFunc("my_whatever"),
                    (lv2, lv3),
                    ty_args=[R.Tensor((32, 32), dtype="float32")],
                )
                R.output(lv4)
            return lv4

    root_fn = Lv0To1["main"]
    dfb = root_fn.body.blocks[0]

    n2binding = name_to_binding(root_fn)

    rwt = DataflowBlockRewrite(dfb, root_fn)
    missing = Var("missing")
    with pytest.raises(RuntimeError, match="Cannot find"):
        rwt.replace_all_uses(missing, missing)

    rwt.replace_all_uses(n2binding["lv0"][0].var, n2binding["lv1"][0].var)
    rwt.replace_all_uses(n2binding["lv1"][0].var, n2binding["lv1"][0].var)
    with pytest.raises(RuntimeError, match=r"is used by 2 vars"):
        rwt.remove_unused(n2binding["lv1"][0].var)
    rwt.remove_unused(n2binding["lv0"][0].var)

    assert_immutability(rwt, dfb, root_fn)

    n2binding_after = name_to_binding(rwt.mutated_root_fn())
    assert "lv0" not in n2binding_after


def test_simple_module_update():
    @tvm_tirx.script.ir_module
    class Identity:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                R.output(lv0)
            return lv0

    root_fn = Identity["main"]
    dfb = root_fn.body.blocks[0]

    rwt = DataflowBlockRewrite(dfb, root_fn)
    rwt.add(name="tmp", expr=root_fn.params[0], is_dfvar=True)

    new_ir = rwt.mutate_irmodule(Identity)

    # immutatbility
    assert new_ir != Identity
    assert 2 == len(new_ir["main"].body.blocks[0].bindings)

    @tvm_tirx.script.ir_module
    class GroundTruth:
        @R.function
        def main(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
            with R.dataflow():
                lv0 = x
                tmp: R.Tensor((32, 32), "float32") = x
                R.output(lv0)
            return lv0

    tvm_tirx.ir.assert_structural_equal(new_ir, GroundTruth)


if __name__ == "__main__":
    tvm_tirx.testing.main()
