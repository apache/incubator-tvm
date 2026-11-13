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
"""Tests for pass-time error enrichment with TVMScript-rendered locations.

A pass body that throws an error carrying a VisitErrorContext (e.g. a relax op
validator) is caught by the leaf pass executor and re-thrown with the failing
pass name plus the offending location rendered as underlined TVMScript.
"""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm_tirx.ir import IRModule


def _bad_matmul_module():
    """Build (programmatically, no TVMScript parse) a module whose `main` binds a
    matmul of incompatible shapes [3, 4] x [5, 6]. The function carries a
    placeholder return type so it constructs; Normalize re-infers and the
    matmul validator fires during the pass."""
    x = relax.Var("x", relax.TensorType([3, 4], "float32"))
    y = relax.Var("y", relax.TensorType([5, 6], "float32"))
    lv = relax.Var("lv")
    body = relax.SeqExpr([relax.BindingBlock([relax.VarBinding(lv, relax.op.matmul(x, y))])], lv)
    func = relax.Function([x, y], body, ret_ty=relax.TensorType([3, 6], "float32"), is_pure=True)
    func = func.with_attr("global_symbol", "main")
    return IRModule({relax.GlobalVar("main"): func})


@pytest.mark.skip_well_formed_check_before_transform
@pytest.mark.skip_well_formed_check_after_transform
def test_pass_error_renders_underlined_tvmscript():
    """End-to-end: a bad matmul through a function pass yields a message naming the
    pass and an underlined TVMScript snippet of the offending binding."""
    mod = _bad_matmul_module()
    with pytest.raises(ValueError) as excinfo:
        relax.transform.Normalize()(mod)
    assert str(excinfo.value) == (
        "Matmul requires the reduction length of the operands to be equal.  However, the LHS "
        "x has shape R.shape([3, 4]), while the RHS y has shape R.shape([5, 6]).  The reduction "
        "dimensions of T.int64(4) and T.int64(5) are not equal.\n\n"
        "Error in pass: Normalize\n"
        "Location (TVMScript):\n"
        "Access path: <root>.body.blocks[0].bindings[0].value\n\n"
        "from __future__ import annotations\n\n"
        "# from tvm.relax import script as R\n\n"
        "@R.function\n"
        'def main(x: R.Tensor((3, 4), dtype="float32"), '
        'y: R.Tensor((5, 6), dtype="float32")) -> R.Tensor((3, 6), dtype="float32"):\n'
        "    lv = R.matmul(x, y)\n"
        "         ^^^^^^^^^^^^^^\n"
        "    return lv"
    )


@pytest.mark.skip_well_formed_check_before_transform
@pytest.mark.skip_well_formed_check_after_transform
def test_sequential_does_not_double_append():
    """Running the failing pass inside a Sequential must not enrich twice — the
    Sequential wrapper does not guard, only the leaf pass does."""
    mod = _bad_matmul_module()
    seq = tvm_tirx.ir.transform.Sequential([relax.transform.Normalize()])
    with pytest.raises(ValueError) as excinfo:
        seq(mod)
    msg = str(excinfo.value)
    assert msg.count("Location (TVMScript):") == 1
    assert "Error in pass: Normalize" in msg


@pytest.mark.skip_well_formed_check_before_transform
@pytest.mark.skip_well_formed_check_after_transform
def test_error_without_resolvable_node_is_not_masked():
    """A pass that throws an error whose node is not findable in the module must
    surface the original message without raising a printer/render error."""

    @tvm_tirx.ir.transform.module_pass(opt_level=0, name="ThrowUnresolvable")
    class ThrowUnresolvable:
        def transform_module(self, mod, ctx):
            # A bare error with no VisitErrorContext payload -> nothing to resolve.
            raise RuntimeError("deliberate failure with no resolvable location")

    mod = _bad_matmul_module()
    with pytest.raises(RuntimeError) as excinfo:
        ThrowUnresolvable()(mod)
    msg = str(excinfo.value)
    assert "deliberate failure with no resolvable location" in msg
    # No context => no location block appended, but also no crash.
    assert "Location (TVMScript):" not in msg


@pytest.mark.skip_well_formed_check_before_transform
@pytest.mark.skip_well_formed_check_after_transform
def test_well_formed_check_propagates_through_pass():
    x = relax.Var("x", relax.TensorType([4], "float32"))
    undefined = relax.Var("undefined", x.ty)
    body = relax.SeqExpr([], undefined)
    func = relax.Function([x], body, ret_ty=x.ty).with_attr("global_symbol", "main")
    mod = IRModule({relax.GlobalVar("main"): func})

    @tvm_tirx.ir.transform.module_pass(opt_level=0, name="CheckWellFormed")
    def check(mod, ctx):
        relax.analysis.check_well_formed(mod)
        return mod

    with pytest.raises(ValueError, match="undefined"):
        relax.analysis.check_well_formed(mod)
    with pytest.raises(ValueError, match="undefined") as error:
        check(mod)
    assert "Error in pass: CheckWellFormed" in str(error.value)


if __name__ == "__main__":
    tvm_tirx.testing.main()
