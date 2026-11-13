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
"""Tests to validate relax fast math tranform pass."""

import pytest

import tvm
import tvm_tirx.testing
from tvm import relax, topi
from tvm.relax import script as R
from tvm.relax.transform import FastMathTransform
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I


def _run_pass_compare_output(Before, Expected):
    fast_mod = FastMathTransform()(Before)
    relax.analysis.check_well_formed(fast_mod)
    assert_structural_equal(Expected, fast_mod)


def test_optimize_transform_layout_pass_one_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            lv1: R.Tensor((16,), dtype="float32") = R.nn.softmax(x)
            lv2: R.Tensor((16,), dtype="float32") = R.exp(lv1)
            lv3: R.Tensor((16,), dtype="float32") = R.erf(lv2)
            lv4: R.Tensor((16,), dtype="float32") = R.tanh(lv3)
            return lv4

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((16,), "float32"))
    with bb.function("main", [x]):
        lv1 = bb.emit_te(topi.nn.fast_softmax, x)
        lv2 = bb.emit_te(topi.fast_exp, lv1)
        lv3 = bb.emit_te(topi.fast_erf, lv2)
        lv4 = bb.emit_te(topi.fast_tanh, lv3)
        bb.emit_func_output(lv4)
    Expected = bb.get()

    _run_pass_compare_output(Before, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
