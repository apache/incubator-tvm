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
# ruff: noqa: E501, F841
import pytest

import tvm_tirx.testing
from tvm.relax import script as R
from tvm.relax.training import AppendLoss
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I


def test_simple():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = x + y
                R.output(gv0)
            return gv0

    @R.function
    def loss(arg1: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            gv0 = R.sum(arg1)
            R.output(gv0)
        return gv0

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")) -> R.Tensor((3, 3), "float32"):
            with R.dataflow():
                gv0: R.Tensor((3, 3), "float32") = R.add(x, y)
                R.output(gv0)
            return gv0

        @R.function
        def main_loss(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                gv0: R.Tensor((3, 3), "float32") = R.add(x, y)
                gv0_1: R.Tensor((), "float32") = R.sum(gv0, axis=None, keepdims=False)
                R.output(gv0_1)
            return gv0_1
    # fmt: on

    After = AppendLoss("main", loss)(Before)
    assert_structural_equal(After, Expected)


def test_num_backbone_outputs():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = R.sum(x)
                gv1 = R.sum(y)
                R.output(gv0, gv1)
            return gv0, gv1

    @R.function
    def loss(arg1: R.Tensor((), "float32"), arg2: R.Tensor((), "float32")):
        with R.dataflow():
            gv0 = R.add(arg1, arg2)
            R.output(gv0)
        return gv0

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tensor((), "float32")):
            with R.dataflow():
                gv0: R.Tensor((), "float32") = R.sum(x, axis=None, keepdims=False)
                gv1: R.Tensor((), "float32") = R.sum(y, axis=None, keepdims=False)
                R.output(gv0, gv1)
            return (gv0, gv1)

        @R.function
        def main_loss(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")) -> R.Tensor((), "float32"):
            with R.dataflow():
                gv0: R.Tensor((), "float32") = R.sum(x, axis=None, keepdims=False)
                gv1: R.Tensor((), "float32") = R.sum(y, axis=None, keepdims=False)
                gv0_1: R.Tensor((), "float32") = R.add(gv0, gv1)
                R.output(gv0_1)
            return gv0_1
    # fmt: on

    After = AppendLoss("main", loss, 2)(Before)
    assert_structural_equal(After, Expected)


def test_extra_params():
    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = R.sum(x)
                gv1 = R.add(x, x)
                gv2 = x
                R.output(gv0, gv1, gv2)
            return gv0, gv1, gv2

    @R.function
    def loss(
        arg1: R.Tensor((), "float32"),
        arg2: R.Tensor((3, 3), "float32"),
        arg3: R.Tensor((3, 3), "float32"),
    ):
        with R.dataflow():
            gv0 = R.add(arg2, arg3)
            gv1 = R.sum(gv0)
            R.output(gv1)
        return gv1

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tensor((3, 3), "float32"), R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0: R.Tensor((), "float32") = R.sum(x, axis=None, keepdims=False)
                gv1: R.Tensor((3, 3), "float32") = R.add(x, x)
                gv2: R.Tensor((3, 3), "float32") = x
                R.output(gv0, gv1, gv2)
            return (gv0, gv1, gv2)

        @R.function
        def main_loss(x: R.Tensor((3, 3), "float32"), arg3: R.Tensor((3, 3), "float32")) -> R.Tuple(R.Tensor((), "float32"), R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0: R.Tensor((), "float32") = R.sum(x, axis=None, keepdims=False)
                gv1: R.Tensor((3, 3), "float32") = R.add(x, x)
                gv2: R.Tensor((3, 3), "float32") = x
                gv0_1: R.Tensor((3, 3), "float32") = R.add(gv1, arg3)
                gv1_1: R.Tensor((), "float32") = R.sum(gv0_1, axis=None, keepdims=False)
                R.output(gv2, gv1_1)
            return (gv1_1, gv2)
    # fmt: on

    After = AppendLoss("main", loss, 2)(Before)
    assert_structural_equal(After, Expected)


def test_error_return_value_vs_parameter():
    # Type not match
    # fmt: off
    @I.ir_module
    class Module1:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = R.sum(x)
                gv1 = R.sum(y)
                R.output(gv0, gv1)
            return gv0, gv1

    @R.function
    def loss1(arg1: R.Tensor((), "float64"), arg2: R.Tensor((), "float64")):
        with R.dataflow():
            gv0 = R.add(arg1, arg2)
            R.output(gv0)
        return gv0
    # fmt: on

    with pytest.raises(RuntimeError):
        AppendLoss("main", loss1, 2)(Module1)

    # The numbers of backbone return value and loss parameter are not enough
    # fmt: off
    @I.ir_module
    class Module2:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = x + y
                R.output(gv0)
            return gv0

    @R.function
    def loss2(arg1: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            gv0 = R.sum(arg1)
            R.output(gv0)
        return gv0
    # fmt: on

    with pytest.raises(RuntimeError):
        AppendLoss("main", loss2, 2)(Module2)

    # Backbone returns nested tuple
    # fmt: off
    @I.ir_module
    class Module3:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = x
                gv1 = y
                gv2 = x + y
                R.output(gv0, gv1, gv2)
            return gv0, (gv1, gv2)

    @R.function
    def loss3(arg1: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            gv0 = R.sum(arg1)
            R.output(gv0)
        return gv0
    # fmt: on

    with pytest.raises(RuntimeError):
        AppendLoss("main", loss3, 1)(Module3)


def test_error_more_blocks():
    # backbone more than one blocks
    # fmt: off
    @I.ir_module
    class Module1:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = x
                R.output(gv0)
            gv1 = gv0
            return gv1

    @R.function
    def loss1(arg: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            gv = R.sum(arg)
            R.output(gv)
        return gv
    # fmt: on

    with pytest.raises(RuntimeError):
        AppendLoss("main", loss1)(Module1)

    # loss more than one blocks
    # fmt: off
    @I.ir_module
    class Module2:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = x
                R.output(gv0)
            return gv0

    @R.function
    def loss2(arg: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            gv = R.sum(arg)
            R.output(gv)
        gv1 = gv
        return gv1
    # fmt: on

    with pytest.raises(RuntimeError):
        AppendLoss("main", loss2)(Module2)


def test_loss_return_value():
    # loss returns non-scalar var
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = x
                R.output(gv0)
            return gv0

    @R.function
    def loss(arg1: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            gv0 = arg1
            R.output(gv0)
        return gv0
    # fmt: on

    with pytest.raises(RuntimeError):
        AppendLoss("main", loss)(Module)

    # loss returns tuple
    # fmt: off
    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                gv0 = x
                R.output(gv0)
            return gv0

    @R.function
    def loss(arg1: R.Tensor((3, 3), "float32")):
        with R.dataflow():
            gv0 = R.sum(arg1)
            gv1 = gv0 + gv0
            R.output(gv0, gv1)
        return gv0, gv1
    # fmt: on

    with pytest.raises(RuntimeError):
        AppendLoss("main", loss)(Module)


if __name__ == "__main__":
    tvm_tirx.testing.main()
