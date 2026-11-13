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
# ruff: noqa: E501
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.training import SetupTrainer
from tvm.relax.training.loss import MSELoss
from tvm.relax.training.optimizer import SGD, MomentumSGD
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I


def test_simple():
    # fmt: off
    @I.ir_module
    class Backbone:
        I.module_attrs({"param_num": 1, "state_num": 0})
        @R.function
        def backbone(x: R.Tensor((2, 2), "float64"), y: R.Tensor((2, 2), "float64")):
            with R.dataflow():
                x1 = x + y
                R.output(x1)
            return x1

    @I.ir_module
    class Expected:
        I.module_attrs({"input_num": 1, "param_num": 1, "state_num": 0})
        @R.function
        def backbone(x: R.Tensor((2, 2), dtype="float64"), y: R.Tensor((2, 2), dtype="float64")) -> R.Tensor((2, 2), dtype="float64"):
            with R.dataflow():
                x1: R.Tensor((2, 2), dtype="float64") = R.add(x, y)
                R.output(x1)
            return x1

        @R.function
        def backbone_loss(x: R.Tensor((2, 2), dtype="float64"), y: R.Tensor((2, 2), dtype="float64"), targets: R.Tensor((2, 2), dtype="float64")) -> R.Tensor((), dtype="float64"):
            with R.dataflow():
                x1: R.Tensor((2, 2), dtype="float64") = R.add(x, y)
                lv: R.Tensor((2, 2), dtype="float64") = R.subtract(x1, targets)
                lv1: R.Tensor((2, 2), dtype="float64") = R.multiply(lv, lv)
                gv: R.Tensor((), dtype="float64") = R.sum(lv1, axis=None, keepdims=False)
                R.output(gv)
            return gv

        @R.function
        def backbone_loss_adjoint(x: R.Tensor((2, 2), dtype="float64"), y: R.Tensor((2, 2), dtype="float64"), targets: R.Tensor((2, 2), dtype="float64")) -> R.Tuple(R.Tensor((), dtype="float64"), R.Tuple(R.Tensor((2, 2), dtype="float64"))):
            with R.dataflow():
                x1: R.Tensor((2, 2), dtype="float64") = R.add(x, y)
                lv: R.Tensor((2, 2), dtype="float64") = R.subtract(x1, targets)
                lv1: R.Tensor((2, 2), dtype="float64") = R.multiply(lv, lv)
                gv: R.Tensor((), dtype="float64") = R.sum(lv1, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float64") = R.ones(R.shape([]), dtype="float64")
                lv1_adjoint: R.Tensor((2, 2), dtype="float64") = R.broadcast_to(gv_adjoint, R.shape([2, 2]))
                lv_adjoint: R.Tensor((2, 2), dtype="float64") = R.multiply(lv1_adjoint, lv)
                lv_1: R.Tensor((2, 2), dtype="float64") = R.multiply(lv1_adjoint, lv)
                lv_adjoint1: R.Tensor((2, 2), dtype="float64") = R.add(lv_adjoint, lv_1)
                x1_adjoint: R.Tensor((2, 2), dtype="float64") = lv_adjoint1
                y_adjoint: R.Tensor((2, 2), dtype="float64") = x1_adjoint
                y_adjoint_out: R.Tensor((2, 2), dtype="float64") = y_adjoint
                R.output(gv, y_adjoint_out)
            return (gv, (y_adjoint_out,))

        @R.function
        def optimizer(params: R.Tuple(R.Tensor((2, 2), dtype="float64")), gradients: R.Tuple(R.Tensor((2, 2), dtype="float64")), optim_states: R.Tuple(R.Tensor((), dtype="int64"))) -> R.Tuple(R.Tuple(R.Tensor((2, 2), dtype="float64")), R.Tuple(R.Tensor((), dtype="int64"))):
            with R.dataflow():
                num_steps: R.Tensor((), dtype="int64") = optim_states[0]
                num_steps_new: R.Tensor((), dtype="int64") = R.add(num_steps, R.const(1, "int64"))
                y: R.Tensor((2, 2), dtype="float64") = params[0]
                y_grad: R.Tensor((2, 2), dtype="float64") = gradients[0]
                lv: R.Tensor((2, 2), dtype="float64") = R.multiply(R.const(0.10000000000000001, "float64"), y_grad)
                y_new: R.Tensor((2, 2), dtype="float64") = R.subtract(y, lv)
                params_new: R.Tuple(R.Tensor((2, 2), dtype="float64")) = (y_new,)
                optim_states_new: R.Tuple(R.Tensor((), dtype="int64")) = (num_steps_new,)
                R.output(params_new, optim_states_new)
            return (params_new, optim_states_new)
    # fmt: on

    ty = relax.TensorType((2, 2), "float64")
    setup_trainer = SetupTrainer(MSELoss(reduction="sum"), SGD(0.1), [ty, ty], legalize=False)
    train_mod = setup_trainer(Backbone)
    assert_structural_equal(train_mod.without_attr("optim_state"), Expected)


def test_states():
    # fmt: off
    @I.ir_module
    class Backbone:
        I.module_attrs({"param_num": 1, "state_num": 1})
        @R.function
        def backbone(x: R.Tensor((2, 2), "float64"), y: R.Tensor((2, 2), "float64"), z: R.Tensor((2, 2), "float64")):
            with R.dataflow():
                x1 = x + y
                z1 = z + R.const(1, "float64")
                R.output(x1, z1)
            return x1, z1

    @I.ir_module
    class Expected:
        I.module_attrs({"input_num": 1, "param_num": 1, "state_num": 1})
        @R.function
        def backbone(x: R.Tensor((2, 2), dtype="float64"), y: R.Tensor((2, 2), dtype="float64"), z: R.Tensor((2, 2), dtype="float64")) -> R.Tuple(R.Tensor((2, 2), dtype="float64"), R.Tensor((2, 2), dtype="float64")):
            with R.dataflow():
                x1: R.Tensor((2, 2), dtype="float64") = R.add(x, y)
                z1: R.Tensor((2, 2), dtype="float64") = R.add(z, R.const(1, "float64"))
                R.output(x1, z1)
            return (x1, z1)

        @R.function
        def backbone_loss(x: R.Tensor((2, 2), dtype="float64"), y: R.Tensor((2, 2), dtype="float64"), z: R.Tensor((2, 2), dtype="float64"), targets: R.Tensor((2, 2), dtype="float64")) -> R.Tuple(R.Tensor((), dtype="float64"), R.Tensor((2, 2), dtype="float64")):
            with R.dataflow():
                x1: R.Tensor((2, 2), dtype="float64") = R.add(x, y)
                z1: R.Tensor((2, 2), dtype="float64") = R.add(z, R.const(1, "float64"))
                lv: R.Tensor((2, 2), dtype="float64") = R.subtract(x1, targets)
                lv1: R.Tensor((2, 2), dtype="float64") = R.multiply(lv, lv)
                gv: R.Tensor((), dtype="float64") = R.sum(lv1, axis=None, keepdims=False)
                R.output(z1, gv)
            return (gv, z1)

        @R.function
        def backbone_loss_adjoint(x: R.Tensor((2, 2), dtype="float64"), y: R.Tensor((2, 2), dtype="float64"), z: R.Tensor((2, 2), dtype="float64"), targets: R.Tensor((2, 2), dtype="float64")) -> R.Tuple(R.Tuple(R.Tensor((), dtype="float64"), R.Tensor((2, 2), dtype="float64")), R.Tuple(R.Tensor((2, 2), dtype="float64"))):
            with R.dataflow():
                x1: R.Tensor((2, 2), dtype="float64") = R.add(x, y)
                z1: R.Tensor((2, 2), dtype="float64") = R.add(z, R.const(1, "float64"))
                lv: R.Tensor((2, 2), dtype="float64") = R.subtract(x1, targets)
                lv1: R.Tensor((2, 2), dtype="float64") = R.multiply(lv, lv)
                gv: R.Tensor((), dtype="float64") = R.sum(lv1, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float64") = R.ones(R.shape([]), dtype="float64")
                lv1_adjoint: R.Tensor((2, 2), dtype="float64") = R.broadcast_to(gv_adjoint, R.shape([2, 2]))
                lv_adjoint: R.Tensor((2, 2), dtype="float64") = R.multiply(lv1_adjoint, lv)
                lv_1: R.Tensor((2, 2), dtype="float64") = R.multiply(lv1_adjoint, lv)
                lv_adjoint1: R.Tensor((2, 2), dtype="float64") = R.add(lv_adjoint, lv_1)
                x1_adjoint: R.Tensor((2, 2), dtype="float64") = lv_adjoint1
                y_adjoint: R.Tensor((2, 2), dtype="float64") = x1_adjoint
                y_adjoint_out: R.Tensor((2, 2), dtype="float64") = y_adjoint
                R.output(z1, gv, y_adjoint_out)
            return ((gv, z1), (y_adjoint_out,))

        @R.function
        def optimizer(params: R.Tuple(R.Tensor((2, 2), dtype="float64")), gradients: R.Tuple(R.Tensor((2, 2), dtype="float64")), optim_states: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((2, 2), dtype="float64"))) -> R.Tuple(R.Tuple(R.Tensor((2, 2), dtype="float64")), R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((2, 2), dtype="float64"))):
            with R.dataflow():
                num_steps: R.Tensor((), dtype="int64") = optim_states[0]
                num_steps_new: R.Tensor((), dtype="int64") = R.add(num_steps, R.const(1, "int64"))
                y: R.Tensor((2, 2), dtype="float64") = params[0]
                y_grad: R.Tensor((2, 2), dtype="float64") = gradients[0]
                y_v: R.Tensor((2, 2), dtype="float64") = optim_states[1]
                lv: R.Tensor((2, 2), dtype="float64") = R.multiply(R.const(0.10000000000000001, "float64"), y_v)
                y_v_new: R.Tensor((2, 2), dtype="float64") = R.add(lv, y_grad)
                lv1: R.Tensor((2, 2), dtype="float64") = R.multiply(R.const(0.10000000000000001, "float64"), y_v_new)
                y_new: R.Tensor((2, 2), dtype="float64") = R.subtract(y, lv1)
                params_new: R.Tuple(R.Tensor((2, 2), dtype="float64")) = (y_new,)
                optim_states_new: R.Tuple(R.Tensor((), dtype="int64"), R.Tensor((2, 2), dtype="float64")) = num_steps_new, y_v_new
                R.output(params_new, optim_states_new)
            return (params_new, optim_states_new)

    # fmt: on

    ty = relax.TensorType((2, 2), "float64")
    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"), MomentumSGD(0.1, 0.1), [ty, ty], legalize=False
    )
    train_mod = setup_trainer(Backbone)
    assert_structural_equal(train_mod.without_attr("optim_state"), Expected)


def test_invalid_mod():
    @I.ir_module
    class NoAttr:
        @R.function
        def backbone(
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
            x: R.Tensor((1, 10), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                gv = R.add(lv0, b0)
                out = R.nn.relu(gv)
                R.output(gv, out)
            return gv, out

    pred_ty = relax.TensorType((1, 5), "float32")
    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.001),
        [pred_ty, pred_ty],
    )

    with pytest.raises((RuntimeError, ValueError)):
        SetupTrainer(
            MSELoss(reduction="sum"),
            SGD(0.001),
            [pred_ty, pred_ty],
        )(NoAttr)

    @I.ir_module
    class WrongFuncName:
        @R.function
        def main(
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
            x: R.Tensor((1, 10), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.add(lv0, b0)
                out = R.nn.relu(lv1)
                R.output(out)
            return out

    with pytest.raises(ValueError):
        setup_trainer(WrongFuncName)


if __name__ == "__main__":
    tvm_tirx.testing.main()
