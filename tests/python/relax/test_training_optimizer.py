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
"""Unit tests for relax optimizer APIs."""

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.training.optimizer import SGD, Adam, MomentumSGD
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script.parser import relax as R


def test_optimizer_error():
    x1 = relax.Var("x1", R.Tensor((3, 3), "float32"))
    x2 = relax.Var("x2", R.Tensor((3, 3), "float64"))
    x3 = relax.Var("x3", R.Tuple([R.Tensor((3, 3), "float32")]))
    x4 = relax.Var("x4", R.Tensor((3, 3), "int64"))
    x5 = relax.Tuple([x1])

    # fine cases
    SGD(0.01).init(x1)
    SGD(0.01).init([x1])
    assert SGD(0.01).init([x2]).dtype == "float64"

    with pytest.raises(ValueError):
        SGD(0.01).init([x1, x1])
    with pytest.raises(ValueError):
        SGD(0.01).init([x1, x2])
    with pytest.raises(ValueError):
        SGD(0.01).init(x3)
    with pytest.raises(ValueError):
        SGD(0.01).init(x4)
    with pytest.raises(ValueError):
        SGD(0.01).init(x5)
    with pytest.raises(
        RuntimeError,
        match="Please call init\\(\\) for the optimizer before calling get_function\\(\\)",
    ):
        SGD(0.01).get_function()


def test_sgd_simple():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    sgd = SGD(0.01).init([x, y]).get_function()

    @R.function
    def sgd_expected(
        params: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        gradients: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        optim_states: R.Tuple(R.Tensor((), "int64")),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        R.Tuple(R.Tensor((), "int64")),
    ):
        R.func_attr({"global_symbol": "SGD"})
        # block 0
        with R.dataflow():
            num_steps: R.Tensor((), "int64") = optim_states[0]
            num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
            x: R.Tensor((3, 3), "float32") = params[0]
            x_grad: R.Tensor((3, 3), "float32") = gradients[0]
            lv: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.01, "float32"), x_grad)
            x_new: R.Tensor((3, 3), "float32") = R.subtract(x, lv)
            y: R.Tensor((3,), "float32") = params[1]
            y_grad: R.Tensor((3,), "float32") = gradients[1]
            lv1: R.Tensor((3,), "float32") = R.multiply(R.const(0.01, "float32"), y_grad)
            y_new: R.Tensor((3,), "float32") = R.subtract(y, lv1)
            params_new: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")) = (
                x_new,
                y_new,
            )
            optim_states_new: R.Tuple(R.Tensor((), "int64")) = (num_steps_new,)
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)

    assert_structural_equal(sgd, sgd_expected)


def test_sgd_complex():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    sgd = SGD(0.01, 0.02).init([x, y]).get_function()

    @R.function
    def sgd_expected(
        params: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        gradients: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        optim_states: R.Tuple(R.Tensor((), "int64")),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        R.Tuple(R.Tensor((), "int64")),
    ):
        R.func_attr({"global_symbol": "SGD"})
        with R.dataflow():
            num_steps: R.Tensor((), "int64") = optim_states[0]
            num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
            x: R.Tensor((3, 3), "float32") = params[0]
            x_grad: R.Tensor((3, 3), "float32") = gradients[0]
            lv: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.02, "float32"), x)
            x_grad_new: R.Tensor((3, 3), "float32") = R.add(lv, x_grad)
            lv1: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.01, "float32"), x_grad_new)
            x_new: R.Tensor((3, 3), "float32") = R.subtract(x, lv1)
            y: R.Tensor((3,), "float32") = params[1]
            y_grad: R.Tensor((3,), "float32") = gradients[1]
            lv2: R.Tensor((3,), "float32") = R.multiply(R.const(0.02, "float32"), y)
            y_grad_new: R.Tensor((3,), "float32") = R.add(lv2, y_grad)
            lv3: R.Tensor((3,), "float32") = R.multiply(R.const(0.01, "float32"), y_grad_new)
            y_new: R.Tensor((3,), "float32") = R.subtract(y, lv3)
            params_new: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")) = (
                x_new,
                y_new,
            )
            optim_states_new: R.Tuple(R.Tensor((), "int64")) = (num_steps_new,)
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)

    assert_structural_equal(sgd, sgd_expected)


def test_momentum_sgd_simple():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    msgd = MomentumSGD(0.01, 0.9).init([x, y]).get_function()

    @R.function
    def msgd_expected(
        params: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        gradients: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        optim_states: R.Tuple(
            R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")
        ),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        R.Tuple(R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
    ):
        R.func_attr({"global_symbol": "MomentumSGD"})
        # block 0
        with R.dataflow():
            num_steps: R.Tensor((), "int64") = optim_states[0]
            num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
            x: R.Tensor((3, 3), "float32") = params[0]
            x_grad: R.Tensor((3, 3), "float32") = gradients[0]
            x_v: R.Tensor((3, 3), "float32") = optim_states[1]
            lv: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.9, "float32"), x_v)
            x_v_new: R.Tensor((3, 3), "float32") = R.add(lv, x_grad)
            lv1: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.01, "float32"), x_v_new)
            x_new: R.Tensor((3, 3), "float32") = R.subtract(x, lv1)
            y: R.Tensor((3,), "float32") = params[1]
            y_grad: R.Tensor((3,), "float32") = gradients[1]
            y_v: R.Tensor((3,), "float32") = optim_states[2]
            lv2: R.Tensor((3,), "float32") = R.multiply(R.const(0.9, "float32"), y_v)
            y_v_new: R.Tensor((3,), "float32") = R.add(lv2, y_grad)
            lv3: R.Tensor((3,), "float32") = R.multiply(R.const(0.01, "float32"), y_v_new)
            y_new: R.Tensor((3,), "float32") = R.subtract(y, lv3)
            params_new: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")) = (
                x_new,
                y_new,
            )
            optim_states_new: R.Tuple(
                R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")
            ) = (num_steps_new, x_v_new, y_v_new)
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)

    assert_structural_equal(msgd, msgd_expected)


def test_momentum_sgd_complex():
    lr, mom, damp, wd, nest = 0.01, 0.9, 0.85, 0.02, False

    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    msgd = MomentumSGD(lr, mom, damp, wd, nest).init([x, y]).get_function()

    @R.function
    def msgd_expected(
        params: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        gradients: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        optim_states: R.Tuple(
            R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")
        ),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        R.Tuple(R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
    ):
        R.func_attr({"global_symbol": "MomentumSGD"})
        # block 0
        with R.dataflow():
            num_steps: R.Tensor((), "int64") = optim_states[0]
            num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
            x: R.Tensor((3, 3), "float32") = params[0]
            x_grad: R.Tensor((3, 3), "float32") = gradients[0]
            x_v: R.Tensor((3, 3), "float32") = optim_states[1]
            lv: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.02, "float32"), x)
            x_grad_new: R.Tensor((3, 3), "float32") = R.add(lv, x_grad)
            lv1: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.9, "float32"), x_v)
            lv2: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.15, "float32"), x_grad_new)
            x_v_new: R.Tensor((3, 3), "float32") = R.add(lv1, lv2)
            lv3: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.01, "float32"), x_v_new)
            x_new: R.Tensor((3, 3), "float32") = R.subtract(x, lv3)
            y: R.Tensor((3,), "float32") = params[1]
            y_grad: R.Tensor((3,), "float32") = gradients[1]
            y_v: R.Tensor((3,), "float32") = optim_states[2]
            lv4: R.Tensor((3,), "float32") = R.multiply(R.const(0.02, "float32"), y)
            y_grad_new: R.Tensor((3,), "float32") = R.add(lv4, y_grad)
            lv5: R.Tensor((3,), "float32") = R.multiply(R.const(0.9, "float32"), y_v)
            lv6: R.Tensor((3,), "float32") = R.multiply(R.const(0.15, "float32"), y_grad_new)
            y_v_new: R.Tensor((3,), "float32") = R.add(lv5, lv6)
            lv7: R.Tensor((3,), "float32") = R.multiply(R.const(0.01, "float32"), y_v_new)
            y_new: R.Tensor((3,), "float32") = R.subtract(y, lv7)
            params_new: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")) = (
                x_new,
                y_new,
            )
            optim_states_new: R.Tuple(
                R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")
            ) = (num_steps_new, x_v_new, y_v_new)
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)

    assert_structural_equal(msgd, msgd_expected)


def test_momentum_sgd_nesterov():
    lr, mom, damp, wd, nest = 0.01, 0.9, 0.85, 0.02, True

    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    msgd = MomentumSGD(lr, mom, damp, wd, nest).init([x, y]).get_function()

    @R.function
    def msgd_expected(
        params: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        gradients: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        optim_states: R.Tuple(
            R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")
        ),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        R.Tuple(R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
    ):
        R.func_attr({"global_symbol": "MomentumSGD"})
        # block 0
        with R.dataflow():
            num_steps: R.Tensor((), "int64") = optim_states[0]
            num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
            x: R.Tensor((3, 3), "float32") = params[0]
            x_grad: R.Tensor((3, 3), "float32") = gradients[0]
            x_v: R.Tensor((3, 3), "float32") = optim_states[1]
            lv: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.02, "float32"), x)
            x_grad_new: R.Tensor((3, 3), "float32") = R.add(lv, x_grad)
            lv1: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.9, "float32"), x_v)
            lv2: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.15, "float32"), x_grad_new)
            x_v_new: R.Tensor((3, 3), "float32") = R.add(lv1, lv2)
            lv3: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.9, "float32"), x_v_new)
            x_g_nest: R.Tensor((3, 3), "float32") = R.add(x_grad_new, lv3)
            lv4: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.01, "float32"), x_g_nest)
            x_new: R.Tensor((3, 3), "float32") = R.subtract(x, lv4)
            y: R.Tensor((3,), "float32") = params[1]
            y_grad: R.Tensor((3,), "float32") = gradients[1]
            y_v: R.Tensor((3,), "float32") = optim_states[2]
            lv5: R.Tensor((3,), "float32") = R.multiply(R.const(0.02, "float32"), y)
            y_grad_new: R.Tensor((3,), "float32") = R.add(lv5, y_grad)
            lv6: R.Tensor((3,), "float32") = R.multiply(R.const(0.9, "float32"), y_v)
            lv7: R.Tensor((3,), "float32") = R.multiply(R.const(0.15, "float32"), y_grad_new)
            y_v_new: R.Tensor((3,), "float32") = R.add(lv6, lv7)
            lv8: R.Tensor((3,), "float32") = R.multiply(R.const(0.9, "float32"), y_v_new)
            y_g_nest: R.Tensor((3,), "float32") = R.add(y_grad_new, lv8)
            lv9: R.Tensor((3,), "float32") = R.multiply(R.const(0.01, "float32"), y_g_nest)
            y_new: R.Tensor((3,), "float32") = R.subtract(y, lv9)
            params_new: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")) = (
                x_new,
                y_new,
            )
            optim_states_new: R.Tuple(
                R.Tensor((), "int64"), R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")
            ) = (num_steps_new, x_v_new, y_v_new)
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)

    assert_structural_equal(msgd, msgd_expected)


def test_adam_simple():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    adam = Adam(0.01).init([x, y]).get_function()

    @R.function
    def adam_expected(
        params: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        gradients: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        optim_states: R.Tuple(
            R.Tensor((), "int64"),
            R.Tensor((), "float32"),
            R.Tensor((), "float32"),
            R.Tensor((3, 3), "float32"),
            R.Tensor((3,), "float32"),
            R.Tensor((3, 3), "float32"),
            R.Tensor((3,), "float32"),
        ),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        R.Tuple(
            R.Tensor((), "int64"),
            R.Tensor((), "float32"),
            R.Tensor((), "float32"),
            R.Tensor((3, 3), "float32"),
            R.Tensor((3,), "float32"),
            R.Tensor((3, 3), "float32"),
            R.Tensor((3,), "float32"),
        ),
    ):
        R.func_attr({"global_symbol": "Adam"})
        # block 0
        with R.dataflow():
            num_steps: R.Tensor((), "int64") = optim_states[0]
            num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
            lv: R.Tensor((), "float32") = optim_states[1]
            beta1_prod: R.Tensor((), "float32") = R.multiply(lv, R.const(0.9, "float32"))
            lv1: R.Tensor((), "float32") = optim_states[2]
            beta2_prod: R.Tensor((), "float32") = R.multiply(lv1, R.const(0.999, "float32"))
            x: R.Tensor((3, 3), "float32") = params[0]
            x_grad: R.Tensor((3, 3), "float32") = gradients[0]
            x_m: R.Tensor((3, 3), "float32") = optim_states[3]
            x_v: R.Tensor((3, 3), "float32") = optim_states[5]
            lv2: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.9, "float32"), x_m)
            lv3: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.1, "float32"), x_grad)
            x_m_new: R.Tensor((3, 3), "float32") = R.add(lv2, lv3)
            lv4: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.999, "float32"), x_v)
            lv5: R.Tensor((3, 3), "float32") = R.multiply(x_grad, x_grad)
            lv6: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.001, "float32"), lv5)
            x_v_new: R.Tensor((3, 3), "float32") = R.add(lv4, lv6)
            lv7: R.Tensor((), "float32") = R.subtract(R.const(1, "float32"), beta1_prod)
            x_m_hat: R.Tensor((3, 3), "float32") = R.divide(x_m_new, lv7)
            lv8: R.Tensor((), "float32") = R.subtract(R.const(1, "float32"), beta2_prod)
            x_v_hat: R.Tensor((3, 3), "float32") = R.divide(x_v_new, lv8)
            lv9: R.Tensor((3, 3), "float32") = R.sqrt(x_v_hat)
            lv10: R.Tensor((3, 3), "float32") = R.add(lv9, R.const(1e-08, "float32"))
            lv11: R.Tensor((3, 3), "float32") = R.divide(x_m_hat, lv10)
            lv12: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.01, "float32"), lv11)
            x_new: R.Tensor((3, 3), "float32") = R.subtract(x, lv12)
            y: R.Tensor((3,), "float32") = params[1]
            y_grad: R.Tensor((3,), "float32") = gradients[1]
            y_m: R.Tensor((3,), "float32") = optim_states[4]
            y_v: R.Tensor((3,), "float32") = optim_states[6]
            lv13: R.Tensor((3,), "float32") = R.multiply(R.const(0.9, "float32"), y_m)
            lv14: R.Tensor((3,), "float32") = R.multiply(R.const(0.1, "float32"), y_grad)
            y_m_new: R.Tensor((3,), "float32") = R.add(lv13, lv14)
            lv15: R.Tensor((3,), "float32") = R.multiply(R.const(0.999, "float32"), y_v)
            lv16: R.Tensor((3,), "float32") = R.multiply(y_grad, y_grad)
            lv17: R.Tensor((3,), "float32") = R.multiply(R.const(0.001, "float32"), lv16)
            y_v_new: R.Tensor((3,), "float32") = R.add(lv15, lv17)
            lv18: R.Tensor((), "float32") = R.subtract(R.const(1, "float32"), beta1_prod)
            y_m_hat: R.Tensor((3,), "float32") = R.divide(y_m_new, lv18)
            lv19: R.Tensor((), "float32") = R.subtract(R.const(1, "float32"), beta2_prod)
            y_v_hat: R.Tensor((3,), "float32") = R.divide(y_v_new, lv19)
            lv20: R.Tensor((3,), "float32") = R.sqrt(y_v_hat)
            lv21: R.Tensor((3,), "float32") = R.add(lv20, R.const(1e-08, "float32"))
            lv22: R.Tensor((3,), "float32") = R.divide(y_m_hat, lv21)
            lv23: R.Tensor((3,), "float32") = R.multiply(R.const(0.01, "float32"), lv22)
            y_new: R.Tensor((3,), "float32") = R.subtract(y, lv23)
            params_new: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")) = (
                x_new,
                y_new,
            )
            optim_states_new: R.Tuple(
                R.Tensor((), "int64"),
                R.Tensor((), "float32"),
                R.Tensor((), "float32"),
                R.Tensor((3, 3), "float32"),
                R.Tensor((3,), "float32"),
                R.Tensor((3, 3), "float32"),
                R.Tensor((3,), "float32"),
            ) = (num_steps_new, beta1_prod, beta2_prod, x_m_new, y_m_new, x_v_new, y_v_new)
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)

    assert_structural_equal(adam, adam_expected)


def test_adam_complex():
    x = relax.Var("x", R.Tensor((3, 3), "float32"))
    y = relax.Var("y", R.Tensor((3,), "float32"))
    adam = Adam(0.01, (0.8, 0.85), 1e-7, 0.1).init([x, y]).get_function()

    @R.function
    def adam_expected(
        params: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        gradients: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        optim_states: R.Tuple(
            R.Tensor((), "int64"),
            R.Tensor((), "float32"),
            R.Tensor((), "float32"),
            R.Tensor((3, 3), "float32"),
            R.Tensor((3,), "float32"),
            R.Tensor((3, 3), "float32"),
            R.Tensor((3,), "float32"),
        ),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")),
        R.Tuple(
            R.Tensor((), "int64"),
            R.Tensor((), "float32"),
            R.Tensor((), "float32"),
            R.Tensor((3, 3), "float32"),
            R.Tensor((3,), "float32"),
            R.Tensor((3, 3), "float32"),
            R.Tensor((3,), "float32"),
        ),
    ):
        R.func_attr({"global_symbol": "Adam"})
        # block 0
        with R.dataflow():
            num_steps: R.Tensor((), "int64") = optim_states[0]
            num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
            lv: R.Tensor((), "float32") = optim_states[1]
            beta1_prod: R.Tensor((), "float32") = R.multiply(lv, R.const(0.8, "float32"))
            lv1: R.Tensor((), "float32") = optim_states[2]
            beta2_prod: R.Tensor((), "float32") = R.multiply(lv1, R.const(0.85, "float32"))
            x: R.Tensor((3, 3), "float32") = params[0]
            x_grad: R.Tensor((3, 3), "float32") = gradients[0]
            x_m: R.Tensor((3, 3), "float32") = optim_states[3]
            x_v: R.Tensor((3, 3), "float32") = optim_states[5]
            lv2: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.1, "float32"), x)
            x_grad_new: R.Tensor((3, 3), "float32") = R.add(lv2, x_grad)
            lv3: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.8, "float32"), x_m)
            lv4: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.2, "float32"), x_grad_new)
            x_m_new: R.Tensor((3, 3), "float32") = R.add(lv3, lv4)
            lv5: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.85, "float32"), x_v)
            lv6: R.Tensor((3, 3), "float32") = R.multiply(x_grad_new, x_grad_new)
            lv7: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.15, "float32"), lv6)
            x_v_new: R.Tensor((3, 3), "float32") = R.add(lv5, lv7)
            lv8: R.Tensor((), "float32") = R.subtract(R.const(1, "float32"), beta1_prod)
            x_m_hat: R.Tensor((3, 3), "float32") = R.divide(x_m_new, lv8)
            lv9: R.Tensor((), "float32") = R.subtract(R.const(1, "float32"), beta2_prod)
            x_v_hat: R.Tensor((3, 3), "float32") = R.divide(x_v_new, lv9)
            lv10: R.Tensor((3, 3), "float32") = R.sqrt(x_v_hat)
            lv11: R.Tensor((3, 3), "float32") = R.add(lv10, R.const(1e-07, "float32"))
            lv12: R.Tensor((3, 3), "float32") = R.divide(x_m_hat, lv11)
            lv13: R.Tensor((3, 3), "float32") = R.multiply(R.const(0.01, "float32"), lv12)
            x_new: R.Tensor((3, 3), "float32") = R.subtract(x, lv13)
            y: R.Tensor((3,), "float32") = params[1]
            y_grad: R.Tensor((3,), "float32") = gradients[1]
            y_m: R.Tensor((3,), "float32") = optim_states[4]
            y_v: R.Tensor((3,), "float32") = optim_states[6]
            lv14: R.Tensor((3,), "float32") = R.multiply(R.const(0.1, "float32"), y)
            y_grad_new: R.Tensor((3,), "float32") = R.add(lv14, y_grad)
            lv15: R.Tensor((3,), "float32") = R.multiply(R.const(0.8, "float32"), y_m)
            lv16: R.Tensor((3,), "float32") = R.multiply(R.const(0.2, "float32"), y_grad_new)
            y_m_new: R.Tensor((3,), "float32") = R.add(lv15, lv16)
            lv17: R.Tensor((3,), "float32") = R.multiply(R.const(0.85, "float32"), y_v)
            lv18: R.Tensor((3,), "float32") = R.multiply(y_grad_new, y_grad_new)
            lv19: R.Tensor((3,), "float32") = R.multiply(R.const(0.15, "float32"), lv18)
            y_v_new: R.Tensor((3,), "float32") = R.add(lv17, lv19)
            lv20: R.Tensor((), "float32") = R.subtract(R.const(1, "float32"), beta1_prod)
            y_m_hat: R.Tensor((3,), "float32") = R.divide(y_m_new, lv20)
            lv21: R.Tensor((), "float32") = R.subtract(R.const(1, "float32"), beta2_prod)
            y_v_hat: R.Tensor((3,), "float32") = R.divide(y_v_new, lv21)
            lv22: R.Tensor((3,), "float32") = R.sqrt(y_v_hat)
            lv23: R.Tensor((3,), "float32") = R.add(lv22, R.const(1e-07, "float32"))
            lv24: R.Tensor((3,), "float32") = R.divide(y_m_hat, lv23)
            lv25: R.Tensor((3,), "float32") = R.multiply(R.const(0.01, "float32"), lv24)
            y_new: R.Tensor((3,), "float32") = R.subtract(y, lv25)
            params_new: R.Tuple(R.Tensor((3, 3), "float32"), R.Tensor((3,), "float32")) = (
                x_new,
                y_new,
            )
            optim_states_new: R.Tuple(
                R.Tensor((), "int64"),
                R.Tensor((), "float32"),
                R.Tensor((), "float32"),
                R.Tensor((3, 3), "float32"),
                R.Tensor((3,), "float32"),
                R.Tensor((3, 3), "float32"),
                R.Tensor((3,), "float32"),
            ) = (num_steps_new, beta1_prod, beta2_prod, x_m_new, y_m_new, x_v_new, y_v_new)
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)

    assert_structural_equal(adam, adam_expected)


def test_adam_float64():
    x = relax.Var("x", R.Tensor((3, 3), "float64"))
    y = relax.Var("y", R.Tensor((3,), "float64"))
    adam = Adam(0.01, (0.8, 0.85), 1e-7, 0.1).init([x, y]).get_function()

    @R.function
    def adam_expected(
        params: R.Tuple(R.Tensor((3, 3), "float64"), R.Tensor((3,), "float64")),
        gradients: R.Tuple(R.Tensor((3, 3), "float64"), R.Tensor((3,), "float64")),
        optim_states: R.Tuple(
            R.Tensor((), "int64"),
            R.Tensor((), "float64"),
            R.Tensor((), "float64"),
            R.Tensor((3, 3), "float64"),
            R.Tensor((3,), "float64"),
            R.Tensor((3, 3), "float64"),
            R.Tensor((3,), "float64"),
        ),
    ) -> R.Tuple(
        R.Tuple(R.Tensor((3, 3), "float64"), R.Tensor((3,), "float64")),
        R.Tuple(
            R.Tensor((), "int64"),
            R.Tensor((), "float64"),
            R.Tensor((), "float64"),
            R.Tensor((3, 3), "float64"),
            R.Tensor((3,), "float64"),
            R.Tensor((3, 3), "float64"),
            R.Tensor((3,), "float64"),
        ),
    ):
        R.func_attr({"global_symbol": "Adam"})
        # block 0
        with R.dataflow():
            num_steps: R.Tensor((), "int64") = optim_states[0]
            num_steps_new: R.Tensor((), "int64") = R.add(num_steps, R.const(1, "int64"))
            lv: R.Tensor((), "float64") = optim_states[1]
            beta1_prod: R.Tensor((), "float64") = R.multiply(lv, R.const(0.8, "float64"))
            lv1: R.Tensor((), "float64") = optim_states[2]
            beta2_prod: R.Tensor((), "float64") = R.multiply(lv1, R.const(0.85, "float64"))
            x: R.Tensor((3, 3), "float64") = params[0]
            x_grad: R.Tensor((3, 3), "float64") = gradients[0]
            x_m: R.Tensor((3, 3), "float64") = optim_states[3]
            x_v: R.Tensor((3, 3), "float64") = optim_states[5]
            lv2: R.Tensor((3, 3), "float64") = R.multiply(R.const(0.1, "float64"), x)
            x_grad_new: R.Tensor((3, 3), "float64") = R.add(lv2, x_grad)
            lv3: R.Tensor((3, 3), "float64") = R.multiply(R.const(0.8, "float64"), x_m)
            lv4: R.Tensor((3, 3), "float64") = R.multiply(R.const(0.2, "float64"), x_grad_new)
            x_m_new: R.Tensor((3, 3), "float64") = R.add(lv3, lv4)
            lv5: R.Tensor((3, 3), "float64") = R.multiply(R.const(0.85, "float64"), x_v)
            lv6: R.Tensor((3, 3), "float64") = R.multiply(x_grad_new, x_grad_new)
            lv7: R.Tensor((3, 3), "float64") = R.multiply(R.const(0.15, "float64"), lv6)
            x_v_new: R.Tensor((3, 3), "float64") = R.add(lv5, lv7)
            lv8: R.Tensor((), "float64") = R.subtract(R.const(1, "float64"), beta1_prod)
            x_m_hat: R.Tensor((3, 3), "float64") = R.divide(x_m_new, lv8)
            lv9: R.Tensor((), "float64") = R.subtract(R.const(1, "float64"), beta2_prod)
            x_v_hat: R.Tensor((3, 3), "float64") = R.divide(x_v_new, lv9)
            lv10: R.Tensor((3, 3), "float64") = R.sqrt(x_v_hat)
            lv11: R.Tensor((3, 3), "float64") = R.add(lv10, R.const(1e-07, "float64"))
            lv12: R.Tensor((3, 3), "float64") = R.divide(x_m_hat, lv11)
            lv13: R.Tensor((3, 3), "float64") = R.multiply(R.const(0.01, "float64"), lv12)
            x_new: R.Tensor((3, 3), "float64") = R.subtract(x, lv13)
            y: R.Tensor((3,), "float64") = params[1]
            y_grad: R.Tensor((3,), "float64") = gradients[1]
            y_m: R.Tensor((3,), "float64") = optim_states[4]
            y_v: R.Tensor((3,), "float64") = optim_states[6]
            lv14: R.Tensor((3,), "float64") = R.multiply(R.const(0.1, "float64"), y)
            y_grad_new: R.Tensor((3,), "float64") = R.add(lv14, y_grad)
            lv15: R.Tensor((3,), "float64") = R.multiply(R.const(0.8, "float64"), y_m)
            lv16: R.Tensor((3,), "float64") = R.multiply(R.const(0.2, "float64"), y_grad_new)
            y_m_new: R.Tensor((3,), "float64") = R.add(lv15, lv16)
            lv17: R.Tensor((3,), "float64") = R.multiply(R.const(0.85, "float64"), y_v)
            lv18: R.Tensor((3,), "float64") = R.multiply(y_grad_new, y_grad_new)
            lv19: R.Tensor((3,), "float64") = R.multiply(R.const(0.15, "float64"), lv18)
            y_v_new: R.Tensor((3,), "float64") = R.add(lv17, lv19)
            lv20: R.Tensor((), "float64") = R.subtract(R.const(1, "float64"), beta1_prod)
            y_m_hat: R.Tensor((3,), "float64") = R.divide(y_m_new, lv20)
            lv21: R.Tensor((), "float64") = R.subtract(R.const(1, "float64"), beta2_prod)
            y_v_hat: R.Tensor((3,), "float64") = R.divide(y_v_new, lv21)
            lv22: R.Tensor((3,), "float64") = R.sqrt(y_v_hat)
            lv23: R.Tensor((3,), "float64") = R.add(lv22, R.const(1e-07, "float64"))
            lv24: R.Tensor((3,), "float64") = R.divide(y_m_hat, lv23)
            lv25: R.Tensor((3,), "float64") = R.multiply(R.const(0.01, "float64"), lv24)
            y_new: R.Tensor((3,), "float64") = R.subtract(y, lv25)
            params_new: R.Tuple(R.Tensor((3, 3), "float64"), R.Tensor((3,), "float64")) = (
                x_new,
                y_new,
            )
            optim_states_new: R.Tuple(
                R.Tensor((), "int64"),
                R.Tensor((), "float64"),
                R.Tensor((), "float64"),
                R.Tensor((3, 3), "float64"),
                R.Tensor((3,), "float64"),
                R.Tensor((3, 3), "float64"),
                R.Tensor((3,), "float64"),
            ) = (num_steps_new, beta1_prod, beta2_prod, x_m_new, y_m_new, x_v_new, y_v_new)
            R.output(params_new, optim_states_new)
        return (params_new, optim_states_new)

    assert_structural_equal(adam, adam_expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
