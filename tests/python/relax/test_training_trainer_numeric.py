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
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.training import SetupTrainer, Trainer
from tvm.relax.training.loss import MSELoss
from tvm.relax.training.optimizer import SGD, Adam
from tvm.testing import device_enabled
from tvm_tirx.script import ir as I


def _get_backbone():
    @I.ir_module
    class MLP:
        I.module_attrs({"param_num": 2, "state_num": 0})

        @R.function
        def backbone(
            x: R.Tensor((1, 10), "float32"),
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.add(lv0, b0)
                out = R.nn.relu(lv1)
                R.output(out)
            return out

    return MLP


def _make_dataset():
    N = 100
    return [[np.ones((1, 10)).astype(np.float32), np.array([[0, 0, 1, 0, 0]], np.float32)]] * N


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_execute():
    target = "llvm"
    dev = tvm_tirx.device_from_target(target)
    backbone = _get_backbone()
    pred_ty = relax.TensorType((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        Adam(0.01),
        [pred_ty, pred_ty],
    )

    train_mod = setup_trainer(backbone)
    ex = tvm.compile(train_mod, target)
    vm = relax.VirtualMachine(ex, dev)

    trainer = Trainer(train_mod, vm, dev, False)
    trainer.zero_init_params()
    trainer.xaiver_uniform_init_params()

    dataset = _make_dataset()
    trainer.predict(dataset[0][0])
    trainer.update(dataset[0][0], dataset[0][1])


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_execute_numeric():
    target = "llvm"
    dev = tvm_tirx.device_from_target(target)
    backbone = _get_backbone()
    pred_ty = relax.TensorType((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.01),
        [pred_ty, pred_ty],
    )

    train_mod = setup_trainer(backbone)
    ex = tvm.compile(train_mod, target)
    vm = relax.VirtualMachine(ex, dev)

    trainer = Trainer(train_mod, vm, dev, False)
    trainer.zero_init_params()

    dataset = _make_dataset()
    for _ in range(2):
        for input, label in dataset:
            loss = trainer.update(input, label)
    tvm_tirx.testing.assert_allclose(loss.numpy(), 3.1974423e-14)

    result = trainer.predict(dataset[0][0])
    result_expected = np.array([[0, 0, 0.9999998, 0, 0]], np.float32)
    tvm_tirx.testing.assert_allclose(result.numpy(), result_expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_load_export_params():
    target = "llvm"
    dev = tvm_tirx.device_from_target(target)
    backbone = _get_backbone()
    pred_ty = relax.TensorType((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.01),
        [pred_ty, pred_ty],
    )

    train_mod = setup_trainer(backbone)
    ex = tvm.compile(train_mod, target)
    vm = relax.VirtualMachine(ex, dev)

    trainer = Trainer(train_mod, vm, dev, False)
    trainer.xaiver_uniform_init_params()

    dataset = _make_dataset()
    for input, label in dataset:
        trainer.update(input, label)

    param_dict = trainer.export_params()
    assert "w0" in param_dict
    assert "b0" in param_dict

    trainer1 = Trainer(train_mod, vm, dev, False)
    trainer1.load_params(param_dict)

    x_sample = dataset[np.random.randint(len(dataset))][0]
    tvm_tirx.testing.assert_allclose(
        trainer.predict(x_sample).numpy(), trainer1.predict(x_sample).numpy()
    )


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_setting_error():
    target = "llvm"
    dev = tvm_tirx.device_from_target(target)
    backbone = _get_backbone()
    pred_ty = relax.TensorType((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.01),
        [pred_ty, pred_ty],
    )

    train_mod = setup_trainer(backbone)
    ex = tvm.compile(train_mod, target)
    vm = relax.VirtualMachine(ex, dev)

    trainer = Trainer(train_mod, vm, dev, False)

    dataset = _make_dataset()
    # parameters are not inited
    with pytest.raises(RuntimeError):
        trainer.predict(dataset[0][0])
    with pytest.raises(RuntimeError):
        trainer.update(dataset[0][0], dataset[0][1])


if __name__ == "__main__":
    tvm_tirx.testing.main()
