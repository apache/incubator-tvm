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
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script import ir as I


@I.ir_module
class Module:
    @R.function
    def forward(
        x: R.Tensor((2, 4), "float32"),
        w: R.Tensor((4, 4), "float32"),
        b: R.Tensor((2, 4), "float32"),
    ) -> R.Tensor((2, 4), "float32"):
        with R.dataflow():
            lv: R.Tensor((2, 4), "float32") = R.matmul(x, w)
            out: R.Tensor((2, 4), "float32") = R.add(lv, b)
            R.output(out)
        return out


def test_l1_loss():
    N = 3
    C = 5
    predictions = relax.TensorType((N, C), "float32")
    targets = relax.TensorType((N, C), "float32")
    l1_loss = relax.training.loss.L1Loss()

    @R.function
    def expected(
        predictions: R.Tensor((3, 5), "float32"), targets: R.Tensor((3, 5), "float32")
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "l1_loss"})
        with R.dataflow():
            lv: R.Tensor((3, 5), "float32") = R.subtract(predictions, targets)
            lv1: R.Tensor((3, 5), "float32") = R.abs(lv)
            gv: R.Tensor((), "float32") = R.mean(lv1, axis=None, keepdims=False)
            R.output(gv)
        return gv

    assert_structural_equal(l1_loss(predictions, targets), expected)


def test_l1_loss_append():
    s = Module["forward"].ret_ty
    l1_loss = relax.training.loss.L1Loss(reduction="sum")
    After = relax.training.AppendLoss("forward", l1_loss(s, s), l1_loss.num_backbone_outputs)(
        Module
    )

    @R.function
    def expected(
        x: R.Tensor((2, 4), "float32"),
        w: R.Tensor((4, 4), "float32"),
        b: R.Tensor((2, 4), "float32"),
        targets: R.Tensor((2, 4), "float32"),
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "forward_loss"})
        with R.dataflow():
            lv: R.Tensor((2, 4), "float32") = R.matmul(x, w)
            out: R.Tensor((2, 4), "float32") = R.add(lv, b)
            lv1: R.Tensor((2, 4), "float32") = R.subtract(out, targets)
            lv11: R.Tensor((2, 4), "float32") = R.abs(lv1)
            gv: R.Tensor((), "float32") = R.sum(lv11, axis=None, keepdims=False)
            R.output(gv)
        return gv

    assert_structural_equal(After["forward_loss"], expected)


def test_mse_loss():
    N = 3
    C = 5
    predictions = relax.TensorType((N, C), "float32")
    targets = relax.TensorType((N, C), "float32")
    mse_loss = relax.training.loss.MSELoss()

    @R.function
    def expected(
        predictions: R.Tensor((3, 5), "float32"), targets: R.Tensor((3, 5), "float32")
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "mse_loss"})
        with R.dataflow():
            lv: R.Tensor((3, 5), "float32") = R.subtract(predictions, targets)
            lv1: R.Tensor((3, 5), "float32") = R.multiply(lv, lv)
            gv: R.Tensor((), "float32") = R.mean(lv1, axis=None, keepdims=False)
            R.output(gv)
        return gv

    assert_structural_equal(mse_loss(predictions, targets), expected)


def test_mse_loss_append():
    s = Module["forward"].ret_ty
    mse_loss = relax.training.loss.MSELoss(reduction="sum")
    After = relax.training.AppendLoss("forward", mse_loss(s, s), mse_loss.num_backbone_outputs)(
        Module
    )

    @R.function
    def expected(
        x: R.Tensor((2, 4), "float32"),
        w: R.Tensor((4, 4), "float32"),
        b: R.Tensor((2, 4), "float32"),
        targets: R.Tensor((2, 4), "float32"),
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "forward_loss"})
        with R.dataflow():
            lv: R.Tensor((2, 4), "float32") = R.matmul(x, w)
            out: R.Tensor((2, 4), "float32") = R.add(lv, b)
            lv1: R.Tensor((2, 4), "float32") = R.subtract(out, targets)
            lv11: R.Tensor((2, 4), "float32") = R.multiply(lv1, lv1)
            gv: R.Tensor((), "float32") = R.sum(lv11, axis=None, keepdims=False)
            R.output(gv)
        return gv

    assert_structural_equal(After["forward_loss"], expected)


def test_cross_entropy_loss():
    N = 3
    C = 5
    predictions = relax.TensorType((N, C), "float32")
    targets = relax.TensorType((N,), "int64")
    weights = relax.TensorType((C,), "float32")
    cross_entropy_loss = relax.training.loss.CrossEntropyLoss(reduction="sum", ignore_index=1)

    @R.function
    def expected(
        predictions: R.Tensor((3, 5), "float32"),
        targets: R.Tensor((3,), "int64"),
        weights: R.Tensor((5,), "float32"),
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "cross_entropy_loss"})
        with R.dataflow():
            lv: R.Tensor((3, 5), "float32") = R.nn.log_softmax(predictions, axis=-1)
            gv: R.Tensor((), "float32") = R.nn.nll_loss(
                lv, targets, weights, reduction="sum", ignore_index=1
            )
            R.output(gv)
        return gv

    assert_structural_equal(cross_entropy_loss(predictions, targets, weights), expected)


def test_cross_entropy_loss_without_weights():
    N = 3
    C = 5
    predictions = relax.TensorType((N, C), "float32")
    targets = relax.TensorType((N,), "int64")
    cross_entropy_loss = relax.training.loss.CrossEntropyLoss()

    @R.function
    def expected(
        predictions: R.Tensor((3, 5), "float32"), targets: R.Tensor((3,), "int64")
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "cross_entropy_loss"})
        with R.dataflow():
            lv: R.Tensor((3, 5), "float32") = R.nn.log_softmax(predictions, axis=-1)
            gv: R.Tensor((), "float32") = R.nn.nll_loss(
                lv, targets, reduction="mean", ignore_index=-100
            )
            R.output(gv)
        return gv

    assert_structural_equal(cross_entropy_loss(predictions, targets), expected)


def test_cross_entropy_loss_append():
    s = Module["forward"].ret_ty
    N = s.shape[0]
    C = s.shape[1]
    targets = relax.TensorType((N,), "int64")
    weights = relax.TensorType((C,), "float32")
    cross_entropy_loss = relax.training.loss.CrossEntropyLoss(reduction="sum", ignore_index=1)
    After = relax.training.AppendLoss(
        "forward", cross_entropy_loss(s, targets, weights), cross_entropy_loss.num_backbone_outputs
    )(Module)

    @R.function
    def expected(
        x: R.Tensor((2, 4), "float32"),
        w: R.Tensor((4, 4), "float32"),
        b: R.Tensor((2, 4), "float32"),
        targets: R.Tensor((2,), "int64"),
        weights: R.Tensor((4,), "float32"),
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "forward_loss"})
        with R.dataflow():
            lv: R.Tensor((2, 4), "float32") = R.matmul(x, w)
            out: R.Tensor((2, 4), "float32") = R.add(lv, b)
            lv1: R.Tensor((2, 4), "float32") = R.nn.log_softmax(out, axis=-1)
            gv: R.Tensor((), "float32") = R.nn.nll_loss(
                lv1, targets, weights, reduction="sum", ignore_index=1
            )
            R.output(gv)
        return gv

    assert_structural_equal(After["forward_loss"], expected)


def test_categorical_cross_entropy_loss():
    N = 3
    C = 5
    predictions = relax.TensorType((N, C), "float32")
    targets = relax.TensorType((N, C), "int64")
    weights = relax.TensorType((C,), "float32")
    categorical_cross_entropy_loss = relax.training.loss.CategoricalCrossEntropyLoss(
        reduction="sum"
    )

    @R.function
    def expected(
        predictions: R.Tensor((3, 5), "float32"),
        targets: R.Tensor((3, 5), "int64"),
        weights: R.Tensor((5,), "float32"),
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "categorical_cross_entropy_loss"})
        with R.dataflow():
            lv: R.Tensor((3, 5), "float32") = R.nn.log_softmax(predictions, axis=-1)
            lv: R.Tensor((), "float32") = -lv * targets.astype("float32")
            gv: R.Tensor((), "float32") = R.sum(lv * weights)
            R.output(gv)
        return gv

    assert_structural_equal(categorical_cross_entropy_loss(predictions, targets, weights), expected)


def test_categorical_cross_entropy_loss_without_weights():
    N = 3
    C = 5
    predictions = relax.TensorType((N, C), "float32")
    targets = relax.TensorType((N, C), "int64")
    categorical_cross_entropy_loss = relax.training.loss.CategoricalCrossEntropyLoss()

    @R.function
    def expected(
        predictions: R.Tensor((3, 5), "float32"), targets: R.Tensor((3, 5), "int64")
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "categorical_cross_entropy_loss"})
        with R.dataflow():
            lv: R.Tensor((3, 5), "float32") = R.nn.log_softmax(predictions, axis=-1)
            gv: R.Tensor((), "float32") = R.mean(-lv * targets.astype("float32"))
            R.output(gv)
        return gv

    assert_structural_equal(categorical_cross_entropy_loss(predictions, targets), expected)


def test_categorical_cross_entropy_loss_with_ignore_index():
    N = 3
    C = 5
    predictions = relax.TensorType((N, C), "float32")
    targets = relax.TensorType((N, C), "int64")
    weights = relax.TensorType((C,), "float32")
    categorical_cross_entropy_loss = relax.training.loss.CategoricalCrossEntropyLoss(
        reduction="sum", ignore_index=1
    )

    @R.function
    def expected(
        predictions: R.Tensor((3, 5), "float32"),
        targets: R.Tensor((3, 5), "int64"),
        weights: R.Tensor((5,), "float32"),
    ) -> R.Tensor((), "float32"):
        R.func_attr({"global_symbol": "categorical_cross_entropy_loss"})
        with R.dataflow():
            lv: R.Tensor((3, 5), "float32") = R.nn.log_softmax(predictions, axis=-1)
            targets = relax.op.reshape(
                relax.op.argmax(targets, axis=1), shape=(targets.ty.shape[0],)
            )
            gv: R.Tensor((), "float32") = R.nn.nll_loss(
                lv, targets, weights, reduction="sum", ignore_index=1
            )
            R.output(gv)
        return gv

    assert_structural_equal(categorical_cross_entropy_loss(predictions, targets, weights), expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
