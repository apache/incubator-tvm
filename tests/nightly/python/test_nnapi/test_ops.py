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
"""NNAPI integration operator tests."""

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.script.relax as R
from test_nnapi.conftest import remote
from test_nnapi.infrastructure import build_and_run


def _build_and_run_network(remote_obj, tracker, mod, input_data):
    """Helper function to build and run a network."""

    def execute_on_host(mod, inputs):
        with tvm_tirx.ir.transform.PassContext(opt_level=3):
            ex = tvm.compile(mod, target="llvm")
        dev = tvm_tirx.cpu(0)
        vm = tvm.relax.VirtualMachine(ex, device=dev)
        output = vm["main"](*inputs)
        return output.numpy()

    outputs = []
    for nnapi in [True, False]:
        if nnapi:
            outputs.append(
                build_and_run(
                    remote_obj,
                    tracker,
                    mod,
                    input_data,
                    enable_nnapi=nnapi,
                )
            )
        else:
            outputs.append(execute_on_host(mod, input_data))
    return outputs


@pytest.mark.parametrize(
    "op",
    [
        R.exp,
        R.log,
        R.negative,
        R.sqrt,
        R.rsqrt,
        R.floor,
        R.nn.relu,
        R.nn.softmax,
        R.sigmoid,
        R.tanh,
        R.abs,
    ],
)
def test_unary(op, input_shape=(1, 2, 8, 5)):
    remote_obj, tracker = remote()

    def create_model() -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(i0: R.Tensor((1, 2, 8, 5), "float32")) -> R.Tensor((1, 2, 8, 5), "float32"):
                with R.dataflow():
                    t0 = op(i0)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[np.random.uniform(size=(1, 2, 8, 5)).astype("float32")],
    )


@pytest.mark.parametrize(
    "op",
    [
        R.power,
        R.greater,
        R.add,
        R.multiply,
        R.subtract,
        R.equal,
        R.less,
        R.less_equal,
        R.not_equal,
        R.maximum,
        R.minimum,
        R.greater_equal,
    ],
)
def test_elementwise_binary(op, input_shape=(1, 2, 8, 5)):
    remote_obj, tracker = remote()

    def create_model() -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((1, 2, 8, 5), "float32"),
                i1: R.Tensor((1, 2, 8, 5), "float32"),
            ) -> R.Tensor((1, 2, 8, 5), "float32"):
                with R.dataflow():
                    t0 = op(i0, i1)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[
            np.random.uniform(size=input_shape).astype("float32"),
            np.random.uniform(size=input_shape).astype("float32"),
        ],
    )


def test_divide(input_shape=(1, 2, 8, 5)):
    remote_obj, tracker = remote()

    def create_model(input_shape) -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((1, 2, 8, 5), "float32"),
                i1: R.Tensor((1, 2, 8, 5), "float32"),
            ) -> R.Tensor((1, 2, 8, 5), "float32"):
                with R.dataflow():
                    t0 = R.divide(i0, i1)
                    R.output(t0)
                return t0

        return Module

    mod = create_model(input_shape)
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[
            np.random.uniform(size=input_shape).astype("float32"),
            np.random.uniform(size=input_shape).astype("float32") + np.ones(input_shape, "float32"),
        ],
    )


def test_matmul():
    remote_obj, tracker = remote()

    def create_model() -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((5, 3, 4), "float32"),
                i1: R.Tensor((5, 4, 8), "float32"),
            ) -> R.Tensor((5, 3, 8), "float32"):
                with R.dataflow():
                    t0 = R.matmul(i0, i1)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[
            np.random.random(size=(5, 3, 4)).astype("float32"),
            np.random.random(size=(5, 4, 8)).astype("float32"),
        ],
    )


def test_permute_dims():
    remote_obj, tracker = remote()

    def create_model() -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((5, 4, 8), "float32"),
            ) -> R.Tensor((8, 5, 4), "float32"):
                with R.dataflow():
                    t0 = R.permute_dims(i0, axes=[2, 0, 1])
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[
            np.random.random(size=(5, 4, 8)).astype("float32"),
        ],
    )


def test_astype():
    remote_obj, tracker = remote()

    def create_model() -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((8, 10, 15), "float32"),
            ) -> R.Tensor((8, 10, 15), "float16"):
                with R.dataflow():
                    t0: R.Tensor((8, 10, 15), "float16") = R.astype(i0, dtype="float16")
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[
            tvm_tirx.runtime.tensor(np.random.uniform(size=(8, 10, 15)).astype("float32")),
        ],
    )


def test_mean():
    remote_obj, tracker = remote()

    def create_model() -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((1, 10, 15), "float32"),
            ) -> R.Tensor((1, 10, 1), "float32"):
                with R.dataflow():
                    t0: R.Tensor((1, 10, 1), "float32") = R.mean(i0, axis=[-1], keepdims=True)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[
            tvm_tirx.runtime.tensor(np.random.uniform(size=(1, 10, 15)).astype("float32")),
        ],
    )


def test_conv2d():
    remote_obj, tracker = remote()

    def create_model() -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((1, 3, 224, 224), "float32"),
                i1: R.Tensor((64, 3, 3, 3), "float32"),
                i2: R.Tensor((1, 64, 1, 1), "float32"),
            ):
                with R.dataflow():
                    t0 = R.nn.conv2d(i0, i1, strides=(1, 1), padding=(1, 1))
                    t0 = R.add(i2, t0)
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[
            np.random.random(size=(1, 3, 224, 224)).astype("float32"),
            np.random.random(size=(64, 3, 3, 3)).astype("float32"),
            np.random.random(size=(1, 64, 1, 1)).astype("float32"),
        ],
    )


def test_max_pool2d():
    remote_obj, tracker = remote()

    def create_model() -> tvm_tirx.IRModule:
        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(
                i0: R.Tensor((1, 1, 28, 28), "float32"),
            ):
                with R.dataflow():
                    t0 = R.nn.max_pool2d(i0, pool_size=(1, 1), strides=(1, 1), padding=(0, 0))
                    R.output(t0)
                return t0

        return Module

    mod = create_model()
    verify(
        remote_obj,
        tracker,
        mod,
        inputs=[
            np.random.random(size=(1, 1, 28, 28)).astype("float32"),
        ],
    )


def verify(remote_obj, tracker, mod, inputs):
    inputs_tvm: list[tvm_tirx.runtime.Tensor] = [tvm_tirx.runtime.tensor(v) for v in inputs]
    outputs = _build_and_run_network(remote_obj, tracker, mod, inputs_tvm)
    nnapi_out = outputs[0]
    expected_out = outputs[1]
    tvm_tirx.testing.assert_allclose(nnapi_out, expected_out, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    tvm_tirx.testing.main()
