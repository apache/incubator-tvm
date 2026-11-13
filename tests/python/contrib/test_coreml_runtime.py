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
import os

import numpy as np
import pytest

import tvm
import tvm_tirx
from tvm import rpc, te
from tvm.contrib import coreml_runtime
from tvm_tirx.support import utils, xcode

proxy_host = os.environ.get("TVM_IOS_RPC_PROXY_HOST", "127.0.0.1")
proxy_port = os.environ.get("TVM_IOS_RPC_PROXY_PORT", 9090)
destination = os.environ.get("TVM_IOS_RPC_DESTINATION", "")
key = "iphone"


@pytest.mark.skip("skip because coremltools is not available in CI")
def test_coreml_runtime():
    import coremltools
    from coremltools.models.neural_network import NeuralNetworkBuilder

    def create_coreml_model():
        shape = (2,)
        alpha = 2

        inputs = [
            ("input0", coremltools.models.datatypes.Array(*shape)),
            ("input1", coremltools.models.datatypes.Array(*shape)),
        ]
        outputs = [
            ("output0", coremltools.models.datatypes.Array(*shape)),
            ("output1", coremltools.models.datatypes.Array(*shape)),
        ]
        builder = NeuralNetworkBuilder(inputs, outputs)
        builder.add_elementwise(
            name="Add", input_names=["input0", "input1"], output_name="output0", mode="ADD"
        )
        builder.add_elementwise(
            name="Mul", alpha=alpha, input_names=["input0"], output_name="output1", mode="MULTIPLY"
        )
        return coremltools.models.MLModel(builder.spec)

    def verify(coreml_model, model_path, dev, session=None):
        coreml_model = create_coreml_model()

        out_spec = coreml_model.output_description._fd_spec
        out_names = [spec.name for spec in out_spec]

        # inference via coremltools
        inputs = {}
        for in_spec in coreml_model.input_description._fd_spec:
            name = in_spec.name
            shape = in_spec.type.multiArrayType.shape
            inputs[name] = np.random.random_sample(shape)

        coreml_outputs = [coreml_model.predict(inputs)[name] for name in out_names]

        # inference via tvm coreml runtime
        runtime = coreml_runtime.create("main", model_path, dev, session=session)
        for name in inputs:
            runtime.set_input(
                name,
                dev.tensor(inputs[name]) if session else tvm_tirx.runtime.tensor(inputs[name], dev),
            )
        runtime.invoke()
        tvm_outputs = [runtime.get_output(i).numpy() for i in range(runtime.get_num_outputs())]

        for c_out, t_out in zip(coreml_outputs, tvm_outputs):
            np.testing.assert_almost_equal(c_out, t_out, 3)

    def check_remote(coreml_model):
        temp = utils.tempdir()
        compiled_model = xcode.compile_coreml(coreml_model, out_dir=temp.temp_dir)
        xcode.popen_test_rpc(
            proxy_host, proxy_port, key, destination=destination, libs=[compiled_model]
        )
        compiled_model = os.path.basename(compiled_model)
        remote = rpc.connect(proxy_host, proxy_port, key=key)
        dev = remote.cpu(0)
        verify(coreml_model, compiled_model, dev, remote)

    def check_local(coreml_model):
        temp = utils.tempdir()
        compiled_model = xcode.compile_coreml(coreml_model, out_dir=temp.temp_dir)
        dev = tvm_tirx.cpu(0)
        verify(coreml_model, compiled_model, dev)

    coreml_model = create_coreml_model()
    check_remote(coreml_model)
    check_local(coreml_model)


if __name__ == "__main__":
    test_coreml_runtime()
