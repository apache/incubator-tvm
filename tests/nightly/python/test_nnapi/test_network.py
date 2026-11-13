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
"""NNAPI network tests."""

import numpy as np
import pytest

pytest.importorskip("onnx")

import onnx

import tvm
import tvm_tirx
from test_nnapi.conftest import remote
from test_nnapi.infrastructure import build_and_run  # , build_and_run_vm
from tvm.contrib.download import download_testdata
from tvm.relax.frontend.onnx import from_onnx
from tvm.testing import env as tvm_env


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


def get_network(name, dtype, input_shape=(1, 3, 224, 224)):
    def download_model(model_url, name):
        model_path = download_testdata(model_url, name + ".onnx", module="onnx")
        onnx_model = onnx.load(model_path)

        shape_dict = {"x": input_shape}
        mod = from_onnx(onnx_model, shape_dict)
        return mod

    def create_model(name):
        if "vgg11" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/vgg11_Opset18_timm/vgg11_Opset18.onnx"
        elif "mobilenetv3" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/mobilenetv3_large_100_miil_Opset17_timm/mobilenetv3_large_100_miil_Opset17.onnx"
        elif "alexnet" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/alexnet_Opset17_torch_hub/alexnet_Opset17.onnx"
        elif "resnet50" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/resnet50_Opset18_timm/resnet50_Opset18.onnx"
        elif "resnet34" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/resnet34_Opset18_timm/resnet34_Opset18.onnx"
        elif "resnet18" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/resnet18_Opset18_timm/resnet18_Opset18.onnx"
        elif "squeezenet" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/squeezenet1_1_Opset18_torch_hub/squeezenet1_1_Opset18.onnx"
        elif "vgg16" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/vgg16_Opset18_timm/vgg16_Opset18.onnx"
        elif "vgg19" == name:
            model_url = "https://github.com/onnx/models/raw/bec48b6a70e5e9042c0badbaafefe4454e072d08/Computer_Vision/vgg19_Opset18_timm/vgg19_Opset18.onnx"
        else:
            assert False, f"Not supported model {name}"

        return download_model(model_url, name)

    mod = create_model(name)
    return mod, {"data": (input_shape, dtype)}


@pytest.mark.parametrize(
    "name",
    [
        "alexnet",
        "vgg11",
        "vgg16",
        "vgg19",
        "resnet18",
        "resnet34",
        "resnet50",
        "squeezenet",
        "mobilenetv3",
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
    ],
)
@pytest.mark.skipif(not tvm_env.build_flag_enabled("USE_NNAPI_CODEGEN"), reason="need nnapi")
def test_network(name, dtype):
    remote_obj, tracker = remote()
    print(f"Network evaluating {name} with dtype {dtype}")
    np.random.seed(0)
    mod, inputs = get_network(name, dtype)
    input_data = {}

    for _name, (shape, _dtype) in inputs.items():
        input_data[_name] = np.random.uniform(-1.0, 1.0, shape).astype(_dtype)

    inputs_tvm: list[tvm_tirx.runtime.Tensor] = [
        tvm_tirx.runtime.tensor(v) for k, v in input_data.items()
    ]
    outputs = _build_and_run_network(remote_obj, tracker, mod, inputs_tvm)
    nnapi_out = outputs[0]
    expected_out = outputs[1]
    tvm_tirx.testing.assert_allclose(nnapi_out, expected_out, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    tvm_tirx.testing.main()
