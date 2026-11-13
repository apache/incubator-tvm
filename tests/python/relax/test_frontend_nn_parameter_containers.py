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

from typing import Any

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm.relax.frontend import nn


class ParamContainerModule(nn.Module):
    def __init__(self):
        self.list_params = nn.ParameterList(
            [
                nn.Parameter((4,), "float32"),
                nn.Parameter((4,), "float32"),
            ]
        )
        self.dict_params = nn.ParameterDict(
            {
                "foo": nn.Parameter((4,), "float32"),
                "bar": nn.Parameter((4,), "float32"),
            }
        )


def test_parameter_list_basic_behavior():
    p0 = nn.Parameter((4,), "float32")
    p1 = nn.Parameter((4,), "float32")
    params = nn.ParameterList([p0])
    params.append(p1)

    assert len(params) == 2
    assert params[0] is p0
    assert list(params) == [p0, p1]

    p2 = nn.Parameter((4,), "float32")
    params[1] = p2
    assert params[1] is p2

    p3 = nn.Parameter((4,), "float32")
    params.extend([p3])
    assert list(params) == [p0, p2, p3]


def test_parameter_dict_basic_behavior():
    p0 = nn.Parameter((4,), "float32")
    p1 = nn.Parameter((4,), "float32")
    params = nn.ParameterDict({"foo": p0})
    params["bar"] = p1

    assert len(params) == 2
    assert params["foo"] is p0
    assert "bar" in params
    assert list(params) == ["foo", "bar"]
    assert list(params.keys()) == ["foo", "bar"]
    assert list(params.values()) == [p0, p1]
    assert list(params.items()) == [("foo", p0), ("bar", p1)]
    assert params.get("foo") is p0

    p2 = nn.Parameter((4,), "float32")
    params.update({"baz": p2})
    assert list(params.keys()) == ["foo", "bar", "baz"]
    assert params.pop("baz") is p2
    params.clear()
    assert len(params) == 0


def test_type_validation():
    with pytest.raises(TypeError):
        nn.ParameterList([object()])

    with pytest.raises(TypeError):
        nn.ParameterDict({"bad": object()})

    with pytest.raises(TypeError):
        nn.ParameterDict({1: nn.Parameter((4,), "float32")})

    with pytest.raises(TypeError):
        nn.ParameterList()[0] = object()


def test_named_parameters_parameters_and_state_dict():
    m = ParamContainerModule()

    expected = [
        "list_params.0",
        "list_params.1",
        "dict_params.foo",
        "dict_params.bar",
    ]

    assert list(m.state_dict().keys()) == expected
    assert [name for name, _ in m.named_parameters()] == expected
    assert len(list(m.parameters())) == 4


def test_nested_traversal_through_module_dict():
    class Inner(nn.Module):
        def __init__(self):
            self.params = nn.ParameterList([nn.Parameter((4,), "float32")])

    class Outer(nn.Module):
        def __init__(self):
            self.blocks = nn.ModuleDict({"inner": Inner()})

    m = Outer()
    assert list(m.state_dict().keys()) == ["blocks.inner.params.0"]


def test_nested_traversal_through_module_list():
    class Inner(nn.Module):
        def __init__(self):
            self.params = nn.ParameterList([nn.Parameter((4,), "float32")])

    class Outer(nn.Module):
        def __init__(self):
            self.blocks = nn.ModuleList([Inner()])

    m = Outer()
    assert list(m.state_dict().keys()) == ["blocks.0.params.0"]


def test_to_dtype():
    m = ParamContainerModule()
    m.to(dtype="float16")

    assert m.list_params[0].dtype == "float16"
    assert m.list_params[1].dtype == "float16"
    assert m.dict_params["foo"].dtype == "float16"
    assert m.dict_params["bar"].dtype == "float16"


def test_load_state_dict():
    m = ParamContainerModule()
    p0 = nn.Parameter((4,), "float32")
    p0.data = np.full((4,), 1.0, dtype="float32")
    p1 = nn.Parameter((4,), "float32")
    p1.data = np.full((4,), 2.0, dtype="float32")
    p2 = nn.Parameter((4,), "float32")
    p2.data = np.full((4,), 3.0, dtype="float32")
    p3 = nn.Parameter((4,), "float32")
    p3.data = np.full((4,), 4.0, dtype="float32")
    state_dict = {
        "list_params.0": p0,
        "list_params.1": p1,
        "dict_params.foo": p2,
        "dict_params.bar": p3,
    }

    missing_keys, unexpected_keys = m.load_state_dict(state_dict)

    assert missing_keys == []
    assert unexpected_keys == []
    tvm_tirx.testing.assert_allclose(m.list_params[0].data.numpy(), np.full((4,), 1.0, "float32"))
    tvm_tirx.testing.assert_allclose(m.list_params[1].data.numpy(), np.full((4,), 2.0, "float32"))
    tvm_tirx.testing.assert_allclose(
        m.dict_params["foo"].data.numpy(), np.full((4,), 3.0, "float32")
    )
    tvm_tirx.testing.assert_allclose(
        m.dict_params["bar"].data.numpy(), np.full((4,), 4.0, "float32")
    )


def test_export_tvm_parameter_names():
    class M(nn.Module):
        def __init__(self):
            self.biases = nn.ParameterList(
                [
                    nn.Parameter((4,), "float32"),
                    nn.Parameter((4,), "float32"),
                ]
            )
            self.scales = nn.ParameterDict({"main": nn.Parameter((4,), "float32")})

        def forward(self, x):
            return x + self.biases[0] + self.biases[1] + self.scales["main"]

    _, params = M().export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((4,), "float32")}},
        debug=False,
    )
    assert [name for name, _ in params] == ["biases.0", "biases.1", "scales.main"]


def test_mutator_parameter_container_names():
    seen = []

    class Recorder(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            seen.append(name)
            return node

    m = ParamContainerModule()
    Recorder().visit_module("", m)

    assert seen == [
        "list_params.0",
        "list_params.1",
        "dict_params.foo",
        "dict_params.bar",
    ]


if __name__ == "__main__":
    tvm_tirx.testing.main()
