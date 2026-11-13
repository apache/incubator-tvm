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

import pytest
import torch

import tvm_tirx
import tvm_tirx.testing
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import spec
from tvm_tirx import tirx


@pytest.mark.parametrize("debug", [True, False])
def test_jit(debug):
    class Layer(nn.Module):
        def __init__(self):
            pass

        def forward(self, x: nn.Tensor):
            y = nn.add(x, x)
            return y

    forward_spec = {"forward": {"x": spec.Tensor([10, 5], dtype="float32")}}
    mod = Layer()

    model = mod.jit(spec=forward_spec, debug=debug)

    x = torch.rand((10, 5), dtype=torch.float32)
    y = model["forward"](x)
    assert isinstance(y, torch.Tensor)
    assert torch.allclose(x + x, y)


@pytest.mark.parametrize("debug", [True, False])
def test_jit_int_input(debug):
    class Layer(nn.Module):
        def __init__(self):
            pass

        def forward(self, x: nn.Tensor, i: tirx.Var):
            y = nn.add(x, x)
            y = nn.reshape(y, (i, 5, 5))
            return y

    forward_spec = {"forward": {"x": spec.Tensor([10, 5], dtype="float32"), "i": int}}
    mod = Layer()

    model = mod.jit(spec=forward_spec, debug=debug)

    x = torch.rand((10, 5), dtype=torch.float32)
    y = model["forward"](x, 2)
    assert isinstance(y, torch.Tensor)
    assert torch.allclose(torch.reshape(x + x, (2, 5, 5)), y)


@pytest.mark.parametrize("debug", [True, False])
def test_jit_with_effect(debug):
    class Layer(nn.Module):
        def __init__(self):
            self.cache = nn.KVCache(10, [10, 5])

        def forward(self, x: nn.Tensor, total_seq_len: tirx.Var):
            self.cache.append(x)
            y = self.cache.view(total_seq_len)
            return y

    forward_spec = {
        "forward": {"x": spec.Tensor([1, 10, 5], dtype="float32"), "total_seq_len": int}
    }
    mod = Layer()

    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        model = mod.jit(spec=forward_spec, debug=debug)

    x0 = torch.rand((1, 10, 5), dtype=torch.float32)
    y = model["forward"](x0, 1)
    assert isinstance(y, torch.Tensor)
    assert torch.allclose(x0, y)

    x1 = torch.rand((1, 10, 5), dtype=torch.float32)
    y = model["forward"](x1, 2)
    assert torch.allclose(torch.concat([x0, x1], dim=0), y)

    x2 = torch.rand((1, 10, 5), dtype=torch.float32)
    y = model["forward"](x2, 3)
    assert torch.allclose(torch.concat([x0, x1, x2], dim=0), y)


@pytest.mark.parametrize("debug", [True, False])
def test_jit_tuple_input(debug):
    class Layer(nn.Module):
        def __init__(self):
            pass

        def forward(self, x: tuple[nn.Tensor, nn.Tensor]):
            assert isinstance(x, tuple)
            x0 = x[0]
            x1 = x[1]
            y0 = nn.add(x0, x1)
            y1 = nn.subtract(x0, x1)
            return (y0, y1)

    forward_spec = {
        "forward": {
            "x": (
                spec.Tensor([10, 5], dtype="float32"),
                spec.Tensor([10, 5], dtype="float32"),
            )
        }
    }
    mod = Layer()

    model = mod.jit(spec=forward_spec, debug=debug)

    x0 = torch.rand((10, 5), dtype=torch.float32)
    x1 = torch.rand((10, 5), dtype=torch.float32)
    x = (x0, x1)
    y = model["forward"](x)

    assert torch.allclose(x0 + x1, y[0])
    assert torch.allclose(x0 - x1, y[1])


@pytest.mark.parametrize("debug", [True, False])
def test_jit_list_input(debug):
    class Layer(nn.Module):
        def __init__(self):
            pass

        def forward(self, x: list[nn.Tensor]):
            assert isinstance(x, list)
            x0 = x[0]
            x1 = x[1]
            y0 = nn.add(x0, x1)
            y1 = nn.subtract(x0, x1)
            return (y0, y1)

    forward_spec = {
        "forward": {
            "x": [
                spec.Tensor([10, 5], dtype="float32"),
                spec.Tensor([10, 5], dtype="float32"),
            ]
        }
    }
    mod = Layer()

    model = mod.jit(spec=forward_spec, debug=debug)

    x0 = torch.rand((10, 5), dtype=torch.float32)
    x1 = torch.rand((10, 5), dtype=torch.float32)
    x = (x0, x1)
    y = model["forward"](x)

    assert torch.allclose(x0 + x1, y[0])
    assert torch.allclose(x0 - x1, y[1])


@pytest.mark.parametrize("debug", [True, False])
def test_jit_tuple_input_with_int(debug):
    class Layer(nn.Module):
        def __init__(self):
            pass

        def forward(self, x: tuple[nn.Tensor, nn.Tensor, int]):
            x0 = x[0]
            x1 = x[1]
            y0 = nn.add(x0, x1)
            y1 = nn.subtract(x0, x1)
            y2 = nn.reshape(x0, (5, x[2], 5))
            return (y0, y1, y2)

    forward_spec = {
        "forward": {
            "x": (spec.Tensor([10, 5], dtype="float32"), spec.Tensor([10, 5], dtype="float32"), int)
        }
    }
    mod = Layer()

    model = mod.jit(spec=forward_spec, debug=debug)

    x0 = torch.rand((10, 5), dtype=torch.float32)
    x1 = torch.rand((10, 5), dtype=torch.float32)
    x = (x0, x1, 2)
    y0, y1, y2 = model["forward"](x)

    assert torch.allclose(x0 + x1, y0)
    assert torch.allclose(x0 - x1, y1)
    assert torch.allclose(torch.reshape(x0, (5, 2, 5)), y2)


if __name__ == "__main__":
    tvm_tirx.testing.main()
