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
# pylint: disable=missing-docstring
import torch
import tvm_ffi

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import op, spec
from tvm_tirx import tirx
from tvm_tirx.runtime import Tensor


def test_debug_print():
    class Layer(nn.Module):
        def forward(self, x: nn.Tensor):  # pylint: disable=invalid-name
            op.print_(x)
            return x

    model = Layer().jit(
        spec={
            "forward": {"x": spec.Tensor([10, 5], dtype="float32")},
        },
        debug=True,
    )
    x = torch.rand((10, 5), dtype=torch.float32)  # pylint: disable=invalid-name
    y = model["forward"](x)  # pylint: disable=invalid-name
    assert isinstance(y, torch.Tensor)


def test_debug_func():
    @tvm_ffi.register_global_func("testing.relax.frontend.nn.test_debug_func")
    def _debug(  # pylint: disable=too-many-arguments
        lineno: str,
        tensor: Tensor,
        const_int: int,
        const_float: float,
        const_str: str,
        var_int: int,
    ) -> None:
        assert "test_frontend_nn_debug.py" in lineno
        assert tensor.shape == (10, 5)
        assert const_int == 1
        assert const_float == 2.0
        assert const_str == "test"
        assert var_int == 8

    class Layer(nn.Module):
        def forward(self, x: nn.Tensor, v: tirx.Var):  # pylint: disable=invalid-name
            op.debug_func("testing.relax.frontend.nn.test_debug_func", x, 1, 2.0, "test", v)
            return x

    model = Layer().jit(
        spec={
            "forward": {
                "x": spec.Tensor([10, 5], dtype="float32"),
                "v": "int",
            },
        },
        debug=True,
    )
    x = torch.rand((10, 5), dtype=torch.float32)  # pylint: disable=invalid-name
    y = model["forward"](x, 8)  # pylint: disable=invalid-name
    assert isinstance(y, torch.Tensor)


if __name__ == "__main__":
    test_debug_print()
    test_debug_func()
