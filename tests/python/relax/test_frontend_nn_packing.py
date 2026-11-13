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
# pylint: disable=invalid-name,missing-docstring
import tvm_tirx
from tvm.relax import script as R
from tvm.relax.frontend import nn
from tvm_tirx.script import ir as I


def _iter_binding_names(mod):
    """Helper function to compare the names of relax variables"""
    for block in mod["forward"].body.blocks:
        for binding in block.bindings:
            yield binding.var.name


def test_nn_export_to_relax():
    class TestModule(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.linear_1 = nn.Linear(in_features, out_features, bias=False)
            self.linear_2 = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: nn.Tensor):
            x1 = self.linear_1(x)
            x2 = self.linear_2(x)
            return x1 + x2

    @I.ir_module
    class ExpectedModule:
        @R.function
        def forward(
            x: R.Tensor((1, 10), dtype="float32"),
            packed_params: R.Tuple(
                R.Tensor((20, 10), dtype="float32"), R.Tensor((20, 10), dtype="float32")
            ),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                linear_1_weight = packed_params[0]
                linear_2_weight = packed_params[1]
                matmul_1_weight = R.permute_dims(linear_1_weight)
                matmul = R.matmul(x, matmul_1_weight)
                matmul_2_weight = R.permute_dims(linear_2_weight)
                matmul1 = R.matmul(x, matmul_2_weight)
                add = R.add(matmul, matmul1)
                gv = add
                R.output(gv)
            return gv

    model = TestModule(10, 20)
    mod, _ = model.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor([1, model.in_features], "float32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            }
        }
    )
    tvm_tirx.ir.assert_structural_equal(mod, ExpectedModule)

    for name, expected_name in zip(_iter_binding_names(mod), _iter_binding_names(ExpectedModule)):
        assert name == expected_name


if __name__ == "__main__":
    tvm_tirx.testing.main()
