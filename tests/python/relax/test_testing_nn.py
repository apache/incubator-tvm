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
# ruff: noqa: RUF005
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.testing import nn
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_emit():
    class ReLU(nn.Module):
        def forward(self, input: relax.Expr) -> relax.Var:
            return nn.emit(relax.op.nn.relu(input))

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((32, 32), dtype="float32")) -> R.Tensor((32, 32), dtype="float32"):
            gv: R.Tensor((32, 32), dtype="float32") = R.nn.relu(x)
            return gv

    bb = relax.BlockBuilder()
    with bb.function("main"):
        model = ReLU()
        x = nn.Placeholder((32, 32), dtype="float32", name="x")
        output = model(x)
        params = [x] + model.parameters()
        bb.emit_func_output(output, params)

    tvm_tirx.ir.assert_structural_equal(bb.get(), Expected)


def test_get_param():
    class Plus1(nn.Module):
        def __init__(self):
            self.const_1 = relax.const(1, "float32")

        def forward(self, input: relax.Expr) -> relax.Var:
            return nn.emit(relax.op.add(input, self.const_1))

    model = Plus1()
    assert model.parameters() == []


def test_define_subroutine():
    """Define subroutines when nn.Module.define_subroutine is True"""

    class Activation(nn.Module):
        define_subroutine = True

        def forward(self, state: relax.Expr) -> relax.Var:
            return relax.op.nn.relu(state)

    class Layer(nn.Module):
        define_subroutine = True

        def __init__(self, in_features, out_features):
            self.weights = nn.Parameter(
                (in_features, out_features), dtype="float32", name="weights"
            )
            self.activation = Activation()

        def forward(self, input: relax.Expr) -> relax.Var:
            state = relax.op.matmul(input, self.weights)
            return self.activation(state)

    batch_size_main = T.dynamic("batch_size", "int64")
    batch_size_layer = T.dynamic("batch_size", "int64")
    batch_size_activation = T.dynamic("batch_size", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            state: R.Tensor((batch_size_main, 64), dtype="float32"),
            weights: R.Tensor((64, 32), dtype="float32"),
        ) -> R.Tensor((batch_size_main, 32), dtype="float32"):
            state = Expected.layer(state, weights)
            return state

        @R.function(private=True)
        def layer(
            state: R.Tensor((batch_size_layer, 64), dtype="float32"),
            weights: R.Tensor((64, 32), dtype="float32"),
        ) -> R.Tensor((batch_size_layer, 32), dtype="float32"):
            state = R.matmul(state, weights)
            state = Expected.activation(state)
            return state

        @R.function(private=True)
        def activation(state: R.Tensor((batch_size_activation, 32), dtype="float32")) -> R.Tensor(
            (batch_size_activation, 32), dtype="float32"
        ):
            state = R.nn.relu(state)
            return state

    model = Layer(64, 32)
    batch_size = tvm_tirx.tirx.Var("batch_size", "int64")
    input = nn.Placeholder((batch_size, 64), dtype="float32", name="input")

    bb = relax.BlockBuilder()
    with bb.function("main", params=[input, *model.parameters()]):
        output = model(input)
        bb.emit_func_output(output)

    tvm_tirx.ir.assert_structural_equal(Expected, bb.get())


if __name__ == "__main__":
    tvm_tirx.testing.main()
