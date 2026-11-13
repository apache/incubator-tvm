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

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.frontend import nn
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import ir as I


def test_linear():
    class Activation(nn.Module):
        define_subroutine = True

        def forward(self, state: relax.Expr) -> relax.Var:
            return nn.op.silu(state)

    class Layer(nn.Module):
        define_subroutine = True

        def __init__(self, in_features, out_features):
            self.weights = nn.Parameter((in_features, out_features), dtype="float32")
            self.activation = Activation()

        def forward(self, input: relax.Expr) -> relax.Var:
            state = nn.op.matmul(input, self.weights)
            return self.activation(state)

    batch_size_forward = I.dynamic("batch_size", "int64")
    batch_size_layer = I.dynamic("batch_size", "int64")
    batch_size_activation = I.dynamic("batch_size", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def forward(
            state: R.Tensor((batch_size_forward, 64), dtype="float32"),
            _io: R.Any,
            weights: R.Tensor((64, 32), dtype="float32"),
        ) -> R.Tuple(R.Tensor((batch_size_forward, 32), dtype="float32"), R.Tuple(R.Any)):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                state = Expected.layer(state, weights)
                dataflow_output = (state, (_io,))
                R.output(dataflow_output)
            return dataflow_output

        @R.function
        def _initialize_effect() -> R.Tuple(R.Any):
            with R.dataflow():
                _io: R.Any = R.null_value()
                lv: R.Tuple(R.Any) = (_io,)
                gv: R.Tuple(R.Any) = lv
                R.output(gv)

            return gv

        @R.function(private=True)
        def layer(
            state: R.Tensor((batch_size_layer, 64), dtype="float32"),
            weights: R.Tensor((64, 32), dtype="float32"),
        ) -> R.Tensor((batch_size_layer, 32), dtype="float32"):
            with R.dataflow():
                state = R.matmul(state, weights)
                state = Expected.activation(state)
                dataflow_output = state
                R.output(dataflow_output)
            return dataflow_output

        @R.function(private=True)
        def activation(
            state: R.Tensor((batch_size_activation, 32), dtype="float32"),
        ) -> R.Tensor((batch_size_activation, 32), dtype="float32"):
            with R.dataflow():
                state = R.nn.silu(state)
                dataflow_output = state
                R.output(dataflow_output)
            return dataflow_output

    mod = Layer(64, 32)
    batch_size = tvm_tirx.tirx.Var("batch_size", "int64")
    tvm_mod, _ = mod.export_tvm(
        spec={"forward": {"input": nn.spec.Tensor((batch_size, 64), "float32")}}, debug=True
    )
    assert_structural_equal(Expected, tvm_mod, True)


def test_different_shapes_produce_distinct_subroutines():
    """Regression test: same Module class with different input shapes
    must generate distinct subroutines, not reuse a cached one."""

    class Linear(nn.Module):
        define_subroutine = True

        def __init__(self, in_features, out_features):
            self.weights = nn.Parameter((in_features, out_features), dtype="float32")

        def forward(self, input: relax.Expr) -> relax.Var:
            return nn.op.matmul(input, self.weights)

    class Model(nn.Module):
        def __init__(self):
            self.linear_a = Linear(32, 16)
            self.linear_b = Linear(64, 16)

        def forward(self, x: relax.Expr, y: relax.Expr) -> relax.Var:
            a = self.linear_a(x)
            b = self.linear_b(y)
            return nn.op.add(a, b)

    mod = Model()
    batch_size = tvm_tirx.tirx.Var("batch_size", "int64")
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor((batch_size, 32), "float32"),
                "y": nn.spec.Tensor((batch_size, 64), "float32"),
            }
        },
        debug=True,
    )

    # Collect all private functions (subroutines) in the module
    subroutine_funcs = [
        func
        for gvar, func in tvm_mod.functions.items()
        if isinstance(func, relax.Function)
        and gvar.name_hint
        not in (
            "forward",
            "_initialize_effect",
        )
    ]

    # There must be two distinct Linear subroutines (one for in_features=32,
    # one for in_features=64), not a single cached one reused for both.
    assert len(subroutine_funcs) == 2, (
        f"Expected 2 distinct subroutines for different input shapes, got {len(subroutine_funcs)}"
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
