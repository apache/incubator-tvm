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

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.frontend import nn
from tvm_tirx import tirx
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_simple():
    """The nn.modules.* may be exported from nn.Module to Relax"""

    slm_mod = nn.modules.ReLU()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((3, 3), "float32")}},
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(x: R.Tensor([3, 3], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                relu = relu
                R.output(relu)
            return relu

    assert_structural_equal(exported_mod, Expected)


def test_custom_module():
    """A user can define their own nn.Module subclasses

    Like the built-in subclasses, these can be exported from nn.Module
    to Relax.
    """

    class Before(nn.Module):
        def forward(self, x: R.Tensor):
            return nn.op.relu(x)

    slm_mod = Before()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((3, 3), "float32")}},
        debug=False,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(x: R.Tensor([3, 3], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                relu = relu
                R.output(relu)
            return relu

    assert_structural_equal(exported_mod, Expected)


def test_debug_effect():
    """Passing debug=True provides an argument for IO effects"""

    slm_mod = nn.modules.ReLU()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((3, 3), "float32")}},
        debug=True,
    )

    @I.ir_module
    class Expected:
        @R.function
        def forward(
            x: R.Tensor([3, 3], dtype="float32"),
            _io: R.Any,
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                relu = R.nn.relu(x)
                output = relu, (_io,)
                R.output(output)
            return output

        @R.function
        def _initialize_effect():
            with R.dataflow():
                _io = R.null_value()
                output = (_io,)
                output = output
                R.output(output)
            return output

    assert_structural_equal(exported_mod, Expected)


def test_dynamic_shape():
    """An argument may have a dynamic shape"""

    slm_mod = nn.modules.ReLU()
    exported_mod, _ = slm_mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor([tirx.Var("batch_size", "int64"), 8], "float32")}},
        debug=False,
    )

    batch_size = T.dynamic("batch_size", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def forward(x: R.Tensor([batch_size, 8], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                relu = relu
                R.output(relu)
            return relu

    assert_structural_equal(exported_mod, Expected)


def test_dynamic_shape_in_multiple_functions():
    """A dynamic shape may be used in multiple functions"""

    class Before(nn.Module):
        def forward_relu(self, x: nn.Tensor):
            return nn.relu(x)

        def forward_silu(self, x: nn.Tensor):
            return nn.silu(x)

    slm_mod = Before()
    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward_relu": {"x": nn.spec.Tensor((tirx.Var("batch_size", "int64"), 8), "float32")},
            "forward_silu": {"x": nn.spec.Tensor((tirx.Var("batch_size", "int64"), 8), "float32")},
        },
        debug=False,
    )

    batch_size_forward_relu = T.dynamic("batch_size", "int64")
    batch_size_forward_silu = T.dynamic("batch_size", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def forward_relu(x: R.Tensor([batch_size_forward_relu, 8], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                relu = R.nn.relu(x)
                relu = relu
                R.output(relu)
            return relu

        @R.function
        def forward_silu(x: R.Tensor([batch_size_forward_silu, 8], dtype="float32")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                silu = R.nn.silu(x)
                silu = silu
                R.output(silu)
            return silu

    assert_structural_equal(exported_mod, Expected)


def test_export_nested_module():
    """nn.Module instances may contain other nn.Module

    When exporting to a Relax IRModule, all `nn.Parameter` instances
    within the `nn.Module` become Relax function parameters.
    """

    class LlamaMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.gate_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.down_proj = nn.Linear(
                intermediate_size,
                hidden_size,
                dtype="float16",
                bias=False,
            )

        def forward(self, x: nn.Tensor):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            return self.down_proj(nn.op.silu(gate) * up)

    hidden_size = 4096
    intermediate_size = 11008
    slm_mod = LlamaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor((tirx.Var("batch_size", "int64"), hidden_size), "float16")
            },
        },
        debug=False,
    )

    batch_size = T.dynamic("batch_size", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def forward(
            x: R.Tensor([batch_size, hidden_size], "float16"),
            gate_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16"),
            up_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16"),
            down_proj_weights: R.Tensor([hidden_size, intermediate_size], "float16"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gate: R.Tensor([batch_size, intermediate_size]) = R.matmul(
                    x, R.permute_dims(gate_proj_weights)
                )
                up: R.Tensor([batch_size, intermediate_size]) = R.matmul(
                    x, R.permute_dims(up_proj_weights)
                )
                down: R.Tensor([batch_size, hidden_size]) = R.matmul(
                    R.nn.silu(gate) * up, R.permute_dims(down_proj_weights)
                )
                down = down
                R.output(down)
            return down

    assert_structural_equal(exported_mod, Expected)


@pytest.mark.xfail(reason="Not yet supported.  See revert https://github.com/apache/tvm/pull/16777")
def test_generate_parameters():
    """Weights may be expressions in terms of other parameters

    Optimizations often require preprocessing of the model weights.

    1. Declare the `nn.Module` members that contain the original model
       weights.  These are used to define the parameter names when
       reading from a Pytorch or Safetensors file.

    2. Declare the `nn.Module` members, with the `weight` field
       in terms of the un-optimized weights.  These `nn.Module`
       do not generate any parameters in the Relax function.

    3. Define the `forward` function in terms of the `nn.Module`
       members for the updated weight tensors.

    The exported Relax function accepts the original model parameters,
    computes the pre-processed weights, and then performs computations
    using the pre-processed weights.

    In this example, the `LiftTransformParams` transform is applied
    immediately, splitting the Relax function into a pre-processing
    step and an execution step.  In practice, this transform would be
    applied much later in an optimization pipeline, to allow optimized
    compute kernels to be recognized.  For example, in some cases
    `R.matmul(x, R.permute_dims(weight))` may be computed more
    efficiently than `R.matmul(x, weight_transpose)`.  For this
    reason, we do *not* apply `LiftTransformParams` as part of the
    export from `nn.Module` to Relax.

    """

    class LlamaMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            # The nn.Linear for the original parameters are present in
            # the model definition, and are still found when
            # collecting a function's parameters.
            self.gate_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )
            self.down_proj = nn.Linear(
                intermediate_size,
                hidden_size,
                dtype="float16",
                bias=False,
            )

            # At runtime, we'd like to have a single concatenated
            # tensor containing both the gate and up projection
            # weights.  We also want to use it in the `forward`
            # function as if it owned its own weights.
            self.gate_up_proj = nn.Linear(
                in_features=hidden_size,
                out_features=intermediate_size,
                dtype="float16",
                bias=False,
            )

            # The weight tensor of `gate_up_proj` can be overwritten
            # in terms of the original `gate_proj` and `up_proj`
            # tensors.
            self.gate_up_proj.weight = nn.op.concat(
                [self.gate_proj.weight, self.up_proj.weight], dim=0, name="gate_up_proj_weights"
            )

        def forward(self, x: nn.Tensor):
            # Even though the `gate_up_proj` weights are defined as an
            # expression rather than a `nn.Parameter`, the `forward`
            # function does not require any special handling for it.
            concat_gate_up = self.gate_up_proj(x)
            gate, up = nn.op.split(concat_gate_up, 2, axis=-1)
            return self.down_proj(nn.op.silu(gate) * up)

    hidden_size = 4096
    intermediate_size = 11008
    slm_mod = LlamaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward": {
                "x": nn.spec.Tensor((tirx.Var("batch_size", "int64"), hidden_size), "float16")
            },
        },
        debug=False,
    )

    batch_size = T.dynamic("batch_size", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def forward(
            x: R.Tensor([batch_size, hidden_size], "float16"),
            # The function's parameters are defined by the
            # `nn.Parameter` instances, and still reference the
            # original `gate_proj` and `up_proj` weights.  This
            # maintains compatibility with named model weights in a
            # Pytorch or Safetensors file.
            gate_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16"),
            up_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16"),
            down_proj_weights: R.Tensor([hidden_size, intermediate_size], "float16"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                # At this stage of compilation, the concatenation is
                # written within the body of the function.  This will
                # later be extracted into a pre-processing step using
                # `relax.transform.LiftTransformParams`.
                gate_up_proj_weights: R.Tensor([intermediate_size * 2, hidden_size], "float16") = (
                    R.concat([gate_proj_weights, up_proj_weights], axis=0)
                )
                gate_up: R.Tensor([batch_size, intermediate_size * 2], "float16") = R.matmul(
                    x, R.permute_dims(gate_up_proj_weights)
                )
                gate_up_split = R.split(gate_up, 2, axis=-1)
                gate = gate_up_split[0]
                up = gate_up_split[1]
                down: R.Tensor([batch_size, hidden_size], "float16") = R.matmul(
                    R.nn.silu(gate) * up, R.permute_dims(down_proj_weights)
                )
                R.output(down)
            return down

    assert_structural_equal(exported_mod, Expected)

    batch_size = T.dynamic("batch_size", "int64")

    @I.ir_module
    class ExpectedAfterLift:
        @R.function
        def forward(
            x: R.Tensor([batch_size, hidden_size], "float16"),
            # After `relax.transform.LiftTransformParams`, the
            # `gate_proj` and `up_proj` weights have been concatenated
            # together.
            gate_up_proj_weights_transpose: R.Tensor(
                [hidden_size, intermediate_size * 2], "float16"
            ),
            down_proj_weights_transpose: R.Tensor([intermediate_size, hidden_size], "float16"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gate_up: R.Tensor([batch_size, intermediate_size * 2], "float16") = R.matmul(
                    x, gate_up_proj_weights_transpose
                )
                gate_up_split = R.split(gate_up, 2, axis=-1)
                gate = gate_up_split[0]
                up = gate_up_split[1]
                down: R.Tensor([batch_size, hidden_size], "float16") = R.matmul(
                    R.nn.silu(gate) * up, down_proj_weights_transpose
                )
                R.output(down)
            return down

        @R.function
        def transform_params(
            model_params: R.Tuple(
                R.Tensor([intermediate_size, hidden_size], "float16"),
                R.Tensor([intermediate_size, hidden_size], "float16"),
                R.Tensor([hidden_size, intermediate_size], "float16"),
            ),
        ):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gate_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16") = (
                    model_params[0]
                )
                up_proj_weights: R.Tensor([intermediate_size, hidden_size], "float16") = (
                    model_params[1]
                )
                gate_up_proj_weights: R.Tensor([intermediate_size * 2, hidden_size], "float16") = (
                    R.concat([gate_proj_weights, up_proj_weights], axis=0)
                )
                gate_up_proj_weights_transpose: R.Tensor(
                    [hidden_size, intermediate_size * 2], "float16"
                ) = R.permute_dims(gate_up_proj_weights)
                down_proj_weights: R.Tensor([hidden_size, intermediate_size], "float16") = (
                    model_params[2]
                )
                down_proj_weights_transpose: R.Tensor(
                    [intermediate_size, hidden_size], "float16"
                ) = R.permute_dims(down_proj_weights)
                output = (gate_up_proj_weights_transpose, down_proj_weights_transpose)
                R.output(output)
            return output

    lifted_mod = relax.transform.LiftTransformParams(shared_transform=True)(exported_mod)
    assert_structural_equal(lifted_mod, ExpectedAfterLift)


def test_linear_dynamic_shape():
    """The weight and bias of nn.Linear have the same out_features

    Even if dynamic, the weight/bias must be the same value.
    """

    n = T.dynamic("n", "int64")

    @R.function
    def forward(
        x: R.Tensor((1, 4), dtype="float32"),
        _io: R.Any,
        weight: R.Tensor((n, 4), dtype="float32"),
        bias: R.Tensor((n,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((1, n), dtype="float32"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            permute_dims: R.Tensor((4, n), dtype="float32") = R.permute_dims(weight, axes=None)
            matmul: R.Tensor((1, n), dtype="float32") = R.matmul(x, permute_dims)
            add: R.Tensor((1, n), dtype="float32") = R.add(matmul, bias)
            gv1: R.Tuple(R.Tensor((1, n), dtype="float32"), R.Tuple(R.Any)) = add, (_io,)
            R.output(gv1)
        return gv1

    mod = nn.modules.Linear(in_features=4, out_features="n", bias=True)
    tvm_mod, _ = mod.export_tvm(
        spec={"forward": {"x": nn.spec.Tensor((1, 4), "float32")}}, debug=True
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


@pytest.mark.parametrize(
    "dynamic_type",
    [
        "same_python_string",
        "different_python_string",
        "same_tir_var",
        "distinct_tir_vars_with_distinct_names",
        pytest.param(
            "distinct_tir_vars_with_same_name",
            marks=pytest.mark.xfail(
                reason="Not yet supported.  See revert https://github.com/apache/tvm/pull/16777"
            ),
        ),
    ],
)
def test_duplicate_names(dynamic_type):
    class Linear(nn.Module):
        def __init__(self, input_size, output_size):
            self.weights = nn.Parameter([output_size, input_size], dtype="float32")

        def forward(self, state: nn.Tensor):
            matmul_weights = nn.op.permute_dims(self.weights)
            return nn.op.matmul(state, matmul_weights)

    class Model(nn.Module):
        def __init__(self, hidden_size, intermediate_size):
            self.embedding = Linear(1024, hidden_size)
            self.up = Linear(hidden_size, intermediate_size)
            self.down = Linear(intermediate_size, hidden_size)

        def forward(self, state: nn.Tensor):
            state = self.embedding(state)
            state = self.up(state)
            state = nn.op.silu(state)
            assert state.dtype == "float32"
            state = self.down(state)
            return state

    if dynamic_type == "same_python_string":
        # Python strings have value equality.  Providing the same name
        # for two different shape parameters results in a single
        # symbolic variable.
        args = ["hidden_size", "hidden_size"]
        expected_num_symbolic_vars = 1
    elif dynamic_type == "different_python_string":
        # Providing two distinct variable names for the two different
        # shape parameters results in two distinct symbolic variables.
        args = ["hidden_size", "intermediate_size"]
        expected_num_symbolic_vars = 2
    elif dynamic_type == "same_tir_var":
        # Symbolic variables can be specified as tirx.Var instances.
        # Providing the same variable for the two different shape
        # parameters uses the symbolic variable in both locations.
        dim = tirx.Var("hidden_size", "int64")
        args = [dim, dim]
        expected_num_symbolic_vars = 1
    elif dynamic_type == "distinct_tir_vars_with_distinct_names":
        # Providing distinct TIR variables for the two different shape
        # parameters uses each TIR variable in the specified location.
        args = [tirx.Var("hidden_size", "int64"), tirx.Var("intermediate_size", "int64")]
        expected_num_symbolic_vars = 2
    elif dynamic_type == "distinct_tir_vars_with_same_name":
        # TIR variable have reference equality.  Even if two different
        # TIR variables have the same name, providing two distinct TIR
        # variables still results in two distinct symbolic variables.
        args = [tirx.Var("hidden_size", "int64"), tirx.Var("hidden_size", "int64")]
        expected_num_symbolic_vars = 2
    else:
        raise ValueError(f"Unexpected dynamic_type: {dynamic_type}")

    slm_mod = Model(*args)

    exported_mod, _ = slm_mod.export_tvm(
        spec={
            "forward": {"state": nn.spec.Tensor(["batch_size", 1024], dtype="float32")},
        },
        debug=False,
    )

    def get_expected_with_intermediate_size():
        batch_size = T.dynamic("batch_size", "int64")
        hidden_size = T.dynamic("hidden_size", "int64")
        intermediate_size = T.dynamic("intermediate_size", "int64")

        @I.ir_module
        class Expected:
            @R.function
            def forward(
                state: R.Tensor([batch_size, 1024], "float32"),
                embedding_weights: R.Tensor([hidden_size, 1024], "float32"),
                up_weights: R.Tensor([intermediate_size, hidden_size], "float32"),
                down_weights: R.Tensor([hidden_size, intermediate_size], "float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    state: R.Tensor([batch_size, hidden_size], "float32") = R.matmul(
                        state, R.permute_dims(embedding_weights)
                    )
                    state: R.Tensor([batch_size, intermediate_size], "float32") = R.matmul(
                        state, R.permute_dims(up_weights)
                    )
                    state: R.Tensor([batch_size, intermediate_size], "float32") = R.nn.silu(state)
                    state: R.Tensor([batch_size, hidden_size], "float32") = R.matmul(
                        state, R.permute_dims(down_weights)
                    )
                    state = state
                    R.output(state)
                return state

        return Expected

    def get_expected_without_intermediate_size():
        batch_size = T.dynamic("batch_size", "int64")
        hidden_size = T.dynamic("hidden_size", "int64")

        @I.ir_module
        class Expected:
            @R.function
            def forward(
                state: R.Tensor([batch_size, 1024], "float32"),
                embedding_weights: R.Tensor([hidden_size, 1024], "float32"),
                up_weights: R.Tensor([hidden_size, hidden_size], "float32"),
                down_weights: R.Tensor([hidden_size, hidden_size], "float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    state: R.Tensor([batch_size, hidden_size], "float32") = R.matmul(
                        state, R.permute_dims(embedding_weights)
                    )
                    state: R.Tensor([batch_size, hidden_size], "float32") = R.matmul(
                        state, R.permute_dims(up_weights)
                    )
                    state: R.Tensor([batch_size, hidden_size], "float32") = R.nn.silu(state)
                    state: R.Tensor([batch_size, hidden_size], "float32") = R.matmul(
                        state, R.permute_dims(down_weights)
                    )
                    state = state
                    R.output(state)
                return state

        return Expected

    if expected_num_symbolic_vars == 1:
        expected = get_expected_without_intermediate_size()
    elif expected_num_symbolic_vars == 2:
        expected = get_expected_with_intermediate_size()
    else:
        raise ValueError(f"Unexpected number of symbolic vars: {expected_num_symbolic_vars}")

    assert_structural_equal(exported_mod["forward"], expected["forward"], True)


if __name__ == "__main__":
    tvm_tirx.testing.main()
