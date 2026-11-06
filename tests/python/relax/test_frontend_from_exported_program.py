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
# ruff: noqa: F401, F841
import operator

import numpy as np
import pytest
import torch
from torch import nn
from torch.export import export
from torch.nn import Module

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.frontend.torch import from_exported_program
from tvm.testing import device_enabled
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def verify_model(
    torch_model,
    example_args,
    binding,
    expected,
    dynamic_shapes=None,
    run_ep_decomposition=True,
    keep_params_as_input=False,
    unwrap_unit_return_tuple=False,
    no_bind_return_tuple=False,
    map_free_vars=False,
    custom_convert_map=None,
):
    exported_program = export(torch_model, args=example_args, dynamic_shapes=dynamic_shapes)
    mod = from_exported_program(
        exported_program,
        run_ep_decomposition=run_ep_decomposition,
        keep_params_as_input=keep_params_as_input,
        unwrap_unit_return_tuple=unwrap_unit_return_tuple,
        no_bind_return_tuple=no_bind_return_tuple,
        custom_convert_map=custom_convert_map,
    )

    binding = {k: tvm_tirx.runtime.tensor(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm_tirx.ir.assert_structural_equal(mod, expected, map_free_vars=map_free_vars)


def verify_model_numerically(torch_model, example_args, rtol=1e-7, atol=1e-7, dynamic_shapes=None):
    """Verify model by comparing numerical outputs between PyTorch and TVM."""
    with torch.no_grad():
        pytorch_output = torch_model(*example_args)

    exported_program = export(torch_model, args=example_args, dynamic_shapes=dynamic_shapes)
    mod = from_exported_program(exported_program)
    target = tvm_tirx.target.Target("llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    tvm_args = [tvm_tirx.runtime.tensor(arg.numpy()) for arg in example_args]
    tvm_output = vm["main"](*tvm_args)

    if hasattr(tvm_output, "numpy"):
        tvm_output_np = tvm_output.numpy()
    else:
        tvm_output_np = tvm_output[0].numpy()

    pytorch_output_np = (
        pytorch_output.numpy()
        if isinstance(pytorch_output, torch.Tensor)
        else pytorch_output[0].numpy()
    )

    assert pytorch_output_np.shape == tvm_output_np.shape, (
        f"Shape mismatch: PyTorch {pytorch_output_np.shape} vs TVM {tvm_output_np.shape}"
    )
    tvm_tirx.testing.assert_allclose(pytorch_output_np, tvm_output_np, rtol=rtol, atol=atol)


operator_basic_unary = [
    (torch.abs, R.abs),
    (torch.acos, R.acos),
    (torch.acosh, R.acosh),
    (torch.asin, R.asin),
    (torch.asinh, R.asinh),
    (torch.atan, R.atan),
    (torch.atanh, R.atanh),
    (torch.bitwise_not, R.bitwise_not),
    (torch.ceil, R.ceil),
    (torch.cos, R.cos),
    (torch.cosh, R.cosh),
    (torch.erf, R.erf),
    (torch.exp, R.exp),
    (torch.floor, R.floor),
    (torch.ops.aten.gelu, R.nn.gelu),
    (torch.log, R.log),
    (torch.neg, R.negative),
    (torch.relu, R.nn.relu),
    (torch.round, R.round),
    (torch.rsqrt, R.rsqrt),
    (torch.sigmoid, R.sigmoid),
    (torch.sin, R.sin),
    (torch.sinh, R.sinh),
    (torch.sign, R.sign),
    (torch.sqrt, R.sqrt),
    (torch.tan, R.tan),
    (torch.tanh, R.tanh),
    (torch.trunc, R.trunc),
]


@pytest.mark.parametrize("pytorch_op, relax_op", operator_basic_unary)
def test_basic_unary_ops(pytorch_op, relax_op):
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    class UnaryOp(Module):
        def forward(self, input):
            return pytorch_op(input)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = relax_op(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(UnaryOp(), example_args, {}, expected)


def test_round():
    class Round(Module):
        def __init__(self, decimals):
            super().__init__()
            self.decimals = decimals

        def forward(self, input):
            return torch.round(input, decimals=self.decimals)

    # Ties-to-even, negative decimals, and scales that overflow to NaN.
    x = torch.tensor(
        [0.5, 1.5, 2.5, 4.5, -0.5, -2.5, 25.0, 125.0, 165.0, 2.25, 0.0],
        dtype=torch.float32,
    )
    for decimals in (0, 1, -1, -2, 309, -309):
        verify_model_numerically(Round(decimals).eval(), (x,), rtol=1e-6, atol=1e-6)


operator_bool_unary = [
    (torch.isinf, R.isinf),
    (torch.isnan, R.isnan),
]


@pytest.mark.parametrize("pytorch_op, relax_op", operator_bool_unary)
def test_bool_unary_ops(pytorch_op, relax_op):
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    class UnaryOp(Module):
        def forward(self, input):
            return pytorch_op(input)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="bool")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = relax_op(input_1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    verify_model(UnaryOp(), example_args, {}, expected)


def test_sqrt_integer_input():
    """Test that sqrt operation works with integer tensors by auto-converting to float."""
    example_args = (torch.tensor([[4, 9, 16, 25]], dtype=torch.int64),)

    class SqrtIntModel(Module):
        def forward(self, input):
            return torch.sqrt(input)

    @tvm_tirx.script.ir_module
    class expected_int64:
        @R.function
        def main(input_1: R.Tensor((1, 4), dtype="int64")) -> R.Tuple(
            R.Tensor((1, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.astype(input_1, dtype="float32")
                lv1: R.Tensor((1, 4), dtype="float32") = R.sqrt(lv)
                gv: R.Tuple(R.Tensor((1, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(SqrtIntModel(), example_args, {}, expected_int64)

    example_args_int32 = (torch.tensor([[1, 4, 9]], dtype=torch.int32),)

    @tvm_tirx.script.ir_module
    class expected_int32:
        @R.function
        def main(input_1: R.Tensor((1, 3), dtype="int32")) -> R.Tuple(
            R.Tensor((1, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3), dtype="float32") = R.astype(input_1, dtype="float32")
                lv1: R.Tensor((1, 3), dtype="float32") = R.sqrt(lv)
                gv: R.Tuple(R.Tensor((1, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(SqrtIntModel(), example_args_int32, {}, expected_int32)


def test_extended_unary_ops():
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    # celu
    class Celu(Module):
        def forward(self, input):
            return torch.nn.functional.celu(input)

    # alpha * min(0, exp(x / alpha) - 1) + max(0, x)
    @tvm_tirx.script.ir_module
    class expected_celu:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(input)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(
                    lv, R.const(1.0, "float32")
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater(
                    input, R.const(0.0, "float32")
                )
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv2, input, lv1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    verify_model(Celu(), example_args, {}, expected_celu)

    # clamp
    class Clamp(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.1, max=0.5)

    @tvm_tirx.script.ir_module
    class expected_clamp:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    input,
                    R.prim_value(T.float64(0.10000000000000001)),
                    R.prim_value(T.float64(0.5)),
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Clamp(), example_args, {}, expected_clamp)

    class ClampMinOnly(Module):
        def forward(self, input):
            return torch.clamp(input, min=0.5, max=None)

    @tvm_tirx.script.ir_module
    class expected_clamp_min_only:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    input, R.prim_value(T.float64(0.5)), R.prim_value(T.float64("inf"))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(ClampMinOnly(), example_args, {}, expected_clamp_min_only)

    class ClampTensors(Module):
        def forward(self, input):
            return torch.clamp(input, min=input, max=input)

    @tvm_tirx.script.ir_module
    class expected_clamp_tensors:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.broadcast_to(
                    input, R.shape([1, 3, 10, 10])
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.maximum(input, lv)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.broadcast_to(
                    input, R.shape([1, 3, 10, 10])
                )
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.minimum(lv1, lv2)
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    lv3, R.prim_value(T.float64("-inf")), R.prim_value(T.float64("inf"))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    verify_model(ClampTensors(), example_args, {}, expected_clamp_tensors)

    # dropout

    class Dropout(Module):
        def forward(self, input):
            return torch.dropout(input, 0.5, train=True)

    class Dropout3(Module):
        def forward(self, input):
            return torch.ops.aten.dropout_(input, 0.5, train=True)

    @tvm_tirx.script.ir_module
    class expected_dropout_for_1_2:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (input,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_dropout_for_3:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.zeros(
                    R.shape([1, 3, 10, 10]), dtype="float32"
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv, R.const(0.5, "float32")
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(input, lv1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    verify_model(Dropout(), example_args, {}, expected_dropout_for_1_2)
    verify_model(Dropout3(), example_args, {}, expected_dropout_for_3)

    # elu
    class Elu(Module):
        def forward(self, input):
            return torch.nn.functional.elu(input)

    @tvm_tirx.script.ir_module
    class expected_elu:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(input)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(
                    R.const(1.0, "float32"), lv
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    R.const(-1.0, "float32"), lv2
                )
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(input)
                lv5: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv3, lv4)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    verify_model(Elu(), example_args, {}, expected_elu)

    # hardsigmoid
    class Hardsigmoid(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardsigmoid(input)

    @tvm_tirx.script.ir_module
    class expected_hardsigmoid:
        @R.function
        def main(inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(
                    inp_0, R.const(3.0, "float32")
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    lv, R.prim_value(0), R.prim_value(T.float64("inf"))
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    lv1, R.prim_value(T.float64("-inf")), R.prim_value(6)
                )
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv2, R.const(6.0, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    verify_model(Hardsigmoid(), example_args, {}, expected_hardsigmoid)

    # hardwish
    class Hardswish(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardswish(input)

    class Hardswish3(torch.nn.Module):
        def forward(self, input):
            return torch.ops.aten.hardswish_(input)

    @tvm_tirx.script.ir_module
    class expected_hardswish:
        @R.function
        def main(inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(
                    inp_0, R.const(3.0, "float32")
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    lv, R.prim_value(0), R.prim_value(T.float64("inf"))
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    lv1, R.prim_value(T.float64("-inf")), R.prim_value(6)
                )
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(inp_0, lv2)
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv3, R.const(6.0, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    verify_model(Hardswish(), example_args, {}, expected_hardswish)
    verify_model(Hardswish3(), example_args, {}, expected_hardswish)

    # isfinite
    class IsFinite(Module):
        def forward(self, input):
            return torch.isfinite(input)

    @tvm_tirx.script.ir_module
    class expected_isfinite:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="bool")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.abs(input)
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.not_equal(
                    lv, R.const(float("inf"), "float32")
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="bool") = R.equal(input, input)
                lv3: R.Tensor((1, 3, 10, 10), dtype="bool") = R.multiply(lv2, lv1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv3,)
                R.output(gv)
            return gv

    verify_model(IsFinite(), example_args, {}, expected_isfinite)

    # log2
    class Log2(Module):
        def forward(self, x):
            return torch.log2(x)

    @tvm_tirx.script.ir_module
    class Expected_log2:
        @R.function
        def main(inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(inp_0)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv, R.const(0.69314718246459961, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(Log2(), example_args, {}, Expected_log2)

    # log10
    class Log10(Module):
        def forward(self, x):
            return torch.log10(x)

    @tvm_tirx.script.ir_module
    class Expected_log10:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(inp_0)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv, R.const(2.302585092994046, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(Log10(), example_args, {}, Expected_log10)

    # log1p
    class Log1p(Module):
        def forward(self, x):
            return torch.log1p(x)

    @tvm_tirx.script.ir_module
    class Expected_log1p:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(
                    R.add(inp_0, R.const(1, "float32"))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Log1p(), example_args, {}, Expected_log1p)

    # reciprocal
    class Reciprocal(Module):
        def forward(self, input):
            return torch.reciprocal(input)

    @tvm_tirx.script.ir_module
    class expected_reciprocal:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    R.const(1.0, "float32"), input_1
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Reciprocal(), example_args, {}, expected_reciprocal)

    # Returns the maximum value of all elements in the input tensor.
    class MaxModel(Module):
        def forward(self, input):
            return torch.max(input)

    @tvm_tirx.script.ir_module
    class expected_max:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.max(input, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(MaxModel(), example_args, {}, expected_max)

    # Returns the minimum value of all elements in the input tensor.
    class MinModel(Module):
        def forward(self, input):
            return torch.min(input)

    @tvm_tirx.script.ir_module
    class expected_min:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.min(input, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(MinModel(), example_args, {}, expected_min)

    # relu6
    class ReLU6_1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu6 = torch.nn.ReLU6()

        def forward(self, x):
            return self.relu6(x)

    class ReLU6_2(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.relu6(x)

    class ReLU6_3(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.relu6_(x)

    @tvm_tirx.script.ir_module
    class expected_relu6_1:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    x, R.prim_value(T.float64(0.0)), R.prim_value(T.float64(6.0))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_relu6_2:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu6(x)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_relu6_3:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    x, R.prim_value(0), R.prim_value(6)
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(ReLU6_1(), example_args, {}, expected_relu6_1)
    verify_model(ReLU6_2(), example_args, {}, expected_relu6_2)
    verify_model(ReLU6_3(), example_args, {}, expected_relu6_3)

    # selu
    class SELU(Module):
        def forward(self, input):
            return torch.nn.functional.selu(input)

    @tvm_tirx.script.ir_module
    class expected_selu:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(input)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(
                    R.const(1.0, "float32"), lv
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    R.const(-1.6732631921768188, "float32"), lv2
                )
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(input)
                lv5: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv3, lv4)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    verify_model(SELU(), example_args, {}, expected_selu)

    # silu
    class SiLU(Module):
        def forward(self, input):
            return torch.nn.functional.silu(input)

    @tvm_tirx.script.ir_module
    class expected_silu:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sigmoid(input)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(input, lv)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(SiLU(), example_args, {}, expected_silu)

    # silu_
    class SiLU_(Module):
        def forward(self, input):
            return torch.ops.aten.silu_(input)

    verify_model(SiLU_(), example_args, {}, expected_silu)

    # square
    class Square(Module):
        def forward(self, input):
            return torch.square(input)

    @tvm_tirx.script.ir_module
    class expected_square:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.power(
                    input, R.const(2.0, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Square(), example_args, {}, expected_square)

    # relu_
    class ReLU_(Module):
        def forward(self, input):
            return torch.relu_(input.clone())

    @tvm_tirx.script.ir_module
    class expected_relu_:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.relu(input)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(ReLU_(), example_args, {}, expected_relu_)


def test_hardtanh():
    class Hardtanh(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.hardtanh(input)

    class Hardtanh3(torch.nn.Module):
        def forward(self, input):
            return torch.ops.aten.hardtanh_(input)

    @tvm_tirx.script.ir_module
    class expected_for_1_2:
        @R.function
        def main(inp_0: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.clip(
                    inp_0, R.prim_value(T.float64(-1.0)), R.prim_value(T.float64(1.0))
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Hardtanh(), example_args, {}, expected_for_1_2)
    # In-place hardtanh_ yields the same program; mutation outputs are dropped.
    verify_model(Hardtanh3(), example_args, {}, expected_for_1_2)


def test_softplus():
    class Softplus(Module):
        def forward(self, input):
            return torch.nn.functional.softplus(input, 1.0, 20.0)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    x, R.const(1.0, "float32")
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(lv)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv1, R.const(1.0, "float32"))
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(lv2)
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.divide(
                    lv3, R.const(1.0, "float32")
                )
                lv5: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater(
                    lv, R.const(20.0, "float32")
                )
                lv6: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv5, x, lv4)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv6,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softplus(), example_args, {}, expected)


def test_leakyrelu():
    class LeakyReLU(Module):
        def forward(self, input):
            return torch.nn.functional.leaky_relu(input, 0.02)

    class LeakyReLU2(Module):
        def forward(self, input):
            return torch.ops.aten.leaky_relu_(input, 0.02)

    @tvm_tirx.script.ir_module
    class expected_for_1_2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.leakyrelu(input_1, alpha=0.02)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(LeakyReLU(), example_args, {}, expected_for_1_2)
    # In-place leaky_relu_ yields the same program; mutation outputs are dropped.
    verify_model(LeakyReLU2(), example_args, {}, expected_for_1_2)


def test_logaddexp():
    class LogAddExp(Module):
        def forward(self, input1, input2):
            return torch.logaddexp(input1, input2)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            input1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            input2: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater_equal(input1, input2)
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv, input1, input2)
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv, input2, input1)
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.abs(input1)
                lv4: R.Tensor((1, 3, 10, 10), dtype="bool") = R.not_equal(
                    lv3, R.const(float("inf"), "float32")
                )
                lv5: R.Tensor((1, 3, 10, 10), dtype="bool") = R.equal(input1, input1)
                lv6: R.Tensor((1, 3, 10, 10), dtype="bool") = R.multiply(lv5, lv4)
                lv7: R.Tensor((1, 3, 10, 10), dtype="bool") = R.logical_not(lv6)
                lv8: R.Tensor((1, 3, 10, 10), dtype="bool") = R.equal(input1, input2)
                lv9: R.Tensor((1, 3, 10, 10), dtype="bool") = R.logical_and(lv7, lv8)
                lv10: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(lv2, lv1)
                lv11: R.Tensor((1, 3, 10, 10), dtype="float32") = R.exp(lv10)
                lv12: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(
                    lv11, R.const(1.0, "float32")
                )
                lv13: R.Tensor((1, 3, 10, 10), dtype="float32") = R.log(lv12)
                lv14: R.Tensor((1, 3, 10, 10), dtype="float32") = R.add(lv1, lv13)
                lv15: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv9, input1, lv14)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv15,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    verify_model(LogAddExp(), example_args, {}, expected)


def test_atan2():
    class Atan2(Module):
        def forward(self, lhs, rhs):
            return torch.atan2(lhs, rhs)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.atan2(lhs, rhs)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    verify_model(Atan2(), example_args, {}, expected)


def test_logical_and():
    class LogicalAnd(Module):
        def forward(self, lhs, rhs):
            return torch.logical_and(lhs, rhs)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(lhs, dtype="bool")
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(rhs, dtype="bool")
                lv2: R.Tensor((1, 3, 10, 10), dtype="bool") = R.logical_and(lv, lv1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    verify_model(LogicalAnd(), example_args, {}, expected)


def test_logical_not():
    class LogicalNot(Module):
        def forward(self, input):
            return torch.logical_not(input)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="bool")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(input, dtype="bool")
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.logical_not(lv)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(LogicalNot(), example_args, {}, expected)


def test_logical_or():
    class LogicalOr(Module):
        def forward(self, lhs, rhs):
            return torch.logical_or(lhs, rhs)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(lhs, dtype="bool")
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(rhs, dtype="bool")
                lv2: R.Tensor((1, 3, 10, 10), dtype="bool") = R.logical_or(lv, lv1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    verify_model(LogicalOr(), example_args, {}, expected)


def test_logical_xor():
    class LogicalXor(Module):
        def forward(self, lhs, rhs):
            return torch.logical_xor(lhs, rhs)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
            rhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(lhs, dtype="bool")
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.astype(rhs, dtype="bool")
                lv2: R.Tensor((1, 3, 10, 10), dtype="bool") = R.logical_xor(lv, lv1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="bool")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
        torch.randn(1, 3, 10, 10, dtype=torch.float32),
    )
    verify_model(LogicalXor(), example_args, {}, expected)


def test_pow_integer():
    class Pow(Module):
        def forward(self, input):
            return input.pow(4)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(input: R.Tensor((4,), dtype="int64")) -> R.Tuple(R.Tensor((4,), dtype="int64")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int64") = R.multiply(input, input)
                lv1: R.Tensor((4,), dtype="int64") = R.multiply(lv, input)
                lv2: R.Tensor((4,), dtype="int64") = R.multiply(lv1, input)
                gv: R.Tuple(R.Tensor((4,), dtype="int64")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.tensor([-1, 1, 2, 3], dtype=torch.int64),)
    verify_model(Pow(), example_args, {}, expected)


def test_logsoftmax():
    class LogSoftmax(Module):
        def forward(self, input):
            return torch.nn.functional.log_softmax(input, dim=1)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.log_softmax(input_1, axis=1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(LogSoftmax(), example_args, {}, expected1)


def test_prelu():
    class Prelu(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = torch.nn.Parameter(torch.tensor([0.25]))

        def forward(self, x):
            return torch.nn.functional.prelu(x, self.alpha)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 1, 1, 1), dtype="float32") = R.reshape(
                    R.const([0.25], dtype="float32"), R.shape([1, 1, 1, 1])
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater(x, R.const(0.0, "float32"))
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(lv, x)
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv1, x, lv2)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Prelu(), example_args, {}, expected)


def test_softmax():
    class Softmax(Module):
        def forward(self, input):
            return torch.nn.functional.softmax(input, dim=1)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.softmax(input_1, axis=1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softmax(), example_args, {}, expected1)


def test_softsign():
    class Softsign(Module):
        def forward(self, input):
            return torch.nn.functional.softsign(input)

    @tvm_tirx.script.ir_module
    class expected_softsign:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            with R.dataflow():
                abs_val = R.abs(input)
                denom = R.add(abs_val, R.const(1.0, "float32"))
                result = R.divide(input, denom)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (result,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softsign(), example_args, {}, expected_softsign)


def test_softshrink():
    class Softshrink(Module):
        def forward(self, input):
            return torch.nn.functional.softshrink(input, lambd=0.5)

    @tvm_tirx.script.ir_module
    class expected_softshrink:
        @R.function
        def main(
            input: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.abs(input)
                lv1: R.Tensor((1, 3, 10, 10), dtype="bool") = R.greater(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.sign(input)
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    lv2, R.const(0.5, "float32")
                )
                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.subtract(input, lv3)
                lv5: R.Tensor((1, 3, 10, 10), dtype="float32") = R.multiply(
                    input, R.const(0.0, "float32")
                )
                lv6: R.Tensor((1, 3, 10, 10), dtype="float32") = R.where(lv1, lv4, lv5)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv6,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Softshrink(), example_args, {}, expected_softshrink)


def test_tril_triu():
    example_args = (torch.randn(10, 10, dtype=torch.float32),)

    class Tril(Module):
        def forward(self, input):
            return torch.tril(input, 1)

    @tvm_tirx.script.ir_module
    class expected_tril:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(10), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((1, 10), dtype="int64") = R.expand_dims(lv, axis=[-2])
                lv2: R.Tensor((10,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(10), R.prim_value(1), dtype="int64"
                )
                lv3: R.Tensor((10, 1), dtype="int64") = R.expand_dims(lv2, axis=[-1])
                lv4: R.Tensor((10, 10), dtype="int64") = R.subtract(lv1, lv3)
                lv5: R.Tensor((10, 10), dtype="bool") = R.less_equal(lv4, R.const(1, "int64"))
                lv6: R.Tensor((), dtype="float32") = R.const(0.0, "float32")
                lv7: R.Tensor((10, 10), dtype="float32") = R.where(lv5, input, lv6)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv7,)
                R.output(gv)
            return gv

    verify_model(Tril(), example_args, {}, expected_tril)

    class Triu(Module):
        def forward(self, input):
            return torch.triu(input, 1)

    @tvm_tirx.script.ir_module
    class expected_triu:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(10), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((1, 10), dtype="int64") = R.expand_dims(lv, axis=[-2])
                lv2: R.Tensor((10,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(10), R.prim_value(1), dtype="int64"
                )
                lv3: R.Tensor((10, 1), dtype="int64") = R.expand_dims(lv2, axis=[-1])
                lv4: R.Tensor((10, 10), dtype="int64") = R.subtract(lv1, lv3)
                lv5: R.Tensor((10, 10), dtype="bool") = R.greater_equal(lv4, R.const(1, "int64"))
                lv6: R.Tensor((), dtype="float32") = R.const(0.0, "float32")
                lv7: R.Tensor((10, 10), dtype="float32") = R.where(lv5, input, lv6)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv7,)
                R.output(gv)
            return gv

    verify_model(Triu(), example_args, {}, expected_triu)


operator_binary_1 = [
    (operator.add, R.add),
    (torch.ops.aten.add_, R.add),
    (torch.ops.aten.bitwise_or_, R.bitwise_or),
    (operator.sub, R.subtract),
    (operator.mul, R.multiply),
    (torch.ops.aten.mul_, R.multiply),
    (operator.truediv, R.divide),
    (operator.floordiv, R.floor_divide),
    (torch.ops.aten.fmod, R.mod),
    (operator.pow, R.power),
    (operator.mod, R.floor_mod),
    (operator.and_, R.bitwise_and),
    (operator.or_, R.bitwise_or),
    (operator.xor, R.bitwise_xor),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_1)
def test_binary1(op, relax_op):
    example_args1 = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )
    example_args2 = (torch.randn(10, 10, dtype=torch.float32),)

    class Binary1(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs, rhs):
            return self.op(lhs, rhs)

    @tvm_tirx.script.ir_module
    class expected_binary1:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype="float32"),
            rhs: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = relax_op(lhs, rhs)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Binary2(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs):
            return self.op(lhs, 1.0)

    @tvm_tirx.script.ir_module
    class expected_binary2:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = relax_op(lhs, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # In-place ops (add_, mul_, ...) produce the same Relax program as their
    # functional counterparts: mutation outputs are dropped by the importer.
    verify_model(Binary1(op), example_args1, {}, expected_binary1)
    verify_model(Binary2(op), example_args2, {}, expected_binary2)


operator_binary_scalar = [
    (torch.ops.aten.add.Scalar, R.add),
    (torch.ops.aten.bitwise_and.Scalar, R.bitwise_and),
    (torch.ops.aten.bitwise_or.Scalar, R.bitwise_or),
    (torch.ops.aten.bitwise_xor.Scalar, R.bitwise_xor),
    (torch.ops.aten.div.Scalar, R.divide),
    (torch.ops.aten.sub.Scalar, R.subtract),
    (torch.ops.aten.mul.Scalar, R.multiply),
    (torch.ops.aten.remainder.Scalar, R.floor_mod),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_scalar)
def test_binary_scalar(op, relax_op):
    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    class BinaryScalar(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs):
            return self.op(lhs, 1.0)

    @tvm_tirx.script.ir_module
    class expected_binary_scalar:
        @R.function
        def main(
            lhs: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = relax_op(lhs, R.const(1.0))
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(BinaryScalar(op), example_args, {}, expected_binary_scalar)


operator_binary_promote = [
    (operator.add, R.add),
    (operator.sub, R.subtract),
    (operator.mul, R.multiply),
    (operator.truediv, R.divide),
    (operator.pow, R.power),
    (operator.mod, R.floor_mod),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_promote)
def test_binary_dtype_promotion(op, relax_op):
    """Ensure binary ops promote differing dtypes following PyTorch rules."""

    class BinaryPromoteLHS(Module):
        def forward(self, x):
            arange_val = torch.arange(x.shape[1])  # int64 by default
            return op(x, arange_val)

    @tvm_tirx.script.ir_module
    class expected_promote_lhs:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(3), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((3,), dtype="float32") = R.astype(lv, dtype="float32")
                lv2: R.Tensor((2, 3), dtype="float32") = relax_op(x, lv1)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    class BinaryPromoteRHS(Module):
        def forward(self, x):
            arange_val = torch.arange(x.shape[1])  # int64 by default
            return op(arange_val, x)

    @tvm_tirx.script.ir_module
    class expected_promote_rhs:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(3), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((3,), dtype="float32") = R.astype(lv, dtype="float32")
                lv2: R.Tensor((2, 3), dtype="float32") = relax_op(lv1, x)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32),)
    verify_model(BinaryPromoteLHS(), example_args, {}, expected_promote_lhs)
    verify_model(BinaryPromoteRHS(), example_args, {}, expected_promote_rhs)


operator_binary_2 = [
    (operator.eq, R.equal),
    (operator.ne, R.not_equal),
    (operator.lt, R.less),
    (operator.le, R.less_equal),
    (operator.gt, R.greater),
    (operator.ge, R.greater_equal),
]


@pytest.mark.parametrize("op, relax_op", operator_binary_2)
def test_binary2(op, relax_op):
    example_args1 = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )
    example_args2 = (torch.randn(10, 10, dtype=torch.float32),)

    class Binary1(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs, rhs):
            return self.op(lhs, rhs)

    @tvm_tirx.script.ir_module
    class expected_binary1:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype="float32"),
            rhs: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="bool") = relax_op(lhs, rhs)
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    class Binary2(Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, lhs):
            return self.op(lhs, 1.0)

    @tvm_tirx.script.ir_module
    class expected_binary2:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="bool") = relax_op(lhs, R.const(1.0))
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Binary1(op), example_args1, {}, expected_binary1)
    verify_model(Binary2(op), example_args2, {}, expected_binary2)


def test_binary3():
    example_args1 = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )
    example_args2 = (torch.randn(10, 10, dtype=torch.float32),)

    # Max
    class Max1(Module):
        def forward(self, x, y):
            return torch.max(x, y)

    @I.ir_module
    class expected_max1:
        @R.function
        def main(
            inp_0: R.Tensor((10, 10), dtype="float32"),
            inp_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.maximum(inp_0, inp_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Max1(), example_args1, {}, expected_max1)

    # Min
    class Min1(Module):
        def forward(self, x, y):
            return torch.min(x, y)

    @I.ir_module
    class expected_min1:
        @R.function
        def main(
            inp_0: R.Tensor((10, 10), dtype="float32"),
            inp_1: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.minimum(inp_0, inp_1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Min1(), example_args1, {}, expected_min1)

    # RSub
    class RSub1(Module):
        def forward(self, x, y):
            return torch.rsub(x, y)

    class RSub2(Module):
        def forward(self, x):
            return torch.rsub(x, 5.0)

    @tvm_tirx.script.ir_module
    class expected_rsub1:
        @R.function
        def main(
            x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.subtract(y, x)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_rsub2:
        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.subtract(R.const(5.0, "float32"), x)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(RSub1(), example_args1, {}, expected_rsub1)
    verify_model(RSub2(), example_args2, {}, expected_rsub2)


# IsIn


def test_isin():
    class IsInModel(torch.nn.Module):
        def forward(self, x, test_elements):
            return torch.isin(x, test_elements)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            x: R.Tensor((10, 10), dtype="float32"), test_elements: R.Tensor((8,), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((10, 10, 1), dtype="float32") = R.reshape(x, R.shape([10, 10, 1]))
                lv1: R.Tensor((10, 10, 8), dtype="bool") = R.equal(lv, test_elements)
                lv2: R.Tensor((10, 10, 8), dtype="int8") = R.astype(lv1, dtype="int8")
                lv3: R.Tensor((10, 10), dtype="int8") = R.max(lv2, axis=[-1], keepdims=False)
                lv4: R.Tensor((10, 10), dtype="bool") = R.astype(lv3, dtype="bool")
                gv: R.Tuple(R.Tensor((10, 10), dtype="bool")) = (lv4,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(8, dtype=torch.float32),
    )
    verify_model(IsInModel(), example_args, {}, expected)


def test_div_mode():
    # Case 1: Basic division (no rounding mode)
    class DivModel(torch.nn.Module):
        def forward(self, a, b):
            return torch.div(a, b)

    @tvm_tirx.script.ir_module
    class expected_div:
        @R.function
        def main(
            a: R.Tensor((64, 64), dtype="float32"), b: R.Tensor((64,), dtype="float32")
        ) -> R.Tuple(R.Tensor((64, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.divide(a, b)
                gv: R.Tuple(R.Tensor((64, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(64, 64, dtype=torch.float32),
        torch.randn(64, dtype=torch.float32),
    )
    verify_model(DivModel(), example_args, {}, expected_div)

    # Case 2: Division with trunc rounding
    class DivTruncModel(torch.nn.Module):
        def forward(self, a, b):
            return torch.div(a, b, rounding_mode="trunc")

    @tvm_tirx.script.ir_module
    class expected_div_trunc:
        @R.function
        def main(
            a: R.Tensor((64, 64), dtype="float32"), b: R.Tensor((64,), dtype="float32")
        ) -> R.Tuple(R.Tensor((64, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.divide(a, b)
                lv1: R.Tensor((64, 64), dtype="float32") = R.trunc(lv)
                gv: R.Tuple(R.Tensor((64, 64), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(DivTruncModel(), example_args, {}, expected_div_trunc)

    # Case 3: Division with floor rounding
    class DivFloorModel(torch.nn.Module):
        def forward(self, a, b):
            return torch.div(a, b, rounding_mode="floor")

    @tvm_tirx.script.ir_module
    class expected_div_floor:
        @R.function
        def main(
            a: R.Tensor((64, 64), dtype="float32"), b: R.Tensor((64,), dtype="float32")
        ) -> R.Tuple(R.Tensor((64, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.floor_divide(a, b)
                gv: R.Tuple(R.Tensor((64, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(DivFloorModel(), example_args, {}, expected_div_floor)

    # Case 4: Integer divisor, whose Python type is not the tensor's dtype
    class DivScalarModel(torch.nn.Module):
        def forward(self, a):
            return torch.div(a, 2, rounding_mode="floor")

    @tvm.script.ir_module
    class expected_div_scalar:
        @R.function
        def main(a: R.Tensor((64, 64), dtype="float32")) -> R.Tuple(
            R.Tensor((64, 64), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((64, 64), dtype="float32") = R.floor_divide(a, R.const(2.0, "float32"))
                gv: R.Tuple(R.Tensor((64, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(
        DivScalarModel(), (torch.randn(64, 64, dtype=torch.float32),), {}, expected_div_scalar
    )


def test_batchnorm2d():
    class BatchNorm2d1(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, input):
            return self.bn(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
            w3: R.Tensor((3,), dtype="float32"),
            w4: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = R.nn.batch_norm(
                    input_1,
                    w1,
                    w2,
                    w3,
                    w4,
                    axis=1,
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                    momentum=0.1,
                    training=False,
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[0]
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    class BatchNorm2dCustom(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3, eps=0.001, momentum=0.01)

        def forward(self, input):
            return self.bn(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
            w3: R.Tensor((3,), dtype="float32"),
            w4: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = R.nn.batch_norm(
                    input_1,
                    w1,
                    w2,
                    w3,
                    w4,
                    axis=1,
                    epsilon=0.001,
                    center=True,
                    scale=True,
                    momentum=0.01,
                    training=False,
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[0]
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model_1 = BatchNorm2d1().eval()
    binding_1 = {
        "w1": model_1.bn.weight.detach().numpy(),
        "w2": model_1.bn.bias.detach().numpy(),
        "w3": model_1.bn.running_mean.detach().numpy(),
        "w4": model_1.bn.running_var.detach().numpy(),
    }
    verify_model(model_1, example_args, binding_1, expected1)

    model_2 = BatchNorm2dCustom().eval()
    binding_2 = {
        "w1": model_2.bn.weight.detach().numpy(),
        "w2": model_2.bn.bias.detach().numpy(),
        "w3": model_2.bn.running_mean.detach().numpy(),
        "w4": model_2.bn.running_var.detach().numpy(),
    }
    verify_model(model_2, example_args, binding_2, expected2)

    class BatchNorm2dTraining(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3, track_running_stats=True)

        def forward(self, input):
            return self.bn(input)

    @tvm_tirx.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((2, 3, 4, 4), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
            w3: R.Tensor((3,), dtype="float32"),
            w4: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 4, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="int64") = R.add(R.const(0, "int64"), R.const(1, "int64"))
                lv1: R.Tuple(
                    R.Tensor((2, 3, 4, 4), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = R.nn.batch_norm(
                    input_1,
                    w1,
                    w2,
                    w3,
                    w4,
                    axis=1,
                    epsilon=1e-5,
                    center=True,
                    scale=True,
                    momentum=0.1,
                    training=True,
                )
                lv2: R.Tensor((2, 3, 4, 4), dtype="float32") = lv1[0]
                lv3: R.Tensor((3,), dtype="float32") = lv1[1]
                lv4: R.Tensor((3,), dtype="float32") = R.zeros(R.shape([3]), dtype="float32")
                lv5: R.Tuple(
                    R.Tensor((2, 3, 4, 4), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = (lv2, lv3, lv4, lv4, lv4)
                lv6: R.Tensor((2, 3, 4, 4), dtype="float32") = lv5[0]
                lv7: R.Tensor((3,), dtype="float32") = lv5[3]
                lv8: R.Tensor((3,), dtype="float32") = lv5[4]
                gv: R.Tuple(R.Tensor((2, 3, 4, 4), dtype="float32")) = (lv6,)
                R.output(gv)
            return gv

    example_args_train = (torch.randn(2, 3, 4, 4, dtype=torch.float32),)

    model_3 = BatchNorm2dTraining()
    model_3.train()  # Set to training mode
    binding_3 = {
        "w1": model_3.bn.weight.detach().numpy(),
        "w2": model_3.bn.bias.detach().numpy(),
        "w3": model_3.bn.running_mean.detach().numpy(),
        "w4": model_3.bn.running_var.detach().numpy(),
    }
    verify_model(model_3, example_args_train, binding_3, expected3)


def test_adaptive_avgpool1d():
    class AdaptiveAvgPool1d(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool1d(input, output_size=5)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 5), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 10), dtype="float32") = R.expand_dims(input_1, axis=[-2])
                lv1: R.Tensor((1, 3, 1, 5), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    lv, output_size=[1, 5], layout="NCHW"
                )
                lv2: R.Tensor((1, 3, 5), dtype="float32") = R.squeeze(lv1, axis=[-2])
                gv: R.Tuple(R.Tensor((1, 3, 5), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool1d(), example_args, {}, expected1)


def test_adaptive_avgpool2d():
    class AdaptiveAvgPool2d(Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool2d(input, [10, 10])

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    input_1, output_size=[10, 10], layout="NCHW", out_layout="NCHW"
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool2d(), example_args, {}, expected1)


def test_adaptive_avgpool3d():
    class AdaptiveAvgPool3d(torch.nn.Module):
        def forward(self, input):
            return torch.nn.functional.adaptive_avg_pool3d(input, [4, 4, 4])

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 4, 4, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = R.nn.adaptive_avg_pool3d(
                    input_1, output_size=[4, 4, 4], layout="NCDHW", out_layout="NCDHW"
                )
                gv: R.Tuple(R.Tensor((1, 3, 4, 4, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 8, 8, 8, dtype=torch.float32),)
    verify_model(AdaptiveAvgPool3d(), example_args, {}, expected1)


def test_addmm():
    class Addmm1(Module):
        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3)

    class Addmm2(Module):
        def forward(self, x1, x2, x3):
            return torch.addmm(x1, x2, x3, beta=0.8, alpha=0.5)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.add(x1, lv)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.multiply(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((10, 10), dtype="float32") = R.multiply(x1, R.const(0.8, "float32"))
                lv3: R.Tensor((10, 10), dtype="float32") = R.add(lv2, lv1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )

    verify_model(Addmm1(), example_args, {}, expected1)
    verify_model(Addmm2(), example_args, {}, expected2)


def test_sparse_addmm():
    class SparseAddmm1(Module):
        def forward(self, x1, x2, x3):
            return torch.sparse.addmm(x1, x2, x3)

    class SparseAddmm2(Module):
        def forward(self, x1, x2, x3):
            return torch.sparse.addmm(x1, x2, x3, beta=0.8, alpha=0.5)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.add(x1, lv)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            x1: R.Tensor((10, 10), dtype="float32"),
            x2: R.Tensor((10, 10), dtype="float32"),
            x3: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.matmul(x2, x3, out_dtype="float32")
                lv1: R.Tensor((10, 10), dtype="float32") = R.multiply(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((10, 10), dtype="float32") = R.multiply(x1, R.const(0.8, "float32"))
                lv3: R.Tensor((10, 10), dtype="float32") = R.add(lv2, lv1)
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
        torch.randn(10, 10, dtype=torch.float32),
    )

    verify_model(SparseAddmm1(), example_args, {}, expected1)
    verify_model(SparseAddmm2(), example_args, {}, expected2)


def test_avg_pool1d():
    class AvgPool1d1(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool1d(kernel_size=1)

        def forward(self, input):
            return self.pool(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input: R.Tensor((1, 3, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 10), dtype="float32") = R.expand_dims(input, axis=[-2])
                lv1: R.Tensor((1, 3, 1, 10), dtype="float32") = R.nn.avg_pool2d(
                    lv,
                    pool_size=[1, 1],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv2: R.Tensor((1, 3, 10), dtype="float32") = R.squeeze(lv1, axis=[-2])
                gv: R.Tuple(R.Tensor((1, 3, 10), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    class AvgPool1d2(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool1d(
                input, kernel_size=3, stride=2, padding=1, ceil_mode=True
            )

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(input: R.Tensor((1, 3, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 6), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 10), dtype="float32") = R.expand_dims(input, axis=[-2])
                lv1: R.Tensor((1, 3, 1, 6), dtype="float32") = R.nn.avg_pool2d(
                    lv,
                    pool_size=[1, 3],
                    strides=[1, 2],
                    dilation=[1, 1],
                    padding=[0, 1, 0, 1],
                    ceil_mode=True,
                    count_include_pad=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv2: R.Tensor((1, 3, 6), dtype="float32") = R.squeeze(lv1, axis=[-2])
                gv: R.Tuple(R.Tensor((1, 3, 6), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    class AvgPool1d4(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool1d(input, kernel_size=2, stride=2, padding=0)

    @tvm_tirx.script.ir_module
    class expected3:
        @R.function
        def main(input: R.Tensor((1, 3, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 5), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 10), dtype="float32") = R.expand_dims(input, axis=[-2])
                lv1: R.Tensor((1, 3, 1, 5), dtype="float32") = R.nn.avg_pool2d(
                    lv,
                    pool_size=[1, 2],
                    strides=[1, 2],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv2: R.Tensor((1, 3, 5), dtype="float32") = R.squeeze(lv1, axis=[-2])
                gv: R.Tuple(R.Tensor((1, 3, 5), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)
    verify_model(AvgPool1d1(), example_args, {}, expected1)
    verify_model(AvgPool1d2(), example_args, {}, expected2)
    verify_model(AvgPool1d4(), example_args, {}, expected3)


def test_avg_pool2d():
    class AvgPool2d1(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.avg_pool2d(
                    input_1,
                    pool_size=[1, 1],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    count_include_pad=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class AvgPool2d2(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool2d(
                input, kernel_size=[4, 4], stride=2, padding=2, ceil_mode=True
            )

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool2d(
                    input_1,
                    pool_size=[4, 4],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[2, 2, 2, 2],
                    ceil_mode=True,
                    count_include_pad=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    class AvgPool2d4(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool2d(input, kernel_size=[2, 1], divisor_override=2)

    @tvm_tirx.script.ir_module
    class expected4:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool2d(
                    input_1,
                    pool_size=[2, 1],
                    strides=[2, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    class AvgPool2d5(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool2d(
                input, kernel_size=[2, 1], divisor_override=2, count_include_pad=False
            )

    @tvm_tirx.script.ir_module
    class expected5:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool2d(
                    input_1,
                    pool_size=[2, 1],
                    strides=[2, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=False,
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(AvgPool2d1(), example_args, {}, expected1)
    verify_model(AvgPool2d2(), example_args, {}, expected2)
    verify_model(AvgPool2d4(), example_args, {}, expected4)
    verify_model(AvgPool2d5(), example_args, {}, expected5)


def test_avg_pool3d():
    class AvgPool3d1(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AvgPool3d(kernel_size=1)

        def forward(self, input):
            return self.pool(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 8, 8, 8), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 8, 8, 8), dtype="float32") = R.nn.avg_pool3d(
                    input_1,
                    pool_size=[1, 1, 1],
                    strides=[1, 1, 1],
                    dilation=[1, 1, 1],
                    padding=[0, 0, 0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv: R.Tuple(R.Tensor((1, 3, 8, 8, 8), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class AvgPool3d2(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool3d(
                input, kernel_size=3, stride=2, padding=1, ceil_mode=True
            )

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool3d(
                    input_1,
                    pool_size=[3, 3, 3],
                    strides=[2, 2, 2],
                    dilation=[1, 1, 1],
                    padding=[1, 1, 1, 1, 1, 1],
                    ceil_mode=True,
                    count_include_pad=True,
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    class AvgPool3d4(Module):
        def forward(self, input):
            return torch.nn.functional.avg_pool3d(input, kernel_size=[2, 1, 2], stride=[2, 1, 2])

    @tvm_tirx.script.ir_module
    class expected3:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")):
            with R.dataflow():
                lv = R.nn.avg_pool3d(
                    input_1,
                    pool_size=[2, 1, 2],
                    strides=[2, 1, 2],
                    dilation=[1, 1, 1],
                    padding=[0, 0, 0, 0, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                gv = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 8, 8, 8, dtype=torch.float32),)
    verify_model(AvgPool3d1(), example_args, {}, expected1)
    verify_model(AvgPool3d2(), example_args, {}, expected2)
    verify_model(AvgPool3d4(), example_args, {}, expected3)


def test_baddbmm():
    class BAddBMM1(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y)

    @tvm_tirx.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((4, 128, 512), dtype="float32"),
            inp_1: R.Tensor((4, 128, 256), dtype="float32"),
            inp_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 128, 512), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(
                    inp_1, inp_2, out_dtype="float32"
                )
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.add(inp_0, lv)
                gv: R.Tuple(R.Tensor((4, 128, 512), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    class BAddBMM2(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=0)

    @tvm_tirx.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((4, 128, 512), dtype="float32"),
            inp_1: R.Tensor((4, 128, 256), dtype="float32"),
            inp_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 128, 512), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(
                    inp_1, inp_2, out_dtype="float32"
                )
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.multiply(
                    lv, R.const(2, "float32")
                )
                gv: R.Tuple(R.Tensor((4, 128, 512), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    class BAddBMM3(Module):
        def forward(self, c, x, y):
            return torch.baddbmm(c, x, y, alpha=2, beta=3)

    @tvm_tirx.script.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((4, 128, 512), dtype="float32"),
            inp_1: R.Tensor((4, 128, 256), dtype="float32"),
            inp_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 128, 512), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(
                    inp_1, inp_2, out_dtype="float32"
                )
                lv1: R.Tensor((4, 128, 512), dtype="float32") = R.multiply(
                    lv, R.const(2, "float32")
                )
                lv2: R.Tensor((4, 128, 512), dtype="float32") = R.multiply(
                    inp_0, R.const(3, "float32")
                )
                lv3: R.Tensor((4, 128, 512), dtype="float32") = R.add(lv2, lv1)
                gv: R.Tuple(R.Tensor((4, 128, 512), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(4, 128, 512, dtype=torch.float32),
        torch.randn(4, 128, 256, dtype=torch.float32),
        torch.randn(4, 256, 512, dtype=torch.float32),
    )
    verify_model(
        BAddBMM1(),
        example_args,
        {},
        Expected1,
        run_ep_decomposition=True,
    )

    verify_model(
        BAddBMM2(),
        example_args,
        {},
        Expected2,
        run_ep_decomposition=True,
    )

    verify_model(
        BAddBMM3(),
        example_args,
        {},
        Expected3,
        run_ep_decomposition=True,
    )


def test_bmm():
    class BMM(Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            input_1: R.Tensor((4, 128, 256), dtype="float32"),
            input_2: R.Tensor((4, 256, 512), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 128, 512), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 128, 512), dtype="float32") = R.matmul(
                    input_1, input_2, out_dtype="float32"
                )
                gv: R.Tuple(R.Tensor((4, 128, 512), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(4, 128, 256, dtype=torch.float32),
        torch.randn(4, 256, 512, dtype=torch.float32),
    )
    verify_model(
        BMM(),
        example_args,
        {},
        Expected,
        run_ep_decomposition=True,
    )


def test_conv_transpose1d():
    class ConvTranspose1d1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(6, 6, 3, bias=True)

        def forward(self, input):
            return self.conv(input)

    class ConvTranspose1d1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 6, 3])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv_transpose1d(input, self.weight, self.bias)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 6, 4), dtype="float32"),
            w1: R.Tensor((6, 6, 3), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 6), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 6), dtype="float32") = R.nn.conv1d_transpose(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    output_padding=[0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="IOW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1)) = R.reshape(w2, [1, 6, 1])
                lv3: R.Tensor((1, 6, 6), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 6), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class ConvTranspose1d2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose1d(6, 6, 3, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 6, 4), dtype="float32"),
            w1: R.Tensor((6, 6, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 6), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 6), dtype="float32") = R.nn.conv1d_transpose(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    output_padding=[0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="IOW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 6, 6), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 6, 4, dtype=torch.float32),)

    model = ConvTranspose1d1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = ConvTranspose1d1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = ConvTranspose1d2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_conv_transpose2d():
    class ConvTranspose2d1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(3, 3, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class ConvTranspose2d1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[3, 3, 7, 7])
            self.bias = torch.randn(size=[3])

        def forward(self, input):
            return torch.nn.functional.conv_transpose2d(input, self.weight, self.bias)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3, 3, 7, 7), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 16, 16), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 3, 16, 16), dtype="float32") = R.nn.conv2d_transpose(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 3, 1, 1)) = R.reshape(w2, [1, 3, 1, 1])
                lv3: R.Tensor((1, 3, 16, 16), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 3, 16, 16), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class ConvTranspose2d2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.ConvTranspose2d(3, 3, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3, 3, 7, 7), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 16, 16), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 3, 16, 16), dtype="float32") = R.nn.conv2d_transpose(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    output_padding=[0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="IOHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 3, 16, 16), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = ConvTranspose2d1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = ConvTranspose2d1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = ConvTranspose2d2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_conv1d():
    class Conv1D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv1D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv1d(input, self.weight, self.bias)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            w1: R.Tensor((6, 3, 7), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
            input_1: R.Tensor((1, 3, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4), dtype="float32") = R.nn.conv1d(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="OIW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1), dtype="float32") = R.reshape(w2, [1, 6, 1])
                lv3: R.Tensor((1, 6, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class Conv1D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            w1: R.Tensor((6, 3, 7), dtype="float32"),
            input_1: R.Tensor((1, 3, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4), dtype="float32") = R.nn.conv1d(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0],
                    dilation=[1],
                    data_layout="NCW",
                    kernel_layout="OIW",
                    out_layout="NCW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 6, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, dtype=torch.float32),)

    model = Conv1D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv1D1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv1D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_conv2d():
    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv2D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv2d(input, self.weight, self.bias)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1, 1)) = R.reshape(w2, [1, 6, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = Conv2D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv2D1Func()
    binding = {"w1": model.weight.numpy(), "w2": model.bias.numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv2D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_conv3d():
    class Conv3D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    class Conv3D1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[6, 3, 7, 7, 7])
            self.bias = torch.randn(size=[6])

        def forward(self, input):
            return torch.nn.functional.conv3d(input, self.weight, self.bias)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7, 7), dtype="float32"),
            w2: R.Tensor((6,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.nn.conv3d(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0, 0],
                    dilation=[1],
                    data_layout="NCDHW",
                    kernel_layout="OIDHW",
                    out_layout="NCDHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1, 1, 1)) = R.reshape(w2, [1, 6, 1, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 4, 4, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class Conv3D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7, 7), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4, 4), dtype="float32")):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4, 4), dtype="float32") = R.nn.conv3d(
                    input_1,
                    w1,
                    strides=[1],
                    padding=[0, 0, 0],
                    dilation=[1],
                    data_layout="NCDHW",
                    kernel_layout="OIDHW",
                    out_layout="NCDHW",
                    out_dtype="float32",
                )
                gv: R.Tuple(R.Tensor((1, 6, 4, 4, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, 10, dtype=torch.float32),)

    model = Conv3D1()
    binding = {"w1": model.conv.weight.detach().numpy(), "w2": model.conv.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv3D1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Conv3D2()
    binding = {"w1": model.conv.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_pad():
    class PadModel(torch.nn.Module):
        def __init__(self, pad, mode="constant", value=0.0):
            super().__init__()
            self.pad = pad
            self.mode = mode
            self.value = value

        def forward(self, x):
            if self.mode == "constant":
                return torch.nn.functional.pad(x, self.pad, mode=self.mode, value=self.value)
            else:
                return torch.nn.functional.pad(x, self.pad, mode=self.mode)

    @tvm_tirx.script.ir_module
    class expected_constant:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 14, 12), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.nn.pad(
                    x,
                    pad_width=[0, 0, 0, 0, 2, 2, 1, 1],
                    pad_mode="constant",
                    pad_value=0.0,
                )
                gv: R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_reflect:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 14, 12), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((14,), dtype="int64") = R.arange(
                    R.prim_value(-2), R.prim_value(12), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((14,), dtype="int64") = R.abs(lv)
                lv2: R.Tensor((14,), dtype="int64") = R.subtract(R.const(9, "int64"), lv1)
                lv3: R.Tensor((14,), dtype="int64") = R.abs(lv2)
                lv4: R.Tensor((14,), dtype="int64") = R.subtract(R.const(9, "int64"), lv3)
                lv5: R.Tensor((1, 3, 14, 10), dtype="float32") = R.take(x, lv4, axis=2, mode="fast")
                lv6: R.Tensor((12,), dtype="int64") = R.arange(
                    R.prim_value(-1), R.prim_value(11), R.prim_value(1), dtype="int64"
                )
                lv7: R.Tensor((12,), dtype="int64") = R.abs(lv6)
                lv8: R.Tensor((12,), dtype="int64") = R.subtract(R.const(9, "int64"), lv7)
                lv9: R.Tensor((12,), dtype="int64") = R.abs(lv8)
                lv10: R.Tensor((12,), dtype="int64") = R.subtract(R.const(9, "int64"), lv9)
                lv11: R.Tensor((1, 3, 14, 12), dtype="float32") = R.take(
                    lv5, lv10, axis=3, mode="fast"
                )
                gv: R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")) = (lv11,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_replicate:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 14, 12), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((14,), dtype="int64") = R.arange(
                    R.prim_value(-2), R.prim_value(12), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((14,), dtype="int64") = R.clip(lv, R.prim_value(0), R.prim_value(9))
                lv2: R.Tensor((1, 3, 14, 10), dtype="float32") = R.take(x, lv1, axis=2, mode="fast")
                lv3: R.Tensor((12,), dtype="int64") = R.arange(
                    R.prim_value(-1), R.prim_value(11), R.prim_value(1), dtype="int64"
                )
                lv4: R.Tensor((12,), dtype="int64") = R.clip(lv3, R.prim_value(0), R.prim_value(9))
                lv5: R.Tensor((1, 3, 14, 12), dtype="float32") = R.take(
                    lv2, lv4, axis=3, mode="fast"
                )
                gv: R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_circular:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 14, 12), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 14, 12), dtype="float32") = R.zeros(
                    R.shape([1, 3, 14, 12]), dtype="float32"
                )

                lv1: R.Tensor((1, 3, 14, 10), dtype="float32") = R.strided_slice(
                    lv,
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    (R.prim_value(11),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.strided_slice(
                    x,
                    (R.prim_value(3),),
                    (R.prim_value(0),),
                    (R.prim_value(10),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = R.strided_slice(
                    lv1,
                    (R.prim_value(2),),
                    (R.prim_value(2),),
                    (R.prim_value(12),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv4: R.Tensor((1, 3, 10, 10), dtype="float32") = R.strided_slice(
                    lv2,
                    (R.prim_value(2),),
                    (R.prim_value(0),),
                    (R.prim_value(10),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv5: R.Tensor((1, 3, 10, 10), dtype="float32") = R.broadcast_to(
                    lv4, R.shape([1, 3, 10, 10])
                )

                lv6: R.Tensor((1, 3, 14, 10), dtype="float32") = R.strided_slice(
                    lv,
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    (R.prim_value(11),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv7: R.Tensor((1, 3, 14, 10), dtype="float32") = R.slice_scatter(
                    lv6, lv5, R.prim_value(2), R.prim_value(12), R.prim_value(1), axis=2
                )

                lv8: R.Tensor((1, 3, 14, 12), dtype="float32") = R.slice_scatter(
                    lv, lv7, R.prim_value(1), R.prim_value(11), R.prim_value(1), axis=3
                )

                lv9: R.Tensor((1, 3, 14, 1), dtype="float32") = R.strided_slice(
                    lv8,
                    (R.prim_value(3),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv10: R.Tensor((1, 3, 14, 1), dtype="float32") = R.strided_slice(
                    lv8,
                    (R.prim_value(3),),
                    (R.prim_value(10),),
                    (R.prim_value(11),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv11: R.Tensor((1, 3, 14, 1), dtype="float32") = R.broadcast_to(
                    lv10, R.shape([1, 3, 14, 1])
                )

                lv12: R.Tensor((1, 3, 14, 12), dtype="float32") = R.slice_scatter(
                    lv8, lv11, R.prim_value(0), R.prim_value(1), R.prim_value(1), axis=3
                )

                lv13: R.Tensor((1, 3, 14, 1), dtype="float32") = R.strided_slice(
                    lv12,
                    (R.prim_value(3),),
                    (R.prim_value(11),),
                    (R.prim_value(12),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv14: R.Tensor((1, 3, 14, 1), dtype="float32") = R.strided_slice(
                    lv12,
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv15: R.Tensor((1, 3, 14, 1), dtype="float32") = R.broadcast_to(
                    lv14, R.shape([1, 3, 14, 1])
                )
                lv16: R.Tensor((1, 3, 14, 12), dtype="float32") = R.slice_scatter(
                    lv12, lv15, R.prim_value(11), R.prim_value(12), R.prim_value(1), axis=3
                )

                lv17: R.Tensor((1, 3, 2, 12), dtype="float32") = R.strided_slice(
                    lv16,
                    (R.prim_value(2),),
                    (R.prim_value(0),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv18: R.Tensor((1, 3, 2, 12), dtype="float32") = R.strided_slice(
                    lv16,
                    (R.prim_value(2),),
                    (R.prim_value(10),),
                    (R.prim_value(12),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv19: R.Tensor((1, 3, 2, 12), dtype="float32") = R.broadcast_to(
                    lv18, R.shape([1, 3, 2, 12])
                )

                lv20: R.Tensor((1, 3, 14, 12), dtype="float32") = R.slice_scatter(
                    lv16, lv19, R.prim_value(0), R.prim_value(2), R.prim_value(1), axis=2
                )
                lv21: R.Tensor((1, 3, 2, 12), dtype="float32") = R.strided_slice(
                    lv20,
                    (R.prim_value(2),),
                    (R.prim_value(12),),
                    (R.prim_value(14),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv22: R.Tensor((1, 3, 2, 12), dtype="float32") = R.strided_slice(
                    lv20,
                    (R.prim_value(2),),
                    (R.prim_value(2),),
                    (R.prim_value(4),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )

                lv23: R.Tensor((1, 3, 2, 12), dtype="float32") = R.broadcast_to(
                    lv22, R.shape([1, 3, 2, 12])
                )

                lv24: R.Tensor((1, 3, 14, 12), dtype="float32") = R.slice_scatter(
                    lv20, lv23, R.prim_value(12), R.prim_value(14), R.prim_value(1), axis=2
                )
                gv: R.Tuple(R.Tensor((1, 3, 14, 12), dtype="float32")) = (lv24,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(PadModel(pad=[1, 1, 2, 2]), example_args, {}, expected_constant)
    verify_model(
        PadModel(pad=[1, 1, 2, 2], mode="reflect"),
        example_args,
        {},
        expected_reflect,
        run_ep_decomposition=True,
    )
    verify_model(
        PadModel(pad=[1, 1, 2, 2], mode="replicate"),
        example_args,
        {},
        expected_replicate,
        run_ep_decomposition=True,
    )
    verify_model(
        PadModel(pad=[1, 1, 2, 2], mode="circular"),
        example_args,
        {},
        expected_circular,
        run_ep_decomposition=True,
    )


def test_pixel_shuffle():
    class PixelShuffle(torch.nn.Module):
        def __init__(self, upscale_factor=2):
            super().__init__()
            self.upscale_factor = upscale_factor

        def forward(self, x):
            return torch.nn.functional.pixel_shuffle(x, self.upscale_factor)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((1, 8, 10, 15), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 2, 20, 30), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2, 10, 15), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 2, 2, 10, 15])
                )
                lv1: R.Tensor((1, 2, 10, 2, 15, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 1, 4, 2, 5, 3]
                )
                lv2: R.Tensor((1, 2, 20, 30), dtype="float32") = R.reshape(
                    lv1, R.shape([1, 2, 20, 30])
                )
                gv: R.Tuple(R.Tensor((1, 2, 20, 30), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 8, 10, 15, dtype=torch.float32),)
    verify_model(PixelShuffle(upscale_factor=2), example_args, {}, expected)


def test_einsum():
    class Einsum(Module):
        def __init__(self, subscripts):
            super().__init__()
            self.subscripts = subscripts

        def forward(self, *args):
            return torch.einsum(self.subscripts, *args)

    @tvm_tirx.script.ir_module
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((4, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.einsum((inp_0,), subscripts="ii")
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((5,), dtype="float32"), inp_1: R.Tensor((4,), dtype="float32")
        ) -> R.Tuple(R.Tensor((5, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 4), dtype="float32") = R.einsum(
                    (inp_0, inp_1), subscripts="i,j->ij"
                )
                gv: R.Tuple(R.Tensor((5, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(4, 4, dtype=torch.float32),)
    verify_model(Einsum("ii"), example_args, {}, Expected1, run_ep_decomposition=False)

    example_args = (torch.randn(5, dtype=torch.float32), torch.randn(4, dtype=torch.float32))
    verify_model(Einsum("i,j->ij"), example_args, {}, Expected2, run_ep_decomposition=False)

    # Default decomposition lowers repeated subscripts through aten.diagonal.
    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")) -> R.Tuple(R.Tensor((3,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="float32") = R.einsum((x,), subscripts="zz->z")
                lv1: R.Tensor((3,), dtype="float32") = R.permute_dims(lv, axes=[0])
                lv2: R.Tensor((3,), dtype="float32") = R.permute_dims(lv1, axes=[0])
                gv: R.Tuple(R.Tensor((3,), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, 3, dtype=torch.float32),)
    verify_model(Einsum("ii->i"), example_args, {}, Expected)

    for subscripts, shapes in [
        ("ii->", [(4, 4)]),
        ("...ii->...i", [(2, 3, 3)]),
        ("abca,abcb->c", [(3, 3, 4, 3), (3, 3, 4, 3)]),
    ]:
        verify_model_numerically(Einsum(subscripts), tuple(torch.randn(*shape) for shape in shapes))

    class DirectDiagonal(Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, x):
            return torch.diagonal(x, self.offset, 0, 1)

    class DirectTrace(Module):
        def forward(self, x):
            return torch.trace(x)

    # For a 3x4 input, 4 and -3 are the first empty diagonals. Larger offsets
    # in either direction check that negative diagonal lengths are clamped to zero.
    for offset in [1, 4, 6, -3, -6]:
        verify_model_numerically(DirectDiagonal(offset), (torch.randn(3, 4),))
    # Float32 trace reductions may accumulate diagonal entries in a different order.
    verify_model_numerically(DirectTrace(), (torch.randn(4, 4),), rtol=1e-6, atol=1e-6)

    # A square diagonal must read the input directly without O(N^2) copies,
    # both before and after legalization.
    n = 8
    exported_program = export(Einsum("ii->i"), args=(torch.randn(n, n),))
    mod = from_exported_program(exported_program)

    def rank2_call_results(ir_mod):
        """Names of calls whose result is a rank-2 (full-size) tensor."""
        results = []
        for func in ir_mod.functions.values():
            if not isinstance(func, relax.Function):
                continue
            for block in func.body.blocks:
                for binding in block.bindings:
                    if not (
                        isinstance(binding.value, relax.Call)
                        and isinstance(binding.value.op, tvm_tirx.ir.Op)
                    ):
                        continue
                    if isinstance(binding.var.ty, relax.TensorType) and binding.var.ty.ndim == 2:
                        results.append(binding.value.op.name)
        return results

    # The diagonal must be the only full-size (N x N) tensor touched: it is the
    # function input read directly by a single repeated-subscript einsum. No
    # call may produce a rank-2 intermediate.
    assert rank2_call_results(mod) == []

    # Sanity check that the graph really performs the diagonal: exactly one
    # einsum on the N x N input producing an N-vector.
    einsum_calls = []
    for block in mod["main"].body.blocks:
        for binding in block.bindings:
            if (
                isinstance(binding.value, relax.Call)
                and isinstance(binding.value.op, tvm_tirx.ir.Op)
                and binding.value.op.name == "relax.einsum"
            ):
                einsum_calls.append(binding.var)
    assert len(einsum_calls) == 1
    assert einsum_calls[0].ty.ndim == 1

    # Legalize and check again on the lowered graph.
    with tvm_tirx.target.Target("llvm"):
        lowered = relax.transform.LegalizeOps()(mod)
    assert rank2_call_results(lowered) == []


def test_outer():
    class Outer(torch.nn.Module):
        def forward(self, x, y):
            return torch.outer(x, y)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32"), y: R.Tensor((4,), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 1), dtype="float32") = R.reshape(x, R.shape([3, 1]))
                lv1: R.Tensor((3, 4), dtype="float32") = R.multiply(lv, y)
                gv: R.Tuple(R.Tensor((3, 4), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(3, dtype=torch.float32),
        torch.randn(4, dtype=torch.float32),
    )
    verify_model(Outer(), example_args, {}, expected)


def test_embedding():
    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, input):
            return self.embedding(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((4,), dtype="int64"), w1: R.Tensor((10, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((4, 3), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int32") = R.astype(input_1, dtype="int32")
                lv1: R.Tensor((4, 3), dtype="float32") = R.take(w1, lv, axis=0)
                gv: R.Tuple(R.Tensor((4, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randint(low=-int(1e5), high=int(1e5), size=(4,), dtype=torch.int64),)

    model = Embedding()
    binding = {"w1": model.embedding.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected1)


def test_groupnorm():
    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.GroupNorm(3, 3)

        def forward(self, input):
            return self.gn(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.group_norm(
                    input_1,
                    w1,
                    w2,
                    num_groups=3,
                    channel_axis=1,
                    axes=[2, 3],
                    epsilon=1.0000000000000001e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = GroupNorm()
    binding = {
        "w1": model.gn.weight.detach().numpy(),
        "w2": model.gn.bias.detach().numpy(),
    }
    verify_model(model, example_args, binding, expected1)


def test_instancenorm2d():
    class InstanceNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.InstanceNorm2d(3)

        def forward(self, input):
            return self.gn(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.instance_norm(
                    input_1,
                    w1,
                    w2,
                    channel_axis=1,
                    axes=[0, 2, 3],
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = InstanceNorm2d()
    binding = {
        "w1": torch.ones(3).detach().numpy(),
        "w2": torch.zeros(3).detach().numpy(),
    }
    verify_model(model, example_args, binding, expected1)


def test_layernorm():
    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm((10, 10))

        def forward(self, input):
            return self.ln(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((10, 10), dtype="float32"),
            w2: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.layer_norm(
                    input_1,
                    w1,
                    w2,
                    axes=[-2, -1],
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = LayerNorm()
    binding = {
        "w1": model.ln.weight.detach().numpy(),
        "w2": model.ln.bias.detach().numpy(),
    }
    verify_model(LayerNorm(), example_args, binding, expected1)


def test_linear():
    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, input):
            return self.linear(input)

    class Dense1Func(Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.randn(size=[7, 10])
            self.bias = torch.randn(size=[7])

        def forward(self, input):
            return torch.nn.functional.linear(input, self.weight, self.bias)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            w1: R.Tensor((7, 10), dtype="float32"),
            w2: R.Tensor((7,), dtype="float32"),
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((30, 10), dtype="float32") = R.reshape(input_1, R.shape([30, 10]))
                lv1: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=[1, 0])
                lv2: R.Tensor((30, 7), dtype="float32") = R.matmul(lv, lv1, out_dtype="float32")
                lv3: R.Tensor((30, 7), dtype="float32") = R.add(w2, lv2)
                lv4: R.Tensor((1, 3, 10, 7), dtype="float32") = R.reshape(
                    lv3, R.shape([1, 3, 10, 7])
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, input):
            return self.linear(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            w1: R.Tensor((7, 10), dtype="float32"),
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 7), dtype="float32") = R.permute_dims(w1, axes=[1, 0])
                lv1: R.Tensor((30, 10), dtype="float32") = R.reshape(input_1, R.shape([30, 10]))
                lv2: R.Tensor((30, 7), dtype="float32") = R.matmul(lv1, lv, out_dtype="float32")
                lv3: R.Tensor((1, 3, 10, 7), dtype="float32") = R.reshape(
                    lv2, R.shape([1, 3, 10, 7])
                )
                gv: R.Tuple(R.Tensor((1, 3, 10, 7), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    model = Dense1()
    binding = {"w1": model.linear.weight.detach().numpy(), "w2": model.linear.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Dense1Func()
    binding = {"w1": model.weight.detach().numpy(), "w2": model.bias.detach().numpy()}
    verify_model(model, example_args, binding, expected1)

    model = Dense2()
    binding = {"w1": model.linear.weight.detach().numpy()}
    verify_model(model, example_args, binding, expected2)


def test_maxpool1d():
    class MaxPool1d(Module):
        def forward(self, input):
            return torch.nn.functional.max_pool1d(input, kernel_size=2)

    class MaxPool1d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool1d(kernel_size=3, stride=2)

        def forward(self, input):
            return self.pool(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 8), dtype="float32") = R.expand_dims(input_1, axis=[-2])
                lv1: R.Tensor((1, 3, 1, 4), dtype="float32") = R.nn.max_pool2d(
                    lv,
                    pool_size=[1, 2],
                    strides=[1, 2],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv2: R.Tensor((1, 3, 1, 4), dtype="float32") = R.zeros_like(lv1)
                lv3: R.Tuple(
                    R.Tensor((1, 3, 1, 4), dtype="float32"),
                    R.Tensor((1, 3, 1, 4), dtype="float32"),
                ) = (lv1, lv2)
                lv4: R.Tensor((1, 3, 1, 4), dtype="float32") = lv3[0]
                lv5: R.Tensor((1, 3, 4), dtype="float32") = R.squeeze(lv4, axis=[-2])
                gv: R.Tuple(R.Tensor((1, 3, 4), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected3:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 10), dtype="float32") = R.expand_dims(input_1, axis=[-2])
                lv1: R.Tensor((1, 3, 1, 4), dtype="float32") = R.nn.max_pool2d(
                    lv,
                    pool_size=[1, 3],
                    strides=[1, 2],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv2: R.Tensor((1, 3, 1, 4), dtype="float32") = R.zeros_like(lv1)
                lv3: R.Tuple(
                    R.Tensor((1, 3, 1, 4), dtype="float32"),
                    R.Tensor((1, 3, 1, 4), dtype="float32"),
                ) = (lv1, lv2)
                lv4: R.Tensor((1, 3, 1, 4), dtype="float32") = lv3[0]
                lv5: R.Tensor((1, 3, 4), dtype="float32") = R.squeeze(lv4, axis=[-2])
                gv: R.Tuple(R.Tensor((1, 3, 4), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    # Example inputs
    example_args1 = (torch.randn(1, 3, 8, dtype=torch.float32),)
    example_args3 = (torch.randn(1, 3, 10, dtype=torch.float32),)

    # Verify the models
    verify_model(MaxPool1d(), example_args1, {}, expected1)
    verify_model(MaxPool1d2(), example_args3, {}, expected3)


def test_maxpool2d():
    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    class MaxPool2d_functional(Module):
        def forward(self, input):
            return torch.nn.functional.max_pool2d(input, kernel_size=[1, 1])

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[1, 1],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.zeros_like(lv)
                lv2: R.Tuple(
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                ) = (lv, lv1)
                lv3: R.Tensor((1, 3, 10, 10), dtype="float32") = lv2[0]
                gv: R.Tuple(R.Tensor((1, 3, 10, 10), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3])

        def forward(self, input):
            return self.pool(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 4, 4), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    dilation=[2, 3],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv1: R.Tensor((1, 3, 4, 4), dtype="float32") = R.zeros_like(lv)
                lv2: R.Tuple(
                    R.Tensor((1, 3, 4, 4), dtype="float32"), R.Tensor((1, 3, 4, 4), dtype="float32")
                ) = (lv, lv1)
                lv3: R.Tensor((1, 3, 4, 4), dtype="float32") = lv2[0]
                gv: R.Tuple(R.Tensor((1, 3, 4, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class MaxPool2d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, input):
            return self.pool(input)

    @tvm_tirx.script.ir_module
    class expected3:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 6, 6), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 6, 6), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[4, 4],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[2, 2, 2, 2],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                lv1: R.Tensor((1, 3, 6, 6), dtype="float32") = R.zeros_like(lv)
                lv2: R.Tuple(
                    R.Tensor((1, 3, 6, 6), dtype="float32"), R.Tensor((1, 3, 6, 6), dtype="float32")
                ) = (lv, lv1)
                lv3: R.Tensor((1, 3, 6, 6), dtype="float32") = lv2[0]
                gv: R.Tuple(R.Tensor((1, 3, 6, 6), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(MaxPool2d(), example_args, {}, expected1)
    verify_model(MaxPool2d_functional(), example_args, {}, expected1)
    verify_model(MaxPool2d2(), example_args, {}, expected2)
    verify_model(MaxPool2d3(), example_args, {}, expected3)


def test_maxpool3d():
    class MaxPool3d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(kernel_size=[1, 1, 1])

        def forward(self, input):
            return self.pool(input)

    class MaxPool3d_functional(Module):
        def forward(self, input):
            return torch.nn.functional.max_pool3d(input, kernel_size=[1, 1, 1])

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 4, 4, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 4, 4, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = R.nn.max_pool3d(
                    input_1,
                    pool_size=[1, 1, 1],
                    strides=[1, 1, 1],
                    dilation=[1, 1, 1],
                    padding=[0, 0, 0, 0, 0, 0],
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                lv1: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = R.zeros_like(lv)
                lv2: R.Tuple(
                    R.Tensor((1, 3, 4, 4, 4), dtype="float32"),
                    R.Tensor((1, 3, 4, 4, 4), dtype="float32"),
                ) = (lv, lv1)
                lv3: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = lv2[0]
                gv: R.Tuple(R.Tensor((1, 3, 4, 4, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class MaxPool3d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(kernel_size=[2, 2, 2], dilation=[2, 2, 2])

        def forward(self, input):
            return self.pool(input)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 8, 8, 8), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 3, 3, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 3, 3, 3), dtype="float32") = R.nn.max_pool3d(
                    input_1,
                    pool_size=[2, 2, 2],
                    strides=[2, 2, 2],
                    dilation=[2, 2, 2],
                    padding=[0, 0, 0, 0, 0, 0],
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                lv1: R.Tensor((1, 3, 3, 3, 3), dtype="float32") = R.zeros_like(lv)
                lv2: R.Tuple(
                    R.Tensor((1, 3, 3, 3, 3), dtype="float32"),
                    R.Tensor((1, 3, 3, 3, 3), dtype="float32"),
                ) = (lv, lv1)
                lv3: R.Tensor((1, 3, 3, 3, 3), dtype="float32") = lv2[0]
                gv: R.Tuple(R.Tensor((1, 3, 3, 3, 3), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    class MaxPool3d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool3d(kernel_size=[3, 3, 3], padding=1, stride=2)

        def forward(self, input):
            return self.pool(input)

    @tvm_tirx.script.ir_module
    class expected3:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 5, 5, 5), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 5, 5), dtype="float32") = R.nn.max_pool3d(
                    input_1,
                    pool_size=[3, 3, 3],
                    strides=[2, 2, 2],
                    dilation=[1, 1, 1],
                    padding=[1, 1, 1, 1, 1, 1],
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                lv1: R.Tensor((1, 3, 5, 5, 5), dtype="float32") = R.zeros_like(lv)
                lv2: R.Tuple(
                    R.Tensor((1, 3, 5, 5, 5), dtype="float32"),
                    R.Tensor((1, 3, 5, 5, 5), dtype="float32"),
                ) = (lv, lv1)
                lv3: R.Tensor((1, 3, 5, 5, 5), dtype="float32") = lv2[0]
                gv: R.Tuple(R.Tensor((1, 3, 5, 5, 5), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    # Example input tensors
    example_args1 = (torch.randn(1, 3, 4, 4, 4, dtype=torch.float32),)
    example_args2 = (torch.randn(1, 3, 8, 8, 8, dtype=torch.float32),)
    example_args3 = (torch.randn(1, 3, 10, 10, 10, dtype=torch.float32),)

    # Verify the models with expected IR modules
    verify_model(MaxPool3d(), example_args1, {}, expected1)
    verify_model(MaxPool3d_functional(), example_args1, {}, expected1)
    verify_model(MaxPool3d2(), example_args2, {}, expected2)
    verify_model(MaxPool3d3(), example_args3, {}, expected3)


def test_scaled_dot_product_attention():
    class Attention1(Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            q: R.Tensor((32, 8, 128, 64), dtype="float32"),
            k: R.Tensor((32, 8, 128, 64), dtype="float32"),
            v: R.Tensor((32, 8, 128, 64), dtype="float32"),
        ) -> R.Tuple(R.Tensor((32, 8, 128, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    q, axes=[0, 2, 1, 3]
                )
                lv1: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    k, axes=[0, 2, 1, 3]
                )
                lv2: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    v, axes=[0, 2, 1, 3]
                )
                lv3: R.Tensor((32, 128, 8, 64), dtype="float32") = R.nn.attention(
                    lv, lv1, lv2, scale=None, causal_mask=None, window_size=None
                )
                lv4: R.Tensor((32, 8, 128, 64), dtype="float32") = R.permute_dims(
                    lv3, axes=[0, 2, 1, 3]
                )
                gv: R.Tuple(R.Tensor((32, 8, 128, 64), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    class Attention2(Module):
        def forward(self, q, k, v, mask):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, mask)

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            q: R.Tensor((32, 8, 128, 64), dtype="float32"),
            k: R.Tensor((32, 8, 128, 64), dtype="float32"),
            v: R.Tensor((32, 8, 128, 64), dtype="float32"),
            mask: R.Tensor((32, 8, 128, 128), dtype="float32"),
        ) -> R.Tuple(R.Tensor((32, 8, 128, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    q, axes=[0, 2, 1, 3]
                )
                lv1: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    k, axes=[0, 2, 1, 3]
                )
                lv2: R.Tensor((32, 128, 8, 64), dtype="float32") = R.permute_dims(
                    v, axes=[0, 2, 1, 3]
                )
                lv3: R.Tensor((32, 128, 8, 64), dtype="float32") = R.nn.attention_bias(
                    lv, lv1, lv2, mask, scale=None, causal_mask=None, window_size=None
                )
                lv4: R.Tensor((32, 8, 128, 64), dtype="float32") = R.permute_dims(
                    lv3, axes=[0, 2, 1, 3]
                )
                gv: R.Tuple(R.Tensor((32, 8, 128, 64), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    verify_model(
        Attention1(),
        (
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
        ),
        {},
        Expected1,
        run_ep_decomposition=False,
    )

    verify_model(
        Attention2(),
        (
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 64, dtype=torch.float32),
            torch.randn(32, 8, 128, 128, dtype=torch.float32),
        ),
        {},
        Expected2,
        run_ep_decomposition=False,
    )

    # Test 2D input (seq_len, head_dim) - bug fix for #18441
    class Attention2D(Module):
        def forward(self, x):
            return torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=False)

    @I.ir_module
    class Expected2D:
        @R.function
        def main(
            x: R.Tensor((8, 32), dtype="float32"),
        ) -> R.Tuple(R.Tensor((8, 32), dtype="float32")):
            with R.dataflow():
                # Expand to add batch dimension for query, key, value separately
                # (8, 32) -> (1, 8, 32)
                lv: R.Tensor((1, 8, 32), dtype="float32") = R.expand_dims(x, axis=[0])
                lv1: R.Tensor((1, 8, 32), dtype="float32") = R.expand_dims(x, axis=[0])
                lv2: R.Tensor((1, 8, 32), dtype="float32") = R.expand_dims(x, axis=[0])
                # Expand to add num_heads dimension: (1, 8, 32) -> (1, 1, 8, 32)
                lv3: R.Tensor((1, 1, 8, 32), dtype="float32") = R.expand_dims(lv, axis=[1])
                lv4: R.Tensor((1, 1, 8, 32), dtype="float32") = R.expand_dims(lv1, axis=[1])
                lv5: R.Tensor((1, 1, 8, 32), dtype="float32") = R.expand_dims(lv2, axis=[1])
                # Attention operation: (1, 1, 8, 32) -> (1, 1, 8, 32)
                lv6: R.Tensor((1, 1, 8, 32), dtype="float32") = R.nn.attention(
                    lv3, lv4, lv5, scale=None, causal_mask=None, window_size=None
                )
                # Squeeze batch and num_heads dimensions: (1, 1, 8, 32) -> (8, 32)
                lv7: R.Tensor((8, 32), dtype="float32") = R.squeeze(lv6, axis=[0, 1])
                gv: R.Tuple(R.Tensor((8, 32), dtype="float32")) = (lv7,)
                R.output(gv)
            return gv

    verify_model(
        Attention2D(),
        (torch.randn(8, 32, dtype=torch.float32),),
        {},
        Expected2D,
        run_ep_decomposition=False,
    )


def test_unbind():
    class Unbind1(Module):
        def forward(self, data):
            return torch.unbind(data)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(data: R.Tensor((3, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(0),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv2: R.Tensor((1, 3, 10, 10), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(0),),
                    (R.prim_value(2),),
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv3: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv, axis=[0])
                lv4: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv1, axis=[0])
                lv5: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv2, axis=[0])
                gv: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv3, lv4, lv5)
                R.output(gv)
            return gv

    class Unbind2(Module):
        def forward(self, data):
            return torch.unbind(data, dim=1)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(data: R.Tensor((3, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
            R.Tensor((3, 10, 10), dtype="float32"),
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((3, 1, 10, 10), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(1),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv1: R.Tensor((3, 1, 10, 10), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv2: R.Tensor((3, 1, 10, 10), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv3: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv, axis=[1])
                lv4: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv5: R.Tensor((3, 10, 10), dtype="float32") = R.squeeze(lv2, axis=[1])
                gv: R.Tuple(
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                    R.Tensor((3, 10, 10), dtype="float32"),
                ) = (lv3, lv4, lv5)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected3:
        @R.function
        def main(data: R.Tensor((3, 1, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 1, 3), dtype="float32") = R.strided_slice(
                    data,
                    (R.prim_value(1),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv1: R.Tensor((3, 3), dtype="float32") = R.squeeze(lv, axis=[1])
                gv: R.Tuple(R.Tensor((3, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, 3, 10, 10, dtype=torch.float32),)
    verify_model(Unbind1(), example_args, {}, expected1)
    verify_model(Unbind2(), example_args, {}, expected2)
    single_dim_args = (torch.randn(3, 1, 3, dtype=torch.float32),)
    verify_model(Unbind2(), single_dim_args, {}, expected3)


def test_interpolate():
    class InterpolateBilinear(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (224, 224), mode="bilinear")

    @tvm_tirx.script.ir_module
    class expected_bilinear:
        @R.function
        def main(input: R.Tensor((1, 3, 112, 112), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 224, 224), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 224, 224), dtype="float32") = R.image.resize2d(
                    input,
                    R.shape([224, 224]),
                    roi=[T.float32(0.0), T.float32(0.0), T.float32(0.0), T.float32(0.0)],
                    layout="NCHW",
                    method="linear",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.75,
                    cubic_exclude=0,
                    extrapolation_value=0.0,
                )
                gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class InterpolateNearest(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (224, 224), mode="nearest")

    @tvm_tirx.script.ir_module
    class expected_nearest:
        @R.function
        def main(input: R.Tensor((1, 3, 112, 112), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 224, 224), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 224, 224), dtype="float32") = R.image.resize2d(
                    input,
                    R.shape([224, 224]),
                    roi=[T.float32(0.0), T.float32(0.0), T.float32(0.0), T.float32(0.0)],
                    layout="NCHW",
                    method="nearest_neighbor",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.75,
                    cubic_exclude=0,
                    extrapolation_value=0.0,
                )
                gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class InterpolateBicubic(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(input, (224, 224), mode="bicubic")

    @I.ir_module
    class expected_bicubic:
        @R.function
        def main(input: R.Tensor((1, 3, 112, 112), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 224, 224), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((224,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(224), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((224,), dtype="float32") = R.astype(lv, dtype="float32")
                lv2: R.Tensor((224,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(224), R.prim_value(1), dtype="int64"
                )
                lv3: R.Tensor((224,), dtype="float32") = R.astype(lv2, dtype="float32")
                lv4: R.Tensor((224,), dtype="float32") = R.add(lv3, R.const(0.5, "float32"))
                lv5: R.Tensor((224,), dtype="float32") = R.multiply(lv4, R.const(0.5, "float32"))
                lv6: R.Tensor((224,), dtype="float32") = R.subtract(lv5, R.const(0.5, "float32"))
                lv7: R.Tensor((224,), dtype="float32") = R.add(lv1, R.const(0.5, "float32"))
                lv8: R.Tensor((224,), dtype="float32") = R.multiply(lv7, R.const(0.5, "float32"))
                lv9: R.Tensor((224,), dtype="float32") = R.subtract(lv8, R.const(0.5, "float32"))
                lv10: R.Tensor((224, 1), dtype="float32") = R.expand_dims(lv9, axis=[-1])
                lv11: R.Tensor((224,), dtype="float32") = R.floor(lv6)
                lv12: R.Tensor((224, 1), dtype="float32") = R.floor(lv10)
                lv13: R.Tensor((224, 1), dtype="float32") = R.subtract(lv10, lv12)
                lv14: R.Tensor((224, 1), dtype="float32") = R.clip(
                    lv13, R.prim_value(T.float64(0.0)), R.prim_value(T.float64(1.0))
                )
                lv15: R.Tensor((224,), dtype="float32") = R.subtract(lv6, lv11)
                lv16: R.Tensor((224,), dtype="float32") = R.clip(
                    lv15, R.prim_value(T.float64(0.0)), R.prim_value(T.float64(1.0))
                )
                lv17: R.Tensor((224,), dtype="int64") = R.astype(lv11, dtype="int64")
                lv18: R.Tensor((224, 1), dtype="int64") = R.astype(lv12, dtype="int64")
                lv19: R.Tensor((224, 1), dtype="int64") = R.subtract(lv18, R.const(1, "int64"))
                lv20: R.Tensor((224, 1), dtype="int64") = R.add(lv18, R.const(1, "int64"))
                lv21: R.Tensor((224, 1), dtype="int64") = R.add(lv18, R.const(2, "int64"))
                lv22: R.Tensor((224,), dtype="int64") = R.subtract(lv17, R.const(1, "int64"))
                lv23: R.Tensor((224,), dtype="int64") = R.add(lv17, R.const(1, "int64"))
                lv24: R.Tensor((224,), dtype="int64") = R.add(lv17, R.const(2, "int64"))
                lv25: R.Tensor((224,), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv16)
                lv26: R.Tensor((448,), dtype="float32") = R.concat((lv16, lv25), axis=0)
                lv27: R.Tensor((2, 224), dtype="float32") = R.reshape(lv26, R.shape([2, 224]))
                lv28: R.Tensor((224,), dtype="float32") = R.add(lv16, R.const(1.0, "float32"))
                lv29: R.Tensor((224,), dtype="float32") = R.subtract(R.const(2.0, "float32"), lv16)
                lv30: R.Tensor((448,), dtype="float32") = R.concat((lv28, lv29), axis=0)
                lv31: R.Tensor((2, 224), dtype="float32") = R.reshape(lv30, R.shape([2, 224]))
                lv32: R.Tensor((2, 224), dtype="float32") = R.multiply(
                    lv31, R.const(-0.75, "float32")
                )
                lv33: R.Tensor((2, 224), dtype="float32") = R.subtract(
                    lv32, R.const(-3.75, "float32")
                )
                lv34: R.Tensor((2, 224), dtype="float32") = R.multiply(lv33, lv31)
                lv35: R.Tensor((2, 224), dtype="float32") = R.add(lv34, R.const(-6.0, "float32"))
                lv36: R.Tensor((2, 224), dtype="float32") = R.multiply(lv35, lv31)
                lv37: R.Tensor((2, 224), dtype="float32") = R.subtract(
                    lv36, R.const(-3.0, "float32")
                )
                lv38: R.Tensor((2, 224), dtype="float32") = R.multiply(
                    lv27, R.const(1.25, "float32")
                )
                lv39: R.Tensor((2, 224), dtype="float32") = R.subtract(
                    lv38, R.const(2.25, "float32")
                )
                lv40: R.Tensor((2, 224), dtype="float32") = R.multiply(lv39, lv27)
                lv41: R.Tensor((2, 224), dtype="float32") = R.multiply(lv40, lv27)
                lv42: R.Tensor((2, 224), dtype="float32") = R.add(lv41, R.const(1.0, "float32"))
                lv43: R.Tensor((1, 224), dtype="float32") = R.strided_slice(
                    lv37,
                    (R.prim_value(0),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv44: R.Tensor((1, 224), dtype="float32") = R.strided_slice(
                    lv37,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv45: R.Tensor((224,), dtype="float32") = R.squeeze(lv43, axis=[0])
                lv46: R.Tensor((224,), dtype="float32") = R.squeeze(lv44, axis=[0])
                lv47: R.Tensor((1, 224), dtype="float32") = R.strided_slice(
                    lv42,
                    (R.prim_value(0),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv48: R.Tensor((1, 224), dtype="float32") = R.strided_slice(
                    lv42,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv49: R.Tensor((224,), dtype="float32") = R.squeeze(lv47, axis=[0])
                lv50: R.Tensor((224,), dtype="float32") = R.squeeze(lv48, axis=[0])
                lv51: R.Tensor((224, 1), dtype="float32") = R.subtract(
                    R.const(1.0, "float32"), lv14
                )
                lv52: R.Tensor((448, 1), dtype="float32") = R.concat((lv14, lv51), axis=0)
                lv53: R.Tensor((2, 224, 1), dtype="float32") = R.reshape(lv52, R.shape([2, 224, 1]))
                lv54: R.Tensor((224, 1), dtype="float32") = R.add(lv14, R.const(1.0, "float32"))
                lv55: R.Tensor((224, 1), dtype="float32") = R.subtract(
                    R.const(2.0, "float32"), lv14
                )
                lv56: R.Tensor((448, 1), dtype="float32") = R.concat((lv54, lv55), axis=0)
                lv57: R.Tensor((2, 224, 1), dtype="float32") = R.reshape(lv56, R.shape([2, 224, 1]))
                lv58: R.Tensor((2, 224, 1), dtype="float32") = R.multiply(
                    lv57, R.const(-0.75, "float32")
                )
                lv59: R.Tensor((2, 224, 1), dtype="float32") = R.subtract(
                    lv58, R.const(-3.75, "float32")
                )
                lv60: R.Tensor((2, 224, 1), dtype="float32") = R.multiply(lv59, lv57)
                lv61: R.Tensor((2, 224, 1), dtype="float32") = R.add(lv60, R.const(-6.0, "float32"))
                lv62: R.Tensor((2, 224, 1), dtype="float32") = R.multiply(lv61, lv57)
                lv63: R.Tensor((2, 224, 1), dtype="float32") = R.subtract(
                    lv62, R.const(-3.0, "float32")
                )
                lv64: R.Tensor((2, 224, 1), dtype="float32") = R.multiply(
                    lv53, R.const(1.25, "float32")
                )
                lv65: R.Tensor((2, 224, 1), dtype="float32") = R.subtract(
                    lv64, R.const(2.25, "float32")
                )
                lv66: R.Tensor((2, 224, 1), dtype="float32") = R.multiply(lv65, lv53)
                lv67: R.Tensor((2, 224, 1), dtype="float32") = R.multiply(lv66, lv53)
                lv68: R.Tensor((2, 224, 1), dtype="float32") = R.add(lv67, R.const(1.0, "float32"))
                lv69: R.Tensor((1, 224, 1), dtype="float32") = R.strided_slice(
                    lv63,
                    (R.prim_value(0),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv70: R.Tensor((1, 224, 1), dtype="float32") = R.strided_slice(
                    lv63,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv71: R.Tensor((224, 1), dtype="float32") = R.squeeze(lv69, axis=[0])
                lv72: R.Tensor((224, 1), dtype="float32") = R.squeeze(lv70, axis=[0])
                lv73: R.Tensor((1, 224, 1), dtype="float32") = R.strided_slice(
                    lv68,
                    (R.prim_value(0),),
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv74: R.Tensor((1, 224, 1), dtype="float32") = R.strided_slice(
                    lv68,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                lv75: R.Tensor((224, 1), dtype="float32") = R.squeeze(lv73, axis=[0])
                lv76: R.Tensor((224, 1), dtype="float32") = R.squeeze(lv74, axis=[0])
                lv77: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv19, R.prim_value(0), R.prim_value(111)
                )
                lv78: R.Tensor((224,), dtype="int64") = R.clip(
                    lv22, R.prim_value(0), R.prim_value(111)
                )
                lv79: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv78, axis=3, mode="fast"
                )
                lv80: R.Tensor((224,), dtype="int64") = R.squeeze(lv77, axis=None)
                lv81: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv79, lv80, axis=2, mode="fast"
                )
                lv82: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv19, R.prim_value(0), R.prim_value(111)
                )
                lv83: R.Tensor((224,), dtype="int64") = R.clip(
                    lv17, R.prim_value(0), R.prim_value(111)
                )
                lv84: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv83, axis=3, mode="fast"
                )
                lv85: R.Tensor((224,), dtype="int64") = R.squeeze(lv82, axis=None)
                lv86: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv84, lv85, axis=2, mode="fast"
                )
                lv87: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv19, R.prim_value(0), R.prim_value(111)
                )
                lv88: R.Tensor((224,), dtype="int64") = R.clip(
                    lv23, R.prim_value(0), R.prim_value(111)
                )
                lv89: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv88, axis=3, mode="fast"
                )
                lv90: R.Tensor((224,), dtype="int64") = R.squeeze(lv87, axis=None)
                lv91: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv89, lv90, axis=2, mode="fast"
                )
                lv92: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv19, R.prim_value(0), R.prim_value(111)
                )
                lv93: R.Tensor((224,), dtype="int64") = R.clip(
                    lv24, R.prim_value(0), R.prim_value(111)
                )
                lv94: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv93, axis=3, mode="fast"
                )
                lv95: R.Tensor((224,), dtype="int64") = R.squeeze(lv92, axis=None)
                lv96: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv94, lv95, axis=2, mode="fast"
                )
                lv97: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv81, lv45)
                lv98: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv86, lv49)
                lv99: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv97, lv98)
                lv100: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv91, lv50)
                lv101: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv99, lv100)
                lv102: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv96, lv46)
                lv103: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv101, lv102)
                lv104: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv18, R.prim_value(0), R.prim_value(111)
                )
                lv105: R.Tensor((224,), dtype="int64") = R.clip(
                    lv22, R.prim_value(0), R.prim_value(111)
                )
                lv106: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv105, axis=3, mode="fast"
                )
                lv107: R.Tensor((224,), dtype="int64") = R.squeeze(lv104, axis=None)
                lv108: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv106, lv107, axis=2, mode="fast"
                )
                lv109: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv18, R.prim_value(0), R.prim_value(111)
                )
                lv110: R.Tensor((224,), dtype="int64") = R.clip(
                    lv17, R.prim_value(0), R.prim_value(111)
                )
                lv111: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv110, axis=3, mode="fast"
                )
                lv112: R.Tensor((224,), dtype="int64") = R.squeeze(lv109, axis=None)
                lv113: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv111, lv112, axis=2, mode="fast"
                )
                lv114: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv18, R.prim_value(0), R.prim_value(111)
                )
                lv115: R.Tensor((224,), dtype="int64") = R.clip(
                    lv23, R.prim_value(0), R.prim_value(111)
                )
                lv116: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv115, axis=3, mode="fast"
                )
                lv117: R.Tensor((224,), dtype="int64") = R.squeeze(lv114, axis=None)
                lv118: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv116, lv117, axis=2, mode="fast"
                )
                lv119: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv18, R.prim_value(0), R.prim_value(111)
                )
                lv120: R.Tensor((224,), dtype="int64") = R.clip(
                    lv24, R.prim_value(0), R.prim_value(111)
                )
                lv121: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv120, axis=3, mode="fast"
                )
                lv122: R.Tensor((224,), dtype="int64") = R.squeeze(lv119, axis=None)
                lv123: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv121, lv122, axis=2, mode="fast"
                )
                lv124: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv108, lv45)
                lv125: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv113, lv49)
                lv126: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv124, lv125)
                lv127: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv118, lv50)
                lv128: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv126, lv127)
                lv129: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv123, lv46)
                lv130: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv128, lv129)
                lv131: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv20, R.prim_value(0), R.prim_value(111)
                )
                lv132: R.Tensor((224,), dtype="int64") = R.clip(
                    lv22, R.prim_value(0), R.prim_value(111)
                )
                lv133: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv132, axis=3, mode="fast"
                )
                lv134: R.Tensor((224,), dtype="int64") = R.squeeze(lv131, axis=None)
                lv135: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv133, lv134, axis=2, mode="fast"
                )
                lv136: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv20, R.prim_value(0), R.prim_value(111)
                )
                lv137: R.Tensor((224,), dtype="int64") = R.clip(
                    lv17, R.prim_value(0), R.prim_value(111)
                )
                lv138: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv137, axis=3, mode="fast"
                )
                lv139: R.Tensor((224,), dtype="int64") = R.squeeze(lv136, axis=None)
                lv140: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv138, lv139, axis=2, mode="fast"
                )
                lv141: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv20, R.prim_value(0), R.prim_value(111)
                )
                lv142: R.Tensor((224,), dtype="int64") = R.clip(
                    lv23, R.prim_value(0), R.prim_value(111)
                )
                lv143: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv142, axis=3, mode="fast"
                )
                lv144: R.Tensor((224,), dtype="int64") = R.squeeze(lv141, axis=None)
                lv145: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv143, lv144, axis=2, mode="fast"
                )
                lv146: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv20, R.prim_value(0), R.prim_value(111)
                )
                lv147: R.Tensor((224,), dtype="int64") = R.clip(
                    lv24, R.prim_value(0), R.prim_value(111)
                )
                lv148: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv147, axis=3, mode="fast"
                )
                lv149: R.Tensor((224,), dtype="int64") = R.squeeze(lv146, axis=None)
                lv150: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv148, lv149, axis=2, mode="fast"
                )
                lv151: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv135, lv45)
                lv152: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv140, lv49)
                lv153: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv151, lv152)
                lv154: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv145, lv50)
                lv155: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv153, lv154)
                lv156: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv150, lv46)
                lv157: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv155, lv156)
                lv158: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv21, R.prim_value(0), R.prim_value(111)
                )
                lv159: R.Tensor((224,), dtype="int64") = R.clip(
                    lv22, R.prim_value(0), R.prim_value(111)
                )
                lv160: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv159, axis=3, mode="fast"
                )
                lv161: R.Tensor((224,), dtype="int64") = R.squeeze(lv158, axis=None)
                lv162: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv160, lv161, axis=2, mode="fast"
                )
                lv163: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv21, R.prim_value(0), R.prim_value(111)
                )
                lv164: R.Tensor((224,), dtype="int64") = R.clip(
                    lv17, R.prim_value(0), R.prim_value(111)
                )
                lv165: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv164, axis=3, mode="fast"
                )
                lv166: R.Tensor((224,), dtype="int64") = R.squeeze(lv163, axis=None)
                lv167: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv165, lv166, axis=2, mode="fast"
                )
                lv168: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv21, R.prim_value(0), R.prim_value(111)
                )
                lv169: R.Tensor((224,), dtype="int64") = R.clip(
                    lv23, R.prim_value(0), R.prim_value(111)
                )
                lv170: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv169, axis=3, mode="fast"
                )
                lv171: R.Tensor((224,), dtype="int64") = R.squeeze(lv168, axis=None)
                lv172: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv170, lv171, axis=2, mode="fast"
                )
                lv173: R.Tensor((224, 1), dtype="int64") = R.clip(
                    lv21, R.prim_value(0), R.prim_value(111)
                )
                lv174: R.Tensor((224,), dtype="int64") = R.clip(
                    lv24, R.prim_value(0), R.prim_value(111)
                )
                lv175: R.Tensor((1, 3, 112, 224), dtype="float32") = R.take(
                    input, lv174, axis=3, mode="fast"
                )
                lv176: R.Tensor((224,), dtype="int64") = R.squeeze(lv173, axis=None)
                lv177: R.Tensor((1, 3, 224, 224), dtype="float32") = R.take(
                    lv175, lv176, axis=2, mode="fast"
                )
                lv178: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv162, lv45)
                lv179: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv167, lv49)
                lv180: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv178, lv179)
                lv181: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv172, lv50)
                lv182: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv180, lv181)
                lv183: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv177, lv46)
                lv184: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv182, lv183)
                lv185: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv103, lv71)
                lv186: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv130, lv75)
                lv187: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv185, lv186)
                lv188: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv157, lv76)
                lv189: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv187, lv188)
                lv190: R.Tensor((1, 3, 224, 224), dtype="float32") = R.multiply(lv184, lv72)
                lv191: R.Tensor((1, 3, 224, 224), dtype="float32") = R.add(lv189, lv190)
                gv: R.Tuple(R.Tensor((1, 3, 224, 224), dtype="float32")) = (lv191,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 112, 112, dtype=torch.float32),)
    verify_model(InterpolateBilinear(), example_args, {}, expected_bilinear)
    verify_model(InterpolateNearest(), example_args, {}, expected_nearest)
    verify_model(InterpolateBicubic(), example_args, {}, expected_bicubic)


def test_interpolate_antialiased():
    """Test bilinear interpolation with antialiasing enabled."""

    class InterpolateBilinearAA(Module):
        def forward(self, input):
            return torch.nn.functional.interpolate(
                input, size=(64, 64), mode="bilinear", align_corners=False, antialias=True
            )

    @tvm_tirx.script.ir_module
    class expected_bilinear_aa:
        @R.function
        def main(input: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 64, 64), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 3, 64, 64), dtype="float32") = R.image.resize2d(
                    input,
                    R.shape([64, 64]),
                    roi=[T.float32(0.0), T.float32(0.0), T.float32(0.0), T.float32(0.0)],
                    layout="NCHW",
                    method="linear",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.75,
                    cubic_exclude=0,
                    extrapolation_value=0.0,
                )
                gv: R.Tuple(R.Tensor((1, 3, 64, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 32, 32, dtype=torch.float32),)
    verify_model(InterpolateBilinearAA(), example_args, {}, expected_bilinear_aa)


def test_mean():
    class Mean(Module):
        def forward(self, input):
            return input.mean(-1)

    class MeanKeepDim(Module):
        def forward(self, input: torch.Tensor):
            return input.mean(-1, keepdim=True)

    class MeanWithoutDim(Module):
        def forward(self, input: torch.Tensor):
            return input.mean()

    @I.ir_module
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((256,), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((256,), dtype="float32") = R.mean(inp_0, axis=[-1], keepdims=False)
                gv: R.Tuple(R.Tensor((256,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((256, 1), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((256, 1), dtype="float32") = R.mean(inp_0, axis=[-1], keepdims=True)
                gv: R.Tuple(R.Tensor((256, 1), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.mean(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(256, 256, dtype=torch.float32),)
    verify_model(Mean(), example_args, {}, Expected1)
    verify_model(MeanKeepDim(), example_args, {}, Expected2)
    verify_model(MeanWithoutDim(), example_args, {}, Expected3)


def test_median():
    class Median(Module):
        def forward(self, input):
            return input.median(-1)

    class MedianKeepDim(Module):
        def forward(self, input):
            return input.median(-1, keepdim=True)

    class MedianWithoutDim(Module):
        def forward(self, input):
            return input.median()

    @I.ir_module
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tuple(R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="int64")) = (
                    R.median(inp_0, axis=[-1], keepdims=False)
                )
                lv1: R.Tensor((256,), dtype="float32") = lv[0]
                lv2: R.Tensor((256,), dtype="int64") = lv[1]
                gv: R.Tuple(R.Tensor((256,), dtype="float32"), R.Tensor((256,), dtype="int64")) = (
                    lv1,
                    lv2,
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((256, 1), dtype="float32"), R.Tensor((256, 1), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((256, 1), dtype="float32"), R.Tensor((256, 1), dtype="int64")
                ) = R.median(inp_0, axis=[-1], keepdims=True)
                lv1: R.Tensor((256, 1), dtype="float32") = lv[0]
                lv2: R.Tensor((256, 1), dtype="int64") = lv[1]
                gv: R.Tuple(
                    R.Tensor((256, 1), dtype="float32"), R.Tensor((256, 1), dtype="int64")
                ) = (lv1, lv2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.median(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(256, 256, dtype=torch.float32),)
    verify_model(Median(), example_args, {}, Expected1)
    verify_model(MedianKeepDim(), example_args, {}, Expected2)
    verify_model(MedianWithoutDim(), example_args, {}, Expected3)


def test_sum():
    class Sum(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1))

    class SumKeepDim(Module):
        def forward(self, x):
            return torch.sum(x, (2, 1), keepdim=True)

    class SumWithoutDim(Module):
        def forward(self, x):
            return torch.sum(x)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 4), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4), dtype="float32") = R.sum(inp_0, axis=[2, 1], keepdims=False)
                gv: R.Tuple(R.Tensor((1, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 1, 1, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 1, 1, 4), dtype="float32") = R.sum(
                    inp_0, axis=[2, 1], keepdims=True
                )
                gv: R.Tuple(R.Tensor((1, 1, 1, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected3:
        @R.function
        def main(inp_0: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.sum(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Sum(), example_args, {}, expected1)
    verify_model(SumKeepDim(), example_args, {}, expected2)
    verify_model(SumWithoutDim(), example_args, {}, expected3)


def test_argmax_argmin():
    example_args = (torch.randn(256, 256, dtype=torch.float32),)

    class Argmax1(Module):
        def forward(self, input):
            return torch.argmax(input, dim=-1)

    class Argmax2(Module):
        def forward(self, input):
            return torch.argmax(input, dim=-1, keepdim=True)

    @tvm_tirx.script.ir_module
    class expected_argmax1:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((256,), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tensor((256,), dtype="int64") = R.argmax(inp_0, axis=-1, keepdims=False)
                gv: R.Tuple(R.Tensor((256,), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_argmax2:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((256, 1), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tensor((256, 1), dtype="int64") = R.argmax(inp_0, axis=-1, keepdims=True)
                gv: R.Tuple(R.Tensor((256, 1), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Argmax1(), example_args, {}, expected_argmax1)
    verify_model(Argmax2(), example_args, {}, expected_argmax2)

    class Argmin1(Module):
        def forward(self, input):
            return torch.argmin(input)

    class Argmin2(Module):
        def forward(self, input):
            return torch.argmin(input, keepdim=True)

    @tvm_tirx.script.ir_module
    class expected_argmin1:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="int64") = R.argmin(inp_0, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected_argmin2:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 1), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 1), dtype="int64") = R.argmin(inp_0, axis=None, keepdims=True)
                gv: R.Tuple(R.Tensor((1, 1), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    verify_model(Argmin1(), example_args, {}, expected_argmin1)
    verify_model(Argmin2(), example_args, {}, expected_argmin2)


def test_cat():
    class Cat0(Module):
        def forward(self, x, y):
            return torch.cat((x, y))

    class Cat1(Module):
        def forward(self, x, y):
            return torch.cat((x, y), dim=1)

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 3), dtype="float32") = R.concat((inp_0, inp_1), axis=0)
                gv: R.Tuple(R.Tensor((4, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
            inp_1: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 6), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 6), dtype="float32") = R.concat((inp_0, inp_1), axis=1)
                gv: R.Tuple(R.Tensor((2, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32), torch.randn(2, 3, dtype=torch.float32))
    verify_model(Cat0(), example_args, {}, Expected1)
    verify_model(Cat1(), example_args, {}, Expected2)


def test_cumsum():
    class Cumsum(Module):
        def forward(self, input):
            return torch.cumsum(input, dim=1, dtype=torch.int32)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 2, 3, 4), dtype="int32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="int32") = R.cumsum(input_1, axis=1, dtype="int32")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="int32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Cumsum(), example_args, {}, expected1)


def test_expand():
    class Expand1(Module):
        def forward(self, x):
            return x.expand(4, 2, 3, 4)

    class Expand2(Module):
        def forward(self, x):
            return x.expand(4, -1, -1, 4)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((4, 2, 3, 4), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 2, 3, 4), dtype="float32") = R.broadcast_to(x, (4, 2, 3, 4))
                gv: R.Tuple(R.Tensor((4, 2, 3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Expand1(), example_args, {}, expected1)
    verify_model(Expand2(), example_args, {}, expected1)


def test_flatten():
    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(2, -1)

        def forward(self, input):
            return self.f(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 100), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 100), dtype="float32") = R.reshape(input_1, (1, 3, 100))
                gv: R.Tuple(R.Tensor((1, 3, 100), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Flatten(), example_args, {}, expected1)


def test_meshgrid():
    class Meshgrid1(Module):
        def forward(self, input1, input2):
            return torch.meshgrid((input1, input2), indexing="ij")

    class Meshgrid2(Module):
        def forward(self, input1, input2):
            return torch.meshgrid((input1, input2), indexing="xy")

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input1: R.Tensor((3,), dtype="float32"), input2: R.Tensor((3,), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 1), dtype="float32") = R.reshape(input1, R.shape([3, 1]))
                lv1: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv, R.shape([3, 3]))
                lv2: R.Tensor((1, 3), dtype="float32") = R.reshape(input2, R.shape([1, 3]))
                lv3: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv2, R.shape([3, 3]))
                gv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv1, lv3)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(
            input1: R.Tensor((3,), dtype="float32"), input2: R.Tensor((3,), dtype="float32")
        ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 1), dtype="float32") = R.reshape(input2, R.shape([3, 1]))
                lv1: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv, R.shape([3, 3]))
                lv2: R.Tensor((1, 3), dtype="float32") = R.reshape(input1, R.shape([1, 3]))
                lv3: R.Tensor((3, 3), dtype="float32") = R.broadcast_to(lv2, R.shape([3, 3]))
                gv: R.Tuple(
                    R.Tensor((3, 3), dtype="float32"), R.Tensor((3, 3), dtype="float32")
                ) = (lv3, lv1)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(3, dtype=torch.float32),
        torch.randn(3, dtype=torch.float32),
    )
    verify_model(Meshgrid1(), example_args, {}, expected1)
    verify_model(Meshgrid2(), example_args, {}, expected2)


def test_permute():
    class Permute1(Module):
        def forward(self, x):
            return x.permute(0, 3, 2, 1)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 4, 3, 2), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tuple(R.Tensor((1, 4, 3, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Permute1(), example_args, {}, expected1)


def test_repeat():
    class Tile1(Module):
        def forward(self, x: torch.Tensor):
            return x.repeat(2)

    class Tile2(Module):
        def forward(self, x: torch.Tensor):
            return x.repeat(4, 2)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32")) -> R.Tuple(R.Tensor((6,), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((6,), dtype="float32") = R.tile(x, 2)
                gv: R.Tuple(R.Tensor((6,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(x: R.Tensor((1, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((4, 6), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 6), dtype="float32") = R.tile(x, [4, 2])
                gv: R.Tuple(R.Tensor((4, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, dtype=torch.float32),)
    verify_model(Tile1(), example_args, {}, expected1)

    example_args = (torch.randn(1, 3, dtype=torch.float32),)
    verify_model(Tile2(), example_args, {}, expected2)


def test_reshape():
    class Reshape(Module):
        def forward(self, x):
            return x.reshape(2, 12)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 12), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tuple(R.Tensor((2, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Reshape(), example_args, {}, expected1)


def test_reshape_as():
    class ReshapeAs(Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor):
            return x.reshape_as(y)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            x: R.Tensor((1, 2, 3, 4), dtype="float32"),
            y: R.Tensor((2, 12), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 12), dtype="float32")):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tuple(R.Tensor((2, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 2, 3, 4, dtype=torch.float32),
        torch.randn(2, 12, dtype=torch.float32),
    )
    verify_model(ReshapeAs(), example_args, {}, expected1)


def test_reshape_zero_sized_dim():
    class Reshape(Module):
        def forward(self, x):
            return x.reshape(0, 4)

    class ReshapeTrailing(Module):
        def forward(self, x):
            return x.reshape(3, 0)

    verify_model_numerically(Reshape(), (torch.randn(2, 0, 4, dtype=torch.float32),))
    verify_model_numerically(ReshapeTrailing(), (torch.randn(0, 3, dtype=torch.float32),))


def test_reshape_zero_sized_dim_symbolic_target():
    # Only a literal is read as "copy the input dimension", so a symbolic dimension in the
    # target is not one: it is carried through, and the literal zero beside it still has to
    # be rewritten. Skipping the rewrite because the target is not fully static reads that
    # zero as a copy and gives a non-empty shape torch never produces.
    class Flatten(Module):
        def forward(self, x):
            return x.flatten(1, 2)

    class KeepBatch(Module):
        def forward(self, x):
            return x.reshape(x.shape[0], 0, 4)

    class ZeroBeforeBatch(Module):
        def forward(self, x):
            return x.reshape(0, x.shape[0])

    batch = torch.export.Dim("batch", min=1, max=64)
    example_args = (torch.randn(3, 2, 0, 4, dtype=torch.float32),)
    for model in (Flatten(), KeepBatch(), ZeroBeforeBatch()):
        verify_model_numerically(model, example_args, dynamic_shapes={"x": {0: batch}})


def test_roll():
    class Roll1(Module):
        def forward(self, x):
            return torch.roll(x, 1)

    class Roll2(Module):
        def forward(self, x):
            return torch.roll(x, -1, 0)

    class Roll3(Module):
        def forward(self, x):
            return torch.roll(x, shifts=(2, 1), dims=(0, 1))

    # Test case 1: torch.roll(x, 1)
    @I.ir_module
    class Expected1:
        @R.function
        def main(x: R.Tensor((4, 2), dtype="int64")) -> R.Tuple(R.Tensor((4, 2), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((8,), dtype="int64") = R.reshape(x, R.shape([8]))
                lv1: R.Tensor((8,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(8), R.prim_value(1), dtype="int64"
                )
                lv2: R.Tensor((8,), dtype="int64") = R.add(lv1, R.const(7, "int64"))
                lv3: R.Tensor((8,), dtype="int64") = R.mod(lv2, R.const(8, "int64"))
                lv4: R.Tensor((8,), dtype="int64") = R.take(lv, lv3, axis=0, mode="fast")
                lv5: R.Tensor((4, 2), dtype="int64") = R.reshape(lv4, R.shape([4, 2]))
                gv: R.Tuple(R.Tensor((4, 2), dtype="int64")) = (lv5,)
                R.output(gv)
            return gv

    # Test case 2: torch.roll(x, -1, 0)
    @I.ir_module
    class Expected2:
        @R.function
        def main(x: R.Tensor((4, 2), dtype="int64")) -> R.Tuple(R.Tensor((4, 2), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(4), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((4,), dtype="int64") = R.add(lv, R.const(1, "int64"))
                lv2: R.Tensor((4,), dtype="int64") = R.mod(lv1, R.const(4, "int64"))
                lv3: R.Tensor((4, 2), dtype="int64") = R.take(x, lv2, axis=0, mode="fast")
                gv: R.Tuple(R.Tensor((4, 2), dtype="int64")) = (lv3,)
                R.output(gv)
            return gv

    # Test case 3: torch.roll(x, shifts=(2,1), dims=(0,1))
    @I.ir_module
    class Expected3:
        @R.function
        def main(x: R.Tensor((4, 2), dtype="int64")) -> R.Tuple(R.Tensor((4, 2), dtype="int64")):
            with R.dataflow():
                # First roll along dim=0 with shift=2
                lv: R.Tensor((4,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(4), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((4,), dtype="int64") = R.add(lv, R.const(2, "int64"))
                lv2: R.Tensor((4,), dtype="int64") = R.mod(lv1, R.const(4, "int64"))
                lv3: R.Tensor((4, 2), dtype="int64") = R.take(x, lv2, axis=0, mode="fast")
                # Second roll along dim=1 with shift=1
                lv4: R.Tensor((2,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(2), R.prim_value(1), dtype="int64"
                )
                lv5: R.Tensor((2,), dtype="int64") = R.add(lv4, R.const(1, "int64"))
                lv6: R.Tensor((2,), dtype="int64") = R.mod(lv5, R.const(2, "int64"))
                lv7: R.Tensor((4, 2), dtype="int64") = R.take(lv3, lv6, axis=1, mode="fast")
                gv: R.Tuple(R.Tensor((4, 2), dtype="int64")) = (lv7,)
                R.output(gv)
            return gv

    # Test inputs
    example_input = torch.randint(0, 10, (4, 2), dtype=torch.int64)

    # Run verification for each case
    verify_model(Roll1(), (example_input,), {}, Expected1)
    verify_model(Roll2(), (example_input,), {}, Expected2)
    verify_model(Roll3(), (example_input,), {}, Expected3)


def test_select_slice():
    class Slice1(Module):
        def forward(self, x):
            return x[0, 1::2, :, :3]

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 10, 3), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((3, 10, 10), dtype="float32") = R.take(
                    x, R.const(0, "int64"), axis=0, mode="fast"
                )
                lv1: R.Tensor((1, 10, 10), dtype="float32") = R.strided_slice(
                    lv,
                    (R.prim_value(0),),
                    (R.prim_value(1),),
                    (R.prim_value(9223372036854775807),),
                    (R.prim_value(2),),
                    assume_inbound=False,
                )
                lv2: R.Tensor((1, 10, 3), dtype="float32") = R.strided_slice(
                    lv1,
                    (R.prim_value(2),),
                    (R.prim_value(0),),
                    (R.prim_value(3),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                gv: R.Tuple(R.Tensor((1, 10, 3), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    class Slice2(Module):
        def forward(self, x):
            return x[:, None, None, :, None]

    @I.ir_module
    class expected2:
        @R.function
        def main(x: R.Tensor((8, 16), dtype="float32")) -> R.Tuple(
            R.Tensor((8, 1, 1, 16, 1), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((8, 1, 16), dtype="float32") = R.expand_dims(x, axis=[1])
                lv1: R.Tensor((8, 1, 1, 16), dtype="float32") = R.expand_dims(lv, axis=[2])
                lv2: R.Tensor((8, 1, 1, 16, 1), dtype="float32") = R.expand_dims(lv1, axis=[4])
                gv: R.Tuple(R.Tensor((8, 1, 1, 16, 1), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Slice1(), example_args, {}, expected1)

    example_args = (torch.randn(8, 16, dtype=torch.float32),)
    verify_model(Slice2(), example_args, {}, expected2)


def test_slice_scatter():
    class SliceScatter1(Module):
        def forward(self, input, src):
            return torch.slice_scatter(input, src, dim=1, start=1, end=7, step=2)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            a: R.Tensor((8, 8, 10, 10), dtype="float32"),
            b: R.Tensor((8, 3, 10, 10), dtype="float32"),
        ) -> R.Tuple(R.Tensor((8, 8, 10, 10), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((8, 8, 10, 10), dtype="float32") = R.slice_scatter(
                    a, b, R.prim_value(1), R.prim_value(7), R.prim_value(2), axis=1
                )
                gv: R.Tuple(R.Tensor((8, 8, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class SliceScatter2(Module):
        def forward(self, input, src):
            return torch.slice_scatter(input, src, dim=0, start=0, end=6, step=1)

    @I.ir_module
    class expected2:
        @R.function
        def main(
            a: R.Tensor((8, 16), dtype="float32"), b: R.Tensor((6, 16), dtype="float32")
        ) -> R.Tuple(R.Tensor((8, 16), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((8, 16), dtype="float32") = R.slice_scatter(
                    a, b, R.prim_value(0), R.prim_value(6), R.prim_value(1), axis=0
                )
                gv: R.Tuple(R.Tensor((8, 16), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class SliceScatterNegative(Module):
        def forward(self, input, src):
            return torch.slice_scatter(input, src, dim=1, start=0, end=-2, step=1)

    @tvm_tirx.script.ir_module
    class expected_slice_scatter:
        @R.function
        def main(
            a: R.Tensor((2, 5), dtype="float32"), b: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tuple(R.Tensor((2, 5), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 5), dtype="float32") = R.slice_scatter(
                    a, b, R.prim_value(0), R.prim_value(3), R.prim_value(1), axis=1
                )
                gv: R.Tuple(R.Tensor((2, 5), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(8, 8, 10, 10, dtype=torch.float32), torch.randn(8, 3, 10, 10))
    verify_model(SliceScatter1(), example_args, {}, expected1)

    example_args = (torch.randn(8, 16, dtype=torch.float32), torch.randn(6, 16))
    verify_model(SliceScatter2(), example_args, {}, expected2)

    example_args = (torch.randn(2, 5, dtype=torch.float32), torch.randn(2, 3, dtype=torch.float32))
    verify_model(SliceScatterNegative(), example_args, {}, expected_slice_scatter)


def test_slice_with_symbolic_end():
    """_slice correctly handles symbolic end values from dynamic shapes."""

    class SliceIdentityModel(torch.nn.Module):
        def forward(self, x):
            # x[:, :x.size(1)] is an identity slice that torch.export emits
            # as slice(x, 1, 0, sym_size_int(x, 1), 1) with dynamic shapes.
            seq_len = x.size(1)
            return x[:, :seq_len] + 0.0  # +0.0 to ensure output is a new tensor

    # The identity slice is elided; only x + 0.0 remains.
    s0 = T.dynamic("s0", "int64")
    s1 = T.dynamic("s1", "int64")

    @I.ir_module
    class ExpectedIdentity:
        @R.function
        def main(x: R.Tensor((s0, s1, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((s0, s1, 4), dtype="float32")
        ):
            R.func_attr({"tir_var_lower_bound": {"s27": 2, "s77": 2}})
            with R.dataflow():
                lv: R.Tensor((s0, s1, 4), dtype="float32") = R.add(x, R.const(0.0, "float32"))
                gv: R.Tuple(R.Tensor((s0, s1, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 8, 4, dtype=torch.float32),)
    batch = torch.export.Dim("batch", min=2)
    seq = torch.export.Dim("seq", min=2)
    dynamic_shapes = {"x": {0: batch, 1: seq}}

    verify_model(
        SliceIdentityModel(),
        example_args,
        {},
        ExpectedIdentity,
        dynamic_shapes=dynamic_shapes,
        map_free_vars=True,
    )

    class SliceStaticModel(torch.nn.Module):
        def forward(self, x):
            # A non-identity static slice
            return x[:, :3]

    @tvm_tirx.script.ir_module
    class ExpectedStatic:
        @R.function
        def main(x: R.Tensor((2, 8, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 3, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 3, 4), dtype="float32") = R.strided_slice(
                    x,
                    axes=[1],
                    begin=[0],
                    end=[3],
                    strides=[1],
                )
                gv: R.Tuple(R.Tensor((2, 3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args_static = (torch.randn(2, 8, 4, dtype=torch.float32),)
    verify_model(SliceStaticModel(), example_args_static, {}, ExpectedStatic)


def test_derived_input_dimension_without_exported_program_decomposition():
    class IdentityPair(torch.nn.Module):
        def forward(self, x, y):
            return x, y

    frames = torch.export.Dim("frames", min=1, max=8)
    exported_program = export(
        IdentityPair(),
        args=(torch.randn(1, 4, 3), torch.randn(1, 8, 3)),
        dynamic_shapes=({1: frames}, {1: 2 * frames}),
    )
    mod = from_exported_program(
        exported_program,
        keep_params_as_input=True,
        run_ep_decomposition=False,
    )

    x_shape = mod["main"].params[0].ty.shape.values
    y_shape = mod["main"].params[1].ty.shape.values
    assert tvm_tirx.sym.Analyzer().can_prove_equal(y_shape[1], x_shape[1] * 2)


def test_expand_with_new_leading_dimension():
    class ExpandLeading(torch.nn.Module):
        def forward(self, x):
            return x.expand(2, -1, -1)

    tokens = torch.export.Dim("tokens", min=1, max=8)
    exported_program = export(
        ExpandLeading(),
        args=(torch.randn(4, 3),),
        dynamic_shapes={"x": {0: tokens}},
    )
    mod = from_exported_program(exported_program)

    input_shape = mod["main"].params[0].ty.shape.values
    output_shape = mod["main"].ret_ty.fields[0].shape.values
    assert tvm_tirx.sym.Analyzer().can_prove_equal(output_shape[0], 2)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(output_shape[1], input_shape[0])
    assert tvm_tirx.sym.Analyzer().can_prove_equal(output_shape[2], input_shape[1])


def test_dynamic_scalar_item_in_shape_operations():
    class DynamicShapeOps(torch.nn.Module):
        def forward(self, x):
            lengths = torch.full(
                (x.shape[0],),
                x.shape[1],
                device=x.device,
                dtype=torch.int64,
            )
            max_len = lengths.max().item()
            positions = torch.arange(max_len, device=x.device)
            mask = positions.unsqueeze(0).expand(x.shape[0], -1) == lengths.unsqueeze(1)
            filled = torch.full(
                (x.shape[0], x.shape[1]),
                x.shape[1],
                device=x.device,
                dtype=torch.int64,
            )
            shifted = torch.arange(x.shape[1] + 1, device=x.device)
            shortened = torch.full_like(lengths, x.shape[1] - 1)
            return mask, filled, shifted, shortened

    example_args = (torch.randn(1, 4, 3, dtype=torch.float32),)
    tokens = torch.export.Dim("tokens", min=1, max=8)
    exported_program = export(
        DynamicShapeOps(),
        args=example_args,
        dynamic_shapes={"x": {1: tokens}},
    )
    mod = from_exported_program(exported_program)

    script = mod.script()
    assert "R.tensor_to_shape" in script
    assert "R.shape_to_tensor" in script

    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())
    for token_count in (4, 6):
        torch_input = torch.randn(1, token_count, 3)
        expected = DynamicShapeOps()(torch_input)
        actual = vm["main"](tvm_tirx.runtime.tensor(torch_input.numpy()))
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_array_equal(actual_value.numpy(), expected_value.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_runtime_scalar_item_integer_value_ranges():
    class RuntimeItems(torch.nn.Module):
        def forward(self, x, signed, unsigned, wide):
            return (
                torch.full_like(x, signed.item(), dtype=torch.int64),
                torch.full_like(x, unsigned.item(), dtype=torch.int64),
                torch.full_like(x, wide.item(), dtype=torch.int64),
            )

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.tensor([[-3]], dtype=torch.int8),
        torch.tensor([200], dtype=torch.uint8),
        torch.tensor(1 << 40, dtype=torch.int64),
    )
    exported_program = export(RuntimeItems(), args=example_args)
    mod = from_exported_program(exported_program, run_ep_decomposition=False)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    expected = RuntimeItems()(*example_args)
    actual = vm["main"](*(tvm_tirx.runtime.tensor(arg.numpy()) for arg in example_args))
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_array_equal(actual_value.numpy(), expected_value.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_dynamic_scalar_fill_operations():
    class DynamicFills(torch.nn.Module):
        def forward(self, x):
            inferred_dtype = torch.full((x.shape[0],), x.shape[1], device=x.device)
            explicit_dtype = torch.full_like(x, x.shape[0], dtype=torch.float64)
            filled = torch.fill(x, x.shape[1])
            filled_inplace = x.clone()
            filled_inplace.fill_(x.shape[0])
            return inferred_dtype, explicit_dtype, filled, filled_inplace

    example_args = (torch.randn(3, 4, dtype=torch.float32),)
    rows = torch.export.Dim("rows", min=1, max=8)
    columns = torch.export.Dim("columns", min=1, max=8)
    exported_program = export(
        DynamicFills(),
        args=example_args,
        dynamic_shapes={"x": {0: rows, 1: columns}},
    )
    mod = from_exported_program(exported_program, run_ep_decomposition=False)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    for shape in ((3, 4), (5, 2)):
        torch_input = torch.randn(shape, dtype=torch.float32)
        expected = DynamicFills()(torch_input)
        actual = vm["main"](tvm_tirx.runtime.tensor(torch_input.numpy()))
        actual_arrays = [value.numpy() for value in actual]

        assert actual_arrays[0].dtype == np.dtype("int64")
        assert actual_arrays[1].dtype == np.dtype("float64")
        for actual_value, expected_value in zip(actual_arrays, expected):
            assert actual_value.dtype == expected_value.numpy().dtype
            np.testing.assert_array_equal(actual_value, expected_value.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_dynamic_boolean_fill_operations():
    class DynamicBooleanFills(torch.nn.Module):
        def forward(self, x):
            value = x.shape[0] == x.shape[1]
            base = x > 0
            filled = torch.fill(base, value)
            filled_inplace = base.clone()
            filled_inplace.fill_(value)
            created = torch.full(
                (x.shape[0], x.shape[1]),
                value,
                device=x.device,
                dtype=torch.bool,
            )
            like = torch.full_like(x, value, dtype=torch.bool)
            return created, like, filled, filled_inplace

    rows = torch.export.Dim("rows", min=1, max=8)
    columns = torch.export.Dim("columns", min=1, max=8)
    example_args = (torch.randn(3, 4, dtype=torch.float32),)
    exported_program = export(
        DynamicBooleanFills(),
        args=example_args,
        dynamic_shapes={"x": {0: rows, 1: columns}},
    )
    mod = from_exported_program(exported_program, run_ep_decomposition=False)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    for shape in ((3, 3), (3, 4)):
        torch_input = torch.randn(shape, dtype=torch.float32)
        expected = DynamicBooleanFills()(torch_input)
        actual = vm["main"](tvm_tirx.runtime.tensor(torch_input.numpy()))
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_array_equal(actual_value.numpy(), expected_value.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_dynamic_scalar_arithmetic():
    class DynamicScalarArithmetic(torch.nn.Module):
        def forward(self, x):
            lengths = torch.full(
                (x.shape[0],),
                x.shape[1],
                device=x.device,
                dtype=torch.int64,
            )
            value = lengths.max().item()
            return (
                torch.full_like(x, -value, dtype=torch.int64),
                torch.arange(value // 2, device=x.device),
                torch.arange(value % 3, device=x.device),
            )

    rows = torch.export.Dim("rows", min=1, max=8)
    columns = torch.export.Dim("columns", min=1, max=8)
    example_args = (torch.randn(3, 4, dtype=torch.float32),)
    exported_program = export(
        DynamicScalarArithmetic(),
        args=example_args,
        dynamic_shapes={"x": {0: rows, 1: columns}},
    )
    mod = from_exported_program(exported_program, run_ep_decomposition=False)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    for shape in ((3, 4), (5, 2)):
        torch_input = torch.randn(shape, dtype=torch.float32)
        expected = DynamicScalarArithmetic()(torch_input)
        actual = vm["main"](tvm_tirx.runtime.tensor(torch_input.numpy()))
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_array_equal(actual_value.numpy(), expected_value.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize(
    ("fill_value", "dtype"),
    [
        (1 << 40, torch.int64),
        (1.0 + 2**-40, torch.float64),
    ],
)
def test_full_and_full_like_python_fill_value_uses_explicit_dtype(fill_value, dtype):
    class FullLike(torch.nn.Module):
        def forward(self, x):
            return (
                torch.full(x.shape, fill_value, dtype=dtype),
                torch.full_like(x, fill_value, dtype=dtype),
            )

    example_args = (torch.randn(2, 3, dtype=torch.float32),)
    exported_program = export(FullLike(), args=example_args)
    mod = from_exported_program(exported_program, run_ep_decomposition=False)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    expected = FullLike()(*example_args)
    actual = vm["main"](tvm_tirx.runtime.tensor(example_args[0].numpy()))
    for actual_value, expected_value in zip(actual, expected):
        assert actual_value.numpy().dtype == expected_value.numpy().dtype
        np.testing.assert_array_equal(actual_value.numpy(), expected_value.numpy())


def test_split():
    class Chunk(Module):
        def forward(self, input):
            return torch.chunk(input, 3, dim=1)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 1, 10, 10), dtype="float32"),
            R.Tensor((1, 1, 10, 10), dtype="float32"),
            R.Tensor((1, 1, 10, 10), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                ) = R.split(input, indices_or_sections=[1, 2], axis=1)
                lv1: R.Tensor((1, 1, 10, 10), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 1, 10, 10), dtype="float32") = lv[1]
                lv3: R.Tensor((1, 1, 10, 10), dtype="float32") = lv[2]
                gv: R.Tuple(
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                    R.Tensor((1, 1, 10, 10), dtype="float32"),
                ) = (lv1, lv2, lv3)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    verify_model(Chunk(), example_args, {}, Expected)


def test_split_int_split_size():
    """x.split(int, dim) must produce chunks of size `split_size` (the last one
    smaller when the dimension is not divisible), matching PyTorch.

    The frontend used to convert the int per-chunk size into a section count and
    pass it as relax.op.split's int argument, which means "split into N equal
    sections"; that yields wrong chunk shapes whenever
    ceil(D / ceil(D / split_size)) != split_size (e.g. split_size > D/2). The
    int branch now builds cumulative cut positions, the same as the list/tuple
    form.
    """

    class Split6(Module):
        def forward(self, input):
            return input.split(6, dim=0)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10,), dtype="float32")) -> R.Tuple(
            R.Tensor((6,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((6,), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                ) = R.split(input, indices_or_sections=[6], axis=0)
                lv1: R.Tensor((6,), dtype="float32") = lv[0]
                lv2: R.Tensor((4,), dtype="float32") = lv[1]
                gv: R.Tuple(
                    R.Tensor((6,), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                ) = (lv1, lv2)
                R.output(gv)
            return gv

    example_args = (torch.arange(10, dtype=torch.float32) + 1,)
    verify_model(Split6(), example_args, {}, Expected)

    # Differential check against native PyTorch for non-divisible sizes and dims.
    class SplitModel(Module):
        def __init__(self, split_size, dim):
            super().__init__()
            self.split_size = split_size
            self.dim = dim

        def forward(self, input):
            return input.split(self.split_size, dim=self.dim)

    def run_tvm(model, args):
        exported_program = export(model, args=args)
        mod = from_exported_program(exported_program)
        ex = relax.build(mod, target="llvm")
        vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
        out = vm["main"](*[tvm_tirx.runtime.tensor(a.numpy()) for a in args])
        if hasattr(out, "numpy"):
            return [out.numpy()]
        return [o.numpy() for o in out]

    for shape, split_size, dim in [
        ((10,), 6, 0),
        ((12, 8), 5, 1),
        ((3, 10), 6, -1),
    ]:
        x = torch.arange(1, int(np.prod(shape)) + 1, dtype=torch.float32).reshape(shape)
        refs = [r.numpy() for r in x.split(split_size, dim)]
        outs = run_tvm(SplitModel(split_size, dim), (x,))
        assert [r.shape for r in refs] == [o.shape for o in outs], (
            f"split shape={shape} s={split_size} dim={dim}: "
            f"torch {[r.shape for r in refs]} vs tvm {[o.shape for o in outs]}"
        )
        for r, o in zip(refs, outs):
            tvm_tirx.testing.assert_allclose(o, r, rtol=1e-7, atol=1e-7)


def test_squeeze():
    class Squeeze1(Module):
        def forward(self, input):
            return input.squeeze(1)

    @tvm_tirx.script.ir_module
    class Expected1:
        @R.function
        def main(inp_0: R.Tensor((3, 1, 4, 1), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 4, 1), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 4, 1), dtype="float32") = R.squeeze(inp_0, axis=[1])
                gv: R.Tuple(R.Tensor((3, 4, 1), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Squeeze2(Module):
        def forward(self, input):
            return input.squeeze()

    @tvm_tirx.script.ir_module
    class Expected2:
        @R.function
        def main(input: R.Tensor((3, 1, 4, 1), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 4), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 4), dtype="float32") = R.squeeze(input, axis=[0, 1, 2, 3])
                gv: R.Tuple(R.Tensor((3, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Squeeze3(Module):
        def forward(self, input):
            return input.squeeze(2)

    @I.ir_module
    class Expected3:
        @R.function
        def main(inp_0: R.Tensor((3, 1, 4, 1), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 1, 4, 1), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 1, 4, 1), dtype="float32") = R.squeeze(inp_0, axis=[2])
                gv: R.Tuple(R.Tensor((3, 1, 4, 1), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(3, 1, 4, 1, dtype=torch.float32),)

    verify_model(Squeeze1(), example_args, {}, Expected1)
    verify_model(Squeeze2(), example_args, {}, Expected2)
    verify_model(Squeeze3(), example_args, {}, Expected3)


def test_stack():
    class Stack0(Module):
        def forward(self, x, y):
            return torch.stack((x, y))  # default dim=0

    class Stack1(Module):
        def forward(self, x, y):
            return torch.stack((x, y), dim=1)

    class Stack3(Module):
        def forward(self, x, y):
            return torch.stack((x, y), dim=-1)  # negative dim

    @I.ir_module
    class Expected0:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 3), dtype="float32") = R.concat((x, y), axis=0)
                lv1: R.Tensor((2, 2, 3), dtype="float32") = R.reshape(lv, R.shape([2, 2, 3]))
                gv: R.Tuple(R.Tensor((2, 2, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected1:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 6), dtype="float32") = R.concat((x, y), axis=1)
                lv1: R.Tensor((2, 2, 3), dtype="float32") = R.reshape(lv, R.shape([2, 2, 3]))
                gv: R.Tuple(R.Tensor((2, 2, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            y: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 1), dtype="float32") = R.expand_dims(x, axis=[2])
                lv1: R.Tensor((2, 3, 1), dtype="float32") = R.expand_dims(y, axis=[2])
                lv2: R.Tensor((2, 3, 2), dtype="float32") = R.concat((lv, lv1), axis=-1)
                gv: R.Tuple(R.Tensor((2, 3, 2), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32), torch.randn(2, 3, dtype=torch.float32))

    verify_model(Stack0(), example_args, {}, Expected0)
    verify_model(Stack1(), example_args, {}, Expected1)
    verify_model(Stack3(), example_args, {}, Expected3)


def test_tile():
    class Tile1(Module):
        def forward(self, x):
            return x.tile((2,))

    class Tile2(Module):
        def forward(self, x):
            return x.tile(4, 2)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 6), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 6), dtype="float32") = R.tile(x, repeats=[1, 2])
                gv: R.Tuple(R.Tensor((1, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(x: R.Tensor((1, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((4, 6), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4, 6), dtype="float32") = R.tile(x, repeats=[4, 2])
                gv: R.Tuple(R.Tensor((4, 6), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, dtype=torch.float32),)
    verify_model(Tile1(), example_args, {}, expected1)
    verify_model(Tile2(), example_args, {}, expected2)


def test_transpose():
    class Transpose(Module):
        def forward(self, x):
            return x.transpose(1, 3)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 4, 3, 2), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 4, 3, 2), dtype="float32") = R.permute_dims(x, axes=[0, 3, 2, 1])
                gv: R.Tuple(R.Tensor((1, 4, 3, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(Transpose(), example_args, {}, expected1)


def test_unsqueeze():
    class Unsqueeze1(Module):
        def forward(self, input):
            return input.unsqueeze(1)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 1, 3, 10, 10), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 1, 3, 10, 10), dtype="float32") = R.expand_dims(input_1, 1)
                gv: R.Tuple(R.Tensor((1, 1, 3, 10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class Unsqueeze2(Module):
        def forward(self, input):
            return input.unsqueeze(-1)

    @tvm_tirx.script.ir_module
    class expected2:
        @R.function
        def main(input_1: R.Tensor((1, 3, 10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 3, 10, 10, 1), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10, 1), dtype="float32") = R.expand_dims(input_1, -1)
                gv: R.Tuple(R.Tensor((1, 3, 10, 10, 1), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)

    verify_model(Unsqueeze1(), example_args, {}, expected1)
    verify_model(Unsqueeze2(), example_args, {}, expected2)


def test_view():
    class View(Module):
        def forward(self, x):
            return x.view(2, 12)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 12), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, (2, 12))
                gv: R.Tuple(R.Tensor((2, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(View(), example_args, {}, expected1)


def test_as_strided():
    class AsStrided(Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, (3, 2, 2), (4, 2, 1))

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 2, 2), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3, 2, 2), dtype="float32") = R.reshape(x, (3, 2, 2))
                gv: R.Tuple(R.Tensor((3, 2, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    class AsStridedNonContiguous(Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, (2, 2, 2), (6, 3, 1))

    class AsStridedWithStorageOffset(Module):
        def forward(self, x):
            return torch.ops.aten.as_strided.default(x, (2, 2), (2, 1), 1)

    example_args = (torch.randn(2, 2, 3, dtype=torch.float32),)
    verify_model(AsStrided(), example_args, {}, Expected)

    exported = export(AsStridedNonContiguous(), args=example_args)
    with pytest.raises(AssertionError, match="non-contiguous stride"):
        from_exported_program(exported)

    example_args = (torch.randn(2, 2, dtype=torch.float32),)
    exported = export(AsStridedWithStorageOffset(), args=example_args)
    with pytest.raises(AssertionError, match="storage_offset"):
        from_exported_program(exported)


def test_arange():
    class Arange(Module):
        def forward(self, input):
            return torch.arange(0, 20, dtype=torch.int32)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((20,), dtype="int32")
        ):
            with R.dataflow():
                lv: R.Tensor((20,), dtype="int32") = R.arange(0, 20, 1, dtype="int32")
                gv: R.Tuple(R.Tensor((20,), dtype="int32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Arange(), example_args, {}, Expected)


def test_hamming_window():
    class HammingWindow(Module):
        def forward(self, input):
            return torch.hamming_window(20, True, dtype=torch.float32)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((20,), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((20,), dtype="float32") = R.hamming_window(
                    R.prim_value(20),
                    R.prim_value(True),
                    R.prim_value(T.float64(0.54000000000000004)),
                    R.prim_value(T.float64(0.46000000000000002)),
                    dtype="float32",
                )
                gv: R.Tuple(R.Tensor((20,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(HammingWindow(), example_args, {}, Expected)


def test_clone():
    class Clone(Module):
        def forward(self, input):
            return torch.clone(input)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            with R.dataflow():
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (input,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Clone(), example_args, {}, Expected)


def test_empty():
    class Empty(Module):
        def forward(self, input):
            return torch.empty((10, 10), dtype=torch.float32)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.zeros(
                    R.shape([10, 10]), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Empty(), example_args, {}, Expected)


def test_fill():
    class Fill(Module):
        def forward(self, input: torch.Tensor):
            return torch.fill(input, 1.5)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((10, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((10, 10), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.full_like(
                    input, R.const(1.5, "float32")
                )
                gv: R.Tuple(R.Tensor((10, 10), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(10, 10, dtype=torch.float32),)
    verify_model(Fill(), example_args, {}, Expected)


def test_fill_inplace():
    class FillInplace(Module):
        def forward(self, input: torch.Tensor):
            input.fill_(42.0)
            return input

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.full_like(input, R.const(42.0, "float32"))
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32),)
    verify_model(FillInplace(), example_args, {}, Expected)


def test_masked_fill():
    class Masked_Fill(Module):
        def forward(self, input: torch.Tensor, mask: torch.Tensor):
            return torch.masked_fill(input, mask, 0)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((128, 128), dtype="float32"), mask: R.Tensor((128, 128), dtype="bool")
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.const(0.0, "float32")
                lv1: R.Tensor((128, 128), dtype="float32") = R.where(mask, lv, input)
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(128, 128, dtype=torch.float32),
        torch.testing.make_tensor((128, 128), dtype=torch.bool, device="cpu"),
    )
    verify_model(Masked_Fill(), example_args, {}, Expected)


def test_masked_fill_inplace():
    class Masked_Fill_Inplace(Module):
        def forward(self, input: torch.Tensor, mask: torch.Tensor):
            return input.masked_fill_(mask, 1.5)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((128, 128), dtype="float32"), mask: R.Tensor((128, 128), dtype="bool")
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.const(1.5, "float32")
                lv1: R.Tensor((128, 128), dtype="float32") = R.where(mask, lv, input)
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(128, 128, dtype=torch.float32),
        torch.testing.make_tensor((128, 128), dtype=torch.bool, device="cpu"),
    )
    verify_model(Masked_Fill_Inplace(), example_args, {}, Expected)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_dynamic_scalar_masked_fill_operations():
    class DynamicMaskedFills(torch.nn.Module):
        def forward(self, x):
            mask = x > 0
            filled = x.masked_fill(mask, x.shape[0])
            filled_inplace = x.clone()
            filled_inplace.masked_fill_(mask, x.shape[1])
            return filled, filled_inplace

    rows = torch.export.Dim("rows", min=1, max=8)
    columns = torch.export.Dim("columns", min=1, max=8)
    example_args = (torch.randn(3, 4, dtype=torch.float32),)
    exported_program = export(
        DynamicMaskedFills(),
        args=example_args,
        dynamic_shapes={"x": {0: rows, 1: columns}},
    )
    mod = from_exported_program(exported_program, run_ep_decomposition=False)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    for shape in ((3, 4), (5, 2)):
        torch_input = torch.randn(shape, dtype=torch.float32)
        expected = DynamicMaskedFills()(torch_input)
        actual = vm["main"](tvm_tirx.runtime.tensor(torch_input.numpy()))
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_array_equal(actual_value.numpy(), expected_value.numpy())


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize(
    ("fill_value", "dtype"),
    [
        (1 << 40, torch.int64),
        (1.0 + 2**-40, torch.float64),
    ],
)
def test_fill_and_masked_fill_python_value_uses_input_dtype(fill_value, dtype):
    class MaskedFills(torch.nn.Module):
        def forward(self, x, mask):
            filled_all = torch.fill(x, fill_value)
            filled_all_inplace = x.clone()
            filled_all_inplace.fill_(fill_value)
            filled = x.masked_fill(mask, fill_value)
            filled_inplace = x.clone()
            filled_inplace.masked_fill_(mask, fill_value)
            return filled_all, filled_all_inplace, filled, filled_inplace

    example_args = (
        torch.arange(6, dtype=dtype).reshape(2, 3),
        torch.tensor([[True, False, True], [False, True, False]]),
    )
    exported_program = export(MaskedFills(), args=example_args)
    mod = from_exported_program(exported_program, run_ep_decomposition=False)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    expected = MaskedFills()(*example_args)
    actual = vm["main"](*(tvm_tirx.runtime.tensor(arg.numpy()) for arg in example_args))
    for actual_value, expected_value in zip(actual, expected):
        assert actual_value.numpy().dtype == expected_value.numpy().dtype
        np.testing.assert_array_equal(actual_value.numpy(), expected_value.numpy())


def test_masked_select():
    class MaskedSelect(Module):
        def forward(self, data: torch.Tensor, mask: torch.Tensor):
            return torch.masked_select(data, mask)

    u0 = T.dynamic("u0", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((2, 3), dtype="float32"), mask: R.Tensor((2, 3), dtype="bool")
        ) -> R.Tuple(R.Tensor(dtype="float32", ndim=1)):
            R.func_attr({"tir_var_lower_bound": {"u0": 0}, "tir_var_upper_bound": {"u0": 6}})
            with R.dataflow():
                lv: R.Tensor((6,), dtype="float32") = R.reshape(data, R.shape([6]))
                lv1: R.Tensor((6,), dtype="bool") = R.reshape(mask, R.shape([6]))
                lv2: R.Tensor(dtype="int64", ndim=2) = R.nonzero(lv1)
                lv3: R.Tensor((1, u0), dtype="int64") = R.match_cast(
                    lv2, R.Tensor((1, u0), dtype="int64")
                )
                lv4: R.Tensor((u0,), dtype="int64") = R.squeeze(lv3, axis=[0])
                lv5: R.Tensor((u0,), dtype="float32") = R.take(lv, lv4, axis=0, mode="fast")
                lv6: T.bool = u0 >= 0
                lv7: T.bool = u0 <= 6
                gv: R.Tuple(R.Tensor((u0,), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.tensor([[True, False, True], [False, True, False]]),
    )
    verify_model(MaskedSelect(), example_args, {}, Expected)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_masked_select_numerically():
    class MaskedSelect(Module):
        def forward(self, data: torch.Tensor, mask: torch.Tensor):
            return torch.masked_select(data, mask)

    example_args = (
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
        torch.tensor([[True, False, True], [False, True, False]]),
    )
    verify_model_numerically(MaskedSelect(), example_args)


def test_new_ones():
    class NewOnes(Module):
        def forward(self, x):
            return x.new_ones(1, 2, 3)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 2, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 2, 3), dtype="float32")
        ):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 2, 3), dtype="float32") = R.full(
                    (1, 2, 3), R.const(1, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((1, 2, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, dtype=torch.float32),)
    verify_model(NewOnes(), example_args, {}, expected1)


def test_new_zeros():
    class NewZeros(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros(1, 128, 128)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((1, 128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 128, 128), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 128, 128), dtype="float32") = R.full(
                    R.shape([1, 128, 128]), R.const(0, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((1, 128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 128, 128, dtype=torch.float32),)
    verify_model(NewZeros(), example_args, {}, expected1)


def test_copy():
    class CopyBroadcast(Module):
        def forward(self, x, src):
            x.copy_(src)
            return x

    @tvm_tirx.script.ir_module
    class expected_copy:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), src: R.Tensor((), dtype="int64")) -> R.Tuple(
            R.Tensor((2, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.astype(src, dtype="float32")
                lv1: R.Tensor((2, 3), dtype="float32") = R.broadcast_to(lv, (2, 3))
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.zeros(2, 3, dtype=torch.float32), torch.tensor(1, dtype=torch.int64))
    verify_model(CopyBroadcast(), example_args, {}, expected_copy)


def test_to_copy():
    class ToHalf(Module):
        def forward(self, x):
            return x.half()

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((1, 2, 3, 4), dtype="float16")
        ):
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 4), dtype="float16") = R.astype(x, dtype="float16")
                gv: R.Tuple(R.Tensor((1, 2, 3, 4), dtype="float16")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, 4, dtype=torch.float32),)
    verify_model(ToHalf(), example_args, {}, Expected)


def test_keep_params():
    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    @tvm_tirx.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            conv_weight: R.Tensor((6, 3, 7, 7), dtype="float32"),
            conv_bias: R.Tensor((6,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")):
            R.func_attr({"num_input": 1})
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    conv_weight,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 6, 1, 1), dtype="float32") = R.reshape(conv_bias, [1, 6, 1, 1])
                lv3: R.Tensor((1, 6, 4, 4), dtype="float32") = R.add(lv1, lv2)
                gv: R.Tuple(R.Tensor((1, 6, 4, 4), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    from tvm.relax.frontend import detach_params

    example_args = (torch.randn(1, 3, 10, 10, dtype=torch.float32),)
    model = Conv2D1()

    exported_program = torch.export.export(model, example_args)
    mod = from_exported_program(exported_program, keep_params_as_input=True)
    mod, params = detach_params(mod)
    tvm_tirx.ir.assert_structural_equal(mod, expected1)
    func = mod["main"]
    params = params["main"]

    assert len(params) == len(func.params) - 1
    for param_var, param_tensor in zip(func.params[1:], params):
        assert tuple(x.value for x in param_var.ty.shape.values) == param_tensor.shape
        assert param_var.ty.dtype == param_tensor.dtype

    tvm_tirx.testing.assert_allclose(params[0].numpy(), model.conv.weight.detach().detach().numpy())
    tvm_tirx.testing.assert_allclose(params[1].numpy(), model.conv.bias.detach().detach().numpy())


def test_unwrap_unit_return_tuple():
    class Identity(Module):
        def forward(self, x):
            return (x,)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(inp_0: R.Tensor((256, 256), dtype="float32")) -> R.Tensor(
            (256, 256), dtype="float32"
        ):
            with R.dataflow():
                gv: R.Tensor((256, 256), dtype="float32") = inp_0
                R.output(gv)
            return gv

    example_args = (torch.randn(256, 256, dtype=torch.float32),)
    verify_model(Identity(), example_args, {}, Expected, unwrap_unit_return_tuple=True)


def test_no_bind_return_tuple():
    class Identity(Module):
        def forward(self, x, y):
            return (x, y)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((256, 256), dtype="float32"),
            inp_1: R.Tensor((256, 256), dtype="float32"),
        ) -> R.Tuple(R.Tensor((256, 256), dtype="float32"), R.Tensor((256, 256), dtype="float32")):
            with R.dataflow():
                gv: R.Tensor((256, 256), dtype="float32") = inp_0
                gv1: R.Tensor((256, 256), dtype="float32") = inp_1
                R.output(gv, gv1)
            return (gv, gv1)

    example_args = (
        torch.randn(256, 256, dtype=torch.float32),
        torch.randn(256, 256, dtype=torch.float32),
    )
    verify_model(Identity(), example_args, {}, Expected, no_bind_return_tuple=True)


def test_register_buffer():
    class ModelWithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("my_buffer", torch.randn(3, 4), persistent=False)

        def forward(self, x):
            return x + self.my_buffer

    example_args = (torch.randn(2, 3, 4),)
    ep = export(ModelWithBuffer(), args=example_args)
    # Just verify that import works.
    from_exported_program(ep)


def test_custom_op():
    class AddOp(Module):
        def forward(self, x, y):
            return torch.ops.aten.add.Tensor(x, y)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5,), dtype="float32"),
            y: R.Tensor((5,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.subtract(x, y)
                gv: R.Tuple(R.Tensor((5,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    from tvm.relax.frontend.torch.exported_program_translator import (
        ExportedProgramImporter,
    )

    def custom_add_converter(node: torch.fx.Node, self: ExportedProgramImporter) -> relax.Var:
        x = self.env[node.args[0]]
        y = self.env[node.args[1]]

        return self.block_builder.emit(R.subtract(x, y))

    example_args = (torch.randn(5, dtype=torch.float32), torch.randn(5, dtype=torch.float32))
    verify_model(
        AddOp(), example_args, {}, Expected, custom_convert_map={"add.Tensor": custom_add_converter}
    )


def test_empty_like():
    class EmptyLike(Module):
        def forward(self, data):
            return torch.empty_like(data)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((5,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.zeros(R.shape([5]), dtype="float32")
                gv: R.Tuple(R.Tensor((5,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, dtype=torch.float32),)

    verify_model(EmptyLike(), example_args, {}, Expected)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_one_hot():
    class OneHot(Module):
        def forward(self, indices):
            return torch.nn.functional.one_hot(indices, num_classes=10)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            indices: R.Tensor((5,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((5, 10), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((5, 10), dtype="int64") = R.one_hot(
                    indices, R.prim_value(1), R.prim_value(0), depth=10, axis=-1
                )
                gv: R.Tuple(R.Tensor((5, 10), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.tensor([0, 1, 5, 8, 9], dtype=torch.int64),)

    verify_model(OneHot(), example_args, {}, Expected, run_ep_decomposition=False)
    verify_model_numerically(OneHot(), example_args)


def test_one_hot_invalid_num_classes():
    class OneHot(Module):
        def forward(self, indices):
            return torch.nn.functional.one_hot(indices, num_classes=0)

    example_args = (torch.randint(0, 5, (5,), dtype=torch.int64),)
    exported_program = export(OneHot(), args=example_args)

    # With the default decomposition, one_hot is rewritten to arange/equal/astype and never
    # reaches this converter. Without it, the non-positive num_classes must be rejected by
    # the frontend instead of failing an internal `depth > 0` check in relax.op.one_hot.
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        from_exported_program(exported_program, run_ep_decomposition=False)


def test_ones_like():
    class OnesLike(Module):
        def forward(self, input):
            return torch.ones_like(input)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((128, 128), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.full_like(
                    input, R.const(1.0, "float32")
                )
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)

    verify_model(OnesLike(), example_args, {}, Expected)


def test_zero_inplace():
    class ZeroInplace(Module):
        def forward(self, input):
            return input.zero_()

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((128, 128), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.full_like(
                    input, R.const(0.0, "float32")
                )
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)

    verify_model(ZeroInplace(), example_args, {}, Expected)


def test_zeros():
    class Zeros(Module):
        def forward(self, input):
            return torch.zeros(5, 2)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 2), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((5, 2), dtype="float32") = R.full(
                    R.shape([5, 2]), R.const(0.0, "float32"), dtype="float32"
                )
                gv: R.Tuple(R.Tensor((5, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)

    verify_model(Zeros(), example_args, {}, Expected)


def test_zeros_like():
    class ZerosLike(Module):
        def forward(self, input):
            return torch.zeros_like(input)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((128, 128), dtype="float32")) -> R.Tuple(
            R.Tensor((128, 128), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float32") = R.full_like(
                    input, R.const(0.0, "float32")
                )
                gv: R.Tuple(R.Tensor((128, 128), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.rand(128, 128, dtype=torch.float32),)
    verify_model(ZerosLike(), example_args, {}, Expected)


def test_randn():
    class Randn(Module):
        def forward(self, input):
            return input + torch.randn(5, 3)

    example_args = (torch.rand(5, 3, dtype=torch.float32),)
    exported_program = export(Randn(), args=example_args)
    mod = from_exported_program(exported_program)
    func = mod["main"]
    ret_ty = func.ret_ty
    assert ret_ty.fields[0].shape[0] == 5
    assert ret_ty.fields[0].shape[1] == 3
    assert ret_ty.fields[0].dtype == "float32"


def test_randn_like():
    class RandnLike(Module):
        def forward(self, input):
            return input + torch.randn_like(input)

    example_args = (torch.rand(4, 6, dtype=torch.float32),)
    exported_program = export(RandnLike(), args=example_args)
    mod = from_exported_program(exported_program)
    func = mod["main"]
    ret_ty = func.ret_ty
    assert ret_ty.fields[0].shape[0] == 4
    assert ret_ty.fields[0].shape[1] == 6
    assert ret_ty.fields[0].dtype == "float32"


def test_type_as():
    class TypeAs(Module):
        def forward(self, input, other):
            return input.type_as(other)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((128, 128), dtype="float32"),
            other: R.Tensor((128, 128), dtype="float16"),
        ) -> R.Tuple(R.Tensor((128, 128), dtype="float16")):
            with R.dataflow():
                lv: R.Tensor((128, 128), dtype="float16") = R.astype(input, dtype="float16")
                gv: R.Tuple(R.Tensor((128, 128), dtype="float16")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.rand(128, 128, dtype=torch.float32),
        torch.rand(128, 128, dtype=torch.float16),
    )

    verify_model(TypeAs(), example_args, {}, Expected)


def test_select():
    class Select(Module):
        def forward(self, input):
            return torch.select(input, 0, 1)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="float32") = R.take(inp_0, R.const(1, "int64"), axis=0)
                gv: R.Tuple(R.Tensor((3,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32),)

    verify_model(Select(), example_args, {}, Expected)


def test_unflatten():
    class Unflatten(Module):
        def forward(self, input):
            return torch.ops.aten.unflatten(input, 1, (3, 5))

    class Unflatten1(Module):
        def forward(self, input):
            return torch.ops.aten.unflatten(input, -2, (3, 5))

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((2, 15, 7), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 5, 7), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 5, 7), dtype="float32") = R.reshape(inp_0, [2, 3, 5, 7])
                gv: R.Tuple(R.Tensor((2, 3, 5, 7), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 15, 7, dtype=torch.float32),)

    verify_model(Unflatten(), example_args, {}, Expected)
    verify_model(Unflatten1(), example_args, {}, Expected)


def test_gather():
    class Gather(Module):
        def __init__(self, axis):
            super().__init__()
            self.axis = axis

        def forward(self, data, indices):
            return torch.gather(data, self.axis, indices)

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.randint(0, 3, (2, 3), dtype=torch.int64),
    )
    for axis in (0, 1, -1, -2):

        @tvm_tirx.script.ir_module
        class Expected:
            @R.function
            def main(
                inp_0: R.Tensor((2, 3), dtype="float32"),
                inp_1: R.Tensor((2, 3), dtype="int64"),
            ) -> R.Tuple(R.Tensor((2, 3), dtype="float32")):
                with R.dataflow():
                    lv: R.Tensor((2, 3), dtype="float32") = R.gather_elements(
                        inp_0, inp_1, axis=axis
                    )
                    gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv,)
                    R.output(gv)
                return gv

        verify_model(Gather(axis), example_args, {}, Expected)


def test_index_put():
    # 1D input
    class IndexPut1D(Module):
        def forward(self, data, indices_0, values):
            indices_tuple = (indices_0,)
            return data.index_put_(indices_tuple, values, accumulate=False)

    example_args_1d = (
        torch.randn(64, dtype=torch.float32),
        torch.randint(0, 64, (128,), dtype=torch.int64),
        torch.randn(128, dtype=torch.float32),
    )

    @I.ir_module
    class Expected1D:
        @R.function
        def main(
            data: R.Tensor((64,), dtype="float32"),
            indices_0: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((64,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64,), dtype="float32") = R.index_put(
                    data, (indices_0,), values, accumulate=False
                )
                gv: R.Tuple(R.Tensor((64,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # 2D input
    class IndexPut2D(Module):
        def forward(self, data, indices_0, indices_1, values):
            indices_tuple = (indices_0, indices_1)
            return data.index_put_(indices_tuple, values, accumulate=False)

    example_args_2d = (
        torch.randn(32, 64, dtype=torch.float32),
        torch.randint(0, 32, (128,), dtype=torch.int64),
        torch.randint(0, 64, (128,), dtype=torch.int64),
        torch.randn(128, dtype=torch.float32),
    )

    @I.ir_module
    class Expected2D:
        @R.function
        def main(
            data: R.Tensor((32, 64), dtype="float32"),
            indices_0: R.Tensor((128,), dtype="int64"),
            indices_1: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((32, 64), dtype="float32") = R.index_put(
                    data, (indices_0, indices_1), values, accumulate=False
                )
                gv: R.Tuple(R.Tensor((32, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # 5D input
    class IndexPut5D(Module):
        def forward(self, data, indices_0, indices_1, indices_2, indices_3, indices_4, values):
            indices_tuple = (indices_0, indices_1, indices_2, indices_3, indices_4)
            return data.index_put_(indices_tuple, values, accumulate=False)

    example_args_5d = (
        torch.randn(4, 8, 16, 32, 64, dtype=torch.float32),
        torch.randint(0, 4, (128,), dtype=torch.int64),
        torch.randint(0, 8, (128,), dtype=torch.int64),
        torch.randint(0, 16, (128,), dtype=torch.int64),
        torch.randint(0, 32, (128,), dtype=torch.int64),
        torch.randint(0, 64, (128,), dtype=torch.int64),
        torch.randn(128, dtype=torch.float32),
    )

    @I.ir_module
    class Expected5D:
        @R.function
        def main(
            data: R.Tensor((4, 8, 16, 32, 64), dtype="float32"),
            indices_0: R.Tensor((128,), dtype="int64"),
            indices_1: R.Tensor((128,), dtype="int64"),
            indices_2: R.Tensor((128,), dtype="int64"),
            indices_3: R.Tensor((128,), dtype="int64"),
            indices_4: R.Tensor((128,), dtype="int64"),
            values: R.Tensor((128,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 8, 16, 32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 8, 16, 32, 64), dtype="float32") = R.index_put(
                    data,
                    (indices_0, indices_1, indices_2, indices_3, indices_4),
                    values,
                    accumulate=False,
                )
                gv: R.Tuple(R.Tensor((4, 8, 16, 32, 64), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    # 2D input with multi-dimensional index (broadcasting)
    # This tests the multi-dimensional index support with broadcasting
    class IndexPutBroadcast1D(Module):
        def forward(self, data, indices_1):
            indices_0 = torch.arange(data.shape[0]).unsqueeze(1)
            values = torch.ones(data.shape[0], len(indices_1), dtype=data.dtype)
            return data.index_put_((indices_0, indices_1), values, accumulate=False)

    example_args_broadcast1 = (
        torch.randn(32, 64, dtype=torch.float32),
        torch.randint(0, 64, (10,), dtype=torch.int64),
    )

    @I.ir_module
    class ExpectedBroadcast1D:
        @R.function
        def main(
            data: R.Tensor((32, 64), dtype="float32"),
            indices_1: R.Tensor((10,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((32,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(32), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((32, 1), dtype="int64") = R.expand_dims(lv, axis=[1])
                lv2: R.Tensor((32, 10), dtype="float32") = R.full(
                    R.shape([32, 10]), R.const(1.0, "float32"), dtype="float32"
                )
                lv3: R.Tensor((32, 64), dtype="float32") = R.index_put(
                    data, (lv1, indices_1), lv2, accumulate=False
                )
                gv: R.Tuple(R.Tensor((32, 64), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    # 2D input with multi-dimensional index (second position)
    class IndexPutBroadcast2D(Module):
        def forward(self, data, indices_0):
            indices_1 = torch.arange(data.shape[1]).unsqueeze(1)
            values = torch.ones(len(indices_0), data.shape[1], dtype=data.dtype)
            return data.index_put_((indices_0, indices_1), values, accumulate=False)

    example_args_broadcast2 = (
        torch.randn(32, 64, dtype=torch.float32),
        torch.randint(0, 32, (10,), dtype=torch.int64),
    )

    @I.ir_module
    class ExpectedBroadcast2D:
        @R.function
        def main(
            data: R.Tensor((32, 64), dtype="float32"),
            indices_0: R.Tensor((10,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((64,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(64), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((64, 1), dtype="int64") = R.expand_dims(lv, axis=[1])
                lv2: R.Tensor((10, 64), dtype="float32") = R.full(
                    R.shape([10, 64]), R.const(1.0, "float32"), dtype="float32"
                )
                lv3: R.Tensor((32, 64), dtype="float32") = R.index_put(
                    data, (indices_0, lv1), lv2, accumulate=False
                )
                gv: R.Tuple(R.Tensor((32, 64), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    # 3D input with mixed 1D and 2D indices
    class IndexPutBroadcast3D(Module):
        def forward(self, data, indices_1):
            indices_0 = torch.arange(data.shape[0]).unsqueeze(1)
            indices_2 = torch.arange(data.shape[2]).unsqueeze(1)
            values = torch.ones(data.shape[0], len(indices_1), data.shape[2], dtype=data.dtype)
            return data.index_put_((indices_0, indices_1, indices_2), values, accumulate=False)

    example_args_broadcast3d = (
        torch.randn(16, 32, 64, dtype=torch.float32),
        torch.randint(0, 32, (10,), dtype=torch.int64),
    )

    @I.ir_module
    class ExpectedBroadcast3D:
        @R.function
        def main(
            data: R.Tensor((16, 32, 64), dtype="float32"),
            indices_1: R.Tensor((10,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((16, 32, 64), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((16,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(16), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((16, 1), dtype="int64") = R.expand_dims(lv, axis=[1])
                lv2: R.Tensor((64,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(64), R.prim_value(1), dtype="int64"
                )
                lv3: R.Tensor((64, 1), dtype="int64") = R.expand_dims(lv2, axis=[1])
                lv4: R.Tensor((16, 10, 64), dtype="float32") = R.full(
                    R.shape([16, 10, 64]), R.const(1.0, "float32"), dtype="float32"
                )
                lv5: R.Tensor((16, 32, 64), dtype="float32") = R.index_put(
                    data, (lv1, indices_1, lv3), lv4, accumulate=False
                )
                gv: R.Tuple(R.Tensor((16, 32, 64), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    # batched indexing with slice (e.g., M[:, rows, cols] = x)
    class IndexPutBatchedWithNone(Module):
        def forward(self, x):
            B = x.size(0)
            M = torch.zeros(B, 11, 11)
            rows = torch.arange(10)
            cols = rows + 1
            M[:, rows, cols] = x  # Batched index assignment
            return M

    example_args_batched_none = (torch.randn(2, 10, dtype=torch.float32),)

    @I.ir_module
    class ExpectedBatchedWithNone:
        @R.function
        def main(x: R.Tensor((2, 10), dtype="float32")) -> R.Tuple(
            R.Tensor((2, 11, 11), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 11, 11), dtype="float32") = R.full(
                    R.shape([2, 11, 11]), R.const(0.0, "float32"), dtype="float32"
                )
                lv1: R.Tensor((10,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(10), R.prim_value(1), dtype="int64"
                )
                lv2: R.Tensor((10,), dtype="int64") = R.add(lv1, R.const(1, "int64"))
                lv3: R.Tensor((2,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(2), R.prim_value(1), dtype="int64"
                )
                lv4: R.Tensor((2, 1), dtype="int64") = R.reshape(lv3, R.shape([2, 1]))
                lv5: R.Tensor((2, 11, 11), dtype="float32") = R.index_put(
                    lv, (lv4, lv1, lv2), x, accumulate=False
                )
                gv: R.Tuple(R.Tensor((2, 11, 11), dtype="float32")) = (lv5,)
                R.output(gv)
            return gv

    # Run verification for each case
    verify_model(IndexPut1D(), example_args_1d, {}, Expected1D)
    verify_model(IndexPut2D(), example_args_2d, {}, Expected2D)
    verify_model(IndexPut5D(), example_args_5d, {}, Expected5D)
    verify_model(IndexPutBroadcast1D(), example_args_broadcast1, {}, ExpectedBroadcast1D)
    verify_model(IndexPutBroadcast2D(), example_args_broadcast2, {}, ExpectedBroadcast2D)
    verify_model(IndexPutBroadcast3D(), example_args_broadcast3d, {}, ExpectedBroadcast3D)
    verify_model(IndexPutBatchedWithNone(), example_args_batched_none, {}, ExpectedBatchedWithNone)


def test_index_put_with_tuple_output():
    class IndexPutTupleOutput(Module):
        def forward(self, x, buf, idx):
            values = x
            buf[..., idx, idx] = values
            return x[..., 1], buf

    example_args = (
        torch.ones(2, 3, 5, dtype=torch.float32),
        torch.zeros(2, 3, 5, 5, dtype=torch.float32),
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
    )

    exported_program = export(IndexPutTupleOutput(), args=example_args)
    mod = from_exported_program(exported_program)

    ret_ty = mod["main"].ret_ty
    assert isinstance(ret_ty, relax.TupleType)

    tensor_fields = [f for f in ret_ty.fields if isinstance(f, relax.TensorType)]
    assert len(tensor_fields) >= 2

    assert any(
        len(f.shape) == 4 and int(f.shape[-2]) == 5 and int(f.shape[-1]) == 5 for f in tensor_fields
    )


def test_index_put_mutation_through_alias_regression():
    class IndexPutAlias(Module):
        def forward(self, x, idx, values):
            y = torch.ops.aten.alias.default(x)
            y[idx] = values
            return x, y

    example_args = (
        torch.zeros(5, dtype=torch.float32),
        torch.tensor([1, 3], dtype=torch.int64),
        torch.tensor([2.0, 4.0], dtype=torch.float32),
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5,), dtype="float32"),
            idx: R.Tensor((2,), dtype="int64"),
            values: R.Tensor((2,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((5,), dtype="float32"),
            R.Tensor((5,), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.index_put(
                    x, (idx,), values, accumulate=False
                )
                # Mutation outputs introduced by functionalization are dropped;
                # only the user outputs (x, y) remain.
                gv: R.Tuple(
                    R.Tensor((5,), dtype="float32"),
                    R.Tensor((5,), dtype="float32"),
                ) = (
                    lv,
                    lv,
                )
                R.output(gv)
            return gv

    verify_model(IndexPutAlias(), example_args, {}, Expected)


def test_flip():
    class Flip0(Module):
        def forward(self, data):
            return torch.flip(data, [0])

    class Flip1(Module):
        def forward(self, data):
            return torch.flip(data, [1])

    @tvm_tirx.script.ir_module
    class Expected0:
        @R.function
        def main(
            inp_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.flip(inp_0, axis=0)
                gv: R.Tuple(R.Tensor((2, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 2), dtype="float32") = R.flip(inp_0, axis=1)
                gv: R.Tuple(R.Tensor((2, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 2, dtype=torch.float32),)

    verify_model(Flip0(), example_args, {}, Expected0)
    verify_model(Flip1(), example_args, {}, Expected1)


def test_flip_multi_axis():
    class FlipMulti(Module):
        def forward(self, data):
            return torch.flip(data, [0, 1])

    class FlipNegMulti(Module):
        def forward(self, data):
            return torch.flip(data, dims=[-1, -2])

    @tvm_tirx.script.ir_module
    class ExpectedMulti:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.flip(inp_0, axis=0)
                lv1: R.Tensor((2, 3), dtype="float32") = R.flip(lv, axis=1)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class ExpectedNegMulti:
        @R.function
        def main(
            inp_0: R.Tensor((2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.flip(inp_0, axis=-1)
                lv1: R.Tensor((2, 3), dtype="float32") = R.flip(lv, axis=-2)
                gv: R.Tuple(R.Tensor((2, 3), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, dtype=torch.float32),)

    verify_model(FlipMulti(), example_args, {}, ExpectedMulti)
    verify_model(FlipNegMulti(), example_args, {}, ExpectedNegMulti)


def test_take():
    class Take(Module):
        def forward(self, data, indices):
            return torch.take(data, indices)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((5,), dtype="float32"),
            indices: R.Tensor((3,), dtype="int64"),
        ) -> R.Tuple(R.Tensor((3,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="float32") = R.reshape(data, R.shape([5]))
                lv1: R.Tensor((3,), dtype="float32") = R.take(lv, indices, axis=0, mode="fast")
                gv: R.Tuple(R.Tensor((3,), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(5, dtype=torch.float32),
        torch.randint(0, 5, (3,), dtype=torch.int64),
    )

    verify_model(Take(), example_args, {}, Expected)


def test_any():
    class AnyAten(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.any(x, dim=1)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="bool"),
        ) -> R.Tuple(R.Tensor((2,), dtype="bool")):
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="int8") = relax.op.astype(x, dtype="int8")
                lv2: R.Tensor((2,), dtype="int8") = relax.op.max(lv, axis=1, keepdims=False)
                lv3: R.Tensor((2,), dtype="bool") = relax.op.astype(lv2, dtype="bool")
                gv: R.Tuple(R.Tensor((2,), dtype="bool")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (torch.tensor([[0, 0, 0], [0, 1, 0]], dtype=torch.bool),)
    verify_model(AnyAten(), example_args, {}, Expected)


def test_std():
    # torch.std(x) defaults to correction=1 (Bessel); decomposes to var.correction + sqrt.
    class Std(Module):
        def forward(self, x):
            return torch.std(x)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.variance(x, axis=None, keepdims=False)
                lv1: R.Tensor((), dtype="float32") = R.multiply(lv, R.const(15.0 / 14.0, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sqrt(lv1)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv2,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Std(), example_args, {}, Expected)


def test_var():
    # torch.var(x) defaults to correction=1 (Bessel).
    class Var(Module):
        def forward(self, x):
            return torch.var(x)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.variance(x, axis=None, keepdims=False)
                lv1: R.Tensor((), dtype="float32") = R.multiply(lv, R.const(15.0 / 14.0, "float32"))
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Var(), example_args, {}, Expected)


def test_var_correction():
    class VarCorrection2(Module):
        def forward(self, x):
            return torch.var(x, dim=-1, correction=2)

    class VarCorrection0(Module):
        def forward(self, x):
            return torch.var(x, dim=1, correction=0)

    @tvm_tirx.script.ir_module
    class Expected2:
        @R.function
        def main(
            x: R.Tensor((2, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2,), dtype="float32") = R.variance(x, axis=[-1], keepdims=False)
                lv1: R.Tensor((2,), dtype="float32") = R.multiply(lv, R.const(5.0 / 3.0, "float32"))
                gv: R.Tuple(R.Tensor((2,), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class Expected0:
        @R.function
        def main(
            x: R.Tensor((2, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2,), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2,), dtype="float32") = R.variance(x, axis=[1], keepdims=False)
                gv: R.Tuple(R.Tensor((2,), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 5, dtype=torch.float32),)
    verify_model(VarCorrection2(), example_args, {}, Expected2)
    verify_model(VarCorrection0(), example_args, {}, Expected0)


@pytest.mark.parametrize(
    "torch_dtype,relax_dtype",
    [(torch.float32, "float32"), (torch.bool, "bool")],
)
def test_prod(torch_dtype, relax_dtype):
    class Prod(Module):
        def forward(self, x):
            return torch.prod(x, dtype=torch_dtype)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((5, 3), dtype=relax_dtype),
        ) -> R.Tuple(R.Tensor((), dtype=relax_dtype)):
            with R.dataflow():
                lv: R.Tensor((), dtype=relax_dtype) = R.prod(x, axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype=relax_dtype)) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.ones(5, 3, dtype=torch_dtype),)
    verify_model(Prod(), example_args, {}, Expected)


def test_cumprod():
    class Cumprod(Module):
        def forward(self, x):
            return torch.cumprod(x, 0)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.cumprod(inp_0, axis=0, exclusive=False)
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_input = torch.randn(5, 3, dtype=torch.float32)
    verify_model(Cumprod(), (example_input,), {}, Expected)


def test_where():
    class Where(Module):
        def forward(self, condition, x, y):
            return torch.where(condition, x, y)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            inp_0: R.Tensor((5, 3), dtype="bool"),
            inp_1: R.Tensor((5, 3), dtype="float32"),
            inp_2: R.Tensor((5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((5, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.where(inp_0, inp_1, inp_2)
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    condition = torch.testing.make_tensor((5, 3), dtype=torch.bool, device="cpu")
    x = torch.randn(5, 3, dtype=torch.float32)
    y = torch.randn(5, 3, dtype=torch.float32)

    verify_model(Where(), (condition, x, y), {}, Expected)


def test_bucketize():
    class Bucketize(Module):
        def forward(self, input_tensor, boundaries):
            return torch.bucketize(input_tensor, boundaries)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((20,), dtype="int64"), boundaries: R.Tensor((10,), dtype="int64")
        ) -> R.Tuple(R.Tensor((20,), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((20,), dtype="int64") = R.bucketize(
                    input, boundaries, out_int32=False, right=False
                )
                gv: R.Tuple(R.Tensor((20,), dtype="int64")) = (lv,)
                R.output(gv)
            return gv

    input_tensor = torch.arange(0, 20)
    boundaries = torch.arange(0, 20, 2)

    verify_model(Bucketize(), (input_tensor, boundaries), {}, Expected)


@pytest.mark.parametrize("right", [False, True])
@pytest.mark.parametrize("out_int32", [False, True])
def test_bucketize_numerically(right, out_int32):
    class Bucketize(Module):
        def forward(self, input_tensor, boundaries):
            return torch.bucketize(input_tensor, boundaries, right=right, out_int32=out_int32)

    input_tensor = torch.tensor([-0.5, 0.0, 0.5, 1.0, 2.0, 2.5], dtype=torch.float32)
    boundaries = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

    verify_model_numerically(Bucketize(), (input_tensor, boundaries))


def test_argsort():
    class Argsort(Module):
        def forward(self, x):
            return torch.argsort(x, dim=1, descending=True)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 3), dtype="float32")) -> R.Tuple(R.Tensor((5, 3), dtype="int64")):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int64") = R.argsort(
                    x, axis=1, descending=True, dtype="int64"
                )
                lv1: R.Tensor((5, 3), dtype="float32") = R.gather_elements(x, lv, axis=1)
                lv2: R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")) = (
                    lv1,
                    lv,
                )
                lv3: R.Tensor((5, 3), dtype="int64") = lv2[1]
                gv: R.Tuple(R.Tensor((5, 3), dtype="int64")) = (lv3,)
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Argsort(), example_args, {}, Expected)


def test_sort():
    class Sort(Module):
        def forward(self, x):
            return torch.sort(x, dim=1, descending=True)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="int64") = R.argsort(
                    x, axis=1, descending=True, dtype="int64"
                )
                lv1: R.Tensor((5, 3), dtype="float32") = R.gather_elements(x, lv, axis=1)
                lv2: R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")) = (
                    lv1,
                    lv,
                )
                lv3: R.Tensor((5, 3), dtype="float32") = lv2[0]
                lv4: R.Tensor((5, 3), dtype="int64") = lv2[1]
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32"), R.Tensor((5, 3), dtype="int64")) = (
                    lv3,
                    lv4,
                )
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Sort(), example_args, {}, Expected)


def test_topk():
    class Topk(Module):
        def forward(self, x):
            return torch.topk(x, k=2, dim=1, largest=True, sorted=True)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tuple(R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")) = (
                    R.topk(x, k=2, axis=1, ret_type="both", largest=True, dtype="int64")
                )
                lv1: R.Tensor((5, 2), dtype="float32") = lv[0]
                lv2: R.Tensor((5, 2), dtype="int64") = lv[1]
                gv: R.Tuple(R.Tensor((5, 2), dtype="float32"), R.Tensor((5, 2), dtype="int64")) = (
                    lv1,
                    lv2,
                )
                R.output(gv)
            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Topk(), example_args, {}, Expected)


def test_dynamic_shape():
    class DynamicModel(torch.nn.Module):
        def forward(self, x1, x2):
            return torch.ops.aten.add.Tensor(x1, x2)

    s0 = T.dynamic("s0", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            lhs: R.Tensor((s0, 4), dtype="float32"),
            rhs: R.Tensor((s0, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((s0, 4), dtype="float32")):
            R.func_attr({"tir_var_lower_bound": {"s24": 0}})
            with R.dataflow():
                lv: R.Tensor((s0, 4), dtype="float32") = R.add(lhs, rhs)
                gv: R.Tuple(R.Tensor((s0, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 4), torch.randn(2, 4))
    batch = torch.export.Dim("batch")
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    verify_model(
        DynamicModel(),
        example_args,
        {},
        Expected,
        dynamic_shapes=dynamic_shapes,
        run_ep_decomposition=True,
        map_free_vars=True,
    )


def test_broadcast_to():
    class BroadcastTo(Module):
        def forward(self, x):
            return torch.broadcast_to(x, (5, 3))

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 1), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((5, 3), dtype="float32") = R.broadcast_to(x, R.shape([5, 3]))
                gv: R.Tuple(R.Tensor((5, 3), dtype="float32")) = (lv,)
                R.output(gv)

            return gv

    example_args = (torch.randn(5, 1, dtype=torch.float32),)
    verify_model(BroadcastTo(), example_args, {}, Expected)


def test_narrow():
    class Narrow(Module):
        def forward(self, x):
            return torch.narrow(x, 1, 0, 2)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((5, 3), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 2), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((5, 2), dtype="float32") = R.strided_slice(
                    x,
                    (R.prim_value(1),),
                    (R.prim_value(0),),
                    (R.prim_value(2),),
                    (R.prim_value(1),),
                    assume_inbound=False,
                )
                gv: R.Tuple(R.Tensor((5, 2), dtype="float32")) = (lv,)
                R.output(gv)

            return gv

    example_args = (torch.randn(5, 3, dtype=torch.float32),)
    verify_model(Narrow(), example_args, {}, Expected)


def test_item():
    class Item(Module):
        def forward(self, x):
            return x.item()

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((1,), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.take(input, R.const(0, "int64"), axis=0)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, dtype=torch.float32),)
    verify_model(Item(), example_args, {}, Expected)


def test_norm():
    class Norm(Module):
        def __init__(self, p, dim=None, keepdim=False):
            super().__init__()
            self.p = p
            self.dim = dim
            self.keepdim = keepdim

        def forward(self, x):
            return torch.norm(x, p=self.p, dim=self.dim, keepdim=self.keepdim)

    @tvm_tirx.script.ir_module
    class Expected1:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.max(R.abs(inp_0), axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class Expected2:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.min(R.abs(inp_0), axis=None, keepdims=False)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class Expected3:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(2, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.power(lv2, R.const(0.5, "float32"))
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class Expected4:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(1.0, "float32"))
                lv2: R.Tensor((), dtype="float32") = R.sum(lv1, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.power(lv2, R.const(1.0, "float32"))
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class Expected5:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 1, 1, 1), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(-4.0, "float32"))
                lv2: R.Tensor((1, 1, 1, 1), dtype="float32") = R.sum(lv1, axis=None, keepdims=True)
                lv3: R.Tensor((1, 1, 1, 1), dtype="float32") = R.power(
                    lv2, R.const(-0.25, "float32")
                )
                gv: R.Tuple(R.Tensor((1, 1, 1, 1), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    @tvm_tirx.script.ir_module
    class Expected6:
        @R.function
        def main(
            inp_0: R.Tensor((1, 3, 5, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 1, 1, 1), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 5, 3), dtype="float32") = R.abs(inp_0)
                lv1: R.Tensor((1, 3, 5, 3), dtype="float32") = R.power(lv, R.const(0.5, "float32"))
                lv2: R.Tensor((1, 1, 1, 1), dtype="float32") = R.sum(lv1, axis=None, keepdims=True)
                lv3: R.Tensor((1, 1, 1, 1), dtype="float32") = R.power(lv2, R.const(2.0, "float32"))
                gv: R.Tuple(R.Tensor((1, 1, 1, 1), dtype="float32")) = (lv3,)
                R.output(gv)
            return gv

    norms = [
        ((float("inf"), None, False), Expected1),
        ((float("-inf"), None, False), Expected2),
        ((float(2), None, False), Expected3),
        ((1.0, None, False), Expected4),
        ((float(-4), None, True), Expected5),
        ((0.5, None, True), Expected6),
    ]

    example_args = (torch.randn(1, 3, 5, 3, dtype=torch.float32),)

    for (p, dim, keepdim), expected in norms:
        verify_model(Norm(p, dim=dim, keepdim=keepdim), example_args, {}, expected)


def test_eye():
    import pytest

    class Eye1(Module):
        def forward(self, input):
            return torch.eye(3, 5, dtype=torch.float32)

    @tvm_tirx.script.ir_module
    class Expected1:
        @R.function
        def main(input: R.Tensor((3, 5), dtype="float32")) -> R.Tuple(
            R.Tensor((3, 5), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="uint8") = R.arange(
                    R.prim_value(0), R.prim_value(3), R.prim_value(1), dtype="uint8"
                )
                lv1: R.Tensor((5,), dtype="uint8") = R.arange(
                    R.prim_value(0), R.prim_value(5), R.prim_value(1), dtype="uint8"
                )
                lv2: R.Tensor((3, 1), dtype="uint8") = R.expand_dims(lv, axis=[-1])
                lv3: R.Tensor((3, 5), dtype="bool") = R.equal(lv2, lv1)
                lv4: R.Tensor((3, 5), dtype="float32") = R.astype(lv3, dtype="float32")
                gv: R.Tuple(R.Tensor((3, 5), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    class Eye2(Module):
        def forward(self, input):
            return torch.eye(5, dtype=torch.float32)

    @tvm_tirx.script.ir_module
    class Expected2:
        @R.function
        def main(input: R.Tensor((5,), dtype="float32")) -> R.Tuple(
            R.Tensor((5, 5), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((5,), dtype="uint8") = R.arange(
                    R.prim_value(0), R.prim_value(5), R.prim_value(1), dtype="uint8"
                )
                lv1: R.Tensor((5,), dtype="uint8") = R.arange(
                    R.prim_value(0), R.prim_value(5), R.prim_value(1), dtype="uint8"
                )
                lv2: R.Tensor((5, 1), dtype="uint8") = R.expand_dims(lv, axis=[-1])
                lv3: R.Tensor((5, 5), dtype="bool") = R.equal(lv2, lv1)
                lv4: R.Tensor((5, 5), dtype="float32") = R.astype(lv3, dtype="float32")
                gv: R.Tuple(R.Tensor((5, 5), dtype="float32")) = (lv4,)
                R.output(gv)
            return gv

    example_args1 = (torch.randn(3, 5, dtype=torch.float32),)
    verify_model(Eye1(), example_args1, {}, Expected1)

    example_args2 = (torch.randn(5, dtype=torch.float32),)
    verify_model(Eye2(), example_args2, {}, Expected2)


def test_cross_entropy():
    class CrossEntropyModule(Module):
        def __init__(self):
            super().__init__()
            self.criterion = nn.CrossEntropyLoss()
            self.target = torch.tensor([0, 1, 2, 1])

        def forward(self, x):
            return self.criterion(x, self.target)

    @tvm_tirx.script.ir_module
    class Expected1:
        @R.function
        def main(x: R.Tensor((4, 3), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 3), dtype="float32") = R.nn.log_softmax(x, axis=1)
                lv1: R.Tensor((4,), dtype="bool") = R.not_equal(
                    R.const([0, 1, 2, 1], dtype="int64"), R.const(-100, "int64")
                )
                lv2: R.Tensor((), dtype="int64") = R.const(0, "int64")
                lv3: R.Tensor((4,), dtype="int64") = R.where(
                    lv1, R.const([0, 1, 2, 1], dtype="int64"), lv2
                )
                lv4: R.Tensor((4, 1), dtype="int64") = R.expand_dims(lv3, axis=[1])
                lv5: R.Tensor((4, 1), dtype="float32") = R.gather_elements(lv, lv4, axis=1)
                lv6: R.Tensor((4,), dtype="float32") = R.squeeze(lv5, axis=[1])
                lv7: R.Tensor((4,), dtype="float32") = R.negative(lv6)
                lv8: R.Tensor((4,), dtype="bool") = R.not_equal(
                    R.const([0, 1, 2, 1], dtype="int64"), R.const(-100, "int64")
                )
                lv9: R.Tensor((), dtype="float32") = R.const(0.0, "float32")
                lv10: R.Tensor((4,), dtype="float32") = R.where(lv8, lv7, lv9)
                lv11: R.Tensor((4,), dtype="bool") = R.not_equal(
                    R.const([0, 1, 2, 1], dtype="int64"), R.const(-100, "int64")
                )
                lv12: R.Tensor((4,), dtype="int64") = R.astype(lv11, dtype="int64")
                lv13: R.Tensor((), dtype="int64") = R.sum(lv12, axis=None, keepdims=False)
                lv14: R.Tensor((), dtype="float32") = R.astype(lv13, dtype="float32")
                lv15: R.Tensor((), dtype="float32") = R.sum(lv10, axis=None, keepdims=False)
                lv16: R.Tensor((), dtype="float32") = R.divide(lv15, lv14)
                gv: R.Tuple(R.Tensor((), dtype="float32")) = (lv16,)
                R.output(gv)
            return gv

    example_args1 = (torch.randn(4, 3, dtype=torch.float32),)
    verify_model(CrossEntropyModule(), example_args1, {}, Expected1)


def test_linspace():
    class Linspace(Module):
        def forward(self, input):
            return torch.linspace(0, 1, steps=9, dtype=torch.float32)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(input: R.Tensor((9, 9), dtype="float32")) -> R.Tuple(
            R.Tensor((9,), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((9,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(9), R.prim_value(1), dtype="int64"
                )
                lv1: R.Tensor((9,), dtype="bool") = R.less(lv, R.const(4, "int64"))
                lv2: R.Tensor((9,), dtype="float32") = R.astype(lv, dtype="float32")
                lv3: R.Tensor((9,), dtype="float32") = R.multiply(lv2, R.const(0.125, "float32"))
                lv4: R.Tensor((9,), dtype="float32") = R.add(lv3, R.const(0.0, "float32"))
                lv5: R.Tensor((9,), dtype="int64") = R.subtract(R.const(8, "int64"), lv)
                lv6: R.Tensor((9,), dtype="float32") = R.astype(lv5, dtype="float32")
                lv7: R.Tensor((9,), dtype="float32") = R.multiply(lv6, R.const(0.125, "float32"))
                lv8: R.Tensor((9,), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv7)
                lv9: R.Tensor((9,), dtype="float32") = R.where(lv1, lv4, lv8)
                gv: R.Tuple(R.Tensor((9,), dtype="float32")) = (lv9,)
                R.output(gv)
            return gv

    example_args = (torch.randn(9, 9, dtype=torch.float32),)
    verify_model(Linspace(), example_args, {}, Expected)


@pytest.mark.parametrize(
    "torch_dtype, relax_dtype",
    [
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.bfloat16, "bfloat16"),
        (torch.int64, "int64"),
        (torch.int32, "int32"),
        (torch.bool, "bool"),
    ],
)
def test_dtypes(torch_dtype, relax_dtype):
    example_args = (
        torch.testing.make_tensor((10, 10), dtype=torch_dtype, device="cpu", low=0, high=10),
        torch.testing.make_tensor((10, 10), dtype=torch_dtype, device="cpu", low=0, high=10),
    )

    class Model(Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor):
            return torch.ops.aten.add(lhs, rhs)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            lhs: R.Tensor((10, 10), dtype=relax_dtype),
            rhs: R.Tensor((10, 10), dtype=relax_dtype),
        ) -> R.Tuple(R.Tensor((10, 10), dtype=relax_dtype)):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype=relax_dtype) = relax.op.add(lhs, rhs)
                gv: R.Tuple(R.Tensor((10, 10), dtype=relax_dtype)) = (lv,)
                R.output(gv)
            return gv

    verify_model(Model(), example_args, {}, Expected)


def test_mm():
    class MatrixMultiply(Module):
        def forward(self, a, b):
            return torch.mm(a, b)

    example_args = (
        torch.randn(2, 3, dtype=torch.float32),
        torch.randn(3, 4, dtype=torch.float32),
    )

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((2, 3), dtype="float32"),
            b: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 4), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 4), dtype="float32") = R.matmul(a, b, out_dtype="float32")
                gv: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    verify_model(MatrixMultiply(), example_args, {}, Expected)


def test_sparse_mm():
    class SparseMatrixMultiply(Module):
        def forward(self, sparse_input, dense_input):
            return torch.sparse.mm(sparse_input, dense_input)

    indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    sparse_input = torch.sparse_coo_tensor(indices, values, size=(3, 100))
    dense_input = torch.randn(100, 50, dtype=torch.float32)

    example_args = (sparse_input, dense_input)

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            sparse_input: R.Tensor((3, 100), dtype="float32"),
            dense_input: R.Tensor((100, 50), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 50), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((3, 50), dtype="float32") = R.full(
                    R.shape([3, 50]), R.const(0.0, "float32"), dtype="float32"
                )
                lv1: R.Tensor((3, 50), dtype="float32") = R.matmul(
                    sparse_input, dense_input, out_dtype="float32"
                )
                gv: R.Tuple(R.Tensor((3, 50), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    verify_model(SparseMatrixMultiply(), example_args, {}, Expected)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_lstm():
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, batch_first, bidirectional):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=batch_first,
                bidirectional=bidirectional,
            )

        def forward(self, x):
            y, _ = self.lstm(x)
            return y

    # Unidirectional LSTM with batch_first=True
    torch.manual_seed(42)
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    verify_model_numerically(LSTM(4, 8, batch_first=True, bidirectional=False), (x,))

    # Unidirectional LSTM with batch_first=False
    torch.manual_seed(43)
    x2 = torch.randn(4, 2, 3, dtype=torch.float32)
    verify_model_numerically(LSTM(3, 6, batch_first=False, bidirectional=False), (x2,))

    # Bidirectional LSTM with batch_first=True
    torch.manual_seed(44)
    x3 = torch.randn(2, 3, 4, dtype=torch.float32)
    verify_model_numerically(LSTM(4, 8, batch_first=True, bidirectional=True), (x3,))

    # Bidirectional LSTM with batch_first=False
    torch.manual_seed(45)
    x4 = torch.randn(4, 2, 3, dtype=torch.float32)
    verify_model_numerically(LSTM(3, 6, batch_first=False, bidirectional=True), (x4,))


def test_tensor_none_tuple():
    example_args = (torch.tensor([1.0, 2.0, 3.0]),)

    class TensorNoneModel(Module):
        def forward(self, x):
            return x + 1, None

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32")) -> R.Tuple(
            R.Tensor((3,), dtype="float32"), R.Any
        ):
            with R.dataflow():
                lv: R.Tensor((3,), dtype="float32") = R.add(x, R.const(1.0, "float32"))
                gv: R.Tuple(R.Tensor((3,), dtype="float32"), R.Any) = (lv, R.null_value())
                R.output(gv)
            return gv

    verify_model(TensorNoneModel(), example_args, {}, Expected)


def test_gru():
    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, batch_first, bidirectional):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=batch_first,
                bidirectional=bidirectional,
            )

        def forward(self, x):
            y, _ = self.gru(x)
            return y

    cases = [
        (42, (2, 3, 4), 4, 8, True, False),
        (43, (4, 2, 3), 3, 6, False, False),
        (44, (2, 3, 4), 4, 5, True, True),
        (45, (4, 2, 3), 3, 4, False, True),
    ]
    for seed, shape, input_size, hidden_size, batch_first, bidirectional in cases:
        torch.manual_seed(seed)
        x = torch.randn(*shape, dtype=torch.float32)
        verify_model_numerically(
            GRU(input_size, hidden_size, batch_first, bidirectional),
            (x,),
            rtol=1e-4,
            atol=1e-5,
        )


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_rnn_tanh():
    target = tvm_tirx.target.Target("llvm")

    def _check(rnn_kwargs, x_shape, seed):
        class RNNWithState(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.RNN(nonlinearity="tanh", num_layers=1, **rnn_kwargs)

            def forward(self, x):
                output, h_n = self.rnn(x)
                return output, h_n

        torch.manual_seed(seed)
        x = torch.randn(*x_shape, dtype=torch.float32)
        model = RNNWithState()
        with torch.no_grad():
            pt_out, pt_hn = model(x)

        exported_program = export(model, args=(x,))
        mod = from_exported_program(exported_program, run_ep_decomposition=False)
        ex = relax.build(mod, target)
        vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
        tvm_outputs = vm["main"](tvm_tirx.runtime.tensor(x.numpy()))
        tvm_out_np = tvm_outputs[0].numpy()
        tvm_hn_np = tvm_outputs[1].numpy()

        assert pt_out.shape == tvm_out_np.shape, (
            f"output shape mismatch: PyTorch {tuple(pt_out.shape)} vs TVM {tvm_out_np.shape}"
        )
        assert pt_hn.shape == tvm_hn_np.shape, (
            f"h_n shape mismatch: PyTorch {tuple(pt_hn.shape)} vs TVM {tvm_hn_np.shape}"
        )
        tvm_tirx.testing.assert_allclose(pt_out.numpy(), tvm_out_np, rtol=1e-4, atol=1e-5)
        tvm_tirx.testing.assert_allclose(pt_hn.numpy(), tvm_hn_np, rtol=1e-4, atol=1e-5)

    # batch_first, unidirectional
    _check(
        {"input_size": 4, "hidden_size": 8, "batch_first": True, "bidirectional": False},
        (2, 3, 4),
        seed=42,
    )
    # seq-first (batch_first=False), unidirectional
    _check(
        {"input_size": 3, "hidden_size": 6, "batch_first": False, "bidirectional": False},
        (4, 2, 3),
        seed=43,
    )
    # bidirectional, batch_first
    _check(
        {"input_size": 4, "hidden_size": 8, "batch_first": True, "bidirectional": True},
        (2, 3, 4),
        seed=44,
    )


def test_dynamic_shape_with_range_constraints():
    class DynamicModel(torch.nn.Module):
        def forward(self, x1, x2):
            return torch.ops.aten.add.Tensor(x1, x2)

    s0 = T.dynamic("s0", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x1: R.Tensor((s0, 4), dtype="float32"), x2: R.Tensor((s0, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((s0, 4), dtype="float32")):
            R.func_attr({"tir_var_lower_bound": {"s24": 1}, "tir_var_upper_bound": {"s24": 64}})
            with R.dataflow():
                lv: R.Tensor((s0, 4), dtype="float32") = R.add(x1, x2)
                gv: R.Tuple(R.Tensor((s0, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(8, 4), torch.randn(8, 4))
    batch = torch.export.Dim("batch", min=1, max=64)
    dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

    verify_model(
        DynamicModel(),
        example_args,
        {},
        Expected,
        dynamic_shapes=dynamic_shapes,
        map_free_vars=True,
    )


def test_dynamic_shape_with_addition_constraints():
    class ConcatModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.cat([x, y], dim=0)

    s0 = T.dynamic("s0", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((s0, 4), dtype="float32"), y: R.Tensor((1 + s0, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((s0 + (1 + s0), 4), dtype="float32")):
            R.func_attr(
                {
                    "tir_var_lower_bound": {"s77": 1},
                    "tir_var_upper_bound": {"s77": 64},
                }
            )
            with R.dataflow():
                lv: R.Tensor((s0 + (1 + s0), 4), dtype="float32") = R.concat((x, y), axis=0)
                gv: R.Tuple(R.Tensor((s0 + (1 + s0), 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    batch = torch.export.Dim("batch", min=1, max=64)
    example_args = (torch.randn(8, 4), torch.randn(9, 4))
    dynamic_shapes = {"x": {0: batch}, "y": {0: batch + 1}}

    verify_model(
        ConcatModel(), example_args, {}, Expected, dynamic_shapes=dynamic_shapes, map_free_vars=True
    )


def test_dynamic_shape_with_multiplication_constraints():
    class ConcatModel(torch.nn.Module):
        def forward(self, x, y):
            return torch.cat([x, y], dim=0)

    s0 = T.dynamic("s0", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((s0, 4), dtype="float32"), y: R.Tensor((2 * s0, 4), dtype="float32")
        ) -> R.Tuple(R.Tensor((s0 + 2 * s0, 4), dtype="float32")):
            R.func_attr(
                {
                    "tir_var_lower_bound": {"s77": 1},
                    "tir_var_upper_bound": {"s77": 64},
                }
            )
            with R.dataflow():
                lv: R.Tensor((s0 + 2 * s0, 4), dtype="float32") = R.concat((x, y), axis=0)
                gv: R.Tuple(R.Tensor((s0 + 2 * s0, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    batch = torch.export.Dim("batch", min=1, max=64)
    example_args = (torch.randn(8, 4), torch.randn(16, 4))
    dynamic_shapes = {"x": {0: batch}, "y": {0: batch * 2}}

    verify_model(
        ConcatModel(), example_args, {}, Expected, dynamic_shapes=dynamic_shapes, map_free_vars=True
    )


def test_dynamic_shape_with_unbounded_constraints():
    class DynamicModel(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.add.Tensor(x, x)

    s0 = T.dynamic("s0", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((s0, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((s0, 4), dtype="float32")
        ):
            R.func_attr({"tir_var_lower_bound": {"s77": 2}})
            with R.dataflow():
                lv: R.Tensor((s0, 4), dtype="float32") = R.add(x, x)
                gv: R.Tuple(R.Tensor((s0, 4), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(8, 4),)
    batch = torch.export.Dim("batch", min=2)
    dynamic_shapes = {"x": {0: batch}}

    verify_model(
        DynamicModel(),
        example_args,
        {},
        Expected,
        dynamic_shapes=dynamic_shapes,
        map_free_vars=True,
    )


def test_sym_size_int():
    class SymSizeInt(Module):
        def forward(self, x):
            shape_dim = torch.ops.aten.sym_size.int(x, 0)
            return x.reshape(shape_dim, -1)

    s0 = T.dynamic("s0", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((s0, 3, 4), dtype="float32")) -> R.Tuple(
            R.Tensor((s0, 12), dtype="float32")
        ):
            R.func_attr({"tir_var_lower_bound": {"s77": 0}})
            with R.dataflow():
                lv: R.Tensor((s0, 12), dtype="float32") = R.reshape(x, R.shape([s0, 12]))
                gv: R.Tuple(R.Tensor((s0, 12), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(2, 3, 4),)
    dynamic_shapes = {"x": {0: torch.export.Dim("dim")}}
    verify_model(
        SymSizeInt(),
        example_args,
        {},
        Expected,
        dynamic_shapes=dynamic_shapes,
        map_free_vars=True,
    )


def test_exponential():
    class Exponential(Module):
        def forward(self, x):
            return x.exponential_()

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 8), dtype="float32")) -> R.Tuple(
            R.Tensor((4, 8), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((4, 8), dtype="float32") = R.zeros_like(x)
                gv: R.Tuple(R.Tensor((4, 8), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (torch.randn(4, 8, dtype=torch.float32),)
    verify_model(Exponential(), example_args, {}, Expected)


def test_max_dim():
    class MaxDim1(Module):
        def forward(self, x):
            return torch.max(x, dim=1)

    class MaxDim2(Module):
        def forward(self, x):
            return torch.max(x, dim=1, keepdim=True)

    @I.ir_module
    class expected1:
        @R.function
        def main(x: R.Tensor((4, 8, 16), dtype="float32")) -> R.Tuple(
            R.Tensor((4, 16), dtype="float32"), R.Tensor((4, 16), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((4, 1, 16), dtype="float32"), R.Tensor((4, 1, 16), dtype="int64")
                ) = R.topk(x, k=1, axis=1, ret_type="both", largest=True, dtype="int64")
                lv1: R.Tensor((4, 1, 16), dtype="float32") = lv[0]
                lv2: R.Tensor((4, 16), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv3: R.Tensor((4, 1, 16), dtype="int64") = lv[1]
                lv4: R.Tensor((4, 16), dtype="int64") = R.squeeze(lv3, axis=[1])
                lv5: R.Tuple(
                    R.Tensor((4, 16), dtype="float32"), R.Tensor((4, 16), dtype="int64")
                ) = (lv2, lv4)
                lv6: R.Tensor((4, 16), dtype="float32") = lv5[0]
                lv7: R.Tensor((4, 16), dtype="int64") = lv5[1]
                gv: R.Tuple(
                    R.Tensor((4, 16), dtype="float32"), R.Tensor((4, 16), dtype="int64")
                ) = (lv6, lv7)
                R.output(gv)
            return gv

    @I.ir_module
    class expected2:
        @R.function
        def main(x: R.Tensor((4, 8, 16), dtype="float32")) -> R.Tuple(
            R.Tensor((4, 1, 16), dtype="float32"), R.Tensor((4, 1, 16), dtype="int64")
        ):
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((4, 1, 16), dtype="float32"), R.Tensor((4, 1, 16), dtype="int64")
                ) = R.topk(x, k=1, axis=1, ret_type="both", largest=True, dtype="int64")
                lv1: R.Tensor((4, 1, 16), dtype="float32") = lv[0]
                lv2: R.Tensor((4, 1, 16), dtype="int64") = lv[1]
                lv3: R.Tuple(
                    R.Tensor((4, 1, 16), dtype="float32"), R.Tensor((4, 1, 16), dtype="int64")
                ) = (lv1, lv2)
                lv4: R.Tensor((4, 1, 16), dtype="float32") = lv3[0]
                lv5: R.Tensor((4, 1, 16), dtype="int64") = lv3[1]
                gv: R.Tuple(
                    R.Tensor((4, 1, 16), dtype="float32"), R.Tensor((4, 1, 16), dtype="int64")
                ) = (lv4, lv5)
                R.output(gv)
            return gv

    example_args = (torch.randn(4, 8, 16, dtype=torch.float32),)
    verify_model(MaxDim1(), example_args, {}, expected1)
    verify_model(MaxDim2(), example_args, {}, expected2)


def test_alias():
    class Alias(Module):
        def forward(self, x):
            return torch.ops.aten.alias(x)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 8), dtype="float32")) -> R.Tuple(
            R.Tensor((4, 8), dtype="float32")
        ):
            with R.dataflow():
                gv: R.Tuple(R.Tensor((4, 8), dtype="float32")) = (x,)
                R.output(gv)
            return gv

    example_args = (torch.randn(4, 8, dtype=torch.float32),)
    verify_model(Alias(), example_args, {}, Expected)


def test_scatter_value():
    class ScatterValue(Module):
        def forward(self, x, index):
            return x.scatter(1, index, 0.5)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((4, 8), dtype="float32"),
            index: R.Tensor((4, 2), dtype="int64"),
        ) -> R.Tuple(R.Tensor((4, 8), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 2), dtype="float32") = R.broadcast_to(
                    R.const(0.5, "float32"), R.shape([4, 2])
                )
                lv1: R.Tensor((4, 8), dtype="float32") = R.scatter_elements(x, index, lv, axis=1)
                gv: R.Tuple(R.Tensor((4, 8), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(4, 8, dtype=torch.float32),
        torch.randint(0, 8, (4, 2), dtype=torch.int64),
    )
    verify_model(ScatterValue(), example_args, {}, Expected)


def test_scatter_src():
    class ScatterSrc(Module):
        def forward(self, x, index, src):
            return x.scatter(1, index, src)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((4, 8), dtype="float32"),
            index: R.Tensor((4, 2), dtype="int64"),
            src: R.Tensor((4, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((4, 8), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((4, 8), dtype="float32") = R.scatter_elements(x, index, src, axis=1)
                gv: R.Tuple(R.Tensor((4, 8), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(4, 8, dtype=torch.float32),
        torch.randint(0, 8, (4, 2), dtype=torch.int64),
        torch.randn(4, 2, dtype=torch.float32),
    )
    verify_model(ScatterSrc(), example_args, {}, Expected)


def test_grid_sample():
    class GridSample(Module):
        def forward(self, input, grid):
            return torch.nn.functional.grid_sample(
                input, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 2, 2, 2), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 3, 2, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 3, 2, 2), dtype="float32") = R.image.grid_sample(
                    input_1,
                    grid,
                    method="bilinear",
                    layout="NCHW",
                    padding_mode="zeros",
                    align_corners=True,
                )
                gv: R.Tuple(R.Tensor((1, 3, 2, 2), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 4, 4, dtype=torch.float32),
        torch.randn(1, 2, 2, 2, dtype=torch.float32),
    )
    verify_model(GridSample(), example_args, {}, expected)


def test_torchvision_roi_align():
    torchvision = pytest.importorskip("torchvision")

    class ROIAlign(Module):
        def forward(self, input, rois):
            return torchvision.ops.roi_align(
                input,
                rois,
                output_size=(3, 3),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=False,
            )

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 8, 8), dtype="float32"),
            rois: R.Tensor((2, 5), dtype="float32"),
        ) -> R.Tuple(R.Tensor((2, 3, 3, 3), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((2, 3, 3, 3), dtype="float32") = R.vision.roi_align(
                    input_1,
                    rois,
                    pooled_size=(3, 3),
                    spatial_scale=1.0,
                    sample_ratio=2,
                    layout="NCHW",
                    mode="avg",
                )
                gv: R.Tuple(R.Tensor((2, 3, 3, 3), dtype="float32")) = (lv,)
                R.output(gv)
            return gv

    example_args = (
        torch.randn(1, 3, 8, 8, dtype=torch.float32),
        torch.tensor([[0.0, 1.0, 1.0, 6.0, 6.0], [0.0, 0.5, 0.5, 7.0, 7.0]], dtype=torch.float32),
    )
    verify_model(ROIAlign(), example_args, {}, expected)


def test_torchvision_roi_align_aligned():
    torchvision = pytest.importorskip("torchvision")

    class ROIAlign(Module):
        def forward(self, input, rois):
            return torchvision.ops.roi_align(
                input,
                rois,
                output_size=(1, 1),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True,
            )

    example_args = (
        torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
        torch.tensor([[0.0, 1.0, 1.0, 1.2, 1.2]], dtype=torch.float32),
    )
    verify_model_numerically(ROIAlign(), example_args, rtol=1e-5, atol=1e-5)


def test_from_exported_program_sparse_csr_buffer():
    class SparseCsrBufferModule(nn.Module):
        def __init__(self):
            super().__init__()
            crow_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
            col_indices = torch.tensor([0, 1], dtype=torch.int64)
            values = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
            csr_tensor = torch.sparse_csr_tensor(
                crow_indices, col_indices, values, dtype=torch.float32
            )
            self.register_buffer("csr_tensor", csr_tensor)
            self.csr_tensor.requires_grad_(True)

        def forward(self, x):
            csr2 = self.csr_tensor.to_sparse(layout=torch.sparse_csr)
            y = torch.matmul(csr2, x)
            return y.sum()

    model = SparseCsrBufferModule().eval()
    x = torch.ones((2, 1), dtype=torch.float32)
    exported_program = export(model, (x,))
    mod = from_exported_program(exported_program)
    assert isinstance(mod, tvm_tirx.IRModule)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("with_message,in_cond", [(False, False), (True, False), (True, True)])
def test_assert_async(with_message, in_cond):
    class AssertAsync(Module):
        def forward(self, x):
            def checked(value):
                if with_message:
                    torch.ops.aten._assert_async.msg(
                        value.clamp(min=0), assert_msg="Positive required: {value}"
                    )
                else:
                    torch.ops.aten._assert_async.default(value)
                return value.clone()

            if in_cond:
                return torch.cond(x >= 0, checked, lambda value: value.clone(), (x,))
            return checked(x)

    # Also cover non-boolean conditions and one-element, non-scalar tensors.
    example_args = (torch.tensor(0.5) if with_message else torch.tensor([True]),)
    model = AssertAsync()
    # PyTorch's decomposition removes the message-less overload.
    mod = from_exported_program(export(model, args=example_args), run_ep_decomposition=with_message)
    assert not mod["main"].is_pure
    vm = relax.VirtualMachine(relax.build(mod, target="llvm"), tvm_tirx.cpu())
    actual = vm["main"](tvm_tirx.runtime.tensor(example_args[0].numpy()))
    np.testing.assert_array_equal(actual[0].numpy(), model(*example_args).numpy())

    invalid = torch.zeros_like(example_args[0])
    with pytest.raises(RuntimeError):
        model(invalid)
    message = "Positive required: [{]value[}]" if with_message else "Assertion Failed"
    with pytest.raises(AssertionError, match=message):
        vm["main"](tvm_tirx.runtime.tensor(invalid.numpy()))

    if in_cond:
        # The unchecked branch must not execute the other branch's assertion.
        negative = -example_args[0]
        actual = vm["main"](tvm_tirx.runtime.tensor(negative.numpy()))
        np.testing.assert_array_equal(actual[0].numpy(), model(negative).numpy())


def test_cond_basic():
    """Basic data-dependent cond with runtime predicate."""

    class CondModel(Module):
        def forward(self, x):
            def true_fn(x):
                return x.cos()

            def false_fn(x):
                return x.sin()

            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def cond_true_branch_0(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((3, 4), dtype="float32"):
            gv: R.Tensor((3, 4), dtype="float32") = R.cos(x)
            gv1: R.Tensor((3, 4), dtype="float32") = gv
            return gv1

        @R.function
        def cond_false_branch_1(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((3, 4), dtype="float32"):
            gv: R.Tensor((3, 4), dtype="float32") = R.sin(x)
            gv1: R.Tensor((3, 4), dtype="float32") = gv
            return gv1

        @R.function
        def main(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32")):
            cls = expected
            gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            gv1: R.Tensor((), dtype="bool") = R.greater(gv, R.const(0.0, "float32"))
            if gv1:
                gv2: R.Tensor((3, 4), dtype="float32") = cls.cond_true_branch_0(x)
                cond_result: R.Tensor((3, 4), dtype="float32") = gv2
            else:
                gv3: R.Tensor((3, 4), dtype="float32") = cls.cond_false_branch_1(x)
                cond_result: R.Tensor((3, 4), dtype="float32") = gv3
            return (cond_result,)

    verify_model(CondModel(), (torch.randn(3, 4),), {}, expected, map_free_vars=True)


def test_cond_shape_predicate():
    """Cond with a shape-derived predicate and dynamic shapes."""

    class CondShapeModel(Module):
        def forward(self, x):
            def true_fn(x):
                return x + 1.0

            def false_fn(x):
                return x - 1.0

            return torch.cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    s77_cond_true_branch_0 = T.dynamic("s77", "int64")
    s77_cond_false_branch_1 = T.dynamic("s77", "int64")
    s77_main = T.dynamic("s77", "int64")

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def cond_true_branch_0(
            x: R.Tensor((s77_cond_true_branch_0, 4), dtype="float32"),
        ) -> R.Tensor((s77_cond_true_branch_0, 4), dtype="float32"):
            gv: R.Tensor((s77_cond_true_branch_0, 4), dtype="float32") = R.add(
                x, R.const(1.0, "float32")
            )
            gv1: R.Tensor((s77_cond_true_branch_0, 4), dtype="float32") = gv
            return gv1

        @R.function
        def cond_false_branch_1(
            x: R.Tensor((s77_cond_false_branch_1, 4), dtype="float32"),
        ) -> R.Tensor((s77_cond_false_branch_1, 4), dtype="float32"):
            gv: R.Tensor((s77_cond_false_branch_1, 4), dtype="float32") = R.subtract(
                x, R.const(1.0, "float32")
            )
            gv1: R.Tensor((s77_cond_false_branch_1, 4), dtype="float32") = gv
            return gv1

        @R.function
        def main(
            x: R.Tensor((s77_main, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((s77_main, 4), dtype="float32")):
            R.func_attr({"tir_var_lower_bound": {"s77": 1}})
            cls = expected
            gv: T.bool = s77_main > 4
            if gv:
                gv1: R.Tensor((s77_main, 4), dtype="float32") = cls.cond_true_branch_0(x)
                cond_result: R.Tensor((s77_main, 4), dtype="float32") = gv1
            else:
                gv2: R.Tensor((s77_main, 4), dtype="float32") = cls.cond_false_branch_1(x)
                cond_result: R.Tensor((s77_main, 4), dtype="float32") = gv2
            return (cond_result,)

    batch = torch.export.Dim("batch", min=1)
    verify_model(
        CondShapeModel(),
        (torch.randn(3, 4),),
        {},
        expected,
        dynamic_shapes={"x": {0: batch}},
        map_free_vars=True,
    )


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_cond_shape_equality_predicate():
    class CondShapeEqualityModel(Module):
        def forward(self, x):
            def true_fn(x):
                return x + 1.0

            def false_fn(x):
                return x - 1.0

            return torch.cond(x.shape[0] == x.shape[1], true_fn, false_fn, (x,))

    rows = torch.export.Dim("rows", min=1, max=8)
    columns = torch.export.Dim("columns", min=1, max=8)
    exported_program = export(
        CondShapeEqualityModel(),
        args=(torch.zeros(3, 3),),
        dynamic_shapes={"x": {0: rows, 1: columns}},
    )
    mod = from_exported_program(exported_program)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    for shape, expected_value in (((3, 3), 1.0), ((2, 3), -1.0)):
        torch_input = torch.zeros(shape, dtype=torch.float32)
        actual = vm["main"](tvm_tirx.runtime.tensor(torch_input.numpy()))[0]
        np.testing.assert_array_equal(actual.numpy(), np.full(shape, expected_value, "float32"))


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_cond_shape_inequality_predicate():
    class CondShapeInequalityModel(Module):
        def forward(self, x):
            def true_fn(x):
                return x + 1.0

            def false_fn(x):
                return x - 1.0

            return torch.cond(x.shape[0] != x.shape[1], true_fn, false_fn, (x,))

    rows = torch.export.Dim("rows", min=1, max=8)
    columns = torch.export.Dim("columns", min=1, max=8)
    exported_program = export(
        CondShapeInequalityModel(),
        args=(torch.zeros(2, 3),),
        dynamic_shapes={"x": {0: rows, 1: columns}},
    )
    mod = from_exported_program(exported_program)
    executable = relax.build(mod, tvm_tirx.target.Target("llvm"))
    vm = relax.VirtualMachine(executable, tvm_tirx.cpu())

    for shape, expected_value in (((2, 3), 1.0), ((3, 3), -1.0)):
        torch_input = torch.zeros(shape, dtype=torch.float32)
        actual = vm["main"](tvm_tirx.runtime.tensor(torch_input.numpy()))[0]
        np.testing.assert_array_equal(actual.numpy(), np.full(shape, expected_value, "float32"))


def test_cond_tuple_output():
    """Cond where both branches return a tuple."""

    class CondTupleModel(Module):
        def forward(self, x):
            def true_fn(x):
                return (x.cos(), x.sin())

            def false_fn(x):
                return (x.sin(), x.cos())

            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def cond_true_branch_0(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 4), dtype="float32")):
            gv: R.Tensor((3, 4), dtype="float32") = R.cos(x)
            gv1: R.Tensor((3, 4), dtype="float32") = R.sin(x)
            gv2: R.Tensor((3, 4), dtype="float32") = gv
            gv3: R.Tensor((3, 4), dtype="float32") = gv1
            return (gv2, gv3)

        @R.function
        def cond_false_branch_1(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 4), dtype="float32")):
            gv: R.Tensor((3, 4), dtype="float32") = R.sin(x)
            gv1: R.Tensor((3, 4), dtype="float32") = R.cos(x)
            gv2: R.Tensor((3, 4), dtype="float32") = gv
            gv3: R.Tensor((3, 4), dtype="float32") = gv1
            return (gv2, gv3)

        @R.function
        def main(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 4), dtype="float32")):
            cls = expected
            gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            gv1: R.Tensor((), dtype="bool") = R.greater(gv, R.const(0.0, "float32"))
            if gv1:
                gv2: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = cls.cond_true_branch_0(x)
                cond_result: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = gv2
            else:
                gv3: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = cls.cond_false_branch_1(x)
                cond_result: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = gv3
            gv4: R.Tensor((3, 4), dtype="float32") = cond_result[0]
            gv5: R.Tensor((3, 4), dtype="float32") = cond_result[1]
            return (gv4, gv5)

    verify_model(CondTupleModel(), (torch.randn(3, 4),), {}, expected, map_free_vars=True)


def test_cond_nested():
    """Nested cond: a cond inside one of the branches."""

    class CondNestedModel(Module):
        def forward(self, x):
            def true_fn(x):
                def inner_true(x):
                    return x * 2.0

                def inner_false(x):
                    return x * 3.0

                return torch.cond(x.sum() > 1, inner_true, inner_false, (x,))

            def false_fn(x):
                return x - 1.0

            return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    example_args = (torch.randn(3, 4),)
    exported_program = export(CondNestedModel(), args=example_args)
    mod = from_exported_program(exported_program)

    assert isinstance(mod, tvm_tirx.IRModule)

    # Should have at least 4 branch functions (2 outer + 2 inner) plus main
    func_names = [gv.name_hint for gv in mod.get_global_vars()]
    branch_funcs = [n for n in func_names if n != "main"]
    assert len(branch_funcs) >= 4, (
        f"Expected at least 4 branch functions for nested cond, got {branch_funcs}"
    )
    # Verify no duplicate function names
    assert len(set(branch_funcs)) == len(branch_funcs), (
        f"Duplicate branch function names: {branch_funcs}"
    )


def test_affine_grid():
    class AffineGrid(Module):
        def forward(self, theta):
            return torch.nn.functional.affine_grid(theta, [1, 3, 16, 16], align_corners=True)

    @tvm_tirx.script.ir_module
    class expected:
        @R.function
        def main(
            theta: R.Tensor((1, 2, 3), dtype="float32"),
        ) -> R.Tuple(R.Tensor((1, 16, 16, 2), dtype="float32")):
            with R.dataflow():
                lv: R.Tensor((1, 2, 16, 16), dtype="float32") = R.image.affine_grid(
                    theta, size=(16, 16)
                )
                lv1: R.Tensor((1, 16, 16, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 2, 3, 1]
                )
                gv: R.Tuple(R.Tensor((1, 16, 16, 2), dtype="float32")) = (lv1,)
                R.output(gv)
            return gv

    example_args = (torch.randn(1, 2, 3, dtype=torch.float32),)
    # Disable decomposition to keep aten.affine_grid_generator as a single op
    verify_model(AffineGrid(), example_args, {}, expected, run_ep_decomposition=False)


def test_affine_grid_numerically():
    """Verify affine_grid numerical correctness: PyTorch vs TVM via our converter."""

    class AffineGrid(Module):
        def forward(self, theta):
            return torch.nn.functional.affine_grid(theta, [2, 3, 8, 12], align_corners=True)

    model = AffineGrid()
    example_args = (torch.randn(2, 2, 3, dtype=torch.float32),)

    with torch.no_grad():
        pytorch_output = model(*example_args)

    exported_program = export(model, args=example_args)
    mod = from_exported_program(exported_program, run_ep_decomposition=False)

    exe = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(exe, tvm_tirx.cpu())

    tvm_args = [tvm_tirx.runtime.tensor(arg.numpy()) for arg in example_args]
    tvm_output = vm["main"](*tvm_args)
    tvm_output_np = tvm_output[0].numpy()

    tvm_tirx.testing.assert_allclose(tvm_output_np, pytorch_output.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm_tirx.testing.main()
