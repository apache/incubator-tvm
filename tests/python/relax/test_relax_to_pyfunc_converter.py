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

"""
Comprehensive test cases for Relax to PyFunc converter.
Tests all major features including basic operations, call_tir, call_dps_packed, and symbolic shapes.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import tvm_ffi

import tvm
import tvm_tirx
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.relax.relax_to_pyfunc_converter import RelaxToPyFuncConverter
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

n_symbolic_add = T.dynamic("n", "int64")
batch_symbolic_matmul = T.dynamic("batch", "int64")
m = T.dynamic("m", "int64")
k = T.dynamic("k", "int64")
n_symbolic_matmul = T.dynamic("n", "int64")
batch_symbolic_expand_dims = T.dynamic("batch", "int64")
seq_len = T.dynamic("seq_len", "int64")


@I.ir_module
class ComprehensiveTestModule:
    """Test module covering all converter features."""

    @Ts.function
    def add_tir(
        x: T.Tensor((5,), "float32"), y: T.Tensor((5,), "float32"), out: T.Tensor((5,), "float32")
    ):
        """TIR function for addition."""

        for i in range(5):
            out[i] = x[i] + y[i]

    @Ts.function
    def mul_tir(
        x: T.Tensor((3, 4), "float32"),
        y: T.Tensor((3, 4), "float32"),
        out: T.Tensor((3, 4), "float32"),
    ):
        """TIR function for multiplication."""

        for i in range(3):
            for j in range(4):
                out[i, j] = x[i, j] * y[i, j]

    @R.function
    def simple_add(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        return R.add(x, y)

    @R.function
    def with_relu(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.nn.relu(x)

    @R.function
    def with_call_tir(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        cls = ComprehensiveTestModule
        return R.call_tir(cls.add_tir, (x, y), ty_args=[R.Tensor((5,), "float32")])

    @R.function
    def with_call_dps_packed(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.call_dps_packed(
            _ExternFunc("my_softmax"), (x, R.prim_value(1)), ty_args=[R.Tensor((5,), "float32")]
        )

    @R.function
    def complex_function(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        added = R.add(x, y)
        relued = R.nn.relu(added)
        cls = ComprehensiveTestModule
        tir_result = R.call_tir(cls.add_tir, (relued, y), ty_args=[R.Tensor((5,), "float32")])
        return R.nn.relu(tir_result)

    @R.function
    def symbolic_add(
        x: R.Tensor((n_symbolic_add,), "float32"), y: R.Tensor((n_symbolic_add,), "float32")
    ) -> R.Tensor((n_symbolic_add,), "float32"):
        return R.add(x, y)

    @R.function
    def symbolic_matmul(
        x: R.Tensor((batch_symbolic_matmul, m, k), "float32"),
        y: R.Tensor((batch_symbolic_matmul, k, n_symbolic_matmul), "float32"),
    ) -> R.Tensor((batch_symbolic_matmul, m, n_symbolic_matmul), "float32"):
        return R.matmul(x, y)

    @R.function
    def symbolic_expand_dims(
        x: R.Tensor((batch_symbolic_expand_dims, seq_len), "float32"),
    ) -> R.Tensor((batch_symbolic_expand_dims, seq_len, 1), "float32"):
        return R.expand_dims(x, axis=2)

    @R.function
    def multi_ops(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")) -> R.Tensor(
        (3, 4), "float32"
    ):
        added = R.add(x, y)
        multiplied = R.multiply(added, y)
        powered = R.power(multiplied, R.const(2.0))
        maxed = R.maximum(powered, x)
        return maxed

    @R.function
    def reduction_ops(x: R.Tensor((5,), "float32")) -> R.Tensor((), "float32"):
        sum_val = R.sum(x)
        mean_val = R.mean(x)
        max_val = R.max(x)
        return R.add(R.add(sum_val, mean_val), max_val)

    @R.function
    def comparison_ops(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "bool"
    ):
        eq_val = R.equal(x, y)
        gt_val = R.greater(x, y)
        return R.logical_and(eq_val, gt_val)

    @R.function
    def test_reshape(x: R.Tensor((2, 3), "float32")) -> R.Tensor((6,), "float32"):
        return R.reshape(x, (6,))

    @R.function
    def test_permute_dims(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((4, 2, 3), "float32"):
        return R.permute_dims(x, axes=[2, 0, 1])

    @R.function
    def test_concat(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor(
        (4, 3), "float32"
    ):
        return R.concat((x, y), axis=0)

    @R.function
    def test_split(x: R.Tensor((4, 3), "float32")) -> R.Tuple:
        return R.split(x, indices_or_sections=2, axis=0)

    @R.function
    def test_stack(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor(
        (2, 2, 3), "float32"
    ):
        return R.stack((x, y), axis=1)

    @R.function
    def test_take(x: R.Tensor((3, 4), "float32"), indices: R.Tensor((2,), "int64")) -> R.Tensor(
        (2,), "float32"
    ):
        return R.take(x, indices, axis=0)

    @R.function
    def test_flip(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        return R.flip(x, axis=1)

    @R.function
    def test_tile(x: R.Tensor((2, 3), "float32")) -> R.Tensor((4, 6), "float32"):
        return R.tile(x, (2, 2))

    @R.function
    def test_repeat(x: R.Tensor((2, 3), "float32")) -> R.Tensor((4, 3), "float32"):
        return R.repeat(x, repeats=2, axis=0)

    @R.function
    def test_expand_dims(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3, 1), "float32"):
        return R.expand_dims(x, axis=2)

    @R.function
    def test_squeeze(x: R.Tensor((2, 3, 1), "float32")) -> R.Tensor((2, 3), "float32"):
        return R.squeeze(x, axis=2)

    @R.function
    def test_sum_with_axis(x: R.Tensor((2, 3), "float32")) -> R.Tensor((3,), "float32"):
        return R.sum(x, axis=0)

    @R.function
    def test_max_with_axis(x: R.Tensor((2, 3), "float32")) -> R.Tensor((3,), "float32"):
        return R.max(x, axis=0)


def create_mock_packed_function():
    """Create a mock packed function for testing."""

    def mock_softmax(x, axis):
        """Mock softmax function that just returns the input."""
        return x

    # Register the function globally
    tvm_ffi.register_global_func("my_softmax", mock_softmax)


class TestRelaxToPyFuncConverter:
    """Comprehensive test class for Relax to PyFunc converter."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.ir_mod = ComprehensiveTestModule
        cls.converter = RelaxToPyFuncConverter(cls.ir_mod)
        create_mock_packed_function()

    def test_basic_operations(self):
        """Test basic arithmetic operations."""
        converted_ir_mod = self.converter.convert(["simple_add", "with_relu"])

        # Test simple_add
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["simple_add"](x, y)
        expected = torch.add(x, y)
        assert torch.allclose(result, expected)

        # Test with_relu
        x_neg = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        result = converted_ir_mod.__pyfuncs__["with_relu"](x_neg)
        expected = torch.nn.functional.relu(x_neg)
        assert torch.allclose(result, expected)

    def test_call_tir(self):
        """Test call_tir functionality with DLPack conversion."""
        converted_ir_mod = self.converter.convert(["with_call_tir"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["with_call_tir"](x, y)
        expected = torch.add(x, y)
        assert torch.allclose(result, expected)
        assert result.shape == expected.shape

    def test_call_dps_packed(self):
        """Test call_dps_packed functionality."""
        converted_ir_mod = self.converter.convert(["with_call_dps_packed"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["with_call_dps_packed"](x)
        expected = x
        assert torch.allclose(result, expected)

    def test_complex_function(self):
        """Test complex function with multiple operations."""
        converted_ir_mod = self.converter.convert(["complex_function"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["complex_function"](x, y)

        # Expected: relu(add(relu(add(x, y)), y))
        step1 = torch.add(x, y)
        step2 = torch.nn.functional.relu(step1)
        step3 = torch.add(step2, y)  # TIR call
        expected = torch.nn.functional.relu(step3)

        assert torch.allclose(result, expected)

    def test_symbolic_shapes(self):
        """Test symbolic shape handling."""
        converted_ir_mod = self.converter.convert(
            ["symbolic_add", "symbolic_matmul", "symbolic_expand_dims"]
        )

        # Test symbolic_add
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        result = converted_ir_mod.__pyfuncs__["symbolic_add"](x, y)
        expected = torch.add(x, y)
        assert torch.allclose(result, expected)

        # Test symbolic_matmul
        x = torch.randn(2, 3, 4, dtype=torch.float32)  # (batch=2, m=3, k=4)
        y = torch.randn(2, 4, 5, dtype=torch.float32)  # (batch=2, k=4, n=5)
        result = converted_ir_mod.__pyfuncs__["symbolic_matmul"](x, y)
        expected = torch.matmul(x, y)
        assert torch.allclose(result, expected)
        assert result.shape == (2, 3, 5)

        # Test symbolic_expand_dims
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        result = converted_ir_mod.__pyfuncs__["symbolic_expand_dims"](x)
        expected = torch.unsqueeze(x, dim=2)
        assert torch.allclose(result, expected)
        assert result.shape == (2, 2, 1)

    def test_multi_operations(self):
        """Test multiple operations in sequence."""
        converted_ir_mod = self.converter.convert(["multi_ops"])

        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            dtype=torch.float32,
        )
        y = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=torch.float32
        )

        result = converted_ir_mod.__pyfuncs__["multi_ops"](x, y)

        # Expected: maximum(power(multiply(add(x, y), y), 2), x)
        step1 = torch.add(x, y)
        step2 = torch.mul(step1, y)
        step3 = torch.pow(step2, 2.0)
        expected = torch.maximum(step3, x)

        assert torch.allclose(result, expected)

    def test_reduction_operations(self):
        """Test reduction operations."""
        converted_ir_mod = self.converter.convert(["reduction_ops"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["reduction_ops"](x)

        # Expected: sum(x) + mean(x) + max(x)
        expected = torch.sum(x) + torch.mean(x) + torch.max(x)

        assert torch.allclose(result, expected)
        assert result.shape == ()

    def test_comparison_operations(self):
        """Test comparison operations."""
        converted_ir_mod = self.converter.convert(["comparison_ops"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        y = torch.tensor([1.0, 2.5, 3.0, 4.5, 5.0], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["comparison_ops"](x, y)

        # Expected: logical_and(equal(x, y), greater(x, y))
        eq_val = torch.eq(x, y)
        gt_val = torch.gt(x, y)
        expected = torch.logical_and(eq_val, gt_val)

        assert torch.allclose(result, expected)
        assert result.dtype == torch.bool

    def test_operator_mapping_completeness(self):
        """Test that operator mapping is comprehensive."""
        operator_map = RelaxToPyFuncConverter._get_op_map()

        # Check that we have a good number of operators
        assert len(operator_map) > 100, f"Expected >100 operators, got {len(operator_map)}"

        # Check key operator categories
        binary_ops = [
            op
            for op in operator_map.keys()
            if op.startswith("relax.") and not op.startswith("relax.nn.")
        ]
        nn_ops = [op for op in operator_map.keys() if op.startswith("relax.nn.")]

        assert len(binary_ops) > 20, f"Expected >20 binary ops, got {len(binary_ops)}"
        assert len(nn_ops) > 30, f"Expected >30 nn ops, got {len(nn_ops)}"

        # Check specific important operators
        important_ops = [
            "relax.add",
            "relax.multiply",
            "relax.nn.relu",
            "relax.nn.softmax",
            "relax.matmul",
            "relax.reshape",
            "relax.sum",
            "relax.mean",
        ]

        for op in important_ops:
            assert op in operator_map, f"Missing important operator: {op}"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        converted_ir_mod = self.converter.convert(["simple_add"])

        # Test with wrong number of arguments
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        with pytest.raises(ValueError, match="Expected 2 arguments"):
            converted_ir_mod.__pyfuncs__["simple_add"](x)  # Missing second argument

        # Test with incompatible shapes - this should raise a RuntimeError
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.tensor([1.0, 2.0], dtype=torch.float32)  # Different shape

        # This should raise a RuntimeError because shapes don't match
        with pytest.raises(RuntimeError, match="The size of tensor a"):
            converted_ir_mod.__pyfuncs__["simple_add"](x, y)

    def test_conversion_metadata(self):
        """Test that conversion preserves metadata correctly."""
        converted_ir_mod = self.converter.convert(["simple_add"])

        # Check that pyfuncs attribute exists
        assert hasattr(converted_ir_mod, "__pyfuncs__")
        assert "simple_add" in converted_ir_mod.__pyfuncs__

        # Check function metadata
        pyfunc = converted_ir_mod.__pyfuncs__["simple_add"]
        assert hasattr(pyfunc, "__name__")
        assert hasattr(pyfunc, "__doc__")
        assert pyfunc.__name__ == "simple_add"

    def test_tensor_operations(self):
        """Test tensor manipulation operations."""
        converted_ir_mod = self.converter.convert(
            [
                "test_reshape",
                "test_permute_dims",
                "test_concat",
                "test_split",
                "test_stack",
                "test_take",
                "test_flip",
                "test_tile",
                "test_repeat",
                "test_expand_dims",
                "test_squeeze",
                "test_sum_with_axis",
                "test_max_with_axis",
            ]
        )

        # Test reshape
        x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        result1 = converted_ir_mod.__pyfuncs__["test_reshape"](x1)
        expected1 = torch.reshape(x1, (6,))
        assert torch.allclose(result1, expected1), "Reshape operation failed"

        # Test permute_dims
        x2 = torch.randn(2, 3, 4)
        result2 = converted_ir_mod.__pyfuncs__["test_permute_dims"](x2)
        expected2 = torch.permute(x2, (2, 0, 1))
        assert torch.allclose(result2, expected2), "Permute_dims operation failed"

        # Test concat
        x3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        y3 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float32)
        result3 = converted_ir_mod.__pyfuncs__["test_concat"](x3, y3)
        expected3 = torch.cat([x3, y3], dim=0)
        assert torch.allclose(result3, expected3), "Concat operation failed"

        # Test split
        x4 = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=torch.float32,
        )
        result4 = converted_ir_mod.__pyfuncs__["test_split"](x4)
        expected4 = torch.split(x4, 2, dim=0)
        assert len(result4) == len(expected4), "Split operation failed - wrong number of tensors"
        for r, e in zip(result4, expected4):
            assert torch.allclose(r, e), "Split operation failed - tensor mismatch"

        # Test stack
        x5 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        y5 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float32)
        result5 = converted_ir_mod.__pyfuncs__["test_stack"](x5, y5)
        expected5 = torch.stack([x5, y5], dim=1)
        assert torch.allclose(result5, expected5), "Stack operation failed"

        # Test take
        x6 = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            dtype=torch.float32,
        )
        indices = torch.tensor([0, 2], dtype=torch.int64)
        result6 = converted_ir_mod.__pyfuncs__["test_take"](x6, indices)
        expected6 = x6[indices]
        assert torch.allclose(result6, expected6), "Take operation failed"

        # Test flip
        x7 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        result7 = converted_ir_mod.__pyfuncs__["test_flip"](x7)
        expected7 = torch.flip(x7, dims=[1])
        assert torch.allclose(result7, expected7), "Flip operation failed"

        # Test tile
        x8 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        result8 = converted_ir_mod.__pyfuncs__["test_tile"](x8)
        expected8 = torch.tile(x8, (2, 2))
        assert torch.allclose(result8, expected8), "Tile operation failed"

        # Test repeat
        x9 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        result9 = converted_ir_mod.__pyfuncs__["test_repeat"](x9)
        expected9 = torch.repeat_interleave(x9, repeats=2, dim=0)
        assert torch.allclose(result9, expected9), "Repeat operation failed"

        # Test expand_dims
        x10 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        result10 = converted_ir_mod.__pyfuncs__["test_expand_dims"](x10)
        expected10 = torch.unsqueeze(x10, dim=2)
        assert torch.allclose(result10, expected10), "Expand_dims operation failed"

        # Test squeeze
        x11 = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=torch.float32)
        result11 = converted_ir_mod.__pyfuncs__["test_squeeze"](x11)
        expected11 = torch.squeeze(x11, dim=2)
        assert torch.allclose(result11, expected11), "Squeeze operation failed"

        # Test sum with axis
        x12 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        result12 = converted_ir_mod.__pyfuncs__["test_sum_with_axis"](x12)
        expected12 = torch.sum(x12, dim=0)
        assert torch.allclose(result12, expected12), "Sum with axis operation failed"

        # Test max with axis
        x13 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        result13 = converted_ir_mod.__pyfuncs__["test_max_with_axis"](x13)
        expected13 = torch.max(x13, dim=0)[0]  # torch.max returns (values, indices)
        assert torch.allclose(result13, expected13), "Max with axis operation failed"


@I.ir_module
class ExtendedOperatorsModule:
    """Extended test module with additional operators not covered in ComprehensiveTestModule."""

    # Unary operations not covered in ComprehensiveTestModule
    @R.function
    def test_abs(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.abs(x)

    @R.function
    def test_neg(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.negative(x)

    @R.function
    def test_exp(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.exp(x)

    @R.function
    def test_log(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.log(x)

    @R.function
    def test_sqrt(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.sqrt(x)

    @R.function
    def test_sin(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.sin(x)

    @R.function
    def test_cos(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.cos(x)

    @R.function
    def test_tanh(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.tanh(x)

    @R.function
    def test_sigmoid(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.sigmoid(x)

    # Comparison operations not covered in ComprehensiveTestModule
    @R.function
    def test_less(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "bool"
    ):
        return R.less(x, y)

    @R.function
    def test_not_equal(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "bool"
    ):
        return R.not_equal(x, y)

    # Binary operations not covered in ComprehensiveTestModule
    @R.function
    def test_multiply(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        return R.multiply(x, y)

    @R.function
    def test_divide(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        return R.divide(x, y)

    @R.function
    def test_power(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        return R.power(x, y)

    @R.function
    def test_maximum(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        return R.maximum(x, y)

    @R.function
    def test_minimum(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        return R.minimum(x, y)

    @R.function
    def test_subtract(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor(
        (5,), "float32"
    ):
        return R.subtract(x, y)

    # Additional tensor operations with different parameters
    @R.function
    def test_transpose_2d(x: R.Tensor((2, 4), "float32")) -> R.Tensor((4, 2), "float32"):
        return R.permute_dims(x, axes=[1, 0])

    @R.function
    def test_mean_axis(x: R.Tensor((2, 3), "float32")) -> R.Tensor((3,), "float32"):
        return R.mean(x, axis=0)

    @R.function
    def test_min_axis(x: R.Tensor((2, 3), "float32")) -> R.Tensor((3,), "float32"):
        return R.min(x, axis=0)

    # Neural network operations not covered in ComprehensiveTestModule
    @R.function
    def test_gelu_nn(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.nn.gelu(x)

    @R.function
    def test_softmax_nn(x: R.Tensor((2, 5), "float32")) -> R.Tensor((2, 5), "float32"):
        return R.nn.softmax(x, axis=1)

    @R.function
    def test_log_softmax_nn(x: R.Tensor((2, 5), "float32")) -> R.Tensor((2, 5), "float32"):
        return R.nn.log_softmax(x, axis=1)

    # Advanced tensor operations with different parameters
    @R.function
    def test_tile_dims(x: R.Tensor((2, 3), "float32")) -> R.Tensor((4, 9), "float32"):
        return R.tile(x, (2, 3))

    @R.function
    def test_repeat_axis(x: R.Tensor((3,), "float32")) -> R.Tensor((6,), "float32"):
        return R.repeat(x, repeats=2, axis=0)


class TestExtendedOperators:
    """Test class for extended operator coverage."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.ir_mod = ExtendedOperatorsModule
        cls.converter = RelaxToPyFuncConverter(cls.ir_mod)

    def test_unary_operations(self):
        """Test unary operations."""
        converted_ir_mod = self.converter.convert(
            ["test_abs", "test_neg", "test_exp", "test_log", "test_sqrt"]
        )

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

        # Test abs
        result = converted_ir_mod.__pyfuncs__["test_abs"](x)
        expected = torch.abs(x)
        assert torch.allclose(result, expected)

        # Test negative
        result = converted_ir_mod.__pyfuncs__["test_neg"](x)
        expected = torch.neg(x)
        assert torch.allclose(result, expected)

        # Test exp
        result = converted_ir_mod.__pyfuncs__["test_exp"](x)
        expected = torch.exp(x)
        assert torch.allclose(result, expected)

        # Test log (with positive values)
        x_pos = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        result = converted_ir_mod.__pyfuncs__["test_log"](x_pos)
        expected = torch.log(x_pos)
        assert torch.allclose(result, expected)

        # Test sqrt
        result = converted_ir_mod.__pyfuncs__["test_sqrt"](x_pos)
        expected = torch.sqrt(x_pos)
        assert torch.allclose(result, expected)

    def test_trigonometric_operations(self):
        """Test trigonometric operations."""
        converted_ir_mod = self.converter.convert(
            ["test_sin", "test_cos", "test_tanh", "test_sigmoid"]
        )

        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float32)

        # Test sin
        result = converted_ir_mod.__pyfuncs__["test_sin"](x)
        expected = torch.sin(x)
        assert torch.allclose(result, expected)

        # Test cos
        result = converted_ir_mod.__pyfuncs__["test_cos"](x)
        expected = torch.cos(x)
        assert torch.allclose(result, expected)

        # Test tanh
        result = converted_ir_mod.__pyfuncs__["test_tanh"](x)
        expected = torch.tanh(x)
        assert torch.allclose(result, expected)

        # Test sigmoid
        result = converted_ir_mod.__pyfuncs__["test_sigmoid"](x)
        expected = torch.sigmoid(x)
        assert torch.allclose(result, expected)

    def test_comparison_operations(self):
        """Test comparison operations not covered in ComprehensiveTestModule."""
        converted_ir_mod = self.converter.convert(["test_less", "test_not_equal"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        y = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32)

        # Test less
        result = converted_ir_mod.__pyfuncs__["test_less"](x, y)
        expected = torch.lt(x, y)
        assert torch.equal(result, expected)

        # Test not equal
        result = converted_ir_mod.__pyfuncs__["test_not_equal"](x, y)
        expected = torch.ne(x, y)
        assert torch.equal(result, expected)

    def test_binary_operations(self):
        """Test binary operations."""
        converted_ir_mod = self.converter.convert(
            [
                "test_multiply",
                "test_divide",
                "test_power",
                "test_maximum",
                "test_minimum",
                "test_subtract",
            ]
        )

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        y = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32)

        # Test multiply
        result = converted_ir_mod.__pyfuncs__["test_multiply"](x, y)
        expected = torch.mul(x, y)
        assert torch.allclose(result, expected)

        # Test divide
        result = converted_ir_mod.__pyfuncs__["test_divide"](x, y)
        expected = torch.div(x, y)
        assert torch.allclose(result, expected)

        # Test power
        result = converted_ir_mod.__pyfuncs__["test_power"](x, y)
        expected = torch.pow(x, y)
        assert torch.allclose(result, expected)

        # Test maximum
        result = converted_ir_mod.__pyfuncs__["test_maximum"](x, y)
        expected = torch.maximum(x, y)
        assert torch.allclose(result, expected)

        # Test minimum
        result = converted_ir_mod.__pyfuncs__["test_minimum"](x, y)
        expected = torch.minimum(x, y)
        assert torch.allclose(result, expected)

        # Test subtract
        result = converted_ir_mod.__pyfuncs__["test_subtract"](x, y)
        expected = torch.sub(x, y)
        assert torch.allclose(result, expected)

    def test_tensor_operations(self):
        """Test tensor operations not covered in ComprehensiveTestModule."""
        converted_ir_mod = self.converter.convert(["test_transpose_2d"])

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)

        # Test transpose
        result = converted_ir_mod.__pyfuncs__["test_transpose_2d"](x)
        expected = torch.transpose(x, 0, 1)
        assert torch.allclose(result, expected)
        assert result.shape == (4, 2)

    def test_reduction_operations(self):
        """Test reduction operations not covered in ComprehensiveTestModule."""
        converted_ir_mod = self.converter.convert(["test_mean_axis", "test_min_axis"])

        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)

        # Test mean
        result = converted_ir_mod.__pyfuncs__["test_mean_axis"](x)
        expected = torch.mean(x, dim=0)
        assert torch.allclose(result, expected)
        assert result.shape == (3,)

        # Test min
        result = converted_ir_mod.__pyfuncs__["test_min_axis"](x)
        expected = torch.min(x, dim=0)[0]
        assert torch.allclose(result, expected)
        assert result.shape == (3,)

    def test_neural_network_operations(self):
        """Test neural network operations not covered in ComprehensiveTestModule."""
        converted_ir_mod = self.converter.convert(
            ["test_gelu_nn", "test_softmax_nn", "test_log_softmax_nn"]
        )

        x = torch.tensor(
            [[-2.0, -1.0, 0.0, 1.0, 2.0], [0.5, 1.5, 2.5, 3.5, 4.5]], dtype=torch.float32
        )

        # Test gelu
        result = converted_ir_mod.__pyfuncs__["test_gelu_nn"](x[0])
        expected = F.gelu(x[0])
        assert torch.allclose(result, expected)

        # Test softmax
        result = converted_ir_mod.__pyfuncs__["test_softmax_nn"](x)
        expected = F.softmax(x, dim=1)
        assert torch.allclose(result, expected)

        # Test log_softmax
        result = converted_ir_mod.__pyfuncs__["test_log_softmax_nn"](x)
        expected = F.log_softmax(x, dim=1)
        assert torch.allclose(result, expected)

    def test_advanced_tensor_operations(self):
        """Test advanced tensor operations with different parameters."""
        converted_ir_mod = self.converter.convert(["test_tile_dims", "test_repeat_axis"])

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)

        # Test tile with different dimensions
        result = converted_ir_mod.__pyfuncs__["test_tile_dims"](x)
        expected = torch.tile(x, (2, 3))
        assert torch.allclose(result, expected)
        assert result.shape == (4, 12)

        # Test repeat with different parameters
        x_1d = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = converted_ir_mod.__pyfuncs__["test_repeat_axis"](x_1d)
        expected = torch.repeat_interleave(x_1d, repeats=2, dim=0)
        assert torch.allclose(result, expected)
        assert result.shape == (6,)


class TestDLPackAndTupleSupport:
    """Test DLPack conversion, tuple handling, and API compatibility features."""

    def test_dlpack_conversion_fallback(self):
        """Test DLPack conversion with numpy fallback."""

        @I.ir_module
        class DLPackTestModule:
            @Ts.function
            def test_tir(
                x: T.Tensor((4,), "float32"),
                y: T.Tensor((4,), "float32"),
                out: T.Tensor((4,), "float32"),
            ):
                for i in range(4):
                    out[i] = x[i] + y[i]

            @R.function
            def test_func(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")) -> R.Tensor(
                (4,), "float32"
            ):
                return R.call_tir(
                    DLPackTestModule.test_tir, (x, y), ty_args=[R.Tensor((4,), "float32")]
                )

        converter = RelaxToPyFuncConverter(DLPackTestModule)
        converted_ir_mod = converter.convert(["test_func"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        y = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["test_func"](x, y)
        expected = torch.add(x, y)

        assert torch.allclose(result, expected), "DLPack conversion with numpy fallback failed"

    def test_tuple_return_handling(self):
        """Test proper handling of tuple returns (e.g., split operation)."""

        @I.ir_module
        class TupleTestModule:
            @R.function
            def test_split(x: R.Tensor((6,), "float32")) -> R.Tuple:
                return R.split(x, indices_or_sections=3, axis=0)

        converter = RelaxToPyFuncConverter(TupleTestModule)
        converted_ir_mod = converter.convert(["test_split"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
        result = converted_ir_mod.__pyfuncs__["test_split"](x)
        expected = torch.split(x, 2, dim=0)

        assert isinstance(result, tuple), "Split should return tuple"
        assert len(result) == len(expected), "Split should return correct number of tensors"
        for r, e in zip(result, expected):
            assert torch.allclose(r, e), "Split tensor values should match"

    def test_tvm_runtime_api_compatibility(self):
        """Test compatibility with tvm_tirx.runtime API instead of deprecated tvm.nd."""

        @I.ir_module
        class RuntimeAPITestModule:
            @Ts.function
            def test_tir(
                x: T.Tensor((3,), "float32"),
                y: T.Tensor((3,), "float32"),
                out: T.Tensor((3,), "float32"),
            ):
                for i in range(3):
                    out[i] = x[i] * y[i]

            @R.function
            def test_func(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")) -> R.Tensor(
                (3,), "float32"
            ):
                return R.call_tir(
                    RuntimeAPITestModule.test_tir, (x, y), ty_args=[R.Tensor((3,), "float32")]
                )

        converter = RelaxToPyFuncConverter(RuntimeAPITestModule)
        converted_ir_mod = converter.convert(["test_func"])

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["test_func"](x, y)
        expected = torch.mul(x, y)

        assert torch.allclose(result, expected)

    def test_packed_function_with_primvalue_args(self):
        """Test packed function calls with Expr arguments."""

        # Register a test packed function
        def test_packed_func(x, axis):
            return x  # Simple identity function

        tvm_ffi.register_global_func("test_packed_func", test_packed_func)

        @I.ir_module
        class PackedFuncTestModule:
            @R.function
            def test_dps(x: R.Tensor((4,), "float32")) -> R.Tensor((4,), "float32"):
                return R.call_dps_packed(
                    _ExternFunc("test_packed_func"),
                    (x, R.const(0)),
                    ty_args=[R.Tensor((4,), "float32")],
                )

        converter = RelaxToPyFuncConverter(PackedFuncTestModule)
        converted_ir_mod = converter.convert(["test_dps"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        result = converted_ir_mod.__pyfuncs__["test_dps"](x)
        expected = x  # Identity function

        assert torch.allclose(result, expected), "Packed function with Expr args failed"

    def test_mixed_tir_and_relax_operations(self):
        """Test mixed TIR and Relax operations in a single function."""

        @I.ir_module
        class MixedOpsTestModule:
            @Ts.function
            def add_tir(
                x: T.Tensor((4,), "float32"),
                y: T.Tensor((4,), "float32"),
                out: T.Tensor((4,), "float32"),
            ):
                for i in range(4):
                    out[i] = x[i] + y[i]

            @R.function
            def test_mixed(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")) -> R.Tensor(
                (4,), "float32"
            ):
                # TIR operation
                tir_result = R.call_tir(
                    MixedOpsTestModule.add_tir, (x, y), ty_args=[R.Tensor((4,), "float32")]
                )
                # Relax operations
                relued = R.nn.relu(tir_result)
                powered = R.power(relued, R.const(2.0))
                return R.nn.gelu(powered)

        converter = RelaxToPyFuncConverter(MixedOpsTestModule)
        converted_ir_mod = converter.convert(["test_mixed"])

        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        y = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

        result = converted_ir_mod.__pyfuncs__["test_mixed"](x, y)

        # Manual computation for expected result
        added = torch.add(x, y)
        relued = F.relu(added)
        powered = torch.pow(relued, 2.0)
        expected = F.gelu(powered)

        assert torch.allclose(result, expected)

    def test_error_handling_improvements(self):
        """Test improved error handling with tensor fallbacks."""

        @I.ir_module
        class ErrorHandlingTestModule:
            @R.function
            def test_error_handling(x: R.Tensor((4,), "float32")) -> R.Tensor((4,), "float32"):
                # This should trigger fallback mechanisms
                return R.nn.relu(x)

        converter = RelaxToPyFuncConverter(ErrorHandlingTestModule)
        converted_ir_mod = converter.convert(["test_error_handling"])

        x = torch.tensor([-2.0, -1.0, 0.0, 1.0], dtype=torch.float32)
        result = converted_ir_mod.__pyfuncs__["test_error_handling"](x)
        expected = F.relu(x)

        assert torch.allclose(result, expected), "Error handling with tensor fallbacks failed"
        assert isinstance(result, torch.Tensor), "Result should be a tensor, not a string"


if __name__ == "__main__":
    tvm_tirx.testing.main()
