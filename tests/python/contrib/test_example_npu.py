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
"""
Tests for Example NPU Backend

This test file demonstrates how to test a custom NPU backend
implementation using TVM's testing infrastructure.
"""

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions, RunCodegen


@tvm_tirx.script.ir_module
class MatmulReLU:
    """Example module with matrix multiplication and ReLU"""

    @R.function
    def main(
        x: R.Tensor((2, 4), "float32"),
        w: R.Tensor((4, 8), "float32"),
    ) -> R.Tensor((2, 8), "float32"):
        with R.dataflow():
            y = relax.op.matmul(x, w)
            z = relax.op.nn.relu(y)
            R.output(z)
        return z


@tvm_tirx.script.ir_module
class Conv2dReLU:
    """Example module with 2D convolution and ReLU"""

    @R.function
    def main(
        x: R.Tensor((1, 3, 32, 32), "float32"),
        w: R.Tensor((16, 3, 3, 3), "float32"),
    ) -> R.Tensor((1, 16, 30, 30), "float32"):
        with R.dataflow():
            y = relax.op.nn.conv2d(x, w)
            z = relax.op.nn.relu(y)
            R.output(z)
        return z


@tvm_tirx.script.ir_module
class MultipleOps:
    """Example module with multiple operations that can be fused"""

    @R.function
    def main(
        x: R.Tensor((1, 16, 32, 32), "float32"),
    ) -> R.Tensor((1, 16, 16, 16), "float32"):
        with R.dataflow():
            # First ReLU
            y = relax.op.nn.relu(x)
            # Max pooling
            z = relax.op.nn.max_pool2d(y, pool_size=(2, 2), strides=(2, 2))
            # Second ReLU
            out = relax.op.nn.relu(z)
            R.output(out)
        return out


@tvm_tirx.script.ir_module
class Softmax:
    """Example module with softmax"""

    @R.function
    def main(x: R.Tensor((2, 8), "float32")) -> R.Tensor((2, 8), "float32"):
        with R.dataflow():
            z = relax.op.nn.softmax(x)
            R.output(z)
        return z


# Check if the example NPU runtime is available
has_example_npu_codegen = tvm_ffi.get_global_func("relax.ext.example_npu", True)
has_example_npu_runtime = tvm_ffi.get_global_func("runtime.ExampleNPUJSONRuntimeCreate", True)
has_example_npu = has_example_npu_codegen and has_example_npu_runtime

example_npu_enabled = pytest.mark.skipif(
    not has_example_npu,
    reason="Example NPU backend not enabled. Compile with the example NPU runtime.",
)


def test_example_npu_patterns_registered():
    """Test that all expected patterns are registered"""
    import tvm.relax.backend.contrib.example_npu  # noqa: F401

    patterns = get_patterns_with_prefix("example_npu")
    pattern_names = {p.name for p in patterns}

    # Core patterns that should always be available
    core_patterns = {
        "example_npu.dense",
        "example_npu.matmul",
        "example_npu.conv1d",
        "example_npu.conv2d",
        "example_npu.max_pool2d",
    }

    assert core_patterns.issubset(pattern_names), (
        f"Missing core patterns: {core_patterns - pattern_names}"
    )

    # Check that at least some activation patterns are available
    activation_patterns = {name for name in pattern_names if "relu" in name or "sigmoid" in name}
    assert len(activation_patterns) > 0, "No activation patterns found"


@example_npu_enabled
def test_example_npu_matmul_relu_partitioning():
    """Test graph partitioning for MatMul + ReLU pattern"""
    import tvm.relax.backend.contrib.example_npu  # noqa: F401

    mod = MatmulReLU
    patterns = get_patterns_with_prefix("example_npu")

    # Partition the graph
    partitioned_mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=False)(mod)
    partitioned_mod = MergeCompositeFunctions()(partitioned_mod)

    # Verify partitioning happened
    assert partitioned_mod is not None

    # Check that composite functions were created
    for gvar, func in partitioned_mod.functions.items():
        if gvar.name_hint != "main":
            # This should be a composite function
            assert "Composite" in str(func)


@example_npu_enabled
def test_example_npu_conv2d_relu_partitioning():
    """Test graph partitioning for Conv2D + ReLU pattern"""
    import tvm.relax.backend.contrib.example_npu  # noqa: F401

    mod = Conv2dReLU
    patterns = get_patterns_with_prefix("example_npu")

    # Partition the graph
    partitioned_mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=False)(mod)
    partitioned_mod = MergeCompositeFunctions()(partitioned_mod)

    assert partitioned_mod is not None


@example_npu_enabled
def test_example_npu_multiple_ops():
    """Test partitioning with multiple fusable operations"""
    import tvm.relax.backend.contrib.example_npu  # noqa: F401

    mod = MultipleOps
    patterns = get_patterns_with_prefix("example_npu")

    # Partition the graph
    partitioned_mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=False)(mod)
    partitioned_mod = MergeCompositeFunctions()(partitioned_mod)

    assert partitioned_mod is not None


@example_npu_enabled
def test_example_npu_softmax_partitioning():
    """Test graph partitioning for softmax pattern"""
    import tvm.relax.backend.contrib.example_npu  # noqa: F401

    mod = Softmax
    patterns = get_patterns_with_prefix("example_npu")

    partitioned_mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=False)(mod)
    partitioned_mod = MergeCompositeFunctions()(partitioned_mod)

    assert partitioned_mod is not None

    for gvar, func in partitioned_mod.functions.items():
        if gvar.name_hint != "main":
            assert "Composite" in str(func)


@example_npu_enabled
def test_example_npu_codegen():
    """Test code generation for the example NPU backend"""
    import tvm.relax.backend.contrib.example_npu  # noqa: F401

    mod = MatmulReLU
    patterns = get_patterns_with_prefix("example_npu")

    # Partition and generate code
    partitioned_mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
    partitioned_mod = MergeCompositeFunctions()(partitioned_mod)
    partitioned_mod = RunCodegen()(partitioned_mod)

    assert partitioned_mod is not None

    # The module should now contain external function calls
    main_func = partitioned_mod["main"]
    assert main_func is not None


@example_npu_enabled
def test_example_npu_runtime_execution():
    """Test end-to-end execution with the example NPU runtime"""

    # Create simple test inputs
    np.random.seed(42)
    x_np = np.random.randn(2, 4).astype("float32")
    w_np = np.random.randn(4, 8).astype("float32")

    # Expected output (computed with NumPy)
    expected = np.maximum(0, np.matmul(x_np, w_np))

    # Build and run with example NPU backend
    mod = MatmulReLU
    patterns = get_patterns_with_prefix("example_npu")

    # Apply transformations
    mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
    mod = MergeCompositeFunctions()(mod)
    mod = RunCodegen()(mod)

    # Build the module
    target = tvm_tirx.target.Target("llvm")
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        built = relax.build(mod, target)

    # Create VM and run
    vm = relax.VirtualMachine(built, tvm_tirx.cpu())

    x_tvm = tvm_tirx.runtime.tensor(x_np, tvm_tirx.cpu())
    w_tvm = tvm_tirx.runtime.tensor(w_np, tvm_tirx.cpu())

    result = vm["main"](x_tvm, w_tvm)

    # Verify the result shape is correct (the runtime is a stub and does not compute numerically)
    assert result.numpy().shape == expected.shape


if __name__ == "__main__":
    # Run tests locally for debugging
    test_example_npu_patterns_registered()

    if has_example_npu:
        print("Example NPU backend is available, running tests...")
        test_example_npu_matmul_relu_partitioning()
        test_example_npu_conv2d_relu_partitioning()
        test_example_npu_softmax_partitioning()
        test_example_npu_multiple_ops()
        test_example_npu_codegen()
        test_example_npu_runtime_execution()
        print("All tests passed!")
    else:
        print("Example NPU backend not available. Compile with example NPU runtime to run tests.")
