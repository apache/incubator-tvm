<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Example NPU Backend

A hands-on example showing how to build a Neural Processing Unit (NPU) backend for TVM's Relax framework using Bring Your Own Codegen (BYOC).

## Context

NPUs are purpose-built accelerators designed around a fixed set of operations common in neural network inference, such as matrix multiplication, convolution, and activation functions. This example shows the architectural patterns you will encounter when building real NPU backends, making it easier to adapt to specific hardware like:

- Mobile NPUs (AMD XDNA, Google Edge TPU, Samsung NPU)
- Dedicated AI chips (Intel Movidius, Qualcomm Hexagon, MediaTek APU)
- Cloud AI accelerators (AWS Inferentia, Google TPU, Microsoft Azure Maia)
- Custom ASIC designs and embedded AI processors

## What This Is

This is an educational template that demonstrates real NPU concepts without requiring actual NPU hardware. It shows developers how to:

- **Pattern-based partitioning**: Identify and group operations that should run on specialized hardware
- **Memory hierarchy management**: Handle different memory tiers (L0/L1/L2/L3) common in NPUs
- **Automatic tiling**: Break large tensors into smaller chunks that fit in on-chip memory
- **Quantization support**: Handle different data precisions efficiently
- **BYOC integration**: Connect custom backends to TVM's compilation pipeline

## Building TVM with Example NPU Support

Add the following flags when configuring TVM with CMake:

```bash
cmake -DTVM_USE_EXAMPLE_NPU_CODEGEN=ON -DTVM_USE_EXAMPLE_NPU_RUNTIME=ON ..
```

## Quick Start

```python
import tvm
from tvm import relax
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import FuseOpsByPattern, RunCodegen

# Import to register patterns
import tvm.relax.backend.contrib.example_npu

# Get available patterns
patterns = get_patterns_with_prefix("example_npu")
print(f"Available patterns: {[p.name for p in patterns]}")

# Your model gets automatically partitioned
# Operations matching patterns get fused into "Composite" functions
# Those get lowered to the example NPU backend
```

The snippet above shows how to discover registered patterns. A minimal runnable example that demonstrates the BYOC flow (partition -> merge -> codegen) looks like this:

```python
import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.relax.backend.pattern_registry import get_patterns_with_prefix
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions, RunCodegen
import tvm.relax.backend.contrib.example_npu  # registers patterns


@tvm.script.ir_module
class MatmulReLU:
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


mod = MatmulReLU
patterns = get_patterns_with_prefix("example_npu")

# Apply partitioning and codegen annotation
mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(mod)
mod = MergeCompositeFunctions()(mod)
mod = RunCodegen()(mod)

print(mod)
```

A compact visualization of the BYOC flow:

```
Model source (Relax)
	│
	▼
Pattern-based partition (FuseOpsByPattern)
	│
	▼
Composite functions (MergeCompositeFunctions)
	│
	▼
Lower/Codegen for example NPU (RunCodegen / relax.ext.example_npu)
	│
	▼
Runtime dispatch to NPU runtime (runtime.ExampleNPUJSONRuntimeCreate)
```

## Supported Operations

The backend recognizes these common neural network patterns:

### Core Operations
- `example_npu.dense` - Dense/fully connected layers
- `example_npu.matmul` - Matrix multiplication operations
- `example_npu.conv1d` - 1D convolution for sequence processing
- `example_npu.conv2d` - 2D convolution for image processing
- `example_npu.depthwise_conv2d` - Depthwise separable convolutions
- `example_npu.max_pool2d` - 2D max pooling
- `example_npu.avg_pool2d` - 2D average pooling
- `example_npu.batch_norm` - Batch normalization
- `example_npu.softmax` - Softmax
- `example_npu.add` - Element-wise addition
- `example_npu.multiply` - Element-wise multiplication
- `example_npu.subtract` - Element-wise subtraction
- `example_npu.divide` - Element-wise division
- `example_npu.relu` - ReLU activation
- `example_npu.gelu` - Gaussian Error Linear Unit
- `example_npu.quantize` - Quantization
- `example_npu.dequantize` - Dequantization

### Build-dependent Operations
These patterns are registered only when the corresponding Relax op is present
in the TVM build:
- `example_npu.relu6` - ReLU6 activation (`relax.nn.relu6`)
- `example_npu.sigmoid` - Sigmoid activation (`relax.nn.sigmoid`)
- `example_npu.tanh` - Hyperbolic tangent (`relax.nn.tanh`)

### Fused Patterns
- `example_npu.conv2d_relu_fused` - Optimized Conv2D+ReLU fusion


## Files

### Backend Implementation
- `patterns.py` - Defines which operations get fused together, along with pattern metadata and architectural annotations used by the partitioner. Includes operator availability checking and NPU-specific constraints.
- `__init__.py` - Registers the backend and its BYOC entry points with TVM so the compiler can discover and use the example NPU.

### Runtime Implementation
- `src/runtime/extra/contrib/example_npu/example_npu_runtime.cc` - C++ runtime implementation that handles JSON-based graph execution for the NPU backend.

### Tests and Examples
- `tests/python/contrib/test_example_npu.py` - Comprehensive test suite containing example IRModules (e.g. `MatmulReLU`, `Conv2dReLU`) and demonstrating the complete BYOC flow from pattern registration to runtime execution.

## Status / Build

- The example backend is an educational, CPU-backed emulation. It does not require real NPU hardware.
- Tests are skipped automatically when the example codegen/runtime are not built into TVM. The test checks for the presence of these global functions before running:

```python
import tvm
import tvm_ffi
has_codegen = tvm_ffi.get_global_func("relax.ext.example_npu", True)
has_runtime = tvm_ffi.get_global_func("runtime.ExampleNPUJSONRuntimeCreate", True)
has_example_npu = has_codegen and has_runtime
```

If `has_example_npu` is False, tests are skipped. This ensures compatibility across different TVM build configurations.

## Testing

Run the tests to see it in action:

```bash
pytest tests/python/contrib/test_example_npu.py -v
```

Tests are skipped if the backend isn't built — see the test file for the exact runtime/codegen checks.

The test suite includes:
- Pattern registration verification (checks that core patterns are available)
- Graph partitioning validation (ensures operations get grouped correctly)
- End-to-end execution testing (verifies runtime integration)
- Build-dependent pattern verification (confirms build-dependent ops register when present)

### Example output

When you run the quick-start snippet or the test, you should see output similar to the following (truncated for brevity):

```
Available patterns: ['example_npu.dense', 'example_npu.matmul', 'example_npu.conv1d', 'example_npu.conv2d', 'example_npu.depthwise_conv2d', 'example_npu.max_pool2d', 'example_npu.avg_pool2d', 'example_npu.batch_norm', 'example_npu.relu', 'example_npu.add', 'example_npu.multiply', 'example_npu.conv2d_relu_fused']

Relax IRModule
def @main(...) -> ...
	%0 = call_extern("relax.ext.example_npu", ...)

# composite functions
def @composite_0(...) /* Composite */ = ...
```

This shows the registered patterns and that matched subgraphs were turned into composite functions and lowered to the example NPU codegen/runtime.

## Key Features Demonstrated

### NPU Architectural Concepts
- **Multi-tier memory hierarchy**: SRAM (256KB), CMX (512KB), and DRAM management
- **Tiling constraints**: 32x32 tiles with 16-element vectors for optimal NPU utilization
- **Quantization support**: INT8/INT16 for inference acceleration, mixed precision handling
- **Specialized execution units**: Matrix engines (16x16), vector units (64-wide), pooling units
- **Power management**: Support for different power modes (high_performance, balanced, low_power)

### Pattern Matching Features
- **Memory constraint hooks**: Placeholder checks where a real backend would reject tensors that exceed on-chip memory; the example accepts all
- **Fusion opportunities**: Identifies conv+activation and other beneficial fusions
- **Layout preferences**: NHWC channel-last layouts preferred by NPUs

### Error Handling
- **Robust exception handling**: Catches specific exception types instead of generic exceptions
- **Comprehensive testing**: Validates both successful cases and error conditions

## Learn More

This backend serves as both a working example and educational resource for understanding NPU integration patterns. The implementation demonstrates vendor-neutral concepts that apply across different NPU architectures, making it a valuable starting point for real NPU backend development.
