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
Example NPU Pattern Table with Architectural Concepts

This module demonstrates NPU-specific architectural patterns that are common
across different NPU vendors, including memory hierarchy, quantization,
tiling, and fusion strategies.
"""

from typing import ClassVar

from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.transform import PatternCheckContext
from tvm_tirx.ir import Op

from ...pattern_registry import register_patterns


# NPU-specific configuration constants (vendor-neutral)
class NPUConfig:
    """NPU architectural parameters common across vendors"""

    # Memory hierarchy sizes (in KB) - typical NPU values
    SRAM_SIZE_KB = 256  # On-chip SRAM/scratchpad
    CMX_SIZE_KB = 512  # Compute memory (near compute units)

    # Tiling constraints
    TILE_HEIGHT = 32
    TILE_WIDTH = 32
    VECTOR_SIZE = 16

    # Supported data types for NPU acceleration
    SUPPORTED_DTYPES: ClassVar[list[str]] = ["int8", "int16", "float16", "float32"]
    QUANTIZED_DTYPES: ClassVar[list[str]] = ["int8", "int16"]

    # NPU execution units
    MATRIX_ENGINE_SIZE = 16  # MxN matrix engine
    VECTOR_ENGINE_WIDTH = 64  # Vector processing width

    # Power modes
    POWER_MODES: ClassVar[list[str]] = ["high_performance", "balanced", "low_power"]


def _check_npu_memory_constraints(
    context: PatternCheckContext,  # pylint: disable=unused-argument
) -> bool:
    """
    Placeholder for NPU memory hierarchy constraint checking.

    A real implementation would inspect the annotated expression's
    TensorType to verify the tensor fits within the NPU's
    on-chip SRAM (L1) or compute memory (L2/CMX). Tensors that
    exceed on-chip capacity require tiling before offload.
    """
    return True


def _check_npu_quantization(
    context: PatternCheckContext,  # pylint: disable=unused-argument
) -> bool:
    """
    Placeholder for NPU quantization requirement checking.

    A real implementation would verify the op's dtype falls within
    the set supported by the NPU (e.g. int8, int16, float16, float32)
    and reject ops with unsupported dtypes so they fall back to CPU.
    """
    return True


def conv2d_relu_fused_pattern():
    """
    NPU-optimized Conv2D+ReLU fusion pattern.

    This is a key NPU optimization - fusing convolution with activation
    avoids memory traffic between operations.
    """

    def _make_conv2d_relu_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        conv = is_op("relax.nn.conv2d")(input_tensor, weight)
        relu = is_op("relax.nn.relu")(conv)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "conv": conv,
            "root": relu,
        }
        return relu, annotations

    def _check_conv2d_relu(context: PatternCheckContext) -> bool:
        """Check if Conv2D+ReLU fusion is beneficial for NPU"""
        if not _check_npu_memory_constraints(context):
            return False
        if not _check_npu_quantization(context):
            return False
        return True

    return ("example_npu.conv2d_relu_fused", *_make_conv2d_relu_pattern(), _check_conv2d_relu)


def matmul_relu_fused_pattern():
    """
    NPU-optimized MatMul+ReLU fusion pattern.

    Fusing the matrix engine output with the activation unit avoids a
    write/read round-trip through L1 SRAM, mirroring the conv2d+relu
    fusion below.
    """

    def _make_matmul_relu_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        matmul = is_op("relax.matmul")(input_tensor, weight)
        relu = is_op("relax.nn.relu")(matmul)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "matmul": matmul,
            "root": relu,
        }
        return relu, annotations

    def _check_matmul_relu(context: PatternCheckContext) -> bool:
        """Check if MatMul+ReLU fusion is beneficial for NPU"""
        if not _check_npu_memory_constraints(context):
            return False
        if not _check_npu_quantization(context):
            return False
        return True

    return ("example_npu.matmul_relu_fused", *_make_matmul_relu_pattern(), _check_matmul_relu)


def matmul_patterns():
    """
    NPU-optimized matrix multiplication patterns.

    NPUs typically have dedicated matrix engines (systolic arrays,
    tensor cores) that require specific layouts and sizes.
    """

    def _make_matmul_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.matmul")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
        }
        return output, annotations

    def _check_matmul(context: PatternCheckContext) -> bool:
        """Check if matmul can use NPU matrix engine"""
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    def _matmul_pattern(pattern_name):
        return (pattern_name, *_make_matmul_pattern(), _check_matmul)

    # Register both common names used for matrix multiplication in patterns/tests
    return [
        _matmul_pattern("example_npu.dense"),
        _matmul_pattern("example_npu.matmul"),
    ]


def conv1d_patterns():
    """
    1D Convolution patterns optimized for NPU execution.

    NPUs handle 1D convolution by mapping to 2D operations
    or using specialized 1D processing units.
    """

    def _make_conv1d_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv1d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
        }
        return output, annotations

    def _check_conv1d(context: PatternCheckContext) -> bool:
        """Check if conv1d can use NPU vector engine"""
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    def _conv1d_pattern(pattern_name):
        return (pattern_name, *_make_conv1d_pattern(), _check_conv1d)

    return [_conv1d_pattern("example_npu.conv1d")]


def conv2d_patterns():
    """
    2D Convolution patterns with NPU tiling and memory management.

    2D convolution is the most important NPU operation, with
    dedicated hardware for efficient processing.
    """

    def _make_conv2d_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv2d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
        }
        return output, annotations

    def _check_conv2d(context: PatternCheckContext) -> bool:
        """Check conv2d NPU constraints"""
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    def _conv2d_pattern(pattern_name):
        return (pattern_name, *_make_conv2d_pattern(), _check_conv2d)

    return [_conv2d_pattern("example_npu.conv2d")]


def depthwise_conv2d_patterns():
    """
    Depthwise convolution - critical for mobile NPUs.

    Many NPUs have specialized units for depthwise operations
    used in MobileNet-style architectures.
    """

    def _make_depthwise_pattern():
        input_tensor = wildcard()
        weight = wildcard()
        output = is_op("relax.nn.conv2d")(input_tensor, weight)

        annotations = {
            "input": input_tensor,
            "weight": weight,
            "root": output,
        }
        return output, annotations

    def _check_depthwise(context: PatternCheckContext) -> bool:
        """Check if this is a depthwise conv that NPU can accelerate"""
        conv_call = context.annotated_expr["root"]
        # groups > 1 distinguishes depthwise/grouped conv from standard conv2d.
        # True depthwise has groups == in_channels; we accept any grouped variant
        # here since the NPU's depthwise unit handles all grouped convolutions.
        if conv_call.attrs.groups <= 1:
            return False
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    return [("example_npu.depthwise_conv2d", *_make_depthwise_pattern(), _check_depthwise)]


def pooling_patterns():
    """
    Pooling operations with NPU memory streaming.

    NPUs often process pooling with the convolution engine
    or dedicated pooling units.
    """

    def _make_maxpool2d_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.max_pool2d")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _make_avgpool2d_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.avg_pool2d")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _check_pooling(context: PatternCheckContext) -> bool:
        """Check pooling NPU constraints"""
        return _check_npu_memory_constraints(context)

    return [
        ("example_npu.max_pool2d", *_make_maxpool2d_pattern(), _check_pooling),
        ("example_npu.avg_pool2d", *_make_avgpool2d_pattern(), _check_pooling),
    ]


def batch_norm_patterns():
    """
    Batch normalization - often fused with conv on NPUs.

    NPUs typically fuse BN into convolution to avoid
    separate memory passes.
    """

    def _make_batch_norm_pattern():
        input_tensor = wildcard()
        gamma = wildcard()
        beta = wildcard()
        moving_mean = wildcard()
        moving_var = wildcard()

        output = is_op("relax.nn.batch_norm")(input_tensor, gamma, beta, moving_mean, moving_var)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _check_batch_norm(context: PatternCheckContext) -> bool:
        """Check if batch norm should be offloaded or fused"""
        return _check_npu_quantization(context)

    return [("example_npu.batch_norm", *_make_batch_norm_pattern(), _check_batch_norm)]


def softmax_patterns():
    """
    Softmax - used in classification heads and attention mechanisms.

    NPUs typically implement softmax via dedicated hardware or
    a combination of exp, sum, and divide operations.
    """

    def _make_softmax_pattern():
        input_tensor = wildcard()
        output = is_op("relax.nn.softmax")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _check_softmax(context: PatternCheckContext) -> bool:
        """Check if softmax can use NPU activation unit"""
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    patterns = []
    try:
        Op.get("relax.nn.softmax")
        patterns.append(("example_npu.softmax", *_make_softmax_pattern(), _check_softmax))
    except (KeyError, AttributeError):
        pass

    return patterns


def activation_patterns():
    """
    NPU activation functions with specialized hardware.

    NPUs have dedicated activation units that can handle
    various functions efficiently.
    """

    def _make_activation_pattern(op_name: str):
        def _pattern():
            input_tensor = wildcard()
            output = is_op(op_name)(input_tensor)

            annotations = {
                "input": input_tensor,
                "root": output,
            }
            return output, annotations

        return _pattern

    def _check_activation(context: PatternCheckContext) -> bool:
        """Check if activation can use NPU activation unit"""
        return _check_npu_quantization(context)

    activations = [
        ("example_npu.relu", "relax.nn.relu"),
        ("example_npu.relu6", "relax.nn.relu6"),
        ("example_npu.sigmoid", "relax.nn.sigmoid"),
        ("example_npu.tanh", "relax.nn.tanh"),
        ("example_npu.gelu", "relax.nn.gelu"),
    ]

    patterns = []
    for pattern_name, op_name in activations:
        try:
            Op.get(op_name)
        except (KeyError, AttributeError):
            continue

        pattern_fn = _make_activation_pattern(op_name)
        patterns.append((pattern_name, *pattern_fn(), _check_activation))

    return patterns


def elementwise_patterns():
    """
    Element-wise operations that NPUs can vectorize.

    NPUs process element-wise ops using vector units
    with SIMD capabilities.
    """

    def _make_elementwise_pattern(op_name: str):
        def _pattern():
            input1 = wildcard()
            input2 = wildcard()
            output = is_op(op_name)(input1, input2)

            annotations = {
                "input1": input1,
                "input2": input2,
                "root": output,
            }
            return output, annotations

        return _pattern

    def _check_elementwise(context: PatternCheckContext) -> bool:
        """Check if elementwise op can use NPU vector unit"""
        return _check_npu_memory_constraints(context) and _check_npu_quantization(context)

    ops = ["relax.add", "relax.multiply", "relax.subtract", "relax.divide"]
    patterns = []
    for op in ops:
        try:
            Op.get(op)
        except (KeyError, AttributeError):
            continue

        op_short = op.split(".")[-1]
        pattern_fn = _make_elementwise_pattern(op)
        patterns.append((f"example_npu.{op_short}", *pattern_fn(), _check_elementwise))

    return patterns


def quantization_patterns():
    """
    Quantization/dequantization patterns for NPU.

    NPUs need explicit quantization boundaries to switch
    between precision levels.
    """

    def _make_quantize_pattern():
        input_tensor = wildcard()
        output = is_op("relax.quantize")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _make_dequantize_pattern():
        input_tensor = wildcard()
        output = is_op("relax.dequantize")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": output,
        }
        return output, annotations

    def _check_quantization(
        context: PatternCheckContext,  # pylint: disable=unused-argument
    ) -> bool:
        """Check quantization operations"""
        return True

    patterns = []

    try:
        Op.get("relax.quantize")
        patterns.append(("example_npu.quantize", *_make_quantize_pattern(), _check_quantization))
    except (KeyError, AttributeError):
        pass

    try:
        Op.get("relax.dequantize")
        patterns.append(
            ("example_npu.dequantize", *_make_dequantize_pattern(), _check_quantization)
        )
    except (KeyError, AttributeError):
        pass

    return patterns


# Register all NPU patterns with architectural awareness
# register_patterns priority: patterns that appear LATER in the list win.
# So we place general / standalone patterns first, and fused (more
# specific) patterns last so they take precedence over their constituents.
register_patterns(
    [
        *quantization_patterns(),
        *elementwise_patterns(),
        *activation_patterns(),
        *softmax_patterns(),
        *batch_norm_patterns(),
        *pooling_patterns(),
        *matmul_patterns(),
        *conv1d_patterns(),
        # Plain conv2d is more general than depthwise (groups>1); list
        # plain first so depthwise wins on grouped convs.
        *conv2d_patterns(),
        *depthwise_conv2d_patterns(),
        # Fused patterns last (highest priority).
        matmul_relu_fused_pattern(),
        conv2d_relu_fused_pattern(),
    ]
)
