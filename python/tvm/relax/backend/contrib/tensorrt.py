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

"""Pattern table and partitioning for the TensorRT BYOC backend.

The composite name of each pattern is "tensorrt.<op>", matching the runtime
converter registered under the same name (the converters are keyed by
"tensorrt." + op_name). ``partition_for_tensorrt`` carves the matched subgraphs
out of the module and annotates them for the ``tensorrt`` codegen.
"""

import tvm_tirx
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.expr import ShapeExpr
from tvm.relax.transform import (
    FuseOpsByPattern,
    MergeCompositeFunctions,
    PatternCheckContext,
)
from tvm_tirx.ir import IRModule

from ..pattern_registry import Pattern, get_patterns_with_prefix, register_patterns


def _op_pattern(composite_name: str, op_name: str, num_args: int) -> Pattern:
    """A pattern matching a single op called with ``num_args`` wildcard arguments."""
    args = [wildcard() for _ in range(num_args)]
    return (composite_name, is_op(op_name)(*args), {})


def _resize2d_pattern() -> Pattern:
    """Match the subset of resize2d that the TensorRT converter implements exactly."""

    data = wildcard()
    size = wildcard()
    root = is_op("relax.image.resize2d")(data, size)

    def check(context: PatternCheckContext) -> bool:
        data_expr = context.annotated_expr["data"]
        size_expr = context.annotated_expr["size"]
        resize = context.annotated_expr["root"]

        # Follow an ANF binding when the static ShapeExpr was assigned to a variable.
        while size_expr in context.matched_bindings:
            size_expr = context.matched_bindings[size_expr]

        if not isinstance(size_expr, ShapeExpr) or len(size_expr.values) != 2:
            return False
        if not all(
            isinstance(dim, tvm_tirx.ir.IntImm) and dim.value > 0 for dim in size_expr.values
        ):
            return False

        attrs = resize.attrs
        if attrs.layout != "NCHW":
            return False
        if data_expr.ty.ndim != 4 or data_expr.ty.shape is None:
            return False

        # The runtime's DLPack-to-TensorRT dtype mapper supports only FP16/FP32.  TensorRT Resize
        # itself preserves its input dtype, so a distinct Relax out_dtype is not supported here.
        if data_expr.ty.dtype not in ("float16", "float32"):
            return False
        if resize.ty.dtype != data_expr.ty.dtype:
            return False

        if attrs.method not in ("nearest_neighbor", "linear", "cubic"):
            return False
        if attrs.coordinate_transformation_mode not in (
            "asymmetric",
            "align_corners",
            "half_pixel",
            "pytorch_half_pixel",
        ):
            return False

        # Relax `round` uses ties-to-even, for which TensorRT has no ResizeRoundMode.  The other
        # modes map exactly to TensorRT's FLOOR, CEIL, HALF_UP, and HALF_DOWN modes.
        if attrs.method == "nearest_neighbor" and attrs.rounding_method not in (
            "floor",
            "ceil",
            "round_prefer_ceil",
            "round_prefer_floor",
        ):
            return False

        return True

    return (
        "tensorrt.image.resize2d",
        root,
        {"data": data, "size": size, "root": root},
        check,
    )


def _tensorrt_patterns() -> list[Pattern]:
    patterns: list[Pattern] = []

    # Activations and unary elementwise ops (single tensor argument).
    for composite, op in [
        ("tensorrt.nn.relu", "relax.nn.relu"),
        ("tensorrt.sigmoid", "relax.sigmoid"),
        ("tensorrt.nn.silu", "relax.nn.silu"),
        ("tensorrt.tanh", "relax.tanh"),
        ("tensorrt.exp", "relax.exp"),
        ("tensorrt.log", "relax.log"),
        ("tensorrt.sqrt", "relax.sqrt"),
        ("tensorrt.abs", "relax.abs"),
        ("tensorrt.negative", "relax.negative"),
        ("tensorrt.sin", "relax.sin"),
        ("tensorrt.cos", "relax.cos"),
        ("tensorrt.atan", "relax.atan"),
        ("tensorrt.ceil", "relax.ceil"),
        ("tensorrt.floor", "relax.floor"),
        ("tensorrt.erf", "relax.erf"),
        ("tensorrt.nn.softmax", "relax.nn.softmax"),
        ("tensorrt.nn.batch_flatten", "relax.nn.batch_flatten"),
        ("tensorrt.expand_dims", "relax.expand_dims"),
        ("tensorrt.squeeze", "relax.squeeze"),
        ("tensorrt.transpose", "relax.permute_dims"),
        ("tensorrt.layout_transform", "relax.layout_transform"),
        ("tensorrt.nn.max_pool2d", "relax.nn.max_pool2d"),
        ("tensorrt.nn.avg_pool2d", "relax.nn.avg_pool2d"),
        ("tensorrt.nn.max_pool3d", "relax.nn.max_pool3d"),
        ("tensorrt.nn.avg_pool3d", "relax.nn.avg_pool3d"),
        ("tensorrt.nn.adaptive_avg_pool2d", "relax.nn.adaptive_avg_pool2d"),
        ("tensorrt.sum", "relax.sum"),
        ("tensorrt.prod", "relax.prod"),
        ("tensorrt.max", "relax.max"),
        ("tensorrt.min", "relax.min"),
        ("tensorrt.mean", "relax.mean"),
        ("tensorrt.concatenate", "relax.concat"),
        ("tensorrt.split", "relax.split"),
    ]:
        patterns.append(_op_pattern(composite, op, 1))

    # Binary elementwise ops (two tensor arguments).
    for composite, op in [
        ("tensorrt.add", "relax.add"),
        ("tensorrt.subtract", "relax.subtract"),
        ("tensorrt.multiply", "relax.multiply"),
        ("tensorrt.divide", "relax.divide"),
        ("tensorrt.power", "relax.power"),
        ("tensorrt.maximum", "relax.maximum"),
        ("tensorrt.minimum", "relax.minimum"),
    ]:
        patterns.append(_op_pattern(composite, op, 2))

    # image.resize2d (data + target-size shape argument).
    patterns.append(_resize2d_pattern())

    # Convolutions and matmul (data + weight).
    for composite, op in [
        ("tensorrt.nn.conv1d", "relax.nn.conv1d"),
        ("tensorrt.nn.conv2d", "relax.nn.conv2d"),
        ("tensorrt.nn.conv3d", "relax.nn.conv3d"),
        ("tensorrt.nn.conv2d_transpose", "relax.nn.conv2d_transpose"),
        ("tensorrt.nn.conv3d_transpose", "relax.nn.conv3d_transpose"),
        ("tensorrt.nn.batch_matmul", "relax.matmul"),
        ("tensorrt.reshape", "relax.reshape"),
    ]:
        patterns.append(_op_pattern(composite, op, 2))

    # layer_norm (data, gamma, beta) and clip (data, min, max).
    patterns.append(_op_pattern("tensorrt.nn.layer_norm", "relax.nn.layer_norm", 3))
    patterns.append(_op_pattern("tensorrt.clip", "relax.clip", 3))

    # strided_slice is called either with or without the optional strides argument.
    patterns.append(_op_pattern("tensorrt.strided_slice", "relax.strided_slice", 5))
    patterns.append(_op_pattern("tensorrt.strided_slice", "relax.strided_slice", 4))

    return patterns


register_patterns(_tensorrt_patterns())


def partition_for_tensorrt(mod: IRModule) -> IRModule:
    """Partition the module, offloading TensorRT-supported subgraphs.

    Parameters
    ----------
    mod : tvm_tirx.ir.IRModule
        The module to partition. Bind model parameters (e.g. via
        ``relax.transform.BindParams``) before calling this so that weights are
        available to TensorRT as constants.

    Returns
    -------
    mod : tvm_tirx.ir.IRModule
        The module with TensorRT-supported subgraphs grouped into composite
        functions annotated for the ``tensorrt`` codegen.

    Notes
    -----
    By default, TensorRT builds engines on the first inference call.
    To build and embed engines before exporting the library, enable
    ``build_at_compile_time`` in the ``relax.ext.tensorrt.options`` PassContext
    configuration while running ``RunCodegen``.

    Compile-time building requires the TensorRT runtime and a CUDA GPU on the
    compilation host. Select the deployment GPU before starting code generation,
    for example with ``CUDA_VISIBLE_DEVICES``. Tensor inputs must have static,
    positive dimensions and FP16 or FP32 dtype. Dynamic inputs and INT8
    calibration are not supported by this mode.

    The exported library contains the serialized engines. No inference call or
    disk engine cache is needed before export. On the first inference call,
    the runtime deserializes these plans instead of rebuilding engines.
    Deployment still requires the TensorRT-enabled TVM runtime, CUDA, and a
    compatible GPU and TensorRT version; the plans are not portable across
    arbitrary GPU architectures or TensorRT versions.

    Examples
    --------
    After binding model weights as constants, partition and generate code on the
    compilation GPU before exporting::

        import tvm, tvm_tirx
        from tvm import relax
        from tvm.relax.backend.contrib.tensorrt import partition_for_tensorrt

        partitioned = partition_for_tensorrt(mod)
        with tvm_tirx.ir.transform.PassContext(
            config={
                "relax.ext.tensorrt.options": {
                    "build_at_compile_time": True,
                }
            }
        ):
            offloaded = relax.transform.RunCodegen()(partitioned)
        executable = tvm.compile(offloaded, target="cuda")
        executable.export_library("model.so")

    Load the library and run inference in a new process on the deployment GPU::

        import tvm, tvm_tirx
        from tvm import relax

        executable = tvm_tirx.runtime.load_module("model.so")
        vm = relax.VirtualMachine(executable, tvm_tirx.cuda(0))
        result = vm["main"](*inputs)
    """
    patterns = get_patterns_with_prefix("tensorrt")
    mod = FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=False)(mod)
    mod = MergeCompositeFunctions()(mod)
    return mod
