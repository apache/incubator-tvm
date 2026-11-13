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

"""Pattern table for NNAPI backend"""

from collections.abc import Mapping

from tvm.relax.dpl.pattern import (
    DFPattern,
    is_op,
    wildcard,
)
from tvm.relax.transform import FuseOpsByPattern, MergeCompositeFunctions
from tvm_tirx.ir import IRModule

from ..pattern_registry import get_patterns_with_prefix, register_patterns


def elementwise_binary_patterns() -> list[tuple[str, DFPattern, Mapping[str, DFPattern]]]:
    """
    Returns a list of tuples representing elementwise binary operation patterns mapped
    between NNAPI and Relax frameworks.
    """

    def _elementwise_binary_pattern(
        pattern_name: str,
        op_name: str,
    ) -> tuple[str, DFPattern, Mapping[str, DFPattern]]:
        input0 = wildcard()
        input1 = wildcard()

        pattern = is_op(op_name)(input0, input1)

        return (pattern_name, pattern, {})

    return [
        _elementwise_binary_pattern("nnapi.add", "relax.add"),
        _elementwise_binary_pattern("nnapi.mul", "relax.multiply"),
        _elementwise_binary_pattern("nnapi.div", "relax.divide"),
        _elementwise_binary_pattern("nnapi.sub", "relax.subtract"),
        _elementwise_binary_pattern("nnapi.pow", "relax.power"),
        _elementwise_binary_pattern("nnapi.equal", "relax.equal"),
        _elementwise_binary_pattern("nnapi.greater", "relax.greater"),
        _elementwise_binary_pattern("nnapi.greater_equal", "relax.greater_equal"),
        _elementwise_binary_pattern("nnapi.less", "relax.less"),
        _elementwise_binary_pattern("nnapi.less_equal", "relax.less_equal"),
        _elementwise_binary_pattern("nnapi.not_equal", "relax.not_equal"),
        _elementwise_binary_pattern("nnapi.maximum", "relax.maximum"),
        _elementwise_binary_pattern("nnapi.minimum", "relax.minimum"),
    ]


def unary_patterns() -> list[tuple[str, DFPattern, Mapping[str, DFPattern]]]:
    """
    Returns a list of tuples representing unary operation patterns mapped
    between NNAPI and Relax frameworks.
    """

    def _unary_pattern(
        pattern_name: str, op_name: str
    ) -> tuple[str, DFPattern, Mapping[str, DFPattern]]:
        input0 = wildcard()
        pattern = is_op(op_name)(input0)
        return (pattern_name, pattern, {})

    return [
        _unary_pattern("nnapi.floor", "relax.floor"),
        _unary_pattern("nnapi.relu", "relax.nn.relu"),
        _unary_pattern("nnapi.logistic", "relax.sigmoid"),
        _unary_pattern("nnapi.softmax", "relax.nn.softmax"),
        _unary_pattern("nnapi.tanh", "relax.tanh"),
        _unary_pattern("nnapi.abs", "relax.abs"),
        _unary_pattern("nnapi.exp", "relax.exp"),
        _unary_pattern("nnapi.log", "relax.log"),
        _unary_pattern("nnapi.neg", "relax.negative"),
        _unary_pattern("nnapi.cast", "relax.astype"),
        _unary_pattern("nnapi.sqrt", "relax.sqrt"),
        _unary_pattern("nnapi.rsqrt", "relax.rsqrt"),
    ]


def matmul_pattern() -> tuple[str, DFPattern, Mapping[str, DFPattern]]:
    """
    Returns a tuple representing matmul operation patterns mapped
    between NNAPI and Relax frameworks.
    """
    input0 = wildcard()
    input1 = wildcard()
    pattern = is_op("relax.matmul")(input0, input1)
    return ("nnapi.batch_matmul", pattern, {})


def permute_dims_pattern() -> tuple[str, DFPattern, Mapping[str, DFPattern]]:
    """
    Returns a tuple representing permute operation patterns mapped
    between NNAPI and Relax frameworks.
    """
    input0 = wildcard()
    pattern = is_op("relax.permute_dims")(input0)
    return ("nnapi.transpose", pattern, {})


def astype_pattern() -> tuple[str, DFPattern, Mapping[str, DFPattern]]:
    """
    Returns a tuple representing astype operation patterns mapped
    between NNAPI and Relax frameworks.
    """
    input0 = wildcard().has_dtype("float16") | wildcard().has_dtype("float32")
    pattern = is_op("relax.astype")(input0).has_dtype("float16") | is_op("relax.astype")(
        input0
    ).has_dtype("float32")

    return ("nnapi.cast", pattern, {})


def mean_pattern() -> tuple[str, DFPattern, Mapping[str, DFPattern]]:
    """
    Returns a tuple representing mean operation patterns mapped
    between NNAPI and Relax frameworks.
    """
    input0 = wildcard()
    pattern = is_op("relax.mean")(input0)

    return ("nnapi.mean", pattern, {})


def conv2d_pattern() -> tuple[str, DFPattern, Mapping[str, DFPattern]]:
    """
    Returns a tuple representing conv2d operation patterns mapped
    between NNAPI and Relax frameworks.
    """
    input0 = wildcard()
    input1 = wildcard()
    input2 = wildcard()
    conv = is_op("relax.nn.conv2d")(input0, input1)
    pattern = is_op("relax.add")(conv, input2)
    return ("nnapi.conv2d", pattern, {})


def max_pool2d_pattern() -> tuple[str, DFPattern, Mapping[str, DFPattern]]:
    """
    Returns a tuple representing max_pool2d operation patterns mapped
    between NNAPI and Relax frameworks.
    """
    input0 = wildcard()
    pattern = is_op("relax.nn.max_pool2d")(input0)
    return ("nnapi.max_pool_2d", pattern, {})


register_patterns(
    [
        *elementwise_binary_patterns(),
        *unary_patterns(),
        matmul_pattern(),
        permute_dims_pattern(),
        astype_pattern(),
        mean_pattern(),
        conv2d_pattern(),
        max_pool2d_pattern(),
    ]
)


def min_feature_level(pattern_name: str) -> int:
    """
    Returns the minimum feature level required to support a given NNAPI operation pattern.

    Args:
        pattern_name (str): The name of the NNAPI operation pattern
        (e.g., "nnapi.add", "nnapi.conv2d").

    Returns:
        int: The minimum feature level for the specified pattern, or 1 if the pattern is not found.
    """

    levels = {
        "nnapi.add": 1,
        "nnapi.average_pool_2d": 1,
        "nnapi.concatenation": 1,
        "nnapi.conv2d": 1,
        "nnapi.depthwise_conv_2d": 1,
        "nnapi.depth_to_space": 1,
        "nnapi.dequantize": 1,
        "nnapi.embedding_lookup": 1,
        "nnapi.floor": 1,
        "nnapi.fully_connected": 1,
        "nnapi.hashtable_lookup": 1,
        "nnapi.l2_normalization": 1,
        "nnapi.l2_pool_2d": 1,
        "nnapi.local_response_normalization": 1,
        "nnapi.logistic": 1,
        "nnapi.lsh_projection": 1,
        "nnapi.lstm": 1,
        "nnapi.max_pool_2d": 1,
        "nnapi.mul": 1,
        "nnapi.relu": 1,
        "nnapi.relu1": 1,
        "nnapi.relu6": 1,
        "nnapi.reshape": 1,
        "nnapi.resize_bilinear": 1,
        "nnapi.rnn": 1,
        "nnapi.softmax": 1,
        "nnapi.space_to_depth": 1,
        "nnapi.svdf": 1,
        "nnapi.tanh": 1,
        "nnapi.batch_to_space_nd": 2,
        "nnapi.div": 2,
        "nnapi.mean": 2,
        "nnapi.pad": 2,
        "nnapi.space_to_batch_nd": 2,
        "nnapi.squeeze": 2,
        "nnapi.strided_slice": 2,
        "nnapi.sub": 2,
        "nnapi.transpose": 2,
        "nnapi.abs": 3,
        "nnapi.argmax": 3,
        "nnapi.argmin": 3,
        "nnapi.axis_aligned_bbox_transform": 3,
        "nnapi.bidirectional_sequence_lstm": 3,
        "nnapi.bidirectional_sequence_rnn": 3,
        "nnapi.box_with_nms_limit": 3,
        "nnapi.cast": 3,
        "nnapi.channel_shuffle": 3,
        "nnapi.detection_postprocessing": 3,
        "nnapi.equal": 3,
        "nnapi.exp": 3,
        "nnapi.expand_dims": 3,
        "nnapi.gather": 3,
        "nnapi.generate_proposals": 3,
        "nnapi.greater": 3,
        "nnapi.greater_equal": 3,
        "nnapi.grouped_conv_2d": 3,
        "nnapi.heatmap_max_keypoint": 3,
        "nnapi.instance_normalization": 3,
        "nnapi.less": 3,
        "nnapi.less_equal": 3,
        "nnapi.log": 3,
        "nnapi.logical_and": 3,
        "nnapi.logical_not": 3,
        "nnapi.logical_or": 3,
        "nnapi.log_softmax": 3,
        "nnapi.maximum": 3,
        "nnapi.minimum": 3,
        "nnapi.neg": 3,
        "nnapi.not_equal": 3,
        "nnapi.pad_v2": 3,
        "nnapi.pow": 3,
        "nnapi.prelu": 3,
        "nnapi.quantize": 3,
        "nnapi.quantized_16bit_lstm": 3,
        "nnapi.random_multinomial": 3,
        "nnapi.reduce_all": 3,
        "nnapi.reduce_any": 3,
        "nnapi.reduce_max": 3,
        "nnapi.reduce_min": 3,
        "nnapi.reduce_prod": 3,
        "nnapi.reduce_sum": 3,
        "nnapi.roi_align": 3,
        "nnapi.roi_pooling": 3,
        "nnapi.rsqrt": 3,
        "nnapi.select": 3,
        "nnapi.sin": 3,
        "nnapi.slice": 3,
        "nnapi.split": 3,
        "nnapi.sqrt": 3,
        "nnapi.tile": 3,
        "nnapi.topk_v2": 3,
        "nnapi.transpose_conv_2d": 3,
        "nnapi.unidirectional_sequence_lstm": 3,
        "nnapi.unidirectional_sequence_rnn": 3,
        "nnapi.resize_nearest_neighbor": 3,
        "nnapi.quantized_lstm": 4,
        "nnapi.if": 4,
        "nnapi.while": 4,
        "nnapi.elu": 4,
        "nnapi.hard_swish": 4,
        "nnapi.fill": 4,
        "nnapi.rank": 4,
        "nnapi.batch_matmul": 6,
        "nnapi.pack": 6,
        "nnapi.mirror_pad": 7,
        "nnapi.reverse": 7,
    }
    return levels[pattern_name]


def partition_for_nnapi(mod: IRModule, feature_level: int | None = None) -> IRModule:
    """Partition the graph greedily offloading supported operators to NNAPI.

    Parameters
    ----------
    mod : tvm_tirx.ir.IRModule
        The module to run passes on.
    feature_level : Optional[int]
        The maximum NNAPI feature level.

    Returns
    -------
    mod : tvm_tirx.ir.IRModule
        Annotated and partitioned module.
    """
    patterns = get_patterns_with_prefix("nnapi")
    if feature_level is not None:
        patterns = [pat for pat in patterns if feature_level >= min_feature_level(pat.name)]
    mod = FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=False)(mod)
    mod = MergeCompositeFunctions()(mod)
    return mod
