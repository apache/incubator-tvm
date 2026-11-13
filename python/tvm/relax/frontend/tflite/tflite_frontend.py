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
# pylint: disable=invalid-name, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel, use-list-literal
# pylint: disable=no-value-for-parameter, unused-variable
# pylint: disable=unexpected-keyword-arg, unused-import, too-many-function-args
# ruff: noqa: RUF005
"""Tensorflow lite frontend."""

import functools
import itertools
import math

import numpy as np

import tvm_tirx
from tvm import relax
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import op as _op
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import TupleType as _TupleType

from .tflite_flexbuffer import FlexBufferDecoder

__all__ = ["from_tflite"]


def to_int_list(np_array):
    """Convert a np array to a python int list.

    Note: This function converts np.int32 to python's int.
    If we don't do this conversion, numpy's automatic upcast will make
    the shape / parameters be converted to int64 IntImm in relax and
    cause problems in relax/TOPI.
    """
    return [int(x) for x in np_array]


class ExprTable:
    """Table storing Relax expressions by names."""

    def __init__(self):
        self.exprs = {}
        self.params = {}
        self.const_ctr = 1
        self.in_padding = False

    def new_const(self, value, shape=None, dtype="float32", source_name=None):
        """Construct a new var expr and add to exprs dictionary"""
        name = f"_param_{self.const_ctr}"
        self.const_ctr += 1
        self.exprs[name] = relax.const(value, dtype)
        self.params[name] = (self.exprs[name], value)

        return self.exprs[name]

    def get_expr(self, name):
        return self.exprs[name]

    def set_expr(self, name, expr, force_override=False):
        # assert isinstance(expr, _expr.Expr)
        # if name exists, we should override the value
        # otherwise, we can not get like x = func(x) work.
        # One example is CoreML preprocess, which will override
        # the same name of input.
        # However, according to git log, Find keras frontend depends
        # on this property, so we add one force_override to control it.
        if name not in self.exprs or force_override:
            self.exprs[name] = expr

    def has_expr(self, name):
        return name in self.exprs


class TensorWrapper:
    """Tensor wrapper for TFLite Tensor"""

    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params


class OperatorConverter:
    """Operator Converted for converting TFLite ops to Relax ops"""

    _SUPPORTED_QUANTIZED_OPS = frozenset(
        {
            "ABS",
            "ADD",
            "AVERAGE_POOL_2D",
            "ATAN2",
            "CEIL",
            "CONCATENATION",
            "CONV_2D",
            "COS",
            "DEPTHWISE_CONV_2D",
            "DEQUANTIZE",
            "DETECTION_POSTPROCESS",
            "DIV",
            "EQUAL",
            "EXP",
            "FLOOR",
            "FLOOR_DIV",
            "FLOOR_MOD",
            "FULLY_CONNECTED",
            "GREATER",
            "GREATER_EQUAL",
            "HARD_SWISH",
            "LEAKY_RELU",
            "LESS",
            "LESS_EQUAL",
            "LOG",
            "LOGISTIC",
            "LOG_SOFTMAX",
            "MAXIMUM",
            "MAX_POOL_2D",
            "MEAN",
            "MINIMUM",
            "MUL",
            "NEG",
            "NOT_EQUAL",
            "POW",
            "QUANTIZE",
            "REDUCE_MAX",
            "REDUCE_MIN",
            "REDUCE_PROD",
            "RELU",
            "RELU6",
            "RELU_N1_TO_1",
            "RESHAPE",
            "RESIZE_BILINEAR",
            "ROUND",
            "RSQRT",
            "SIN",
            "SOFTMAX",
            "SQRT",
            "SQUARED_DIFFERENCE",
            "SUB",
            "SUM",
            "TAN",
            "TANH",
            "TRANSPOSE_CONV",
        }
    )

    def __init__(self, model, subgraph, exp_tab, ctx, conversion_state=None):
        from tflite.ActivationFunctionType import ActivationFunctionType
        from tflite.BuiltinOperator import BuiltinOperator
        from tflite.BuiltinOptions import BuiltinOptions

        self.model = model
        self.subgraph = subgraph
        self.exp_tab = exp_tab
        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())
        self.prefetched_nodes = {}
        self.allow_custom_ops = False
        self.bb = ctx
        if conversion_state is None:
            conversion_state = {
                "lowered_subgraphs": {},
                "lowered_if_functions": {},
                "lowered_while_functions": {},
                "lowering_stack": [],
                "module_builder": ctx,
                "resource_values": {},
                "hashtable_values": {},
                "in_call_once_init": False,
            }
        else:
            conversion_state.setdefault("module_builder", ctx)
            conversion_state.setdefault("resource_values", {})
            conversion_state.setdefault("hashtable_values", {})
            conversion_state.setdefault("in_call_once_init", False)
        self.conversion_state = conversion_state
        self.resource_handles = {}
        self.hashtable_handles = {}

        # Add more operators
        self.convert_map = {
            "ABS": functools.partial(self._convert_unary_elemwise, relax_op=_op.abs),
            "ADD": functools.partial(self._convert_elemwise, relax_op=_op.add),
            "ADD_N": self.convert_add_n,
            "ARG_MAX": functools.partial(self._convert_arg_min_max, relax_op=_op.argmax),
            "ARG_MIN": functools.partial(self._convert_arg_min_max, relax_op=_op.argmin),
            "ASSIGN_VARIABLE": self.convert_assign_variable,
            "ATAN2": functools.partial(self._convert_elemwise, relax_op=_op.atan2),
            "AVERAGE_POOL_2D": functools.partial(self.convert_pool2d, pool_type="average"),
            "BATCH_TO_SPACE_ND": self.convert_batch_to_space_nd,
            "BATCH_MATMUL": self.convert_batch_matmul,
            "BIDIRECTIONAL_SEQUENCE_LSTM": self.convert_bidirectional_sequence_lstm,
            "BIDIRECTIONAL_SEQUENCE_RNN": self.convert_bidirectional_sequence_rnn,
            "BITCAST": self.convert_bitcast,
            "BITWISE_XOR": functools.partial(self._convert_elemwise, relax_op=_op.bitwise_xor),
            "BROADCAST_ARGS": self.convert_broadcast_args,
            "BROADCAST_TO": self.convert_broadcast_to,
            "BUCKETIZE": self.convert_bucketize,
            "CALL": self.convert_call,
            "CALL_ONCE": self.convert_call_once,
            "COMPLEX_ABS": self.convert_complex_abs,
            "CAST": self.convert_cast,
            "CEIL": functools.partial(self._convert_unary_elemwise, relax_op=_op.ceil),
            "CONCATENATION": self.convert_concatenation,
            "CONV_2D": functools.partial(self.convert_conv, conv_type="conv2d"),
            "CONV_3D": self.convert_conv3d,
            "CONV_3D_TRANSPOSE": self.convert_conv3d_transpose,
            "COS": functools.partial(self._convert_unary_elemwise, relax_op=_op.cos),
            "CUMSUM": self.convert_cumsum,
            "DENSIFY": self.convert_densify,
            "DEPTH_TO_SPACE": self.convert_depth_to_space,
            "DEPTHWISE_CONV_2D": functools.partial(self.convert_conv, conv_type="depthwise"),
            "DELEGATE": functools.partial(self.convert_operator_marker, op_name="DELEGATE"),
            "DEQUANTIZE": self.convert_dequantize,
            "DETECTION_POSTPROCESS": self.convert_detection_postprocess,
            "DILATE": self.convert_dilate,
            "DIV": functools.partial(self._convert_elemwise, relax_op=_op.divide),
            "ELU": self.convert_elu,
            "EMBEDDING_LOOKUP": self.convert_embedding_lookup,
            "EMBEDDING_LOOKUP_SPARSE": self.convert_embedding_lookup_sparse,
            "EQUAL": functools.partial(
                self._convert_elemwise, relax_op=_op.equal, comparison_op=True
            ),
            "EXP": functools.partial(self._convert_unary_elemwise, relax_op=_op.exp),
            "EXPAND_DIMS": self.convert_expand_dims,
            "FAKE_QUANT": self.convert_fake_quant,
            "FILL": self.convert_fill,
            "FLOOR_DIV": functools.partial(self._convert_elemwise, relax_op=_op.floor_divide),
            "FLOOR_MOD": functools.partial(self._convert_elemwise, relax_op=_op.floor_mod),
            "FLOOR": functools.partial(self._convert_unary_elemwise, relax_op=_op.floor),
            "FULLY_CONNECTED": self.convert_fully_connected,
            "GATHER": self.convert_gather,
            "GATHER_ND": self.convert_gather_nd,
            "GREATER_EQUAL": functools.partial(
                self._convert_elemwise, relax_op=_op.greater_equal, comparison_op=True
            ),
            "GREATER": functools.partial(
                self._convert_elemwise, relax_op=_op.greater, comparison_op=True
            ),
            "GELU": self.convert_gelu,
            "HARD_SWISH": self.convert_hard_swish,
            "HASHTABLE": self.convert_hashtable,
            "HASHTABLE_FIND": self.convert_hashtable_find,
            "HASHTABLE_IMPORT": self.convert_hashtable_import,
            "HASHTABLE_LOOKUP": self.convert_hashtable_lookup,
            "HASHTABLE_SIZE": self.convert_hashtable_size,
            "IF": self.convert_if,
            "IMAG": self.convert_imag,
            "L2_NORMALIZATION": self.convert_l2_normalization,
            "L2_POOL_2D": functools.partial(self.convert_pool2d, pool_type="l2"),
            "LEAKY_RELU": self.convert_leaky_relu,
            "LESS_EQUAL": functools.partial(
                self._convert_elemwise, relax_op=_op.less_equal, comparison_op=True
            ),
            "LESS": functools.partial(
                self._convert_elemwise, relax_op=_op.less, comparison_op=True
            ),
            "LOCAL_RESPONSE_NORMALIZATION": self.convert_lrn,
            "LOG": functools.partial(self._convert_unary_elemwise, relax_op=_op.log),
            "LOG_SOFTMAX": self.convert_log_softmax,
            "LOGICAL_AND": functools.partial(
                self._convert_logical_binary, relax_op=_op.logical_and
            ),
            "LOGICAL_NOT": self.convert_logical_not,
            "LOGICAL_OR": functools.partial(self._convert_logical_binary, relax_op=_op.logical_or),
            "LOGISTIC": self.convert_logistic,
            "LSTM": self.convert_lstm,
            "MATRIX_DIAG": self.convert_matrix_diag,
            "MATRIX_SET_DIAG": self.convert_matrix_set_diag,
            "MAX_POOL_2D": functools.partial(self.convert_pool2d, pool_type="max"),
            "MAXIMUM": functools.partial(self._convert_elemwise, relax_op=_op.maximum),
            "MEAN": functools.partial(self._convert_reduce, relax_op=_op.mean),
            "MINIMUM": functools.partial(self._convert_elemwise, relax_op=_op.minimum),
            "MIRROR_PAD": self.convert_mirror_pad,
            "MUL": functools.partial(self._convert_elemwise, relax_op=_op.multiply),
            "MULTINOMIAL": self.convert_multinomial,
            "NEG": functools.partial(self._convert_unary_elemwise, relax_op=_op.negative),
            "NOT_EQUAL": functools.partial(
                self._convert_elemwise, relax_op=_op.not_equal, comparison_op=True
            ),
            "ONE_HOT": self.convert_one_hot,
            "PACK": self.convert_pack,
            "PAD": self.convert_pad,
            "PADV2": self.convert_pad,
            "PLACEHOLDER_FOR_GREATER_OP_CODES": functools.partial(
                self.convert_operator_marker, op_name="PLACEHOLDER_FOR_GREATER_OP_CODES"
            ),
            "POW": functools.partial(self._convert_elemwise, relax_op=_op.power),
            "PRELU": self.convert_prelu,
            "RANGE": self.convert_range,
            "RANK": self.convert_rank,
            "QUANTIZE": self.convert_quantize,
            "RANDOM_STANDARD_NORMAL": self.convert_random_standard_normal,
            "RANDOM_UNIFORM": self.convert_random_uniform,
            "READ_VARIABLE": self.convert_read_variable,
            "REAL": self.convert_real,
            "REDUCE_ALL": functools.partial(self._convert_reduce_bool, relax_op=_op.min),
            "REDUCE_ANY": functools.partial(self._convert_reduce_bool, relax_op=_op.max),
            "REDUCE_MAX": functools.partial(self._convert_reduce, relax_op=_op.max),
            "REDUCE_MIN": functools.partial(self._convert_reduce, relax_op=_op.min),
            "REDUCE_PROD": functools.partial(self._convert_reduce, relax_op=_op.prod),
            "RELU": self.convert_relu,
            "RELU_0_TO_1": self.convert_relu_0_to_1,
            "RELU6": self.convert_relu6,
            "RELU_N1_TO_1": self.convert_relu_n1_to_1,
            "RESHAPE": self.convert_reshape,
            "RESIZE_BILINEAR": self.convert_resize_bilinear,
            "RESIZE_NEAREST_NEIGHBOR": self.convert_resize_nearest_neighbor,
            "RFFT2D": self.convert_rfft2d,
            "RIGHT_SHIFT": functools.partial(self._convert_elemwise, relax_op=_op.right_shift),
            "ROUND": functools.partial(self._convert_unary_elemwise, relax_op=_op.round),
            "RSQRT": functools.partial(self._convert_unary_elemwise, relax_op=_op.rsqrt),
            "REVERSE_SEQUENCE": self.convert_reverse_sequence,
            "REVERSE_V2": self.convert_reverse_v2,
            "SCATTER_ND": self.convert_scatter_nd,
            "SELECT": self.convert_select,
            "SELECT_V2": self.convert_select,
            "SEGMENT_SUM": functools.partial(
                self._convert_segment_op, op_name="SEGMENT_SUM", reduction="add"
            ),
            "SHAPE": self.convert_shape,
            "SIGN": functools.partial(self._convert_unary_elemwise, relax_op=_op.sign),
            "SIN": functools.partial(self._convert_unary_elemwise, relax_op=_op.sin),
            "SLICE": self.convert_slice,
            "SOFTMAX": self.convert_softmax,
            "SPACE_TO_BATCH_ND": self.convert_space_to_batch_nd,
            "SPACE_TO_DEPTH": self.convert_space_to_depth,
            "SPARSE_TO_DENSE": self.convert_sparse_to_dense,
            "SPLIT": self.convert_split,
            "SPLIT_V": self.convert_split_v,
            "SQRT": functools.partial(self._convert_unary_elemwise, relax_op=_op.sqrt),
            "SQUARE": self.convert_square,
            "SQUARED_DIFFERENCE": self.convert_squared_difference,
            "STABLEHLO_ABS": functools.partial(self._convert_stablehlo_unary, relax_op=_op.abs),
            "STABLEHLO_ADD": functools.partial(self._convert_stablehlo_binary, relax_op=_op.add),
            "STABLEHLO_AND": self._convert_stablehlo_and,
            "STABLEHLO_BROADCAST_IN_DIM": self._convert_stablehlo_broadcast_in_dim,
            "STABLEHLO_CBRT": self._convert_stablehlo_cbrt,
            "STABLEHLO_CLAMP": self._convert_stablehlo_clamp,
            "STABLEHLO_COMPARE": self._convert_stablehlo_compare,
            "STABLEHLO_COMPOSITE": self._convert_stablehlo_composite,
            "STABLEHLO_CONCATENATE": self._convert_stablehlo_concatenate,
            "STABLEHLO_CONVOLUTION": self._convert_stablehlo_convolution,
            "STABLEHLO_CONVERT": self._convert_stablehlo_convert,
            "STABLEHLO_COSINE": functools.partial(self._convert_stablehlo_unary, relax_op=_op.cos),
            "STABLEHLO_CUSTOM_CALL": self._convert_stablehlo_custom_call,
            "STABLEHLO_DIVIDE": functools.partial(
                self._convert_stablehlo_binary, relax_op=_op.divide
            ),
            "STABLEHLO_DOT_GENERAL": self._convert_stablehlo_dot_general,
            "STABLEHLO_DYNAMIC_SLICE": self._convert_stablehlo_dynamic_slice,
            "STABLEHLO_DYNAMIC_UPDATE_SLICE": self._convert_stablehlo_dynamic_update_slice,
            "STABLEHLO_EXPONENTIAL": functools.partial(
                self._convert_stablehlo_unary, relax_op=_op.exp
            ),
            "STABLEHLO_FLOOR": functools.partial(self._convert_stablehlo_unary, relax_op=_op.floor),
            "STABLEHLO_GATHER": self._convert_stablehlo_gather,
            "STABLEHLO_IOTA": self._convert_stablehlo_iota,
            "STABLEHLO_LOG": functools.partial(self._convert_stablehlo_unary, relax_op=_op.log),
            "STABLEHLO_LOGISTIC": functools.partial(
                self._convert_stablehlo_unary, relax_op=_op.sigmoid
            ),
            "STABLEHLO_MAXIMUM": functools.partial(
                self._convert_stablehlo_binary, relax_op=_op.maximum
            ),
            "STABLEHLO_MINIMUM": functools.partial(
                self._convert_stablehlo_binary, relax_op=_op.minimum
            ),
            "STABLEHLO_MULTIPLY": functools.partial(
                self._convert_stablehlo_binary, relax_op=_op.multiply
            ),
            "STABLEHLO_NEGATE": functools.partial(
                self._convert_stablehlo_unary, relax_op=_op.negative
            ),
            "STABLEHLO_OR": self._convert_stablehlo_or,
            "STABLEHLO_PAD": self._convert_stablehlo_pad,
            "STABLEHLO_POWER": functools.partial(
                self._convert_stablehlo_binary, relax_op=_op.power
            ),
            "STABLEHLO_REDUCE": self._convert_stablehlo_reduce,
            "STABLEHLO_REDUCE_WINDOW": self._convert_stablehlo_reduce_window,
            "STABLEHLO_REMAINDER": self._convert_stablehlo_remainder,
            "STABLEHLO_RESHAPE": self._convert_stablehlo_reshape,
            "STABLEHLO_RNG_BIT_GENERATOR": self._convert_stablehlo_rng_bit_generator,
            "STABLEHLO_RSQRT": functools.partial(self._convert_stablehlo_unary, relax_op=_op.rsqrt),
            "STABLEHLO_SCATTER": self._convert_stablehlo_scatter,
            "STABLEHLO_SELECT": functools.partial(
                self._convert_stablehlo_ternary, relax_op=_op.where
            ),
            "STABLEHLO_SHIFT_LEFT": functools.partial(
                self._convert_stablehlo_binary, relax_op=_op.left_shift
            ),
            "STABLEHLO_SLICE": self._convert_stablehlo_slice,
            "STABLEHLO_SORT": self._convert_stablehlo_sort,
            "STABLEHLO_SUBTRACT": functools.partial(
                self._convert_stablehlo_binary, relax_op=_op.subtract
            ),
            "STABLEHLO_TANH": functools.partial(self._convert_stablehlo_unary, relax_op=_op.tanh),
            "STABLEHLO_TRANSPOSE": self._convert_stablehlo_transpose,
            "STABLEHLO_WHILE": self._convert_stablehlo_while,
            "SQUEEZE": self.convert_squeeze,
            "STRIDED_SLICE": self.convert_strided_slice,
            "SUB": functools.partial(self._convert_elemwise, relax_op=_op.subtract),
            "SUM": functools.partial(self._convert_reduce, relax_op=_op.sum),
            "SVDF": self.convert_svdf,
            "TAN": functools.partial(self._convert_unary_elemwise, relax_op=_op.tan),
            "TANH": self.convert_tanh,
            "TILE": self.convert_tile,
            "TOPK_V2": self.convert_topk_v2,
            "TRANSPOSE_CONV": self.convert_transpose_conv,
            "TRANSPOSE": self.convert_transpose,
            "UNPACK": self.convert_unpack,
            "UNIDIRECTIONAL_SEQUENCE_RNN": self.convert_unidirectional_sequence_rnn,
            "UNIQUE": self.convert_unique,
            "UNSORTED_SEGMENT_MAX": functools.partial(
                self._convert_segment_op, op_name="UNSORTED_SEGMENT_MAX", reduction="max"
            ),
            "UNSORTED_SEGMENT_MIN": functools.partial(
                self._convert_segment_op, op_name="UNSORTED_SEGMENT_MIN", reduction="min"
            ),
            "UNSORTED_SEGMENT_PROD": functools.partial(
                self._convert_segment_op, op_name="UNSORTED_SEGMENT_PROD", reduction="mul"
            ),
            "UNSORTED_SEGMENT_SUM": functools.partial(
                self._convert_segment_op, op_name="UNSORTED_SEGMENT_SUM", reduction="add"
            ),
            "UNIDIRECTIONAL_SEQUENCE_LSTM": self.convert_unidirectional_sequence_lstm,
            "VAR_HANDLE": self.convert_var_handle,
            "WHERE": self.convert_select,
            "WHILE": self.convert_while,
            "ZEROS_LIKE": self.convert_zeros_like,
            "NON_MAX_SUPPRESSION_V4": self.convert_nms_v4,
            "NON_MAX_SUPPRESSION_V5": self.convert_nms_v5,
        }

    def check_unsupported_ops(self):
        """Check unsupported TFLite ops in our converter."""
        unsupported_ops_set = set()
        dynamic_range_ops_set = set()
        unsupported_quantized_ops_set = set()
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            if op_code_str not in self.convert_map:
                unsupported_ops_set.add(op_code_str)
                continue

            # Trying to exclude "dynamic range quantization" optimized ops as not supported in TVM
            input_tensors = self.get_input_tensors(op)
            output_tensors = self.get_output_tensors(op)
            qnn_in_cnt = len([_.qnn_params for _ in input_tensors[0:1] if _.qnn_params is not None])
            qnn_weight_cnt = len(
                [_.qnn_params for _ in input_tensors[1:] if _.qnn_params is not None]
            )
            qnn_out_cnt = len([_.qnn_params for _ in output_tensors if _.qnn_params is not None])

            if qnn_in_cnt == 0 and qnn_out_cnt == 0 and qnn_weight_cnt > 0:
                dynamic_range_ops_set.add(op_code_str)

            if (
                qnn_in_cnt + qnn_weight_cnt + qnn_out_cnt > 0
                and op_code_str not in self._SUPPORTED_QUANTIZED_OPS
            ):
                unsupported_quantized_ops_set.add(op_code_str)

        raise_msg = ""

        if unsupported_ops_set:
            ops = str(list(unsupported_ops_set)).strip("[,]")
            raise_msg += f"The following operators are not supported in frontend TFLite: {ops}\n"

        if dynamic_range_ops_set:
            ops = str(list(dynamic_range_ops_set)).strip("[,]")
            raise_msg += (
                f"The following operators are likely to have dynamic range quantization: {ops}. "
                f"If you are running an optimized graph, please turn off dynamic range "
                f"quantization or use full integer quantization\n"
            )

        if unsupported_quantized_ops_set:
            ops = ", ".join(f"'{op}'" for op in sorted(unsupported_quantized_ops_set))
            raise_msg += (
                f"The following quantized TFLite operators are not supported in frontend "
                f"TFLite yet: {ops}. Quantized operators require explicit QDQ lowering "
                f"to avoid applying Relax ops directly to quantized integer tensors.\n"
            )

        if len(raise_msg) > 0:
            raise NotImplementedError(raise_msg)

    def convert_operator_marker(self, op, op_name):
        """Reject TFLite marker builtins with an explicit diagnostic."""
        raise NotImplementedError(
            f"TFLite operator marker {op_name} is not a Relax tensor operator"
        )

    def unbind(self, data, axis=1):
        """
        This is a modified version compared to the one in common.py.
        The onnx version takes a relax.Expr.Call, the tflite
        version a TensorWrapper. Also this version by default splits
        along axis 1 and not axis 0 as the onnx version.

         Parameters
         ----------
         data : tvm.relax.frontend.tflite.TensorWrapper
             Input tensor
         axis : int
             Axis along which tensor is split.
         Returns
         -------
         result : List[relax.Expr]
             The sequence of computed tensors
        """
        shape = to_int_list(self.get_tensor_shape(data))
        if axis >= len(shape):
            msg = "Please check input dim, it shouldn't be greater than or equal to rank."
            raise AttributeError(msg)

        selections = shape[axis]
        shape.pop(axis)
        timestep = 0  # Reshape to make time step as the first dim
        shape.insert(timestep, selections)
        res_split = relax.op.split(
            relax.op.reshape(self.get_expr(data.tensor_idx), tuple(shape)), selections, timestep
        )
        ret = []
        for i in range(selections):
            ret.append(_op.squeeze(res_split[i], axis=[timestep]))
        return relax.Tuple(relax.Tuple(ret), selections)

    def _infer_shape(self, arg):
        return self.bb.normalize(arg).ty.shape

    def convert_op_to_relax(self):
        """Convert TFLite ops to relax ops"""
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            output_tensors = self.get_output_tensors(op)

            from tflite.Operator import Operator

            assert isinstance(op, Operator)
            ret = self.convert_map[op_code_str](op=op)

            # In case the Op can be prefetched, the output can be optimized out
            if ret is None:
                continue

            ret = self.bb.normalize(ret)

            if len(output_tensors) == 1:
                tensor_idx = output_tensors[0].tensor_idx
                self.exp_tab.set_expr(get_tensor_name(self.subgraph, tensor_idx), ret)
            else:
                for idx, output_tensor in enumerate(output_tensors):
                    self.exp_tab.set_expr(
                        get_tensor_name(self.subgraph, output_tensor.tensor_idx), ret[idx]
                    )

    @staticmethod
    def _decode_tflite_string(value):
        """Decode a TFLite string field."""
        if value is None:
            return ""
        if isinstance(value, bytes | bytearray):
            return value.decode("utf-8")
        return str(value)

    def _get_var_handle_resource_key(self, op, fallback_tensor=None):
        """Return a stable resource key for a VAR_HANDLE op."""
        container = ""
        shared_name = ""
        if op.BuiltinOptions() is not None:
            try:
                from tflite.VarHandleOptions import VarHandleOptions

                opts = self._get_builtin_options(op, VarHandleOptions)
                if hasattr(opts, "Container"):
                    container = self._decode_tflite_string(opts.Container())
                if hasattr(opts, "SharedName"):
                    shared_name = self._decode_tflite_string(opts.SharedName())
            except (ImportError, ModuleNotFoundError):
                pass

        if container or shared_name:
            return (container, shared_name)
        if fallback_tensor is not None:
            return ("", get_tensor_name(self.subgraph, fallback_tensor.tensor_idx))
        raise NotImplementedError("VAR_HANDLE requires VarHandleOptions")

    def _get_resource_key_for_handle(self, tensor, op_name):
        tensor_name = get_tensor_name(self.subgraph, tensor.tensor_idx)
        if tensor_name not in self.resource_handles:
            raise NotImplementedError(
                f"{op_name} requires a VAR_HANDLE in the same TFLite subgraph"
            )
        return self.resource_handles[tensor_name]

    def convert_var_handle(self, op):
        """Convert a TFLite VAR_HANDLE into an importer-local resource handle."""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 0 or len(output_tensors) != 1:
            raise NotImplementedError("VAR_HANDLE expects no inputs and one output")

        resource_key = self._get_var_handle_resource_key(op, output_tensors[0])
        resource_tensor_name = get_tensor_name(self.subgraph, output_tensors[0].tensor_idx)
        self.resource_handles[resource_tensor_name] = resource_key
        return None

    def convert_assign_variable(self, op):
        """Convert the CALL_ONCE initialization subset of ASSIGN_VARIABLE."""
        if not self.conversion_state["in_call_once_init"]:
            raise NotImplementedError(
                "ASSIGN_VARIABLE outside CALL_ONCE initialization is not supported by the "
                "Relax TFLite frontend yet because it requires mutable resource state modeling."
            )

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 2 or len(output_tensors) != 0:
            raise NotImplementedError(
                "ASSIGN_VARIABLE expects a resource handle and value input with no outputs"
            )

        resource_key = self._get_resource_key_for_handle(input_tensors[0], "ASSIGN_VARIABLE")
        self.conversion_state["resource_values"][resource_key] = self.get_tensor_expr(
            input_tensors[1]
        )
        return None

    def convert_read_variable(self, op):
        """Convert READ_VARIABLE for resources initialized by CALL_ONCE."""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 1 or len(output_tensors) != 1:
            raise NotImplementedError("READ_VARIABLE expects one input and one output")

        resource_key = self._get_resource_key_for_handle(input_tensors[0], "READ_VARIABLE")
        resource_values = self.conversion_state["resource_values"]
        if resource_key not in resource_values:
            raise NotImplementedError(
                "READ_VARIABLE requires a resource initialized by a supported CALL_ONCE subgraph"
            )
        return resource_values[resource_key]

    def _is_tflite_string_type(self, tensor_type):
        from tflite.TensorType import TensorType

        return hasattr(TensorType, "STRING") and tensor_type == TensorType.STRING

    def _is_supported_hashtable_type_pair(self, key_dtype, value_dtype):
        from tflite.TensorType import TensorType

        return (key_dtype == TensorType.INT64 and self._is_tflite_string_type(value_dtype)) or (
            self._is_tflite_string_type(key_dtype) and value_dtype == TensorType.INT64
        )

    def _get_hashtable_key(self, op, fallback_tensor=None):
        """Return a stable key and TFLite dtype pair for a HASHTABLE resource."""
        table_id = None
        key_dtype = None
        value_dtype = None
        if op.BuiltinOptions() is not None:
            try:
                from tflite.HashtableOptions import HashtableOptions

                opts = self._get_builtin_options(op, HashtableOptions)
                table_id = int(opts.TableId())
                key_dtype = int(opts.KeyDtype())
                value_dtype = int(opts.ValueDtype())
            except (ImportError, ModuleNotFoundError):
                pass

        if key_dtype is None or value_dtype is None:
            raise NotImplementedError("HASHTABLE requires HashtableOptions")
        if not self._is_supported_hashtable_type_pair(key_dtype, value_dtype):
            raise NotImplementedError(
                "TFLite HASHTABLE only supports int64/string or string/int64 tables"
            )

        if table_id is not None:
            return table_id, key_dtype, value_dtype
        if fallback_tensor is not None:
            return (
                get_tensor_name(self.subgraph, fallback_tensor.tensor_idx),
                key_dtype,
                value_dtype,
            )
        raise NotImplementedError("HASHTABLE requires HashtableOptions")

    def _get_hashtable_info_for_handle(self, tensor, op_name):
        tensor_name = get_tensor_name(self.subgraph, tensor.tensor_idx)
        if tensor_name not in self.hashtable_handles:
            raise NotImplementedError(f"{op_name} requires a HASHTABLE in the same TFLite subgraph")
        return self.hashtable_handles[tensor_name]

    @staticmethod
    def _get_tensor_shape_tuple(tensor_wrapper):
        if tensor_wrapper.tensor.ShapeLength() == 0:
            return ()
        return tuple(int(dim) for dim in tensor_wrapper.tensor.ShapeAsNumpy())

    @staticmethod
    def _has_tensor_buffer_data(tensor_wrapper):
        return (
            tensor_wrapper.buffer is not None
            and hasattr(tensor_wrapper.buffer, "DataLength")
            and tensor_wrapper.buffer.DataLength() > 0
        )

    def _get_string_tensor_value(self, tensor_wrapper, op_name):
        """Decode a constant TFLite string tensor buffer."""
        if not self._is_tflite_string_type(tensor_wrapper.tensor.Type()):
            raise NotImplementedError(f"{op_name} requires a TensorType.STRING tensor")
        if not self._has_tensor_buffer_data(tensor_wrapper):
            raise NotImplementedError(f"{op_name} requires a constant string tensor")

        data = bytes(tensor_wrapper.buffer.DataAsNumpy())
        if len(data) < 4:
            raise NotImplementedError(f"{op_name} has an invalid string tensor buffer")

        count = int(np.frombuffer(data, dtype="<i4", count=1)[0])
        if count < 0:
            raise NotImplementedError(f"{op_name} has an invalid string tensor count")

        header_size = 4 * (count + 2)
        if len(data) < header_size:
            raise NotImplementedError(f"{op_name} has an invalid string tensor offsets")

        offsets = np.frombuffer(data, dtype="<i4", count=count + 1, offset=4).astype(np.int64)
        if np.any(offsets < header_size) or np.any(offsets > len(data)):
            raise NotImplementedError(f"{op_name} has out-of-bounds string tensor offsets")
        if np.any(offsets[:-1] > offsets[1:]):
            raise NotImplementedError(f"{op_name} has non-monotonic string tensor offsets")

        try:
            values = [
                data[int(offsets[i]) : int(offsets[i + 1])].decode("utf-8") for i in range(count)
            ]
        except UnicodeDecodeError as e:
            raise NotImplementedError(f"{op_name} has invalid UTF-8 string data: {e}") from e
        shape = self._get_tensor_shape_tuple(tensor_wrapper)
        expected_count = math.prod(shape) if shape else 1
        if expected_count != count:
            raise NotImplementedError(
                f"{op_name} string tensor buffer count does not match its shape"
            )
        return np.array(values, dtype=object).reshape(shape)

    def convert_hashtable(self, op):
        """Convert a TFLite HASHTABLE into an importer-local table handle."""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 0 or len(output_tensors) != 1:
            raise NotImplementedError("HASHTABLE expects no inputs and one output")

        table_key, key_dtype, value_dtype = self._get_hashtable_key(op, output_tensors[0])
        table_tensor_name = get_tensor_name(self.subgraph, output_tensors[0].tensor_idx)
        self.hashtable_handles[table_tensor_name] = {
            "table_key": table_key,
            "key_dtype": key_dtype,
            "value_dtype": value_dtype,
        }
        return None

    def convert_hashtable_import(self, op):
        """Convert static metadata for the CALL_ONCE HASHTABLE_IMPORT subset."""
        if not self.conversion_state["in_call_once_init"]:
            raise NotImplementedError(
                "HASHTABLE_IMPORT outside CALL_ONCE initialization is not supported by the "
                "Relax TFLite frontend yet because it requires mutable resource state modeling."
            )

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 3 or len(output_tensors) != 0:
            raise NotImplementedError(
                "HASHTABLE_IMPORT expects table, keys, and values inputs with no outputs"
            )

        table_info = self._get_hashtable_info_for_handle(input_tensors[0], "HASHTABLE_IMPORT")
        key_tensor = input_tensors[1]
        value_tensor = input_tensors[2]
        if (
            key_tensor.tensor.Type() != table_info["key_dtype"]
            or value_tensor.tensor.Type() != table_info["value_dtype"]
        ):
            raise NotImplementedError("HASHTABLE_IMPORT key/value dtypes mismatch")
        key_shape = self._get_tensor_shape_tuple(key_tensor)
        value_shape = self._get_tensor_shape_tuple(value_tensor)
        if key_shape != value_shape:
            raise NotImplementedError("HASHTABLE_IMPORT requires keys and values same shape")
        if not self._has_tensor_buffer_data(key_tensor) or not self._has_tensor_buffer_data(
            value_tensor
        ):
            raise NotImplementedError("HASHTABLE_IMPORT requires constant keys and values")

        if self._is_tflite_string_type(table_info["key_dtype"]):
            keys = self._get_string_tensor_value(key_tensor, "HASHTABLE_IMPORT")
        else:
            keys = self.get_tensor_value(key_tensor)
        if self._is_tflite_string_type(table_info["value_dtype"]):
            values = self._get_string_tensor_value(value_tensor, "HASHTABLE_IMPORT")
        else:
            values = self.get_tensor_value(value_tensor)

        if np.unique(keys).size != keys.size:
            raise NotImplementedError("HASHTABLE_IMPORT with duplicate keys is not supported")

        hashtable_values = self.conversion_state["hashtable_values"]
        table_key = table_info["table_key"]
        if table_key not in hashtable_values:
            hashtable_values[table_key] = {
                "size": math.prod(key_shape) if key_shape else 1,
                "key_dtype": table_info["key_dtype"],
                "value_dtype": table_info["value_dtype"],
                "keys": keys,
                "values": values,
            }
        return None

    def convert_hashtable_find(self, op):
        """Convert the constant-foldable string-to-int64 HASHTABLE_FIND subset."""
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 3 or len(output_tensors) != 1:
            raise NotImplementedError(
                "HASHTABLE_FIND expects table, query, and default inputs with one output"
            )

        table_tensor, query_tensor, default_tensor = input_tensors
        output_tensor = output_tensors[0]
        table_info = self._get_hashtable_info_for_handle(table_tensor, "HASHTABLE_FIND")
        table_key = table_info["table_key"]
        hashtable_values = self.conversion_state["hashtable_values"]
        if table_key not in hashtable_values:
            raise NotImplementedError(
                "HASHTABLE_FIND requires a table initialized by a supported CALL_ONCE subgraph"
            )
        table_values = hashtable_values[table_key]

        if (
            query_tensor.tensor.Type() != table_values["key_dtype"]
            or default_tensor.tensor.Type() != table_values["value_dtype"]
            or output_tensor.tensor.Type() != table_values["value_dtype"]
        ):
            raise NotImplementedError("HASHTABLE_FIND key/value dtypes mismatch")

        if not (
            self._is_tflite_string_type(table_values["key_dtype"])
            and table_values["value_dtype"] == TensorType.INT64
        ):
            raise NotImplementedError(
                "HASHTABLE_FIND only supports constant string -> int64 tables"
            )
        if not self._has_tensor_buffer_data(query_tensor):
            raise NotImplementedError("HASHTABLE_FIND with runtime string queries is not supported")
        if not self._has_tensor_buffer_data(default_tensor):
            raise NotImplementedError("HASHTABLE_FIND requires constant default values")

        query_shape = self._get_tensor_shape_tuple(query_tensor)
        output_shape = self._get_tensor_shape_tuple(output_tensor)
        if output_shape != query_shape:
            raise NotImplementedError("HASHTABLE_FIND output shape must match query shape")

        query_values = self._get_string_tensor_value(query_tensor, "HASHTABLE_FIND")
        default_values = self.get_tensor_value(default_tensor)
        default_shape = self._get_tensor_shape_tuple(default_tensor)
        if default_shape == () or default_values.size == 1:
            result = np.full(output_shape, int(default_values.item()), dtype=np.int64)
        elif default_shape == query_shape:
            result = default_values.astype(np.int64).copy()
        else:
            raise NotImplementedError(
                "HASHTABLE_FIND default value must be scalar or match query shape"
            )

        table_map = dict(
            zip(
                table_values["keys"].reshape(-1).tolist(),
                table_values["values"].reshape(-1).astype(np.int64).tolist(),
            )
        )
        for index, key in np.ndenumerate(query_values):
            if key in table_map:
                result[index] = table_map[key]
        return relax.const(result.astype(np.int64), "int64")

    def convert_hashtable_lookup(self, op):
        """Convert TFLite HASHTABLE_LOOKUP for non-string value tensors."""
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 3 or len(output_tensors) != 2:
            raise NotImplementedError(
                "HASHTABLE_LOOKUP expects lookup, key, and value inputs with two outputs"
            )

        lookup_tensor, key_tensor, value_tensor = input_tensors
        output_tensor, hits_tensor = output_tensors

        if (
            lookup_tensor.tensor.Type() != TensorType.INT32
            or key_tensor.tensor.Type() != TensorType.INT32
        ):
            raise NotImplementedError("HASHTABLE_LOOKUP requires int32 lookup and key tensors")
        if self._is_tflite_string_type(value_tensor.tensor.Type()):
            raise NotImplementedError(
                "HASHTABLE_LOOKUP with TensorType.STRING values is not supported"
            )
        if value_tensor.tensor.Type() != output_tensor.tensor.Type():
            raise NotImplementedError(
                "HASHTABLE_LOOKUP output dtype must match the value tensor dtype"
            )
        if hits_tensor.tensor.Type() != TensorType.UINT8:
            raise NotImplementedError("HASHTABLE_LOOKUP hits output must be uint8")

        lookup_shape = to_int_list(self.get_tensor_shape(lookup_tensor))
        key_shape = to_int_list(self.get_tensor_shape(key_tensor))
        value_shape = to_int_list(self.get_tensor_shape(value_tensor))
        output_shape = to_int_list(self.get_tensor_shape(output_tensor))
        hits_shape = to_int_list(self.get_tensor_shape(hits_tensor))

        if len(lookup_shape) != 1 or len(key_shape) != 1 or len(value_shape) < 1:
            raise NotImplementedError(
                "HASHTABLE_LOOKUP requires rank-1 lookup/key and rank>=1 value tensors"
            )
        if key_shape[0] != value_shape[0]:
            raise NotImplementedError(
                "HASHTABLE_LOOKUP requires key and value tensors to agree on row count"
            )
        if key_shape[0] == 0:
            raise NotImplementedError("HASHTABLE_LOOKUP requires a non-empty key/value table")
        if output_shape != [lookup_shape[0]] + value_shape[1:]:
            raise NotImplementedError(
                "HASHTABLE_LOOKUP output shape must match lookup count and value tail shape"
            )
        if hits_shape != [lookup_shape[0]]:
            raise NotImplementedError("HASHTABLE_LOOKUP hits output shape must match lookup count")

        lookup = self.get_tensor_expr(lookup_tensor)
        key = self.get_tensor_expr(key_tensor)
        value = self.get_tensor_expr(value_tensor)

        positions = relax.op.bucketize(lookup, key, out_int32=True, right=False)
        candidate_keys = relax.op.take(key, positions, axis=0, mode="clip")
        in_range = relax.op.less(positions, relax.const(key_shape[0], "int32"))
        found = relax.op.logical_and(in_range, relax.op.equal(candidate_keys, lookup))

        gathered_values = relax.op.take(value, positions, axis=0, mode="clip")
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())
        zero_values = relax.op.zeros(output_shape, output_dtype)

        if len(value_shape) > 1:
            found_values = relax.op.expand_dims(found, axis=list(range(1, len(value_shape))))
            found_values = relax.op.broadcast_to(found_values, output_shape)
        else:
            found_values = found

        output = relax.op.where(found_values, gathered_values, zero_values)
        hits = relax.op.astype(found, "uint8")
        return relax.Tuple([output, hits])

    def convert_hashtable_size(self, op):
        """Convert HASHTABLE_SIZE for a statically imported TFLite hashtable."""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 1 or len(output_tensors) != 1:
            raise NotImplementedError("HASHTABLE_SIZE expects one input and one output")

        from tflite.TensorType import TensorType

        if output_tensors[0].tensor.Type() != TensorType.INT64:
            raise NotImplementedError("HASHTABLE_SIZE output must be int64")
        table_info = self._get_hashtable_info_for_handle(input_tensors[0], "HASHTABLE_SIZE")
        table_key = table_info["table_key"]
        hashtable_values = self.conversion_state["hashtable_values"]
        if table_key not in hashtable_values:
            raise NotImplementedError(
                "HASHTABLE_SIZE requires a table initialized by a supported CALL_ONCE subgraph"
            )
        return relax.const(np.array([hashtable_values[table_key]["size"]], dtype=np.int64), "int64")

    def get_op_code_str(self, op):
        """Get TFLite ops string representation"""

        from tflite.BuiltinOperator import BuiltinOperator

        op_code_list_idx = op.OpcodeIndex()

        op_c = self.model.OperatorCodes(op_code_list_idx)
        # In TFlite 2.4.x there was a change where the type of the field that contained
        # the builtin code changed from int8 to int32 in the flat buffer representation.
        # However, to retain support for old flat buffers that were created, they retained
        # the original 8 bit field, but named it "deprecated_builtin_code" in TFLite 2.4.
        # This means that the API function BuiltinCode() which originally returned the value
        # of the 8 bit field would now look for the value in the new int32 field in the
        # schema and DeprecatedBuiltinCode() will look at the old 8 bit field.
        # In TFLite 2.4, if the opcode value is less than 127, it can be in either field
        # (however, if it is only in the "builtin_code" field, the model is not backward
        # compatible), so similarly to TFLite 2.4 reader, we'll pick the higher value of the
        # two fields.
        # Remember however that this value came into existence only after Tensorflow
        # lite 2.4.x and hence encase it in a try -except block.
        # Phew !
        try:
            opc = max(op_c.DeprecatedBuiltinCode(), op_c.BuiltinCode())
        except AttributeError:
            # In versions before 2.4 the int8 field that holds the builtin code is accessed
            # by BuiltinCode() and DeprecatedBuiltinCode() doesn't exist
            opc = op_c.BuiltinCode()

        op_code_id = opc
        try:
            op_code_str = self.builtin_op_code[op_code_id]
        except KeyError:
            raise NotImplementedError(
                "TFLite operator with code "
                + str(op_code_id)
                + " is not supported by this version of the fbs schema."
            )
        if op_code_id == BuiltinOperator.CUSTOM:
            # Custom operator
            custom_op_code_str = self.model.OperatorCodes(op_code_list_idx).CustomCode()

            if self.allow_custom_ops:
                return "CUSTOM"

            if custom_op_code_str == b"TFLite_Detection_PostProcess":
                return "DETECTION_POSTPROCESS"

            raise NotImplementedError("Custom operators are currently not supported")
        return op_code_str

    def get_input_tensors(self, op):
        operator_inputs = op.InputsAsNumpy()
        return self.get_tensors(operator_inputs)

    def get_output_tensors(self, op):
        operator_outputs = op.OutputsAsNumpy()
        return self.get_tensors(operator_outputs)

    def get_tensors(self, tensors_idx_list):
        """Get tensor wrapper list from given TFLite tensor index list"""
        return_list = list()
        for tensor_idx in self._indices_or_empty(tensors_idx_list):
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, 0, 0))
                continue

            tensor = self.subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                # TFLite supports both per-tensor and per-axis (aka channel) quantization.  For
                # per-tensor quantization, scale and zero points are scalar values.  For per-axis
                # quantization, scale and zero points for the weights are tensors (activations are
                # per-tensor quantized). However, the TFLite quantization spec puts restrictions on
                # zero points for per-axis quantization.  Specifically, the zero point is a tensor
                # but all values are 0. More information can be found here -
                # https://www.tensorflow.org/lite/performance/quantization_spec

                tflite_scale = tflite_qnn_params.ScaleAsNumpy()
                tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
                is_qnn_params_valid = True

                # Handle Per-axis and per-tensor cases
                if isinstance(tflite_scale, np.ndarray):
                    assert isinstance(tflite_zero_point, np.ndarray)

                    # Tensor - Per-axis quantization
                    if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                        scale = tflite_scale
                        # Ensure that all zero points are zeros
                        zero_point = tflite_zero_point
                        if not np.all(zero_point == 0):
                            raise AttributeError(
                                "TFLite per-axis quantization restricts all zero points to be"
                                + " 0, but a non-zero value is observed"
                            )
                        zero_point = int(zero_point[0])

                    # Scalar - Per-tensor quantization
                    elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                        scale = float(tflite_scale[0])
                        zero_point = int(tflite_zero_point[0])

                    else:
                        raise NotImplementedError(
                            f"Quantized type {type(tflite_scale)} (scale) and  "
                            f"{type(tflite_zero_point)} (zero point) not supported"
                        )
                elif tflite_scale == 0 and tflite_zero_point == 0:
                    # Handle corner case for ops like quantized reshape whose second operand (shape)
                    # has zero scale and zero zero point. This is not used.
                    is_qnn_params_valid = False
                else:
                    raise NotImplementedError(f"Quantized type {type(tflite_scale)} not supported")

                # Check that the scale and zero points are valid.
                if is_qnn_params_valid:
                    qnn_params = dict()
                    qnn_params["scale"] = relax.const(scale, "float32")
                    qnn_params["zero_point"] = relax.const(zero_point, "int32")
                    qnn_params["axis"] = int(tflite_qnn_params.QuantizedDimension())
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
        return return_list

    def get_tensor_type_as_numpy(self, tensor_wrapper):
        """Returns np.dtype out of TensorType"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        from tflite.TensorType import TensorType

        return {
            TensorType.UINT8: np.uint8,
            TensorType.INT8: np.int8,
            TensorType.INT16: np.int16,
            TensorType.FLOAT16: np.float16,
            TensorType.FLOAT32: np.float32,
            TensorType.INT32: np.int32,
            TensorType.INT64: np.int64,
            TensorType.UINT32: np.uint32,
            TensorType.UINT64: np.uint64,
            TensorType.BOOL: np.bool_,
            TensorType.COMPLEX64: np.complex64,
        }[tensor_wrapper.tensor.Type()]

    # pylint: disable=no-else-return
    def get_tensor_value(self, tensor_wrapper, is_sparse=False):
        """Get tensor buffer value from given tensor wrapper.

        Returns the raw TFLite buffer reinterpreted via ``get_tensor_type_as_numpy``: for
        COMPLEX64 this is a ``np.complex64`` ndarray with the TFLite shape (no pair axis).
        This is distinct from ``get_tensor_expr``, which returns the *Relax* representation
        (float32 real/imag pair with a trailing (2,) axis for COMPLEX64). Use
        ``get_tensor_expr`` for Relax IR construction; use ``get_tensor_value`` only when
        you need the literal TFLite buffer (e.g. fft_length constant parsing).
        """
        assert isinstance(tensor_wrapper, TensorWrapper)

        dtype = self.get_tensor_type_as_numpy(tensor_wrapper)
        data = tensor_wrapper.buffer.DataAsNumpy()

        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = to_int_list(self.get_tensor_shape(tensor_wrapper))
        else:
            shape = []

        if is_sparse:
            return np.frombuffer(data, dtype=dtype)
        else:
            return np.frombuffer(data, dtype=dtype).reshape(shape)

    def get_tensor_type_str(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""

        from tflite.TensorType import TensorType

        if tensor_type == TensorType.INT8:
            return "int8"
        if tensor_type == TensorType.INT16:
            return "int16"
        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT16:
            return "float16"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        if tensor_type == TensorType.UINT32:
            return "uint32"
        if tensor_type == TensorType.UINT64:
            return "uint64"
        if tensor_type == TensorType.BOOL:
            return "bool"
        if tensor_type == TensorType.COMPLEX64:
            return "complex64"
        raise NotImplementedError(f"Tensor type {tensor_type!s} is currently not supported")

    def _is_tflite_complex64_type(self, tensor_type):
        """Return whether a TFLite tensor type is COMPLEX64."""
        from tflite.TensorType import TensorType

        return tensor_type == TensorType.COMPLEX64

    def _unwrap_tflite_tensor(self, tensor):
        """Return the underlying tflite.Tensor, unwrapping TensorWrapper if needed."""
        if isinstance(tensor, TensorWrapper):
            return tensor.tensor
        return tensor

    def _get_relax_tensor_dtype(self, tensor):
        """Return the Relax dtype used to represent a TFLite tensor."""
        tensor = self._unwrap_tflite_tensor(tensor)
        tensor_type = tensor.Type() if hasattr(tensor, "Type") else tensor
        if self._is_tflite_complex64_type(tensor_type):
            return "float32"
        return self.get_tensor_type_str(tensor_type)

    def _get_relax_tensor_shape(self, tensor):
        """Return the Relax shape used to represent a TFLite tensor.

        For COMPLEX64 tensors, the trailing (2,) axis encodes the real/imag pair.
        Returns an empty tuple () for rank-0 tensors. Shape elements are Python ints
        (not numpy scalars) so the result is safe to feed into TIRX ``T.Tensor(shape, ...)``.
        """
        tensor = self._unwrap_tflite_tensor(tensor)
        shape = to_int_list(tensor.ShapeAsNumpy()) if tensor.ShapeLength() > 0 else ()
        if self._is_tflite_complex64_type(tensor.Type()):
            return tuple(shape) + (2,)
        return tuple(shape)

    def _get_shape_expr_from_tensor(self, shape_tensor, prefix):
        """Convert a TFLite shape tensor to a Relax shape expression."""
        if self.has_expr(shape_tensor.tensor_idx):
            dims_expr = self.get_expr(shape_tensor.tensor_idx)
            dims_ndim = int(self.get_tensor_shape(shape_tensor)[0])
            dims_dtype = self.get_tensor_type_str(shape_tensor.tensor.Type())
            dims_expr = self.bb.match_cast(dims_expr, relax.TensorType([dims_ndim], dims_dtype))
            dims_expr = self.bb.normalize(relax.op.astype(dims_expr, "int64"))
            shape_dataflow_var = self.bb.emit(relax.op.tensor_to_shape(dims_expr))
            shape_vars = [tirx.Var(f"{prefix}_{i}", "int64") for i in range(dims_ndim)]
            self.bb.match_cast(shape_dataflow_var, relax.ShapeType(shape_vars))
            return relax.ShapeExpr(shape_vars), shape_vars

        dims = to_int_list(self.get_tensor_value(shape_tensor))
        return dims, dims

    def flatten_to_nd(self, x, nd=3):
        """Flatten input tensor to nd rank"""
        shape = x.ty.shape
        ndims = len(shape)
        if ndims == nd:
            return x
        new_shape = [-1] + [int(shape[i]) for i in range(ndims - nd + 1, ndims)]
        return relax.op.reshape(x, new_shape)

    def has_same_qnn_params(self, lhs_tensor, rhs_tensor):
        lhs_scale = lhs_tensor.qnn_params["scale"]
        rhs_scale = rhs_tensor.qnn_params["scale"]
        lhs_zero_point = lhs_tensor.qnn_params["zero_point"]
        rhs_zero_point = rhs_tensor.qnn_params["zero_point"]
        # 0.1 + 0.2 != 0.3
        return np.allclose(
            lhs_scale.value.numpy(), rhs_scale.value.numpy(), rtol=1e-5, atol=1e-5
        ) and np.allclose(
            lhs_zero_point.value.numpy(), rhs_zero_point.value.numpy(), rtol=1e-5, atol=1e-5
        )

    def quantize(self, expr, tensor_to_quantize):
        """Helper function to quantize a tensor with Relax"""
        tensor_type = tensor_to_quantize.tensor.Type()
        tensor_type_str = self.get_tensor_type_str(tensor_type)
        quantized = relax.op.quantize(
            data=expr,
            scale=tensor_to_quantize.qnn_params["scale"],
            zero_point=tensor_to_quantize.qnn_params["zero_point"],
            axis=tensor_to_quantize.qnn_params["axis"],
            out_dtype=tensor_type_str,
        )
        return quantized

    def dequantize(self, expr, tensor):
        """Helper function to dequantize a tensor with Relax"""
        dequantized = relax.op.dequantize(
            data=expr,
            scale=tensor.qnn_params["scale"],
            zero_point=tensor.qnn_params["zero_point"],
            axis=tensor.qnn_params["axis"],
        )
        return dequantized

    def is_quantized(self, op):
        """Check if an input tensor is quantized."""
        input_tensors = self.get_input_tensors(op)
        first_tensor = input_tensors[0]
        return first_tensor.qnn_params is not None

    def convert_qnn_fused_activation_function(
        self, expr, fused_activation_fn, scale, zero_point, dtype
    ):
        """Convert TFLite fused activation function. The expr is an input quantized tensor with
        scale and zero point"""

        from tflite.ActivationFunctionType import ActivationFunctionType

        # Quantize a float value to an quantized integer value
        def quantize(x):
            return float(round(x / scale) + zero_point)

        # Get min/max of the output dtype. This will be used to ensure that clip a_min/a_max are not
        # beyond the dtype range.
        qmin = float(tvm_tirx.tirx.min_value(ty=dtype).value)
        qmax = float(tvm_tirx.tirx.max_value(ty=dtype).value)

        # The input expr is a quantized tensor with its scale and zero point. We calculate the
        # suitable clip off points based on these scale and zero point.
        if fused_activation_fn == ActivationFunctionType.NONE:
            return expr
        if fused_activation_fn == ActivationFunctionType.RELU6:
            return relax.op.clip(expr, min=max(qmin, quantize(0)), max=min(qmax, quantize(6.0)))
        if fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
            return relax.op.clip(expr, min=max(qmin, quantize(-1.0)), max=min(qmax, quantize(1.0)))
        if fused_activation_fn == ActivationFunctionType.RELU:
            return relax.op.clip(expr, min=max(qmin, quantize(0.0)), max=qmax)

        fused_activation_fn_str = self.activation_fn_type[fused_activation_fn]
        raise NotImplementedError(
            f"Quantized activation {fused_activation_fn_str} is not supported yet."
        )

    def convert_reshape(self, op):
        """Convert TFLite reshape"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.ReshapeOptions import ReshapeOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (1, 2), "input tensors should not be empty"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "There should be only 1 output tensor"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        if len(input_tensors) == 2:
            shape_tensor = input_tensors[1]
            if self.has_expr(shape_tensor.tensor_idx):
                target_expr = self.get_expr(shape_tensor.tensor_idx)
                """
                target_value, success = try_infer_value(
                    target_expr,
                    parameters={
                        k: tvm_tirx.runtime.tensor(np.array(v)) for k, v in self.exp_tab.params.items()
                    },
                )
                """
                target_value, success = target_expr, False
                if success:
                    # convert to flattened list
                    from itertools import chain

                    try:
                        target_shape = list(chain(*target_value))
                    except TypeError:
                        target_shape = list(chain(target_value))
                else:
                    target_shape = target_expr
            else:
                target_shape = self.get_tensor_value(shape_tensor)
                # convert to flattened list
                from itertools import chain

                try:
                    target_shape = list(chain(*target_shape))
                except TypeError:
                    target_shape = list(chain(target_shape))

        else:
            assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
            op_options = op.BuiltinOptions()
            reshape_options = ReshapeOptions()
            reshape_options.Init(op_options.Bytes, op_options.Pos)
            target_shape = to_int_list(reshape_options.NewShapeAsNumpy())

        in_expr = self.get_expr(input_tensor_idx)

        # If the tensors are quantized, ensure that input/output qnn params are same.

        input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
        if input_tensor.qnn_params and input_tensor_type_str == "int8":
            # TFLite 2.x quantization spec requires qnn params to be same and dtype to be int8.
            # For TFLite 1.x, dtype can be uint8 and qnn params can be different
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(input_tensor, output_tensor), (
                "TFLite reshape requires input and output scale and zero points to be equal"
            )

        if input_tensor.qnn_params and input_tensor_type_str == "uint8":
            output_tensor = output_tensors[0]
            if not self.has_same_qnn_params(input_tensor, output_tensor):
                in_f32 = self.dequantize(in_expr, input_tensor)
                out = relax.op.reshape(in_f32, shape=relax.ShapeExpr(target_shape))
                out = self.quantize(out, output_tensor)
                return out

        out = relax.op.reshape(in_expr, shape=relax.ShapeExpr(target_shape))
        return out

    def _convert_resize(self, method, op):
        """Generic method to Convert TFLite RESIZE operators"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.ResizeBilinearOptions import ResizeBilinearOptions

        # ResizeNearestNeighborOptions was added in tflite v1.13
        tflite_ver = 1120
        if hasattr(BuiltinOptions, "ResizeNearestNeighborOptions"):
            tflite_ver = 1130

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # images, 4-D Tensor with shape NHWC.
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # size - 1-D int32 Tensor of 2 elements: new_height, new_width
        target_size = tuple(self.get_tensor_value(input_tensors[1]))

        # Options - align_corners (bool)
        resize_options = None
        align_corners = False
        bilinear_method = method == "linear"
        if bilinear_method:
            assert op.BuiltinOptionsType() == BuiltinOptions.ResizeBilinearOptions
            resize_options = ResizeBilinearOptions()
        elif tflite_ver >= 1130:
            assert op.BuiltinOptionsType() == BuiltinOptions.ResizeNearestNeighborOptions
            from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions

            resize_options = ResizeNearestNeighborOptions()

        if resize_options is not None:
            op_options = op.BuiltinOptions()
            resize_options.Init(op_options.Bytes, op_options.Pos)
            align_corners = resize_options.AlignCorners()
            half_pixel_centers = resize_options.HalfPixelCenters()

        # Use layout NHWC
        coord_trans = "align_corners" if align_corners else "asymmetric"
        coord_trans = "half_pixel" if half_pixel_centers else coord_trans

        rounding_method = ""
        if method == "nearest_neighbor":
            if not align_corners and half_pixel_centers:
                rounding_method = "round_prefer_ceil"

        if bilinear_method and input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.image.resize2d(
            in_expr, target_size, None, "NHWC", method, coord_trans, rounding_method
        )
        if bilinear_method and output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_resize_bilinear(self, op):
        """Convert TFLite RESIZE_BILINEAR"""
        return self._convert_resize("linear", op)

    def convert_resize_nearest_neighbor(self, op):
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR"""
        return self._convert_resize("nearest_neighbor", op)

    def convert_l2_normalization(self, op):
        """Convert TFLite L2_NORMALIZATION"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.L2NormOptions import L2NormOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.L2NormOptions
        op_options = op.BuiltinOptions()
        l2_norm_options = L2NormOptions()
        l2_norm_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = l2_norm_options.FusedActivationFunction()

        # TFLite supports normalization only over the last dim
        input_tensor_rank = len(input_tensor.tensor.ShapeAsNumpy())

        if self.is_quantized(op):
            raise NotImplementedError(
                "TFLite quantized L2_NORMALIZATION operator is not supported yet."
            )

        # TFL uses only the default epsilon value
        # Implement L2 normalization: output = input / sqrt(sum(input^2) + eps)
        # L2 normalization is applied along the last axis
        squared = relax.op.square(in_expr)
        sum_squared = relax.op.sum(squared, axis=input_tensor_rank - 1, keepdims=True)
        denom = relax.op.sqrt(relax.op.add(sum_squared, relax.const(1e-12, "float32")))
        out = relax.op.divide(in_expr, denom)

        # if we have fused activation fn
        if output_tensor.qnn_params:
            raise NotImplementedError(
                "TFLite quantized L2_NORMALIZATION operator is not supported yet."
            )
        out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_lrn(self, op):
        """Convert TFLite LOCAL_RESPONSE_NORMALIZATION"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.LocalResponseNormalizationOptions import LocalResponseNormalizationOptions

        if self.is_quantized(op):
            raise NotImplementedError("TFlite quantized LRN operator is not supported yet.")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.LocalResponseNormalizationOptions
        op_options = op.BuiltinOptions()
        lrn_options = LocalResponseNormalizationOptions()
        lrn_options.Init(op_options.Bytes, op_options.Pos)
        radius = lrn_options.Radius()
        bias = lrn_options.Bias()
        alpha = lrn_options.Alpha()
        beta = lrn_options.Beta()
        size = (radius * 2) + 1
        alpha = alpha * size
        axis = 3  # NHWC format
        data_shape = to_int_list(self.get_tensor_shape(input_tensor))
        in_type = self.get_tensor_type_str(input_tensor.tensor.Type())

        # Relax currently does not expose a dedicated LRN op. Implement NHWC channel LRN
        # by pooling squared values over the channel axis.
        squared = self.bb.normalize(relax.op.square(in_expr))
        squared_2d = _op.reshape(squared, [-1, data_shape[axis], 1, 1])
        pooled = self.bb.normalize(
            relax.op.nn.avg_pool2d(
                squared_2d,
                pool_size=[size, 1],
                strides=[1, 1],
                padding=[radius, 0, radius, 0],
                layout="NHWC",
                count_include_pad=True,
            )
        )
        pooled = self.bb.normalize(_op.reshape(pooled, data_shape))
        denom = relax.op.power(
            relax.op.add(
                relax.const(bias, in_type), relax.op.multiply(relax.const(alpha, in_type), pooled)
            ),
            relax.const(beta, in_type),
        )
        out = relax.op.divide(in_expr, denom)

        return out

    def convert_logistic(self, op):
        """Convert TFLite LOGISTIC"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.sigmoid(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_softmax(self, op):
        """Convert TFLite softmax"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        params = {"axis": -1}  # -1 is channel
        in_expr = self.get_expr(input_tensor_idx)

        # TODO - Naive softmax int8 implementation leads to bad accuracy. Currently, we can
        # dequantize to FP32 and perform softmax on FP32. We can investigate an integer only softmax
        # implementation in future.
        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        out = relax.op.nn.softmax(in_expr, **params)

        # Go back to integer dataype if the original operator was quantized.
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_tanh(self, op):
        """Convert TFLite TANH"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.tanh(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_range(self, op):
        """Convert TFLite Range.

        Constant bounds lower directly to ``relax.op.arange``. Runtime (dynamic)
        scalar bounds are handled by computing the element count in-graph,
        lifting it to a symbolic output dimension, and rebuilding the values as
        ``arange(0, count) * delta + start`` (see ``_convert_dynamic_range``).
        """

        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        start, limit, delta = input_tensors[0], input_tensors[1], input_tensors[2]

        # out type inference
        if delta.tensor.Type() == TensorType.FLOAT32:
            out_type = self.get_tensor_type_str(delta.tensor.Type())
        else:
            out_type = self.get_tensor_type_str(start.tensor.Type())

        def is_dynamic(tensor):
            return self.has_expr(tensor.tensor_idx) and not isinstance(
                self.get_expr(tensor.tensor_idx), tvm_tirx.ir.GenericConst
            )

        def static_scalar(tensor):
            if self.has_expr(tensor.tensor_idx):
                value = self.get_expr(tensor.tensor_idx).value.numpy()
            else:
                value = self.get_tensor_value(tensor)
            # TFLite RANGE operands are scalar tensors in the flatbuffer.
            assert value.size == 1, "RANGE scalar input must have exactly one element"
            return value.item()

        if not (is_dynamic(start) or is_dynamic(limit) or is_dynamic(delta)):
            return relax.op.arange(
                static_scalar(start), static_scalar(limit), static_scalar(delta), out_type
            )

        return self._convert_dynamic_range(start, limit, delta, out_type)

    def _scalar_tensor_to_dim(self, expr, name):
        """Lift a runtime scalar Relax expr to a symbolic ``tirx.Var`` dimension.

        Mirrors the ``tensor_to_shape`` + ``match_cast`` bridge used by
        ``_get_shape_expr_from_tensor`` so a data-dependent scalar can be used as
        a ``Expr`` (e.g. an output length). The scalar is cast to int64 first.
        """
        expr = self.bb.normalize(relax.op.astype(expr, "int64"))
        expr = self.bb.normalize(relax.op.reshape(expr, (1,)))
        expr = self.bb.match_cast(expr, relax.TensorType([1], "int64"))
        shape_var = self.bb.emit(relax.op.tensor_to_shape(expr))
        dim = tirx.Var(name, "int64")
        self.bb.match_cast(shape_var, relax.ShapeType([dim]))
        return dim

    def _convert_dynamic_range(self, start, limit, delta, out_type):
        """RANGE with dynamic (runtime) scalar bounds, for int and float dtypes.

        ``relax.op.arange`` only accepts compile-time ``Expr`` bounds, and its
        struct-info length formula lacks a negative-step branch, so feeding
        symbolic bounds directly would mis-declare descending ranges. Instead the
        element count ``max(0, ceil((limit - start) / delta))`` is computed
        in-graph and lifted to one symbolic dimension ``L`` (so the declared and
        runtime lengths match by construction); values are rebuilt as
        ``arange(0, L) * delta + start``.
        """
        # int ranges work in int64 for an exact, sign-agnostic count; float
        # ranges work in the output float dtype.
        work_type = out_type if out_type.startswith("float") else "int64"

        def scalar_expr(tensor):
            return self.bb.normalize(relax.op.astype(self.get_tensor_expr(tensor), work_type))

        start_e = scalar_expr(start)
        limit_e = scalar_expr(limit)
        delta_e = scalar_expr(delta)

        if work_type.startswith("float"):
            count = relax.op.ceil(relax.op.divide(relax.op.subtract(limit_e, start_e), delta_e))
        else:
            # ceil((limit - start) / delta) == -floordiv(start - limit, delta),
            # which stays exact and handles negative delta without a float cast.
            count = relax.op.negative(
                relax.op.floor_divide(relax.op.subtract(start_e, limit_e), delta_e)
            )
        count = relax.op.maximum(count, relax.const(0, work_type))
        dim = self._scalar_tensor_to_dim(count, "range_len")

        positions = self.bb.normalize(
            relax.op.astype(relax.op.arange(0, dim, 1, "int64"), work_type)
        )
        out = relax.op.add(relax.op.multiply(positions, delta_e), start_e)
        return out if work_type == out_type else relax.op.astype(out, out_type)

    def convert_rank(self, op):
        """Convert TFLite RANK."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_dtype = self.get_tensor_type_str(output_tensors[0].tensor.Type())

        rank = len(self.get_tensor_shape(input_tensors[0]))
        return relax.const(rank, dtype=output_dtype)

    def convert_shape(self, op):
        """Convert TFLite Shape"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.ShapeOptions import ShapeOptions
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.ShapeOptions
        op_options = op.BuiltinOptions()
        shape_options = ShapeOptions()
        shape_options.Init(op_options.Bytes, op_options.Pos)

        # SHAPE must materialize as a tensor output in Relax, not just symbolic shape info.
        out = relax.op.shape_to_tensor(relax.op.shape_of(self.get_tensor_expr(input_tensors[0])))
        if shape_options.OutType() == TensorType.INT32:
            out = relax.op.astype(out, "int32")

        return out

    def convert_bucketize(self, op):
        """Convert TFLite BUCKETIZE."""
        from tflite.BucketizeOptions import BucketizeOptions
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.BucketizeOptions
        op_options = op.BuiltinOptions()
        bucketize_options = BucketizeOptions()
        bucketize_options.Init(op_options.Bytes, op_options.Pos)

        boundaries = self.bb.normalize(
            relax.const(bucketize_options.BoundariesAsNumpy(), dtype="float32")
        )
        out_tensor = self.get_output_tensors(op)[0]
        out_int32 = out_tensor.tensor.Type() == TensorType.INT32
        return relax.op.bucketize(
            self.get_tensor_expr(input_tensors[0]), boundaries, out_int32=out_int32, right=False
        )

    def convert_relu(self, op):
        """Convert TFLite ReLU"""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_f32 = self.dequantize(in_expr, input_tensor)
            out = relax.op.nn.relu(in_f32)
            out = self.quantize(out, output_tensor)
        else:
            out = relax.op.nn.relu(in_expr)

        return out

    def convert_relu_0_to_1(self, op):
        """Convert TFLite RELU_0_TO_1."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_f32 = self.dequantize(in_expr, input_tensor)
            out = relax.op.clip(in_f32, 0, 1)
            out = self.quantize(out, output_tensor)
        else:
            out = relax.op.clip(in_expr, 0, 1)

        return out

    def convert_hard_swish(self, op):
        """Convert TFLite Hard swish"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        def _relu6(data):
            return relax.op.clip(data, min=0.0, max=6.0)

        def _hard_swish(data):
            return data * _relu6(data + relax.const(3.0)) / relax.const(6.0)

        # Dequantize if the input is quantized.
        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        # Perform hardswish
        out = _hard_swish(in_expr)

        # Go back to integer dataype if the original operator was quantized.
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_relu6(self, op):
        """Convert TFLite ReLU6"""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_f32 = self.dequantize(in_expr, input_tensor)
            out = relax.op.clip(in_f32, min=0, max=6)
            out = self.quantize(out, output_tensor)
        else:
            out = relax.op.clip(in_expr, min=0, max=6)

        return out

    def convert_leaky_relu(self, op):
        """Convert TFLite LEAKY_RELU"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.LeakyReluOptions import LeakyReluOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.LeakyReluOptions
        op_options = op.BuiltinOptions()
        leaky_relu_options = LeakyReluOptions()
        leaky_relu_options.Init(op_options.Bytes, op_options.Pos)
        alpha_tensor = leaky_relu_options.Alpha()

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.nn.leakyrelu(in_expr, alpha_tensor)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_relu_n1_to_1(self, op):
        """Convert TFLite RELU_N1_TO_1"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_f32 = self.dequantize(in_expr, input_tensor)
            out = relax.op.clip(in_f32, min=-1, max=1)
            out = self.quantize(out, output_tensor)
        else:
            out = relax.op.clip(in_expr, min=-1, max=1)

        return out

    def convert_log_softmax(self, op):
        """Convert TFLite LOG_SOFTMAX"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax.op.nn.log_softmax(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_concatenation(self, op):
        """Convert TFLite concatenation"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.ConcatenationOptions import ConcatenationOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors should greater than 1"
        in_exprs = [self.get_tensor_expr(_) for _ in input_tensors]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.ConcatenationOptions
        op_options = op.BuiltinOptions()
        concatenation_options = ConcatenationOptions()
        concatenation_options.Init(op_options.Bytes, op_options.Pos)
        concatenation_axis = concatenation_options.Axis()
        fused_activation_fn = concatenation_options.FusedActivationFunction()

        if not input_tensors[0].qnn_params:
            out = relax.op.concat(in_exprs, axis=concatenation_axis)
        else:
            in_f32s = [
                self.dequantize(expr, tensor) for expr, tensor in zip(in_exprs, input_tensors)
            ]
            out = relax.op.concat(in_f32s, axis=concatenation_axis)
            out = self.quantize(out, output_tensor)

        # Handle fused activations
        if output_tensor.qnn_params:
            scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=scale_val,
                zero_point=zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def _convert_unary_elemwise(self, op, relax_op):
        """Generic method to convert TFLite unary elemwise functions"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relax_op(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def _convert_stablehlo_unary(self, op, relax_op):
        """Convert a unary StableHLO TFLite builtin operator.

        StableHLO builtins do not have TFLite fused activation attributes. Keep
        this path independent from the regular TFLite elemwise/QNN helpers so
        StableHLO semantics are mapped directly to Relax operators.
        """
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        assert len(self.get_output_tensors(op)) == 1, "output tensors length should be 1"

        in_expr = self.get_tensor_expr(input_tensors[0])
        return relax_op(in_expr)

    def _convert_stablehlo_binary(self, op, relax_op):
        """Convert a binary StableHLO TFLite builtin operator.

        StableHLO builtins do not have TFLite fused activation attributes. Keep
        this path independent from the regular TFLite elemwise/QNN helpers so
        StableHLO semantics are mapped directly to Relax operators.
        """
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        assert len(self.get_output_tensors(op)) == 1, "output tensors length should be 1"

        lhs_expr = self.get_tensor_expr(input_tensors[0])
        rhs_expr = self.get_tensor_expr(input_tensors[1])
        return relax_op(lhs_expr, rhs_expr)

    def _convert_stablehlo_and(self, op):
        """Convert StableHLO AND for bool and integer tensors."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        assert len(self.get_output_tensors(op)) == 1, "output tensors length should be 1"

        lhs = self.get_tensor_expr(input_tensors[0])
        rhs = self.get_tensor_expr(input_tensors[1])
        dtype = lhs.ty.dtype.dtype
        if dtype == "bool":
            op_fn = _op.logical_and
        elif dtype.startswith(("int", "uint")):
            op_fn = _op.bitwise_and
        else:
            raise NotImplementedError(f"STABLEHLO_AND with dtype {dtype} is not supported")
        return self.bb.normalize(op_fn(lhs, rhs))

    def _convert_stablehlo_or(self, op):
        """Convert StableHLO OR for bool and integer tensors."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        assert len(self.get_output_tensors(op)) == 1, "output tensors length should be 1"

        lhs = self.get_tensor_expr(input_tensors[0])
        rhs = self.get_tensor_expr(input_tensors[1])
        dtype = lhs.ty.dtype.dtype
        if dtype == "bool":
            op_fn = _op.logical_or
        elif dtype.startswith(("int", "uint")):
            op_fn = _op.bitwise_or
        else:
            raise NotImplementedError(f"STABLEHLO_OR with dtype {dtype} is not supported")
        return self.bb.normalize(op_fn(lhs, rhs))

    def _convert_stablehlo_ternary(self, op, relax_op):
        """Convert a ternary StableHLO TFLite builtin operator.

        StableHLO builtins do not have TFLite fused activation attributes. Keep
        this path independent from the regular TFLite elemwise/QNN helpers so
        StableHLO semantics are mapped directly to Relax operators.
        """
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        assert len(self.get_output_tensors(op)) == 1, "output tensors length should be 1"

        arg0 = self.get_tensor_expr(input_tensors[0])
        arg1 = self.get_tensor_expr(input_tensors[1])
        arg2 = self.get_tensor_expr(input_tensors[2])
        return relax_op(arg0, arg1, arg2)

    def _get_stablehlo_options(self, op, options_cls):
        """Parse BuiltinOptions2 for a StableHLO TFLite builtin operator.

        Returns an initialized options object of the given class.
        """
        from tflite.BuiltinOptions2 import BuiltinOptions2

        op_options = op.BuiltinOptions2()
        if op_options is None:
            # A malformed flatbuffer may declare a BuiltinOptions2 type without
            # carrying the actual options table. Fail cleanly instead of raising
            # an opaque AttributeError when accessing the missing payload.
            raise NotImplementedError(
                f"{options_cls.__name__} is required but missing from the operator"
            )
        # Look up the expected BuiltinOptions2 enum value by matching the class
        # name to an enum member (e.g. StablehloConcatenateOptions → 1).
        options_type = getattr(BuiltinOptions2, options_cls.__name__, None)
        if options_type is not None:
            assert op.BuiltinOptions2Type() == options_type, (
                f"Unexpected BuiltinOptions2 type: expected "
                f"{options_cls.__name__}, got {op.BuiltinOptions2Type()}"
            )
        result = options_cls()
        result.Init(op_options.Bytes, op_options.Pos)
        return result

    def _get_static_tensor_shape(self, tensor, op_name):
        """Return a statically-known TFLite tensor shape as Python ints."""
        try:
            return [int(dim) for dim in self.get_tensor_shape(tensor)]
        except (TypeError, ValueError) as err:
            raise NotImplementedError(f"{op_name} requires statically-known tensor shapes") from err

    def _get_stablehlo_i64_vector(self, vector, default):
        """Convert an optional StableHLO int64 vector field to a Python int list."""
        if vector is None or isinstance(vector, int):
            return list(default)
        return [int(v) for v in vector]

    def _ensure_stablehlo_float_dtype(self, expr, op_name):
        """Return expr dtype if the StableHLO subset supports it."""
        dtype = expr.ty.dtype.dtype
        if not dtype.startswith("float"):
            raise NotImplementedError(f"{op_name} with dtype {dtype} is not supported")
        return dtype

    def _convert_stablehlo_cbrt(self, op):
        """Convert STABLEHLO_CBRT to a sign-preserving Relax expression."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        assert len(self.get_output_tensors(op)) == 1

        data = self.get_tensor_expr(input_tensors[0])
        dtype = self._ensure_stablehlo_float_dtype(data, "STABLEHLO_CBRT")
        zero = relax.const(0, dtype)
        exponent = relax.const(1.0 / 3.0, dtype)

        is_negative = self.bb.normalize(relax.op.less(data, zero))
        negative_base = self.bb.normalize(relax.op.negative(data))
        negative_root = self.bb.normalize(relax.op.power(negative_base, exponent))
        negative_result = self.bb.normalize(relax.op.negative(negative_root))
        positive_result = self.bb.normalize(relax.op.power(data, exponent))
        return self.bb.normalize(relax.op.where(is_negative, negative_result, positive_result))

    def _convert_stablehlo_remainder(self, op):
        """Convert STABLEHLO_REMAINDER to truncating remainder for float tensors."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        assert len(self.get_output_tensors(op)) == 1

        lhs = self.get_tensor_expr(input_tensors[0])
        rhs = self.get_tensor_expr(input_tensors[1])
        self._ensure_stablehlo_float_dtype(lhs, "STABLEHLO_REMAINDER")
        self._ensure_stablehlo_float_dtype(rhs, "STABLEHLO_REMAINDER")

        quotient = self.bb.normalize(relax.op.divide(lhs, rhs))
        truncated = self.bb.normalize(relax.op.trunc(quotient))
        product = self.bb.normalize(relax.op.multiply(rhs, truncated))
        return self.bb.normalize(relax.op.subtract(lhs, product))

    def _get_stablehlo_simple_body_op(self, body_subgraph_index, parent_op_name, input_count):
        """Return the single operator from a simple StableHLO body subgraph."""
        if body_subgraph_index <= 0 or body_subgraph_index >= self.model.SubgraphsLength():
            raise NotImplementedError(f"{parent_op_name} requires a valid non-main body subgraph")

        body_subgraph = self.model.Subgraphs(body_subgraph_index)
        if (
            body_subgraph.InputsLength() != input_count
            or body_subgraph.OutputsLength() != 1
            or body_subgraph.OperatorsLength() != 1
        ):
            raise NotImplementedError(f"{parent_op_name} only supports single-op body subgraphs")

        return body_subgraph.Operators(0)

    def _check_stablehlo_reduce_init(
        self, init_tensor, reducer_name, parent_op_name="STABLEHLO_REDUCE"
    ):
        """Validate that the StableHLO reduce init value matches the Relax identity."""
        if self.has_expr(init_tensor.tensor_idx):
            raise NotImplementedError(f"{parent_op_name} with dynamic init values is not supported")

        init_value = np.asarray(self.get_tensor_value(init_tensor))
        if init_value.shape not in [(), (1,)]:
            raise NotImplementedError(f"{parent_op_name} requires scalar init values")

        dtype = init_value.dtype
        scalar = init_value.item()
        if reducer_name == "STABLEHLO_ADD":
            is_identity = bool(np.isclose(scalar, 0))
        elif reducer_name == "STABLEHLO_MULTIPLY":
            is_identity = bool(np.isclose(scalar, 1))
        elif reducer_name == "STABLEHLO_MAXIMUM":
            if np.issubdtype(dtype, np.floating):
                is_identity = bool(np.isneginf(scalar))
            elif np.issubdtype(dtype, np.integer):
                is_identity = scalar == np.iinfo(dtype).min
            else:
                is_identity = False
        elif reducer_name == "STABLEHLO_MINIMUM":
            if np.issubdtype(dtype, np.floating):
                is_identity = bool(np.isposinf(scalar))
            elif np.issubdtype(dtype, np.integer):
                is_identity = scalar == np.iinfo(dtype).max
            else:
                is_identity = False
        else:
            raise NotImplementedError(f"{parent_op_name} reducer {reducer_name} is not supported")

        if not is_identity:
            raise NotImplementedError(
                f"{parent_op_name} init value must match the reducer identity"
            )

    def _convert_stablehlo_reduce(self, op):
        """Convert the single-input STABLEHLO_REDUCE subset to Relax reductions."""
        from tflite.StablehloReduceOptions import StablehloReduceOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        assert len(self.get_output_tensors(op)) == 1

        opts = self._get_stablehlo_options(op, StablehloReduceOptions)
        dimensions = self._get_stablehlo_i64_vector(opts.DimensionsAsNumpy(), [])
        body_op = self._get_stablehlo_simple_body_op(
            int(opts.BodySubgraphIndex()), "STABLEHLO_REDUCE", 2
        )
        reducer_name = self.get_op_code_str(body_op)

        reducers = {
            "STABLEHLO_ADD": relax.op.sum,
            "STABLEHLO_MAXIMUM": relax.op.max,
            "STABLEHLO_MINIMUM": relax.op.min,
            "STABLEHLO_MULTIPLY": relax.op.prod,
        }
        if reducer_name not in reducers:
            raise NotImplementedError(f"STABLEHLO_REDUCE reducer {reducer_name} is not supported")

        self._check_stablehlo_reduce_init(input_tensors[1], reducer_name)
        data = self.get_tensor_expr(input_tensors[0])
        return self.bb.normalize(reducers[reducer_name](data, axis=dimensions, keepdims=False))

    def _convert_stablehlo_reduce_window(self, op):
        """Convert the NHWC 2D max-pool STABLEHLO_REDUCE_WINDOW subset."""
        from tflite.StablehloReduceWindowOptions import StablehloReduceWindowOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        assert len(self.get_output_tensors(op)) == 1

        opts = self._get_stablehlo_options(op, StablehloReduceWindowOptions)
        body_op = self._get_stablehlo_simple_body_op(
            int(opts.BodySubgraphIndex()), "STABLEHLO_REDUCE_WINDOW", 2
        )
        reducer_name = self.get_op_code_str(body_op)
        if reducer_name != "STABLEHLO_MAXIMUM":
            raise NotImplementedError(
                "STABLEHLO_REDUCE_WINDOW only supports MAXIMUM reducer windows"
            )
        self._check_stablehlo_reduce_init(input_tensors[1], reducer_name, "STABLEHLO_REDUCE_WINDOW")

        data_shape = self._get_static_tensor_shape(input_tensors[0], "STABLEHLO_REDUCE_WINDOW")
        if len(data_shape) != 4:
            raise NotImplementedError("STABLEHLO_REDUCE_WINDOW only supports 4D input")

        window_dimensions = self._get_stablehlo_i64_vector(opts.WindowDimensionsAsNumpy(), [])
        window_strides = self._get_stablehlo_i64_vector(
            opts.WindowStridesAsNumpy(), [1] * len(window_dimensions)
        )
        base_dilations = self._get_stablehlo_i64_vector(
            opts.BaseDilationsAsNumpy(), [1] * len(window_dimensions)
        )
        window_dilations = self._get_stablehlo_i64_vector(
            opts.WindowDilationsAsNumpy(), [1] * len(window_dimensions)
        )
        padding = self._get_stablehlo_i64_vector(
            opts.PaddingAsNumpy(), [0] * (2 * len(window_dimensions))
        )

        if (
            len(window_dimensions) != 4
            or len(window_strides) != 4
            or len(base_dilations) != 4
            or len(window_dilations) != 4
            or len(padding) != 8
        ):
            raise NotImplementedError(
                "STABLEHLO_REDUCE_WINDOW only supports rank-4 window attributes"
            )
        if window_dimensions[0] != 1 or window_dimensions[3] != 1:
            raise NotImplementedError(
                "STABLEHLO_REDUCE_WINDOW only supports pooling over spatial dimensions"
            )
        if window_strides[0] != 1 or window_strides[3] != 1:
            raise NotImplementedError(
                "STABLEHLO_REDUCE_WINDOW only supports unit batch/channel strides"
            )
        if base_dilations != [1, 1, 1, 1]:
            raise NotImplementedError("STABLEHLO_REDUCE_WINDOW with base dilation is not supported")
        if padding[0] != 0 or padding[1] != 0 or padding[6] != 0 or padding[7] != 0:
            raise NotImplementedError("STABLEHLO_REDUCE_WINDOW only supports spatial padding")

        data = self.get_tensor_expr(input_tensors[0])
        return self.bb.normalize(
            relax.op.nn.max_pool2d(
                data,
                pool_size=[window_dimensions[1], window_dimensions[2]],
                strides=[window_strides[1], window_strides[2]],
                padding=[padding[2], padding[4], padding[3], padding[5]],
                dilation=[window_dilations[1], window_dilations[2]],
                layout="NHWC",
                out_layout="NHWC",
            )
        )

    def _convert_stablehlo_scatter(self, op):
        """Convert the canonical point-update STABLEHLO_SCATTER subset."""
        from tflite.StablehloScatterOptions import StablehloScatterOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"
        assert len(self.get_output_tensors(op)) == 1

        opts = self._get_stablehlo_options(op, StablehloScatterOptions)
        operand_shape = self._get_static_tensor_shape(input_tensors[0], "STABLEHLO_SCATTER")
        indices_shape = self._get_static_tensor_shape(input_tensors[1], "STABLEHLO_SCATTER")
        updates_shape = self._get_static_tensor_shape(input_tensors[2], "STABLEHLO_SCATTER")
        operand_rank = len(operand_shape)
        indices_rank = len(indices_shape)

        update_window_dims = self._get_stablehlo_i64_vector(opts.UpdateWindowDimsAsNumpy(), [])
        inserted_window_dims = self._get_stablehlo_i64_vector(opts.InsertedWindowDimsAsNumpy(), [])
        scatter_dims_to_operand_dims = self._get_stablehlo_i64_vector(
            opts.ScatterDimsToOperandDimsAsNumpy(), []
        )
        index_vector_dim = int(opts.IndexVectorDim())

        if indices_rank == 0 or index_vector_dim != indices_rank - 1:
            raise NotImplementedError(
                "STABLEHLO_SCATTER only supports trailing index-vector dimensions"
            )
        if update_window_dims:
            raise NotImplementedError(
                "STABLEHLO_SCATTER only supports point updates without update windows"
            )
        if inserted_window_dims != list(range(operand_rank)):
            raise NotImplementedError(
                "STABLEHLO_SCATTER only supports point updates for every operand dimension"
            )
        if scatter_dims_to_operand_dims != list(range(operand_rank)):
            raise NotImplementedError(
                "STABLEHLO_SCATTER only supports canonical scatter-to-operand dimensions"
            )
        if indices_shape[-1] != operand_rank or updates_shape != indices_shape[:-1]:
            raise NotImplementedError(
                "STABLEHLO_SCATTER requires point update shapes to match scatter indices"
            )

        body_op = self._get_stablehlo_simple_body_op(
            int(opts.UpdateComputationSubgraphIndex()), "STABLEHLO_SCATTER", 2
        )
        reducer_name = self.get_op_code_str(body_op)
        reductions = {
            "STABLEHLO_ADD": "add",
            "STABLEHLO_MAXIMUM": "max",
            "STABLEHLO_MINIMUM": "min",
            "STABLEHLO_MULTIPLY": "mul",
        }
        if reducer_name not in reductions:
            raise NotImplementedError(f"STABLEHLO_SCATTER reducer {reducer_name} is not supported")

        operand = self.get_tensor_expr(input_tensors[0])
        indices = self.get_tensor_expr(input_tensors[1])
        updates = self.get_tensor_expr(input_tensors[2])
        return self.bb.normalize(
            relax.op.scatter_nd(operand, indices, updates, reductions[reducer_name])
        )

    def _convert_stablehlo_composite(self, op):
        """Convert STABLEHLO_COMPOSITE by inlining a simple decomposition subgraph."""
        from tflite.StableHLOCompositeOptions import StableHLOCompositeOptions

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(output_tensors) != 1:
            raise NotImplementedError(
                "STABLEHLO_COMPOSITE only supports single-output decompositions"
            )

        opts = self._get_stablehlo_options(op, StableHLOCompositeOptions)
        composite_name = opts.Name()
        composite_name = (
            composite_name.decode("utf-8") if composite_name is not None else "<unnamed>"
        )
        if opts.CompositeAttributesLength() != 0:
            raise NotImplementedError(
                f"STABLEHLO_COMPOSITE {composite_name} with composite attributes is not supported"
            )

        decomposition_subgraph_index = int(opts.DecompositionSubgraphIndex())
        if (
            decomposition_subgraph_index <= 0
            or decomposition_subgraph_index >= self.model.SubgraphsLength()
        ):
            raise NotImplementedError(
                f"STABLEHLO_COMPOSITE {composite_name} requires a valid decomposition subgraph"
            )
        decomposition_subgraph = self.model.Subgraphs(decomposition_subgraph_index)
        if decomposition_subgraph.InputsLength() != len(input_tensors):
            raise NotImplementedError(
                f"STABLEHLO_COMPOSITE {composite_name} decomposition input count mismatch"
            )
        if decomposition_subgraph.OutputsLength() != 1:
            raise NotImplementedError(
                f"STABLEHLO_COMPOSITE {composite_name} only supports single-output decompositions"
            )

        decomposition_exp_tab = ExprTable()
        decomposition_converter = OperatorConverter(
            self.model, decomposition_subgraph, decomposition_exp_tab, self.bb
        )
        for decomposition_input_idx, composite_input in zip(
            decomposition_subgraph.InputsAsNumpy(), input_tensors
        ):
            decomposition_input_name = get_tensor_name(
                decomposition_subgraph, int(decomposition_input_idx)
            )
            decomposition_exp_tab.set_expr(
                decomposition_input_name,
                self.get_tensor_expr(composite_input),
                force_override=True,
            )

        decomposition_converter.check_unsupported_ops()
        decomposition_converter.convert_op_to_relax()
        decomposition_output_idx = int(decomposition_subgraph.Outputs(0))
        decomposition_output_tensor = decomposition_converter.get_tensors(
            [decomposition_output_idx]
        )[0]
        for const_expr, value in decomposition_exp_tab.params.values():
            param_name = f"_param_{self.exp_tab.const_ctr}"
            self.exp_tab.const_ctr += 1
            self.exp_tab.params[param_name] = (const_expr, value)
        return decomposition_converter.get_tensor_expr(decomposition_output_tensor)

    def _convert_stablehlo_sort(self, op):
        """Convert the single-input STABLEHLO_SORT subset to Relax sort."""
        from tflite.StablehloCompareOptions import StablehloCompareOptions
        from tflite.StablehloComparisonDirection import StablehloComparisonDirection
        from tflite.StablehloComparisonType import StablehloComparisonType
        from tflite.StablehloSortOptions import StablehloSortOptions

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 1 or len(output_tensors) != 1:
            raise NotImplementedError(
                "STABLEHLO_SORT only supports single-input single-output sort"
            )

        opts = self._get_stablehlo_options(op, StablehloSortOptions)
        if opts.IsStable():
            raise NotImplementedError("STABLEHLO_SORT stable sort is not supported")

        body_op = self._get_stablehlo_simple_body_op(
            int(opts.ComparatorSubgraphIndex()), "STABLEHLO_SORT", 2
        )
        comparator_name = self.get_op_code_str(body_op)
        if comparator_name != "STABLEHLO_COMPARE":
            raise NotImplementedError(
                f"STABLEHLO_SORT comparator {comparator_name} is not supported"
            )

        compare_opts = self._get_stablehlo_options(body_op, StablehloCompareOptions)
        if (
            compare_opts.CompareType()
            == StablehloComparisonType.STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER
        ):
            raise NotImplementedError("STABLEHLO_SORT with TOTALORDER comparator is not supported")

        direction = compare_opts.ComparisonDirection()
        _DIR = StablehloComparisonDirection
        if direction == _DIR.STABLEHLO_COMPARISON_DIRECTION_LT:
            descending = False
        elif direction == _DIR.STABLEHLO_COMPARISON_DIRECTION_GT:
            descending = True
        else:
            raise NotImplementedError("STABLEHLO_SORT only supports LT or GT comparators")

        data = self.get_tensor_expr(input_tensors[0])
        return self.bb.normalize(
            relax.op.sort(data, axis=int(opts.Dimension()), descending=descending)
        )

    def _convert_stablehlo_custom_call(self, op):
        """Convert supported annotation-only STABLEHLO_CUSTOM_CALL targets."""
        from tflite.StablehloCustomCallOptions import StablehloCustomCallOptions

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        opts = self._get_stablehlo_options(op, StablehloCustomCallOptions)
        call_target_name = self._decode_tflite_string(opts.CallTargetName())

        if call_target_name == "Sharding":
            # TensorFlow treats Sharding custom calls as metadata annotations
            # and may erase them by replacing the op with its input. Mirror
            # that identity semantics for the safe single-input/single-output
            # subset. The sharding spec in backend_config is intentionally
            # dropped for single-device import. TFLite has no runtime kernel
            # for general STABLEHLO_CUSTOM_CALL targets.
            if opts.HasSideEffect():
                raise NotImplementedError(
                    "STABLEHLO_CUSTOM_CALL Sharding with side effects is not supported"
                )
            if opts.CalledComputationsLength() != 0:
                raise NotImplementedError(
                    "STABLEHLO_CUSTOM_CALL Sharding with called computations is not supported"
                )
            if len(input_tensors) != 1 or len(output_tensors) != 1:
                raise NotImplementedError(
                    "STABLEHLO_CUSTOM_CALL Sharding requires one input and one output"
                )
            self._check_tensor_metadata_match(
                input_tensors[0], output_tensors[0], "STABLEHLO_CUSTOM_CALL", "Sharding"
            )
            return self.get_tensor_expr(input_tensors[0])

        target = call_target_name or "<empty>"
        raise NotImplementedError(f"STABLEHLO_CUSTOM_CALL target {target} is not supported")

    def _convert_stablehlo_rng_bit_generator(self, op):
        """Convert STABLEHLO_RNG_BIT_GENERATOR to a bit-exact call_tir kernel."""
        from tflite.RngAlgorithm import RngAlgorithm
        from tflite.StablehloRngBitGeneratorOptions import StablehloRngBitGeneratorOptions

        op_name = "STABLEHLO_RNG_BIT_GENERATOR"
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 1 or len(output_tensors) != 2:
            raise NotImplementedError(f"{op_name} expects one input and two outputs")

        opts = self._get_stablehlo_options(op, StablehloRngBitGeneratorOptions)
        algorithm_enum = opts.Algorithm()
        # DEFAULT resolves to PHILOX in the TFLite runtime kernel.
        if algorithm_enum == RngAlgorithm.THREEFRY:
            algorithm = "threefry"
        elif algorithm_enum in (RngAlgorithm.PHILOX, RngAlgorithm.DEFAULT):
            algorithm = "philox"
        else:
            raise NotImplementedError(f"{op_name} algorithm {algorithm_enum} is not supported")

        state_tensor = input_tensors[0]
        if self.get_tensor_type_str(state_tensor.tensor.Type()) != "uint64":
            raise NotImplementedError(f"{op_name} requires a uint64 initial state")
        state_shape = self._get_static_tensor_shape(state_tensor, op_name)
        if len(state_shape) != 1:
            raise NotImplementedError(f"{op_name} requires a 1-D initial state")
        state_len = int(state_shape[0])
        # State-length constraints mirror the TFLite runtime kernel.
        if algorithm == "threefry" and state_len != 2:
            raise NotImplementedError(f"{op_name} THREEFRY requires a u64[2] state")
        if algorithm == "philox" and state_len not in (2, 3):
            raise NotImplementedError(f"{op_name} PHILOX requires a u64[2] or u64[3] state")

        out_state_tensor, out_tensor = output_tensors
        if self.get_tensor_type_str(out_state_tensor.tensor.Type()) != "uint64":
            raise NotImplementedError(f"{op_name} output state must be uint64")
        out_state_shape = self._get_static_tensor_shape(out_state_tensor, op_name)
        if list(out_state_shape) != list(state_shape):
            raise NotImplementedError(f"{op_name} output state shape must match the initial state")
        out_dtype = self.get_tensor_type_str(out_tensor.tensor.Type())
        if out_dtype not in ("int32", "int64", "uint32", "uint64"):
            raise NotImplementedError(f"{op_name} output dtype {out_dtype} is not supported")
        out_shape = tuple(self._get_static_tensor_shape(out_tensor, op_name))

        function = _build_stablehlo_rng_bit_generator_function(
            algorithm, state_len, out_dtype, out_shape
        )
        module_builder = self.conversion_state["module_builder"]
        func_name = f"tflite_stablehlo_rng_{algorithm}_{out_state_tensor.tensor_idx}"
        gv = module_builder.add_func(function, func_name)
        state_expr = self.get_tensor_expr(state_tensor)
        call = relax.call_tir(
            gv,
            [state_expr],
            ty_args=[
                _TupleType(
                    [
                        relax.TensorType(tuple(state_shape), "uint64"),
                        relax.TensorType(out_shape, out_dtype),
                    ]
                )
            ],
        )
        return self.bb.normalize(call)

    def _convert_stablehlo_while(self, op):
        """Convert STABLEHLO_WHILE to a recursive Relax private function."""
        from tflite.StablehloWhileOptions import StablehloWhileOptions

        opts = self._get_stablehlo_options(op, StablehloWhileOptions)
        return self._convert_while_like(
            op,
            "STABLEHLO_WHILE",
            int(opts.CondSubgraphIndex()),
            int(opts.BodySubgraphIndex()),
            "tflite_stablehlo_while",
        )

    def _get_builtin_options(self, op, options_cls):
        """Parse BuiltinOptions for a TFLite builtin operator."""
        from tflite.BuiltinOptions import BuiltinOptions

        op_options = op.BuiltinOptions()
        if op_options is None:
            raise NotImplementedError(f"{options_cls.__name__} is required")

        options_type = getattr(BuiltinOptions, options_cls.__name__, None)
        if options_type is not None and op.BuiltinOptionsType() != options_type:
            raise NotImplementedError(
                f"Unexpected BuiltinOptions type: expected "
                f"{options_cls.__name__}, got {op.BuiltinOptionsType()}"
            )
        result = options_cls()
        result.Init(op_options.Bytes, op_options.Pos)
        return result

    def _get_subgraph(self, subgraph_index, op_name, allow_main=False):
        """Return a validated TFLite subgraph by index."""
        if subgraph_index < 0 or subgraph_index >= self.model.SubgraphsLength():
            raise NotImplementedError(f"{op_name} requires a valid subgraph index")
        if not allow_main and subgraph_index == 0:
            raise NotImplementedError(f"{op_name} cannot target the main subgraph")
        return self.model.Subgraphs(subgraph_index)

    def _make_tuple_or_single(self, exprs):
        """Return a single expression or Relax tuple for a list of expressions."""
        if len(exprs) == 1:
            return exprs[0]
        return relax.Tuple(exprs)

    def _indices_or_empty(self, indices):
        """Return a TFLite index vector, using an empty list for absent vectors."""
        return indices if indices is not None else []

    def _check_subgraph_io(self, subgraph_index, op_name, input_count=None, output_count=None):
        """Validate a referenced subgraph's input and output counts."""
        subgraph = self._get_subgraph(subgraph_index, op_name)
        if input_count is not None and subgraph.InputsLength() != input_count:
            raise NotImplementedError(f"{op_name} subgraph input count mismatch")
        if output_count is not None and subgraph.OutputsLength() != output_count:
            raise NotImplementedError(f"{op_name} subgraph output count mismatch")
        return subgraph

    def _check_subgraph_interface(
        self,
        subgraph_index,
        op_name,
        input_tensors=None,
        output_tensors=None,
        input_count=None,
        output_count=None,
    ):
        """Validate a referenced subgraph's arity and tensor metadata."""
        if input_tensors is not None:
            input_count = len(input_tensors)
        if output_tensors is not None:
            output_count = len(output_tensors)

        subgraph = self._check_subgraph_io(
            subgraph_index, op_name, input_count=input_count, output_count=output_count
        )
        if input_tensors is not None:
            self._check_subgraph_tensor_metadata(
                subgraph,
                op_name,
                "subgraph input",
                subgraph.InputsAsNumpy(),
                input_tensors,
            )
        if output_tensors is not None:
            self._check_subgraph_tensor_metadata(
                subgraph,
                op_name,
                "subgraph output",
                subgraph.OutputsAsNumpy(),
                output_tensors,
            )
        return subgraph

    def _get_tensor_metadata(self, tensor):
        """Return static shape and dtype metadata for a TFLite tensor."""
        if isinstance(tensor, TensorWrapper):
            tensor = tensor.tensor
        shape = self._get_relax_tensor_shape(tensor)
        dtype = self._get_relax_tensor_dtype(tensor)
        return shape, dtype

    def _check_tensor_metadata_match(self, actual, expected, op_name, tensor_role):
        """Validate that two TFLite tensors have matching static metadata."""
        if self._get_tensor_metadata(actual) != self._get_tensor_metadata(expected):
            raise NotImplementedError(f"{op_name} {tensor_role} tensor metadata mismatch")

    def _check_subgraph_tensor_metadata(
        self, subgraph, op_name, tensor_role, subgraph_indices, expected_tensors
    ):
        """Validate referenced subgraph tensor metadata against caller tensors."""
        for subgraph_index, expected_tensor in zip(
            self._indices_or_empty(subgraph_indices), expected_tensors
        ):
            self._check_tensor_metadata_match(
                subgraph.Tensors(int(subgraph_index)),
                expected_tensor,
                op_name,
                tensor_role,
            )

    def _require_scalar_bool_tensor(self, tensor, op_name):
        """Validate that a TFLite tensor is a scalar bool tensor."""
        if isinstance(tensor, TensorWrapper):
            tensor = tensor.tensor
        dtype = self.get_tensor_type_str(tensor.Type())
        if dtype != "bool" or tensor.ShapeLength() != 0:
            raise NotImplementedError(f"{op_name} requires a scalar bool condition")

    def _get_subgraph_params(self, subgraph):
        """Create Relax parameters for a TFLite subgraph."""
        params = []
        exp_tab = ExprTable()
        for input_index in self._indices_or_empty(subgraph.InputsAsNumpy()):
            tensor = subgraph.Tensors(int(input_index))
            input_name = get_tensor_name(subgraph, int(input_index))
            shape = self._get_relax_tensor_shape(tensor)
            dtype = self._get_relax_tensor_dtype(tensor)
            param = relax.Var(input_name, relax.TensorType(shape=shape, dtype=dtype))
            exp_tab.set_expr(input_name, param)
            params.append(param)
        return params, exp_tab

    def _get_tensor_param(self, tensor_wrapper):
        """Create a Relax parameter from TFLite tensor metadata."""
        name = get_tensor_name(self.subgraph, tensor_wrapper.tensor_idx)
        shape = self._get_relax_tensor_shape(tensor_wrapper)
        dtype = self._get_relax_tensor_dtype(tensor_wrapper)
        return relax.Var(name, relax.TensorType(shape=shape, dtype=dtype))

    def _lower_subgraph_to_function(self, subgraph_index, function_name_hint, op_name="CALL"):
        """Lower a TFLite subgraph into a private Relax function."""
        lowered_subgraphs = self.conversion_state["lowered_subgraphs"]
        if subgraph_index in lowered_subgraphs:
            return lowered_subgraphs[subgraph_index]

        lowering_stack = self.conversion_state["lowering_stack"]
        if subgraph_index in lowering_stack:
            raise NotImplementedError(f"Recursive TFLite {op_name} subgraphs are not supported")

        subgraph = self._get_subgraph(subgraph_index, op_name)
        lowering_stack.append(subgraph_index)
        try:
            params, subgraph_exp_tab = self._get_subgraph_params(subgraph)
            subgraph_bb = relax.BlockBuilder()
            with subgraph_bb.function(function_name_hint, params=params, private=True):
                with subgraph_bb.dataflow():
                    subgraph_converter = type(self)(
                        self.model,
                        subgraph,
                        subgraph_exp_tab,
                        subgraph_bb,
                        self.conversion_state,
                    )
                    subgraph_converter.check_unsupported_ops()
                    subgraph_converter.convert_op_to_relax()
                    output_tensors = subgraph_converter.get_tensors(subgraph.OutputsAsNumpy())
                    outputs = [
                        subgraph_converter.get_tensor_expr(tensor) for tensor in output_tensors
                    ]
                    output = subgraph_bb.emit_output(self._make_tuple_or_single(outputs))
                subgraph_bb.emit_func_output(output)

            subgraph_mod = subgraph_bb.get()
            module_builder = self.conversion_state["module_builder"]
            gv = module_builder.add_func(subgraph_mod[function_name_hint], function_name_hint)
            lowered_subgraphs[subgraph_index] = gv
            return gv
        finally:
            lowering_stack.pop()

    def _bind_call_outputs(self, call, output_count):
        """Return per-output expressions from a single or tuple-valued call."""
        if output_count == 1:
            return [call]
        return [call[index] for index in range(output_count)]

    def _lower_if_to_function(
        self,
        then_subgraph_index,
        else_subgraph_index,
        input_tensors,
        branch_input_count,
        output_count,
    ):
        """Lower a TFLite IF op into a private Relax function."""
        cache_key = (then_subgraph_index, else_subgraph_index, branch_input_count, output_count)
        lowered_if_functions = self.conversion_state["lowered_if_functions"]
        if cache_key in lowered_if_functions:
            return lowered_if_functions[cache_key]

        then_func = self._lower_subgraph_to_function(
            then_subgraph_index,
            f"tflite_if_then_subgraph_{then_subgraph_index}",
            op_name="IF",
        )
        else_func = self._lower_subgraph_to_function(
            else_subgraph_index,
            f"tflite_if_else_subgraph_{else_subgraph_index}",
            op_name="IF",
        )
        if_name = f"tflite_if_subgraph_{then_subgraph_index}_{else_subgraph_index}"
        params = [self._get_tensor_param(tensor) for tensor in input_tensors]
        cond = params[0]
        branch_args = params[1:]

        if_bb = relax.BlockBuilder()
        with if_bb.function(if_name, params=params, private=True):
            result = relax.IfExpr(
                cond,
                relax.Call(then_func, branch_args),
                relax.Call(else_func, branch_args),
            )
            if_bb.emit_func_output(result)
        if_func = if_bb.get()[if_name]
        module_builder = self.conversion_state["module_builder"]
        gv = module_builder.add_func(if_func, if_name)
        lowered_if_functions[cache_key] = gv
        return gv

    def _lower_while_to_function(
        self,
        cond_subgraph_index,
        body_subgraph_index,
        loop_var_count,
        cond_func,
        body_func,
        body_subgraph,
        function_prefix="tflite_while",
    ):
        """Lower a TFLite WHILE op into a recursive private Relax function."""
        cache_key = (function_prefix, cond_subgraph_index, body_subgraph_index, loop_var_count)
        lowered_while_functions = self.conversion_state["lowered_while_functions"]
        if cache_key in lowered_while_functions:
            return lowered_while_functions[cache_key]

        loop_name = f"{function_prefix}_subgraph_{cond_subgraph_index}_{body_subgraph_index}"
        params, _ = self._get_subgraph_params(body_subgraph)
        dummy_body = self._make_tuple_or_single(params)
        module_builder = self.conversion_state["module_builder"]
        loop_gv = module_builder.add_func(relax.Function(params, dummy_body), loop_name)
        lowered_while_functions[cache_key] = loop_gv

        loop_bb = relax.BlockBuilder()
        with loop_bb.function(loop_name, params=params, private=True):
            cond = loop_bb.emit(relax.Call(cond_func, params), "while_cond")
            next_state = relax.Call(body_func, params)
            next_args = self._bind_call_outputs(next_state, loop_var_count)
            true_branch = relax.Call(loop_gv, next_args)
            false_branch = self._make_tuple_or_single(params)
            result = relax.IfExpr(cond, true_branch, false_branch)
            loop_bb.emit_func_output(result)
        loop_func = loop_bb.get()[loop_name]
        module_builder.update_func(loop_gv, loop_func)
        return loop_gv

    def convert_call(self, op):
        """Convert TFLite CALL to a Relax private function call."""
        from tflite.CallOptions import CallOptions

        opts = self._get_builtin_options(op, CallOptions)
        subgraph_index = int(opts.Subgraph())
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        self._check_subgraph_interface(
            subgraph_index,
            "CALL",
            input_tensors=input_tensors,
            output_tensors=output_tensors,
        )

        callee = self._lower_subgraph_to_function(
            subgraph_index, f"tflite_call_subgraph_{subgraph_index}", op_name="CALL"
        )
        args = [self.get_tensor_expr(tensor) for tensor in input_tensors]
        return relax.Call(callee, args)

    def convert_if(self, op):
        """Convert TFLite IF to Relax If with private branch functions."""
        from tflite.IfOptions import IfOptions

        opts = self._get_builtin_options(op, IfOptions)
        then_subgraph_index = int(opts.ThenSubgraphIndex())
        else_subgraph_index = int(opts.ElseSubgraphIndex())
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) < 1:
            raise NotImplementedError("IF requires a condition input")

        self._require_scalar_bool_tensor(input_tensors[0], "IF")
        branch_input_count = len(input_tensors) - 1
        output_count = len(output_tensors)
        branch_input_tensors = input_tensors[1:]
        self._check_subgraph_interface(
            then_subgraph_index,
            "IF",
            input_tensors=branch_input_tensors,
            output_tensors=output_tensors,
        )
        self._check_subgraph_interface(
            else_subgraph_index,
            "IF",
            input_tensors=branch_input_tensors,
            output_tensors=output_tensors,
        )

        if_func = self._lower_if_to_function(
            then_subgraph_index,
            else_subgraph_index,
            input_tensors,
            branch_input_count,
            output_count,
        )
        args = [self.get_tensor_expr(tensor) for tensor in input_tensors]
        return relax.Call(if_func, args)

    def _convert_while_like(
        self, op, op_name, cond_subgraph_index, body_subgraph_index, function_prefix
    ):
        """Convert a TFLite while-like operator with referenced cond/body subgraphs."""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        loop_var_count = len(input_tensors)
        if loop_var_count == 0:
            raise NotImplementedError(f"{op_name} requires loop-carried inputs")
        if len(output_tensors) != loop_var_count:
            raise NotImplementedError(f"{op_name} output count must match input count")

        cond_subgraph = self._check_subgraph_interface(
            cond_subgraph_index,
            op_name,
            input_tensors=input_tensors,
            output_count=1,
        )
        body_subgraph = self._check_subgraph_interface(
            body_subgraph_index,
            op_name,
            input_tensors=input_tensors,
            output_tensors=input_tensors,
        )
        for input_tensor, output_tensor in zip(input_tensors, output_tensors):
            self._check_tensor_metadata_match(input_tensor, output_tensor, op_name, "loop state")
        cond_output = cond_subgraph.Tensors(int(cond_subgraph.Outputs(0)))
        self._require_scalar_bool_tensor(cond_output, op_name)

        cond_func = self._lower_subgraph_to_function(
            cond_subgraph_index,
            f"{function_prefix}_cond_subgraph_{cond_subgraph_index}",
            op_name=op_name,
        )
        body_func = self._lower_subgraph_to_function(
            body_subgraph_index,
            f"{function_prefix}_body_subgraph_{body_subgraph_index}",
            op_name=op_name,
        )

        loop_gv = self._lower_while_to_function(
            cond_subgraph_index,
            body_subgraph_index,
            loop_var_count,
            cond_func,
            body_func,
            body_subgraph,
            function_prefix=function_prefix,
        )

        args = [self.get_tensor_expr(tensor) for tensor in input_tensors]
        return relax.Call(loop_gv, args)

    def convert_while(self, op):
        """Convert TFLite WHILE to a recursive Relax private function."""
        from tflite.WhileOptions import WhileOptions

        opts = self._get_builtin_options(op, WhileOptions)
        return self._convert_while_like(
            op,
            "WHILE",
            int(opts.CondSubgraphIndex()),
            int(opts.BodySubgraphIndex()),
            "tflite_while",
        )

    def convert_call_once(self, op):
        """Convert TFLite CALL_ONCE for no-op and resource-variable initialization subsets."""
        from tflite.CallOnceOptions import CallOnceOptions

        opts = self._get_builtin_options(op, CallOnceOptions)
        init_subgraph_index = int(opts.InitSubgraphIndex())
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 0 or len(output_tensors) != 0:
            raise NotImplementedError("CALL_ONCE with inputs or outputs is not supported")

        init_subgraph = self._get_subgraph(init_subgraph_index, "CALL_ONCE")
        if init_subgraph.InputsLength() != 0 or init_subgraph.OutputsLength() != 0:
            raise NotImplementedError("CALL_ONCE with non-empty init subgraph I/O is not supported")
        if init_subgraph.OperatorsLength() != 0:
            self._convert_call_once_init_subgraph(init_subgraph)
        return None

    def _convert_call_once_init_subgraph(self, init_subgraph):
        """Convert the resource-variable initialization subset of a CALL_ONCE subgraph."""
        supported_init_ops = {"VAR_HANDLE", "ASSIGN_VARIABLE", "HASHTABLE", "HASHTABLE_IMPORT"}
        for op_idx in range(init_subgraph.OperatorsLength()):
            op_name = self.get_op_code_str(init_subgraph.Operators(op_idx))
            if op_name not in supported_init_ops:
                raise NotImplementedError(
                    f"CALL_ONCE init subgraph operator {op_name} is not supported"
                )

        old_in_call_once_init = self.conversion_state["in_call_once_init"]
        self.conversion_state["in_call_once_init"] = True
        try:
            # The supported init ops below only update importer state and return None.
            # If future CALL_ONCE ops emit Relax bindings, revisit sharing the parent builder.
            subgraph_converter = type(self)(
                self.model,
                init_subgraph,
                ExprTable(),
                self.bb,
                self.conversion_state,
            )
            subgraph_converter.check_unsupported_ops()
            subgraph_converter.convert_op_to_relax()
        finally:
            self.conversion_state["in_call_once_init"] = old_in_call_once_init

    def _convert_stablehlo_convert(self, op):
        """Convert STABLEHLO_CONVERT to Relax (astype).

        Reads the output tensor dtype from the TFLite schema and applies
        relax.op.astype.  This path is intentionally separate from the
        generic _convert_stablehlo_unary helper because the output dtype
        is operator-level metadata, not a Relax op parameter.
        """
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        in_expr = self.get_tensor_expr(input_tensors[0])
        output_dtype = self.get_tensor_type_str(output_tensors[0].tensor.Type())
        return self.bb.normalize(relax.op.astype(in_expr, output_dtype))

    def _convert_stablehlo_clamp(self, op):
        """Convert STABLEHLO_CLAMP to Relax.

        StableHLO clamp(min, operand, max) → R.minimum(R.maximum(operand, min), max).
        """
        # NOTE: R.clip is not used here because it only accepts scalar Expr
        # min/max, not tensor inputs.
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        assert len(self.get_output_tensors(op)) == 1

        min_expr = self.get_tensor_expr(input_tensors[0])
        operand_expr = self.get_tensor_expr(input_tensors[1])
        max_expr = self.get_tensor_expr(input_tensors[2])

        clamped = self.bb.normalize(relax.op.maximum(operand_expr, min_expr))
        return self.bb.normalize(relax.op.minimum(clamped, max_expr))

    def _convert_stablehlo_concatenate(self, op):
        """Convert STABLEHLO_CONCATENATE to Relax."""
        from tflite.StablehloConcatenateOptions import StablehloConcatenateOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors length should be >= 1"
        assert len(self.get_output_tensors(op)) == 1

        opts = self._get_stablehlo_options(op, StablehloConcatenateOptions)
        dim = opts.Dimension()

        in_exprs = [self.get_tensor_expr(t) for t in input_tensors]
        return self.bb.normalize(relax.op.concat(in_exprs, axis=dim))

    def _convert_stablehlo_broadcast_in_dim(self, op):
        """Convert STABLEHLO_BROADCAST_IN_DIM to Relax."""
        from tflite.StablehloBroadcastInDimOptions import StablehloBroadcastInDimOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1

        opts = self._get_stablehlo_options(op, StablehloBroadcastInDimOptions)
        broadcast_dims = [int(d) for d in opts.BroadcastDimensionsAsNumpy()]

        in_expr = self.get_tensor_expr(input_tensors[0])
        input_shape = [int(d) for d in self.get_tensor_shape(input_tensors[0])]
        output_shape = [int(d) for d in self.get_tensor_shape(output_tensors[0])]

        # Map input dims to output dims via broadcast_dims, filling
        # unmapped positions with 1 so broadcast_to covers them.
        intermediate_shape = [1] * len(output_shape)
        for i, d in enumerate(broadcast_dims):
            intermediate_shape[d] = input_shape[i]

        reshaped = self.bb.normalize(relax.op.reshape(in_expr, intermediate_shape))
        return self.bb.normalize(relax.op.broadcast_to(reshaped, output_shape))

    def _convert_stablehlo_reshape(self, op):
        """Convert STABLEHLO_RESHAPE to Relax."""
        input_tensors = self.get_input_tensors(op)
        if len(input_tensors) != 1:
            raise ValueError(
                f"STABLEHLO_RESHAPE expects exactly 1 input tensor, but got {len(input_tensors)}"
            )
        output_tensors = self.get_output_tensors(op)
        if len(output_tensors) != 1:
            raise ValueError(
                f"STABLEHLO_RESHAPE expects exactly 1 output tensor, but got {len(output_tensors)}"
            )

        in_expr = self.get_tensor_expr(input_tensors[0])
        output_shape = self._get_relax_tensor_shape(output_tensors[0])
        return self.bb.normalize(relax.op.reshape(in_expr, output_shape))

    def _convert_stablehlo_slice(self, op):
        """Convert STABLEHLO_SLICE to Relax."""
        from tflite.StablehloSliceOptions import StablehloSliceOptions

        input_tensors = self.get_input_tensors(op)
        if len(input_tensors) != 1:
            raise ValueError(
                f"STABLEHLO_SLICE expects exactly 1 input tensor, but got {len(input_tensors)}"
            )
        output_tensors = self.get_output_tensors(op)
        if len(output_tensors) != 1:
            raise ValueError(
                f"STABLEHLO_SLICE expects exactly 1 output tensor, but got {len(output_tensors)}"
            )

        opts = self._get_stablehlo_options(op, StablehloSliceOptions)
        begin = [int(d) for d in opts.StartIndicesAsNumpy()]
        end = [int(d) for d in opts.LimitIndicesAsNumpy()]
        strides = [int(d) for d in opts.StridesAsNumpy()]
        axes = list(range(len(begin)))

        in_expr = self.get_tensor_expr(input_tensors[0])
        return self.bb.normalize(
            relax.op.strided_slice(in_expr, axes=axes, begin=begin, end=end, strides=strides)
        )

    def _convert_stablehlo_transpose(self, op):
        """Convert STABLEHLO_TRANSPOSE to Relax."""
        from tflite.StablehloTransposeOptions import StablehloTransposeOptions

        input_tensors = self.get_input_tensors(op)
        if len(input_tensors) != 1:
            raise ValueError(
                f"STABLEHLO_TRANSPOSE expects exactly 1 input tensor, but got {len(input_tensors)}"
            )
        output_tensors = self.get_output_tensors(op)
        if len(output_tensors) != 1:
            raise ValueError(
                "STABLEHLO_TRANSPOSE expects exactly 1 output tensor, "
                f"but got {len(output_tensors)}"
            )

        opts = self._get_stablehlo_options(op, StablehloTransposeOptions)
        permutation = [int(d) for d in opts.PermutationAsNumpy()]
        if self._is_tflite_complex64_type(input_tensors[0].tensor.Type()):
            permutation.append(len(permutation))

        in_expr = self.get_tensor_expr(input_tensors[0])
        return self.bb.normalize(relax.op.permute_dims(in_expr, axes=permutation))

    def _convert_stablehlo_iota(self, op):
        """Convert STABLEHLO_IOTA to Relax (arange + broadcast)."""
        from tflite.StablehloIotaOptions import StablehloIotaOptions

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1

        opts = self._get_stablehlo_options(op, StablehloIotaOptions)
        iota_dim = opts.IotaDimension()

        output_tensor = output_tensors[0]
        output_shape = [int(d) for d in self.get_tensor_shape(output_tensor)]
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())

        # arange along the iota dimension
        size = output_shape[iota_dim]
        arange_1d = self.bb.normalize(relax.op.arange(0, size, 1, output_dtype))

        # reshape to [1, ..., size, ..., 1]
        broadcast_shape = [1] * len(output_shape)
        broadcast_shape[iota_dim] = size
        arange_reshaped = self.bb.normalize(relax.op.reshape(arange_1d, broadcast_shape))

        # broadcast to full output shape
        return self.bb.normalize(relax.op.broadcast_to(arange_reshaped, output_shape))

    def _convert_stablehlo_compare(self, op):
        """Convert STABLEHLO_COMPARE to Relax binary comparison ops."""
        from tflite.StablehloCompareOptions import StablehloCompareOptions
        from tflite.StablehloComparisonDirection import StablehloComparisonDirection

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2
        assert len(self.get_output_tensors(op)) == 1

        from tflite.StablehloComparisonType import StablehloComparisonType

        opts = self._get_stablehlo_options(op, StablehloCompareOptions)
        direction = opts.ComparisonDirection()
        compare_type = opts.CompareType()

        # TOTALORDER compare is not expressible via Relax comparison ops.
        if compare_type == StablehloComparisonType.STABLEHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER:
            raise NotImplementedError(
                "STABLEHLO_COMPARE with TOTALORDER comparison type is not supported"
            )

        _DIR = StablehloComparisonDirection
        direction_map = {
            _DIR.STABLEHLO_COMPARISON_DIRECTION_EQ: relax.op.equal,
            _DIR.STABLEHLO_COMPARISON_DIRECTION_NE: relax.op.not_equal,
            _DIR.STABLEHLO_COMPARISON_DIRECTION_GE: relax.op.greater_equal,
            _DIR.STABLEHLO_COMPARISON_DIRECTION_GT: relax.op.greater,
            _DIR.STABLEHLO_COMPARISON_DIRECTION_LE: relax.op.less_equal,
            _DIR.STABLEHLO_COMPARISON_DIRECTION_LT: relax.op.less,
        }
        relax_fn = direction_map.get(direction)
        if relax_fn is None:
            raise NotImplementedError(f"Unsupported StableHLO comparison direction: {direction}")

        lhs = self.get_tensor_expr(input_tensors[0])
        rhs = self.get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax_fn(lhs, rhs))

    def _convert_stablehlo_pad(self, op):
        """Convert STABLEHLO_PAD to Relax (nn.pad).

        Maps edge padding to R.nn.pad with constant mode.  Interior padding
        (dilation) is not supported in the first version.
        """
        from tflite.StablehloPadOptions import StablehloPadOptions

        input_tensors = self.get_input_tensors(op)
        # operand + padding_value
        assert len(input_tensors) == 2, "input tensors length should be 2"
        assert len(self.get_output_tensors(op)) == 1

        opts = self._get_stablehlo_options(op, StablehloPadOptions)
        edge_low = [int(d) for d in opts.EdgePaddingLowAsNumpy()]
        edge_high = [int(d) for d in opts.EdgePaddingHighAsNumpy()]
        interior = [int(d) for d in opts.InteriorPaddingAsNumpy()]

        if any(d != 0 for d in interior):
            raise NotImplementedError(
                "STABLEHLO_PAD with interior (dilation) padding is not supported"
            )
        if any(d < 0 for d in edge_low) or any(d < 0 for d in edge_high):
            raise NotImplementedError(
                "STABLEHLO_PAD with negative edge padding (crop) is not supported"
            )

        operand = self.get_tensor_expr(input_tensors[0])

        # R.nn.pad only supports a static Python float pad_value.
        pad_value_tensor = input_tensors[1]
        if not self.has_expr(pad_value_tensor.tensor_idx):
            pad_val = float(self.get_tensor_value(pad_value_tensor))
        else:
            raise NotImplementedError("STABLEHLO_PAD with dynamic padding value is not supported")

        # R.nn.pad with flat pad_width: [lo0, hi0, lo1, hi1, ...]
        pad_width = []
        for lo, hi in zip(edge_low, edge_high):
            pad_width.extend([lo, hi])

        return self.bb.normalize(relax.op.nn.pad(operand, pad_width=pad_width, pad_value=pad_val))

    def _convert_stablehlo_dynamic_slice(self, op):
        """Convert STABLEHLO_DYNAMIC_SLICE to Relax (dynamic_strided_slice).

        Start indices are assumed to be constant (non-dynamic) values stored
        in the flatbuffer.  Truly dynamic (runtime) start indices require
        Relax arithmetic to compute begin/end from scalar inputs and are not
        yet supported.
        """
        from tflite.StablehloDynamicSliceOptions import StablehloDynamicSliceOptions

        input_tensors = self.get_input_tensors(op)
        # operand + N start-index scalars
        assert len(input_tensors) >= 2
        ndim = len(input_tensors) - 1
        assert len(self.get_output_tensors(op)) == 1

        opts = self._get_stablehlo_options(op, StablehloDynamicSliceOptions)
        slice_sizes = [int(d) for d in opts.SliceSizesAsNumpy()]
        assert len(slice_sizes) == ndim

        operand = self.get_tensor_expr(input_tensors[0])

        # Build constant 1D tensors for begin, end, strides
        # (assumes start values are constant in the flatbuffer)
        # TODO: support dynamic start indices via Relax arithmetic
        if any(self.has_expr(t.tensor_idx) for t in input_tensors[1:]):
            raise NotImplementedError(
                "STABLEHLO_DYNAMIC_SLICE with dynamic start indices is not supported"
            )
        start_vals = [int(self.get_tensor_value(t)) for t in input_tensors[1:]]
        operand_shape = [int(d) for d in self.get_tensor_shape(input_tensors[0])]
        for start, size, dim in zip(start_vals, slice_sizes, operand_shape):
            if start < 0 or start + size > dim:
                raise NotImplementedError(
                    "STABLEHLO_DYNAMIC_SLICE with out-of-bounds start indices is not supported"
                )
        end_vals = [s + sz for s, sz in zip(start_vals, slice_sizes)]
        stride_vals = [1] * ndim

        def _const_1d(values, dtype="int64"):
            arr = np.array(values, dtype=dtype)
            return self.bb.normalize(relax.const(arr, dtype=dtype))

        begin = _const_1d(start_vals)
        end = _const_1d(end_vals)
        strides = _const_1d(stride_vals)

        return self.bb.normalize(relax.op.dynamic_strided_slice(operand, begin, end, strides))

    def _convert_stablehlo_dynamic_update_slice(self, op):
        """Convert STABLEHLO_DYNAMIC_UPDATE_SLICE to Relax.

        Lowers to ``relax.op.scatter_nd``. Constant start indices build the index
        grid at compile time; runtime (dynamic) start indices build it in-graph
        with ``arange`` + broadcast, clamping each start to
        ``[0, operand_dim - update_dim]`` per StableHLO semantics.
        """
        input_tensors = self.get_input_tensors(op)
        # operand + update + N start-index scalars
        assert len(input_tensors) >= 3, "input tensors length should be >= 3"
        assert len(self.get_output_tensors(op)) == 1

        operand_tensor = input_tensors[0]
        update_tensor = input_tensors[1]
        start_tensors = input_tensors[2:]

        op_name = "STABLEHLO_DYNAMIC_UPDATE_SLICE"
        operand_shape = self._get_static_tensor_shape(operand_tensor, op_name)
        update_shape = self._get_static_tensor_shape(update_tensor, op_name)
        rank = len(operand_shape)
        if len(update_shape) != rank or len(start_tensors) != rank:
            raise NotImplementedError(
                "STABLEHLO_DYNAMIC_UPDATE_SLICE requires operand, update, "
                "and start-index ranks to match"
            )
        for dim, size in zip(operand_shape, update_shape):
            if size > dim:
                raise NotImplementedError(
                    "STABLEHLO_DYNAMIC_UPDATE_SLICE update shape must be smaller than "
                    "or equal to operand shape for all dimensions"
                )

        operand = self.get_tensor_expr(operand_tensor)
        update = self.get_tensor_expr(update_tensor)

        if any(self.has_expr(t.tensor_idx) for t in start_tensors):
            indices = self._build_dynamic_update_slice_indices(
                start_tensors, operand_shape, update_shape, rank
            )
            return self.bb.normalize(relax.op.scatter_nd(operand, indices, update, "update"))

        start_vals = [int(np.asarray(self.get_tensor_value(t)).item()) for t in start_tensors]
        for start, size, dim in zip(start_vals, update_shape, operand_shape):
            if start < 0 or start + size > dim:
                raise NotImplementedError(
                    "STABLEHLO_DYNAMIC_UPDATE_SLICE with out-of-bounds update "
                    "indices is not supported"
                )

        update_indices = np.indices(update_shape, dtype=np.int64)
        for axis, start in enumerate(start_vals):
            update_indices[axis] += start
        update_indices = np.moveaxis(update_indices, 0, -1)

        indices = self.bb.normalize(relax.const(update_indices, dtype="int64"))
        return self.bb.normalize(relax.op.scatter_nd(operand, indices, update, "update"))

    def _build_dynamic_update_slice_indices(self, start_tensors, operand_shape, update_shape, rank):
        """Build the scatter_nd index grid for runtime DYNAMIC_UPDATE_SLICE starts.

        Returns an int64 tensor of shape ``(*update_shape, rank)`` where axis ``a``
        holds ``arange(update_shape[a]) + clamp(start[a], 0, operand_dim - update_dim)``,
        broadcast over the other axes (StableHLO clamps out-of-range starts).
        """
        axis_indices = []
        for axis in range(rank):
            start_expr = self.bb.normalize(
                relax.op.astype(self.get_tensor_expr(start_tensors[axis]), "int64")
            )
            max_start = operand_shape[axis] - update_shape[axis]
            start_expr = relax.op.maximum(start_expr, relax.const(0, "int64"))
            start_expr = relax.op.minimum(start_expr, relax.const(max_start, "int64"))

            base = relax.op.arange(0, update_shape[axis], 1, "int64")
            idx = relax.op.add(base, start_expr)

            broadcast_shape = [1] * rank
            broadcast_shape[axis] = update_shape[axis]
            idx = self.bb.normalize(relax.op.reshape(idx, broadcast_shape))
            idx = self.bb.normalize(relax.op.broadcast_to(idx, update_shape))
            idx = self.bb.normalize(relax.op.expand_dims(idx, axis=-1))
            axis_indices.append(idx)

        return self.bb.normalize(relax.op.concat(axis_indices, axis=-1))

    def _convert_stablehlo_dot_general(self, op):
        """Convert the canonical 2D STABLEHLO_DOT_GENERAL subset to Relax matmul."""
        from tflite.StablehloDotGeneralOptions import StablehloDotGeneralOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        assert len(self.get_output_tensors(op)) == 1

        opts = self._get_stablehlo_options(op, StablehloDotGeneralOptions)
        lhs_batch_dims = self._get_stablehlo_i64_vector(opts.LhsBatchingDimensionsAsNumpy(), [])
        rhs_batch_dims = self._get_stablehlo_i64_vector(opts.RhsBatchingDimensionsAsNumpy(), [])
        lhs_contract_dims = self._get_stablehlo_i64_vector(
            opts.LhsContractingDimensionsAsNumpy(), []
        )
        rhs_contract_dims = self._get_stablehlo_i64_vector(
            opts.RhsContractingDimensionsAsNumpy(), []
        )

        lhs_shape = self._get_static_tensor_shape(input_tensors[0], "STABLEHLO_DOT_GENERAL")
        rhs_shape = self._get_static_tensor_shape(input_tensors[1], "STABLEHLO_DOT_GENERAL")
        if len(lhs_shape) != 2 or len(rhs_shape) != 2:
            raise NotImplementedError("STABLEHLO_DOT_GENERAL only supports 2D matmul")
        if lhs_batch_dims or rhs_batch_dims:
            raise NotImplementedError(
                "STABLEHLO_DOT_GENERAL with batching dimensions is not supported"
            )
        if lhs_contract_dims != [1] or rhs_contract_dims != [0]:
            raise NotImplementedError(
                "STABLEHLO_DOT_GENERAL only supports canonical contracting dimensions"
            )

        lhs = self.get_tensor_expr(input_tensors[0])
        rhs = self.get_tensor_expr(input_tensors[1])
        return self.bb.normalize(relax.op.matmul(lhs, rhs))

    def _convert_stablehlo_convolution(self, op):
        """Convert the canonical 2D NHWC/HWIO STABLEHLO_CONVOLUTION subset."""
        from tflite.StablehloConvolutionOptions import StablehloConvolutionOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        assert len(self.get_output_tensors(op)) == 1

        opts = self._get_stablehlo_options(op, StablehloConvolutionOptions)
        input_spatial_dims = self._get_stablehlo_i64_vector(
            opts.InputSpatialDimensionsAsNumpy(), []
        )
        kernel_spatial_dims = self._get_stablehlo_i64_vector(
            opts.KernelSpatialDimensionsAsNumpy(), []
        )
        output_spatial_dims = self._get_stablehlo_i64_vector(
            opts.OutputSpatialDimensionsAsNumpy(), []
        )
        if input_spatial_dims != [1, 2]:
            raise NotImplementedError("STABLEHLO_CONVOLUTION only supports NHWC input layout")
        if kernel_spatial_dims != [0, 1]:
            raise NotImplementedError("STABLEHLO_CONVOLUTION only supports HWIO kernel layout")
        if output_spatial_dims != [1, 2]:
            raise NotImplementedError("STABLEHLO_CONVOLUTION only supports NHWC output layout")

        if (
            int(opts.InputBatchDimension()) != 0
            or int(opts.InputFeatureDimension()) != 3
            or int(opts.KernelInputFeatureDimension()) != 2
            or int(opts.KernelOutputFeatureDimension()) != 3
            or int(opts.OutputBatchDimension()) != 0
            or int(opts.OutputFeatureDimension()) != 3
        ):
            raise NotImplementedError(
                "STABLEHLO_CONVOLUTION only supports canonical NHWC/HWIO dimension numbers"
            )
        if int(opts.BatchGroupCount()) != 1:
            raise NotImplementedError(
                "STABLEHLO_CONVOLUTION with batch_group_count > 1 is not supported"
            )
        if int(opts.FeatureGroupCount()) != 1:
            raise NotImplementedError(
                "STABLEHLO_CONVOLUTION with feature_group_count > 1 is not supported"
            )

        data_shape = self._get_static_tensor_shape(input_tensors[0], "STABLEHLO_CONVOLUTION")
        kernel_shape = self._get_static_tensor_shape(input_tensors[1], "STABLEHLO_CONVOLUTION")
        if len(data_shape) != 4 or len(kernel_shape) != 4:
            raise NotImplementedError("STABLEHLO_CONVOLUTION only supports 2D convolution")
        if data_shape[3] != kernel_shape[2]:
            raise NotImplementedError(
                "STABLEHLO_CONVOLUTION input channels must match kernel input channels"
            )

        window_strides = self._get_stablehlo_i64_vector(opts.WindowStridesAsNumpy(), [1, 1])
        padding = self._get_stablehlo_i64_vector(opts.PaddingAsNumpy(), [0, 0, 0, 0])
        lhs_dilation = self._get_stablehlo_i64_vector(opts.LhsDilationAsNumpy(), [1, 1])
        rhs_dilation = self._get_stablehlo_i64_vector(opts.RhsDilationAsNumpy(), [1, 1])
        window_reversal = opts.WindowReversalAsNumpy()
        window_reversal = (
            [False, False] if window_reversal is None else [bool(v) for v in window_reversal]
        )

        if len(window_strides) != 2 or len(rhs_dilation) != 2:
            raise NotImplementedError("STABLEHLO_CONVOLUTION only supports two spatial dimensions")
        if lhs_dilation != [1, 1]:
            raise NotImplementedError("STABLEHLO_CONVOLUTION with lhs dilation is not supported")
        if any(window_reversal):
            raise NotImplementedError("STABLEHLO_CONVOLUTION with window reversal is not supported")
        if len(padding) != 4:
            raise NotImplementedError("STABLEHLO_CONVOLUTION only supports 2D low/high padding")

        # StableHLO stores padding as [low_h, high_h, low_w, high_w].
        relax_padding = [padding[0], padding[2], padding[1], padding[3]]
        data = self.get_tensor_expr(input_tensors[0])
        kernel = self.get_tensor_expr(input_tensors[1])
        self._ensure_stablehlo_float_dtype(data, "STABLEHLO_CONVOLUTION")
        self._ensure_stablehlo_float_dtype(kernel, "STABLEHLO_CONVOLUTION")
        return self.bb.normalize(
            relax.op.nn.conv2d(
                data,
                kernel,
                strides=window_strides,
                padding=relax_padding,
                dilation=rhs_dilation,
                data_layout="NHWC",
                kernel_layout="HWIO",
            )
        )

    def _convert_stablehlo_gather(self, op):
        """Convert STABLEHLO_GATHER to Relax (take-equivalent subset only).

        Only handles gather patterns equivalent to R.take along a single axis.
        Multi-dimensional gathers, index_vector_dim != rank(indices)-1, and
        non-trivial slice_sizes raise NotImplementedError.
        """
        from tflite.StablehloGatherOptions import StablehloGatherOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1

        opts = self._get_stablehlo_options(op, StablehloGatherOptions)
        offset_dims = [int(d) for d in opts.OffsetDimsAsNumpy()]
        collapsed_slice_dims = [int(d) for d in opts.CollapsedSliceDimsAsNumpy()]
        start_index_map = [int(d) for d in opts.StartIndexMapAsNumpy()]
        slice_sizes = [int(d) for d in opts.SliceSizesAsNumpy()]
        index_vector_dim = int(opts.IndexVectorDim())

        data_tensor, indices_tensor = input_tensors
        data_shape = [int(d) for d in self.get_tensor_shape(data_tensor)]
        indices_shape = [int(d) for d in self.get_tensor_shape(indices_tensor)]
        output_shape = [int(d) for d in self.get_tensor_shape(output_tensors[0])]

        if len(start_index_map) != 1:
            raise NotImplementedError("STABLEHLO_GATHER only supports one start_index_map entry")
        axis = start_index_map[0]
        if axis < 0 or axis >= len(data_shape):
            raise NotImplementedError(f"Unsupported STABLEHLO_GATHER axis: {axis}")
        if collapsed_slice_dims != [axis]:
            raise NotImplementedError(
                "STABLEHLO_GATHER only supports collapsed_slice_dims matching the gather axis"
            )
        if len(slice_sizes) != len(data_shape):
            raise NotImplementedError("STABLEHLO_GATHER slice_sizes must match operand rank")
        for i, (size, dim) in enumerate(zip(slice_sizes, data_shape)):
            expected = 1 if i == axis else dim
            if size != expected:
                raise NotImplementedError(
                    "STABLEHLO_GATHER only supports take-equivalent slice_sizes"
                )
        if index_vector_dim != len(indices_shape) - 1:
            raise NotImplementedError("STABLEHLO_GATHER only supports trailing index_vector_dim")
        if not indices_shape or indices_shape[index_vector_dim] != 1:
            raise NotImplementedError("STABLEHLO_GATHER only supports index vector size 1")

        indices_batch_shape = indices_shape[:index_vector_dim]
        expected_offset_dims = list(range(axis)) + list(
            range(axis + len(indices_batch_shape), len(data_shape) + len(indices_batch_shape) - 1)
        )
        if offset_dims != expected_offset_dims:
            raise NotImplementedError(
                "STABLEHLO_GATHER offset_dims do not match Relax take output layout"
            )

        expected_output_shape = data_shape[:axis] + indices_batch_shape + data_shape[axis + 1 :]
        if output_shape != expected_output_shape:
            raise NotImplementedError(
                "STABLEHLO_GATHER output shape does not match Relax take semantics"
            )

        data = self.get_tensor_expr(data_tensor)
        indices = self.get_tensor_expr(indices_tensor)
        indices = self.bb.normalize(relax.op.reshape(indices, indices_batch_shape))
        return self.bb.normalize(relax.op.take(data, indices, axis=axis, mode="fast"))

    def convert_elu(self, op):
        """Convert TFLite ELU"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        exp_type = self.get_tensor_type_str(input_tensor.tensor.Type())
        out = relax.const(-1.0, exp_type) * relax.op.nn.relu(
            relax.const(1.0, exp_type) - relax.op.exp(in_expr)
        ) + relax.op.nn.relu(in_expr)

        return out

    def convert_gelu(self, op):
        """Convert TFLite GELU"""
        if self.is_quantized(op):
            raise NotImplementedError(
                "The TFLite to Relax converter does not support quantized GELU operator yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        in_type = self.get_tensor_type_str(input_tensor.tensor.Type())

        return in_expr * (
            relax.const(0.5, dtype=in_type)
            + relax.op.erf(in_expr * relax.const(0.5**0.5, dtype=in_type))
            * relax.const(0.5, dtype=in_type)
        )

    def convert_square(self, op):
        """Convert TFLite SQUARE"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        exp_type = self.get_tensor_type_str(output_tensor.tensor.Type())
        out = relax.op.power(in_expr, relax.const(2, exp_type))

        return out

    def _convert_elemwise(self, op, relax_op, comparison_op=False):
        """Generic method to Convert TFLite elemwise"""

        from tflite.AddOptions import AddOptions
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.DivOptions import DivOptions
        from tflite.MulOptions import MulOptions
        from tflite.SubOptions import SubOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        rhs_tensor = input_tensors[1]
        lhs_expr = self.get_tensor_expr(lhs_tensor)
        rhs_expr = self.get_tensor_expr(rhs_tensor)
        input_is_quantized = lhs_tensor.qnn_params is not None or rhs_tensor.qnn_params is not None

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_is_quantized:
            if lhs_tensor.qnn_params:
                lhs_expr = self.dequantize(lhs_expr, lhs_tensor)
            if rhs_tensor.qnn_params:
                rhs_expr = self.dequantize(rhs_expr, rhs_tensor)

        out = relax_op(lhs_expr, rhs_expr)

        # Options (fused_activation_function)
        options = None
        if op.BuiltinOptionsType() == BuiltinOptions.AddOptions:
            options = AddOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.SubOptions:
            options = SubOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.MulOptions:
            options = MulOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.DivOptions:
            options = DivOptions()

        if options is not None:
            op_options = op.BuiltinOptions()
            options.Init(op_options.Bytes, op_options.Pos)
            fused_activation_fn = options.FusedActivationFunction()

            out = self.convert_fused_activation_function(out, fused_activation_fn)

        if input_is_quantized and not comparison_op:
            if not output_tensor.qnn_params:
                raise AttributeError(
                    "Quantized TFLite elemwise operator output must have quantization parameters"
                )
            out = self.quantize(out, output_tensor)
        return out

    def convert_add_n(self, op):
        """Convert TFLite ADD_N"""
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        input_tensors = self.get_input_tensors(op)
        assert not input_tensors[0].qnn_params, "TFLite does not support quantized ADD_N."
        lhs_expr = self.get_tensor_expr(input_tensors[0])
        for rhs_tensor in input_tensors[1:]:
            assert not rhs_tensor.qnn_params, "TFLite does not support quantized ADD_N"
            rhs_expr = self.get_tensor_expr(rhs_tensor)
            lhs_expr = relax.op.add(lhs_expr, rhs_expr)
        return lhs_expr

    def convert_cumsum(self, op):
        """Convert TFLite CUMSUM"""
        if self.is_quantized(op):
            raise NotImplementedError(
                "The TFLite to Relax converter does not support quantized CUMSUM operator yet."
            )

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.CumsumOptions import CumsumOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        input_expr = self.get_tensor_expr(input_tensors[0])

        if self.has_expr(input_tensors[1].tensor_idx):
            raise NotImplementedError(
                "The TFLite to Relax converter does not support dynamic axis for CUMSUM yet."
            )
        axis = self.get_tensor_value(input_tensors[1])
        if isinstance(axis, np.ndarray):
            assert axis.size == 1, "only one value is expected."
            axis = int(axis.flat[0])

        assert op.BuiltinOptionsType() == BuiltinOptions.CumsumOptions
        op_options = op.BuiltinOptions()
        cumsum_options = CumsumOptions()
        cumsum_options.Init(op_options.Bytes, op_options.Pos)
        exclusive = cumsum_options.Exclusive()
        if cumsum_options.Reverse():
            raise NotImplementedError(
                "The TFLite to Relax converter does not support reverse CUMSUM operator yet."
            )

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        out_dtype = self.get_tensor_type_str(output_tensors[0].tensor.Type())

        out = relax.op.cumsum(input_expr, axis, out_dtype, exclusive)

        return out

    def convert_squared_difference(self, op):
        """Convert TFLite SQUARED DIFFERENCE"""
        # Check if the input tensor is quantized, call QNN op
        # (https://github.com/tensorflow/tflite-micro/blob/bc35c3ed9c7ab93b3a13b46fce936f854bcfce2c
        # /tensorflow/lite/micro/kernels/squared_difference.cc#L157)
        if self.is_quantized(op):
            input_tensors = self.get_input_tensors(op)
            output_tensors = self.get_output_tensors(op)
            lhs_expr = self.get_tensor_expr(input_tensors[0])
            rhs_expr = self.get_tensor_expr(input_tensors[1])
            assert len(input_tensors) == 2, "input tensors length should be 2"
            assert len(output_tensors) == 1, "output tensors length should be 1"
            lhs_expr_f32 = self.dequantize(lhs_expr, input_tensors[0])
            rhs_expr_f32 = self.dequantize(rhs_expr, input_tensors[1])
            out_f32 = relax.op.subtract(lhs_expr_f32, rhs_expr_f32)
            return self.quantize(out_f32 * out_f32, output_tensors[0])

        difference = self._convert_elemwise(op, _op.subtract)
        # _convert_elemwise has guaranteed only have one output tensor
        exp_type = self.get_tensor_type_str(self.get_output_tensors(op)[0].tensor.Type())
        out = relax.op.power(difference, relax.const(2, exp_type))
        return out

    def _convert_logical_binary(self, relax_op, op):
        """Generic method to convert logical binary ops"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        lhs_expr = self.get_tensor_expr(lhs_tensor)
        rhs_tensor = input_tensors[1]
        rhs_expr = self.get_tensor_expr(rhs_tensor)
        out = relax_op(lhs_expr, rhs_expr)

        return out

    def convert_logical_not(self, op):
        """Convert tflite LOGICAL_NOT"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        data = self.get_expr(input_tensors[0].tensor_idx)
        out = relax.op.logical_not(data)

        return out

    def convert_gather(self, op):
        """Method to Convert TFLite GATHER operator"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.GatherOptions import GatherOptions
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        data = self.get_tensor_expr(input_tensors[0])
        indices = input_tensors[1]
        indices_type = indices.tensor.Type()
        assert indices_type in (TensorType.INT32, TensorType.INT64)

        assert op.BuiltinOptionsType() == BuiltinOptions.GatherOptions
        op_options = op.BuiltinOptions()
        gather_options = GatherOptions()
        gather_options.Init(op_options.Bytes, op_options.Pos)
        axis = gather_options.Axis()

        # Check the indices are with in bounds.
        data_shape = to_int_list(self.get_tensor_shape(input_tensors[0]))
        data_dim = len(data_shape)

        axis = data_dim + axis if axis < 0 else axis
        assert axis >= 0, "Axis out of bounds"
        assert axis < data_dim, "Axis out of bounds"

        if self.has_expr(indices.tensor_idx):
            indices_expr = relax.op.astype(self.get_expr(indices.tensor_idx), "int32")
        else:
            indices_val = self.get_tensor_value(indices)
            indices_expr = self.exp_tab.new_const(
                indices_val,
                dtype=self.get_tensor_type_str(indices_type),
                source_name=indices.tensor.Name(),
            )
            indices_shape = list(indices_val.shape)
            indices_len = len(indices_shape)

            out_shape = data_shape[:axis] + indices_shape[:] + data_shape[axis + 1 :]

            loopover = [range(s) for s in out_shape]
            for idx in list(itertools.product(*loopover)):
                real_indices = (
                    list(idx[:axis])
                    + [indices_val[idx[axis : axis + indices_len]]]
                    + list(idx[axis + indices_len :])
                )
                if np.any(np.subtract(data_shape, real_indices) < 0):
                    raise ValueError("TFLite out of bound indices are not supported.")

        # Use mode 'fast' since indices are already checked within bounds.
        out = relax.op.take(data, indices_expr, axis=axis, mode="fast")
        return out

    def convert_gather_nd(self, op):
        """Method to Convert TFLite GATHER_ND operator"""

        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        for t in input_tensors:
            assert not t.qnn_params, "Quantized input is not expected."

        data = self.get_tensor_expr(input_tensors[0])
        indices = self.get_tensor_expr(input_tensors[1])

        indices_type = input_tensors[1].tensor.Type()
        assert indices_type in (TensorType.INT32, TensorType.INT64)

        indices_dims = len(self._infer_shape(indices))
        indices_t = relax.op.permute_dims(indices, axes=[-1] + list(range(indices_dims - 1)))
        if indices_type == TensorType.INT32:
            # Relax gather_nd requires int64 indices.
            indices_t = relax.op.astype(indices_t, "int64")

        out = relax.op.gather_nd(data, indices_t)
        return out

    def convert_strided_slice(self, op):
        """Method to Convert TFLite STRIDED_SLICE operator.
        NOTE: Eventhough tensorflow supports begin_mask, end_mask, ellipsis_mask, new_axis_mask
        and shrink_axis_mask, tflite doesn't support these and expect these values to be zero.
        But in future, they may open up the mask implementation, so kept the implementation
        same as tensorflow.

        This op extracts a slice of size (end - begin) / stride from the given input tensor.
        Starting at the location specified by begin the slice continues by adding stride to the
        index until all dimensions are not less than end. Note that a stride can be negative,
        which causes a reverse slice.

        For slice input[val0, val1, ..., valn], begin/end/strides will be vectors of length n.

        In each mask field(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
        the ith bit will correspond to the ith val.

        If the ith bit of begin_mask is set, begin[i] is ignored and the fullest possible range
        in that dimension is used instead.

        If the ith bit of ellipsis_mask is set, as many unspecified dimensions as needed will be
        inserted between other dimensions. Only one non-zero bit is allowed in ellipsis_mask.

        If the ith bit of new_axis_mask is set, then begin, end, and stride are ignored and a
        new length 1 dimension is added at this point in the output tensor.

        If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks
        the dimensionality by 1, taking on the value at index begin[i]. end[i] and strides[i]
        are ignored in this case.
        begin and end are zero-indexed. strides entries must be non-zero.

        TVM Relax implementation of doesn't support mask, so the mask values are processed in
        this function and begin/end/strides are updated accordingly. If any mask is present, and
        since tvm doesn't support mask computation directly, the output need a final reshape.
        """

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.StridedSliceOptions import StridedSliceOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 4, "input tensors length should be 4"

        data_expr = self.get_expr(input_tensors[0].tensor_idx)

        begin = list(self.get_tensor_value(input_tensors[1]))
        end = list(self.get_tensor_value(input_tensors[2]))
        stride = list(self.get_tensor_value(input_tensors[3]))

        assert op.BuiltinOptionsType() == BuiltinOptions.StridedSliceOptions
        op_options = op.BuiltinOptions()
        options = StridedSliceOptions()
        options.Init(op_options.Bytes, op_options.Pos)
        begin_mask = options.BeginMask()
        end_mask = options.EndMask()
        ellipsis_mask = options.EllipsisMask()
        new_axis_mask = options.NewAxisMask()
        shrink_axis_mask = options.ShrinkAxisMask()

        data_shape = to_int_list(self.get_tensor_shape(input_tensors[0]))
        data_dim = len(data_shape)
        stride_dim = len(stride)

        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            # Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                # Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= 1 << stride_dim
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    # Identify the end index for applying ellipsis_mask
                    to_index = min(
                        ((data_dim - (stride_dim - index)) + 1 + new_axes_after_ellipsis), data_dim
                    )
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask & new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = data_shape[final_index] if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        if stride[index] < 0:
                            # Relax negative-step slicing excludes the end index, so an
                            # unspecified lower bound needs one extra step past index 0.
                            m_end[final_index] = -data_shape[final_index] - 1
                        else:
                            m_end[final_index] = data_shape[final_index]
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        # Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = (
                            data_shape[final_index] + begin[index]
                            if begin[index] < 0
                            else begin[index]
                        )
                        m_end[final_index] = m_begin[final_index] + 1
                        m_stride[final_index] = 1
                        fshape_indices.append(-2)
                    else:
                        fshape_indices.append(final_index)

                    final_index += 1
            return m_begin, m_end, m_stride, fshape_indices

        fshape_indices = None
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
            begin, end, stride, fshape_indices = _transform_mask(stride_dim, ellipsis_mask)

        begin = [int(i) for i in begin]
        end = [int(i) for i in end]
        stride = [int(i) for i in stride]
        axes = list(range(len(begin)))
        out = relax.op.strided_slice(data_expr, axes=axes, begin=begin, end=end, strides=stride)
        out_shape = self.bb.normalize(out).ty.shape
        if not fshape_indices:
            fshape_indices = range(len(out_shape))

        # Create final output shape.
        final_output = []
        final_len = len(fshape_indices)
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
                final_len += 1
            elif gather_index == -2:
                final_len -= 1
            else:
                final_output.append(out_shape[gather_index])

        if final_len == 0:
            return relax.op.squeeze(out, axis=tuple(range(len(fshape_indices))))

        if not final_output:
            return out
        return relax.op.reshape(out, shape=tuple(final_output))

    def convert_zeros_like(self, op):
        """Convert TFLite ZEROS LIKE"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = relax.op.zeros_like(in_expr)

        return out

    def convert_fill(self, op):
        """Convert TFLite FILL"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        dims_tensor = input_tensors[0]
        in_value_expr = self.get_expr(input_tensors[1].tensor_idx)

        out_shape, _ = self._get_shape_expr_from_tensor(dims_tensor, "fill_dim")
        out = relax.op.full(out_shape, in_value_expr)

        return out

    def _get_random_options(self, op):
        """Return the seed pair for random TFLite operators.

        The runtime imports seeded TFLite random ops with stateless semantics, so identical
        non-zero seed pairs produce identical results on every invocation. The seed pair
        (0, 0) is forwarded as the TF non-deterministic case.
        """
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.RandomOptions import RandomOptions

        if op.BuiltinOptionsType():
            assert op.BuiltinOptionsType() == BuiltinOptions.RandomOptions
            random_options = RandomOptions()
            op_options = op.BuiltinOptions()
            random_options.Init(op_options.Bytes, op_options.Pos)
            return int(random_options.Seed()), int(random_options.Seed2())
        return 0, 0

    def _check_random_output_dtype(self, op_name, output_dtype, supported_dtypes):
        if output_dtype not in supported_dtypes:
            supported = ", ".join(supported_dtypes)
            raise NotImplementedError(
                f"The TFLite {op_name} converter currently supports output dtype(s) "
                f"{supported} only, but got {output_dtype}."
            )

    def convert_random_uniform(self, op):
        """Convert TFLite RANDOM_UNIFORM using stateless seeded RNG semantics."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())
        self._check_random_output_dtype("RANDOM_UNIFORM", output_dtype, ["float32"])

        out_shape, _ = self._get_shape_expr_from_tensor(input_tensors[0], "random_uniform_dim")
        seed, seed2 = self._get_random_options(op)
        return relax.op.call_dps_packed(
            _ExternFunc("tvm.contrib.random.uniform"),
            (seed, seed2, 0.0, 1.0),
            ty_args=[relax.TensorType(out_shape, output_dtype)],
        )

    def convert_random_standard_normal(self, op):
        """Convert TFLite RANDOM_STANDARD_NORMAL using stateless seeded RNG semantics."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())
        self._check_random_output_dtype("RANDOM_STANDARD_NORMAL", output_dtype, ["float32"])

        out_shape, _ = self._get_shape_expr_from_tensor(
            input_tensors[0], "random_standard_normal_dim"
        )
        seed, seed2 = self._get_random_options(op)
        return relax.op.call_dps_packed(
            _ExternFunc("tvm.contrib.random.normal"),
            (seed, seed2, 0.0, 1.0),
            ty_args=[relax.TensorType(out_shape, output_dtype)],
        )

    def convert_multinomial(self, op):
        """Convert TFLite MULTINOMIAL using stateless seeded RNG semantics."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        logits_tensor, num_samples_tensor = input_tensors
        logits_expr = self.get_tensor_expr(logits_tensor)
        batch_size = self.get_tensor_shape(logits_tensor)[0]
        if self.has_expr(num_samples_tensor.tensor_idx):
            scalar_expr = self.get_expr(num_samples_tensor.tensor_idx)
            scalar_dtype = self.get_tensor_type_str(num_samples_tensor.tensor.Type())
            scalar_expr = self.bb.match_cast(scalar_expr, relax.TensorType([], scalar_dtype))
            scalar_expr = self.bb.normalize(relax.op.astype(scalar_expr, "int64"))
            scalar_expr = self.bb.normalize(relax.op.reshape(scalar_expr, [1]))
            shape_dataflow_var = self.bb.emit(relax.op.tensor_to_shape(scalar_expr))
            num_samples = tirx.Var("multinomial_num_samples", "int64")
            self.bb.match_cast(shape_dataflow_var, relax.ShapeType([num_samples]))
        else:
            value = self.get_tensor_value(num_samples_tensor)
            assert value.size == 1, (
                "TFLite MULTINOMIAL num_samples must be a scalar tensor, "
                f"but got {value.size} values"
            )
            num_samples = int(value.item())
        output_batch = batch_size * num_samples

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())
        self._check_random_output_dtype("MULTINOMIAL", output_dtype, ["int32", "int64"])

        seed, seed2 = self._get_random_options(op)
        uniform_sample = relax.op.call_dps_packed(
            _ExternFunc("tvm.contrib.random.uniform"),
            (seed, seed2, 0.0, 1.0),
            ty_args=[relax.TensorType([output_batch, 1], "float32")],
        )
        sample_indices = relax.op.reshape(
            relax.op.broadcast_to(
                relax.op.expand_dims(relax.op.arange(batch_size, dtype="int64"), axis=[1]),
                relax.ShapeExpr([batch_size, num_samples]),
            ),
            relax.ShapeExpr([output_batch, 1]),
        )
        sampled = relax.op.multinomial_from_uniform(
            relax.op.nn.softmax(logits_expr, axis=-1),
            uniform_sample,
            sample_indices,
            dtype=output_dtype,
        )
        return relax.op.reshape(sampled, relax.ShapeExpr([batch_size, num_samples]))

    def _convert_reduce(self, relax_op, op):
        """Generic method to Convert TFLite REDUCE operators"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.ReducerOptions import ReducerOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # input_tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # axis
        axis_value = self.get_tensor_value(input_tensors[1])
        axis = tuple(axis_value) if len(axis_value.shape) > 0 else tuple((axis_value.item(),))

        # Options - keep_dims (bool)
        # In case Options are not present, set keep_dims to False(default)
        if op.BuiltinOptionsType():
            assert op.BuiltinOptionsType() == BuiltinOptions.ReducerOptions
            reduce_options = ReducerOptions()
            op_options = op.BuiltinOptions()
            reduce_options.Init(op_options.Bytes, op_options.Pos)
            keep_dims = reduce_options.KeepDims()
        else:
            keep_dims = False

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        out = relax_op(in_expr, axis, keep_dims)

        # Finally if the reduce is quantized. Quantize the output.
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def _convert_reduce_bool(self, relax_op, op):
        """Convert TFLite REDUCE_ANY / REDUCE_ALL (bool-only ops).

        Relax max/min are undefined on bool, so cast through int8.
        """
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.ReducerOptions import ReducerOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        axis_value = self.get_tensor_value(input_tensors[1])
        axis = tuple(axis_value) if len(axis_value.shape) > 0 else tuple((axis_value.item(),))

        if op.BuiltinOptionsType():
            assert op.BuiltinOptionsType() == BuiltinOptions.ReducerOptions
            reduce_options = ReducerOptions()
            op_options = op.BuiltinOptions()
            reduce_options.Init(op_options.Bytes, op_options.Pos)
            keep_dims = reduce_options.KeepDims()
        else:
            keep_dims = False

        in_expr = relax.op.astype(in_expr, "int8")
        out = relax_op(in_expr, axis, keep_dims)
        return relax.op.astype(out, "bool")

    def _convert_arg_min_max(self, op, relax_op):
        """Generic method converting TFLite arg_min_max"""

        from tflite.ArgMaxOptions import ArgMaxOptions
        from tflite.ArgMinOptions import ArgMinOptions
        from tflite.BuiltinOptions import BuiltinOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "two input tensor arguments expected"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "one output tensor expected"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        axis_tensor = input_tensors[1]
        # In Tensorflow, `axis` argument is a Tensor, not attribute. We
        # support the case where it inputs from a scalar constant.
        axis_value = self.get_tensor_value(axis_tensor)
        assert axis_value.size == 1
        axis_value = axis_value.item()

        if op.BuiltinOptionsType() == BuiltinOptions.ArgMinOptions:
            arg_min_max_options = ArgMinOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.ArgMaxOptions:
            arg_min_max_options = ArgMaxOptions()
        op_options = op.BuiltinOptions()
        arg_min_max_options.Init(op_options.Bytes, op_options.Pos)

        # set keepdims to True since tflite 1.13 removes all dims of size 1
        # WARNING: all other versions of tflite > 1.13 need keepdims=False
        out = relax_op(in_expr, axis=axis_value, keepdims=False)

        return out

    def convert_fully_connected(self, op):
        """Convert TFLite fully connected"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.FullyConnectedOptions import FullyConnectedOptions
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (2, 3), "input tensors length should be two or three"

        input_tensor = input_tensors[0]
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        weight_tensor_shape = to_int_list(self.get_tensor_shape(weight_tensor))

        # Weight should have only 2 dimensions(TFLite convention)
        assert len(weight_tensor_shape) == 2, "Weight should be only 2-dim"

        # Input shape: [i_batch_size, ..., n_inputs]
        # Filter shape: [n_inputs, n_units]
        #
        # As we will transform Fully_Connected Input to MatMul
        # Weight require a transpose
        in_expr = self.get_tensor_expr(input_tensor)

        # TODO: Change the output shape calculation based on keep_dim option
        assert op.BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions
        op_options = op.BuiltinOptions()
        fully_connected_options = FullyConnectedOptions()
        fully_connected_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = fully_connected_options.FusedActivationFunction()
        keep_num_dims = fully_connected_options.KeepNumDims()

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (
            TensorType.INT8,
            TensorType.UINT8,
            TensorType.FLOAT32,
        )

        weight_expr = self.get_tensor_expr(weight_tensor)
        weight_expr = relax.op.permute_dims(weight_expr, [1, 0])

        if input_tensor.qnn_params:
            # Dequantize input and weight (OC remapped from axis 0 to 1)
            in_f32 = self.dequantize(in_expr, input_tensor)
            weight_axis = weight_tensor.qnn_params["axis"]
            if weight_axis != 0:
                raise AttributeError(
                    f"FC weight QuantizedDimension() must be 0 (output-channel "
                    f"axis in [OC,IC] layout), got {weight_axis}"
                )
            w_f32 = relax.op.dequantize(
                weight_expr,
                scale=weight_tensor.qnn_params["scale"],
                zero_point=weight_tensor.qnn_params["zero_point"],
                axis=1,
            )
            out = relax.op.matmul(in_f32, w_f32)
        else:
            out = relax.op.matmul(in_expr, weight_expr)

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            if bias_tensor.tensor_idx != -1:
                bias_tensor_type = bias_tensor.tensor.Type()
                # bias tensor type should be INT32 (quantization) or FLOAT32
                assert bias_tensor_type in (
                    TensorType.INT32,
                    TensorType.INT64,
                    TensorType.FLOAT32,
                )
                bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
                if self.has_expr(bias_tensor.tensor_idx):
                    bias_expr = self.get_expr(bias_tensor.tensor_idx)
                else:
                    bias_expr = self.exp_tab.new_const(
                        self.get_tensor_value(bias_tensor),
                        dtype=bias_tensor_type_str,
                        source_name=bias_tensor.tensor.Name(),
                    )
                if bias_tensor.qnn_params:
                    bias_expr = self.dequantize(bias_expr, bias_tensor)
                elif input_tensor.qnn_params and bias_tensor_type in (
                    TensorType.INT32,
                    TensorType.INT64,
                ):
                    bias_scale = relax.op.multiply(
                        input_tensor.qnn_params["scale"],
                        weight_tensor.qnn_params["scale"],
                    )
                    bias_expr = relax.op.dequantize(
                        bias_expr,
                        scale=bias_scale,
                        zero_point=relax.const(0, "int32"),
                        axis=0,
                    )
                out = relax.op.add(out, bias_expr)

        # Finally if the dense is quantized. Quantize the output.
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

            # Call activation function
            output_scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            output_zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=output_scale_val,
                zero_point=output_zero_point_val,
                dtype=output_tensor_type_str,
            )

        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        # Change the output shape calculation based on keep_dim option
        if keep_num_dims:
            input_shape = self._infer_shape(self.get_tensor_expr(input_tensor))
            output_shape = to_int_list(input_shape)[:-1] + [weight_tensor_shape[0]]
            out = relax.op.reshape(out, output_shape)

        return out

    def convert_squeeze(self, op):
        """Convert TFLite squeeze"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.SqueezeOptions import SqueezeOptions

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        assert len(output_tensors) == 1, "output tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.SqueezeOptions
        op_options = op.BuiltinOptions()
        squeeze_options = SqueezeOptions()
        squeeze_options.Init(op_options.Bytes, op_options.Pos)
        squeeze_axis = squeeze_options.SqueezeDimsAsNumpy()

        in_expr = self.get_expr(input_tensor_idx)
        out = relax.op.squeeze(in_expr, axis=tuple(squeeze_axis))

        return out

    def convert_fused_activation_function(self, in_expr, fused_activation_fn):
        """Convert TFLite fused activation function"""

        from tflite.ActivationFunctionType import ActivationFunctionType

        if fused_activation_fn == ActivationFunctionType.NONE:
            return in_expr
        if fused_activation_fn == ActivationFunctionType.RELU6:
            return relax.op.clip(in_expr, min=0, max=6)
        if fused_activation_fn == ActivationFunctionType.RELU:
            return relax.op.nn.relu(in_expr)
        if fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
            return relax.op.clip(in_expr, min=-1, max=1)
        if fused_activation_fn == ActivationFunctionType.TANH:
            return relax.op.tanh(in_expr)
        fused_activation_fn_str = self.activation_fn_type[fused_activation_fn]
        raise NotImplementedError(
            f"Fused activation {fused_activation_fn_str} is not supported yet."
        )

    def convert_conv(self, op, conv_type):
        """convolution implementation."""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.Conv2DOptions import Conv2DOptions
        from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
        from tflite.Padding import Padding
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        is_depthwise_conv = False
        if conv_type == "conv2d":
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == "depthwise":
            is_depthwise_conv = True
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
            depth_multiplier = conv_options.DepthMultiplier()
        else:
            raise NotImplementedError(f"Operator {conv_type} is not supported for frontend TFLite.")

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        _, input_h, input_w, input_c = to_int_list(self.get_tensor_shape(input_tensor))

        if is_depthwise_conv:
            # TFLite depthwise convolution kernel layout is:
            # 1 KH KW C(input_c * depth_multiplier)
            _, kernel_h, kernel_w, in_channels = to_int_list(self.get_tensor_shape(weight_tensor))
            assert in_channels == input_c * depth_multiplier
            # Relax grouped convolution expects one input channel per group in
            # HWOI when input_c > 1.  The single-channel case remains HWIO.
            depthwise_weight_shape = (
                (kernel_h, kernel_w, input_c, depth_multiplier)
                if input_c == 1
                else (kernel_h, kernel_w, in_channels, 1)
            )
        else:
            output_channels, kernel_h, kernel_w, in_channels = to_int_list(
                self.get_tensor_shape(weight_tensor)
            )

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {
            # "kernel_size": [kernel_h, kernel_w],
            "strides": [stride_h, stride_w],
            "dilation": [dilation_h, dilation_w],
            "padding": [0, 0],
            "data_layout": "NHWC",
        }

        if is_depthwise_conv:
            params["groups"] = int(input_c)
            # If number of input channels is 1, treat as normal
            # convolution.
            params["kernel_layout"] = "HWIO" if input_c == 1 else "HWOI"
        else:
            # params["channels"] = int(output_channels)
            params["kernel_layout"] = "HWIO"
            if input_c != in_channels:
                assert input_c % in_channels == 0, (
                    "Input channels is not divisible of kernel in_channels."
                )
                params["groups"] = int(input_c / in_channels)

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (
            TensorType.INT8,
            TensorType.UINT8,
            TensorType.FLOAT32,
        )
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        per_channel_depthwise = (
            is_depthwise_conv
            and input_tensor.qnn_params
            and weight_tensor.qnn_params
            and weight_tensor.tensor.Quantization().ScaleLength() > 1
        )

        in_expr = self.get_expr(input_tensor_idx)

        # TFLite converts float32 models to float16 models by introducing
        # a Dequantize op in every op that contains a float32 values.
        # (weights, biases, and constants etc. )
        # So conv op may have weight and bias as tensors instead of values.
        if self.has_expr(weight_tensor.tensor_idx):
            weight_expr = self.get_expr(weight_tensor.tensor_idx)
            if is_depthwise_conv:
                if not per_channel_depthwise:
                    weight_expr = relax.op.reshape(weight_expr, depthwise_weight_shape)
            else:
                weight_expr = relax.op.permute_dims(weight_expr, axes=(1, 2, 3, 0))
        else:
            if self.is_prefetched(weight_tensor.tensor_idx):
                weight_value = self.get_prefetched_node(weight_tensor.tensor_idx)
            else:
                weight_value = self.get_tensor_value(weight_tensor)

            # TFLite kernel layout:
            # convolution:
            # OC KH KW IC, we require KH KW IC OC (HWIO)
            # depthwise convolution:
            # 1 KH KW C(input_c * depth_multiplier), we require HWIO
            # [KH,KW,1,M] for one input channel, otherwise HWOI [KH,KW,C*M,1].
            if is_depthwise_conv:
                if not per_channel_depthwise:
                    weight_value = weight_value.reshape(depthwise_weight_shape)
            else:
                weight_value = weight_value.transpose((1, 2, 3, 0))

            weight_expr = self.exp_tab.new_const(
                weight_value, dtype=weight_tensor_type_str, source_name=weight_tensor.tensor.Name()
            )

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, dilated_kernel_h, stride_h)

            pad_left, pad_right = get_pad_value(input_w, dilated_kernel_w, stride_w)
            do_pad = not (pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0)
            if do_pad:
                params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]

        else:
            raise NotImplementedError(
                f"Padding format {padding} is not supported for operator Conv."
            )

        if input_tensor.qnn_params:
            # Dequantize input activation
            in_f32 = self.dequantize(in_expr, input_tensor)
            # Dequantize weight with per-channel axis remap.
            # TFLite weight original layout: [OC, KH, KW, IC]
            # After transpose to HWIO: [KH, KW, IC, OC]
            # QuantizedDimension() == 0 (OC in original) → axis 3 in HWIO.
            weight_axis = weight_tensor.qnn_params["axis"]
            if is_depthwise_conv:
                if per_channel_depthwise:
                    if weight_axis != 3:
                        raise AttributeError(
                            "Per-channel DepthwiseConv2D weight QuantizedDimension() must be 3 "
                            f"(the flattened output-channel axis), got {weight_axis}"
                        )
                    scale_count = weight_tensor.tensor.Quantization().ScaleLength()
                    if scale_count != in_channels:
                        raise AttributeError(
                            "Per-channel DepthwiseConv2D weight scale count must match "
                            f"input_channels * depth_multiplier ({in_channels}), got {scale_count}"
                        )
                    # Dequantize while the TFLite [1, KH, KW, C*M] channel axis is
                    # still intact.  Reshaping first would split that axis into C and M,
                    # which a single-axis dequantize cannot represent.
                    w_f32 = relax.op.dequantize(
                        weight_expr,
                        scale=weight_tensor.qnn_params["scale"],
                        zero_point=weight_tensor.qnn_params["zero_point"],
                        axis=3,
                    )
                    w_f32 = relax.op.reshape(w_f32, depthwise_weight_shape)
                else:
                    if weight_axis != 0:
                        raise AttributeError(
                            "Per-tensor DepthwiseConv2D weight QuantizedDimension() must be 0, "
                            f"got {weight_axis}"
                        )
                    w_f32 = relax.op.dequantize(
                        weight_expr,
                        scale=weight_tensor.qnn_params["scale"],
                        zero_point=weight_tensor.qnn_params["zero_point"],
                        axis=0,
                    )
            else:
                if weight_axis != 0:
                    raise AttributeError(
                        f"Conv2D weight QuantizedDimension() must be 0 (output-channel "
                        f"axis in [OC,KH,KW,IC] layout), got {weight_axis}"
                    )
                w_f32 = relax.op.dequantize(
                    weight_expr,
                    scale=weight_tensor.qnn_params["scale"],
                    zero_point=weight_tensor.qnn_params["zero_point"],
                    axis=3,
                )
            # Float convolution
            out = relax.op.nn.conv2d(in_f32, w_f32, **params)
        else:
            out = relax.op.nn.conv2d(in_expr, weight_expr, **params)

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (int8 qnn) or INT64 (int16 qnn) or FLOAT32
            assert bias_tensor_type in (
                TensorType.INT32,
                TensorType.INT64,
                TensorType.FLOAT32,
            )
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            if self.has_expr(bias_tensor.tensor_idx):
                bias_expr = self.get_expr(bias_tensor.tensor_idx)
            else:
                bias_expr = self.exp_tab.new_const(
                    self.get_tensor_value(bias_tensor),
                    dtype=bias_tensor_type_str,
                    source_name=bias_tensor.tensor.Name(),
                )
            # For quantized conv, INT32/INT64 bias must be dequantized
            # to float32 before adding to the float conv output.
            if bias_tensor.qnn_params:
                bias_expr = self.dequantize(bias_expr, bias_tensor)
            elif input_tensor.qnn_params and bias_tensor_type in (
                TensorType.INT32,
                TensorType.INT64,
            ):
                bias_expr = relax.op.dequantize(
                    bias_expr,
                    scale=relax.op.multiply(
                        input_tensor.qnn_params["scale"],
                        weight_tensor.qnn_params["scale"],
                    ),
                    zero_point=relax.const(0, "int32"),
                    axis=0,
                )
            out = relax.op.add(out, bias_expr)

        # Handle fused activation.
        if output_tensor.qnn_params:
            # Quantize the float output using the output tensor's qnn params.
            out = self.quantize(out, output_tensor)

            # Call quantized activation function
            output_scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            output_zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=output_scale_val,
                zero_point=output_zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)
        return out

    def convert_conv3d(self, op):
        """3D convolution implementation."""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.Conv3DOptions import Conv3DOptions
        from tflite.Padding import Padding
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.Conv3DOptions
        op_options = op.BuiltinOptions()
        conv3d_options = Conv3DOptions()
        conv3d_options.Init(op_options.Bytes, op_options.Pos)

        stride_d = conv3d_options.StrideD()
        stride_h = conv3d_options.StrideH()
        stride_w = conv3d_options.StrideW()
        dilation_d = conv3d_options.DilationDFactor()
        dilation_h = conv3d_options.DilationHFactor()
        dilation_w = conv3d_options.DilationWFactor()
        padding = conv3d_options.Padding()
        fused_activation_fn = conv3d_options.FusedActivationFunction()

        _, input_d, input_h, input_w, input_c = to_int_list(self.get_tensor_shape(input_tensor))
        # TFLite Conv3D kernel layout is already DHWIO:
        # KD KH KW IC OC
        kernel_d, kernel_h, kernel_w, in_channels, output_channels = to_int_list(
            self.get_tensor_shape(weight_tensor)
        )

        dilated_kernel_d = dilation_d * (kernel_d - 1) + 1
        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {
            "strides": [stride_d, stride_h, stride_w],
            "dilation": [dilation_d, dilation_h, dilation_w],
            "padding": [0, 0, 0, 0, 0, 0],
            "data_layout": "NDHWC",
        }

        params["kernel_layout"] = "DHWIO"
        if input_c != in_channels:
            assert input_c % in_channels == 0, (
                "Input channels is not divisible by kernel in_channels."
            )
            params["groups"] = int(input_c / in_channels)

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (
            TensorType.INT8,
            TensorType.UINT8,
            TensorType.FLOAT32,
        )
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        in_expr = self.get_expr(input_tensor_idx)

        # TFLite Conv3D kernel is already in DHWIO layout, no transpose needed.
        if self.has_expr(weight_tensor.tensor_idx):
            weight_expr = self.get_expr(weight_tensor.tensor_idx)
        else:
            if self.is_prefetched(weight_tensor.tensor_idx):
                weight_value = self.get_prefetched_node(weight_tensor.tensor_idx)
            else:
                weight_value = self.get_tensor_value(weight_tensor)

            weight_expr = self.exp_tab.new_const(
                weight_value, dtype=weight_tensor_type_str, source_name=weight_tensor.tensor.Name()
            )

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_front, pad_back = get_pad_value(input_d, dilated_kernel_d, stride_d)
            pad_top, pad_bottom = get_pad_value(input_h, dilated_kernel_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, dilated_kernel_w, stride_w)

            do_pad = not (
                pad_front == 0
                and pad_back == 0
                and pad_top == 0
                and pad_bottom == 0
                and pad_left == 0
                and pad_right == 0
            )
            if do_pad:
                params["padding"] = [pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right]
        else:
            raise NotImplementedError(
                f"Padding format {padding} is not supported for operator Conv3D."
            )

        if input_tensor.qnn_params:
            raise NotImplementedError(
                "Quantized Conv3D is not yet supported in the Relax frontend."
            )

        out = relax.op.nn.conv3d(in_expr, weight_expr, **params)

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            if bias_tensor.tensor_idx != -1:
                bias_tensor_type = bias_tensor.tensor.Type()
                # bias tensor type should be INT32 (int8 qnn) or INT64 (int16 qnn) or FLOAT32
                assert bias_tensor_type in (TensorType.INT32, TensorType.INT64, TensorType.FLOAT32)
                bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
                if self.has_expr(bias_tensor.tensor_idx):
                    bias_expr = self.get_expr(bias_tensor.tensor_idx)
                else:
                    bias_expr = self.exp_tab.new_const(
                        self.get_tensor_value(bias_tensor),
                        dtype=bias_tensor_type_str,
                        source_name=bias_tensor.tensor.Name(),
                    )
                out = relax.op.add(out, bias_expr)

        # Handle fused activation.
        if output_tensor.qnn_params:
            raise NotImplementedError(
                "Quantized Conv3D is not yet supported in the Relax frontend."
            )

        out = self.convert_fused_activation_function(out, fused_activation_fn)
        return out

    def convert_conv3d_transpose(self, op):
        """3D transposed convolution implementation."""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.Conv3DOptions import Conv3DOptions
        from tflite.Padding import Padding
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 3, "input tensors length should be >= 3"

        # TFLite CONV_3D_TRANSPOSE input order:
        # [0] output_shape, [1] weight, [2] data, [3] bias (optional)
        weight_tensor = input_tensors[1]
        input_tensor = input_tensors[2]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.Conv3DOptions
        op_options = op.BuiltinOptions()
        conv3d_options = Conv3DOptions()
        conv3d_options.Init(op_options.Bytes, op_options.Pos)

        stride_d = conv3d_options.StrideD()
        stride_h = conv3d_options.StrideH()
        stride_w = conv3d_options.StrideW()
        dilation_d = conv3d_options.DilationDFactor()
        dilation_h = conv3d_options.DilationHFactor()
        dilation_w = conv3d_options.DilationWFactor()
        padding = conv3d_options.Padding()
        fused_activation_fn = conv3d_options.FusedActivationFunction()

        _, input_d, input_h, input_w, input_c = to_int_list(self.get_tensor_shape(input_tensor))

        # TFLite Conv3DTranspose kernel layout is DHWOI:
        # KD KH KW OC IC
        kernel_d, kernel_h, kernel_w, output_channels, in_channels = to_int_list(
            self.get_tensor_shape(weight_tensor)
        )

        dilated_kernel_d = dilation_d * (kernel_d - 1) + 1
        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {
            "strides": [stride_d, stride_h, stride_w],
            "dilation": [dilation_d, dilation_h, dilation_w],
            "padding": [0, 0, 0, 0, 0, 0],
            "output_padding": [0, 0, 0],
            "data_layout": "NDHWC",
            "kernel_layout": "DHWOI",
        }

        if input_c != in_channels:
            assert input_c % in_channels == 0, (
                "Input channels is not divisible by kernel in_channels."
            )
            params["groups"] = int(input_c / in_channels)

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (
            TensorType.INT8,
            TensorType.UINT8,
            TensorType.FLOAT32,
        )
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        in_expr = self.get_expr(input_tensor_idx)

        # TFLite Conv3DTranspose kernel is already in DHWOI layout, no transpose needed.
        if self.has_expr(weight_tensor.tensor_idx):
            weight_expr = self.get_expr(weight_tensor.tensor_idx)
        else:
            if self.is_prefetched(weight_tensor.tensor_idx):
                weight_value = self.get_prefetched_node(weight_tensor.tensor_idx)
            else:
                weight_value = self.get_tensor_value(weight_tensor)

            weight_expr = self.exp_tab.new_const(
                weight_value, dtype=weight_tensor_type_str, source_name=weight_tensor.tensor.Name()
            )

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            # For transposed convolution with SAME padding:
            # target output_size = input_size * stride
            # total_pad = max(0, dilated_kernel - stride)
            for dim_kernel, dim_stride, label in [
                (dilated_kernel_d, stride_d, "D"),
                (dilated_kernel_h, stride_h, "H"),
                (dilated_kernel_w, stride_w, "W"),
            ]:
                total_pad = max(0, dim_kernel - dim_stride)
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
                idx = {"D": 0, "H": 1, "W": 2}[label]
                params["padding"][idx] = pad_before
                params["padding"][idx + 3] = pad_after

                # output_padding handles the case when stride > dilated_kernel
                output_pad = max(0, dim_stride - dim_kernel)
                params["output_padding"][idx] = output_pad
        else:
            raise NotImplementedError(
                f"Padding format {padding} is not supported for operator Conv3DTranspose."
            )

        if input_tensor.qnn_params:
            raise NotImplementedError(
                "Quantized Conv3DTranspose is not yet supported in the Relax frontend."
            )

        out = relax.op.nn.conv3d_transpose(in_expr, weight_expr, **params)

        # if we have bias (input_tensors[3])
        if len(input_tensors) >= 4:
            bias_tensor = input_tensors[3]
            if bias_tensor.tensor_idx != -1:
                bias_tensor_type = bias_tensor.tensor.Type()
                # bias tensor type should be INT32 (int8 qnn) or INT64 (int16 qnn) or FLOAT32
                assert bias_tensor_type in (TensorType.INT32, TensorType.INT64, TensorType.FLOAT32)
                bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
                if self.has_expr(bias_tensor.tensor_idx):
                    bias_expr = self.get_expr(bias_tensor.tensor_idx)
                else:
                    bias_expr = self.exp_tab.new_const(
                        self.get_tensor_value(bias_tensor),
                        dtype=bias_tensor_type_str,
                        source_name=bias_tensor.tensor.Name(),
                    )
                out = relax.op.add(out, bias_expr)

        # Handle fused activation.
        if output_tensor.qnn_params:
            raise NotImplementedError(
                "Quantized Conv3DTranspose is not yet supported in the Relax frontend."
            )

        out = self.convert_fused_activation_function(out, fused_activation_fn)
        return out

    def convert_split(self, op):
        """split implementation."""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.SplitOptions import SplitOptions

        input_tensors = self.get_input_tensors(op)

        assert len(input_tensors) == 2, "input tensors length should be == 2"

        axis_tensor = input_tensors[0]
        split_axis = self.get_tensor_value(axis_tensor)
        input_tensor = input_tensors[1]
        input_tensor_idx = input_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.SplitOptions
        op_options = op.BuiltinOptions()
        split_options = SplitOptions()
        split_options.Init(op_options.Bytes, op_options.Pos)
        num_splits = split_options.NumSplits()

        in_expr = self.get_expr(input_tensor_idx)
        out = relax.op.split(in_expr, num_splits, axis=int(split_axis))
        # Relay does not like a TupleWrapper of 1 element, further this
        # only shows up with tf1.13 if we use a split with num_splits==1.
        # In tf 1.14 this doesn't appear as it is automatically a reshape
        # operation.
        if isinstance(out, relax.Tuple):
            if out.size == 1:
                out = out[0]

        return out

    def convert_split_v(self, op):
        """SPLIT_V implementation."""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)

        assert len(input_tensors) == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)

        axis_tensor = input_tensors[2]
        split_axis = int(self.get_tensor_value(axis_tensor))

        size_splits_tensor = input_tensors[1]

        if self.has_expr(size_splits_tensor.tensor_idx):
            # Dynamic size_splits case: decompose into dynamic strided slices.
            size_splits_expr = self.get_expr(size_splits_tensor.tensor_idx)
            cumsum = relax.op.cumsum(size_splits_expr, axis=0, dtype="int64")
            # Pad a leading zero so that cumsum[i-1] can be read uniformly
            # via strided_slice even for i == 0.
            zero = relax.const(np.array([0], dtype="int64"), "int64")
            padded_cumsum = relax.op.concat([zero, cumsum], axis=0)
            # TFLite fixes the tuple arity in the graph, even when the split
            # sizes themselves are supplied at runtime.
            num_splits = len(output_tensors)
            rank = len(in_expr.ty.shape)

            # end_base is the full input shape; only split_axis changes per slice.
            end_base = relax.op.shape_to_tensor(relax.op.shape_of(in_expr))
            begin_base = relax.const(np.zeros((rank,), dtype="int64"), "int64")
            strides = relax.const(np.ones((rank,), dtype="int64"), "int64")
            scatter_idx = relax.const([split_axis], "int64")

            outputs = []
            for i in range(num_splits):
                start_val = relax.op.strided_slice(padded_cumsum, axes=[0], begin=[i], end=[i + 1])
                end_val = relax.op.strided_slice(
                    padded_cumsum, axes=[0], begin=[i + 1], end=[i + 2]
                )

                begin = relax.op.scatter_elements(begin_base, scatter_idx, start_val)
                end = relax.op.scatter_elements(end_base, scatter_idx, end_val)
                slice_i = relax.op.dynamic_strided_slice(in_expr, begin, end, strides)
                outputs.append(slice_i)

            out = relax.Tuple(outputs)
        else:
            # Static size_splits case
            size_splits = list(self.get_tensor_value(size_splits_tensor))
            size_splits = tuple(np.cumsum(size_splits)[:-1])
            out = relax.op.split(in_expr, size_splits, axis=split_axis)

        # Relay does not like a TupleWrapper of 1 element, further this
        # only shows up with tf1.13 if we use a split with num_splits==1.
        # In tf 1.14 this doesn't appear as it is automatically a reshape
        # operation.
        if isinstance(out, relax.Tuple) and len(out.fields) == 1:
            out = out[0]

        return out

    def convert_slice(self, op):
        """Convert TFLite SLICE"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be == 3"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        begin = list(self.get_tensor_value(input_tensors[1]))
        size = list(self.get_tensor_value(input_tensors[2]))
        # strided_slice(Relax) needs the slice's end indices, not the size
        end = size
        input_tensor_shape = to_int_list(self.get_tensor_shape(input_tensor))
        input_tensor_rank = len(input_tensor_shape)
        for i in range(input_tensor_rank):
            if size[i] == -1:
                end[i] = input_tensor_shape[i]
            else:
                end[i] += begin[i]

        # Create axes list for all dimensions being sliced
        axes = list(range(input_tensor_rank))
        begin = [int(v) for v in begin]
        end = [int(v) for v in end]
        out = relax.op.strided_slice(in_expr, axes=axes, begin=begin, end=end)
        return out

    def convert_scatter_nd(self, op):
        """Convert TFLite SCATTER_ND"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "SCATTER_ND should have 3 input tensors"
        indices = self.get_tensor_expr(input_tensors[0])
        updates = self.get_tensor_expr(input_tensors[1])
        shape_tensor = input_tensors[2]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "SCATTER_ND should have 1 output tensor"
        updates_dtype = self.get_tensor_type_str(output_tensors[0].tensor.Type())

        if self.has_expr(shape_tensor.tensor_idx):
            shape_expr = self.get_expr(shape_tensor.tensor_idx)
            shape_expr = self.bb.normalize(relax.op.astype(shape_expr, "int64"))
            shape = self.bb.emit_output(relax.op.tensor_to_shape(shape_expr))
        else:
            shape = to_int_list(self.get_tensor_value(shape_tensor))

        indices_dims = len(self._infer_shape(indices))
        indices = relax.op.permute_dims(indices, axes=[-1] + list(range(indices_dims - 1)))

        data = relax.op.zeros(shape, updates_dtype)
        return relax.op.scatter_nd(data, indices, updates, "update")

    def convert_unique(self, op):
        """Convert TFLite UNIQUE."""
        from tflite.TensorType import TensorType
        from tflite.UniqueOptions import UniqueOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "UNIQUE should have 1 input tensor"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 2, "UNIQUE should have 2 output tensors"

        unique_options = UniqueOptions()
        op_options = op.BuiltinOptions()
        unique_options.Init(op_options.Bytes, op_options.Pos)

        unique = self.bb.normalize(
            relax.op.unique(
                self.get_tensor_expr(input_tensors[0]),
                sorted=False,
                return_index=False,
                return_inverse=True,
                return_counts=False,
                axis=None,
            )
        )
        values = self.bb.emit(relax.TupleGetItem(unique, 0))
        inverse_indices = self.bb.emit(relax.TupleGetItem(unique, 1))

        idx_out_type = unique_options.IdxOutType()
        if idx_out_type == TensorType.INT32:
            inverse_indices = self.bb.emit(relax.op.astype(inverse_indices, "int32"))

        return relax.Tuple([values, inverse_indices])

    def _get_segment_scatter_base(self, output_shape, output_dtype, reduction):
        """Create the identity base tensor for scatter-based segment reductions."""
        if reduction == "add":
            return relax.op.zeros(output_shape, output_dtype)
        if reduction == "mul":
            return relax.op.full(output_shape, relax.const(1, output_dtype), output_dtype)
        if reduction == "max":
            np_dtype = np.dtype(output_dtype)
            if np.issubdtype(np_dtype, np.floating):
                identity = np.finfo(np_dtype).min
            elif np.issubdtype(np_dtype, np.integer):
                identity = np.iinfo(np_dtype).min
            else:
                raise NotImplementedError(
                    f"UNSORTED_SEGMENT_MAX does not support output dtype {output_dtype}."
                )
            return relax.op.full(output_shape, relax.const(identity, output_dtype), output_dtype)
        if reduction == "min":
            np_dtype = np.dtype(output_dtype)
            if np.issubdtype(np_dtype, np.floating):
                identity = np.finfo(np_dtype).max
            elif np.issubdtype(np_dtype, np.integer):
                identity = np.iinfo(np_dtype).max
            else:
                raise NotImplementedError(
                    f"UNSORTED_SEGMENT_MIN does not support output dtype {output_dtype}."
                )
            return relax.op.full(output_shape, relax.const(identity, output_dtype), output_dtype)

        raise ValueError(f"Unsupported segment reduction mode: {reduction}")

    def _get_segment_num_segments(self, op_name, input_tensors):
        if op_name == "SEGMENT_SUM":
            segment_ids_tensor = input_tensors[1]
            if self.has_expr(segment_ids_tensor.tensor_idx):
                raise NotImplementedError(
                    "TFLite SEGMENT_SUM with runtime segment_ids is not supported, "
                    "because TFLite does not encode a reliable output segment count."
                )
            segment_ids = self.get_tensor_value(segment_ids_tensor)
            if np.any(segment_ids < 0):
                raise NotImplementedError(
                    "TFLite SEGMENT_SUM with negative segment ids is not supported."
                )
            return int(np.max(segment_ids)) + 1 if segment_ids.size else 0

        num_segments_tensor = input_tensors[2]
        if self.has_expr(num_segments_tensor.tensor_idx):
            raise NotImplementedError(
                f"TFLite {op_name} with runtime num_segments is not supported."
            )
        num_segments_value = self.get_tensor_value(num_segments_tensor)
        assert num_segments_value.size == 1, f"{op_name} num_segments should be a scalar tensor"
        num_segments = int(num_segments_value.item())
        assert num_segments >= 0, f"{op_name} num_segments should be non-negative"
        return num_segments

    def _convert_segment_op(self, op, op_name, reduction):
        """Convert TFLite segment ops through relax.op.scatter_nd."""
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        expected_inputs = 2 if op_name == "SEGMENT_SUM" else 3
        assert len(input_tensors) == expected_inputs, (
            f"{op_name} should have {expected_inputs} input tensors"
        )

        data_tensor = input_tensors[0]
        segment_ids_tensor = input_tensors[1]
        for t in input_tensors:
            assert not t.qnn_params, "Quantized input is not expected."

        segment_ids_type = segment_ids_tensor.tensor.Type()
        assert segment_ids_type in (TensorType.INT32, TensorType.INT64)
        if op_name != "SEGMENT_SUM":
            num_segments_type = input_tensors[2].tensor.Type()
            assert num_segments_type in (TensorType.INT32, TensorType.INT64)
        if not self.has_expr(segment_ids_tensor.tensor_idx):
            segment_ids_value = self.get_tensor_value(segment_ids_tensor)
            if np.any(segment_ids_value < 0):
                raise NotImplementedError(
                    f"TFLite {op_name} with negative segment ids is not supported."
                )

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, f"{op_name} should have 1 output tensor"
        output_tensor = output_tensors[0]
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())

        data_shape = to_int_list(self.get_tensor_shape(data_tensor))
        segment_ids_shape = to_int_list(self.get_tensor_shape(segment_ids_tensor))
        segment_ids_rank = len(segment_ids_shape)
        assert data_shape[:segment_ids_rank] == segment_ids_shape, (
            f"{op_name} requires segment_ids shape to match a prefix of data shape"
        )
        num_segments = self._get_segment_num_segments(op_name, input_tensors)
        output_shape = [num_segments] + data_shape[segment_ids_rank:]

        data = self.get_tensor_expr(data_tensor)
        segment_ids = self.get_tensor_expr(segment_ids_tensor)
        indices = relax.op.expand_dims(segment_ids, axis=[segment_ids_rank])

        base = self._get_segment_scatter_base(output_shape, output_dtype, reduction)
        return relax.op.scatter_nd(base, indices, data, reduction)

    def convert_select(self, op):
        """Convert TFLite SELECT"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be == 3"
        cond = self.get_tensor_expr(input_tensors[0])
        x = self.get_tensor_expr(input_tensors[1])
        y = self.get_tensor_expr(input_tensors[2])

        out = relax.op.where(cond, x, y)

        return out

    def convert_transpose(self, op):
        """transpose implementation."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        in_expr = self.get_expr(input_tensor_idx)

        # axis
        in_axis = tuple(self.get_tensor_value(input_tensors[1]))

        if not in_axis:
            out = relax.op.permute_dims(in_expr)
        else:
            out = relax.op.permute_dims(in_expr, in_axis)

        return out

    def convert_reverse_sequence(self, op):
        """Convert TFLite REVERSE_SEQUENCE"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.ReverseSequenceOptions import ReverseSequenceOptions

        if self.is_quantized(op):
            raise NotImplementedError(
                "TFLite does not support quantized REVERSE_SEQUENCE operator yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        in_expr = self.get_tensor_expr(input_tensors[0])
        length_expr = self.get_tensor_expr(input_tensors[1])

        assert op.BuiltinOptionsType() == BuiltinOptions.ReverseSequenceOptions
        op_options = op.BuiltinOptions()
        options = ReverseSequenceOptions()
        options.Init(op_options.Bytes, op_options.Pos)
        batch_axis = options.BatchDim()
        seq_axis = options.SeqDim()

        return relax.op.reverse_sequence(in_expr, length_expr, seq_axis, batch_axis)

    def convert_bitcast(self, op):
        """Convert TFLite BITCAST"""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        assert len(output_tensors) == 1, "output tensors length should be 1"

        in_expr = self.get_tensor_expr(input_tensors[0])
        input_dtype = self.get_tensor_type_str(input_tensors[0].tensor.Type())
        output_dtype = self.get_tensor_type_str(output_tensors[0].tensor.Type())
        input_shape = to_int_list(self.get_tensor_shape(input_tensors[0]))
        output_shape = to_int_list(self.get_tensor_shape(output_tensors[0]))

        input_nbytes = int(np.prod(input_shape)) * np.dtype(input_dtype).itemsize
        output_nbytes = int(np.prod(output_shape)) * np.dtype(output_dtype).itemsize
        assert input_nbytes == output_nbytes, (
            "TFLite BITCAST requires input.nbytes == output.nbytes, "
            f"but got input={input_nbytes} bytes, output={output_nbytes} bytes"
        )

        return relax.op.memory.view(in_expr, shape=output_shape, dtype=output_dtype)

    def convert_rfft2d(self, op):
        """Convert TFLite RFFT2D for static no-padding/no-truncation shapes."""
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        if len(input_tensors) != 2 or len(output_tensors) != 1:
            raise NotImplementedError("RFFT2D expects two inputs and one output")

        data_tensor, fft_length_tensor = input_tensors
        output_tensor = output_tensors[0]
        if data_tensor.tensor.Type() != TensorType.FLOAT32:
            raise NotImplementedError("RFFT2D input must be float32")
        if not self._is_tflite_complex64_type(output_tensor.tensor.Type()):
            raise NotImplementedError("RFFT2D output must be COMPLEX64")
        if (
            data_tensor.tensor.Sparsity() is not None
            or fft_length_tensor.tensor.Sparsity() is not None
        ):
            raise NotImplementedError("RFFT2D does not support sparse inputs")

        input_shape = tuple(to_int_list(self.get_tensor_shape(data_tensor)))
        tflite_output_shape = tuple(to_int_list(self.get_tensor_shape(output_tensor)))
        if len(input_shape) < 2:
            raise NotImplementedError("RFFT2D input rank must be at least 2")

        try:
            fft_length_value = self.get_tensor_value_or_prefetched(fft_length_tensor)
        except (ValueError, TypeError):
            raise NotImplementedError("RFFT2D requires a constant fft_length") from None
        if fft_length_value is None:
            raise NotImplementedError("RFFT2D requires a constant fft_length")
        # Reject non-integer fft_length tensors before astype("int64") can silently
        # truncate (e.g. float32 2.7 -> int64 2).
        if not np.issubdtype(fft_length_value.dtype, np.integer):
            raise NotImplementedError(
                f"RFFT2D fft_length must be an integer tensor, got dtype {fft_length_value.dtype!r}"
            )
        fft_length = fft_length_value.astype("int64")
        if tuple(fft_length.shape) != (2,):
            raise NotImplementedError("RFFT2D fft_length must be a length-2 tensor")

        height = int(fft_length[0])
        width = int(fft_length[1])
        if height <= 0 or width <= 0:
            raise NotImplementedError(
                f"RFFT2D fft_length must be positive, got ({height}, {width})"
            )
        if height != int(input_shape[-2]) or width != int(input_shape[-1]):
            raise NotImplementedError(
                "RFFT2D currently supports fft_length matching the input spatial shape"
            )
        expected_tflite_output_shape = input_shape[:-2] + (height, width // 2 + 1)
        if tflite_output_shape != expected_tflite_output_shape:
            raise NotImplementedError("RFFT2D output shape does not match fft_length")

        relax_output_shape = self._get_relax_tensor_shape(output_tensor)
        # Dispatch: power-of-2 sizes use the O(N^2 log N) Cooley-Tukey FFT kernel;
        # the remaining (odd / non-power-of-2) shapes fall back to the O(N^4) DFT
        # reference. Both kernels share the same call_tir contract, so the
        # downstream code is kernel-agnostic.
        if _is_power_of_2(height) and _is_power_of_2(width):
            function = _build_tflite_rfft2d_fft_function(input_shape, relax_output_shape)
        else:
            function = _build_tflite_rfft2d_function(input_shape, relax_output_shape)
        module_builder = self.conversion_state["module_builder"]
        func_name = f"tflite_rfft2d_{output_tensor.tensor_idx}"
        gv = module_builder.add_func(function, func_name)
        data_expr = self.get_tensor_expr(data_tensor)
        call = relax.call_tir(
            gv,
            [data_expr],
            ty_args=[relax.TensorType(relax_output_shape, "float32")],
        )
        return self.bb.normalize(call)

    def convert_broadcast_args(self, op):
        """Convert TFLite BROADCAST_ARGS"""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        assert len(output_tensors) == 1, "output tensors length should be 1"

        s0 = self.get_tensor_expr(input_tensors[0])
        s1 = self.get_tensor_expr(input_tensors[1])
        s0_len = to_int_list(self.get_tensor_shape(input_tensors[0]))[0]
        s1_len = to_int_list(self.get_tensor_shape(input_tensors[1]))[0]
        out_dtype = self.get_tensor_type_str(input_tensors[0].tensor.Type())

        # Left-pad the shorter input with 1s to length target_len.
        target_len = tirx.max(s0_len, s1_len)
        one = relax.const(1, dtype=out_dtype)
        s0 = relax.op.concat(
            [relax.op.full([target_len - s0_len], one, dtype=out_dtype), s0], axis=0
        )
        s1 = relax.op.concat(
            [relax.op.full([target_len - s1_len], one, dtype=out_dtype), s1], axis=0
        )
        # Per-dim broadcast. If either side is 1 take the other, else elementwise max.
        s0_is_one = relax.op.equal(s0, one)
        s1_is_one = relax.op.equal(s1, one)
        return relax.op.where(
            s0_is_one,
            s1,
            relax.op.where(s1_is_one, s0, relax.op.maximum(s0, s1)),
        )

    def convert_cast(self, op):
        """Convert TFLite CAST"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.CastOptions import CastOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # MLIR-based converter outputs no BuiltinOptions for Cast operator. In this
        # case the output type can be derived from the Cast operator output tensor.
        # When TOCO converter is used there will be "normal" BuiltinOptions.CastOptions
        # with output type.
        if op.BuiltinOptions() is not None:
            assert op.BuiltinOptionsType() == BuiltinOptions.CastOptions
            op_options = op.BuiltinOptions()
            cast_options = CastOptions()
            cast_options.Init(op_options.Bytes, op_options.Pos)
            cast_dtype = cast_options.OutDataType()
        else:
            cast_dtype = self.get_output_tensors(op)[0].tensor.Type()

        out = relax.op.astype(in_expr, self.get_tensor_type_str(cast_dtype))

        return out

    def convert_tile(self, op):
        """tile implementation."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        in_expr = self.get_expr(input_tensor_idx)

        reps = tuple(self.get_tensor_value(input_tensors[1]))

        out = relax.op.tile(in_expr, reps)

        return out

    def convert_topk_v2(self, op):
        """Convert TFLite TOPK_v2"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)
        k = self.get_tensor_value(input_tensors[1])
        out = relax.op.topk(in_expr, int(k))

        return out

    def convert_pool2d(self, op, pool_type):
        """pool2d implementation."""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.Padding import Padding
        from tflite.Pool2DOptions import Pool2DOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool2d_options = Pool2DOptions()
        pool2d_options.Init(op_options.Bytes, op_options.Pos)
        stride_h = pool2d_options.StrideH()
        stride_w = pool2d_options.StrideW()
        padding = pool2d_options.Padding()
        filter_h = pool2d_options.FilterHeight()
        filter_w = pool2d_options.FilterWidth()
        fused_activation_fn = pool2d_options.FusedActivationFunction()

        params = {
            "pool_size": (filter_h, filter_w),
            "strides": (stride_h, stride_w),
            "padding": [0, 0],
            "layout": "NHWC",
        }

        in_expr = self.get_expr(input_tensor_idx)

        _, input_h, input_w, _ = to_int_list(self.get_tensor_shape(input_tensor))

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, filter_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, filter_w, stride_w)
            params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
        else:
            raise NotImplementedError(
                f"Padding format {padding} for operator Pool2D is not supported."
            )

        if pool_type == "average":
            if input_tensor.qnn_params:
                assert self.has_same_qnn_params(input_tensor, output_tensor), (
                    "TFLite avg_pool2dreshape requires input and output scale"
                    "and zero points to be equal"
                )
                out = relax.op.astype(in_expr, "int32")
                out = relax.op.nn.avg_pool2d(out, **params)
                out = relax.op.astype(out, output_tensor_type_str)
            else:
                out = relax.op.nn.avg_pool2d(in_expr, **params)
        elif pool_type == "max":
            if input_tensor.qnn_params:
                assert self.has_same_qnn_params(input_tensor, output_tensor), (
                    "qnn.op.max_pool2d requires input and output qnn params to be same"
                )
            out = relax.op.nn.max_pool2d(in_expr, **params)
        elif pool_type == "l2":
            # L2_POOL_2D is equivalent to square_root(avg_pool(square(in_data)))
            # TFLite does not have support for quantised L2_POOL_2D op.
            assert not input_tensor.qnn_params, (
                "As TFLite does not have support for quantized L2_POOL_2D, \
                Quantized input is not expected."
            )
            exp_type = self.get_tensor_type_str(output_tensor.tensor.Type())
            square_exp = relax.op.power(in_expr, relax.const(2, exp_type))
            avg_pool_exp = relax.op.nn.avg_pool2d(square_exp, **params)
            out = relax.op.sqrt(avg_pool_exp)
        else:
            raise NotImplementedError(
                f"Operator {pool_type} pool is not supported for frontend TFLite."
            )

        # Handle fused activations
        if output_tensor.qnn_params:
            scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=scale_val,
                zero_point=zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_pad(self, op):
        """Convert TFLite PAD/PADV2 \
           TFLite treats PAD and PADV2 operators identically"""

        input_tensors = self.get_input_tensors(op)

        # TFLite PAD/PADV2 only supports CONSTANT mode
        assert len(input_tensors) == 2 or len(input_tensors) == 3, (
            "input tensor's length should be 2 for PAD and 3 for PADV2"
        )

        if len(input_tensors) == 3:
            assert input_tensors[0].tensor.Type() == input_tensors[2].tensor.Type(), (
                "constant_values tensor must be of same type as input tensor"
            )

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])

        # convert list of lists to tuple of tuples
        paddings = []
        for val in pad_list:
            paddings += val.tolist()

        # Set the pad value, by default 0, unless constant_values parameter is provided
        pad_value = 0

        if input_tensor.qnn_params:
            # Check that input and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(input_tensor, output_tensor), (
                "TFLite PADV2 requires input and output scale and zero points to be equal"
            )

            # The pad value for quantized pad is the input zero point by default.
            pad_value = float(input_tensor.qnn_params["zero_point"].value.numpy())

        if len(input_tensors) == 3:
            pad_value = self.get_tensor_value(input_tensors[2])
            if isinstance(pad_value, np.ndarray):
                pad_value = pad_value.tolist()
            if isinstance(pad_value, list):
                assert len(pad_value) == 1, "Only one constant value is expected."
                pad_value = pad_value[0]
            if input_tensor.qnn_params:
                # Check that input tensor and constant_values have same qnn params.
                assert self.has_same_qnn_params(input_tensor, input_tensors[2]), (
                    "TFLite PADV2 requires input and constant_values tensors' \
                        scale and zero points to be equal"
                )

        out = relax.op.nn.pad(in_expr, pad_width=paddings, pad_value=pad_value)
        return out

    def convert_mirror_pad(self, op):
        """Convert TFLite MIRROR_PAD"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.MirrorPadOptions import MirrorPadOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])
        # convert list of lists to tuple of tuples
        paddings = tuple(tuple(val.astype(np.int32)) for val in pad_list)

        assert op.BuiltinOptionsType() == BuiltinOptions.MirrorPadOptions
        op_options = op.BuiltinOptions()
        mirror_pad_options = MirrorPadOptions()
        mirror_pad_options.Init(op_options.Bytes, op_options.Pos)
        mode_byte = mirror_pad_options.Mode()

        mode = "REFLECT" if mode_byte == 0 else "SYMMETRIC"
        if mode == "SYMMETRIC":
            raise NotImplementedError("MIRROR_PAD with SYMMETRIC mode is not yet supported.")
        # Flatten tuple-of-tuples to a list for relax.op.nn.pad
        flat_pads = [int(v) for pair in paddings for v in pair]
        out = relax.op.nn.pad(in_expr, flat_pads, pad_mode="reflect")

        return out

    def convert_pack(self, op):
        """Convert TFLite pack"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.PackOptions import PackOptions

        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        if input_tensors[0].qnn_params:
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(input_tensors[0], output_tensor), (
                "TFLite pack requires input and output scale and zero points to be equal"
            )

            for input_tensor in input_tensors:
                assert self.has_same_qnn_params(input_tensors[0], input_tensor), (
                    "TFLite pack requires all input tensors to have same scale and zero point"
                )

        assert op.BuiltinOptionsType() == BuiltinOptions.PackOptions
        op_options = op.BuiltinOptions()
        pack_options = PackOptions()
        pack_options.Init(op_options.Bytes, op_options.Pos)
        pack_axis = pack_options.Axis()
        pack_values_count = pack_options.ValuesCount()
        assert len(input_tensors) == pack_values_count, "Discordance in input values count"

        in_exprs = [self.get_tensor_expr(_) for _ in input_tensors]
        in_exprs_reshaped = [_op.expand_dims(_, axis=pack_axis) for _ in in_exprs]
        out = relax.op.concat(in_exprs_reshaped, pack_axis)
        return out

    def convert_unpack(self, op):
        """Convert TFLite unpack"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.UnpackOptions import UnpackOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        assert op.BuiltinOptionsType() == BuiltinOptions.UnpackOptions
        op_options = op.BuiltinOptions()
        unpack_options = UnpackOptions()
        unpack_options.Init(op_options.Bytes, op_options.Pos)
        num_unpacks = unpack_options.Num()
        unpack_axis = unpack_options.Axis()

        # Relax doesn't support 'unpack' operator so we use 'split' & 'squeeze' instead.
        # We have to do 'squeeze' along the split axis.
        # Relax expects squeeze_axis to be List.
        squeeze_axis = [unpack_axis]

        # Relax doesn't like TupleWrapper of 1 element so we isolate the case of unpacking
        # a tensor by an axis with len(axis) == 1. For reference see convert_split().
        # Such unpacking will result in the same tensor so we omit 'split' and only squeeze
        # along the axis of dim == 1.
        if num_unpacks == 1:
            squeezed = relax.op.squeeze(in_expr, axis=squeeze_axis)
            if isinstance(squeezed, relax.Tuple):
                squeezed = squeezed[0]
        else:
            splitted = relax.op.split(in_expr, indices_or_sections=num_unpacks, axis=unpack_axis)
            squeezed = relax.Tuple(
                [
                    _op.squeeze(relax.TupleGetItem(splitted, i), axis=squeeze_axis)
                    for i in range(num_unpacks)
                ]
            )

        return squeezed

    def convert_lstm(self, op):
        """Convert TFLite LSTM (single-step).

        Standard LSTM cell with FULL kernel and coupled input-forget gate.
        Peephole, projection, and layer norm are not supported.

        Inputs (24 tensors, many optional):
          [0]  input                      [batch, input_size]
          [1]  input_to_input_weights     (optional, -1 => coupled)
          [2]  input_to_forget_weights    [num_units, input_size]
          [3]  input_to_cell_weights      [num_units, input_size]
          [4]  input_to_output_weights    [num_units, input_size]
          [5]  recurrent_to_input_weights (optional)
          [6]  recurrent_to_forget_weights [num_units, num_units]
          [7]  recurrent_to_cell_weights  [num_units, num_units]
          [8]  recurrent_to_output_weights [num_units, num_units]
          [9-11] cell_to_*_weights        (optional, not supported)
          [12] input_gate_bias            (optional)
          [13] forget_gate_bias           [num_units]
          [14] cell_bias                  [num_units]
          [15] output_gate_bias           [num_units]
          [16-17] projection_weights/bias (optional, not supported)
          [18] output_state               [batch, num_units]
          [19] cell_state                 [batch, num_units]
          [20-23] layer_norm              (optional, not supported)

        Output:
          [0] output  [batch, num_units]

        Cell (coupled input-forget):
          f = sigmoid(x @ W_f.T + h @ R_f.T + b_f)
          i = 1 - f
          g = tanh(x @ W_c.T + h @ R_c.T + b_c)
          o = sigmoid(x @ W_o.T + h @ R_o.T + b_o)
          c_new = f * c_prev + i * g
          h_new = fused_activation(o * tanh(c_new))
        """
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.LSTMOptions import LSTMOptions

        if self.is_quantized(op):
            raise NotImplementedError("TFLite quantized LSTM is not supported yet.")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 24, (
            f"input tensors length should be 24, got {len(input_tensors)}"
        )

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) >= 1, "output tensors length should be at least 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.LSTMOptions
        op_options = op.BuiltinOptions()
        lstm_opts = LSTMOptions()
        lstm_opts.Init(op_options.Bytes, op_options.Pos)

        fused_activation_fn = lstm_opts.FusedActivationFunction()
        cell_clip = lstm_opts.CellClip()
        proj_clip = lstm_opts.ProjClip()

        in_expr = self.get_tensor_expr(input_tensors[0])

        # Only coupled input-forget gate is supported.
        if input_tensors[1].tensor_idx != -1 or input_tensors[5].tensor_idx != -1:
            raise NotImplementedError("Only coupled input-forget LSTM is supported.")

        # Peephole, projection, and layer norm are not modeled yet.
        if (
            any(t.tensor_idx != -1 for t in input_tensors[9:12])
            or any(t.tensor_idx != -1 for t in input_tensors[16:18])
            or any(t.tensor_idx != -1 for t in input_tensors[20:24])
        ):
            raise NotImplementedError(
                "Peephole, projection, and layer norm LSTM are not supported yet."
            )

        # Weights.
        w_f = self.get_tensor_expr(input_tensors[2])
        w_c = self.get_tensor_expr(input_tensors[3])
        w_o = self.get_tensor_expr(input_tensors[4])

        r_f = self.get_tensor_expr(input_tensors[6])
        r_c = self.get_tensor_expr(input_tensors[7])
        r_o = self.get_tensor_expr(input_tensors[8])

        # Biases.
        b_f = self.get_tensor_expr(input_tensors[13])
        b_c = self.get_tensor_expr(input_tensors[14])
        b_o = self.get_tensor_expr(input_tensors[15])

        # State inputs.
        h_prev = self.get_tensor_expr(input_tensors[18])
        c_prev = self.get_tensor_expr(input_tensors[19])

        # Coupled input-forget gate.
        f = relax.op.sigmoid(
            relax.op.add(
                relax.op.add(
                    relax.op.matmul(in_expr, relax.op.permute_dims(w_f)),
                    relax.op.matmul(h_prev, relax.op.permute_dims(r_f)),
                ),
                b_f,
            )
        )
        i = relax.op.subtract(
            relax.const(1.0, "float32"),
            f,
        )

        # Cell candidate.
        g = relax.op.tanh(
            relax.op.add(
                relax.op.add(
                    relax.op.matmul(in_expr, relax.op.permute_dims(w_c)),
                    relax.op.matmul(h_prev, relax.op.permute_dims(r_c)),
                ),
                b_c,
            )
        )

        # Output gate.
        o = relax.op.sigmoid(
            relax.op.add(
                relax.op.add(
                    relax.op.matmul(in_expr, relax.op.permute_dims(w_o)),
                    relax.op.matmul(h_prev, relax.op.permute_dims(r_o)),
                ),
                b_o,
            )
        )

        # Cell state update with optional clipping.
        c_new = relax.op.add(
            relax.op.multiply(f, c_prev),
            relax.op.multiply(i, g),
        )
        if cell_clip > 0:
            c_new = relax.op.clip(c_new, -cell_clip, cell_clip)

        # Hidden state.
        # TFLite applies the fused activation to the cell state before the
        # output gate multiply.
        h_new = relax.op.multiply(
            o, self.convert_fused_activation_function(c_new, fused_activation_fn)
        )
        if proj_clip > 0:
            h_new = relax.op.clip(h_new, -proj_clip, proj_clip)

        # Update state tensors in the expression table for subsequent ops.
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[18].tensor_idx),
            h_new,
            force_override=True,
        )
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[19].tensor_idx),
            c_new,
            force_override=True,
        )

        return h_new

    def convert_svdf(self, op):
        """Convert TFLite SVDF (single-step).

        Structured-Vectorized Bidirectional Filter for keyword spotting.

        Inputs (5 tensors):
          [0] input           [batch, input_size]
          [1] feature_weights [num_filters, input_size]
          [2] time_weights    [num_filters, memory_size]
          [3] bias            [num_filters]           (optional)
          [4] state           [batch, num_filters * memory_size]  (variable)

        Output:
          [0] output  [batch, num_units]

        Computation:
          feat = x @ W_feat.T                              # feature projection
          state_r = reshape(state, [B, F, memory_size])    # ring buffer
          time = sum(state_r * time_weights, axis=-1)      # time filtering
          out = activation(sum(reshape(time, [B, U, rank]), axis=-1) + bias)
        """
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.SVDFOptions import SVDFOptions

        if self.is_quantized(op):
            raise NotImplementedError("TFLite quantized SVDF is not supported yet.")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 5, (
            f"input tensors length should be 5, got {len(input_tensors)}"
        )

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) >= 1, "output tensors length should be at least 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.SVDFOptions
        op_options = op.BuiltinOptions()
        svdf_opts = SVDFOptions()
        svdf_opts.Init(op_options.Bytes, op_options.Pos)

        rank = svdf_opts.Rank()
        fused_activation_fn = svdf_opts.FusedActivationFunction()

        in_expr = self.get_tensor_expr(input_tensors[0])
        feat_weights = self.get_tensor_expr(input_tensors[1])
        time_weights = self.get_tensor_expr(input_tensors[2])

        batch_size = self.get_tensor_shape(input_tensors[0])[0]
        if isinstance(batch_size, np.integer | int):
            batch_size = int(batch_size)
        num_filters = to_int_list(self.get_tensor_shape(input_tensors[1]))[0]
        if num_filters % rank != 0:
            raise NotImplementedError("SVDF num_filters must be divisible by rank.")
        num_units = num_filters // rank
        memory_size = to_int_list(self.get_tensor_shape(input_tensors[2]))[1]

        # Feature projection: [batch, input_size] @ [input_size, num_filters]
        feat = relax.op.matmul(in_expr, relax.op.permute_dims(feat_weights))

        # Time filtering: reshape state -> weight -> reduce.
        state_expr = self.get_tensor_expr(input_tensors[4])
        state_3d = relax.op.reshape(state_expr, (batch_size, num_filters, memory_size))

        # time_weights: [num_filters, memory_size], broadcast to [1, num_filters, memory_size]
        tw_3d = relax.op.reshape(time_weights, (1, num_filters, memory_size))
        time_weighted = relax.op.multiply(state_3d, tw_3d)
        time_output = relax.op.sum(time_weighted, axis=-1, keepdims=False)
        reduced = relax.op.reshape(time_output, (batch_size, num_units, rank))
        result = relax.op.sum(reduced, axis=-1, keepdims=False)

        # Add bias if present
        if input_tensors[3].tensor_idx != -1:
            bias_expr = self.get_tensor_expr(input_tensors[3])
            result = relax.op.add(result, bias_expr)

        result = self.convert_fused_activation_function(result, fused_activation_fn)

        # Update state tensor in the expression table for subsequent steps.
        # SVDF state is a FIFO ring-buffer: shift left by 1, append new feat.
        feat_3d = relax.op.expand_dims(feat, axis=-1)
        if memory_size > 1:
            shifted_state = relax.op.strided_slice(
                state_3d, axes=[2], begin=[1], end=[int(memory_size)]
            )
            new_state_3d = relax.op.concat([shifted_state, feat_3d], axis=2)
        else:
            new_state_3d = feat_3d
        new_state = relax.op.reshape(new_state_3d, (batch_size, num_filters * memory_size))
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[4].tensor_idx),
            new_state,
            force_override=True,
        )

        return result

    def convert_unidirectional_sequence_rnn(self, op):
        """Convert TFLite UNIDIRECTIONAL_SEQUENCE_RNN.

        Inputs (5 tensors):
          [0] input          [batch, time, input_size]  (or [time, batch, input_size] if time_major)
          [1] input_weights  [num_units, input_size]
          [2] recurrent_weights [num_units, num_units]
          [3] bias           [num_units]
          [4] hidden_state   [batch, num_units]  (variable, zero-initialised)

        Output:
          [0] output  [batch, time, num_units]

        Cell equation:
          h_t = fused_activation(x_t @ W.T + h_{t-1} @ Wr.T + b)
        """
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.SequenceRNNOptions import SequenceRNNOptions

        if self.is_quantized(op):
            raise NotImplementedError(
                "TFLite quantized UNIDIRECTIONAL_SEQUENCE_RNN is not supported yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 5, "input tensors length should be 5"

        input_tensor = input_tensors[0]
        weights_tensor = input_tensors[1]
        recurrent_tensor = input_tensors[2]
        bias_tensor = input_tensors[3]
        hidden_state_tensor = input_tensors[4]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) >= 1, "output tensors length should be at least 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.SequenceRNNOptions
        op_options = op.BuiltinOptions()
        seq_rnn_options = SequenceRNNOptions()
        seq_rnn_options.Init(op_options.Bytes, op_options.Pos)
        time_major = seq_rnn_options.TimeMajor()
        fused_activation_fn = seq_rnn_options.FusedActivationFunction()

        # Constant weight/bias expressions.
        weights_expr = self.get_tensor_expr(weights_tensor)  # [num_units, input_size]
        recurrent_expr = self.get_tensor_expr(recurrent_tensor)  # [num_units, num_units]

        # bias is optional (tensor_idx == -1 when absent); default to zeros.
        if bias_tensor.tensor_idx != -1:
            bias_expr = self.get_tensor_expr(bias_tensor)  # [num_units]
        else:
            num_units = int(self.get_tensor_shape(weights_tensor)[0])
            bias_dtype = self.get_tensor_type_str(weights_tensor.tensor.Type())
            bias_expr = relax.op.zeros((num_units,), dtype=bias_dtype)

        # Transpose to [input_size, num_units] and [num_units, num_units] for x @ W.T.
        w_t = relax.op.permute_dims(weights_expr)
        wr_t = relax.op.permute_dims(recurrent_expr)

        # Resolve the input expression; normalise to batch-major [batch, time, input_size].
        # Only the time dimension must be static (needed for unrolling); batch may be dynamic.
        in_expr = self.get_tensor_expr(input_tensor)
        in_shape = self.get_tensor_shape(input_tensor)
        if time_major:
            in_expr = relax.op.permute_dims(in_expr, [1, 0, 2])
            num_steps = int(in_shape[0])
        else:
            num_steps = int(in_shape[1])

        # Initial hidden state: use the model's tensor value when available (non-zero init or
        # graph input), otherwise fall back to zeros for the common variable-tensor case.
        h_dtype = self.get_tensor_type_str(hidden_state_tensor.tensor.Type())
        if self.has_expr(hidden_state_tensor.tensor_idx) or (
            hidden_state_tensor.buffer is not None and hidden_state_tensor.buffer.DataLength() > 0
        ):
            h = self.get_tensor_expr(hidden_state_tensor)
        else:
            h_shape = tuple(to_int_list(self.get_tensor_shape(hidden_state_tensor)))
            h = relax.op.zeros(h_shape, dtype=h_dtype)

        # Unroll over the time axis.
        # relax.op.split with 1 section returns the tensor directly; handle uniformly.
        if num_steps == 1:
            steps = [relax.op.squeeze(in_expr, axis=[1])]
        else:
            splits = relax.op.split(in_expr, num_steps, axis=1)
            steps = [relax.op.squeeze(splits[i], axis=[1]) for i in range(num_steps)]

        outputs = []
        for x_t in steps:  # x_t: [batch, input_size]
            gates = relax.op.add(
                relax.op.add(relax.op.matmul(x_t, w_t), relax.op.matmul(h, wr_t)),
                bias_expr,
            )
            h = self.convert_fused_activation_function(gates, fused_activation_fn)
            outputs.append(h)

        # Stack timestep outputs: [batch, time, num_units].
        return relax.op.stack(outputs, axis=1)

    def convert_unidirectional_sequence_lstm(self, op):
        """Convert TFLite UNIDIRECTIONAL_SEQUENCE_LSTM.

        Inputs (24 tensors, same layout as single-step LSTM):
          [0]  input                       [batch, time, input_size]
          [1]  input_to_input_weights      [num_units, input_size]   (optional)
          [2]  input_to_forget_weights     [num_units, input_size]
          [3]  input_to_cell_weights       [num_units, input_size]
          [4]  input_to_output_weights     [num_units, input_size]
          [5]  recurrent_to_input_weights  [num_units, num_units]   (optional)
          [6]  recurrent_to_forget_weights [num_units, num_units]
          [7]  recurrent_to_cell_weights   [num_units, num_units]
          [8]  recurrent_to_output_weights [num_units, num_units]
          [9]  cell_to_input_weights       [num_units]              (optional)
          [10] cell_to_forget_weights      [num_units]              (optional)
          [11] cell_to_output_weights      [num_units]              (optional)
          [12] input_gate_bias             [num_units]              (optional)
          [13] forget_gate_bias            [num_units]
          [14] cell_gate_bias              [num_units]
          [15] output_gate_bias            [num_units]
          [16] projection_weights          [num_units, num_units]   (optional)
          [17] projection_bias             [num_units]              (optional)
          [18] output_state                [batch, num_units]       (variable)
          [19] cell_state                  [batch, num_units]       (variable)
          [20-23] optional layer norm weights

        Output:
          [0] output  [batch, time, num_units]

        Uses coupled input-forget gate (i = 1 - f) for the FULL kernel.
        """
        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.UnidirectionalSequenceLSTMOptions import UnidirectionalSequenceLSTMOptions

        if self.is_quantized(op):
            raise NotImplementedError(
                "TFLite quantized UNIDIRECTIONAL_SEQUENCE_LSTM is not supported yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 24, (
            f"input tensors length should be 24, got {len(input_tensors)}"
        )

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) >= 1, "output tensors length should be at least 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.UnidirectionalSequenceLSTMOptions
        op_options = op.BuiltinOptions()
        lstm_opts = UnidirectionalSequenceLSTMOptions()
        lstm_opts.Init(op_options.Bytes, op_options.Pos)
        time_major = lstm_opts.TimeMajor()
        fused_activation_fn = lstm_opts.FusedActivationFunction()
        cell_clip = lstm_opts.CellClip()
        proj_clip = lstm_opts.ProjClip()

        # Only coupled input-forget gate is supported.
        if input_tensors[1].tensor_idx != -1 or input_tensors[5].tensor_idx != -1:
            raise NotImplementedError("Only coupled input-forget LSTM is supported.")
        if any(input_tensors[idx].tensor_idx != -1 for idx in [9, 10, 11]):
            raise NotImplementedError("TFLite peephole LSTM is not supported yet.")
        if any(input_tensors[idx].tensor_idx != -1 for idx in [16, 17]):
            raise NotImplementedError("TFLite projection LSTM is not supported yet.")
        if any(input_tensors[idx].tensor_idx != -1 for idx in [20, 21, 22, 23]):
            raise NotImplementedError("TFLite layer-norm LSTM is not supported yet.")

        # Weights (transposed once outside the loop).
        w_f_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[2]))
        w_c_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[3]))
        w_o_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[4]))
        r_f_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[6]))
        r_c_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[7]))
        r_o_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[8]))

        # Biases.
        b_f = self.get_tensor_expr(input_tensors[13])
        b_c = self.get_tensor_expr(input_tensors[14])
        b_o = self.get_tensor_expr(input_tensors[15])

        # Initial states.
        h = self.get_tensor_expr(input_tensors[18])
        c = self.get_tensor_expr(input_tensors[19])

        # Resolve the input expression; normalise to batch-major [batch, time, input_size].
        in_expr = self.get_tensor_expr(input_tensors[0])
        in_shape = self.get_tensor_shape(input_tensors[0])
        if time_major:
            in_expr = relax.op.permute_dims(in_expr, [1, 0, 2])
            num_steps = int(in_shape[0])
        else:
            num_steps = int(in_shape[1])

        # Unroll over the time axis.
        if num_steps == 1:
            steps = [relax.op.squeeze(in_expr, axis=[1])]
        else:
            splits = relax.op.split(in_expr, num_steps, axis=1)
            steps = [relax.op.squeeze(splits[i], axis=[1]) for i in range(num_steps)]

        one = relax.const(1.0, "float32")
        outputs = []
        for x_t in steps:
            f = relax.op.sigmoid(
                relax.op.add(
                    relax.op.add(
                        relax.op.matmul(x_t, w_f_t),
                        relax.op.matmul(h, r_f_t),
                    ),
                    b_f,
                )
            )
            i = relax.op.subtract(one, f)
            g = self.convert_fused_activation_function(
                relax.op.add(
                    relax.op.add(relax.op.matmul(x_t, w_c_t), relax.op.matmul(h, r_c_t)),
                    b_c,
                ),
                fused_activation_fn,
            )
            o = relax.op.sigmoid(
                relax.op.add(
                    relax.op.add(
                        relax.op.matmul(x_t, w_o_t),
                        relax.op.matmul(h, r_o_t),
                    ),
                    b_o,
                )
            )

            c_new = relax.op.add(relax.op.multiply(f, c), relax.op.multiply(i, g))
            if cell_clip > 0.0:
                c_new = relax.op.clip(c_new, -cell_clip, cell_clip)

            h_new = relax.op.multiply(
                o, self.convert_fused_activation_function(c_new, fused_activation_fn)
            )
            if proj_clip > 0.0:
                h_new = relax.op.clip(h_new, -proj_clip, proj_clip)
            outputs.append(h_new)
            h, c = h_new, c_new

        h_out = relax.op.stack(outputs, axis=1)
        if time_major:
            h_out = relax.op.permute_dims(h_out, [1, 0, 2])

        # Update state tensors in the expression table for subsequent ops.
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[18].tensor_idx),
            h,
            force_override=True,
        )
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[19].tensor_idx),
            c,
            force_override=True,
        )

        return h_out

    def convert_bidirectional_sequence_rnn(self, op):
        """Convert TFLite BIDIRECTIONAL_SEQUENCE_RNN.

        Inputs (9 tensors, aux_input not supported):
          [0] input                [batch, time, input_size]
          [1] fw_weights           [num_units, input_size]
          [2] fw_recurrent_weights [num_units, num_units]
          [3] fw_bias              [num_units]
          [4] fw_hidden_state      [batch, num_units]         (variable)
          [5] bw_weights           [num_units, input_size]
          [6] bw_recurrent_weights [num_units, num_units]
          [7] bw_bias              [num_units]
          [8] bw_hidden_state      [batch, num_units]         (variable)

        Output (merge_outputs=True):
          [0] output  [batch, time, 2 * num_units]  (fw and bw concatenated)

        Output (merge_outputs=False):
          [0] fw_output  [batch, time, num_units]
          [1] bw_output  [batch, time, num_units]
        """
        from tflite.BidirectionalSequenceRNNOptions import BidirectionalSequenceRNNOptions
        from tflite.BuiltinOptions import BuiltinOptions

        if self.is_quantized(op):
            raise NotImplementedError(
                "TFLite quantized BIDIRECTIONAL_SEQUENCE_RNN is not supported yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 12, (
            f"input tensors length should be 12, got {len(input_tensors)}"
        )

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) >= 1, "output tensors length should be at least 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.BidirectionalSequenceRNNOptions
        op_options = op.BuiltinOptions()
        rnn_opts = BidirectionalSequenceRNNOptions()
        rnn_opts.Init(op_options.Bytes, op_options.Pos)
        time_major = rnn_opts.TimeMajor()
        fused_activation_fn = rnn_opts.FusedActivationFunction()
        merge_outputs = rnn_opts.MergeOutputs()
        if any(input_tensors[idx].tensor_idx != -1 for idx in [9, 10, 11]):
            raise NotImplementedError(
                "TFLite BIDIRECTIONAL_SEQUENCE_RNN aux input is not supported yet."
            )

        # Forward weights and biases.
        fw_weights_expr = self.get_tensor_expr(input_tensors[1])
        fw_recurrent_expr = self.get_tensor_expr(input_tensors[2])
        fw_bias_expr = self.get_tensor_expr(input_tensors[3])
        fw_w_t = relax.op.permute_dims(fw_weights_expr)
        fw_wr_t = relax.op.permute_dims(fw_recurrent_expr)

        # Backward weights and biases.
        bw_weights_expr = self.get_tensor_expr(input_tensors[5])
        bw_recurrent_expr = self.get_tensor_expr(input_tensors[6])
        bw_bias_expr = self.get_tensor_expr(input_tensors[7])
        bw_w_t = relax.op.permute_dims(bw_weights_expr)
        bw_wr_t = relax.op.permute_dims(bw_recurrent_expr)

        # Resolve the input expression; normalise to batch-major [batch, time, input_size].
        in_expr = self.get_tensor_expr(input_tensors[0])
        in_shape = self.get_tensor_shape(input_tensors[0])
        if time_major:
            in_expr = relax.op.permute_dims(in_expr, [1, 0, 2])
            num_steps = int(in_shape[0])
        else:
            num_steps = int(in_shape[1])

        # Initial hidden states.
        def _get_hidden_state(tensor):
            if self.has_expr(tensor.tensor_idx) or (
                tensor.buffer is not None and tensor.buffer.DataLength() > 0
            ):
                return self.get_tensor_expr(tensor)
            dtype = self.get_tensor_type_str(tensor.tensor.Type())
            h_shape = tuple(to_int_list(self.get_tensor_shape(tensor)))
            return relax.op.zeros(h_shape, dtype=dtype)

        fw_h = _get_hidden_state(input_tensors[4])
        bw_h = _get_hidden_state(input_tensors[8])

        # Unroll over the time axis.
        if num_steps == 1:
            steps = [relax.op.squeeze(in_expr, axis=[1])]
        else:
            splits = relax.op.split(in_expr, num_steps, axis=1)
            steps = [relax.op.squeeze(splits[i], axis=[1]) for i in range(num_steps)]

        # Forward pass.
        fw_outputs = []
        for x_t in steps:
            gates = relax.op.add(
                relax.op.add(relax.op.matmul(x_t, fw_w_t), relax.op.matmul(fw_h, fw_wr_t)),
                fw_bias_expr,
            )
            fw_h = self.convert_fused_activation_function(gates, fused_activation_fn)
            fw_outputs.append(fw_h)

        # Backward pass (process steps in reverse).
        bw_outputs = []
        for x_t in reversed(steps):
            gates = relax.op.add(
                relax.op.add(relax.op.matmul(x_t, bw_w_t), relax.op.matmul(bw_h, bw_wr_t)),
                bw_bias_expr,
            )
            bw_h = self.convert_fused_activation_function(gates, fused_activation_fn)
            bw_outputs.append(bw_h)
        bw_outputs.reverse()

        fw_stacked = relax.op.stack(fw_outputs, axis=1)  # [batch, time, num_units]
        bw_stacked = relax.op.stack(bw_outputs, axis=1)  # [batch, time, num_units]
        if time_major:
            fw_stacked = relax.op.permute_dims(fw_stacked, [1, 0, 2])
            bw_stacked = relax.op.permute_dims(bw_stacked, [1, 0, 2])

        # Update state tensors in the expression table for subsequent ops.
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[4].tensor_idx),
            fw_h,
            force_override=True,
        )
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[8].tensor_idx),
            bw_h,
            force_override=True,
        )

        if merge_outputs:
            return relax.op.concat([fw_stacked, bw_stacked], axis=-1)
        else:
            return relax.Tuple([fw_stacked, bw_stacked])

    def convert_bidirectional_sequence_lstm(self, op):
        """Convert TFLite BIDIRECTIONAL_SEQUENCE_LSTM.

        Inputs (48 tensors, indices 0-17 forward LSTM, 18-34 backward LSTM, 35-38 states,
        39-47 optional aux inputs, which are not supported):

        Forward LSTM cell (indices 0-17, same layout as single-step LSTM):
          [0]  input (shared)              [batch, time, input_size]
          [1]  fw_input_to_input_weights   (optional)
          [2]  fw_input_to_forget_weights
          [3]  fw_input_to_cell_weights
          [4]  fw_input_to_output_weights
          [5]  fw_recurrent_to_input_wts   (optional)
          [6]  fw_recurrent_to_forget_wts
          [7]  fw_recurrent_to_cell_wts
          [8]  fw_recurrent_to_output_wts
          [9-11] fw cell_to_*_weights      (optional, not supported)
          [12] fw_input_gate_bias          (optional)
          [13] fw_forget_gate_bias
          [14] fw_cell_gate_bias
          [15] fw_output_gate_bias
          [16] fw_projection_weights       (optional, not supported)
          [17] fw_projection_bias          (optional, not supported)

        Backward LSTM cell (indices 18-34, same layout as fw):
          [19] bw_input_to_forget_weights
          [20] bw_input_to_cell_weights
          [21] bw_input_to_output_weights
          [23] bw_recurrent_to_forget_wts
          [24] bw_recurrent_to_cell_wts
          [25] bw_recurrent_to_output_wts
          [30] bw_forget_gate_bias
          [31] bw_cell_gate_bias
          [32] bw_output_gate_bias

        State tensors:
          [35] fw_activation_state  [batch, num_units]
          [36] fw_cell_state        [batch, num_units]
          [37] bw_activation_state  [batch, num_units]
          [38] bw_cell_state        [batch, num_units]

        Output (merge_outputs=True):
          [0] output  [batch, time, 2 * num_units]

        Output (merge_outputs=False):
          [0] fw_output  [batch, time, num_units]
          [1] bw_output  [batch, time, num_units]
        """
        from tflite.BidirectionalSequenceLSTMOptions import BidirectionalSequenceLSTMOptions
        from tflite.BuiltinOptions import BuiltinOptions

        if self.is_quantized(op):
            raise NotImplementedError(
                "TFLite quantized BIDIRECTIONAL_SEQUENCE_LSTM is not supported yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 48, (
            f"input tensors length should be 48, got {len(input_tensors)}"
        )

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) >= 1, "output tensors length should be at least 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.BidirectionalSequenceLSTMOptions
        op_options = op.BuiltinOptions()
        lstm_opts = BidirectionalSequenceLSTMOptions()
        lstm_opts.Init(op_options.Bytes, op_options.Pos)
        time_major = lstm_opts.TimeMajor()
        fused_activation_fn = lstm_opts.FusedActivationFunction()
        merge_outputs = lstm_opts.MergeOutputs()
        cell_clip = lstm_opts.CellClip()
        proj_clip = lstm_opts.ProjClip()

        # ── Forward LSTM weights (transposed once outside the loop) ──
        if input_tensors[1].tensor_idx != -1 or input_tensors[5].tensor_idx != -1:
            raise NotImplementedError("Only coupled input-forget LSTM is supported.")
        if any(input_tensors[idx].tensor_idx != -1 for idx in [9, 10, 11]):
            raise NotImplementedError("TFLite peephole LSTM is not supported yet.")
        if any(input_tensors[idx].tensor_idx != -1 for idx in [16, 17]):
            raise NotImplementedError("TFLite projection LSTM is not supported yet.")

        fw_w_f_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[2]))
        fw_w_c_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[3]))
        fw_w_o_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[4]))
        fw_r_f_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[6]))
        fw_r_c_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[7]))
        fw_r_o_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[8]))
        fw_b_f = self.get_tensor_expr(input_tensors[13])
        fw_b_c = self.get_tensor_expr(input_tensors[14])
        fw_b_o = self.get_tensor_expr(input_tensors[15])

        # ── Backward LSTM weights (transposed once outside the loop) ──
        if input_tensors[18].tensor_idx != -1 or input_tensors[22].tensor_idx != -1:
            raise NotImplementedError("Only coupled input-forget LSTM is supported.")
        if any(input_tensors[idx].tensor_idx != -1 for idx in [26, 27, 28]):
            raise NotImplementedError("TFLite peephole LSTM is not supported yet.")
        if any(input_tensors[idx].tensor_idx != -1 for idx in [33, 34]):
            raise NotImplementedError("TFLite projection LSTM is not supported yet.")
        if any(input_tensors[idx].tensor_idx != -1 for idx in range(39, 48)):
            raise NotImplementedError(
                "TFLite BIDIRECTIONAL_SEQUENCE_LSTM aux input is not supported yet."
            )

        bw_w_f_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[19]))
        bw_w_c_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[20]))
        bw_w_o_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[21]))
        bw_r_f_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[23]))
        bw_r_c_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[24]))
        bw_r_o_t = relax.op.permute_dims(self.get_tensor_expr(input_tensors[25]))
        bw_b_f = self.get_tensor_expr(input_tensors[30])
        bw_b_c = self.get_tensor_expr(input_tensors[31])
        bw_b_o = self.get_tensor_expr(input_tensors[32])

        # ── Initial states ──
        fw_h = self.get_tensor_expr(input_tensors[35])
        fw_c = self.get_tensor_expr(input_tensors[36])
        bw_h = self.get_tensor_expr(input_tensors[37])
        bw_c = self.get_tensor_expr(input_tensors[38])

        # ── Unroll input ──
        in_expr = self.get_tensor_expr(input_tensors[0])
        in_shape = self.get_tensor_shape(input_tensors[0])
        if time_major:
            in_expr = relax.op.permute_dims(in_expr, [1, 0, 2])
            num_steps = int(in_shape[0])
        else:
            num_steps = int(in_shape[1])

        if num_steps == 1:
            steps = [relax.op.squeeze(in_expr, axis=[1])]
        else:
            splits = relax.op.split(in_expr, num_steps, axis=1)
            steps = [relax.op.squeeze(splits[i], axis=[1]) for i in range(num_steps)]

        one = relax.const(1.0, "float32")

        def _lstm_step(x_t, h, c, w_f_t, w_c_t, w_o_t, r_f_t, r_c_t, r_o_t, b_f, b_c, b_o):
            """Single LSTM step with coupled input-forget gate."""
            f = relax.op.sigmoid(
                relax.op.add(
                    relax.op.add(
                        relax.op.matmul(x_t, w_f_t),
                        relax.op.matmul(h, r_f_t),
                    ),
                    b_f,
                )
            )
            i = relax.op.subtract(one, f)
            g = self.convert_fused_activation_function(
                relax.op.add(
                    relax.op.add(relax.op.matmul(x_t, w_c_t), relax.op.matmul(h, r_c_t)),
                    b_c,
                ),
                fused_activation_fn,
            )
            o = relax.op.sigmoid(
                relax.op.add(
                    relax.op.add(
                        relax.op.matmul(x_t, w_o_t),
                        relax.op.matmul(h, r_o_t),
                    ),
                    b_o,
                )
            )
            c_new = relax.op.add(relax.op.multiply(f, c), relax.op.multiply(i, g))
            if cell_clip > 0.0:
                c_new = relax.op.clip(c_new, -cell_clip, cell_clip)
            h_new = relax.op.multiply(
                o, self.convert_fused_activation_function(c_new, fused_activation_fn)
            )
            if proj_clip > 0.0:
                h_new = relax.op.clip(h_new, -proj_clip, proj_clip)
            return h_new, c_new

        # ── Forward pass ──
        fw_outputs = []
        for x_t in steps:
            fw_h, fw_c = _lstm_step(
                x_t,
                fw_h,
                fw_c,
                fw_w_f_t,
                fw_w_c_t,
                fw_w_o_t,
                fw_r_f_t,
                fw_r_c_t,
                fw_r_o_t,
                fw_b_f,
                fw_b_c,
                fw_b_o,
            )
            fw_outputs.append(fw_h)

        # ── Backward pass ──
        bw_outputs = []
        for x_t in reversed(steps):
            bw_h, bw_c = _lstm_step(
                x_t,
                bw_h,
                bw_c,
                bw_w_f_t,
                bw_w_c_t,
                bw_w_o_t,
                bw_r_f_t,
                bw_r_c_t,
                bw_r_o_t,
                bw_b_f,
                bw_b_c,
                bw_b_o,
            )
            bw_outputs.append(bw_h)
        bw_outputs.reverse()

        fw_stacked = relax.op.stack(fw_outputs, axis=1)
        bw_stacked = relax.op.stack(bw_outputs, axis=1)
        if time_major:
            fw_stacked = relax.op.permute_dims(fw_stacked, [1, 0, 2])
            bw_stacked = relax.op.permute_dims(bw_stacked, [1, 0, 2])

        # Update state tensors in the expression table for subsequent ops.
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[35].tensor_idx),
            fw_h,
            force_override=True,
        )
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[36].tensor_idx),
            fw_c,
            force_override=True,
        )
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[37].tensor_idx),
            bw_h,
            force_override=True,
        )
        self.exp_tab.set_expr(
            get_tensor_name(self.subgraph, input_tensors[38].tensor_idx),
            bw_c,
            force_override=True,
        )

        if merge_outputs:
            return relax.op.concat([fw_stacked, bw_stacked], axis=-1)
        else:
            return relax.Tuple([fw_stacked, bw_stacked])

    def convert_batch_to_space_nd(self, op):
        """batch_to_space_nd implementation."""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)

        block_shape = to_int_list(self.get_tensor_value(input_tensors[1]))
        crops = self.get_tensor_value(input_tensors[2])
        crop_begin = to_int_list(crops[:, 0])
        crop_end = to_int_list(crops[:, 1])

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_shape = to_int_list(self.get_tensor_shape(output_tensor))
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())

        out = relax.op.call_dps_packed(
            _ExternFunc("topi.nn.batch_to_space_nd"),
            (
                in_expr,
                relax.ShapeExpr(block_shape),
                relax.ShapeExpr(crop_begin),
                relax.ShapeExpr(crop_end),
            ),
            ty_args=[relax.TensorType(output_shape, output_dtype)],
        )

        return out

    def convert_broadcast_to(self, op):
        """Convert TFLite BROADCAST_TO"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        data = self.get_tensor_expr(input_tensors[0])
        shape_tensor = input_tensors[1]
        if self.has_expr(shape_tensor.tensor_idx):
            shape_expr = self.get_expr(shape_tensor.tensor_idx)
            shape = self.bb.emit_output(relax.op.tensor_to_shape(shape_expr))
        else:
            shape = to_int_list(self.get_tensor_value(shape_tensor))
        return relax.op.broadcast_to(data, shape)

    def convert_embedding_lookup(self, op):
        """Convert TFLite EMBEDDING_LOOKUP"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        params = self.get_tensor_expr(input_tensors[0])
        indices_tensor = input_tensors[1]
        if self.has_expr(indices_tensor.tensor_idx):
            indices = relax.op.astype(self.get_expr(indices_tensor.tensor_idx), "int32")
        else:
            indices = self.get_tensor_expr(indices_tensor)
        return relax.op.take(params, indices, axis=0)

    def convert_embedding_lookup_sparse(self, op):
        """Convert TFLite EMBEDDING_LOOKUP_SPARSE."""
        from tflite.CombinerType import CombinerType
        from tflite.EmbeddingLookupSparseOptions import EmbeddingLookupSparseOptions
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 5, "EMBEDDING_LOOKUP_SPARSE should have 5 input tensors"
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "EMBEDDING_LOOKUP_SPARSE should have 1 output tensor"

        ids_tensor, indices_tensor, dense_shape_tensor, weights_tensor, params_tensor = (
            input_tensors
        )
        output_tensor = output_tensors[0]

        for tensor in input_tensors:
            assert not tensor.qnn_params, "Quantized input is not expected."

        assert ids_tensor.tensor.Type() == TensorType.INT32
        assert indices_tensor.tensor.Type() == TensorType.INT32
        assert dense_shape_tensor.tensor.Type() == TensorType.INT32
        assert weights_tensor.tensor.Type() == TensorType.FLOAT32
        assert params_tensor.tensor.Type() == TensorType.FLOAT32
        assert output_tensor.tensor.Type() == TensorType.FLOAT32

        ids_shape = to_int_list(self.get_tensor_shape(ids_tensor))
        indices_shape = to_int_list(self.get_tensor_shape(indices_tensor))
        dense_shape_shape = to_int_list(self.get_tensor_shape(dense_shape_tensor))
        weights_shape = to_int_list(self.get_tensor_shape(weights_tensor))
        params_shape = to_int_list(self.get_tensor_shape(params_tensor))

        assert len(ids_shape) == 1, "EMBEDDING_LOOKUP_SPARSE ids must be rank 1"
        assert len(indices_shape) == 2, "EMBEDDING_LOOKUP_SPARSE indices must be rank 2"
        assert len(dense_shape_shape) == 1, "EMBEDDING_LOOKUP_SPARSE dense_shape must be rank 1"
        assert len(weights_shape) == 1, "EMBEDDING_LOOKUP_SPARSE weights must be rank 1"
        assert len(params_shape) >= 2, "EMBEDDING_LOOKUP_SPARSE params must be rank >= 2"
        assert indices_shape[0] == ids_shape[0], (
            "EMBEDDING_LOOKUP_SPARSE ids and indices must agree on lookup count"
        )
        assert weights_shape[0] == ids_shape[0], (
            "EMBEDDING_LOOKUP_SPARSE ids and weights must agree on lookup count"
        )

        if self.has_expr(dense_shape_tensor.tensor_idx):
            raise NotImplementedError(
                "TFLite EMBEDDING_LOOKUP_SPARSE with runtime dense_shape is not supported."
            )

        dense_shape = to_int_list(self.get_tensor_value(dense_shape_tensor))
        lookup_rank = indices_shape[1]
        assert len(dense_shape) == lookup_rank, (
            "EMBEDDING_LOOKUP_SPARSE dense_shape length must match indices width"
        )
        assert lookup_rank >= 1, "EMBEDDING_LOOKUP_SPARSE indices width must be positive"
        if not self.has_expr(ids_tensor.tensor_idx):
            ids_value = self.get_tensor_value(ids_tensor)
            if np.any(ids_value < 0):
                raise NotImplementedError(
                    "TFLite EMBEDDING_LOOKUP_SPARSE with negative ids is not supported."
                )

        params = self.get_tensor_expr(params_tensor)
        ids = self.get_tensor_expr(ids_tensor)
        weights = self.get_tensor_expr(weights_tensor)
        indices = self.get_tensor_expr(indices_tensor)

        ids = relax.op.astype(ids, "int32")
        lookup = relax.op.take(params, ids, axis=0)

        embedding_tail_shape = params_shape[1:]
        output_prefix_shape = dense_shape[:-1]
        output_shape = output_prefix_shape + embedding_tail_shape

        # Aggregation buckets are defined by every sparse index dimension except the last one.
        bucket_indices = relax.op.strided_slice(indices, axes=[1], begin=[0], end=[lookup_rank - 1])

        weight_expand_shape = [ids_shape[0]] + [1] * len(embedding_tail_shape)
        weighted_lookup = relax.op.multiply(lookup, relax.op.reshape(weights, weight_expand_shape))

        value_base = relax.const(np.zeros(output_shape, dtype=np.float32), "float32")
        summed_lookup = relax.op.scatter_nd(value_base, bucket_indices, weighted_lookup, "add")

        op_options = op.BuiltinOptions()
        sparse_options = EmbeddingLookupSparseOptions()
        sparse_options.Init(op_options.Bytes, op_options.Pos)
        combiner = sparse_options.Combiner()
        if combiner == CombinerType.SUM:
            return summed_lookup

        count_shape = output_prefix_shape
        count_base = relax.const(np.zeros(count_shape, dtype=np.float32), "float32")
        bucket_count_updates = relax.const(np.ones(ids_shape, dtype=np.float32), "float32")
        bucket_counts = relax.op.scatter_nd(count_base, bucket_indices, bucket_count_updates, "add")
        if combiner == CombinerType.MEAN:
            denominator_updates = weights
        elif combiner == CombinerType.SQRTN:
            denominator_updates = relax.op.multiply(weights, weights)
        else:
            raise NotImplementedError(
                f"Unsupported TFLite EMBEDDING_LOOKUP_SPARSE combiner value {combiner}"
            )

        denominator = relax.op.scatter_nd(count_base, bucket_indices, denominator_updates, "add")
        if combiner == CombinerType.SQRTN:
            denominator = relax.op.sqrt(denominator)

        broadcast_shape = count_shape + [1] * len(embedding_tail_shape)
        denominator = relax.op.reshape(denominator, broadcast_shape)
        denominator = relax.op.broadcast_to(denominator, output_shape)
        normalized = relax.op.divide(summed_lookup, denominator)
        bucket_counts = relax.op.reshape(bucket_counts, broadcast_shape)
        bucket_counts = relax.op.broadcast_to(bucket_counts, output_shape)
        return relax.op.where(
            relax.op.greater(bucket_counts, relax.const(0.0, "float32")), normalized, value_base
        )

    def convert_batch_matmul(self, op):
        """batch_matmul implementation."""

        from tflite.BatchMatMulOptions import BatchMatMulOptions

        input_tensors = self.get_input_tensors(op)

        assert len(input_tensors) == 2, "two input tensor arguments expected"

        if self.is_quantized(op):
            raise NotImplementedError(
                "Quantized BATCH_MATMUL is not yet supported in the Relax frontend"
            )

        batch_matmul_options = BatchMatMulOptions()
        op_options = op.BuiltinOptions()
        batch_matmul_options.Init(op_options.Bytes, op_options.Pos)

        input_a = self.get_expr(input_tensors[0].tensor_idx)
        input_b = self.get_expr(input_tensors[1].tensor_idx)

        shape_a = list(input_a.ty.shape)
        shape_b = list(input_b.ty.shape)
        rank_a = len(shape_a)
        rank_b = len(shape_b)

        if rank_a > 2 or rank_b > 2:
            # Broadcast batch dimensions
            new_a_shape = [1] * max(0, rank_b - rank_a) + [int(s) for s in shape_a]
            new_b_shape = [1] * max(0, rank_a - rank_b) + [int(s) for s in shape_b]
            max_rank = max(rank_a, rank_b)

            batch_shape = [max(new_a_shape[i], new_b_shape[i]) for i in range(max_rank - 2)]

            a_broadcast = batch_shape + [int(shape_a[-2]), int(shape_a[-1])]
            b_broadcast = batch_shape + [int(shape_b[-2]), int(shape_b[-1])]

            if [int(s) for s in shape_a] != a_broadcast:
                input_a = relax.op.broadcast_to(input_a, a_broadcast)
            if [int(s) for s in shape_b] != b_broadcast:
                input_b = relax.op.broadcast_to(input_b, b_broadcast)

            input_a = self.flatten_to_nd(input_a, 3)
            input_b = self.flatten_to_nd(input_b, 3)

            adj_x = batch_matmul_options.AdjX()
            adj_y = batch_matmul_options.AdjY()

            if adj_x:
                input_a = relax.op.permute_dims(input_a, [0, 2, 1])
            if adj_y:
                input_b = relax.op.permute_dims(input_b, [0, 2, 1])

            output = relax.op.matmul(input_a, input_b)

            # Compute output matmul dims from original shapes
            m_dim = int(shape_a[-1]) if adj_x else int(shape_a[-2])
            n_dim = int(shape_b[-2]) if adj_y else int(shape_b[-1])
            final_shape = [int(s) for s in shape_a[: rank_a - 2]] + [m_dim, n_dim]
            return relax.op.reshape(output, final_shape)

        # rank <= 2: use matmul directly
        if batch_matmul_options.AdjX():
            input_a = relax.op.permute_dims(input_a)
        if batch_matmul_options.AdjY():
            input_b = relax.op.permute_dims(input_b)
        return relax.op.matmul(input_a, input_b)

    def convert_space_to_batch_nd(self, op):
        """space_to_batch_nd implementation."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)

        block_shape = to_int_list(self.get_tensor_value(input_tensors[1]))
        paddings = self.get_tensor_value(input_tensors[2])
        pad_before = to_int_list(paddings[:, 0])
        pad_after = to_int_list(paddings[:, 1])

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_shape = to_int_list(self.get_tensor_shape(output_tensor))
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())

        out = relax.op.call_dps_packed(
            _ExternFunc("topi.nn.space_to_batch_nd"),
            (
                in_expr,
                relax.ShapeExpr(block_shape),
                relax.ShapeExpr(pad_before),
                relax.ShapeExpr(pad_after),
                0.0,
            ),
            ty_args=[relax.TensorType(output_shape, output_dtype)],
        )

        return out

    def convert_depth_to_space(self, op):
        """Convert TFLite DEPTH_TO_SPACE"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.DepthToSpaceOptions import DepthToSpaceOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.DepthToSpaceOptions
        op_options = op.BuiltinOptions()
        depth_to_space_options = DepthToSpaceOptions()
        depth_to_space_options.Init(op_options.Bytes, op_options.Pos)
        block_size = depth_to_space_options.BlockSize()

        # TFLite uses NHWC layout: (N, H, W, C) -> (N, H*bs, W*bs, C/(bs*bs))
        input_shape = self.get_tensor_shape(input_tensor)
        n, h, w, c = input_shape
        out_c = c // (block_size**2)
        out = relax.op.reshape(in_expr, (n, h, w, block_size, block_size, out_c))
        out = relax.op.permute_dims(out, [0, 1, 3, 2, 4, 5])
        out = relax.op.reshape(out, (n, h * block_size, w * block_size, out_c))

        return out

    def convert_space_to_depth(self, op):
        """Convert TFLite SPACE_TO_DEPTH"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.SpaceToDepthOptions import SpaceToDepthOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.SpaceToDepthOptions
        op_options = op.BuiltinOptions()
        space_to_depth_options = SpaceToDepthOptions()
        space_to_depth_options.Init(op_options.Bytes, op_options.Pos)
        block_size = space_to_depth_options.BlockSize()

        # TFLite uses NHWC layout: (N, H, W, C) -> (N, H/bs, W/bs, C*bs*bs)
        input_shape = self.get_tensor_shape(input_tensor)
        n, h, w, c = input_shape
        out = relax.op.reshape(
            in_expr, (n, h // block_size, block_size, w // block_size, block_size, c)
        )
        out = relax.op.permute_dims(out, [0, 1, 3, 2, 4, 5])
        out = relax.op.reshape(
            out, (n, h // block_size, w // block_size, c * block_size * block_size)
        )

        return out

    def convert_sparse_to_dense(self, op):
        """Convert TFLite SPARSE_TO_DENSE"""
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 4, "input tensors length should be 4"

        indices, values = input_tensors[0], input_tensors[2]
        default_value = input_tensors[3]
        output_shape = input_tensors[1]

        for t in input_tensors:
            assert not t.qnn_params, "Quantized input is not expected."

        for t in [indices, output_shape]:
            t_type = t.tensor.Type()
            assert t_type in (TensorType.INT32, TensorType.INT64)

        output_tensors = self.get_output_tensors(op)
        output_tensor = output_tensors[0]
        output_shape_val = to_int_list(self.get_tensor_shape(output_tensor))
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())

        indices_expr = self.get_tensor_expr(indices)
        values_expr = self.get_tensor_expr(values)
        default_value_expr = self.get_tensor_expr(default_value)
        output_shape_expr = relax.const(list(self.get_tensor_value(output_shape)), "int32")

        out = relax.op.call_dps_packed(
            _ExternFunc("topi.sparse_to_dense"),
            (indices_expr, output_shape_expr, values_expr, default_value_expr),
            ty_args=[relax.TensorType(output_shape_val, output_dtype)],
        )

        return out

    def convert_prelu(self, op):
        """Convert TFLite PReLU"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        input_tensor = input_tensors[0]
        alpha_tensor = input_tensors[1]
        data_shape = to_int_list(self.get_tensor_shape(input_tensor))
        in_expr = self.get_tensor_expr(input_tensor)
        alpha_expr = self.get_tensor_expr(alpha_tensor)
        alpha_expr = self.bb.normalize(relax.op.broadcast_to(alpha_expr, data_shape))
        alpha_expr = self.bb.normalize(relax.op.reshape(alpha_expr, [-1]))
        out = relax.op.nn.prelu(_op.reshape(in_expr, [-1]), alpha_expr, axis=0)
        out = relax.op.reshape(out, data_shape)
        return out

    def convert_transpose_conv(self, op):
        """Convert TFLite TRANSPOSE_CONV"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.Padding import Padding
        from tflite.TensorType import TensorType
        from tflite.TransposeConvOptions import TransposeConvOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 3, "input tensors length should be >= 3"

        # Input (data) Tensor. NHWC layout
        input_tensor = input_tensors[2]
        _, _, _, input_c = to_int_list(self.get_tensor_shape(input_tensor))
        # Weights tensor. TFLite uses OHWI layout
        weights_tensor = input_tensors[1]
        out_channels, kernel_h, kernel_w, in_channels = to_int_list(
            self.get_tensor_shape(weights_tensor)
        )

        assert input_c == in_channels, (
            "Input channel in the filter should match to channel in the input"
        )
        # output_shape Tensor. NHWC layout
        output_shape_tensor = input_tensors[0]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        assert op.BuiltinOptionsType() == BuiltinOptions.TransposeConvOptions
        op_options = op.BuiltinOptions()
        deconv_options = TransposeConvOptions()
        deconv_options.Init(op_options.Bytes, op_options.Pos)

        padding = deconv_options.Padding()
        stride_h = deconv_options.StrideH()
        stride_w = deconv_options.StrideW()
        assert padding in (
            Padding.VALID,
            Padding.SAME,
        ), f"Padding format {padding} is not supported for operator TRANSPOSE_CONV"

        # Data
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # Weights
        weights_tensor_type = weights_tensor.tensor.Type()
        # weights tensor type should be UINT8 (quantization) or FLOAT32
        assert weights_tensor_type in (
            TensorType.INT8,
            TensorType.UINT8,
            TensorType.FLOAT32,
        )
        weight_tensor_type_str = self.get_tensor_type_str(weights_tensor_type)

        if self.has_expr(weights_tensor.tensor_idx):
            weight_expr_iohw = self.get_expr(weights_tensor.tensor_idx)
            weight_expr_iohw = relax.op.permute_dims(weight_expr_iohw, axes=(3, 0, 1, 2))
        else:
            weight_value_ohwi = self.get_tensor_value_or_prefetched(weights_tensor)
            # Relax kernel_layout should be OIHW
            # Relax weights layout should be different from kernel_layout - it should be IOHW
            weight_value_iohw = np.transpose(weight_value_ohwi, (3, 0, 1, 2))
            weight_expr_iohw = self.exp_tab.new_const(
                weight_value_iohw,
                dtype=weight_tensor_type_str,
                source_name=weights_tensor.tensor.Name(),
            )

        # Output shape value
        output_shape_value = self.get_tensor_value(output_shape_tensor)
        # Relax expects filter output channel to match to output tensor channel.
        assert out_channels == output_shape_value[3], (
            "Output channel in the filter should match to channel in the output_shape"
        )

        if padding == Padding.SAME:
            output_h, output_w = output_shape_value[1], output_shape_value[2]
            pad_top, pad_bottom = get_pad_value(output_h, kernel_h, stride_h)
            pad_left, pad_right = get_pad_value(output_w, kernel_w, stride_w)
            padding = (pad_top, pad_left, pad_bottom, pad_right)
        else:
            padding = (0, 0, 0, 0)

        if input_tensor.qnn_params:
            in_f32 = self.dequantize(in_expr, input_tensor)
            weight_axis = weights_tensor.qnn_params["axis"]
            if weight_axis != 0:
                raise AttributeError(
                    f"TransposeConv weight QuantizedDimension() must be 0 "
                    f"(output-channel axis in OHWI layout), got {weight_axis}"
                )
            w_f32 = relax.op.dequantize(
                weight_expr_iohw,
                scale=weights_tensor.qnn_params["scale"],
                zero_point=weights_tensor.qnn_params["zero_point"],
                axis=1,
            )
            out = relax.op.nn.conv2d_transpose(
                in_f32,
                w_f32,
                strides=(stride_h, stride_w),
                padding=padding,
                data_layout="NHWC",
                kernel_layout="IOHW",
                out_dtype="float32",
            )
        else:
            out = relax.op.nn.conv2d_transpose(
                in_expr,
                weight_expr_iohw,
                strides=(stride_h, stride_w),
                padding=padding,
                data_layout="NHWC",
                kernel_layout="IOHW",
                out_dtype=output_tensor_type_str,
            )

        # Checking if there is a fused bias
        if len(input_tensors) == 4:
            bias_tensor = input_tensors[3]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (quantization) or FLOAT32
            assert bias_tensor_type in (
                TensorType.INT32,
                TensorType.INT64,
                TensorType.FLOAT32,
            )
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            if self.has_expr(bias_tensor.tensor_idx):
                bias_expr = self.get_expr(bias_tensor.tensor_idx)
            else:
                bias_expr = self.exp_tab.new_const(
                    self.get_tensor_value(bias_tensor),
                    dtype=bias_tensor_type_str,
                    source_name=bias_tensor.tensor.Name(),
                )
            if bias_tensor.qnn_params:
                bias_expr = self.dequantize(bias_expr, bias_tensor)
            elif input_tensor.qnn_params and bias_tensor_type in (
                TensorType.INT32,
                TensorType.INT64,
            ):
                bias_scale = relax.op.multiply(
                    input_tensor.qnn_params["scale"],
                    weights_tensor.qnn_params["scale"],
                )
                bias_expr = relax.op.dequantize(
                    bias_expr,
                    scale=bias_scale,
                    zero_point=relax.const(0, "int32"),
                    axis=0,
                )
            out = relax.op.add(out, bias_expr)

        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_quantize(self, op):
        """Convert TFLite Quantize"""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
        in_expr = self.get_tensor_expr(input_tensor)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # The output must be quantized
        assert output_tensor.qnn_params

        # TFLite Quantize op can also act as Requantize op
        if input_tensor_type_str == "float32":
            out = self.quantize(in_expr, output_tensor)
        else:
            in_f32 = self.dequantize(in_expr, input_tensor)
            out = self.quantize(in_f32, output_tensor)
        return out

    def convert_dequantize(self, op):
        """Convert TFLite Dequantize"""
        from tflite.TensorType import TensorType

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]

        if input_tensor.tensor.Type() == TensorType.FLOAT16:
            dtype = self.get_tensor_type_str(input_tensor.tensor.Type())
            input_value = self.get_tensor_value(input_tensor)
            in_expr = self.exp_tab.new_const(
                input_value, dtype=dtype, source_name=input_tensor.tensor.Name()
            )
            out = relax.op.astype(in_expr, "float32")
            return out

        in_expr = self.get_expr(input_tensor.tensor_idx)

        # The input must be quantized
        assert input_tensor.qnn_params
        # Dequantize the input.
        out = self.dequantize(in_expr, input_tensor)

        return out

    def convert_dilate(self, op):
        """Convert TFLite DILATE"""
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"
        assert len(output_tensors) == 1, "output tensors length should be 1"

        in_expr = self.get_tensor_expr(input_tensors[0])
        in_shape = to_int_list(self.get_tensor_shape(input_tensors[0]))
        in_dtype = self.get_tensor_type_str(input_tensors[0].tensor.Type())
        n_dims = len(in_shape)

        dilations_tensor = input_tensors[1]
        padding_expr = self.get_tensor_expr(input_tensors[2])

        # Runtime dilations bind tensor values to TIR Vars for symbolic
        # per-axis math.
        if self.has_expr(dilations_tensor.tensor_idx):
            dilations_expr = self.get_expr(dilations_tensor.tensor_idx)
            dilations_expr = self.bb.match_cast(dilations_expr, relax.TensorType([n_dims], "int32"))
            dilations_int64 = self.bb.normalize(relax.op.astype(dilations_expr, "int64"))
            shape_var = self.bb.emit(relax.op.tensor_to_shape(dilations_int64))
            stride_vars = [tirx.Var(f"dilate_stride_{i}", "int64") for i in range(n_dims)]
            self.bb.match_cast(shape_var, relax.ShapeType(stride_vars))
            strides = stride_vars
        else:
            strides = to_int_list(self.get_tensor_value(dilations_tensor))

        # Per axis: reshape to add a size-1 stride-axis, concat (s-1) padding
        # values along it, reshape to merge axes (length d*s), trim trailing
        # pad to TFLite's output dim formula (d-1)*s + 1.
        result = in_expr
        current_shape = list(in_shape)
        axes = list(range(n_dims))
        ones = [1] * n_dims
        for axis in range(n_dims):
            d = current_shape[axis]
            s = strides[axis]
            expanded_shape = current_shape[: axis + 1] + [1] + current_shape[axis + 1 :]
            expanded = relax.op.reshape(result, expanded_shape)
            pad_shape = list(expanded_shape)
            pad_shape[axis + 1] = s - 1
            pad = relax.op.full(pad_shape, padding_expr, dtype=in_dtype)
            concatted = relax.op.concat([expanded, pad], axis=axis + 1)
            merged_shape = list(current_shape)
            merged_shape[axis] = d * s
            merged = relax.op.reshape(concatted, merged_shape)
            # (d - 1) * s + 1 is the output dim along this axis.
            final_dim = (d - 1) * s + 1
            end = list(merged_shape)
            end[axis] = final_dim
            result = relax.op.strided_slice(
                merged, axes=axes, begin=[0] * n_dims, end=end, strides=ones
            )
            current_shape = list(merged_shape)
            current_shape[axis] = final_dim

        return result

    def convert_detection_postprocess(self, op):
        """Convert TFLite_Detection_PostProcess"""
        flexbuffer = op.CustomOptionsAsNumpy().tobytes()
        custom_options = FlexBufferDecoder(flexbuffer).decode()

        use_regular_nms = bool(custom_options.get("use_regular_nms", False))

        required_attrs = [
            "num_classes",
            "max_detections",
            "detections_per_class",
            "nms_iou_threshold",
            "nms_score_threshold",
            "x_scale",
            "y_scale",
            "w_scale",
            "h_scale",
        ]
        missing_attrs = [key for key in required_attrs if key not in custom_options]
        if missing_attrs:
            raise ValueError(
                "DETECTION_POSTPROCESS custom options miss required attributes: "
                + ", ".join(missing_attrs)
            )

        num_classes = int(custom_options["num_classes"])
        max_detections = int(custom_options["max_detections"])
        detections_per_class = int(custom_options["detections_per_class"])
        iou_threshold = float(custom_options["nms_iou_threshold"])
        score_threshold = float(custom_options["nms_score_threshold"])
        x_scale = float(custom_options["x_scale"])
        y_scale = float(custom_options["y_scale"])
        w_scale = float(custom_options["w_scale"])
        h_scale = float(custom_options["h_scale"])

        if num_classes <= 0:
            raise ValueError("DETECTION_POSTPROCESS requires num_classes > 0.")
        if max_detections <= 0:
            raise ValueError("DETECTION_POSTPROCESS requires max_detections > 0.")
        if detections_per_class <= 0:
            raise ValueError("DETECTION_POSTPROCESS requires detections_per_class > 0.")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("DETECTION_POSTPROCESS requires nms_iou_threshold in [0, 1].")
        if x_scale <= 0.0 or y_scale <= 0.0 or w_scale <= 0.0 or h_scale <= 0.0:
            raise ValueError("DETECTION_POSTPROCESS requires x/y/w/h_scale to be > 0.")

        inputs = self.get_input_tensors(op)
        assert len(inputs) == 3, "inputs length should be 3"
        cls_pred = self.get_expr(inputs[1].tensor_idx)
        loc_prob = self.get_expr(inputs[0].tensor_idx)
        batch_size = inputs[1].tensor.Shape(0)
        input_num_classes = int(inputs[1].tensor.Shape(2))
        label_offset = input_num_classes - num_classes
        if label_offset not in (0, 1):
            raise ValueError(
                "DETECTION_POSTPROCESS class predictions must contain num_classes "
                "or num_classes + 1 entries"
            )
        anchor_values = self.get_tensor_value(inputs[2])
        anchor_boxes = len(anchor_values)
        anchor_type = self.get_tensor_type_str(inputs[2].tensor.Type())
        anchor_expr = self.exp_tab.new_const(
            anchor_values, dtype=anchor_type, source_name=inputs[2].tensor.Name()
        )

        if inputs[0].qnn_params:
            loc_prob = self.dequantize(loc_prob, inputs[0])
        if inputs[1].qnn_params:
            cls_pred = self.dequantize(cls_pred, inputs[1])
        if inputs[2].qnn_params:
            anchor_expr = self.dequantize(anchor_expr, inputs[2])

        if label_offset:
            cls_pred = relax.op.strided_slice(
                cls_pred,
                axes=[2],
                begin=[label_offset],
                end=[label_offset + num_classes],
            )

        # loc_prob coords are in yxhw format
        # need to convert to xywh
        loc_coords = relax.op.split(loc_prob, 4, axis=2)
        loc_prob = relax.op.concat(
            [loc_coords[1], loc_coords[0], loc_coords[3], loc_coords[2]], axis=2
        )
        # reshape loc_prob tensor so is can be consumed by
        # multibox_transform_loc
        loc_prob = relax.op.reshape(loc_prob, [batch_size, anchor_boxes * 4])

        # anchor coords are in yxhw format
        # need to convert to ltrb
        anchor_coords = relax.op.split(anchor_expr, 4, axis=1)
        anchor_y = anchor_coords[0]
        anchor_x = anchor_coords[1]
        anchor_h = anchor_coords[2]
        anchor_w = anchor_coords[3]
        plus_half = relax.const(0.5, dtype="float32")
        minus_half = relax.const(-0.5, dtype="float32")
        anchor_l = relax.op.add(anchor_x, relax.op.multiply(anchor_w, minus_half))
        anchor_r = relax.op.add(anchor_x, relax.op.multiply(anchor_w, plus_half))
        anchor_t = relax.op.add(anchor_y, relax.op.multiply(anchor_h, minus_half))
        anchor_b = relax.op.add(anchor_y, relax.op.multiply(anchor_h, plus_half))
        anchor_expr = relax.op.concat([anchor_l, anchor_t, anchor_r, anchor_b], axis=1)
        anchor_expr = relax.op.expand_dims(anchor_expr, 0)

        # attributes for multibox_transform_loc
        multibox_transform_loc_attrs = {}
        multibox_transform_loc_attrs["clip"] = False
        multibox_transform_loc_attrs["threshold"] = 0.0 if use_regular_nms else score_threshold
        multibox_transform_loc_attrs["variances"] = (
            1 / x_scale,
            1 / y_scale,
            1 / w_scale,
            1 / h_scale,
        )
        # TFLite DetectionPostProcess consumes probabilities, not logits.  Any
        # optional background class was sliced above, so class 0 is now a real
        # foreground class and must be kept for both NMS implementations.
        multibox_transform_loc_attrs["keep_background"] = True
        multibox_transform_loc_attrs["apply_softmax"] = False

        multibox_res = self.bb.emit(
            relax.op.vision.multibox_transform_loc(
                # reshape cls_pred so it can be consumed by
                # multibox_transform_loc
                relax.op.permute_dims(cls_pred, [0, 2, 1]),
                loc_prob,
                anchor_expr,
                **multibox_transform_loc_attrs,
            )
        )
        transformed_boxes = self.bb.emit(relax.TupleGetItem(multibox_res, 0))
        transformed_scores = self.bb.emit(relax.TupleGetItem(multibox_res, 1))

        if use_regular_nms:
            nms_out = self.bb.emit(
                relax.op.vision.all_class_non_max_suppression(
                    transformed_boxes,
                    transformed_scores,
                    relax.const(detections_per_class, "int64"),
                    relax.const(iou_threshold, "float32"),
                    relax.const(score_threshold, "float32"),
                    output_format="tensorflow",
                )
            )
            selected_indices = self.bb.emit(relax.TupleGetItem(nms_out, 0))
            selected_scores = self.bb.emit(relax.TupleGetItem(nms_out, 1))
            num_detections = self.bb.emit(relax.TupleGetItem(nms_out, 2))
            class_id_from_score = None
        else:
            topk_res = self.bb.emit(
                relax.op.topk(transformed_scores, k=1, axis=1, ret_type="both", largest=True)
            )
            max_scores = self.bb.emit(relax.TupleGetItem(topk_res, 0))
            class_id_from_score = self.bb.emit(relax.TupleGetItem(topk_res, 1))
            nms_out = self.bb.emit(
                relax.op.vision.all_class_non_max_suppression(
                    transformed_boxes,
                    max_scores,
                    relax.const(max_detections, "int64"),
                    relax.const(iou_threshold, "float32"),
                    relax.const(score_threshold, "float32"),
                    output_format="tensorflow",
                )
            )
            selected_indices = self.bb.emit(relax.TupleGetItem(nms_out, 0))
            selected_scores = self.bb.emit(relax.TupleGetItem(nms_out, 1))
            num_detections = self.bb.emit(relax.TupleGetItem(nms_out, 2))
            class_id_from_score = relax.op.squeeze(class_id_from_score, axis=[1])

        selected_score_slots = selected_scores.ty.shape.values[1]
        selected_detection_positions = relax.op.expand_dims(
            relax.op.arange(selected_score_slots, dtype="int64"), axis=0
        )
        selected_valid_detection_mask = relax.op.less(
            selected_detection_positions, relax.op.expand_dims(num_detections, axis=1)
        )
        masked_selected_scores = relax.op.where(
            selected_valid_detection_mask,
            selected_scores,
            relax.const(-1.0, "float32"),
        )
        topk_scores_res = self.bb.emit(
            relax.op.topk(
                masked_selected_scores, k=max_detections, axis=1, ret_type="both", largest=True
            )
        )
        detection_scores = self.bb.emit(relax.TupleGetItem(topk_scores_res, 0))
        top_positions = self.bb.emit(relax.TupleGetItem(topk_scores_res, 1))
        num_detections = relax.op.minimum(
            num_detections, relax.const([max_detections], dtype="int64")
        )
        detection_positions = relax.op.expand_dims(
            relax.op.arange(max_detections, dtype="int64"), axis=0
        )
        valid_detection_mask = relax.op.less(
            detection_positions, relax.op.expand_dims(num_detections, axis=1)
        )
        top_positions_expanded = relax.op.expand_dims(top_positions, axis=2)
        top_positions_for_pairs = relax.op.repeat(top_positions_expanded, 2, axis=2)
        top_index_pairs = relax.op.gather_elements(
            selected_indices, top_positions_for_pairs, axis=1
        )
        top_box_ids = relax.op.squeeze(
            relax.op.strided_slice(top_index_pairs, axes=[2], begin=[1], end=[2]),
            axis=[2],
        )
        top_box_ids_for_gather = relax.op.expand_dims(relax.op.astype(top_box_ids, "int64"), axis=2)
        detection_boxes = relax.op.gather_nd(
            transformed_boxes, top_box_ids_for_gather, batch_dims=1
        )

        if use_regular_nms:
            detection_classes = relax.op.squeeze(
                relax.op.strided_slice(top_index_pairs, axes=[2], begin=[0], end=[1]),
                axis=[2],
            )
            detection_classes = relax.op.astype(detection_classes, "int32")
        else:
            top_box_ids_for_class = relax.op.expand_dims(
                relax.op.astype(top_box_ids, "int64"), axis=2
            )
            detection_classes = relax.op.gather_nd(
                class_id_from_score, top_box_ids_for_class, batch_dims=1
            )

        detection_mask = relax.op.expand_dims(valid_detection_mask, axis=2)
        detection_boxes = relax.op.where(
            detection_mask,
            detection_boxes,
            relax.op.zeros((batch_size, max_detections, 4), dtype="float32"),
        )
        detection_classes = relax.op.where(
            valid_detection_mask,
            detection_classes,
            relax.op.zeros((batch_size, max_detections), dtype="int32"),
        )
        detection_scores = relax.op.where(
            valid_detection_mask,
            detection_scores,
            relax.op.zeros((batch_size, max_detections), dtype="float32"),
        )
        detection_classes = relax.op.astype(detection_classes, "float32")
        num_detections = relax.op.astype(num_detections, "float32")
        return relax.Tuple([detection_boxes, detection_classes, detection_scores, num_detections])

    def convert_nms_v4(self, op):
        """Convert TFLite NonMaxSuppressionV4"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 5, "input tensor length should be 5"

        boxes = self.get_tensor_expr(input_tensors[0])
        scores = self.get_tensor_expr(input_tensors[1])

        max_output_size = self.get_tensor_value(input_tensors[2])
        iou_threshold = self.get_tensor_value(input_tensors[3])
        score_threshold = self.get_tensor_value(input_tensors[4])

        if isinstance(max_output_size, np.ndarray):
            assert max_output_size.size == 1, "only one value is expected."
            max_output_size = int(max_output_size)

        if isinstance(iou_threshold, np.ndarray):
            assert iou_threshold.size == 1, "only one value is expected."
            iou_threshold = float(iou_threshold)

        if isinstance(score_threshold, np.ndarray):
            assert score_threshold.size == 1, "only one value is expected."
            score_threshold = float(score_threshold)

        scores_expand = relax.op.expand_dims(scores, axis=-1)
        data = relax.op.concat([scores_expand, boxes], axis=-1)
        data = relax.op.expand_dims(data, axis=0)

        valid_counts_ret = relax.op.vision.get_valid_counts(
            data, score_threshold=score_threshold, id_index=-1, score_index=0
        )
        count = valid_counts_ret[0]
        data = valid_counts_ret[1]
        indices = valid_counts_ret[2]

        nms_ret = relax.op.vision.non_max_suppression(
            data=data,
            valid_count=count,
            indices=indices,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            force_suppress=True,
            top_k=-1,
            coord_start=1,
            score_index=0,
            id_index=-1,
            return_indices=True,
            invalid_to_bottom=False,
        )

        selected_indices = relax.op.squeeze(nms_ret[0], axis=[0])
        selected_indices = relax.op.strided_slice(
            selected_indices, axes=[0], begin=[0], end=[max_output_size]
        )
        num_valid = relax.op.reshape(nms_ret[1], [])

        return relax.Tuple([selected_indices, num_valid])

    def convert_nms_v5(self, op):
        """Convert TFLite NonMaxSuppressionV5"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 6, "input tensor length should be 6"

        boxes = self.get_tensor_expr(input_tensors[0])
        scores = self.get_tensor_expr(input_tensors[1])

        max_output_size = self.get_tensor_value(input_tensors[2])
        iou_threshold = self.get_tensor_value(input_tensors[3])
        score_threshold = self.get_tensor_value(input_tensors[4])
        soft_nms_sigma = self.get_tensor_value(input_tensors[5])

        if isinstance(max_output_size, np.ndarray):
            assert max_output_size.size == 1, "only one value is expected."
            max_output_size = int(max_output_size)

        if isinstance(iou_threshold, np.ndarray):
            assert iou_threshold.size == 1, "only one value is expected."
            iou_threshold = float(iou_threshold)

        if isinstance(score_threshold, np.ndarray):
            assert score_threshold.size == 1, "only one value is expected."
            score_threshold = float(score_threshold)

        if isinstance(soft_nms_sigma, np.ndarray):
            assert soft_nms_sigma.size == 1, "only one value is expected."
            soft_nms_sigma = float(soft_nms_sigma)

        scores_expand = relax.op.expand_dims(scores, axis=-1)
        data = relax.op.concat([scores_expand, boxes], axis=-1)
        data = relax.op.expand_dims(data, axis=0)

        valid_counts_ret = relax.op.vision.get_valid_counts(
            data, score_threshold=score_threshold, id_index=-1, score_index=0
        )
        count = valid_counts_ret[0]
        data = valid_counts_ret[1]
        indices = valid_counts_ret[2]

        nms_ret = relax.op.vision.non_max_suppression(
            data=data,
            valid_count=count,
            indices=indices,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            force_suppress=True,
            top_k=-1,
            coord_start=1,
            score_index=0,
            id_index=-1,
            return_indices=True,
            invalid_to_bottom=False,
            soft_nms_sigma=soft_nms_sigma,
            score_threshold=score_threshold,
        )

        if soft_nms_sigma > 0.0:
            processed_data = relax.op.squeeze(nms_ret[0], axis=[0])
            indices_from_nms = nms_ret[1]
            num_valid_from_nms = nms_ret[2]
        else:
            indices_from_nms = nms_ret[0]
            num_valid_from_nms = nms_ret[1]

        selected_indices = relax.op.squeeze(indices_from_nms, axis=[0])
        selected_indices = relax.op.strided_slice(
            selected_indices, axes=[0], begin=[0], end=[max_output_size]
        )
        num_valid = relax.op.reshape(num_valid_from_nms, [])

        if soft_nms_sigma > 0.0:
            # Extract decayed scores from the processed data (score_index=0)
            selected_scores = relax.op.strided_slice(processed_data, axes=[1], begin=[0], end=[1])
            selected_scores = relax.op.squeeze(selected_scores, axis=[1])
            selected_scores = relax.op.strided_slice(
                selected_scores, axes=[0], begin=[0], end=[max_output_size]
            )
            selected_scores = relax.op.clip(
                selected_scores, min=0.0, max=float(np.finfo("float32").max)
            )
        else:
            # Clamp out-of-bound padded indices to prevent take() crash.
            num_boxes = int(self.get_tensor_shape(input_tensors[0])[0])
            safe_indices = relax.op.clip(selected_indices, min=0, max=num_boxes - 1)
            selected_scores = relax.op.take(scores, safe_indices, axis=0)

        out = relax.Tuple([selected_indices, selected_scores, num_valid])
        return out

    def convert_expand_dims(self, op):
        """Convert TFLite EXPAND_DIMS"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        if input_tensors[0].qnn_params:
            # Check that input and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            assert self.has_same_qnn_params(input_tensors[0], output_tensors[0]), (
                "TFLite EXPAND_DIMS requires input and output tensors' \
                    scale and zero points to be equal"
            )

        input_expr = self.get_tensor_expr(input_tensors[0])
        axis = self.get_tensor_value(input_tensors[1])
        if isinstance(axis, np.ndarray):
            assert axis.size == 1, "only one value is expected."
            axis = int(axis.flat[0])

        ndims = len(input_tensors[0].tensor.ShapeAsNumpy())
        assert -1 - ndims <= axis <= ndims, "axis out of range"

        out = relax.op.expand_dims(input_expr, axis, 1)

        return out

    def convert_one_hot(self, op):
        """Convert TFLite ONE_HOT"""

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.OneHotOptions import OneHotOptions

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 4, "Input tensor's length should be 4"

        # Ensuring input isn't quantized
        assert all(not i.qnn_params for i in input_tensors), "Quantized input is not expected."

        # TFlite ONE_HOT requires both on_value
        # and off_value, making dtype redundant.
        indices = input_tensors[0]
        depth = input_tensors[1]
        on_value = input_tensors[2]
        off_value = input_tensors[3]

        assert on_value.tensor.Type() == off_value.tensor.Type(), (
            "on_value and off_value should be the same type"
        )

        # Getting relax expr for indices
        indices_expr = self.get_expr(indices.tensor_idx)

        # Getting depth value
        depth = self.get_tensor_value(depth)
        if isinstance(depth, np.ndarray):
            depth = int(depth)

        # Getting Axis from Option (Attributes)
        assert op.BuiltinOptionsType() == BuiltinOptions.OneHotOptions
        op_options = op.BuiltinOptions()
        one_hot_options = OneHotOptions()
        one_hot_options.Init(op_options.Bytes, op_options.Pos)
        axis = one_hot_options.Axis()

        # Extract scalar values for on_value and off_value as Expr
        dtype = self.get_tensor_type_str(on_value.tensor.Type())
        on_val = self.get_tensor_value(on_value).item()
        off_val = self.get_tensor_value(off_value).item()
        if "float" in dtype:
            on_prim = relax.prim_value(tvm_tirx.ir.FloatImm(dtype, float(on_val)))
            off_prim = relax.prim_value(tvm_tirx.ir.FloatImm(dtype, float(off_val)))
        else:
            on_prim = relax.prim_value(tvm_tirx.ir.IntImm(dtype, int(on_val)))
            off_prim = relax.prim_value(tvm_tirx.ir.IntImm(dtype, int(off_val)))

        out = relax.op.one_hot(indices_expr, on_prim, off_prim, depth, axis)

        return out

    def convert_reverse_v2(self, op):
        """Convert TFLite REVERSE_V2"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensor's length should be 2"

        input_expr = self.get_expr(input_tensors[0].tensor_idx)

        # Getting axis value
        axis = self.get_tensor_value(input_tensors[1])
        if isinstance(axis, np.ndarray):
            assert len(axis) == 1, "TFLite does not support multi-axis yet"
            axis = int(axis.flat[0])

        out = relax.op.flip(input_expr, axis)
        return out

    def convert_matrix_set_diag(self, op):
        """Convert TFLite MATRIX_SET_DIAG"""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensor's length should be 2"

        assert input_tensors[0].tensor.Type() == input_tensors[1].tensor.Type(), (
            "input and diagonal should be the same type of tensors"
        )

        if input_tensors[0].qnn_params:
            # Check that input and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            assert self.has_same_qnn_params(input_tensors[0], output_tensors[0]), (
                "TFLite MATRIX_SET_DIAG requires input and output tensors' \
                    scale and zero points to be equal"
            )

            # Check that input and diagonal tensor have same qnn params.
            assert self.has_same_qnn_params(input_tensors[0], input_tensors[1]), (
                "TFLite MATRIX_SET_DIAG requires input and diagonal tensors' \
                    scale and zero points to be equal"
            )

        input_expr = self.get_tensor_expr(input_tensors[0])
        diagonal_expr = self.get_tensor_expr(input_tensors[1])

        output_tensors = self.get_output_tensors(op)
        output_tensor = output_tensors[0]
        output_shape = to_int_list(self.get_tensor_shape(output_tensor))
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())

        # topi.matrix_set_diag(
        #     input, diagonal, k1, k2, super_diag_right_align, sub_diag_right_align
        # )
        # TFLite MATRIX_SET_DIAG only sets the main diagonal, so k1=0, k2=0
        out = relax.op.call_dps_packed(
            _ExternFunc("topi.matrix_set_diag"),
            (
                input_expr,
                diagonal_expr,
                relax.const(0),
                relax.const(0),
                relax.const(False),
                relax.const(False),
            ),
            ty_args=[relax.TensorType(output_shape, output_dtype)],
        )
        return out

    def convert_matrix_diag(self, op):
        """Convert TFLite MATRIX_DIAG"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensor's length should be 1"

        diagonal = input_tensors[0]

        if diagonal.qnn_params:
            # Check that diagonal and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            assert self.has_same_qnn_params(diagonal, output_tensors[0]), (
                "TFLite MATRIX_DIAG requires diagonal and output tensors' \
                    scale and zero points to be equal"
            )

        output_tensors = self.get_output_tensors(op)
        output_tensor = output_tensors[0]
        output_shape = to_int_list(self.get_tensor_shape(output_tensor))
        output_dtype = self.get_tensor_type_str(output_tensor.tensor.Type())

        diagonal_expr = self.get_tensor_expr(diagonal)
        zeros_expr = relax.op.zeros(output_shape, output_dtype)

        # topi.matrix_set_diag(
        #     input, diagonal, k1, k2, super_diag_right_align, sub_diag_right_align
        # )
        # TFLite MATRIX_DIAG only sets the main diagonal, so k1=0, k2=0
        out = relax.op.call_dps_packed(
            _ExternFunc("topi.matrix_set_diag"),
            (
                zeros_expr,
                diagonal_expr,
                relax.const(0),
                relax.const(0),
                relax.const(False),
                relax.const(False),
            ),
            ty_args=[relax.TensorType(output_shape, output_dtype)],
        )
        return out

    def convert_densify(self, op):
        """Convert TFLite DENSIFY"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        sparse_weight_tensor = input_tensors[0]
        sparse_weight_tensor_type_str = self.get_tensor_type_str(sparse_weight_tensor.tensor.Type())

        # NOTE: With current implementation in TFLite, Densify Op does not need to be present
        # in runtime.
        # TODO(ANSHUMAN87): we need to use the sparse_indices output
        # from below function and use that in sparse_to_dense Op.
        # Once the stack corruption issue is resolved in sparse_to_dense Op.
        _, dense_weight = prepare_dense_matrix_from_sparse(
            sparse_weight_tensor.tensor,
            self.get_tensor_value(sparse_weight_tensor, is_sparse=True),
            sparse_weight_tensor_type_str,
        )

        self.set_prefetched_node(output_tensor.tensor_idx, dense_weight)

    def convert_fake_quant(self, op):
        """Convert TFLite FAKE_QUANT"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.FakeQuantOptions import FakeQuantOptions

        assert op.BuiltinOptionsType() == BuiltinOptions.FakeQuantOptions

        op_options = op.BuiltinOptions()
        fake_quant_options = FakeQuantOptions()
        fake_quant_options.Init(op_options.Bytes, op_options.Pos)

        opt_min = fake_quant_options.Min()
        opt_max = fake_quant_options.Max()
        narrow_range = fake_quant_options.NarrowRange()
        num_bits = fake_quant_options.NumBits()

        assert 2 <= num_bits <= 16

        quant_min = 1 if narrow_range else 0
        quant_max = (1 << num_bits) - 1
        scale = (opt_max - opt_min) / (quant_max - quant_min)

        zero_point_from_min = quant_min - opt_min / scale
        if zero_point_from_min <= quant_min:
            nudged_zero_point = quant_min
        elif zero_point_from_min >= quant_max:
            nudged_zero_point = quant_max
        else:
            nudged_zero_point = round(zero_point_from_min)

        nudged_min = (quant_min - nudged_zero_point) * scale
        nudged_max = (quant_max - nudged_zero_point) * scale

        nudged_min_expr = relax.const(nudged_min, "float32")
        clamped = relax.op.clip(in_expr, nudged_min, nudged_max)
        clamped_shifted = relax.op.subtract(clamped, nudged_min_expr)

        half = relax.const(0.5, "float32")
        one = relax.const(1.0, "float32")
        scale_expr = relax.const(scale, "float32")
        inv_scale = relax.op.divide(one, scale_expr)
        rounded = relax.op.floor(_op.add(_op.multiply(clamped_shifted, inv_scale), half))
        return relax.op.add(_op.multiply(rounded, scale_expr), nudged_min_expr)

    def convert_real(self, op):
        """Convert TFLite REAL op.

        TFLite complex64 tensors are represented as float32[..., 2] in Relax,
        where index 0 = real part, index 1 = imaginary part along the last axis
        """
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = self.get_expr(input_tensors[0].tensor_idx)
        # slice last axis at index 0, and squeeze to remove the last axis
        real = _op.strided_slice(input_tensor, begin=[0], end=[1], strides=[1], axes=[-1])
        return _op.squeeze(real, axis=[-1])

    def convert_imag(self, op):
        """Convert TFLite IMAG op.

        See convert_real for representation of complex64 tensors in Relax.
        """
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = self.get_expr(input_tensors[0].tensor_idx)
        # slice last axis at index 1, and squeeze to remove the last axis
        imag = _op.strided_slice(input_tensor, begin=[1], end=[2], strides=[1], axes=[-1])
        return _op.squeeze(imag, axis=[-1])

    def convert_complex_abs(self, op):
        """Convert TFLite COMPLEX_ABS op: sqrt(real^2 + imag^2)

        See convert_real for the float32[..., 2] complex representation convention.
        """
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = self.get_expr(input_tensors[0].tensor_idx)
        real = self.bb.emit(
            _op.strided_slice(input_tensor, begin=[0], end=[1], strides=[1], axes=[-1])
        )
        real = self.bb.emit(_op.squeeze(real, axis=[-1]))
        imag = self.bb.emit(
            _op.strided_slice(input_tensor, begin=[1], end=[2], strides=[1], axes=[-1])
        )
        imag = self.bb.emit(_op.squeeze(imag, axis=[-1]))
        real_sq = self.bb.emit(_op.multiply(real, real))
        imag_sq = self.bb.emit(_op.multiply(imag, imag))
        sum_expr = self.bb.emit(_op.add(real_sq, imag_sq))
        return _op.sqrt(sum_expr)

    def get_expr(self, input_tensor_idx):
        return self.exp_tab.get_expr(get_tensor_name(self.subgraph, input_tensor_idx))

    def has_expr(self, input_tensor_idx):
        return self.exp_tab.has_expr(get_tensor_name(self.subgraph, input_tensor_idx))

    def is_prefetched(self, input_tensor_idx):
        return (
            self.prefetched_nodes.get(get_tensor_name(self.subgraph, input_tensor_idx)) is not None
        )

    def set_prefetched_node(self, input_tensor_idx, value):
        self.prefetched_nodes[get_tensor_name(self.subgraph, input_tensor_idx)] = value

    def get_prefetched_node(self, input_tensor_idx):
        return self.prefetched_nodes[get_tensor_name(self.subgraph, input_tensor_idx)]

    def get_tensor_value_or_prefetched(self, tensor, is_sparse=False):
        if self.is_prefetched(tensor.tensor_idx):
            return self.get_prefetched_node(tensor.tensor_idx)
        return self.get_tensor_value(tensor, is_sparse)

    def get_tensor_expr(self, tensor, is_sparse=False):
        """Return the Relax expr for tensor."""
        if self.has_expr(tensor.tensor_idx):
            return self.get_expr(tensor.tensor_idx)

        type_str = self.get_tensor_type_str(tensor.tensor.Type())
        value = self.get_tensor_value_or_prefetched(tensor, is_sparse)
        # complex64 constants have no native Relax dtype. Reinterpret the
        # interleaved float32 storage as float32[..., 2] to match the
        # convention used for input tensors.
        if type_str == "complex64":
            value = value.view(np.float32).reshape(value.shape + (2,))
            type_str = "float32"
        return self.exp_tab.new_const(value, dtype=type_str, source_name=tensor.tensor.Name())

    def get_tensor_shape(self, tensor_wrapper):
        """Returns the TFLite tensor shape, inferring it if the TFLite shape is empty.

        This returns the *raw TFLite* shape (no pair axis), even for COMPLEX64 tensors.
        It is distinct from ``_get_relax_tensor_shape``, which returns the *Relax*
        representation (TFLite shape with a trailing (2,) axis appended for COMPLEX64).
        Operators that build Relax IR for COMPLEX64 inputs should use
        ``_get_relax_tensor_shape``; operators that need the TFLite shape for validation
        (e.g. comparing against the model's declared output shape) should use this method.
        """
        assert isinstance(tensor_wrapper, TensorWrapper), "Expecting TensorWrapper here"
        return (
            tensor_wrapper.tensor.ShapeAsNumpy()
            if tensor_wrapper.tensor.ShapeLength() > 0
            else self._infer_shape(self.get_tensor_expr(tensor_wrapper))
        )


def _is_power_of_2(n):
    """Return True iff ``n`` is a positive power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def _bit_reversal_swap_pairs(n):
    """Return the (i, j) index pairs (i < j) for the bit-reversal permutation of length n.

    For a Cooley-Tukey radix-2 FFT, the input must be permuted by bit-reversing
    each index in log2(n) bits before the butterfly stages. Precomputing the
    swap pairs as constants is much cheaper in TIR than computing the
    bit-reverse on the fly.
    """
    assert _is_power_of_2(n), f"bit-reversal requires power of 2, got {n}"
    length = n.bit_length() - 1  # log2(n)
    swaps = []
    for i in range(1, n):
        j = 0
        for k in range(length):
            if i & (1 << k):
                j |= 1 << (length - 1 - k)
        if i < j:
            swaps.append((i, j))
    return swaps


def _build_tflite_rfft2d_function(input_shape, output_pair_shape):
    """Build a reference TIR kernel for TFLite RFFT2D.

    The TFLite frontend represents complex tensors as float32 real/imag pairs
    with a trailing dimension of size 2 because TVM does not have a native
    complex64 dtype. This kernel computes the unnormalized 2-D real FFT over
    the last two input dimensions and writes that pair representation.

    All trig and accumulation are in float32, so the result agrees with
    ``np.fft.rfft2`` to about ``1e-5`` absolute tolerance for typical input
    sizes. Higher-precision backends should override this kernel.

    Notes
    -----
    This is a **naive O(B * H * W * H * W) DFT**, not an FFT. For an input of
    spatial shape (H, W) the inner sum runs H*W times per output position, and
    there are H*W' output positions per batch (W' = W // 2 + 1). This is
    intentionally simple for correctness validation against
    ``np.fft.rfft2``; production use cases with large spatial dimensions should
    override the kernel with an FFT-based implementation. The outer
    (batch, out_y, out_x) iteration is structured as S-TIR spatial axes so a
    downstream ``tvm.tir.schedule`` pass can parallelize it.
    """
    from tvm_tirx.script.parser import tirx as T

    batch = 1
    for dim in input_shape[:-2]:
        batch *= int(dim)
    height = int(input_shape[-2])
    width = int(input_shape[-1])
    out_width = int(output_pair_shape[-2])
    input_total = batch * height * width
    output_complex_total = batch * height * out_width
    neg_two_pi = np.float32(-2.0 * math.pi)

    @Ts.function(private=True, check_well_formed=False)
    def kernel(
        data: T.Tensor(input_shape, "float32"), output: T.Tensor(output_pair_shape, "float32")
    ):
        # Flat 1D aliases of the multi-dim buffers. The kernel is rank-agnostic
        # over the leading batch dimensions, so collapsing the index space
        # avoids special-casing 2D / 3D / 4D input shapes.
        data_flat = T.decl_tensor(
            data.data, ty_args=[T.Tensor((input_total,), "float32", data=data.data)]
        )
        output_flat = T.decl_tensor(
            output.data,
            ty_args=[T.Tensor((output_complex_total * 2,), "float32", data=output.data)],
        )
        neg_two_pi_const = T.float32(neg_two_pi)

        for b_idx, out_y, out_x in T.grid(batch, height, out_width):
            with Ts.sblock("rfft2d"):
                v_b, v_oy, v_ox = Ts.axis.remap("SSS", [b_idx, out_y, out_x])
                real_sum: T.float32 = T.float32(0)
                imag_sum: T.float32 = T.float32(0)
                input_base = v_b * height * width
                for in_y, in_x in T.grid(height, width):
                    phase_y = T.Cast("float32", v_oy) * T.Cast("float32", in_y) / T.float32(height)
                    phase_x = T.Cast("float32", v_ox) * T.Cast("float32", in_x) / T.float32(width)
                    angle = neg_two_pi_const * (phase_y + phase_x)
                    value = data_flat[input_base + in_y * width + in_x]
                    real_sum = real_sum + value * T.cos(angle)
                    imag_sum = imag_sum + value * T.sin(angle)
                flat_out_idx = ((v_b * height + v_oy) * out_width + v_ox) * 2
                output_flat[flat_out_idx] = real_sum
                output_flat[flat_out_idx + 1] = imag_sum

    return kernel


def _build_tflite_rfft2d_fft_function(input_shape, output_pair_shape):
    """Build a 2D Cooley-Tukey FFT TIR kernel for TFLite RFFT2D.

    Precondition: both ``input_shape[-2]`` (height) and ``input_shape[-1]``
    (width) must be positive powers of 2. The frontend dispatches to this
    kernel via ``_is_power_of_2`` checks; the DFT reference kernel handles
    the remaining cases (odd / non-power-of-2 sizes).

    Algorithm
    ---------
    1. Copy the real input into a scratch complex buffer (imag = 0) of shape
       ``(B * H * W,)``.
    2. For each batch and each row, run an in-place radix-2 1D FFT of length
       ``W`` along the width axis.
    3. For each batch and each column, run an in-place radix-2 1D FFT of
       length ``H`` along the height axis (with stride ``W``).
    4. Write the first ``W // 2 + 1`` complex bins per row to the output
       pair representation.

    The bit-reversal permutation required by iterative Cooley-Tukey is done
    by emitting the (i, j) swap pairs directly in the TIR source (one
    inlined swap per pair), avoiding the need for runtime index tables.

    Complexity is ``O(B * H * W * (log2(H) + log2(W)))``, vs the DFT
    reference kernel's ``O(B * H * W * H * W)``.
    """
    from tvm_tirx.script.parser import tirx as T

    batch = 1
    for dim in input_shape[:-2]:
        batch *= int(dim)
    height = int(input_shape[-2])
    width = int(input_shape[-1])
    out_width = int(output_pair_shape[-2])
    input_total = batch * height * width
    output_complex_total = batch * height * out_width
    # Cast to Python float so the f-string-interpolated repr is a plain number
    # (np.float32's repr is "np.float32(...)", which the TIR parser can't resolve).
    neg_two_pi = float(np.float32(-2.0 * math.pi))
    log2_w = int(math.log2(width))
    log2_h = int(math.log2(height))

    if not (_is_power_of_2(height) and _is_power_of_2(width)):
        raise ValueError(
            f"_build_tflite_rfft2d_fft_function requires power-of-2 height and width, "
            f"got H={height}, W={width}"
        )

    # Precompute the bit-reversal swap pairs at Python level. These are
    # constant for a given FFT length and will be inlined in the TIR source.
    # Each emitted line is indented 16 spaces (4 levels: top → b_idx loop →
    # sblock → row/col loop body) so it lands inside the for loop when
    # concatenated into the function source.
    row_swap_stmts = []
    for i, j in _bit_reversal_swap_pairs(width):
        row_swap_stmts.append(
            f"                i_idx = row_base + {i}\n"
            f"                j_idx = row_base + {j}\n"
            f"                tmp_r = scratch_real[i_idx]\n"
            f"                scratch_real[i_idx] = scratch_real[j_idx]\n"
            f"                scratch_real[j_idx] = tmp_r\n"
            f"                tmp_i = scratch_imag[i_idx]\n"
            f"                scratch_imag[i_idx] = scratch_imag[j_idx]\n"
            f"                scratch_imag[j_idx] = tmp_i\n"
        )
    row_swaps_code = "".join(row_swap_stmts) if row_swap_stmts else "                pass\n"

    col_swap_stmts = []
    for i, j in _bit_reversal_swap_pairs(height):
        col_swap_stmts.append(
            f"                i_idx = col_base + {i * width}\n"
            f"                j_idx = col_base + {j * width}\n"
            f"                tmp_r = scratch_real[i_idx]\n"
            f"                scratch_real[i_idx] = scratch_real[j_idx]\n"
            f"                scratch_real[j_idx] = tmp_r\n"
            f"                tmp_i = scratch_imag[i_idx]\n"
            f"                scratch_imag[i_idx] = scratch_imag[j_idx]\n"
            f"                scratch_imag[j_idx] = tmp_i\n"
        )
    col_swaps_code = "".join(col_swap_stmts) if col_swap_stmts else "                pass\n"

    # Build the per-stage butterfly code with the stage loop fully unrolled
    # at function-construction time. After unrolling, all loop bounds
    # (block_start, k) are compile-time integers, so the TIR parser doesn't
    # need to reason about runtime loop extents and the scheduler can
    # see static twiddle factors instead of runtime trig calls.
    def _stage_stmts(stage_count, length, indent, stride=1, base_expr="row_base"):
        """Generate fully-unrolled Cooley-Tukey butterfly stage bodies.

        ``base_expr`` is the TIR expression holding the base offset of the
        FFT being transformed (e.g. ``"row_base"`` for rows or
        ``"col_base"`` for columns). ``stride`` is the integer distance
        between adjacent butterfly taps: 1 for the row-FFT (contiguous
        elements) and ``width`` for the column-FFT (strided access).
        """
        out = []
        for stage in range(1, stage_count + 1):
            m_val = 1 << stage
            half_val = m_val >> 1
            for block_start in range(0, length, m_val):
                for k in range(half_val):
                    if stride == 1:
                        a_idx = f"{base_expr} + {block_start} + {k}"
                        b_idx_expr = f"{base_expr} + {block_start} + {k} + {half_val}"
                    else:
                        a_idx = f"{base_expr} + ({block_start} + {k}) * {stride}"
                        b_idx_expr = f"{base_expr} + ({block_start} + {k} + {half_val}) * {stride}"
                    angle_val = float(np.float32(neg_two_pi * k / m_val))
                    w_real_val = float(np.float32(math.cos(angle_val)))
                    w_imag_val = float(np.float32(math.sin(angle_val)))
                    out.extend(
                        [
                            f"{indent}a_idx = {a_idx}\n",
                            f"{indent}b_idx_local = {b_idx_expr}\n",
                            f"{indent}t_real = scratch_real[b_idx_local] * T.float32({w_real_val!r}) - scratch_imag[b_idx_local] * T.float32({w_imag_val!r})\n",
                            f"{indent}t_imag = scratch_real[b_idx_local] * T.float32({w_imag_val!r}) + scratch_imag[b_idx_local] * T.float32({w_real_val!r})\n",
                            f"{indent}u_real = scratch_real[a_idx]\n",
                            f"{indent}u_imag = scratch_imag[a_idx]\n",
                            f"{indent}scratch_real[a_idx] = u_real + t_real\n",
                            f"{indent}scratch_imag[a_idx] = u_imag + t_imag\n",
                            f"{indent}scratch_real[b_idx_local] = u_real - t_real\n",
                            f"{indent}scratch_imag[b_idx_local] = u_imag - t_imag\n",
                        ]
                    )
        return "".join(out)

    row_stages_code = _stage_stmts(
        log2_w, width, "                ", stride=1, base_expr="row_base"
    )
    col_stages_code = _stage_stmts(
        log2_h, height, "                ", stride=width, base_expr="col_base"
    )

    # Build the function source. The bit-reversal swaps are inlined (one
    # unconditional block per (i, j) pair) and the butterfly stages are
    # fully unrolled, so the TIR parser sees ordinary statements rather than
    # runtime table lookups or runtime-magnitude loops. The body is wrapped
    # in a single S-TIR block over the batch dimension so the Relax
    # pipeline (which expects an SBlockRealize at the function body) accepts
    # this kernel.
    function_source = (
        "from tvm_tirx.script.parser import tirx as T\n"
        "from tvm.s_tir import script as Ts\n"
        "@Ts.function(private=True, check_well_formed=False)\n"
        "def kernel(\n"
        f"    data: T.Tensor({tuple(int(x) for x in input_shape)}, 'float32'),\n"
        f"    output: T.Tensor({tuple(int(x) for x in output_pair_shape)}, 'float32'),\n"
        "):\n"
        f"    data_flat = T.decl_tensor(data.data, ty_args=[T.Tensor(({input_total},), 'float32')])\n"
        f"    output_flat = T.decl_tensor(output.data, ty_args=[T.Tensor(({output_complex_total * 2},), 'float32')])\n"
        f"    scratch_real = T.alloc_tensor(ty_args=[T.Tensor(({input_total},), 'float32')])\n"
        f"    scratch_imag = T.alloc_tensor(ty_args=[T.Tensor(({input_total},), 'float32')])\n"
        f"    for b_idx in T.serial({batch}):\n"
        f"        with Ts.sblock('rfft2d_fft'):\n"
        f"            v_b = Ts.axis.remap('S', [b_idx])\n"
        f"            # Initialize scratch from real input; imag = 0.\n"
        f"            for i in T.serial({height * width}):\n"
        f"                src = v_b * {height * width} + i\n"
        f"                scratch_real[src] = data_flat[src]\n"
        f"                scratch_imag[src] = T.float32(0)\n"
        f"            # FFT along width axis (one 1D FFT per row).\n"
        f"            for row in T.serial({height}):\n"
        f"                row_base = v_b * {height * width} + row * {width}\n"
        f"                # Bit-reversal permutation (inlined swaps).\n"
        f"{row_swaps_code}"
        f"                # Cooley-Tukey butterfly stages (fully unrolled).\n"
        f"{row_stages_code}"
        f"            # FFT along height axis (one 1D FFT per column, stride = width).\n"
        f"            for col in T.serial({width}):\n"
        f"                col_base = v_b * {height * width} + col\n"
        f"                # Bit-reversal permutation (strided, inlined swaps).\n"
        f"{col_swaps_code}"
        f"                # Cooley-Tukey butterfly stages (strided, fully unrolled).\n"
        f"{col_stages_code}"
        f"            # Write the first out_width complex bins per row to output.\n"
        f"            for row in T.serial({height}):\n"
        f"                for out_x in T.serial({out_width}):\n"
        f"                    src = v_b * {height * width} + row * {width} + out_x\n"
        f"                    dst = ((v_b * {height} + row) * {out_width} + out_x) * 2\n"
        f"                    output_flat[dst] = scratch_real[src]\n"
        f"                    output_flat[dst + 1] = scratch_imag[src]\n"
    )

    namespace = {"T": T, "tirx": T}
    # Register the generated source in linecache so the TIR parser (which calls
    # inspect.getsourcelines) can find it. The fake filename is stable across
    # calls — we include input/output shapes so the cache is self-disambiguating
    # for any callers who want to introspect the generated source.
    import linecache as _linecache

    fake_file = f"<tflite_rfft2d_fft_function H={height} W={width} outW={out_width}>"
    _linecache.cache[fake_file] = (
        len(function_source.splitlines()),
        None,
        [line + "\n" for line in function_source.splitlines()],
        fake_file,
    )
    code = compile(function_source, fake_file, "exec")
    exec(code, namespace)
    return namespace["kernel"]


# Constants for the Random123 counter-based PRNGs used by STABLEHLO_RNG_BIT_GENERATOR,
# matching tensorflow/lite/kernels/rng_util.cc.
_STABLEHLO_RNG_THREEFRY_PARITY = 0x1BD11BDA
_STABLEHLO_RNG_PHILOX_MUL_A = 0xD2511F53
_STABLEHLO_RNG_PHILOX_MUL_B = 0xCD9E8D57
_STABLEHLO_RNG_PHILOX_WEYL_A = 0x9E3779B9
_STABLEHLO_RNG_PHILOX_WEYL_B = 0xBB67AE85


def _build_stablehlo_rng_bit_generator_function(algorithm, state_len, out_dtype, out_shape):
    """Build a bit-exact TIR kernel for STABLEHLO_RNG_BIT_GENERATOR.

    Mirrors the TFLite runtime kernel (tensorflow/lite/kernels/rng_bit_generator.cc),
    implementing the Random123 Threefry2x32 (20 rounds) and Philox4x32 (10 rounds)
    counter-based PRNGs. The kernel reinterprets the uint64 state as uint32 words,
    advances a 64-bit block counter, and packs the generated words into the output
    tensor. The updated state keeps the key unchanged and only advances the counter,
    which is the only behaviour the runtime relies on.
    """
    from tvm_tirx.script.parser import tirx as T

    total = 1
    for dim in out_shape:
        total *= int(dim)
    is_64bit = out_dtype in ("int64", "uint64")
    block_words = 2 if algorithm == "threefry" else 4
    out_word_count = total * (2 if is_64bit else 1)
    num_blocks = (out_word_count + block_words - 1) // block_words
    writes_per_block = block_words // (2 if is_64bit else 1)
    parity = _STABLEHLO_RNG_THREEFRY_PARITY
    mul_a, mul_b = _STABLEHLO_RNG_PHILOX_MUL_A, _STABLEHLO_RNG_PHILOX_MUL_B
    weyl_a, weyl_b = _STABLEHLO_RNG_PHILOX_WEYL_A, _STABLEHLO_RNG_PHILOX_WEYL_B

    def _u32(value):
        return T.Cast("uint32", value)

    def _u64(value):
        return T.Cast("uint64", value)

    def _store_value(words, write_index):
        # Pack the generated uint32 words into one output element, reinterpreting
        # the bit pattern into the (possibly signed) output dtype.
        if is_64bit:
            low = _u64(words[2 * write_index])
            high = _u64(words[2 * write_index + 1])
            return T.reinterpret(low | (high << T.uint64(32)), ty=out_dtype)
        return T.reinterpret(words[write_index], ty=out_dtype)

    if algorithm == "threefry":

        @Ts.function(private=True)
        def kernel(
            initial_state: T.Tensor((state_len,), "uint64"),
            output_state: T.Tensor((state_len,), "uint64"),
            output: T.Tensor(out_shape, out_dtype),
        ):
            # A single opaque structured block keeps the imperative kernel as a
            # well-formed block-structured Function, as required by the Relax
            # pipeline (e.g. HasReshapePattern).
            with Ts.sblock("rng_bit_generator"):
                state_key = initial_state[0]
                state_counter = initial_state[1]
                key_0 = _u32(state_key & T.uint64(0xFFFFFFFF))
                key_1 = _u32(state_key >> T.uint64(32))
                output_state[0] = state_key
                output_state[1] = state_counter + T.uint64(num_blocks)
                out_flat = T.decl_tensor(
                    output.data, ty_args=[T.Tensor((total,), out_dtype, data=output.data)]
                )
                keys = T.alloc_tensor(ty_args=[T.Tensor((3,), "uint32", scope="local")])
                rotations = T.alloc_tensor(ty_args=[T.Tensor((8,), "uint32", scope="local")])
                ctr = T.alloc_tensor(ty_args=[T.Tensor((2,), "uint32", scope="local")])
                keys[0] = key_0
                keys[1] = key_1
                keys[2] = key_0 ^ key_1 ^ T.uint32(parity)
                rotations[0] = T.uint32(13)
                rotations[1] = T.uint32(15)
                rotations[2] = T.uint32(26)
                rotations[3] = T.uint32(6)
                rotations[4] = T.uint32(17)
                rotations[5] = T.uint32(29)
                rotations[6] = T.uint32(16)
                rotations[7] = T.uint32(24)
                for block in T.serial(num_blocks):
                    counter = state_counter + _u64(block)
                    ctr[0] = _u32(counter & T.uint64(0xFFFFFFFF)) + key_0
                    ctr[1] = _u32(counter >> T.uint64(32)) + key_1
                    for group in T.serial(5):
                        for step in T.serial(4):
                            rot = rotations[(group * 4 + step) % 8]
                            ctr[0] = ctr[0] + ctr[1]
                            ctr[1] = (ctr[1] << rot) | (ctr[1] >> (T.uint32(32) - rot))
                            ctr[1] = ctr[1] ^ ctr[0]
                        ctr[0] = ctr[0] + keys[(group + 1) % 3]
                        ctr[1] = ctr[1] + keys[(group + 2) % 3] + _u32(group + 1)
                    for write_index in T.serial(writes_per_block):
                        element = block * writes_per_block + write_index
                        if element < total:
                            out_flat[element] = _store_value(ctr, write_index)

        return kernel

    @Ts.function(private=True)
    def kernel(
        initial_state: T.Tensor((state_len,), "uint64"),
        output_state: T.Tensor((state_len,), "uint64"),
        output: T.Tensor(out_shape, out_dtype),
    ):
        with Ts.sblock("rng_bit_generator"):
            state_key = initial_state[0]
            state_counter = initial_state[1]
            key_0 = _u32(state_key & T.uint64(0xFFFFFFFF))
            key_1 = _u32(state_key >> T.uint64(32))
            output_state[0] = state_key
            output_state[1] = state_counter + T.uint64(num_blocks)
            out_flat = T.decl_tensor(
                output.data, ty_args=[T.Tensor((total,), out_dtype, data=output.data)]
            )
            ctr = T.alloc_tensor(ty_args=[T.Tensor((4,), "uint32", scope="local")])
            keys = T.alloc_tensor(ty_args=[T.Tensor((2,), "uint32", scope="local")])
            high_ctr = T.alloc_tensor(ty_args=[T.Tensor((2,), "uint32", scope="local")])
            if state_len == 3:
                # PHILOX u64[3]: the third state word feeds the high counter and
                # is passed through to the output state unchanged.
                high_state = initial_state[2]
                output_state[2] = high_state
                high_ctr[0] = _u32(high_state & T.uint64(0xFFFFFFFF))
                high_ctr[1] = _u32(high_state >> T.uint64(32))
            else:
                high_ctr[0] = key_0
                high_ctr[1] = key_1
            for block in T.serial(num_blocks):
                counter = state_counter + _u64(block)
                ctr[0] = _u32(counter & T.uint64(0xFFFFFFFF))
                ctr[1] = _u32(counter >> T.uint64(32))
                ctr[2] = high_ctr[0]
                ctr[3] = high_ctr[1]
                keys[0] = key_0
                keys[1] = key_1
                for _round in T.serial(10):
                    prod_0 = T.uint64(mul_a) * _u64(ctr[0])
                    prod_1 = T.uint64(mul_b) * _u64(ctr[2])
                    new_0 = _u32(prod_1 >> T.uint64(32)) ^ ctr[1] ^ keys[0]
                    new_1 = _u32(prod_1 & T.uint64(0xFFFFFFFF))
                    new_2 = _u32(prod_0 >> T.uint64(32)) ^ ctr[3] ^ keys[1]
                    new_3 = _u32(prod_0 & T.uint64(0xFFFFFFFF))
                    ctr[0] = new_0
                    ctr[1] = new_1
                    ctr[2] = new_2
                    ctr[3] = new_3
                    keys[0] = keys[0] + T.uint32(weyl_a)
                    keys[1] = keys[1] + T.uint32(weyl_b)
                for write_index in T.serial(writes_per_block):
                    element = block * writes_per_block + write_index
                    if element < total:
                        out_flat[element] = _store_value(ctr, write_index)

    return kernel


# pylint: disable=no-else-return
def prepare_dense_matrix_from_sparse(sparse_tensor, sparse_tensor_value, sparse_tensor_type):
    """Prepare sparse indices and dense matrix from TFLite sparse parameters."""
    # The function is implemented based on TFLite sparse parameter specifications
    # Please refer
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs#L89
    # for details about each parameters
    sparsity = sparse_tensor.Sparsity()
    dense_shape = sparse_tensor.ShapeAsNumpy()
    orig_rank = len(dense_shape)

    # The traversal order of the dimensions defined in the `shape` field of the to be dense tensor.
    traversal_order = sparsity.TraversalOrderAsNumpy()

    # For an n-dimensional tensor with a k-dimensional block (0 <= k <= n),
    # stores how a block dimension in (dn, ..., dn+k-1) maps to the original
    # tensor dimension in (d0, ..., dn). It's stored in the order of (dn, ..., dn+k-1).
    # If not block-sparse, this field is NULL.
    block_map = sparsity.BlockMapAsNumpy()

    total_rank = sparsity.TraversalOrderLength()
    dense_mat = np.full(shape=dense_shape, fill_value=0, dtype=sparse_tensor_type).flatten()

    from enum import Enum

    # NOTE: Here the Vector term is borrowed from TFLite spec.
    class VectorType(Enum):
        Empty = 0
        Int32 = 1
        Uint16 = 2
        Uint8 = 3

    def _get_vector_flag(v_type):
        if VectorType(v_type) == VectorType.Int32:
            return N.Int32Flags
        elif VectorType(v_type) == VectorType.Uint16:
            return N.Uint16Flags
        elif VectorType(v_type) == VectorType.Uint8:
            return N.Uint8Flags
        else:
            raise NotImplementedError(f"The provided type {v_type} is not supported")

    def _get_flattened_index(indices, shape):
        index = 0
        sub_elements = 1
        for i in reversed(range(0, len(dense_shape))):
            index += indices[i] * sub_elements
            sub_elements *= shape[i]
        return index

    # DimensionMetadata per dimension: the metadata needed for
    #     each dimension to locate the non-zero values in the original dense tensor
    #     inline with traversal order parameter.
    #
    # sp_format has 2 possible values: {DENSE = 0, SPARSE_CSR = 1}
    # If format = DENSE{0} : DenseSize represents size of that dimension
    # If format = SPARSE_CSR{1} : array_segments represents how to segment the indices array,
    #      each segment corresponds to one element in the previous dimension. array_indices
    #      represents the index of the non-zero elements within this dimension
    #      (as those in the CSR matrix format, where the first array is row pointers
    #       and the second array is column indices).
    sp_format = np.zeros(sparsity.DimMetadataLength())
    dim_metadata = [None] * (2 * sparsity.DimMetadataLength())

    # Below loop will fetch all meta data per dimension based on format type
    # Dense or Sparse and will put it in an agnostic array for easy access
    # while preparing dense buffer or indices.
    for i in range(sparsity.DimMetadataLength()):
        sp_format[i] = sparsity.DimMetadata(i).Format()
        if sp_format[i] == 0:
            dim_metadata[2 * i] = [sparsity.DimMetadata(i).DenseSize()]
        else:
            from flatbuffers import number_types as N

            dim_metadata[2 * i] = (
                sparsity.DimMetadata(i)
                .ArraySegments()
                .GetVectorAsNumpy(
                    flags=_get_vector_flag(sparsity.DimMetadata(i).ArraySegmentsType()), off=4
                )
            )
            dim_metadata[2 * i + 1] = (
                sparsity.DimMetadata(i)
                .ArrayIndices()
                .GetVectorAsNumpy(
                    flags=_get_vector_flag(sparsity.DimMetadata(i).ArrayIndicesType()), off=4
                )
            )

    block_dim = 0
    block_size = np.zeros(sparsity.BlockMapLength())

    # Block size parameter if encoded in BSR format
    for i in range(orig_rank):
        if block_dim < sparsity.BlockMapLength() and block_map[block_dim] == i:
            orig_dim = traversal_order[orig_rank + block_dim]
            block_size[block_dim] = sparsity.DimMetadata(orig_dim).DenseSize()
            block_dim += 1

    indices_list = []

    # Below function iterates through each applicable indices per dimension
    # based on format type specified and finally produce the dense matrix and the NZ indices.
    def _def_prepare_dense_matrix_from_sparse(indices, level, prev_idx):
        if level == len(indices):
            start_pos = 0
            orig_idx = np.zeros(orig_rank, dtype="int32")
            while start_pos < orig_rank:
                orig_idx[traversal_order[start_pos]] = indices[start_pos]
                start_pos += 1
            while start_pos < len(indices):
                block_idx = traversal_order[start_pos] - orig_rank
                orig_dim = block_map[block_idx]
                orig_idx[orig_dim] = orig_idx[orig_dim] * block_size[block_idx] + indices[start_pos]
                start_pos += 1
            indices_list.append(orig_idx)
            nonlocal value_idx
            dense_mat[_get_flattened_index(orig_idx, dense_shape)] = sparse_tensor_value[value_idx]
            value_idx += 1
        else:
            metadata_idx = 2 * level
            if sp_format[level] == 0:
                shape_of_level = dim_metadata[metadata_idx][0]
                for idx in range(shape_of_level):
                    indices[level] = idx
                    _def_prepare_dense_matrix_from_sparse(
                        indices, level + 1, prev_idx * shape_of_level + idx
                    )
            else:
                array_segments = dim_metadata[metadata_idx]
                array_indices = dim_metadata[metadata_idx + 1]
                for idx in range(array_segments[prev_idx], array_segments[prev_idx + 1]):
                    indices[level] = array_indices[idx]
                    _def_prepare_dense_matrix_from_sparse(indices, level + 1, idx)

    indices = np.zeros(total_rank)
    value_idx = 0
    _def_prepare_dense_matrix_from_sparse(indices, 0, 0)
    return np.array(indices_list, dtype="int32"), dense_mat.reshape(dense_shape)


def get_scalar_from_constant(expr):
    """Returns scalar value from Relax constant scalar."""
    assert isinstance(expr, tvm_tirx.ir.GenericConst) and not expr.value.shape, (
        "Expr is not a constant scalar."
    )
    value = expr.value.numpy()
    assert value.dtype == np.dtype(np.int32) or value.dtype == np.dtype(np.float32), (
        "value must be float32/int32"
    )
    return value.item(0)


def get_tensor_from_constant(expr):
    """Returns tensor of values from Relax constant node."""
    assert isinstance(expr, relax.const)
    value = expr.value.numpy()
    assert value.dtype == np.dtype(np.int32) or value.dtype == np.dtype(np.float32), (
        "value must be float32/int32"
    )
    return value


def build_str_map(obj):
    """Build string map of TFLite enum int value

    Parameters
    ----------
    obj:
        TFLite class which contains enum int value, such as BuiltInOptions

    Returns
    -------
        String representation map of TFLite class enum int value
    """
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith("_"):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret


# SAME padding: https://www.tensorflow.org/api_guides/python/nn
def get_pad_value(data, kernel, stride):
    """Get the pad tuple of value for SAME padding

    Parameters
    ----------
    data:
        1D input data

    kernel:
        1D input kernel

    stride:
        1D input stride

    Returns
    -------
        pad tuple of value
    """

    out = math.ceil(float(data) / float(stride))
    pad = max(0, (out - 1) * stride + kernel - data)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after


def get_tensor_name(subgraph, tensor_idx):
    """Get the tensor name.

    Parameters
    ----------
    subgraph:
        tflite.Subgraph.Subgraph

    tensor:
        tensor index in subgraph

    Returns
    -------
        tensor name in UTF-8 encoding
    """
    tensor_name = subgraph.Tensors(tensor_idx).Name()
    if tensor_name is not None:
        tensor_name = tensor_name.decode("utf-8")
    else:
        tensor_name = "tvmgen_tensor_" + str(tensor_idx)
    return tensor_name


def _decode_type(n):
    _tflite_m = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
        12: "uint64",
        15: "uint32",
    }
    return _tflite_m[n]


def _input_type(model):
    subgraph_count = model.SubgraphsLength()
    assert subgraph_count > 0
    shape_dict = {}
    dtype_dict = {}
    subgraph = model.Subgraphs(0)
    inputs_count = subgraph.InputsLength()
    # TFLite subgraphs can validly have zero inputs (e.g. constant-only RANGE models).
    for input_index in range(inputs_count):
        input_ = subgraph.Inputs(input_index)
        assert subgraph.TensorsLength() > input_
        tensor = subgraph.Tensors(input_)
        input_shape = tuple(tensor.ShapeAsNumpy())
        tensor_type = tensor.Type()
        input_name = get_tensor_name(subgraph, input_)
        input_dtype = _decode_type(tensor_type)
        shape_dict[input_name] = input_shape
        dtype_dict[input_name] = input_dtype

    return shape_dict, dtype_dict


def from_tflite(
    model,
    shape_dict: dict[str, tuple[int]] | None = None,
    dtype_dict: dict[str, str] | None = None,
    op_converter=OperatorConverter,
):
    """Convert from tflite model into compatible relax Function.

    Parameters
    ----------
    model:
        tflite.Model or tflite.Model.Model (depending on tflite version)

    shape_dict : dict of str to int list/tuple
        Input shapes of the model.

    dtype_dict : dict of str to str
        Input types of the model.

    Returns
    -------
    mod : tvm_tirx.IRModule
        The relax module for compilation.

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relax

    Examples
    --------
    Users can use TFLiteConverter to convert a concrete function followed by
    tflite.Model.Model.GetRootAsModel to get a tflite.Model object
    The following codes show how to convert a simple concrete function to a Relax program.

    .. code-block:: python

        import tensorflow as tf
        from tvm.relax.frontend.tflite import from_tflite


        # A concrete function be defined from network architecture or invoking a model
        # from keras applications

        # Define a tf.Module containing a tf.function with network definition
        class Conv2DModule(tf.Module):
            @tf.function(
                input_signature=[
                    tf.TensorSpec(shape=(1, 128, 128, 32), dtype=tf.float32),
                    tf.TensorSpec(shape=(3, 3, 32, 32), dtype=tf.float32),
                ]
            )
            def func(self, data, kernel):
                return tf.nn.conv2d(
                    input=(1, 128, 128, 32),
                    filters=(3, 3, 32, 32),
                    data_format="NHWC",
                    strides=(1, 1, 1, 1),
                    padding="SAME",
                )
        concrete_func = Conv2DModule().func.get_concrete_function()

        # Alternatively you can use tensorflow.keras.application wrapped arounf a tf.function
        class NetworkModule(tf.Module):
            def __init__(self):
                self.model = net(weights="imagenet", include_top=True)

            @tf.function
            def func(self, data):
                return self.model(data, training=False)

        model = NetworkModule()
        concrete_func = model.func.get_concrete_function(
            tf.TensorSpec(shape=shape, dtype=tf.float32)
        )


        # Which ever the way we have created the concrete function make tflite.Model as shown below

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = tf.lite.Model.Model.GetRootAsModel(converter.convert(), 0)

        # Call the imported to conver the tflite.Model to relax.Module
        mod = from_tflite(tflite_model)

        # Print out the imported model.
        print(mod.script())

    """
    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite

        assert isinstance(model, tflite.Model)
    except ImportError as err:
        raise ImportError(
            "tflite is required by the TFLite frontend. Install it with: pip install tflite"
        ) from err
    except TypeError:
        import tflite.Model

        assert isinstance(model, tflite.Model.Model)

    _shape_dict, _dtype_dict = _input_type(model)
    if shape_dict is not None:
        _shape_dict.update(shape_dict)
    if dtype_dict is not None:
        _dtype_dict.update(dtype_dict)

    # Only Subgraphs(0) is converted into Relax main. Additional subgraphs are
    # region/control-flow bodies referenced by specific TFLite ops and are
    # consumed by those op converters as needed.
    assert model.SubgraphsLength() >= 1, "TFLite model must contain at least one subgraph"
    subgraph = model.Subgraphs(0)

    # model inputs / outputs
    model_inputs = subgraph.InputsAsNumpy()
    model_outputs = subgraph.OutputsAsNumpy()

    bb = relax.BlockBuilder()  # pylint: disable=invalid-name

    with bb.function("main"):
        input_list = []
        with bb.dataflow() as df:  # noqa: F841  # pylint: disable=invalid-name, unused-variable
            exp_tab = ExprTable()
            for model_input in model_inputs:
                model_input_name = get_tensor_name(subgraph, model_input)
                shape = _shape_dict[model_input_name] if model_input_name in _shape_dict else None
                dtype = (
                    _dtype_dict[model_input_name] if model_input_name in _dtype_dict else "float32"
                )
                if dtype == "complex64":
                    dtype = "float32"
                    if shape is not None:
                        shape = tuple(shape) + (2,)
                if dtype == "string":
                    # Relax has no string tensor type, so TFLite TensorType.STRING graph
                    # inputs cannot be represented. This also covers runtime string queries
                    # for ops like HASHTABLE_FIND, whose constant-foldable subset is handled
                    # in the op converter.
                    raise NotImplementedError(
                        "Relax TFLite frontend does not support TensorType.STRING graph inputs "
                        "(e.g. runtime string queries)"
                    )
                input_var = relax.Var(
                    name=model_input_name, ty=relax.TensorType(shape=shape, dtype=dtype)
                )
                exp_tab.set_expr(model_input_name, input_var)
                input_list.append(input_var)

            # op code in model
            op_converter = op_converter(model, subgraph, exp_tab, bb)
            op_converter.check_unsupported_ops()
            op_converter.convert_op_to_relax()

            # params and outputs
            # Resolve outputs through tensor wrappers so constant/prefetched outputs are handled.
            output_tensors = op_converter.get_tensors(model_outputs)
            outputs = [op_converter.get_tensor_expr(tensor) for tensor in output_tensors]
            outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)
            output_var = bb.emit_output(outputs)

        bb.emit_func_output(output_var, input_list)

        relax_mod = bb.get()
        # Attach attributes.
        param_value_list = []
        if exp_tab.params:
            _, param_value_list = map(list, zip(*exp_tab.params.values()))
        func_attrs = {}
        func_attrs["num_input"] = len(input_list)
        func_attrs["params"] = [tvm_tirx.runtime.tensor(arr) for arr in param_value_list]
        relax_mod["main"] = relax_mod["main"].with_attrs(func_attrs)

        return relax_mod
