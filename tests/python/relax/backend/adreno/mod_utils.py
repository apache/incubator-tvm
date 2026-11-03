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
# ruff: noqa: E501, F401, F841
"""CLML integration operator tests."""

import json

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax, rpc
from tvm.relax import script as R
from tvm.relax.backend.adreno import clml
from tvm.relax.script import ir_builder as relax_builder
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder


def get_relax_conv2d_mod(
    data_shape,
    weight_shape,
    stride,
    dilation,
    padding,
    weight_layout="OIHW",
    groups=1,
    dtype="float32",
    has_bias=False,
    has_bn=False,
    has_activation=False,
    has_pad=False,
    is_depthwise=False,
):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            if has_pad:
                p = (0, 0, 0, 0, padding[0], padding[0], padding[1], padding[1])
                orig_data = R.arg_("data", R.Tensor(data_shape, dtype))
                data = R.nn.pad(orig_data, pad_width=p, pad_value=0.0)
                padding = (0, 0, 0, 0)
            else:
                data = R.arg_("data", R.Tensor(data_shape, dtype))
            weight = R.arg_("weight", R.Tensor(weight_shape, dtype))
            if has_bias:
                bias = R.arg_("bias", R.Tensor((1, weight_shape[0], 1, 1), dtype))

            is_depthwise = data_shape[1] == weight_shape[0] == groups

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d(
                        data,
                        weight,
                        out_dtype=dtype,
                        strides=stride,
                        dilation=dilation,
                        padding=padding,
                        data_layout="NCHW",
                        kernel_layout=weight_layout,
                        groups=groups,
                    )
                )
                if has_bias:
                    output = R.emit(output + bias)
                if has_bn:
                    gamma = R.arg_("gamma", R.Tensor((weight_shape[0],), dtype))
                    beta = R.arg_("beta", R.Tensor((weight_shape[0],), dtype))
                    mean = R.arg_("mean", R.Tensor((weight_shape[0],), dtype))
                    variance = R.arg_("variance", R.Tensor((weight_shape[0],), dtype))
                    output = R.emit(
                        R.nn.batch_norm(
                            output,
                            gamma,
                            beta,
                            mean,
                            variance,
                            axis=1,
                            epsilon=1e-5,
                            training=False,
                        )[0]
                    )
                if has_activation:
                    output = R.emit(R.nn.relu(output))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_clml_conv2d_codegen(
    data_shape,
    weight_shape,
    stride,
    dilation,
    padding,
    weight_layout="OIHW",
    groups=1,
    dtype="float32",
    has_bias=False,
    has_bn=False,
    has_activation=False,
    has_pad=False,
    is_depthwise=False,
):
    kernel_h, kernel_w = weight_shape[2], weight_shape[3]
    channels = weight_shape[0]
    if len(padding) == 2:
        padding = (padding[0], padding[1], padding[0], padding[1])
    output_height = ((data_shape[2] - kernel_h + padding[0] + padding[2]) / stride[0]) + 1
    output_width = ((data_shape[3] - kernel_w + padding[1] + padding[3]) / stride[1]) + 1
    output_shape = (1, channels, int(output_height), int(output_width))
    out_dtype = dtype
    is_depthwise = data_shape[1] == channels == groups

    weight_layout = "IOHW" if is_depthwise else "OIHW"
    if weight_layout == "OIHW":
        weight_shape = (channels, data_shape[1] // groups, kernel_h, kernel_w)
    else:
        weight_shape = (data_shape[1] // groups, channels, kernel_h, kernel_w)

    if is_depthwise:
        name = "openclml.nn.depthwise_conv2d"
    else:
        name = "openclml.nn.conv2d"

    node = {
        "op": "kernel",
        "name": "",
        "inputs": [],
        "attrs": {
            "groups": groups,
            "num_outputs": 1,
            "data_layout": "NCHW",
            "kernel_layout": weight_layout,
            "dilation": dilation,
            "out_layout": "NCHW",
            "out_dtype": out_dtype,
            "shape": [list(output_shape)],
            "dtype": [dtype],
            "padding": padding,
            "strides": stride,
        },
    }

    if has_activation:
        node["attrs"]["activation_type"] = "relu"

    nodes = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [list(data_shape)], "dtype": [str(dtype)]},
        },
    ]

    nodes.append(
        {
            "op": "const",
            "name": "",
            "attrs": {"shape": [list(weight_shape)], "dtype": [str(dtype)]},
        }
    )

    if has_bias:
        bias_dtype = dtype
        nodes.append(
            {
                "op": "const",
                "name": "",
                "attrs": {
                    "shape": [[1, weight_shape[1] if is_depthwise else weight_shape[0], 1, 1]],
                    "dtype": [bias_dtype],
                },
            }
        )

    if has_bn:
        bn_shape = [1, weight_shape[0], 1, 1]
        # conv2d + bn --> conv2d + Add due to OptimizeBatchNorm transformation Pass
        nodes.append(
            {
                "name": "",
                "op": "const",
                "attrs": {"dtype": [dtype], "shape": [[1, weight_shape[0], 1, 1]]},
            },
        )

    input_idx = 0
    for _ in range(len(nodes)):
        node["inputs"].append([input_idx, 0, 0])
        input_idx += 1
    node["attrs"]["num_inputs"] = len(nodes)
    nodes.append(node)
    return nodes


def get_relax_conv2d_transpose_mod(
    data_shape,
    weight_shape,
    channels,
    stride,
    padding,
    dtype="float32",
):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            data = R.arg_("data", R.Tensor(data_shape, dtype))
            weight = R.arg_("weight", R.Tensor(weight_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d_transpose(
                        data,
                        weight,
                        groups=1,
                        strides=stride,
                        padding=padding,
                        kernel_layout="OIHW",
                        data_layout="NCHW",
                    )
                )
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_conv2d_transpose_expected_codegen(
    dshape, kshape, channels, kernel_size, strides, padding, dilation, dtype, output_shape
):
    attrs = {
        "data_layout": "NCHW",
        "kernel_layout": "OIHW",
        "groups": 1,
        "dilation": dilation,
        "num_inputs": 2,
        "num_outputs": 1,
        "padding": padding,
        "shape": [list(output_shape)],
        "dtype": [dtype],
        "strides": strides,
        "out_dtype": "",
        "out_layout": "NCHW",
        "output_padding": [0, 0],
    }

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [list(dshape)], "dtype": [str(dtype)]},
        },
        {
            "op": "const",
            "name": "",
            "attrs": {"shape": [list(kshape)], "dtype": [str(dtype)]},
        },
        {
            "op": "kernel",
            "name": "",
            "inputs": [[0, 0, 0], [1, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


def get_batchnorm_mod(data_shape, channels, axis, epsilon, dtype):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            data = R.arg_("data", R.Tensor(data_shape, dtype))
            gamma = R.arg_("gamma", R.Tensor((channels,), dtype))
            beta = R.arg_("beta", R.Tensor((channels,), dtype))
            mean = R.arg_("moving_mean", R.Tensor((channels,), dtype))
            variance = R.arg_("moving_var", R.Tensor((channels,), dtype))
            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.batch_norm(data, gamma, beta, mean, variance, axis, epsilon)[0]
                )
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

        func = builder.get()
        return tvm_tirx.IRModule({"main": func})


def get_binary_op_mod(a_shape, b_shape, op, dtype):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            a = R.arg_("a", R.Tensor(a_shape, dtype))
            b = R.arg_("b", R.Tensor(b_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(op(a, b))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()

    low, high = 0, 1
    a_data = np.random.uniform(low, high, size=(a_shape)).astype(dtype)
    b_data = np.random.uniform(low, high, size=(b_shape)).astype(dtype)

    return (tvm_tirx.IRModule({"main": func}), (a_data, b_data))


def get_unary_op_mod(a_shape, op, dtype):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            a = R.arg_("a", R.Tensor(a_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(op(a))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()

    low, high = 0, 1
    a_data = np.random.uniform(low, high, size=(a_shape)).astype(dtype)

    return (tvm_tirx.IRModule({"main": func}), (a_data,))


def get_relax_maxpool_mod(
    data_shape, dtype, pool_size, stride=None, dilation=(1, 1), padding=(0, 0), has_pad=False
):
    """
    Args:
        data_shape (tuple): Input tensor shape
        pool_size (tuple): Pooling window size (height, width)
        stride (tuple, optional): Stride of pooling operation. Defaults to pool_size.
        dilation (tuple, optional): Dilation rate. Defaults to (1, 1).
        padding (tuple, optional): Padding for the input tensor. Defaults to (0, 0).
        dtype (str, optional): Data type. Defaults to "float32".
        has_pad (bool, optional): Whether to apply explicit padding. Defaults to False.

    Returns:
        tvm_tirx.IRModule: Relax MaxPool module
    """
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")

            if has_pad:
                p = (0, 0, 0, 0, padding[0], padding[1], padding[0], padding[1])
                orig_data = R.arg_("data", R.Tensor(data_shape, dtype))
                data = R.nn.pad(orig_data, pad_width=p, pad_value=float("-inf"))
                padding = (0, 0)
            else:
                data = R.arg_("data", R.Tensor(data_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.max_pool2d(
                        data,
                        pool_size=pool_size,
                        strides=stride,
                        dilation=dilation,
                        padding=padding,
                        layout="NCHW",
                    )
                )
                R.output(output)
            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_maxpool_expected_codegen(input_shape, pool_size, stride, padding, pool_type, dtype):
    import math

    adjusted_input_shape = [
        input_shape[0],
        input_shape[1],
        input_shape[2] + padding[0] + padding[1],
        input_shape[3] + padding[2] + padding[3],
    ]

    pool_height = math.floor(((adjusted_input_shape[2] - pool_size[0]) / stride[0]) + 1)
    pool_width = math.floor(((adjusted_input_shape[3] - pool_size[1]) / stride[1]) + 1)
    output_shape = [adjusted_input_shape[0], adjusted_input_shape[1], pool_height, pool_width]

    attrs = {
        "ceil_mode": 0,
        "dilation": [1, 1],
        "layout": "NCHW",
        "num_inputs": 1,
        "num_outputs": 1,
        "out_layout": "NCHW",
        "padding": list(padding),
        "pool_size": pool_size,
        "shape": [list(output_shape)],
        "dtype": [dtype],
        "strides": stride,
        "count_include_pad": 0,
    }
    if sum(padding):
        attrs["count_include_pad"] = 0

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [list(adjusted_input_shape)], "dtype": [str(dtype)]},
        },
        {
            "op": "kernel",
            "name": "",
            "inputs": [[0, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


def get_relax_avgpool_mod(data_shape, dtype, pool_size, stride, dilation, padding, has_pad):
    """
    Args:
        data_shape (tuple): Input tensor shape
        pool_size (tuple): Pooling window size (height, width)
        stride (tuple, optional): Stride of pooling operation. Defaults to pool_size.
        dilation (tuple, optional): Dilation rate. Defaults to (1, 1).
        padding (tuple, optional): Padding for the input tensor. Defaults to (0, 0).
        dtype (str, optional): Data type. Defaults to "float32".
        has_pad (bool, optional): Whether to apply explicit padding. Defaults to False.
        count_include_pad (bool, optional): Whether to include padding in averaging. Defaults to True.

    Returns:
        tvm_tirx.IRModule: Relax AvgPool module
    """
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")

            if has_pad:
                p = (0, 0, 0, 0, padding[0], padding[1], padding[0], padding[1])
                orig_data = R.arg_("data", R.Tensor(data_shape, dtype))
                data = R.nn.pad(orig_data, pad_width=p, pad_value=0.0)
                padding = (0, 0)
            else:
                data = R.arg_("data", R.Tensor(data_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.avg_pool2d(
                        data,
                        pool_size=pool_size,
                        strides=stride,
                        dilation=dilation,
                        padding=padding,
                        layout="NCHW",
                    )
                )
                R.output(output)
            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_avgpool_expected_codegen(input_shape, pool_size, stride, padding, pool_type, dtype):
    import math

    adjusted_input_shape = [
        input_shape[0],
        input_shape[1],
        input_shape[2] + padding[0] + padding[1],
        input_shape[3] + padding[2] + padding[3],
    ]

    pool_height = math.floor(((adjusted_input_shape[2] - pool_size[0]) / stride[0]) + 1)
    pool_width = math.floor(((adjusted_input_shape[3] - pool_size[1]) / stride[1]) + 1)
    output_shape = [adjusted_input_shape[0], adjusted_input_shape[1], pool_height, pool_width]

    attrs = {
        "ceil_mode": 0,
        "dilation": [1, 1],
        "layout": "NCHW",
        "num_inputs": 1,
        "num_outputs": 1,
        "out_layout": "NCHW",
        "padding": list(padding),
        "pool_size": pool_size,
        "shape": [list(output_shape)],
        "dtype": [dtype],
        "strides": stride,
        "count_include_pad": 0,
    }
    if sum(padding):
        attrs["count_include_pad"] = 0

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [list(adjusted_input_shape)], "dtype": [str(dtype)]},
        },
        {
            "op": "kernel",
            "name": "",
            "inputs": [[0, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


def get_relax_reshape_mod(input_shape, output_shape, dtype):
    """
    Args:
        input_shape (tuple): Input tensor shape
        output_shape (tuple): Desired output tensor shape
        dtype (str, optional): Data type. Defaults to "float32".

    Returns:
        tvm_tirx.IRModule: Relax Reshape module
    """
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            data = R.arg_("data", R.Tensor(input_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(R.reshape(data, output_shape))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_relax_reshape_codegen(input_shape, output_shape, dtype):
    def compute_output_shape(input_shape, output_shape):
        input_elements = np.prod(input_shape)
        specified_elements = np.prod([dim for dim in output_shape if dim != -1])
        missing_dim = input_elements // specified_elements
        return [int(dim) if dim != -1 else int(missing_dim) for dim in output_shape]

    expected_output_shape = compute_output_shape(input_shape, output_shape)

    expected_codegen_str = [
        {
            "attrs": {
                "dtype": [dtype],
                "shape": [list(input_shape)],
            },
            "name": "",
            "op": "input",
        },
        {
            "attrs": {
                "dtype": [dtype],
                "num_inputs": 1,
                "num_outputs": 1,
                "shape": [expected_output_shape],
            },
            "inputs": [[0, 0, 0]],
            "name": "",
            "op": "kernel",
        },
    ]
    return expected_codegen_str


def get_relax_global_avgpool_mod(data_shape, keepdims, dtype):
    """
    Create a Relax module for Global Average Pooling (GAP).

    Args:
        data_shape (tuple): Input tensor shape (N, C, H, W)
        dtype (str): Data type

    Returns:
        tvm_tirx.IRModule: Relax GAP module
    """
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            data = R.arg_("data", R.Tensor(data_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(R.mean(data, axis=[2, 3], keepdims=keepdims))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_global_avgpool_expected_codegen(input_shape, keep_dims, dtype):
    """
    Generate expected codegen for Global Average Pooling.

    Args:
        input_shape (tuple): Input shape (N, C, H, W)
        dtype (str): Data type

    Returns:
        dict: Expected codegen output
    """
    output_shape = (
        [input_shape[0], input_shape[1]]
        if not keep_dims
        else [input_shape[0], input_shape[1], 1, 1]
    )
    attrs = {
        "num_inputs": 1,
        "num_outputs": 1,
        "shape": [list(output_shape)],
        "dtype": [dtype],
        "axis": [2, 3],
        "keepdims": 1 if keep_dims else 0,
    }

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [list(input_shape)], "dtype": [str(dtype)]},
        },
        {"op": "kernel", "name": "", "inputs": [[0, 0, 0]], "attrs": attrs},
    ]
    return exp_codegen


def get_relax_global_maxpool_mod(data_shape, keepdims, dtype):
    """
    Create a Relax module for Global Average Pooling (GAP).

    Args:
        data_shape (tuple): Input tensor shape (N, C, H, W)
        dtype (str): Data type

    Returns:
        tvm_tirx.IRModule: Relax GAP module
    """
    N, C, H, W = data_shape
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            data = R.arg_("data", R.Tensor(data_shape, dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.max_pool2d(
                        data, pool_size=(H, W), strides=(1, 1), padding=(0, 0), layout="NCHW"
                    )
                )
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_global_maxpool_expected_codegen(input_shape, pool_size, stride, padding, pool_type, dtype):
    import math

    adjusted_input_shape = [
        input_shape[0],
        input_shape[1],
        input_shape[2] + padding[0] + padding[1],
        input_shape[3] + padding[2] + padding[3],
    ]

    output_shape = [adjusted_input_shape[0], adjusted_input_shape[1], 1, 1]

    attrs = {
        "ceil_mode": 0,
        "dilation": [1, 1],
        "layout": "NCHW",
        "num_inputs": 1,
        "num_outputs": 1,
        "out_layout": "NCHW",
        "padding": padding,
        "pool_size": pool_size,
        "shape": [list(output_shape)],
        "dtype": [dtype],
        "strides": stride,
        "count_include_pad": 0,
    }
    if sum(padding):
        attrs["count_include_pad"] = 0

    exp_codegen = [
        {
            "op": "input",
            "name": "",
            "attrs": {"shape": [list(adjusted_input_shape)], "dtype": [str(dtype)]},
        },
        {
            "op": "kernel",
            "name": "",
            "inputs": [[0, 0, 0]],
            "attrs": attrs,
        },
    ]
    return exp_codegen


def get_dequant_matmul_module(K, N):
    seq_len = T.dynamic("seq_len", "int64")

    @I.ir_module
    class DequantMatmul:
        @R.function
        def main(
            input: R.Tensor((1, seq_len, K), dtype="float16"),
            weight: R.Tensor((K // 8, N), dtype="uint32"),
            scale: R.Tensor((K // 32, N), dtype="float16"),
        ):
            cls = DequantMatmul
            with R.dataflow():
                lv2 = relax.call_tir(
                    cls.dequantize,
                    (weight, scale),
                    ty_args=[R.Tensor((K, N), dtype="float16")],
                )
                gv: R.Tensor((1, seq_len, N), dtype="float16") = relax.op.matmul(
                    input, lv2, out_dtype="float16"
                )
                R.output(gv)
            return gv

        @Ts.function
        def dequantize(
            lm_head_q_weight1: T.Tensor((T.int64(K // 8), T.int64(N)), "uint32"),
            lm_head_q_scale1: T.Tensor((T.int64(K // 32), T.int64(N)), "float16"),
            dequantize: T.Tensor((T.int64(K), T.int64(N)), "float16"),
        ):
            T.func_attr({"tirx.noalias": T.bool(True)})

            # with Ts.sblock("root"):
            compute = T.alloc_tensor(ty_args=[T.Tensor((T.int64(K), T.int64(N)), "float16")])
            for i0, i1 in T.grid(T.int64(K), T.int64(N)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(lm_head_q_weight1[v_i0 // T.int64(8), v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Cast(
                        "float16",
                        T.bitwise_and(
                            T.shift_right(
                                lm_head_q_weight1[v_i0 // T.int64(8), v_i1],
                                T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4)),
                            ),
                            T.uint32(15),
                        ),
                    )
            for i0, i1 in T.grid(T.int64(K), T.int64(N)):
                with Ts.sblock("dequantize"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(compute[v_i0, v_i1], lm_head_q_scale1[v_i0 // T.int64(32), v_i1])
                    Ts.writes(dequantize[v_i0, v_i1])
                    dequantize[v_i0, v_i1] = (
                        compute[v_i0, v_i1] - T.float16(7.0)
                    ) * lm_head_q_scale1[v_i0 // T.int64(32), v_i1]

    return DequantMatmul


def get_dequant_vec_matmul_module(K, N):
    vocab_size_main = T.dynamic("vocab_size", "int64")
    vocab_size_dequantize = T.dynamic("vocab_size", "int64")

    @I.ir_module
    class DequantVecMatmul:
        @R.function
        def main(
            input: R.Tensor((1, 1, K), dtype="float16"),
            weight: R.Tensor((K // 8, vocab_size_main), dtype="uint32"),
            scale: R.Tensor((K // 32, vocab_size_main), dtype="float16"),
        ):
            cls = DequantVecMatmul
            with R.dataflow():
                lv2 = relax.call_tir(
                    cls.dequantize,
                    (weight, scale),
                    ty_args=[R.Tensor((K, vocab_size_main), dtype="float16")],
                )
                gv: R.Tensor((1, 1, vocab_size_main), dtype="float16") = relax.op.matmul(
                    input, lv2, out_dtype="float16"
                )
                R.output(gv)
            return gv

        @Ts.function
        def dequantize(
            lm_head_q_weight1: T.Tensor((T.int64(K // 8), vocab_size_dequantize), "uint32"),
            lm_head_q_scale1: T.Tensor((T.int64(K // 32), vocab_size_dequantize), "float16"),
            dequantize: T.Tensor((T.int64(K), vocab_size_dequantize), "float16"),
        ):
            T.func_attr({"tirx.noalias": T.bool(True)})

            # with Ts.sblock("root"):
            compute = T.alloc_tensor(
                ty_args=[T.Tensor((T.int64(K), vocab_size_dequantize), "float16")]
            )
            for i0, i1 in T.grid(T.int64(K), vocab_size_dequantize):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(lm_head_q_weight1[v_i0 // T.int64(8), v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Cast(
                        "float16",
                        T.bitwise_and(
                            T.shift_right(
                                lm_head_q_weight1[v_i0 // T.int64(8), v_i1],
                                T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4)),
                            ),
                            T.uint32(15),
                        ),
                    )
            for i0, i1 in T.grid(T.int64(K), vocab_size_dequantize):
                with Ts.sblock("dequantize"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(compute[v_i0, v_i1], lm_head_q_scale1[v_i0 // T.int64(32), v_i1])
                    Ts.writes(dequantize[v_i0, v_i1])
                    dequantize[v_i0, v_i1] = (
                        compute[v_i0, v_i1] - T.float16(7.0)
                    ) * lm_head_q_scale1[v_i0 // T.int64(32), v_i1]

    return DequantVecMatmul
