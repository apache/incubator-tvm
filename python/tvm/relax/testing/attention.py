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

"""Relax script for attention module."""

import tvm_tirx
from tvm.relax import script as R
from tvm.relax.script import ir_builder as relax_builder
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder


def get_relax_attention_module(
    q_shape,
    k_shape,
    v_shape,
    *,
    dtype,
    bias_shape=None,
    qk_scale=None,
    causal_mask=None,
    window_size=None,
):  # pylint: disable=too-many-arguments, too-many-locals, invalid-name
    """Get a relax module for attention."""

    if qk_scale is not None:
        qk_scale = T.FloatImm("float32", qk_scale)

    if window_size is not None:
        window_size = T.IntImm("int32", window_size)

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            q = R.arg_("q", R.Tensor(q_shape, dtype))
            k = R.arg_("k", R.Tensor(k_shape, dtype))
            v = R.arg_("v", R.Tensor(v_shape, dtype))
            bias = None
            if bias_shape is not None and bias_shape != "none":
                bias = R.arg_("bias", R.Tensor(bias_shape, dtype))

            with R.dataflow() as frame:
                result = R.emit(R.nn.attention(q, k, v, bias, qk_scale, causal_mask, window_size))
                R.output(result)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_relax_stacked_attention_module(
    qkv,
    b,
    s,
    n,
    h,
    h_v,
    op,
    bias=None,
    qk_scale=None,
    single_shape=False,
    layout="BS3NH",
    causal_mask=None,
):  # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, invalid-name
    # pylint: disable=too-many-statements
    """Get a relax module for stacked attention."""
    dtype = str(qkv.dtype)
    assert layout in ["BS3NH", "SBN3H"]

    if qk_scale is not None:
        qk_scale = T.FloatImm("float32", qk_scale)

    if single_shape:
        if layout == "BS3NH":
            qk_shape = R.shape([b, s, n, h])
        elif layout == "SBN3H":
            qk_shape = R.shape([b, s, n, h])
        v_shape = qk_shape
    else:
        if layout == "BS3NH":
            qk_shape = [b, s, n, h]
            v_shape = [b, s, n, h_v]
        elif layout == "SBN3H":
            qk_shape = [s, b, n, h]
            v_shape = [s, b, n, h_v]

    if layout == "BS3NH":
        split_axis = 2
        split_sections = [n * h, n * h * 2]
    elif layout == "SBN3H":
        split_axis = 3
        split_sections = [h, h * 2]

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            qkv = R.arg_("qkv", R.Tensor(qkv.shape, dtype))
            if bias is not None:
                bias = R.arg_("bias", R.Tensor(bias.shape, dtype))
            with R.dataflow() as frame:
                if op == "split":
                    qkv_tuple = R.split(qkv, split_sections, axis=split_axis)
                    q = qkv_tuple[0]
                    k = qkv_tuple[1]
                    v = qkv_tuple[2]
                elif op == "strided_slice":
                    q = R.strided_slice(qkv, [split_axis], [0], [split_sections[0]], [1])
                    k = R.strided_slice(
                        qkv, [split_axis], [split_sections[0]], [split_sections[1]], [1]
                    )
                    v = R.strided_slice(
                        qkv,
                        [split_axis],
                        [split_sections[1]],
                        [int(qkv.ty.shape[split_axis])],
                        [1],
                    )
                else:
                    raise NotImplementedError()
                if layout == "BS3NH":
                    q = R.reshape(q, qk_shape)
                    k = R.reshape(k, qk_shape)
                    v = R.reshape(v, v_shape)
                elif layout == "SBN3H":
                    q = R.permute_dims(q, [1, 0, 2, 3])
                    k = R.permute_dims(k, [1, 0, 2, 3])
                    v = R.permute_dims(v, [1, 0, 2, 3])
                result = R.emit(R.nn.attention(q, k, v, bias, qk_scale, causal_mask))
                if layout == "SBN3H":
                    result = R.emit(R.permute_dims(result, [1, 0, 2, 3]))
                R.output(result)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})
