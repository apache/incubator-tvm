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
# ruff: noqa: RUF005
"""Utilities to construct matmul workloads."""

import tvm_tirx
from tvm.relax import script as R
from tvm.relax.script import ir_builder as relax_builder
from tvm_tirx.script.ir_builder import IRBuilder


def get_relax_matmul_module(
    x_shape,
    y_shape,
    in_dtype,
    out_dtype=None,
    transposed_y=False,
    bias_shape=None,
    activation=None,
    residual_bin_op=None,
    residual_activation=None,
):
    """Create a matmul op followd by epilogue operations."""
    out_dtype = out_dtype if out_dtype is not None else in_dtype
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            x = R.arg_("x", R.Tensor(x_shape, in_dtype))
            y = R.arg_("y", R.Tensor(y_shape, in_dtype))
            if bias_shape is not None:
                bias = R.arg_("bias", R.Tensor(bias_shape, out_dtype))

            with R.dataflow() as frame:
                if transposed_y:
                    axes = list(range(len(y_shape) - 2)) + [-1, -2]
                    y = R.emit(R.permute_dims(y, axes=axes))
                result = R.emit(R.matmul(x, y, out_dtype=out_dtype))
                if bias_shape is not None:
                    result = R.emit(result + bias)
                if activation is not None:
                    result = R.emit(activation(result))
                if residual_bin_op is not None:
                    result = R.emit(residual_bin_op(result, x))
                    if residual_activation is not None:
                        result = R.emit(residual_activation(result))
                R.output(result)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})
