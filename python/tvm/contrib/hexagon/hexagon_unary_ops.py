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
# pylint: disable=missing-docstring, invalid-name
import logging

import numpy as np
from scipy import special

from tvm import te

logger = logging.getLogger(__name__)

######################################################################
#################### FUNCTION FOR LUT and Take Op ####################
######################################################################


def saturate(x: te.Tensor, dtype: str):
    """Saturate value for the specified data type"""
    return te.max(te.min_value(ty=dtype), te.min(x, te.max_value(ty=dtype)))


def hardswish_func(x):
    x_2 = np.add(x, 3.0)
    x_2 = np.clip(x_2, 0.0, 6.0)
    return x * x_2 / 6.0


def LUT_generation(inp_scale, inp_zp, out_scale, out_zp, op_name) -> None:
    LUT = []
    for i in range(256):
        i = np.int32(i)
        # converting the constants to the numpy value
        if inp_zp.data.shape == ():
            i_zp = inp_zp.data.numpy()[()]
        if inp_scale.data.shape == ():
            i_scale = inp_scale.data.numpy()[()]
        if out_zp.data.shape == ():
            o_zp = out_zp.data.numpy()[()]
        if out_scale.data.shape == ():
            o_scale = out_scale.data.numpy()[()]
        # Dequantization followed by computing the op value
        dequant = (i - i_zp) * i_scale
        if "tanh" in op_name:
            op_val = np.tanh(dequant)
        elif "rsqrt" in op_name:
            op_val = 1 / np.sqrt(dequant)
        elif "sqrt" in op_name:
            op_val = np.sqrt(dequant)
        elif "exp" in op_name:
            op_val = np.exp(dequant)
        elif "erf" in op_name:
            op_val = special.erf(dequant)
        elif "sigmoid" in op_name:
            op_val = 1 / (1 + np.exp(np.negative(dequant)))
        elif "hardswish" in op_name:
            op_val = hardswish_func(dequant)
        elif "log" in op_name:
            op_val = np.log(dequant)
        elif "abs" in op_name:
            op_val = np.abs(dequant)
        else:
            logger.error("Error op is other than unary op")

        # Quantizing the value generated and appending in the Look Up Table
        quant = np.round((op_val) / o_scale) + o_zp
        val = np.maximum(0, np.minimum(quant, 255)).astype(np.uint8)
        LUT.append(val)
    return LUT


def generate_take_function(inp, ty):
    # Generating the take op
    N, H, W, C = inp.ty.shape
    data = te.placeholder((N, H, W, C), dtype=ty.dtype, name="data")
    LUT_func = te.placeholder((256,), dtype="uint8", name="LUT")
    take = te.compute(
        ty.shape,
        lambda *indices: saturate((LUT_func[data[indices].astype("uint8")]), ty.dtype).astype(
            ty.dtype
        ),
        name="take_op",
    )
    mod = te.create_function([data, LUT_func, take])
    return mod
