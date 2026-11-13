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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Utility functions for bitserial operators"""

import numpy as np

import tvm_tirx
from tvm import te
from tvm.topi.transform import concatenate

from ..utils import get_const_int


def bitpack(data, bits, pack_axis, bit_axis, pack_type, name="QuantizeInput"):
    """Packs data into format necessary for bitserial computation

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tvm tensor
    bits : int
        Number of bits to use for packing
    pack_axis : int
       index of the axis to pack in data
    bit_axis : int
       index of axis to place bit axis in resulting packed data
    pack_type : str
        Data type for packing, must be one of: ['uint8', 'uint16', 'uint32', 'uint64']
    name : Optional[str] = "QuantizeInput"
        Name for the operation
    """
    ishape = data.shape
    n = len(ishape)
    if pack_type == "uint8":
        data_width = 8
    elif pack_type == "uint16":
        data_width = 16
    elif pack_type == "uint32":
        data_width = 32
    elif pack_type == "uint64":
        data_width = 64

    # Data must be in multiples of the data_width
    assert get_const_int(ishape[pack_axis]) % data_width == 0, "Not a multiple of word size"

    shape_vec = list(ishape)
    shape_vec[pack_axis] = shape_vec[pack_axis] // data_width
    shape_vec.insert(bit_axis, 1)
    bitserial_oshape = tuple(shape_vec)
    masks = np.array([0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80])

    # pack axis shifts if bit axis comes before
    if bit_axis <= pack_axis:
        pack_axis += 1

    def _bitpack(*indices):
        packed_data = [tvm_tirx.tirx.const(0, pack_type)] * bits
        for k in range(data_width):
            # Translate indices for packed data back to original
            idx = [0] * n
            j = 0
            for i in range(n + 1):
                if i == bit_axis:
                    continue
                if i == pack_axis:
                    idx[j] = indices[i] * data_width + k
                else:
                    idx[j] = indices[i]
                j += 1

            element = data(*idx)
            for b in range(bits):
                extracted_bit = ((element & tvm_tirx.tirx.const(masks[b], "int32")) >> b).astype(
                    pack_type
                )
                packed_data[b] = packed_data[b] | extracted_bit
                if k < data_width - 1:
                    packed_data[b] = packed_data[b] << 1

            if k == data_width - 1:
                return tuple(packed_data)
        return tuple(packed_data)

    output_tuple = te.compute(bitserial_oshape, _bitpack, name=name, tag="bitpack")

    if bits > 1:
        return concatenate(output_tuple, axis=bit_axis)
    return output_tuple


def binary_op_multiplier(pack_dtype):
    """ "Returns number of bits packed into
    pack_dtype: string
        pack type for the operator (must be a uint)"""
    return int(pack_dtype[4:])
