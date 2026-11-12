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
# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
# pylint: disable=broad-exception-raised, use-list-literal
"""Tensorflow lite frontend helper to parse custom options in Flexbuffer format."""

import struct
from enum import IntEnum


class BitWidth(IntEnum):
    """Flexbuffer bit width schema from flexbuffers.h"""

    BIT_WIDTH_8 = 0
    BIT_WIDTH_16 = 1
    BIT_WIDTH_32 = 2
    BIT_WIDTH_64 = 3


class FlexBufferType(IntEnum):
    """Flexbuffer type schema from flexbuffers.h"""

    FBT_NULL = 0
    FBT_INT = 1
    FBT_UINT = 2
    FBT_FLOAT = 3
    # Types above stored inline, types below store an offset.
    FBT_KEY = 4
    FBT_STRING = 5
    FBT_INDIRECT_INT = 6
    FBT_INDIRECT_UINT = 7
    FBT_INDIRECT_FLOAT = 8
    FBT_MAP = 9
    FBT_VECTOR = 10  # Untyped.
    FBT_VECTOR_INT = 11  # Typed any size (stores no type table).
    FBT_VECTOR_UINT = 12
    FBT_VECTOR_FLOAT = 13
    FBT_VECTOR_KEY = 14
    FBT_VECTOR_STRING = 15
    FBT_VECTOR_INT2 = 16  # Typed tuple (no type table, no size field).
    FBT_VECTOR_UINT2 = 17
    FBT_VECTOR_FLOAT2 = 18
    FBT_VECTOR_INT3 = 19  # Typed triple (no type table, no size field).
    FBT_VECTOR_UINT3 = 20
    FBT_VECTOR_FLOAT3 = 21
    FBT_VECTOR_INT4 = 22  # Typed quad (no type table, no size field).
    FBT_VECTOR_UINT4 = 23
    FBT_VECTOR_FLOAT4 = 24
    FBT_BLOB = 25
    FBT_BOOL = 26
    FBT_VECTOR_BOOL = 36  # To Allow the same type of conversion of type to vector type


class FlexBufferDecoder:
    """
    This implements partial flexbuffer deserialization to be able
    to read custom options. It is not intended to be a general
    purpose flexbuffer deserializer and as such only supports a
    limited number of types and assumes the data is a flat map.
    """

    def __init__(self, buffer):
        self.buffer = buffer

    def indirect_jump(self, offset, byte_width):
        """Helper function to read the offset value and jump"""
        unpack_str = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[byte_width]
        back_jump = struct.unpack(unpack_str, self.buffer[offset : offset + byte_width])[0]
        return offset - back_jump

    def decode_keys(self, end, size, byte_width):
        """Decodes the flexbuffer type vector. Map keys are stored in this form"""
        # Keys are strings here. The format is all strings separated by null, followed by back
        # offsets for each of the string. For example, (str1)\0(str1)\0(offset1)(offset2) The end
        # pointer is pointing at the end of all strings
        keys = list()
        for i in range(0, size):
            offset_pos = end + i * byte_width
            start_index = self.indirect_jump(offset_pos, byte_width)
            str_size = self.buffer[start_index:].find(b"\0")
            assert str_size != -1
            s = self.buffer[start_index : start_index + str_size].decode("utf-8")
            keys.append(s)
        return keys

    def decode_vector(self, end, size, byte_width):
        """Decodes the flexbuffer vector"""
        # Each entry in the vector can have different datatype. Each entry is of fixed length. The
        # format is a sequence of all values followed by a sequence of datatype of all values. For
        # example - (4)(3.56)(int)(float) The end here points to the start of the values.
        # Each type byte contains: (type << 2) | bit_width, where bit_width determines actual size.
        values = list()
        for i in range(0, size):
            value_type_pos = end + size * byte_width + i
            value_type_packed = self.buffer[value_type_pos]
            value_type = FlexBufferType(value_type_packed >> 2)
            value_bit_width = BitWidth(value_type_packed & 3)
            value_byte_width = 1 << value_bit_width
            value_bytes = self.buffer[
                end + i * byte_width : end + i * byte_width + value_byte_width
            ]
            if value_type == FlexBufferType.FBT_BOOL:
                value = bool(value_bytes[0])
            elif value_type == FlexBufferType.FBT_INT:
                fmt = {1: "<b", 2: "<h", 4: "<i", 8: "<q"}[value_byte_width]
                value = struct.unpack(fmt, value_bytes)[0]
            elif value_type == FlexBufferType.FBT_UINT:
                fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[value_byte_width]
                value = struct.unpack(fmt, value_bytes)[0]
            elif value_type == FlexBufferType.FBT_FLOAT:
                fmt = {4: "<f", 8: "<d"}[value_byte_width]
                value = struct.unpack(fmt, value_bytes)[0]
            else:
                raise Exception
            values.append(value)
        return values

    def decode_map(self, end, byte_width, parent_byte_width):
        """Decodes the flexbuffer map and returns a dict"""
        mid_loc = self.indirect_jump(end, parent_byte_width)
        size_fmt = {1: "<b", 2: "<h", 4: "<i", 8: "<q"}[byte_width]
        map_size = struct.unpack(size_fmt, self.buffer[mid_loc - byte_width : mid_loc])[0]

        # Find keys. The map header also stores the byte width of the keys vector, which is
        # wider than one byte once a key string sits more than 255 bytes before it.
        keys_offset = mid_loc - byte_width * 3
        keys_end = self.indirect_jump(keys_offset, byte_width)
        keys_byte_width_fmt = {1: "<B", 2: "<H", 4: "<I", 8: "<Q"}[byte_width]
        keys_byte_width = struct.unpack(
            keys_byte_width_fmt, self.buffer[mid_loc - byte_width * 2 : mid_loc - byte_width]
        )[0]
        keys = self.decode_keys(keys_end, map_size, keys_byte_width)

        # Find values
        values_end = self.indirect_jump(end, parent_byte_width)
        values = self.decode_vector(values_end, map_size, byte_width)
        return dict(zip(keys, values))

    def decode(self):
        """Decode the buffer. Decoding is partially implemented"""
        root_end = len(self.buffer) - 1
        root_byte_width = self.buffer[root_end]
        root_end -= 1
        root_packed_type = self.buffer[root_end]
        root_end -= root_byte_width

        root_type = FlexBufferType(root_packed_type >> 2)
        byte_width = 1 << BitWidth(root_packed_type & 3)

        if root_type == FlexBufferType.FBT_MAP:
            return self.decode_map(root_end, byte_width, root_byte_width)
        raise NotImplementedError("Flexbuffer Decoding is partially imlpemented.")
