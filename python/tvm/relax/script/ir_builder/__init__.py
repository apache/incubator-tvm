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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name
"""Public exports for the Relax builder."""

import sys as _sys

from tvm_tirx.script.ir_builder.ir import constexpr
from tvm_tirx.script.parser import register_namespace as _register_namespace
from tvm_tirx.tirx.script.ir_builder import expose_script_op as _expose_script_op

from . import distributed as dist
from . import ir as _native
from . import op as _op
from . import parser_protocol as _protocol
from .ir import *
from .op import *
from .parser_protocol import *
from .parser_protocol import (
    __tvm_value_if__,
    supports_mutable_declarations,
)

__all__ = [*_native.__all__, *_op.__all__, *_protocol.__all__, "constexpr", "dist"]

# Install explicit builders before requesting generated inspection operators.
_register_namespace("relax", _sys.modules[__name__])
for _name in (
    "tensor_dtype_code",
    "tensor_dtype_bits",
    "tensor_dtype_lanes",
    "tensor_ndim",
    "tensor_byte_offset",
    "tensor_elem_offset",
    "tensor_shape_i",
    "tensor_stride_i",
):
    _expose_script_op(f"relax.inspect.{_name}")
