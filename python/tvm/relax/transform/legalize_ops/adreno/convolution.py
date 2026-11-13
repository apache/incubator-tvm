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
"""A Convolution impl for Adreno GPU."""

from tvm import relax, topi


def conv2d_NCHWc_OIHWo(bb: relax.BlockBuilder, call: relax.Call) -> relax.Expr:
    return bb.call_te(
        topi.nn.conv2d_NCHWc_OIHWo,
        data=call.args[0],
        kernel=call.args[1],
        stride=call.attrs.strides,
        padding=call.attrs.padding,
        dilation=call.attrs.dilation,
        layout=call.attrs.data_layout,
        out_layout=call.attrs.out_layout,
        # out_dtype=call.attrs.out_dtype,
        ty_args=call.ty_args,
        function_name_hint="conv2d_NCHWc_OIHWo",
    )
