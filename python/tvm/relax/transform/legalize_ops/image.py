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
# pylint: disable=invalid-name
"""Default legalization function for image operators."""

import tvm_tirx
from tvm import topi
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr
from .common import register_legalize


@register_legalize("relax.image.resize2d")
def _image_resize2d(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.image.resize2d,
        call.args[0],
        roi=call.attrs.roi,
        size=call.args[1],
        layout=call.attrs.layout,
        method=call.attrs.method,
        coordinate_transformation_mode=call.attrs.coordinate_transformation_mode,
        rounding_method=call.attrs.rounding_method,
        bicubic_alpha=call.attrs.cubic_alpha,
        bicubic_exclude=call.attrs.cubic_exclude,
        extrapolation_value=call.attrs.extrapolation_value,
    )


@register_legalize("relax.image.grid_sample")
def _image_grid_sample(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.image.grid_sample,
        call.args[0],
        call.args[1],
        method=call.attrs.method,
        layout=call.attrs.layout,
        padding_mode=call.attrs.padding_mode,
        align_corners=call.attrs.align_corners,
    )


@register_legalize("relax.image.affine_grid")
def _image_affine_grid(bb: BlockBuilder, call: Call) -> Expr:
    for v in call.args[1].values:
        if not isinstance(v, int | tvm_tirx.ir.IntImm):
            raise ValueError(
                f"affine_grid legalization requires static target_shape, got symbolic value: {v}"
            )
    target_shape = [int(v) for v in call.args[1].values]
    return bb.call_te(
        topi.image.affine_grid,
        call.args[0],
        target_shape=target_shape,
        align_corners=call.attrs.align_corners,
    )


@register_legalize("relax.image.resize3d")
def _image_resize3d(bb: BlockBuilder, call: Call) -> Expr:
    return bb.call_te(
        topi.image.resize3d,
        call.args[0],
        roi=call.attrs.roi,
        size=call.args[1],
        layout=call.attrs.layout,
        method=call.attrs.method,
        coordinate_transformation_mode=call.attrs.coordinate_transformation_mode,
        rounding_method=call.attrs.rounding_method,
        bicubic_alpha=call.attrs.cubic_alpha,
        bicubic_exclude=call.attrs.cubic_exclude,
        extrapolation_value=call.attrs.extrapolation_value,
    )
