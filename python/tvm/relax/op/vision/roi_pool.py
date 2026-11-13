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
"""ROI Pool operator"""

import tvm_ffi

from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC as _UNKNOWN_LOC
from tvm_tirx.ir.location import Location as _Location

from ..base import Expr


@tvm_ffi.register_object("relax.attrs.ROIPoolAttrs")
class ROIPoolAttrs(Attrs):
    """Attributes for vision.roi_pool"""


def roi_pool(
    data: Expr,
    rois: Expr,
    pooled_size: int | tuple[int, int] | list[int],
    spatial_scale: float,
    layout: str = "NCHW",
    *,
    ty=None,
    loc: _Location = _UNKNOWN_LOC,
):
    """ROI Pool operator.

    Parameters
    ----------
    data : relax.Expr
        4-D input tensor.

    rois : relax.Expr
        2-D input tensor with shape `(num_roi, 5)` in
        `[batch_idx, x1, y1, x2, y2]` format.

    pooled_size : Union[int, Tuple[int, int], List[int]]
        Output pooled size.

    spatial_scale : float
        Ratio of input feature map height (or width) to raw image height (or width).

    layout : str, optional
        Layout of the input data. Currently only `NCHW` is supported.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(pooled_size, int):
        pooled_size = (pooled_size, pooled_size)
    return _Call(
        "relax.vision.roi_pool",
        [data, rois],
        attrs=_make_attrs(
            "relax.attrs.ROIPoolAttrs",
            pooled_size=pooled_size,
            spatial_scale=spatial_scale,
            layout=layout,
        ),
        loc=loc,
        ty=ty,
    )
