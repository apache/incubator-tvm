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
# pylint: disable=redefined-builtin
"""Operators for distributed Relax."""

from tvm.relax.distributed import DeviceMesh, Placement
from tvm_tirx.ir import Call, Op
from tvm_tirx.ir._expose_api import _make_op_api
from tvm_tirx.ir.attrs import make_node
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ...expr import Expr


def annotate_sharding(
    input: Expr,
    device_mesh: DeviceMesh,
    placement: Placement,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Annotate sharding plan for tensor

    Parameters
    ----------
    input : relax.Expr
      The input tensor.
    device_mesh: DeviceMesh
      The device mesh of the sharding plan
    placement: Placement
      The placement of the sharding plan

    Returns
    -------
    result : relax.Expr
      The tensor unmodified.
    """
    return Call(
        "relax.dist.annotate_sharding",
        [input],
        attrs=make_node(
            "relax.attrs.DistributionAttrs", device_mesh=device_mesh, placement=placement
        ),
        loc=loc,
        ty=ty,
    )  # type: ignore


def redistribute(
    input: Expr,
    device_mesh: DeviceMesh,
    placement: Placement,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Redistribute tensor

    Parameters
    ----------
    input : relax.Expr
      The input tensor.
    device_mesh: DeviceMesh
      The device mesh after redistribution
    placement: Placement
      The placement after redistribution
    Returns
    -------
    result : relax.Expr
      The tensor after redistribution.
    """
    return Call(
        "relax.dist.redistribute",
        [input],
        attrs=make_node(
            "relax.attrs.DistributionAttrs", device_mesh=device_mesh, placement=placement
        ),
        loc=loc,
        ty=ty,
    )  # type: ignore


call_tir_local_view = _make_op_api(Op.get("relax.dist.call_tir_local_view"), __name__)


def redistribute_replica_to_shard(
    input: Expr, num_workers: int, axis: int, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Slice tensor into several parts along one axis,
        and each worker takes one part.
        input.ty.shape[axis] % num_workers == 0 is required.
        Each worker must have an identical copy of the input.
        This is a specialized version of redistribute op.

    Parameters
    ----------
    input : relax.Expr
      The buffer to be sliced into equal parts.

    num_worker : int
      The number of workers, i.e. the number of parts the given buffer should be sliced into.

    axis : int
      The axis of the tensor to be sliced.

    Returns
    -------
    result : relax.Expr
      Sliced Tensor kept by each device.
    """
    return Call(
        "relax.dist.redistribute_replica_to_shard",
        [input],
        attrs=make_node("relax.attrs.ScatterCollectiveAttrs", num_workers=num_workers, axis=axis),
        loc=loc,
        ty=ty,
    )
