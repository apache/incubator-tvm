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
"""Distributed Relax expression operators."""

from tvm.relax.distributed import DeviceMesh, Placement
from tvm.relax.expr import Expr
from tvm.relax.op import call_tir as call_tir
from tvm.relax.op.distributed import annotate_sharding as _annotate_sharding
from tvm.relax.op.distributed import call_tir_local_view, redistribute_replica_to_shard
from tvm.relax.op.distributed import redistribute as _redistribute
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from .ir import _lookup_device_mesh

py_str = str


def annotate_sharding(
    value: Expr,
    device_mesh: py_str | DeviceMesh,
    placement: py_str | Placement,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    if isinstance(device_mesh, py_str):
        device_mesh = _lookup_device_mesh(device_mesh)
    if isinstance(placement, py_str):
        placement = Placement.from_text(placement)
    return _annotate_sharding(value, device_mesh, placement, ty=ty, loc=loc)


def redistribute(
    value: Expr,
    device_mesh: py_str | DeviceMesh,
    placement: py_str | Placement,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    if isinstance(device_mesh, py_str):
        device_mesh = _lookup_device_mesh(device_mesh)
    if isinstance(placement, py_str):
        placement = Placement.from_text(placement)
    return _redistribute(value, device_mesh, placement, ty=ty, loc=loc)


__all__ = [
    "annotate_sharding",
    "call_tir",
    "call_tir_local_view",
    "redistribute",
    "redistribute_replica_to_shard",
]
