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
"""Default legalization function for distir-related operators."""

from tvm import relax
from tvm_tirx import tirx
from tvm_tirx.ir import Call

from ...block_builder import BlockBuilder
from ...expr import Expr
from ...op import call_pure_packed
from ...type import ShapeType
from .common import register_legalize


@register_legalize("relax.dist.redistribute_replica_to_shard")
def _redistribute_replica_to_shard(_bb: BlockBuilder, call: Call) -> Expr:
    num_workers = call.attrs.num_workers
    axis = call.attrs.axis
    worker_id_symbol = tirx.Var("worker_id", "int64")
    worker_id_var = _bb.emit(call_pure_packed("runtime.disco.worker_id", ty_args=[ShapeType(None)]))
    _bb.match_cast(worker_id_var, ShapeType([worker_id_symbol]))

    split_axis_size = call.args[0].ty.shape[axis]
    return relax.op.strided_slice(
        call.args[0],
        axes=[axis],
        begin=[worker_id_symbol * split_axis_size // num_workers],
        end=[(worker_id_symbol + 1) * split_axis_size // num_workers],
        assume_inbound=True,
    )
