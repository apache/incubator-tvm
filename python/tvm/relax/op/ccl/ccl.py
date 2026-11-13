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
"""Relax Collective Communications Library (CCL) operators"""

import tvm_ffi

from tvm_tirx.ir import Attrs
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ...expr import Expr


@tvm_ffi.register_object("relax.attrs.AllReduceAttrs")
class AllReduceAttrs(Attrs):
    """Attributes used in allreduce operator"""


def allreduce(
    x, op_type: str = "sum", in_group: bool = True, *, ty=None, loc: Location = UNKNOWN_LOC
):  # pylint: disable=invalid-name
    """Allreduce operator

    Parameters
    ----------
    x : relax.Expr
      The input tensor.

    op_type : str
      The type of reduction operation to be applied to the input data.
      Now "sum", "prod", "min", "max" and "avg" are supported.

    in_group : bool
      Whether the reduction operation performs globally or in group as default.

    Returns
    -------
    result : relax.Expr
      The result of allreduce.
    """
    supported_op_types = ["sum", "prod", "min", "max", "avg"]
    assert op_type in supported_op_types, (
        "Allreduce only supports limited reduction operations, "
        f"including {supported_op_types}, but got {op_type}."
    )
    return _Call(
        "relax.ccl.allreduce",
        [x],
        attrs=_make_attrs("relax.attrs.AllReduceAttrs", op_type=op_type, in_group=in_group),
        loc=loc,
        ty=ty,
    )  # type: ignore # pylint: disable=no-member


@tvm_ffi.register_object("relax.attrs.AllGatherAttrs")
class AllGatherAttrs(Attrs):
    """Attributes used in allgather operator"""


def allgather(x, num_workers: int, in_group: bool = True, *, ty=None, loc: Location = UNKNOWN_LOC):  # pylint: disable=invalid-name
    """AllGather operator

    Parameters
    ----------
    x : relax.Expr
      The input tensor.

    num_worker : int
      The number of workers to gather data from.

    in_group : bool
      Whether the gather operation performs globally or in group as default.

    Returns
    -------
    result : relax.Expr
      The result of allgather.
    """
    return _Call(
        "relax.ccl.allgather",
        [x],
        attrs=_make_attrs("relax.attrs.AllGatherAttrs", num_workers=num_workers, in_group=in_group),
        loc=loc,
        ty=ty,
    )  # type: ignore # pylint: disable=no-member


def broadcast_from_worker0(x: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Broadcast data from worker-0 to all other workers.

    Parameters
    ----------
    x : relax.Expr
      The tensor to be broadcast.

    Returns
    -------
    result : relax.Expr
      The same tensor, which has been broadcast to all other workers.
    """
    return _Call(
        "relax.ccl.broadcast_from_worker0",
        [x],
        loc=loc,
        ty=ty,
    )


@tvm_ffi.register_object("relax.attrs.ScatterCollectiveAttrs")
class ScatterCollectiveAttrs(Attrs):
    """Attributes used in scatter collective operators"""


def scatter_from_worker0(
    x: Expr, num_workers: int, axis: int = 0, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Perform a scatter operation from worker-0, chunking the given buffer into equal parts.

    Parameters
    ----------
    x : relax.Expr
      The buffer to be divided into equal parts and sent to each worker accordingly.

    num_worker : int
      The number of workers, i.e. the number of parts the given buffer should be chunked into.

    axis : int
      The dimension of the tensor to be scattered. Default is 0.

    Returns
    -------
    result : relax.Expr
      Chunked Tensor received by different workers.
    """
    return _Call(
        "relax.ccl.scatter_from_worker0",
        [x],
        attrs=_make_attrs("relax.attrs.ScatterCollectiveAttrs", num_workers=num_workers, axis=axis),
        loc=loc,
        ty=ty,
    )
