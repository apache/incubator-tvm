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
"""Analysis used in TensorIR scheduling"""

import tvm_ffi

from tvm_tirx.ir import For, Var
from tvm_tirx.tirx.expr import Expr
from tvm_tirx.tirx.function import Function
from tvm_tirx.tirx.index_map import IndexMap

from . import _ffi_api
from .schedule import SBlockRV, Schedule


def suggest_index_map(
    tensor: Var,
    indices: list[Expr],
    loops: list[For],
    predicate: Expr,
) -> IndexMap | None:
    """Provided the access pattern to a tensor, suggest one of the possible layout
    transformation to maximize the locality of the access pattern.

    Parameters
    ----------
    tensor : Var
        The tensor to be transformed.
    indices : List[Expr]
        The access pattern to the tensor.
    loops : List[For]
        The loops above the tensor.
    predicate : Expr
        The predicate of the access.

    Returns
    -------
    index_map : Optional[IndexMap]
        The suggested index map. None if no transformation is suggested.
    """
    return _ffi_api.SuggestIndexMap(  # type: ignore # pylint: disable=no-member
        tensor,
        indices,
        loops,
        predicate,
    )


@tvm_ffi.register_object("s_tir.schedule.TensorizeInfo")
class TensorizeInfo(tvm_ffi.Object):
    """Necessary information used for tensorization."""


def get_tensorize_loop_mapping(
    sch: Schedule, block: SBlockRV, desc_func: Function, allow_padding: bool = False
) -> TensorizeInfo | None:
    """Establish a mapping between loops in a target block and an intrinsic description

    Parameters
    ----------
    sch : Schedule
        The schedule to be tensorized
    block : SBlockRV
        The target block to match against
    desc_func : Function
        The function describing the computation to be tensorized
    allow_padding : bool
        Whether to allow padding the block iters to match the intrinsic description
    Returns
    -------
    tensorize_info : Optional[TensorizeInfo]
        TensorizeInfo structure if a valid mapping is found, None otherwise
    """
    return _ffi_api.GetTensorizeLoopMapping(sch, block, desc_func, allow_padding)  # type: ignore


@tvm_ffi.register_object("s_tir.schedule.AutoTensorizeMappingInfo")
class AutoTensorizeMappingInfo(tvm_ffi.Object):
    """Necessary information used to perform transformations for tensorization."""


def get_auto_tensorize_mapping_info(
    sch: Schedule, block: SBlockRV, desc_func: Function
) -> AutoTensorizeMappingInfo | None:
    """Get mapping info between a target block and an intrinsic description including layout
    transformations to apply.

    Parameters
    ----------
    sch : Schedule
        The schedule to be tensorized
    block : SBlockRV
        The compute block for auto tensorization
    desc_func : Function
        The function describing the computation to be tensorized

    Returns
    -------
    auto_tensorize_mapping_info : Optional[AutoTensorizeMappingInfo]
        AutoTensorizeMappingInfo structure if potential mappings found, None otherwise.

    Note
    ----
    Returning a valid AutoTensorizeMappingInfo doesn't guarantee the block can be tensorized.
    We will need to apply the suggested layout transformations and then match against the tensor
    intrinsics.
    """
    return _ffi_api.GetAutoTensorizeMappingInfo(sch, block, desc_func)  # type: ignore


def has_block(sch: Schedule, block_name: str) -> bool:
    """Query if the given block name exists in the module associated with the provided schedule.

    Parameters
    ----------
    sch : Schedule
        The schedule
    block_name : str
        The name of the block to query

    Returns
    -------
    yes/no: bool
        True if the given block exists in the schedule.
    """
    return _ffi_api.HasBlock(sch, block_name)  # type: ignore


def is_output_block(sch: Schedule, block: SBlockRV) -> bool:
    """Check whether the given block is an output block

    Parameters
    ----------
    sch : Schedule
        The schedule object of the block
    block : SBlockRV
        The blockRV to be checked

    Returns
    -------
    yes/no : bool
        True if the given block is an output block

    """
    return _ffi_api.IsOutputBlock(sch, block)  # type: ignore
