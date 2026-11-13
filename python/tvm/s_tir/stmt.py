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
"""Schedulable TensorIR statement nodes."""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import tvm_ffi

from tvm.s_tir import IterVar
from tvm_tirx.ir import Expr, Location, Scriptable, SeqStmt, Stmt, TensorRegion, Var, const
from tvm_tirx.ir.location import UNKNOWN_LOC
from tvm_tirx.tirx import TensorType

from . import _ffi_api


@tvm_ffi.register_object("s_tir.MatchTensorRegion")
class MatchTensorRegion(tvm_ffi.Object, Scriptable):
    """MatchTensorRegion node.

    Parameters
    ----------
    tensor : Var
        The target tensor

    source : TensorRegion
        The region of source tensor
    """

    tensor: Var
    source: TensorRegion

    def __init__(self, tensor: Var, source: TensorRegion) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.MatchTensorRegion,
            tensor,
            source,  # type: ignore
        )


@tvm_ffi.register_object("s_tir.SBlock")
class SBlock(Stmt):
    """SBlock node.

    Parameters
    ----------
    iter_vars : List[IterVar]
        The block Variable.

    reads : List[TensorRegion]
        The read tensor regions of the block.

    writes: List[TensorRegion]
        The write tensor regions of the block.

    name_hint: str
        the name_hint of the block.

    body: Stmt | Sequence[Stmt]
        The body of the block.

    init: Stmt | Sequence[Stmt] | None
        The init block of the reduction block

    alloc_tensors: Optional[list[Var]]
        The tensor allocations

    match_tensors: Optional[List[MatchTensorRegion]]
        The subregion tensor match

    annotations: Optional[Mapping[str, tvm_ffi.Object]]
        Additional annotation hints.

    loc : Location, optional
        The location of this block in the source code.
    """

    iter_vars: list[IterVar]
    reads: list[TensorRegion]
    writes: list[TensorRegion]
    name_hint: str
    body: SeqStmt
    init: SeqStmt | None
    alloc_tensors: list[Var]
    match_tensors: list[MatchTensorRegion]
    annotations: Mapping[str, tvm_ffi.Object]
    loc: Location

    def __init__(
        self,
        iter_vars: list[IterVar],
        reads: list[TensorRegion],
        writes: list[TensorRegion],
        name_hint: str,
        body: Stmt | Sequence[Stmt],
        init: Stmt | Sequence[Stmt] | None = None,
        alloc_tensors: list[Var] | None = None,
        match_tensors: list[MatchTensorRegion] | None = None,
        annotations: Mapping[str, tvm_ffi.Object] | None = None,
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        if alloc_tensors is None:
            alloc_tensors = []
        if match_tensors is None:
            match_tensors = []
        if annotations is None:
            annotations = {}
        self.__init_handle_by_constructor__(
            _ffi_api.SBlock,  # type: ignore
            iter_vars,
            reads,
            writes,
            name_hint,
            body,
            init,
            alloc_tensors,
            match_tensors,
            annotations,
            loc,
        )  # type: ignore


@tvm_ffi.register_object("s_tir.SBlockRealize")
class SBlockRealize(Stmt):
    """SBlockRealize node.

    Parameters
    ----------
    iter_values : List[Expr]
        The binding values of the block var.

    predicate : Union[Expr, bool]
        The predicate of the block.

    block : SBlock
        The block to realize

    loc : Location, optional
        The location of this block_realize in the source code.
    """

    iter_values: list[Expr]
    predicate: Expr
    block: SBlock
    loc: Location

    def __init__(
        self,
        iter_values: list[Expr],
        predicate: Expr | bool,
        block: SBlock,
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        if isinstance(predicate, bool):
            predicate = const(predicate, "bool")
        self.__init_handle_by_constructor__(
            _ffi_api.SBlockRealize,  # type: ignore
            iter_values,
            predicate,
            block,
            loc,
        )  # type: ignore


class ShapeStridePattern(NamedTuple):
    """Shape, actual strides, and element offset of a regular memory layout."""

    shape: Sequence[Expr]
    strides: Sequence[Expr]
    elem_offset: Expr


def match_shape_stride_pattern(tensor_type: TensorType) -> ShapeStridePattern | None:
    """Match absent or shape-aligned memory-only TileLayouts, without symbolic proofs."""
    result = _ffi_api.MatchShapeStridePattern(tensor_type)
    return None if result is None else ShapeStridePattern(*result)
