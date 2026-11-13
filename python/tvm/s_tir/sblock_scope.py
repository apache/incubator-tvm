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
"""Definition of two pillar data structure for TensorIR scheduling: StmtSRef, SBlockScope."""

from enum import IntEnum
from typing import Optional

import tvm_ffi
from tvm_ffi import register_object

from tvm.s_tir import SBlock
from tvm_tirx.ir import For

from . import _ffi_api


@register_object("s_tir.StmtSRef")
class StmtSRef(tvm_ffi.Object):
    """An object that refers to schedulable elements in the TensorIR, aka "sref".

    Glossary
    - SBlock sref: An StmtSref that points to a TensorIR block.
    - Loop sref: An StmtSRef that points to a TensorIR for loop.
    - Parent sref: The parent sref of an sref is the block/loop sref that points to its closest
    schedulable statement of its ancestors on the TensorIR AST.
    - Root sref: Sref to the root block. Every sref has exactly one parent sref
    except for root sref.
    - Sref tree: The parent-children-relationship of srefs that forms a tree,
    uniquely determined by the TensorIR AST.
    """

    seq_index: int

    @property
    def stmt(self) -> SBlock | For | None:
        """The block/for stmt the object refers to"""
        return _ffi_api.StmtSRefStmt(self)  # type: ignore # pylint: disable=no-member

    @property
    def parent(self) -> Optional["StmtSRef"]:
        """The parent sref"""
        return _ffi_api.StmtSRefParent(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def inline_mark() -> "StmtSRef":
        """A special StmtSRef, which doesn't point to any stmt in the AST,
        only serving as a "mark" to hint compute-at to do the work of compute-inline"""
        return _ffi_api.StmtSRefInlineMark()  # type: ignore # pylint: disable=no-member

    @staticmethod
    def root_mark() -> "StmtSRef":
        """A special StmtSRef, which doesn't point to any stmt in the AST,
        only serving as a "mark" to hint compute-at to do nothing"""
        return _ffi_api.StmtSRefRootMark()  # type: ignore # pylint: disable=no-member


class DepKind(IntEnum):
    """Type of dependency.

    Attributes
    ----------
    RAW : int = 0
        Read-after-write dependency
    WAW : int = 1
        Write-after-write dependency
    WAR : int = 2
        Write-after-read dependency. Not supported in TensorIR for now.
    OPAQUE: int = 3
        Opaque dependency
    """

    RAW = 0
    WAW = 1
    WAR = 2
    OPAQUE = 3


@register_object("s_tir.Dependency")
class Dependency(tvm_ffi.Object):
    """A tuple (src, dst, kind) representing certain types of dependency.
    For example, (A, B, kRAW) means block B depends on block A, and the dependency kind is
    read-after-write, which means block B reads the result written by block A.

    Parameters
    ----------
    src : StmtSRef
        The source of the dependency relation
    dst : StmtSRef
        The destination of the dependency relation
    kind : DepKind
        The dependency kind
    """

    src: StmtSRef
    dst: StmtSRef
    kind: DepKind


@register_object("s_tir.SBlockScope")
class SBlockScope(tvm_ffi.Object):
    """An object corresponds to each block sref in the sref tree, which
    tracks the producer-consumer dependency between blocks.

    Glossary:

    - SBlock scope: A contiguous subtree of the sref tree, rooted at
      each SBlock sref, whose components are:

      - scope root: a SBlock sref
      - internal srefs: loop srefs
      - scope leaves: SBlock srefs

    - Child SBlock: The scope leaf SBlocks under the scope root or a specific internal sref
    """

    def get_deps_by_src(self, block: StmtSRef) -> list[Dependency]:
        """Get all dependencies whose `src` is the target`block`.

        Parameters
        ----------
        block: StmtSRef
            The queried block

        Returns
        -------
        blocks: List[Dependency]
            The dependencies
        """
        return _ffi_api.SBlockScopeGetDepsBySrc(self, block)  # type: ignore # pylint: disable=no-member

    def get_deps_by_dst(self, block: StmtSRef) -> list[Dependency]:
        """Get all dependencies whose `dst` is the target `block`.

        Parameters
        ----------
        block: StmtSRef
            The queried block

        Returns
        -------
        blocks: List[Dependency]
            The dependencies
        """
        return _ffi_api.SBlockScopeGetDepsByDst(self, block)  # type: ignore # pylint: disable=no-member
