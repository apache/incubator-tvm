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

"""The Graph Matching Context Manager for Dataflow Pattern Language."""

import tvm_ffi

from ..expr import DataflowBlock, Var
from . import _ffi as ffi
from .pattern import DFPattern


class PatternContext(tvm_ffi.Object):
    """A context object for doing graph (topogical) pattern matching."""

    def __init__(self, incremental=False):
        """
        Initialize the PatternContext

        Parameters
        ----------
        incremental : bool, optional
            perform incremental matching based on the recent context, by default False
        """
        self.__init_handle_by_constructor__(ffi.PatternContext, incremental)  # type: ignore

    def __enter__(self):
        """Enter the context"""
        ffi.enter_context(self)  # type: ignore
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context"""
        ffi.exit_context(self)  # type: ignore

    @staticmethod
    def current() -> "PatternContext":
        """
        Get the current context

        Returns
        -------
        PatternContext
            The current context
        """
        return ffi.current_context()  # type: ignore

    def match_dfb(
        self,
        dfb: DataflowBlock,
    ) -> dict[DFPattern, Var]:
        """
        Match a DataflowBlock via a graph of DFPattern and corresponding constraints

        Parameters
        ----------
        dfb : DataflowBlock
            The DataflowBlock to match

        Returns
        -------
        Dict[DFPattern, Var]
            The mapping from DFPattern to matched expression
        """
        return ffi.match_dfb(self, dfb)  # type: ignore
