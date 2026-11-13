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
"""Iteration-axis metadata for schedulable tensor computations."""

import tvm_ffi

from tvm_tirx import ir
from tvm_tirx.ir import ExprWithOp, OpaqueExpr, Var
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from . import _ffi_api


@tvm_ffi.register_object("s_tir.IterVar")
class IterVar(OpaqueExpr, ExprWithOp):
    """Represent iteration variable.

    IterVar represents axis iterations in the computation.
    It may appear as a primitive-valued expression in TE construction. CreateFunction
    lowers value occurrences to the underlying variable, while retaining block axis
    metadata. Use ``.var`` for primitive analyses outside TE lowering.

    Parameters
    ----------
    dom : Range
        The domain of the iteration.

    var : Union[Var, str]
        The internal variable that is used for iteration.

    iter_type : int
        The iteration type.

    thread_tag : str
        The thread type tag.

    loc : Location, optional
        The location of this expression in the source code.

    See Also
    --------
    te.thread_axis: Create thread axis IterVar.
    te.reduce_axis: Create reduce axis IterVar.
    """

    DataPar = 0
    ThreadIndex = 1
    CommReduce = 2
    Ordered = 3
    Opaque = 4
    Unrolled = 5
    Vectorized = 6
    Parallelized = 7
    Tensorized = 8

    dom: ir.Range
    var: Var
    iter_type: int
    thread_tag: str

    def __init__(
        self,
        dom: ir.Range,
        var: Var | str,
        iter_type: int,
        thread_tag: str = "",
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        if dom is not None:
            if isinstance(dom, list | tuple):
                if len(dom) != 2:
                    raise TypeError("need to be list of ranges")
                dom = ir.Range(dom[0], dom[1])

            if not isinstance(dom, ir.Range):
                raise TypeError("dom need to be Range")

        name = var if var is not None else "iter"
        dtype = "int32" if dom is None else dom.extent.ty
        var = Var(name, ty=dtype, loc=loc) if not isinstance(var, Var) else var
        if dom is not None:
            assert var.ty == dom.extent.ty, "IterVar's Var type must match its domain's extent type"
        self.__init_handle_by_constructor__(
            _ffi_api.IterVar,
            dom,
            var,
            iter_type,
            thread_tag,
            loc,  # type: ignore
        )
