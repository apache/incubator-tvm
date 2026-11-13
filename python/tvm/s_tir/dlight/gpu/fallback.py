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
# pylint: disable=missing-docstring
"""A fallback schedule rule for GPU operators."""

import tvm_ffi

import tvm_tirx
from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.ir import Range
from tvm_tirx.target import Target

from .. import base
from ..analysis import normalize_function
from ..base import try_inline
from .base import GPUScheduleRule


def _has_internal_thread_env(stmt: tvm_tirx.ir.Stmt) -> bool:
    """Check whether a statement already launches GPU threads internally,
    e.g. via `T.launch_thread` regions or nested
    thread-bound loops. Such blocks manage their own thread environment
    and must not be wrapped in an additional thread binding."""
    found = False

    def visit_region(node: tvm_tirx.ir.RegionStmt):
        nonlocal found
        if node.op.name == "tirx.launch_thread":
            found = True

    def visit_for(node: tvm_tirx.ir.For):
        nonlocal found
        if node.annotations.get("thread_binding") is not None:
            found = True

    tvm_ffi.structural_walk(
        stmt,
        [(tvm_tirx.ir.RegionStmt, visit_region), (tvm_tirx.ir.For, visit_for)],
        order="post",
    )
    return found


def _has_zero_extent_loop(stmt: tvm_tirx.ir.Stmt) -> bool:
    """Check whether a statement contains a statically empty loop."""
    found = False

    def visit_for(node: tvm_tirx.ir.For):
        nonlocal found
        if (
            isinstance(Range.get_extent_from_begin_end(node.begin, node.end), tvm_tirx.ir.IntImm)
            and Range.get_extent_from_begin_end(node.begin, node.end).value == 0
        ):
            found = True

    tvm_ffi.structural_walk(stmt, (tvm_tirx.ir.For, visit_for), order="post")
    return found


class Fallback(GPUScheduleRule):
    """
    A fallback schedule rule for all GPU operators. It will try to inline all the blocks first,
    and then apply a simple block/grid mapping to the spatial loops on top of the remaining blocks.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule:
        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None
        if _has_zero_extent_loop(func.body):
            return None
        max_threads_per_block = base.max_threads_per_block(target)

        sch = s_tir.Schedule(func)
        block_infos = normalize_function(sch)

        if block_infos is None:
            return None

        block_infos = try_inline(sch, block_infos)
        reduction_blocks: list[tuple[s_tir.schedule.SBlockRV, s_tir.schedule.LoopRV]] = []
        for block in block_infos:
            s_loops: list[s_tir.schedule.LoopRV] = []
            r_loops: list[s_tir.schedule.LoopRV] = []
            o_loops: list[s_tir.schedule.LoopRV] = []
            dom_kind = block.dom_kind()
            block = block.block_rv

            if any(
                sch.get(loop_rv).annotations.get("thread_binding") is not None
                for loop_rv in sch.get_loops(block)
            ):
                continue

            if len(sch.get_loops(block)) == 0 and _has_internal_thread_env(sch.get(block).body):
                # The block (e.g. an opaque sort kernel) launches its own
                # threads; binding an outer loop would conflict with them.
                continue

            for loop, iter_type in zip(sch.get_loops(block), dom_kind):
                {"S": s_loops, "R": r_loops, "O": o_loops}[iter_type].append(loop)

            if not s_loops:
                s_loops.append(sch.add_unit_loop(block))
            sch.reorder(*s_loops, *r_loops, *o_loops)
            bx, tx = sch.split(  # pylint: disable=invalid-name
                sch.fuse(*s_loops),
                factors=[None, max_threads_per_block],
            )
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

            if len(r_loops) > 0:
                reduction_blocks.append((block, r_loops[0]))

        for block, r_loop in reduction_blocks:
            sch.decompose_reduction(block, r_loop)

        return sch
