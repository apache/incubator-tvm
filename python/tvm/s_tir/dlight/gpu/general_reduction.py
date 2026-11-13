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
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

import tvm_ffi

from tvm import s_tir
from tvm_tirx import sym, tirx
from tvm_tirx.ir import Range
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.target import Target

from ..analysis import SBlockInfo, get_root_block, normalize_function
from ..base import get_extent, try_inline_contiguous_spatial
from .base import GPUScheduleRule

_SCOPE_GLOBAL = MemoryScope("global")

# Largest producer reduction extent for which this serial path is applied.
#
# This is a POLICY CONSTANT, not the output of a model. The bound below is the only
# number in this file that is not derived from the IR, so it is written out with what
# is and is not known about it.
#
# What is measured (issue apache/tvm#20401, RTX A6000, whole-graph VM latency):
#   * R = 4   : default 251.70 us, serial 10.30 us  -> serial wins
#   * R = 289 : default 403.01 us, serial  32.90 us -> serial wins
# Both are single measurements of a real crossover region. Neither locates where the
# serial path stops winning.
#
# Why there is no crossover estimate here. A two-term model (serial cost ~ R rounds;
# default cost ~ R/len_tx rounds plus one block-wide barrier, barrier cost B) has one
# free parameter. Fitting B to the R = 289 point reproduces that point by construction
# and predicts nothing else; with one measurement the model is unfalsifiable, so it is
# not a basis for choosing this bound. Two further terms are omitted and are both
# R-dependent in the direction that favours the default path:
#
#   * Memory coalescing. At a given load instruction the default path's neighbouring
#     lanes read neighbouring addresses; the serial path's lanes stride by the window
#     width. Measured per instruction from the lowered index maps (see
#     `evidence-sector-efficiency.py`): at R = 289 the default path uses 70.8% of the
#     sectors it fetches and the serial path 12.5%. This is the most likely reason a
#     deep reduction eventually favours the default path, and no barrier-only term can
#     express it. Note it is not yet binding at R = 289: serial wins there by 12x
#     despite fetching 5.7x the bytes.
#   * Wave quantization. The serial path always launches ceil(num_outputs / len_tx)
#     blocks (576 here) regardless of R, against ~504-672 resident block slots on an
#     84-SM A6000. That is a tail effect of at most one partial wave, and it does not
#     vary with R, so it cannot by itself move the crossover in R.
#
# How the value was chosen. It covers every measured point with one doubling of margin
# and stays well inside the region where the barrier/idle-thread cost is known to
# dominate. It is deliberately conservative: if the true crossover is below 512 this
# leaves some serial wins unclaimed, but it never applies the serial path to a case
# measured to be worse. The opposite error -- a bound above the true crossover -- would
# reintroduce the regression. A reduction-size sweep is required to replace this with a
# measured crossover, and the regression test at the boundary pins the current choice.
_SERIAL_REDUCTION_MAX_EXTENT = 512


def _static_extent(block_info: SBlockInfo, kind: str) -> int | None:
    """Product of the extents of the iterators of `kind` ("S" or "R"), or None if dynamic."""
    extent = 1
    for iter_info in block_info.iters:
        if iter_info.kind != kind:
            continue
        if not isinstance(iter_info.dom, int):
            return None
        extent *= iter_info.dom
    return extent


class GeneralReduction(GPUScheduleRule):
    """General Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None

        if target.kind.name == "cuda":
            len_tx = 256
            unroll_depth = 256
        elif target.kind.name == "opencl":
            len_tx = 256
            unroll_depth = 64
        else:
            len_tx = 64
            unroll_depth = 64

        sch = s_tir.Schedule(func)
        block_infos = normalize_function(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        dom_kind = block_infos[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        # Align the number of block iters of the last block.
        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter < len(dom_kind):
            # If the last block is a scalar value, there is nothing left to
            # tile/parallelise, and  `iters` is an empty tuple.
            # Add a unit thread loop so the final write happens inside a valid
            # GPU thread environment.
            if num_last_block_iter == 0:
                # Allocation planning can move a reduction tensor inside its
                # producer kernel when all surrounding loops are trivial. Give
                # tensors private to that kernel an explicit local scope, while
                # preserving global scope for tensors accessed by another block.
                blocks = [sch.get(info.block_rv) for info in block_infos]
                alloc_tensors = list(sch.get(get_root_block(sch)).alloc_tensors)
                analyzer = sym.Analyzer()
                for block_index, (info, block) in enumerate(zip(block_infos[:-1], blocks[:-1])):
                    loops = sch.get_loops(info.block_rv)
                    if not all(
                        analyzer.can_prove_equal(
                            Range.get_extent_from_begin_end(sch.get(loop).begin, sch.get(loop).end),
                            1,
                        )
                        for loop in loops
                    ):
                        continue

                    other_block_tensors = [
                        region.source
                        for other_index, other_block in enumerate(blocks)
                        if other_index != block_index
                        for region in (*other_block.reads, *other_block.writes)
                    ]
                    for tensor_index, write in enumerate(block.writes):
                        tensor = write.source
                        is_allocated = any(tensor.same_as(other) for other in alloc_tensors)
                        is_cross_block = any(tensor.same_as(other) for other in other_block_tensors)
                        if tensor.scope() == _SCOPE_GLOBAL and is_allocated and not is_cross_block:
                            sch.set_scope(block_infos[block_index].block_rv, tensor_index, "local")

                # Put every block (both the running reductions and the final
                # scalar write) inside a trivial GPU thread. The very first block
                # gets a `blockIdx.x` wrapper so that kernels still have a unique
                # block scope.
                for i, info in enumerate(block_infos):
                    loop_rv = sch.add_unit_loop(info.block_rv)
                    if i == 0:
                        sch.bind(loop_rv, "blockIdx.x")
                    else:
                        sch.bind(loop_rv, "threadIdx.x")

                return sch

            def f_layout_mapping(*iters):
                analyzer = sym.Analyzer()
                # Try to match the iters of last block to the iters of the first block.
                # For matched positions, use the iter from the input `iters`.
                # For unmatched positions, use a new iter which is constant 0.
                num_matched = 0
                target_layout_iters = []
                for block_iter in block_infos[0].iters:
                    if num_matched < len(iters) and analyzer.can_prove_equal(
                        block_iter.dom, block_infos[-1].iters[num_matched].dom
                    ):
                        target_layout_iters.append(iters[num_matched])
                        num_matched += 1
                    else:
                        target_layout_iters.append(tirx.const(0, iters[0].ty))

                # If all the iters of the last block can match, return the new layout.
                if num_matched == len(iters):
                    return target_layout_iters
                # Otherwise, fallback to appending zeros in the beginning.
                return [tirx.const(0, iters[0].ty)] * (len(dom_kind) - num_last_block_iter) + list(
                    iters
                )

            index_map = tirx.IndexMap.from_func(f_layout_mapping, ndim=num_last_block_iter)
            sch.transform_block_layout(block_infos[-1].block_rv, index_map)

        try:
            # TODO: fix num_leading_s = 0 case
            assert num_trailing_r > 0
            for block in block_infos[1:-1]:
                assert block.dom_kind() == dom_kind
            assert block_infos[-1].is_injective()
            assert len(block_infos[-1].dom_kind()) <= len(dom_kind)
        except AssertionError:
            return None

        if "R" not in block_infos[-1].dom_kind():
            # The final block is a spatial block.
            # It is possible that the loop order of the last block is not the same as
            # previous blocks.
            # Thus we reorder spatial loops to align with reduction loops for followup schedule.
            # We first collect all the tensors written by reduction blocks,
            # then in the final block, any index of those tensors are spatial.
            reduced_tensors = []
            for block_info in block_infos[:-1]:
                for tensor_write in sch.get(block_info.block_rv).writes:
                    reduced_tensors.append(tensor_write.source)

            spatial_block = sch.get(block_infos[-1].block_rv)
            spatial_loops = set()
            block_var_to_loop_var = {}
            loops = sch.get_loops(block_infos[-1].block_rv)
            for block_iter, loop_rv in zip(spatial_block.iter_vars, loops):
                block_var_to_loop_var[block_iter.var] = sch.get(loop_rv).var

            def _visit_expr(e: tirx.Var):
                if e in block_var_to_loop_var:
                    spatial_loops.add(block_var_to_loop_var[e])

            for tensor_read in spatial_block.reads:
                tensor = tensor_read.source
                if tensor in reduced_tensors:
                    for read_range in tensor_read.region:
                        tvm_ffi.structural_walk(
                            read_range.begin, (tirx.Var, _visit_expr), order="post"
                        )
                        tvm_ffi.structural_walk(
                            read_range.extent,
                            (tirx.Var, _visit_expr),
                            order="post",
                        )

            s_loops = []
            other_loops = []
            for loop_rv in loops:
                loop = sch.get(loop_rv)
                if (
                    loop.var in spatial_loops
                    or Range.get_extent_from_begin_end(loop.begin, loop.end) == 1
                ):
                    s_loops.append(loop_rv)
                else:
                    other_loops.append(loop_rv)
            sch.reorder(*s_loops, *other_loops)

        small_reduction = self._schedule_small_reduction(sch, block_infos, len_tx, unroll_depth)
        if small_reduction is not None:
            return small_reduction

        loops = sch.get_loops(block_infos[-1].block_rv)
        bx = sch.fuse(*loops[:num_leading_s])
        r_loop, tx = sch.split(loops[-1], [None, len_tx])
        sch.reorder(tx, r_loop)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.annotate(r_loop, ann_key="tirx.auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(r_loop, ann_key="tirx.unroll_explicit", ann_val=1)

        for block in reversed(block_infos[:-1]):
            block = block.block_rv
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, tensor_index=i, storage_scope="shared")
            sch.compute_at(block, bx, preserve_unit_loops=True)
            r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
            r_loop, tx = sch.split(r_loop, [None, len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(tx, "threadIdx.x")
            sch.annotate(r_loop, ann_key="tirx.auto_unroll_max_step", ann_val=unroll_depth)
            sch.annotate(r_loop, ann_key="tirx.unroll_explicit", ann_val=1)

        # TODO: It's just a workaround to avoid unroll spatial loops, because of the bug of
        # the pass lower-thread-allreduce. We should fix it in the future.
        # sch.annotate(bx, ann_key="tirx.auto_unroll_max_step", ann_val=unroll_depth)
        # sch.annotate(bx, ann_key="tirx.unroll_explicit", ann_val=1)
        return sch

    def _schedule_small_reduction(  # pylint: disable=too-many-locals
        self,
        sch: s_tir.Schedule,
        block_infos: list[SBlockInfo],
        len_tx: int,
        unroll_depth: int,
    ) -> s_tir.Schedule | None:
        """Fold a shallow reduction serially, one output per thread.

        The default path below sizes the thread dimension from the loop extents it finds
        and never compares it against the depth of the reduction. For an output-heavy,
        reduction-light op that is a loss: `split(loop_of_extent_4, [None, 256])` starts
        256 threads to do 4 elements of work, the block dimension ends up sized by the
        output count, and every block pays a shared-memory round trip and a block-wide
        barrier to combine a handful of values. For a reduction shallow enough that the
        per-output serial work is cheaper than that barrier, it is better folded serially
        inside one thread, keeping the output that thread already owns in a register.

        How shallow is "shallow enough" is the policy constant
        `_SERIAL_REDUCTION_MAX_EXTENT`, which is set from the two measurements in
        apache/tvm#20401 and documented there. It is a conservative bound on a crossover
        that has not been located, not a modelled optimum.

        Following Kennedy and McKinley (LCPC 1993), distributing a fused loop is only
        profitable while it does not give up parallelism or duplicate work, so this path is
        taken only when all of the following hold. Otherwise it returns None without
        touching `sch`, and the default path runs unchanged.

        * every producer's reduction is static and at most `_SERIAL_REDUCTION_MAX_EXTENT`;
        * the outputs are static and fill at least one block of `len_tx` threads;
        * every producer has exactly as many spatial elements as the final block, and each
          later block reads a producer value at a single point. A producer broadcast over a
          reduced axis (softmax, layer norm statistics) is therefore left alone, because one
          thread per output would recompute it once per output element.
        """
        producers = block_infos[:-1]
        for info in producers:
            reduction_extent = _static_extent(info, "R")
            if reduction_extent is None or reduction_extent > _SERIAL_REDUCTION_MAX_EXTENT:
                return None

        spatial_loops = sch.get_loops(block_infos[-1].block_rv)
        num_outputs = 1
        for loop_rv in spatial_loops:
            extent = get_extent(sch, loop_rv)
            if not isinstance(extent, int):
                return None
            num_outputs *= extent
        if num_outputs < len_tx:
            return None

        analyzer = sym.Analyzer()
        produced = []
        for info in block_infos:
            block = sch.get(info.block_rv)
            for read in block.reads:
                if any(read.source.same_as(t) for t in produced) and not all(
                    analyzer.can_prove_equal(rng.extent, 1) for rng in read.region
                ):
                    return None
            if info is not block_infos[-1]:
                if _static_extent(info, "S") != num_outputs:
                    return None
                produced.extend(write.source for write in block.writes)

        fused = sch.fuse(*spatial_loops)
        bx, tx = sch.split(fused, [None, len_tx])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        for info in reversed(producers):
            block = info.block_rv
            # Each thread owns its element, so keep it in a register, not shared memory.
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, tensor_index=i, storage_scope="local")
            # Compute at the thread loop, not the block loop: computing at `bx` would
            # make every thread redo the whole block's reductions.
            sch.compute_at(block, tx, preserve_unit_loops=True)
            inner_loops = sch.get_loops(block)
            sch.annotate(
                inner_loops[-1],
                ann_key="tirx.auto_unroll_max_step",
                ann_val=unroll_depth,
            )
            sch.annotate(inner_loops[-1], ann_key="tirx.unroll_explicit", ann_val=1)
        return sch
