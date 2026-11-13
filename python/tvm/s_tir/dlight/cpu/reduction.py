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
"""CPU reduction rule for operators including softmax, layer norm, RMS norm, etc."""

from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.codegen import llvm
from tvm_tirx.target import Target

from ..analysis import normalize_function
from ..base import get_extent
from .base import CPUScheduleRule


def _get_num_leading_s(dom_kind: str) -> int:
    """Count leading spatial ('S') axes in a dom_kind string."""
    return len(dom_kind) - len(dom_kind.lstrip("S"))


class Reduction(CPUScheduleRule):
    """CPU reduction rule for softmax, layer norm, RMS norm, and similar operators.

    Targets patterns with a mix of reduction (SR) and injective (SS) blocks,
    where all blocks share the same leading spatial axes.
    Example: softmax = maxelem(SR) -> exp(SS) -> expsum(SR) -> norm(SS).

    Schedule strategy:
      1. Parallelize leading spatial axes (batch dimension).
      2. Move all blocks under the spatial loop via compute_at.
      3. Vectorize injective blocks (exp, delta, norm) on their inner axis.
      4. Split reduction inner axis to VLEN-sized chunks and annotate for
         LLVM unrolling, preventing harmful full-unroll by the backend.

    Note: vectorized reduction via rfactor is not used here because TVM's
    rfactor primitive requires the reduction block to be the first child of
    its enclosing loop, which is incompatible with compute_at when multiple
    blocks share the same spatial loop. A follow-up using RVV reduction
    intrinsics (vfredmax/vfredusum) via tensorize can address this.
    """

    def apply(  # pylint: disable=too-many-locals,too-many-return-statements,too-many-branches
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None

        sch = s_tir.Schedule(func)
        block_infos = normalize_function(sch)
        if block_infos is None or len(block_infos) < 2:
            return None

        # Must have at least one reduction block and last block must be injective.
        if not any(not bi.is_injective() for bi in block_infos):
            return None
        if not block_infos[-1].is_injective():
            return None

        # Every block must start with at least one spatial axis, and all blocks
        # must agree on the minimum number of leading spatial axes.
        num_leading_s = None
        for bi in block_infos:
            dk = bi.dom_kind()
            if not dk or dk[0] != "S":
                return None
            n = _get_num_leading_s(dk)
            num_leading_s = n if num_leading_s is None else min(num_leading_s, n)
        if not num_leading_s:
            return None

        # Infer dtype from the last block's write tensor.
        last_block_stmt = sch.get(block_infos[-1].block_rv)
        dtype_bits = last_block_stmt.writes[0].source.dtype.bits if last_block_stmt.writes else 32

        # Determine vector lanes from target VLEN.
        vlen_bits = llvm.get_vector_width(target)
        if vlen_bits <= 0:
            vlen_bits = 128
        vec_lanes = max(vlen_bits // dtype_bits, 2)

        # --- Phase 1: Parallelize spatial on the last block ---
        last_block = block_infos[-1]
        loops = sch.get_loops(last_block.block_rv)
        if num_leading_s > 1:
            spatial = sch.fuse(*loops[:num_leading_s])
        else:
            spatial = loops[0]
        sch.parallel(spatial)

        # --- Phase 2: Vectorize the last (injective) block ---
        self._vectorize_inner(sch, last_block.block_rv, vec_lanes)

        # --- Phase 3: compute_at all preceding blocks under spatial ---
        for block_info in reversed(block_infos[:-1]):
            sch.compute_at(block_info.block_rv, spatial, preserve_unit_loops=True)

        # --- Phase 4: Vectorize injective, split+unroll reduction blocks ---
        for block_info in block_infos[:-1]:
            if block_info.is_injective():
                self._vectorize_inner(sch, block_info.block_rv, vec_lanes)
            else:
                self._unroll_reduction_inner(sch, block_info.block_rv, vec_lanes)

        return sch

    @staticmethod
    def _vectorize_inner(sch, block_rv, vec_lanes):
        """Split the innermost loop to vec_lanes and vectorize."""
        block_loops = sch.get_loops(block_rv)
        if len(block_loops) <= 1:
            return
        inner = block_loops[-1]
        extent = get_extent(sch, inner)
        if isinstance(extent, int):
            if extent > vec_lanes:
                _, vec_loop = sch.split(inner, factors=[None, vec_lanes])
                sch.vectorize(vec_loop)
            elif extent >= 2:
                sch.vectorize(inner)
        else:
            _, vec_loop = sch.split(inner, factors=[None, vec_lanes])
            sch.vectorize(vec_loop)

    @staticmethod
    def _unroll_reduction_inner(sch, block_rv, vec_lanes):
        """Split the reduction inner loop and annotate for unrolling."""
        block_loops = sch.get_loops(block_rv)
        if len(block_loops) <= 1:
            return
        inner = block_loops[-1]
        extent = get_extent(sch, inner)
        if isinstance(extent, int) and extent <= vec_lanes:
            return
        _, inner_loop = sch.split(inner, factors=[None, vec_lanes])
        sch.annotate(inner_loop, ann_key="tirx.auto_unroll_max_step", ann_val=vec_lanes)
        sch.annotate(inner_loop, ann_key="tirx.unroll_explicit", ann_val=1)
