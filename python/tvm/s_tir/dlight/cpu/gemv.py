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
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""A rule for GEMV and DecodeGEMV."""

from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.target import Target

from ..analysis import SBlockInfo, normalize_function
from ..analysis.gemv import is_gemv, normalize
from ..base import get_extent, try_inline_contiguous_spatial
from .base import CPUScheduleRule


class GEMV(CPUScheduleRule):
    """A rule for GEMV and DecodeGEMV."""

    def apply(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements, no-else-return
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None
        sch = s_tir.Schedule(func)
        block_infos = normalize_function(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None:
            return None
        if len(block_infos) == 1:
            epilogue = None
        elif len(block_infos) == 2:
            epilogue = block_infos[1]
            if not epilogue.is_injective():
                return None
        else:
            return None

        block_info = block_infos[0]
        if len(block_info.iters) not in [2, 3]:
            # either [B, S, R] = [B, S, R] * [B, R]
            # or [S, R] = [S, R] * [R]
            return None
        block = block_info.block_rv
        vector_input_tensors = is_gemv(sch, block_info)
        if vector_input_tensors is None:
            return None

        # Step 1. Normalize the block, merge spatial and reduction iters
        is_inner_reduction = normalize(sch, block_info)

        # Step 2. Do the scheduling
        if is_inner_reduction is None:
            return None
        elif is_inner_reduction:
            return self.sch_inner_reduction(sch, target, block, vector_input_tensors, epilogue)
        else:
            # sch_outer reduction
            return None

    def sch_inner_reduction(  # pylint: disable=too-many-arguments, too-many-positional-arguments, invalid-name, unused-argument
        self,
        sch: s_tir.Schedule,
        target: Target,
        block: s_tir.schedule.SBlockRV,
        vector_input_tensors: list[tirx.Var],
        epilogue_info: SBlockInfo | None,
    ):
        """Schedule the inner reduction block."""

        def apply(  # pylint: disable=unused-variable, too-many-locals
            sch: s_tir.Schedule,
            gemv,
            vector_width: int = 8,
            parallel_threads: int = 8,
            unroll_factor: int = 256,
        ):
            batch, s, r, c = sch.get_loops(block)
            len_batch, len_s, len_r, len_c = (
                get_extent(sch, batch),
                get_extent(sch, s),
                get_extent(sch, r),
                get_extent(sch, c),
            )
            len_S = len_batch * len_s
            len_R = len_r * len_c

            if isinstance(len_S, int) and isinstance(len_R, int):
                if len_S > len_R:
                    tile_s, tile_r = 128, 64  # Larger tiling for s-axis when len_S is larger
                else:
                    tile_s, tile_r = 64, 128  # Larger tiling for r-axis when len_R is larger
            else:
                tile_s, tile_r = 64, 64  # Default tile sizes for unknown extents

            tile_c = min(vector_width, len_c)  # Ensure c-axis tiling aligns with SIMD vector width

            # Apply loop tiling (improves cache locality)
            s_outer, s_inner = sch.split(s, factors=[None, tile_s])
            r_outer, r_inner = sch.split(r, factors=[None, tile_r])
            c_outer, c_inner = sch.split(c, factors=[None, tile_c])

            # Apply vectorization (SIMD optimization)
            sch.vectorize(s_inner)  # Vectorize computation along c-axis for AVX/NEON

            # Enable parallel execution
            sch.parallel(s_outer)  # Parallelize along the s-axis (major computation loop)

            # Apply loop unrolling for better CPU performance
            sch.annotate(r_outer, "tirx.auto_unroll_max_step", unroll_factor)
            sch.annotate(r_outer, "tirx.unroll_explicit", 1)
            return sch

        return apply(
            sch,
            gemv=block,
        )
