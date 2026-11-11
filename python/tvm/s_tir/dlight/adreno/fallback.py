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

# licensed to the apache software foundation (asf) under one
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
"""Dlight Adreno Fallback Schedules"""

from functools import cache

from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.target import Target

from .. import analysis
from .base import AdrenoScheduleRule


@cache
def _texture_scopes():
    return tuple(
        MemoryScope(name)
        for name in (
            "texture",
            "global.texture",
            "global.texture-1d",
            "global.texture-nchw",
            "global.texture-nhwc",
            "global.texture-weight",
        )
    )


def _assert_gpu_target(target: Target):
    if "gpu" not in target.keys:
        raise ValueError(f"Expect a GPU target, but got {target}")


def get_max_threads_per_block(target: Target) -> int:
    _assert_gpu_target(target)
    max_threads_per_block = None
    for name in ["max_threads_per_block", "max_num_threads"]:
        if max_threads_per_block is None:
            max_threads_per_block = target.attrs.get(name, None)
    if max_threads_per_block is None:
        max_threads_per_block = 64
    return int(max_threads_per_block)


# pylint: disable=invalid-name,missing-function-docstring,unused-variable,unused-import
class Fallback(AdrenoScheduleRule):
    """Texture Based Fallback Schedule(s) for Adreno"""

    @staticmethod
    def schedule_inline_blocks(
        sch: s_tir.Schedule, blocks: list[s_tir.schedule.SBlockRV]
    ) -> list[s_tir.schedule.SBlockRV]:
        """
        Auto Inlines Injective and Element-wise Operations while trying to omit data pad blocks...
        """

        if blocks is None:
            root_blk = analysis.get_root_block(sch)
            blocks = sch.get_child_blocks(root_blk)

        remaining_blocks = []
        for blk in blocks:
            block_info = analysis.get_sblock_info(sch, blk)
            if block_info.is_injective() and not block_info.is_data_pad(sch):
                if len(sch.get_consumers(blk)) == 1:
                    try:
                        sch.compute_inline(blk)
                    except Exception:  # pylint: disable=broad-exception-caught
                        remaining_blocks.append(blk)
                elif len(sch.get_producers(blk)) == 1:
                    inlined_once = False
                    try:
                        # Would cause an issue inlining to producer with multiple consumers
                        while (
                            len(sch.get_producers(blk)) == 1
                            and len(sch.get_consumers(sch.get_producers(blk)[0])) == 1
                        ):
                            sch.reverse_compute_inline(blk)
                            inlined_once = True
                    except Exception:  # pylint: disable=broad-exception-caught
                        break
                    if not inlined_once:
                        remaining_blocks.append(blk)
                else:
                    remaining_blocks.append(blk)
            else:
                remaining_blocks.append(blk)
        return remaining_blocks

    @staticmethod
    def schedule_default(sch: s_tir.Schedule, blk: s_tir.schedule.SBlockRV, *, target=None):
        block_info = analysis.get_sblock_info(sch, blk)

        s_loops, r_loops, o_loops = [], [], []
        v_loop = block_info.write_tensors(sch)[0].assoc_lps[-1]

        for iter_info in block_info.iters:
            if sch.get(iter_info.loop_rv) == sch.get(v_loop):
                continue
            {"S": s_loops, "R": r_loops, "O": o_loops}.get(iter_info.kind).append(iter_info.loop_rv)

        iter_vars = analysis.collect_block_iter_vars_used_in_access_region(
            sch.get(blk), block_info.write_tensors(sch)[0].tensor_region.region
        )
        o_outer = [lp for lp in o_loops if sch.get(lp).var in iter_vars]
        o_inner = [lp for lp in o_loops if sch.get(lp).var not in iter_vars]

        # Can't change loop order for opaque loops
        if o_loops != o_outer + o_inner:
            return

        o_outer.append(v_loop)
        sch.reorder(*s_loops, *o_outer, *r_loops, *o_inner)

        assert s_loops
        tgt = target

        b = sch.fuse(*s_loops)
        tx_extent = get_max_threads_per_block(tgt) if tgt is not None else 256
        bx, tx = sch.split(b, [None, tx_extent])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        if len(r_loops) > 1:
            lp = [*s_loops, *o_outer][-1]
            init_block = sch.decompose_reduction(blk, lp)
            wblk = sch.cache_write(blk, 0, "local")
            sch.compute_at(wblk, lp)
            if v_loop:
                sch.vectorize(sch.get_loops(init_block)[-1])
                sch.vectorize(sch.get_loops(wblk)[-1])
        elif v_loop is not None:
            sch.vectorize(v_loop)

    @staticmethod
    def schedule_fallback(sch, *, target=None):
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        schedule_blocks = [
            blk
            for blk in blocks
            if analysis.get_sblock_info(sch, blk).is_reduction()
            or analysis.get_sblock_info(sch, blk).is_data_pad(sch)
        ]
        remaining_blocks = [blk for blk in blocks if blk not in schedule_blocks]

        for blk in schedule_blocks:
            Fallback.schedule_default(sch, blk, target=target)
        remaining_blocks = Fallback.schedule_inline_blocks(sch, remaining_blocks)
        # TODO: Analyze unscheduled blocks to schedule instead of relying on remaining
        for blk in remaining_blocks:
            Fallback.schedule_default(sch, blk, target=target)

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        # pylint: disable=invalid-name

        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None

        sch = s_tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if any(len(sch.get_child_blocks(block)) != 0 for block in blocks):
            return None

        block_infos = [analysis.get_sblock_info(sch, block) for block in blocks]
        if not any(
            block.write_tensors(sch)[0].get_scope() in _texture_scopes() for block in block_infos
        ):
            return None

        Fallback.schedule_fallback(sch, target=target)
        return sch
