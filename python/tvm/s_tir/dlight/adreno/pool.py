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
# ruff: noqa: F841
"""Pool schedule rule for Adreno operators."""

from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.ir import Range
from tvm_tirx.target import Target

from .. import analysis
from .base import AdrenoScheduleRule


# pylint: disable=invalid-name, unused-variable
class Pool2D(AdrenoScheduleRule):
    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule:
        sch = s_tir.Schedule(func)
        root = sch.get_sblock(name="root", func_name="main")

        blocks = sch.get_child_blocks(root)
        blocks_names = [sch.get(blk).name_hint for blk in blocks]

        if "adaptive_pool_sum" not in blocks_names and "pool_max" not in blocks_names:
            return None

        def schedule_pad(blk: s_tir.schedule.SBlockRV):
            lps, veclp = sch.get_loops(blk)[:-1], sch.get_loops(blk)[-1]
            sch.vectorize(veclp)
            b = sch.fuse(*lps)
            tx_extent = min(
                int(Range.get_extent_from_begin_end(sch.get(b).begin, sch.get(b).end))
                & ~int(Range.get_extent_from_begin_end(sch.get(b).begin, sch.get(b).end) - 1),
                256,
            )
            bx, tx = sch.split(b, [None, tx_extent])
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

        def schedule_max_pool(blk: s_tir.schedule.SBlockRV):
            block_info = analysis.get_sblock_info(sch, blk)
            iters_kind = "".join([_iter.kind for _iter in block_info.iters])
            if iters_kind != "SSSSSRR":
                return None

            lps = sch.get_loops(blk)
            block_lps, vec_lp, red_lps = lps[:4], lps[4], lps[5:]
            write_blk = sch.cache_write(blk, 0, "local")
            sch.reverse_compute_at(write_blk, vec_lp)
            b = sch.fuse(*block_lps)
            tx_extent = min(
                int(Range.get_extent_from_begin_end(sch.get(b).begin, sch.get(b).end))
                & ~int(Range.get_extent_from_begin_end(sch.get(b).begin, sch.get(b).end) - 1),
                256,
            )
            bx, tx = sch.split(b, [None, tx_extent])
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")
            sch.vectorize(vec_lp)

            return True

        passed_reduction = False
        for blk in blocks:
            if sch.get(blk).name_hint == "pad_temp":
                schedule_pad(blk)
            elif (
                sch.get(blk).name_hint == "adaptive_pool_sum"
                or sch.get(blk).name_hint == "pool_max"
            ):
                ok = schedule_max_pool(blk)
                if not ok:
                    return None
                passed_reduction = True
            else:
                try:
                    if passed_reduction:
                        sch.reverse_compute_inline(blk)
                    else:
                        sch.compute_inline(blk)
                except Exception:  # pylint: disable=broad-except
                    pass
        return sch
