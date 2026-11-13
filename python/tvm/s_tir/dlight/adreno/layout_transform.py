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
# ruff: noqa: F841

# Licensed to the apache software foundation (asf) under one
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

# pylint: disable=invalid-name, unused-variable

"Schedules for Texture Based Layout Transforms"

from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.ir import Range
from tvm_tirx.target import Target

from .. import analysis
from .base import AdrenoScheduleRule


class LayoutTransform(AdrenoScheduleRule):
    """Texture based Layout Transform Dlight Schedule for Adreno"""

    def __init__(self, use_op_name=True):
        self.use_op_name = use_op_name

    # TODO: Try using Coalesced Writes...
    def apply(  # pylint: disable=too-many-locals
        self,
        func: tirx.Function | s_tir.Schedule,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        # pylint: disable=invalid-name
        if not (isinstance(func, tirx.Function | s_tir.Schedule)) or not self.is_target_available(
            target
        ):
            return None

        if isinstance(func, tirx.Function):
            sch = s_tir.Schedule(func)
            sch.work_on("main")
        elif isinstance(func, s_tir.Schedule):
            sch = func

        root_block = analysis.get_root_block(sch, sch.func_working_on)

        if len(sch.get_child_blocks(root_block)) != 1:
            return None

        blk = sch.get_child_blocks(root_block)[0]
        block_info = analysis.get_sblock_info(sch, blk)
        if not (
            (self.use_op_name and block_info.name == "te_layout_transform")
            or (not self.use_op_name and block_info.is_layout_transform(sch))
        ):
            return None

        read_tensor, write_tensor = (
            block_info.read_tensors(sch)[0],
            block_info.write_tensors(sch)[0],
        )
        lps = block_info.get_loops()
        lpv_read, lpv_write = (
            read_tensor.assoc_lps[-1],
            write_tensor.assoc_lps[-1],
        )

        if lpv_read is None or lpv_write is None:
            return None

        vlen_read, vlen_write = read_tensor.get_vecsize(), write_tensor.get_vecsize()
        local_cache = sch.get(lpv_read) != sch.get(lpv_write) or vlen_read != vlen_write
        block_loops = [
            lp
            for lp in lps
            if sch.get(lp) != sch.get(lpv_read) and sch.get(lp) != sch.get(lpv_write)
        ]
        vec_loops = (
            [lpv_read, lpv_write] if sch.get(lpv_read) != sch.get(lpv_write) else (lpv_read,)
        )
        sch.reorder(*block_loops, *vec_loops)
        if local_cache:
            if sch.get(lpv_read) != sch.get(lpv_write):
                blp_read, vlp_read = sch.split(
                    lpv_read, [None, vlen_read], preserve_unit_iters=True
                )
                blp_write, vlp_write = sch.split(
                    lpv_write, [None, vlen_write], preserve_unit_iters=True
                )
                sch.reorder(blp_read, blp_write, vlp_read, vlp_write)
                block_loops += [blp_read, blp_write]
                rblk = sch.cache_read(blk, 0, "local")
                sch.compute_at(rblk, block_loops[-1], preserve_unit_loops=True)
                sch.vectorize(sch.get_loops(rblk)[-1])
                sch.vectorize(vlp_write)
            else:
                if vlen_read > vlen_write:
                    read_lp, vec_lp = sch.split(blk, [None, vlen_write], preserve_unit_iters=True)
                    rblk = sch.cache_read(blk, 0, "local")
                    sch.compute_at(rblk, read_lp, preserve_unit_loops=True)
                    sch.vectorize(sch.get_loops(rblk)[-1])
                    sch.vectorize(vec_lp)
                else:
                    rblk = sch.cache_read(blk, 0, "local")
                    sch.compute_at(rblk, block_loops[-1], preserve_unit_loops=True)
                    _, vread_lp = sch.split(
                        sch.get_loops(rblk)[-1], vlen_read, preserve_unit_iters=True
                    )
                    sch.vectorize(vread_lp)
                    sch.vectorize(vlp_write)
        else:
            blp, vlp = sch.split(lpv_read, [None, vlen_read], preserve_unit_iters=True)
            block_loops += [blp]
            sch.vectorize(vlp)

        b = sch.fuse(*block_loops)
        tx_extent = min(Range.get_extent_from_begin_end(sch.get(b).begin, sch.get(b).end), 256)
        candidates = [1, 2, 4, 8, 16, 32]
        bx, tx = sch.split(b, [None, 256], preserve_unit_iters=True)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        return sch
