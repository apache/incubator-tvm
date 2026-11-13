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
# pylint: disable=missing-docstring, invalid-name
"""A Conv2d schedule rule for Adreno GPU operators."""

from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.target import Target

from .. import analysis
from .base import AdrenoScheduleRule
from .utils import schedule_default, schedule_inline_blocks


class Conv2d(AdrenoScheduleRule):
    """The schedule rule for convolution computation"""

    @staticmethod
    def schedule_conv2d(sch: s_tir.Schedule, blk: s_tir.schedule.SBlockRV):
        n, oc, oh, ow, ob, ic, kh, kw = sch.get_loops(blk)

        bz, vz, tz = sch.split(oc, [None, 8, 1], preserve_unit_iters=True)
        by, vy, ty = sch.split(oh, [None, 1, 16], preserve_unit_iters=True)
        bx, vx, tx = sch.split(ow, [None, 1, 16], preserve_unit_iters=True)

        bz = sch.fuse(n, bz, preserve_unit_iters=True)
        sch.reorder(bz, by, bx, vz, vy, vx, tz, ty, tx, ob)
        sch.bind(bz, "blockIdx.z")
        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(vz, "vthread.z")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(tz, "threadIdx.z")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        rblk = sch.cache_read(blk, 0, "local")
        ico, icb = sch.split(ic, [None, 4], preserve_unit_iters=True)
        sch.reorder(ico, kh, kw, icb, ob)

        sch.compute_at(rblk, kw, preserve_unit_loops=True)
        sch.vectorize(sch.get_loops(rblk)[-1])
        wblk = sch.cache_write(blk, 0, "local")
        sch.reverse_compute_at(wblk, tx, preserve_unit_loops=True)
        sch.vectorize(sch.get_loops(wblk)[-1])
        init_blk = sch.decompose_reduction(blk, tx)
        sch.vectorize(sch.get_loops(init_blk)[-1])

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tirx.Function | s_tir.Schedule,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule | None:
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
        blocks = sch.get_child_blocks(root_block)
        reduction_blocks = list(
            filter(lambda block: analysis.get_sblock_info(sch, block).is_reduction(), blocks)
        )
        remaining_blocks = [blk for blk in blocks if blk not in reduction_blocks]

        def is_convolution(blk):
            block_info = analysis.get_sblock_info(sch, blk)
            return "conv2d_NCHWc" in block_info.name

        if len(reduction_blocks) != 1 or not is_convolution(reduction_blocks[0]):
            return None

        conv_blk = reduction_blocks[0]
        Conv2d.schedule_conv2d(sch, conv_blk)
        remaining_blocks = schedule_inline_blocks(sch, remaining_blocks)
        schedule_default(sch, remaining_blocks, target=target)

        return sch
