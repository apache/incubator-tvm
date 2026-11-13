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
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

import tvm_tirx
from tvm import s_tir
from tvm.s_tir import Schedule
from tvm.s_tir.schedule import SBlockRV
from tvm_tirx import sym, tirx
from tvm_tirx.ir import TensorLoad
from tvm_tirx.target import Target

from ..analysis import detect_dominant_read, normalize_function
from ..base import try_inline_contiguous_spatial
from .base import GPUScheduleRule


class Transpose(GPUScheduleRule):
    """Schedule rule for transpose"""

    def is_transpose(self, sch: Schedule, block_rv: SBlockRV):
        block = sch.get(block_rv)
        if len(block.body) == 1 and isinstance(block.body[0], tvm_tirx.ir.TensorStore):
            store = block.body[0]
            rhs = store.value
            if isinstance(rhs, TensorLoad):
                lhs_indices = store.indices
                rhs_indices = rhs.indices
                if list(lhs_indices) != list(rhs_indices) and set(lhs_indices) == set(rhs_indices):
                    return True
        return False

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tirx.Function,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        # pylint: disable=invalid-name
        if not isinstance(func, tirx.Function) or not self.is_target_available(target):
            return None
        if target.kind.name == "cuda":
            len_tx = 16
            len_ty = 8
            unroll_depth = 256
        elif target.kind.name == "opencl":
            len_tx = 16
            len_ty = 8
            unroll_depth = 64
        else:
            len_tx = 8
            len_ty = 4
            unroll_depth = 64
        len_vec = 4

        sch = s_tir.Schedule(func)
        blocks = normalize_function(sch)
        transpose_block_idx = -1
        for idx, block in reversed(list(enumerate(blocks))):
            if self.is_transpose(sch, block.block_rv):
                transpose_block_idx = idx
                break
            if not block.is_injective():
                return None
        if transpose_block_idx == -1:
            return None
        transpose_block = blocks[transpose_block_idx].block_rv

        prologue = None  # the optional decoding block
        if transpose_block_idx > 0:
            spatials = try_inline_contiguous_spatial(sch, blocks[: transpose_block_idx - 1])
            assert len(spatials) == 0
            prologue = blocks[transpose_block_idx - 1].block_rv

        loops = sch.get_loops(transpose_block)
        if len(loops) != 2:
            # transpose with more than 2 axes is not supported
            return None

        c_factor = 1
        if prologue is not None:
            block_stmt = sch.get(prologue)
            result = sym.normalize_to_iter_sum(
                detect_dominant_read(block_stmt),
                input_iters={i.var: i.dom for i in block_stmt.iter_vars},
            )
            if len(result.args) > 0:
                c_factor = int(result.args[0].lower_factor)

        i, j = loops
        i, vi = sch.split(i, factors=[None, c_factor], preserve_unit_iters=True)
        bi, ti = sch.split(i, factors=[None, len_ty], preserve_unit_iters=True)
        bj, tj = sch.split(j, factors=[None, len_tx], preserve_unit_iters=True)
        sch.reorder(bi, bj, ti, tj, vi)
        sch.bind(bi, "blockIdx.y")
        sch.bind(bj, "blockIdx.x")
        sch.bind(ti, "threadIdx.y")
        sch.bind(tj, "threadIdx.x")
        len_vec = min(len_vec, c_factor)
        _, vi = sch.split(vi, factors=[None, len_vec])
        if len_vec > 1:
            sch.vectorize(vi)

        cache_read = sch.cache_read(transpose_block, read_tensor_index=0, storage_scope="shared")
        sch.compute_at(cache_read, bj)
        loops = sch.get_loops(cache_read)[2:]
        fused = sch.fuse(*loops)
        _, ty, tx, v = sch.split(fused, factors=[None, len_ty, len_tx, c_factor])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.unroll(v)
        sch.storage_align(block=cache_read, tensor_index=0, axis=0, factor=32, offset=1)

        sch.annotate(bi, ann_key="tirx.auto_unroll_max_step", ann_val=unroll_depth)
        sch.annotate(bi, ann_key="tirx.unroll_explicit", ann_val=1)

        if prologue is not None:
            sch.compute_inline(prologue)
        return sch
