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
# ruff: noqa: E741
"""A rule for GEMV and DecodeGEMV."""

from functools import reduce

import tvm_tirx
from tvm import s_tir
from tvm_tirx import tirx
from tvm_tirx.ir import Range
from tvm_tirx.target import Target

from ..analysis import (
    SBlockInfo,
    get_max_shared_memory_per_block,
    is_broadcast_epilogue,
    is_gemv,
    normalize,
    normalize_function,
)
from ..base import auto_vectorize, get_bytes, get_extent, try_inline_contiguous_spatial
from .base import GPUScheduleRule


class GEMV(GPUScheduleRule):
    """A rule for GEMV and DecodeGEMV."""

    def apply(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
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
            ret = self.sch_outer_reduction(sch, target, block, vector_input_tensors, epilogue)
            if ret is None:
                return self.sch_outer_reduction_fallback(
                    sch, target, block, vector_input_tensors, epilogue
                )
            return sch

    def sch_inner_reduction(  # pylint: disable=too-many-arguments, invalid-name, unused-argument
        self,
        sch: s_tir.Schedule,
        target: Target,
        block: s_tir.schedule.SBlockRV,
        vector_input_tensors: list[tirx.Var],
        epilogue_info: SBlockInfo | None,
    ):
        """Schedule the inner reduction block."""

        def get_max_factor(n, factors):
            factors = sorted(factors, reverse=True)
            for factor in factors:
                if n % factor == 0:
                    return factor
            return 1

        def apply(
            sch: s_tir.Schedule,
            gemv,
            TAG_S,
            TAG_R,
            TS,
            TR,
            TILE_S,
            TILE_R,
            VEC_LOAD,
            VEC_C,
            LOAD_V_SHARED,
            LOAD_V_VEC,
            UNROLL,
            SUPPORT_WARP_SHUFFLE,
        ):
            # rfactor: reduce to tx * vec_c
            _, s, r, c = sch.get_loops(block=gemv)
            s = sch.fuse(_, s)
            r = sch.fuse(r, c)
            bx, ts, tile_s = sch.split(s, factors=[None, TS, TILE_S], preserve_unit_iters=True)
            r, tr, tile_r_vec_n, vec_c = sch.split(
                r, factors=[None, TR, TILE_R // VEC_C, VEC_C], preserve_unit_iters=True
            )
            sch.reorder(r, tile_r_vec_n, tr, vec_c)
            tr_vec_c = sch.fuse(tr, vec_c)
            rf = sch.rfactor(tr_vec_c, 0)

            # rfactor: reduce to tx
            bx, ts, tile_s, tr_vec_c = sch.get_loops(block=gemv)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            rf2 = sch.rfactor(tr, 0)

            # bind, vectorize compute
            bx, ts, tile_s, r, tile_r_vec_n, tr_vec_c = sch.get_loops(block=rf)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            sch.reorder(bx, ts, tr, r, tile_s, tile_r_vec_n, vec_c)
            sch.bind(bx, "blockIdx.x")
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            sch.vectorize(vec_c)

            shared_mem_usage = 0
            for tensor in vector_input_tensors:
                dtype_bytes = get_bytes(tensor.dtype)
                tensor_size = (
                    reduce(
                        lambda x, y: x * y,
                        tensor.shape,
                        tvm_tirx.ir.IntImm(tensor.shape[0].ty, 1),
                    )
                    * dtype_bytes
                )
                shared_mem_usage += tensor_size
                if not SUPPORT_WARP_SHUFFLE:
                    # When warp shuffle is not able, cross-thread allreduce
                    # is implemented with shared memory.
                    shared_mem_usage += TS * TR * dtype_bytes

            max_smem = get_max_shared_memory_per_block(target)
            LOAD_V_SHARED = (
                LOAD_V_SHARED
                and isinstance(shared_mem_usage, tvm_tirx.ir.IntImm)
                and shared_mem_usage.value <= max_smem
            )

            Aq_local = sch.cache_read(rf, read_tensor_index=1, storage_scope="local")
            num_cache_loops = len(sch.get_loops(block=Aq_local))
            sch.compute_at(Aq_local, r, preserve_unit_loops=True)
            cache_loops = sch.get_loops(block=Aq_local)[-num_cache_loops:]
            if len(cache_loops) == 1:
                fused_load = cache_loops[0]
            else:
                fused_load = sch.fuse(*cache_loops[-2:])
            aq_vec_len = max(1, VEC_LOAD // get_bytes(sch.get(Aq_local).reads[0].source.dtype))
            fused_load, vec_load = sch.split(
                fused_load, factors=[None, aq_vec_len], preserve_unit_iters=True
            )
            sch.vectorize(vec_load)

            # load vector into shared memory, shape should be the whole vector
            if LOAD_V_SHARED:
                if len(vector_input_tensors) != 1:
                    return None
                V_shared = sch.cache_read(rf, read_tensor_index=0, storage_scope="shared")
                sch.compute_at(V_shared, tr, preserve_unit_loops=True)
                l = sch.get_loops(block=V_shared)[-1]
                loop: tvm_tirx.ir.For = sch.get(l)
                if isinstance(
                    Range.get_extent_from_begin_end(loop.begin, loop.end), tvm_tirx.ir.IntImm
                ):
                    # avoid introducing predicates when vector length is too large
                    vec_length = max(
                        min(
                            get_max_factor(
                                (int)(Range.get_extent_from_begin_end(loop.begin, loop.end)),
                                [TS * TR * 1, TS * TR * 2, TS * TR * 4, TS * TR * 8],
                            )
                            // TS
                            // TR,
                            LOAD_V_VEC,
                        ),
                        1,
                    )
                else:
                    vec_length = LOAD_V_VEC
                if TAG_R == "threadIdx.x":
                    _, ty, tx, vec = sch.split(
                        l, factors=[None, TS, TR, vec_length], preserve_unit_iters=True
                    )
                else:
                    _, ty, tx, vec = sch.split(
                        l, factors=[None, TR, TS, vec_length], preserve_unit_iters=True
                    )
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                sch.vectorize(vec)

            # reduce tile_s * tr * vec to tile_s * tr
            sch.reverse_compute_at(rf2, loop=bx, preserve_unit_loops=True)
            tr, vec_c, *ts_tile_s = sch.get_loops(block=rf2)[1:]
            ts_tile_s = sch.fuse(*ts_tile_s)
            ts_o, ts_i, tile_s = sch.split(
                ts_tile_s, factors=[None, TS, TILE_S], preserve_unit_iters=True
            )
            tile_s, vec_s = sch.split(
                tile_s,
                factors=[None, get_max_factor(TILE_S, [1, 2, 4, 8])],
                preserve_unit_iters=True,
            )
            assert (
                Range.get_extent_from_begin_end(sch.get(ts_o).begin, sch.get(ts_o).end).value == 1
            )
            ts = sch.fuse(ts_o, ts_i)
            sch.reorder(ts, tr, tile_s, vec_s, vec_c)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            sch.vectorize(vec_s)

            # reduce tile_s * tr to tile_s
            sch.reverse_compute_at(gemv, loop=bx, preserve_unit_loops=True)
            tr, *ts_tile_s = sch.get_loops(block=gemv)[1:]
            ts_tile_s = sch.fuse(*ts_tile_s)
            ts_o, ts_i, tile_s = sch.split(
                ts_tile_s, factors=[None, TS, TILE_S], preserve_unit_iters=True
            )
            assert (
                Range.get_extent_from_begin_end(sch.get(ts_o).begin, sch.get(ts_o).end).value == 1
            )
            ts = sch.fuse(ts_o, ts_i)
            sch.reorder(tile_s, ts, tr)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)

            sch.decompose_reduction(rf, loop=sch.get_loops(block=rf)[3])
            sch.decompose_reduction(rf2, loop=sch.get_loops(block=rf2)[-1])

            sch.set_scope(rf, tensor_index=0, storage_scope="local")
            sch.set_scope(rf2, tensor_index=0, storage_scope="local")

            unroll_factor = UNROLL

            sch.annotate(
                block_or_loop=sch.get_loops(rf)[3],
                ann_key="tirx.auto_unroll_max_step",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf)[3], ann_key="tirx.unroll_explicit", ann_val=1
            )

            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[3],
                ann_key="tirx.auto_unroll_max_step",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[3], ann_key="tirx.unroll_explicit", ann_val=1
            )

            if LOAD_V_SHARED:
                sch.annotate(
                    block_or_loop=sch.get_loops(V_shared)[-4],
                    ann_key="tirx.unroll_explicit",
                    ann_val=unroll_factor,
                )

            # Schedule epilogue
            if epilogue_info is not None:
                epilogue = epilogue_info.block_rv
                if is_broadcast_epilogue(sch, block, epilogue):
                    sch.reverse_compute_at(epilogue, bx)
                    sch.set_scope(block, 0, "shared")
                    _, _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                    _, tx = sch.split(sch.fuse(*s), factors=[None, TR])
                    sch.bind(tx, "threadIdx.x")
                else:
                    sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
                    ts_tile_s = sch.fuse(*sch.get_loops(epilogue)[1:])
                    ts_tile_s = sch.get_loops(epilogue)[-1]
                    ts_o, ts_i, tile_s = sch.split(
                        ts_tile_s, factors=[None, TS, TILE_S], preserve_unit_iters=True
                    )
                    assert (
                        Range.get_extent_from_begin_end(
                            sch.get(ts_o).begin, sch.get(ts_o).end
                        ).value
                        == 1
                    )
                    ts = sch.fuse(ts_o, ts_i)
                    sch.bind(ts, TAG_S)
                    sch.set_scope(block, 0, "local")
            # pylint: enable=invalid-name
            return sch

        # Specify the `len_tx` and `len_ty` according to the loop extent
        batch, s, r, c = sch.get_loops(block=block)
        len_batch, len_s, len_r, len_c = (
            get_extent(sch, batch),
            get_extent(sch, s),
            get_extent(sch, r),
            get_extent(sch, c),
        )
        len_S = len_batch * len_s
        len_R = len_r * len_c

        TAG_S, TAG_R = "threadIdx.y", "threadIdx.x"
        SUPPORT_WARP_SHUFFLE = False
        VEC_LOAD = 1
        if target.kind.name == "cuda":
            VEC_C = 4
            LOAD_V_SHARED = True
            LOAD_V_VEC = 8
            VEC_LOAD = 4
            UNROLL = 256
            SUPPORT_WARP_SHUFFLE = True
            if isinstance(len_S, int):
                TS, TR = 16, 32
            else:
                TS, TR = 1, 64
        elif target.kind.name == "metal":
            # Note that the following tile size is tuned on M2 Ultra for 7B
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            VEC_C = 1
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 256
            SUPPORT_WARP_SHUFFLE = True
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 4, 16
                else:
                    TS, TR = 2, 64
            else:
                TS, TR = 1, 64
        elif target.kind.name == "rocm":
            VEC_C = 4
            # TODO: set LOAD_V_SHARED = False for now
            # rocm might have some issues when load/store of shared do not belong to same data type
            # and only works for certain vector lens, our commonly useful vector lens are in 4
            LOAD_V_SHARED = False
            LOAD_V_VEC = 8
            UNROLL = 256
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 1, 128
                else:
                    TS, TR = 8, 64
            else:
                TS, TR = 1, 64
        elif target.kind.name == "opencl" and (
            ("android" in str(target.host)) or ("adreno" in str(target.attrs))
        ):
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            VEC_C = 8
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 8
            TS, TR = 2, 32
        elif target.kind.name == "vulkan":
            VEC_C = 4
            LOAD_V_SHARED = True
            LOAD_V_VEC = 4
            UNROLL = 256
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 4, 32
                else:
                    TS, TR = 16, 32
            else:
                TS, TR = 1, 64
        elif target.kind.name == "opencl" and "mali" in str(target.attrs):
            VEC_C = 8
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 64
            TS, TR = 1, 64
        elif target.kind.name == "webgpu":
            VEC_C = 1
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 64
            if isinstance(len_S, int):
                # Note that the following tile size is tuned on Apple M4 Max for q4f16_1
                TS, TR = 8, 16
            else:
                # symbolic S: TS > 1 writes past the last row when S % TS != 0
                TS, TR = 1, 64
        else:
            VEC_C = 1
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 64
            TS, TR = 1, 64

        while TS * TR > int(target.attrs["max_num_threads"]):
            if TS > 1:
                TS //= 2
            else:
                TR //= 2

        TILE_S, TILE_R = (
            1,
            (
                len_c
                if len_c > 1
                else max(get_max_factor(len_r, [TR * 1, TR * 2, TR * 4, TR * 8]) // TR, 1)
            ),
        )
        VEC_C = min(get_max_factor(TILE_R, [1, 2, 4, 8]), VEC_C)

        return apply(
            sch,
            gemv=block,
            TAG_S=TAG_S,
            TAG_R=TAG_R,
            TS=TS,
            TR=TR,
            TILE_S=TILE_S,
            TILE_R=TILE_R,
            VEC_LOAD=VEC_LOAD,
            VEC_C=VEC_C,
            LOAD_V_SHARED=LOAD_V_SHARED,
            LOAD_V_VEC=LOAD_V_VEC,
            UNROLL=UNROLL,
            SUPPORT_WARP_SHUFFLE=SUPPORT_WARP_SHUFFLE,
        )

    def sch_outer_reduction(  # pylint: disable=too-many-arguments, invalid-name, unused-argument
        self,
        sch: s_tir.Schedule,
        target: Target,
        block: s_tir.schedule.SBlockRV,
        vector_input_tensors: list[tirx.Var],
        epilogue_info: SBlockInfo | None,
    ):
        """Schedule the outer reduction block."""

        def get_max_factor(n, factors):
            factors = sorted(factors, reverse=True)
            for factor in factors:
                if n % factor == 0:
                    return factor
            return 1

        def apply(
            sch: s_tir.Schedule,
            gemv,
            TAG_S,
            TAG_R,
            TS,
            TR,
            SCALE_PACK,
            DEC_PACK,
            VEC_LOAD,
            VEC_C,
            LOAD_V_SHARED,
            LOAD_V_VEC,
            UNROLL,
            LOAD_V_TILE,
        ):
            # rfactor: reduce to tx * vec_c
            batch, s, r, c = sch.get_loops(block=gemv)
            s = sch.fuse(batch, s)
            r = sch.fuse(r, c)
            bx, ts = sch.split(s, factors=[None, TS], preserve_unit_iters=True)
            r, v_tile, tr, tile_r, vec_c = sch.split(
                r, factors=[None, LOAD_V_TILE, TR, SCALE_PACK, DEC_PACK], preserve_unit_iters=True
            )
            sch.reorder(bx, ts, r, v_tile, tile_r, tr, vec_c)
            tr_vec_c = sch.fuse(tr, vec_c)
            rf = sch.rfactor(tr_vec_c, 0)

            # rfactor: reduce to tx
            bx, ts, tr_vec_c = sch.get_loops(block=gemv)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            rf2 = sch.rfactor(tr, 0)

            # bind, vectorize compute
            bx, ts, r, v_tile, tile_r, tr_vec_c = sch.get_loops(block=rf)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, DEC_PACK])
            sch.reorder(bx, ts, tr, r, v_tile, tile_r, vec_c)
            # sch.bind(batch, "blockIdx.z")
            sch.bind(bx, "blockIdx.x")
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            auto_vectorize(sch, vec_c, VEC_C)

            # decompose independent scale read to outer loop
            block_rf_stmt = sch.get(rf)
            if len(block_rf_stmt.reads) >= 3:
                As_local = sch.cache_read(rf, read_tensor_index=2, storage_scope="local")
                sch.compute_at(As_local, v_tile, preserve_unit_loops=True)
                # *tile_thr, vec_s = sch.get_loops(block=As_local)
                # sch.vectorize(vec_s)

            Aq_local = sch.cache_read(rf, read_tensor_index=1, storage_scope="local")
            sch.compute_at(Aq_local, tile_r, preserve_unit_loops=True)
            # *tile_thr, vec_s = sch.get_loops(block=Aq_local)
            # sch.vectorize(vec_s)

            if LOAD_V_SHARED:
                V_shared = sch.cache_read(rf, read_tensor_index=0, storage_scope="shared")
                sch.compute_at(V_shared, r, preserve_unit_loops=True)
                l = sch.get_loops(block=V_shared)[-1]
                _, v_tile, ts, tr, vec = sch.split(
                    l, factors=[None, LOAD_V_TILE, TS, TR, LOAD_V_VEC], preserve_unit_iters=True
                )
                sch.bind(tr, TAG_R)
                sch.bind(ts, TAG_S)
                auto_vectorize(sch, vec, LOAD_V_VEC)

            # reduce tile_s * tr * vec to tile_s * tr
            sch.reverse_compute_at(rf2, loop=bx, preserve_unit_loops=True)
            tr, vec_c, ts = sch.get_loops(block=rf2)[1:]
            sch.reorder(ts, tr, vec_c)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)

            # reduce tile_s * tr to tile_s
            sch.reverse_compute_at(gemv, loop=bx, preserve_unit_loops=True)
            tr, ts = sch.get_loops(block=gemv)[1:]
            sch.reorder(ts, tr)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)

            sch.decompose_reduction(rf, loop=sch.get_loops(block=rf)[2])
            sch.decompose_reduction(rf2, loop=sch.get_loops(block=rf2)[-1])

            sch.set_scope(rf, tensor_index=0, storage_scope="local")
            sch.set_scope(rf2, tensor_index=0, storage_scope="local")

            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[3],
                ann_key="tirx.auto_unroll_max_step",
                ann_val=UNROLL,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[3], ann_key="tirx.unroll_explicit", ann_val=1
            )

            # Schedule epilogue
            if epilogue_info is not None:
                epilogue = epilogue_info.block_rv
                if is_broadcast_epilogue(sch, block, epilogue):
                    sch.reverse_compute_at(epilogue, bx)
                    sch.set_scope(block, 0, "shared")
                    _, _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                    _, ts = sch.split(sch.fuse(*s), factors=[None, TS])
                    sch.bind(ts, TAG_S)
                else:
                    sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
                    ts_tile_s = sch.fuse(*sch.get_loops(epilogue)[1:])
                    ts_tile_s = sch.get_loops(epilogue)[-1]
                    ts, _ = sch.split(ts_tile_s, factors=[TS, None], preserve_unit_iters=True)
                    sch.bind(ts, TAG_S)
                    sch.set_scope(block, 0, "local")
            return sch

        # Specify the `len_tx` and `len_ty` according to the loop extent
        batch, s, r, c = sch.get_loops(block=block)
        _, len_s, len_r, len_c = (
            get_extent(sch, batch),
            get_extent(sch, s),
            get_extent(sch, r),
            get_extent(sch, c),
        )

        DEC_PACK = 8
        SCALE_PACK = 4

        if target.kind.name == "opencl" and (
            ("android" in str(target.host)) or ("adreno" in str(target.attrs))
        ):
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            VEC_C = 8
            UNROLL = 8
            TS, TR = 64, 4
            LOAD_V_SHARED = False
            LOAD_V_VEC = 4
            LOAD_V_TILE = 8
        elif target.kind.name == "metal":
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            VEC_C = 4
            UNROLL = 8
            TS, TR = 128, 4
            LOAD_V_SHARED = False
            LOAD_V_VEC = 4
            LOAD_V_TILE = 4
        else:
            return None

        if LOAD_V_SHARED is False:
            LOAD_V_TILE = 1

        if not isinstance(len_r, int) or len_r < LOAD_V_TILE * TR * SCALE_PACK * DEC_PACK:
            return None

        if not isinstance(len_s, int):
            TS, TR = 256, 1
            LOAD_V_SHARED = True

        if isinstance(len_s, int) and len_s > 96000:
            return None

        _, TILE_R = (
            1,
            (
                len_c
                if len_c > 1
                else max(get_max_factor(len_r, [TR * 1, TR * 2, TR * 4, TR * 8]) // TR, 1)
            ),
        )
        LOAD_V_VEC = min(get_max_factor(TILE_R, [1, 2, 4, 8]), LOAD_V_VEC)
        VEC_LOAD = 1

        return apply(
            sch,
            gemv=block,
            TAG_S=TAG_S,
            TAG_R=TAG_R,
            TS=TS,
            TR=TR,
            SCALE_PACK=SCALE_PACK,
            DEC_PACK=DEC_PACK,
            VEC_LOAD=VEC_LOAD,
            VEC_C=VEC_C,
            LOAD_V_SHARED=LOAD_V_SHARED,
            LOAD_V_VEC=LOAD_V_VEC,
            UNROLL=UNROLL,
            LOAD_V_TILE=LOAD_V_TILE,
        )

    def sch_outer_reduction_fallback(  # pylint: disable=too-many-arguments, invalid-name, unused-argument
        self,
        sch: s_tir.Schedule,
        target: Target,
        block: s_tir.schedule.SBlockRV,
        vector_input_tensors: list[tirx.Var],
        epilogue_info: SBlockInfo | None,
    ):
        """Schedule the outer reduction block."""
        # NOTE: Only Android is supported so far
        if not (
            target.kind.name == "opencl"
            and (("android" in str(target.host)) or ("adreno" in str(target.attrs)))
        ):
            return None
        batch, s, r, c = sch.get_loops(block)
        len_s = get_extent(sch, s)

        # The config is designed for Adreno
        LOAD_V_SHARED = 1
        tx_len = 128
        vec_len = (4 if len_s > 4096 else 2) if isinstance(len_s, int) else 1
        inner_r = 4

        bx, tx, vec = sch.split(s, factors=[None, tx_len, vec_len])
        r0, r1 = sch.split(r, factors=[None, inner_r])
        sch.bind(batch, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        sch.reorder(bx, tx, r0, r1, c, vec)

        sch.annotate(tx, ann_key="tirx.auto_unroll_max_step", ann_val=8)
        sch.annotate(tx, ann_key="tirx.unroll_explicit", ann_val=1)

        if LOAD_V_SHARED:
            V_shared = sch.cache_read(block, vector_input_tensors[0], storage_scope="shared")
            sch.compute_at(V_shared, bx, preserve_unit_loops=True)
            l = sch.get_loops(block=V_shared)[-1]
            _, tx, vec_r = sch.split(l, factors=[None, tx_len, 8], preserve_unit_iters=True)
            sch.bind(tx, "threadIdx.x")
            sch.vectorize(vec_r)

        sch.vectorize(vec)

        # Schedule epilogue
        if epilogue_info is not None:
            sch.reverse_compute_at(epilogue_info.block_rv, bx, preserve_unit_loops=True)
            ts_tile_s = sch.get_loops(epilogue_info.block_rv)[-1]
            ts, vec = sch.split(ts_tile_s, factors=[tx_len, vec_len], preserve_unit_iters=True)
            sch.bind(ts, "threadIdx.x")
            sch.vectorize(vec)
            sch.set_scope(block, 0, "local")

        sch.decompose_reduction(block, r0)

        return sch
