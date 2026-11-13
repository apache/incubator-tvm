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
# ruff: noqa: E741
"""A rule for low-batch GEMM / decode-GEMM using GEMV schedule."""

from functools import reduce
from typing import Literal

import tvm_ffi

import tvm_tirx
from tvm import s_tir
from tvm_tirx import sym, tirx
from tvm_tirx.ir import Range
from tvm_tirx.target import Target

from ..analysis import (
    SBlockInfo,
    collect_block_iter_vars_used_in_access_region,
    collect_vars_used_in_prim_expr,
    get_max_shared_memory_per_block,
    is_broadcast_epilogue,
    normalize_function,
)
from ..analysis.common_analysis import _tensor_region_offset
from ..base import auto_vectorize, get_bytes, get_extent, try_inline_contiguous_spatial
from .base import GPUScheduleRule


def _get_reduction_expr(block: s_tir.SBlock) -> tirx.Expr | None:
    # Detect and return `Y` in `X[...] = X[...] + Y`
    if len(block.body) != 1:
        return None
    tensor_store = block.body[0]
    if not isinstance(tensor_store, tvm_tirx.ir.TensorStore):
        return None
    if not isinstance(tensor_store.value, tirx.Add):
        return None
    if not tvm_ffi.structural_equal(
        tensor_store.value.a,
        tvm_tirx.ir.TensorLoad(tensor_store.dest, tensor_store.indices),
        map_free_vars=True,
    ):
        return None
    return tensor_store.value.b


def _has_pad_einsum_compatible_access(block: s_tir.SBlock) -> bool:
    """Check the point-access restriction required by ``Schedule.pad_einsum``."""
    return all(
        isinstance(dim.extent, tvm_tirx.ir.IntImm)
        and int(dim.extent) == 1
        and isinstance(dim.begin, tvm_tirx.ir.IntImm | tirx.Var)
        for region in [*block.reads, *block.writes]
        for dim in region.region
    )


def is_gemv(sch: s_tir.Schedule, block_info: SBlockInfo) -> list[tirx.Var] | None:
    """Check if the block is a low batch GEMM.

    Parameters
    ----------

    sch : s_tir.Schedule
        The schedule

    block_info : SBlockInfo
        The block info to be checked


    Returns
    -------
    ret : Optional[List[tirx.Var]]
        The vector-like tensors used in the low batch GEMM if it is a low batch GEMM,
        otherwise None.
    """
    block = block_info.block_rv
    block_stmt = sch.get(block)
    conditions = []
    conditions.append(block_info.is_reduction())
    conditions.append(len(block_stmt.reads) >= 2)
    conditions.append(len(block_stmt.writes) == 1)
    conditions.append(_get_reduction_expr(block_stmt) is not None)
    conditions.append(_has_pad_einsum_compatible_access(block_stmt))
    conditions.append(
        len(collect_block_iter_vars_used_in_access_region(block_stmt, block_stmt.writes[0].region))
        > 0
    )
    if not all(conditions):
        return None
    const_iter_vars = set(
        iter_var.var
        for iter_var in block_stmt.iter_vars
        if isinstance(iter_var.dom.extent, tvm_tirx.ir.IntImm)
    )
    if len(block_stmt.iter_vars) - len(const_iter_vars) != 1:
        return None
    symbolic_iter_var = next(
        iter_var
        for iter_var in block_stmt.iter_vars
        if not isinstance(iter_var.dom.extent, tvm_tirx.ir.IntImm)
    )
    if symbolic_iter_var.iter_type != s_tir.IterVar.DataPar:
        return None
    ret = [
        read.source
        for read in block_stmt.reads
        if len(
            collect_block_iter_vars_used_in_access_region(block_stmt, read.region) & const_iter_vars
        )
        < len(const_iter_vars)
        and len(
            collect_block_iter_vars_used_in_access_region(block_stmt, read.region) & const_iter_vars
        )
        > 0
    ]
    return ret if 0 < len(ret) < len(block_stmt.reads) else None


def detect_dominant_read(block: s_tir.SBlock, const_iter_vars: set[tirx.Var]) -> tirx.Expr:
    """Detect the dominant read indices in the block."""
    dominant_read = None
    num_read_iters = -1
    for tensor_region in block.reads:
        tir_vars = (
            collect_block_iter_vars_used_in_access_region(block, tensor_region.region)
            & const_iter_vars
        )
        if num_read_iters < len(tir_vars):
            num_read_iters = len(tir_vars)
            dominant_read = tensor_region
    assert dominant_read is not None
    return _tensor_region_offset(dominant_read)


def normalize(
    sch: s_tir.Schedule,
    block_info: SBlockInfo,
) -> bool | None:
    """Normalize the main block."""
    block_stmt: s_tir.SBlock = sch.get(block_info.block_rv)
    const_iter_vars = set(
        iter_var.var
        for iter_var in block_stmt.iter_vars
        if isinstance(iter_var.dom.extent, tvm_tirx.ir.IntImm)
    )
    dynamic_iter_vars = set(
        iter_var.var for iter_var in block_stmt.iter_vars if iter_var.var not in const_iter_vars
    )
    access = sym.normalize_to_iter_sum(
        detect_dominant_read(block_stmt, const_iter_vars),
        input_iters={i.var: i.dom for i in block_stmt.iter_vars},
    )
    tensors_use_vars = [
        collect_block_iter_vars_used_in_access_region(block_stmt, tensor.region)
        for tensor in block_stmt.writes
    ]
    tensors_use_vars.extend(
        [
            collect_block_iter_vars_used_in_access_region(block_stmt, tensor.region)
            for tensor in block_stmt.reads
        ]
    )
    if collect_vars_used_in_prim_expr(access.base) & set(
        iter_var.var for iter_var in block_stmt.iter_vars
    ):
        return None
    iter_to_info = {i.var: i for i in block_info.iters}
    batch_loops, s_loops, r_loops = [], [], []
    inner_axis = access.args[-1].source.source
    is_inner_reduction = iter_to_info[inner_axis].kind == "R"

    for split_expr in access.args:
        var = split_expr.source.source
        info = iter_to_info.get(var)
        loop = info.loop_rv
        is_reduction = info.kind == "R"
        # No C loops as we do not compute_inline weights into main block
        if is_reduction:
            r_loops.append(loop)
        elif all([var in tensor_vars for tensor_vars in tensors_use_vars]):
            batch_loops.append(loop)
        else:
            s_loops.append(loop)

    assert s_loops
    assert r_loops
    dynamic_loops = [iter_to_info[var].loop_rv for var in dynamic_iter_vars]
    assert len(dynamic_loops) == 1
    sch.reorder(*dynamic_loops, *s_loops, *r_loops)
    sch.fuse(*s_loops)
    sch.fuse(*r_loops)
    return is_inner_reduction


class LowBatchGEMV(GPUScheduleRule):
    """A rule for low batch GEMM / decode-GEMM."""

    def __init__(self, bucket=4):
        self.bucket = bucket

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
        if block_infos is None:
            return None
        reduction_block_infos = [
            block_info for block_info in block_infos if block_info.is_reduction()
        ]
        if len(reduction_block_infos) != 1:
            return None
        reduction_block_info = reduction_block_infos[0]
        vector_input_tensors = is_gemv(sch, reduction_block_info)
        if vector_input_tensors is None:
            return None
        batch_pad = self.bucket
        pad_value = [
            iter.dom if isinstance(iter.dom, int) else batch_pad
            for iter in reduction_block_info.iters
        ]
        sch.pad_einsum(reduction_block_info.block_rv, pad_value)
        block_infos = normalize_function(sch)
        dequantize_block = None
        pad_input_block = None
        for block_info in block_infos:
            if "dequantize" in block_info.name:
                dequantize_block = block_info.block_rv
            elif "pad" in block_info.name and len(sch.get_producers(block_info.block_rv)) == 0:
                pad_input_block = block_info.block_rv
        block_infos = [
            block_info
            for block_info in block_infos
            if "pad" not in block_info.name and "dequantize" not in block_info.name
        ]
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
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
            self.sch_inner_reduction(
                sch,
                target,
                block,
                dequantize_block,
                pad_input_block,
                vector_input_tensors,
                epilogue,
                batch_pad,
            )
            return sch
        elif self.bucket <= 4:
            self.sch_outer_reduction(
                sch,
                target,
                block,
                dequantize_block,
                pad_input_block,
                vector_input_tensors,
                epilogue,
                batch_pad,
            )
            return sch
        else:
            return None

    def sch_inner_reduction(  # pylint: disable=too-many-arguments, invalid-name, unused-argument
        self,
        sch: s_tir.Schedule,
        target: Target,
        block: s_tir.schedule.SBlockRV,
        dequantize_block: s_tir.schedule.SBlockRV | None,
        pad_input_block: s_tir.schedule.SBlockRV | None,
        vector_input_tensors: list[tirx.Var],
        epilogue_info: SBlockInfo | None,
        batch_pad: int,
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
        ):
            # rfactor: reduce to tx * vec_c

            _, s, r = sch.get_loops(block=gemv)
            bx, ts, tile_s = sch.split(s, factors=[None, TS, TILE_S], preserve_unit_iters=True)
            r, tr, tile_r_vec_n, vec_c = sch.split(
                r, factors=[None, TR, TILE_R // VEC_C, VEC_C], preserve_unit_iters=True
            )
            sch.reorder(r, tile_r_vec_n, tr, vec_c)
            tr_vec_c = sch.fuse(tr, vec_c)
            rf = sch.rfactor(tr_vec_c, 0)

            # rfactor: reduce to tx
            _, bx, ts, tile_s, tr_vec_c = sch.get_loops(block=gemv)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            rf2 = sch.rfactor(tr, 0)
            # bind, vectorize compute
            batch_loop, bx, ts, tile_s, r, tile_r_vec_n, tr_vec_c = sch.get_loops(block=rf)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            sch.reorder(bx, ts, tr, r, tile_s, tile_r_vec_n, vec_c)
            sch.bind(bx, "blockIdx.x")
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            sch.vectorize(vec_c)
            by, batch = sch.split(batch_loop, factors=[None, batch_pad])
            sch.bind(by, "blockIdx.y")
            sch.reorder(bx, ts, tr, r, batch)

            shared_mem_usage = 0
            for tensor in vector_input_tensors:
                tensor_size = reduce(
                    lambda x, y: x * y, tensor.shape, tvm_tirx.ir.IntImm(tensor.shape[0].ty, 1)
                ) * get_bytes(tensor.dtype)
                shared_mem_usage += tensor_size
            max_smem = get_max_shared_memory_per_block(target)
            LOAD_V_SHARED = (
                LOAD_V_SHARED
                and isinstance(shared_mem_usage, tvm_tirx.ir.IntImm)
                and shared_mem_usage.value <= max_smem
            )

            # vectorize load A
            # (TODO) this is now actually problematic since the number of loops is dependent on the
            # number of dimensions of A_q
            if dequantize_block is not None:
                sch.compute_at(dequantize_block, r, preserve_unit_loops=True)
                sch.set_scope(dequantize_block, 0, "local")

                s_local, r_local = sch.get_loops(block=dequantize_block)[-2:]
                s_local, vec_load = sch.split(
                    s_local, factors=[None, VEC_LOAD], preserve_unit_iters=True
                )
                sch.reorder(s_local, r_local, vec_load)  # either s_local or r_local should be 1
                sch.vectorize(vec_load)

            # load vector into shared memory, shape should be the whole vector
            if LOAD_V_SHARED:
                assert len(vector_input_tensors) == 1
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
            if pad_input_block is not None:
                sch.compute_inline(pad_input_block)

            # reduce tile_s * tr * vec to tile_s * tr
            sch.reverse_compute_at(rf2, loop=bx, preserve_unit_loops=True)
            tr, vec_c, batch_loop, *ts_tile_s = sch.get_loops(block=rf2)[2:]
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
            sch.reorder(ts, tr, tile_s, batch_loop, vec_s, vec_c)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            sch.vectorize(vec_s)

            # reduce tile_s * tr to tile_s
            sch.reverse_compute_at(gemv, loop=bx, preserve_unit_loops=True)

            tr, batch_loop, *ts_tile_s = sch.get_loops(block=gemv)[2:]
            ts_tile_s = sch.fuse(*ts_tile_s)
            ts_o, ts_i, tile_s = sch.split(
                ts_tile_s, factors=[None, TS, TILE_S], preserve_unit_iters=True
            )
            assert (
                Range.get_extent_from_begin_end(sch.get(ts_o).begin, sch.get(ts_o).end).value == 1
            )
            ts = sch.fuse(ts_o, ts_i)
            sch.reorder(tile_s, batch_loop, ts, tr)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)

            sch.decompose_reduction(rf, loop=sch.get_loops(block=rf)[4])
            sch.decompose_reduction(rf2, loop=sch.get_loops(block=rf2)[-1])

            sch.set_scope(rf, tensor_index=0, storage_scope="local")
            sch.set_scope(rf2, tensor_index=0, storage_scope="local")

            unroll_factor = UNROLL

            sch.annotate(
                block_or_loop=sch.get_loops(rf)[4],
                ann_key="tirx.auto_unroll_max_step",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf)[4], ann_key="tirx.unroll_explicit", ann_val=1
            )

            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[4],
                ann_key="tirx.auto_unroll_max_step",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[4], ann_key="tirx.unroll_explicit", ann_val=1
            )

            if LOAD_V_SHARED:
                sch.annotate(
                    block_or_loop=sch.get_loops(V_shared)[-4],
                    ann_key="tirx.unroll_explicit",
                    ann_val=unroll_factor,
                )

            epilogue = sch.get_consumers(gemv)
            # Schedule epilogue
            if epilogue:
                epilogue = epilogue[0]
                if is_broadcast_epilogue(sch, block, epilogue):
                    sch.reverse_compute_at(epilogue, bx)
                    sch.set_scope(block, 0, "shared")
                    _, _, _, *s = sch.get_loops(epilogue)  # pylint: disable=invalid-name
                    _, tx = sch.split(sch.fuse(*s), factors=[None, TS])
                    sch.bind(tx, TAG_S)
                else:
                    sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
                    ts_tile_s = sch.fuse(*sch.get_loops(epilogue)[3:])
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

            return sch

        # Specify the `len_tx` and `len_ty` according to the loop extent
        _, s, r = sch.get_loops(block=block)
        len_s, len_r = get_extent(sch, s), get_extent(sch, r)

        TAG_S, TAG_R = "threadIdx.y", "threadIdx.x"
        if target.kind.name == "cuda":
            VEC_C = 4
            LOAD_V_SHARED = True
            LOAD_V_VEC = 8
            UNROLL = 256
            if isinstance(len_s, int):
                if len_s > len_r:
                    TS, TR = 4, 64
                else:
                    TS, TR = 16, 32
        elif target.kind.name == "metal":
            VEC_C = 4
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 8
            if isinstance(len_s, int):
                if len_s > len_r:
                    TS, TR = 8, 32
                else:
                    TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
                    TS, TR = 8, 32
        elif target.kind.name == "rocm":
            VEC_C = 4
            LOAD_V_SHARED = True
            LOAD_V_VEC = 8
            UNROLL = 256
            if isinstance(len_s, int):
                if len_s > len_r:
                    TS, TR = 1, 128
                else:
                    TS, TR = 8, 64
        elif target.kind.name == "opencl" and "android" in str(target.host):
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
            if isinstance(len_s, int):
                if len_s > len_r:
                    TS, TR = 4, 32
                else:
                    TS, TR = 16, 32
        elif target.kind.name == "opencl" and "mali" in str(target.attrs):
            VEC_C = 8
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 64
            TS, TR = 1, 64
        else:
            VEC_C = 1
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 64
            TS, TR = 1, 64

        if not isinstance(len_s, int):
            TS, TR = 1, 64

        while TS * TR > int(target.attrs["max_num_threads"]):
            if TS > 1:
                TS //= 2
            else:
                TR //= 2

        TILE_S, TILE_R = 2, max(get_max_factor(len_r, [TR * 1, TR * 2, TR * 4, TR * 8]) // TR, 1)
        VEC_C = min(get_max_factor(TILE_R, [1, 2, 4, 8]), VEC_C)
        VEC_LOAD = 1
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
        )

    def sch_outer_reduction(  # pylint: disable=too-many-arguments, invalid-name, unused-argument
        self,
        sch: s_tir.Schedule,
        target: Target,
        block: s_tir.schedule.SBlockRV,
        dequantize_block: s_tir.schedule.SBlockRV | None,
        pad_input_block: s_tir.schedule.SBlockRV | None,
        vector_input_tensors: list[tirx.Var],
        epilogue_info: SBlockInfo | None,
        batch_pad: int,
    ):
        """Schedule the outer reduction block."""

        # Need to detect from the block
        DEC_PACK = 8
        SCALE_PACK = 4

        def apply(
            sch: s_tir.Schedule,
            main_block: s_tir.schedule.SBlockRV,
            TAG_S: Literal["threadIdx.x", "threadIdx.y"],
            TAG_R: Literal["threadIdx.x", "threadIdx.y"],
            TS: int,
            TR: int,
            VEC: int,
            UNROLL: int,
        ):
            # rfactor: reduce to tx * vec_c
            b, s, r = sch.get_loops(main_block)
            by, batch = sch.split(b, [None, batch_pad], preserve_unit_iters=True)
            bx, ts = sch.split(s, [None, TS], preserve_unit_iters=True)
            r, tr, scale_c, vec_c = sch.split(
                r, [None, TR, SCALE_PACK, DEC_PACK], preserve_unit_iters=True
            )
            sch.reorder(by, bx, ts, r, batch, scale_c, tr, vec_c)
            tr_vec_c = sch.fuse(tr, vec_c)
            rf = sch.rfactor(tr_vec_c, 0)

            # rfactor: reduce to tx
            by, bx, ts, batch, tr_vec_c = sch.get_loops(block=main_block)
            tr, vec_c = sch.split(tr_vec_c, [TR, DEC_PACK], preserve_unit_iters=True)
            rf2 = sch.rfactor(tr, 0)

            # bind, vectorize compute
            by, bx, ts, r, batch, scale_c, tr_vec_c = sch.get_loops(block=rf)
            tr, vec_c = sch.split(tr_vec_c, [TR, DEC_PACK], preserve_unit_iters=True)
            sch.reorder(by, bx, ts, tr, r, scale_c, batch, vec_c)
            sch.bind(by, "blockIdx.y")
            sch.bind(bx, "blockIdx.x")
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            auto_vectorize(sch, vec_c, VEC)

            if dequantize_block is not None:
                sch.compute_at(dequantize_block, scale_c, preserve_unit_loops=True)
                sch.set_scope(dequantize_block, 0, "local")
                auto_vectorize(sch, sch.fuse(*sch.get_loops(dequantize_block)[6:]), VEC)

                B0_local = sch.cache_read(dequantize_block, 0, "local")
                sch.compute_at(B0_local, r, preserve_unit_loops=True)
                auto_vectorize(sch, sch.fuse(*sch.get_loops(B0_local)[5:]), VEC)

                B1_local = sch.cache_read(dequantize_block, 1, "local")
                sch.compute_at(B1_local, r, preserve_unit_loops=True)
                auto_vectorize(sch, sch.fuse(*sch.get_loops(B1_local)[5:]), VEC)
            else:
                # Only support quantized workloads for now
                sch = None
                return

            if LOAD_V_SHARED:
                sch.set_scope(pad_input_block, 0, "shared")
                sch.compute_at(pad_input_block, r, preserve_unit_loops=True)
                sch.storage_align(pad_input_block, 0, axis=-2, factor=8, offset=1)
                tr, ts, v = sch.split(sch.fuse(*sch.get_loops(pad_input_block)[5:]), [TR, TS, None])
                sch.bind(tr, TAG_R)
                sch.bind(ts, TAG_S)
                auto_vectorize(sch, v, VEC)
            else:
                sch.compute_inline(pad_input_block)

            # reduce tile_s * tr * vec to tile_s * tr
            sch.reverse_compute_at(rf2, bx, preserve_unit_loops=True)
            tr, vec_c, batch, ts = sch.get_loops(rf2)[2:]
            sch.reorder(ts, tr, batch, vec_c)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)

            # reduce tile_s * tr to tile_s
            sch.reverse_compute_at(main_block, bx, preserve_unit_loops=True)
            tr, batch, ts = sch.get_loops(main_block)[2:]
            sch.reorder(batch, ts, tr)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            # unroll(batch, 1)

            sch.decompose_reduction(rf, loop=sch.get_loops(block=rf)[4])
            sch.decompose_reduction(rf2, loop=sch.get_loops(block=rf2)[4])

            sch.set_scope(rf, tensor_index=0, storage_scope="local")
            sch.set_scope(rf2, tensor_index=0, storage_scope="local")

            epilogue = sch.get_consumers(main_block)
            # Schedule epilogue
            if epilogue:
                epilogue = epilogue[0]
                if is_broadcast_epilogue(  # pylint: disable=no-else-raise
                    sch, main_block, epilogue
                ):
                    raise NotImplementedError
                else:
                    sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
                    batch, ts = sch.get_loops(epilogue)[2:]
                    sch.bind(ts, TAG_S)
                    sch.set_scope(main_block, 0, "local")

        if target.kind.name == "metal":
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            TS, TR = 64, 4
            LOAD_V_SHARED = True
            VEC = 4
            UNROLL = 8
        else:
            # fallback configuration
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            TS, TR = 32, 4
            LOAD_V_SHARED = False
            VEC = 1
            UNROLL = 64

        return apply(
            sch,
            block,
            TAG_S,
            TAG_R,
            TS,
            TR,
            VEC,
            UNROLL,
        )
