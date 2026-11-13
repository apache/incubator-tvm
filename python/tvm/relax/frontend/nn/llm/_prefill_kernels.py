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
# ruff: noqa: E501
# fmt: off

"""Prefill attention kernels for (paged/ragged/MLA/dense) KV storage.

All of the ``@T.function`` factories below share the same online-softmax
skeleton that is built up from ``@T.macro`` helpers in
``_kernel_common._make_prefill_macros``. Each kernel only supplies the
K/V loading path that is specific to its storage layout.
"""

# pylint: disable=too-many-statements,too-many-arguments,invalid-name,line-too-long
import math
from typing import Any, Literal

import tvm
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

from ._kernel_common import (
    _alloc_mha_qkvo_buffers,
    _alloc_mla_qkvo_buffers,
    _alloc_softmax_state_buffers,
    _alloc_tile_walk_state,
    _causal_mask,
    _causal_or_sliding_cross_mask,
    _get_kv_chunk_len,
    _get_prefill_kernel_config,
    _get_seq_offset,
    _length_info_buffer,
    _make_prefill_macros,
    _rope,
    _schedule_prefill_kernel,
)
from ._prefill_parallel import (
    _attention_prefill_paged_parallel,
    _attention_prefill_ragged_parallel,
)


def _attention_prefill_cpu(
    h_kv,
    h_q,
    d,
    dtype,
    sliding_window: bool,
    rope_scaling: dict[str, Any],
    page_size: int = 16,
    sliding_window_size: int = 1024,
):
    global_symbol = "batch_prefill_paged_kv_cpu"
    if sliding_window:
        global_symbol += "_sliding_window"

    group_size = h_q // h_kv

    # pylint: disable=too-many-branches
    batch_size = T.dynamic("batch_size", "int32")
    total_len = T.dynamic("total_len", "int32")
    nnz_pages = T.dynamic("nnz_pages", "int32")
    max_num_pages = T.dynamic("max_num_pages", "int32")
    q_indptr_elem_offset = T.dynamic("q_indptr_elem_offset", "int32")
    page_indptr_elem_offset = T.dynamic("page_indptr_elem_offset", "int32")
    page_values_elem_offset = T.dynamic("page_values_elem_offset", "int32")
    k_rope_pos_offset_elem_offset = T.dynamic("k_rope_pos_offset_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    length_info_elem_offset = T.dynamic("length_info_elem_offset", "int32")
    @Ts.function
    def batch_prefill_paged_kv_cpu(
        q: T.Tensor((total_len, h_q, d), dtype), # [total_len, h_q, d]
        q_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (q_indptr_elem_offset))), # [batch_size + 1]
        pages: T.Tensor((max_num_pages, 2, h_kv, page_size, d), dtype), # [max_num_pages, 2, h_kv, page_size, d]
        page_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (page_indptr_elem_offset))), # [batch_size + 1]
        page_values: T.Tensor((nnz_pages,), 'int32', layout=T.TileLayout(T.S[(nnz_pages,)] + (page_values_elem_offset))), # [nnz_pages]
        length_info: _length_info_buffer(batch_size, sliding_window, length_info_elem_offset), # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset: T.Tensor((batch_size,), 'int32', layout=T.TileLayout(T.S[(batch_size,)] + (k_rope_pos_offset_elem_offset))), # [b]
        q_rope_position: T.Tensor((total_len,), 'int32', layout=T.TileLayout(T.S[(total_len,)] + (q_rope_position_elem_offset))), # [total_len]
        output: T.Tensor((total_len, h_q, d), dtype), # [total_len, h_q, d]
        lse: T.Tensor((total_len, h_q), 'float32'), # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"global_symbol": global_symbol})

          # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".

        for h_qo in T.serial(h_q):
            for b_idx in T.serial(batch_size):
                with Ts.sblock("attn"):
                    O_local = Ts.sblock_alloc_tensor((d, ), "float32")
                    Q_local = Ts.sblock_alloc_tensor((d, ), "float32")
                    K_local = Ts.sblock_alloc_tensor((d, ), "float32")
                    V_local = Ts.sblock_alloc_tensor((d, ), "float32")

                    kv_chunk_len = Ts.sblock_alloc_tensor((1, ), "int32")

                    m_val = Ts.sblock_alloc_tensor((1, ), "float32")
                    new_m = Ts.sblock_alloc_tensor((1, ), "float32")
                    d_val = Ts.sblock_alloc_tensor((1, ), "float32")
                    S_val = Ts.sblock_alloc_tensor((1, ), "float32")
                    scale_O = Ts.sblock_alloc_tensor((1, ), "float32")
                    factor = Ts.sblock_alloc_tensor((1, ), "float32")
                    cur_page_indptr_begin: T.let[T.int32] = page_indptr[b_idx]
                    cur_page_indptr_end: T.let[T.int32] = page_indptr[b_idx + 1]
                    #max_kv_len: T.let[T.int32] = max_num_pages * page_size
                    kv_chunk_len[0] = T.if_then_else(
                        cur_page_indptr_begin != cur_page_indptr_end,
                        _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, b_idx, length_info, sliding_window),
                        0
                    )

                    for q_idx in T.serial(q_indptr[b_idx + 1] - q_indptr[b_idx]):
                        #init m, d, O
                        m_val[0] = -5e4
                        d_val[0] = 1.0
                        for d_idx in T.serial(d):
                            O_local[d_idx] = 0.0
                        curl_q: T.let[T.int32] = q_indptr[b_idx] + q_idx

                        for d_idx in T.serial(d):

                            Q_local[d_idx] = T.if_then_else(
                                rotary_mode == 1,
                                _rope(q, q_rope_position[curl_q], d, rope_theta, rope_scale, (curl_q, h_qo, d_idx), dtype, rope_scaling),
                                q[curl_q, h_qo, d_idx]
                            )
                        for row_idx in T.serial(max_num_pages * page_size):
                            if row_idx < kv_chunk_len[0]:
                                page_no: T.let[T.int32()] = page_values[cur_page_indptr_begin + (_get_seq_offset(row_idx, b_idx, length_info, sliding_window) // page_size)]
                                page_offset: T.let[T.int32()] = _get_seq_offset(row_idx, b_idx, length_info, sliding_window) % page_size

                                # Load KV
                                for d_idx in T.serial(d):
                                    K_local[d_idx] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(pages, k_rope_pos_offset[b_idx] + row_idx, d, rope_theta, rope_scale, (page_no, 0, h_qo // group_size, page_offset, d_idx), dtype, rope_scaling),
                                        pages[page_no, 0, h_qo // group_size, page_offset, d_idx]
                                    )
                                    V_local[d_idx] = pages[page_no, 1, h_qo // group_size, page_offset, d_idx]

                                # Compute S
                                # Q[i] * K[i] * sm_scale
                                S_val[0] = 0.0
                                for d_idx in T.serial(d):
                                    S_val[0] += Q_local[d_idx] * K_local[d_idx]
                                S_val[0] *= sm_scale * math.log2(math.exp(1))

                                # update m_val, d_val , O_local
                                new_m[0] = m_val[0]
                                if _causal_or_sliding_cross_mask(
                                    causal,
                                    row=q_idx,
                                    col=row_idx,
                                    kv_len=kv_chunk_len[0],
                                    qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx],
                                    sliding_window_size=(sliding_window_size if sliding_window else 0),
                                ):
                                    new_m[0] = T.max(m_val[0], S_val[0])
                                else:
                                    S_val[0] = -5e4
                                # update d_val
                                d_val[0] *= T.exp2(m_val[0] - new_m[0])
                                d_val[0] += T.exp2(S_val[0] - new_m[0])

                                # restore O_local then update O_local
                                scale_O[0] = T.exp2(m_val[0] - new_m[0])
                                m_val[0] = new_m[0]
                                factor[0] = T.exp2(S_val[0] - m_val[0])
                                for d_idx in T.serial(d):
                                    O_local[d_idx] = O_local[d_idx] * scale_O[0]

                                for d_idx in T.serial(d):
                                    O_local[d_idx] += V_local[d_idx] * factor[0]
                        # Store Output
                        for d_idx in T.serial(d):
                            O_local[d_idx] = O_local[d_idx] /d_val[0]
                            output[curl_q, h_qo, d_idx] = O_local[d_idx]
                        lse[curl_q, h_qo] = m_val[0] + T.log2(d_val[0])
    return batch_prefill_paged_kv_cpu

def _attention_prefill(
    h_kv,
    h_q,
    d,
    dtype,
    sliding_window: bool,
    rope_scaling: dict[str, Any],
    target: Target,
    page_size: int = 16,
    sliding_window_size: int = 1024,
):
    if target.kind.name == "webgpu" and dtype == "float32" and d == 512:
        return _attention_prefill_paged_parallel(
            h_kv, h_q, rope_scaling, sliding_window, page_size, sliding_window_size
        )
    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(h_kv, h_q, d, dtype, target)

    global_symbol = "batch_prefill_paged_kv"
    if sliding_window:
        global_symbol += "_sliding_window"

    init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, _, advance_tile_batch, paged_store_output_lse, *_ = _make_prefill_macros(tile_x, tile_y, tile_z, tile_y, bdx, num_warps, group_size)

    # pylint: disable=too-many-branches
    batch_size = T.dynamic("batch_size", "int32")
    total_len = T.dynamic("total_len", "int32")
    nnz_pages = T.dynamic("nnz_pages", "int32")
    max_num_pages = T.dynamic("max_num_pages", "int32")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    q_indptr_elem_offset = T.dynamic("q_indptr_elem_offset", "int32")
    page_indptr_elem_offset = T.dynamic("page_indptr_elem_offset", "int32")
    page_values_elem_offset = T.dynamic("page_values_elem_offset", "int32")
    k_rope_pos_offset_elem_offset = T.dynamic("k_rope_pos_offset_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    length_info_elem_offset = T.dynamic("length_info_elem_offset", "int32")
    @Ts.function
    def batch_prefill_paged_kv(
        q: T.Tensor((total_len, h_q, d), dtype), # [total_len, h_q, d]
        q_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (q_indptr_elem_offset))), # [batch_size + 1]
        pages: T.Tensor((max_num_pages, 2, h_kv, page_size, d), dtype, layout=T.TileLayout(T.S[(max_num_pages, 2, h_kv, page_size, d)] + (pages_elem_offset))), # [max_num_pages, 2, h_kv, page_size, d]
        page_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (page_indptr_elem_offset))), # [batch_size + 1]
        page_values: T.Tensor((nnz_pages,), 'int32', layout=T.TileLayout(T.S[(nnz_pages,)] + (page_values_elem_offset))), # [nnz_pages]
        length_info: _length_info_buffer(batch_size, sliding_window, length_info_elem_offset), # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset: T.Tensor((batch_size,), 'int32', layout=T.TileLayout(T.S[(batch_size,)] + (k_rope_pos_offset_elem_offset))), # [b]
        q_rope_position: T.Tensor((total_len,), 'int32', layout=T.TileLayout(T.S[(total_len,)] + (q_rope_position_elem_offset))), # [total_len]
        output: T.Tensor((total_len, h_q, d), dtype), # [total_len, h_q, d]
        lse: T.Tensor((total_len, h_q), 'float32'), # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"global_symbol": global_symbol})

          # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".

        # kernel code
        for lbx in T.parallel(NUM_BLKS, annotations={"thread_binding": "blockIdx.x"}):
            for lby in T.parallel(h_kv, annotations={"thread_binding": "blockIdx.y"}):
                for lty in T.parallel(num_warps, annotations={"thread_binding": "threadIdx.y"}):
                    for ltx in T.parallel(bdx, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("attn"):
                            bx, by, ty, tx = Ts.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            Ts.reads()
                            Ts.writes()
                            tile_id, batch_idx, batch_tiles, batch_rows, iterator, kv_chunk_len = _alloc_tile_walk_state()
                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d, d, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                                _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                            )

                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.gpu_thread_invariant(batch_idx[0] < batch_size):
                                advance_tile_batch(tile_id, batch_idx, batch_tiles, batch_rows, q_indptr, batch_size)

                                if T.gpu_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.let[T.int32] = batch_idx[0]
                                    LH_start: T.let[T.int32] = tile_id[0] * tile_x
                                    q_indptr_val: T.let[T.int32] = q_indptr[b_idx]

                                    cur_page_indptr_begin: T.let[T.int32] = page_indptr[b_idx]
                                    cur_page_indptr_end: T.let[T.int32] = page_indptr[b_idx + 1]
                                    kv_chunk_len[0] = T.if_then_else(
                                        cur_page_indptr_begin != cur_page_indptr_end,
                                        _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, b_idx, length_info, sliding_window),
                                        0
                                    )
                                    T.gpu_storage_sync("shared")

                                    init_states(m_smem, d_smem, O_local, ty, tx)

                                    # Load Q from gmem to smem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with Ts.sblock("Q_load"):
                                            i, j = Ts.axis.remap("SS", [li, lj])
                                            Ts.reads()
                                            Ts.writes()
                                            cur_L: T.let[T.int32] = q_indptr_val + (LH_start + i) // group_size
                                            cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype, rope_scaling),
                                                    q[cur_L, cur_H_qo, j]
                                                )
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.gpu_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.let[T.int32] = iterator * tile_z
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with Ts.sblock("K_load"):
                                                i, j = Ts.axis.remap("SS", [lz, ly])
                                                Ts.reads()
                                                Ts.writes()
                                                cur_L: T.let[T.int32] = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    seq_offset: T.let[T.int32()] = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                    page_no: T.let[T.int32()] = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, page_size)]  # type: ignore
                                                    page_offset: T.let[T.int32()] = T.floormod(seq_offset, page_size)  # type: ignore
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(pages, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (page_no, 0, by, page_offset, j), dtype, rope_scaling),
                                                        pages[page_no, 0, by, page_offset, j]
                                                    )
                                                else:
                                                    K_smem[i, j] = 0.0
                                        T.gpu_storage_sync("shared")
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with Ts.sblock("V_load"):
                                                i, j = Ts.axis.remap("SS", [lz, ly])
                                                Ts.reads()
                                                Ts.writes()
                                                cur_L: T.let[T.int32] = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    seq_offset: T.let[T.int32()] = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                    page_no: T.let[T.int32()] = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, page_size)]  # type: ignore
                                                    page_offset: T.let[T.int32()] = T.floormod(seq_offset, page_size)  # type: ignore
                                                    V_smem[i, j] = pages[page_no, 1, by, page_offset, j]
                                                else:
                                                    V_smem[i, j] = 0.0
                                        T.gpu_storage_sync("shared")

                                        compute_s_gemm(Q_smem, K_smem, S_local, S_smem, sm_scale)
                                        softmax_update_causal(S_smem, m_smem, d_smem, m_prev_smem, m_new, m_prev, d_new, ty, tx, LH_start, L_kv_start, causal, kv_chunk_len[0], q_indptr[b_idx + 1] - q_indptr[b_idx], sliding_window_size if sliding_window else 0)
                                        compute_o_gemm(S_smem, V_smem, O_local, m_prev_smem, m_smem)

                                    paged_store_output_lse(output, lse, O_local, m_smem, d_smem, q_indptr, b_idx, by, LH_start)

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # pylint: enable=too-many-branches
    sch = tvm.s_tir.Schedule(batch_prefill_paged_kv)
    sch = _schedule_prefill_kernel(
        sch, LOAD_VEC, bdx, num_warps, tile_x, tile_y, tile_z, False, False
    )
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)

def _attention_sequence_prefill(h_kv, h_q, d, dtype, target: Target, causal=0, sm_scale=1.0):
    _, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(h_kv, h_q, d, dtype, target)
    init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, *_ = _make_prefill_macros(tile_x, tile_y, tile_z, tile_y, bdx, num_warps, group_size)

    batch_size = T.dynamic("batch_size", "int32")
    qo_len = T.dynamic("qo_len", "int32")
    kv_len = T.dynamic("kv_len", "int32")
    @Ts.function
    def batch_sequence_prefill_kv(  # pylint: disable=too-many-branches
        q: T.Tensor((batch_size, qo_len, h_q, d), dtype), # [total_len, h_q, d]
        k: T.Tensor((batch_size, kv_len, h_kv, d), dtype), # [total_len, h_kv, d]
        v: T.Tensor((batch_size, kv_len, h_kv, d), dtype), # [total_len, h_kv, d]
        output: T.Tensor((batch_size, qo_len, h_q, d), dtype), # [total_len, h_q, d]
        lse: T.Tensor((batch_size, qo_len, h_q), dtype) # [total_len, h_q]
    ):

          # pylint: disable=unused-variable

        batch_tiles: T.let[T.int32] = T.ceildiv(qo_len * group_size, tile_x)

        # kernel code
        for lbx in T.parallel(T.cast(batch_size, ty="int32") * batch_tiles, annotations={"thread_binding": "blockIdx.x"}):
            for lby in T.parallel(h_kv, annotations={"thread_binding": "blockIdx.y"}):
                for lty in T.parallel(num_warps, annotations={"thread_binding": "threadIdx.y"}):
                    for ltx in T.parallel(bdx, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("attn"):
                            vbx, by, ty, tx = Ts.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            Ts.reads()
                            Ts.writes()

                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d, d, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                                _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                            )

                            b_idx: T.let[T.int32] = vbx // batch_tiles
                            tile_id: T.let[T.int32] = vbx % batch_tiles
                            LH_start: T.let[T.int32] = tile_id * tile_x
                            T.gpu_storage_sync("shared")

                            init_states(m_smem, d_smem, O_local, ty, tx)

                            # Load Q from gmem to smem
                            for li, lj in T.grid(tile_x, tile_y):
                                with Ts.sblock("Q_load"):
                                    i, j = Ts.axis.remap("SS", [li, lj])
                                    Ts.reads()
                                    Ts.writes()
                                    cur_L: T.let[T.int32] = (LH_start + i) // group_size
                                    cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        Q_smem[i, j] = q[b_idx, cur_L, cur_H_qo, j]
                                    else:
                                        Q_smem[i, j] = 0.0
                            T.gpu_storage_sync("shared")

                            for iterator in T.serial(T.ceildiv(kv_len, tile_z)):
                                L_kv_start: T.let[T.int32] = iterator * tile_z
                                L_kv_base: T.let[T.int32] = 0
                                for lz, ly in T.grid(tile_z, tile_y):
                                    with Ts.sblock("K_load"):
                                        i, j = Ts.axis.remap("SS", [lz, ly])
                                        Ts.reads()
                                        Ts.writes()
                                        cur_L: T.let[T.int32] = L_kv_start + i
                                        if cur_L < kv_len:
                                            K_smem[i, j] = k[
                                                b_idx, L_kv_base + cur_L, by, j
                                            ]
                                        else:
                                            K_smem[i, j] = 0.0
                                T.gpu_storage_sync("shared")
                                for lz, ly in T.grid(tile_z, tile_y):
                                    with Ts.sblock("V_load"):
                                        i, j = Ts.axis.remap("SS", [lz, ly])
                                        Ts.reads()
                                        Ts.writes()
                                        cur_L: T.let[T.int32] = L_kv_start + i
                                        if cur_L < kv_len:
                                            V_smem[i, j] = v[b_idx, L_kv_base + cur_L, by, j]
                                        else:
                                            V_smem[i, j] = 0.0
                                T.gpu_storage_sync("shared")

                                compute_s_gemm(Q_smem, K_smem, S_local, S_smem, sm_scale)
                                softmax_update_causal(S_smem, m_smem, d_smem, m_prev_smem, m_new, m_prev, d_new, ty, tx, LH_start, L_kv_start, causal, kv_len, qo_len, 0)
                                compute_o_gemm(S_smem, V_smem, O_local, m_prev_smem, m_smem)

                            # Store O from smem to gmem
                            for li, lj in T.grid(tile_x, tile_y):
                                with Ts.sblock("O_store"):
                                    i, j = Ts.axis.remap("SS", [li, lj])
                                    cur_L: T.let[T.int32] = 0 + (LH_start + i) // group_size
                                    cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        output[b_idx, cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                            # Store LSE to gmem
                            for li in T.grid(tile_x):
                                with Ts.sblock("lse_store"):
                                    i = Ts.axis.remap("S", [li])
                                    cur_L: T.let[T.int32] = 0 + (LH_start + i) // group_size
                                    cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        lse[b_idx, cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

    # pylint: enable=too-many-branches
    sch = tvm.s_tir.Schedule(batch_sequence_prefill_kv)
    sch = _schedule_prefill_kernel(sch, LOAD_VEC, bdx, num_warps, tile_x, tile_y, tile_z, False, False)
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)

def _attention_sequence_prefill_with_mask(
    h_kv, h_q, d, dtype, target: Target, sm_scale=1.0, *,
    mask_mode: Literal["padded", "causal_padded_left"] = "padded",
):
    """Tiled sequence prefill kernel with a per-batch padding mask.

    Supports two mask regimes selected by ``mask_mode``:

    * ``"padded"`` (default) — bidirectional attention with right-padding.
      For batch ``b``, positions ``[0, valid_lens[b])`` are real and
      positions ``[valid_lens[b], seq_len)`` are padding. This is the
      encoder-style batch regime.
    * ``"causal_padded_left"`` — causal attention with left-padding. For
      batch ``b``, positions ``[seq_len - valid_lens[b], seq_len)`` are
      real and positions ``[0, seq_len - valid_lens[b])`` are padding;
      the causal constraint additionally keeps ``col <= row`` within the
      valid range. This is the decoder-embedding batch regime, where
      last-token pooling is a cheap slice of the final row.

    In both modes the kernel takes an extra ``valid_lens`` buffer of shape
    ``(batch_size,)`` and applies the mask inside the QKV load path and the
    online softmax update, so no explicit mask tensor broadcast or additive
    bias is needed on the host side. Padding queries and keys/values are
    zeroed at load time; masked ``(row, col)`` pairs are excluded from the
    max/sum of the online softmax via a ``-inf`` slot. ``valid_len`` is the
    per-batch real token count shared by Q and K/V; cross-attention with
    independent Q/K validity is out of scope.
    """
    _, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(h_kv, h_q, d, dtype, target)
    (
        init_states, compute_s_gemm, _, compute_o_gemm, softmax_update_valid_length,
        _, _, softmax_update_causal_padded_left,
    ) = _make_prefill_macros(tile_x, tile_y, tile_z, tile_y, bdx, num_warps, group_size)

    softmax_update = (
        softmax_update_valid_length
        if mask_mode == "padded"
        else softmax_update_causal_padded_left
    )

    def _q_row_valid(row, valid_len, qo_len):
        # Row-validity predicate for Q load (TIR expression); mask_mode is
        # captured at closure time so the function body stays specialised.
        if mask_mode == "padded":
            return tirx.And(row < qo_len, row < valid_len)
        pad = qo_len - valid_len
        return tirx.And(row < qo_len, row >= pad)

    def _kv_col_valid(col, valid_len, kv_len):
        # Column-validity predicate for K/V load (TIR expression).
        if mask_mode == "padded":
            return tirx.And(col < kv_len, col < valid_len)
        pad = kv_len - valid_len
        return tirx.And(col < kv_len, col >= pad)

    batch_size = T.dynamic("batch_size", "int32")
    qo_len = T.dynamic("qo_len", "int32")
    kv_len = T.dynamic("kv_len", "int32")
    @Ts.function
    def batch_sequence_prefill_kv_masked(  # pylint: disable=too-many-branches
        q: T.Tensor((batch_size, qo_len, h_q, d), dtype), # [batch_size, qo_len, h_q, d]
        k: T.Tensor((batch_size, kv_len, h_kv, d), dtype), # [batch_size, kv_len, h_kv, d]
        v: T.Tensor((batch_size, kv_len, h_kv, d), dtype), # [batch_size, kv_len, h_kv, d]
        valid_lens: T.Tensor((batch_size,), 'int32'), # [batch_size], int32
        output: T.Tensor((batch_size, qo_len, h_q, d), dtype), # [batch_size, qo_len, h_q, d]
        lse: T.Tensor((batch_size, qo_len, h_q), dtype) # [batch_size, qo_len, h_q]
    ):

        batch_tiles: T.let[T.int32] = T.ceildiv(qo_len * group_size, tile_x)

        for lbx in T.parallel(T.cast(batch_size, ty="int32") * batch_tiles, annotations={"thread_binding": "blockIdx.x"}):
            for lby in T.parallel(h_kv, annotations={"thread_binding": "blockIdx.y"}):
                for lty in T.parallel(num_warps, annotations={"thread_binding": "threadIdx.y"}):
                    for ltx in T.parallel(bdx, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("attn"):
                            vbx, by, ty, tx = Ts.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            Ts.reads()
                            Ts.writes()

                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d, d, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                                _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                            )

                            b_idx: T.let[T.int32] = vbx // batch_tiles
                            valid_len: T.let[T.int32] = valid_lens[b_idx]
                            tile_id: T.let[T.int32] = vbx % batch_tiles
                            LH_start: T.let[T.int32] = tile_id * tile_x
                            T.gpu_storage_sync("shared")

                            init_states(m_smem, d_smem, O_local, ty, tx)

                            # Load Q; rows outside the valid range are zeroed so they contribute nothing downstream.
                            for li, lj in T.grid(tile_x, tile_y):
                                with Ts.sblock("Q_load"):
                                    i, j = Ts.axis.remap("SS", [li, lj])
                                    Ts.reads()
                                    Ts.writes()
                                    cur_L: T.let[T.int32] = (LH_start + i) // group_size
                                    cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                    if _q_row_valid(cur_L, valid_len, qo_len):
                                        Q_smem[i, j] = q[b_idx, cur_L, cur_H_qo, j]
                                    else:
                                        Q_smem[i, j] = 0.0
                            T.gpu_storage_sync("shared")

                            for iterator in T.serial(T.ceildiv(kv_len, tile_z)):
                                L_kv_start: T.let[T.int32] = iterator * tile_z
                                L_kv_base: T.let[T.int32] = 0
                                for lz, ly in T.grid(tile_z, tile_y):
                                    with Ts.sblock("K_load"):
                                        i, j = Ts.axis.remap("SS", [lz, ly])
                                        Ts.reads()
                                        Ts.writes()
                                        cur_L: T.let[T.int32] = L_kv_start + i
                                        if _kv_col_valid(cur_L, valid_len, kv_len):
                                            K_smem[i, j] = k[b_idx, L_kv_base + cur_L, by, j]
                                        else:
                                            K_smem[i, j] = 0.0
                                T.gpu_storage_sync("shared")
                                for lz, ly in T.grid(tile_z, tile_y):
                                    with Ts.sblock("V_load"):
                                        i, j = Ts.axis.remap("SS", [lz, ly])
                                        Ts.reads()
                                        Ts.writes()
                                        cur_L: T.let[T.int32] = L_kv_start + i
                                        if _kv_col_valid(cur_L, valid_len, kv_len):
                                            V_smem[i, j] = v[b_idx, L_kv_base + cur_L, by, j]
                                        else:
                                            V_smem[i, j] = 0.0
                                T.gpu_storage_sync("shared")

                                compute_s_gemm(Q_smem, K_smem, S_local, S_smem, sm_scale)
                                softmax_update(S_smem, m_smem, d_smem, m_prev_smem, m_new, m_prev, d_new, ty, tx, LH_start, L_kv_start, valid_len, qo_len, kv_len)
                                compute_o_gemm(S_smem, V_smem, O_local, m_prev_smem, m_smem)

                            # Store O
                            for li, lj in T.grid(tile_x, tile_y):
                                with Ts.sblock("O_store"):
                                    i, j = Ts.axis.remap("SS", [li, lj])
                                    cur_L: T.let[T.int32] = 0 + (LH_start + i) // group_size
                                    cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        output[b_idx, cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                            # Store LSE
                            for li in T.grid(tile_x):
                                with Ts.sblock("lse_store"):
                                    i = Ts.axis.remap("S", [li])
                                    cur_L: T.let[T.int32] = 0 + (LH_start + i) // group_size
                                    cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        lse[b_idx, cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

    sch = tvm.s_tir.Schedule(batch_sequence_prefill_kv_masked)
    sch = _schedule_prefill_kernel(sch, LOAD_VEC, bdx, num_warps, tile_x, tile_y, tile_z, False, False)
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)

def _attention_prefill_ragged_cpu(h_kv, h_q, d_qk, d_v, dtype, rope_scaling: dict[str, Any]):
    group_size = h_q // h_kv

    batch_size = T.dynamic("batch_size", "int32")
    qo_len = T.dynamic("qo_len", "int32")
    kv_len = T.dynamic("kv_len", "int32")
    q_indptr_elem_offset = T.dynamic("q_indptr_elem_offset", "int32")
    kv_indptr_elem_offset = T.dynamic("kv_indptr_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    k_rope_pos_offset_elem_offset = T.dynamic("k_rope_pos_offset_elem_offset", "int32")
    @Ts.function
    def batch_prefill_ragged_kv(  # pylint: disable=too-many-branches
        q: T.Tensor((qo_len, h_q, d_qk), dtype),  # [total_len, h_q, d_qk]
        q_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (q_indptr_elem_offset))),  # [batch_size + 1]
        k: T.Tensor((kv_len, h_kv, d_qk), dtype),  # [total_len, h_kv, d_qk]
        v: T.Tensor((kv_len, h_kv, d_v), dtype),  # [total_len, h_kv, d_v]
        kv_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (kv_indptr_elem_offset))),  # [batch_size + 1]
        q_rope_position: T.Tensor((qo_len,), 'int32', layout=T.TileLayout(T.S[(qo_len,)] + (q_rope_position_elem_offset))),  # [total_q_len]
        k_rope_pos_offset: T.Tensor((batch_size,), 'int32', layout=T.TileLayout(T.S[(batch_size,)] + (k_rope_pos_offset_elem_offset))),  # [b]
        output: T.Tensor((qo_len, h_q, d_v), dtype),  # [total_len, h_q, d_v]
        lse: T.Tensor((qo_len, h_q), 'float32'),  # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):

          # pylint: disable=unused-variable

        for b in T.serial(batch_size):
            with Ts.sblock("attn"):
                softmax_sum = Ts.sblock_alloc_tensor([h_q], "float32")
                m_prev = Ts.sblock_alloc_tensor([h_q], "float32")
                m_new = Ts.sblock_alloc_tensor([h_q], "float32")
                d_prev = Ts.sblock_alloc_tensor([h_q], "float32")
                d_new = Ts.sblock_alloc_tensor([h_q], "float32")
                p_sum = Ts.sblock_alloc_tensor([d_v], "float32")
                max_score = Ts.sblock_alloc_tensor([h_q], "float32")
                attention_scores = Ts.sblock_alloc_tensor([kv_len, h_q], "float32")
                exp_scores = Ts.sblock_alloc_tensor([kv_len, h_q], "float32")
                attention_score = Ts.sblock_alloc_tensor([1], "float32")
                query_val = Ts.sblock_alloc_tensor([1], "float32")
                key_val = Ts.sblock_alloc_tensor([1], "float32")
                result = Ts.sblock_alloc_tensor([1], "float32")

                for q_idx in T.serial(q_indptr[b + 1] - q_indptr[b]):
                    for i in T.serial(h_q):
                        max_score[i] = -5e4
                        m_prev[i] = -5e4
                        d_prev[i] = 1.0

                    for k_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                        for h in T.serial(h_q):
                            h_kv_idx: T.let[T.int32] = h // group_size

                            if _causal_mask(
                                causal,
                                row=q_idx,
                                col=k_idx,
                                kv_len=kv_indptr[b + 1] - kv_indptr[b],
                                qo_len=q_indptr[b + 1] - q_indptr[b],
                            ):
                                result[0] = 0.0
                                for d_idx in T.serial(d_qk):
                                    query_val[0] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(q, q_rope_position[q_indptr[b] + q_idx], d_qk, rope_theta, rope_scale, (q_indptr[b] + q_idx, h, d_idx), dtype, rope_scaling),
                                        q[q_indptr[b] + q_idx, h, d_idx],
                                    )

                                    key_val[0] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(k, k_rope_pos_offset[b] + k_idx, d_qk, rope_theta, rope_scale, (kv_indptr[b] + k_idx, h_kv_idx, d_idx), dtype, rope_scaling),
                                        k[kv_indptr[b] + k_idx, h_kv_idx, d_idx],
                                    )

                                    result[0] += query_val[0] * key_val[0]
                                attention_score[0] = result[0] * math.log2(math.exp(1)) * sm_scale
                            else:
                                attention_score[0] = -5e4 * math.log2(math.exp(1)) * sm_scale
                            attention_scores[k_idx, h] = attention_score[0]
                            max_score[h] = T.max(max_score[h], attention_score[0])
                            m_new[h] = T.max(m_prev[h], max_score[h])

                    for h in T.serial(h_q):
                        d_new[h] = d_prev[h] * T.exp2(m_prev[h] - m_new[h])

                    for h in T.serial(h_q):
                        softmax_sum[h] = 0.0
                        for k_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                            exp_scores[k_idx, h] = T.exp2(attention_scores[k_idx, h] - m_new[h])
                            softmax_sum[h] += exp_scores[k_idx, h]
                        d_new[h] += softmax_sum[h]

                    for h in T.serial(h_q):
                        h_kv_idx: T.let[T.int32] = h // group_size
                        for i in T.serial(d_v):
                            p_sum[i] = 0.0
                        for v_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                            weight: T.let[T.float32] = exp_scores[v_idx, h] / d_new[h]
                            for i in T.serial(d_v):
                                p_sum[i] += v[kv_indptr[b] + v_idx, h_kv_idx, i] * weight
                        for i in T.serial(d_v):
                            output[q_indptr[b] + q_idx, h, i] = p_sum[i]
                        lse[q_indptr[b] + q_idx, h] = m_new[h] + T.log2(d_new[h])
    return batch_prefill_ragged_kv

def _attention_prefill_ragged(h_kv, h_q, d_qk, d_v, dtype, rope_scaling: dict[str, Any], target: Target):
    if target.kind.name == "webgpu" and dtype == "float32" and d_qk == d_v == 512:
        return _attention_prefill_ragged_parallel(h_kv, h_q, rope_scaling)
    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(h_kv, h_q, d_qk, dtype, target, d_v=d_v)
    init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, _, advance_tile_batch, paged_store_output_lse, *_ = _make_prefill_macros(tile_x, tile_y, tile_z, d_v, bdx, num_warps, group_size)

    batch_size = T.dynamic("batch_size", "int32")
    qo_len = T.dynamic("qo_len", "int32")
    kv_len = T.dynamic("kv_len", "int32")
    q_indptr_elem_offset = T.dynamic("q_indptr_elem_offset", "int32")
    kv_indptr_elem_offset = T.dynamic("kv_indptr_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    k_rope_pos_offset_elem_offset = T.dynamic("k_rope_pos_offset_elem_offset", "int32")
    @Ts.function
    def batch_prefill_ragged_kv(  # pylint: disable=too-many-branches
        q: T.Tensor((qo_len, h_q, d_qk), dtype), # [total_len, h_q, d_qk]
        q_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (q_indptr_elem_offset))), # [batch_size + 1]
        k: T.Tensor((kv_len, h_kv, d_qk), dtype), # [total_len, h_kv, d_qk]
        v: T.Tensor((kv_len, h_kv, d_v), dtype), # [total_len, h_kv, d_v]
        kv_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (kv_indptr_elem_offset))), # [batch_size + 1]
        q_rope_position: T.Tensor((qo_len,), 'int32', layout=T.TileLayout(T.S[(qo_len,)] + (q_rope_position_elem_offset))), # [total_q_len]
        k_rope_pos_offset: T.Tensor((batch_size,), 'int32', layout=T.TileLayout(T.S[(batch_size,)] + (k_rope_pos_offset_elem_offset))), # [b]
        output: T.Tensor((qo_len, h_q, d_v), dtype), # [total_len, h_q, d_v]
        lse: T.Tensor((qo_len, h_q), 'float32'), # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32
    ):

          # pylint: disable=unused-variable

        # kernel code
        for lbx in T.parallel(NUM_BLKS, annotations={"thread_binding": "blockIdx.x"}):
            for lby in T.parallel(h_kv, annotations={"thread_binding": "blockIdx.y"}):
                for lty in T.parallel(num_warps, annotations={"thread_binding": "threadIdx.y"}):
                    for ltx in T.parallel(bdx, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("attn"):
                            bx, by, ty, tx = Ts.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            Ts.reads()
                            Ts.writes()
                            tile_id, batch_idx, batch_tiles, batch_rows, iterator, kv_chunk_len = _alloc_tile_walk_state()
                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d_qk, d_v, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                                _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                            )

                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.gpu_thread_invariant(batch_idx[0] < batch_size):
                                advance_tile_batch(tile_id, batch_idx, batch_tiles, batch_rows, q_indptr, batch_size)

                                if T.gpu_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.let[T.int32] = batch_idx[0]
                                    q_indptr_val: T.let[T.int32] = q_indptr[b_idx]
                                    LH_start: T.let[T.int32] = tile_id[0] * tile_x

                                    kv_chunk_len[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                    T.gpu_storage_sync("shared")

                                    init_states(m_smem, d_smem, O_local, ty, tx)

                                    # Load Q from gmem to smem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with Ts.sblock("Q_load"):
                                            i, j = Ts.axis.remap("SS", [li, lj])
                                            Ts.reads()
                                            Ts.writes()
                                            cur_L: T.let[T.int32] = q_indptr_val + (LH_start + i) // group_size
                                            cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d_qk, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype, rope_scaling),
                                                    q[cur_L, cur_H_qo, j]
                                                )
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.gpu_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.let[T.int32] = iterator * tile_z
                                        L_kv_base: T.let[T.int32] = kv_indptr[b_idx]
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with Ts.sblock("K_load"):
                                                i, j = Ts.axis.remap("SS", [lz, ly])
                                                cur_L: T.let[T.int32] = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(k, k_rope_pos_offset[b_idx] + cur_L, d_qk, rope_theta, rope_scale, (L_kv_base + cur_L, by, j), dtype, rope_scaling),
                                                        k[L_kv_base + cur_L, by, j]
                                                    )
                                                else:
                                                    K_smem[i, j] = 0.0
                                        T.gpu_storage_sync("shared")
                                        for lz, ly in T.grid(tile_z, d_v):
                                            with Ts.sblock("V_load"):
                                                i, j = Ts.axis.remap("SS", [lz, ly])
                                                Ts.reads()
                                                Ts.writes()
                                                cur_L: T.let[T.int32] = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    V_smem[i, j] = v[L_kv_base + cur_L, by, j]
                                                else:
                                                    V_smem[i, j] = 0.0
                                        T.gpu_storage_sync("shared")

                                        compute_s_gemm(Q_smem, K_smem, S_local, S_smem, sm_scale)
                                        softmax_update_causal(S_smem, m_smem, d_smem, m_prev_smem, m_new, m_prev, d_new, ty, tx, LH_start, L_kv_start, causal, kv_chunk_len[0], q_indptr[b_idx + 1] - q_indptr[b_idx], 0)
                                        compute_o_gemm(S_smem, V_smem, O_local, m_prev_smem, m_smem)

                                    paged_store_output_lse(output, lse, O_local, m_smem, d_smem, q_indptr, b_idx, by, LH_start)

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # pylint: enable=too-many-branches
    sch = tvm.s_tir.Schedule(batch_prefill_ragged_kv)
    sch = _schedule_prefill_kernel(sch, LOAD_VEC, bdx, num_warps, tile_x, d_v, tile_z, True, False)
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)

def _attention_prefill_mla(h_q, d_latent, d_rope, dtype, sliding_window: bool, target: Target, page_size: int = 16):
    d_qk = d_latent + d_rope
    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(1, h_q, d_qk, dtype, target, d_v=d_latent, merged_kv=True)
    init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, _, advance_tile_batch, paged_store_output_lse, *_ = _make_prefill_macros(tile_x, tile_y, tile_z, d_latent, bdx, num_warps, group_size)

    global_symbol = "batch_prefill_paged_kv_mla"
    if sliding_window:
        global_symbol += "_sliding_window"

    # pylint: disable=too-many-branches
    batch_size = T.dynamic("batch_size", "int32")
    total_len = T.dynamic("total_len", "int32")
    nnz_pages = T.dynamic("nnz_pages", "int32")
    max_num_pages = T.dynamic("max_num_pages", "int32")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    q_indptr_elem_offset = T.dynamic("q_indptr_elem_offset", "int32")
    page_indptr_elem_offset = T.dynamic("page_indptr_elem_offset", "int32")
    page_values_elem_offset = T.dynamic("page_values_elem_offset", "int32")
    length_info_elem_offset = T.dynamic("length_info_elem_offset", "int32")
    @Ts.function
    def batch_prefill_paged_kv_mla(
        q: T.Tensor((total_len, h_q, d_qk), dtype), # [total_len, h_q, d_qk]
        q_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (q_indptr_elem_offset))), # [batch_size + 1]
        pages: T.Tensor((max_num_pages, page_size, d_qk), dtype, layout=T.TileLayout(T.S[(max_num_pages, page_size, d_qk)] + (pages_elem_offset))), # [max_num_pages, page_size, d_qk]
        page_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (page_indptr_elem_offset))), # [batch_size + 1]
        page_values: T.Tensor((nnz_pages,), 'int32', layout=T.TileLayout(T.S[(nnz_pages,)] + (page_values_elem_offset))), # [nnz_pages]
        length_info: _length_info_buffer(batch_size, sliding_window, length_info_elem_offset), # [b] when sliding window = False, or otherwise [3, b]
        output: T.Tensor((total_len, h_q, d_latent), dtype), # [total_len, h_q, d_latent]
        lse: T.Tensor((total_len, h_q), 'float32'), # [total_len, h_q]
        causal: T.int32,
        sm_scale: T.float32,
    ):
        T.func_attr({"global_symbol": global_symbol})

          # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".

        # kernel code
        for lbx in T.parallel(NUM_BLKS, annotations={"thread_binding": "blockIdx.x"}):
            for lty in T.parallel(num_warps, annotations={"thread_binding": "threadIdx.y"}):
                for ltx in T.parallel(bdx, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("attn"):
                        bx, ty, tx = Ts.axis.remap("SSS", [lbx, lty, ltx])
                        Ts.reads()
                        Ts.writes()
                        tile_id, batch_idx, batch_tiles, batch_rows, iterator, kv_chunk_len = _alloc_tile_walk_state()
                        Q_smem, KV_smem, O_local = _alloc_mla_qkvo_buffers(tile_x, tile_z, d_qk, d_latent, dtype)
                        S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                            _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                        )

                        tile_id[0] = bx
                        batch_idx[0] = 0
                        batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                        while T.gpu_thread_invariant(batch_idx[0] < batch_size):
                            advance_tile_batch(tile_id, batch_idx, batch_tiles, batch_rows, q_indptr, batch_size)

                            if T.gpu_thread_invariant(batch_idx[0] < batch_size):
                                b_idx: T.let[T.int32] = batch_idx[0]
                                LH_start: T.let[T.int32] = tile_id[0] * tile_x
                                q_indptr_val: T.let[T.int32] = q_indptr[b_idx]

                                cur_page_indptr_begin: T.let[T.int32] = page_indptr[b_idx]
                                cur_page_indptr_end: T.let[T.int32] = page_indptr[b_idx + 1]
                                kv_chunk_len[0] = T.if_then_else(
                                    cur_page_indptr_begin != cur_page_indptr_end,
                                    _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, b_idx, length_info, sliding_window),
                                    0
                                )
                                T.gpu_storage_sync("shared")

                                init_states(m_smem, d_smem, O_local, ty, tx)

                                # Load Q from gmem to smem
                                for li, lj in T.grid(tile_x, tile_y):
                                    with Ts.sblock("Q_load"):
                                        i, j = Ts.axis.remap("SS", [li, lj])
                                        Ts.reads()
                                        Ts.writes()
                                        cur_L: T.let[T.int32] = q_indptr_val + (LH_start + i) // group_size
                                        cur_H_qo: T.let[T.int32] = (LH_start + i) % group_size
                                        if cur_L < q_indptr[b_idx + 1]:
                                            Q_smem[i, j] = q[cur_L, cur_H_qo, j]
                                        else:
                                            Q_smem[i, j] = 0.0
                                T.gpu_storage_sync("shared")

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                    L_kv_start: T.let[T.int32] = iterator * tile_z
                                    for lz, ly in T.grid(tile_z, tile_y):
                                        with Ts.sblock("KV_load"):
                                            i, j = Ts.axis.remap("SS", [lz, ly])
                                            Ts.reads()
                                            Ts.writes()
                                            cur_L: T.let[T.int32] = L_kv_start + i
                                            if cur_L < kv_chunk_len[0]:
                                                seq_offset: T.let[T.int32()] = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                page_no: T.let[T.int32()] = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, page_size)]  # type: ignore
                                                page_offset: T.let[T.int32()] = T.floormod(seq_offset, page_size)  # type: ignore
                                                KV_smem[i, j] = pages[page_no, page_offset, j]
                                            else:
                                                KV_smem[i, j] = 0.0
                                    T.gpu_storage_sync("shared")

                                    # MLA shares the same buffer for K and V (V = KV_smem[:, :d_latent])
                                    compute_s_gemm(Q_smem, KV_smem, S_local, S_smem, sm_scale)

                                    softmax_update_causal(
                                        S_smem, m_smem, d_smem, m_prev_smem,
                                        m_new, m_prev, d_new,
                                        ty, tx, LH_start, L_kv_start,
                                        causal, kv_chunk_len[0], q_indptr[b_idx + 1] - q_indptr[b_idx],
                                        0,
                                    )

                                    compute_o_gemm(S_smem, KV_smem, O_local, m_prev_smem, m_smem)

                                # MLA has no blockIdx.y binding; pass by=0 so the
                                # by*group_size term in the shared epilogue drops.
                                paged_store_output_lse(
                                    output, lse, O_local, m_smem, d_smem,
                                    q_indptr, b_idx, 0, LH_start,
                                )

                                # move to next tile
                                tile_id[0] += NUM_BLKS
    # pylint: enable=too-many-branches
    sch = tvm.s_tir.Schedule(batch_prefill_paged_kv_mla)
    sch = _schedule_prefill_kernel(sch, LOAD_VEC, bdx, num_warps, tile_x, d_latent, tile_z, False, True)
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)
