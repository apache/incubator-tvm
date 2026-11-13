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

"""Single-token decode attention kernels and attention-state merge helpers.

Contents:
- ``_attention_decode_cpu`` / ``_attention_decode`` — paged-KV decode (one Q token
  per sequence), CPU scalar and GPU allreduce variants.
- ``_merge_state_inplace_cpu`` / ``_merge_state_inplace`` — combine two
  log-sum-exp attention outputs in place. Used by multi-stage decoding and by
  the distributed KV-transfer path.
"""

# pylint: disable=too-many-statements,too-many-arguments,invalid-name,line-too-long
import math
from typing import Any

from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

from ._kernel_common import (
    _get_kv_chunk_len,
    _get_seq_offset,
    _length_info_buffer,
    _rope,
    _var,
    _var_cpu,
    check_thread_limits,
    get_max_num_threads_per_block,
)


def _attention_decode_cpu(num_kv_heads, num_qo_heads, head_dim, qkv_dtype, sliding_window: bool, rope_scaling: dict[str, Any], page_size: int = 16):
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim
    group_size = num_qo_heads // num_kv_heads

    global_symbol = "batch_decode_paged_kv_cpu"
    if sliding_window:
        global_symbol += "_sliding_window"

    B = T.dynamic("B", "int32")
    nnz_pages = T.dynamic("nnz_pages", "int32")
    max_num_pages = T.dynamic("max_num_pages", "int32")
    page_indptr_elem_offset = T.dynamic("page_indptr_elem_offset", "int32")
    page_values_elem_offset = T.dynamic("page_values_elem_offset", "int32")
    k_rope_pos_offset_elem_offset = T.dynamic("k_rope_pos_offset_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    length_info_elem_offset = T.dynamic("length_info_elem_offset", "int32")
    @Ts.function
    def batch_decode_paged_kv(
        Q: T.Tensor((B, H_qo, D), qkv_dtype),
        pages: T.Tensor((max_num_pages, 2, H_kv, page_size, D), qkv_dtype),
        page_table_indptr: T.Tensor((B + 1,), 'int32', layout=T.TileLayout(T.S[(B + 1,)] + (page_indptr_elem_offset))),
        page_table_values: T.Tensor((nnz_pages,), 'int32', layout=T.TileLayout(T.S[(nnz_pages,)] + (page_values_elem_offset))),
        length_info: _length_info_buffer(B, sliding_window, length_info_elem_offset),  # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset: T.Tensor((B,), 'int32', layout=T.TileLayout(T.S[(B,)] + (k_rope_pos_offset_elem_offset))),
        q_rope_position: T.Tensor((B,), 'int32', layout=T.TileLayout(T.S[(B,)] + (q_rope_position_elem_offset))),
        output: T.Tensor((B, H_qo, D), qkv_dtype),
        lse: T.Tensor((B, H_qo), 'float32'),
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"tirx.is_scheduled": True, "global_symbol": global_symbol})

          # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".

        for b in T.serial(B):
            with Ts.sblock("attn"):
                O_local = Ts.sblock_alloc_tensor((D,), "float32")
                Q_local = Ts.sblock_alloc_tensor((D,), "float32")
                K_local = Ts.sblock_alloc_tensor((D,), "float32")
                V_local = Ts.sblock_alloc_tensor((D,), "float32")

                kv_chunk_len = Ts.sblock_alloc_tensor((1,), "int32")

                m_val = Ts.sblock_alloc_tensor((1,), "float32")
                new_m = Ts.sblock_alloc_tensor((1,), "float32")
                d_val = Ts.sblock_alloc_tensor((1,), "float32")
                S_val = Ts.sblock_alloc_tensor((1,), "float32")
                scale_O = Ts.sblock_alloc_tensor((1,), "float32")
                factor = Ts.sblock_alloc_tensor((1,), "float32")

                cur_page_indptr_begin: T.let[T.int32] = page_table_indptr[b]
                cur_page_indptr_end: T.let[T.int32] = page_table_indptr[b + 1]

                kv_chunk_len[0] = T.if_then_else(
                    cur_page_indptr_begin != cur_page_indptr_end,
                    _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, b, length_info, sliding_window),
                    0,
                )

                for h_qo in T.serial(H_qo):
                    m_val[0] = -5e4
                    d_val[0] = 1.0

                    for d in T.serial(D):
                        O_local[d] = 0.0

                    for d in T.serial(D):
                        Q_local[d] = T.if_then_else(
                            rotary_mode == 1,
                            _rope(Q, q_rope_position[b], head_dim, rope_theta, rope_scale, (b, h_qo, d), qkv_dtype, rope_scaling),
                            Q[b, h_qo, d],
                        )

                    for row_idx in T.serial(kv_chunk_len[0]):
                        seq_offset: T.let[T.int32()] = _get_seq_offset(row_idx, b, length_info, sliding_window)
                        page_no: T.let[T.int32()] = page_table_values[cur_page_indptr_begin + (seq_offset // page_size)]
                        page_offset: T.let[T.int32()] = seq_offset % page_size

                        for d in T.serial(D):
                            K_local[d] = T.if_then_else(
                                rotary_mode == 1,
                                _rope(pages, k_rope_pos_offset[b] + row_idx, head_dim, rope_theta, rope_scale, (page_no, 0, h_qo // group_size, page_offset, d), qkv_dtype, rope_scaling),
                                pages[page_no, 0, h_qo // group_size, page_offset, d],
                            )
                        S_val[0] = 0.0
                        for d in T.serial(D):
                            S_val[0] += Q_local[d] * K_local[d]
                        S_val[0] *= sm_scale * math.log2(math.exp(1))

                        new_m[0] = T.max(m_val[0], S_val[0])
                        d_val[0] = (d_val[0] * T.exp2(m_val[0] - new_m[0])) + T.exp2(S_val[0] - new_m[0])

                        scale_O[0] = T.exp2(m_val[0] - new_m[0])

                        for d in T.serial(D):
                            O_local[d] = O_local[d] * scale_O[0]

                        m_val[0] = new_m[0]
                        for d in T.serial(D):
                            V_local[d] = pages[page_no, 1, h_qo // group_size, page_offset, d]

                        factor[0] = T.exp2(S_val[0] - m_val[0])
                        for d in T.serial(D):
                            O_local[d] = O_local[d] + V_local[d] * factor[0]
                    for d in T.serial(D):
                        O_local[d] = O_local[d] / d_val[0]
                        output[b, h_qo, d] = O_local[d]
                    lse[b, h_qo] = m_val[0] + T.log2(d_val[0])

    return batch_decode_paged_kv

def _attention_decode(num_kv_heads, num_qo_heads, head_dim, qkv_dtype, sliding_window: bool, rope_scaling: dict[str, Any], target: Target, page_size: int = 16):
    qkv_dtype_bytes = 2
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim

    THREAD_LIMIT = 512
    TILE_SIZE_PER_BDX = 2
    if target.kind.name == "opencl" and (("android" in str(target.host)) or ("adreno" in str(target.attrs))):
        # Keeping lower thread limit for this kernel on adreno target
        # to avoid register spill
        THREAD_LIMIT = 256
        TILE_SIZE_PER_BDX = 1
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    thread_limit = min(max_num_threads_per_block, THREAD_LIMIT)

    GROUP_SIZE = H_qo // H_kv
    VEC_SIZE = min(max(8 // qkv_dtype_bytes, D // 32), 4)
    bdx = D // VEC_SIZE
    bdy = GROUP_SIZE
    while bdx * bdy > thread_limit and bdy > 1:
        bdy //= 2
    gdz = GROUP_SIZE // bdy
    threads_per_CTA = max(thread_limit, bdx * bdy)
    bdz = threads_per_CTA // (bdx * bdy)
    tile_size_per_bdx = TILE_SIZE_PER_BDX if GROUP_SIZE == 1 else 1
    check_thread_limits(target, bdx=bdx, bdy=bdy, bdz=bdz, gdz=1)

    global_symbol = "batch_decode_paged_kv"
    if sliding_window:
        global_symbol += "_sliding_window"

    # pylint: disable=too-many-branches
    B = T.dynamic("B", "int32")
    nnz_pages = T.dynamic("nnz_pages", "int32")
    max_num_pages = T.dynamic("max_num_pages", "int32")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    page_indptr_elem_offset = T.dynamic("page_indptr_elem_offset", "int32")
    page_values_elem_offset = T.dynamic("page_values_elem_offset", "int32")
    k_rope_pos_offset_elem_offset = T.dynamic("k_rope_pos_offset_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    length_info_elem_offset = T.dynamic("length_info_elem_offset", "int32")
    @Ts.function
    def batch_decode_paged_kv(
        Q: T.Tensor((B, H_qo, D), qkv_dtype),
        pages: T.Tensor((max_num_pages, 2, H_kv, page_size, D), qkv_dtype, layout=T.TileLayout(T.S[(max_num_pages, 2, H_kv, page_size, D)] + (pages_elem_offset))),
        page_table_indptr: T.Tensor((B + 1,), 'int32', layout=T.TileLayout(T.S[(B + 1,)] + (page_indptr_elem_offset))),
        page_table_values: T.Tensor((nnz_pages,), 'int32', layout=T.TileLayout(T.S[(nnz_pages,)] + (page_values_elem_offset))),
        length_info: _length_info_buffer(B, sliding_window, length_info_elem_offset), # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset: T.Tensor((B,), 'int32', layout=T.TileLayout(T.S[(B,)] + (k_rope_pos_offset_elem_offset))),
        q_rope_position: T.Tensor((B,), 'int32', layout=T.TileLayout(T.S[(B,)] + (q_rope_position_elem_offset))),
        output: T.Tensor((B, H_qo, D), qkv_dtype),
        lse: T.Tensor((B, H_qo), 'float32'),
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"tirx.is_scheduled": True, "global_symbol": global_symbol})

          # pylint: disable=unused-variable

        for bx in T.parallel(B, annotations={"thread_binding": "blockIdx.x"}):
            for fused_by_bz in T.parallel(H_kv * gdz, annotations={"thread_binding": "blockIdx.y"}):
                for ty in T.parallel(bdy, annotations={"thread_binding": "threadIdx.y"}):
                    for tx in T.parallel(bdx, annotations={"thread_binding": "threadIdx.x"}):
                        for tz in T.parallel(bdz, annotations={"thread_binding": "threadIdx.z"}):
                            with Ts.sblock("attn"):
                                Q_local = Ts.sblock_alloc_tensor((VEC_SIZE,), qkv_dtype, scope="local")
                                kv_chunk_len = Ts.sblock_alloc_tensor((1,), "int32", scope="local")
                                K_smem = Ts.sblock_alloc_tensor((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                V_smem = Ts.sblock_alloc_tensor((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                O_allreduce = Ts.sblock_alloc_tensor((bdz, bdy, D), "float32", scope="shared")
                                md_allreduce = Ts.sblock_alloc_tensor((bdz, bdy, 2), "float32", scope="shared")
                                S_reduce_local = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                t0 = Ts.sblock_alloc_tensor((1,), "float32", scope="local")

                                S_local = Ts.sblock_alloc_tensor((bdy * tile_size_per_bdx), "float32", scope="local")
                                QK_local = Ts.sblock_alloc_tensor((VEC_SIZE,), "float32", scope="local")
                                V_local = Ts.sblock_alloc_tensor((VEC_SIZE,), qkv_dtype, scope="local")
                                m_prev = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                d_prev = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                other_m = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                other_d = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                exp_mprev = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                exp_otherm = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                other_o = Ts.sblock_alloc_tensor((VEC_SIZE,), "float32", scope="local")
                                st_m = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                st_d = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                                O_local = Ts.sblock_alloc_tensor((VEC_SIZE,), "float32", scope="local")

                                by: T.let[T.int32] = fused_by_bz % H_kv
                                bz: T.let[T.int32] = fused_by_bz // H_kv
                                batch_idx: T.let[T.int32] = bx
                                cur_page_indptr_begin: T.let[T.int32] = page_table_indptr[batch_idx]
                                cur_page_indptr_end: T.let[T.int32] = page_table_indptr[batch_idx + 1]
                                kv_chunk_len[0] = T.if_then_else(
                                    cur_page_indptr_begin != cur_page_indptr_end,
                                    _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, batch_idx, length_info, sliding_window),
                                    0
                                )

                                # init states
                                st_m[0] = -5e4
                                st_d[0] = 1.0
                                for vec in T.vectorized(VEC_SIZE):
                                    O_local[vec] = 0.0

                                # load q
                                for vec in T.vectorized(VEC_SIZE):
                                    Q_local[vec] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(Q, q_rope_position[batch_idx], head_dim, rope_theta, rope_scale, (bx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec), qkv_dtype, rope_scaling),
                                        Q[bx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec]
                                    )

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_size_per_bdx * bdy * bdz)):
                                    tile_start_s: T.let[T.int32()] = (tz * bdy + ty) * tile_size_per_bdx  # type: ignore
                                    tile_start_g: T.let[T.int32()] = ((iterator * bdz + tz) * bdy + ty) * tile_size_per_bdx  # type: ignore
                                    # load KV from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        with Ts.sblock("KV_load"):
                                            Ts.reads()
                                            Ts.writes()
                                            row_g: T.let[T.int32()] = tile_start_g + j  # type: ignore
                                            if row_g < kv_chunk_len[0]:
                                                seq_offset: T.let[T.int32()] = _get_seq_offset(row_g, batch_idx, length_info, sliding_window)  # type: ignore
                                                page_no: T.let[T.int32()] = page_table_values[cur_page_indptr_begin + T.floordiv(seq_offset, page_size)]  # type: ignore
                                                page_offset: T.let[T.int32()] = T.floormod(seq_offset, page_size)  # type: ignore
                                                for vec in T.vectorized(VEC_SIZE):
                                                    K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(pages, k_rope_pos_offset[batch_idx] + row_g, head_dim, rope_theta, rope_scale, (page_no, 0, by, page_offset, tx * VEC_SIZE + vec), qkv_dtype, rope_scaling),
                                                        pages[page_no, 0, by, page_offset, tx * VEC_SIZE + vec]
                                                    )
                                                    V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = pages[page_no, 1, by, page_offset, tx * VEC_SIZE + vec]
                                            else:
                                                for vec in T.vectorized(VEC_SIZE):
                                                    K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                                    V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.gpu_storage_sync("shared")
                                    # compute QK
                                    m_prev[0] = st_m[0]
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        # compute S = Q * K * sm_scale
                                        for vec in T.vectorized(VEC_SIZE):
                                            QK_local[vec] = T.cast(Q_local[vec], ty="float32") * T.cast(K_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec], ty="float32") * sm_scale * math.log2(math.exp(1))
                                        S_reduce_local[0] = 0
                                        for vec in T.unroll(VEC_SIZE):
                                            S_reduce_local[0] += QK_local[vec]

                                        with Ts.sblock("block_cross_thread"):
                                            Ts.reads(S_reduce_local[0])
                                            Ts.writes(t0[0])
                                            T.gpu_thread_allreduce(T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)), (T.float32(0),), (S_reduce_local[0],), True, (t0[0],), (tx,))

                                        S_local[j] = -5e4
                                        if (iterator * bdz + tz) * bdy * tile_size_per_bdx + j < kv_chunk_len[0]:
                                            S_local[j] = t0[0]
                                        # update st_m
                                        st_m[0] = T.max(st_m[0], S_local[j])

                                    # update st_d, st_O
                                    o_scale: T.let[T.float32] = T.exp2(m_prev[0] - st_m[0])
                                    st_d[0] *= o_scale
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        S_local[j] = T.exp2(S_local[j] - st_m[0])
                                        st_d[0] += S_local[j]
                                    for j in T.vectorized(VEC_SIZE):
                                        O_local[j] *= o_scale

                                    # load V from shared memory to local memory
                                    # compute O
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        for vec in T.vectorized(VEC_SIZE):
                                            V_local[vec] = V_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec]
                                        for vec in T.vectorized(VEC_SIZE):
                                            O_local[vec] += T.cast(V_local[vec], ty="float32") * S_local[j]

                                if bdz > 1:
                                    # allreduce over bdz
                                    for vec in T.vectorized(VEC_SIZE):
                                        O_allreduce[tz, ty, tx * VEC_SIZE + vec] = O_local[vec]
                                    md_allreduce[tz, ty, 0] = st_m[0]
                                    md_allreduce[tz, ty, 1] = st_d[0]
                                    T.gpu_storage_sync("shared")

                                    st_m[0] = -5e4
                                    st_d[0] = 1.0
                                    for vec in T.vectorized(VEC_SIZE):
                                        O_local[vec] = 0.0

                                    for j in T.serial(bdz):
                                        m_prev[0] = st_m[0]
                                        d_prev[0] = st_d[0]
                                        other_m[0] = md_allreduce[j, ty, 0]
                                        other_d[0] = md_allreduce[j, ty, 1]
                                        for vec in T.vectorized(VEC_SIZE):
                                            other_o[vec] = O_allreduce[j, ty, tx * VEC_SIZE + vec]
                                        st_m[0] = T.max(st_m[0], other_m[0])
                                        st_d[0] = d_prev[0] * T.exp2(m_prev[0] - st_m[0]) + other_d[0] * T.exp2(other_m[0] - st_m[0])
                                        exp_mprev[0] = T.exp2(m_prev[0] - st_m[0])
                                        exp_otherm[0] = T.exp2(other_m[0] - st_m[0])
                                        for vec in T.vectorized(VEC_SIZE):
                                            O_local[vec] = O_local[vec] * exp_mprev[0] + other_o[vec] * exp_otherm[0]

                                # normalize O
                                for vec in T.vectorized(VEC_SIZE):
                                    O_local[vec] /= st_d[0]

                                # store O to global memory
                                for vec in T.vectorized(VEC_SIZE):
                                    output[batch_idx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec] = O_local[vec]

                                # store lse to global memory
                                lse[batch_idx, by * GROUP_SIZE + bz * bdy + ty] = st_m[0] + T.log2(st_d[0])
    # pylint: enable=too-many-branches
    return batch_decode_paged_kv

def _merge_state_inplace_cpu(v_dtype):
    N = T.dynamic("N", "int32")
    H = T.dynamic("H", "int32")
    D = T.dynamic("D", "int32")
    @Ts.function
    def merge_state_inplace_cpu(
        V: T.Tensor((N, H, D), v_dtype),
        S: T.Tensor((N, H), 'float32'),
        V_other: T.Tensor((N, H, D), v_dtype),
        S_other: T.Tensor((N, H), 'float32'),
    ):
        T.func_attr({"tirx.is_scheduled": True})

        for n in T.serial(N):
            for h in T.serial(H):
                with Ts.sblock("merge"):
                    s_val = _var_cpu("float32")
                    s_other_val = _var_cpu("float32")
                    s_max = _var_cpu("float32")
                    scale = _var_cpu("float32")
                    other_scale = _var_cpu("float32")

                    s_val[0] = S[n, h]
                    s_other_val[0] = S_other[n, h]
                    s_max[0] = T.max(s_val[0], s_other_val[0])
                    s_val[0] = T.exp2(s_val[0] - s_max[0])
                    s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                    scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                    other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])
                    for d in T.serial(D):
                        V[n, h, d] = V[n, h, d] * scale[0] + V_other[n, h, d] * other_scale[0]
                    S[n, h] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]

    return merge_state_inplace_cpu

def _merge_state_inplace(num_heads, head_dim, v_dtype, target: Target, global_symbol: str | None = None):
    v_dtype_bytes = 2
    VEC_SIZE = min(max(8 // v_dtype_bytes, head_dim // 32), 4)
    bdx = head_dim // VEC_SIZE
    bdy = num_heads
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    while bdx * bdy > max_num_threads_per_block and bdy > 1:
        bdy //= 2
    gdy = num_heads // bdy
    check_thread_limits(target, bdx=bdx, bdy=bdy, bdz=1, gdz=1)

    N = T.dynamic("N", "int32")
    H = T.dynamic("H", "int32")
    D = T.dynamic("D", "int32")
    @Ts.function
    def merge_state_inplace(
        V: T.Tensor((N, H, D), v_dtype),
        S: T.Tensor((N, H), 'float32'),
        V_other: T.Tensor((N, H, D), v_dtype),
        S_other: T.Tensor((N, H), 'float32'),
    ):
        T.func_attr({"tirx.is_scheduled": True})

        for bx in T.parallel(N, annotations={"thread_binding": "blockIdx.x"}):
            for by in T.parallel(gdy, annotations={"thread_binding": "blockIdx.y"}):
                for ty in T.parallel(bdy, annotations={"thread_binding": "threadIdx.y"}):
                    for tx in T.parallel(bdx, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("merge"):
                            s_val = _var("float32")
                            s_other_val = _var("float32")
                            s_max = _var("float32")
                            scale = _var("float32")
                            other_scale = _var("float32")

                            v_vec = Ts.sblock_alloc_tensor((VEC_SIZE,), v_dtype, scope="local")
                            v_other_vec = Ts.sblock_alloc_tensor((VEC_SIZE,), v_dtype, scope="local")

                            s_val[0] = S[bx, ty + by * bdy]
                            s_other_val[0] = S_other[bx, ty + by * bdy]
                            s_max[0] = T.max(s_val[0], s_other_val[0])
                            s_val[0] = T.exp2(s_val[0] - s_max[0])
                            s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                            scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                            other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])

                            # load v
                            for vec in T.vectorized(VEC_SIZE):
                                v_vec[vec] = V[bx, ty + by * bdy, tx * VEC_SIZE + vec]
                            # load v_other
                            for vec in T.vectorized(VEC_SIZE):
                                v_other_vec[vec] = V_other[bx, ty + by * bdy, tx * VEC_SIZE + vec]

                            # merge
                            for vec in T.serial(VEC_SIZE):
                                v_vec[vec] = v_vec[vec] * scale[0] + v_other_vec[vec] * other_scale[0]

                            # store v
                            for vec in T.vectorized(VEC_SIZE):
                                V[bx, ty + by * bdy, tx * VEC_SIZE + vec] = v_vec[vec]

                            # store s
                            # Every threadIdx.x thread of this block reads the same S element
                            # above, so none may overwrite it until all of them have read it.
                            # Without this barrier the threads only stay in step while bdx fits
                            # in one warp, and a head_dim above 128 makes bdx wider than that.
                            T.gpu_storage_sync("shared")
                            if tx == 0:
                                S[bx, ty + by * bdy] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]

    func = merge_state_inplace
    if global_symbol:
        func = func.with_attr("global_symbol", global_symbol)
    return func
