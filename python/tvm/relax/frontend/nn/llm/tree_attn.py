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
# pylint: disable=invalid-name
# ruff: noqa: E501

"""Operators for tree attention."""

import math
from typing import Any

from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

# The helpers below are shared with the main KV-cache kernels. They live in
# ``_kernel_common`` so that ``kv_cache.py``, this file, and the split kernel
# modules can all pull from a single source of truth.
from ._kernel_common import (
    _alloc_mha_qkvo_buffers,
    _alloc_softmax_state_buffers,
    _alloc_tile_walk_state,
    _get_kv_chunk_len,
    _get_prefill_kernel_config,
    _get_seq_offset,
    _length_info_buffer,
    _rope,
)

# mypy: disable-error-code="attr-defined,valid-type,no-redef"
# pylint: disable=too-many-statements,too-many-locals,too-many-arguments


def _check_tree_order(tree_order_indptr, tree_order, batch, row, col, kv_len, qo_len):
    tree_order_len = tree_order_indptr[batch + 1] - tree_order_indptr[batch]

    tree_start = kv_len - tree_order_len
    child_idx_in_tree = row + tree_order_len - qo_len
    parent_idx_in_tree = col - tree_start
    return tirx.all(
        col < kv_len,
        tirx.any(
            col < tree_start,
            tirx.all(
                tree_order[tree_order_indptr[batch] + child_idx_in_tree, 0]
                >= tree_order[tree_order_indptr[batch] + parent_idx_in_tree, 0],
                tree_order[tree_order_indptr[batch] + child_idx_in_tree, 0]
                < tree_order[tree_order_indptr[batch] + parent_idx_in_tree, 1],
            ),
        ),
    )


def tree_attn_cpu(h_kv, h_q, d, dtype, rope_scaling: dict[str, Any]):
    """Generate tree attention kernel for batched tree attention.

    Parameters
    ----------
    h_kv : int
        Number of heads for key and value.
    h_q : int
        Number of heads for query.
    d : int
        Hidden dimension.
    dtype : str
        Data type.
    target : Target
        The target device.

    Returns
    -------
    mod : tvm_tirx.IRModule
        The generated IR module.
    """
    group_size = h_q // h_kv

    # fmt: off
    qo_len = T.dynamic("qo_len", "int32")
    kv_len = T.dynamic("kv_len", "int32")
    q_indptr_elem_offset = T.dynamic("q_indptr_elem_offset", "int32")
    kv_indptr_elem_offset = T.dynamic("kv_indptr_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    mn_indptr_elem_offset = T.dynamic("mn_indptr_elem_offset", "int32")
    mask_elem_offset = T.dynamic("mask_elem_offset", "int32")
    tree_size = T.dynamic("tree_size", "int32")
    batch_size_plus_1 = T.dynamic("batch_size_plus_1", "int32")
    @Ts.function
    def batch_tree_attn(  # pylint: disable=too-many-branches,line-too-long
        q: T.Tensor((qo_len, h_q, d), dtype),  # [total_len, h_q, d]
        q_indptr: T.Tensor((batch_size_plus_1,), 'int32', layout=T.TileLayout(T.S[(batch_size_plus_1,)] + (q_indptr_elem_offset))),  # [batch_size + 1]
        k: T.Tensor((kv_len, h_kv, d), dtype),  # [total_len, h_kv, d]
        v: T.Tensor((kv_len, h_kv, d), dtype),  # [total_len, h_kv, d]
        kv_indptr: T.Tensor((batch_size_plus_1,), 'int32', layout=T.TileLayout(T.S[(batch_size_plus_1,)] + (kv_indptr_elem_offset))),  # [batch_size + 1], kv_indptr should be the same as q_indptr in this case
        q_rope_position: T.Tensor((qo_len,), 'int32', layout=T.TileLayout(T.S[(qo_len,)] + (q_rope_position_elem_offset))),  # [total_q_len]
        mn_indptr: T.Tensor((batch_size_plus_1,), 'int32', layout=T.TileLayout(T.S[(batch_size_plus_1,)] + (mn_indptr_elem_offset))),  # [batch_size + 1]
        mask: T.Tensor((tree_size, 2), 'int32', layout=T.TileLayout(T.S[(tree_size, 2)] + (mask_elem_offset))),  # [mn_indptr[batch_size]]
        output: T.Tensor((qo_len, h_q, d), dtype),  # [total_len, h_q, d]
        lse: T.Tensor((qo_len, h_q), 'float32'),  # [total_len, h_q]
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):

          # pylint: disable=unused-variable

        for b in T.serial(batch_size_plus_1 - 1):
            with Ts.sblock("attn"):

                softmax_sum = Ts.sblock_alloc_tensor([h_q], "float32")
                m_prev = Ts.sblock_alloc_tensor([h_q], "float32")
                m_new = Ts.sblock_alloc_tensor([h_q], "float32")
                d_prev = Ts.sblock_alloc_tensor([h_q], "float32")
                d_new = Ts.sblock_alloc_tensor([h_q], "float32")
                p_sum = Ts.sblock_alloc_tensor([d], "float32")

                max_score = Ts.sblock_alloc_tensor([h_q], "float32")
                attention_scores = Ts.sblock_alloc_tensor([kv_len, h_q], "float32")
                exp_scores = Ts.sblock_alloc_tensor([kv_len, h_q], "float32")
                attention_score = Ts.sblock_alloc_tensor(
                    [
                        1,
                    ],
                    "float32",
                )
                query_val = Ts.sblock_alloc_tensor(
                    [
                        1,
                    ],
                    "float32",
                )
                key_val = Ts.sblock_alloc_tensor(
                    [
                        1,
                    ],
                    "float32",
                )
                result = Ts.sblock_alloc_tensor(
                    [
                        1,
                    ],
                    "float32",
                )

                for q_idx in T.serial(q_indptr[b + 1] - q_indptr[b]):
                    for i in T.serial(h_q):
                        max_score[i] = -5e4
                        m_prev[i] = -5e4
                        d_prev[i] = 1.0

                    for k_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                        for h in T.serial(h_q):
                            h_kv_idx: T.let[T.int32] = h // group_size

                            if _check_tree_order(
                                row=q_idx,
                                col=k_idx,
                                batch=b,
                                tree_order=mask,
                                tree_order_indptr=mn_indptr,
                                kv_len=kv_indptr[b + 1] - kv_indptr[b],
                                qo_len=q_indptr[b + 1] - q_indptr[b],
                            ):
                                result[0] = 0.0
                                for d_idx in T.serial(d):
                                    query_val[0] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(
                                            q,
                                            q_rope_position[q_indptr[b] + q_idx],
                                            d,
                                            rope_theta,
                                            rope_scale,
                                            (q_indptr[b] + q_idx, h, d_idx),
                                            dtype,
                                            rope_scaling,
                                        ),
                                        q[q_indptr[b] + q_idx, h, d_idx],
                                    )

                                    key_val[0] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(
                                            k,
                                            q_rope_position[kv_indptr[b] + k_idx],
                                            d,
                                            rope_theta,
                                            rope_scale,
                                            (kv_indptr[b] + k_idx, h_kv_idx, d_idx),
                                            dtype,
                                            rope_scaling,
                                        ),
                                        k[kv_indptr[b] + k_idx, h_kv_idx, d_idx],
                                    )

                                    result[0] += query_val[0] * key_val[0]
                                attention_score[0] = (
                                    result[0] * math.log2(math.exp(1)) * sm_scale
                                )
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
                        for i in T.serial(d):
                            p_sum[i] = 0.0
                        for v_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                            weight: T.let[T.float32] = exp_scores[v_idx, h] / d_new[h]
                            for i in T.serial(d):
                                p_sum[i] += v[kv_indptr[b] + v_idx, h_kv_idx, i] * weight
                        for i in T.serial(d):
                            output[q_indptr[b] + q_idx, h, i] = p_sum[i]
                        lse[q_indptr[b] + q_idx, h] = m_new[h] + T.log2(d_new[h])

    # fmt: on
    # pylint: enable=line-too-long,too-many-branches
    return batch_tree_attn


def tree_attn(h_kv, h_q, d, dtype, rope_scaling: dict[str, Any], target: Target):  # pylint: disable=unused-argument
    """Generate tree attention kernel for batched tree attention.

    Parameters
    ----------
    h_kv : int
        Number of heads for key and value.
    h_q : int
        Number of heads for query.
    d : int
        Hidden dimension.
    dtype : str
        Data type.
    target : Target
        The target device.

    Returns
    -------
    mod : tvm_tirx.IRModule
        The generated IR module.
    """
    # pylint: disable=invalid-name,line-too-long
    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = (
        _get_prefill_kernel_config(h_kv, h_q, d, dtype, target)
    )

    # fmt: off
    qo_len = T.dynamic("qo_len", "int32")
    kv_len = T.dynamic("kv_len", "int32")
    q_indptr_elem_offset = T.dynamic("q_indptr_elem_offset", "int32")
    kv_indptr_elem_offset = T.dynamic("kv_indptr_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    mn_indptr_elem_offset = T.dynamic("mn_indptr_elem_offset", "int32")
    mask_elem_offset = T.dynamic("mask_elem_offset", "int32")
    tree_size = T.dynamic("tree_size", "int32")
    batch_size_plus_1 = T.dynamic("batch_size_plus_1", "int32")
    @Ts.function
    def batch_tree_attn(  # pylint: disable=too-many-branches
        q: T.Tensor((qo_len, h_q, d), dtype), # [total_len, h_q, d]
        q_indptr: T.Tensor((batch_size_plus_1,), 'int32', layout=T.TileLayout(T.S[(batch_size_plus_1,)] + (q_indptr_elem_offset))), # [batch_size + 1]
        k: T.Tensor((kv_len, h_kv, d), dtype), # [total_len, h_kv, d]
        v: T.Tensor((kv_len, h_kv, d), dtype), # [total_len, h_kv, d]
        kv_indptr: T.Tensor((batch_size_plus_1,), 'int32', layout=T.TileLayout(T.S[(batch_size_plus_1,)] + (kv_indptr_elem_offset))), # [batch_size + 1], kv_indptr should be the same as q_indptr in this case
        q_rope_position: T.Tensor((qo_len,), 'int32', layout=T.TileLayout(T.S[(qo_len,)] + (q_rope_position_elem_offset))), # [total_q_len]
        mn_indptr: T.Tensor((batch_size_plus_1,), 'int32', layout=T.TileLayout(T.S[(batch_size_plus_1,)] + (mn_indptr_elem_offset))), # [batch_size + 1]
        mask: T.Tensor((tree_size, 2), 'int32', layout=T.TileLayout(T.S[(tree_size, 2)] + (mask_elem_offset))), # [mn_indptr[batch_size]]
        output: T.Tensor((qo_len, h_q, d), dtype), # [total_len, h_q, d]
        lse: T.Tensor((qo_len, h_q), 'float32'), # [total_len, h_q]
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
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
                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d, d, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)

                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.gpu_thread_invariant(batch_idx[0] < batch_size_plus_1 - 1):
                                # advance to next tile
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size_plus_1 - 1:
                                    tile_id[0] -= batch_tiles[0]
                                    batch_idx[0] += 1
                                    if batch_idx[0] < batch_size_plus_1 - 1:
                                        b_idx: T.let[T.int32] = batch_idx[0]
                                        batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                if T.gpu_thread_invariant(batch_idx[0] < batch_size_plus_1 - 1):
                                    b_idx: T.let[T.int32()] = batch_idx[0]
                                    LH_start: T.let[T.int32()] = tile_id[0] * tile_x
                                    q_indptr_val: T.let[T.int32] = q_indptr[b_idx]

                                    kv_chunk_len[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                    T.gpu_storage_sync("shared")

                                    # init states
                                    for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                        row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
                                        if row < tile_x:
                                            m_smem[row] = -5e4
                                            d_smem[row] = 1.0

                                    for li, lj in T.grid(tile_x, tile_y):
                                        with Ts.sblock("O_init"):
                                            i, j = Ts.axis.remap("SS", [li, lj])
                                            O_local[i, j] = 0.0
                                    T.gpu_storage_sync("shared")

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
                                        L_kv_base: T.let[T.int32] = kv_indptr[b_idx]
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with Ts.sblock("KV_load"):
                                                i, j = Ts.axis.remap("SS", [lz, ly])
                                                Ts.reads()
                                                Ts.writes()
                                                cur_L: T.let[T.int32] = L_kv_base + L_kv_start + i
                                                if L_kv_start + i < kv_chunk_len[0]:
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(k, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, by, j), dtype, rope_scaling),
                                                        k[cur_L, by, j]
                                                    )
                                                    V_smem[i, j] = v[cur_L, by, j]
                                                else:
                                                    K_smem[i, j] = 0.0
                                                    V_smem[i, j] = 0.0
                                        T.gpu_storage_sync("shared")

                                        # Compute S
                                        with Ts.sblock():
                                            for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                                                with Ts.sblock("S_gemm"):
                                                    i, j, k_axis = Ts.axis.remap("SSR", [li, lj, lk])
                                                    with Ts.init():
                                                        S_local[i, j] = 0.0
                                                    S_local[i, j] += T.cast(Q_smem[i, k_axis], ty="float32") * T.cast(K_smem[j, k_axis], ty="float32") * sm_scale * math.log2(math.exp(1))
                                        T.gpu_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with Ts.sblock("S_store"):
                                                i, j = Ts.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.gpu_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with Ts.sblock("update1"):
                                                    m_prev[i] = m_smem[row]
                                                    m_new[i] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    row_: T.let[T.int32] = (LH_start + row) // group_size
                                                    for j in T.serial(tile_z):
                                                        if _check_tree_order(
                                                            row=row_,
                                                            col=L_kv_start + j,
                                                            batch=b_idx,
                                                            tree_order=mask,
                                                            tree_order_indptr=mn_indptr,
                                                            qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx],
                                                            kv_len=kv_chunk_len[0]):
                                                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
                                            with Ts.sblock("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        row_: T.let[T.int32] = (LH_start + row) // group_size
                                                        if _check_tree_order(
                                                            row=row_,
                                                            col=L_kv_start + j,
                                                            batch=b_idx,
                                                            tree_order=mask,
                                                            tree_order_indptr=mn_indptr,
                                                            qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx],
                                                            kv_len=kv_chunk_len[0]):
                                                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with Ts.sblock("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[i] += S_smem[row, j]
                                                    m_smem[row] = m_new[i]
                                                    d_smem[row] = d_new[i]
                                                    m_prev_smem[row] = m_prev[i]
                                        T.gpu_storage_sync("shared")

                                        # Update O
                                        with Ts.sblock():
                                            for li, lj, lk in T.grid(tile_x, tile_y, tile_z):
                                                with Ts.sblock("O_gemm"):
                                                    i, j, k_axis = Ts.axis.remap("SSR", [li, lj, lk])
                                                    with Ts.init():
                                                        O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                                                    O_local[i, j] += S_smem[i, k_axis] * T.cast(V_smem[k_axis, j], ty="float32")

                                    # Store O from smem to gmem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with Ts.sblock("O_store"):
                                            i, j = Ts.axis.remap("SS", [li, lj])
                                            cur_L: T.let[T.int32] = q_indptr[b_idx] + (LH_start + i) // group_size
                                            cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                output[cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                                    # Store LSE to gmem
                                    for li in T.grid(tile_x):
                                        with Ts.sblock("lse_store"):
                                            i = Ts.axis.remap("S", [li])
                                            cur_L: T.let[T.int32] = q_indptr[b_idx] + (LH_start + i) // group_size
                                            cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # fmt: on
    # pylint: enable=line-too-long,too-many-branches
    sch = s_tir.Schedule(batch_tree_attn)

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = math.ceil(math.sqrt(cnt))
        while (cnt % tile_y != 0 or y % tile_y != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: s_tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, bdx, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: s_tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(  # pylint: disable=unused-argument
        sch: s_tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
        loop_x, loop_y, loop_z = sch.get_loops(block)[-3:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        ko, ki = sch.split(loop_z, factors=[None, r_len])
        if k_major:
            sch.reorder(ko, xi, yi, ki)
        else:
            sch.reorder(ko, ki, xi, yi)
        sch.decompose_reduction(block, ty)

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
    apply_to_gemm(sch, sch.get_sblock("S_gemm"), tile_s, 0, 1, k_major=True)
    apply_to_gemm(sch, sch.get_sblock("O_gemm"), tile_o, 2, 3, k_major=False)
    apply_to_so_ewise(sch, sch.get_sblock("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_sblock("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_sblock("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_sblock("Q_load"))
    apply_to_qkv_load(sch, sch.get_sblock("KV_load"))

    apply_to_md(sch, sch.get_sblock("lse_store"))
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)


def tree_attn_with_paged_kv_cache_cpu(h_kv, h_q, d, dtype, rope_scaling: dict[str, Any]):
    """Generate tree attention kernel for batched tree attention with paged key-value cache.

    Parameters
    ----------
    h_kv : int
        Number of heads for key and value.
    h_q : int
        Number of heads for query.
    d : int
        Hidden dimension.
    dtype : str
        Data type.
    target : Target
        The target device.

    Returns
    -------
    mod : tvm_tirx.IRModule
        The generated IR module.
    """
    global_symbol = "tree_attn_paged_kv_cpu"
    sliding_window = False
    group_size = h_q // h_kv

    # pylint: disable=line-too-long,too-many-branches
    # fmt: off
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
    tree_order_elem_offset = T.dynamic("tree_order_elem_offset", "int32")
    tree_order_indptr_elem_offset = T.dynamic("tree_order_indptr_elem_offset", "int32")
    total_tree_order_len = T.dynamic("total_tree_order_len", "int32")
    @Ts.function
    def tree_attn_paged_kv_cpu(
        q: T.Tensor((total_len, h_q, d), dtype), # [total_len, h_q, d]
        q_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (q_indptr_elem_offset))), # [batch_size + 1]
        pages: T.Tensor((max_num_pages, 2, h_kv, 16, d), dtype), # [max_num_pages, 2, h_kv, page_size, d]
        page_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (page_indptr_elem_offset))), # [batch_size + 1]
        page_values: T.Tensor((nnz_pages,), 'int32', layout=T.TileLayout(T.S[(nnz_pages,)] + (page_values_elem_offset))), # [nnz_pages]
        length_info: _length_info_buffer(batch_size, sliding_window, length_info_elem_offset), # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset: T.Tensor((batch_size,), 'int32', layout=T.TileLayout(T.S[(batch_size,)] + (k_rope_pos_offset_elem_offset))), # [b]
        q_rope_position: T.Tensor((total_len,), 'int32', layout=T.TileLayout(T.S[(total_len,)] + (q_rope_position_elem_offset))), # [total_len]
        output: T.Tensor((total_len, h_q, d), dtype), # [total_len, h_q, d]
        lse: T.Tensor((total_len, h_q), 'float32'), # [total_len, h_q]
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
        tree_order_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (tree_order_indptr_elem_offset))),  # [batch_size + 1]
        tree_order: T.Tensor((total_tree_order_len, 2), 'int32', layout=T.TileLayout(T.S[(total_tree_order_len, 2)] + (tree_order_elem_offset))),  # [total_len, 2]
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

        T.assert_(
            rotary_mode == T.int32(0), "Inline rotary mode is not supported in tree attention."
        )

        for h_qo in T.serial(h_q):
            for b_idx in T.serial(batch_size):
                with Ts.sblock("attn"):
                    Ts.reads()
                    Ts.writes()
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
                    kv_chunk_len[0] = T.if_then_else(
                        cur_page_indptr_begin != cur_page_indptr_end,
                        _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, 16, b_idx, length_info, sliding_window),
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
                        for row_idx in T.serial(max_num_pages * 16):
                            if row_idx < kv_chunk_len[0]:
                                page_no: T.let[T.int32()] = page_values[cur_page_indptr_begin + (_get_seq_offset(row_idx, b_idx, length_info, sliding_window) // 16)]
                                page_offset: T.let[T.int32()] = _get_seq_offset(row_idx, b_idx, length_info, sliding_window) % 16

                                # Load KV
                                for d_idx in T.serial(d):
                                    K_local[d_idx] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(pages, k_rope_pos_offset[b_idx] + row_idx, d, rope_theta, rope_scale, (page_no, 0, h_qo // group_size, page_offset, d_idx), dtype, rope_scaling),
                                        pages[page_no, 0, h_qo // group_size, page_offset, d_idx]
                                    )
                                    V_local[d_idx] = pages[page_no, 1, h_qo // group_size, page_offset, d_idx]

                                # Compute S
                                S_val[0] = 0.0
                                for d_idx in T.serial(d):
                                    S_val[0] += Q_local[d_idx] * K_local[d_idx]
                                S_val[0] *= sm_scale * math.log2(math.exp(1))

                                # update m_val, d_val , O_local
                                if _check_tree_order(
                                    tree_order_indptr=tree_order_indptr,
                                    tree_order=tree_order,
                                    batch=b_idx,
                                    row=q_idx,
                                    col=row_idx,
                                    kv_len=kv_chunk_len[0],
                                    qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx],
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
                                    O_local[d_idx] = O_local[d_idx] * scale_O[d_idx]

                                for d_idx in T.serial(d):
                                    O_local[d_idx] += V_local[d_idx] * factor[0]
                        # Store Output
                        for d_idx in T.serial(d):
                            O_local[d_idx] = O_local[d_idx] /d_val[0]
                            output[curl_q, h_qo, d_idx] = O_local[d_idx]
                        lse[curl_q, h_qo] = m_val[0] + T.log2(d_val[0])
    return tree_attn_paged_kv_cpu


def tree_attn_with_paged_kv_cache(
    h_kv, h_q, d, dtype, rope_scaling: dict[str, Any], target: Target
):
    """Generate tree attention kernel for batched tree attention with paged key-value cache.

    Parameters
    ----------
    h_kv : int
        Number of heads for key and value.
    h_q : int
        Number of heads for query.
    d : int
        Hidden dimension.
    dtype : str
        Data type.
    target : Target
        The target device.

    Returns
    -------
    mod : tvm_tirx.IRModule
        The generated IR module.
    """
    # pylint: disable=invalid-name, line-too-long
    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = (
        _get_prefill_kernel_config(h_kv, h_q, d, dtype, target)
    )

    global_symbol = "tree_attn_paged_kv"
    sliding_window = False  # Sliding window is not supported in this kernel.

    # fmt: off
    batch_size = T.dynamic("batch_size", "int32")
    total_len = T.dynamic("total_len", "int32")
    nnz_pages = T.dynamic("nnz_pages", "int32")
    max_num_pages = T.dynamic("max_num_pages", "int32")
    q_indptr_elem_offset = T.dynamic("q_indptr_elem_offset", "int32")
    k_rope_pos_offset_elem_offset = T.dynamic("k_rope_pos_offset_elem_offset", "int32")
    q_rope_position_elem_offset = T.dynamic("q_rope_position_elem_offset", "int32")
    page_indptr_elem_offset = T.dynamic("page_indptr_elem_offset", "int32")
    page_values_elem_offset = T.dynamic("page_values_elem_offset", "int32")
    length_info_elem_offset = T.dynamic("length_info_elem_offset", "int32")
    tree_order_elem_offset = T.dynamic("tree_order_elem_offset", "int32")
    tree_order_indptr_elem_offset = T.dynamic("tree_order_indptr_elem_offset", "int32")
    total_tree_order_len = T.dynamic("total_tree_order_len", "int32")
    @Ts.function
    def tree_attn_paged_kv(
        q: T.Tensor((total_len, h_q, d), dtype),  # [total_len, h_q, d]
        q_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (q_indptr_elem_offset))),  # [batch_size + 1]
        pages: T.Tensor((max_num_pages, 2, h_kv, 16, d), dtype),  # [max_num_pages, 2, h_kv, page_size, d]
        page_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (page_indptr_elem_offset))),  # [batch_size + 1]
        page_values: T.Tensor((nnz_pages,), 'int32', layout=T.TileLayout(T.S[(nnz_pages,)] + (page_values_elem_offset))),  # [nnz_pages]
        length_info: _length_info_buffer(batch_size, sliding_window, length_info_elem_offset),  # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset: T.Tensor((batch_size,), 'int32', layout=T.TileLayout(T.S[(batch_size,)] + (k_rope_pos_offset_elem_offset))),  # [b]
        q_rope_position: T.Tensor((total_len,), 'int32', layout=T.TileLayout(T.S[(total_len,)] + (q_rope_position_elem_offset))),  # [total_len]
        output: T.Tensor((total_len, h_q, d), dtype),  # [total_len, h_q, d]
        lse: T.Tensor((total_len, h_q), 'float32'),  # [total_len, h_q]
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
        tree_order_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (tree_order_indptr_elem_offset))),  # [batch_size + 1]
        tree_order: T.Tensor((total_tree_order_len, 2), 'int32', layout=T.TileLayout(T.S[(total_tree_order_len, 2)] + (tree_order_elem_offset))),  # [total_len, 2]
    ):
        # pylint: disable=unused-variable, too-many-branches
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

        T.assert_(
            rotary_mode == T.int32(0), "Inline rotary mode is not supported in tree attention."
        )

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
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)

                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.gpu_thread_invariant(batch_idx[0] < batch_size):
                                # advance to next tile
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] -= batch_tiles[0]
                                    batch_idx[0] += 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.let[T.int32] = batch_idx[0]
                                        batch_rows[0] = (
                                            q_indptr[b_idx + 1] - q_indptr[b_idx]
                                        ) * group_size
                                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                if T.gpu_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.let[T.int32()] = batch_idx[0]
                                    LH_start: T.let[T.int32()] = tile_id[0] * tile_x
                                    q_indptr_val: T.let[T.int32] = q_indptr[b_idx]

                                    cur_page_indptr_begin: T.let[T.int32] = page_indptr[b_idx]
                                    cur_page_indptr_end: T.let[T.int32] = page_indptr[b_idx + 1]
                                    kv_chunk_len[0] = T.if_then_else(
                                        cur_page_indptr_begin != cur_page_indptr_end,
                                        _get_kv_chunk_len(
                                            cur_page_indptr_end - cur_page_indptr_begin,
                                            16,
                                            b_idx,
                                            length_info,
                                            sliding_window,
                                        ),
                                        0,
                                    )
                                    T.gpu_storage_sync("shared")

                                    # init states
                                    for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                        row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
                                        if row < tile_x:
                                            m_smem[row] = -5e4
                                            d_smem[row] = 1.0

                                    for li, lj in T.grid(tile_x, tile_y):
                                        with Ts.sblock("O_init"):
                                            i, j = Ts.axis.remap("SS", [li, lj])
                                            O_local[i, j] = 0.0
                                    T.gpu_storage_sync("shared")

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
                                                    _rope(
                                                        q,
                                                        q_rope_position[cur_L],
                                                        d,
                                                        rope_theta,
                                                        rope_scale,
                                                        (cur_L, cur_H_qo, j),
                                                        dtype,
                                                        rope_scaling,
                                                    ),
                                                    q[cur_L, cur_H_qo, j],
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
                                                    page_no: T.let[T.int32()] = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, 16)]  # type: ignore
                                                    page_offset: T.let[T.int32()] = T.floormod(seq_offset, 16)  # type: ignore
                                                    K_smem[i, j] = pages[
                                                        page_no, 0, by, page_offset, j
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
                                                if cur_L < kv_chunk_len[0]:
                                                    seq_offset: T.let[T.int32()] = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                    page_no: T.let[T.int32()] = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, 16)]  # type: ignore
                                                    page_offset: T.let[T.int32()] = T.floormod(seq_offset, 16)  # type: ignore
                                                    V_smem[i, j] = pages[
                                                        page_no, 1, by, page_offset, j
                                                    ]
                                                else:
                                                    V_smem[i, j] = 0.0
                                        T.gpu_storage_sync("shared")

                                        # Compute S
                                        with Ts.sblock():
                                            for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                                                with Ts.sblock("S_gemm"):
                                                    i, j, k = Ts.axis.remap("SSR", [li, lj, lk])
                                                    with Ts.init():
                                                        S_local[i, j] = 0.0
                                                    S_local[i, j] += (
                                                        T.cast(Q_smem[i, k], ty="float32")
                                                        * T.cast(K_smem[j, k], ty="float32")
                                                        * sm_scale
                                                        * math.log2(math.exp(1))
                                                    )
                                        T.gpu_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with Ts.sblock("S_store"):
                                                i, j = Ts.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.gpu_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with Ts.sblock("update1"):
                                                    m_prev[i] = m_smem[row]
                                                    m_new[i] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    row_: T.let[T.int32] = (LH_start + row) // group_size
                                                    for j in T.serial(tile_z):
                                                        if _check_tree_order(
                                                            tree_order_indptr=tree_order_indptr,
                                                            tree_order=tree_order,
                                                            batch=b_idx,
                                                            row=row_,
                                                            col=L_kv_start + j,
                                                            kv_len=kv_chunk_len[0],
                                                            qo_len=q_indptr[b_idx + 1]
                                                            - q_indptr[b_idx],
                                                        ):
                                                            m_new[i] = T.max(
                                                                m_new[i], S_smem[row, j]
                                                            )
                                                    d_new[i] = d_smem[row] * T.exp2(
                                                        m_prev[i] - m_new[i]
                                                    )

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
                                            with Ts.sblock("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        row_: T.let[T.int32] = (
                                                            LH_start + row
                                                        ) // group_size
                                                        if _check_tree_order(
                                                            tree_order_indptr=tree_order_indptr,
                                                            tree_order=tree_order,
                                                            batch=b_idx,
                                                            row=row_,
                                                            col=L_kv_start + j,
                                                            kv_len=kv_chunk_len[0],
                                                            qo_len=q_indptr[b_idx + 1]
                                                            - q_indptr[b_idx],
                                                        ):
                                                            S_smem[row, j] = T.exp2(
                                                                S_smem[row, j] - m_new[i]
                                                            )
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with Ts.sblock("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[i] += S_smem[row, j]
                                                    m_smem[row] = m_new[i]
                                                    d_smem[row] = d_new[i]
                                                    m_prev_smem[row] = m_prev[i]
                                        T.gpu_storage_sync("shared")

                                        # Update O
                                        with Ts.sblock():
                                            for li, lj, lk in T.grid(tile_x, tile_y, tile_z):
                                                with Ts.sblock("O_gemm"):
                                                    i, j, k = Ts.axis.remap("SSR", [li, lj, lk])
                                                    with Ts.init():
                                                        O_local[i, j] *= T.exp2(
                                                            m_prev_smem[i] - m_smem[i]
                                                        )
                                                    O_local[i, j] += S_smem[i, k] * T.cast(
                                                        V_smem[k, j], ty = "float32"
                                                    )

                                    # Store O from smem to gmem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with Ts.sblock("O_store"):
                                            i, j = Ts.axis.remap("SS", [li, lj])
                                            cur_L: T.let[T.int32] = (
                                                q_indptr[b_idx] + (LH_start + i) // group_size
                                            )
                                            cur_H_qo: T.let[T.int32] = (
                                                by * group_size + (LH_start + i) % group_size
                                            )
                                            if cur_L < q_indptr[b_idx + 1]:
                                                output[cur_L, cur_H_qo, j] = (
                                                    O_local[i, j] / d_smem[i]
                                                )

                                    # Store LSE to gmem
                                    for li in T.grid(tile_x):
                                        with Ts.sblock("lse_store"):
                                            i = Ts.axis.remap("S", [li])
                                            cur_L: T.let[T.int32] = (
                                                q_indptr[b_idx] + (LH_start + i) // group_size
                                            )
                                            cur_H_qo: T.let[T.int32] = (
                                                by * group_size + (LH_start + i) % group_size
                                            )
                                            if cur_L < q_indptr[b_idx + 1]:
                                                lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS

    # fmt: on
    # pylint: enable=line-too-long,too-many-branches
    sch = s_tir.Schedule(tree_attn_paged_kv)

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = math.ceil(math.sqrt(cnt))
        while (cnt % tile_y != 0 or y % tile_y != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: s_tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, bdx, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: s_tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(  # pylint: disable=unused-argument
        sch: s_tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
        loop_x, loop_y, loop_z = sch.get_loops(block)[-3:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        ko, ki = sch.split(loop_z, factors=[None, r_len])
        if k_major:
            sch.reorder(ko, xi, yi, ki)
        else:
            sch.reorder(ko, ki, xi, yi)
        sch.decompose_reduction(block, ty)

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
    apply_to_gemm(sch, sch.get_sblock("S_gemm"), tile_s, 0, 1, k_major=True)
    apply_to_gemm(sch, sch.get_sblock("O_gemm"), tile_o, 2, 3, k_major=False)
    apply_to_so_ewise(sch, sch.get_sblock("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_sblock("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_sblock("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_sblock("Q_load"))
    apply_to_qkv_load(sch, sch.get_sblock("K_load"))
    apply_to_qkv_load(sch, sch.get_sblock("V_load"))
    apply_to_md(sch, sch.get_sblock("lse_store"))
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)
