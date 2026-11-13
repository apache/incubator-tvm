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
# ruff: noqa: E501, E731, RUF005
# fmt: off

"""Shared TIR helpers used by KV-cache / attention kernels in this package.

This module consolidates constructs reused by the prefill/decode/paged/tree
attention kernels so each kernel file can focus on its own specialised logic.

Contents:
- Thread-limit checks (``get_max_num_threads_per_block``, ``check_thread_limits``)
- KV-cache enums (``AttnKind``, ``RopeMode``)
- Small TVMScript helpers (``_var``, ``_var_cpu``, ``_causal_mask``, ``_rope``)
- Length-info accessors for sliding-window-aware indexing
- Buffer allocators for the tiled online-softmax state used by every prefill kernel
- ``_make_prefill_macros`` — the ``@T.macro`` bundle invoked by the prefill kernels
- Tiling config (``_get_prefill_kernel_config``) and scheduling (``_schedule_prefill_kernel``)
"""
# pylint: disable=too-many-statements,too-many-arguments,invalid-name,line-too-long
import enum
import math
from typing import Any

import tvm_ffi

import tvm
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import Range
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

from .position_embedding import switch_rope_freq_func


def _var(dtype):
    return Ts.sblock_alloc_tensor((1,), dtype, scope="local")

def _var_cpu(dtype):
    return Ts.sblock_alloc_tensor((1,), dtype)

def get_max_num_threads_per_block(target: Target) -> int:
    """
    max(max_num_threads, max_threads_per_block); if latter does not exist, return max_num_threads.
    We add this method since some targets have both fields and `max_threads_per_block` is larger.
    """
    max_num_threads = int(target.attrs["max_num_threads"])
    max_threads_per_block = target.attrs.get("max_threads_per_block", None)
    if max_threads_per_block is None:
        return max_num_threads
    return max(max_num_threads, max_threads_per_block)

def check_thread_limits(target: Target, bdx: int, bdy: int, bdz: int, gdz: int):
    """
    Check whether max num threads exceeded given a target.

    Parameters
    ----------
    bdx: threadIdx.x
    bdy: threadIdx.y
    bdz: threadIdx.z
    gdz: blockIdx.z
    """
    max_num_threads_per_block = get_max_num_threads_per_block(target)

    assert bdx * bdy * bdz <= max_num_threads_per_block, (
        f"{target.kind} max num threads exceeded: {bdx}*{bdy}*{bdz}>{max_num_threads_per_block}"
    )

    if target.kind.name == "webgpu":
        # https://gpuweb.github.io/gpuweb/#dom-supported-limits-maxcomputeworkgroupsizez
        assert bdz <= 64, f"webgpu's threadIdx.z cannot exceed 64, but got bdz={bdz}"
        assert gdz == 1, f"webgpu's blockIdx.z should be 1, but got gdz={gdz}"

class AttnKind(enum.IntEnum):
    """The attention kind class.
    MHA denotes multi-head attention, multi-query attention or grouped query attention.
    MLA denotes multi-head latent attention.
    """

    MHA = 0
    MLA = 1
    MHA_SLIDING = 3

class RopeMode(enum.IntEnum):
    """The RoPE mode of the Paged KV cache.
    If it is none, the KV cache will not apply RoPE to q and k.
    If it is normal, RoPE will be applied to k before adding k to cache.
    Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
    """

    NONE = 0
    NORMAL = 1
    INLINE = 2

def _rope(buffer: T.Tensor, offset: tirx.Var, rotary_dim: int, theta: tirx.Var, scale: tirx.Var, indices: tuple[tirx.Var, ...], qkv_dtype: str, rope_scaling: dict[str, Any]):
    d = indices[-1]
    cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(offset * scale, d, rotary_dim, theta, "float32")
    cos = cos_freq * buffer[indices].astype("float32")
    sin = sin_freq * tirx.if_then_else(
        d < rotary_dim // 2,
        -buffer[indices[:-1] + (d + rotary_dim // 2,)],
        buffer[indices[:-1] + (d - rotary_dim // 2,)],
    ).astype("float32")
    expr = (cos + sin).astype(qkv_dtype)
    for var, value in var_map.items():
        expr = tirx.Let(var, value, expr)
    return expr

def _causal_mask(causal, row, col, kv_len, qo_len):
    return T.if_then_else(
        causal > 0,
        col < kv_len - qo_len + row + 1,
        col < kv_len,
    )

def _causal_or_sliding_cross_mask(causal, row, col, kv_len, qo_len, sliding_window_size):
    visible_past = T.max(sliding_window_size - row - 1, 0)
    return T.if_then_else(
        tirx.And(causal > 0, sliding_window_size > 0),
        tirx.And(col < kv_len, col >= T.max(kv_len - visible_past, 0)),
        _causal_mask(causal, row, col, kv_len, qo_len),
    )

def _length_info_buffer(batch_size, sliding_window, elem_offset):
    return (
        T.Tensor( (3, batch_size), "int32", layout=T.TileLayout(T.S[(3, batch_size)] + (elem_offset)))
        if sliding_window
        else T.Tensor( (batch_size,), "int32", layout=T.TileLayout(T.S[(batch_size,)] + (elem_offset)))
    )

def _get_kv_chunk_len(num_pages, page_size, seq_id, length_info, sliding_window):
    if not sliding_window:
        return (num_pages - 1) * page_size + length_info[seq_id]
    # ((num_pages - 1) * page_size + last_page_len) - sliding_window_offset + sink_size
    return (num_pages - 1) * page_size + length_info[0, seq_id] - length_info[1, seq_id] + length_info[2, seq_id]

def _get_seq_offset(pos, seq_id, length_info, sliding_window):
    if not sliding_window:
        return pos
    # pos if pos < sink_size else pos - sink_size + sliding_window_offset
    return T.if_then_else(
        pos < length_info[2, seq_id],
        pos,
        pos - length_info[2, seq_id] + length_info[1, seq_id],
    )

def _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps):
    """Allocate the shared/local online-softmax working state used by every tiled prefill kernel.

    Returns ``(S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new)``.
    """
    S_smem = Ts.sblock_alloc_tensor((tile_x, tile_z), "float32", scope="shared")
    S_local = Ts.sblock_alloc_tensor((tile_x, tile_z), "float32", scope="local")
    m_smem = Ts.sblock_alloc_tensor((tile_x,), "float32", scope="shared")
    m_prev_smem = Ts.sblock_alloc_tensor((tile_x,), "float32", scope="shared")
    d_smem = Ts.sblock_alloc_tensor((tile_x,), "float32", scope="shared")
    md_shape = (math.ceil(tile_x / (bdx * num_warps)),)
    m_new = Ts.sblock_alloc_tensor(md_shape, "float32", scope="local")
    m_prev = Ts.sblock_alloc_tensor(md_shape, "float32", scope="local")
    d_new = Ts.sblock_alloc_tensor(md_shape, "float32", scope="local")
    return S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new

def _alloc_mha_qkvo_buffers(tile_x, tile_z, d_qk, d_v, dtype):
    """Allocate Q/K/V shared + O local buffers for standard MHA/GQA prefill kernels."""
    Q_smem = Ts.sblock_alloc_tensor((tile_x, d_qk), dtype, scope="shared")
    K_smem = Ts.sblock_alloc_tensor((tile_z, d_qk), dtype, scope="shared")
    V_smem = Ts.sblock_alloc_tensor((tile_z, d_v), dtype, scope="shared")
    O_local = Ts.sblock_alloc_tensor((tile_x, d_v), "float32", scope="local")
    return Q_smem, K_smem, V_smem, O_local

def _alloc_mla_qkvo_buffers(tile_x, tile_z, d_qk, d_latent, dtype):
    """Allocate Q + combined KV shared + O local for MLA prefill (V reuses the KV buffer)."""
    Q_smem = Ts.sblock_alloc_tensor((tile_x, d_qk), dtype, scope="shared")
    KV_smem = Ts.sblock_alloc_tensor((tile_z, d_qk), dtype, scope="shared")
    O_local = Ts.sblock_alloc_tensor((tile_x, d_latent), "float32", scope="local")
    return Q_smem, KV_smem, O_local

def _alloc_tile_walk_state():
    """Return (tile_id, batch_idx, batch_tiles, batch_rows, iterator, kv_chunk_len) int32 scalars for the paged/ragged/MLA tile-walk state machine."""
    return _var("int32"), _var("int32"), _var("int32"), _var("int32"), _var("int32"), _var("int32")

def _make_prefill_macros(tile_x, tile_y, tile_z, tile_o, bdx, num_warps, group_size):
    """Build @T.macro helpers shared across tiled online-softmax prefill kernels.

    Parameters
    ----------
    tile_x : int  # query/output row tile
    tile_y : int  # QK reduction dim (head_dim for MHA, d_qk for MLA/ragged)
    tile_z : int  # key/value column tile
    tile_o : int  # output/V column dim (d for MHA/sequence, d_v for ragged, d_latent for MLA)
    """
    @T.macro
    def init_states(
        m_smem: T.Tensor, d_smem: T.Tensor, O_local: T.Tensor, ty: T.int32, tx: T.int32,
    ):
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                m_smem[row] = -5e4
                d_smem[row] = 1.0
        for li, lj in T.grid(tile_x, tile_o):
            with Ts.sblock("O_init"):
                i, j = Ts.axis.remap("SS", [li, lj])
                O_local[i, j] = 0.0
        T.gpu_storage_sync("shared")

    @T.macro
    def compute_s_gemm(
        Q_smem: T.Tensor, K_smem: T.Tensor, S_local: T.Tensor, S_smem: T.Tensor, sm_scale: T.float32,
    ):
        with Ts.sblock():
            for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                with Ts.sblock("S_gemm"):
                    i, j, k = Ts.axis.remap("SSR", [li, lj, lk])
                    with Ts.init():
                        S_local[i, j] = 0.0
                    S_local[i, j] += T.cast(Q_smem[i, k], ty="float32") * T.cast(K_smem[j, k], ty="float32") * sm_scale * math.log2(math.exp(1))
        T.gpu_storage_sync("shared")
        for li, lj in T.grid(tile_x, tile_z):
            with Ts.sblock("S_store"):
                i, j = Ts.axis.remap("SS", [li, lj])
                S_smem[i, j] = S_local[i, j]
        T.gpu_storage_sync("shared")

    @T.macro
    def softmax_update_causal(
        S_smem: T.Tensor, m_smem: T.Tensor, d_smem: T.Tensor, m_prev_smem: T.Tensor,
        m_new: T.Tensor, m_prev: T.Tensor, d_new: T.Tensor,
        ty: T.int32, tx: T.int32, LH_start: T.int32, L_kv_start: T.int32,
        causal: T.int32, kv_len: T.int32, qo_len: T.int32, sliding_window_size: T.int32,
    ):
        # Phase 1: compute m_new = max(masked S over kv tile), d_new = d_prev * exp2(m_prev - m_new)
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with Ts.sblock("update1"):
                    m_prev[i] = m_smem[row]
                    m_new[i] = m_smem[row]
                    row_: T.let[T.int32] = (LH_start + row) // group_size
                    for j in T.serial(tile_z):
                        if _causal_or_sliding_cross_mask(causal, row=row_, col=L_kv_start + j, kv_len=kv_len, qo_len=qo_len, sliding_window_size=sliding_window_size):
                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])
        # Phase 2: exp-and-scale S_smem; masked-out entries use -inf
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
            with Ts.sblock("update"):
                for j in T.serial(tile_z):
                    # predicate sits inside loop so sync stays outside conditional branches
                    if row < tile_x:
                        row_: T.let[T.int32] = (LH_start + row) // group_size
                        if _causal_or_sliding_cross_mask(causal, row=row_, col=L_kv_start + j, kv_len=kv_len, qo_len=qo_len, sliding_window_size=sliding_window_size):
                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                        else:
                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])
        # Phase 3: d_new += sum(S_smem[row, :]); write m/d/m_prev back to smem
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

    @T.macro
    def compute_o_gemm(
        S_smem: T.Tensor, V_smem: T.Tensor, O_local: T.Tensor,
        m_prev_smem: T.Tensor, m_smem: T.Tensor,
    ):
        with Ts.sblock():
            for li, lj, lk in T.grid(tile_x, tile_o, tile_z):
                with Ts.sblock("O_gemm"):
                    i, j, k = Ts.axis.remap("SSR", [li, lj, lk])
                    with Ts.init():
                        O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                    O_local[i, j] += S_smem[i, k] * T.cast(V_smem[k, j], ty="float32")

    @T.macro
    def paged_store_output_lse(
        output: T.Tensor, lse: T.Tensor, O_local: T.Tensor, m_smem: T.Tensor, d_smem: T.Tensor,
        q_indptr: T.Tensor, b_idx: T.int32, by: T.int32, LH_start: T.int32,
    ):
        """Paged-style (q_indptr-based) O_store + lse_store epilogue.

        Used by paged prefill, ragged prefill and MLA prefill. MLA passes ``by=0`` so
        the ``by * group_size`` term drops to zero at compile time.
        """
        for li, lj in T.grid(tile_x, tile_o):
            with Ts.sblock("O_store"):
                i, j = Ts.axis.remap("SS", [li, lj])
                cur_L: T.let[T.int32] = q_indptr[b_idx] + (LH_start + i) // group_size
                cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                if cur_L < q_indptr[b_idx + 1]:
                    output[cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]
        for li in T.grid(tile_x):
            with Ts.sblock("lse_store"):
                i = Ts.axis.remap("S", [li])
                cur_L: T.let[T.int32] = q_indptr[b_idx] + (LH_start + i) // group_size
                cur_H_qo: T.let[T.int32] = by * group_size + (LH_start + i) % group_size
                if cur_L < q_indptr[b_idx + 1]:
                    lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

    @T.macro
    def advance_tile_batch(
        tile_id: T.Tensor, batch_idx: T.Tensor, batch_tiles: T.Tensor, batch_rows: T.Tensor,
        q_indptr: T.Tensor, batch_size: T.int32,
    ):
        """Advance tile_id/batch_idx past exhausted batches.

        After the loop, either batch_idx[0] >= batch_size (all tiles consumed) or
        tile_id[0] < batch_tiles[0] (the current batch still has work to do).
        """
        while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
            tile_id[0] -= batch_tiles[0]
            batch_idx[0] += 1
            if batch_idx[0] < batch_size:
                b_idx: T.let[T.int32] = batch_idx[0]
                batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

    @T.macro
    def softmax_update_valid_length(
        S_smem: T.Tensor, m_smem: T.Tensor, d_smem: T.Tensor, m_prev_smem: T.Tensor,
        m_new: T.Tensor, m_prev: T.Tensor, d_new: T.Tensor,
        ty: T.int32, tx: T.int32, LH_start: T.int32, L_kv_start: T.int32,
        valid_len: T.int32, qo_len: T.int32, kv_len: T.int32,
    ):
        # Same three-phase online softmax as softmax_update_causal but with a
        # per-batch right-padding mask in place of causal masking.
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with Ts.sblock("update1"):
                    m_prev[i] = m_smem[row]
                    m_new[i] = m_smem[row]
                    row_: T.let[T.int32] = (LH_start + row) // group_size
                    for j in T.serial(tile_z):
                        if tirx.And(tirx.And(row_ < qo_len, row_ < valid_len), L_kv_start + j < valid_len):
                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
            with Ts.sblock("update"):
                for j in T.serial(tile_z):
                    if row < tile_x:
                        row_: T.let[T.int32] = (LH_start + row) // group_size
                        if tirx.And(tirx.And(row_ < qo_len, row_ < valid_len), L_kv_start + j < valid_len):
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

    @T.macro
    def softmax_update_causal_padded_left(
        S_smem: T.Tensor, m_smem: T.Tensor, d_smem: T.Tensor, m_prev_smem: T.Tensor,
        m_new: T.Tensor, m_prev: T.Tensor, d_new: T.Tensor,
        ty: T.int32, tx: T.int32, LH_start: T.int32, L_kv_start: T.int32,
        valid_len: T.int32, qo_len: T.int32, kv_len: T.int32,
    ):
        # Three-phase online softmax with left-padding + causal mask. Real
        # queries occupy [qo_len - valid_len, qo_len); real keys occupy
        # [kv_len - valid_len, kv_len). Causal keeps
        # col <= row + (kv_len - qo_len) within those valid suffixes.
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with Ts.sblock("update1"):
                    m_prev[i] = m_smem[row]
                    m_new[i] = m_smem[row]
                    row_: T.let[T.int32] = (LH_start + row) // group_size
                    pad_q: T.let[T.int32] = qo_len - valid_len
                    pad_kv: T.let[T.int32] = kv_len - valid_len
                    for j in T.serial(tile_z):
                        col_: T.let[T.int32] = L_kv_start + j
                        if tirx.And(tirx.And(row_ < qo_len, row_ >= pad_q), tirx.And(col_ >= pad_kv, col_ < kv_len - qo_len + row_ + 1)):
                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.let[T.int32] = i * bdx * num_warps + ty * bdx + tx
            with Ts.sblock("update"):
                for j in T.serial(tile_z):
                    if row < tile_x:
                        row_: T.let[T.int32] = (LH_start + row) // group_size
                        pad_q: T.let[T.int32] = qo_len - valid_len
                        pad_kv: T.let[T.int32] = kv_len - valid_len
                        col_: T.let[T.int32] = L_kv_start + j
                        if tirx.And(tirx.And(row_ < qo_len, row_ >= pad_q), tirx.And(col_ >= pad_kv, col_ < kv_len - qo_len + row_ + 1)):
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

    return init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, softmax_update_valid_length, advance_tile_batch, paged_store_output_lse, softmax_update_causal_padded_left

def _get_prefill_shared_memory_usage(
    tile_x, tile_z, d, dtype, *, d_v=None, merged_kv=False
):
    """Return shared bytes, where ``d`` is Q/K width and ``d_v`` is V/output width.

    ``merged_kv`` denotes MLA's single shared KV buffer. Otherwise K and V occupy
    separate buffers, and ``d_v`` defaults to ``d`` for standard attention.
    """
    if d_v is None:
        d_v = d
    dtype_bytes = (tvm_ffi.dtype(dtype).bits + 7) // 8
    kv_elements = tile_z * d if merged_kv else tile_z * (d + d_v)
    qkv_bytes = (tile_x * d + kv_elements) * dtype_bytes
    softmax_bytes = (tile_x * tile_z + 3 * tile_x) * 4
    return qkv_bytes + softmax_bytes

def _get_prefill_vector_size(extent, load_vec):
    """Return the scheduler's vector width for a contiguous extent."""
    return min(load_vec, extent & ~(extent - 1))

def _get_prefill_tile_size(x, y, num_threads):
    """Return the scheduler's per-thread 2D tile, or ``None`` if none is legal."""
    if (x * y) % num_threads != 0:
        return None
    elements_per_thread = (x * y) // num_threads
    inner_y = math.ceil(math.sqrt(elements_per_thread))
    while inner_y <= elements_per_thread:
        if elements_per_thread % inner_y == 0:
            inner_x = elements_per_thread // inner_y
            if y % inner_y == 0 and x % inner_x == 0:
                return inner_x, inner_y
        inner_y += 1
    return None

def _get_prefill_load_config(x, y, num_threads, load_vec):
    """Return ``(vector width, tile x, tile y)`` for a scheduled load, if legal."""
    if (x * y) % num_threads != 0:
        return None
    elements_per_thread = (x * y) // num_threads
    vec_size = min(
        _get_prefill_vector_size(y, load_vec),
        _get_prefill_vector_size(elements_per_thread, load_vec),
    )
    tile = _get_prefill_tile_size(x, y // vec_size, num_threads)
    if tile is None:
        return None
    return vec_size, *tile

def _is_prefill_kernel_config_legal(
    tile_x, tile_y, tile_z, d_v, load_vec, bdx, num_warps, merged_kv
):
    """Check the factorization assumptions made by the prefill schedulers."""
    num_threads = bdx * num_warps
    return all(
        (
            _get_prefill_tile_size(tile_x, tile_z, num_threads) is not None,
            _get_prefill_tile_size(tile_x, d_v, num_threads) is not None,
            _get_prefill_load_config(tile_x, tile_y, num_threads, load_vec) is not None,
            _get_prefill_load_config(tile_z, tile_y, num_threads, load_vec) is not None,
            merged_kv
            or _get_prefill_load_config(tile_z, d_v, num_threads, load_vec) is not None,
        )
    )

def _fit_prefill_config_to_shared_memory(
    tile_x,
    tile_y,
    tile_z,
    preferred_tile_z,
    max_tile_x,
    d,
    d_v,
    dtype,
    load_vec,
    bdx,
    num_warps,
    merged_kv,
    max_shared_memory_per_block,
):
    """Reduce the query and key tiles until the prefill kernel fits shared memory."""
    if (
        _get_prefill_shared_memory_usage(
            tile_x, tile_z, d, dtype, d_v=d_v, merged_kv=merged_kv
        )
        <= max_shared_memory_per_block
        and _is_prefill_kernel_config_legal(
            tile_x, tile_y, tile_z, d_v, load_vec, bdx, num_warps, merged_kv
        )
    ):
        return tile_x, num_warps, tile_z

    candidate_num_warps = sorted({num_warps, min(num_warps, 2), 1}, reverse=True)
    for warps in candidate_num_warps:
        for candidate_tile_z in range(tile_z, 0, -1):
            if not _is_prefill_kernel_config_legal(
                tile_x,
                tile_y,
                candidate_tile_z,
                d_v,
                load_vec,
                bdx,
                warps,
                merged_kv,
            ):
                continue
            if (
                _get_prefill_shared_memory_usage(
                    tile_x,
                    candidate_tile_z,
                    d,
                    dtype,
                    d_v=d_v,
                    merged_kv=merged_kv,
                )
                <= max_shared_memory_per_block
            ):
                return tile_x, warps, candidate_tile_z

    # If the heuristic query tile cannot be scheduled, prefer the established
    # two-warp low-storage configuration, then try nearby query tiles in either
    # direction, preferring the larger tile on ties. For each query tile, avoid
    # enlarging the original key tile unless thread factorization requires it.
    fallback_num_warps = sorted(
        candidate_num_warps,
        key=lambda warps: (warps != min(num_warps, 2), -warps),
    )
    alternative_tile_x = sorted(
        (candidate for candidate in range(1, max_tile_x + 1) if candidate != tile_x),
        key=lambda candidate: (abs(candidate - tile_x), -candidate),
    )
    fallback_tile_z_ceiling = max(tile_z, bdx * num_warps)
    candidate_tile_z_ranges = (
        range(preferred_tile_z, 0, -1),
        range(preferred_tile_z + 1, fallback_tile_z_ceiling + 1),
    )
    for warps in fallback_num_warps:
        for candidate_tile_x in alternative_tile_x:
            for candidate_tile_z_range in candidate_tile_z_ranges:
                for candidate_tile_z in candidate_tile_z_range:
                    if not _is_prefill_kernel_config_legal(
                        candidate_tile_x,
                        tile_y,
                        candidate_tile_z,
                        d_v,
                        load_vec,
                        bdx,
                        warps,
                        merged_kv,
                    ):
                        continue
                    if (
                        _get_prefill_shared_memory_usage(
                            candidate_tile_x,
                            candidate_tile_z,
                            d,
                            dtype,
                            d_v=d_v,
                            merged_kv=merged_kv,
                        )
                        <= max_shared_memory_per_block
                    ):
                        return candidate_tile_x, warps, candidate_tile_z

    required = _get_prefill_shared_memory_usage(
        tile_x, tile_z, d, dtype, d_v=d_v, merged_kv=merged_kv
    )
    raise ValueError(
        "Unable to find a legal prefill tile within the target's shared-memory limit: "
        f"initial tile requires {required} bytes, target allows "
        f"{max_shared_memory_per_block} bytes"
    )

def _get_prefill_kernel_config(
    h_kv, h_q, d, dtype, target: Target, *, d_v=None, merged_kv=False
):
    if d_v is None:
        d_v = d
    NUM_BLKS = 16
    dtype_bytes = (tvm_ffi.dtype(dtype).bits + 7) // 8
    LOAD_VEC = 8 // dtype_bytes  # 8 bytes
    group_size = h_q // h_kv

    bdx = 32
    num_warps = 4
    # Preserve the largest query tile considered by the existing heuristic.
    max_tile_x = 64 // dtype_bytes
    tile_x, tile_y, tile_z = (
        max_tile_x // max(d // 128, 1),
        d,
        max_tile_x // max(d // 128, 1),
    )
    original_tile_y = tile_y
    original_tile_z = tile_z
    while (tile_x * tile_z) % (bdx * num_warps) != 0:
        tile_z += original_tile_z
    if target.kind.name != "metal":
        while (tile_x * tile_y) % (bdx * num_warps) != 0:
            tile_y += original_tile_y

    # Preserve the established WebGPU config, which targets WebGPU's portable limits.
    if (
        target.kind.name == "webgpu"
        and ((d + 127) // 128) * ((tvm_ffi.dtype(dtype).bits + 15) // 16) >= 4
    ):
        tile_z = 8
        num_warps = 2
        if dtype == "float32" and d == 512 and d_v == 512:
            # A 4x4 tile needs 16 threads and 24,688 shared bytes. The usual
            # 64-thread configuration cannot fit these wide FP32 heads.
            bdx = 16
            num_warps = 1
            tile_x = 4
            tile_z = 4
    if target.kind.name == "metal":
        max_shared_memory_per_block = int(target.attrs["max_shared_memory_per_block"])
        tile_x, num_warps, tile_z = _fit_prefill_config_to_shared_memory(
            tile_x,
            tile_y,
            tile_z,
            original_tile_z,
            max_tile_x,
            d,
            d_v,
            dtype,
            LOAD_VEC,
            bdx,
            num_warps,
            merged_kv,
            max_shared_memory_per_block,
        )
    if target.kind.name == "opencl" and (
        ("android" in str(target.host)) or ("adreno" in str(target.attrs))
    ):
        LOAD_VEC = 16 // ((tvm_ffi.dtype(dtype).bits + 7) // 8)  # 16 bytes
        NUM_BLKS = group_size * 8

    check_thread_limits(target, bdx=bdx, bdy=num_warps, bdz=1, gdz=1)
    if not _is_prefill_kernel_config_legal(
        tile_x, tile_y, tile_z, d_v, LOAD_VEC, bdx, num_warps, merged_kv
    ):
        raise ValueError(
            "Prefill tile is incompatible with the scheduler's thread factorization: "
            f"tile=({tile_x}, {tile_y}, {tile_z}, {d_v}), "
            f"threads={bdx * num_warps}"
        )

    return NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z

def _schedule_prefill_kernel(sch: s_tir.Schedule, load_vec, bdx, num_warps, tile_x, tile_y, tile_z, transform_k_load: bool, merged_qk_load: bool) -> tvm.s_tir.Schedule:
    get_extent = lambda *lps: [int(Range.get_extent_from_begin_end(sch.get(lp).begin, sch.get(lp).end)) for lp in lps]

    def apply_to_qkv_load(sch: s_tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        x_extent, y_extent = get_extent(loop_x, loop_y)
        load_config = _get_prefill_load_config(
            x_extent, y_extent, bdx * num_warps, load_vec
        )
        assert load_config is not None
        vec_size, tile_x, tile_y = load_config
        yo, yv = sch.split(loop_y, [None, vec_size])
        xo, xi = sch.split(loop_x, [tile_x, None])
        yo, yi = sch.split(yo, [tile_y, None])
        sch.reorder(xi, yi, xo, yo)
        t = sch.fuse(xi, yi)
        ty, tx = sch.split(t, [num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(yv)

    def apply_to_so_ewise(sch: s_tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        yiv_extent = _get_prefill_vector_size(tile[1], load_vec)
        yio, yiv = sch.split(yi, [None, yiv_extent])
        sch.unroll(yio)
        sch.vectorize(yiv)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(sch: s_tir.Schedule, block, tile, r_len=16, k_major=False):
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
        yiv_extent = _get_prefill_vector_size(tile[1], load_vec)
        yio, yiv = sch.split(yi, [None, yiv_extent])
        sch.unroll(yio)
        sch.vectorize(yiv)
        sch.unroll(xi)
        sch.decompose_reduction(block, ty)

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    if transform_k_load and not merged_qk_load:
        sch.transform_layout("K_load", ("write", 0), lambda i, j: (j, i))
    tile_s = _get_prefill_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = _get_prefill_tile_size(tile_x, tile_y, bdx * num_warps)
    assert tile_s is not None and tile_o is not None
    apply_to_gemm(sch, sch.get_sblock("S_gemm"), tile_s, k_major=True)
    apply_to_gemm(sch, sch.get_sblock("O_gemm"), tile_o, k_major=False)
    apply_to_so_ewise(sch, sch.get_sblock("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_sblock("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_sblock("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_sblock("Q_load"))
    if not merged_qk_load:
        apply_to_qkv_load(sch, sch.get_sblock("K_load"))
        apply_to_qkv_load(sch, sch.get_sblock("V_load"))
    else:
        apply_to_qkv_load(sch, sch.get_sblock("KV_load"))
    apply_to_md(sch, sch.get_sblock("lse_store"))
    return sch
