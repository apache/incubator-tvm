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

"""TIR kernels that operate on paged KV-cache storage (without doing attention).

This module contains:
- Append helpers that transpose/write new K/V tokens into the paged layout
  (``_kv_cache_transpose_append`` and its MLA variant).
- Debug helpers that extract K/V from the paged layout for inspection
  (``_kv_cache_debug_get_kv``, ``_kv_cache_debug_get_kv_mla``).
- Copy helpers used by the cache runtime for forking/sharing pages
  (``_copy_single_page``, ``_copy_single_page_mla``, ``_copy_single_page_cpu``).
- Compact helpers that reorganise pages after removals
  (``_compact_kv_copy``, ``_compact_kv_copy_cpu``).
"""
from __future__ import annotations

# pylint: disable=too-many-statements,too-many-arguments,invalid-name,line-too-long
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

from ._kernel_common import get_max_num_threads_per_block


def _kv_cache_transpose_append(num_key_value_heads, head_dim, dtype, page_size: int = 16):
    """Return the TIR function that appends new k/v data to PagedKVCache."""

    ntoken = T.dynamic("ntoken", "int64")
    num_pages = T.dynamic("num_pages", "int64")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    position_map_elem_offset = T.dynamic("position_map_elem_offset", "int32")
    @Ts.function
    def tir_kv_cache_transpose_append(
        pages: T.Tensor((num_pages, 2, num_key_value_heads, page_size, head_dim), dtype, layout=T.TileLayout(T.S[(num_pages, 2, num_key_value_heads, page_size, head_dim)] + (pages_elem_offset))),
        k_data: T.Tensor((ntoken, num_key_value_heads, head_dim), dtype),
        v_data: T.Tensor((ntoken, num_key_value_heads, head_dim), dtype),
        position_map: T.Tensor((ntoken,), 'int32', layout=T.TileLayout(T.S[(ntoken,)] + (position_map_elem_offset))),
    ):
        T.func_attr({"tirx.noalias": True})

        for global_pos, h, f in T.grid(ntoken, num_key_value_heads, head_dim):
            if position_map[global_pos] != T.int32(-1):
                with Ts.sblock("k_transpose_append"):
                    vgpos, vh, vf = Ts.axis.remap("SSS", [global_pos, h, f])
                    Ts.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    Ts.writes(pages[position_map[vgpos] // page_size, 0, vh, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore
                    pages[T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vf] = k_data[vgpos, vh, vf]
                with Ts.sblock("v_transpose_append"):
                    vgpos, vh, vf = Ts.axis.remap("SSS", [global_pos, h, f])
                    Ts.reads(position_map[vgpos], v_data[vgpos, vh, vf])
                    Ts.writes(pages[position_map[vgpos] // page_size, 1, vh, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos] # type: ignore[name-defined,no-redef]
                    pages[T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vf] = v_data[vgpos, vh, vf]

    return tir_kv_cache_transpose_append

def _kv_cache_transpose_append_mla(d_qk: int, dtype, page_size: int = 16):
    """Return the TIR function that appends new compressed KV data to PagedKVCache for MLA."""

    ntoken = T.dynamic("ntoken", "int64")
    num_pages = T.dynamic("num_pages", "int64")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    position_map_elem_offset = T.dynamic("position_map_elem_offset", "int32")
    @Ts.function
    def tir_kv_cache_transpose_append_mla(
        pages: T.Tensor((num_pages, page_size, d_qk), dtype, layout=T.TileLayout(T.S[(num_pages, page_size, d_qk)] + (pages_elem_offset))),
        kv_data: T.Tensor((ntoken, d_qk), dtype),
        position_map: T.Tensor((ntoken,), 'int32', layout=T.TileLayout(T.S[(ntoken,)] + (position_map_elem_offset))),
    ):
        T.func_attr({"tirx.noalias": True})

        for global_pos, f in T.grid(ntoken, d_qk):
            if position_map[global_pos] != T.int32(-1):
                with Ts.sblock("k_transpose_append"):
                    vgpos, vf = Ts.axis.remap("SS", [global_pos, f])
                    Ts.reads(position_map[vgpos], kv_data[vgpos, vf])
                    Ts.writes(pages[position_map[vgpos] // page_size, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore
                    pages[T.floordiv(position, page_size), T.floormod(position, page_size), vf] = kv_data[vgpos, vf]

    return tir_kv_cache_transpose_append_mla

def _kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype):
    """Return the TIR function that fetches the k/v data on given positions and layer."""

    seqlen = T.dynamic("seqlen", "int64")
    page_size = T.dynamic("page_size", "int64")
    num_pages = T.dynamic("num_pages", "int64")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    position_map_elem_offset = T.dynamic("position_map_elem_offset", "int64")
    @Ts.function
    def tir_kv_cache_debug_get_kv(
        pages: T.Tensor((num_pages, 2, num_key_value_heads, page_size, head_dim), dtype, layout=T.TileLayout(T.S[(num_pages, 2, num_key_value_heads, page_size, head_dim)] + (pages_elem_offset))),
        position_map: T.Tensor((seqlen,), 'int32', layout=T.TileLayout(T.S[(seqlen,)] + (position_map_elem_offset))),
        k_data: T.Tensor((num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype),
        v_data: T.Tensor((num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype),
        layer_id: T.int64,
    ):
        T.func_attr({"tirx.noalias": True})

        for p, h, d in T.grid(seqlen, num_key_value_heads, head_dim):
            with Ts.sblock("copy0"):
                vp, vh, vd = Ts.axis.remap("SSS", [p, h, d])
                Ts.reads(position_map[vp], pages[position_map[vp] // page_size, 0:2, vh, position_map[vp] % page_size, vd])
                Ts.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int32 = position_map[vp] # type: ignore[name-defined]
                k_data[layer_id, vp, vh, vd] = pages[T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vd]
                v_data[layer_id, vp, vh, vd] = pages[T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vd]

    return tir_kv_cache_debug_get_kv

def _kv_cache_debug_get_kv_mla(num_hidden_layers, d_qk, dtype):
    """Return the TIR function that fetches the k/v data on given positions and layer."""

    seqlen = T.dynamic("seqlen", "int64")
    page_size = T.dynamic("page_size", "int64")
    num_pages = T.dynamic("num_pages", "int64")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    position_map_elem_offset = T.dynamic("position_map_elem_offset", "int64")
    @Ts.function
    def tir_kv_cache_debug_get_kv_mla(
        pages: T.Tensor((num_pages, page_size, d_qk), dtype, layout=T.TileLayout(T.S[(num_pages, page_size, d_qk)] + (pages_elem_offset))),
        position_map: T.Tensor((seqlen,), 'int32', layout=T.TileLayout(T.S[(seqlen,)] + (position_map_elem_offset))),
        compressed_kv_with_k_pe_data: T.Tensor((num_hidden_layers, seqlen, d_qk), dtype),
        layer_id: T.int64,
    ):
        T.func_attr({"tirx.noalias": True})

        for p, d in T.grid(seqlen, d_qk):
            with Ts.sblock("copy0"):
                vp, vd = Ts.axis.remap("SS", [p, d])
                Ts.reads(position_map[vp], pages[position_map[vp] // page_size, position_map[vp] % page_size, vd])
                Ts.writes(compressed_kv_with_k_pe_data[layer_id, vp, vd])
                position: T.int32 = position_map[vp] # type: ignore[name-defined]
                compressed_kv_with_k_pe_data[layer_id, vp, vd] = pages[T.floordiv(position, page_size), T.floormod(position, page_size), vd]

    return tir_kv_cache_debug_get_kv_mla

def _copy_single_page(num_heads, page_size, head_dim, dtype, target: Target):
    tx = get_max_num_threads_per_block(target)

    num_pages = T.dynamic("num_pages", "int32")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    @Ts.function
    def copy_single_page(pages: T.Tensor((num_pages, 2, num_heads, page_size, head_dim), dtype, layout=T.TileLayout(T.S[(num_pages, 2, num_heads, page_size, head_dim)] + (pages_elem_offset))), src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tirx.is_scheduled": True})

        for b in T.parallel((copy_length * num_heads * head_dim + tx - 1) // tx, annotations={"thread_binding": "blockIdx.x"}):
            for t in T.parallel(tx, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("copy"):
                    Ts.where(b * tx + t < copy_length * num_heads * head_dim)
                    vh = Ts.axis.spatial(num_heads, T.Cast("int32", (b * tx + t) // (copy_length * head_dim)))
                    vp = Ts.axis.spatial(copy_length, (b * tx + t) % (copy_length * head_dim) // head_dim)
                    vd = Ts.axis.spatial(head_dim, T.Cast("int32", (b * tx + t) % head_dim))
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]

    return copy_single_page

def _copy_single_page_mla(page_size, head_dim, dtype, target: Target):
    tx = get_max_num_threads_per_block(target)

    num_pages = T.dynamic("num_pages", "int32")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    @Ts.function
    def copy_single_page_mla(pages: T.Tensor((num_pages, page_size, head_dim), dtype, layout=T.TileLayout(T.S[(num_pages, page_size, head_dim)] + (pages_elem_offset))), src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tirx.is_scheduled": True})

        for b in T.parallel((copy_length * head_dim + tx - 1) // tx, annotations={"thread_binding": "blockIdx.x"}):
            for t in T.parallel(tx, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("copy"):
                    Ts.where(b * tx + t < copy_length * head_dim)
                    vp = Ts.axis.spatial(copy_length, (b * tx + t) // head_dim)
                    vd = Ts.axis.spatial(head_dim, T.Cast("int32", (b * tx + t) % head_dim))
                    pages[tgt_page_id, vp, vd] = pages[src_page_id, vp, vd]

    return copy_single_page_mla

def _copy_single_page_cpu(num_heads, page_size, head_dim, dtype):
    tx = 1

    num_pages = T.dynamic("num_pages", "int32")
    @Ts.function
    def copy_single_page_cpu(pages: T.Tensor((num_pages, 2, num_heads, page_size, head_dim), dtype), src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tirx.is_scheduled": True})

        for b in T.serial((copy_length * num_heads * head_dim + tx - 1) // tx):
            for t in T.serial(tx):
                with Ts.sblock("copy"):
                    Ts.where(b * tx + t < copy_length * num_heads * head_dim)
                    vh = Ts.axis.spatial(num_heads, T.Cast("int32", (b * tx + t) // (copy_length * head_dim)))
                    vp = Ts.axis.spatial(copy_length, (b * tx + t) % (copy_length * head_dim) // head_dim)
                    vd = Ts.axis.spatial(head_dim, T.Cast("int32", (b * tx + t) % head_dim))
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]

    return copy_single_page_cpu

def _compact_kv_copy(num_heads, head_dim, dtype, target: Target, page_size: int = 16):
    tx = get_max_num_threads_per_block(target)

    num_pages = T.dynamic("num_pages", "int32")
    total_copy_length = T.dynamic("total_copy_length", "int32")
    copy_length_indptr_elem_offset = T.dynamic("copy_length_indptr_elem_offset", "int32")
    copy_src_dst_pos_elem_offset = T.dynamic("copy_src_dst_pos_elem_offset", "int32")
    pages_elem_offset = T.dynamic("pages_elem_offset", "int64")
    @Ts.function
    def compact_kv_copy(pages: T.Tensor((num_pages, 2, num_heads, page_size, head_dim), dtype, layout=T.TileLayout(T.S[(num_pages, 2, num_heads, page_size, head_dim)] + (pages_elem_offset))), copy_length_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (copy_length_indptr_elem_offset))), copy_src_dst_pos: T.Tensor((2, total_copy_length), 'int32', layout=T.TileLayout(T.S[(2, total_copy_length)] + (copy_src_dst_pos_elem_offset))), batch_size: T.int32):  # noqa: F821
        T.func_attr({"tirx.is_scheduled": True})

        with Ts.sblock("root"):
            for bhd_o in T.parallel((batch_size * num_heads * head_dim + tx - 1) // tx, annotations={"thread_binding": "blockIdx.x"}):
                for bhd_i in T.parallel(tx, annotations={"thread_binding": "threadIdx.x"}):
                    b: T.int32 = (bhd_o * tx + bhd_i) // (num_heads * head_dim)
                    h: T.int32 = (bhd_o * tx + bhd_i) // head_dim % num_heads
                    d: T.int32 = (bhd_o * tx + bhd_i) % head_dim
                    if (bhd_o * tx + bhd_i) < batch_size * num_heads * head_dim:
                        for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // page_size, 0, h, dst_pos % page_size, d] = pages[src_pos // page_size, 0, h, src_pos % page_size, d]
                            pages[dst_pos // page_size, 1, h, dst_pos % page_size, d] = pages[src_pos // page_size, 1, h, src_pos % page_size, d]

    return compact_kv_copy

def _compact_kv_copy_cpu(num_heads, head_dim, dtype, page_size: int = 16):
    tx = 8

    num_pages = T.dynamic("num_pages", "int32")
    total_copy_length = T.dynamic("total_copy_length", "int32")
    copy_length_indptr_elem_offset = T.dynamic("copy_length_indptr_elem_offset", "int32")
    copy_src_dst_pos_elem_offset = T.dynamic("copy_src_dst_pos_elem_offset", "int32")
    @Ts.function
    def compact_kv_copy_cpu(pages: T.Tensor((num_pages, 2, num_heads, page_size, head_dim), dtype), copy_length_indptr: T.Tensor((batch_size + 1,), 'int32', layout=T.TileLayout(T.S[(batch_size + 1,)] + (copy_length_indptr_elem_offset))), copy_src_dst_pos: T.Tensor((2, total_copy_length), 'int32', layout=T.TileLayout(T.S[(2, total_copy_length)] + (copy_src_dst_pos_elem_offset))), batch_size: T.int32):  # noqa: F821
        T.func_attr({"tirx.is_scheduled": True})

        with Ts.sblock("root"):
            for bhd_o in T.serial((batch_size * num_heads * head_dim + tx - 1) // tx):
                for bhd_i in T.serial(tx):
                    b: T.int32 = (bhd_o * tx + bhd_i) // (num_heads * head_dim)
                    h: T.int32 = (bhd_o * tx + bhd_i) // head_dim % num_heads
                    d: T.int32 = (bhd_o * tx + bhd_i) % head_dim
                    if (bhd_o * tx + bhd_i) < batch_size * num_heads * head_dim:
                        for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // page_size, 0, h, dst_pos % page_size, d] = pages[src_pos // page_size, 0, h, src_pos % page_size, d]
                            pages[dst_pos // page_size, 1, h, dst_pos % page_size, d] = pages[src_pos // page_size, 1, h, src_pos % page_size, d]

    return compact_kv_copy_cpu
