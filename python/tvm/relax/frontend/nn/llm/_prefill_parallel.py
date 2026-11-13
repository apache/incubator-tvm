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
"""Parallel dot-product prefill for wide FP32 WebGPU attention heads."""

from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T

from ._kernel_common import _get_kv_chunk_len, _get_seq_offset, _length_info_buffer, _rope


def _parallel_prefill_body(
    h_kv,
    h_q,
    rope_scaling,
    paged=False,
    sliding_window=False,
    page_size=16,
    sliding_window_size=1024,
    partitions=1,
):
    # One workgroup handles one query/head. Four groups of 32 threads each
    # reduce a 512-element Q/K dot product, then update the online softmax.
    dtype = "float32"

    def kv_length(length, batch, length_info):
        return (
            _get_kv_chunk_len(length, page_size, batch, length_info, sliding_window)
            if paged
            else length
        )

    def kv_page(indptr, indices, length_info, batch, key):
        if paged:
            slot = _get_seq_offset(key, batch, length_info, sliding_window)
            return indices[indptr[batch] + slot // page_size]
        return indptr[batch] + key

    def kv_offset(length_info, batch, key):
        return _get_seq_offset(key, batch, length_info, sliding_window) % page_size if paged else 0

    def kv_index(page, offset, head, feature, kind):
        return (page, kind, head, offset, feature) if paged else (page, head, feature)

    def load_key(
        source, page, offset, position, head, feature, rotary_mode, rope_scale, rope_theta
    ):
        index = kv_index(page, offset, head, feature, 0)
        return T.if_then_else(
            rotary_mode == 1,
            _rope(source, position, 512, rope_theta, rope_scale, index, dtype, rope_scaling),
            source[index],
        )

    def load_value(source, page, offset, head, feature):
        return source[kv_index(page, offset, head, feature, 1)]

    @T.macro
    def body(
        q,
        q_indptr,
        k,
        v,
        kv_indptr,
        q_rope_position,
        k_rope_pos_offset,
        output,
        lse,
        causal,
        rotary_mode,
        rope_scale,
        rope_theta,
        sm_scale,
        page_values,
        length_info,
        part_idx=0,
    ):
        for row in T.parallel(q.shape[0], annotations={"thread_binding": "blockIdx.x"}):
            for head in T.parallel(h_q, annotations={"thread_binding": "blockIdx.y"}):
                for lane in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("attention"):
                        Ts.reads()
                        Ts.writes()
                        partial = Ts.sblock_alloc_tensor((128,), "float32", scope="shared")
                        probabilities = Ts.sblock_alloc_tensor((4,), "float32", scope="shared")
                        state = Ts.sblock_alloc_tensor((3,), "float32", scope="shared")
                        query = Ts.sblock_alloc_tensor((16,), "float32", scope="local")
                        accum = Ts.sblock_alloc_tensor((4,), "float32", scope="local")
                        score = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                        weights = Ts.sblock_alloc_tensor((4,), "float32", scope="local")
                        denominator = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                        rescale = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                        b = Ts.sblock_alloc_tensor((1,), "int32", scope="local")
                        b[0] = 0
                        while row >= q_indptr[b[0] + 1]:
                            b[0] += 1
                        begin: T.let[T.int32] = kv_indptr[b[0]]
                        count: T.let[T.int32] = kv_length(
                            kv_indptr[b[0] + 1] - begin, b[0], length_info
                        )
                        limit: T.let[T.int32] = T.if_then_else(
                            causal != 0 and not sliding_window,
                            T.min(
                                count,
                                row
                                - q_indptr[b[0]]
                                + count
                                - (q_indptr[b[0] + 1] - q_indptr[b[0]])
                                + 1,
                            ),
                            count,
                        )
                        first: T.let[T.int32] = T.if_then_else(
                            causal != 0 and sliding_window,
                            T.max(
                                count - T.max(sliding_window_size - (row - q_indptr[b[0]]) - 1, 0),
                                0,
                            ),
                            0,
                        )
                        width: T.let[T.int32] = T.ceildiv(T.max(limit - first, 0), partitions)
                        partition_first: T.let[T.int32] = T.min(limit, first + part_idx * width)
                        partition_limit: T.let[T.int32] = T.min(
                            limit, first + (part_idx + 1) * width
                        )
                        kh: T.let[T.int32] = head // (h_q // h_kv)
                        for j in T.unroll(16):
                            feature: T.let[T.int32] = j * 32 + lane % 32
                            query[j] = T.if_then_else(
                                rotary_mode == 1,
                                _rope(
                                    q,
                                    q_rope_position[row],
                                    512,
                                    rope_theta,
                                    rope_scale,
                                    (row, head, feature),
                                    dtype,
                                    rope_scaling,
                                ),
                                q[row, head, feature],
                            )
                        for j in T.unroll(4):
                            accum[j] = 0
                        if lane == 0:
                            state[0] = -50000
                            state[1] = 0
                        T.gpu_storage_sync("shared")
                        for tile in T.serial(T.ceildiv(partition_limit - partition_first, 4)):
                            key: T.let[T.int32] = partition_first + tile * 4 + lane // 32
                            score[0] = 0
                            if key < partition_limit:
                                page: T.let[T.int32] = kv_page(
                                    kv_indptr, page_values, length_info, b[0], key
                                )
                                offset: T.let[T.int32] = kv_offset(length_info, b[0], key)
                                for j in T.unroll(16):
                                    feature: T.let[T.int32] = j * 32 + lane % 32
                                    score[0] += query[j] * load_key(
                                        k,
                                        page,
                                        offset,
                                        k_rope_pos_offset[b[0]] + key,
                                        kh,
                                        feature,
                                        rotary_mode,
                                        rope_scale,
                                        rope_theta,
                                    )
                            partial[lane] = score[0]
                            T.gpu_storage_sync("shared")
                            for reduction in T.unroll(5):
                                width: T.let[T.int32] = 16 >> reduction
                                if lane % 32 < width:
                                    partial[lane] += partial[lane + width]
                                T.gpu_storage_sync("shared")
                            if lane == 0:
                                score[0] = state[0]
                                for j in T.unroll(4):
                                    weights[j] = T.if_then_else(
                                        partition_first + tile * 4 + j < partition_limit,
                                        partial[j * 32] * sm_scale * 1.4426950408889634,
                                        T.float32(-50000),
                                    )
                                    score[0] = T.max(score[0], weights[j])
                                rescale[0] = T.exp2(state[0] - score[0])
                                denominator[0] = state[1] * rescale[0]
                                for j in T.unroll(4):
                                    weights[j] = T.if_then_else(
                                        partition_first + tile * 4 + j < partition_limit,
                                        T.exp2(weights[j] - score[0]),
                                        T.float32(0),
                                    )
                                    denominator[0] += weights[j]
                            T.gpu_storage_sync("shared")
                            if lane == 0:
                                for j in T.unroll(4):
                                    probabilities[j] = weights[j]
                                state[0] = score[0]
                                state[1] = denominator[0]
                                state[2] = rescale[0]
                            T.gpu_storage_sync("shared")
                            for feature in T.unroll(4):
                                accum[feature] *= state[2]
                                for j in T.unroll(4):
                                    if partition_first + tile * 4 + j < partition_limit:
                                        page: T.let[T.int32] = kv_page(
                                            kv_indptr,
                                            page_values,
                                            length_info,
                                            b[0],
                                            partition_first + tile * 4 + j,
                                        )
                                        offset: T.let[T.int32] = kv_offset(
                                            length_info, b[0], partition_first + tile * 4 + j
                                        )
                                        accum[feature] += probabilities[j] * load_value(
                                            v, page, offset, kh, feature * 128 + lane
                                        )
                            T.gpu_storage_sync("shared")
                        for feature in T.unroll(4):
                            output[row, head, feature * 128 + lane] = T.if_then_else(
                                state[1] > 0, accum[feature] / state[1], T.float32(0)
                            )
                        if lane == 0:
                            lse[row, head] = T.if_then_else(
                                state[1] > 0, state[0] + T.log2(state[1]), T.float32(-50000)
                            )

    return body


def _attention_prefill_ragged_parallel(h_kv, h_q, rope_scaling):
    batch, qlen, klen = [T.dynamic(x, "int32") for x in ["batch_size", "qo_len", "kv_len"]]
    qi, ki, qp, kp = [
        T.dynamic(x, "int32")
        for x in [
            "q_indptr_elem_offset",
            "kv_indptr_elem_offset",
            "q_rope_position_elem_offset",
            "k_rope_pos_offset_elem_offset",
        ]
    ]
    body = _parallel_prefill_body(h_kv, h_q, rope_scaling)

    @Ts.function
    def batch_prefill_ragged_kv(
        q: T.Tensor((qlen, h_q, 512), "float32"),
        q_indptr: T.Tensor((batch + 1,), "int32", layout=T.TileLayout(T.S[(batch + 1,)] + (qi))),
        k: T.Tensor((klen, h_kv, 512), "float32"),
        v: T.Tensor((klen, h_kv, 512), "float32"),
        kv_indptr: T.Tensor((batch + 1,), "int32", layout=T.TileLayout(T.S[(batch + 1,)] + (ki))),
        q_rope_position: T.Tensor((qlen,), "int32", layout=T.TileLayout(T.S[(qlen,)] + (qp))),
        k_rope_pos_offset: T.Tensor((batch,), "int32", layout=T.TileLayout(T.S[(batch,)] + (kp))),
        output: T.Tensor((qlen, h_q, 512), "float32"),
        lse: T.Tensor((qlen, h_q), "float32"),
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        body(
            q,
            q_indptr,
            k,
            v,
            kv_indptr,
            q_rope_position,
            k_rope_pos_offset,
            output,
            lse,
            causal,
            rotary_mode,
            rope_scale,
            rope_theta,
            sm_scale,
            kv_indptr,
            kv_indptr,
        )

    return batch_prefill_ragged_kv.with_attr("tirx.is_scheduled", True)


def _attention_prefill_paged_parallel(
    h_kv, h_q, rope_scaling, sliding_window, page_size=16, sliding_window_size=1024
):
    if not sliding_window:
        return _attention_prefill_paged_split(
            h_kv, h_q, rope_scaling, sliding_window, page_size, sliding_window_size
        )
    batch, qlen, num_pages, nnz_pages = [
        T.dynamic(x, "int32") for x in ["batch_size", "qo_len", "num_pages", "nnz_pages"]
    ]
    qi, pi, pv, li, qp, kp, po = [
        T.dynamic(x, "int32")
        for x in [
            "q_indptr_elem_offset",
            "page_indptr_elem_offset",
            "page_values_elem_offset",
            "length_info_elem_offset",
            "q_rope_position_elem_offset",
            "k_rope_pos_offset_elem_offset",
            "pages_elem_offset",
        ]
    ]
    body = _parallel_prefill_body(
        h_kv, h_q, rope_scaling, True, sliding_window, page_size, sliding_window_size
    )
    symbol = "batch_prefill_paged_kv_sliding_window" if sliding_window else "batch_prefill_paged_kv"

    @Ts.function
    def batch_prefill_paged_kv(
        q: T.Tensor((qlen, h_q, 512), "float32"),
        q_indptr: T.Tensor((batch + 1,), "int32", layout=T.TileLayout(T.S[(batch + 1,)] + (qi))),
        pages: T.Tensor(
            (num_pages, 2, h_kv, page_size, 512),
            "float32",
            layout=T.TileLayout(T.S[(num_pages, 2, h_kv, page_size, 512)] + (po)),
        ),
        page_indptr: T.Tensor((batch + 1,), "int32", layout=T.TileLayout(T.S[(batch + 1,)] + (pi))),
        page_values: T.Tensor((nnz_pages,), "int32", layout=T.TileLayout(T.S[(nnz_pages,)] + (pv))),
        length_info: _length_info_buffer(batch, sliding_window, li),
        k_rope_pos_offset: T.Tensor((batch,), "int32", layout=T.TileLayout(T.S[(batch,)] + (kp))),
        q_rope_position: T.Tensor((qlen,), "int32", layout=T.TileLayout(T.S[(qlen,)] + (qp))),
        output: T.Tensor((qlen, h_q, 512), "float32"),
        lse: T.Tensor((qlen, h_q), "float32"),
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"global_symbol": symbol})
        body(
            q,
            q_indptr,
            pages,
            pages,
            page_indptr,
            q_rope_position,
            k_rope_pos_offset,
            output,
            lse,
            causal,
            rotary_mode,
            rope_scale,
            rope_theta,
            sm_scale,
            page_values,
            length_info,
        )

    return batch_prefill_paged_kv.with_attr("tirx.is_scheduled", True)


def _attention_prefill_paged_split(
    h_kv, h_q, rope_scaling, sliding_window, page_size=16, sliding_window_size=1024
):
    batch, qlen, num_pages, nnz_pages = [
        T.dynamic(x, "int32") for x in ["batch_size", "qo_len", "num_pages", "nnz_pages"]
    ]
    qi, pi, pv, li, qp, kp, po = [
        T.dynamic(x, "int32")
        for x in [
            "q_indptr_elem_offset",
            "page_indptr_elem_offset",
            "page_values_elem_offset",
            "length_info_elem_offset",
            "q_rope_position_elem_offset",
            "k_rope_pos_offset_elem_offset",
            "pages_elem_offset",
        ]
    ]
    body = _parallel_prefill_body(
        h_kv, h_q, rope_scaling, True, False, page_size, sliding_window_size, partitions=4
    )
    symbol = "batch_prefill_paged_kv_sliding_window" if sliding_window else "batch_prefill_paged_kv"

    batch_prefill_paged_kv_part_output_offset = T.dynamic(
        "batch_prefill_paged_kv_part_output_offset", "int32"
    )
    batch_prefill_paged_kv_part_lse_offset = T.dynamic(
        "batch_prefill_paged_kv_part_lse_offset", "int32"
    )

    @Ts.function
    def batch_prefill_paged_kv(
        q: T.Tensor((qlen, h_q, 512), "float32"),
        q_indptr: T.Tensor((batch + 1,), "int32", layout=T.TileLayout(T.S[(batch + 1,)] + (qi))),
        pages: T.Tensor(
            (num_pages, 2, h_kv, page_size, 512),
            "float32",
            layout=T.TileLayout(T.S[(num_pages, 2, h_kv, page_size, 512)] + (po)),
        ),
        page_indptr: T.Tensor((batch + 1,), "int32", layout=T.TileLayout(T.S[(batch + 1,)] + (pi))),
        page_values: T.Tensor((nnz_pages,), "int32", layout=T.TileLayout(T.S[(nnz_pages,)] + (pv))),
        length_info: _length_info_buffer(batch, sliding_window, li),
        k_rope_pos_offset: T.Tensor((batch,), "int32", layout=T.TileLayout(T.S[(batch,)] + (kp))),
        q_rope_position: T.Tensor((qlen,), "int32", layout=T.TileLayout(T.S[(qlen,)] + (qp))),
        output: T.Tensor((qlen, h_q, 512), "float32"),
        lse: T.Tensor((qlen, h_q), "float32"),
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"global_symbol": symbol})
        # Launch each KV partition separately so long contexts do not produce
        # one long-running dispatch. Merge their softmax-weighted outputs below.
        temporary = Ts.sblock_alloc_tensor(
            (T.int64(4), T.Cast("int64", qlen), T.int64(h_q), T.int64(512)), "float32"
        )
        temporary_lse = Ts.sblock_alloc_tensor(
            (T.int64(4), T.Cast("int64", qlen), T.int64(h_q)), "float32"
        )
        for part_idx in T.serial(4):
            with Ts.sblock("part_idx"):
                part_output = Ts.match_tensor(
                    temporary[part_idx, 0:qlen, 0:h_q, 0:512],
                    (qlen, h_q, 512),
                    layout=T.TileLayout(
                        T.S[(qlen, h_q, 512)] + (batch_prefill_paged_kv_part_output_offset)
                    ),
                )
                part_lse = Ts.match_tensor(
                    temporary_lse[part_idx, 0:qlen, 0:h_q],
                    (qlen, h_q),
                    layout=T.TileLayout(
                        T.S[(qlen, h_q)] + (batch_prefill_paged_kv_part_lse_offset)
                    ),
                )
                body(
                    q,
                    q_indptr,
                    pages,
                    pages,
                    page_indptr,
                    q_rope_position,
                    k_rope_pos_offset,
                    part_output,
                    part_lse,
                    causal,
                    rotary_mode,
                    rope_scale,
                    rope_theta,
                    sm_scale,
                    page_values,
                    length_info,
                    part_idx,
                )
        for row in T.parallel(qlen, annotations={"thread_binding": "blockIdx.x"}):
            for head in T.parallel(h_q, annotations={"thread_binding": "blockIdx.y"}):
                for lane in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("merge"):
                        Ts.reads()
                        Ts.writes()
                        weights = Ts.sblock_alloc_tensor((4,), "float32", scope="shared")
                        local_weights = Ts.sblock_alloc_tensor((4,), "float32", scope="local")
                        maximum = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                        denominator = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                        value = Ts.sblock_alloc_tensor((1,), "float32", scope="local")
                        if lane == 0:
                            maximum[0] = T.float32(-50000)
                            for part in T.unroll(4):
                                local_weights[part] = temporary_lse[part, row, head]
                                maximum[0] = T.max(maximum[0], local_weights[part])
                            denominator[0] = 0
                            for part in T.unroll(4):
                                local_weights[part] = T.if_then_else(
                                    local_weights[part] > T.float32(-49999),
                                    T.exp2(local_weights[part] - maximum[0]),
                                    T.float32(0),
                                )
                                denominator[0] += local_weights[part]
                            for part in T.unroll(4):
                                weights[part] = local_weights[part] / T.max(
                                    denominator[0], T.float32(1)
                                )
                            lse[row, head] = T.if_then_else(
                                denominator[0] > 0,
                                maximum[0] + T.log2(denominator[0]),
                                T.float32(-50000),
                            )
                        T.gpu_storage_sync("shared")
                        for feature in T.unroll(4):
                            value[0] = 0
                            for part in T.unroll(4):
                                value[0] += (
                                    weights[part] * temporary[part, row, head, feature * 128 + lane]
                                )
                            output[row, head, feature * 128 + lane] = value[0]

    # Let the caller assign a symbol when scratch buffers are lifted.
    return batch_prefill_paged_kv.without_attr("global_symbol").with_attr("tirx.is_scheduled", True)
