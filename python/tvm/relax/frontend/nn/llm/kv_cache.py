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
# ruff: noqa: E501, RUF012
# fmt: off

"""Attention KV cache modeling.

This module exposes the public ``PagedKVCache`` classes (``FlashInferPagedKVCache``
and ``TIRPagedKVCache``). The kernel factories that build the underlying TIR
functions are split across sibling private modules:

- ``_kernel_common``: shared helpers (enums, RoPE, mask, tile allocators,
  ``@T.macro`` bundle, tiling config, scheduling).
- ``_page_kernels``: page management (append, debug, copy, compact).
- ``_prefill_kernels``: prefill attention kernels (paged/ragged/MLA/dense).
- ``_decode_kernels``: decode attention kernels and state-merge helpers.

The private-named kernel factories are re-exported from this module so the
test suite can continue to import them via ``tvm.relax.frontend.nn.llm.kv_cache``.
"""
# pylint: disable=too-many-statements,too-many-arguments,invalid-name,line-too-long
import math
from typing import Any, Literal

import tvm_ffi

import tvm_tirx
from tvm import relax as rx
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax.frontend.nn import Object, Tensor
from tvm_tirx import tirx
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.target import Target

# Re-export enums + kernel factories so existing ``from kv_cache import ...``
# users (test suite, tree_attn.py, mlc-llm, etc.) continue to work after the
# split. These names are referenced in ``__all__`` below to signal to linters
# that the imports are intentional public API (not dead code).
from ._decode_kernels import (
    _attention_decode,
    _attention_decode_cpu,
    _merge_state_inplace,
    _merge_state_inplace_cpu,
)
from ._kernel_common import AttnKind, RopeMode
from ._page_kernels import (
    _compact_kv_copy,
    _compact_kv_copy_cpu,
    _copy_single_page,
    _copy_single_page_cpu,
    _copy_single_page_mla,
    _kv_cache_debug_get_kv,
    _kv_cache_debug_get_kv_mla,
    _kv_cache_transpose_append,
    _kv_cache_transpose_append_mla,
)
from ._prefill_kernels import (
    _attention_prefill,
    _attention_prefill_cpu,
    _attention_prefill_mla,
    _attention_prefill_ragged,
    _attention_prefill_ragged_cpu,
    _attention_sequence_prefill,
    _attention_sequence_prefill_with_mask,
)
from .position_embedding import llama_rope_with_position_map
from .tree_attn import (
    tree_attn,
    tree_attn_cpu,
    tree_attn_with_paged_kv_cache,
    tree_attn_with_paged_kv_cache_cpu,
)

__all__ = [
    "AttnKind",
    "FlashInferPagedKVCache",
    "PagedKVCache",
    "RopeMode",
    "TIRPagedKVCache",
    "_attention_decode",
    "_attention_decode_cpu",
    "_attention_prefill",
    "_attention_prefill_cpu",
    "_attention_prefill_mla",
    "_attention_prefill_ragged",
    "_attention_prefill_ragged_cpu",
    "_attention_sequence_prefill",
    "_attention_sequence_prefill_with_mask",
    "_compact_kv_copy",
    "_compact_kv_copy_cpu",
    "_copy_single_page",
    "_copy_single_page_cpu",
    "_copy_single_page_mla",
    "_kv_cache_debug_get_kv",
    "_kv_cache_debug_get_kv_mla",
    "_kv_cache_transpose_append",
    "_kv_cache_transpose_append_mla",
    "_merge_state_inplace",
    "_merge_state_inplace_cpu",
    "llama_rope_with_position_map",
    "tree_attn",
    "tree_attn_cpu",
    "tree_attn_with_paged_kv_cache",
    "tree_attn_with_paged_kv_cache_cpu",
]


class PagedKVCache(Object):  # pylint: disable=too-few-public-methods
    """The Paged KV Cache used in LLM batching for efficient attention computation."""

    extern_mods: list[tvm_ffi.Module] = []

    def attention_with_fused_qkv(
        self,
        layer_id: int,
        qkv: Tensor,
        num_qo_heads: int,
        sm_scale: float,
    ) -> Tensor:
        """Compute attention with the given fused q/k/v data and in-cache k/v data
        on the specified layer. Rotary position embeddings are applied to k/v
        within this function.

        - For prefill, the input qkv and output tensor have shape
        (1, total_seq_len) for the first two dimensions.
        - For decode, the input qkv and output tensor have shape
        (batch_size, 1) for the first two dimensions.
        - The input qkv have `2 * num_qo_heads + num_kv_heads` at the third dim.
        - The output tensor have `num_qo_heads` at the third dim.
        - The input qkv and output tensor have `head_dim` at the last dim.
        """
        # pylint: disable=protected-access
        b, s, _, d = qkv._expr.ty.shape
        qkv = qkv.reshape(b * s, qkv.shape[2], d)
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_dps_packed(
                    _ExternFunc("vm.builtin.attention_kv_cache_attention_with_fused_qkv"),
                    [
                        self._expr,
                        rx.prim_value(layer_id),  # type: ignore[arg-type]
                        rx.prim_value(sm_scale),
                        qkv._expr,
                    ],
                    ty_args=[rx.TensorType((b * s, num_qo_heads, d), qkv.dtype)],
                )
            )
        ).reshape(b, s, num_qo_heads, d)

    def self_attention(  # pylint: disable=too-many-locals
        self,
        layer_id: int,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        sm_scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Fine-grained API that computes ragged self attention with Q/K/V data."""
        # pylint: disable=protected-access
        b, s, h_qo, d_qk = q._expr.ty.shape
        _, _, h_kv, d_v = v._expr.ty.shape
        q = q.reshape(b * s, h_qo, d_qk)
        k = k.reshape(b * s, h_kv, d_qk)
        v = v.reshape(b * s, h_kv, d_v)
        bb = rx.BlockBuilder.current()
        attn_results = bb.emit(
            rx.call_dps_packed(
                _ExternFunc("vm.builtin.attention_kv_cache_self_attention"),
                [
                    self._expr,
                    rx.prim_value(layer_id),  # type: ignore[arg-type]
                    rx.prim_value(sm_scale),
                    q._expr,
                    k._expr,
                    v._expr,
                ],
                ty_args=[_TupleType([rx.TensorType((b * s, h_qo, d_v), q.dtype), rx.TensorType((b * s, h_qo), "float32")])],
            )
        )
        assert isinstance(attn_results.ty, rx.TupleType)
        assert len(attn_results.ty.fields) == 2
        o = Tensor(_expr=bb.emit(rx.TupleGetItem(attn_results, 0))).reshape(b, s, h_qo, d_v)
        lse = Tensor(_expr=bb.emit(rx.TupleGetItem(attn_results, 1))).reshape(b, s, h_qo)
        return o, lse

    def cross_attention(
        self,
        layer_id: int,
        q: Tensor,
        v_head_dim: int,
        sm_scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Fine-grained API that computes paged cross attention with Q and in-cache KV data."""
        # pylint: disable=protected-access
        b, s, h_qo, d_qk = q._expr.ty.shape
        q = q.reshape(b * s, h_qo, d_qk)
        bb = rx.BlockBuilder.current()
        attn_results = bb.emit(
            rx.call_dps_packed(
                _ExternFunc("vm.builtin.attention_kv_cache_cross_attention"),
                [
                    self._expr,
                    rx.prim_value(layer_id),  # type: ignore[arg-type]
                    rx.prim_value(sm_scale),
                    q._expr,
                ],
                ty_args=[_TupleType([rx.TensorType((b * s, h_qo, v_head_dim), q.dtype), rx.TensorType((b * s, h_qo), "float32")])],
            )
        )
        assert isinstance(attn_results.ty, rx.TupleType)
        assert len(attn_results.ty.fields) == 2
        o = Tensor(_expr=bb.emit(rx.TupleGetItem(attn_results, 0))).reshape(b, s, h_qo, v_head_dim)
        lse = Tensor(_expr=bb.emit(rx.TupleGetItem(attn_results, 1))).reshape(b, s, h_qo)
        return o, lse

    def attention_with_shared_kv(
        self,
        source_layer_id: int,
        q: Tensor,
        current_k: Tensor,
        current_v: Tensor,
        sm_scale: float,
    ) -> Tensor:
        """Compute attention using K/V shared by another logical layer.

        This operation does not append to the KV cache. The cache-owning source layer must run
        before this operation in the active forward pass. ``current_k`` and ``current_v`` are that
        source layer's K/V for the active chunk. During prefill they are combined with the source
        layer's paged past K/V; during decode the source layer has already appended them and the
        operation reads the cache directly.
        """
        # pylint: disable=protected-access
        b, s, h_qo, d_qk = q._expr.ty.shape
        _, _, h_kv, _ = current_k._expr.ty.shape
        _, _, _, d_v = current_v._expr.ty.shape
        q = q.reshape(b * s, h_qo, d_qk)
        current_k = current_k.reshape(b * s, h_kv, d_qk)
        current_v = current_v.reshape(b * s, h_kv, d_v)
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_dps_packed(
                    _ExternFunc("vm.builtin.attention_kv_cache_attention_with_shared_kv"),
                    [
                        self._expr,
                        rx.prim_value(source_layer_id),  # type: ignore[arg-type]
                        rx.prim_value(sm_scale),
                        q._expr,
                        current_k._expr,
                        current_v._expr,
                    ],
                    ty_args=[rx.TensorType((b * s, h_qo, d_v), q.dtype)],
                )
            )
        ).reshape(b, s, h_qo, d_v)

    def append_mla_kv(self, layer_id: int, kv: Tensor) -> "PagedKVCache":
        """Fine-grained API that appends the MLA K/V data to KV cache."""
        # pylint: disable=protected-access
        b, s, _, d_qk = kv._expr.ty.shape
        kv = kv.reshape(b * s, d_qk)
        return PagedKVCache(
            _expr=rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_append_mla_kv",
                self._expr,
                rx.prim_value(layer_id),  # type: ignore[arg-type]
                kv._expr,
                ty_args=rx.AnyType(),
            ),
            _name="paged_kv_cache",
        )

    def merge_attn_output_inplace(
        self,
        o_self_attn: Tensor,
        lse_self_attn: Tensor,
        o_cross_attn: Tensor,
        lse_cross_attn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fine-grained API that merges the attention output from two sources.
        The first two tensors will be inplace updated.
        """
        # pylint: disable=protected-access
        b, s, h_qo, d_v = o_self_attn._expr.ty.shape
        o_self_attn = o_self_attn.reshape(b * s, h_qo, d_v)
        lse_self_attn = lse_self_attn.reshape(b * s, h_qo)
        o_cross_attn = o_cross_attn.reshape(b * s, h_qo, d_v)
        lse_cross_attn = lse_cross_attn.reshape(b * s, h_qo)
        bb = rx.BlockBuilder.current()
        merge_results = bb.emit(
            rx.call_pure_packed(
                "vm.builtin.attention_kv_cache_merge_attn_output_inplace",
                self._expr,
                o_self_attn._expr,
                lse_self_attn._expr,
                o_cross_attn._expr,
                lse_cross_attn._expr,
                ty_args=rx.TupleType(
                    [o_self_attn._expr.ty, lse_self_attn._expr.ty]
                ),
            )
        )
        assert isinstance(merge_results.ty, rx.TupleType)
        assert len(merge_results.ty.fields) == 2
        o_self_attn = Tensor(_expr=bb.emit(rx.TupleGetItem(merge_results, 0))).reshape(
            b, s, h_qo, d_v
        )
        lse_self_attn = Tensor(_expr=bb.emit(rx.TupleGetItem(merge_results, 1))).reshape(b, s, h_qo)
        return o_self_attn, lse_self_attn

    def get_query_positions(self, total_length: tirx.Expr) -> Tensor:
        """Get the in-sequence positions of each slot in the query,
        which are needed for applying positional embeddings in some models.

        Parameters
        ----------
        total_length : tirx.Expr
            The summed-up total sequence length of queries in
            the batch being forwarded.

        Returns
        -------
        q_positions : Tensor
            The in-sequence query positions, in shape `(total_length,)`
        """
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_pure_packed(
                    "vm.builtin.attention_kv_cache_get_query_positions",
                    self._expr,
                    ty_args=rx.TensorType((total_length,), "int32"),
                )
            )
        )

    # pylint: enable=protected-access


def _prepare_yarn_rope_scaling(rope_scaling: dict[str, Any] | None, rope_theta: float | None) -> dict[str, Any] | None:
    """Ensure Yarn-specific scaling configs include the theta metadata."""
    if rope_scaling is None:
        return None
    if rope_scaling.get("rope_type") != "yarn":
        return rope_scaling

    rope_scaling_updated = dict(rope_scaling)
    if "inv_theta_log_scale" not in rope_scaling_updated and rope_theta is not None:
        theta_value = float(rope_theta)
        rope_scaling_updated["inv_theta_log_scale"] = 1.0 / (2 * math.log(theta_value))
    return rope_scaling_updated


class FlashInferPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using FlashInfer (CUDA) kernels."""

    def __init__(  # pylint: disable=too-many-locals
        self,
        attn_kind: Literal["mha", "mla"] | list[Literal["mha", "mla", "mha_sliding"]],
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
        layer_partition: rx.ShapeExpr,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        mla_original_qk_head_dim: int,
        mla_original_v_head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        rope_scaling: dict[str, Any],
        rope_ext_factors: rx.Expr,
        rotary_dim: int,
        enable_disaggregation: bool,
        dtype: str,
        target: Target,
        name: str = "paged_kv_cache",
        *,
        layer_sliding_window_size: int = 1024,
    ) -> None:
        """Create a paged KV cache object with FlashInfer kernels.

        Parameters
        ----------
        max_batch_size : tirx.Var
            The maximum allowed batch size of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        max_total_seq_len : tirx.Var
            The maximum allowed total sequence length of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        prefill_chunk_size : tirx.Var
            The maximum total sequence length in a prefill.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        page_size : tirx.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        support_sliding_window : tirx.Var
            0 or 1, denoting whether the KV cache supports sliding window.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        layer_partition : rx.ShapeExpr
            The KV cache layer partition for pipeline stages.
            It is an indptr array, denoting the starting layer of each pipeline stage.
        rope_mode : RopeMode
            The RoPE mode of the Paged KV cache.
            If it is normal, RoPE will be applied to k before adding k to cache.
            Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        rope_scaling: Dict[str, Any]
            The RoPE scaling information dict.
        rope_ext_factors: rx.Expr
            The RoPE extension factors when "longrope" mode RoPE scaling is enabled.
        rotary_dim : int
            The number of dimensions in the embedding that RoPE is applied to.
        enable_disaggregation : bool
            Whether to enable disaggregation in the KV cache.
        layer_sliding_window_size : int
            Window size used by layers whose attention kind is ``mha_sliding``.
        """
        assert rope_mode != RopeMode.INLINE, "FlashInfer RoPE does not support inline mode."
        rope_scaling = _prepare_yarn_rope_scaling(rope_scaling, rope_theta)

        attn_kind_single = attn_kind[0] if isinstance(attn_kind, list) else attn_kind
        if attn_kind_single == "mha_sliding":
            attn_kind_single = "mha"
        flashinfer_prefill_mods = rx.backend.cuda.flashinfer.gen_flashinfer_prefill_module(
            dtype_q=dtype,
            dtype_kv=dtype,
            dtype_o=dtype,
            qk_head_dim=(qk_head_dim if attn_kind_single == "mha" else mla_original_qk_head_dim),
            v_head_dim=(v_head_dim if attn_kind_single == "mha" else mla_original_v_head_dim),
            enable_inline_rope=False,
            return_static_libs=True,
        )
        flashinfer_decode_mods = (
            rx.backend.cuda.flashinfer.gen_flashinfer_decode_module(
                dtype_q=dtype,
                dtype_kv=dtype,
                dtype_o=dtype,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
                enable_inline_rope=False,
                return_static_libs=True,
            )
            if attn_kind_single == "mha"
            else []
        )
        flashinfer_mla_mods = (
            rx.backend.cuda.flashinfer.gen_flashinfer_mla_module(
                dtype_q=dtype,
                dtype_kv=dtype,
                dtype_o=dtype,
                head_dim_ckv=v_head_dim,
                head_dim_kpe=qk_head_dim - v_head_dim,
                return_static_libs=True,
            )
            if attn_kind_single == "mla"
            else []
        )
        self.extern_mods = flashinfer_prefill_mods + flashinfer_decode_mods + flashinfer_mla_mods

        bb = rx.BlockBuilder.current()
        mha_functions = (
            [
                rx.Tuple([tvm_tirx.ir.StringImm("flashinfer"), rx.ExternFunc("batch_prefill_paged_run"), rx.ExternFunc("batch_prefill_plan")]),
                rx.Tuple([tvm_tirx.ir.StringImm("flashinfer"), rx.ExternFunc("batch_decode_run"), rx.ExternFunc("batch_decode_plan")]),
                rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, target, sliding_window_size=layer_sliding_window_size), "tir_attention_prefill_sliding_window")]),
                rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, target), "tir_attention_decode_sliding_window")]),
                rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(tree_attn_with_paged_kv_cache(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask_with_paged_kv_cache")]),
                rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(tree_attn(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask")]),
            ]
            if attn_kind_single == "mha"
            else [rx.Tuple([]) for _ in range(6)]
        )
        ragged_prefill_function = rx.Tuple([tvm_tirx.ir.StringImm("flashinfer"), rx.ExternFunc("batch_prefill_ragged_run"), rx.ExternFunc("batch_prefill_plan")]) if attn_kind_single == "mha" else rx.Tuple([tvm_tirx.ir.StringImm("flashinfer"), rx.ExternFunc("batch_prefill_ragged_run"), rx.ExternFunc("batch_prefill_plan"), rx.prim_value(mla_original_qk_head_dim), rx.prim_value(mla_original_v_head_dim)])
        mla_function = rx.Tuple([tvm_tirx.ir.StringImm("flashinfer"), rx.ExternFunc("batch_mla_run"), rx.ExternFunc("batch_mla_plan")] if attn_kind_single == "mla" else [])
        attn_merge_functions = [
            bb.add_func(_merge_state_inplace(num_attention_heads, v_head_dim, dtype, target, "tir_attention_merge_state"), "tir_attention_merge_state"),
        ]
        if attn_kind_single == "mla":
            attn_merge_functions.append(bb.add_func(_merge_state_inplace(num_attention_heads, mla_original_v_head_dim, dtype, target, "tir_attention_merge_state_mla"), "tir_attention_merge_state_mla"))

        if isinstance(attn_kind, list):
            attn_kind = [int(getattr(AttnKind, layer_kind.upper())) for layer_kind in attn_kind]
        else:
            attn_kind = [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]

        args = [
            rx.ShapeExpr(
                [
                    max_batch_size,
                    max_total_seq_len,
                    prefill_chunk_size,
                    page_size,
                    support_sliding_window,
                    layer_sliding_window_size,
                ]
            ),
            layer_partition,
            rx.prim_value(num_attention_heads),
            rx.prim_value(num_key_value_heads),
            rx.prim_value(qk_head_dim),
            rx.prim_value(v_head_dim),
            rx.ShapeExpr(attn_kind),
            rx.prim_value(enable_disaggregation),
            rx.prim_value(rope_mode),
            rx.prim_value(rope_scale),
            rx.prim_value(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_kv_cache_transpose_append_mla(qk_head_dim, dtype), "kv_cache_transpose_append_mla"),
            ragged_prefill_function,
            *mha_functions,
            mla_function,
            rx.Tuple(attn_merge_functions),
            bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, qk_head_dim, num_attention_heads, num_key_value_heads, dtype, rope_scaling, rotary_dim), "tir_split_rotary"),
            bb.add_func(_copy_single_page(num_key_value_heads, page_size, qk_head_dim, dtype, target) if attn_kind_single == "mha" else _copy_single_page_mla(page_size, qk_head_dim, dtype, target), "kv_cache_copy_single_page"),
            bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, qk_head_dim, dtype), "kv_cache_debug_get_kv"),
            bb.add_func(_compact_kv_copy(num_key_value_heads, qk_head_dim, dtype, target), "kv_cache_compact_kv_copy"),
        ]
        super().__init__(
            _expr=rx.call_pure_packed(
                "vm.builtin.paged_attention_kv_cache_create",
                *args,
                ty_args=rx.AnyType(),
            ),
            _name=name,
        )


class TIRPagedKVCache(PagedKVCache):  # pylint: disable=too-few-public-methods
    """Paged KV cache using TIR kernels."""

    def __init__(  # pylint: disable=too-many-locals
        self,
        attn_kind: Literal["mha", "mla"] | list[Literal["mha", "mla", "mha_sliding"]],
        max_batch_size: tirx.Var,
        max_total_seq_len: tirx.Var,
        prefill_chunk_size: tirx.Var,
        page_size: tirx.Var,
        support_sliding_window: tirx.Var,
        layer_partition: rx.ShapeExpr,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        mla_original_qk_head_dim: int,
        mla_original_v_head_dim: int,
        rope_mode: RopeMode,
        rope_scale: int,
        rope_theta: int,
        rope_scaling: dict[str, Any],
        rope_ext_factors: rx.Expr,
        rotary_dim: int,
        enable_disaggregation: bool,
        dtype: str,
        target: Target,
        name: str = "paged_kv_cache",
        *,
        layer_sliding_window_size: int = 1024,
    ) -> None:
        """Create a paged KV cache object with TIR kernels.

        Parameters
        ----------
        max_batch_size : tirx.Var
            The maximum allowed batch size of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        max_total_seq_len : tirx.Var
            The maximum allowed total sequence length of the KV cache.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        prefill_chunk_size : tirx.Var
            The maximum total sequence length in a prefill.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        page_size : tirx.Var
            The size (a.k.a. number of tokens) of each page.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        support_sliding_window : tirx.Var
            0 or 1, denoting whether the KV cache supports sliding window.
            It is a symbolic variable whose concrete value is specified
            at runtime.
        layer_partition : rx.ShapeExpr
            The KV cache layer partition for pipeline stages.
            It is an indptr array, denoting the starting layer of each pipeline stage.
        rope_mode : RopeMode
            The RoPE mode of the Paged KV cache.
            If it is normal, RoPE will be applied to k before adding k to cache.
            Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
        rope_scale : int
            The scale of rotary position embedding.
        rope_theta : int
            The base of rotary position embedding.
        rope_scaling: Dict[str, Any]
            The RoPE scaling information dict.
        rope_ext_factors: rx.Expr
            The RoPE extension factors when "longrope" mode RoPE scaling is enabled.
        rotary_dim : int
            The number of dimensions in the embedding that RoPE is applied to.
        enable_disaggregation : bool
            Whether to enable disaggregation in the KV cache.
        target : Target
            The target to build the model to.
        layer_sliding_window_size : int
            Window size used by layers whose attention kind is ``mha_sliding``.
        """
        rope_scaling = _prepare_yarn_rope_scaling(rope_scaling, rope_theta)
        attn_kind_single = attn_kind[0] if isinstance(attn_kind, list) else attn_kind
        if attn_kind_single == "mha_sliding":
            attn_kind_single = "mha"
        if isinstance(attn_kind, list):
            attn_kind = [int(getattr(AttnKind, layer_kind.upper())) for layer_kind in attn_kind]
        else:
            attn_kind = [int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]
        bb = rx.BlockBuilder.current()
        args = [
            rx.ShapeExpr(
                [
                    max_batch_size,
                    max_total_seq_len,
                    prefill_chunk_size,
                    page_size,
                    support_sliding_window,
                    layer_sliding_window_size,
                ]
            ),
            layer_partition,
            rx.prim_value(num_attention_heads),
            rx.prim_value(num_key_value_heads),
            rx.prim_value(qk_head_dim),
            rx.prim_value(v_head_dim),
            rx.ShapeExpr(attn_kind),
            rx.prim_value(enable_disaggregation),
            rx.prim_value(rope_mode),
            rx.prim_value(rope_scale),
            rx.prim_value(rope_theta),
            rope_ext_factors,
            rx.op.zeros((), dtype),
            bb.add_func(_kv_cache_transpose_append(num_key_value_heads, qk_head_dim, dtype), "kv_cache_transpose_append"),
            bb.add_func(_kv_cache_transpose_append_mla(qk_head_dim, dtype), "kv_cache_transpose_append_mla"),
        ]

        if target.kind.name == "llvm":
            if attn_kind_single == "mla":
                raise ValueError("MLA is not supported in TIR kernels for now.")
            args.extend(
                [
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_prefill_ragged_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, v_head_dim, dtype, rope_scaling), "tir_attention_prefill_ragged_cpu")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_prefill_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, False, rope_scaling), "tir_attention_prefill_cpu")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_decode_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, False, rope_scaling), "tir_attention_decode_cpu")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_prefill_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, sliding_window_size=layer_sliding_window_size), "tir_attention_prefill_cpu_sliding_window")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_decode_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling), "tir_attention_decode_cpu_sliding_window")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(tree_attn_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling), "tir_attention_prefill_with_tree_mask_cpu")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(tree_attn_with_paged_kv_cache_cpu(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling), "tir_attention_prefill_with_tree_mask_with_paged_kv_cache_cpu")]),
                    rx.Tuple([]),  # f_mla_prefill
                    rx.Tuple([bb.add_func(_merge_state_inplace_cpu(dtype), "tir_attention_merge_state_cpu")]),
                    bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, qk_head_dim, num_attention_heads, num_key_value_heads, dtype, rope_scaling, rotary_dim), "tir_split_rotary"),
                    bb.add_func(_copy_single_page_cpu(num_key_value_heads, page_size, qk_head_dim, dtype), "kv_cache_copy_single_page_cpu"),
                    bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, qk_head_dim, dtype), "kv_cache_debug_get_kv"),
                    bb.add_func(_compact_kv_copy_cpu(num_key_value_heads, qk_head_dim, dtype), "kv_cache_compact_kv_copy_cpu"),
                ]
            )
        else:
            ragged_qk_head_dim = qk_head_dim if attn_kind_single == "mha" else mla_original_qk_head_dim
            ragged_v_head_dim = v_head_dim if attn_kind_single == "mha" else mla_original_v_head_dim
            args.append(rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_prefill_ragged(num_key_value_heads if attn_kind_single == "mha" else num_attention_heads, num_attention_heads, ragged_qk_head_dim, ragged_v_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_ragged")]))
            mha_functions = (
                [
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, False, rope_scaling, target), "tir_attention_prefill")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, False, rope_scaling, target), "tir_attention_decode")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_prefill(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, target, sliding_window_size=layer_sliding_window_size), "tir_attention_prefill_sliding_window")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_decode(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, True, rope_scaling, target), "tir_attention_decode_sliding_window")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(tree_attn_with_paged_kv_cache(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask_with_paged_kv_cache")]),
                    rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(tree_attn(num_key_value_heads, num_attention_heads, qk_head_dim, dtype, rope_scaling, target), "tir_attention_prefill_with_tree_mask")]),
                ]
                if attn_kind_single == "mha"
                else [rx.Tuple([]) for _ in range(6)]
            )
            mla_function = rx.Tuple([tvm_tirx.ir.StringImm("tirx"), bb.add_func(_attention_prefill_mla(num_attention_heads, v_head_dim, qk_head_dim - v_head_dim, dtype, False, target), "tir_attention_prefill_mla")] if attn_kind_single == "mla" else [])
            attn_merge_functions = [
                bb.add_func(_merge_state_inplace(num_attention_heads, v_head_dim, dtype, target, "tir_attention_merge_state"), "tir_attention_merge_state"),
            ]
            if attn_kind_single == "mla":
                attn_merge_functions.append(bb.add_func(_merge_state_inplace(num_attention_heads, mla_original_v_head_dim, dtype, target, "tir_attention_merge_state_mla"), "tir_attention_merge_state_mla"))
            args.extend(mha_functions)
            args.append(mla_function)
            args.extend(
                [
                    rx.Tuple(attn_merge_functions),
                    bb.add_func(llama_rope_with_position_map(rope_theta, rope_scale, qk_head_dim, num_attention_heads, num_key_value_heads, dtype, rope_scaling, rotary_dim), "tir_split_rotary"),
                    bb.add_func(_copy_single_page(num_key_value_heads, page_size, qk_head_dim, dtype, target) if attn_kind_single == "mha" else _copy_single_page_mla(page_size, qk_head_dim, dtype, target), "kv_cache_copy_single_page"),
                    bb.add_func(_kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, qk_head_dim, dtype), "kv_cache_debug_get_kv"),
                    bb.add_func(_compact_kv_copy(num_key_value_heads, qk_head_dim, dtype, target), "kv_cache_compact_kv_copy"),
                ]
            )

        super().__init__(
            _expr=rx.call_pure_packed(
                "vm.builtin.paged_attention_kv_cache_create",
                *args,
                ty_args=rx.AnyType(),
            ),
            _name=name,
        )
