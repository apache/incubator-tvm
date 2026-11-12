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

"""Operators for positional embeddings, e.g. RoPE."""

import math
from collections.abc import Callable
from functools import partial
from typing import Any

from tvm.relax.frontend.nn import Tensor, op
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# pylint: disable=invalid-name


def rope_freq_default(s: tirx.Var, d: tirx.Var, d_range: int, theta: float, dtype: str):
    """Compute the inverse frequency of RoPE and then return the cosine and sine of it.

    Parameters
    ----------
    s : tirx.Var
        The position index.

    d : tirx.Var
        The dimension index.

    d_range : int
        The maximum dimension index.

    theta : float
        The theta value in RoPE, which controls the frequency.

    dtype : str
        The data type of the output.

    Returns
    -------
    cos_freq : Tensor
        The cosine of the inverse frequency.

    sin_freq : Tensor
        The sine of the inverse frequency.

    var_map: Dict[tirx.Var, tirx.Expr]
        The common expression map.
    """
    freq = s / tirx.power(theta, d * 2 % d_range / tirx.const(d_range, "float32"))
    freq_var = tirx.Var("freq", "float32")
    cos_freq = tirx.cos(freq_var).astype(dtype)
    sin_freq = tirx.sin(freq_var).astype(dtype)
    return cos_freq, sin_freq, {freq_var: freq}


def rope_freq_gptj(s: tirx.Var, d: tirx.Var, d_range: int, theta: float, dtype: str):
    """Compute the inverse frequency of RoPE for gptj RoPE scaling."""
    freq = s / tirx.power(theta, 2 * (d // 2) % d_range / tirx.const(d_range, "float32"))
    freq_var = tirx.Var("freq", "float32")
    cos_freq = tirx.cos(freq_var).astype(dtype)
    sin_freq = tirx.sin(freq_var).astype(dtype)
    return cos_freq, sin_freq, {freq_var: freq}


def rope_freq_llama4(  # pylint: disable=too-many-arguments,too-many-locals
    s: tirx.Var,
    d: tirx.Var,
    d_range: int,
    theta: float,
    dtype: str,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: float,
):
    """Compute the inverse frequency of RoPE for llama4 RoPE scaling."""
    orig_freq = tirx.const(1, "float32") / tirx.power(
        theta, 2 * (d // 2) / tirx.const(d_range, "float32")
    )
    orig_freq_var = tirx.Var("orig_freq", "float32")

    llama4_inv_scaling_factor = 1.0 / factor

    if high_freq_factor == low_freq_factor:
        wavelength = tirx.const(2 * math.pi, "float32") / orig_freq_var
        threshold_wavelen = tirx.const(
            original_max_position_embeddings / low_freq_factor, "float32"
        )

        scaled_freq = tirx.if_then_else(
            wavelength > threshold_wavelen, orig_freq_var / factor, orig_freq_var
        )
        smoothed_freq = s * scaled_freq

    else:
        # Original smooth interpolation logic
        inv_diff_freq_factor = 1.0 / (high_freq_factor - low_freq_factor)

        llama4_alpha = original_max_position_embeddings / (2 * math.pi) * inv_diff_freq_factor
        llama4_beta = low_freq_factor * inv_diff_freq_factor
        smooth = tirx.max(0.0, tirx.min(1.0, llama4_alpha * orig_freq_var - llama4_beta))
        smoothed_freq = s * (
            (1.0 - smooth) * orig_freq_var * llama4_inv_scaling_factor + smooth * orig_freq_var
        )

    smoothed_freq_var = tirx.Var("smoothed_freq", "float32")
    cos_freq = tirx.cos(smoothed_freq_var).astype(dtype)
    sin_freq = tirx.sin(smoothed_freq_var).astype(dtype)
    return (
        cos_freq,
        sin_freq,
        {smoothed_freq_var: smoothed_freq, orig_freq_var: orig_freq},
    )


def rope_freq_llama3(  # pylint: disable=too-many-arguments,too-many-locals
    s: tirx.Var,
    d: tirx.Var,
    d_range: int,
    theta: float,
    dtype: str,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: float,
):
    """Compute the inverse frequency of RoPE for llama3 RoPE scaling."""
    orig_freq = tirx.const(1, "float32") / tirx.power(
        theta, d * 2 % d_range / tirx.const(d_range, "float32")
    )
    orig_freq_var = tirx.Var("orig_freq", "float32")
    inv_diff_freq_factor = 1.0 / (high_freq_factor - low_freq_factor)
    llama3_inv_scaling_factor = 1.0 / factor
    llama3_alpha = original_max_position_embeddings / (2 * math.pi) * inv_diff_freq_factor
    llama3_beta = low_freq_factor * inv_diff_freq_factor
    smooth = tirx.max(0.0, tirx.min(1.0, llama3_alpha * orig_freq_var - llama3_beta))
    smoothed_freq = s * (
        (1.0 - smooth) * orig_freq_var * llama3_inv_scaling_factor + smooth * orig_freq_var
    )
    smoothed_freq_var = tirx.Var("smoothed_freq", "float32")
    cos_freq = tirx.cos(smoothed_freq_var).astype(dtype)
    sin_freq = tirx.sin(smoothed_freq_var).astype(dtype)
    return (
        cos_freq,
        sin_freq,
        {smoothed_freq_var: smoothed_freq, orig_freq_var: orig_freq},
    )


def rope_freq_longrope(  # pylint: disable=too-many-arguments
    s: tirx.Var,
    d: tirx.Var,
    d_range: int,
    theta: float,
    dtype: str,
    max_position_embeddings: int,
    original_max_position_embeddings: int,
    ext_factors: T.Tensor | None = None,
):
    """Compute the inverse frequency of RoPE for longrope scaling."""
    scale = max_position_embeddings / original_max_position_embeddings
    scaling_factor = (
        math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))
        if scale > 1.0
        else 1.0
    )
    divisor = tirx.power(theta, d * 2 % d_range / tirx.const(d_range, "float32"))
    if ext_factors is not None:
        divisor = ext_factors[d % (d_range // 2)] * divisor
    freq = s / divisor
    freq_var = tirx.Var("freq", "float32")
    cos_freq = (tirx.cos(freq_var) * scaling_factor).astype(dtype)
    sin_freq = (tirx.sin(freq_var) * scaling_factor).astype(dtype)
    return cos_freq, sin_freq, {freq_var: freq}


def yarn_find_correction_dim(
    num_rotations: int,
    d: tirx.Var,
    max_position_embeddings: int,
    inv_theta_log_scale: float | tirx.Expr | None = None,
):
    """Inverse dim formula to find dim based on number of rotations"""
    return (
        d * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) * inv_theta_log_scale
    )


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    d: tirx.Var,
    max_position_embeddings: int,
    inv_theta_log_scale: float | tirx.Expr | None = None,
):
    """Find the correction range based on the number of rotations"""
    low = yarn_find_correction_dim(
        low_rot, d, max_position_embeddings, inv_theta_log_scale=inv_theta_log_scale
    )
    high = yarn_find_correction_dim(
        high_rot, d, max_position_embeddings, inv_theta_log_scale=inv_theta_log_scale
    )
    return tirx.max(low, 0), tirx.min(high, d - 1)


def rope_freq_yarn(
    s: tirx.Var,
    d: tirx.Var,
    d_range: int,
    theta: float | tirx.Expr,
    dtype: str,
    original_max_position_embeddings: int,
    scaling_factor: float,
    beta_fast: int,
    beta_slow: int,
    inv_theta_log_scale: float | tirx.Expr | None = None,
):  # pylint: disable=too-many-arguments, too-many-locals
    """Compute the inverse frequency of RoPE for yarn RoPE scaling."""

    exponent = d * 2 % d_range / tirx.const(d_range, "float32")
    freq_power = tirx.power(theta, exponent)
    freq_extra = tirx.const(1, "float32") / freq_power
    freq_inter = tirx.const(1, "float32") / (scaling_factor * freq_power)

    low, high = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        d_range,
        original_max_position_embeddings,
        inv_theta_log_scale=inv_theta_log_scale,
    )
    high = tirx.if_then_else(low == high, high + 0.001, high)
    # Both coordinates of a rotary pair need the same interpolation weight.
    inv_freq_mask = tirx.const(1, "float32") - tirx.max(
        tirx.min((d % (d_range // 2) - low) / (high - low), 1.0), 0.0
    ).astype("float32")
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    freq = s * inv_freq
    freq_var = tirx.Var("freq", "float32")
    cos_freq = tirx.cos(freq_var).astype(dtype)
    sin_freq = tirx.sin(freq_var).astype(dtype)
    return cos_freq, sin_freq, {freq_var: freq}


def switch_rope_freq_func(rope_scaling: dict[str, Any]) -> Callable:
    """Return the RoPE inverse frequency computation function based
    on the given RoPE scaling.
    """
    if "rope_type" not in rope_scaling:
        return rope_freq_default
    if rope_scaling["rope_type"] == "gptj":
        return rope_freq_gptj
    if rope_scaling["rope_type"] == "llama3":
        return partial(
            rope_freq_llama3,
            factor=rope_scaling["factor"],
            low_freq_factor=rope_scaling["low_freq_factor"],
            high_freq_factor=rope_scaling["high_freq_factor"],
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
        )
    if rope_scaling["rope_type"] == "llama4":
        return partial(
            rope_freq_llama4,
            factor=rope_scaling["factor"],
            low_freq_factor=rope_scaling["low_freq_factor"],
            high_freq_factor=rope_scaling["high_freq_factor"],
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
        )
    if rope_scaling["rope_type"] == "longrope":
        return partial(
            rope_freq_longrope,
            max_position_embeddings=rope_scaling["max_position_embeddings"],
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
        )
    if rope_scaling["rope_type"] == "yarn":
        inv_theta_log_scale = rope_scaling.get("inv_theta_log_scale")
        assert inv_theta_log_scale is not None, "inv_theta_log_scale must be precomputed for YaRN"
        return partial(
            rope_freq_yarn,
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            scaling_factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
            inv_theta_log_scale=inv_theta_log_scale,
        )
    raise ValueError(f"Unsupported RoPE scaling type: {rope_scaling['rope_type']}")


# mypy: disable-error-code="attr-defined"


def llama_rope(  # pylint: disable=too-many-arguments
    qkv: Tensor,
    total_seq_len: tirx.Var,
    theta: float,
    scale: float,
    num_q_heads: int,
    num_kv_heads: int,
    rope_scaling: dict[str, Any],
    rotary_dim: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Llama-style RoPE. Given a fused QKV tensor, it returns three tensors, Q, K, and V, where Q
    and K are rotated by RoPE while V remains unchanged.

    Parameters
    ----------
    qkv : Tensor
        The fused QKV tensor of shape: [batch_size, seq_len, #q_heads + #kv_heads * 2, head_dim]

    total_seq_len : tirx.Var
        The total sequence length after being concatenated with KVCache. It is used to compute the
        offset of RoPE.

    theta : float
        The theta value, or "base" in RoPE, which controls the frequency.

    scale : float
        The RoPE scaling factor.

    num_q_heads : int
        The number of query heads.

    num_kv_heads : int
        The number of key/value heads. It differs from `num_q_heads` in group-query attention.

    rope_scaling : Dict
        The configuration of RoPE scaling.

    rotary_dim : Optional[int]
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.

    Returns
    -------
    q : Tensor
        The query tensor of shape [batch_size, seq_len, #q_heads, head_dim] w/ RoPE applied

    k : Tensor
        The key tensor of shape [batch_size, seq_len, #kv_heads, head_dim] w/ RoPE applied

    v : Tensor
        The value tensor of shape [batch_size, seq_len, #kv_heads, head_dim] w/o RoPE applied
    """
    _, _, fused_heads, head_dim = qkv.shape
    assert fused_heads == num_q_heads + num_kv_heads * 2
    if rotary_dim is None:
        rotary_dim = head_dim
    dtype = qkv.dtype
    scale = tirx.const(scale, dtype)

    def _rope(  # pylint: disable=too-many-arguments
        x: T.Tensor,
        b: tirx.Var,
        s: tirx.Var,
        h: tirx.Var,
        d: tirx.Var,
        offset: tirx.Var,
    ):
        cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(
            (s + offset) * scale, d, rotary_dim, theta, dtype
        )
        cos = cos_freq * x[b, s, h, d]
        if rope_scaling["rope_type"] == "gptj":
            sin = sin_freq * tirx.if_then_else(
                d % 2 == 0,
                -x[b, s, h, d + 1],
                x[b, s, h, d - 1],
            )
        else:
            sin = sin_freq * tirx.if_then_else(
                d < rotary_dim // 2,
                -x[b, s, h, d + rotary_dim // 2],
                x[b, s, h, d - rotary_dim // 2],
            )
        expr = cos + sin
        for var, value in var_map.items():
            expr = tirx.Let(var, value, expr)
        return expr

    batch_size = T.dynamic("batch_size", "int64")
    seq_len = T.dynamic("seq_len", "int64")

    @Ts.function(private=True)
    def fused_rope(  # pylint: disable=too-many-locals
        qkv: T.Tensor((batch_size, seq_len, fused_heads, head_dim), dtype),
        q: T.Tensor((batch_size, seq_len, num_q_heads, head_dim), dtype),
        k: T.Tensor((batch_size, seq_len, num_kv_heads, head_dim), dtype),
        v: T.Tensor((batch_size, seq_len, num_kv_heads, head_dim), dtype),
        total_seq_len: T.int64,
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tirx.noalias": True,
            }
        )

        for iters in T.grid(batch_size, seq_len, fused_heads, head_dim):
            with Ts.sblock("llama_fused_rope"):
                b, s, h, d = Ts.axis.remap("SSSS", iters)
                if h < num_q_heads:
                    q[b, s, h, d] = T.if_then_else(
                        d < rotary_dim,
                        _rope(qkv, b, s, h, d, total_seq_len - seq_len),
                        qkv[b, s, h, d],
                    )
                elif h < num_q_heads + num_kv_heads:
                    k[b, s, h - num_q_heads, d] = T.if_then_else(
                        d < rotary_dim,
                        _rope(qkv, b, s, h, d, total_seq_len - seq_len),
                        qkv[b, s, h, d],
                    )
                else:
                    v[b, s, h - (num_q_heads + num_kv_heads), d] = qkv[b, s, h, d]

    b, s, _, _ = qkv.shape
    return op.tensor_ir_op(  # pylint: disable=no-member
        fused_rope,
        "llama_rope",
        args=[qkv, total_seq_len],
        out=(
            Tensor.placeholder((b, s, num_q_heads, head_dim), dtype),
            Tensor.placeholder((b, s, num_kv_heads, head_dim), dtype),
            Tensor.placeholder((b, s, num_kv_heads, head_dim), dtype),
        ),
    )


def llama_rope_with_position_map(  # pylint: disable=too-many-arguments
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
    rope_scaling: dict[str, Any],
    rotary_dim: int | None = None,
):
    """Return the TIR function that computes Llama-style RoPE with q position map.

    Parameters
    ----------
    theta : float
        The theta value, or "base" in RoPE, which controls the frequency.

    scale : float
        The RoPE scaling factor.

    head_dim : int
        The number of features on each head.

    num_q_heads : int
        The number of query heads.

    num_kv_heads : int
        The number of key/value heads. It differs from `num_q_heads` in group-query attention.

    dtype : str
        The dtype of qkv data.

    rope_scaling : Dict
        The configuration of RoPE scaling.

    rotary_dim : int
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.
    """
    fused_heads = num_q_heads + num_kv_heads * 2
    if rotary_dim is None:
        rotary_dim = head_dim
    scale = tirx.const(scale, "float32")
    is_longrope_scaling = rope_scaling.get("rope_type") == "longrope"
    if is_longrope_scaling and "original_max_position_embeddings" in rope_scaling:
        original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
    else:
        original_max_position_embeddings = 0

    def _rope(  # pylint: disable=too-many-arguments
        x: T.Tensor,
        s: tirx.Var,
        h: tirx.Var,
        d: tirx.Var,
        pos: tirx.Var,
        ext_factors: T.Tensor | None = None,
    ):
        kwargs = {}
        if ext_factors is not None:
            kwargs["ext_factors"] = ext_factors
        cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(
            pos * scale, d, rotary_dim, theta, "float32", **kwargs
        )
        cos = cos_freq * x[s, h, d].astype("float32")
        if "rope_type" in rope_scaling and rope_scaling["rope_type"] == "gptj":
            sin = sin_freq * tirx.if_then_else(
                d % 2 == 0,
                -x[s, h, d + 1],
                x[s, h, d - 1],
            ).astype("float32")
        else:
            sin = sin_freq * tirx.if_then_else(
                d < rotary_dim // 2,
                -x[s, h, d + rotary_dim // 2],
                x[s, h, d - rotary_dim // 2],
            ).astype("float32")
        expr = (cos + sin).astype(dtype)
        for var, value in var_map.items():
            expr = tirx.Let(var, value, expr)
        return expr

    seq_len = T.dynamic("seq_len", "int32")
    position_map_elem_offset = T.dynamic("position_map_elem_offset", "int32")

    @Ts.function
    def fused_rope(  # pylint: disable=too-many-locals
        qkv: T.Tensor((seq_len, fused_heads, head_dim), dtype),
        position_map: T.Tensor(
            (seq_len,), "int32", layout=T.TileLayout(T.S[(seq_len,)] + (position_map_elem_offset))
        ),
        q: T.Tensor((seq_len, num_q_heads, head_dim), dtype),
        k: T.Tensor((seq_len, num_kv_heads, head_dim), dtype),
        v: T.Tensor((seq_len, num_kv_heads, head_dim), dtype),
        apply_rope: T.int64,
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tirx.noalias": True,
            }
        )

        for iters in T.grid(seq_len, fused_heads, head_dim):
            with Ts.sblock("llama_fused_rope"):
                s, h, d = Ts.axis.remap("SSS", iters)
                if h < num_q_heads:
                    q[s, h, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                elif h < num_q_heads + num_kv_heads:
                    k[s, h - num_q_heads, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                else:
                    v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]

    seq_len = T.dynamic("seq_len", "int64")
    position_map_elem_offset = T.dynamic("position_map_elem_offset", "int64")

    @Ts.function
    def fused_rope_longrope_scaling(  # pylint: disable=too-many-locals
        qkv: T.Tensor((seq_len, fused_heads, head_dim), dtype),
        position_map: T.Tensor(
            (seq_len,), "int32", layout=T.TileLayout(T.S[(seq_len,)] + (position_map_elem_offset))
        ),
        q: T.Tensor((seq_len, num_q_heads, head_dim), dtype),
        k: T.Tensor((seq_len, num_kv_heads, head_dim), dtype),
        v: T.Tensor((seq_len, num_kv_heads, head_dim), dtype),
        ext_factors: T.Tensor((rotary_dim,), "float32"),  # type: ignore
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tirx.noalias": True,
            }
        )

        # long factors is the first half, short factors is the second half
        long_factors = T.decl_tensor(
            ext_factors.data,
            ty_args=[T.Tensor((rotary_dim // 2,), "float32", data=ext_factors.data)],
        )
        short_factors = T.decl_tensor(
            ext_factors.data,
            ty_args=[
                T.Tensor(
                    (rotary_dim // 2,),
                    "float32",
                    layout=T.TileLayout(T.S[(rotary_dim // 2,)] + (rotary_dim // 2)),
                    data=ext_factors.data,
                )
            ],
        )

        if seq_len > original_max_position_embeddings:
            for iters in T.grid(seq_len, fused_heads, head_dim):
                with Ts.sblock("llama_fused_rope"):
                    s, h, d = Ts.axis.remap("SSS", iters)
                    if h < num_q_heads:
                        q[s, h, d] = T.if_then_else(
                            d < rotary_dim,
                            _rope(
                                qkv,
                                s,
                                h,
                                d,
                                position_map[s],
                                long_factors if is_longrope_scaling else None,
                            ),
                            qkv[s, h, d],
                        )
                    elif h < num_q_heads + num_kv_heads:
                        k[s, h - num_q_heads, d] = T.if_then_else(
                            d < rotary_dim,
                            _rope(
                                qkv,
                                s,
                                h,
                                d,
                                position_map[s],
                                long_factors if is_longrope_scaling else None,
                            ),
                            qkv[s, h, d],
                        )
                    else:
                        v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]
        else:
            for iters in T.grid(seq_len, fused_heads, head_dim):
                with Ts.sblock("llama_fused_rope"):
                    s, h, d = Ts.axis.remap("SSS", iters)
                    if h < num_q_heads:
                        q[s, h, d] = T.if_then_else(
                            d < rotary_dim,
                            _rope(
                                qkv,
                                s,
                                h,
                                d,
                                position_map[s],
                                short_factors if is_longrope_scaling else None,
                            ),
                            qkv[s, h, d],
                        )
                    elif h < num_q_heads + num_kv_heads:
                        k[s, h - num_q_heads, d] = T.if_then_else(
                            d < rotary_dim,
                            _rope(
                                qkv,
                                s,
                                h,
                                d,
                                position_map[s],
                                short_factors if is_longrope_scaling else None,
                            ),
                            qkv[s, h, d],
                        )
                    else:
                        v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]

    if is_longrope_scaling:
        return fused_rope_longrope_scaling
    return fused_rope


def llama4_rope_with_position_map(  # pylint: disable=too-many-arguments
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
    rope_scaling: dict[str, Any],
    rotary_dim: int | None = None,
):
    """Return the TIR function that computes Llama-style RoPE with q position map.

    Parameters
    ----------
    theta : float
        The theta value, or "base" in RoPE, which controls the frequency.

    scale : float
        The RoPE scaling factor.

    head_dim : int
        The number of features on each head.

    num_q_heads : int
        The number of query heads.

    num_kv_heads : int
        The number of key/value heads. It differs from `num_q_heads` in group-query attention.

    dtype : str
        The dtype of qkv data.

    rope_scaling : Dict
        The configuration of RoPE scaling.

    rotary_dim : int
        The number of dimensions in the embedding that RoPE is applied to. By default, the
        rotary_dim is the same as head_dim.
    """
    fused_heads = num_q_heads + num_kv_heads * 2
    if rotary_dim is None:
        rotary_dim = head_dim
    scale = tirx.const(scale, "float32")
    is_longrope_scaling = rope_scaling.get("rope_type") == "longrope"
    if is_longrope_scaling and "original_max_position_embeddings" in rope_scaling:
        original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]
    else:
        original_max_position_embeddings = 0

    def _rope(  # pylint: disable=too-many-arguments
        x: T.Tensor,
        s: tirx.Var,
        h: tirx.Var,
        d: tirx.Var,
        pos: tirx.Var,
        ext_factors: T.Tensor | None = None,
    ):
        kwargs = {}
        if ext_factors is not None:
            kwargs["ext_factors"] = ext_factors
        cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(
            pos * scale, d, rotary_dim, theta, "float32", **kwargs
        )
        cos = cos_freq * x[s, h, d].astype("float32")
        if "rope_type" in rope_scaling and rope_scaling["rope_type"] == "gptj":
            sin = sin_freq * tirx.if_then_else(
                d % 2 == 0,
                -x[s, h, d + 1],
                x[s, h, d - 1],
            ).astype("float32")
        else:
            # Data layout is different for llama4 vs llama3
            sin = sin_freq * tirx.if_then_else(
                d % 2 == 0,
                -x[s, h, d + 1],
                x[s, h, d - 1],
            ).astype("float32")
        expr = (cos + sin).astype(dtype)
        for var, value in var_map.items():
            expr = tirx.Let(var, value, expr)
        return expr

    seq_len = T.dynamic("seq_len", "int32")
    position_map_elem_offset = T.dynamic("position_map_elem_offset", "int32")

    @Ts.function(private=True)
    def fused_rope(  # pylint: disable=too-many-locals
        qkv: T.Tensor((seq_len, fused_heads, head_dim), dtype),
        position_map: T.Tensor(
            (seq_len,), "int32", layout=T.TileLayout(T.S[(seq_len,)] + (position_map_elem_offset))
        ),
        q: T.Tensor((seq_len, num_q_heads, head_dim), dtype),
        k: T.Tensor((seq_len, num_kv_heads, head_dim), dtype),
        v: T.Tensor((seq_len, num_kv_heads, head_dim), dtype),
        apply_rope: T.int64,
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tirx.noalias": True,
            }
        )

        for iters in T.grid(seq_len, fused_heads, head_dim):
            with Ts.sblock("llama_fused_rope"):
                s, h, d = Ts.axis.remap("SSS", iters)
                if h < num_q_heads:
                    q[s, h, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                elif h < num_q_heads + num_kv_heads:
                    k[s, h - num_q_heads, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                else:
                    v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]

    seq_len = T.dynamic("seq_len", "int64")
    position_map_elem_offset = T.dynamic("position_map_elem_offset", "int64")

    @Ts.function
    def fused_rope_longrope_scaling(  # pylint: disable=too-many-locals
        qkv: T.Tensor((seq_len, fused_heads, head_dim), dtype),
        position_map: T.Tensor(
            (seq_len,), "int32", layout=T.TileLayout(T.S[(seq_len,)] + (position_map_elem_offset))
        ),
        q: T.Tensor((seq_len, num_q_heads, head_dim), dtype),
        k: T.Tensor((seq_len, num_kv_heads, head_dim), dtype),
        v: T.Tensor((seq_len, num_kv_heads, head_dim), dtype),
        ext_factors: T.Tensor((rotary_dim,), "float32"),  # type: ignore
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tirx.noalias": True,
            }
        )

        # long factors is the first half, short factors is the second half
        long_factors = T.decl_tensor(
            ext_factors.data,
            ty_args=[T.Tensor((rotary_dim // 2,), "float32", data=ext_factors.data)],
        )
        short_factors = T.decl_tensor(
            ext_factors.data,
            ty_args=[
                T.Tensor(
                    (rotary_dim // 2,),
                    "float32",
                    layout=T.TileLayout(T.S[(rotary_dim // 2,)] + (rotary_dim // 2)),
                    data=ext_factors.data,
                )
            ],
        )

        if seq_len > original_max_position_embeddings:
            for iters in T.grid(seq_len, fused_heads, head_dim):
                with Ts.sblock("llama_fused_rope"):
                    s, h, d = Ts.axis.remap("SSS", iters)
                    if h < num_q_heads:
                        q[s, h, d] = T.if_then_else(
                            d < rotary_dim,
                            _rope(
                                qkv,
                                s,
                                h,
                                d,
                                position_map[s],
                                long_factors if is_longrope_scaling else None,
                            ),
                            qkv[s, h, d],
                        )
                    elif h < num_q_heads + num_kv_heads:
                        k[s, h - num_q_heads, d] = T.if_then_else(
                            d < rotary_dim,
                            _rope(
                                qkv,
                                s,
                                h,
                                d,
                                position_map[s],
                                long_factors if is_longrope_scaling else None,
                            ),
                            qkv[s, h, d],
                        )
                    else:
                        v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]
        else:
            for iters in T.grid(seq_len, fused_heads, head_dim):
                with Ts.sblock("llama_fused_rope"):
                    s, h, d = Ts.axis.remap("SSS", iters)
                    if h < num_q_heads:
                        q[s, h, d] = T.if_then_else(
                            d < rotary_dim,
                            _rope(
                                qkv,
                                s,
                                h,
                                d,
                                position_map[s],
                                short_factors if is_longrope_scaling else None,
                            ),
                            qkv[s, h, d],
                        )
                    elif h < num_q_heads + num_kv_heads:
                        k[s, h - num_q_heads, d] = T.if_then_else(
                            d < rotary_dim,
                            _rope(
                                qkv,
                                s,
                                h,
                                d,
                                position_map[s],
                                short_factors if is_longrope_scaling else None,
                            ),
                            qkv[s, h, d],
                        )
                    else:
                        v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]

    if is_longrope_scaling:
        return fused_rope_longrope_scaling
    return fused_rope
