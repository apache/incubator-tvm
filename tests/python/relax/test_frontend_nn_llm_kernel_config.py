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

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax.frontend.nn.llm._decode_kernels import _merge_state_inplace
from tvm.relax.frontend.nn.llm._kernel_common import (
    _get_prefill_kernel_config,
    _get_prefill_shared_memory_usage,
)
from tvm.relax.frontend.nn.llm._prefill_kernels import (
    _attention_prefill,
    _attention_prefill_mla,
    _attention_prefill_ragged,
)
from tvm.relax.frontend.nn.llm.tree_attn import tree_attn, tree_attn_with_paged_kv_cache


def _get_allocated_shared_memory(func):
    mod = tvm_tirx.IRModule.from_expr(func)
    mod = tvm.s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    allocated_bytes = tvm.s_tir.analysis.calculate_allocated_bytes(mod)
    (function_allocations,) = allocated_bytes.values()
    return function_allocations["shared"]


def test_wide_head_prefill_fits_metal_shared_memory():
    target = tvm_tirx.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=512,
        dtype="float16",
        target=target,
    )

    _, _, _, _, num_warps, tile_x, _, tile_z = config
    assert num_warps == 2
    assert tile_z == 8
    assert _get_prefill_shared_memory_usage(tile_x, tile_z, 512, "float16") == 24_928
    assert 24_928 <= int(target.attrs["max_shared_memory_per_block"])


@pytest.mark.parametrize("sliding_window", [False, True])
def test_wide_fp32_head_prefill_fits_webgpu_shared_memory(sliding_window):
    target = tvm_tirx.target.Target("webgpu")
    config = _get_prefill_kernel_config(1, 8, 512, "float32", target)
    func = _attention_prefill(1, 8, 512, "float32", sliding_window, {}, target)

    assert config == (16, 2, 8, 16, 1, 4, 512, 4)
    assert _get_allocated_shared_memory(func) <= int(target.attrs["max_shared_memory_per_block"])
    mod = tvm_tirx.tirx.compile(func.with_attr("global_symbol", "main"), target=target).mod
    assert mod.imports[0].inspect_source()


def test_non_power_of_two_head_prefill_reduces_query_tile_for_metal():
    target = tvm_tirx.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=384,
        dtype="float16",
        target=target,
    )
    func = _attention_prefill(1, 8, 384, "float16", False, {}, target)

    assert config == (16, 4, 8, 32, 2, 8, 384, 8)
    assert func.attrs["tirx.is_scheduled"]
    assert _get_allocated_shared_memory(func) == 18_784
    assert 18_784 <= int(target.attrs["max_shared_memory_per_block"])
    assert tvm_tirx.tirx.compile(func, target=target).mod.imports[0].inspect_source()


def test_reduced_query_tile_keeps_the_real_head_dimension():
    target = tvm_tirx.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=416,
        dtype="float16",
        target=target,
    )
    func = _attention_prefill(1, 8, 416, "float16", False, {}, target)

    assert config == (16, 4, 8, 32, 2, 8, 416, 8)
    assert func.attrs["tirx.is_scheduled"]
    assert _get_allocated_shared_memory(func) == 20_320
    assert 20_320 <= int(target.attrs["max_shared_memory_per_block"])
    assert tvm_tirx.tirx.compile(func, target=target).mod.imports[0].inspect_source()


def test_fallback_can_select_larger_nearby_query_tile():
    target = tvm_tirx.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=640,
        dtype="float16",
        target=target,
    )
    func = _attention_prefill(1, 8, 640, "float16", False, {}, target)

    assert config == (16, 4, 8, 32, 2, 8, 640, 8)
    assert func.attrs["tirx.is_scheduled"]
    assert _get_allocated_shared_memory(func) == 31_072
    assert 31_072 <= int(target.attrs["max_shared_memory_per_block"])
    assert tvm_tirx.tirx.compile(func, target=target).mod.imports[0].inspect_source()


@pytest.mark.parametrize(
    ("d", "dtype", "expected_config", "expected_shared_memory"),
    [
        (672, "float16", (16, 4, 8, 32, 2, 8, 672, 8), 32_608),
        (768, "float16", (16, 4, 8, 32, 1, 4, 768, 8), 30_896),
        (384, "float32", (16, 2, 8, 32, 1, 4, 384, 8), 30_896),
    ],
)
def test_fallback_can_expand_key_tile_for_factorization(
    d, dtype, expected_config, expected_shared_memory
):
    target = tvm_tirx.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=d,
        dtype=dtype,
        target=target,
    )
    func = _attention_prefill(1, 8, d, dtype, False, {}, target)

    assert config == expected_config
    assert func.attrs["tirx.is_scheduled"]
    assert _get_allocated_shared_memory(func) == expected_shared_memory
    assert expected_shared_memory <= int(target.attrs["max_shared_memory_per_block"])
    assert tvm_tirx.tirx.compile(func, target=target).mod.imports[0].inspect_source()


def test_normal_head_prefill_keeps_existing_metal_config():
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=256,
        dtype="float16",
        target=tvm_tirx.target.Target("metal"),
    )

    assert config == (16, 4, 8, 32, 4, 16, 256, 16)


def test_wide_head_prefill_keeps_existing_webgpu_config():
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=512,
        dtype="float16",
        target=tvm_tirx.target.Target("webgpu"),
    )

    assert config == (16, 4, 8, 32, 2, 8, 512, 8)


def test_wide_head_prefill_uses_target_shared_memory_limit():
    target = tvm_tirx.target.Target({"kind": "metal", "max_shared_memory_per_block": 65_536})
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=512,
        dtype="float16",
        target=target,
    )

    assert config == (16, 4, 8, 32, 4, 8, 512, 16)
    assert _get_prefill_shared_memory_usage(8, 16, 512, "float16") == 41_568
    assert 41_568 <= int(target.attrs["max_shared_memory_per_block"])


def test_wide_head_prefill_rejects_unachievable_shared_memory_limit():
    target = tvm_tirx.target.Target({"kind": "metal", "max_shared_memory_per_block": 1_024})

    with pytest.raises(ValueError, match="target allows 1024 bytes"):
        _get_prefill_kernel_config(
            h_kv=1,
            h_q=8,
            d=512,
            dtype="float16",
            target=target,
        )


def test_ragged_prefill_accounts_for_wider_value_dimension():
    target = tvm_tirx.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=256,
        dtype="float16",
        target=target,
        d_v=512,
    )
    func = _attention_prefill_ragged(1, 8, 256, 512, "float16", {}, target)

    assert config == (16, 4, 8, 32, 4, 16, 256, 8)
    assert _get_prefill_shared_memory_usage(16, 8, 256, "float16", d_v=512) == 21_184
    assert _get_allocated_shared_memory(func) == 21_184
    assert 21_184 <= int(target.attrs["max_shared_memory_per_block"])


def test_mla_prefill_accounts_for_merged_kv_buffer():
    target = tvm_tirx.target.Target("metal")
    config = _get_prefill_kernel_config(
        h_kv=1,
        h_q=8,
        d=576,
        dtype="float16",
        target=target,
        d_v=512,
        merged_kv=True,
    )
    func = _attention_prefill_mla(8, 512, 64, "float16", False, target)

    assert config == (16, 4, 8, 32, 4, 8, 576, 16)
    assert (
        _get_prefill_shared_memory_usage(8, 16, 576, "float16", d_v=512, merged_kv=True) == 28_256
    )
    assert _get_allocated_shared_memory(func) == 28_256
    assert 28_256 <= int(target.attrs["max_shared_memory_per_block"])


@pytest.mark.parametrize("kernel", [tree_attn, tree_attn_with_paged_kv_cache])
def test_wide_head_tree_attention_has_legal_metal_schedule(kernel):
    target = tvm_tirx.target.Target("metal")
    func = kernel(1, 8, 512, "float16", {}, target)

    assert func.attrs["tirx.is_scheduled"]
    assert _get_allocated_shared_memory(func) == 24_928
    assert 24_928 <= int(target.attrs["max_shared_memory_per_block"])


@pytest.mark.parametrize("head_dim", [128, 256, 512])
def test_merge_state_waits_before_overwriting_the_shared_lse(head_dim):
    """merge_state_inplace must synchronize between reading and overwriting S.

    Every threadIdx.x thread of a block reads the same S element and later writes it back. The
    threads only advance in step while bdx fits in one warp, and bdx is head_dim // VEC_SIZE, so a
    head_dim above 128 spreads them over several warps. Without a barrier a warp that finishes
    early overwrites S while another warp has yet to read it, and that warp then rescales its slice
    of V with an already merged LSE, corrupting whole warp-sized spans of the output.
    """
    target = tvm_tirx.target.Target("metal")
    func = _merge_state_inplace(8, head_dim, "float16", target, global_symbol="merge")
    lines = [
        line.strip()
        for line in tvm_tirx.tirx.compile(func, target=target)
        .mod.imports[0]
        .inspect_source()
        .splitlines()
    ]

    loads = [i for i, line in enumerate(lines) if "= S_ptr[" in line]
    stores = [i for i, line in enumerate(lines) if line.startswith("S_ptr[")]
    assert loads and stores, "expected the kernel to both read and write the shared LSE"

    last_load, first_store = max(loads), min(stores)
    assert any(
        last_load < i < first_store for i, line in enumerate(lines) if "threadgroup_barrier" in line
    ), "no barrier between reading the shared LSE and overwriting it"
    assert any("threadIdx.x) == 0" in line for line in lines[last_load:first_store]), (
        "the shared LSE should be stored by a single thread"
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
