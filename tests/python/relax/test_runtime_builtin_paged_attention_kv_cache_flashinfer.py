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
import functools

# ruff: noqa: E741
import pytest
import torch
import tvm_ffi
from tvm_ffi import Shape

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.frontend.nn.llm.kv_cache import (
    AttnKind,
    RopeMode,
    _compact_kv_copy,
    _copy_single_page,
    _kv_cache_debug_get_kv,
    _kv_cache_transpose_append,
    _merge_state_inplace,
    llama_rope_with_position_map,
)
from tvm.s_tir import dlight as dl


def has_flashinfer():
    """Check whether FlashInfer (with the JIT module generator) is available."""
    try:
        from flashinfer.jit import gen_customize_batch_prefill_module  # noqa: F401

        return True
    except ImportError:
        return False


reserved_nseq = 32
maximum_total_seq_length = 2048
prefill_chunk_size = 512
page_size = 16
num_layers = 4
num_qo_heads = 32
num_kv_heads = 4
head_dim = 128
sm_scale = head_dim ** (-0.5)
rope_scale = 1.0
rope_theta = 1e4
dtype = "float16"
dtype_torch = getattr(torch, dtype)
device = None
device_torch = torch.device("cuda")

fclear = None
fadd_sequence = None
fremove_sequence = None
ffork_sequence = None
fpopn = None
fbegin_forward = None
fend_forward = None
fattention = None
fattention_with_fuse_qkv = None
fdebug_get_kv = None

fattention_prefill = None
fattention_decode = None
fattention_prefill_ragged = None
fattention_prefill_plan = None
fattention_decode_plan = None
fattention_prefill_ragged_plan = None
fattention_merge_state = None

ftranspose_append = None
fsplit_rotary = None
fcopy_single_page = None
fcopy_cache = None
fcompact_copy = None


def set_global_func(rope_mode: RopeMode, target):
    global fclear, fadd_sequence, fremove_sequence, ffork_sequence, fpopn
    global fbegin_forward, fend_forward, fattention, fattention_with_fuse_qkv, fdebug_get_kv
    global fattention_prefill, fattention_decode, fattention_prefill_ragged
    global fattention_prefill_plan, fattention_decode_plan, fattention_prefill_ragged_plan
    global fattention_merge_state, fsplit_rotary, fcopy_single_page
    global ftranspose_append, fcopy_cache, fcompact_copy

    fclear = tvm_ffi.get_global_func("vm.builtin.kv_state_clear")
    fadd_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_add_sequence")
    fremove_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_remove_sequence")
    ffork_sequence = tvm_ffi.get_global_func("vm.builtin.kv_state_fork_sequence")
    fpopn = tvm_ffi.get_global_func("vm.builtin.kv_state_popn")
    fbegin_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm_ffi.get_global_func("vm.builtin.kv_state_end_forward")
    fattention_with_fuse_qkv = tvm_ffi.get_global_func(
        "vm.builtin.attention_kv_cache_attention_with_fused_qkv"
    )
    fdebug_get_kv = tvm_ffi.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv")

    flashinfer_prefill_mod = relax.backend.cuda.flashinfer.gen_flashinfer_prefill_module(
        dtype_q=dtype,
        dtype_kv=dtype,
        dtype_o=dtype,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        enable_inline_rope=rope_mode == RopeMode.INLINE,
    )[0]
    flashinfer_decode_mod = relax.backend.cuda.flashinfer.gen_flashinfer_decode_module(
        dtype_q=dtype,
        dtype_kv=dtype,
        dtype_o=dtype,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        enable_inline_rope=rope_mode == RopeMode.INLINE,
    )[0]

    fattention_prefill = flashinfer_prefill_mod["batch_prefill_paged_run"]
    fattention_prefill_plan = flashinfer_prefill_mod["batch_prefill_plan"]
    fattention_prefill_ragged = flashinfer_prefill_mod["batch_prefill_ragged_run"]
    fattention_prefill_ragged_plan = flashinfer_prefill_mod["batch_prefill_plan"]
    fattention_decode = flashinfer_decode_mod["batch_decode_run"]
    fattention_decode_plan = flashinfer_decode_mod["batch_decode_plan"]

    builts = []
    for tir_func in [
        _kv_cache_transpose_append(num_kv_heads, head_dim, dtype),
        _merge_state_inplace(num_qo_heads, head_dim, dtype, target),
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype, {}
        ),
        _copy_single_page(num_kv_heads, page_size, head_dim, dtype, target),
        _kv_cache_debug_get_kv(num_layers, num_kv_heads, head_dim, dtype),
        _compact_kv_copy(num_kv_heads, head_dim, dtype, target),
    ]:
        mod = tvm_tirx.IRModule({"main": tir_func})
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback(), target=target)(mod)
        f = tvm_tirx.tirx.compile(mod["main"], target=target).mod
        builts.append(f.main)

    (
        ftranspose_append,
        fattention_merge_state,
        fsplit_rotary,
        fcopy_single_page,
        fcopy_cache,
        fcompact_copy,
    ) = builts


def create_kv_cache(rope_mode):
    fcreate = tvm_ffi.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    support_sliding_window = 0
    cache = fcreate(
        tvm_ffi.Shape(
            [
                reserved_nseq,
                maximum_total_seq_length,
                prefill_chunk_size,
                page_size,
                support_sliding_window,
            ]
        ),
        tvm_ffi.Shape([0, num_layers]),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,  # v_head_dim
        tvm_ffi.Shape([int(AttnKind.MHA) for _ in range(num_layers)]),
        False,  # enable_kv_transfer
        rope_mode,
        rope_scale,
        rope_theta,
        None,  # rope_ext_factors
        tvm_tirx.runtime.empty((), dtype, device=device),
        ftranspose_append,
        None,  # f_transpose_append_mla
        ["flashinfer", fattention_prefill_ragged, fattention_prefill_ragged_plan],
        ["flashinfer", fattention_prefill, fattention_prefill_plan],
        ["flashinfer", fattention_decode, fattention_decode_plan],
        [],  # fattn_prefill_sliding_window
        [],  # fattn_decode_sliding_window
        [],  # fattn_prefill_with_tree_mask_paged_kv_cache
        [],  # fattn_prefill_with_tree_mask
        [],  # f_mla_prefill
        [fattention_merge_state],
        fsplit_rotary,
        fcopy_single_page,
        fcopy_cache,
        fcompact_copy,
    )
    return cache


@pytest.fixture(params=[RopeMode.NONE, RopeMode.NORMAL, RopeMode.INLINE])
def kv_cache_and_rope_mode(request):
    if not has_flashinfer():
        pytest.skip("FlashInfer is not available")
    if request.param == RopeMode.INLINE:
        # FlashInfer does not support inline RoPE (see the assertion in
        # tvm/relax/frontend/nn/llm/kv_cache.py); models pair FlashInfer with
        # NORMAL/NONE rope and apply rotary embedding in a separate kernel.
        pytest.skip("FlashInfer does not support inline RoPE mode")
    target = tvm_tirx.testing.run_with_gpu_lock(_get_cuda_target)
    set_global_func(request.param, target)
    return request.param


def _get_cuda_target():
    return tvm_tirx.target.Target.from_device(tvm_tirx.cuda())


def _run_with_kv_cache(test):
    @functools.wraps(test)
    def wrapper(kv_cache_and_rope_mode):
        def run_and_check():
            global device
            device = tvm_tirx.cuda()
            try:
                cache = create_kv_cache(kv_cache_and_rope_mode)
                return test((cache, kv_cache_and_rope_mode))
            finally:
                device = None

        return tvm_tirx.testing.run_with_gpu_lock(run_and_check)

    return wrapper


def verify_cached_kv(kv_cache, seq_ids, expected_k, expected_v):
    for seq_id in seq_ids:
        keys_expected = expected_k[seq_id]
        values_expected = expected_v[seq_id]
        assert keys_expected.shape == values_expected.shape
        seq_length = expected_k[seq_id].shape[1]
        keys = tvm_tirx.runtime.empty(keys_expected.shape, dtype=dtype, device=device)
        values = tvm_tirx.runtime.empty(values_expected.shape, dtype=dtype, device=device)
        fdebug_get_kv(kv_cache, seq_id, 0, seq_length, keys, values)
        torch.testing.assert_close(
            torch.from_numpy(keys.numpy()).to(device_torch), keys_expected, rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            torch.from_numpy(values.numpy()).to(device_torch), values_expected, rtol=1e-3, atol=1e-3
        )


def f_apply_rotary(x, offset, scale, theta):
    # x: (N, H, D)
    assert len(x.shape) == 3
    nfeat = x.shape[-1]
    nfeat_half = x.shape[-1] // 2
    x_dtype = x.dtype
    x = x.to(torch.float32)
    y = torch.cat([-x[:, :, nfeat_half:], x[:, :, :nfeat_half]], dim=-1)

    inv_freq = scale / (
        theta ** (torch.arange(0, nfeat, 2, device=device_torch, dtype=torch.float32) / nfeat)
    )
    t = torch.arange(offset, offset + x.shape[0], device=device_torch, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_values = torch.cos(emb)
    sin_values = torch.sin(emb)

    return torch.einsum("ij,ikj->ikj", cos_values, x).to(x_dtype) + torch.einsum(
        "ij,ikj->ikj", sin_values, y
    ).to(x_dtype)


def apply_attention(
    kv_cache,
    rope_mode: RopeMode,
    batch: list[tuple[int | tuple[int, int, int], int]],
    cached_k: dict[int, torch.Tensor],
    cached_v: dict[int, torch.Tensor],
) -> None:
    seq_ids = []
    append_lengths = []
    for i, (seq_id, append_length) in enumerate(batch):
        fork_parent_id = None
        if isinstance(seq_id, tuple):
            # Fork sequence
            seq_id, fork_parent_id, fork_pos = seq_id
            batch[i] = (seq_id, append_length)
        seq_ids.append(seq_id)
        append_lengths.append(append_length)
        if fork_parent_id is not None:
            assert fork_parent_id in cached_k
            assert seq_id not in cached_k
            ffork_sequence(kv_cache, fork_parent_id, seq_id, fork_pos)
            if fork_pos == -1:
                cached_k[seq_id] = cached_k[fork_parent_id]
                cached_v[seq_id] = cached_v[fork_parent_id]
            else:
                cached_k[seq_id] = cached_k[fork_parent_id][::, :fork_pos]
                cached_v[seq_id] = cached_v[fork_parent_id][::, :fork_pos]
        elif seq_id not in cached_k:
            fadd_sequence(kv_cache, seq_id)
            cached_k[seq_id] = torch.zeros(
                (num_layers, 0, num_kv_heads, head_dim), dtype=dtype_torch, device=device_torch
            )
            cached_v[seq_id] = torch.zeros(
                (num_layers, 0, num_kv_heads, head_dim), dtype=dtype_torch, device=device_torch
            )

    fbegin_forward(kv_cache, Shape(seq_ids), Shape(append_lengths))

    global_new_q = torch.zeros(
        (num_layers, 0, num_qo_heads, head_dim), dtype=dtype_torch, device=device_torch
    )
    global_new_k = torch.zeros(
        (num_layers, 0, num_kv_heads, head_dim), dtype=dtype_torch, device=device_torch
    )
    global_new_v = torch.zeros(
        (num_layers, 0, num_kv_heads, head_dim), dtype=dtype_torch, device=device_torch
    )

    q_array = []
    for seq_id, append_length in batch:
        new_q = torch.rand(
            num_layers,
            append_length,
            num_qo_heads,
            head_dim,
            dtype=dtype_torch,
            device=device_torch,
        )
        new_k = torch.rand(
            num_layers,
            append_length,
            num_kv_heads,
            head_dim,
            dtype=dtype_torch,
            device=device_torch,
        )
        new_v = torch.rand(
            num_layers,
            append_length,
            num_kv_heads,
            head_dim,
            dtype=dtype_torch,
            device=device_torch,
        )
        new_q = new_q * 2 - 1
        new_k = new_k * 2 - 1
        new_v = new_v * 2 - 1
        q_array.append(new_q)

        cached_k[seq_id] = torch.cat(
            [
                cached_k[seq_id],
                torch.stack(
                    [
                        (
                            new_k[l]
                            if rope_mode != RopeMode.NORMAL
                            else f_apply_rotary(
                                new_k[l], cached_k[seq_id].shape[1], rope_scale, rope_theta
                            )
                        )
                        for l in range(num_layers)
                    ],
                    dim=0,
                ),
            ],
            dim=1,
        )
        cached_v[seq_id] = torch.cat([cached_v[seq_id], new_v], dim=1)
        global_new_q = torch.cat([global_new_q, new_q], dim=1)
        global_new_k = torch.cat([global_new_k, new_k], dim=1)
        global_new_v = torch.cat([global_new_v, new_v], dim=1)

    for layer_id in range(num_layers):
        queries_np = global_new_q[layer_id]
        keys_np = global_new_k[layer_id]
        values_np = global_new_v[layer_id]
        qkv = tvm_tirx.runtime.tensor(
            torch.cat([queries_np, keys_np, values_np], dim=1).cpu().numpy(), device
        )
        outputs = tvm_tirx.runtime.empty(queries_np.shape, dtype, device=device)
        fattention_with_fuse_qkv(kv_cache, layer_id, sm_scale, qkv, outputs)

        # Compute attention expected results.
        outputs = torch.from_numpy(outputs.numpy()).unsqueeze(0).to(device_torch)
        sum_length = 0
        for i, (seq_id, append_length) in enumerate(batch):
            assert cached_k[seq_id].shape[1] == cached_v[seq_id].shape[1] >= append_length

            rope_offset = cached_k[seq_id].shape[1] - append_length
            q_seq = (
                q_array[i][layer_id]
                if rope_mode == RopeMode.NONE
                else f_apply_rotary(
                    q_array[i][layer_id],
                    rope_offset,
                    rope_scale,
                    rope_theta,
                )
            ).permute(1, 0, 2)
            k_seq = (
                cached_k[seq_id][layer_id]
                if rope_mode != RopeMode.INLINE
                else f_apply_rotary(cached_k[seq_id][layer_id], 0, rope_scale, rope_theta)
            ).permute(1, 2, 0)
            v_seq = cached_v[seq_id][layer_id].permute(1, 0, 2)

            k_seq = k_seq.repeat_interleave(num_qo_heads // num_kv_heads, dim=0)
            v_seq = v_seq.repeat_interleave(num_qo_heads // num_kv_heads, dim=0)
            softmax_input = (q_seq.to(torch.float32) @ k_seq.to(torch.float32)) / (head_dim**0.5)
            softmax_shape = softmax_input.shape
            length_diff = softmax_shape[-1] - softmax_shape[-2]
            assert length_diff >= 0
            mask = torch.tril(
                torch.full_like(softmax_input, torch.finfo(torch.float32).max), diagonal=length_diff
            ) + torch.triu(
                torch.full_like(softmax_input, torch.finfo(torch.float32).min),
                diagonal=length_diff + 1,
            )
            softmax_input = torch.minimum(softmax_input, mask)

            results = torch.unsqueeze(
                (
                    torch.nn.functional.softmax(softmax_input, dim=-1) @ v_seq.to(torch.float32)
                ).permute(1, 0, 2),
                dim=0,
            ).to(dtype_torch)

            torch.testing.assert_close(
                outputs[:, sum_length : sum_length + append_length, ...],
                results,
                rtol=1e-3,
                atol=1e-3,
            )
            sum_length += append_length
    fend_forward(kv_cache)

    # Verify
    verify_cached_kv(kv_cache, seq_ids, cached_k, cached_v)


@_run_with_kv_cache
def test_paged_attention_kv_cache_prefill_and_decode(kv_cache_and_rope_mode):
    kv_cache, rope_mode = kv_cache_and_rope_mode
    fclear(kv_cache)

    # Prefill.
    operation_seq = [[(0, 6)], [(1, 8)], [(2, 11)], [(3, 16)], [(4, 19), (5, 20)]]
    operation_seq += [[(6, 21), (7, 24)], [(2, 5), (4, 7), (8, 24)]]
    operation_seq += [[(6, 13)], [(8, 19)], [(0, 1)], [(1, 3), (3, 8), (5, 12), (7, 11)]]
    # Decode
    operation_seq += [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]
    operation_seq += [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]
    operation_seq += [[(0, 1), (2, 1), (4, 1), (6, 1), (8, 1)]]
    operation_seq += [[(4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]

    cached_k = {}
    cached_v = {}
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)


@_run_with_kv_cache
def test_paged_attention_kv_cache_remove_sequence(kv_cache_and_rope_mode):
    kv_cache, rope_mode = kv_cache_and_rope_mode
    fclear(kv_cache)

    num_sequences = 5
    batch = [(seq_id, 1) for seq_id in range(num_sequences)]
    cached_k = {}
    cached_v = {}
    for seq_id_to_remove in range(num_sequences):
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)
        # Remove sequence.
        fremove_sequence(kv_cache, seq_id_to_remove)
        cached_k.pop(seq_id_to_remove)
        cached_v.pop(seq_id_to_remove)
        verify_cached_kv(
            kv_cache,
            seq_ids=[seq_id for seq_id in range(num_sequences) if seq_id != seq_id_to_remove],
            expected_k=cached_k,
            expected_v=cached_v,
        )


@_run_with_kv_cache
def test_paged_attention_kv_cache_fork_sequence(kv_cache_and_rope_mode):
    kv_cache, rope_mode = kv_cache_and_rope_mode
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 60), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)
    # Fork existing sequences.
    apply_attention(kv_cache, rope_mode, [((4, 3, -1), 35)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((5, 0, -1), 20)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((6, 5, -1), 102)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((7, 0, -1), 3)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((8, 5, -1), 71), ((9, 5, -1), 20)], cached_k, cached_v)
    # 0 <- 5 <- 6,8,9
    # 0 <- 7
    # 3 <- 4
    # Mixture of decode and prefill.
    operation_seq = [
        [(2, 1), (4, 1), (7, 1), (6, 1), (8, 1), (9, 1)],
        [(7, 1), (6, 1), (8, 1), (9, 1)],
        [(7, 1), (1, 1), (6, 1), (2, 1), (8, 1), (4, 1), (9, 1)],
        [(7, 10), (6, 2), (8, 3), (9, 4)],
    ]
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)

    apply_attention(kv_cache, rope_mode, [((10, 1, 33), 11)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((11, 0, 60), 45), ((12, 0, 15), 14)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((13, 0, 16), 19), ((14, 0, 17), 19)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((15, 5, 60), 8), ((16, 5, 80), 10)], cached_k, cached_v)
    apply_attention(
        kv_cache,
        rope_mode,
        [((17, 5, 75), 11), ((18, 5, 76), 45), ((19, 5, 77), 14)],
        cached_k,
        cached_v,
    )

    operation_seq = [
        [(6, 1), (11, 1), (13, 1), (9, 1)],
        [(10, 1), (16, 1), (18, 1), (19, 1)],
        [(8, 1), (15, 1), (17, 1), (12, 1), (14, 1)],
        [(10, 10), (6, 2), (8, 3), (19, 4)],
    ]
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)

    for i in range(19, -1, -1):
        fremove_sequence(kv_cache, i)
        cached_k.pop(i)
        cached_v.pop(i)
        verify_cached_kv(kv_cache, seq_ids=list(range(i)), expected_k=cached_k, expected_v=cached_v)

    # Test fork after page recycle
    apply_attention(kv_cache, rope_mode, [(0, 7), (1, 24)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((2, 1, -1), 10)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((3, 0, -1), 20)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [(2, 1), (3, 1)], cached_k, cached_v)

    apply_attention(kv_cache, rope_mode, [(10, 7), (11, 24)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((12, 11, -1), 200)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [(10, 1), (12, 1)], cached_k, cached_v)


@_run_with_kv_cache
def test_paged_attention_kv_cache_popn(kv_cache_and_rope_mode):
    kv_cache, rope_mode = kv_cache_and_rope_mode
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 35), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((4, 3, -1), 35)], cached_k, cached_v)

    popn_operations = [(0, 17), (1, 57), (2, 16), (3, 0), (4, 19)]
    for seq_id, pop_length in popn_operations:
        fpopn(kv_cache, seq_id, pop_length)
        if pop_length != 0:
            cached_k[seq_id] = cached_k[seq_id][:, :-pop_length, ...]
            cached_v[seq_id] = cached_v[seq_id][:, :-pop_length, ...]
        verify_cached_kv(kv_cache, seq_ids=list(range(5)), expected_k=cached_k, expected_v=cached_v)


if __name__ == "__main__":
    tvm_tirx.testing.main()
