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
import itertools

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax.frontend.nn.llm._kernel_common import _get_prefill_kernel_config
from tvm.relax.frontend.nn.llm._prefill_kernels import _attention_prefill_ragged
from tvm.testing import device_enabled
from tvm_tirx import ir


@pytest.mark.parametrize("group_size", [1, 3, 8])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_ragged_causal_loop_bound(group_size, dtype):
    target = tvm_tirx.target.Target("webgpu")
    config = _get_prefill_kernel_config(1, group_size, 64, dtype, target)
    tile_x, _, tile_z = config[-3:]
    func = _attention_prefill_ragged(1, group_size, 64, 64, dtype, {}, target)
    bindings = {}
    loops = []

    def collect(node):
        if isinstance(node, ir.Bind):
            bindings[node.var.name] = node.value
        if isinstance(node, ir.For) and node.var.name == "iterator":
            loops.append(node)

    tvm_ffi.structural_walk(func.body, collect)
    assert len(loops) == 1
    analyzer = tvm_tirx.sym.Analyzer()
    # PagedKVCache passes the same indptr for Q and KV, so each sequence has q_len == kv_len.
    # Also cover longer KV sequences, supported by the kernel's separate indptr arrays.
    for seq_len, kv_extra, causal in itertools.product((1, 7, 33, 65, 129), (0, 37), (0, 1)):
        kv_len = seq_len + kv_extra
        for start in range(0, seq_len * group_size, tile_x):
            values = {"q_len": seq_len, "LH_start": start, "causal": causal}

            def substitute(node):
                # q_len is replaced below, leaving only kv_chunk_len[0] as a tensor load.
                if isinstance(node, ir.TensorLoad):
                    return ir.IntImm("int32", kv_len)
                if isinstance(node, ir.Var):
                    if node.name in values:
                        return ir.IntImm("int32", values[node.name])
                    if node.name in ("q_end", "kv_stop"):
                        return bindings[node.name]
                return node

            end = loops[0].end
            for _ in range(3):
                end = tvm_ffi.structural_map(end, substitute)
            actual = int(analyzer.simplify(end))
            # Derive the last visible key independently of the kernel's bound expression.
            last_row = min(start + tile_x, seq_len * group_size) - 1
            last_key = kv_len - seq_len + last_row // group_size if causal else kv_len - 1
            assert actual == last_key // tile_z + 1, (seq_len, kv_len, causal, start, group_size)


def test_ragged_prefill_webgpu_codegen():
    target = tvm_tirx.target.Target("webgpu")
    func = _attention_prefill_ragged(1, 3, 64, 32, "float32", {}, target)
    compiled = tvm_tirx.tirx.compile(func.with_attr("global_symbol", "main"), target=target)
    assert compiled.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.parametrize("device_type", ["cuda", "metal"])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_ragged_causal_runtime(device_type, dtype):
    if not device_enabled(device_type):
        pytest.skip(f"{device_type} not enabled")
    device = tvm_ffi.device(device_type)
    target = tvm_tirx.target.Target.from_device(device)
    # Lengths span several query and KV tiles so later KV tiles are actually skipped.
    lengths = [1, 5, 37, 129]
    indptr = np.array([0, *np.cumsum(lengths)], dtype="int32")
    rng = np.random.default_rng(127)
    q = rng.normal(0, 0.3, (indptr[-1], 8, 32)).astype(dtype)
    k = rng.normal(0, 0.3, (indptr[-1], 2, 32)).astype(dtype)
    v = rng.normal(size=(indptr[-1], 2, 64)).astype(dtype)
    scale = 32**-0.5
    reference = np.zeros((indptr[-1], 8, 64))
    reference_lse = np.zeros((indptr[-1], 8))
    for b, n in enumerate(lengths):
        rows = slice(indptr[b], indptr[b + 1])
        for head in range(8):
            # The kernel works in base 2: LSE is log2(sum(exp2(score))).
            scores = q[rows, head].astype("float64") @ k[rows, head // 4].astype("float64").T
            scores = scores * scale / np.log(2)
            scores[np.triu_indices(n, 1)] = -np.inf
            maximum = scores.max(axis=1, keepdims=True)
            weights = np.exp2(scores - maximum)
            reference[rows, head] = (
                weights @ v[rows, head // 4] / weights.sum(axis=1, keepdims=True)
            )
            reference_lse[rows, head] = maximum[:, 0] + np.log2(weights.sum(axis=1))
    func = _attention_prefill_ragged(2, 8, 32, 64, dtype, {}, target)
    module = tvm.compile(func.with_attr("global_symbol", "main"), target=target)

    def run_and_check():
        arrays = [
            q,
            indptr,
            k,
            v,
            indptr,
            np.arange(indptr[-1], dtype="int32"),
            np.zeros(len(lengths), dtype="int32"),
        ]
        args = [tvm_tirx.runtime.tensor(a, device) for a in arrays]
        output = tvm_tirx.runtime.tensor(np.full(reference.shape, np.nan, dtype=dtype), device)
        lse = tvm_tirx.runtime.tensor(np.full(reference_lse.shape, np.nan, dtype="float32"), device)
        module(*args, output, lse, 1, 0, 1.0, 10000.0, scale)
        rtol, atol = (2e-3, 2e-3) if dtype == "float16" else (2e-4, 2e-5)
        np.testing.assert_allclose(output.numpy(), reference, rtol=rtol, atol=atol)
        np.testing.assert_allclose(lse.numpy(), reference_lse, rtol=0, atol=2e-5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
