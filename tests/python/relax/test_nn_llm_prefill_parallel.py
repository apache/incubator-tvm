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
import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax.frontend.nn.llm._prefill_kernels import (
    _attention_prefill,
    _attention_prefill_ragged,
)
from tvm.relax.frontend.nn.llm._prefill_parallel import (
    _attention_prefill_paged_parallel,
    _attention_prefill_ragged_parallel,
)
from tvm.testing import device_enabled


@pytest.mark.parametrize("layout", ["ragged", "paged", "sliding"])
def test_webgpu_parallel_prefill_compiles(layout):
    target = tvm_tirx.target.Target("webgpu")
    if layout == "ragged":
        func = _attention_prefill_ragged(1, 8, 512, 512, "float32", {}, target)
    else:
        func = _attention_prefill(1, 8, 512, "float32", layout == "sliding", {}, target)
    compiled = tvm_tirx.tirx.compile(func.with_attr("global_symbol", "main"), target=target)
    assert compiled.mod.imports[0].inspect_source()


def rope(values, positions):
    frequency = positions[:, None] / 10000 ** (np.arange(256, dtype="float64") / 256)
    angles = np.concatenate([frequency, frequency], axis=-1)[:, None, :]
    rotated = np.concatenate([-values[..., 256:], values[..., :256]], axis=-1)
    return values * np.cos(angles) + rotated * np.sin(angles)


@pytest.mark.gpu
@pytest.mark.parametrize("device_type", ["cuda", "metal"])
@pytest.mark.parametrize("layout", ["ragged", "paged", "sliding"])
@pytest.mark.parametrize("rotary", [0, 1])
@pytest.mark.parametrize("causal", [0, 1])
@pytest.mark.parametrize("first_kv_length", [1, 23])
def test_parallel_prefill(device_type, layout, rotary, causal, first_kv_length):
    if not device_enabled(device_type):
        pytest.skip(f"{device_type} not enabled")
    device = tvm_ffi.device(device_type)
    target = tvm_tirx.target.Target.from_device(device)
    rng = np.random.default_rng(127)
    lengths = [(7, first_kv_length), (17, 513)]
    qi = np.array([0, 7, 24], dtype="int32")
    ki = np.array([0, first_kv_length, first_kv_length + 513], dtype="int32")
    q = rng.normal(0, 0.3, (24, 8, 512)).astype("float32")
    k = rng.normal(0, 0.3, (first_kv_length + 513, 2, 512)).astype("float32")
    v = rng.normal(size=(first_kv_length + 513, 2, 512)).astype("float32")
    kp = np.array([32, 1234], dtype="int32")
    qp = np.concatenate(
        [np.arange(nk - nq, nk) + kp[b] for b, (nq, nk) in enumerate(lengths)]
    ).astype("int32")
    reference = np.zeros_like(q)
    reference_lse = np.full((24, 8), -50000, dtype="float32")
    scale = 512**-0.5
    for b, (nq, nk) in enumerate(lengths):
        query = q[qi[b] : qi[b + 1]].astype("float64")
        keys = k[ki[b] : ki[b + 1]].astype("float64")
        if rotary:
            query = rope(query, qp[qi[b] : qi[b + 1]])
            keys = rope(keys, np.arange(nk) + kp[b])
        keys = np.repeat(keys, 4, axis=1)
        values = np.repeat(v[ki[b] : ki[b + 1]], 4, axis=1)
        scores = np.einsum("qhd,khd->hqk", query, keys) * scale
        mask = np.ones((nq, nk), dtype=bool)
        if causal:
            if layout == "sliding":
                mask = np.arange(nk)[None, :] >= np.maximum(
                    nk - np.maximum(15 - np.arange(nq)[:, None], 0), 0
                )
            else:
                mask = np.arange(nk)[None, :] <= np.arange(nq)[:, None] + nk - nq
        for row in range(nq):
            if not mask[row].any():
                continue
            scores_row = scores[:, row, mask[row]]
            maximum = scores_row.max(axis=-1, keepdims=True)
            weights = np.exp(scores_row - maximum)
            denominator = weights.sum(axis=-1, keepdims=True)
            reference[qi[b] + row] = np.einsum(
                "hk,khd->hd", weights / denominator, values[mask[row]]
            )
            reference_lse[qi[b] + row] = ((maximum + np.log(denominator)) / np.log(2))[:, 0]
    output = tvm_tirx.runtime.empty(q.shape, "float32", device)
    lse = tvm_tirx.runtime.empty(reference_lse.shape, "float32", device)

    def tensor(value):
        return tvm_tirx.runtime.tensor(value, device)

    if layout == "ragged":
        func = _attention_prefill_ragged_parallel(2, 8, {})
        args = [tensor(q), tensor(qi), tensor(k), tensor(v), tensor(ki), tensor(qp), tensor(kp)]
    else:
        sliding = layout == "sliding"
        offsets = [3, 17] if sliding else [0, 0]
        sinks = [0, 2] if sliding else [0, 0]
        physical_lengths = [
            nk + offset - sink for (_, nk), offset, sink in zip(lengths, offsets, sinks)
        ]
        counts = [(n + 15) // 16 for n in physical_lengths]
        pi = np.array([0, counts[0], sum(counts)], dtype="int32")
        indices = rng.permutation(sum(counts)).astype("int32")
        pages = np.zeros((sum(counts), 2, 2, 16, 512), dtype="float32")
        for b, (_, nk) in enumerate(lengths):
            for row in range(nk):
                slot = row if row < sinks[b] else row - sinks[b] + offsets[b]
                page = indices[pi[b] + slot // 16]
                pages[page, 0, :, slot % 16] = k[ki[b] + row]
                pages[page, 1, :, slot % 16] = v[ki[b] + row]
        last = [(n - 1) % 16 + 1 for n in physical_lengths]
        info = np.array([last, offsets, sinks] if sliding else last, dtype="int32")
        func = _attention_prefill_paged_parallel(2, 8, {}, sliding, sliding_window_size=16)
        args = [
            tensor(q),
            tensor(qi),
            tensor(pages),
            tensor(pi),
            tensor(indices),
            tensor(info),
            tensor(kp),
            tensor(qp),
        ]
    module = tvm.compile(
        tvm_tirx.IRModule({"main": func.with_attr("global_symbol", "main")}), target=target
    )
    module(*args, output, lse, causal, rotary, 1.0, 10000.0, scale)
    tvm_tirx.testing.assert_allclose(output.numpy(), reference, rtol=2e-4, atol=2e-5)
    tvm_tirx.testing.assert_allclose(lse.numpy(), reference_lse, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    tvm_tirx.testing.main()
