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

"""YaRN frequencies must agree for the two coordinates of each rotary pair."""

import math

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm.relax.frontend.nn.llm.position_embedding import (
    llama_rope_with_position_map,
    switch_rope_freq_func,
)
from tvm.s_tir import dlight as dl
from tvm_tirx import tirx

TARGETS = [
    "llvm",
    pytest.param("cuda", marks=pytest.mark.gpu),
    pytest.param("metal", marks=pytest.mark.gpu),
]


def _yarn_config(factor):
    return {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": 16384,
        "beta_fast": 32,
        "beta_slow": 1,
        "inv_theta_log_scale": 1 / (2 * math.log(1_000_000)),
    }


def _build_rope(kernel, target):
    if not tvm_tirx.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")
    mod = tvm_tirx.IRModule({"main": kernel.with_attr("global_symbol", "main")})
    if target != "llvm":

        def get_target():
            return tvm_tirx.target.Target.from_device(tvm_tirx.device_from_target(target))

        target = tvm_tirx.testing.run_with_gpu_lock(get_target)
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    return tvm_tirx.tirx.compile(mod["main"], target=target).mod


@pytest.mark.parametrize("rotary_dim", [64, 128, 192])
@pytest.mark.parametrize("factor", [4, 16])
def test_yarn_rotary_pairs(rotary_dim, factor):
    analyzer = tvm_tirx.sym.Analyzer()
    rope_freq = switch_rope_freq_func(_yarn_config(factor))
    frequencies = []
    for d in range(rotary_dim):
        _, _, values = rope_freq(
            tirx.const(1, "float32"),
            tirx.const(d, "int32"),
            rotary_dim,
            1_000_000,
            "float32",
        )
        (frequency,) = values.values()
        frequencies.append(analyzer.simplify(frequency))

    half = rotary_dim // 2
    for d in range(half):
        assert analyzer.can_prove_equal(frequencies[d], frequencies[d + half]), d


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(("head_dim", "rotary_dim"), [(128, 64), (128, 128), (192, 192)])
@pytest.mark.parametrize("factor", [1, 16])
@pytest.mark.parametrize("apply_rope", [0, 1])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_yarn_fused_rope_preserves_norm(target, head_dim, rotary_dim, factor, apply_rope, dtype):
    kernel = llama_rope_with_position_map(
        theta=1_000_000,
        scale=1,
        head_dim=head_dim,
        num_q_heads=2,
        num_kv_heads=1,
        dtype=dtype,
        rope_scaling=_yarn_config(factor),
        rotary_dim=rotary_dim,
    )
    module = _build_rope(kernel, target)
    positions = np.array([0, 1, 16, 4095, 16383, 16384, 262143], dtype="int32")
    qkv = np.random.default_rng(0).normal(size=(len(positions), 4, head_dim)).astype(dtype)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target)
        q = tvm_tirx.runtime.empty((len(positions), 2, head_dim), dtype, dev)
        k = tvm_tirx.runtime.empty((len(positions), 1, head_dim), dtype, dev)
        v = tvm_tirx.runtime.empty((len(positions), 1, head_dim), dtype, dev)
        module["main"](
            tvm_tirx.runtime.tensor(qkv, dev),
            tvm_tirx.runtime.tensor(positions, dev),
            q,
            k,
            v,
            apply_rope,
        )
        dev.sync()
        for actual, original in [(q.numpy(), qkv[:, :2]), (k.numpy(), qkv[:, 2:3])]:
            if apply_rope:
                np.testing.assert_allclose(
                    np.square(actual.astype("float32")).sum(axis=-1),
                    np.square(original.astype("float32")).sum(axis=-1),
                    rtol=5e-4 if dtype == "float16" else 2e-6,
                    atol=2e-5,
                )
                np.testing.assert_array_equal(actual[..., rotary_dim:], original[..., rotary_dim:])
            else:
                np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(v.numpy(), qkv[:, 3:])

    if target == "llvm":
        run_and_check()
    else:
        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "beta_fast,indices,frequencies",
    [
        (32, [0, 32, 63], [1.0, 0.00032175992234979915, 7.755861004698247e-8]),
        (1, [0, 36, 37], [1.0, 0.00042169650342858224, 2.1238802055890998e-5]),
    ],
)
def test_yarn_known_frequencies(target, beta_fast, indices, frequencies):
    config = _yarn_config(16)
    config["beta_fast"] = beta_fast
    kernel = llama_rope_with_position_map(
        theta=1_000_000,
        scale=1,
        head_dim=128,
        num_q_heads=1,
        num_kv_heads=1,
        dtype="float32",
        rope_scaling=config,
    )
    module = _build_rope(kernel, target)
    # Keep the tested angles below pi for CUDA's default fast math.
    # Low frequencies still use position 4096 to expose interpolation errors.
    positions = np.array([1, 4096, 4096], dtype="int32")
    qkv = np.zeros((len(positions), 3, 128), dtype="float32")
    expected = np.zeros((len(positions), 128), dtype="float32")
    # The equal-beta case has a zero-width ramp between coordinates 36 and 37.
    for row, (position, index, frequency) in enumerate(zip(positions, indices, frequencies)):
        angle = position * frequency
        qkv[row, :, index] = 1
        expected[row, index] = math.cos(angle)
        expected[row, index + 64] = math.sin(angle)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target)
        q, k, v = [
            tvm_tirx.runtime.empty((len(positions), 1, 128), "float32", dev) for _ in range(3)
        ]
        module["main"](
            tvm_tirx.runtime.tensor(qkv, dev),
            tvm_tirx.runtime.tensor(positions, dev),
            q,
            k,
            v,
            1,
        )
        dev.sync()
        for output in [q, k]:
            np.testing.assert_allclose(output.numpy()[:, 0], expected, rtol=1e-5, atol=1e-6)

    if target == "llvm":
        run_and_check()
    else:
        tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
