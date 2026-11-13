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
# ruff: noqa: F401, F841
"""Test sharded loader"""

# pylint: disable=missing-docstring
import json
import tempfile

import numpy as np
import pytest
import tvm_ffi
from tvm_ffi import Shape, register_global_func

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm.contrib import tvmjs
from tvm.relax import script as R
from tvm.runtime import disco as di
from tvm.s_tir import dlight as dl
from tvm_tirx.script import ir as I
from tvm_tirx.target import Target
from tvm_tirx.testing import env

# `runtime.disco.compiled_ccl` is registered together with the CCL runtime
# functions, so its absence means the disco CCL runtime is not in this build.
_compiled_ccl = tvm_ffi.get_global_func("runtime.disco.compiled_ccl", allow_missing=True)
if _compiled_ccl is None or _compiled_ccl() != "nccl":
    pytest.skip("Disco NCCL support is not available", allow_module_level=True)

# All tests in this file shard across two GPUs.
pytestmark = [
    pytest.mark.skipif(not env.has_multi_gpu(), reason="need multiple gpus"),
]


def _run_with_nccl_session(devices, func):
    def run_and_check():
        sess = di.ThreadedSession(num_workers=len(devices))
        try:
            sess.init_ccl("nccl", *devices)
            return func(sess)
        finally:
            sess.shutdown()

    return tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@register_global_func("tests.disco.shard_dim_0", override=True)
def _shard_dim_0(src, num_shards, tgt):
    s_0, s_1 = src.shape
    tgt.copyfrom(src.numpy().reshape(num_shards, s_0 // num_shards, s_1))


@register_global_func("tests.disco.shard_dim_1", override=True)
def _shard_dim_1(src, num_shards, tgt):
    s_0, s_1 = src.shape
    tgt.copyfrom(src.numpy().reshape(s_0, num_shards, s_1 // num_shards).transpose(1, 0, 2))


@register_global_func("tests.disco.shard_qkv_0", override=True)
def _shard_qkv_0(src, num_shards, q_heads, kv_heads, tgt):
    total_dim, hidden_size = src.shape
    head_dim = total_dim // (q_heads + kv_heads + kv_heads)
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    w_q = src.numpy()[:q_dim, :].reshape(
        num_shards,
        q_heads // num_shards,
        head_dim,
        hidden_size,
    )
    w_k = src.numpy()[q_dim : q_dim + kv_dim, :].reshape(
        num_shards,
        kv_heads // num_shards,
        head_dim,
        hidden_size,
    )
    w_v = src.numpy()[q_dim + kv_dim :, :].reshape(
        num_shards,
        kv_heads // num_shards,
        head_dim,
        hidden_size,
    )
    w_qkv = np.concatenate([w_q, w_k, w_v], axis=1)
    tgt.copyfrom(w_qkv)


@register_global_func("tests.disco.shard_qkv_1", override=True)
def _shard_qkv_1(src, tgt):
    s, _, _, h = src.shape  # pylint: disable=invalid-name
    tgt.copyfrom(src.numpy().reshape(s, -1, h))


def _create_loader(sess, path, param_dict, shard_info):
    path_tensor_cache = path + "/tensor-cache.json"
    tvmjs.dump_tensor_cache(param_dict, path, encode_format="raw")
    with open(path_tensor_cache, encoding="utf-8") as i_f:
        tensor_cache = i_f.read()
    loader_create = sess.get_global_func("runtime.disco.ShardLoader")
    loader = loader_create(path_tensor_cache, tensor_cache, json.dumps(shard_info), None)
    return loader


def _simulate_presharded_weights(base_path, param_dict, num_shards, shard_info):
    """Create fake weights to simulate those produced MLC-LLM's pre-sharding"""

    sharded_params = {}

    for key, ndarray in param_dict.items():
        assert key in shard_info, f"ShardInfo lacks shard info about param: {key}"
        shard_dim = shard_info[key]
        sharded_params[key] = [
            tvm_tirx.runtime.tensor(np_shard)
            for np_shard in np.split(ndarray, num_shards, axis=shard_dim)
        ]

    # Re-order so that the parameter order is sorted first by shard,
    # then by parameter.  This matches the ordering used by MLC-LLM,
    # and avoids having *.bin files that must be accessed by more than
    # one worker.
    sharded_params = {
        f"{key}_shard-{i + 1}-of-{num_shards}": shards[i]
        for i in range(num_shards)
        for key, shards in sharded_params.items()
    }

    tvmjs.dump_tensor_cache(
        sharded_params,
        base_path,
        encode_format="raw",
    )


def test_load_shard():
    devices = [0, 1]
    num_shards = len(devices)
    param_dict = {
        "x_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "x_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = {
        "x_0": [
            [
                "tests.disco.shard_dim_1",
                [(num_shards, 64, 64), "float16"],
                num_shards,
            ],
        ],
        "x_1": [
            [
                "tests.disco.shard_dim_0",
                [(num_shards, 16, 128), "float32"],
                num_shards,
            ]
        ],
    }
    with tempfile.TemporaryDirectory() as path:

        def run_test(sess):
            loader = _create_loader(sess, path, param_dict, shard_info)
            loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoad")
            d_0 = loader_load(loader, Shape([0]))
            d_1 = loader_load(loader, Shape([1]))
            np.testing.assert_equal(
                param_dict["x_0"][:, 0:64],
                d_0.debug_get_from_remote(0).numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_0"][:, 64:128],
                d_0.debug_get_from_remote(1).numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_1"][0:16, :],
                d_1.debug_get_from_remote(0).numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_1"][16:32, :],
                d_1.debug_get_from_remote(1).numpy(),
            )

        _run_with_nccl_session(devices, run_test)


def _create_presharded_loader(sess, path):
    path_tensor_cache = path + "/tensor-cache.json"
    with open(path_tensor_cache, encoding="utf-8") as i_f:
        tensor_cache = i_f.read()
    loader_create = sess.get_global_func("runtime.disco.ShardLoader")
    loader = loader_create(path_tensor_cache, tensor_cache, json.dumps({}), None)
    return loader


def test_load_presharded():
    devices = [0, 1]
    param_dict = {
        "x_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "x_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = {
        "x_0": 1,
        "x_1": 0,
    }

    with tempfile.TemporaryDirectory() as path:
        _simulate_presharded_weights(path, param_dict, len(devices), shard_info)

        def run_test(sess):
            loader = _create_presharded_loader(sess, path)
            loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadPresharded")

            d_0 = loader_load(loader, Shape([0]))
            d_1 = loader_load(loader, Shape([1]))

            np.testing.assert_equal(
                param_dict["x_0"][:, 0:64],
                d_0.debug_get_from_remote(0).numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_0"][:, 64:128],
                d_0.debug_get_from_remote(1).numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_1"][0:16, :],
                d_1.debug_get_from_remote(0).numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_1"][16:32, :],
                d_1.debug_get_from_remote(1).numpy(),
            )

        _run_with_nccl_session(devices, run_test)


def test_load_shard_in_relax():
    devices = [0, 1]
    num_shards = len(devices)
    param_dict = {
        "x_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "x_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = {
        "x_0": [
            [
                "tests.disco.shard_dim_1",
                [(num_shards, 64, 64), "float16"],
                num_shards,
            ],
        ],
        "x_1": [
            [
                "tests.disco.shard_dim_0",
                [(num_shards, 16, 128), "float32"],
                num_shards,
            ]
        ],
    }

    # pylint: disable=invalid-name
    @I.ir_module
    class Module:  # pylint: disable=too-few-public-methods
        @R.function
        def main(
            loader: R.Any,
        ) -> R.Tuple(R.Tensor((64, 64), "float32"), R.Tensor((16, 128), "float32")):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                lv0: R.Tensor((64, 64), "float32") = R.call_pure_packed(
                    "runtime.disco.ShardLoaderLoad",
                    loader,
                    R.shape([0]),
                    ty_args=R.Tensor((64, 64), "float32"),
                )
                lv1: R.Tensor((16, 128), "float32") = R.call_pure_packed(
                    "runtime.disco.ShardLoaderLoad",
                    loader,
                    R.shape([1]),
                    ty_args=R.Tensor((16, 128), "float32"),
                )
                lv2 = (lv0, lv1)
                R.output(lv2)
            return lv2

    # pylint: enable=invalid-name
    def relax_build(mod, target):
        mod = rx.get_pipeline("zero", target=target)(mod)  # pylint: disable=no-value-for-parameter
        return tvm.compile(mod, target="cuda")

    target = Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": 49152,
            "max_threads_per_block": 1024,
            "thread_warp_size": 32,
            "registers_per_block": 65536,
            "arch": "sm_80",
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        dso_path = tmpdir + "/test.so"
        relax_build(Module, target).export_library(dso_path)

        def run_test(sess):
            mod = sess.load_vm_module(dso_path)
            loader = _create_loader(sess, tmpdir, param_dict, shard_info)
            result = mod["main"](loader)
            np.testing.assert_equal(
                param_dict["x_0"][:, 0:64],
                result.debug_get_from_remote(0)[0].numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_0"][:, 64:128],
                result.debug_get_from_remote(1)[0].numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_1"][0:16, :],
                result.debug_get_from_remote(0)[1].numpy(),
            )
            np.testing.assert_equal(
                param_dict["x_1"][16:32, :],
                result.debug_get_from_remote(1)[1].numpy(),
            )

        _run_with_nccl_session(devices, run_test)


def test_load_shard_all():
    devices = [0, 1]
    num_shards = len(devices)
    param_dict = {
        "param_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "param_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = {
        "param_0": [
            [
                "tests.disco.shard_dim_1",
                [(num_shards, 64, 64), "float16"],
                num_shards,
            ],
        ],
        "param_1": [
            [
                "tests.disco.shard_dim_0",
                [(2, 16, 128), "float32"],
                num_shards,
            ]
        ],
    }
    with tempfile.TemporaryDirectory() as path:

        def run_test(sess):
            loader = _create_loader(sess, path, param_dict, shard_info)
            loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAll")
            params = loader_load(loader)
            p_0 = params.debug_get_from_remote(0)
            p_1 = params.debug_get_from_remote(1)
            np.testing.assert_equal(param_dict["param_0"][:, 0:64], p_0[0].numpy())
            np.testing.assert_equal(param_dict["param_0"][:, 64:128], p_1[0].numpy())
            np.testing.assert_equal(param_dict["param_1"][0:16, :], p_0[1].numpy())
            np.testing.assert_equal(param_dict["param_1"][16:32, :], p_1[1].numpy())

        _run_with_nccl_session(devices, run_test)


def test_load_all_presharded():
    devices = [0, 1]
    num_shards = len(devices)
    param_dict = {
        "param_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "param_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = {
        "param_0": 0,
        "param_1": 1,
    }
    with tempfile.TemporaryDirectory() as path:
        _simulate_presharded_weights(path, param_dict, len(devices), shard_info)

        def run_test(sess):
            loader = _create_presharded_loader(sess, path)
            loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAllPresharded")
            params = loader_load(loader)

            p_0 = params.debug_get_from_remote(0)
            p_1 = params.debug_get_from_remote(1)

            np.testing.assert_equal(param_dict["param_0"][0:32, :], p_0[0].numpy())
            np.testing.assert_equal(param_dict["param_0"][32:64, :], p_1[0].numpy())
            np.testing.assert_equal(param_dict["param_1"][:, 0:64], p_0[1].numpy())
            np.testing.assert_equal(param_dict["param_1"][:, 64:128], p_1[1].numpy())

        _run_with_nccl_session(devices, run_test)


def test_load_shard_broadcast():
    devices = [0, 1]
    param_dict = {
        "param_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "param_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = {}
    with tempfile.TemporaryDirectory() as path:

        def run_test(sess):
            loader = _create_loader(sess, path, param_dict, shard_info)
            loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAll")
            params = loader_load(loader)
            p_0 = params.debug_get_from_remote(0)
            p_1 = params.debug_get_from_remote(1)
            np.testing.assert_equal(param_dict["param_0"], p_0[0].numpy())
            np.testing.assert_equal(param_dict["param_0"], p_1[0].numpy())
            np.testing.assert_equal(param_dict["param_1"], p_0[1].numpy())
            np.testing.assert_equal(param_dict["param_1"], p_1[1].numpy())

        _run_with_nccl_session(devices, run_test)


def test_load_qkv_proj_shard():  # pylint: disable=too-many-locals
    devices = [0, 1]
    num_shards = len(devices)
    q_heads = 8
    kv_heads = 10
    head_dim = 10
    hidden_size = 20
    w_q = np.random.uniform(size=[q_heads * head_dim, hidden_size]).astype("float16")
    w_k = np.random.uniform(size=[kv_heads * head_dim, hidden_size]).astype("float16")
    w_v = np.random.uniform(size=[kv_heads * head_dim, hidden_size]).astype("float16")
    w_qkv = np.concatenate([w_q, w_k, w_v], axis=0)
    param_dict = {"w_qkv": w_qkv}
    np_qkv = np.concatenate(
        [
            w_q.reshape((num_shards, q_heads // num_shards, head_dim, hidden_size)),
            w_k.reshape((num_shards, kv_heads // num_shards, head_dim, hidden_size)),
            w_v.reshape((num_shards, kv_heads // num_shards, head_dim, hidden_size)),
        ],
        axis=1,
    ).reshape((num_shards, -1, hidden_size))

    shard_info = {
        "w_qkv": [
            [
                "tests.disco.shard_qkv_0",
                [
                    (num_shards, (q_heads + kv_heads * 2) // num_shards, head_dim, hidden_size),
                    "float16",
                ],
                num_shards,
                q_heads,
                kv_heads,
            ],
            [
                "tests.disco.shard_qkv_1",
                [
                    (num_shards, (q_heads + kv_heads * 2) // num_shards * head_dim, hidden_size),
                    "float16",
                ],
            ],
        ],
    }

    with tempfile.TemporaryDirectory() as path:

        def run_test(sess):
            loader = _create_loader(sess, path, param_dict, shard_info)
            loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoad")
            d_0 = loader_load(loader, Shape([0]))
            np.testing.assert_equal(
                np_qkv[0],
                d_0.debug_get_from_remote(0).numpy(),
            )
            np.testing.assert_equal(
                np_qkv[1],
                d_0.debug_get_from_remote(1).numpy(),
            )

        _run_with_nccl_session(devices, run_test)


if __name__ == "__main__":
    tvm_tirx.testing.main()
