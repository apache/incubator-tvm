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
from tvm import relax
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

has_vllm = tvm_ffi.get_global_func("tvm.contrib.vllm.single_query_cached_kv_attention", True)

vllm_enabled = pytest.mark.skipif(
    not has_vllm,
    reason="VLLM not enabled.",
)

pytestmark = [
    vllm_enabled,
    pytest.mark.gpu,
    pytest.mark.skipif(not env.has_cuda(), reason="need cuda"),
]


def build_and_run(mod, inputs_np, target, legalize=True):
    if legalize:
        mod = relax.transform.LegalizeOps()(mod)

        mod = tvm.s_tir.transform.DefaultGPUSchedule(target=tvm_tirx.target.Target("cuda"))(mod)

    with tvm_tirx.ir.transform.PassContext():
        ex = tvm.compile(mod, target)

    dev = tvm_tirx.cuda(0)
    vm = relax.VirtualMachine(ex, dev)
    f = vm["main"]
    inputs = [tvm_tirx.runtime.tensor(inp, dev) for inp in inputs_np]

    out = f(*inputs)

    if isinstance(out, tvm_ffi.Array):
        return [arr.numpy() for arr in out]

    return out.numpy()


def test_attention():
    num_seqs = T.dynamic("num_seqs", "int64")
    num_blocks = T.dynamic("num_blocks", "int64")
    max_num_blocks_per_seq = T.dynamic("max_num_blocks_per_seq", "int64")

    @I.ir_module
    class ModulePagedAttentionV1:
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def main(
            query: R.Tensor((num_seqs, 1, 64), dtype="float16"),
            key_cache: R.Tensor((num_blocks, 1, 8, 16, 8), dtype="float16"),
            value_cache: R.Tensor((num_blocks, 1, 64, 16), dtype="float16"),
            block_tables: R.Tensor((num_seqs, max_num_blocks_per_seq), dtype="int32"),
            context_lens: R.Tensor((num_seqs,), dtype="int32"),
        ) -> R.Tensor((num_seqs, 1, 64), dtype="float16"):
            with R.dataflow():
                max_len = R.to_vdevice(R.max(context_lens), "llvm:0")
                out = R.call_dps_packed(
                    _ExternFunc("tvm.contrib.vllm.single_query_cached_kv_attention_v1"),
                    [
                        query,
                        key_cache,
                        value_cache,
                        block_tables,
                        context_lens,
                        16,
                        max_len,
                    ],
                    ty_args=[query.ty],
                )
                R.output(out)
            return out

    num_seqs = T.dynamic("num_seqs", "int64")
    num_blocks = T.dynamic("num_blocks", "int64")
    max_num_blocks_per_seq = T.dynamic("max_num_blocks_per_seq", "int64")

    @I.ir_module
    class ModulePagedAttentionV2:
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def main(
            query: R.Tensor((num_seqs, 1, 64), dtype="float16"),
            key_cache: R.Tensor((num_blocks, 1, 8, 16, 8), dtype="float16"),
            value_cache: R.Tensor((num_blocks, 1, 64, 16), dtype="float16"),
            block_tables: R.Tensor((num_seqs, max_num_blocks_per_seq), dtype="int32"),
            context_lens: R.Tensor((num_seqs,), dtype="int32"),
        ) -> R.Tensor((num_seqs, 1, 64), dtype="float16"):
            with R.dataflow():
                max_len = R.to_vdevice(R.max(context_lens), "llvm:0")
                # alloc workspace
                exp_sums = R.zeros((num_seqs, 1, 1), "float32")
                max_logits = R.zeros((num_seqs, 1, 1), "float32")
                tmp_out = R.zeros((num_seqs, 1, 1, 64), "float16")

                out = R.call_dps_packed(
                    _ExternFunc("tvm.contrib.vllm.single_query_cached_kv_attention_v2"),
                    [
                        query,
                        key_cache,
                        value_cache,
                        block_tables,
                        context_lens,
                        16,
                        max_len,
                        exp_sums,
                        max_logits,
                        tmp_out,
                    ],
                    ty_args=[query.ty],
                )
                R.output(out)
            return out

    np.random.seed(0)
    num_heads = 1
    head_dim = 64
    vec_size = 8
    block_size = 16
    num_seqs = 2
    num_blocks = 1
    query = np.random.randn(num_seqs, num_heads, head_dim).astype("float16")
    key_cache = np.random.randn(
        num_blocks, num_heads, head_dim // vec_size, block_size, vec_size
    ).astype("float16")
    value_cache = np.random.randn(num_blocks, num_heads, head_dim, block_size).astype("float16")
    block_tables = np.array([[0], [0]]).astype("int32")
    context_lens = np.array([3, 5]).astype("int32")

    out_v1 = build_and_run(
        ModulePagedAttentionV1,
        [query, key_cache, value_cache, block_tables, context_lens],
        "cuda",
        legalize=True,
    )

    out_v2 = build_and_run(
        ModulePagedAttentionV2,
        [query, key_cache, value_cache, block_tables, context_lens],
        "cuda",
        legalize=True,
    )

    ref = np.array(
        [
            [
                [
                    0.28271484375,
                    0.197021484375,
                    -0.278564453125,
                    0.444580078125,
                    -0.47802734375,
                    -0.7548828125,
                    -0.84228515625,
                    -0.80322265625,
                    0.478759765625,
                    0.195068359375,
                    -0.59521484375,
                    0.779296875,
                    0.35888671875,
                    -0.158935546875,
                    -0.6103515625,
                    0.188720703125,
                    0.410400390625,
                    0.28662109375,
                    0.40283203125,
                    -1.23046875,
                    -0.01043701171875,
                    -0.0794677734375,
                    -0.0350341796875,
                    0.12005615234375,
                    0.63671875,
                    0.368896484375,
                    -0.58642578125,
                    -0.34228515625,
                    -0.552734375,
                    0.947265625,
                    -0.079833984375,
                    0.85302734375,
                    0.1947021484375,
                    0.16748046875,
                    -0.083984375,
                    -0.75244140625,
                    -0.568359375,
                    -1.45703125,
                    -1.021484375,
                    -0.2235107421875,
                    -0.98828125,
                    -0.87109375,
                    -0.43359375,
                    -0.3271484375,
                    0.0557861328125,
                    -0.269287109375,
                    -1.009765625,
                    0.1387939453125,
                    -0.0831298828125,
                    0.27978515625,
                    -0.9736328125,
                    0.7802734375,
                    -0.1329345703125,
                    -0.5927734375,
                    -1.6923828125,
                    1.1904296875,
                    -1.3759765625,
                    -1.080078125,
                    -0.53173828125,
                    0.28466796875,
                    -2.02734375,
                    -0.377685546875,
                    -0.81201171875,
                    -0.7412109375,
                ]
            ],
            [
                [
                    0.482177734375,
                    0.114501953125,
                    -0.265869140625,
                    -1.154296875,
                    0.28857421875,
                    0.71240234375,
                    -1.1767578125,
                    0.187744140625,
                    -0.23486328125,
                    0.07135009765625,
                    -0.34521484375,
                    0.444091796875,
                    -0.09130859375,
                    0.900390625,
                    -0.043701171875,
                    0.61279296875,
                    0.1201171875,
                    0.443603515625,
                    -0.4150390625,
                    -0.9560546875,
                    -0.1917724609375,
                    0.0863037109375,
                    0.267578125,
                    0.04931640625,
                    -0.32666015625,
                    0.5859375,
                    -0.57421875,
                    0.29541015625,
                    -0.26220703125,
                    1.177734375,
                    0.11309814453125,
                    0.81201171875,
                    0.346435546875,
                    0.53271484375,
                    -0.0009765625,
                    -0.35205078125,
                    -0.1298828125,
                    -1.2431640625,
                    -0.2196044921875,
                    0.31640625,
                    -0.40869140625,
                    0.25244140625,
                    -0.9853515625,
                    0.284912109375,
                    0.399169921875,
                    -1.1435546875,
                    0.305419921875,
                    0.300048828125,
                    -0.84521484375,
                    -0.5166015625,
                    -0.787109375,
                    0.1011962890625,
                    -1.0302734375,
                    -1.35546875,
                    -0.0556640625,
                    1.0791015625,
                    -0.047607421875,
                    -0.498046875,
                    -0.055999755859375,
                    -0.35009765625,
                    -1.4296875,
                    0.350341796875,
                    -1.16796875,
                    -0.576171875,
                ]
            ],
        ]
    ).astype("float16")

    # from vllm import attention_ops
    # import torch
    #
    # def to_torch(arr):
    #     return torch.from_numpy(arr).to("cuda")
    #
    # ref = to_torch(np.zeros_like(query))
    # attention_ops.single_query_cached_kv_attention(
    #     ref,
    #     to_torch(query),
    #     to_torch(key_cache),
    #     to_torch(value_cache),
    #     num_kv_heads,
    #     query.shape[-1] ** -0.5,  # scale
    #     to_torch(block_tables),
    #     to_torch(context_lens),
    #     value_cache.shape[-1],  # block_size,
    #     np.max(context_lens),
    #     None,
    #     )
    # ref = ref.cpu().numpy()

    # print(ref.tolist())

    for out in [out_v1, out_v2]:
        assert np.max(np.abs(ref - out)) == 0.0


def test_cache():
    num_tokens = T.dynamic("num_tokens", "int64")
    num_blocks = T.dynamic("num_blocks", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(
            key: R.Tensor((num_tokens, 1, 8), dtype="float16"),
            value: R.Tensor((num_tokens, 1, 8), dtype="float16"),
            key_cache: R.Tensor((num_blocks, 1, 1, 16, 8), dtype="float16"),
            value_cache: R.Tensor((num_blocks, 1, 8, 16), dtype="float16"),
            slot_mapping: R.Tensor((num_tokens,), dtype="int32"),
        ) -> R.Tuple(
            [
                R.Tensor((num_blocks, 1, 8, 16, 8), dtype="float16"),
                R.Tensor((num_blocks, 1, 8, 16), dtype="float16"),
            ]
        ):
            with R.dataflow():
                kv = R.call_pure_packed(
                    "tvm.contrib.vllm.reshape_and_cache",
                    key,
                    value,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    ty_args=[key_cache.ty, value_cache.ty],
                )
                out = (kv[0], kv[1])
                R.output(out)
            return out

    np.random.seed(0)
    num_heads = 1
    head_dim = 8
    vec_size = 8
    block_size = 16
    num_tokens = 8
    num_blocks = 1
    key = np.random.randn(num_tokens, num_heads, head_dim).astype("float16")
    value = np.random.randn(num_tokens, num_heads, head_dim).astype("float16")
    key_cache_before = np.random.randn(
        num_blocks, num_heads, head_dim // vec_size, block_size, vec_size
    ).astype("float16")
    value_cache_before = np.random.randn(num_blocks, num_heads, head_dim, block_size).astype(
        "float16"
    )
    slot_mapping = np.arange(num_tokens).astype("int32")

    key_cache = key_cache_before.copy()
    value_cache = value_cache_before.copy()

    out_key_cache, out_value_cache = build_and_run(
        Module,
        [key, value, key_cache, value_cache, slot_mapping],
        "cuda",
    )

    ref_key_cache = np.array(
        [
            [
                [
                    [
                        [
                            1.763671875,
                            0.400146484375,
                            0.978515625,
                            2.240234375,
                            1.8671875,
                            -0.97705078125,
                            0.9501953125,
                            -0.1513671875,
                        ],
                        [
                            -0.10321044921875,
                            0.41064453125,
                            0.14404296875,
                            1.4541015625,
                            0.76123046875,
                            0.1217041015625,
                            0.44384765625,
                            0.333740234375,
                        ],
                        [
                            1.494140625,
                            -0.2052001953125,
                            0.31298828125,
                            -0.85400390625,
                            -2.552734375,
                            0.65380859375,
                            0.8642578125,
                            -0.7421875,
                        ],
                        [
                            2.26953125,
                            -1.4541015625,
                            0.045745849609375,
                            -0.1871337890625,
                            1.533203125,
                            1.4697265625,
                            0.1549072265625,
                            0.378173828125,
                        ],
                        [
                            -0.8876953125,
                            -1.98046875,
                            -0.347900390625,
                            0.1563720703125,
                            1.23046875,
                            1.2021484375,
                            -0.38720703125,
                            -0.30224609375,
                        ],
                        [
                            -1.048828125,
                            -1.419921875,
                            -1.7060546875,
                            1.951171875,
                            -0.509765625,
                            -0.43798828125,
                            -1.2529296875,
                            0.77734375,
                        ],
                        [
                            -1.6142578125,
                            -0.2127685546875,
                            -0.8955078125,
                            0.386962890625,
                            -0.5107421875,
                            -1.1806640625,
                            -0.0281829833984375,
                            0.42822265625,
                        ],
                        [
                            0.0665283203125,
                            0.302490234375,
                            -0.63427734375,
                            -0.36279296875,
                            -0.67236328125,
                            -0.359619140625,
                            -0.81298828125,
                            -1.7265625,
                        ],
                        [
                            -0.039276123046875,
                            -1.16796875,
                            0.5234375,
                            -0.1715087890625,
                            0.77197265625,
                            0.82373046875,
                            2.1640625,
                            1.3369140625,
                        ],
                        [
                            -0.369140625,
                            -0.2393798828125,
                            1.099609375,
                            0.6552734375,
                            0.64013671875,
                            -1.6171875,
                            -0.024322509765625,
                            -0.73779296875,
                        ],
                        [
                            0.280029296875,
                            -0.09814453125,
                            0.91015625,
                            0.317138671875,
                            0.7861328125,
                            -0.46630859375,
                            -0.9443359375,
                            -0.41015625,
                        ],
                        [
                            -0.0170135498046875,
                            0.379150390625,
                            2.259765625,
                            -0.042266845703125,
                            -0.9560546875,
                            -0.345947265625,
                            -0.463623046875,
                            0.4814453125,
                        ],
                        [
                            -1.541015625,
                            0.063232421875,
                            0.156494140625,
                            0.232177734375,
                            -0.59716796875,
                            -0.2379150390625,
                            -1.423828125,
                            -0.493408203125,
                        ],
                        [
                            -0.54296875,
                            0.416015625,
                            -1.15625,
                            0.78125,
                            1.494140625,
                            -2.0703125,
                            0.42626953125,
                            0.6767578125,
                        ],
                        [
                            -0.63720703125,
                            -0.397216796875,
                            -0.1329345703125,
                            -0.2978515625,
                            -0.30908203125,
                            -1.67578125,
                            1.15234375,
                            1.080078125,
                        ],
                        [
                            -0.8134765625,
                            -1.466796875,
                            0.52099609375,
                            -0.57568359375,
                            0.1419677734375,
                            -0.3193359375,
                            0.69140625,
                            0.69482421875,
                        ],
                    ]
                ]
            ]
        ]
    ).astype("float16")

    ref_value_cache = np.array(
        [
            [
                [
                    [
                        0.1773681640625,
                        1.1396484375,
                        -1.1650390625,
                        -1.0703125,
                        0.010498046875,
                        -1.1728515625,
                        -0.861328125,
                        0.37646484375,
                        -1.9365234375,
                        0.188720703125,
                        0.52392578125,
                        0.08843994140625,
                        -0.310791015625,
                        0.097412109375,
                        0.39892578125,
                        -2.7734375,
                    ],
                    [
                        -0.40185546875,
                        -1.234375,
                        0.90087890625,
                        1.0546875,
                        1.7861328125,
                        1.943359375,
                        1.91015625,
                        -1.099609375,
                        -0.11053466796875,
                        1.0205078125,
                        -0.69189453125,
                        1.5361328125,
                        0.286376953125,
                        0.60888671875,
                        -1.044921875,
                        1.2109375,
                    ],
                    [
                        -1.6298828125,
                        0.40234375,
                        0.465576171875,
                        -0.403076171875,
                        0.126953125,
                        -0.41357421875,
                        -0.26806640625,
                        0.29833984375,
                        0.09771728515625,
                        0.5830078125,
                        -0.3994140625,
                        0.3701171875,
                        -1.306640625,
                        1.658203125,
                        -0.1181640625,
                        -0.68017578125,
                    ],
                    [
                        0.462890625,
                        -0.6845703125,
                        -1.5361328125,
                        1.22265625,
                        0.402099609375,
                        -0.74755859375,
                        0.80224609375,
                        1.326171875,
                        -1.126953125,
                        -0.73046875,
                        -0.384765625,
                        0.0943603515625,
                        -0.04217529296875,
                        -0.286865234375,
                        -0.061614990234375,
                        -0.1072998046875,
                    ],
                    [
                        -0.9072265625,
                        -0.87060546875,
                        1.48828125,
                        0.208251953125,
                        1.8828125,
                        1.9228515625,
                        0.947265625,
                        -0.6943359375,
                        -0.70458984375,
                        0.943359375,
                        0.7470703125,
                        -1.1884765625,
                        0.7734375,
                        -1.18359375,
                        -2.658203125,
                        0.6064453125,
                    ],
                    [
                        0.05194091796875,
                        -0.57861328125,
                        1.8955078125,
                        0.9765625,
                        -1.34765625,
                        1.48046875,
                        -0.155029296875,
                        -0.149658203125,
                        -0.44091796875,
                        -0.2802734375,
                        -0.36474609375,
                        0.15673828125,
                        0.57861328125,
                        0.349609375,
                        -0.76416015625,
                        -1.4375,
                    ],
                    [
                        0.72900390625,
                        -0.3115234375,
                        1.1787109375,
                        0.3564453125,
                        -1.2705078125,
                        1.8671875,
                        0.6142578125,
                        -0.43505859375,
                        0.6982421875,
                        0.0037708282470703125,
                        0.931640625,
                        0.33984375,
                        -0.01568603515625,
                        0.160888671875,
                        -0.190673828125,
                        -0.394775390625,
                    ],
                    [
                        0.1290283203125,
                        0.05615234375,
                        -0.179931640625,
                        0.70654296875,
                        0.96923828125,
                        0.90625,
                        0.92236328125,
                        1.849609375,
                        0.6435546875,
                        -1.5703125,
                        -0.2069091796875,
                        0.88037109375,
                        -1.6982421875,
                        0.38720703125,
                        -2.255859375,
                        -1.0224609375,
                    ],
                ]
            ]
        ]
    ).astype("float16")

    # from vllm import cache_ops
    # import torch

    # def to_torch(arr):
    #     return torch.from_numpy(arr).to("cuda")

    # ref_key_cache = to_torch(key_cache_before.copy())
    # ref_value_cache = to_torch(value_cache_before.copy())

    # cache_ops.reshape_and_cache(
    #     to_torch(key),
    #     to_torch(value),
    #     ref_key_cache,
    #     ref_value_cache,
    #     to_torch(slot_mapping),
    # )

    # ref_key_cache = ref_key_cache.cpu().numpy()
    # ref_value_cache = ref_value_cache.cpu().numpy()

    assert np.max(np.abs(out_key_cache - ref_key_cache)) == 0
    assert np.max(np.abs(out_value_cache - ref_value_cache)) == 0


def test_reconstruct_from_cache():
    num_heads = 1
    head_dim = 8
    vec_size = 8
    block_size = 16
    num_tokens = 8
    num_blocks = 1

    reshape_and_cache_func = tvm_ffi.get_global_func("tvm.contrib.vllm.reshape_and_cache")
    reconstruct_from_cache_func = tvm_ffi.get_global_func("tvm.contrib.vllm.reconstruct_from_cache")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        key = tvm_tirx.runtime.tensor(
            np.random.randn(num_tokens, num_heads, head_dim).astype("float16"), dev
        )
        value = tvm_tirx.runtime.tensor(
            np.random.randn(num_tokens, num_heads, head_dim).astype("float16"), dev
        )
        slot_mapping = tvm_tirx.runtime.tensor(np.arange(num_tokens).astype("int32"), dev)

        k_cache = tvm_tirx.runtime.tensor(
            np.random.randn(
                num_blocks, num_heads, head_dim // vec_size, block_size, vec_size
            ).astype("float16"),
            dev,
        )
        v_cache = tvm_tirx.runtime.tensor(
            np.random.randn(num_blocks, num_heads, head_dim, block_size).astype("float16"), dev
        )

        reshape_and_cache_func(key, value, k_cache, v_cache, slot_mapping)
        out = reconstruct_from_cache_func(k_cache, v_cache, slot_mapping)

        np.testing.assert_equal(key.numpy(), out[0].numpy())
        np.testing.assert_equal(value.numpy(), out[1].numpy())

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
