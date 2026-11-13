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
# ruff: noqa: F841

#  type: ignore

from __future__ import annotations

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.s_tir import script as Ts
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def test_mlp():
    @I.ir_module
    class MLP:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                    R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((128, 128), "float32"),
            weight1: R.Tensor((128, 128), "float32"),
            weight2: R.Tensor((128, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            lv0 = R.matmul(x, weight1)
            lv1 = R.nn.gelu(lv0)
            lv2 = R.dist.annotate_sharding(lv1, device_mesh="mesh[0]", placement="S[1]")
            lv3 = R.matmul(lv2, weight2)
            return lv3

    @I.ir_module
    class ShardedMLP:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
            weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        ) -> R.DTensor((128, 128), "float32", "mesh[0]", "R"):
            lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.matmul(x, weight1)
            lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
            lv3: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.matmul(lv1, weight2)
            return lv3

    after = relax.distributed.transform.PropagateSharding()(MLP)
    assert_structural_equal(after, ShardedMLP)


def test_mlp_with_tuple():
    @I.ir_module
    class MLPWithTuple:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                    R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @Ts.function(private=True)
        def split1(
            A: T.Tensor((128, 128), "float32"),
            T_split: T.Tensor((64, 128), "float32"),
            T_split_1: T.Tensor((64, 128), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax1, ax2 in T.grid(64, 128):
                with Ts.sblock("T_split"):
                    v_ax1, v_ax2 = Ts.axis.remap("SS", [ax1, ax2])
                    T_split[v_ax1, v_ax2] = A[v_ax1, v_ax2]
            for ax1, ax2 in T.grid(64, 128):
                with Ts.sblock("T_split_1"):
                    v_ax1, v_ax2 = Ts.axis.remap("SS", [ax1, ax2])
                    T_split_1[v_ax1, v_ax2] = A[v_ax1 + 64, v_ax2]

        @R.function
        def foo(
            x: R.Tensor((128, 128), "float32"),
            weight_packed: R.Tuple(
                R.Tensor((128, 128), "float32"), R.Tensor((128, 128), "float32")
            ),
        ) -> R.Tensor((64, 128), "float32"):
            cls = MLPWithTuple
            weight1 = weight_packed[0]
            lv0 = R.matmul(x, weight1)
            lv1 = R.nn.gelu(lv0)
            lv_tuple = R.call_tir(
                cls.split1,
                [lv1],
                ty_args=[
                    _TupleType([R.Tensor((64, 128), "float32"), R.Tensor((64, 128), "float32")])
                ],
            )
            lv2 = lv_tuple[0]
            lv3 = R.dist.annotate_sharding(lv2, device_mesh="mesh[0]", placement="S[1]")
            weight2 = weight_packed[1]
            lv4 = R.matmul(lv3, weight2)
            return lv4

    @I.ir_module
    class ShardedMLPWithTuple:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @Ts.function(private=True)
        def split1(
            A: T.Tensor((128, 128), "float32"),
            T_split: T.Tensor((64, 128), "float32"),
            T_split_1: T.Tensor((64, 128), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax1, ax2 in T.grid(64, 128):
                with Ts.sblock("T_split"):
                    v_ax1, v_ax2 = Ts.axis.remap("SS", [ax1, ax2])
                    Ts.reads(A[v_ax1, v_ax2])
                    Ts.writes(T_split[v_ax1, v_ax2])
                    T_split[v_ax1, v_ax2] = A[v_ax1, v_ax2]
            for ax1, ax2 in T.grid(64, 128):
                with Ts.sblock("T_split_1"):
                    v_ax1, v_ax2 = Ts.axis.remap("SS", [ax1, ax2])
                    Ts.reads(A[v_ax1 + 64, v_ax2])
                    Ts.writes(T_split_1[v_ax1, v_ax2])
                    T_split_1[v_ax1, v_ax2] = A[v_ax1 + 64, v_ax2]

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            weight_packed: R.Tuple(
                R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
                R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
            ),
        ) -> R.DTensor((64, 128), "float32", "mesh[0]", "R"):
            cls = ShardedMLPWithTuple
            weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = weight_packed[0]
            lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.matmul(x, weight1)
            lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
            gv = R.dist.call_tir(
                cls.split1,
                (lv1,),
                ty_args=[
                    _TupleType(
                        [
                            R.DTensor((64, 128), "float32", "mesh[0]", "S[1]"),
                            R.DTensor((64, 128), "float32", "mesh[0]", "S[1]"),
                        ]
                    )
                ],
            )
            lv2: R.DTensor((64, 128), "float32", "mesh[0]", "S[1]") = gv[0]
            weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]") = weight_packed[1]
            lv4: R.DTensor((64, 128), "float32", "mesh[0]", "R") = R.matmul(lv2, weight2)
            return lv4

    after = relax.distributed.transform.PropagateSharding()(MLPWithTuple)
    assert_structural_equal(after, ShardedMLPWithTuple)


def test_mlp_const():
    @I.ir_module
    class MLPWithConst:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                    R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((128, 128), "float32"),
            weight1: R.Tensor((128, 128), "float32"),
            weight2: R.Tensor((128, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            lv0 = R.matmul(x, weight1)
            lv1 = R.nn.gelu(lv0)
            lv2 = R.add(lv1, R.const(2, "float32"))
            lv3 = R.dist.annotate_sharding(lv2, device_mesh="mesh[0]", placement="S[1]")
            lv4 = R.matmul(lv3, weight2)
            return lv4

    @I.ir_module
    class ShardedMLPWithConst:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
            weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        ) -> R.DTensor((128, 128), "float32", "mesh[0]", "R"):
            lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.matmul(x, weight1)
            lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
            lv2: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.add(
                lv1, R.dist.const(2, R.DTensor((), "float32", "mesh[0]", "R"))
            )
            lv4: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.matmul(lv2, weight2)
            return lv4

    after = relax.distributed.transform.PropagateSharding()(MLPWithConst)
    assert_structural_equal(after, ShardedMLPWithConst)


def test_mlp_dynamic_shape():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    k0 = T.dynamic("k0", "int64")
    k1 = T.dynamic("k1", "int64")

    @I.ir_module
    class MLPDynamicShape:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                    R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((m, k0), "float32"),
            weight1: R.Tensor((k0, k1), "float32"),
            weight2: R.Tensor((k1, n), "float32"),
        ) -> R.Tensor((m, n), "float32"):
            lv0 = R.matmul(x, weight1)
            lv1 = R.nn.gelu(lv0)
            lv2 = R.dist.annotate_sharding(lv1, device_mesh="mesh[0]", placement="S[1]")
            lv3 = R.matmul(lv2, weight2)
            return lv3

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    k0 = T.dynamic("k0", "int64")
    k1 = T.dynamic("k1", "int64")

    @I.ir_module
    class ShardedMLPDynamicShape:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @R.function
        def foo(
            x: R.DTensor((m, k0), "float32", "mesh[0]", "R"),
            weight1: R.DTensor((k0, k1), "float32", "mesh[0]", "S[1]"),
            weight2: R.DTensor((k1, n), "float32", "mesh[0]", "S[0]"),
        ) -> R.DTensor((m, n), "float32", "mesh[0]", "R"):
            lv0: R.DTensor((m, k1), "float32", "mesh[0]", "S[1]") = R.matmul(x, weight1)
            lv1: R.DTensor((m, k1), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
            lv3: R.DTensor((m, n), "float32", "mesh[0]", "R") = R.matmul(lv1, weight2)
            return lv3

    after = relax.distributed.transform.PropagateSharding()(MLPDynamicShape)
    assert_structural_equal(after, ShardedMLPDynamicShape)


def test_mlp_pipeline_parallelism():
    @I.ir_module
    class PipelineMLP:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                    R.device_mesh((2,), I.Range(4, 6)),  # mesh[1]
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((128, 128), "float32"),
            weight1: R.Tensor((128, 128), "float32"),
            weight2: R.Tensor((128, 128), "float32"),
            weight3: R.Tensor((128, 128), "float32"),
            weight4: R.Tensor((128, 128), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            lv0 = R.matmul(x, weight1)
            lv1 = R.nn.gelu(lv0)
            lv2 = R.dist.annotate_sharding(lv1, device_mesh="mesh[0]", placement="S[1]")
            lv3 = R.matmul(lv2, weight2)
            lv4 = R.dist.annotate_sharding(lv3, device_mesh="mesh[1]", placement="R")
            lv5 = R.matmul(lv4, weight3)
            lv6 = R.nn.gelu(lv5)
            lv7 = R.dist.annotate_sharding(lv6, device_mesh="mesh[1]", placement="S[1]")
            lv8 = R.matmul(lv7, weight4)
            return lv8

    # from tvm.script import ir as I
    # from tvm.script import relax as R

    @I.ir_module
    class ShardedPipelineMLP:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((2,), I.Range(4, 6))]}
        )

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
            weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
            weight3: R.DTensor((128, 128), "float32", "mesh[1]", "S[1]"),
            weight4: R.DTensor((128, 128), "float32", "mesh[1]", "S[0]"),
        ) -> R.DTensor((128, 128), "float32", "mesh[1]", "R"):
            lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.matmul(x, weight1)
            lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
            lv3: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.matmul(lv1, weight2)
            lv4: R.DTensor((128, 128), "float32", "mesh[1]", "R") = R.dist.redistribute(
                lv3, device_mesh="mesh[1]", placement="R"
            )
            lv5: R.DTensor((128, 128), "float32", "mesh[1]", "S[1]") = R.matmul(lv4, weight3)
            lv6: R.DTensor((128, 128), "float32", "mesh[1]", "S[1]") = R.nn.gelu(lv5)
            lv8: R.DTensor((128, 128), "float32", "mesh[1]", "R") = R.matmul(lv6, weight4)
            return lv8

    after = relax.distributed.transform.PropagateSharding()(PipelineMLP)
    assert_structural_equal(after, ShardedPipelineMLP)


def test_decoder_layer():
    @I.ir_module
    class LlamaAttentionLayer:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                    R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @Ts.function
        def rms_norm(
            A: T.Tensor((T.int64(1), 256, T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(4096),), "float16"),
            rms_norm_1: T.Tensor((T.int64(1), 256, T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            Ared_temp = Ts.sblock_alloc_tensor((T.int64(1), 256))
            for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
                with Ts.sblock("Ared_temp"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSR", [bsz, i, k])
                    Ts.reads(A[v_bsz, v_i, v_k])
                    Ts.writes(Ared_temp[v_bsz, v_i])
                    with Ts.init():
                        Ared_temp[v_bsz, v_i] = T.float32(0)
                    Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                        "float32", A[v_bsz, v_i, v_k]
                    ) * T.Cast("float32", A[v_bsz, v_i, v_k])
            for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
                with Ts.sblock("rms_norm"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSS", [bsz, i, k])
                    Ts.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                    Ts.writes(rms_norm_1[v_bsz, v_i, v_k])
                    rms_norm_1[v_bsz, v_i, v_k] = T.Cast(
                        "float16",
                        T.Cast("float32", B[v_k])
                        * (
                            T.Cast("float32", A[v_bsz, v_i, v_k])
                            / T.sqrt(
                                Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                                + T.float32(9.9999999999999995e-07)
                            )
                        ),
                    )

        @Ts.function
        def rotary_embedding(
            A: T.Tensor((T.int64(1), 256, T.int64(32), T.int64(128)), "float16"),
            B: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            C: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            rotary: T.Tensor((T.int64(1), 256, T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for i0, i1, i2, i3 in T.grid(T.int64(1), 256, T.int64(32), T.int64(128)):
                with Ts.sblock("rotary"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(
                        B[256 + v_i1 - 256, v_i3],
                        A[v_i0, v_i1, v_i2, v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129)],
                        C[256 + v_i1 - 256, v_i3],
                    )
                    Ts.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                    rotary[v_i0, v_i1, v_i2, v_i3] = B[256 + v_i1 - 256, v_i3] * A[
                        v_i0, v_i1, v_i2, v_i3
                    ] + C[256 + v_i1 - 256, v_i3] * T.Select(
                        T.int64(64) <= v_i3,
                        A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                        A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                    )

        @R.function(pure=False)
        def foo(
            input_tokens: R.Tensor((1, 256, 4096), dtype="float16"),
            mask: R.Tensor((1, 1, 256, 256), dtype="float16"),
            div_const: R.Tensor((1, 32, 256, 256), dtype="float16"),
            maximum_const: R.Tensor((1, 32, 256, 256), dtype="float16"),
            kv_cache: R.Tuple(R.Any, R.Any),
            linear_weight: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight1: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight2: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight3: R.Tensor((4096, 4096), dtype="float16"),
            rms_norm_weight: R.Tensor((4096,), dtype="float16"),
            cos_cached: R.Tensor((2048, 128), dtype="float16"),
            sin_cached: R.Tensor((2048, 128), dtype="float16"),
        ):
            cls = LlamaAttentionLayer
            lv6 = R.call_tir(
                cls.rms_norm,
                (input_tokens, rms_norm_weight),
                ty_args=[R.Tensor((1, 256, 4096), dtype="float16")],
            )
            lv7: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight, axes=None)
            lv7_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv7, "mesh[0]", "S[1]"
            )
            lv8: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv7_copy)
            lv9: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
                lv8, R.shape([1, 256, 32, 128])
            )
            lv10: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(
                linear_weight1, axes=None
            )
            lv10_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv10, "mesh[0]", "S[1]"
            )
            lv11: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv10_copy)
            lv12: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
                lv11, R.shape([1, 256, 32, 128])
            )
            lv13: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(
                linear_weight2, axes=None
            )
            lv13_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv13, "mesh[0]", "S[1]"
            )
            lv14: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv13_copy)
            lv15: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
                lv14, R.shape([1, 256, 32, 128])
            )
            lv16 = R.call_tir(
                cls.rotary_embedding,
                (lv9, cos_cached, sin_cached),
                ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")],
            )
            lv17 = R.call_tir(
                cls.rotary_embedding,
                (lv12, cos_cached, sin_cached),
                ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")],
            )
            lv18: R.Tensor((256, 32, 128), dtype="float16") = R.reshape(
                lv17, R.shape([256, 32, 128])
            )
            lv19: R.Tensor((256, 32, 128), dtype="float16") = R.reshape(
                lv15, R.shape([256, 32, 128])
            )
            lv20: R.Any = kv_cache[0]
            lv21: R.Any = R.call_packed(
                "vm.builtin.attention_kv_cache_append", lv20, lv18, ty_args=(R.Any,)
            )
            lv22: R.Any = kv_cache[1]
            lv23: R.Any = R.call_packed(
                "vm.builtin.attention_kv_cache_append", lv22, lv19, ty_args=(R.Any,)
            )
            lv24: R.Tensor((256, 32, 128), dtype="float16") = R.call_packed(
                "vm.builtin.attention_kv_cache_view",
                lv21,
                R.shape([256, 32, 128]),
                ty_args=(R.Tensor((256, 32, 128), dtype="float16"),),
            )
            lv25: R.Tensor((256, 32, 128), dtype="float16") = R.call_packed(
                "vm.builtin.attention_kv_cache_view",
                lv23,
                R.shape([256, 32, 128]),
                ty_args=(R.Tensor((256, 32, 128), dtype="float16"),),
            )
            lv26: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
                lv24, R.shape([1, 256, 32, 128])
            )
            lv27: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
                lv25, R.shape([1, 256, 32, 128])
            )
            lv28: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(
                lv16, axes=[0, 2, 1, 3]
            )
            lv29: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(
                lv26, axes=[0, 2, 1, 3]
            )
            lv30: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(
                lv27, axes=[0, 2, 1, 3]
            )
            lv31: R.Tensor((1, 32, 128, 256), dtype="float16") = R.permute_dims(
                lv29, axes=[0, 1, 3, 2]
            )
            lv32: R.Tensor((1, 32, 256, 256), dtype="float16") = R.matmul(lv28, lv31)
            lv33: R.Tensor((1, 32, 256, 256), dtype="float16") = R.divide(lv32, div_const)
            lv34: R.Tensor((1, 32, 256, 256), dtype="float16") = R.maximum(lv33, maximum_const)
            lv35: R.Tensor((1, 32, 256, 256), dtype="float16") = R.minimum(lv34, mask)
            # lv36: R.Tensor((1, 32, 256, 256), dtype="float32") = R.astype(lv35, dtype="float32")
            lv37: R.Tensor((1, 32, 256, 256), dtype="float16") = R.nn.softmax(lv35, axis=-1)
            # lv38: R.Tensor((1, 32, 256, 256), dtype="float16") = R.astype(lv37, dtype="float16")
            lv39: R.Tensor((1, 32, 256, 128), dtype="float16") = R.matmul(lv37, lv30)
            lv40: R.Tensor((1, 256, 32, 128), dtype="float16") = R.permute_dims(
                lv39, axes=[0, 2, 1, 3]
            )
            lv41: R.Tensor((1, 256, 4096), dtype="float16") = R.reshape(
                lv40, R.shape([1, 256, 4096])
            )
            lv42: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(
                linear_weight3, axes=None
            )
            lv43: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv41, lv42)
            lv44: R.Tensor((1, 256, 4096), dtype="float16") = R.add(input_tokens, lv43)
            gv = lv44

            return gv

    @I.ir_module
    class ShardedLlamaAttentionLayer:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @Ts.function
        def rms_norm(
            A: T.Tensor((T.int64(1), 256, T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(4096),), "float16"),
            rms_norm_1: T.Tensor((T.int64(1), 256, T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            Ared_temp = Ts.sblock_alloc_tensor((T.int64(1), 256))
            for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
                with Ts.sblock("Ared_temp"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSR", [bsz, i, k])
                    Ts.reads(A[v_bsz, v_i, v_k])
                    Ts.writes(Ared_temp[v_bsz, v_i])
                    with Ts.init():
                        Ared_temp[v_bsz, v_i] = T.float32(0)
                    Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                        "float32", A[v_bsz, v_i, v_k]
                    ) * T.Cast("float32", A[v_bsz, v_i, v_k])
            for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
                with Ts.sblock("rms_norm"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSS", [bsz, i, k])
                    Ts.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                    Ts.writes(rms_norm_1[v_bsz, v_i, v_k])
                    rms_norm_1[v_bsz, v_i, v_k] = T.Cast(
                        "float16",
                        T.Cast("float32", B[v_k])
                        * (
                            T.Cast("float32", A[v_bsz, v_i, v_k])
                            / T.sqrt(
                                Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                                + T.float32(9.9999999999999995e-07)
                            )
                        ),
                    )

        @Ts.function
        def rotary_embedding(
            A: T.Tensor((T.int64(1), 256, T.int64(32), T.int64(128)), "float16"),
            B: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            C: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            rotary: T.Tensor((T.int64(1), 256, T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3 in T.grid(T.int64(1), 256, T.int64(32), T.int64(128)):
                with Ts.sblock("rotary"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(
                        B[256 + v_i1 - 256, v_i3],
                        A[v_i0, v_i1, v_i2, v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129)],
                        C[256 + v_i1 - 256, v_i3],
                    )
                    Ts.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                    rotary[v_i0, v_i1, v_i2, v_i3] = B[256 + v_i1 - 256, v_i3] * A[
                        v_i0, v_i1, v_i2, v_i3
                    ] + C[256 + v_i1 - 256, v_i3] * T.Select(
                        T.int64(64) <= v_i3,
                        A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                        A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                    )

        @R.function(pure=False)
        def foo(
            input_tokens: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R"),
            mask: R.DTensor((1, 1, 256, 256), "float16", "mesh[0]", "R"),
            div_const: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]"),
            maximum_const: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]"),
            kv_cache: R.Tuple(R.Any, R.Any),
            linear_weight: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight1: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight2: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight3: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]"),
            rms_norm_weight: R.DTensor((4096,), "float16", "mesh[0]", "R"),
            cos_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
            sin_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
        ) -> R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R"):
            cls = ShardedLlamaAttentionLayer
            lv6 = R.dist.call_tir(
                cls.rms_norm,
                (input_tokens, rms_norm_weight),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            lv7: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                linear_weight, axes=None
            )
            lv8: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(lv6, lv7)
            lv9: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv8, R.shape([1, 256, 32, 128])
            )
            lv10: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                linear_weight1, axes=None
            )
            lv11: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(lv6, lv10)
            lv12: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv11, R.shape([1, 256, 32, 128])
            )
            lv13: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                linear_weight2, axes=None
            )
            lv14: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(lv6, lv13)
            lv15: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv14, R.shape([1, 256, 32, 128])
            )
            lv16 = R.dist.call_tir(
                cls.rotary_embedding,
                (lv9, cos_cached, sin_cached),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv17 = R.dist.call_tir(
                cls.rotary_embedding,
                (lv12, cos_cached, sin_cached),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv18: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.reshape(
                lv17, R.shape([256, 32, 128])
            )
            lv19: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.reshape(
                lv15, R.shape([256, 32, 128])
            )
            lv20: R.Any = kv_cache[0]
            lv21: R.Any = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_append",
                lv20,
                lv18,
                ty_args=(R.Any,),
            )
            lv22: R.Any = kv_cache[1]
            lv23: R.Any = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_append",
                lv22,
                lv19,
                ty_args=(R.Any,),
            )
            lv24: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_view",
                lv21,
                R.shape([256, 32, 128]),
                ty_args=(R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]"),),
            )
            lv25: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_view",
                lv23,
                R.shape([256, 32, 128]),
                ty_args=(R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]"),),
            )
            lv26: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv24, R.shape([1, 256, 32, 128])
            )
            lv27: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv25, R.shape([1, 256, 32, 128])
            )
            lv28: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                lv16, axes=[0, 2, 1, 3]
            )
            lv29: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                lv26, axes=[0, 2, 1, 3]
            )
            lv30: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                lv27, axes=[0, 2, 1, 3]
            )
            lv31: R.DTensor((1, 32, 128, 256), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                lv29, axes=[0, 1, 3, 2]
            )
            lv32: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.matmul(lv28, lv31)
            lv33: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.divide(
                lv32, div_const
            )
            lv34: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.maximum(
                lv33, maximum_const
            )
            lv35: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.minimum(lv34, mask)
            lv37: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.nn.softmax(
                lv35, axis=-1
            )
            lv39: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = R.matmul(lv37, lv30)
            lv40: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.permute_dims(
                lv39, axes=[0, 2, 1, 3]
            )
            lv41: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv40, R.shape([1, 256, 4096])
            )
            lv42: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]") = R.permute_dims(
                linear_weight3, axes=None
            )
            lv43: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = R.matmul(lv41, lv42)
            lv44: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = R.add(input_tokens, lv43)
            gv: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = lv44
            return gv

    after = relax.distributed.transform.PropagateSharding()(LlamaAttentionLayer)
    assert_structural_equal(after, ShardedLlamaAttentionLayer)


# PropagateSharding should analyze TIR funtions
# and successfully propagate sharding annotations through them
def test_decoder_layer_tir():
    @I.ir_module
    class LlamaAttentionLayerTIR:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @Ts.function(private=True)
        def add(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
            T_add: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(256), T.int64(4096)):
                with Ts.sblock("T_add"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2 = Ts.axis.remap("SS", [ax1, ax2])
                    Ts.reads(A[T.int64(0), v_ax1, v_ax2], B[T.int64(0), v_ax1, v_ax2])
                    Ts.writes(T_add[T.int64(0), v_ax1, v_ax2])
                    T_add[T.int64(0), v_ax1, v_ax2] = (
                        A[T.int64(0), v_ax1, v_ax2] + B[T.int64(0), v_ax1, v_ax2]
                    )

        @Ts.function(private=True)
        def divide(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
            T_divide: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_divide"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    Ts.reads(A[T.int64(0), v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_divide[T.int64(0), v_ax1, v_ax2, v_ax3])
                    T_divide[T.int64(0), v_ax1, v_ax2, v_ax3] = (
                        A[T.int64(0), v_ax1, v_ax2, v_ax3] / B[T.int64(0), v_ax1, v_ax2, v_ax3]
                    )

        @Ts.function(private=True)
        def matmul(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(4096), T.int64(4096)), "float16"),
            matmul: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(256), T.int64(4096), T.int64(4096)):
                with Ts.sblock("matmul"):
                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i1, v_i2, v_k = Ts.axis.remap("SSR", [i1, i2, k])
                    Ts.reads(A[T.int64(0), v_i1, v_k], B[v_k, v_i2])
                    Ts.writes(matmul[T.int64(0), v_i1, v_i2])
                    with Ts.init():
                        matmul[T.int64(0), v_i1, v_i2] = T.float16(0)
                    matmul[T.int64(0), v_i1, v_i2] = (
                        matmul[T.int64(0), v_i1, v_i2] + A[T.int64(0), v_i1, v_k] * B[v_k, v_i2]
                    )

        @Ts.function(private=True)
        def matmul1(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(128)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(32), T.int64(128), T.int64(256)), "float16"),
            matmul: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3, k in T.grid(
                T.int64(1), T.int64(32), T.int64(256), T.int64(256), T.int64(128)
            ):
                with Ts.sblock("matmul"):
                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSR", [i1, i2, i3, k])
                    Ts.reads(A[T.int64(0), v_i1, v_i2, v_k], B[T.int64(0), v_i1, v_k, v_i3])
                    Ts.writes(matmul[T.int64(0), v_i1, v_i2, v_i3])
                    with Ts.init():
                        matmul[T.int64(0), v_i1, v_i2, v_i3] = T.float16(0)
                    matmul[T.int64(0), v_i1, v_i2, v_i3] = (
                        matmul[T.int64(0), v_i1, v_i2, v_i3]
                        + A[T.int64(0), v_i1, v_i2, v_k] * B[T.int64(0), v_i1, v_k, v_i3]
                    )

        @Ts.function(private=True)
        def matmul2(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(128)), "float16"),
            matmul: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3, k in T.grid(
                T.int64(1), T.int64(32), T.int64(256), T.int64(128), T.int64(256)
            ):
                with Ts.sblock("matmul"):
                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSR", [i1, i2, i3, k])
                    Ts.reads(A[T.int64(0), v_i1, v_i2, v_k], B[T.int64(0), v_i1, v_k, v_i3])
                    Ts.writes(matmul[T.int64(0), v_i1, v_i2, v_i3])
                    with Ts.init():
                        matmul[T.int64(0), v_i1, v_i2, v_i3] = T.float16(0)
                    matmul[T.int64(0), v_i1, v_i2, v_i3] = (
                        matmul[T.int64(0), v_i1, v_i2, v_i3]
                        + A[T.int64(0), v_i1, v_i2, v_k] * B[T.int64(0), v_i1, v_k, v_i3]
                    )

        @Ts.function(private=True)
        def maximum(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
            T_maximum: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_maximum"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    Ts.reads(A[T.int64(0), v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_maximum[T.int64(0), v_ax1, v_ax2, v_ax3])
                    T_maximum[T.int64(0), v_ax1, v_ax2, v_ax3] = T.max(
                        A[T.int64(0), v_ax1, v_ax2, v_ax3], B[T.int64(0), v_ax1, v_ax2, v_ax3]
                    )

        @Ts.function(private=True)
        def minimum(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(1), T.int64(256), T.int64(256)), "float16"),
            T_minimum: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_minimum"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    Ts.reads(
                        A[T.int64(0), v_ax1, v_ax2, v_ax3], B[T.int64(0), T.int64(0), v_ax2, v_ax3]
                    )
                    Ts.writes(T_minimum[T.int64(0), v_ax1, v_ax2, v_ax3])
                    T_minimum[T.int64(0), v_ax1, v_ax2, v_ax3] = T.min(
                        A[T.int64(0), v_ax1, v_ax2, v_ax3], B[T.int64(0), T.int64(0), v_ax2, v_ax3]
                    )

        @Ts.function(private=True)
        def reshape(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
            T_reshape: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(32), T.int64(128)):
                with Ts.sblock("T_reshape"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    Ts.reads(A[T.int64(0), v_ax1, v_ax2 * T.int64(128) + v_ax3])
                    Ts.writes(T_reshape[T.int64(0), v_ax1, v_ax2, v_ax3])
                    T_reshape[T.int64(0), v_ax1, v_ax2, v_ax3] = A[
                        T.int64(0), v_ax1, v_ax2 * T.int64(128) + v_ax3
                    ]

        @Ts.function(private=True)
        def reshape1(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
            T_reshape: T.Tensor((T.int64(256), T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(32), T.int64(128)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]

        @Ts.function(private=True)
        def reshape2(
            A: T.Tensor((T.int64(256), T.int64(32), T.int64(128)), "float16"),
            T_reshape: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(32), T.int64(128)):
                with Ts.sblock("T_reshape"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    Ts.reads(A[v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_reshape[T.int64(0), v_ax1, v_ax2, v_ax3])
                    T_reshape[T.int64(0), v_ax1, v_ax2, v_ax3] = A[v_ax1, v_ax2, v_ax3]

        @Ts.function(private=True)
        def reshape3(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
            T_reshape: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(256), T.int64(4096)):
                with Ts.sblock("T_reshape"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2 = Ts.axis.remap("SS", [ax1, ax2])
                    Ts.reads(A[T.int64(0), v_ax1, v_ax2 // T.int64(128), v_ax2 % T.int64(128)])
                    Ts.writes(T_reshape[T.int64(0), v_ax1, v_ax2])
                    T_reshape[T.int64(0), v_ax1, v_ax2] = A[
                        T.int64(0), v_ax1, v_ax2 // T.int64(128), v_ax2 % T.int64(128)
                    ]

        @Ts.function
        def rms_norm(
            A: T.Tensor((T.int64(1), 256, T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(4096),), "float16"),
            rms_norm_1: T.Tensor((T.int64(1), 256, T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            Ared_temp = Ts.sblock_alloc_tensor((T.int64(1), 256))
            for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
                with Ts.sblock("Ared_temp"):
                    v_bsz = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i, v_k = Ts.axis.remap("SR", [i, k])
                    Ts.reads(A[T.int64(0), v_i, v_k])
                    Ts.writes(Ared_temp[T.int64(0), v_i])
                    with Ts.init():
                        Ared_temp[T.int64(0), v_i] = T.float32(0)
                    Ared_temp[T.int64(0), v_i] = Ared_temp[T.int64(0), v_i] + T.Cast(
                        "float32", A[T.int64(0), v_i, v_k]
                    ) * T.Cast("float32", A[T.int64(0), v_i, v_k])
            for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
                with Ts.sblock("rms_norm"):
                    v_bsz = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i, v_k = Ts.axis.remap("SS", [i, k])
                    Ts.reads(B[v_k], A[T.int64(0), v_i, v_k], Ared_temp[T.int64(0), v_i])
                    Ts.writes(rms_norm_1[T.int64(0), v_i, v_k])
                    rms_norm_1[T.int64(0), v_i, v_k] = T.Cast(
                        "float16",
                        T.Cast("float32", B[v_k])
                        * (
                            T.Cast("float32", A[T.int64(0), v_i, v_k])
                            / T.sqrt(
                                Ared_temp[T.int64(0), v_i] * T.float32(0.000244140625)
                                + T.float32(9.9999999999999995e-07)
                            )
                        ),
                    )

        @Ts.function
        def rotary_embedding(
            A: T.Tensor((T.int64(1), 256, T.int64(32), T.int64(128)), "float16"),
            B: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            C: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            rotary: T.Tensor((T.int64(1), 256, T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3 in T.grid(T.int64(1), 256, T.int64(32), T.int64(128)):
                with Ts.sblock("rotary"):
                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i1, v_i2, v_i3 = Ts.axis.remap("SSS", [i1, i2, i3])
                    Ts.reads(
                        B[v_i1, v_i3],
                        A[
                            T.int64(0),
                            v_i1,
                            v_i2,
                            v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129),
                        ],
                        C[v_i1, v_i3],
                    )
                    Ts.writes(rotary[T.int64(0), v_i1, v_i2, v_i3])
                    rotary[T.int64(0), v_i1, v_i2, v_i3] = B[v_i1, v_i3] * A[
                        T.int64(0), v_i1, v_i2, v_i3
                    ] + C[v_i1, v_i3] * T.Select(
                        T.int64(64) <= v_i3,
                        A[T.int64(0), v_i1, v_i2, v_i3 - T.int64(64)],
                        A[T.int64(0), v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                    )

        @Ts.function(private=True)
        def softmax(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"),
            T_softmax_norm: T.Tensor(
                (T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"
            ),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_softmax_maxelem = Ts.sblock_alloc_tensor(
                (T.int64(1), T.int64(32), T.int64(256)), "float16"
            )
            T_softmax_exp = Ts.sblock_alloc_tensor(
                (T.int64(1), T.int64(32), T.int64(256), T.int64(256)), "float16"
            )
            T_softmax_expsum = Ts.sblock_alloc_tensor(
                (T.int64(1), T.int64(32), T.int64(256)), "float16"
            )
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_maxelem"):
                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i1, v_i2, v_k = Ts.axis.remap("SSR", [i1, i2, k])
                    Ts.reads(A[T.int64(0), v_i1, v_i2, v_k])
                    Ts.writes(T_softmax_maxelem[T.int64(0), v_i1, v_i2])
                    with Ts.init():
                        T_softmax_maxelem[T.int64(0), v_i1, v_i2] = T.float16(-65504)
                    T_softmax_maxelem[T.int64(0), v_i1, v_i2] = T.max(
                        T_softmax_maxelem[T.int64(0), v_i1, v_i2], A[T.int64(0), v_i1, v_i2, v_k]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_exp"):
                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i1, v_i2, v_i3 = Ts.axis.remap("SSS", [i1, i2, i3])
                    Ts.reads(
                        A[T.int64(0), v_i1, v_i2, v_i3], T_softmax_maxelem[T.int64(0), v_i1, v_i2]
                    )
                    Ts.writes(T_softmax_exp[T.int64(0), v_i1, v_i2, v_i3])
                    T_softmax_exp[T.int64(0), v_i1, v_i2, v_i3] = T.exp(
                        A[T.int64(0), v_i1, v_i2, v_i3] - T_softmax_maxelem[T.int64(0), v_i1, v_i2]
                    )
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_expsum"):
                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i1, v_i2, v_k = Ts.axis.remap("SSR", [i1, i2, k])
                    Ts.reads(T_softmax_exp[T.int64(0), v_i1, v_i2, v_k])
                    Ts.writes(T_softmax_expsum[T.int64(0), v_i1, v_i2])
                    with Ts.init():
                        T_softmax_expsum[T.int64(0), v_i1, v_i2] = T.float16(0)
                    T_softmax_expsum[T.int64(0), v_i1, v_i2] = (
                        T_softmax_expsum[T.int64(0), v_i1, v_i2]
                        + T_softmax_exp[T.int64(0), v_i1, v_i2, v_k]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_norm"):
                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_i1, v_i2, v_i3 = Ts.axis.remap("SSS", [i1, i2, i3])
                    Ts.reads(
                        T_softmax_exp[T.int64(0), v_i1, v_i2, v_i3],
                        T_softmax_expsum[T.int64(0), v_i1, v_i2],
                    )
                    Ts.writes(T_softmax_norm[T.int64(0), v_i1, v_i2, v_i3])
                    Ts.sblock_attr({"axis": 3})
                    T_softmax_norm[T.int64(0), v_i1, v_i2, v_i3] = (
                        T_softmax_exp[T.int64(0), v_i1, v_i2, v_i3]
                        / T_softmax_expsum[T.int64(0), v_i1, v_i2]
                    )

        @Ts.function(private=True)
        def transpose(
            A: T.Tensor((T.int64(4096), T.int64(4096)), "float16"),
            T_transpose: T.Tensor((T.int64(4096), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax1, v_ax0])
                    Ts.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = A[v_ax1, v_ax0]

        @Ts.function(private=True)
        def transpose1(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
            T_transpose: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(128)):
                with Ts.sblock("T_transpose"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    Ts.reads(A[T.int64(0), v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose[T.int64(0), v_ax1, v_ax2, v_ax3])
                    T_transpose[T.int64(0), v_ax1, v_ax2, v_ax3] = A[
                        T.int64(0), v_ax2, v_ax1, v_ax3
                    ]

        @Ts.function(private=True)
        def transpose2(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(128)), "float16"),
            T_transpose: T.Tensor((T.int64(1), T.int64(32), T.int64(128), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(128), T.int64(256)):
                with Ts.sblock("T_transpose"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    Ts.reads(A[T.int64(0), v_ax1, v_ax3, v_ax2])
                    Ts.writes(T_transpose[T.int64(0), v_ax1, v_ax2, v_ax3])
                    T_transpose[T.int64(0), v_ax1, v_ax2, v_ax3] = A[
                        T.int64(0), v_ax1, v_ax3, v_ax2
                    ]

        @Ts.function(private=True)
        def transpose3(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(128)), "float16"),
            T_transpose: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(32), T.int64(128)):
                with Ts.sblock("T_transpose"):
                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                    v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    Ts.reads(A[T.int64(0), v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose[T.int64(0), v_ax1, v_ax2, v_ax3])
                    T_transpose[T.int64(0), v_ax1, v_ax2, v_ax3] = A[
                        T.int64(0), v_ax2, v_ax1, v_ax3
                    ]

        @R.function(pure=False)
        def foo(
            input_tokens: R.Tensor((1, 256, 4096), dtype="float16"),
            mask: R.Tensor((1, 1, 256, 256), dtype="float16"),
            div_const: R.Tensor((1, 32, 256, 256), dtype="float16"),
            maximum_const: R.Tensor((1, 32, 256, 256), dtype="float16"),
            kv_cache: R.Tuple(R.Any, R.Any),
            linear_weight: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight1: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight2: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight3: R.Tensor((4096, 4096), dtype="float16"),
            rms_norm_weight: R.Tensor((4096,), dtype="float16"),
            cos_cached: R.Tensor((2048, 128), dtype="float16"),
            sin_cached: R.Tensor((2048, 128), dtype="float16"),
        ) -> R.Tensor((1, 256, 4096), dtype="float16"):
            cls = LlamaAttentionLayerTIR
            lv6 = R.call_tir(
                cls.rms_norm,
                (input_tokens, rms_norm_weight),
                ty_args=[R.Tensor((1, 256, 4096), dtype="float16")],
            )
            lv7 = R.call_tir(
                cls.transpose, (linear_weight,), ty_args=[R.Tensor((4096, 4096), dtype="float16")]
            )
            lv7_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv7, device_mesh="mesh[0]", placement="S[1]"
            )
            lv8 = R.call_tir(
                cls.matmul, (lv6, lv7_copy), ty_args=[R.Tensor((1, 256, 4096), dtype="float16")]
            )
            lv9 = R.call_tir(
                cls.reshape, (lv8,), ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")]
            )
            lv10 = R.call_tir(
                cls.transpose, (linear_weight1,), ty_args=[R.Tensor((4096, 4096), dtype="float16")]
            )
            lv10_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv10, device_mesh="mesh[0]", placement="S[1]"
            )
            lv11 = R.call_tir(
                cls.matmul, (lv6, lv10_copy), ty_args=[R.Tensor((1, 256, 4096), dtype="float16")]
            )
            lv12 = R.call_tir(
                cls.reshape, (lv11,), ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")]
            )
            lv13 = R.call_tir(
                cls.transpose, (linear_weight2,), ty_args=[R.Tensor((4096, 4096), dtype="float16")]
            )
            lv13_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv13, device_mesh="mesh[0]", placement="S[1]"
            )
            lv14 = R.call_tir(
                cls.matmul, (lv6, lv13_copy), ty_args=[R.Tensor((1, 256, 4096), dtype="float16")]
            )
            lv15 = R.call_tir(
                cls.reshape, (lv14,), ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")]
            )
            lv16 = R.call_tir(
                cls.rotary_embedding,
                (lv9, cos_cached, sin_cached),
                ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")],
            )
            lv17 = R.call_tir(
                cls.rotary_embedding,
                (lv12, cos_cached, sin_cached),
                ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")],
            )
            lv18 = R.call_tir(
                cls.reshape1, (lv17,), ty_args=[R.Tensor((256, 32, 128), dtype="float16")]
            )
            lv19 = R.call_tir(
                cls.reshape1, (lv15,), ty_args=[R.Tensor((256, 32, 128), dtype="float16")]
            )
            lv20: R.Any = kv_cache[0]
            lv21: R.Any = R.call_packed(
                "vm.builtin.attention_kv_cache_append", lv20, lv18, ty_args=(R.Any,)
            )
            lv22: R.Any = kv_cache[1]
            lv23: R.Any = R.call_packed(
                "vm.builtin.attention_kv_cache_append", lv22, lv19, ty_args=(R.Any,)
            )
            lv24: R.Tensor((256, 32, 128), dtype="float16") = R.call_packed(
                "vm.builtin.attention_kv_cache_view",
                lv21,
                R.shape([256, 32, 128]),
                ty_args=(R.Tensor((256, 32, 128), dtype="float16"),),
            )
            lv25: R.Tensor((256, 32, 128), dtype="float16") = R.call_packed(
                "vm.builtin.attention_kv_cache_view",
                lv23,
                R.shape([256, 32, 128]),
                ty_args=(R.Tensor((256, 32, 128), dtype="float16"),),
            )
            lv26 = R.call_tir(
                cls.reshape2, (lv24,), ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")]
            )
            lv27 = R.call_tir(
                cls.reshape2, (lv25,), ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")]
            )
            lv28 = R.call_tir(
                cls.transpose1, (lv16,), ty_args=[R.Tensor((1, 32, 256, 128), dtype="float16")]
            )
            lv29 = R.call_tir(
                cls.transpose1, (lv26,), ty_args=[R.Tensor((1, 32, 256, 128), dtype="float16")]
            )
            lv30 = R.call_tir(
                cls.transpose1, (lv27,), ty_args=[R.Tensor((1, 32, 256, 128), dtype="float16")]
            )
            lv31 = R.call_tir(
                cls.transpose2, (lv29,), ty_args=[R.Tensor((1, 32, 128, 256), dtype="float16")]
            )
            lv32 = R.call_tir(
                cls.matmul1, (lv28, lv31), ty_args=[R.Tensor((1, 32, 256, 256), dtype="float16")]
            )
            lv33 = R.call_tir(
                cls.divide,
                (lv32, div_const),
                ty_args=[R.Tensor((1, 32, 256, 256), dtype="float16")],
            )
            lv34 = R.call_tir(
                cls.maximum,
                (lv33, maximum_const),
                ty_args=[R.Tensor((1, 32, 256, 256), dtype="float16")],
            )
            lv35 = R.call_tir(
                cls.minimum, (lv34, mask), ty_args=[R.Tensor((1, 32, 256, 256), dtype="float16")]
            )
            lv37 = R.call_tir(
                cls.softmax, (lv35,), ty_args=[R.Tensor((1, 32, 256, 256), dtype="float16")]
            )
            lv39 = R.call_tir(
                cls.matmul2, (lv37, lv30), ty_args=[R.Tensor((1, 32, 256, 128), dtype="float16")]
            )
            lv40 = R.call_tir(
                cls.transpose3, (lv39,), ty_args=[R.Tensor((1, 256, 32, 128), dtype="float16")]
            )
            lv41 = R.call_tir(
                cls.reshape3, (lv40,), ty_args=[R.Tensor((1, 256, 4096), dtype="float16")]
            )
            lv42 = R.call_tir(
                cls.transpose, (linear_weight3,), ty_args=[R.Tensor((4096, 4096), dtype="float16")]
            )
            lv43 = R.call_tir(
                cls.matmul, (lv41, lv42), ty_args=[R.Tensor((1, 256, 4096), dtype="float16")]
            )
            lv44 = R.call_tir(
                cls.add, (input_tokens, lv43), ty_args=[R.Tensor((1, 256, 4096), dtype="float16")]
            )
            gv: R.Tensor((1, 256, 4096), dtype="float16") = lv44
            return gv

    # the below uses global vars that are not yet defined but the definitions
    # will be added later
    @I.ir_module(check_well_formed=False)
    class ShardedLlamaAttentionLayerTIR:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @R.function(pure=False)
        def foo(
            input_tokens: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R"),
            mask: R.DTensor((1, 1, 256, 256), "float16", "mesh[0]", "R"),
            div_const: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]"),
            maximum_const: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]"),
            kv_cache: R.Tuple(R.Any, R.Any),
            linear_weight: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight1: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight2: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight3: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]"),
            rms_norm_weight: R.DTensor((4096,), "float16", "mesh[0]", "R"),
            cos_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
            sin_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
        ) -> R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R"):
            cls = ShardedLlamaAttentionLayerTIR
            lv6 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("rms_norm"),
                (input_tokens, rms_norm_weight),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            lv7 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose"),
                (linear_weight,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
            )
            lv8 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("matmul"),
                (lv6, lv7),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
            )
            lv9 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("reshape"),
                (lv8,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv10 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose"),
                (linear_weight1,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
            )
            lv11 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("matmul"),
                (lv6, lv10),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
            )
            lv12 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("reshape"),
                (lv11,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv13 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose"),
                (linear_weight2,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
            )
            lv14 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("matmul"),
                (lv6, lv13),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
            )
            lv15 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("reshape"),
                (lv14,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv16 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("rotary_embedding"),
                (lv9, cos_cached, sin_cached),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv17 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("rotary_embedding"),
                (lv12, cos_cached, sin_cached),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv18 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("reshape1"),
                (lv17,),
                ty_args=[R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv19 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("reshape1"),
                (lv15,),
                ty_args=[R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv20: R.Any = kv_cache[0]
            lv21: R.Any = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_append",
                lv20,
                lv18,
                ty_args=(R.Any,),
            )
            lv22: R.Any = kv_cache[1]
            lv23: R.Any = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_append",
                lv22,
                lv19,
                ty_args=(R.Any,),
            )
            lv24: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_view",
                lv21,
                R.shape([256, 32, 128]),
                ty_args=(R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]"),),
            )
            lv25: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_view",
                lv23,
                R.shape([256, 32, 128]),
                ty_args=(R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]"),),
            )
            lv26 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("reshape2"),
                (lv24,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv27 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("reshape2"),
                (lv25,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv28 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose1"),
                (lv16,),
                ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv29 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose1"),
                (lv26,),
                ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv30 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose1"),
                (lv27,),
                ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv31 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose2"),
                (lv29,),
                ty_args=[R.DTensor((1, 32, 128, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv32 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("matmul1"),
                (lv28, lv31),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv33 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("divide"),
                (lv32, div_const),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv34 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("maximum"),
                (lv33, maximum_const),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv35 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("minimum"),
                (lv34, mask),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv37 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("softmax"),
                (lv35,),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv39 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("matmul2"),
                (lv37, lv30),
                ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv40 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose3"),
                (lv39,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv41 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("reshape3"),
                (lv40,),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
            )
            lv42 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("transpose"),
                (linear_weight3,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]")],
            )
            lv43 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("matmul"),
                (lv41, lv42),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            lv44 = R.dist.call_tir(
                LlamaAttentionLayerTIR.get_global_var("add"),
                (input_tokens, lv43),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            gv: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = lv44
            return gv

    for gv, func in LlamaAttentionLayerTIR.functions_items():
        if gv.name_hint != "foo":
            ShardedLlamaAttentionLayerTIR[gv] = func

    after = relax.distributed.transform.PropagateSharding()(LlamaAttentionLayerTIR)
    assert_structural_equal(after, ShardedLlamaAttentionLayerTIR)


def test_decoder_layer_dynamic_shape():
    rms_norm_n = T.dynamic("n", "int64")
    rotary_embedding_n = T.dynamic("n", "int64")
    foo_n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class LlamaAttentionLayerDynamicShape:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                    R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @Ts.function
        def rms_norm(
            A: T.Tensor((T.int64(1), rms_norm_n, T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(4096),), "float16"),
            rms_norm_1: T.Tensor((T.int64(1), rms_norm_n, T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            Ared_temp = Ts.sblock_alloc_tensor((T.int64(1), rms_norm_n))
            for bsz, i, k in T.grid(T.int64(1), rms_norm_n, T.int64(4096)):
                with Ts.sblock("Ared_temp"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSR", [bsz, i, k])
                    Ts.reads(A[v_bsz, v_i, v_k])
                    Ts.writes(Ared_temp[v_bsz, v_i])
                    with Ts.init():
                        Ared_temp[v_bsz, v_i] = T.float32(0)
                    Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                        "float32", A[v_bsz, v_i, v_k]
                    ) * T.Cast("float32", A[v_bsz, v_i, v_k])
            for bsz, i, k in T.grid(T.int64(1), rms_norm_n, T.int64(4096)):
                with Ts.sblock("rms_norm"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSS", [bsz, i, k])
                    Ts.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                    Ts.writes(rms_norm_1[v_bsz, v_i, v_k])
                    rms_norm_1[v_bsz, v_i, v_k] = T.Cast(
                        "float16",
                        T.Cast("float32", B[v_k])
                        * (
                            T.Cast("float32", A[v_bsz, v_i, v_k])
                            / T.sqrt(
                                Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                                + T.float32(9.9999999999999995e-07)
                            )
                        ),
                    )

        @Ts.function
        def rotary_embedding(
            A: T.Tensor((T.int64(1), rotary_embedding_n, T.int64(32), T.int64(128)), "float16"),
            B: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            C: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            m: T.int64,
            rotary: T.Tensor(
                (T.int64(1), rotary_embedding_n, T.int64(32), T.int64(128)), "float16"
            ),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for i0, i1, i2, i3 in T.grid(T.int64(1), rotary_embedding_n, T.int64(32), T.int64(128)):
                with Ts.sblock("rotary"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(
                        B[m + v_i1 - rotary_embedding_n, v_i3],
                        A[v_i0, v_i1, v_i2, v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129)],
                        C[m + v_i1 - rotary_embedding_n, v_i3],
                    )
                    Ts.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                    rotary[v_i0, v_i1, v_i2, v_i3] = B[m + v_i1 - rotary_embedding_n, v_i3] * A[
                        v_i0, v_i1, v_i2, v_i3
                    ] + C[m + v_i1 - rotary_embedding_n, v_i3] * T.Select(
                        T.int64(64) <= v_i3,
                        A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                        A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                    )

        @R.function(pure=False)
        def foo(
            input_tokens: R.Tensor((1, foo_n, 4096), dtype="float16"),
            mask: R.Tensor((1, 1, foo_n, m), dtype="float16"),
            kv_cache: R.Tuple(R.Any, R.Any),
            linear_weight: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight1: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight2: R.Tensor((4096, 4096), dtype="float16"),
            linear_weight3: R.Tensor((4096, 4096), dtype="float16"),
            rms_norm_weight: R.Tensor((4096,), dtype="float16"),
            cos_cached: R.Tensor((2048, 128), dtype="float16"),
            sin_cached: R.Tensor((2048, 128), dtype="float16"),
        ):
            cls = LlamaAttentionLayerDynamicShape
            lv6 = R.call_tir(
                cls.rms_norm,
                (input_tokens, rms_norm_weight),
                ty_args=[R.Tensor((1, foo_n, 4096), dtype="float16")],
            )
            lv7: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight, axes=None)
            lv7_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv7, "mesh[0]", "S[1]"
            )
            lv8: R.Tensor((1, foo_n, 4096), dtype="float16") = R.matmul(lv6, lv7_copy)
            lv9: R.Tensor((1, foo_n, 32, 128), dtype="float16") = R.reshape(
                lv8, R.shape([1, foo_n, 32, 128])
            )
            lv10: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(
                linear_weight1, axes=None
            )
            lv10_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv10, "mesh[0]", "S[1]"
            )
            lv11: R.Tensor((1, foo_n, 4096), dtype="float16") = R.matmul(lv6, lv10_copy)
            lv12: R.Tensor((1, foo_n, 32, 128), dtype="float16") = R.reshape(
                lv11, R.shape([1, foo_n, 32, 128])
            )
            lv13: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(
                linear_weight2, axes=None
            )
            lv13_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
                lv13, "mesh[0]", "S[1]"
            )
            lv14: R.Tensor((1, foo_n, 4096), dtype="float16") = R.matmul(lv6, lv13_copy)
            lv15: R.Tensor((1, foo_n, 32, 128), dtype="float16") = R.reshape(
                lv14, R.shape([1, foo_n, 32, 128])
            )
            lv16 = R.call_tir(
                cls.rotary_embedding,
                (lv9, cos_cached, sin_cached, m),
                ty_args=[R.Tensor((1, foo_n, 32, 128), dtype="float16")],
            )
            lv17 = R.call_tir(
                cls.rotary_embedding,
                (lv12, cos_cached, sin_cached, m),
                ty_args=[R.Tensor((1, foo_n, 32, 128), dtype="float16")],
            )
            lv18: R.Tensor((foo_n, 32, 128), dtype="float16") = R.reshape(
                lv17, R.shape([foo_n, 32, 128])
            )
            lv19: R.Tensor((foo_n, 32, 128), dtype="float16") = R.reshape(
                lv15, R.shape([foo_n, 32, 128])
            )
            lv20: R.Any = kv_cache[0]
            lv21: R.Any = R.call_packed(
                "vm.builtin.attention_kv_cache_append", lv20, lv18, ty_args=(R.Any,)
            )
            lv22: R.Any = kv_cache[1]
            lv23: R.Any = R.call_packed(
                "vm.builtin.attention_kv_cache_append", lv22, lv19, ty_args=(R.Any,)
            )
            lv24: R.Tensor((m, 32, 128), dtype="float16") = R.call_packed(
                "vm.builtin.attention_kv_cache_view",
                lv21,
                R.shape([m, 32, 128]),
                ty_args=(R.Tensor((m, 32, 128), dtype="float16"),),
            )
            lv25: R.Tensor((m, 32, 128), dtype="float16") = R.call_packed(
                "vm.builtin.attention_kv_cache_view",
                lv23,
                R.shape([m, 32, 128]),
                ty_args=(R.Tensor((m, 32, 128), dtype="float16"),),
            )
            lv26: R.Tensor((1, m, 32, 128), dtype="float16") = R.reshape(
                lv24, R.shape([1, m, 32, 128])
            )
            lv27: R.Tensor((1, m, 32, 128), dtype="float16") = R.reshape(
                lv25, R.shape([1, m, 32, 128])
            )
            lv28: R.Tensor((1, 32, foo_n, 128), dtype="float16") = R.permute_dims(
                lv16, axes=[0, 2, 1, 3]
            )
            lv29: R.Tensor((1, 32, m, 128), dtype="float16") = R.permute_dims(
                lv26, axes=[0, 2, 1, 3]
            )
            lv30: R.Tensor((1, 32, m, 128), dtype="float16") = R.permute_dims(
                lv27, axes=[0, 2, 1, 3]
            )
            lv31: R.Tensor((1, 32, 128, m), dtype="float16") = R.permute_dims(
                lv29, axes=[0, 1, 3, 2]
            )
            lv32: R.Tensor((1, 32, foo_n, m), dtype="float16") = R.matmul(lv28, lv31)
            lv33: R.Tensor((1, 32, foo_n, m), dtype="float16") = R.divide(
                lv32, R.const(8, dtype="float16")
            )  # just choose some random value
            lv34: R.Tensor((1, 32, foo_n, m), dtype="float16") = R.maximum(
                lv33, R.const(1, dtype="float16")
            )  # just choose some random value
            lv35: R.Tensor((1, 32, foo_n, m), dtype="float16") = R.minimum(lv34, mask)
            # lv36: R.Tensor((1, 32, n, m), dtype="float32") = R.astype(lv35, dtype="float32")
            lv37: R.Tensor((1, 32, foo_n, m), dtype="float16") = R.nn.softmax(lv35, axis=-1)
            # lv38: R.Tensor((1, 32, n, m), dtype="float16") = R.astype(lv37, dtype="float16")
            lv39: R.Tensor((1, 32, foo_n, 128), dtype="float16") = R.matmul(lv37, lv30)
            lv40: R.Tensor((1, foo_n, 32, 128), dtype="float16") = R.permute_dims(
                lv39, axes=[0, 2, 1, 3]
            )
            lv41: R.Tensor((1, foo_n, 4096), dtype="float16") = R.reshape(
                lv40, R.shape([1, foo_n, 4096])
            )
            lv42: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(
                linear_weight3, axes=None
            )
            lv43: R.Tensor((1, foo_n, 4096), dtype="float16") = R.matmul(lv41, lv42)
            lv44: R.Tensor((1, foo_n, 4096), dtype="float16") = R.add(input_tokens, lv43)
            gv = lv44

            return gv

    rms_norm_n = T.dynamic("n", "int64")
    rotary_embedding_n = T.dynamic("n", "int64")
    foo_n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @I.ir_module
    class ShardedLlamaAttentionLayerDynamicShape:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @Ts.function
        def rms_norm(
            A: T.Tensor((T.int64(1), rms_norm_n, T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(4096),), "float16"),
            rms_norm_1: T.Tensor((T.int64(1), rms_norm_n, T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            Ared_temp = Ts.sblock_alloc_tensor((T.int64(1), rms_norm_n))
            for bsz, i, k in T.grid(T.int64(1), rms_norm_n, T.int64(4096)):
                with Ts.sblock("Ared_temp"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSR", [bsz, i, k])
                    Ts.reads(A[v_bsz, v_i, v_k])
                    Ts.writes(Ared_temp[v_bsz, v_i])
                    with Ts.init():
                        Ared_temp[v_bsz, v_i] = T.float32(0)
                    Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                        "float32", A[v_bsz, v_i, v_k]
                    ) * T.Cast("float32", A[v_bsz, v_i, v_k])
            for bsz, i, k in T.grid(T.int64(1), rms_norm_n, T.int64(4096)):
                with Ts.sblock("rms_norm"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSS", [bsz, i, k])
                    Ts.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                    Ts.writes(rms_norm_1[v_bsz, v_i, v_k])
                    rms_norm_1[v_bsz, v_i, v_k] = T.Cast(
                        "float16",
                        T.Cast("float32", B[v_k])
                        * (
                            T.Cast("float32", A[v_bsz, v_i, v_k])
                            / T.sqrt(
                                Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                                + T.float32(9.9999999999999995e-07)
                            )
                        ),
                    )

        @Ts.function
        def rotary_embedding(
            A: T.Tensor((T.int64(1), rotary_embedding_n, T.int64(32), T.int64(128)), "float16"),
            B: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            C: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            m: T.int64,
            rotary: T.Tensor(
                (T.int64(1), rotary_embedding_n, T.int64(32), T.int64(128)), "float16"
            ),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for i0, i1, i2, i3 in T.grid(T.int64(1), rotary_embedding_n, T.int64(32), T.int64(128)):
                with Ts.sblock("rotary"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(
                        B[m + v_i1 - rotary_embedding_n, v_i3],
                        A[v_i0, v_i1, v_i2, v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129)],
                        C[m + v_i1 - rotary_embedding_n, v_i3],
                    )
                    Ts.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                    rotary[v_i0, v_i1, v_i2, v_i3] = B[m + v_i1 - rotary_embedding_n, v_i3] * A[
                        v_i0, v_i1, v_i2, v_i3
                    ] + C[m + v_i1 - rotary_embedding_n, v_i3] * T.Select(
                        T.int64(64) <= v_i3,
                        A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                        A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                    )

        @R.function(pure=False)
        def foo(
            input_tokens: R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "R"),
            mask: R.DTensor((1, 1, foo_n, m), "float16", "mesh[0]", "R"),
            kv_cache: R.Tuple(R.Any, R.Any),
            linear_weight: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight1: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight2: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
            linear_weight3: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]"),
            rms_norm_weight: R.DTensor((4096,), "float16", "mesh[0]", "R"),
            cos_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
            sin_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
        ) -> R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "R"):
            cls = ShardedLlamaAttentionLayerDynamicShape
            lv6 = R.dist.call_tir(
                cls.rms_norm,
                (input_tokens, rms_norm_weight),
                ty_args=[R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "R")],
            )
            lv7: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                linear_weight, axes=None
            )
            lv8: R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(lv6, lv7)
            lv9: R.DTensor((1, foo_n, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv8, R.shape([1, foo_n, 32, 128])
            )
            lv10: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                linear_weight1, axes=None
            )
            lv11: R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(lv6, lv10)
            lv12: R.DTensor((1, foo_n, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv11, R.shape([1, foo_n, 32, 128])
            )
            lv13: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                linear_weight2, axes=None
            )
            lv14: R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(lv6, lv13)
            lv15: R.DTensor((1, foo_n, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv14, R.shape([1, foo_n, 32, 128])
            )
            lv16 = R.dist.call_tir(
                cls.rotary_embedding,
                (lv9, cos_cached, sin_cached, m),
                ty_args=[R.DTensor((1, foo_n, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv17 = R.dist.call_tir(
                cls.rotary_embedding,
                (lv12, cos_cached, sin_cached, m),
                ty_args=[R.DTensor((1, foo_n, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv18: R.DTensor((foo_n, 32, 128), "float16", "mesh[0]", "S[1]") = R.reshape(
                lv17, R.shape([foo_n, 32, 128])
            )
            lv19: R.DTensor((foo_n, 32, 128), "float16", "mesh[0]", "S[1]") = R.reshape(
                lv15, R.shape([foo_n, 32, 128])
            )
            lv20: R.Any = kv_cache[0]
            lv21: R.Any = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_append",
                lv20,
                lv18,
                ty_args=(R.Any,),
            )
            lv22: R.Any = kv_cache[1]
            lv23: R.Any = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_append",
                lv22,
                lv19,
                ty_args=(R.Any,),
            )
            lv24: R.DTensor((m, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_view",
                lv21,
                R.shape([m, 32, 128]),
                ty_args=(R.DTensor((m, 32, 128), "float16", "mesh[0]", "S[1]"),),
            )
            lv25: R.DTensor((m, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
                "vm.builtin.distributed.attention_kv_cache_view",
                lv23,
                R.shape([m, 32, 128]),
                ty_args=(R.DTensor((m, 32, 128), "float16", "mesh[0]", "S[1]"),),
            )
            lv26: R.DTensor((1, m, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv24, R.shape([1, m, 32, 128])
            )
            lv27: R.DTensor((1, m, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv25, R.shape([1, m, 32, 128])
            )
            lv28: R.DTensor((1, 32, foo_n, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                lv16, axes=[0, 2, 1, 3]
            )
            lv29: R.DTensor((1, 32, m, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                lv26, axes=[0, 2, 1, 3]
            )
            lv30: R.DTensor((1, 32, m, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                lv27, axes=[0, 2, 1, 3]
            )
            lv31: R.DTensor((1, 32, 128, m), "float16", "mesh[0]", "S[1]") = R.permute_dims(
                lv29, axes=[0, 1, 3, 2]
            )
            lv32: R.DTensor((1, 32, foo_n, m), "float16", "mesh[0]", "S[1]") = R.matmul(lv28, lv31)
            lv33: R.DTensor((1, 32, foo_n, m), "float16", "mesh[0]", "S[1]") = R.divide(
                lv32, R.dist.const(8, R.DTensor((), "float16", "mesh[0]", "R"))
            )
            lv34: R.DTensor((1, 32, foo_n, m), "float16", "mesh[0]", "S[1]") = R.maximum(
                lv33, R.dist.const(1, R.DTensor((), "float16", "mesh[0]", "R"))
            )
            lv35: R.DTensor((1, 32, foo_n, m), "float16", "mesh[0]", "S[1]") = R.minimum(lv34, mask)
            lv37: R.DTensor((1, 32, foo_n, m), "float16", "mesh[0]", "S[1]") = R.nn.softmax(
                lv35, axis=-1
            )
            lv39: R.DTensor((1, 32, foo_n, 128), "float16", "mesh[0]", "S[1]") = R.matmul(
                lv37, lv30
            )
            lv40: R.DTensor((1, foo_n, 32, 128), "float16", "mesh[0]", "S[2]") = R.permute_dims(
                lv39, axes=[0, 2, 1, 3]
            )
            lv41: R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "S[2]") = R.reshape(
                lv40, R.shape([1, foo_n, 4096])
            )
            lv42: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]") = R.permute_dims(
                linear_weight3, axes=None
            )
            lv43: R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "R") = R.matmul(lv41, lv42)
            lv44: R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "R") = R.add(input_tokens, lv43)
            gv: R.DTensor((1, foo_n, 4096), "float16", "mesh[0]", "R") = lv44
            return gv

    after = relax.distributed.transform.PropagateSharding()(LlamaAttentionLayerDynamicShape)
    assert_structural_equal(after, ShardedLlamaAttentionLayerDynamicShape)


if __name__ == "__main__":
    tvm_tirx.testing.main()
