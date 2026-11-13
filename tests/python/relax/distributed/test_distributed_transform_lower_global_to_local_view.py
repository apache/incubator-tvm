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
# ruff: noqa: F401

#  type: ignore
from __future__ import annotations

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.s_tir import script as Ts
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def test_mlp():
    @I.ir_module
    class MLP:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @Ts.function(private=True)
        def gelu(
            A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            T_multiply: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_multiply_1 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(128)))
            compute = Ts.sblock_alloc_tensor((T.int64(128), T.int64(128)))
            T_multiply_2 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(128)))
            T_add = Ts.sblock_alloc_tensor((T.int64(128), T.int64(128)))
            for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T.float32(0.70710678118654757)
            for i0, i1 in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_multiply_1[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.erf(T_multiply_1[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock("T_multiply_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(compute[v_ax0, v_ax1])
                    Ts.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = compute[v_ax0, v_ax1] * T.float32(0.5)
            for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_2[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(0.5) + T_multiply_2[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock("T_multiply_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]

        @Ts.function(private=True)
        def matmul(
            A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            matmul_1: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(128)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(A[v_i0, v_k], B[v_k, v_i1])
                    Ts.writes(matmul_1[v_i0, v_i1])
                    with Ts.init():
                        matmul_1[v_i0, v_i1] = T.float32(0)
                    matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
            weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        ) -> R.DTensor((128, 128), "float32", "mesh[0]", "R"):
            cls = MLP
            lv0 = R.dist.call_tir(
                cls.matmul,
                (x, weight1),
                ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "S[1]")],
            )
            lv1 = R.dist.call_tir(
                cls.gelu, (lv0,), ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "S[1]")]
            )
            lv2: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = lv1
            lv3 = R.dist.call_tir(
                cls.matmul,
                (lv2, weight2),
                ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "R")],
            )
            return lv3

    @I.ir_module
    class Expected:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @Ts.function(private=True)
        def gelu1(
            A: T.Tensor((T.int64(128), T.int64(64)), "float32"),
            T_multiply: T.Tensor((T.int64(128), T.int64(64)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_multiply_1 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            compute = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            T_multiply_2 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            T_add = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T.float32(0.70710678118654757)
            for i0, i1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_multiply_1[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.erf(T_multiply_1[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(compute[v_ax0, v_ax1])
                    Ts.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = compute[v_ax0, v_ax1] * T.float32(0.5)
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_2[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(0.5) + T_multiply_2[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]

        @Ts.function(private=True)
        def matmul1(
            A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            B: T.Tensor((T.int64(128), T.int64(64)), "float32"),
            matmul_1: T.Tensor((T.int64(128), T.int64(64)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(64), T.int64(128)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(A[v_i0, v_k], B[v_k, v_i1])
                    Ts.writes(matmul_1[v_i0, v_i1])
                    with Ts.init():
                        matmul_1[v_i0, v_i1] = T.float32(0)
                    matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

        @Ts.function(private=True)
        def matmul2(
            A: T.Tensor((T.int64(128), T.int64(64)), "float32"),
            B: T.Tensor((T.int64(64), T.int64(128)), "float32"),
            matmul_1: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(64)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(A[v_i0, v_k], B[v_k, v_i1])
                    Ts.writes(matmul_1[v_i0, v_i1])
                    with Ts.init():
                        matmul_1[v_i0, v_i1] = T.float32(0)
                    matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
            weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        ) -> R.DTensor((128, 128), "float32", "mesh[0]", "R"):
            cls = Expected
            lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.dist.call_tir_local_view(
                cls.matmul1,
                (x, weight1),
                ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "S[1]")],
            )
            lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.dist.call_tir_local_view(
                cls.gelu1, (lv0,), ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "S[1]")]
            )
            lv2: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = lv1
            gv: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.dist.call_tir_local_view(
                cls.matmul2,
                (lv2, weight2),
                ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "R")],
            )
            lv3: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.ccl.allreduce(
                gv, op_type="sum", in_group=False
            )
            return lv3

    mod = MLP
    mod = relax.distributed.transform.LowerGlobalViewToLocalView()(mod)
    mod = relax.transform.DeadCodeElimination()(mod)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_llama_attention():
    @I.ir_module
    class LlamaAttentionLayer:
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
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2], B[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_add[v_ax0, v_ax1, v_ax2])
                    T_add[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] + B[v_ax0, v_ax1, v_ax2]

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
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = (
                        A[v_ax0, v_ax1, v_ax2, v_ax3] / B[v_ax0, v_ax1, v_ax2, v_ax3]
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
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
                    Ts.writes(matmul[v_i0, v_i1, v_i2])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2] = (
                        matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]
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
                    v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    Ts.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2, v_i3] = (
                        matmul[v_i0, v_i1, v_i2, v_i3]
                        + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]
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
                    v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    Ts.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2, v_i3] = (
                        matmul[v_i0, v_i1, v_i2, v_i3]
                        + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]
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
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_maximum[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_maximum[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(
                        A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3]
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
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, T.int64(0), v_ax2, v_ax3])
                    Ts.writes(T_minimum[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_minimum[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(
                        A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, T.int64(0), v_ax2, v_ax3]
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
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(
                        A[
                            T.int64(0),
                            ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax1)
                            % T.int64(256),
                            (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                        T.int64(0),
                        ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax1) % T.int64(256),
                        (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
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
                    Ts.reads(
                        A[
                            T.int64(0),
                            ((v_ax2 // T.int64(128) + v_ax1) // T.int64(32) + v_ax0) % T.int64(256),
                            (v_ax2 // T.int64(128) + v_ax1) % T.int64(32),
                            v_ax2 % T.int64(128),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = A[
                        T.int64(0),
                        ((v_ax2 // T.int64(128) + v_ax1) // T.int64(32) + v_ax0) % T.int64(256),
                        (v_ax2 // T.int64(128) + v_ax1) % T.int64(32),
                        v_ax2 % T.int64(128),
                    ]

        @Ts.function(private=True)
        def reshape2(
            A: T.Tensor((T.int64(256), T.int64(32), T.int64(128)), "float16"),
            T_reshape: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(32), T.int64(128)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(
                        A[
                            ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax1) % T.int64(256),
                            (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                            v_ax3 % T.int64(128),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                        ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax1) % T.int64(256),
                        (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                        v_ax3 % T.int64(128),
                    ]

        @Ts.function(private=True)
        def reshape3(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
            T_reshape: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(256), T.int64(4096)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(
                        A[
                            T.int64(0),
                            (v_ax2 // T.int64(4096) + v_ax1) % T.int64(256),
                            v_ax2 % T.int64(4096) // T.int64(128),
                            v_ax2 % T.int64(128),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = A[
                        T.int64(0),
                        (v_ax2 // T.int64(4096) + v_ax1) % T.int64(256),
                        v_ax2 % T.int64(4096) // T.int64(128),
                        v_ax2 % T.int64(128),
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
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_k])
                    Ts.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                    with Ts.init():
                        T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float16(-65504)
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(
                        T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_exp"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                    Ts.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                    T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(
                        A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]
                    )
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_expsum"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                    Ts.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                    with Ts.init():
                        T_softmax_expsum[v_i0, v_i1, v_i2] = T.float16(0)
                    T_softmax_expsum[v_i0, v_i1, v_i2] = (
                        T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(
                        T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2]
                    )
                    Ts.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                    Ts.sblock_attr({"axis": 3})
                    T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = (
                        T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
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
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

        @Ts.function(private=True)
        def transpose2(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(128)), "float16"),
            T_transpose: T.Tensor((T.int64(1), T.int64(32), T.int64(128), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(128), T.int64(256)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax3, v_ax2])
                    Ts.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax3, v_ax2]

        @Ts.function(private=True)
        def transpose3(
            A: T.Tensor((T.int64(1), T.int64(32), T.int64(256), T.int64(128)), "float16"),
            T_transpose: T.Tensor((T.int64(1), T.int64(256), T.int64(32), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(32), T.int64(128)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

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
            cls = LlamaAttentionLayer
            lv6 = R.dist.call_tir(
                cls.rms_norm,
                (input_tokens, rms_norm_weight),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            lv7 = R.dist.call_tir(
                cls.transpose,
                (linear_weight,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
            )
            lv8 = R.dist.call_tir(
                cls.matmul,
                (lv6, lv7),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
            )
            lv9 = R.dist.call_tir(
                cls.reshape,
                (lv8,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv10 = R.dist.call_tir(
                cls.transpose,
                (linear_weight1,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
            )
            lv11 = R.dist.call_tir(
                cls.matmul,
                (lv6, lv10),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
            )
            lv12 = R.dist.call_tir(
                cls.reshape,
                (lv11,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv13 = R.dist.call_tir(
                cls.transpose,
                (linear_weight2,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
            )
            lv14 = R.dist.call_tir(
                cls.matmul,
                (lv6, lv13),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
            )
            lv15 = R.dist.call_tir(
                cls.reshape,
                (lv14,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
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
            lv18 = R.dist.call_tir(
                cls.reshape1,
                (lv17,),
                ty_args=[R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv19 = R.dist.call_tir(
                cls.reshape1,
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
                cls.reshape2,
                (lv24,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv27 = R.dist.call_tir(
                cls.reshape2,
                (lv25,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv28 = R.dist.call_tir(
                cls.transpose1,
                (lv16,),
                ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv29 = R.dist.call_tir(
                cls.transpose1,
                (lv26,),
                ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv30 = R.dist.call_tir(
                cls.transpose1,
                (lv27,),
                ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv31 = R.dist.call_tir(
                cls.transpose2,
                (lv29,),
                ty_args=[R.DTensor((1, 32, 128, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv32 = R.dist.call_tir(
                cls.matmul1,
                (lv28, lv31),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv33 = R.dist.call_tir(
                cls.divide,
                (lv32, div_const),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv34 = R.dist.call_tir(
                cls.maximum,
                (lv33, maximum_const),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv35 = R.dist.call_tir(
                cls.minimum,
                (lv34, mask),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv37 = R.dist.call_tir(
                cls.softmax,
                (lv35,),
                ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
            )
            lv39 = R.dist.call_tir(
                cls.matmul2,
                (lv37, lv30),
                ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
            )
            lv40 = R.dist.call_tir(
                cls.transpose3,
                (lv39,),
                ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
            )
            lv41 = R.dist.call_tir(
                cls.reshape3,
                (lv40,),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
            )
            lv42 = R.dist.call_tir(
                cls.transpose,
                (linear_weight3,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]")],
            )
            lv43 = R.dist.call_tir(
                cls.matmul,
                (lv41, lv42),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            lv44 = R.dist.call_tir(
                cls.add,
                (input_tokens, lv43),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            gv: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = lv44
            return gv

    @I.ir_module
    class Expected:
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
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2], B[v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_add[v_ax0, v_ax1, v_ax2])
                    T_add[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] + B[v_ax0, v_ax1, v_ax2]

        @Ts.function(private=True)
        def divide1(
            A: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
            T_divide: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(256), T.int64(256)):
                with Ts.sblock("T_divide"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = (
                        A[v_ax0, v_ax1, v_ax2, v_ax3] / B[v_ax0, v_ax1, v_ax2, v_ax3]
                    )

        @Ts.function(private=True)
        def matmul11(
            A: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(128)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(16), T.int64(128), T.int64(256)), "float16"),
            matmul: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3, k in T.grid(
                T.int64(1), T.int64(16), T.int64(256), T.int64(256), T.int64(128)
            ):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    Ts.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2, v_i3] = (
                        matmul[v_i0, v_i1, v_i2, v_i3]
                        + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]
                    )

        @Ts.function(private=True)
        def matmul21(
            A: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(128)), "float16"),
            matmul: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3, k in T.grid(
                T.int64(1), T.int64(16), T.int64(256), T.int64(128), T.int64(256)
            ):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    Ts.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2, v_i3] = (
                        matmul[v_i0, v_i1, v_i2, v_i3]
                        + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]
                    )

        @Ts.function(private=True)
        def matmul3(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(4096), T.int64(2048)), "float16"),
            matmul: T.Tensor((T.int64(1), T.int64(256), T.int64(2048)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(256), T.int64(2048), T.int64(4096)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
                    Ts.writes(matmul[v_i0, v_i1, v_i2])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2] = (
                        matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]
                    )

        @Ts.function(private=True)
        def matmul4(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(2048)), "float16"),
            B: T.Tensor((T.int64(2048), T.int64(4096)), "float16"),
            matmul: T.Tensor((T.int64(1), T.int64(256), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(256), T.int64(4096), T.int64(2048)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
                    Ts.writes(matmul[v_i0, v_i1, v_i2])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2] = (
                        matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]
                    )

        @Ts.function(private=True)
        def maximum1(
            A: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
            T_maximum: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(256), T.int64(256)):
                with Ts.sblock("T_maximum"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_maximum[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_maximum[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(
                        A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3]
                    )

        @Ts.function(private=True)
        def minimum1(
            A: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
            B: T.Tensor((T.int64(1), T.int64(1), T.int64(256), T.int64(256)), "float16"),
            T_minimum: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(256), T.int64(256)):
                with Ts.sblock("T_minimum"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, T.int64(0), v_ax2, v_ax3])
                    Ts.writes(T_minimum[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_minimum[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(
                        A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, T.int64(0), v_ax2, v_ax3]
                    )

        @Ts.function(private=True)
        def reshape11(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(16), T.int64(128)), "float16"),
            T_reshape: T.Tensor((T.int64(256), T.int64(16), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2 in T.grid(T.int64(256), T.int64(16), T.int64(128)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(
                        A[
                            T.int64(0),
                            ((v_ax2 // T.int64(128) + v_ax1) // T.int64(32) + v_ax0) % T.int64(256),
                            (v_ax2 // T.int64(128) + v_ax1) % T.int64(32),
                            v_ax2 % T.int64(128),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = A[
                        T.int64(0),
                        ((v_ax2 // T.int64(128) + v_ax1) // T.int64(32) + v_ax0) % T.int64(256),
                        (v_ax2 // T.int64(128) + v_ax1) % T.int64(32),
                        v_ax2 % T.int64(128),
                    ]

        @Ts.function(private=True)
        def reshape21(
            A: T.Tensor((T.int64(256), T.int64(16), T.int64(128)), "float16"),
            T_reshape: T.Tensor((T.int64(1), T.int64(256), T.int64(16), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(16), T.int64(128)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(
                        A[
                            ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax1) % T.int64(256),
                            (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                            v_ax3 % T.int64(128),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                        ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax1) % T.int64(256),
                        (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                        v_ax3 % T.int64(128),
                    ]

        @Ts.function(private=True)
        def reshape31(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(16), T.int64(128)), "float16"),
            T_reshape: T.Tensor((T.int64(1), T.int64(256), T.int64(2048)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(256), T.int64(2048)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(
                        A[
                            T.int64(0),
                            (v_ax2 // T.int64(4096) + v_ax1) % T.int64(256),
                            v_ax2 % T.int64(4096) // T.int64(128),
                            v_ax2 % T.int64(128),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = A[
                        T.int64(0),
                        (v_ax2 // T.int64(4096) + v_ax1) % T.int64(256),
                        v_ax2 % T.int64(4096) // T.int64(128),
                        v_ax2 % T.int64(128),
                    ]

        @Ts.function(private=True)
        def reshape4(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(2048)), "float16"),
            T_reshape: T.Tensor((T.int64(1), T.int64(256), T.int64(16), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(16), T.int64(128)):
                with Ts.sblock("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(
                        A[
                            T.int64(0),
                            ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax1)
                            % T.int64(256),
                            (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                        ]
                    )
                    Ts.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                        T.int64(0),
                        ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax1) % T.int64(256),
                        (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
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

        @Ts.function
        def rotary_embedding1(
            A: T.Tensor((T.int64(1), 256, T.int64(16), T.int64(128)), "float16"),
            B: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            C: T.Tensor((T.int64(2048), T.int64(128)), "float16"),
            rotary: T.Tensor((T.int64(1), 256, T.int64(16), T.int64(128)), "float16"),
        ):
            T.func_attr({"global_symbol": "rotary_embedding", "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3 in T.grid(T.int64(1), 256, T.int64(16), T.int64(128)):
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

        @Ts.function(private=True)
        def softmax1(
            A: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"),
            T_softmax_norm: T.Tensor(
                (T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"
            ),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_softmax_maxelem = Ts.sblock_alloc_tensor(
                (T.int64(1), T.int64(16), T.int64(256)), "float16"
            )
            T_softmax_exp = Ts.sblock_alloc_tensor(
                (T.int64(1), T.int64(16), T.int64(256), T.int64(256)), "float16"
            )
            T_softmax_expsum = Ts.sblock_alloc_tensor(
                (T.int64(1), T.int64(16), T.int64(256)), "float16"
            )
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(16), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_maxelem"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_k])
                    Ts.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
                    with Ts.init():
                        T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float16(-65504)
                    T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(
                        T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_exp"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
                    Ts.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
                    T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(
                        A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]
                    )
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(16), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_expsum"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
                    Ts.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
                    with Ts.init():
                        T_softmax_expsum[v_i0, v_i1, v_i2] = T.float16(0)
                    T_softmax_expsum[v_i0, v_i1, v_i2] = (
                        T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(16), T.int64(256), T.int64(256)):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(
                        T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2]
                    )
                    Ts.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
                    Ts.sblock_attr({"axis": 3})
                    T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = (
                        T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
                    )

        @Ts.function(private=True)
        def transpose11(
            A: T.Tensor((T.int64(1), T.int64(256), T.int64(16), T.int64(128)), "float16"),
            T_transpose: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(256), T.int64(128)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

        @Ts.function(private=True)
        def transpose21(
            A: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(128)), "float16"),
            T_transpose: T.Tensor((T.int64(1), T.int64(16), T.int64(128), T.int64(256)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16), T.int64(128), T.int64(256)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax3, v_ax2])
                    Ts.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax3, v_ax2]

        @Ts.function(private=True)
        def transpose31(
            A: T.Tensor((T.int64(1), T.int64(16), T.int64(256), T.int64(128)), "float16"),
            T_transpose: T.Tensor((T.int64(1), T.int64(256), T.int64(16), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(16), T.int64(128)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]

        @Ts.function(private=True)
        def transpose4(
            A: T.Tensor((T.int64(2048), T.int64(4096)), "float16"),
            T_transpose: T.Tensor((T.int64(4096), T.int64(2048)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(4096), T.int64(2048)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax1, v_ax0])
                    Ts.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = A[v_ax1, v_ax0]

        @Ts.function(private=True)
        def transpose5(
            A: T.Tensor((T.int64(4096), T.int64(2048)), "float16"),
            T_transpose: T.Tensor((T.int64(2048), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(2048), T.int64(4096)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax1, v_ax0])
                    Ts.writes(T_transpose[v_ax0, v_ax1])
                    T_transpose[v_ax0, v_ax1] = A[v_ax1, v_ax0]

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
            cls = Expected
            lv6: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = R.dist.call_tir_local_view(
                cls.rms_norm,
                (input_tokens, rms_norm_weight),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            lv7: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.dist.call_tir_local_view(
                cls.transpose4,
                (linear_weight,),
                ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
            )
            lv8: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.matmul3,
                    (lv6, lv7),
                    ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv9: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.reshape4,
                    (lv8,),
                    ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv10: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.transpose4,
                    (linear_weight1,),
                    ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv11: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.matmul3,
                    (lv6, lv10),
                    ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv12: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.reshape4,
                    (lv11,),
                    ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv13: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.transpose4,
                    (linear_weight2,),
                    ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv14: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.matmul3,
                    (lv6, lv13),
                    ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv15: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.reshape4,
                    (lv14,),
                    ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv16: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.rotary_embedding1,
                    (lv9, cos_cached, sin_cached),
                    ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv17: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.rotary_embedding1,
                    (lv12, cos_cached, sin_cached),
                    ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv18: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.reshape11,
                    (lv17,),
                    ty_args=[R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv19: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.reshape11,
                    (lv15,),
                    ty_args=[R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]")],
                )
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
            lv26: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.reshape21,
                    (lv24,),
                    ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv27: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.reshape21,
                    (lv25,),
                    ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv28: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.transpose11,
                    (lv16,),
                    ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv29: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.transpose11,
                    (lv26,),
                    ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv30: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.transpose11,
                    (lv27,),
                    ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv31: R.DTensor((1, 32, 128, 256), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.transpose21,
                    (lv29,),
                    ty_args=[R.DTensor((1, 32, 128, 256), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv32: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.matmul11,
                    (lv28, lv31),
                    ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv33: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.divide1,
                    (lv32, div_const),
                    ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv34: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.maximum1,
                    (lv33, maximum_const),
                    ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv35: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.minimum1,
                    (lv34, mask),
                    ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv37: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.softmax1,
                    (lv35,),
                    ty_args=[R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv39: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = (
                R.dist.call_tir_local_view(
                    cls.matmul21,
                    (lv37, lv30),
                    ty_args=[R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]")],
                )
            )
            lv40: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.transpose31,
                    (lv39,),
                    ty_args=[R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv41: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = (
                R.dist.call_tir_local_view(
                    cls.reshape31,
                    (lv40,),
                    ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]")],
                )
            )
            lv42: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]") = (
                R.dist.call_tir_local_view(
                    cls.transpose5,
                    (linear_weight3,),
                    ty_args=[R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]")],
                )
            )
            gv: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = R.dist.call_tir_local_view(
                cls.matmul4,
                (lv41, lv42),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            lv43: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = R.ccl.allreduce(
                gv, op_type="sum", in_group=False
            )
            lv44: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = R.dist.call_tir_local_view(
                cls.add,
                (input_tokens, lv43),
                ty_args=[R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R")],
            )
            gv_1: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = lv44
            return gv_1

    mod = relax.distributed.transform.LowerGlobalViewToLocalView()(LlamaAttentionLayer)
    mod = relax.transform.DeadCodeElimination()(mod)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    test_mlp()
    test_llama_attention()
