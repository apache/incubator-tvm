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
# pylint: disable=missing-docstring,line-too-long,invalid-name,too-few-public-methods,too-many-locals
# ruff: noqa: E501, F841

import tvm_tirx.testing
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def test_decode_gemv_1():
    # NK layout + K as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def func(W: T.Tensor((4096, 512), "uint32"), S: T.Tensor((4096, 128), "float16"), V: T.Tensor((1, 1, 4096), "float16"), C: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            # with Ts.sblock("root"):
            B = Ts.sblock_alloc_tensor((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    Ts.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    Ts.writes(C[v_i0, v_i1, v_i2])
                    with Ts.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @I.ir_module
    class After:
        @Ts.function
        def func(W: T.Tensor((4096, 512), 'uint32'), S: T.Tensor((4096, 128), 'float16'), V: T.Tensor((1, 1, 4096), 'float16'), C: T.Tensor((1, 1, 4096), 'float16')):
            T.func_attr({"global_symbol": "main", "tirx.is_scheduled": True, "tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                C_rf_local = Ts.sblock_alloc_tensor((512, 1, 1, 4096), "float16", scope="local")
                for ax0_fused in T.parallel(4096, annotations={"thread_binding": "blockIdx.x"}):
                    for ax1_0_fused_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("matmul_rf_init"):
                            vax1_0_fused_1 = Ts.axis.spatial(512, ax1_0_fused_1)
                            v0 = Ts.axis.spatial(4096, ax0_fused)
                            Ts.reads()
                            Ts.writes(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                            C_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float16(0)
                        for ax1_0_fused_0 in range(1):
                            for ax1_1 in range(8):
                                with Ts.sblock("matmul_rf_update"):
                                    vax1_0_fused_1 = Ts.axis.spatial(512, ax1_0_fused_1)
                                    v0 = Ts.axis.spatial(4096, ax0_fused)
                                    vax1_0_fused_0 = Ts.axis.reduce(1, ax1_0_fused_0)
                                    vax1_1 = Ts.axis.reduce(8, ax1_1)
                                    Ts.reads(C_rf_local[vax1_0_fused_1, 0, 0, v0], V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1], W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                                    Ts.writes(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                                    C_rf_local[vax1_0_fused_1, 0, 0, v0] = C_rf_local[vax1_0_fused_1, 0, 0, v0] + V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                    for ax1_fused in range(1):
                        for ax0 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("matmul"):
                                vax1_0_fused_1 = Ts.axis.reduce(512, ax0)
                                v0 = Ts.axis.spatial(4096, ax0_fused)
                                Ts.reads(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                                Ts.writes(C[0, 0, v0])
                                with Ts.init():
                                    C[0, 0, v0] = T.float16(0)
                                C[0, 0, v0] = C[0, 0, v0] + C_rf_local[vax1_0_fused_1, 0, 0, v0]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_2():
    # KN layout + K as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def func(W: T.Tensor((512, 4096), "uint32"), S: T.Tensor((128, 4096), "float16"), V: T.Tensor((1, 1, 4096), "float16"), C: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            # with Ts.sblock("root"):
            B = Ts.sblock_alloc_tensor((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(W[v_i // 8, v_j], S[v_i // 32, v_j])
                    Ts.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i // 8, v_j], T.Cast("uint32", v_i % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i // 32, v_j]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(V[v_i0, v_i1, v_k], B[v_k, v_i2])
                    Ts.writes(C[v_i0, v_i1, v_i2])
                    with Ts.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_k, v_i2]

    @I.ir_module
    class After:
        @Ts.function
        def func(W: T.Tensor((512, 4096), "uint32"), S: T.Tensor((128, 4096), "float16"), V: T.Tensor((1, 1, 4096), "float16"), C: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            C_rf_local = Ts.sblock_alloc_tensor((16, 1, 1, 4096), "float16", scope="local")
            for i2_i0_i1_fused_0 in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for i2_i0_i1_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for k_0_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("matmul_rf_init"):
                            vk_0_fused_1 = Ts.axis.spatial(16, k_0_fused_1)
                            v_i2 = Ts.axis.spatial(4096, i2_i0_i1_fused_0 * 16 + i2_i0_i1_fused_1)
                            C_rf_local[vk_0_fused_1, 0, 0, v_i2] = T.float16(0)
                        for k_0_fused_0, k_1 in T.grid(32, 8):
                            with Ts.sblock("matmul_rf_update"):
                                vk_0_fused_1 = Ts.axis.spatial(16, k_0_fused_1)
                                v_i2 = Ts.axis.spatial(4096, i2_i0_i1_fused_0 * 16 + i2_i0_i1_fused_1)
                                vk_0_fused_0, vk_1 = Ts.axis.remap("RR", [k_0_fused_0, k_1])
                                C_rf_local[vk_0_fused_1, 0, 0, v_i2] = C_rf_local[vk_0_fused_1, 0, 0, v_i2] + V[0, 0, vk_0_fused_0 * 128 + vk_0_fused_1 * 8 + vk_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[(vk_0_fused_0 * 128 + vk_0_fused_1 * 8 + vk_1) // 8, v_i2], T.Cast("uint32", (vk_0_fused_0 * 128 + vk_0_fused_1 * 8 + vk_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[(vk_0_fused_0 * 128 + vk_0_fused_1 * 8 + vk_1) // 32, v_i2])
                for ax1_ax2_ax3_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("matmul"):
                            vk_0_fused_1 = Ts.axis.reduce(16, ax0_fused)
                            v_i2 = Ts.axis.spatial(4096, i2_i0_i1_fused_0 * 16 + ax1_ax2_ax3_fused)
                            with Ts.init():
                                C[0, 0, v_i2] = T.float16(0)
                            C[0, 0, v_i2] = C[0, 0, v_i2] + C_rf_local[vk_0_fused_1, 0, 0, v_i2]

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_3():
    # NK layout + N as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def func(W: T.Tensor((512, 4096), "uint32"), S: T.Tensor((128, 4096), "float16"), V: T.Tensor((1, 1, 4096), "float16"), C: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            # with Ts.sblock("root"):
            B = Ts.sblock_alloc_tensor((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(W[v_i // 8, v_j], S[v_i // 32, v_j])
                    Ts.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i // 8, v_j], T.Cast("uint32", v_i % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i // 32, v_j]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    Ts.writes(C[v_i0, v_i1, v_i2])
                    with Ts.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_i2, v_k]

    @I.ir_module
    class After:
        @Ts.function
        def func(W: T.Tensor((512, 4096), 'uint32'), S: T.Tensor((128, 4096), 'float16'), V: T.Tensor((1, 1, 4096), 'float16'), C: T.Tensor((1, 1, 4096), 'float16')):
            T.func_attr({"global_symbol": "main", "tirx.is_scheduled": True, "tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                C_rf_local = Ts.sblock_alloc_tensor((1024, 1, 1, 4096), "float16", scope="local")
                for ax0_0_fused in T.parallel(512, annotations={"thread_binding": "blockIdx.x"}):
                    for ax1_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax0_1_init in range(8):
                            with Ts.sblock("matmul_rf_init"):
                                vax1_fused_1 = Ts.axis.spatial(1024, ax1_fused_1)
                                v0 = Ts.axis.spatial(4096, ax0_0_fused * 8 + ax0_1_init)
                                Ts.reads()
                                Ts.writes(C_rf_local[vax1_fused_1, 0, 0, v0])
                                C_rf_local[vax1_fused_1, 0, 0, v0] = T.float16(0)
                        for ax1_fused_0 in range(4):
                            for ax0_1 in range(8):
                                with Ts.sblock("matmul_rf_update"):
                                    vax1_fused_1 = Ts.axis.spatial(1024, ax1_fused_1)
                                    v0 = Ts.axis.spatial(4096, ax0_0_fused * 8 + ax0_1)
                                    vax1_fused_0 = Ts.axis.reduce(4, ax1_fused_0)
                                    Ts.reads(C_rf_local[vax1_fused_1, 0, 0, v0], V[0, 0, vax1_fused_0 * 1024 + vax1_fused_1], W[v0 // 8, vax1_fused_0 * 1024 + vax1_fused_1], S[v0 // 32, vax1_fused_0 * 1024 + vax1_fused_1])
                                    Ts.writes(C_rf_local[vax1_fused_1, 0, 0, v0])
                                    C_rf_local[vax1_fused_1, 0, 0, v0] = C_rf_local[vax1_fused_1, 0, 0, v0] + V[0, 0, vax1_fused_0 * 1024 + vax1_fused_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v0 // 8, vax1_fused_0 * 1024 + vax1_fused_1], T.Cast("uint32", v0 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0 // 32, vax1_fused_0 * 1024 + vax1_fused_1])
                    for ax1_fused_0 in range(1):
                        for ax0 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                            for ax1_fused_1 in range(8):
                                with Ts.sblock("matmul"):
                                    vax1_fused_1 = Ts.axis.reduce(1024, ax0)
                                    v0 = Ts.axis.spatial(4096, ax0_0_fused * 8 + ax1_fused_1)
                                    Ts.reads(C_rf_local[vax1_fused_1, 0, 0, v0])
                                    Ts.writes(C[0, 0, v0])
                                    with Ts.init():
                                        C[0, 0, v0] = T.float16(0)
                                    C[0, 0, v0] = C[0, 0, v0] + C_rf_local[vax1_fused_1, 0, 0, v0]

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_4():
    # KN layout + N as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def func(W: T.Tensor((4096, 512), "uint32"), S: T.Tensor((4096, 128), "float16"), V: T.Tensor((1, 1, 4096), "float16"), C: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            # with Ts.sblock("root"):
            B = Ts.sblock_alloc_tensor((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    Ts.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(V[v_i0, v_i1, v_k], B[v_k, v_i2])
                    Ts.writes(C[v_i0, v_i1, v_i2])
                    with Ts.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_k, v_i2]

    @I.ir_module
    class After:
        @Ts.function
        def func(W: T.Tensor((4096, 512), "uint32"), S: T.Tensor((4096, 128), "float16"), V: T.Tensor((1, 1, 4096), "float16"), C: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            C_rf_local = Ts.sblock_alloc_tensor((16, 1, 1, 4096), "float16", scope="local")
            for i2_0_i0_i1_fused_0 in T.parallel(32, annotations={"thread_binding": "blockIdx.x"}):
                for i2_0_i0_i1_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for k_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        for i2_1_init in range(8):
                            with Ts.sblock("matmul_rf_init"):
                                vk_fused_1 = Ts.axis.spatial(16, k_fused_1)
                                v_i2 = Ts.axis.spatial(4096, i2_0_i0_i1_fused_0 * 128 + i2_0_i0_i1_fused_1 * 8 + i2_1_init)
                                C_rf_local[vk_fused_1, 0, 0, v_i2] = T.float16(0)
                        for k_fused_0, i2_1 in T.grid(256, 8):
                            with Ts.sblock("matmul_rf_update"):
                                vk_fused_1 = Ts.axis.spatial(16, k_fused_1)
                                v_i2 = Ts.axis.spatial(4096, i2_0_i0_i1_fused_0 * 128 + i2_0_i0_i1_fused_1 * 8 + i2_1)
                                vk_fused_0 = Ts.axis.reduce(256, k_fused_0)
                                C_rf_local[vk_fused_1, 0, 0, v_i2] = C_rf_local[vk_fused_1, 0, 0, v_i2] + V[0, 0, vk_fused_0 * 16 + vk_fused_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[vk_fused_0 * 16 + vk_fused_1, v_i2 // 8], T.Cast("uint32", v_i2 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[vk_fused_0 * 16 + vk_fused_1, v_i2 // 32])
                for ax1_ax2_ax3_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_ax2_ax3_fused_1 in range(8):
                        for ax0_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                            with Ts.sblock("matmul"):
                                vk_fused_1 = Ts.axis.reduce(16, ax0_fused)
                                v_i2 = Ts.axis.spatial(4096, i2_0_i0_i1_fused_0 * 128 + ax1_ax2_ax3_fused_0 * 8 + ax1_ax2_ax3_fused_1)
                                with Ts.init():
                                    C[0, 0, v_i2] = T.float16(0)
                                C[0, 0, v_i2] = C[0, 0, v_i2] + C_rf_local[vk_fused_1, 0, 0, v_i2]

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_sigmoid():
    # NK layout + K as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def func(W: T.Tensor((4096, 512), "uint32"), S: T.Tensor((4096, 128), "float16"), V: T.Tensor((1, 1, 4096), "float16"), D: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            # with Ts.sblock("root"):
            B = Ts.sblock_alloc_tensor((4096, 4096), "float16")
            C = Ts.sblock_alloc_tensor((1, 1, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    Ts.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    Ts.writes(C[v_i0, v_i1, v_i2])
                    with Ts.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_i2, v_k]
            for i0, i1, i2 in T.grid(1, 1, 4096):
                with Ts.sblock("sigmoid"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(C[v_i0, v_i1, v_i2])
                    Ts.writes(D[v_i0, v_i1, v_i2])
                    D[v_i0, v_i1, v_i2] = T.sigmoid(C[v_i0, v_i1, v_i2])

    @I.ir_module
    class After:
        @Ts.function
        def func(W: T.Tensor((4096, 512), 'uint32'), S: T.Tensor((4096, 128), 'float16'), V: T.Tensor((1, 1, 4096), 'float16'), D: T.Tensor((1, 1, 4096), 'float16')):
            T.func_attr({"global_symbol": "main", "tirx.is_scheduled": True, "tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                C_local = Ts.sblock_alloc_tensor((1, 1, 4096), "float16", scope="local")
                C_rf_local = Ts.sblock_alloc_tensor((512, 1, 1, 4096), "float16", scope="local")
                for ax0_fused in T.parallel(4096, annotations={"thread_binding": "blockIdx.x"}):
                    for ax1_0_fused_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("matmul_rf_init"):
                            vax1_0_fused_1 = Ts.axis.spatial(512, ax1_0_fused_1)
                            v0 = Ts.axis.spatial(4096, ax0_fused)
                            Ts.reads()
                            Ts.writes(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                            C_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float16(0)
                        for ax1_0_fused_0 in range(1):
                            for ax1_1 in range(8):
                                with Ts.sblock("matmul_rf_update"):
                                    vax1_0_fused_1 = Ts.axis.spatial(512, ax1_0_fused_1)
                                    v0 = Ts.axis.spatial(4096, ax0_fused)
                                    vax1_0_fused_0 = Ts.axis.reduce(1, ax1_0_fused_0)
                                    vax1_1 = Ts.axis.reduce(8, ax1_1)
                                    Ts.reads(C_rf_local[vax1_0_fused_1, 0, 0, v0], V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1], W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                                    Ts.writes(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                                    C_rf_local[vax1_0_fused_1, 0, 0, v0] = C_rf_local[vax1_0_fused_1, 0, 0, v0] + V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                    for ax1_fused in range(1):
                        for ax0 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("matmul"):
                                vax1_0_fused_1 = Ts.axis.reduce(512, ax0)
                                v0 = Ts.axis.spatial(4096, ax0_fused)
                                Ts.reads(C_rf_local[vax1_0_fused_1, 0, 0, v0])
                                Ts.writes(C_local[0, 0, v0])
                                with Ts.init():
                                    C_local[0, 0, v0] = T.float16(0)
                                C_local[0, 0, v0] = C_local[0, 0, v0] + C_rf_local[vax1_0_fused_1, 0, 0, v0]
                    for ax0 in range(1):
                        with Ts.sblock("sigmoid"):
                            v0 = Ts.axis.spatial(4096, ax0_fused + ax0)
                            Ts.reads(C_local[0, 0, v0])
                            Ts.writes(D[0, 0, v0])
                            D[0, 0, v0] = T.sigmoid(C_local[0, 0, v0])

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_decode_gemv_1_fp32():
    # NK layout + K as decode dim
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def func(W: T.Tensor((4096, 512), "uint32"), S: T.Tensor((4096, 128), "float16"), V: T.Tensor((1, 1, 4096), "float16"), C: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            # with Ts.sblock("root"):
            B = Ts.sblock_alloc_tensor((4096, 4096), "float16")
            C_fp32 = Ts.sblock_alloc_tensor((1, 1, 4096), "float32")
            for i, j in T.grid(4096, 4096):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    Ts.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    Ts.writes(C_fp32[v_i0, v_i1, v_i2])
                    with Ts.init():
                        C_fp32[v_i0, v_i1, v_i2] = T.float16(0)
                    C_fp32[v_i0, v_i1, v_i2] = C_fp32[v_i0, v_i1, v_i2] + T.Cast("float32", V[v_i0, v_i1, v_k]) * T.Cast("float32", B[v_i2, v_k])
            for i0, i1, i2 in T.grid(1, 1, 4096):
                with Ts.sblock("cast"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(C_fp32[v_i0, v_i1, v_i2])
                    Ts.writes(C[v_i0, v_i1, v_i2])
                    C[v_i0, v_i1, v_i2] = T.Cast("float16", C_fp32[v_i0, v_i1, v_i2])

    @I.ir_module
    class After:
        @Ts.function
        def func(W: T.Tensor((4096, 512), 'uint32'), S: T.Tensor((4096, 128), 'float16'), V: T.Tensor((1, 1, 4096), 'float16'), C: T.Tensor((1, 1, 4096), 'float16')):
            T.func_attr({"global_symbol": "main", "tirx.is_scheduled": True, "tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                C_fp32_local = Ts.sblock_alloc_tensor((1, 1, 4096), scope="local")
                C_fp32_rf_local = Ts.sblock_alloc_tensor((512, 1, 1, 4096), scope="local")
                for ax0_fused in T.parallel(4096, annotations={"thread_binding": "blockIdx.x"}):
                    for ax1_0_fused_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("matmul_rf_init"):
                            vax1_0_fused_1 = Ts.axis.spatial(512, ax1_0_fused_1)
                            v0 = Ts.axis.spatial(4096, ax0_fused)
                            Ts.reads()
                            Ts.writes(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0])
                            C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float32(0)
                        for ax1_0_fused_0 in range(1):
                            for ax1_1 in range(8):
                                with Ts.sblock("matmul_rf_update"):
                                    vax1_0_fused_1 = Ts.axis.spatial(512, ax1_0_fused_1)
                                    v0 = Ts.axis.spatial(4096, ax0_fused)
                                    vax1_0_fused_0 = Ts.axis.reduce(1, ax1_0_fused_0)
                                    vax1_1 = Ts.axis.reduce(8, ax1_1)
                                    Ts.reads(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0], V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1], W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                                    Ts.writes(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0])
                                    C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] = C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0] + T.Cast("float32", V[0, 0, vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1]) * T.Cast("float32", (T.Cast("float16", T.bitwise_and(T.shift_right(W[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 8], T.Cast("uint32", (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v0, (vax1_0_fused_0 * 4096 + vax1_0_fused_1 * 8 + vax1_1) // 32])
                    for ax1_fused in range(1):
                        for ax0 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("matmul"):
                                vax1_0_fused_1 = Ts.axis.reduce(512, ax0)
                                v0 = Ts.axis.spatial(4096, ax0_fused)
                                Ts.reads(C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0])
                                Ts.writes(C_fp32_local[0, 0, v0])
                                with Ts.init():
                                    C_fp32_local[0, 0, v0] = T.float32(0)
                                C_fp32_local[0, 0, v0] = C_fp32_local[0, 0, v0] + C_fp32_rf_local[vax1_0_fused_1, 0, 0, v0]
                    for ax0 in range(1):
                        with Ts.sblock("cast"):
                            v0 = Ts.axis.spatial(4096, ax0_fused + ax0)
                            Ts.reads(C_fp32_local[0, 0, v0])
                            Ts.writes(C[0, 0, v0])
                            C[0, 0, v0] = T.Cast("float16", C_fp32_local[0, 0, v0])

    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_reduction_no_spatial():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((1, 1, 4096), "float16"), B: T.Tensor((4096,), "float16"), rms_norm: T.Tensor((1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            Ared_temp = Ts.sblock_alloc_tensor((1, 1))
            for ax0 in range(4096):
                with Ts.sblock("Ared_temp"):
                    v0 = Ts.axis.reduce(4096, ax0)
                    with Ts.init():
                        Ared_temp[0, 0] = T.float32(0)
                    Ared_temp[0, 0] = Ared_temp[0, 0] + T.Cast("float32", A[0, 0, v0]) * T.Cast("float32", A[0, 0, v0])
            for ax0 in range(4096):
                with Ts.sblock("rms_norm"):
                    v0 = Ts.axis.spatial(4096, ax0)
                    rms_norm[0, v0] = T.Cast("float16", T.Cast("float32", B[v0]) * (T.Cast("float32", A[0, 0, v0]) / T.sqrt(Ared_temp[0, 0] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

    @I.ir_module
    class After:
        @Ts.function
        def main(A: T.Tensor((1, 1, 4096), 'float16'), B: T.Tensor((4096,), 'float16'), rms_norm: T.Tensor((1, 4096), 'float16')):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                Ared_temp_shared = Ts.sblock_alloc_tensor((1, 1), scope="shared")
                Ared_temp_rf_local = Ts.sblock_alloc_tensor((1024, 1, 1), scope="local")
                for ax0_fused in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                    for ax1_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("Ared_temp_rf_init"):
                            vax1_fused_1 = Ts.axis.spatial(1024, ax1_fused_1)
                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                            Ts.reads()
                            Ts.writes(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                            Ared_temp_rf_local[vax1_fused_1, 0, 0] = T.float32(0)
                        for ax1_fused_0 in range(4):
                            for u in range(1):
                                with Ts.sblock("Ared_temp_rf_update"):
                                    vax1_fused_1 = Ts.axis.spatial(1024, ax1_fused_1)
                                    v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                    vax1_fused_0 = Ts.axis.reduce(4, ax1_fused_0)
                                    Ts.reads(Ared_temp_rf_local[vax1_fused_1, 0, 0], A[0, 0, vax1_fused_0 * 1024 + vax1_fused_1])
                                    Ts.writes(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                                    Ared_temp_rf_local[vax1_fused_1, 0, 0] = Ared_temp_rf_local[vax1_fused_1, 0, 0] + T.Cast("float32", A[0, 0, vax1_fused_0 * 1024 + vax1_fused_1]) * T.Cast("float32", A[0, 0, vax1_fused_0 * 1024 + vax1_fused_1])
                    for ax1_fused in range(T.int64(1)):
                        for ax0 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("Ared_temp"):
                                vax1_fused_1 = Ts.axis.reduce(1024, ax0)
                                v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                Ts.reads(Ared_temp_rf_local[vax1_fused_1, 0, 0])
                                Ts.writes(Ared_temp_shared[0, 0])
                                with Ts.init():
                                    Ared_temp_shared[0, 0] = T.float32(0)
                                Ared_temp_shared[0, 0] = Ared_temp_shared[0, 0] + Ared_temp_rf_local[vax1_fused_1, 0, 0]
                    for ax0_fused_0 in range(4):
                        for ax0_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("rms_norm"):
                                v0 = Ts.axis.spatial(4096, ax0_fused_0 * 1024 + ax0_fused_1)
                                Ts.reads(B[v0], A[0, 0, v0], Ared_temp_shared[0, 0])
                                Ts.writes(rms_norm[0, v0])
                                rms_norm[0, v0] = T.Cast("float16", T.Cast("float32", B[v0]) * (T.Cast("float32", A[0, 0, v0]) / T.sqrt(Ared_temp_shared[0, 0] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))
    # fmt: on
    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, After)


def test_spatial_inner_no_broadcasting():
    # fmt: off
    @I.ir_module
    class Module:
        @Ts.function
        def main(lv575: T.Tensor((1376, 4096), "uint32"), lv576: T.Tensor((344, 4096), "float16"), lv574: T.Tensor((1, 1, 11008), "float16"), lv570: T.Tensor((1, 1, 4096), "float16"), p_output0_intermediate: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"tirx.noalias": True})
            p_output0_intermediate_1 = Ts.sblock_alloc_tensor((11008, 4096), "float16")
            var_matmul_intermediate = Ts.sblock_alloc_tensor((1, 1, 4096), "float16")
            for i, j in T.grid(11008, 4096):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(lv575[v_i // 8, v_j], lv576[v_i // 32, v_j])
                    Ts.writes(p_output0_intermediate_1[v_i, v_j])
                    p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv575[v_i // 8, v_j], T.Cast("uint32", v_i % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[v_i // 32, v_j]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 11008):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(lv574[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
                    Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                    with Ts.init():
                        var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv574[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
            for ax0, ax1, ax2 in T.grid(1, 1, 4096):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(lv570[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                    Ts.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                    p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv570[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(lv575: T.Tensor((1376, 4096), "uint32"), lv576: T.Tensor((344, 4096), "float16"), lv574: T.Tensor((1, 1, 11008), "float16"), lv570: T.Tensor((1, 1, 4096), "float16"), p_output0_intermediate: T.Tensor((1, 1, 4096), "float16")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            var_matmul_intermediate_local = Ts.sblock_alloc_tensor((1, 1, 4096), "float16", scope="local")
            var_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((16, 1, 1, 4096), "float16", scope="local")
            for ax0_fused_0 in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for ax0_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_0_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("matmul_rf_init"):
                            vax1_0_fused_1 = Ts.axis.spatial(16, ax1_0_fused_1)
                            v0 = Ts.axis.spatial(4096, ax0_fused_0 * 16 + ax0_fused_1)
                            Ts.reads()
                            Ts.writes(var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                            var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] = T.float16(0)
                        for ax1_0_fused_0, ax1_1 in T.grid(86, 8):
                            with Ts.sblock("matmul_rf_update"):
                                vax1_0_fused_1 = Ts.axis.spatial(16, ax1_0_fused_1)
                                v0 = Ts.axis.spatial(4096, ax0_fused_0 * 16 + ax0_fused_1)
                                vax1_0_fused_0, vax1_1 = Ts.axis.remap("RR", [ax1_0_fused_0, ax1_1])
                                Ts.reads(var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0], lv574[0, 0, vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1], lv575[(vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) // 8, v0], lv576[(vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) // 32, v0])
                                Ts.writes(var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                                var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] = var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0] + lv574[0, 0, vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv575[(vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) // 8, v0], T.Cast("uint32", (vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[(vax1_0_fused_0 * 128 + vax1_0_fused_1 * 8 + vax1_1) // 32, v0])
                for ax1_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("matmul"):
                            vax1_0_fused_1 = Ts.axis.reduce(16, ax0)
                            v0 = Ts.axis.spatial(4096, ax0_fused_0 * 16 + ax1_fused)
                            Ts.reads(var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0])
                            Ts.writes(var_matmul_intermediate_local[0, 0, v0])
                            with Ts.init():
                                var_matmul_intermediate_local[0, 0, v0] = T.float16(0)
                            var_matmul_intermediate_local[0, 0, v0] = var_matmul_intermediate_local[0, 0, v0] + var_matmul_intermediate_rf_local[vax1_0_fused_1, 0, 0, v0]
                for ax0_fused_0_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_fused_1 in range(1):
                        with Ts.sblock("T_add"):
                            v0 = Ts.axis.spatial(4096, ax0_fused_0 * 16 + ax0_fused_0_1 + ax0_fused_1)
                            Ts.reads(lv570[0, 0, v0], var_matmul_intermediate_local[0, 0, v0])
                            Ts.writes(p_output0_intermediate[0, 0, v0])
                            p_output0_intermediate[0, 0, v0] = lv570[0, 0, v0] + var_matmul_intermediate_local[0, 0, v0]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_spatial_inner_broadcasting():
    # fmt: off
    @I.ir_module
    class Module:
        @Ts.function
        def main(A: T.Tensor((256, 256), "float32"), B: T.Tensor((256, 256), "float32")):
            T.func_attr({"tirx.noalias": True})
            temp_local = Ts.sblock_alloc_tensor((256,))
            for j in T.serial(256):
                for k in T.serial(256):
                    with Ts.sblock("sum"):
                        vj, vk = Ts.axis.remap("SR", [j, k])
                        Ts.reads(A[vk, vj])
                        Ts.writes(temp_local[vj])
                        with Ts.init():
                            temp_local[vj] = T.float32(0)
                        temp_local[vj] = temp_local[vj] + A[vk, vj]
            for i, j in T.grid(256, 256):
                with Ts.sblock("add"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Ts.reads(temp_local[vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] + temp_local[vj]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((256, 256), "float32"), B: T.Tensor((256, 256), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            temp_local_shared = Ts.sblock_alloc_tensor((256,), scope="shared")
            temp_local_rf_local = Ts.sblock_alloc_tensor((16, 256), scope="local")
            for ax0_fused_0 in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
                for ax0_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("sum_rf_init"):
                            vax1_fused_1 = Ts.axis.spatial(16, ax1_fused_1)
                            v0 = Ts.axis.spatial(256, ax0_fused_0 * 16 + ax0_fused_1)
                            Ts.reads()
                            Ts.writes(temp_local_rf_local[vax1_fused_1, v0])
                            temp_local_rf_local[vax1_fused_1, v0] = T.float32(0)
                        for ax1_fused_0, u in T.grid(16, 1):
                            with Ts.sblock("sum_rf_update"):
                                vax1_fused_1 = Ts.axis.spatial(16, ax1_fused_1)
                                v0 = Ts.axis.spatial(256, ax0_fused_0 * 16 + ax0_fused_1)
                                vax1_fused_0 = Ts.axis.reduce(16, ax1_fused_0)
                                Ts.reads(temp_local_rf_local[vax1_fused_1, v0], A[vax1_fused_0 * 16 + vax1_fused_1, v0])
                                Ts.writes(temp_local_rf_local[vax1_fused_1, v0])
                                temp_local_rf_local[vax1_fused_1, v0] = temp_local_rf_local[vax1_fused_1, v0] + A[vax1_fused_0 * 16 + vax1_fused_1, v0]
                for ax1_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("sum"):
                            vax1_fused_1 = Ts.axis.reduce(16, ax0)
                            v0 = Ts.axis.spatial(256, ax0_fused_0 * 16 + ax1_fused)
                            Ts.reads(temp_local_rf_local[vax1_fused_1, v0])
                            Ts.writes(temp_local_shared[v0])
                            with Ts.init():
                                temp_local_shared[v0] = T.float32(0)
                            temp_local_shared[v0] = temp_local_shared[v0] + temp_local_rf_local[vax1_fused_1, v0]
                for ax0_ax1_fused_0 in range(16):
                    for ax0_ax1_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                        for ax0_ax1_fused_2 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                            with Ts.sblock("add"):
                                v0 = Ts.axis.spatial(256, (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) // 16)
                                v1 = Ts.axis.spatial(256, ax0_fused_0 * 16 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 16 + ax0_ax1_fused_2) % 16)
                                Ts.reads(temp_local_shared[v1])
                                Ts.writes(B[v0, v1])
                                B[v0, v1] = A[v0, v1] + temp_local_shared[v1]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_reduction_inner_no_broadcasting():
    # fmt: off
    @I.ir_module
    class Module:
        @Ts.function
        def main(A: T.Tensor((256, 256), "float32"), B: T.Tensor((256,), "float32")):
            T.func_attr({"tirx.noalias": True})
            temp_local = Ts.sblock_alloc_tensor((256,))
            for i in T.serial(256):
                for k in T.serial(256):
                    with Ts.sblock("sum"):
                        vi, vk = Ts.axis.remap("SR", [i, k])
                        Ts.reads(A[vi, vk])
                        Ts.writes(temp_local[vi])
                        with Ts.init():
                            temp_local[vi] = T.float32(0)
                        temp_local[vi] = temp_local[vi] + A[vi, vk]
            for i in T.grid(256):
                with Ts.sblock("add"):
                    vi = Ts.axis.remap("S", [i])
                    Ts.reads(temp_local[vi])
                    Ts.writes(B[vi,])
                    B[vi] = temp_local[vi] + T.float32(1)

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((256, 256), "float32"), B: T.Tensor((256,), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            temp_local_local = Ts.sblock_alloc_tensor((256,), scope="local")
            temp_local_rf_local = Ts.sblock_alloc_tensor((256, 256), scope="local")
            for ax0_fused in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for ax1_fused_1 in T.parallel(256, annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                    with Ts.sblock("sum_rf_init"):
                        vax1_fused_1, v0 = Ts.axis.remap("SS", [ax1_fused_1, ax0_fused])
                        Ts.reads()
                        Ts.writes(temp_local_rf_local[vax1_fused_1, v0])
                        temp_local_rf_local[vax1_fused_1, v0] = T.float32(0)
                    for ax1_fused_0, u in T.grid(1, 1):
                        with Ts.sblock("sum_rf_update"):
                            vax1_fused_1, v0, vax1_fused_0 = Ts.axis.remap("SSR", [ax1_fused_1, ax0_fused, ax1_fused_0])
                            Ts.reads(temp_local_rf_local[vax1_fused_1, v0], A[v0, vax1_fused_0 * 256 + vax1_fused_1])
                            Ts.writes(temp_local_rf_local[vax1_fused_1, v0])
                            temp_local_rf_local[vax1_fused_1, v0] = temp_local_rf_local[vax1_fused_1, v0] + A[v0, vax1_fused_0 * 256 + vax1_fused_1]
                for ax1_fused in range(1):
                    for ax0 in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("sum"):
                            vax1_fused_1, v0 = Ts.axis.remap("RS", [ax0, ax0_fused])
                            Ts.reads(temp_local_rf_local[vax1_fused_1, v0])
                            Ts.writes(temp_local_local[v0])
                            with Ts.init():
                                temp_local_local[v0] = T.float32(0)
                            temp_local_local[v0] = temp_local_local[v0] + temp_local_rf_local[vax1_fused_1, v0]
                for ax0 in range(1):
                    with Ts.sblock("add"):
                        v0 = Ts.axis.spatial(256, ax0_fused + ax0)
                        Ts.reads(temp_local_local[v0])
                        Ts.writes(B[v0])
                        B[v0] = temp_local_local[v0] + T.float32(1)
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Module)  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_reduction_inner_no_broadcasting2():
    # fmt: off
    @I.ir_module
    class Module:
        @Ts.function
        def main(lv9: T.Tensor((2560, 320), "uint32"), lv10: T.Tensor((2560, 80), "float16"), lv1: T.Tensor((1, 2560), "float16"), p_output0_intermediate: T.Tensor((1, 2560), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            p_output0_intermediate_1 = Ts.sblock_alloc_tensor((2560, 2560), "float16")
            var_matmul_intermediate = Ts.sblock_alloc_tensor((1, 2560), "float16")
            for i, j in T.grid(2560, 2560):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(lv9[v_i, v_j // 8], lv10[v_i, v_j // 32])
                    Ts.writes(p_output0_intermediate_1[v_i, v_j])
                    p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv9[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv10[v_i, v_j // 32]
            for i0, i1, k in T.grid(1, 2560, 2560):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(lv1[v_i0, v_k], p_output0_intermediate_1[v_k, v_i1])
                    Ts.writes(var_matmul_intermediate[v_i0, v_i1])
                    with Ts.init():
                        var_matmul_intermediate[v_i0, v_i1] = T.float16(0)
                    var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv1[v_i0, v_k] * p_output0_intermediate_1[v_k, v_i1]
            for i0, i1 in T.grid(1, 2560):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(var_matmul_intermediate[v_i0, v_i1])
                    Ts.writes(p_output0_intermediate[v_i0, v_i1])
                    p_output0_intermediate[v_i0, v_i1] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1])

    @I.ir_module
    class Expected:
        @Ts.function
        def main(lv9: T.Tensor((2560, 320), "uint32"), lv10: T.Tensor((2560, 80), "float16"), lv1: T.Tensor((1, 2560), "float16"), p_output0_intermediate: T.Tensor((1, 2560), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            # with Ts.sblock("root"):
            var_matmul_intermediate_local = Ts.sblock_alloc_tensor((1, 2560), "float16", scope="local")
            var_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((16, 1, 2560), "float16", scope="local")
            for ax0_0_fused_0 in T.parallel(20, annotations={"thread_binding": "blockIdx.x"}):
                for ax0_0_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        for ax0_1_init in range(8):
                            with Ts.sblock("matmul_rf_init"):
                                vax1_fused_1 = Ts.axis.spatial(16, ax1_fused_1)
                                v0 = Ts.axis.spatial(2560, ax0_0_fused_0 * 128 + ax0_0_fused_1 * 8 + ax0_1_init)
                                Ts.reads()
                                Ts.writes(var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0])
                                var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0] = T.float16(0)
                        for ax1_fused_0, ax0_1 in T.grid(160, 8):
                            with Ts.sblock("matmul_rf_update"):
                                vax1_fused_1 = Ts.axis.spatial(16, ax1_fused_1)
                                v0 = Ts.axis.spatial(2560, ax0_0_fused_0 * 128 + ax0_0_fused_1 * 8 + ax0_1)
                                vax1_fused_0 = Ts.axis.reduce(160, ax1_fused_0)
                                Ts.reads(var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0], lv1[0, vax1_fused_0 * 16 + vax1_fused_1], lv9[vax1_fused_0 * 16 + vax1_fused_1, v0 // 8], lv10[vax1_fused_0 * 16 + vax1_fused_1, v0 // 32])
                                Ts.writes(var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0])
                                var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0] = var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0] + lv1[0, vax1_fused_0 * 16 + vax1_fused_1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv9[vax1_fused_0 * 16 + vax1_fused_1, v0 // 8], T.Cast("uint32", v0 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv10[vax1_fused_0 * 16 + vax1_fused_1, v0 // 32])
                for ax1_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_fused_1 in range(8):
                        for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                            with Ts.sblock("matmul"):
                                vax1_fused_1 = Ts.axis.reduce(16, ax0)
                                v0 = Ts.axis.spatial(2560, ax0_0_fused_0 * 128 + ax1_fused_0 * 8 + ax1_fused_1)
                                Ts.reads(var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0])
                                Ts.writes(var_matmul_intermediate_local[0, v0])
                                with Ts.init():
                                    var_matmul_intermediate_local[0, v0] = T.float16(0)
                                var_matmul_intermediate_local[0, v0] = var_matmul_intermediate_local[0, v0] + var_matmul_intermediate_rf_local[vax1_fused_1, 0, v0]
                for ax0_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_fused_1 in range(8):
                        with Ts.sblock("compute"):
                            v0 = Ts.axis.spatial(2560, ax0_0_fused_0 * 128 + ax0_fused_0 * 8 + ax0_fused_1)
                            Ts.reads(var_matmul_intermediate_local[0, v0])
                            Ts.writes(p_output0_intermediate[0, v0])
                            p_output0_intermediate[0, v0] = T.Cast("float32", var_matmul_intermediate_local[0, v0])
    # fmt: on

    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=Target("nvidia/geforce-rtx-3090-ti"))(
        Module
    )  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_reduction_inner_spatial_choose_perfect_factor():
    # fmt: off
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @Ts.function
        def main(A: T.Tensor((T.int64(1), T.int64(32), T.int64(1), n), 'float16'), B: T.Tensor((T.int64(1), T.int64(32), n, T.int64(100)), 'float16'), matmul: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(100)), "float16")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(100), n):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    Ts.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                    matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((T.int64(1), T.int64(32), T.int64(1), n), 'float16'), B: T.Tensor((T.int64(1), T.int64(32), n, T.int64(100)), 'float16'), matmul: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(100)), "float16")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            matmul_rf_local = Ts.sblock_alloc_tensor((T.int64(16), T.int64(1), T.int64(32), T.int64(1), T.int64(100)), "float16", scope="local")
            for ax0_ax1_fused_0 in T.parallel(T.int64(320), annotations={"thread_binding": "blockIdx.x"}):
                for ax0_ax1_fused_1 in T.parallel(T.int64(10), annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("matmul_rf_init"):
                            vax2_fused_1 = Ts.axis.spatial(T.int64(16), ax2_fused_1)
                            v0 = Ts.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(10) + ax0_ax1_fused_1) // T.int64(100))
                            v1 = Ts.axis.spatial(T.int64(100), (ax0_ax1_fused_0 * T.int64(10) + ax0_ax1_fused_1) % T.int64(100))
                            Ts.reads()
                            Ts.writes(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                            matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                        for ax2_fused_0, u in T.grid((n + T.int64(15)) // T.int64(16), 1):
                            with Ts.sblock("matmul_rf_update"):
                                vax2_fused_1 = Ts.axis.spatial(T.int64(16), ax2_fused_1)
                                v0 = Ts.axis.spatial(T.int64(32), (ax0_ax1_fused_0 * T.int64(10) + ax0_ax1_fused_1) // T.int64(100))
                                v1 = Ts.axis.spatial(T.int64(100), (ax0_ax1_fused_0 * T.int64(10) + ax0_ax1_fused_1) % T.int64(100))
                                vax2_fused_0 = Ts.axis.reduce((n + T.int64(15)) // T.int64(16), ax2_fused_0)
                                Ts.where(ax2_fused_0 * T.int64(16) + ax2_fused_1 < n)
                                Ts.reads(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1], A[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1], B[T.int64(0), v0, vax2_fused_0 * T.int64(16) + vax2_fused_1, v1])
                                Ts.writes(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                                matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] + A[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1] * B[T.int64(0), v0, vax2_fused_0 * T.int64(16) + vax2_fused_1, v1]
                for ax1_ax2_fused in T.parallel(T.int64(10), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("matmul"):
                            vax2_fused_1 = Ts.axis.reduce(T.int64(16), ax0)
                            v0 = Ts.axis.spatial(T.int64(32), ax0_ax1_fused_0 // T.int64(10))
                            v1 = Ts.axis.spatial(T.int64(100), ax0_ax1_fused_0 % T.int64(10) * T.int64(10) + ax1_ax2_fused)
                            Ts.reads(matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                            Ts.writes(matmul[T.int64(0), v0, T.int64(0), v1])
                            with Ts.init():
                                matmul[T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                            matmul[T.int64(0), v0, T.int64(0), v1] = matmul[T.int64(0), v0, T.int64(0), v1] + matmul_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1]
    # fmt: on

    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=Target("nvidia/geforce-rtx-3090-ti"))(
        Module
    )  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_reduction_inner_spatial_non_affine_write_back_falls_back():
    # The partial innermost tile leaves three div/mod components in the write-back binding.
    @I.ir_module
    class Before:
        @Ts.function
        def main(
            A: T.Tensor((2, 4, 20), "float32"),
            W: T.Tensor((3,), "float32"),
            C: T.Tensor((2, 2, 20), "float32"),
        ):
            for n, y, x, k in T.grid(2, 2, 20, 3):
                with Ts.sblock("conv"):
                    vn, vy, vx, vk = Ts.axis.remap("SSSR", [n, y, x, k])
                    Ts.reads(A[vn, vy + vk, vx], W[vk])
                    Ts.writes(C[vn, vy, vx])
                    with Ts.init():
                        C[vn, vy, vx] = T.float32(0)
                    C[vn, vy, vx] += A[vn, vy + vk, vx] * W[vk]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(
            A: T.Tensor((2, 4, 20), "float32"),
            W: T.Tensor((3,), "float32"),
            C: T.Tensor((2, 2, 20), "float32"),
        ):
            T.func_attr({"tirx.is_scheduled": True})
            for ax0_ax1_ax2_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for ax0_ax1_ax2_fused_1 in T.parallel(
                    1024, annotations={"thread_binding": "threadIdx.x"}
                ):
                    with Ts.sblock("conv_init"):
                        v0 = Ts.axis.spatial(
                            2, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) // 40
                        )
                        v1 = Ts.axis.spatial(
                            2, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 40 // 20
                        )
                        v2 = Ts.axis.spatial(
                            20, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 20
                        )
                        Ts.where(ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 < 80)
                        Ts.reads()
                        Ts.writes(C[v0, v1, v2])
                        C[v0, v1, v2] = T.float32(0)
                    for ax3 in range(3):
                        with Ts.sblock("conv_update"):
                            v0 = Ts.axis.spatial(
                                2, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) // 40
                            )
                            v1 = Ts.axis.spatial(
                                2,
                                (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 40 // 20,
                            )
                            v2 = Ts.axis.spatial(
                                20, (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1) % 20
                            )
                            v3 = Ts.axis.reduce(3, ax3)
                            Ts.where(ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 < 80)
                            Ts.reads(C[v0, v1, v2], A[v0, v1 + v3, v2], W[v3])
                            Ts.writes(C[v0, v1, v2])
                            C[v0, v1, v2] += A[v0, v1 + v3, v2] * W[v3]

    assert (
        dl.gpu.Reduction().apply(Before["main"], Target("nvidia/geforce-rtx-3090-ti"), False)
        is None
    )
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.Reduction(), dl.gpu.Fallback(), target=Target("nvidia/geforce-rtx-3090-ti")
    )(Before)
    assert_structural_equal(mod, Expected)


def test_reduction_inner_spatial_non_affine_without_mixed_access():
    # The failure depends on the spatial extents, not mixed spatial/reduction indexing.
    @I.ir_module
    class Before:
        @Ts.function
        def main(
            A: T.Tensor((2, 2, 3, 20), "float32"),
            C: T.Tensor((2, 2, 20), "float32"),
        ):
            for n, y, x, k in T.grid(2, 2, 20, 3):
                with Ts.sblock("sum"):
                    vn, vy, vx, vk = Ts.axis.remap("SSSR", [n, y, x, k])
                    Ts.reads(A[vn, vy, vk, vx])
                    Ts.writes(C[vn, vy, vx])
                    with Ts.init():
                        C[vn, vy, vx] = T.float32(0)
                    C[vn, vy, vx] += A[vn, vy, vk, vx]

    assert (
        dl.gpu.Reduction().apply(Before["main"], Target("nvidia/geforce-rtx-3090-ti"), False)
        is None
    )
    mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Fallback(),
        target=Target("nvidia/geforce-rtx-3090-ti"),
    )(Before)
    assert "tirx.is_scheduled" in mod["main"].attrs


def test_reduction_inner_spatial_reordered_access_declines():
    # `_normalize` fuses the spatial loops in dominant-read order: n, x, y.
    @I.ir_module
    class Before:
        @Ts.function
        def main(
            A: T.Tensor((2, 16, 3, 20), "float32"),
            C: T.Tensor((2, 20, 16), "float32"),
        ):
            for n, y, x, k in T.grid(2, 20, 16, 3):
                with Ts.sblock("sum"):
                    vn, vy, vx, vk = Ts.axis.remap("SSSR", [n, y, x, k])
                    Ts.reads(A[vn, vx, vk, vy])
                    Ts.writes(C[vn, vy, vx])
                    with Ts.init():
                        C[vn, vy, vx] = T.float32(0)
                    C[vn, vy, vx] += A[vn, vx, vk, vy]

    assert (
        dl.gpu.Reduction().apply(Before["main"], Target("nvidia/geforce-rtx-3090-ti"), False)
        is None
    )


def test_reduction_inner_spatial_affine_write_back_still_applies():
    # An exact innermost tile leaves only two div/mod components.
    @I.ir_module
    class Before:
        @Ts.function
        def main(
            A: T.Tensor((2, 4, 16), "float32"),
            W: T.Tensor((3,), "float32"),
            C: T.Tensor((2, 2, 16), "float32"),
        ):
            for n, y, x, k in T.grid(2, 2, 16, 3):
                with Ts.sblock("conv"):
                    vn, vy, vx, vk = Ts.axis.remap("SSSR", [n, y, x, k])
                    Ts.reads(A[vn, vy + vk, vx], W[vk])
                    Ts.writes(C[vn, vy, vx])
                    with Ts.init():
                        C[vn, vy, vx] = T.float32(0)
                    C[vn, vy, vx] += A[vn, vy + vk, vx] * W[vk]

    sch = dl.gpu.Reduction().apply(Before["main"], Target("nvidia/geforce-rtx-3090-ti"), False)
    assert sch is not None, "Reduction rule should still handle affine write-back blocks"


def test_reduction_inner_spatial_uses_normalized_split_extent():
    # x // 2 normalizes the x extent from 8 to 4, which must also be used for tx.
    @I.ir_module
    class Before:
        @Ts.function
        def main(
            A: T.Tensor((2, 3, 3, 4), "float32"),
            C: T.Tensor((2, 3, 8), "float32"),
        ):
            for n, y, x, k in T.grid(2, 3, 8, 3):
                with Ts.sblock("sum"):
                    vn, vy, vx, vk = Ts.axis.remap("SSSR", [n, y, x, k])
                    Ts.reads(A[vn, vy, vk, vx // 2])
                    Ts.writes(C[vn, vy, vx])
                    with Ts.init():
                        C[vn, vy, vx] = T.float32(0)
                    C[vn, vy, vx] += A[vn, vy, vk, vx // 2]

    sch = dl.gpu.Reduction().apply(Before["main"], Target("nvidia/geforce-rtx-3090-ti"), False)
    assert sch is not None, "Reduction should use the normalized split extent for tx"
    assert 'parallel(0, 4, annotations={"thread_binding": "threadIdx.x"})' in sch.mod.script()


def test_repeat_transpose_gemv():
    # fmt: off

    kv_seq_len = T.dynamic("kv_seq_len", "int64")

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def fused_relax_repeat_relax_permute_dims_relax_matmul1(lv716: T.Tensor((T.int64(1), kv_seq_len, T.int64(8), T.int64(128)), 'float16'), astype66: T.Tensor((T.int64(1), T.int64(32), T.int64(1), kv_seq_len), 'float16'), var_matmul_intermediate: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            var_T_repeat_intermediate = Ts.sblock_alloc_tensor((T.int64(1), kv_seq_len, T.int64(32), T.int64(128)), "float16")
            var_T_transpose_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), kv_seq_len, T.int64(128)), "float16")
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), kv_seq_len, T.int64(32), T.int64(128)):
                with Ts.sblock("T_repeat"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(lv716[v_ax0, v_ax1, v_ax2 // T.int64(4), v_ax3])
                    Ts.writes(var_T_repeat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    var_T_repeat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv716[v_ax0, v_ax1, v_ax2 // T.int64(4), v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), kv_seq_len, T.int64(128)):
                with Ts.sblock("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(var_T_repeat_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
                    Ts.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_repeat_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
            for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), kv_seq_len):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    Ts.reads(astype66[v_i0, v_i1, v_i2, v_k], var_T_transpose_intermediate[v_i0, v_i1, v_k, v_i3])
                    Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                    with Ts.init():
                        var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                    var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + astype66[v_i0, v_i1, v_i2, v_k] * var_T_transpose_intermediate[v_i0, v_i1, v_k, v_i3]
    kv_seq_len = T.dynamic("kv_seq_len", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def fused_relax_repeat_relax_permute_dims_relax_matmul1(lv716: T.Tensor((T.int64(1), kv_seq_len, T.int64(8), T.int64(128)), 'float16'), astype66: T.Tensor((T.int64(1), T.int64(32), T.int64(1), kv_seq_len), 'float16'), var_matmul_intermediate: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            var_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((T.int64(16), T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16", scope="local")
            for ax0_0_ax1_fused_0 in T.parallel(T.int64(64), annotations={"thread_binding": "blockIdx.x"}):
                for ax0_0_ax1_fused_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                        for ax0_1_init in range(T.int64(4)):
                            with Ts.sblock("matmul_rf_init"):
                                vax2_fused_1 = Ts.axis.spatial(T.int64(16), ax2_fused_1)
                                v0 = Ts.axis.spatial(T.int64(32), (ax0_0_ax1_fused_0 * T.int64(16) + ax0_0_ax1_fused_1) // T.int64(128) * T.int64(4) + ax0_1_init)
                                v1 = Ts.axis.spatial(T.int64(128), (ax0_0_ax1_fused_0 * T.int64(16) + ax0_0_ax1_fused_1) % T.int64(128))
                                Ts.reads()
                                Ts.writes(var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                                var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                        for ax2_fused_0, ax0_1 in T.grid((kv_seq_len + T.int64(15)) // T.int64(16), T.int64(4)):
                            with Ts.sblock("matmul_rf_update"):
                                vax2_fused_1 = Ts.axis.spatial(T.int64(16), ax2_fused_1)
                                v0 = Ts.axis.spatial(T.int64(32), (ax0_0_ax1_fused_0 * T.int64(16) + ax0_0_ax1_fused_1) // T.int64(128) * T.int64(4) + ax0_1)
                                v1 = Ts.axis.spatial(T.int64(128), (ax0_0_ax1_fused_0 * T.int64(16) + ax0_0_ax1_fused_1) % T.int64(128))
                                vax2_fused_0 = Ts.axis.reduce((kv_seq_len + T.int64(15)) // T.int64(16), ax2_fused_0)
                                Ts.where(ax2_fused_0 * T.int64(16) + ax2_fused_1 < kv_seq_len)
                                Ts.reads(var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1], astype66[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1], lv716[T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1, v0 // T.int64(4), v1])
                                Ts.writes(var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                                var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] = var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1] + astype66[T.int64(0), v0, T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1] * lv716[T.int64(0), vax2_fused_0 * T.int64(16) + vax2_fused_1, v0 // T.int64(4), v1]
                for ax1_0_ax2_fused in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_1 in range(T.int64(4)):
                        for ax0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                            with Ts.sblock("matmul"):
                                vax2_fused_1 = Ts.axis.reduce(T.int64(16), ax0)
                                v0 = Ts.axis.spatial(T.int64(32), ax0_0_ax1_fused_0 // T.int64(8) * T.int64(4) + ax1_1)
                                v1 = Ts.axis.spatial(T.int64(128), ax0_0_ax1_fused_0 % T.int64(8) * T.int64(16) + ax1_0_ax2_fused)
                                Ts.reads(var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1])
                                Ts.writes(var_matmul_intermediate[T.int64(0), v0, T.int64(0), v1])
                                with Ts.init():
                                    var_matmul_intermediate[T.int64(0), v0, T.int64(0), v1] = T.float16(0)
                                var_matmul_intermediate[T.int64(0), v0, T.int64(0), v1] = var_matmul_intermediate[T.int64(0), v0, T.int64(0), v1] + var_matmul_intermediate_rf_local[vax2_fused_1, T.int64(0), v0, T.int64(0), v1]
    # fmt: on

    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=Target("nvidia/geforce-rtx-3090-ti"))(
        Before
    )  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_gemv_dyn_shape_epilogue():
    vocab_size = T.dynamic("vocab_size", "int64")

    @I.ir_module
    class Module:
        @Ts.function(private=True)
        def main(
            A: T.Tensor((T.int64(4096), vocab_size), "float16"),
            B: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
            C: T.Tensor((T.int64(1), T.int64(1), vocab_size), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            C_temp = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), vocab_size), "float16")
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), vocab_size, T.int64(4096)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(B[v_i0, v_i1, v_k], A[v_k, v_i2])
                    Ts.writes(C_temp[v_i0, v_i1, v_i2])
                    with Ts.init():
                        C_temp[v_i0, v_i1, v_i2] = T.float16(0)
                    C_temp[v_i0, v_i1, v_i2] = (
                        C_temp[v_i0, v_i1, v_i2] + B[v_i0, v_i1, v_k] * A[v_k, v_i2]
                    )
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), vocab_size):
                with Ts.sblock("epilogue"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(C_temp[v_i0, v_i1, v_i2])
                    Ts.writes(C[v_i0, v_i1, v_i2])
                    C[v_i0, v_i1, v_i2] = T.Cast("float32", C_temp[v_i0, v_i1, v_i2])

    # fmt: off
    vocab_size = T.dynamic("vocab_size", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def main(A: T.Tensor((T.int64(4096), vocab_size), 'float16'), B: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"), C: T.Tensor((T.int64(1), T.int64(1), vocab_size), "float32")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

            # with Ts.sblock("root"):
            C_temp_local = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), vocab_size), "float16", scope="local")
            C_temp_rf_local = Ts.sblock_alloc_tensor((T.int64(16), T.int64(1), T.int64(1), vocab_size), "float16", scope="local")
            for ax0_fused_0 in T.parallel(vocab_size, annotations={"thread_binding": "blockIdx.x"}):
                for ax0_fused_1 in T.parallel(T.int64(1), annotations={"thread_binding": "threadIdx.x"}):
                    for ax1_fused_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("matmul_rf_init"):
                            vax1_fused_1 = Ts.axis.spatial(T.int64(16), ax1_fused_1)
                            v0 = Ts.axis.spatial(vocab_size, ax0_fused_0 + ax0_fused_1)
                            Ts.reads()
                            Ts.writes(C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = T.float16(0)
                        for ax1_fused_0, u in T.grid(T.int64(256), 1):
                            with Ts.sblock("matmul_rf_update"):
                                vax1_fused_1 = Ts.axis.spatial(T.int64(16), ax1_fused_1)
                                v0 = Ts.axis.spatial(vocab_size, ax0_fused_0 + ax0_fused_1)
                                vax1_fused_0 = Ts.axis.reduce(T.int64(256), ax1_fused_0)
                                Ts.reads(C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0], B[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1], A[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0])
                                Ts.writes(C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                                C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] = C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0] + B[T.int64(0), T.int64(0), vax1_fused_0 * T.int64(16) + vax1_fused_1] * A[vax1_fused_0 * T.int64(16) + vax1_fused_1, v0]
                for ax1_fused in T.parallel(T.int64(1), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("matmul"):
                            vax1_fused_1, v0 = Ts.axis.remap("RS", [ax0, ax0_fused_0])
                            Ts.reads(C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0])
                            Ts.writes(C_temp_local[T.int64(0), T.int64(0), v0])
                            with Ts.init():
                                C_temp_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                            C_temp_local[T.int64(0), T.int64(0), v0] = C_temp_local[T.int64(0), T.int64(0), v0] + C_temp_rf_local[vax1_fused_1, T.int64(0), T.int64(0), v0]
                for ax0_fused_0_1 in T.parallel(T.int64(1), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0_fused_1 in range(T.int64(1)):
                        with Ts.sblock("epilogue"):
                            v0 = Ts.axis.spatial(vocab_size, ax0_fused_0)
                            Ts.reads(C_temp_local[T.int64(0), T.int64(0), v0])
                            Ts.writes(C[T.int64(0), T.int64(0), v0])
                            C[T.int64(0), T.int64(0), v0] = T.Cast("float32", C_temp_local[T.int64(0), T.int64(0), v0])
    # fmt: on

    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=Target("nvidia/geforce-rtx-3090-ti"))(
        Module
    )  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_gemv_output_one_element():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((T.int64(1), T.int64(2048)), "float16"), weight: T.Tensor((T.int64(1), T.int64(2048)), "float16"), out: T.Tensor((T.int64(1), T.int64(1)), "float16")):
            T.func_attr({"tirx.noalias": True})
            NT_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1)), "float16")
            for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(2048)):
                with Ts.sblock("NT_matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    with Ts.init():
                        NT_matmul_intermediate[v_i0, v_i1] = T.float16(0)
                    NT_matmul_intermediate[v_i0, v_i1] = NT_matmul_intermediate[v_i0, v_i1] + A[v_i0, v_k] * weight[v_i1, v_k]
            for i0, i1 in T.grid(T.int64(1), T.int64(1)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    out[v_i0, v_i1] = T.sigmoid(NT_matmul_intermediate[v_i0, v_i1])

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def main(A: T.Tensor((T.int64(1), T.int64(2048)), "float16"), weight: T.Tensor((T.int64(1), T.int64(2048)), "float16"), out: T.Tensor((T.int64(1), T.int64(1)), "float16")):
            T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
            NT_matmul_intermediate_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1)), "float16", scope="shared")
            NT_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((T.int64(1024), T.int64(1), T.int64(1)), "float16", scope="local")
            for ax0_fused in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for ax1_fused_1 in T.parallel(T.int64(1024), annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                    with Ts.sblock("NT_matmul_rf_init"):
                        vax1_fused_1 = Ts.axis.spatial(T.int64(1024), ax1_fused_1)
                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                        NT_matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0)] = T.float16(0)
                    for ax1_fused_0, u in T.grid(T.int64(2), 1):
                        with Ts.sblock("NT_matmul_rf_update"):
                            vax1_fused_1 = Ts.axis.spatial(T.int64(1024), ax1_fused_1)
                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                            vax1_fused_0 = Ts.axis.reduce(T.int64(2), ax1_fused_0)
                            NT_matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0)] = NT_matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0)] + A[T.int64(0), vax1_fused_0 * T.int64(1024) + vax1_fused_1] * weight[T.int64(0), vax1_fused_0 * T.int64(1024) + vax1_fused_1]
                for ax1_fused in range(T.int64(1)):
                    for ax0 in T.parallel(T.int64(1024), annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("NT_matmul"):
                            vax1_fused_1 = Ts.axis.reduce(T.int64(1024), ax0)
                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                            with Ts.init():
                                NT_matmul_intermediate_shared[T.int64(0), T.int64(0)] = T.float16(0)
                            NT_matmul_intermediate_shared[T.int64(0), T.int64(0)] = NT_matmul_intermediate_shared[T.int64(0), T.int64(0)] + NT_matmul_intermediate_rf_local[vax1_fused_1, T.int64(0), T.int64(0)]
                for ax0_fused_0 in range(T.int64(1)):
                    for ax0_fused_1 in T.parallel(T.int64(1024), annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("compute"):
                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                            Ts.where(ax0_fused_0 * T.int64(1024) + ax0_fused_1 < T.int64(1))
                            out[T.int64(0), T.int64(0)] = T.sigmoid(NT_matmul_intermediate_shared[T.int64(0), T.int64(0)])
    # fmt: on

    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=Target("nvidia/geforce-rtx-3090-ti"))(
        Before
    )  # pylint: disable=not-callable
    assert_structural_equal(mod, Expected)


def test_no_reduction_loop_check():
    # The normalized prime func will not contain a reduction loop since its extent is one.
    # This checks that the Reduction schedule is correctly not applied in this case
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def matmul(lv43: T.Tensor((T.int64(1), T.int64(32), T.int64(1)), "float16"), lv44: T.Tensor((T.int64(1), T.int64(1), T.int64(1)), "float16"), matmul: T.Tensor((T.int64(1), T.int64(32), T.int64(1)), "float16")):
            T.func_attr({"op_pattern": 4, "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(1)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                    Ts.reads(lv43[v_i0, v_i1, v_k], lv44[v_i0, v_k, v_i2])
                    Ts.writes(matmul[v_i0, v_i1, v_i2])
                    with Ts.init():
                        matmul[v_i0, v_i1, v_i2] = T.float16(0.0)
                    matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + lv43[v_i0, v_i1, v_k] * lv44[v_i0, v_k, v_i2]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), target=target)(Before)  # pylint: disable=not-callable
    assert_structural_equal(mod, Before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
