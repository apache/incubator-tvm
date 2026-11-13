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
# pylint: disable=missing-docstring
# ruff: noqa: E501
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def test_matmul():
    # fmt: off
    m = T.dynamic("m", "int64")

    @Ts.function(private=True)
    def before(inp0: T.Tensor((T.int64(1), m, T.int64(4096)), "float32"), inp1: T.Tensor((T.int64(4096), T.int64(4096)), "float32"), matmul: T.Tensor((T.int64(1), m, T.int64(4096)), "float32")):

        for i0, i1, i2, k in T.grid(T.int64(1), m, T.int64(4096), T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                with Ts.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + inp0[v_i0, v_i1, v_k] * inp1[v_k, v_i2]

    m = T.dynamic("m", "int64")

    @Ts.function(private=True)
    def expected(inp0: T.Tensor((T.int64(1), m, T.int64(4096)), "float32"), inp1: T.Tensor((T.int64(4096), T.int64(4096)), "float32"), matmul: T.Tensor((T.int64(1), m, T.int64(4096)), "float32")):
        T.func_attr({"tirx.is_scheduled": True})

        # with Ts.sblock("root"):
        matmul_reindex_pad_local = Ts.sblock_alloc_tensor((T.int64(1), (m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="local")
        inp0_reindex_pad_shared = Ts.sblock_alloc_tensor((T.int64(1), (m + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="shared")
        inp1_reindex_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4096), T.int64(4096)), scope="shared")
        for ax0_ax2_0_fused in T.parallel(T.int64(64), annotations={"thread_binding": "blockIdx.y"}):
            for ax1_0 in T.parallel((m + T.int64(31)) // T.int64(32), annotations={"thread_binding": "blockIdx.x"}):
                for ax2_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.y"}):
                    for ax1_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.x"}):
                        for ax2_2 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                            for ax1_2 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with Ts.sblock("matmul_init"):
                                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = Ts.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            Ts.reads()
                                            Ts.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                            matmul_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0 in range(T.int64(256)):
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("inp0_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(inp0[v0, v1, v2])
                                                        Ts.writes(inp0_reindex_pad_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        inp0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("inp1_reindex_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(inp1[v2, v1])
                                                        Ts.writes(inp1_reindex_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        inp1_reindex_shared[v0, v1, v2] = inp1[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with Ts.sblock("matmul_update"):
                                                v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = Ts.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = Ts.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                                Ts.reads(matmul_reindex_pad_local[T.int64(0), v1, v2], inp0_reindex_pad_shared[T.int64(0), v1, v3], inp1_reindex_shared[T.int64(0), v2, v3])
                                                Ts.writes(matmul_reindex_pad_local[T.int64(0), v1, v2])
                                                matmul_reindex_pad_local[T.int64(0), v1, v2] = matmul_reindex_pad_local[T.int64(0), v1, v2] + inp0_reindex_pad_shared[T.int64(0), v1, v3] * inp1_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with Ts.sblock("matmul_reindex_pad_local"):
                                            v0 = Ts.axis.spatial(T.int64(1), ax0)
                                            v1 = Ts.axis.spatial((m + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            Ts.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < m)
                                            Ts.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            Ts.writes(matmul[T.int64(0), v1, v2])
                                            matmul[T.int64(0), v1, v2] = matmul_reindex_pad_local[v0, v1, v2]
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), target=Target("nvidia/geforce-gtx-1080-ti"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_matmul_int32():
    # fmt: off
    m = T.dynamic("m", "int32")

    @Ts.function(private=True)
    def func(inp0: T.Tensor((1, m, 4096), "float32"), inp1: T.Tensor((4096, 4096), "float32"), matmul: T.Tensor((1, m, 4096), "float32")):

        for i0, i1, i2, k in T.grid(1, m, 4096, 4096):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                with Ts.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + inp0[v_i0, v_i1, v_k] * inp1[v_k, v_i2]

    m = T.dynamic("m", "int32")

    @Ts.function(private=True)
    def expected(inp0: T.Tensor((1, m, 4096), "float32"), inp1: T.Tensor((4096, 4096), "float32"), matmul: T.Tensor((1, m, 4096), "float32")):
        T.func_attr({"tirx.is_scheduled": True})

        # with Ts.sblock("root"):
        matmul_reindex_pad_local = Ts.sblock_alloc_tensor((1, (m + 31) // 32 * 32, 4096), scope="local")
        inp0_reindex_pad_shared = Ts.sblock_alloc_tensor((1, (m + 31) // 32 * 32, 4096), scope="shared")
        inp1_reindex_shared = Ts.sblock_alloc_tensor((1, 4096, 4096), scope="shared")
        for ax0_ax2_0_fused in T.parallel(64, annotations={"thread_binding": "blockIdx.y"}):
            for ax1_0 in T.parallel((m + 31) // 32, annotations={"thread_binding": "blockIdx.x"}):
                for ax2_1 in T.parallel(1, annotations={"thread_binding": "vthread.y"}):
                    for ax1_1 in T.parallel(1, annotations={"thread_binding": "vthread.x"}):
                        for ax2_2 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                            for ax1_2 in T.parallel(8, annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(4, 2):
                                    for ax2_3_1_init in T.vectorized(2):
                                        with Ts.sblock("matmul_init"):
                                            v0 = Ts.axis.spatial(1, 0)
                                            v1 = Ts.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3_init)
                                            v2 = Ts.axis.spatial(4096, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0_init * 2 + ax2_3_1_init)
                                            Ts.reads()
                                            Ts.writes(matmul_reindex_pad_local[0, v1, v2])
                                            matmul_reindex_pad_local[0, v1, v2] = T.float32(0)
                                for ax3_0 in range(256):
                                    for ax0_ax1_ax2_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(2):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with Ts.sblock("inp0_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(1, 0)
                                                        v1 = Ts.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = Ts.axis.spatial(4096, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 32 + ax0_ax1_ax2_fused_1 * 4 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        Ts.reads(inp0[v0, v1, v2])
                                                        Ts.writes(inp0_reindex_pad_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        inp0_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
                                    for ax0_ax1_ax2_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(4):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(2):
                                                    with Ts.sblock("inp1_reindex_shared"):
                                                        v0 = Ts.axis.spatial(1, 0)
                                                        v1 = Ts.axis.spatial(4096, ax0_ax2_0_fused * 64 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) // 16)
                                                        v2 = Ts.axis.spatial(4096, ax3_0 * 16 + (ax0_ax1_ax2_fused_0 * 64 + ax0_ax1_ax2_fused_1 * 8 + ax0_ax1_ax2_fused_2 * 2 + ax0_ax1_ax2_fused_3) % 16)
                                                        Ts.reads(inp1[v2, v1])
                                                        Ts.writes(inp1_reindex_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        inp1_reindex_shared[v0, v1, v2] = inp1[v2, v1]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(16, 4, 2):
                                        for ax2_3_1 in T.vectorized(2):
                                            with Ts.sblock("matmul_update"):
                                                v0 = Ts.axis.spatial(1, 0)
                                                v1 = Ts.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_1 * 32 + ax1_2 * 4 + ax1_3)
                                                v2 = Ts.axis.spatial(4096, ax0_ax2_0_fused * 64 + ax2_1 * 64 + ax2_2 * 4 + ax2_3_0 * 2 + ax2_3_1)
                                                v3 = Ts.axis.reduce(4096, ax3_0 * 16 + ax3_1)
                                                Ts.reads(matmul_reindex_pad_local[0, v1, v2], inp0_reindex_pad_shared[0, v1, v3], inp1_reindex_shared[0, v2, v3])
                                                Ts.writes(matmul_reindex_pad_local[0, v1, v2])
                                                matmul_reindex_pad_local[0, v1, v2] = matmul_reindex_pad_local[0, v1, v2] + inp0_reindex_pad_shared[0, v1, v3] * inp1_reindex_shared[0, v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(1, 4, 2):
                                    for ax2_1_1 in T.vectorized(2):
                                        with Ts.sblock("matmul_reindex_pad_local"):
                                            v0 = Ts.axis.spatial(1, ax0)
                                            v1 = Ts.axis.spatial((m + 31) // 32 * 32, ax1_0 * 32 + ax1_2 * 4 + ax1)
                                            v2 = Ts.axis.spatial(4096, ax0_ax2_0_fused * 64 + ax2_2 * 4 + ax2_0 * 2 + ax2_1_1)
                                            Ts.where(ax1_0 * 32 + ax1_2 * 4 + ax1 < m)
                                            Ts.reads(matmul_reindex_pad_local[v0, v1, v2])
                                            Ts.writes(matmul[0, v1, v2])
                                            matmul[0, v1, v2] = matmul_reindex_pad_local[v0, v1, v2]
    # fmt: on

    mod = tvm_tirx.IRModule({"main": func})
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), target=Target("nvidia/geforce-gtx-1080-ti"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_fused_matmul():
    # fmt: off
    @Ts.function(private=True)
    def before(W: T.Tensor((T.int64(512), T.int64(4096)), "uint32"), S: T.Tensor((T.int64(128), T.int64(4096)), "uint32"), A: T.Tensor((T.int64(1), T.int64(32), T.int64(4096)), "float32"), C: T.Tensor((T.int64(1), T.int64(32), T.int64(4096)), "float32"), Out: T.Tensor((T.int64(1), T.int64(32), T.int64(4096)), "float32")):
        var_decode_intermediate = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(4096)))
        var_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), T.int64(4096)))
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(W[v_i // T.int64(8), v_j], S[v_i // T.int64(32), v_j])
                Ts.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(W[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret( T.shift_left(T.bitwise_and(S[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16)), ty="float32") + T.reinterpret( T.shift_left(T.bitwise_and(T.shift_right(S[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)), ty="float32")
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(4096), T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(A[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(4096)):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(C[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                Ts.writes(Out[v_ax0, v_ax1, v_ax2])
                Out[v_ax0, v_ax1, v_ax2] = C[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @Ts.function(private=True)
    def expected(W: T.Tensor((T.int64(512), T.int64(4096)), "uint32"), S: T.Tensor((T.int64(128), T.int64(4096)), "uint32"), A: T.Tensor((T.int64(1), T.int64(32), T.int64(4096)), "float32"), C: T.Tensor((T.int64(1), T.int64(32), T.int64(4096)), "float32"), Out: T.Tensor((T.int64(1), T.int64(32), T.int64(4096)), "float32")):
        T.func_attr({"tirx.is_scheduled": True})
        # with Ts.sblock("root"):
        var_matmul_intermediate_reindex_local = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), T.int64(4096)), scope="local")
        A_reindex_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32), T.int64(4096)), scope="shared")
        var_decode_intermediate_reindex_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4096), T.int64(4096)), scope="shared")
        for ax0_ax2_0_fused in T.parallel(T.int64(64), annotations={"thread_binding": "blockIdx.y"}):
            for ax1_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for ax2_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.y"}):
                    for ax1_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.x"}):
                        for ax2_2 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                            for ax1_2 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with Ts.sblock("matmul_init"):
                                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = Ts.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            Ts.reads()
                                            Ts.writes(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2])
                                            var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0 in range(T.int64(256)):
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("A_reindex_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial(T.int64(32), (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(A[v0, v1, v2])
                                                        Ts.writes(A_reindex_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        A_reindex_shared[v0, v1, v2] = A[v0, v1, v2]
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("var_decode_intermediate_reindex_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(W[v2 // T.int64(8), v1], S[v2 // T.int64(32), v1])
                                                        Ts.writes(var_decode_intermediate_reindex_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        var_decode_intermediate_reindex_shared[v0, v1, v2] = T.Cast("float32", T.bitwise_and(T.shift_right(W[v2 // T.int64(8), v1], T.Cast("uint32", v2 % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret( T.shift_left(T.bitwise_and(S[v2 // T.int64(32), v1], T.uint32(65535)), T.uint32(16)), ty="float32") + T.reinterpret( T.shift_left(T.bitwise_and(T.shift_right(S[v2 // T.int64(32), v1], T.uint32(16)), T.uint32(65535)), T.uint32(16)), ty="float32")
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with Ts.sblock("matmul_update"):
                                                v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = Ts.axis.spatial(T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = Ts.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                                Ts.reads(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2], A_reindex_shared[T.int64(0), v1, v3], var_decode_intermediate_reindex_shared[T.int64(0), v2, v3])
                                                Ts.writes(var_matmul_intermediate_reindex_local[T.int64(0), v1, v2])
                                                var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] = var_matmul_intermediate_reindex_local[T.int64(0), v1, v2] + A_reindex_shared[T.int64(0), v1, v3] * var_decode_intermediate_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with Ts.sblock("var_matmul_intermediate_reindex_local"):
                                            v0 = Ts.axis.spatial(T.int64(1), ax0)
                                            v1 = Ts.axis.spatial(T.int64(32), ax1_2 * T.int64(4) + ax1)
                                            v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            Ts.reads(C[T.int64(0), v1, v2], var_matmul_intermediate_reindex_local[v0, v1, v2])
                                            Ts.writes(Out[T.int64(0), v1, v2])
                                            Out[T.int64(0), v1, v2] = C[T.int64(0), v1, v2] + var_matmul_intermediate_reindex_local[v0, v1, v2]

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), target=Target("nvidia/geforce-gtx-1080-ti"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_skip_gemv():
    # fmt: off
    @Ts.function(private=True)
    def before(W: T.Tensor((T.int64(512), T.int64(4096)), "uint32"), S: T.Tensor((T.int64(128), T.int64(4096)), "uint32"), A: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float32"), C: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float32"), Out: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float32")):
        T.func_attr({"tirx.noalias": True})
        var_decode_intermediate = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(4096)))
        var_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(4096)))
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(W[v_i // T.int64(8), v_j], S[v_i // T.int64(32), v_j])
                Ts.writes(var_decode_intermediate[v_i, v_j])
                var_decode_intermediate[v_i, v_j] = T.Cast("float32", T.bitwise_and(T.shift_right(W[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) * T.reinterpret( T.shift_left(T.bitwise_and(S[v_i // T.int64(32), v_j], T.uint32(65535)), T.uint32(16)), ty="float32") + T.reinterpret( T.shift_left(T.bitwise_and(T.shift_right(S[v_i // T.int64(32), v_j], T.uint32(16)), T.uint32(65535)), T.uint32(16)), ty="float32")
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(A[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
                Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(C[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                Ts.writes(Out[v_ax0, v_ax1, v_ax2])
                Out[v_ax0, v_ax1, v_ax2] = C[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    # fmt: on

    expected = before

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), target=Target("nvidia/geforce-gtx-1080-ti"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_output_fp32():
    # fmt: off
    n = T.dynamic("n", "int64")

    @Ts.function(private=True)
    def before(lv13: T.Tensor((T.int64(4096), T.int64(512)), "uint32"), lv14: T.Tensor((T.int64(4096), T.int64(128)), "float16"), lv48: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16'), lv13_1: T.Tensor((T.int64(4096),), "float16"), lv3: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16'), p_output0_intermediate: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16')):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        p_output0_intermediate_1 = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(4096)), "float16")
        var_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), n, T.int64(4096)))
        var_compute_intermediate = Ts.sblock_alloc_tensor((T.int64(4096),))
        var_T_add_intermediate = Ts.sblock_alloc_tensor((T.int64(1), n, T.int64(4096)))
        var_compute_intermediate_1 = Ts.sblock_alloc_tensor((T.int64(1), n, T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv13[v_i, v_j // T.int64(8)], lv14[v_i, v_j // T.int64(32)])
                Ts.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv13[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv14[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv48[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
                Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + T.Cast("float32", lv48[v_i0, v_i1, v_k]) * T.Cast("float32", p_output0_intermediate_1[v_k, v_i2])
        for i0 in range(T.int64(4096)):
            with Ts.sblock("compute"):
                v_i0 = Ts.axis.spatial(T.int64(4096), i0)
                Ts.reads(lv13_1[v_i0])
                Ts.writes(var_compute_intermediate[v_i0])
                var_compute_intermediate[v_i0] = T.Cast("float32", lv13_1[v_i0])
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], var_compute_intermediate[v_ax2])
                Ts.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + var_compute_intermediate[v_ax2]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(4096)):
            with Ts.sblock("compute_1"):
                v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(var_T_add_intermediate[v_i0, v_i1, v_i2])
                Ts.writes(var_compute_intermediate_1[v_i0, v_i1, v_i2])
                var_compute_intermediate_1[v_i0, v_i1, v_i2] = T.Cast("float16", var_T_add_intermediate[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
            with Ts.sblock("T_add_1"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(var_compute_intermediate_1[v_ax0, v_ax1, v_ax2], lv3[v_ax0, v_ax1, v_ax2])
                Ts.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = var_compute_intermediate_1[v_ax0, v_ax1, v_ax2] + lv3[v_ax0, v_ax1, v_ax2]

    n = T.dynamic("n", "int64")

    @Ts.function(private=True)
    def expected(lv13: T.Tensor((T.int64(4096), T.int64(512)), "uint32"), lv14: T.Tensor((T.int64(4096), T.int64(128)), "float16"), lv48: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16'), lv13_1: T.Tensor((T.int64(4096),), "float16"), lv3: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16'), p_output0_intermediate: T.Tensor((T.int64(1), n, T.int64(4096)), 'float16')):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        var_matmul_intermediate_reindex_pad_local = Ts.sblock_alloc_tensor((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), scope="local")
        lv48_reindex_pad_shared = Ts.sblock_alloc_tensor((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(4096)), "float16", scope="shared")
        p_output0_intermediate_1_reindex_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(4096), T.int64(4096)), "float16", scope="shared")
        for ax0_ax2_0_fused in T.parallel(T.int64(64), annotations={"thread_binding": "blockIdx.y"}):
            for ax1_0 in T.parallel((n + T.int64(31)) // T.int64(32), annotations={"thread_binding": "blockIdx.x"}):
                for ax2_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.y"}):
                    for ax1_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.x"}):
                        for ax2_2 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                            for ax1_2 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with Ts.sblock("matmul_init"):
                                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = Ts.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            Ts.reads()
                                            Ts.writes(var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                            var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = T.float32(0)
                                for ax3_0 in range(T.int64(256)):
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("lv48_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(lv48[v0, v1, v2])
                                                        Ts.writes(lv48_reindex_pad_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        lv48_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < n, lv48[v0, v1, v2], T.float16(0))
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("p_output0_intermediate_1_reindex_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(4096), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(lv13[v2, v1 // T.int64(8)], lv14[v2, v1 // T.int64(32)])
                                                        Ts.writes(p_output0_intermediate_1_reindex_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        p_output0_intermediate_1_reindex_shared[v0, v1, v2] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv13[v2, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv14[v2, v1 // T.int64(32)]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with Ts.sblock("matmul_update"):
                                                v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = Ts.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = Ts.axis.reduce(T.int64(4096), ax3_0 * T.int64(16) + ax3_1)
                                                Ts.reads(var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2], lv48_reindex_pad_shared[T.int64(0), v1, v3], p_output0_intermediate_1_reindex_shared[T.int64(0), v2, v3])
                                                Ts.writes(var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                                var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = var_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] + T.Cast("float32", lv48_reindex_pad_shared[T.int64(0), v1, v3]) * T.Cast("float32", p_output0_intermediate_1_reindex_shared[T.int64(0), v2, v3])
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with Ts.sblock("var_matmul_intermediate_reindex_pad_local"):
                                            v0 = Ts.axis.spatial(T.int64(1), ax0)
                                            v1 = Ts.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = Ts.axis.spatial(T.int64(4096), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            Ts.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < n)
                                            Ts.reads(var_matmul_intermediate_reindex_pad_local[v0, v1, v2], lv13_1[v2], lv3[T.int64(0), v1, v2])
                                            Ts.writes(p_output0_intermediate[T.int64(0), v1, v2])
                                            p_output0_intermediate[T.int64(0), v1, v2] = T.Cast("float16", var_matmul_intermediate_reindex_pad_local[v0, v1, v2] + T.Cast("float32", lv13_1[v2])) + lv3[T.int64(0), v1, v2]

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), target=Target("nvidia/geforce-gtx-1080-ti"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_inline_consumer_chain():
    # fmt: off
    n = T.dynamic("n", "int64")

    @Ts.function(private=True)
    def before(lv26: T.Tensor((n, T.int64(2048)), 'float16'), lv9: T.Tensor((T.int64(2048), T.int64(2048)), "float16"), lv52: T.Tensor((T.int64(1), n, T.int64(2048)), "float32"), var_T_multiply_intermediate: T.Tensor((n, T.int64(2048)), 'float16')):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        var_NT_matmul_intermediate = Ts.sblock_alloc_tensor((n, T.int64(2048)), "float16")
        compute = Ts.sblock_alloc_tensor((n, T.int64(2048)), "float16")
        var_T_multiply_intermediate_1 = Ts.sblock_alloc_tensor((n, T.int64(2048)), "float16")
        var_T_squeeze_intermediate = Ts.sblock_alloc_tensor((n, T.int64(2048)))
        var_compute_intermediate = Ts.sblock_alloc_tensor((n, T.int64(2048)), "float16")
        for i0, i1, k in T.grid(n, T.int64(2048), T.int64(2048)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                Ts.reads(lv26[v_i0, v_k], lv9[v_i1, v_k])
                Ts.writes(var_NT_matmul_intermediate[v_i0, v_i1])
                with Ts.init():
                    var_NT_matmul_intermediate[v_i0, v_i1] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1] = var_NT_matmul_intermediate[v_i0, v_i1] + lv26[v_i0, v_k] * lv9[v_i1, v_k]
        for i0, i1 in T.grid(n, T.int64(2048)):
            with Ts.sblock("compute"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(var_NT_matmul_intermediate[v_i0, v_i1])
                Ts.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.sigmoid(var_NT_matmul_intermediate[v_i0, v_i1])
        for ax0, ax1 in T.grid(n, T.int64(2048)):
            with Ts.sblock("T_multiply"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(var_NT_matmul_intermediate[v_ax0, v_ax1], compute[v_ax0, v_ax1])
                Ts.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1])
                var_T_multiply_intermediate_1[v_ax0, v_ax1] = var_NT_matmul_intermediate[v_ax0, v_ax1] * compute[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(n, T.int64(2048)):
            with Ts.sblock("T_squeeze"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(lv52[T.int64(0), v_ax0, v_ax1])
                Ts.writes(var_T_squeeze_intermediate[v_ax0, v_ax1])
                var_T_squeeze_intermediate[v_ax0, v_ax1] = lv52[T.int64(0), v_ax0, v_ax1]
        for i0, i1 in T.grid(n, T.int64(2048)):
            with Ts.sblock("compute_1"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(var_T_squeeze_intermediate[v_i0, v_i1])
                Ts.writes(var_compute_intermediate[v_i0, v_i1])
                var_compute_intermediate[v_i0, v_i1] = T.Cast("float16", var_T_squeeze_intermediate[v_i0, v_i1])
        for ax0, ax1 in T.grid(n, T.int64(2048)):
            with Ts.sblock("T_multiply_1"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(var_compute_intermediate[v_ax0, v_ax1], var_T_multiply_intermediate_1[v_ax0, v_ax1])
                Ts.writes(var_T_multiply_intermediate[v_ax0, v_ax1])
                var_T_multiply_intermediate[v_ax0, v_ax1] = var_compute_intermediate[v_ax0, v_ax1] * var_T_multiply_intermediate_1[v_ax0, v_ax1]

    n = T.dynamic("n", "int64")

    @Ts.function(private=True)
    def expected(lv26: T.Tensor((n, T.int64(2048)), 'float16'), lv9: T.Tensor((T.int64(2048), T.int64(2048)), "float16"), lv52: T.Tensor((T.int64(1), n, T.int64(2048)), "float32"), var_T_multiply_intermediate: T.Tensor((n, T.int64(2048)), 'float16')):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        var_NT_matmul_intermediate_reindex_pad_local = Ts.sblock_alloc_tensor((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2048)), "float16", scope="local")
        lv26_reindex_pad_shared = Ts.sblock_alloc_tensor((T.int64(1), (n + T.int64(31)) // T.int64(32) * T.int64(32), T.int64(2048)), "float16", scope="shared")
        lv9_reindex_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(2048), T.int64(2048)), "float16", scope="shared")
        for ax0_ax2_0_fused in T.parallel(T.int64(32), annotations={"thread_binding": "blockIdx.y"}):
            for ax1_0 in T.parallel((n + T.int64(31)) // T.int64(32), annotations={"thread_binding": "blockIdx.x"}):
                for ax2_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.y"}):
                    for ax1_1 in T.parallel(T.int64(1), annotations={"thread_binding": "vthread.x"}):
                        for ax2_2 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                            for ax1_2 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x", "tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax1_3_init, ax2_3_0_init in T.grid(T.int64(4), T.int64(2)):
                                    for ax2_3_1_init in T.vectorized(T.int64(2)):
                                        with Ts.sblock("NT_matmul_init"):
                                            v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = Ts.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3_init)
                                            v2 = Ts.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0_init * T.int64(2) + ax2_3_1_init)
                                            Ts.reads()
                                            Ts.writes(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                            var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = T.float16(0)
                                for ax3_0 in range(T.int64(128)):
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(2)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("lv26_reindex_pad_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(2048), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(32) + ax0_ax1_ax2_fused_1 * T.int64(4) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(lv26[v1, v2])
                                                        Ts.writes(lv26_reindex_pad_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        lv26_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < n, lv26[v1, v2], T.float16(0))
                                    for ax0_ax1_ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                        for ax0_ax1_ax2_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                                            for ax0_ax1_ax2_fused_2 in range(T.int64(4)):
                                                for ax0_ax1_ax2_fused_3 in T.vectorized(T.int64(2)):
                                                    with Ts.sblock("lv9_reindex_shared"):
                                                        v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                        v1 = Ts.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) // T.int64(16))
                                                        v2 = Ts.axis.spatial(T.int64(2048), ax3_0 * T.int64(16) + (ax0_ax1_ax2_fused_0 * T.int64(64) + ax0_ax1_ax2_fused_1 * T.int64(8) + ax0_ax1_ax2_fused_2 * T.int64(2) + ax0_ax1_ax2_fused_3) % T.int64(16))
                                                        Ts.reads(lv9[v1, v2])
                                                        Ts.writes(lv9_reindex_shared[v0, v1, v2])
                                                        Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 2]]})
                                                        lv9_reindex_shared[v0, v1, v2] = lv9[v1, v2]
                                    for ax3_1, ax1_3, ax2_3_0 in T.grid(T.int64(16), T.int64(4), T.int64(2)):
                                        for ax2_3_1 in T.vectorized(T.int64(2)):
                                            with Ts.sblock("NT_matmul_update"):
                                                v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                v1 = Ts.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_1 * T.int64(32) + ax1_2 * T.int64(4) + ax1_3)
                                                v2 = Ts.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_1 * T.int64(64) + ax2_2 * T.int64(4) + ax2_3_0 * T.int64(2) + ax2_3_1)
                                                v3 = Ts.axis.reduce(T.int64(2048), ax3_0 * T.int64(16) + ax3_1)
                                                Ts.reads(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2], lv26_reindex_pad_shared[T.int64(0), v1, v3], lv9_reindex_shared[T.int64(0), v2, v3])
                                                Ts.writes(var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2])
                                                var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] = var_NT_matmul_intermediate_reindex_pad_local[T.int64(0), v1, v2] + lv26_reindex_pad_shared[T.int64(0), v1, v3] * lv9_reindex_shared[T.int64(0), v2, v3]
                                for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(4), T.int64(2)):
                                    for ax2_1_1 in T.vectorized(T.int64(2)):
                                        with Ts.sblock("var_NT_matmul_intermediate_reindex_pad_local"):
                                            v0 = Ts.axis.spatial(T.int64(1), ax0)
                                            v1 = Ts.axis.spatial((n + T.int64(31)) // T.int64(32) * T.int64(32), ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1)
                                            v2 = Ts.axis.spatial(T.int64(2048), ax0_ax2_0_fused * T.int64(64) + ax2_2 * T.int64(4) + ax2_0 * T.int64(2) + ax2_1_1)
                                            Ts.reads(lv52[T.int64(0), v1, v2], var_NT_matmul_intermediate_reindex_pad_local[v0, v1, v2])
                                            Ts.where(ax1_0 * T.int64(32) + ax1_2 * T.int64(4) + ax1 < n)
                                            Ts.writes(var_T_multiply_intermediate[v1, v2])
                                            var_T_multiply_intermediate[v1, v2] = T.Cast("float16", lv52[T.int64(0), v1, v2]) * (var_NT_matmul_intermediate_reindex_pad_local[v0, v1, v2] * T.sigmoid(var_NT_matmul_intermediate_reindex_pad_local[v0, v1, v2]))

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), target=Target("nvidia/geforce-gtx-1080-ti"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_matmul_android():
    # fmt: off
    m = T.dynamic("m", "int64")

    @Ts.function(private=True)
    def before(inp0: T.Tensor((T.int64(1), m, T.int64(4096)), "float32"), inp1: T.Tensor((T.int64(4096), T.int64(4096)), "float32"), matmul: T.Tensor((T.int64(1), m, T.int64(4096)), "float32")):

        for i0, i1, i2, k in T.grid(T.int64(1), m, T.int64(4096), T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                with Ts.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + inp0[v_i0, v_i1, v_k] * inp1[v_k, v_i2]

    m = T.dynamic("m", "int64")

    @Ts.function(private=True)
    def expected(inp0: T.Tensor((T.int64(1), m, T.int64(4096)), "float32"), inp1: T.Tensor((T.int64(4096), T.int64(4096)), "float32"), matmul: T.Tensor((T.int64(1), m, T.int64(4096)), "float32")):
        T.func_attr({"tirx.is_scheduled": True})

        # with Ts.sblock("root"):
        inp0_reindex_pad = Ts.sblock_alloc_tensor((T.int64(1), (m + T.int64(15)) // T.int64(16), T.int64(4096), T.int64(16)))
        matmul_pad_local = Ts.sblock_alloc_tensor((T.int64(1), (m + T.int64(15)) // T.int64(16) * T.int64(16), T.int64(4096)), scope="local")
        inp0_reindex_pad_local = Ts.sblock_alloc_tensor((T.int64(1), (m + T.int64(15)) // T.int64(16), T.int64(4096), T.int64(16)), scope="local")
        for i0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.z"}):
            for i1_0 in T.parallel(((m + T.int64(15)) // T.int64(16) * T.int64(16) + T.int64(63)) // T.int64(64), annotations={"thread_binding": "blockIdx.y"}):
                for i2_0 in T.parallel(T.int64(128), annotations={"thread_binding": "blockIdx.x"}):
                    for i1_1 in T.parallel(T.int64(4), annotations={"thread_binding": "threadIdx.y"}):
                        for i2_1 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                            for i1_2 in T.vectorized(T.int64(16)):
                                with Ts.sblock("inp0_reindex_pad"):
                                    v0 = Ts.axis.spatial(T.int64(1), i0)
                                    v1 = Ts.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), i1_0 * T.int64(64) + i1_1 * T.int64(16) + i1_2)
                                    v2 = Ts.axis.spatial(T.int64(4096), i2_0 * T.int64(32) + i2_1)
                                    Ts.where((i1_0 * T.int64(4) + i1_1) * T.int64(16) + i1_2 < (m + T.int64(15)) // T.int64(16) * T.int64(16))
                                    Ts.reads(inp0[v0, v1, v2])
                                    Ts.writes(inp0_reindex_pad[v0, v1 // T.int64(16), v2, v1 % T.int64(16)])
                                    inp0_reindex_pad[v0, v1 // T.int64(16), v2, v1 % T.int64(16)] = T.if_then_else(v1 < m, inp0[v0, v1, v2], T.float32(0))
        for i2_0 in T.parallel(T.int64(16), annotations={"thread_binding": "blockIdx.x"}):
            for i0_i1_fused_0 in T.parallel(((m + T.int64(15)) // T.int64(16) * T.int64(16) + T.int64(63)) // T.int64(64), annotations={"thread_binding": "blockIdx.y"}):
                for i2_1 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                    for i0_i1_fused_1 in T.parallel(T.int64(4), annotations={"thread_binding": "threadIdx.y"}):
                        for i0_i1_fused_2_init in T.unroll(T.int64(16)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with Ts.sblock("matmul_init"):
                                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = Ts.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + i0_i1_fused_2_init)
                                    v_i2 = Ts.axis.spatial(T.int64(4096), i2_0 * T.int64(256) + i2_1 * T.int64(8) + i2_2_init)
                                    Ts.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1) * T.int64(16) + i0_i1_fused_2_init < (m + T.int64(15)) // T.int64(16) * T.int64(16))
                                    Ts.reads()
                                    Ts.writes(matmul_pad_local[v_i0, v_i1, v_i2])
                                    matmul_pad_local[v_i0, v_i1, v_i2] = T.float32(0)
                        for k_0, k_1 in T.grid(T.int64(128), T.int64(4)):
                            for k_2 in T.unroll(T.int64(8)):
                                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                                    for ax3 in T.vectorized(T.int64(16)):
                                        with Ts.sblock("inp0_reindex_pad_local"):
                                            v0 = Ts.axis.spatial(T.int64(1), ax0)
                                            v1 = Ts.axis.spatial((m + T.int64(15)) // T.int64(16), i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 + ax1)
                                            v2 = Ts.axis.spatial(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                            v3 = Ts.axis.spatial(T.int64(16), ax3)
                                            Ts.where(i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 < (m + T.int64(15)) // T.int64(16))
                                            Ts.reads(inp0_reindex_pad[v0, v1, v2, v3])
                                            Ts.writes(inp0_reindex_pad_local[v0, v1, v2, v3])
                                            inp0_reindex_pad_local[v0, v1, v2, v3] = inp0_reindex_pad[v0, v1, v2, v3]
                                for i0_i1_fused_2 in T.unroll(T.int64(16)):
                                    for i2_2 in T.vectorized(T.int64(8)):
                                        with Ts.sblock("matmul_update"):
                                            v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                            v_i1 = Ts.axis.spatial((m + T.int64(15)) // T.int64(16) * T.int64(16), i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + i0_i1_fused_2)
                                            v_i2 = Ts.axis.spatial(T.int64(4096), i2_0 * T.int64(256) + i2_1 * T.int64(8) + i2_2)
                                            v_k = Ts.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                            Ts.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1) * T.int64(16) + i0_i1_fused_2 < (m + T.int64(15)) // T.int64(16) * T.int64(16))
                                            Ts.reads(matmul_pad_local[v_i0, v_i1, v_i2], inp0_reindex_pad_local[v_i0, v_i1 // T.int64(16), v_k, v_i1 % T.int64(16)], inp1[v_k, v_i2])
                                            Ts.writes(matmul_pad_local[v_i0, v_i1, v_i2])
                                            matmul_pad_local[v_i0, v_i1, v_i2] = matmul_pad_local[v_i0, v_i1, v_i2] + inp0_reindex_pad_local[v_i0, v_i1 // T.int64(16), v_k, v_i1 % T.int64(16)] * inp1[v_k, v_i2]
                        for ax0 in T.unroll(T.int64(16)):
                            for ax1 in T.vectorized(T.int64(8)):
                                with Ts.sblock("matmul_pad"):
                                    v0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = Ts.axis.spatial(m, i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0)
                                    v2 = Ts.axis.spatial(T.int64(4096), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax1)
                                    Ts.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 - (m + T.int64(15)) // T.int64(16) < T.int64(0) or i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0 == T.int64(0)) and i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0 < m)
                                    Ts.reads(matmul_pad_local[v0, v1, v2])
                                    Ts.writes(matmul[v0, v1, v2])
                                    matmul[v0, v1, v2] = matmul_pad_local[v0, v1, v2]

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul(), target=Target('opencl', host={'kind': 'llvm', 'mtriple': 'aarch64-linux-android'}))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_fused_dequant_matmul_android():
    # fmt: off
    seq_len = T.dynamic("seq_len", "int64")

    @Ts.function(private=True)
    def before(lv452: T.Tensor((T.int64(512), T.int64(12288)), "uint32"), lv453: T.Tensor((T.int64(128), T.int64(12288)), "float16"), rms_norm130: T.Tensor((T.int64(1), seq_len, T.int64(4096)), 'float16'), transformer_h_0_attn_c_attn_bias3: T.Tensor((T.int64(12288),), "float16"), T_add_intermediate_intermediate: T.Tensor((T.int64(1), seq_len, T.int64(12288)), 'float16')):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        compute = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(12288)), "float16")
        dequantize_intermediate_intermediate = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(12288)), "float16")
        matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), seq_len, T.int64(12288)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with Ts.sblock("compute"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(lv452[v_i0 // T.int64(8), v_i1])
                Ts.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv452[v_i0 // T.int64(8), v_i1], T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(4096), T.int64(12288)):
            with Ts.sblock("dequantize"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(compute[v_i0, v_i1], lv453[v_i0 // T.int64(32), v_i1])
                Ts.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * lv453[v_i0 // T.int64(32), v_i1]
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(12288), T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(rms_norm130[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_k, v_i2])
                Ts.writes(matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                matmul_intermediate[v_i0, v_i1, v_i2] = matmul_intermediate[v_i0, v_i1, v_i2] + rms_norm130[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(12288)):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(matmul_intermediate[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                Ts.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]

    seq_len = T.dynamic("seq_len", "int64")

    @Ts.function(private=True)
    def expected(lv452: T.Tensor((T.int64(512), T.int64(12288)), "uint32"), lv453: T.Tensor((T.int64(128), T.int64(12288)), "float16"), rms_norm130: T.Tensor((T.int64(1), seq_len, T.int64(4096)), 'float16'), transformer_h_0_attn_c_attn_bias3: T.Tensor((T.int64(12288),), "float16"), T_add_intermediate_intermediate: T.Tensor((T.int64(1), seq_len, T.int64(12288)), 'float16')):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        dequantize_intermediate_intermediate_local = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(12288)), "float16", scope="local")
        rms_norm130_reindex_pad = Ts.sblock_alloc_tensor((T.int64(1), (seq_len + T.int64(15)) // T.int64(16), T.int64(4096), T.int64(16)), "float16")
        matmul_intermediate_pad_local = Ts.sblock_alloc_tensor((T.int64(1), (seq_len + T.int64(15)) // T.int64(16) * T.int64(16), T.int64(12288)), "float16", scope="local")
        rms_norm130_reindex_pad_local = Ts.sblock_alloc_tensor((T.int64(1), (seq_len + T.int64(15)) // T.int64(16), T.int64(4096), T.int64(16)), "float16", scope="local")
        lv452_local = Ts.sblock_alloc_tensor((T.int64(512), T.int64(12288)), "uint32", scope="local")
        lv453_local = Ts.sblock_alloc_tensor((T.int64(128), T.int64(12288)), "float16", scope="local")
        for i0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.z"}):
            for i1_0 in T.parallel(((seq_len + T.int64(15)) // T.int64(16) * T.int64(16) + T.int64(63)) // T.int64(64), annotations={"thread_binding": "blockIdx.y"}):
                for i2_0 in T.parallel(T.int64(128), annotations={"thread_binding": "blockIdx.x"}):
                    for i1_1 in T.parallel(T.int64(4), annotations={"thread_binding": "threadIdx.y"}):
                        for i2_1 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                            for i1_2 in T.vectorized(T.int64(16)):
                                with Ts.sblock("rms_norm130_reindex_pad"):
                                    v0 = Ts.axis.spatial(T.int64(1), i0)
                                    v1 = Ts.axis.spatial((seq_len + T.int64(15)) // T.int64(16) * T.int64(16), i1_0 * T.int64(64) + i1_1 * T.int64(16) + i1_2)
                                    v2 = Ts.axis.spatial(T.int64(4096), i2_0 * T.int64(32) + i2_1)
                                    Ts.where((i1_0 * T.int64(4) + i1_1) * T.int64(16) + i1_2 < (seq_len + T.int64(15)) // T.int64(16) * T.int64(16))
                                    Ts.reads(rms_norm130[v0, v1, v2])
                                    Ts.writes(rms_norm130_reindex_pad[v0, v1 // T.int64(16), v2, v1 % T.int64(16)])
                                    rms_norm130_reindex_pad[v0, v1 // T.int64(16), v2, v1 % T.int64(16)] = T.if_then_else(v1 < seq_len, rms_norm130[v0, v1, v2], T.float16(0))
        for i2_0 in T.parallel(T.int64(48), annotations={"thread_binding": "blockIdx.x"}):
            for i0_i1_fused_0 in T.parallel(((seq_len + T.int64(15)) // T.int64(16) * T.int64(16) + T.int64(63)) // T.int64(64), annotations={"thread_binding": "blockIdx.y"}):
                for i2_1 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                    for i0_i1_fused_1 in T.parallel(T.int64(4), annotations={"thread_binding": "threadIdx.y"}):
                        for i0_i1_fused_2_init in T.unroll(T.int64(16)):
                            for i2_2_init in T.vectorized(T.int64(8)):
                                with Ts.sblock("matmul_init"):
                                    v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                    v_i1 = Ts.axis.spatial((seq_len + T.int64(15)) // T.int64(16) * T.int64(16), i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + i0_i1_fused_2_init)
                                    v_i2 = Ts.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + i2_2_init)
                                    Ts.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1) * T.int64(16) + i0_i1_fused_2_init < (seq_len + T.int64(15)) // T.int64(16) * T.int64(16))
                                    Ts.reads()
                                    Ts.writes(matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                    matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = T.float16(0)
                        for k_0 in range(T.int64(128)):
                            for ax0 in range(T.int64(1)):
                                for ax1 in T.vectorized(T.int64(8)):
                                    with Ts.sblock("lv453_local"):
                                        v0 = Ts.axis.spatial(T.int64(128), k_0 + ax0)
                                        v1 = Ts.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax1)
                                        Ts.reads(lv453[v0, v1])
                                        Ts.writes(lv453_local[v0, v1])
                                        lv453_local[v0, v1] = lv453[v0, v1]
                            for k_1 in range(T.int64(4)):
                                for ax0 in range(T.int64(1)):
                                    for ax1 in T.vectorized(T.int64(8)):
                                        with Ts.sblock("lv452_local"):
                                            v0 = Ts.axis.spatial(T.int64(512), k_0 * T.int64(4) + k_1 + ax0)
                                            v1 = Ts.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax1)
                                            Ts.reads(lv452[v0, v1])
                                            Ts.writes(lv452_local[v0, v1])
                                            lv452_local[v0, v1] = lv452[v0, v1]
                                for k_2 in T.unroll(T.int64(8)):
                                    for ax0 in T.vectorized(T.int64(8)):
                                        with Ts.sblock("dequantize"):
                                            v_i0 = Ts.axis.spatial(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                            v_i1 = Ts.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax0)
                                            Ts.reads(lv452_local[v_i0 // T.int64(8), v_i1], lv453_local[v_i0 // T.int64(32), v_i1])
                                            Ts.writes(dequantize_intermediate_intermediate_local[v_i0, v_i1])
                                            dequantize_intermediate_intermediate_local[v_i0, v_i1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv452_local[v_i0 // T.int64(8), v_i1], T.Cast("uint32", v_i0 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7)) * lv453_local[v_i0 // T.int64(32), v_i1]
                                    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                                        for ax3 in T.vectorized(T.int64(16)):
                                            with Ts.sblock("rms_norm130_reindex_pad_local"):
                                                v0 = Ts.axis.spatial(T.int64(1), ax0)
                                                v1 = Ts.axis.spatial((seq_len + T.int64(15)) // T.int64(16), i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 + ax1)
                                                v2 = Ts.axis.spatial(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2 + ax2)
                                                v3 = Ts.axis.spatial(T.int64(16), ax3)
                                                Ts.where(i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 < (seq_len + T.int64(15)) // T.int64(16))
                                                Ts.reads(rms_norm130_reindex_pad[v0, v1, v2, v3])
                                                Ts.writes(rms_norm130_reindex_pad_local[v0, v1, v2, v3])
                                                rms_norm130_reindex_pad_local[v0, v1, v2, v3] = rms_norm130_reindex_pad[v0, v1, v2, v3]
                                    for i0_i1_fused_2 in T.unroll(T.int64(16)):
                                        for i2_2 in T.vectorized(T.int64(8)):
                                            with Ts.sblock("matmul_update"):
                                                v_i0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                                v_i1 = Ts.axis.spatial((seq_len + T.int64(15)) // T.int64(16) * T.int64(16), i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + i0_i1_fused_2)
                                                v_i2 = Ts.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + i2_2)
                                                v_k = Ts.axis.reduce(T.int64(4096), k_0 * T.int64(32) + k_1 * T.int64(8) + k_2)
                                                Ts.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1) * T.int64(16) + i0_i1_fused_2 < (seq_len + T.int64(15)) // T.int64(16) * T.int64(16))
                                                Ts.reads(matmul_intermediate_pad_local[v_i0, v_i1, v_i2], rms_norm130_reindex_pad_local[v_i0, v_i1 // T.int64(16), v_k, v_i1 % T.int64(16)], dequantize_intermediate_intermediate_local[v_k, v_i2])
                                                Ts.writes(matmul_intermediate_pad_local[v_i0, v_i1, v_i2])
                                                matmul_intermediate_pad_local[v_i0, v_i1, v_i2] = matmul_intermediate_pad_local[v_i0, v_i1, v_i2] + rms_norm130_reindex_pad_local[v_i0, v_i1 // T.int64(16), v_k, v_i1 % T.int64(16)] * dequantize_intermediate_intermediate_local[v_k, v_i2]
                        for ax0 in T.unroll(T.int64(16)):
                            for ax1 in T.vectorized(T.int64(8)):
                                with Ts.sblock("T_add"):
                                    v_ax0 = Ts.axis.spatial(T.int64(1), T.int64(0))
                                    v_ax1 = Ts.axis.spatial(seq_len, i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0)
                                    v_ax2 = Ts.axis.spatial(T.int64(12288), i2_0 * T.int64(256) + i2_1 * T.int64(8) + ax1)
                                    Ts.where((i0_i1_fused_0 * T.int64(4) + i0_i1_fused_1 - (seq_len + T.int64(15)) // T.int64(16) < T.int64(0) or i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0 == T.int64(0)) and i0_i1_fused_0 * T.int64(64) + i0_i1_fused_1 * T.int64(16) + ax0 < seq_len)
                                    Ts.reads(matmul_intermediate_pad_local[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_attn_bias3[v_ax2])
                                    Ts.writes(T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2])
                                    T_add_intermediate_intermediate[v_ax0, v_ax1, v_ax2] = matmul_intermediate_pad_local[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_attn_bias3[v_ax2]
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        target=Target("opencl", host={"kind": "llvm", "mtriple": "aarch64-linux-android"}),
    )(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
