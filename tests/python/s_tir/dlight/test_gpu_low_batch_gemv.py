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

from unittest import mock

import tvm_tirx.testing
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm.s_tir.dlight.gpu import low_batch_gemv
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def test_batch_decode_gemv():
    # fmt: off

    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def before(lv429: T.Tensor((T.int64(4096), T.int64(3584)), "uint32"), lv430: T.Tensor((T.int64(4096), T.int64(896)), "float16"), lv807: T.Tensor((batch_size, T.int64(1), T.int64(28672)), 'float16'), NT_matmul_intermediate: T.Tensor((batch_size, T.int64(1), T.int64(4096)), 'float16')):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        compute = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(28672)), "float16")
        dequantize_intermediate_intermediate = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(28672)), "float16")
        for i0, i1 in T.grid(T.int64(4096), T.int64(28672)):
            with Ts.sblock("compute"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(lv429[v_i0, v_i1 // T.int64(8)])
                Ts.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv429[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(4096), T.int64(28672)):
            with Ts.sblock("dequantize"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(compute[v_i0, v_i1], lv430[v_i0, v_i1 // T.int64(32)])
                Ts.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * lv430[v_i0, v_i1 // T.int64(32)]
        for i0, i1, i2, k in T.grid(batch_size, T.int64(1), T.int64(4096), T.int64(28672)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv807[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_i2, v_k])
                Ts.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv807[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_i2, v_k]

    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def expected(lv429: T.Tensor((T.int64(4096), T.int64(3584)), "uint32"), lv430: T.Tensor((T.int64(4096), T.int64(896)), "float16"), lv807: T.Tensor((batch_size, T.int64(1), T.int64(28672)), 'float16'), NT_matmul_intermediate: T.Tensor((batch_size, T.int64(1), T.int64(4096)), 'float16')):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        dequantize_intermediate_intermediate_local = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(28672)), "float16", scope="local")
        NT_matmul_intermediate_pad_local = Ts.sblock_alloc_tensor(((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        NT_matmul_intermediate_pad_rf_local = Ts.sblock_alloc_tensor((T.int64(128), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        NT_matmul_intermediate_pad_rf_local_1 = Ts.sblock_alloc_tensor((T.int64(32), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        for ax0_0 in T.parallel((batch_size + T.int64(3)) // T.int64(4), annotations={"thread_binding": "blockIdx.y"}):
            for u_fused_ax1_fused_fused_0 in T.parallel(T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
                for u_fused_ax1_fused_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.y"}):
                        for ax0_1_init, u_fused_ax1_fused_fused_2_init in T.grid(T.int64(4), T.int64(2)):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                                with Ts.sblock("NT_matmul_rf_init"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = Ts.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                    v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1_init)
                                    v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2_init)
                                    Ts.reads()
                                    Ts.writes(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1])
                                    NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] = T.float16(0)
                        for ax2_fused_u_fused_0 in T.serial(T.int64(112), annotations={"tirx.auto_unroll_max_step": 8, "tirx.unroll_explicit": 1}):
                            for ax0_0_1, ax1 in T.grid(T.int64(2), T.int64(8)):
                                for ax0_1 in T.vectorized(T.int64(1)):
                                    with Ts.sblock("dequantize"):
                                        v0 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + ax0_0_1 + ax0_1)
                                        v1 = Ts.axis.spatial(T.int64(28672), ax2_fused_u_fused_0 * T.int64(256) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(8) + ax1)
                                        Ts.reads(lv429[v0, v1 // T.int64(8)], lv430[v0, v1 // T.int64(32)])
                                        Ts.writes(dequantize_intermediate_intermediate_local[v0, v1])
                                        dequantize_intermediate_intermediate_local[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv429[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7)) * lv430[v0, v1 // T.int64(32)]
                            for ax0_1, u_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(4), T.int64(2), T.int64(2)):
                                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                    with Ts.sblock("NT_matmul_rf_update"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = Ts.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                        v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1)
                                        v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2)
                                        vax2_fused_u_fused_0, vax2_fused_u_fused_2 = Ts.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                        Ts.reads(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1], lv807[v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], dequantize_intermediate_intermediate_local[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)])
                                        Ts.writes(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1])
                                        NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] = NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] + T.if_then_else(v0 < batch_size, lv807[v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], T.float16(0)) * dequantize_intermediate_intermediate_local[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)]
                for ax3_fused_0_ax3_fused_1_fused in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.y"}):
                        for ax3_fused_2_0 in T.serial(T.int64(1), annotations={"tirx.auto_unroll_max_step": 8, "tirx.unroll_explicit": 1}):
                            for ax2 in T.serial(T.int64(0), T.int64(4)):
                                for ax3_fused_2_1 in T.vectorized(T.int64(2)):
                                    with Ts.sblock("NT_matmul_rf_init"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = Ts.axis.spatial(T.int64(32), ax0)
                                        v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                        v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                        Ts.reads()
                                        Ts.writes(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                        NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] = T.float16(0)
                                    for ax1 in T.serial(T.int64(0), T.int64(4)):
                                        with Ts.sblock("NT_matmul_rf_update"):
                                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                            v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                            v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                            Ts.reads(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1], NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, T.int64(0), v1])
                                            Ts.writes(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                            NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] = NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] + NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, T.int64(0), v1]
                for ax2_fused_2, ax1 in T.grid(T.int64(2), T.int64(4)):
                    for ax2_fused_0_ax2_fused_1_fused in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                        for ax0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.y"}):
                            with Ts.sblock("NT_matmul"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = Ts.axis.reduce(T.int64(32), ax0)
                                v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax1)
                                v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax2_fused_0_ax2_fused_1_fused * T.int64(2) + ax2_fused_2)
                                Ts.reads(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                Ts.writes(NT_matmul_intermediate_pad_local[v0, T.int64(0), v1])
                                with Ts.init():
                                    NT_matmul_intermediate_pad_local[v0, T.int64(0), v1] = T.float16(0)
                                NT_matmul_intermediate_pad_local[v0, T.int64(0), v1] = NT_matmul_intermediate_pad_local[v0, T.int64(0), v1] + NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1]
                for ax0 in T.serial(T.int64(0), T.int64(4)):
                    for ax1_fused_0_ax1_fused_1_fused in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                        for ax1_fused_2 in T.serial(T.int64(0), T.int64(2)):
                            with Ts.sblock("NT_matmul_intermediate_pad"):
                                v0 = Ts.axis.spatial(batch_size, ax0_0 * T.int64(4) + ax0)
                                v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax1_fused_0_ax1_fused_1_fused * T.int64(2) + ax1_fused_2)
                                Ts.where((ax0_0 - (batch_size + T.int64(3)) // T.int64(4) < T.int64(0) or ax0_0 * T.int64(4) + ax0 == T.int64(0)) and ax0_0 * T.int64(4) + ax0 < batch_size)
                                Ts.reads(NT_matmul_intermediate_pad_local[v0, T.int64(0), v1])
                                Ts.writes(NT_matmul_intermediate[v0, T.int64(0), v1])
                                NT_matmul_intermediate[v0, T.int64(0), v1] = NT_matmul_intermediate_pad_local[v0, T.int64(0), v1]

    # fmt: on
    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4), target=Target("metal"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_batch_gemv():
    N = 4096
    K = 4096

    # fmt: off
    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def before(A: T.Tensor((batch_size, T.int64(1), T.int64(K)), 'float16'), B: T.Tensor((T.int64(N), T.int64(K)), "float16"), NT_matmul: T.Tensor((batch_size, T.int64(1), T.int64(N)), 'float16')):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        for i0, i1, i2, k in T.grid(batch_size, T.int64(1), T.int64(N), T.int64(K)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                Ts.writes(NT_matmul[v_i0, v_i1, v_i2])
                with Ts.init():
                    NT_matmul[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul[v_i0, v_i1, v_i2] = NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]

    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def expected(A: T.Tensor((batch_size, T.int64(1), T.int64(4096)), 'float16'), B: T.Tensor((T.int64(4096), T.int64(4096)), "float16"), NT_matmul: T.Tensor((batch_size, T.int64(1), T.int64(4096)), 'float16')):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        NT_matmul_pad_local = Ts.sblock_alloc_tensor(((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        NT_matmul_pad_rf_local = Ts.sblock_alloc_tensor((T.int64(128), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        NT_matmul_pad_rf_local_1 = Ts.sblock_alloc_tensor((T.int64(32), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1), T.int64(4096)), "float16", scope="local")
        for ax0_0 in T.parallel((batch_size + T.int64(3)) // T.int64(4), annotations={"thread_binding": "blockIdx.y"}):
            for u_fused_ax1_fused_fused_0 in T.parallel(T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
                for u_fused_ax1_fused_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.y"}):
                        for ax0_1_init, u_fused_ax1_fused_fused_2_init in T.grid(T.int64(4), T.int64(2)):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                                with Ts.sblock("NT_matmul_rf_init"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = Ts.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                    v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1_init)
                                    v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2_init)
                                    Ts.reads()
                                    Ts.writes(NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1])
                                    NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] = T.float16(0)
                        for ax2_fused_u_fused_0 in T.serial(T.int64(16), annotations={"tirx.auto_unroll_max_step": 8, "tirx.unroll_explicit": 1}):
                            for ax0_1, u_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(4), T.int64(2), T.int64(2)):
                                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                    with Ts.sblock("NT_matmul_rf_update"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = Ts.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                        v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1)
                                        v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2)
                                        vax2_fused_u_fused_0, vax2_fused_u_fused_2 = Ts.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                        Ts.reads(NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1], A[v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], B[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)])
                                        Ts.writes(NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1])
                                        NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] = NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, T.int64(0), v1] + T.if_then_else(v0 < batch_size, A[v0, T.int64(0), vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], T.float16(0)) * B[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)]
                for ax3_fused_0_ax3_fused_1_fused in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.y"}):
                        for ax3_fused_2_0 in T.serial(T.int64(1), annotations={"tirx.auto_unroll_max_step": 8, "tirx.unroll_explicit": 1}):
                            for ax2 in T.serial(T.int64(0), T.int64(4)):
                                for ax3_fused_2_1 in T.vectorized(T.int64(2)):
                                    with Ts.sblock("NT_matmul_rf_init"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = Ts.axis.spatial(T.int64(32), ax0)
                                        v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                        v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                        Ts.reads()
                                        Ts.writes(NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                        NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] = T.float16(0)
                                    for ax1 in T.serial(T.int64(0), T.int64(4)):
                                        with Ts.sblock("NT_matmul_rf_update"):
                                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                            v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                            v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                            Ts.reads(NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1], NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, T.int64(0), v1])
                                            Ts.writes(NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                            NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] = NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1] + NT_matmul_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, T.int64(0), v1]
                for ax2_fused_2, ax1 in T.grid(T.int64(2), T.int64(4)):
                    for ax2_fused_0_ax2_fused_1_fused in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                        for ax0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.y"}):
                            with Ts.sblock("NT_matmul"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = Ts.axis.reduce(T.int64(32), ax0)
                                v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax1)
                                v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax2_fused_0_ax2_fused_1_fused * T.int64(2) + ax2_fused_2)
                                Ts.reads(NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1])
                                Ts.writes(NT_matmul_pad_local[v0, T.int64(0), v1])
                                with Ts.init():
                                    NT_matmul_pad_local[v0, T.int64(0), v1] = T.float16(0)
                                NT_matmul_pad_local[v0, T.int64(0), v1] = NT_matmul_pad_local[v0, T.int64(0), v1] + NT_matmul_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, T.int64(0), v1]
                for ax0 in T.serial(T.int64(0), T.int64(4)):
                    for ax1_fused_0_ax1_fused_1_fused in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                        for ax1_fused_2 in T.serial(T.int64(0), T.int64(2)):
                            with Ts.sblock("NT_matmul_pad"):
                                v0 = Ts.axis.spatial(batch_size, ax0_0 * T.int64(4) + ax0)
                                v1 = Ts.axis.spatial(T.int64(4096), u_fused_ax1_fused_fused_0 * T.int64(16) + ax1_fused_0_ax1_fused_1_fused * T.int64(2) + ax1_fused_2)
                                Ts.where((ax0_0 - (batch_size + T.int64(3)) // T.int64(4) < T.int64(0) or ax0_0 * T.int64(4) + ax0 == T.int64(0)) and ax0_0 * T.int64(4) + ax0 < batch_size)
                                Ts.reads(NT_matmul_pad_local[v0, T.int64(0), v1])
                                Ts.writes(NT_matmul[v0, T.int64(0), v1])
                                NT_matmul[v0, T.int64(0), v1] = NT_matmul_pad_local[v0, T.int64(0), v1]
    # fmt: on
    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4), target=Target("metal"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_reduction_symbolic_var():
    # fmt: off
    kv_seq_len = T.dynamic("kv_seq_len", "int64")

    @Ts.function(private=True)
    def before(A: T.Tensor((T.int64(1), T.int64(32), T.int64(1), kv_seq_len), "float32"), B: T.Tensor((T.int64(1), T.int64(32), kv_seq_len, T.int64(128)), "float32"), matmul: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), kv_seq_len):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                Ts.writes(matmul[v_i0, v_i1, v_i2, v_i3])
                with Ts.init():
                    matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]
    # fmt: on
    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4), target=Target("metal"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], before)


def test_small_spatial_axis():
    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def func(
        A: T.Tensor((batch_size, T.int64(4096)), "float16"),
        B: T.Tensor((T.int64(8), T.int64(4096)), "float16"),
        C: T.Tensor((batch_size, T.int64(8)), "float16"),
    ):
        T.func_attr({"tirx.noalias": True})

        for i0, i1, k in T.grid(batch_size, T.int64(8), T.int64(4096)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                Ts.reads(A[v_i0, v_k], B[v_i1, v_k])
                Ts.writes(C[v_i0, v_i1])
                with Ts.init():
                    C[v_i0, v_i1] = T.float16(0)
                C[v_i0, v_i1] = C[v_i0, v_i1] + A[v_i0, v_k] * B[v_i1, v_k]

    # fmt: off
    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def expected(A: T.Tensor((batch_size, T.int64(4096)), 'float16'), B: T.Tensor((T.int64(8), T.int64(4096)), "float16"), C: T.Tensor((batch_size, T.int64(8)), 'float16')):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        C_pad_local = Ts.sblock_alloc_tensor(((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(8)), "float16", scope="local")
        C_pad_rf_local = Ts.sblock_alloc_tensor((T.int64(128), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(8)), "float16", scope="local")
        C_pad_rf_local_1 = Ts.sblock_alloc_tensor((T.int64(32), (batch_size + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(8)), "float16", scope="local")
        for ax0_0 in T.parallel((batch_size + T.int64(3)) // T.int64(4), annotations={"thread_binding": "blockIdx.y"}):
            for u_fused_ax1_fused_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for u_fused_ax1_fused_fused_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                    for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                        for ax0_1_init, u_fused_ax1_fused_fused_2_init in T.grid(T.int64(4), T.int64(2)):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                                with Ts.sblock("NT_matmul_rf_init"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = Ts.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                    v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1_init)
                                    v1 = Ts.axis.spatial(T.int64(8), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2_init)
                                    Ts.where((u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1) * T.int64(2) + u_fused_ax1_fused_fused_2_init < T.int64(8))
                                    Ts.reads()
                                    Ts.writes(C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1])
                                    C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1] = T.float16(0)
                        for ax2_fused_u_fused_0 in T.serial(T.int64(16), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            for ax0_1, u_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(4), T.int64(2), T.int64(2)):
                                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                    with Ts.sblock("NT_matmul_rf_update"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = Ts.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                        v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1)
                                        v1 = Ts.axis.spatial(T.int64(8), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2)
                                        vax2_fused_u_fused_0, vax2_fused_u_fused_2 = Ts.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                        Ts.where((u_fused_ax1_fused_fused_0 * T.int64(16) + u_fused_ax1_fused_fused_1) * T.int64(2) + u_fused_ax1_fused_fused_2 < T.int64(8))
                                        Ts.reads(C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1], A[v0, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], B[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)])
                                        Ts.writes(C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1])
                                        C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1] = C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, v0, v1] + T.if_then_else(v0 < batch_size, A[v0, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], T.float16(0)) * B[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)]
                for ax3_fused_0_ax3_fused_1_fused in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                    for ax0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                        for ax3_fused_2_0 in T.serial(T.int64(1), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                            for ax2 in T.serial(T.int64(0), T.int64(4)):
                                for ax3_fused_2_1 in T.vectorized(T.int64(2)):
                                    with Ts.sblock("NT_matmul_rf_init"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = Ts.axis.spatial(T.int64(32), ax0)
                                        v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                        v1 = Ts.axis.spatial(T.int64(8), ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                        Ts.where((T.Mul(T.int64(0), T.int64(16)) + ax3_fused_0_ax3_fused_1_fused % T.int64(16)) * T.int64(2) + (ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1) < T.int64(8))
                                        Ts.reads()
                                        Ts.writes(C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1])
                                        C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1] = T.float16(0)
                                    for ax1 in T.serial(T.int64(0), T.int64(4)):
                                        with Ts.sblock("NT_matmul_rf_update"):
                                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                            v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                            v1 = Ts.axis.spatial(T.int64(8), ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                            Ts.where((T.Mul(T.int64(0), T.int64(16)) + ax3_fused_0_ax3_fused_1_fused % T.int64(16)) * T.int64(2) + (ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1) < T.int64(8))
                                            Ts.reads(C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1], C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, v1])
                                            Ts.writes(C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1])
                                            C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1] = C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1] + C_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, v0, v1]
                for ax2_fused_2, ax1 in T.grid(T.int64(2), T.int64(4)):
                    for ax2_fused_0_ax2_fused_1_fused in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                        for ax0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("NT_matmul"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = Ts.axis.reduce(T.int64(32), ax0)
                                v0 = Ts.axis.spatial((batch_size + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax1)
                                v1 = Ts.axis.spatial(T.int64(8), ax2_fused_0_ax2_fused_1_fused * T.int64(2) + ax2_fused_2)
                                Ts.where((T.Mul(T.int64(0), T.int64(16)) + ax2_fused_0_ax2_fused_1_fused % T.int64(16)) * T.int64(2) + ax2_fused_2 < T.int64(8))
                                Ts.reads(C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1])
                                Ts.writes(C_pad_local[v0, v1])
                                with Ts.init():
                                    C_pad_local[v0, v1] = T.float16(0)
                                C_pad_local[v0, v1] = C_pad_local[v0, v1] + C_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, v0, v1]
                for ax0 in T.serial(T.int64(0), T.int64(4)):
                    for ax1_fused_0_ax1_fused_1_fused in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                        for ax1_fused_2 in T.serial(T.int64(0), T.int64(2)):
                            with Ts.sblock("C_pad"):
                                v0 = Ts.axis.spatial(batch_size, ax0_0 * T.int64(4) + ax0)
                                v1 = Ts.axis.spatial(T.int64(8), ax1_fused_0_ax1_fused_1_fused * T.int64(2) + ax1_fused_2)
                                Ts.where((ax0_0 - (batch_size + T.int64(3)) // T.int64(4) < T.int64(0) or ax0_0 * T.int64(4) + ax0 == T.int64(0)) and ax0_0 * T.int64(4) + ax0 < batch_size and (T.Mul(T.int64(0), T.int64(16)) + ax1_fused_0_ax1_fused_1_fused % T.int64(16)) * T.int64(2) + ax1_fused_2 < T.int64(8))
                                Ts.reads(C_pad_local[v0, v1])
                                Ts.writes(C[v0, v1])
                                C[v0, v1] = C_pad_local[v0, v1]
    # fmt: on

    mod = tvm_tirx.IRModule({"main": func})
    mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4), target=Target("cuda"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_outer_reduction():
    # fmt: off
    batch_size = T.dynamic("batch_size", "int32")

    @Ts.function(private=True)
    def before(
        B0: T.Tensor((512, 6144), "uint32"),
        B1: T.Tensor((128, 6144), "float16"),
        A: T.Tensor((batch_size, 1, 4096), 'float16'),
        C: T.Tensor((batch_size, 1, 6144), 'float16')
    ):

        compute = Ts.sblock_alloc_tensor((4096, 6144), "float16")
        B = Ts.sblock_alloc_tensor((4096, 6144), "float16")
        for i0, i1 in T.grid(4096, 6144):
            with Ts.sblock("compute"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(B0[v_i0 // 8, v_i1], T.Cast("uint32", v_i0 % 8 * 4)), T.uint32(15)))
        for i0, i1 in T.grid(4096, 6144):
            with Ts.sblock("dequantize"):
                v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                B[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * B1[v_i0 // 32, v_i1]
        for i0, i1, i2, k in T.grid(batch_size, 1, 6144, 4096):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                with Ts.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]

    batch_size = T.dynamic("batch_size", "int32")

    @Ts.function(private=True)
    def expected(B0: T.Tensor((512, 6144), "uint32"), B1: T.Tensor((128, 6144), "float16"), A: T.Tensor((batch_size, 1, 4096), 'float16'), C: T.Tensor((batch_size, 1, 6144), 'float16')):
        T.func_attr({"tirx.is_scheduled": True})

        # with Ts.sblock("root"):
        B_local = Ts.sblock_alloc_tensor((4096, 6144), "float16", scope="local")
        A_pad_shared = Ts.sblock_alloc_tensor(((batch_size + 3) // 4 * 4, 1, 4096), "float16", scope="shared")
        C_pad_local = Ts.sblock_alloc_tensor(((batch_size + 3) // 4 * 4, 1, 6144), "float16", scope="local")
        C_pad_rf_local = Ts.sblock_alloc_tensor((32, (batch_size + 3) // 4 * 4, 1, 6144), "float16", scope="local")
        C_pad_rf_local_1 = Ts.sblock_alloc_tensor((4, (batch_size + 3) // 4 * 4, 1, 6144), "float16", scope="local")
        B0_local = Ts.sblock_alloc_tensor((512, 6144), "uint32", scope="local")
        B1_local = Ts.sblock_alloc_tensor((128, 6144), "float16", scope="local")
        for ax0_0 in T.parallel((batch_size + 3) // 4, annotations={"thread_binding": "blockIdx.y"}):
            for ax1_fused_0 in T.parallel(96, annotations={"thread_binding": "blockIdx.x"}):
                for ax1_fused_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_1_ax2_fused_3_fused_0 in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                        for ax0_1_init, ax2_fused_1_ax2_fused_3_fused_1_0_init in T.grid(4, 2):
                            for ax2_fused_1_ax2_fused_3_fused_1_1_init in T.vectorized(4):
                                with Ts.sblock("matmul_rf_init"):
                                    vax2_fused_1_ax2_fused_3_fused = Ts.axis.spatial(32, ax2_fused_1_ax2_fused_3_fused_0 * 8 + ax2_fused_1_ax2_fused_3_fused_1_0_init * 4 + ax2_fused_1_ax2_fused_3_fused_1_1_init)
                                    v0 = Ts.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax0_1_init)
                                    v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                    Ts.reads()
                                    Ts.writes(C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1])
                                    C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1] = T.float16(0)
                        for ax2_fused_0 in range(32):
                            for ax0_ax1_fused in T.vectorized(4):
                                with Ts.sblock("B0_local"):
                                    v0 = Ts.axis.spatial(512, ax2_fused_0 * 16 + ax2_fused_1_ax2_fused_3_fused_0 * 4 + ax0_ax1_fused)
                                    v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                    Ts.reads(B0[v0, v1])
                                    Ts.writes(B0_local[v0, v1])
                                    B0_local[v0, v1] = B0[v0, v1]
                            for ax0_ax1_fused in T.vectorized(1):
                                with Ts.sblock("B1_local"):
                                    v0 = Ts.axis.spatial(128, ax2_fused_0 * 4 + ax2_fused_1_ax2_fused_3_fused_0)
                                    v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                    Ts.reads(B1[v0, v1])
                                    Ts.writes(B1_local[v0, v1])
                                    B1_local[v0, v1] = B1[v0, v1]
                            for ax0_ax1_fused_0 in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                                for ax0_ax1_fused_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                                    for ax0_ax1_fused_2 in T.vectorized(2):
                                        with Ts.sblock("A_pad"):
                                            v0 = Ts.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 128)
                                            v1 = Ts.axis.spatial(4096, ax2_fused_0 * 128 + (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 128)
                                            Ts.reads(A[v0, 0, v1])
                                            Ts.writes(A_pad_shared[v0, 0, v1])
                                            Ts.sblock_attr({"tensor_dim_align": [[0, 1, 8, 1]]})
                                            A_pad_shared[v0, 0, v1] = T.if_then_else(v0 < batch_size, A[v0, 0, v1], T.float16(0))
                            for ax2_fused_2 in range(4):
                                for ax0_ax1_fused_0 in range(2):
                                    for ax0_ax1_fused_1 in T.vectorized(4):
                                        with Ts.sblock("dequantize"):
                                            v0 = Ts.axis.spatial(4096, ax2_fused_0 * 128 + ax2_fused_1_ax2_fused_3_fused_0 * 32 + ax2_fused_2 * 8 + ax0_ax1_fused_0 * 4 + ax0_ax1_fused_1)
                                            v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                            Ts.reads(B0_local[v0 // 8, v1], B1_local[v0 // 32, v1])
                                            Ts.writes(B_local[v0, v1])
                                            B_local[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(B0_local[v0 // 8, v1], T.Cast("uint32", v0 % 8 * 4)), T.uint32(15))) - T.float16(7)) * B1_local[v0 // 32, v1]
                                for ax0_1, ax2_fused_1_ax2_fused_3_fused_1_0 in T.grid(4, 2):
                                    for ax2_fused_1_ax2_fused_3_fused_1_1 in T.vectorized(4):
                                        with Ts.sblock("matmul_rf_update"):
                                            vax2_fused_1_ax2_fused_3_fused = Ts.axis.spatial(32, ax2_fused_1_ax2_fused_3_fused_0 * 8 + ax2_fused_1_ax2_fused_3_fused_1_0 * 4 + ax2_fused_1_ax2_fused_3_fused_1_1)
                                            v0 = Ts.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax0_1)
                                            v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax1_fused_1)
                                            vax2_fused_0, vax2_fused_2 = Ts.axis.remap("RR", [ax2_fused_0, ax2_fused_2])
                                            Ts.reads(C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1], A_pad_shared[v0, 0, vax2_fused_0 * 128 + vax2_fused_1_ax2_fused_3_fused // 8 * 32 + vax2_fused_2 * 8 + vax2_fused_1_ax2_fused_3_fused % 8], B_local[vax2_fused_0 * 128 + vax2_fused_1_ax2_fused_3_fused // 8 * 32 + vax2_fused_2 * 8 + vax2_fused_1_ax2_fused_3_fused % 8, v1])
                                            Ts.writes(C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1])
                                            C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1] = C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused, v0, 0, v1] + A_pad_shared[v0, 0, vax2_fused_0 * 128 + vax2_fused_1_ax2_fused_3_fused // 8 * 32 + vax2_fused_2 * 8 + vax2_fused_1_ax2_fused_3_fused % 8] * B_local[vax2_fused_0 * 128 + vax2_fused_1_ax2_fused_3_fused // 8 * 32 + vax2_fused_2 * 8 + vax2_fused_1_ax2_fused_3_fused % 8, v1]
                for ax3 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0 in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                        for ax2_init in range(4):
                            with Ts.sblock("matmul_rf_init"):
                                vax2_fused_1_ax2_fused_3_fused_0 = Ts.axis.spatial(4, ax0)
                                v0 = Ts.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax2_init)
                                v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax3)
                                Ts.reads()
                                Ts.writes(C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1])
                                C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1] = T.float16(0)
                        for ax2, ax1 in T.grid(4, 8):
                            with Ts.sblock("matmul_rf_update"):
                                vax2_fused_1_ax2_fused_3_fused_0, vax2_fused_1_ax2_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                v0 = Ts.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax2)
                                v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax3)
                                Ts.reads(C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1], C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused_0 * 8 + vax2_fused_1_ax2_fused_3_fused_1, v0, 0, v1])
                                Ts.writes(C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1])
                                C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1] = C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1] + C_pad_rf_local[vax2_fused_1_ax2_fused_3_fused_0 * 8 + vax2_fused_1_ax2_fused_3_fused_1, v0, 0, v1]
                for ax1 in range(4):
                    for ax2 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                        for ax0 in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                            with Ts.sblock("matmul"):
                                vax2_fused_1_ax2_fused_3_fused_0 = Ts.axis.reduce(4, ax0)
                                v0 = Ts.axis.spatial((batch_size + 3) // 4 * 4, ax0_0 * 4 + ax1)
                                v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax2)
                                Ts.reads(C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1])
                                Ts.writes(C_pad_local[v0, 0, v1])
                                with Ts.init():
                                    C_pad_local[v0, 0, v1] = T.float16(0)
                                C_pad_local[v0, 0, v1] = C_pad_local[v0, 0, v1] + C_pad_rf_local_1[vax2_fused_1_ax2_fused_3_fused_0, v0, 0, v1]
                for ax0 in range(4):
                    for ax1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("C_pad"):
                            v0 = Ts.axis.spatial(batch_size, ax0_0 * 4 + ax0)
                            v1 = Ts.axis.spatial(6144, ax1_fused_0 * 64 + ax1)
                            Ts.where((ax0_0 - (batch_size + 3) // 4 < 0 or ax0_0 * 4 + ax0 == 0) and ax0_0 * 4 + ax0 < batch_size)
                            Ts.reads(C_pad_local[v0, 0, v1])
                            Ts.writes(C[v0, 0, v1])
                            C[v0, 0, v1] = C_pad_local[v0, 0, v1]
    # fmt: on
    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4), target=Target("metal"))(mod)  # pylint: disable=not-callable
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_low_batch_gemv_cuda_target_without_max_shared_memory_per_block():
    # fmt: off
    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def before(A: T.Tensor((batch_size, T.int64(1), T.int64(128)), 'float16'), B: T.Tensor((T.int64(128), T.int64(128)), "float16"), C: T.Tensor((batch_size, T.int64(1), T.int64(128)), 'float16')):
        T.func_attr({"tir.noalias": True})

        for i0, i1, i2, k in T.grid(batch_size, T.int64(1), T.int64(128), T.int64(128)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                Ts.writes(C[v_i0, v_i1, v_i2])
                with Ts.init():
                    C[v_i0, v_i1, v_i2] = T.float16(0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]
    # fmt: on

    target = Target({"kind": "cuda", "max_num_threads": 1024})
    assert target.attrs.get("max_shared_memory_per_block", None) is None

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.gpu.LowBatchGEMV(4), target=target)(mod)
    assert mod["main"].attrs["tirx.is_scheduled"] == 1


def test_low_batch_gemv_rejects_non_einsum_tensor_access():
    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def before(
        A: T.Tensor((batch_size, 8), "float16"),
        B: T.Tensor((4, batch_size + 8), "float16"),
        C: T.Tensor((batch_size, 4), "float16"),
    ):
        for i, j, k in T.grid(batch_size, 4, 8):
            with Ts.sblock("attention_score"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                Ts.reads(A[vi, vk], B[vj, T.max(vi + vk - 7, 0)])
                Ts.writes(C[vi, vj])
                with Ts.init():
                    C[vi, vj] = T.float16(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, T.max(vi + vk - 7, 0)]

    target = Target("webgpu")
    result = dl.gpu.LowBatchGEMV(4).apply(before, target, False)
    assert result is None


def test_low_batch_gemv_broadcast_epilogue():
    # fmt: off
    batch_size = T.dynamic("batch_size", "int64")

    @Ts.function(private=True)
    def before(
        A: T.Tensor((T.int64(1), batch_size, T.int64(1), T.int64(128)), 'float16'),
        B: T.Tensor((T.int64(128), T.int64(128)), "float16"),
        C: T.Tensor((T.int64(1), batch_size, T.int64(2), T.int64(3), T.int64(128)), 'float32'),
    ):
        T.func_attr({"tirx.noalias": True})

        C_temp = Ts.sblock_alloc_tensor((T.int64(1), batch_size, T.int64(1), T.int64(128)), "float16")
        for i0, i1, i2, i3, k in T.grid(
            T.int64(1), batch_size, T.int64(1), T.int64(128), T.int64(128)
        ):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i3, v_k])
                Ts.writes(C_temp[v_i0, v_i1, v_i2, v_i3])
                with Ts.init():
                    C_temp[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                C_temp[v_i0, v_i1, v_i2, v_i3] = (
                    C_temp[v_i0, v_i1, v_i2, v_i3]
                    + A[v_i0, v_i1, v_i2, v_k] * B[v_i3, v_k]
                )
        for i0, i1, i2, i3, i4 in T.grid(
            T.int64(1), batch_size, T.int64(2), T.int64(3), T.int64(128)
        ):
            with Ts.sblock("broadcast_epilogue"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(C_temp[v_i0, v_i1, T.int64(0), v_i4])
                Ts.writes(C[v_i0, v_i1, v_i2, v_i3, v_i4])
                C[v_i0, v_i1, v_i2, v_i3, v_i4] = T.Cast(
                    "float32", C_temp[v_i0, v_i1, T.int64(0), v_i4]
                )
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with mock.patch.object(low_batch_gemv, "is_broadcast_epilogue", return_value=True) as check:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.LowBatchGEMV(4), target=Target("nvidia/geforce-rtx-3090-ti")
        )(mod)

    check.assert_called_once()
    assert mod["main"].attrs["tirx.is_scheduled"] == 1


if __name__ == "__main__":
    tvm_tirx.testing.main()
