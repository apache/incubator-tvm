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
# ruff: noqa: E501, F841

from __future__ import annotations

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def test_gemv_rejects_composite_normalized_axis():
    @Ts.function(private=True)
    def before(
        data: T.Tensor((1, 64, n), "float32"),  # noqa: F821
        weight: T.Tensor((64, 1, 512), "float32"),
        output: T.Tensor((1, 1, n * 256), "float32"),  # noqa: F821
        n: T.int64,
    ):
        for w, rc, rw in T.grid(n * 256, 64, 512):
            with Ts.sblock("conv1d_transpose"):
                vw, vrc, vrw = Ts.axis.remap("SRR", [w, rc, rw])
                Ts.reads(data[0, vrc, (vw + vrw - 383) // 256], weight[vrc, 0, 511 - vrw])
                Ts.writes(output[0, 0, vw])
                with Ts.init():
                    output[0, 0, vw] = T.float32(0)
                output[0, 0, vw] += (
                    data[0, vrc, (vw + vrw - 383) // 256] * weight[vrc, 0, 511 - vrw]
                )

    mod = tvm_tirx.IRModule({"main": before})
    with Target("webgpu"):
        scheduled = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm_tirx.ir.assert_structural_equal(scheduled["main"], before)


def test_gemv_basic():
    # fmt: off
    n = T.dynamic("n", "int32")

    @Ts.function(private=True)
    def before(lv1637: T.Tensor((1, 32, 1, 128), "float16"), lv1638: T.Tensor((1, 32, n, 128), 'float16'), lv1614: T.Tensor((1, 1, 1, n), 'float16'), var_compute_intermediate: T.Tensor((1, 32, 1, n), "float32")):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        var_NT_matmul_intermediate = Ts.sblock_alloc_tensor((1, 32, 1, n), "float16")
        var_T_divide_intermediate = Ts.sblock_alloc_tensor((1, 32, 1, n), "float16")
        var_T_maximum_intermediate = Ts.sblock_alloc_tensor((1, 32, 1, n), "float16")
        var_T_minimum_intermediate = Ts.sblock_alloc_tensor((1, 32, 1, n), "float16")
        for i0, i1, i2, i3, k in T.grid(1, 32, 1, n, 128):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(lv1637[v_i0, v_i1, v_i2, v_k], lv1638[v_i0, v_i1, v_i3, v_k])
                Ts.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
                with Ts.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv1637[v_i0, v_i1, v_i2, v_k] * lv1638[v_i0, v_i1, v_i3, v_k]
        for ax0, ax1, ax2, ax3 in T.grid(1, 32, 1, n):
            with Ts.sblock("T_divide"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                Ts.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
        for ax0, ax1, ax2, ax3 in T.grid(1, 32, 1, n):
            with Ts.sblock("T_maximum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                Ts.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
        for ax0, ax1, ax2, ax3 in T.grid(1, 32, 1, n):
            with Ts.sblock("T_minimum"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1614[v_ax0, 0, v_ax2, v_ax3])
                Ts.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1614[v_ax0, 0, v_ax2, v_ax3])
        for i0, i1, i2, i3 in T.grid(1, 32, 1, n):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
                Ts.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])

    n = T.dynamic("n", "int32")

    @Ts.function(private=True)
    def expected(lv1637: T.Tensor((1, 32, 1, 128), "float16"), lv1638: T.Tensor((1, 32, n, 128), 'float16'), lv1614: T.Tensor((1, 1, 1, n), 'float16'), var_compute_intermediate: T.Tensor((1, 32, 1, n), "float32")):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        var_NT_matmul_intermediate_local = Ts.sblock_alloc_tensor((1, 32, 1, n), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((128, 1, 32, 1, n), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = Ts.sblock_alloc_tensor((64, 1, 32, 1, n), "float16", scope="local")
        lv1638_local = Ts.sblock_alloc_tensor((1, 32, n, 128), "float16", scope="local")
        lv1637_shared = Ts.sblock_alloc_tensor((1, 32, 1, 128), "float16", scope="shared")
        for ax0_fused_ax1_fused_fused_0 in T.parallel(n * 32, annotations={"thread_binding": "blockIdx.x"}):
            for ax0_fused_ax1_fused_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.y"}):
                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0, ax1, ax2 in T.grid(1, 1, 1):
                        for ax3_0 in T.serial(1, annotations={"tirx.unroll_explicit": 256}):
                            for ax3_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.y"}):
                                for ax3_2 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                                    for ax3_3 in T.vectorized(2):
                                        with Ts.sblock("lv1637_shared"):
                                            v0 = Ts.axis.spatial(1, ax0)
                                            v1 = Ts.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n + ax1)
                                            v2 = Ts.axis.spatial(1, ax2)
                                            v3 = Ts.axis.spatial(128, ax3_0 * 128 + ax3_1 * 128 + ax3_2 * 2 + ax3_3)
                                            Ts.reads(lv1637[v0, v1, v2, v3])
                                            Ts.writes(lv1637_shared[v0, v1, v2, v3])
                                            lv1637_shared[v0, v1, v2, v3] = lv1637[v0, v1, v2, v3]
                    for ax0_fused_ax1_fused_fused_2_init in range(1):
                        for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(2):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = Ts.axis.spatial(128, ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                v0 = Ts.axis.spatial(32, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2_init) // n)
                                v1 = Ts.axis.spatial(n, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2_init) % n)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1])
                                var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1] = T.float16(0)
                    for ax2_fused_u_fused_0 in T.serial(1, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax0, ax1, ax2_ax3_fused_0 in T.grid(1, 1, 1):
                            for ax2_ax3_fused_1 in T.vectorized(2):
                                with Ts.sblock("lv1638_local"):
                                    v0 = Ts.axis.spatial(1, ax0)
                                    v1 = Ts.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n + ax1)
                                    v2 = Ts.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                                    v3 = Ts.axis.spatial(128, ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + ax2_ax3_fused_0 * 2 + ax2_ax3_fused_1)
                                    Ts.reads(lv1638[v0, v1, v2, v3])
                                    Ts.writes(lv1638_local[v0, v1, v2, v3])
                                    lv1638_local[v0, v1, v2, v3] = lv1638[v0, v1, v2, v3]
                        for ax0_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(1, 1):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(2):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = Ts.axis.spatial(128, ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                    v0 = Ts.axis.spatial(32, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2) // n)
                                    v1 = Ts.axis.spatial(n, (ax0_fused_ax1_fused_fused_0 + ax0_fused_ax1_fused_fused_1 + ax0_fused_ax1_fused_fused_2) % n)
                                    vax2_fused_u_fused_2, vax2_fused_u_fused_0 = Ts.axis.remap("RR", [ax2_fused_u_fused_2, ax2_fused_u_fused_0])
                                    Ts.reads(var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1], lv1637_shared[0, v0, 0, vax2_fused_u_fused_0 * 128 + vax2_fused_u_fused_2 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused], lv1638_local[0, v0, v1, vax2_fused_u_fused_0 * 128 + vax2_fused_u_fused_2 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1])
                                    var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1] = var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, 0, v0, 0, v1] + lv1637_shared[0, v0, 0, vax2_fused_u_fused_0 * 128 + vax2_fused_u_fused_2 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused] * lv1638_local[0, v0, v1, vax2_fused_u_fused_0 * 128 + vax2_fused_u_fused_2 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused]
            for ax2_ax3_fused_0 in T.parallel(1, annotations={"thread_binding": "threadIdx.y"}):
                for ax0 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_ax3_fused_1_0 in T.serial(1, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax2_ax3_fused_1_1 in T.vectorized(1):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = Ts.axis.spatial(64, ax0)
                                v0 = Ts.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n)
                                v1 = Ts.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1])
                                var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1] = T.float16(0)
                            for ax1 in range(2):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                    v0 = Ts.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n)
                                    v1 = Ts.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                                    Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1], var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, 0, v0, 0, v1])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1])
                                    var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1] = var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1] + var_NT_matmul_intermediate_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * 2 + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, 0, v0, 0, v1]
            for ax1_ax2_fused_1 in range(1):
                for ax1_ax2_fused_0 in T.parallel(1, annotations={"thread_binding": "threadIdx.y"}):
                    for ax0 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("NT_matmul"):
                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = Ts.axis.reduce(64, ax0)
                            v0 = Ts.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n)
                            v1 = Ts.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                            Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1])
                            Ts.writes(var_NT_matmul_intermediate_local[0, v0, 0, v1])
                            with Ts.init():
                                var_NT_matmul_intermediate_local[0, v0, 0, v1] = T.float16(0)
                            var_NT_matmul_intermediate_local[0, v0, 0, v1] = var_NT_matmul_intermediate_local[0, v0, 0, v1] + var_NT_matmul_intermediate_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, 0, v0, 0, v1]
            for ax0_ax1_fused_0 in T.parallel(1, annotations={"thread_binding": "threadIdx.y"}):
                for ax0_ax1_fused_1 in range(1):
                    with Ts.sblock("compute"):
                        v0 = Ts.axis.spatial(32, ax0_fused_ax1_fused_fused_0 // n)
                        v1 = Ts.axis.spatial(n, ax0_fused_ax1_fused_fused_0 % n)
                        Ts.reads(var_NT_matmul_intermediate_local[0, v0, 0, v1], lv1614[0, 0, 0, v1])
                        Ts.writes(var_compute_intermediate[0, v0, 0, v1])
                        var_compute_intermediate[0, v0, 0, v1] = T.Cast("float32", T.min(T.max(var_NT_matmul_intermediate_local[0, v0, 0, v1] * T.float16(0.088397790055248615), T.float16(-65504)), lv1614[0, 0, 0, v1]))

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_decode_gemv_256_threads():
    # fmt: off
    @Ts.function(private=True)
    def before(lv571: T.Tensor((22016, 512), "uint32"), lv572: T.Tensor((22016, 128), "float16"), lv1654: T.Tensor((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Tensor((1, 1, 22016), "float16")):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        p_output0_intermediate = Ts.sblock_alloc_tensor((22016, 4096), "float16")
        for i, j in T.grid(22016, 4096):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv571[v_i, v_j // 8], lv572[v_i, v_j // 32])
                Ts.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v_i, v_j // 32]
        for i0, i1, i2, k in T.grid(1, 1, 22016, 4096):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                Ts.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @Ts.function(private=True)
    def expected(lv571: T.Tensor((22016, 512), "uint32"), lv572: T.Tensor((22016, 128), "float16"), lv1654: T.Tensor((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Tensor((1, 1, 22016), "float16")):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        # with Ts.sblock("root"):
        var_NT_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((16, 1, 1, 22016), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = Ts.sblock_alloc_tensor((16, 1, 1, 22016), "float16", scope="local")
        lv571_local = Ts.sblock_alloc_tensor((22016, 512), "uint32", scope="local")
        for u_fused_ax0_fused_fused_0 in T.parallel(5504, annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_ax0_fused_fused_1 in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                    for u_fused_ax0_fused_fused_2_init in range(1):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(1):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(16, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = T.float16(0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(32, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax0_ax1_fused in T.serial(1):
                            for ax0_1 in T.vectorized(1):
                                with Ts.sblock("lv571_local"):
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1)
                                    v1 = Ts.axis.spatial(512, ax1_0_fused_ax1_1_fused_0 * 16 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0)
                                    Ts.reads(lv571[v0, v1])
                                    Ts.writes(lv571_local[v0, v1])
                                    lv571_local[v0, v1] = lv571[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(1, 8):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(1):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(16, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = Ts.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    Ts.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0], lv1654[0, 0, vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2], lv571_local[v0, vax1_0_fused_ax1_1_fused_0 * 16 + vax1_0_fused_ax1_1_fused_2 // 8 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused], lv572[v0, (vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2) // 32])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] + lv1654[0, 0, vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv571_local[v0, vax1_0_fused_ax1_1_fused_0 * 16 + vax1_0_fused_ax1_1_fused_2 // 8 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v0, (vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2) // 32])
            for ax2_fused_0 in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                    for ax2_fused_1_0 in T.serial(1, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax2_fused_1_1 in T.vectorized(1):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.spatial(16, ax0)
                                v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = T.float16(0)
                            for ax1 in range(1):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                    Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0], var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0]
            for ax1_fused_1 in range(1):
                for ax1_fused_0 in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                        with Ts.sblock("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.reduce(16, ax0)
                            v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 4 + ax1_fused_0 + ax1_fused_1)
                            Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                            Ts.writes(var_NT_matmul_intermediate[0, 0, v0])
                            with Ts.init():
                                var_NT_matmul_intermediate[0, 0, v0] = T.float16(0)
                            var_NT_matmul_intermediate[0, 0, v0] = var_NT_matmul_intermediate[0, 0, v0] + var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0]

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with Target("apple/m1-gpu-restricted"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_decode_gemv_webgpu():
    # fmt: off
    @Ts.function(private=True)
    def before(lv571: T.Tensor((22016, 512), "uint32"), lv572: T.Tensor((22016, 128), "float16"), lv1654: T.Tensor((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Tensor((1, 1, 22016), "float16")):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        p_output0_intermediate = Ts.sblock_alloc_tensor((22016, 4096), "float16")
        for i, j in T.grid(22016, 4096):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv571[v_i, v_j // 8], lv572[v_i, v_j // 32])
                Ts.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v_i, v_j // 32]
        for i0, i1, i2, k in T.grid(1, 1, 22016, 4096):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                Ts.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @Ts.function(private=True)
    def expected(lv571: T.Tensor((22016, 512), "uint32"), lv572: T.Tensor((22016, 128), "float16"), lv1654: T.Tensor((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Tensor((1, 1, 22016), "float16")):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        # with Ts.sblock("root"):
        var_NT_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((16, 1, 1, 22016), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = Ts.sblock_alloc_tensor((16, 1, 1, 22016), "float16", scope="local")
        lv571_local = Ts.sblock_alloc_tensor((22016, 512), "uint32", scope="local")
        for u_fused_ax0_fused_fused_0 in T.parallel(2752, annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_ax0_fused_fused_1 in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for u_fused_ax0_fused_fused_2_init in range(1):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(1):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(16, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 8 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = T.float16(0.0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(32, annotations={"auto_unroll_max_step": 64, "unroll_explicit": 1}):
                        for ax0_ax1_fused_0 in range(1):
                            for ax0_ax1_fused_1 in T.vectorized(1):
                                with Ts.sblock("lv571_local"):
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 8 + u_fused_ax0_fused_fused_1)
                                    v1 = Ts.axis.spatial(512, ax1_0_fused_ax1_1_fused_0 * 16 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0)
                                    Ts.reads(lv571[v0, v1])
                                    Ts.writes(lv571_local[v0, v1])
                                    lv571_local[v0, v1] = lv571[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(1, 8):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(1):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(16, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 8 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = Ts.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    Ts.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0], lv1654[0, 0, vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2], lv571_local[v0, vax1_0_fused_ax1_1_fused_0 * 16 + vax1_0_fused_ax1_1_fused_2 // 8 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused], lv572[v0, (vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2) // 32])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] + lv1654[0, 0, vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2] * ((T.Cast("float16", lv571_local[v0, vax1_0_fused_ax1_1_fused_0 * 16 + vax1_0_fused_ax1_1_fused_2 // 8 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused] >> T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2) % 8) * T.uint32(4) & T.uint32(15)) - T.float16(7.0)) * lv572[v0, (vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused * 8 + vax1_0_fused_ax1_1_fused_2) // 32])
            for ax2_fused_0_ax2_fused_1_fused in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_2_0 in T.serial(1, annotations={"auto_unroll_max_step": 64, "unroll_explicit": 1}):
                        for ax2_fused_2_1 in T.vectorized(1):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.spatial(16, ax0)
                                v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 8 + ax2_fused_0_ax2_fused_1_fused + ax2_fused_2_0 + ax2_fused_2_1)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = T.float16(0.0)
                            for ax1 in range(1):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 8 + ax2_fused_0_ax2_fused_1_fused + ax2_fused_2_0 + ax2_fused_2_1)
                                    Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0], var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0]
            for ax1_fused_2 in range(1):
                for ax1_fused_0_ax1_fused_1_fused in T.parallel(8, annotations={"thread_binding": "threadIdx.y"}):
                    for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.reduce(16, ax0)
                            v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 8 + ax1_fused_0_ax1_fused_1_fused + ax1_fused_2)
                            Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                            Ts.writes(var_NT_matmul_intermediate[0, 0, v0])
                            with Ts.init():
                                var_NT_matmul_intermediate[0, 0, v0] = T.float16(0.0)
                            var_NT_matmul_intermediate[0, 0, v0] = var_NT_matmul_intermediate[0, 0, v0] + var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0]
    # fmt: on

    mod = tvm.IRModule({"main": before})
    with Target("webgpu"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_decode_gemv1():
    # fmt: off

    @Ts.function(private=True)
    def before(lv571: T.Tensor((22016, 512), "uint32"), lv572: T.Tensor((22016, 128), "float16"), lv1654: T.Tensor((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Tensor((1, 1, 22016), "float16")):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        p_output0_intermediate = Ts.sblock_alloc_tensor((22016, 4096), "float16")
        for i, j in T.grid(22016, 4096):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv571[v_i, v_j // 8], lv572[v_i, v_j // 32])
                Ts.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v_i, v_j // 32]
        for i0, i1, i2, k in T.grid(1, 1, 22016, 4096):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                Ts.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @Ts.function(private=True)
    def expected(lv571: T.Tensor((22016, 512), "uint32"), lv572: T.Tensor((22016, 128), "float16"), lv1654: T.Tensor((1, 1, 4096), "float16"), var_NT_matmul_intermediate: T.Tensor((1, 1, 22016), "float16")):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        # with Ts.sblock("root"):
        var_NT_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((128, 1, 1, 22016), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = Ts.sblock_alloc_tensor((32, 1, 1, 22016), "float16", scope="local")
        lv571_local = Ts.sblock_alloc_tensor((22016, 512), "uint32", scope="local")
        lv1654_shared = Ts.sblock_alloc_tensor((1, 1, 4096), "float16", scope="shared")
        for u_fused_ax0_fused_fused_0 in T.parallel(1376, annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_ax0_fused_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0, ax1 in T.grid(1, 1):
                        for ax2_0 in T.serial(1, annotations={"tirx.unroll_explicit": 256}):
                            for ax2_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                                for ax2_2 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                                    for ax2_3 in T.vectorized(8):
                                        with Ts.sblock("lv1654_shared"):
                                            v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                                            v2 = Ts.axis.spatial(4096, ax2_0 * 4096 + ax2_1 * 256 + ax2_2 * 8 + ax2_3)
                                            Ts.reads(lv1654[v0, v1, v2])
                                            Ts.writes(lv1654_shared[v0, v1, v2])
                                            lv1654_shared[v0, v1, v2] = lv1654[v0, v1, v2]
                    for u_fused_ax0_fused_fused_2_init in range(1):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(4):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(128, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = T.float16(0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(16, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax0_ax1_fused_0 in range(1):
                            for ax0_ax1_fused_1 in T.vectorized(1):
                                with Ts.sblock("lv571_local"):
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1)
                                    v1 = Ts.axis.spatial(512, ax1_0_fused_ax1_1_fused_0 * 32 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0)
                                    Ts.reads(lv571[v0, v1])
                                    Ts.writes(lv571_local[v0, v1])
                                    lv571_local[v0, v1] = lv571[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(1, 2):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(4):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(128, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = Ts.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    Ts.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0], lv1654_shared[0, 0, vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4], lv571_local[v0, vax1_0_fused_ax1_1_fused_0 * 32 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 + vax1_0_fused_ax1_1_fused_2 // 2], lv572[v0, (vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) // 32])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] + lv1654_shared[0, 0, vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv571_local[v0, vax1_0_fused_ax1_1_fused_0 * 32 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 + vax1_0_fused_ax1_1_fused_2 // 2], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv572[v0, (vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) // 32])
            for ax2_fused_0_ax2_fused_1_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                for ax0 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_2_0 in T.serial(1, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax2_fused_2_1 in T.vectorized(1):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.spatial(32, ax0)
                                v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 16 + ax2_fused_0_ax2_fused_1_fused + ax2_fused_2_0 + ax2_fused_2_1)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = T.float16(0)
                            for ax1 in range(4):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                    v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 16 + ax2_fused_0_ax2_fused_1_fused + ax2_fused_2_0 + ax2_fused_2_1)
                                    Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0], var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0]
            for ax1_fused_2 in range(1):
                for ax1_fused_0_ax1_fused_1_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                    for ax0 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.reduce(32, ax0)
                            v0 = Ts.axis.spatial(22016, u_fused_ax0_fused_fused_0 * 16 + ax1_fused_0_ax1_fused_1_fused + ax1_fused_2)
                            Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                            Ts.writes(var_NT_matmul_intermediate[0, 0, v0])
                            with Ts.init():
                                var_NT_matmul_intermediate[0, 0, v0] = T.float16(0)
                            var_NT_matmul_intermediate[0, 0, v0] = var_NT_matmul_intermediate[0, 0, v0] + var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0]

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_decode_gemv2():
    # fmt: off

    @Ts.function(private=True)
    def before(lv771: T.Tensor((32000, 512), "uint32"), lv772: T.Tensor((32000, 128), "float16"), lv3216: T.Tensor((1, 1, 4096), "float16"), p_output0_intermediate: T.Tensor((1, 1, 32000), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        p_output0_intermediate_1 = Ts.sblock_alloc_tensor((32000, 4096), "float16")
        var_NT_matmul_intermediate = Ts.sblock_alloc_tensor((1, 1, 32000), "float16")
        for i, j in T.grid(32000, 4096):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv771[v_i, v_j // 8], lv772[v_i, v_j // 32])
                Ts.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv771[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv772[v_i, v_j // 32]
        for i0, i1, i2, k in T.grid(1, 1, 32000, 4096):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv3216[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                Ts.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv3216[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for i0, i1, i2 in T.grid(1, 1, 32000):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                Ts.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

    @Ts.function(private=True)
    def expected(lv771: T.Tensor((32000, 512), "uint32"), lv772: T.Tensor((32000, 128), "float16"), lv3216: T.Tensor((1, 1, 4096), "float16"), p_output0_intermediate: T.Tensor((1, 1, 32000), "float32")):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        # with Ts.sblock("root"):
        var_NT_matmul_intermediate_local = Ts.sblock_alloc_tensor((1, 1, 32000), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((128, 1, 1, 32000), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = Ts.sblock_alloc_tensor((32, 1, 1, 32000), "float16", scope="local")
        lv771_local = Ts.sblock_alloc_tensor((32000, 512), "uint32", scope="local")
        lv3216_shared = Ts.sblock_alloc_tensor((1, 1, 4096), "float16", scope="shared")
        for u_fused_ax0_fused_fused_0 in T.parallel(2000, annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_ax0_fused_fused_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                    for ax0, ax1 in T.grid(1, 1):
                        for ax2_0 in T.serial(1, annotations={"tirx.unroll_explicit": 256}):
                            for ax2_1 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                                for ax2_2 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                                    for ax2_3 in T.vectorized(8):
                                        with Ts.sblock("lv3216_shared"):
                                            v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                                            v2 = Ts.axis.spatial(4096, ax2_0 * 4096 + ax2_1 * 256 + ax2_2 * 8 + ax2_3)
                                            Ts.reads(lv3216[v0, v1, v2])
                                            Ts.writes(lv3216_shared[v0, v1, v2])
                                            lv3216_shared[v0, v1, v2] = lv3216[v0, v1, v2]
                    for u_fused_ax0_fused_fused_2_init in range(1):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(4):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(128, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = Ts.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = T.float16(0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(16, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax0_ax1_fused_0 in range(1):
                            for ax0_ax1_fused_1 in T.vectorized(1):
                                with Ts.sblock("lv771_local"):
                                    v0 = Ts.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1)
                                    v1 = Ts.axis.spatial(512, ax1_0_fused_ax1_1_fused_0 * 32 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0)
                                    Ts.reads(lv771[v0, v1])
                                    Ts.writes(lv771_local[v0, v1])
                                    lv771_local[v0, v1] = lv771[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(1, 2):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(4):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(128, ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = Ts.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = Ts.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    Ts.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0], lv3216_shared[0, 0, vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4], lv771_local[v0, vax1_0_fused_ax1_1_fused_0 * 32 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 + vax1_0_fused_ax1_1_fused_2 // 2], lv772[v0, (vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) // 32])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, 0, 0, v0] + lv3216_shared[0, 0, vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv771_local[v0, vax1_0_fused_ax1_1_fused_0 * 32 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 + vax1_0_fused_ax1_1_fused_2 // 2], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv772[v0, (vax1_0_fused_ax1_1_fused_0 * 256 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // 4 * 8 + vax1_0_fused_ax1_1_fused_2 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % 4) // 32])
            for ax2_fused_0_ax2_fused_1_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                for ax0 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_2_0 in T.serial(1, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax2_fused_2_1 in T.vectorized(1):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.spatial(32, ax0)
                                v0 = Ts.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 16 + ax2_fused_0_ax2_fused_1_fused + ax2_fused_2_0 + ax2_fused_2_1)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = T.float16(0)
                            for ax1 in range(4):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                    v0 = Ts.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 16 + ax2_fused_0_ax2_fused_1_fused + ax2_fused_2_0 + ax2_fused_2_1)
                                    Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0], var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] = var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, 0, 0, v0]
            for ax1_fused_2 in range(1):
                for ax1_fused_0_ax1_fused_1_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                    for ax0 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.reduce(32, ax0)
                            v0 = Ts.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 16 + ax1_fused_0_ax1_fused_1_fused + ax1_fused_2)
                            Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0])
                            Ts.writes(var_NT_matmul_intermediate_local[0, 0, v0])
                            with Ts.init():
                                var_NT_matmul_intermediate_local[0, 0, v0] = T.float16(0)
                            var_NT_matmul_intermediate_local[0, 0, v0] = var_NT_matmul_intermediate_local[0, 0, v0] + var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, 0, 0, v0]
            for ax0_fused_0_ax0_fused_1_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                for ax0_fused_2 in range(1):
                    with Ts.sblock("compute"):
                        v0 = Ts.axis.spatial(32000, u_fused_ax0_fused_fused_0 * 16 + ax0_fused_0_ax0_fused_1_fused + ax0_fused_2)
                        Ts.reads(var_NT_matmul_intermediate_local[0, 0, v0])
                        Ts.writes(p_output0_intermediate[0, 0, v0])
                        p_output0_intermediate[0, 0, v0] = T.Cast("float32", var_NT_matmul_intermediate_local[0, 0, v0])

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_decode_gemv3():
    # fmt: off

    @Ts.function(private=True)
    def before(lv575: T.Tensor((T.int64(4096), T.int64(1376)), "uint32"), lv576: T.Tensor((T.int64(4096), T.int64(344)), "float16"), lv574: T.Tensor((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv570: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        p_output0_intermediate_1 = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(11008)), "float16")
        var_NT_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(4096), T.int64(11008)):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv575[v_i, v_j // T.int64(8)], lv576[v_i, v_j // T.int64(32)])
                Ts.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv575[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv574[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                Ts.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv574[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(lv570[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                Ts.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv570[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @Ts.function(private=True)
    def expected(lv575: T.Tensor((T.int64(4096), T.int64(1376)), "uint32"), lv576: T.Tensor((T.int64(4096), T.int64(344)), "float16"), lv574: T.Tensor((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv570: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        # with Ts.sblock("root"):
        var_NT_matmul_intermediate_local = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((T.int64(128), T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
        var_NT_matmul_intermediate_rf_local_1 = Ts.sblock_alloc_tensor((T.int64(32), T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
        lv575_local = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(1376)), "uint32", scope="local")
        lv574_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(11008)), "float16", scope="shared")
        for u_fused_ax0_fused_fused_0 in T.parallel(T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_ax0_fused_fused_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                    for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                        for ax2_0 in T.serial(T.int64(22), annotations={"tirx.unroll_explicit": 256}):
                            for ax2_1 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                                for ax2_2 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                                    for ax2_3 in T.vectorized(T.int64(1)):
                                        with Ts.sblock("lv574_shared"):
                                            v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                                            v2 = Ts.axis.spatial(T.int64(11008), ax2_0 * T.int64(512) + ax2_1 * T.int64(32) + ax2_2 + ax2_3)
                                            Ts.where((ax2_0 * T.int64(16) + ax2_1) * T.int64(32) + ax2_2 + ax2_3 < T.int64(11008))
                                            Ts.reads(lv574[v0, v1, v2])
                                            Ts.writes(lv574_shared[v0, v1, v2])
                                            lv574_shared[v0, v1, v2] = lv574[v0, v1, v2]
                    for u_fused_ax0_fused_fused_2_init in T.serial(T.int64(0), T.int64(1)):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(T.int64(128), ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                                v0 = Ts.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0])
                                var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] = T.float16(0)
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(T.int64(43), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax0_ax1_fused_0 in T.serial(T.int64(0), T.int64(1)):
                            for ax0_ax1_fused_1 in T.vectorized(T.int64(1)):
                                with Ts.sblock("lv575_local"):
                                    v0 = Ts.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1)
                                    v1 = Ts.axis.spatial(T.int64(1376), ax1_0_fused_ax1_1_fused_0 * T.int64(32) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0)
                                    Ts.reads(lv575[v0, v1])
                                    Ts.writes(lv575_local[v0, v1])
                                    lv575_local[v0, v1] = lv575[v0, v1]
                        for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(T.int64(1), T.int64(2)):
                            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = Ts.axis.spatial(T.int64(128), ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                    v0 = Ts.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = Ts.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                    Ts.reads(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0], lv574_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)], lv575_local[v0, vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) + vax1_0_fused_ax1_1_fused_2 // T.int64(2)], lv576[v0, (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) // T.int64(32)])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0])
                                    var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] = var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] + lv574_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv575_local[v0, vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) + vax1_0_fused_ax1_1_fused_2 // T.int64(2)], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576[v0, (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) // T.int64(32)])
            for ax2_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                for ax0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                    for ax2_fused_1_0 in T.serial(T.int64(1), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax2_fused_1_1 in T.vectorized(T.int64(1)):
                            with Ts.sblock("NT_matmul_rf_init"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.spatial(T.int64(32), ax0)
                                v0 = Ts.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                Ts.reads()
                                Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                                var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] = T.float16(0)
                            for ax1 in T.serial(T.int64(0), T.int64(4)):
                                with Ts.sblock("NT_matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                    v0 = Ts.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                    Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0], var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, T.int64(0), T.int64(0), v0])
                                    Ts.writes(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                                    var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] = var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] + var_NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, T.int64(0), T.int64(0), v0]
            for ax1_fused_1 in T.serial(T.int64(0), T.int64(1)):
                for ax1_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                    for ax0 in T.parallel(T.int64(32), annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("NT_matmul"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = Ts.axis.reduce(T.int64(32), ax0)
                            v0 = Ts.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax1_fused_0 + ax1_fused_1)
                            Ts.reads(var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                            Ts.writes(var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                            with Ts.init():
                                var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                            var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] = var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] + var_NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0]
            for ax0_fused_0 in T.parallel(T.int64(16), annotations={"thread_binding": "threadIdx.y"}):
                for ax0_fused_1 in T.serial(T.int64(0), T.int64(1)):
                    with Ts.sblock("T_add"):
                        v0 = Ts.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax0_fused_0 + ax0_fused_1)
                        Ts.reads(lv570[T.int64(0), T.int64(0), v0], var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        Ts.writes(p_output0_intermediate[T.int64(0), T.int64(0), v0])
                        p_output0_intermediate[T.int64(0), T.int64(0), v0] = lv570[T.int64(0), T.int64(0), v0] + var_NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0]

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_autogptq_decode_gemv():
    # fmt: off
    @Ts.function(private=True)
    def func(lv9: T.Tensor((T.int64(512), T.int64(4096)), "uint32"), lv10: T.Tensor((T.int64(32), T.int64(512)), "uint32"), lv11: T.Tensor((T.int64(32), T.int64(4096)), "float16"), lv12: T.Tensor((T.int64(4096),), "uint32"), lv8: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv1613: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        decode_intermediate = Ts.sblock_alloc_tensor((T.int64(4096), T.int64(4096)), "float16")
        var_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16")
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv9[v_i // T.int64(8), v_j], lv10[lv12[v_i], v_j // T.int64(8)], lv12[v_i], lv11[lv12[v_i], v_j])
                Ts.writes(decode_intermediate[v_i, v_j])
                decode_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv9[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8) * T.int64(4))), T.uint32(15))) - (T.Cast("float16", T.bitwise_and(T.shift_right(lv10[lv12[v_i], v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8) * T.int64(4))), T.uint32(15))) + T.float16(1))) * lv11[lv12[v_i], v_j]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv8[v_i0, v_i1, v_k], decode_intermediate[v_k, v_i2])
                Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv8[v_i0, v_i1, v_k] * decode_intermediate[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(lv1613[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                Ts.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv1613[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
    # fmt: on

    # The GeMV rule does not yet support the inner dim being grouped.
    # So the rule is expected to skip transforming this function.
    mod = tvm_tirx.IRModule({"main": func})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], func)


def test_outer_reduction_adreno():
    # fmt: off
    @Ts.function(private=True)
    def before(
        lv575: T.Tensor((1376, 4096), "uint32"),
        lv576: T.Tensor((344, 4096), "float16"),
        lv574: T.Tensor((1, 1, 11008), "float16"),
        lv570: T.Tensor((1, 1, 4096), "float16"),
        p_output0_intermediate: T.Tensor((1, 1, 4096), "float16"),
    ):
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        p_output0_intermediate_1 = Ts.sblock_alloc_tensor((11008, 4096), "float16")
        var_matmul_intermediate = Ts.sblock_alloc_tensor((1, 1, 4096), "float16")
        for i, j in T.grid(11008, 4096):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv575[v_i // 8, v_j], T.Cast("uint32", v_i % 8) * T.uint32(4)), T.uint32(15)))- T.float16(7)) * lv576[v_i // 32, v_j]
        for i0, i1, i2, k in T.grid(1, 1, 4096, 11008):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                with Ts.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv574[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(1, 1, 4096):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv570[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @Ts.function(private=True)
    def expected(lv575: T.Tensor((1376, 4096), "uint32"), lv576: T.Tensor((344, 4096), "float16"), lv574: T.Tensor((1, 1, 11008), "float16"), lv570: T.Tensor((1, 1, 4096), "float16"), p_output0_intermediate: T.Tensor((1, 1, 4096), "float16")):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})
        # with Ts.sblock("root"):
        var_matmul_intermediate_local = Ts.sblock_alloc_tensor((1, 1, 4096), "float16", scope="local")
        var_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((32, 1, 1, 4096), "float16", scope="local")
        var_matmul_intermediate_rf_local_1 = Ts.sblock_alloc_tensor((4, 1, 1, 4096), "float16", scope="local")
        lv576_local = Ts.sblock_alloc_tensor((344, 4096), "float16", scope="local")
        lv575_local = Ts.sblock_alloc_tensor((1376, 4096), "uint32", scope="local")
        for u_fused_ax0_fused_fused_0 in T.parallel(64, annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_ax0_fused_fused_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                for ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0_init in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                    for ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1_init in T.vectorized(8):
                        with Ts.sblock("matmul_rf_init"):
                            vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused = Ts.axis.spatial(32, ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0_init * 8 + ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1_init)
                            v0 = Ts.axis.spatial(4096, u_fused_ax0_fused_fused_0 * 64 + u_fused_ax0_fused_fused_1)
                            Ts.reads()
                            Ts.writes(var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, 0, 0, v0])
                            var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, 0, 0, v0] = T.float16(0)
                for ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                    for ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_1 in T.grid(86, 1):
                        for ax0, ax1 in T.grid(1, 1):
                            with Ts.sblock("lv576_local"):
                                v0 = Ts.axis.spatial(344, ax1_0_fused_ax1_1_fused_0 * 4 + ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 + ax0)
                                v1 = Ts.axis.spatial(4096, u_fused_ax0_fused_fused_0 * 64 + u_fused_ax0_fused_fused_1 + ax1)
                                Ts.reads(lv576[v0, v1])
                                Ts.writes(lv576_local[v0, v1])
                                lv576_local[v0, v1] = lv576[v0, v1]
                        for ax1_0_fused_ax1_1_fused_3 in range(4):
                            for ax0, ax1 in T.grid(1, 1):
                                with Ts.sblock("lv575_local"):
                                    v0 = Ts.axis.spatial(1376, ax1_0_fused_ax1_1_fused_0 * 16 + ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 * 4 + ax1_0_fused_ax1_1_fused_3 + ax0)
                                    v1 = Ts.axis.spatial(4096, u_fused_ax0_fused_fused_0 * 64 + u_fused_ax0_fused_fused_1 + ax1)
                                    Ts.reads(lv575[v0, v1])
                                    Ts.writes(lv575_local[v0, v1])
                                    lv575_local[v0, v1] = lv575[v0, v1]
                            for ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1 in T.vectorized(8):
                                with Ts.sblock("matmul_rf_update"):
                                    vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused = Ts.axis.spatial(32, ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 * 8 + ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1)
                                    v0 = Ts.axis.spatial(4096, u_fused_ax0_fused_fused_0 * 64 + u_fused_ax0_fused_fused_1)
                                    vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_1, vax1_0_fused_ax1_1_fused_3 = Ts.axis.remap("RRR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_1, ax1_0_fused_ax1_1_fused_3])
                                    Ts.reads(var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, 0, 0, v0], lv574[0, 0, vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1 * 128 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // 8 * 32 + vax1_0_fused_ax1_1_fused_3 * 8 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused % 8], lv575_local[vax1_0_fused_ax1_1_fused_0 * 16 + vax1_0_fused_ax1_1_fused_1 * 16 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // 8 * 4 + vax1_0_fused_ax1_1_fused_3, v0], lv576_local[vax1_0_fused_ax1_1_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1 * 4 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // 8 + vax1_0_fused_ax1_1_fused_3 // 4, v0])
                                    Ts.writes(var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, 0, 0, v0])
                                    var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, 0, 0, v0] = var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, 0, 0, v0] + lv574[0, 0, vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1 * 128 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // 8 * 32 + vax1_0_fused_ax1_1_fused_3 * 8 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused % 8] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv575_local[vax1_0_fused_ax1_1_fused_0 * 16 + vax1_0_fused_ax1_1_fused_1 * 16 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // 8 * 4 + vax1_0_fused_ax1_1_fused_3, v0], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * 128 + vax1_0_fused_ax1_1_fused_1 * 128 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // 8 * 32 + vax1_0_fused_ax1_1_fused_3 * 8 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused % 8) % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv576_local[vax1_0_fused_ax1_1_fused_0 * 4 + vax1_0_fused_ax1_1_fused_1 * 4 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // 8 + vax1_0_fused_ax1_1_fused_3 // 4, v0])
            for ax2 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                for ax0 in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                    with Ts.sblock("matmul_rf_init"):
                        vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 = Ts.axis.spatial(4, ax0)
                        v0 = Ts.axis.spatial(4096, u_fused_ax0_fused_fused_0 * 64 + ax2)
                        Ts.reads()
                        Ts.writes(var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, 0, 0, v0])
                        var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, 0, 0, v0] = T.float16(0)
                    for ax1 in T.serial(8, annotations={"tirx.auto_unroll_max_step": 8, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("matmul_rf_update"):
                            vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                            v0 = Ts.axis.spatial(4096, u_fused_ax0_fused_fused_0 * 64 + ax2)
                            Ts.reads(var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, 0, 0, v0], var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 * 8 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1, 0, 0, v0])
                            Ts.writes(var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, 0, 0, v0])
                            var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, 0, 0, v0] = var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, 0, 0, v0] + var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 * 8 + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1, 0, 0, v0]
            for ax1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                for ax0 in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                    with Ts.sblock("matmul"):
                        vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 = Ts.axis.reduce(4, ax0)
                        v0 = Ts.axis.spatial(4096, u_fused_ax0_fused_fused_0 * 64 + ax1)
                        Ts.reads(var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, 0, 0, v0])
                        Ts.writes(var_matmul_intermediate_local[0, 0, v0])
                        with Ts.init():
                            var_matmul_intermediate_local[0, 0, v0] = T.float16(0)
                        var_matmul_intermediate_local[0, 0, v0] = var_matmul_intermediate_local[0, 0, v0] + var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, 0, 0, v0]
            for ax0_fused_0 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                for ax0_fused_1 in range(1):
                    with Ts.sblock("T_add"):
                        v0 = Ts.axis.spatial(4096, u_fused_ax0_fused_fused_0 * 64 + ax0_fused_0 + ax0_fused_1)
                        Ts.reads(lv570[0, 0, v0], var_matmul_intermediate_local[0, 0, v0])
                        Ts.writes(p_output0_intermediate[0, 0, v0])
                        p_output0_intermediate[0, 0, v0] = lv570[0, 0, v0] + var_matmul_intermediate_local[0, 0, v0]
    # fmt: on
    mod = tvm_tirx.IRModule({"main": before})
    with Target("opencl", host={"kind": "llvm", "mtriple": "aarch64-linux-android"}):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_outer_reduction_adreno_dynamic():
    # fmt: off
    v = T.dynamic("v", "int64")

    @Ts.function(private=True)
    def before(lv612: T.Tensor((T.int64(512), v), 'uint32'), lv613: T.Tensor((T.int64(128), v), 'float16'), lv1607: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Tensor((T.int64(1), T.int64(1), v), "float32")):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        p_output0_intermediate_1 = Ts.sblock_alloc_tensor((T.int64(4096), v), "float16")
        var_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), v), "float16")
        for i, j in T.grid(T.int64(4096), v):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv612[v_i // T.int64(8), v_j], lv613[v_i // T.int64(32), v_j])
                Ts.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv612[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv613[v_i // T.int64(32), v_j]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), v, T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
                Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), v):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
                Ts.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1, v_i2])

    v = T.dynamic("v", "int64")

    @Ts.function(private=True)
    def expected(lv612: T.Tensor((T.int64(512), v), 'uint32'), lv613: T.Tensor((T.int64(128), v), 'float16'), lv1607: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Tensor((T.int64(1), T.int64(1), v), "float32")):
        T.func_attr({"tirx.is_scheduled": True, "tirx.noalias": True})

        # with Ts.sblock("root"):
        var_matmul_intermediate_local = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), v), "float16", scope="local")
        var_matmul_intermediate_rf_local = Ts.sblock_alloc_tensor((T.int64(8), T.int64(1), T.int64(1), v), "float16", scope="local")
        var_matmul_intermediate_rf_local_1 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(1), v), "float16", scope="local")
        lv613_local = Ts.sblock_alloc_tensor((T.int64(128), v), "float16", scope="local")
        lv612_local = Ts.sblock_alloc_tensor((T.int64(512), v), "uint32", scope="local")
        lv1607_shared = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="shared")
        for u_fused_ax0_fused_fused_0 in T.parallel((v + T.int64(255)) // T.int64(256), annotations={"thread_binding": "blockIdx.x"}):
            for u_fused_ax0_fused_fused_1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                for ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0_init in T.parallel(T.int64(1), annotations={"thread_binding": "threadIdx.y"}):
                    for ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1_init in T.vectorized(T.int64(8)):
                        with Ts.sblock("matmul_rf_init"):
                            vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused = Ts.axis.spatial(T.int64(8), ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0_init * T.int64(8) + ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1_init)
                            v0 = Ts.axis.spatial(v, u_fused_ax0_fused_fused_0 * T.int64(256) + u_fused_ax0_fused_fused_1)
                            Ts.where(u_fused_ax0_fused_fused_0 * T.int64(256) + u_fused_ax0_fused_fused_1 < v)
                            Ts.reads()
                            Ts.writes(var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, T.int64(0), T.int64(0), v0])
                            var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, T.int64(0), T.int64(0), v0] = T.float16(0)
                for ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "threadIdx.y"}):
                    for ax1_0_fused_ax1_1_fused_0 in T.serial(T.int64(0), T.int64(128)):
                        for ax0, ax1, ax2_0, ax2_1 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                            for ax2_2 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                                for ax2_3 in T.parallel(T.int64(1), annotations={"thread_binding": "threadIdx.y"}):
                                    for ax2_4 in T.vectorized(T.int64(4)):
                                        with Ts.sblock("lv1607_shared"):
                                            v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                                            v2 = Ts.axis.spatial(T.int64(4096), ax1_0_fused_ax1_1_fused_0 * T.int64(32) + (ax2_0 * T.int64(1024) + ax2_1 * T.int64(1024) + ax2_2 * T.int64(4) + ax2_3 * T.int64(4) + ax2_4))
                                            Ts.where(((ax2_0 + ax2_1) * T.int64(256) + ax2_2 + ax2_3) * T.int64(4) + ax2_4 < T.int64(32))
                                            Ts.reads(lv1607[v0, v1, v2])
                                            Ts.writes(lv1607_shared[v0, v1, v2])
                                            lv1607_shared[v0, v1, v2] = lv1607[v0, v1, v2]
                        for ax1_0_fused_ax1_1_fused_1 in T.serial(T.int64(0), T.int64(1)):
                            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                                with Ts.sblock("lv613_local"):
                                    v0 = Ts.axis.spatial(T.int64(128), ax1_0_fused_ax1_1_fused_0 + ax0)
                                    v1 = Ts.axis.spatial(v, u_fused_ax0_fused_fused_0 * T.int64(256) + u_fused_ax0_fused_fused_1 + ax1)
                                    Ts.where(u_fused_ax0_fused_fused_0 * T.int64(256) + u_fused_ax0_fused_fused_1 < v)
                                    Ts.reads(lv613[v0, v1])
                                    Ts.writes(lv613_local[v0, v1])
                                    lv613_local[v0, v1] = lv613[v0, v1]
                            for ax1_0_fused_ax1_1_fused_3 in T.serial(T.int64(0), T.int64(4)):
                                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                                    with Ts.sblock("lv612_local"):
                                        v0 = Ts.axis.spatial(T.int64(512), ax1_0_fused_ax1_1_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_3 + ax0)
                                        v1 = Ts.axis.spatial(v, u_fused_ax0_fused_fused_0 * T.int64(256) + u_fused_ax0_fused_fused_1 + ax1)
                                        Ts.where(u_fused_ax0_fused_fused_0 * T.int64(256) + u_fused_ax0_fused_fused_1 < v)
                                        Ts.reads(lv612[v0, v1])
                                        Ts.writes(lv612_local[v0, v1])
                                        lv612_local[v0, v1] = lv612[v0, v1]
                                for ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1 in T.vectorized(T.int64(8)):
                                    with Ts.sblock("matmul_rf_update"):
                                        vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused = Ts.axis.spatial(T.int64(8), ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 * T.int64(8) + ax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1)
                                        v0 = Ts.axis.spatial(v, u_fused_ax0_fused_fused_0 * T.int64(256) + u_fused_ax0_fused_fused_1)
                                        vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_1, vax1_0_fused_ax1_1_fused_3 = Ts.axis.remap("RRR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_1, ax1_0_fused_ax1_1_fused_3])
                                        Ts.where(u_fused_ax0_fused_fused_0 * T.int64(256) + u_fused_ax0_fused_fused_1 < v)
                                        Ts.reads(var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, T.int64(0), T.int64(0), v0], lv1607_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1 * T.int64(32) + vax1_0_fused_ax1_1_fused_3 * T.int64(8) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused], lv612_local[vax1_0_fused_ax1_1_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1 * T.int64(4) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // T.int64(8) + vax1_0_fused_ax1_1_fused_3, v0], lv613_local[(vax1_0_fused_ax1_1_fused_3 * T.int64(8) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused) // T.int64(32) + vax1_0_fused_ax1_1_fused_0 + vax1_0_fused_ax1_1_fused_1, v0])
                                        Ts.writes(var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, T.int64(0), T.int64(0), v0])
                                        var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, T.int64(0), T.int64(0), v0] = var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused, T.int64(0), T.int64(0), v0] + lv1607_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1 * T.int64(32) + vax1_0_fused_ax1_1_fused_3 * T.int64(8) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv612_local[vax1_0_fused_ax1_1_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1 * T.int64(4) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused // T.int64(8) + vax1_0_fused_ax1_1_fused_3, v0], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1 * T.int64(32) + vax1_0_fused_ax1_1_fused_3 * T.int64(8) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused) % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv613_local[(vax1_0_fused_ax1_1_fused_3 * T.int64(8) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused) // T.int64(32) + vax1_0_fused_ax1_1_fused_0 + vax1_0_fused_ax1_1_fused_1, v0])
            for ax2 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                for ax0 in T.parallel(T.int64(1), annotations={"thread_binding": "threadIdx.y"}):
                    with Ts.sblock("matmul_rf_init"):
                        vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 = Ts.axis.spatial(T.int64(1), ax0)
                        v0 = Ts.axis.spatial(v, u_fused_ax0_fused_fused_0 * T.int64(256) + ax2)
                        Ts.where(u_fused_ax0_fused_fused_0 * T.int64(256) + ax2 < v)
                        Ts.reads()
                        Ts.writes(var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, T.int64(0), T.int64(0), v0])
                        var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, T.int64(0), T.int64(0), v0] = T.float16(0)
                    for ax1 in T.serial(T.int64(8), annotations={"tirx.auto_unroll_max_step": 8, "tirx.unroll_explicit": 1}):
                        with Ts.sblock("matmul_rf_update"):
                            vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                            v0 = Ts.axis.spatial(v, u_fused_ax0_fused_fused_0 * T.int64(256) + ax2)
                            Ts.where(u_fused_ax0_fused_fused_0 * T.int64(256) + ax2 < v)
                            Ts.reads(var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, T.int64(0), T.int64(0), v0], var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 * T.int64(8) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1, T.int64(0), T.int64(0), v0])
                            Ts.writes(var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, T.int64(0), T.int64(0), v0])
                            var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, T.int64(0), T.int64(0), v0] = var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, T.int64(0), T.int64(0), v0] + var_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 * T.int64(8) + vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_1, T.int64(0), T.int64(0), v0]
            for ax1 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                for ax0 in T.parallel(T.int64(1), annotations={"thread_binding": "threadIdx.y"}):
                    with Ts.sblock("matmul"):
                        vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0 = Ts.axis.reduce(T.int64(1), ax0)
                        v0 = Ts.axis.spatial(v, u_fused_ax0_fused_fused_0 * T.int64(256) + ax1)
                        Ts.where(u_fused_ax0_fused_fused_0 * T.int64(256) + ax1 < v)
                        Ts.reads(var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, T.int64(0), T.int64(0), v0])
                        Ts.writes(var_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        with Ts.init():
                            var_matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                        var_matmul_intermediate_local[T.int64(0), T.int64(0), v0] = var_matmul_intermediate_local[T.int64(0), T.int64(0), v0] + var_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_2_ax1_0_fused_ax1_1_fused_4_fused_0, T.int64(0), T.int64(0), v0]
            for ax0_fused_0 in T.parallel(T.int64(256), annotations={"thread_binding": "threadIdx.x"}):
                for ax0_fused_1 in T.serial(T.int64(0), T.int64(1)):
                    with Ts.sblock("compute"):
                        v0 = Ts.axis.spatial(v, u_fused_ax0_fused_fused_0 * T.int64(256) + ax0_fused_0 + ax0_fused_1)
                        Ts.where(u_fused_ax0_fused_fused_0 * T.int64(256) + (ax0_fused_0 + ax0_fused_1) < v)
                        Ts.reads(var_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        Ts.writes(p_output0_intermediate[T.int64(0), T.int64(0), v0])
                        p_output0_intermediate[T.int64(0), T.int64(0), v0] = T.Cast("float32", var_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with Target("opencl", host={"kind": "llvm", "mtriple": "aarch64-linux-android"}):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
        tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_blockized_gemv():
    # fmt: off
    @Ts.function(private=True)
    def before(x: T.Tensor((1, 4096), "float16"), w: T.Tensor((8, 16384, 4096), "float16"), indptr: T.Tensor((2,), "int32"), o: T.Tensor((2, 16384), "float16")):
        # with Ts.sblock("root"):
        for expert_id in T.parallel(2, annotations={"thread_binding": "blockIdx.y"}):
            with Ts.sblock("gemv_o"):
                v_expert_id_o = Ts.axis.spatial(2, expert_id)
                vi_o = Ts.axis.spatial(1, 0)
                vj_o = Ts.axis.reduce(1, 0)
                Ts.reads(x[0, 0:4096], w[indptr[v_expert_id_o], 0:16384, 0:4096], indptr[v_expert_id_o])
                Ts.writes(o[v_expert_id_o, 0:16384])
                for i, j in T.grid(16384, 4096):
                    with Ts.sblock("gemv"):
                        vi_i, vj_i = Ts.axis.remap("SR", [i, j])
                        Ts.reads(x[0, vj_i], w[indptr[v_expert_id_o], vi_i, vj_i], indptr[v_expert_id_o])
                        Ts.writes(o[v_expert_id_o, vi_i])
                        with Ts.init():
                            o[v_expert_id_o, vi_i] = T.float16(0)
                        o[v_expert_id_o, vi_i] = o[v_expert_id_o, vi_i] + x[0, vj_i] * w[indptr[v_expert_id_o], vi_i, vj_i]

    @Ts.function(private=True)
    def expected(x: T.Tensor((1, 4096), "float16"), w: T.Tensor((8, 16384, 4096), "float16"), indptr: T.Tensor((2,), "int32"), o: T.Tensor((2, 16384), "float16")):
        T.func_attr({"tirx.is_scheduled": True})
        # with Ts.sblock("root"):
        for expert_id in T.parallel(2, annotations={"thread_binding": "blockIdx.y"}):
            with Ts.sblock("gemv_o"):
                v_expert_id_o = Ts.axis.spatial(2, expert_id)
                vi_o = Ts.axis.spatial(1, 0)
                vj_o = Ts.axis.reduce(1, 0)
                Ts.reads(x[0, 0:4096], w[indptr[v_expert_id_o], 0:16384, 0:4096], indptr[v_expert_id_o])
                Ts.writes(o[v_expert_id_o, 0:16384])
                o_rf_local = Ts.sblock_alloc_tensor((16, 2, 16384), "float16", scope="local")
                o_rf_local_1 = Ts.sblock_alloc_tensor((16, 2, 16384), "float16", scope="local")
                w_local = Ts.sblock_alloc_tensor((1, 16384, 4096), "float16", scope="local")
                for u_fused_ax0_fused_fused_0 in T.parallel(4096, annotations={"thread_binding": "blockIdx.x"}):
                    for u_fused_ax0_fused_fused_1 in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                        for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                            for u_fused_ax0_fused_fused_2_init in range(1):
                                for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1_init in T.vectorized(1):
                                    with Ts.sblock("gemv_rf_init"):
                                        vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused = Ts.axis.spatial(16, ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1_init)
                                        v0 = Ts.axis.spatial(16384, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                                        Ts.reads()
                                        Ts.writes(o_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, v_expert_id_o, v0])
                                        o_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, v_expert_id_o, v0] = T.float16(0)
                            for ax1_fused_u_fused_0 in T.serial(32, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax0 in range(1):
                                    for ax1_ax2_fused_0 in range(8):
                                        for ax1_ax2_fused_1 in T.vectorized(1):
                                            with Ts.sblock("w_local"):
                                                v0 = Ts.axis.spatial(1, ax0)
                                                v1 = Ts.axis.spatial(16384, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1)
                                                v2 = Ts.axis.spatial(4096, ax1_fused_u_fused_0 * 128 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 8 + ax1_ax2_fused_0 + ax1_ax2_fused_1)
                                                Ts.reads(w[indptr[v_expert_id_o] + v0, v1, v2])
                                                Ts.writes(w_local[v0, v1, v2])
                                                w_local[v0, v1, v2] = w[indptr[v_expert_id_o] + v0, v1, v2]
                                for u_fused_ax0_fused_fused_2, ax1_fused_u_fused_2 in T.grid(1, 8):
                                    for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1 in T.vectorized(1):
                                        with Ts.sblock("gemv_rf_update"):
                                            vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused = Ts.axis.spatial(16, ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1)
                                            v0 = Ts.axis.spatial(16384, u_fused_ax0_fused_fused_0 * 4 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                            vax1_fused_u_fused_0, vax1_fused_u_fused_2 = Ts.axis.remap("RR", [ax1_fused_u_fused_0, ax1_fused_u_fused_2])
                                            Ts.reads(o_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, v_expert_id_o, v0], x[0, vax1_fused_u_fused_0 * 128 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused * 8 + vax1_fused_u_fused_2], w_local[0, v0, vax1_fused_u_fused_0 * 128 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused * 8 + vax1_fused_u_fused_2], indptr[v_expert_id_o])
                                            Ts.writes(o_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, v_expert_id_o, v0])
                                            o_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, v_expert_id_o, v0] = o_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, v_expert_id_o, v0] + x[0, vax1_fused_u_fused_0 * 128 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused * 8 + vax1_fused_u_fused_2] * w_local[0, v0, vax1_fused_u_fused_0 * 128 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused * 8 + vax1_fused_u_fused_2]
                    for ax2_fused_0 in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                        for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                            for ax2_fused_1_0 in T.serial(1, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax2_fused_1_1 in T.vectorized(1):
                                    with Ts.sblock("gemv_rf_init"):
                                        vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 = Ts.axis.spatial(16, ax0)
                                        v0 = Ts.axis.spatial(16384, u_fused_ax0_fused_fused_0 * 4 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                        Ts.reads()
                                        Ts.writes(o_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v_expert_id_o, v0])
                                        o_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v_expert_id_o, v0] = T.float16(0)
                                    for ax1 in range(1):
                                        with Ts.sblock("gemv_rf_update"):
                                            vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1 = Ts.axis.remap("SR", [ax0, ax1])
                                            v0 = Ts.axis.spatial(16384, u_fused_ax0_fused_fused_0 * 4 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                            Ts.reads(o_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v_expert_id_o, v0], o_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, v_expert_id_o, v0])
                                            Ts.writes(o_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v_expert_id_o, v0])
                                            o_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v_expert_id_o, v0] = o_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v_expert_id_o, v0] + o_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, v_expert_id_o, v0]
                    for ax1_fused_1 in range(1):
                        for ax1_fused_0 in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                            for ax0 in T.parallel(16, annotations={"thread_binding": "threadIdx.y"}):
                                with Ts.sblock("gemv"):
                                    vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 = Ts.axis.reduce(16, ax0)
                                    v0 = Ts.axis.spatial(16384, u_fused_ax0_fused_fused_0 * 4 + ax1_fused_0 + ax1_fused_1)
                                    Ts.reads(o_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v_expert_id_o, v0])
                                    Ts.writes(o[v_expert_id_o, v0])
                                    with Ts.init():
                                        o[v_expert_id_o, v0] = T.float16(0)
                                    o[v_expert_id_o, v0] = o[v_expert_id_o, v0] + o_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v_expert_id_o, v0]
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with Target("metal"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
        tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_func_to_skip():
    @Ts.function
    def before(
        data_buf: T.Tensor((seq_len * T.int64(8),), "int32", align=8),  # noqa: F821
        output_buf: T.Tensor((seq_len * T.int64(8),), "int32", align=8),  # noqa: F821
        seq_len: T.int64,
    ):
        with Ts.sblock("exclusive_scan_thrust"):
            Ts.reads()
            Ts.writes()
            T.call_packed(
                "tvm.contrib.thrust.sum_scan",
                T.stack_make_dltensor(
                    data_buf.data, T.stack_make_shape(seq_len * T.int64(8)), 0, 1, 0, T.int64(0)
                ),
                T.stack_make_dltensor(
                    output_buf.data,
                    T.stack_make_shape(seq_len * T.int64(8)),
                    0,
                    1,
                    0,
                    T.int64(0),
                ),
                T.bool(False),
            )

    # This function should be skipped.
    mod = tvm_tirx.IRModule({"main": before})
    with Target("metal"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
        tvm_tirx.ir.assert_structural_equal(mod["main"], before)


def test_gemv_cuda_target_without_max_shared_memory_per_block():
    # fmt: off
    @Ts.function(private=True)
    def before(
        A: T.Tensor((1, 1, 1, 128), "float16"),
        B: T.Tensor((1, 1, 64, 128), "float16"),
        C: T.Tensor((1, 1, 1, 64), "float16"),
    ):
        T.func_attr({"tirx.noalias": True})
        for i0, i1, i2, i3, k in T.grid(1, 1, 1, 64, 128):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                Ts.writes(C[v_i0, v_i1, v_i2, v_i3])
                with Ts.init():
                    C[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                C[v_i0, v_i1, v_i2, v_i3] = C[v_i0, v_i1, v_i2, v_i3] + A[
                    v_i0, v_i1, v_i2, v_k
                ] * B[v_i0, v_i1, v_i3, v_k]

    # fmt: on

    target = Target({"kind": "cuda", "max_num_threads": 1024})
    assert target.attrs.get("max_shared_memory_per_block", None) is None

    mod = tvm_tirx.IRModule({"main": before})
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)

    assert mod["main"].attrs["tirx.is_scheduled"] == 1


def test_gemv_rank_one_vector_input():
    @Ts.function(private=True)
    def before(
        matrix: T.Tensor((2, 2), "float32"),
        vector: T.Tensor((2,), "float32"),
        output: T.Tensor((2,), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        for i, k in T.grid(2, 2):
            with Ts.sblock("gemv"):
                vi, vk = Ts.axis.remap("SR", [i, k])
                Ts.reads(matrix[vi, vk], vector[vk])
                Ts.writes(output[vi])
                with Ts.init():
                    output[vi] = T.float32(0)
                output[vi] += matrix[vi, vk] * vector[vk]

    mod = tvm_tirx.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)

    assert mod["main"].attrs["tirx.is_scheduled"] == 1
    sch = tvm.s_tir.Schedule(mod)
    vector_local = sch.get_sblock("vector_local")
    vector_load_loop = sch.get(sch.get_loops(vector_local)[-1])
    assert vector_load_loop.kind == tvm_tirx.ir.ForKind.VECTORIZED


def test_gemv_broadcast_epilogue():
    # fmt: off
    n = T.dynamic("n", "int32")

    @Ts.function(private=True)
    def before(
        A: T.Tensor((1, 32, 1, 128), "float16"),
        B: T.Tensor((1, 32, n, 128), 'float16'),
        C: T.Tensor((1, 32, 2, 3, n), 'float32'),
    ):
        T.func_attr({"tirx.noalias": True})

        C_temp = Ts.sblock_alloc_tensor((1, 32, 1, n), "float16")
        for i0, i1, i2, i3, k in T.grid(1, 32, 1, n, 128):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                Ts.writes(C_temp[v_i0, v_i1, v_i2, v_i3])
                with Ts.init():
                    C_temp[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                C_temp[v_i0, v_i1, v_i2, v_i3] = C_temp[
                    v_i0, v_i1, v_i2, v_i3
                ] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]
        for i0, i1, i2, i3, i4 in T.grid(1, 32, 2, 3, n):
            with Ts.sblock("broadcast_epilogue"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap(
                    "SSSSS", [i0, i1, i2, i3, i4]
                )
                Ts.reads(C_temp[v_i0, v_i1, 0, v_i4])
                Ts.writes(C[v_i0, v_i1, v_i2, v_i3, v_i4])
                C[v_i0, v_i1, v_i2, v_i3, v_i4] = T.Cast(
                    "float32", C_temp[v_i0, v_i1, 0, v_i4]
                )

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    with Target("nvidia/geforce-rtx-3090-ti"):
        mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)

    assert mod["main"].attrs["tirx.is_scheduled"] == 1


if __name__ == "__main__":
    tvm_tirx.testing.main()
