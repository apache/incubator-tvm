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
# ruff: noqa: E501, F401, F841
from __future__ import annotations

import pytest

import tvm
import tvm_tirx.testing
from tvm.s_tir import dlight as dl
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


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
        var_NT_matmul_intermediate = Ts.sblock_alloc_tensor((1, 32, 1, n), "float16")
        for ax0_fused in range(32):
            for ax1_fused_0 in T.parallel((n + 63) // 64):
                for ax1_fused_1 in T.vectorized(64):
                    for ax2_fused_0 in T.serial(2, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax2_fused_1, u_0, u_1 in T.grid(64, 1, 1):
                            with Ts.sblock("NT_matmul"):
                                v0 = Ts.axis.spatial(32, ax0_fused)
                                v1 = Ts.axis.spatial(n, ax1_fused_0 * 64 + ax1_fused_1)
                                v2 = Ts.axis.reduce(128, ax2_fused_0 * 64 + ax2_fused_1)
                                Ts.where(ax1_fused_0 * 64 + ax1_fused_1 < n)
                                Ts.reads(lv1637[0, v0, 0, v2], lv1638[0, v0, v1, v2])
                                Ts.writes(var_NT_matmul_intermediate[0, v0, 0, v1])
                                with Ts.init():
                                    var_NT_matmul_intermediate[0, v0, 0, v1] = T.float16(0.0)
                                var_NT_matmul_intermediate[0, v0, 0, v1] = var_NT_matmul_intermediate[0, v0, 0, v1] + lv1637[0, v0, 0, v2] * lv1638[0, v0, v1, v2]
        for ax0, ax1 in T.grid(32, n):
            with Ts.sblock("compute"):
                v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(var_NT_matmul_intermediate[0, v0, 0, v1], lv1614[0, 0, 0, v1])
                Ts.writes(var_compute_intermediate[0, v0, 0, v1])
                var_compute_intermediate[0, v0, 0, v1] = T.Cast("float32", T.min(T.max(var_NT_matmul_intermediate[0, v0, 0, v1] * T.float16(0.088397790055248615), T.float16(-65504.0)), lv1614[0, 0, 0, v1]))

    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
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
        for u_fused in range(1):
            for ax0_fused_0 in T.parallel(172):
                for ax0_fused_1 in T.vectorized(128):
                    for ax1_0_fused_0 in T.serial(8, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax1_0_fused_1, ax1_1_0, ax1_1_1 in T.grid(64, 1, 8):
                            with Ts.sblock("NT_matmul"):
                                v0 = Ts.axis.spatial(22016, ax0_fused_0 * 128 + ax0_fused_1)
                                v1 = Ts.axis.reduce(4096, ax1_0_fused_0 * 512 + ax1_0_fused_1 * 8 + ax1_1_0 * 8 + ax1_1_1)
                                Ts.reads(lv1654[0, 0, v1], lv571[v0, v1 // 8], lv572[v0, v1 // 32])
                                Ts.writes(var_NT_matmul_intermediate[0, 0, v0])
                                with Ts.init():
                                    var_NT_matmul_intermediate[0, 0, v0] = T.float16(0.0)
                                var_NT_matmul_intermediate[0, 0, v0] = var_NT_matmul_intermediate[0, 0, v0] + lv1654[0, 0, v1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v0, v1 // 8], T.Cast("uint32", v1 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7.0)) * lv572[v0, v1 // 32])
        # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target('llvm'))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


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
        for u_fused in range(1):
            for ax0_fused_0 in T.parallel(172):
                for ax0_fused_1 in T.vectorized(128):
                    for ax1_0_fused_0 in T.serial(8, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax1_0_fused_1, ax1_1_0, ax1_1_1 in T.grid(64, 1, 8):
                            with Ts.sblock("NT_matmul"):
                                v0 = Ts.axis.spatial(22016, ax0_fused_0 * 128 + ax0_fused_1)
                                v1 = Ts.axis.reduce(4096, ax1_0_fused_0 * 512 + ax1_0_fused_1 * 8 + ax1_1_0 * 8 + ax1_1_1)
                                Ts.reads(lv1654[0, 0, v1], lv571[v0, v1 // 8], lv572[v0, v1 // 32])
                                Ts.writes(var_NT_matmul_intermediate[0, 0, v0])
                                with Ts.init():
                                    var_NT_matmul_intermediate[0, 0, v0] = T.float16(0.0)
                                var_NT_matmul_intermediate[0, 0, v0] = var_NT_matmul_intermediate[0, 0, v0] + lv1654[0, 0, v1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv571[v0, v1 // 8], T.Cast("uint32", v1 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7.0)) * lv572[v0, v1 // 32])
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
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
        var_NT_matmul_intermediate = Ts.sblock_alloc_tensor((1, 1, 32000), "float16")
        for u_fused in range(1):
            for ax0_fused_0 in T.parallel(250):
                for ax0_fused_1 in T.vectorized(128):
                    for ax1_0_fused_0 in T.serial(8, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax1_0_fused_1, ax1_1_0, ax1_1_1 in T.grid(64, 1, 8):
                            with Ts.sblock("NT_matmul"):
                                v0 = Ts.axis.spatial(32000, ax0_fused_0 * 128 + ax0_fused_1)
                                v1 = Ts.axis.reduce(4096, ax1_0_fused_0 * 512 + ax1_0_fused_1 * 8 + ax1_1_0 * 8 + ax1_1_1)
                                Ts.reads(lv3216[0, 0, v1], lv771[v0, v1 // 8], lv772[v0, v1 // 32])
                                Ts.writes(var_NT_matmul_intermediate[0, 0, v0])
                                with Ts.init():
                                    var_NT_matmul_intermediate[0, 0, v0] = T.float16(0.0)
                                var_NT_matmul_intermediate[0, 0, v0] = var_NT_matmul_intermediate[0, 0, v0] + lv3216[0, 0, v1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv771[v0, v1 // 8], T.Cast("uint32", v1 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7.0)) * lv772[v0, v1 // 32])
        for ax0 in range(32000):
            with Ts.sblock("compute"):
                v0 = Ts.axis.spatial(32000, ax0)
                Ts.reads(var_NT_matmul_intermediate[0, 0, v0])
                Ts.writes(p_output0_intermediate[0, 0, v0])
                p_output0_intermediate[0, 0, v0] = T.Cast("float32", var_NT_matmul_intermediate[0, 0, v0])
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
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
        var_NT_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16")
        for u_fused in range(1):
            for ax0_fused_0 in T.parallel(T.int64(64)):
                for ax0_fused_1 in T.vectorized(T.int64(64)):
                    for ax1_0_fused_0 in T.serial(T.int64(11), annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                        for ax1_0_fused_1, ax1_1_0, ax1_1_1 in T.grid(T.int64(128), T.int64(1), T.int64(8)):
                            with Ts.sblock("NT_matmul"):
                                v0 = Ts.axis.spatial(T.int64(4096), ax0_fused_0 * T.int64(64) + ax0_fused_1)
                                v1 = Ts.axis.reduce(T.int64(11008), (ax1_0_fused_0 * T.int64(128) + ax1_0_fused_1) * T.int64(8) + ax1_1_0 * T.int64(8) + ax1_1_1)
                                Ts.where(ax1_0_fused_0 * T.int64(128) + ax1_0_fused_1 < T.int64(1376))
                                Ts.reads(lv574[T.int64(0), T.int64(0), v1], lv575[v0, v1 // T.int64(8)], lv576[v0, v1 // T.int64(32)])
                                Ts.writes(var_NT_matmul_intermediate[T.int64(0), T.int64(0), v0])
                                with Ts.init():
                                    var_NT_matmul_intermediate[T.int64(0), T.int64(0), v0] = T.float16(0.0)
                                var_NT_matmul_intermediate[T.int64(0), T.int64(0), v0] = var_NT_matmul_intermediate[T.int64(0), T.int64(0), v0] + lv574[T.int64(0), T.int64(0), v1] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv575[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7.0)) * lv576[v0, v1 // T.int64(32)])
        for ax0 in range(T.int64(4096)):
            with Ts.sblock("T_add"):
                v0 = Ts.axis.spatial(T.int64(4096), ax0)
                Ts.reads(lv570[T.int64(0), T.int64(0), v0], var_NT_matmul_intermediate[T.int64(0), T.int64(0), v0])
                Ts.writes(p_output0_intermediate[T.int64(0), T.int64(0), v0])
                p_output0_intermediate[T.int64(0), T.int64(0), v0] = lv570[T.int64(0), T.int64(0), v0] + var_NT_matmul_intermediate[T.int64(0), T.int64(0), v0]
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
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
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
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
        T.func_attr({"tirx.noalias": True})
        # with Ts.sblock("root"):
        p_output0_intermediate_1 = Ts.sblock_alloc_tensor((11008, 4096), "float16")
        var_matmul_intermediate = Ts.sblock_alloc_tensor((1, 1, 4096), "float16")
        for i, j in T.grid(11008, 4096):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv575[v_i // 8, v_j], lv576[v_i // 32, v_j])
                Ts.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv575[v_i // 8, v_j], T.Cast("uint32", v_i % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7.0)) * lv576[v_i // 32, v_j]
        for i0, i1, i2, k in T.grid(1, 1, 4096, 11008):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv574[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
                Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0.0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv574[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
        for ax0, ax1, ax2 in T.grid(1, 1, 4096):
            with Ts.sblock("T_add"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(lv570[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                Ts.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv570[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
    # fmt: on
    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
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
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        p_output0_intermediate_1 = Ts.sblock_alloc_tensor((T.int64(4096), v), "float16")
        var_matmul_intermediate = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), v), "float16")
        for i, j in T.grid(T.int64(4096), v):
            with Ts.sblock("decode"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                Ts.reads(lv612[v_i // T.int64(8), v_j], lv613[v_i // T.int64(32), v_j])
                Ts.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv612[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7.0)) * lv613[v_i // T.int64(32), v_j]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), v, T.int64(4096)):
            with Ts.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
                Ts.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
                with Ts.init():
                    var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0.0)
                var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), v):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
                Ts.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1, v_i2])
    # fmt: on

    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
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
                for u_fused in range(1):
                    for ax0_fused_0 in T.parallel(128):
                        for ax0_fused_1 in T.vectorized(128):
                            for ax1_fused_0 in T.serial(64, annotations={"tirx.auto_unroll_max_step": 256, "tirx.unroll_explicit": 1}):
                                for ax1_fused_1, u_0, u_1 in T.grid(64, 1, 1):
                                    with Ts.sblock("gemv"):
                                        v0 = Ts.axis.spatial(16384, ax0_fused_0 * 128 + ax0_fused_1)
                                        v1 = Ts.axis.reduce(4096, ax1_fused_0 * 64 + ax1_fused_1)
                                        Ts.reads(x[0, v1], w[indptr[v_expert_id_o], v0, v1], indptr[v_expert_id_o])
                                        Ts.writes(o[v_expert_id_o, v0])
                                        with Ts.init():
                                            o[v_expert_id_o, v0] = T.float16(0.0)
                                        o[v_expert_id_o, v0] = o[v_expert_id_o, v0] + x[0, v1] * w[indptr[v_expert_id_o], v0, v1]
    # fmt: on
    mod = tvm_tirx.IRModule({"main": before})
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
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
    mod = dl.ApplyDefaultSchedule(dl.cpu.GEMV(), target=Target("llvm"))(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], before)


if __name__ == "__main__":
    tvm_tirx.testing.main()
