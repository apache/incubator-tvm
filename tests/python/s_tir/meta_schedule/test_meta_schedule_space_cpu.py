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
# ruff: noqa: E501, E741, F401
"""Tests for MetaSchedule search space on CPU"""

from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    print_sketches,
)
from tvm.s_tir.meta_schedule.testing.te_workload import create_te_workload
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def _target():
    return Target("aws/cpu/c5.9xlarge")


def _design_space(mod):
    return generate_design_space(
        kind="llvm",
        mod=mod,
        target=_target(),
        types=ms.ScheduleRule,
    )


def test_cpu_c1d():
    # fmt: off
    @Ts.function
    def c1d_0(inputs: T.Tensor((1, 256, 64), "float32"), weight: T.Tensor((3, 64, 128), "float32"), conv1d_nlc: T.Tensor((1, 128, 128), "float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel":288, "meta_schedule.unroll_explicit":512, "meta_schedule.vectorize":64})
            PadInput = Ts.sblock_alloc_tensor((1, 258, 64), dtype="float32")
            conv1d_nlc_global = Ts.sblock_alloc_tensor((1, 128, 128), dtype="float32")
            for i0, i1, i2 in T.grid(1, 258, 64):
                with Ts.sblock("PadInput"):
                    v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(inputs[v_i0, v_i1 - 1, v_i2])
                    Ts.writes(PadInput[v_i0, v_i1, v_i2])
                    PadInput[v_i0, v_i1, v_i2] = T.if_then_else(1 <= v_i1 and v_i1 < 257, inputs[v_i0, v_i1 - 1, v_i2], T.float32(0))
            for n_0, l_0, co_0, n_1, l_1, co_1 in T.grid(1, 1, 2, 1, 1, 8):
                for rl_0, rc_0, n_2, l_2, co_2, rl_1, rc_1, n_3, l_3, co_3 in T.grid(1, 64, 1, 64, 8, 3, 1, 1, 2, 1):
                    with Ts.sblock("conv1d_nlc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_l = Ts.axis.spatial(128, l_0 * 128 + l_1 * 128 + l_2 * 2 + l_3)
                        v_co = Ts.axis.spatial(128, co_0 * 64 + co_1 * 8 + co_2 + co_3)
                        v_rl = Ts.axis.reduce(3, rl_0 * 3 + rl_1)
                        v_rc = Ts.axis.reduce(64, rc_0 + rc_1)
                        Ts.reads(PadInput[v_n, v_l * 2 + v_rl, v_co // 128 * 64 + v_rc], weight[v_rl, v_rc, v_co])
                        Ts.writes(conv1d_nlc_global[v_n, v_l, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv1d_nlc_global[v_n, v_l, v_co] = T.float32(0)
                        conv1d_nlc_global[v_n, v_l, v_co] = conv1d_nlc_global[v_n, v_l, v_co] + PadInput[v_n, v_l * 2 + v_rl, v_co // 128 * 64 + v_rc] * weight[v_rl, v_rc, v_co]
                for ax0, ax1, ax2 in T.grid(1, 128, 8):
                    with Ts.sblock("conv1d_nlc_global"):
                        v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                        v2 = Ts.axis.spatial(128, co_0 * 64 + co_1 * 8 + ax2)
                        Ts.reads(conv1d_nlc_global[v0, v1, v2])
                        Ts.writes(conv1d_nlc[v0, v1, v2])
                        conv1d_nlc[v0, v1, v2] = conv1d_nlc_global[v0, v1, v2]
    @Ts.function
    def c1d_1(inputs: T.Tensor((1, 256, 64), "float32"), weight: T.Tensor((3, 64, 128), "float32"), conv1d_nlc: T.Tensor((1, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 258, 64))
            conv1d_nlc_global = Ts.sblock_alloc_tensor((1, 128, 128))
            for n_0, l_0, co_0 in T.grid(1, 1, 2):
                for n_1, l_1, co_1 in T.grid(1, 1, 8):
                    for ax0, ax1, ax2 in T.grid(1, 257, 64):
                        with Ts.sblock("PadInput"):
                            v_i0 = Ts.axis.spatial(1, ax0)
                            v_i1 = Ts.axis.spatial(258, ax1)
                            v_i2 = Ts.axis.spatial(64, ax2)
                            Ts.reads(inputs[v_i0, v_i1 - 1, v_i2])
                            Ts.writes(PadInput[v_i0, v_i1, v_i2])
                            PadInput[v_i0, v_i1, v_i2] = T.if_then_else(1 <= v_i1 and v_i1 < 257, inputs[v_i0, v_i1 - 1, v_i2], T.float32(0))
                    for rl_0, rc_0, n_2, l_2, co_2, rl_1, rc_1, n_3, l_3, co_3 in T.grid(1, 64, 1, 64, 8, 3, 1, 1, 2, 1):
                        with Ts.sblock("conv1d_nlc"):
                            v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                            v_l = Ts.axis.spatial(128, l_0 * 128 + l_1 * 128 + l_2 * 2 + l_3)
                            v_co = Ts.axis.spatial(128, co_0 * 64 + co_1 * 8 + co_2 + co_3)
                            v_rl = Ts.axis.reduce(3, rl_0 * 3 + rl_1)
                            v_rc = Ts.axis.reduce(64, rc_0 + rc_1)
                            Ts.reads(PadInput[v_n, v_l * 2 + v_rl, v_co // 128 * 64 + v_rc], weight[v_rl, v_rc, v_co])
                            Ts.writes(conv1d_nlc_global[v_n, v_l, v_co])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                conv1d_nlc_global[v_n, v_l, v_co] = T.float32(0)
                            conv1d_nlc_global[v_n, v_l, v_co] = conv1d_nlc_global[v_n, v_l, v_co] + PadInput[v_n, v_l * 2 + v_rl, v_co // 128 * 64 + v_rc] * weight[v_rl, v_rc, v_co]
                for ax0, ax1, ax2 in T.grid(1, 128, 64):
                    with Ts.sblock("conv1d_nlc_global"):
                        v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                        v2 = Ts.axis.spatial(128, co_0 * 64 + ax2)
                        Ts.reads(conv1d_nlc_global[v0, v1, v2])
                        Ts.writes(conv1d_nlc[v0, v1, v2])
                        conv1d_nlc[v0, v1, v2] = conv1d_nlc_global[v0, v1, v2]

    @Ts.function
    def c1d_2(inputs: T.Tensor((1, 256, 64), "float32"), weight: T.Tensor((3, 64, 128), "float32"), conv1d_nlc: T.Tensor((1, 128, 128), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            for n_0, l_0, co_0, n_1, l_1, co_1, rl_0, rc_0, n_2, l_2, co_2, rl_1, rc_1, n_3, l_3, co_3 in T.grid(1, 1, 2, 1, 1, 8, 1, 64, 1, 64, 8, 3, 1, 1, 2, 1):
                with Ts.sblock("conv1d_nlc"):
                    v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                    v_l = Ts.axis.spatial(128, l_0 * 128 + l_1 * 128 + l_2 * 2 + l_3)
                    v_co = Ts.axis.spatial(128, co_0 * 64 + co_1 * 8 + co_2 + co_3)
                    v_rl = Ts.axis.reduce(3, rl_0 * 3 + rl_1)
                    v_rc = Ts.axis.reduce(64, rc_0 + rc_1)
                    Ts.reads(inputs[v_n, v_l * 2 + v_rl - 1, v_co // 128 * 64 + v_rc], weight[v_rl, v_rc, v_co])
                    Ts.writes(conv1d_nlc[v_n, v_l, v_co])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        conv1d_nlc[v_n, v_l, v_co] = T.float32(0)
                    conv1d_nlc[v_n, v_l, v_co] = conv1d_nlc[v_n, v_l, v_co] + T.if_then_else(1 <= v_l * 2 + v_rl and v_l * 2 + v_rl < 257, inputs[v_n, v_l * 2 + v_rl - 1, v_co // 128 * 64 + v_rc], T.float32(0)) * weight[v_rl, v_rc, v_co]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 64, 2]),
        ("SamplePerfectTile", [2, 8, 8, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [64, 1]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 64, 2]),
        ("SamplePerfectTile", [2, 8, 8, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [64, 1]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 5),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 64, 2]),
        ("SamplePerfectTile", [2, 8, 8, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [64, 1]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -2),
    ]

    mod = create_te_workload("C1D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c1d_0, c1d_1, c1d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_c2d():
    # fmt: off
    @Ts.function
    def c2d_0(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 230, 230, 3))
            conv2d_nhwc_global = Ts.sblock_alloc_tensor((1, 112, 112, 64))
            for n_0, h_0, w_0, co_0, n_1, h_1, w_1 in T.grid(1, 7, 4, 2, 1, 1, 28):
                for ax0, ax1, ax2, ax3 in T.grid(1, 37, 7, 3):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(230, h_0 * 32 + ax1)
                        v_i2 = Ts.axis.spatial(230, w_0 * 56 + w_1 * 2 + ax2)
                        v_i3 = Ts.axis.spatial(3, ax3)
                        Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(3 <= v_i1 and v_i1 < 227 and 3 <= v_i2 and v_i2 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3], T.float32(0))
                for co_1 in range(8):
                    for rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(7, 7, 1, 1, 2, 1, 1, 1, 1, 3, 1, 8, 1, 4):
                        with Ts.sblock("conv2d_nhwc"):
                            v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                            v_h = Ts.axis.spatial(112, h_0 * 16 + h_1 * 16 + h_2 * 8 + h_3)
                            v_w = Ts.axis.spatial(112, w_0 * 28 + w_1 + w_2 + w_3)
                            v_co = Ts.axis.spatial(64, co_0 * 32 + co_1 * 4 + co_2 * 4 + co_3)
                            v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                            v_rw = Ts.axis.reduce(7, rw_0 + rw_1)
                            v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                            Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                            Ts.writes(conv2d_nhwc_global[v_n, v_h, v_w, v_co])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                conv2d_nhwc_global[v_n, v_h, v_w, v_co] = T.float32(0)
                            conv2d_nhwc_global[v_n, v_h, v_w, v_co] = conv2d_nhwc_global[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 16, 1, 4):
                        with Ts.sblock("conv2d_nhwc_global"):
                            v0 = Ts.axis.spatial(1, ax0)
                            v1 = Ts.axis.spatial(112, h_0 * 16 + ax1)
                            v2 = Ts.axis.spatial(112, w_0 * 28 + w_1 + ax2)
                            v3 = Ts.axis.spatial(64, co_0 * 32 + co_1 * 4 + ax3)
                            Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                            Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                            conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def c2d_1(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 230, 230, 3))
            conv2d_nhwc_global = Ts.sblock_alloc_tensor((1, 112, 112, 64))
            for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
                with Ts.sblock("PadInput"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3])
                    Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                    PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(3 <= v_i1 and v_i1 < 227 and 3 <= v_i2 and v_i2 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3], T.float32(0))
            for n_0, h_0, w_0, co_0 in T.grid(1, 7, 4, 2):
                for n_1, h_1, w_1, co_1, rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(1, 1, 28, 8, 7, 7, 1, 1, 2, 1, 1, 1, 1, 3, 1, 8, 1, 4):
                    with Ts.sblock("conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(112, h_0 * 16 + h_1 * 16 + h_2 * 8 + h_3)
                        v_w = Ts.axis.spatial(112, w_0 * 28 + w_1 + w_2 + w_3)
                        v_co = Ts.axis.spatial(64, co_0 * 32 + co_1 * 4 + co_2 * 4 + co_3)
                        v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                        v_rw = Ts.axis.reduce(7, rw_0 + rw_1)
                        v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                        Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                        Ts.writes(conv2d_nhwc_global[v_n, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv2d_nhwc_global[v_n, v_h, v_w, v_co] = T.float32(0)
                        conv2d_nhwc_global[v_n, v_h, v_w, v_co] = conv2d_nhwc_global[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 16, 28, 32):
                    with Ts.sblock("conv2d_nhwc_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(112, h_0 * 16 + ax1)
                        v2 = Ts.axis.spatial(112, w_0 * 28 + ax2)
                        v3 = Ts.axis.spatial(64, co_0 * 32 + ax3)
                        Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                        Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                        conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def c2d_2(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 230, 230, 3))
            for n_0, h_0 in T.grid(1, 7):
                for ax0, ax1, ax2, ax3 in T.grid(1, 37, 229, 3):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(230, h_0 * 32 + ax1)
                        v_i2 = Ts.axis.spatial(230, ax2)
                        v_i3 = Ts.axis.spatial(3, ax3)
                        Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(3 <= v_i1 and v_i1 < 227 and 3 <= v_i2 and v_i2 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3], T.float32(0))
                for w_0, co_0, n_1, h_1, w_1, co_1, rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(4, 2, 1, 1, 28, 8, 7, 7, 1, 1, 2, 1, 1, 1, 1, 3, 1, 8, 1, 4):
                    with Ts.sblock("conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(112, h_0 * 16 + h_1 * 16 + h_2 * 8 + h_3)
                        v_w = Ts.axis.spatial(112, w_0 * 28 + w_1 + w_2 + w_3)
                        v_co = Ts.axis.spatial(64, co_0 * 32 + co_1 * 4 + co_2 * 4 + co_3)
                        v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                        v_rw = Ts.axis.reduce(7, rw_0 + rw_1)
                        v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                        Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                        Ts.writes(conv2d_nhwc[v_n, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv2d_nhwc[v_n, v_h, v_w, v_co] = T.float32(0)
                        conv2d_nhwc[v_n, v_h, v_w, v_co] = conv2d_nhwc[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 1, 2, 8]),
        ("SamplePerfectTile", [4, 28, 1, 1]),
        ("SamplePerfectTile", [2, 8, 1, 4]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", 6),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 1, 2, 8]),
        ("SamplePerfectTile", [4, 28, 1, 1]),
        ("SamplePerfectTile", [2, 8, 1, 4]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 1, 2, 8]),
        ("SamplePerfectTile", [4, 28, 1, 1]),
        ("SamplePerfectTile", [2, 8, 1, 4]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 1),
    ]

    mod = create_te_workload("C2D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c2d_0, c2d_1, c2d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_c3d():
    # fmt: off
    @Ts.function
    def c3d_0(inputs: T.Tensor((1, 16, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 7, 3, 64), "float32"), conv3d_ndhwc: T.Tensor((1, 8, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 22, 230, 230, 3))
            conv3d_ndhwc_global = Ts.sblock_alloc_tensor((1, 8, 112, 112, 64))
            for n_0, d_0, h_0, w_0, co_0 in T.grid(1, 2, 4, 1, 2):
                for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 13, 61, 229, 3):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(22, d_0 * 8 + ax1)
                        v_i2 = Ts.axis.spatial(230, h_0 * 56 + ax2)
                        v_i3 = Ts.axis.spatial(230, ax3)
                        v_i4 = Ts.axis.spatial(3, ax4)
                        Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3 - 3, v_i4])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3, v_i4])
                        PadInput[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(3 <= v_i1 and v_i1 < 19 and 3 <= v_i2 and v_i2 < 227 and 3 <= v_i3 and v_i3 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3 - 3, v_i4], T.float32(0))
                for n_1, d_1, h_1, w_1, co_1 in T.grid(1, 4, 4, 14, 1):
                    for rd_0, rh_0, rw_0, rc_0, n_2, d_2, h_2, w_2, co_2, rd_1, rh_1, rw_1, rc_1, n_3, d_3, h_3, w_3, co_3 in T.grid(1, 7, 7, 3, 1, 1, 1, 1, 32, 7, 1, 1, 1, 1, 1, 7, 8, 1):
                        with Ts.sblock("conv3d_ndhwc"):
                            v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                            v_d = Ts.axis.spatial(8, d_0 * 4 + d_1 + d_2 + d_3)
                            v_h = Ts.axis.spatial(112, h_0 * 28 + h_1 * 7 + h_2 * 7 + h_3)
                            v_w = Ts.axis.spatial(112, w_0 * 112 + w_1 * 8 + w_2 * 8 + w_3)
                            v_co = Ts.axis.spatial(64, co_0 * 32 + co_1 * 32 + co_2 + co_3)
                            v_rd = Ts.axis.reduce(7, rd_0 * 7 + rd_1)
                            v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                            v_rw = Ts.axis.reduce(7, rw_0 + rw_1)
                            v_rc = Ts.axis.reduce(3, rc_0 + rc_1)
                            Ts.reads(PadInput[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight[v_rd, v_rh, v_rw, v_rc, v_co])
                            Ts.writes(conv3d_ndhwc_global[v_n, v_d, v_h, v_w, v_co])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                conv3d_ndhwc_global[v_n, v_d, v_h, v_w, v_co] = T.float32(0)
                            conv3d_ndhwc_global[v_n, v_d, v_h, v_w, v_co] = conv3d_ndhwc_global[v_n, v_d, v_h, v_w, v_co] + PadInput[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight[v_rd, v_rh, v_rw, v_rc, v_co]
                    for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 1, 7, 8, 32):
                        with Ts.sblock("conv3d_ndhwc_global"):
                            v0 = Ts.axis.spatial(1, ax0)
                            v1 = Ts.axis.spatial(8, d_0 * 4 + d_1 + ax1)
                            v2 = Ts.axis.spatial(112, h_0 * 28 + h_1 * 7 + ax2)
                            v3 = Ts.axis.spatial(112, w_1 * 8 + ax3)
                            v4 = Ts.axis.spatial(64, co_0 * 32 + ax4)
                            Ts.reads(conv3d_ndhwc_global[v0, v1, v2, v3, v4])
                            Ts.writes(conv3d_ndhwc[v0, v1, v2, v3, v4])
                            conv3d_ndhwc[v0, v1, v2, v3, v4] = conv3d_ndhwc_global[v0, v1, v2, v3, v4]
    @Ts.function
    def c3d_1(inputs: T.Tensor((1, 16, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 7, 3, 64), "float32"), conv3d_ndhwc: T.Tensor((1, 8, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 22, 230, 230, 3))
            conv3d_ndhwc_global = Ts.sblock_alloc_tensor((1, 8, 112, 112, 64))
            for n_0, d_0, h_0, w_0, co_0 in T.grid(1, 2, 4, 1, 2):
                for n_1, d_1, h_1, w_1 in T.grid(1, 4, 4, 14):
                    for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 7, 19, 21, 3):
                        with Ts.sblock("PadInput"):
                            v_i0 = Ts.axis.spatial(1, ax0)
                            v_i1 = Ts.axis.spatial(22, d_0 * 8 + d_1 * 2 + ax1)
                            v_i2 = Ts.axis.spatial(230, h_0 * 56 + h_1 * 14 + ax2)
                            v_i3 = Ts.axis.spatial(230, w_1 * 16 + ax3)
                            v_i4 = Ts.axis.spatial(3, ax4)
                            Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3 - 3, v_i4])
                            Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3, v_i4])
                            PadInput[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(3 <= v_i1 and v_i1 < 19 and 3 <= v_i2 and v_i2 < 227 and 3 <= v_i3 and v_i3 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3 - 3, v_i4], T.float32(0))
                    for co_1, rd_0, rh_0, rw_0, rc_0, n_2, d_2, h_2, w_2, co_2, rd_1, rh_1, rw_1, rc_1, n_3, d_3, h_3, w_3, co_3 in T.grid(1, 1, 7, 7, 3, 1, 1, 1, 1, 32, 7, 1, 1, 1, 1, 1, 7, 8, 1):
                        with Ts.sblock("conv3d_ndhwc"):
                            v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                            v_d = Ts.axis.spatial(8, d_0 * 4 + d_1 + d_2 + d_3)
                            v_h = Ts.axis.spatial(112, h_0 * 28 + h_1 * 7 + h_2 * 7 + h_3)
                            v_w = Ts.axis.spatial(112, w_0 * 112 + w_1 * 8 + w_2 * 8 + w_3)
                            v_co = Ts.axis.spatial(64, co_0 * 32 + co_1 * 32 + co_2 + co_3)
                            v_rd = Ts.axis.reduce(7, rd_0 * 7 + rd_1)
                            v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                            v_rw = Ts.axis.reduce(7, rw_0 + rw_1)
                            v_rc = Ts.axis.reduce(3, rc_0 + rc_1)
                            Ts.reads(PadInput[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight[v_rd, v_rh, v_rw, v_rc, v_co])
                            Ts.writes(conv3d_ndhwc_global[v_n, v_d, v_h, v_w, v_co])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                conv3d_ndhwc_global[v_n, v_d, v_h, v_w, v_co] = T.float32(0)
                            conv3d_ndhwc_global[v_n, v_d, v_h, v_w, v_co] = conv3d_ndhwc_global[v_n, v_d, v_h, v_w, v_co] + PadInput[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight[v_rd, v_rh, v_rw, v_rc, v_co]
                for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 4, 28, 112, 32):
                    with Ts.sblock("conv3d_ndhwc_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(8, d_0 * 4 + ax1)
                        v2 = Ts.axis.spatial(112, h_0 * 28 + ax2)
                        v3 = Ts.axis.spatial(112, ax3)
                        v4 = Ts.axis.spatial(64, co_0 * 32 + ax4)
                        Ts.reads(conv3d_ndhwc_global[v0, v1, v2, v3, v4])
                        Ts.writes(conv3d_ndhwc[v0, v1, v2, v3, v4])
                        conv3d_ndhwc[v0, v1, v2, v3, v4] = conv3d_ndhwc_global[v0, v1, v2, v3, v4]
    @Ts.function
    def c3d_2(inputs: T.Tensor((1, 16, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 7, 3, 64), "float32"), conv3d_ndhwc: T.Tensor((1, 8, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 22, 230, 230, 3))
            for n_0, d_0, h_0, w_0, co_0, n_1, d_1, h_1, w_1 in T.grid(1, 2, 4, 1, 2, 1, 4, 4, 14):
                for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 7, 19, 21, 3):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(22, d_0 * 8 + d_1 * 2 + ax1)
                        v_i2 = Ts.axis.spatial(230, h_0 * 56 + h_1 * 14 + ax2)
                        v_i3 = Ts.axis.spatial(230, w_1 * 16 + ax3)
                        v_i4 = Ts.axis.spatial(3, ax4)
                        Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3 - 3, v_i4])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3, v_i4])
                        PadInput[v_i0, v_i1, v_i2, v_i3, v_i4] = T.if_then_else(3 <= v_i1 and v_i1 < 19 and 3 <= v_i2 and v_i2 < 227 and 3 <= v_i3 and v_i3 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3 - 3, v_i4], T.float32(0))
                for co_1, rd_0, rh_0, rw_0, rc_0, n_2, d_2, h_2, w_2, co_2, rd_1, rh_1, rw_1, rc_1, n_3, d_3, h_3, w_3, co_3 in T.grid(1, 1, 7, 7, 3, 1, 1, 1, 1, 32, 7, 1, 1, 1, 1, 1, 7, 8, 1):
                    with Ts.sblock("conv3d_ndhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_d = Ts.axis.spatial(8, d_0 * 4 + d_1 + d_2 + d_3)
                        v_h = Ts.axis.spatial(112, h_0 * 28 + h_1 * 7 + h_2 * 7 + h_3)
                        v_w = Ts.axis.spatial(112, w_0 * 112 + w_1 * 8 + w_2 * 8 + w_3)
                        v_co = Ts.axis.spatial(64, co_0 * 32 + co_1 * 32 + co_2 + co_3)
                        v_rd = Ts.axis.reduce(7, rd_0 * 7 + rd_1)
                        v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                        v_rw = Ts.axis.reduce(7, rw_0 + rw_1)
                        v_rc = Ts.axis.reduce(3, rc_0 + rc_1)
                        Ts.reads(PadInput[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight[v_rd, v_rh, v_rw, v_rc, v_co])
                        Ts.writes(conv3d_ndhwc[v_n, v_d, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv3d_ndhwc[v_n, v_d, v_h, v_w, v_co] = T.float32(0)
                        conv3d_ndhwc[v_n, v_d, v_h, v_w, v_co] = conv3d_ndhwc[v_n, v_d, v_h, v_w, v_co] + PadInput[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight[v_rd, v_rh, v_rw, v_rc, v_co]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [4, 4, 1, 7]),
        ("SamplePerfectTile", [1, 14, 1, 8]),
        ("SamplePerfectTile", [2, 1, 32, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [3, 1]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 4),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [4, 4, 1, 7]),
        ("SamplePerfectTile", [1, 14, 1, 8]),
        ("SamplePerfectTile", [2, 1, 32, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [3, 1]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 8),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [4, 4, 1, 7]),
        ("SamplePerfectTile", [1, 14, 1, 8]),
        ("SamplePerfectTile", [2, 1, 32, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [3, 1]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", 8),
    ]

    mod = create_te_workload("C3D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c3d_0, c3d_1, c3d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_cap():
    # fmt: off
    @Ts.function
    def cap_0(inputs: T.Tensor((1, 16, 16, 4, 4, 32), "float32"), weight: T.Tensor((3, 3, 4, 4, 32, 32), "float32"), conv2d_capsule_nhwijc: T.Tensor((1, 8, 8, 4, 4, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 18, 18, 4, 4, 32))
            conv2d_capsule_nhwijc_global = Ts.sblock_alloc_tensor((1, 8, 8, 4, 4, 32))
            for n_0, h_0, w_0, cap_i_0, cap_j_0, co_0, n_1, h_1 in T.grid(1, 2, 1, 1, 1, 1, 1, 4):
                for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 3, 17, 4, 4, 32):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(18, h_0 * 8 + h_1 * 2 + ax1)
                        v_i2 = Ts.axis.spatial(18, ax2)
                        v_i3, v_i4, v_i5 = Ts.axis.remap("SSS", [ax3, ax4, ax5])
                        Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3, v_i4, v_i5])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5])
                        PadInput[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5] = T.if_then_else(1 <= v_i1 and v_i1 < 17 and 1 <= v_i2 and v_i2 < 17, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3, v_i4, v_i5], T.float32(0))
                for w_1, cap_i_1, cap_j_1, co_1 in T.grid(4, 1, 4, 2):
                    for rh_0, rw_0, cap_k_0, rc_0, n_2, h_2, w_2, cap_i_2, cap_j_2, co_2, rh_1, rw_1, cap_k_1, rc_1, n_3, h_3, w_3, cap_i_3, cap_j_3, co_3 in T.grid(1, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 32, 1, 1, 1, 4, 1, 16):
                        with Ts.sblock("conv2d_capsule_nhwijc"):
                            v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                            v_h = Ts.axis.spatial(8, h_0 * 4 + h_1 + h_2 + h_3)
                            v_w = Ts.axis.spatial(8, w_0 * 8 + w_1 * 2 + w_2 + w_3)
                            v_cap_i = Ts.axis.spatial(4, cap_i_0 * 4 + cap_i_1 * 4 + cap_i_2 * 4 + cap_i_3)
                            v_cap_j = Ts.axis.spatial(4, cap_j_0 * 4 + cap_j_1 + cap_j_2 + cap_j_3)
                            v_co = Ts.axis.spatial(32, co_0 * 32 + co_1 * 16 + co_2 * 16 + co_3)
                            v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                            v_rw = Ts.axis.reduce(3, rw_0 + rw_1)
                            v_cap_k = Ts.axis.reduce(4, cap_k_0 + cap_k_1)
                            v_rc = Ts.axis.reduce(32, rc_0 * 32 + rc_1)
                            Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc], weight[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co])
                            Ts.writes(conv2d_capsule_nhwijc_global[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                conv2d_capsule_nhwijc_global[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = T.float32(0)
                            conv2d_capsule_nhwijc_global[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = conv2d_capsule_nhwijc_global[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc] * weight[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co]
                    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 1, 2, 4, 1, 16):
                        with Ts.sblock("conv2d_capsule_nhwijc_global"):
                            v0 = Ts.axis.spatial(1, ax0)
                            v1 = Ts.axis.spatial(8, h_0 * 4 + h_1 + ax1)
                            v2 = Ts.axis.spatial(8, w_1 * 2 + ax2)
                            v3 = Ts.axis.spatial(4, ax3)
                            v4 = Ts.axis.spatial(4, cap_j_1 + ax4)
                            v5 = Ts.axis.spatial(32, co_1 * 16 + ax5)
                            Ts.reads(conv2d_capsule_nhwijc_global[v0, v1, v2, v3, v4, v5])
                            Ts.writes(conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5])
                            conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5] = conv2d_capsule_nhwijc_global[v0, v1, v2, v3, v4, v5]
    @Ts.function
    def cap_1(inputs: T.Tensor((1, 16, 16, 4, 4, 32), "float32"), weight: T.Tensor((3, 3, 4, 4, 32, 32), "float32"), conv2d_capsule_nhwijc: T.Tensor((1, 8, 8, 4, 4, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 18, 18, 4, 4, 32))
            conv2d_capsule_nhwijc_global = Ts.sblock_alloc_tensor((1, 8, 8, 4, 4, 32))
            for n_0, h_0, w_0, cap_i_0, cap_j_0, co_0 in T.grid(1, 2, 1, 1, 1, 1):
                for n_1, h_1, w_1, cap_i_1, cap_j_1, co_1 in T.grid(1, 4, 4, 1, 4, 2):
                    for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 3, 5, 4, 4, 32):
                        with Ts.sblock("PadInput"):
                            v_i0 = Ts.axis.spatial(1, ax0)
                            v_i1 = Ts.axis.spatial(18, h_0 * 8 + h_1 * 2 + ax1)
                            v_i2 = Ts.axis.spatial(18, w_1 * 4 + ax2)
                            v_i3, v_i4, v_i5 = Ts.axis.remap("SSS", [ax3, ax4, ax5])
                            Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3, v_i4, v_i5])
                            Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5])
                            PadInput[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5] = T.if_then_else(1 <= v_i1 and v_i1 < 17 and 1 <= v_i2 and v_i2 < 17, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3, v_i4, v_i5], T.float32(0))
                    for rh_0, rw_0, cap_k_0, rc_0, n_2, h_2, w_2, cap_i_2, cap_j_2, co_2, rh_1, rw_1, cap_k_1, rc_1, n_3, h_3, w_3, cap_i_3, cap_j_3, co_3 in T.grid(1, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 32, 1, 1, 1, 4, 1, 16):
                        with Ts.sblock("conv2d_capsule_nhwijc"):
                            v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                            v_h = Ts.axis.spatial(8, h_0 * 4 + h_1 + h_2 + h_3)
                            v_w = Ts.axis.spatial(8, w_0 * 8 + w_1 * 2 + w_2 + w_3)
                            v_cap_i = Ts.axis.spatial(4, cap_i_0 * 4 + cap_i_1 * 4 + cap_i_2 * 4 + cap_i_3)
                            v_cap_j = Ts.axis.spatial(4, cap_j_0 * 4 + cap_j_1 + cap_j_2 + cap_j_3)
                            v_co = Ts.axis.spatial(32, co_0 * 32 + co_1 * 16 + co_2 * 16 + co_3)
                            v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                            v_rw = Ts.axis.reduce(3, rw_0 + rw_1)
                            v_cap_k = Ts.axis.reduce(4, cap_k_0 + cap_k_1)
                            v_rc = Ts.axis.reduce(32, rc_0 * 32 + rc_1)
                            Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc], weight[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co])
                            Ts.writes(conv2d_capsule_nhwijc_global[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                conv2d_capsule_nhwijc_global[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = T.float32(0)
                            conv2d_capsule_nhwijc_global[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = conv2d_capsule_nhwijc_global[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc] * weight[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co]
                for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 4, 8, 4, 4, 32):
                    with Ts.sblock("conv2d_capsule_nhwijc_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(8, h_0 * 4 + ax1)
                        v2, v3, v4, v5 = Ts.axis.remap("SSSS", [ax2, ax3, ax4, ax5])
                        Ts.reads(conv2d_capsule_nhwijc_global[v0, v1, v2, v3, v4, v5])
                        Ts.writes(conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5])
                        conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5] = conv2d_capsule_nhwijc_global[v0, v1, v2, v3, v4, v5]
    @Ts.function
    def cap_2(inputs: T.Tensor((1, 16, 16, 4, 4, 32), "float32"), weight: T.Tensor((3, 3, 4, 4, 32, 32), "float32"), conv2d_capsule_nhwijc: T.Tensor((1, 8, 8, 4, 4, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 18, 18, 4, 4, 32))
            for i0, i1, i2, i3, i4, i5 in T.grid(1, 18, 18, 4, 4, 32):
                with Ts.sblock("PadInput"):
                    v_i0, v_i1, v_i2, v_i3, v_i4, v_i5 = Ts.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3, v_i4, v_i5])
                    Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5])
                    PadInput[v_i0, v_i1, v_i2, v_i3, v_i4, v_i5] = T.if_then_else(1 <= v_i1 and v_i1 < 17 and 1 <= v_i2 and v_i2 < 17, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3, v_i4, v_i5], T.float32(0))
            for n_0, h_0, w_0, cap_i_0, cap_j_0, co_0, n_1, h_1, w_1, cap_i_1, cap_j_1, co_1, rh_0, rw_0, cap_k_0, rc_0, n_2, h_2, w_2, cap_i_2, cap_j_2, co_2, rh_1, rw_1, cap_k_1, rc_1, n_3, h_3, w_3, cap_i_3, cap_j_3, co_3 in T.grid(1, 2, 1, 1, 1, 1, 1, 4, 4, 1, 4, 2, 1, 3, 4, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 32, 1, 1, 1, 4, 1, 16):
                with Ts.sblock("conv2d_capsule_nhwijc"):
                    v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                    v_h = Ts.axis.spatial(8, h_0 * 4 + h_1 + h_2 + h_3)
                    v_w = Ts.axis.spatial(8, w_0 * 8 + w_1 * 2 + w_2 + w_3)
                    v_cap_i = Ts.axis.spatial(4, cap_i_0 * 4 + cap_i_1 * 4 + cap_i_2 * 4 + cap_i_3)
                    v_cap_j = Ts.axis.spatial(4, cap_j_0 * 4 + cap_j_1 + cap_j_2 + cap_j_3)
                    v_co = Ts.axis.spatial(32, co_0 * 32 + co_1 * 16 + co_2 * 16 + co_3)
                    v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                    v_rw = Ts.axis.reduce(3, rw_0 + rw_1)
                    v_cap_k = Ts.axis.reduce(4, cap_k_0 + cap_k_1)
                    v_rc = Ts.axis.reduce(32, rc_0 * 32 + rc_1)
                    Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc], weight[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co])
                    Ts.writes(conv2d_capsule_nhwijc[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        conv2d_capsule_nhwijc[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = T.float32(0)
                    conv2d_capsule_nhwijc[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = conv2d_capsule_nhwijc[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc] * weight[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 1]),
        ("SamplePerfectTile", [1, 1, 1, 4]),
        ("SamplePerfectTile", [1, 4, 1, 1]),
        ("SamplePerfectTile", [1, 2, 1, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 32]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 7),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 1]),
        ("SamplePerfectTile", [1, 1, 1, 4]),
        ("SamplePerfectTile", [1, 4, 1, 1]),
        ("SamplePerfectTile", [1, 2, 1, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 32]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 11),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 4, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 1]),
        ("SamplePerfectTile", [1, 1, 1, 4]),
        ("SamplePerfectTile", [1, 4, 1, 1]),
        ("SamplePerfectTile", [1, 2, 1, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 32]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -1),
    ]
    mod = create_te_workload("CAP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cap_0, cap_1, cap_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_dep():
    # fmt: off
    @Ts.function
    def dep_0(placeholder: T.Tensor((1, 112, 112, 32), "float32"), placeholder_1: T.Tensor((1, 3, 3, 32), "float32"), depth_conv2d_nhwc: T.Tensor((1, 112, 112, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 114, 114, 32))
            depth_conv2d_nhwc_global = Ts.sblock_alloc_tensor((1, 112, 112, 32))
            for i0, i1, i2, i3 in T.grid(1, 114, 114, 32):
                with Ts.sblock("PadInput"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(placeholder[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                    Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                    PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 113 and 1 <= v_i2 and v_i2 < 113, placeholder[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float32(0))
            for n_0, h_0, w_0, c_0, n_1, h_1, w_1, c_1 in T.grid(1, 1, 1, 1, 1, 4, 4, 8):
                for rh_0, rw_0, n_2, h_2, w_2, c_2, rh_1, rw_1, n_3, h_3, w_3, c_3 in T.grid(1, 1, 1, 2, 7, 2, 3, 3, 1, 14, 4, 2):
                    with Ts.sblock("depth_conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(112, h_0 * 112 + h_1 * 28 + h_2 * 14 + h_3)
                        v_w = Ts.axis.spatial(112, w_0 * 112 + w_1 * 28 + w_2 * 4 + w_3)
                        v_c = Ts.axis.spatial(32, c_0 * 32 + c_1 * 4 + c_2 * 2 + c_3)
                        v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                        v_rw = Ts.axis.reduce(3, rw_0 * 3 + rw_1)
                        Ts.reads(PadInput[v_n, v_h + v_rh, v_w + v_rw, v_c], placeholder_1[0, v_rh, v_rw, v_c])
                        Ts.writes(depth_conv2d_nhwc_global[v_n, v_h, v_w, v_c])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            depth_conv2d_nhwc_global[v_n, v_h, v_w, v_c] = T.float32(0)
                        depth_conv2d_nhwc_global[v_n, v_h, v_w, v_c] = depth_conv2d_nhwc_global[v_n, v_h, v_w, v_c] + PadInput[v_n, v_h + v_rh, v_w + v_rw, v_c] * placeholder_1[0, v_rh, v_rw, v_c]
                for ax0, ax1, ax2, ax3 in T.grid(1, 28, 28, 4):
                    with Ts.sblock("depth_conv2d_nhwc_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(112, h_1 * 28 + ax1)
                        v2 = Ts.axis.spatial(112, w_1 * 28 + ax2)
                        v3 = Ts.axis.spatial(32, c_1 * 4 + ax3)
                        Ts.reads(depth_conv2d_nhwc_global[v0, v1, v2, v3])
                        Ts.writes(depth_conv2d_nhwc[v0, v1, v2, v3])
                        depth_conv2d_nhwc[v0, v1, v2, v3] = depth_conv2d_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def dep_1(placeholder: T.Tensor((1, 112, 112, 32), "float32"), placeholder_1: T.Tensor((1, 3, 3, 32), "float32"), depth_conv2d_nhwc: T.Tensor((1, 112, 112, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 114, 114, 32))
            depth_conv2d_nhwc_global = Ts.sblock_alloc_tensor((1, 112, 112, 32))
            for i0, i1, i2, i3 in T.grid(1, 114, 114, 32):
                with Ts.sblock("PadInput"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(placeholder[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                    Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                    PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 113 and 1 <= v_i2 and v_i2 < 113, placeholder[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float32(0))
            for n_0, h_0, w_0, c_0 in T.grid(1, 1, 1, 1):
                for n_1, h_1, w_1, c_1, rh_0, rw_0, n_2, h_2, w_2, c_2, rh_1, rw_1, n_3, h_3, w_3, c_3 in T.grid(1, 4, 4, 8, 1, 1, 1, 2, 7, 2, 3, 3, 1, 14, 4, 2):
                    with Ts.sblock("depth_conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(112, h_0 * 112 + h_1 * 28 + h_2 * 14 + h_3)
                        v_w = Ts.axis.spatial(112, w_0 * 112 + w_1 * 28 + w_2 * 4 + w_3)
                        v_c = Ts.axis.spatial(32, c_0 * 32 + c_1 * 4 + c_2 * 2 + c_3)
                        v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                        v_rw = Ts.axis.reduce(3, rw_0 * 3 + rw_1)
                        Ts.reads(PadInput[v_n, v_h + v_rh, v_w + v_rw, v_c], placeholder_1[0, v_rh, v_rw, v_c])
                        Ts.writes(depth_conv2d_nhwc_global[v_n, v_h, v_w, v_c])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            depth_conv2d_nhwc_global[v_n, v_h, v_w, v_c] = T.float32(0)
                        depth_conv2d_nhwc_global[v_n, v_h, v_w, v_c] = depth_conv2d_nhwc_global[v_n, v_h, v_w, v_c] + PadInput[v_n, v_h + v_rh, v_w + v_rw, v_c] * placeholder_1[0, v_rh, v_rw, v_c]
                for ax0, ax1, ax2, ax3 in T.grid(1, 112, 112, 32):
                    with Ts.sblock("depth_conv2d_nhwc_global"):
                        v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                        Ts.reads(depth_conv2d_nhwc_global[v0, v1, v2, v3])
                        Ts.writes(depth_conv2d_nhwc[v0, v1, v2, v3])
                        depth_conv2d_nhwc[v0, v1, v2, v3] = depth_conv2d_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def dep_2(placeholder: T.Tensor((1, 112, 112, 32), "float32"), placeholder_1: T.Tensor((1, 3, 3, 32), "float32"), depth_conv2d_nhwc: T.Tensor((1, 112, 112, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 114, 114, 32))
            for n_0, h_0, w_0, c_0, n_1, h_1 in T.grid(1, 1, 1, 1, 1, 4):
                for ax0, ax1, ax2, ax3 in T.grid(1, 30, 114, 32):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(114, h_1 * 28 + ax1)
                        v_i2, v_i3 = Ts.axis.remap("SS", [ax2, ax3])
                        Ts.reads(placeholder[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 113 and 1 <= v_i2 and v_i2 < 113, placeholder[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float32(0))
                for w_1, c_1, rh_0, rw_0, n_2, h_2, w_2, c_2, rh_1, rw_1, n_3, h_3, w_3, c_3 in T.grid(4, 8, 1, 1, 1, 2, 7, 2, 3, 3, 1, 14, 4, 2):
                    with Ts.sblock("depth_conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(112, h_0 * 112 + h_1 * 28 + h_2 * 14 + h_3)
                        v_w = Ts.axis.spatial(112, w_0 * 112 + w_1 * 28 + w_2 * 4 + w_3)
                        v_c = Ts.axis.spatial(32, c_0 * 32 + c_1 * 4 + c_2 * 2 + c_3)
                        v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                        v_rw = Ts.axis.reduce(3, rw_0 * 3 + rw_1)
                        Ts.reads(PadInput[v_n, v_h + v_rh, v_w + v_rw, v_c], placeholder_1[0, v_rh, v_rw, v_c])
                        Ts.writes(depth_conv2d_nhwc[v_n, v_h, v_w, v_c])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            depth_conv2d_nhwc[v_n, v_h, v_w, v_c] = T.float32(0)
                        depth_conv2d_nhwc[v_n, v_h, v_w, v_c] = depth_conv2d_nhwc[v_n, v_h, v_w, v_c] + PadInput[v_n, v_h + v_rh, v_w + v_rw, v_c] * placeholder_1[0, v_rh, v_rw, v_c]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 14]),
        ("SamplePerfectTile", [1, 4, 7, 4]),
        ("SamplePerfectTile", [1, 8, 2, 2]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 14]),
        ("SamplePerfectTile", [1, 4, 7, 4]),
        ("SamplePerfectTile", [1, 8, 2, 2]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 2, 14]),
        ("SamplePerfectTile", [1, 4, 7, 4]),
        ("SamplePerfectTile", [1, 8, 2, 2]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 5),
    ]
    mod = create_te_workload("DEP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dep_0, dep_1, dep_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_dil():
    # fmt: off
    @Ts.function
    def dil_0(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 109, 109, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 230, 230, 3))
            conv2d_nhwc_global = Ts.sblock_alloc_tensor((1, 109, 109, 64))
            for n_0, h_0, w_0, co_0, n_1, h_1, w_1, co_1 in T.grid(1, 109, 1, 4, 1, 1, 1, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 13, 229, 3):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(230, h_0 * 2 + ax1)
                        v_i2 = Ts.axis.spatial(230, ax2)
                        v_i3 = Ts.axis.spatial(3, ax3)
                        Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(3 <= v_i1 and v_i1 < 227 and 3 <= v_i2 and v_i2 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3], T.float32(0))
                for rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(7, 1, 1, 1, 1, 109, 8, 1, 7, 3, 1, 1, 1, 1):
                    with Ts.sblock("conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(109, h_0 + h_1 + h_2 + h_3)
                        v_w = Ts.axis.spatial(109, w_0 * 109 + w_1 * 109 + w_2 + w_3)
                        v_co = Ts.axis.spatial(64, co_0 * 16 + co_1 * 8 + co_2 + co_3)
                        v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                        v_rw = Ts.axis.reduce(7, rw_0 * 7 + rw_1)
                        v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                        Ts.reads(PadInput[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                        Ts.writes(conv2d_nhwc_global[v_n, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv2d_nhwc_global[v_n, v_h, v_w, v_co] = T.float32(0)
                        conv2d_nhwc_global[v_n, v_h, v_w, v_co] = conv2d_nhwc_global[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 1, 109, 8):
                    with Ts.sblock("conv2d_nhwc_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(109, h_0 + ax1)
                        v2 = Ts.axis.spatial(109, ax2)
                        v3 = Ts.axis.spatial(64, co_0 * 16 + co_1 * 8 + ax3)
                        Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                        Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                        conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def dil_1(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 109, 109, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 230, 230, 3))
            conv2d_nhwc_global = Ts.sblock_alloc_tensor((1, 109, 109, 64))
            for n_0, h_0, w_0, co_0 in T.grid(1, 109, 1, 4):
                for n_1, h_1, w_1, co_1, rh_0 in T.grid(1, 1, 1, 2, 7):
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 229, 3):
                        with Ts.sblock("PadInput"):
                            v_i0 = Ts.axis.spatial(1, ax0)
                            v_i1 = Ts.axis.spatial(230, h_0 * 2 + rh_0 * 2 + ax1)
                            v_i2 = Ts.axis.spatial(230, ax2)
                            v_i3 = Ts.axis.spatial(3, ax3)
                            Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3])
                            Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                            PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(3 <= v_i1 and v_i1 < 227 and 3 <= v_i2 and v_i2 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3], T.float32(0))
                    for rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(1, 1, 1, 1, 109, 8, 1, 7, 3, 1, 1, 1, 1):
                        with Ts.sblock("conv2d_nhwc"):
                            v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                            v_h = Ts.axis.spatial(109, h_0 + h_1 + h_2 + h_3)
                            v_w = Ts.axis.spatial(109, w_0 * 109 + w_1 * 109 + w_2 + w_3)
                            v_co = Ts.axis.spatial(64, co_0 * 16 + co_1 * 8 + co_2 + co_3)
                            v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                            v_rw = Ts.axis.reduce(7, rw_0 * 7 + rw_1)
                            v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                            Ts.reads(PadInput[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                            Ts.writes(conv2d_nhwc_global[v_n, v_h, v_w, v_co])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                conv2d_nhwc_global[v_n, v_h, v_w, v_co] = T.float32(0)
                            conv2d_nhwc_global[v_n, v_h, v_w, v_co] = conv2d_nhwc_global[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 1, 109, 16):
                    with Ts.sblock("conv2d_nhwc_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(109, h_0 + ax1)
                        v2 = Ts.axis.spatial(109, ax2)
                        v3 = Ts.axis.spatial(64, co_0 * 16 + ax3)
                        Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                        Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                        conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def dil_2(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 109, 109, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 230, 230, 3))
            for n_0, h_0 in T.grid(1, 109):
                for ax0, ax1, ax2, ax3 in T.grid(1, 13, 229, 3):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(230, h_0 * 2 + ax1)
                        v_i2 = Ts.axis.spatial(230, ax2)
                        v_i3 = Ts.axis.spatial(3, ax3)
                        Ts.reads(inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(3 <= v_i1 and v_i1 < 227 and 3 <= v_i2 and v_i2 < 227, inputs[v_i0, v_i1 - 3, v_i2 - 3, v_i3], T.float32(0))
                for w_0, co_0, n_1, h_1, w_1, co_1, rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(1, 4, 1, 1, 1, 2, 7, 1, 1, 1, 1, 109, 8, 1, 7, 3, 1, 1, 1, 1):
                    with Ts.sblock("conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(109, h_0 + h_1 + h_2 + h_3)
                        v_w = Ts.axis.spatial(109, w_0 * 109 + w_1 * 109 + w_2 + w_3)
                        v_co = Ts.axis.spatial(64, co_0 * 16 + co_1 * 8 + co_2 + co_3)
                        v_rh = Ts.axis.reduce(7, rh_0 + rh_1)
                        v_rw = Ts.axis.reduce(7, rw_0 * 7 + rw_1)
                        v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                        Ts.reads(PadInput[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                        Ts.writes(conv2d_nhwc[v_n, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv2d_nhwc[v_n, v_h, v_w, v_co] = T.float32(0)
                        conv2d_nhwc[v_n, v_h, v_w, v_co] = conv2d_nhwc[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [109, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 109, 1]),
        ("SamplePerfectTile", [4, 2, 8, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 7),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [109, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 109, 1]),
        ("SamplePerfectTile", [4, 2, 8, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 8),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [109, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 109, 1]),
        ("SamplePerfectTile", [4, 2, 8, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 7]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 1),
    ]
    mod = create_te_workload("DIL", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dil_0, dil_1, dil_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_gmm():
    # fmt: off
    @Ts.function
    def gmm_0(X: T.Tensor((1, 128, 128), "float32"), Y: T.Tensor((1, 128, 128), "float32"), Z: T.Tensor((1, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            Z_global = Ts.sblock_alloc_tensor((1, 128, 128))
            for b_0, i_0, j_0, b_1, i_1, j_1 in T.grid(1, 4, 2, 1, 1, 8):
                for k_0, b_2, i_2, j_2, k_1, b_3, i_3, j_3 in T.grid(128, 1, 16, 1, 1, 1, 2, 8):
                    with Ts.sblock("Z"):
                        v_b = Ts.axis.spatial(1, b_0 + b_1 + b_2 + b_3)
                        v_i = Ts.axis.spatial(128, i_0 * 32 + i_1 * 32 + i_2 * 2 + i_3)
                        v_j = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + j_2 * 8 + j_3)
                        v_k = Ts.axis.reduce(128, k_0 + k_1)
                        Ts.reads(X[v_b, v_i, v_k], Y[v_b, v_k, v_j])
                        Ts.writes(Z_global[v_b, v_i, v_j])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            Z_global[v_b, v_i, v_j] = T.float32(0)
                        Z_global[v_b, v_i, v_j] = Z_global[v_b, v_i, v_j] + X[v_b, v_i, v_k] * Y[v_b, v_k, v_j]
                for ax0, ax1, ax2 in T.grid(1, 32, 8):
                    with Ts.sblock("Z_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(128, i_0 * 32 + ax1)
                        v2 = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + ax2)
                        Ts.reads(Z_global[v0, v1, v2])
                        Ts.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_global[v0, v1, v2]
    @Ts.function
    def gmm_1(X: T.Tensor((1, 128, 128), "float32"), Y: T.Tensor((1, 128, 128), "float32"), Z: T.Tensor((1, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            Z_global = Ts.sblock_alloc_tensor((1, 128, 128))
            for b_0, i_0, j_0 in T.grid(1, 4, 2):
                for b_1, i_1, j_1, k_0, b_2, i_2, j_2, k_1, b_3, i_3, j_3 in T.grid(1, 1, 8, 128, 1, 16, 1, 1, 1, 2, 8):
                    with Ts.sblock("Z"):
                        v_b = Ts.axis.spatial(1, b_0 + b_1 + b_2 + b_3)
                        v_i = Ts.axis.spatial(128, i_0 * 32 + i_1 * 32 + i_2 * 2 + i_3)
                        v_j = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + j_2 * 8 + j_3)
                        v_k = Ts.axis.reduce(128, k_0 + k_1)
                        Ts.reads(X[v_b, v_i, v_k], Y[v_b, v_k, v_j])
                        Ts.writes(Z_global[v_b, v_i, v_j])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            Z_global[v_b, v_i, v_j] = T.float32(0)
                        Z_global[v_b, v_i, v_j] = Z_global[v_b, v_i, v_j] + X[v_b, v_i, v_k] * Y[v_b, v_k, v_j]
                for ax0, ax1, ax2 in T.grid(1, 32, 64):
                    with Ts.sblock("Z_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(128, i_0 * 32 + ax1)
                        v2 = Ts.axis.spatial(128, j_0 * 64 + ax2)
                        Ts.reads(Z_global[v0, v1, v2])
                        Ts.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_global[v0, v1, v2]
    @Ts.function
    def gmm_2(X: T.Tensor((1, 128, 128), "float32"), Y: T.Tensor((1, 128, 128), "float32"), Z: T.Tensor((1, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            for b_0, i_0, j_0, b_1, i_1, j_1, k_0, b_2, i_2, j_2, k_1, b_3, i_3, j_3 in T.grid(1, 4, 2, 1, 1, 8, 128, 1, 16, 1, 1, 1, 2, 8):
                with Ts.sblock("Z"):
                    v_b = Ts.axis.spatial(1, b_0 + b_1 + b_2 + b_3)
                    v_i = Ts.axis.spatial(128, i_0 * 32 + i_1 * 32 + i_2 * 2 + i_3)
                    v_j = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + j_2 * 8 + j_3)
                    v_k = Ts.axis.reduce(128, k_0 + k_1)
                    Ts.reads(X[v_b, v_i, v_k], Y[v_b, v_k, v_j])
                    Ts.writes(Z[v_b, v_i, v_j])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        Z[v_b, v_i, v_j] = T.float32(0)
                    Z[v_b, v_i, v_j] = Z[v_b, v_i, v_j] + X[v_b, v_i, v_k] * Y[v_b, v_k, v_j]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 16, 2]),
        ("SamplePerfectTile", [2, 8, 1, 8]),
        ("SamplePerfectTile", [128, 1]),
        ("SampleCategorical", 1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 16, 2]),
        ("SamplePerfectTile", [2, 8, 1, 8]),
        ("SamplePerfectTile", [128, 1]),
        ("SampleCategorical", 1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 16, 2]),
        ("SamplePerfectTile", [2, 8, 1, 8]),
        ("SamplePerfectTile", [128, 1]),
        ("SampleCategorical", 1),
    ]
    mod = create_te_workload("GMM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[gmm_0, gmm_1, gmm_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_grp():
    # fmt: off
    @Ts.function
    def grp_0(inputs: T.Tensor((1, 56, 56, 64), "float32"), weight: T.Tensor((3, 3, 16, 128), "float32"), conv2d_nhwc: T.Tensor((1, 28, 28, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 58, 58, 64))
            conv2d_nhwc_global = Ts.sblock_alloc_tensor((1, 28, 28, 128))
            for n_0, h_0, w_0, co_0 in T.grid(1, 7, 1, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 9, 57, 32):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(58, h_0 * 8 + ax1)
                        v_i2 = Ts.axis.spatial(58, ax2)
                        v_i3 = Ts.axis.spatial(64, co_0 * 32 + ax3)
                        Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 57 and 1 <= v_i2 and v_i2 < 57, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float32(0))
                for n_1, h_1, w_1, co_1 in T.grid(1, 4, 1, 1):
                    for rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(1, 3, 8, 1, 1, 4, 4, 3, 1, 2, 1, 1, 7, 16):
                        with Ts.sblock("conv2d_nhwc"):
                            v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                            v_h = Ts.axis.spatial(28, h_0 * 4 + h_1 + h_2 + h_3)
                            v_w = Ts.axis.spatial(28, w_0 * 28 + w_1 * 28 + w_2 * 7 + w_3)
                            v_co = Ts.axis.spatial(128, co_0 * 64 + co_1 * 64 + co_2 * 16 + co_3)
                            v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                            v_rw = Ts.axis.reduce(3, rw_0 + rw_1)
                            v_rc = Ts.axis.reduce(16, rc_0 * 2 + rc_1)
                            Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                            Ts.writes(conv2d_nhwc_global[v_n, v_h, v_w, v_co])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                conv2d_nhwc_global[v_n, v_h, v_w, v_co] = T.float32(0)
                            conv2d_nhwc_global[v_n, v_h, v_w, v_co] = conv2d_nhwc_global[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 28, 64):
                        with Ts.sblock("conv2d_nhwc_global"):
                            v0 = Ts.axis.spatial(1, ax0)
                            v1 = Ts.axis.spatial(28, h_0 * 4 + h_1 + ax1)
                            v2 = Ts.axis.spatial(28, ax2)
                            v3 = Ts.axis.spatial(128, co_0 * 64 + ax3)
                            Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                            Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                            conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def grp_1(inputs: T.Tensor((1, 56, 56, 64), "float32"), weight: T.Tensor((3, 3, 16, 128), "float32"), conv2d_nhwc: T.Tensor((1, 28, 28, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 58, 58, 64))
            conv2d_nhwc_global = Ts.sblock_alloc_tensor((1, 28, 28, 128))
            for i0, i1, i2, i3 in T.grid(1, 58, 58, 64):
                with Ts.sblock("PadInput"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                    Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                    PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 57 and 1 <= v_i2 and v_i2 < 57, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float32(0))
            for n_0, h_0, w_0, co_0 in T.grid(1, 7, 1, 2):
                for n_1, h_1, w_1, co_1, rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(1, 4, 1, 1, 1, 3, 8, 1, 1, 4, 4, 3, 1, 2, 1, 1, 7, 16):
                    with Ts.sblock("conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(28, h_0 * 4 + h_1 + h_2 + h_3)
                        v_w = Ts.axis.spatial(28, w_0 * 28 + w_1 * 28 + w_2 * 7 + w_3)
                        v_co = Ts.axis.spatial(128, co_0 * 64 + co_1 * 64 + co_2 * 16 + co_3)
                        v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                        v_rw = Ts.axis.reduce(3, rw_0 + rw_1)
                        v_rc = Ts.axis.reduce(16, rc_0 * 2 + rc_1)
                        Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                        Ts.writes(conv2d_nhwc_global[v_n, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv2d_nhwc_global[v_n, v_h, v_w, v_co] = T.float32(0)
                        conv2d_nhwc_global[v_n, v_h, v_w, v_co] = conv2d_nhwc_global[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 4, 28, 64):
                    with Ts.sblock("conv2d_nhwc_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(28, h_0 * 4 + ax1)
                        v2 = Ts.axis.spatial(28, ax2)
                        v3 = Ts.axis.spatial(128, co_0 * 64 + ax3)
                        Ts.reads(conv2d_nhwc_global[v0, v1, v2, v3])
                        Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                        conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def grp_2(inputs: T.Tensor((1, 56, 56, 64), "float32"), weight: T.Tensor((3, 3, 16, 128), "float32"), conv2d_nhwc: T.Tensor((1, 28, 28, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 58, 58, 64))
            for n_0, h_0, w_0, co_0, n_1, h_1, w_1, co_1, rh_0, rw_0 in T.grid(1, 7, 1, 2, 1, 4, 1, 1, 1, 3):
                for ax0, ax1, ax2, ax3 in T.grid(1, 3, 55, 32):
                    with Ts.sblock("PadInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(58, h_0 * 8 + h_1 * 2 + ax1)
                        v_i2 = Ts.axis.spatial(58, rw_0 + ax2)
                        v_i3 = Ts.axis.spatial(64, co_0 * 32 + ax3)
                        Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 57 and 1 <= v_i2 and v_i2 < 57, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float32(0))
                for rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(8, 1, 1, 4, 4, 3, 1, 2, 1, 1, 7, 16):
                    with Ts.sblock("conv2d_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(28, h_0 * 4 + h_1 + h_2 + h_3)
                        v_w = Ts.axis.spatial(28, w_0 * 28 + w_1 * 28 + w_2 * 7 + w_3)
                        v_co = Ts.axis.spatial(128, co_0 * 64 + co_1 * 64 + co_2 * 16 + co_3)
                        v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1)
                        v_rw = Ts.axis.reduce(3, rw_0 + rw_1)
                        v_rc = Ts.axis.reduce(16, rc_0 * 2 + rc_1)
                        Ts.reads(PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc], weight[v_rh, v_rw, v_rc, v_co])
                        Ts.writes(conv2d_nhwc[v_n, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv2d_nhwc[v_n, v_h, v_w, v_co] = T.float32(0)
                        conv2d_nhwc[v_n, v_h, v_w, v_co] = conv2d_nhwc[v_n, v_h, v_w, v_co] + PadInput[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc] * weight[v_rh, v_rw, v_rc, v_co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 4, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4, 7]),
        ("SamplePerfectTile", [2, 1, 4, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [8, 2]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", 3),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 4, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4, 7]),
        ("SamplePerfectTile", [2, 1, 4, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [8, 2]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [7, 4, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4, 7]),
        ("SamplePerfectTile", [2, 1, 4, 16]),
        ("SamplePerfectTile", [1, 3]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [8, 2]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", 9),
    ]
    mod = create_te_workload("GRP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[grp_0, grp_1, grp_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_t2d():
    # fmt: off
    @Ts.function
    def t2d_0(inputs: T.Tensor((1, 4, 4, 512), "float32"), weight: T.Tensor((4, 4, 512, 256), "float32"), conv2d_transpose_nhwc: T.Tensor((1, 8, 8, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 6, 6, 512))
            conv2d_transpose_nhwc_global = Ts.sblock_alloc_tensor((1, 8, 8, 256))
            for i0, i1, i2, i3 in T.grid(1, 6, 6, 512):
                with Ts.sblock("PadInput"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                    Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                    PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 5 and 1 <= v_i2 and v_i2 < 5, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float32(0))
            for n_0, h_0, w_0, co_0, n_1, h_1, w_1, co_1 in T.grid(1, 1, 2, 8, 1, 4, 1, 4):
                for rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(2, 2, 64, 1, 1, 1, 1, 2, 2, 8, 1, 2, 4, 8):
                    with Ts.sblock("conv2d_transpose_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(8, h_0 * 8 + h_1 * 2 + h_2 * 2 + h_3)
                        v_w = Ts.axis.spatial(8, w_0 * 4 + w_1 * 4 + w_2 * 4 + w_3)
                        v_co = Ts.axis.spatial(256, co_0 * 32 + co_1 * 8 + co_2 * 8 + co_3)
                        v_rh = Ts.axis.reduce(4, rh_0 * 2 + rh_1)
                        v_rw = Ts.axis.reduce(4, rw_0 * 2 + rw_1)
                        v_rc = Ts.axis.reduce(512, rc_0 * 8 + rc_1)
                        Ts.reads(PadInput[v_n, (v_h + v_rh) // 2, (v_w + v_rw) // 2, v_rc], weight[3 - v_rh, 3 - v_rw, v_rc, v_co])
                        Ts.writes(conv2d_transpose_nhwc_global[v_n, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv2d_transpose_nhwc_global[v_n, v_h, v_w, v_co] = T.float32(0)
                        conv2d_transpose_nhwc_global[v_n, v_h, v_w, v_co] = conv2d_transpose_nhwc_global[v_n, v_h, v_w, v_co] + T.if_then_else((v_h + v_rh) % 2 == 0 and (v_w + v_rw) % 2 == 0, PadInput[v_n, (v_h + v_rh) // 2, (v_w + v_rw) // 2, v_rc], T.float32(0)) * weight[3 - v_rh, 3 - v_rw, v_rc, v_co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 2, 4, 8):
                    with Ts.sblock("conv2d_transpose_nhwc_global"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(8, h_1 * 2 + ax1)
                        v2 = Ts.axis.spatial(8, w_0 * 4 + ax2)
                        v3 = Ts.axis.spatial(256, co_0 * 32 + co_1 * 8 + ax3)
                        Ts.reads(conv2d_transpose_nhwc_global[v0, v1, v2, v3])
                        Ts.writes(conv2d_transpose_nhwc[v0, v1, v2, v3])
                        conv2d_transpose_nhwc[v0, v1, v2, v3] = conv2d_transpose_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def t2d_1(inputs: T.Tensor((1, 4, 4, 512), "float32"), weight: T.Tensor((4, 4, 512, 256), "float32"), conv2d_transpose_nhwc: T.Tensor((1, 8, 8, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            PadInput = Ts.sblock_alloc_tensor((1, 6, 6, 512))
            conv2d_transpose_nhwc_global = Ts.sblock_alloc_tensor((1, 8, 8, 256))
            for n_0, h_0, w_0, co_0 in T.grid(1, 1, 2, 8):
                for ax0, ax1, ax2, ax3 in T.grid(1, 6, 4, 512):
                    with Ts.sblock("PadInput"):
                        v_i0, v_i1 = Ts.axis.remap("SS", [ax0, ax1])
                        v_i2 = Ts.axis.spatial(6, w_0 * 2 + ax2)
                        v_i3 = Ts.axis.spatial(512, ax3)
                        Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                        Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                        PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(1 <= v_i1 and v_i1 < 5 and 1 <= v_i2 and v_i2 < 5, inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3], T.float32(0))
                for n_1, h_1, w_1, co_1, rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(1, 4, 1, 4, 2, 2, 64, 1, 1, 1, 1, 2, 2, 8, 1, 2, 4, 8):
                    with Ts.sblock("conv2d_transpose_nhwc"):
                        v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                        v_h = Ts.axis.spatial(8, h_0 * 8 + h_1 * 2 + h_2 * 2 + h_3)
                        v_w = Ts.axis.spatial(8, w_0 * 4 + w_1 * 4 + w_2 * 4 + w_3)
                        v_co = Ts.axis.spatial(256, co_0 * 32 + co_1 * 8 + co_2 * 8 + co_3)
                        v_rh = Ts.axis.reduce(4, rh_0 * 2 + rh_1)
                        v_rw = Ts.axis.reduce(4, rw_0 * 2 + rw_1)
                        v_rc = Ts.axis.reduce(512, rc_0 * 8 + rc_1)
                        Ts.reads(PadInput[v_n, (v_h + v_rh) // 2, (v_w + v_rw) // 2, v_rc], weight[3 - v_rh, 3 - v_rw, v_rc, v_co])
                        Ts.writes(conv2d_transpose_nhwc_global[v_n, v_h, v_w, v_co])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            conv2d_transpose_nhwc_global[v_n, v_h, v_w, v_co] = T.float32(0)
                        conv2d_transpose_nhwc_global[v_n, v_h, v_w, v_co] = conv2d_transpose_nhwc_global[v_n, v_h, v_w, v_co] + T.if_then_else((v_h + v_rh) % 2 == 0 and (v_w + v_rw) % 2 == 0, PadInput[v_n, (v_h + v_rh) // 2, (v_w + v_rw) // 2, v_rc], T.float32(0)) * weight[3 - v_rh, 3 - v_rw, v_rc, v_co]
                for ax0, ax1, ax2, ax3 in T.grid(1, 8, 4, 32):
                    with Ts.sblock("conv2d_transpose_nhwc_global"):
                        v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                        v2 = Ts.axis.spatial(8, w_0 * 4 + ax2)
                        v3 = Ts.axis.spatial(256, co_0 * 32 + ax3)
                        Ts.reads(conv2d_transpose_nhwc_global[v0, v1, v2, v3])
                        Ts.writes(conv2d_transpose_nhwc[v0, v1, v2, v3])
                        conv2d_transpose_nhwc[v0, v1, v2, v3] = conv2d_transpose_nhwc_global[v0, v1, v2, v3]
    @Ts.function
    def t2d_2(inputs: T.Tensor((1, 4, 4, 512), "float32"), weight: T.Tensor((4, 4, 512, 256), "float32"), conv2d_transpose_nhwc: T.Tensor((1, 8, 8, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            for n_0, h_0, w_0, co_0, n_1, h_1, w_1, co_1, rh_0, rw_0, rc_0, n_2, h_2, w_2, co_2, rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3 in T.grid(1, 1, 2, 8, 1, 4, 1, 4, 2, 2, 64, 1, 1, 1, 1, 2, 2, 8, 1, 2, 4, 8):
                with Ts.sblock("conv2d_transpose_nhwc"):
                    v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2 + n_3)
                    v_h = Ts.axis.spatial(8, h_0 * 8 + h_1 * 2 + h_2 * 2 + h_3)
                    v_w = Ts.axis.spatial(8, w_0 * 4 + w_1 * 4 + w_2 * 4 + w_3)
                    v_co = Ts.axis.spatial(256, co_0 * 32 + co_1 * 8 + co_2 * 8 + co_3)
                    v_rh = Ts.axis.reduce(4, rh_0 * 2 + rh_1)
                    v_rw = Ts.axis.reduce(4, rw_0 * 2 + rw_1)
                    v_rc = Ts.axis.reduce(512, rc_0 * 8 + rc_1)
                    Ts.reads(inputs[v_n, (v_h + v_rh) // 2 - 1, (v_w + v_rw) // 2 - 1, v_rc], weight[3 - v_rh, 3 - v_rw, v_rc, v_co])
                    Ts.writes(conv2d_transpose_nhwc[v_n, v_h, v_w, v_co])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        conv2d_transpose_nhwc[v_n, v_h, v_w, v_co] = T.float32(0)
                    conv2d_transpose_nhwc[v_n, v_h, v_w, v_co] = conv2d_transpose_nhwc[v_n, v_h, v_w, v_co] + T.if_then_else((v_h + v_rh) % 2 == 0 and (v_w + v_rw) % 2 == 0, T.if_then_else(1 <= (v_h + v_rh) // 2 and (v_h + v_rh) // 2 < 5 and 1 <= (v_w + v_rw) // 2 and (v_w + v_rw) // 2 < 5, inputs[v_n, (v_h + v_rh) // 2 - 1, (v_w + v_rw) // 2 - 1, v_rc], T.float32(0)), T.float32(0)) * weight[3 - v_rh, 3 - v_rw, v_rc, v_co]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 1, 2]),
        ("SamplePerfectTile", [2, 1, 1, 4]),
        ("SamplePerfectTile", [8, 4, 1, 8]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [64, 8]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 1, 2]),
        ("SamplePerfectTile", [2, 1, 1, 4]),
        ("SamplePerfectTile", [8, 4, 1, 8]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [64, 8]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 3),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 1, 2]),
        ("SamplePerfectTile", [2, 1, 1, 4]),
        ("SamplePerfectTile", [8, 4, 1, 8]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [2, 2]),
        ("SamplePerfectTile", [64, 8]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -2),
    ]
    mod = create_te_workload("T2D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[t2d_0, t2d_1, t2d_2],
        expected_decisions=[decision_0, decision_1, decision_2],
        debug_mask=0,
    )


def test_cpu_nrm():
    # fmt: off
    @Ts.function
    def nrm_0(A: T.Tensor((1, 256, 256), "float32"), D: T.Tensor(1, "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            C = Ts.sblock_alloc_tensor((1,))
            C_rf = Ts.sblock_alloc_tensor((1, 32768))
            for b, i_j_fused_0, i_j_fused_1 in T.grid(1, 32768, 2):
                with Ts.sblock("C_rf"):
                    vi_j_fused_0, v_b, vi_j_fused_1 = Ts.axis.remap("SSR", [i_j_fused_0, b, i_j_fused_1])
                    Ts.reads(A[v_b, (vi_j_fused_0 * 2 + vi_j_fused_1) // 256, (vi_j_fused_0 * 2 + vi_j_fused_1) % 256])
                    Ts.writes(C_rf[v_b, vi_j_fused_0])
                    with Ts.init():
                        C_rf[v_b, vi_j_fused_0] = T.float32(0)
                    C_rf[v_b, vi_j_fused_0] = C_rf[v_b, vi_j_fused_0] + A[v_b, (vi_j_fused_0 * 2 + vi_j_fused_1) // 256, (vi_j_fused_0 * 2 + vi_j_fused_1) % 256] * A[v_b, (vi_j_fused_0 * 2 + vi_j_fused_1) // 256, (vi_j_fused_0 * 2 + vi_j_fused_1) % 256]
            for b, i_j_fused_0 in T.grid(1, 32768):
                with Ts.sblock("C"):
                    vi_j_fused_0, v_b = Ts.axis.remap("RS", [i_j_fused_0, b])
                    Ts.reads(C_rf[v_b, vi_j_fused_0])
                    Ts.writes(C[v_b])
                    with Ts.init():
                        C[v_b] = T.float32(0)
                    C[v_b] = C[v_b] + C_rf[v_b, vi_j_fused_0]
            for b in range(1):
                with Ts.sblock("D"):
                    v_b = Ts.axis.spatial(1, b)
                    Ts.reads(C[v_b])
                    Ts.writes(D[v_b])
                    D[v_b] = T.sqrt(C[v_b])
    @Ts.function
    def nrm_1(A: T.Tensor((1, 256, 256), "float32"), D: T.Tensor(1, "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            C = Ts.sblock_alloc_tensor((1,))
            C_rf = Ts.sblock_alloc_tensor((1, 2))
            for b, i_j_fused_0, i_j_fused_1 in T.grid(1, 32768, 2):
                with Ts.sblock("C_rf"):
                    vi_j_fused_1, v_b, vi_j_fused_0 = Ts.axis.remap("SSR", [i_j_fused_1, b, i_j_fused_0])
                    Ts.reads(A[v_b, (vi_j_fused_0 * 2 + vi_j_fused_1) // 256, (vi_j_fused_0 * 2 + vi_j_fused_1) % 256])
                    Ts.writes(C_rf[v_b, vi_j_fused_1])
                    with Ts.init():
                        C_rf[v_b, vi_j_fused_1] = T.float32(0)
                    C_rf[v_b, vi_j_fused_1] = C_rf[v_b, vi_j_fused_1] + A[v_b, (vi_j_fused_0 * 2 + vi_j_fused_1) // 256, (vi_j_fused_0 * 2 + vi_j_fused_1) % 256] * A[v_b, (vi_j_fused_0 * 2 + vi_j_fused_1) // 256, (vi_j_fused_0 * 2 + vi_j_fused_1) % 256]
            for b, i_j_fused_1 in T.grid(1, 2):
                with Ts.sblock("C"):
                    vi_j_fused_1, v_b = Ts.axis.remap("RS", [i_j_fused_1, b])
                    Ts.reads(C_rf[v_b, vi_j_fused_1])
                    Ts.writes(C[v_b])
                    with Ts.init():
                        C[v_b] = T.float32(0)
                    C[v_b] = C[v_b] + C_rf[v_b, vi_j_fused_1]
            for b in range(1):
                with Ts.sblock("D"):
                    v_b = Ts.axis.spatial(1, b)
                    Ts.reads(C[v_b])
                    Ts.writes(D[v_b])
                    D[v_b] = T.sqrt(C[v_b])
    @Ts.function
    def nrm_2(A: T.Tensor((1, 256, 256), "float32"), D: T.Tensor(1, "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            C = Ts.sblock_alloc_tensor((1,))
            for b, i, j in T.grid(1, 256, 256):
                with Ts.sblock("C"):
                    v_b, v_i, v_j = Ts.axis.remap("SRR", [b, i, j])
                    Ts.reads(A[v_b, v_i, v_j])
                    Ts.writes(C[v_b])
                    with Ts.init():
                        C[v_b] = T.float32(0)
                    C[v_b] = C[v_b] + A[v_b, v_i, v_j] * A[v_b, v_i, v_j]
            for b in range(1):
                with Ts.sblock("D"):
                    v_b = Ts.axis.spatial(1, b)
                    Ts.reads(C[v_b])
                    Ts.writes(D[v_b])
                    D[v_b] = T.sqrt(C[v_b])
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [32768, 2]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [32768, 2]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SampleCategorical", 0),
        ("SampleComputeLocation", -1),
    ]
    mod = create_te_workload("NRM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[nrm_0, nrm_1, nrm_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_sfm():
    # fmt: off
    @Ts.function
    def sfm_0(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum_rf = Ts.sblock_alloc_tensor((256, 16))
            T_softmax_maxelem_rf = Ts.sblock_alloc_tensor((256, 4))
            for i0, k_0, k_1 in T.grid(256, 4, 64):
                with Ts.sblock("T_softmax_maxelem_rf"):
                    vk_0, v_i0, vk_1 = Ts.axis.remap("SSR", [k_0, i0, k_1])
                    Ts.reads(A[v_i0, vk_0 * 64 + vk_1])
                    Ts.writes(T_softmax_maxelem_rf[v_i0, vk_0])
                    with Ts.init():
                        T_softmax_maxelem_rf[v_i0, vk_0] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem_rf[v_i0, vk_0] = T.max(T_softmax_maxelem_rf[v_i0, vk_0], A[v_i0, vk_0 * 64 + vk_1])
            for i0, k_0 in T.grid(256, 4):
                with Ts.sblock("T_softmax_maxelem"):
                    vk_0, v_i0 = Ts.axis.remap("RS", [k_0, i0])
                    Ts.reads(T_softmax_maxelem_rf[v_i0, vk_0])
                    Ts.writes(T_softmax_maxelem[v_i0])
                    with Ts.init():
                        T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], T_softmax_maxelem_rf[v_i0, vk_0])
            for i0, k_0, k_1 in T.grid(256, 16, 16):
                with Ts.sblock("T_softmax_expsum_rf"):
                    vk_0, v_i0, vk_1 = Ts.axis.remap("SSR", [k_0, i0, k_1])
                    Ts.reads(A[v_i0, vk_0 * 16 + vk_1], T_softmax_maxelem[v_i0])
                    Ts.writes(T_softmax_expsum_rf[v_i0, vk_0])
                    with Ts.init():
                        T_softmax_expsum_rf[v_i0, vk_0] = T.float32(0)
                    T_softmax_expsum_rf[v_i0, vk_0] = T_softmax_expsum_rf[v_i0, vk_0] + T.exp(A[v_i0, vk_0 * 16 + vk_1] - T_softmax_maxelem[v_i0])
            for i0, i1 in T.grid(256, 256):
                for ax0, ax1 in T.grid(16, 1):
                    with Ts.sblock("T_softmax_expsum"):
                        vk_0 = Ts.axis.reduce(16, ax0)
                        v_i0 = Ts.axis.spatial(256, i0 + ax1)
                        Ts.reads(T_softmax_expsum_rf[v_i0, vk_0])
                        Ts.writes(T_softmax_expsum[v_i0])
                        with Ts.init():
                            T_softmax_expsum[v_i0] = T.float32(0)
                        T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_expsum_rf[v_i0, vk_0]
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum[v_i0])
                    Ts.writes(T_softmax_norm[v_i0, v_i1])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_1(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 16, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_exp = Ts.sblock_alloc_tensor((256, 256))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum_rf = Ts.sblock_alloc_tensor((256, 16))
            T_softmax_maxelem_rf = Ts.sblock_alloc_tensor((256, 64))
            for i0 in range(256):
                for ax0, ax1, ax2 in T.grid(64, 1, 4):
                    with Ts.sblock("T_softmax_maxelem_rf"):
                        vk_1 = Ts.axis.spatial(64, ax0)
                        v_i0 = Ts.axis.spatial(256, i0 + ax1)
                        vk_0 = Ts.axis.reduce(4, ax2)
                        Ts.reads(A[v_i0, vk_0 * 64 + vk_1])
                        Ts.writes(T_softmax_maxelem_rf[v_i0, vk_1])
                        with Ts.init():
                            T_softmax_maxelem_rf[v_i0, vk_1] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_rf[v_i0, vk_1] = T.max(T_softmax_maxelem_rf[v_i0, vk_1], A[v_i0, vk_0 * 64 + vk_1])
                for i1 in range(256):
                    for ax0, ax1 in T.grid(64, 1):
                        with Ts.sblock("T_softmax_maxelem"):
                            vk_1 = Ts.axis.reduce(64, ax0)
                            v_i0 = Ts.axis.spatial(256, i0 + ax1)
                            Ts.reads(T_softmax_maxelem_rf[v_i0, vk_1])
                            Ts.writes(T_softmax_maxelem[v_i0])
                            with Ts.init():
                                T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], T_softmax_maxelem_rf[v_i0, vk_1])
                    with Ts.sblock("T_softmax_exp"):
                        v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                        Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0])
                        Ts.writes(T_softmax_exp[v_i0, v_i1])
                        T_softmax_exp[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0])
            for i0, k_0, k_1 in T.grid(256, 16, 16):
                with Ts.sblock("T_softmax_expsum_rf"):
                    vk_0, v_i0, vk_1 = Ts.axis.remap("SSR", [k_0, i0, k_1])
                    Ts.reads(T_softmax_exp[v_i0, vk_0 * 16 + vk_1])
                    Ts.writes(T_softmax_expsum_rf[v_i0, vk_0])
                    with Ts.init():
                        T_softmax_expsum_rf[v_i0, vk_0] = T.float32(0)
                    T_softmax_expsum_rf[v_i0, vk_0] = T_softmax_expsum_rf[v_i0, vk_0] + T_softmax_exp[v_i0, vk_0 * 16 + vk_1]
            for i0, k_0 in T.grid(256, 16):
                with Ts.sblock("T_softmax_expsum"):
                    vk_0, v_i0 = Ts.axis.remap("RS", [k_0, i0])
                    Ts.reads(T_softmax_expsum_rf[v_i0, vk_0])
                    Ts.writes(T_softmax_expsum[v_i0])
                    with Ts.init():
                        T_softmax_expsum[v_i0] = T.float32(0)
                    T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_expsum_rf[v_i0, vk_0]
            for i0, i1 in T.grid(256, 256):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_softmax_exp[v_i0, v_i1], T_softmax_expsum[v_i0])
                    Ts.writes(T_softmax_norm[v_i0, v_i1])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm[v_i0, v_i1] = T_softmax_exp[v_i0, v_i1] / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_2(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum_rf = Ts.sblock_alloc_tensor((256, 16))
            for i0, k in T.grid(256, 256):
                with Ts.sblock("T_softmax_maxelem"):
                    v_i0, v_k = Ts.axis.remap("SR", [i0, k])
                    Ts.reads(A[v_i0, v_k])
                    Ts.writes(T_softmax_maxelem[v_i0])
                    with Ts.init():
                        T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
            for i0, k_0, k_1 in T.grid(256, 16, 16):
                with Ts.sblock("T_softmax_expsum_rf"):
                    vk_0, v_i0, vk_1 = Ts.axis.remap("SSR", [k_0, i0, k_1])
                    Ts.reads(A[v_i0, vk_0 * 16 + vk_1], T_softmax_maxelem[v_i0])
                    Ts.writes(T_softmax_expsum_rf[v_i0, vk_0])
                    with Ts.init():
                        T_softmax_expsum_rf[v_i0, vk_0] = T.float32(0)
                    T_softmax_expsum_rf[v_i0, vk_0] = T_softmax_expsum_rf[v_i0, vk_0] + T.exp(A[v_i0, vk_0 * 16 + vk_1] - T_softmax_maxelem[v_i0])
            for i0, k_0 in T.grid(256, 16):
                with Ts.sblock("T_softmax_expsum"):
                    vk_0, v_i0 = Ts.axis.remap("RS", [k_0, i0])
                    Ts.reads(T_softmax_expsum_rf[v_i0, vk_0])
                    Ts.writes(T_softmax_expsum[v_i0])
                    with Ts.init():
                        T_softmax_expsum[v_i0] = T.float32(0)
                    T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_expsum_rf[v_i0, vk_0]
            for i0, i1 in T.grid(256, 256):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum[v_i0])
                    Ts.writes(T_softmax_norm[v_i0, v_i1])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_3(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_exp = Ts.sblock_alloc_tensor((256, 256))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum_rf = Ts.sblock_alloc_tensor((256, 16))
            T_softmax_maxelem_rf = Ts.sblock_alloc_tensor((256, 256))
            for i0, i1 in T.grid(256, 256):
                for ax0, ax1, ax2 in T.grid(256, 1, 1):
                    with Ts.sblock("T_softmax_maxelem_rf"):
                        vk_0 = Ts.axis.spatial(256, ax0)
                        v_i0 = Ts.axis.spatial(256, i0 + ax1)
                        vk_1 = Ts.axis.reduce(1, ax2)
                        Ts.reads(A[v_i0, vk_0 + vk_1])
                        Ts.writes(T_softmax_maxelem_rf[v_i0, vk_0])
                        with Ts.init():
                            T_softmax_maxelem_rf[v_i0, vk_0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_rf[v_i0, vk_0] = T.max(T_softmax_maxelem_rf[v_i0, vk_0], A[v_i0, vk_0 + vk_1])
                for ax0, ax1 in T.grid(256, 1):
                    with Ts.sblock("T_softmax_maxelem"):
                        vk_0 = Ts.axis.reduce(256, ax0)
                        v_i0 = Ts.axis.spatial(256, i0 + ax1)
                        Ts.reads(T_softmax_maxelem_rf[v_i0, vk_0])
                        Ts.writes(T_softmax_maxelem[v_i0])
                        with Ts.init():
                            T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], T_softmax_maxelem_rf[v_i0, vk_0])
                for ax0, ax1 in T.grid(1, 256):
                    with Ts.sblock("T_softmax_exp"):
                        v_i0 = Ts.axis.spatial(256, i0 + ax0)
                        v_i1 = Ts.axis.spatial(256, ax1)
                        Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0])
                        Ts.writes(T_softmax_exp[v_i0, v_i1])
                        T_softmax_exp[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0])
                for ax0 in range(16):
                    for ax0_1, ax1, ax2 in T.grid(1, 1, 16):
                        with Ts.sblock("T_softmax_expsum_rf"):
                            vk_1 = Ts.axis.spatial(16, ax0 + ax0_1)
                            v_i0 = Ts.axis.spatial(256, i0 + ax1)
                            vk_0 = Ts.axis.reduce(16, ax2)
                            Ts.reads(T_softmax_exp[v_i0, vk_0 * 16 + vk_1])
                            Ts.writes(T_softmax_expsum_rf[v_i0, vk_1])
                            with Ts.init():
                                T_softmax_expsum_rf[v_i0, vk_1] = T.float32(0)
                            T_softmax_expsum_rf[v_i0, vk_1] = T_softmax_expsum_rf[v_i0, vk_1] + T_softmax_exp[v_i0, vk_0 * 16 + vk_1]
                    for ax1 in range(1):
                        with Ts.sblock("T_softmax_expsum"):
                            vk_1 = Ts.axis.reduce(16, ax0)
                            v_i0 = Ts.axis.spatial(256, i0 + ax1)
                            Ts.reads(T_softmax_expsum_rf[v_i0, vk_1])
                            Ts.writes(T_softmax_expsum[v_i0])
                            with Ts.init():
                                T_softmax_expsum[v_i0] = T.float32(0)
                            T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_expsum_rf[v_i0, vk_1]
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_softmax_exp[v_i0, v_i1], T_softmax_expsum[v_i0])
                    Ts.writes(T_softmax_norm[v_i0, v_i1])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm[v_i0, v_i1] = T_softmax_exp[v_i0, v_i1] / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_4(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 0, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_exp = Ts.sblock_alloc_tensor((256, 256))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum_rf = Ts.sblock_alloc_tensor((256, 16))
            T_softmax_maxelem_rf = Ts.sblock_alloc_tensor((256, 1))
            for i0 in range(256):
                for ax0, ax1, ax2 in T.grid(1, 1, 256):
                    with Ts.sblock("T_softmax_maxelem_rf"):
                        vk_1 = Ts.axis.spatial(1, ax0)
                        v_i0 = Ts.axis.spatial(256, i0 + ax1)
                        vk_0 = Ts.axis.reduce(256, ax2)
                        Ts.reads(A[v_i0, vk_0 + vk_1])
                        Ts.writes(T_softmax_maxelem_rf[v_i0, vk_1])
                        with Ts.init():
                            T_softmax_maxelem_rf[v_i0, vk_1] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_rf[v_i0, vk_1] = T.max(T_softmax_maxelem_rf[v_i0, vk_1], A[v_i0, vk_0 + vk_1])
                for k_1 in range(1):
                    with Ts.sblock("T_softmax_maxelem"):
                        vk_1, v_i0 = Ts.axis.remap("RS", [k_1, i0])
                        Ts.reads(T_softmax_maxelem_rf[v_i0, vk_1])
                        Ts.writes(T_softmax_maxelem[v_i0])
                        with Ts.init():
                            T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], T_softmax_maxelem_rf[v_i0, vk_1])
            for i0, i1 in T.grid(256, 256):
                with Ts.sblock("T_softmax_exp"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0])
                    Ts.writes(T_softmax_exp[v_i0, v_i1])
                    T_softmax_exp[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0])
            for i0, k_0, k_1 in T.grid(256, 16, 16):
                with Ts.sblock("T_softmax_expsum_rf"):
                    vk_1, v_i0, vk_0 = Ts.axis.remap("SSR", [k_1, i0, k_0])
                    Ts.reads(T_softmax_exp[v_i0, vk_0 * 16 + vk_1])
                    Ts.writes(T_softmax_expsum_rf[v_i0, vk_1])
                    with Ts.init():
                        T_softmax_expsum_rf[v_i0, vk_1] = T.float32(0)
                    T_softmax_expsum_rf[v_i0, vk_1] = T_softmax_expsum_rf[v_i0, vk_1] + T_softmax_exp[v_i0, vk_0 * 16 + vk_1]
            for i0, k_1 in T.grid(256, 16):
                with Ts.sblock("T_softmax_expsum"):
                    vk_1, v_i0 = Ts.axis.remap("RS", [k_1, i0])
                    Ts.reads(T_softmax_expsum_rf[v_i0, vk_1])
                    Ts.writes(T_softmax_expsum[v_i0])
                    with Ts.init():
                        T_softmax_expsum[v_i0] = T.float32(0)
                    T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_expsum_rf[v_i0, vk_1]
            for i0, i1 in T.grid(256, 256):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_softmax_exp[v_i0, v_i1], T_softmax_expsum[v_i0])
                    Ts.writes(T_softmax_norm[v_i0, v_i1])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm[v_i0, v_i1] = T_softmax_exp[v_i0, v_i1] / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_5(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_exp = Ts.sblock_alloc_tensor((256, 256))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum_rf = Ts.sblock_alloc_tensor((256, 16))
            for i0 in range(256):
                for ax0, ax1 in T.grid(1, 256):
                    with Ts.sblock("T_softmax_maxelem"):
                        v_i0 = Ts.axis.spatial(256, i0 + ax0)
                        v_k = Ts.axis.reduce(256, ax1)
                        Ts.reads(A[v_i0, v_k])
                        Ts.writes(T_softmax_maxelem[v_i0])
                        with Ts.init():
                            T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
                for ax0, ax1 in T.grid(1, 256):
                    with Ts.sblock("T_softmax_exp"):
                        v_i0 = Ts.axis.spatial(256, i0 + ax0)
                        v_i1 = Ts.axis.spatial(256, ax1)
                        Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0])
                        Ts.writes(T_softmax_exp[v_i0, v_i1])
                        T_softmax_exp[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0])
                for ax0 in range(16):
                    for ax0_1, ax1, ax2 in T.grid(1, 1, 16):
                        with Ts.sblock("T_softmax_expsum_rf"):
                            vk_1 = Ts.axis.spatial(16, ax0 + ax0_1)
                            v_i0 = Ts.axis.spatial(256, i0 + ax1)
                            vk_0 = Ts.axis.reduce(16, ax2)
                            Ts.reads(T_softmax_exp[v_i0, vk_0 * 16 + vk_1])
                            Ts.writes(T_softmax_expsum_rf[v_i0, vk_1])
                            with Ts.init():
                                T_softmax_expsum_rf[v_i0, vk_1] = T.float32(0)
                            T_softmax_expsum_rf[v_i0, vk_1] = T_softmax_expsum_rf[v_i0, vk_1] + T_softmax_exp[v_i0, vk_0 * 16 + vk_1]
                    for ax1 in range(1):
                        with Ts.sblock("T_softmax_expsum"):
                            vk_1 = Ts.axis.reduce(16, ax0)
                            v_i0 = Ts.axis.spatial(256, i0 + ax1)
                            Ts.reads(T_softmax_expsum_rf[v_i0, vk_1])
                            Ts.writes(T_softmax_expsum[v_i0])
                            with Ts.init():
                                T_softmax_expsum[v_i0] = T.float32(0)
                            T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_expsum_rf[v_i0, vk_1]
                for i1 in range(256):
                    with Ts.sblock("T_softmax_norm"):
                        v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                        Ts.reads(T_softmax_exp[v_i0, v_i1], T_softmax_expsum[v_i0])
                        Ts.writes(T_softmax_norm[v_i0, v_i1])
                        Ts.sblock_attr({"axis": 1})
                        T_softmax_norm[v_i0, v_i1] = T_softmax_exp[v_i0, v_i1] / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_6(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            T_softmax_maxelem_rf = Ts.sblock_alloc_tensor((256, 64))
            for i0 in range(256):
                for ax0, ax1, ax2 in T.grid(64, 1, 4):
                    with Ts.sblock("T_softmax_maxelem_rf"):
                        vk_0 = Ts.axis.spatial(64, ax0)
                        v_i0 = Ts.axis.spatial(256, i0 + ax1)
                        vk_1 = Ts.axis.reduce(4, ax2)
                        Ts.reads(A[v_i0, vk_0 * 4 + vk_1])
                        Ts.writes(T_softmax_maxelem_rf[v_i0, vk_0])
                        with Ts.init():
                            T_softmax_maxelem_rf[v_i0, vk_0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_rf[v_i0, vk_0] = T.max(T_softmax_maxelem_rf[v_i0, vk_0], A[v_i0, vk_0 * 4 + vk_1])
                for k_0 in range(64):
                    with Ts.sblock("T_softmax_maxelem"):
                        vk_0, v_i0 = Ts.axis.remap("RS", [k_0, i0])
                        Ts.reads(T_softmax_maxelem_rf[v_i0, vk_0])
                        Ts.writes(T_softmax_maxelem[v_i0])
                        with Ts.init():
                            T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], T_softmax_maxelem_rf[v_i0, vk_0])
            for i0, k in T.grid(256, 256):
                with Ts.sblock("T_softmax_expsum"):
                    v_i0, v_k = Ts.axis.remap("SR", [i0, k])
                    Ts.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                    Ts.writes(T_softmax_expsum[v_i0])
                    with Ts.init():
                        T_softmax_expsum[v_i0] = T.float32(0)
                    T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
            for i0, i1 in T.grid(256, 256):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum[v_i0])
                    Ts.writes(T_softmax_norm[v_i0, v_i1])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_7(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            T_softmax_maxelem_rf = Ts.sblock_alloc_tensor((256, 4))
            for i0, k_0, k_1 in T.grid(256, 64, 4):
                with Ts.sblock("T_softmax_maxelem_rf"):
                    vk_1, v_i0, vk_0 = Ts.axis.remap("SSR", [k_1, i0, k_0])
                    Ts.reads(A[v_i0, vk_0 * 4 + vk_1])
                    Ts.writes(T_softmax_maxelem_rf[v_i0, vk_1])
                    with Ts.init():
                        T_softmax_maxelem_rf[v_i0, vk_1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem_rf[v_i0, vk_1] = T.max(T_softmax_maxelem_rf[v_i0, vk_1], A[v_i0, vk_0 * 4 + vk_1])
            for i0, k_1 in T.grid(256, 4):
                with Ts.sblock("T_softmax_maxelem"):
                    vk_1, v_i0 = Ts.axis.remap("RS", [k_1, i0])
                    Ts.reads(T_softmax_maxelem_rf[v_i0, vk_1])
                    Ts.writes(T_softmax_maxelem[v_i0])
                    with Ts.init():
                        T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], T_softmax_maxelem_rf[v_i0, vk_1])
            for i0, i1 in T.grid(256, 256):
                for ax0, ax1 in T.grid(1, 256):
                    with Ts.sblock("T_softmax_expsum"):
                        v_i0 = Ts.axis.spatial(256, i0 + ax0)
                        v_k = Ts.axis.reduce(256, ax1)
                        Ts.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                        Ts.writes(T_softmax_expsum[v_i0])
                        with Ts.init():
                            T_softmax_expsum[v_i0] = T.float32(0)
                        T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum[v_i0])
                    Ts.writes(T_softmax_norm[v_i0, v_i1])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_8(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_exp = Ts.sblock_alloc_tensor((256, 256))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            for i0 in range(256):
                for ax0, ax1 in T.grid(1, 256):
                    with Ts.sblock("T_softmax_maxelem"):
                        v_i0 = Ts.axis.spatial(256, i0 + ax0)
                        v_k = Ts.axis.reduce(256, ax1)
                        Ts.reads(A[v_i0, v_k])
                        Ts.writes(T_softmax_maxelem[v_i0])
                        with Ts.init():
                            T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
                for i1 in range(256):
                    with Ts.sblock("T_softmax_exp"):
                        v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                        Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0])
                        Ts.writes(T_softmax_exp[v_i0, v_i1])
                        T_softmax_exp[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0])
            for i0, k in T.grid(256, 256):
                with Ts.sblock("T_softmax_expsum"):
                    v_i0, v_k = Ts.axis.remap("SR", [i0, k])
                    Ts.reads(T_softmax_exp[v_i0, v_k])
                    Ts.writes(T_softmax_expsum[v_i0])
                    with Ts.init():
                        T_softmax_expsum[v_i0] = T.float32(0)
                    T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_exp[v_i0, v_k]
            for i0, i1 in T.grid(256, 256):
                with Ts.sblock("T_softmax_norm"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_softmax_exp[v_i0, v_i1], T_softmax_expsum[v_i0])
                    Ts.writes(T_softmax_norm[v_i0, v_i1])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm[v_i0, v_i1] = T_softmax_exp[v_i0, v_i1] / T_softmax_expsum[v_i0]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [16, 16]),
        ("SamplePerfectTile", [4, 64]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -2),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
    ]
    decision_1 = [
        ("SamplePerfectTile", [16, 16]),
        ("SamplePerfectTile", [4, 64]),
        ("SampleCategorical", 1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 0),
    ]
    decision_2 = [
        ("SamplePerfectTile", [16, 16]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -2),
        ("SampleComputeLocation", -1),
    ]
    decision_3 = [
        ("SamplePerfectTile", [16, 16]),
        ("SamplePerfectTile", [256, 1]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 2),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 1),
    ]
    decision_4 = [
        ("SamplePerfectTile", [16, 16]),
        ("SamplePerfectTile", [256, 1]),
        ("SampleCategorical", 0),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", 0),
    ]
    decision_5 = [
        ("SamplePerfectTile", [16, 16]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 0),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", 0),
        ("SampleComputeLocation", 0),
    ]
    decision_6 = [
        ("SamplePerfectTile", [64, 4]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -2),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", 0),
    ]
    decision_7 = [
        ("SamplePerfectTile", [64, 4]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 1),
        ("SampleComputeLocation", -2),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
    ]
    decision_8 = [
        ("SampleCategorical", 3),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", -1),
        ("SampleComputeLocation", 0),
    ]
    mod = create_te_workload("SFM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[sfm_0, sfm_1, sfm_2, sfm_3, sfm_4, sfm_5, sfm_6, sfm_7, sfm_8],
        expected_decisions=[
            decision_0,
            decision_1,
            decision_2,
            decision_3,
            decision_4,
            decision_5,
            decision_6,
            decision_7,
            decision_8,
        ],
    )


def test_cpu_cbr():
    # fmt: off
    @Ts.function
    def cbr_0(data: T.Tensor((1, 224, 224, 3), "float32"), kernel: T.Tensor((7, 7, 3, 64), "float32"), bias: T.Tensor(64, "float32"), bn_offset: T.Tensor(64, "float32"), bn_scale: T.Tensor(64, "float32"), compute: T.Tensor((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            Conv2dOutput = Ts.sblock_alloc_tensor((1, 112, 112, 64))
            for nn_0, yy_0, xx_0, ff_0, nn_1, yy_1, xx_1, ff_1, ry_0, rx_0, rc_0, nn_2, yy_2, xx_2, ff_2, ry_1, rx_1, rc_1, nn_3, yy_3, xx_3, ff_3 in T.grid(1, 2, 7, 1, 1, 2, 2, 32, 7, 7, 1, 1, 1, 4, 1, 1, 1, 3, 1, 28, 2, 2):
                with Ts.sblock("Conv2dOutput"):
                    v_nn = Ts.axis.spatial(1, nn_0 + nn_1 + nn_2 + nn_3)
                    v_yy = Ts.axis.spatial(112, yy_0 * 56 + yy_1 * 28 + yy_2 * 28 + yy_3)
                    v_xx = Ts.axis.spatial(112, xx_0 * 16 + xx_1 * 8 + xx_2 * 2 + xx_3)
                    v_ff = Ts.axis.spatial(64, ff_0 * 64 + ff_1 * 2 + ff_2 * 2 + ff_3)
                    v_ry = Ts.axis.reduce(7, ry_0 + ry_1)
                    v_rx = Ts.axis.reduce(7, rx_0 + rx_1)
                    v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                    Ts.reads(data[v_nn, v_yy * 2 + v_ry - 3, v_xx * 2 + v_rx - 3, v_rc], kernel[v_ry, v_rx, v_rc, v_ff])
                    Ts.writes(Conv2dOutput[v_nn, v_yy, v_xx, v_ff])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        Conv2dOutput[v_nn, v_yy, v_xx, v_ff] = T.float32(0)
                    Conv2dOutput[v_nn, v_yy, v_xx, v_ff] = Conv2dOutput[v_nn, v_yy, v_xx, v_ff] + T.if_then_else(3 <= v_yy * 2 + v_ry and v_yy * 2 + v_ry < 227 and 3 <= v_xx * 2 + v_rx and v_xx * 2 + v_rx < 227, data[v_nn, v_yy * 2 + v_ry - 3, v_xx * 2 + v_rx - 3, v_rc], T.float32(0)) * kernel[v_ry, v_rx, v_rc, v_ff]
            for i0, i1, i2, i3 in T.grid(1, 112, 112, 64):
                with Ts.sblock("compute"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(Conv2dOutput[v_i0, v_i1, v_i2, v_i3], bias[v_i3], bn_scale[v_i3], bn_offset[v_i3])
                    Ts.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.max((Conv2dOutput[v_i0, v_i1, v_i2, v_i3] + bias[v_i3]) * bn_scale[v_i3] + bn_offset[v_i3], T.float32(0))
    @Ts.function
    def cbr_1(data: T.Tensor((1, 224, 224, 3), "float32"), kernel: T.Tensor((7, 7, 3, 64), "float32"), bias: T.Tensor(64, "float32"), bn_offset: T.Tensor(64, "float32"), bn_scale: T.Tensor(64, "float32"), compute: T.Tensor((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            PaddedInput = Ts.sblock_alloc_tensor((1, 230, 230, 3))
            Conv2dOutput = Ts.sblock_alloc_tensor((1, 112, 112, 64))
            for nn_0, yy_0 in T.grid(1, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 117, 229, 3):
                    with Ts.sblock("PaddedInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(230, yy_0 * 112 + ax1)
                        v_i2 = Ts.axis.spatial(230, ax2)
                        v_i3 = Ts.axis.spatial(3, ax3)
                        Ts.reads(data[v_i0, v_i1 - 3, v_i2 - 3, v_i3])
                        Ts.writes(PaddedInput[v_i0, v_i1, v_i2, v_i3])
                        PaddedInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(3 <= v_i1 and v_i1 < 227 and 3 <= v_i2 and v_i2 < 227, data[v_i0, v_i1 - 3, v_i2 - 3, v_i3], T.float32(0))
                for xx_0, ff_0, nn_1, yy_1, xx_1, ff_1 in T.grid(7, 1, 1, 2, 2, 32):
                    for ry_0, rx_0, rc_0, nn_2, yy_2, xx_2, ff_2, ry_1, rx_1, rc_1, nn_3, yy_3, xx_3, ff_3 in T.grid(7, 7, 1, 1, 1, 4, 1, 1, 1, 3, 1, 28, 2, 2):
                        with Ts.sblock("Conv2dOutput"):
                            v_nn = Ts.axis.spatial(1, nn_0 + nn_1 + nn_2 + nn_3)
                            v_yy = Ts.axis.spatial(112, yy_0 * 56 + yy_1 * 28 + yy_2 * 28 + yy_3)
                            v_xx = Ts.axis.spatial(112, xx_0 * 16 + xx_1 * 8 + xx_2 * 2 + xx_3)
                            v_ff = Ts.axis.spatial(64, ff_0 * 64 + ff_1 * 2 + ff_2 * 2 + ff_3)
                            v_ry = Ts.axis.reduce(7, ry_0 + ry_1)
                            v_rx = Ts.axis.reduce(7, rx_0 + rx_1)
                            v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                            Ts.reads(PaddedInput[v_nn, v_yy * 2 + v_ry, v_xx * 2 + v_rx, v_rc], kernel[v_ry, v_rx, v_rc, v_ff])
                            Ts.writes(Conv2dOutput[v_nn, v_yy, v_xx, v_ff])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                Conv2dOutput[v_nn, v_yy, v_xx, v_ff] = T.float32(0)
                            Conv2dOutput[v_nn, v_yy, v_xx, v_ff] = Conv2dOutput[v_nn, v_yy, v_xx, v_ff] + PaddedInput[v_nn, v_yy * 2 + v_ry, v_xx * 2 + v_rx, v_rc] * kernel[v_ry, v_rx, v_rc, v_ff]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 28, 8, 2):
                        with Ts.sblock("compute"):
                            v_i0 = Ts.axis.spatial(1, ax0)
                            v_i1 = Ts.axis.spatial(112, yy_0 * 56 + yy_1 * 28 + ax1)
                            v_i2 = Ts.axis.spatial(112, xx_0 * 16 + xx_1 * 8 + ax2)
                            v_i3 = Ts.axis.spatial(64, ff_1 * 2 + ax3)
                            Ts.reads(Conv2dOutput[v_i0, v_i1, v_i2, v_i3], bias[v_i3], bn_scale[v_i3], bn_offset[v_i3])
                            Ts.writes(compute[v_i0, v_i1, v_i2, v_i3])
                            compute[v_i0, v_i1, v_i2, v_i3] = T.max((Conv2dOutput[v_i0, v_i1, v_i2, v_i3] + bias[v_i3]) * bn_scale[v_i3] + bn_offset[v_i3], T.float32(0))
    @Ts.function
    def cbr_2(data: T.Tensor((1, 224, 224, 3), "float32"), kernel: T.Tensor((7, 7, 3, 64), "float32"), bias: T.Tensor(64, "float32"), bn_offset: T.Tensor(64, "float32"), bn_scale: T.Tensor(64, "float32"), compute: T.Tensor((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            PaddedInput = Ts.sblock_alloc_tensor((1, 230, 230, 3))
            Conv2dOutput = Ts.sblock_alloc_tensor((1, 112, 112, 64))
            for nn_0, yy_0 in T.grid(1, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 117, 229, 3):
                    with Ts.sblock("PaddedInput"):
                        v_i0 = Ts.axis.spatial(1, ax0)
                        v_i1 = Ts.axis.spatial(230, yy_0 * 112 + ax1)
                        v_i2 = Ts.axis.spatial(230, ax2)
                        v_i3 = Ts.axis.spatial(3, ax3)
                        Ts.reads(data[v_i0, v_i1 - 3, v_i2 - 3, v_i3])
                        Ts.writes(PaddedInput[v_i0, v_i1, v_i2, v_i3])
                        PaddedInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(3 <= v_i1 and v_i1 < 227 and 3 <= v_i2 and v_i2 < 227, data[v_i0, v_i1 - 3, v_i2 - 3, v_i3], T.float32(0))
                for xx_0, ff_0 in T.grid(7, 1):
                    for nn_1, yy_1, xx_1, ff_1, ry_0, rx_0, rc_0, nn_2, yy_2, xx_2, ff_2, ry_1, rx_1, rc_1, nn_3, yy_3, xx_3, ff_3 in T.grid(1, 2, 2, 32, 7, 7, 1, 1, 1, 4, 1, 1, 1, 3, 1, 28, 2, 2):
                        with Ts.sblock("Conv2dOutput"):
                            v_nn = Ts.axis.spatial(1, nn_0 + nn_1 + nn_2 + nn_3)
                            v_yy = Ts.axis.spatial(112, yy_0 * 56 + yy_1 * 28 + yy_2 * 28 + yy_3)
                            v_xx = Ts.axis.spatial(112, xx_0 * 16 + xx_1 * 8 + xx_2 * 2 + xx_3)
                            v_ff = Ts.axis.spatial(64, ff_0 * 64 + ff_1 * 2 + ff_2 * 2 + ff_3)
                            v_ry = Ts.axis.reduce(7, ry_0 + ry_1)
                            v_rx = Ts.axis.reduce(7, rx_0 + rx_1)
                            v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1)
                            Ts.reads(PaddedInput[v_nn, v_yy * 2 + v_ry, v_xx * 2 + v_rx, v_rc], kernel[v_ry, v_rx, v_rc, v_ff])
                            Ts.writes(Conv2dOutput[v_nn, v_yy, v_xx, v_ff])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                Conv2dOutput[v_nn, v_yy, v_xx, v_ff] = T.float32(0)
                            Conv2dOutput[v_nn, v_yy, v_xx, v_ff] = Conv2dOutput[v_nn, v_yy, v_xx, v_ff] + PaddedInput[v_nn, v_yy * 2 + v_ry, v_xx * 2 + v_rx, v_rc] * kernel[v_ry, v_rx, v_rc, v_ff]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 56, 16, 64):
                        with Ts.sblock("compute"):
                            v_i0 = Ts.axis.spatial(1, ax0)
                            v_i1 = Ts.axis.spatial(112, yy_0 * 56 + ax1)
                            v_i2 = Ts.axis.spatial(112, xx_0 * 16 + ax2)
                            v_i3 = Ts.axis.spatial(64, ax3)
                            Ts.reads(Conv2dOutput[v_i0, v_i1, v_i2, v_i3], bias[v_i3], bn_scale[v_i3], bn_offset[v_i3])
                            Ts.writes(compute[v_i0, v_i1, v_i2, v_i3])
                            compute[v_i0, v_i1, v_i2, v_i3] = T.max((Conv2dOutput[v_i0, v_i1, v_i2, v_i3] + bias[v_i3]) * bn_scale[v_i3] + bn_offset[v_i3], T.float32(0))
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 2, 1, 28]),
        ("SamplePerfectTile", [7, 2, 4, 2]),
        ("SamplePerfectTile", [1, 32, 1, 2]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", -2),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 2, 1, 28]),
        ("SamplePerfectTile", [7, 2, 4, 2]),
        ("SamplePerfectTile", [1, 32, 1, 2]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 2, 1, 28]),
        ("SamplePerfectTile", [7, 2, 4, 2]),
        ("SamplePerfectTile", [1, 32, 1, 2]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [7, 1]),
        ("SamplePerfectTile", [1, 3]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 1),
    ]
    mod = create_te_workload("CBR", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cbr_0, cbr_1, cbr_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_tbg():
    # fmt: off
    @Ts.function
    def tbg_0(query: T.Tensor((1, 128, 12, 64), "float32"), value: T.Tensor((1, 128, 12, 64), "float32"), C: T.Tensor((1, 12, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            query_T = Ts.sblock_alloc_tensor((1, 12, 128, 64))
            value_T = Ts.sblock_alloc_tensor((1, 12, 64, 128))
            C_global = Ts.sblock_alloc_tensor((1, 12, 128, 128))
            for b_0, h_0, i_0, j_0, b_1, h_1, i_1 in T.grid(1, 1, 1, 2, 1, 6, 2):
                for ax0, ax1, ax2, ax3 in T.grid(1, 2, 64, 64):
                    with Ts.sblock("value_T"):
                        v_b = Ts.axis.spatial(1, ax0)
                        v_h = Ts.axis.spatial(12, h_1 * 2 + ax1)
                        v_d = Ts.axis.spatial(64, ax2)
                        v_l = Ts.axis.spatial(128, j_0 * 64 + ax3)
                        Ts.reads(value[v_b, v_l, v_h, v_d])
                        Ts.writes(value_T[v_b, v_h, v_d, v_l])
                        value_T[v_b, v_h, v_d, v_l] = value[v_b, v_l, v_h, v_d]
                for ax0, ax1, ax2, ax3 in T.grid(1, 2, 64, 64):
                    with Ts.sblock("query_T"):
                        v_b = Ts.axis.spatial(1, ax0)
                        v_h = Ts.axis.spatial(12, h_1 * 2 + ax1)
                        v_l = Ts.axis.spatial(128, i_1 * 64 + ax2)
                        v_d = Ts.axis.spatial(64, ax3)
                        Ts.reads(query[v_b, v_l, v_h, v_d])
                        Ts.writes(query_T[v_b, v_h, v_l, v_d])
                        query_T[v_b, v_h, v_l, v_d] = query[v_b, v_l, v_h, v_d]
                for j_1 in range(8):
                    for k_0, b_2, h_2, i_2, j_2, k_1, b_3, h_3, i_3, j_3 in T.grid(1, 1, 2, 2, 4, 64, 1, 1, 32, 2):
                        with Ts.sblock("C"):
                            v_b = Ts.axis.spatial(1, b_0 + b_1 + b_2 + b_3)
                            v_h = Ts.axis.spatial(12, h_0 * 12 + h_1 * 2 + h_2 + h_3)
                            v_i = Ts.axis.spatial(128, i_0 * 128 + i_1 * 64 + i_2 * 32 + i_3)
                            v_j = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + j_2 * 2 + j_3)
                            v_k = Ts.axis.reduce(64, k_0 * 64 + k_1)
                            Ts.reads(query_T[v_b, v_h, v_i, v_k], value_T[v_b, v_h, v_k, v_j])
                            Ts.writes(C_global[v_b, v_h, v_i, v_j])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                C_global[v_b, v_h, v_i, v_j] = T.float32(0)
                            C_global[v_b, v_h, v_i, v_j] = C_global[v_b, v_h, v_i, v_j] + query_T[v_b, v_h, v_i, v_k] * value_T[v_b, v_h, v_k, v_j]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 2, 64, 8):
                        with Ts.sblock("C_global"):
                            v0 = Ts.axis.spatial(1, ax0)
                            v1 = Ts.axis.spatial(12, h_1 * 2 + ax1)
                            v2 = Ts.axis.spatial(128, i_1 * 64 + ax2)
                            v3 = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + ax3)
                            Ts.reads(C_global[v0, v1, v2, v3])
                            Ts.writes(C[v0, v1, v2, v3])
                            C[v0, v1, v2, v3] = C_global[v0, v1, v2, v3]
    @Ts.function
    def tbg_1(query: T.Tensor((1, 128, 12, 64), "float32"), value: T.Tensor((1, 128, 12, 64), "float32"), C: T.Tensor((1, 12, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 64, "meta_schedule.vectorize": 64})
            query_T = Ts.sblock_alloc_tensor((1, 12, 128, 64))
            value_T = Ts.sblock_alloc_tensor((1, 12, 64, 128))
            C_global = Ts.sblock_alloc_tensor((1, 12, 128, 128))
            for b, h, l, d in T.grid(1, 12, 128, 64):
                with Ts.sblock("query_T"):
                    v_b, v_h, v_l, v_d = Ts.axis.remap("SSSS", [b, h, l, d])
                    Ts.reads(query[v_b, v_l, v_h, v_d])
                    Ts.writes(query_T[v_b, v_h, v_l, v_d])
                    query_T[v_b, v_h, v_l, v_d] = query[v_b, v_l, v_h, v_d]
            for b_0, h_0, i_0, j_0 in T.grid(1, 1, 1, 2):
                for b_1, h_1, i_1, j_1, k_0, b_2, h_2, i_2, j_2, k_1 in T.grid(1, 6, 2, 8, 1, 1, 2, 2, 4, 64):
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 2):
                        with Ts.sblock("value_T"):
                            v_b = Ts.axis.spatial(1, ax0)
                            v_h = Ts.axis.spatial(12, h_1 * 2 + h_2 + ax1)
                            v_d = Ts.axis.spatial(64, k_1 + ax2)
                            v_l = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + j_2 * 2 + ax3)
                            Ts.reads(value[v_b, v_l, v_h, v_d])
                            Ts.writes(value_T[v_b, v_h, v_d, v_l])
                            value_T[v_b, v_h, v_d, v_l] = value[v_b, v_l, v_h, v_d]
                    for b_3, h_3, i_3, j_3 in T.grid(1, 1, 32, 2):
                        with Ts.sblock("C"):
                            v_b = Ts.axis.spatial(1, b_0 + b_1 + b_2 + b_3)
                            v_h = Ts.axis.spatial(12, h_0 * 12 + h_1 * 2 + h_2 + h_3)
                            v_i = Ts.axis.spatial(128, i_0 * 128 + i_1 * 64 + i_2 * 32 + i_3)
                            v_j = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + j_2 * 2 + j_3)
                            v_k = Ts.axis.reduce(64, k_0 * 64 + k_1)
                            Ts.reads(query_T[v_b, v_h, v_i, v_k], value_T[v_b, v_h, v_k, v_j])
                            Ts.writes(C_global[v_b, v_h, v_i, v_j])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            with Ts.init():
                                C_global[v_b, v_h, v_i, v_j] = T.float32(0)
                            C_global[v_b, v_h, v_i, v_j] = C_global[v_b, v_h, v_i, v_j] + query_T[v_b, v_h, v_i, v_k] * value_T[v_b, v_h, v_k, v_j]
                for ax0, ax1, ax2, ax3 in T.grid(1, 12, 128, 64):
                    with Ts.sblock("C_global"):
                        v0, v1, v2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                        v3 = Ts.axis.spatial(128, j_0 * 64 + ax3)
                        Ts.reads(C_global[v0, v1, v2, v3])
                        Ts.writes(C[v0, v1, v2, v3])
                        C[v0, v1, v2, v3] = C_global[v0, v1, v2, v3]
    @Ts.function
    def tbg_2(query: T.Tensor((1, 128, 12, 64), "float32"), value: T.Tensor((1, 128, 12, 64), "float32"), C: T.Tensor((1, 12, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.parallel": 288, "meta_schedule.unroll_explicit": 512, "meta_schedule.vectorize": 64})
            value_T = Ts.sblock_alloc_tensor((1, 12, 64, 128))
            for b_0, h_0, i_0, j_0, b_1, h_1, i_1, j_1 in T.grid(1, 1, 1, 2, 1, 6, 2, 8):
                for ax0, ax1, ax2, ax3 in T.grid(1, 2, 64, 8):
                    with Ts.sblock("value_T"):
                        v_b = Ts.axis.spatial(1, ax0)
                        v_h = Ts.axis.spatial(12, h_1 * 2 + ax1)
                        v_d = Ts.axis.spatial(64, ax2)
                        v_l = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + ax3)
                        Ts.reads(value[v_b, v_l, v_h, v_d])
                        Ts.writes(value_T[v_b, v_h, v_d, v_l])
                        value_T[v_b, v_h, v_d, v_l] = value[v_b, v_l, v_h, v_d]
                for k_0, b_2, h_2, i_2, j_2, k_1, b_3, h_3, i_3, j_3 in T.grid(1, 1, 2, 2, 4, 64, 1, 1, 32, 2):
                    with Ts.sblock("C"):
                        v_b = Ts.axis.spatial(1, b_0 + b_1 + b_2 + b_3)
                        v_h = Ts.axis.spatial(12, h_0 * 12 + h_1 * 2 + h_2 + h_3)
                        v_i = Ts.axis.spatial(128, i_0 * 128 + i_1 * 64 + i_2 * 32 + i_3)
                        v_j = Ts.axis.spatial(128, j_0 * 64 + j_1 * 8 + j_2 * 2 + j_3)
                        v_k = Ts.axis.reduce(64, k_0 * 64 + k_1)
                        Ts.reads(query[v_b, v_i, v_h, v_k], value_T[v_b, v_h, v_k, v_j])
                        Ts.writes(C[v_b, v_h, v_i, v_j])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        with Ts.init():
                            C[v_b, v_h, v_i, v_j] = T.float32(0)
                        C[v_b, v_h, v_i, v_j] = C[v_b, v_h, v_i, v_j] + query[v_b, v_i, v_h, v_k] * value_T[v_b, v_h, v_k, v_j]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 6, 2, 1]),
        ("SamplePerfectTile", [1, 2, 2, 32]),
        ("SamplePerfectTile", [2, 8, 4, 2]),
        ("SamplePerfectTile", [1, 64]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 6),
        ("SampleComputeLocation", 6),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 6, 2, 1]),
        ("SamplePerfectTile", [1, 2, 2, 32]),
        ("SamplePerfectTile", [2, 8, 4, 2]),
        ("SamplePerfectTile", [1, 64]),
        ("SampleCategorical", 2),
        ("SampleComputeLocation", 13),
        ("SampleComputeLocation", -1),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 6, 2, 1]),
        ("SamplePerfectTile", [1, 2, 2, 32]),
        ("SamplePerfectTile", [2, 8, 4, 2]),
        ("SamplePerfectTile", [1, 64]),
        ("SampleCategorical", 3),
        ("SampleComputeLocation", 7),
        ("SampleComputeLocation", -2),
    ]
    mod = create_te_workload("TBG", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[tbg_0, tbg_1, tbg_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


if __name__ == "__main__":
    test_cpu_c1d()
    test_cpu_c2d()
    test_cpu_c3d()
    test_cpu_cap()
    test_cpu_dep()
    test_cpu_dil()
    test_cpu_gmm()
    test_cpu_grp()
    test_cpu_t2d()
    test_cpu_nrm()
    test_cpu_sfm()
    test_cpu_cbr()
    test_cpu_tbg()
