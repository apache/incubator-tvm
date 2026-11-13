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
# ruff: noqa: E501, F401
"""Tests for MetaSchedule search space on CUDA"""

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
    return Target("nvidia/geforce-rtx-2080")  # disable async trace using sm75


def _design_space(mod):
    return generate_design_space(
        kind="cuda",
        mod=mod,
        target=_target(),
        types=ms.ScheduleRule,
    )


def test_cuda_c1d():
    # fmt: off
    @Ts.function
    def c1d_0(inputs: T.Tensor((1, 256, 64), "float32"), weight: T.Tensor((3, 64, 128), "float32"), conv1d_nlc: T.Tensor((1, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 16})
            conv1d_nlc_local = Ts.sblock_alloc_tensor((1, 128, 128), scope="local")
            PadInput_shared = Ts.sblock_alloc_tensor((1, 258, 64), scope="shared")
            weight_shared = Ts.sblock_alloc_tensor((3, 64, 128), scope="shared")
            for n_0_l_0_co_0_fused in T.parallel(4, annotations={"thread_binding": "blockIdx.x"}):
                for n_1_l_1_co_1_fused in T.parallel(16, annotations={"thread_binding": "vthread.x"}):
                    for n_2_l_2_co_2_fused in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                        for rl_0, rc_0 in T.grid(1, 16):
                            for ax0_ax1_ax2_fused in range(260):
                                with Ts.sblock("PadInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(258, n_0_l_0_co_0_fused * 64 + ax0_ax1_ax2_fused // 4)
                                    v2 = Ts.axis.spatial(64, rc_0 * 4 + ax0_ax1_ax2_fused % 4)
                                    Ts.reads(inputs[v0, v1 - 1, v2])
                                    Ts.writes(PadInput_shared[v0, v1, v2])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                    PadInput_shared[v0, v1, v2] = T.if_then_else(1 <= v1 and v1 < 257, inputs[v0, v1 - 1, v2], T.float32(0))
                            for ax0_ax1_ax2_fused in range(1536):
                                with Ts.sblock("weight_shared"):
                                    v0 = Ts.axis.spatial(3, ax0_ax1_ax2_fused // 512)
                                    v1 = Ts.axis.spatial(64, rc_0 * 4 + ax0_ax1_ax2_fused % 512 // 128)
                                    v2 = Ts.axis.spatial(128, ax0_ax1_ax2_fused % 128)
                                    Ts.reads(weight[v0, v1, v2])
                                    Ts.writes(weight_shared[v0, v1, v2])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 3})
                                    weight_shared[v0, v1, v2] = weight[v0, v1, v2]
                            for rl_1, rc_1, n_3, l_3, co_3, rl_2, rc_2, n_4, l_4, co_4 in T.grid(1, 2, 1, 1, 2, 3, 2, 1, 4, 8):
                                with Ts.sblock("conv1d_nlc"):
                                    v_n = Ts.axis.spatial(1, n_3 + n_4)
                                    v_l = Ts.axis.spatial(128, n_0_l_0_co_0_fused * 32 + n_1_l_1_co_1_fused // 2 * 4 + l_3 * 4 + l_4)
                                    v_co = Ts.axis.spatial(128, n_1_l_1_co_1_fused % 2 * 64 + n_2_l_2_co_2_fused * 16 + co_3 * 8 + co_4)
                                    v_rl = Ts.axis.reduce(3, rl_0 * 3 + rl_1 * 3 + rl_2)
                                    v_rc = Ts.axis.reduce(64, rc_0 * 4 + rc_1 * 2 + rc_2)
                                    Ts.reads(PadInput_shared[v_n, v_l * 2 + v_rl, v_co // 128 * 64 + v_rc], weight_shared[v_rl, v_rc, v_co])
                                    Ts.writes(conv1d_nlc_local[v_n, v_l, v_co])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        conv1d_nlc_local[v_n, v_l, v_co] = T.float32(0)
                                    conv1d_nlc_local[v_n, v_l, v_co] = conv1d_nlc_local[v_n, v_l, v_co] + PadInput_shared[v_n, v_l * 2 + v_rl, v_co // 128 * 64 + v_rc] * weight_shared[v_rl, v_rc, v_co]
                        for ax0, ax1, ax2 in T.grid(1, 4, 16):
                            with Ts.sblock("conv1d_nlc_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(128, n_0_l_0_co_0_fused * 32 + n_1_l_1_co_1_fused // 2 * 4 + ax1)
                                v2 = Ts.axis.spatial(128, n_1_l_1_co_1_fused % 2 * 64 + n_2_l_2_co_2_fused * 16 + ax2)
                                Ts.reads(conv1d_nlc_local[v0, v1, v2])
                                Ts.writes(conv1d_nlc[v0, v1, v2])
                                conv1d_nlc[v0, v1, v2] = conv1d_nlc_local[v0, v1, v2]
    # fmt: on

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 8, 1, 1, 4]),
        ("SamplePerfectTile", [1, 2, 4, 2, 8]),
        ("SamplePerfectTile", [1, 1, 3]),
        ("SamplePerfectTile", [16, 2, 2]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]

    mod = create_te_workload("C1D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c1d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_c2d():
    # fmt: off
    @Ts.function
    def c2d_0(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 16})
            conv2d_nhwc_local = Ts.sblock_alloc_tensor((1, 112, 112, 64), scope="local")
            PadInput_shared = Ts.sblock_alloc_tensor((1, 230, 230, 3), scope="shared")
            weight_shared = Ts.sblock_alloc_tensor((7, 7, 3, 64), scope="shared")
            for n_0_h_0_w_0_co_0_fused in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
                for n_1_h_1_w_1_co_1_fused in T.parallel(56, annotations={"thread_binding": "vthread.x"}):
                    for n_2_h_2_w_2_co_2_fused in T.parallel(14, annotations={"thread_binding": "threadIdx.x"}):
                        for rh_0, rw_0, rc_0 in T.grid(1, 1, 1):
                            for ax0_ax1_ax2_ax3_fused in range(80379):
                                with Ts.sblock("PadInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(230, ax0_ax1_ax2_ax3_fused // 351)
                                    v2 = Ts.axis.spatial(230, n_0_h_0_w_0_co_0_fused // 8 * 112 + ax0_ax1_ax2_ax3_fused % 351 // 3)
                                    v3 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 3)
                                    Ts.reads(inputs[v0, v1 - 3, v2 - 3, v3])
                                    Ts.writes(PadInput_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, inputs[v0, v1 - 3, v2 - 3, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(1176):
                                with Ts.sblock("weight_shared"):
                                    v0 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_fused // 168)
                                    v1 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_fused % 168 // 24)
                                    v2 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 24 // 8)
                                    v3 = Ts.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 8 * 8 + ax0_ax1_ax2_ax3_fused % 8)
                                    Ts.reads(weight[v0, v1, v2, v3])
                                    Ts.writes(weight_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(1, 7, 1, 1, 8, 4, 1, 7, 1, 3, 1, 1, 1, 2):
                                with Ts.sblock("conv2d_nhwc"):
                                    v_n = Ts.axis.spatial(1, n_3 + n_4)
                                    v_h = Ts.axis.spatial(112, n_2_h_2_w_2_co_2_fused * 8 + h_3 + h_4)
                                    v_w = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused // 8 * 56 + n_1_h_1_w_1_co_1_fused // 4 * 4 + w_3 + w_4)
                                    v_co = Ts.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 8 * 8 + n_1_h_1_w_1_co_1_fused % 4 * 2 + co_3 * 2 + co_4)
                                    v_rh = Ts.axis.reduce(7, rh_0 * 7 + rh_1 * 7 + rh_2)
                                    v_rw = Ts.axis.reduce(7, rw_0 * 7 + rw_1 + rw_2)
                                    v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1 * 3 + rc_2)
                                    Ts.reads(PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight_shared[v_rh, v_rw, v_rc, v_co])
                                    Ts.writes(conv2d_nhwc_local[v_n, v_h, v_w, v_co])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        conv2d_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                    conv2d_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_nhwc_local[v_n, v_h, v_w, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight_shared[v_rh, v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 8, 4, 2):
                            with Ts.sblock("conv2d_nhwc_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(112, n_2_h_2_w_2_co_2_fused * 8 + ax1)
                                v2 = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused // 8 * 56 + n_1_h_1_w_1_co_1_fused // 4 * 4 + ax2)
                                v3 = Ts.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 8 * 8 + n_1_h_1_w_1_co_1_fused % 4 * 2 + ax3)
                                Ts.reads(conv2d_nhwc_local[v0, v1, v2, v3])
                                Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 14, 8, 1]),
        ("SamplePerfectTile", [2, 14, 1, 4, 1]),
        ("SamplePerfectTile", [8, 4, 1, 1, 2]),
        ("SamplePerfectTile", [1, 1, 7]),
        ("SamplePerfectTile", [1, 7, 1]),
        ("SamplePerfectTile", [1, 1, 3]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 1),
    ]

    mod = create_te_workload("C2D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c2d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_c3d():
    # fmt: off
    @Ts.function
    def c3d_0(inputs: T.Tensor((1, 16, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 7, 3, 64), "float32"), conv3d_ndhwc: T.Tensor((1, 8, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 16})
            conv3d_ndhwc_local = Ts.sblock_alloc_tensor((1, 8, 112, 112, 64), scope="local")
            PadInput_shared = Ts.sblock_alloc_tensor((1, 22, 230, 230, 3), scope="shared")
            weight_shared = Ts.sblock_alloc_tensor((7, 7, 7, 3, 64), scope="shared")
            for n_0_d_0_h_0_w_0_co_0_fused in T.parallel(2, annotations={"thread_binding": "blockIdx.x"}):
                for n_1_d_1_h_1_w_1_co_1_fused in T.parallel(8, annotations={"thread_binding": "vthread.x"}):
                    for n_2_d_2_h_2_w_2_co_2_fused in T.parallel(392, annotations={"thread_binding": "threadIdx.x"}):
                        for rd_0, rh_0, rw_0, rc_0 in T.grid(1, 1, 1, 1):
                            for ax0_ax1_ax2_ax3_ax4_fused in range(1687959):
                                with Ts.sblock("PadInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(22, ax0_ax1_ax2_ax3_ax4_fused // 80379)
                                    v2 = Ts.axis.spatial(230, ax0_ax1_ax2_ax3_ax4_fused % 80379 // 351)
                                    v3 = Ts.axis.spatial(230, n_0_d_0_h_0_w_0_co_0_fused * 112 + ax0_ax1_ax2_ax3_ax4_fused % 351 // 3)
                                    v4 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_ax4_fused % 3)
                                    Ts.reads(inputs[v0, v1 - 3, v2 - 3, v3 - 3, v4])
                                    Ts.writes(PadInput_shared[v0, v1, v2, v3, v4])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                    PadInput_shared[v0, v1, v2, v3, v4] = T.if_then_else(3 <= v1 and v1 < 19 and 3 <= v2 and v2 < 227 and 3 <= v3 and v3 < 227, inputs[v0, v1 - 3, v2 - 3, v3 - 3, v4], T.float32(0))
                            for ax0_ax1_ax2_ax3_ax4_fused in range(65856):
                                with Ts.sblock("weight_shared"):
                                    v0 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused // 9408)
                                    v1 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused % 9408 // 1344)
                                    v2 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_ax4_fused % 1344 // 192)
                                    v3 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_ax4_fused % 192 // 64)
                                    v4 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_ax4_fused % 64)
                                    Ts.reads(weight[v0, v1, v2, v3, v4])
                                    Ts.writes(weight_shared[v0, v1, v2, v3, v4])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 3})
                                    weight_shared[v0, v1, v2, v3, v4] = weight[v0, v1, v2, v3, v4]
                            for rd_1, rh_1, rw_1, rc_1, n_3, d_3, h_3, w_3, co_3, rd_2, rh_2, rw_2, rc_2, n_4, d_4, h_4, w_4, co_4 in T.grid(7, 7, 1, 3, 1, 2, 2, 1, 32, 1, 1, 7, 1, 1, 1, 2, 4, 1):
                                with Ts.sblock("conv3d_ndhwc"):
                                    v_n = Ts.axis.spatial(1, n_3 + n_4)
                                    v_d = Ts.axis.spatial(8, n_2_d_2_h_2_w_2_co_2_fused // 98 * 2 + d_3 + d_4)
                                    v_h = Ts.axis.spatial(112, n_1_d_1_h_1_w_1_co_1_fused // 2 * 28 + n_2_d_2_h_2_w_2_co_2_fused % 98 // 14 * 4 + h_3 * 2 + h_4)
                                    v_w = Ts.axis.spatial(112, n_0_d_0_h_0_w_0_co_0_fused * 56 + n_1_d_1_h_1_w_1_co_1_fused % 2 * 28 + n_2_d_2_h_2_w_2_co_2_fused % 14 // 2 * 4 + w_3 * 4 + w_4)
                                    v_co = Ts.axis.spatial(64, n_2_d_2_h_2_w_2_co_2_fused % 2 * 32 + co_3 + co_4)
                                    v_rd = Ts.axis.reduce(7, rd_0 * 7 + rd_1 + rd_2)
                                    v_rh = Ts.axis.reduce(7, rh_0 * 7 + rh_1 + rh_2)
                                    v_rw = Ts.axis.reduce(7, rw_0 * 7 + rw_1 * 7 + rw_2)
                                    v_rc = Ts.axis.reduce(3, rc_0 * 3 + rc_1 + rc_2)
                                    Ts.reads(PadInput_shared[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight_shared[v_rd, v_rh, v_rw, v_rc, v_co])
                                    Ts.writes(conv3d_ndhwc_local[v_n, v_d, v_h, v_w, v_co])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        conv3d_ndhwc_local[v_n, v_d, v_h, v_w, v_co] = T.float32(0)
                                    conv3d_ndhwc_local[v_n, v_d, v_h, v_w, v_co] = conv3d_ndhwc_local[v_n, v_d, v_h, v_w, v_co] + PadInput_shared[v_n, v_d * 2 + v_rd, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight_shared[v_rd, v_rh, v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 2, 4, 4, 32):
                            with Ts.sblock("conv3d_ndhwc_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(8, n_2_d_2_h_2_w_2_co_2_fused // 98 * 2 + ax1)
                                v2 = Ts.axis.spatial(112, n_1_d_1_h_1_w_1_co_1_fused // 2 * 28 + n_2_d_2_h_2_w_2_co_2_fused % 98 // 14 * 4 + ax2)
                                v3 = Ts.axis.spatial(112, n_0_d_0_h_0_w_0_co_0_fused * 56 + n_1_d_1_h_1_w_1_co_1_fused % 2 * 28 + n_2_d_2_h_2_w_2_co_2_fused % 14 // 2 * 4 + ax3)
                                v4 = Ts.axis.spatial(64, n_2_d_2_h_2_w_2_co_2_fused % 2 * 32 + ax4)
                                Ts.reads(conv3d_ndhwc_local[v0, v1, v2, v3, v4])
                                Ts.writes(conv3d_ndhwc[v0, v1, v2, v3, v4])
                                conv3d_ndhwc[v0, v1, v2, v3, v4] = conv3d_ndhwc_local[v0, v1, v2, v3, v4]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4, 2, 1]),
        ("SamplePerfectTile", [1, 4, 7, 2, 2]),
        ("SamplePerfectTile", [2, 2, 7, 1, 4]),
        ("SamplePerfectTile", [1, 1, 2, 32, 1]),
        ("SamplePerfectTile", [1, 7, 1]),
        ("SamplePerfectTile", [1, 7, 1]),
        ("SamplePerfectTile", [1, 1, 7]),
        ("SamplePerfectTile", [1, 3, 1]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]
    mod = create_te_workload("C3D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[c3d_0],
        expected_decisions=[decision_0],
    )


def test_cuda_cap():
    # fmt: off
    @Ts.function
    def cap_0(inputs: T.Tensor((1, 16, 16, 4, 4, 32), "float32"), weight: T.Tensor((3, 3, 4, 4, 32, 32), "float32"), conv2d_capsule_nhwijc: T.Tensor((1, 8, 8, 4, 4, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 64})
            conv2d_capsule_nhwijc_local = Ts.sblock_alloc_tensor((1, 8, 8, 4, 4, 32), scope="local")
            PadInput_shared = Ts.sblock_alloc_tensor((1, 18, 18, 4, 4, 32), scope="shared")
            weight_shared = Ts.sblock_alloc_tensor((3, 3, 4, 4, 32, 32), scope="shared")
            for n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for n_1_h_1_w_1_cap_i_1_cap_j_1_co_1_fused in T.parallel(1, annotations={"thread_binding": "vthread.x"}):
                    for n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                        for rh_0, rw_0, cap_k_0, rc_0 in T.grid(3, 3, 2, 8):
                            for ax0_ax1_ax2_ax3_ax4_ax5_fused in range(48):
                                with Ts.sblock("PadInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(18, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused // 64 * 4 + rh_0 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 48 // 16)
                                    v2 = Ts.axis.spatial(18, T.Add(n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 64 // 8 * 2 + rw_0, 0))
                                    v3 = Ts.axis.spatial(4, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 8 // 4 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 16 // 8)
                                    v4 = Ts.axis.spatial(4, cap_k_0 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 8 // 4)
                                    v5 = Ts.axis.spatial(32, rc_0 * 4 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 4)
                                    Ts.reads(inputs[v0, v1 - 1, v2 - 1, v3, v4, v5])
                                    Ts.writes(PadInput_shared[v0, v1, v2, v3, v4, v5])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3, v4, v5] = T.if_then_else(1 <= v1 and v1 < 17 and 1 <= v2 and v2 < 17, inputs[v0, v1 - 1, v2 - 1, v3, v4, v5], T.float32(0))
                            for ax0_ax1_ax2_ax3_ax4_ax5_fused in range(256):
                                with Ts.sblock("weight_shared"):
                                    v0, v1 = Ts.axis.remap("SS", [rh_0, rw_0])
                                    v2 = Ts.axis.spatial(4, cap_k_0 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_fused // 128)
                                    v3 = Ts.axis.spatial(4, ax0_ax1_ax2_ax3_ax4_ax5_fused % 128 // 32)
                                    v4 = Ts.axis.spatial(32, rc_0 * 4 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 32 // 8)
                                    v5 = Ts.axis.spatial(32, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 4 * 8 + ax0_ax1_ax2_ax3_ax4_ax5_fused % 8)
                                    Ts.reads(weight[v0, v1, v2, v3, v4, v5])
                                    Ts.writes(weight_shared[v0, v1, v2, v3, v4, v5])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                    weight_shared[v0, v1, v2, v3, v4, v5] = weight[v0, v1, v2, v3, v4, v5]
                            for rh_1, rw_1, cap_k_1, rc_1, n_3, h_3, w_3, cap_i_3, cap_j_3, co_3, rh_2, rw_2, cap_k_2, rc_2, n_4, h_4, w_4, cap_i_4, cap_j_4, co_4 in T.grid(1, 1, 1, 4, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 8):
                                with Ts.sblock("conv2d_capsule_nhwijc"):
                                    v_n = Ts.axis.spatial(1, n_3 + n_4)
                                    v_h = Ts.axis.spatial(8, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused // 64 * 2 + h_3 + h_4)
                                    v_w = Ts.axis.spatial(8, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 64 // 8 + w_3 + w_4)
                                    v_cap_i = Ts.axis.spatial(4, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 8 // 4 * 2 + n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused // 2 + cap_i_3 + cap_i_4)
                                    v_cap_j = Ts.axis.spatial(4, n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused % 2 * 2 + cap_j_3 * 2 + cap_j_4)
                                    v_co = Ts.axis.spatial(32, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 4 * 8 + co_3 * 8 + co_4)
                                    v_rh = Ts.axis.reduce(3, rh_0 + rh_1 + rh_2)
                                    v_rw = Ts.axis.reduce(3, rw_0 + rw_1 + rw_2)
                                    v_cap_k = Ts.axis.reduce(4, cap_k_0 * 2 + cap_k_1 * 2 + cap_k_2)
                                    v_rc = Ts.axis.reduce(32, rc_0 * 4 + rc_1 + rc_2)
                                    Ts.reads(PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc], weight_shared[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co])
                                    Ts.writes(conv2d_capsule_nhwijc_local[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        conv2d_capsule_nhwijc_local[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = T.float32(0)
                                    conv2d_capsule_nhwijc_local[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] = conv2d_capsule_nhwijc_local[v_n, v_h, v_w, v_cap_i, v_cap_j, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_cap_i, v_cap_k, v_rc] * weight_shared[v_rh, v_rw, v_cap_k, v_cap_j, v_rc, v_co]
                        for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 2, 1, 1, 2, 8):
                            with Ts.sblock("conv2d_capsule_nhwijc_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(8, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused // 64 * 2 + ax1)
                                v2 = Ts.axis.spatial(8, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 64 // 8 + ax2)
                                v3 = Ts.axis.spatial(4, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 8 // 4 * 2 + n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused // 2 + ax3)
                                v4 = Ts.axis.spatial(4, n_2_h_2_w_2_cap_i_2_cap_j_2_co_2_fused % 2 * 2 + ax4)
                                v5 = Ts.axis.spatial(32, n_0_h_0_w_0_cap_i_0_cap_j_0_co_0_fused % 4 * 8 + ax5)
                                Ts.reads(conv2d_capsule_nhwijc_local[v0, v1, v2, v3, v4, v5])
                                Ts.writes(conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5])
                                conv2d_capsule_nhwijc[v0, v1, v2, v3, v4, v5] = conv2d_capsule_nhwijc_local[v0, v1, v2, v3, v4, v5]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 1, 2, 1]),
        ("SamplePerfectTile", [8, 1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 1, 2, 1, 1]),
        ("SamplePerfectTile", [1, 1, 2, 1, 2]),
        ("SamplePerfectTile", [4, 1, 1, 1, 8]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SamplePerfectTile", [2, 1, 2]),
        ("SamplePerfectTile", [8, 4, 1]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 3),
    ]
    mod = create_te_workload("CAP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cap_0],
        expected_decisions=[decision_0],
    )


def test_cuda_dep():
    # fmt: off
    @Ts.function
    def dep_0(placeholder: T.Tensor((1, 112, 112, 32), "float32"), placeholder_1: T.Tensor((1, 3, 3, 32), "float32"), depth_conv2d_nhwc: T.Tensor((1, 112, 112, 32), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 16})
            depth_conv2d_nhwc_local = Ts.sblock_alloc_tensor((1, 112, 112, 32), scope="local")
            PadInput_shared = Ts.sblock_alloc_tensor((1, 114, 114, 32), scope="shared")
            placeholder_shared = Ts.sblock_alloc_tensor((1, 3, 3, 32), scope="shared")
            for n_0_h_0_w_0_c_0_fused in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for n_1_h_1_w_1_c_1_fused in T.parallel(8, annotations={"thread_binding": "vthread.x"}):
                    for n_2_h_2_w_2_c_2_fused in T.parallel(14, annotations={"thread_binding": "threadIdx.x"}):
                        for rh_0, rw_0 in T.grid(1, 1):
                            for ax0_ax1_ax2_ax3_fused in range(415872):
                                with Ts.sblock("PadInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(114, ax0_ax1_ax2_ax3_fused // 3648)
                                    v2 = Ts.axis.spatial(114, ax0_ax1_ax2_ax3_fused % 3648 // 32)
                                    v3 = Ts.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 32)
                                    Ts.reads(placeholder[v0, v1 - 1, v2 - 1, v3])
                                    Ts.writes(PadInput_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 3})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(1 <= v1 and v1 < 113 and 1 <= v2 and v2 < 113, placeholder[v0, v1 - 1, v2 - 1, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(288):
                                with Ts.sblock("placeholder_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused // 96)
                                    v2 = Ts.axis.spatial(3, ax0_ax1_ax2_ax3_fused % 96 // 32)
                                    v3 = Ts.axis.spatial(32, ax0_ax1_ax2_ax3_fused % 32)
                                    Ts.reads(placeholder_1[v0, v1, v2, v3])
                                    Ts.writes(placeholder_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 3})
                                    placeholder_shared[v0, v1, v2, v3] = placeholder_1[v0, v1, v2, v3]
                            for rh_1, rw_1, n_3, h_3, w_3, c_3, rh_2, rw_2, n_4, h_4, w_4, c_4 in T.grid(3, 1, 1, 4, 16, 8, 1, 3, 1, 7, 1, 1):
                                with Ts.sblock("depth_conv2d_nhwc"):
                                    v_n = Ts.axis.spatial(1, n_3 + n_4)
                                    v_h = Ts.axis.spatial(112, n_1_h_1_w_1_c_1_fused // 2 * 28 + h_3 * 7 + h_4)
                                    v_w = Ts.axis.spatial(112, n_2_h_2_w_2_c_2_fused // 2 * 16 + w_3 + w_4)
                                    v_c = Ts.axis.spatial(32, n_1_h_1_w_1_c_1_fused % 2 * 16 + n_2_h_2_w_2_c_2_fused % 2 * 8 + c_3 + c_4)
                                    v_rh = Ts.axis.reduce(3, rh_0 * 3 + rh_1 + rh_2)
                                    v_rw = Ts.axis.reduce(3, rw_0 * 3 + rw_1 * 3 + rw_2)
                                    Ts.reads(PadInput_shared[v_n, v_h + v_rh, v_w + v_rw, v_c], placeholder_shared[0, v_rh, v_rw, v_c])
                                    Ts.writes(depth_conv2d_nhwc_local[v_n, v_h, v_w, v_c])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        depth_conv2d_nhwc_local[v_n, v_h, v_w, v_c] = T.float32(0)
                                    depth_conv2d_nhwc_local[v_n, v_h, v_w, v_c] = depth_conv2d_nhwc_local[v_n, v_h, v_w, v_c] + PadInput_shared[v_n, v_h + v_rh, v_w + v_rw, v_c] * placeholder_shared[0, v_rh, v_rw, v_c]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 28, 16, 8):
                            with Ts.sblock("depth_conv2d_nhwc_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(112, n_1_h_1_w_1_c_1_fused // 2 * 28 + ax1)
                                v2 = Ts.axis.spatial(112, n_2_h_2_w_2_c_2_fused // 2 * 16 + ax2)
                                v3 = Ts.axis.spatial(32, n_1_h_1_w_1_c_1_fused % 2 * 16 + n_2_h_2_w_2_c_2_fused % 2 * 8 + ax3)
                                Ts.reads(depth_conv2d_nhwc_local[v0, v1, v2, v3])
                                Ts.writes(depth_conv2d_nhwc[v0, v1, v2, v3])
                                depth_conv2d_nhwc[v0, v1, v2, v3] = depth_conv2d_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 4, 1, 4, 7]),
        ("SamplePerfectTile", [1, 1, 7, 16, 1]),
        ("SamplePerfectTile", [1, 2, 2, 8, 1]),
        ("SamplePerfectTile", [1, 3, 1]),
        ("SamplePerfectTile", [1, 1, 3]),
        ("SampleCategorical", 2),
        ("SampleCategorical", 2),
        ("SampleCategorical", 1),
    ]
    mod = create_te_workload("DEP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dep_0],
        expected_decisions=[decision_0],
    )


def test_cuda_dil():
    # fmt: off
    @Ts.function
    def dil_0(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 109, 109, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 512})
            conv2d_nhwc_local = Ts.sblock_alloc_tensor((1, 109, 109, 64), scope="local")
            PadInput_shared = Ts.sblock_alloc_tensor((1, 230, 230, 3), scope="shared")
            weight_shared = Ts.sblock_alloc_tensor((7, 7, 3, 64), scope="shared")
            for n_0_h_0_w_0_co_0_fused in T.parallel(218, annotations={"thread_binding": "blockIdx.x"}):
                for n_1_h_1_w_1_co_1_fused in T.parallel(109, annotations={"thread_binding": "vthread.x"}):
                    for n_2_h_2_w_2_co_2_fused in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                        for rh_0, rw_0, rc_0 in T.grid(7, 7, 3):
                            for ax0_ax1_ax2_ax3_fused in range(217):
                                with Ts.sblock("PadInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(230, T.Add(n_0_h_0_w_0_co_0_fused // 2 * 2 + rh_0 * 2, 0))
                                    v2 = Ts.axis.spatial(230, rw_0 * 2 + ax0_ax1_ax2_ax3_fused % 217)
                                    v3 = Ts.axis.spatial(3, T.Add(rc_0, 0))
                                    Ts.reads(inputs[v0, v1 - 3, v2 - 3, v3])
                                    Ts.writes(PadInput_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, inputs[v0, v1 - 3, v2 - 3, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(32):
                                with Ts.sblock("weight_shared"):
                                    v0, v1, v2 = Ts.axis.remap("SSS", [rh_0, rw_0, rc_0])
                                    v3 = Ts.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 2 * 32 + ax0_ax1_ax2_ax3_fused)
                                    Ts.reads(weight[v0, v1, v2, v3])
                                    Ts.writes(weight_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 4):
                                with Ts.sblock("conv2d_nhwc"):
                                    v_n = Ts.axis.spatial(1, n_3 + n_4)
                                    v_h = Ts.axis.spatial(109, n_0_h_0_w_0_co_0_fused // 2 + h_3 + h_4)
                                    v_w = Ts.axis.spatial(109, n_1_h_1_w_1_co_1_fused + w_3 + w_4)
                                    v_co = Ts.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 2 * 32 + co_3 * 4 + co_4)
                                    v_rh = Ts.axis.reduce(7, rh_0 + rh_1 + rh_2)
                                    v_rw = Ts.axis.reduce(7, rw_0 + rw_1 + rw_2)
                                    v_rc = Ts.axis.reduce(3, rc_0 + rc_1 + rc_2)
                                    Ts.reads(PadInput_shared[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc], weight_shared[v_rh, v_rw, v_rc, v_co])
                                    Ts.writes(conv2d_nhwc_local[v_n, v_h, v_w, v_co])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        conv2d_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                    conv2d_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_nhwc_local[v_n, v_h, v_w, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh * 2, v_w * 2 + v_rw * 2, v_co // 64 * 3 + v_rc] * weight_shared[v_rh, v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 32):
                            with Ts.sblock("conv2d_nhwc_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(109, n_0_h_0_w_0_co_0_fused // 2 + ax1)
                                v2 = Ts.axis.spatial(109, n_1_h_1_w_1_co_1_fused + ax2)
                                v3 = Ts.axis.spatial(64, n_0_h_0_w_0_co_0_fused % 2 * 32 + ax3)
                                Ts.reads(conv2d_nhwc_local[v0, v1, v2, v3])
                                Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [109, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 109, 1, 1, 1]),
        ("SamplePerfectTile", [2, 1, 1, 8, 4]),
        ("SamplePerfectTile", [7, 1, 1]),
        ("SamplePerfectTile", [7, 1, 1]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 6),
    ]
    mod = create_te_workload("DIL", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[dil_0],
        expected_decisions=[decision_0],
    )


def test_cuda_gmm():
    # fmt: off
    @Ts.function
    def gmm_0(X: T.Tensor((1, 128, 128), "float32"), Y: T.Tensor((1, 128, 128), "float32"), Z: T.Tensor((1, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 1024})
            Z_local = Ts.sblock_alloc_tensor((1, 128, 128), scope="local")
            X_shared = Ts.sblock_alloc_tensor((1, 128, 128), scope="shared")
            Y_shared = Ts.sblock_alloc_tensor((1, 128, 128), scope="shared")
            for b_0_i_0_j_0_fused in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for b_1_i_1_j_1_fused in T.parallel(32, annotations={"thread_binding": "vthread.x"}):
                    for b_2_i_2_j_2_fused in T.parallel(2, annotations={"thread_binding": "threadIdx.x"}):
                        for k_0 in range(1):
                            for ax0_ax1_ax2_fused in range(16384):
                                with Ts.sblock("X_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(128, ax0_ax1_ax2_fused // 128)
                                    v2 = Ts.axis.spatial(128, ax0_ax1_ax2_fused % 128)
                                    Ts.reads(X[v0, v1, v2])
                                    Ts.writes(X_shared[v0, v1, v2])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                    X_shared[v0, v1, v2] = X[v0, v1, v2]
                            for ax0_ax1_ax2_fused in range(16384):
                                with Ts.sblock("Y_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(128, ax0_ax1_ax2_fused // 128)
                                    v2 = Ts.axis.spatial(128, ax0_ax1_ax2_fused % 128)
                                    Ts.reads(Y[v0, v1, v2])
                                    Ts.writes(Y_shared[v0, v1, v2])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                    Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                            for k_1, b_3, i_3, j_3, k_2, b_4, i_4, j_4 in T.grid(32, 1, 2, 64, 4, 1, 2, 1):
                                with Ts.sblock("Z"):
                                    v_b = Ts.axis.spatial(1, b_3 + b_4)
                                    v_i = Ts.axis.spatial(128, b_1_i_1_j_1_fused * 4 + i_3 * 2 + i_4)
                                    v_j = Ts.axis.spatial(128, b_2_i_2_j_2_fused * 64 + j_3 + j_4)
                                    v_k = Ts.axis.reduce(128, k_0 * 128 + k_1 * 4 + k_2)
                                    Ts.reads(X_shared[v_b, v_i, v_k], Y_shared[v_b, v_k, v_j])
                                    Ts.writes(Z_local[v_b, v_i, v_j])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        Z_local[v_b, v_i, v_j] = T.float32(0)
                                    Z_local[v_b, v_i, v_j] = Z_local[v_b, v_i, v_j] + X_shared[v_b, v_i, v_k] * Y_shared[v_b, v_k, v_j]
                        for ax0, ax1, ax2 in T.grid(1, 4, 64):
                            with Ts.sblock("Z_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(128, b_1_i_1_j_1_fused * 4 + ax1)
                                v2 = Ts.axis.spatial(128, b_2_i_2_j_2_fused * 64 + ax2)
                                Ts.reads(Z_local[v0, v1, v2])
                                Ts.writes(Z[v0, v1, v2])
                                Z[v0, v1, v2] = Z_local[v0, v1, v2]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 32, 1, 2, 2]),
        ("SamplePerfectTile", [1, 1, 2, 64, 1]),
        ("SamplePerfectTile", [1, 32, 4]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 0),
        ("SampleCategorical", 7),
    ]
    mod = create_te_workload("GMM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[gmm_0],
        expected_decisions=[decision_0],
    )


def test_cuda_grp():
    # fmt: off
    @Ts.function
    def grp_0(inputs: T.Tensor((1, 56, 56, 64), "float32"), weight: T.Tensor((3, 3, 16, 128), "float32"), conv2d_nhwc: T.Tensor((1, 28, 28, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 16})
            conv2d_nhwc_local = Ts.sblock_alloc_tensor((1, 28, 28, 128), scope="local")
            PadInput_shared = Ts.sblock_alloc_tensor((1, 58, 58, 64), scope="shared")
            weight_shared = Ts.sblock_alloc_tensor((3, 3, 16, 128), scope="shared")
            for n_0_h_0_w_0_co_0_fused in T.parallel(2, annotations={"thread_binding": "blockIdx.x"}):
                for n_1_h_1_w_1_co_1_fused in T.parallel(1, annotations={"thread_binding": "vthread.x"}):
                    for n_2_h_2_w_2_co_2_fused in T.parallel(112, annotations={"thread_binding": "threadIdx.x"}):
                        for rh_0, rw_0, rc_0 in T.grid(3, 3, 1):
                            for ax0_ax1_ax2_ax3_fused in range(95040):
                                with Ts.sblock("PadInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(58, n_0_h_0_w_0_co_0_fused * 28 + rh_0 + ax0_ax1_ax2_ax3_fused % 95040 // 3520)
                                    v2 = Ts.axis.spatial(58, rw_0 + ax0_ax1_ax2_ax3_fused % 3520 // 64)
                                    v3 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 64)
                                    Ts.reads(inputs[v0, v1 - 1, v2 - 1, v3])
                                    Ts.writes(PadInput_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(1 <= v1 and v1 < 57 and 1 <= v2 and v2 < 57, inputs[v0, v1 - 1, v2 - 1, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(2048):
                                with Ts.sblock("weight_shared"):
                                    v0, v1 = Ts.axis.remap("SS", [rh_0, rw_0])
                                    v2 = Ts.axis.spatial(16, ax0_ax1_ax2_ax3_fused // 128)
                                    v3 = Ts.axis.spatial(128, ax0_ax1_ax2_ax3_fused % 128)
                                    Ts.reads(weight[v0, v1, v2, v3])
                                    Ts.writes(weight_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(1, 1, 2, 1, 2, 1, 2, 1, 1, 8, 1, 7, 4, 4):
                                with Ts.sblock("conv2d_nhwc"):
                                    v_n = Ts.axis.spatial(1, n_3 + n_4)
                                    v_h = Ts.axis.spatial(28, n_0_h_0_w_0_co_0_fused * 14 + h_3 * 7 + h_4)
                                    v_w = Ts.axis.spatial(28, n_2_h_2_w_2_co_2_fused // 16 * 4 + w_3 * 4 + w_4)
                                    v_co = Ts.axis.spatial(128, n_2_h_2_w_2_co_2_fused % 16 * 8 + co_3 * 4 + co_4)
                                    v_rh = Ts.axis.reduce(3, rh_0 + rh_1 + rh_2)
                                    v_rw = Ts.axis.reduce(3, rw_0 + rw_1 + rw_2)
                                    v_rc = Ts.axis.reduce(16, rc_0 * 16 + rc_1 * 8 + rc_2)
                                    Ts.reads(PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc], weight_shared[v_rh, v_rw, v_rc, v_co])
                                    Ts.writes(conv2d_nhwc_local[v_n, v_h, v_w, v_co])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        conv2d_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                    conv2d_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_nhwc_local[v_n, v_h, v_w, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 32 * 16 + v_rc] * weight_shared[v_rh, v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 14, 4, 8):
                            with Ts.sblock("conv2d_nhwc_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(28, n_0_h_0_w_0_co_0_fused * 14 + ax1)
                                v2 = Ts.axis.spatial(28, n_2_h_2_w_2_co_2_fused // 16 * 4 + ax2)
                                v3 = Ts.axis.spatial(128, n_2_h_2_w_2_co_2_fused % 16 * 8 + ax3)
                                Ts.reads(conv2d_nhwc_local[v0, v1, v2, v3])
                                Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                                conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [2, 1, 1, 2, 7]),
        ("SamplePerfectTile", [1, 1, 7, 1, 4]),
        ("SamplePerfectTile", [1, 1, 16, 2, 4]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SamplePerfectTile", [1, 2, 8]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 0),
        ("SampleCategorical", 1),
    ]
    mod = create_te_workload("GRP", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[grp_0],
        expected_decisions=[decision_0],
    )


def test_cuda_t2d():
    # fmt: off
    @Ts.function
    def t2d_0(inputs: T.Tensor((1, 4, 4, 512), "float32"), weight: T.Tensor((4, 4, 512, 256), "float32"), conv2d_transpose_nhwc: T.Tensor((1, 8, 8, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 64})
            conv2d_transpose_nhwc_local = Ts.sblock_alloc_tensor((1, 8, 8, 256), scope="local")
            PadInput_shared = Ts.sblock_alloc_tensor((1, 6, 6, 512), scope="shared")
            weight_shared = Ts.sblock_alloc_tensor((4, 4, 512, 256), scope="shared")
            for n_0_h_0_w_0_co_0_fused in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for n_1_h_1_w_1_co_1_fused in T.parallel(2, annotations={"thread_binding": "vthread.x"}):
                    for n_2_h_2_w_2_co_2_fused in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                        for rh_0, rw_0, rc_0 in T.grid(4, 1, 16):
                            for ax0_ax1_ax2_ax3_fused in range(rh_0 % 2 * 96 + 96):
                                with Ts.sblock("PadInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(6, n_0_h_0_w_0_co_0_fused // 64 + rh_0 // 2 + ax0_ax1_ax2_ax3_fused % (rh_0 % 2 * 96 + 96) // 96)
                                    v2 = Ts.axis.spatial(6, n_0_h_0_w_0_co_0_fused % 64 // 16 + ax0_ax1_ax2_ax3_fused % 96 // 32)
                                    v3 = Ts.axis.spatial(512, rc_0 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    Ts.reads(inputs[v0, v1 - 1, v2 - 1, v3])
                                    Ts.writes(PadInput_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                    PadInput_shared[v0, v1, v2, v3] = T.if_then_else(1 <= v1 and v1 < 5 and 1 <= v2 and v2 < 5, inputs[v0, v1 - 1, v2 - 1, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(2048):
                                with Ts.sblock("weight_shared"):
                                    v0 = Ts.axis.spatial(4, rh_0 * -1 + 3)
                                    v1 = Ts.axis.spatial(4, ax0_ax1_ax2_ax3_fused // 512)
                                    v2 = Ts.axis.spatial(512, rc_0 * 32 + ax0_ax1_ax2_ax3_fused % 512 // 16)
                                    v3 = Ts.axis.spatial(256, n_0_h_0_w_0_co_0_fused % 16 * 16 + ax0_ax1_ax2_ax3_fused % 16)
                                    Ts.reads(weight[v0, v1, v2, v3])
                                    Ts.writes(weight_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                    weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                            for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(1, 1, 4, 1, 2, 1, 8, 1, 4, 8, 1, 1, 2, 1):
                                with Ts.sblock("conv2d_transpose_nhwc"):
                                    v_n = Ts.axis.spatial(1, n_3 + n_4)
                                    v_h = Ts.axis.spatial(8, n_0_h_0_w_0_co_0_fused // 64 * 2 + h_3 + h_4)
                                    v_w = Ts.axis.spatial(8, n_0_h_0_w_0_co_0_fused % 64 // 16 * 2 + w_3 * 2 + w_4)
                                    v_co = Ts.axis.spatial(256, n_0_h_0_w_0_co_0_fused % 16 * 16 + n_1_h_1_w_1_co_1_fused * 8 + co_3 + co_4)
                                    v_rh = Ts.axis.reduce(4, rh_0 + rh_1 + rh_2)
                                    v_rw = Ts.axis.reduce(4, rw_0 * 4 + rw_1 * 4 + rw_2)
                                    v_rc = Ts.axis.reduce(512, rc_0 * 32 + rc_1 * 8 + rc_2)
                                    Ts.reads(PadInput_shared[v_n, (v_h + v_rh) // 2, (v_w + v_rw) // 2, v_rc], weight_shared[3 - v_rh, 3 - v_rw, v_rc, v_co])
                                    Ts.writes(conv2d_transpose_nhwc_local[v_n, v_h, v_w, v_co])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        conv2d_transpose_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                    conv2d_transpose_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_transpose_nhwc_local[v_n, v_h, v_w, v_co] + T.if_then_else((v_h + v_rh) % 2 == 0 and (v_w + v_rw) % 2 == 0, PadInput_shared[v_n, (v_h + v_rh) // 2, (v_w + v_rw) // 2, v_rc], T.float32(0)) * weight_shared[3 - v_rh, 3 - v_rw, v_rc, v_co]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 2, 2, 8):
                            with Ts.sblock("conv2d_transpose_nhwc_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(8, n_0_h_0_w_0_co_0_fused // 64 * 2 + ax1)
                                v2 = Ts.axis.spatial(8, n_0_h_0_w_0_co_0_fused % 64 // 16 * 2 + ax2)
                                v3 = Ts.axis.spatial(256, n_0_h_0_w_0_co_0_fused % 16 * 16 + n_1_h_1_w_1_co_1_fused * 8 + ax3)
                                Ts.reads(conv2d_transpose_nhwc_local[v0, v1, v2, v3])
                                Ts.writes(conv2d_transpose_nhwc[v0, v1, v2, v3])
                                conv2d_transpose_nhwc[v0, v1, v2, v3] = conv2d_transpose_nhwc_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [4, 1, 1, 2, 1]),
        ("SamplePerfectTile", [4, 1, 1, 1, 2]),
        ("SamplePerfectTile", [16, 2, 1, 8, 1]),
        ("SamplePerfectTile", [4, 1, 1]),
        ("SamplePerfectTile", [1, 1, 4]),
        ("SamplePerfectTile", [16, 4, 8]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 3),
    ]
    mod = create_te_workload("T2D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[t2d_0],
        expected_decisions=[decision_0],
        debug_mask=0,
    )


def test_cuda_nrm():
    # fmt: off
    @Ts.function
    def nrm_0(A: T.Tensor((1, 256, 256), "float32"), D: T.Tensor(1, "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 512})
            C = Ts.sblock_alloc_tensor((1,))
            for b_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for b_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                    for i, j in T.grid(256, 256):
                        with Ts.sblock("C"):
                            v_b = Ts.axis.spatial(1, 0)
                            v_i, v_j = Ts.axis.remap("RR", [i, j])
                            Ts.reads(A[v_b, v_i, v_j])
                            Ts.writes(C[v_b])
                            with Ts.init():
                                C[v_b] = T.float32(0)
                            C[v_b] = C[v_b] + A[v_b, v_i, v_j] * A[v_b, v_i, v_j]
            for b_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for b_fused_1 in T.parallel(1, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("D"):
                        v_b = Ts.axis.spatial(1, 0)
                        Ts.reads(C[v_b])
                        Ts.writes(D[v_b])
                        D[v_b] = T.sqrt(C[v_b])
    @Ts.function
    def nrm_1(A: T.Tensor((1, 256, 256), "float32"), D: T.Tensor(1, "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 1024})
            C_shared = Ts.sblock_alloc_tensor((1,), scope="shared")
            for b_0_fused in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for ax0, ax1_ax2_fused_0 in T.grid(1, 512):
                    for ax1_ax2_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("C"):
                            v_b = Ts.axis.spatial(1, ax0)
                            v_i = Ts.axis.reduce(256, (ax1_ax2_fused_0 * 128 + ax1_ax2_fused_1) // 256)
                            v_j = Ts.axis.reduce(256, (ax1_ax2_fused_0 * 128 + ax1_ax2_fused_1) % 256)
                            Ts.reads(A[v_b, v_i, v_j])
                            Ts.writes(C_shared[v_b])
                            with Ts.init():
                                C_shared[v_b] = T.float32(0)
                            C_shared[v_b] = C_shared[v_b] + A[v_b, v_i, v_j] * A[v_b, v_i, v_j]
                for b_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("D"):
                        v_b = Ts.axis.spatial(1, b_1)
                        Ts.where(T.Mul(0, 128) + b_1 < 1)
                        Ts.reads(C_shared[v_b])
                        Ts.writes(D[v_b])
                        D[v_b] = T.sqrt(C_shared[v_b])
    # fmt: on
    decision_0 = [
        ("SampleCategorical", 6),
    ]
    decision_1 = [
        ("SampleCategorical", 5),
        ("SampleCategorical", 7),
    ]
    mod = create_te_workload("NRM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[nrm_0, nrm_1],
        expected_decisions=[decision_0, decision_1],
    )


def test_cuda_sfm():
    # fmt: off
    @Ts.function
    def sfm_0(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 0})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            for i0_fused_0 in T.parallel(2, annotations={"thread_binding": "blockIdx.x"}):
                for i0_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                    for k in range(256):
                        with Ts.sblock("T_softmax_maxelem"):
                            v_i0 = Ts.axis.spatial(256, i0_fused_0 * 128 + i0_fused_1)
                            v_k = Ts.axis.reduce(256, k)
                            Ts.reads(A[v_i0, v_k])
                            Ts.writes(T_softmax_maxelem[v_i0])
                            with Ts.init():
                                T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
            for i0_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
                for i0_fused_1 in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                    for k in range(256):
                        with Ts.sblock("T_softmax_expsum"):
                            v_i0 = Ts.axis.spatial(256, i0_fused_0 * 256 + i0_fused_1)
                            v_k = Ts.axis.reduce(256, k)
                            Ts.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                            Ts.writes(T_softmax_expsum[v_i0])
                            with Ts.init():
                                T_softmax_expsum[v_i0] = T.float32(0)
                            T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
            for i0_i1_fused_0 in T.parallel(1024, annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_norm"):
                        v_i0 = Ts.axis.spatial(256, (i0_i1_fused_0 * 64 + i0_i1_fused_1) // 256)
                        v_i1 = Ts.axis.spatial(256, (i0_i1_fused_0 * 64 + i0_i1_fused_1) % 256)
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
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 16})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum = Ts.sblock_alloc_tensor((256,))
            for i0_fused in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for k_0 in range(64):
                    for k_1 in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("T_softmax_maxelem"):
                            v_i0 = Ts.axis.spatial(256, i0_fused)
                            v_k = Ts.axis.reduce(256, k_0 * 4 + k_1)
                            Ts.reads(A[v_i0, v_k])
                            Ts.writes(T_softmax_maxelem[v_i0])
                            with Ts.init():
                                T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
            for i0_fused_0 in T.parallel(4, annotations={"thread_binding": "blockIdx.x"}):
                for i0_fused_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                    for k in range(256):
                        with Ts.sblock("T_softmax_expsum"):
                            v_i0 = Ts.axis.spatial(256, i0_fused_0 * 64 + i0_fused_1)
                            v_k = Ts.axis.reduce(256, k)
                            Ts.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                            Ts.writes(T_softmax_expsum[v_i0])
                            with Ts.init():
                                T_softmax_expsum[v_i0] = T.float32(0)
                            T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
            for i0_i1_fused_0 in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("T_softmax_norm"):
                        v_i0 = Ts.axis.spatial(256, (i0_i1_fused_0 * 256 + i0_i1_fused_1) // 256)
                        v_i1 = Ts.axis.spatial(256, (i0_i1_fused_0 * 256 + i0_i1_fused_1) % 256)
                        Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum[v_i0])
                        Ts.writes(T_softmax_norm[v_i0, v_i1])
                        Ts.sblock_attr({"axis": 1})
                        T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum[v_i0]
    @Ts.function
    def sfm_2(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 512})
            T_softmax_maxelem = Ts.sblock_alloc_tensor((256,))
            T_softmax_expsum_shared = Ts.sblock_alloc_tensor((256,), scope="shared")
            for i0_fused_0 in T.parallel(8, annotations={"thread_binding": "blockIdx.x"}):
                for i0_fused_1 in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                    for k in range(256):
                        with Ts.sblock("T_softmax_maxelem"):
                            v_i0 = Ts.axis.spatial(256, i0_fused_0 * 32 + i0_fused_1)
                            v_k = Ts.axis.reduce(256, k)
                            Ts.reads(A[v_i0, v_k])
                            Ts.writes(T_softmax_maxelem[v_i0])
                            with Ts.init():
                                T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
            for i0_fused in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for ax0, ax1_0 in T.grid(1, 1):
                    for ax1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("T_softmax_expsum"):
                            v_i0 = Ts.axis.spatial(256, i0_fused + ax0)
                            v_k = Ts.axis.reduce(256, ax1_0 * 512 + ax1_1)
                            Ts.where(ax1_0 * 512 + ax1_1 < 256)
                            Ts.reads(A[v_i0, v_k], T_softmax_maxelem[v_i0])
                            Ts.writes(T_softmax_expsum_shared[v_i0])
                            with Ts.init():
                                T_softmax_expsum_shared[v_i0] = T.float32(0)
                            T_softmax_expsum_shared[v_i0] = T_softmax_expsum_shared[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem[v_i0])
                for i1_0 in range(1):
                    for i1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("T_softmax_norm"):
                            v_i0 = Ts.axis.spatial(256, i0_fused)
                            v_i1 = Ts.axis.spatial(256, i1_0 * 512 + i1_1)
                            Ts.where(i1_0 * 512 + i1_1 < 256)
                            Ts.reads(A[v_i0, v_i1], T_softmax_maxelem[v_i0], T_softmax_expsum_shared[v_i0])
                            Ts.writes(T_softmax_norm[v_i0, v_i1])
                            Ts.sblock_attr({"axis": 1})
                            T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0]) / T_softmax_expsum_shared[v_i0]
    @Ts.function
    def sfm_3(A: T.Tensor((256, 256), "float32"), T_softmax_norm: T.Tensor((256, 256), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 0})
            T_softmax_maxelem_shared = Ts.sblock_alloc_tensor((256,), scope="shared")
            T_softmax_expsum_shared = Ts.sblock_alloc_tensor((256,), scope="shared")
            for i0_fused in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                for ax0, ax1_0 in T.grid(1, 1):
                    for ax1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("T_softmax_maxelem"):
                            v_i0 = Ts.axis.spatial(256, i0_fused + ax0)
                            v_k = Ts.axis.reduce(256, ax1_0 * 512 + ax1_1)
                            Ts.where(ax1_0 * 512 + ax1_1 < 256)
                            Ts.reads(A[v_i0, v_k])
                            Ts.writes(T_softmax_maxelem_shared[v_i0])
                            with Ts.init():
                                T_softmax_maxelem_shared[v_i0] = T.float32(-3.4028234663852886e+38)
                            T_softmax_maxelem_shared[v_i0] = T.max(T_softmax_maxelem_shared[v_i0], A[v_i0, v_k])
                for ax0, ax1_0 in T.grid(1, 1):
                    for ax1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("T_softmax_expsum"):
                            v_i0 = Ts.axis.spatial(256, i0_fused + ax0)
                            v_k = Ts.axis.reduce(256, ax1_0 * 512 + ax1_1)
                            Ts.where(ax1_0 * 512 + ax1_1 < 256)
                            Ts.reads(A[v_i0, v_k], T_softmax_maxelem_shared[v_i0])
                            Ts.writes(T_softmax_expsum_shared[v_i0])
                            with Ts.init():
                                T_softmax_expsum_shared[v_i0] = T.float32(0)
                            T_softmax_expsum_shared[v_i0] = T_softmax_expsum_shared[v_i0] + T.exp(A[v_i0, v_k] - T_softmax_maxelem_shared[v_i0])
                for i1_0 in range(1):
                    for i1_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("T_softmax_norm"):
                            v_i0 = Ts.axis.spatial(256, i0_fused)
                            v_i1 = Ts.axis.spatial(256, i1_0 * 512 + i1_1)
                            Ts.where(i1_0 * 512 + i1_1 < 256)
                            Ts.reads(A[v_i0, v_i1], T_softmax_maxelem_shared[v_i0], T_softmax_expsum_shared[v_i0])
                            Ts.writes(T_softmax_norm[v_i0, v_i1])
                            Ts.sblock_attr({"axis": 1})
                            T_softmax_norm[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem_shared[v_i0]) / T_softmax_expsum_shared[v_i0]
    # fmt: on
    decision_0 = [
        ("SampleCategorical", 0),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
    ]
    decision_1 = [
        ("SampleCategorical", 0),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
        ("SampleCategorical", 1),
    ]
    decision_2 = [
        ("SampleCategorical", 7),
        ("SampleCategorical", 6),
        ("SampleCategorical", 0),
    ]
    decision_3 = [
        ("SampleCategorical", 7),
        ("SampleCategorical", 0),
        ("SampleCategorical", 0),
    ]
    mod = create_te_workload("SFM", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[sfm_0, sfm_1, sfm_2, sfm_3],
        expected_decisions=[decision_0, decision_1, decision_2, decision_3],
    )


def test_cuda_cbr():
    # fmt: off
    @Ts.function
    def cbr_0(data: T.Tensor((1, 224, 224, 3), "float32"), kernel: T.Tensor((7, 7, 3, 64), "float32"), bias: T.Tensor(64, "float32"), bn_offset: T.Tensor(64, "float32"), bn_scale: T.Tensor(64, "float32"), compute: T.Tensor((1, 112, 112, 64), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 512})
            Conv2dOutput_local = Ts.sblock_alloc_tensor((1, 112, 112, 64), scope="local")
            PaddedInput_shared = Ts.sblock_alloc_tensor((1, 230, 230, 3), scope="shared")
            kernel_shared = Ts.sblock_alloc_tensor((7, 7, 3, 64), scope="shared")
            for nn_0_yy_0_xx_0_ff_0_fused in T.parallel(14, annotations={"thread_binding": "blockIdx.x"}):
                for nn_1_yy_1_xx_1_ff_1_fused in T.parallel(4, annotations={"thread_binding": "vthread.x"}):
                    for nn_2_yy_2_xx_2_ff_2_fused in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                        for ry_0, rx_0, rc_0 in T.grid(7, 1, 3):
                            for ax0_ax1_ax2_ax3_fused in range(8251):
                                with Ts.sblock("PaddedInput_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(230, ry_0 + ax0_ax1_ax2_ax3_fused // 37)
                                    v2 = Ts.axis.spatial(230, nn_0_yy_0_xx_0_ff_0_fused // 2 * 32 + ax0_ax1_ax2_ax3_fused % 37)
                                    v3 = Ts.axis.spatial(3, rc_0)
                                    Ts.reads(data[v0, v1 - 3, v2 - 3, v3])
                                    Ts.writes(PaddedInput_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                    PaddedInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, data[v0, v1 - 3, v2 - 3, v3], T.float32(0))
                            for ax0_ax1_ax2_ax3_fused in range(224):
                                with Ts.sblock("kernel_shared"):
                                    v0 = Ts.axis.spatial(7, ry_0)
                                    v1 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_fused // 32)
                                    v2 = Ts.axis.spatial(3, rc_0)
                                    v3 = Ts.axis.spatial(64, nn_0_yy_0_xx_0_ff_0_fused % 2 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    Ts.reads(kernel[v0, v1, v2, v3])
                                    Ts.writes(kernel_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                    kernel_shared[v0, v1, v2, v3] = kernel[v0, v1, v2, v3]
                            for ry_1, rx_1, rc_1, nn_3, yy_3, xx_3, ff_3, ry_2, rx_2, rc_2, nn_4, yy_4, xx_4, ff_4 in T.grid(1, 1, 1, 1, 1, 1, 2, 1, 7, 1, 1, 7, 1, 8):
                                with Ts.sblock("Conv2dOutput"):
                                    v_nn = Ts.axis.spatial(1, nn_3 + nn_4)
                                    v_yy = Ts.axis.spatial(112, nn_1_yy_1_xx_1_ff_1_fused // 2 * 56 + nn_2_yy_2_xx_2_ff_2_fused // 16 * 7 + yy_3 * 7 + yy_4)
                                    v_xx = Ts.axis.spatial(112, nn_0_yy_0_xx_0_ff_0_fused // 2 * 16 + nn_2_yy_2_xx_2_ff_2_fused % 16 + xx_3 + xx_4)
                                    v_ff = Ts.axis.spatial(64, nn_0_yy_0_xx_0_ff_0_fused % 2 * 32 + nn_1_yy_1_xx_1_ff_1_fused % 2 * 16 + ff_3 * 8 + ff_4)
                                    v_ry = Ts.axis.reduce(7, ry_0 + ry_1 + ry_2)
                                    v_rx = Ts.axis.reduce(7, rx_0 * 7 + rx_1 * 7 + rx_2)
                                    v_rc = Ts.axis.reduce(3, rc_0 + rc_1 + rc_2)
                                    Ts.reads(PaddedInput_shared[v_nn, v_yy * 2 + v_ry, v_xx * 2 + v_rx, v_rc], kernel_shared[v_ry, v_rx, v_rc, v_ff])
                                    Ts.writes(Conv2dOutput_local[v_nn, v_yy, v_xx, v_ff])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        Conv2dOutput_local[v_nn, v_yy, v_xx, v_ff] = T.float32(0)
                                    Conv2dOutput_local[v_nn, v_yy, v_xx, v_ff] = Conv2dOutput_local[v_nn, v_yy, v_xx, v_ff] + PaddedInput_shared[v_nn, v_yy * 2 + v_ry, v_xx * 2 + v_rx, v_rc] * kernel_shared[v_ry, v_rx, v_rc, v_ff]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 7, 1, 16):
                            with Ts.sblock("Conv2dOutput_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(112, nn_1_yy_1_xx_1_ff_1_fused // 2 * 56 + nn_2_yy_2_xx_2_ff_2_fused // 16 * 7 + ax1)
                                v2 = Ts.axis.spatial(112, nn_0_yy_0_xx_0_ff_0_fused // 2 * 16 + nn_2_yy_2_xx_2_ff_2_fused % 16 + ax2)
                                v3 = Ts.axis.spatial(64, nn_0_yy_0_xx_0_ff_0_fused % 2 * 32 + nn_1_yy_1_xx_1_ff_1_fused % 2 * 16 + ax3)
                                Ts.reads(Conv2dOutput_local[v0, v1, v2, v3], bias[v3], bn_scale[v3], bn_offset[v3])
                                Ts.writes(compute[v0, v1, v2, v3])
                                compute[v0, v1, v2, v3] = T.max((Conv2dOutput_local[v0, v1, v2, v3] + bias[v3]) * bn_scale[v3] + bn_offset[v3], T.float32(0))
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 2, 8, 1, 7]),
        ("SamplePerfectTile", [7, 1, 16, 1, 1]),
        ("SamplePerfectTile", [2, 2, 1, 2, 8]),
        ("SamplePerfectTile", [7, 1, 1]),
        ("SamplePerfectTile", [1, 1, 7]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SampleCategorical", 0),
        ("SampleCategorical", 0),
        ("SampleCategorical", 6),
    ]
    mod = create_te_workload("CBR", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cbr_0],
        expected_decisions=[decision_0],
    )


def test_cuda_tbg():
    # fmt: off
    @Ts.function
    def tbg_0(query: T.Tensor((1, 128, 12, 64), "float32"), value: T.Tensor((1, 128, 12, 64), "float32"), C: T.Tensor((1, 12, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 1024})
            C_local = Ts.sblock_alloc_tensor((1, 12, 128, 128), scope="local")
            query_T_shared = Ts.sblock_alloc_tensor((1, 12, 128, 64), scope="shared")
            value_T_shared = Ts.sblock_alloc_tensor((1, 12, 64, 128), scope="shared")
            for b_0_h_0_i_0_j_0_fused in T.parallel(4, annotations={"thread_binding": "blockIdx.x"}):
                for b_1_h_1_i_1_j_1_fused in T.parallel(192, annotations={"thread_binding": "vthread.x"}):
                    for b_2_h_2_i_2_j_2_fused in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                        for k_0 in range(8):
                            for ax0_ax1_ax2_ax3_fused in range(12288):
                                with Ts.sblock("query_T_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(12, ax0_ax1_ax2_ax3_fused // 1024)
                                    v2 = Ts.axis.spatial(128, ax0_ax1_ax2_ax3_fused % 1024 // 8)
                                    v3 = Ts.axis.spatial(64, k_0 * 8 + ax0_ax1_ax2_ax3_fused % 8)
                                    Ts.reads(query[v0, v2, v1, v3])
                                    Ts.writes(query_T_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 3})
                                    query_T_shared[v0, v1, v2, v3] = query[v0, v2, v1, v3]
                            for ax0_ax1_ax2_ax3_fused in range(3072):
                                with Ts.sblock("value_T_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(12, ax0_ax1_ax2_ax3_fused // 256)
                                    v2 = Ts.axis.spatial(64, k_0 * 8 + ax0_ax1_ax2_ax3_fused % 256 // 32)
                                    v3 = Ts.axis.spatial(128, b_0_h_0_i_0_j_0_fused * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    Ts.reads(value[v0, v3, v1, v2])
                                    Ts.writes(value_T_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                    value_T_shared[v0, v1, v2, v3] = value[v0, v3, v1, v2]
                            for k_1, b_3, h_3, i_3, j_3, k_2, b_4, h_4, i_4, j_4 in T.grid(4, 1, 2, 1, 1, 2, 1, 1, 4, 1):
                                with Ts.sblock("C"):
                                    v_b = Ts.axis.spatial(1, b_3 + b_4)
                                    v_h = Ts.axis.spatial(12, b_1_h_1_i_1_j_1_fused // 32 * 2 + h_3 + h_4)
                                    v_i = Ts.axis.spatial(128, b_1_h_1_i_1_j_1_fused % 32 // 8 * 32 + b_2_h_2_i_2_j_2_fused // 4 * 4 + i_3 * 4 + i_4)
                                    v_j = Ts.axis.spatial(128, b_0_h_0_i_0_j_0_fused * 32 + b_1_h_1_i_1_j_1_fused % 8 * 4 + b_2_h_2_i_2_j_2_fused % 4 + j_3 + j_4)
                                    v_k = Ts.axis.reduce(64, k_0 * 8 + k_1 * 2 + k_2)
                                    Ts.reads(query_T_shared[v_b, v_h, v_i, v_k], value_T_shared[v_b, v_h, v_k, v_j])
                                    Ts.writes(C_local[v_b, v_h, v_i, v_j])
                                    Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with Ts.init():
                                        C_local[v_b, v_h, v_i, v_j] = T.float32(0)
                                    C_local[v_b, v_h, v_i, v_j] = C_local[v_b, v_h, v_i, v_j] + query_T_shared[v_b, v_h, v_i, v_k] * value_T_shared[v_b, v_h, v_k, v_j]
                        for ax0, ax1, ax2, ax3 in T.grid(1, 2, 4, 1):
                            with Ts.sblock("C_local"):
                                v0 = Ts.axis.spatial(1, ax0)
                                v1 = Ts.axis.spatial(12, b_1_h_1_i_1_j_1_fused // 32 * 2 + ax1)
                                v2 = Ts.axis.spatial(128, b_1_h_1_i_1_j_1_fused % 32 // 8 * 32 + b_2_h_2_i_2_j_2_fused // 4 * 4 + ax2)
                                v3 = Ts.axis.spatial(128, b_0_h_0_i_0_j_0_fused * 32 + b_1_h_1_i_1_j_1_fused % 8 * 4 + b_2_h_2_i_2_j_2_fused % 4 + ax3)
                                Ts.reads(C_local[v0, v1, v2, v3])
                                Ts.writes(C[v0, v1, v2, v3])
                                C[v0, v1, v2, v3] = C_local[v0, v1, v2, v3]
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 6, 1, 2, 1]),
        ("SamplePerfectTile", [1, 4, 8, 1, 4]),
        ("SamplePerfectTile", [4, 8, 4, 1, 1]),
        ("SamplePerfectTile", [8, 4, 2]),
        ("SampleCategorical", 2),
        ("SampleCategorical", 3),
        ("SampleCategorical", 7),
    ]
    mod = create_te_workload("TBG", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[tbg_0],
        expected_decisions=[decision_0],
    )


if __name__ == "__main__":
    test_cuda_c1d()
    test_cuda_c2d()
    test_cuda_c3d()
    test_cuda_cap()
    test_cuda_dep()
    test_cuda_dil()
    test_cuda_gmm()
    test_cuda_grp()
    test_cuda_t2d()
    test_cuda_nrm()
    test_cuda_sfm()
    test_cuda_cbr()
    test_cuda_tbg()
