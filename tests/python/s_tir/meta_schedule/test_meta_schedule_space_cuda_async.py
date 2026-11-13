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
    return Target("nvidia/geforce-rtx-3070")


def _design_space(mod):
    return generate_design_space(
        kind="cuda",
        mod=mod,
        target=_target(),
        types=ms.ScheduleRule,
    )


def get_c2d_function(stage: int):
    if stage == 0:
        # fmt: off
        @Ts.function
        def c2d(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                Ts.sblock_attr({"meta_schedule.unroll_explicit": 1024})
                conv2d_nhwc_local = Ts.sblock_alloc_tensor((1, 112, 112, 64), scope="local")
                PadInput_shared = Ts.sblock_alloc_tensor((1, 230, 230, 3), scope="shared")
                weight_shared = Ts.sblock_alloc_tensor((7, 7, 3, 64), scope="shared")
                for n_0_h_0_w_0_co_0_fused in T.parallel(112, annotations={"thread_binding": "blockIdx.x"}):
                    for n_1_h_1_w_1_co_1_fused in T.parallel(8, annotations={"thread_binding": "vthread.x"}):
                        for n_2_h_2_w_2_co_2_fused in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                            for rh_0, rw_0, rc_0 in T.grid(1, 1, 3):
                                for ax0_ax1_ax2_ax3_fused in range(693):
                                    with Ts.sblock("PadInput_shared"):
                                        v0 = Ts.axis.spatial(1, 0)
                                        v1 = Ts.axis.spatial(230, n_0_h_0_w_0_co_0_fused // 8 * 16 + ax0_ax1_ax2_ax3_fused // 33)
                                        v2 = Ts.axis.spatial(230, n_0_h_0_w_0_co_0_fused % 8 * 28 + ax0_ax1_ax2_ax3_fused % 33)
                                        v3 = Ts.axis.spatial(3, rc_0)
                                        Ts.reads(inputs[v0, v1 - 3, v2 - 3, v3])
                                        Ts.writes(PadInput_shared[v0, v1, v2, v3])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                        PadInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, inputs[v0, v1 - 3, v2 - 3, v3], T.float32(0))
                                for ax0_ax1_ax2_ax3_fused in range(3136):
                                    with Ts.sblock("weight_shared"):
                                        v0 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_fused // 448)
                                        v1 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_fused % 448 // 64)
                                        v2 = Ts.axis.spatial(3, rc_0)
                                        v3 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 64)
                                        Ts.reads(weight[v0, v1, v2, v3])
                                        Ts.writes(weight_shared[v0, v1, v2, v3])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch": 3})
                                        weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                                for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(7, 1, 1, 1, 1, 14, 1, 1, 7, 1, 1, 1, 1, 1):
                                    with Ts.sblock("conv2d_nhwc"):
                                        v_n = Ts.axis.spatial(1, n_3 + n_4)
                                        v_h = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused // 8 * 8 + n_1_h_1_w_1_co_1_fused // 4 * 4 + n_2_h_2_w_2_co_2_fused // 16 + h_3 + h_4)
                                        v_w = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused % 8 * 14 + w_3 + w_4)
                                        v_co = Ts.axis.spatial(64, n_1_h_1_w_1_co_1_fused % 4 * 16 + n_2_h_2_w_2_co_2_fused % 16 + co_3 + co_4)
                                        v_rh = Ts.axis.reduce(7, rh_0 * 7 + rh_1 + rh_2)
                                        v_rw = Ts.axis.reduce(7, rw_0 * 7 + rw_1 * 7 + rw_2)
                                        v_rc = Ts.axis.reduce(3, rc_0 + rc_1 + rc_2)
                                        Ts.reads(PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight_shared[v_rh, v_rw, v_rc, v_co])
                                        Ts.writes(conv2d_nhwc_local[v_n, v_h, v_w, v_co])
                                        Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                        with Ts.init():
                                            conv2d_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                        conv2d_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_nhwc_local[v_n, v_h, v_w, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight_shared[v_rh, v_rw, v_rc, v_co]
                            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 14, 1):
                                with Ts.sblock("conv2d_nhwc_local"):
                                    v0 = Ts.axis.spatial(1, ax0)
                                    v1 = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused // 8 * 8 + n_1_h_1_w_1_co_1_fused // 4 * 4 + n_2_h_2_w_2_co_2_fused // 16 + ax1)
                                    v2 = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused % 8 * 14 + ax2)
                                    v3 = Ts.axis.spatial(64, n_1_h_1_w_1_co_1_fused % 4 * 16 + n_2_h_2_w_2_co_2_fused % 16 + ax3)
                                    Ts.reads(conv2d_nhwc_local[v0, v1, v2, v3])
                                    Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                                    conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_local[v0, v1, v2, v3]
        # fmt: on
    else:
        # fmt: off
        @Ts.function
        def c2d(inputs: T.Tensor((1, 224, 224, 3), "float32"), weight: T.Tensor((7, 7, 3, 64), "float32"), conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                Ts.sblock_attr({"meta_schedule.unroll_explicit": 1024})
                conv2d_nhwc_local = Ts.sblock_alloc_tensor((1, 112, 112, 64), scope="local")
                PadInput_shared = Ts.sblock_alloc_tensor((1, 230, 230, 3), scope="shared")
                weight_shared = Ts.sblock_alloc_tensor((7, 7, 3, 64), scope="shared")
                for n_0_h_0_w_0_co_0_fused in T.parallel(112, annotations={"thread_binding": "blockIdx.x"}):
                    for n_1_h_1_w_1_co_1_fused in T.parallel(8, annotations={"thread_binding": "vthread.x"}):
                        for n_2_h_2_w_2_co_2_fused in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                            for rh_0_rw_0_rc_0_fused in T.serial(3, annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, stage - 2]}):
                                for ax0_ax1_ax2_ax3_fused in range(693):
                                    with Ts.sblock("PadInput_shared"):
                                        v0 = Ts.axis.spatial(1, 0)
                                        v1 = Ts.axis.spatial(230, n_0_h_0_w_0_co_0_fused // 8 * 16 + ax0_ax1_ax2_ax3_fused // 33)
                                        v2 = Ts.axis.spatial(230, n_0_h_0_w_0_co_0_fused % 8 * 28 + ax0_ax1_ax2_ax3_fused % 33)
                                        v3 = Ts.axis.spatial(3, rh_0_rw_0_rc_0_fused)
                                        Ts.reads(inputs[v0, v1 - 3, v2 - 3, v3])
                                        Ts.writes(PadInput_shared[v0, v1, v2, v3])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                        PadInput_shared[v0, v1, v2, v3] = T.if_then_else(3 <= v1 and v1 < 227 and 3 <= v2 and v2 < 227, inputs[v0, v1 - 3, v2 - 3, v3], T.float32(0))
                                for ax0_ax1_ax2_ax3_fused in range(3136):
                                    with Ts.sblock("weight_shared"):
                                        v0 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_fused // 448)
                                        v1 = Ts.axis.spatial(7, ax0_ax1_ax2_ax3_fused % 448 // 64)
                                        v2 = Ts.axis.spatial(3, rh_0_rw_0_rc_0_fused)
                                        v3 = Ts.axis.spatial(64, ax0_ax1_ax2_ax3_fused % 64)
                                        Ts.reads(weight[v0, v1, v2, v3])
                                        Ts.writes(weight_shared[v0, v1, v2, v3])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch": 3})
                                        weight_shared[v0, v1, v2, v3] = weight[v0, v1, v2, v3]
                                for rh_1, rw_1, rc_1, n_3, h_3, w_3, co_3, rh_2, rw_2, rc_2, n_4, h_4, w_4, co_4 in T.grid(7, 1, 1, 1, 1, 14, 1, 1, 7, 1, 1, 1, 1, 1):
                                    with Ts.sblock("conv2d_nhwc"):
                                        v_n = Ts.axis.spatial(1, n_3 + n_4)
                                        v_h = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused // 8 * 8 + n_1_h_1_w_1_co_1_fused // 4 * 4 + n_2_h_2_w_2_co_2_fused // 16 + h_3 + h_4)
                                        v_w = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused % 8 * 14 + w_3 + w_4)
                                        v_co = Ts.axis.spatial(64, n_1_h_1_w_1_co_1_fused % 4 * 16 + n_2_h_2_w_2_co_2_fused % 16 + co_3 + co_4)
                                        v_rh = Ts.axis.reduce(7, rh_1 + rh_2)
                                        v_rw = Ts.axis.reduce(7, rw_1 * 7 + rw_2)
                                        v_rc = Ts.axis.reduce(3, rh_0_rw_0_rc_0_fused + rc_1 + rc_2)
                                        Ts.reads(PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc], weight_shared[v_rh, v_rw, v_rc, v_co])
                                        Ts.writes(conv2d_nhwc_local[v_n, v_h, v_w, v_co])
                                        Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                        with Ts.init():
                                            conv2d_nhwc_local[v_n, v_h, v_w, v_co] = T.float32(0)
                                        conv2d_nhwc_local[v_n, v_h, v_w, v_co] = conv2d_nhwc_local[v_n, v_h, v_w, v_co] + PadInput_shared[v_n, v_h * 2 + v_rh, v_w * 2 + v_rw, v_co // 64 * 3 + v_rc] * weight_shared[v_rh, v_rw, v_rc, v_co]
                            for ax0, ax1, ax2, ax3 in T.grid(1, 1, 14, 1):
                                with Ts.sblock("conv2d_nhwc_local"):
                                    v0 = Ts.axis.spatial(1, ax0)
                                    v1 = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused // 8 * 8 + n_1_h_1_w_1_co_1_fused // 4 * 4 + n_2_h_2_w_2_co_2_fused // 16 + ax1)
                                    v2 = Ts.axis.spatial(112, n_0_h_0_w_0_co_0_fused % 8 * 14 + ax2)
                                    v3 = Ts.axis.spatial(64, n_1_h_1_w_1_co_1_fused % 4 * 16 + n_2_h_2_w_2_co_2_fused % 16 + ax3)
                                    Ts.reads(conv2d_nhwc_local[v0, v1, v2, v3])
                                    Ts.writes(conv2d_nhwc[v0, v1, v2, v3])
                                    conv2d_nhwc[v0, v1, v2, v3] = conv2d_nhwc_local[v0, v1, v2, v3]
        # fmt: on
    return c2d


def test_cuda_c2d():
    c2d_decision = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [14, 2, 4, 1, 1]),
        ("SamplePerfectTile", [8, 1, 1, 14, 1]),
        ("SamplePerfectTile", [1, 4, 16, 1, 1]),
        ("SamplePerfectTile", [1, 7, 1]),
        ("SamplePerfectTile", [1, 1, 7]),
        ("SamplePerfectTile", [3, 1, 1]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 2),
        ("SampleCategorical", 7),
    ]

    mod = create_te_workload("C2D", 0)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[
            get_c2d_function(stage=0),
            get_c2d_function(stage=4),
            get_c2d_function(stage=5),
        ],
        expected_decisions=[c2d_decision, c2d_decision, c2d_decision],
    )


def get_gmm_function(stage: int):
    if stage == 0:
        # fmt: off
        @Ts.function
        def gmm(X: T.Tensor((1, 1024, 1024), "float32"), Y: T.Tensor((1, 1024, 1024), "float32"), Z: T.Tensor((1, 1024, 1024), "float32")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                Ts.sblock_attr({"meta_schedule.unroll_explicit": 16})
                Z_local = Ts.sblock_alloc_tensor((1, 1024, 1024), scope="local")
                X_shared = Ts.sblock_alloc_tensor((1, 1024, 1024), scope="shared")
                Y_shared = Ts.sblock_alloc_tensor((1, 1024, 1024), scope="shared")
                for b_0_i_0_j_0_fused in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                    for b_1_i_1_j_1_fused in T.parallel(32, annotations={"thread_binding": "vthread.x"}):
                        for b_2_i_2_j_2_fused in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                            for k_0 in range(64):
                                for ax0_ax1_ax2_fused in range(1024):
                                    with Ts.sblock("X_shared"):
                                        v0 = Ts.axis.spatial(1, 0)
                                        v1 = Ts.axis.spatial(1024, b_0_i_0_j_0_fused // 16 * 64 + ax0_ax1_ax2_fused // 16)
                                        v2 = Ts.axis.spatial(1024, k_0 * 16 + ax0_ax1_ax2_fused % 16)
                                        Ts.reads(X[v0, v1, v2])
                                        Ts.writes(X_shared[v0, v1, v2])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                        X_shared[v0, v1, v2] = X[v0, v1, v2]
                                for ax0_ax1_ax2_fused in range(1024):
                                    with Ts.sblock("Y_shared"):
                                        v0 = Ts.axis.spatial(1, 0)
                                        v1 = Ts.axis.spatial(1024, k_0 * 16 + ax0_ax1_ax2_fused // 64)
                                        v2 = Ts.axis.spatial(1024, b_0_i_0_j_0_fused % 16 * 64 + ax0_ax1_ax2_fused % 64)
                                        Ts.reads(Y[v0, v1, v2])
                                        Ts.writes(Y_shared[v0, v1, v2])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                        Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                                for k_1, b_3, i_3, j_3, k_2, b_4, i_4, j_4 in T.grid(2, 1, 1, 1, 8, 1, 1, 2):
                                    with Ts.sblock("Z"):
                                        v_b = Ts.axis.spatial(1, b_3 + b_4)
                                        v_i = Ts.axis.spatial(1024, b_0_i_0_j_0_fused // 16 * 64 + b_1_i_1_j_1_fused // 4 * 8 + b_2_i_2_j_2_fused // 8 + i_3 + i_4)
                                        v_j = Ts.axis.spatial(1024, b_0_i_0_j_0_fused % 16 * 64 + b_1_i_1_j_1_fused % 4 * 16 + b_2_i_2_j_2_fused % 8 * 2 + j_3 * 2 + j_4)
                                        v_k = Ts.axis.reduce(1024, k_0 * 16 + k_1 * 8 + k_2)
                                        Ts.reads(X_shared[v_b, v_i, v_k], Y_shared[v_b, v_k, v_j])
                                        Ts.writes(Z_local[v_b, v_i, v_j])
                                        Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                        with Ts.init():
                                            Z_local[v_b, v_i, v_j] = T.float32(0)
                                        Z_local[v_b, v_i, v_j] = Z_local[v_b, v_i, v_j] + X_shared[v_b, v_i, v_k] * Y_shared[v_b, v_k, v_j]
                            for ax0, ax1, ax2 in T.grid(1, 1, 2):
                                with Ts.sblock("Z_local"):
                                    v0 = Ts.axis.spatial(1, ax0)
                                    v1 = Ts.axis.spatial(1024, b_0_i_0_j_0_fused // 16 * 64 + b_1_i_1_j_1_fused // 4 * 8 + b_2_i_2_j_2_fused // 8 + ax1)
                                    v2 = Ts.axis.spatial(1024, b_0_i_0_j_0_fused % 16 * 64 + b_1_i_1_j_1_fused % 4 * 16 + b_2_i_2_j_2_fused % 8 * 2 + ax2)
                                    Ts.reads(Z_local[v0, v1, v2])
                                    Ts.writes(Z[v0, v1, v2])
                                    Z[v0, v1, v2] = Z_local[v0, v1, v2]
        # fmt: on
    else:
        # fmt: off
        @Ts.function
        def gmm(X: T.Tensor((1, 1024, 1024), "float32"), Y: T.Tensor((1, 1024, 1024), "float32"), Z: T.Tensor((1, 1024, 1024), "float32")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                Ts.sblock_attr({"meta_schedule.unroll_explicit": 16})
                Z_local = Ts.sblock_alloc_tensor((1, 1024, 1024), scope="local")
                X_shared = Ts.sblock_alloc_tensor((1, 1024, 1024), scope="shared")
                Y_shared = Ts.sblock_alloc_tensor((1, 1024, 1024), scope="shared")
                for b_0_i_0_j_0_fused in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
                    for b_1_i_1_j_1_fused in T.parallel(32, annotations={"thread_binding": "vthread.x"}):
                        for b_2_i_2_j_2_fused in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                            for k_0_fused in T.serial(64, annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, stage - 2]}):
                                for ax0_ax1_ax2_fused in range(1024):
                                    with Ts.sblock("X_shared"):
                                        v0 = Ts.axis.spatial(1, 0)
                                        v1 = Ts.axis.spatial(1024, b_0_i_0_j_0_fused // 16 * 64 + ax0_ax1_ax2_fused // 16)
                                        v2 = Ts.axis.spatial(1024, k_0_fused * 16 + ax0_ax1_ax2_fused % 16)
                                        Ts.reads(X[v0, v1, v2])
                                        Ts.writes(X_shared[v0, v1, v2])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                        X_shared[v0, v1, v2] = X[v0, v1, v2]
                                for ax0_ax1_ax2_fused in range(1024):
                                    with Ts.sblock("Y_shared"):
                                        v0 = Ts.axis.spatial(1, 0)
                                        v1 = Ts.axis.spatial(1024, k_0_fused * 16 + ax0_ax1_ax2_fused // 64)
                                        v2 = Ts.axis.spatial(1024, b_0_i_0_j_0_fused % 16 * 64 + ax0_ax1_ax2_fused % 64)
                                        Ts.reads(Y[v0, v1, v2])
                                        Ts.writes(Y_shared[v0, v1, v2])
                                        Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                        Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                                for k_1, b_3, i_3, j_3, k_2, b_4, i_4, j_4 in T.grid(2, 1, 1, 1, 8, 1, 1, 2):
                                    with Ts.sblock("Z"):
                                        v_b = Ts.axis.spatial(1, b_3 + b_4)
                                        v_i = Ts.axis.spatial(1024, b_0_i_0_j_0_fused // 16 * 64 + b_1_i_1_j_1_fused // 4 * 8 + b_2_i_2_j_2_fused // 8 + i_3 + i_4)
                                        v_j = Ts.axis.spatial(1024, b_0_i_0_j_0_fused % 16 * 64 + b_1_i_1_j_1_fused % 4 * 16 + b_2_i_2_j_2_fused % 8 * 2 + j_3 * 2 + j_4)
                                        v_k = Ts.axis.reduce(1024, k_0_fused * 16 + k_1 * 8 + k_2)
                                        Ts.reads(X_shared[v_b, v_i, v_k], Y_shared[v_b, v_k, v_j])
                                        Ts.writes(Z_local[v_b, v_i, v_j])
                                        Ts.sblock_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                        with Ts.init():
                                            Z_local[v_b, v_i, v_j] = T.float32(0)
                                        Z_local[v_b, v_i, v_j] = Z_local[v_b, v_i, v_j] + X_shared[v_b, v_i, v_k] * Y_shared[v_b, v_k, v_j]
                            for ax0, ax1, ax2 in T.grid(1, 1, 2):
                                with Ts.sblock("Z_local"):
                                    v0 = Ts.axis.spatial(1, ax0)
                                    v1 = Ts.axis.spatial(1024, b_0_i_0_j_0_fused // 16 * 64 + b_1_i_1_j_1_fused // 4 * 8 + b_2_i_2_j_2_fused // 8 + ax1)
                                    v2 = Ts.axis.spatial(1024, b_0_i_0_j_0_fused % 16 * 64 + b_1_i_1_j_1_fused % 4 * 16 + b_2_i_2_j_2_fused % 8 * 2 + ax2)
                                    Ts.reads(Z_local[v0, v1, v2])
                                    Ts.writes(Z[v0, v1, v2])
                                    Z[v0, v1, v2] = Z_local[v0, v1, v2]
        # fmt: on
    return gmm


def test_cuda_gmm():
    gmm_decision = [
        ("SamplePerfectTile", [1, 1, 1, 1, 1]),
        ("SamplePerfectTile", [16, 8, 8, 1, 1]),
        ("SamplePerfectTile", [16, 4, 8, 1, 2]),
        ("SamplePerfectTile", [64, 2, 8]),
        ("SampleCategorical", 3),
        ("SampleCategorical", 3),
        ("SampleCategorical", 1),
    ]

    mod = create_te_workload("GMM", 3)
    actual = _design_space(mod)
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[
            get_gmm_function(stage=0),
            get_gmm_function(stage=4),
            get_gmm_function(stage=5),
        ],
        expected_decisions=[gmm_decision, gmm_decision, gmm_decision],
    )


if __name__ == "__main__":
    test_cuda_c2d()
    test_cuda_gmm()
