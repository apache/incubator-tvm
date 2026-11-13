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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# ruff: noqa: E741, F401
import tvm
import tvm_tirx.testing
from tvm import te
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.testing import te_workload
from tvm.s_tir.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    print_sketches,
)
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def test_cpu_matmul():
    @Ts.function
    def cpu_matmul_0(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        C: T.Tensor((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C_global = Ts.sblock_alloc_tensor([512, 512], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1 in T.grid(1, 8, 8, 1):
            for i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(16, 2, 8, 32, 32, 8):
                with Ts.sblock("C"):
                    i = Ts.axis.spatial(512, i0_0 * 512 + i0_1 * 64 + i0_2 * 32 + i0_3)
                    j = Ts.axis.spatial(512, i1_0 * 64 + i1_1 * 64 + i1_2 * 8 + i1_3)
                    k = Ts.axis.reduce(512, i2_0 * 32 + i2_1)
                    Ts.reads(A[i, k], B[k, j])
                    Ts.writes(C_global[i, j])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        C_global[i, j] = T.float32(0)
                    C_global[i, j] = C_global[i, j] + A[i, k] * B[k, j]
            for ax0, ax1 in T.grid(64, 64):
                with Ts.sblock("C_global"):
                    v0 = Ts.axis.spatial(512, i0_1 * 64 + ax0)
                    v1 = Ts.axis.spatial(512, i1_0 * 64 + ax1)
                    Ts.reads(C_global[v0, v1])
                    Ts.writes(C[v0, v1])
                    C[v0, v1] = C_global[v0, v1]

    @Ts.function
    def cpu_matmul_1(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        C: T.Tensor((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C_global = Ts.sblock_alloc_tensor([512, 512], dtype="float32")
        for i0_0, i1_0 in T.grid(1, 8):
            for i0_1, i1_1, i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(8, 1, 16, 2, 8, 32, 32, 8):
                with Ts.sblock("C"):
                    i = Ts.axis.spatial(512, i0_0 * 512 + i0_1 * 64 + i0_2 * 32 + i0_3)
                    j = Ts.axis.spatial(512, i1_0 * 64 + i1_1 * 64 + i1_2 * 8 + i1_3)
                    k = Ts.axis.reduce(512, i2_0 * 32 + i2_1)
                    Ts.reads(A[i, k], B[k, j])
                    Ts.writes(C_global[i, j])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        C_global[i, j] = T.float32(0)
                    C_global[i, j] = C_global[i, j] + A[i, k] * B[k, j]
            for ax0, ax1 in T.grid(512, 64):
                with Ts.sblock("C_global"):
                    v0 = Ts.axis.spatial(512, ax0)
                    v1 = Ts.axis.spatial(512, i1_0 * 64 + ax1)
                    Ts.reads(C_global[v0, v1])
                    Ts.writes(C[v0, v1])
                    C[v0, v1] = C_global[v0, v1]

    @Ts.function
    def cpu_matmul_2(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        C: T.Tensor((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for i0_0, i1_0, i0_1, i1_1, i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(
            1, 8, 8, 1, 16, 2, 8, 32, 32, 8
        ):
            with Ts.sblock("C"):
                i = Ts.axis.spatial(512, i0_0 * 512 + i0_1 * 64 + i0_2 * 32 + i0_3)
                j = Ts.axis.spatial(512, i1_0 * 64 + i1_1 * 64 + i1_2 * 8 + i1_3)
                k = Ts.axis.reduce(512, i2_0 * 32 + i2_1)
                Ts.reads(A[i, k], B[k, j])
                Ts.writes(C[i, j])
                Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                with Ts.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[k, j]

    decision_0 = [
        ("SamplePerfectTile", [1, 8, 2, 32]),
        ("SamplePerfectTile", [8, 1, 8, 8]),
        ("SamplePerfectTile", [16, 32]),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 8, 2, 32]),
        ("SamplePerfectTile", [8, 1, 8, 8]),
        ("SamplePerfectTile", [16, 32]),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 8, 2, 32]),
        ("SamplePerfectTile", [8, 1, 8, 8]),
        ("SamplePerfectTile", [16, 32]),
    ]

    mod = te.create_function(te_workload.matmul(512, 512, 512))
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target("llvm"),
        types=ms.schedule_rule.MultiLevelTiling,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cpu_matmul_0, cpu_matmul_1, cpu_matmul_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cpu_matmul_relu():
    @Ts.function
    def cpu_matmul_relu_0(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        compute: T.Tensor((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C = Ts.sblock_alloc_tensor([512, 512], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1, i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(
            256, 4, 1, 4, 64, 1, 32, 8, 2, 1
        ):
            with Ts.sblock("C"):
                i = Ts.axis.spatial(512, i0_0 * 2 + i0_1 * 2 + i0_2 * 2 + i0_3)
                j = Ts.axis.spatial(512, i1_0 * 128 + i1_1 * 32 + i1_2 + i1_3)
                k = Ts.axis.reduce(512, i2_0 * 8 + i2_1)
                Ts.reads(A[i, k], B[k, j])
                Ts.writes(C[i, j])
                Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                with Ts.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
        for i0, i1 in T.grid(512, 512):
            with Ts.sblock("compute"):
                i0_4, i1_4 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(C[i0_4, i1_4])
                Ts.writes(compute[i0_4, i1_4])
                compute[i0_4, i1_4] = T.max(C[i0_4, i1_4], T.float32(0))

    @Ts.function
    def cpu_matmul_relu_1(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        compute: T.Tensor((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C = Ts.sblock_alloc_tensor([512, 512], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1 in T.grid(256, 4, 1, 4):
            for i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(64, 1, 32, 8, 2, 1):
                with Ts.sblock("C"):
                    i = Ts.axis.spatial(512, i0_0 * 2 + i0_1 * 2 + i0_2 * 2 + i0_3)
                    j = Ts.axis.spatial(512, i1_0 * 128 + i1_1 * 32 + i1_2 + i1_3)
                    k = Ts.axis.reduce(512, i2_0 * 8 + i2_1)
                    Ts.reads(A[i, k], B[k, j])
                    Ts.writes(C[i, j])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]
            for ax0, ax1 in T.grid(2, 32):
                with Ts.sblock("compute"):
                    i0 = Ts.axis.spatial(512, i0_0 * 2 + ax0)
                    i1 = Ts.axis.spatial(512, i1_0 * 128 + i1_1 * 32 + ax1)
                    Ts.reads(C[i0, i1])
                    Ts.writes(compute[i0, i1])
                    compute[i0, i1] = T.max(C[i0, i1], T.float32(0))

    @Ts.function
    def cpu_matmul_relu_2(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        compute: T.Tensor((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C = Ts.sblock_alloc_tensor([512, 512], dtype="float32")
        for i0_0, i1_0 in T.grid(256, 4):
            for i0_1, i1_1, i2_0, i0_2, i1_2, i2_1, i0_3, i1_3 in T.grid(1, 4, 64, 1, 32, 8, 2, 1):
                with Ts.sblock("C"):
                    i = Ts.axis.spatial(512, i0_0 * 2 + i0_1 * 2 + i0_2 * 2 + i0_3)
                    j = Ts.axis.spatial(512, i1_0 * 128 + i1_1 * 32 + i1_2 + i1_3)
                    k = Ts.axis.reduce(512, i2_0 * 8 + i2_1)
                    Ts.reads(A[i, k], B[k, j])
                    Ts.writes(C[i, j])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                    with Ts.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]
            for ax0, ax1 in T.grid(2, 128):
                with Ts.sblock("compute"):
                    i0 = Ts.axis.spatial(512, i0_0 * 2 + ax0)
                    i1 = Ts.axis.spatial(512, i1_0 * 128 + ax1)
                    Ts.reads(C[i0, i1])
                    Ts.writes(compute[i0, i1])
                    compute[i0, i1] = T.max(C[i0, i1], T.float32(0))

    decision_0 = [
        ("SamplePerfectTile", [256, 1, 1, 2]),
        ("SamplePerfectTile", [4, 4, 32, 1]),
        ("SamplePerfectTile", [64, 8]),
    ]
    decision_1 = [
        ("SamplePerfectTile", [256, 1, 1, 2]),
        ("SamplePerfectTile", [4, 4, 32, 1]),
        ("SamplePerfectTile", [64, 8]),
    ]
    decision_2 = [
        ("SamplePerfectTile", [256, 1, 1, 2]),
        ("SamplePerfectTile", [4, 4, 32, 1]),
        ("SamplePerfectTile", [64, 8]),
    ]
    mod = te.create_function(te_workload.matmul_relu(512, 512, 512))
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target("llvm"),
        types=ms.schedule_rule.MultiLevelTiling,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cpu_matmul_relu_0, cpu_matmul_relu_1, cpu_matmul_relu_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def test_cuda_matmul():
    @Ts.function
    def cuda_matmul_0(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        C: T.Tensor((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C_local = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="local")
        A_shared = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="shared")
        B_shared = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.parallel(128, annotations={"thread_binding": "blockIdx.x"}):
            for i0_1_i1_1_fused in T.parallel(8, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_fused in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
                    for i2_0 in T.serial(128):
                        for ax0_ax1_fused in T.serial(256):
                            with Ts.sblock("A_shared"):
                                v0 = Ts.axis.spatial(
                                    512, i0_0_i1_0_fused // 16 * 64 + ax0_ax1_fused // 4
                                )
                                v1 = Ts.axis.spatial(512, i2_0 * 4 + ax0_ax1_fused % 4)
                                Ts.reads(A[v0, v1])
                                Ts.writes(A_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in T.serial(128):
                            with Ts.sblock("B_shared"):
                                v0 = Ts.axis.spatial(512, i2_0 * 4 + ax0_ax1_fused // 32)
                                v1 = Ts.axis.spatial(
                                    512, i0_0_i1_0_fused % 16 * 32 + ax0_ax1_fused % 32
                                )
                                Ts.reads(B[v0, v1])
                                Ts.writes(B_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(2, 1, 1, 2, 16, 4):
                            with Ts.sblock("C"):
                                i = Ts.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused // 16 * 64
                                    + i0_1_i1_1_fused // 2 * 16
                                    + i0_3 * 16
                                    + i0_4,
                                )
                                j = Ts.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused % 16 * 32
                                    + i0_1_i1_1_fused % 2 * 16
                                    + i0_2_i1_2_fused * 4
                                    + i1_3 * 4
                                    + i1_4,
                                )
                                k = Ts.axis.reduce(512, i2_0 * 4 + i2_1 * 2 + i2_2)
                                Ts.reads(A_shared[i, k], B_shared[k, j])
                                Ts.writes(C_local[i, j])
                                Ts.sblock_attr(
                                    {
                                        "meta_schedule.thread_extent_high_inclusive": 1024,
                                        "meta_schedule.thread_extent_low_inclusive": 32,
                                        "meta_schedule.tiling_structure": "SSSRRSRS",
                                    }
                                )
                                with Ts.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(16, 4):
                        with Ts.sblock("C_local"):
                            v0 = Ts.axis.spatial(
                                512, i0_0_i1_0_fused // 16 * 64 + i0_1_i1_1_fused // 2 * 16 + ax0
                            )
                            v1 = Ts.axis.spatial(
                                512,
                                i0_0_i1_0_fused % 16 * 32
                                + i0_1_i1_1_fused % 2 * 16
                                + i0_2_i1_2_fused * 4
                                + ax1,
                            )
                            Ts.reads(C_local[v0, v1])
                            Ts.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]

    decision_0 = [
        ("SamplePerfectTile", [8, 4, 1, 1, 16]),
        ("SamplePerfectTile", [16, 2, 4, 1, 4]),
        ("SamplePerfectTile", [128, 2, 2]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 0),
    ]
    mod = te.create_function(te_workload.matmul(512, 512, 512))
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-2080"),  # disable async trace using sm75
        types=ms.schedule_rule.MultiLevelTiling,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cuda_matmul_0],
        expected_decisions=[decision_0],
    )


def test_cuda_matmul_relu():
    @Ts.function
    def cuda_matmul_relu_0(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        compute: T.Tensor((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        C = Ts.sblock_alloc_tensor([512, 512], dtype="float32")
        C_local = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="local")
        A_shared = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="shared")
        B_shared = Ts.sblock_alloc_tensor([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.parallel(64, annotations={"thread_binding": "blockIdx.x"}):
            for i0_1_i1_1_fused in T.parallel(64, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_fused in T.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                    for i2_0 in T.serial(8):
                        for ax0_ax1_fused in T.serial(4096):
                            with Ts.sblock("A_shared"):
                                v0 = Ts.axis.spatial(
                                    512, i0_0_i1_0_fused // 8 * 64 + ax0_ax1_fused // 64
                                )
                                v1 = Ts.axis.spatial(512, i2_0 * 64 + ax0_ax1_fused % 64)
                                Ts.reads(A[v0, v1])
                                Ts.writes(A_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in T.serial(4096):
                            with Ts.sblock("B_shared"):
                                v0 = Ts.axis.spatial(512, i2_0 * 64 + ax0_ax1_fused // 64)
                                v1 = Ts.axis.spatial(
                                    512, i0_0_i1_0_fused % 8 * 64 + ax0_ax1_fused % 64
                                )
                                Ts.reads(B[v0, v1])
                                Ts.writes(B_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(8, 2, 1, 8, 2, 2):
                            with Ts.sblock("C"):
                                i = Ts.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused // 8 * 64
                                    + i0_1_i1_1_fused // 8 * 8
                                    + i0_2_i1_2_fused // 4 * 4
                                    + i0_3 * 2
                                    + i0_4,
                                )
                                j = Ts.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused % 8 * 64
                                    + i0_1_i1_1_fused % 8 * 8
                                    + i0_2_i1_2_fused % 4 * 2
                                    + i1_3 * 2
                                    + i1_4,
                                )
                                k = Ts.axis.reduce(512, i2_0 * 64 + i2_1 * 8 + i2_2)
                                Ts.reads(A_shared[i, k], B_shared[k, j])
                                Ts.writes(C_local[i, j])
                                Ts.sblock_attr(
                                    {
                                        "meta_schedule.thread_extent_high_inclusive": 1024,
                                        "meta_schedule.thread_extent_low_inclusive": 32,
                                        "meta_schedule.tiling_structure": "SSSRRSRS",
                                    }
                                )
                                with Ts.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(4, 2):
                        with Ts.sblock("C_local"):
                            v0 = Ts.axis.spatial(
                                512,
                                i0_0_i1_0_fused // 8 * 64
                                + i0_1_i1_1_fused // 8 * 8
                                + i0_2_i1_2_fused // 4 * 4
                                + ax0,
                            )
                            v1 = Ts.axis.spatial(
                                512,
                                i0_0_i1_0_fused % 8 * 64
                                + i0_1_i1_1_fused % 8 * 8
                                + i0_2_i1_2_fused % 4 * 2
                                + ax1,
                            )
                            Ts.reads(C_local[v0, v1])
                            Ts.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
        for i0, i1 in T.grid(512, 512):
            with Ts.sblock("compute"):
                i0_1, i1_1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(C[i0_1, i1_1])
                Ts.writes(compute[i0_1, i1_1])
                compute[i0_1, i1_1] = T.max(C[i0_1, i1_1], T.float32(0))

    decision_0 = [
        ("SamplePerfectTile", [8, 8, 2, 2, 2]),
        ("SamplePerfectTile", [8, 8, 4, 1, 2]),
        ("SamplePerfectTile", [8, 8, 8]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 3),
    ]
    mod = te.create_function(te_workload.matmul_relu(512, 512, 512))
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-2080"),  # disable async trace using sm75
        types=ms.schedule_rule.MultiLevelTiling,
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cuda_matmul_relu_0],
        expected_decisions=[decision_0],
    )


def test_cuda_sum_with_trivial_block_iter():
    @Ts.function
    def sum_with_trivial_block_iter(
        A: T.Tensor((1, 64, 768), "float32"),
        B: T.Tensor((1, 64, 1), "float32"),
    ) -> None:
        for i0, i1, i2, i3 in T.grid(1, 64, 1, 768):
            with Ts.sblock("sum"):
                ax0, ax1, ax2, k2 = Ts.axis.remap("SSSR", [i0, i1, i2, i3])
                Ts.reads(A[ax0, ax1, k2])
                Ts.writes(B[ax0, ax1, ax2])
                with Ts.init():
                    B[ax0, ax1, ax2] = T.float32(0)
                B[ax0, ax1, ax2] = B[ax0, ax1, ax2] + A[ax0, ax1, k2]

    # Expect nothing to happen - the rule is not supposed to be applied in this case
    mod = sum_with_trivial_block_iter
    (sch,) = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-3080"),
        types=ms.schedule_rule.MultiLevelTiling,
    )
    assert not sch.trace.simplified(remove_postproc=True).insts


def test_multi_level_tiling_hexagon():
    @Ts.function
    def cpu_conv2d_nhwc(
        inputs: T.Tensor((1, 56, 56, 64), "float16"),
        weight: T.Tensor((3, 3, 64, 64), "float16"),
        conv2d_nhwc: T.Tensor((1, 56, 56, 64), "float16"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        PadInput = Ts.sblock_alloc_tensor((1, 58, 58, 64), "float16")
        for i0, i1, i2, i3 in T.grid(1, 58, 58, 64):
            with Ts.sblock("PadInput"):
                v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3])
                Ts.writes(PadInput[v_i0, v_i1, v_i2, v_i3])
                PadInput[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                    1 <= v_i1 and v_i1 < 57 and 1 <= v_i2 and v_i2 < 57,
                    inputs[v_i0, v_i1 - 1, v_i2 - 1, v_i3],
                    T.float16(0),
                )
        for (
            n_0,
            h_0,
            w_0,
            co_0,
            rh_0,
            rw_0,
            rc_0,
            n_1,
            h_1,
            w_1,
            co_1,
            rh_1,
            rw_1,
            rc_1,
            n_2,
            h_2,
            w_2,
            co_2,
        ) in T.grid(1, 1, 2, 1, 3, 3, 16, 1, 14, 2, 1, 1, 1, 4, 1, 4, 14, 64):
            with Ts.sblock("conv2d_nhwc"):
                v_n = Ts.axis.spatial(1, n_0 + n_1 + n_2)
                v_h = Ts.axis.spatial(56, h_0 * 56 + h_1 * 4 + h_2)
                v_w = Ts.axis.spatial(56, w_0 * 28 + w_1 * 14 + w_2)
                v_co = Ts.axis.spatial(64, co_0 * 64 + co_1 * 64 + co_2)
                v_rh = Ts.axis.reduce(3, rh_0 + rh_1)
                v_rw = Ts.axis.reduce(3, rw_0 + rw_1)
                v_rc = Ts.axis.reduce(64, rc_0 * 4 + rc_1)
                Ts.reads(
                    PadInput[v_n, v_h + v_rh, v_w + v_rw, v_co // 64 * 64 + v_rc],
                    weight[v_rh, v_rw, v_rc, v_co],
                )
                Ts.writes(conv2d_nhwc[v_n, v_h, v_w, v_co])
                Ts.sblock_attr({"meta_schedule.tiling_structure": "SRSRS"})
                with Ts.init():
                    conv2d_nhwc[v_n, v_h, v_w, v_co] = T.float16(0)
                conv2d_nhwc[v_n, v_h, v_w, v_co] = (
                    conv2d_nhwc[v_n, v_h, v_w, v_co]
                    + PadInput[v_n, v_h + v_rh, v_w + v_rw, v_co // 64 * 64 + v_rc]
                    * weight[v_rh, v_rw, v_rc, v_co]
                )

    target_hexagon = Target("qcom/hexagon-v69")

    I = 64
    O = 64
    H = 56
    W = 56

    mod = te.create_function(
        te_workload.conv2d_nhwc(1, H, W, I, O, 3, 1, 1, 1, in_dtype="float16", out_dtype="float16")
    )

    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target(target_hexagon, host=target_hexagon),
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTilingWideVector(
                structure="SRSRS",
                vector_length_in_bits=1024,
                max_innermost_factor=64,
                reuse_read=None,
                reuse_write=None,
            )
        ],
    )

    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1]),
        ("SamplePerfectTile", [1, 14, 4]),
        ("SamplePerfectTile", [2, 2, 14]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [3, 1]),
        ("SamplePerfectTile", [16, 4]),
    ]

    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[cpu_conv2d_nhwc],
        expected_decisions=[decision_0],
    )


def test_cache_read_specify_consumer():
    @Ts.function
    def cache_read_specify_consumer_0(
        A: T.Tensor((512, 512), "float32"),
        B: T.Tensor((512, 512), "float32"),
        T_add: T.Tensor((512, 512), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        C = Ts.sblock_alloc_tensor((512, 512))
        C_local = Ts.sblock_alloc_tensor((512, 512), scope="local")
        A_shared = Ts.sblock_alloc_tensor((512, 512), scope="shared")
        B_shared = Ts.sblock_alloc_tensor((512, 512), scope="shared")
        for i_0_j_0_fused in T.parallel(2, annotations={"thread_binding": "blockIdx.x"}):
            for i_1_j_1_fused in T.parallel(512, annotations={"thread_binding": "vthread.x"}):
                for i_2_j_2_fused in T.parallel(16, annotations={"thread_binding": "threadIdx.x"}):
                    for k_0 in range(2):
                        for ax0_ax1_fused in range(131072):
                            with Ts.sblock("A_shared"):
                                v0 = Ts.axis.spatial(512, ax0_ax1_fused // 256)
                                v1 = Ts.axis.spatial(512, k_0 * 256 + ax0_ax1_fused % 256)
                                Ts.reads(A[v0, v1])
                                Ts.writes(A_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 2})
                                A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused in range(65536):
                            with Ts.sblock("B_shared"):
                                v0 = Ts.axis.spatial(512, k_0 * 256 + ax0_ax1_fused // 256)
                                v1 = Ts.axis.spatial(512, i_0_j_0_fused * 256 + ax0_ax1_fused % 256)
                                Ts.reads(B[v0, v1])
                                Ts.writes(B_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 3})
                                B_shared[v0, v1] = B[v0, v1]
                        for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(64, 1, 1, 4, 1, 16):
                            with Ts.sblock("C"):
                                v_i = Ts.axis.spatial(
                                    512,
                                    i_1_j_1_fused // 8 * 8 + i_2_j_2_fused // 2 + i_3 + i_4,
                                )
                                v_j = Ts.axis.spatial(
                                    512,
                                    i_0_j_0_fused * 256
                                    + i_1_j_1_fused % 8 * 32
                                    + i_2_j_2_fused % 2 * 16
                                    + j_3 * 16
                                    + j_4,
                                )
                                v_k = Ts.axis.reduce(512, k_0 * 256 + k_1 * 4 + k_2)
                                Ts.reads(A_shared[v_i, v_k], B_shared[v_k, v_j])
                                Ts.writes(C_local[v_i, v_j])
                                Ts.sblock_attr(
                                    {
                                        "meta_schedule.thread_extent_high_inclusive": 1024,
                                        "meta_schedule.thread_extent_low_inclusive": 32,
                                        "meta_schedule.tiling_structure": "SSSRRSRS",
                                    }
                                )
                                with Ts.init():
                                    C_local[v_i, v_j] = T.float32(0)
                                C_local[v_i, v_j] = (
                                    C_local[v_i, v_j] + A_shared[v_i, v_k] * B_shared[v_k, v_j]
                                )
                    for ax0, ax1 in T.grid(1, 16):
                        with Ts.sblock("C_local"):
                            v0 = Ts.axis.spatial(
                                512,
                                i_1_j_1_fused // 8 * 8 + i_2_j_2_fused // 2 + ax0,
                            )
                            v1 = Ts.axis.spatial(
                                512,
                                i_0_j_0_fused * 256
                                + i_1_j_1_fused % 8 * 32
                                + i_2_j_2_fused % 2 * 16
                                + ax1,
                            )
                            Ts.reads(C_local[v0, v1])
                            Ts.writes(C[v0, v1])
                            C[v0, v1] = C_local[v0, v1]
        for ax0, ax1 in T.grid(512, 512):
            with Ts.sblock("T_add"):
                v_ax0 = Ts.axis.spatial(512, ax0)
                v_ax1 = Ts.axis.spatial(512, ax1)
                Ts.reads(C[v_ax0, v_ax1], A[v_ax0, v_ax1])
                Ts.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = C[v_ax0, v_ax1] + A[v_ax0, v_ax1]

    decision_0 = [
        ("SamplePerfectTile", [1, 64, 8, 1, 1]),
        ("SamplePerfectTile", [2, 8, 2, 1, 16]),
        ("SamplePerfectTile", [2, 64, 4]),
        ("SampleCategorical", 1),
        ("SampleCategorical", 2),
    ]
    A, B, C = te_workload.matmul(512, 512, 512)
    mod = te.create_function([A, B, C + A])

    space = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/geforce-rtx-2080"),  # disable async trace using sm75
        types=ms.schedule_rule.MultiLevelTiling,
    )
    check_sketches(
        mod,
        sketches=space,
        expected_mods=[cache_read_specify_consumer_0],
        expected_decisions=[decision_0],
    )


def test_max_pool_blocked():
    # fmt off
    @Ts.function
    def pool_blocked_cache_read_write(
        X: T.Tensor((1, 2, 8, 8, 8, 8, 32), "uint8"),
        pool: T.Tensor((1, 2, 4, 4, 8, 8, 32), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        pool_global = Ts.sblock_alloc_tensor((1, 2, 4, 4, 8, 8, 32), "uint8")
        X_global = Ts.sblock_alloc_tensor((1, 2, 8, 8, 8, 8, 32), "uint8")
        for b_0, c_o_0, h_o_0, w_o_0, h_i_0, w_i_0, c_i_0 in T.grid(1, 2, 4, 1, 8, 1, 4):
            for ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused in range(896):
                with Ts.sblock("X_global"):
                    v0 = Ts.axis.spatial(1, 0)
                    v1 = Ts.axis.spatial(2, c_o_0)
                    v2 = Ts.axis.spatial(8, h_o_0 * 2)
                    v3 = Ts.axis.spatial(8, ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused // 128)
                    v4 = Ts.axis.spatial(
                        8, h_i_0 % 4 * 2 + ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 128 // 64
                    )
                    v5 = Ts.axis.spatial(8, ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 64 // 8)
                    v6 = Ts.axis.spatial(32, c_i_0 * 8 + ax0_ax1_ax2_ax3_ax4_ax5_ax6_fused % 8)
                    Ts.reads(X[v0, v1, v2, v3, v4, v5, v6])
                    Ts.writes(X_global[v0, v1, v2, v3, v4, v5, v6])
                    X_global[v0, v1, v2, v3, v4, v5, v6] = X[v0, v1, v2, v3, v4, v5, v6]
            for wh, ww, b_1, c_o_1, h_o_1, w_o_1, h_i_1, w_i_1, c_i_1 in T.grid(
                2, 2, 1, 1, 1, 4, 1, 8, 8
            ):
                with Ts.sblock("pool"):
                    v_b = Ts.axis.spatial(1, b_0 + b_1)
                    v_c_o = Ts.axis.spatial(2, c_o_0 + c_o_1)
                    v_h_o = Ts.axis.spatial(4, h_o_0 + h_o_1)
                    v_w_o = Ts.axis.spatial(4, w_o_0 * 4 + w_o_1)
                    v_h_i = Ts.axis.spatial(8, h_i_0 + h_i_1)
                    v_w_i = Ts.axis.spatial(8, w_i_0 * 8 + w_i_1)
                    v_c_i = Ts.axis.spatial(32, c_i_0 * 8 + c_i_1)
                    v_wh, v_ww = Ts.axis.remap("RR", [wh, ww])
                    Ts.reads(
                        X_global[
                            v_b,
                            v_c_o,
                            v_h_i // 8 * 2 + v_h_o * 2,
                            v_w_i // 8 * 2 + v_w_o * 2,
                            v_h_i % 4 * 2 + v_wh,
                            v_w_i % 4 * 2 + v_ww,
                            v_c_i,
                        ]
                    )
                    Ts.writes(pool_global[v_b, v_c_o, v_h_o, v_w_o, v_h_i, v_w_i, v_c_i])
                    Ts.sblock_attr({"meta_schedule.tiling_structure": "SRS"})
                    with Ts.init():
                        pool_global[v_b, v_c_o, v_h_o, v_w_o, v_h_i, v_w_i, v_c_i] = T.uint8(0)
                    pool_global[v_b, v_c_o, v_h_o, v_w_o, v_h_i, v_w_i, v_c_i] = T.max(
                        pool_global[v_b, v_c_o, v_h_o, v_w_o, v_h_i, v_w_i, v_c_i],
                        X_global[
                            v_b,
                            v_c_o,
                            v_h_i // 8 * 2 + v_h_o * 2,
                            v_w_i // 8 * 2 + v_w_o * 2,
                            v_h_i % 4 * 2 + v_wh,
                            v_w_i % 4 * 2 + v_ww,
                            v_c_i,
                        ],
                    )
            for ax0, ax1, ax2, ax3, ax4, ax5, ax6 in T.grid(1, 1, 1, 4, 1, 8, 8):
                with Ts.sblock("pool_global"):
                    v0 = Ts.axis.spatial(1, ax0)
                    v1 = Ts.axis.spatial(2, c_o_0 + ax1)
                    v2 = Ts.axis.spatial(4, h_o_0 + ax2)
                    v3 = Ts.axis.spatial(4, ax3)
                    v4 = Ts.axis.spatial(8, h_i_0 + ax4)
                    v5 = Ts.axis.spatial(8, ax5)
                    v6 = Ts.axis.spatial(32, c_i_0 * 8 + ax6)
                    Ts.reads(pool_global[v0, v1, v2, v3, v4, v5, v6])
                    Ts.writes(pool[v0, v1, v2, v3, v4, v5, v6])
                    pool[v0, v1, v2, v3, v4, v5, v6] = pool_global[v0, v1, v2, v3, v4, v5, v6]

    # fmt on

    def max_pool_blocked_compute(height, width, channel):
        ishape = (1, channel // 32, height // 8, width // 8, 8, 8, 32)
        oshape = (1, channel // 32, height // 8 // 2, width // 8 // 2, 8, 8, 32)
        X = te.placeholder(ishape, name="X", dtype="uint8")

        window_h = te.reduce_axis((0, 2), name="wh")
        window_w = te.reduce_axis((0, 2), name="ww")

        out = te.compute(
            oshape,
            lambda b, c_o, h_o, w_o, h_i, w_i, c_i: te.max(
                X[
                    b,
                    c_o,
                    (h_o * 8 + h_i) // 8 * 2,
                    (w_o * 8 + w_i) // 8 * 2,
                    (h_o * 8 + h_i) % 4 * 2 + window_h,
                    (w_o * 8 + w_i) % 4 * 2 + window_w,
                    c_i,
                ],
                axis=[window_h, window_w],
            ),
            name="pool",
        )
        return [X, out]

    height = width = 64
    channel = 64

    mod = te.create_function(max_pool_blocked_compute(height, width, channel))

    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target("llvm"),
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTiling(
                structure="SRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=ms.schedule_rule.ReuseType(
                    req="must",
                    levels=[1],
                    scope="global",
                ),
                reuse_write=ms.schedule_rule.ReuseType(req="must", levels=[1], scope="global"),
                filter_fn=lambda sch, block_rv: sch.get(block_rv).name_hint == "pool",
            )
        ],
    )

    decision = [
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [2, 1]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 4]),
        ("SamplePerfectTile", [8, 1]),
        ("SamplePerfectTile", [1, 8]),
        ("SamplePerfectTile", [4, 8]),
    ]

    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[pool_blocked_cache_read_write],
        expected_decisions=[decision],
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
