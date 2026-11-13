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
# ruff: noqa: E501, F401, F841
import pytest

import tvm
import tvm.s_tir.meta_schedule as ms
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import Schedule
from tvm.s_tir import script as Ts
from tvm.s_tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN
from tvm_tirx.codegen import llvm
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target
from tvm_tirx.tirx import floordiv, floormod


# fmt: off
@tvm_tirx.script.ir_module
class Dense:
    @Ts.function
    def main(
        p0: T.Tensor((128, 128), "float32"),
        p1: T.Tensor((128, 128), "float32"),
        T_matmul_NT: T.Tensor((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"layout_free_tensors": [1], "tirx.noalias": True, "global_symbol": "main"})
        # body
        # with Ts.sblock("root")
        for i0, i1, i2 in T.grid(128, 128, 128):
            with Ts.sblock("T_matmul_NT"):
                i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
                Ts.reads(p0[i, k], p1[j, k])
                Ts.writes(T_matmul_NT[i, j])
                Ts.sblock_attr({"layout_free_placeholders": []})
                with Ts.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1[j, k]

@tvm_tirx.script.ir_module
class DenseAdd:
    @Ts.function
    def main(
        p0: T.Tensor((128, 128), "float32"),
        p1: T.Tensor((128, 128), "float32"),
        T_add: T.Tensor((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
        # body
        # with Ts.sblock("root")
        T_matmul_NT = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        compile_engine_const = Ts.sblock_alloc_tensor([], dtype="float32")
        for i0, i1, i2 in T.grid(128, 128, 128):
            with Ts.sblock("T_matmul_NT"):
                i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
                Ts.reads(p0[i, k], p1[j, k])
                Ts.writes(T_matmul_NT[i, j])
                Ts.sblock_attr({"layout_free_placeholders": []})
                with Ts.init():
                    T_matmul_NT[i, j] = T.float32(0)
                T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1[j, k]
        with Ts.sblock("compile_engine_const"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const[()])
            compile_engine_const[()] = T.float32(1)
        for i0, i1 in T.grid(128, 128):
            with Ts.sblock("T_add"):
                ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(T_matmul_NT[ax0, ax1], compile_engine_const[()])
                Ts.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + compile_engine_const[()]

@tvm_tirx.script.ir_module
class DenseAdd_scheduled_cpu:
    @Ts.function
    def main(
        p0: T.Tensor((128, 128), "float32"),
        p1: T.Tensor((128, 128), "float32"),
        T_add: T.Tensor((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
        # body
        # with Ts.sblock("root")
        T_matmul_NT_global = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        p1_global = Ts.sblock_alloc_tensor([2, 128, 64], dtype="float32")
        for ax0, ax1 in T.grid(128, 128):
            with Ts.sblock("p1_global"):
                v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(p1[v0, v1])
                Ts.writes(p1_global[v0 // 64, v1, v0 % 64])
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc": True})
                p1_global[v0 // 64, v1, v0 % 64] = p1[v0, v1]
        for i0_0_i1_0_fused_fused in T.parallel(4):
            for i0_1, i1_1 in T.grid(8, 1):
                for i0_2_init, i1_2_init, i0_3_init in T.grid(4, 1, 2):
                    for i1_3_fused_init in T.vectorized(64):
                        with Ts.sblock("T_matmul_NT_init"):
                            i = Ts.axis.spatial(
                                128,
                                i0_0_i1_0_fused_fused // 2 * 64
                                + i0_1 * 8
                                + i0_2_init * 2
                                + i0_3_init,
                            )
                            j = Ts.axis.spatial(
                                128,
                                i0_0_i1_0_fused_fused % 2 * 64
                                + i1_1 * 64
                                + i1_2_init * 64
                                + i1_3_fused_init,
                            )
                            Ts.reads()
                            Ts.writes(T_matmul_NT_global[i, j])
                            Ts.sblock_attr(
                                {
                                    "layout_free_placeholders": [],
                                    "meta_schedule.tiling_structure": "SSRSRS",
                                }
                            )
                            T_matmul_NT_global[i, j] = T.float32(0)
                for i2_0, i0_2, i1_2, i2_1, i0_3 in T.grid(128, 4, 1, 1, 2):
                    for i1_3_fused in T.vectorized(64):
                        with Ts.sblock("T_matmul_NT_update"):
                            i = Ts.axis.spatial(
                                128, i0_0_i1_0_fused_fused // 2 * 64 + i0_1 * 8 + i0_2 * 2 + i0_3
                            )
                            j = Ts.axis.spatial(
                                128,
                                i0_0_i1_0_fused_fused % 2 * 64 + i1_1 * 64 + i1_2 * 64 + i1_3_fused,
                            )
                            k = Ts.axis.reduce(128, i2_0 + i2_1)
                            Ts.reads(
                                T_matmul_NT_global[i, j], p0[i, k], p1_global[j // 64, k, j % 64]
                            )
                            Ts.writes(T_matmul_NT_global[i, j])
                            Ts.sblock_attr(
                                {
                                    "layout_free_placeholders": [],
                                    "meta_schedule.tiling_structure": "SSRSRS",
                                }
                            )
                            T_matmul_NT_global[i, j] = (
                                T_matmul_NT_global[i, j] + p0[i, k] * p1_global[j // 64, k, j % 64]
                            )
            for ax0 in T.serial(64):
                for ax1_fused in T.vectorized(64):
                    with Ts.sblock("T_matmul_NT_global"):
                        v0 = Ts.axis.spatial(128, i0_0_i1_0_fused_fused // 2 * 64 + ax0)
                        v1 = Ts.axis.spatial(128, i0_0_i1_0_fused_fused % 2 * 64 + ax1_fused)
                        Ts.reads(T_matmul_NT_global[v0, v1])
                        Ts.writes(T_add[v0, v1])
                        T_add[v0, v1] = T_matmul_NT_global[v0, v1] + T.float32(1)

@tvm_tirx.script.ir_module
class DenseAdd_cpu_no_write_cache:
    @Ts.function
    def main(p0: T.Tensor((128, 128), "float32"), p1: T.Tensor((128, 128), "float32"), T_add: T.Tensor((128, 128), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
        # body
        # with Ts.sblock("root")
        T_matmul_NT = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        p1_global = Ts.sblock_alloc_tensor([8, 4, 16, 32], dtype="float32")
        for ax0, ax1 in T.grid(128, 128):
            with Ts.sblock("p1_global"):
                v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(p1[v0, v1])
                Ts.writes(p1_global[v1 // 16, v0 // 32, v1 % 16, v0 % 32])
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc": True})
                p1_global[v1 // 16, v0 // 32, v1 % 16, v0 % 32] = p1[v0, v1]
        for i0_0_i1_0_i0_1_i1_1_fused in T.parallel(16, annotations={"tirx.auto_unroll_max_step":16, "tirx.unroll_explicit":1}):
            for i0_2_init, i1_2_init, i0_3_init in T.grid(4, 4, 2):
                for i1_3_fused_init in T.vectorized(32):
                    with Ts.sblock("T_matmul_NT_init"):
                        i = Ts.axis.spatial(128, i0_0_i1_0_i0_1_i1_1_fused * 8 + i0_2_init * 2 + i0_3_init)
                        j = Ts.axis.spatial(128, i1_2_init * 32 + i1_3_fused_init)
                        Ts.reads()
                        Ts.writes(T_matmul_NT[i, j])
                        Ts.sblock_attr({"layout_free_placeholders":[], "meta_schedule.tiling_structure":"SSRSRS"})
                        T_matmul_NT[i, j] = T.float32(0)
            for i2_0, i0_2, i1_2, i2_1, i0_3 in T.grid(8, 4, 4, 16, 2):
                for i1_3_fused in T.vectorized(32):
                    with Ts.sblock("T_matmul_NT_update"):
                        i = Ts.axis.spatial(128, i0_0_i1_0_i0_1_i1_1_fused * 8 + i0_2 * 2 + i0_3)
                        j = Ts.axis.spatial(128, i1_2 * 32 + i1_3_fused)
                        k = Ts.axis.reduce(128, i2_0 * 16 + i2_1)
                        Ts.reads(T_matmul_NT[i, j], p0[i, k], p1_global[k // 16, j // 32, k % 16, j % 32])
                        Ts.writes(T_matmul_NT[i, j])
                        Ts.sblock_attr({"layout_free_placeholders":[], "meta_schedule.tiling_structure":"SSRSRS"})
                        T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1_global[k // 16, j // 32, k % 16, j % 32]
        for i0_i1_fused in T.parallel(16384):
            with Ts.sblock("T_add"):
                ax0 = Ts.axis.spatial(128, i0_i1_fused // 128)
                ax1 = Ts.axis.spatial(128, i0_i1_fused % 128)
                Ts.reads(T_matmul_NT[ax0, ax1])
                Ts.writes(T_add[ax0, ax1])
                T_add[ax0, ax1] = T_matmul_NT[ax0, ax1] + T.float32(1)

@tvm_tirx.script.ir_module
class DenseAdd_scheduled_gpu:
    @Ts.function
    def main(
        p0: T.Tensor((128, 128), "float32"),
        p1: T.Tensor((128, 128), "float32"),
        T_add: T.Tensor((128, 128), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
        # body
        # with Ts.sblock("root")
        T_matmul_NT_local = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="local")
        p0_shared = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="shared")
        p1_shared = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.parallel(32, annotations={"thread_binding": "blockIdx.x", "tirx.auto_unroll_max_step": 64, "tirx.unroll_explicit": 1}):
            for i0_1_i1_1_fused in T.parallel(1, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_fused in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                    for i0_3_init, i1_3_init, i0_4_init, i1_4_init in T.grid(1, 4, 1, 1):
                        with Ts.sblock("T_matmul_NT_init"):
                            i = Ts.axis.spatial(
                                128,
                                i0_0_i1_0_fused // 4 * 16
                                + i0_2_i1_2_fused // 8
                                + i0_3_init
                                + i0_4_init,
                            )
                            j = Ts.axis.spatial(
                                128,
                                i0_0_i1_0_fused % 4 * 32
                                + i0_2_i1_2_fused % 8 * 4
                                + i1_3_init
                                + i1_4_init,
                            )
                            Ts.reads()
                            Ts.writes(T_matmul_NT_local[i, j])
                            Ts.sblock_attr(
                                {
                                    "layout_free_placeholders": [],
                                    "meta_schedule.thread_extent_high_inclusive": 256,
                                    "meta_schedule.thread_extent_low_inclusive": 16,
                                    "meta_schedule.tiling_structure": "SSSRRSRS",
                                }
                            )
                            T_matmul_NT_local[i, j] = T.float32(0)
                    for i2_0 in T.serial(32):
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_fused_2 in T.vectorized(2):
                                    with Ts.sblock("p0_shared"):
                                        Ts.where(
                                            (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1) * 2
                                            + ax0_ax1_fused_2
                                            < 64
                                        )
                                        v0 = Ts.axis.spatial(
                                            128,
                                            i0_0_i1_0_fused // 4 * 16
                                            + (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 2
                                                + ax0_ax1_fused_2
                                            )
                                            // 4,
                                        )
                                        v1 = Ts.axis.spatial(
                                            128,
                                            i2_0 * 4
                                            + (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 2
                                                + ax0_ax1_fused_2
                                            )
                                            % 4,
                                        )
                                        Ts.reads(p0[v0, v1])
                                        Ts.writes(p0_shared[v0, v1])
                                        p0_shared[v0, v1] = p0[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(1):
                            for ax0_ax1_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_fused_2 in T.vectorized(4):
                                    with Ts.sblock("p1_shared"):
                                        Ts.where(
                                            (ax0_ax1_fused_0 * 128 + ax0_ax1_fused_1) * 4
                                            + ax0_ax1_fused_2
                                            < 128
                                        )
                                        v0 = Ts.axis.spatial(
                                            128,
                                            i0_0_i1_0_fused % 4 * 32
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 4
                                                + ax0_ax1_fused_2
                                            )
                                            // 4,
                                        )
                                        v1 = Ts.axis.spatial(
                                            128,
                                            i2_0 * 4
                                            + (
                                                ax0_ax1_fused_0 * 512
                                                + ax0_ax1_fused_1 * 4
                                                + ax0_ax1_fused_2
                                            )
                                            % 4,
                                        )
                                        Ts.reads(p1[v0, v1])
                                        Ts.writes(p1_shared[v0, v1])
                                        p1_shared[v0, v1] = p1[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(1, 1, 4, 4, 1, 1):
                            with Ts.sblock("T_matmul_NT_update"):
                                i = Ts.axis.spatial(
                                    128,
                                    i0_0_i1_0_fused // 4 * 16 + i0_2_i1_2_fused // 8 + i0_3 + i0_4,
                                )
                                j = Ts.axis.spatial(
                                    128,
                                    i0_0_i1_0_fused % 4 * 32
                                    + i0_2_i1_2_fused % 8 * 4
                                    + i1_3
                                    + i1_4,
                                )
                                k = Ts.axis.reduce(128, i2_0 * 4 + i2_1 * 4 + i2_2)
                                Ts.reads(T_matmul_NT_local[i, j], p0_shared[i, k], p1_shared[j, k])
                                Ts.writes(T_matmul_NT_local[i, j])
                                Ts.sblock_attr(
                                    {
                                        "layout_free_placeholders": [],
                                        "meta_schedule.thread_extent_high_inclusive": 256,
                                        "meta_schedule.thread_extent_low_inclusive": 16,
                                        "meta_schedule.tiling_structure": "SSSRRSRS",
                                    }
                                )
                                T_matmul_NT_local[i, j] = (
                                    T_matmul_NT_local[i, j] + p0_shared[i, k] * p1_shared[j, k]
                                )
                    for ax0, ax1 in T.grid(1, 4):
                        with Ts.sblock("T_matmul_NT_local"):
                            v0 = Ts.axis.spatial(
                                128, i0_0_i1_0_fused // 4 * 16 + i0_2_i1_2_fused // 8 + ax0
                            )
                            v1 = Ts.axis.spatial(
                                128, i0_0_i1_0_fused % 4 * 32 + i0_2_i1_2_fused % 8 * 4 + ax1
                            )
                            Ts.reads(T_matmul_NT_local[v0, v1])
                            Ts.writes(T_add[v0, v1])
                            T_add[v0, v1] = T_matmul_NT_local[v0, v1] + T.float32(1)

@tvm_tirx.script.ir_module
class Conv2dInt8:
    @Ts.function
    def main(p0: T.Tensor((16, 56, 56, 64), "int8"), p1: T.Tensor((256, 1, 1, 64), "int8"), p2: T.Tensor((1, 1, 1, 256), "int32"), p3: T.Tensor((1, 1, 1, 256), "int32"), p4: T.Tensor((1, 1, 1, 256), "int64"), p5: T.Tensor((1, 1, 1, 256), "int64"), p6: T.Tensor((1, 1, 1, 256), "int64"), p7: T.Tensor((), "int32"), p8: T.Tensor(1, "int32"), compute: T.Tensor((16, 56, 56, 256), "int32")) -> None:
        # function attr dict
        T.func_attr({"tirx.noalias": True, "global_symbol": "main"})
        # body
        # with Ts.sblock("root")
        pad_temp = Ts.sblock_alloc_tensor([16, 56, 56, 64], dtype="int8")
        conv2d_nhwc = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_subtract = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_cast = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int64")
        T_multiply = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int64")
        T_add_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int64")
        T_right_shift = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int64")
        T_cast_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add_2 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_cast_2 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="uint8")
        T_cast_3 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_subtract_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 64):
            with Ts.sblock("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p0[i0_1, i1_1, i2_1, i3_1])
                Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with Ts.sblock("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                Ts.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                Ts.writes(conv2d_nhwc[nn, yy, xx, ff])
                with Ts.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], ty="int32") * T.cast(p1[ff, ry, rx, rc], ty="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_subtract"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                Ts.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast[ax0, ax1, ax2, ax3])
                T_cast[ax0, ax1, ax2, ax3] = T.cast(T_add[ax0, ax1, ax2, ax3], ty="int64")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_multiply"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_cast[ax0, ax1, ax2, ax3], p4[0, 0, 0, ax3])
                Ts.writes(T_multiply[ax0, ax1, ax2, ax3])
                T_multiply[ax0, ax1, ax2, ax3] = T_cast[ax0, ax1, ax2, ax3] * p4[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_multiply[ax0, ax1, ax2, ax3], p5[0, 0, 0, ax3])
                Ts.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = T_multiply[ax0, ax1, ax2, ax3] + p5[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_right_shift"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add_1[ax0, ax1, ax2, ax3], p6[0, 0, 0, ax3])
                Ts.writes(T_right_shift[ax0, ax1, ax2, ax3])
                T_right_shift[ax0, ax1, ax2, ax3] = T.shift_right(T_add_1[ax0, ax1, ax2, ax3], p6[0, 0, 0, ax3])
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_right_shift[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast_1[ax0, ax1, ax2, ax3])
                T_cast_1[ax0, ax1, ax2, ax3] = T.cast(T_right_shift[ax0, ax1, ax2, ax3], ty="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add_2"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p7[()], T_cast_1[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_2[ax0, ax1, ax2, ax3])
                T_add_2[ax0, ax1, ax2, ax3] = p7[()] + T_cast_1[ax0, ax1, ax2, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute"):
                i0_2, i1_2, i2_2, i3_2 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add_2[i0_2, i1_2, i2_2, i3_2])
                Ts.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.max(T.min(T_add_2[i0_2, i1_2, i2_2, i3_2], 255), 0)
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast_2"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                Ts.reads(compute_1[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast_2[ax0, ax1, ax2, ax3])
                T_cast_2[ax0, ax1, ax2, ax3] = T.cast(compute_1[ax0, ax1, ax2, ax3], ty="uint8")
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast_3"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                Ts.reads(T_cast_2[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast_3[ax0, ax1, ax2, ax3])
                T_cast_3[ax0, ax1, ax2, ax3] = T.cast(T_cast_2[ax0, ax1, ax2, ax3], ty="int32")
        for i0_5, i1_5, i2_5, i3_5 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_subtract_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_5, i1_5, i2_5, i3_5])
                Ts.reads(T_cast_3[ax0, ax1, ax2, ax3], p8[0])
                Ts.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = T_cast_3[ax0, ax1, ax2, ax3] - p8[0]
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_1"):
                i0_7, i1_7, i2_7, i3_7 = Ts.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                Ts.reads(T_subtract_1[i0_7, i1_7, i2_7, i3_7])
                Ts.writes(compute[i0_7, i1_7, i2_7, i3_7])
                compute[i0_7, i1_7, i2_7, i3_7] = T.Cast("int32", T.shift_right(T.Cast("int64", T_subtract_1[i0_7, i1_7, i2_7, i3_7]) * T.Cast("int64", 1963325822), (31) - (1)))

@tvm_tirx.script.ir_module
class Conv2dInt8_target:
    @Ts.function
    def main(p0: T.Tensor((16, 56, 56, 64), "int8"), p1: T.Tensor((256, 1, 1, 64), "int8"), p2: T.Tensor((1, 1, 1, 256), "int32"), p3: T.Tensor((1, 1, 1, 256), "int32"), p4: T.Tensor((1, 1, 1, 256), "int64"), p5: T.Tensor((1, 1, 1, 256), "int64"), p6: T.Tensor((1, 1, 1, 256), "int64"), p7: T.Tensor((), "int32"), p8: T.Tensor(1, "int32"), p9: T.Tensor((16, 56, 56, 256), "int32"), compute: T.Tensor((16, 56, 56, 256), "uint8")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        pad_temp = Ts.sblock_alloc_tensor([16, 56, 56, 64], dtype="int8")
        conv2d_nhwc = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_subtract = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_cast = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int64")
        T_multiply = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int64")
        T_add_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int64")
        T_right_shift = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int64")
        T_cast_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add_2 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_cast_2 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="uint8")
        T_cast_3 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_subtract_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_2 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add_3 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_3 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_cast_4 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="uint8")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 64):
            with Ts.sblock("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p0[i0_1, i1_1, i2_1, i3_1])
                Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with Ts.sblock("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                Ts.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                Ts.writes(conv2d_nhwc[nn, yy, xx, ff])
                with Ts.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], ty="int32") * T.cast(p1[ff, ry, rx, rc], ty="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_subtract"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                Ts.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast[ax0, ax1, ax2, ax3])
                T_cast[ax0, ax1, ax2, ax3] = T.cast(T_add[ax0, ax1, ax2, ax3], ty="int64")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_multiply"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_cast[ax0, ax1, ax2, ax3], p4[0, 0, 0, ax3])
                Ts.writes(T_multiply[ax0, ax1, ax2, ax3])
                T_multiply[ax0, ax1, ax2, ax3] = T_cast[ax0, ax1, ax2, ax3] * p4[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_multiply[ax0, ax1, ax2, ax3], p5[0, 0, 0, ax3])
                Ts.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = T_multiply[ax0, ax1, ax2, ax3] + p5[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_right_shift"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add_1[ax0, ax1, ax2, ax3], p6[0, 0, 0, ax3])
                Ts.writes(T_right_shift[ax0, ax1, ax2, ax3])
                T_right_shift[ax0, ax1, ax2, ax3] = T.shift_right(T_add_1[ax0, ax1, ax2, ax3], p6[0, 0, 0, ax3])
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_right_shift[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast_1[ax0, ax1, ax2, ax3])
                T_cast_1[ax0, ax1, ax2, ax3] = T.cast(T_right_shift[ax0, ax1, ax2, ax3], ty="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add_2"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p7[()], T_cast_1[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_2[ax0, ax1, ax2, ax3])
                T_add_2[ax0, ax1, ax2, ax3] = p7[()] + T_cast_1[ax0, ax1, ax2, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute"):
                i0_2, i1_2, i2_2, i3_2 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add_2[i0_2, i1_2, i2_2, i3_2])
                Ts.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.max(T.min(T_add_2[i0_2, i1_2, i2_2, i3_2], 255), 0)
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast_2"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                Ts.reads(compute_1[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast_2[ax0, ax1, ax2, ax3])
                T_cast_2[ax0, ax1, ax2, ax3] = T.cast(compute_1[ax0, ax1, ax2, ax3], ty="uint8")
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast_3"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                Ts.reads(T_cast_2[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast_3[ax0, ax1, ax2, ax3])
                T_cast_3[ax0, ax1, ax2, ax3] = T.cast(T_cast_2[ax0, ax1, ax2, ax3], ty="int32")
        for i0_5, i1_5, i2_5, i3_5 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_subtract_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_5, i1_5, i2_5, i3_5])
                Ts.reads(T_cast_3[ax0, ax1, ax2, ax3], p8[0])
                Ts.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = T_cast_3[ax0, ax1, ax2, ax3] - p8[0]
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_1"):
                i0_7, i1_7, i2_7, i3_7 = Ts.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                Ts.reads(T_subtract_1[i0_7, i1_7, i2_7, i3_7])
                Ts.writes(compute_2[i0_7, i1_7, i2_7, i3_7])
                compute_2[i0_7, i1_7, i2_7, i3_7] = T.Cast("int32", T.shift_right(T.Cast("int64", T_subtract_1[i0_7, i1_7, i2_7, i3_7]) * T.Cast("int64", 1098990753), (31) - (1)))
        for i0_8, i1_8, i2_8, i3_8 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add_3"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_8, i1_8, i2_8, i3_8])
                Ts.reads(compute_2[ax0, ax1, ax2, ax3], p9[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_3[ax0, ax1, ax2, ax3])
                T_add_3[ax0, ax1, ax2, ax3] = compute_2[ax0, ax1, ax2, ax3] + p9[ax0, ax1, ax2, ax3]
        for i0_9, i1_9, i2_9, i3_9 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_2"):
                i0_10, i1_10, i2_10, i3_10 = Ts.axis.remap("SSSS", [i0_9, i1_9, i2_9, i3_9])
                Ts.reads(T_add_3[i0_10, i1_10, i2_10, i3_10])
                Ts.writes(compute_3[i0_10, i1_10, i2_10, i3_10])
                compute_3[i0_10, i1_10, i2_10, i3_10] = T.max(T.min(T_add_3[i0_10, i1_10, i2_10, i3_10], 255), 0)
        for i0_11, i1_11, i2_11, i3_11 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_cast_4"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_11, i1_11, i2_11, i3_11])
                Ts.reads(compute_3[ax0, ax1, ax2, ax3])
                Ts.writes(T_cast_4[ax0, ax1, ax2, ax3])
                T_cast_4[ax0, ax1, ax2, ax3] = T.cast(compute_3[ax0, ax1, ax2, ax3], ty="uint8")
        for i0_12, i1_12, i2_12, i3_12 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_3"):
                i0_13, i1_13, i2_13, i3_13 = Ts.axis.remap("SSSS", [i0_12, i1_12, i2_12, i3_12])
                Ts.reads(T_cast_4[i0_13, i1_13, i2_13, i3_13])
                Ts.writes(compute[i0_13, i1_13, i2_13, i3_13])
                compute[i0_13, i1_13, i2_13, i3_13] = T.max(T.min(T_cast_4[i0_13, i1_13, i2_13, i3_13], T.uint8(255)), T.uint8(0))

C_s0 = T.dynamic("C_s0", "int32")
C_s1 = T.dynamic("C_s1", "int32")
A_3_s0 = T.dynamic("A_3_s0", "int32")
A_3_s1 = T.dynamic("A_3_s1", "int32")
C_4_s0 = T.dynamic("C_4_s0", "int32")
C_4_s1 = T.dynamic("C_4_s1", "int32")
A_s0 = T.dynamic("A_s0", "int32")
A_s1 = T.dynamic("A_s1", "int32")
C_1_s0 = T.dynamic("C_1_s0", "int32")
C_1_s1 = T.dynamic("C_1_s1", "int32")
A_1_s0 = T.dynamic("A_1_s0", "int32")
A_1_s1 = T.dynamic("A_1_s1", "int32")
C_2_s0 = T.dynamic("C_2_s0", "int32")
C_2_s1 = T.dynamic("C_2_s1", "int32")
A_2_s0 = T.dynamic("A_2_s0", "int32")
A_2_s1 = T.dynamic("A_2_s1", "int32")
B_s0 = T.dynamic("B_s0", "int32")
B_s1 = T.dynamic("B_s1", "int32")
C_3_s0 = T.dynamic("C_3_s0", "int32")
C_3_s1 = T.dynamic("C_3_s1", "int32")


@tvm_tirx.script.ir_module
class Conv2dInt8_NCHWc:
    @Ts.function
    def main(p0: T.Tensor((1, 32, 7, 7, 16), "uint8"), p1: T.Tensor((128, 32, 1, 1, 4, 16, 4), "int8"), p2: T.Tensor((1, 128, 1, 1, 16), "int32"), p3: T.Tensor((1, 128, 1, 1, 16), "float32"), p4: T.Tensor(1, "float32"), p5: T.Tensor((1, 128, 7, 7, 16), "int32"), compute: T.Tensor((1, 128, 7, 7, 16), "uint8")) -> None:
        # function attr dict
        T.func_attr({"tirx.noalias": True, "global_symbol": "main"})
        # body
        # with Ts.sblock("root")
        compile_engine_const = Ts.sblock_alloc_tensor([], dtype="float32")
        conv2d_NCHWc_int8 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_add = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_cast = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_multiply = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_1 = Ts.sblock_alloc_tensor([], dtype="float32")
        T_add_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_floor = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_cast_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        compute_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_cast_2 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="uint8")
        T_cast_3 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_subtract = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_multiply_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_2 = Ts.sblock_alloc_tensor([], dtype="float32")
        T_add_2 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_floor_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_cast_4 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_add_3 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        compute_2 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_cast_5 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="uint8")
        with Ts.sblock("compile_engine_const"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const[()])
            compile_engine_const[()] = T.float32(0.94537687301635742)
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 128, 7, 7, 16, 1, 1, 32, 4, 4):
            with Ts.sblock("conv2d_NCHWc_int8"):
                n, oc_chunk, oh, ow, oc_block, kh, kw, ic_outer, ic_f_inner, ic_s_inner = Ts.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                Ts.reads(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner])
                Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                Ts.sblock_attr({"schedule_rule":"meta_schedule.conv2d_NCHWc_int8", "workload":["conv2d_NCHWc_int8.x86", ["TENSOR", [1, 32, 7, 7, 16], "uint8"], ["TENSOR", [128, 32, 1, 1, 4, 16, 4], "int8"], [1, 1], [0, 0, 0, 0], [1, 1], "NCHW16c", "NCHW16c", "int32"]})
                with Ts.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] + T.cast(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], ty="int32") * T.cast(p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner], ty="int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(conv2d_NCHWc_int8[ax0, ax1, ax2, ax3, ax4], p2[ax0, ax1, 0, 0, ax4])
                Ts.writes(T_add[ax0, ax1, ax2, ax3, ax4])
                T_add[ax0, ax1, ax2, ax3, ax4] = conv2d_NCHWc_int8[ax0, ax1, ax2, ax3, ax4] + p2[ax0, ax1, 0, 0, ax4]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast[ax0, ax1, ax2, ax3, ax4])
                T_cast[ax0, ax1, ax2, ax3, ax4] = T.cast(T_add[ax0, ax1, ax2, ax3, ax4], ty="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_multiply"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast[ax0, ax1, ax2, ax3, ax4], p3[ax0, ax1, 0, 0, ax4])
                Ts.writes(T_multiply[ax0, ax1, ax2, ax3, ax4])
                T_multiply[ax0, ax1, ax2, ax3, ax4] = T_cast[ax0, ax1, ax2, ax3, ax4] * p3[ax0, ax1, 0, 0, ax4]
        with Ts.sblock("compile_engine_const_1"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const_1[()])
            compile_engine_const_1[()] = T.float32(54.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_multiply[ax0, ax1, ax2, ax3, ax4], compile_engine_const_1[()])
                Ts.writes(T_add_1[ax0, ax1, ax2, ax3, ax4])
                T_add_1[ax0, ax1, ax2, ax3, ax4] = T_multiply[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_1[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_floor"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add_1[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_floor[ax0, ax1, ax2, ax3, ax4])
                T_floor[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_1[ax0, ax1, ax2, ax3, ax4])
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_1"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_floor[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_1[ax0, ax1, ax2, ax3, ax4])
                T_cast_1[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor[ax0, ax1, ax2, ax3, ax4], ty="int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("compute"):
                i0_1, i1_1, i2_1, i3_1, i4_1 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_1[i0_1, i1_1, i2_1, i3_1, i4_1])
                Ts.writes(compute_1[i0_1, i1_1, i2_1, i3_1, i4_1])
                compute_1[i0_1, i1_1, i2_1, i3_1, i4_1] = T.max(T.min(T_cast_1[i0_1, i1_1, i2_1, i3_1, i4_1], 255), 0)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_2"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(compute_1[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_2[ax0, ax1, ax2, ax3, ax4])
                T_cast_2[ax0, ax1, ax2, ax3, ax4] = T.cast(compute_1[ax0, ax1, ax2, ax3, ax4], ty="uint8")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_3"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_2[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_3[ax0, ax1, ax2, ax3, ax4])
                T_cast_3[ax0, ax1, ax2, ax3, ax4] = T.cast(T_cast_2[ax0, ax1, ax2, ax3, ax4], ty="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_subtract"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_3[ax0, ax1, ax2, ax3, ax4], p4[0])
                Ts.writes(T_subtract[ax0, ax1, ax2, ax3, ax4])
                T_subtract[ax0, ax1, ax2, ax3, ax4] = T_cast_3[ax0, ax1, ax2, ax3, ax4] - p4[0]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_multiply_1"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(compile_engine_const[()], T_subtract[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_multiply_1[ax0, ax1, ax2, ax3, ax4])
                T_multiply_1[ax0, ax1, ax2, ax3, ax4] = compile_engine_const[()] * T_subtract[ax0, ax1, ax2, ax3, ax4]
        with Ts.sblock("compile_engine_const_2"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const_2[()])
            compile_engine_const_2[()] = T.float32(0.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add_2"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_multiply_1[ax0, ax1, ax2, ax3, ax4], compile_engine_const_2[()])
                Ts.writes(T_add_2[ax0, ax1, ax2, ax3, ax4])
                T_add_2[ax0, ax1, ax2, ax3, ax4] = T_multiply_1[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_2[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_floor_1"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add_2[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_floor_1[ax0, ax1, ax2, ax3, ax4])
                T_floor_1[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_2[ax0, ax1, ax2, ax3, ax4])
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_4"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_floor_1[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_4[ax0, ax1, ax2, ax3, ax4])
                T_cast_4[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor_1[ax0, ax1, ax2, ax3, ax4], ty="int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add_3"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_4[ax0, ax1, ax2, ax3, ax4], p5[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_add_3[ax0, ax1, ax2, ax3, ax4])
                T_add_3[ax0, ax1, ax2, ax3, ax4] = T_cast_4[ax0, ax1, ax2, ax3, ax4] + p5[ax0, ax1, ax2, ax3, ax4]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("compute_1"):
                i0_2, i1_2, i2_2, i3_2, i4_2 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add_3[i0_2, i1_2, i2_2, i3_2, i4_2])
                Ts.writes(compute_2[i0_2, i1_2, i2_2, i3_2, i4_2])
                compute_2[i0_2, i1_2, i2_2, i3_2, i4_2] = T.max(T.min(T_add_3[i0_2, i1_2, i2_2, i3_2, i4_2], 255), 0)
        for i0_3, i1_3, i2_3, i3_3, i4_3 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_5"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0_3, i1_3, i2_3, i3_3, i4_3])
                Ts.reads(compute_2[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_5[ax0, ax1, ax2, ax3, ax4])
                T_cast_5[ax0, ax1, ax2, ax3, ax4] = T.cast(compute_2[ax0, ax1, ax2, ax3, ax4], ty="uint8")
        for i0_4, i1_4, i2_4, i3_4, i4_4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("compute_2"):
                i0_5, i1_5, i2_5, i3_5, i4_5 = Ts.axis.remap("SSSSS", [i0_4, i1_4, i2_4, i3_4, i4_4])
                Ts.reads(T_cast_5[i0_5, i1_5, i2_5, i3_5, i4_5])
                Ts.writes(compute[i0_5, i1_5, i2_5, i3_5, i4_5])
                compute[i0_5, i1_5, i2_5, i3_5, i4_5] = T.max(T.min(T_cast_5[i0_5, i1_5, i2_5, i3_5, i4_5], T.uint8(255)), T.uint8(0))

@tvm_tirx.script.ir_module
class Conv2dInt8_NCHWc_target:
    @Ts.function
    def main(p0: T.Tensor((1, 32, 7, 7, 16), "uint8"), p1: T.Tensor((128, 32, 1, 1, 4, 16, 4), "int8"), p2: T.Tensor((1, 128, 1, 1, 16), "int32"), p3: T.Tensor((1, 128, 1, 1, 16), "float32"), p4: T.Tensor(1, "float32"), p5: T.Tensor((1, 128, 7, 7, 16), "uint8"), T_cast: T.Tensor((1, 128, 7, 7, 16), "int32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        compile_engine_const = Ts.sblock_alloc_tensor([], dtype="float32")
        conv2d_NCHWc_int8 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_add = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_cast_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_multiply = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_1 = Ts.sblock_alloc_tensor([], dtype="float32")
        T_add_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_floor = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_cast_2 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        compute = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_cast_3 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="uint8")
        T_cast_4 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_subtract = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_multiply_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_2 = Ts.sblock_alloc_tensor([], dtype="float32")
        T_add_2 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_floor_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_cast_5 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        compile_engine_const_3 = Ts.sblock_alloc_tensor([], dtype="float32")
        T_cast_6 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_multiply_2 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        compile_engine_const_4 = Ts.sblock_alloc_tensor([], dtype="float32")
        T_add_3 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_floor_2 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="float32")
        T_cast_7 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_add_4 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        compute_1 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
        T_cast_8 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="uint8")
        compute_2 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="uint8")
        with Ts.sblock("compile_engine_const"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const[()])
            compile_engine_const[()] = T.float32(0.95489668846130371)
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 128, 7, 7, 16, 1, 1, 32, 4, 4):
            with Ts.sblock("conv2d_NCHWc_int8"):
                n, oc_chunk, oh, ow, oc_block, kh, kw, ic_outer, ic_f_inner, ic_s_inner = Ts.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                Ts.reads(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner])
                Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                Ts.sblock_attr({"schedule_rule":"meta_schedule.conv2d_NCHWc_int8", "workload":["conv2d_NCHWc_int8.x86", ["TENSOR", [1, 32, 7, 7, 16], "uint8"], ["TENSOR", [128, 32, 1, 1, 4, 16, 4], "int8"], [1, 1], [0, 0, 0, 0], [1, 1], "NCHW16c", "NCHW16c", "int32"]})
                with Ts.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] + T.cast(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner], ty="int32") * T.cast(p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner], ty="int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(conv2d_NCHWc_int8[ax0, ax1, ax2, ax3, ax4], p2[ax0, ax1, 0, 0, ax4])
                Ts.writes(T_add[ax0, ax1, ax2, ax3, ax4])
                T_add[ax0, ax1, ax2, ax3, ax4] = conv2d_NCHWc_int8[ax0, ax1, ax2, ax3, ax4] + p2[ax0, ax1, 0, 0, ax4]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_1[ax0, ax1, ax2, ax3, ax4])
                T_cast_1[ax0, ax1, ax2, ax3, ax4] = T.cast(T_add[ax0, ax1, ax2, ax3, ax4], ty="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_multiply"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_1[ax0, ax1, ax2, ax3, ax4], p3[ax0, ax1, 0, 0, ax4])
                Ts.writes(T_multiply[ax0, ax1, ax2, ax3, ax4])
                T_multiply[ax0, ax1, ax2, ax3, ax4] = T_cast_1[ax0, ax1, ax2, ax3, ax4] * p3[ax0, ax1, 0, 0, ax4]
        with Ts.sblock("compile_engine_const_1"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const_1[()])
            compile_engine_const_1[()] = T.float32(65.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_multiply[ax0, ax1, ax2, ax3, ax4], compile_engine_const_1[()])
                Ts.writes(T_add_1[ax0, ax1, ax2, ax3, ax4])
                T_add_1[ax0, ax1, ax2, ax3, ax4] = T_multiply[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_1[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_floor"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add_1[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_floor[ax0, ax1, ax2, ax3, ax4])
                T_floor[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_1[ax0, ax1, ax2, ax3, ax4])
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_1"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_floor[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_2[ax0, ax1, ax2, ax3, ax4])
                T_cast_2[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor[ax0, ax1, ax2, ax3, ax4], ty="int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("compute"):
                i0_1, i1_1, i2_1, i3_1, i4_1 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_2[i0_1, i1_1, i2_1, i3_1, i4_1])
                Ts.writes(compute[i0_1, i1_1, i2_1, i3_1, i4_1])
                compute[i0_1, i1_1, i2_1, i3_1, i4_1] = T.max(T.min(T_cast_2[i0_1, i1_1, i2_1, i3_1, i4_1], 255), 0)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_2"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(compute[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_3[ax0, ax1, ax2, ax3, ax4])
                T_cast_3[ax0, ax1, ax2, ax3, ax4] = T.cast(compute[ax0, ax1, ax2, ax3, ax4], ty="uint8")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_3"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_3[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_4[ax0, ax1, ax2, ax3, ax4])
                T_cast_4[ax0, ax1, ax2, ax3, ax4] = T.cast(T_cast_3[ax0, ax1, ax2, ax3, ax4], ty="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_subtract"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_4[ax0, ax1, ax2, ax3, ax4], p4[0])
                Ts.writes(T_subtract[ax0, ax1, ax2, ax3, ax4])
                T_subtract[ax0, ax1, ax2, ax3, ax4] = T_cast_4[ax0, ax1, ax2, ax3, ax4] - p4[0]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_multiply_1"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(compile_engine_const[()], T_subtract[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_multiply_1[ax0, ax1, ax2, ax3, ax4])
                T_multiply_1[ax0, ax1, ax2, ax3, ax4] = compile_engine_const[()] * T_subtract[ax0, ax1, ax2, ax3, ax4]
        with Ts.sblock("compile_engine_const_2"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const_2[()])
            compile_engine_const_2[()] = T.float32(0.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add_2"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_multiply_1[ax0, ax1, ax2, ax3, ax4], compile_engine_const_2[()])
                Ts.writes(T_add_2[ax0, ax1, ax2, ax3, ax4])
                T_add_2[ax0, ax1, ax2, ax3, ax4] = T_multiply_1[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_2[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_floor_1"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add_2[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_floor_1[ax0, ax1, ax2, ax3, ax4])
                T_floor_1[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_2[ax0, ax1, ax2, ax3, ax4])
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_4"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_floor_1[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_5[ax0, ax1, ax2, ax3, ax4])
                T_cast_5[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor_1[ax0, ax1, ax2, ax3, ax4], ty="int32")
        with Ts.sblock("compile_engine_const_3"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const_3[()])
            compile_engine_const_3[()] = T.float32(0.71245479583740234)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_5"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(p5[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_6[ax0, ax1, ax2, ax3, ax4])
                T_cast_6[ax0, ax1, ax2, ax3, ax4] = T.cast(p5[ax0, ax1, ax2, ax3, ax4], ty="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_multiply_2"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(compile_engine_const_3[()], T_cast_6[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_multiply_2[ax0, ax1, ax2, ax3, ax4])
                T_multiply_2[ax0, ax1, ax2, ax3, ax4] = compile_engine_const_3[()] * T_cast_6[ax0, ax1, ax2, ax3, ax4]
        with Ts.sblock("compile_engine_const_4"):
            vi = Ts.axis.spatial(1, 0)
            Ts.reads()
            Ts.writes(compile_engine_const_4[()])
            compile_engine_const_4[()] = T.float32(0.5)
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add_3"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_multiply_2[ax0, ax1, ax2, ax3, ax4], compile_engine_const_4[()])
                Ts.writes(T_add_3[ax0, ax1, ax2, ax3, ax4])
                T_add_3[ax0, ax1, ax2, ax3, ax4] = T_multiply_2[ax0, ax1, ax2, ax3, ax4] + compile_engine_const_4[()]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_floor_2"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add_3[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_floor_2[ax0, ax1, ax2, ax3, ax4])
                T_floor_2[ax0, ax1, ax2, ax3, ax4] = T.floor(T_add_3[ax0, ax1, ax2, ax3, ax4])
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_6"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_floor_2[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_7[ax0, ax1, ax2, ax3, ax4])
                T_cast_7[ax0, ax1, ax2, ax3, ax4] = T.cast(T_floor_2[ax0, ax1, ax2, ax3, ax4], ty="int32")
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_add_4"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_cast_5[ax0, ax1, ax2, ax3, ax4], T_cast_7[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_add_4[ax0, ax1, ax2, ax3, ax4])
                T_add_4[ax0, ax1, ax2, ax3, ax4] = T_cast_5[ax0, ax1, ax2, ax3, ax4] + T_cast_7[ax0, ax1, ax2, ax3, ax4]
        for i0, i1, i2, i3, i4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("compute_1"):
                i0_2, i1_2, i2_2, i3_2, i4_2 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                Ts.reads(T_add_4[i0_2, i1_2, i2_2, i3_2, i4_2])
                Ts.writes(compute_1[i0_2, i1_2, i2_2, i3_2, i4_2])
                compute_1[i0_2, i1_2, i2_2, i3_2, i4_2] = T.max(T.min(T_add_4[i0_2, i1_2, i2_2, i3_2, i4_2], 255), 0)
        for i0_3, i1_3, i2_3, i3_3, i4_3 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_7"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0_3, i1_3, i2_3, i3_3, i4_3])
                Ts.reads(compute_1[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast_8[ax0, ax1, ax2, ax3, ax4])
                T_cast_8[ax0, ax1, ax2, ax3, ax4] = T.cast(compute_1[ax0, ax1, ax2, ax3, ax4], ty="uint8")
        for i0_4, i1_4, i2_4, i3_4, i4_4 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("compute_2"):
                i0_5, i1_5, i2_5, i3_5, i4_5 = Ts.axis.remap("SSSSS", [i0_4, i1_4, i2_4, i3_4, i4_4])
                Ts.reads(T_cast_8[i0_5, i1_5, i2_5, i3_5, i4_5])
                Ts.writes(compute_2[i0_5, i1_5, i2_5, i3_5, i4_5])
                compute_2[i0_5, i1_5, i2_5, i3_5, i4_5] = T.max(T.min(T_cast_8[i0_5, i1_5, i2_5, i3_5, i4_5], T.uint8(255)), T.uint8(0))
        for i0_6, i1_6, i2_6, i3_6, i4_6 in T.grid(1, 128, 7, 7, 16):
            with Ts.sblock("T_cast_8"):
                ax0, ax1, ax2, ax3, ax4 = Ts.axis.remap("SSSSS", [i0_6, i1_6, i2_6, i3_6, i4_6])
                Ts.reads(compute_2[ax0, ax1, ax2, ax3, ax4])
                Ts.writes(T_cast[ax0, ax1, ax2, ax3, ax4])
                T_cast[ax0, ax1, ax2, ax3, ax4] = T.cast(compute_2[ax0, ax1, ax2, ax3, ax4], ty="int32")

def get_conv2d_vnni_mod(intrin_id):
    conv2dint8_nchwc_scheduled_a_offset = T.dynamic("conv2dint8_nchwc_scheduled_a_offset", "int32")
    conv2dint8_nchwc_scheduled_b_offset = T.dynamic("conv2dint8_nchwc_scheduled_b_offset", "int32")
    conv2dint8_nchwc_scheduled_c_offset = T.dynamic("conv2dint8_nchwc_scheduled_c_offset", "int32")

    @tvm_tirx.script.ir_module
    class Conv2dInt8_NCHWc_scheduled:
        @Ts.function
        def main(p0: T.Tensor((1, 32, 7, 7, 16), "uint8"), p1: T.Tensor((128, 32, 1, 1, 4, 16, 4), "int8"), p2: T.Tensor((1, 128, 1, 1, 16), "int32"), p3: T.Tensor((1, 128, 1, 1, 16), "float32"), p4: T.Tensor(1, "float32"), p5: T.Tensor((1, 128, 7, 7, 16), "uint8"), T_cast: T.Tensor((1, 128, 7, 7, 16), "int32")) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            # body
            # with Ts.sblock("root")
            conv2d_NCHWc_int8 = Ts.sblock_alloc_tensor([1, 128, 7, 7, 16], dtype="int32")
            for i0_0_i1_0_i2_0_i3_0_i4_0_0_i0_1_i1_1_fused in T.parallel(128, annotations={"tirx.auto_unroll_max_step":64, "tirx.unroll_explicit":1}):
                for i2_1, i3_1, i4_0_1 in T.grid(7, 1, 1):
                    for i0_2_init, i1_2_init, i2_2_init, i3_2_init, i4_0_2_init, i0_3_init, i1_3_init, i2_3_init, i3_3_init, i4_0_3_init in T.grid(1, 1, 1, 1, 1, 1, 1, 1, 7, 1):
                        with Ts.sblock("conv2d_NCHWc_int8_o_init"):
                            n = Ts.axis.spatial(1, i0_2_init + i0_3_init)
                            oc_chunk = Ts.axis.spatial(128, i0_0_i1_0_i2_0_i3_0_i4_0_0_i0_1_i1_1_fused + i1_2_init + i1_3_init)
                            oh = Ts.axis.spatial(7, i2_1 + i2_2_init + i2_3_init)
                            ow = Ts.axis.spatial(7, i3_1 * 7 + i3_2_init * 7 + i3_3_init)
                            oc_block_o = Ts.axis.spatial(1, i4_0_1 + i4_0_2_init + i4_0_3_init)
                            Ts.reads()
                            Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16])
                            for i4_1 in T.vectorized(16):
                                with Ts.sblock("conv2d_NCHWc_int8_init"):
                                    oc_block_i_init = Ts.axis.spatial(16, i4_1)
                                    Ts.reads()
                                    Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init])
                                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                    for i5_0, i6_0, i7_0, i8_0, i9_0_0, i0_2, i1_2, i2_2, i3_2, i4_0_2, i5_1, i6_1, i7_1, i8_1, i9_0_1, i0_3, i1_3, i2_3, i3_3, i4_0_3 in T.grid(1, 1, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 7, 1):
                        with Ts.sblock("conv2d_NCHWc_int8_o_update"):
                            n = Ts.axis.spatial(1, i0_2 + i0_3)
                            oc_chunk = Ts.axis.spatial(128,  i0_0_i1_0_i2_0_i3_0_i4_0_0_i0_1_i1_1_fused + i1_2 + i1_3)
                            oh = Ts.axis.spatial(7, i2_1 + i2_2 + i2_3)
                            ow = Ts.axis.spatial(7, i3_1 * 7 + i3_2 * 7 + i3_3)
                            oc_block_o = Ts.axis.spatial(1, i4_0_1 + i4_0_2 + i4_0_3)
                            kh = Ts.axis.reduce(1, i5_0 + i5_1)
                            kw = Ts.axis.reduce(1, i6_0 + i6_1)
                            ic_outer = Ts.axis.reduce(32, i7_0 * 8 + i7_1)
                            ic_f_inner = Ts.axis.reduce(4, i8_0 + i8_1)
                            ic_s_inner_o = Ts.axis.reduce(1, i9_0_0 + i9_0_1)
                            Ts.reads(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16], p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4], p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0 : 16, 0 : 4])
                            Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16])
                            A = Ts.match_tensor(p0[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4], [4], dtype="uint8", layout=T.TileLayout(T.S[[4]] + (conv2dint8_nchwc_scheduled_a_offset)))
                            B = Ts.match_tensor(p1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0 : 16, 0 : 4], [16, 4], dtype="int8", layout=T.TileLayout(T.S[[16, 4]] + (conv2dint8_nchwc_scheduled_b_offset)))
                            C = Ts.match_tensor(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0 : 16], [16], dtype="int32", layout=T.TileLayout(T.S[[16]] + (conv2dint8_nchwc_scheduled_c_offset)))
                            A_u8x4 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.uint8x4, scope="local")])
                            A_u8x4[0] = A[T.ramp(0, 1, 4)]
                            A_i32 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32, scope="local")])
                            A_i32[0] = T.reinterpret(A_u8x4[0], ty="int32")
                            B_i8x64 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int8x64, scope="local")])
                            B_i8x64[0] = B[0, T.ramp(0, 1, 64)]
                            B_i32x16 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32x16, scope="local")])
                            B_i32x16[0] = T.reinterpret(B_i8x64[0], ty="int32x16")
                            C_i32x16 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32x16, scope="local")])
                            C_i32x16[0] = C[T.ramp(0, 1, 16)]
                            C[T.ramp(0, 1, 16)] = T.call_llvm_pure_intrin(T.uint32(intrin_id), C_i32x16[0], T.broadcast(A_i32[0], 16), B_i32x16[0], ty="int32x16")
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 1, 7):
                        for ax4_fused in T.vectorized(16):
                            with Ts.sblock("T_cast_8"):
                                ax0_1 = Ts.axis.spatial(1, ax0)
                                ax1_1 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_i3_0_i4_0_0_i0_1_i1_1_fused + ax1)
                                ax2_1 = Ts.axis.spatial(7, i2_1 + ax2)
                                ax3_1, ax4 = Ts.axis.remap("SS", [ax3, ax4_fused])
                                Ts.reads(conv2d_NCHWc_int8[ax0_1, ax1_1, ax2_1, ax3_1, ax4], p2[ax0_1, ax1_1, 0, 0, ax4], p3[ax0_1, ax1_1, 0, 0, ax4], p4[0], p5[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                                Ts.writes(T_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4])
                                T_cast[ax0_1, ax1_1, ax2_1, ax3_1, ax4] = T.cast(T.max(T.min(T.cast(T.max(T.min(T.cast(T.floor(T.float32(0.95489668846130371) * (T.cast(T.cast(T.max(T.min(T.cast(T.floor(T.cast(conv2d_NCHWc_int8[ax0_1, ax1_1, ax2_1, ax3_1, ax4] + p2[ax0_1, ax1_1, 0, 0, ax4], ty="float32") * p3[ax0_1, ax1_1, 0, 0, ax4] + T.float32(65.5)), ty="int32"), 255), 0), ty="uint8"), ty="float32") - p4[0]) + T.float32(0.5)), ty="int32") + T.cast(T.floor(T.float32(0.71245479583740234) * T.cast(p5[ax0_1, ax1_1, ax2_1, ax3_1, ax4], ty="float32") + T.float32(0.5)), ty="int32"), 255), 0), ty="uint8"), T.uint8(255)), T.uint8(0)), ty="int32")

    return Conv2dInt8_NCHWc_scheduled

@tvm_tirx.script.ir_module
class Conv2dWinogradAddRelu:
    @Ts.function
    def main(p0: T.Tensor((1, 56, 56, 64), "float32"), p1: T.Tensor((6, 6, 64, 64), "float32"), p2: T.Tensor((1, 1, 1, 64), "float32"), T_relu: T.Tensor((1, 56, 56, 64), "float32")) -> None:
        # function attr dict
        T.func_attr({"layout_free_tensors": [1], "tirx.noalias": True, "global_symbol": "main"})
        # body
        # with Ts.sblock("root")
        data_pad = Ts.sblock_alloc_tensor([1, 58, 58, 64], dtype="float32")
        input_tile = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
        B = Ts.sblock_alloc_tensor([6, 6], dtype="float32")
        data_pack = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
        bgemm = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
        A = Ts.sblock_alloc_tensor([6, 4], dtype="float32")
        inverse = Ts.sblock_alloc_tensor([4, 4, 196, 64], dtype="float32")
        conv2d_winograd = Ts.sblock_alloc_tensor([1, 56, 56, 64], dtype="float32")
        T_add = Ts.sblock_alloc_tensor([1, 56, 56, 64], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 58, 58, 64):
            with Ts.sblock("data_pad"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p0[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                Ts.writes(data_pad[i0_1, i1_1, i2_1, i3_1])
                Ts.sblock_attr({"schedule_rule":"None"})
                data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 57 and 1 <= i2_1 and i2_1 < 57, p0[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0))
        for i0, i1, i2, i3 in T.grid(6, 6, 196, 64):
            with Ts.sblock("input_tile"):
                eps, nu, p, ci = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(data_pad[p // 196, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu, ci])
                Ts.writes(input_tile[eps, nu, p, ci])
                Ts.sblock_attr({"schedule_rule":"None"})
                input_tile[eps, nu, p, ci] = data_pad[p // 196, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu, ci]
        for i0, i1 in T.grid(6, 6):
            with Ts.sblock("B"):
                i, j = Ts.axis.remap("SS", [i0, i1])
                Ts.reads()
                Ts.writes(B[i, j])
                Ts.sblock_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
                B[i, j] = T.Select(i % 6 == 5 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 5 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 6 == 5, T.float32(1.5), T.Select(i % 6 == 4 and j % 6 == 4, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 3, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 2, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 6 == 5, T.float32(-2), T.Select(i % 6 == 3 and j % 6 == 4, T.float32(-0.5), T.Select(i % 6 == 3 and j % 6 == 3, T.float32(2), T.Select(i % 6 == 3 and j % 6 == 2, T.float32(2.5), T.Select(i % 6 == 3 and j % 6 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 6 == 0, T.float32(1.5), T.Select(i % 6 == 2 and j % 6 == 5, T.float32(-1.5), T.Select(i % 6 == 2 and j % 6 == 4, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 3, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 2, T.float32(0.5), T.Select(i % 6 == 2 and j % 6 == 1, T.float32(-2.5), T.Select(i % 6 == 2 and j % 6 == 0, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 4, T.float32(0.5), T.Select(i % 6 == 1 and j % 6 == 3, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 2, T.float32(-1), T.Select(i % 6 == 1 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 0, T.float32(-1.5), T.Select(i % 6 == 0 and j % 6 == 5, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(6, 6, 196, 64, 6, 6):
            with Ts.sblock("data_pack"):
                eps, nu, p, ci, r_a, r_b = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                Ts.reads(input_tile[r_a, r_b, p, ci], B[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(eps, nu) : T.max(eps, nu) + 1])
                Ts.writes(data_pack[eps, nu, p, ci])
                Ts.sblock_attr({"auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_data_pack.cuda"})
                with Ts.init():
                    data_pack[eps, nu, p, ci] = T.float32(0)
                data_pack[eps, nu, p, ci] = data_pack[eps, nu, p, ci] + input_tile[r_a, r_b, p, ci] * B[r_a, eps] * B[r_b, nu]
        for i0, i1, i2, i3, i4 in T.grid(6, 6, 196, 64, 64):
            with Ts.sblock("bgemm"):
                eps, nu, p, co, ci = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, i4])
                Ts.reads(data_pack[eps, nu, p, ci], p1[eps, nu, co, ci])
                Ts.writes(bgemm[eps, nu, p, co])
                Ts.sblock_attr({"layout_free_placeholders":[]})
                with Ts.init():
                    bgemm[eps, nu, p, co] = T.float32(0)
                bgemm[eps, nu, p, co] = bgemm[eps, nu, p, co] + data_pack[eps, nu, p, ci] * p1[eps, nu, co, ci]
        for i0, i1 in T.grid(6, 4):
            with Ts.sblock("A"):
                i, j = Ts.axis.remap("SS", [i0, i1])
                Ts.reads()
                Ts.writes(A[i, j])
                Ts.sblock_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
                A[i, j] = T.Select(i % 6 == 5 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 5 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 4 == 3, T.float32(-8), T.Select(i % 6 == 4 and j % 4 == 2, T.float32(4), T.Select(i % 6 == 4 and j % 4 == 1, T.float32(-2), T.Select(i % 6 == 4 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 4 == 3, T.float32(0.125), T.Select(i % 6 == 3 and j % 4 == 2, T.float32(0.25), T.Select(i % 6 == 3 and j % 4 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 1, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 3, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 1, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 0 and j % 4 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(4, 4, 196, 64, 6, 6):
            with Ts.sblock("inverse"):
                vh, vw, p, co, r_a, r_b = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                Ts.reads(bgemm[r_a, r_b, p, co], A[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(vh, vw) : T.max(vh, vw) + 1])
                Ts.writes(inverse[vh, vw, p, co])
                Ts.sblock_attr({"auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_inverse.cuda"})
                with Ts.init():
                    inverse[vh, vw, p, co] = T.float32(0)
                inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co] * A[r_a, vh] * A[r_b, vw]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with Ts.sblock("conv2d_winograd"):
                n, h, w, co = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co])
                Ts.writes(conv2d_winograd[n, h, w, co])
                conv2d_winograd[n, h, w, co] = inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(conv2d_winograd[ax0, ax1, ax2, ax3], p2[ax0, 0, 0, ax3])
                Ts.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = conv2d_winograd[ax0, ax1, ax2, ax3] + p2[ax0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with Ts.sblock("T_relu"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add[ax0, ax1, ax2, ax3])
                Ts.writes(T_relu[ax0, ax1, ax2, ax3])
                T_relu[ax0, ax1, ax2, ax3] = T.max(T_add[ax0, ax1, ax2, ax3], T.float32(0))

@tvm_tirx.script.ir_module
class Conv2dWinogradAddResidualRelu:
    @Ts.function
    def main(p0: T.Tensor((1, 56, 56, 64), "float32"), p1: T.Tensor((6, 6, 64, 64), "float32"), p2: T.Tensor((1, 1, 1, 64), "float32"), p3: T.Tensor((1, 56, 56, 64), "float32"), T_relu: T.Tensor((1, 56, 56, 64), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
        # body
        # with Ts.sblock("root")
        data_pad = Ts.sblock_alloc_tensor([1, 58, 58, 64], dtype="float32")
        input_tile = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
        B = Ts.sblock_alloc_tensor([6, 6], dtype="float32")
        data_pack = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
        bgemm = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
        A = Ts.sblock_alloc_tensor([6, 4], dtype="float32")
        inverse = Ts.sblock_alloc_tensor([4, 4, 196, 64], dtype="float32")
        conv2d_winograd = Ts.sblock_alloc_tensor([1, 56, 56, 64], dtype="float32")
        T_add = Ts.sblock_alloc_tensor([1, 56, 56, 64], dtype="float32")
        T_add_1 = Ts.sblock_alloc_tensor([1, 56, 56, 64], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 58, 58, 64):
            with Ts.sblock("data_pad"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p0[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                Ts.writes(data_pad[i0_1, i1_1, i2_1, i3_1])
                Ts.sblock_attr({"schedule_rule":"None"})
                data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 57 and 1 <= i2_1 and i2_1 < 57, p0[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0))
        for i0, i1, i2, i3 in T.grid(6, 6, 196, 64):
            with Ts.sblock("input_tile"):
                eps, nu, p, ci = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(data_pad[p // 196, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu, ci])
                Ts.writes(input_tile[eps, nu, p, ci])
                Ts.sblock_attr({"schedule_rule":"None"})
                input_tile[eps, nu, p, ci] = data_pad[p // 196, p % 196 // 14 * 4 + eps, p % 14 * 4 + nu, ci]
        for i0, i1 in T.grid(6, 6):
            with Ts.sblock("B"):
                i, j = Ts.axis.remap("SS", [i0, i1])
                Ts.reads()
                Ts.writes(B[i, j])
                Ts.sblock_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
                B[i, j] = T.Select(i % 6 == 5 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 5 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 6 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 6 == 5, T.float32(1.5), T.Select(i % 6 == 4 and j % 6 == 4, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 3, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 2, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 4 and j % 6 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 6 == 5, T.float32(-2), T.Select(i % 6 == 3 and j % 6 == 4, T.float32(-0.5), T.Select(i % 6 == 3 and j % 6 == 3, T.float32(2), T.Select(i % 6 == 3 and j % 6 == 2, T.float32(2.5), T.Select(i % 6 == 3 and j % 6 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 6 == 0, T.float32(1.5), T.Select(i % 6 == 2 and j % 6 == 5, T.float32(-1.5), T.Select(i % 6 == 2 and j % 6 == 4, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 3, T.float32(-1), T.Select(i % 6 == 2 and j % 6 == 2, T.float32(0.5), T.Select(i % 6 == 2 and j % 6 == 1, T.float32(-2.5), T.Select(i % 6 == 2 and j % 6 == 0, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 5, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 4, T.float32(0.5), T.Select(i % 6 == 1 and j % 6 == 3, T.float32(-2), T.Select(i % 6 == 1 and j % 6 == 2, T.float32(-1), T.Select(i % 6 == 1 and j % 6 == 1, T.float32(1), T.Select(i % 6 == 1 and j % 6 == 0, T.float32(-1.5), T.Select(i % 6 == 0 and j % 6 == 5, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 4, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(6, 6, 196, 64, 6, 6):
            with Ts.sblock("data_pack"):
                eps, nu, p, ci, r_a, r_b = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                Ts.reads(input_tile[r_a, r_b, p, ci], B[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(eps, nu) : T.max(eps, nu) + 1])
                Ts.writes(data_pack[eps, nu, p, ci])
                Ts.sblock_attr({"auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_data_pack.cuda"})
                with Ts.init():
                    data_pack[eps, nu, p, ci] = T.float32(0)
                data_pack[eps, nu, p, ci] = data_pack[eps, nu, p, ci] + input_tile[r_a, r_b, p, ci] * B[r_a, eps] * B[r_b, nu]
        for i0, i1, i2, i3, i4 in T.grid(6, 6, 196, 64, 64):
            with Ts.sblock("bgemm"):
                eps, nu, p, co, ci = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, i4])
                Ts.reads(data_pack[eps, nu, p, ci], p1[eps, nu, co, ci])
                Ts.writes(bgemm[eps, nu, p, co])
                Ts.sblock_attr({"layout_free_placeholders":[]})
                with Ts.init():
                    bgemm[eps, nu, p, co] = T.float32(0)
                bgemm[eps, nu, p, co] = bgemm[eps, nu, p, co] + data_pack[eps, nu, p, ci] * p1[eps, nu, co, ci]
        for i0, i1 in T.grid(6, 4):
            with Ts.sblock("A"):
                i, j = Ts.axis.remap("SS", [i0, i1])
                Ts.reads()
                Ts.writes(A[i, j])
                Ts.sblock_attr({"const_matrix":True, "schedule_rule":"meta_schedule.compute_inline"})
                A[i, j] = T.Select(i % 6 == 5 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 5 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 5 and j % 4 == 0, T.float32(0), T.Select(i % 6 == 4 and j % 4 == 3, T.float32(-8), T.Select(i % 6 == 4 and j % 4 == 2, T.float32(4), T.Select(i % 6 == 4 and j % 4 == 1, T.float32(-2), T.Select(i % 6 == 4 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 3 and j % 4 == 3, T.float32(0.125), T.Select(i % 6 == 3 and j % 4 == 2, T.float32(0.25), T.Select(i % 6 == 3 and j % 4 == 1, T.float32(0.5), T.Select(i % 6 == 3 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 3, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 1, T.float32(1), T.Select(i % 6 == 2 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 3, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 2, T.float32(1), T.Select(i % 6 == 1 and j % 4 == 1, T.float32(-1), T.Select(i % 6 == 1 and j % 4 == 0, T.float32(1), T.Select(i % 6 == 0 and j % 4 == 3, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 2, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 1, T.float32(0), T.Select(i % 6 == 0 and j % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
        for i0, i1, i2, i3, i4, i5 in T.grid(4, 4, 196, 64, 6, 6):
            with Ts.sblock("inverse"):
                vh, vw, p, co, r_a, r_b = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                Ts.reads(bgemm[r_a, r_b, p, co], A[T.min(r_a, r_b) : T.max(r_a, r_b) + 1, T.min(vh, vw) : T.max(vh, vw) + 1])
                Ts.writes(inverse[vh, vw, p, co])
                Ts.sblock_attr({"auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_inverse.cuda"})
                with Ts.init():
                    inverse[vh, vw, p, co] = T.float32(0)
                inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co] * A[r_a, vh] * A[r_b, vw]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with Ts.sblock("conv2d_winograd"):
                n, h, w, co = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co])
                Ts.writes(conv2d_winograd[n, h, w, co])
                conv2d_winograd[n, h, w, co] = inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(conv2d_winograd[ax0, ax1, ax2, ax3], p2[ax0, 0, 0, ax3])
                Ts.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = conv2d_winograd[ax0, ax1, ax2, ax3] + p2[ax0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add[ax0, ax1, ax2, ax3], p3[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = T_add[ax0, ax1, ax2, ax3] + p3[ax0, ax1, ax2, ax3]
        for i0, i1, i2, i3 in T.grid(1, 56, 56, 64):
            with Ts.sblock("T_relu"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add_1[ax0, ax1, ax2, ax3])
                Ts.writes(T_relu[ax0, ax1, ax2, ax3])
                T_relu[ax0, ax1, ax2, ax3] = T.max(T_add_1[ax0, ax1, ax2, ax3], T.float32(0))

@tvm_tirx.script.ir_module
class Conv2dWinogradAddResidualRelu_scheduled:
    @Ts.function
    def main(p0: T.Tensor((1, 56, 56, 64), "float32"), p1: T.Tensor((6, 6, 64, 64), "float32"), p2: T.Tensor((1, 1, 1, 64), "float32"), p3: T.Tensor((1, 56, 56, 64), "float32"), T_relu: T.Tensor((1, 56, 56, 64), "float32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
        # body
        # with Ts.sblock("root")
        input_tile_local = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32", scope="local")
        data_pack = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
        bgemm = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32")
        inverse = Ts.sblock_alloc_tensor([4, 4, 196, 64], dtype="float32")
        bgemm_local = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32", scope="local")
        data_pack_shared = Ts.sblock_alloc_tensor([6, 6, 196, 64], dtype="float32", scope="shared")
        p1_shared = Ts.sblock_alloc_tensor([6, 6, 64, 64], dtype="float32", scope="shared")
        for i2_0_i3_0_i2_1_i3_1_fused_0 in T.parallel(98, annotations={"thread_binding": "blockIdx.x", "tirx.auto_unroll_max_step":1024, "tirx.unroll_explicit":1}):
            for i2_0_i3_0_i2_1_i3_1_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                for ax0, ax1, ax2, ax3 in T.grid(6, 6, 1, 1):
                    with Ts.sblock("input_tile"):
                        eps, nu = Ts.axis.remap("SS", [ax0, ax1])
                        p = Ts.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) // 896 * 14 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 112 // 8 + ax2)
                        ci = Ts.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 896 // 112 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8 + ax3)
                        Ts.reads(p0[p // 196, p % 196 // 14 * 4 + eps - 1, p % 14 * 4 + nu - 1, ci])
                        Ts.writes(input_tile_local[eps, nu, p, ci])
                        Ts.sblock_attr({"schedule_rule":"None"})
                        input_tile_local[eps, nu, p, ci] = T.if_then_else(1 <= p % 196 // 14 * 4 + eps and p % 196 // 14 * 4 + eps < 57 and 1 <= p % 14 * 4 + nu and p % 14 * 4 + nu < 57, p0[p // 196, p % 196 // 14 * 4 + eps - 1, p % 14 * 4 + nu - 1, ci], T.float32(0))
                for i0 in T.unroll(6):
                    for i1 in T.unroll(6):
                        with Ts.sblock("data_pack_init"):
                            eps, nu = Ts.axis.remap("SS", [i0, i1])
                            p = Ts.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) // 896 * 14 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 112 // 8)
                            ci = Ts.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 896 // 112 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8)
                            Ts.reads()
                            Ts.writes(data_pack[eps, nu, p, ci])
                            Ts.sblock_attr({"auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_data_pack.cuda"})
                            data_pack[eps, nu, p, ci] = T.float32(0)
                        for i4 in T.unroll(6):
                            for i5 in T.unroll(6):
                                with Ts.sblock("data_pack_update"):
                                    eps, nu = Ts.axis.remap("SS", [i0, i1])
                                    p = Ts.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) // 896 * 14 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 112 // 8)
                                    ci = Ts.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 896 // 112 * 8 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 128 + i2_0_i3_0_i2_1_i3_1_fused_1) % 8)
                                    r_a, r_b = Ts.axis.remap("RR", [i4, i5])
                                    Ts.reads(data_pack[eps, nu, p, ci], input_tile_local[r_a, r_b, p, ci])
                                    Ts.writes(data_pack[eps, nu, p, ci])
                                    Ts.sblock_attr({"auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_data_pack.cuda"})
                                    data_pack[eps, nu, p, ci] = data_pack[eps, nu, p, ci] + input_tile_local[r_a, r_b, p, ci] * T.Select(r_a % 6 == 5 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 5 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 5 and eps % 6 == 0, T.float32(0), T.Select(r_a % 6 == 4 and eps % 6 == 5, T.float32(1.5), T.Select(r_a % 6 == 4 and eps % 6 == 4, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 3, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 2, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 4 and eps % 6 == 0, T.float32(1), T.Select(r_a % 6 == 3 and eps % 6 == 5, T.float32(-2), T.Select(r_a % 6 == 3 and eps % 6 == 4, T.float32(-0.5), T.Select(r_a % 6 == 3 and eps % 6 == 3, T.float32(2), T.Select(r_a % 6 == 3 and eps % 6 == 2, T.float32(2.5), T.Select(r_a % 6 == 3 and eps % 6 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and eps % 6 == 0, T.float32(1.5), T.Select(r_a % 6 == 2 and eps % 6 == 5, T.float32(-1.5), T.Select(r_a % 6 == 2 and eps % 6 == 4, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 3, T.float32(-1), T.Select(r_a % 6 == 2 and eps % 6 == 2, T.float32(0.5), T.Select(r_a % 6 == 2 and eps % 6 == 1, T.float32(-2.5), T.Select(r_a % 6 == 2 and eps % 6 == 0, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 5, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 4, T.float32(0.5), T.Select(r_a % 6 == 1 and eps % 6 == 3, T.float32(-2), T.Select(r_a % 6 == 1 and eps % 6 == 2, T.float32(-1), T.Select(r_a % 6 == 1 and eps % 6 == 1, T.float32(1), T.Select(r_a % 6 == 1 and eps % 6 == 0, T.float32(-1.5), T.Select(r_a % 6 == 0 and eps % 6 == 5, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 4, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 3, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 2, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 1, T.float32(0), T.Select(r_a % 6 == 0 and eps % 6 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 5 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 5 and nu % 6 == 0, T.float32(0), T.Select(r_b % 6 == 4 and nu % 6 == 5, T.float32(1.5), T.Select(r_b % 6 == 4 and nu % 6 == 4, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 3, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 2, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 4 and nu % 6 == 0, T.float32(1), T.Select(r_b % 6 == 3 and nu % 6 == 5, T.float32(-2), T.Select(r_b % 6 == 3 and nu % 6 == 4, T.float32(-0.5), T.Select(r_b % 6 == 3 and nu % 6 == 3, T.float32(2), T.Select(r_b % 6 == 3 and nu % 6 == 2, T.float32(2.5), T.Select(r_b % 6 == 3 and nu % 6 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and nu % 6 == 0, T.float32(1.5), T.Select(r_b % 6 == 2 and nu % 6 == 5, T.float32(-1.5), T.Select(r_b % 6 == 2 and nu % 6 == 4, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 3, T.float32(-1), T.Select(r_b % 6 == 2 and nu % 6 == 2, T.float32(0.5), T.Select(r_b % 6 == 2 and nu % 6 == 1, T.float32(-2.5), T.Select(r_b % 6 == 2 and nu % 6 == 0, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 5, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 4, T.float32(0.5), T.Select(r_b % 6 == 1 and nu % 6 == 3, T.float32(-2), T.Select(r_b % 6 == 1 and nu % 6 == 2, T.float32(-1), T.Select(r_b % 6 == 1 and nu % 6 == 1, T.float32(1), T.Select(r_b % 6 == 1 and nu % 6 == 0, T.float32(-1.5), T.Select(r_b % 6 == 0 and nu % 6 == 5, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 4, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 3, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 2, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 1, T.float32(0), T.Select(r_b % 6 == 0 and nu % 6 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
        for i0_0_i1_0_i2_0_i3_0_fused in T.parallel(168, annotations={"thread_binding": "blockIdx.x", "tirx.auto_unroll_max_step":1024, "tirx.unroll_explicit":1}):
            for i0_1_i1_1_i2_1_i3_1_fused in T.parallel(4, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_i2_2_i3_2_fused in T.parallel(48, annotations={"thread_binding": "threadIdx.x"}):
                    for i0_3_init, i1_3_init, i2_3_init, i3_3_init, i0_4_init, i1_4_init, i2_4_init, i3_4_init in T.grid(1, 1, 14, 1, 1, 1, 1, 1):
                        with Ts.sblock("bgemm_init"):
                            eps = Ts.axis.spatial(6, i0_1_i1_1_i2_1_i3_1_fused // 2 * 3 + i0_2_i1_2_i2_2_i3_2_fused // 16 + i0_3_init + i0_4_init)
                            nu = Ts.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28 + i1_3_init + i1_4_init)
                            p = Ts.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 28 // 4 * 28 + i0_1_i1_1_i2_1_i3_1_fused % 2 * 14 + i2_3_init + i2_4_init)
                            co = Ts.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 4 * 16 + i0_2_i1_2_i2_2_i3_2_fused % 16 + i3_3_init + i3_4_init)
                            Ts.reads()
                            Ts.writes(bgemm_local[eps, nu, p, co])
                            Ts.sblock_attr({"layout_free_placeholders":[], "meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                            bgemm_local[eps, nu, p, co] = T.float32(0)
                    for i4_0 in T.serial(2):
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(28):
                            for ax0_ax1_ax2_ax3_fused_1 in T.parallel(48, annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(4):
                                    with Ts.sblock("data_pack_shared"):
                                        v0 = Ts.axis.spatial(6, (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) // 896)
                                        v1 = Ts.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28)
                                        v2 = Ts.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 28 // 4 * 28 + (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 896 // 32)
                                        v3 = Ts.axis.spatial(64, i4_0 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 32)
                                        Ts.reads(data_pack[v0, v1, v2, v3])
                                        Ts.writes(data_pack_shared[v0, v1, v2, v3])
                                        data_pack_shared[v0, v1, v2, v3] = data_pack[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(16):
                            for ax0_ax1_ax2_ax3_fused_1 in T.parallel(48, annotations={"thread_binding": "threadIdx.x"}):
                                for ax0_ax1_ax2_ax3_fused_2 in T.vectorized(4):
                                    with Ts.sblock("p1_shared"):
                                        v0 = Ts.axis.spatial(6, (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) // 512)
                                        v1 = Ts.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28)
                                        v2 = Ts.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 4 * 16 + (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 512 // 32)
                                        v3 = Ts.axis.spatial(64, i4_0 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 192 + ax0_ax1_ax2_ax3_fused_1 * 4 + ax0_ax1_ax2_ax3_fused_2) % 32)
                                        Ts.reads(p1[v0, v1, v2, v3])
                                        Ts.writes(p1_shared[v0, v1, v2, v3])
                                        p1_shared[v0, v1, v2, v3] = p1[v0, v1, v2, v3]
                        for i4_1, i0_3, i1_3, i2_3, i3_3, i4_2, i0_4, i1_4, i2_4, i3_4 in T.grid(2, 1, 1, 14, 1, 16, 1, 1, 1, 1):
                            with Ts.sblock("bgemm_update"):
                                eps = Ts.axis.spatial(6, i0_1_i1_1_i2_1_i3_1_fused // 2 * 3 + i0_2_i1_2_i2_2_i3_2_fused // 16 + i0_3 + i0_4)
                                nu = Ts.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28 + i1_3 + i1_4)
                                p = Ts.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 28 // 4 * 28 + i0_1_i1_1_i2_1_i3_1_fused % 2 * 14 + i2_3 + i2_4)
                                co = Ts.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 4 * 16 + i0_2_i1_2_i2_2_i3_2_fused % 16 + i3_3 + i3_4)
                                ci = Ts.axis.reduce(64, i4_0 * 32 + i4_1 * 16 + i4_2)
                                Ts.reads(bgemm_local[eps, nu, p, co], data_pack_shared[eps, nu, p, ci], p1_shared[eps, nu, co, ci])
                                Ts.writes(bgemm_local[eps, nu, p, co])
                                Ts.sblock_attr({"layout_free_placeholders":[], "meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32, "meta_schedule.tiling_structure":"SSSRRSRS"})
                                bgemm_local[eps, nu, p, co] = bgemm_local[eps, nu, p, co] + data_pack_shared[eps, nu, p, ci] * p1_shared[eps, nu, co, ci]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 1, 14, 1):
                        with Ts.sblock("bgemm_local"):
                            v0 = Ts.axis.spatial(6, i0_1_i1_1_i2_1_i3_1_fused // 2 * 3 + i0_2_i1_2_i2_2_i3_2_fused // 16 + ax0)
                            v1 = Ts.axis.spatial(6, i0_0_i1_0_i2_0_i3_0_fused // 28 + ax1)
                            v2 = Ts.axis.spatial(196, i0_0_i1_0_i2_0_i3_0_fused % 28 // 4 * 28 + i0_1_i1_1_i2_1_i3_1_fused % 2 * 14 + ax2)
                            v3 = Ts.axis.spatial(64, i0_0_i1_0_i2_0_i3_0_fused % 4 * 16 + i0_2_i1_2_i2_2_i3_2_fused % 16 + ax3)
                            Ts.reads(bgemm_local[v0, v1, v2, v3])
                            Ts.writes(bgemm[v0, v1, v2, v3])
                            bgemm[v0, v1, v2, v3] = bgemm_local[v0, v1, v2, v3]
        for i2_0_i3_0_i2_1_i3_1_fused_0 in T.parallel(25, annotations={"thread_binding": "blockIdx.x", "tirx.auto_unroll_max_step":1024, "tirx.unroll_explicit":1}):
            for i2_0_i3_0_i2_1_i3_1_fused_1 in T.parallel(512, annotations={"thread_binding": "threadIdx.x"}):
                for i0 in T.unroll(4):
                    for i1 in T.unroll(4):
                        with Ts.sblock("inverse_init"):
                            Ts.where(i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1 < 12544)
                            vh, vw = Ts.axis.remap("SS", [i0, i1])
                            p = Ts.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) // 448 * 7 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 224 // 32)
                            co = Ts.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 448 // 224 * 32 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 32)
                            Ts.reads()
                            Ts.writes(inverse[vh, vw, p, co])
                            Ts.sblock_attr({"auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_inverse.cuda"})
                            inverse[vh, vw, p, co] = T.float32(0)
                        for i4 in T.unroll(6):
                            for i5 in T.unroll(6):
                                with Ts.sblock("inverse_update"):
                                    Ts.where(i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1 < 12544)
                                    vh, vw = Ts.axis.remap("SS", [i0, i1])
                                    p = Ts.axis.spatial(196, (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) // 448 * 7 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 224 // 32)
                                    co = Ts.axis.spatial(64, (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 448 // 224 * 32 + (i2_0_i3_0_i2_1_i3_1_fused_0 * 512 + i2_0_i3_0_i2_1_i3_1_fused_1) % 32)
                                    r_a, r_b = Ts.axis.remap("RR", [i4, i5])
                                    Ts.reads(inverse[vh, vw, p, co], bgemm[r_a, r_b, p, co])
                                    Ts.writes(inverse[vh, vw, p, co])
                                    Ts.sblock_attr({"auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"], "schedule_rule":"meta_schedule.winograd_inverse.cuda"})
                                    inverse[vh, vw, p, co] = inverse[vh, vw, p, co] + bgemm[r_a, r_b, p, co] * T.Select(r_a % 6 == 5 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 5 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 5 and vh % 4 == 0, T.float32(0), T.Select(r_a % 6 == 4 and vh % 4 == 3, T.float32(-8), T.Select(r_a % 6 == 4 and vh % 4 == 2, T.float32(4), T.Select(r_a % 6 == 4 and vh % 4 == 1, T.float32(-2), T.Select(r_a % 6 == 4 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 3 and vh % 4 == 3, T.float32(0.125), T.Select(r_a % 6 == 3 and vh % 4 == 2, T.float32(0.25), T.Select(r_a % 6 == 3 and vh % 4 == 1, T.float32(0.5), T.Select(r_a % 6 == 3 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 3, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 1, T.float32(1), T.Select(r_a % 6 == 2 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 3, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 2, T.float32(1), T.Select(r_a % 6 == 1 and vh % 4 == 1, T.float32(-1), T.Select(r_a % 6 == 1 and vh % 4 == 0, T.float32(1), T.Select(r_a % 6 == 0 and vh % 4 == 3, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 2, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 1, T.float32(0), T.Select(r_a % 6 == 0 and vh % 4 == 0, T.float32(1), T.float32(0))))))))))))))))))))))))) * T.Select(r_b % 6 == 5 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 5 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 5 and vw % 4 == 0, T.float32(0), T.Select(r_b % 6 == 4 and vw % 4 == 3, T.float32(-8), T.Select(r_b % 6 == 4 and vw % 4 == 2, T.float32(4), T.Select(r_b % 6 == 4 and vw % 4 == 1, T.float32(-2), T.Select(r_b % 6 == 4 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 3 and vw % 4 == 3, T.float32(0.125), T.Select(r_b % 6 == 3 and vw % 4 == 2, T.float32(0.25), T.Select(r_b % 6 == 3 and vw % 4 == 1, T.float32(0.5), T.Select(r_b % 6 == 3 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 3, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 1, T.float32(1), T.Select(r_b % 6 == 2 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 3, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 2, T.float32(1), T.Select(r_b % 6 == 1 and vw % 4 == 1, T.float32(-1), T.Select(r_b % 6 == 1 and vw % 4 == 0, T.float32(1), T.Select(r_b % 6 == 0 and vw % 4 == 3, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 2, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 1, T.float32(0), T.Select(r_b % 6 == 0 and vw % 4 == 0, T.float32(1), T.float32(0)))))))))))))))))))))))))
        for i0_i1_i2_i3_fused_0 in T.parallel(1568, annotations={"thread_binding": "blockIdx.x", "tirx.auto_unroll_max_step":1024, "tirx.unroll_explicit":1}):
            for i0_i1_i2_i3_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("conv2d_winograd"):
                    n = Ts.axis.spatial(1, 0)
                    h = Ts.axis.spatial(56, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) // 3584)
                    w = Ts.axis.spatial(56, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) % 3584 // 64)
                    co = Ts.axis.spatial(64, (i0_i1_i2_i3_fused_0 * 128 + i0_i1_i2_i3_fused_1) % 64)
                    Ts.reads(inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co], p2[n, 0, 0, co], p3[n, h, w, co])
                    Ts.writes(T_relu[n, h, w, co])
                    T_relu[n, h, w, co] = T.max(inverse[h % 4, w % 4, n * 196 + h // 4 * 14 + w // 4, co] + p2[n, 0, 0, co] + p3[n, h, w, co], T.float32(0))

@tvm_tirx.script.ir_module
class Conv2dInt8_with_predicate:
    @Ts.function
    def main(p0: T.Tensor((16, 56, 56, 64), "int8"), p1: T.Tensor((256, 1, 1, 64), "int8"), p2: T.Tensor((1, 1, 1, 256), "int32"), p3: T.Tensor((1, 1, 1, 256), "int32"), p4: T.Tensor(256, "int32"), p5: T.Tensor(256, "int32"), p6: T.Tensor(256, "int32"), p7: T.Tensor((), "int32"), p8: T.Tensor(1, "int32"), compute: T.Tensor((16, 56, 56, 256), "int32")) -> None:
        # function attr dict
        T.func_attr({"tirx.noalias": True, "global_symbol": "main"})
        # body
        # with Ts.sblock("root")
        pad_temp = Ts.sblock_alloc_tensor([16, 56, 56, 64], dtype="int8")
        conv2d_nhwc = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_subtract = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_2 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_subtract_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 64):
            with Ts.sblock("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p0[i0_1, i1_1, i2_1, i3_1])
                Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with Ts.sblock("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                Ts.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                Ts.writes(conv2d_nhwc[nn, yy, xx, ff])
                with Ts.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], ty="int32") * T.cast(p1[ff, ry, rx, rc], ty="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_subtract"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                Ts.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute"):
                i0_2, i1_2, i2_2, i3_2 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2])
                Ts.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.Cast("int32", T.shift_right(T.shift_left(T.Cast("int64", T_add[i0_2, i1_2, i2_2, i3_2]) * T.Cast("int64", p4[i3_2]), p5[i3_2]), (31) + (p6[i3_2])))
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                Ts.reads(p7[()], compute_1[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = p7[()] + compute_1[ax0, ax1, ax2, ax3]
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_1"):
                i0_5, i1_5, i2_5, i3_5 = Ts.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                Ts.reads(T_add_1[i0_5, i1_5, i2_5, i3_5])
                Ts.writes(compute_2[i0_5, i1_5, i2_5, i3_5])
                compute_2[i0_5, i1_5, i2_5, i3_5] = T.max(T.min(T_add_1[i0_5, i1_5, i2_5, i3_5], 255), 0)
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_subtract_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                Ts.reads(compute_2[ax0, ax1, ax2, ax3], p8[0])
                Ts.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = compute_2[ax0, ax1, ax2, ax3] - p8[0]
        for i0_7, i1_7, i2_7, i3_7 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_2"):
                i0_8, i1_8, i2_8, i3_8 = Ts.axis.remap("SSSS", [i0_7, i1_7, i2_7, i3_7])
                Ts.reads(T_subtract_1[i0_8, i1_8, i2_8, i3_8])
                Ts.writes(compute[i0_8, i1_8, i2_8, i3_8])
                compute[i0_8, i1_8, i2_8, i3_8] = T.Cast("int32", T.shift_right(T.Cast("int64", T_subtract_1[i0_8, i1_8, i2_8, i3_8]) * T.Cast("int64", 1963325822), (31) - (1)))

@tvm_tirx.script.ir_module
class Conv2dInt8_with_predicate_target:
    @Ts.function
    def main(p0: T.Tensor((16, 56, 56, 64), "int8"), p1: T.Tensor((256, 1, 1, 64), "int8"), p2: T.Tensor((1, 1, 1, 256), "int32"), p3: T.Tensor((1, 1, 1, 256), "int32"), p4: T.Tensor(256, "int32"), p5: T.Tensor(256, "int32"), p6: T.Tensor(256, "int32"), p7: T.Tensor((), "int32"), p8: T.Tensor(1, "int32"), p9: T.Tensor((16, 56, 56, 256), "int32"), compute: T.Tensor((16, 56, 56, 256), "int32")) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        pad_temp = Ts.sblock_alloc_tensor([16, 56, 56, 64], dtype="int8")
        conv2d_nhwc = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_subtract = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_2 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_subtract_1 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_3 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        compute_4 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        T_add_2 = Ts.sblock_alloc_tensor([16, 56, 56, 256], dtype="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 64):
            with Ts.sblock("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(p0[i0_1, i1_1, i2_1, i3_1])
                Ts.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = p0[i0_1, i1_1, i2_1, i3_1]
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(16, 56, 56, 256, 1, 1, 64):
            with Ts.sblock("conv2d_nhwc"):
                nn, yy, xx, ff, ry, rx, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                Ts.reads(pad_temp[nn, yy + ry, xx + rx, rc], p1[ff, ry, rx, rc])
                Ts.writes(conv2d_nhwc[nn, yy, xx, ff])
                with Ts.init():
                    conv2d_nhwc[nn, yy, xx, ff] = 0
                conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + T.cast(pad_temp[nn, yy + ry, xx + rx, rc], ty="int32") * T.cast(p1[ff, ry, rx, rc], ty="int32")
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_subtract"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(conv2d_nhwc[ax0, ax1, ax2, ax3], p2[0, 0, 0, ax3])
                Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                T_subtract[ax0, ax1, ax2, ax3] = conv2d_nhwc[ax0, ax1, ax2, ax3] - p2[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_subtract[ax0, ax1, ax2, ax3], p3[0, 0, 0, ax3])
                Ts.writes(T_add[ax0, ax1, ax2, ax3])
                T_add[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] + p3[0, 0, 0, ax3]
        for i0, i1, i2, i3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute"):
                i0_2, i1_2, i2_2, i3_2 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(T_add[i0_2, i1_2, i2_2, i3_2], p4[i3_2], p5[i3_2], p6[i3_2])
                Ts.writes(compute_1[i0_2, i1_2, i2_2, i3_2])
                compute_1[i0_2, i1_2, i2_2, i3_2] = T.Cast("int32", T.shift_right(T.shift_left(T.Cast("int64", T_add[i0_2, i1_2, i2_2, i3_2]) * T.Cast("int64", p4[i3_2]), p5[i3_2]), (31) + (p6[i3_2])))
        for i0_3, i1_3, i2_3, i3_3 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3_3])
                Ts.reads(p7[()], compute_1[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_1[ax0, ax1, ax2, ax3])
                T_add_1[ax0, ax1, ax2, ax3] = p7[()] + compute_1[ax0, ax1, ax2, ax3]
        for i0_4, i1_4, i2_4, i3_4 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_1"):
                i0_5, i1_5, i2_5, i3_5 = Ts.axis.remap("SSSS", [i0_4, i1_4, i2_4, i3_4])
                Ts.reads(T_add_1[i0_5, i1_5, i2_5, i3_5])
                Ts.writes(compute_2[i0_5, i1_5, i2_5, i3_5])
                compute_2[i0_5, i1_5, i2_5, i3_5] = T.max(T.min(T_add_1[i0_5, i1_5, i2_5, i3_5], 255), 0)
        for i0_6, i1_6, i2_6, i3_6 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_subtract_1"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_6, i1_6, i2_6, i3_6])
                Ts.reads(compute_2[ax0, ax1, ax2, ax3], p8[0])
                Ts.writes(T_subtract_1[ax0, ax1, ax2, ax3])
                T_subtract_1[ax0, ax1, ax2, ax3] = compute_2[ax0, ax1, ax2, ax3] - p8[0]
        for i0_7, i1_7, i2_7, i3_7 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_2"):
                i0_8, i1_8, i2_8, i3_8 = Ts.axis.remap("SSSS", [i0_7, i1_7, i2_7, i3_7])
                Ts.reads(T_subtract_1[i0_8, i1_8, i2_8, i3_8])
                Ts.writes(compute_3[i0_8, i1_8, i2_8, i3_8])
                compute_3[i0_8, i1_8, i2_8, i3_8] = T.Cast("int32", T.shift_right(T.Cast("int64", T_subtract_1[i0_8, i1_8, i2_8, i3_8]) * T.Cast("int64", 1457846997), (31) - (0)))
        for i0_9, i1_9, i2_9, i3_9 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_3"):
                i0_10, i1_10, i2_10, i3_10 = Ts.axis.remap("SSSS", [i0_9, i1_9, i2_9, i3_9])
                Ts.reads(p9[i0_10, i1_10, i2_10, i3_10])
                Ts.writes(compute_4[i0_10, i1_10, i2_10, i3_10])
                compute_4[i0_10, i1_10, i2_10, i3_10] = T.Cast("int32", T.shift_right(T.Cast("int64", p9[i0_10, i1_10, i2_10, i3_10]) * T.Cast("int64", 2101000910), (31) - (0)))
        for i0_11, i1_11, i2_11, i3_11 in T.grid(16, 56, 56, 256):
            with Ts.sblock("T_add_2"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0_11, i1_11, i2_11, i3_11])
                Ts.reads(compute_3[ax0, ax1, ax2, ax3], compute_4[ax0, ax1, ax2, ax3])
                Ts.writes(T_add_2[ax0, ax1, ax2, ax3])
                T_add_2[ax0, ax1, ax2, ax3] = compute_3[ax0, ax1, ax2, ax3] + compute_4[ax0, ax1, ax2, ax3]
        for i0_12, i1_12, i2_12, i3_12 in T.grid(16, 56, 56, 256):
            with Ts.sblock("compute_4"):
                i0_13, i1_13, i2_13, i3_13 = Ts.axis.remap("SSSS", [i0_12, i1_12, i2_12, i3_12])
                Ts.reads(T_add_2[i0_13, i1_13, i2_13, i3_13])
                Ts.writes(compute[i0_13, i1_13, i2_13, i3_13])
                compute[i0_13, i1_13, i2_13, i3_13] = T.max(T.min(T_add_2[i0_13, i1_13, i2_13, i3_13], 255), 0)

@tvm_tirx.script.ir_module
class Conv2dInt8_with_predicate_scheduled:
    @Ts.function
    def main(p0: T.Tensor((16, 56, 56, 64), "int8"), p1: T.Tensor((256, 1, 1, 64), "int8"), p2: T.Tensor((1, 1, 1, 256), "int32"), p3: T.Tensor((1, 1, 1, 256), "int32"), p4: T.Tensor((256,), "int32"), p5: T.Tensor((256,), "int32"), p6: T.Tensor((256,), "int32"), p7: T.Tensor((), "int32"), p8: T.Tensor((1,), "int32"), p9: T.Tensor((16, 56, 56, 256), "int32"), compute: T.Tensor((16, 56, 56, 256), "int32")):
        T.func_attr({"tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            Ts.sblock_attr({"meta_schedule.unroll_explicit": 1024})
            conv2d_nhwc_reindex_shared = Ts.sblock_alloc_tensor((50176, 256), "int32", scope="shared")
            conv2d_nhwc_reindex_shared_wmma_accumulator = Ts.sblock_alloc_tensor((50176, 256), "int32", scope="wmma.accumulator")
            pad_temp_reindex_shared = Ts.sblock_alloc_tensor((50176, 64), "int8", scope="shared")
            p1_reindex_shared = Ts.sblock_alloc_tensor((1, 1, 256, 64), "int8", scope="shared")
            pad_temp_reindex_shared_wmma_matrix_a = Ts.sblock_alloc_tensor((50176, 64), "int8", scope="wmma.matrix_a")
            p1_reindex_shared_wmma_matrix_b = Ts.sblock_alloc_tensor((1, 1, 256, 64), "int8", scope="wmma.matrix_b")
            for ax2_0_0_ax3_0_0_fused in T.parallel(32, annotations={"thread_binding": "blockIdx.y"}):
                for ax2_0_1_ax3_0_1_fused in T.parallel(196, annotations={"thread_binding": "blockIdx.x"}):
                    for ax2_0_2_ax3_0_2_fused in T.parallel(4, annotations={"thread_binding": "threadIdx.y"}):
                        for ax0_0, ax1_0, ax4_0_0 in T.grid(1, 1, 2):
                            for ax0_ax1_fused in range(1024):
                                with Ts.sblock("pad_temp_reindex_shared"):
                                    v0 = Ts.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0_ax1_fused // 32)
                                    v1 = Ts.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_fused % 32)
                                    Ts.reads(p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                                    Ts.writes(pad_temp_reindex_shared[v0, v1])
                                    Ts.sblock_attr({"tensor_dim_align": [[0, 0, 32, 16]], "meta_schedule.cooperative_fetch": 4})
                                    pad_temp_reindex_shared[v0, v1] = p0[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1]
                            for ax0_ax1_ax2_ax3_fused in range(2048):
                                with Ts.sblock("p1_reindex_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(1, 0)
                                    v2 = Ts.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + ax0_ax1_ax2_ax3_fused // 32)
                                    v3 = Ts.axis.spatial(64, ax4_0_0 * 32 + ax0_ax1_ax2_ax3_fused % 32)
                                    Ts.reads(p1[v2, v0, v1, v3])
                                    Ts.writes(p1_reindex_shared[v0, v1, v2, v3])
                                    Ts.sblock_attr({"tensor_dim_align": [[0, 2, 32, 16]], "meta_schedule.cooperative_fetch": 3})
                                    p1_reindex_shared[v0, v1, v2, v3] = p1[v2, v0, v1, v3]
                            for ax0_1, ax1_1, ax4_0_1 in T.grid(1, 1, 2):
                                for ax0_0_1, ax1_0_1 in T.grid(1, 1):
                                    with Ts.sblock("pad_temp_reindex_shared_wmma.matrix_a_o"):
                                        v0_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax0_0_1)
                                        v1_o = Ts.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1 + ax1_0_1)
                                        Ts.reads(pad_temp_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        Ts.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_s8_a_shared"})
                                        for ax0_1_1, ax1_1_1 in T.grid(16, 16):
                                            with Ts.sblock("pad_temp_reindex_shared_wmma.matrix_a"):
                                                v0_i, v1_i = Ts.axis.remap("SS", [ax0_1_1, ax1_1_1])
                                                Ts.reads(pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                Ts.writes(pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                                pad_temp_reindex_shared_wmma_matrix_a[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = pad_temp_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                                for ax0, ax1, ax2_0, ax3_0 in T.grid(1, 1, 2, 1):
                                    with Ts.sblock("p1_reindex_shared_wmma.matrix_b_o"):
                                        v0_o, v1_o = Ts.axis.remap("SS", [ax0, ax1])
                                        v2_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax2_0)
                                        v3_o = Ts.axis.spatial(4, ax4_0_0 * 2 + ax4_0_1 + ax3_0)
                                        Ts.reads(p1_reindex_shared[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        Ts.writes(p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize": "wmma_load_16x16x16_s8_b_trans_shared"})
                                        for ax2_1, ax3_1 in T.grid(16, 16):
                                            with Ts.sblock("p1_reindex_shared_wmma.matrix_b"):
                                                v2_i, v3_i = Ts.axis.remap("SS", [ax2_1, ax3_1])
                                                Ts.reads(p1_reindex_shared[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                Ts.writes(p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i] = p1_reindex_shared[v0_o, v1_o, v2_o * 16 + v2_i, v3_o * 16 + v3_i]
                                for ax2_0_3, ax3_0_3, ax0_2, ax1_2, ax4_0_2, ax2_0_4, ax3_0_4 in T.grid(1, 1, 1, 1, 1, 1, 2):
                                    with Ts.sblock("conv2d_nhwc_o"):
                                        v0_o = Ts.axis.spatial(1, ax0_0 + ax0_1 + ax0_2)
                                        v1_o = Ts.axis.spatial(1, ax1_0 + ax1_1 + ax1_2)
                                        v2_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax2_0_3 + ax2_0_4)
                                        v3_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax3_0_3 * 2 + ax3_0_4)
                                        v4_o = Ts.axis.reduce(4, ax4_0_0 * 2 + ax4_0_1 + ax4_0_2)
                                        Ts.reads(pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16:v2_o * 16 + 16, v4_o * 16:v4_o * 16 + 16], p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v3_o * 16:v3_o * 16 + 16, v4_o * 16:v4_o * 16 + 16])
                                        Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16:v2_o * 16 + 16, v3_o * 16:v3_o * 16 + 16])
                                        Ts.sblock_attr({"meta_schedule.auto_tensorize": "wmma_sync_16x16x16_s8s8s32_trans", "meta_schedule.auto_tensorize_init": "wmma_fill_16x16x16_s32", "meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "warp_execution": 1})
                                        with Ts.init():
                                            for ax2_1, ax3_1 in T.grid(16, 16):
                                                with Ts.sblock("conv2d_nhwc_init"):
                                                    v2_i_init, v3_i_init = Ts.axis.remap("SS", [ax2_1, ax3_1])
                                                    Ts.reads()
                                                    Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init])
                                                    conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i_init, v3_o * 16 + v3_i_init] = 0
                                        for ax2_1, ax3_1, ax4_1 in T.grid(16, 16, 16):
                                            with Ts.sblock("conv2d_nhwc"):
                                                v2_i, v3_i, v4_i = Ts.axis.remap("SSR", [ax2_1, ax3_1, ax4_1])
                                                Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i], pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i], p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v3_o * 16 + v3_i, v4_o * 16 + v4_i])
                                                Ts.writes(conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i])
                                                Ts.sblock_attr({"meta_schedule.tiling_structure": "SSSRRSRS"})
                                                conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v2_o * 16 + v2_i, v3_o * 16 + v3_i] + T.Cast("int32", pad_temp_reindex_shared_wmma_matrix_a[v2_o * 16 + v2_i, v4_o * 16 + v4_i]) * T.Cast("int32", p1_reindex_shared_wmma_matrix_b[v0_o, v1_o, v3_o * 16 + v3_i, v4_o * 16 + v4_i])
                        for ax0_0, ax1_0 in T.grid(1, 2):
                            with Ts.sblock("conv2d_nhwc_reindex_shared_wmma.accumulator_o"):
                                v0_o = Ts.axis.spatial(3136, ax2_0_0_ax3_0_0_fused // 4 * 392 + ax2_0_1_ax3_0_1_fused * 2 + ax2_0_2_ax3_0_2_fused // 2 + ax0_0)
                                v1_o = Ts.axis.spatial(16, ax2_0_0_ax3_0_0_fused % 4 * 4 + ax2_0_2_ax3_0_2_fused % 2 * 2 + ax1_0)
                                Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                Ts.writes(conv2d_nhwc_reindex_shared[v0_o * 16:v0_o * 16 + 16, v1_o * 16:v1_o * 16 + 16])
                                Ts.sblock_attr({"meta_schedule.auto_tensorize": "wmma_store_16x16x16_s32_shared"})
                                for ax0_1, ax1_1 in T.grid(16, 16):
                                    with Ts.sblock("conv2d_nhwc_reindex_shared_wmma.accumulator"):
                                        v0_i, v1_i = Ts.axis.remap("SS", [ax0_1, ax1_1])
                                        Ts.reads(conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        Ts.writes(conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i])
                                        conv2d_nhwc_reindex_shared[v0_o * 16 + v0_i, v1_o * 16 + v1_i] = conv2d_nhwc_reindex_shared_wmma_accumulator[v0_o * 16 + v0_i, v1_o * 16 + v1_i]
                    for ax0, ax1_0, ax1_1, ax1_2, ax1_3 in T.grid(32, 1, 4, 32, 2):
                        with Ts.sblock("conv2d_nhwc_reindex_shared"):
                            v0 = Ts.axis.spatial(50176, ax2_0_0_ax3_0_0_fused // 4 * 6272 + ax2_0_1_ax3_0_1_fused * 32 + ax0)
                            v1 = Ts.axis.spatial(256, ax2_0_0_ax3_0_0_fused % 4 * 64 + (ax1_0 * 256 + ax1_1 * 64 + ax1_2 * 2 + ax1_3))
                            Ts.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 2 + ax1_3 < 64)
                            Ts.reads(p7[()], conv2d_nhwc_reindex_shared[v0, v1], p2[0, 0, 0, v1], p3[0, 0, 0, v1], p4[v1], p5[v1], p6[v1], p8[0], p9[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            Ts.writes(compute[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1])
                            compute[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1] = T.max(T.min(T.Cast("int32", T.shift_right(T.Cast("int64", T.max(T.min(p7[()] + T.Cast("int32", T.shift_right(T.shift_left(T.Cast("int64", conv2d_nhwc_reindex_shared[v0, v1] - p2[0, 0, 0, v1] + p3[0, 0, 0, v1]) * T.Cast("int64", p4[v1]), p5[v1]), (31) + (p6[v1]))), 255), 0) - p8[0]) * T.Cast("int64", 1457846997), (31) - (0))) + T.Cast("int32", T.shift_right(T.Cast("int64", p9[v0 // 3136, v0 % 3136 // 56, v0 % 56, v1]) * T.Cast("int64", 2101000910), (31) - (0))), 255), 0)

# fmt: on
def verify(anchor_mod, anchor_trace_fun, target_mod, target, ref):
    anchor_sch = Schedule(anchor_mod)
    anchor_trace_fun(anchor_sch)
    anchor_trace = anchor_sch.trace

    sch = Schedule(target_mod)

    ms.trace_apply.schedule_using_anchor_trace(sch, anchor_trace, Target(target))

    tvm_tirx.ir.assert_structural_equal(ref, sch.mod)


def test_dense_add_cpu():
    def apply_anchor_trace(sch: Schedule) -> None:
        b0 = sch.get_sblock(name="T_matmul_NT", func_name="main")
        b1 = sch.get_sblock(name="root", func_name="main")
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
        l2, l3, l4 = sch.get_loops(block=b0)
        v5, v6, v7, v8 = sch.sample_perfect_tile(
            loop=l2, n=4, max_innermost_factor=64, decision=[2, 8, 4, 2]
        )
        l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8], preserve_unit_iters=True)
        v13, v14, v15, v16 = sch.sample_perfect_tile(
            loop=l3, n=4, max_innermost_factor=64, decision=[2, 1, 1, 64]
        )
        l17, l18, l19, l20 = sch.split(
            loop=l3, factors=[v13, v14, v15, v16], preserve_unit_iters=True
        )
        v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64, decision=[128, 1])
        l23, l24 = sch.split(loop=l4, factors=[v21, v22], preserve_unit_iters=True)
        sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
        b25 = sch.cache_write(block=b0, write_tensor_index=0, storage_scope="global")
        sch.reverse_compute_at(block=b25, loop=l17, preserve_unit_loops=True, index=-1)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=160)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
        v26 = sch.sample_categorical(
            candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0
        )
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)
        sch.enter_postproc()
        b27 = sch.get_sblock(name="root", func_name="main")
        sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.parallel")
        sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.vectorize")
        sch.unannotate(block_or_loop=b27, ann_key="meta_schedule.unroll_explicit")
        b28, b29 = sch.get_child_blocks(b27)
        l30, l31, l32, l33, l34, l35, l36, l37, l38, l39 = sch.get_loops(block=b28)
        l40 = sch.fuse(l30, l31, preserve_unit_iters=True)
        sch.parallel(loop=l40)
        l41 = sch.fuse(l39, preserve_unit_iters=True)
        sch.vectorize(loop=l41)
        l42, l43, l44 = sch.get_loops(block=b29)
        l45 = sch.fuse(l42, preserve_unit_iters=True)
        sch.parallel(loop=l45)
        l46 = sch.fuse(l44, preserve_unit_iters=True)
        sch.vectorize(loop=l46)
        b47 = sch.get_sblock(name="T_matmul_NT", func_name="main")
        l48, l49, l50, l51, l52, l53, l54, l55, l56 = sch.get_loops(block=b47)
        b57 = sch.decompose_reduction(block=b47, loop=l51)
        b58 = sch.get_sblock(name="T_matmul_NT_update", func_name="main")
        b59 = sch.cache_read(block=b58, read_tensor_index=2, storage_scope="global")
        sch.transform_layout(
            block=b58,
            tensor=("read", 2),
            index_map=tvm_tirx.tirx.IndexMap.from_func(
                lambda i0, i1: (
                    floordiv(i0, 64),
                    i1,
                    floormod(i0, 64),
                ),
                inverse_index_map=lambda i0, i1, i2: (
                    ((i0 * 64) + i2),
                    i1,
                ),
                index_dtype="int32",
            ),
            pad_value=None,
        )
        sch.annotate(
            block_or_loop=b59, ann_key="meta_schedule.layout_rewrite_preproc", ann_val=True
        )

    verify(Dense, apply_anchor_trace, DenseAdd, "llvm", DenseAdd_scheduled_cpu)


def test_dense_add_cpu_no_write_cache():
    def apply_trace(sch):
        b0 = sch.get_sblock(name="T_matmul_NT", func_name="main")
        b1 = sch.get_sblock(name="root", func_name="main")
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
        l2, l3, l4 = sch.get_loops(block=b0)
        v5, v6, v7, v8 = sch.sample_perfect_tile(
            loop=l2, n=4, max_innermost_factor=64, decision=[4, 4, 4, 2]
        )
        l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8], preserve_unit_iters=True)
        v13, v14, v15, v16 = sch.sample_perfect_tile(
            loop=l3, n=4, max_innermost_factor=64, decision=[1, 1, 4, 32]
        )
        l17, l18, l19, l20 = sch.split(
            loop=l3, factors=[v13, v14, v15, v16], preserve_unit_iters=True
        )
        v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64, decision=[8, 16])
        l23, l24 = sch.split(loop=l4, factors=[v21, v22], preserve_unit_iters=True)
        sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=160)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=64)
        v25 = sch.sample_categorical(
            candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1
        )
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v25)
        sch.enter_postproc()
        b26 = sch.get_sblock(name="root", func_name="main")
        sch.unannotate(block_or_loop=b26, ann_key="meta_schedule.parallel")
        sch.unannotate(block_or_loop=b26, ann_key="meta_schedule.vectorize")
        sch.unannotate(block_or_loop=b26, ann_key="meta_schedule.unroll_explicit")
        (b27,) = sch.get_child_blocks(b26)
        l28, l29, l30, l31, l32, l33, l34, l35, l36, l37 = sch.get_loops(block=b27)
        l38 = sch.fuse(l28, l29, l30, l31, preserve_unit_iters=True)
        sch.parallel(loop=l38)
        l39 = sch.fuse(l37, preserve_unit_iters=True)
        sch.vectorize(loop=l39)
        sch.annotate(block_or_loop=l38, ann_key="tirx.auto_unroll_max_step", ann_val=16)
        sch.annotate(block_or_loop=l38, ann_key="tirx.unroll_explicit", ann_val=1)
        b40 = sch.get_sblock(name="T_matmul_NT", func_name="main")
        l41, l42, l43, l44, l45, l46, l47 = sch.get_loops(block=b40)
        b48 = sch.decompose_reduction(block=b40, loop=l42)
        b49 = sch.get_sblock(name="T_matmul_NT_update", func_name="main")
        b50 = sch.cache_read(block=b49, read_tensor_index=2, storage_scope="global")
        sch.transform_layout(
            block=b49,
            tensor=("read", 2),
            index_map=tvm_tirx.tirx.IndexMap.from_func(
                lambda i0, i1: (
                    floordiv(i1, 16),
                    floordiv(i0, 32),
                    floormod(i1, 16),
                    floormod(i0, 32),
                ),
                inverse_index_map=lambda i0, i1, i2, i3: (
                    ((i1 * 32) + i3),
                    ((i0 * 16) + i2),
                ),
                index_dtype="int32",
            ),
            pad_value=None,
        )
        sch.annotate(
            block_or_loop=b50, ann_key="meta_schedule.layout_rewrite_preproc", ann_val=True
        )

    verify(Dense, apply_trace, DenseAdd, "llvm", DenseAdd_cpu_no_write_cache)


def test_dense_add_gpu():
    def apply_anchor_trace(sch: Schedule) -> None:
        b0 = sch.get_sblock(name="T_matmul_NT", func_name="main")
        b1 = sch.get_sblock(name="root", func_name="main")
        sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        l2, l3, l4 = sch.get_loops(block=b0)
        v5, v6, v7, v8, v9 = sch.sample_perfect_tile(
            loop=l2, n=5, max_innermost_factor=64, decision=[8, 1, 16, 1, 1]
        )
        l10, l11, l12, l13, l14 = sch.split(
            loop=l2, factors=[v5, v6, v7, v8, v9], preserve_unit_iters=True
        )
        v15, v16, v17, v18, v19 = sch.sample_perfect_tile(
            loop=l3, n=5, max_innermost_factor=64, decision=[4, 1, 8, 4, 1]
        )
        l20, l21, l22, l23, l24 = sch.split(
            loop=l3, factors=[v15, v16, v17, v18, v19], preserve_unit_iters=True
        )
        v25, v26, v27 = sch.sample_perfect_tile(
            loop=l4, n=3, max_innermost_factor=64, decision=[32, 1, 4]
        )
        l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27], preserve_unit_iters=True)
        sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
        l31 = sch.fuse(l10, l20, preserve_unit_iters=True)
        sch.bind(loop=l31, thread_axis="blockIdx.x")
        l32 = sch.fuse(l11, l21, preserve_unit_iters=True)
        sch.bind(loop=l32, thread_axis="vthread.x")
        l33 = sch.fuse(l12, l22, preserve_unit_iters=True)
        sch.bind(loop=l33, thread_axis="threadIdx.x")
        sch.annotate(
            block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=16
        )
        sch.annotate(
            block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=256
        )
        b34 = sch.cache_write(block=b0, write_tensor_index=0, storage_scope="local")
        sch.reverse_compute_at(block=b34, loop=l33, preserve_unit_loops=True, index=-1)
        b35 = sch.cache_read(
            block=b0, read_tensor_index=0, storage_scope="shared", consumer_blocks=[b0]
        )
        sch.compute_at(block=b35, loop=l28, preserve_unit_loops=True, index=-1)
        l36, l37, l38, l39, l40, l41 = sch.get_loops(block=b35)
        l42 = sch.fuse(l40, l41, preserve_unit_iters=True)
        v43 = sch.sample_categorical(
            candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1
        )
        sch.annotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
        b44 = sch.cache_read(
            block=b0, read_tensor_index=1, storage_scope="shared", consumer_blocks=[b0]
        )
        sch.compute_at(block=b44, loop=l28, preserve_unit_loops=True, index=-1)
        l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b44)
        l51 = sch.fuse(l49, l50, preserve_unit_iters=True)
        v52 = sch.sample_categorical(
            candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3
        )
        sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v52)
        v53 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024],
            probs=[
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
            ],
            decision=2,
        )
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v53)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch")
        l54, l55, l56, l57, l58 = sch.get_loops(block=b35)
        l59, l60, l61 = sch.split(loop=l58, factors=[None, 128, 2], preserve_unit_iters=True)
        sch.vectorize(loop=l61)
        sch.bind(loop=l60, thread_axis="threadIdx.x")
        sch.unannotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch")
        l62, l63, l64, l65, l66 = sch.get_loops(block=b44)
        l67, l68, l69 = sch.split(loop=l66, factors=[None, 128, 4], preserve_unit_iters=True)
        sch.vectorize(loop=l69)
        sch.bind(loop=l68, thread_axis="threadIdx.x")
        b70 = sch.get_sblock(name="root", func_name="main")
        sch.unannotate(block_or_loop=b70, ann_key="meta_schedule.unroll_explicit")
        b71, b72, b73, b74 = sch.get_child_blocks(b70)
        l75, l76, l77, l78, l79, l80, l81 = sch.get_loops(block=b71)
        sch.annotate(block_or_loop=l75, ann_key="tirx.auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l75, ann_key="tirx.unroll_explicit", ann_val=1)
        l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b72)
        sch.annotate(block_or_loop=l82, ann_key="tirx.auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l82, ann_key="tirx.unroll_explicit", ann_val=1)
        l89, l90, l91, l92, l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b73)
        sch.annotate(block_or_loop=l89, ann_key="tirx.auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l89, ann_key="tirx.unroll_explicit", ann_val=1)
        l99, l100, l101, l102, l103 = sch.get_loops(block=b74)
        sch.annotate(block_or_loop=l99, ann_key="tirx.auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l99, ann_key="tirx.unroll_explicit", ann_val=1)
        b104 = sch.get_sblock(name="T_matmul_NT", func_name="main")
        l105, l106, l107, l108, l109, l110, l111, l112, l113, l114 = sch.get_loops(block=b104)
        b115 = sch.decompose_reduction(block=b104, loop=l108)

    verify(Dense, apply_anchor_trace, DenseAdd, "cuda", DenseAdd_scheduled_gpu)


def test_conv2d_int8_vnni():
    def apply_trace(sch):
        b0 = sch.get_sblock(name="compile_engine_const", func_name="main")
        b1 = sch.get_sblock(name="conv2d_NCHWc_int8", func_name="main")
        b2 = sch.get_sblock(name="T_add", func_name="main")
        b3 = sch.get_sblock(name="T_cast", func_name="main")
        b4 = sch.get_sblock(name="T_multiply", func_name="main")
        b5 = sch.get_sblock(name="compile_engine_const_1", func_name="main")
        b6 = sch.get_sblock(name="T_add_1", func_name="main")
        b7 = sch.get_sblock(name="T_floor", func_name="main")
        b8 = sch.get_sblock(name="T_cast_1", func_name="main")
        b9 = sch.get_sblock(name="compute", func_name="main")
        b10 = sch.get_sblock(name="T_cast_2", func_name="main")
        b11 = sch.get_sblock(name="T_cast_3", func_name="main")
        b12 = sch.get_sblock(name="T_subtract", func_name="main")
        b13 = sch.get_sblock(name="T_multiply_1", func_name="main")
        b14 = sch.get_sblock(name="compile_engine_const_2", func_name="main")
        b15 = sch.get_sblock(name="T_add_2", func_name="main")
        b16 = sch.get_sblock(name="T_floor_1", func_name="main")
        b17 = sch.get_sblock(name="T_cast_4", func_name="main")
        b18 = sch.get_sblock(name="T_add_3", func_name="main")
        b19 = sch.get_sblock(name="compute_1", func_name="main")
        b20 = sch.get_sblock(name="T_cast_5", func_name="main")
        b21 = sch.get_sblock(name="root", func_name="main")
        sch.compute_inline(block=b20)
        sch.compute_inline(block=b19)
        sch.compute_inline(block=b18)
        sch.compute_inline(block=b17)
        sch.compute_inline(block=b16)
        sch.compute_inline(block=b15)
        sch.compute_inline(block=b14)
        sch.compute_inline(block=b13)
        sch.compute_inline(block=b12)
        sch.compute_inline(block=b11)
        sch.compute_inline(block=b10)
        sch.compute_inline(block=b9)
        sch.compute_inline(block=b8)
        sch.compute_inline(block=b7)
        sch.compute_inline(block=b6)
        sch.compute_inline(block=b5)
        sch.compute_inline(block=b4)
        sch.compute_inline(block=b3)
        sch.compute_inline(block=b2)
        sch.compute_inline(block=b0)
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")
        l22, l23, l24, l25, l26, l27, l28, l29, l30, l31 = sch.get_loops(block=b1)
        l32, l33 = sch.split(loop=l31, factors=[None, 4], preserve_unit_iters=True)
        l34, l35 = sch.split(loop=l26, factors=[None, 16], preserve_unit_iters=True)
        l36, l37, l38, l39, l40, l41, l42, l43, l44, l45, l46, l47 = sch.get_loops(block=b1)
        sch.reorder(l42, l43, l44, l45, l46, l35, l33)
        b48 = sch.blockize(target=l35)
        sch.annotate(block_or_loop=b48, ann_key="meta_schedule.auto_tensorize", ann_val=VNNI_INTRIN)
        l49, l50, l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b48)
        v59, v60, v61, v62 = sch.sample_perfect_tile(
            loop=l49, n=4, max_innermost_factor=64, decision=[1, 1, 1, 1]
        )
        l63, l64, l65, l66 = sch.split(
            loop=l49, factors=[v59, v60, v61, v62], preserve_unit_iters=True
        )
        v67, v68, v69, v70 = sch.sample_perfect_tile(
            loop=l50, n=4, max_innermost_factor=64, decision=[4, 32, 1, 1]
        )
        l71, l72, l73, l74 = sch.split(
            loop=l50, factors=[v67, v68, v69, v70], preserve_unit_iters=True
        )
        v75, v76, v77, v78 = sch.sample_perfect_tile(
            loop=l51, n=4, max_innermost_factor=64, decision=[1, 7, 1, 1]
        )
        l79, l80, l81, l82 = sch.split(
            loop=l51, factors=[v75, v76, v77, v78], preserve_unit_iters=True
        )
        v83, v84, v85, v86 = sch.sample_perfect_tile(
            loop=l52, n=4, max_innermost_factor=64, decision=[1, 1, 1, 7]
        )
        l87, l88, l89, l90 = sch.split(
            loop=l52, factors=[v83, v84, v85, v86], preserve_unit_iters=True
        )
        v91, v92, v93, v94 = sch.sample_perfect_tile(
            loop=l53, n=4, max_innermost_factor=64, decision=[1, 1, 1, 1]
        )
        l95, l96, l97, l98 = sch.split(
            loop=l53, factors=[v91, v92, v93, v94], preserve_unit_iters=True
        )
        v99, v100 = sch.sample_perfect_tile(loop=l54, n=2, max_innermost_factor=64, decision=[1, 1])
        l101, l102 = sch.split(loop=l54, factors=[v99, v100], preserve_unit_iters=True)
        v103, v104 = sch.sample_perfect_tile(
            loop=l55, n=2, max_innermost_factor=64, decision=[1, 1]
        )
        l105, l106 = sch.split(loop=l55, factors=[v103, v104], preserve_unit_iters=True)
        v107, v108 = sch.sample_perfect_tile(
            loop=l56, n=2, max_innermost_factor=64, decision=[4, 8]
        )
        l109, l110 = sch.split(loop=l56, factors=[v107, v108], preserve_unit_iters=True)
        v111, v112 = sch.sample_perfect_tile(
            loop=l57, n=2, max_innermost_factor=64, decision=[4, 1]
        )
        l113, l114 = sch.split(loop=l57, factors=[v111, v112], preserve_unit_iters=True)
        v115, v116 = sch.sample_perfect_tile(
            loop=l58, n=2, max_innermost_factor=64, decision=[1, 1]
        )
        l117, l118 = sch.split(loop=l58, factors=[v115, v116], preserve_unit_iters=True)
        sch.reorder(
            l63,
            l71,
            l79,
            l87,
            l95,
            l64,
            l72,
            l80,
            l88,
            l96,
            l101,
            l105,
            l109,
            l113,
            l117,
            l65,
            l73,
            l81,
            l89,
            l97,
            l102,
            l106,
            l110,
            l114,
            l118,
            l66,
            l74,
            l82,
            l90,
            l98,
        )
        (b119,) = sch.get_consumers(block=b48)
        sch.reverse_compute_at(block=b119, loop=l96, preserve_unit_loops=True, index=-1)
        sch.annotate(block_or_loop=b21, ann_key="meta_schedule.parallel", ann_val=96)
        sch.annotate(block_or_loop=b21, ann_key="meta_schedule.vectorize", ann_val=64)
        v120 = sch.sample_categorical(
            candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=2
        )
        sch.annotate(block_or_loop=b21, ann_key="meta_schedule.unroll_explicit", ann_val=v120)
        sch.enter_postproc()
        b121 = sch.get_sblock(name="root", func_name="main")
        sch.unannotate(block_or_loop=b121, ann_key="meta_schedule.parallel")
        sch.unannotate(block_or_loop=b121, ann_key="meta_schedule.vectorize")
        sch.unannotate(block_or_loop=b121, ann_key="meta_schedule.unroll_explicit")
        b122, b123 = sch.get_child_blocks(b121)
        (
            l124,
            l125,
            l126,
            l127,
            l128,
            l129,
            l130,
            l131,
            l132,
            l133,
            l134,
            l135,
            l136,
            l137,
            l138,
            l139,
            l140,
            l141,
            l142,
            l143,
            l144,
            l145,
            l146,
            l147,
            l148,
            l149,
            l150,
            l151,
            l152,
            l153,
        ) = sch.get_loops(block=b122)
        l154 = sch.fuse(l124, l125, l126, l127, l128, l129, l130, preserve_unit_iters=True)
        sch.parallel(loop=l154)
        sch.annotate(block_or_loop=l154, ann_key="tirx.auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l154, ann_key="tirx.unroll_explicit", ann_val=1)
        l155, l156, l157, l158, l159, l160, l161, l162, l163 = sch.get_loops(block=b123)
        l164 = sch.fuse(l163, preserve_unit_iters=True)
        sch.vectorize(loop=l164)
        sch.annotate(block_or_loop=l155, ann_key="tirx.auto_unroll_max_step", ann_val=64)
        sch.annotate(block_or_loop=l155, ann_key="tirx.unroll_explicit", ann_val=1)
        b165 = sch.get_sblock(name="conv2d_NCHWc_int8_o", func_name="main")
        (
            l166,
            l167,
            l168,
            l169,
            l170,
            l171,
            l172,
            l173,
            l174,
            l175,
            l176,
            l177,
            l178,
            l179,
            l180,
            l181,
            l182,
            l183,
            l184,
            l185,
            l186,
            l187,
            l188,
            l189,
        ) = sch.get_loops(block=b165)
        b190 = sch.decompose_reduction(block=b165, loop=l170)
        sch.unannotate(block_or_loop=b190, ann_key="meta_schedule.auto_tensorize")
        sch.annotate(block_or_loop=b190, ann_key="meta_schedule.auto_tensorize", ann_val="")
        b191 = sch.get_sblock(name="conv2d_NCHWc_int8_o_init", func_name="main")
        sch.unannotate(block_or_loop=b191, ann_key="meta_schedule.auto_tensorize")
        (b192,) = sch.get_child_blocks(b191)
        (l193,) = sch.get_loops(block=b192)
        sch.vectorize(loop=l193)
        b194 = sch.get_sblock(name="conv2d_NCHWc_int8_o_update", func_name="main")
        sch.unannotate(block_or_loop=b194, ann_key="meta_schedule.auto_tensorize")
        sch.tensorize(block_or_loop=b194, tensor_intrin=VNNI_INTRIN)

    vnni_id = llvm.lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512")
    verify(
        Conv2dInt8_NCHWc,
        apply_trace,
        Conv2dInt8_NCHWc_target,
        {"kind": "llvm", "mcpu": "cascadelake"},
        get_conv2d_vnni_mod(vnni_id),
    )


def test_winograd_gpu():
    def apply_trace(sch):
        b0 = sch.get_sblock(name="B", func_name="main")
        b1 = sch.get_sblock(name="data_pack", func_name="main")
        b2 = sch.get_sblock(name="bgemm", func_name="main")
        b3 = sch.get_sblock(name="A", func_name="main")
        b4 = sch.get_sblock(name="inverse", func_name="main")
        b5 = sch.get_sblock(name="conv2d_winograd", func_name="main")
        b6 = sch.get_sblock(name="T_add", func_name="main")
        b7 = sch.get_sblock(name="T_relu", func_name="main")
        b8 = sch.get_sblock(name="root", func_name="main")
        sch.compute_inline(block=b0)
        (b9,) = sch.get_producers(block=b1)
        (b10,) = sch.get_producers(block=b9)
        l11, l12, l13, l14, l15, l16 = sch.get_loops(block=b1)
        v17, v18 = sch.sample_perfect_tile(
            loop=l13, n=2, max_innermost_factor=64, decision=[14, 14]
        )
        l19, l20 = sch.split(loop=l13, factors=[v17, v18], preserve_unit_iters=True)
        v21, v22 = sch.sample_perfect_tile(loop=l14, n=2, max_innermost_factor=64, decision=[8, 8])
        l23, l24 = sch.split(loop=l14, factors=[v21, v22], preserve_unit_iters=True)
        sch.unroll(loop=l11)
        sch.unroll(loop=l12)
        sch.unroll(loop=l15)
        sch.unroll(loop=l16)
        sch.reorder(l19, l23, l20, l24, l11, l12, l15, l16)
        sch.compute_at(block=b9, loop=l24, preserve_unit_loops=True, index=-1)
        sch.set_scope(block=b9, tensor_index=0, storage_scope="local")
        sch.compute_inline(block=b10)
        l25, l26, l27, l28, l29, l30, l31, l32 = sch.get_loops(block=b1)
        l33 = sch.fuse(l25, l26, l27, l28, preserve_unit_iters=True)
        v34 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        l35, l36 = sch.split(loop=l33, factors=[None, v34], preserve_unit_iters=True)
        sch.bind(loop=l35, thread_axis="blockIdx.x")
        sch.bind(loop=l36, thread_axis="threadIdx.x")
        sch.compute_inline(block=b3)
        l37, l38, l39, l40, l41, l42 = sch.get_loops(block=b4)
        v43, v44 = sch.sample_perfect_tile(loop=l39, n=2, max_innermost_factor=64, decision=[28, 7])
        l45, l46 = sch.split(loop=l39, factors=[v43, v44], preserve_unit_iters=True)
        v47, v48 = sch.sample_perfect_tile(loop=l40, n=2, max_innermost_factor=64, decision=[2, 32])
        l49, l50 = sch.split(loop=l40, factors=[v47, v48], preserve_unit_iters=True)
        sch.unroll(loop=l37)
        sch.unroll(loop=l38)
        sch.unroll(loop=l41)
        sch.unroll(loop=l42)
        sch.reorder(l45, l49, l46, l50, l37, l38, l41, l42)
        l51, l52, l53, l54, l55, l56, l57, l58 = sch.get_loops(block=b4)
        l59 = sch.fuse(l51, l52, l53, l54, preserve_unit_iters=True)
        v60 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=4,
        )
        l61, l62 = sch.split(loop=l59, factors=[None, v60], preserve_unit_iters=True)
        sch.bind(loop=l61, thread_axis="blockIdx.x")
        sch.bind(loop=l62, thread_axis="threadIdx.x")
        sch.annotate(block_or_loop=b2, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        l63, l64, l65, l66, l67 = sch.get_loops(block=b2)
        v68, v69, v70, v71, v72 = sch.sample_perfect_tile(
            loop=l63, n=5, max_innermost_factor=64, decision=[1, 2, 3, 1, 1]
        )
        l73, l74, l75, l76, l77 = sch.split(
            loop=l63, factors=[v68, v69, v70, v71, v72], preserve_unit_iters=True
        )
        v78, v79, v80, v81, v82 = sch.sample_perfect_tile(
            loop=l64, n=5, max_innermost_factor=64, decision=[6, 1, 1, 1, 1]
        )
        l83, l84, l85, l86, l87 = sch.split(
            loop=l64, factors=[v78, v79, v80, v81, v82], preserve_unit_iters=True
        )
        v88, v89, v90, v91, v92 = sch.sample_perfect_tile(
            loop=l65, n=5, max_innermost_factor=64, decision=[7, 2, 1, 14, 1]
        )
        l93, l94, l95, l96, l97 = sch.split(
            loop=l65, factors=[v88, v89, v90, v91, v92], preserve_unit_iters=True
        )
        v98, v99, v100, v101, v102 = sch.sample_perfect_tile(
            loop=l66, n=5, max_innermost_factor=64, decision=[4, 1, 16, 1, 1]
        )
        l103, l104, l105, l106, l107 = sch.split(
            loop=l66, factors=[v98, v99, v100, v101, v102], preserve_unit_iters=True
        )
        v108, v109, v110 = sch.sample_perfect_tile(
            loop=l67, n=3, max_innermost_factor=64, decision=[2, 2, 16]
        )
        l111, l112, l113 = sch.split(loop=l67, factors=[v108, v109, v110], preserve_unit_iters=True)
        sch.reorder(
            l73,
            l83,
            l93,
            l103,
            l74,
            l84,
            l94,
            l104,
            l75,
            l85,
            l95,
            l105,
            l111,
            l112,
            l76,
            l86,
            l96,
            l106,
            l113,
            l77,
            l87,
            l97,
            l107,
        )
        l114 = sch.fuse(l73, l83, l93, l103, preserve_unit_iters=True)
        sch.bind(loop=l114, thread_axis="blockIdx.x")
        l115 = sch.fuse(l74, l84, l94, l104, preserve_unit_iters=True)
        sch.bind(loop=l115, thread_axis="vthread.x")
        l116 = sch.fuse(l75, l85, l95, l105, preserve_unit_iters=True)
        sch.bind(loop=l116, thread_axis="threadIdx.x")
        sch.annotate(
            block_or_loop=b2, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32
        )
        sch.annotate(
            block_or_loop=b2, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024
        )
        b117 = sch.cache_write(block=b2, write_tensor_index=0, storage_scope="local")
        sch.reverse_compute_at(block=b117, loop=l116, preserve_unit_loops=True, index=-1)
        b118 = sch.cache_read(
            block=b2, read_tensor_index=0, storage_scope="shared", consumer_blocks=[b2]
        )
        sch.compute_at(block=b118, loop=l111, preserve_unit_loops=True, index=-1)
        l119, l120, l121, l122, l123, l124, l125, l126 = sch.get_loops(block=b118)
        l127 = sch.fuse(l123, l124, l125, l126, preserve_unit_iters=True)
        v128 = sch.sample_categorical(
            candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3
        )
        sch.annotate(block_or_loop=b118, ann_key="meta_schedule.cooperative_fetch", ann_val=v128)
        b129 = sch.cache_read(
            block=b2, read_tensor_index=1, storage_scope="shared", consumer_blocks=[b2]
        )
        sch.compute_at(block=b129, loop=l111, preserve_unit_loops=True, index=-1)
        l130, l131, l132, l133, l134, l135, l136, l137 = sch.get_loops(block=b129)
        l138 = sch.fuse(l134, l135, l136, l137, preserve_unit_iters=True)
        v139 = sch.sample_categorical(
            candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=3
        )
        sch.annotate(block_or_loop=b129, ann_key="meta_schedule.cooperative_fetch", ann_val=v139)
        sch.reverse_compute_inline(block=b7)
        sch.reverse_compute_inline(block=b6)
        v140 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024],
            probs=[
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
            ],
            decision=4,
        )
        sch.annotate(block_or_loop=b8, ann_key="meta_schedule.unroll_explicit", ann_val=v140)
        l141, l142, l143, l144 = sch.get_loops(block=b5)
        l145 = sch.fuse(l141, l142, l143, l144, preserve_unit_iters=True)
        v146 = sch.sample_categorical(
            candidates=[32, 64, 128, 256, 512, 1024],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        l147, l148 = sch.split(loop=l145, factors=[None, v146], preserve_unit_iters=True)
        sch.bind(loop=l147, thread_axis="blockIdx.x")
        sch.bind(loop=l148, thread_axis="threadIdx.x")
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b118, ann_key="meta_schedule.cooperative_fetch")
        l149, l150, l151, l152, l153 = sch.get_loops(block=b118)
        l154, l155, l156 = sch.split(loop=l153, factors=[None, 48, 4], preserve_unit_iters=True)
        sch.vectorize(loop=l156)
        sch.bind(loop=l155, thread_axis="threadIdx.x")
        sch.unannotate(block_or_loop=b129, ann_key="meta_schedule.cooperative_fetch")
        l157, l158, l159, l160, l161 = sch.get_loops(block=b129)
        l162, l163, l164 = sch.split(loop=l161, factors=[None, 48, 4], preserve_unit_iters=True)
        sch.vectorize(loop=l164)
        sch.bind(loop=l163, thread_axis="threadIdx.x")
        b165 = sch.get_sblock(name="root", func_name="main")
        sch.unannotate(block_or_loop=b165, ann_key="meta_schedule.unroll_explicit")
        b166, b167, b168, b169, b170, b171, b172, b173 = sch.get_child_blocks(b165)
        l174, l175, l176, l177, l178, l179 = sch.get_loops(block=b166)
        sch.annotate(block_or_loop=l174, ann_key="tirx.auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l174, ann_key="tirx.unroll_explicit", ann_val=1)
        l180, l181, l182, l183, l184, l185 = sch.get_loops(block=b167)
        sch.annotate(block_or_loop=l180, ann_key="tirx.auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l180, ann_key="tirx.unroll_explicit", ann_val=1)
        l186, l187, l188, l189, l190, l191, l192 = sch.get_loops(block=b168)
        sch.annotate(block_or_loop=l186, ann_key="tirx.auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l186, ann_key="tirx.unroll_explicit", ann_val=1)
        l193, l194, l195, l196, l197, l198, l199 = sch.get_loops(block=b169)
        sch.annotate(block_or_loop=l193, ann_key="tirx.auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l193, ann_key="tirx.unroll_explicit", ann_val=1)
        (
            l200,
            l201,
            l202,
            l203,
            l204,
            l205,
            l206,
            l207,
            l208,
            l209,
            l210,
            l211,
            l212,
            l213,
        ) = sch.get_loops(block=b170)
        sch.annotate(block_or_loop=l200, ann_key="tirx.auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l200, ann_key="tirx.unroll_explicit", ann_val=1)
        l214, l215, l216, l217, l218, l219, l220 = sch.get_loops(block=b171)
        sch.annotate(block_or_loop=l214, ann_key="tirx.auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l214, ann_key="tirx.unroll_explicit", ann_val=1)
        l221, l222, l223, l224, l225, l226 = sch.get_loops(block=b172)
        sch.annotate(block_or_loop=l221, ann_key="tirx.auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l221, ann_key="tirx.unroll_explicit", ann_val=1)
        l227, l228 = sch.get_loops(block=b173)
        sch.annotate(block_or_loop=l227, ann_key="tirx.auto_unroll_max_step", ann_val=1024)
        sch.annotate(block_or_loop=l227, ann_key="tirx.unroll_explicit", ann_val=1)
        b229 = sch.get_sblock(name="data_pack", func_name="main")
        l230, l231, l232, l233, l234, l235 = sch.get_loops(block=b229)
        b236 = sch.decompose_reduction(block=b229, loop=l234)
        b237 = sch.get_sblock(name="bgemm", func_name="main")
        (
            l238,
            l239,
            l240,
            l241,
            l242,
            l243,
            l244,
            l245,
            l246,
            l247,
            l248,
            l249,
            l250,
            l251,
        ) = sch.get_loops(block=b237)
        b252 = sch.decompose_reduction(block=b237, loop=l241)
        b253 = sch.get_sblock(name="inverse", func_name="main")
        l254, l255, l256, l257, l258, l259 = sch.get_loops(block=b253)
        b260 = sch.decompose_reduction(block=b253, loop=l258)

    verify(
        Conv2dWinogradAddRelu,
        apply_trace,
        Conv2dWinogradAddResidualRelu,
        "cuda",
        Conv2dWinogradAddResidualRelu_scheduled,
    )


def test_inline_order():
    # In this test, the order of applying AutoInline is tested.
    # We need to make sure that the last block in Conv2dInt8_with_predicate_target,
    # "compute_4", is AutoInline-ed after all other blocks have been processed.
    #
    # Otherwise, if the order is "T_add_2" -> "compute_4" -> "compute_3", "compute_4" is neither
    # inlined (because this is the last block) nor reverse-inlined
    # (because it has multiple producers). This results in the "compute_4" block being
    # reverse-inlined at the very end of ScheduleUsingAnchorTrace, where its producer block
    # "conv2d_nhwc_reindex_shared" has the predicate
    # Ts.where(((ax1_0 * 4 + ax1_1) * 32 + ax1_2) * 2 + ax1_3 < 64) due to anchor-block scheduling
    # (see Conv2dInt8_with_predicate_scheduled). ReverseComputeInline cannot be applied in
    # such cases.

    def apply_trace(sch: Schedule) -> None:
        b0 = sch.get_sblock(name="pad_temp", func_name="main")
        b1 = sch.get_sblock(name="conv2d_nhwc", func_name="main")
        b2 = sch.get_sblock(name="T_subtract", func_name="main")
        b3 = sch.get_sblock(name="T_add", func_name="main")
        b4 = sch.get_sblock(name="compute", func_name="main")
        b5 = sch.get_sblock(name="T_add_1", func_name="main")
        b6 = sch.get_sblock(name="compute_1", func_name="main")
        b7 = sch.get_sblock(name="T_subtract_1", func_name="main")
        b8 = sch.get_sblock(name="compute_2", func_name="main")
        b9 = sch.get_sblock(name="root", func_name="main")
        sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
        b10 = sch.reindex(block=b1, tensor=("write", 0))
        b11 = sch.reindex(block=b1, tensor=("read", 0))
        b12 = sch.reindex(block=b1, tensor=("read", 1))
        sch.transform_layout(
            block=b1,
            tensor=("read", 0),
            index_map=lambda nn, yy, xx, rc: (
                (((nn * 3136) + (yy * 56)) + xx),
                rc,
            ),
            pad_value=None,
        )
        sch.transform_layout(
            block=b1,
            tensor=("read", 1),
            index_map=lambda ff, ry, rx, rc: (
                ry,
                rx,
                ff,
                rc,
            ),
            pad_value=None,
        )
        sch.transform_layout(
            block=b1,
            tensor=("write", 0),
            index_map=lambda nn, yy, xx, ff: (
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
            ),
            pad_value=None,
        )
        sch.transform_block_layout(
            block=b10,
            index_map=lambda nn, yy, xx, ff: (
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
            ),
        )
        sch.transform_block_layout(
            block=b11,
            index_map=lambda nn, yy, xx, rc: (
                (((nn * 3136) + (yy * 56)) + xx),
                rc,
            ),
        )
        sch.transform_block_layout(
            block=b12,
            index_map=lambda ff, ry, rx, rc: (
                ry,
                rx,
                ff,
                rc,
            ),
        )
        sch.transform_block_layout(
            block=b1,
            index_map=lambda nn, yy, xx, ff, ry, rx, rc: (
                ry,
                rx,
                (((nn * 3136) + (yy * 56)) + xx),
                ff,
                rc,
            ),
        )
        l13, l14, l15, l16, l17 = sch.get_loops(block=b1)
        l18, l19 = sch.split(loop=l17, factors=[None, 16], preserve_unit_iters=True)
        l20, l21 = sch.split(loop=l16, factors=[None, 16], preserve_unit_iters=True)
        l22, l23 = sch.split(loop=l15, factors=[None, 16], preserve_unit_iters=True)
        l24, l25, l26, l27, l28, l29, l30, l31 = sch.get_loops(block=b1)
        sch.reorder(l28, l30, l23, l21, l19)
        b32 = sch.blockize(target=l23)
        sch.annotate(
            block_or_loop=b32,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_sync_16x16x16_s8s8s32_trans",
        )
        sch.annotate(
            block_or_loop=b32,
            ann_key="meta_schedule.auto_tensorize_init",
            ann_val="wmma_fill_16x16x16_s32",
        )
        sch.annotate(block_or_loop=b32, ann_key="warp_execution", ann_val=1)
        l33, l34, l35, l36, l37 = sch.get_loops(block=b32)
        v38, v39, v40 = sch.sample_perfect_tile(
            loop=l33, n=3, max_innermost_factor=4, decision=[1, 1, 1]
        )
        l41, l42, l43 = sch.split(loop=l33, factors=[v38, v39, v40], preserve_unit_iters=True)
        v44, v45, v46 = sch.sample_perfect_tile(
            loop=l34, n=3, max_innermost_factor=4, decision=[1, 1, 1]
        )
        l47, l48, l49 = sch.split(loop=l34, factors=[v44, v45, v46], preserve_unit_iters=True)
        v50, v51, v52, v53, v54 = sch.sample_perfect_tile(
            loop=l35, n=5, max_innermost_factor=4, decision=[8, 196, 2, 1, 1]
        )
        l55, l56, l57, l58, l59 = sch.split(
            loop=l35, factors=[v50, v51, v52, v53, v54], preserve_unit_iters=True
        )
        v60, v61, v62, v63, v64 = sch.sample_perfect_tile(
            loop=l36, n=5, max_innermost_factor=4, decision=[4, 1, 2, 1, 2]
        )
        l65, l66, l67, l68, l69 = sch.split(
            loop=l36, factors=[v60, v61, v62, v63, v64], preserve_unit_iters=True
        )
        v70, v71, v72 = sch.sample_perfect_tile(
            loop=l37, n=3, max_innermost_factor=4, decision=[2, 2, 1]
        )
        l73, l74, l75 = sch.split(loop=l37, factors=[v70, v71, v72], preserve_unit_iters=True)
        sch.reorder(
            l55,
            l65,
            l56,
            l66,
            l57,
            l67,
            l41,
            l47,
            l73,
            l42,
            l48,
            l74,
            l58,
            l68,
            l43,
            l49,
            l75,
            l59,
            l69,
        )
        l76 = sch.fuse(l55, l65, preserve_unit_iters=True)
        sch.bind(loop=l76, thread_axis="blockIdx.y")
        l77 = sch.fuse(l56, l66, preserve_unit_iters=True)
        sch.bind(loop=l77, thread_axis="blockIdx.x")
        l78 = sch.fuse(l57, l67, preserve_unit_iters=True)
        sch.bind(loop=l78, thread_axis="threadIdx.y")
        sch.annotate(
            block_or_loop=b32, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32
        )
        sch.annotate(
            block_or_loop=b32, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024
        )
        b79 = sch.cache_write(block=b32, write_tensor_index=0, storage_scope="shared")
        sch.reverse_compute_at(block=b79, loop=l77, preserve_unit_loops=True, index=-1)
        b80 = sch.cache_write(block=b32, write_tensor_index=0, storage_scope="wmma.accumulator")
        sch.reverse_compute_at(block=b80, loop=l78, preserve_unit_loops=True, index=-1)
        v81 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=1,
        )
        sch.annotate(block_or_loop=b79, ann_key="meta_schedule.cooperative_fetch", ann_val=v81)
        sch.reverse_compute_inline(block=b10)
        l82, l83, l84, l85, l86 = sch.get_loops(block=b80)
        l87, l88 = sch.split(loop=l86, factors=[None, 16], preserve_unit_iters=True)
        l89, l90 = sch.split(loop=l85, factors=[None, 16], preserve_unit_iters=True)
        l91, l92, l93, l94, l95, l96, l97 = sch.get_loops(block=b80)
        sch.reorder(l96, l90, l88)
        b98 = sch.blockize(target=l90)
        sch.annotate(
            block_or_loop=b98,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_store_16x16x16_s32_shared",
        )
        b99 = sch.cache_read(
            block=b32, read_tensor_index=0, storage_scope="shared", consumer_blocks=[b32]
        )
        sch.compute_at(block=b99, loop=l73, preserve_unit_loops=True, index=-1)
        l100, l101, l102, l103, l104, l105, l106, l107 = sch.get_loops(block=b99)
        l108 = sch.fuse(l106, l107, preserve_unit_iters=True)
        v109 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=3,
        )
        sch.annotate(block_or_loop=b99, ann_key="meta_schedule.cooperative_fetch", ann_val=v109)
        b110 = sch.cache_read(
            block=b32, read_tensor_index=1, storage_scope="shared", consumer_blocks=[b32]
        )
        sch.compute_at(block=b110, loop=l73, preserve_unit_loops=True, index=-1)
        l111, l112, l113, l114, l115, l116, l117, l118, l119, l120 = sch.get_loops(block=b110)
        l121 = sch.fuse(l117, l118, l119, l120, preserve_unit_iters=True)
        v122 = sch.sample_categorical(
            candidates=[1, 2, 3, 4, 8, 16],
            probs=[
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
                0.16666666666666666,
            ],
            decision=2,
        )
        sch.annotate(block_or_loop=b110, ann_key="meta_schedule.cooperative_fetch", ann_val=v122)
        b123 = sch.cache_read(block=b32, read_tensor_index=0, storage_scope="wmma.matrix_a")
        sch.compute_at(block=b123, loop=l74, preserve_unit_loops=True, index=-1)
        l124, l125, l126, l127, l128, l129, l130, l131, l132, l133, l134 = sch.get_loops(block=b123)
        l135, l136 = sch.split(loop=l134, factors=[None, 16], preserve_unit_iters=True)
        l137, l138 = sch.split(loop=l133, factors=[None, 16], preserve_unit_iters=True)
        (
            l139,
            l140,
            l141,
            l142,
            l143,
            l144,
            l145,
            l146,
            l147,
            l148,
            l149,
            l150,
            l151,
        ) = sch.get_loops(block=b123)
        sch.reorder(l150, l138, l136)
        b152 = sch.blockize(target=l138)
        sch.annotate(
            block_or_loop=b152,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_load_16x16x16_s8_a_shared",
        )
        b153 = sch.cache_read(block=b32, read_tensor_index=1, storage_scope="wmma.matrix_b")
        sch.compute_at(block=b153, loop=l74, preserve_unit_loops=True, index=-1)
        (
            l154,
            l155,
            l156,
            l157,
            l158,
            l159,
            l160,
            l161,
            l162,
            l163,
            l164,
            l165,
            l166,
        ) = sch.get_loops(block=b153)
        l167, l168 = sch.split(loop=l166, factors=[None, 16], preserve_unit_iters=True)
        l169, l170 = sch.split(loop=l165, factors=[None, 16], preserve_unit_iters=True)
        (
            l171,
            l172,
            l173,
            l174,
            l175,
            l176,
            l177,
            l178,
            l179,
            l180,
            l181,
            l182,
            l183,
            l184,
            l185,
        ) = sch.get_loops(block=b153)
        sch.reorder(l184, l170, l168)
        b186 = sch.blockize(target=l170)
        sch.annotate(
            block_or_loop=b186,
            ann_key="meta_schedule.auto_tensorize",
            ann_val="wmma_load_16x16x16_s8_b_trans_shared",
        )
        sch.compute_inline(block=b11)
        sch.compute_inline(block=b12)
        sch.storage_align(block=b99, tensor_index=0, axis=-2, factor=32, offset=16)
        sch.storage_align(block=b110, tensor_index=0, axis=-2, factor=32, offset=16)
        sch.reverse_compute_inline(block=b8)
        sch.reverse_compute_inline(block=b7)
        sch.reverse_compute_inline(block=b6)
        sch.reverse_compute_inline(block=b5)
        sch.reverse_compute_inline(block=b4)
        sch.reverse_compute_inline(block=b3)
        sch.reverse_compute_inline(block=b2)
        sch.compute_inline(block=b0)

        v187 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024],
            probs=[
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
                0.20000000000000001,
            ],
            decision=4,
        )
        sch.annotate(block_or_loop=b9, ann_key="meta_schedule.unroll_explicit", ann_val=v187)
        sch.enter_postproc()
        sch.unannotate(block_or_loop=b79, ann_key="meta_schedule.cooperative_fetch")
        l188, l189, l190, l191 = sch.get_loops(block=b79)

        l192, l193, l194, l195 = sch.split(
            loop=l191, factors=[None, 4, 32, 2], preserve_unit_iters=True
        )

    verify(
        Conv2dInt8_with_predicate,
        apply_trace,
        Conv2dInt8_with_predicate_target,
        "cuda",
        Conv2dInt8_with_predicate_scheduled,
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
