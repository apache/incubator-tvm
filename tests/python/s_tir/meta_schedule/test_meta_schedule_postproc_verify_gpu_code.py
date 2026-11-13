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
# ruff: noqa: E501, F401
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def _target() -> Target:
    return Target("nvidia/geforce-rtx-3080")


def _create_context(mod, target) -> ms.TuneContext:
    return ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[ms.postproc.VerifyGPUCode()],
            mutator_probs={},
        ),
        task_name="test",
    )


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable,misplaced-comparison-constant
# fmt: off

@tvm_tirx.script.ir_module
class Conv2dCuda0:
    @Ts.function
    def main(A: T.Tensor([14 * 14 * 256 * 256], dtype='float32'), B: T.Tensor([14 * 14 * 512 * 256], dtype='float32')) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition

        # body
        blockIdx_z = T.launch_thread("blockIdx.z", 196)
        B_local = T.alloc_tensor(ty_args=[T.Tensor([64], "float32", scope="local")])
        Apad_shared = T.alloc_tensor(ty_args=[T.Tensor([512], "float32", scope="shared")])
        Apad_shared_local = T.alloc_tensor(ty_args=[T.Tensor([8], "float32", scope="local")])
        blockIdx_y = T.launch_thread("blockIdx.y", 8)
        blockIdx_x = T.launch_thread("blockIdx.x", 4)
        threadIdx_y = T.launch_thread("threadIdx.y", 8)
        threadIdx_x = T.launch_thread("threadIdx.x", 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            B_local[ff_c_init * 8 + nn_c_init] = T.float32(0)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                Apad_shared[T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4)] = T.if_then_else(
                    1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15,
                    A[T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4)],
                    T.broadcast(T.float32(0), 4),

                )
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    Apad_shared_local[ax3] = Apad_shared[rc_inner * 64 + threadIdx_x * 8 + ax3]
                for ff_c, nn_c in T.grid(8, 8):
                    B_local[ff_c * 8 + nn_c] = B_local[ff_c * 8 + nn_c] + Apad_shared_local[nn_c]
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            B[blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner] = B_local[ff_inner_inner_inner * 8 + nn_inner_inner_inner]

@tvm_tirx.script.ir_module
class Conv2dCuda1:
    @Ts.function
    def main(A: T.Tensor([14 * 14 * 256 * 256], dtype='float32'), B: T.Tensor([14 * 14 * 512 * 256], dtype='float32')) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition

        # body
        blockIdx_z = T.launch_thread("blockIdx.z", 196)
        B_local = T.alloc_tensor(ty_args=[T.Tensor([6400000], "float32", scope="local")])
        Apad_shared = T.alloc_tensor(ty_args=[T.Tensor([512], "float32", scope="shared")])
        Apad_shared_local = T.alloc_tensor(ty_args=[T.Tensor([8], "float32", scope="local")])
        blockIdx_y = T.launch_thread("blockIdx.y", 8)
        blockIdx_x = T.launch_thread("blockIdx.x", 4)
        threadIdx_y = T.launch_thread("threadIdx.y", 8)
        threadIdx_x = T.launch_thread("threadIdx.x", 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            B_local[ff_c_init * 8 + nn_c_init] = T.float32(0)
            # Access of the last element of B_local prevents tensor
            # compacting from reducing the amount of shared memory
            # used.
            B_local[6400000-1 + ff_c_init*8] = 0.0
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                Apad_shared[T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4)] = T.if_then_else(
                    1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15,
                    A[T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4)],
                    T.broadcast(T.float32(0), 4),

                )
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    Apad_shared_local[ax3] = Apad_shared[rc_inner * 64 + threadIdx_x * 8 + ax3]
                for ff_c, nn_c in T.grid(8, 8):
                    B_local[ff_c * 8 + nn_c] = B_local[ff_c * 8 + nn_c] + Apad_shared_local[nn_c]
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            B[blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner] = B_local[ff_inner_inner_inner * 8 + nn_inner_inner_inner]

@tvm_tirx.script.ir_module
class Conv2dCuda2:
    @Ts.function
    def main(A: T.Tensor([14 * 14 * 256 * 256], dtype='float32'), B: T.Tensor([14 * 14 * 512 * 256], dtype='float32')) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition

        # body
        blockIdx_z = T.launch_thread("blockIdx.z", 196)
        B_local = T.alloc_tensor(ty_args=[T.Tensor([64], "float32", scope="local")])
        Apad_shared = T.alloc_tensor(ty_args=[T.Tensor([512000], "float32", scope="shared")])
        Apad_shared_local = T.alloc_tensor(ty_args=[T.Tensor([8], "float32", scope="local")])
        blockIdx_y = T.launch_thread("blockIdx.y", 8)
        blockIdx_x = T.launch_thread("blockIdx.x", 4)
        threadIdx_y = T.launch_thread("threadIdx.y", 8)
        threadIdx_x = T.launch_thread("threadIdx.x", 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            B_local[ff_c_init * 8 + nn_c_init] = T.float32(0)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                Apad_shared[T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4)] = T.if_then_else(
                    1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15,
                    A[T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4)],
                    T.broadcast(T.float32(0), 4),

                )
                # Access of the last element of Apad_shared prevents
                # tensor compacting from reducing the amount of shared
                # memory used.
                Apad_shared[512000-1] = 0.0
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    Apad_shared_local[ax3] = Apad_shared[rc_inner * 64 + threadIdx_x * 8 + ax3]
                for ff_c, nn_c in T.grid(8, 8):
                    B_local[ff_c * 8 + nn_c] = B_local[ff_c * 8 + nn_c] + Apad_shared_local[nn_c]
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            B[blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner] = B_local[ff_inner_inner_inner * 8 + nn_inner_inner_inner]

@tvm_tirx.script.ir_module
class Conv2dCuda3:
    @Ts.function
    def main(A: T.Tensor([14 * 14 * 256 * 256], dtype='float32'), B: T.Tensor([14 * 14 * 512 * 256], dtype='float32')) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition

        # body
        blockIdx_z = T.launch_thread("blockIdx.z", 196)
        B_local = T.alloc_tensor(ty_args=[T.Tensor([64], "float32", scope="local")])
        Apad_shared = T.alloc_tensor(ty_args=[T.Tensor([512], "float32", scope="shared")])
        Apad_shared_local = T.alloc_tensor(ty_args=[T.Tensor([8], "float32", scope="local")])
        blockIdx_y = T.launch_thread("blockIdx.y", 8)
        blockIdx_x = T.launch_thread("blockIdx.x", 4)
        threadIdx_y = T.launch_thread("threadIdx.y", 8)
        threadIdx_x = T.launch_thread("threadIdx.x", 800000)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            B_local[ff_c_init * 8 + nn_c_init] = T.float32(0)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                Apad_shared[T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4)] = T.if_then_else(
                    1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15,
                    A[T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4)],
                    T.broadcast(T.float32(0), 4),

                )
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    Apad_shared_local[ax3] = Apad_shared[rc_inner * 64 + threadIdx_x * 8 + ax3]
                for ff_c, nn_c in T.grid(8, 8):
                    B_local[ff_c * 8 + nn_c] = B_local[ff_c * 8 + nn_c] + Apad_shared_local[nn_c]
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            B[blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner] = B_local[ff_inner_inner_inner * 8 + nn_inner_inner_inner]

@Ts.function
def GmmCuda0(X: T.Tensor((1, 128, 128), "float32"), Y: T.Tensor((1, 128, 128), "float32"), Z: T.Tensor((1, 128, 128), "float32")) -> None:
    Z_local = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="local")
    X_shared = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="shared")
    Y_shared = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="shared")
    for i0_0_i1_0_i2_0_fused in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
        for i0_1_i1_1_i2_1_fused in T.parallel(1, annotations={"thread_binding": "vthread.x"}):
            for i0_2_i1_2_i2_2_fused in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                for i1_3_init, i2_4_init in T.grid(4, 2):
                    with Ts.sblock("Z_init"):
                        b = Ts.axis.spatial(1, 0)
                        i = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3_init)
                        j = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4_init)
                        Ts.reads()
                        Ts.writes(Z_local[b, i, j])
                        Z_local[b, i, j] = T.float32(0)
                for i3_0 in T.serial(4):
                    for ax0_ax1_ax2_fused_0 in T.serial(4):
                        for ax0_ax1_ax2_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                            for ax0_ax1_ax2_fused_2 in T.vectorized(2):
                                with Ts.sblock("X_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) // 32)
                                    v2 = Ts.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) % 32)
                                    Ts.reads(X[v0, v1, v2])
                                    Ts.writes(X_shared[v0, v1, v2])
                                    X_shared[v0, v1, v2] = X[v0, v1, v2]
                    for ax0_ax1_ax2_fused_0 in T.serial(8):
                        for ax0_ax1_ax2_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("Y_shared"):
                                v0 = Ts.axis.spatial(1, 0)
                                v1 = Ts.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) // 32)
                                v2 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) % 32)
                                Ts.reads(Y[v0, v1, v2])
                                Ts.writes(Y_shared[v0, v1, v2])
                                Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                    for i3_1, i0_3, i1_3, i2_3, i3_2, i0_4, i1_4, i2_4 in T.grid(1, 1, 4, 1, 32, 1, 1, 2):
                        with Ts.sblock("Z_update"):
                            b = Ts.axis.spatial(1, 0)
                            i = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3)
                            j = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4)
                            k = Ts.axis.reduce(128, i3_0 * 32 + i3_2)
                            Ts.reads(Z_local[b, i, j], X_shared[b, i, k], Y_shared[b, k, j])
                            Ts.writes(Z_local[b, i, j])
                            Z_local[b, i, j] = Z_local[b, i, j] + X_shared[b, i, k] * Y_shared[b, k, j]
                for ax0, ax1, ax2 in T.grid(1, 4, 2):
                    with Ts.sblock("Z_local"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + ax1)
                        v2 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + ax2)
                        Ts.reads(Z_local[v0, v1, v2])
                        Ts.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_local[v0, v1, v2]

@Ts.function
def GmmCuda1(X: T.Tensor((1, 128, 128), "float32"), Y: T.Tensor((1, 128, 128), "float32"), Z: T.Tensor((1, 128, 128), "float32")) -> None:
    Z_local = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="local")
    X_shared = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="shared")
    Y_shared = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="shared")
    for i0_0_i1_0_i2_0_fused in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
        for i0_1_i1_1_i2_1_fused in T.parallel(1, annotations={"thread_binding": "vthread.x"}):
            for i0_2_i1_2_i2_2_fused in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                for i1_3_init, i2_4_init in T.grid(4, 2):
                    with Ts.sblock("Z_init"):
                        b = Ts.axis.spatial(1, 0)
                        i = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3_init)
                        j = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4_init)
                        Ts.reads()
                        Ts.writes(Z_local[b, i, j])
                        Z_local[b, i, j] = T.float32(0)
                for i3_0 in T.serial(4):
                    for ax0_ax1_ax2_fused_0 in T.serial(4):
                        for ax0_ax1_ax2_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                            for ax0_ax1_ax2_fused_2 in T.vectorized(2):
                                with Ts.sblock("X_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) // 32)
                                    v2 = Ts.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) % 32)
                                    Ts.reads(X[v0, v1, v2])
                                    Ts.writes(X_shared[v0, v1, v2])
                                    X_shared[v0, v1, v2] = X[v0, v1, v2]
                    for ax0_ax1_ax2_fused_0 in T.serial(8):
                        for ax0_ax1_ax2_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("Y_shared"):
                                v0 = Ts.axis.spatial(1, 0)
                                v1 = Ts.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) // 32)
                                v2 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) % 32)
                                Ts.reads(Y[v0, v1, v2])
                                Ts.writes(Y_shared[v0, v1, v2])
                                Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                    for i3_1, i0_3, i1_3, i2_3, i3_2, i0_4, i1_4, i2_4 in T.grid(1, 1, 4, 1, 32, 1, 1, 2):
                        with Ts.sblock("Z_update"):
                            b = Ts.axis.spatial(1, 0)
                            i = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3)
                            j = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4)
                            k = Ts.axis.reduce(128, i3_0 * 32 + i3_2)
                            Ts.sblock_attr({
                                "meta_schedule.thread_extent_low_inclusive": 0,
                                "meta_schedule.thread_extent_high_inclusive": 32,
                            })
                            Ts.reads(Z_local[b, i, j], X_shared[b, i, k], Y_shared[b, k, j])
                            Ts.writes(Z_local[b, i, j])
                            Z_local[b, i, j] = Z_local[b, i, j] + X_shared[b, i, k] * Y_shared[b, k, j]
                for ax0, ax1, ax2 in T.grid(1, 4, 2):
                    with Ts.sblock("Z_local"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + ax1)
                        v2 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + ax2)
                        Ts.reads(Z_local[v0, v1, v2])
                        Ts.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_local[v0, v1, v2]

@Ts.function
def GmmCuda2(X: T.Tensor((1, 128, 128), "float32"), Y: T.Tensor((1, 128, 128), "float32"), Z: T.Tensor((1, 128, 128), "float32")) -> None:
    Z_local = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="local")
    X_shared = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="shared")
    Y_shared = Ts.sblock_alloc_tensor([1, 128, 128], dtype="float32", scope="shared")
    for i0_0_i1_0_i2_0_fused in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
        for i0_1_i1_1_i2_1_fused in T.parallel(1, annotations={"thread_binding": "vthread.x"}):
            for i0_2_i1_2_i2_2_fused in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                for i1_3_init, i2_4_init in T.grid(4, 2):
                    with Ts.sblock("Z_init"):
                        b = Ts.axis.spatial(1, 0)
                        i = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3_init)
                        j = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4_init)
                        Ts.reads()
                        Ts.writes(Z_local[b, i, j])
                        Z_local[b, i, j] = T.float32(0)
                for i3_0 in T.serial(4):
                    for ax0_ax1_ax2_fused_0 in T.serial(4):
                        for ax0_ax1_ax2_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                            for ax0_ax1_ax2_fused_2 in T.vectorized(2):
                                with Ts.sblock("X_shared"):
                                    v0 = Ts.axis.spatial(1, 0)
                                    v1 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) // 32)
                                    v2 = Ts.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 2 + ax0_ax1_ax2_fused_2) % 32)
                                    Ts.reads(X[v0, v1, v2])
                                    Ts.writes(X_shared[v0, v1, v2])
                                    X_shared[v0, v1, v2] = X[v0, v1, v2]
                    for ax0_ax1_ax2_fused_0 in T.serial(8):
                        for ax0_ax1_ax2_fused_1 in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                            with Ts.sblock("Y_shared"):
                                v0 = Ts.axis.spatial(1, 0)
                                v1 = Ts.axis.spatial(128, i3_0 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) // 32)
                                v2 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + (ax0_ax1_ax2_fused_0 * 128 + ax0_ax1_ax2_fused_1) % 32)
                                Ts.reads(Y[v0, v1, v2])
                                Ts.writes(Y_shared[v0, v1, v2])
                                Y_shared[v0, v1, v2] = Y[v0, v1, v2]
                    for i3_1, i0_3, i1_3, i2_3, i3_2, i0_4, i1_4, i2_4 in T.grid(1, 1, 4, 1, 32, 1, 1, 2):
                        with Ts.sblock("Z_update"):
                            b = Ts.axis.spatial(1, 0)
                            i = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + i1_3)
                            j = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + i2_4)
                            k = Ts.axis.reduce(128, i3_0 * 32 + i3_2)
                            Ts.sblock_attr({
                                "meta_schedule.thread_extent_low_inclusive": 1024,
                                "meta_schedule.thread_extent_high_inclusive": 1024,
                            })
                            Ts.reads(Z_local[b, i, j], X_shared[b, i, k], Y_shared[b, k, j])
                            Ts.writes(Z_local[b, i, j])
                            Z_local[b, i, j] = Z_local[b, i, j] + X_shared[b, i, k] * Y_shared[b, k, j]
                for ax0, ax1, ax2 in T.grid(1, 4, 2):
                    with Ts.sblock("Z_local"):
                        v0 = Ts.axis.spatial(1, ax0)
                        v1 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused // 4 * 32 + i0_2_i1_2_i2_2_fused // 16 * 4 + ax1)
                        v2 = Ts.axis.spatial(128, i0_0_i1_0_i2_0_fused % 4 * 32 + i0_2_i1_2_i2_2_fused % 16 * 2 + ax2)
                        Ts.reads(Z_local[v0, v1, v2])
                        Ts.writes(Z[v0, v1, v2])
                        Z[v0, v1, v2] = Z_local[v0, v1, v2]

s0 = T.dynamic("s0", "int32")
s0_1 = T.dynamic("s0_1", "int32")
s0_2 = T.dynamic("s0_2", "int32")
s1 = T.dynamic("s1", "int32")
s1_1 = T.dynamic("s1_1", "int32")
s1_2 = T.dynamic("s1_2", "int32")


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,not-callable,misplaced-comparison-constant


@pytest.mark.parametrize("mod", [Conv2dCuda0, Conv2dCuda1, GmmCuda0])
def test_postproc_check_pass(mod):
    ctx = _create_context(mod, target=_target())
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    assert ctx.space_generator.postprocs[0].apply(sch)


@pytest.mark.parametrize(
    "mod",
    [
        Conv2dCuda2,  # Should fail due to too much local memory per block (large Apad_shared allocation)
        Conv2dCuda3,  # Should fail due to too many threads per block (large threadIdx.x extent)
        GmmCuda1,
        GmmCuda2,
    ],
)
def test_postproc_check_fail(mod):
    ctx = _create_context(mod, target=_target())
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    assert not ctx.space_generator.postprocs[0].apply(sch)


if __name__ == "__main__":
    tvm_tirx.testing.main()
