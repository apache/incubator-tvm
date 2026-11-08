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
# pylint: disable=missing-function-docstring,missing-module-docstring
# ruff: noqa: E501, F401
from __future__ import annotations

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import te
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

@Ts.function
def two_elementwise(A: T.Tensor((128, 128), 'float32'), C: T.Tensor((128, 128), 'float32')) -> None:

    B = Ts.sblock_alloc_tensor((128, 128), "float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def two_elementwise_after_compute_at(A: T.Tensor((128, 128), 'float32'), C: T.Tensor((128, 128), 'float32')) -> None:

    B = Ts.sblock_alloc_tensor((128, 128), "float32")

    for i in range(0, 128):
        for ax0, ax1 in T.grid(1, 128):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i + ax0)
                vj = Ts.axis.S(128, ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def blockized_1(A: T.Tensor([128, 128], 'float32'), C: T.Tensor([128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([128, 128], "float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(8, 8):
        with Ts.sblock("C_outer"):
            vi_o, vj_o = Ts.axis.remap("SS", [i, j])
            Ts.reads([B[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16,
            ]])
            Ts.writes([C[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16
            ]])
            for i_i, j_i in T.grid(16, 16):
                with Ts.sblock("C_inner"):
                    vi = Ts.axis.S(128, vi_o * 16 + i_i)
                    vj = Ts.axis.S(128, vj_o * 16 + j_i)
                    C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def blockized_after_compute_at(A: T.Tensor([128, 128], 'float32'), C: T.Tensor([128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([128, 128], "float32")

    for i0_0, i1_0 in T.grid(8, 8):
        for ax0, ax1 in T.grid(16, 16):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i0_0 * 16 + ax0)
                vj = Ts.axis.S(128, i1_0 * 16 + ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        with Ts.sblock("C_outer"):
            vi_o, vj_o = Ts.axis.remap("SS", [i0_0, i1_0])
            Ts.reads([B[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16,
            ]])
            Ts.writes([C[
                vi_o * 16 : vi_o * 16 + 16,
                vj_o * 16 : vj_o * 16 + 16
            ]])
            for i0_1, i1_1 in T.grid(16, 16):
                with Ts.sblock("C_inner"):
                    vi = Ts.axis.S(128, vi_o * 16 + i0_1)
                    vj = Ts.axis.S(128, vj_o * 16 + i1_1)
                    C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def blockized_2(A: T.Tensor([128, 128], 'float32'), C: T.Tensor([128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([128, 128], "float32")

    for i_o, j_o in T.grid(8, 8):
        with Ts.sblock("B_outer"):
            vio, vjo = Ts.axis.remap("SS", [i_o, j_o])
            Ts.reads([A[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16,
            ]])
            Ts.writes([B[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16
            ]])
            for i_i, j_i in T.grid(16, 16):
                with Ts.sblock("B_inner"):
                    vi = Ts.axis.S(128, vio * 16 + i_i)
                    vj = Ts.axis.S(128, vjo * 16 + j_i)
                    B[vi, vj] = A[vi, vj] * 2.0
    for i_o, j_o, i_i, j_i in T.grid(4, 4, 32, 32):
        with Ts.sblock("C"):
            vi = Ts.axis.S(128, i_o * 32 + i_i)
            vj = Ts.axis.S(128, j_o * 32 + j_i)
            C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def blockized_2_after_reverse_compute_at(A: T.Tensor([128, 128], 'float32'), C: T.Tensor([128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([128, 128], "float32")

    for i_o, j_o in T.grid(8, 8):
        with Ts.sblock("B_outer"):
            vio, vjo = Ts.axis.remap("SS", [i_o, j_o])
            Ts.reads([A[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16,
            ]])
            Ts.writes([B[
                vio * 16 : vio * 16 + 16,
                vjo * 16 : vjo * 16 + 16
            ]])
            for i_i, j_i in T.grid(16, 16):
                with Ts.sblock("B_inner"):
                    vi = Ts.axis.S(128, vio * 16 + i_i)
                    vj = Ts.axis.S(128, vjo * 16 + j_i)
                    B[vi, vj] = A[vi, vj] * 2.0
        for ax0, ax1 in T.grid(16, 16):
            with Ts.sblock("C"):
                vi = Ts.axis.S(128, i_o * 16 + ax0)
                vj = Ts.axis.S(128, j_o * 16 + ax1)
                Ts.reads([B[vi, vj]])
                Ts.writes([C[vi, vj]])
                C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def blockized_2_after_compute_at(A: T.Tensor([128, 128], 'float32'), C: T.Tensor([128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([128, 128], "float32")

    for i_o, j_o in T.grid(4, 4):
        for ax0, ax1 in T.grid(2, 2):
            with Ts.sblock("blockized_B"):
                vio = Ts.axis.S(8, i_o * 2 + ax0)
                vjo = Ts.axis.S(8, j_o * 2 + ax1)
                Ts.reads([A[
                    vio * 16 : vio * 16 + 16,
                    vjo * 16 : vjo * 16 + 16,
                ]])
                Ts.writes([B[
                    vio * 16 : vio * 16 + 16,
                    vjo * 16 : vjo * 16 + 16,
                ]])
                for i_i, j_i in T.grid(16, 16):
                    with Ts.sblock("B"):
                        vi = Ts.axis.S(128, vio * 16 + i_i)
                        vj = Ts.axis.S(128, vjo * 16 + j_i)
                        B[vi, vj] = A[vi, vj] * 2.0
        for i_i, j_i in T.grid(32, 32):
            with Ts.sblock("C"):
                vi = Ts.axis.S(128, i_o * 32 + i_i)
                vj = Ts.axis.S(128, j_o * 32 + j_i)
                C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def cuda_matmul_0(A: T.Tensor([2048, 2048], 'float32'), B: T.Tensor([2048, 2048], 'float32'), C: T.Tensor([2048, 2048], 'float32')) -> None:  # pylint: disable=undefined-loop-variable

    A_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    B_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    A_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    B_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    C_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("A_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("A_shared_local"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            A_shared_local[v0, v1] = A_shared[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared_local"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared_local[v0, v1] = B_shared[v0, v1]
    for i, j, k in T.grid(2048, 2048, 2048):
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C_local[vi, vj] = 0.0
            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
    for by in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.y"}):
        for bx in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.x"}):
            for vy in T.parallel(0, 2, annotations={"thread_binding": "vthread.y"}):
                for vx in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                    for ty in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.y"}):
                        for tx in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                            for i, j in T.grid(4, 4):
                                with Ts.sblock("C_local"):
                                    v0_4 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1_4 = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0_4, v1_4] = C_local[v0_4, v1_4]

@Ts.function
def cuda_matmul_0_after_compute_at(A: T.Tensor([2048, 2048], 'float32'), B: T.Tensor([2048, 2048], 'float32'), C: T.Tensor([2048, 2048], 'float32')) -> None:  # pylint: disable=undefined-loop-variable

    A_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    B_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    A_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    B_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    C_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("A_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("A_shared_local"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            A_shared_local[v0, v1] = A_shared[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared_local"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.y"}):
        for bx in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.x"}):
            for vy in T.parallel(0, 2, annotations={"thread_binding": "vthread.y"}):
                for vx in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                    for ty in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.y"}):
                        for tx in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                            for i, j, k in T.grid(4, 4, 2048):
                                with Ts.sblock("C"):
                                    vi = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    vj = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    vk = Ts.axis.R(2048, k)
                                    with Ts.init():
                                        C_local[vi, vj] = 0.0
                                    C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with Ts.sblock("C_local"):
                                    vi = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    vj = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[vi, vj] = C_local[vi, vj]

@Ts.function
def cuda_matmul_1(A: T.Tensor([2048, 2048], 'float32'), B: T.Tensor([2048, 2048], 'float32'), C: T.Tensor([2048, 2048], 'float32')) -> None:  # pylint: disable=undefined-loop-variable

    A_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    B_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    A_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    B_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    C_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("A_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("A_shared_local"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            A_shared_local[v0, v1] = A_shared[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared_local"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.y"}):
        for bx in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.x"}):
            for vy in T.parallel(0, 2, annotations={"thread_binding": "vthread.y"}):
                for vx in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                    for ty in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.y"}):
                        for tx in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                            for k_0 in T.serial(0, 256):
                                for k_1 in T.unroll(0, 8):
                                    for _, i, j in T.grid(1, 4, 4):
                                        with Ts.sblock("C"):
                                            vi = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = Ts.axis.R(2048, k_0 * 8 + k_1)
                                            with Ts.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with Ts.sblock("C_local"):
                                    vi = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    vj = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[vi, vj] = C_local[vi, vj]

@Ts.function
def cuda_matmul_2(A: T.Tensor([2048, 2048], 'float32'), B: T.Tensor([2048, 2048], 'float32'), C: T.Tensor([2048, 2048], 'float32')) -> None:  # pylint: disable=undefined-loop-variable

    A_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    B_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    A_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    B_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    C_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("A_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared_local"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared_local[v0, v1] = B_shared[v0, v1]
    for by in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.y"}):
        for bx in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.x"}):
            for vy in T.parallel(0, 2, annotations={"thread_binding": "vthread.y"}):
                for vx in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                    for ty in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.y"}):
                        for tx in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                            for k_0 in T.serial(0, 256):
                                for k_1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with Ts.sblock("A_shared_local"):
                                            v0 = Ts.axis.S(2048, k_0 * 8 + k_1 + i)
                                            v1 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with Ts.sblock("C"):
                                            vi = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = Ts.axis.R(2048, k_0 * 8 + k_1)
                                            with Ts.init():
                                                C_local[vi, vj] = T.float32(0)
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with Ts.sblock("C_local"):
                                    v0 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1 = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]

@Ts.function
def cuda_matmul_3(A: T.Tensor([2048, 2048], 'float32'), B: T.Tensor([2048, 2048], 'float32'), C: T.Tensor([2048, 2048], 'float32')) -> None:  # pylint: disable=undefined-loop-variable

    A_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    B_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    A_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    B_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    C_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("A_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            A_shared[v0, v1] = A[v0, v1]
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for by in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.y"}):
        for bx in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.x"}):
            for vy in T.parallel(0, 2, annotations={"thread_binding": "vthread.y"}):
                for vx in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                    for ty in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.y"}):
                        for tx in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                            for k0 in T.serial(0, 256):
                                for k1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with Ts.sblock("A_shared_local"):
                                            v0 = Ts.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with Ts.sblock("B_shared_local"):
                                            v0 = Ts.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with Ts.sblock("C"):
                                            vi = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = Ts.axis.R(2048, k0 * 8 + k1)
                                            with Ts.init():
                                                C_local[vi, vj] = T.float32(0)
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with Ts.sblock("C_local"):
                                    v0 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1 = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]

@Ts.function
def cuda_matmul_4(A: T.Tensor([2048, 2048], 'float32'), B: T.Tensor([2048, 2048], 'float32'), C: T.Tensor([2048, 2048], 'float32')) -> None:  # pylint: disable=undefined-loop-variable

    A_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    B_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    A_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    B_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    C_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    for i, j in T.grid(2048, 2048):
        with Ts.sblock("B_shared"):
            v0, v1 = Ts.axis.remap("SS", [i, j])
            B_shared[v0, v1] = B[v0, v1]
    for by in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.y"}):
        for bx in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.x"}):
            for vy in T.parallel(0, 2, annotations={"thread_binding": "vthread.y"}):
                for vx in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                    for ty in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.y"}):
                        for tx in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                            for k0 in T.serial(0, 256):
                                for i, j in T.grid(8, 64):
                                    with Ts.sblock("A_shared"):
                                        v0 = Ts.axis.S(2048, k0 * 8 + i)
                                        v1 = Ts.axis.S(2048, by * 64 + j)
                                        A_shared[v0, v1] = A[v0, v1]
                                for k1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with Ts.sblock("A_shared_local"):
                                            v0 = Ts.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with Ts.sblock("B_shared_local"):
                                            v0 = Ts.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with Ts.sblock("C"):
                                            vi = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = Ts.axis.R(2048, k0 * 8 + k1)
                                            with Ts.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with Ts.sblock("C_local"):
                                    v0 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1 = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]

@Ts.function
def cuda_matmul_5(A: T.Tensor([2048, 2048], 'float32'), B: T.Tensor([2048, 2048], 'float32'), C: T.Tensor([2048, 2048], 'float32')) -> None:  # pylint: disable=undefined-loop-variable

    A_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    B_shared = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="shared")
    A_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    B_shared_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    C_local = Ts.sblock_alloc_tensor([2048, 2048], "float32", scope="local")
    for by in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.y"}):
        for bx in T.parallel(0, 32, annotations={"thread_binding": "blockIdx.x"}):
            for vy in T.parallel(0, 2, annotations={"thread_binding": "vthread.y"}):
                for vx in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                    for ty in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.y"}):
                        for tx in T.parallel(0, 8, annotations={"thread_binding": "threadIdx.x"}):
                            for k0 in T.serial(0, 256):
                                for i, j in T.grid(8, 64):
                                    with Ts.sblock("A_shared"):
                                        v0 = Ts.axis.S(2048, k0 * 8 + i)
                                        v1 = Ts.axis.S(2048, by * 64 + j)
                                        A_shared[v0, v1] = A[v0, v1]
                                for i, j in T.grid(8, 64):
                                    with Ts.sblock("B_shared"):
                                        v0 = Ts.axis.S(2048, k0 * 8 + i)
                                        v1 = Ts.axis.S(2048, bx * 64 + j)
                                        B_shared[v0, v1] = B[v0, v1]
                                for k1 in T.unroll(0, 8):
                                    for i, j in T.grid(1, 4):
                                        with Ts.sblock("A_shared_local"):
                                            v0 = Ts.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + j)
                                            A_shared_local[v0, v1] = A_shared[v0, v1]
                                    for i, j in T.grid(1, 4):
                                        with Ts.sblock("B_shared_local"):
                                            v0 = Ts.axis.S(2048, k0 * 8 + k1 + i)
                                            v1 = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            B_shared_local[v0, v1] = B_shared[v0, v1]
                                    for _, i, j in T.grid(1, 4, 4):
                                        with Ts.sblock("C"):
                                            vi = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                            vj = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                            vk = Ts.axis.R(2048, k0 * 8 + k1)
                                            with Ts.init():
                                                C_local[vi, vj] = 0.0
                                            C_local[vi, vj] = C_local[vi, vj] + A_shared_local[vk, vi] * B_shared_local[vk, vj]
                            for i, j in T.grid(4, 4):
                                with Ts.sblock("C_local"):
                                    v0 = Ts.axis.S(2048, by * 64 + vy * 32 + ty * 4 + i)
                                    v1 = Ts.axis.S(2048, bx * 64 + vx * 32 + tx * 4 + j)
                                    C[v0, v1] = C_local[v0, v1]

@Ts.function
def tiled(A: T.Tensor([128, 128], 'float32'), C: T.Tensor([128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([128, 128], "float32")

    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, i_0 * 16 + i_1)
            vj = Ts.axis.S(128, j_0 * 16 + j_1)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def tiled_after_reverse_compute_at(A: T.Tensor([128, 128], 'float32'), C: T.Tensor([128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([128, 128], "float32")

    for i_0, j_0, i_1 in T.grid(8, 8, 16):
        for j_1 in T.serial(0, 16):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i_0 * 16 + i_1)
                vj = Ts.axis.S(128, j_0 * 16 + j_1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j_1 in T.serial(0, 16):
            with Ts.sblock("C"):
                vi = Ts.axis.S(128, i_0 * 16 + i_1)
                vj = Ts.axis.S(128, j_0 * 16 + j_1)
                C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def tiled_trivial_binding(A: T.Tensor([1, 128, 128], 'float32'), C: T.Tensor([1, 128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([1, 128, 128], "float32")

    for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, i_0 * 16 + i_1)
            vj = Ts.axis.S(128, j_0 * 16 + j_1)
            B[0, vi, vj] = A[0, vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[0, vi, vj] = B[0, vi, vj] + 1.0

@Ts.function
def tiled_trivial_binding_after_reverse_compute_at(A: T.Tensor([1, 128, 128], 'float32'), C: T.Tensor([1, 128, 128], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([1, 128, 128], "float32")

    for i_0, j_0, i_1 in T.grid(8, 8, 16):
        for j_1 in T.serial(0, 16):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i_0 * 16 + i_1)
                vj = Ts.axis.S(128, j_0 * 16 + j_1)
                B[0, vi, vj] = A[0, vi, vj] * 2.0
        for j_1 in T.serial(0, 16):
            with Ts.sblock("C"):
                vi = Ts.axis.S(128, i_0 * 16 + i_1)
                vj = Ts.axis.S(128, j_0 * 16 + j_1)
                C[0, vi, vj] = B[0, vi, vj] + 1.0

@Ts.function
def factorized(A: T.Tensor([16, 16, 16], 'float32'), B: T.Tensor([16], 'float32')) -> None:

    B_rf_local = Ts.sblock_alloc_tensor([16, 16], "float32", scope="local")
    for j in T.parallel(0, 16, annotations={"thread_binding": "blockIdx.x"}):
        for i_o in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
            for i_i, k in T.grid(4, 16):
                with Ts.sblock("B_rf"):
                    vi = Ts.axis.S(16, i_o * 4 + i_i)
                    vj, vk = Ts.axis.remap("SR", [j, k])
                    with Ts.init():
                        B_rf_local[vi, vj] = 0.0
                    B_rf_local[vi, vj] = B_rf_local[vi, vj] + A[vj, vi, vk]
    for i, k in T.grid(16, 16):
        with Ts.sblock("B"):
            vi, vk = Ts.axis.remap("SR", [i, k])
            with Ts.init():
                B[vi] = 0.0
            B[vi] = B[vi] + B_rf_local[vk, vi]

@Ts.function
def factorized_after_reverse_compute_at(A: T.Tensor([16, 16, 16], 'float32'), B: T.Tensor([16], 'float32')) -> None:

    B_rf_local = Ts.sblock_alloc_tensor([16, 16], "float32", scope="local")
    for j in T.parallel(0, 16, annotations={"thread_binding": "blockIdx.x"}):
        for i_o in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
            for i_i, k in T.grid(4, 16):
                with Ts.sblock("B_rf"):
                    vi = Ts.axis.S(16, i_o * 4 + i_i)
                    vj = Ts.axis.S(16, j)
                    vk = Ts.axis.R(16, k)
                    with Ts.init():
                        B_rf_local[vi, vj] = 0.0
                    B_rf_local[vi, vj] = B_rf_local[vi, vj] + A[vj, vi, vk]
            for k in T.serial(0, 4):
                with Ts.sblock("B"):
                    vi = Ts.axis.S(16, j)
                    vk = Ts.axis.R(16, i_o * 4 + k)
                    with Ts.init():
                        B[vi] = 0.0
                    B[vi] = B[vi] + B_rf_local[vk, vi]

@Ts.function
def not_all_compact_data_flow(A: T.Tensor((128, 128), 'float32'), C: T.Tensor((128, 128), 'float32')):

    B = Ts.sblock_alloc_tensor((128, 128), "float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 64):
        with Ts.sblock("C_1"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj * 2] = B[vi, vj * 2] + 1.0
        with Ts.sblock("C_2"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj * 2 + 1] = B[vi, vj * 2 + 1] * 2.0

@Ts.function
def not_all_compact_data_flow_after_compute_at(A: T.Tensor((128, 128), 'float32'), C: T.Tensor((128, 128), 'float32')):

    B = Ts.sblock_alloc_tensor((128, 128), "float32")

    for i, j in T.grid(128, 64):
        for t in range(2):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i)
                vj = Ts.axis.S(128, j * 2 + t)
                B[vi, vj] = A[vi, vj]
        with Ts.sblock("C_1"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj * 2] = B[vi, vj * 2] + 1.0
        with Ts.sblock("C_2"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj * 2 + 1] = B[vi, vj * 2 + 1] * 2.0

@Ts.function
def fail_subtree_compact_dataflow(A: T.Tensor((128, 128), 'float32'), C: T.Tensor((128, 128), 'float32')) -> None:

    B = Ts.sblock_alloc_tensor((128, 128), "float32")

    for i in range(0, 128):
        for j in range(0, 64):
            with Ts.sblock("B_0"):
                vi = Ts.axis.S(128, i)
                vj = Ts.axis.S(128, j)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 64):
            with Ts.sblock("B_1"):
                vi = Ts.axis.S(128, i)
                vj = Ts.axis.S(128, j + 64)
                B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def fail_all_consumers_under_loop(A: T.Tensor((128, 128), 'float32'), C: T.Tensor((128, 128), 'float32'), D: T.Tensor((128, 128), 'float32')) -> None:

    B = Ts.sblock_alloc_tensor((128, 128), "float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("D"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def fail_all_producers_under_loop(A: T.Tensor((128, 128), 'float32'), D: T.Tensor((128, 128), 'float32')) -> None:

    B = Ts.sblock_alloc_tensor((128, 128), "float32")
    C = Ts.sblock_alloc_tensor((128, 128), "float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("D"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = B[vi, vj] + C[vi, vj]

@Ts.function
def read_out_of_bound(A: T.Tensor([16], 'float32'), C: T.Tensor([16], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([16], "float32")

    for i in T.serial(0, 16):
        with Ts.sblock("B"):
            v = Ts.axis.S(16, i)
            B[v] = A[v]
    for j in T.serial(0, 16):
        with Ts.sblock("C"):
            v = Ts.axis.S(16, j)
            Ts.reads(B[v : v + 2])
            C[v] = T.if_then_else(v < 15, T.max(B[v], B[v + 1]), B[v])

@Ts.function
def read_out_of_bound_after_compute_at(A: T.Tensor([16], 'float32'), C: T.Tensor([16], 'float32')) -> None:

    B = Ts.sblock_alloc_tensor([16], "float32")

    for j in T.serial(0, 16):
        for i in T.serial(0, 2):
            with Ts.sblock("B"):
                v = Ts.axis.S(16, j + i)
                Ts.where(j + i < 16)
                B[v] = A[v]
        with Ts.sblock("C"):
            v = Ts.axis.S(16, j)
            Ts.reads([B[v : v + 2]])
            C[v] = T.if_then_else(v < 15, T.max(B[v], B[v + 1]), B[v])

@Ts.function
def multi_reduction(A: T.Tensor((16, 16), "float32"), C: T.Tensor((), "float32")):
    B = Ts.sblock_alloc_tensor((16, ), dtype="float32")
    for i, k in T.grid(16, 16):
        with Ts.sblock("B"):
            vi, vk = Ts.axis.remap("SR", [i, k])
            with Ts.init():
                B[vi] = 0.0
            B[vi] += A[vi, vk]
    for k in T.grid(16):
        with Ts.sblock("C"):
            vk = Ts.axis.remap("R", [k])
            with Ts.init():
                C[()] = 0.0
            C[()] += B[vk]

@Ts.function
def multi_reduction_after_compute_at(
    A: T.Tensor((16, 16), "float32"),
    C:T.Tensor((), "float32"),
):
    B = Ts.sblock_alloc_tensor((16, ), dtype="float32")
    for k in T.grid(16):
        for kk in T.grid(16):
            with Ts.sblock("B"):
                vi, vk = Ts.axis.remap("SR", [k, kk])
                with Ts.init():
                    B[vi] = 0.0
                B[vi] += A[vi, vk]
        with Ts.sblock("C"):
            vk = Ts.axis.remap("R", [k])
            with Ts.init():
                C[()] = 0.0
            C[()] += B[vk]

@Ts.function
def tiled_pooling_read_cache(X: T.Tensor([224, 224], dtype='float32'), Y: T.Tensor([224, 224], dtype='float32')) -> None:

    cache = Ts.sblock_alloc_tensor([224, 224], dtype="float32")
    for hh, ww in T.grid(224, 224):
        with Ts.sblock("cache"):
            h, w = Ts.axis.remap("SS", [hh, ww])
            cache[h, w] = X[h, w]
    for hh_0, ww_0, hh_1, ww_1, khh, kww in T.grid(28, 28, 8, 8, 3, 3):
        with Ts.sblock("compute"):
            h = Ts.axis.spatial(224, hh_0 * 8 + hh_1)
            w = Ts.axis.spatial(224, ww_0 * 8 + ww_1)
            kh, kw = Ts.axis.remap("RR", [khh, kww])
            with Ts.init():
                Y[h, w] = 0.0
            Y[h, w] = T.max(Y[h, w], T.if_then_else(
                T.likely(1 <= h + kh) and \
                T.likely(h + kh < 225) and \
                T.likely(1 <= w + kw) and \
                T.likely(w + kw < 225),
                cache[h + kh - 1, w + kw - 1], 0.0))

@Ts.function
def tiled_pooling_read_cache_after_compute_at(X: T.Tensor([224, 224], dtype='float32'), Y: T.Tensor([224, 224], dtype='float32')) -> None:

    cache = Ts.sblock_alloc_tensor([224, 224], dtype="float32")
    for hh_0, ww_0 in T.grid(28, 28):
        for ax0, ax1 in T.grid(10, 10):
            with Ts.sblock("cache"):
                h = Ts.axis.spatial(224, hh_0 * 8 - 1 + ax0)
                w = Ts.axis.spatial(224, ww_0 * 8 - 1 + ax1)
                Ts.where(1 <= hh_0 * 8 + ax0 and hh_0 * 8 + ax0 < 225 and 1 <= ww_0 * 8 + ax1 and ww_0 * 8 + ax1 < 225)
                cache[h, w] = X[h, w]
        for hh_1, ww_1, khh, kww in T.grid(8, 8, 3, 3):
            with Ts.sblock("compute"):
                h = Ts.axis.spatial(224, hh_0 * 8 + hh_1)
                w = Ts.axis.spatial(224, ww_0 * 8 + ww_1)
                kh, kw = Ts.axis.remap("RR", [khh, kww])
                with Ts.init():
                    Y[h, w] = 0.0
                Y[h, w] = T.max(Y[h, w], T.if_then_else(
                    T.likely(1 <= h + kh) and \
                    T.likely(h + kh < 225) and \
                    T.likely(1 <= w + kw) and \
                    T.likely(w + kw < 225),
                    cache[h + kh - 1, w + kw - 1], 0.0))

@Ts.function
def non_uniform_tiled_conv(x: T.Tensor((1, 3, 100, 100), "float32"),
                           w: T.Tensor((16, 3, 3, 3), "float32"),
                           y: T.Tensor((1, 16, 98, 98), "float32")) -> None:
    x_global = Ts.sblock_alloc_tensor([1, 3, 100, 100], dtype="float32")
    for ax0, ax1, ax2, ax3 in T.grid(1, 3, 100, 100):
        with Ts.sblock("cache"):
            v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
    for h_o, w_o, n, c_o, h_i, w_i, c_i, kh, kw in T.grid(7, 7, 1, 16, 15, 15, 3, 3, 3):
        with Ts.sblock("compute"):
            nn = Ts.axis.spatial(1, 0)
            cc = Ts.axis.spatial(16, c_o)
            hh = Ts.axis.spatial(98, h_o * 15 + h_i)
            ww = Ts.axis.spatial(98, w_o * 15 + w_i)
            rc, rh, rw = Ts.axis.remap("RRR", [c_i, kh, kw])
            Ts.where(h_o * 15 + h_i < 98 and w_o * 15 + w_i < 98)
            with Ts.init():
                y[nn, cc, hh, ww] = T.float32(0)
            y[nn, cc, hh, ww] = y[nn, cc, hh, ww] + \
                x_global[nn, cc // 16 * 3 + rc, hh + rh, ww + rw] * w[cc, rc, rh, rw]

@Ts.function
def non_uniform_tiled_conv_after_compute_at(x: T.Tensor((1, 3, 100, 100), "float32"),
                                            w: T.Tensor((16, 3, 3, 3), "float32"),
                                            y: T.Tensor((1, 16, 98, 98), "float32")) -> None:
    x_global = Ts.sblock_alloc_tensor([1, 3, 100, 100], dtype="float32")
    for h_o, w_o in T.grid(7, 7):
        for ax0, ax1, ax2 in T.grid(3, 17, 17):
            with Ts.sblock("cache"):
                v0 = Ts.axis.spatial(1, 0)
                v1 = Ts.axis.spatial(3, ax0)
                v2 = Ts.axis.spatial(100, h_o * 15 + ax1)
                v3 = Ts.axis.spatial(100, w_o * 15 + ax2)
                Ts.where(h_o * 15 + ax1 < 100 and w_o * 15 + ax2 < 100)
                x_global[v0, v1, v2, v3] = x[v0, v1, v2, v3]
        for n, c_o, h_i, w_i, c_i, kh, kw in T.grid(1, 16, 15, 15, 3, 3, 3):
            with Ts.sblock("compute"):
                nn = Ts.axis.spatial(1, 0)
                cc = Ts.axis.spatial(16, c_o)
                hh = Ts.axis.spatial(98, h_o * 15 + h_i)
                ww = Ts.axis.spatial(98, w_o * 15 + w_i)
                rc, rh, rw = Ts.axis.remap("RRR", [c_i, kh, kw])
                Ts.where(h_o * 15 + h_i < 98 and w_o * 15 + w_i < 98)
                with Ts.init():
                    y[nn, cc, hh, ww] = T.float32(0)
                y[nn, cc, hh, ww] = y[nn, cc, hh, ww] + \
                    x_global[nn, cc // 16 * 3 + rc, hh + rh, ww + rw] * w[cc, rc, rh, rw]

@Ts.function
def concat_two_elemwise(x: T.Tensor((16,), "float32"),
                        y: T.Tensor((8,), "float32"),
                        T_concat: T.Tensor((24,), "float32")) -> None:
    T_add_1 = Ts.sblock_alloc_tensor([16], dtype="float32")
    T_add_2 = Ts.sblock_alloc_tensor([8], dtype="float32")
    for i in T.serial(16):
        with Ts.sblock("T_add_1"):
            ax = Ts.axis.spatial(16, i)
            T_add_1[ax] = x[ax] + T.float32(1)
    for i in T.serial(8):
        with Ts.sblock("T_add_2"):
            ax = Ts.axis.spatial(8, i)
            T_add_2[ax] = y[ax] + T.float32(2)
    for i in T.serial(24):
        with Ts.sblock("T_concat"):
            ax = Ts.axis.spatial(24, i)
            T_concat[ax] = T.if_then_else(16 <= ax, T_add_2[ax - 16], T_add_1[ax])

@Ts.function
def concat_two_elemwise_after_compute_at(x: T.Tensor((16,), "float32"),
                                         y: T.Tensor((8,), "float32"),
                                         T_concat: T.Tensor((24,), "float32")) -> None:
    T_add_1 = Ts.sblock_alloc_tensor([16], dtype="float32")
    T_add_2 = Ts.sblock_alloc_tensor([8], dtype="float32")
    for i in T.serial(24):
        with Ts.sblock("T_add_1"):
            ax = Ts.axis.spatial(16, i)
            Ts.where(i < 16)
            T_add_1[ax] = x[ax] + T.float32(1)
        with Ts.sblock("T_add_2"):
            ax = Ts.axis.spatial(8, i - 16)
            Ts.where(16 <= i)
            T_add_2[ax] = y[ax] + T.float32(2)
        with Ts.sblock("T_concat"):
            ax = Ts.axis.spatial(24, i)
            T_concat[ax] = T.if_then_else(16 <= ax, T_add_2[ax - 16], T_add_1[ax])

@Ts.function
def floordiv_and_floormod_indices(X: T.Tensor([16, 16], "float32"), Y: T.Tensor([256], "float32")) -> None:

    temp = Ts.sblock_alloc_tensor([16, 16])
    for i, j in T.grid(16, 16):
        with Ts.sblock("A"):
            v_i, v_j = Ts.axis.remap("SS", [i, j])
            temp[v_i, v_j] = X[v_j, v_i] + 1.0
    for i in T.serial(0, 256):
        with Ts.sblock("B"):
            v_i = Ts.axis.remap("S", [i])
            Y[v_i] = temp[v_i // 16, v_i % 16]

@Ts.function
def floordiv_and_floormod_indices_after_reverse_compute_at(X: T.Tensor([16, 16], dtype='float32'), Y: T.Tensor([256], dtype='float32')) -> None:

    temp = Ts.sblock_alloc_tensor([16, 16], dtype="float32")
    for i in T.serial(0, 16):
        for j in T.serial(0, 16):
            with Ts.sblock("A"):
                v_i, v_j = Ts.axis.remap("SS", [i, j])
                temp[v_i, v_j] = X[v_j, v_i] + T.float32(1)
        for ax0 in T.serial(0, 16):
            with Ts.sblock("B"):
                v_i = Ts.axis.spatial(256, i * 16 + ax0)
                Y[v_i] = temp[v_i // 16, v_i % 16]

@Ts.function
def recursive_floordiv_floormod(A: T.Tensor((16, 64, 1, 8, 8, 32), "float32"),
                                C: T.Tensor((3, 512, 512), "float32")) -> None:
    T.func_attr({"tirx.noalias": True})
    # with Ts.sblock("root"):
    B = Ts.sblock_alloc_tensor((1, 128, 16, 8, 2, 32, 2), "float32")
    for axis1, axis2, axis3, axis4, axis5, axis6, axis7 in T.grid(1, 128, 16, 8, 2, 32, 2):
        with Ts.sblock("In"):
            v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7 = Ts.axis.remap("SSSSSSS", [axis1, axis2, axis3, axis4, axis5, axis6, axis7])
            Ts.reads(A[(v_axis2 * 4 + v_axis5 * 2 + v_axis7) // 32, (v_axis3 * 32 + v_axis6) // 8, (v_axis1 * 8 + v_axis4) // 8, (v_axis3 * 32 + v_axis6) % 8, v_axis1 * 8 + v_axis4, (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 32])
            Ts.writes(B[v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7])
            B[v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7] = A[(v_axis2 * 4 + v_axis5 * 2 + v_axis7) // 32, (v_axis3 * 32 + v_axis6) // 8, (v_axis1 * 8 + v_axis4) // 8, (v_axis3 * 32 + v_axis6) % 8, v_axis1 * 8 + v_axis4, (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 32] + 3
    for ax1, ax2, ax3 in T.grid(3, 512, 512):
        with Ts.sblock("Out"):
            v1, v2, v3 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
            Ts.reads(B[v1 // 8, v2 // 4, v3 // 32, v1, v2 % 4 // 2, v3 % 32, v2 % 2])
            Ts.writes(C[v1, v2, v3])
            C[v1, v2, v3] = B[v1 // 8, v2 // 4, v3 // 32, v1, v2 % 4 // 2, v3 % 32, v2 % 2] * 2

@Ts.function
def recursive_floordiv_floormod_after_reverse_compute_at(A: T.Tensor((16, 64, 1, 8, 8, 32), "float32"), C: T.Tensor((3, 512, 512), "float32")) -> None:
    T.func_attr({"tirx.noalias": True})
    # with Ts.sblock("root"):
    B = Ts.sblock_alloc_tensor((1, 128, 16, 8, 2, 32, 2))
    for axis1, axis2, axis3 in T.grid(1, 128, 16):
        for axis4, axis5, axis6, axis7 in T.grid(8, 2, 32, 2):
            with Ts.sblock("In"):
                v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7 = Ts.axis.remap("SSSSSSS", [axis1, axis2, axis3, axis4, axis5, axis6, axis7])
                Ts.reads(A[(v_axis2 * 4 + v_axis5 * 2 + v_axis7) // 32, (v_axis3 * 32 + v_axis6) // 8, (v_axis1 * 8 + v_axis4) // 8, (v_axis3 * 32 + v_axis6) % 8, v_axis1 * 8 + v_axis4, (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 32])
                Ts.writes(B[v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7])
                B[v_axis1, v_axis2, v_axis3, v_axis4, v_axis5, v_axis6, v_axis7] = A[(v_axis2 * 4 + v_axis5 * 2 + v_axis7) // 32, (v_axis3 * 32 + v_axis6) // 8, (v_axis1 * 8 + v_axis4) // 8, (v_axis3 * 32 + v_axis6) % 8, v_axis1 * 8 + v_axis4, (v_axis2 * 4 + v_axis5 * 2 + v_axis7) % 32] + T.float32(3)
        for ax0, ax1, ax2 in T.grid(3, 4, 32):
            with Ts.sblock("Out"):
                v1 = Ts.axis.spatial(3, ax0)
                v2 = Ts.axis.spatial(512, axis2 * 4 + ax1)
                v3 = Ts.axis.spatial(512, axis3 * 32 + ax2)
                Ts.reads(B[v1 // 8, v2 // 4, v3 // 32, v1, v2 % 4 // 2, v3 % 32, v2 % 2])
                Ts.writes(C[v1, v2, v3])
                C[v1, v2, v3] = B[v1 // 8, v2 // 4, v3 // 32, v1, v2 % 4 // 2, v3 % 32, v2 % 2] * T.float32(2)

@Ts.function
def tiled_repeat_op(x: T.Tensor((4,), "float32"), T_repeat: T.Tensor((64,), "float32")) -> None:
    T_add = Ts.sblock_alloc_tensor([4], dtype="float32")
    for i0 in T.serial(4):
        with Ts.sblock("T_add"):
            ax0 = Ts.axis.spatial(4, i0)
            T_add[ax0] = x[ax0] + 1.0
    for i0_0, i0_1 in T.grid(8, 8):
        with Ts.sblock("T_repeat"):
            ax0 = Ts.axis.spatial(64, i0_0 * 8 + i0_1)
            T_repeat[ax0] = T_add[ax0 // 16]

@Ts.function
def tiled_repeat_op_after_compute_at(x: T.Tensor((4,), "float32"), T_repeat: T.Tensor((64,), "float32")) -> None:
    T_add = Ts.sblock_alloc_tensor([4], dtype="float32")
    for i0_0 in T.serial(8):
        with Ts.sblock("T_add"):
            ax0 = Ts.axis.spatial(4, i0_0 // 2)
            T_add[ax0] = x[ax0] + T.float32(1)
        for i0_1 in T.serial(8):
            with Ts.sblock("T_repeat"):
                ax0 = Ts.axis.spatial(64, i0_0 * 8 + i0_1)
                T_repeat[ax0] = T_add[ax0 // 16]

@Ts.function
def static_bound(A: T.Tensor((32, 1), "float32"), C: T.Tensor((32, 1), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((32, 1), "float32")
    for i, j in T.grid(32, 1):
        with Ts.sblock("B"):
            vi = Ts.axis.spatial(32, i)
            vj = Ts.axis.spatial(1, j)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(32, 32):
        with Ts.sblock("C"):
            vi = Ts.axis.spatial(32, i)
            vj = Ts.axis.spatial(1, j)
            Ts.where(j < 1)
            C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def static_bound_after_compute_at(A: T.Tensor((32, 1), "float32"), C: T.Tensor((32, 1), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((32, 1), "float32")
    for i in range(32):
        for ax0, ax1 in T.grid(1, 1):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(32, i + ax0)
                vj = Ts.axis.spatial(1, ax1)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(32):
            with Ts.sblock("C"):
                vi = Ts.axis.spatial(32, i)
                vj = Ts.axis.spatial(1, j)
                Ts.where(j < 1)
                C[vi, vj] = B[vi, vj] + 1.0
# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


def test_compute_at_two_elementwise(use_block_name):
    sch = tvm.s_tir.Schedule(two_elementwise, debug_mask="all")
    block = "B" if use_block_name else sch.get_sblock("B")
    loop, _ = sch.get_loops("C" if use_block_name else sch.get_sblock("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(two_elementwise_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_compute_at_blockized_1(use_block_name):
    sch = tvm.s_tir.Schedule(blockized_1, debug_mask="all")
    block = sch.get_sblock("B")
    _, loop = sch.get_loops(sch.get_sblock("C_outer"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(blockized_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=blockized_1)


def test_compute_at_blockized_2(use_block_name):
    sch = tvm.s_tir.Schedule(blockized_2, debug_mask="all")
    block = sch.get_sblock("B_outer")
    _, loop, _, _ = sch.get_loops(sch.get_sblock("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(blockized_2_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=blockized_2)


def test_compute_at_cuda_matmul_0(use_block_name):
    sch = tvm.s_tir.Schedule(cuda_matmul_0, debug_mask="all")
    block = sch.get_sblock("C")
    _, _, _, _, _, loop, _, _ = sch.get_loops(sch.get_sblock("C_local"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(cuda_matmul_0_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_0)


def test_compute_at_cuda_matmul_1(use_block_name):
    sch = tvm.s_tir.Schedule(cuda_matmul_1, debug_mask="all")
    block = sch.get_sblock("A_shared_local")
    _, _, _, _, _, _, _, loop, _, _, _ = sch.get_loops(sch.get_sblock("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(cuda_matmul_2, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_1)


def test_compute_at_cuda_matmul_2(use_block_name):
    sch = tvm.s_tir.Schedule(cuda_matmul_2, debug_mask="all")
    block = sch.get_sblock("B_shared_local")
    _, _, _, _, _, _, _, loop, _, _, _ = sch.get_loops(sch.get_sblock("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(cuda_matmul_3, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_2)


def test_compute_at_cuda_matmul_3(use_block_name):
    sch = tvm.s_tir.Schedule(cuda_matmul_3, debug_mask="all")
    block = sch.get_sblock("A_shared")
    _, _, _, _, _, _, loop, _, _, _, _ = sch.get_loops(sch.get_sblock("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(cuda_matmul_4, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_3)


def test_compute_at_cuda_matmul_4(use_block_name):
    sch = tvm.s_tir.Schedule(cuda_matmul_4, debug_mask="all")
    block = sch.get_sblock("B_shared")
    _, _, _, _, _, _, loop, _, _, _, _ = sch.get_loops(sch.get_sblock("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(cuda_matmul_5, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=cuda_matmul_4)


def test_compute_at_reduction_block(use_block_name):
    sch = tvm.s_tir.Schedule(multi_reduction, debug_mask="all")
    block = sch.get_sblock("B")
    (loop,) = sch.get_loops(sch.get_sblock("C"))
    sch.compute_at(block, loop, preserve_unit_loops=False)
    assert_structural_equal_ignore_global_symbol(multi_reduction_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=multi_reduction)


def test_compute_at_tiled_pooling_read_cache(use_block_name):
    sch = tvm.s_tir.Schedule(tiled_pooling_read_cache, debug_mask="all")
    compute = sch.get_sblock("compute")
    _, w_o, _, _, _, _ = sch.get_loops(compute)
    cache = sch.get_sblock("cache")
    sch.compute_at(cache, w_o)
    assert_structural_equal_ignore_global_symbol(
        tiled_pooling_read_cache_after_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=tiled_pooling_read_cache)


def test_compute_at_non_uniform_tiled_conv(use_block_name):
    sch = tvm.s_tir.Schedule(non_uniform_tiled_conv, debug_mask="all")
    compute = sch.get_sblock("compute")
    sch.compute_at(sch.get_sblock("cache"), sch.get_loops(compute)[1])
    assert_structural_equal_ignore_global_symbol(
        non_uniform_tiled_conv_after_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=non_uniform_tiled_conv)


def test_compute_at_concat(use_block_name):
    sch = tvm.s_tir.Schedule(concat_two_elemwise, debug_mask="all")
    concat = sch.get_sblock("T_concat")
    add1 = sch.get_sblock("T_add_1")
    add2 = sch.get_sblock("T_add_2")
    axis = sch.get_loops(concat)[0]
    sch.compute_at(add1, axis)
    sch.compute_at(add2, axis)
    assert_structural_equal_ignore_global_symbol(
        concat_two_elemwise_after_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=concat_two_elemwise)


def test_compute_at_tiled_repeat_op(use_block_name):
    sch = tvm.s_tir.Schedule(tiled_repeat_op, debug_mask="all")
    outer_ax, _ = sch.get_loops(sch.get_sblock("T_repeat"))
    sch.compute_at(sch.get_sblock("T_add"), outer_ax)
    assert_structural_equal_ignore_global_symbol(tiled_repeat_op_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=tiled_repeat_op)


def test_compute_at_rev_iter():
    @Ts.function
    def before(X: T.Tensor((10, 10), "float32"), Z: T.Tensor((10, 10), "float32")):
        Y = Ts.sblock_alloc_tensor([10, 10], "float32")
        for i, j in T.grid(10, 10):
            with Ts.sblock("b0"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Y[9 - vi, 9 - vj] = X[vi, vj] + 1.0
        for i, j in T.grid(10, 10):
            with Ts.sblock("b1"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vj, vi] + 2.0

    @Ts.function
    def after(X: T.Tensor((10, 10), "float32"), Z: T.Tensor((10, 10), "float32")):
        Y = Ts.sblock_alloc_tensor([10, 10], "float32")
        for i in range(10):
            for j in range(10):
                with Ts.sblock("b0"):
                    vi = Ts.axis.spatial(10, j)
                    vj = Ts.axis.spatial(10, 9 - i)
                    Y[9 - vi, 9 - vj] = X[vi, vj] + 1.0
            for j in range(10):
                with Ts.sblock("b1"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Z[vi, vj] = Y[vj, vi] + 2.0

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    axis = sch.get_loops(sch.get_sblock("b1"))[0]
    sch.compute_at(sch.get_sblock("b0"), axis)
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=before)


def test_reverse_compute_at_tiled(use_block_name):
    sch = tvm.s_tir.Schedule(tiled, debug_mask="all")
    block = sch.get_sblock("C")
    _, _, loop, _ = sch.get_loops(sch.get_sblock("B"))
    sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
    assert_structural_equal_ignore_global_symbol(tiled_after_reverse_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=tiled)


def test_reverse_compute_at_tiled_trivial_binding(use_block_name):
    sch = tvm.s_tir.Schedule(tiled_trivial_binding, debug_mask="all")
    block = sch.get_sblock("C")
    _, _, loop, _ = sch.get_loops(sch.get_sblock("B"))
    sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
    assert_structural_equal_ignore_global_symbol(
        tiled_trivial_binding_after_reverse_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=tiled_trivial_binding)


def test_reverse_compute_at_blockized_2(use_block_name):
    sch = tvm.s_tir.Schedule(blockized_2, debug_mask="all")
    block = sch.get_sblock("C")
    _, loop = sch.get_loops(sch.get_sblock("B_outer"))
    sch.reverse_compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(
        blockized_2_after_reverse_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=blockized_2)


def test_reverse_compute_at_factorized(use_block_name):
    sch = tvm.s_tir.Schedule(factorized, debug_mask="all")
    block = sch.get_sblock("B")
    _, loop, _, _ = sch.get_loops(sch.get_sblock("B_rf"))
    sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
    assert_structural_equal_ignore_global_symbol(
        factorized_after_reverse_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=factorized)


def test_reverse_compute_at_floordiv_and_floormod_indices(use_block_name):
    sch = tvm.s_tir.Schedule(floordiv_and_floormod_indices, debug_mask="all")
    A = sch.get_sblock("A")
    B = sch.get_sblock("B")
    sch.reverse_compute_at(B, sch.get_loops(A)[0])
    assert_structural_equal_ignore_global_symbol(
        floordiv_and_floormod_indices_after_reverse_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=floordiv_and_floormod_indices)


def test_reverse_compute_at_floordiv_and_floormod_recursive(use_block_name):
    sch = tvm.s_tir.Schedule(recursive_floordiv_floormod, debug_mask="all")
    write_block = sch.get_sblock("Out")
    sch.reverse_compute_at(write_block, sch.get_loops("In")[2])
    assert_structural_equal_ignore_global_symbol(
        recursive_floordiv_floormod_after_reverse_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=recursive_floordiv_floormod)


def test_read_out_of_bound(use_block_name):
    sch = tvm.s_tir.Schedule(read_out_of_bound, debug_mask="all")
    block = sch.get_sblock("B")
    (loop,) = sch.get_loops(sch.get_sblock("C"))
    sch.compute_at(block, loop)
    assert_structural_equal_ignore_global_symbol(
        read_out_of_bound_after_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=read_out_of_bound)


def test_compact_dataflow(use_block_name):
    sch = tvm.s_tir.Schedule(not_all_compact_data_flow, debug_mask="all")
    block = sch.get_sblock("B")
    _, loop = sch.get_loops(sch.get_sblock("C_1"))
    sch.compute_at(block, loop)
    assert_structural_equal_ignore_global_symbol(
        not_all_compact_data_flow_after_compute_at, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=not_all_compact_data_flow)


def test_compute_at_simplify_static_bound(use_block_name):
    sch = tvm.s_tir.Schedule(static_bound, debug_mask="all")
    block = sch.get_sblock("B")
    loop, _ = sch.get_loops(sch.get_sblock("C"))
    sch.compute_at(block, loop, preserve_unit_loops=True)
    assert_structural_equal_ignore_global_symbol(static_bound_after_compute_at, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=static_bound)


def test_compute_at_simplify_symbolic_predicate():
    @tvm_tirx.script.ir_module
    class Before:
        @Ts.function
        def main(
            X: T.Tensor((T.int64(8), n * 32), "float32"),  # noqa: F821
            Y: T.Tensor((T.int64(8), n * 32), "float32"),  # noqa: F821
            n: T.int64,
        ):
            for i, k in T.grid(T.int64(8), n * 32):
                with Ts.sblock("Y"):
                    vi, vk = Ts.axis.remap("SS", [i, k])
                    Y[vi, vk] = X[vi, vk]

    @tvm_tirx.script.ir_module
    class After:
        @Ts.function
        def main(
            X: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
            Y: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
            n: T.int64,
        ):
            X_global = Ts.sblock_alloc_tensor((T.int64(8), n * T.int64(32)))

            for i, k_0 in T.grid(T.int64(8), n):
                for ax0 in range(T.int64(32)):
                    with Ts.sblock("X_global"):
                        v0 = Ts.axis.spatial(T.int64(8), i)
                        v1 = Ts.axis.spatial(n * T.int64(32), k_0 * T.int64(32) + ax0)
                        X_global[v0, v1] = X[v0, v1]
                for k_1 in range(T.int64(32)):
                    with Ts.sblock("Y"):
                        vi = Ts.axis.spatial(T.int64(8), i)
                        vk = Ts.axis.spatial(n * T.int64(32), k_0 * T.int64(32) + k_1)
                        Y[vi, vk] = X_global[vi, vk]

    sch = tvm.s_tir.Schedule(Before, debug_mask="all")
    block = sch.get_sblock("Y")
    i, k = sch.get_loops(sch.get_sblock("Y"))
    ko, ki = sch.split(k, [None, 32])
    XX = sch.cache_read(block, 0, "global")
    sch.compute_at(XX, ko)
    tvm_tirx.ir.assert_structural_equal(sch.mod, After)


def test_compute_at_non_perfect_channel_group(use_block_name):
    @Ts.function
    def grouped_channel_bias(
        X: T.Tensor((720, 8, 8), "float32"), Y: T.Tensor((720, 8, 8), "float32")
    ):
        B = Ts.sblock_alloc_tensor([45], dtype="float32", scope="global")
        for i in T.grid(45):
            with Ts.sblock("init"):
                vi = Ts.axis.remap("S", [i])
                B[vi] = vi
        for c_o, h, w, c_i in T.grid(2, 8, 8, 360):
            with Ts.sblock("compute"):
                hh, ww = Ts.axis.remap("SS", [h, w])
                cc = Ts.axis.spatial(720, c_o * 360 + c_i)
                Y[cc, hh, ww] = X[cc, hh, ww] + B[cc // 16]

    @Ts.function
    def grouped_channel_bias_non_perfect_tiled(
        X: T.Tensor((720, 8, 8), "float32"), Y: T.Tensor((720, 8, 8), "float32")
    ):
        B = Ts.sblock_alloc_tensor([45], dtype="float32")
        for c_o in range(2):
            for ax0 in range(23):
                with Ts.sblock("init"):
                    vi = Ts.axis.spatial(45, c_o * 22 + ax0)
                    B[vi] = vi
            for h, w, c_i in T.grid(8, 8, 360):
                with Ts.sblock("compute"):
                    hh, ww = Ts.axis.remap("SS", [h, w])
                    cc = Ts.axis.spatial(720, c_o * 360 + c_i)
                    Y[cc, hh, ww] = X[cc, hh, ww] + B[cc // 16]

    sch = tvm.s_tir.Schedule(grouped_channel_bias, debug_mask="all")
    loop = sch.get_loops(sch.get_sblock("compute"))[0]
    sch.compute_at(sch.get_sblock("init"), loop)
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], grouped_channel_bias_non_perfect_tiled
    )


def test_fail_subtree_complete_block(use_block_name):
    sch = tvm.s_tir.Schedule(fail_subtree_compact_dataflow, debug_mask="all")
    block = sch.get_sblock("B_0")
    loop, _ = sch.get_loops(sch.get_sblock("C"))
    with pytest.raises(tvm.s_tir.ScheduleError, match="complete block"):
        sch.compute_at(block, loop)


def test_fail_not_in_same_scope(use_block_name):
    sch = tvm.s_tir.Schedule(blockized_1, debug_mask="all")
    block = "B" if use_block_name else sch.get_sblock("B")
    loop, _ = sch.get_loops(sch.get_sblock("C_inner"))
    with pytest.raises(tvm.s_tir.ScheduleError, match="same block scope"):
        sch.compute_at(block, loop)


def test_fail_loop_is_ancestor_of_block(use_block_name):
    sch = tvm.s_tir.Schedule(two_elementwise, debug_mask="all")
    block = "B" if use_block_name else sch.get_sblock("B")
    loop, _ = sch.get_loops(sch.get_sblock("B"))
    with pytest.raises(tvm.s_tir.ScheduleError, match="ancestor of block"):
        sch.compute_at(block, loop)


def test_fail_output_block(use_block_name):
    sch = tvm.s_tir.Schedule(tiled, debug_mask="all")
    block = "C" if use_block_name else sch.get_sblock("C")
    loop, _, _, _ = sch.get_loops(sch.get_sblock("B"))
    with pytest.raises(tvm.s_tir.ScheduleError, match="output block"):
        sch.compute_at(block, loop)


def test_fail_all_consumers_under_loop(use_block_name):
    sch = tvm.s_tir.Schedule(fail_all_consumers_under_loop, debug_mask="all")
    block = "B" if use_block_name else sch.get_sblock("B")
    loop, _ = sch.get_loops(sch.get_sblock("C"))
    with pytest.raises(tvm.s_tir.ScheduleError, match="requires all the consumer"):
        sch.compute_at(block, loop)


def test_fail_all_producers_under_loop(use_block_name):
    sch = tvm.s_tir.Schedule(fail_all_producers_under_loop, debug_mask="all")
    block = "D" if use_block_name else sch.get_sblock("D")
    loop, _ = sch.get_loops(sch.get_sblock("C"))
    with pytest.raises(tvm.s_tir.ScheduleError, match="requires all the producer"):
        sch.reverse_compute_at(block, loop)


def test_compute_at_int64_loop(use_block_name):
    def _create_function():
        n = te.var("n", dtype="int64")
        m = te.var("m", dtype="int64")
        A = te.placeholder((n, m), name="A", dtype="float32")
        B = te.placeholder((n, m), name="B", dtype="float32")
        C = te.compute((n, m), lambda i, j: A[i, j] + B[i, j], name="C")
        D = te.compute((n, m), lambda i, j: C[i, j] + 1.0, name="D")
        return te.create_function([A, B, D])

    mod = _create_function()
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    block_d = "D" if use_block_name else sch.get_sblock("D")
    i, _ = sch.get_loops(block_d)
    sch.compute_at(block_c, i)
    verify_trace_roundtrip(sch=sch, mod=mod)


def test_compute_at_to_index():
    @Ts.function
    def multi_producers_conv(
        data: T.Tensor((1, 3, 224, 224), "int8"),
        w: T.Tensor((16, 3, 7, 7), "int8"),
        conv: T.Tensor((1, 16, 112, 112), "int32"),
    ) -> None:
        pad = Ts.sblock_alloc_tensor([1, 3, 230, 230], dtype="int8")
        wbuf = Ts.sblock_alloc_tensor([16, 3, 7, 7], dtype="int8")
        for i0, i1, i2, i3 in T.grid(1, 3, 230, 230):
            with Ts.sblock("pad"):
                i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                Ts.reads(data[i0_1, i1_1, i2_1 - 3, i3_1 - 3])
                Ts.writes(pad[i0_1, i1_1, i2_1, i3_1])
                pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                    3 <= i2_1 and i2_1 < 227 and 3 <= i3_1 and i3_1 < 227,
                    data[i0_1, i1_1, i2_1 - 3, i3_1 - 3],
                    T.int8(0),
                )
        for i0 in T.serial(1):
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 7, 7):
                with Ts.sblock("wbuf"):
                    v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w[v0, v1, v2, v3])
                    Ts.writes(wbuf[v0, v1, v2, v3])
                    wbuf[v0, v1, v2, v3] = w[v0, v1, v2, v3]
            for i1, i2, i3, i4, i5, i6 in T.grid(16, 112, 112, 3, 7, 7):
                with Ts.sblock("conv"):
                    nn, ff, yy, xx, rc, ry, rx = Ts.axis.remap(
                        "SSSSRRR", [i0, i1, i2, i3, i4, i5, i6]
                    )
                    Ts.reads(pad[nn, rc, yy * 2 + ry, xx * 2 + rx], wbuf[ff, rc, ry, rx])
                    Ts.writes(conv[nn, ff, yy, xx])
                    with Ts.init():
                        conv[nn, ff, yy, xx] = 0
                    conv[nn, ff, yy, xx] = conv[nn, ff, yy, xx] + T.cast(
                        pad[nn, rc, yy * 2 + ry, xx * 2 + rx], ty="int32"
                    ) * T.cast(wbuf[ff, rc, ry, rx], ty="int32")

    @Ts.function
    def multi_producers_after_compute_at(
        data: T.Tensor((1, 3, 224, 224), "int8"),
        w: T.Tensor((16, 3, 7, 7), "int8"),
        conv: T.Tensor((1, 16, 112, 112), "int32"),
    ) -> None:
        pad = Ts.sblock_alloc_tensor([1, 3, 230, 230], dtype="int8")
        wbuf = Ts.sblock_alloc_tensor([16, 3, 7, 7], dtype="int8")
        for i0 in T.serial(1):
            for ax0, ax1, ax2 in T.grid(3, 229, 229):
                with Ts.sblock("pad"):
                    i0_1 = Ts.axis.spatial(1, 0)
                    i1_1 = Ts.axis.spatial(3, ax0)
                    i2_1 = Ts.axis.spatial(230, ax1)
                    i3_1 = Ts.axis.spatial(230, ax2)
                    Ts.reads(data[i0_1, i1_1, i2_1 - 3, i3_1 - 3])
                    Ts.writes(pad[i0_1, i1_1, i2_1, i3_1])
                    pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                        3 <= i2_1 and i2_1 < 227 and 3 <= i3_1 and i3_1 < 227,
                        data[i0_1, i1_1, i2_1 - 3, i3_1 - 3],
                        T.int8(0),
                    )
            for ax0, ax1, ax2, ax3 in T.grid(16, 3, 7, 7):
                with Ts.sblock("wbuf"):
                    v0, v1, v2, v3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(w[v0, v1, v2, v3])
                    Ts.writes(wbuf[v0, v1, v2, v3])
                    wbuf[v0, v1, v2, v3] = w[v0, v1, v2, v3]
            for i1, i2, i3, i4, i5, i6 in T.grid(16, 112, 112, 3, 7, 7):
                with Ts.sblock("conv"):
                    nn, ff, yy, xx, rc, ry, rx = Ts.axis.remap(
                        "SSSSRRR", [i0, i1, i2, i3, i4, i5, i6]
                    )
                    Ts.reads(pad[nn, rc, yy * 2 + ry, xx * 2 + rx], wbuf[ff, rc, ry, rx])
                    Ts.writes(conv[nn, ff, yy, xx])
                    with Ts.init():
                        conv[nn, ff, yy, xx] = 0
                    conv[nn, ff, yy, xx] = conv[nn, ff, yy, xx] + T.cast(
                        pad[nn, rc, yy * 2 + ry, xx * 2 + rx], ty="int32"
                    ) * T.cast(wbuf[ff, rc, ry, rx], ty="int32")

    sch = tvm.s_tir.Schedule(multi_producers_conv, debug_mask="all")
    block_c = sch.get_sblock("pad")
    axis = sch.get_loops("conv")[0]
    sch.compute_at(block_c, axis, index=-2)
    assert_structural_equal_ignore_global_symbol(multi_producers_after_compute_at, sch.mod["main"])


def test_reverse_compute_at_to_index():
    @Ts.function
    def main(A: T.Tensor((128, 128), "float32"), D: T.Tensor((128, 128), "float32")) -> None:
        B = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        C = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i_0, j_0, i_1 in T.grid(8, 8, 16):
            for j_1 in T.serial(16):
                with Ts.sblock("B"):
                    vi = Ts.axis.spatial(128, i_0 * 16 + i_1)
                    vj = Ts.axis.spatial(128, j_0 * 16 + j_1)
                    Ts.reads(A[vi, vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * T.float32(2)
            for ax0 in T.serial(16):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(128, i_0 * 16 + i_1)
                    vj = Ts.axis.spatial(128, j_0 * 16 + ax0)
                    Ts.reads(B[vi, vj])
                    Ts.writes(C[vi, vj])
                    C[vi, vj] = B[vi, vj] + T.float32(1)
        for i, j in T.grid(128, 128):
            with Ts.sblock("D"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(B[vi, vj])
                Ts.writes(D[vi, vj])
                D[vi, vj] = B[vi, vj] + T.float32(1)

    @Ts.function
    def main_reverse_compute_at(
        A: T.Tensor((128, 128), "float32"), D: T.Tensor((128, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        C = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i_0, j_0, i_1 in T.grid(8, 8, 16):
            for j_1 in T.serial(16):
                with Ts.sblock("B"):
                    vi = Ts.axis.spatial(128, i_0 * 16 + i_1)
                    vj = Ts.axis.spatial(128, j_0 * 16 + j_1)
                    Ts.reads(A[vi, vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * T.float32(2)
            for ax0 in T.serial(16):
                with Ts.sblock("D"):
                    vi = Ts.axis.spatial(128, i_0 * 16 + i_1)
                    vj = Ts.axis.spatial(128, j_0 * 16 + ax0)
                    Ts.reads(B[vi, vj])
                    Ts.writes(D[vi, vj])
                    D[vi, vj] = B[vi, vj] + T.float32(1)
            for ax0 in T.serial(16):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(128, i_0 * 16 + i_1)
                    vj = Ts.axis.spatial(128, j_0 * 16 + ax0)
                    Ts.reads(B[vi, vj])
                    Ts.writes(C[vi, vj])
                    C[vi, vj] = B[vi, vj] + T.float32(1)

    sch = tvm.s_tir.Schedule(main, debug_mask="all")
    block_c = sch.get_sblock("D")
    axis = sch.get_loops("B")[2]
    sch.reverse_compute_at(block_c, axis, index=1)
    assert_structural_equal_ignore_global_symbol(main_reverse_compute_at, sch.mod["main"])


def test_reverse_compute_at_with_unit_loop():
    @Ts.function
    def main(A: T.Tensor((128, 128), "float32"), D: T.Tensor((1, 2, 1), "float32")) -> None:
        B = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i_0, j_0, i_1 in T.grid(T.int64(8), T.int64(8), T.int64(16)):
            for j_1 in T.serial(T.int64(16)):
                with Ts.sblock("B"):
                    vi = Ts.axis.spatial(T.int64(128), i_0 * T.int64(16) + i_1)
                    vj = Ts.axis.spatial(T.int64(128), j_0 * T.int64(16) + j_1)
                    Ts.reads(A[vi, vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * T.float32(2)
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(2), T.int64(1)):
            with Ts.sblock("D"):
                v0, v1, v2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                Ts.reads(B[v0, v1])
                Ts.writes(D[v0, v1, v2])
                D[v0, v1, v2] = B[v0, v1] + T.float32(1)

    @Ts.function
    def main_reverse_compute_at(
        A: T.Tensor((128, 128), "float32"), D: T.Tensor((1, 2, 1), "float32")
    ):
        B = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i_0, j_0, i_1 in T.grid(T.int64(8), T.int64(8), T.int64(16)):
            for j_1 in T.serial(T.int64(16)):
                with Ts.sblock("B"):
                    vi = Ts.axis.spatial(T.int64(128), i_0 * T.int64(16) + i_1)
                    vj = Ts.axis.spatial(T.int64(128), j_0 * T.int64(16) + j_1)
                    Ts.reads(A[vi, vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = A[vi, vj] * T.float32(2)
            for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(16), T.int64(1)):
                with Ts.sblock("D"):
                    Ts.where(
                        i_0 * T.int64(16) + i_1 < T.int64(1)
                        and j_0 * T.int64(16) + ax1 < T.int64(2)
                    )
                    v0 = Ts.axis.spatial(T.int64(1), i_0 * T.int64(16) + i_1 + ax0)
                    v1 = Ts.axis.spatial(T.int64(2), j_0 * T.int64(16) + ax1)
                    v2 = Ts.axis.spatial(T.int64(1), ax2)
                    Ts.reads(B[v0, v1])
                    Ts.writes(D[v0, v1, v2])
                    D[v0, v1, v2] = B[v0, v1] + T.float32(1)

    sch = tvm.s_tir.Schedule(main, debug_mask="all")
    block_d = sch.get_sblock("D")
    axis = sch.get_loops("B")[2]
    sch.reverse_compute_at(block_d, axis, preserve_unit_loops=True, index=1)
    assert_structural_equal_ignore_global_symbol(main_reverse_compute_at, sch.mod["main"])


def test_reverse_compute_at_layout_trans():
    @Ts.function
    def before(A: T.Tensor((1, 3, 5, 5, 16), "float32"), C: T.Tensor((1, 6, 5, 5, 8), "float32")):
        B = Ts.sblock_alloc_tensor((1, 3, 5, 5, 16))
        for i0, i1, i2, i3, i4 in T.grid(1, 3, 5, 5, 16):
            with Ts.sblock("compute"):
                v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                B[v_i0, v_i1, v_i2, v_i3, v_i4] = A[v_i0, v_i1, v_i2, v_i3, v_i4] + T.float32(1)
        for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 6, 5, 5, 8):
            with Ts.sblock("T_layout_trans"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = Ts.axis.remap(
                    "SSSSS", [ax0, ax1, ax2, ax3, ax4]
                )
                C[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = B[
                    v_ax0, (v_ax1 * 8 + v_ax4) // 16, v_ax2, v_ax3, (v_ax1 * 8 + v_ax4) % 16
                ]

    @Ts.function
    def after(A: T.Tensor((1, 3, 5, 5, 16), "float32"), C: T.Tensor((1, 6, 5, 5, 8), "float32")):
        B = Ts.sblock_alloc_tensor((1, 3, 5, 5, 16))
        for i0, i1 in T.grid(1, 3):
            for i2, i3, i4 in T.grid(5, 5, 16):
                with Ts.sblock("compute"):
                    v_i0, v_i1, v_i2, v_i3, v_i4 = Ts.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    B[v_i0, v_i1, v_i2, v_i3, v_i4] = A[v_i0, v_i1, v_i2, v_i3, v_i4] + T.float32(1)
            for ax0, ax1, ax2, ax3 in T.grid(2, 5, 5, 8):
                with Ts.sblock("T_layout_trans"):
                    v_ax0 = Ts.axis.spatial(1, 0)
                    v_ax1 = Ts.axis.spatial(6, i1 * 2 + ax0)
                    v_ax2, v_ax3, v_ax4 = Ts.axis.remap("SSS", [ax1, ax2, ax3])
                    C[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = B[
                        v_ax0, (v_ax1 * 8 + v_ax4) // 16, v_ax2, v_ax3, (v_ax1 * 8 + v_ax4) % 16
                    ]

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    trans = sch.get_sblock("T_layout_trans")
    axis = sch.get_loops("compute")[1]
    sch.reverse_compute_at(trans, axis)
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=before)


def test_shape_var_as_bound():
    # fmt: off
    n = T.dynamic("n", "int32")

    @Ts.function
    def before(A: T.Tensor((32, 1, 128), "float32"), B: T.Tensor((32, n, 128), "float32"), C: T.Tensor((32, 1, n), "float32")):

        # with Ts.sblock("root"):
        C_rf = Ts.sblock_alloc_tensor((128, 32, 1, n))
        for ax0_ax1_fused, ax2_fused_1, ax2_fused_0 in T.grid(n * 32, 128, 1):
            with Ts.sblock("NT_matmul_rf"):
                vax2_fused_1 = Ts.axis.spatial(128, ax2_fused_1)
                v0 = Ts.axis.spatial(32, ax0_ax1_fused // n)
                v1 = Ts.axis.spatial(n, ax0_ax1_fused % n)
                vax2_fused_0 = Ts.axis.reduce(1, ax2_fused_0)
                Ts.reads(A[v0, 0, vax2_fused_0 * 128 + vax2_fused_1], B[v0, v1, vax2_fused_0 * 128 + vax2_fused_1])
                Ts.writes(C_rf[vax2_fused_1, v0, 0, v1])
                with Ts.init():
                    C_rf[vax2_fused_1, v0, 0, v1] = T.float32(0)
                C_rf[vax2_fused_1, v0, 0, v1] = C_rf[vax2_fused_1, v0, 0, v1] + A[v0, 0, vax2_fused_0 * 128 + vax2_fused_1] * B[v0, v1, vax2_fused_0 * 128 + vax2_fused_1]
        for ax0_ax1_fused, ax2_fused_1 in T.grid(n * 32, 128):
            with Ts.sblock("NT_matmul"):
                vax2_fused_1 = Ts.axis.reduce(128, ax2_fused_1)
                v0 = Ts.axis.spatial(32, ax0_ax1_fused // n)
                v1 = Ts.axis.spatial(n, ax0_ax1_fused % n)
                Ts.reads(C_rf[vax2_fused_1, v0, 0, v1])
                Ts.writes(C[v0, 0, v1])
                with Ts.init():
                    C[v0, 0, v1] = T.float32(0)
                C[v0, 0, v1] = C[v0, 0, v1] + C_rf[vax2_fused_1, v0, 0, v1]

    n = T.dynamic("n", "int32")

    @Ts.function
    def expected(A: T.Tensor((32, 1, 128), "float32"), B: T.Tensor((32, n, 128), "float32"), C: T.Tensor((32, 1, n), "float32")):

        # with Ts.sblock("root"):
        C_rf = Ts.sblock_alloc_tensor((128, 32, 1, n))
        for ax0_ax1_fused in range(n * 32):
            for ax2_fused_1, ax2_fused_0 in T.grid(128, 1):
                with Ts.sblock("NT_matmul_rf"):
                    vax2_fused_1 = Ts.axis.spatial(128, ax2_fused_1)
                    v0 = Ts.axis.spatial(32, ax0_ax1_fused // n)
                    v1 = Ts.axis.spatial(n, ax0_ax1_fused % n)
                    vax2_fused_0 = Ts.axis.reduce(1, ax2_fused_0)
                    Ts.reads(A[v0, 0, vax2_fused_0 * 128 + vax2_fused_1], B[v0, v1, vax2_fused_0 * 128 + vax2_fused_1])
                    Ts.writes(C_rf[vax2_fused_1, v0, 0, v1])
                    with Ts.init():
                        C_rf[vax2_fused_1, v0, 0, v1] = T.float32(0)
                    C_rf[vax2_fused_1, v0, 0, v1] = C_rf[vax2_fused_1, v0, 0, v1] + A[v0, 0, vax2_fused_0 * 128 + vax2_fused_1] * B[v0, v1, vax2_fused_0 * 128 + vax2_fused_1]
            for ax0, ax1, ax2 in T.grid(128, 1, 1):
                with Ts.sblock("NT_matmul"):
                    vax2_fused_1 = Ts.axis.reduce(128, ax0)
                    v0 = Ts.axis.spatial(32, ax0_ax1_fused // n + ax1)
                    v1 = Ts.axis.spatial(n, ax0_ax1_fused % n + ax2)
                    Ts.reads(C_rf[vax2_fused_1, v0, 0, v1])
                    Ts.writes(C[v0, 0, v1])
                    with Ts.init():
                        C[v0, 0, v1] = T.float32(0)
                    C[v0, 0, v1] = C[v0, 0, v1] + C_rf[vax2_fused_1, v0, 0, v1]
    # fmt: on
    sch = tvm.s_tir.Schedule(before.with_attr("global_symbol", "main"), debug_mask="all")
    block = sch.get_sblock("NT_matmul")
    loop, _, _ = sch.get_loops(sch.get_sblock("NT_matmul_rf"))
    sch.reverse_compute_at(block, loop, preserve_unit_loops=True)
    tvm_tirx.ir.assert_structural_equal(
        sch.mod["main"], expected.with_attr("global_symbol", "main"), True
    )


def test_compute_at_sliced_concatenate():
    @Ts.function
    def before():
        X = Ts.sblock_alloc_tensor((1, 16, 28, 64), "float32")
        Y = Ts.sblock_alloc_tensor((1, 32, 28, 64), "float32")
        Z = Ts.sblock_alloc_tensor((1, 53, 28, 64), "float32")
        Concat = Ts.sblock_alloc_tensor((1, 101, 28, 64), "float32")
        Slice = Ts.sblock_alloc_tensor((1, 87, 28, 64), "float32")
        for ax0, ax1, ax2, ax3 in T.grid(1, 16, 28, 64):
            with Ts.sblock("compute"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                X[v_ax0, v_ax1, v_ax2, v_ax3] = 1.0
        for ax0, ax1, ax2, ax3 in T.grid(1, 101, 28, 64):
            with Ts.sblock("T_concat"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(
                    85 <= v_ax1,
                    X[v_ax0, v_ax1 - 85, v_ax2, v_ax3],
                    T.if_then_else(
                        53 <= v_ax1,
                        Y[v_ax0, v_ax1 - 53, v_ax2, v_ax3],
                        Z[v_ax0, v_ax1, v_ax2, v_ax3],
                    ),
                )
        for ax0, ax1, ax2, ax3 in T.grid(1, 87, 28, 64):
            with Ts.sblock("T_strided_slice"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Slice[v_ax0, v_ax1, v_ax2, v_ax3] = Concat[v_ax0, v_ax1, v_ax2, v_ax3]

    @Ts.function
    def expect():
        X = Ts.sblock_alloc_tensor((1, 16, 28, 64))
        Y = Ts.sblock_alloc_tensor((1, 32, 28, 64))
        Z = Ts.sblock_alloc_tensor((1, 53, 28, 64))
        Concat = Ts.sblock_alloc_tensor((1, 101, 28, 64))
        Slice = Ts.sblock_alloc_tensor((1, 87, 28, 64))
        for ax0 in range(1):
            for ax0_1, ax1, ax2 in T.grid(2, 28, 64):
                with Ts.sblock("compute"):
                    v_ax0 = Ts.axis.spatial(1, 0)
                    v_ax1 = Ts.axis.spatial(16, ax0_1)
                    v_ax2, v_ax3 = Ts.axis.remap("SS", [ax1, ax2])
                    X[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(1)
            for ax0_1, ax1, ax2 in T.grid(87, 28, 64):
                with Ts.sblock("T_concat"):
                    v_ax0 = Ts.axis.spatial(1, 0)
                    v_ax1 = Ts.axis.spatial(101, ax0_1)
                    v_ax2, v_ax3 = Ts.axis.remap("SS", [ax1, ax2])
                    Concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(
                        85 <= v_ax1,
                        X[v_ax0, v_ax1 - 85, v_ax2, v_ax3],
                        T.if_then_else(
                            53 <= v_ax1,
                            Y[v_ax0, v_ax1 - 53, v_ax2, v_ax3],
                            Z[v_ax0, v_ax1, v_ax2, v_ax3],
                        ),
                    )
            for ax1, ax2, ax3 in T.grid(87, 28, 64):
                with Ts.sblock("T_strided_slice"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Slice[v_ax0, v_ax1, v_ax2, v_ax3] = Concat[v_ax0, v_ax1, v_ax2, v_ax3]

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    blk1 = sch.get_sblock("compute")
    blk2 = sch.get_sblock("T_concat")
    blk3 = sch.get_sblock("T_strided_slice")
    loop = sch.get_loops(blk3)[0]
    sch.compute_at(blk2, loop)
    sch.compute_at(blk1, loop)
    after = sch.mod["main"]
    assert_structural_equal_ignore_global_symbol(expect, after)
    verify_trace_roundtrip(sch=sch, mod=before)


def _llvm_allclose(func, inputs, expected):
    if not tvm_tirx.runtime.enabled("llvm"):
        return
    compiled = tvm.compile(func, target="llvm")
    outputs = []
    args = []
    for array in inputs:
        args.append(tvm_tirx.runtime.tensor(array))
    for array in expected:
        buf = tvm_tirx.runtime.tensor(np.zeros_like(array))
        outputs.append(buf)
        args.append(buf)
    compiled(*args)
    for got, ref in zip(outputs, expected):
        tvm_tirx.testing.assert_allclose(got.numpy(), ref)


def test_compute_at_rfactor_keeps_tail_predicate():
    """`split` of a non-divisible reduction stores the tail on the block.

    `rfactor` keeps that guard, and `compute_at` must keep it too. Rebuilding
    the rfactor block from its write region alone drops the guard and reads
    past the input. `debug_mask` stays off: `rfactor` still records affine and
    region-cover flags that a fresh analysis rejects for this predicate.
    """

    @Ts.function
    def before(X: T.Tensor((4, 77), "float32"), Y: T.Tensor((4,), "float32")):
        for r, k in T.grid(4, 77):
            with Ts.sblock("Y"):
                v_r, v_k = Ts.axis.remap("SR", [r, k])
                with Ts.init():
                    Y[v_r] = T.float32(0)
                Y[v_r] = Y[v_r] + X[v_r, v_k]

    @Ts.function
    def expected(X: T.Tensor((4, 77), "float32"), Y: T.Tensor((4,), "float32")):
        Y_rf = Ts.sblock_alloc_tensor((64, 4), "float32")
        for r, k_1 in T.grid(4, 64):
            for ax0 in range(2):
                with Ts.sblock("Y_rf"):
                    vk_1 = Ts.axis.spatial(64, k_1)
                    v_r = Ts.axis.spatial(4, r)
                    vk_0 = Ts.axis.reduce(2, ax0)
                    Ts.where(ax0 * 64 + k_1 < 77)
                    with Ts.init():
                        Y_rf[vk_1, v_r] = T.float32(0)
                    Y_rf[vk_1, v_r] = Y_rf[vk_1, v_r] + X[v_r, vk_0 * 64 + vk_1]
            with Ts.sblock("Y"):
                vk_1 = Ts.axis.reduce(64, k_1)
                v_r = Ts.axis.spatial(4, r)
                with Ts.init():
                    Y[v_r] = T.float32(0)
                Y[v_r] = Y[v_r] + Y_rf[vk_1, v_r]

    sch = tvm.s_tir.Schedule(before, debug_mask="none")
    block = sch.get_sblock("Y")
    _, k = sch.get_loops(block)
    sch.split(k, factors=[None, 64])
    rf = sch.rfactor(sch.get_loops(block)[-1], factor_axis=0)
    sch.compute_at(rf, sch.get_loops(block)[-1])
    assert_structural_equal_ignore_global_symbol(expected, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=before, debug_mask="none")

    x = (np.arange(4 * 77, dtype="float32") % 7).reshape(4, 77)
    _llvm_allclose(sch.mod["main"], [x], [x.sum(axis=1)])


def test_reverse_compute_at_keeps_tail_predicate():
    """`reverse_compute_at` rebuilds loops in the same helper as `compute_at`.

    A consumer whose split does not divide its domain must keep the tail guard
    when that guard is not implied by the regenerated loop extent.
    """

    @Ts.function
    def before(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        B = Ts.sblock_alloc_tensor((128,), "float32")
        for i in range(128):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(128, i)
                B[vi] = A[vi]
        for j0, j1 in T.grid(2, 64):
            with Ts.sblock("C"):
                vj = Ts.axis.spatial(128, j0 * 64 + j1)
                Ts.where(j0 * 64 + j1 < 77)
                C[vj] = B[vj] * T.float32(2)

    @Ts.function
    def expected(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        B = Ts.sblock_alloc_tensor((128,), "float32")
        for i in range(128):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(128, i)
                B[vi] = A[vi]
            with Ts.sblock("C"):
                vj = Ts.axis.spatial(128, i)
                Ts.where(i < 77)
                C[vj] = B[vj] * T.float32(2)

    sch = tvm.s_tir.Schedule(before, debug_mask="none")
    sch.reverse_compute_at(sch.get_sblock("C"), sch.get_loops("B")[0])
    assert_structural_equal_ignore_global_symbol(expected, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=before, debug_mask="none")

    x = np.arange(128, dtype="float32")
    ref = np.zeros(128, dtype="float32")
    ref[:77] = x[:77] * np.float32(2)
    _llvm_allclose(sch.mod["main"], [x], [ref])


if __name__ == "__main__":
    tvm_tirx.testing.main()
