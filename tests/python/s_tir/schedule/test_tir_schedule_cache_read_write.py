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
# ruff: noqa: E741, F401
from __future__ import annotations

import sys

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable

########## Function before schedule ##########


@Ts.function
def elementwise(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def elementwise_shape_int64(
    A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    C: T.Tensor((T.int64(128), T.int64(128)), "float32"),
) -> None:
    B = Ts.sblock_alloc_tensor((T.int64(128), T.int64(128)))

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def elementwise_reindex_cache_read(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
):
    B = Ts.sblock_alloc_tensor((128, 128))
    B_shared = Ts.sblock_alloc_tensor((128, 64, 2), scope="shared")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(B[vi, vj])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with Ts.sblock("B_shared"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B[vi, vj])
            Ts.writes(B_shared[vj, vi // 2, vi % 2])
            B_shared[vj, vi // 2, vi % 2] = B[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B_shared[vj, vi // 2, vi % 2])
            Ts.writes(C[vi, vj])
            C[vi, vj] = B_shared[vj, vi // 2, vi % 2] + T.float32(1)


@Ts.function
def elementwise_reindex_cache_write(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
):
    B = Ts.sblock_alloc_tensor((128, 128))
    B_shared = Ts.sblock_alloc_tensor((128, 128), scope="shared")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(B_shared[vj, vi])
            B_shared[vj, vi] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with Ts.sblock("B_shared"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B_shared[vj, vi])
            Ts.writes(B[vi, vj])
            B[vi, vj] = B_shared[vj, vi]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B[vi, vj])
            Ts.writes(C[vi, vj])
            C[vi, vj] = B[vi, vj] + T.float32(1)


@Ts.function
def reduce(A: T.Tensor((128, 128, 128, 128), "float32"), C: T.Tensor((128, 128), "float32")):
    B = Ts.sblock_alloc_tensor((128, 128, 128), dtype="float32")
    for i, j, k in T.grid(128, 128, 128):
        for l in range(128):
            with Ts.sblock("B"):
                vi, vj, vk, vl = Ts.axis.remap("SSSR", [i, j, k, l])
                with Ts.init():
                    B[vi, vj, vk] = T.float32(0)
                B[vi, vj, vk] = B[vi, vj, vk] + A[vi, vj, vk, vl]
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + B[vi, vj, vk]


@Ts.function
def reduce_reindex_cache_write_0(
    A: T.Tensor((128, 128, 128, 128), "float32"), C: T.Tensor((128, 128), "float32")
):
    B = Ts.sblock_alloc_tensor((128, 128, 128))
    B_shared = Ts.sblock_alloc_tensor((128, 128, 128), scope="shared")
    for i, j, k in T.grid(128, 128, 128):
        for l in range(128):
            with Ts.sblock("B"):
                vi, vj, vk, vl = Ts.axis.remap("SSSR", [i, j, k, l])
                Ts.reads(A[vi, vj, vk, vl])
                Ts.writes(B_shared[vj, vi, vk])
                with Ts.init():
                    B_shared[vj, vi, vk] = T.float32(0)
                B_shared[vj, vi, vk] = B_shared[vj, vi, vk] + A[vi, vj, vk, vl]
        with Ts.sblock("B_shared"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            Ts.reads(B_shared[vj, vi, vk])
            Ts.writes(B[vi, vj, vk])
            B[vi, vj, vk] = B_shared[vj, vi, vk]
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(B[vi, vj, vk])
            Ts.writes(C[vi, vj])
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + B[vi, vj, vk]


@Ts.function
def reduce_reindex_cache_write_1(
    A: T.Tensor((128, 128, 128, 128), "float32"), C: T.Tensor((128, 128), "float32")
):
    B = Ts.sblock_alloc_tensor((128, 128, 128))
    B_shared = Ts.sblock_alloc_tensor((128, 128, 128), scope="shared")
    C_shared = Ts.sblock_alloc_tensor((128, 128), scope="shared")
    for i, j, k in T.grid(128, 128, 128):
        for l in range(128):
            with Ts.sblock("B"):
                vi, vj, vk, vl = Ts.axis.remap("SSSR", [i, j, k, l])
                Ts.reads(A[vi, vj, vk, vl])
                Ts.writes(B_shared[vj, vi, vk])
                with Ts.init():
                    B_shared[vj, vi, vk] = T.float32(0)
                B_shared[vj, vi, vk] = B_shared[vj, vi, vk] + A[vi, vj, vk, vl]
        with Ts.sblock("B_shared"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            Ts.reads(B_shared[vj, vi, vk])
            Ts.writes(B[vi, vj, vk])
            B[vi, vj, vk] = B_shared[vj, vi, vk]
        with Ts.sblock("C"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(B[vi, vj, vk])
            Ts.writes(C_shared[vj, vi])
            with Ts.init():
                C_shared[vj, vi] = T.float32(0)
            C_shared[vj, vi] = C_shared[vj, vi] + B[vi, vj, vk]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C_shared"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(C_shared[vj, vi])
            Ts.writes(C[vi, vj])
            C[vi, vj] = C_shared[vj, vi]


@Ts.function
def func_nested_seq(B: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    A = Ts.sblock_alloc_tensor((128, 128))

    for i, j in T.grid(128, 128):
        with Ts.sblock("A"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            A[vi, vj] = 2.0
    for i, j in T.grid(8, 8):
        for x, y in T.grid(16, 16):
            with Ts.sblock("B0"):
                vi = Ts.axis.S(128, i * 16 + x)
                vj = Ts.axis.S(128, j * 16 + y)
                B[vi, vj] = 1.0
        for x, y in T.grid(16, 16):
            with Ts.sblock("B1"):
                vi = Ts.axis.S(128, i * 16 + x)
                vj = Ts.axis.S(128, j * 16 + y)
                B[vi, vj] = A[vi, vj] + B[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0


@Ts.function
def access_under_scope(
    B: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    A = Ts.sblock_alloc_tensor((128, 128))

    for i0, j0 in T.grid(8, 8):
        with Ts.sblock("scope"):
            i, j = Ts.axis.remap("SS", [i0, j0])
            for x, y in T.grid(16, 16):
                with Ts.sblock("A"):
                    vi = Ts.axis.S(128, i * 16 + x)
                    vj = Ts.axis.S(128, j * 16 + y)
                    A[vi, vj] = 1.0
            for x, y in T.grid(16, 16):
                with Ts.sblock("B"):
                    vi = Ts.axis.S(128, i * 16 + x)
                    vj = Ts.axis.S(128, j * 16 + y)
                    B[vi, vj] = A[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0


opaque_access_a0_offset = T.dynamic("opaque_access_a0_offset", "int32")
opaque_access_c0_offset = T.dynamic("opaque_access_c0_offset", "int32")


@Ts.function
def opaque_access(
    A: T.Tensor((128, 128), dtype="float16"),
    B: T.Tensor((128, 128), dtype="float16"),
    C: T.Tensor((128, 128), dtype="float16"),
    D: T.Tensor((128, 128), dtype="float16"),
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("load_store"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(D[vi, vj])
            D[vi, vj] = A[vi, vj]
    for i, j in T.grid(8, 8):
        with Ts.sblock("opaque"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            Ts.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.evaluate(
                T.gpu_load_matrix_sync(
                    B.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.ptr_byte_offset(A.data, (vi * 2048 + vj * 16) * 2, ty=A.data.ty),
                    128,
                    "row_major",
                )
            )
    for i, j in T.grid(8, 8):
        with Ts.sblock("match_tensor"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            Ts.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            A0 = Ts.match_tensor(
                A[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                layout=T.TileLayout(T.S[(16, 16) : [128, 1]] + (opaque_access_a0_offset)),
            )
            C0 = Ts.match_tensor(
                C[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                layout=T.TileLayout(T.S[(16, 16) : [128, 1]] + (opaque_access_c0_offset)),
            )
            T.evaluate(
                T.gpu_load_matrix_sync(
                    C0.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    A0.ptr_to([0] * len(A0.shape)),
                    128,
                    "row_major",
                )
            )


@Ts.function
def func_multi_consumer() -> None:
    A = Ts.sblock_alloc_tensor(128)
    B = Ts.sblock_alloc_tensor(128)
    C = Ts.sblock_alloc_tensor(128)
    for i in T.grid(8):
        for j in T.grid(16):
            with Ts.sblock("A"):
                vi = Ts.axis.S(128, i * 16 + j)
                A[vi] = 1.0
        for j in T.grid(16):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i * 16 + j)
                B[vi] = A[vi] + 1.0
    for i in T.grid(128):
        with Ts.sblock("C"):
            vi = Ts.axis.S(128, i)
            C[vi] = A[vi]


@Ts.function
def reindex_cache_read_multi_consumer() -> None:
    A = Ts.sblock_alloc_tensor((128,))
    B = Ts.sblock_alloc_tensor((128,))
    C = Ts.sblock_alloc_tensor((128,))
    A_shared = Ts.sblock_alloc_tensor((4, 32), scope="shared")
    for i in range(8):
        for j in range(16):
            with Ts.sblock("A"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                Ts.reads()
                Ts.writes(A[vi])
                A[vi] = T.float32(1)
        for j in range(16):
            with Ts.sblock("A_shared"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                Ts.reads(A[vi])
                Ts.writes(A_shared[vi // 32, vi % 32])
                A_shared[vi // 32, vi % 32] = A[vi]
        for j in range(16):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                Ts.reads(A_shared[vi // 32, vi % 32])
                Ts.writes(B[vi])
                B[vi] = A_shared[vi // 32, vi % 32] + T.float32(1)
    for i in range(128):
        with Ts.sblock("C"):
            vi = Ts.axis.spatial(128, i)
            Ts.reads(A[vi])
            Ts.writes(C[vi])
            C[vi] = A[vi]


@Ts.function
def func_multi_producer() -> None:
    A = Ts.sblock_alloc_tensor(128)
    B = Ts.sblock_alloc_tensor(128)
    for i in range(128):
        with Ts.sblock("A0"):
            vi = Ts.axis.S(128, i)
            A[vi] = 1.0
    for i in range(128):
        with Ts.sblock("A1"):
            vi = Ts.axis.S(128, i)
            A[vi] = 2.0
    for i in range(128):
        with Ts.sblock("B"):
            vi = Ts.axis.S(128, i)
            B[vi] = A[vi]


@Ts.function
def func_with_block_predicate() -> None:
    A = Ts.sblock_alloc_tensor(120)
    B = Ts.sblock_alloc_tensor(120)
    for i, j in T.grid(16, 8):
        with Ts.sblock("producer"):
            Ts.where(i * 8 + j < 120)
            ax = Ts.axis.S(120, i * 8 + j)
            A[ax] = 0.0
    for i, j in T.grid(16, 8):
        with Ts.sblock("consumer"):
            Ts.where(i * 8 + j < 120)
            ax = Ts.axis.S(120, i * 8 + j)
            B[ax] = A[ax] + 1.0


@Ts.function
def inplace_func(data_io: T.Tensor((64), "int32")):
    data_1d = Ts.sblock_alloc_tensor([64], dtype="int32")
    for i0 in T.serial(64):
        with Ts.sblock("copy_in"):
            v0 = Ts.axis.remap("S", [i0])
            data_1d[v0] = data_io[v0]
    for i0 in T.serial(1):
        with Ts.sblock("ext_call"):
            Ts.reads(data_1d[:64])
            Ts.writes(data_1d[:64])
            T.evaluate(T.call_extern("call_impl", data_1d.data, ty=""))
    for i0 in T.serial(64):
        with Ts.sblock("copy_out"):
            v0 = Ts.axis.remap("S", [i0])
            data_io[v0] = data_1d[v0]


@Ts.function
def inplace_call(data_io: T.Tensor((64), "int32")):
    for i0 in T.serial(1):
        with Ts.sblock("ext_call"):
            Ts.reads(data_io[:64])
            Ts.writes(data_io[:64])
            T.evaluate(T.call_extern("call_impl", data_io.data, ty=""))


@Ts.function
def cache_read_nested_seq_target(
    B: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    A = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
    A_global = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("A"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads()
            Ts.writes(A[vi, vj])
            A[vi, vj] = T.float32(2)
    for i, j in T.grid(8, 8):
        for x, y in T.grid(16, 16):
            with Ts.sblock("B0"):
                vi = Ts.axis.spatial(128, i * 16 + x)
                vj = Ts.axis.spatial(128, j * 16 + y)
                Ts.reads()
                Ts.writes(B[vi, vj])
                B[vi, vj] = T.float32(1)
        for x, y in T.grid(16, 16):
            with Ts.sblock("B1"):
                vi = Ts.axis.spatial(128, i * 16 + x)
                vj = Ts.axis.spatial(128, j * 16 + y)
                Ts.reads(A[vi, vj], B[vi, vj])
                Ts.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] + B[vi, vj]
    for ax0, ax1 in T.grid(128, 128):
        with Ts.sblock("A_global"):
            v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
            Ts.reads(A[v0, v1])
            Ts.writes(A_global[v0, v1])
            A_global[v0, v1] = A[v0, v1]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A_global[vi, vj])
            Ts.writes(C[vi, vj])
            C[vi, vj] = A_global[vi, vj] * T.float32(2)


@Ts.function
def nested_tensor_access(
    A: T.Tensor((T.int64(7), T.int64(512)), dtype="float32"),
    B: T.Tensor(T.int64(1), dtype="int32"),
    C: T.Tensor((T.int64(1), T.int64(512)), dtype="float32"),
):
    for ax0, ax1 in T.grid(T.int64(1), T.int64(512)):
        with Ts.sblock("C"):
            v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
            Ts.reads(A[B[v_ax0], v_ax1], B[v_ax0])
            Ts.writes(C[v_ax0, v_ax1])
            C[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]


########## Expected function after cache_read ##########


@Ts.function
def cache_read_elementwise(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    A_global = Ts.sblock_alloc_tensor((128, 128))
    B_local = Ts.sblock_alloc_tensor((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with Ts.sblock("A_global"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            A_global[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A_global[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("B_local"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_local[vi, vj] = B[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B_local[vi, vj] + 1.0


@Ts.function
def cache_read_under_scope(
    B: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    A = Ts.sblock_alloc_tensor((128, 128))

    A_global = Ts.sblock_alloc_tensor((128, 128))

    for i0, j0 in T.grid(8, 8):
        with Ts.sblock("scope"):
            i, j = Ts.axis.remap("SS", [i0, j0])
            A_local = Ts.sblock_alloc_tensor((16, 16), scope="local")
            for x, y in T.grid(16, 16):
                with Ts.sblock("A"):
                    vi = Ts.axis.S(128, i * 16 + x)
                    vj = Ts.axis.S(128, j * 16 + y)
                    A[vi, vj] = 1.0
            for x, y in T.grid(16, 16):
                with Ts.sblock("A_local"):
                    vi = Ts.axis.S(16, x)
                    vj = Ts.axis.S(16, y)
                    A_local[vi, vj] = A[i * 16 + vi, j * 16 + vj]
            for x, y in T.grid(16, 16):
                with Ts.sblock("B"):
                    vi = Ts.axis.S(128, i * 16 + x)
                    vj = Ts.axis.S(128, j * 16 + y)
                    B[vi, vj] = A_local[vi - i * 16, vj - j * 16] + 1.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("A_global"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            A_global[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A_global[vi, vj] * 2.0


cache_read_opaque_access_a0_offset = T.dynamic("cache_read_opaque_access_a0_offset", "int32")
cache_read_opaque_access_c0_offset = T.dynamic("cache_read_opaque_access_c0_offset", "int32")


@Ts.function
def cache_read_opaque_access(
    A: T.Tensor((128, 128), dtype="float16"),
    B: T.Tensor((128, 128), dtype="float16"),
    C: T.Tensor((128, 128), dtype="float16"),
    D: T.Tensor((128, 128), dtype="float16"),
) -> None:
    A_global = Ts.sblock_alloc_tensor((128, 128), dtype="float16")

    for i, j in T.grid(128, 128):
        with Ts.sblock("A_global"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            A_global[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("load_store"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A_global[vi, vj])
            Ts.writes(D[vi, vj])
            D[vi, vj] = A_global[vi, vj]
    for i, j in T.grid(8, 8):
        with Ts.sblock("opaque"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            Ts.writes(B[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.evaluate(
                T.gpu_load_matrix_sync(
                    B.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.ptr_byte_offset(
                        A_global.data, (vi * 2048 + vj * 16) * 2, ty=A_global.data.ty
                    ),
                    128,
                    "row_major",
                )
            )
    for i, j in T.grid(8, 8):
        with Ts.sblock("match_tensor"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            Ts.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            A0 = Ts.match_tensor(
                A_global[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                layout=T.TileLayout(
                    T.S[(16, 16) : [128, 1]] + (cache_read_opaque_access_a0_offset)
                ),
            )
            C0 = Ts.match_tensor(
                C[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                layout=T.TileLayout(
                    T.S[(16, 16) : [128, 1]] + (cache_read_opaque_access_c0_offset)
                ),
            )
            T.evaluate(
                T.gpu_load_matrix_sync(
                    C0.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    A0.ptr_to([0] * len(A0.shape)),
                    128,
                    "row_major",
                )
            )


@Ts.function
def cache_read_multi_consumer() -> None:
    A = Ts.sblock_alloc_tensor(128)
    B = Ts.sblock_alloc_tensor(128)
    C = Ts.sblock_alloc_tensor(128)
    A_global = Ts.sblock_alloc_tensor(128)
    for i in T.grid(8):
        for j in T.grid(16):
            with Ts.sblock("A"):
                vi = Ts.axis.S(128, i * 16 + j)
                A[vi] = 1.0
        for j in T.grid(16):
            with Ts.sblock("A"):
                vi = Ts.axis.S(128, i * 16 + j)
                A_global[vi] = A[vi]
        for j in T.grid(16):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i * 16 + j)
                B[vi] = A_global[vi] + 1.0

    for i in T.grid(128):
        with Ts.sblock("C"):
            vi = Ts.axis.S(128, i)
            C[vi] = A_global[vi]


@Ts.function
def cache_read_multi_consumer_target() -> None:
    A = Ts.sblock_alloc_tensor(128)
    B = Ts.sblock_alloc_tensor(128)
    C = Ts.sblock_alloc_tensor(128)
    A_global = Ts.sblock_alloc_tensor(128)
    for i in T.grid(8):
        for j in T.grid(16):
            with Ts.sblock("A"):
                vi = Ts.axis.S(128, i * 16 + j)
                A[vi] = 1.0
        for j in T.grid(16):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i * 16 + j)
                B[vi] = A[vi] + 1.0

    for i in T.grid(128):
        with Ts.sblock("A"):
            vi = Ts.axis.S(128, i)
            A_global[vi] = A[vi]
    for i in T.grid(128):
        with Ts.sblock("C"):
            vi = Ts.axis.S(128, i)
            C[vi] = A_global[vi]


@Ts.function
def continuous_cache_read(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    B_shared = Ts.sblock_alloc_tensor((128, 128), scope="shared")
    B_local = Ts.sblock_alloc_tensor((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("B_shared"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_shared[vi, vj] = B[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("B_local"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_local[vi, vj] = B_shared[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B_local[vi, vj] + 1.0


@Ts.function
def block_predicate_cache_read() -> None:
    A = Ts.sblock_alloc_tensor([120], dtype="float32")
    B = Ts.sblock_alloc_tensor([120], dtype="float32")
    A_shared = Ts.sblock_alloc_tensor([120], dtype="float32", scope="shared")
    for i, j in T.grid(16, 8):
        with Ts.sblock("producer"):
            ax = Ts.axis.spatial(120, i * 8 + j)
            Ts.where(i * 8 + j < 120)
            A[ax] = T.float32(0)
    for ax0 in T.serial(120):
        with Ts.sblock("A_shared"):
            v0 = Ts.axis.spatial(120, ax0)
            A_shared[v0] = A[v0]
    for i, j in T.grid(16, 8):
        with Ts.sblock("consumer"):
            ax = Ts.axis.spatial(120, i * 8 + j)
            Ts.where(i * 8 + j < 120)
            B[ax] = A_shared[ax] + T.float32(1)


@Ts.function
def cache_read_shape_int64(
    A: T.Tensor((T.int64(128), T.int64(128)), dtype="float32"),
    C: T.Tensor((T.int64(128), T.int64(128)), dtype="float32"),
) -> None:
    B = Ts.sblock_alloc_tensor([T.int64(128), T.int64(128)], dtype="float32")
    A_global = Ts.sblock_alloc_tensor([T.int64(128), T.int64(128)], dtype="float32")
    for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
        with Ts.sblock("A_global"):
            v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
            Ts.reads(A[v0, v1])
            Ts.writes(A_global[v0, v1])
            A_global[v0, v1] = A[v0, v1]
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A_global[vi, vj])
            Ts.writes(B[vi, vj])
            B[vi, vj] = A_global[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(B[vi, vj])
            Ts.writes(C[vi, vj])
            C[vi, vj] = B[vi, vj] + T.float32(1)


@Ts.function
def cache_read_inplace(data_io: T.Tensor(64, "int32")) -> None:
    data_1d = Ts.sblock_alloc_tensor([64], dtype="int32")
    data_io_local = Ts.sblock_alloc_tensor([64], dtype="int32", scope="local")
    for ax0 in T.serial(64):
        with Ts.sblock("data_io_local"):
            v0 = Ts.axis.spatial(64, ax0)
            Ts.reads(data_io[v0])
            Ts.writes(data_io_local[v0])
            data_io_local[v0] = data_io[v0]
    for i0 in T.serial(64):
        with Ts.sblock("copy_in"):
            v0 = Ts.axis.spatial(64, i0)
            Ts.reads(data_io_local[v0])
            Ts.writes(data_1d[v0])
            data_1d[v0] = data_io_local[v0]
    for i0 in T.serial(1):
        with Ts.sblock("ext_call"):
            Ts.reads(data_1d[0:64])
            Ts.writes(data_1d[0:64])
            T.evaluate(T.call_extern("call_impl", data_1d.data, ty=""))
    for i0 in T.serial(64):
        with Ts.sblock("copy_out"):
            v0 = Ts.axis.spatial(64, i0)
            Ts.reads(data_1d[v0])
            Ts.writes(data_io[v0])
            data_io[v0] = data_1d[v0]


@Ts.function
def cache_inplace_tensor(data_io: T.Tensor(64, "int32")) -> None:
    data_io_local = Ts.sblock_alloc_tensor([64], dtype="int32", scope="local")
    data_io_global = Ts.sblock_alloc_tensor([64], dtype="int32")
    data_io_global_1 = Ts.sblock_alloc_tensor([64], dtype="int32")
    for ax0 in T.serial(64):
        with Ts.sblock("data_io_global"):
            v0 = Ts.axis.spatial(64, ax0)
            Ts.reads(data_io[v0])
            Ts.writes(data_io_global[v0])
            data_io_global[v0] = data_io[v0]
    for i0 in T.serial(1):
        for ax0 in T.serial(64):
            with Ts.sblock("data_io_local"):
                v0 = Ts.axis.spatial(64, ax0)
                Ts.reads(data_io_global[v0])
                Ts.writes(data_io_local[v0])
                data_io_local[v0] = data_io_global[v0]
        with Ts.sblock("ext_call"):
            Ts.reads(data_io_local[0:64])
            Ts.writes(data_io_local[0:64])
            T.evaluate(T.call_extern("call_impl", data_io_local.data, ty=""))
        for ax0 in T.serial(64):
            with Ts.sblock("data_io_local"):
                v0 = Ts.axis.spatial(64, ax0)
                Ts.reads(data_io_local[v0])
                Ts.writes(data_io_global_1[v0])
                data_io_global_1[v0] = data_io_local[v0]
    for ax0 in T.serial(64):
        with Ts.sblock("data_io_global"):
            v0 = Ts.axis.spatial(64, ax0)
            Ts.reads(data_io_global_1[v0])
            Ts.writes(data_io[v0])
            data_io[v0] = data_io_global_1[v0]


@Ts.function
def cache_read_nested_tensor_access(
    A: T.Tensor((T.int64(7), T.int64(512)), dtype="float32"),
    B: T.Tensor(T.int64(1), dtype="int32"),
    C: T.Tensor((T.int64(1), T.int64(512)), dtype="float32"),
):
    B_global = Ts.sblock_alloc_tensor((T.int64(1),), "int32")
    for ax0 in range(T.int64(1)):
        with Ts.sblock("B_global"):
            v0 = Ts.axis.spatial(T.int64(1), ax0)
            Ts.reads(B[v0])
            Ts.writes(B_global[v0])
            B_global[v0] = B[v0]
    for ax0, ax1 in T.grid(T.int64(1), T.int64(512)):
        with Ts.sblock("C"):
            v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
            Ts.reads(A[B_global[v_ax0], v_ax1], B_global[v_ax0])
            Ts.writes(C[v_ax0, v_ax1])
            C[v_ax0, v_ax1] = A[B_global[v_ax0], v_ax1]


########## Expected function after cache_write ##########


@Ts.function
def cache_write_elementwise(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))
    B_global = Ts.sblock_alloc_tensor((128, 128), scope="local")
    C_local = Ts.sblock_alloc_tensor((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock("B_global"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_global[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = B_global[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C_local"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C_local[vi, vj] = B[vi, vj] + 1.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = C_local[vi, vj]


@Ts.function
def cache_write_under_scope(
    B: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    A = Ts.sblock_alloc_tensor((128, 128))

    A_global = Ts.sblock_alloc_tensor((128, 128))

    for i0, j0 in T.grid(8, 8):
        with Ts.sblock("scope"):
            i, j = Ts.axis.remap("SS", [i0, j0])
            A_local = Ts.sblock_alloc_tensor((16, 16), scope="local")
            B_global = Ts.sblock_alloc_tensor((16, 16))
            for x, y in T.grid(16, 16):
                with Ts.sblock("A_local"):
                    vi = Ts.axis.S(128, i * 16 + x)
                    vj = Ts.axis.S(128, j * 16 + y)
                    A_local[vi - i * 16, vj - j * 16] = 1.0
            for x, y in T.grid(16, 16):
                with Ts.sblock("A"):
                    vi = Ts.axis.S(16, x)
                    vj = Ts.axis.S(16, y)
                    A_global[i * 16 + vi, j * 16 + vj] = A_local[vi, vj]
            for x, y in T.grid(16, 16):
                with Ts.sblock("B"):
                    vi = Ts.axis.S(128, i * 16 + x)
                    vj = Ts.axis.S(128, j * 16 + y)
                    B_global[vi - i * 16, vj - j * 16] = A_global[vi, vj] + 1.0
            for x, y in T.grid(16, 16):
                with Ts.sblock("B_global"):
                    vi = Ts.axis.S(16, x)
                    vj = Ts.axis.S(16, y)
                    B[i * 16 + vi, j * 16 + vj] = B_global[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("A_global"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            A[vi, vj] = A_global[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] * 2.0


cache_write_opaque_access_a0_offset = T.dynamic("cache_write_opaque_access_a0_offset", "int32")
cache_write_opaque_access_c0_offset = T.dynamic("cache_write_opaque_access_c0_offset", "int32")


@Ts.function
def cache_write_opaque_access(
    A: T.Tensor((128, 128), dtype="float16"),
    B: T.Tensor((128, 128), dtype="float16"),
    C: T.Tensor((128, 128), dtype="float16"),
    D: T.Tensor((128, 128), dtype="float16"),
) -> None:
    D_global = Ts.sblock_alloc_tensor((128, 128), dtype="float16")
    B_global = Ts.sblock_alloc_tensor((128, 128), dtype="float16")
    C_global = Ts.sblock_alloc_tensor((128, 128), dtype="float16")

    for i, j in T.grid(128, 128):
        with Ts.sblock("load_store"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi, vj])
            Ts.writes(D_global[vi, vj])
            D_global[vi, vj] = A[vi, vj]
    for i, j in T.grid(8, 8):
        with Ts.sblock("opaque"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            Ts.writes(B_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            T.evaluate(
                T.gpu_load_matrix_sync(
                    B_global.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    T.ptr_byte_offset(A.data, (vi * 2048 + vj * 16) * 2, ty=A.data.ty),
                    128,
                    "row_major",
                )
            )
    for i, j in T.grid(8, 8):
        with Ts.sblock("match_tensor"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            Ts.writes(C_global[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            A0 = Ts.match_tensor(
                A[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                layout=T.TileLayout(
                    T.S[(16, 16) : [128, 1]] + (cache_write_opaque_access_a0_offset)
                ),
            )
            C0 = Ts.match_tensor(
                C_global[
                    vi * 16 : vi * 16 + 16,
                    vj * 16 : vj * 16 + 16,
                ],
                (16, 16),
                "float16",
                layout=T.TileLayout(
                    T.S[(16, 16) : [128, 1]] + (cache_write_opaque_access_c0_offset)
                ),
            )
            T.evaluate(
                T.gpu_load_matrix_sync(
                    C0.data,
                    16,
                    16,
                    16,
                    vi * 8 + vj,
                    A0.ptr_to([0] * len(A0.shape)),
                    128,
                    "row_major",
                )
            )

    for i, j in T.grid(128, 128):
        with Ts.sblock("D"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = D_global[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = B_global[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = C_global[vi, vj]


@Ts.function
def cache_write_multi_consumer() -> None:
    A = Ts.sblock_alloc_tensor(128)
    B = Ts.sblock_alloc_tensor(128)
    C = Ts.sblock_alloc_tensor(128)
    A_global = Ts.sblock_alloc_tensor(128)
    for i in T.grid(8):
        for j in T.grid(16):
            with Ts.sblock("A_global"):
                vi = Ts.axis.S(128, i * 16 + j)
                A_global[vi] = 1.0
        for j in T.grid(16):
            with Ts.sblock("A"):
                vi = Ts.axis.S(128, i * 16 + j)
                A[vi] = A_global[vi]
        for j in T.grid(16):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i * 16 + j)
                B[vi] = A[vi] + 1.0

    for i in T.grid(128):
        with Ts.sblock("C"):
            vi = Ts.axis.S(128, i)
            C[vi] = A[vi]


@Ts.function
def cache_write_multi_consumer_B_consume_cache():
    A = Ts.sblock_alloc_tensor([128], dtype="float32")
    B = Ts.sblock_alloc_tensor([128], dtype="float32")
    C = Ts.sblock_alloc_tensor([128], dtype="float32")
    A_global = Ts.sblock_alloc_tensor([128], dtype="float32")
    for i in T.serial(8):
        for j in T.serial(16):
            with Ts.sblock("A"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                A_global[vi] = 1.0
        for j in T.serial(16):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                B[vi] = A_global[vi] + 1.0
    for ax0 in T.serial(128):
        with Ts.sblock("A_global"):
            v0 = Ts.axis.spatial(128, ax0)
            A[v0] = A_global[v0]
    for i in T.serial(128):
        with Ts.sblock("C"):
            vi = Ts.axis.spatial(128, i)
            C[vi] = A[vi]


@Ts.function
def cache_write_multi_consumer_C_consume_cache():
    A = Ts.sblock_alloc_tensor([128], dtype="float32")
    B = Ts.sblock_alloc_tensor([128], dtype="float32")
    C = Ts.sblock_alloc_tensor([128], dtype="float32")
    A_global = Ts.sblock_alloc_tensor([128], dtype="float32")
    for i in T.serial(8):
        for j in T.serial(16):
            with Ts.sblock("A"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                A_global[vi] = T.float32(1)
        for ax0 in T.serial(16):
            with Ts.sblock("A_global"):
                v0 = Ts.axis.spatial(128, i * 16 + ax0)
                A[v0] = A_global[v0]
        for j in T.serial(16):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                B[vi] = A[vi] + T.float32(1)
    for i in T.serial(128):
        with Ts.sblock("C"):
            vi = Ts.axis.spatial(128, i)
            C[vi] = A_global[vi]


@Ts.function
def cache_write_multi_consumer_all_consume_cache():
    A = Ts.sblock_alloc_tensor([128], dtype="float32")
    B = Ts.sblock_alloc_tensor([128], dtype="float32")
    C = Ts.sblock_alloc_tensor([128], dtype="float32")
    A_global = Ts.sblock_alloc_tensor([128], dtype="float32")
    for i in T.serial(8):
        for j in T.serial(16):
            with Ts.sblock("A"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                A_global[vi] = T.float32(1)
        for j in T.serial(16):
            with Ts.sblock("B"):
                vi = Ts.axis.spatial(128, i * 16 + j)
                B[vi] = A_global[vi] + T.float32(1)
    for i in T.serial(128):
        with Ts.sblock("C"):
            vi = Ts.axis.spatial(128, i)
            C[vi] = A_global[vi]
    for ax0 in T.serial(128):
        with Ts.sblock("A_global"):
            v0 = Ts.axis.spatial(128, ax0)
            A[v0] = A_global[v0]


@Ts.function
def continuous_cache_write(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128))

    B_shared = Ts.sblock_alloc_tensor((128, 128), scope="shared")
    B_local = Ts.sblock_alloc_tensor((128, 128), scope="local")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_local[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B_shared[vi, vj] = B_local[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = B_shared[vi, vj]
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def block_predicate_cache_write_intermediate_buf() -> None:
    A = Ts.sblock_alloc_tensor([120], dtype="float32")
    B = Ts.sblock_alloc_tensor([120], dtype="float32")
    A_shared = Ts.sblock_alloc_tensor([120], dtype="float32", scope="shared")
    for i, j in T.grid(16, 8):
        with Ts.sblock("producer"):
            ax = Ts.axis.spatial(120, i * 8 + j)
            Ts.where(i * 8 + j < 120)
            A_shared[ax] = T.float32(0)
    for ax0 in T.serial(120):
        with Ts.sblock("A_shared"):
            v0 = Ts.axis.spatial(120, ax0)
            A[v0] = A_shared[v0]
    for i, j in T.grid(16, 8):
        with Ts.sblock("consumer"):
            ax = Ts.axis.spatial(120, i * 8 + j)
            Ts.where(i * 8 + j < 120)
            B[ax] = A[ax] + 1.0


@Ts.function
def block_predicate_cache_write_output_buf() -> None:
    A = Ts.sblock_alloc_tensor([120], dtype="float32")
    B = Ts.sblock_alloc_tensor([120], dtype="float32")
    B_shared = Ts.sblock_alloc_tensor([120], dtype="float32", scope="shared")
    for i, j in T.grid(16, 8):
        with Ts.sblock("producer"):
            ax = Ts.axis.spatial(120, i * 8 + j)
            Ts.where(i * 8 + j < 120)
            A[ax] = T.float32(0)
    for i, j in T.grid(16, 8):
        with Ts.sblock("consumer"):
            ax = Ts.axis.spatial(120, i * 8 + j)
            Ts.where(i * 8 + j < 120)
            B_shared[ax] = A[ax] + T.float32(1)
    for ax0 in T.serial(120):
        with Ts.sblock("B_shared"):
            v0 = Ts.axis.spatial(120, ax0)
            B[v0] = B_shared[v0]


@Ts.function
def symbolic_matmul_blocked(
    A: T.Tensor(((n + 31) // 32 * 32, 4), "float32"),  # noqa: F821
    B: T.Tensor((4, (n + 31) // 32 * 32), "float32"),  # noqa: F821
    C: T.Tensor(((n + 31) // 32 * 32, (n + 31) // 32 * 32), "float32"),  # noqa: F821
    n: T.int32,
):
    for i0_0, i1_0 in T.grid((n + 31) // 32, (n + 31) // 32):
        with Ts.sblock("matmul_o"):
            v_i0_o, v_i1_o = Ts.axis.remap("SS", [i0_0, i1_0])
            Ts.reads(
                A[v_i0_o * 32 : v_i0_o * 32 + 32, 0:4],
                B[0:4, v_i1_o * 32 : v_i1_o * 32 + 32],
            )
            Ts.writes(C[v_i0_o * 32 : v_i0_o * 32 + 32, v_i1_o * 32 : v_i1_o * 32 + 32])
            for i0_1, i1_1, k in T.grid(32, 32, 4):
                with Ts.sblock("matmul"):
                    v_i0_i, v_i1_i, v_k_i = Ts.axis.remap("SSR", [i0_1, i1_1, k])
                    Ts.reads(A[v_i0_o * 32 + v_i0_i, v_k_i], B[v_k_i, v_i1_o * 32 + v_i1_i])
                    Ts.writes(C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i])
                    with Ts.init():
                        C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = T.float32(0)
                    C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = (
                        C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i]
                        + A[v_i0_o * 32 + v_i0_i, v_k_i] * B[v_k_i, v_i1_o * 32 + v_i1_i]
                    )


@Ts.function
def symbolic_matmul_blocked_cache_read(
    A: T.Tensor(((n + 31) // 32 * 32, 4), "float32"),  # noqa: F821
    B: T.Tensor((4, (n + 31) // 32 * 32), "float32"),  # noqa: F821
    C: T.Tensor(((n + 31) // 32 * 32, (n + 31) // 32 * 32), "float32"),  # noqa: F821
    n: T.int32,
):
    for i0_0, i1_0 in T.grid((n + 31) // 32, (n + 31) // 32):
        with Ts.sblock("matmul_o"):
            v_i0_o, v_i1_o = Ts.axis.remap("SS", [i0_0, i1_0])
            Ts.reads(
                A[v_i0_o * 32 : v_i0_o * 32 + 32, 0:4],
                B[0:4, v_i1_o * 32 : v_i1_o * 32 + 32],
            )
            Ts.writes(C[v_i0_o * 32 : v_i0_o * 32 + 32, v_i1_o * 32 : v_i1_o * 32 + 32])
            A_shared = Ts.sblock_alloc_tensor((32, 4), scope="shared")
            for ax0, ax1 in T.grid(32, 4):
                with Ts.sblock("A_shared"):
                    v0 = Ts.axis.spatial(32, ax0)
                    v1 = Ts.axis.spatial(4, ax1)
                    Ts.reads(A[v_i0_o * 32 + v0, v1])
                    Ts.writes(A_shared[v0, v1])
                    A_shared[v0, v1] = A[v_i0_o * 32 + v0, v1]
            for i0_1, i1_1, k in T.grid(32, 32, 4):
                with Ts.sblock("matmul"):
                    v_i0_i, v_i1_i, v_k_i = Ts.axis.remap("SSR", [i0_1, i1_1, k])
                    Ts.reads(A_shared[v_i0_i, v_k_i], B[v_k_i, v_i1_o * 32 + v_i1_i])
                    Ts.writes(C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i])
                    with Ts.init():
                        C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = T.float32(0)
                    C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i] = (
                        C[v_i0_o * 32 + v_i0_i, v_i1_o * 32 + v_i1_i]
                        + A_shared[v_i0_i, v_k_i] * B[v_k_i, v_i1_o * 32 + v_i1_i]
                    )


@Ts.function
def symbolic_matmul_blocked_cache_write(
    A: T.Tensor(((n + 31) // 32 * 32, 4), "float32"),  # noqa: F821
    B: T.Tensor((4, (n + 31) // 32 * 32), "float32"),  # noqa: F821
    C: T.Tensor(((n + 31) // 32 * 32, (n + 31) // 32 * 32), "float32"),  # noqa: F821
    n: T.int32,
):
    for i0_0, i1_0 in T.grid((n + 31) // 32, (n + 31) // 32):
        with Ts.sblock("matmul_o"):
            v_i0_o, v_i1_o = Ts.axis.remap("SS", [i0_0, i1_0])
            Ts.reads(
                A[v_i0_o * 32 : v_i0_o * 32 + 32, 0:4],
                B[0:4, v_i1_o * 32 : v_i1_o * 32 + 32],
            )
            Ts.writes(C[v_i0_o * 32 : v_i0_o * 32 + 32, v_i1_o * 32 : v_i1_o * 32 + 32])
            C_pad_local = Ts.sblock_alloc_tensor((32, 32), scope="local")
            for i0_1, i1_1, k in T.grid(32, 32, 4):
                with Ts.sblock("matmul"):
                    v_i0_i, v_i1_i, v_k_i = Ts.axis.remap("SSR", [i0_1, i1_1, k])
                    Ts.reads(A[v_i0_o * 32 + v_i0_i, v_k_i], B[v_k_i, v_i1_o * 32 + v_i1_i])
                    Ts.writes(C_pad_local[v_i0_i, v_i1_i])
                    with Ts.init():
                        C_pad_local[v_i0_i, v_i1_i] = T.float32(0)
                    C_pad_local[v_i0_i, v_i1_i] = (
                        C_pad_local[v_i0_i, v_i1_i]
                        + A[v_i0_o * 32 + v_i0_i, v_k_i] * B[v_k_i, v_i1_o * 32 + v_i1_i]
                    )
            for ax0, ax1 in T.grid(32, 32):
                with Ts.sblock("C_pad_local"):
                    v0 = Ts.axis.spatial(32, ax0)
                    v1 = Ts.axis.spatial(32, ax1)
                    Ts.reads(C_pad_local[v0, v1])
                    Ts.writes(C[v_i0_o * 32 + v0, v_i1_o * 32 + v1])
                    C[v_i0_o * 32 + v0, v_i1_o * 32 + v1] = C_pad_local[v0, v1]


########## Testcases for cache_read ##########


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


def test_cache_read_elementwise(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    block_c = sch.get_sblock("C")
    tensor_b_name = sch.get(block_b).writes[0].source.name
    if use_block_name:
        cached_a = sch.cache_read("B", "A", "global")
        cached_b = sch.cache_read("C", tensor_b_name, "local")
    else:
        cached_a = sch.cache_read(block_b, 0, "global")
        cached_b = sch.cache_read(block_c, 0, "local")
    assert sch.get(cached_a) == sch.get(sch.get_sblock("A_global"))
    assert sch.get(cached_b) == sch.get(sch.get_sblock(tensor_b_name + "_local"))
    assert sch.get(block_b) == sch.get(sch.get_sblock("B"))
    assert sch.get(block_c) == sch.get(sch.get_sblock("C"))
    assert_structural_equal_ignore_global_symbol(cache_read_elementwise, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_read_under_scope(use_block_name):
    sch = tvm.s_tir.Schedule(access_under_scope, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.cache_read(block_b, 0, "local")
    sch.cache_read(block_c, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_under_scope, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=access_under_scope)


def test_cache_read_opaque_access(use_block_name):
    sch = tvm.s_tir.Schedule(opaque_access, debug_mask="all")
    block = "load_store" if use_block_name else sch.get_sblock("load_store")
    sch.cache_read(block, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_opaque_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_cache_read_location(use_block_name):
    sch = tvm.s_tir.Schedule(func_multi_consumer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    sch.cache_read(block_b, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_multi_consumer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Test that specific consumer block targeting works.
    sch = tvm.s_tir.Schedule(func_multi_consumer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.cache_read(block_b, 0, "global", consumer_blocks=[block_c])
    assert_structural_equal_ignore_global_symbol(cache_read_multi_consumer_target, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Also test setting multiple consumers yields same result as unspecified.
    sch = tvm.s_tir.Schedule(func_multi_consumer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.cache_read(block_b, 0, "global", consumer_blocks=[block_b, block_c])
    assert_structural_equal_ignore_global_symbol(cache_read_multi_consumer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)


def test_continuous_cache_read(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.cache_read(block_c, 0, "shared")
    sch.cache_read(block_c, 0, "local")
    assert_structural_equal_ignore_global_symbol(continuous_cache_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_read_with_block_predicate(use_block_name):
    sch = tvm.s_tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = "consumer" if use_block_name else sch.get_sblock("consumer")
    sch.cache_read(block, 0, "shared")
    assert_structural_equal_ignore_global_symbol(block_predicate_cache_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)


def test_cache_read_non_int32_shape(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise_shape_int64, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    sch.cache_read(block_b, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_shape_int64, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise_shape_int64)


def test_cache_read_nested_tensor_access(use_block_name):
    sch = tvm.s_tir.Schedule(nested_tensor_access, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.cache_read(block_c, 1, "global")
    assert_structural_equal_ignore_global_symbol(cache_read_nested_tensor_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=nested_tensor_access)


def test_cache_read_fail_multi_producer(use_block_name):
    sch = tvm.s_tir.Schedule(func_multi_producer, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.cache_read(block_b, 0, "global")


def test_cache_read_fail_index_out_of_bound(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.cache_read(block_b, 1, "global")


def test_cache_read_fail_invalid_storage_scope(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.cache_read(block_b, 0, "test_scope")


def test_inplace_cache_read():
    sch = tvm.s_tir.Schedule(inplace_func, debug_mask="all")
    block = sch.get_sblock("copy_in")
    sch.cache_read(block, 0, "local", [block])
    assert_structural_equal_ignore_global_symbol(cache_read_inplace, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=inplace_func)


def test_cache_inplace():
    # cache_inplace could introduce WAR, which is expected but stage pipeline property changes
    debug_mask = tvm.s_tir.schedule.state.ScheduleDebugMask.VERIFY_SREF_TREE
    sch = tvm.s_tir.Schedule(inplace_call, debug_mask=debug_mask)
    block = sch.get_sblock("ext_call")
    blocks = sch.cache_inplace(block, 0, "local")
    block = sch.cache_read(blocks[0], 0, "global", [blocks[0]])
    block = sch.cache_write(blocks[1], 0, "global")

    assert_structural_equal_ignore_global_symbol(cache_inplace_tensor, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=inplace_call, debug_mask=debug_mask)


def test_cache_read_nested_seq(use_block_name):
    sch = tvm.s_tir.Schedule(func_nested_seq, debug_mask="all")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.cache_read(block_c, 0, "global", consumer_blocks=[block_c])
    assert_structural_equal_ignore_global_symbol(cache_read_nested_seq_target, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_nested_seq)


########## Testcases for cache_write ##########


def test_cache_write_elementwise(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    block_c = sch.get_sblock("C")
    tensor_b_name = sch.get(block_b).writes[0].source.name
    cached_b = sch.cache_write("B" if use_block_name else block_b, 0, "local")
    cached_c = sch.cache_write("C" if use_block_name else block_c, 0, "global")
    assert sch.get(cached_b) == sch.get(sch.get_sblock(tensor_b_name + "_local"))
    assert sch.get(cached_c) == sch.get(sch.get_sblock("C_global"))
    assert sch.get(block_b) == sch.get(sch.get_sblock("B"))
    assert sch.get(block_c) == sch.get(sch.get_sblock("C"))
    assert_structural_equal_ignore_global_symbol(cache_write_elementwise, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_write_under_scope(use_block_name):
    sch = tvm.s_tir.Schedule(access_under_scope, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_sblock("A")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_scope = sch.get_sblock("scope")
    sch.cache_write(block_a, 0, "local")
    sch.cache_write(block_b, 0, "global")
    sch.cache_write(block_scope, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_write_under_scope, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=access_under_scope)


def test_cache_write_opaque_access(use_block_name):
    sch = tvm.s_tir.Schedule(opaque_access, debug_mask="all")
    block_store = "load_store" if use_block_name else sch.get_sblock("load_store")
    block_opaque = "opaque" if use_block_name else sch.get_sblock("opaque")
    block_match_tensor = "match_tensor" if use_block_name else sch.get_sblock("match_tensor")
    sch.cache_write(block_store, 0, "global")
    sch.cache_write(block_opaque, 0, "global")
    sch.cache_write(block_match_tensor, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_write_opaque_access, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=opaque_access)


def test_cache_write_location(use_block_name):
    sch = tvm.s_tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_sblock("A")
    sch.cache_write(block_a, 0, "global")
    assert_structural_equal_ignore_global_symbol(cache_write_multi_consumer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Test that specific consumer block targeting works.
    # B read cache tensor and C read original output tensor
    sch = tvm.s_tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_sblock("A")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    sch.cache_write(block_a, 0, "global", consumer_blocks=[block_b])
    assert_structural_equal_ignore_global_symbol(
        cache_write_multi_consumer_B_consume_cache, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Test that specific consumer block targeting works.
    # B read original output tensor and C read cache tensor
    sch = tvm.s_tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_sblock("A")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.cache_write(block_a, 0, "global", consumer_blocks=[block_c])
    assert_structural_equal_ignore_global_symbol(
        cache_write_multi_consumer_C_consume_cache, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)

    # Test that specific consumer block targeting works.
    # B and C read cache tensor
    sch = tvm.s_tir.Schedule(func_multi_consumer, debug_mask="all")
    block_a = "A" if use_block_name else sch.get_sblock("A")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    block_c = "C" if use_block_name else sch.get_sblock("C")
    sch.cache_write(block_a, 0, "global", consumer_blocks=[block_b, block_c])
    assert_structural_equal_ignore_global_symbol(
        cache_write_multi_consumer_all_consume_cache, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_multi_consumer)


def test_continuous_cache_write(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    sch.cache_write(block_b, 0, "shared")
    sch.cache_write(block_b, 0, "local")
    assert_structural_equal_ignore_global_symbol(continuous_cache_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_cache_write_with_block_predicate(use_block_name):
    # cache write for intermediate tensor
    sch = tvm.s_tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = "producer" if use_block_name else sch.get_sblock("producer")
    sch.cache_write(block, 0, "shared")
    assert_structural_equal_ignore_global_symbol(
        block_predicate_cache_write_intermediate_buf, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)
    # cache write for external tensor
    sch = tvm.s_tir.Schedule(func_with_block_predicate, debug_mask="all")
    block = "consumer" if use_block_name else sch.get_sblock("consumer")
    sch.cache_write(block, 0, "shared")
    assert_structural_equal_ignore_global_symbol(
        block_predicate_cache_write_output_buf, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=func_with_block_predicate)


def test_cache_write_fail_multi_producer(use_block_name):
    sch = tvm.s_tir.Schedule(func_multi_producer, debug_mask="all")
    block_a0 = "A0" if use_block_name else sch.get_sblock("A0")
    block_a1 = "A1" if use_block_name else sch.get_sblock("A1")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.cache_write(block_a0, 0, "global")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.cache_write(block_a1, 0, "global")


def test_cache_write_fail_index_out_of_bound(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.cache_write(block_b, 1, "global")


def test_cache_write_fail_invalid_storage_scope(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block_b = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.cache_write(block_b, 0, "test_scope")


def test_reindex_cache_read():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    sch.reindex_cache_read("C", 0, "shared", lambda i, j: (j, i // 2, i % 2))
    assert_structural_equal_ignore_global_symbol(elementwise_reindex_cache_read, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reindex_cache_read_multi_consumer():
    sch = tvm.s_tir.Schedule(func_multi_consumer)
    sch.reindex_cache_read("B", 0, "shared", lambda i: (i // 32, i % 32))
    assert_structural_equal_ignore_global_symbol(reindex_cache_read_multi_consumer, sch.mod["main"])
    # NOTE(zihao): we do not verify trace roundtrip because of in set analysis issues.


def test_reindex_cache_read_fail_not_match():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reindex_cache_read(
            "C",
            0,
            "shared",
            lambda i, j: j * 2,
        )


def test_reindex_cache_read_failed_not_single_point():
    sch = tvm.s_tir.Schedule(access_under_scope, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reindex_cache_read("scope", 0, "shared", lambda i, j: (i, j))


def test_reindex_cache_write():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    sch.reindex_cache_write("B", 0, "shared", lambda i, j: (j, i))
    assert_structural_equal_ignore_global_symbol(elementwise_reindex_cache_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_reindex_cache_write_reduce():
    sch = tvm.s_tir.Schedule(reduce, debug_mask="all")
    sch.reindex_cache_write("B", 0, "shared", lambda i, j, k, l: (j, i, k))
    assert_structural_equal_ignore_global_symbol(reduce_reindex_cache_write_0, sch.mod["main"])
    sch.reindex_cache_write("C", 0, "shared", lambda i, j, k: [j, i])
    assert_structural_equal_ignore_global_symbol(reduce_reindex_cache_write_1, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=reduce)


def test_reindex_cache_write_fail_not_match():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reindex_cache_write(
            "B",
            0,
            "shared",
            lambda i, j: i,
        )


def test_reindex_cache_write_fail_not_single_point():
    sch = tvm.s_tir.Schedule(access_under_scope, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.reindex_cache_write("scope", 0, "shared", lambda i, j: (i, j))


def test_symbolic_matmul_blocked_cache_read(use_block_name):
    sch = tvm.s_tir.Schedule(symbolic_matmul_blocked, debug_mask="all")
    block = "matmul" if use_block_name else sch.get_sblock("matmul")
    sch.cache_read(block=block, read_tensor_index=0, storage_scope="shared")
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], symbolic_matmul_blocked_cache_read
    )
    verify_trace_roundtrip(sch=sch, mod=symbolic_matmul_blocked)


def test_symbolic_matmul_blocked_cache_write(use_block_name):
    sch = tvm.s_tir.Schedule(symbolic_matmul_blocked, debug_mask="all")
    block = "matmul" if use_block_name else sch.get_sblock("matmul")
    sch.cache_write(block=block, write_tensor_index=0, storage_scope="local")
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], symbolic_matmul_blocked_cache_write
    )
    verify_trace_roundtrip(sch=sch, mod=symbolic_matmul_blocked)


def test_cache_write_with_nested_block_predicate():
    @Ts.function
    def main(A_buf: T.Tensor((12, 24), "float32"), C_buf: T.Tensor((10, 20), "float32")) -> None:
        for i, j in T.grid(12, 24):
            with Ts.sblock("compute"):
                vi, vj = Ts.axis.remap("SS", [i, j])

                with Ts.sblock("inner"):
                    Ts.where(vi < 10 and vj < 20)
                    C_buf[vi, vj] = A_buf[vi, vj] * 2.0

    @Ts.function
    def expected(A_buf: T.Tensor((12, 24), "float32"), C_buf: T.Tensor((10, 20), "float32")):
        with Ts.sblock("root"):
            C_buf_local = Ts.sblock_alloc_tensor((10, 20), scope="local")
            for i, j in T.grid(12, 24):
                with Ts.sblock("compute"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A_buf[vi, vj])
                    Ts.writes(C_buf_local[vi, vj])
                    with Ts.sblock("inner"):
                        Ts.where(vi < 10 and vj < 20)
                        Ts.reads(A_buf[vi, vj])
                        Ts.writes(C_buf_local[vi, vj])
                        C_buf_local[vi, vj] = A_buf[vi, vj] * T.float32(2)
            for ax0, ax1 in T.grid(10, 20):
                with Ts.sblock("C_buf_local"):
                    v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(C_buf_local[v0, v1])
                    Ts.writes(C_buf[v0, v1])
                    C_buf[v0, v1] = C_buf_local[v0, v1]

    sch = tvm.s_tir.Schedule(main)
    block = sch.get_sblock("compute")
    sch.cache_write(block, 0, "local")
    assert_structural_equal_ignore_global_symbol(expected, sch.mod["main"])


def test_cache_read_with_nested_block_predicate():
    @Ts.function
    def main(A_buf: T.Tensor((12, 24), "float32"), C_buf: T.Tensor((10, 20), "float32")) -> None:
        for i, j in T.grid(12, 24):
            with Ts.sblock("compute"):
                vi, vj = Ts.axis.remap("SS", [i, j])

                with Ts.sblock("inner"):
                    Ts.where(vi < 10 and vj < 20)
                    C_buf[vi, vj] = A_buf[vi, vj] * 2.0

    @Ts.function
    def expected(A_buf: T.Tensor((12, 24), "float32"), C_buf: T.Tensor((10, 20), "float32")):
        with Ts.sblock("root"):
            A_buf_local = Ts.sblock_alloc_tensor((10, 20), scope="local")
            for ax0, ax1 in T.grid(10, 20):
                with Ts.sblock("A_buf_local"):
                    v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A_buf[v0, v1])
                    Ts.writes(A_buf_local[v0, v1])
                    A_buf_local[v0, v1] = A_buf[v0, v1]
            for i, j in T.grid(12, 24):
                with Ts.sblock("compute"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A_buf_local[vi, vj])
                    Ts.writes(C_buf[vi, vj])
                    with Ts.sblock("inner"):
                        Ts.where(vi < 10 and vj < 20)
                        Ts.reads(A_buf_local[vi, vj])
                        Ts.writes(C_buf[vi, vj])
                        C_buf[vi, vj] = A_buf_local[vi, vj] * T.float32(2)

    sch = tvm.s_tir.Schedule(main)
    block = sch.get_sblock("compute")
    sch.cache_read(block, 0, "local")
    assert_structural_equal_ignore_global_symbol(expected, sch.mod["main"])


def test_cache_write_sibling_nested_block_predicates_use_union():
    """Regression: cache_write with sibling nested blocks must union their predicates.

    Two sibling nested sblocks access the same tensor under *different* predicates:
      left  block: Ts.where(vi < 8)   — writes rows 0-7, all columns
      top   block: Ts.where(vj < 16)  — writes all rows, columns 0-15

    The cache must cover the UNION of both access sets.  The bounding box of that
    union is (12, 24) — the full tensor shape.

    Bug: CollectNestedBlockPredicates ANDs the predicates of all found nested blocks,
    giving (vi < 8) AND (vj < 16).  RelaxTensorRegion under that intersection predicate
    yields the bounding box of the *intersection* instead: (8, 16), which is too small.
    The "left" block then writes C_buf_local[vi, vj] for vi in [8,12) — indices that
    were never loaded into C_buf_local — resulting in incorrect output.
    """

    @Ts.function
    def main(A_buf: T.Tensor((12, 24), "float32"), C_buf: T.Tensor((12, 24), "float32")) -> None:
        for i, j in T.grid(12, 24):
            with Ts.sblock("compute"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                with Ts.sblock("left"):
                    Ts.where(vi < 8)
                    C_buf[vi, vj] = A_buf[vi, vj] * 2.0
                with Ts.sblock("top"):
                    Ts.where(vj < 16)
                    C_buf[vi, vj] = A_buf[vi, vj] * 3.0

    sch = tvm.s_tir.Schedule(main)
    block = sch.get_sblock("compute")
    sch.cache_write(block, 0, "local")

    # Extract the alloc tensor shape from the resulting IR.
    result_script = sch.mod["main"].script()
    # The cache must be large enough to hold the union of both write regions.
    # Union bounding box = full (12, 24).  The buggy AND gives (8, 16).
    assert "sblock_alloc_tensor((12, 24)" in result_script, (
        f"Expected cache shape (12, 24) covering the union of both write regions, "
        f"but got a smaller shape. Full IR:\n{result_script}"
    )


def test_cache_read_sibling_nested_block_predicates_use_union():
    """Regression: cache_read with sibling nested blocks must union their predicates.

    Two sibling nested sblocks read the same input tensor under different predicates:
      left  block: Ts.where(vi < 8)   — reads rows 0-7, all columns
      top   block: Ts.where(vj < 16)  — reads all rows, columns 0-15

    The cache must cover the UNION of both read sets.  The bounding box of that
    union is (12, 24) — the full tensor shape.

    Bug: CollectNestedBlockPredicates ANDs the two predicates, giving (vi < 8) AND
    (vj < 16).  Case 2 of CacheRead calls RelaxTensorRegion under that intersection
    predicate, producing a cache of shape (8, 16).  The "left" block then tries to
    read A_buf_local[vi, vj] for vi in [8,12) — indices outside the cache — which
    is incorrect.
    """

    @Ts.function
    def main(A_buf: T.Tensor((12, 24), "float32"), C_buf: T.Tensor((12, 24), "float32")) -> None:
        for i, j in T.grid(12, 24):
            with Ts.sblock("compute"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                with Ts.sblock("left"):
                    Ts.where(vi < 8)
                    C_buf[vi, vj] = A_buf[vi, vj] * 2.0
                with Ts.sblock("top"):
                    Ts.where(vj < 16)
                    C_buf[vi, vj] = A_buf[vi, vj] * 3.0

    sch = tvm.s_tir.Schedule(main)
    block = sch.get_sblock("compute")
    sch.cache_read(block, 0, "local")

    result_script = sch.mod["main"].script()
    # Cache must cover the union bounding box (12, 24).  Buggy AND gives (8, 16).
    assert "sblock_alloc_tensor((12, 24)" in result_script, (
        f"Expected cache shape (12, 24) covering the union of both read regions, "
        f"but got a smaller shape. Full IR:\n{result_script}"
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
