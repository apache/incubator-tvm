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
# ruff: noqa: F401
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

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@Ts.function
def matmul_before(
    A: T.Tensor((128, 127), "float32"),
    B: T.Tensor((127, 127), "float32"),
    C: T.Tensor((128, 127), "float32"),
) -> None:
    A_shared = Ts.sblock_alloc_tensor((128, 127), "float32", scope="shared")
    B_shared = Ts.sblock_alloc_tensor((127, 127), "float32", scope="shared")
    C_shared = Ts.sblock_alloc_tensor((128, 127), "float32", scope="shared")
    for i0, i1 in T.grid(128, 127):
        with Ts.sblock("A"):
            i, j = Ts.axis.remap("SS", [i0, i1])
            A_shared[i, j] = A[i, j]
    for i0, i1 in T.grid(127, 127):
        with Ts.sblock("B"):
            i, j = Ts.axis.remap("SS", [i0, i1])
            B_shared[i, j] = B[i, j]
    for i0, i1, i2 in T.grid(128, 127, 127):
        with Ts.sblock("C_shared"):
            i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
            with Ts.init():
                C_shared[i, j] = T.float32(0)
            C_shared[i, j] = C_shared[i, j] + A_shared[i, k] * B_shared[k, j]
    for i0, i1 in T.grid(128, 127):
        with Ts.sblock("C"):
            i, j = Ts.axis.remap("SS", [i0, i1])
            C[i, j] = C_shared[i, j]


@Ts.function
def matmul_expected(
    A: T.Tensor((128, 127), "float32"),
    B: T.Tensor((127, 127), "float32"),
    C: T.Tensor((128, 127), "float32"),
) -> None:
    A_shared_padded = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="shared")
    B_shared_padded = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="shared")
    C_shared_padded = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="shared")
    for i0, i1 in T.grid(128, 128):
        with Ts.sblock("A"):
            i, j = Ts.axis.remap("SS", [i0, i1])
            Ts.reads(A[i, j])
            Ts.writes(A_shared_padded[i, j])
            A_shared_padded[i, j] = T.if_then_else(j < 127, A[i, j], T.float32(0))
    for i0, i1 in T.grid(128, 128):
        with Ts.sblock("B"):
            i, j = Ts.axis.remap("SS", [i0, i1])
            Ts.reads(B[i, j])
            Ts.writes(B_shared_padded[i, j])
            B_shared_padded[i, j] = T.if_then_else(i < 127 and j < 127, B[i, j], T.float32(0))
    for i0, i1, i2 in T.grid(128, 128, 128):
        with Ts.sblock("C_shared"):
            i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
            Ts.reads(A_shared_padded[i, k], B_shared_padded[k, j])
            Ts.writes(C_shared_padded[i, j])
            with Ts.init():
                C_shared_padded[i, j] = T.float32(0)
            C_shared_padded[i, j] = (
                C_shared_padded[i, j] + A_shared_padded[i, k] * B_shared_padded[k, j]
            )
    for i0, i1 in T.grid(128, 127):
        with Ts.sblock("C"):
            i, j = Ts.axis.remap("SS", [i0, i1])
            Ts.reads(C_shared_padded[i, j])
            Ts.writes(C[i, j])
            C[i, j] = C_shared_padded[i, j]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def test_pad_matmul():
    # pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg

    n = T.dynamic("n", "int32")

    @Ts.function
    def matmul_before(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((n, 128), "float32"),
        C: T.Tensor((128, n), "float32"),
    ) -> None:
        for i0, i1, i2 in T.grid(128, n, 128):
            with Ts.sblock("C"):
                i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
                with Ts.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[j, k]

    n = T.dynamic("n", "int32")

    @Ts.function
    def matmul_after(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((n, 128), "float32"),
        C: T.Tensor((128, n), "float32"),
    ):
        B_pad = Ts.sblock_alloc_tensor(((n + 31) // 32 * 32, 128))
        C_pad = Ts.sblock_alloc_tensor((128, (n + 31) // 32 * 32))
        for i0, i1 in T.grid((n + 31) // 32 * 32, 128):
            with Ts.sblock("B_pad"):
                v0, v1 = Ts.axis.remap("SS", [i0, i1])
                B_pad[v0, v1] = T.if_then_else(v0 < n, B[v0, v1], T.float32(0))
        for i0, i1, i2 in T.grid(128, (n + 31) // 32 * 32, 128):
            with Ts.sblock("C"):
                i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
                Ts.reads(A[i, k], B_pad[j, k])
                Ts.writes(C_pad[i, j])
                with Ts.init():
                    C_pad[i, j] = T.float32(0)
                C_pad[i, j] = C_pad[i, j] + A[i, k] * B_pad[j, k]
        for i0, i1 in T.grid(128, n):
            with Ts.sblock("C_pad"):
                v0, v1 = Ts.axis.remap("SS", [i0, i1])
                C[v0, v1] = C_pad[v0, v1]

    sch = tvm.s_tir.Schedule(matmul_before, debug_mask="all")
    C = sch.get_sblock("C")
    sch.pad_einsum(C, [32, 32, 32])
    assert_structural_equal_ignore_global_symbol(matmul_after, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=matmul_before)


def test_pad_matmul_2():
    n = T.dynamic("n", "int32")

    @Ts.function
    def before(
        A: T.Tensor((1, n, 4096), "float32"),
        B: T.Tensor((11008, 4096), "float32"),
        M: T.Tensor((1, n, 11008), "float32"),
        D: T.Tensor((1, n, 11008), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})

        C = Ts.sblock_alloc_tensor((1, n, 11008))
        for i0, i1, i2, k in T.grid(1, n, 11008, 4096):
            with Ts.sblock("C"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                Ts.writes(C[v_i0, v_i1, v_i2])
                with Ts.init():
                    C[v_i0, v_i1, v_i2] = T.float32(0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(1, n, 11008):
            with Ts.sblock("D"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                D[v_ax0, v_ax1, v_ax2] = M[v_ax0, v_ax1, v_ax2] * C[v_ax0, v_ax1, v_ax2]

    n = T.dynamic("n", "int32")

    @Ts.function
    def after(
        A: T.Tensor((1, n, 4096), "float32"),
        B: T.Tensor((11008, 4096), "float32"),
        M: T.Tensor((1, n, 11008), "float32"),
        D: T.Tensor((1, n, 11008), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})

        # with Ts.sblock("root"):
        C = Ts.sblock_alloc_tensor((1, n, 11008))
        A_pad = Ts.sblock_alloc_tensor((1, (n + 31) // 32 * 32, 4096))
        C_pad = Ts.sblock_alloc_tensor((1, (n + 31) // 32 * 32, 11008))
        for i0, i1, i2 in T.grid(1, (n + 31) // 32 * 32, 4096):
            with Ts.sblock("A_pad"):
                v0, v1, v2 = Ts.axis.remap("SSS", [i0, i1, i2])
                A_pad[v0, v1, v2] = T.if_then_else(v1 < n, A[v0, v1, v2], T.float32(0))
        for i0, i1, i2, k in T.grid(1, (n + 31) // 32 * 32, 11008, 4096):
            with Ts.sblock("C"):
                v_i0, v_i1, v_i2, v_k = Ts.axis.remap("SSSR", [i0, i1, i2, k])
                Ts.reads(A_pad[v_i0, v_i1, v_k], B[v_i2, v_k])
                Ts.writes(C_pad[v_i0, v_i1, v_i2])
                with Ts.init():
                    C_pad[v_i0, v_i1, v_i2] = T.float32(0)
                C_pad[v_i0, v_i1, v_i2] = (
                    C_pad[v_i0, v_i1, v_i2] + A_pad[v_i0, v_i1, v_k] * B[v_i2, v_k]
                )
        for i0, i1, i2 in T.grid(1, n, 11008):
            with Ts.sblock("C_pad"):
                v0, v1, v2 = Ts.axis.remap("SSS", [i0, i1, i2])
                C[v0, v1, v2] = C_pad[v0, v1, v2]
        for ax0, ax1, ax2 in T.grid(1, n, 11008):
            with Ts.sblock("D"):
                v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                D[v_ax0, v_ax1, v_ax2] = M[v_ax0, v_ax1, v_ax2] * C[v_ax0, v_ax1, v_ax2]

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    C = sch.get_sblock("C")
    sch.pad_einsum(C, [1, 32, 32, 32])
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=before)


def test_pad_rms():
    n = T.dynamic("n", "int32")

    @Ts.function
    def before(
        A: T.Tensor((1, n, 4096), "float32"),
        W: T.Tensor((4096,), "float32"),
        Result: T.Tensor((1, n, 4096), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})

        S = Ts.sblock_alloc_tensor((1, n), "float32")
        for bsz, i, k in T.grid(1, n, 4096):
            with Ts.sblock("S"):
                v_bsz, v_i, v_k = Ts.axis.remap("SSR", [bsz, i, k])
                Ts.reads(A[v_bsz, v_i, v_k])
                Ts.writes(S[v_bsz, v_i])
                with Ts.init():
                    S[v_bsz, v_i] = T.float32(0)
                S[v_bsz, v_i] = S[v_bsz, v_i] + A[v_bsz, v_i, v_k] * A[v_bsz, v_i, v_k]
        for bsz, i, k in T.grid(1, n, 4096):
            with Ts.sblock("R"):
                v_bsz, v_i, v_k = Ts.axis.remap("SSS", [bsz, i, k])
                Result[v_bsz, v_i, v_k] = W[v_k] * (
                    A[v_bsz, v_i, v_k]
                    / T.sqrt(S[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(1e-6))
                )

    n = T.dynamic("n", "int32")

    @Ts.function
    def after(
        A: T.Tensor((1, n, 4096), "float32"),
        W: T.Tensor((4096,), "float32"),
        Result: T.Tensor((1, n, 4096), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})

        S = Ts.sblock_alloc_tensor((1, n))
        A_pad = Ts.sblock_alloc_tensor((1, (n + 31) // 32 * 32, 4096))
        S_pad = Ts.sblock_alloc_tensor((1, (n + 31) // 32 * 32))
        for i0, i1, i2 in T.grid(1, (n + 31) // 32 * 32, 4096):
            with Ts.sblock("A_pad"):
                v0, v1, v2 = Ts.axis.remap("SSS", [i0, i1, i2])
                A_pad[v0, v1, v2] = T.if_then_else(v1 < n, A[v0, v1, v2], T.float32(0))
        for bsz, i, k in T.grid(1, (n + 31) // 32 * 32, 4096):
            with Ts.sblock("S"):
                v_bsz, v_i, v_k = Ts.axis.remap("SSR", [bsz, i, k])
                Ts.reads(A_pad[v_bsz, v_i, v_k])
                Ts.writes(S_pad[v_bsz, v_i])
                with Ts.init():
                    S_pad[v_bsz, v_i] = T.float32(0)
                S_pad[v_bsz, v_i] = (
                    S_pad[v_bsz, v_i] + A_pad[v_bsz, v_i, v_k] * A_pad[v_bsz, v_i, v_k]
                )
        for i0, i1 in T.grid(1, n):
            with Ts.sblock("S_pad"):
                v0, v1 = Ts.axis.remap("SS", [i0, i1])
                S[v0, v1] = S_pad[v0, v1]
        for bsz, i, k in T.grid(1, n, 4096):
            with Ts.sblock("R"):
                v_bsz, v_i, v_k = Ts.axis.remap("SSS", [bsz, i, k])
                Result[v_bsz, v_i, v_k] = W[v_k] * (
                    A[v_bsz, v_i, v_k]
                    / T.sqrt(S[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(1e-6))
                )

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    C = sch.get_sblock("S")
    sch.pad_einsum(C, [1, 32, 1])
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=before)


if __name__ == "__main__":
    test_pad_matmul()
    test_pad_matmul_2()
    test_pad_rms()
