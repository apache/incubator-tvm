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
# ruff: noqa: E501, F401, F841
import numpy as np
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


@Ts.function
def matmul_clipping_before(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
    lower: T.float32,
    upper: T.float32,
) -> None:
    """Original function with separate reduction and clipping epilogue blocks."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with Ts.sblock("clipping"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = T.min(T.max(temp[vi, vj], lower), upper)


@Ts.function
def matmul_clipping_expected(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
    lower: T.float32,
    upper: T.float32,
) -> None:
    """Expected function after fusion (Clipping)."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(A[vi, vk], B[vj, vk])
            Ts.writes(D[vi, vj])
            with Ts.init():
                D[vi, vj] = T.min(T.max(T.float32(0), lower), upper)
            D[vi, vj] = T.min(T.max(D[vi, vj] + A[vi, vk] * B[vj, vk], lower), upper)


def test_matmul_clipping():
    """Test fusion of matmul with clipping epilogue."""
    sch = tvm.s_tir.Schedule(matmul_clipping_before, debug_mask="all")
    sch.fuse_reduction_epilogue("matmul", "clipping")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_clipping_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_clipping_before)


@Ts.function
def matmul_clipping_before_per_iteration(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
) -> None:
    """Original function with per-iteration clipping (same semantics as fused)."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    lower = T.float32(-5.0)
    upper = T.float32(5.0)
    for i, j in T.grid(16, 16):
        with Ts.sblock("init"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            temp[vi, vj] = T.min(T.max(T.float32(0), lower), upper)  # Clip init

    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            # Per-iteration clipping
            temp[vi, vj] = T.min(T.max(temp[vi, vj] + A[vi, vk] * B[vj, vk], lower), upper)

    for i, j in T.grid(16, 16):
        with Ts.sblock("copy"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj]


def test_matmul_clipping_correctness_unified():
    """Test that original and fused produce identical results with per-iteration clipping."""
    A_np = np.random.randn(16, 16).astype("float32")
    B_np = np.random.randn(16, 16).astype("float32")
    lower = -5.0
    upper = 5.0

    # NumPy reference for per-iteration clipping
    D_ref = np.clip(0.0, lower, upper)  # init with clipping
    for k in range(16):
        D_ref = np.clip(D_ref + np.outer(A_np[:, k], B_np[:, k]), lower, upper)

    # TVM execution (original with per-iteration clipping)
    mod_original = tvm.compile(matmul_clipping_before_per_iteration, target="llvm")
    D_original_tvm = tvm_tirx.runtime.tensor(np.zeros((16, 16), dtype="float32"))
    mod_original(
        tvm_tirx.runtime.tensor(A_np),
        tvm_tirx.runtime.tensor(B_np),
        D_original_tvm,
    )

    # TVM execution (fused)
    sch = tvm.s_tir.Schedule(matmul_clipping_before)
    sch.fuse_reduction_epilogue("matmul", "clipping")
    mod_fused = tvm.compile(sch.mod["main"], target="llvm")
    D_fused_tvm = tvm_tirx.runtime.tensor(np.zeros((16, 16), dtype="float32"))
    # Pass scalar values directly as Python floats
    mod_fused(
        tvm_tirx.runtime.tensor(A_np),
        tvm_tirx.runtime.tensor(B_np),
        D_fused_tvm,
        lower,
        upper,
    )

    D_original = D_original_tvm.numpy()
    D_fused = D_fused_tvm.numpy()

    # Now both should match exactly
    np.testing.assert_allclose(D_original, D_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(D_fused, D_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(D_original, D_fused, rtol=1e-5, atol=1e-6)


@Ts.function
def matmul_clipping_multiple_epilogue_before(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
    E: T.Tensor((16, 16), "float32"),
    lower: T.float32,
    upper: T.float32,
) -> None:
    """Original function with separate reduction and multiple epilogue blocks (one with clipping, one without)."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with Ts.sblock("clipping"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = T.min(T.max(temp[vi, vj], lower), upper)

    for i, j in T.grid(16, 16):
        with Ts.sblock("copy"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            E[vi, vj] = temp[vi, vj]


@Ts.function
def matmul_clipping_multiple_epilogue_expected(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
    E: T.Tensor((16, 16), "float32"),
    lower: T.float32,
    upper: T.float32,
) -> None:
    """Expected function after fusion (Clipping) with multiple epilogue blocks."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(A[vi, vk], B[vj, vk])
            Ts.writes(D[vi, vj])
            with Ts.init():
                D[vi, vj] = T.min(T.max(T.float32(0), lower), upper)
            D[vi, vj] = T.min(T.max(D[vi, vj] + A[vi, vk] * B[vj, vk], lower), upper)
    for i, j in T.grid(16, 16):
        with Ts.sblock("copy"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(temp[vi, vj])
            Ts.writes(E[vi, vj])
            E[vi, vj] = temp[vi, vj]


def test_matmul_clipping_multiple_epilogue():
    """Test fusion with multiple epilogue blocks - one with clipping, one without.

    Following the same pattern as test_fuse_reduction_epilogue_multiple_epilogue,
    this test verifies that fusion works correctly when there are multiple
    epilogue blocks. The temp tensor is kept because the second epilogue block
    still needs it.
    """
    sch = tvm.s_tir.Schedule(matmul_clipping_multiple_epilogue_before, debug_mask="all")
    sch.fuse_reduction_epilogue("matmul", "clipping")
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], matmul_clipping_multiple_epilogue_expected
    )
    verify_trace_roundtrip(sch=sch, mod=matmul_clipping_multiple_epilogue_before)

    mod = tvm.compile(sch.mod["main"], target="llvm")
    assert mod is not None


# Test commutative variants of clipping patterns
@pytest.mark.parametrize(
    "pattern_func",
    [
        lambda temp, lower, upper: T.min(T.max(temp, lower), upper),  # min(max(temp, lower), upper)
        lambda temp, lower, upper: T.min(upper, T.max(temp, lower)),  # min(upper, max(temp, lower))
        lambda temp, lower, upper: T.min(T.max(lower, temp), upper),  # min(max(lower, temp), upper)
        lambda temp, lower, upper: T.max(T.min(temp, upper), lower),  # max(min(temp, upper), lower)
        lambda temp, lower, upper: T.max(lower, T.min(temp, upper)),  # max(lower, min(temp, upper))
    ],
)
def test_matmul_clipping_commutative_variants(pattern_func):
    """Test that all commutative variants of clipping patterns are recognized."""
    lower = -5.0
    upper = 5.0

    @Ts.function
    def test_func(
        A: T.Tensor((8, 8), "float32"),
        B: T.Tensor((8, 8), "float32"),
        D: T.Tensor((8, 8), "float32"),
    ) -> None:
        temp = Ts.sblock_alloc_tensor((8, 8), dtype="float32")
        for i, j, k in T.grid(8, 8, 8):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    temp[vi, vj] = T.float32(0)
                temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]

        for i, j in T.grid(8, 8):
            with Ts.sblock("clipping"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                D[vi, vj] = pattern_func(temp[vi, vj], T.float32(lower), T.float32(upper))

    sch = tvm.s_tir.Schedule(test_func, debug_mask="all")
    # Should not raise an error - all variants should be recognized
    sch.fuse_reduction_epilogue("matmul", "clipping")
    verify_trace_roundtrip(sch=sch, mod=test_func)


if __name__ == "__main__":
    tvm_tirx.testing.main()
