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
def matmul_bias_relu_before(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
) -> None:
    """Original function with separate reduction and epilogue blocks (Bias + ReLU)."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with Ts.sblock("bias_relu"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = T.max(temp[vi, vj] + C[vi, vj], T.float32(0))


@Ts.function
def matmul_bias_relu_before_per_iteration(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
) -> None:
    """Original function with per-iteration ReLU (same semantics as fused)."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j in T.grid(16, 16):
        with Ts.sblock("init"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            temp[vi, vj] = T.max(C[vi, vj], T.float32(0))  # ReLU on bias

    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            # Per-iteration ReLU
            temp[vi, vj] = T.max(temp[vi, vj] + A[vi, vk] * B[vj, vk], T.float32(0))

    for i, j in T.grid(16, 16):
        with Ts.sblock("copy"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj]


@Ts.function
def matmul_bias_relu_expected(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
) -> None:
    """Expected function after fusion (Bias + ReLU)."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            Ts.writes(D[vi, vj])
            with Ts.init():
                D[vi, vj] = T.max(C[vi, vj], T.float32(0))
            D[vi, vj] = T.max(D[vi, vj] + A[vi, vk] * B[vj, vk], T.float32(0))


def test_matmul_bias_relu():
    """Test fusion of matmul with bias + ReLU epilogue."""
    sch = tvm.s_tir.Schedule(matmul_bias_relu_before, debug_mask="all")
    sch.fuse_reduction_epilogue("matmul", "bias_relu")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_relu_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_relu_before)


def test_matmul_bias_relu_correctness_unified():
    """Test that original and fused produce identical results with per-iteration ReLU."""
    A_np = np.random.randn(16, 16).astype("float32")
    B_np = np.random.randn(16, 16).astype("float32")
    C_np = np.random.randn(16, 16).astype("float32")

    # NumPy reference for per-iteration ReLU
    # Simulate per-iteration ReLU behavior
    # Original code computes A[vi, vk] * B[vj, vk] which is A[i, k] * B[j, k]
    # For each k: add outer product of A[:, k] and B[:, k]
    D_ref = np.maximum(C_np, 0)  # init with ReLU on bias
    for k in range(16):
        # A[:, k] is shape (16,), B[:, k] is shape (16,)
        # Outer product: A[:, k] * B[:, k] for all i, j = A[i, k] * B[j, k]
        # Using broadcasting: A[:, k:k+1] * B[:, k:k+1].T gives (16, 1) * (1, 16) = (16, 16)
        D_ref = np.maximum(D_ref + np.outer(A_np[:, k], B_np[:, k]), 0)

    # TVM execution (original with per-iteration ReLU)
    mod_original = tvm.compile(matmul_bias_relu_before_per_iteration, target="llvm")
    D_original_tvm = tvm_tirx.runtime.tensor(np.zeros((16, 16), dtype="float32"))
    mod_original(
        tvm_tirx.runtime.tensor(A_np),
        tvm_tirx.runtime.tensor(B_np),
        tvm_tirx.runtime.tensor(C_np),
        D_original_tvm,
    )

    # TVM execution (fused)
    sch = tvm.s_tir.Schedule(matmul_bias_relu_before)
    sch.fuse_reduction_epilogue("matmul", "bias_relu")
    mod_fused = tvm.compile(sch.mod["main"], target="llvm")
    D_fused_tvm = tvm_tirx.runtime.tensor(np.zeros((16, 16), dtype="float32"))
    mod_fused(
        tvm_tirx.runtime.tensor(A_np),
        tvm_tirx.runtime.tensor(B_np),
        tvm_tirx.runtime.tensor(C_np),
        D_fused_tvm,
    )

    D_original = D_original_tvm.numpy()
    D_fused = D_fused_tvm.numpy()

    # Now both should match exactly
    np.testing.assert_allclose(D_original, D_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(D_fused, D_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(D_original, D_fused, rtol=1e-5, atol=1e-6)


@Ts.function
def matmul_bias_relu_multiple_epilogue_before(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
    E: T.Tensor((16, 16), "float32"),
) -> None:
    """Original function with separate reduction and multiple epilogue blocks (one with ReLU, one without)."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]

    for i, j in T.grid(16, 16):
        with Ts.sblock("bias_relu"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = T.max(temp[vi, vj] + C[vi, vj], T.float32(0))

    for i, j in T.grid(16, 16):
        with Ts.sblock("bias"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            E[vi, vj] = temp[vi, vj] + C[vi, vj]


@Ts.function
def matmul_bias_relu_multiple_epilogue_expected(
    A: T.Tensor((16, 16), "float32"),
    B: T.Tensor((16, 16), "float32"),
    C: T.Tensor((16, 16), "float32"),
    D: T.Tensor((16, 16), "float32"),
    E: T.Tensor((16, 16), "float32"),
) -> None:
    """Expected function after fusion (Bias + ReLU) with multiple epilogue blocks."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="float32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("matmul"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            Ts.writes(D[vi, vj])
            with Ts.init():
                D[vi, vj] = T.max(C[vi, vj], T.float32(0))
            D[vi, vj] = T.max(D[vi, vj] + A[vi, vk] * B[vj, vk], T.float32(0))
    for i, j in T.grid(16, 16):
        with Ts.sblock("bias"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(temp[vi, vj], C[vi, vj])
            Ts.writes(E[vi, vj])
            E[vi, vj] = temp[vi, vj] + C[vi, vj]


def test_matmul_bias_relu_multiple_epilogue():
    """Test fusion with multiple epilogue blocks - one with ReLU, one without.

    Following the same pattern as test_fuse_reduction_epilogue_multiple_epilogue,
    this test verifies that fusion works correctly when there are multiple
    epilogue blocks. The temp tensor is kept because the second epilogue block
    still needs it.
    """
    sch = tvm.s_tir.Schedule(matmul_bias_relu_multiple_epilogue_before, debug_mask="all")
    sch.fuse_reduction_epilogue("matmul", "bias_relu")
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], matmul_bias_relu_multiple_epilogue_expected
    )
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_relu_multiple_epilogue_before)

    mod = tvm.compile(sch.mod["main"], target="llvm")
    assert mod is not None


if __name__ == "__main__":
    tvm_tirx.testing.main()
