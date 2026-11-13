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
# ruff: noqa: F401, F841
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
def matmul_bias_before(
    A: T.Tensor((16, 16), "int8"),
    B: T.Tensor((16, 16), "int8"),
    C: T.Tensor((16, 16), "int32"),
    D: T.Tensor((16, 16), "int32"),
) -> None:
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("multiply"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.int32(0)
            temp[vi, vj] = temp[vi, vj] + T.cast(A[vi, vk], ty="int32") * T.cast(
                B[vj, vk], ty="int32"
            )
    for i, j in T.grid(16, 16):
        with Ts.sblock("add"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj] + C[vi, vj]


@Ts.function
def matmul_bias_expected(
    A: T.Tensor((16, 16), "int8"),
    B: T.Tensor((16, 16), "int8"),
    C: T.Tensor((16, 16), "int32"),
    D: T.Tensor((16, 16), "int32"),
) -> None:
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("multiply"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            Ts.writes(D[vi, vj])
            with Ts.init():
                D[vi, vj] = C[vi, vj]
            D[vi, vj] = D[vi, vj] + T.cast(A[vi, vk], ty="int32") * T.cast(B[vj, vk], ty="int32")


@Ts.function
def matmul_bias_fp32_before(
    A: T.Tensor((32, 32), "float32"),
    B: T.Tensor((32, 32), "float32"),
    C: T.Tensor((32, 32), "float32"),
    D: T.Tensor((32, 32), "float32"),
) -> None:
    temp = Ts.sblock_alloc_tensor((32, 32), dtype="float32")
    for i, j, k in T.grid(32, 32, 32):
        with Ts.sblock("multiply"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.float32(0)
            temp[vi, vj] = temp[vi, vj] + A[vi, vk] * B[vj, vk]
    for i, j in T.grid(32, 32):
        with Ts.sblock("add"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj] + C[vi, vj]


@Ts.function
def matmul_bias_fp32_expected(
    A: T.Tensor((32, 32), "float32"),
    B: T.Tensor((32, 32), "float32"),
    C: T.Tensor((32, 32), "float32"),
    D: T.Tensor((32, 32), "float32"),
) -> None:
    temp = Ts.sblock_alloc_tensor((32, 32), dtype="float32")
    for i, j, k in T.grid(32, 32, 32):
        with Ts.sblock("multiply"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            Ts.writes(D[vi, vj])
            with Ts.init():
                D[vi, vj] = C[vi, vj]
            D[vi, vj] = D[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.function
def matmul_bias_multiple_epilogue_before(
    A: T.Tensor((16, 16), "int8"),
    B: T.Tensor((16, 16), "int8"),
    C: T.Tensor((16, 16), "int32"),
    D: T.Tensor((16, 16), "int32"),
    E: T.Tensor((16, 16), "int32"),
) -> None:
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("multiply"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.int32(0)
            temp[vi, vj] = temp[vi, vj] + T.cast(A[vi, vk], ty="int32") * T.cast(
                B[vj, vk], ty="int32"
            )
    for i, j in T.grid(16, 16):
        with Ts.sblock("add"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            D[vi, vj] = temp[vi, vj] + C[vi, vj]
    for i, j in T.grid(16, 16):
        with Ts.sblock("add2"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            E[vi, vj] = temp[vi, vj] + C[vi, vj]


@Ts.function
def matmul_bias_multiple_epilogue_expected(
    A: T.Tensor((16, 16), "int8"),
    B: T.Tensor((16, 16), "int8"),
    C: T.Tensor((16, 16), "int32"),
    D: T.Tensor((16, 16), "int32"),
    E: T.Tensor((16, 16), "int32"),
) -> None:
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("multiply"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            Ts.writes(D[vi, vj])
            with Ts.init():
                D[vi, vj] = C[vi, vj]
            D[vi, vj] = D[vi, vj] + T.cast(A[vi, vk], ty="int32") * T.cast(B[vj, vk], ty="int32")
    for i, j in T.grid(16, 16):
        with Ts.sblock("add2"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(temp[vi, vj], C[vi, vj])
            Ts.writes(E[vi, vj])
            E[vi, vj] = temp[vi, vj] + C[vi, vj]


def test_fuse_reduction_epilogue_basic():
    sch = tvm.s_tir.Schedule(matmul_bias_before, debug_mask="all")
    sch.fuse_reduction_epilogue("multiply", "add")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_before)


def test_fuse_reduction_epilogue_fp32():
    sch = tvm.s_tir.Schedule(matmul_bias_fp32_before, debug_mask="all")
    sch.fuse_reduction_epilogue("multiply", "add")
    assert_structural_equal_ignore_global_symbol(sch.mod["main"], matmul_bias_fp32_expected)
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_fp32_before)


def test_fuse_reduction_epilogue_numerical_correctness():
    sch_original = tvm.s_tir.Schedule(matmul_bias_before, debug_mask="all")
    mod_original = tvm.compile(sch_original.mod["main"], target="llvm")

    sch_fused = tvm.s_tir.Schedule(matmul_bias_before, debug_mask="all")
    sch_fused.fuse_reduction_epilogue("multiply", "add")
    mod_fused = tvm.compile(sch_fused.mod["main"], target="llvm")

    A_np = np.random.randint(-128, 127, size=(16, 16), dtype="int8")
    B_np = np.random.randint(-128, 127, size=(16, 16), dtype="int8")
    C_np = np.random.randint(-1000, 1000, size=(16, 16), dtype="int32")

    expected = (A_np.astype("int32") @ B_np.T.astype("int32")) + C_np

    D_original_tvm = tvm_tirx.runtime.tensor(np.zeros((16, 16), dtype="int32"))
    D_fused_tvm = tvm_tirx.runtime.tensor(np.zeros((16, 16), dtype="int32"))

    mod_original(
        tvm_tirx.runtime.tensor(A_np),
        tvm_tirx.runtime.tensor(B_np),
        tvm_tirx.runtime.tensor(C_np),
        D_original_tvm,
    )

    mod_fused(
        tvm_tirx.runtime.tensor(A_np),
        tvm_tirx.runtime.tensor(B_np),
        tvm_tirx.runtime.tensor(C_np),
        D_fused_tvm,
    )

    D_original = D_original_tvm.numpy()
    D_fused = D_fused_tvm.numpy()

    tvm_tirx.testing.assert_allclose(D_original, expected, rtol=1e-5)
    tvm_tirx.testing.assert_allclose(D_fused, expected, rtol=1e-5)
    tvm_tirx.testing.assert_allclose(D_fused, D_original, rtol=1e-5)


def test_fuse_reduction_epilogue_multiple_epilogue():
    sch = tvm.s_tir.Schedule(matmul_bias_multiple_epilogue_before, debug_mask="all")
    sch.fuse_reduction_epilogue("multiply", "add")
    assert_structural_equal_ignore_global_symbol(
        sch.mod["main"], matmul_bias_multiple_epilogue_expected
    )
    verify_trace_roundtrip(sch=sch, mod=matmul_bias_multiple_epilogue_before)

    mod = tvm.compile(sch.mod["main"], target="llvm")
    assert mod is not None


@Ts.function
def matmul_bias_invalid_multiple_use_before(
    A: T.Tensor((16, 16), "int8"),
    B: T.Tensor((16, 16), "int8"),
    C1: T.Tensor((16, 16), "int32"),
    C2: T.Tensor((16, 16), "int32"),
    D: T.Tensor((16, 16), "int32"),
) -> None:
    """Epilogue uses the reduction result twice; fusion must be rejected."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("multiply"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.int32(0)
            temp[vi, vj] = temp[vi, vj] + T.cast(A[vi, vk], ty="int32") * T.cast(
                B[vj, vk], ty="int32"
            )
    for i, j in T.grid(16, 16):
        with Ts.sblock("bad_epilogue"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            # temp[vi, vj] is used twice in the epilogue expression
            D[vi, vj] = (temp[vi, vj] + C1[vi, vj]) * (temp[vi, vj] + C2[vi, vj])


def test_fuse_reduction_epilogue_reject_multiple_use():
    """fusion should be rejected when the reduction result appears more than once."""
    sch = tvm.s_tir.Schedule(matmul_bias_invalid_multiple_use_before, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.fuse_reduction_epilogue("multiply", "bad_epilogue")


@Ts.function
def matmul_bias_invalid_scaling_before(
    A: T.Tensor((16, 16), "int8"),
    B: T.Tensor((16, 16), "int8"),
    C: T.Tensor((16, 16), "int32"),
    D: T.Tensor((16, 16), "int32"),
) -> None:
    """Epilogue scales the reduction result; fusion must be rejected."""
    temp = Ts.sblock_alloc_tensor((16, 16), dtype="int32")
    for i, j, k in T.grid(16, 16, 16):
        with Ts.sblock("multiply"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                temp[vi, vj] = T.int32(0)
            temp[vi, vj] = temp[vi, vj] + T.cast(A[vi, vk], ty="int32") * T.cast(
                B[vj, vk], ty="int32"
            )
    for i, j in T.grid(16, 16):
        with Ts.sblock("scaled_epilogue"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            # temp[vi, vj] is scaled by 2 before adding bias; this must not be fused.
            D[vi, vj] = temp[vi, vj] * T.int32(2) + C[vi, vj]


def test_fuse_reduction_epilogue_reject_scaling():
    """fusion should be rejected when the reduction result is scaled by Mul/Div/Mod."""
    sch = tvm.s_tir.Schedule(matmul_bias_invalid_scaling_before, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.fuse_reduction_epilogue("multiply", "scaled_epilogue")


if __name__ == "__main__":
    tvm_tirx.testing.main()
