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
# pylint: disable=missing-docstring
import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import te
from tvm.s_tir.tensor_intrin.rocm import (
    ROCM_MFMA_f16f16f32_INTRIN,
    ROCM_MFMA_f32f32f32_INTRIN,
    ROCM_MFMA_fill_16x16_f32_INTRIN,
    ROCM_MFMA_fill_16x16_i32_INTRIN,
    ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN,
    ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN,
    ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN,
    ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN,
    ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
    ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN,
    ROCM_MFMA_s8s8s32_INTRIN,
    ROCM_MFMA_STORE_16x16_f32_INTRIN,
    ROCM_MFMA_STORE_16x16_s32_INTRIN,
    shared_4x16_to_local_64x1_layout_B,
    shared_16x4_to_local_64x1_layout_A,
    shared_16x16_to_local_64x4_layout_A,
    shared_16x16_to_local_64x4_layout_B,
    shared_16x16_to_local_64x4_layout_C,
)
from tvm.s_tir.testing.tir import mfma_schedule
from tvm_tirx.testing import env

M = 1024
N = 1024
K = 1024
measure_perf = False
gflops = (N * M * K) * 2 / 1e9


def matmul(m, n, k, in_dtype, out_dtype, b_transposed):
    b_shape = (n, k) if b_transposed else (k, n)
    a = te.placeholder((m, k), name="A", dtype=in_dtype)
    b = te.placeholder(b_shape, name="B", dtype=in_dtype)
    k = te.reduce_axis((0, k), name="k")

    def maybe_cast(v):
        if in_dtype != out_dtype:
            return tvm_tirx.tirx.Cast(out_dtype, v)
        return v

    def maybe_swap(i, j):
        if b_transposed:
            return j, i
        return i, j

    c = te.compute(
        (m, n),
        lambda i, j: te.sum(maybe_cast(a[i, k]) * maybe_cast(b[maybe_swap(k, j)]), axis=[k]),
        name="C",
    )
    return (a, b, c)


def run_test(
    k_inner,
    in_dtype,
    out_dtype,
    b_transposed,
    i_factors,
    j_factors,
    k_factors,
    index_map_A,
    index_map_B,
    index_map_C,
    ldmatrix_a_intrin,
    ldmatrix_b_intrin,
    mma_intrin,
    mma_fill_intrin,
    mma_store_intrin,
):
    sch = mfma_schedule(
        te.create_function(matmul(M, N, K, in_dtype, out_dtype, b_transposed)),
        k_inner,
        in_dtype,
        b_transposed,
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        ldmatrix_a_intrin,
        ldmatrix_b_intrin,
        mma_intrin,
        mma_fill_intrin,
        mma_store_intrin,
    )

    f = tvm.compile(sch.mod["main"], target="rocm").jit()

    if in_dtype == "float32":
        a_np = np.random.uniform(size=(M, K)).astype("float32")

        if b_transposed:
            b_np = np.random.uniform(size=(N, K)).astype("float32")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                out_dtype
            )
        else:
            b_np = np.random.uniform(size=(K, N)).astype("float32")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype(out_dtype)
    elif in_dtype == "float16":
        a_np = np.random.uniform(size=(M, K)).astype("float16")

        if b_transposed:
            b_np = np.random.uniform(size=(N, K)).astype("float16")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                out_dtype
            )
        else:
            b_np = np.random.uniform(size=(K, N)).astype("float16")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype(out_dtype)
    else:
        a_np = np.random.randint(-128, 128, (M, K)).astype("int8")

        if b_transposed:
            b_np = np.random.randint(-128, 128, (N, K)).astype("int8")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32").transpose()).astype(
                "int32"
            )
        else:
            b_np = np.random.randint(-128, 128, (K, N)).astype("int8")
            c_np = np.dot(a_np.astype("float32"), b_np.astype("float32")).astype("int32")

    def run_and_check(measure=False):
        dev = tvm_tirx.rocm(0)
        a = tvm_tirx.runtime.tensor(a_np, dev)
        b = tvm_tirx.runtime.tensor(b_np, dev)
        c = tvm_tirx.runtime.tensor(np.zeros((M, N), dtype=out_dtype), dev)
        if measure:
            return tvm.runtime.time_evaluator(f, f.entry_name, dev, number=500)(a, b, c)
        f(a, b, c)
        dev.sync()
        tvm_tirx.testing.assert_allclose(c.numpy(), c_np, rtol=1e-2, atol=1e-2)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)
    return lambda: tvm_tirx.testing.run_with_gpu_lock(run_and_check, True)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_matrixcore(), reason="need matrixcore")
def test_i8i8i32_m16n16k16():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_A(i % 16, j % 16),
        )

    def index_map_B(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_B(i % 16, j % 16),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_C(i % 16, j % 16),
        )

    k_inner = 16
    in_dtype = "int8"
    out_dtype = "int32"
    i_factors, j_factors, k_factors = [1, 8, 2, 4, 1], [1, 16, 2, 1, 2], [32, 2, 1]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN,
        ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN,
        ROCM_MFMA_s8s8s32_INTRIN,
        ROCM_MFMA_fill_16x16_i32_INTRIN,
        ROCM_MFMA_STORE_16x16_s32_INTRIN,
    )

    if measure_perf and timer:
        print("test_i8i8i32_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_matrixcore(), reason="need matrixcore")
def test_f16f16f32_m16n16k16():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_A(i % 16, j % 16),
        )

    def index_map_B(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_B(i % 16, j % 16),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_C(i % 16, j % 16),
        )

    k_inner = 16
    in_dtype = "float16"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [1, 8, 2, 4, 1], [1, 16, 2, 1, 2], [32, 2, 1]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN,
        ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
        ROCM_MFMA_f16f16f32_INTRIN,
        ROCM_MFMA_fill_16x16_f32_INTRIN,
        ROCM_MFMA_STORE_16x16_f32_INTRIN,
    )

    if measure_perf and timer:
        print("f16f16f32_m16n16k16: %f GFLOPS" % (gflops / (timer().mean)))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_matrixcore(), reason="need matrixcore")
def test_f32f32f32_m16n16k4():
    def index_map_A(i, j):
        return (
            i // 16,
            j // 4,
            *shared_16x4_to_local_64x1_layout_A(i % 16, j % 4),
        )

    def index_map_B(i, j):
        return (
            i // 4,
            j // 16,
            *shared_4x16_to_local_64x1_layout_B(i % 4, j % 16),
        )

    def index_map_C(i, j):
        return (
            i // 16,
            j // 16,
            *shared_16x16_to_local_64x4_layout_C(i % 16, j % 16),
        )

    k_inner = 4
    in_dtype = "float32"
    out_dtype = "float32"
    i_factors, j_factors, k_factors = [4, 2, 1, 4, 2], [4, 2, 2, 1, 4], [128, 2, 1]

    timer = run_test(
        k_inner,
        in_dtype,
        out_dtype,
        False,  # b_transposed
        i_factors,
        j_factors,
        k_factors,
        index_map_A,
        index_map_B,
        index_map_C,
        ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN,
        ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN,
        ROCM_MFMA_f32f32f32_INTRIN,
        ROCM_MFMA_fill_16x16_f32_INTRIN,
        ROCM_MFMA_STORE_16x16_f32_INTRIN,
    )

    if measure_perf and timer:
        print("test_f32f32f32_m16n16k4: %f GFLOPS" % (gflops / (timer().mean)))


@pytest.mark.parametrize(
    "k_inner,in_dtype,out_dtype,load_a,load_b,mfma,fill,store",
    [
        (
            4,
            "float32",
            "float32",
            ROCM_MFMA_LOAD_16x4_A_SHARED_f32_INTRIN,
            ROCM_MFMA_LOAD_16x4_B_SHARED_f32_INTRIN,
            ROCM_MFMA_f32f32f32_INTRIN,
            ROCM_MFMA_fill_16x16_f32_INTRIN,
            ROCM_MFMA_STORE_16x16_f32_INTRIN,
        ),
        (
            16,
            "float16",
            "float32",
            ROCM_MFMA_LOAD_16x16_A_SHARED_f16_INTRIN,
            ROCM_MFMA_LOAD_16x16_B_SHARED_f16_INTRIN,
            ROCM_MFMA_f16f16f32_INTRIN,
            ROCM_MFMA_fill_16x16_f32_INTRIN,
            ROCM_MFMA_STORE_16x16_f32_INTRIN,
        ),
        (
            16,
            "int8",
            "int32",
            ROCM_MFMA_LOAD_16x16_A_SHARED_s8_INTRIN,
            ROCM_MFMA_LOAD_16x16_B_SHARED_s8_INTRIN,
            ROCM_MFMA_s8s8s32_INTRIN,
            ROCM_MFMA_fill_16x16_i32_INTRIN,
            ROCM_MFMA_STORE_16x16_s32_INTRIN,
        ),
    ],
)
def test_mfma_per_lane_storage(k_inner, in_dtype, out_dtype, load_a, load_b, mfma, fill, store):
    """Tensorize and lower multi-tile, multi-wave GEMM without ROCm hardware."""
    from tvm.s_tir.pipeline import _lower_s_tir

    layout_a = (
        shared_16x4_to_local_64x1_layout_A if k_inner == 4 else shared_16x16_to_local_64x4_layout_A
    )
    layout_b = (
        shared_4x16_to_local_64x1_layout_B if k_inner == 4 else shared_16x16_to_local_64x4_layout_B
    )

    def index_map_a(i, j):
        return i // 16, j // k_inner, *layout_a(i % 16, j % k_inner)

    def index_map_b(i, j):
        return i // k_inner, j // 16, *layout_b(i % k_inner, j % 16)

    def index_map_c(i, j):
        return i // 16, j // 16, *shared_16x16_to_local_64x4_layout_C(i % 16, j % 16)

    sch = mfma_schedule(
        te.create_function(matmul(64, 64, 2 * k_inner, in_dtype, out_dtype, False)),
        k_inner,
        in_dtype,
        False,
        [2, 1, 1, 2, 1],
        [1, 1, 2, 1, 2],
        [1, 2, 1],
        index_map_a,
        index_map_b,
        index_map_c,
        load_a,
        load_b,
        mfma,
        fill,
        store,
    )
    lowered = _lower_s_tir()(sch.mod)["main"]
    allocations = []
    accesses = []

    def check(node):
        if isinstance(node, tvm_tirx.ir.Bind) and isinstance(node.var.ty, tvm_tirx.tirx.TensorType):
            assert str(node.var.ty.scope) != "warp"
            if str(node.var.ty.scope) == "local":
                allocations.append(tuple(int(extent) for extent in node.var.ty.shape))
        if isinstance(node, tvm_tirx.ir.TensorLoad | tvm_tirx.ir.TensorStore):
            tensor = node.source if isinstance(node, tvm_tirx.ir.TensorLoad) else node.dest
            if str(tensor.ty.scope) == "local":
                accesses.append(node)
                # All load/fill/MFMA/store operations address this lane's fragment.
                assert int(node.indices[-2]) == 0

    tvm_ffi.structural_walk(lowered.body, check)
    values_per_lane = k_inner // 4
    assert sorted(allocations) == sorted(
        [(2, 1, 1, values_per_lane), (1, 2, 1, values_per_lane), (2, 2, 1, 4)]
    )
    assert len(accesses) >= 8


if __name__ == "__main__":
    tvm_tirx.testing.main()
