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

import ml_dtypes
import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm.contrib.pickle_memoize import memoize
from tvm.testing import env as tvm_env
from tvm_tirx.testing import env


def get_random_tensor(shape, dtype):
    if dtype == "int8":
        return np.random.randint(-128, 128, shape).astype(dtype)
    elif dtype == "uint8":
        return np.random.randint(0, 256, shape).astype(dtype)
    return np.random.uniform(-1, 1, shape).astype(dtype)


def verify_group_gemm(
    func_name, M, N, K, num_groups, x_dtype, weight_dtype, out_dtype, use_scale, rtol, atol
):
    group_gemm_func = tvm_ffi.get_global_func(func_name, allow_missing=True)
    if group_gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    @memoize("tvm.contrib.cutlass.test_group_gemm_sm90")
    def get_ref_data():
        assert M % num_groups == 0
        M_per_group = M // num_groups
        a_np = get_random_tensor((M, K), x_dtype)
        b_np = get_random_tensor((num_groups, N, K), weight_dtype)
        indptr_np = np.arange(1, num_groups + 1).astype("int64") * M_per_group
        c_np = np.concatenate(
            [a_np[i * M_per_group : (i + 1) * M_per_group] @ b_np[i].T for i in range(num_groups)],
            axis=0,
        )
        return a_np, b_np, indptr_np, c_np

    def to_numpy_dtype(dtype):
        mapping = {"float8_e5m2": ml_dtypes.float8_e5m2, "float8_e4m3fn": ml_dtypes.float8_e4m3fn}
        return mapping.get(dtype, dtype)

    a_np, b_np, indptr_np, c_np = get_ref_data()

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        a_nd = tvm_tirx.runtime.tensor(a_np.astype(to_numpy_dtype(x_dtype)), device=dev)
        b_nd = tvm_tirx.runtime.tensor(b_np.astype(to_numpy_dtype(weight_dtype)), device=dev)
        c_nd = tvm_tirx.runtime.empty(c_np.shape, dtype=out_dtype, device=dev)
        indptr_nd = tvm_tirx.runtime.tensor(indptr_np, device=dev)
        workspace = tvm_tirx.runtime.empty((4096 * 1024,), dtype="uint8", device=dev)
        if use_scale:
            scale = tvm_tirx.runtime.tensor(np.array([1.0], dtype="float32"), device=dev)
            group_gemm_func(a_nd, b_nd, indptr_nd, workspace, scale, c_nd)
        else:
            group_gemm_func(a_nd, b_nd, indptr_nd, workspace, c_nd)
        tvm_tirx.testing.assert_allclose(c_nd.numpy(), c_np, rtol=rtol, atol=atol)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.skipif(not tvm_env.build_flag_enabled("USE_CUTLASS"), reason="need cutlass")
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_group_gemm_sm90():
    verify_group_gemm(
        "cutlass.group_gemm",
        8,
        128,
        128,
        4,
        "float16",
        "float16",
        "float16",
        False,
        rtol=1e-3,
        atol=1e-3,
    )
    verify_group_gemm(
        "cutlass.group_gemm_e5m2_e5m2_fp16",
        8,
        16,
        16,
        4,
        "float8_e5m2",
        "float8_e5m2",
        "float16",
        True,
        rtol=1e-1,
        atol=1,
    )
    verify_group_gemm(
        "cutlass.group_gemm_e4m3_e4m3_fp16",
        8,
        16,
        16,
        4,
        "float8_e4m3fn",
        "float8_e4m3fn",
        "float16",
        True,
        rtol=1e-1,
        atol=1,
    )


@pytest.mark.skipif(not tvm_env.build_flag_enabled("USE_CUTLASS"), reason="need cutlass")
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_group_gemm_sm100():
    verify_group_gemm(
        "cutlass.group_gemm",
        8,
        128,
        128,
        4,
        "bfloat16",
        "bfloat16",
        "bfloat16",
        False,
        rtol=1e-2,
        atol=1e-3,
    )


def rowwise_quant_fp8_e4m3(shape: tuple[int, int], block_size: tuple[int, int], dtype: str):
    x_full_np = (np.random.rand(*shape) * 2 - 1).astype(dtype)
    x_scale_shape = (
        *shape[:-1],
        (shape[-1] + block_size[1] - 1) // block_size[1],
    )
    # For each (block_size[1]) block, compute the max abs value of `w_full_np`
    x_max_abs_np = np.zeros(x_scale_shape, dtype="float32")
    for i in range(x_scale_shape[-1]):
        x_max_abs_np[..., i] = np.max(
            np.abs(x_full_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])]),
            axis=-1,
        )[0]
    # Scale is the `x_max_abs_np` divided by the max value of quant_dtype in ml_dtypes
    fp8_max = float(ml_dtypes.finfo("float8_e4m3fn").max)
    x_scale_np = x_max_abs_np / fp8_max
    # `x_np` is the `x_full_np` divided by the `x_scale_np` (with block awareness),
    # clamped to (-fp8_max, fp8_max), and cast to `quant_dtype`
    x_np = np.zeros_like(x_full_np, dtype="float8_e4m3fn")
    for i in range(x_scale_shape[-1]):
        x_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])] = np.clip(
            x_full_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])]
            / x_scale_np[..., i : i + 1],
            -fp8_max,
            fp8_max,
        )

    x_scale_np = np.random.rand(*x_scale_np.shape).astype("float32") / fp8_max
    for i in range(x_scale_shape[-1]):
        x_full_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])] = (
            x_np[..., i * block_size[1] : min((i + 1) * block_size[1], shape[-1])].astype(
                x_scale_np.dtype
            )
            * x_scale_np[..., i : i + 1]
        )
    return x_np, x_scale_np


def blockwise_quant_fp8_e4m3(shape: tuple[int, int], block_size: tuple[int, int], dtype: str):
    w_full_np = (np.random.rand(*shape) * 2 - 1).astype(dtype)
    w_scale_shape = (
        *shape[:-2],
        (shape[-2] + block_size[0] - 1) // block_size[0],
        (shape[-1] + block_size[1] - 1) // block_size[1],
    )
    # For each (block_size[0], block_size[1]) block, compute the max abs value of `w_full_np`
    w_max_abs_np = np.zeros(w_scale_shape, dtype="float32")
    for i in range(w_scale_shape[-2]):
        for j in range(w_scale_shape[-1]):
            block_shape = (
                *shape[:-2],
                min(block_size[0], shape[-2] - i * block_size[0]),
                min(block_size[1], shape[-1] - j * block_size[1]),
            )
            w_max_abs_np[..., i, j] = np.max(
                np.abs(
                    w_full_np[
                        ...,
                        i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                    ]
                ).reshape(*shape[:-2], block_shape[-2] * block_shape[-1]),
                axis=-1,
            )
    # Scale is the `w_max_abs_np` divided by the max value of quant_dtype in ml_dtypes
    fp8_max = float(ml_dtypes.finfo("float8_e4m3fn").max)
    w_scale_np = w_max_abs_np / fp8_max
    # `w_np` is the `w_full_np` divided by the `w_scale_np` (with block awareness),
    # clamped to (-fp8_max, fp8_max), and cast to `quant_dtype`
    w_np = np.zeros_like(w_full_np, dtype="float8_e4m3fn")
    if len(w_scale_shape) == 2:
        for i in range(w_scale_shape[-2]):
            for j in range(w_scale_shape[-1]):
                w_np[
                    i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                    j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                ] = np.clip(
                    w_full_np[
                        i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                    ]
                    / w_scale_np[..., i, j],
                    -fp8_max,
                    fp8_max,
                )
    else:
        for e in range(w_scale_shape[0]):
            for i in range(w_scale_shape[-2]):
                for j in range(w_scale_shape[-1]):
                    w_np[
                        e,
                        i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                        j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                    ] = np.clip(
                        w_full_np[
                            e,
                            i * block_size[0] : min((i + 1) * block_size[0], shape[-2]),
                            j * block_size[1] : min((j + 1) * block_size[1], shape[-1]),
                        ]
                        / w_scale_np[e, i, j],
                        -fp8_max,
                        fp8_max,
                    )

    w_scale_np = np.random.rand(*w_scale_np.shape).astype("float32") / fp8_max
    return w_np, w_scale_np


def blockwise_matmul(
    x_fp8_np: np.ndarray,
    x_scale_np: np.ndarray,
    w_np: np.ndarray,
    w_scale_np: np.ndarray,
    block_size: tuple[int, int],
    dtype: str,
):
    o_np = np.zeros((x_fp8_np.shape[0], w_np.shape[0]), dtype=dtype)
    for j in range(w_scale_np.shape[0]):
        for k in range(w_scale_np.shape[1]):
            o_np[:, j * block_size[0] : min((j + 1) * block_size[0], w_np.shape[0])] += (
                np.matmul(
                    x_fp8_np[
                        :, k * block_size[1] : min((k + 1) * block_size[1], x_fp8_np.shape[1])
                    ].astype(dtype),
                    w_np[
                        j * block_size[0] : min((j + 1) * block_size[0], w_np.shape[0]),
                        k * block_size[1] : min((k + 1) * block_size[1], w_np.shape[1]),
                    ].T.astype(dtype),
                )
                * x_scale_np[:, k : k + 1]
                * w_scale_np[j, k]
            )
    return o_np


def blockwise_bmm(
    x_fp8_np: np.ndarray,
    x_scale_np: np.ndarray,
    w_np: np.ndarray,
    w_scale_np: np.ndarray,
    block_size: tuple[int, int],
    dtype: str,
):
    o_np = np.zeros((x_fp8_np.shape[0], x_fp8_np.shape[1], w_np.shape[1]), dtype=dtype)
    for j in range(w_scale_np.shape[1]):
        for k in range(w_scale_np.shape[2]):
            o_np[..., j * block_size[0] : min((j + 1) * block_size[0], w_np.shape[1])] += (
                np.matmul(
                    x_fp8_np[
                        ..., k * block_size[1] : min((k + 1) * block_size[1], x_fp8_np.shape[2])
                    ].astype(dtype),
                    w_np[
                        ...,
                        j * block_size[0] : min((j + 1) * block_size[0], w_np.shape[1]),
                        k * block_size[1] : min((k + 1) * block_size[1], w_np.shape[2]),
                    ]
                    .transpose(0, 2, 1)
                    .astype(dtype),
                )
                * x_scale_np[..., k : k + 1]
                * w_scale_np[..., j : j + 1, k : k + 1]
            )
    return o_np


@pytest.mark.skipif(not tvm_env.build_flag_enabled("USE_CUTLASS"), reason="need cutlass")
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_fp8_e4m3_groupwise_scaled_gemm():
    M = 16
    N = 4608
    K = 896
    block_size = (128, 128)
    assert N % 128 == 0 and K % 128 == 0  # Only support N/K are multiple of 128

    func_name = "cutlass.groupwise_scaled_gemm_e4m3fn_e4m3fn"
    gemm_func = tvm_ffi.get_global_func(func_name, allow_missing=True)
    if gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    dtype = "bfloat16"
    x_np, x_scale_np = rowwise_quant_fp8_e4m3((M, K), block_size, dtype)
    w_np, w_scale_np = blockwise_quant_fp8_e4m3((N, K), block_size, dtype)
    o_np = blockwise_matmul(x_np, x_scale_np, w_np, w_scale_np, block_size, dtype)

    def run_and_check():
        device = tvm_tirx.cuda(0)
        x_tvm = tvm_tirx.runtime.tensor(x_np, device=device)
        x_scale_tvm = tvm_tirx.runtime.tensor(x_scale_np.T, device=device)
        w_tvm = tvm_tirx.runtime.tensor(w_np, device=device)
        w_scale_tvm = tvm_tirx.runtime.tensor(w_scale_np, device=device)
        workspace = tvm_tirx.runtime.empty((4096 * 1024,), dtype="uint8", device=device)
        o_tvm = tvm_tirx.runtime.empty((M, N), dtype=dtype, device=device)
        gemm_func(
            x_tvm,
            w_tvm,
            x_scale_tvm,
            w_scale_tvm,
            workspace,
            block_size[0],
            block_size[1],
            o_tvm,
        )
        tvm_tirx.testing.assert_allclose(o_tvm.numpy(), o_np, rtol=1e-4, atol=0.5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.skipif(not tvm_env.build_flag_enabled("USE_CUTLASS"), reason="need cutlass")
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_fp8_e4m3_groupwise_scaled_bmm():
    B = 16
    M = 40
    N = 512
    K = 128
    block_size = (128, 128)
    assert N % 128 == 0 and K % 128 == 0  # Only support N/K are multiple of 128

    func_name = "cutlass.groupwise_scaled_bmm_e4m3fn_e4m3fn"
    gemm_func = tvm_ffi.get_global_func(func_name, allow_missing=True)
    if gemm_func is None:
        print(f"Skipped as {func_name} is not available")
        return

    dtype = "bfloat16"
    x_np, x_scale_np = rowwise_quant_fp8_e4m3((B, M, K), block_size, dtype)
    w_np, w_scale_np = blockwise_quant_fp8_e4m3((B, N, K), block_size, dtype)
    o_np = blockwise_bmm(x_np, x_scale_np, w_np, w_scale_np, block_size, dtype)

    def run_and_check():
        device = tvm_tirx.cuda(0)
        x_tvm = tvm_tirx.runtime.tensor(x_np, device=device)
        x_scale_tvm = tvm_tirx.runtime.tensor(x_scale_np.transpose(0, 2, 1), device=device)
        w_tvm = tvm_tirx.runtime.tensor(w_np, device=device)
        w_scale_tvm = tvm_tirx.runtime.tensor(w_scale_np, device=device)
        workspace = tvm_tirx.runtime.empty((4096 * 1024,), dtype="uint8", device=device)
        o_tvm = tvm_tirx.runtime.empty((B, M, N), dtype=dtype, device=device)
        gemm_func(
            x_tvm,
            w_tvm,
            x_scale_tvm,
            w_scale_tvm,
            workspace,
            block_size[0],
            block_size[1],
            o_tvm,
        )
        tvm_tirx.testing.assert_allclose(o_tvm.numpy(), o_np, rtol=1e-4, atol=0.5)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
