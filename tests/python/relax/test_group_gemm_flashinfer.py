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
# ruff: noqa: E501, F401, RUF005

"""Test for FlashInfer GroupedGemm TVM integration"""

import math

import numpy as np
import pytest
import torch
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax

DEFAULT_WORKSPACE_SIZE = 32 * 1024 * 1024
fp8_dtype = "float8_e4m3fn"


###########################################
################# Helpers #################
###########################################
def has_flashinfer():
    """Check if FlashInfer is available with the SM100 grouped-gemm symbol."""
    try:
        from flashinfer.gemm import (  # pylint: disable=import-outside-toplevel,unused-import
            gen_gemm_sm100_module,
        )

        from tvm.relax.backend.cuda import (  # pylint: disable=import-outside-toplevel
            flashinfer,
        )

        return True
    except ImportError:
        return False


def has_cutlass():
    """Check if CUTLASS is available for SM90+ operations"""
    if not tvm_ffi.get_global_func("device_api.cuda", True):
        return False
    try:
        import pynvml  # pylint: disable=import-outside-toplevel

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        return major >= 9  # SM90+
    except Exception:
        return False


def calc_diff(x: np.ndarray, y: np.ndarray):
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def quantize_fp8(x, scale_shape, tile_shape, scale_major_mode):
    from einops import rearrange, reduce, repeat

    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim == len(scale_shape) == len(tile_shape)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_amax = torch.tensor(fp8_info.max, device=x.device, dtype=torch.float32)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        s0, s1 = scale_shape
        t0, t1 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(x, "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(x_scale, "s0 s1 -> (s0 t0) (s1 t1)", t0=t0, t1=t1)
        else:
            # Handle column-major tiling
            x_tiled = rearrange(x, "(s1 t0) (s0 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
            abs_max = reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 -> s1 s0")
            scales_repeated = repeat(scales_permuted, "s1 s0 -> (s1 t0) (s0 t1)", t0=t0, t1=t1)

    elif ndim == 3:
        s0, s1, s2 = scale_shape
        t0, t1, t2 = tile_shape
        if scale_major_mode == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = rearrange(
                x, "(s0 t0) (s1 t1) (s2 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = repeat(
                x_scale, "s0 s1 s2 -> (s0 t0) (s1 t1) (s2 t2)", t0=t0, t1=t1, t2=t2
            )
        else:
            # Handle layout where the last two axes are swapped
            x_tiled = rearrange(
                x, "(s0 t0) (s2 t1) (s1 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = reduce(x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max").clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))
            # Permute scale axes before repeating to match layout
            scales_permuted = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1")
            scales_repeated = repeat(
                scales_permuted,
                "s0 s2 s1 -> (s0 t0) (s2 t1) (s1 t2)",
                t0=t0,
                t1=t1,
                t2=t2,
            )
    # 3. Final Quantization
    # Divide the original tensor by the broadcasted scales
    x_fp32 = x / (scales_repeated + 1e-8)

    # Convert the result to the target FP8 format
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    return x_fp8, x_scale


def dequantize_fp8(x, x_scale, scale_major_mode):
    from einops import rearrange

    """
    Quantizes a 2D or 3D tensor to FP8.

    Args:
        x (torch.Tensor): The 2D or 3D input tensor.
        scale_shape (tuple): The shape of the scale tensor.
        tile_shape (tuple): The shape of the tiles.
        scale_major_mode (str): The tiling order, "K" for row-major like,
                                or another value for column-major like.

    Returns:
        tuple: A tuple containing the quantized FP8 tensor and the
               calculated float32 scales.
    """
    # 1. Assertions and Initial Setup
    ndim = x.ndim
    assert ndim == len(x_scale.shape)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        if scale_major_mode == "K":
            s0, s1 = x_scale.shape
        else:
            s1, s0 = x_scale.shape
        x = rearrange(x.to(torch.float32), "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1)
        if scale_major_mode == "K":
            x_scale = rearrange(x_scale, "s0 s1 -> s0 s1 1 1")
        else:
            x_scale = rearrange(x_scale, "s0 s1 -> s1 s0 1 1")
        out = rearrange(x * x_scale, "s0 s1 t0 t1 -> (s0 t0) (s1 t1)")
    elif ndim == 3:
        if scale_major_mode == "K":
            s0, s1, s2 = x_scale.shape
        else:
            s0, s2, s1 = x_scale.shape
        x = rearrange(
            x.to(torch.float32),
            "(s0 t0) (s1 t1) (s2 t2)-> s0 s1 s2 t0 t1 t2",
            s0=s0,
            s1=s1,
            s2=s2,
        )
        if scale_major_mode == "K":
            x_scale = rearrange(x_scale, "s0 s1 s2 -> s0 s1 s2 1 1 1")
        else:
            x_scale = rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1 1 1 1")
        out = rearrange(x * x_scale, "s0 s1 s2 t0 t1 t2 -> (s0 t0) (s1 t1) (s2 t2)")

    return out


###########################################
########### Refernce generation ###########
###########################################
def compute_reference_grouped_gemm(
    a_fp32: torch.Tensor,  # (total_m, k)
    b_fp32: torch.Tensor,  # (batch_size, n, k)
    m_indptr: torch.Tensor,
    dtype_out: str,  # (total_m, n)
):
    """Compute reference result using PyTorch operations"""
    """Compute reference result using original FP32 tensors"""

    total_m, k = a_fp32.shape
    batch_size, n, k2 = b_fp32.shape
    assert k == k2

    # Perform grouped GEMM computation directly on original FP32 data
    results = []

    for i in range(batch_size):
        start_m = m_indptr[i].item()
        end_m = m_indptr[i + 1].item()

        # Extract group's portion of A
        a_group = a_fp32[start_m:end_m, :]  # [m_sizes[i], k]
        b_group = b_fp32[i]

        # Multiply with shared B matrix
        result_group = torch.mm(a_group, b_group.T)  # [m_sizes[i], n]
        results.append(result_group)

    result_fp32 = torch.cat(results, dim=0)

    # Convert to output dtype
    if dtype_out == "bfloat16":
        result = result_fp32.to(torch.bfloat16)
    elif dtype_out == "float16":
        result = result_fp32.to(torch.float16)
    else:
        result = result_fp32

    return result


###########################################
########### Test data generation ##########
###########################################
def generate_test_data(
    m_sizes: list,
    batch_size: int,
    n: int,
    k: int,
    dtype_a: str,
    dtype_b: str,
    dtype_out: str,
    scale_granularity_m: int,
    scale_granularity_n: int,
    scale_granularity_k: int,
    scale_major_mode: str,
    device: tvm_tirx.runtime.Device,
):
    """Generate test data for grouped GEMM operations"""
    assert batch_size == len(m_sizes), (
        f"batch_size ({batch_size}) must equal len(m_sizes) ({len(m_sizes)})"
    )

    # print(f"Device object: {device}")
    torch_device = torch.device(f"cuda:{device.index}")

    cum_m = [0] + list(np.cumsum(m_sizes))
    total_m = cum_m[-1]

    # Generate input matrices A and B (where we assert of form fp8) random data in fp32 first, then convert
    assert dtype_a == "float8_e4m3fn"
    a_fp32 = torch.randn(total_m, k, device=torch_device, dtype=torch.float32)

    assert dtype_b == "float8_e4m3fn"
    b_fp32 = torch.randn(batch_size, n, k, device=torch_device, dtype=torch.float32) / math.sqrt(k)

    if scale_major_mode == "K":  # K mode:
        scale_a_shape = (total_m // scale_granularity_m, k // scale_granularity_k)
        scale_b_shape = (batch_size, n // scale_granularity_n, k // scale_granularity_k)

    else:  # MN mode
        scale_a_shape = (k // scale_granularity_k, total_m // scale_granularity_m)
        scale_b_shape = (batch_size, k // scale_granularity_k, n // scale_granularity_n)

    tile_a_shape = (scale_granularity_m, scale_granularity_k)
    tile_b_shape = (1, scale_granularity_n, scale_granularity_k)

    # quantize A, B
    a_quantized, scale_a = quantize_fp8(a_fp32, scale_a_shape, tile_a_shape, scale_major_mode)
    b_quantized, scale_b = quantize_fp8(b_fp32, scale_b_shape, tile_b_shape, scale_major_mode)

    if dtype_a == "float8_e4m3fn":
        a_tvm = tvm_tirx.runtime.tensor(
            a_quantized.view(torch.uint8).cpu().numpy().view(fp8_dtype), device=device
        )
    else:
        a_tvm = tvm_ffi.from_dlpack(a_quantized)

    if dtype_b == "float8_e4m3fn":
        b_tvm = tvm_tirx.runtime.tensor(
            b_quantized.view(torch.uint8).cpu().numpy().view(fp8_dtype), device=device
        )
    else:
        b_tvm = tvm_ffi.from_dlpack(b_quantized)

    scale_a_tvm = tvm_ffi.from_dlpack(scale_a)
    scale_b_tvm = tvm_ffi.from_dlpack(scale_b)

    # Create m_indptr for grouped operation
    m_indptr = torch.tensor(cum_m, device=torch_device, dtype=torch.int32)
    m_indptr_tvm = tvm_tirx.runtime.tensor(m_indptr.cpu().numpy(), device)

    return {
        "a": a_tvm,
        "b": b_tvm,
        "torch_a": a_fp32,
        "torch_b": b_fp32,
        "scale_a": scale_a_tvm,
        "scale_b": scale_b_tvm,
        "m_indptr": m_indptr_tvm,
        "m_sizes": m_sizes,
        "n": n,
        "k": k,
        "total_m": total_m,
        "torch_scale_a": scale_a,
        "torch_scale_b": scale_b,
        "torch_m_indptr": m_indptr,
    }


###########################################
############### Test driver ###############
###########################################
@pytest.mark.skipif(not has_flashinfer(), reason="FlashInfer not available")
@pytest.mark.skipif(not has_cutlass(), reason="CUTLASS SM90+ not available")
@pytest.mark.parametrize(
    "dtype_a,dtype_b,dtype_out",
    [
        ("float8_e4m3fn", "float8_e4m3fn", "bfloat16"),
        ("float8_e4m3fn", "float8_e4m3fn", "float16"),
    ],
)
@pytest.mark.parametrize(
    "scale_granularity_m,scale_granularity_n,scale_granularity_k",
    [
        (1, 128, 128),  # Row-wise A, block-wise B
    ],
)
@pytest.mark.parametrize("scale_major_mode", ["K", "MN"])
@pytest.mark.parametrize("mma_sm", [1, 2])
@pytest.mark.parametrize(
    "test_case",
    [
        {"batch_size": 4, "m_sizes": [128, 256, 192, 320], "n": 512, "k": 1024},
        {"batch_size": 2, "m_sizes": [64, 128], "n": 256, "k": 512},
        {"batch_size": 3, "m_sizes": [256, 256, 128], "n": 768, "k": 768},
        {"batch_size": 2, "m_sizes": [20, 36], "n": 768, "k": 768},
    ],
)
def test_grouped_gemm_correctness(
    dtype_a,
    dtype_b,
    dtype_out,
    scale_granularity_m,
    scale_granularity_n,
    scale_granularity_k,
    scale_major_mode,
    mma_sm,
    test_case,
):
    """Test correctness of GroupedGemm operations"""
    target = tvm_tirx.target.Target.from_device(tvm_tirx.cuda(0))

    # Generate the module
    mod = relax.backend.cuda.flashinfer.gen_grouped_gemm_module(target=target)[0]

    # Load the module
    grouped_gemm_fn = mod["group_gemm_fp8_nt_groupwise"]

    def run_and_check():
        device = tvm_tirx.cuda(0)
        test_data = generate_test_data(
            batch_size=test_case["batch_size"],
            m_sizes=test_case["m_sizes"],
            n=test_case["n"],
            k=test_case["k"],
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_out=dtype_out,
            scale_granularity_m=scale_granularity_m,
            scale_granularity_n=scale_granularity_n,
            scale_granularity_k=scale_granularity_k,
            scale_major_mode=scale_major_mode,
            device=device,
        )

        output_shape = (test_data["total_m"], test_data["n"])
        if dtype_out == "bfloat16":
            output = tvm_tirx.runtime.empty(output_shape, dtype="bfloat16", device=device)
        elif dtype_out == "float16":
            output = tvm_tirx.runtime.empty(output_shape, dtype="float16", device=device)
        else:
            output = tvm_tirx.runtime.empty(output_shape, dtype="float32", device=device)

        int_workspace = tvm_tirx.runtime.empty(
            (DEFAULT_WORKSPACE_SIZE,), dtype="int32", device=device
        )
        float_workspace = tvm_tirx.runtime.empty(
            (DEFAULT_WORKSPACE_SIZE,), dtype="float32", device=device
        )

        grouped_gemm_fn(
            int_workspace,
            float_workspace,
            test_data["a"],
            test_data["b"],
            test_data["scale_a"],
            test_data["scale_b"],
            output,
            test_data["m_indptr"],
            test_data["n"],
            test_data["k"],
            scale_granularity_m,
            scale_granularity_n,
            scale_granularity_k,
            scale_major_mode,
            mma_sm,
        )

        reference = compute_reference_grouped_gemm(
            test_data["torch_a"],
            test_data["torch_b"],
            test_data["torch_m_indptr"],
            dtype_out,
        )
        output_torch = torch.as_tensor(output, device=test_data["torch_a"].device)
        assert output_torch.shape == reference.shape, (
            f"Shape mismatch: got {output_torch.shape}, expected {reference.shape}"
        )
        diff = calc_diff(output_torch.cpu().double().numpy(), reference.cpu().double().numpy())
        assert diff < 1e-3, f"diff too large {diff}"

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
