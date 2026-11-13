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
from tvm.s_tir.schedule.schedule import ScheduleError
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


@Ts.function
def transpose_elementwise(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")
) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vj, vi] * 2.0


@Ts.function
def transpose_elementwise_reindex_read(
    A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")
) -> None:
    A_reindex = Ts.sblock_alloc_tensor((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("A_reindex"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            A_reindex[vi, vj] = A[vj, vi]
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A_reindex[vi, vj] * 2.0


@Ts.function
def conv2d_nhwc(
    Input: T.Tensor((1, 224, 224, 3), "float32"),
    Weight: T.Tensor((7, 7, 3, 64), "float32"),
    Conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32"),
) -> None:
    PadInput = Ts.sblock_alloc_tensor([1, 230, 230, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with Ts.sblock("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                ((((i1_1 >= 3) and (i1_1 < 227)) and (i2_1 >= 3)) and (i2_1 < 227)),
                Input[i0_1, (i1_1 - 3), (i2_1 - 3), i3_1],
                T.float32(0),
            )
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with Ts.sblock("conv2d_nhwc"):
            n, h, w, co, rh, rw, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            with Ts.init():
                Conv2d_nhwc[n, h, w, co] = T.float32(0)
            Conv2d_nhwc[n, h, w, co] = Conv2d_nhwc[n, h, w, co] + (
                PadInput[n, ((h * 2) + rh), ((w * 2) + rw), ((T.floordiv(co, 64) * 3) + rc)]
                * Weight[rh, rw, rc, co]
            )


@Ts.function
def conv2d_nhwc_reindex_data(
    Input: T.Tensor((1, 224, 224, 3), "float32"),
    Weight: T.Tensor((7, 7, 3, 64), "float32"),
    Conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32"),
) -> None:
    PadInput = Ts.sblock_alloc_tensor([1, 230, 230, 3], dtype="float32")
    ReindexInput = Ts.sblock_alloc_tensor([1, 112, 112, 7, 7, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with Ts.sblock("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                ((((i1_1 >= 3) and (i1_1 < 227)) and (i2_1 >= 3)) and (i2_1 < 227)),
                Input[i0_1, (i1_1 - 3), (i2_1 - 3), i3_1],
                T.float32(0),
            )
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 112, 112, 7, 7, 3):
        with Ts.sblock("ReindexInput"):
            n, h, w, rh, rw, rc = Ts.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
            ReindexInput[n, h, w, rh, rw, rc] = PadInput[n, ((h * 2) + rh), ((w * 2) + rw), rc]
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with Ts.sblock("conv2d_nhwc"):
            n, h, w, co, rh, rw, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            with Ts.init():
                Conv2d_nhwc[n, h, w, co] = T.float32(0)
            Conv2d_nhwc[n, h, w, co] = Conv2d_nhwc[n, h, w, co] + (
                ReindexInput[n, h, w, rh, rw, rc] * Weight[rh, rw, rc, co]
            )


@Ts.function
def conv2d_nhwc_reindex_weight(
    inputs: T.Tensor([1, 224, 224, 3], dtype="float32"),
    weight: T.Tensor([7, 7, 3, 64], dtype="float32"),
    conv2d_nhwc: T.Tensor([1, 112, 112, 64], dtype="float32"),
) -> None:
    PadInput = Ts.sblock_alloc_tensor([1, 230, 230, 3], dtype="float32")
    weight_reindex = Ts.sblock_alloc_tensor([64, 7, 7, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with Ts.sblock("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            Ts.reads(inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1])
            Ts.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                i1_1 >= 3 and i1_1 < 227 and i2_1 >= 3 and i2_1 < 227,
                inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1],
                T.float32(0),
            )
    for ax3, ax4, ax5, ax6 in T.grid(64, 7, 7, 3):
        with Ts.sblock("weight_reindex"):
            v3, v4, v5, v6 = Ts.axis.remap("SSSS", [ax3, ax4, ax5, ax6])
            Ts.reads(weight[v4, v5, v6, v3])
            Ts.writes(weight_reindex[v3, v4, v5, v6])
            weight_reindex[v3, v4, v5, v6] = weight[v4, v5, v6, v3]
    for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 112, 112, 64, 7, 7, 3):
        with Ts.sblock("conv2d_nhwc"):
            n, h, w, co, rh, rw, rc = Ts.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
            Ts.reads(
                PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc],
                weight_reindex[co, rh, rw, rc],
            )
            Ts.writes(conv2d_nhwc[n, h, w, co])
            with Ts.init():
                conv2d_nhwc[n, h, w, co] = T.float32(0)
            conv2d_nhwc[n, h, w, co] = (
                conv2d_nhwc[n, h, w, co]
                + PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc]
                * weight_reindex[co, rh, rw, rc]
            )


@Ts.function
def matmul(
    A: T.Tensor((512, 512), "float32"),
    B: T.Tensor((512, 512), "float32"),
    C: T.Tensor((512, 512), "float32"),
) -> None:
    for i0, i1, i2 in T.grid(512, 512, 512):
        with Ts.sblock("matmul"):
            i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
            Ts.reads(C[i, j], A[i, k], B[k, j])
            Ts.writes(C[i, j])
            with Ts.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


@Ts.function
def matmul_reindex_write(
    A: T.Tensor((512, 512), "float32"),
    B: T.Tensor((512, 512), "float32"),
    C: T.Tensor((512, 512), "float32"),
) -> None:
    C_reindex = Ts.sblock_alloc_tensor([512, 512], dtype="float32")
    for i0, i1, i2 in T.grid(512, 512, 512):
        with Ts.sblock("matmul"):
            i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
            Ts.reads(C_reindex[i, j], A[i, k], B[k, j])
            Ts.writes(C_reindex[i, j])
            with Ts.init():
                C_reindex[i, j] = T.float32(0)
            C_reindex[i, j] = C_reindex[i, j] + A[i, k] * B[k, j]
    for i0, i1 in T.grid(512, 512):
        with Ts.sblock("C_reindex"):
            v0, v1 = Ts.axis.remap("SS", [i0, i1])
            Ts.reads(C_reindex[v0, v1])
            Ts.writes(C[v0, v1])
            C[v0, v1] = C_reindex[v0, v1]


@Ts.function
def multiple_read(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vj, vi] + A[vi, vj]


@Ts.function
def mixed_dtype(
    p0: T.Tensor((T.int64(2), 1280), "float16"),
    p1: T.Tensor((1280, 1280), "float16"),
    T_matmul_NT: T.Tensor((T.int64(2), 1280), "float16"),
) -> None:
    for i0, i1, i2 in T.grid(T.int64(2), 1280, 1280):
        with Ts.sblock("T_matmul_NT"):
            i = Ts.axis.spatial(T.int64(2), i0)
            j, k = Ts.axis.remap("SR", [i1, i2])
            Ts.reads(p0[i, k], p1[j, k])
            Ts.writes(T_matmul_NT[i, j])
            with Ts.init():
                T_matmul_NT[i, j] = T.float16(0)
            T_matmul_NT[i, j] = T_matmul_NT[i, j] + p0[i, k] * p1[j, k]


@Ts.function
def mixed_dtype_reindex_write(
    p0: T.Tensor((T.int64(2), 1280), "float16"),
    p1: T.Tensor((1280, 1280), "float16"),
    T_matmul_NT: T.Tensor((T.int64(2), 1280), "float16"),
) -> None:
    T_matmul_NT_reindex = Ts.sblock_alloc_tensor([T.int64(2), 1280], dtype="float16")
    for i0, i1, i2 in T.grid(T.int64(2), 1280, 1280):
        with Ts.sblock("T_matmul_NT"):
            i = Ts.axis.spatial(T.int64(2), i0)
            j, k = Ts.axis.remap("SR", [i1, i2])
            Ts.reads(p0[i, k], p1[j, k])
            Ts.writes(T_matmul_NT_reindex[i, j])
            with Ts.init():
                T_matmul_NT_reindex[i, j] = T.float16(0)
            T_matmul_NT_reindex[i, j] = T_matmul_NT_reindex[i, j] + p0[i, k] * p1[j, k]
    for ax0, ax1 in T.grid(T.int64(2), 1280):
        with Ts.sblock("T_matmul_NT_reindex"):
            v0 = Ts.axis.spatial(T.int64(2), ax0)
            v1 = Ts.axis.remap("S", [ax1])
            Ts.reads(T_matmul_NT_reindex[v0, v1])
            Ts.writes(T_matmul_NT[v0, v1])
            T_matmul_NT[v0, v1] = T_matmul_NT_reindex[v0, v1]


@Ts.function
def matmul_unit_dim(
    A: T.Tensor((1, 512), "float32"),
    B: T.Tensor((512, 1), "float32"),
    C: T.Tensor((1, 1), "float32"),
) -> None:
    for i0, i1, i2 in T.grid(1, 1, 512):
        with Ts.sblock("matmul"):
            i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
            Ts.reads(C[i, j], A[i, k], B[k, j])
            Ts.writes(C[i, j])
            with Ts.init():
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]


@Ts.function
def matmul_unit_dim_reindex_write(
    A: T.Tensor((1, 512), "float32"),
    B: T.Tensor((512, 1), "float32"),
    C: T.Tensor((1, 1), "float32"),
) -> None:
    C_reindex = Ts.sblock_alloc_tensor([1, 1], dtype="float32")
    for i0, i1, i2 in T.grid(1, 1, 512):
        with Ts.sblock("matmul"):
            i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
            Ts.reads(C_reindex[i, j], A[i, k], B[k, j])
            Ts.writes(C_reindex[i, j])
            with Ts.init():
                C_reindex[i, j] = T.float32(0)
            C_reindex[i, j] = C_reindex[i, j] + A[i, k] * B[k, j]
    for i0, i1 in T.grid(1, 1):
        with Ts.sblock("C_reindex"):
            v0, v1 = Ts.axis.remap("SS", [i0, i1])
            Ts.reads(C_reindex[v0, v1])
            Ts.writes(C[v0, v1])
            C[v0, v1] = C_reindex[v0, v1]


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


@pytest.fixture(
    params=[pytest.param(False, id="tensor_index"), pytest.param(True, id="tensor_name")],
    scope="session",
)
def use_tensor_name(request):
    return request.param


def test_reindex_read_basic(use_block_name, use_tensor_name):
    sch = tvm.s_tir.Schedule(transpose_elementwise)
    block = "B" if use_block_name else sch.get_sblock("B")
    buf = "A" if use_tensor_name else ("read", 0)
    sch.reindex(block, buf)
    assert_structural_equal_ignore_global_symbol(
        transpose_elementwise_reindex_read, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=transpose_elementwise)


def test_conv2d_reindex_weight(use_block_name, use_tensor_name):
    sch = tvm.s_tir.Schedule(conv2d_nhwc)
    block = "conv2d_nhwc" if use_block_name else sch.get_sblock("conv2d_nhwc")
    buf = "Weight" if use_tensor_name else ("read", 1)
    sch.reindex(block, buf)
    assert_structural_equal_ignore_global_symbol(conv2d_nhwc_reindex_weight, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=conv2d_nhwc)


def test_conv2d_reindex_data(use_block_name, use_tensor_name):
    sch = tvm.s_tir.Schedule(conv2d_nhwc)
    block = "conv2d_nhwc" if use_block_name else sch.get_sblock("conv2d_nhwc")
    tensor_name = sch.get(sch.get_sblock("conv2d_nhwc")).reads[0].source.name
    buf = tensor_name if use_tensor_name else ("read", 0)
    sch.reindex(block, buf)
    assert_structural_equal_ignore_global_symbol(conv2d_nhwc_reindex_data, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=conv2d_nhwc)


def test_matmul_reindex_write(use_block_name, use_tensor_name):
    sch = tvm.s_tir.Schedule(matmul)
    block = "matmul" if use_block_name else sch.get_sblock("matmul")
    buf = "C" if use_tensor_name else ("write", 0)
    sch.reindex(block, buf)
    assert_structural_equal_ignore_global_symbol(matmul_reindex_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=matmul)


def test_reindex_fail_multiple_read(use_block_name, use_tensor_name):
    sch = tvm.s_tir.Schedule(multiple_read)
    block = "B" if use_block_name else sch.get_sblock("B")
    buf = "A" if use_tensor_name else ("read", 0)
    with pytest.raises(ScheduleError):
        sch.reindex(block, buf)


def test_reindex_mixed_dtype(use_block_name, use_tensor_name):
    sch = tvm.s_tir.Schedule(mixed_dtype)
    block = "T_matmul_NT" if use_block_name else sch.get_sblock("T_matmul_NT")
    buf = "T_matmul_NT" if use_tensor_name else ("write", 0)
    sch.reindex(block, buf)
    assert_structural_equal_ignore_global_symbol(mixed_dtype_reindex_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=mixed_dtype)


def test_matmul_unit_dim_reindex_write(use_block_name, use_tensor_name):
    sch = tvm.s_tir.Schedule(matmul_unit_dim)
    block = "matmul" if use_block_name else sch.get_sblock("matmul")
    buf = "C" if use_tensor_name else ("write", 0)
    sch.reindex(block, buf)
    assert_structural_equal_ignore_global_symbol(matmul_unit_dim_reindex_write, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=matmul_unit_dim)


if __name__ == "__main__":
    tvm_tirx.testing.main()
