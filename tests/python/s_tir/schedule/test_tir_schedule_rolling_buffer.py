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
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


def check_rolling_buffer(
    sch: s_tir.Schedule, origin: tirx.Function, expected: tirx.Function, check_run=False
):
    scheduled = sch.mod["main"]
    assert_structural_equal_ignore_global_symbol(scheduled, expected)
    verify_trace_roundtrip(sch, origin)
    if check_run:
        in_tensor = origin.params[0]
        out_tensor = origin.params[1]
        in_shape = [int(_) for _ in in_tensor.shape]
        out_shape = [int(_) for _ in out_tensor.shape]
        x = tvm_tirx.runtime.tensor(np.random.uniform(0, 64, in_shape).astype(in_tensor.dtype))
        y0 = tvm_tirx.runtime.tensor(np.zeros(out_shape).astype(out_tensor.dtype))
        y1 = tvm_tirx.runtime.tensor(np.zeros(out_shape).astype(out_tensor.dtype))
        f_origin = tvm.compile(origin)
        f_scheduled = tvm.compile(scheduled)
        f_origin(x, y0)
        f_scheduled(x, y1)
        tvm_tirx.testing.assert_allclose(y0.numpy(), y1.numpy())


def _tile_nd(s, tile, block_name):
    outer_indices = []
    inner_indices = []
    block = s.get_sblock(block_name)
    loops = s.get_loops(block)
    for i, size in enumerate(tile):
        outer, inner = s.split(loops[i], [None, size])
        outer_indices.append(outer)
        inner_indices.append(inner)

    s.reorder(*outer_indices, *inner_indices)
    return outer_indices, inner_indices


def test_1d_rolling_buffer():
    @Ts.function
    def before(A: T.Tensor((4, 12), "int32"), C: T.Tensor((4, 8), "int32")):
        B = Ts.sblock_alloc_tensor((4, 10), "int32")
        for c in T.serial(4):
            for i in T.serial(0, 10):
                for k in T.serial(3):
                    with Ts.sblock("B"):
                        cc, vi, vk = Ts.axis.remap("SSR", [c, i, k])
                        with Ts.init():
                            B[cc, vi] = 0
                        B[cc, vi] = B[cc, vi] + A[cc, vi + vk]
            for i in T.serial(0, 8):
                for k in T.serial(3):
                    with Ts.sblock("C"):
                        cc, vi, vk = Ts.axis.remap("SSR", [c, i, k])
                        with Ts.init():
                            C[cc, vi] = 0
                        C[cc, vi] = C[cc, vi] + B[cc, vi + vk]

    @Ts.function
    def expected(A: T.Tensor((4, 12), "int32"), C: T.Tensor((4, 8), "int32")):
        B = Ts.sblock_alloc_tensor([4, 6], dtype="int32")
        for c, i_0 in T.grid(4, 2):
            for ax0, ax1 in T.grid(6, 3):
                with Ts.sblock("B"):
                    Ts.where(i_0 < 1 or 2 <= ax0)
                    cc = Ts.axis.spatial(4, c)
                    vi = Ts.axis.opaque(10, i_0 * 4 + ax0)
                    vk = Ts.axis.reduce(3, ax1)
                    Ts.reads(A[cc, vi + vk])
                    Ts.writes(B[cc, vi % 6])
                    with Ts.init():
                        B[cc, vi % 6] = 0
                    B[cc, vi % 6] = B[cc, vi % 6] + A[cc, vi + vk]
            for i_1, k in T.grid(4, 3):
                with Ts.sblock("C"):
                    cc = Ts.axis.spatial(4, c)
                    vi = Ts.axis.opaque(8, i_0 * 4 + i_1)
                    vk = Ts.axis.reduce(3, k)
                    Ts.reads(B[cc, (vi + vk) % 6])
                    Ts.writes(C[cc, vi])
                    with Ts.init():
                        C[cc, vi] = 0
                    C[cc, vi] = C[cc, vi] + B[cc, (vi + vk) % 6]

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    _, i, _ = sch.get_loops(sch.get_sblock("C"))
    io, _ = sch.split(i, [2, 4])
    sch.compute_at(sch.get_sblock("B"), io)
    sch.rolling_buffer(sch.get_sblock("B"), 0)
    check_rolling_buffer(sch, before, expected, check_run=True)


@Ts.function
def cascade_2_max_pool2d(A: T.Tensor((1, 12, 12, 16), "int8"), C: T.Tensor((1, 8, 8, 16), "int8")):
    B = Ts.sblock_alloc_tensor([1, 10, 10, 16], dtype="int8")
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 10, 10, 16, 3, 3):
        with Ts.sblock("B"):
            ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with Ts.init():
                B[ax0, ax1, ax2, ax3] = T.int8(-128)
            B[ax0, ax1, ax2, ax3] = T.max(B[ax0, ax1, ax2, ax3], A[ax0, ax1 + rv0, ax2 + rv1, ax3])
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 8, 8, 16, 3, 3):
        with Ts.sblock("C"):
            ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with Ts.init():
                C[ax0, ax1, ax2, ax3] = T.int8(-128)
            C[ax0, ax1, ax2, ax3] = T.max(C[ax0, ax1, ax2, ax3], B[ax0, ax1 + rv0, ax2 + rv1, ax3])


@Ts.function
def cascade_3_max_pool2d_with_stride(
    A: T.Tensor((1, 24, 24, 16), "int8"), C: T.Tensor((1, 8, 8, 16), "int8")
):
    B_0 = Ts.sblock_alloc_tensor([1, 22, 22, 16], dtype="int8")
    B_1 = Ts.sblock_alloc_tensor([1, 10, 10, 16], dtype="int8")
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 22, 22, 16, 3, 3):
        with Ts.sblock("B_0"):
            ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with Ts.init():
                B_0[ax0, ax1, ax2, ax3] = T.int8(-128)
            B_0[ax0, ax1, ax2, ax3] = T.max(
                B_0[ax0, ax1, ax2, ax3], A[ax0, ax1 + rv0, ax2 + rv1, ax3]
            )
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 10, 10, 16, 3, 3):
        with Ts.sblock("B_1"):
            ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with Ts.init():
                B_1[ax0, ax1, ax2, ax3] = T.int8(-128)
            B_1[ax0, ax1, ax2, ax3] = T.max(
                B_1[ax0, ax1, ax2, ax3], B_0[ax0, ax1 * 2 + rv0, ax2 * 2 + rv1, ax3]
            )
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 8, 8, 16, 3, 3):
        with Ts.sblock("C"):
            ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with Ts.init():
                C[ax0, ax1, ax2, ax3] = T.int8(-128)
            C[ax0, ax1, ax2, ax3] = T.max(
                C[ax0, ax1, ax2, ax3], B_1[ax0, ax1 + rv0, ax2 + rv1, ax3]
            )


def test_cascade_max_pool2d_w_tiled():
    @Ts.function
    def expected(A: T.Tensor((1, 12, 12, 16), "int8"), C: T.Tensor((1, 8, 8, 16), "int8")):
        B = Ts.sblock_alloc_tensor([1, 10, 6, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 1, 2, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(10, 6, 16, 3, 3):
                with Ts.sblock("B"):
                    Ts.where(i2_0 < 1 or 2 <= ax1)
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.spatial(10, ax0)
                    ax2_1 = Ts.axis.opaque(10, i2_0 * 4 + ax1)
                    ax3_1, rv0, rv1 = Ts.axis.remap("SRR", [ax2, ax3, ax4])
                    Ts.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    Ts.writes(B[ax0_1, ax1_1, ax2_1 % 6, ax3_1])
                    with Ts.init():
                        B[ax0_1, ax1_1, ax2_1 % 6, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1, ax2_1 % 6, ax3_1] = T.max(
                        B[ax0_1, ax1_1, ax2_1 % 6, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 8, 4, 16, 3, 3):
                with Ts.sblock("C"):
                    ax0 = Ts.axis.spatial(1, i0_0 + i0_1)
                    ax1 = Ts.axis.spatial(8, i1_0 * 8 + i1_1)
                    ax2 = Ts.axis.opaque(8, i2_0 * 4 + i2_1)
                    ax3 = Ts.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(B[ax0, ax1 + rv0, (ax2 + rv1) % 6, ax3])
                    Ts.writes(C[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, ax1 + rv0, (ax2 + rv1) % 6, ax3]
                    )

    sch = tvm.s_tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    oi, _ = _tile_nd(sch, [1, 8, 4, 16], "C")
    sch.compute_at(sch.get_sblock("B"), oi[-1])
    sch.rolling_buffer(sch.get_sblock("B"), 0)
    check_rolling_buffer(sch, cascade_2_max_pool2d, expected, check_run=True)


def test_cascade_max_pool2d_h_tiled():
    @Ts.function
    def expected(A: T.Tensor((1, 12, 12, 16), "int8"), C: T.Tensor((1, 8, 8, 16), "int8")):
        B = Ts.sblock_alloc_tensor([1, 6, 10, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 2, 1, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(6, 10, 16, 3, 3):
                with Ts.sblock("B"):
                    Ts.where(i1_0 < 1 or 2 <= ax0)
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.opaque(10, i1_0 * 4 + ax0)
                    ax2_1 = Ts.axis.spatial(10, ax1)
                    ax3_1, rv0, rv1 = Ts.axis.remap("SRR", [ax2, ax3, ax4])
                    Ts.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    Ts.writes(B[ax0_1, ax1_1 % 6, ax2_1, ax3_1])
                    with Ts.init():
                        B[ax0_1, ax1_1 % 6, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1 % 6, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1 % 6, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 4, 8, 16, 3, 3):
                with Ts.sblock("C"):
                    ax0 = Ts.axis.spatial(1, i0_0 + i0_1)
                    ax1 = Ts.axis.opaque(8, i1_0 * 4 + i1_1)
                    ax2 = Ts.axis.spatial(8, i2_0 * 8 + i2_1)
                    ax3 = Ts.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(B[ax0, (ax1 + rv0) % 6, ax2 + rv1, ax3])
                    Ts.writes(C[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, (ax1 + rv0) % 6, ax2 + rv1, ax3]
                    )

    sch = tvm.s_tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    io, _ = _tile_nd(sch, [1, 4, 8, 16], "C")
    sch.compute_at(sch.get_sblock("B"), io[-1])
    sch.rolling_buffer(sch.get_sblock("B"), 0)
    check_rolling_buffer(sch, cascade_2_max_pool2d, expected, check_run=True)


def test_cascade_max_pool2d_h_w_c_tiled():
    @Ts.function
    def expected(A: T.Tensor((1, 12, 12, 16), "int8"), C: T.Tensor((1, 8, 8, 16), "int8")):
        B = Ts.sblock_alloc_tensor([1, 6, 10, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 2, 2, 2):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(6, 6, 8, 3, 3):
                with Ts.sblock("B"):
                    Ts.where((i1_0 < 1 or 2 <= ax0) and (i2_0 < 1 or 2 <= ax1))
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.opaque(10, i1_0 * 4 + ax0)
                    ax2_1 = Ts.axis.spatial(10, i2_0 * 4 + ax1)
                    ax3_1 = Ts.axis.spatial(16, i3_0 * 8 + ax2)
                    rv0, rv1 = Ts.axis.remap("RR", [ax3, ax4])
                    Ts.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    Ts.writes(B[ax0_1, ax1_1 % 6, ax2_1, ax3_1])
                    with Ts.init():
                        B[ax0_1, ax1_1 % 6, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1 % 6, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1 % 6, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 4, 4, 8, 3, 3):
                with Ts.sblock("C"):
                    ax0 = Ts.axis.spatial(1, i0_0 + i0_1)
                    ax1 = Ts.axis.opaque(8, i1_0 * 4 + i1_1)
                    ax2 = Ts.axis.spatial(8, i2_0 * 4 + i2_1)
                    ax3 = Ts.axis.spatial(16, i3_0 * 8 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(B[ax0, (ax1 + rv0) % 6, ax2 + rv1, ax3])
                    Ts.writes(C[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, (ax1 + rv0) % 6, ax2 + rv1, ax3]
                    )

    sch = tvm.s_tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    io, _ = _tile_nd(sch, [1, 4, 4, 8], "C")
    sch.compute_at(sch.get_sblock("B"), io[-1])
    sch.rolling_buffer(sch.get_sblock("B"), 0)
    check_rolling_buffer(sch, cascade_2_max_pool2d, expected, check_run=True)


def test_cascade_max_pool2d_non_perfect_tiled():
    @Ts.function
    def expected(A: T.Tensor((1, 12, 12, 16), "int8"), C: T.Tensor((1, 8, 8, 16), "int8")) -> None:
        B = Ts.sblock_alloc_tensor([1, 8, 10, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 2, 2, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(8, 8, 16, 3, 3):
                with Ts.sblock("B"):
                    Ts.where(
                        i1_0 * 6 + ax0 < 10
                        and i2_0 * 6 + ax1 < 10
                        and (i1_0 < 1 or 2 <= ax0)
                        and (i2_0 < 1 or 2 <= ax1)
                    )
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.opaque(10, i1_0 * 6 + ax0)
                    ax2_1 = Ts.axis.spatial(10, i2_0 * 6 + ax1)
                    ax3_1, rv0, rv1 = Ts.axis.remap("SRR", [ax2, ax3, ax4])
                    Ts.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    Ts.writes(B[ax0_1, ax1_1 % 8, ax2_1, ax3_1])
                    with Ts.init():
                        B[ax0_1, ax1_1 % 8, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1 % 8, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1 % 8, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 6, 6, 16, 3, 3):
                with Ts.sblock("C"):
                    Ts.where(i1_0 * 6 + i1_1 < 8 and i2_0 * 6 + i2_1 < 8)
                    ax0 = Ts.axis.spatial(1, i0_0 + i0_1)
                    ax1 = Ts.axis.opaque(8, i1_0 * 6 + i1_1)
                    ax2 = Ts.axis.spatial(8, i2_0 * 6 + i2_1)
                    ax3 = Ts.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(B[ax0, (ax1 + rv0) % 8, ax2 + rv1, ax3])
                    Ts.writes(C[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, (ax1 + rv0) % 8, ax2 + rv1, ax3]
                    )

    sch = tvm.s_tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    io, _ = _tile_nd(sch, [1, 6, 6, 16], "C")
    sch.compute_at(sch.get_sblock("B"), io[-1])
    sch.rolling_buffer(sch.get_sblock("B"), 0)
    check_rolling_buffer(sch, cascade_2_max_pool2d, expected, check_run=True)


def test_cascade_3_max_pool2d_with_stride():
    @Ts.function
    def expected(A: T.Tensor((1, 24, 24, 16), "int8"), C: T.Tensor((1, 8, 8, 16), "int8")) -> None:
        B_0 = Ts.sblock_alloc_tensor([1, 13, 22, 16], dtype="int8")
        B_1 = Ts.sblock_alloc_tensor([1, 6, 10, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 2, 2, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(13, 13, 16, 3, 3):
                with Ts.sblock("B_0"):
                    Ts.where((i1_0 < 1 or 5 <= ax0) and (i2_0 < 1 or 5 <= ax1))
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.opaque(22, i1_0 * 8 + ax0)
                    ax2_1 = Ts.axis.spatial(22, i2_0 * 8 + ax1)
                    ax3_1, rv0, rv1 = Ts.axis.remap("SRR", [ax2, ax3, ax4])
                    Ts.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    Ts.writes(B_0[ax0_1, ax1_1 % 13, ax2_1, ax3_1])
                    with Ts.init():
                        B_0[ax0_1, ax1_1 % 13, ax2_1, ax3_1] = T.int8(-128)
                    B_0[ax0_1, ax1_1 % 13, ax2_1, ax3_1] = T.max(
                        B_0[ax0_1, ax1_1 % 13, ax2_1, ax3_1],
                        A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1],
                    )
            for ax0, ax1, ax2, ax3, ax4 in T.grid(6, 6, 16, 3, 3):
                with Ts.sblock("B_1"):
                    Ts.where((i1_0 < 1 or 2 <= ax0) and (i2_0 < 1 or 2 <= ax1))
                    ax0_2 = Ts.axis.spatial(1, 0)
                    ax1_2 = Ts.axis.opaque(10, i1_0 * 4 + ax0)
                    ax2_2 = Ts.axis.spatial(10, i2_0 * 4 + ax1)
                    ax3_2, rv0, rv1 = Ts.axis.remap("SRR", [ax2, ax3, ax4])
                    Ts.reads(B_0[ax0_2, (ax1_2 * 2 + rv0) % 13, ax2_2 * 2 + rv1, ax3_2])
                    Ts.writes(B_1[ax0_2, ax1_2 % 6, ax2_2, ax3_2])
                    with Ts.init():
                        B_1[ax0_2, ax1_2 % 6, ax2_2, ax3_2] = T.int8(-128)
                    B_1[ax0_2, ax1_2 % 6, ax2_2, ax3_2] = T.max(
                        B_1[ax0_2, ax1_2 % 6, ax2_2, ax3_2],
                        B_0[ax0_2, (ax1_2 * 2 + rv0) % 13, ax2_2 * 2 + rv1, ax3_2],
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 4, 4, 16, 3, 3):
                with Ts.sblock("C"):
                    ax0_3 = Ts.axis.spatial(1, i0_0 + i0_1)
                    ax1_3 = Ts.axis.opaque(8, i1_0 * 4 + i1_1)
                    ax2_3 = Ts.axis.spatial(8, i2_0 * 4 + i2_1)
                    ax3_3 = Ts.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(B_1[ax0_3, (ax1_3 + rv0) % 6, ax2_3 + rv1, ax3_3])
                    Ts.writes(C[ax0_3, ax1_3, ax2_3, ax3_3])
                    with Ts.init():
                        C[ax0_3, ax1_3, ax2_3, ax3_3] = T.int8(-128)
                    C[ax0_3, ax1_3, ax2_3, ax3_3] = T.max(
                        C[ax0_3, ax1_3, ax2_3, ax3_3],
                        B_1[ax0_3, (ax1_3 + rv0) % 6, ax2_3 + rv1, ax3_3],
                    )

    sch = tvm.s_tir.Schedule(cascade_3_max_pool2d_with_stride, debug_mask="all")
    io, _ = _tile_nd(sch, [1, 4, 4, 16], "C")
    sch.compute_at(sch.get_sblock("B_1"), io[-1])
    sch.compute_at(sch.get_sblock("B_0"), io[-1])
    sch.rolling_buffer(sch.get_sblock("B_0"), 0)
    sch.rolling_buffer(sch.get_sblock("B_1"), 0)
    check_rolling_buffer(sch, cascade_3_max_pool2d_with_stride, expected, check_run=True)


def test_upscale():
    @Ts.function
    def before(A: T.Tensor((1, 16, 16, 16), "int8"), C: T.Tensor((1, 24, 24, 16), "int8")) -> None:
        B = Ts.sblock_alloc_tensor([1, 14, 14, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 5, 5, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(5, 5, 16, 3, 3):
                with Ts.sblock("B"):
                    Ts.where(i1_0 * 5 // 2 + ax0 < 14 and i2_0 * 5 // 2 + ax1 < 14)
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.spatial(14, i1_0 * 5 // 2 + ax0)
                    ax2_1 = Ts.axis.spatial(14, i2_0 * 5 // 2 + ax1)
                    ax3_1 = Ts.axis.spatial(16, ax2)
                    rv0, rv1 = Ts.axis.remap("RR", [ax3, ax4])
                    Ts.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    Ts.writes(B[ax0_1, ax1_1, ax2_1, ax3_1])
                    with Ts.init():
                        B[ax0_1, ax1_1, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 5, 5, 16, 3, 3):
                with Ts.sblock("C"):
                    Ts.where(i1_0 * 5 + i1_1 < 24 and i2_0 * 5 + i2_1 < 24)
                    ax0 = Ts.axis.spatial(1, i0_0 + i0_1)
                    ax1 = Ts.axis.spatial(24, i1_0 * 5 + i1_1)
                    ax2 = Ts.axis.spatial(24, i2_0 * 5 + i2_1)
                    ax3 = Ts.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(B[ax0, ax1 // 2 + rv0, ax2 // 2 + rv1, ax3])
                    Ts.writes(C[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, ax1 // 2 + rv0, ax2 // 2 + rv1, ax3]
                    )

    @Ts.function
    def expected(
        A: T.Tensor((1, 16, 16, 16), "int8"), C: T.Tensor((1, 24, 24, 16), "int8")
    ) -> None:
        B = Ts.sblock_alloc_tensor([1, 5, 14, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 5, 5, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(5, 5, 16, 3, 3):
                with Ts.sblock("B"):
                    Ts.where(
                        i1_0 * 5 // 2 + ax0 < 14
                        and i2_0 * 5 // 2 + ax1 < 14
                        and (i1_0 < 1 or 2 <= ax0)
                        and (i2_0 < 1 or 2 <= ax1)
                    )
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.opaque(14, i1_0 * 5 // 2 + ax0)
                    ax2_1 = Ts.axis.spatial(14, i2_0 * 5 // 2 + ax1)
                    ax3_1 = Ts.axis.spatial(16, ax2)
                    rv0, rv1 = Ts.axis.remap("RR", [ax3, ax4])
                    Ts.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    Ts.writes(B[ax0_1, ax1_1 % 5, ax2_1, ax3_1])
                    with Ts.init():
                        B[ax0_1, ax1_1 % 5, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1 % 5, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1 % 5, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 5, 5, 16, 3, 3):
                with Ts.sblock("C"):
                    Ts.where(i1_0 * 5 + i1_1 < 24 and i2_0 * 5 + i2_1 < 24)
                    ax0 = Ts.axis.spatial(1, i0_0 + i0_1)
                    ax1 = Ts.axis.opaque(24, i1_0 * 5 + i1_1)
                    ax2 = Ts.axis.spatial(24, i2_0 * 5 + i2_1)
                    ax3 = Ts.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(B[ax0, (ax1 // 2 + rv0) % 5, ax2 // 2 + rv1, ax3])
                    Ts.writes(C[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, (ax1 // 2 + rv0) % 5, ax2 // 2 + rv1, ax3]
                    )

    sch = tvm.s_tir.Schedule(before, debug_mask="all")
    sch.rolling_buffer(sch.get_sblock("B"), 0)
    check_rolling_buffer(sch, before, expected, check_run=True)


def test_fail_rolling_buffer_multi_writers():
    @Ts.function
    def func_multi_writers(
        A: T.Tensor((1, 12, 12, 16), "int8"), C: T.Tensor((1, 12, 12, 16), "int8")
    ):
        B = Ts.sblock_alloc_tensor([1, 12, 12, 16], dtype="int8")
        for i0, i1, i2, i3 in T.grid(1, 3, 3, 1):
            for ax0, ax1, ax2 in T.grid(6, 6, 16):
                with Ts.sblock("B_writer_0"):
                    ax0_1 = Ts.axis.spatial(1, i0)
                    ax1_1 = Ts.axis.spatial(12, i1 * 4 + ax0)
                    ax2_1 = Ts.axis.spatial(12, i2 * 4 + ax1)
                    ax3_1 = Ts.axis.spatial(16, ax2)
                    with Ts.init():
                        B[ax0_1, ax1_1, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1, ax2_1, ax3_1] = A[ax0_1, ax1_1, ax2_1, ax3_1] + T.int8(1)
            for ax0, ax1, ax2 in T.grid(6, 6, 16):
                with Ts.sblock("B_writer_1"):
                    ax0_2 = Ts.axis.spatial(1, i0)
                    ax1_2 = Ts.axis.spatial(12, i1 * 4 + ax0)
                    ax2_2 = Ts.axis.spatial(12, i2 * 4 + ax1)
                    ax3_2 = Ts.axis.spatial(16, ax2)
                    with Ts.init():
                        B[ax0_2, ax1_2, ax2_2, ax3_2] = T.int8(-128)
                    B[ax0_2, ax1_2, ax2_2, ax3_2] = B[ax0_2, ax1_2, ax2_2, ax3_2] + A[
                        ax0_2, ax1_2, ax2_2, ax3_2
                    ] * T.int8(2)
            for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 4, 4, 16, 3, 3):
                with Ts.sblock("C"):
                    ax0_3 = Ts.axis.spatial(1, i0 + ax0)
                    ax1_3 = Ts.axis.spatial(12, i1 * 4 + ax1)
                    ax2_3 = Ts.axis.spatial(12, i2 * 4 + ax2)
                    ax3_3 = Ts.axis.spatial(16, i3 * 16 + ax3)
                    rv0, rv1 = Ts.axis.remap("RR", [ax4, ax5])
                    with Ts.init():
                        C[ax0_3, ax1_3, ax2_3, ax3_3] = T.int8(-128)
                    C[ax0_3, ax1_3, ax2_3, ax3_3] = T.max(
                        C[ax0_3, ax1_3, ax2_3, ax3_3], B[ax0_3, ax1_3 + rv0, ax2_3 + rv1, ax3_3]
                    )

    sch = tvm.s_tir.Schedule(func_multi_writers, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.rolling_buffer(sch.get_sblock("B_writer_0"), 0)


def test_fail_rolling_buffer_not_match():
    @Ts.function
    def func_non_overlap(
        A: T.Tensor((1, 12, 12, 16), "int8"), C: T.Tensor((1, 12, 12, 16), "int8")
    ):
        B = Ts.sblock_alloc_tensor([1, 12, 12, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 3, 3, 1):
            for ax0, ax1, ax2 in T.grid(4, 4, 16):
                with Ts.sblock("B"):
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.spatial(12, i1_0 * 4 + ax0)
                    ax2_1 = Ts.axis.spatial(12, i2_0 * 4 + ax1)
                    ax3 = Ts.axis.spatial(16, ax2)
                    Ts.reads(A[ax0_1, ax1_1, ax2_1, ax3])
                    Ts.writes(B[ax0_1, ax1_1, ax2_1, ax3])
                    with Ts.init():
                        B[ax0_1, ax1_1, ax2_1, ax3] = T.int8(-128)
                    B[ax0_1, ax1_1, ax2_1, ax3] = A[ax0_1, ax1_1, ax2_1, ax3]
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 4, 4, 16, 1, 1):
                with Ts.sblock("C"):
                    ax0 = Ts.axis.spatial(1, i0_0 + i0_1)
                    ax1 = Ts.axis.spatial(12, i1_0 * 4 + i1_1)
                    ax2 = Ts.axis.spatial(12, i2_0 * 4 + i2_1)
                    ax3 = Ts.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(B[ax0, ax1 + rv0, ax2 + rv1, ax3])
                    Ts.writes(C[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, ax1 + rv0, ax2 + rv1, ax3]
                    )

    sch = tvm.s_tir.Schedule(func_non_overlap, debug_mask="all")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.rolling_buffer(sch.get_sblock("B"), 0)


def test_fail_rolling_buffer_injection_invalid():
    sch = tvm.s_tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    # Block B is not compute_at to Block C, so rolling_buffer injection is invalid.
    _, _ = _tile_nd(sch, [1, 4, 8, 16], "C")
    _, _ = _tile_nd(sch, [1, 4, 8, 16], "B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.rolling_buffer(sch.get_sblock("B"), 0)


if __name__ == "__main__":
    tvm_tirx.testing.main()
