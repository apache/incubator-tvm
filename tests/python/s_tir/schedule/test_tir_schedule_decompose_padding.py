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
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def check_decompose_padding(origin, scheduled, expected, check_run=False):
    assert_structural_equal_ignore_global_symbol(scheduled, expected)
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


def test_int64_indices_batch_decompose_padding():
    @Ts.function
    def before_decompose(
        x: T.Tensor((T.int64(1), T.int64(128), T.int64(128)), "int32"),
        y: T.Tensor((T.int64(1), T.int64(140), T.int64(128)), "int32"),
    ):
        for b, i, j in T.grid(T.int64(1), T.int64(140), T.int64(128)):
            with Ts.sblock("block"):
                vb, vi, vj = Ts.axis.remap("SSS", [b, i, j])
                y[vb, vi, vj] = T.if_then_else(vi < T.int64(128), x[vb, vi, vj], 0)

    @Ts.function
    def after_decompose(
        x: T.Tensor((T.int64(1), T.int64(128), T.int64(128)), "int32"),
        y: T.Tensor((T.int64(1), T.int64(140), T.int64(128)), "int32"),
    ):
        # with Ts.sblock("root"):
        for b, i in T.grid(T.int64(1), T.int64(140)):
            # Use T.serial(T.int64(0), T.int64(128)) so iter_var dom.min is int64
            # (matches schedule output; `range(T.int64(...))` would emit an int32 min).
            for j in T.serial(T.int64(0), T.int64(128)):
                with Ts.sblock("block_pad_const"):
                    vb = Ts.axis.spatial(T.int64(1), T.int64(0))
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Ts.reads()
                    Ts.writes(y[vb, vi, vj])
                    y[vb, vi, vj] = 0
            for j in T.serial(T.int64(0), T.int64(128)):
                with Ts.sblock("block"):
                    vb = Ts.axis.spatial(T.int64(1), T.int64(0))
                    vi = Ts.axis.spatial(T.int64(128), i)
                    vj = Ts.axis.spatial(T.int64(128), j)
                    Ts.where(i < T.int64(128))
                    Ts.reads(x[vb, vi, vj])
                    Ts.writes(y[vb, vi, vj])
                    y[vb, vi, vj] = x[vb, vi, vj]

    sch = tvm.s_tir.Schedule(before_decompose, debug_mask="all")
    block = sch.get_sblock("block")
    sch.decompose_padding(block, sch.get_loops(block)[2])
    check_decompose_padding(before_decompose, sch.mod["main"], after_decompose, check_run=False)


def test_1d_decompose_padding():
    @Ts.function
    def before_decompose(x: T.Tensor(128, "int32"), y: T.Tensor(140, "int32")):
        for i in range(140):
            with Ts.sblock("block"):
                vi = Ts.axis.remap("S", [i])
                y[vi] = T.if_then_else(vi >= 6 and vi < 134, x[vi - 6], 0)

    @Ts.function
    def after_decompose(x: T.Tensor(128, "int32"), y: T.Tensor(140, "int32")):
        for i in T.serial(140):
            with Ts.sblock("block_pad_const"):
                vi = Ts.axis.spatial(140, i)
                Ts.reads()
                Ts.writes(y[vi])
                y[vi] = 0
        for i in T.serial(128):
            with Ts.sblock("block"):
                vi = Ts.axis.spatial(128, i)
                Ts.reads(x[vi])
                Ts.writes(y[vi + 6])
                y[vi + 6] = x[vi]

    sch = tvm.s_tir.Schedule(before_decompose, debug_mask="all")
    block = sch.get_sblock("block")
    sch.decompose_padding(block, sch.get_loops(block)[0])
    check_decompose_padding(before_decompose, sch.mod["main"], after_decompose, check_run=False)


@Ts.function
def sum_pool_2d(
    x: T.Tensor((1, 16, 225, 225), "int8"), tensor: T.Tensor((1, 16, 225, 225), "int8")
):
    pad_temp = Ts.sblock_alloc_tensor([1, 16, 231, 231], dtype="int8")
    for i0, i1, i2, i3 in T.grid(1, 16, 231, 231):
        with Ts.sblock("pad_temp"):
            ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            pad_temp[ax0, ax1, ax2, ax3] = T.if_then_else(
                3 <= ax2 and ax2 < 228 and 3 <= ax3 and ax3 < 228,
                x[ax0, ax1, ax2 - 3, ax3 - 3],
                T.int8(0),
            )
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 16, 225, 225, 7, 7):
        with Ts.sblock("tensor"):
            ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with Ts.init():
                tensor[ax0, ax1, ax2, ax3] = T.int8(0)
            tensor[ax0, ax1, ax2, ax3] = (
                tensor[ax0, ax1, ax2, ax3] + pad_temp[ax0, ax1, ax2 + rv0, ax3 + rv1]
            )


def test_decompose_hw_padding_direct():
    """Case 0. direct decompose"""

    @Ts.function
    def pooling_decompose_0(
        x: T.Tensor((1, 16, 225, 225), "int8"), tensor: T.Tensor((1, 16, 225, 225), "int8")
    ):
        pad_temp = Ts.sblock_alloc_tensor([1, 16, 231, 231], dtype="int8")
        for i0, i1, i2, i3 in T.grid(1, 16, 231, 231):
            with Ts.sblock("pad_temp_pad_const"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                pad_temp[ax0, ax1, ax2, ax3] = T.int8(0)
        for i0, i1, i2, i3 in T.grid(1, 16, 225, 225):
            with Ts.sblock("pad_temp"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                pad_temp[ax0, ax1, ax2 + 3, ax3 + 3] = x[ax0, ax1, ax2, ax3]
        for i0, i1, i2, i3, i4, i5 in T.grid(1, 16, 225, 225, 7, 7):
            with Ts.sblock("tensor"):
                ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                with Ts.init():
                    tensor[ax0, ax1, ax2, ax3] = T.int8(0)
                tensor[ax0, ax1, ax2, ax3] = (
                    tensor[ax0, ax1, ax2, ax3] + pad_temp[ax0, ax1, ax2 + rv0, ax3 + rv1]
                )

    sch = tvm.s_tir.Schedule(sum_pool_2d, debug_mask="all")
    pad = sch.get_sblock("pad_temp")
    sch.decompose_padding(pad, sch.get_loops(pad)[0])
    check_decompose_padding(sum_pool_2d, sch.mod["main"], pooling_decompose_0, check_run=True)


def test_decompose_hw_padding_tiled():
    """Case 1. tiling and then decompose"""

    @Ts.function
    def pooling_decompose_1(
        x: T.Tensor((1, 16, 225, 225), "int8"), tensor: T.Tensor((1, 16, 225, 225), "int8")
    ) -> None:
        pad_temp = Ts.sblock_alloc_tensor([1, 16, 231, 231], dtype="int8")
        for i0, i2_0, i3_0 in T.grid(1, 3, 3):
            for ax0, ax1, ax2 in T.grid(16, 81, 81):
                with Ts.sblock("pad_temp_pad_const"):
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.spatial(16, ax0)
                    ax2_1 = Ts.axis.spatial(231, i2_0 * 75 + ax1)
                    ax3 = Ts.axis.spatial(231, i3_0 * 75 + ax2)
                    Ts.reads()
                    Ts.writes(pad_temp[ax0_1, ax1_1, ax2_1, ax3])
                    pad_temp[ax0_1, ax1_1, ax2_1, ax3] = T.int8(0)
            for ax0, ax1, ax2 in T.grid(16, 81, 81):
                with Ts.sblock("pad_temp"):
                    ax0_2 = Ts.axis.spatial(1, 0)
                    ax1_2 = Ts.axis.spatial(16, ax0)
                    ax2_2 = Ts.axis.spatial(225, i2_0 * 75 + ax1 - 3)
                    ax3 = Ts.axis.spatial(225, i3_0 * 75 + ax2 - 3)
                    Ts.where(
                        3 <= i2_0 * 75 + ax1
                        and i2_0 * 75 + ax1 < 228
                        and 3 <= i3_0 * 75 + ax2
                        and i3_0 * 75 + ax2 < 228
                    )
                    Ts.reads(x[ax0_2, ax1_2, ax2_2, ax3])
                    Ts.writes(pad_temp[ax0_2, ax1_2, ax2_2 + 3, ax3 + 3])
                    pad_temp[ax0_2, ax1_2, ax2_2 + 3, ax3 + 3] = x[ax0_2, ax1_2, ax2_2, ax3]
            for i1, i2_1, i3_1, i4, i5 in T.grid(16, 75, 75, 7, 7):
                with Ts.sblock("tensor"):
                    ax0_3, ax1_3 = Ts.axis.remap("SS", [i0, i1])
                    ax2_3 = Ts.axis.spatial(225, i2_0 * 75 + i2_1)
                    ax3 = Ts.axis.spatial(225, i3_0 * 75 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(pad_temp[ax0_3, ax1_3, ax2_3 + rv0, ax3 + rv1])
                    Ts.writes(tensor[ax0_3, ax1_3, ax2_3, ax3])
                    with Ts.init():
                        tensor[ax0_3, ax1_3, ax2_3, ax3] = T.int8(0)
                    tensor[ax0_3, ax1_3, ax2_3, ax3] = (
                        tensor[ax0_3, ax1_3, ax2_3, ax3]
                        + pad_temp[ax0_3, ax1_3, ax2_3 + rv0, ax3 + rv1]
                    )

    sch = tvm.s_tir.Schedule(sum_pool_2d, debug_mask="all")
    block = sch.get_sblock("tensor")
    pad = sch.get_sblock("pad_temp")
    n, c, h, w, kh, kw = sch.get_loops(block)
    ho, hi = sch.split(h, [3, 75])
    wo, wi = sch.split(w, [3, 75])
    sch.reorder(n, ho, wo, c, hi, wi, kh, kw)
    sch.compute_at(sch.get_sblock("pad_temp"), wo)
    sch.decompose_padding(pad, sch.get_loops(pad)[3])
    check_decompose_padding(sum_pool_2d, sch.mod["main"], pooling_decompose_1, check_run=True)


def test_decompose_hw_padding_tiled_and_lift_pad():
    """Case 2. tiling and then decompose, lift const pad values to outer loop"""

    @Ts.function
    def pooling_decompose_2(
        x: T.Tensor((1, 16, 225, 225), "int8"), tensor: T.Tensor((1, 16, 225, 225), "int8")
    ) -> None:
        pad_temp = Ts.sblock_alloc_tensor([1, 16, 231, 231], dtype="int8")
        for i0, i2_0, i3_0, ax0, ax1, ax2 in T.grid(1, 3, 3, 16, 81, 81):
            with Ts.sblock("pad_temp_pad_const"):
                ax0_1 = Ts.axis.spatial(1, 0)
                ax1_1 = Ts.axis.spatial(16, ax0)
                ax2_1 = Ts.axis.spatial(231, i2_0 * 75 + ax1)
                ax3 = Ts.axis.spatial(231, i3_0 * 75 + ax2)
                Ts.reads()
                Ts.writes(pad_temp[ax0_1, ax1_1, ax2_1, ax3])
                pad_temp[ax0_1, ax1_1, ax2_1, ax3] = T.int8(0)
        for i0, i2_0, i3_0 in T.grid(1, 3, 3):
            for ax0, ax1, ax2 in T.grid(16, 81, 81):
                with Ts.sblock("pad_temp"):
                    ax0_2 = Ts.axis.spatial(1, 0)
                    ax1_2 = Ts.axis.spatial(16, ax0)
                    ax2_2 = Ts.axis.spatial(225, i2_0 * 75 + ax1 - 3)
                    ax3 = Ts.axis.spatial(225, i3_0 * 75 + ax2 - 3)
                    Ts.where(
                        3 <= i2_0 * 75 + ax1
                        and i2_0 * 75 + ax1 < 228
                        and 3 <= i3_0 * 75 + ax2
                        and i3_0 * 75 + ax2 < 228
                    )
                    Ts.reads(x[ax0_2, ax1_2, ax2_2, ax3])
                    Ts.writes(pad_temp[ax0_2, ax1_2, ax2_2 + 3, ax3 + 3])
                    pad_temp[ax0_2, ax1_2, ax2_2 + 3, ax3 + 3] = x[ax0_2, ax1_2, ax2_2, ax3]
            for i1, i2_1, i3_1, i4, i5 in T.grid(16, 75, 75, 7, 7):
                with Ts.sblock("tensor"):
                    ax0_3, ax1_3 = Ts.axis.remap("SS", [i0, i1])
                    ax2_3 = Ts.axis.spatial(225, i2_0 * 75 + i2_1)
                    ax3 = Ts.axis.spatial(225, i3_0 * 75 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.reads(pad_temp[ax0_3, ax1_3, ax2_3 + rv0, ax3 + rv1])
                    Ts.writes(tensor[ax0_3, ax1_3, ax2_3, ax3])
                    with Ts.init():
                        tensor[ax0_3, ax1_3, ax2_3, ax3] = T.int8(0)
                    tensor[ax0_3, ax1_3, ax2_3, ax3] = (
                        tensor[ax0_3, ax1_3, ax2_3, ax3]
                        + pad_temp[ax0_3, ax1_3, ax2_3 + rv0, ax3 + rv1]
                    )

    sch = tvm.s_tir.Schedule(sum_pool_2d, debug_mask="all")
    block = sch.get_sblock("tensor")
    pad = sch.get_sblock("pad_temp")
    n, c, h, w, kh, kw = sch.get_loops(block)
    ho, hi = sch.split(h, [3, 75])
    wo, wi = sch.split(w, [3, 75])
    sch.reorder(n, ho, wo, c, hi, wi, kh, kw)
    sch.compute_at(sch.get_sblock("pad_temp"), wo)
    sch.decompose_padding(pad, sch.get_loops(pad)[0])
    check_decompose_padding(sum_pool_2d, sch.mod["main"], pooling_decompose_2, check_run=True)


def test_decompose_hw_padding_non_perfect_tiled():
    """Case 3. non-perfect tiling and then decompose"""

    @Ts.function
    def pooling_decompose_3(
        x: T.Tensor((1, 16, 225, 225), "int8"), tensor: T.Tensor((1, 16, 225, 225), "int8")
    ) -> None:
        pad_temp = Ts.sblock_alloc_tensor([1, 16, 231, 231], dtype="int8")
        for i0, i2_0, i3_0 in T.grid(1, 3, 3):
            for ax0, ax1, ax2 in T.grid(16, 86, 86):
                with Ts.sblock("pad_temp_pad_const"):
                    ax0_1 = Ts.axis.spatial(1, 0)
                    ax1_1 = Ts.axis.spatial(16, ax0)
                    ax2_1 = Ts.axis.spatial(231, i2_0 * 80 + ax1)
                    ax3 = Ts.axis.spatial(231, i3_0 * 80 + ax2)
                    Ts.where(i2_0 * 80 + ax1 < 231 and i3_0 * 80 + ax2 < 231)
                    Ts.reads()
                    Ts.writes(pad_temp[ax0_1, ax1_1, ax2_1, ax3])
                    pad_temp[ax0_1, ax1_1, ax2_1, ax3] = T.int8(0)
            for ax0, ax1, ax2 in T.grid(16, 86, 86):
                with Ts.sblock("pad_temp"):
                    ax0_2 = Ts.axis.spatial(1, 0)
                    ax1_2 = Ts.axis.spatial(16, ax0)
                    ax2_2 = Ts.axis.spatial(225, i2_0 * 80 + ax1 - 3)
                    ax3 = Ts.axis.spatial(225, i3_0 * 80 + ax2 - 3)
                    Ts.where(
                        3 <= i2_0 * 80 + ax1
                        and i2_0 * 80 + ax1 < 228
                        and 3 <= i3_0 * 80 + ax2
                        and i3_0 * 80 + ax2 < 228
                        and i2_0 * 80 + ax1 < 231
                        and i3_0 * 80 + ax2 < 231
                    )
                    Ts.reads(x[ax0_2, ax1_2, ax2_2, ax3])
                    Ts.writes(pad_temp[ax0_2, ax1_2, ax2_2 + 3, ax3 + 3])
                    pad_temp[ax0_2, ax1_2, ax2_2 + 3, ax3 + 3] = x[ax0_2, ax1_2, ax2_2, ax3]
            for i1, i2_1, i3_1, i4, i5 in T.grid(16, 80, 80, 7, 7):
                with Ts.sblock("tensor"):
                    ax0_3, ax1_3 = Ts.axis.remap("SS", [i0, i1])
                    ax2_3 = Ts.axis.spatial(225, i2_0 * 80 + i2_1)
                    ax3 = Ts.axis.spatial(225, i3_0 * 80 + i3_1)
                    rv0, rv1 = Ts.axis.remap("RR", [i4, i5])
                    Ts.where(i2_0 * 80 + i2_1 < 225 and i3_0 * 80 + i3_1 < 225)
                    Ts.reads(pad_temp[ax0_3, ax1_3, ax2_3 + rv0, ax3 + rv1])
                    Ts.writes(tensor[ax0_3, ax1_3, ax2_3, ax3])
                    with Ts.init():
                        tensor[ax0_3, ax1_3, ax2_3, ax3] = T.int8(0)
                    tensor[ax0_3, ax1_3, ax2_3, ax3] = (
                        tensor[ax0_3, ax1_3, ax2_3, ax3]
                        + pad_temp[ax0_3, ax1_3, ax2_3 + rv0, ax3 + rv1]
                    )

    sch = tvm.s_tir.Schedule(sum_pool_2d, debug_mask="all")
    block = sch.get_sblock("tensor")
    pad = sch.get_sblock("pad_temp")
    n, c, h, w, kh, kw = sch.get_loops(block)
    ho, hi = sch.split(h, [None, 80])
    wo, wi = sch.split(w, [None, 80])
    sch.reorder(n, ho, wo, c, hi, wi, kh, kw)
    sch.compute_at(sch.get_sblock("pad_temp"), wo)
    sch.decompose_padding(pad, sch.get_loops(pad)[3])
    check_decompose_padding(sum_pool_2d, sch.mod["main"], pooling_decompose_3, check_run=True)


def test_decompose_wrt_single_child_subtree():
    """Test the case when the decompose position is under the single child subtree"""

    @Ts.function
    def pad_op(
        x: T.Tensor((1, 16, 225, 225), "int8"),
        y: T.Tensor((1, 16, 231, 231), dtype="int8"),
    ):
        for i0, i1, i2, i3 in T.grid(1, 16, 231, 231):
            with Ts.sblock("pad_temp"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                y[ax0, ax1, ax2, ax3] = T.if_then_else(
                    3 <= ax2 and ax2 < 228 and 3 <= ax3 and ax3 < 228,
                    x[ax0, ax1, ax2 - 3, ax3 - 3],
                    T.int8(0),
                )

    @Ts.function
    def pad_op_after(
        x: T.Tensor((1, 16, 225, 225), "int8"), y: T.Tensor((1, 16, 231, 231), "int8")
    ):
        for i0, i1 in T.grid(1, 16):
            for i2, i3 in T.grid(231, 231):
                with Ts.sblock("pad_temp_pad_const"):
                    ax0 = Ts.axis.spatial(1, 0)
                    ax1, ax2, ax3 = Ts.axis.remap("SSS", [i1, i2, i3])
                    y[ax0, ax1, ax2, ax3] = T.int8(0)
            for i2, i3 in T.grid(225, 225):
                with Ts.sblock("pad_temp"):
                    ax0 = Ts.axis.spatial(1, 0)
                    ax1, ax2, ax3 = Ts.axis.remap("SSS", [i1, i2, i3])
                    y[ax0, ax1, ax2 + 3, ax3 + 3] = x[ax0, ax1, ax2, ax3]

    sch = tvm.s_tir.Schedule(pad_op, debug_mask="all")
    pad = sch.get_sblock("pad_temp")
    _, _, h, _ = sch.get_loops(pad)
    sch.decompose_padding(pad, h)
    check_decompose_padding(pad_op, sch.mod["main"], pad_op_after, check_run=True)


def test_not_to_decompose_trivial_predicate():
    """Test the case when the padding condition is trivial"""

    @Ts.function
    def trivial_pad(
        x: T.Tensor((1, 16, 225, 225), "int8"), y: T.Tensor([1, 16, 225, 225], dtype="int8")
    ):
        for i0, i1, i2, i3 in T.grid(1, 16, 225, 225):
            with Ts.sblock("pad_temp"):
                ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                y[ax0, ax1, ax2, ax3] = T.if_then_else(
                    0 <= ax2 and ax2 < 225 and 0 <= ax3 and ax3 < 225,
                    x[ax0, ax1, ax2, ax3],
                    T.int8(0),
                )

    sch = tvm.s_tir.Schedule(trivial_pad, debug_mask="all")
    pad = sch.get_sblock("pad_temp")
    _, _, h, _ = sch.get_loops(pad)
    assert not sch.can_decompose_padding(pad, h)


if __name__ == "__main__":
    tvm_tirx.testing.main()
