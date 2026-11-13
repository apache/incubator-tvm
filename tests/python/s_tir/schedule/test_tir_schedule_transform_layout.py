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
# ruff: noqa: E501, E731, F401
import sys

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
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

def packed_index_map_func(m, n):
    return m // 16, n // 16, m % 16, n % 16

@Ts.function
def two_elementwise(A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def two_elementwise_transformed_intermediate_tensor(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((8, 8, 16, 16), "float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi // 16, vj // 16, vi % 16, vj % 16] + 1.0

@Ts.function
def two_elementwise_transformed_input_tensor(
    A: T.Tensor((8, 8, 16, 16), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi // 16, vj // 16, vi % 16, vj % 16] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0

@Ts.function
def two_elementwise_transformed_output_tensor(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((8, 8, 16, 16), "float32")
) -> None:
    B = Ts.sblock_alloc_tensor((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi // 16, vj // 16, vi % 16, vj % 16] = B[vi, vj] + 1.0

@Ts.function
def elementwise(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")) -> None:
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

@Ts.function
def elementwise_transformed(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")) -> None:
    for i in range(16384):
        with Ts.sblock("B"):
            vi = Ts.axis.remap("S", [i])
            B[vi // 128, vi % 128] = A[vi // 128, vi % 128] * 2.0

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
def conv2d_nhwc_transformed(
    Input: T.Tensor((1, 224, 224, 3), "float32"),
    Weight: T.Tensor((7, 7, 3, 64), "float32"),
    Conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32"),
) -> None:
    PadInput = Ts.sblock_alloc_tensor([1, 230, 230, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with Ts.sblock("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            Ts.reads(Input[i0_1, i1_1 - 3, i2_1 - 3, i3_1])
            Ts.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                i1_1 >= 3 and i1_1 < 227 and i2_1 >= 3 and i2_1 < 227,
                Input[i0_1, i1_1 - 3, i2_1 - 3, i3_1],
                T.float32(0),

            )
    for ax0, ax1, ax2 in T.grid(12544, 64, 147):
        with Ts.sblock("conv2d_nhwc"):
            v0, v1, v2 = Ts.axis.remap("SSR", [ax0, ax1, ax2])
            with Ts.init():
                Conv2d_nhwc[0, v0 // 112, v0 % 112, v1] = T.float32(0)
            Conv2d_nhwc[0, v0 // 112, v0 % 112, v1] = Conv2d_nhwc[0, v0 // 112, v0 % 112, v1] + PadInput[0, v0 // 112 * 2 + v2 // 21, v0 % 112 * 2 + v2 % 21 // 3, v2 % 3] * Weight[v2 // 21, v2 % 21 // 3, v2 % 3, v1]

@Ts.function
def two_elementwise_unit_dim(A: T.Tensor((1, 128), "float32"), C: T.Tensor((1, 128), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((1, 128), "float32")
    for i, j in T.grid(1, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(1, 128):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0

# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_block_name(request):
    return request.param


def test_two_elementwise_transform_intermediate_tensor(use_block_name):
    sch = tvm.s_tir.Schedule(two_elementwise, debug_mask="all")

    if use_block_name:
        sch.transform_layout(
            block="B",
            tensor=sch.get(sch.get_sblock("B")).writes[0].source.name,
            index_map=packed_index_map_func,
        )
    else:
        block = sch.get_sblock("B")
        sch.transform_layout(block, ("write", 0), packed_index_map_func)

    assert_structural_equal_ignore_global_symbol(
        two_elementwise_transformed_intermediate_tensor, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_transform_layout_with_sampling():
    sch = tvm.s_tir.Schedule(two_elementwise, debug_mask="all")
    block_b = sch.get_sblock("B")
    loop = sch.get_loops(block_b)[-1]
    j0, j1, j2 = sch.sample_perfect_tile(loop, 3, decision=[4, 8, 4])
    sch.transform_layout(block_b, ("write", 0), lambda i, j: (i, j // (j1 * j2), j % (j1 * j2)))
    verify_trace_roundtrip(sch=sch, mod=two_elementwise, text_format="json")


def test_two_elementwise_transform_input_tensor(use_block_name):
    sch = tvm.s_tir.Schedule(two_elementwise, debug_mask="all")

    if use_block_name:
        sch.transform_layout(
            index_map=packed_index_map_func,
            block="B",
            tensor="A",
        )
    else:
        block = sch.get_sblock("B")
        sch.transform_layout(block, ("read", 0), packed_index_map_func)

    assert_structural_equal_ignore_global_symbol(
        two_elementwise_transformed_input_tensor, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_output_tensor(use_block_name):
    sch = tvm.s_tir.Schedule(two_elementwise, debug_mask="all")

    if use_block_name:
        sch.transform_layout(
            index_map=packed_index_map_func,
            block="C",
            tensor="C",
        )
    else:
        block = sch.get_sblock("C")
        sch.transform_layout(block, ("write", 0), packed_index_map_func)

    assert_structural_equal_ignore_global_symbol(
        two_elementwise_transformed_output_tensor, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_unit_dim(use_block_name):
    sch = tvm.s_tir.Schedule(two_elementwise_unit_dim, debug_mask="all")
    index_map = lambda i, j: (i, j)

    if use_block_name:
        sch.transform_layout(
            index_map=index_map,
            block="B",
            tensor=sch.get(sch.get_sblock("B")).writes[0].source.name,
        )
    else:
        block = sch.get_sblock("B")
        sch.transform_layout(block, ("write", 0), index_map)

    assert_structural_equal_ignore_global_symbol(two_elementwise_unit_dim, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise_unit_dim)


def test_simplify():
    sch = tvm.s_tir.Schedule(two_elementwise, debug_mask="all")

    i, j = sch.get_loops(sch.get_sblock("C"))
    i, i_inner = sch.split(i, factors=[None, 16])
    j, j_inner = sch.split(j, factors=[None, 16])

    sch.reorder(
        i,
        j,
        i_inner,
        j_inner,
    )

    block_outer = sch.blockize(i_inner)

    B = sch.cache_read(block_outer, 0, "global")
    sch.transform_layout(B, ("write", 0), lambda i, j: (i // 16, j // 16, i % 16, j % 16))

    @Ts.function
    def ref(B: T.Tensor((8, 8, 16, 16), "float32"), C: T.Tensor((128, 128), "float32")):
        for i_0, j_0 in T.grid(8, 8):
            with Ts.sblock("C_o"):
                vi_o, vj_o = Ts.axis.remap("SS", [i_0, j_0])
                Ts.reads(B[vi_o, vj_o, 0:16, 0:16])
                Ts.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i_1, j_1])
                        Ts.reads(B[vi_o, vj_o, vi, vj])
                        Ts.writes(C[vi_o * 16 + vi, vj_o * 16 + vj])
                        C[vi_o * 16 + vi, vj_o * 16 + vj] = B[vi_o, vj_o, vi, vj] + T.float32(1)

                        # Without simplification
                        # Ts.reads(B[vi // 16 + vi_o, vj // 16 + vj_o, vi % 16, vj % 16])
                        # C[...] = B[vi // 16 + vi_o, vj // 16 + vj_o, vi % 16, vj % 16] + T.float32(1)

    expected = tvm_tirx.tirx.Function(
        [param for param in ref.params if tvm_tirx.tirx.is_tensor_var(param)],
        ref.body[0].block.body,
    )
    actual_block = sch.get(block_outer)
    actual = tvm_tirx.tirx.Function(
        [actual_block.reads[0].source, actual_block.writes[0].source],
        sch.get(sch.get_loops(block_outer)[0]),
    )
    tvm_tirx.ir.assert_structural_equal(expected, actual)


def test_var_args_sugar():
    @Ts.function
    def summation_3d(
        A: T.Tensor((1024, 1024, 32), "float32"), B: T.Tensor((1,), "float32")
    ) -> None:
        B[0] = 0
        for i, j, k in T.grid(1024, 1024, 32):
            with Ts.sblock("compute"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                B[0] = B[0] + A[vi, vj, vk]

    @Ts.function
    def summation_3d_split(
        A: T.Tensor((1024, 1024, 8, 4), "float32"), B: T.Tensor((1,), "float32")
    ) -> None:
        B[0] = 0
        for i, j, k in T.grid(1024, 1024, 32):
            with Ts.sblock("compute"):
                vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                B[0] = B[0] + A[vi, vj, vk // 4, vk % 4]

    sch = tvm.s_tir.Schedule(summation_3d, debug_mask="all")
    sch.transform_layout(
        index_map=lambda *indices, k: [*indices, k // 4, k % 4], block="compute", tensor="A"
    )
    assert_structural_equal_ignore_global_symbol(summation_3d_split, sch.mod["main"])


def test_transform_block_layout_basic(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block = "B" if use_block_name else sch.get_sblock("B")
    sch.transform_block_layout(block, lambda i, j: (i * 128 + j,))
    assert_structural_equal_ignore_global_symbol(elementwise_transformed, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=elementwise)


def test_transform_block_layout_conv2d_nhwc(use_block_name):
    sch = tvm.s_tir.Schedule(conv2d_nhwc, debug_mask="all")
    block = "conv2d_nhwc" if use_block_name else sch.get_sblock("conv2d_nhwc")
    sch.transform_block_layout(
        block,
        lambda n, h, w, co, rh, rw, rc: (n * 112 * 112 + h * 112 + w, co, rh * 7 * 3 + rw * 3 + rc),
    )
    assert_structural_equal_ignore_global_symbol(conv2d_nhwc_transformed, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=conv2d_nhwc)


def test_transform_block_layout_unit_dim(use_block_name):
    sch = tvm.s_tir.Schedule(two_elementwise_unit_dim, debug_mask="all")
    block = "B" if use_block_name else sch.get_sblock("B")
    sch.transform_block_layout(block, lambda i, j: (j, i))

    @Ts.function
    def two_elementwise_unit_dim_transformed(
        A: T.Tensor((1, 128), "float32"), C: T.Tensor((1, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor((1, 128), "float32")
        for j, i in T.grid(128, 1):
            with Ts.sblock("B"):
                vj, vi = Ts.axis.remap("SS", [j, i])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(1, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    assert_structural_equal_ignore_global_symbol(
        two_elementwise_unit_dim_transformed, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=two_elementwise_unit_dim)


def test_transform_block_layout_fail_non_affine(use_block_name):
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    block = "B" if use_block_name else sch.get_sblock("B")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.transform_block_layout(block, lambda i, j: (i + j,))


def test_transform_block_layout_fail_mixed_iter_type(use_block_name):
    sch = tvm.s_tir.Schedule(conv2d_nhwc, debug_mask="all")
    block = "conv2d_nhwc" if use_block_name else sch.get_sblock("conv2d_nhwc")
    with pytest.raises(tvm.s_tir.ScheduleError):
        sch.transform_block_layout(
            block,
            lambda n, h, w, co, rh, rw, rc: (n * 112 * 112 + h * 112 + w, co * 7 + rh, rw * 3 + rc),
        )


def test_transform_block_layout_int64_extent(use_block_name):
    @Ts.function
    def elementwise_int64_extent(
        A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0

    @Ts.function
    def elementwise_int64_extent_transformed(
        A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        # T.serial with explicit int64 min so the iter_var dom is all-int64
        # (`range(T.int64(...))` would emit an int32 min).
        for i in T.serial(T.int64(0), T.int64(16384)):
            with Ts.sblock("B"):
                vi = Ts.axis.remap("S", [i])
                B[vi // T.int64(128), vi % T.int64(128)] = (
                    A[vi // T.int64(128), vi % T.int64(128)] * 2.0
                )

    sch = tvm.s_tir.Schedule(elementwise_int64_extent, debug_mask="all")
    block = "B" if use_block_name else sch.get_sblock("B")
    sch.transform_block_layout(block, lambda i, j: (i * 128 + j,))
    assert_structural_equal_ignore_global_symbol(
        elementwise_int64_extent_transformed, sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=elementwise_int64_extent)


@pytest.mark.parametrize("pad_value", [None, 42])
def test_no_padding(pad_value):
    """Transformations without padding do not depend on pad_value."""

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor(16, "int32")
            for i in T.serial(16):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi] = 0

    @I.ir_module
    class Expected:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor([4, 4], "int32")
            for i in T.serial(16):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi // 4, vi % 4] = 0

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // 4, i % 4],
        pad_value=pad_value,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("pad_value", [None, 42])
def test_no_padding_multiple_usage(pad_value):
    """Transformations without padding do not depend on pad_value.

    Like test_no_padding, but the tensor A shows up in multiple
    locations.  To remain internally consistent, all instances of the
    tensor should be rewritten.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor(16, "int32")
            for i in T.serial(16):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi] = 0

            B = Ts.sblock_alloc_tensor(16, "int32")
            for i in T.serial(16):
                with Ts.sblock("other"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi]

    @I.ir_module
    class Expected:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor([4, 4], "int32")
            for i in T.serial(16):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi // 4, vi % 4] = 0

            B = Ts.sblock_alloc_tensor(16, "int32")
            for i in T.serial(16):
                with Ts.sblock("other"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi // 4, vi % 4]

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // 4, i % 4],
        pad_value=pad_value,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.parametrize("pad_value", [None, 42])
def test_no_padding_opaque_block(pad_value):
    """Transformations without padding do not depend on pad_value.

    Like test_no_padding, but tensor access is done in an opaque block.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor(16, "int32")
            for i in T.serial(16):
                with Ts.sblock("block"):
                    A[i] = 0

    @I.ir_module
    class Expected:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor([4, 4], "int32")
            for i in T.serial(16):
                with Ts.sblock("block"):
                    A[i // 4, i % 4] = 0

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // 4, i % 4],
        pad_value=pad_value,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_error_if_padding_forbidden():
    """Unless padding is explicitly enabled, should raise error"""

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi] = 0

    sch = tvm.s_tir.Schedule(Before)
    with pytest.raises(tvm.s_tir.schedule.schedule.ScheduleError):
        sch.transform_layout(
            "block",
            sch.get(sch.get_sblock("block")).writes[0].source.name,
            lambda i: [i // 4, i % 4],
            pad_value=None,
        )


def test_implicit_padding_assume_injective():
    """When pad_value is None and assume_injective_transform is set, the tensor can be implicitly
    padded. The padded region is not accessed because the original loop extent is not changed.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi] = 0

    @I.ir_module
    class Expected:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor([4, 4], "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi // 4, vi % 4] = 0

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // 4, i % 4],
        pad_value=None,
        assume_injective_transform=True,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_error_on_wrong_padding_type():
    """The padding must have the same dtype as the tensor"""

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi] = 0

    sch = tvm.s_tir.Schedule(Before)
    with pytest.raises(tvm.s_tir.schedule.schedule.ScheduleError):
        sch.transform_layout(
            "block",
            sch.get(sch.get_sblock("block")).writes[0].source.name,
            lambda i: [i // 4, i % 4],
            pad_value=tvm_tirx.ir.IntImm("int8", 0),
        )


def test_error_on_non_matching_types():
    """The padding must have the same dtype as the tensor"""

    @I.ir_module
    class Before:
        @Ts.function
        def main():
            A = Ts.sblock_alloc_tensor(14, "float32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi] = 0

    sch = tvm.s_tir.Schedule(Before)
    with pytest.raises(tvm.s_tir.schedule.schedule.ScheduleError):
        sch.transform_layout(
            "block",
            sch.get(sch.get_sblock("block")).writes[0].source.name,
            lambda i: [i // 4, i % 4],
            pad_value=0,
        )


@pytest.mark.parametrize("dtype", ["int32", "int8"])
def test_padded_transform_if_then_else(dtype):
    """Use if_then_else to represent padding, if possible.

    For a block that is a producer of the pre-transformation tensor,
    which visits all indices according to a row-major traversal, and
    which has no effect other than producing the transformed tensor,
    transform the loop iterators to be a row-major traversal of the
    post-transformation tensor, with padding represented by
    `T.if_then_else`.
    """

    @Ts.function(private=True)
    def before_func(A: T.Tensor(14, dtype)):
        B = Ts.sblock_alloc_tensor(14, dtype)
        for i in T.serial(14):
            with Ts.sblock("block"):
                vi = Ts.axis.remap("S", [i])
                B[vi] = A[vi]

    pad_value_imm = tvm_tirx.ir.IntImm(dtype, 0)

    @Ts.function(private=True)
    def expected_func(A: T.Tensor(14, dtype)):
        B = Ts.sblock_alloc_tensor([4, 4], dtype)
        for i, j in T.grid(4, 4):
            with Ts.sblock("block"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = T.if_then_else(vi == 3 and 2 <= vj, pad_value_imm, A[vi * 4 + vj])

    Before = tvm_tirx.IRModule({"main": before_func})
    Expected = tvm_tirx.IRModule({"main": expected_func})

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // 4, i % 4],
        pad_value=0,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_padded_transform_without_loop():
    """Handle padded writes without a loop

    The statement being replaced may be something other than a
    for-loop, such as if a loop has already been unrolled.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(14, "int32")):
            with Ts.sblock("root"):
                Ts.reads()
                Ts.writes()
                with Ts.sblock("block"):
                    A[0] = 0

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((4, 4), "int32")):
            with Ts.sblock("block"):
                A[0, 0] = 0

            for i, j in T.grid(4, 4):
                with Ts.sblock("tensor_A_padding"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Ts.where(i == 3 and 2 <= j)
                    A[vi, vj] = 0

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        "A",
        lambda i: [i // 4, i % 4],
        pad_value=0,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_padded_transform_if_then_else_reduction():
    """Like test_padded_transform_if_then_else, but with a reduction axis"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((14, 32), "int32")):
            B = Ts.sblock_alloc_tensor(14, "int32")
            for i, k in T.grid(14, 32):
                with Ts.sblock("block"):
                    vi, vk = Ts.axis.remap("SR", [i, k])
                    with Ts.init():
                        B[vi] = 0
                    B[vi] = B[vi] + A[vi, vk]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((14, 32), "int32")):
            B = Ts.sblock_alloc_tensor([4, 4], "int32")
            for i, j, k in T.grid(4, 4, 32):
                with Ts.sblock("block"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        B[vi, vj] = T.if_then_else(vi == 3 and 2 <= vj, 0, 0)
                    B[vi, vj] = T.if_then_else(
                        vi == 3 and 2 <= vj, 0, B[vi, vj] + A[vi * 4 + vj, vk]
                    )

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // 4, i % 4],
        pad_value=0,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_padded_transform_if_then_else_reduction_opaque():
    """Like test_padded_transform_if_then_else_reduction, but with opaque blocks"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor((14, 32), "int32")):
            B = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                B[i] = 0
                for k in T.serial(32):
                    with Ts.sblock("block"):
                        B[i] = B[i] + A[i, k]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((14, 32), "int32")):
            B = Ts.sblock_alloc_tensor([4, 4], "int32")
            for i, j in T.grid(4, 4):
                B[i, j] = T.if_then_else(i == 3 and 2 <= j, 0, 0)
                for k in T.serial(32):
                    with Ts.sblock("block"):
                        B[i, j] = T.if_then_else(i == 3 and 2 <= j, 0, B[i, j] + A[i * 4 + j, k])

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // 4, i % 4],
        pad_value=0,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_padded_transform_post_proc_if_required_due_to_side_effects():
    """Set the transformation padding in a post-processing block.

    Like test_padded_transform_if_then_else, but the block that produces B
    also has the effect of setting `C`.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(14, "int32")):
            B = Ts.sblock_alloc_tensor(14, "int32")
            C = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi]
                    C[vi] = 0

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor(14, "int32")):
            B = Ts.sblock_alloc_tensor([4, 4], "int32")
            C = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi // 4, vi % 4] = A[vi]
                    C[vi] = 0

            for i, j in T.grid(4, 4):
                with Ts.sblock("block_pad_B"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    Ts.where(i == 3 and 2 <= j)
                    B[vi, vj] = 0

    sch = tvm.s_tir.Schedule(Before)
    # Both local outputs are anonymous; select the first write by native identity.
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source,
        lambda i: [i // 4, i % 4],
        pad_value=0,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_padded_transform_of_input_creates_assumption():
    """Transformation of an input tensor places T.assume locally"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(14, "int32"), B: T.Tensor(14, "int32")):
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((4, 4), "int32"), B: T.Tensor(14, "int32")):
            for i, j in T.grid(4, 4):
                with Ts.sblock("tensor_A_assumption"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    T.evaluate(T.assume(not (vi == 3 and 2 <= vj) or A[vi, vj] == 42))

            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi // 4, vi % 4]

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        "A",
        lambda i: [i // 4, i % 4],
        pad_value=42,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_padded_transform_non_constant_value():
    """Allow an expression to specify the pad value.

    Like test_padded_transform_if_then_else, but the pad value depends on
    the indices.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(14, "int32")):
            B = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor(14, "int32")):
            B = Ts.sblock_alloc_tensor([4, 4], "int32")
            for i, j in T.grid(4, 4):
                with Ts.sblock("block"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = T.if_then_else(vi == 3 and 2 <= vj, vi + vj, A[vi * 4 + vj])

    sch = tvm.s_tir.Schedule(Before)
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // 4, i % 4],
        pad_value=lambda i, j: i + j,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


@pytest.mark.xfail(reason="Not yet implemented")
def test_padded_transform_repeated_tensor_element():
    """Allow an expression to specify the pad value.

    Like test_padded_transform_of_input_creates_assumption, but the pad
    value depends on another portion of the tensor.  In this case, the
    padding at the end of A contains repeated elements from the
    beginning of A.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(14, "int32")):
            B = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((4, 4), "int32")):
            for i, j in T.grid(4, 4):
                with Ts.sblock("tensor_A_assumption"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    T.evaluate(
                        T.assume(
                            not (vi == 3 and 2 <= vj)
                            or A[vi, vj] == A[((4 * vi + j) % 14) // 4, ((4 * vi + j) % 14) % 4]
                        )
                    )

            B = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.grid(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi // 4, vi % 4]

    sch = tvm.s_tir.Schedule(Before)
    A = sch.get(sch.get_sblock("block")).reads[0].source
    sch.transform_layout(
        "block",
        "A",
        lambda i: [i // 4, i % 4],
        pad_value=lambda i, j: A[(4 * i + j) % 14],
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_pad_value_may_not_reference_other_tensor():
    """Allow an expression to specify the pad value.

    Like test_padded_transform_repeated_tensor_element, but the pad value depends on
    a different tensor, which is not allowed.
    """

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(14, "int32")):
            B = Ts.sblock_alloc_tensor(14, "int32")
            for i in T.serial(14):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi]

    sch = tvm.s_tir.Schedule(Before)
    A = sch.get(sch.get_sblock("block")).reads[0].source
    other = tirx.Var("other", tirx.TensorType(1, A.ty.dtype))
    with pytest.raises(tvm.s_tir.schedule.schedule.ScheduleError):
        sch.transform_layout(
            "block",
            "A",
            lambda i: [i // 4, i % 4],
            pad_value=lambda i, j: other[0],
        )


def test_transform_layout_with_var():
    """Layout transform with dynamic parameter in transform"""

    @I.ir_module
    class Before:
        @Ts.function
        def main(A: T.Tensor(16, "int32"), n: T.int32):
            B = Ts.sblock_alloc_tensor(16, "int32")
            for i in T.serial(16):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor(16, "int32"), n: T.int32):
            B = Ts.sblock_alloc_tensor([(-16 % n + 16) // n, n], dtype="int32")
            for i, j in T.grid((-16 % n + 16) // n, n):
                with Ts.sblock("block"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = T.if_then_else(
                        # Checks if the transform introduced padding
                        -16 % n != 0
                        # If so, is vi in the last group (which may
                        # include padding).
                        and (vj + vi * n) // n == 16 // n
                        # And is vj within the padding
                        and 16 % n <= (vj + vi * n) % n,
                        0,
                        A[vj + vi * n],
                    )

    sch = tvm.s_tir.Schedule(Before)
    n = sch.mod["main"].params[1]
    sch.transform_layout(
        "block",
        sch.get(sch.get_sblock("block")).writes[0].source.name,
        lambda i: [i // n, i % n],
        pad_value=0,
    )
    After = sch.mod
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_index_map_dtype_legalize():
    """Test dtype legalization of the index map indices."""

    @Ts.function
    def func(A: T.Tensor(T.int64(58), "int32")):
        for i in T.serial(T.int64(58)):
            with Ts.sblock("block"):
                vi = Ts.axis.remap("S", [i])
                Ts.writes(A[vi])
                A[vi] = 0

    sch = tvm.s_tir.Schedule(func)

    # # The following error is raised from the IterVar constructor without the dtype legalization.
    # # RuntimeError: Check failed: dom->extent.dtype() == var.dtype() (int64 vs. int32) :
    # # The dtype of the extent of an IterVar (int64) must match its associated Var's dtype (int32)
    sch.transform_layout(
        sch.get_sblock("block"), tensor="A", index_map=lambda h: [h // 8, h % 8], pad_value=0
    )


def test_index_map_dtype_legalize_with_constant():
    """Legalization of inverse containing a constant output

    The index map `lambda i,j: [i, j//8, j % 8]` has an inverse `lambda i,j,k: [i, 8*j+k]`.
    """

    @Ts.function
    def func(A: T.Tensor(T.int64(16), "int32")):
        for i in T.grid(T.int64(16)):
            with Ts.sblock("block"):
                vi = Ts.axis.remap("S", [i])
                A[vi] = 0

    sch = tvm.s_tir.Schedule(func)

    # Triggering the error requires an IndexMap that introduces padding
    func = lambda i: [
        # And a constant to be one of the output indices.
        tirx.const(0, i.ty.dtype),
        (i + 1) // 8,
        (i + 1) % 8,
    ]

    # Previously, the legalization was only handled by propagating the
    # dtype of the indices to the transformed indices.  As a result,
    # output indices whose value did not depend on the input index
    # would be left with the incorrect dtype.

    # Prior to the bugfix, this resulted in the following error is
    # raised from the IterVar constructor.
    #
    # RuntimeError: Check failed: dom->extent.dtype() == var.dtype() (int64 vs. int32) :
    # The dtype of the extent of an IterVar (int64) must match its associated Var's dtype (int32)
    sch.transform_layout(block="block", tensor="A", index_map=func, pad_value=0)


def test_transform_layout_with_symbolic_bound():
    # fmt: off
    # pylint: disable=invalid-name,line-too-long,too-many-locals
    n = T.dynamic("n", "int64")

    @Ts.function
    def before(A: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), 'float16'), B: T.Tensor((T.int64(1), T.int64(32), n, T.int64(128)), 'float16'), C: T.Tensor((T.int64(1), T.int64(32), T.int64(1), n), 'float16')):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n, T.int64(128)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                Ts.writes(C[v_i0, v_i1, v_i2, v_i3])
                with Ts.init():
                    C[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
                C[v_i0, v_i1, v_i2, v_i3] = C[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]

    n = T.dynamic("n", "int64")

    @Ts.function
    def after(A: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), 'float16'), B: T.Tensor((T.int64(1), T.int64(32), n, T.int64(128)), 'float16'), C: T.Tensor((n * T.int64(32),), 'float16')):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n, T.int64(128)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                Ts.writes(C[v_i1 * n + v_i3])
                with Ts.init():
                    C[v_i1 * n + v_i3] = T.float16(0)
                C[v_i1 * n + v_i3] = C[v_i1 * n + v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]
    # pylint: enable=invalid-name,line-too-long,too-many-locals
    # fmt: on
    # pylint: disable=invalid-name
    _, _, n, _ = before.params[1].ty.shape
    sch = tvm.s_tir.Schedule(before)
    block = sch.get_sblock("NT_matmul")
    sch.transform_layout(
        block,
        ("write", 0),
        lambda x, y, z, w: x * 32 * n + y * n + z * n + w,
        assume_injective_transform=True,
    )
    # pylint: enable=invalid-name
    tvm_tirx.ir.assert_structural_equal(after, sch.mod["main"])


def test_transform_block_layout_with_symbolic_bound():
    # fmt: off
    # pylint: disable=invalid-name,line-too-long,too-many-locals
    n = T.dynamic("n", "int64")

    @Ts.function
    def before(A: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), 'float16'), B: T.Tensor((T.int64(1), T.int64(32), n, T.int64(128)), 'float16'), C: T.Tensor((n * T.int64(32),), 'float16')):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n, T.int64(128)):
            with Ts.sblock("NT_matmul"):
                v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_i3, v_k])
                Ts.writes(C[v_i1 * n + v_i3])
                with Ts.init():
                    C[v_i1 * n + v_i3] = T.float16(0)
                C[v_i1 * n + v_i3] = C[v_i1 * n + v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_i3, v_k]

    n = T.dynamic("n", "int64")

    @Ts.function
    def after(A: T.Tensor((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), 'float16'), B: T.Tensor((T.int64(1), T.int64(32), n, T.int64(128)), 'float16'), C: T.Tensor((n * T.int64(32),), 'float16')):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for ax0, ax1 in T.grid(n * T.int64(32), T.int64(128)):
            with Ts.sblock("NT_matmul"):
                v0, v1 = Ts.axis.remap("SR", [ax0, ax1])
                Ts.reads(A[T.int64(0), v0 // n, T.int64(0), v1], B[T.int64(0), v0 // n, v0 % n, v1])
                Ts.writes(C[v0])
                with Ts.init():
                    C[v0] = T.float16(0)
                C[v0] = C[v0] + A[T.int64(0), v0 // n, T.int64(0), v1] * B[T.int64(0), v0 // n, v0 % n, v1]
    # pylint: enable=invalid-name,line-too-long,too-many-locals
    # fmt: on
    # pylint: disable=invalid-name
    _, _, n, _ = before.params[1].ty.shape
    sch = tvm.s_tir.Schedule(before)
    block = sch.get_sblock("NT_matmul")
    sch.transform_block_layout(
        block,
        lambda x, y, z, w, k: (
            x * 32 * n + y * n + z * n + w,
            k,
        ),
    )
    # pylint: enable=invalid-name
    tvm_tirx.ir.assert_structural_equal(after, sch.mod["main"])


if __name__ == "__main__":
    tvm_tirx.testing.main()
