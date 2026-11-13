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
# ruff: noqa: F401, F841

import pytest
from tvm_ffi import structural_walk

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import Schedule, TensorIntrin
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.testing import te_workload
from tvm.s_tir.schedule.analysis import (
    TensorizeInfo,
    get_auto_tensorize_mapping_info,
    get_tensorize_loop_mapping,
    is_output_block,
    suggest_index_map,
)
from tvm.s_tir.tensor_intrin.x86 import dot_product_16x4_u8i8i32_desc
from tvm.te import create_function
from tvm_tirx.ir import Evaluate, For, ForKind
from tvm_tirx.ir.prim import expr_deep_equal
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import IndexMap, TensorType, Var, floordiv, floormod


def _make_vars(*args: str) -> list[Var]:
    return [Var(arg, ty="int32") for arg in args]


def _make_loops(loop_vars: list[Var], extents: list[int]) -> list[For]:
    assert len(loop_vars) == len(extents)
    return [
        For(
            var=loop_var,
            begin=0,
            end=extent,
            step=tvm_tirx.ir.IntImm(loop_var.ty, 1),
            kind=ForKind.DEFAULT,
            body=Evaluate(0),
        )
        for loop_var, extent in zip(loop_vars, extents)
    ]


def test_suggest_index_map_simple():
    i, j = _make_vars("i", "j")
    index_map = suggest_index_map(
        tensor=Var("tensor", TensorType([8, 256], "float32")),
        indices=[
            floordiv(i, 16) * 4 + floordiv(j, 16),
            floormod(i, 16) * 16 + floormod(j, 16),
        ],
        loops=_make_loops(
            loop_vars=[i, j],
            extents=[32, 64],
        ),
        predicate=True,
    )
    expected_index_map = IndexMap.from_func(
        lambda x, y: [
            floordiv(x, 4),
            floordiv(y, 16),
            floormod(x, 4),
            floormod(y, 16),
        ],
    )
    assert index_map.is_equivalent_to(expected_index_map)


def test_suggest_index_map_bijective():
    i, j = _make_vars("i", "j")
    index_map = suggest_index_map(
        tensor=Var("tensor", TensorType([8], "float32")),
        indices=[floormod(j, 4) * 2 + i],
        loops=_make_loops(
            loop_vars=[i, j],
            extents=[2, 32],
        ),
        predicate=True,
    )
    expected_index_map = IndexMap.from_func(
        lambda x: [
            floormod(x, 2),
            floordiv(x, 2),
        ],
    )
    assert index_map.is_equivalent_to(expected_index_map)


def test_suggest_index_map_winograd():
    """use case in winograd conv where the indices are complicated"""
    fused_outer, i3_3_fused, i4_0, i4_1 = _make_vars("fused_outer", "i3_3_fused", "i4_0", "i4_1")
    eps = floordiv(fused_outer, 336) * 2 + floordiv(floormod(fused_outer, 16), 8)
    nu = floordiv(floormod(fused_outer, 336), 112) * 2 + floordiv(floormod(fused_outer, 8), 4)
    co = floormod(fused_outer, 4) * 32 + i3_3_fused
    ci = (i4_0 * 32) + i4_1
    tensor = Var("tensor", TensorType([6, 6, 128, 128], "float32"))
    index_map = suggest_index_map(
        tensor=tensor,
        indices=[eps, nu, co, ci],
        loops=_make_loops(
            loop_vars=[fused_outer, i3_3_fused, i4_0, i4_1],
            extents=[1008, 32, 4, 32],
        ),
        predicate=True,
    )
    expected_index_map = IndexMap.from_func(
        lambda i0, i1, i2, i3: (
            floordiv(i0, 2),
            floordiv(i1, 2),
            floormod(i0, 2),
            floormod(i1, 2) * 4 + floordiv(i2, 32),
            floormod(i2, 32),
            floordiv(i3, 32),
            floormod(i3, 32),
        )
    )
    assert index_map.is_equivalent_to(expected_index_map)
    inverse_index_map = index_map.inverse(tensor.shape)
    expected_inverse_index_map = IndexMap.from_func(
        lambda i0, i1, i2, i3, i4, i5, i6: (
            ((i0 * 2) + i2),
            i1 * 2 + floordiv(i3, 4),
            floormod(i3, 4) * 32 + i4,
            ((i5 * 32) + i6),
        )
    )
    assert inverse_index_map.is_equivalent_to(expected_inverse_index_map)


@tvm_tirx.script.ir_module
class DenseTIRModule:
    @Ts.function
    def main(
        placeholder: T.Tensor((1024, 1024), "uint8"),
        placeholder_1: T.Tensor((64, 256, 16, 4), "int8"),
        compute: T.Tensor((1024, 1024), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            for i0, i1, i2 in T.grid(1024, 1024, 1024):
                with Ts.sblock("compute"):
                    i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
                    Ts.reads(placeholder[i, k], placeholder_1[j // 16, k // 4, j % 16, k % 4])
                    Ts.writes(compute[i, j])
                    with Ts.init():
                        compute[i, j] = 0
                    compute[i, j] = compute[i, j] + T.cast(placeholder[i, k], ty="int32") * T.cast(
                        placeholder_1[j // 16, k // 4, j % 16, k % 4], ty="int32"
                    )


@tvm_tirx.script.ir_module
class Conv2dNCHWcTIRModule:
    @Ts.function
    def main(
        placeholder: T.Tensor((1, 4, 56, 56, 16), "uint8"),
        placeholder_1: T.Tensor((16, 4, 1, 1, 4, 16, 4), "int8"),
        conv2d_NCHWc_int8: T.Tensor((1, 16, 56, 56, 16), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 16, 56, 56, 16, 1, 1, 4, 4, 4):
            with Ts.sblock("conv2d_NCHWc_int8"):
                (
                    n,
                    oc_chunk,
                    oh,
                    ow,
                    oc_block,
                    kh,
                    kw,
                    ic_outer,
                    ic_f_inner,
                    ic_s_inner,
                ) = Ts.axis.remap("SSSSSRRRRR", [i0, i1, i2, i3, i4, i5, i6, i7, i8, i9])
                Ts.reads(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                )
                Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                with Ts.init():
                    conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = 0
                conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[
                    n, oc_chunk, oh, ow, oc_block
                ] + T.cast(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                    ty="int32",
                ) * T.cast(
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner],
                    ty="int32",
                )


def collect_loops(function):
    loops = []
    structural_walk(function.body, (tvm_tirx.ir.For, loops.append), order="pre")

    return loops


def test_get_tensorize_loop_mapping_dense_16x4():
    s = Schedule(DenseTIRModule)
    block = s.get_sblock("compute")

    info = get_tensorize_loop_mapping(s, block, dot_product_16x4_u8i8i32_desc)

    assert isinstance(info, TensorizeInfo)

    desc_loop_to_sref = dict((v, k) for k, v in info.loop_map.items())

    desc_loops = collect_loops(dot_product_16x4_u8i8i32_desc)
    _, loop_j, loop_k = s.get_loops(block)

    assert desc_loops[0] in desc_loop_to_sref and desc_loops[1] in desc_loop_to_sref
    assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(loop_j)
    assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(loop_k)


def test_get_tensorize_loop_mapping_conv2d_nchwc_16x4():
    s = Schedule(Conv2dNCHWcTIRModule)
    block = s.get_sblock("conv2d_NCHWc_int8")

    info = get_tensorize_loop_mapping(s, block, dot_product_16x4_u8i8i32_desc)

    desc_loop_to_sref = dict((v, k) for k, v in info.loop_map.items())

    desc_loops = collect_loops(dot_product_16x4_u8i8i32_desc)

    # i4 corresonds to the inner output channel axis of the NCHWc output tensor
    # for i0, i1, i2, i3, i4, i5, i6, i7, i8, i9 in T.grid(1, 16, 56, 56, 16, 1, 1, 4, 4, 4):
    _, _, _, _, i4, _, _, _, _, i9 = s.get_loops(block)

    assert desc_loops[0] in desc_loop_to_sref and desc_loops[1] in desc_loop_to_sref
    assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(i4)
    assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(i9)


def test_get_tensorize_loop_mapping_matmul_mma():
    matmul_16x16x16xf16f16f16_desc_a_offset = T.dynamic(
        "matmul_16x16x16xf16f16f16_desc_a_offset", "int32"
    )
    matmul_16x16x16xf16f16f16_desc_b_offset = T.dynamic(
        "matmul_16x16x16xf16f16f16_desc_b_offset", "int32"
    )
    matmul_16x16x16xf16f16f16_desc_c_offset = T.dynamic(
        "matmul_16x16x16xf16f16f16_desc_c_offset", "int32"
    )

    @Ts.function
    def matmul_16x16x16xf16f16f16_desc(
        A: T.Tensor(
            (16, 16),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(16, 16)] + (matmul_16x16x16xf16f16f16_desc_a_offset)),
        ),
        B: T.Tensor(
            (16, 16),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(16, 16)] + (matmul_16x16x16xf16f16f16_desc_b_offset)),
        ),
        C: T.Tensor(
            (16, 16),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(16, 16)] + (matmul_16x16x16xf16f16f16_desc_c_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
            Ts.writes(C[0:16, 0:16])
            for i, j, k in T.grid(16, 16, 16):
                with Ts.sblock("update"):
                    vii, vjj, vkk = Ts.axis.remap("SSR", [i, j, k])
                    C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]

    matmul = create_function(
        te_workload.matmul_relu(
            n=512,
            m=512,
            k=512,
        )
    )

    s = Schedule(matmul)
    block = s.get_sblock("C")
    i0, i1, i2 = s.get_loops(block)
    desc_loops = collect_loops(matmul_16x16x16xf16f16f16_desc)

    for do_reorder in [False, True]:
        # Mapping should be invariant to the loop permutation
        if do_reorder:
            s.reorder(i2, i0, i1)

        info = get_tensorize_loop_mapping(s, block, matmul_16x16x16xf16f16f16_desc)
        assert info is not None
        desc_loop_to_sref = dict((v, k) for k, v in info.loop_map.items())

        for i in range(3):
            assert desc_loops[i] in desc_loop_to_sref

        assert s.get(desc_loop_to_sref[desc_loops[0]]) == s.get(i0)
        assert s.get(desc_loop_to_sref[desc_loops[1]]) == s.get(i1)
        assert s.get(desc_loop_to_sref[desc_loops[2]]) == s.get(i2)


def matmul_16x8x8_desc(out_dtype):
    desc_a_offset = T.dynamic("desc_a_offset", "int32")
    desc_b_offset = T.dynamic("desc_b_offset", "int32")
    desc_c_offset = T.dynamic("desc_c_offset", "int32")

    @Ts.function
    def desc(
        A: T.Tensor((16, 8), "float16", layout=T.TileLayout(T.S[(16, 8)] + (desc_a_offset))),
        B: T.Tensor((8, 8), "float16", layout=T.TileLayout(T.S[(8, 8)] + (desc_b_offset))),
        C: T.Tensor((16, 8), out_dtype, layout=T.TileLayout(T.S[(16, 8)] + (desc_c_offset))),
    ):
        with Ts.sblock("root"):
            Ts.reads(C[0:16, 0:8], A[0:16, 0:8], B[0:8, 0:8])
            Ts.writes(C[0:16, 0:8])
            for i, j, k in T.grid(16, 8, 8):
                with Ts.sblock("update"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    C[vi, vj] += T.cast(A[vi, vk], ty=out_dtype) * T.cast(B[vk, vj], ty=out_dtype)

    return desc


def test_get_tensorize_loop_mapping_padding_matmul():
    matmul = create_function(
        te_workload.matmul_relu(
            n=127,
            m=256,
            k=65,
            in_dtype="float16",
            out_dtype="float16",
        )
    )
    s = Schedule(matmul)
    block = s.get_sblock("C")

    desc = matmul_16x8x8_desc("float16")
    info = get_tensorize_loop_mapping(s, block, desc, allow_padding=True)
    assert info is not None
    expected_padding = [16, 1, 8]
    actual_padding = info.block_iter_paddings
    assert actual_padding is not None
    assert len(actual_padding) == len(expected_padding)
    for actual, expected in zip(actual_padding, expected_padding):
        assert actual == expected


def check_index_map(workload, block_name, desc_func, expected_index_map):
    s = Schedule(workload)
    block = s.get_sblock(block_name)
    info = get_auto_tensorize_mapping_info(s, block, desc_func)
    if expected_index_map is None:
        assert info is None
        return
    assert len(info.mappings) == 1
    assert IndexMap.from_func(expected_index_map).is_equivalent_to(info.mappings[0])


def test_get_auto_tensorize_mapping_info_conv2d():
    conv2d = create_function(
        te_workload.conv2d_nhwc(4, 16, 16, 64, 64, 3, 1, 1, in_dtype="float16", out_dtype="float32")
    )
    check_index_map(
        conv2d,
        "conv2d_nhwc",
        matmul_16x8x8_desc("float32"),
        lambda n, h, w, c, rh, rw, rc: (n * 256 + h * 16 + w, c, rh * 192 + rw * 64 + rc),
    )


def test_get_auto_tensorize_mapping_info_conv2d_unit_batch():
    conv2d = create_function(
        te_workload.conv2d_nhwc(1, 16, 16, 64, 64, 3, 1, 1, in_dtype="float16", out_dtype="float32")
    )
    check_index_map(
        conv2d,
        "conv2d_nhwc",
        matmul_16x8x8_desc("float32"),
        lambda n, h, w, c, rh, rw, rc: (n * 256 + h * 16 + w, c, rh * 192 + rw * 64 + rc),
    )


@pytest.mark.parametrize("b,m,n,k", [(1, 512, 512, 512), (16, 32, 32, 32)])
def test_get_auto_tensorize_mapping_info_batch_matmul(b, m, n, k):
    matmul = create_function(
        te_workload.batch_matmul_nkkm(b, m, n, k, in_dtype="float16", out_dtype="float32")
    )
    check_index_map(matmul, "Z", matmul_16x8x8_desc("float32"), lambda b, m, n, k: (b, m, n, k))


@pytest.mark.parametrize(
    "n,m,k,expected",
    [
        (
            512,
            512,
            512,
            lambda n, m, k: (
                n,
                m,
                k,
            ),
        ),
        (1, 32, 32, lambda n, m, k: (n, m, k)),
    ],
)
def test_get_auto_tensorize_mapping_info_matmul(n, m, k, expected):
    matmul = create_function(te_workload.matmul(n, m, k, in_dtype="float16", out_dtype="float32"))
    check_index_map(matmul, "C", matmul_16x8x8_desc("float32"), expected)


def test_is_output_block():
    @Ts.function
    def two_elementwise(
        A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor((128, 128), "float32")

        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 1.0

    sch = tvm.s_tir.Schedule(two_elementwise)
    block_rv = sch.get_sblock("C")
    assert is_output_block(sch, block_rv)


def test_empty_grid():
    @Ts.function
    def foo(out: T.Tensor((T.int64(1), T.int64(8), T.int64(8)), "int32")):
        act = Ts.sblock_alloc_tensor((1, 8, 8), "int32")
        for z2, y2, x2 in T.grid(1, 8, 8):
            with Ts.sblock("b0"):
                az, ay, ax = Ts.axis.remap("SSS", [z2, y2, x2])
                Ts.writes(act[az, ay, ax])
                act[az, ay, az] = T.int32(0)
        # Empty grid:
        for z1, y1, x1 in T.grid(0, 8, 8):
            with Ts.sblock("b1"):
                az, ay, ax = Ts.axis.remap("SSS", [z1, y1, x1])
                Ts.reads(act[az + 1, ay, ax])
                Ts.writes(out[az, ay, ax])
                out[az, ay, ax] = act[az + 1, ay, ax]
        # The block below is not needed to show the bug, but the 'out'
        # tensor would be undefined without it.
        for z2, y2, x2 in T.grid(1, 8, 8):
            with Ts.sblock("b2"):
                az, ay, ax = Ts.axis.remap("SSS", [z2, y2, x2])
                Ts.writes(out[az, ay, ax])
                out[az, ay, az] = T.int32(0)

    # This caused a crash before.
    sch = tvm.s_tir.Schedule(foo)


if __name__ == "__main__":
    tvm_tirx.testing.main()
