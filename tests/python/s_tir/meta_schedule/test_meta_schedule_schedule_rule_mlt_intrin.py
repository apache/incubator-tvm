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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# ruff: noqa: E501, F401, F841
from tvm import te
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
    print_sketches,
)
from tvm.s_tir.tensor_intrin.arm_cpu import DP4A_S8S8S32_INTRIN
from tvm.s_tir.tensor_intrin.x86 import AVX512_DOT_16x4_INTRIN as AVX512_INTRIN
from tvm.s_tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target


def test_x86_conv2d_nchwc(
    intrin=VNNI_INTRIN, target={"kind": "llvm", "mcpu": "cascadelake", "num-cores": 4}
):
    @Ts.function
    def conv2d_nchwc(
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

    # fmt: off
    @Ts.function
    def x86_conv2d_nchwc_0(placeholder: T.Tensor((1, 4, 56, 56, 16), "uint8"), placeholder_1: T.Tensor((16, 4, 1, 1, 4, 16, 4), "int8"), conv2d_NCHWc_int8: T.Tensor((1, 16, 56, 56, 16), "int32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # with Ts.sblock("root"):
        conv2d_NCHWc_int8_global = Ts.sblock_alloc_tensor((1, 16, 56, 56, 16), "int32")
        for i0_0, i1_0, i2_0, i3_0, i4_0_0, i0_1, i1_1, i2_1, i3_1, i4_0_1 in T.grid(1, 8, 28, 56, 1, 1, 2, 1, 1, 1):
            for i5_0, i6_0, i7_0, i8_0, i9_0_0, i0_2, i1_2, i2_2, i3_2, i4_0_2, i5_1, i6_1, i7_1, i8_1, i9_0_1, i0_3, i1_3, i2_3, i3_3, i4_0_3 in T.grid(1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1):
                with Ts.sblock("conv2d_NCHWc_int8_o"):
                    n = Ts.axis.spatial(1, i0_0 + i0_1 + i0_2 + i0_3)
                    oc_chunk = Ts.axis.spatial(16, i1_0 * 2 + i1_1 + i1_2 + i1_3)
                    oh = Ts.axis.spatial(56, i2_0 * 2 + i2_1 * 2 + i2_2 + i2_3)
                    ow = Ts.axis.spatial(56, i3_0 + i3_1 + i3_2 + i3_3)
                    oc_block_o = Ts.axis.spatial(1, i4_0_0 + i4_0_1 + i4_0_2 + i4_0_3)
                    kh = Ts.axis.reduce(1, i5_0 + i5_1)
                    kw = Ts.axis.reduce(1, i6_0 + i6_1)
                    ic_outer = Ts.axis.reduce(4, i7_0 * 4 + i7_1)
                    ic_f_inner = Ts.axis.reduce(4, i8_0 + i8_1)
                    ic_s_inner_o = Ts.axis.reduce(1, i9_0_0 + i9_0_1)
                    Ts.reads(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4:ic_f_inner * 4 + 4], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4])
                    Ts.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, 0:16])
                    Ts.sblock_attr({"meta_schedule.auto_tensorize": intrin})
                    with Ts.init():
                        for i4_1 in range(16):
                            with Ts.sblock("conv2d_NCHWc_int8_init"):
                                oc_block_i_init = Ts.axis.spatial(16, i4_1)
                                Ts.reads()
                                Ts.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i_init])
                                conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                    for i4_1, i9_1 in T.grid(16, 4):
                        with Ts.sblock("conv2d_NCHWc_int8"):
                            oc_block_i, ic_s_inner_i = Ts.axis.remap("SR", [i4_1, i9_1])
                            Ts.reads(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i], placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
                            Ts.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i] = conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i] + T.Cast("int32", placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i]) * T.Cast("int32", placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
            for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 1, 2, 1, 16):
                with Ts.sblock("conv2d_NCHWc_int8_global"):
                    v0 = Ts.axis.spatial(1, ax0)
                    v1 = Ts.axis.spatial(16, i1_0 * 2 + i1_1 + ax1)
                    v2 = Ts.axis.spatial(56, i2_0 * 2 + ax2)
                    v3 = Ts.axis.spatial(56, i3_0 + ax3)
                    v4 = Ts.axis.spatial(16, ax4)
                    Ts.reads(conv2d_NCHWc_int8_global[v0, v1, v2, v3, v4])
                    Ts.writes(conv2d_NCHWc_int8[v0, v1, v2, v3, v4])
                    conv2d_NCHWc_int8[v0, v1, v2, v3, v4] = conv2d_NCHWc_int8_global[v0, v1, v2, v3, v4]

    @Ts.function
    def x86_conv2d_nchwc_1(placeholder: T.Tensor((1, 4, 56, 56, 16), "uint8"), placeholder_1: T.Tensor((16, 4, 1, 1, 4, 16, 4), "int8"), conv2d_NCHWc_int8: T.Tensor((1, 16, 56, 56, 16), "int32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # with Ts.sblock("root"):
        conv2d_NCHWc_int8_global = Ts.sblock_alloc_tensor((1, 16, 56, 56, 16), "int32")
        for i0_0, i1_0, i2_0, i3_0, i4_0_0 in T.grid(1, 8, 28, 56, 1):
            for i0_1, i1_1, i2_1, i3_1, i4_0_1, i5_0, i6_0, i7_0, i8_0, i9_0_0, i0_2, i1_2, i2_2, i3_2, i4_0_2, i5_1, i6_1, i7_1, i8_1, i9_0_1, i0_3, i1_3, i2_3, i3_3, i4_0_3 in T.grid(1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1):
                with Ts.sblock("conv2d_NCHWc_int8_o"):
                    n = Ts.axis.spatial(1, i0_0 + i0_1 + i0_2 + i0_3)
                    oc_chunk = Ts.axis.spatial(16, i1_0 * 2 + i1_1 + i1_2 + i1_3)
                    oh = Ts.axis.spatial(56, i2_0 * 2 + i2_1 * 2 + i2_2 + i2_3)
                    ow = Ts.axis.spatial(56, i3_0 + i3_1 + i3_2 + i3_3)
                    oc_block_o = Ts.axis.spatial(1, i4_0_0 + i4_0_1 + i4_0_2 + i4_0_3)
                    kh = Ts.axis.reduce(1, i5_0 + i5_1)
                    kw = Ts.axis.reduce(1, i6_0 + i6_1)
                    ic_outer = Ts.axis.reduce(4, i7_0 * 4 + i7_1)
                    ic_f_inner = Ts.axis.reduce(4, i8_0 + i8_1)
                    ic_s_inner_o = Ts.axis.reduce(1, i9_0_0 + i9_0_1)
                    Ts.reads(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4:ic_f_inner * 4 + 4], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4])
                    Ts.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, 0:16])
                    Ts.sblock_attr({"meta_schedule.auto_tensorize": intrin})
                    with Ts.init():
                        for i4_1 in range(16):
                            with Ts.sblock("conv2d_NCHWc_int8_init"):
                                oc_block_i_init = Ts.axis.spatial(16, i4_1)
                                Ts.reads()
                                Ts.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i_init])
                                conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                    for i4_1, i9_1 in T.grid(16, 4):
                        with Ts.sblock("conv2d_NCHWc_int8"):
                            oc_block_i, ic_s_inner_i = Ts.axis.remap("SR", [i4_1, i9_1])
                            Ts.reads(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i], placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
                            Ts.writes(conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i])
                            Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                            conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i] = conv2d_NCHWc_int8_global[n, oc_chunk, oh, ow, oc_block_i] + T.Cast("int32", placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i]) * T.Cast("int32", placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
            for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 2, 2, 1, 16):
                with Ts.sblock("conv2d_NCHWc_int8_global"):
                    v0 = Ts.axis.spatial(1, ax0)
                    v1 = Ts.axis.spatial(16, i1_0 * 2 + ax1)
                    v2 = Ts.axis.spatial(56, i2_0 * 2 + ax2)
                    v3 = Ts.axis.spatial(56, i3_0 + ax3)
                    v4 = Ts.axis.spatial(16, ax4)
                    Ts.reads(conv2d_NCHWc_int8_global[v0, v1, v2, v3, v4])
                    Ts.writes(conv2d_NCHWc_int8[v0, v1, v2, v3, v4])
                    conv2d_NCHWc_int8[v0, v1, v2, v3, v4] = conv2d_NCHWc_int8_global[v0, v1, v2, v3, v4]

    @Ts.function
    def x86_conv2d_nchwc_2(placeholder: T.Tensor((1, 4, 56, 56, 16), "uint8"), placeholder_1: T.Tensor((16, 4, 1, 1, 4, 16, 4), "int8"), conv2d_NCHWc_int8: T.Tensor((1, 16, 56, 56, 16), "int32")) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # with Ts.sblock("root"):
        for i0_0, i1_0, i2_0, i3_0, i4_0_0, i0_1, i1_1, i2_1, i3_1, i4_0_1, i5_0, i6_0, i7_0, i8_0, i9_0_0, i0_2, i1_2, i2_2, i3_2, i4_0_2, i5_1, i6_1, i7_1, i8_1, i9_0_1, i0_3, i1_3, i2_3, i3_3, i4_0_3 in T.grid(1, 8, 28, 56, 1, 1, 2, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1):
            with Ts.sblock("conv2d_NCHWc_int8_o"):
                n = Ts.axis.spatial(1, i0_0 + i0_1 + i0_2 + i0_3)
                oc_chunk = Ts.axis.spatial(16, i1_0 * 2 + i1_1 + i1_2 + i1_3)
                oh = Ts.axis.spatial(56, i2_0 * 2 + i2_1 * 2 + i2_2 + i2_3)
                ow = Ts.axis.spatial(56, i3_0 + i3_1 + i3_2 + i3_3)
                oc_block_o = Ts.axis.spatial(1, i4_0_0 + i4_0_1 + i4_0_2 + i4_0_3)
                kh = Ts.axis.reduce(1, i5_0 + i5_1)
                kw = Ts.axis.reduce(1, i6_0 + i6_1)
                ic_outer = Ts.axis.reduce(4, i7_0 * 4 + i7_1)
                ic_f_inner = Ts.axis.reduce(4, i8_0 + i8_1)
                ic_s_inner_o = Ts.axis.reduce(1, i9_0_0 + i9_0_1)
                Ts.reads(placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4:ic_f_inner * 4 + 4], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4])
                Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16])
                Ts.sblock_attr({"meta_schedule.auto_tensorize": intrin})
                with Ts.init():
                    for i4_1 in range(16):
                        with Ts.sblock("conv2d_NCHWc_int8_init"):
                            oc_block_i_init = Ts.axis.spatial(16, i4_1)
                            Ts.reads()
                            Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init])
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i_init] = 0
                for i4_1, i9_1 in T.grid(16, 4):
                    with Ts.sblock("conv2d_NCHWc_int8"):
                        oc_block_i, ic_s_inner_i = Ts.axis.remap("SR", [i4_1, i9_1])
                        Ts.reads(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i], placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i], placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
                        Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i] = conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_i] + T.Cast("int32", placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner_i]) * T.Cast("int32", placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block_i, ic_s_inner_i])
    # fmt: on
    decision_0 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [8, 2, 1, 1]),
        ("SamplePerfectTile", [28, 1, 2, 1]),
        ("SamplePerfectTile", [56, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 4]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 1]),
    ]
    decision_1 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [8, 2, 1, 1]),
        ("SamplePerfectTile", [28, 1, 2, 1]),
        ("SamplePerfectTile", [56, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 4]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 1]),
    ]
    decision_2 = [
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [8, 2, 1, 1]),
        ("SamplePerfectTile", [28, 1, 2, 1]),
        ("SamplePerfectTile", [56, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1, 1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 1]),
        ("SamplePerfectTile", [1, 4]),
        ("SamplePerfectTile", [4, 1]),
        ("SamplePerfectTile", [1, 1]),
    ]

    mod = conv2d_nchwc
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target(target),
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                intrin,
                structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=ms.schedule_rule.ReuseType(req="may", levels=[1, 2], scope="global"),
            ),
        ],
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[x86_conv2d_nchwc_0, x86_conv2d_nchwc_1, x86_conv2d_nchwc_2],
        expected_decisions=[decision_0, decision_1, decision_2],
    )


def _check_dp4a_dense(m, n, k, in_dtype, out_dtype, expected_mods, expected_decisions):
    def _dense(m, n, k, in_dtype, out_dtype):
        X = te.placeholder((m, k), name="X", dtype=in_dtype)
        W = te.placeholder((n, k), name="W", dtype=in_dtype)
        ak = te.reduce_axis((0, k), name="k")
        matmul = te.compute(
            (m, n),
            lambda i, j: te.sum(
                X[i, ak].astype(out_dtype) * W[j, ak].astype(out_dtype),
                axis=ak,
            ),
            name="compute",
        )
        return te.create_function([X, W, matmul])

    mod = _dense(m, n, k, in_dtype, out_dtype)
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target({"kind": "cuda", "arch": "sm_70"}),
        types=None,
        sch_rules=[
            ms.schedule_rule.MultiLevelTilingWithIntrin(
                DP4A_S8S8S32_INTRIN,
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
                max_innermost_factor=64,
                vector_load_lens=[1, 2, 3, 4],
                reuse_read=ms.schedule_rule.ReuseType(req="must", levels=[4], scope="shared"),
                reuse_write=ms.schedule_rule.ReuseType(req="must", levels=[3], scope="local"),
            )
        ],
    )
    if expected_mods is None:
        assert expected_decisions is None
        assert len(actual) == 1
        assert_structural_equal(mod, actual[0].mod["main"])
    else:
        check_sketches(mod, actual, expected_mods, expected_decisions)


def test_dp4a_dense():
    @Ts.function
    def dp4a_dense_0(
        X: T.Tensor((128, 128), "int8"),
        W: T.Tensor((128, 128), "int8"),
        compute: T.Tensor((128, 128), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # with Ts.sblock("root"):
        compute_local = Ts.sblock_alloc_tensor((128, 128), "int32", scope="local")
        X_shared = Ts.sblock_alloc_tensor((128, 128), "int8", scope="shared")
        W_shared = Ts.sblock_alloc_tensor((128, 128), "int8", scope="shared")
        for i_0_j_0_fused in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for i_1_j_1_fused in T.parallel(512, annotations={"thread_binding": "vthread.x"}):
                for i_2_j_2_fused in T.parallel(2, annotations={"thread_binding": "threadIdx.x"}):
                    for k_0_0 in range(1):
                        for ax0_ax1_fused in range(16384):
                            with Ts.sblock("X_shared"):
                                v0 = Ts.axis.spatial(128, ax0_ax1_fused // 128)
                                v1 = Ts.axis.spatial(128, ax0_ax1_fused % 128)
                                Ts.reads(X[v0, v1])
                                Ts.writes(X_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                X_shared[v0, v1] = X[v0, v1]
                        for ax0_ax1_fused in range(16384):
                            with Ts.sblock("W_shared"):
                                v0 = Ts.axis.spatial(128, ax0_ax1_fused // 128)
                                v1 = Ts.axis.spatial(128, ax0_ax1_fused % 128)
                                Ts.reads(W[v0, v1])
                                Ts.writes(W_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                W_shared[v0, v1] = W[v0, v1]
                        for k_0_1, i_3, j_3, k_0_2, i_4, j_4 in T.grid(1, 2, 4, 32, 2, 1):
                            with Ts.sblock("compute_o"):
                                v_i = Ts.axis.spatial(
                                    128, i_1_j_1_fused // 32 * 8 + i_2_j_2_fused * 4 + i_3 * 2 + i_4
                                )
                                v_j = Ts.axis.spatial(128, i_1_j_1_fused % 32 * 4 + j_3 + j_4)
                                v_k_o = Ts.axis.reduce(32, k_0_0 * 32 + k_0_1 * 32 + k_0_2)
                                Ts.reads(
                                    X_shared[v_i, v_k_o * 4 : v_k_o * 4 + 4],
                                    W_shared[v_j, v_k_o * 4 : v_k_o * 4 + 4],
                                )
                                Ts.writes(compute_local[v_i, v_j])
                                Ts.sblock_attr({"meta_schedule.auto_tensorize": "dp4a_s8s8s32"})
                                with Ts.init():
                                    with Ts.sblock("compute_init"):
                                        Ts.reads()
                                        Ts.writes(compute_local[v_i, v_j])
                                        compute_local[v_i, v_j] = 0
                                for k_1 in range(4):
                                    with Ts.sblock("compute"):
                                        v_k_i = Ts.axis.reduce(4, k_1)
                                        Ts.reads(
                                            compute_local[v_i, v_j],
                                            X_shared[v_i, v_k_o * 4 + v_k_i],
                                            W_shared[v_j, v_k_o * 4 + v_k_i],
                                        )
                                        Ts.writes(compute_local[v_i, v_j])
                                        Ts.sblock_attr(
                                            {"meta_schedule.tiling_structure": "SSSRRSRS"}
                                        )
                                        compute_local[v_i, v_j] = compute_local[v_i, v_j] + T.Cast(
                                            "int32", X_shared[v_i, v_k_o * 4 + v_k_i]
                                        ) * T.Cast("int32", W_shared[v_j, v_k_o * 4 + v_k_i])
                    for ax0, ax1 in T.grid(4, 4):
                        with Ts.sblock("compute_local"):
                            v0 = Ts.axis.spatial(
                                128, i_1_j_1_fused // 32 * 8 + i_2_j_2_fused * 4 + ax0
                            )
                            v1 = Ts.axis.spatial(128, i_1_j_1_fused % 32 * 4 + ax1)
                            Ts.reads(compute_local[v0, v1])
                            Ts.writes(compute[v0, v1])
                            compute[v0, v1] = compute_local[v0, v1]

    decision_0 = [
        ("SamplePerfectTile", [1, 16, 2, 2, 2]),
        ("SamplePerfectTile", [1, 32, 1, 4, 1]),
        ("SamplePerfectTile", [1, 1, 32]),
        ("SampleCategorical", 0),
        ("SampleCategorical", 0),
    ]
    _check_dp4a_dense(
        m=128,
        n=128,
        k=128,
        in_dtype="int8",
        out_dtype="int32",
        expected_mods=[dp4a_dense_0],
        expected_decisions=[decision_0],
    )


def test_dp4a_dense_no_tensorize_1():
    _check_dp4a_dense(
        m=128,
        n=128,
        k=128,
        in_dtype="float32",
        out_dtype="float32",
        expected_mods=None,
        expected_decisions=None,
    )


def test_dp4a_dense_no_tensorize_2():
    _check_dp4a_dense(
        m=127,
        n=127,
        k=127,
        in_dtype="int8",
        out_dtype="int32",
        expected_mods=None,
        expected_decisions=None,
    )


if __name__ == "__main__":
    test_x86_conv2d_nchwc()
    test_x86_conv2d_nchwc(AVX512_INTRIN, {"kind": "llvm", "mcpu": "skylake-avx512", "num-cores": 4})
    test_dp4a_dense()
    test_dp4a_dense_no_tensorize_1()
    test_dp4a_dense_no_tensorize_2()
