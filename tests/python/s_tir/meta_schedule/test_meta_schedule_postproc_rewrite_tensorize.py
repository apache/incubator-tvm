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
# ruff: noqa: F401, F841
import tvm
import tvm_tirx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.tensor_intrin import dot_product_common, x86
from tvm_tirx.script import tirx as T


@tvm_tirx.script.ir_module
class Conv2dNCHWcVNNIModuleTiled:
    @Ts.function
    def main(
        placeholder: T.Tensor((1, 4, 56, 56, 16), "uint8"),
        placeholder_1: T.Tensor((16, 4, 1, 1, 4, 16, 4), "int8"),
        conv2d_NCHWc_int8: T.Tensor((1, 16, 56, 56, 16), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for (
            i0_0,
            i1_0,
            i2_0,
            i3_0,
            i4_0_0,
            i0_1,
            i1_1,
            i2_1,
            i3_1,
            i4_0_1,
            i5_0,
            i6_0,
            i7_0,
            i8_0,
            i9_0_0,
            i0_2,
            i1_2,
            i2_2,
            i3_2,
            i4_0_2,
            i5_1,
            i6_1,
            i7_1,
            i8_1,
            i9_0_1,
            i0_3,
            i1_3,
            i2_3,
            i3_3,
            i4_0_3,
        ) in T.grid(
            1,
            1,
            2,
            1,
            1,
            1,
            4,
            1,
            14,
            1,
            1,
            1,
            4,
            1,
            1,
            1,
            4,
            7,
            1,
            1,
            1,
            1,
            1,
            4,
            1,
            1,
            1,
            4,
            4,
            1,
        ):
            with Ts.sblock("conv2d_NCHWc_int8_o"):
                n = Ts.axis.spatial(1, 0)
                oc_chunk = Ts.axis.spatial(16, i1_1 * 4 + i1_2)
                oh = Ts.axis.spatial(56, i2_0 * 28 + i2_2 * 4 + i2_3)
                ow = Ts.axis.spatial(56, i3_1 * 4 + i3_3)
                oc_block_o = Ts.axis.spatial(1, 0)
                kh = Ts.axis.reduce(1, 0)
                kw = Ts.axis.reduce(1, 0)
                ic_outer, ic_f_inner = Ts.axis.remap("RR", [i7_0, i8_1])
                ic_s_inner_o = Ts.axis.reduce(1, 0)
                Ts.reads(
                    placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4],
                    placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4],
                )
                Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16])
                Ts.sblock_attr({"meta_schedule.auto_tensorize": "dot_16x4_vnni"})
                with Ts.init():
                    for i4_1 in T.serial(16):
                        with Ts.sblock("conv2d_NCHWc_int8_init"):
                            oc_block_init = Ts.axis.spatial(16, i4_1)
                            Ts.reads()
                            Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_init])
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_init] = 0
                for i4_1, i9_1 in T.grid(16, 4):
                    with Ts.sblock("conv2d_NCHWc_int8"):
                        oc_block, ic_s_inner = Ts.axis.remap("SR", [i4_1, i9_1])
                        Ts.reads(
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block],
                            placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                            placeholder_1[
                                oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner
                            ],
                        )
                        Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block])
                        Ts.sblock_attr({"meta_schedule.tiling_structure": "SSRSRS"})
                        conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block] = conv2d_NCHWc_int8[
                            n, oc_chunk, oh, ow, oc_block
                        ] + T.cast(
                            placeholder[n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 + ic_s_inner],
                            ty="int32",
                        ) * T.cast(
                            placeholder_1[
                                oc_chunk, ic_outer, kh, kw, ic_f_inner, oc_block, ic_s_inner
                            ],
                            ty="int32",
                        )


conv2dnchwcvnnimoduletensorized_a_offset = T.dynamic(
    "conv2dnchwcvnnimoduletensorized_a_offset", "int32"
)
conv2dnchwcvnnimoduletensorized_b_offset = T.dynamic(
    "conv2dnchwcvnnimoduletensorized_b_offset", "int32"
)
conv2dnchwcvnnimoduletensorized_c_offset = T.dynamic(
    "conv2dnchwcvnnimoduletensorized_c_offset", "int32"
)


@tvm_tirx.script.ir_module
class Conv2dNCHWcVNNIModuleTensorized:
    @Ts.function
    def main(
        placeholder: T.Tensor((1, 4, 56, 56, 16), "uint8"),
        placeholder_1: T.Tensor((16, 4, 1, 1, 4, 16, 4), "int8"),
        conv2d_NCHWc_int8: T.Tensor((1, 16, 56, 56, 16), "int32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for i0_0, i1_0, i2_0, i3_0, i4_0_0, i0_1, i1_1, i2_1, i3_1, i4_0_1, i5_0, i6_0 in T.grid(
            1, 1, 2, 1, 1, 1, 4, 1, 14, 1, 1, 1
        ):
            for i1_2_init, i2_2_init, i2_3_init, i3_3_init in T.grid(4, 7, 4, 4):
                with Ts.sblock("conv2d_NCHWc_int8_o_init"):
                    n = Ts.axis.spatial(1, 0)
                    oc_chunk = Ts.axis.spatial(16, i1_1 * 4 + i1_2_init)
                    oh = Ts.axis.spatial(56, i2_0 * 28 + i2_2_init * 4 + i2_3_init)
                    ow = Ts.axis.spatial(56, i3_1 * 4 + i3_3_init)
                    oc_block_o = Ts.axis.spatial(1, 0)
                    Ts.reads()
                    Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16])
                    for i4_1 in T.vectorized(16):
                        with Ts.sblock("conv2d_NCHWc_int8_init"):
                            oc_block_init = Ts.axis.spatial(16, i4_1)
                            Ts.reads()
                            Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_init])
                            conv2d_NCHWc_int8[n, oc_chunk, oh, ow, oc_block_init] = 0
            for (
                i7_0,
                i8_0,
                i9_0_0,
                i0_2,
                i1_2,
                i2_2,
                i3_2,
                i4_0_2,
                i5_1,
                i6_1,
                i7_1,
                i8_1,
                i9_0_1,
                i0_3,
                i1_3,
                i2_3,
                i3_3,
                i4_0_3,
            ) in T.grid(4, 1, 1, 1, 4, 7, 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 4, 1):
                with Ts.sblock("conv2d_NCHWc_int8_o_update"):
                    n = Ts.axis.spatial(1, 0)
                    oc_chunk = Ts.axis.spatial(16, i1_1 * 4 + i1_2)
                    oh = Ts.axis.spatial(56, i2_0 * 28 + i2_2 * 4 + i2_3)
                    ow = Ts.axis.spatial(56, i3_1 * 4 + i3_3)
                    oc_block_o = Ts.axis.spatial(1, 0)
                    kh = Ts.axis.reduce(1, 0)
                    kw = Ts.axis.reduce(1, 0)
                    ic_outer, ic_f_inner = Ts.axis.remap("RR", [i7_0, i8_1])
                    ic_s_inner_o = Ts.axis.reduce(1, 0)
                    Ts.reads(
                        conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16],
                        placeholder[
                            n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4
                        ],
                        placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4],
                    )
                    Ts.writes(conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16])
                    A = Ts.match_tensor(
                        placeholder[
                            n, ic_outer, oh + kh, ow + kw, ic_f_inner * 4 : ic_f_inner * 4 + 4
                        ],
                        [4],
                        dtype="uint8",
                        layout=T.TileLayout(T.S[[4]] + (conv2dnchwcvnnimoduletensorized_a_offset)),
                    )
                    B = Ts.match_tensor(
                        placeholder_1[oc_chunk, ic_outer, kh, kw, ic_f_inner, 0:16, 0:4],
                        [16, 4],
                        dtype="int8",
                        layout=T.TileLayout(
                            T.S[[16, 4]] + (conv2dnchwcvnnimoduletensorized_b_offset)
                        ),
                    )
                    C = Ts.match_tensor(
                        conv2d_NCHWc_int8[n, oc_chunk, oh, ow, 0:16],
                        [16],
                        dtype="int32",
                        layout=T.TileLayout(T.S[[16]] + (conv2dnchwcvnnimoduletensorized_c_offset)),
                    )
                    A_u8x4 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.uint8x4, scope="local")])
                    A_u8x4[0] = A[T.Ramp(0, 1, 4)]
                    A_i32 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32, scope="local")])
                    A_i32[0] = T.reinterpret(A_u8x4[0], ty="int32")
                    B_i8x64 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int8x64, scope="local")])
                    B_i8x64[0] = B[0, T.Ramp(0, 1, 64)]
                    B_i32x16 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32x16, scope="local")])
                    B_i32x16[0] = T.reinterpret(B_i8x64[0], ty="int32x16")
                    C_i32x16 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32x16, scope="local")])
                    C_i32x16[0] = C[T.Ramp(0, 1, 16)]
                    C[T.ramp(0, 1, 16)] = T.call_llvm_pure_intrin(
                        T.llvm_lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512"),
                        C_i32x16[0],
                        T.broadcast(A_i32[0], 16),
                        B_i32x16[0],
                        ty="int32x16",
                    )


@tvm_tirx.script.ir_module
class DenseDP4ATiled:
    @Ts.function
    def main(
        X: T.Tensor((128, 128), "int8"),
        W: T.Tensor((128, 128), "int8"),
        compute: T.Tensor((128, 128), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        compute_local = Ts.sblock_alloc_tensor([128, 128], dtype="int32", scope="local")
        X_shared = Ts.sblock_alloc_tensor([128, 128], dtype="int8", scope="shared")
        W_shared = Ts.sblock_alloc_tensor([128, 128], dtype="int8", scope="shared")
        for i0_0_i1_0_fused in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
            for i0_1_i1_1_fused in T.parallel(2, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_fused in T.parallel(2, annotations={"thread_binding": "threadIdx.x"}):
                    for i2_0_0 in T.serial(2):
                        for ax0_ax1_fused in T.serial(1024):
                            with Ts.sblock("X_shared"):
                                v0 = Ts.axis.spatial(
                                    128, i0_0_i1_0_fused // 2 * 16 + ax0_ax1_fused // 64
                                )
                                v1 = Ts.axis.spatial(128, i2_0_0 * 64 + ax0_ax1_fused % 64)
                                Ts.reads(X[v0, v1])
                                Ts.writes(X_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                X_shared[v0, v1] = X[v0, v1]
                        for ax0_ax1_fused in T.serial(4096):
                            with Ts.sblock("W_shared"):
                                v0 = Ts.axis.spatial(
                                    128, i0_0_i1_0_fused % 2 * 64 + ax0_ax1_fused // 64
                                )
                                v1 = Ts.axis.spatial(128, i2_0_0 * 64 + ax0_ax1_fused % 64)
                                Ts.reads(W[v0, v1])
                                Ts.writes(W_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                W_shared[v0, v1] = W[v0, v1]
                        for i2_0_1, i0_3, i1_3, i2_0_2, i0_4, i1_4 in T.grid(2, 4, 16, 8, 4, 1):
                            with Ts.sblock("compute_o"):
                                i = Ts.axis.spatial(
                                    128, i0_0_i1_0_fused // 2 * 16 + i0_3 * 4 + i0_4
                                )
                                j = Ts.axis.spatial(
                                    128,
                                    i0_0_i1_0_fused % 2 * 64
                                    + i0_1_i1_1_fused * 32
                                    + i0_2_i1_2_fused * 16
                                    + i1_3,
                                )
                                k_o = Ts.axis.reduce(32, i2_0_0 * 16 + i2_0_1 * 8 + i2_0_2)
                                Ts.reads(
                                    X_shared[i, k_o * 4 : k_o * 4 + 4],
                                    W_shared[j, k_o * 4 : k_o * 4 + 4],
                                )
                                Ts.writes(compute_local[i, j])
                                Ts.sblock_attr({"meta_schedule.auto_tensorize": "dp4a_s8s8s32"})
                                with Ts.init():
                                    with Ts.sblock("compute_init"):
                                        Ts.reads()
                                        Ts.writes(compute_local[i, j])
                                        compute_local[i, j] = 0
                                for i2_1 in T.serial(4):
                                    with Ts.sblock("compute"):
                                        k = Ts.axis.reduce(4, i2_1)
                                        Ts.reads(
                                            compute_local[i, j],
                                            X_shared[i, k_o * 4 + k],
                                            W_shared[j, k_o * 4 + k],
                                        )
                                        Ts.writes(compute_local[i, j])
                                        Ts.sblock_attr(
                                            {"meta_schedule.tiling_structure": "SSSRRSRS"}
                                        )
                                        compute_local[i, j] = compute_local[i, j] + T.cast(
                                            X_shared[i, k_o * 4 + k], ty="int32"
                                        ) * T.cast(W_shared[j, k_o * 4 + k], ty="int32")
                    for ax0, ax1 in T.grid(16, 16):
                        with Ts.sblock("compute_local"):
                            v0 = Ts.axis.spatial(128, i0_0_i1_0_fused // 2 * 16 + ax0)
                            v1 = Ts.axis.spatial(
                                128,
                                i0_0_i1_0_fused % 2 * 64
                                + i0_1_i1_1_fused * 32
                                + i0_2_i1_2_fused * 16
                                + ax1,
                            )
                            Ts.reads(compute_local[v0, v1])
                            Ts.writes(compute[v0, v1])
                            compute[v0, v1] = compute_local[v0, v1]


densedp4atensorized_a_offset = T.dynamic("densedp4atensorized_a_offset", "int32")
densedp4atensorized_b_offset = T.dynamic("densedp4atensorized_b_offset", "int32")
densedp4atensorized_c_offset = T.dynamic("densedp4atensorized_c_offset", "int32")


@tvm_tirx.script.ir_module
class DenseDP4ATensorized:
    @Ts.function
    def main(
        X: T.Tensor((128, 128), "int8"),
        W: T.Tensor((128, 128), "int8"),
        compute: T.Tensor((128, 128), "int32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        compute_local = Ts.sblock_alloc_tensor([128, 128], dtype="int32", scope="local")
        X_shared = Ts.sblock_alloc_tensor([128, 128], dtype="int8", scope="shared")
        W_shared = Ts.sblock_alloc_tensor([128, 128], dtype="int8", scope="shared")
        for i0_0_i1_0_fused in T.parallel(16, annotations={"thread_binding": "blockIdx.x"}):
            for i0_1_i1_1_fused in T.parallel(2, annotations={"thread_binding": "vthread.x"}):
                for i0_2_i1_2_fused in T.parallel(2, annotations={"thread_binding": "threadIdx.x"}):
                    for i0_3_init, i1_3_init, i0_4_init in T.grid(4, 16, 4):
                        with Ts.sblock("compute_o_init"):
                            i = Ts.axis.spatial(
                                128, i0_0_i1_0_fused // 2 * 16 + i0_3_init * 4 + i0_4_init
                            )
                            j = Ts.axis.spatial(
                                128,
                                i0_0_i1_0_fused % 2 * 64
                                + i0_1_i1_1_fused * 32
                                + i0_2_i1_2_fused * 16
                                + i1_3_init,
                            )
                            Ts.reads()
                            Ts.writes(compute_local[i, j])
                            Ts.sblock_attr({"meta_schedule.auto_tensorize": ""})
                            with Ts.sblock("compute_init"):
                                Ts.reads()
                                Ts.writes(compute_local[i, j])
                                compute_local[i, j] = 0
                    for i2_0_0 in T.serial(2):
                        for ax0_ax1_fused in T.serial(1024):
                            with Ts.sblock("X_shared"):
                                v0 = Ts.axis.spatial(
                                    128, i0_0_i1_0_fused // 2 * 16 + ax0_ax1_fused // 64
                                )
                                v1 = Ts.axis.spatial(128, i2_0_0 * 64 + ax0_ax1_fused % 64)
                                Ts.reads(X[v0, v1])
                                Ts.writes(X_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 4})
                                X_shared[v0, v1] = X[v0, v1]
                        for ax0_ax1_fused in T.serial(4096):
                            with Ts.sblock("W_shared"):
                                v0 = Ts.axis.spatial(
                                    128, i0_0_i1_0_fused % 2 * 64 + ax0_ax1_fused // 64
                                )
                                v1 = Ts.axis.spatial(128, i2_0_0 * 64 + ax0_ax1_fused % 64)
                                Ts.reads(W[v0, v1])
                                Ts.writes(W_shared[v0, v1])
                                Ts.sblock_attr({"meta_schedule.cooperative_fetch": 1})
                                W_shared[v0, v1] = W[v0, v1]
                        for i2_0_1, i0_3, i1_3, i2_0_2, i0_4, i1_4 in T.grid(2, 4, 16, 8, 4, 1):
                            with Ts.sblock("compute_o_update"):
                                i = Ts.axis.spatial(
                                    128, i0_0_i1_0_fused // 2 * 16 + i0_3 * 4 + i0_4
                                )
                                j = Ts.axis.spatial(
                                    128,
                                    i0_0_i1_0_fused % 2 * 64
                                    + i0_1_i1_1_fused * 32
                                    + i0_2_i1_2_fused * 16
                                    + i1_3,
                                )
                                k_o = Ts.axis.reduce(32, i2_0_0 * 16 + i2_0_1 * 8 + i2_0_2)
                                Ts.reads(
                                    compute_local[i, j],
                                    X_shared[i, k_o * 4 : k_o * 4 + 4],
                                    W_shared[j, k_o * 4 : k_o * 4 + 4],
                                )
                                Ts.writes(compute_local[i, j])
                                A = Ts.match_tensor(
                                    X_shared[i, k_o * 4 : k_o * 4 + 4],
                                    [4],
                                    dtype="int8",
                                    scope="shared",
                                    align=4,
                                    layout=T.TileLayout(T.S[[4]] + (densedp4atensorized_a_offset)),
                                )
                                B = Ts.match_tensor(
                                    W_shared[j, k_o * 4 : k_o * 4 + 4],
                                    [4],
                                    dtype="int8",
                                    scope="shared",
                                    align=4,
                                    layout=T.TileLayout(T.S[[4]] + (densedp4atensorized_b_offset)),
                                )
                                C = Ts.match_tensor(
                                    compute_local[i, j],
                                    [1],
                                    dtype="int32",
                                    scope="local",
                                    align=4,
                                    layout=T.TileLayout(T.S[[1]] + (densedp4atensorized_c_offset)),
                                )
                                C[0] = C[0] + T.call_pure_extern(
                                    "__dp4a", A[T.ramp(0, 1, 4)], B[T.ramp(0, 1, 4)], 0, ty="int32"
                                )
                    for ax0, ax1 in T.grid(16, 16):
                        with Ts.sblock("compute_local"):
                            v0 = Ts.axis.spatial(128, i0_0_i1_0_fused // 2 * 16 + ax0)
                            v1 = Ts.axis.spatial(
                                128,
                                i0_0_i1_0_fused % 2 * 64
                                + i0_1_i1_1_fused * 32
                                + i0_2_i1_2_fused * 16
                                + ax1,
                            )
                            Ts.reads(compute_local[v0, v1])
                            Ts.writes(compute[v0, v1])
                            compute[v0, v1] = compute_local[v0, v1]


def _create_context(mod, target, postprocs) -> ms.TuneContext:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=postprocs,
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


def test_rewrite_tensorize_conv2d_nchwc_vnni():
    mod = Conv2dNCHWcVNNIModuleTiled
    target = tvm_tirx.target.Target({"kind": "llvm", "mcpu": "cascadelake", "num-cores": 4})
    ctx = _create_context(
        mod,
        target,
        [
            ms.postproc.RewriteReductionBlock(),
            ms.postproc.RewriteTensorize(True),
        ],
    )
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()

    for proc in ctx.space_generator.postprocs:
        proc.apply(sch)

    tvm_tirx.ir.assert_structural_equal(sch.mod, Conv2dNCHWcVNNIModuleTensorized)


def test_rewrite_tensorize_dense_dp4a():
    mod = DenseDP4ATiled
    target = tvm_tirx.target.Target("nvidia/geforce-rtx-3070")
    ctx = _create_context(
        mod,
        target,
        [
            ms.postproc.RewriteCooperativeFetch(),
            ms.postproc.RewriteReductionBlock(),
            ms.postproc.RewriteTensorize(),
        ],
    )
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()

    for proc in ctx.space_generator.postprocs:
        proc.apply(sch)

    tvm_tirx.ir.assert_structural_equal(sch.mod, DenseDP4ATensorized)


if __name__ == "__main__":
    test_rewrite_tensorize_conv2d_nchwc_vnni()
    test_rewrite_tensorize_dense_dp4a()
