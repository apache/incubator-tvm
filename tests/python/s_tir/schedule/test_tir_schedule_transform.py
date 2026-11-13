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
import tvm_tirx
from tvm.s_tir import Schedule
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.transform import tile_with_tensor_intrin
from tvm.s_tir.tensor_intrin.x86 import AVX512_DOT_16x4_INTRIN, VNNI_DOT_16x4_INTRIN
from tvm_tirx.script import tirx as T


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
class DenseTIRModuleTiled:
    @Ts.function
    def main(
        placeholder: T.Tensor((1024, 1024), "uint8"),
        placeholder_1: T.Tensor((64, 256, 16, 4), "int8"),
        compute: T.Tensor((1024, 1024), "int32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for i0, i1_0, i2_0, i1_1, i2_1 in T.grid(1024, 64, 256, 16, 4):
            with Ts.sblock("compute"):
                i = Ts.axis.spatial(1024, i0)
                j = Ts.axis.spatial(1024, i1_0 * 16 + i1_1)
                k = Ts.axis.reduce(1024, i2_0 * 4 + i2_1)
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


@tvm_tirx.script.ir_module
class Conv2dNCHWcTIRModuleTiled:
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
        for i0, i1, i2, i3, i4_0, i5, i6, i7, i8, i9_0, i4_1, i9_1 in T.grid(
            1, 16, 56, 56, 1, 1, 1, 4, 4, 1, 16, 4
        ):
            with Ts.sblock("conv2d_NCHWc_int8"):
                n, oc_chunk, oh, ow = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                oc_block = Ts.axis.spatial(16, i4_0 * 16 + i4_1)
                kh, kw, ic_outer, ic_f_inner = Ts.axis.remap("RRRR", [i5, i6, i7, i8])
                ic_s_inner = Ts.axis.reduce(4, i9_0 * 4 + i9_1)
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


def test_tile_with_tensor_intrin_dense(intrin=VNNI_DOT_16x4_INTRIN):
    s = Schedule(DenseTIRModule)
    block = s.get_sblock("compute")

    tiled_loop = tile_with_tensor_intrin(s, block, intrin)

    _, _, _, i1_1, _ = s.get_loops(block)

    assert s.get(tiled_loop) == s.get(i1_1)
    tvm_tirx.ir.assert_structural_equal(s.mod, DenseTIRModuleTiled)


def test_tile_with_tensor_intrin_conv2d_nchwc(intrin=VNNI_DOT_16x4_INTRIN):
    s = Schedule(Conv2dNCHWcTIRModule)
    block = s.get_sblock("conv2d_NCHWc_int8")
    tiled_loop = tile_with_tensor_intrin(s, block, intrin)
    tiled_loops = s.get_loops(block)
    assert len(tiled_loops) == 12
    assert s.get(tiled_loop) == s.get(tiled_loops[-2])
    tvm_tirx.ir.assert_structural_equal(s.mod, Conv2dNCHWcTIRModuleTiled)


if __name__ == "__main__":
    test_tile_with_tensor_intrin_dense()
    test_tile_with_tensor_intrin_dense(AVX512_DOT_16x4_INTRIN)
    test_tile_with_tensor_intrin_conv2d_nchwc()
    test_tile_with_tensor_intrin_conv2d_nchwc(AVX512_DOT_16x4_INTRIN)
