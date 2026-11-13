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
# pylint: disable=invalid-name,missing-function-docstring
"""Intrinsics for x86 tensorization."""

from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T

from .. import TensorIntrin

# Tensorized intrinsic description and VNNI-specific implementation.
# Equivalent to the ones in topi/x86/tensor_intrin.py


dot_product_16x4_u8i8i32_desc_a_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_desc_a_offset", "int32"
)
dot_product_16x4_u8i8i32_desc_b_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_desc_b_offset", "int32"
)
dot_product_16x4_u8i8i32_desc_c_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_desc_c_offset", "int32"
)


@Ts.function
def dot_product_16x4_u8i8i32_desc(
    A: T.Tensor(
        (4,), "uint8", layout=T.TileLayout(T.S[(4,)] + (dot_product_16x4_u8i8i32_desc_a_offset))
    ),
    B: T.Tensor(
        (16, 4),
        "int8",
        layout=T.TileLayout(T.S[(16, 4)] + (dot_product_16x4_u8i8i32_desc_b_offset)),
    ),
    C: T.Tensor(
        (16,), "int32", layout=T.TileLayout(T.S[(16,)] + (dot_product_16x4_u8i8i32_desc_c_offset))
    ),
) -> None:
    with Ts.sblock("root"):
        Ts.reads(C[0:16], A[0:4], B[0:16, 0:4])
        Ts.writes(C[0:16])
        for i in T.serial(0, 16):
            for k in T.serial(0, 4):
                with Ts.sblock("update"):
                    vi, vk = Ts.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], ty="int32") * T.cast(B[vi, vk], ty="int32")


dot_product_16x4_u8i8i32_vnni_a_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_vnni_a_offset", "int32"
)
dot_product_16x4_u8i8i32_vnni_b_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_vnni_b_offset", "int32"
)
dot_product_16x4_u8i8i32_vnni_c_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_vnni_c_offset", "int32"
)


@Ts.function
def dot_product_16x4_u8i8i32_vnni(
    A: T.Tensor(
        (4,), "uint8", layout=T.TileLayout(T.S[(4,)] + (dot_product_16x4_u8i8i32_vnni_a_offset))
    ),
    B: T.Tensor(
        (16, 4),
        "int8",
        layout=T.TileLayout(T.S[(16, 4)] + (dot_product_16x4_u8i8i32_vnni_b_offset)),
    ),
    C: T.Tensor(
        (16,), "int32", layout=T.TileLayout(T.S[(16,)] + (dot_product_16x4_u8i8i32_vnni_c_offset))
    ),
) -> None:
    with Ts.sblock("root"):
        Ts.reads(C[0:16], A[0:4], B[0:16, 0:4])
        Ts.writes(C[0:16])

        A_u8x4: T.uint8x4 = A[T.Ramp(0, 1, 4)]
        A_i32: T.int32 = T.reinterpret(A_u8x4, ty="int32")

        B_i8x64: T.int8x64 = B[0, T.Ramp(0, 1, 64)]
        B_i32x16: T.int32x16 = T.reinterpret(B_i8x64, ty="int32x16")
        C_i32x16: T.int32x16 = C[T.Ramp(0, 1, 16)]

        C[T.ramp(T.int32(0), 1, 16)] = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512"),
            C_i32x16,
            T.broadcast(A_i32, 16),
            B_i32x16,
            ty="int32x16",
        )


dot_product_16x4_u8i8i32_avx512_a_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_avx512_a_offset", "int32"
)
dot_product_16x4_u8i8i32_avx512_b_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_avx512_b_offset", "int32"
)
dot_product_16x4_u8i8i32_avx512_c_offset = T.dynamic(
    "dot_product_16x4_u8i8i32_avx512_c_offset", "int32"
)


@Ts.function
def dot_product_16x4_u8i8i32_avx512(
    A: T.Tensor(
        (4,), "uint8", layout=T.TileLayout(T.S[(4,)] + (dot_product_16x4_u8i8i32_avx512_a_offset))
    ),
    B: T.Tensor(
        (16, 4),
        "int8",
        layout=T.TileLayout(T.S[(16, 4)] + (dot_product_16x4_u8i8i32_avx512_b_offset)),
    ),
    C: T.Tensor(
        (16,), "int32", layout=T.TileLayout(T.S[(16,)] + (dot_product_16x4_u8i8i32_avx512_c_offset))
    ),
) -> None:
    with Ts.sblock("root"):
        Ts.reads(C[0:16], A[0:4], B[0:16, 0:4])
        Ts.writes(C[0:16])

        A_u8x4 = A[T.Ramp(0, 1, 4)]
        A_i32 = T.reinterpret(A_u8x4, ty="int32")
        A_brdcst = T.broadcast(A_i32, 16)
        A_u8x64 = T.reinterpret(A_brdcst, ty="uint8x64")

        B_i8x64 = B[0, T.Ramp(0, 1, 64)]

        Red = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.x86.avx512.pmaddubs.w.512"),
            A_u8x64,
            B_i8x64,
            ty="int16x32",
        )

        C[T.ramp(T.int32(0), 1, 16)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.x86.avx512.pmaddw.d.512"),
            Red,
            T.int16x32(1),
            ty="int32x16",
        )


VNNI_DOT_16x4_INTRIN = "dot_16x4_vnni"

TensorIntrin.register(
    VNNI_DOT_16x4_INTRIN, dot_product_16x4_u8i8i32_desc, dot_product_16x4_u8i8i32_vnni
)

AVX512_DOT_16x4_INTRIN = "dot_16x4_avx512"

TensorIntrin.register(
    AVX512_DOT_16x4_INTRIN, dot_product_16x4_u8i8i32_desc, dot_product_16x4_u8i8i32_avx512
)
