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
"""Intrinsics for Hexagon tensorization."""

from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T

from .. import TensorIntrin


def generate_dma_load_intrin(
    size: int,
    dtype: str,
):
    """Generator of dma_load intrins"""

    sync_dma_load_desc_a_offset = T.dynamic("sync_dma_load_desc_a_offset", "int32")
    sync_dma_load_desc_c_offset = T.dynamic("sync_dma_load_desc_c_offset", "int32")

    @Ts.function
    def sync_dma_load_desc(
        A: T.Tensor(
            size,
            dtype,
            layout=T.TileLayout(T.S[size] + (sync_dma_load_desc_a_offset)),
            scope="global",
        ),
        C: T.Tensor(
            size,
            dtype,
            layout=T.TileLayout(T.S[size] + (sync_dma_load_desc_c_offset)),
            scope="global.vtcm",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:size])
            Ts.writes(C[0:size])
            for i in T.serial(size):
                with Ts.sblock("load"):
                    vii = Ts.axis.remap("S", [i])
                    C[vii] = A[vii]

    sync_dma_load_impl_a_offset = T.dynamic("sync_dma_load_impl_a_offset", "int32")
    sync_dma_load_impl_c_offset = T.dynamic("sync_dma_load_impl_c_offset", "int32")

    @Ts.function
    def sync_dma_load_impl(
        A: T.Tensor(
            size,
            dtype,
            layout=T.TileLayout(T.S[size] + (sync_dma_load_impl_a_offset)),
            scope="global",
        ),
        C: T.Tensor(
            size,
            dtype,
            layout=T.TileLayout(T.S[size] + (sync_dma_load_impl_c_offset)),
            scope="global.vtcm",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:size])
            Ts.writes(C[0:size])
            T.evaluate(
                T.call_packed(
                    "device_api.hexagon.dma_copy_dltensor",
                    T.stack_make_dltensor(
                        T.address_of(C[0]),
                        T.stack_make_shape(size),
                        0,
                        1,
                        C.dtype,
                        0,
                    ),
                    T.stack_make_dltensor(
                        T.address_of(A[0]),
                        T.stack_make_shape(size),
                        0,
                        1,
                        A.dtype,
                        0,
                    ),
                    T.cast(size, ty="int"),
                    False,  # Do not use experimental bypass mode.
                )
            )

    return sync_dma_load_desc, sync_dma_load_impl


def generate_dot_product_32x4_u8u8i32(mem_scope="global"):
    dot_product_32x4_u8u8i32_desc_a_offset = T.dynamic(
        "dot_product_32x4_u8u8i32_desc_a_offset", "int32"
    )
    dot_product_32x4_u8u8i32_desc_b_offset = T.dynamic(
        "dot_product_32x4_u8u8i32_desc_b_offset", "int32"
    )
    dot_product_32x4_u8u8i32_desc_c_offset = T.dynamic(
        "dot_product_32x4_u8u8i32_desc_c_offset", "int32"
    )

    @Ts.function
    def dot_product_32x4_u8u8i32_desc(
        A: T.Tensor(
            (4,),
            "uint8",
            layout=T.TileLayout(T.S[(4,)] + (dot_product_32x4_u8u8i32_desc_a_offset)),
            scope=mem_scope,
        ),
        B: T.Tensor(
            (32, 4),
            "uint8",
            layout=T.TileLayout(T.S[(32, 4)] + (dot_product_32x4_u8u8i32_desc_b_offset)),
            scope=mem_scope,
        ),
        C: T.Tensor(
            (32,),
            "int32",
            layout=T.TileLayout(T.S[(32,)] + (dot_product_32x4_u8u8i32_desc_c_offset)),
            scope=mem_scope,
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32], A[0:4], B[0:32, 0:4])
            Ts.writes(C[0:32])
            for i in T.serial(0, 32):
                for k in T.serial(0, 4):
                    with Ts.sblock("update"):
                        vi, vk = Ts.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], ty="int32") * T.cast(B[vi, vk], ty="int32")

    dot_product_32x4_u8u8i32_vrmpy_a_offset = T.dynamic(
        "dot_product_32x4_u8u8i32_vrmpy_a_offset", "int32"
    )
    dot_product_32x4_u8u8i32_vrmpy_b_offset = T.dynamic(
        "dot_product_32x4_u8u8i32_vrmpy_b_offset", "int32"
    )
    dot_product_32x4_u8u8i32_vrmpy_c_offset = T.dynamic(
        "dot_product_32x4_u8u8i32_vrmpy_c_offset", "int32"
    )

    @Ts.function
    def dot_product_32x4_u8u8i32_vrmpy(
        A: T.Tensor(
            (4,),
            "uint8",
            layout=T.TileLayout(T.S[(4,)] + (dot_product_32x4_u8u8i32_vrmpy_a_offset)),
            scope=mem_scope,
        ),
        B: T.Tensor(
            (32, 4),
            "uint8",
            layout=T.TileLayout(T.S[(32, 4)] + (dot_product_32x4_u8u8i32_vrmpy_b_offset)),
            scope=mem_scope,
        ),
        C: T.Tensor(
            (32,),
            "int32",
            layout=T.TileLayout(T.S[(32,)] + (dot_product_32x4_u8u8i32_vrmpy_c_offset)),
            scope=mem_scope,
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32], A[0:4], B[0:32, 0:4])
            Ts.writes(C[0:32])

            A_u8x4 = A[T.Ramp(0, 1, 4)]
            A_i32 = T.reinterpret(A_u8x4, ty="int32")

            B_i8x128 = B[0, T.Ramp(0, 1, 128)]
            B_i32x32 = T.reinterpret(B_i8x128, ty="int32x32")

            C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                C[T.ramp(T.int32(0), 1, 32)],
                B_i32x32,
                A_i32,
                ty="int32x32",
            )

    return dot_product_32x4_u8u8i32_desc, dot_product_32x4_u8u8i32_vrmpy


def generate_dot_product_32x4_u8i8i32(mem_scope="global"):
    dot_product_32x4_u8i8i32_desc_a_offset = T.dynamic(
        "dot_product_32x4_u8i8i32_desc_a_offset", "int32"
    )
    dot_product_32x4_u8i8i32_desc_b_offset = T.dynamic(
        "dot_product_32x4_u8i8i32_desc_b_offset", "int32"
    )
    dot_product_32x4_u8i8i32_desc_c_offset = T.dynamic(
        "dot_product_32x4_u8i8i32_desc_c_offset", "int32"
    )

    @Ts.function
    def dot_product_32x4_u8i8i32_desc(
        A: T.Tensor(
            (4,),
            "uint8",
            layout=T.TileLayout(T.S[(4,)] + (dot_product_32x4_u8i8i32_desc_a_offset)),
            scope=mem_scope,
        ),
        B: T.Tensor(
            (32, 4),
            "int8",
            layout=T.TileLayout(T.S[(32, 4)] + (dot_product_32x4_u8i8i32_desc_b_offset)),
            scope=mem_scope,
        ),
        C: T.Tensor(
            (32,),
            "int32",
            layout=T.TileLayout(T.S[(32,)] + (dot_product_32x4_u8i8i32_desc_c_offset)),
            scope=mem_scope,
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32], A[0:4], B[0:32, 0:4])
            Ts.writes(C[0:32])
            for i in T.serial(0, 32):
                for k in T.serial(0, 4):
                    with Ts.sblock("update"):
                        vi, vk = Ts.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], ty="int32") * T.cast(B[vi, vk], ty="int32")

    dot_product_32x4_u8i8i32_vrmpy_a_offset = T.dynamic(
        "dot_product_32x4_u8i8i32_vrmpy_a_offset", "int32"
    )
    dot_product_32x4_u8i8i32_vrmpy_b_offset = T.dynamic(
        "dot_product_32x4_u8i8i32_vrmpy_b_offset", "int32"
    )
    dot_product_32x4_u8i8i32_vrmpy_c_offset = T.dynamic(
        "dot_product_32x4_u8i8i32_vrmpy_c_offset", "int32"
    )

    @Ts.function
    def dot_product_32x4_u8i8i32_vrmpy(
        A: T.Tensor(
            (4,),
            "uint8",
            layout=T.TileLayout(T.S[(4,)] + (dot_product_32x4_u8i8i32_vrmpy_a_offset)),
            scope=mem_scope,
        ),
        B: T.Tensor(
            (32, 4),
            "int8",
            layout=T.TileLayout(T.S[(32, 4)] + (dot_product_32x4_u8i8i32_vrmpy_b_offset)),
            scope=mem_scope,
        ),
        C: T.Tensor(
            (32,),
            "int32",
            layout=T.TileLayout(T.S[(32,)] + (dot_product_32x4_u8i8i32_vrmpy_c_offset)),
            scope=mem_scope,
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32], A[0:4], B[0:32, 0:4])
            Ts.writes(C[0:32])

            A_u8x4 = A[T.Ramp(0, 1, 4)]
            A_i32 = T.reinterpret(A_u8x4, ty="int32")

            B_i8x128 = B[0, T.Ramp(0, 1, 128)]
            B_i32x32 = T.reinterpret(B_i8x128, ty="int32x32")

            C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpybusv.acc.128B"),
                C[T.ramp(T.int32(0), 1, 32)],
                T.broadcast(A_i32, 32),
                B_i32x32,
                ty="int32x32",
            )

    return dot_product_32x4_u8i8i32_desc, dot_product_32x4_u8i8i32_vrmpy


def generate_dot_product_32x2_i16i16i32(mem_scope="global"):
    dot_product_32x2_i16i16i32_desc_a_offset = T.dynamic(
        "dot_product_32x2_i16i16i32_desc_a_offset", "int32"
    )
    dot_product_32x2_i16i16i32_desc_b_offset = T.dynamic(
        "dot_product_32x2_i16i16i32_desc_b_offset", "int32"
    )
    dot_product_32x2_i16i16i32_desc_c_offset = T.dynamic(
        "dot_product_32x2_i16i16i32_desc_c_offset", "int32"
    )

    @Ts.function
    def dot_product_32x2_i16i16i32_desc(
        A: T.Tensor(
            (2,),
            "int16",
            layout=T.TileLayout(T.S[(2,)] + (dot_product_32x2_i16i16i32_desc_a_offset)),
            scope=mem_scope,
        ),
        B: T.Tensor(
            (32, 2),
            "int16",
            layout=T.TileLayout(T.S[(32, 2)] + (dot_product_32x2_i16i16i32_desc_b_offset)),
            scope=mem_scope,
        ),
        C: T.Tensor(
            (32,),
            "int32",
            layout=T.TileLayout(T.S[(32,)] + (dot_product_32x2_i16i16i32_desc_c_offset)),
            scope=mem_scope,
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32], A[0:2], B[0:32, 0:2])
            Ts.writes(C[0:32])
            for i in T.serial(0, 32):
                for k in T.serial(0, 2):
                    with Ts.sblock("update"):
                        vi, vk = Ts.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], ty="int32") * T.cast(B[vi, vk], ty="int32")

    dot_product_32x2_i16i16i32_vdmpy_a_offset = T.dynamic(
        "dot_product_32x2_i16i16i32_vdmpy_a_offset", "int32"
    )
    dot_product_32x2_i16i16i32_vdmpy_b_offset = T.dynamic(
        "dot_product_32x2_i16i16i32_vdmpy_b_offset", "int32"
    )
    dot_product_32x2_i16i16i32_vdmpy_c_offset = T.dynamic(
        "dot_product_32x2_i16i16i32_vdmpy_c_offset", "int32"
    )

    @Ts.function
    def dot_product_32x2_i16i16i32_vdmpy(
        A: T.Tensor(
            (2,),
            "int16",
            layout=T.TileLayout(T.S[(2,)] + (dot_product_32x2_i16i16i32_vdmpy_a_offset)),
            scope=mem_scope,
        ),
        B: T.Tensor(
            (32, 2),
            "int16",
            layout=T.TileLayout(T.S[(32, 2)] + (dot_product_32x2_i16i16i32_vdmpy_b_offset)),
            scope=mem_scope,
        ),
        C: T.Tensor(
            (32,),
            "int32",
            layout=T.TileLayout(T.S[(32,)] + (dot_product_32x2_i16i16i32_vdmpy_c_offset)),
            scope=mem_scope,
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32], A[0:2], B[0:32, 0:2])
            Ts.writes(C[0:32])

            A_i16x2 = A[T.Ramp(0, 1, 2)]
            A_i32 = T.reinterpret(A_i16x2, ty="int32")

            B_i16x64 = B[0, T.Ramp(0, 1, 64)]
            B_i32x32 = T.reinterpret(B_i16x64, ty="int32x32")

            C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vdmpyhvsat.acc.128B"),
                C[T.ramp(T.int32(0), 1, 32)],
                T.Broadcast(A_i32, 32),
                B_i32x32,
                ty="int32x32",
            )

    return dot_product_32x2_i16i16i32_desc, dot_product_32x2_i16i16i32_vdmpy


VRMPY_u8u8i32_INTRIN = "dot_32x4_u8u8i32_vrmpy"

TensorIntrin.register(VRMPY_u8u8i32_INTRIN, *generate_dot_product_32x4_u8u8i32())

VRMPY_u8i8i32_INTRIN = "dot_32x4_u8i8i32_vrmpy"

TensorIntrin.register(VRMPY_u8i8i32_INTRIN, *generate_dot_product_32x4_u8i8i32())

VDMPY_i16i16i32_INTRIN = "dot_product_32x2_i16i16i32_vdmpy"

TensorIntrin.register(VDMPY_i16i16i32_INTRIN, *generate_dot_product_32x2_i16i16i32())

VRMPY_u8u8i32_VTCM_INTRIN = "dot_32x4_u8u8i32_vtcm_vrmpy"
TensorIntrin.register(VRMPY_u8u8i32_VTCM_INTRIN, *generate_dot_product_32x4_u8u8i32("global.vtcm"))

VRMPY_u8i8i32_VTCM_INTRIN = "dot_32x4_u8i8i32_vtcm_vrmpy"
TensorIntrin.register(VRMPY_u8i8i32_VTCM_INTRIN, *generate_dot_product_32x4_u8i8i32("global.vtcm"))

DMA_READ_128_u8 = "dma_read_128_u8"
TensorIntrin.register(DMA_READ_128_u8, *generate_dma_load_intrin(128, "uint8"))

DMA_READ_128_i8 = "dma_read_128_i8"
TensorIntrin.register(DMA_READ_128_i8, *generate_dma_load_intrin(128, "int8"))
