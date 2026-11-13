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
# ruff: noqa: E501

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir, te
from tvm.s_tir import match_shape_stride_pattern
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)
from tvm.s_tir.tensor_intrin.arm_cpu import (
    DP4A_S8S8S32_INTRIN,
    DP4A_S8U8S32_INTRIN,
    DP4A_U8S8S32_INTRIN,
    DP4A_U8U8U32_INTRIN,
    ARM_DOT_4x4_i8_NEON_INTRIN,
    ARM_DOT_4x4_i8_SDOT_INTRIN,
)
from tvm.s_tir.tensor_intrin.hexagon import VDMPY_i16i16i32_INTRIN, VRMPY_u8u8i32_INTRIN
from tvm.s_tir.tensor_intrin.rocm import AMDGPU_SDOT4_INTRIN
from tvm.s_tir.tensor_intrin.x86 import AVX512_DOT_16x4_INTRIN, VNNI_DOT_16x4_INTRIN
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

mma_desc_a_offset = T.dynamic("mma_desc_a_offset", "int32")
mma_desc_b_offset = T.dynamic("mma_desc_b_offset", "int32")
mma_desc_c_offset = T.dynamic("mma_desc_c_offset", "int32")

@Ts.function
def mma_desc(A: T.Tensor((16, 16), "float32", align=64,  layout=T.TileLayout(T.S[(16, 16)] + (mma_desc_a_offset))), B: T.Tensor((16, 16), "float32", align=64,  layout=T.TileLayout(T.S[(16, 16)] + (mma_desc_b_offset))), C: T.Tensor((16, 16), "float32", align=64,  layout=T.TileLayout(T.S[(16, 16)] + (mma_desc_c_offset)))) -> None:

    with Ts.sblock("root"):
        Ts.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        Ts.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with Ts.sblock("update"):
                vii, vjj, vkk = Ts.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]

mma_intrin_a_stride = T.dynamic("mma_intrin_a_stride", "int32")
mma_intrin_a_offset = T.dynamic("mma_intrin_a_offset", "int32")
mma_intrin_b_stride = T.dynamic("mma_intrin_b_stride", "int32")
mma_intrin_b_offset = T.dynamic("mma_intrin_b_offset", "int32")
mma_intrin_c_stride = T.dynamic("mma_intrin_c_stride", "int32")
mma_intrin_c_offset = T.dynamic("mma_intrin_c_offset", "int32")

@Ts.function
def mma_intrin(A: T.Tensor((16, 16), "float32", layout=T.TileLayout(T.S[(16, 16) : [mma_intrin_a_stride, 1]] + (mma_intrin_a_offset)), align=64), B: T.Tensor((16, 16), "float32", layout=T.TileLayout(T.S[(16, 16) : [mma_intrin_b_stride, 1]] + (mma_intrin_b_offset)), align=64), C: T.Tensor((16, 16), "float32", layout=T.TileLayout(T.S[(16, 16) : [mma_intrin_c_stride, 1]] + (mma_intrin_c_offset)), align=64)) -> None:

    with Ts.sblock("root"):
        Ts.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        Ts.writes(C[0 : 16, 0 : 16])
        A_pattern = T.meta_var(match_shape_stride_pattern(A.ty))
        B_pattern = T.meta_var(match_shape_stride_pattern(B.ty))
        C_pattern = T.meta_var(match_shape_stride_pattern(C.ty))
        T.evaluate(
            T.gpu_mma_sync(
                C.data,
                C_pattern.elem_offset // 256,
                A.data,
                A_pattern.elem_offset // 256,
                B.data,
                B_pattern.elem_offset // 256,
                C.data,
                C_pattern.elem_offset // 256,

            )
        )

@Ts.function
def dot_product_desc(A: T.Tensor((4,), "float32"), B: T.Tensor((4,), "float32"), C: T.Tensor((), "float32")) -> None:

    with Ts.sblock("root"):
        Ts.reads(C[()], A[0 : 4], B[0 : 4])
        Ts.writes(C[()])
        for i in range(0, 4):
            with Ts.sblock("update"):
                vi = Ts.axis.remap("R", [i])
                C[()] = C[()] + A[vi] * B[vi]

dot_product_intrin_a_offset = T.dynamic("dot_product_intrin_a_offset", "int32")
dot_product_intrin_b_offset = T.dynamic("dot_product_intrin_b_offset", "int32")
dot_product_intrin_c_offset = T.dynamic("dot_product_intrin_c_offset", "int32")

@Ts.function
def dot_product_intrin(A: T.Tensor((4,), "float32",  layout=T.TileLayout(T.S[(4,)] + (dot_product_intrin_a_offset))), B: T.Tensor((4,), "float32",  layout=T.TileLayout(T.S[(4,)] + (dot_product_intrin_b_offset))), C: T.Tensor((), "float32",  layout=T.TileLayout(T.S[()] + (dot_product_intrin_c_offset)))) -> None:

    with Ts.sblock("root"):
        Ts.reads(C[()], A[0 : 4], B[0 : 4])
        Ts.writes(C[()])
        A_pattern = T.meta_var(match_shape_stride_pattern(A.ty))
        B_pattern = T.meta_var(match_shape_stride_pattern(B.ty))
        C_pattern = T.meta_var(match_shape_stride_pattern(C.ty))
        T.evaluate(
            T.call_extern(
                "vec4add",
                C.data,
                C_pattern.elem_offset,
                A.data,
                A_pattern.elem_offset,
                B.data,
                B_pattern.elem_offset,
                ty="int32",
            )
        )

dot_product_intrin_annotated_a_offset = T.dynamic("dot_product_intrin_annotated_a_offset", "int32")
dot_product_intrin_annotated_b_offset = T.dynamic("dot_product_intrin_annotated_b_offset", "int32")
dot_product_intrin_annotated_c_offset = T.dynamic("dot_product_intrin_annotated_c_offset", "int32")

@Ts.function
def dot_product_intrin_annotated(A: T.Tensor((4,), "float32",  layout=T.TileLayout(T.S[(4,)] + (dot_product_intrin_annotated_a_offset))), B: T.Tensor((4,), "float32",  layout=T.TileLayout(T.S[(4,)] + (dot_product_intrin_annotated_b_offset))), C: T.Tensor((), "float32",  layout=T.TileLayout(T.S[()] + (dot_product_intrin_annotated_c_offset)))) -> None:

    with Ts.sblock("root"):
        Ts.reads(C[()], A[0 : 4], B[0 : 4])
        Ts.writes(C[()])
        Ts.sblock_attr({"test_annotation": True})
        A_pattern = T.meta_var(match_shape_stride_pattern(A.ty))
        B_pattern = T.meta_var(match_shape_stride_pattern(B.ty))
        C_pattern = T.meta_var(match_shape_stride_pattern(C.ty))
        T.evaluate(
            T.call_extern(
                "vec4add",
                C.data,
                C_pattern.elem_offset,
                A.data,
                A_pattern.elem_offset,
                B.data,
                B_pattern.elem_offset,
                ty="int32",
            )
        )

outer_product_desc_a_offset = T.dynamic("outer_product_desc_a_offset", "int32")
outer_product_desc_b_offset = T.dynamic("outer_product_desc_b_offset", "int32")
outer_product_desc_c_offset = T.dynamic("outer_product_desc_c_offset", "int32")

@Ts.function
def outer_product_desc(A: T.Tensor((16, 1), "float32",  layout=T.TileLayout(T.S[(16, 1)] + (outer_product_desc_a_offset))), B: T.Tensor((16, 1), "float32",  layout=T.TileLayout(T.S[(16, 1)] + (outer_product_desc_b_offset))), C: T.Tensor((16, 16), "float32",  layout=T.TileLayout(T.S[(16, 16)] + (outer_product_desc_c_offset)))) -> None:

    with Ts.sblock("root"):
        Ts.reads(
            C[0 : 16, 0 : 16],
            A[0 : 16, 0 : 1],
            B[0 : 16, 0 : 1],
        )
        Ts.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with Ts.sblock("update"):
                vii, vjj = Ts.axis.remap("SS", [i, j])
                C[vii, vjj] = C[vii, vjj] + A[vii, 0] * B[vjj, 0]

outer_product_intrin_a_stride = T.dynamic("outer_product_intrin_a_stride", "int32")
outer_product_intrin_a_offset = T.dynamic("outer_product_intrin_a_offset", "int32")
outer_product_intrin_b_stride = T.dynamic("outer_product_intrin_b_stride", "int32")
outer_product_intrin_b_offset = T.dynamic("outer_product_intrin_b_offset", "int32")
outer_product_intrin_c_stride = T.dynamic("outer_product_intrin_c_stride", "int32")
outer_product_intrin_c_offset = T.dynamic("outer_product_intrin_c_offset", "int32")

@Ts.function
def outer_product_intrin(A: T.Tensor((16, 1), "float32",  layout=T.TileLayout(T.S[(16, 1) : [outer_product_intrin_a_stride, 1]] + (outer_product_intrin_a_offset))), B: T.Tensor((16, 1), "float32",  layout=T.TileLayout(T.S[(16, 1) : [outer_product_intrin_b_stride, 1]] + (outer_product_intrin_b_offset))), C: T.Tensor((16, 16), "float32",  layout=T.TileLayout(T.S[(16, 16) : [outer_product_intrin_c_stride, 1]] + (outer_product_intrin_c_offset)))) -> None:

    with Ts.sblock("root"):
        Ts.reads(
            C[0 : 16, 0 : 16],
            A[0 : 16, 0 : 1],
            B[0 : 16, 0 : 1],
        )
        Ts.writes(C[0 : 16, 0 : 16])
        A_pattern = T.meta_var(match_shape_stride_pattern(A.ty))
        B_pattern = T.meta_var(match_shape_stride_pattern(B.ty))
        C_pattern = T.meta_var(match_shape_stride_pattern(C.ty))
        T.evaluate(
            T.call_extern(
                "outer_product",
                C.data,
                C_pattern.elem_offset,
                A.data,
                A_pattern.elem_offset,
                B.data,
                B_pattern.elem_offset,
                ty="int32",
            )
        )

@Ts.function
def matmul(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

A_elem_offset = T.dynamic("A_elem_offset", "int32")
B_elem_offset = T.dynamic("B_elem_offset", "int32")
C_elem_offset = T.dynamic("C_elem_offset", "int32")

tensorized_matmul_a_sub_stride = T.dynamic("tensorized_matmul_a_sub_stride", "int32")
tensorized_matmul_b_sub_stride = T.dynamic("tensorized_matmul_b_sub_stride", "int32")
tensorized_matmul_c_sub_stride = T.dynamic("tensorized_matmul_c_sub_stride", "int32")

@Ts.function
def tensorized_matmul(A: T.Tensor([128, 128], "float32",  align=64), B: T.Tensor([128, 128], "float32",  align=64), C: T.Tensor([128, 128], "float32",  align=64)) -> None:

    for i_outer, j_outer in T.grid(8, 8):
        for i_inner_init, j_inner_init in T.grid(16, 16):
            with Ts.sblock("init"):
                vi_init = Ts.axis.S(128, ((i_outer * 16) + i_inner_init))
                vj_init = Ts.axis.S(128, ((j_outer * 16) + j_inner_init))
                C[vi_init, vj_init] = T.float32(0)
        for k_outer in T.grid(8):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i_outer, j_outer, k_outer])
                Ts.reads(
                    [
                        C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                        A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                        B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    ]
                )
                Ts.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                A_sub = Ts.match_tensor(
                    A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    layout=T.TileLayout(T.S[[16, 16]:[tensorized_matmul_a_sub_stride, 1]] + (A_elem_offset)),
                )
                B_sub = Ts.match_tensor(
                    B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    layout=T.TileLayout(T.S[[16, 16]:[tensorized_matmul_b_sub_stride, 1]] + (B_elem_offset)),
                )
                C_sub = Ts.match_tensor(
                    C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    [16, 16],
                    layout=T.TileLayout(T.S[[16, 16]:[tensorized_matmul_c_sub_stride, 1]] + (C_elem_offset)),
                )
                A_sub_pattern = T.meta_var(match_shape_stride_pattern(A_sub.ty))
                B_sub_pattern = T.meta_var(match_shape_stride_pattern(B_sub.ty))
                C_sub_pattern = T.meta_var(match_shape_stride_pattern(C_sub.ty))
                T.evaluate(
                    T.gpu_mma_sync(
                        C_sub.data,
                        T.floordiv(C_sub_pattern.elem_offset, 256),
                        A_sub.data,
                        T.floordiv(A_sub_pattern.elem_offset, 256),
                        B_sub.data,
                        T.floordiv(B_sub_pattern.elem_offset, 256),
                        C_sub.data,
                        T.floordiv(C_sub_pattern.elem_offset, 256),

                    )
                )

@Ts.function
def batch_matmul(
    A: T.Tensor((16, 128, 128), "float32"),
    B: T.Tensor((16, 128, 128), "float32"),
    C: T.Tensor((16, 128, 128), "float32"),
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with Ts.sblock("init"):
            vn, vi, vj = Ts.axis.remap("SSS", [n, i, j])
            C[vn, vi, vj] = T.float32(0)

    for n, i, j, k in T.grid(16, 128, 128, 128):
        with Ts.sblock("update"):
            vn, vi, vj, vk = Ts.axis.remap("SSSR", [n, i, j, k])
            C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]

A_elem_offset = T.dynamic("A_elem_offset", "int32")
B_elem_offset = T.dynamic("B_elem_offset", "int32")
C_elem_offset = T.dynamic("C_elem_offset", "int32")

tensorized_batch_matmul_mma_a_sub_stride = T.dynamic("tensorized_batch_matmul_mma_a_sub_stride", "int32")
tensorized_batch_matmul_mma_b_sub_stride = T.dynamic("tensorized_batch_matmul_mma_b_sub_stride", "int32")
tensorized_batch_matmul_mma_c_sub_stride = T.dynamic("tensorized_batch_matmul_mma_c_sub_stride", "int32")

@Ts.function
def tensorized_batch_matmul_mma(
    A: T.Tensor((16, 128, 128), "float32"),
    B: T.Tensor((16, 128, 128), "float32"),
    C: T.Tensor((16, 128, 128), "float32"),
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with Ts.sblock("init"):
            vn, vi, vj = Ts.axis.remap("SSS", [n, i, j])
            Ts.reads()
            Ts.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
    for n in range(0, 16):
        for i, j, k in T.grid(8, 8, 8):
            with Ts.sblock("update"):
                vn, vi, vj, vk = Ts.axis.remap("SSSR", [n, i, j, k])
                Ts.reads(
                    C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    A[vn : vn + 1, vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    B[vn : vn + 1, vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                )
                Ts.writes(C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                A_sub = Ts.match_tensor(
                    A[vn : vn + 1, vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    (16, 16),
                    layout=T.TileLayout(T.S[(16, 16):[tensorized_batch_matmul_mma_a_sub_stride, 1]] + (A_elem_offset)),
                )
                B_sub = Ts.match_tensor(
                    B[vn : vn + 1, vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    (16, 16),
                    layout=T.TileLayout(T.S[(16, 16):[tensorized_batch_matmul_mma_b_sub_stride, 1]] + (B_elem_offset)),
                )
                C_sub = Ts.match_tensor(
                    C[vn : vn + 1, vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    (16, 16),
                    layout=T.TileLayout(T.S[(16, 16):[tensorized_batch_matmul_mma_c_sub_stride, 1]] + (C_elem_offset)),
                )
                A_sub_pattern = T.meta_var(match_shape_stride_pattern(A_sub.ty))
                B_sub_pattern = T.meta_var(match_shape_stride_pattern(B_sub.ty))
                C_sub_pattern = T.meta_var(match_shape_stride_pattern(C_sub.ty))
                T.evaluate(
                    T.gpu_mma_sync(
                        C_sub.data,
                        T.floordiv(C_sub_pattern.elem_offset, 256),
                        A_sub.data,
                        T.floordiv(A_sub_pattern.elem_offset, 256),
                        B_sub.data,
                        T.floordiv(B_sub_pattern.elem_offset, 256),
                        C_sub.data,
                        T.floordiv(C_sub_pattern.elem_offset, 256),

                    )
                )

tensorized_batch_matmul_dot_product_a_1_offset = T.dynamic("tensorized_batch_matmul_dot_product_a_1_offset", "int32")
tensorized_batch_matmul_dot_product_b_1_offset = T.dynamic("tensorized_batch_matmul_dot_product_b_1_offset", "int32")
tensorized_batch_matmul_dot_product_c_1_offset = T.dynamic("tensorized_batch_matmul_dot_product_c_1_offset", "int32")

@Ts.function
def tensorized_batch_matmul_dot_product(
    A: T.Tensor((16, 128, 128), "float32"),
    B: T.Tensor((16, 128, 128), "float32"),
    C: T.Tensor((16, 128, 128), "float32"),
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with Ts.sblock("init"):
            vn, vi, vj = Ts.axis.remap("SSS", [n, i, j])
            Ts.reads()
            Ts.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
    for n, i, j, k_0 in T.grid(16, 128, 128, 32):
        with Ts.sblock("blockized_update"):
            vn, vi, vj, vko = Ts.axis.remap("SSSR", [n, i, j, k_0])
            Ts.reads(
                C[vn, vi, vj], A[vn, vi, vko * 4 : vko * 4 + 4], B[vn, vj, vko * 4 : vko * 4 + 4]
            )
            Ts.writes(C[vn, vi, vj])
            A_1 = Ts.match_tensor(
                A[vn, vi, vko * 4 : vko * 4 + 4], [4], dtype="float32", layout=T.TileLayout(T.S[[4]] + (tensorized_batch_matmul_dot_product_a_1_offset))
            )
            B_1 = Ts.match_tensor(
                B[vn, vj, vko * 4 : vko * 4 + 4], [4], dtype="float32", layout=T.TileLayout(T.S[[4]] + (tensorized_batch_matmul_dot_product_b_1_offset))
            )
            C_1 = Ts.match_tensor(C[vn, vi, vj], [], dtype="float32", layout=T.TileLayout(T.S[[]] + (tensorized_batch_matmul_dot_product_c_1_offset)))
            A_1_pattern = T.meta_var(match_shape_stride_pattern(A_1.ty))
            B_1_pattern = T.meta_var(match_shape_stride_pattern(B_1.ty))
            C_1_pattern = T.meta_var(match_shape_stride_pattern(C_1.ty))
            T.evaluate(
                T.call_extern(
                    "vec4add",
                    C_1.data,
                    C_1_pattern.elem_offset,
                    A_1.data,
                    A_1_pattern.elem_offset,
                    B_1.data,
                    B_1_pattern.elem_offset,
                    ty="int32",
                )
            )

tensorized_batch_matmul_outer_product_a_1_stride = T.dynamic("tensorized_batch_matmul_outer_product_a_1_stride", "int32")
tensorized_batch_matmul_outer_product_a_1_offset = T.dynamic("tensorized_batch_matmul_outer_product_a_1_offset", "int32")
tensorized_batch_matmul_outer_product_b_1_stride = T.dynamic("tensorized_batch_matmul_outer_product_b_1_stride", "int32")
tensorized_batch_matmul_outer_product_b_1_offset = T.dynamic("tensorized_batch_matmul_outer_product_b_1_offset", "int32")
tensorized_batch_matmul_outer_product_c_1_stride = T.dynamic("tensorized_batch_matmul_outer_product_c_1_stride", "int32")
tensorized_batch_matmul_outer_product_c_1_offset = T.dynamic("tensorized_batch_matmul_outer_product_c_1_offset", "int32")

@Ts.function
def tensorized_batch_matmul_outer_product(
    A: T.Tensor((16, 128, 128), "float32"),
    B: T.Tensor((16, 128, 128), "float32"),
    C: T.Tensor((16, 128, 128), "float32"),
) -> None:
    for n, i, j in T.grid(16, 128, 128):
        with Ts.sblock("init"):
            vn, vi, vj = Ts.axis.remap("SSS", [n, i, j])
            Ts.reads()
            Ts.writes(C[vn, vi, vj])
            C[vn, vi, vj] = T.float32(0)
    for n, i_0, j_0, k in T.grid(16, 8, 8, 128):
        with Ts.sblock("blockized_update"):
            vn, vio, vjo, vk = Ts.axis.remap("SSSR", [n, i_0, j_0, k])
            Ts.reads(
                C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                A[vn, vio * 16 : vio * 16 + 16, vk],
                B[vn, vjo * 16 : vjo * 16 + 16, vk],
            )
            Ts.writes(C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
            A_1 = Ts.match_tensor(A[vn, vio * 16 : vio * 16 + 16, vk], [16, 1], layout=T.TileLayout(T.S[[16, 1]:[tensorized_batch_matmul_outer_product_a_1_stride, 1]] + (tensorized_batch_matmul_outer_product_a_1_offset)), dtype="float32")
            B_1 = Ts.match_tensor(B[vn, vjo * 16 : vjo * 16 + 16, vk], [16, 1], layout=T.TileLayout(T.S[[16, 1]:[tensorized_batch_matmul_outer_product_b_1_stride, 1]] + (tensorized_batch_matmul_outer_product_b_1_offset)), dtype="float32"
            )
            C_1 = Ts.match_tensor(
                C[vn, vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16], [16, 16], layout=T.TileLayout(T.S[[16, 16]:[tensorized_batch_matmul_outer_product_c_1_stride, 1]] + (tensorized_batch_matmul_outer_product_c_1_offset)), dtype="float32"
            )
            A_1_pattern = T.meta_var(match_shape_stride_pattern(A_1.ty))
            B_1_pattern = T.meta_var(match_shape_stride_pattern(B_1.ty))
            C_1_pattern = T.meta_var(match_shape_stride_pattern(C_1.ty))
            T.evaluate(
                T.call_extern("outer_product", C_1.data, C_1_pattern.elem_offset, A_1.data, A_1_pattern.elem_offset,
                              B_1.data, B_1_pattern.elem_offset, ty="int32"
                )
            )

annotated_mma_desc_a_offset = T.dynamic("annotated_mma_desc_a_offset", "int32")
annotated_mma_desc_b_offset = T.dynamic("annotated_mma_desc_b_offset", "int32")
annotated_mma_desc_c_offset = T.dynamic("annotated_mma_desc_c_offset", "int32")

@Ts.function
def annotated_mma_desc(A: T.Tensor((16, 16), "float32", align=64,  layout=T.TileLayout(T.S[(16, 16)] + (annotated_mma_desc_a_offset))), B: T.Tensor((16, 16), "float32", align=64,  layout=T.TileLayout(T.S[(16, 16)] + (annotated_mma_desc_b_offset))), C: T.Tensor((16, 16), "float32", align=64,  layout=T.TileLayout(T.S[(16, 16)] + (annotated_mma_desc_c_offset)))) -> None:

    with Ts.sblock("root"):
        Ts.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0 : 16, 0 : 16])
        Ts.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with Ts.sblock("update"):
                Ts.sblock_attr({"test_annotation": True})
                vii, vjj, vkk = Ts.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]

@Ts.function
def annotated_matmul(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            Ts.sblock_attr({"test_annotation": True})
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

A_elem_offset = T.dynamic("A_elem_offset", "int32")
B_elem_offset = T.dynamic("B_elem_offset", "int32")
C_elem_offset = T.dynamic("C_elem_offset", "int32")

annotated_tensorized_matmul_a_sub_stride = T.dynamic("annotated_tensorized_matmul_a_sub_stride", "int32")
annotated_tensorized_matmul_b_sub_stride = T.dynamic("annotated_tensorized_matmul_b_sub_stride", "int32")
annotated_tensorized_matmul_c_sub_stride = T.dynamic("annotated_tensorized_matmul_c_sub_stride", "int32")

@Ts.function
def annotated_tensorized_matmul(A: T.Tensor([128, 128], "float32",  align=64), B: T.Tensor([128, 128], "float32",  align=64), C: T.Tensor([128, 128], "float32",  align=64)) -> None:

    for i_outer, j_outer in T.grid(8, 8):
        for i_inner_init, j_inner_init in T.grid(16, 16):
            with Ts.sblock("init"):
                vi_init = Ts.axis.S(128, ((i_outer * 16) + i_inner_init))
                vj_init = Ts.axis.S(128, ((j_outer * 16) + j_inner_init))
                Ts.sblock_attr({"test_annotation": True})
                C[vi_init, vj_init] = T.float32(0)
        for k_outer in T.grid(8):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i_outer, j_outer, k_outer])
                Ts.reads(
                    [
                        C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                        A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                        B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    ]
                )
                Ts.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                A_sub = Ts.match_tensor(
                    A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    layout=T.TileLayout(T.S[[16, 16]:[annotated_tensorized_matmul_a_sub_stride, 1]] + (A_elem_offset)),
                )
                B_sub = Ts.match_tensor(
                    B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                    [16, 16],
                    layout=T.TileLayout(T.S[[16, 16]:[annotated_tensorized_matmul_b_sub_stride, 1]] + (B_elem_offset)),
                )
                C_sub = Ts.match_tensor(
                    C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                    [16, 16],
                    layout=T.TileLayout(T.S[[16, 16]:[annotated_tensorized_matmul_c_sub_stride, 1]] + (C_elem_offset)),
                )
                A_sub_pattern = T.meta_var(match_shape_stride_pattern(A_sub.ty))
                B_sub_pattern = T.meta_var(match_shape_stride_pattern(B_sub.ty))
                C_sub_pattern = T.meta_var(match_shape_stride_pattern(C_sub.ty))
                T.evaluate(
                    T.gpu_mma_sync(
                        C_sub.data,
                        T.floordiv(C_sub_pattern.elem_offset, 256),
                        A_sub.data,
                        T.floordiv(A_sub_pattern.elem_offset, 256),
                        B_sub.data,
                        T.floordiv(B_sub_pattern.elem_offset, 256),
                        C_sub.data,
                        T.floordiv(C_sub_pattern.elem_offset, 256),

                    )
                )

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

s_tir.TensorIntrin.register("test_mma_intrin", mma_desc, mma_intrin)
s_tir.TensorIntrin.register("test_annotated_mma_intrin", annotated_mma_desc, mma_intrin)
s_tir.TensorIntrin.register("test_dot_product_intrin", dot_product_desc, dot_product_intrin)
s_tir.TensorIntrin.register("test_outer_product_intrin", outer_product_desc, outer_product_intrin)
s_tir.TensorIntrin.register("test_dot_product_intrin_annotated", dot_product_desc, dot_product_intrin_annotated)

def test_tensorize_matmul():
    func = matmul
    # schedule
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    update = s.get_sblock("update")
    i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)
    s.tensorize(ii, "test_mma_intrin")
    assert_structural_equal_ignore_global_symbol(tensorized_matmul, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)

def test_tensorize_batch_matmul():
    func = batch_matmul
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    update = s.get_sblock("update")
    _, i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.tensorize(ii, "test_mma_intrin")
    assert_structural_equal_ignore_global_symbol(tensorized_batch_matmul_mma, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=batch_matmul)

def test_tensorize_dot_product():
    func = batch_matmul
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    C = s.get_sblock("update")
    _, _, _, k = s.get_loops(C)
    _, ki = s.split(k, factors=[None, 4])
    s.tensorize(ki, "test_dot_product_intrin")
    assert_structural_equal_ignore_global_symbol(tensorized_batch_matmul_dot_product, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)

def test_tensorize_outer_product():
    func = batch_matmul
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    C = s.get_sblock("update")
    _, i, j, k = s.get_loops(C)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    s.reorder(io, jo, k, ii, ji)
    s.tensorize(ii, "test_outer_product_intrin")
    assert_structural_equal_ignore_global_symbol(tensorized_batch_matmul_outer_product, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)

def test_tensorize_with_annotation():
    func = annotated_matmul
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    update = s.get_sblock("update")
    i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)
    s.tensorize(ii, "test_annotated_mma_intrin")
    assert_structural_equal_ignore_global_symbol(annotated_tensorized_matmul, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)

def test_tensorize_intrinsic_with_annotation():
    func = matmul
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    update = s.get_sblock("update")
    _, _, k = s.get_loops(update)
    ko, ki = s.split(k, factors=[None, 4])
    s.decompose_reduction(update, ko)
    s.tensorize(ki, "test_dot_product_intrin_annotated")

    b = s.get(s.get_sblock("update_update_o"))
    assert b.annotations["test_annotation"] == T.bool(True)
    verify_trace_roundtrip(sch=s, mod=func)

def get_matmul_packed(m, n, k, lhs_type, rhs_dtype="int8"):
    X = te.placeholder((m, k), name="X", dtype=lhs_type)
    W = te.placeholder((n, k), name="W", dtype=rhs_dtype)

    ak = te.reduce_axis((0, k), name="k")
    matmul = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32") * W[j, ak].astype("int32"),
            axis=ak,
        ),
        name="compute",
    )

    return te.create_function([X, W, matmul])

def tensorize_16x4_test(intrin=VNNI_DOT_16x4_INTRIN):
    m, n, k = 128, 128, 128

    func = get_matmul_packed(m, n, k, "uint8")

    sch = tvm.s_tir.Schedule(func, debug_mask="all")
    block = sch.get_sblock("compute")
    sch.transform_layout(block, "W", lambda i, j: [i//16, j//4, i%16, j%4])
    _, j, k = sch.get_loops(block)

    _, ji = sch.split(j, factors=[None, 16])
    ko, ki = sch.split(k, factors=[None, 4])
    sch.reorder(ko, ji, ki)

    sch.decompose_reduction(block, ko)
    sch.tensorize(ji, intrin)

    verify_trace_roundtrip(sch=sch, mod=func)

def test_tensorize_vnni():
    tensorize_16x4_test()

def test_tensorize_avx512():
    tensorize_16x4_test(AVX512_DOT_16x4_INTRIN)

def test_tensorize_arm_dot():
    m, n, k = 128, 128, 128

    func = get_matmul_packed(m, n, k, "int8")

    for intrin in [ARM_DOT_4x4_i8_SDOT_INTRIN, ARM_DOT_4x4_i8_NEON_INTRIN]:
        sch = tvm.s_tir.Schedule(func, debug_mask="all")
        block = sch.get_sblock("compute")
        sch.transform_layout(block, "W", lambda i, j: [i//4, j//4, i%4, j%4])
        _, j, k = sch.get_loops(block)

        _, ji = sch.split(j, factors=[None, 4])
        ko, ki = sch.split(k, factors=[None, 4])
        sch.reorder(ko, ji, ki)

        sch.decompose_reduction(block, ko)
        sch.tensorize(ji, intrin)

        verify_trace_roundtrip(sch=sch, mod=func)

def test_tensorize_vrmpy():
    m, n, k = 128, 128, 128

    func = get_matmul_packed(m, n, k, "uint8", "uint8")

    sch = tvm.s_tir.Schedule(func, debug_mask="all")
    block = sch.get_sblock("compute")
    sch.transform_layout(block, "W", lambda i, j: [i//32, j//4, i%32, j%4])
    _, j, k = sch.get_loops(block)

    _, ji = sch.split(j, factors=[None, 32])
    ko, ki = sch.split(k, factors=[None, 4])
    sch.reorder(ko, ji, ki)

    sch.decompose_reduction(block, ko)
    sch.tensorize(ji, VRMPY_u8u8i32_INTRIN)

    verify_trace_roundtrip(sch=sch, mod=func)

def test_tensorize_vdmpy():
    m, n, k = 128, 128, 128

    func = get_matmul_packed(m, n, k, "int16", "int16")

    sch = tvm.s_tir.Schedule(func, debug_mask="all")
    block = sch.get_sblock("compute")
    sch.transform_layout(block, "W", lambda i, j: [i//32, j//2, i%32, j%2])
    _, j, k = sch.get_loops(block)

    _, ji = sch.split(j, factors=[None, 32])
    ko, ki = sch.split(k, factors=[None, 2])
    sch.reorder(ko, ji, ki)

    sch.decompose_reduction(block, ko)
    sch.tensorize(ji, VDMPY_i16i16i32_INTRIN)

    verify_trace_roundtrip(sch=sch, mod=func)

def test_tensorize_dp4a():
    # pylint: disable=too-many-locals
    def _test_intrin(dtype_a, dtype_b, dtype_c, intrin):
        m, n, k = 128, 128, 128
        X = te.placeholder((m, k), name="X", dtype=dtype_a)
        W = te.placeholder((n, k), name="W", dtype=dtype_b)
        ak = te.reduce_axis((0, k), name="k")

        matmul = te.compute(
            (m, n),
            lambda i, j: te.sum(
                X[i, ak].astype(dtype_c) * W[j, ak].astype(dtype_c),
                axis=ak,
            ),
            name="compute",
        )

        func = te.create_function([X, W, matmul])

        sch = tvm.s_tir.Schedule(func, debug_mask="all")
        block = sch.get_sblock("compute")
        i, j, k = sch.get_loops(block)

        by, ty, yi = sch.split(i, factors=sch.sample_perfect_tile(i, n=3))
        bx, tx, xi = sch.split(j, factors=sch.sample_perfect_tile(j, n=3))
        ko, ki = sch.split(k, [None, 4])
        ko, kt = sch.split(ko, factors=sch.sample_perfect_tile(ko, n=2))

        sch.reorder(by, bx, ty, tx, yi, xi)
        CC = sch.cache_write(block, 0, "local")
        sch.reverse_compute_at(CC, tx)

        def fetch_to_shared(block, idx):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, ko, True)
            return block_read

        fetch_to_shared(block, 0)
        fetch_to_shared(block, 1)

        sch.decompose_reduction(block, ko)
        sch.tensorize(ki, intrin)

        verify_trace_roundtrip(sch=sch, mod=func)

    for args in [
        ("int8", "int8", "int32", AMDGPU_SDOT4_INTRIN),
        ("int8", "int8", "int32", DP4A_S8S8S32_INTRIN),
        ("int8", "uint8", "int32", DP4A_S8U8S32_INTRIN),
        ("uint8", "int8", "int32", DP4A_U8S8S32_INTRIN),
        ("uint8", "uint8", "uint32", DP4A_U8U8U32_INTRIN),
    ]:
        _test_intrin(*args)

def test_tensor_intrin_look_up():
    intrin_name = 'non_existent_intrin'
    assert s_tir.TensorIntrin.get(intrin_name, allow_missing=True) is None
    with pytest.raises(ValueError):
        s_tir.TensorIntrin.get(intrin_name)

def test_tensorize_matmul_mixed_dtype():
    # fmt: off
    @Ts.function
    def matmul_int64_shape(
        A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        C: T.Tensor((T.int64(128), T.int64(128)), "float32")
    ) -> None:
        for i_0, j_0 in T.grid(T.int64(8), T.int64(8)):
            for i_1_init, j_1_init in T.grid(T.int64(16), T.int64(16)):
                with Ts.sblock("init"):
                    vi = Ts.axis.spatial(T.int64(128), i_0 * T.int64(16) + i_1_init)
                    vj = Ts.axis.spatial(T.int64(128), j_0 * T.int64(16) + j_1_init)
                    C[vi, vj] = T.float32(0)
            for k_0, i_1, j_1, k_1 in T.grid(T.int64(8), T.int64(16), T.int64(16), T.int64(16)):
                with Ts.sblock("update"):
                    vi = Ts.axis.spatial(T.int64(128), i_0 * T.int64(16) + i_1)
                    vj = Ts.axis.spatial(T.int64(128), j_0 * T.int64(16) + j_1)
                    vk = Ts.axis.reduce(T.int64(128), k_0 * T.int64(16) + k_1)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    A_elem_offset = T.dynamic("A_elem_offset", "int64")
    B_elem_offset = T.dynamic("B_elem_offset", "int64")
    C_elem_offset = T.dynamic("C_elem_offset", "int64")

    tensorized_matmul_int64_shape_a_sub_stride = T.dynamic("tensorized_matmul_int64_shape_a_sub_stride", "int64")
    tensorized_matmul_int64_shape_b_sub_stride = T.dynamic("tensorized_matmul_int64_shape_b_sub_stride", "int64")
    tensorized_matmul_int64_shape_c_sub_stride = T.dynamic("tensorized_matmul_int64_shape_c_sub_stride", "int64")

    @Ts.function
    def tensorized_matmul_int64_shape(
        A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        C: T.Tensor((T.int64(128), T.int64(128)), "float32")
    ) -> None:
        for i_outer, j_outer in T.grid(T.int64(8), T.int64(8)):
            for i_inner_init, j_inner_init in T.grid(T.int64(16), T.int64(16)):
                with Ts.sblock("init"):
                    vi = Ts.axis.spatial(T.int64(128), i_outer * T.int64(16) + i_inner_init)
                    vj = Ts.axis.spatial(T.int64(128), j_outer * T.int64(16) + j_inner_init)
                    C[vi, vj] = T.float32(0)
            for k_outer in T.grid(T.int64(8)):
                with Ts.sblock("update"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i_outer, j_outer, k_outer])
                    Ts.reads(
                        [
                            C[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vj * T.int64(16) : vj * T.int64(16) + T.int64(16)],
                            A[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vk * T.int64(16) : vk * T.int64(16) + T.int64(16)],
                            B[vj * T.int64(16) : vj * T.int64(16) + T.int64(16), vk * T.int64(16) : vk * T.int64(16) + T.int64(16)],
                        ]
                    )
                    Ts.writes(C[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vj * T.int64(16) : vj * T.int64(16) + T.int64(16)])
                    A_sub = Ts.match_tensor(
                        A[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vk * T.int64(16) : vk * T.int64(16) + T.int64(16)],
                        [T.int64(16), T.int64(16)],
                        layout=T.TileLayout(T.S[[T.int64(16), T.int64(16)]:[tensorized_matmul_int64_shape_a_sub_stride, T.int64(1)]] + (A_elem_offset)),
                    )
                    B_sub = Ts.match_tensor(
                        B[vj * T.int64(16) : vj * T.int64(16) + T.int64(16), vk * T.int64(16) : vk * T.int64(16) + T.int64(16)],
                        [T.int64(16), T.int64(16)],
                        layout=T.TileLayout(T.S[[T.int64(16), T.int64(16)]:[tensorized_matmul_int64_shape_b_sub_stride, T.int64(1)]] + (B_elem_offset)),
                    )
                    C_sub = Ts.match_tensor(
                        C[vi * T.int64(16) : vi * T.int64(16) + T.int64(16), vj * T.int64(16) : vj * T.int64(16) + T.int64(16)],
                        [T.int64(16), T.int64(16)],
                        layout=T.TileLayout(T.S[[T.int64(16), T.int64(16)]:[tensorized_matmul_int64_shape_c_sub_stride, T.int64(1)]] + (C_elem_offset)),
                    )
                    A_sub_pattern = T.meta_var(match_shape_stride_pattern(A_sub.ty))
                    B_sub_pattern = T.meta_var(match_shape_stride_pattern(B_sub.ty))
                    C_sub_pattern = T.meta_var(match_shape_stride_pattern(C_sub.ty))
                    T.evaluate(
                        T.gpu_mma_sync(
                            C_sub.data,
                            T.floordiv(C_sub_pattern.elem_offset, T.int64(256)),
                            A_sub.data,
                            T.floordiv(A_sub_pattern.elem_offset, T.int64(256)),
                            B_sub.data,
                            T.floordiv(B_sub_pattern.elem_offset, T.int64(256)),
                            C_sub.data,
                            T.floordiv(C_sub_pattern.elem_offset, T.int64(256)),

                        )
                    )
    # fmt: on

    s = tvm.s_tir.Schedule(matmul_int64_shape, debug_mask="all")
    update = s.get_sblock("update")
    ii = s.get_loops(update)[-3]
    s.tensorize(ii, "test_mma_intrin")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], tensorized_matmul_int64_shape)
    verify_trace_roundtrip(sch=s, mod=matmul_int64_shape)

def _tir_packed_int_to_int_to_float(storage_nbit: int):
    storage_dtype = "int" + str(storage_nbit)

    def f_convert(nbit: int, val: tirx.Expr, pos: tirx.Expr, dtype: str):
        assert val.ty.dtype == storage_dtype
        mask = tirx.const((1 << nbit) - 1, "int32")
        unextended = (val >> (pos.astype("int32") * tirx.const(nbit, "int32"))) & mask
        return tirx.Cast(dtype, (unextended << tirx.const(32 - nbit, "int32")) >> tirx.const(32 - nbit, "int32"))

    return f_convert

@Ts.function
def decode_i4s_to_f16_desc(Compressed: T.Tensor([1], dtype='int32', scope='local'), Decompressed: T.Tensor([8], dtype='float16', scope='local')) -> None:

    with Ts.sblock("root"):
        Ts.reads(Compressed[0:1])
        Ts.writes(Decompressed[0:8])
        for i in T.grid(8):
            with Ts.sblock("decode"):
                vi = Ts.axis.remap("S", [i])
                Decompressed[vi] = _tir_packed_int_to_int_to_float(32)(
                    4,
                    Compressed[vi // 8],
                    vi % 8,
                    dtype="float16",
                )

@Ts.function
def decode_i4s_to_f16_impl(Compressed: T.Tensor([1], dtype='int32', scope='local'), Decompressed: T.Tensor([8], dtype='float16', scope='local')) -> None:

    with Ts.sblock("root"):
        Ts.reads(Compressed[0:1])
        Ts.writes(Decompressed[0:8])
        T.call_extern(

            "test_decode_i4s_to_f16",
            Compressed.data,
            Decompressed.data,
            8,
         ty="handle")

s_tir.TensorIntrin.register("test_decode_i4s_to_f16_intrin", decode_i4s_to_f16_desc, decode_i4s_to_f16_impl)

def test_tensorize_arith_simplification():
    # fmt: off
    @Ts.function
    def decode_i4s_to_int32_to_f16():
        B_decode_local = Ts.sblock_alloc_tensor((16384, 16384), "float16", scope="local")
        B_local = Ts.sblock_alloc_tensor((16384, 2048), "int32", scope="local")
        for ax0_0 in T.parallel(8192, annotations={"thread_binding": "blockIdx.x"}):
            for ax0_1 in T.parallel(2, annotations={"thread_binding": "threadIdx.y"}):
                for ax1_0 in range(32):
                    for ax1_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                        for ax0, ax1 in T.grid(1, 8):
                            with Ts.sblock("B_decode_local"):
                                v0 = Ts.axis.spatial(16384, ax0_0 * 2 + ax0_1 + ax0)
                                v1 = Ts.axis.spatial(16384, ax1_0 * 512 + ax1_1 * 8 + ax1)
                                Ts.reads(B_local[v0, v1 // 8])
                                Ts.writes(B_decode_local[v0, v1])
                                B_decode_local[v0, v1] = T.Cast("float16", T.shift_right(T.shift_left(T.bitwise_and(T.shift_right(B_local[v0, v1 // 8], v1 % 8 * 4), 15), 28), 28))

    @Ts.function
    def tensorized_decode_i4s_to_int32_to_f16():
        B_decode_local = Ts.sblock_alloc_tensor((16384, 16384), "float16", scope="local")
        B_local = Ts.sblock_alloc_tensor((16384, 2048), "int32", scope="local")
        for ax0_0 in T.parallel(8192, annotations={"thread_binding": "blockIdx.x"}):
            for ax0_1 in T.parallel(2, annotations={"thread_binding": "threadIdx.y"}):
                for ax1_0 in range(32):
                    for ax1_1 in T.parallel(64, annotations={"thread_binding": "threadIdx.x"}):
                        for ax0 in range(1):
                            with Ts.sblock("B_decode_local_o"):
                                v0_o = Ts.axis.spatial(16384, ax0_0 * 2 + ax0_1 + ax0)
                                v1_o = Ts.axis.spatial(2048, ax1_0 * 64 + ax1_1)
                                Ts.reads(B_local[v0_o, v1_o])
                                Ts.writes(B_decode_local[v0_o, v1_o * 8:v1_o * 8 + 8])
                                Compressed = Ts.match_tensor(B_local[v0_o, v1_o], (1,), "int32", scope="local")
                                Decompressed = Ts.match_tensor(B_decode_local[v0_o, v1_o * 8:v1_o * 8 + 8], (8,), "float16", scope="local")
                                T.call_extern( "test_decode_i4s_to_f16", Compressed.data, Decompressed.data, 8, ty="handle")

    s = tvm.s_tir.Schedule(decode_i4s_to_int32_to_f16, debug_mask="all")
    update = s.get_sblock("B_decode_local")
    ii = s.get_loops(update)[-1]
    s.tensorize(ii, "test_decode_i4s_to_f16_intrin")
    assert_structural_equal_ignore_global_symbol(s.mod["main"], tensorized_decode_i4s_to_int32_to_f16)
    verify_trace_roundtrip(sch=s, mod=decode_i4s_to_int32_to_f16)

if __name__ == "__main__":
    tvm_tirx.testing.main()
