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
# pylint: disable=invalid-name,missing-function-docstring,unused-import
# ruff: noqa: E501, F401
"""Intrinsics for ARM tensorization."""

from tvm.s_tir import match_shape_stride_pattern
from tvm.s_tir import script as Ts
from tvm.s_tir.script.ir_builder import function as build_function
from tvm_tirx import tirx
from tvm_tirx.codegen import llvm
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder

from .. import TensorIntrin
from .dot_product_common import (
    DP4A_S8S8S32_INTRIN,
    DP4A_S8U8S32_INTRIN,
    DP4A_U8S8S32_INTRIN,
    DP4A_U8U8U32_INTRIN,
)

# TODO(masahi): Parametrize the TVMScript description of dot product by
# shape and dtype, and share the common description with x86.


neon_4x4_i8i8i32_desc_a_offset = T.dynamic("neon_4x4_i8i8i32_desc_a_offset", "int32")
neon_4x4_i8i8i32_desc_b_offset = T.dynamic("neon_4x4_i8i8i32_desc_b_offset", "int32")
neon_4x4_i8i8i32_desc_c_offset = T.dynamic("neon_4x4_i8i8i32_desc_c_offset", "int32")


@Ts.function
def neon_4x4_i8i8i32_desc(
    A: T.Tensor((4,), "int8", layout=T.TileLayout(T.S[(4,)] + (neon_4x4_i8i8i32_desc_a_offset))),
    B: T.Tensor(
        (4, 4), "int8", layout=T.TileLayout(T.S[(4, 4)] + (neon_4x4_i8i8i32_desc_b_offset))
    ),
    C: T.Tensor((4,), "int32", layout=T.TileLayout(T.S[(4,)] + (neon_4x4_i8i8i32_desc_c_offset))),
) -> None:
    with Ts.sblock("root"):
        Ts.reads(C[0:4], A[0:4], B[0:4, 0:4])
        Ts.writes(C[0:4])
        for i in T.serial(0, 4):
            for k in T.serial(0, 4):
                with Ts.sblock("update"):
                    vi, vk = Ts.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], ty="int32") * T.cast(B[vi, vk], ty="int32")


neon_4x4_i8i8i32_impl_a_offset = T.dynamic("neon_4x4_i8i8i32_impl_a_offset", "int32")
neon_4x4_i8i8i32_impl_b_offset = T.dynamic("neon_4x4_i8i8i32_impl_b_offset", "int32")
neon_4x4_i8i8i32_impl_c_offset = T.dynamic("neon_4x4_i8i8i32_impl_c_offset", "int32")


@Ts.function
def neon_4x4_i8i8i32_impl(
    A: T.Tensor((4,), "int8", layout=T.TileLayout(T.S[(4,)] + (neon_4x4_i8i8i32_impl_a_offset))),
    B: T.Tensor(
        (4, 4), "int8", layout=T.TileLayout(T.S[(4, 4)] + (neon_4x4_i8i8i32_impl_b_offset))
    ),
    C: T.Tensor((4,), "int32", layout=T.TileLayout(T.S[(4,)] + (neon_4x4_i8i8i32_impl_c_offset))),
) -> None:
    with Ts.sblock("root"):
        Ts.reads(C[0:4], A[0:4], B[0:4, 0:4])
        Ts.writes(C[0:4])

        A_int8 = A[T.Ramp(0, 1, 4)]
        re_int32 = T.reinterpret(A_int8, ty="int32")
        vec_ai32 = T.broadcast(re_int32, 2)
        vec_a = T.reinterpret(vec_ai32, ty="int8x8")

        vec_b = B[0, T.Ramp(0, 1, 16)]

        # TODO(masahi): Remove duplication when inlined function call is supported
        vec_b_low = T.vector_low(vec_b, ty="int8x8")

        multiply_low = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            vec_a,
            vec_b_low,
            ty="int16x8",
        )

        pairwise_reduction_low = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            multiply_low,
            ty="int32x4",
        )

        vec_b_high = T.vector_high(vec_b, ty="int8x8")

        multiply_high = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            vec_a,
            vec_b_high,
            ty="int16x8",
        )

        pairwise_reduction_high = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            multiply_high,
            ty="int32x4",
        )

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.addp.v4i32"),
            pairwise_reduction_low,
            pairwise_reduction_high,
            ty="int32x4",
        )


def get_dotprod_intrin(in_dtype, out_dtype):
    if in_dtype == "uint8":
        instr = "udot.v4u32.v16u8"
    else:  # if in_dtype == "int8"
        instr = "sdot.v4i32.v16i8"

    out_dtype_x4 = f"{out_dtype}x4"
    in_dtype_x16 = f"{in_dtype}x16"

    dot_prod_desc_a_offset = T.dynamic("dot_prod_desc_a_offset", "int32")
    dot_prod_desc_b_offset = T.dynamic("dot_prod_desc_b_offset", "int32")
    dot_prod_desc_c_offset = T.dynamic("dot_prod_desc_c_offset", "int32")

    @Ts.function
    def dot_prod_desc(
        A: T.Tensor(
            (4,), dtype=in_dtype, layout=T.TileLayout(T.S[(4,)] + (dot_prod_desc_a_offset))
        ),
        B: T.Tensor(
            (4, 4), dtype=in_dtype, layout=T.TileLayout(T.S[(4, 4)] + (dot_prod_desc_b_offset))
        ),
        C: T.Tensor(
            (4,), dtype=out_dtype, layout=T.TileLayout(T.S[(4,)] + (dot_prod_desc_c_offset))
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:4], A[0:4], B[0:4, 0:4])
            Ts.writes(C[0:4])
            for i in T.serial(0, 4):
                for k in T.serial(0, 4):
                    with Ts.sblock("update"):
                        vi, vk = Ts.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], ty=out_dtype) * T.cast(
                            B[vi, vk], ty=out_dtype
                        )

    dot_prod_impl_a_offset = T.dynamic("dot_prod_impl_a_offset", "int32")
    dot_prod_impl_b_offset = T.dynamic("dot_prod_impl_b_offset", "int32")
    dot_prod_impl_c_offset = T.dynamic("dot_prod_impl_c_offset", "int32")

    @Ts.function
    def dot_prod_impl(
        A: T.Tensor(
            (4,), dtype=in_dtype, layout=T.TileLayout(T.S[(4,)] + (dot_prod_impl_a_offset))
        ),
        B: T.Tensor(
            (4, 4), dtype=in_dtype, layout=T.TileLayout(T.S[(4, 4)] + (dot_prod_impl_b_offset))
        ),
        C: T.Tensor(
            (4,), dtype=out_dtype, layout=T.TileLayout(T.S[(4,)] + (dot_prod_impl_c_offset))
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:4], A[0:4], B[0:4, 0:4])
            Ts.writes(C[0:4])

            A_i8x4 = A[T.Ramp(0, 1, 4)]
            A_i32 = T.reinterpret(A_i8x4, ty=out_dtype)
            vec_ai32 = T.broadcast(A_i32, 4)
            vec_a = T.reinterpret(vec_ai32, ty=in_dtype_x16)

            vec_b = B[0, T.Ramp(0, 1, 16)]

            vec_c = C[T.Ramp(0, 1, 4)]

            C[T.ramp(T.int32(0), 1, 4)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id(f"llvm.aarch64.neon.{instr}"),
                vec_c,
                vec_a,
                vec_b,
                ty=out_dtype_x4,
            )

    return dot_prod_desc, dot_prod_impl


def _create_ptrue_mask(dtype):
    """
    Creates a mask that enables all lanes of a scalable vector.
    """
    return T.broadcast(T.bool(True), tirx.get_vscale_expr(dtype))


def _create_active_lane_mask(tensor, relative_offsets, vertical_limit):
    """
    Get the active lane mask intrinsic call for predicated accesses.

    Parameters
    ----------
    tensor : tvm_tirx.ir.Var
        The tensor the tensor access will be performed on.
    relative_offsets : Tuple[Expr, Expr]
        The vertical and horizontal offsets into the accumulator tile.
    vertical_limit : Expr
        An absolute offset specifying the limit at which rows should be stored.

    Returns
    -------
    Expr
        The active lane mask intrinsic.
    """
    vertical_offset, horizontal_offset = relative_offsets
    tensor_pattern = match_shape_stride_pattern(tensor.ty)
    assert tensor_pattern is not None
    stride = tensor_pattern.strides[0]

    # The base is the offset of the first value we wish to store
    base = T.int32(
        tensor_pattern.elem_offset
        + vertical_offset * stride
        + horizontal_offset * tensor_pattern.strides[1]
    )

    # The limit is the maximum offset in the current row of 'base' that we wish to allow values
    # to be stored. Calculating this limit is a bit tricky since we can only request offsets of
    # elements in the tensorized tile of the output tensor. One way to calculate this is to find
    # the offset of the first value in the row of the output tensor that 'base' is in and add
    # 'stride' to it.
    limit = (
        base
        - T.int32(horizontal_offset)
        - T.int32(tensor_pattern.elem_offset % stride)
        + T.int32(stride)
    )
    limit = T.Min(limit, T.Cast("int32", vertical_limit) * stride)

    return T.get_active_lane_mask(
        T.Cast("int32", base), T.Cast("int32", limit), ty="uint1xvscalex4"
    )


def get_sme_transpose_interleave_2svlx2svl_fp32_intrin(cols, rows):
    """
    Transpose a matrix of size 2SVL x 2SVL (where 'SVL' is the Scalable Vector Length) using
    the Scalable Matrix Extension (SME).

    This is completed by loading rows of the input matrix into the accumulator tile,
    then storing the columns. The SME accumulator tile is divided into a series of sub-tiles
    which must be loaded to / stored from independently.

    Example
    -------
    An example case for float32. In this instance the accumulator tile is divided into 4
    sub-tiles of size SVLxSVL numbered 0-3. We start by loading rows of A, each SVL in length,
    into each of the sub-tiles. In the diagram below, each load for a sub-tile is sequenced by
    a, b, ... till the tile is full.

    The columns of each sub-tile are then stored into A_t. Note that to perform a transpose,
    the contents of sub-tile 1 and 2 are stored in opposite locations - see the diagram
    below.

    ::

        A:                                  Accumulator tile:                     A_t:
                    2SVL                                2SVL                               2SVL
             +----------------+                 +-----------------+                +-------------------+
             | --0a--  --1a-- |                 |                 |                | |  |     |  |     |
             | --0b--  --1b-- |                 |    0       1    |                | 0a 0b .. 2a 2b .. |
             |   ...     ...  | ld1w.horiz      |                 | st1w.vert      | |  |     |  |     |
        2SVL | --2a--  --3a-- |   ====>    2SVL |                 |   ====>   2SVL | |  |     |  |     |
             | --2a--  --3b-- |                 |    2       3    |                | 1a 1b .. 3a 3b .. |
             |   ...     ...  |                 |                 |                | |  |     |  |     |
             +----------------+                 +-----------------+                +-------------------+

    Returns
    -------
    intrin : TensorIntrin
        The SME TensorIntrin that can be used in tensorizing a schedule.

    """
    SVF = tirx.get_vscale_expr("float32")
    SVF2 = 2 * SVF

    desc_a_offset = T.dynamic("desc_a_offset", "int32")
    desc_a_t_offset = T.dynamic("desc_a_t_offset", "int32")

    @Ts.function
    def desc(
        A: T.Tensor(
            (SVF2, SVF2), dtype="float32", layout=T.TileLayout(T.S[(SVF2, SVF2)] + (desc_a_offset))
        ),
        A_t: T.Tensor(
            (SVF2, SVF2),
            dtype="float32",
            layout=T.TileLayout(T.S[(SVF2, SVF2)] + (desc_a_t_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:SVF2, 0:SVF2])
            Ts.writes(A_t[0:SVF2, 0:SVF2])
            for k, m in T.grid(SVF2, SVF2):
                with Ts.sblock("transpose"):
                    v_m, v_k = Ts.axis.remap("SS", [m, k])
                    A_t[v_k, v_m] = A[v_m, v_k]

    impl_a_offset = T.dynamic("impl_a_offset", "int32")
    impl_a_t_offset = T.dynamic("impl_a_t_offset", "int32")
    impl_a_stride = T.dynamic("impl_a_stride", "int32")
    impl_a_t_stride = T.dynamic("impl_a_t_stride", "int32")

    def impl():
        sub_tile_count = 4

        with IRBuilder() as ib:
            with build_function():
                A = T.arg_(
                    "a",
                    T.Tensor(
                        (SVF2, SVF2),
                        "float32",
                        layout=T.TileLayout(
                            T.S[(SVF2, SVF2) : [impl_a_stride, 1]] + (impl_a_offset)
                        ),
                    ),
                )
                A_t = T.arg_(
                    "a_t",
                    T.Tensor(
                        (SVF2, SVF2),
                        "float32",
                        layout=T.TileLayout(
                            T.S[(SVF2, SVF2) : [impl_a_t_stride, 1]] + (impl_a_t_offset)
                        ),
                    ),
                )

                with Ts.sblock("root"):
                    Ts.reads(A[0:SVF2, 0:SVF2])
                    Ts.writes(A_t[0:SVF2, 0:SVF2])

                    # Load rows of the input matrix
                    with T.serial(0, SVF) as slice_idx:
                        for sub_tile_idx in range(0, sub_tile_count):
                            row_offset = SVF if sub_tile_idx >= (sub_tile_count // 2) else 0
                            col_offset = SVF if sub_tile_idx % 2 else 0

                            input_ptr = A.ptr_to([slice_idx + row_offset, col_offset])
                            sub_tile = T.int32(sub_tile_idx)
                            predicate = _create_active_lane_mask(
                                A, (row_offset + slice_idx, col_offset), cols
                            )
                            T.evaluate(
                                T.call_llvm_intrin(
                                    "llvm.aarch64.sme.ld1w.horiz",
                                    predicate,
                                    input_ptr,
                                    sub_tile,
                                    slice_idx,
                                    ty="void",
                                )
                            )

                    # Store columns to the output matrix
                    with T.serial(0, SVF) as slice_idx:
                        for sub_tile_idx in range(0, sub_tile_count):
                            col_offset = SVF if sub_tile_idx >= (sub_tile_count // 2) else 0
                            row_offset = SVF if sub_tile_idx % 2 else 0

                            output_ptr = A_t.ptr_to([slice_idx + row_offset, col_offset])
                            sub_tile = T.int32(sub_tile_idx)
                            predicate = _create_active_lane_mask(
                                A_t, (row_offset + slice_idx, col_offset), rows
                            )
                            T.evaluate(
                                T.call_llvm_intrin(
                                    "llvm.aarch64.sme.st1w.vert",
                                    predicate,
                                    output_ptr,
                                    sub_tile,
                                    slice_idx,
                                    ty="void",
                                )
                            )

        return ib.get()

    return desc, impl()


def get_sme_transpose_interleave_block2_2svl_fp16_intrin():
    # pylint: disable=line-too-long
    """
    Transpose and block pack a matrix of size 2SVL x 1SVL (where 'SVL' is the Scalable Vector
    Length for the fp16 datatype) using the Scalable Matrix Extension (SME).

    Rows of the fp16 input matrix are loaded into the accumulator tile and columns are stored
    as fp32 SVL length vectors to the output matrix. When loading, the accumulator tile is
    interpreted to be of shape 2 * 8 * vscale x 8 * vscale. When  storing, we interpret the
    accumulator tile to be of shape 2 * 4 * vscale x 2 * 4 * vscale.

    Example
    -------
    In the fp16 instance, the accumulator tile consists of two sub-tiles numbered 0-1. Rows
    of A are loaded onto the accumulator tile by interleaving rows in the first half (0, SVL//2]
    of the tile and rows in the second half (SVL//2, SVL]. Columns of fp32 values are stored
    into the output tensor. The fp32 store is used to group pairs of consecutive values together,
    resulting in the arrangement displayed below::

        A:                                Accumulator tile:
             +----------------+            +----------------+
             |-------0a-------|            |-------0a-------|
             |-------0b-------|            |-------0x-------|
             |      ...       |            |-------0b-------|            A_t:
             |-------0x-------|            |-------0y-------|             +------------------------------------------------+
             |-------0y-------|            |      ...       |             |0a.0 0a.1 0b.0 0b.1    | 1a.0 1a.1 1b.0 1b.1    |
             |      ...       | ld1h.horiz |                |  st1w.vert  |0x.0 0x.1 0y.0 0y.1    | 1x.0 1x.1 1y.0 1y.1    |
             |================|   ====>    |================|    ====>    |0a.2 0a.3 0b.2 0b.3 ...| 1a.2 1a.3 1b.2 1b.3 ...|
             |-------1a-------|            |-------1a-------|             |0x.2 0x.3 0y.2 0y.3    | 1x.2 1x.3 1y.2 1y.3    |
             |-------1b-------|            |-------1x-------|             |...  ...  ...  ...     | ...  ...  ...  ...     |
             |      ...       |            |-------1b-------|             +------------------------------------------------+
             |-------1x-------|            |-------1y-------|
             |-------1y-------|            |      ...       |
             |      ...       |            |                |
             +----------------+            +----------------+

    In the A_t output matrix in the diagram above, .x is used to denote the offset into the
    labelled row.

    Returns
    -------
    intrin : TensorIntrin
        The SME TensorIntrin that can be used in tensorizing a schedule.

    """
    # pylint: enable=line-too-long
    SVF = tirx.get_vscale_expr("float16")
    SVF2 = 2 * SVF

    desc_a_offset_1 = T.dynamic("desc_a_offset_1", "int32")
    desc_a_t_offset_1 = T.dynamic("desc_a_t_offset_1", "int32")

    @Ts.function
    def desc(
        A: T.Tensor(
            (SVF2, SVF), dtype="float16", layout=T.TileLayout(T.S[(SVF2, SVF)] + (desc_a_offset_1))
        ),
        A_t: T.Tensor(
            (SVF, SVF2),
            dtype="float16",
            layout=T.TileLayout(T.S[(SVF, SVF2)] + (desc_a_t_offset_1)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:SVF2, 0:SVF])
            Ts.writes(A_t[0:SVF, 0:SVF2])
            for k, m in T.grid(SVF, SVF2):
                with Ts.sblock("transpose"):
                    v_m, v_k = Ts.axis.remap("SS", [m, k])
                    A_t[v_k, v_m] = A[v_m, v_k]

    impl_a_offset = T.dynamic("impl_a_offset", "int32")
    impl_a_t_offset = T.dynamic("impl_a_t_offset", "int32")
    impl_a_stride = T.dynamic("impl_a_stride", "int32")
    impl_a_t_stride = T.dynamic("impl_a_t_stride", "int32")

    def impl():
        with IRBuilder() as ib:
            with build_function():
                A = T.arg_(
                    "a",
                    T.Tensor(
                        (SVF2, SVF),
                        "float16",
                        layout=T.TileLayout(
                            T.S[(SVF2, SVF) : [impl_a_stride, 1]] + (impl_a_offset)
                        ),
                    ),
                )
                A_t = T.arg_(
                    "a_t",
                    T.Tensor(
                        (SVF, SVF2),
                        "float16",
                        layout=T.TileLayout(
                            T.S[(SVF, SVF2) : [impl_a_t_stride, 1]] + (impl_a_t_offset)
                        ),
                    ),
                )

                ptrue_fp16 = _create_ptrue_mask("float16")
                ptrue_fp32 = _create_ptrue_mask("float32")

                with Ts.sblock("root"):
                    Ts.reads(A[0:SVF2, 0:SVF])
                    Ts.writes(A_t[0:SVF, 0:SVF2])

                    # Load rows of the input matrix
                    with T.serial(SVF // 2) as slice_idx:
                        for sub_tile_idx in range(2):
                            input_ptr = A.ptr_to([slice_idx + SVF * sub_tile_idx, 0])
                            T.evaluate(
                                T.call_llvm_intrin(
                                    "llvm.aarch64.sme.ld1h.horiz",
                                    ptrue_fp16,
                                    input_ptr,
                                    sub_tile_idx,
                                    slice_idx * 2,
                                    ty="void",
                                )
                            )
                            input_ptr = A.ptr_to([slice_idx + SVF * sub_tile_idx + SVF // 2, 0])
                            T.evaluate(
                                T.call_llvm_intrin(
                                    "llvm.aarch64.sme.ld1h.horiz",
                                    ptrue_fp16,
                                    input_ptr,
                                    sub_tile_idx,
                                    slice_idx * 2 + 1,
                                    ty="void",
                                )
                            )

                    # Store columns to the output matrix
                    with T.serial(SVF // 2) as slice_idx:
                        for sub_tile_idx in range(2):
                            output_ptr = A_t.ptr_to([slice_idx * 2, SVF * sub_tile_idx])
                            T.evaluate(
                                T.call_llvm_intrin(
                                    "llvm.aarch64.sme.st1w.vert",
                                    ptrue_fp32,
                                    output_ptr,
                                    sub_tile_idx,
                                    slice_idx,
                                    ty="void",
                                )
                            )
                            output_ptr = A_t.ptr_to([slice_idx * 2 + 1, SVF * sub_tile_idx])
                            T.evaluate(
                                T.call_llvm_intrin(
                                    "llvm.aarch64.sme.st1w.vert",
                                    ptrue_fp32,
                                    output_ptr,
                                    sub_tile_idx + 2,
                                    slice_idx,
                                    ty="void",
                                )
                            )

        return ib.get()

    return desc, impl()


def get_transpose_interleave_intrin_name(in_dtype, out_dtype, extent_cols, extent_rows):
    if in_dtype == "float32" and out_dtype == "float32":
        sme_transpose_interleave_intrin_name = (
            ARM_SME_2SVLx2SVL_FP32_TRANSPOSE_INTERLEAVE + f"_{extent_cols}_{extent_rows}"
        )
        TensorIntrin.register(
            sme_transpose_interleave_intrin_name,
            *get_sme_transpose_interleave_2svlx2svl_fp32_intrin(extent_cols, extent_rows),
            override=True,
        )
        return sme_transpose_interleave_intrin_name
    elif in_dtype == "float16" and out_dtype == "float32":
        return ARM_SME_BLOCK2_2SVLx1SVL_FP16_TRANSPOSE_INTERLEAVE
    else:
        raise ValueError("Input/output data type combination not supported.")


def get_sme_gemm_interleaved_mopa_2svlx2svl_intrin(M, K, in_dtype):
    """
    Compute a GEMM of size 2SVL x 2SVL (where 'SVL' is the Scalable Vector Length using
    outer product operations from the Scalable Matrix Extension (SME).

    The inputs A and B are expected to be of size K x 2SVL and produce a result C of
    size 2SVL x 2SVL.

    The SME accumulator tile is divided into sub-tiles, each of which is utilized to
    calculate the outer-product using columns / rows of A and B respectively. For each
    sub-tile, elements in the first column of input matrix A (accessed sequentially due
    to being transpose-interleaved) and first row of input matrix B are used to calculate
    an outer-product. This is then accumulated with the result of performing an
    outer-product on the second column and row of A and B respectively. This process is
    repeated K times. Finally, the results of the accumulation are stored.

    Note: The input tensor 'A' must be transpose-interleaved.

    Example
    -------

    Diagram showing outer-product performed on each of the accumulator sub-tiles
    for the fp32 datatype:

    ::

                           SVL           SVL
                    +----------------------------+
                    |       l     |       h      | K
                K   +----------------------------+
             +---+  +----------------------------+
             |   |  |  0:            1:          |-+
             |   |  |  mopa(l, l)    mopa(l, h)  | |-+
           l |   |  |                            | | |
             |   |  |                            | | |
             |---|  |                            | | |
             |   |  |  2:            3:          | | |
           h |   |  |  mopa(h, l)    mopa(h, h)  | | |
             |   |  |                            | | |
             |   |  |                            | | |
             +---+  +----------------------------+ | |
                      +----------------------------+ |
                         +---------------------------+
                                        (accumulate K times)

    Pseudo code computing 2SVL x 2SVL GEMM for fp32 inputs:

    .. code-block:: c

        // Number of fp32 elements in a scalable vector
        int SVF = SVL / 32;

        // Reset the accumulator tile
        sme.zero();

        // Calculate outer products and accumulate
        for (k = 0; k < K; k++) {
            float32xSVF A_row_0 = A[k][0];
            float32xSVF A_row_1 = A[k][SVF];
            float32xSVF B_row_0 = B[k][0];
            float32xSVF B_row_1 = B[k][SVF];

            float32xSVFxSVF sub_tile_0 += sme.mopa(A_row_0, B_row_0);
            float32xSVFxSVF sub_tile_1 += sme.mopa(A_row_0, B_row_1);
            float32xSVFxSVF sub_tile_2 += sme.mopa(A_row_1, B_row_0);
            float32xSVFxSVF sub_tile_3 += sme.mopa(A_row_1, B_row_1);
        }

        // Store the results of accumulation
        for (i = 0; i < SVF; i++) {
            C[i][0] = sme.horiz(sub_tile_0[i]);
            C[i][0] = sme.horiz(sub_tile_0[i + SVF]);
            C[i + SVF][0] = sme.horiz(sub_tile_0[i]);
            C[i + SVF][0] = sme.horiz(sub_tile_0[i + SVF]);
        }

    Notes:

    - Recall that A has been transposed beforehand such that each column is now accessed
      by row.
    - 'sme.zero' resets the accumulator tile to contain all zero's.
    - 'sme.mopa' is the outer product and accumulate intrinsic.
    - 'sme.horiz' stores rows of an accumulator sub-tile to memory.

    Returns
    -------
    intrin : TensorIntrin
        The SME TensorIntrin that can be used in tensorizing a schedule.

    """
    SVF = tirx.get_vscale_expr("float32")
    SVF2 = 2 * SVF
    fmopa_intrin = (
        "llvm.aarch64.sme.mopa" if in_dtype == "float32" else "llvm.aarch64.sme.mopa.wide"
    )

    desc_a_offset_2 = T.dynamic("desc_a_offset_2", "int32")
    desc_b_offset = T.dynamic("desc_b_offset", "int32")
    desc_c_offset = T.dynamic("desc_c_offset", "int32")

    @Ts.function
    def desc(
        A: T.Tensor(
            (K, SVF2), dtype=in_dtype, layout=T.TileLayout(T.S[(K, SVF2)] + (desc_a_offset_2))
        ),
        B: T.Tensor(
            (K, SVF2), dtype=in_dtype, layout=T.TileLayout(T.S[(K, SVF2)] + (desc_b_offset))
        ),
        C: T.Tensor(
            (SVF2, SVF2), dtype="float32", layout=T.TileLayout(T.S[(SVF2, SVF2)] + (desc_c_offset))
        ),
    ):
        with Ts.sblock("root"):
            Ts.reads(C[0:SVF2, 0:SVF2], A[0:K, 0:SVF2], B[0:K, 0:SVF2])
            Ts.writes(C[0:SVF2, 0:SVF2])
            for m, n, k in T.grid(SVF2, SVF2, K):
                with Ts.sblock("gemm"):
                    v_m, v_n, v_k = Ts.axis.remap("SSR", [m, n, k])
                    C[v_m, v_n] += T.Cast("float32", A[v_k, v_m]) * T.Cast("float32", B[v_k, v_n])

    impl_a_offset = T.dynamic("impl_a_offset", "int32")
    impl_b_offset = T.dynamic("impl_b_offset", "int32")
    impl_c_offset = T.dynamic("impl_c_offset", "int32")
    impl_a_stride = T.dynamic("impl_a_stride", "int32")
    impl_b_stride = T.dynamic("impl_b_stride", "int32")
    impl_c_stride = T.dynamic("impl_c_stride", "int32")

    def impl():
        sub_tile_count = 4

        with IRBuilder() as ib:
            with build_function():
                A = T.arg_(
                    "a",
                    T.Tensor(
                        (K, SVF2),
                        in_dtype,
                        layout=T.TileLayout(T.S[(K, SVF2) : [impl_a_stride, 1]] + (impl_a_offset)),
                    ),
                )
                B = T.arg_(
                    "b",
                    T.Tensor(
                        (K, SVF2),
                        in_dtype,
                        layout=T.TileLayout(T.S[(K, SVF2) : [impl_b_stride, 1]] + (impl_b_offset)),
                    ),
                )
                C = T.arg_(
                    "c",
                    T.Tensor(
                        (SVF2, SVF2),
                        "float32",
                        layout=T.TileLayout(
                            T.S[(SVF2, SVF2) : [impl_c_stride, 1]] + (impl_c_offset)
                        ),
                    ),
                )

                ptrue = _create_ptrue_mask(in_dtype)

                with Ts.sblock("root"):
                    Ts.reads(C[0:SVF2, 0:SVF2], A[0:K, 0:SVF2], B[0:K, 0:SVF2])
                    Ts.writes(C[0:SVF2, 0:SVF2])

                    # Iterate over the reduction axis applying outer product and accumulate
                    rows_per_iter = 1 if in_dtype == "float32" else 2
                    with T.serial(T.ceildiv(K, rows_per_iter)) as k:
                        k_row = k * rows_per_iter
                        in_dtype_svf = tirx.get_vscale_expr(in_dtype)

                        # Ideally we'd rely on predicating the loads and use the same predicate
                        # for the outer product operation. However, support for predicated
                        # tensors is not currently supported by multiple lowering passes such as
                        # "LowerMatchTensor", therefore the predicate is passed directly to the
                        # outer product operation for now.
                        if in_dtype == "float32":
                            a_low = (
                                T.TensorLoad(A, [k_row, T.Ramp(0, 1, in_dtype_svf)]),
                                _create_active_lane_mask(A, (k_row, 0), K),
                            )
                            b_low = (
                                T.TensorLoad(B, [k_row, T.Ramp(0, 1, in_dtype_svf)]),
                                _create_active_lane_mask(B, (k_row, 0), K),
                            )
                            a_high = (
                                T.TensorLoad(A, [k_row, T.Ramp(in_dtype_svf, 1, in_dtype_svf)]),
                                _create_active_lane_mask(A, (k_row, in_dtype_svf), K),
                            )
                            b_high = (
                                T.TensorLoad(B, [k_row, T.Ramp(in_dtype_svf, 1, in_dtype_svf)]),
                                _create_active_lane_mask(B, (k_row, in_dtype_svf), K),
                            )
                        else:
                            a_low = (T.TensorLoad(A, [k_row, T.Ramp(0, 1, in_dtype_svf)]), ptrue)
                            b_low = (T.TensorLoad(B, [k_row, T.Ramp(0, 1, in_dtype_svf)]), ptrue)
                            a_high = (
                                T.TensorLoad(A, [k_row + 1, T.Ramp(0, 1, in_dtype_svf)]),
                                ptrue,
                            )
                            b_high = (
                                T.TensorLoad(B, [k_row + 1, T.Ramp(0, 1, in_dtype_svf)]),
                                ptrue,
                            )

                        input_combinations = [
                            (a_low, b_low),
                            (a_low, b_high),
                            (a_high, b_low),
                            (a_high, b_high),
                        ]
                        for sub_tile_idx in range(0, sub_tile_count):
                            sub_tile = T.int32(sub_tile_idx)
                            input_1 = input_combinations[sub_tile_idx][0]
                            input_2 = input_combinations[sub_tile_idx][1]

                            T.evaluate(
                                T.call_llvm_intrin(
                                    fmopa_intrin,
                                    sub_tile,
                                    input_1[1],
                                    input_2[1],
                                    input_1[0],
                                    input_2[0],
                                    ty="void",
                                )
                            )

                    # Store the accumulated tile results
                    with T.serial(SVF) as slice_idx:
                        for sub_tile_idx in range(sub_tile_count):
                            vert_offset = SVF if sub_tile_idx >= (sub_tile_count // 2) else 0
                            horiz_offset = SVF if sub_tile_idx % 2 else 0
                            output_ptr = C.ptr_to([slice_idx + vert_offset, horiz_offset])

                            T.evaluate(
                                T.call_llvm_intrin(
                                    "llvm.aarch64.sme.st1w.horiz",
                                    _create_active_lane_mask(
                                        C, (vert_offset + slice_idx, horiz_offset), M
                                    ),
                                    output_ptr,
                                    T.int32(sub_tile_idx),
                                    T.int32(slice_idx),
                                    ty="void",
                                )
                            )

            return ib.get()

    return desc, impl()


def get_sme_init_intrin():
    """
    Reset the entire matrix tile storage to 0.
    """
    SVF2 = 2 * 4 * T.vscale()

    desc_c_offset_1 = T.dynamic("desc_c_offset_1", "int32")

    @Ts.function
    def desc(
        C: T.Tensor(
            (SVF2, SVF2), "float32", layout=T.TileLayout(T.S[(SVF2, SVF2)] + (desc_c_offset_1))
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes(C[0:SVF2, 0:SVF2])
            for m, n in T.grid(SVF2, SVF2):
                with Ts.sblock("init"):
                    v_m, v_n = Ts.axis.remap("SS", [m, n])
                    C[v_m, v_n] = T.float32(0)

    impl_c_offset = T.dynamic("impl_c_offset", "int32")

    @Ts.function
    def impl(
        C: T.Tensor(
            (SVF2, SVF2), "float32", layout=T.TileLayout(T.S[(SVF2, SVF2)] + (impl_c_offset))
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes(C[0:SVF2, 0:SVF2])
            clear_all_tiles = T.int32(255)
            T.evaluate(T.call_llvm_intrin("llvm.aarch64.sme.zero", clear_all_tiles, ty="void"))

    return desc, impl


ARM_DOT_4x4_i8_NEON_INTRIN = "dot_4x4_i8i8s32_neon"
ARM_DOT_4x4_i8_SDOT_INTRIN = "dot_4x4_i8i8s32_sdot"
ARM_DOT_4x4_u8_UDOT_INTRIN = "dot_4x4_u8u8u32_udot"
ARM_DOT_4x4_u8_HDOT_INTRIN = "dot_4x4_u8u8i32_hdot"

TensorIntrin.register(ARM_DOT_4x4_i8_NEON_INTRIN, neon_4x4_i8i8i32_desc, neon_4x4_i8i8i32_impl)
TensorIntrin.register(ARM_DOT_4x4_i8_SDOT_INTRIN, *get_dotprod_intrin("int8", "int32"))
TensorIntrin.register(ARM_DOT_4x4_u8_UDOT_INTRIN, *get_dotprod_intrin("uint8", "uint32"))
TensorIntrin.register(ARM_DOT_4x4_u8_HDOT_INTRIN, *get_dotprod_intrin("uint8", "int32"))

ARM_SME_INIT = "sme_init"
ARM_SME_2SVLx2SVL_FP32_TRANSPOSE_INTERLEAVE = "sme_2svlx2svl_fp32_transpose_interleave"
ARM_SME_BLOCK2_2SVLx1SVL_FP16_TRANSPOSE_INTERLEAVE = (
    "sme_block2_2svlx1svl_fp16_transpose_interleave"
)
ARM_SME_2SVLx2SVL_GEMM_INTERLEAVED_MOPA = "sme_2svlx2svl_gemm_interleaved_mopa"

# The following tensor intrinsics use LLVM intrinsics that are only available
# in versions of LLVM >= 15. Installations with older versions of LLVM will
# not be able to use them.
if llvm.version_major() >= 15:
    TensorIntrin.register(
        ARM_SME_BLOCK2_2SVLx1SVL_FP16_TRANSPOSE_INTERLEAVE,
        *get_sme_transpose_interleave_block2_2svl_fp16_intrin(),
    )
    TensorIntrin.register(ARM_SME_INIT, *get_sme_init_intrin())
