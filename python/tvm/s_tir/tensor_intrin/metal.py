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
# pylint: disable=invalid-name,missing-function-docstring,unused-variable
"""Intrinsics for tensorization on Apple GPU."""

from typing import Literal

from tvm.s_tir import TensorIntrin, match_shape_stride_pattern
from tvm.s_tir import script as Ts
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx import Expr, Function, Var

_SCOPE_GLOBAL = MemoryScope("global")
_SCOPE_SHARED = MemoryScope("shared")

######## simdgroup matrix intrinsics ########


def get_simdgroup_index(tensor: Var, stride: Expr, col: int, row: int):
    """Compute simdgroup index using elem_offset of the tensor"""

    # NOTE: Need further check the usage between `col`` and `row`
    # Currently, Metal only supports 8x8, which means the values of `col` and `row` are the same
    tensor_pattern = match_shape_stride_pattern(tensor.ty)
    assert tensor_pattern is not None
    frag_index_m = tensor_pattern.elem_offset // stride // col
    frag_index_n = tensor_pattern.elem_offset % stride // row

    num_fragments_per_row = stride // row
    return frag_index_m * num_fragments_per_row + frag_index_n


def get_make_filled_simdgroup_matrix_intrin(
    dtype: str, col: int = 8, row: int = 8
) -> tuple[Function, Function]:
    desc_a_offset = T.dynamic("desc_a_offset", "int32")

    @Ts.function
    def desc(
        A: T.Tensor(
            (col, row),
            dtype,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(col, row)] + (desc_a_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes(A[0:col, 0:row])
            for i, j in T.grid(col, row):
                with Ts.sblock("init"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    A[vi, vj] = T.float32(0)

    d0 = T.dynamic("d0", "int32")
    d1 = T.dynamic("d1", "int32")

    impl_a_offset = T.dynamic("impl_a_offset", "int32")

    @Ts.function
    def impl(
        A: T.Tensor(
            (col, row),
            dtype,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(col, row) : [d1, d0]] + (impl_a_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes(A[0:col, 0:row])
            T.metal.make_filled_simdgroup_matrix(
                A.data,
                index=get_simdgroup_index(A, d1, col, row),
                value=T.float32(0),
                col=col,
                row=row,
            )

    return desc, impl


def get_simdgroup_load_intrin(
    dtype: str,
    scope: Literal["global", "shared"],
    col: int = 8,
    row: int = 8,
    transpose_matrix: bool = False,
) -> tuple[Function, Function]:
    align = col * row

    desc_a_offset_1 = T.dynamic("desc_a_offset_1", "int32")
    desc_c_offset = T.dynamic("desc_c_offset", "int32")

    @Ts.function
    def desc(
        A: T.Tensor(
            (col, row),
            dtype,
            align=align,
            scope=scope,
            layout=T.TileLayout(T.S[(col, row)] + (desc_a_offset_1)),
        ),
        C: T.Tensor(
            (col, row),
            dtype,
            align=align,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(col, row)] + (desc_c_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:col, 0:row])
            Ts.writes(C[0:col, 0:row])
            for i, j in T.grid(col, row):
                with Ts.sblock("load"):
                    vii, vjj = Ts.axis.remap("SS", [i, j])
                    if T.constexpr(transpose_matrix):
                        # C[vii, vjj] = A[vjj, vii]
                        C[vjj, vii] = A[vii, vjj]
                    else:
                        C[vii, vjj] = A[vii, vjj]

    s0 = T.dynamic("s0", "int32")
    s1 = T.dynamic("s1", "int32")
    d0 = T.dynamic("d0", "int32")
    d1 = T.dynamic("d1", "int32")

    impl_a_offset_1 = T.dynamic("impl_a_offset_1", "int32")
    impl_c_offset = T.dynamic("impl_c_offset", "int32")

    @Ts.function
    def impl(
        A: T.Tensor(
            (col, row),
            dtype,
            align=align,
            scope=scope,
            layout=T.TileLayout(T.S[(col, row) : [s1, s0]] + (impl_a_offset_1)),
        ),
        C: T.Tensor(
            (col, row),
            dtype,
            align=align,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(col, row) : [d1, d0]] + (impl_c_offset)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:col, 0:row])
            Ts.writes(C[0:col, 0:row])
            T.metal.simdgroup_load(
                C.data,
                index=get_simdgroup_index(C, d1, col, row),
                ptr=A.ptr_to([0, 0]),
                stride=s1,
                col=col,
                row=row,
                transpose_matrix=transpose_matrix,
            )

    return desc, impl


def get_simdgroup_store_intrin(
    dtype: str,
    scope: Literal["global", "shared"],
    col: int = 8,
    row: int = 8,
    transpose_matrix: bool = False,
) -> tuple[Function, Function]:
    align = col * row

    desc_a_offset_2 = T.dynamic("desc_a_offset_2", "int32")
    desc_c_offset_1 = T.dynamic("desc_c_offset_1", "int32")

    @Ts.function
    def desc(
        A: T.Tensor(
            (col, row),
            dtype,
            align=align,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(col, row)] + (desc_a_offset_2)),
        ),
        C: T.Tensor(
            (col, row),
            dtype,
            align=align,
            scope=scope,
            layout=T.TileLayout(T.S[(col, row)] + (desc_c_offset_1)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:col, 0:row])
            Ts.writes(C[0:col, 0:row])
            for i, j in T.grid(col, row):
                with Ts.sblock("store"):
                    vii, vjj = Ts.axis.remap("SS", [i, j])
                    if T.constexpr(transpose_matrix):
                        C[vjj, vii] = A[vii, vjj]
                    else:
                        C[vii, vjj] = A[vii, vjj]

    s0 = T.dynamic("s0", "int32")
    s1 = T.dynamic("s1", "int32")
    d0 = T.dynamic("d0", "int32")
    d1 = T.dynamic("d1", "int32")

    impl_a_offset_2 = T.dynamic("impl_a_offset_2", "int32")
    impl_c_offset_1 = T.dynamic("impl_c_offset_1", "int32")

    @Ts.function
    def impl(
        A: T.Tensor(
            (col, row),
            dtype,
            align=align,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(col, row) : [s1, s0]] + (impl_a_offset_2)),
        ),
        C: T.Tensor(
            (col, row),
            dtype,
            align=align,
            scope=scope,
            layout=T.TileLayout(T.S[(col, row) : [d1, d0]] + (impl_c_offset_1)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(A[0:col, 0:row])
            Ts.writes(C[0:col, 0:row])
            T.metal.simdgroup_store(
                A.data,
                index=get_simdgroup_index(A, s1, col, row),
                ptr=C.ptr_to([0, 0]),
                stride=d1,
                col=col,
                row=row,
                transpose_matrix=transpose_matrix,
            )

    return desc, impl


def get_simdgroup_multiply_accumulate_intrin(
    m_dim: int, n_dim: int, k_dim: int, dtype: str
) -> tuple[Function, Function]:
    desc_a_offset_3 = T.dynamic("desc_a_offset_3", "int32")
    desc_b_offset = T.dynamic("desc_b_offset", "int32")
    desc_c_offset_2 = T.dynamic("desc_c_offset_2", "int32")

    @Ts.function
    def desc(
        A: T.Tensor(
            (m_dim, k_dim),
            dtype,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(m_dim, k_dim)] + (desc_a_offset_3)),
        ),
        B: T.Tensor(
            (k_dim, n_dim),
            dtype,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(k_dim, n_dim)] + (desc_b_offset)),
        ),
        C: T.Tensor(
            (m_dim, n_dim),
            dtype,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(m_dim, n_dim)] + (desc_c_offset_2)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:k_dim, 0:n_dim])
            Ts.writes(C[0:m_dim, 0:n_dim])
            for i, j, k in T.grid(m_dim, n_dim, k_dim):
                with Ts.sblock(""):
                    vii, vjj, vkk = Ts.axis.remap("SSR", [i, j, k])
                    C[vii, vjj] += A[vii, vkk] * B[vkk, vjj]

    a0 = T.dynamic("a0", "int32")
    a1 = T.dynamic("a1", "int32")
    b0 = T.dynamic("b0", "int32")
    b1 = T.dynamic("b1", "int32")
    c0 = T.dynamic("c0", "int32")
    c1 = T.dynamic("c1", "int32")

    impl_a_offset_3 = T.dynamic("impl_a_offset_3", "int32")
    impl_b_offset = T.dynamic("impl_b_offset", "int32")
    impl_c_offset_2 = T.dynamic("impl_c_offset_2", "int32")

    @Ts.function
    def impl(
        A: T.Tensor(
            (m_dim, k_dim),
            dtype,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(m_dim, k_dim) : [a1, a0]] + (impl_a_offset_3)),
        ),
        B: T.Tensor(
            (k_dim, n_dim),
            dtype,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(k_dim, n_dim) : [b1, b0]] + (impl_b_offset)),
        ),
        C: T.Tensor(
            (m_dim, n_dim),
            dtype,
            scope="metal.simdgroup",
            layout=T.TileLayout(T.S[(m_dim, n_dim) : [c1, c0]] + (impl_c_offset_2)),
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:m_dim, 0:n_dim], A[0:m_dim, 0:k_dim], B[0:k_dim, 0:n_dim])
            Ts.writes(C[0:m_dim, 0:n_dim])
            T.metal.simdgroup_multiply_accumulate(
                C.data,
                get_simdgroup_index(C, c1, m_dim, n_dim),
                A.data,
                get_simdgroup_index(A, a1, m_dim, k_dim),
                B.data,
                get_simdgroup_index(B, b1, k_dim, n_dim),
                C.data,
                get_simdgroup_index(C, c1, m_dim, n_dim),
            )

    return desc, impl


# Make filled simdgroup matrix intrinsics

SIMDGROUP_MAKE_FILLED_8x8x8_f16_INTRIN = "simdgroup_make_filled_8x8x8_f16"
TensorIntrin.register(
    SIMDGROUP_MAKE_FILLED_8x8x8_f16_INTRIN,
    *get_make_filled_simdgroup_matrix_intrin("float16", 8, 8),
)

SIMDGROUP_FILLED_8x8x8_f32_INTRIN = "simdgroup_fill_8x8x8_f32"
TensorIntrin.register(
    SIMDGROUP_FILLED_8x8x8_f32_INTRIN, *get_make_filled_simdgroup_matrix_intrin("float32", 8, 8)
)

SIMDGROUP_FILLED_8x8x8_bf16_INTRIN = "simdgroup_fill_8x8x8_bf16"
TensorIntrin.register(
    SIMDGROUP_FILLED_8x8x8_bf16_INTRIN, *get_make_filled_simdgroup_matrix_intrin("bfloat16", 8, 8)
)

# Load intrinsics

SIMDGROUP_LOAD_8x8x8_f16_SHARED_INTRIN = "simdgroup_load_8x8x8_f16_shared"
TensorIntrin.register(
    SIMDGROUP_LOAD_8x8x8_f16_SHARED_INTRIN,
    *get_simdgroup_load_intrin("float16", "shared", 8, 8, False),
)

SIMDGROUP_LOAD_8x8x8_f16_SHARED_TRANS_INTRIN = "simdgroup_load_8x8x8_f16_shared_trans"
TensorIntrin.register(
    SIMDGROUP_LOAD_8x8x8_f16_SHARED_TRANS_INTRIN,
    *get_simdgroup_load_intrin("float16", "shared", 8, 8, True),
)

# Store intrinsics

SIMDGROUP_STORE_8x8x8_f16_GLOBAL_INTRIN = "simdgroup_store_8x8x8_f16_global"
TensorIntrin.register(
    SIMDGROUP_STORE_8x8x8_f16_GLOBAL_INTRIN,
    *get_simdgroup_store_intrin("float16", "global", 8, 8, False),
)

SIMDGROUP_STORE_8x8x8_f16_SHARED_INTRIN = "simdgroup_store_8x8x8_f16_shared"
TensorIntrin.register(
    SIMDGROUP_STORE_8x8x8_f16_SHARED_INTRIN,
    *get_simdgroup_store_intrin("float16", "shared", 8, 8, False),
)
# Multiply accumulate intrinsics

SIMDGROUP_MULTI_ACC_8x8x8_f16_INTRIN = "simdgroup_multiply_accumulate_8x8x8_f16"
TensorIntrin.register(
    SIMDGROUP_MULTI_ACC_8x8x8_f16_INTRIN,
    *get_simdgroup_multiply_accumulate_intrin(8, 8, 8, "float16"),
)


def get_simdgroup_intrin_group(
    load_scope: Literal["shared"],
    store_scope: Literal["global", "shared"],
    dtype: str,
    trans_a: bool = False,
    trans_b: bool = False,
) -> dict[str, str]:
    """Get a group of intrinsics for tensorization on Apple GPU.

    Parameters
    ----------
    load_scope : Literal["shared"]
        The memory scope of the input tensor.

    store_scope : Literal["global", "shared"]
        The memory scope of the result tensor.

    dtype : str
        The data type of the input and output tensors.

    trans_a : bool
        Whether the input matrix A is transposed.

    trans_b : bool
        Whether the input matrix B is transposed.

    Returns
    -------
    ret : Dict[str, str]
        A group of tensor intrinsics.
    """
    load_scope = MemoryScope(load_scope)
    store_scope = MemoryScope(store_scope)
    assert load_scope in [_SCOPE_SHARED]
    assert store_scope in [_SCOPE_GLOBAL, _SCOPE_SHARED]
    assert dtype in ["float16", "bfloat16", "float32"]

    shape = "8x8x8"
    dtype = "f16" if dtype == "float16" else "bf16" if dtype == "bfloat16" else "f32"
    trans_a = "_trans" if trans_a else ""
    trans_b = "_trans" if trans_b else ""

    # e.g. simdgroup_load_8x8x8_f16_shared
    load_a_intrin = f"simdgroup_load_{shape}_{dtype}_{load_scope.name}{trans_a}"
    # e.g. simdgroup_load_8x8x8_f16_shared_trans
    load_b_intrin = f"simdgroup_load_{shape}_{dtype}_{load_scope.name}{trans_b}"
    # e.g. simdgroup_multiply_accumulate_8x8x8_f16
    compute_intrin = f"simdgroup_multiply_accumulate_{shape}_{dtype}"
    # e.g. simdgroup_make_filled_8x8x8_f16
    init_intrin = f"simdgroup_make_filled_{shape}_{dtype}"
    # e.g. simdgroup_store_8x8x8_f16_global
    store_intrin = f"simdgroup_store_{shape}_{dtype}_{store_scope.name}"

    return {
        "init": init_intrin,
        "load_a": load_a_intrin,
        "load_b": load_b_intrin,
        "compute": compute_intrin,
        "store": store_intrin,
    }
