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

from __future__ import annotations

import pytest

import tvm
import tvm.s_tir
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import match_shape_stride_pattern
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


def _check(original, transformed):
    mod = tvm_tirx.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LowerMatchTensor()(mod)
    mod = tvm.s_tir.transform.StmtSimplify()(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"))


def _check_fail(original):
    mod = tvm_tirx.IRModule.from_expr(original)
    with pytest.raises(RuntimeError):
        mod = tvm.s_tir.transform.LowerMatchTensor()(mod)


tensor_load_store_sub_a_offset = T.dynamic("tensor_load_store_sub_a_offset", "int32")
tensor_load_store_sub_c_offset = T.dynamic("tensor_load_store_sub_c_offset", "int32")


@Ts.function
def tensor_load_store(
    A: T.Tensor((16, 16, 16), "float32"), C: T.Tensor((16, 16), "float32")
) -> None:
    for i, j, k in T.grid(4, 16, 8):
        with Ts.sblock():
            Ts.reads(C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2])
            Ts.writes(A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2])
            sub_A = Ts.match_tensor(
                A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2],
                (4, 1, 2),
                layout=T.TileLayout(
                    T.S[(4, 1, 2) : (256, 16, 1)] + (tensor_load_store_sub_a_offset)
                ),
            )
            sub_C = Ts.match_tensor(
                C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2],
                (4, 2),
                layout=T.TileLayout(T.S[(4, 2) : (16, 1)] + (tensor_load_store_sub_c_offset)),
            )
            for ii, kk in T.grid(4, 2):
                sub_A[ii, 0, kk] += sub_C[ii, kk]


@Ts.function
def transformed_tensor_load_store(
    A: T.Tensor((16, 16, 16), "float32"), C: T.Tensor((16, 16), "float32")
) -> None:
    for i, j, k in T.grid(4, 16, 8):
        with Ts.sblock():
            Ts.reads(C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2])
            Ts.writes(A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2])
            for ii, kk in T.grid(4, 2):
                A[i * 4 + ii, j, k * 2 + kk] += C[i * 4 + ii, k * 2 + kk]


# Dummy intrinsic whose arguments exercise match_tensor fields.  TVMScript
# evaluates the call eagerly (to 0), so it must NOT be registered as an op:
# registering "tirx.intrin_test" only leaves a category-less op in the tirx
# registry, breaking the exactly-one-category invariant for later tests.
def intrin_test(data, elem_offset, stride_0, stride_1, shape_0, shape_1):
    return 0


Bs_0 = T.dynamic("Bs_0", "int32")
Bs_1 = T.dynamic("Bs_1", "int32")


opaque_access_sub_a_offset = T.dynamic("opaque_access_sub_a_offset", "int32")
opaque_access_sub_b_offset = T.dynamic("opaque_access_sub_b_offset", "int32")


@Ts.function
def opaque_access(
    A: T.Tensor((32, 64, 128), "float32"), B: T.Tensor((64, 64, 64), "float32")
) -> None:
    for i, j, k in T.grid(2, 64, 8):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16])
            sub_A = Ts.match_tensor(
                A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16],
                (16, 1, 16),
                layout=T.TileLayout(
                    T.S[(16, 1, 16) : [8192, 128, 1]] + (opaque_access_sub_a_offset)
                ),
            )
            sub_A_pattern = T.meta_var(match_shape_stride_pattern(sub_A.ty))
            T.evaluate(
                intrin_test(
                    sub_A.data,
                    sub_A_pattern.elem_offset,
                    sub_A_pattern.strides[0],
                    sub_A_pattern.strides[1],
                    sub_A.shape[0],
                    sub_A.shape[1],
                )
            )
    for i, j, k in T.grid(64, 2, 8):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8])
            sub_B = Ts.match_tensor(
                B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8],
                (32, 8),
                layout=T.TileLayout(T.S[(32, 8) : [Bs_0, Bs_1]] + (opaque_access_sub_b_offset)),
            )
            sub_B_pattern = T.meta_var(match_shape_stride_pattern(sub_B.ty))
            T.evaluate(
                intrin_test(
                    sub_B.data,
                    sub_B_pattern.elem_offset,
                    sub_B_pattern.strides[0],
                    sub_B_pattern.strides[1],
                    sub_B.shape[0],
                    sub_B.shape[1],
                )
            )


@Ts.function
def transformed_opaque_access(
    A: T.Tensor((32, 64, 128), "float32"), B: T.Tensor((64, 64, 64), "float32")
) -> None:
    for i, j, k in T.grid(2, 64, 8):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16])
            T.evaluate(
                intrin_test(
                    A.data,
                    i * 131072 + j * 128 + k * 16,
                    8192,
                    128,
                    16,
                    1,
                )
            )
    for i, j, k in T.grid(64, 2, 8):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8])
            T.evaluate(
                intrin_test(
                    B.data,
                    i * 4096 + j * 2048 + k * 8,
                    64,
                    1,
                    32,
                    8,
                )
            )


opaque_tensor_data_projection_sub_a_offset = T.dynamic(
    "opaque_tensor_data_projection_sub_a_offset", "int32"
)


@Ts.function
def opaque_tensor_data_projection(A: T.Tensor((16,), "float32")) -> None:
    with Ts.sblock():
        Ts.reads([])
        Ts.writes(A[4:8])
        sub_A = Ts.match_tensor(
            A[4:8],
            (4,),
            layout=T.TileLayout(T.S[(4,)] + (opaque_tensor_data_projection_sub_a_offset)),
        )
        sub_A_pattern = T.meta_var(match_shape_stride_pattern(sub_A.ty))
        T.evaluate(T.call_extern("consume", sub_A.data, sub_A_pattern.elem_offset, ty="int32"))


@Ts.function
def transformed_opaque_tensor_data_projection(A: T.Tensor((16,), "float32")) -> None:
    with Ts.sblock():
        Ts.reads([])
        Ts.writes(A[4:8])
        T.evaluate(T.call_extern("consume", A.data, 4, ty="int32"))


As_0 = T.dynamic("As_0", "int32")
As_1 = T.dynamic("As_1", "int32")


high_dim_opaque_access_sub_a_offset = T.dynamic("high_dim_opaque_access_sub_a_offset", "int32")


@Ts.function
def high_dim_opaque_access(A: T.Tensor((16, 32, 64), "float32")) -> None:
    for i, j, k in T.grid(16, 2, 4):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
            sub_A = Ts.match_tensor(
                A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                (16, 16),
                layout=T.TileLayout(
                    T.S[(16, 16) : [As_0, As_1]] + (high_dim_opaque_access_sub_a_offset)
                ),
            )
            sub_A_pattern = T.meta_var(match_shape_stride_pattern(sub_A.ty))
            T.evaluate(
                intrin_test(
                    sub_A.data,
                    sub_A_pattern.elem_offset,
                    sub_A_pattern.strides[0],
                    sub_A_pattern.strides[1],
                    sub_A.shape[0],
                    sub_A.shape[1],
                )
            )


@Ts.function
def transformed_high_dim_opaque_access(A: T.Tensor((16, 32, 64), "float32")) -> None:
    for i, j, k in T.grid(16, 2, 4):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
            T.evaluate(
                intrin_test(
                    A.data,
                    i * 2048 + j * 1024 + k * 16,
                    64,
                    1,
                    16,
                    16,
                )
            )


As_0 = T.dynamic("As_0", "int32")
As_1 = T.dynamic("As_1", "int32")


high_dim_opaque_access_with_source_strides_sub_a_offset = T.dynamic(
    "high_dim_opaque_access_with_source_strides_sub_a_offset", "int32"
)


@Ts.function
def high_dim_opaque_access_with_source_strides(
    A: T.Tensor((16, 32, 64), "float32", layout=T.TileLayout(T.S[(16, 32, 64) : [2576, 80, 1]])),
) -> None:
    for i, j, k in T.grid(16, 2, 4):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
            sub_A = Ts.match_tensor(
                A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                (16, 16),
                layout=T.TileLayout(
                    T.S[(16, 16) : [As_0, As_1]]
                    + (high_dim_opaque_access_with_source_strides_sub_a_offset)
                ),
            )
            sub_A_pattern = T.meta_var(match_shape_stride_pattern(sub_A.ty))
            T.evaluate(
                intrin_test(
                    sub_A.data,
                    sub_A_pattern.elem_offset,
                    sub_A_pattern.strides[0],
                    sub_A_pattern.strides[1],
                    sub_A.shape[0],
                    sub_A.shape[1],
                )
            )


@Ts.function
def transformed_high_dim_opaque_access_with_source_strides(
    A: T.Tensor((16, 32, 64), "float32", layout=T.TileLayout(T.S[(16, 32, 64) : [2576, 80, 1]])),
) -> None:
    for i, j, k in T.grid(16, 2, 4):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
            T.evaluate(
                intrin_test(
                    A.data,
                    i * 2576 + j * 1280 + k * 16,
                    80,
                    1,
                    16,
                    16,
                )
            )


As_0 = T.dynamic("As_0", "int32")
As_1 = T.dynamic("As_1", "int32")
Ass_0 = T.dynamic("Ass_0", "int32")
Ass_1 = T.dynamic("Ass_1", "int32")


recursive_match_sub_a_offset = T.dynamic("recursive_match_sub_a_offset", "int32")
recursive_match_sub_b_offset = T.dynamic("recursive_match_sub_b_offset", "int32")
recursive_match_sub_sub_a_offset = T.dynamic("recursive_match_sub_sub_a_offset", "int32")
recursive_match_sub_sub_b_offset = T.dynamic("recursive_match_sub_sub_b_offset", "int32")


@Ts.function
def recursive_match(
    A: T.Tensor((64, 64, 64), "float32"), B: T.Tensor((64, 64, 64), "float32")
) -> None:
    for i, j, k in T.grid(64, 4, 4):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(
                [
                    A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                    B[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                ]
            )
            sub_A = Ts.match_tensor(
                A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                (16, 16),
                layout=T.TileLayout(T.S[(16, 16) : [As_0, As_1]] + (recursive_match_sub_a_offset)),
            )
            sub_B = Ts.match_tensor(
                B[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                (16, 16),
                layout=T.TileLayout(T.S[(16, 16) : (64, 1)] + (recursive_match_sub_b_offset)),
            )
            for jj, kk in T.grid(4, 4):
                with Ts.sblock():
                    Ts.reads([])
                    Ts.writes(
                        [
                            sub_A[jj * 4 : jj * 4 + 4, kk * 4 : kk * 4 + 4],
                            sub_B[jj * 4 : jj * 4 + 4, kk * 4 : kk * 4 + 4],
                        ]
                    )
                    sub_sub_A = Ts.match_tensor(
                        sub_A[jj * 4 : jj * 4 + 4, kk * 4 : kk * 4 + 4],
                        (4, 4),
                        layout=T.TileLayout(
                            T.S[(4, 4) : [Ass_0, Ass_1]] + (recursive_match_sub_sub_a_offset)
                        ),
                    )
                    sub_sub_B = Ts.match_tensor(
                        sub_B[jj * 4 : jj * 4 + 4, kk * 4 : kk * 4 + 4],
                        (4, 4),
                        layout=T.TileLayout(
                            T.S[(4, 4) : (64, 1)] + (recursive_match_sub_sub_b_offset)
                        ),
                    )
                    sub_sub_A_pattern = T.meta_var(match_shape_stride_pattern(sub_sub_A.ty))
                    T.evaluate(
                        intrin_test(
                            sub_sub_A.data,
                            sub_sub_A_pattern.elem_offset,
                            sub_sub_A_pattern.strides[0],
                            sub_sub_A_pattern.strides[1],
                            sub_sub_A.shape[0],
                            sub_sub_A.shape[1],
                        )
                    )
                    for jjj, kkk in T.grid(4, 4):
                        sub_sub_B[jjj, kkk] = 1


@Ts.function
def transformed_recursive_match(
    A: T.Tensor((64, 64, 64), "float32"), B: T.Tensor((64, 64, 64), "float32")
) -> None:
    for i, j, k in T.grid(64, 4, 4):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(
                [
                    A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                    B[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                ]
            )
            for jj, kk in T.grid(4, 4):
                with Ts.sblock():
                    Ts.reads([])
                    Ts.writes(
                        [
                            A[
                                i,
                                j * 16 + jj * 4 : j * 16 + jj * 4 + 4,
                                k * 16 + kk * 4 : k * 16 + kk * 4 + 4,
                            ],
                            B[
                                i,
                                j * 16 + jj * 4 : j * 16 + jj * 4 + 4,
                                k * 16 + kk * 4 : k * 16 + kk * 4 + 4,
                            ],
                        ]
                    )
                    T.evaluate(
                        intrin_test(
                            A.data,
                            i * 4096 + j * 1024 + jj * 256 + k * 16 + kk * 4,
                            64,
                            1,
                            4,
                            4,
                        )
                    )
                    for jjj, kkk in T.grid(4, 4):
                        B[i, j * 16 + jj * 4 + jjj, k * 16 + kk * 4 + kkk] = 1


Bs_0 = T.dynamic("Bs_0", "int32")
Bs_1 = T.dynamic("Bs_1", "int32")


symbolic_match_sub_a_offset = T.dynamic("symbolic_match_sub_a_offset", "int32")
symbolic_match_sub_b_offset = T.dynamic("symbolic_match_sub_b_offset", "int32")


@Ts.function
def symbolic_match(
    A: T.Tensor((n * m, m), "float32"),  # noqa: F821
    B: T.Tensor((n * 2, m * 4), "float32"),  # noqa: F821
    n: T.int32,
    m: T.int32,
) -> None:
    for i in range(0, n):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes([A[i * m : i * m + n, 0:m], B[i * n : i * n + 2, 0 : m * 4]])
            sub_A = Ts.match_tensor(
                A[i * m : i * m + m, 0:m],
                (m, m),
                layout=T.TileLayout(T.S[(m, m)] + (symbolic_match_sub_a_offset)),
            )
            sub_B = Ts.match_tensor(
                B[i * n : i * n + 2, 0 : m * 4],
                (2, m * 4),
                layout=T.TileLayout(T.S[(2, m * 4) : [Bs_0, Bs_1]] + (symbolic_match_sub_b_offset)),
            )
            for ii, jj in T.grid(m, m):
                sub_A[ii, jj] = 1
            for j in range(0, 4):
                sub_B_pattern = T.meta_var(match_shape_stride_pattern(sub_B.ty))
                T.evaluate(
                    intrin_test(
                        sub_B.data,
                        sub_B_pattern.elem_offset,
                        sub_B_pattern.strides[0],
                        sub_B_pattern.strides[1],
                        sub_B.shape[0],
                        sub_B.shape[1],
                    )
                )


@Ts.function
def transformed_symbolic_match(
    A: T.Tensor((n * m, m), "float32"),  # noqa: F821
    B: T.Tensor((n * 2, m * 4), "float32"),  # noqa: F821
    n: T.int32,
    m: T.int32,
) -> None:
    for i in range(0, n):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes([A[i * m : i * m + n, 0:m], B[i * n : i * n + 2, 0 : m * 4]])
            for ii, jj in T.grid(m, m):
                A[i * m + ii, jj] = 1
            for j in range(0, 4):
                T.evaluate(
                    intrin_test(
                        B.data,
                        i * n * (m * 4),
                        m * 4,
                        1,
                        2,
                        m * 4,
                    )
                )


rank0_tensor_sub_a_offset = T.dynamic("rank0_tensor_sub_a_offset", "int32")
rank0_tensor_sub_b_offset = T.dynamic("rank0_tensor_sub_b_offset", "int32")


@Ts.function
def rank0_tensor(A: T.Tensor((8, 8), "float32"), B: T.Tensor((8, 8), "float32")) -> None:
    for i, j in T.grid(8, 8):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes([A[i, j], B[i, j]])
            sub_A = Ts.match_tensor(
                A[i, j], (), layout=T.TileLayout(T.S[()] + (rank0_tensor_sub_a_offset))
            )
            sub_B = Ts.match_tensor(
                B[i, j], (), layout=T.TileLayout(T.S[()] + (rank0_tensor_sub_b_offset))
            )
            sub_A[()] = 1
            sub_B_pattern = T.meta_var(match_shape_stride_pattern(sub_B.ty))
            T.evaluate(
                intrin_test(
                    sub_B.data,
                    sub_B_pattern.elem_offset,
                    0,
                    0,
                    0,
                    0,
                )
            )


@Ts.function
def transformed_rank0_tensor(
    A: T.Tensor((8, 8), "float32"), B: T.Tensor((8, 8), "float32")
) -> None:
    for i, j in T.grid(8, 8):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes([A[i, j], B[i, j]])
            A[i, j] = 1
            T.evaluate(
                intrin_test(
                    B.data,
                    i * 8 + j,
                    0,
                    0,
                    0,
                    0,
                )
            )


@Ts.function
def fail_match_load(A: T.Tensor((8, 8), "float32")) -> None:
    for i, j in T.grid(8, 8):
        with Ts.sblock():
            Ts.reads(A[i, j])
            Ts.writes([])
            sub_A = Ts.match_tensor(A[i, j], ())
            T.evaluate(sub_A[()])


@Ts.function
def fail_match_store(A: T.Tensor((8, 8), "float32")) -> None:
    for i, j in T.grid(8, 8):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i, j])
            sub_A = Ts.match_tensor(A[i, j], ())
            sub_A[()] = 1


# well-formed checker complains about redefinition of a stride variable
stride = T.dynamic("stride", "int32")


fail_tensor_bind_sub_a_offset = T.dynamic("fail_tensor_bind_sub_a_offset", "int32")


@Ts.function(check_well_formed=False)
def fail_tensor_bind(A: T.Tensor((8, 8), "float32")) -> None:
    for i, j in T.grid(8, 2):
        with Ts.sblock():
            sub_A = Ts.match_tensor(
                A[i, j * 4 : j * 4 + 4],
                (1, 4),
                layout=T.TileLayout(
                    T.S[(1, 4) : [stride, stride]] + (fail_tensor_bind_sub_a_offset)
                ),
            )
            for jj in range(0, 4):
                sub_A[i, j * 4 + jj] = 1


# well-formed checker complains about redefinition of a stride variable
fail_match_func_param_sub_a_offset = T.dynamic("fail_match_func_param_sub_a_offset", "int32")


@Ts.function(check_well_formed=False)
def fail_match_func_param(A: T.Tensor((8, 8), "float32"), m: T.int32, n: T.int32) -> None:
    for i, j in T.grid(8, 2):
        with Ts.sblock():
            sub_A = Ts.match_tensor(
                A[i, j * 4 : j * 4 + 4],
                (1, 4),
                layout=T.TileLayout(T.S[(1, 4) : [m, n]] + (fail_match_func_param_sub_a_offset)),
            )
            for jj in range(0, 4):
                sub_A[i, j * 4 + jj] = 1


def test_tensor_load_store():
    _check(tensor_load_store, transformed_tensor_load_store)


def test_opaque_access():
    _check(opaque_access, transformed_opaque_access)
    _check(opaque_tensor_data_projection, transformed_opaque_tensor_data_projection)


def test_high_dim_opaque_access():
    _check(high_dim_opaque_access, transformed_high_dim_opaque_access)
    _check(
        high_dim_opaque_access_with_source_strides,
        transformed_high_dim_opaque_access_with_source_strides,
    )


def test_recursive_match():
    _check(recursive_match, transformed_recursive_match)


def test_symbolic_match():
    _check(symbolic_match, transformed_symbolic_match)


def test_rank0_tensor():
    _check(rank0_tensor, transformed_rank0_tensor)


def test_fail_load_store():
    _check_fail(fail_match_load)
    _check_fail(fail_match_store)


def test_fail_tensor_bind():
    _check_fail(fail_tensor_bind)


def test_fail_match_func_param():
    _check_fail(fail_match_func_param)


scalar_match_tensor_type_coercion_scalar_buf_offset = T.dynamic(
    "scalar_match_tensor_type_coercion_scalar_buf_offset", "int32"
)


@Ts.function
def scalar_match_tensor_type_coercion(A: T.Tensor((8, 8), "float32")) -> None:
    for i, j in T.grid(8, 8):
        with Ts.sblock(""):
            vi = Ts.axis.spatial(8, i)
            vj = Ts.axis.spatial(8, j)
            Ts.reads()
            Ts.writes(A[vi, vj])
            # Create scalar match tensor from single element - this triggers type coercion
            scalar_buf = Ts.match_tensor(
                A[vi, vj],
                (),
                layout=T.TileLayout(
                    T.S[()] + (scalar_match_tensor_type_coercion_scalar_buf_offset)
                ),
            )
            scalar_buf[()] = T.float32(1.0)


@Ts.function
def transformed_scalar_match_tensor_type_coercion(A: T.Tensor((8, 8), "float32")) -> None:
    for i, j in T.grid(8, 8):
        with Ts.sblock(""):
            vi = Ts.axis.spatial(8, i)
            vj = Ts.axis.spatial(8, j)
            Ts.reads()
            Ts.writes(A[vi, vj])
            # Scalar match_tensor eliminated, direct assignment
            A[vi, vj] = T.float32(1.0)


def test_scalar_match_tensor_type_coercion():
    _check(scalar_match_tensor_type_coercion, transformed_scalar_match_tensor_type_coercion)


masked_match_tensor_sub_a_offset = T.dynamic("masked_match_tensor_sub_a_offset", "int32")


@Ts.function
def masked_match_tensor(A: T.Tensor((8,), "float32")) -> None:
    with Ts.sblock():
        Ts.reads(A[2:6])
        sub_A = Ts.match_tensor(
            A[2:6], (4,), layout=T.TileLayout(T.S[(4,)] + (masked_match_tensor_sub_a_offset))
        )
        mask = T.meta_var(T.Broadcast(T.bool(True), 4))
        T.evaluate(T.masked_load(sub_A, T.Ramp(0, 1, 4), mask, ty="float32x4"))


def test_masked_match_tensor_fails_explicitly():
    mod = tvm_tirx.IRModule.from_expr(masked_match_tensor)
    with pytest.raises(RuntimeError, match="Predicated tensor access is not currently supported"):
        tvm.s_tir.transform.LowerMatchTensor()(mod)


if __name__ == "__main__":
    tvm_tirx.testing.main()
