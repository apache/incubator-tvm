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
"""S-TIR script dynamic shape."""

from __future__ import annotations

import sys

import pytest

import tvm
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm_tirx.script import from_source
from tvm_tirx.script import tirx as T


def test_tir_tensor_region_extent_correct_dtype():
    @Ts.function
    def func(A: T.Tensor((T.int64(16), T.int64(1)), "float32")):
        for i in T.grid(T.int64(16)):
            with Ts.sblock("block"):
                vi = Ts.axis.remap("S", [i])
                Ts.reads(A[vi, T.int64(0) : T.int64(1)])
                T.evaluate(0)

    assert func.body[0].block.body[0].body[0].block.reads[0].region[0].extent.ty.dtype == "int64"


N = T.dynamic("N", "int32")


M = T.dynamic("M", "int32")


K = T.dynamic("K", "int32")


@Ts.function
def gemm_dyn_shape(
    A: T.Tensor((N, K), "float32"), B: T.Tensor((K, M), "float32"), C: T.Tensor((N, M), "float32")
):
    for i, j, k in T.grid(N, M, K):
        with Ts.sblock("gemm"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_dynamic_shape_gemm():
    gemm_dyn_shape_roundtrip = from_source(
        gemm_dyn_shape.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    )
    assert_structural_equal_ignore_global_symbol(gemm_dyn_shape, gemm_dyn_shape_roundtrip)


@Ts.function
def tensor_int64(
    A: T.Tensor((T.int64(128), T.int64(128)), dtype="float32"),
    C: T.Tensor((T.int64(128), T.int64(128)), dtype="float32"),
) -> None:
    B = Ts.sblock_alloc_tensor((T.int64(128), T.int64(128)), dtype="float32")

    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(T.int64(128), T.int64(128)):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@Ts.function
def tensor_int64_after_roundtrip(
    A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    C: T.Tensor((T.int64(128), T.int64(128)), "float32"),
) -> None:
    B = Ts.sblock_alloc_tensor((T.int64(128), T.int64(128)), dtype="float32")
    for i, j in T.grid(128, 128):
        with Ts.sblock("B"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(T.int64(128), T.int64(128)):
        with Ts.sblock("C"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


def test_tensor_int64():
    original = tensor_int64
    after_roundtrip = tensor_int64_after_roundtrip
    assert_structural_equal_ignore_global_symbol(original, after_roundtrip, True)


def test_int64_loop():
    @Ts.function
    def int64_grid(
        A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + 1.0

    @Ts.function
    def int64_grid_expanded(
        A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    ) -> None:
        for i in range(T.int64(0), T.int64(128)):
            for j in range(T.int64(0), T.int64(128)):
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(T.int64(128), i)
                    vj = Ts.axis.spatial(T.int64(128), j)
                    B[vi, vj] = A[vi, vj] + 1.0

    assert_structural_equal_ignore_global_symbol(int64_grid, int64_grid_expanded)


def parse_tensorslice_as_range_bound():
    # apparently the use of i in the "outer" block when it is defined outside of a block is wrong
    @Ts.function(check_well_formed=False)
    def segment_sum(
        A: T.Tensor([m], dtype="float32"),  # noqa: F821
        B: T.Tensor([n], dtype="float32"),  # noqa: F821
        indptr: T.Tensor([n + 1], dtype="int32"),  # noqa: F821
        n: T.int32,
        m: T.int32,
    ) -> None:
        for i in T.serial(n):
            with Ts.sblock("outer"):
                vi = Ts.axis.spatial(n, i)
                Ts.reads(indptr[i : i + 2], B[vi], A[indptr[i] : indptr[i + 1]])
                Ts.writes(B[vi])
                for j in T.serial(indptr[i], indptr[i + 1]):
                    with Ts.sblock("inner"):
                        vj = Ts.axis.reduce(m, j)
                        Ts.reads(B[vi], A[vj])
                        Ts.writes(B[vi])
                        with Ts.init():
                            B[vi] = T.float32(0)
                        B[vi] = B[vi] + A[vj]

    return segment_sum


@pytest.mark.parametrize(
    "ir_generator",
    [parse_tensorslice_as_range_bound],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_dynamic_shape(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


if __name__ == "__main__":
    tvm_tirx.testing.main()
