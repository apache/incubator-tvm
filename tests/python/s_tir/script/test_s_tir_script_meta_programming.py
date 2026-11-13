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
"""S-TIR script meta programming."""

from __future__ import annotations

import numpy

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm_tirx.script import tirx as T


def test_meta_programming_matmul():
    def matmul_generator(M: int, N: int, K: int, dtype: str):
        @Ts.function
        def matmul(
            A: T.Tensor([M, K], dtype=dtype),
            B: T.Tensor([N, K], dtype=dtype),
            C: T.Tensor([M, N], dtype=dtype),
        ) -> None:
            for i, j, k in T.grid(M, N, K):
                with Ts.sblock():
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        return matmul

    @Ts.function
    def matmul_128_128_128_fp16(
        A: T.Tensor([128, 128], dtype="float16"),
        B: T.Tensor([128, 128], dtype="float16"),
        C: T.Tensor([128, 128], dtype="float16"),
    ) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock():
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    f = matmul_generator(128, 128, 128, "float16").with_attr("global_symbol", "main")
    tvm_tirx.ir.assert_structural_equal(
        f, matmul_128_128_128_fp16.with_attr("global_symbol", "main")
    )


def test_meta_programming_uncaptured_var():
    def generate_erf(dtype):
        @Ts.function
        def main(A: T.Tensor((1,), dtype), C: T.Tensor((1,), dtype)):
            for i in range(1):
                with Ts.sblock("C"):
                    C[i] = T.erf(A[i])

        return main

    @Ts.function
    def fp32(A: T.Tensor((1,), "float32"), C: T.Tensor((1,), "float32")):
        for i in range(1):
            with Ts.sblock("C"):
                C[i] = T.erf(A[i])

    @Ts.function
    def fp16(A: T.Tensor((1,), "float16"), C: T.Tensor((1,), "float16")):
        for i in range(1):
            with Ts.sblock("C"):
                C[i] = T.erf(A[i])

    f1 = generate_erf("float32").with_attr("global_symbol", "main")
    tvm_tirx.ir.assert_structural_equal(f1, fp32.with_attr("global_symbol", "main"))
    f2 = generate_erf("float16").with_attr("global_symbol", "main")
    tvm_tirx.ir.assert_structural_equal(f2, fp16.with_attr("global_symbol", "main"))


def test_tir_macro_signature():
    @T.inline
    def assign(i, *args, t1, **kwargs):
        vi, vj, vk = Ts.axis.remap("SSR", [i, args[0], args[1]])
        kwargs["t3"][vi, vj] = kwargs["t3"][vi, vj] + t1[vi, vk] * kwargs["t2"][vj, vk]

    @Ts.function(private=True)
    def matmul_w_macro(
        A: T.Tensor([128, 128], "float32"),
        B: T.Tensor([128, 128], "float32"),
        C: T.Tensor([128, 128], "float32"),
    ) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("update"):
                assign(i, j, k, t1=A, t2=B, t3=C)

    @Ts.function(private=True)
    def matmul_no_macro(
        A: T.Tensor([128, 128], "float32"),
        B: T.Tensor([128, 128], "float32"),
        C: T.Tensor([128, 128], "float32"),
    ) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    tvm_tirx.ir.assert_structural_equal(matmul_no_macro, matmul_w_macro)


def test_tir_macro_in_class():
    class Object:
        def __init__(self, x: T.Tensor):
            self.local_x = Ts.sblock_alloc_tensor(x.shape, x.dtype)

        @T.inline
        def load(self, x: T.Tensor):
            N, M = T.meta_var(self.local_x.shape)
            for i, j in T.grid(N, M):
                with Ts.sblock("update"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    self.local_x[vi, vj] = x[vi, vj]

    @Ts.function(private=True)
    def func_w_macro(A: T.Tensor([128, 128], "float32")):
        o1 = T.meta_var(Object(A))
        o1.load(A)
        o2 = T.meta_var(Object(A))
        o2.load(o1.local_x)

    @Ts.function(private=True)
    def func_no_macro(A: T.Tensor([128, 128], "float32")):
        local_a = Ts.sblock_alloc_tensor([128, 128])
        N, M = local_a.shape
        for i, j in T.grid(N, M):
            with Ts.sblock("update"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                local_a[vi, vj] = A[vi, vj]
        local_b = Ts.sblock_alloc_tensor([128, 128])
        N, M = local_b.shape
        for i, j in T.grid(N, M):
            with Ts.sblock("update"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                local_b[vi, vj] = local_a[vi, vj]

    tvm_tirx.ir.assert_structural_equal(func_no_macro, func_w_macro)


def test_tir_starred_for_loop():
    dims = (128, 128)

    @Ts.function(private=True)
    def starred(A: T.Tensor([*dims, 128], "int32"), B: T.Tensor(dims, "int32")):
        for *spatial, reduction in T.grid(*A.shape):
            with Ts.sblock("reduce"):
                with Ts.init():
                    B[spatial] = T.int32(0)
                B[spatial] = B[spatial] + A[(*spatial, reduction)]

    @Ts.function(private=True)
    def non_starred(A: T.Tensor([128, 128, 128], "int32"), B: T.Tensor([128, 128], "int32")):
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock("reduce"):
                with Ts.init():
                    B[i, j] = T.int32(0)
                B[i, j] = B[i, j] + A[i, j, k]

    tvm_tirx.ir.assert_structural_equal(starred, non_starred)


def test_tir_macro_block_name_suffix():
    @T.inline
    def operation(A, idx):
        with Ts.sblock("op"):
            v = Ts.axis.remap("S", [idx])
            A[v] = A[v] * T.float32(2)

    @Ts.function(private=True)
    def func_w_macro(A: T.Tensor([10], "float32")) -> None:
        for i in T.serial(0, 10):
            operation(A, i)
            operation(A, i)
            operation(A, i)

    @Ts.function(private=True)
    def expected(A: T.Tensor([10], "float32")) -> None:
        for i in T.serial(0, 10):
            with Ts.sblock("op"):
                v = Ts.axis.remap("S", [i])
                A[v] = A[v] * T.float32(2)
            with Ts.sblock("op_1"):
                v = Ts.axis.remap("S", [i])
                A[v] = A[v] * T.float32(2)
            with Ts.sblock("op_2"):
                v = Ts.axis.remap("S", [i])
                A[v] = A[v] * T.float32(2)

    tvm_tirx.ir.assert_structural_equal(func_w_macro, expected)


np_array = numpy.array([0, 1, 2, 3])


@Ts.function
def matmul(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_multi_element_array_in_outmost_namespace():
    func = matmul
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)


def test_func_call():
    def shared_16x16_to_ldmatrix_32x8_layout(i, j):
        thread_id = (i % 8) * 4 + (j % 8) // 2
        return thread_id, (j // 8) * 4 + (i // 8) * 2 + (j % 2)

    mma_sync_m16n16k16_desc_a_offset = T.dynamic("mma_sync_m16n16k16_desc_a_offset", "int32")
    mma_sync_m16n16k16_desc_b_offset = T.dynamic("mma_sync_m16n16k16_desc_b_offset", "int32")
    mma_sync_m16n16k16_desc_c_offset = T.dynamic("mma_sync_m16n16k16_desc_c_offset", "int32")

    @Ts.function
    def mma_sync_m16n16k16_desc(
        A: T.Tensor(
            (32, 8),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(32, 8)] + ((16) * (mma_sync_m16n16k16_desc_a_offset))),
            scope="warp",
        ),
        B: T.Tensor(
            (32, 8),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(32, 8)] + ((16) * (mma_sync_m16n16k16_desc_b_offset))),
            scope="warp",
        ),
        C: T.Tensor(
            (32, 8),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(32, 8)] + ((16) * (mma_sync_m16n16k16_desc_c_offset))),
            scope="warp",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32, 0:8], A[0:32, 0:8], B[0:32, 0:8])
            Ts.writes(C[0:32, 0:8])
            for i, j, k in T.grid(16, 16, 16):
                with Ts.sblock("C"):
                    i, j, k = Ts.axis.remap("SSR", [i, j, k])
                    indices_C = shared_16x16_to_ldmatrix_32x8_layout(i, j)
                    indices_A = shared_16x16_to_ldmatrix_32x8_layout(i, k)
                    indices_B = shared_16x16_to_ldmatrix_32x8_layout(k, j)

                    Ts.reads(
                        C[indices_C[0], indices_C[1]],
                        A[indices_A[0], indices_A[1]],
                        B[indices_B[0], indices_B[1]],
                    )
                    Ts.writes(C[indices_C[0], indices_C[1]])

                    C[indices_C[0], indices_C[1]] += (
                        A[indices_A[0], indices_A[1]] * B[indices_B[0], indices_B[1]]
                    )

    mma_sync_m16n16k16_desc_manual_a_offset = T.dynamic(
        "mma_sync_m16n16k16_desc_manual_a_offset", "int32"
    )
    mma_sync_m16n16k16_desc_manual_b_offset = T.dynamic(
        "mma_sync_m16n16k16_desc_manual_b_offset", "int32"
    )
    mma_sync_m16n16k16_desc_manual_c_offset = T.dynamic(
        "mma_sync_m16n16k16_desc_manual_c_offset", "int32"
    )

    @Ts.function
    def mma_sync_m16n16k16_desc_manual(
        A: T.Tensor(
            (32, 8),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(32, 8)] + ((16) * (mma_sync_m16n16k16_desc_manual_a_offset))),
            scope="warp",
        ),
        B: T.Tensor(
            (32, 8),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(32, 8)] + ((16) * (mma_sync_m16n16k16_desc_manual_b_offset))),
            scope="warp",
        ),
        C: T.Tensor(
            (32, 8),
            "float16",
            align=64,
            layout=T.TileLayout(T.S[(32, 8)] + ((16) * (mma_sync_m16n16k16_desc_manual_c_offset))),
            scope="warp",
        ),
    ) -> None:
        with Ts.sblock("root"):
            Ts.reads(C[0:32, 0:8], A[0:32, 0:8], B[0:32, 0:8])
            Ts.writes(C[0:32, 0:8])
            for i, j, k in T.grid(16, 16, 16):
                with Ts.sblock("C"):
                    i, j, k = Ts.axis.remap("SSR", [i, j, k])
                    Ts.reads(
                        C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2],
                        A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2],
                        B[k % 8 * 4 + j % 8 // 2, j // 8 * 4 + k // 8 * 2 + j % 2],
                    )
                    Ts.writes(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2])
                    C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] = (
                        C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2]
                        + A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2]
                        * B[k % 8 * 4 + j % 8 // 2, j // 8 * 4 + k // 8 * 2 + j % 2]
                    )

    assert_structural_equal_ignore_global_symbol(
        mma_sync_m16n16k16_desc, mma_sync_m16n16k16_desc_manual
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
