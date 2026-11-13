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

import tvm
import tvm.s_tir
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


def _check(original, transformed):
    func = original
    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    # These expected fixtures include the subsequent lowering of thread loops.
    mod = tvm.s_tir.transform.LowerThreadBinding()(mod)
    mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
    tvm_tirx.ir.assert_structural_equal(
        mod["main"], transformed.with_attr("global_symbol", "main"), True
    )


@Ts.function
def compacted_elementwise_func(
    A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")
) -> None:
    for i in range(0, 16):
        with Ts.sblock():
            Ts.reads(A[i, 0:16])
            Ts.writes(C[i, 0:16])
            B = Ts.sblock_alloc_tensor([1, 16], "float32", scope="global")
            for j in range(0, 16):
                with Ts.sblock():
                    Ts.reads(A[i, j])
                    Ts.writes(B[0, j])
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with Ts.sblock():
                    Ts.reads(B[0, j])
                    Ts.writes(C[i, j])
                    C[i, j] = B[0, j] * 2.0


@Ts.function
def transformed_elementwise_func(
    A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")
) -> None:
    for i in T.serial(0, 16):
        B_new = T.alloc_tensor(ty_args=[T.Tensor([1, 16], "float32")])
        for j in T.serial(0, 16):
            B_new[0, j] = A[i, j] + 1.0
        for j in T.serial(0, 16):
            C[i, j] = B_new[0, j] * 2.0


@Ts.function
def compacted_gpu_func(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
    for i0 in T.parallel(0, 4, annotations={"thread_binding": "blockIdx.x"}):
        for i1 in T.parallel(0, 2, annotations={"thread_binding": "threadIdx.x"}):
            for i2 in T.parallel(0, 2, annotations={"thread_binding": "vthread"}):
                with Ts.sblock():
                    Ts.reads(A[i0 * 4 + i1 * 2 + i2, 0:16])
                    Ts.writes(C[i0 * 4 + i1 * 2 + i2, 0:16])
                    B = Ts.sblock_alloc_tensor([1, 16], "float32", scope="local")
                    for j in range(0, 16):
                        with Ts.sblock():
                            Ts.reads(A[i0 * 4 + i1 * 2 + i2, j])
                            Ts.writes(B[0, j])
                            B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
                    for j in range(0, 16):
                        with Ts.sblock():
                            Ts.reads(B[0, j])
                            Ts.writes(C[i0 * 4 + i1 * 2 + i2, j])
                            C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0


@Ts.function
def transformed_gpu_func(
    A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")
) -> None:
    i0 = T.launch_thread("blockIdx.x", 4)
    i1 = T.launch_thread("threadIdx.x", 2)
    i2 = T.launch_thread("vthread", 2)
    B = T.alloc_tensor(ty_args=[T.Tensor([1, 16], "float32", scope="local")])
    for j in range(0, 16):
        B[0, j] = A[i0 * 4 + i1 * 2 + i2, j] + 1.0
    for j in range(0, 16):
        C[i0 * 4 + i1 * 2 + i2, j] = B[0, j] * 2.0


@Ts.function
def compacted_symbolic_func(
    A: T.Tensor((n, m), "float32"),  # noqa: F821
    C: T.Tensor((n, m), "float32"),  # noqa: F821
    n: T.int32,
    m: T.int32,
) -> None:
    for i in range(0, n):
        with Ts.sblock():
            Ts.reads(A[i, m])
            Ts.writes(C[i, m])
            B = Ts.sblock_alloc_tensor((m,), "float32", scope="global")
            for j in range(0, m):
                with Ts.sblock():
                    Ts.reads(A[i, j])
                    Ts.writes(B[j])
                    B[j] = A[i, j] + 1.0
            for j in range(0, m):
                with Ts.sblock():
                    Ts.reads(B[j])
                    Ts.writes(C[i, j])
                    C[i, j] = B[j] * 2.0


@Ts.function
def transformed_symbolic_func(
    A: T.Tensor((n, m), "float32"),  # noqa: F821
    C: T.Tensor((n, m), "float32"),  # noqa: F821
    n: T.int32,
    m: T.int32,
) -> None:
    for i in range(0, n):
        B = T.alloc_tensor(ty_args=[T.Tensor([m], "float32")])
        for j in range(0, m):
            B[j] = A[i, j] + 1.0
        for j in range(0, m):
            C[i, j] = B[j] * 2.0


@Ts.function
def compacted_predicate_func(A: T.Tensor(32, "float32"), C: T.Tensor(32, "float32")) -> None:
    for i, j in T.grid(5, 7):
        with Ts.sblock():
            Ts.reads(A[i * 7 + j])
            Ts.writes(C[i * 7 + j])
            Ts.where(i * 7 + j < 32)
            C[i * 7 + j] = A[i * 7 + j] + 1.0


@Ts.function
def transformed_predicate_func(A: T.Tensor(32, "float32"), C: T.Tensor(32, "float32")) -> None:
    for i, j in T.grid(5, 7):
        if i * 7 + j < 32:
            C[i * 7 + j] = A[i * 7 + j] + 1.0


@Ts.function
def compacted_unit_loop_func(A: T.Tensor(32, "float32"), C: T.Tensor(32, "float32")) -> None:
    for x, y, z in T.grid(4, 1, 8):
        with Ts.sblock():
            Ts.reads(A[x * 8 + y * 8 + z])
            Ts.writes(C[x * 8 + y * 8 + z])
            C[x * 8 + y * 8 + z] = A[x * 8 + y * 8 + z] + 1.0


@Ts.function
def transformed_unit_loop_func(A: T.Tensor(32, "float32"), C: T.Tensor(32, "float32")) -> None:
    for x, z in T.grid(4, 8):
        C[x * 8 + z] = A[x * 8 + z] + 1.0


@Ts.function
def compacted_multi_alloc_func(A: T.Tensor(32, "float32"), D: T.Tensor(32, "float32")) -> None:
    for i in range(0, 32):
        with Ts.sblock():
            Ts.reads(A[i])
            Ts.writes(D[i])
            B = Ts.sblock_alloc_tensor((32,), scope="global")
            C = Ts.sblock_alloc_tensor((32,), scope="global")
            B[i] = A[i] + 1.0
            C[i] = A[i] + B[i]
            D[i] = C[i] * 2.0


@Ts.function
def transformed_multi_alloc_func(A: T.Tensor(32, "float32"), D: T.Tensor(32, "float32")) -> None:
    for i in range(0, 32):
        B = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32")])
        C = T.alloc_tensor(ty_args=[T.Tensor((32,), "float32")])
        B[i] = A[i] + 1.0
        C[i] = A[i] + B[i]
        D[i] = C[i] * 2.0


@Ts.function
def compacted_strided_tensor_func(
    A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")
) -> None:
    for i0 in range(0, 4):
        with Ts.sblock():
            Ts.reads(A[i0 * 4 : i0 * 4 + 4, 0:16])
            Ts.writes(C[i0 * 4 : i0 * 4 + 4, 0:16])
            B = Ts.sblock_alloc_tensor(
                [4, 16], "float32", layout=T.TileLayout(T.S[[4, 16] : [17, 1]]), scope="global"
            )
            for i1 in range(0, 4):
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[i0 * 4 + i1, j])
                        Ts.writes(B[i1, j])
                        B[i1, j] = A[i0 * 4 + i1, j] + 1.0
            for i1 in range(0, 4):
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(B[i1, j])
                        Ts.writes(C[i0 * 4 + i1, j])
                        C[i0 * 4 + i1, j] = B[i1, j] * 2.0


@Ts.function
def transformed_strided_tensor_func(
    A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")
) -> None:
    # body
    for i0 in T.serial(4):
        B = T.alloc_tensor(
            ty_args=[T.Tensor([4, 16], "float32", layout=T.TileLayout(T.S[[4, 16] : [17, 1]]))]
        )
        for i1, j in T.grid(4, 16):
            B[i1, j] = A[i0 * 4 + i1, j] + T.float32(1)
        for i1, j in T.grid(4, 16):
            C[i0 * 4 + i1, j] = B[i1, j] * T.float32(2)


n = T.dynamic("n", "int32")


@Ts.function
def compacted_symbolic_strided_tensor_func(A: T.Tensor((1, n, 10240), "float32")) -> None:
    padded_size = T.meta_var(T.min((n + 63) // 64 * 64, 96))
    # with Ts.sblock("root"):
    for i, j, k in T.grid(((n + 63) // 64 * 4 + 7) // 8, 2, 160):
        with Ts.sblock(""):
            A_pad_shared_dyn = Ts.sblock_alloc_tensor(
                (1, padded_size, 64),
                layout=T.TileLayout(T.S[(1, padded_size, 64) : (72 * padded_size, 72, 1)]),
                scope="shared.dyn",
            )
            for ax0, ax1 in T.grid(96, 64):
                with Ts.sblock("A_pad_shared.dyn"):
                    Ts.where(i * 128 + j * 32 + ax0 < (n + 63) // 64 * 64)
                    A_pad_shared_dyn[0, ax0, ax1] = T.if_then_else(
                        i * 128 + j * 32 + ax0 < n,
                        A[0, i * 128 + j * 32 + ax0, k * 64 + ax1],
                        T.float32(0),
                    )


n = T.dynamic("n", "int32")


@Ts.function
def transformed_symbolic_strided_tensor_func(A: T.Tensor((1, n, 10240), "float32")):
    padded_size = T.min((n + 63) // 64 * 64, 96)
    for i, j, k in T.grid(((n + 63) // 64 * 4 + 7) // 8, 2, 160):
        A_pad_shared_dyn = T.alloc_tensor(
            ty_args=[
                T.Tensor(
                    (1, padded_size, 64),
                    "float32",
                    layout=T.TileLayout(T.S[(1, padded_size, 64) : (72 * padded_size, 72, 1)]),
                    scope="shared.dyn",
                )
            ]
        )
        for ax0, ax1 in T.grid(96, 64):
            if i * 128 + j * 32 + ax0 < (n + 63) // 64 * 64:
                A_pad_shared_dyn[0, ax0, ax1] = T.if_then_else(
                    i * 128 + j * 32 + ax0 < n,
                    A[0, i * 128 + j * 32 + ax0, k * 64 + ax1],
                    T.float32(0),
                )


@Ts.function
def boolean_handling_before(a: T.Tensor(10, "bool"), b: T.Tensor(10, "bool")) -> None:
    for i0 in T.serial(10):
        with Ts.sblock("b"):
            Ts.reads(a[i0])
            Ts.writes(b[i0])
            b[i0] = a[i0]


@Ts.function
def boolean_handling_after(a: T.Tensor(10, "bool"), b: T.Tensor(10, "bool")) -> None:
    # body
    for i0 in T.serial(10):
        b[i0] = a[i0]


def test_elementwise():
    _check(compacted_elementwise_func, transformed_elementwise_func)


def test_gpu_workload():
    _check(compacted_gpu_func, transformed_gpu_func)


def test_symbolic_shape():
    _check(compacted_symbolic_func, transformed_symbolic_func)


def test_predicate():
    _check(compacted_predicate_func, transformed_predicate_func)


def test_unit_loops():
    _check(compacted_unit_loop_func, transformed_unit_loop_func)


def test_multi_alloc():
    _check(compacted_multi_alloc_func, transformed_multi_alloc_func)


def test_strided_tensor():
    _check(compacted_strided_tensor_func, transformed_strided_tensor_func)


def test_symbolic_strided_tensor():
    _check(compacted_symbolic_strided_tensor_func, transformed_symbolic_strided_tensor_func)


def test_preserved_annotations():
    @Ts.function
    def before(A: T.Tensor(8, "float32"), B: T.Tensor(8, "float32")):
        for i in T.serial(8, annotations={"k_0": 1, "k_1": [2, 3], "k_2": 3.14}):
            with Ts.sblock("block"):
                Ts.sblock_attr({"k_3": "oops"})
                B[i] = A[i] + 1.0

    @Ts.function
    def after(A: T.Tensor(8, "float32"), B: T.Tensor(8, "float32")):
        for i in T.serial(8, annotations={"k_0": 1, "k_1": [2, 3], "k_2": 3.14}):
            B[i] = A[i] + 1.0

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], after.with_attr("global_symbol", "main"))


def test_preserved_none_pragma_annotation():
    @Ts.function
    def before(A: T.Tensor(8, "float32"), B: T.Tensor(8, "float32")):
        for i in T.serial(8, annotations={"tirx.unroll_explicit": None}):
            with Ts.sblock("block"):
                B[i] = A[i] + 1.0

    @Ts.function
    def after(A: T.Tensor(8, "float32"), B: T.Tensor(8, "float32")):
        for i in T.serial(8, annotations={"tirx.unroll_explicit": None}):
            B[i] = A[i] + 1.0

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    tvm_tirx.ir.assert_structural_equal(
        mod["main"],
        after.with_attr("global_symbol", "main"),
    )


def test_allocation_type_alignment():
    @Ts.function
    def before(A: T.Tensor(4, "float32")):
        with Ts.sblock("root"):
            B = Ts.sblock_alloc_tensor((4,), "float32", scope="shared", align=128)
            for i in range(4):
                B[i] = A[i]
            for i in range(4):
                A[i] = B[i]

    @Ts.function
    def after(A: T.Tensor(4, "float32")):
        B = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="shared", align=128)])
        for i in range(4):
            B[i] = A[i]
        for i in range(4):
            A[i] = B[i]

    _check(before, after)


def test_boolean_handling():
    _check(boolean_handling_before, boolean_handling_after)


if __name__ == "__main__":
    tvm_tirx.testing.main()
