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
# ruff: noqa: E501
from __future__ import annotations

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


class BaseCompactTest:
    """Base testcase class. The inherit testcase should include:
    - `before` and `expected` function used to check structural equality for the transformation.
    - `is_lower_order_free` tag, defaults to True, denotes that we would check
       (LowerOpaqueBlock . CompactTensorAllocation)(before) ==
       (CompactTensorAllocation . LowerOpaqueBlock)(before)
    - `is_strict` tag, defaults to True, controls the `is_strict` option of the compaction pass.
    """

    def test_compact(self):
        is_lower_order_free = getattr(self, "is_lower_order_free", True)
        is_strict = getattr(self, "is_strict_mode", True)

        before = tvm_tirx.IRModule.from_expr(self.before.with_attr("global_symbol", "main"))
        expected = tvm_tirx.IRModule.from_expr(self.expected.with_attr("global_symbol", "main"))
        simplify = tvm_tirx.ir.transform.Sequential(
            [s_tir.transform.StmtSimplify(), tirx.transform.RemoveNoOp()]
        )
        after = simplify(s_tir.transform.CompactTensorAllocation(is_strict=is_strict)(before))
        expected = simplify(expected)
        try:
            tvm_tirx.ir.assert_structural_equal(after, expected)
        except ValueError as err:
            script = tvm_tirx.IRModule(
                {"expected": expected["main"], "after": after["main"], "before": before["main"]}
            ).script()
            raise ValueError(
                f"Function after simplification did not match expected:\n{script}"
            ) from err

        if not is_lower_order_free:
            return
        lower_before_compact = s_tir.transform.LowerOpaqueBlock()(before)
        lower_before_compact = s_tir.transform.CompactTensorAllocation(is_strict=is_strict)(
            lower_before_compact
        )
        lower_before_compact = simplify(lower_before_compact)
        lower_after_compact = s_tir.transform.LowerOpaqueBlock()(after)
        lower_after_compact = simplify(lower_after_compact)
        try:
            tvm_tirx.ir.assert_structural_equal(lower_before_compact, lower_after_compact)
        except ValueError as err:
            script = tvm_tirx.IRModule(
                {
                    "lower_before_compact": lower_before_compact["main"],
                    "lower_after_compact": lower_after_compact["main"],
                    "before": before["main"],
                }
            ).script()
            raise ValueError(
                f"Function after simplification did not match expected:\n{script}"
            ) from err


class TestElemwise(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i in range(0, 16):
            with Ts.sblock():
                Ts.reads(A[i, 0:16])
                Ts.writes(C[i, 0:16])
                B = Ts.sblock_alloc_tensor((16, 16), "float32")
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[i, j])
                        Ts.writes(B[i, j])
                        B[i, j] = A[i, j] + 1.0
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(B[i, j])
                        Ts.writes(C[i, j])
                        C[i, j] = B[i, j] * 2.0

    @Ts.function
    def expected(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i in range(0, 16):
            with Ts.sblock():
                Ts.reads(A[i, 0:16])
                Ts.writes(C[i, 0:16])
                B = Ts.sblock_alloc_tensor((1, 16), "float32")
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


class TestUnschedulableFunc(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i in range(0, 16):
            with Ts.sblock():
                Ts.reads(A[i, 0:16])
                Ts.writes(C[i, 0:16])
                B = Ts.sblock_alloc_tensor((16, 16), "float32")
                for j in range(0, 16):
                    T.evaluate(T.call_extern("dummy_extern_function", B.data, ty="int32"))
                    B[i, j] = A[i, j] + 1.0
                for j in range(0, 16):
                    C[i, j] = B[i, j] * 2.0

    expected = before


class TestParamTensorAccess(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((20, 20), "float32"), B: T.Tensor((20, 20), "float32")) -> None:
        for i in range(0, 16):
            with Ts.sblock():
                Ts.reads(A[i, 0:16])
                Ts.writes(B[i, 0:16])
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[i, j])
                        Ts.writes(B[i, j])
                        B[i, j] = A[i, j] + 1.0

    expected = before


class TestSharedMem(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i0 in T.parallel(0, 2, annotations={"thread_binding": "blockIdx.x"}):
            for i1 in T.parallel(0, 2, annotations={"thread_binding": "vthread"}):
                for i2 in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock():
                        Ts.reads(A[i0 * 8 + i1 * 4 + i2, 0:16])
                        Ts.writes(C[i0 * 8 + i1 * 4 + i2, 0:16])
                        B = Ts.sblock_alloc_tensor((16, 16), "float32", scope="shared")
                        for j in range(0, 16):
                            with Ts.sblock():
                                Ts.reads(A[i0 * 8 + i1 * 4 + i2, j])
                                Ts.writes(B[i0 * 8 + i1 * 4 + i2, j])
                                B[i0 * 8 + i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                        for j in range(0, 16):
                            with Ts.sblock():
                                Ts.reads(B[i0 * 8 + i1 * 4 + i2, j])
                                Ts.writes(C[i0 * 8 + i1 * 4 + i2, j])
                                C[i0 * 8 + i1 * 4 + i2, j] = B[i0 * 8 + i1 * 4 + i2, j] * 2.0

    @Ts.function
    def expected(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i0 in T.parallel(0, 2, annotations={"thread_binding": "blockIdx.x"}):
            for i1 in T.parallel(0, 2, annotations={"thread_binding": "vthread"}):
                for i2 in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock():
                        Ts.reads(A[i0 * 8 + i1 * 4 + i2, 0:16])
                        Ts.writes(C[i0 * 8 + i1 * 4 + i2, 0:16])
                        B = Ts.sblock_alloc_tensor((8, 16), "float32", scope="shared")
                        for j in range(0, 16):
                            with Ts.sblock():
                                Ts.reads(A[i0 * 8 + i1 * 4 + i2, j])
                                Ts.writes(B[i1 * 4 + i2, j])
                                B[i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                        for j in range(0, 16):
                            with Ts.sblock():
                                Ts.reads(B[i1 * 4 + i2, j])
                                Ts.writes(C[i0 * 8 + i1 * 4 + i2, j])
                                C[i0 * 8 + i1 * 4 + i2, j] = B[i1 * 4 + i2, j] * 2.0


class TestWrapMem(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i0 in T.parallel(0, 2, annotations={"thread_binding": "blockIdx.x"}):
            for i1 in T.parallel(0, 2, annotations={"thread_binding": "vthread"}):
                for i2 in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock():
                        Ts.reads(A[i0 * 8 + i1 * 4 + i2, 0:16])
                        Ts.writes(C[i0 * 8 + i1 * 4 + i2, 0:16])
                        B = Ts.sblock_alloc_tensor((16, 16), "float32", scope="warp")
                        for j in range(0, 16):
                            with Ts.sblock():
                                Ts.reads(A[i0 * 8 + i1 * 4 + i2, j])
                                Ts.writes(B[i0 * 8 + i1 * 4 + i2, j])
                                B[i0 * 8 + i1 * 4 + i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                        for j in range(0, 16):
                            with Ts.sblock():
                                Ts.reads(B[i0 * 8 + i1 * 4 + i2, j])
                                Ts.writes(C[i0 * 8 + i1 * 4 + i2, j])
                                C[i0 * 8 + i1 * 4 + i2, j] = B[i0 * 8 + i1 * 4 + i2, j] * 2.0

    @Ts.function
    def expected(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i0 in T.parallel(0, 2, annotations={"thread_binding": "blockIdx.x"}):
            for i1 in T.parallel(0, 2, annotations={"thread_binding": "vthread"}):
                for i2 in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock():
                        Ts.reads(A[i0 * 8 + i1 * 4 + i2, 0:16])
                        Ts.writes(C[i0 * 8 + i1 * 4 + i2, 0:16])
                        B = Ts.sblock_alloc_tensor((4, 16), "float32", scope="warp")
                        for j in range(0, 16):
                            with Ts.sblock():
                                Ts.reads(A[i0 * 8 + i1 * 4 + i2, j])
                                Ts.writes(B[i2, j])
                                B[i2, j] = A[i0 * 8 + i1 * 4 + i2, j] + 1.0
                        for j in range(0, 16):
                            with Ts.sblock():
                                Ts.reads(B[i2, j])
                                Ts.writes(C[i0 * 8 + i1 * 4 + i2, j])
                                C[i0 * 8 + i1 * 4 + i2, j] = B[i2, j] * 2.0


class TestSymbolic(BaseCompactTest):
    @Ts.function
    def before(
        A: T.Tensor((n * 8,), "float32"),  # noqa: F821
        C: T.Tensor((n * 8,), "float32"),  # noqa: F821
        n: T.int32,
    ) -> None:
        for i in range(0, n):
            with Ts.sblock():
                Ts.reads(A[i * 8 : i * 8 + 8])
                Ts.writes(C[i * 8 : i * 8 + 8])
                B = Ts.sblock_alloc_tensor((n * 8,), "float32")
                for j in range(0, 8):
                    with Ts.sblock():
                        Ts.reads(A[i * 8 + j])
                        Ts.writes(B[i * 8 + j])
                        B[i * 8 + j] = A[i * 8 + j] + 1.0
                for j in range(0, 8):
                    with Ts.sblock():
                        Ts.reads(B[i * 8 + j])
                        Ts.writes(C[i * 8 + j])
                        C[i * 8 + j] = B[i * 8 + j] * 2.0

    @Ts.function
    def expected(
        A: T.Tensor((n * 8,), "float32"),  # noqa: F821
        C: T.Tensor((n * 8,), "float32"),  # noqa: F821
        n: T.int32,
    ) -> None:
        for i in range(0, n):
            with Ts.sblock():
                Ts.reads(A[i * 8 : i * 8 + 8])
                Ts.writes(C[i * 8 : i * 8 + 8])
                B = Ts.sblock_alloc_tensor((8,), "float32")
                for j in range(0, 8):
                    with Ts.sblock():
                        Ts.reads(A[i * 8 + j])
                        Ts.writes(B[j])
                        B[j] = A[i * 8 + j] + 1.0
                for j in range(0, 8):
                    with Ts.sblock():
                        Ts.reads(B[j])
                        Ts.writes(C[i * 8 + j])
                        C[i * 8 + j] = B[j] * 2.0


class TestComplexFunc(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((8, 8), "float32"), C: T.Tensor((8, 8), "float32"), n: T.int32) -> None:
        for i in range(0, 8):
            with Ts.sblock():
                Ts.reads(A[0, 8])
                Ts.writes(C[0, 8])
                B = Ts.sblock_alloc_tensor((8, 8), "float32")
                for j in range(0, 4):
                    with Ts.sblock():
                        D = Ts.sblock_alloc_tensor((8, 8), "float32")
                        Ts.reads(A[i, j])
                        Ts.writes(B[i, j])
                        for k in range(4, 8):
                            D[k, j] = 1.0
                        for k in range(2, 4):
                            B[i, j] = A[i, j] + D[k, j]
                for j in range(3, 5):
                    with Ts.sblock():
                        Ts.reads(B[i, j])
                        Ts.writes(C[i, j])
                        C[i, j] = B[i, j]
                for j in range(6, 8):
                    with Ts.sblock():
                        Ts.reads(B[i, j])
                        Ts.writes(C[i, j])
                        C[i, j] = B[i, j]

    @Ts.function
    def expected(
        A: T.Tensor((8, 8), "float32"), C: T.Tensor((8, 8), "float32"), n: T.int32
    ) -> None:
        for i in range(0, 8):
            with Ts.sblock():
                Ts.reads(A[0, 8])
                Ts.writes(C[0, 8])
                B = Ts.sblock_alloc_tensor((1, 8), "float32")
                for j in range(0, 4):
                    with Ts.sblock():
                        D = Ts.sblock_alloc_tensor((6, 1), "float32")
                        Ts.reads(A[i, j])
                        Ts.writes(B[0, j])
                        for k in range(4, 8):
                            D[k - 2, 0] = 1.0
                        for k in range(2, 4):
                            B[0, j] = A[i, j] + D[k - 2, 0]
                for j in range(3, 5):
                    with Ts.sblock():
                        Ts.reads(B[0, j])
                        Ts.writes(C[i, j])
                        C[i, j] = B[0, j]
                for j in range(6, 8):
                    with Ts.sblock():
                        Ts.reads(B[0, j])
                        Ts.writes(C[i, j])
                        C[i, j] = B[0, j]


class TestMatchTensor(BaseCompactTest):
    is_lower_order_free = False

    @Ts.function
    def before(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i in range(0, 16):
            with Ts.sblock():
                A0 = Ts.match_tensor(A[i, 0:16], (16))
                C0 = Ts.match_tensor(C[i, 0:16], (16))
                B = Ts.sblock_alloc_tensor((16, 16))
                with Ts.sblock():
                    B0 = Ts.match_tensor(B[i, 0:16], (16))
                    for j in range(0, 16):
                        with Ts.sblock():
                            A1 = Ts.match_tensor(A0[j], ())
                            B1 = Ts.match_tensor(B0[j], ())
                            B1[()] = A1[()] + 1.0
                for j in range(0, 16):
                    with Ts.sblock():
                        C1 = Ts.match_tensor(C0[j], ())
                        B2 = Ts.match_tensor(B[i, j], ())
                        C1[()] = B2[()] * 2.0

    @Ts.function
    def expected(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i in range(0, 16):
            with Ts.sblock():
                A0 = Ts.match_tensor(A[i, 0:16], (16))
                C0 = Ts.match_tensor(C[i, 0:16], (16))
                B = Ts.sblock_alloc_tensor((1, 16))
                with Ts.sblock():
                    B0 = Ts.match_tensor(B[0, 0:16], (16))
                    for j in range(0, 16):
                        with Ts.sblock():
                            A1 = Ts.match_tensor(A0[j], ())
                            B1 = Ts.match_tensor(B0[j], ())
                            B1[()] = A1[()] + 1.0
                for j in range(0, 16):
                    with Ts.sblock():
                        C1 = Ts.match_tensor(C0[j], ())
                        B2 = Ts.match_tensor(B[0, j], ())
                        C1[()] = B2[()] * 2.0


class TestStorageAlign(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i in range(0, 16):
            with Ts.sblock():
                Ts.reads(A[i, 0:16])
                Ts.writes(C[i, 0:16])
                B = Ts.sblock_alloc_tensor((16, 16), "float32")
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[i, j])
                        Ts.writes(B[i, j])
                        Ts.sblock_attr({"tensor_dim_align": [[0, 0, 16, 15]]})
                        B[i, j] = A[i, j] + 1.0
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(B[i, j])
                        Ts.writes(C[i, j])
                        C[i, j] = B[i, j] * 2.0

    @Ts.function
    def expected(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i in range(0, 16):
            with Ts.sblock():
                Ts.reads(A[i, 0:16])
                Ts.writes(C[i, 0:16])
                B = Ts.sblock_alloc_tensor(
                    (1, 16), layout=T.TileLayout(T.S[(1, 16) : (31, 1)]), dtype="float32"
                )
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[i, j])
                        Ts.writes(B[0, j])
                        Ts.sblock_attr({"tensor_dim_align": [[0, 0, 16, 15]]})
                        B[0, j] = A[i, j] + 1.0
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(B[0, j])
                        Ts.writes(C[i, j])
                        C[i, j] = B[0, j] * 2.0


class TestPaddingPattern(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((16, 16), "float32"), C: T.Tensor((20, 20), "float32")) -> None:
        with Ts.sblock():
            B = Ts.sblock_alloc_tensor((20, 20), dtype="float32")
            for i, j in T.grid(16, 16):
                with Ts.sblock():
                    B[i, j] = A[i, j]
            for i, j in T.grid(20, 20):
                with Ts.sblock():
                    C[i, j] = T.if_then_else(
                        2 <= i and i < 18 and 2 <= j and j < 18,
                        B[i - 2, j - 2],
                        0.0,
                    )

    @Ts.function
    def expected(
        A: T.Tensor([16, 16], dtype="float32"), C: T.Tensor([20, 20], dtype="float32")
    ) -> None:
        with Ts.sblock():
            B = Ts.sblock_alloc_tensor([16, 16], dtype="float32")
            for i, j in T.grid(16, 16):
                with Ts.sblock():
                    B[i, j] = A[i, j]
            for i, j in T.grid(20, 20):
                with Ts.sblock():
                    C[i, j] = T.if_then_else(
                        2 <= i and i < 18 and 2 <= j and j < 18,
                        B[i - 2, j - 2],
                        0.0,
                    )


class TestPaddingPatternInlined(BaseCompactTest):
    @Ts.function
    def before(
        X: T.Tensor([224, 224], dtype="float32"), Y: T.Tensor([224, 224], dtype="float32")
    ) -> None:
        cache = Ts.sblock_alloc_tensor([224, 224], dtype="float32")
        for h, w in T.grid(224, 224):
            with Ts.sblock("cache"):
                cache[h, w] = X[h, w]
        for h, w, kh, kw in T.grid(224, 224, 3, 3):
            with Ts.sblock("compute"):
                Y[h, w] = T.max(
                    Y[h, w],
                    T.if_then_else(
                        T.likely(1 <= h + kh)
                        and T.likely(h + kh < 225)
                        and T.likely(1 <= w + kw)
                        and T.likely(w + kw < 225),
                        cache[h + kh - 1, w + kw - 1],
                        0.0,
                    ),
                )

    @Ts.function
    def expected(X: T.Tensor((224, 224), "float32"), Y: T.Tensor((224, 224), "float32")) -> None:
        cache = Ts.sblock_alloc_tensor([224, 224], dtype="float32")
        for h, w in T.grid(224, 224):
            with Ts.sblock("cache"):
                cache[h, w] = X[h, w]
        for h, w, kh, kw in T.grid(224, 224, 3, 3):
            with Ts.sblock("compute"):
                Y[h, w] = T.max(
                    Y[h, w],
                    T.if_then_else(
                        T.likely(1 <= h + kh)
                        and T.likely(h + kh < 225)
                        and T.likely(1 <= w + kw)
                        and T.likely(w + kw < 225),
                        cache[h + kh - 1, w + kw - 1],
                        0.0,
                    ),
                )


class TestMemAccessInBranch(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((224, 224), "float32")) -> None:
        with Ts.sblock():
            B1 = Ts.sblock_alloc_tensor((224, 224), dtype="float32")
            B2 = Ts.sblock_alloc_tensor((224, 224), dtype="float32")
            B3 = Ts.sblock_alloc_tensor((224, 224), dtype="float32")
            B4 = Ts.sblock_alloc_tensor((224, 224), dtype="float32")
            for i in range(0, 224):
                for j in range(0, 224):
                    with Ts.sblock():
                        if i < 112 and j < 112:
                            B1[i, j] = A[i, j] * 2.0
                        else:
                            B2[i, j] = A[i, j] + 3.0
            for i in range(0, 224):
                for j in range(0, 224):
                    with Ts.sblock():
                        if i < 112 or j < 112:
                            B3[i, j] = A[i, j] * 2.0
                        else:
                            B4[i, j] = A[i, j] + 3.0

    @Ts.function
    def expected(A: T.Tensor([224, 224], dtype="float32")) -> None:
        with Ts.sblock():
            B1 = Ts.sblock_alloc_tensor([112, 112], dtype="float32")
            B2 = Ts.sblock_alloc_tensor([224, 224], dtype="float32")
            B3 = Ts.sblock_alloc_tensor([224, 224], dtype="float32")
            B4 = Ts.sblock_alloc_tensor([112, 112], dtype="float32")
            for i, j in T.grid(224, 224):
                with Ts.sblock():
                    if i < 112 and j < 112:
                        B1[i, j] = A[i, j] * 2.0
                    else:
                        B2[i, j] = A[i, j] + 3.0
            for i, j in T.grid(224, 224):
                with Ts.sblock():
                    if i < 112 or j < 112:
                        B3[i, j] = A[i, j] * 2.0
                    else:
                        B4[i - 112, j - 112] = A[i, j] + 3.0


class TestAnnotatedOpaqueAccess(BaseCompactTest):
    is_lower_order_free = False

    @Ts.function
    def before(A: T.Tensor((1024,), "float32")) -> None:
        with Ts.sblock():
            B = Ts.sblock_alloc_tensor((1024,), dtype="float32")
            C = Ts.sblock_alloc_tensor((1024,), dtype="float32")
            for i in range(0, 512):
                with Ts.sblock():
                    # no annotation, opaque access will cover full region
                    Ts.reads([])
                    Ts.writes([])
                    T.evaluate(T.call_extern("opaque_extern_function", A.data, B.data, ty="int32"))
                    B[i] = A[i]
                with Ts.sblock():
                    # treat opaque access only access annotated regions, even if
                    # they are not compatible with actual tensor accesses.
                    Ts.reads([B[i]])
                    Ts.writes([C[i : i + 9]])
                    T.evaluate(T.call_extern("opaque_extern_function", B.data, C.data, ty="int32"))
                    C[i] = B[i]

    @Ts.function
    def expected(A: T.Tensor((1024,), "float32")) -> None:
        with Ts.sblock():
            B = Ts.sblock_alloc_tensor((1024,), dtype="float32")
            C = Ts.sblock_alloc_tensor((520,), dtype="float32")
            for i in range(0, 512):
                with Ts.sblock():
                    # no annotation, opaque access will cover full region
                    Ts.reads([])
                    Ts.writes([])
                    T.evaluate(T.call_extern("opaque_extern_function", A.data, B.data, ty="int32"))
                    B[i] = A[i]
                with Ts.sblock():
                    # treat opaque access only access annotated regions, even if
                    # they are not compatible with actual tensor accesses.
                    Ts.reads([B[i]])
                    Ts.writes([C[i : i + 9]])
                    T.evaluate(T.call_extern("opaque_extern_function", B.data, C.data, ty="int32"))
                    C[i] = B[i]


class TestSparseReadCache(BaseCompactTest):
    @Ts.function
    def before(
        A_data: T.Tensor((819,), "float32"),
        B: T.Tensor((128,), "float32"),
        A_indptr: T.Tensor((129,), "int32"),
    ) -> None:
        for i in T.serial(128):
            with Ts.sblock("rowsum_outer"):
                Ts.reads(
                    A_indptr[i : i + 1],
                    A_data[A_indptr[i] + 0 : A_indptr[i] + (A_indptr[i + 1] - A_indptr[i])],
                )
                Ts.writes(B[i])
                with Ts.sblock("rowsum_init"):
                    Ts.reads()
                    Ts.writes(B[i])
                    B[i] = T.float32(0)
                for k in T.serial(A_indptr[i + 1] - A_indptr[i]):
                    with Ts.sblock():
                        Ts.reads(A_indptr[i], A_data[A_indptr[i] + k], B[i])
                        Ts.writes(B[i])
                        A_data_local = Ts.sblock_alloc_tensor([819], dtype="float32", scope="local")
                        with Ts.sblock("A_data_cache_read"):
                            Ts.reads(A_indptr[i], A_data[A_indptr[i] + k])
                            Ts.writes(A_data_local[A_indptr[i] + k])
                            A_data_local[A_indptr[i] + k] = A_data[A_indptr[i] + k]
                        with Ts.sblock("rowsum_inner"):
                            Ts.reads(B[i], A_indptr[i], A_data[A_indptr[i] + k])
                            Ts.writes(B[i])
                            B[i] = B[i] + A_data_local[A_indptr[i] + k]

    @Ts.function
    def expected(
        A_data: T.Tensor((819,), "float32"),
        B: T.Tensor((128,), "float32"),
        A_indptr: T.Tensor((129,), "int32"),
    ) -> None:
        for i in T.serial(128):
            with Ts.sblock("rowsum_outer"):
                Ts.reads(
                    A_indptr[i : i + 1],
                    A_data[A_indptr[i] + 0 : A_indptr[i] + 0 + (A_indptr[i + 1] - A_indptr[i])],
                )
                Ts.writes(B[i])
                with Ts.sblock("rowsum_init"):
                    Ts.reads()
                    Ts.writes(B[i])
                    B[i] = T.float32(0)
                for k in T.serial(A_indptr[i + 1] - A_indptr[i]):
                    with Ts.sblock():
                        Ts.reads(A_indptr[i], A_data[A_indptr[i] + k], B[i])
                        Ts.writes(B[i])
                        A_data_local = Ts.sblock_alloc_tensor([1], dtype="float32", scope="local")
                        with Ts.sblock("A_data_cache_read"):
                            Ts.reads(A_indptr[i], A_data[A_indptr[i] + k])
                            Ts.writes(A_data_local[T.min(A_indptr[i] + k, 0)])
                            A_data_local[T.min(A_indptr[i] + k, 0)] = A_data[A_indptr[i] + k]
                        with Ts.sblock("rowsum_inner"):
                            Ts.reads(B[i], A_indptr[i], A_data[A_indptr[i] + k])
                            Ts.writes(B[i])
                            B[i] = B[i] + A_data_local[T.min(A_indptr[i] + k, 0)]


class TestDataDependentRegion(BaseCompactTest):
    """Partial code of NMS, the `argsort_nms_cpu`'s region depends on inner allocated tensor
    `nkeep`'s value, thus the tensor should not be compacted with data dependent region extent."""

    @Ts.function
    def before(
        p0: T.Tensor((30,), "float32"),
        p1: T.Tensor((1,), "int32"),
        hybrid_nms: T.Tensor((30,), "float32"),
    ):
        argsort_nms_cpu = T.alloc_tensor(ty_args=[T.Tensor([5], "int32", scope="global")])
        for i in range(1):
            nkeep = T.alloc_tensor(ty_args=[T.Tensor([1], "int32", scope="global")])
            if 0 < p1[i]:
                nkeep[0] = p1[i]
                if 2 < nkeep[0]:
                    nkeep[0] = 2
                for j in T.parallel(nkeep[0]):
                    for k in range(6):
                        hybrid_nms[i * 30 + j * 6 + k] = p0[
                            i * 30 + argsort_nms_cpu[i * 5 + j] * 6 + k
                        ]
                    hybrid_nms[i * 5 + j] = argsort_nms_cpu[i * 5 + j]
                if 2 < p1[i]:
                    for j in T.parallel(p1[i] - nkeep[0]):
                        for k in range(6):
                            hybrid_nms[i * 30 + j * 6 + nkeep[0] * 6 + k] = T.float32(-1)
                        hybrid_nms[i * 5 + j + nkeep[0]] = -1

    expected = before


class TestNarrowShape(BaseCompactTest):
    @Ts.function
    def before(A: T.Tensor((10,), "float32"), B: T.Tensor((10,), "float32")) -> None:
        B_cache = Ts.sblock_alloc_tensor(10, "float32")
        for j in T.serial(3):
            for k in T.serial(4):
                with Ts.sblock("B_cache"):
                    Ts.where(j * 4 + k < 10)
                    B_cache[j * 4 + k] = B[j]
        for i in T.serial(10):
            A[i] = B_cache[i] + T.float32(1)

    @Ts.function
    def expected(A: T.Tensor((10,), "float32"), B: T.Tensor((10,), "float32")) -> None:
        B_cache = Ts.sblock_alloc_tensor([10], dtype="float32")
        for j, k in T.grid(3, 4):
            with Ts.sblock("B_cache"):
                Ts.where(j * 4 + k < 10)
                Ts.reads(B[j])
                Ts.writes(B_cache[j * 4 + k])
                B_cache[j * 4 + k] = B[j]
        for i in T.serial(10):
            A[i] = B_cache[i] + T.float32(1)


class TestLetBinding(BaseCompactTest):
    @Ts.function
    def before():
        A = Ts.sblock_alloc_tensor((64, 8), "float32")
        B = Ts.sblock_alloc_tensor((64, 8), "float32")
        C = Ts.sblock_alloc_tensor((8, 8), "float32")
        for rk in range(64):
            for rii, rjj in T.grid(8, 8):
                C[rii, rjj] = T.float32(0)
            for riijj in T.serial(8 * 8):
                rii: T.let[T.int32] = riijj // 8
                rjj: T.let[T.int32] = riijj % 8
                C[rii, rjj] += A[rk, rii] * B[rk, rjj]

    expected = before


class TestNonIndexLetBinding(BaseCompactTest):
    @Ts.function
    def before():
        A = Ts.sblock_alloc_tensor((64), "float32")
        x1: T.let[T.float16] = T.call_extern("get", ty="float16")
        x2: T.let[T.float32] = T.call_extern("get", ty="float32")
        x3: T.let[T.float64] = T.call_extern("get", ty="float64")
        x4: T.let[T.uint8] = T.call_extern("get", ty="uint8")
        x5: T.let[T.int32x16] = T.call_extern("get", ty="int32x16")
        x6: T.let[T.handle] = T.call_extern("get", ty="handle")
        x7: T.let = T.call_extern("get", ty="")
        for rk in range(64):
            A[rk] = T.call_extern("load_ptr", x1, x2, x3, x4, x5, x6, x7, ty="float32")

    expected = before


class TestSpatialTiledPadPooling(BaseCompactTest):
    @Ts.function
    def before(X: T.Tensor((64, 112, 112), "int32"), Y: T.Tensor((64, 56, 56), "int32")) -> None:
        for h_o, w_o in T.grid(14, 14):
            with Ts.sblock():
                X_cache = Ts.sblock_alloc_tensor([112, 112, 64], dtype="int32")
                for ax0, ax1, ax2 in T.grid(64, 9, 9):
                    with Ts.sblock("cache"):
                        Ts.where(1 <= h_o * 8 + ax1 and 1 <= w_o * 8 + ax2)
                        Ts.reads(X[ax0, h_o * 8 - 1 + ax1, w_o * 8 - 1 + ax2])
                        Ts.writes(X_cache[h_o * 8 - 1 + ax1, w_o * 8 - 1 + ax2, ax0])
                        X_cache[h_o * 8 - 1 + ax1, w_o * 8 - 1 + ax2, ax0] = X[
                            ax0, h_o * 8 - 1 + ax1, w_o * 8 - 1 + ax2
                        ]
                for h_i, w_i, kh, kw, c in T.grid(4, 4, 3, 3, 64):
                    with Ts.sblock("compute"):
                        Ts.reads(
                            X_cache[(h_o * 4 + h_i) * 2 + kh - 1, (w_o * 4 + w_i) * 2 + kw - 1, c]
                        )
                        Ts.writes(Y[h_o * 4 + h_i, w_o * 4 + w_i, c])
                        if kh == 0 and kw == 0:
                            Y[h_o * 4 + h_i, w_o * 4 + w_i, c] = 0
                        Y[h_o * 4 + h_i, w_o * 4 + w_i, c] = T.max(
                            Y[h_o * 4 + h_i, w_o * 4 + w_i, c],
                            T.if_then_else(
                                T.likely(1 <= (h_o * 4 + h_i) * 2 + kh)
                                and T.likely((h_o * 4 + h_i) * 2 + kh < 113)
                                and T.likely(1 <= (w_o * 4 + w_i) * 2 + kw)
                                and T.likely((w_o * 4 + w_i) * 2 + kw < 113),
                                X_cache[
                                    (h_o * 4 + h_i) * 2 + kh - 1,
                                    (w_o * 4 + w_i) * 2 + kw - 1,
                                    c,
                                ],
                                0,
                            ),
                        )

    @Ts.function
    def expected(X: T.Tensor((64, 112, 112), "int32"), Y: T.Tensor((64, 56, 56), "int32")) -> None:
        for h_o, w_o in T.grid(14, 14):
            with Ts.sblock():
                Ts.reads(X[0:64, h_o * 8 - 1 : h_o * 8 + 8, w_o * 8 - 1 : w_o * 8 + 8])
                Ts.writes(Y[h_o * 4 : h_o * 4 + 4, w_o * 4 : w_o * 4 + 4, 0:64])
                X_cache = Ts.sblock_alloc_tensor([9, 9, 64], dtype="int32")
                for ax0, ax1, ax2 in T.grid(64, 9, 9):
                    with Ts.sblock("cache"):
                        Ts.where(1 <= h_o * 8 + ax1 and 1 <= w_o * 8 + ax2)
                        Ts.reads(X[ax0, h_o * 8 + ax1 - 1, w_o * 8 + ax2 - 1])
                        Ts.writes(
                            X_cache[
                                h_o * 8 + ax1 - T.max(0, h_o * 8 - 1) - 1,
                                w_o * 8 + ax2 - T.max(0, w_o * 8 - 1) - 1,
                                ax0,
                            ]
                        )
                        X_cache[
                            h_o * 8 + ax1 - T.max(0, h_o * 8 - 1) - 1,
                            w_o * 8 + ax2 - T.max(0, w_o * 8 - 1) - 1,
                            ax0,
                        ] = X[ax0, h_o * 8 + ax1 - 1, w_o * 8 + ax2 - 1]
                for h_i, w_i, kh, kw, c in T.grid(4, 4, 3, 3, 64):
                    with Ts.sblock("compute"):
                        Ts.reads(
                            X_cache[
                                h_o * 8 + h_i * 2 + kh - T.max(0, h_o * 8 - 1) - 1,
                                w_o * 8 + w_i * 2 + kw - T.max(0, w_o * 8 - 1) - 1,
                                c,
                            ]
                        )
                        Ts.writes(Y[h_o * 4 + h_i, w_o * 4 + w_i, c])
                        if kh == 0 and kw == 0:
                            Y[h_o * 4 + h_i, w_o * 4 + w_i, c] = 0
                        Y[h_o * 4 + h_i, w_o * 4 + w_i, c] = T.max(
                            Y[h_o * 4 + h_i, w_o * 4 + w_i, c],
                            T.if_then_else(
                                T.likely(1 <= h_o * 8 + h_i * 2 + kh)
                                and T.likely(1 <= w_o * 8 + w_i * 2 + kw),
                                X_cache[
                                    h_o * 8 + h_i * 2 + kh - T.max(0, h_o * 8 - 1) - 1,
                                    w_o * 8 + w_i * 2 + kw - T.max(0, w_o * 8 - 1) - 1,
                                    c,
                                ],
                                0,
                            ),
                        )


class TestComplexCase1(BaseCompactTest):
    """Meta-schedule matmul case for compact shared A, B matrix"""

    # fmt: off
    @Ts.function
    def before(A: T.Tensor((960, 770), "float32"), B: T.Tensor((770, 2304), "float32"), C: T.Tensor((960, 2304), "float32")) -> None:
        for bx in T.parallel(144, annotations={"thread_binding": "blockIdx.x"}):
            for vx in T.parallel(2, annotations={"thread_binding": "vthread.x"}):
                for tx_p in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock():
                        for k_0 in T.serial(193):
                            with Ts.sblock():
                                A_shared = Ts.sblock_alloc_tensor([960, 770], dtype="float32", scope="shared")
                                B_shared = Ts.sblock_alloc_tensor([770, 2304], dtype="float32", scope="shared")
                                for _u in T.serial(1):
                                    for tx in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                                        for vec in T.vectorized(3):
                                            with Ts.sblock("A_shared"):
                                                Ts.where(bx // 18 * 128 + ((_u * 256 + tx) * 3 + vec) // 4 < 960 and k_0 * 4 + ((_u * 256 + tx) * 3 + vec) % 4 < 770 and (_u * 256 + tx) * 3 + vec < 512)
                                                A_shared[bx // 18 * 128 + (_u * 768 + tx * 3 + vec) // 4, k_0 * 4 + (_u * 768 + tx * 3 + vec) % 4] = A[bx // 18 * 128 + (_u * 768 + tx * 3 + vec) // 4, k_0 * 4 + (_u * 768 + tx * 3 + vec) % 4]
                                for _u in T.serial(1):
                                    for tx in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                                        for vec in T.vectorized(4):
                                            with Ts.sblock("B_shared"):
                                                Ts.where(k_0 * 4 + ((_u * 256 + tx) * 4 + vec) // 128 < 770 and (_u * 256 + tx) * 4 + vec < 512)
                                                B_shared[k_0 * 4 + (_u * 1024 + tx * 4 + vec) // 128, bx % 18 * 128 + (_u * 1024 + tx * 4 + vec) % 128] = B[k_0 * 4 + (_u * 1024 + tx * 4 + vec) // 128, bx % 18 * 128 + (_u * 1024 + tx * 4 + vec) % 128]
                                for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(1, 8, 1, 4, 2, 2):
                                    with Ts.sblock("update_update"):
                                        C[(((bx // 18 + 0) * 8 + tx_p // 32) * 8 + i_3) * 2 + i_4, ((bx % 18 * 2 + vx % 2) * 32 + tx_p % 32 + j_3) * 2 + j_4] = C[(((bx // 18 + 0) * 8 + tx_p // 32) * 8 + i_3) * 2 + i_4, ((bx % 18 * 2 + vx % 2) * 32 + tx_p % 32 + j_3) * 2 + j_4] + A_shared[(((bx // 18 + 0) * 8 + tx_p // 32) * 8 + i_3) * 2 + i_4, (k_0 + k_1) * 4 + k_2] * B_shared[(k_0 + k_1) * 4 + k_2, ((bx % 18 * 2 + vx % 2) * 32 + tx_p % 32 + j_3) * 2 + j_4]

    @Ts.function
    def expected(A: T.Tensor((960, 770), "float32"), B: T.Tensor((770, 2304), "float32"), C: T.Tensor((960, 2304), "float32")) -> None:
        for bx in T.parallel(144, annotations={"thread_binding": "blockIdx.x"}):
            for vx in T.parallel(2, annotations={"thread_binding": "vthread.x"}):
                for tx_p in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock():
                        for k_0 in T.serial(193):
                            with Ts.sblock():
                                A_shared = Ts.sblock_alloc_tensor([128, 4], dtype="float32", scope="shared")
                                B_shared = Ts.sblock_alloc_tensor([4, 128], dtype="float32", scope="shared")
                                for v_u in T.serial(1):
                                    for tx in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                                        for vec in T.vectorized(3):
                                            with Ts.sblock("A_shared"):
                                                Ts.where(bx // 18 * 128 + (tx * 3 + vec) // 4 < 960 and k_0 * 4 + (tx * 3 + vec) % 4 < 770 and tx * 3 + vec < 512)
                                                A_shared[(tx * 3 + vec) // 4, (tx * 3 + vec) % 4] = A[bx // 18 * 128 + (tx * 3 + vec) // 4, k_0 * 4 + (tx * 3 + vec) % 4]
                                for v_u in T.serial(1):
                                    for tx in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                                        for vec in T.vectorized(4):
                                            with Ts.sblock("B_shared"):
                                                Ts.where(k_0 * 4 + tx // 32 < 770 and tx * 4 + vec < 512)
                                                B_shared[tx // 32, tx % 32 * 4 + vec] = B[k_0 * 4 + tx // 32, bx % 18 * 128 + tx % 32 * 4 + vec]
                                for k_1, i_3, j_3, k_2, i_4, j_4 in T.grid(1, 8, 1, 4, 2, 2):
                                    with Ts.sblock("update_update"):
                                        C[bx // 18 * 128 + tx_p // 32 * 16 + i_3 * 2 + i_4, bx % 18 * 128 + vx * 64 + tx_p % 32 * 2 + j_4] = C[bx // 18 * 128 + tx_p // 32 * 16 + i_3 * 2 + i_4, bx % 18 * 128 + vx * 64 + tx_p % 32 * 2 + j_4] + A_shared[tx_p // 32 * 16 + i_3 * 2 + i_4, k_2] * B_shared[k_2, vx * 64 + tx_p % 32 * 2 + j_4]
    # fmt: on


class TestDependentTensorIndices(BaseCompactTest):
    """Check the upper bound on different indices could be independently estimated."""

    @Ts.function
    def before():
        """This is a diagnal tensor access pattern"""
        for i in range(8):
            with Ts.sblock():
                A = Ts.sblock_alloc_tensor((256, 256), "float32")
                for j, k in T.grid(8, 8):
                    with Ts.sblock():
                        Ts.where(j * 8 + k < 60)
                        A[i * 64 + j * 8 + k, i * 64 + j * 8 + k] = 1.0

    @Ts.function
    def expected() -> None:
        for i in T.serial(8):
            with Ts.sblock():
                A = Ts.sblock_alloc_tensor([60, 60], dtype="float32")
                for j, k in T.grid(8, 8):
                    with Ts.sblock():
                        Ts.where(j * 8 + k < 60)
                        A[j * 8 + k, j * 8 + k] = 1.0


class TestDependentTensorIndicesOfPackedMatmul(BaseCompactTest):
    """Check the outer dimension of the packed M-dim should be compacted to 1 wrt split condition."""

    @Ts.function
    def before(
        A: T.Tensor((1020, 64), "float32"),
        B: T.Tensor((1000, 64), "float32"),
        C: T.Tensor((1020, 1000), "float32"),
    ):
        for i0, i1 in T.grid(4, 1):
            with Ts.sblock():
                C_local2 = Ts.sblock_alloc_tensor(
                    [4, 1, 16, 1000, 16], dtype="float32", scope="local"
                )
                C_local1 = Ts.sblock_alloc_tensor([1020, 1000], dtype="float32", scope="local")
                for ax0, ax1, ax2 in T.grid(255, 1000, 64):
                    with Ts.sblock("matmul"):
                        if ax2 == 0:
                            C_local1[i0 * 255 + ax0, ax1] = 0
                        C_local1[i0 * 255 + ax0, ax1] = (
                            C_local1[i0 * 255 + ax0, ax1] + A[i0 * 255 + ax0, ax2] * B[ax1, ax2]
                        )
                for ax0, ax1 in T.grid(255, 1000):
                    with Ts.sblock("st1"):
                        C_local2[
                            (i0 * 255 + ax0) // 255,
                            0,
                            (i0 * 255 + ax0) % 255 // 16,
                            ax1,
                            (i0 * 255 + ax0) % 255 % 16,
                        ] = C_local1[i0 * 255 + ax0, ax1]
                for ax0, ax1, ax2 in T.grid(16, 16, 1000):
                    with Ts.sblock("st2"):
                        Ts.where(ax0 * 16 + ax1 < 255)
                        C[i0 * 255 + (ax0 * 16 + ax1), i1 * 1000 + ax2] = C_local2[
                            (i0 * 255 + ax0 * 16 + ax1) // 255,
                            0,
                            (i0 * 255 + ax0 * 16 + ax1) % 255 // 16,
                            i1 * 1000 + ax2,
                            (i0 * 255 + ax0 * 16 + ax1) % 255 % 16,
                        ]

    @Ts.function
    def expected(
        A: T.Tensor((1020, 64), "float32"),
        B: T.Tensor((1000, 64), "float32"),
        C: T.Tensor((1020, 1000), "float32"),
    ) -> None:
        for i0, i1 in T.grid(4, 1):
            with Ts.sblock():
                C_local2 = Ts.sblock_alloc_tensor(
                    [1, 1, 16, 1000, 16], dtype="float32", scope="local"
                )
                C_local1 = Ts.sblock_alloc_tensor([255, 1000], dtype="float32", scope="local")
                for ax0, ax1, ax2 in T.grid(255, 1000, 64):
                    with Ts.sblock("matmul"):
                        if ax2 == 0:
                            C_local1[ax0, ax1] = 0
                        C_local1[ax0, ax1] = (
                            C_local1[ax0, ax1] + A[i0 * 255 + ax0, ax2] * B[ax1, ax2]
                        )
                for ax0, ax1 in T.grid(255, 1000):
                    with Ts.sblock("st1"):
                        C_local2[0, 0, ax0 // 16, ax1, ax0 % 16] = C_local1[ax0, ax1]
                for ax0, ax1, ax2 in T.grid(16, 16, 1000):
                    with Ts.sblock("st2"):
                        Ts.where(ax0 * 16 + ax1 < 255)
                        C[i0 * 255 + ax0 * 16 + ax1, ax2] = C_local2[
                            (ax0 * 16 + ax1) // 255,
                            0,
                            (ax0 * 16 + ax1) % 255 // 16,
                            ax2,
                            (ax0 * 16 + ax1) % 255 % 16,
                        ]


class TestTileAwareCompaction(BaseCompactTest):
    """Each partitioned tile could be independently compacted."""

    # it is not an opaque block case intentionally
    is_lower_order_free = False

    @property
    def before(self):
        @Ts.function
        def main(
            A: T.Tensor((128, 128), "float32"),
            B: T.Tensor((128, 128), "float32"),
            C: T.Tensor((128, 128), "float32"),
        ):
            for i_0 in range(5, annotations={"loop_partition_hint": 1}):
                for j_0 in range(5, annotations={"loop_partition_hint": 1}):
                    A_local = T.alloc_tensor(
                        ty_args=[T.Tensor((26, 128), "float32", scope="local")]
                    )
                    B_local = T.alloc_tensor(
                        ty_args=[T.Tensor((128, 26), "float32", scope="local")]
                    )
                    C_local = T.alloc_tensor(ty_args=[T.Tensor((26, 26), "float32", scope="local")])
                    for ax0, ax1 in T.grid(26, 128):
                        if i_0 * 26 + ax0 < 128:
                            A_local[ax0, ax1] = A[i_0 * 26 + ax0, ax1]
                    for ax0, ax1 in T.grid(128, 26):
                        if j_0 * 26 + ax1 < 128:
                            B_local[ax0, ax1] = B[ax0, j_0 * 26 + ax1]
                    for i_1, j_1, k in T.grid(26, 26, 128):
                        if i_0 * 26 + i_1 < 128 and j_0 * 26 + j_1 < 128:
                            if k == 0:
                                C_local[i_1, j_1] = T.float32(0)
                            C_local[i_1, j_1] = (
                                C_local[i_1, j_1] + A_local[i_1, k] * B_local[k, j_1]
                            )
                    for ax0, ax1 in T.grid(26, 26):
                        if i_0 * 26 + ax0 < 128 and j_0 * 26 + ax1 < 128:
                            C[i_0 * 26 + ax0, j_0 * 26 + ax1] = C_local[ax0, ax1]

        # Get partitioned workload to compact
        mod = tvm_tirx.IRModule.from_expr(main)
        with tvm_tirx.ir.transform.PassContext(
            config={"s_tir.LoopPartition": {"partition_const_loop": True}}
        ):
            mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
            mod = tvm.s_tir.transform.LoopPartition()(mod)

        return mod["main"]

    @Ts.function
    def expected(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        for i_0 in range(4):
            for j_0 in range(4):
                A_local_tile0 = T.alloc_tensor(
                    ty_args=[T.Tensor((26, 128), "float32", scope="local")]
                )
                B_local_tile0 = T.alloc_tensor(
                    ty_args=[T.Tensor((128, 26), "float32", scope="local")]
                )
                C_local_tile0 = T.alloc_tensor(
                    ty_args=[T.Tensor((26, 26), "float32", scope="local")]
                )
                for ax0, ax1 in T.grid(26, 128):
                    A_local_tile0[ax0, ax1] = A[i_0 * 26 + ax0, ax1]
                for ax0, ax1 in T.grid(128, 26):
                    B_local_tile0[ax0, ax1] = B[ax0, j_0 * 26 + ax1]
                for i_1, j_1, k in T.grid(26, 26, 128):
                    if k == 0:
                        C_local_tile0[i_1, j_1] = T.float32(0)
                    C_local_tile0[i_1, j_1] = (
                        C_local_tile0[i_1, j_1] + A_local_tile0[i_1, k] * B_local_tile0[k, j_1]
                    )
                for ax0, ax1 in T.grid(26, 26):
                    C[i_0 * 26 + ax0, j_0 * 26 + ax1] = C_local_tile0[ax0, ax1]

            A_local_tile1 = T.alloc_tensor(ty_args=[T.Tensor((26, 128), "float32", scope="local")])
            B_local_tile1 = T.alloc_tensor(ty_args=[T.Tensor((128, 24), "float32", scope="local")])
            C_local_tile1 = T.alloc_tensor(ty_args=[T.Tensor((26, 24), "float32", scope="local")])
            for ax0, ax1 in T.grid(26, 128):
                A_local_tile1[ax0, ax1] = A[i_0 * 26 + ax0, ax1]
            for ax0, ax1 in T.grid(128, 26):
                if ax1 < 24:
                    B_local_tile1[ax0, ax1] = B[ax0, ax1 + 104]
            for i_1, j_1, k in T.grid(26, 26, 128):
                if j_1 < 24:
                    if k == 0:
                        C_local_tile1[i_1, j_1] = T.float32(0)
                    C_local_tile1[i_1, j_1] = (
                        C_local_tile1[i_1, j_1] + A_local_tile1[i_1, k] * B_local_tile1[k, j_1]
                    )
            for ax0, ax1 in T.grid(26, 26):
                if ax1 < 24:
                    C[i_0 * 26 + ax0, ax1 + 104] = C_local_tile1[ax0, ax1]

        for j_0 in range(4):
            A_local_tile2 = T.alloc_tensor(ty_args=[T.Tensor((24, 128), "float32", scope="local")])
            B_local_tile2 = T.alloc_tensor(ty_args=[T.Tensor((128, 26), "float32", scope="local")])
            C_local_tile2 = T.alloc_tensor(ty_args=[T.Tensor((24, 26), "float32", scope="local")])
            for ax0, ax1 in T.grid(26, 128):
                if ax0 < 24:
                    A_local_tile2[ax0, ax1] = A[ax0 + 104, ax1]
            for ax0, ax1 in T.grid(128, 26):
                B_local_tile2[ax0, ax1] = B[ax0, j_0 * 26 + ax1]
            for i_1, j_1, k in T.grid(26, 26, 128):
                if i_1 < 24:
                    if k == 0:
                        C_local_tile2[i_1, j_1] = T.float32(0)
                    C_local_tile2[i_1, j_1] = (
                        C_local_tile2[i_1, j_1] + A_local_tile2[i_1, k] * B_local_tile2[k, j_1]
                    )
            for ax0, ax1 in T.grid(26, 26):
                if ax0 < 24:
                    C[ax0 + 104, j_0 * 26 + ax1] = C_local_tile2[ax0, ax1]

        A_local_tile3 = T.alloc_tensor(ty_args=[T.Tensor((24, 128), "float32", scope="local")])
        B_local_tile3 = T.alloc_tensor(ty_args=[T.Tensor((128, 24), "float32", scope="local")])
        C_local_tile3 = T.alloc_tensor(ty_args=[T.Tensor((24, 24), "float32", scope="local")])
        for ax0, ax1 in T.grid(26, 128):
            if ax0 < 24:
                A_local_tile3[ax0, ax1] = A[ax0 + 104, ax1]
        for ax0, ax1 in T.grid(128, 26):
            if ax1 < 24:
                B_local_tile3[ax0, ax1] = B[ax0, ax1 + 104]
        for i_1, j_1, k in T.grid(26, 26, 128):
            if i_1 < 24 and j_1 < 24:
                if k == 0:
                    C_local_tile3[i_1, j_1] = T.float32(0)
                C_local_tile3[i_1, j_1] = (
                    C_local_tile3[i_1, j_1] + A_local_tile3[i_1, k] * B_local_tile3[k, j_1]
                )
        for ax0, ax1 in T.grid(26, 26):
            if ax0 < 24 and ax1 < 24:
                C[ax0 + 104, ax1 + 104] = C_local_tile3[ax0, ax1]


class TestNonStrictCompactionForPaddedMatmul(BaseCompactTest):
    is_strict_mode = False

    @Ts.function
    def before(
        A: T.Tensor((127, 127), "float32"),
        B: T.Tensor((127, 127), "float32"),
        C: T.Tensor((127, 127), "float32"),
    ):
        """A mock workload where the intermediate tensor allocation is not enought originally"""
        for i_0, j_0 in T.grid(4, 4):
            with Ts.sblock(""):
                Ts.reads(A[i_0 * 32 : i_0 * 32 + 32, 0:128], B[0:128, j_0 * 32 : j_0 * 32 + 32])
                Ts.writes(C[i_0 * 32 : i_0 * 32 + 32, j_0 * 32 : j_0 * 32 + 32])
                A_local = Ts.sblock_alloc_tensor((127, 127), scope="local")
                B_local = Ts.sblock_alloc_tensor((127, 127), scope="local")
                C_local = Ts.sblock_alloc_tensor((127, 127), scope="local")
                for ax0, ax1 in T.grid(32, 128):
                    with Ts.sblock("A_local"):
                        A_local[i_0 * 32 + ax0, ax1] = T.if_then_else(
                            i_0 * 32 + ax0 < 127, A[i_0 * 32 + ax0, ax1], 0.0
                        )
                for ax0, ax1 in T.grid(128, 32):
                    with Ts.sblock("B_local"):
                        B_local[ax0, j_0 * 32 + ax1] = T.if_then_else(
                            j_0 * 32 + ax1 < 127, B[ax0, j_0 * 32 + ax1], 0.0
                        )
                for i_1, j_1, k in T.grid(32, 32, 128):
                    with Ts.sblock("compute"):
                        Ts.where(i_0 * 32 + i_1 < 127 and j_0 * 32 + j_1 < 127)
                        if k == 0:
                            C_local[i_0 * 32 + i_1, j_0 * 32 + j_1] = T.float32(0)
                        C_local[i_0 * 32 + i_1, j_0 * 32 + j_1] = (
                            C_local[i_0 * 32 + i_1, j_0 * 32 + j_1]
                            + A_local[i_0 * 32 + i_1, k] * B_local[k, j_0 * 32 + j_1]
                        )
                for ax0, ax1 in T.grid(32, 32):
                    with Ts.sblock("C_local"):
                        Ts.where(i_0 * 32 + ax0 < 127 and j_0 * 32 + ax1 < 127)
                        C[i_0 * 32 + ax0, j_0 * 32 + ax1] = C_local[i_0 * 32 + ax0, j_0 * 32 + ax1]

    @Ts.function
    def expected(
        A: T.Tensor((127, 127), "float32"),
        B: T.Tensor((127, 127), "float32"),
        C: T.Tensor((127, 127), "float32"),
    ):
        for i_0, j_0 in T.grid(4, 4):
            with Ts.sblock(""):
                Ts.reads(A[i_0 * 32 : i_0 * 32 + 32, 0:128], B[0:128, j_0 * 32 : j_0 * 32 + 32])
                Ts.writes(C[i_0 * 32 : i_0 * 32 + 32, j_0 * 32 : j_0 * 32 + 32])
                A_local = Ts.sblock_alloc_tensor((32, 128), scope="local")
                B_local = Ts.sblock_alloc_tensor((128, 32), scope="local")
                C_local = Ts.sblock_alloc_tensor((32, 32), scope="local")
                for ax0, ax1 in T.grid(32, 128):
                    with Ts.sblock("A_local"):
                        A_local[ax0, ax1] = T.if_then_else(
                            i_0 * 32 + ax0 < 127, A[i_0 * 32 + ax0, ax1], T.float32(0)
                        )
                for ax0, ax1 in T.grid(128, 32):
                    with Ts.sblock("B_local"):
                        B_local[ax0, ax1] = T.if_then_else(
                            j_0 * 32 + ax1 < 127, B[ax0, j_0 * 32 + ax1], T.float32(0)
                        )
                for i_1, j_1, k in T.grid(32, 32, 128):
                    with Ts.sblock("compute"):
                        Ts.where(i_0 * 32 + i_1 < 127 and j_0 * 32 + j_1 < 127)
                        if k == 0:
                            C_local[i_1, j_1] = T.float32(0)
                        C_local[i_1, j_1] = C_local[i_1, j_1] + A_local[i_1, k] * B_local[k, j_1]
                for ax0, ax1 in T.grid(32, 32):
                    with Ts.sblock("C_local"):
                        Ts.where(i_0 * 32 + ax0 < 127 and j_0 * 32 + ax1 < 127)
                        C[i_0 * 32 + ax0, j_0 * 32 + ax1] = C_local[ax0, ax1]


class TestNotCompactAliasTensor(BaseCompactTest):
    # it is not testcase on block form
    is_lower_order_free = False

    @Ts.function
    def before():
        """Partially accessed tensor, but should not compact
        because existence of aliasing tensor B."""
        data = T.alloc_tensor(ty_args=[T.Tensor((1024,), "int8")])
        A = T.decl_tensor(data.data, ty_args=[T.Tensor([1024], "int8", data=data.data)])
        B = T.decl_tensor(data.data, ty_args=[T.Tensor([512], "float16", data=data.data)])
        for i in range(10):
            A[i] = A[i] + T.int8(1)
        for i in range(10):
            B[i] = B[i] + T.float16(1)

    expected = before


class TestNotCompactTensorWithDifferentDtype(BaseCompactTest):
    # it is not testcase on block form
    is_lower_order_free = False

    @Ts.function
    def before():
        """Partially accessed tensor, but should not compact
        because existence of aliasing tensor B."""
        data = T.alloc_tensor(ty_args=[T.Tensor((1024,), "int8")])
        A = T.decl_tensor(data.data, ty_args=[T.Tensor([256], "int32", data=data.data)])
        for i in range(10):
            A[i] = A[i] + 1

    expected = before


class TestNonBoolCondition(BaseCompactTest):
    # it is not testcase on block form
    is_lower_order_free = False

    @Ts.function
    def before():
        A = T.alloc_tensor(ty_args=[T.Tensor([12], "int32")])
        for i in range(10):
            if i:
                A[i] = A[i] + 1

    @Ts.function
    def expected():
        A = T.alloc_tensor(ty_args=[T.Tensor((9,), "int32")])
        for i in range(10):
            if i:
                A[i - 1] = A[i - 1] + 1


def test_loop_var_does_not_escape_compacted_tensor_extent():
    n = T.dynamic("n")

    @Ts.function(private=True)
    def before(A: T.Tensor((n,), "int32")):
        tmp = T.alloc_tensor(ty_args=[T.Tensor((n,), "int32")])
        for i in range(n):
            length: T.let[T.int64] = T.ceildiv(n, T.shift_left(T.int64(1), i + 1))
            for j in range(length):
                tmp[j] = A[j]

    after = s_tir.transform.CompactTensorAllocation()(tvm_tirx.IRModule.from_expr(before))
    s_tir.analysis.check_well_formed(after)


class TestCompactSymbolicBound0:
    """Test symbolic bound that get compacted to constant"""

    @Ts.function
    def before(
        X: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
        Y: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
        n: T.int64,
    ):
        for i, k_0 in T.grid(T.int64(8), n):
            with Ts.sblock(""):
                X_global = Ts.sblock_alloc_tensor((T.int64(8), n * T.int64(32)))
                for ax0 in range(T.int64(32)):
                    with Ts.sblock("X_global"):
                        X_global[i, k_0 * T.int64(32) + ax0] = X[i, k_0 * T.int64(32) + ax0]
                for k_1 in range(T.int64(32)):
                    with Ts.sblock("Y"):
                        Y[i, k_0 * T.int64(32) + k_1] = X_global[i, k_0 * T.int64(32) + k_1]

    @Ts.function
    def expected(
        X: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
        Y: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
        n: T.int64,
    ):
        for i, k_0 in T.grid(T.int64(8), n):
            with Ts.sblock(""):
                X_global = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32)))
                for ax0 in range(T.int64(32)):
                    with Ts.sblock("X_global"):
                        X_global[T.int64(0), ax0] = X[i, k_0 * T.int64(32) + ax0]
                for k_1 in range(T.int64(32)):
                    with Ts.sblock("Y"):
                        Y[i, k_0 * T.int64(32) + k_1] = X_global[T.int64(0), k_1]


class TestCompactSymbolicBound1:
    """Test symbolic bound that get compacted to constant"""

    @Ts.function
    def before(
        X: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
        Y: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
        n: T.int64,
    ):
        for i, k_0 in T.grid(T.int64(8), n):
            with Ts.sblock(""):
                X_global = Ts.sblock_alloc_tensor((T.int64(8), n * T.int64(32)))
                with Ts.sblock("X_global"):
                    for x0 in range(T.int64(32)):
                        X_global[i, k_0 * T.int64(32) + x0] = X[i, k_0 * T.int64(32) + x0]
                with Ts.sblock("Y"):
                    for x1 in range(T.int64(32)):
                        Y[i, k_0 * T.int64(32) + x1] = X_global[i, k_0 * T.int64(32) + x1]

    @Ts.function
    def expected(
        X: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
        Y: T.Tensor((T.int64(8), n * T.int64(32)), "float32"),  # noqa: F821
        n: T.int64,
    ):
        # with Ts.sblock("root"):
        for i, k_0 in T.grid(T.int64(8), n):
            with Ts.sblock(""):
                X_global = Ts.sblock_alloc_tensor((T.int64(1), T.int64(32)))
                with Ts.sblock("X_global"):
                    for x0 in range(T.int64(32)):
                        X_global[T.int64(0), x0] = X[i, k_0 * T.int64(32) + x0]
                with Ts.sblock("Y"):
                    for x1 in range(T.int64(32)):
                        Y[i, k_0 * T.int64(32) + x1] = X_global[T.int64(0), x1]


class TestSymbolicDiagMaskCase:
    """Test symbolic allocation not too complex"""

    @Ts.function
    def before(A: T.Tensor((1, 1, n, n), "float32"), n: T.int32):  # noqa: F821
        B = Ts.sblock_alloc_tensor((n, n))
        for i in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
            for j in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                for k in range((n * n + 65535) // 65536):
                    with Ts.sblock("make_diag_mask_te"):
                        Ts.where((k * 256 + i) * 256 + j < n * n)
                        Ts.reads()
                        Ts.writes(B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n])
                        B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n] = T.Select(
                            (k * 65536 + i * 256 + j) // n < (k * 65536 + i * 256 + j) % n,
                            T.float32(-3.4028234663852886e38),
                            T.float32(3.4028234663852886e38),
                        )
        for i in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
            for j in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                for k in range((n * n + 65535) // 65536):
                    with Ts.sblock("T_broadcast_to"):
                        Ts.where((k * 256 + i) * 256 + j < n * n)
                        Ts.reads(B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n])
                        Ts.writes(
                            A[0, 0, (k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n]
                        )
                        A[0, 0, (k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n] = B[
                            (k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n
                        ]

    @Ts.function
    def expected(A: T.Tensor((1, 1, n, n), "float32"), n: T.int32):  # noqa: F821
        B = Ts.sblock_alloc_tensor((n, n))
        for i in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
            for j in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                for k in range((n * n + 65535) // 65536):
                    with Ts.sblock("make_diag_mask_te"):
                        Ts.where(k * 65536 + i * 256 + j < n * n)
                        Ts.reads()
                        Ts.writes(B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n])
                        B[(k * 65536 + i * 256 + j) // n, (k * 65536 + i * 256 + j) % n] = T.Select(
                            (k * 65536 + i * 256 + j) // n < (k * 65536 + i * 256 + j) % n,
                            T.float32(-3.4028234663852886e38),
                            T.float32(3.4028234663852886e38),
                        )
        for i in T.parallel(256, annotations={"thread_binding": "blockIdx.x"}):
            for k in T.parallel(256, annotations={"thread_binding": "threadIdx.x"}):
                for k in range((n * n + 65535) // 65536):
                    with Ts.sblock("T_broadcast_to"):
                        Ts.where(k * 65536 + i * 256 + k < n * n)
                        Ts.reads(B[(k * 65536 + i * 256 + k) // n, (k * 65536 + i * 256 + k) % n])
                        Ts.writes(
                            A[0, 0, (k * 65536 + i * 256 + k) // n, (k * 65536 + i * 256 + k) % n]
                        )
                        A[0, 0, (k * 65536 + i * 256 + k) // n, (k * 65536 + i * 256 + k) % n] = B[
                            (k * 65536 + i * 256 + k) // n, (k * 65536 + i * 256 + k) % n
                        ]


if __name__ == "__main__":
    tvm_tirx.testing.main()
