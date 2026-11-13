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

import enum

import pytest

import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


class OpPatternKind(enum.IntEnum):
    kElemWise = 0
    kBroadcast = 1
    kInjective = 2
    kCommReduce = 3
    kOutEWiseFusable = 4
    kTuple = 7
    kOpaque = 8


def test_annotate_opkind_outewisefusable():
    m = T.dynamic("m", "int32")
    n = T.dynamic("n", "int32")
    k = T.dynamic("k", "int32")

    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def tir_matmul(
            A: T.Tensor((m, n), "float32"),
            B: T.Tensor((n, k), "float32"),
            C: T.Tensor((m, k), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})

            for i, j, k_index in T.grid(m, k, n):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k_index])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_matmul"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


@pytest.mark.parametrize(
    "cast_pattern",
    [
        lambda a, b: T.Cast("float32", a * b),
        lambda a, b: T.Cast("float32", a) * T.Cast("float32", b),
        lambda a, b: T.Cast("float32", T.Cast("float16", a * b)),
    ],
)
def test_annotate_opkind_outewisefusable_with_cast(cast_pattern):
    m = T.dynamic("m", "int32")
    n = T.dynamic("n", "int32")
    k = T.dynamic("k", "int32")

    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def tir_matmul(
            A: T.Tensor((m, n), "float16"),
            B: T.Tensor((n, k), "float16"),
            C: T.Tensor((m, k), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})

            for i, j, k_index in T.grid(m, k, n):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k_index])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + cast_pattern(A[vi, vk], B[vk, vj])

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_matmul"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


def test_annotate_opkind_outewisefusable_int_var_signature():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def tir_matmul(
            A: T.Tensor((m, n), "float32"),  # noqa: F821
            B: T.Tensor((n, k), "float32"),  # noqa: F821
            C: T.Tensor((m, k), "float32"),  # noqa: F821
            m: T.int64,
            n: T.int64,
            k: T.int64,
        ):
            T.func_attr({"global_symbol": "tir_matmul"})

            for i, j, k_index in T.grid(m, k, n):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k_index])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_matmul"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


def test_annotate_opkind_reduce():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def sum(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16,), "float32")) -> None:
            T.func_attr({"global_symbol": "elemwise"})

            for i, j in T.grid(16, 16):
                with Ts.sblock("matmul"):
                    vi, vj = Ts.axis.remap("SR", [i, j])
                    with Ts.init():
                        B[vi] = 0.0
                    B[vi] += A[vi, vj]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["sum"].attrs["op_pattern"] == OpPatternKind.kCommReduce


def test_annotate_opkind_ewise():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def elemwise(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")) -> None:
            T.func_attr({"global_symbol": "elemwise"})

            for i, j in T.grid(16, 16):
                with Ts.sblock("matmul"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + 1.0

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["elemwise"].attrs["op_pattern"] == OpPatternKind.kElemWise


def test_annotate_opkind_broadcast():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def broadcast(
            A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16, 16, 16), "float32")
        ) -> None:
            T.func_attr({"global_symbol": "elemwise"})

            for i0, j0, i1, j1 in T.grid(16, 16, 16, 16):
                with Ts.sblock("matmul"):
                    vi0, vj0, vi1, vj1 = Ts.axis.remap("SSSS", [i0, j0, i1, j1])
                    B[vi0, vj0, vi1, vj1] = A[vj0, vj1]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["broadcast"].attrs["op_pattern"] == OpPatternKind.kBroadcast


def test_annotate_opkind_injective():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def injective(
            A: T.Tensor((4, 4, 4, 4), "float32"), B: T.Tensor((16, 16), "float32")
        ) -> None:
            T.func_attr({"global_symbol": "elemwise"})

            for i, j in T.grid(16, 16):
                with Ts.sblock("matmul"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi // 4, vj // 4, vi % 4, vj % 4]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["injective"].attrs["op_pattern"] == OpPatternKind.kInjective


def test_annotate_opkind_bias_add():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def tir_bias_add(
            A: T.Tensor((1, 1000), "float32"),
            B: T.Tensor((1000,), "float32"),
            C: T.Tensor((1, 1000), "float32"),
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "tir_bias_add", "tirx.noalias": True})
            # body
            # with Ts.sblock("root")
            for i0, i1 in T.grid(1, 1000):
                with Ts.sblock("T_add"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[ax0, ax1], B[ax1])
                    Ts.writes(C[ax0, ax1])
                    C[ax0, ax1] = A[ax0, ax1] + B[ax1]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["tir_bias_add"].attrs["op_pattern"] == OpPatternKind.kElemWise


def test_annotate_opkind_add_broadcast_with_unit_shape():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def add_with_unit_dim_len_broadcast(
            A: T.Tensor((1, 64, 112, 112), "float32"),
            B: T.Tensor((64, 1, 1), "float32"),
            C: T.Tensor((1, 64, 112, 112), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "add5", "tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(1, 64, 112, 112):
                with Ts.sblock("T_add"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(A[ax0, ax1, ax2, ax3], B[ax1, 0, 0])
                    Ts.writes(C[ax0, ax1, ax2, ax3])
                    C[ax0, ax1, ax2, ax3] = A[ax0, ax1, ax2, ax3] + B[ax1, 0, 0]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["add_with_unit_dim_len_broadcast"].attrs["op_pattern"] == OpPatternKind.kElemWise


def test_annotate_opkind_add_zero_dim_element_wise():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def add_zero_dim(
            A: T.Tensor((128,), "float32"),
            B: T.Tensor((), "float32"),
            C: T.Tensor((128,), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "add8", "tirx.noalias": True})
            for i0 in T.serial(128):
                with Ts.sblock("T_add"):
                    ax0 = Ts.axis.spatial(128, i0)
                    Ts.reads(A[ax0], B[()])
                    Ts.writes(C[ax0])
                    C[ax0] = A[ax0] + B[()]

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["add_zero_dim"].attrs["op_pattern"] == OpPatternKind.kElemWise


def test_annotate_opkind_pooling():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def max_pool2d(
            rxplaceholder_1: T.Tensor((1, 64, 112, 112), "float32"),
            tensor_1: T.Tensor((1, 64, 56, 56), "float32"),
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "max_pool2d", "T.noalias": True})
            # body
            # with Ts.sblock("root")
            pad_temp_1 = Ts.sblock_alloc_tensor([1, 64, 114, 114], dtype="float32")
            for i0, i1, i2, i3 in T.grid(1, 64, 114, 114):
                with Ts.sblock("pad_temp"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_1[ax0, ax1, ax2 - 1, ax3 - 1])
                    Ts.writes(pad_temp_1[ax0, ax1, ax2, ax3])
                    pad_temp_1[ax0, ax1, ax2, ax3] = T.if_then_else(
                        1 <= ax2 and ax2 < 113 and 1 <= ax3 and ax3 < 113,
                        rxplaceholder_1[ax0, ax1, ax2 - 1, ax3 - 1],
                        T.float32(-3.4028234663852886e38),
                    )
            for i0, i1, i2, i3, i4, i5 in T.grid(1, 64, 56, 56, 3, 3):
                with Ts.sblock("tensor"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(
                        tensor_1[ax0, ax1, ax2, ax3],
                        pad_temp_1[ax0, ax1, ax2 * 2 + rv0, ax3 * 2 + rv1],
                    )
                    Ts.writes(tensor_1[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        tensor_1[ax0, ax1, ax2, ax3] = T.float32(-3.4028234663852886e38)
                    tensor_1[ax0, ax1, ax2, ax3] = T.max(
                        tensor_1[ax0, ax1, ax2, ax3],
                        pad_temp_1[ax0, ax1, ax2 * 2 + rv0, ax3 * 2 + rv1],
                    )

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["max_pool2d"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


def test_annotate_opkind_softmax():
    @tvm_tirx.script.ir_module
    class InputModule:
        @Ts.function
        def softmax(
            rxplaceholder_1: T.Tensor((16, 16), "float32"),
            T_softmax_norm_1: T.Tensor((16, 16), "float32"),
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "softmax", "T.noalias": True})
            # body
            # with Ts.sblock("root")
            T_softmax_maxelem_1 = Ts.sblock_alloc_tensor([16], dtype="float32")
            T_softmax_exp_1 = Ts.sblock_alloc_tensor([16, 16], dtype="float32")
            T_softmax_expsum_1 = Ts.sblock_alloc_tensor([16], dtype="float32")
            for i0_7, i1_3 in T.grid(16, 16):
                with Ts.sblock("T_softmax_maxelem"):
                    i0_8, k = Ts.axis.remap("SR", [i0_7, i1_3])
                    Ts.reads(T_softmax_maxelem_1[i0_8], rxplaceholder_1[i0_8, k])
                    Ts.writes(T_softmax_maxelem_1[i0_8])
                    with Ts.init():
                        T_softmax_maxelem_1[i0_8] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem_1[i0_8] = T.max(
                        T_softmax_maxelem_1[i0_8], rxplaceholder_1[i0_8, k]
                    )
            for i0_9, i1_4 in T.grid(16, 16):
                with Ts.sblock("T_softmax_exp"):
                    i0_10, i1_5 = Ts.axis.remap("SS", [i0_9, i1_4])
                    Ts.reads(rxplaceholder_1[i0_10, i1_5], T_softmax_maxelem_1[i0_10])
                    Ts.writes(T_softmax_exp_1[i0_10, i1_5])
                    T_softmax_exp_1[i0_10, i1_5] = T.exp(
                        rxplaceholder_1[i0_10, i1_5] - T_softmax_maxelem_1[i0_10]
                    )
            for i0_11, i1_6 in T.grid(16, 16):
                with Ts.sblock("T_softmax_expsum"):
                    i0_12, k = Ts.axis.remap("SR", [i0_11, i1_6])
                    Ts.reads(T_softmax_expsum_1[i0_12], T_softmax_exp_1[i0_12, k])
                    Ts.writes(T_softmax_expsum_1[i0_12])
                    with Ts.init():
                        T_softmax_expsum_1[i0_12] = T.float32(0)
                    T_softmax_expsum_1[i0_12] = (
                        T_softmax_expsum_1[i0_12] + T_softmax_exp_1[i0_12, k]
                    )
            for i0_13, i1_7 in T.grid(16, 16):
                with Ts.sblock("T_softmax_norm"):
                    i0_14, i1_8 = Ts.axis.remap("SS", [i0_13, i1_7])
                    Ts.reads(T_softmax_exp_1[i0_14, i1_8], T_softmax_expsum_1[i0_14])
                    Ts.writes(T_softmax_norm_1[i0_14, i1_8])
                    Ts.sblock_attr({"axis": 1})
                    T_softmax_norm_1[i0_14, i1_8] = (
                        T_softmax_exp_1[i0_14, i1_8] / T_softmax_expsum_1[i0_14]
                    )

    mod = InputModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["softmax"].attrs["op_pattern"] == OpPatternKind.kOutEWiseFusable


def test_multiple_bufer_stores_fallback():
    cumsummodule_rxplaceholder_offset = T.dynamic("cumsummodule_rxplaceholder_offset", "int32")

    @tvm_tirx.script.ir_module
    class CumsumModule:
        @Ts.function
        def cumsum(
            rxplaceholder: T.Tensor(
                [10, 16],
                dtype="float32",
                layout=T.TileLayout(T.S[[10, 16]] + (cumsummodule_rxplaceholder_offset)),
            ),
            out_buf: T.Tensor(160, "float32"),
        ):
            with Ts.sblock("cumsum_generic"):
                Ts.reads(rxplaceholder[0:10, 0:16])
                Ts.writes(out_buf[0:160])
                for fused in T.parallel(1):
                    out_buf[fused * 160] = rxplaceholder[fused * 160 // 16, fused * 160 % 16]
                    for v_k in T.serial(159):
                        out_buf[fused * 160 + (v_k + 1)] = (
                            out_buf[fused * 160 + (v_k + 1 - 1)]
                            + rxplaceholder[
                                (fused * 160 + (v_k + 1)) // 16,
                                (fused * 160 + (v_k + 1)) % 16,
                            ]
                        )

    mod = CumsumModule
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["cumsum"].attrs["op_pattern"] == OpPatternKind.kOpaque


def test_sum_sqsum():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def sum_sqsum(
            A: T.Tensor((32, 64), "float32"),
            vsum: T.Tensor((32,), "float32"),
            sqsum: T.Tensor((32,), "float32"),
        ):
            for ax0, k0 in T.grid(32, 64):
                with Ts.sblock("block"):
                    v_ax0, v_k0 = Ts.axis.remap("SR", [ax0, k0])
                    Ts.reads(A[v_ax0, v_k0])
                    Ts.writes(vsum[v_ax0], sqsum[v_ax0])
                    with Ts.init():
                        vsum[v_ax0] = T.float32(0)
                        sqsum[v_ax0] = T.float32(0)
                    v_vsum: T.let[T.float32] = vsum[v_ax0] + A[v_ax0, v_k0]
                    v_sqsum: T.let[T.float32] = sqsum[v_ax0] + A[v_ax0, v_k0] * A[v_ax0, v_k0]
                    vsum[v_ax0] = v_vsum
                    sqsum[v_ax0] = v_sqsum

    mod = Module
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["sum_sqsum"].attrs["op_pattern"] == OpPatternKind.kCommReduce


def test_no_tensor_stores():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def no_tensor_stores(A: T.Tensor((32, 64), "float32"), vsum: T.Tensor((32,), "float32")):
            for ax0, k0 in T.grid(32, 64):
                with Ts.sblock("block"):
                    v_ax0, v_k0 = Ts.axis.remap("SR", [ax0, k0])
                    Ts.reads(A[v_ax0, v_k0])
                    Ts.writes(vsum[v_ax0])
                    # absence of buffer stores usually happens when there is an external call for
                    # computation. We assume opaque in all such cases.
                    T.call_packed("some_func")

    mod = Module
    new_mod = relax.transform.AnnotateTIROpPattern()(mod)
    assert new_mod["no_tensor_stores"].attrs["op_pattern"] == OpPatternKind.kOpaque


if __name__ == "__main__":
    tvm_tirx.testing.main()
