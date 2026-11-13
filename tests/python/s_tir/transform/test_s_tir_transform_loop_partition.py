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
# ruff: noqa: F401
import numpy
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.ir.module import IRModule
from tvm_tirx.script import tirx as T


def collect_visit(stmt, f):
    ret = []
    tvm_ffi.structural_walk(stmt, lambda x: ret.append(f(x)))
    return ret


def test_multi_loop():
    @Ts.function
    def func(n: T.int64, m: T.int64):
        for i in range(4):
            for j in T.serial(n):
                for k in T.serial(m):
                    if T.likely(i * m + j + k < n):
                        T.evaluate(m)
                    else:
                        T.evaluate(n)

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LoopPartition()(mod)
    stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert not any(collect_visit(stmt[0].body[0], lambda x: isinstance(x, tvm_tirx.ir.If)))


def test_multi_if():
    @Ts.function
    def func(n: T.int64, m: T.int64):
        for i in range(4):
            for j in T.serial(n):
                for k in T.serial(m):
                    if T.likely(i * m + j + k < n):
                        T.evaluate(m)
                    else:
                        T.evaluate(n)
                    if T.likely(i * m + j - k < n):
                        T.evaluate(m)
                    else:
                        T.evaluate(n)

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LoopPartition()(mod)
    stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert not any(collect_visit(stmt[0].body[0], lambda x: isinstance(x, tvm_tirx.ir.If)))


def test_condition():
    @Ts.function
    def func(m: T.int64, n: T.int64):
        for i in T.serial(T.truncdiv(n + 3, 4)):
            for j in range(4):
                T.evaluate(T.Select(T.likely(i * 4 + j < n), m, n))

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LoopPartition()(mod)
    stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert not any(collect_visit(stmt[0], lambda x: isinstance(x, tvm_tirx.tirx.Select)))


def test_condition_EQ():
    @Ts.function
    def func(m: T.int64, n: T.int64):
        for i in range(10):
            T.evaluate(T.Select(T.likely(i == 5), m, n))

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tvm_tirx.ir.transform.PassContext(
        config={"s_tir.LoopPartition": {"partition_const_loop": True}}
    ):
        mod = tvm.s_tir.transform.LoopPartition()(mod)
        stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert not any(collect_visit(stmt[0], lambda x: isinstance(x, tvm_tirx.tirx.Select)))


def test_everything_during_deduction():
    @Ts.function
    def func(m: T.int64, n: T.int64):
        for i in T.serial(n):
            for j in range(32):
                if T.likely(T.truncdiv(i, j) < m):
                    # this guard will produce everything during deduction
                    T.evaluate(m)

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LoopPartition()(mod)
    stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert isinstance(stmt[0].body[0].body[0], tvm_tirx.ir.If)


def test_oneD_pool():
    @Ts.function
    def func(m: T.int64, data: T.handle("float32"), out: T.handle("float32")):
        data_ptr = T.decl_tensor(data, ty_args=[T.Tensor((16,), "float32", data=data)])
        out_ptr = T.decl_tensor(out, ty_args=[T.Tensor((16,), "float32", data=out)])
        for ow in range(16):
            for kw in range(3):
                if T.likely(ow > 0):
                    if T.likely(ow < 15):
                        out_ptr[ow] = T.max(out_ptr[ow], data_ptr[ow + kw - 1])
        for ow in range(16):
            for kw in range(3):
                if T.likely(ow < 1):
                    if T.likely(kw > 0):
                        out_ptr[ow] = T.max(out_ptr[ow], data_ptr[ow + kw - 1])
        for ow in range(16):
            for kw in range(3):
                if T.likely(ow > 14):
                    if T.likely(kw < 2):
                        out_ptr[ow] = T.max(out_ptr[ow], data_ptr[ow + kw - 1])

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))

    with tvm_tirx.ir.transform.PassContext(
        config={"s_tir.LoopPartition": {"partition_const_loop": True}}
    ):
        mod = tvm.s_tir.transform.LoopPartition()(mod)
        stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm_tirx.ir.If)))


def test_cce_loop_1():
    n = 514
    m = 514

    @Ts.function
    def func(A: T.Tensor((n * m,), "float16"), B: T.Tensor((n * m,), "float16")):
        for i in range(11):
            for j in range(160):
                if T.likely(i * 160 + j < 1600):
                    A[(i + 1) * m + j + 1] = (
                        B[i * m + j + 1] + B[(i + 1) * m + j + 1] + B[(i + 2) * m + j + 1]
                    )

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tvm_tirx.ir.transform.PassContext(
        config={"s_tir.LoopPartition": {"partition_const_loop": True}}
    ):
        mod = tvm.s_tir.transform.LoopPartition()(mod)
        stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm_tirx.ir.If)))


def test_cce_loop_2():
    length = 112
    tile = 32
    loop = (length + tile - 1) // tile

    @Ts.function(private=True)
    def func():
        for i in range(loop):
            if T.likely(i * tile + tile > length):
                T.evaluate(T.call_extern("cce_intrisic", i * tile, length, ty="float32"))
            else:
                T.evaluate(T.call_extern("cce_intrisic", i * tile, i * tile + tile, ty="float32"))

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with tvm_tirx.ir.transform.PassContext(
        config={"s_tir.LoopPartition": {"partition_const_loop": True}}
    ):
        mod = tvm.s_tir.transform.LoopPartition()(mod)
        stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm_tirx.ir.If)))


def test_cce_loop_3():
    loop1 = 4
    loop2 = 9998
    tile = 39991

    @Ts.function(private=True)
    def func():
        for i in range(loop2):
            for j in range(loop1):
                if T.likely(i * loop1 + j < tile):
                    T.evaluate(T.call_extern("cce_intrisic", i, ty="float16"))

    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))

    with tvm_tirx.ir.transform.PassContext(
        config={"s_tir.LoopPartition": {"partition_const_loop": True}}
    ):
        mod = tvm.s_tir.transform.LoopPartition()(mod)
        stmt = tvm_tirx.tirx.transform.StmtSimplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm_tirx.ir.If)))


@Ts.function
def partitioned_concat(
    A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32"), C: T.Tensor((32,), "float32")
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    for i in T.serial(0, 16):
        C[i] = A[i]
    for i in T.serial(0, 16):
        C[i + 16] = B[i + 16]


def partition_from_scheduled_tir(function, pass_cfg, do_flatten=True):
    with tvm_tirx.ir.transform.PassContext(config=pass_cfg):
        mod = IRModule.from_expr(function.with_attr("global_symbol", "main"))
        mod = tvm.s_tir.transform.LoopPartition()(mod)
        mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
        mod = tvm.s_tir.transform.LowerThreadBinding()(mod)
        if do_flatten:
            mod = tvm_tirx.tirx.transform.FlattenTensor()(mod)
        mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
        mod = tvm_tirx.tirx.transform.RemoveNoOp()(mod)
        return mod


@Ts.function
def partitioned_concat_3(
    placeholder: T.Tensor((1, 64, 28, 28), "int8"),
    placeholder_1: T.Tensor((1, 32, 28, 28), "int8"),
    placeholder_2: T.Tensor((1, 32, 28, 28), "int8"),
    T_concat: T.Tensor((1, 128, 28, 28), "int8"),
) -> None:
    placeholder_flat = T.decl_tensor(
        placeholder.data, ty_args=[T.Tensor([50176], "int8", data=placeholder.data)]
    )
    placeholder_1_flat = T.decl_tensor(
        placeholder_1.data, ty_args=[T.Tensor([25088], "int8", data=placeholder_1.data)]
    )
    placeholder_2_flat = T.decl_tensor(
        placeholder_2.data, ty_args=[T.Tensor([25088], "int8", data=placeholder_2.data)]
    )
    T_concat_flat = T.decl_tensor(
        T_concat.data, ty_args=[T.Tensor([100352], "int8", data=T_concat.data)]
    )
    for i1, i2, i3 in T.grid(64, 28, 28):
        T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_flat[i1 * 784 + i2 * 28 + i3]
    for i1, i2, i3 in T.grid(32, 28, 28):
        T_concat_flat[i1 * 784 + i2 * 28 + i3 + 50176] = placeholder_1_flat[i1 * 784 + i2 * 28 + i3]
    for i1, i2, i3 in T.grid(32, 28, 28):
        T_concat_flat[i1 * 784 + i2 * 28 + i3 + 75264] = placeholder_2_flat[i1 * 784 + i2 * 28 + i3]


@Ts.function
def concat_func_3(
    placeholder: T.Tensor((1, 64, 28, 28), "int8"),
    placeholder_1: T.Tensor((1, 32, 28, 28), "int8"),
    placeholder_2: T.Tensor((1, 32, 28, 28), "int8"),
    T_concat: T.Tensor((1, 128, 28, 28), "int8"),
) -> None:
    placeholder_flat = T.decl_tensor(
        placeholder.data, ty_args=[T.Tensor([50176], "int8", data=placeholder.data)]
    )
    placeholder_1_flat = T.decl_tensor(
        placeholder_1.data, ty_args=[T.Tensor([25088], "int8", data=placeholder_1.data)]
    )
    placeholder_2_flat = T.decl_tensor(
        placeholder_2.data, ty_args=[T.Tensor([25088], "int8", data=placeholder_2.data)]
    )
    T_concat_flat = T.decl_tensor(
        T_concat.data, ty_args=[T.Tensor([100352], "int8", data=T_concat.data)]
    )
    for i1 in T.serial(128, annotations={"loop_partition_hint": 1}):
        for i2, i3 in T.grid(28, 28):
            if 96 <= i1:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_2_flat[
                    i1 * 784 + i2 * 28 + i3 - 75264
                ]
            if 64 <= i1 and i1 < 96:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_1_flat[
                    i1 * 784 + i2 * 28 + i3 - 50176
                ]
            if i1 < 64:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_flat[i1 * 784 + i2 * 28 + i3]


def test_condition_mutually_exclusive():
    mod = partition_from_scheduled_tir(
        concat_func_3, {"s_tir.LoopPartition": {"partition_const_loop": True}}
    )
    tvm_tirx.ir.assert_structural_equal(
        mod["main"], partitioned_concat_3.with_attr("global_symbol", "main")
    )


def test_loop_partition_unroll_hint():
    @Ts.function
    def main(
        A_arg: T.Tensor((1, 3, 224, 224), "int8"), B_arg: T.Tensor((1, 224, 7, 16), "int8")
    ) -> None:
        A = T.decl_tensor(A_arg.data, ty_args=[T.Tensor(150528, "int8", data=A_arg.data)])
        B = T.decl_tensor(B_arg.data, ty_args=[T.Tensor(25088, "int8", data=B_arg.data)])
        for ax0 in T.serial(
            112,
            annotations={"loop_partition_hint": True},
        ):
            for ax1, ax2, ax3 in T.grid(224, 7, 16):
                if 3 <= ax0 * 2 + ax2 and ax0 * 2 + ax2 < 227 and ax3 < 3:
                    B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax0 * 2 + ax2 - 3]

    @Ts.function
    def partitioned_main(
        A_arg: T.Tensor((1, 3, 224, 224), "int8"), B_arg: T.Tensor((1, 224, 7, 16), "int8")
    ) -> None:
        A = T.decl_tensor(A_arg.data, ty_args=[T.Tensor(150528, "int8", data=A_arg.data)])
        B = T.decl_tensor(B_arg.data, ty_args=[T.Tensor(25088, "int8", data=B_arg.data)])
        # body
        for ax1, ax2, ax3 in T.grid(224, 7, 16):
            if 3 <= ax2 and ax3 < 3:
                B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax2 - 3]
        for ax1, ax2, ax3 in T.grid(224, 7, 16):
            if 1 <= ax2 and ax3 < 3:
                B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax2 - 1]
        for ax0, ax1, ax2, ax3 in T.grid(109, 224, 7, 16):
            if ax3 < 3:
                B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax0 * 2 + ax2 + 1]
        for ax1, ax2, ax3 in T.grid(224, 7, 16):
            if ax2 < 5 and ax3 < 3:
                B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax2 + 219]

    mod = partition_from_scheduled_tir(
        main,
        {
            "s_tir.LoopPartition": {
                "partition_const_loop": True,
                "unroll_loop_with_partition_hint_no_interval": True,
            }
        },
    )
    mod = tvm_tirx.tirx.transform.UnrollLoop()(mod)
    mod = tvm_tirx.tirx.transform.RemoveNoOp()(mod)
    mod = tvm_tirx.tirx.transform.StmtSimplify()(mod)
    tvm_tirx.ir.assert_structural_equal(
        mod["main"], partitioned_main.with_attr("global_symbol", "main")
    )


def test_loop_partition_recursive_unroll_hint():
    @Ts.function
    def main():
        placeholder_0_dm = T.alloc_tensor(ty_args=[T.Tensor([1, 32, 32, 16], "int8")])
        for i3_0 in T.serial(5, annotations={"loop_partition_hint": 1}):
            for i2_0 in T.serial(2, annotations={"loop_partition_hint": 1}):
                pad_temp = T.alloc_tensor(ty_args=[T.Tensor([1, 16, 16, 16], "int8")])
                for ax0, ax1, ax2 in T.grid(16, 16, 16):
                    if (
                        6 <= i2_0 * 4 + ax0
                        and i2_0 * 4 + ax0 < 26
                        and 6 <= i3_0 * 4 + ax1
                        and i3_0 * 4 + ax1 < 26
                    ):
                        pad_temp[
                            0,
                            i2_0 * 4 + ax0 - 6 + 6 - i2_0 * 4,
                            i3_0 * 4 + ax1 - 6 + 6 - i3_0 * 4,
                            ax2,
                        ] = placeholder_0_dm[
                            0,
                            i2_0 * 4 + ax0 - 6 - -6,
                            i3_0 * 4 + ax1 - 6 - -6,
                            ax2,
                        ]

    @Ts.function
    def partitioned_main():
        placeholder_0_dm = T.alloc_tensor(ty_args=[T.Tensor((16384,), "int8")])
        for i3_0 in T.unroll(2):
            for i2_0 in T.unroll(2):
                pad_temp = T.alloc_tensor(ty_args=[T.Tensor((4096,), "int8")])
                for ax0, ax1, ax2 in T.grid(16, 16, 16):
                    if 6 <= i2_0 * 4 + ax0 and 6 <= i3_0 * 4 + ax1:
                        pad_temp[ax0 * 256 + ax1 * 16 + ax2] = placeholder_0_dm[
                            i2_0 * 2048 + ax0 * 512 + i3_0 * 64 + ax1 * 16 + ax2
                        ]
        for i2_0 in T.unroll(2):
            pad_temp = T.alloc_tensor(ty_args=[T.Tensor((4096,), "int8")])
            for ax0, ax1, ax2 in T.grid(16, 16, 16):
                if 6 <= i2_0 * 4 + ax0:
                    pad_temp[ax0 * 256 + ax1 * 16 + ax2] = placeholder_0_dm[
                        i2_0 * 2048 + ax0 * 512 + ax1 * 16 + ax2 + 128
                    ]
        for i3_0 in T.unroll(2):
            for i2_0 in T.unroll(2):
                pad_temp = T.alloc_tensor(ty_args=[T.Tensor((4096,), "int8")])
                for ax0, ax1, ax2 in T.grid(16, 16, 16):
                    if 6 <= i2_0 * 4 + ax0 and i3_0 * 4 + ax1 < 14:
                        pad_temp[ax0 * 256 + ax1 * 16 + ax2] = placeholder_0_dm[
                            i2_0 * 2048 + ax0 * 512 + i3_0 * 64 + ax1 * 16 + ax2 + 192
                        ]

    mod = partition_from_scheduled_tir(
        main,
        {
            "s_tir.LoopPartition": {
                "partition_const_loop": True,
                "unroll_loop_with_partition_hint_no_interval": True,
            }
        },
    )
    tvm_tirx.ir.assert_structural_equal(
        mod["main"], partitioned_main.with_attr("global_symbol", "main")
    )


def test_loop_partition_keep_loop_annotations():
    @Ts.function
    def before(A: T.Tensor(160, "int32"), B: T.Tensor(160, "int32")) -> None:
        for i in T.serial(
            160,
            annotations={"loop_partition_hint": True, "key": "value"},
        ):
            if i < 10:
                B[i] = A[i] + 1
            elif 10 <= i and i < 150:
                B[i] = A[i] + 2
            else:
                B[i] = A[i] + 3

    @Ts.function
    def after(A: T.Tensor(160, "int32"), B: T.Tensor(160, "int32")) -> None:
        A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((160,), "int32", data=A.data)])
        B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((160,), "int32", data=B.data)])
        for i in T.serial(10, annotations={"key": "value"}):
            B_1[i] = A_1[i] + 1
        for i in T.serial(140, annotations={"key": "value"}):
            B_1[i + 10] = A_1[i + 10] + 2
        for i in T.serial(10, annotations={"key": "value"}):
            B_1[i + 150] = A_1[i + 150] + 3

    mod = partition_from_scheduled_tir(
        before,
        {
            "s_tir.LoopPartition": {
                "partition_const_loop": True,
            }
        },
    )
    tvm_tirx.ir.assert_structural_equal(mod["main"], after.with_attr("global_symbol", "main"))


def test_loop_partition_with_unit_loop_in_condition():
    @Ts.function
    def before(
        placeholder: T.Tensor((50176,), "int8"),
        placeholder_1: T.Tensor((25088,), "int8"),
        placeholder_2: T.Tensor((25088,), "int8"),
        T_concat: T.Tensor((100352,), "int8"),
    ) -> None:
        for k in range(1, annotations={"preserve_unit_loop": True}):
            for i1 in T.serial(128, annotations={"loop_partition_hint": 1}):
                for i2, i3 in T.grid(28, 28):
                    if 96 <= k * 128 + i1:
                        T_concat[k * i1 * 784 + i2 * 28 + i3] = placeholder_2[
                            i1 * 784 + i2 * 28 + i3 - 75264
                        ]
                    if 64 <= k * 128 + i1 and k * 128 + i1 < 96:
                        T_concat[i1 * 784 + i2 * 28 + i3] = placeholder_1[
                            i1 * 784 + i2 * 28 + i3 - 50176
                        ]
                    if k * 128 + i1 < 64:
                        T_concat[i1 * 784 + i2 * 28 + i3] = placeholder[i1 * 784 + i2 * 28 + i3]

    @Ts.function
    def after(
        placeholder: T.Tensor(50176, "int8"),
        placeholder_1: T.Tensor(25088, "int8"),
        placeholder_2: T.Tensor(25088, "int8"),
        T_concat: T.Tensor(100352, "int8"),
    ) -> None:
        placeholder_3 = T.decl_tensor(
            placeholder.data, ty_args=[T.Tensor((50176,), "int8", data=placeholder.data)]
        )
        placeholder_1_1 = T.decl_tensor(
            placeholder_1.data, ty_args=[T.Tensor((25088,), "int8", data=placeholder_1.data)]
        )
        placeholder_2_1 = T.decl_tensor(
            placeholder_2.data, ty_args=[T.Tensor((25088,), "int8", data=placeholder_2.data)]
        )
        T_concat_1 = T.decl_tensor(
            T_concat.data, ty_args=[T.Tensor((100352,), "int8", data=T_concat.data)]
        )
        for k in T.serial(1, annotations={"preserve_unit_loop": True}):
            for i1, i2, i3 in T.grid(64, 28, 28):
                T_concat_1[i1 * 784 + i2 * 28 + i3] = placeholder_3[i1 * 784 + i2 * 28 + i3]
            for i1, i2, i3 in T.grid(32, 28, 28):
                T_concat_1[i1 * 784 + i2 * 28 + i3 + 50176] = placeholder_1_1[
                    i1 * 784 + i2 * 28 + i3
                ]
            for i1, i2, i3 in T.grid(32, 28, 28):
                T_concat_1[i2 * 28 + i3] = placeholder_2_1[i1 * 784 + i2 * 28 + i3]

    mod = partition_from_scheduled_tir(
        before,
        {
            "s_tir.LoopPartition": {
                "partition_const_loop": True,
            }
        },
    )
    tvm_tirx.ir.assert_structural_equal(mod["main"], after.with_attr("global_symbol", "main"))


@Ts.function
def concat_func_single_point(
    placeholder: T.Tensor((28, 64), "int8"),
    placeholder_1: T.Tensor((28, 1), "int8"),
    placeholder_2: T.Tensor((28, 63), "int8"),
    T_concat: T.Tensor((28, 128), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in T.serial(128, annotations={"loop_partition_hint": 1}):
            if i1 > 63:
                T_concat[i0, i1] = placeholder[i0, i1 - 64]
            elif i1 == 63:
                T_concat[i0, i1] = placeholder_1[i0, i1 - 63]
            else:
                T_concat[i0, i1] = placeholder_2[i0, i1]


@Ts.function
def expected_partitioned_concat_single_point(
    placeholder: T.Tensor((28, 64), "int8"),
    placeholder_1: T.Tensor((28, 1), "int8"),
    placeholder_2: T.Tensor((28, 63), "int8"),
    T_concat: T.Tensor((28, 128), "int8"),
):
    placeholder_3 = T.decl_tensor(
        placeholder.data, ty_args=[T.Tensor((1792,), "int8", data=placeholder.data)]
    )
    placeholder_1_1 = T.decl_tensor(
        placeholder_1.data, ty_args=[T.Tensor((28,), "int8", data=placeholder_1.data)]
    )
    placeholder_2_1 = T.decl_tensor(
        placeholder_2.data, ty_args=[T.Tensor((1764,), "int8", data=placeholder_2.data)]
    )
    T_concat_1 = T.decl_tensor(
        T_concat.data, ty_args=[T.Tensor((3584,), "int8", data=T_concat.data)]
    )
    for i0 in range(28):
        for i1 in range(63):
            T_concat_1[i0 * 128 + i1] = placeholder_2_1[i0 * 63 + i1]
        T_concat_1[i0 * 128 + 63] = placeholder_1_1[i0]
        for i1 in range(64):
            T_concat_1[i0 * 128 + i1 + 64] = placeholder_3[i0 * 64 + i1]


@Ts.function
def concat_func_start_point_equality(
    placeholder: T.Tensor((28, 64), "int8"),
    placeholder_1: T.Tensor((28, 1), "int8"),
    placeholder_2: T.Tensor((28, 63), "int8"),
    T_concat: T.Tensor((28, 128), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in range(128, annotations={"loop_partition_hint": 1}):
            if i1 == 0:
                # Special case for i1 == 0
                T_concat[i0, i1] = placeholder_1[i0, 0]
            elif i1 < 64:
                # Normal case for i1 in [1, 63]
                T_concat[i0, i1] = placeholder_2[i0, i1]
            else:
                # Case for i1 in [64, 127]
                T_concat[i0, i1] = placeholder[i0, i1 - 64]


@Ts.function
def concat_func_start_point_equality_expected(
    placeholder: T.Tensor((28, 64), "int8"),
    placeholder_1: T.Tensor((28, 1), "int8"),
    placeholder_2: T.Tensor((28, 63), "int8"),
    T_concat: T.Tensor((28, 128), "int8"),
):
    placeholder_3 = T.decl_tensor(
        placeholder.data, ty_args=[T.Tensor((1792,), "int8", data=placeholder.data)]
    )
    placeholder_1_1 = T.decl_tensor(
        placeholder_1.data, ty_args=[T.Tensor((28,), "int8", data=placeholder_1.data)]
    )
    placeholder_2_1 = T.decl_tensor(
        placeholder_2.data, ty_args=[T.Tensor((1764,), "int8", data=placeholder_2.data)]
    )
    T_concat_1 = T.decl_tensor(
        T_concat.data, ty_args=[T.Tensor((3584,), "int8", data=T_concat.data)]
    )
    for i0 in range(28):
        T_concat_1[i0 * 128] = placeholder_1_1[i0]
        for i1 in range(63):
            T_concat_1[i0 * 128 + i1 + 1] = placeholder_2_1[i0 * 63 + i1 + 1]
        for i1 in range(64):
            T_concat_1[i0 * 128 + i1 + 64] = placeholder_3[i0 * 64 + i1]


@Ts.function
def concat_func_end_point_equality(
    placeholder: T.Tensor((28, 64), "int8"),
    placeholder_1: T.Tensor((28, 1), "int8"),
    placeholder_2: T.Tensor((28, 63), "int8"),
    T_concat: T.Tensor((28, 128), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in range(128, annotations={"loop_partition_hint": 1}):
            if i1 == 127:
                # Explicit equality check for the end point i1 == 127
                T_concat[i0, i1] = placeholder_1[i0, 0]
            elif i1 >= 64:
                # Case for i1 in [64, 126]
                T_concat[i0, i1] = placeholder[i0, i1 - 64]
            else:
                # Case for i1 in [0, 63]
                T_concat[i0, i1] = placeholder_2[i0, i1]


@Ts.function
def concat_func_end_point_equality_expected(
    placeholder: T.Tensor((28, 64), "int8"),
    placeholder_1: T.Tensor((28, 1), "int8"),
    placeholder_2: T.Tensor((28, 63), "int8"),
    T_concat: T.Tensor((28, 128), "int8"),
):
    placeholder_3 = T.decl_tensor(
        placeholder.data, ty_args=[T.Tensor((1792,), "int8", data=placeholder.data)]
    )
    placeholder_1_1 = T.decl_tensor(
        placeholder_1.data, ty_args=[T.Tensor((28,), "int8", data=placeholder_1.data)]
    )
    placeholder_2_1 = T.decl_tensor(
        placeholder_2.data, ty_args=[T.Tensor((1764,), "int8", data=placeholder_2.data)]
    )
    T_concat_1 = T.decl_tensor(
        T_concat.data, ty_args=[T.Tensor((3584,), "int8", data=T_concat.data)]
    )
    for i0 in range(28):
        for i1 in range(64):
            T_concat_1[i0 * 128 + i1] = placeholder_2_1[i0 * 63 + i1]
        for i1 in range(63):
            T_concat_1[i0 * 128 + i1 + 64] = placeholder_3[i0 * 64 + i1]
        T_concat_1[i0 * 128 + 127] = placeholder_1_1[i0]


@Ts.function
def concat_func_edge_equalities(
    placeholder: T.Tensor((28, 64), "int8"),
    placeholder_1: T.Tensor((28, 1), "int8"),
    placeholder_2: T.Tensor((28, 1), "int8"),
    T_concat: T.Tensor((28, 66), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in range(66, annotations={"loop_partition_hint": 1}):  # Loop from 0 to 65 inclusive
            if i1 == 0:
                # Handle equality at the start of the range: i1 == 0
                T_concat[i0, i1] = placeholder_2[i0, 0]
            elif i1 == 65:
                # Handle equality at the end of the range: i1 == 65
                T_concat[i0, i1] = placeholder_1[i0, 0]
            else:
                # Copying from placeholder (from 0 to 63)
                T_concat[i0, i1] = placeholder[i0, i1 - 1]


@Ts.function
def concat_func_edge_equalities_expected(
    placeholder: T.Tensor((28, 64), "int8"),
    placeholder_1: T.Tensor((28, 1), "int8"),
    placeholder_2: T.Tensor((28, 1), "int8"),
    T_concat: T.Tensor((28, 66), "int8"),
):
    placeholder_3 = T.decl_tensor(
        placeholder.data, ty_args=[T.Tensor((1792,), "int8", data=placeholder.data)]
    )
    placeholder_1_1 = T.decl_tensor(
        placeholder_1.data, ty_args=[T.Tensor((28,), "int8", data=placeholder_1.data)]
    )
    placeholder_2_1 = T.decl_tensor(
        placeholder_2.data, ty_args=[T.Tensor((28,), "int8", data=placeholder_2.data)]
    )
    T_concat_1 = T.decl_tensor(
        T_concat.data, ty_args=[T.Tensor((1848,), "int8", data=T_concat.data)]
    )
    for i0 in range(28):
        T_concat_1[i0 * 66] = placeholder_2_1[i0]
        for i1 in range(64):
            T_concat_1[i0 * 66 + i1 + 1] = placeholder_3[i0 * 64 + i1]
        T_concat_1[i0 * 66 + 65] = placeholder_1_1[i0]


@Ts.function
def concat_five_tensors_with_equalities(
    tensor_a: T.Tensor((28, 1), "int8"),  # Used for i1 == 0
    tensor_b: T.Tensor((28, 63), "int8"),  # Fills i1 from 1 to 63
    tensor_c: T.Tensor((28, 1), "int8"),  # Used for i1 == 64
    tensor_d: T.Tensor((28, 63), "int8"),  # Fills i1 from 65 to 128
    tensor_e: T.Tensor((28, 1), "int8"),  # Used for i1 == 129
    T_concat: T.Tensor((28, 129), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in range(130, annotations={"loop_partition_hint": 1}):
            if i1 == 0:
                T_concat[i0, i1] = tensor_a[i0, 0]
            elif i1 == 64:
                T_concat[i0, i1] = tensor_c[i0, 0]
            elif i1 == 129:
                T_concat[i0, i1] = tensor_e[i0, 0]
            elif i1 < 64:
                T_concat[i0, i1] = tensor_b[i0, i1 - 1]
            else:  # i1 > 64 and i1 < 128
                T_concat[i0, i1] = tensor_d[i0, i1 - 65]


@Ts.function
def concat_five_tensors_with_equalities_expected(
    tensor_a: T.Tensor((28, 1), "int8"),  # Used for i1 == 0
    tensor_b: T.Tensor((28, 63), "int8"),  # Fills i1 from 1 to 63
    tensor_c: T.Tensor((28, 1), "int8"),  # Used for i1 == 64
    tensor_d: T.Tensor((28, 63), "int8"),  # Fills i1 from 65 to 128
    tensor_e: T.Tensor((28, 1), "int8"),  # Used for i1 == 129
    T_concat: T.Tensor((28, 129), "int8"),
):
    tensor_a_1 = T.decl_tensor(tensor_a.data, ty_args=[T.Tensor((28,), "int8", data=tensor_a.data)])
    tensor_b_1 = T.decl_tensor(
        tensor_b.data, ty_args=[T.Tensor((1764,), "int8", data=tensor_b.data)]
    )
    tensor_c_1 = T.decl_tensor(tensor_c.data, ty_args=[T.Tensor((28,), "int8", data=tensor_c.data)])
    tensor_d_1 = T.decl_tensor(
        tensor_d.data, ty_args=[T.Tensor((1764,), "int8", data=tensor_d.data)]
    )
    tensor_e_1 = T.decl_tensor(tensor_e.data, ty_args=[T.Tensor((28,), "int8", data=tensor_e.data)])
    T_concat_1 = T.decl_tensor(
        T_concat.data, ty_args=[T.Tensor((3612,), "int8", data=T_concat.data)]
    )
    for i0 in range(28):
        T_concat_1[i0 * 129] = tensor_a_1[i0]
        for i1 in range(63):
            T_concat_1[i0 * 129 + i1 + 1] = tensor_b_1[i0 * 63 + i1]
        T_concat_1[i0 * 129 + 64] = tensor_c_1[i0]
        for i1 in range(64):
            T_concat_1[i0 * 129 + i1 + 65] = tensor_d_1[i0 * 63 + i1]
        T_concat_1[i0 * 129 + 129] = tensor_e_1[i0]


@Ts.function
def nested_partition_with_single_points(A: T.Tensor((25,), "int32")):
    for i in T.serial(5, annotations={"loop_partition_hint": 1}):
        if i == 1:
            for j in T.serial(5, annotations={"loop_partition_hint": 1}):
                if j > 2:
                    A[i * 5 + j] = i * 5 + j
        else:
            for j in T.serial(5, annotations={"loop_partition_hint": 1}):
                if j > 2:
                    A[i * 5 + j] = i * 15 + j


@Ts.function
def nested_partition_with_single_points_expected(A: T.Tensor((25,), "int32")):
    A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((25,), "int32", data=A.data)])
    for j in range(2):
        A_1[j + 3] = j + 3
    for j in range(2):
        A_1[j + 8] = j + 8
    for i, j in T.grid(3, 2):
        A_1[i * 5 + j + 13] = i * 15 + j + 33


@pytest.mark.parametrize(
    "origin,expected",
    [
        (concat_func_single_point, expected_partitioned_concat_single_point),
        (concat_func_start_point_equality, concat_func_start_point_equality_expected),
        (concat_func_end_point_equality, concat_func_end_point_equality_expected),
        (concat_func_edge_equalities, concat_func_edge_equalities_expected),
        (concat_five_tensors_with_equalities, concat_five_tensors_with_equalities_expected),
        (nested_partition_with_single_points, nested_partition_with_single_points_expected),
    ],
)
def test_single_point_partition(origin, expected):
    origin = origin.with_attr({"global_symbol": "main"})
    expected = expected.with_attr({"global_symbol": "main"})
    mod = partition_from_scheduled_tir(
        origin,
        {
            "s_tir.LoopPartition": {
                "partition_const_loop": True,
                "unroll_loop_with_partition_hint_no_interval": True,
            }
        },
    )
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_equation_on_floordiv():
    @Ts.function
    def before(A: T.Tensor((2, 2, 20), "int32")):
        for i in T.serial(5, annotations={"loop_partition_hint": 1}):
            if i == 1:
                for vv in T.vectorized(640, annotations={"loop_partition_hint": 1}):
                    if i * 2 + vv // 320 == 3:
                        A[i - 1, i * 2 + vv // 320 - 3, vv % 320 // 16] = 1

    @Ts.function
    def expected(A: T.Tensor((2, 2, 20), "int32")):
        for vv in T.vectorized(320):
            A[0, 0, vv // 16] = 1

    expected = expected.with_attr({"global_symbol": "main"})
    after = partition_from_scheduled_tir(
        before.with_attr("global_symbol", "main"), {}, do_flatten=False
    )
    tvm_tirx.ir.assert_structural_equal(after["main"], expected)


def test_ignore_loop_partition_hint():
    """Skip unroll body and prologue for pipeline case"""

    @Ts.function
    def before(A: T.Tensor((10), "float32"), D: T.Tensor((10), "float32")):
        B = T.alloc_tensor(ty_args=[T.Tensor([2], "float32")])
        C = T.alloc_tensor(ty_args=[T.Tensor([2], "float32")])
        for i in T.serial(12, annotations={"loop_partition_hint": 1}):
            if T.ignore_loop_partition(i < 10):
                B[i % 2] = A[i] + 1.0
            if T.ignore_loop_partition(1 <= i and i < 11):
                C[(i - 1) % 2] = B[(i - 1) % 2] + 2.0
            if 2 <= i:
                D[i - 2] = C[i % 2] + 3.0

    @Ts.function
    def expected(A: T.Tensor((10), "float32"), D: T.Tensor((10), "float32")):
        B = T.alloc_tensor(ty_args=[T.Tensor([2], "float32")])
        C = T.alloc_tensor(ty_args=[T.Tensor([2], "float32")])
        for i in range(2):
            B[i] = A[i] + 1.0
            if i == 1:
                C[i - 1] = B[i - 1] + 2.0
        for i in T.serial(10):
            if i < 8:
                B[i % 2] = A[i + 2] + 1.0
            if i < 9:
                C[(i + 1) % 2] = B[(i + 1) % 2] + 2.0
            D[i] = C[i % 2] + 3.0

    expected = expected.with_attr({"global_symbol": "main"})
    after = partition_from_scheduled_tir(
        before.with_attr({"global_symbol": "main"}), {}, do_flatten=False
    )
    tvm_tirx.ir.assert_structural_equal(after["main"], expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
