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
import numpy as np

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm.s_tir.tensor_intrin.hexagon import VRMPY_u8u8i32_INTRIN
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T


def _check(original, transformed):
    func = original
    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.PlanAndUpdateTensorAllocationLocation()(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"))


@Ts.function
def element_func(A: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
    B = Ts.sblock_alloc_tensor((16, 16))
    for i0 in range(0, 16):
        for j0 in range(0, 16):
            with Ts.sblock():
                i, j = Ts.axis.remap("SS", [i0, j0])
                B[i, j] = A[i, j] + 1.0
        for j0 in range(0, 16):
            with Ts.sblock():
                i, j = Ts.axis.remap("SS", [i0, j0])
                C[i, j] = B[i, j] * 2.0


@Ts.function
def transformed_element_func(
    A: T.Tensor([16, 16], "float32"), C: T.Tensor([16, 16], "float32")
) -> None:
    for i_0 in range(0, 16):
        with Ts.sblock():
            Ts.reads([A[i_0, 0:16]])
            Ts.writes([C[i_0, 0:16]])
            B = Ts.sblock_alloc_tensor([16, 16])
            for j_0 in T.serial(0, 16):
                with Ts.sblock():
                    i, j = Ts.axis.remap("SS", [i_0, j_0])
                    B[i, j] = A[i, j] + 1.0
            for j_0 in T.serial(0, 16):
                with Ts.sblock():
                    i, j = Ts.axis.remap("SS", [i_0, j_0])
                    C[i, j] = B[i, j] * 2.0


@Ts.function
def original_func() -> None:
    A = Ts.sblock_alloc_tensor((128, 128), "float32")
    for i0, j0 in T.grid(128, 128):
        with Ts.sblock():
            i, j = Ts.axis.remap("SS", [i0, j0])
            A[i, j] = T.float32(0)
    for i0, j0, k0 in T.grid(32, 32, 32):
        with Ts.sblock():
            i, j, k = Ts.axis.remap("SSR", [i0, j0, k0])
            B = Ts.sblock_alloc_tensor((128, 128), "float32")
            C = Ts.sblock_alloc_tensor((128, 128), "float32")
            D = Ts.sblock_alloc_tensor((128, 128), "float32")
            if k == 0:
                for ii, jj in T.grid(4, 4):
                    B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
            for ii, jj in T.grid(4, 4):
                for kk in range(0, 4):
                    B[i * 4 + ii, j * 4 + jj] += C[i * 4 + ii, k * 4 + kk]
                for kk in range(0, 4):
                    B[i * 4 + ii, j * 4 + jj] += (
                        D[j * 4 + jj, k * 4 + kk] * C[i * 4 + ii, k * 4 + kk]
                    )


@Ts.function
def transformed_func() -> None:
    A = Ts.sblock_alloc_tensor([128, 128])
    for i0, j0 in T.grid(128, 128):
        with Ts.sblock():
            i, j = Ts.axis.remap("SS", [i0, j0])
            A[i, j] = T.float32(0)
    for i0, j0, k0 in T.grid(32, 32, 32):
        with Ts.sblock():
            i, j, k = Ts.axis.remap("SSR", [i0, j0, k0])
            B = Ts.sblock_alloc_tensor([128, 128])
            if k == 0:
                for ii, jj in T.grid(4, 4):
                    B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
            for ii, jj in T.grid(4, 4):
                with Ts.sblock(""):
                    Ts.reads([B[((i * 4) + ii), ((j * 4) + jj)]])
                    Ts.writes([B[((i * 4) + ii), ((j * 4) + jj)]])
                    C = Ts.sblock_alloc_tensor([128, 128])
                    for kk in T.serial(0, 4):
                        B[((i * 4) + ii), ((j * 4) + jj)] = (
                            B[((i * 4) + ii), ((j * 4) + jj)] + C[((i * 4) + ii), ((k * 4) + kk)]
                        )
                    for kk in T.serial(0, 4):
                        with Ts.sblock(""):
                            Ts.reads(
                                [
                                    B[((i * 4) + ii), ((j * 4) + jj)],
                                    C[((i * 4) + ii), ((k * 4) + kk)],
                                ]
                            )
                            Ts.writes([B[((i * 4) + ii), ((j * 4) + jj)]])
                            D = Ts.sblock_alloc_tensor([128, 128])
                            B[((i * 4) + ii), ((j * 4) + jj)] = B[
                                ((i * 4) + ii), ((j * 4) + jj)
                            ] + (
                                D[((j * 4) + jj), ((k * 4) + kk)]
                                * C[((i * 4) + ii), ((k * 4) + kk)]
                            )


@Ts.function
def match_tensor_func() -> None:
    C = Ts.sblock_alloc_tensor((128, 128))
    for i in range(128):
        with Ts.sblock():
            vi = Ts.axis.S(128, i)
            C0 = Ts.match_tensor(C[vi, 0:128], (128))
            for j in range(128):
                with Ts.sblock():
                    jj = Ts.axis.S(128, j)
                    C1 = Ts.match_tensor(C0[jj], ())
                    C1[()] = 0


@Ts.function
def transformed_match_tensor_func() -> None:
    for i in range(0, 128):
        with Ts.sblock():
            vi = Ts.axis.S(128, i)
            C = Ts.sblock_alloc_tensor((128, 128))
            C0 = Ts.match_tensor(C[vi, 0:128], (128))
            for j in range(128):
                with Ts.sblock():
                    jj = Ts.axis.S(128, j)
                    C1 = Ts.match_tensor(C0[jj], ())
                    C1[()] = 0


@Ts.function
def opaque_access(A: T.Tensor([1024], "float32"), B: T.Tensor([1024], "float32")) -> None:
    A_cache = Ts.sblock_alloc_tensor([1024])
    for i in T.serial(0, 8):
        with Ts.sblock():
            vi = Ts.axis.S(8, i)
            with Ts.sblock():
                v = Ts.axis.S(8, vi)
                Ts.reads([A[(v * 128) : ((v * 128) + 128)]])
                Ts.writes([A_cache[(v * 128) : ((v * 128) + 128)]])
                T.evaluate(
                    T.call_extern(
                        "test",
                        A_cache.data,
                        (v * 128),
                        128,
                        A.data,
                        (v * 128),
                        128,
                        ty="float32",
                    )
                )
            for j in T.serial(0, 128):
                with Ts.sblock():
                    v = Ts.axis.S(1024, vi * 128 + j)
                    Ts.reads([A_cache[v]])
                    Ts.writes([B[v]])
                    B[v] = A_cache[v]


@Ts.function
def transformed_opaque_access(
    A: T.Tensor([1024], "float32"), B: T.Tensor([1024], "float32")
) -> None:
    for i in T.serial(0, 8):
        with Ts.sblock():
            vi = Ts.axis.S(8, i)
            Ts.reads(A[vi * 128 : vi * 128 + 128])
            Ts.writes(B[vi * 128 : vi * 128 + 128])
            A_cache = Ts.sblock_alloc_tensor([1024])
            with Ts.sblock():
                v = Ts.axis.S(8, vi)
                Ts.reads([A[v * 128 : v * 128 + 128]])
                Ts.writes([A_cache[v * 128 : v * 128 + 128]])
                T.evaluate(
                    T.call_extern(
                        "test", A_cache.data, v * 128, 128, A.data, v * 128, 128, ty="float32"
                    )
                )
            for j in T.serial(0, 128):
                with Ts.sblock():
                    v = Ts.axis.S(1024, vi * 128 + j)
                    Ts.reads([A_cache[v]])
                    Ts.writes([B[v]])
                    B[v] = A_cache[v]


def test_elementwise():
    _check(element_func, transformed_element_func)


def test_locate_tensor_allocation():
    _check(original_func, transformed_func)


def test_match_tensor_allocation():
    _check(match_tensor_func, transformed_match_tensor_func)


def test_opaque_access():
    _check(opaque_access, transformed_opaque_access)


def test_loop_carried_dependency():
    """The tensor allocation should be above opaque iter var's loop scopes
    such that tensor accesses with loop carried dependencies are covered,
    and the allocate tensor should keep the order."""

    @Ts.function
    def before(A: T.Tensor((8, 8, 8), "int32"), B: T.Tensor((8, 8, 8), "int32")):
        C = Ts.sblock_alloc_tensor([8, 8, 8], dtype="int32")
        D = Ts.sblock_alloc_tensor([8, 8, 8], dtype="int32")
        for i in T.serial(8):
            for j in T.serial(8):
                for k in T.serial(8):
                    with Ts.sblock("b0"):
                        vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                        C[vi, vj, vk] = A[vi, vj, vk] + 1
                for k in T.serial(8):
                    with Ts.sblock("b1"):
                        vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                        D[vi, vj, vk] = A[vi, vj, vk] + 2
                for k in T.serial(8):
                    with Ts.sblock("b2"):
                        vi, vk = Ts.axis.remap("SS", [i, k])
                        vj = Ts.axis.opaque(8, j)
                        B[vi, vj, vk] = (
                            C[vi, vj, vk]
                            + T.if_then_else(0 < vj, C[vi, vj - 1, vk], 0)
                            + D[vi, vj, vk]
                        )

    @Ts.function
    def after(A: T.Tensor((8, 8, 8), "int32"), B: T.Tensor((8, 8, 8), "int32")) -> None:
        for i in T.serial(8):
            with Ts.sblock():
                Ts.reads(A[i, 0:8, 0:8])
                Ts.writes(B[i, 0:8, 0:8])
                C = Ts.sblock_alloc_tensor([8, 8, 8], dtype="int32")
                D = Ts.sblock_alloc_tensor([8, 8, 8], dtype="int32")
                for j in T.serial(8):
                    for k in T.serial(8):
                        with Ts.sblock("b0"):
                            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                            C[vi, vj, vk] = A[vi, vj, vk] + 1
                    for k in T.serial(8):
                        with Ts.sblock("b1"):
                            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                            D[vi, vj, vk] = A[vi, vj, vk] + 2
                    for k in T.serial(8):
                        with Ts.sblock("b2"):
                            vi, vk = Ts.axis.remap("SS", [i, k])
                            vj = Ts.axis.opaque(8, j)
                            B[vi, vj, vk] = (
                                C[vi, vj, vk]
                                + T.if_then_else(0 < vj, C[vi, vj - 1, vk], 0)
                                + D[vi, vj, vk]
                            )

    _check(before, after)


def test_1D_cascade_op_rolling_buffer():
    """The intermediate tensor must be allocated above rolling tensor's rolling loop,
    which is marked as opaque in consumer block's iter mappings."""

    @Ts.function
    def before(A: T.Tensor((4, 16), "int32"), C: T.Tensor((4, 8), "int32")):
        B = Ts.sblock_alloc_tensor((4, 6), "int32")
        for c in T.serial(4):
            for i in T.serial(0, 2):
                for j in T.serial(0, 6):
                    for k in T.serial(3):
                        with Ts.sblock("P1"):
                            Ts.where(i < 1 or j >= 2)
                            cc, vi, vj, vk = Ts.axis.remap("SSSR", [c, i, j, k])
                            if vk == 0:
                                B[cc, T.floormod(vi * 4 + vj, 6)] = 0
                            B[cc, T.floormod(vi * 4 + vj, 6)] = (
                                B[cc, T.floormod(vi * 4 + vj, 6)] + A[cc, vi * 4 + vj + vk]
                            )
                for j in T.serial(0, 4):
                    for k in T.serial(3):
                        with Ts.sblock("P2"):
                            vi = Ts.axis.opaque(2, i)
                            cc, vj, vk = Ts.axis.remap("SSR", [c, j, k])
                            if vk == 0:
                                C[cc, vi * 4 + vj] = 0
                            C[cc, vi * 4 + vj] = (
                                C[cc, vi * 4 + vj] + B[cc, T.floormod(vi * 4 + vj + vk, 6)]
                            )

    @Ts.function
    def after(A: T.Tensor((4, 16), "int32"), C: T.Tensor((4, 8), "int32")):
        for c in T.serial(4):
            with Ts.sblock():
                Ts.reads(A[c, 0:12], C[c, 0:8])
                Ts.writes(C[c, 0:8])
                B = Ts.sblock_alloc_tensor([4, 6], dtype="int32")
                for i in T.serial(2):
                    for j, k in T.grid(6, 3):
                        with Ts.sblock("P1"):
                            Ts.where(i < 1 or j >= 2)
                            cc, vi, vj, vk = Ts.axis.remap("SSSR", [c, i, j, k])
                            if vk == 0:
                                B[cc, (vi * 4 + vj) % 6] = 0
                            B[cc, (vi * 4 + vj) % 6] = (
                                B[cc, (vi * 4 + vj) % 6] + A[cc, vi * 4 + vj + vk]
                            )
                    for j, k in T.grid(4, 3):
                        with Ts.sblock("P2"):
                            vi = Ts.axis.opaque(2, i)
                            cc, vj, vk = Ts.axis.remap("SSR", [c, j, k])
                            if vk == 0:
                                C[cc, vi * 4 + vj] = 0
                            C[cc, vi * 4 + vj] = C[cc, vi * 4 + vj] + B[cc, (vi * 4 + vj + vk) % 6]

    _check(before, after)


def test_tensor_conditional_lowering():
    """Tensors passed as pointer arguments are unmodified

    Confirm that the `tirx.PlanAndUpdateTensorAllocationLocation` pass
    leaves (Tensor nodes corresponding to pointer-typed Function arguments)
    unchanged, rather than lowering them to `reads`, `writes`, and `alloc_tensor` nodes.
    """

    @Ts.function
    def before(A: T.handle("float32")):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        for i in range(1):
            A_1 = T.decl_tensor(A, ty_args=[T.Tensor((1,), "float32", data=A)])
            A_1[i] = 0

    after = before
    _check(before, after)


def test_dltensor_tensor_is_unlowered():
    """Tensors allocated with a Bind are unmodified

    Confirm that the `tirx.PlanAndUpdateTensorAllocationLocation` pass
    leaves (Tensor nodes corresponding to Function DLTensor arguments)
    unchanged, rather than lowering them to `reads`, `writes`, and
    `alloc_tensor` nodes.
    """

    @Ts.function
    def before(dlpack_handle: T.handle, axis: T.int64) -> T.int64:
        ndim: T.int32 = T.abi_field_get(dlpack_handle, 0, 5, ty="int32")
        stride_ptr: T.let[T.handle("int64")] = T.abi_field_get(
            dlpack_handle, 0, 4, ty=T.handle("int64").ty
        )
        if T.isnullptr(stride_ptr):
            shape_ptr: T.let[T.handle("int64")] = T.abi_field_get(
                dlpack_handle, 0, 3, ty=T.handle("int64").ty
            )
            shape = T.decl_tensor(shape_ptr, ty_args=[T.Tensor(ndim, "int64", data=shape_ptr)])
            product = T.alloc_tensor(ty_args=[T.Tensor([], "int64")])
            product[()] = 1
            for dim in range(axis + 1, ndim):
                product[()] = product[()] * shape[dim]
            return product[()]
        else:
            strides = T.decl_tensor(stride_ptr, ty_args=[T.Tensor(ndim, "int64", data=stride_ptr)])
            stride: T.int64 = strides[axis]
            return stride

    after = before
    _check(before, after)


def test_reduce_tensor_dominate_reduce_loops():
    """Reduction write tensor allocation should dominate all reduce loops"""

    @Ts.function
    def before(x: T.Tensor((256, 256, 256), "float32"), x_red: T.Tensor((256, 256), "float32")):
        x_red_ = Ts.sblock_alloc_tensor((256, 256))
        for ax0_0, k1_0, ax1_0 in T.grid(4, 4, 4):
            for ax0_1, k1_1, ax1_1 in T.grid(64, 64, 64):
                with Ts.sblock("x_red"):
                    v_ax0 = Ts.axis.spatial(256, ax0_0 * 64 + ax0_1)
                    v_ax1 = Ts.axis.spatial(256, ax1_0 * 64 + ax1_1)
                    v_k1 = Ts.axis.reduce(256, k1_0 * 64 + k1_1)
                    if v_k1 == 0:
                        x_red_[v_ax0, v_ax1] = T.float32(0.0)
                    x_red_[v_ax0, v_ax1] = x_red_[v_ax0, v_ax1] + x[v_ax0, v_k1, v_ax1]
            for ax0, ax1 in T.grid(64, 64):
                with Ts.sblock("x_red_"):
                    v0 = Ts.axis.spatial(256, ax0_0 * 64 + ax0)
                    v1 = Ts.axis.spatial(256, ax1_0 * 64 + ax1)
                    x_red[v0, v1] = x_red_[v0, v1]

    @Ts.function
    def after(x: T.Tensor((256, 256, 256), "float32"), x_red: T.Tensor((256, 256), "float32")):
        for ax0_0 in range(4):
            with Ts.sblock(""):
                Ts.reads(x[ax0_0 * 64 : ax0_0 * 64 + 64, 0:256, 0:256])
                Ts.writes(x_red[ax0_0 * 64 : ax0_0 * 64 + 64, 0:256])
                x_red_ = Ts.sblock_alloc_tensor((256, 256))
                for k1_0, ax1_0 in T.grid(4, 4):
                    for ax0_1, k1_1, ax1_1 in T.grid(64, 64, 64):
                        with Ts.sblock("x_red"):
                            v_ax0 = Ts.axis.spatial(256, ax0_0 * 64 + ax0_1)
                            v_ax1 = Ts.axis.spatial(256, ax1_0 * 64 + ax1_1)
                            v_k1 = Ts.axis.reduce(256, k1_0 * 64 + k1_1)
                            Ts.reads(x_red_[v_ax0, v_ax1], x[v_ax0, v_k1, v_ax1])
                            Ts.writes(x_red_[v_ax0, v_ax1])
                            if v_k1 == 0:
                                x_red_[v_ax0, v_ax1] = T.float32(0.0)
                            x_red_[v_ax0, v_ax1] = x_red_[v_ax0, v_ax1] + x[v_ax0, v_k1, v_ax1]
                    for ax0, ax1 in T.grid(64, 64):
                        with Ts.sblock("x_red_"):
                            v0 = Ts.axis.spatial(256, ax0_0 * 64 + ax0)
                            v1 = Ts.axis.spatial(256, ax1_0 * 64 + ax1)
                            Ts.reads(x_red_[v0, v1])
                            Ts.writes(x_red[v0, v1])
                            x_red[v0, v1] = x_red_[v0, v1]

    _check(before, after)


if __name__ == "__main__":
    tvm_tirx.testing.main()
