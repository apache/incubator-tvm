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
import sys

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


def _check(original, transformed):
    mod = tvm_tirx.IRModule.from_expr(original.with_attr("global_symbol", "main"))
    mod = tvm.s_tir.transform.LowerThreadBinding()(mod)
    mod = tvm.s_tir.transform.StmtSimplify()(mod)
    tvm_tirx.ir.assert_structural_equal(
        mod["main"], transformed.with_attr("global_symbol", "main"), True
    )


def _check_fail(original):
    mod = tvm_tirx.IRModule.from_expr(
        original.with_attr("target", tvm_tirx.target.Target("cuda", host="llvm"))
    )
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    mod = tvm.s_tir.transform.LowerThreadBinding()(mod)
    with pytest.raises(ValueError, match="Incompatible launch extents for threadIdx.x"):
        tvm_tirx.tirx.transform.SplitHostDevice()(mod)


@Ts.function
def element_wise_thread_x(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i in T.parallel(0, 128, annotations={"thread_binding": "blockIdx.x"}):
        for j0_0 in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
            for j0_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for j1_0 in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
            for j1_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


@Ts.function
def lowered_element_wise_thread_x(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    with T.launch_thread("blockIdx.x", 128) as i:
        with T.launch_thread("threadIdx.x", 4) as j0_0:
            for j0_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        with T.launch_thread("threadIdx.x", 4) as j1_0:
            for j1_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


@Ts.function
def element_wise_thread_x_different_dtype(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
) -> None:
    for i in T.parallel(128, annotations={"thread_binding": "blockIdx.x"}):
        for j0_0 in T.parallel(4, annotations={"thread_binding": "threadIdx.x"}):
            for j0_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for j1_0 in T.parallel(T.int64(4), annotations={"thread_binding": "threadIdx.x"}):
            for j1_1 in T.serial(T.int64(32)):
                with Ts.sblock(""):
                    C[i, j1_0 * T.int64(32) + j1_1] = B[i, j1_0 * T.int64(32) + j1_1] + 1.0


@Ts.function
def lowered_element_wise_thread_x_different_dtype(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
) -> None:
    with T.launch_thread("blockIdx.x", 128) as i:
        with T.launch_thread("threadIdx.x", 4) as j0_0:
            for j0_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        with T.launch_thread("threadIdx.x", T.int64(4)) as j1_0:
            for j1_1 in T.serial(T.int64(32)):
                with Ts.sblock(""):
                    C[i, j1_0 * T.int64(32) + j1_1] = B[i, j1_0 * T.int64(32) + j1_1] + 1.0


@Ts.function
def element_wise_launch_thread_x(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    i = T.launch_thread("blockIdx.x", 128)
    j0_0 = T.launch_thread("threadIdx.x", 4)
    j1_0 = T.launch_thread("threadIdx.x", 4)

    for j0_1 in T.serial(0, 32):
        with Ts.sblock(""):
            B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
    for j1_1 in T.serial(0, 32):
        with Ts.sblock(""):
            C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


@Ts.function
def lowered_element_wise_launch_thread_x(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    i = T.launch_thread("blockIdx.x", 128)
    j0_0 = T.launch_thread("threadIdx.x", 4)
    j1_0 = T.launch_thread("threadIdx.x", 4)

    for j0_1 in T.serial(0, 32):
        with Ts.sblock(""):
            B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
    for j1_1 in T.serial(0, 32):
        with Ts.sblock(""):
            C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


@Ts.function
def element_wise_vthread_x(
    A: T.Tensor([128, 128], "float32"), B: T.Tensor([128, 128], "float32")
) -> None:
    for i_0 in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
        for i_1 in T.parallel(0, 64, annotations={"thread_binding": "threadIdx.x"}):
            for j_0 in T.parallel(0, 2, annotations={"thread_binding": "vthread.x"}):
                for j_1 in T.serial(0, 64):
                    with Ts.sblock(""):
                        B[i_0 * 64 + i_1, j_0 * 64 + j_1] = A[i_0 * 64 + i_1, j_0 * 64 + j_1] * 2.0


@Ts.function
def lowered_element_wise_vthread_x(
    A: T.Tensor([128, 128], "float32"), B: T.Tensor([128, 128], "float32")
) -> None:
    with T.launch_thread("vthread.x", 2) as i_0:
        with T.launch_thread("threadIdx.x", 64) as i_1:
            with T.launch_thread("vthread.x", 2) as j_0:
                for j_1 in T.serial(0, 64):
                    with Ts.sblock(""):
                        B[i_0 * 64 + i_1, j_0 * 64 + j_1] = A[i_0 * 64 + i_1, j_0 * 64 + j_1] * 2.0


@Ts.function
def element_wise_two_thread_x_in_same_kernel_not_equal(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 64], "float32"),
) -> None:
    for i in T.parallel(0, 128, annotations={"thread_binding": "blockIdx.x"}):
        for j0 in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
            B[i, j0] = A[i, j0] * 2.0
        for j1 in T.parallel(0, 64, annotations={"thread_binding": "threadIdx.x"}):
            C[i, j1] = A[i, j1] + 1.0


@Ts.function
def element_wise_kernels_with_different_size(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([256, 256], "float32"),
    D: T.Tensor([256, 256], "float32"),
) -> None:
    for i0 in T.parallel(0, 128, annotations={"thread_binding": "blockIdx.x"}):
        for j0 in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
            B[i0, j0] = A[i0, j0] * 2.0
    for i1 in T.parallel(0, 256, annotations={"thread_binding": "blockIdx.x"}):
        for j1 in T.parallel(0, 256, annotations={"thread_binding": "threadIdx.x"}):
            D[i1, j1] = C[i1, j1] + 1.0


@Ts.function
def lowered_element_wise_kernels_with_different_size(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([256, 256], "float32"),
    D: T.Tensor([256, 256], "float32"),
) -> None:
    with T.launch_thread("blockIdx.x", 128) as i0:
        with T.launch_thread("threadIdx.x", 128) as j0:
            B[i0, j0] = A[i0, j0] * 2.0
    with T.launch_thread("blockIdx.x", 256) as i1:
        with T.launch_thread("threadIdx.x", 256) as j1:
            D[i1, j1] = C[i1, j1] + 1.0


@Ts.function
def element_wise_implicit_block(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.y"}):
        for j0_0 in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
            for j0_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        for j1_0 in T.parallel(0, 4, annotations={"thread_binding": "threadIdx.x"}):
            for j1_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


@Ts.function
def lowered_element_wise_implicit_block(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    with T.launch_thread("threadIdx.y", 128) as i:
        with T.launch_thread("threadIdx.x", 4) as j0_0:
            for j0_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    B[i, j0_0 * 32 + j0_1] = A[i, j0_0 * 32 + j0_1] * 2.0
        with T.launch_thread("threadIdx.x", 4) as j1_0:
            for j1_1 in T.serial(0, 32):
                with Ts.sblock(""):
                    C[i, j1_0 * 32 + j1_1] = B[i, j1_0 * 32 + j1_1] + 1.0


def test_thread_x():
    _check(element_wise_thread_x, lowered_element_wise_thread_x)


def test_thread_x_different_dtype():
    _check(element_wise_thread_x_different_dtype, lowered_element_wise_thread_x_different_dtype)


def test_launch_thread_x():
    _check(element_wise_launch_thread_x, lowered_element_wise_launch_thread_x)


def test_vthread_x():
    _check(element_wise_vthread_x, lowered_element_wise_vthread_x)


def test_two_thread_x_in_same_kernel_not_equal():
    _check_fail(element_wise_two_thread_x_in_same_kernel_not_equal)


def test_kernels_with_different_size():
    _check(
        element_wise_kernels_with_different_size, lowered_element_wise_kernels_with_different_size
    )


def test_implicit_block():
    _check(element_wise_implicit_block, lowered_element_wise_implicit_block)


def test_inner_binding_with_annotation():
    @Ts.function
    def inner_binding_with_annotation(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        for bx in T.parallel(32, annotations={"thread_binding": "blockIdx.x"}):
            for tx in T.parallel(
                2, annotations={"thread_binding": "threadIdx.x", "my_annotation": 1}
            ):
                with Ts.sblock("block"):
                    v = Ts.axis.spatial(64, bx * 2 + tx)
                    B[v] = A[v]

    @Ts.function
    def lowered_inner_binding_with_annotation(
        A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")
    ):
        with T.launch_thread("blockIdx.x", 32) as blockIdx_x:
            with T.launch_thread("threadIdx.x", 2) as threadIdx_x:
                for var in T.serial(1, annotations={"my_annotation": 1}):
                    with Ts.sblock("block"):
                        v = Ts.axis.spatial(64, blockIdx_x * 2 + threadIdx_x)
                        Ts.reads(A[v])
                        Ts.writes(B[v])
                        B[v] = A[v]

    _check(inner_binding_with_annotation, lowered_inner_binding_with_annotation)


if __name__ == "__main__":
    tvm_tirx.testing.main()
