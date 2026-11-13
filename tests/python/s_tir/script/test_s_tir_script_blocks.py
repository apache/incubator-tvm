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
# ruff: noqa: F841
"""S-TIR script blocks."""

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import assert_structural_equal_ignore_global_symbol
from tvm_tirx.ir import Range, TensorRegion, TensorRegionType
from tvm_tirx.script import tirx as T


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


@Ts.function
def matmul_original(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j in T.grid(32, 32):
        with Ts.sblock("init"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            for ii, jj in T.grid(4, 4):
                C[vi * 4 + ii, vj * 4 + jj] = T.float32(0)

        for k in range(0, 32):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                for ii, jj, kk in T.grid(4, 4, 4):
                    C[vi * 4 + ii, vj * 4 + jj] = (
                        C[vi * 4 + ii, vj * 4 + jj]
                        + A[vi * 4 + ii, vk * 4 + kk] * B[vj * 4 + jj, vk * 4 + kk]
                    )


@Ts.function
def elementwise_with_root(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    with Ts.sblock():
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.float32(1)
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + T.float32(1)


@Ts.function
def func_with_part_access_region(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    with Ts.sblock():
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi, vj])
                B[vi, vj] = A[vi, vj] + T.float32(1)

        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + T.float32(1)


def test_complete_matmul():
    func = matmul
    A, B, C = [x for x in func.params if tvm_tirx.tirx.is_tensor_var(x)]

    block = func.body[0].block.body[0].body[0].body[0].body[0].block
    assert isinstance(block, tvm.s_tir.SBlock)
    vi, vj, vk = [x.var for x in block.iter_vars]
    access_A = TensorRegion(
        A, [Range.from_min_extent(vi, 1), Range.from_min_extent(vk, 1)], ty=TensorRegionType()
    )
    access_B = TensorRegion(
        B, [Range.from_min_extent(vj, 1), Range.from_min_extent(vk, 1)], ty=TensorRegionType()
    )
    access_C = TensorRegion(
        C, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)], ty=TensorRegionType()
    )
    tvm_tirx.ir.assert_structural_equal(block.reads, [access_A, access_B])
    tvm_tirx.ir.assert_structural_equal(block.writes, [access_C])


def test_complete_matmul_original():
    func = matmul_original
    A, B, C = [x for x in func.params if tvm_tirx.tirx.is_tensor_var(x)]

    block1 = func.body[0].block.body[0].body[0].body[0].block
    assert isinstance(block1, tvm.s_tir.SBlock)
    vi, vj = [x.var for x in block1.iter_vars]
    access_C = TensorRegion(
        C,
        [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vj * 4, 4)],
        ty=TensorRegionType(),
    )
    tvm_tirx.ir.assert_structural_equal(block1.reads, [])
    tvm_tirx.ir.assert_structural_equal(block1.writes, [access_C])

    block2 = func.body[0].block.body[0].body[0].body[1].body[0].block
    assert isinstance(block2, tvm.s_tir.SBlock)
    vi, vj, vk = [x.var for x in block2.iter_vars]
    access_A = TensorRegion(
        A,
        [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vk * 4, 4)],
        ty=TensorRegionType(),
    )
    access_B = TensorRegion(
        B,
        [Range.from_min_extent(vj * 4, 4), Range.from_min_extent(vk * 4, 4)],
        ty=TensorRegionType(),
    )
    access_C = TensorRegion(
        C,
        [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vj * 4, 4)],
        ty=TensorRegionType(),
    )
    tvm_tirx.ir.assert_structural_equal(block2.reads, [access_C, access_A, access_B])
    tvm_tirx.ir.assert_structural_equal(block2.writes, [access_C])


def _check_elementwise(func):
    A, B, C = [x for x in func.params if tvm_tirx.tirx.is_tensor_var(x)]

    root_block = func.body[0].block
    assert len(root_block.reads) == 0
    assert len(root_block.writes) == 0

    block1 = func.body[0].block.body[0].body[0].body[0].block
    assert isinstance(block1, tvm.s_tir.SBlock)
    vi, vj = [x.var for x in block1.iter_vars]

    tvm_tirx.ir.assert_structural_equal(
        block1.reads,
        [
            TensorRegion(
                A,
                [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)],
                ty=TensorRegionType(),
            )
        ],
    )
    tvm_tirx.ir.assert_structural_equal(
        block1.writes,
        [
            TensorRegion(
                B,
                [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)],
                ty=TensorRegionType(),
            )
        ],
    )

    block2 = func.body[0].block.body[1].body[0].body[0].block
    assert isinstance(block2, tvm.s_tir.SBlock)
    vi, vj = [x.var for x in block2.iter_vars]
    tvm_tirx.ir.assert_structural_equal(
        block2.reads,
        [
            TensorRegion(
                B,
                [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)],
                ty=TensorRegionType(),
            )
        ],
    )
    tvm_tirx.ir.assert_structural_equal(
        block2.writes,
        [
            TensorRegion(
                C,
                [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)],
                ty=TensorRegionType(),
            )
        ],
    )


def test_complete_with_root():
    _check_elementwise(elementwise_with_root)


def test_complete_part_region():
    _check_elementwise(func_with_part_access_region)


@Ts.function
def func_with_tensorslice_indices(
    data_buf: T.Tensor((16, 16), "float32"), index_buf: T.Tensor((1,), "int32")
) -> None:
    out_buf = Ts.sblock_alloc_tensor((16, 16), "float32")

    for i, j in T.grid(16, 16):
        with Ts.sblock():
            vi, vj = Ts.axis.remap("SS", [i, j])
            out_buf[vi, vj] = data_buf[vi, index_buf[0]]


@Ts.function
def expected_tensorslice_indices(
    data_buf: T.Tensor([16, 16], "float32", align=64),
    index_buf: T.Tensor([1], dtype="int32", align=64),
) -> None:
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        out_buf = Ts.sblock_alloc_tensor([16, 16], align=64)
        for i0, i1 in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i0, i1])
                Ts.reads([data_buf[vi, index_buf[0]], index_buf[0]])
                Ts.writes([out_buf[vi, vj]])
                out_buf[vi, vj] = data_buf[vi, index_buf[0]]


@Ts.function
def func_with_recursive_tensorslice_indices(
    data_buf: T.Tensor((16, 16), "float32"), index_buf: T.Tensor((1,), "int32")
) -> None:
    out_buf = Ts.sblock_alloc_tensor((16, 16), "float32")

    for i, j in T.grid(16, 16):
        with Ts.sblock():
            vi, vj = Ts.axis.remap("SS", [i, j])
            out_buf[vi, vj] = data_buf[index_buf[index_buf[0]], index_buf[0]]


@Ts.function
def expected_recursive_tensorslice_indices(
    data_buf: T.Tensor([16, 16], "float32", align=64),
    index_buf: T.Tensor([1], dtype="int32", align=64),
) -> None:
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        out_buf = Ts.sblock_alloc_tensor([16, 16], align=64)
        for i0, i1 in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(
                    [
                        data_buf[index_buf[index_buf[0]], index_buf[0]],
                        index_buf[T.min(index_buf[0], 0) : T.max(index_buf[0], 0) + 1],
                    ]
                )
                Ts.writes([out_buf[vi, vj]])
                out_buf[vi, vj] = data_buf[index_buf[index_buf[0]], index_buf[0]]


def test_complete_tensor_indices():
    new_func = tvm_tirx.script.from_source(
        func_with_tensorslice_indices.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    ).with_attr("global_symbol", "main")
    tvm_tirx.ir.assert_structural_equal(
        new_func, expected_tensorslice_indices.with_attr("global_symbol", "main")
    )
    new_func = tvm_tirx.script.from_source(
        func_with_recursive_tensorslice_indices.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    ).with_attr("global_symbol", "main")
    tvm_tirx.ir.assert_structural_equal(
        new_func, expected_recursive_tensorslice_indices.with_attr("global_symbol", "main")
    )


@Ts.function
def match_tensor_func(A: T.Tensor((16, 16), "float32")) -> None:
    for i in range(0, 16):
        with Ts.sblock():
            A0 = Ts.match_tensor(A[i, 0:16], (16))
            with Ts.sblock():
                for j in range(0, 16):
                    with Ts.sblock():
                        A1 = Ts.match_tensor(A0[j], ())
                        A1[()] = 1.0


@Ts.function
def expected_match_tensor_func(A: T.Tensor((16, 16), "float32")) -> None:
    for i in range(0, 16):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i, 0:16])
            A0 = Ts.match_tensor(A[i, 0:16], (16))
            with Ts.sblock():
                Ts.reads([])
                Ts.writes(A0[0:16])
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads([])
                        Ts.writes(A0[j])
                        A1 = Ts.match_tensor(A0[j], ())
                        A1[()] = 1.0


def test_complete_match_tensor():
    tvm_tirx.ir.assert_structural_equal(
        match_tensor_func.with_attr("global_symbol", "main"),
        expected_match_tensor_func.with_attr("global_symbol", "main"),
    )


@Ts.function
def alloc_tensor_func(
    A: T.Tensor([2, 2], dtype="float32"), B: T.Tensor([2, 2], dtype="float32")
) -> None:
    C = Ts.sblock_alloc_tensor([2, 2], dtype="float32")
    A[(0, 0)] = T.float32(2)
    C[(0, 0)] = A[(0, 0)] + B[(0, 0)]
    B[(0, 0)] = C[(0, 0)]


@Ts.function
def expect_alloc_tensor_func(
    A: T.Tensor([2, 2], dtype="float32", align=64),
    B: T.Tensor([2, 2], dtype="float32", align=64),
) -> None:
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        C = Ts.sblock_alloc_tensor(
            [2, 2],
            dtype="float32",
            align=64,
        )
        A[(0, 0)] = T.float32(2)
        C[(0, 0)] = A[(0, 0)] + B[(0, 0)]
        B[(0, 0)] = C[(0, 0)]


def test_complete_alloc_tensor():
    rt_func = tvm_tirx.script.from_source(
        alloc_tensor_func.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    ).with_attr("global_symbol", "main")
    tvm_tirx.ir.assert_structural_equal(
        rt_func, expect_alloc_tensor_func.with_attr("global_symbol", "main")
    )


@Ts.function
def alloc_zero_dim_tensor(
    A: T.Tensor([], dtype="float32"), B: T.Tensor([], dtype="float32")
) -> None:
    # body
    # tirx.with block("root")
    C = Ts.sblock_alloc_tensor([], dtype="float32")
    A[()] = T.float32(2)
    C[()] = A[()] + B[()]
    B[()] = C[()]


@Ts.function
def alloc_zero_dim_tensor_block(A: T.Tensor((), "float32"), B: T.Tensor((), "float32")) -> None:
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        C = Ts.sblock_alloc_tensor((), "float32")
        A[()] = T.float32(2)
        C[()] = A[()] + B[()]
        B[()] = C[()]


def _check_alloc_zero_dim_tensor(f):
    dtype = "float32"
    ctx = tvm_tirx.cpu()

    np_data = np.zeros(shape=()).astype(dtype)
    np_out = np.zeros(shape=()).astype(dtype)
    tvm_data = tvm_tirx.runtime.tensor(np_data, ctx)
    tvm_out = tvm_tirx.runtime.tensor(np_out, ctx)

    # np func exection
    np_inter = np.array(1)
    np_data[()] = 2.0
    np_inter[()] = np_data[()] + np_out[()]
    np_out[()] = np_inter[()]

    # tvm func execution
    f(tvm_data, tvm_out)
    tvm_tirx.testing.assert_allclose(tvm_out.numpy(), np_out, rtol=1e-5)


def test_alloc_zero_dim_tensor_round_trip():
    func = alloc_zero_dim_tensor
    func_with_block = alloc_zero_dim_tensor_block
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    )
    rt_func_with_block = tvm_tirx.script.from_source(
        func_with_block.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    )
    rt_mod = tvm.compile(rt_func, "llvm")
    rt_mod_with_block = tvm.compile(rt_func_with_block, "llvm")
    tvm_tirx.ir.assert_structural_equal(
        func.with_attr("global_symbol", "main"), func_with_block.with_attr("global_symbol", "main")
    )
    tvm_tirx.ir.assert_structural_equal(
        rt_func.with_attr("global_symbol", "main"),
        rt_func_with_block.with_attr("global_symbol", "main"),
    )
    _check_alloc_zero_dim_tensor(rt_mod)
    _check_alloc_zero_dim_tensor(rt_mod_with_block)


def test_different_dtype_assignment_to_var():
    @Ts.function
    def test_case():
        a = Ts.sblock_alloc_tensor((10, 10), dtype="int8")

    @Ts.function
    def func_ref():
        a = Ts.sblock_alloc_tensor([10, 10], dtype="int8")
        T.evaluate(0)

    tvm_tirx.ir.assert_structural_equal(
        test_case.with_attr("global_symbol", "main"), func_ref.with_attr("global_symbol", "main")
    )


def roundtrip_matmul():
    @Ts.function
    def roundtrip_matmul(
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

    return roundtrip_matmul


def roundtrip_matmul_original():
    @Ts.function
    def roundtrip_matmul_original(
        A: T.Tensor([128, 128], "float32"),
        B: T.Tensor([128, 128], "float32"),
        C: T.Tensor([128, 128], "float32"),
    ) -> None:
        for i, j in T.grid(128, 128):
            with Ts.sblock("init"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = T.float32(0)

            for k in range(128):
                with Ts.sblock("update"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    return roundtrip_matmul_original


def element_wise():
    @Ts.function
    def element_wise(
        A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
    ) -> None:
        B = Ts.sblock_alloc_tensor((128, 128), "float32")

        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * T.float32(2)
        for i, j in T.grid(128, 128):
            with Ts.sblock("C"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + T.float32(1)

    return element_wise


def predicate():
    @Ts.function
    def predicate(B: T.Tensor((16, 16), "float32"), C: T.Tensor((16, 16), "float32")) -> None:
        for i, jo, ji in T.grid(16, 4, 5):
            with Ts.sblock("update"):
                vi = Ts.axis.S(16, i)
                vj = Ts.axis.S(16, jo * 4 + ji)
                Ts.where(jo * 4 + ji < 16)
                C[vi, vj] = B[vi, vj] + T.float32(1)

    return predicate


def test_module_define():
    func1 = tvm_tirx.ir.IRModule({"matmul": roundtrip_matmul()})["matmul"]
    func2 = tvm_tirx.ir.IRModule({"element_wise": element_wise()})["element_wise"]
    func3 = tvm_tirx.ir.IRModule({"predicate": predicate()})["predicate"]
    mod1 = tvm_tirx.ir.IRModule({"func1": func1, "func2": func2, "func3": func3})
    mod2 = tvm_tirx.ir.IRModule(
        {"func1": roundtrip_matmul(), "func2": element_wise(), "func3": predicate()}
    )
    tvm_tirx.ir.assert_structural_equal(mod1, mod2)


def test_matmul_original():
    func = roundtrip_matmul_original()
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body[0].block, s_tir.SBlock)
    assert isinstance(rt_func.body[0].block.body[0], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[0].body[0], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[0].body[0].body, tvm_tirx.ir.SeqStmt)
    assert isinstance(rt_func.body[0].block.body[0].body[0].body[0].block, s_tir.SBlock)
    assert isinstance(rt_func.body[0].block.body[0].body[0].body[1], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[0].body[0].body[1].body[0].block, s_tir.SBlock)


def test_element_wise():
    func = element_wise()
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body[0].block, s_tir.SBlock)
    assert isinstance(rt_func.body[0].block.body, tvm_tirx.ir.SeqStmt)
    assert isinstance(rt_func.body[0].block.body[0], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[0].body[0], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[0].body[0].body[0].block, s_tir.SBlock)

    assert isinstance(rt_func.body[0].block.body[1], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[1].body[0], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[1].body[0].body[0].block, s_tir.SBlock)


def test_predicate():
    func = predicate()
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body[0].block, s_tir.SBlock)
    assert isinstance(rt_func.body[0].block.body[0], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[0].body[0], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[0].body[0].body[0], tvm_tirx.ir.For)
    assert isinstance(rt_func.body[0].block.body[0].body[0].body[0].body[0].block, s_tir.SBlock)


def match_tensor_region():
    @Ts.function
    def match_tensor_region(
        A: T.Tensor((16, 16, 16), "float32"), B: T.Tensor(1, "float32")
    ) -> None:
        for i, j in T.grid(16, 4):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                C = Ts.match_tensor(A[0:16, vi, vj * 4 : vj * 4 + 4], (16, 1, 4))
                for ii in range(4):
                    with Ts.sblock():
                        vii = Ts.axis.S(4, ii)
                        D = Ts.match_tensor(C[vii * 4 : vii * 4 + 4, 0, 0:4], (4, 1, 4))
                        for i, j in T.grid(4, 4):
                            B[0] += D[i, 0, j]

    return match_tensor_region


def test_match_tensor_region():
    func = match_tensor_region()
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body[0], s_tir.SBlockRealize)
    root = rt_func.body[0].block

    assert isinstance(root.body[0], tvm_tirx.ir.For)
    assert isinstance(root.body[0].body[0], tvm_tirx.ir.For)
    assert isinstance(root.body[0].body[0].body[0], s_tir.SBlockRealize)
    outer_block = root.body[0].body[0].body[0].block
    assert len(outer_block.match_tensors) == 1
    tensor_C = outer_block.match_tensors[0].tensor
    tvm_tirx.ir.assert_structural_equal(tensor_C.shape, [T.int32(16), T.int32(1), T.int32(4)])

    assert isinstance(outer_block.body[0], tvm_tirx.ir.For)
    assert isinstance(outer_block.body[0].body[0], s_tir.SBlockRealize)
    inner_block = outer_block.body[0].body[0].block
    assert len(inner_block.match_tensors) == 1
    tensor_D = inner_block.match_tensors[0].tensor
    tvm_tirx.ir.assert_structural_equal(tensor_D.shape, [T.int32(4), T.int32(1), T.int32(4)])


def block_elements():
    @Ts.function
    def block_elements(A: T.Tensor((16, 16), "float32"), B: T.Tensor((1, 1), "float32")) -> None:
        with Ts.sblock("update"):
            vi = Ts.axis.S(1, 0)
            Ts.where(True)
            Ts.reads(A[0:16, 0:16])
            Ts.writes(B[0, 0])
            Ts.sblock_attr({"attr_key": "attr_value"})
            C = Ts.sblock_alloc_tensor((4, 4), dtype="float32")
            D = Ts.match_tensor(A[0:4, 0], (4, 1))
            with Ts.init():
                B[0, 0] = T.float32(0)
            B[0, 0] = A[0, 0] + B[0, 0] + C[1, 1] + D[2, 0]

    return block_elements


def test_block_elements():
    func = block_elements()
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body[0].block, s_tir.SBlock)
    assert isinstance(rt_func.body[0].block.body[0], s_tir.SBlockRealize)
    assert isinstance(rt_func.body[0].block.body[0].block, s_tir.SBlock)
    block = rt_func.body[0].block.body[0].block
    assert isinstance(block.body[0], tvm_tirx.ir.TensorStore)
    assert isinstance(block.init[0], tvm_tirx.ir.TensorStore)
    assert len(block.annotations) == 1
    assert block.annotations["attr_key"] == "attr_value"


def opaque_block():
    @Ts.function
    def opaque_block(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")) -> None:
        for i in range(16):
            for j in range(16):
                with Ts.sblock():
                    Ts.reads([])
                    Ts.writes(A[i, j])
                    A[i, j] = T.float32(0)
            with Ts.sblock():
                Ts.reads([A[i, 0:16]])
                Ts.writes([B[i, 0:16]])
                for j in range(16):
                    B[i, j] = A[i, j]

    return opaque_block


def test_opaque_block():
    func = opaque_block()
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)

    root_block = rt_func.body[0].block
    assert isinstance(root_block, s_tir.SBlock)
    assert isinstance(root_block.body[0], tvm_tirx.ir.For)
    assert isinstance(root_block.body[0].body[0], tvm_tirx.ir.For)
    assert isinstance(root_block.body[0].body[0].body[0], s_tir.SBlockRealize)
    assert isinstance(root_block.body[0].body[0].body[0].block, s_tir.SBlock)
    assert len(root_block.body[0].body[0].body[0].block.iter_vars) == 0
    assert isinstance(root_block.body[0].body[1], s_tir.SBlockRealize)
    assert isinstance(root_block.body[0].body[1].block, s_tir.SBlock)
    assert len(root_block.body[0].body[1].block.iter_vars) == 0


def rank0():
    @Ts.function
    def rank0(A: T.Tensor((), "float32")) -> None:
        B = Ts.sblock_alloc_tensor((), "float32")
        A[()] = 2
        B[()] = A[()]

    return rank0


def rank0_block():
    @Ts.function
    def rank0_block(A: T.Tensor((), "float32")) -> None:
        B = Ts.sblock_alloc_tensor((), "float32")
        B[()] = A[()]

        with Ts.sblock("update"):
            Ts.reads([A[()]])
            Ts.writes([B[()]])
            for i in range(1):
                B[()] = A[()]

    return rank0_block


def nontrivial_range_axis():
    @Ts.function
    def nontrivial_range_axis(A: T.Tensor(10, "float32")) -> None:
        for i in range(10):
            with Ts.sblock("block"):
                vi = Ts.axis.spatial((1, 11), i + 1)
                A[vi - 1] = A[vi - 1] + 1.0

    return nontrivial_range_axis


def func_root_attr():
    @Ts.function
    def func_root_attr():
        with Ts.sblock("root"):
            Ts.sblock_attr({"a": "0"})
            T.evaluate(0)

    return func_root_attr


def func_trivial_root_block():
    @Ts.function
    def func(A: T.Tensor(1, "int32")):
        with Ts.sblock("root"):
            A[0] = 0

    return func


def func_nested_root_block():
    @Ts.function
    def func(A: T.Tensor(1, "int32")):
        with Ts.sblock("root"):
            with Ts.sblock("block"):
                A[0] = 0

    return func


def int64_support():
    @Ts.function
    def elementwise_shape_int64(
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

    return elementwise_shape_int64


def func_attr_with_list():
    @Ts.function
    def func(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        D: T.Tensor((128, 128), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
        C = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
        for i0, i1, i2 in T.grid(128, 128, 128):
            with Ts.sblock("C"):
                x, y, k = Ts.axis.remap("SSR", [i0, i1, i2])
                with Ts.init():
                    C[x, y] = T.float32(0)
                C[x, y] = C[x, y] + A[x, k] * B[y, k]
        for i0, i1 in T.grid(128, 128):
            with Ts.sblock("D"):
                Ts.sblock_attr({"layout_free_placeholders": [C]})
                x, y = Ts.axis.remap("SS", [i0, i1])
                D[x, y] = C[x, y] + T.float32(1)

    return func


@Ts.function
def transformed_matmul_no_syntax_sugar(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with Ts.sblock("update"):
            vi, vj = Ts.axis.remap("SS", [i0, i1])
            vk = Ts.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            Ts.reads([C[vi, vj], A[vi, vk], B[vj, vk]])
            Ts.writes([C[vi, vj], A[vi, vk]])
            with Ts.init():
                C[vi, vj] = 0.0
            A[vi, vk] = A[vi, vk] + B[vj, vk]
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@Ts.function
def transformed_matmul_syntax_sugar(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with Ts.sblock("update"):
            vi, vj = Ts.axis.remap("SS", [i0, i1])
            vk = Ts.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            Ts.reads(C[vi, vj], A[vi, vk], B[vj, vk])
            Ts.writes(C[vi, vj], A[vi, vk])
            with Ts.init():
                C[vi, vj] = 0.0
            A[vi, vk] = A[vi, vk] + B[vj, vk]
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


def test_reads_writes_syntax_sugar():
    assert_structural_equal_ignore_global_symbol(
        transformed_matmul_no_syntax_sugar, transformed_matmul_syntax_sugar
    )


def test_match_tensor_region_has_implicit_shape_dtype():
    @Ts.function
    def explicit_shape_dtype(A: T.Tensor((16, 64), "int32")):
        with Ts.sblock():
            B = Ts.match_tensor(A[8:16, 32:64], shape=(8, 32), dtype="int32")
            T.evaluate(0)

    @Ts.function
    def implicit_shape_dtype(A: T.Tensor((16, 64), "int32")):
        with Ts.sblock():
            B = Ts.match_tensor(A[8:16, 32:64])
            T.evaluate(0)

    assert_structural_equal_ignore_global_symbol(explicit_shape_dtype, implicit_shape_dtype)


@pytest.mark.parametrize(
    "ir_generator",
    [
        roundtrip_matmul,
        rank0,
        rank0_block,
        nontrivial_range_axis,
        func_root_attr,
        func_trivial_root_block,
        func_nested_root_block,
        int64_support,
        func_attr_with_list,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_blocks(ir_generator):
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


# Import-time construction also checks the annotated S-TIR API.
@Ts.function
def element_wise_storage_align(
    A: T.Tensor([128, 128], "float32", align=64),
    C: T.Tensor([128, 128], "float32", align=64),
) -> None:
    # body
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        B = Ts.sblock_alloc_tensor([128, 128], align=64)
        for i0 in T.serial(0, 128):
            for ax1 in T.serial(0, 128):
                with Ts.sblock("B"):
                    vi = Ts.axis.S(128, i0)
                    vj = Ts.axis.S(128, ax1)
                    Ts.reads([A[vi, vj]])
                    Ts.writes([B[vi, vj]])
                    Ts.sblock_attr({"tensor_dim_align": [[0, 0, 128, 127]]})
                    B[vi, vj] = A[vi, vj] * T.float32(2)
            for i1 in T.serial(0, 128):
                with Ts.sblock("C"):
                    vi_1, vj_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads([B[vi_1, vj_1]])
                    Ts.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = B[vi_1, vj_1] + T.float32(1)


# Import-time construction also checks the annotated S-TIR API.
@Ts.function
def loop_split(
    A: T.Tensor([128, 128], dtype="float32"), B: T.Tensor([128], dtype="float32")
) -> None:
    for i, ko in T.grid(128, 4):
        for ki in T.parallel(0, 32, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("B"):
                vi = Ts.axis.S(128, i)
                vk = Ts.axis.R(128, ko * 32 + ki)
                Ts.reads([B[vi], A[vi, vk]])
                Ts.writes([B[vi]])
                with Ts.init():
                    B[vi] = T.float32(0)
                B[vi] = B[vi] + A[vi, vk]


# Import-time construction also checks the annotated S-TIR API.
@Ts.function
def different_access_indices(
    A: T.Tensor([128, 128, 128], dtype="float32"), B: T.Tensor([128, 128], dtype="float32")
) -> None:
    for i, j in T.grid(128, 128):
        for k in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("B"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                Ts.reads([B[vi, vj], A[vi, vj, vk]])
                Ts.writes(
                    [
                        B[
                            T.min(vj, vi) : T.min(vj, vi)  # type: ignore[misc]
                            + (T.max(vj, vi) + 1 - T.min(vj, vi)),
                            T.min(vi, vj) : T.min(vi, vj)  # type: ignore[misc]
                            + (T.max(vi, vj) + 1 - T.min(vi, vj)),
                        ]
                    ]
                )
                with Ts.init():
                    B[vj, vi] = T.exp(B[vj, vi])
                B[vi, vj] = B[vi, vj] + A[vi, vj, vk]


if __name__ == "__main__":
    tvm_tirx.testing.main()
