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
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx.ir import Range, TensorRegion, TensorRegionType
from tvm_tirx.script import tirx as T


@Ts.function
def func() -> None:
    A = Ts.sblock_alloc_tensor((128, 128), "float32")
    B = Ts.sblock_alloc_tensor((128, 128), "float32")
    C = Ts.sblock_alloc_tensor((128, 128), "float32")
    D = Ts.sblock_alloc_tensor((128, 128), "float32")
    with Ts.sblock():
        # Need add read/write region manually to avoid triggering block access region detector
        Ts.reads([B[0, 0], C[0:16, 0:16], A[4:12, 4:12]])
        Ts.writes([A[0:12, 0:12]])
        for i, j in T.grid(8, 8):
            A[i, j] = B[0, 0] + C[0, 0]
        for i, j in T.grid(2, 2):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads([A[vi * 4 + 4 : vi * 4 + 8, vj * 4 + 4 : vj * 4 + 8], C[12:16, 12:16]])
                Ts.writes([A[vi * 4 + 4 : vi * 4 + 8, vj * 4 + 4 : vj * 4 + 8]])
                for i, j in T.grid(4, 4):
                    A[vi * 4 + 4 + i, vj * 4 + 4 + j] += C[i + 12, j + 12]
        T.evaluate(D.data)


@Ts.function
def masked_access_func() -> None:
    A = Ts.sblock_alloc_tensor((16,), "float32")
    B = Ts.sblock_alloc_tensor((16,), "float32")
    with Ts.sblock():
        mask = T.meta_var(T.Broadcast(T.bool(True), 4))
        value = T.meta_var(T.masked_load(A, T.Ramp(4, 1, 4), mask, ty="float32x4"))
        T.masked_store(B, value, T.Ramp(8, 1, 4), mask)


@Ts.function
def match_tensor_func() -> None:
    with Ts.sblock("root"):
        A = Ts.sblock_alloc_tensor((128, 128), "float32")
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        Ts.reads([])
        Ts.writes([])
        # Need add read/write region manually to avoid triggering block access region detector
        for i, j in T.grid(8, 8):
            with Ts.sblock("block"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(B[vi * 16 + 2 : vi * 16 + 12, vj * 16 + 2 : vj * 16 + 16])
                Ts.writes(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                AA = Ts.match_tensor(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16], (16, 16))
                B0 = Ts.match_tensor(
                    B[vi * 16 + 2 : vi * 16 + 6, vj * 16 + 2 : vj * 16 + 6], (4, 4)
                )
                B1 = Ts.match_tensor(
                    B[vi * 16 + 8 : vi * 16 + 12, vj * 16 + 8 : vj * 16 + 16], (4, 8)
                )
                for ii, jj in T.grid(16, 16):
                    with Ts.sblock("AAA"):
                        vii, vjj = Ts.axis.remap("SS", [ii, jj])
                        Ts.reads([])
                        Ts.writes(AA[vii, vjj])
                        AAA = Ts.match_tensor(AA[vii, vjj], ())
                        AAA[()] = 1.0
                T.evaluate(B0.data)
                T.evaluate(B1.data)


@Ts.function
def opaque_block_func() -> None:
    with Ts.sblock("root"):
        A = Ts.sblock_alloc_tensor((16, 16), "float32")
        B = Ts.sblock_alloc_tensor((16, 16), "float32")
        Ts.reads([])
        Ts.writes([])
        # Need add read/write region manually to avoid triggering block access region detector
        for i in range(0, 16):
            with Ts.sblock():
                Ts.reads(A[i, 0:16])
                Ts.writes([B[i, 0:16]])
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads(A[i, j])
                        Ts.writes(B[i, j])
                        B[i, j] = A[i, j] + 1.0


@Ts.function
def opaque_access_func() -> None:
    A = Ts.sblock_alloc_tensor([1024])
    B = Ts.sblock_alloc_tensor([1024])
    for i in T.serial(0, 8):
        with Ts.sblock():
            v = Ts.axis.S(8, i)
            Ts.reads([A[v * 128 : v * 128 + 128]])
            Ts.writes([B[v * 128 : v * 128 + 128]])
            T.evaluate(
                T.call_extern("test", B.data, v * 128, 128, A.data, v * 128, 128, ty="float32")
            )


@Ts.function
def opaque_access_with_pointer_func() -> None:
    A = Ts.sblock_alloc_tensor([1024])
    B = Ts.sblock_alloc_tensor([1024])
    C = Ts.sblock_alloc_tensor([1024])
    with Ts.sblock("opaque"):
        Ts.reads(A[0:1024], C[0:1024])
        Ts.writes(B[0:1024], C[0:1024])
        T.evaluate(A.ptr_to([0] * len(A.shape)))
        T.evaluate(B.ptr_to([0] * len(B.shape)))
        T.evaluate(C.ptr_to([0] * len(C.shape)))


@Ts.function
def decl_tensor_alias_func(
    A: T.Tensor((16,), "float32"),
    B: T.Tensor((16,), "float32"),
) -> None:
    with Ts.sblock("alias"):
        Ts.reads(A[0])
        Ts.writes(B[0])
        A_view = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "float32", data=A.data)])
        B[0] = A[0] + A_view[0]


@Ts.function
def access_in_if_then_else_func() -> None:
    A = Ts.sblock_alloc_tensor([8])
    B = Ts.sblock_alloc_tensor([8])
    with Ts.sblock():
        Ts.reads([A[0:5]])
        Ts.writes([B[0:8]])
        for i in T.serial(0, 8):
            B[i] = T.if_then_else(i < 5, A[i], 0.0)


@Ts.function
def access_in_branch_func() -> None:
    A = Ts.sblock_alloc_tensor([8])
    B = Ts.sblock_alloc_tensor([8])
    with Ts.sblock():
        Ts.reads([A[0:7]])
        Ts.writes([B[0:8]])
        for i in T.serial(0, 8):
            if i < 5:
                B[i] = A[i] + 1.0
            else:
                B[i] = A[i - 1]


@Ts.function
def gemm() -> None:
    A = Ts.sblock_alloc_tensor([16, 16], "float32")
    B = Ts.sblock_alloc_tensor([16, 16], "float32")
    C = Ts.sblock_alloc_tensor([16, 16], "float32")
    for i, j, k, ii, jj in T.grid(4, 4, 16, 4, 4):
        with Ts.sblock("update"):
            vi = Ts.axis.S(16, i * 4 + ii)
            vj = Ts.axis.S(16, j * 4 + jj)
            vk = Ts.axis.R(16, k)
            Ts.reads(A[vi, vk], B[vj, vk])
            Ts.writes(C[vi, vj])
            with Ts.init():
                C[vi, vj] = 0
            C[vi, vj] += A[vi, vk] * B[vj, vk]


@Ts.function
def decomposed_gemm() -> None:
    A = Ts.sblock_alloc_tensor([16, 16], "float32")
    B = Ts.sblock_alloc_tensor([16, 16], "float32")
    C = Ts.sblock_alloc_tensor([16, 16], "float32")
    for i, j in T.grid(4, 4):
        for ii, jj in T.grid(4, 4):
            with Ts.sblock("init"):
                vi = Ts.axis.S(16, i * 4 + ii)
                vj = Ts.axis.S(16, j * 4 + jj)
                Ts.reads([])
                Ts.writes(C[vi, vj])
                C[vi, vj] = 0
        for k, ii, jj in T.grid(16, 4, 4):
            with Ts.sblock("update"):
                vi = Ts.axis.S(16, i * 4 + ii)
                vj = Ts.axis.S(16, j * 4 + jj)
                vk = Ts.axis.R(16, k)
                Ts.reads(C[vi, vj], A[vi, vk], B[vj, vk])
                Ts.writes(C[vi, vj])
                C[vi, vj] += A[vi, vk] * B[vj, vk]


@Ts.function
def access_of_padding_pattern() -> None:
    X = Ts.sblock_alloc_tensor([28, 28])
    X_pad = Ts.sblock_alloc_tensor([32, 32])
    Y = Ts.sblock_alloc_tensor([28, 28])
    for i, j in T.grid(32, 32):
        with Ts.sblock("padding"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads([X[vi - 2, vj - 2]])
            Ts.writes([X_pad[vi, vj]])
            X_pad[vi, vj] = T.if_then_else(
                2 <= vi and vi < 30 and 2 <= vj and vj < 30, X[vi - 2, vj - 2], 0.0
            )
        with Ts.sblock("padding_reverse"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads([X_pad[vi, vj]])
            Ts.writes([Y[vi - 2, vj - 2]])
            if 2 <= vi and vi < 30 and 2 <= vj and vj < 30:
                Y[vi - 2, vj - 2] = X_pad[vi, vj]


def test_block_access_region_detector():
    block = func.body[0].block.body[0].block
    alloc_tensors = func.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}
    ret = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)

    tvm_tirx.ir.assert_structural_equal(block.reads, ret[0])
    tvm_tirx.ir.assert_structural_equal(block.writes, ret[1])
    D = alloc_tensors[-1]
    tvm_tirx.ir.assert_structural_equal(
        [TensorRegion(D, [Range(0, 128), Range(0, 128)], ty=TensorRegionType())], ret[2]
    )


def test_masked_access_is_not_opaque():
    root = masked_access_func.body[0].block
    block = root.body[0].block
    A, B = root.alloc_tensors
    reads, writes, opaque = s_tir.analysis.get_sblock_access_region(block, {A: A, B: B})
    tvm_tirx.ir.assert_structural_equal(
        reads, [TensorRegion(A, [Range.from_min_extent(4, 4)], ty=TensorRegionType())]
    )
    tvm_tirx.ir.assert_structural_equal(
        writes, [TensorRegion(B, [Range.from_min_extent(8, 4)], ty=TensorRegionType())]
    )
    tvm_tirx.ir.assert_structural_equal(opaque, [])


def test_opaque_block():
    alloc_tensors = opaque_block_func.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}

    block0 = opaque_block_func.body[0].block.body[0].body[0].block
    ret = s_tir.analysis.get_sblock_access_region(block0, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(block0.reads, ret[0])
    tvm_tirx.ir.assert_structural_equal(block0.writes, ret[1])

    block1 = block0.body[0].body[0].block
    ret = s_tir.analysis.get_sblock_access_region(block1, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(block1.reads, ret[0])
    tvm_tirx.ir.assert_structural_equal(block1.writes, ret[1])


def test_opaque_access():
    block = opaque_access_func.body[0].block.body[0].body[0].block
    alloc_tensors = opaque_access_func.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}

    ret0 = s_tir.analysis.get_sblock_read_write_region(block, tensor_var_map)
    ret1 = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
    with pytest.raises(ValueError):
        tvm_tirx.ir.assert_structural_equal(ret0[0], ret1[0])
    with pytest.raises(ValueError):
        tvm_tirx.ir.assert_structural_equal(ret0[1], ret1[1])


def test_opaque_access_with_pointer():
    block = opaque_access_with_pointer_func.body[0].block.body[0].block
    alloc_tensors = opaque_access_with_pointer_func.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}

    reads, writes = s_tir.analysis.get_sblock_read_write_region(block, tensor_var_map)
    direct_reads, direct_writes, opaque = s_tir.analysis.get_sblock_access_region(
        block, tensor_var_map
    )
    assert not direct_reads
    assert not direct_writes
    assert len(opaque) == 3
    tvm_tirx.ir.assert_structural_equal(reads, opaque)
    tvm_tirx.ir.assert_structural_equal(writes, opaque)


def test_decl_tensor_alias_is_not_an_opaque_access():
    block = decl_tensor_alias_func.body[0].block
    tensor_var_map = {
        buf: buf for buf in decl_tensor_alias_func.params if tvm_tirx.tirx.is_tensor_var(buf)
    }

    reads, writes, opaque = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(block.reads, reads)
    tvm_tirx.ir.assert_structural_equal(block.writes, writes)
    tvm_tirx.ir.assert_structural_equal([], opaque)


def test_match_tensor():
    root_block = match_tensor_func.body[0].block
    block = root_block.body[0].body[0].body[0].block
    block_inner = block.body[0].body[0].body[0].block
    alloc_tensors = match_tensor_func.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}

    # Check block
    ret = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(block.writes, ret[1])
    # B is opaque access
    tvm_tirx.ir.assert_structural_equal(block.reads, ret[2])

    # Check inner block AAA without updating tensor_var_map
    ret = s_tir.analysis.get_sblock_access_region(block_inner, tensor_var_map)
    # Since AA is not in the tensor_var_map, region of AA will not be collected.
    tvm_tirx.ir.assert_structural_equal([], ret[1])

    # Check inner block AAA
    for match_tensor in block.match_tensors:
        target_tensor = match_tensor.tensor
        tensor_var_map[target_tensor] = target_tensor

    ret = s_tir.analysis.get_sblock_access_region(block_inner, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(block_inner.reads, ret[0])
    tvm_tirx.ir.assert_structural_equal(block_inner.writes, ret[1])


def test_access_in_if_then_else_func():
    block = access_in_if_then_else_func.body[0].block.body[0].block
    alloc_tensors = access_in_if_then_else_func.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}
    ret0 = s_tir.analysis.get_sblock_read_write_region(block, tensor_var_map)
    ret1 = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(ret0[0], ret1[0])
    tvm_tirx.ir.assert_structural_equal(ret0[1], ret1[1])


def test_access_in_branch_func():
    block = access_in_branch_func.body[0].block.body[0].block
    alloc_tensors = access_in_branch_func.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}
    ret0 = s_tir.analysis.get_sblock_read_write_region(block, tensor_var_map)
    ret1 = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(ret0[0], ret1[0])
    tvm_tirx.ir.assert_structural_equal(ret0[1], ret1[1])


def test_access_of_padding_pattern():
    s = tvm.s_tir.schedule.Schedule(access_of_padding_pattern)
    alloc_tensors = s.get_sref(s.get_sblock("root")).stmt.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}

    def do_compare_tensor_region(region, expect):
        assert region.source == expect.source
        analyzer = tvm_tirx.sym.Analyzer()
        for observed_range, expected_range in zip(region.region, expect.region):
            analyzer.can_prove_equal(observed_range.begin, expected_range.begin)
            analyzer.can_prove_equal(
                observed_range.extent,
                expected_range.extent,
            )

    def do_check_block(block_name):
        block = s.get_sref(s.get_sblock(block_name)).stmt
        expect_reads = block.reads
        expect_writes = block.writes
        ret = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
        for i, read in enumerate(ret[0]):
            do_compare_tensor_region(read, expect_reads[i])
        for i, write in enumerate(ret[1]):
            do_compare_tensor_region(write, expect_writes[i])

    do_check_block("padding")
    do_check_block("padding_reverse")


def test_access_of_reduction():
    block = gemm.body[0].block.body[0].body[0].body[0].body[0].body[0].body[0].block
    alloc_tensors = gemm.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}
    ret = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(block.reads, ret[0])
    tvm_tirx.ir.assert_structural_equal(block.writes, ret[1])


def test_access_of_decompose_reduction():
    init = decomposed_gemm.body[0].block.body[0].body[0].body[0].body[0].body[0].block
    update = decomposed_gemm.body[0].block.body[0].body[0].body[1].body[0].body[0].body[0].block
    alloc_tensors = decomposed_gemm.body[0].block.alloc_tensors
    tensor_var_map = {buf: buf for buf in alloc_tensors}
    for block in [init, update]:
        ret = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
        tvm_tirx.ir.assert_structural_equal(block.reads, ret[0])
        tvm_tirx.ir.assert_structural_equal(block.writes, ret[1])


def test_tensor_access_with_let_binding():
    @Ts.function
    def func(
        storage: T.Tensor((16, 16, 16), "float32"),
        seq_slot_ids: T.Tensor((16,), "int32"),
        history_slot_ids: T.Tensor((16,), "int32"),
        output: T.Tensor((16, 16), "float32"),
    ):
        for i, s in T.grid(16, 16):
            with Ts.sblock("copy"):
                vi, vs = Ts.axis.remap("SS", [i, s])
                Ts.reads(
                    seq_slot_ids[vi],
                    history_slot_ids[vi],
                    storage[seq_slot_ids[vi], history_slot_ids[vi], vs],
                )
                Ts.writes(output[vi, vs])
                seq_id: T.let[T.int32] = seq_slot_ids[vi]
                history_id: T.let[T.int32] = history_slot_ids[vi]
                output[vi, vs] = storage[seq_id, history_id, vs]

    block = func.body[0].block.body[0].body[0].body[0].block
    tensor_var_map = {buf: buf for buf in func.params if tvm_tirx.tirx.is_tensor_var(buf)}
    ret = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(block.reads, ret[0])
    tvm_tirx.ir.assert_structural_equal(block.writes, ret[1])


def test_tensor_access_with_nested_let_binding():
    @Ts.function
    def func(
        A: T.Tensor((16, 16), "float32"),
        B: T.Tensor((16, 16), "float32"),
        C: T.Tensor((16, 16), "float32"),
    ):
        for i, s in T.grid(16, 16):
            with Ts.sblock("copy"):
                vi, vs = Ts.axis.remap("SS", [i, s])
                Ts.reads(A[vi, vs], B[vi, vs])
                Ts.writes(C[vi, vs])
                vi1: T.let[T.int32] = vi
                vi2: T.let[T.int32] = vi1
                vs1: T.let[T.int32] = vs
                vs2: T.let[T.int32] = vs1
                vs3: T.let[T.int32] = vs2
                C[vi, vs1] = A[vi1, vs2] + B[vi2, vs3]

    block = func.body[0].block.body[0].body[0].body[0].block
    tensor_var_map = {buf: buf for buf in func.params if tvm_tirx.tirx.is_tensor_var(buf)}
    ret = s_tir.analysis.get_sblock_access_region(block, tensor_var_map)
    tvm_tirx.ir.assert_structural_equal(block.reads, ret[0])
    tvm_tirx.ir.assert_structural_equal(block.writes, ret[1])


@pytest.mark.parametrize("case", ["coupled", "equal", "nonlinear", "empty", "unbounded", "rounded"])
def test_conditional_inequality_access_regions(case):
    # Retain the live cases from the former sym inequality solver tests through
    # the block-access consumer, including its conservative unresolved fallback.
    tirx = tvm_tirx.tirx
    x, y, z = [tirx.Var(name, "int32") for name in ("x", "y", "z")]
    free_var = tirx.Var("free_var", "int32")
    unbounded_extent = tirx.Cast("int32", 1 + tirx.log(free_var))
    cases = {
        "coupled": (
            [x, y],
            [(-100, 200), (0, 10)],
            tirx.all(x + y <= 20, x - y >= 10),
            [(-100, 200), (0, 10)],
        ),
        "equal": (
            [x, y],
            [(-100, 200), (-100, 200)],
            tirx.all(x + y >= 10, x - y >= 2, x <= 6),
            [(6, 1), (4, 1)],
        ),
        "nonlinear": (
            [x, y, z],
            [(-100, 200)] * 3,
            tirx.all(x <= 6, x >= 6, x - z * y >= 0, x - z * y <= 0),
            [(-100, 200)] * 3,
        ),
        "empty": (
            [x],
            [(-20, 41)],
            tirx.all(-x - 4 <= -5 * x + 2, x * 4 + 5 <= x * 5),
            [(-20, 41)],
        ),
        "unbounded": ([x], [(0, unbounded_extent)], x > 3, [(0, 256)]),
        "rounded": ([x], [(-20, 41)], tirx.all(x * 3 >= -7, x * 2 <= 9), [(-2, 7)]),
    }
    variables, domains, condition, expected = cases[case]
    inside = tirx.Var("inside", tirx.TensorType([256] * len(variables), "float32"))
    outside = tirx.Var("outside", tirx.TensorType([256] * len(variables), "float32"))
    body = tvm_tirx.ir.SeqStmt(
        [
            tvm_tirx.ir.If(condition, tvm_tirx.ir.Evaluate(inside[tuple(variables)]), None),
            tvm_tirx.ir.Evaluate(outside[tuple(variables)]),
        ]
    )
    for var, (minimum, extent) in reversed(list(zip(variables, domains))):
        body = tvm_tirx.ir.For(
            var,
            minimum,
            minimum + extent,
            tvm_tirx.ir.IntImm(var.ty, 1),
            tvm_tirx.ir.ForKind.DEFAULT,
            body,
        )
    block = s_tir.SBlock([], [], [], "conditional", body)
    # Unbounded access sets conservatively cover the whole tensor.
    outside_expected = [(0, 256)] if case == "unbounded" else domains
    reads, writes, opaque = s_tir.analysis.get_sblock_access_region(
        block, {inside: inside, outside: outside}
    )
    tvm_tirx.ir.assert_structural_equal(
        reads,
        [
            TensorRegion(
                inside,
                [Range.from_min_extent(*bounds) for bounds in expected],
                ty=TensorRegionType(),
            ),
            # Leaving the conditional scope must restore the original domains.
            TensorRegion(
                outside,
                [Range.from_min_extent(*bounds) for bounds in outside_expected],
                ty=TensorRegionType(),
            ),
        ],
    )
    assert not writes
    assert not opaque


if __name__ == "__main__":
    tvm_tirx.testing.main()
