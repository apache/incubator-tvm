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
from tvm.s_tir import script as Ts
from tvm_tirx.ir import TensorRegion, TensorRegionType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_tensor_region_bounds_are_visited():
    data = tvm_tirx.tirx.Var(
        "data", tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("int32"), scope="global")
    )
    tensor = tvm_tirx.tirx.Var("tensor", tvm_tirx.tirx.TensorType([4], "int32", data=data))
    undefined = tvm_tirx.tirx.Var("undefined", "int32")
    region = TensorRegion(
        tensor, [tvm_tirx.ir.Range.from_min_extent(undefined, 4)], ty=TensorRegionType()
    )
    block = tvm.s_tir.SBlock([], [region], [], "region", tvm_tirx.ir.Evaluate(0))
    func = tvm_tirx.tirx.Function([tensor], block)
    with pytest.raises(RuntimeError):
        tvm.s_tir.analysis.check_well_formed(func)


def test_fail_use_out_loop_var():
    @Ts.function(check_well_formed=False)
    def element_wise(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
    ):
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                # we cannot use `i` since it's defined outside the block
                B[vi, vj] = A[i, vj] * 2.0

    with pytest.raises(RuntimeError):
        tvm.s_tir.analysis.check_well_formed(element_wise)


def test_block_match_tensor_defines_tensor_obj():
    """In a block, Ts.match_tensor defines a tensor view"""

    @I.ir_module
    class mod:
        @Ts.function
        def func(A: T.Tensor([256, 256], "float32")):
            for (*iters,) in T.grid(16, 16, 16, 16):
                with Ts.sblock("compute"):
                    tile_i, tile_j, i, j = Ts.axis.remap("SSSS", iters)
                    B = Ts.match_tensor(
                        A[tile_i * 16 : (tile_i + 1) * 16, tile_j * 16 : (tile_j + 1) * 16],
                        dtype="float32",
                    )
                    B[i, j] = 0.0

    assert tvm.s_tir.analysis.check_well_formed(mod) is None


def test_block_match_tensor_defines_symbolic_variables():
    """In a block, Ts.match_tensor may define symbolic variables"""

    elem_offset = T.dynamic("elem_offset", "int32")

    @I.ir_module
    class mod:
        @Ts.function
        def func(A: T.Tensor([256, 256], "int32")):
            for (*iters,) in T.grid(16, 16, 16, 16):
                with Ts.sblock("compute"):
                    tile_i, tile_j, i, j = Ts.axis.remap("SSSS", iters)

                    B = Ts.match_tensor(
                        A[tile_i * 16 : (tile_i + 1) * 16, tile_j * 16 : (tile_j + 1) * 16],
                        dtype="float32",
                        layout=T.TileLayout(T.S[(16, 16) : (256, 1)] + elem_offset),
                    )

                    B[i, j] = elem_offset

    tvm.s_tir.analysis.check_well_formed(mod)


def test_match_tensor_in_block_is_well_formed():
    """SBlock::match_tensors introduces a tensor into scope for the block body."""

    @I.ir_module
    class mod:
        @Ts.function
        def func(A: T.Tensor((128, 128), "float32")):
            for (*iters,) in T.grid(8, 8, 16, 16):
                with Ts.sblock("compute"):
                    ti, tj, i, j = Ts.axis.remap("SSSS", iters)
                    A_tile = Ts.match_tensor(
                        A[ti * 16 : (ti + 1) * 16, tj * 16 : (tj + 1) * 16],
                        dtype="float32",
                    )
                    A_tile[i, j] = A_tile[i, j] * 2.0

    tvm.s_tir.analysis.check_well_formed(mod)


def test_error_undeclared_tensor_in_schedulable_tir():
    """In schedule-level TIR (with SBlock nodes), all tensors must be declared."""
    # Manually construct a TensorStore that uses a tensor without any declaration
    # inside a block context.
    n = tvm_tirx.tirx.Var("n", "int32")
    A = tvm_tirx.tirx.Var("A", tvm_tirx.tirx.TensorType([n], "float32"))
    i = tvm_tirx.tirx.Var("i", "int32")

    # Create an undeclared tensor using an explicit data pointer that is NOT
    # a function parameter and NOT wrapped with DeclTensor.
    B_data = tvm_tirx.tirx.Var("B_data", tvm_tirx.ir.PtrType(tvm_tirx.ir.PrimType("float32")))
    B = tvm_tirx.tirx.Var("B", tvm_tirx.tirx.TensorType([n], "float32", data=B_data))

    # Build a block that writes to B without any declaration of B.
    bi = tvm_tirx.tirx.Var("bi", "int32")
    block = tvm.s_tir.SBlock(
        iter_vars=[tvm.s_tir.IterVar(tvm_tirx.ir.Range(0, n), bi, 0)],  # 0 = kDataPar
        reads=[TensorRegion(A, [tvm_tirx.ir.Range(bi, bi + 1)], ty=TensorRegionType())],
        writes=[TensorRegion(B, [tvm_tirx.ir.Range(bi, bi + 1)], ty=TensorRegionType())],
        body=tvm_tirx.ir.TensorStore(B, [bi], tvm_tirx.ir.TensorLoad(A, [bi])),
        name_hint="write_B",
    )
    block_realize = tvm.s_tir.SBlockRealize(
        iter_values=[i],
        predicate=tvm_tirx.tirx.const(True, "bool"),
        block=block,
    )

    function = tvm_tirx.tirx.Function(
        params=[A, B_data],
        body=tvm_tirx.ir.For(
            i, 0, n, tvm_tirx.ir.IntImm(i.ty, 1), tvm_tirx.ir.ForKind.DEFAULT, block_realize
        ),
        # Note: B is NOT a function parameter, so its declaration scope is only
        # within a DeclTensor node (which we intentionally omit here).
    )

    # B is used in the block but was never declared — should fail.
    with pytest.raises((ValueError, RuntimeError), match="tensor B.*without a prior DeclTensor"):
        tvm.s_tir.analysis.check_well_formed(function)
