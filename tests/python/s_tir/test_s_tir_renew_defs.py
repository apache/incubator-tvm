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

import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import SBlock, match_shape_stride_pattern
from tvm.s_tir import script as Ts
from tvm_tirx.ir import Var
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.function import Function


def _check_func_signature_remap(lhs: Function, rhs: Function):
    assert lhs != rhs
    for x, y in zip(lhs.params, rhs.params):
        assert x != y
        assert tvm_tirx.tirx.is_tensor_var(x) == tvm_tirx.tirx.is_tensor_var(y)


def _check_tensor_decl(lhs: Var, rhs: Var):
    assert lhs != rhs
    assert lhs.data != rhs.data


def _check_block_signature_remap(lhs: SBlock, rhs: SBlock):
    assert lhs != rhs
    for x, y in zip(lhs.iter_vars, rhs.iter_vars):
        assert x != y
        assert x.var != y.var
    for x, y in zip(lhs.alloc_tensors, rhs.alloc_tensors):
        _check_tensor_decl(x, y)
    for x, y in zip(lhs.match_tensors, rhs.match_tensors):
        assert x != y
        _check_tensor_decl(x.tensor, y.tensor)


def test_simple():
    @Ts.function
    # Var A should be remapped
    def elementwise(A: T.Tensor((128, 128), "float32")):
        # Var B should be remapped
        B = Ts.sblock_alloc_tensor((128, 128), "float32")
        # i, j should be remapped
        for i, j in T.grid(128, 128):
            with Ts.sblock("B"):
                # vi, vj should be remapped
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi, vj])
                Ts.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj] * 2.0

    f1 = elementwise
    f2 = tvm_tirx.tirx.renew_def(f1)
    tvm_tirx.ir.assert_structural_equal(f1, f2)

    _check_func_signature_remap(f1, f2)
    # check root block
    _check_block_signature_remap(f1.body[0].block, f2.body[0].block)
    # check remap of i
    assert f1.body[0].block.body[0].var != f2.body[0].block.body[0].var
    # check remap of j
    assert f1.body[0].block.body[0].body[0].var != f2.body[0].block.body[0].body[0].var

    # check inner block
    def _get_sblock(f):
        return f.body[0].block.body[0].body[0].body[0].block

    _check_block_signature_remap(_get_sblock(f1), _get_sblock(f2))


def test_match_tensor():
    # well-formed checker complains about multiple definitions for variable A0_s1,
    # likely stemming from strides=[s, s]
    s = T.dynamic("s", "int32")
    e = T.dynamic("e", "int32")

    @Ts.function(check_well_formed=False)
    # A and B should be remapped
    def func_match_tensor(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        with Ts.sblock("root"):
            # A0 should be remapped
            A0 = Ts.match_tensor(
                A[0:128, 0:128],
                shape=(128, 128),
                dtype="float32",
                # s and e should be remapped
                layout=T.TileLayout(T.S[(128, 128) : [s, s]] + (e)),
            )
            for i, j in T.grid(128, 128):
                with Ts.sblock("B"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A0[vi, vj] * 2.0

    f1 = func_match_tensor
    f2 = tvm_tirx.tirx.renew_def(f1)
    tvm_tirx.ir.assert_structural_equal(f1, f2)

    _check_func_signature_remap(f1, f2)
    _check_block_signature_remap(f1.body[0].block, f2.body[0].block)
    assert f1.body[0].block.body[0].var != f2.body[0].block.body[0].var

    def _get_sblock(f):
        return f.body[0].block

    block1 = _get_sblock(f1)
    block2 = _get_sblock(f2)
    _check_block_signature_remap(block1, block2)

    matched_tensor1 = block1.match_tensors[0].tensor
    matched_tensor2 = block2.match_tensors[0].tensor
    # Stride var s should be remapped
    matched_tensor1_pattern = match_shape_stride_pattern(matched_tensor1.ty)
    matched_tensor2_pattern = match_shape_stride_pattern(matched_tensor2.ty)
    assert matched_tensor1_pattern.strides[0] != matched_tensor2_pattern.strides[0]
    assert matched_tensor1_pattern.strides[1] != matched_tensor2_pattern.strides[1]
    # s should be only remapped once
    assert matched_tensor1_pattern.strides[0] == matched_tensor1_pattern.strides[1]
    assert matched_tensor2_pattern.strides[0] == matched_tensor2_pattern.strides[1]
    # Element-offset var e should be remapped
    assert matched_tensor1_pattern.elem_offset != matched_tensor2_pattern.elem_offset


def test_tensor_params():
    m = T.dynamic("m")

    @Ts.function
    def main(A: T.Tensor((m * 2,), "float32"), B: T.Tensor((m, 2), "float32")):
        for i, j in T.grid(m, 2):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi * 2 + vj]

    f1 = main
    f2 = tvm_tirx.tirx.renew_def(main)
    tvm_tirx.ir.assert_structural_equal(f1, f2)
    assert f1.params[1].shape[0] != f2.params[1].shape[0]


def test_gather():
    @Ts.function(private=True)
    def take(
        A: T.Tensor((4096, 4096), "float16"),
        B: T.Tensor((1,), "int32"),
        T_take: T.Tensor((1, 4096), "float16"),
    ):
        for ax0, ax1 in T.grid(1, 4096):
            with Ts.sblock("T_take"):
                v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(A[B[v_ax0], v_ax1], B[v_ax0])
                Ts.writes(T_take[v_ax0, v_ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    f1 = take
    f2 = tvm_tirx.tirx.renew_def(take)
    tvm_tirx.ir.assert_structural_equal(f1, f2)


if __name__ == "__main__":
    tvm_tirx.testing.main()
