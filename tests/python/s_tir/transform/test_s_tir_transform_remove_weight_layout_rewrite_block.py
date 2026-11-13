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

import tvm
import tvm_tirx
from tvm.s_tir import script as Ts
from tvm_tirx.ir.module import IRModule
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.function import Function


def _check(before, expect):
    if isinstance(before, Function):
        before = IRModule({"main": before.with_attr("global_symbol", "main")})
    if isinstance(expect, Function):
        expect = IRModule({"main": expect.with_attr("global_symbol", "main")})

    mod = tvm.s_tir.transform.RemoveWeightLayoutRewriteBlock()(before)
    tvm_tirx.ir.assert_structural_equal(mod, expect)


def test_matmul():
    @Ts.function
    def before(
        A: T.Tensor((16, 16), "float32"),
        B: T.Tensor((16, 16), "float32"),
        C: T.Tensor((16, 16), "float32"),
    ) -> None:
        T.func_attr({"layout_free_tensors": [1]})
        B_ = Ts.sblock_alloc_tensor([16, 4, 4], dtype="float32")
        for i0_o, i1_o in T.grid(16, 16):
            with Ts.sblock("layout_rewrite"):
                i0, i1 = Ts.axis.remap("SS", [i0_o, i1_o])
                Ts.reads(B[i0, i1])
                Ts.writes(B_[i1, i0 // 4, i0 % 4])
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc": True})
                B_[i1, i0 // 4, i0 % 4] = B[i0, i1]
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with Ts.sblock("matmul"):
                vi = Ts.axis.spatial(16, i0 * 4 + i1)
                vj = Ts.axis.spatial(16, j)
                vk = Ts.axis.reduce(16, k0 * 4 + k1)
                Ts.reads(A[vi, vk], B_[vj, vk // 4, vk % 4])
                Ts.writes(C[vi, vj])
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B_[vj, vk // 4, vk % 4]

    @Ts.function
    def after(
        A: T.Tensor((16, 16), "float32"),
        B: T.Tensor((16, 4, 4), "float32"),
        C: T.Tensor((16, 16), "float32"),
    ) -> None:
        T.func_attr({"layout_free_tensors": [1]})
        for i0_o, i1_o in T.grid(16, 16):
            with Ts.sblock("layout_rewrite"):
                i0, i1 = Ts.axis.remap("SS", [i0_o, i1_o])
                Ts.reads()
                Ts.writes()
                Ts.sblock_attr({"meta_schedule.layout_rewrite_preproc": True})
                T.evaluate(0)
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with Ts.sblock("matmul"):
                vi = Ts.axis.spatial(16, i0 * 4 + i1)
                vj = Ts.axis.spatial(16, j)
                vk = Ts.axis.reduce(16, k0 * 4 + k1)
                Ts.reads(A[vi, vk], B[vj, vk // 4, vk % 4])
                Ts.writes(C[vi, vj])
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk // 4, vk % 4]

    _check(before, after)


if __name__ == "__main__":
    test_matmul()
