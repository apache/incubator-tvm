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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
# ruff: noqa: F401
import pytest

import tvm
import tvm_tirx
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm_tirx.script.ir_module
class Module:
    @Ts.function
    def scale_by_two(a: T.Tensor((128,), "int8"), c: T.Tensor((128,), "int8")):
        for i in T.serial(128):
            with Ts.sblock("C"):
                c[i] = a[i] * T.int8(2)

    @Ts.function
    def scale_by_two_three(a: T.Tensor((128,), "int8"), c: T.Tensor((128,), "int8")):
        B = Ts.sblock_alloc_tensor([128], dtype="int8", scope="global.vtcm")
        for i in T.serial(128):
            with Ts.sblock("B"):
                B[i] = a[i] * T.int8(2)
        for i in T.serial(128):
            with Ts.sblock("C"):
                c[i] = B[i] * T.int8(3)

# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


@pytest.mark.parametrize(
    "function,size", [(Module["scale_by_two"], 128), (Module["scale_by_two_three"], 256)]
)
def test_scale_by(function, size):
    """Test calculate allocated bytes per scope"""
    mod = tvm_tirx.IRModule.from_expr(function.with_attr("global_symbol", "main"))
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    block_c = sch.get_sblock("C")
    (flat,) = sch.get_loops(block_c)
    cache_block = sch.cache_read(block_c, 0, storage_scope="global.vtcm")
    sch.compute_at(cache_block, flat)

    mod = sch.mod
    mod = tvm.s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    sizes = tvm.s_tir.analysis.calculate_allocated_bytes(mod["main"])
    assert "main" in sizes, 'Calls with Function is expected to return with function key as "main"'
    sizes = sizes["main"]
    assert sizes.get("global.vtcm", 0) == size


@Ts.function
def matmul_mix_scope(
    A: T.Tensor([128, 128], "float32", scope="global"),
    B: T.Tensor([128, 128], "float32", scope="global"),
    C: T.Tensor([128, 128], "float32", scope="global"),
) -> None:
    A_allocated = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="global.texture")
    B_allocated = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="global.texture")
    C_allocated = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="global")

    for i, j in T.grid(128, 128):
        with Ts.sblock("A.allocated"):
            A_allocated[i, j] = A[i, j]
    for i, j in T.grid(128, 128):
        with Ts.sblock("B.allocated"):
            B_allocated[i, j] = B[i, j]

    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C_allocated[vi, vj] = 0.0
            C_allocated[vi, vj] = C[vi, vj] + A_allocated[vi, vk] * B_allocated[vj, vk]

    for i, j in T.grid(128, 128):
        with Ts.sblock("C"):
            C[i, j] = C_allocated[i, j]


@pytest.mark.parametrize(
    "scope,size", [("global", 65536), ("global.texture", 131072), ("global.texture-nhwc", 0)]
)
def test_matmul_mix_scope(scope, size):
    """Test calculate allocated bytes per scope"""
    mod = tvm_tirx.IRModule({"main": matmul_mix_scope})
    mod = tvm.s_tir.transform.LowerInitBlock()(mod)
    mod = tvm.s_tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    sizes = tvm.s_tir.analysis.calculate_allocated_bytes(mod["main"])
    assert "main" in sizes, 'Calls with Function is expected to return with function key as "main"'
    sizes = sizes["main"]
    assert sizes.get(scope, 0) == size


def test_full_mod_calculator():
    def apply_schedule(sch, func_name):
        sch.work_on(func_name)
        block_c = sch.get_sblock("C")
        sch.cache_read(block_c, 0, storage_scope="global.vtcm")

    sch = tvm.s_tir.Schedule(Module, debug_mask="all")
    apply_schedule(sch, "scale_by_two")
    apply_schedule(sch, "scale_by_two_three")
    mod = tvm.s_tir.transform.ConvertBlocksToOpaque()(sch.mod)
    mod = tvm.s_tir.transform.LowerOpaqueBlock()(mod)
    sizes = tvm.s_tir.analysis.calculate_allocated_bytes(mod)
    assert "scale_by_two" in sizes, "Values for scale_by_two not found"
    scale_by_two_sizes = sizes["scale_by_two"]
    assert "global.vtcm" in scale_by_two_sizes, (
        "Expected global.vtcm allocation to be calculated scale_by_two"
    )
    assert scale_by_two_sizes["global.vtcm"] == 128, "Expected the calculated size to be 128"
    scale_by_two_three_sizes = sizes["scale_by_two_three"]
    assert "global.vtcm" in scale_by_two_three_sizes, (
        "Expected global.vtcm allocation to be calculated scale_by_two_three"
    )
    assert scale_by_two_three_sizes["global.vtcm"] == 256, "Expected the calculated size to be 256"


if __name__ == "__main__":
    tvm_tirx.testing.main()
