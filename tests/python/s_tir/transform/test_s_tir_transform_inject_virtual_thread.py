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
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _is_tensor_binding(node, *op_names):
    return (
        isinstance(node, tvm_tirx.ir.Bind)
        and isinstance(node.value, tvm_tirx.ir.Call)
        and isinstance(node.value.op, tvm_tirx.ir.Op)
        and node.value.op.name in op_names
    )


def test_vthread():
    """Test virtual thread injection with vthread"""
    n = 100
    m = 4
    nthread = 2

    @I.ir_module
    class Module:
        @Ts.function
        def main(A: T.handle("float32"), C: T.handle("float32")):
            A_buf = T.decl_tensor(A, ty_args=[T.Tensor((n * nthread,), "float32", data=A)])
            C_buf = T.decl_tensor(C, ty_args=[T.Tensor((n * nthread,), "float32", data=C)])
            for i in range(n):
                vt_x = T.launch_thread("vthread", nthread)
                vt_y = T.launch_thread("vthread", nthread)
                B = T.alloc_tensor(ty_args=[T.Tensor((m,), "float32", scope="shared")])
                B[i] = A_buf[i * nthread + vt_x]
                T.evaluate(T.call_extern("Run", B.ptr_to([0] * len(B.shape)), vt_x, ty="int32"))
                C_buf[i * nthread + vt_x] = B[i] + T.float32(1)

    # For vthread, expected allocation is m * nthread
    B_expected_alloc = m * nthread

    stmt = tvm.s_tir.transform.InjectVirtualThread()(Module)["main"]

    # Find allocate nodes
    allocates = []

    def find_allocates(node):
        if _is_tensor_binding(node, "tirx.alloc_tensor"):
            allocates.append(node)

    tvm_ffi.structural_walk(stmt.body, find_allocates)
    assert len(allocates) == 1
    assert list(allocates[0].var.ty.shape) == [B_expected_alloc]


def test_vthread_extern():
    """Test virtual thread injection with extern call"""
    n = 100
    m = 4
    nthread = 2

    @I.ir_module
    class Module:
        @Ts.function
        def main():
            T.func_attr({"global_symbol": "main"})
            for i in range(n):
                vt_x = T.launch_thread("vthread", nthread)
                vt_y = T.launch_thread("vthread", nthread)
                A = T.alloc_tensor(ty_args=[T.Tensor((m,), "float32", scope="shared")])
                B = T.alloc_tensor(ty_args=[T.Tensor((m,), "float32", scope="shared")])
                C = T.alloc_tensor(ty_args=[T.Tensor((m,), "float32", scope="shared")])
                A[vt_x] = T.Cast("float32", vt_x) + T.float32(1)
                B[vt_y] = T.Cast("float32", vt_y) + T.float32(1)
                T.evaluate(
                    T.call_extern(
                        "Run",
                        A.ptr_to([0] * len(A.shape)),
                        B.ptr_to([0] * len(B.shape)),
                        C.ptr_to([0] * len(C.shape)),
                        ty="int32",
                    )
                )

    # The opaque call may write through any pointer.  Without direction metadata,
    # all three tensors conservatively depend on both virtual thread axes.
    expected_alloc = m * nthread * nthread

    stmt = tvm.s_tir.transform.InjectVirtualThread()(Module)["main"]

    # Find allocate nodes
    allocates = []

    def find_allocates(node):
        if _is_tensor_binding(node, "tirx.alloc_tensor"):
            allocates.append(node)

    tvm_ffi.structural_walk(stmt.body, find_allocates)
    assert len(allocates) == 3
    # Check that we have the expected extents (order may vary)
    extents = sorted([int(a.var.ty.shape[0]) for a in allocates])
    assert extents == [expected_alloc] * 3


def test_vthread_if_then_else():
    """Test virtual thread injection with if-then-else"""
    nthread = 2

    @I.ir_module
    class Module:
        @Ts.function
        def main(A: T.handle("float32")):
            T.func_attr({"global_symbol": "main"})
            A_buf = T.decl_tensor(A, ty_args=[T.Tensor((100 * nthread,), "float32", data=A)])
            for i in range(100):
                vt = T.launch_thread("vthread", nthread)
                B = T.alloc_tensor(ty_args=[T.Tensor((128,), "float32", scope="shared")])
                if i == 0:
                    B[i] = A_buf[i * nthread + vt]
                else:
                    B[i] = A_buf[i * nthread + vt] + T.float32(1)
                if i == 0:
                    B[i] = A_buf[i * nthread + vt] + T.float32(2)

    stmt = tvm.s_tir.transform.InjectVirtualThread()(Module)["main"]

    # Find If nodes
    if_nodes = []

    def find_ifs(node):
        if isinstance(node, tvm_tirx.ir.If):
            if_nodes.append(node)

    tvm_ffi.structural_walk(stmt.body, find_ifs)

    assert len(if_nodes) == 2
    # First if has else_case, second does not
    assert if_nodes[0].else_case is not None
    assert if_nodes[1].else_case is None


def test_vthread_simplified():
    """Indices resulting from vthread injection should simplified

    This ensures that downstream passes that check for Ramp nodes do
    not need to each simplify the indices.
    """

    @Ts.function
    def before_func():
        vthread = T.launch_thread("vthread", 4)
        B = T.alloc_tensor(ty_args=[T.Tensor((4,), "int32", scope="shared")])
        B[T.ramp(0, 1, 4)] = T.broadcast(vthread, 4)

    @Ts.function
    def expected_func():
        B = T.alloc_tensor(ty_args=[T.Tensor((16,), "int32", scope="shared")])
        # The indices for B should each be a single Ramp node, and
        # should not be the sum of a Ramp and Broadcast node.
        B[T.ramp(T.Mul(0, 4), 1, 4)] = T.broadcast(0, 4)
        B[T.ramp(T.Mul(1, 4), 1, 4)] = T.broadcast(1, 4)
        B[T.ramp(T.Mul(2, 4), 1, 4)] = T.broadcast(2, 4)
        B[T.ramp(T.Mul(3, 4), 1, 4)] = T.broadcast(3, 4)

    before_mod = tvm_tirx.IRModule.from_expr(before_func.with_attr("global_symbol", "main"))
    after_mod = tvm.s_tir.transform.InjectVirtualThread()(before_mod)
    after_func = after_mod["main"]

    tvm_tirx.ir.assert_structural_equal(
        after_func, expected_func.with_attr("global_symbol", "main")
    )


def test_vthread_vectorized():
    """Use of vthread is compatible with vector allocations"""

    @Ts.function
    def before_func():
        vthread = T.launch_thread("vthread", 4)
        B = T.alloc_tensor(ty_args=[T.Tensor((4,), "int32", scope="shared")])
        B[T.ramp(0, 1, 4)] = T.broadcast(vthread, 4)

    before_mod = tvm_tirx.IRModule.from_expr(before_func.with_attr("global_symbol", "main"))
    intermediate_mod = tvm.s_tir.transform.InjectVirtualThread()(before_mod)
    after_mod = tvm_tirx.tirx.transform.StorageRewrite()(intermediate_mod)
    after_func = after_mod["main"]

    # Verify the vectorized allocation has the expected shape and dtype
    allocate_node = None

    def visitor(op):
        nonlocal allocate_node
        if _is_tensor_binding(op, "tirx.alloc_tensor") and "shared" in str(op.var.data.ty):
            allocate_node = op

    tvm_ffi.structural_walk(after_func.body, visitor)
    assert allocate_node is not None
    assert list(allocate_node.var.ty.shape) == [4]
    assert allocate_node.var.ty.dtype == "int32x4"


def test_vthread_rewrites_masked_accesses():
    @Ts.function
    def before_func():
        vthread = T.launch_thread("vthread", 2)
        B = T.alloc_tensor(ty_args=[T.Tensor((4,), "float32", scope="shared")])
        mask = T.meta_var(T.Broadcast(T.bool(True), 4))
        loaded = T.meta_var(T.masked_load(B, T.Ramp(0, 1, 4), mask, ty="float32x4"))
        value = T.meta_var(loaded + T.Broadcast(T.Cast("float32", vthread), 4))
        T.masked_store(B, value, T.Ramp(0, 1, 4), mask)

    after = tvm.s_tir.transform.InjectVirtualThread()(
        tvm_tirx.IRModule.from_expr(before_func.with_attr("global_symbol", "main"))
    )["main"]
    masked_calls = []

    def visitor(node):
        if isinstance(node, tvm_tirx.ir.Call) and node.op.name in {
            "tirx.masked_load",
            "tirx.masked_store",
        }:
            masked_calls.append(node)

    tvm_ffi.structural_walk(after.body, visitor)
    assert len(masked_calls) == 4
    assert all(list(call.args[0].ty.shape) == [8] for call in masked_calls)
    analyzer = tvm_tirx.sym.Analyzer()
    assert sorted(int(analyzer.simplify(call.args[-2].base)) for call in masked_calls) == [
        0,
        0,
        4,
        4,
    ]


if __name__ == "__main__":
    tvm_tirx.testing.main()
