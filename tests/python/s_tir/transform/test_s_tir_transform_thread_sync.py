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
# ruff: noqa: F821, F841
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def run_passes(func: tvm_tirx.tirx.Function):
    mod = tvm_tirx.IRModule.from_expr(func)

    cuda_target = tvm_tirx.target.Target("cuda", host="llvm")

    mod = tvm_tirx.tirx.transform.Apply(
        lambda f: f.with_attr({"global_symbol": "test", "target": cuda_target})
    )(mod)

    mod = tvm_tirx.tirx.transform.SplitHostDevice()(mod)
    return tvm.s_tir.transform.ThreadSync("shared")(mod)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_sync_read_thread_id_independent_location():
    @Ts.function(check_well_formed=False)
    def func(p0_arg: T.Tensor((1, 2, 1, 1), "float32"), p1: T.Tensor(2, "float32")) -> None:
        p0 = T.Var("p0", T.Tensor([2], dtype="float32", data=p0_arg.data))
        result_local = Ts.sblock_alloc_tensor([1], dtype="float32", scope="local")
        temp_shared = Ts.sblock_alloc_tensor([1], dtype="float32", scope="shared")
        blockIdx_x = T.launch_thread("blockIdx.x", 8)
        threadIdx_x = T.launch_thread("threadIdx.x", 4)
        result_local[0] = T.float32(0)
        if threadIdx_x < 1:
            temp_shared[0] = p0[0]
        result_local[0] = result_local[0] + temp_shared[0] * p1[0]
        if threadIdx_x < 1:
            temp_shared[0] = p0[1]
        result_local[0] = result_local[0] + temp_shared[0] * p1[1]

    mod = run_passes(func)
    assert 'T.gpu_storage_sync("shared")' in str(mod)


def test_sync_inside_condition():
    @Ts.function
    def func1(A: T.Tensor((4, 4), "float32")) -> None:
        A_shared = T.alloc_tensor(ty_args=[T.Tensor((4, 4), "float32", scope="shared")])
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 32)
        if A[0, 0] > 1.0:
            for i, j in T.grid(4, 4):
                A_shared[i, j] = A[i, j]
            for i, j in T.grid(4, 4):
                A[i, j] = A_shared[i, j] + 1.0

    @Ts.function
    def func2(A: T.Tensor((4, 4), "float32")) -> None:
        A_shared = T.alloc_tensor(ty_args=[T.Tensor((4, 4), "float32", scope="shared")])
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 32)
        if T.gpu_thread_invariant(A[0, 0] > 1.0):
            for i, j in T.grid(4, 4):
                A_shared[i, j] = A[i, j]
            for i, j in T.grid(4, 4):
                A[i, j] = A_shared[i, j] + 1.0

    @Ts.function
    def func3(A: T.Tensor((4, 4), "float32")) -> None:
        A_shared = T.alloc_tensor(ty_args=[T.Tensor((4, 4), "float32", scope="shared")])
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 32)
        while T.gpu_thread_invariant(A[0, 0] > 1.0):
            for i, j in T.grid(4, 4):
                A_shared[i, j] = A[i, j]
            for i, j in T.grid(4, 4):
                A[i, j] = A_shared[i, j] + 1.0

    with pytest.raises(RuntimeError):
        s_tir.transform.ThreadSync("shared")(tvm_tirx.IRModule.from_expr(func1))

    for func in (func2, func3):
        mod = s_tir.transform.ThreadSync("shared")(tvm_tirx.IRModule.from_expr(func))
        assert 'T.gpu_storage_sync("shared")' in mod.script()


def test_sync_shared_dyn():
    @Ts.function(private=True)
    def func(A: T.Tensor((4, 4), "float32"), E: T.Tensor((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B = T.alloc_tensor(ty_args=[T.Tensor((24,), "float32", scope="shared.dyn")])
        C = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        D = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32", scope="shared.dyn")])
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((24,), "float32", scope="shared.dyn")])
        A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "float32", data=A.data)])
        B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1 = T.decl_tensor(C.data, ty_args=[T.Tensor((1,), "float32", scope="local")])
        C_1[0] = B_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1 = T.decl_tensor(D.data, ty_args=[T.Tensor((16,), "float32", scope="shared.dyn")])
        D_1[threadIdx_x] = C_1[0]
        E_1 = T.decl_tensor(E.data, ty_args=[T.Tensor((16,), "float32", data=E.data)])
        E_1[threadIdx_x] = D_1[threadIdx_x]

    @Ts.function(private=True)
    def expected(A: T.Tensor((4, 4), "float32"), E: T.Tensor((4, 4), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        B_1 = T.alloc_tensor(ty_args=[T.Tensor((24,), "float32", scope="shared.dyn")])
        C_1 = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        D_1 = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32", scope="shared.dyn")])
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        B_1_1 = T.decl_tensor(B_1.data, ty_args=[T.Tensor((24,), "float32", scope="shared.dyn")])
        A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), "float32", data=A.data)])
        B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4] = A_1[threadIdx_x]
        C_1_1 = T.decl_tensor(C_1.data, ty_args=[T.Tensor((1,), "float32", scope="local")])
        C_1_1[0] = B_1_1[threadIdx_x // 4 * 6 + threadIdx_x % 4]
        D_1_1 = T.decl_tensor(D_1.data, ty_args=[T.Tensor((16,), "float32", scope="shared.dyn")])
        T.evaluate(T.call_intrin("tirx.gpu_storage_sync", "shared.dyn", ty="int32"))
        D_1_1[threadIdx_x] = C_1_1[0]
        E_1 = T.decl_tensor(E.data, ty_args=[T.Tensor((16,), "float32", data=E.data)])
        E_1[threadIdx_x] = D_1_1[threadIdx_x]

    mod = tvm_tirx.IRModule({"main": func})
    mod = tvm.s_tir.transform.ThreadSync("shared.dyn")(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


def test_sync_shared_aliasing_tensor_views():
    @Ts.function(private=True)
    def func(A: T.Tensor((64,), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        shared_storage = T.alloc_tensor(ty_args=[T.Tensor((32,), "float16", scope="shared")])
        local = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        shared_half = T.decl_tensor(
            shared_storage.data, ty_args=[T.Tensor((32,), "float16", scope="shared")]
        )
        shared_float = T.decl_tensor(
            shared_storage.data, ty_args=[T.Tensor((16,), "float32", scope="shared")]
        )
        for i in range(2):
            shared_half[threadIdx_x] = T.Cast("float16", A[i * 32 + threadIdx_x])
            T.gpu_storage_sync("shared")
            local[0] = shared_float[threadIdx_x % 16]
            A[i * 32 + threadIdx_x] = local[0]

    mod = tvm_tirx.IRModule({"main": func})
    mod = tvm.s_tir.transform.ThreadSync("shared")(mod)

    # In addition to the explicit write-to-read barrier, the shared physical
    # storage needs a read-to-next-write barrier across loop iterations.
    assert str(mod["main"]).count("T.gpu_storage_sync") == 2


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_sync_bind():
    @Ts.function(private=True)
    def func(A: T.Tensor((16 * 512), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 16)
        A_shared = T.alloc_tensor(ty_args=[T.Tensor((512,), "float32", scope="shared")])
        in_thread_A_temp = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        cross_thread_A_temp = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        A_shared_1 = T.decl_tensor(
            A_shared.data, ty_args=[T.Tensor((512,), "float32", scope="shared")]
        )
        for ax0 in range(512):
            A_shared_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1 = T.decl_tensor(
            in_thread_A_temp.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
        )
        in_thread_A_temp_1[0] = T.float32(0)
        A_temp_1 = in_thread_A_temp_1[0] + A_shared_1[threadIdx_x]
        in_thread_A_temp_1[0] = A_temp_1
        A_temp_2 = in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 128]
        in_thread_A_temp_1[0] = A_temp_2
        A_temp_3 = in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 256]
        in_thread_A_temp_1[0] = A_temp_3
        A_temp_4 = in_thread_A_temp_1[0] + A_shared_1[threadIdx_x + 384]
        in_thread_A_temp_1[0] = A_temp_4
        cross_thread_A_temp_1 = T.decl_tensor(
            cross_thread_A_temp.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
        )
        T.gpu_thread_allreduce(
            T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
            (T.float32(0),),
            (in_thread_A_temp_1[0],),
            T.bool(True),
            (cross_thread_A_temp_1[0],),
            (threadIdx_x,),
        )

    @Ts.function(private=True)
    def expected(A: T.Tensor((8192,), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 16)
        A_shared_1 = T.alloc_tensor(ty_args=[T.Tensor((512,), "float32", scope="shared")])
        in_thread_A_temp_1 = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        cross_thread_A_temp_1 = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        A_shared_1_1 = T.decl_tensor(
            A_shared_1.data, ty_args=[T.Tensor((512,), "float32", scope="shared")]
        )
        for ax0 in range(512):
            A_shared_1_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1_1 = T.decl_tensor(
            in_thread_A_temp_1.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
        )
        in_thread_A_temp_1_1[0] = T.float32(0)
        T.evaluate(T.call_intrin("tirx.gpu_storage_sync", "shared", ty="int32"))
        A_temp_1 = in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x]
        in_thread_A_temp_1_1[0] = A_temp_1
        A_temp_2 = in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 128]
        in_thread_A_temp_1_1[0] = A_temp_2
        A_temp_3 = in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 256]
        in_thread_A_temp_1_1[0] = A_temp_3
        A_temp_4 = in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 384]
        in_thread_A_temp_1_1[0] = A_temp_4
        cross_thread_A_temp_1_1 = T.decl_tensor(
            cross_thread_A_temp_1.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
        )
        T.gpu_thread_allreduce(
            T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
            (T.float32(0),),
            (in_thread_A_temp_1_1[0],),
            T.bool(True),
            (cross_thread_A_temp_1_1[0],),
            (threadIdx_x,),
        )

    mod = tvm_tirx.IRModule({"main": func})
    mod = tvm.s_tir.transform.ThreadSync("shared")(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], expected)


@pytest.mark.parametrize("loop", [T.serial, T.unroll])
def test_hoist_if_preserves_sync_placement(loop):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((4,), "int32")):
            B = T.alloc_tensor(ty_args=[T.Tensor((4,), "int32", scope="shared")])
            tx = T.launch_thread("threadIdx.x", 4)
            B[tx] = A[tx]
            for i in loop(1, 4):
                if tx == 0:
                    B[i] = B[i - 1] + 1
            A[tx] = B[tx]

    # Each iteration must keep its own guard so synchronization can be inserted
    # between shared-memory accesses, outside conditional control flow.
    finish = tvm_tirx.ir.transform.Sequential(
        [tvm_tirx.tirx.transform.UnrollLoop(), tvm.s_tir.transform.ThreadSync("shared")]
    )
    Expected = finish(Before)
    After = finish(tvm_tirx.tirx.transform.HoistIf()(Before))
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    test_thread_storage_sync()
    test_sync_else_branch()
    test_sync_read_thread_id_independent_location()
    test_sync_shared_dyn()
    test_sync_bind()
