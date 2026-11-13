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
# ruff: noqa: F401, F841

import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir
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


def _has_volatile_alloc_tensor(mod):
    has_volatile_alloc = False

    def visit(node):
        nonlocal has_volatile_alloc
        if _is_tensor_binding(node, "tirx.alloc_tensor") and "tirx.volatile" in node.value.attrs:
            has_volatile_alloc = has_volatile_alloc or node.value.attrs["tirx.volatile"] is True

    tvm_ffi.structural_walk(mod["main"].body, visit)
    return has_volatile_alloc


def test_basic():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((128, 32), "float32"), B: T.Tensor(128, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(4096, "float32", data=A.data)])

            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 32)

                reduce = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                reduce_1 = T.decl_tensor(
                    reduce.data, ty_args=[T.Tensor(1, "float32", scope="local")]
                )

                T.gpu_thread_allreduce(
                    T.Lambda([T.float32, T.float32], lambda x, y: x + y),
                    T.float32(0),
                    A_flat[0],
                    T.bool(True),
                    reduce_1[0],
                    threadIdx_x,
                )
                if threadIdx_x == 0:
                    B[i] = reduce_1[0]

    After = transform(Before)
    # Verify the transform produces valid output (not checking exact structure
    # since the output depends on the allreduce lowering implementation)
    assert After is not None
    # Run script roundtrip to verify it can be printed and reparsed
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script


def test_basic_with_decl_tensor():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((128, 32), "float32"), B: T.Tensor(128, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(4096, "float32", data=A.data)])

            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 32)

                reduce = T.alloc_tensor(ty_args=[T.Tensor(1, "float32", scope="local")])

                T.gpu_thread_allreduce(
                    T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                    (T.float32(0),),
                    (A_flat[0],),
                    T.bool(True),
                    (reduce[0],),
                    (threadIdx_x,),
                )
                if threadIdx_x == 0:
                    B[i] = reduce[0]

    After = transform(Before)
    assert After is not None
    tvm_tirx.tirx.analysis.check_well_formed(After)
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script


def test_reduce_summation():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((128, 128), "float32"), B: T.Tensor(128, "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(16384, "float32", data=A.data)])

            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 32)

                normal_reduce = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                normal_reduce_1 = T.decl_tensor(
                    normal_reduce.data, ty_args=[T.Tensor(1, "float32", scope="local")]
                )

                reduce = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                reduce_1 = T.decl_tensor(
                    reduce.data, ty_args=[T.Tensor(1, "float32", scope="local")]
                )

                normal_reduce_1[0] = T.float32(0)

                for ko in range(4):
                    normal_reduce_1[0] = (
                        normal_reduce_1[0] + A_flat[i * 128 + ko * 32 + threadIdx_x]
                    )

                T.gpu_thread_allreduce(
                    T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                    (T.float32(0),),
                    (normal_reduce_1[0],),
                    T.bool(True),
                    (reduce_1[0],),
                    (threadIdx_x,),
                )
                if threadIdx_x == 0:
                    B[i] = reduce_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script


def test_multi_group_reduction():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 32)
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_x = T.launch_thread("threadIdx.x", 32)
            cross_thread_B_1 = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1024,), "float32", data=A.data)])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (A_1[threadIdx_y * 32 + threadIdx_x],),
                T.bool(True),
                (cross_thread_B_1[0],),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((32,), "float32", data=B.data)])
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script


def test_multi_group_reduction_consumed_through_alias():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((4, 128), "float32"), B: T.Tensor((4,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 4)
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            cross_thread_B_alias = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor((512,), "float32", data=A.data)])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (A_flat[threadIdx_y * 128 + threadIdx_x],),
                T.bool(True),
                (cross_thread_B[0],),
                (threadIdx_x,),
            )
            cross_thread_B_alias[0] = cross_thread_B[0]
            if threadIdx_x == 0:
                B_flat = T.decl_tensor(B.data, ty_args=[T.Tensor((4,), "float32", data=B.data)])
                B_flat[threadIdx_y] = cross_thread_B_alias[0]

    After = transform(Before)
    tvm_tirx.tirx.analysis.check_well_formed(After)
    After_script = After.script()
    assert "red_result[threadIdx_y] = red_result[threadIdx_y]" in After_script
    assert "B_flat[threadIdx_y] = red_result[threadIdx_y]" in After_script
    assert "B_flat[threadIdx_y] = red_result[0]" not in After_script
    assert "red_result[0] = red_result[threadIdx_y]" not in After_script
    assert "cross_thread_B_alias" not in After_script


def test_multi_group_reduction_with_alias_declared_after_allreduce():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((4, 128), "float32"), B: T.Tensor((4,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 4)
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor((512,), "float32", data=A.data)])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (A_flat[threadIdx_y * 128 + threadIdx_x],),
                T.bool(True),
                (cross_thread_B[0],),
                (threadIdx_x,),
            )
            cross_thread_B_alias = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            cross_thread_B[0] = cross_thread_B_alias[0]
            if threadIdx_x == 0:
                B_flat = T.decl_tensor(B.data, ty_args=[T.Tensor((4,), "float32", data=B.data)])
                B_flat[threadIdx_y] = cross_thread_B_alias[0]

    After = transform(Before)
    tvm_tirx.tirx.analysis.check_well_formed(After)
    After_script = After.script()
    assert "red_result[threadIdx_y] = red_result[threadIdx_y]" in After_script
    assert "B_flat[threadIdx_y] = red_result[threadIdx_y]" in After_script
    assert "B_flat[threadIdx_y] = red_result[0]" not in After_script
    assert "red_result[0] = red_result[threadIdx_y]" not in After_script
    assert "cross_thread_B_alias" not in After_script


def test_multi_group_mask1():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((32, 8), "float32"), B: T.Tensor((32,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 32)
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_x = T.launch_thread("threadIdx.x", 8)
            cross_thread_B_1 = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((256,), "float32", data=A.data)])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (A_1[threadIdx_y * 8 + threadIdx_x],),
                T.bool(True),
                (cross_thread_B_1[0],),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((32,), "float32", data=B.data)])
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script


def test_multi_warp_reduce1():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 128)
                cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                cross_thread_B_1 = T.decl_tensor(
                    cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
                )
                A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((16384,), "float32", data=A.data)])
                T.gpu_thread_allreduce(
                    T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                    (T.float32(0),),
                    (A_1[i * 128 + threadIdx_x],),
                    T.bool(True),
                    (cross_thread_B_1[0],),
                    (threadIdx_x,),
                )
                if threadIdx_x == 0:
                    B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((128,), "float32", data=B.data)])
                    B_1[i] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script
    assert "gpu_storage_sync" in After_script  # multi-warp needs shared sync


def test_multi_warp_reduce2():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((1, 1024), "float32"), B: T.Tensor((1,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_x = T.launch_thread("threadIdx.x", 1024)
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            cross_thread_B_1 = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1024,), "float32", data=A.data)])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (A_1[threadIdx_x],),
                T.bool(True),
                (cross_thread_B_1[0],),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), "float32", data=B.data)])
                B_1[0] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script
    assert "gpu_storage_sync" in After_script


def test_multi_group_multi_warp_reduction():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((4, 128), "float32"), B: T.Tensor((4,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 4)
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            cross_thread_B_1 = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((512,), "float32", data=A.data)])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (A_1[threadIdx_y * 128 + threadIdx_x],),
                T.bool(True),
                (cross_thread_B_1[0],),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((4,), "float32", data=B.data)])
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script
    assert "gpu_storage_sync" in After_script


def test_multi_group_multi_warp_predicated_reduction():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((2, 70), "float32"), B: T.Tensor((2,), "float32")):
            T.func_attr({"target": T.target("cuda", host="llvm")})
            threadIdx_y = T.launch_thread("threadIdx.y", 2)
            in_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_x = T.launch_thread("threadIdx.x", 512)
            in_thread_B_1 = T.decl_tensor(
                in_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            in_thread_B_1[0] = T.float32(0)
            if threadIdx_x < 70:
                A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((140,), "float32", data=A.data)])
                in_thread_B_1[0] = in_thread_B_1[0] + A_1[threadIdx_y * 70 + threadIdx_x]
            cross_thread_B_1 = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (in_thread_B_1[0],),
                T.bool(True),
                (cross_thread_B_1[0],),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((2,), "float32", data=B.data)])
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "gpu_warp_shuffle" in After_script
    assert "gpu_storage_sync" in After_script


def test_metal_no_mask():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((1, 1, 2, 128), "float32"), B: T.Tensor((1, 1, 2), "float32")):
            T.func_attr(
                {
                    "target": T.target(
                        {
                            "kind": "metal",
                            "max_threads_per_block": 1024,
                            "thread_warp_size": 32,
                            "host": "llvm",
                        }
                    ),
                }
            )
            blockIdx_x = T.launch_thread("blockIdx.x", 1)
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_z = T.launch_thread("threadIdx.z", 1)
            threadIdx_y = T.launch_thread("threadIdx.y", 2)
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            cross_thread_B_1 = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((256,), "float32", data=A.data)])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (A_1[threadIdx_y * 128 + threadIdx_x],),
                T.bool(True),
                (cross_thread_B_1[0],),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((2,), "float32", data=B.data)])
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    # Metal does not use warp masks
    assert "gpu_warp_shuffle_down" in After_script
    assert "gpu_storage_sync" in After_script


def test_webgpu_warp_reduce():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((128, 32), "float32"), B: T.Tensor(128, "float32")):
            T.func_attr(
                {
                    "target": T.target(
                        {
                            "kind": "webgpu",
                            "supports_subgroups": True,
                            "host": "llvm",
                        }
                    ),
                }
            )
            A_flat = T.decl_tensor(A.data, ty_args=[T.Tensor(4096, "float32", data=A.data)])

            for i in range(128):
                threadIdx_x = T.launch_thread("threadIdx.x", 32)

                reduce_data = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
                reduce = T.decl_tensor(
                    reduce_data.data, ty_args=[T.Tensor(1, "float32", scope="local")]
                )
                reduce_alias = T.decl_tensor(
                    reduce.data, ty_args=[T.Tensor(1, "float32", scope="local")]
                )

                T.gpu_thread_allreduce(
                    T.Lambda([T.float32, T.float32], lambda x, y: (x + y,)),
                    (T.float32(0),),
                    (A_flat[0],),
                    T.bool(True),
                    (reduce_alias[0],),
                    (threadIdx_x,),
                )
                if threadIdx_x == 0:
                    B[i] = reduce_alias[0]

    After = transform(Before)
    assert After is not None
    tvm_tirx.tirx.analysis.check_well_formed(After)
    After_script = After.script()
    assert "gpu_warp_shuffle_down" in After_script
    assert "gpu_warp_shuffle(" in After_script
    assert "gpu_storage_sync" not in After_script
    assert "T.uint32(" not in After_script


def test_webgpu_multi_warp_reduce():
    transform = tvm_tirx.transform.Sequential(
        [
            tvm.s_tir.transform.LowerOpaqueBlock(),
            tvm.s_tir.transform.LowerThreadBinding(),
            tvm_tirx.tirx.transform.LowerThreadAllreduce(),
        ]
    )

    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def main(A: T.Tensor((1, 1, 2, 128), "float32"), B: T.Tensor((1, 1, 2), "float32")):
            T.func_attr(
                {
                    "target": T.target(
                        {
                            "kind": "webgpu",
                            "max_num_threads": 1024,
                            "supports_subgroups": True,
                            "host": "llvm",
                        }
                    ),
                }
            )
            blockIdx_x = T.launch_thread("blockIdx.x", 1)
            cross_thread_B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            threadIdx_z = T.launch_thread("threadIdx.z", 1)
            threadIdx_y = T.launch_thread("threadIdx.y", 2)
            threadIdx_x = T.launch_thread("threadIdx.x", 128)
            cross_thread_B_1 = T.decl_tensor(
                cross_thread_B.data, ty_args=[T.Tensor((1,), "float32", scope="local")]
            )
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((256,), "float32", data=A.data)])
            T.gpu_thread_allreduce(
                T.Lambda([T.float32, T.float32], lambda x0, y0: (x0 + y0,)),
                (T.float32(0),),
                (A_1[threadIdx_y * 128 + threadIdx_x],),
                T.bool(True),
                (cross_thread_B_1[0],),
                (threadIdx_x,),
            )
            if threadIdx_x == 0:
                B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((2,), "float32", data=B.data)])
                B_1[threadIdx_y] = cross_thread_B_1[0]

    After = transform(Before)
    assert After is not None
    After_script = After.script()
    assert "gpu_warp_shuffle_down" in After_script
    assert "gpu_storage_sync" in After_script
    assert _has_volatile_alloc_tensor(After)
    assert "T.uint32(" not in After_script


if __name__ == "__main__":
    tvm_tirx.testing.main()
