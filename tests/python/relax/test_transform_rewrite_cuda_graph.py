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
# ruff: noqa: E501, F841

import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


@pytest.fixture(autouse=True)
def enable_cuda_graph():
    """Enable cuda graph transform for all tests in this file"""
    with tvm_tirx.ir.transform.PassContext(config={"relax.backend.use_cuda_graph": True}):
        yield


def test_rewrite_cuda_graph():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tirx.noalias": True, "global_symbol": "exp"})
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("compute"):
                        i0 = Ts.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = Ts.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2,4), dtype="float32"):
            # force_pure is expected because purity checking should be disabled before this pass
            R.func_attr({"relax.force_pure": True})
            cls = Before
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _1: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            storage1: R.Any = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([2, 4]), "float32")
            _2: R.Tuple = R.call_tir_packed(cls.exp, (alloc, alloc1))
            _3: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _4: R.Tuple = R.call_tir_packed(cls.exp, (alloc1, alloc2))
            _5: R.Tuple = R.memory.kill_tensor(alloc1)
            storage2: R.Any = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc3: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage2, 0, R.shape([2, 4]), "float32")
            _6: R.Tuple = R.call_tir_packed(cls.exp, (alloc2, alloc3))
            _7: R.Tuple = R.memory.kill_tensor(alloc2)
            alloc4: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), "float32", 0)
            _8 = R.call_tir_packed(cls.exp, (alloc3, alloc4))
            _9: R.Tuple = R.memory.kill_tensor(alloc3)
            _10: R.Tuple = R.memory.kill_storage(storage)
            _11: R.Tuple = R.memory.kill_storage(storage1)
            _12: R.Tuple = R.memory.kill_storage(storage2)
            return alloc4

    @I.ir_module
    class Expected:
        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tirx.noalias": True, "global_symbol": "exp"})
            # body
            # with Ts.sblock("root")
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("compute"):
                        i0 = Ts.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = Ts.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        Ts.reads(rxplaceholder[i0, i1])
                        Ts.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            storage1: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            storage2: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv: R.Tuple(R.Any, R.Any, R.Any) = (storage, storage1, storage2)
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(alloc: R.Tensor((2, 4), dtype="float32"), alloc1: R.Tensor((2, 4), dtype="float32"), storage: R.Any, storage2: R.Any) -> R.Tuple(R.Tensor((2, 4), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            _2: R.Tuple = R.call_tir_packed(cls.exp, (alloc, alloc1))
            _3: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _4: R.Tuple = R.call_tir_packed(cls.exp, (alloc1, alloc2))
            _5: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage2, 0, R.shape([2, 4]), "float32")
            _6: R.Tuple = R.call_tir_packed(cls.exp, (alloc2, alloc3))
            _7: R.Tuple = R.memory.kill_tensor(alloc2)
            gv: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (alloc3,)
            return gv

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2,4), dtype="float32"):
            # this comes after RemovePurityChecking, so we expect purity to be forced
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Any, R.Any, R.Any) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.get_cached_alloc", (cls.cuda_graph_alloc, R.prim_value(0)), ty_args=(R.Tuple(R.Any, R.Any, R.Any),))
            storage: R.Any = gv[0]
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            storage1: R.Any = gv[1]
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            storage2: R.Any = gv[2]
            gv1: R.Tuple(R.Tensor((2, 4), dtype="float32")) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (cls.main_cuda_graph_capture, (alloc, alloc1, storage, storage2), R.prim_value(0)), ty_args=(R.Tuple(R.Tensor((2, 4), dtype="float32")),))
            alloc3: R.Tensor((2, 4), dtype="float32") = gv1[0]
            alloc4: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _6: R.Tuple = R.call_tir_packed(cls.exp, (alloc3, alloc4))
            _7: R.Tuple = R.memory.kill_tensor(alloc3)
            _8: R.Tuple = R.memory.kill_storage(storage)
            _9: R.Tuple = R.memory.kill_storage(storage1)
            _10: R.Tuple = R.memory.kill_storage(storage2)
            return alloc4
    # fmt: on

    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_tuple():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tirx.noalias": True, "global_symbol": "exp"})
            # body
            # with Ts.sblock("root")
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("compute"):
                        i0 = Ts.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = Ts.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        Ts.reads(rxplaceholder[i0, i1])
                        Ts.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Before
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            storage1: R.Any = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([2, 4]), "float32")
            _: R.Tuple = R.call_tir_packed(cls.exp, (alloc, alloc1))
            lv0 = (alloc1,)
            lv1 = (lv0,)
            lv2 = lv1[0]
            lv3 = lv2[0]
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _1: R.Tuple = R.call_tir_packed(cls.exp, (lv3, alloc2))
            _2: R.Tuple = R.memory.kill_tensor(alloc)
            _3: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _4: R.Tuple = R.call_tir_packed(cls.exp, (alloc2, alloc3))
            _5: R.Tuple = R.memory.kill_tensor(alloc2)
            _6: R.Tuple = R.memory.kill_storage(storage)
            _7: R.Tuple = R.memory.kill_storage(storage1)
            return alloc3

    @I.ir_module
    class Expected:
        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            T.func_attr({"global_symbol": "exp", "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("compute"):
                        i0 = Ts.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = Ts.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        Ts.reads(rxplaceholder[i0, i1])
                        Ts.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            storage1: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv: R.Tuple(R.Any, R.Any) = (storage, storage1)
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(alloc: R.Tensor((2, 4), dtype="float32"), alloc1: R.Tensor((2, 4), dtype="float32"), storage: R.Any) -> R.Tuple(R.Tensor((2, 4), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            _: R.Tuple = R.call_tir_packed(cls.exp, (alloc, alloc1))
            lv0: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (alloc1,)
            lv1: R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"))) = (lv0,)
            lv2: R.Tuple(R.Tensor((2, 4), dtype="float32")) = lv1[0]
            lv3: R.Tensor((2, 4), dtype="float32") = lv2[0]
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.exp, (lv3, alloc2))
            _2: R.Tuple = R.memory.kill_tensor(alloc)
            _3: R.Tuple = R.memory.kill_tensor(alloc1)
            gv: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (alloc2,)
            return gv

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Any, R.Any) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.get_cached_alloc", (cls.cuda_graph_alloc, R.prim_value(0)), ty_args=(R.Tuple(R.Any, R.Any),))
            storage: R.Any = gv[0]
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            storage1: R.Any = gv[1]
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            gv1: R.Tuple(R.Tensor((2, 4), dtype="float32")) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (cls.main_cuda_graph_capture, (alloc, alloc1, storage), R.prim_value(0)), ty_args=(R.Tuple(R.Tensor((2, 4), dtype="float32")),))
            alloc2: R.Tensor((2, 4), dtype="float32") = gv1[0]
            alloc3: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _4: R.Tuple = R.call_tir_packed(cls.exp, (alloc2, alloc3))
            _5: R.Tuple = R.memory.kill_tensor(alloc2)
            _6: R.Tuple = R.memory.kill_storage(storage)
            _7: R.Tuple = R.memory.kill_storage(storage1)
            return alloc3
    # fmt: on

    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_vm_builtin():
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tirx.noalias": True, "global_symbol": "exp"})
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("compute"):
                        i0 = Ts.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = Ts.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2,4), dtype="float32"):
            # force_pure is expected because purity checking should be disabled before this pass
            R.func_attr({"relax.force_pure": True})
            cls = Before
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _1: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            storage1: R.Any = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([2, 4]), "float32")
            _2: R.Tuple = R.call_tir_packed(cls.exp, (alloc, alloc1))
            _3: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            lv: R.Tensor((2, 4), dtype="float32") = alloc2
            _4: R.Tuple = R.call_packed("vm.builtin.dummy", (x, lv), ty_args=R.Tuple())
            _5: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), "float32", 0)
            _6 = R.call_tir_packed(cls.exp, (alloc2, alloc3))
            _7: R.Tuple = R.memory.kill_tensor(alloc2)
            _8: R.Tuple = R.memory.kill_storage(storage)
            return alloc3

    @I.ir_module
    class Expected:
        @Ts.function
        def exp(rxplaceholder: T.Tensor((T.int64(2), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            T.func_attr({"global_symbol": "exp", "tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0_i1_fused_0 in T.parallel(T.int64(1), annotations={"thread_binding": "blockIdx.x"}):
                for i0_i1_fused_1 in T.parallel(T.int64(8), annotations={"thread_binding": "threadIdx.x"}):
                    with Ts.sblock("compute"):
                        i0 = Ts.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = Ts.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        Ts.reads(rxplaceholder[i0, i1])
                        Ts.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            storage1: R.Any = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv: R.Tuple(R.Any, R.Any) = (storage, storage1)
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(alloc: R.Tensor((2, 4), dtype="float32"), alloc1: R.Tensor((2, 4), dtype="float32"), storage: R.Any) -> R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            _2: R.Tuple = R.call_tir_packed(cls.exp, (alloc, alloc1))
            _3: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            lv: R.Tensor((2, 4), dtype="float32") = alloc2
            gv: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")) = (lv, alloc2)
            return gv

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2,4), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Any, R.Any) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.get_cached_alloc", (cls.cuda_graph_alloc, R.prim_value(0)), ty_args=(R.Tuple(R.Any, R.Any),))
            storage: R.Any = gv[0]
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _1: R.Tuple = R.call_tir_packed(cls.exp, (x, alloc))
            storage1: R.Any = gv[1]
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            gv1: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (cls.main_cuda_graph_capture, (alloc, alloc1, storage), R.prim_value(0)), ty_args=(R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")),))
            alloc2: R.Tensor((2, 4), dtype="float32") = gv1[1]
            lv: R.Tensor((2, 4), dtype="float32") = gv1[0]
            _4: R.Tuple = R.call_packed("vm.builtin.dummy", (x, lv), ty_args=(R.Tuple,))
            _5: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _6: R.Tuple = R.call_tir_packed(cls.exp, (alloc2, alloc3))
            _7: R.Tuple = R.memory.kill_tensor(alloc2)
            _8: R.Tuple = R.memory.kill_storage(storage)
            return alloc3
    # fmt: on

    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_capture_fixed_inputs():
    @tvm_tirx.script.ir_module
    class Conv2dx3:
        @R.function
        def main(
            data: R.Tensor((16, 32, 32, 16), "float16"),
            weight1: R.Tensor((16, 3, 3, 16), "float16"),
            weight2: R.Tensor((16, 3, 3, 16), "float16"),
            weight3: R.Tensor((16, 3, 3, 16), "float16"),
            gamma: R.Tensor((16,), "float16"),
            beta: R.Tensor((16,), "float16"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                conv1 = R.nn.relu(
                    R.nn.conv2d(
                        data, weight1, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    )
                )

                ###############################################################################
                # The second conv2d and layer norm can be captured into a graph
                conv2 = R.nn.relu(
                    R.nn.conv2d(
                        conv1, weight2, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    )
                )
                ln = R.nn.layer_norm(conv2, gamma, beta, axes=[-1])
                ###############################################################################

                conv3 = R.nn.relu(
                    R.nn.conv2d(
                        ln, weight3, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    )
                )
                R.output(conv3)

            return conv3

    @I.ir_module
    class Expected:
        @Ts.function
        def fused_conv2d_relu(
            data: T.Tensor((T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16"),
            weight1: T.Tensor((T.int64(16), T.int64(3), T.int64(3), T.int64(16)), "float16"),
            var_compute_intermediate: T.Tensor(
                (T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16"
            ),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            pad_temp = Ts.sblock_alloc_tensor(
                (T.int64(16), T.int64(34), T.int64(34), T.int64(16)), "float16"
            )
            var_conv2d_nhwc_intermediate = Ts.sblock_alloc_tensor(
                (T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16"
            )
            for i0, i1, i2, i3 in T.grid(T.int64(16), T.int64(34), T.int64(34), T.int64(16)):
                with Ts.sblock("pad_temp"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(data[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3])
                    Ts.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
                    pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                        T.int64(1) <= v_i1
                        and v_i1 < T.int64(33)
                        and T.int64(1) <= v_i2
                        and v_i2 < T.int64(33),
                        data[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3],
                        T.float16(0),
                    )
            for nn, yy, xx, ff, ry, rx, rc in T.grid(
                T.int64(16),
                T.int64(32),
                T.int64(32),
                T.int64(16),
                T.int64(3),
                T.int64(3),
                T.int64(16),
            ):
                with Ts.sblock("conv2d_nhwc"):
                    v_nn, v_yy, v_xx, v_ff, v_ry, v_rx, v_rc = Ts.axis.remap(
                        "SSSSRRR", [nn, yy, xx, ff, ry, rx, rc]
                    )
                    Ts.reads(
                        pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc],
                        weight1[v_ff, v_ry, v_rx, v_rc],
                    )
                    Ts.writes(var_conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff])
                    with Ts.init():
                        var_conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff] = T.float16(0)
                    var_conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff] = (
                        var_conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff]
                        + pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc]
                        * weight1[v_ff, v_ry, v_rx, v_rc]
                    )
            for i0, i1, i2, i3 in T.grid(T.int64(16), T.int64(32), T.int64(32), T.int64(16)):
                with Ts.sblock("compute"):
                    v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(var_conv2d_nhwc_intermediate[v_i0, v_i1, v_i2, v_i3])
                    Ts.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
                    var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(
                        var_conv2d_nhwc_intermediate[v_i0, v_i1, v_i2, v_i3], T.float16(0)
                    )

        @Ts.function
        def layer_norm(
            A: T.Tensor((T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16"),
            B: T.Tensor((T.int64(16),), "float16"),
            C: T.Tensor((T.int64(16),), "float16"),
            T_layer_norm: T.Tensor((T.int64(16), T.int64(32), T.int64(32), T.int64(16)), "float16"),
        ):
            T.func_attr({"op_pattern": 4, "tirx.noalias": True})
            # with Ts.sblock("root"):
            A_red_temp_v0 = Ts.sblock_alloc_tensor((T.int64(16), T.int64(32), T.int64(32)))
            A_red_temp_v1 = Ts.sblock_alloc_tensor((T.int64(16), T.int64(32), T.int64(32)))
            for ax0, ax1, ax2, k3 in T.grid(T.int64(16), T.int64(32), T.int64(32), T.int64(16)):
                with Ts.sblock("A_red_temp"):
                    v_ax0, v_ax1, v_ax2, v_k3 = Ts.axis.remap("SSSR", [ax0, ax1, ax2, k3])
                    Ts.reads(A[v_ax0, v_ax1, v_ax2, v_k3])
                    Ts.writes(
                        A_red_temp_v0[v_ax0, v_ax1, v_ax2], A_red_temp_v1[v_ax0, v_ax1, v_ax2]
                    )
                    with Ts.init():
                        A_red_temp_v0[v_ax0, v_ax1, v_ax2] = T.float32(0)
                        A_red_temp_v1[v_ax0, v_ax1, v_ax2] = T.float32(0)
                    v_A_red_temp_v0: T.let[T.float32] = A_red_temp_v0[v_ax0, v_ax1, v_ax2] + T.Cast(
                        "float32", A[v_ax0, v_ax1, v_ax2, v_k3]
                    )
                    v_A_red_temp_v1: T.let[T.float32] = A_red_temp_v1[v_ax0, v_ax1, v_ax2] + T.Cast(
                        "float32", A[v_ax0, v_ax1, v_ax2, v_k3]
                    ) * T.Cast("float32", A[v_ax0, v_ax1, v_ax2, v_k3])
                    A_red_temp_v0[v_ax0, v_ax1, v_ax2] = v_A_red_temp_v0
                    A_red_temp_v1[v_ax0, v_ax1, v_ax2] = v_A_red_temp_v1
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(32), T.int64(32), T.int64(16)):
                with Ts.sblock("T_layer_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(
                        A[v_ax0, v_ax1, v_ax2, v_ax3],
                        A_red_temp_v0[v_ax0, v_ax1, v_ax2],
                        A_red_temp_v1[v_ax0, v_ax1, v_ax2],
                        B[v_ax3],
                        C[v_ax3],
                    )
                    Ts.writes(T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3] = (
                        T.Cast(
                            "float16",
                            (
                                T.Cast("float32", A[v_ax0, v_ax1, v_ax2, v_ax3])
                                - A_red_temp_v0[v_ax0, v_ax1, v_ax2] * T.float32(0.0625)
                            )
                            * T.rsqrt(
                                A_red_temp_v1[v_ax0, v_ax1, v_ax2] * T.float32(0.0625)
                                - A_red_temp_v0[v_ax0, v_ax1, v_ax2]
                                * T.float32(0.0625)
                                * (A_red_temp_v0[v_ax0, v_ax1, v_ax2] * T.float32(0.0625))
                                + T.float32(1.0000000000000001e-05)
                            ),
                        )
                        * B[v_ax3]
                        + C[v_ax3]
                    )

        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage: R.Any = R.memory.alloc_storage(
                R.shape([524288]), R.prim_value(0), R.str("global"), R.dtype("float16")
            )
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([524288]), R.prim_value(0), R.str("global"), R.dtype("float16")
            )
            gv: R.Tuple(R.Any, R.Any) = storage, storage1
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(
            lv: R.Tensor((16, 32, 32, 16), dtype="float16"),
            lv1: R.Tensor((16, 3, 3, 16), dtype="float16"),
            alloc1: R.Tensor((16, 32, 32, 16), dtype="float16"),
            alloc: R.Tensor((16, 32, 32, 16), dtype="float16"),
            params: R.Tuple(
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16,), dtype="float16"),
                R.Tensor((16,), dtype="float16"),
            ),
            storage: R.Any,
        ) -> R.Tuple(
            R.Tensor((16, 32, 32, 16), dtype="float16"),
            R.Tensor((16, 3, 3, 16), dtype="float16"),
            R.Tensor((16, 32, 32, 16), dtype="float16"),
        ):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            _1: R.Tuple = R.call_tir_packed(cls.fused_conv2d_relu, (lv, lv1, alloc1))
            _: R.Tuple = R.memory.kill_tensor(alloc)
            lv1_1: R.Tensor((16, 32, 32, 16), dtype="float16") = alloc1
            lv2: R.Tensor((16,), dtype="float16") = params[3]
            lv3: R.Tensor((16,), dtype="float16") = params[4]
            alloc2: R.Tensor((16, 32, 32, 16), dtype="float16") = R.memory.alloc_tensor(
                storage, R.prim_value(0), R.shape([16, 32, 32, 16]), R.dtype("float16")
            )
            _2: R.Tuple = R.call_tir_packed(cls.layer_norm, (lv1_1, lv2, lv3, alloc2))
            _1_1: R.Tuple = R.memory.kill_tensor(alloc1)
            ln: R.Tensor((16, 32, 32, 16), dtype="float16") = alloc2
            lv4: R.Tensor((16, 3, 3, 16), dtype="float16") = params[2]
            gv: R.Tuple(
                R.Tensor((16, 32, 32, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 32, 32, 16), dtype="float16"),
            ) = (ln, lv4, alloc2)
            return gv

        @R.function
        def main_transform_params(
            params: R.Tuple(
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16,), dtype="float16"),
                R.Tensor((16,), dtype="float16"),
            ),
        ) -> R.Tuple(
            R.Tensor((16, 3, 3, 16), dtype="float16"),
            R.Tensor((16, 3, 3, 16), dtype="float16"),
            R.Tensor((16, 3, 3, 16), dtype="float16"),
            R.Tensor((16,), dtype="float16"),
            R.Tensor((16,), dtype="float16"),
        ):
            R.func_attr({"relax.force_pure": True})
            lv: R.Tensor((16, 3, 3, 16), dtype="float16") = params[0]
            lv1: R.Tensor((16, 3, 3, 16), dtype="float16") = params[1]
            lv2: R.Tensor((16, 3, 3, 16), dtype="float16") = params[2]
            lv3: R.Tensor((16,), dtype="float16") = params[3]
            lv4: R.Tensor((16,), dtype="float16") = params[4]
            gv: R.Tuple(
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16,), dtype="float16"),
                R.Tensor((16,), dtype="float16"),
            ) = (lv, lv1, lv2, lv3, lv4)
            return gv

        @R.function
        def main(
            data: R.Tensor((16, 32, 32, 16), dtype="float16"),
            params: R.Tuple(
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16,), dtype="float16"),
                R.Tensor((16,), dtype="float16"),
            ),
        ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
            R.func_attr({"num_input": 1, "relax.force_pure": True})
            cls = Expected
            lv: R.Tensor((16, 3, 3, 16), dtype="float16") = params[0]
            gv: R.Tuple(R.Any, R.Any) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.get_cached_alloc",
                (cls.cuda_graph_alloc, R.prim_value(0)),
                ty_args=(R.Tuple(R.Any, R.Any),),
            )
            storage: R.Any = gv[0]
            alloc: R.Tensor((16, 32, 32, 16), dtype="float16") = R.memory.alloc_tensor(
                storage, R.prim_value(0), R.shape([16, 32, 32, 16]), R.dtype("float16")
            )
            _: R.Tuple = R.call_tir_packed(cls.fused_conv2d_relu, (data, lv, alloc))
            lv_1: R.Tensor((16, 32, 32, 16), dtype="float16") = alloc
            lv1: R.Tensor((16, 3, 3, 16), dtype="float16") = params[1]
            storage1: R.Any = gv[1]
            alloc1: R.Tensor((16, 32, 32, 16), dtype="float16") = R.memory.alloc_tensor(
                storage1, R.prim_value(0), R.shape([16, 32, 32, 16]), R.dtype("float16")
            )
            gv1: R.Tuple(
                R.Tensor((16, 32, 32, 16), dtype="float16"),
                R.Tensor((16, 3, 3, 16), dtype="float16"),
                R.Tensor((16, 32, 32, 16), dtype="float16"),
            ) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.run_or_capture",
                (
                    cls.main_cuda_graph_capture,
                    (lv_1, lv1, alloc1, alloc, params, storage),
                    R.prim_value(0),
                ),
                ty_args=(
                    R.Tuple(
                        R.Tensor((16, 32, 32, 16), dtype="float16"),
                        R.Tensor((16, 3, 3, 16), dtype="float16"),
                        R.Tensor((16, 32, 32, 16), dtype="float16"),
                    ),
                ),
            )
            alloc2: R.Tensor((16, 32, 32, 16), dtype="float16") = gv1[2]
            ln: R.Tensor((16, 32, 32, 16), dtype="float16") = gv1[0]
            lv4: R.Tensor((16, 3, 3, 16), dtype="float16") = gv1[1]
            alloc3: R.Tensor((16, 32, 32, 16), dtype="float16") = R.builtin.alloc_tensor(
                R.shape([16, 32, 32, 16]), R.dtype("float16"), R.prim_value(0)
            )
            _3: R.Tuple = R.call_tir_packed(cls.fused_conv2d_relu, (ln, lv4, alloc3))
            _2: R.Tuple = R.memory.kill_tensor(alloc2)
            gv_1: R.Tensor((16, 32, 32, 16), dtype="float16") = alloc3
            _3_1: R.Tuple = R.memory.kill_storage(storage)
            _4: R.Tuple = R.memory.kill_storage(storage1)
            return gv_1

    mod = tvm_tirx.ir.transform.Sequential(
        [
            relax.pipeline.get_pipeline(),
            relax.transform.LiftTransformParams(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
            relax.transform.StaticPlanBlockMemory(),
        ]
    )(Conv2dx3)

    mod["main"] = mod["main"].with_attr({"num_input": 1})
    after = relax.transform.RewriteCUDAGraph()(mod)
    tvm_tirx.ir.assert_structural_equal(after, after)


def test_null_value():
    @I.ir_module
    class Before:
        @R.function
        def main() -> R.Tuple(R.Any):
            _io: R.Any = R.null_value()
            lv: R.Tuple(R.Any) = (_io,)
            gv: R.Tuple(R.Any) = lv
            return gv

    Expected = Before

    After = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_transform_is_no_op_when_disabled():
    @I.ir_module
    class Before:
        @R.function
        def main():
            storage = R.memory.alloc_storage(R.shape([8]), 0, "global", "float32")
            alloc3 = R.memory.alloc_tensor(storage, 0, R.shape([8]), "float32")
            return ()

    with tvm_tirx.ir.transform.PassContext(config={"relax.backend.use_cuda_graph": True}):
        AfterWhenEnabled = relax.transform.RewriteCUDAGraph()(Before)
    with tvm_tirx.ir.transform.PassContext(config={"relax.backend.use_cuda_graph": False}):
        AfterWhenDisabled = relax.transform.RewriteCUDAGraph()(Before)

    assert not tvm_ffi.structural_equal(Before, AfterWhenEnabled)
    tvm_tirx.ir.assert_structural_equal(Before, AfterWhenDisabled)


def test_static_args():
    @I.ir_module
    class Before:
        @R.function(pure=False)
        def main():
            storage0 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float32")
            alloc0 = R.memory.alloc_tensor(storage0, 0, R.shape([8]), "float32")
            _ = R.call_packed("dummy_func", alloc0, R.dtype("float32"), R.str("string"))
            return ()

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any):
            R.func_attr({"relax.force_pure": True})
            storage0: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            gv: R.Tuple(R.Any) = (storage0,)
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(alloc0: R.Tensor((8,), dtype="float32")) -> R.Tuple:
            R.func_attr({"relax.force_pure": True})
            _: R.Any = R.call_packed("dummy_func", alloc0, R.dtype("float32"), R.str("string"))
            gv: R.Tuple = ()
            return gv

        @R.function(pure=False)
        def main() -> R.Tuple:
            cls = Expected
            gv: R.Tuple(R.Any) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.get_cached_alloc",
                (cls.cuda_graph_alloc, R.prim_value(0)),
                ty_args=(R.Tuple(R.Any),),
            )
            storage0: R.Any = gv[0]
            alloc0: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(
                storage0, R.prim_value(0), R.shape([8]), R.dtype("float32")
            )
            gv1: R.Tuple = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.run_or_capture",
                (cls.main_cuda_graph_capture, (alloc0,), R.prim_value(0)),
                ty_args=(R.Tuple,),
            )
            return ()

    mod = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_dynamic_capture():
    m_add_one = T.dynamic("m", "int64")
    m_main = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @Ts.function
        def add_one(x: T.Tensor((m_add_one,), "float32"), y: T.Tensor((m_add_one,), "float32")):
            # Use T.serial with explicit int64 min so the inner sblock iter_var
            # dom is all-int64 (matches what Expected emits via Ts.axis.spatial(m, i)).
            for i in T.serial(T.int64(0), m_add_one):
                with Ts.sblock("add"):
                    vi = Ts.axis.remap("S", [i])
                    y[vi] = x[vi] + T.float32(1)

        @R.function
        def main(x: R.Tensor((m_main,), "float32")) -> R.Tensor((m_main,), "float32"):
            R.func_attr(
                {"relax.rewrite_cuda_graph.capture_symbolic_vars": ["m"], "relax.force_pure": True}
            )
            storage: R.Any = R.memory.alloc_storage(
                R.shape([16]), 0, "global", "float32"
            )  # assume m is upper-bounded
            alloc1: R.Tensor((m_main,), "float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([m_main]), "float32"
            )
            _ = R.call_tir_packed(Before.add_one, (x, alloc1))
            storage1: R.Any = R.memory.alloc_storage(R.shape([16]), 0, "global", "float32")
            alloc2: R.Tensor((m_main,), "float32") = R.memory.alloc_tensor(
                storage1, 0, R.shape([m_main]), "float32"
            )
            _ = R.call_tir_packed(Before.add_one, (alloc1, alloc2))
            alloc3: R.Tensor((m_main,), "float32") = R.builtin.alloc_tensor(
                R.shape([m_main]), "float32", 0, "global"
            )
            _ = R.call_tir_packed(Before.add_one, (alloc2, alloc3))
            return alloc3

    m_add_one = T.dynamic("m", "int64")
    m_main_cuda_graph_capture = T.dynamic("m", "int64")
    m_main = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def add_one(x: T.Tensor((m_add_one,), "float32"), y: T.Tensor((m_add_one,), "float32")):
            # with Ts.sblock("root"):
            for i in T.serial(T.int64(0), m_add_one):
                with Ts.sblock("add"):
                    vi = Ts.axis.spatial(m_add_one, i)
                    Ts.reads(x[vi])
                    Ts.writes(y[vi])
                    y[vi] = x[vi] + T.float32(1)

        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage: R.Any = R.memory.alloc_storage(
                R.shape([16]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([16]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            gv: R.Tuple(R.Any, R.Any) = storage, storage1
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(
            alloc1: R.Tensor((m_main_cuda_graph_capture,), dtype="float32"),
            alloc2: R.Tensor((m_main_cuda_graph_capture,), dtype="float32"),
            shape_expr: R.Shape([m_main_cuda_graph_capture]),
        ):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            R.call_tir_packed(cls.add_one, (alloc1, alloc2))
            gv = ()
            return ()

        @R.function
        def main(x: R.Tensor((m_main,), dtype="float32")) -> R.Tensor((m_main,), dtype="float32"):
            R.func_attr(
                {"relax.force_pure": True, "relax.rewrite_cuda_graph.capture_symbolic_vars": ["m"]}
            )
            cls = Expected
            gv: R.Tuple(R.Any, R.Any) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.get_cached_alloc",
                (cls.cuda_graph_alloc, R.prim_value(0)),
                ty_args=(R.Tuple(R.Any, R.Any),),
            )
            storage: R.Any = gv[0]
            alloc1: R.Tensor((m_main,), dtype="float32") = R.memory.alloc_tensor(
                storage, R.prim_value(0), R.shape([m_main]), R.dtype("float32")
            )
            R.call_tir_packed(cls.add_one, (x, alloc1))
            storage1: R.Any = gv[1]
            alloc2: R.Tensor((m_main,), dtype="float32") = R.memory.alloc_tensor(
                storage1, R.prim_value(0), R.shape([m_main]), R.dtype("float32")
            )
            R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.run_or_capture",
                (
                    cls.main_cuda_graph_capture,
                    (alloc1, alloc2, R.shape([m_main])),
                    R.prim_value(0),
                    R.shape([m_main]),
                ),
                ty_args=(R.Tuple,),
            )
            alloc3: R.Tensor((m_main,), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([m_main]), R.dtype("float32"), R.prim_value(0), R.str("global")
            )
            R.call_tir_packed(cls.add_one, (alloc2, alloc3))
            return alloc3

    mod = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_merge_alloc_funcs():
    @I.ir_module
    class Before:
        @R.function
        def func1():
            R.func_attr({"relax.force_pure": True})
            storage1 = R.memory.alloc_storage(R.shape([128]), 0, "global", "float32")
            storage2 = R.memory.alloc_storage(R.shape([256]), 0, "global", "float32")
            storage3 = R.memory.alloc_storage(R.shape([512]), 0, "ipc_memory", "float32")
            alloc1 = R.memory.alloc_tensor(storage1, 0, R.shape([128]), "float32")
            alloc2 = R.memory.alloc_tensor(storage2, 0, R.shape([256]), "float32")
            alloc3 = R.memory.alloc_tensor(storage3, 0, R.shape([512]), "float32")
            R.call_packed("dummy", alloc1, alloc2, alloc3, ty_args=(R.Tuple,))
            return ()

        @R.function
        def func2():
            R.func_attr({"relax.force_pure": True})
            storage1 = R.memory.alloc_storage(R.shape([192]), 0, "global", "float32")
            storage2 = R.memory.alloc_storage(R.shape([64]), 0, "global", "float32")
            storage3 = R.memory.alloc_storage(R.shape([1024]), 0, "ipc_memory", "float32")
            storage4 = R.memory.alloc_storage(R.shape([512]), 0, "global", "float32")
            alloc1 = R.memory.alloc_tensor(storage1, 0, R.shape([192]), "float32")
            alloc2 = R.memory.alloc_tensor(storage2, 0, R.shape([64]), "float32")
            alloc3 = R.memory.alloc_tensor(storage3, 0, R.shape([1024]), "float32")
            alloc4 = R.memory.alloc_tensor(storage4, 0, R.shape([512]), "float32")
            R.call_packed("dummy", alloc1, alloc2, alloc3, alloc4, ty_args=(R.Tuple,))
            return ()

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any, R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage4: R.Any = R.memory.alloc_storage(
                R.shape([512]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([192]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            storage2: R.Any = R.memory.alloc_storage(
                R.shape([64]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            storage3: R.Any = R.memory.alloc_storage(
                R.shape([1024]), R.prim_value(0), R.str("ipc_memory"), R.dtype("float32")
            )
            gv: R.Tuple(R.Any, R.Any, R.Any, R.Any) = (
                storage4,
                storage1,
                storage2,
                storage3,
            )
            return gv

        @R.function
        def func1() -> R.Tuple:
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Any, R.Any, R.Any, R.Any) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.get_cached_alloc",
                (cls.cuda_graph_alloc, R.prim_value(0)),
                ty_args=(R.Tuple(R.Any, R.Any, R.Any, R.Any),),
            )
            storage1: R.Any = gv[1]
            storage2: R.Any = gv[0]
            storage3: R.Any = gv[3]
            alloc1: R.Tensor((128,), dtype="float32") = R.memory.alloc_tensor(
                storage1, R.prim_value(0), R.shape([128]), R.dtype("float32")
            )
            alloc2: R.Tensor((256,), dtype="float32") = R.memory.alloc_tensor(
                storage2, R.prim_value(0), R.shape([256]), R.dtype("float32")
            )
            alloc3: R.Tensor((512,), dtype="float32") = R.memory.alloc_tensor(
                storage3, R.prim_value(0), R.shape([512]), R.dtype("float32")
            )
            R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.run_or_capture",
                (cls.func1_cuda_graph_capture, (alloc1, alloc2, alloc3), R.prim_value(0)),
                ty_args=(R.Tuple,),
            )
            return ()

        @R.function(private=True)
        def func1_cuda_graph_capture(
            alloc1: R.Tensor((128,), dtype="float32"),
            alloc2: R.Tensor((256,), dtype="float32"),
            alloc3: R.Tensor((512,), dtype="float32"),
        ) -> R.Tuple:
            R.func_attr({"relax.force_pure": True})
            R.call_packed("dummy", alloc1, alloc2, alloc3, ty_args=(R.Tuple,))
            ()
            return ()

        @R.function
        def func2() -> R.Tuple:
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv2: R.Tuple(R.Any, R.Any, R.Any, R.Any) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.get_cached_alloc",
                (cls.cuda_graph_alloc, R.prim_value(0)),
                ty_args=(R.Tuple(R.Any, R.Any, R.Any, R.Any),),
            )
            storage11: R.Any = gv2[1]
            storage21: R.Any = gv2[2]
            storage31: R.Any = gv2[3]
            storage4: R.Any = gv2[0]
            alloc1: R.Tensor((192,), dtype="float32") = R.memory.alloc_tensor(
                storage11, R.prim_value(0), R.shape([192]), R.dtype("float32")
            )
            alloc2: R.Tensor((64,), dtype="float32") = R.memory.alloc_tensor(
                storage21, R.prim_value(0), R.shape([64]), R.dtype("float32")
            )
            alloc3: R.Tensor((1024,), dtype="float32") = R.memory.alloc_tensor(
                storage31, R.prim_value(0), R.shape([1024]), R.dtype("float32")
            )
            alloc4: R.Tensor((512,), dtype="float32") = R.memory.alloc_tensor(
                storage4, R.prim_value(0), R.shape([512]), R.dtype("float32")
            )
            R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.run_or_capture",
                (cls.func2_cuda_graph_capture, (alloc1, alloc2, alloc3, alloc4), R.prim_value(1)),
                ty_args=(R.Tuple,),
            )
            return ()

        @R.function(private=True)
        def func2_cuda_graph_capture(
            alloc1: R.Tensor((192,), dtype="float32"),
            alloc2: R.Tensor((64,), dtype="float32"),
            alloc3: R.Tensor((1024,), dtype="float32"),
            alloc4: R.Tensor((512,), dtype="float32"),
        ) -> R.Tuple:
            R.func_attr({"relax.force_pure": True})
            R.call_packed("dummy", alloc1, alloc2, alloc3, alloc4, ty_args=(R.Tuple,))
            ()
            return ()

    After = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_disable_capture_output():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((8,), "float32")) -> R.Tuple(R.Tensor((8,), "float32")):
            R.func_attr({"relax.force_pure": True})
            storage1 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float32")
            alloc1 = R.memory.alloc_tensor(storage1, 0, R.shape([8]), "float32")
            _ = R.call_packed("dummy", x, alloc1, ty_args=(R.Tuple,))
            storage2 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float32")
            alloc2 = R.memory.alloc_tensor(storage2, 0, R.shape([8]), "float32")
            _1 = R.call_packed("dummy", alloc1, alloc2, ty_args=(R.Tuple,))
            storage3 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float32")
            alloc3 = R.memory.alloc_tensor(storage3, 0, R.shape([8]), "float32")
            _2 = R.call_packed("dummy", alloc2, alloc3, ty_args=(R.Tuple,))
            gv = (alloc3,)
            return gv

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            storage2: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            gv: R.Tuple(R.Any, R.Any) = storage1, storage2
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(
            alloc1: R.Tensor((8,), dtype="float32"), alloc2: R.Tensor((8,), dtype="float32")
        ) -> R.Tuple:
            R.func_attr({"relax.force_pure": True})
            R.call_packed("dummy", alloc1, alloc2, ty_args=(R.Tuple,))
            ()
            return ()

        @R.function
        def main(x: R.Tensor((8,), dtype="float32")) -> R.Tuple(R.Tensor((8,), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Any, R.Any) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.get_cached_alloc",
                (cls.cuda_graph_alloc, R.prim_value(0)),
                ty_args=(R.Tuple(R.Any, R.Any),),
            )
            storage1: R.Any = gv[0]
            alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(
                storage1, R.prim_value(0), R.shape([8]), R.dtype("float32")
            )
            R.call_packed("dummy", x, alloc1, ty_args=(R.Tuple,))
            storage2: R.Any = gv[1]
            alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(
                storage2, R.prim_value(0), R.shape([8]), R.dtype("float32")
            )
            R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.run_or_capture",
                (cls.main_cuda_graph_capture, (alloc1, alloc2), R.prim_value(0)),
                ty_args=(R.Tuple,),
            )
            storage3: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            alloc3: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(
                storage3, R.prim_value(0), R.shape([8]), R.dtype("float32")
            )
            R.call_packed("dummy", alloc2, alloc3, ty_args=(R.Tuple,))
            gv = (alloc3,)
            return gv

    After = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_static_input_with_symbolic_shape():
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((8,), "float16"), w: R.Tensor((m,))):
            R.func_attr({"relax.force_pure": True, "num_input": 1})
            storage1 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float16")
            alloc1 = R.memory.alloc_tensor(storage1, 0, R.shape([8]), "float16")
            _ = R.call_packed("dummy", x, w, alloc1, ty_args=(R.Tuple,))
            storage2 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float16")
            alloc2 = R.memory.alloc_tensor(storage2, 0, R.shape([8]), "float16")
            _1 = R.call_packed("dummy", alloc1, w, alloc2, ty_args=(R.Tuple,))
            storage3 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float16")
            alloc3 = R.memory.alloc_tensor(storage3, 0, R.shape([8]), "float16")
            _2 = R.call_packed("dummy", alloc2, w, alloc3, ty_args=(R.Tuple,))
            gv = (alloc3,)
            return gv

    m_main_cuda_graph_capture = T.dynamic("m", "int64")
    m_main = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16")
            )
            storage2: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16")
            )
            gv: R.Tuple(R.Any, R.Any) = storage1, storage2
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(
            alloc1: R.Tensor((8,), dtype="float16"),
            w: R.Tensor((m_main_cuda_graph_capture,)),
            alloc2: R.Tensor((8,), dtype="float16"),
            shape_expr: R.Shape([m_main_cuda_graph_capture]),
        ) -> R.Tuple:
            R.func_attr({"relax.force_pure": True})
            R.call_packed("dummy", alloc1, w, alloc2, ty_args=(R.Tuple,))
            ()
            return ()

        @R.function
        def main(x: R.Tensor((8,), dtype="float16"), w: R.Tensor((m_main,))) -> R.Tuple(
            R.Tensor((8,), dtype="float16")
        ):
            R.func_attr({"num_input": 1, "relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Any, R.Any) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.get_cached_alloc",
                (cls.cuda_graph_alloc, R.prim_value(0)),
                ty_args=(R.Tuple(R.Any, R.Any),),
            )
            storage1: R.Any = gv[0]
            alloc1: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(
                storage1, R.prim_value(0), R.shape([8]), R.dtype("float16")
            )
            R.call_packed("dummy", x, w, alloc1, ty_args=(R.Tuple,))
            storage2: R.Any = gv[1]
            alloc2: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(
                storage2, R.prim_value(0), R.shape([8]), R.dtype("float16")
            )
            R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.run_or_capture",
                (
                    cls.main_cuda_graph_capture,
                    (alloc1, w, alloc2, R.shape([m_main])),
                    R.prim_value(0),
                    R.shape([m_main]),
                ),
                ty_args=(R.Tuple,),
            )
            storage3: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16")
            )
            alloc3: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(
                storage3, R.prim_value(0), R.shape([8]), R.dtype("float16")
            )
            R.call_packed("dummy", alloc2, w, alloc3, ty_args=(R.Tuple,))
            gv_1: R.Tuple(R.Tensor((8,), dtype="float16")) = (alloc3,)
            return gv_1

    After = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()


def test_match_cast_of_captured_value():
    m = T.dynamic("m", "int64")

    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((8,), "float16")):
            R.func_attr({"relax.force_pure": True, "num_input": 1})
            storage1 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float16")
            alloc1 = R.memory.alloc_tensor(storage1, 0, R.shape([8]), "float16")
            _ = R.call_packed("dummy", x, alloc1, ty_args=(R.Tuple,))
            storage2 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float16")
            alloc2 = R.memory.alloc_tensor(storage2, 0, R.shape([8]), "float16")
            _1 = R.call_packed("dummy", alloc1, alloc2, ty_args=(R.Tuple,))
            lv = R.reshape(alloc2, R.shape([2, 4]))
            lv1 = R.match_cast(lv, R.Tensor((m, 4), "float16"))
            storage3 = R.memory.alloc_storage(R.shape([8]), 0, "global", "float16")
            alloc3 = R.memory.alloc_tensor(storage3, 0, R.shape([8]), "float16")
            _2 = R.call_packed("dummy", lv1, alloc3, ty_args=(R.Tuple,))
            gv = (alloc3,)
            return gv

    @I.ir_module
    class Expected:
        @R.function(private=True)
        def cuda_graph_alloc() -> R.Tuple(R.Any, R.Any):
            R.func_attr({"relax.force_pure": True})
            storage1: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16")
            )
            storage2: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16")
            )
            gv: R.Tuple(R.Any, R.Any) = storage1, storage2
            return gv

        @R.function(private=True)
        def main_cuda_graph_capture(
            alloc1: R.Tensor((8,), dtype="float16"), alloc2: R.Tensor((8,), dtype="float16")
        ) -> R.Tuple(R.Tensor((2, 4), dtype="float16")):
            R.func_attr({"relax.force_pure": True})
            R.call_packed("dummy", alloc1, alloc2, ty_args=(R.Tuple,))
            lv: R.Tensor((2, 4), dtype="float16") = R.reshape(alloc2, R.shape([2, 4]))
            gv: R.Tuple(R.Tensor((2, 4), dtype="float16")) = (lv,)
            return gv

        @R.function
        def main(x: R.Tensor((8,), dtype="float16")) -> R.Tuple(R.Tensor((8,), dtype="float16")):
            R.func_attr({"num_input": 1, "relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Any, R.Any) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.get_cached_alloc",
                (cls.cuda_graph_alloc, R.prim_value(0)),
                ty_args=(R.Tuple(R.Any, R.Any),),
            )
            storage1: R.Any = gv[0]
            alloc1: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(
                storage1, R.prim_value(0), R.shape([8]), R.dtype("float16")
            )
            R.call_packed("dummy", x, alloc1, ty_args=(R.Tuple,))
            storage2: R.Any = gv[1]
            alloc2: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(
                storage2, R.prim_value(0), R.shape([8]), R.dtype("float16")
            )
            gv1: R.Tuple(R.Tensor((2, 4), dtype="float16")) = R.call_builtin_with_ctx(
                "vm.builtin.cuda_graph.run_or_capture",
                (cls.main_cuda_graph_capture, (alloc1, alloc2), R.prim_value(0)),
                ty_args=(R.Tuple(R.Tensor((2, 4), dtype="float16")),),
            )
            lv: R.Tensor((2, 4), dtype="float16") = gv1[0]
            lv1: R.Tensor((m, 4), dtype="float16") = R.match_cast(
                lv, R.Tensor((m, 4), dtype="float16")
            )
            storage3: R.Any = R.memory.alloc_storage(
                R.shape([8]), R.prim_value(0), R.str("global"), R.dtype("float16")
            )
            alloc3: R.Tensor((8,), dtype="float16") = R.memory.alloc_tensor(
                storage3, R.prim_value(0), R.shape([8]), R.dtype("float16")
            )
            R.call_packed("dummy", lv1, alloc3, ty_args=(R.Tuple,))
            gv_1: R.Tuple(R.Tensor((8,), dtype="float16")) = (alloc3,)
            return gv_1

    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)
