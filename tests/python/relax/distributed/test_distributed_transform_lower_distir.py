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

#  type: ignore
from __future__ import annotations

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.s_tir import script as Ts
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.ir import assert_structural_equal
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def test_mlp():
    @I.ir_module
    class MLP:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @Ts.function(private=True)
        def gelu1(
            A: T.Tensor((T.int64(128), T.int64(64)), "float32"),
            T_multiply: T.Tensor((T.int64(128), T.int64(64)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_multiply_1 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            compute = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            T_multiply_2 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            T_add = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T.float32(0.70710678118654757)
            for i0, i1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_multiply_1[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.erf(T_multiply_1[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(compute[v_ax0, v_ax1])
                    Ts.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = compute[v_ax0, v_ax1] * T.float32(0.5)
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_2[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(0.5) + T_multiply_2[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]

        @Ts.function(private=True)
        def matmul1(
            A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            B: T.Tensor((T.int64(128), T.int64(64)), "float32"),
            matmul_1: T.Tensor((T.int64(128), T.int64(64)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(64), T.int64(128)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(A[v_i0, v_k], B[v_k, v_i1])
                    Ts.writes(matmul_1[v_i0, v_i1])
                    with Ts.init():
                        matmul_1[v_i0, v_i1] = T.float32(0)
                    matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

        @Ts.function(private=True)
        def matmul2(
            A: T.Tensor((T.int64(128), T.int64(64)), "float32"),
            B: T.Tensor((T.int64(64), T.int64(128)), "float32"),
            matmul_1: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(64)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(A[v_i0, v_k], B[v_k, v_i1])
                    Ts.writes(matmul_1[v_i0, v_i1])
                    with Ts.init():
                        matmul_1[v_i0, v_i1] = T.float32(0)
                    matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
            weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        ) -> R.DTensor((128, 128), "float32", "mesh[0]", "R"):
            R.func_attr({"num_input": 1})
            cls = MLP
            lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.dist.call_tir_local_view(
                cls.matmul1,
                (x, weight1),
                ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "S[1]")],
            )
            lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.dist.call_tir_local_view(
                cls.gelu1, (lv0,), ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "S[1]")]
            )
            lv2: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = lv1
            gv: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.dist.call_tir_local_view(
                cls.matmul2,
                (lv2, weight2),
                ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "R")],
            )
            lv3: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.ccl.allreduce(
                gv, op_type="sum"
            )
            return lv3

    @I.ir_module(check_well_formed=False)
    class LoweredMLP:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @R.function
        def foo(
            x: R.Tensor((128, 128), dtype="float32"),
            weight1: R.Tensor((128, 128), dtype="float32"),
            weight2: R.Tensor((128, 128), dtype="float32"),
        ) -> R.Tensor((128, 128), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = LoweredMLP
            gv: R.Tensor((128, 128), dtype="float32") = R.ccl.broadcast_from_worker0(x)
            gv1: R.Tensor((128, 64), dtype="float32") = R.ccl.scatter_from_worker0(
                weight1, num_workers=2, axis=1
            )
            gv2: R.Tensor((64, 128), dtype="float32") = R.ccl.scatter_from_worker0(
                weight2, num_workers=2, axis=0
            )
            lv0 = R.call_tir(
                MLP.get_global_var("matmul1"),
                (gv, gv1),
                ty_args=[R.Tensor((128, 64), dtype="float32")],
            )
            lv1 = R.call_tir(
                MLP.get_global_var("gelu1"), (lv0,), ty_args=[R.Tensor((128, 64), dtype="float32")]
            )
            lv2: R.Tensor((128, 64), dtype="float32") = lv1
            gv_1 = R.call_tir(
                MLP.get_global_var("matmul2"),
                (lv2, gv2),
                ty_args=[R.Tensor((128, 128), dtype="float32")],
            )
            lv3: R.Tensor((128, 128), dtype="float32") = R.ccl.allreduce(gv_1, op_type="sum")
            return lv3

    for gv, func in MLP.functions_items():
        if gv.name_hint != "foo":
            LoweredMLP[gv] = func

    mod = MLP
    mod = relax.distributed.transform.LowerDistIR()(mod)
    tvm_tirx.ir.assert_structural_equal(mod, LoweredMLP)


def test_mlp_with_tuple():
    @I.ir_module
    class MLPWithTuple:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @Ts.function(private=True)
        def gelu1(
            A: T.Tensor((T.int64(128), T.int64(64)), "float32"),
            T_multiply: T.Tensor((T.int64(128), T.int64(64)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            T_multiply_1 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            compute = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            T_multiply_2 = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            T_add = Ts.sblock_alloc_tensor((T.int64(128), T.int64(64)))
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T.float32(0.70710678118654757)
            for i0, i1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(T_multiply_1[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.erf(T_multiply_1[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(compute[v_ax0, v_ax1])
                    Ts.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = compute[v_ax0, v_ax1] * T.float32(0.5)
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_2[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(0.5) + T_multiply_2[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("T_multiply_2"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]

        @Ts.function(private=True)
        def matmul11(
            A: T.Tensor((T.int64(64), T.int64(64)), "float32"),
            B: T.Tensor((T.int64(64), T.int64(128)), "float32"),
            matmul: T.Tensor((T.int64(64), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(64), T.int64(128), T.int64(64)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(A[v_i0, v_k], B[v_k, v_i1])
                    Ts.writes(matmul[v_i0, v_i1])
                    with Ts.init():
                        matmul[v_i0, v_i1] = T.float32(0)
                    matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

        @Ts.function(private=True)
        def matmul2(
            A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            B: T.Tensor((T.int64(128), T.int64(64)), "float32"),
            matmul: T.Tensor((T.int64(128), T.int64(64)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(64), T.int64(128)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_k = Ts.axis.remap("SSR", [i0, i1, k])
                    Ts.reads(A[v_i0, v_k], B[v_k, v_i1])
                    Ts.writes(matmul[v_i0, v_i1])
                    with Ts.init():
                        matmul[v_i0, v_i1] = T.float32(0)
                    matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

        @Ts.function(private=True)
        def split11(
            A: T.Tensor((128, 64), "float32"),
            T_split: T.Tensor((64, 64), "float32"),
            T_split_1: T.Tensor((64, 64), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax1, ax2 in T.grid(64, 64):
                with Ts.sblock("T_split"):
                    v_ax1, v_ax2 = Ts.axis.remap("SS", [ax1, ax2])
                    Ts.reads(A[v_ax1, v_ax2])
                    Ts.writes(T_split[v_ax1, v_ax2])
                    T_split[v_ax1, v_ax2] = A[v_ax1, v_ax2]
            for ax1, ax2 in T.grid(64, 64):
                with Ts.sblock("T_split_1"):
                    v_ax1, v_ax2 = Ts.axis.remap("SS", [ax1, ax2])
                    Ts.reads(A[v_ax1 + 64, v_ax2])
                    Ts.writes(T_split_1[v_ax1, v_ax2])
                    T_split_1[v_ax1, v_ax2] = A[v_ax1 + 64, v_ax2]

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            weight_packed: R.Tuple(
                R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
                R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
            ),
        ) -> R.DTensor((64, 128), "float32", "mesh[0]", "R"):
            cls = MLPWithTuple
            weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = weight_packed[0]
            lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.dist.call_tir_local_view(
                cls.matmul2,
                (x, weight1),
                ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "S[1]")],
            )
            lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.dist.call_tir_local_view(
                cls.gelu1, (lv0,), ty_args=[R.DTensor((128, 128), "float32", "mesh[0]", "S[1]")]
            )
            gv: R.Tuple(
                R.DTensor((64, 128), "float32", "mesh[0]", "S[1]"),
                R.DTensor((64, 128), "float32", "mesh[0]", "S[1]"),
            ) = R.dist.call_tir_local_view(
                cls.split11,
                (lv1,),
                ty_args=[
                    _TupleType(
                        [
                            R.DTensor((64, 128), "float32", "mesh[0]", "S[1]"),
                            R.DTensor((64, 128), "float32", "mesh[0]", "S[1]"),
                        ]
                    )
                ],
            )
            lv2: R.DTensor((64, 128), "float32", "mesh[0]", "S[1]") = gv[0]
            lv3: R.DTensor((64, 128), "float32", "mesh[0]", "S[1]") = lv2
            weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]") = weight_packed[1]
            gv_1: R.DTensor((64, 128), "float32", "mesh[0]", "R") = R.dist.call_tir_local_view(
                cls.matmul11,
                (lv3, weight2),
                ty_args=[R.DTensor((64, 128), "float32", "mesh[0]", "R")],
            )
            lv4: R.DTensor((64, 128), "float32", "mesh[0]", "R") = R.ccl.allreduce(
                gv_1, op_type="sum"
            )
            return lv4

    @I.ir_module(check_well_formed=False)
    class LoweredMLPWithTuple:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
        )

        @R.function
        def foo(
            x: R.Tensor((128, 128), dtype="float32"),
            weight_packed: R.Tuple(
                R.Tensor((128, 128), dtype="float32"), R.Tensor((128, 128), dtype="float32")
            ),
        ) -> R.Tensor((64, 128), dtype="float32"):
            cls = LoweredMLPWithTuple
            gv: R.Tensor((128, 128), dtype="float32") = R.ccl.broadcast_from_worker0(x)
            gv1: R.Tensor((128, 128), dtype="float32") = weight_packed[0]
            gv2: R.Tensor((128, 64), dtype="float32") = R.ccl.scatter_from_worker0(
                gv1, num_workers=2, axis=1
            )
            gv3: R.Tensor((128, 128), dtype="float32") = weight_packed[1]
            gv4: R.Tensor((64, 128), dtype="float32") = R.ccl.scatter_from_worker0(
                gv3, num_workers=2, axis=0
            )
            lv0 = R.call_tir(
                MLPWithTuple.get_global_var("matmul2"),
                (gv, gv2),
                ty_args=[R.Tensor((128, 64), dtype="float32")],
            )
            lv1 = R.call_tir(
                MLPWithTuple.get_global_var("gelu1"),
                (lv0,),
                ty_args=[R.Tensor((128, 64), dtype="float32")],
            )
            gv_1 = R.call_tir(
                MLPWithTuple.get_global_var("split11"),
                (lv1,),
                ty_args=[
                    _TupleType(
                        [R.Tensor((64, 64), dtype="float32"), R.Tensor((64, 64), dtype="float32")]
                    )
                ],
            )
            lv2: R.Tensor((64, 64), dtype="float32") = gv_1[0]
            lv3: R.Tensor((64, 64), dtype="float32") = lv2
            gv_1_1 = R.call_tir(
                MLPWithTuple.get_global_var("matmul11"),
                (lv3, gv4),
                ty_args=[R.Tensor((64, 128), dtype="float32")],
            )
            lv4: R.Tensor((64, 128), dtype="float32") = R.ccl.allreduce(gv_1_1, op_type="sum")
            return lv4

    for gv, func in MLPWithTuple.functions_items():
        if gv.name_hint != "foo":
            LoweredMLPWithTuple[gv] = func

    mod = MLPWithTuple
    mod = relax.distributed.transform.LowerDistIR()(mod)
    tvm_tirx.ir.assert_structural_equal(mod, LoweredMLPWithTuple)


if __name__ == "__main__":
    tvm_tirx.testing.main()
