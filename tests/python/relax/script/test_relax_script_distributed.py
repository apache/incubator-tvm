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
"""Relax script distributed."""

from __future__ import annotations

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import Call, SeqExpr, VarBinding
from tvm.relax import script as R
from tvm.relax.distributed import DeviceMesh
from tvm.s_tir import script as Ts
from tvm_tirx import IRModule
from tvm_tirx.ir import Range
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _check(
    parsed: relax.Function | IRModule,
    expect: relax.Function | IRModule | None = None,
):
    test = parsed.script(show_meta=True)
    roundtrip_mod = tvm_tirx.script.from_source(
        test,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "R": tvm.relax.script,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(parsed, roundtrip_mod)
    if expect:
        tvm_tirx.ir.assert_structural_equal(parsed, expect)


def test_call_tir_dtensor():
    @I.ir_module
    class TestModule:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2, 2), I.Range(0, 4)),  # mesh[0]
                    R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @Ts.function
        def tir_func(
            x: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            y: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock():
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"),
        ) -> R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"):
            gv0 = R.dist.call_tir(
                TestModule.tir_func,
                x,
                ty_args=[
                    R.DTensor(
                        shape=(128, 128),
                        dtype="float32",
                        device_mesh="mesh[0]",
                        placement="S[0], R",
                    )
                ],
            )
            return gv0

    device_mesh_list = [DeviceMesh((2, 2), Range(0, 4)), DeviceMesh((1,), Range(4, 5))]
    foo_func = TestModule["foo"]
    params = foo_func.params
    assert len(params) == 1
    assert params[0].ty == R.DTensor(
        (128, 128), "float32", device_mesh_list[0], placement="S[0], R"
    )
    assert foo_func.ret_ty == R.DTensor(
        (128, 128), "float32", device_mesh_list[0], placement="S[0], R"
    )
    assert isinstance(foo_func.body, SeqExpr)
    assert len(foo_func.body.blocks[0].bindings) == 1
    assert isinstance(foo_func.body.blocks[0].bindings[0], VarBinding)
    value = foo_func.body.blocks[0].bindings[0].value
    assert isinstance(value, Call)
    assert value.ty_args[0] == R.DTensor(
        (128, 128), "float32", device_mesh_list[0], placement="S[0], R"
    )
    _check(TestModule)


def test_explicit_device_id():
    @I.ir_module
    class TestModule:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2, 2), [0, 1, 2, 3]),  # mesh[0]
                    R.device_mesh(
                        (1,),
                        [
                            4,
                        ],
                    ),  # mesh[1]
                ]
            }
        )

        @Ts.function
        def tir_func(
            x: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            y: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock():
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"),
        ) -> R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"):
            gv0 = R.dist.call_tir(
                TestModule.tir_func,
                x,
                ty_args=[
                    R.DTensor(
                        shape=(128, 128),
                        dtype="float32",
                        device_mesh="mesh[0]",
                        placement="S[0], R",
                    )
                ],
            )
            return gv0

    _check(TestModule)


def test_constant():
    @I.ir_module
    class TestModule:
        I.module_attrs({"device_num": 10})
        I.module_global_infos(
            {
                "mesh": [
                    R.device_mesh((2, 2), I.Range(0, 4)),  # mesh[0]
                    R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
                ]
            }
        )

        @Ts.function
        def tir_func(
            x: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            y: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock():
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(
            x: R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"),
        ) -> R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"):
            gv0 = R.dist.call_tir(
                TestModule.tir_func,
                x,
                ty_args=[
                    R.DTensor(
                        shape=(128, 128),
                        dtype="float32",
                        device_mesh="mesh[0]",
                        placement="S[0], R",
                    )
                ],
            )
            gv1 = R.add(gv0, R.dist.const(1.0, ty=R.DTensor((), "float32", "mesh[0]", "R, R")))
            return gv1

    _check(TestModule)


if __name__ == "__main__":
    tvm_tirx.testing.main()
