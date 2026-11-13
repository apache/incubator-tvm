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
"""Relax script basic usage."""

from __future__ import annotations

import sys

import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax, topi
from tvm.relax import DummyGlobalInfo, VDevice
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx import IRModule, tirx
from tvm_tirx.ir import TupleType as _TupleType
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
    if isinstance(parsed, IRModule) and isinstance(roundtrip_mod, IRModule):
        relax.analysis.check_well_formed(parsed)
        relax.analysis.check_well_formed(roundtrip_mod)
    if expect:
        tvm_tirx.ir.assert_structural_equal(parsed, expect)


def test_simple_func():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
        R.func_attr({"Primitive": True})
        gv0 = R.call_dps_packed(
            _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
        )
        gv1 = R.call_dps_packed(
            _ExternFunc("extern_dps_func"), gv0, ty_args=[R.Tensor((128, 128), dtype="float32")]
        )
        return gv1

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), attrs={"Primitive": True}):
        y = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
        )
        out = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_dps_func"), y, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
        )
        bb.emit_func_output(out)

    _check(foo, bb.get()["foo"])


def test_simple_module():
    @I.ir_module
    class TestModule:
        @Ts.function(private=True)
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
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            cls = TestModule
            gv0 = R.call_tir(cls.tir_func, x, ty_args=[R.Tensor((128, 128), dtype="float32")])
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        out = bb.emit_te(lambda x: x + 1, x, function_name_hint="tir_func")
        bb.emit_func_output(out)

    _check(TestModule, bb.get())


def test_module_with_attr_and_global_info():
    @I.ir_module
    class TestModule:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "dummy": [
                    R.dummy_global_info(),  # dummy[0]
                    R.dummy_global_info(),  # dummy[1]
                ]
            }
        )

        @Ts.function(private=True)
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
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            cls = TestModule
            gv0 = R.call_tir(cls.tir_func, x, ty_args=[R.Tensor((128, 128), dtype="float32")])
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        out = bb.emit_te(lambda x: x + 1, x, function_name_hint="tir_func")
        bb.emit_func_output(out)
    mod = bb.get()
    mod.update_global_info("dummy", [DummyGlobalInfo(), DummyGlobalInfo()])
    mod = mod.with_attr("attr", 10)
    _check(TestModule, mod)


def test_global_info_vdevice():
    vdevices = [
        VDevice("llvm"),
        VDevice("cuda", 0),
        VDevice({"kind": "cuda", "arch": "sm_80"}, 0),
        VDevice("metal", 0, "global"),
    ]

    @I.ir_module
    class TestModule:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                    R.vdevice("cuda", 0),
                    R.vdevice({"kind": "cuda", "arch": "sm_80"}, 0),
                    R.vdevice("metal", 0, "global"),
                ]
            }
        )

        @Ts.function(private=True)
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
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            cls = TestModule
            gv0 = R.call_tir(cls.tir_func, x, ty_args=[R.Tensor((128, 128), dtype="float32")])
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        out = bb.emit_te(lambda x: x + 1, x, function_name_hint="tir_func")
        bb.emit_func_output(out)
    mod = bb.get()
    mod.update_global_info("vdevice", vdevices)
    mod = mod.with_attr("attr", 10)
    _check(TestModule, mod)


def test_relax_tensor_op():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")) -> R.Tensor((4, 4), "float32"):
        y = R.add(x, x)
        z = R.multiply(x, y)
        return z

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        y = bb.emit(relax.op.add(x, x))
        z = bb.emit(relax.op.multiply(x, y))
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_relax_base_op():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        alloc = R.builtin.alloc_tensor(R.shape([4, 4]), runtime_device_index=0, dtype="float32")
        shape = R.shape_of(alloc)
        return shape

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        alloc = bb.emit(relax.op.builtin.alloc_tensor(relax.ShapeExpr((4, 4)), "float32", 0))
        shape = bb.emit(relax.op.shape_of(alloc))
        bb.emit_func_output(shape)

    _check(foo, bb.get()["foo"])


def test_relax_shape_to_tensor():
    @R.function
    def foo(x: R.Shape((4, 4))):
        tensor = R.shape_to_tensor(x)
        return tensor

    x = relax.Var("x", R.Shape((4, 4)))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        tensor = bb.emit(relax.op.shape_to_tensor(x))
        bb.emit_func_output(tensor)

    _check(foo, bb.get()["foo"])


def test_tuple_return():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        gv0 = R.call_dps_packed(
            _ExternFunc("extern_func_0"), x, ty_args=[R.Tensor((4, 4), dtype="float32")]
        )
        gv1 = R.call_dps_packed(
            _ExternFunc("extern_func_1"), x, ty_args=[R.Tensor((4, 4), dtype="float32")]
        )
        return (gv0, gv1)

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        gv0 = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_func_0"), x, ty_args=[R.Tensor((4, 4), dtype="float32")]
            )
        )
        gv1 = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_func_1"), x, ty_args=[R.Tensor((4, 4), dtype="float32")]
            )
        )
        bb.emit_func_output(relax.Tuple((gv0, gv1)))

    _check(foo, bb.get()["foo"])


def test_tuple_return_2():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @R.function
    def foo(x: R.Tensor("float32", ndim=2)):
        x0 = R.match_cast(x, R.Tensor((n, m), "float32"))
        return (x0, R.shape([n + 1, m, 1]))

    x = relax.Var("x", R.Tensor("float32", ndim=2))
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        x0 = bb.match_cast(x, R.Tensor((n, m), "float32"))
        bb.emit_func_output(relax.Tuple([x0, relax.ShapeExpr([n + 1, m, 1])]))

    _check(foo, bb.get()["foo"])


def test_tuple_binding():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @R.function
    def foo(x: R.Tensor("float32", ndim=2)):
        x0 = R.match_cast(x, R.Tensor((n, m), "float32"))
        t0 = (x, x0)
        t1 = (x, R.shape([n, m]), t0)
        return t1

    x = relax.Var("x", R.Tensor("float32", ndim=2))
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        x0 = bb.match_cast(x, R.Tensor((n, m), "float32"))
        t0 = bb.emit(relax.Tuple([x, x0]))
        t1 = bb.emit(relax.Tuple([x, relax.ShapeExpr([n, m]), t0]))
        bb.emit_func_output(t1)

    _check(foo, bb.get()["foo"])


def test_tuple_get_item():
    @R.function
    def foo(x: R.Tensor, y: R.Tensor):
        t1 = (x, y)
        t2 = (x, y)
        a = t1[0]
        b = R.TupleGetItem(t2, 1)
        c = R.add(a, b)
        return c

    x = relax.Var("x", R.Tensor())
    y = relax.Var("y", R.Tensor())
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        t1 = bb.emit(relax.Tuple([x, y]))
        t2 = bb.emit(relax.Tuple([x, y]))
        a = bb.emit(relax.TupleGetItem(t1, 0))
        b = bb.emit(relax.TupleGetItem(t2, 1))
        c = bb.emit(relax.op.add(a, b))
        bb.emit_func_output(c)

    _check(foo, bb.get()["foo"])


def test_dataflow_block():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        with R.dataflow():
            lv0 = R.call_dps_packed(
                _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
            lv1 = R.call_dps_packed(
                _ExternFunc("extern_func"), lv0, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
            gv = lv1
            R.output(gv)
        return gv

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        with bb.dataflow():
            lv0 = bb.emit(
                relax.call_dps_packed(
                    _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
            )
            lv1 = bb.emit(
                relax.call_dps_packed(
                    _ExternFunc("extern_func"), lv0, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
            )
            gv = bb.emit_output(lv1)
        bb.emit_func_output(gv)

    _check(foo, bb.get()["foo"])


def test_dataflow_block_advanced():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv0 = R.call_dps_packed(
            _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
        )
        gv1 = R.call_dps_packed(
            _ExternFunc("extern_func"), gv0, ty_args=[R.Tensor((128, 128), dtype="float32")]
        )
        with R.dataflow():
            lv0 = R.call_dps_packed(
                _ExternFunc("extern_func"), gv1, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
            lv1 = R.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv2 = R.call_dps_packed(
                _ExternFunc("extern_func"), lv0, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
            gv2 = R.call_dps_packed(
                _ExternFunc("extern_func"), gv2, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
            gv3 = R.match_cast(gv2, R.Tensor((m, n), "float32"))
            gv3 = R.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv4 = gv3
            gv5 = gv2
            R.output(gv5, gv4)
        gv6 = R.call_dps_packed(
            _ExternFunc("extern_func"), gv5, ty_args=[R.Tensor((128, 128), dtype="float32")]
        )
        gv7 = R.call_dps_packed(
            _ExternFunc("extern_func"), gv6, ty_args=[R.Tensor((128, 128), dtype="float32")]
        )
        return gv7

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    m = tirx.Var("m", ty="int64")
    n = tirx.Var("n", ty="int64")
    with bb.function("foo", (x,)):
        gv0 = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
        )
        gv1 = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_func"), gv0, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
        )
        with bb.dataflow():
            lv0 = bb.emit(
                relax.call_dps_packed(
                    _ExternFunc("extern_func"), gv1, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
            )
            lv1 = bb.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv2 = bb.emit(
                relax.call_dps_packed(
                    _ExternFunc("extern_func"), lv0, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
            )
            gv21 = bb.emit(
                relax.call_dps_packed(
                    _ExternFunc("extern_func"), gv2, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
            )
            gv3 = bb.match_cast(gv21, R.Tensor((m, n), "float32"))
            gv31 = bb.match_cast(lv0, R.Tensor((m, n), "float32"))
            gv32 = bb.emit_output(gv31)
            gv22 = bb.emit_output(gv21)
        gv4 = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_func"), gv22, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
        )
        gv5 = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_func"), gv4, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
        )
        bb.emit_func_output(gv5)

    _check(foo, bb.get()["foo"])


def test_return_without_binding():
    @R.function
    def foo(x: R.Tensor((128, 128), "float32")):
        return x

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        bb.emit_func_output(x)

    _check(foo, bb.get()["foo"])


def test_tensor_type_without_args():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        v = R.call_dps_packed(
            _ExternFunc("extern_relu"), x, ty_args=[R.Tensor((32, 32), dtype="float32")]
        )
        return v

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        v = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("extern_relu"), x, ty_args=[R.Tensor((32, 32), dtype="float32")]
            )
        )
        bb.emit_func_output(v)

    _check(foo, bb.get()["foo"])


def test_tensor_with_vdevice():
    vdevices = [
        VDevice("llvm"),
        VDevice("cuda", 0),
        VDevice("metal", 0, "global"),
        VDevice({"kind": "cuda", "arch": "sm_80"}, 0),
    ]

    @I.ir_module
    class TestModule:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                    R.vdevice("cuda", 0),
                    R.vdevice("metal", 0, "global"),
                    R.vdevice({"kind": "cuda", "arch": "sm_80"}, 0),
                ]
            }
        )

        @R.function
        def foo(
            a: R.Tensor((128, 128), "float32", "cuda:1"),
            b: R.Tensor((128, 128), "float32", "llvm"),
            c: R.Tensor((128, 128), "float32", "vdevice:3"),
        ) -> R.Tensor((128, 128), "float32", "cuda:1"):
            s = R.add(a, c)
            return s

    a = relax.Var("a", R.Tensor((128, 128), "float32", vdevices[3]))
    b = relax.Var("b", R.Tensor((128, 128), "float32", vdevices[0]))
    c = relax.Var("c", R.Tensor((128, 128), "float32", vdevices[3]))
    bb = relax.BlockBuilder()
    with bb.function("foo", (a, b, c)):
        out = bb.emit(relax.op.add(a, c))
        bb.emit_func_output(out)
    mod = bb.get()
    mod = mod.with_attr("attr", 10)
    mod.update_global_info("vdevice", vdevices)

    _check(TestModule, mod)


def test_direct_return():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor((32, 32), "float32"):
        return x

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        bb.emit_func_output(x)

    _check(foo, bb.get()["foo"])


def test_call_packed():
    @R.function(pure=False)
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        z = R.call_packed("vm.builtin.copy", x, ty_args=R.Tensor((32, 32), "float32"))
        return z

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x), pure=False):
        z = bb.emit(
            relax.Call(
                relax.ExternFunc("vm.builtin.copy"),
                (x,),
                None,
                ty_args=[R.Tensor((32, 32), "float32")],
            )
        )
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_call_packed_without_ty_args():
    @R.function(pure=False)
    def foo(x: R.Any) -> R.Any:
        z = R.call_packed("test", x)
        return z

    x = relax.Var("x", R.Any())
    bb = relax.BlockBuilder()
    with bb.function("foo", (x), pure=False):
        z = bb.emit(
            relax.Call(
                relax.ExternFunc("test"),
                (x,),
                None,
                ty_args=[],
            )
        )
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_object_proxy_compat_alias():
    @R.function
    def foo(x: R.Object) -> R.Object:
        return x

    assert isinstance(foo.ret_ty, relax.AnyType)


def test_annotation():
    m = T.dynamic("m", "int64")

    @R.function(pure=False)
    def foo(
        x: R.Tensor((32, m), "float32"),
        y: R.Tensor((m,), "float32"),
        r: R.Tensor(dtype="int64"),
    ) -> R.Any:
        z: R.Tensor((32, m), "float32") = R.multiply(x, y)
        w: R.Tensor(ndim=2) = R.multiply(z, z)
        q: R.Tensor = R.add(w, w)
        t = R.add(w, z)
        sh: R.Shape = R.call_packed("shape_of", x, ty_args=R.Shape)
        lv: R.Tensor(sh, dtype="float32") = R.reshape(x, sh)
        o: R.Any = R.call_packed("contrib.tensor_array_stack", x, y, ty_args=R.Any)
        return o

    def _check_ty(binding, expected_ty):
        tvm_tirx.ir.assert_structural_equal(binding.var.ty, expected_ty)
        tvm_tirx.ir.assert_structural_equal(binding.value.ty, expected_ty)

    # Cannot use block builder here because we need to check the annotated type,
    # which may be inconsistent with deduced type.
    assert isinstance(foo.ret_ty, relax.AnyType)
    m = relax.get_shape_of(foo.params[0])[1]
    bindings = foo.body.blocks[0].bindings
    sh = bindings[4].var

    _check_ty(bindings[0], relax.TensorType([32, m], "float32"))
    _check_ty(bindings[1], relax.TensorType(dtype=None, ndim=2))
    _check_ty(bindings[2], relax.TensorType(dtype=None, ndim=-1))
    _check_ty(bindings[3], relax.TensorType(dtype=None, ndim=2))
    _check_ty(bindings[4], relax.ShapeType(ndim=-1))
    _check_ty(bindings[5], relax.TensorType(sh))
    _check_ty(bindings[6], relax.AnyType())


def test_call_dps_packed_empty_shape():
    @R.function
    def foo(x: R.Tensor((), "float32")):
        z = R.call_dps_packed(_ExternFunc("scalar_add"), x, ty_args=[R.Tensor((), dtype="float32")])
        return z

    (z_bind,) = foo.body.blocks[0].bindings
    shape_expr = z_bind.value.ty_args[0].shape

    assert isinstance(shape_expr, relax.ShapeExpr)
    assert len(shape_expr.values) == 0


def test_call_tir_empty_tuple_arg():
    bb = relax.BlockBuilder()
    dummy_param = relax.Var("dummy_param", R.Tensor(()))
    with bb.function("foo", [dummy_param], {"global_symbol": "foo"}):
        output = bb.emit_te(topi.full, shape=(16, 32), dtype="float32", fill_value=1.0)
        bb.emit_func_output(output)

    _check(bb.get())


def test_call_tir_with_grad():
    @I.ir_module
    class Module:
        @Ts.function
        def identity_tir(
            A: T.Tensor([54, 96], "float32"), B: T.Tensor([54, 96], "float32")
        ) -> None:
            for i, j in T.grid(54, 96):
                with Ts.sblock("compute"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj]

        @R.function
        def main(v0: R.Tensor([54, 96], "float32")):
            cls = Module
            out = R.call_tir_with_grad(
                cls.identity_tir,
                (v0,),
                ty_args=[R.Tensor((54, 96), "float32")],
                te_grad_name="identity_k_grad",
                te_grad_kwargs={"k": 1.0},
            )
            return out

    _check(Module)


def test_call_tir_inplace():
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def copy(
            A: T.Tensor((2, 3), "int32"),
            B: T.Tensor((2, 3), "int32"),
            out1: T.Tensor((2, 3), "int32"),
        ):
            # copies the contents of B into A and out1
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(B[ax0, ax1])
                    Ts.writes(A[ax0, ax1], out1[ax0, ax1])
                    A[ax0, ax1] = B[ax0, ax1]
                    out1[ax0, ax1] = B[ax0, ax1]

        @R.function
        def main(x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")) -> R.Tuple(
            R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")
        ):
            res = R.call_tir_inplace(
                Module.copy,
                (x, y),
                inplace_indices=[0, -1],
                ty_args=[_TupleType([R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")])],
            )
            return res

    _check(Module)


def test_if_branch():
    @R.function
    def foo(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")) -> R.Tensor((1,), "float32"):
        if cond:
            w = R.add(x, x)
            y = R.multiply(w, w)
        else:
            w = R.multiply(x, x)
            y = R.add(w, w)
        return y

    cond, x = foo.params
    y_bind = foo.body.blocks[0].bindings[0]
    y, ite = y_bind.var, y_bind.value

    assert isinstance(y, relax.Var)
    assert y.name == "y"

    assert isinstance(ite, relax.IfExpr)
    assert isinstance(ite.true_branch, relax.SeqExpr)
    assert isinstance(ite.false_branch, relax.SeqExpr)

    def check_call(call, op, args):
        assert isinstance(call, relax.Call)
        if isinstance(op, str):
            assert call.op.name == op
        else:
            assert call.op == op
        tvm_tirx.ir.assert_structural_equal(call.args, args)

    w_bind = ite.true_branch.blocks[0].bindings[0]
    # the seq exprts in the branches are normalized to bind any call
    # in the seq expr "body" to a var
    y_bind = ite.true_branch.blocks[-1].bindings[-1]
    assert w_bind.var.name == "w"
    check_call(w_bind.value, "relax.add", [x, x])
    check_call(y_bind.value, "relax.multiply", [w_bind.var, w_bind.var])

    w_bind = ite.false_branch.blocks[0].bindings[0]
    y_bind = ite.false_branch.blocks[-1].bindings[-1]
    assert w_bind.var.name == "w"
    check_call(w_bind.value, "relax.multiply", [x, x])
    check_call(y_bind.value, "relax.add", [w_bind.var, w_bind.var])


def test_scalar_tensor_as_branch_condition():
    """Branch condition can be 0-d tensor"""

    @R.function
    def func(cond: R.Tensor([], "bool"), x: R.Tensor((1,), "float32")):
        if cond:
            out = R.add(x, x)
        else:
            out = R.multiply(x, x)
        return out

    if_else = func.body.blocks[0].bindings[0].value
    assert isinstance(if_else.cond, relax.Var)
    tvm_tirx.ir.assert_structural_equal(if_else.cond.ty, R.Tensor([], "bool"))


def test_prim_value_as_branch_condition():
    """In addition to scalar tensor, can use a primitive scalar condition"""

    @R.function
    def func(cond: T.bool, x: R.Tensor((1,), "float32")):
        if cond:
            out = R.add(x, x)
        else:
            out = R.multiply(x, x)
        return out

    if_else = func.body.blocks[0].bindings[0].value
    assert isinstance(if_else.cond, relax.Var)
    tvm_tirx.ir.assert_structural_equal(if_else.cond.ty, tvm_tirx.ir.PrimType("bool"))


def test_computed_prim_value_as_branch_condition():
    """The primitive scalar condition may be computed within the function"""

    N = T.dynamic("N", "int64")

    @R.function
    def func(x: R.Tensor([N], "float32")):
        if R.prim_value(N % 16 == 0):
            out = R.call_pure_packed("fast_vectorized_impl", x, ty_args=[x.ty])
        else:
            out = R.call_pure_packed("slow_non_vectorized_impl", x, ty_args=[x.ty])
        return out

    N = func.params[0].ty.shape[0]
    if_else = func.body.blocks[0].bindings[0].value
    assert tvm_tirx.ir.is_prim_expr(if_else.cond)
    tvm_tirx.ir.assert_structural_equal(N % 16 == 0, if_else.cond)
    tvm_tirx.ir.assert_structural_equal(if_else.cond.ty, tvm_tirx.ir.PrimType("bool"))


def test_tir_expr_as_branch_condition():
    """Syntactic sugar, use Expr directly"""

    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def sugared(x: R.Tensor([N], "float32")):
        if N % 16 == 0:
            out = R.call_pure_packed("fast_vectorized_impl", x, ty_args=[x.ty])
        else:
            out = R.call_pure_packed("slow_non_vectorized_impl", x, ty_args=[x.ty])
        return out

    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def unsugared(x: R.Tensor([N], "float32")):
        if R.prim_value(N % 16 == 0):
            out = R.call_pure_packed("fast_vectorized_impl", x, ty_args=[x.ty])
        else:
            out = R.call_pure_packed("slow_non_vectorized_impl", x, ty_args=[x.ty])
        return out

    tvm_tirx.ir.assert_structural_equal(unsugared, sugared)


def test_scalar_tensor_as_assert_condition():
    """Branch condition can be 0-d tensor"""

    @R.function(pure=False)
    def func(cond: R.Tensor([], "bool"), x: R.Tensor((1,), "float32")):
        _ = R.assert_op(cond)
        out = R.add(x, x)
        return out

    assert_op = func.body.blocks[0].bindings[0].value
    condition = assert_op.args[0]
    assert isinstance(condition, relax.Var)
    tvm_tirx.ir.assert_structural_equal(condition.ty, R.Tensor([], "bool"))


def test_prim_value_as_assert_condition():
    """In addition to scalar tensor, can use a primitive scalar condition"""

    @R.function(pure=False)
    def func(cond: T.bool, x: R.Tensor((1,), "float32")):
        _ = R.assert_op(cond)
        out = R.add(x, x)
        return out

    assert_op = func.body.blocks[0].bindings[0].value
    condition = assert_op.args[0]
    assert isinstance(condition, relax.Var)
    tvm_tirx.ir.assert_structural_equal(condition.ty, tvm_tirx.ir.PrimType("bool"))


def test_computed_prim_value_as_assert_condition():
    """The primitive scalar condition may be computed within the function"""

    N = T.dynamic("N", "int64")

    @R.function(pure=False)
    def func(x: R.Tensor([N], "float32")):
        _ = R.assert_op(R.prim_value(N % 16 == 0))
        out = R.call_packed("fast_vectorized_impl", x, ty_args=[x.ty])
        return out

    N = func.params[0].ty.shape[0]
    assert_op = func.body.blocks[0].bindings[0].value
    condition = assert_op.args[0]
    assert tvm_tirx.ir.is_prim_expr(condition)
    tvm_tirx.ir.assert_structural_equal(N % 16 == 0, condition)
    tvm_tirx.ir.assert_structural_equal(condition.ty, tvm_tirx.ir.PrimType("bool"))


def test_tir_expr_as_assert_condition():
    """Syntactic sugar, use Expr directly"""

    N = T.dynamic("N", "int64")

    @R.function(pure=False, private=True)
    def sugared(x: R.Tensor([N], "float32")):
        _ = R.assert_op(N % 16 == 0)
        out = R.call_packed("fast_vectorized_impl", x, ty_args=[x.ty])
        return out

    N = T.dynamic("N", "int64")

    @R.function(pure=False, private=True)
    def unsugared(x: R.Tensor([N], "float32")):
        _ = R.assert_op(R.prim_value(N % 16 == 0))
        out = R.call_packed("fast_vectorized_impl", x, ty_args=[x.ty])
        return out

    tvm_tirx.ir.assert_structural_equal(unsugared, sugared)


def test_empty_tuple():
    @R.function
    def foo(x: R.Tuple()):
        y: R.Tuple() = ()
        return y

    x = relax.Var("x", relax.TupleType([]))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        y = bb.emit(relax.Tuple([]))
        bb.emit_func_output(y)

    _check(foo, bb.get()["foo"])


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_arith_operators():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor((m, n), "float32"), y: R.Tensor((m, n), "float32")):
        a0 = -x
        a1 = x + y
        a2 = x - y
        a3 = x * y
        a4 = x / y
        a5 = x // y
        a6 = x**y

        c0 = x > y
        c1 = x < y
        c2 = x >= y
        c3 = x <= y

        tuple_expr = ((x, x), y)
        t0 = tuple_expr[0]
        t1 = tuple_expr[1]
        t2 = tuple_expr[0][0]  # <= Will normalize to two bindings
        return (a0, a1, a2, a3, a4, a5, a6, c0, c1, c2, c3, t0, t1, t2)

    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x = relax.Var("x", relax.TensorType([m, n], "float32"))
    y = relax.Var("y", relax.TensorType([m, n], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        a0 = bb.emit(relax.op.negative(x))
        a1 = bb.emit(relax.op.add(x, y))
        a2 = bb.emit(relax.op.subtract(x, y))
        a3 = bb.emit(relax.op.multiply(x, y))
        a4 = bb.emit(relax.op.divide(x, y))
        a5 = bb.emit(relax.op.floor_divide(x, y))
        a6 = bb.emit(relax.op.power(x, y))

        c0 = bb.emit(relax.op.greater(x, y))
        c1 = bb.emit(relax.op.less(x, y))
        c2 = bb.emit(relax.op.greater_equal(x, y))
        c3 = bb.emit(relax.op.less_equal(x, y))

        tuple_expr = bb.emit(relax.Tuple((relax.Tuple((x, x)), y)))
        t0 = bb.emit(relax.TupleGetItem(tuple_expr, 0))
        t1 = bb.emit(relax.TupleGetItem(tuple_expr, 1))
        tmp = bb.emit(relax.TupleGetItem(tuple_expr, 0))
        t2 = bb.emit(relax.TupleGetItem(tmp, 0))
        bb.emit_func_output(relax.Tuple((a0, a1, a2, a3, a4, a5, a6, c0, c1, c2, c3, t0, t1, t2)))

    _check(foo, bb.get()["foo"])


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_memory_ops():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor((m, n), dtype="float32")):
        storage = R.memory.alloc_storage(
            R.shape([4 * m * n]), virtual_device_index=0, storage_scope="global", dtype="float32"
        )
        alloc = R.memory.alloc_tensor(storage, offset=0, shape=R.shape([m, n]), dtype="float32")
        tensor = R.builtin.alloc_tensor(R.shape([m, n]), dtype="float32", runtime_device_index=0)
        gv = tensor
        return alloc, gv

    _check(foo)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_vm_ops():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function(pure=False)
    def foo(x: R.Tensor((m, n), dtype="float32")):
        storage = R.vm.alloc_storage(R.shape([4 * m * n]), runtime_device_index=0, dtype="uint8")
        alloc = R.vm.alloc_tensor(storage, offset=0, shape=R.shape([m, n]), dtype="float32")
        tensor = R.builtin.alloc_tensor(R.shape([m, n]), dtype="float32", runtime_device_index=0)
        tir_dym = R.vm.call_tir_dyn("te_func", (x, tensor, R.ShapeExpr((m, n))))
        return alloc, tir_dym

    _check(foo)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_builtin_ops():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor((m, n), dtype="float32")):
        tensor = R.builtin.stop_lift_params(x)
        gv = tensor
        return gv

    _check(foo)


def test_prim_value():
    @R.function(pure=False)
    def foo():
        gv = R.call_packed("test", 1, ty_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_string_imm():
    @R.function(pure=False)
    def foo():
        gv = R.call_packed("test", "hello", ty_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_datatype_imm():
    @R.function(pure=False)
    def foo():
        gv = R.call_packed("test", R.dtype("float32"), ty_args=R.Tensor((32, 32), "float32"))
        return gv

    _check(foo)


def test_function_void_return_type():
    @tvm_tirx.script.ir_module
    class Foo:
        @R.function
        def main(x: R.Tensor((3, 3), dtype="float32")):
            res = Foo.mul(x)
            return res

        @R.function
        def mul(x: R.Tensor((3, 3), dtype="float32")):
            res = R.multiply(x, x)
            return res

    _check(Foo)
    # Since the return type of function `mul` is not annotated,
    # the function `main` regards it as a generic return type.
    assert isinstance(Foo["main"].ret_ty, relax.AnyType)
    assert isinstance(Foo["mul"].ret_ty, relax.TensorType)

    @tvm_tirx.script.ir_module
    class Bar:
        @R.function
        def main(x1: R.Tensor((3, 3), dtype="float32")):
            res1 = Bar.mul(x1)
            return res1

        @R.function
        def mul(x: R.Tensor((3, 3), dtype="float32")) -> None:
            res = R.multiply(x, x)
            return res

    # Since the return type of function `mul` is not annotated,
    # the function `main` regards it as a generic return type.
    _check(Bar)
    tvm_tirx.ir.assert_structural_equal(Bar["main"].ret_ty, relax.TupleType([]))
    tvm_tirx.ir.assert_structural_equal(Bar["mul"].ret_ty, relax.TupleType([]))


def test_unit_tuple_on_rhs_of_assign():
    @I.ir_module
    class Module:
        @R.function
        def main(input: R.Tensor((5, 5))) -> R.Tuple(R.Tensor((5, 5))):
            gv = (input,)
            return gv

    _check(Module)


def test_empty_tuple_on_rhs_of_assign():
    @I.ir_module
    class Module:
        @R.function
        def main(input: R.Tensor((5, 5))) -> R.Tuple():
            gv = ()
            return gv

    _check(Module)


def test_global_var_ty():
    @I.ir_module
    class Module:
        @R.function
        def foo(x: R.Tensor((128, 128), "float32")):
            gv0 = R.emit_te(topi.add, x, x)
            return gv0

    target_ty = R.Callable(
        (R.Tensor((128, 128), dtype="float32"),), R.Tensor((128, 128), dtype="float32")
    )
    gv = Module.get_global_var("foo")
    tvm_tirx.ir.assert_structural_equal(gv.ty, target_ty)
    tvm_tirx.ir.assert_structural_equal(Module["foo"].ty, target_ty)
    _check(Module)


def test_assert_op():
    @I.ir_module
    class AssertOp:
        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.assert_op(R.const(False, dtype="bool"), "x: {}", x)
            return x

    _check(AssertOp)


def test_assert_outside_of_class():
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        y = R.assert_op(R.const(False, dtype="bool"), "x: {}", x)
        return x

    # this just makes sure that the machinery regarding the pure attribute parses
    # in the case where the function is outside of a class too
    _check(func)


def test_impure_inner_function():
    @R.function
    def f(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
        # we will not actually call it
        @R.function(pure=False)
        def g(y: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            z = R.assert_op(R.const(False, dtype="bool"), "y: {}", y)
            return y

        return x

    assert f.is_pure
    # definition of g
    assert not f.body.blocks[0].bindings[0].value.is_pure

    # make sure we are not incorrectly passing state for inner functions
    _check(f)


def test_impure_inner_function_in_class():
    @I.ir_module
    class ImpureInner:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            # we will not actually call it
            @R.function(pure=False)
            def g(y: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                z = R.assert_op(R.const(False, dtype="bool"), "y: {}", y)
                return y

            return x

    assert ImpureInner["main"].is_pure
    # definition of g
    assert not ImpureInner["main"].body.blocks[0].bindings[0].value.is_pure

    # make sure we are not incorrectly passing state for inner functions
    _check(ImpureInner)


def test_print():
    @I.ir_module
    class Print:
        @R.function(pure=False)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.print("x: {}", x)
            return x

    _check(Print)


def test_parse_multiple_pure_and_impure_funcs():
    @I.ir_module
    class Mixture:
        @R.function(pure=False)
        def print(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.print("x: {}", x)
            return x

        @R.function(pure=False)
        def assert_func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.assert_op(R.const(False, dtype="bool"), "x: {}", x)
            return x

        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            return x

    assert not Mixture["print"].is_pure
    assert not Mixture["assert_func"].is_pure
    assert Mixture["main"].is_pure
    _check(Mixture)


def test_function_with_void_return_type_may_be_used_as_statements():
    """Void return of calls do not need to be assigned"""

    @I.ir_module
    class Unsugared:
        @R.function(pure=False)
        def print(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.print("x: {}", x)
            return x

        @R.function(pure=False)
        def assert_func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.assert_op(R.const(False, dtype="bool"), "x: {}", x)
            return x

    @I.ir_module
    class Sugared:
        @R.function(pure=False)
        def print(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.print("x: {}", x)
            return x

        @R.function(pure=False)
        def assert_func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.assert_op(R.const(False, dtype="bool"), "x: {}", x)
            return x

    tvm_tirx.ir.assert_structural_equal(Unsugared, Sugared)


def test_function_with_void_return_type_in_if_else():
    """Last statement in if/else may be a void return"""

    @I.ir_module
    class Unsugared:
        @R.function(pure=False)
        def conditional(x: R.Tensor((), "int32"), condition: R.Tensor((), "bool")) -> R.Tensor(
            (), "int32"
        ):
            if condition:
                y = R.print("True condition: {}", x)
            else:
                y = R.print("False condition: {}", x)
            return x

    @I.ir_module
    class Sugared:
        @R.function(pure=False)
        def conditional(x: R.Tensor((), "int32"), condition: R.Tensor((), "bool")) -> R.Tensor(
            (), "int32"
        ):
            if condition:
                R.print("True condition: {}", x)
            else:
                R.print("False condition: {}", x)
            return x

    _check(Sugared, Unsugared)


def test_call_pure_packed():
    @R.function
    def foo(x: R.Tensor((32, 32), "float32")) -> R.Tensor:
        z = R.call_pure_packed("vm.builtin.copy", x, ty_args=R.Tensor((32, 32), "float32"))
        return z

    x = relax.Var("x", R.Tensor((32, 32), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x)):
        z = bb.emit(
            R.call_pure_packed("vm.builtin.copy", x, ty_args=[R.Tensor((32, 32), "float32")])
        )
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_call_pure_packed_returning_object():
    @R.function
    def foo() -> R.Any:
        z = R.call_pure_packed("dummy_func", ty_args=R.Any)
        return z

    bb = relax.BlockBuilder()
    with bb.function("foo", params=[]):
        z = bb.emit(R.call_pure_packed("dummy_func", ty_args=[relax.AnyType()]))
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_private_function():
    @I.ir_module
    class Addition:
        @R.function(private=True)
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            return y

    x = relax.Var("x", R.Tensor((), "int32"))
    bb = relax.BlockBuilder()
    with bb.function("main", (x), private=True):
        y = bb.emit(R.add(x, x))
        bb.emit_func_output(y)

    _check(Addition, bb.get())


def test_function_attributes_are_defined():
    """func.attrs defaults to an empty DictAttrs"""

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    main_m = T.dynamic("m", "int64")
    main_n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor, shape: R.Shape([main_m, main_n])):
            output = Module.subroutine(x, shape)
            return output

        @R.function
        def subroutine(x: R.Tensor, _: R.Shape([m, n])) -> R.Tensor([m, n]):
            q = x
            z = R.match_cast(q, R.Tensor((m, n)))
            w = z
            return w

    for gvar, func in Module.functions.items():
        assert func.attrs is not None


def test_primitive_assignments_emit_fresh_bindings():
    """Primitive expressions follow ordinary Relax assignment semantics."""

    @R.function(private=True)
    def func(scalar: T.int64, tensor: R.Tensor([1], "float32")):
        alias = scalar
        arithmetic = alias + 1
        integer: T.int32 = 2
        floating: T.float32 = 2.5
        boolean: T.bool = True
        compatibility = R.prim_value(arithmetic)
        tensor_alias = tensor
        return tensor_alias

    bindings = func.body.blocks[0].bindings
    assert len(bindings) == 7
    for binding in bindings:
        assert isinstance(binding, relax.VarBinding)
        assert not binding.var.same_as(binding.value)

    alias, arithmetic, integer, floating, boolean, compatibility, tensor_alias = bindings
    assert alias.value.same_as(func.params[0])
    assert compatibility.value.same_as(arithmetic.var)
    assert tensor_alias.value.same_as(func.params[1])
    for binding, dtype in [
        (integer, "int32"),
        (floating, "float32"),
        (boolean, "bool"),
    ]:
        assert isinstance(binding.var.ty, tvm_tirx.ir.PrimType)
        assert binding.var.ty.dtype == dtype
        assert tvm_tirx.ir.is_prim_expr(binding.value)
        assert not isinstance(binding.value, tvm_tirx.ir.GenericConst)

    source = func.script(extra_config={"relax.show_all_ty": False})
    assert "R.prim_value" not in source
    for annotation in [
        "alias: T.int64",
        "integer: T.int32",
        "floating: T.float32",
        "boolean: T.bool",
    ]:
        assert annotation in source
    _check(func)


def test_primitive_if_emits_fresh_result():
    """A primitive If has a fresh Relax result and typed branch terminators."""

    @R.function(private=True)
    def func(cond: T.bool, lhs: T.int64, rhs: T.int64):
        if cond:
            output = lhs
        else:
            output = rhs
        return output

    outer_binding = func.body.blocks[0].bindings[0]
    assert isinstance(outer_binding, relax.VarBinding)
    assert isinstance(outer_binding.value, relax.IfExpr)
    assert isinstance(outer_binding.var.ty, tvm_tirx.ir.PrimType)
    assert not outer_binding.var.same_as(func.params[1])
    assert not outer_binding.var.same_as(func.params[2])
    for branch, param in [
        (outer_binding.value.true_branch, func.params[1]),
        (outer_binding.value.false_branch, func.params[2]),
    ]:
        assert not branch.blocks
        assert branch.body.same_as(param)

    source = func.script(extra_config={"relax.show_all_ty": False})
    assert "R.prim_value" not in source
    assert source.count("output: T.int64") == 2
    _check(func)


def test_return_from_dataflow_block():
    """Return statements imply

    The `R.output` statement in a `R.dataflow()` block marks a
    variable that should be a `relax.Var` instead of a
    `relax.DataflowVar`, allowing it to be used outside of the
    `DataflowBlock` that defined it.  A relax function's output is not
    part of any binding, and must not contain any `DataflowVar`, so
    these are exposed implicitly.

    """

    @R.function(private=True)
    def output_then_return(A: R.Tensor([16], "float16")):
        with R.dataflow():
            B = R.add(A, A)
            C = R.multiply(B, B)
            R.output(C)

        return C

    @R.function(private=True)
    def return_inside_dataflow(A: R.Tensor([16], "float16")):
        with R.dataflow():
            B = R.add(A, A)
            C = R.multiply(B, B)
            return C

    tvm_tirx.ir.assert_structural_equal(output_then_return, return_inside_dataflow)


def relax_match_cast_ty_proxy():
    """Default type constructors may be used as expressions

    This is a regression test.  The TVMScript parser allows Type
    to be specified using a default-constructible class
    (e.g. `R.Tensor` or `R.Shape`) rather than an instance of that
    class (e.g. `R.Tensor()` or `R.Shape()`).  In previous
    implementations, this was only handled when the `Type` was
    used in an annotation context.  However, a `Type` may also
    appear as an argument, which is passed to `R.match_cast`.  Use of
    a default-constructible class must be handled in this context as
    well.
    """

    def make_ir_generator(proxy_subclass):
        def inner():
            @R.function
            def func(A: R.Any):
                B = R.match_cast(A, proxy_subclass)
                return B

            return func

        inner.__name__ = subclass.__name__
        return inner

    # Prim and DTensor require arguments; the remaining public type
    # constructors also work as bare values in match_cast expressions.
    subclasses = [R.Any, R.Tensor, R.Callable, R.Tuple, R.Shape]

    for subclass in subclasses:
        yield make_ir_generator(subclass)


def relax_float_symbolic_var():
    """Relax scalar variables may use any dtype."""

    @R.function
    def func(value: T.float16):
        return value

    return func


@pytest.mark.parametrize(
    "ir_generator",
    [*relax_match_cast_ty_proxy(), relax_float_symbolic_var],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_basic_usage(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
            "Ts": tvm.s_tir.script,
            "R": tvm.relax.script,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


def test_tuple_literals_preserve_expression_structure():
    @R.function(private=True)
    def literal(x: T.int64):
        empty = ()
        singleton = (x,)
        nested = (empty, singleton)
        return nested

    @R.function(private=True)
    def constructed(x: T.int64):
        empty = R.tuple()
        singleton = R.tuple(x)
        nested = R.tuple(empty, singleton)
        return nested

    tvm_tirx.ir.assert_structural_equal(literal, constructed)
    for show_all_ty in [False, True]:
        source = literal.script(extra_config={"relax.show_all_ty": show_all_ty})
        assert "R.tuple" not in source
        tvm_tirx.ir.assert_structural_equal(
            literal, tvm_tirx.script.from_source(source, extra_vars={"I": I, "R": R, "T": T})
        )


def test_primitive_scalar_conversion_depends_on_context():
    @R.function(private=True)
    def tensor():
        return 1

    @R.function(private=True)
    def primitive():
        return R.prim_value(1)

    @R.function(private=True)
    def typed():
        value: T.int32 = 1
        return value

    assert isinstance(tensor.ret_ty, relax.TensorType)
    assert primitive.ret_ty.dtype == "int64"
    assert typed.ret_ty.dtype == "int32"
    _check(typed)


if __name__ == "__main__":
    tvm_tirx.testing.main()
