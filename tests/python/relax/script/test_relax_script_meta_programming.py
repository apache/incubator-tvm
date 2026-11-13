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
"""Relax script meta programming."""

from __future__ import annotations

import sys
from typing import TypeVar

import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax, topi
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx import IRModule, ir
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


def test_emit_te_function_attrs():
    @I.ir_module
    class TestModule:
        @Ts.function(private=True)
        def plus_one(
            x: T.Tensor((T.int64(128), T.int64(128)), "float32"),
            y: T.Tensor((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"some_attr": "foo", "another_attr": True, "tirx.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock():
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    y[vi, vj] = x[vi, vj] + 1.0

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
            cls = TestModule
            gv0 = R.call_tir(cls.plus_one, x, ty_args=[R.Tensor((128, 128), dtype="float32")])
            return gv0

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        out = bb.emit_te(
            lambda x: x + 1,
            x,
            function_name_hint="plus_one",
            function_attrs={"some_attr": "foo", "another_attr": True},
        )
        bb.emit_func_output(out)
    _check(TestModule, bb.get())


def test_emit_te():
    @I.ir_module
    class EmitTE:
        @R.function
        def main(x: R.Tensor((10, 20), "float32")) -> R.Tensor((10, 20), dtype="float32"):
            lv1 = R.emit_te(topi.add, x, x)
            out = R.emit_te(topi.multiply, lv1, lv1)
            return out

    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorType([10, 20], "float32"))
    with bb.function("main", [x], {"global_symbol": "main"}):
        lv1 = bb.emit_te(topi.add, x, x)
        out = bb.emit_te(topi.multiply, lv1, lv1)
        bb.emit_func_output(out)

    _check(EmitTE, bb.get())


def test_shadowing():
    @R.function
    def foo(x: R.Tensor((4, 4), "float32")):
        y = R.add(x, x)
        z = R.multiply(x, y)
        y = R.add(x, y)
        y = z
        y = R.multiply(y, x)
        z = y
        return z

    x = relax.Var("x", R.Tensor((4, 4), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x,)):
        y = bb.emit(relax.op.add(x, x))
        z = bb.emit(relax.op.multiply(x, y))
        y = bb.emit(relax.op.add(x, y))
        y = bb.emit(z)
        y = bb.emit(relax.op.multiply(y, x))
        z = bb.emit(y)
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


def test_local_function():
    @R.function
    def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor(
        (2, 3), "float32"
    ):
        @R.function
        def outer_func(
            c1: R.Tensor((2, 3), "float32"),
        ) -> R.Callable((R.Tensor(None, "float32", ndim=2),), R.Tensor(None, "float32", ndim=2)):
            @R.function
            def inner_func(x1: R.Tensor((2, 3), "float32")):
                s: R.Tensor((2, 3), "float32") = R.add(x1, c1)
                return s

            return inner_func

        in_call = outer_func(x)
        res = in_call(y)
        return res

    main_bindings = main.body.blocks[0].bindings
    assert len(main_bindings) == 3
    outer_func = main_bindings[0].value
    assert isinstance(outer_func, relax.Function)

    outer_func_bindings = outer_func.body.blocks[0].bindings
    assert len(outer_func_bindings) == 1
    inner_func = outer_func_bindings[0].value
    assert isinstance(inner_func, relax.Function)


def test_inline_function():
    with pytest.raises(TypeError, match="unexpected keyword argument.*local"):

        @I.ir_module
        class TestModule:
            @R.function
            def f(x: R.Tensor((128, 128), "float32"), y: R.Tensor((128, 128), "float32")):
                @Ts.function
                def my_matmul(
                    A: T.Tensor((128, 128), "float32"),
                    B: T.Tensor((128, 128), "float32"),
                    C: T.Tensor((128, 128), "float32"),
                ) -> None:
                    for i, j, k in T.grid(128, 128, 128):
                        with Ts.sblock():
                            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                            with Ts.init():
                                C[vi, vj] = 0.0
                            C[vi, vj] += A[vi, vk] * B[vj, vk]

                z = relax.call_tir(
                    my_matmul, (x, y), ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
                return z


def test_cross_function_call():
    @I.ir_module
    class Mod0:
        @R.function
        def foo(x: R.Tensor((10, 5), "float32")):
            s = R.add(x, x)
            return s

        @R.function
        def main(x: R.Tensor((10, 5), "float32")):
            cls = Mod0
            inner = cls.foo
            gv1 = inner(x)
            gv2 = Mod0.foo(x)
            return (inner, gv1, gv2)

    @I.ir_module
    class Mod1:
        @R.function
        def main(x: R.Tensor((10, 5), "float32")):
            cls = Mod1
            inner = cls.foo
            gv1 = inner(x)
            gv2 = Mod1.foo(x)
            return (inner, gv1, gv2)

        @R.function
        def foo(x: R.Tensor((10, 5), "float32")) -> R.Tensor((10, 5), "float32"):
            s = R.add(x, x)
            return s


def test_class_normalize():
    @tvm_tirx.script.ir_module
    class InputModule:
        @R.function
        def mul_add(x: R.Tensor) -> R.Tensor:
            return R.multiply(R.add(x, x), R.add(x, x))

    # The parser automatically normalizes the input AST to the following ANF form
    @tvm_tirx.script.ir_module
    class OutputModule:
        @R.function
        def mul_add(x: R.Tensor) -> R.Tensor:
            gv = R.add(x, x)
            gv1 = R.add(x, x)
            return R.multiply(gv, gv1)

    _check(InputModule, OutputModule)


def test_context_aware_parsing(monkeypatch):
    @tvm_tirx.script.ir_module
    class Module:
        @Ts.function
        def add(
            X: T.Tensor([T.int64(2), T.int64(4)], "float32"),
            Y: T.Tensor((), "float32"),
            Z: T.Tensor([T.int64(2), T.int64(4)], "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc = R.builtin.alloc_tensor(R.shape([2, 4]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = R.call_tir_packed(cls.add, (x, R.const(1, "float32"), alloc))
            return alloc

    _check(Module)

    # Break the env settings, but context-aware parsing can still handle it
    def _break_env(self, *args):
        raise RuntimeError("Fail to pass context-aware parsing")

    monkeypatch.setattr(tvm_tirx.ir.GlobalVar, "__call__", _break_env)

    _check(Module)


def test_macro_hygienic():
    x = R.prim_value(2)

    @R.macro(hygienic=True)
    def alloc_and_shape(dtype: str):
        alloc = R.builtin.alloc_tensor(R.shape([4, 4]), runtime_device_index=x, dtype=dtype)
        shape = R.shape_of(alloc)
        return shape

    x = R.prim_value(1)

    @R.function(private=True)
    def func(z: R.Tensor((4, 4), "float32")):
        shape = alloc_and_shape(dtype="float32")
        return shape

    @R.function(private=True)
    def expect(z: R.Tensor((4, 4), dtype="float32")) -> R.Shape([4, 4]):
        alloc: R.Tensor((4, 4), dtype="float32") = R.builtin.alloc_tensor(
            R.shape([4, 4]),
            R.dtype("float32"),
            R.prim_value(2),  # Make sure prim_value is 2
        )
        shape: R.Shape([4, 4]) = R.shape_of(alloc)
        shape_1: R.Shape([4, 4]) = shape
        return shape_1

    _check(func, expect)


def test_macro_non_hygienic():
    global global_x_var  # Lookup doesn't find this variable if it's not global

    global_x_var = R.prim_value(2)

    @R.macro(hygienic=False)
    def alloc_and_shape(dtype: str):
        alloc = R.builtin.alloc_tensor(
            R.shape([4, 4]), runtime_device_index=global_x_var, dtype=dtype
        )
        shape = R.shape_of(alloc)
        return shape

    global_x_var = R.prim_value(1)

    @R.function(private=True)
    def func(z: R.Tensor((4, 4), "float32")):
        shape = alloc_and_shape(dtype="float32")
        return shape

    @R.function(private=True)
    def expect(z: R.Tensor((4, 4), dtype="float32")) -> R.Shape([4, 4]):
        alloc: R.Tensor((4, 4), dtype="float32") = R.builtin.alloc_tensor(
            R.shape([4, 4]),
            R.dtype("float32"),
            R.prim_value(1),  # Make sure prim_value is 1
        )
        shape: R.Shape([4, 4]) = R.shape_of(alloc)
        shape_1: R.Shape([4, 4]) = shape
        return shape_1

    _check(func, expect)


def test_macro_no_variable_leak():
    with pytest.raises(NameError):

        @R.macro(hygienic=True)
        def add_two(value):
            x = value + R.const(1)  # `x` defined in macro
            y = x + R.const(1)
            return y

        @R.function(private=True)
        def func(t: R.Tensor((), "int32")):
            u = add_two(t)
            return x  # Should be undefined here  # noqa: F821


def test_reused_extern_func():
    """ExternFunc lookups can become bindings in EliminateCommonSubexpr"""

    @R.function(private=True)
    def parsed(x: R.Tensor((128, 128), "float32")) -> R.Tensor((128, 128), "float32"):
        func = R.ExternFunc("extern_func")
        gv0 = R.call_dps_packed(func, x, ty_args=[R.Tensor((128, 128), dtype="float32")])
        gv1 = R.call_dps_packed(func, gv0, ty_args=[R.Tensor((128, 128), dtype="float32")])
        return gv1

    x = relax.Var("x", R.Tensor((128, 128), "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x], private=True):
        func = bb.emit(relax.ExternFunc("extern_func"))
        y = bb.emit(relax.call_dps_packed(func, x, ty_args=[R.Tensor((128, 128), "float32")]))
        z = bb.emit(relax.call_dps_packed(func, y, ty_args=[R.Tensor((128, 128), "float32")]))
        bb.emit_func_output(z)

    expected = bb.get()["main"]

    _check(parsed, expected)


def test_extern_func_in_module():
    """Module-level parsing may produce function bindings"""

    @I.ir_module
    class parsed_module:
        my_ext = R.ExternFunc("my_ext")

        @R.function
        def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
            return a

    @R.function
    def func(a: R.Tensor((10, 10))) -> R.Tensor((10, 10)):
        return a

    expected = tvm_tirx.IRModule({"my_ext": relax.ExternFunc("my_ext"), "func": func})

    _check(parsed_module, expected)


def test_define_relax_function_using_global_var():
    """A @R.function may call a GlobalVar

    When parsing a @R.function, the function's body may reference
    GlobalVar instances available in the calling python scope.  The
    resulting function should pass TVMScript's well-formed check, as
    the GlobalVar may be available in the IRModule for which the
    function is being defined.
    """

    @I.ir_module
    class DefinedAllAtOnce:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return DefinedAllAtOnce.subroutine(A, B)

        @R.function(private=True)
        def subroutine(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            return R.matmul(A, B)

    @I.ir_module
    class MainDefinedLater:
        @R.function(private=True)
        def subroutine(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            return R.matmul(A, B)

    subroutine_gvar = MainDefinedLater.get_global_var("subroutine")

    @R.function
    def main(A: R.Tensor, B: R.Tensor):
        return subroutine_gvar(A, B)

    MainDefinedLater["main"] = main

    tvm_tirx.ir.assert_structural_equal(DefinedAllAtOnce, MainDefinedLater)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_shared_meta_var_uses_ordinary_relax_bindings():
    """Identity calls retain ordinary fresh Relax bindings and strict roundtrip."""

    assert I.meta_var is T.meta_var

    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def func(A: R.Tensor([N], "float32")):
        via_i = I.meta_var(N)
        via_t = T.meta_var(via_i)
        output = R.reshape(A, R.shape([via_t]))
        return output

    bindings = func.body.blocks[0].bindings
    assert len(bindings) == 3
    via_i, via_t, output = bindings
    assert via_i.value.same_as(func.params[0].ty.shape[0])
    assert via_t.value.same_as(via_i.var)
    assert not via_i.var.same_as(via_i.value)
    assert not via_t.var.same_as(via_t.value)
    assert func.body.body.same_as(output.var)
    source = func.script(extra_config={"relax.show_all_ty": False})
    assert "meta_var" not in source
    _check(func)

    symbol = T.dynamic("symbol", "int64")
    with pytest.raises(RuntimeError, match="Invalid annotation"):

        @R.function(private=True)
        def mismatched_binding():
            value: T.float32 = symbol
            return value


def test_return_annotation_keeps_local_symbols_and_unused_captures():
    n = ir.Var("n", "int64")
    unused = TypeVar("unused", bound=int)

    @R.function
    def main(x: R.Tensor((n,), "float32")) -> R.Tensor(
        ((lambda local: local if I.constexpr(True) else unused)(n),), "float32"
    ):
        return x

    assert main.ret_ty.shape[0].same_as(n)


if __name__ == "__main__":
    tvm_tirx.testing.main()
