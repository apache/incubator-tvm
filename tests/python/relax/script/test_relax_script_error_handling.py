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
"""Relax script error handling."""

from __future__ import annotations

import inspect

import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx import IRModule
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


def test_call_tir_requires_global_var():
    with pytest.raises(RuntimeError, match="first argument to be a GlobalVar"):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            # call_tir requires a GlobalVar rather than a packed-function name.
            gv0 = gv1 = R.call_tir(
                "extern_func", x, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
            return gv0


def test_mismatch_cast_dims_and_ndim():
    with pytest.raises(Exception):

        @R.function
        def f(
            x: R.Tensor((2, 3), "float32", ndim=3),
        ):  # error: ndim and the shape dims are mismatch
            return x


def test_unexpected_num_kw_args():
    with pytest.raises(Exception):

        @R.function
        def f(x: R.Tensor(dtype="float32", ndim=1, foo=2)):  # error: unexpected kw args foo
            return x


def test_unexpected_ndim():
    with pytest.raises(Exception):

        @R.function
        # error: dim is expected to be non-negative int or -1 for unknown
        def f(x: R.Tensor(dtype="float32", ndim=-2)):
            return x


def test_unexpected_ndim_type():
    with pytest.raises(Exception):

        @R.function
        def f(x: R.Tensor(dtype="float32", ndim="1")):  # error: dim is expected to be int
            return x


def test_unexpected_tir_cast_args():
    with pytest.raises(TypeError):
        m = T.dynamic("m", "int64")

        @R.function
        def f(x: R.Tensor((m,), "float32")):
            # tirx.cast expects 2 arguments, but got 3
            return R.call_tir(
                "foo",
                (x,),
                ty_args=[
                    R.Tensor(
                        (T.cast("int32", loc=1, ty=m),),
                        dtype="float32",
                    )
                ],
            )


def test_unexpected_tir_args():
    with pytest.raises(TypeError):
        m = T.dynamic("m", "int64")

        @tvm_tirx.script.ir_module
        class TestWellCallTIR:
            @Ts.function
            def tir_addone(A: T.Tensor((16, 16), "int32"), B: T.Tensor((16, 16), "int32")) -> None:
                T.func_attr({"global_symbol": "tir_addone"})
                for i, j in T.grid(16, 16):
                    with Ts.sblock("tir_addone"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] + T.int32(1)

            @R.function
            def foo(x: R.Tensor((m, m), "float32")):
                # tirx.max expects 2 arguments, but got 1
                gv = R.call_tir(tir_addone, (x,), ty_args=[R.Tensor((T.max(16),), dtype="float32")])  # noqa: F821
                return gv

    with pytest.raises(TypeError):
        m = T.dynamic("m", "int64")

        @R.function
        def f(x: R.Tensor((m, m), "float32")):
            # call_tir expected a tirx function
            return relax.call_tir(
                "extern_func", (x,), ty_args=[R.Tensor((T.max(m),), dtype="float32")]
            )


def test_func_type_annotation_fail():
    with pytest.raises(SyntaxError, match="Parameter 'x' requires an annotation") as error:

        @R.function
        def f(x, y):  # error: the parameter type annotation is missing
            z = R.add(x, y)
            y = z
            return y

    lines, start = inspect.getsourcelines(test_func_type_annotation_fail)
    line_index = next(i for i, line in enumerate(lines) if "def f(x, y):" in line)
    assert error.value.filename == __file__
    assert error.value.lineno == error.value.end_lineno == start + line_index
    assert error.value.offset == lines[line_index].index("x, y") + 1
    assert error.value.end_offset == error.value.offset + 1


def test_if_mismatch_var_fail():
    with pytest.raises(SyntaxError, match="same named output") as error:

        @R.function
        def f(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                z = R.add(w, w)  # error: The binding var is expected to `y`
            return z

    lines, start = inspect.getsourcelines(test_if_mismatch_var_fail)
    last = next(i for i, line in enumerate(lines) if "z = R.add(w, w)" in line)
    assert error.value.filename == __file__
    assert error.value.lineno == error.value.end_lineno == start + last
    assert error.value.offset == lines[last].index("z = R.add(w, w)") + 1
    assert (
        error.value.end_offset == lines[last].index("z = R.add(w, w)") + len("z = R.add(w, w)") + 1
    )


def test_unassigned_call_fail():
    with pytest.raises(ValueError):

        @R.function
        def f(x: R.Tensor):
            R.add(x, x)
            return x


def test_incorrect_tensor_shape():
    with pytest.raises(RuntimeError):

        @R.function
        def f(x: R.Tensor([16])):
            y: R.Tensor(16) = R.add(x, x)
            return y


def test_dataflow_binding_after_output():
    with pytest.raises(ValueError, match="New binding is not allowed after dataflow block output"):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv = R.call_dps_packed(
                    _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
                R.output(gv)
                lv = R.call_dps_packed(
                    _ExternFunc("extern_func"), gv, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
            return gv


def test_dataflow_output_global_var():
    with pytest.raises(
        ValueError, match="An output variable is not emitted by this dataflow block"
    ):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv0 = R.call_dps_packed(
                _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
            with R.dataflow():
                gv1 = R.call_dps_packed(
                    _ExternFunc("extern_func"), gv0, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
                R.output(gv0, gv1)
            return gv1


def test_dataflow_multiple_output():
    with pytest.raises(
        ValueError, match="not allowed for a dataflow block to have multiple output"
    ):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv = R.call_dps_packed(
                    _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
                )
                R.output(gv)
                R.output(gv)
            return gv


def test_dataflow_output_outside_dataflow_block():
    with pytest.raises(ValueError, match="`R.output` should appear inside a dataflow block"):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv = R.call_dps_packed(
                _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )
            R.output(gv)
            return gv


def test_dataflow_scope_fail():
    with pytest.raises(ValueError):

        @R.function
        def f(x: R.Tensor(ndim=2)):
            with R.dataflow():
                y = R.add(x, x)
                z = R.multiply(y, x)
                w = R.add(z, x)
                R.output(y, w)
            t = R.multiply(y, z)  # z is not in the outer scope
            return t


def test_multiple_return():
    with pytest.raises(ValueError):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")):
            return x
            return x


def test_function_without_return():
    with pytest.raises(ValueError, match="A Relax function must have a return value"):

        @R.function
        def foo(x: R.Tensor((128, 128), "float32")):
            gv0 = R.call_dps_packed(
                _ExternFunc("extern_func"), x, ty_args=[R.Tensor((128, 128), dtype="float32")]
            )


def test_annotate_override():
    @R.function
    def foo(x: R.Tensor):
        y = x
        # z will be treated as Any even though it's a tensor
        z: R.Any = R.add(x, y)
        return z

    assert isinstance(foo.ret_ty, relax.AnyType)
    y_bind, z_bind = foo.body.blocks[0].bindings
    assert isinstance(y_bind.var.ty, relax.TensorType)
    assert isinstance(z_bind.var.ty, relax.AnyType)

    with pytest.raises(RuntimeError):

        @R.function
        def test(x: R.Tensor):
            # Error: x is of Tensor Type, which can not annotate to R.Shape.
            z: R.Shape = x
            return z

    @R.function
    def bar(x: R.Tensor):
        # x is of Tensor Type, the annotation of `z` is ignored.
        z: R.Any = x
        return z

    assert isinstance(bar.ret_ty, relax.TensorType)
    (z_bind,) = bar.body.blocks[0].bindings
    assert isinstance(z_bind.var.ty, relax.TensorType)


def test_call_tir_inplace_with_tuple_var_raises_error():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")):
            cls = Module
            args = (x, y)
            res = R.call_tir_inplace(
                cls.copy,
                # The `args` tuple must be an in-line tuple, not a
                # reference to a tuple.  This error should be
                # caught by explicit module validation.
                args,
                inplace_indices=[0, -1],
                ty_args=[_TupleType([R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")])],
            )
            return res

        @Ts.function
        def copy(
            A: T.Tensor((2, 3), "int32"),
            B: T.Tensor((2, 3), "int32"),
            out1: T.Tensor((2, 3), "int32"),
        ):
            # copies the contents of B into A and out1
            T.func_attr({"tirx.noalias": True})
            for (*iters,) in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    i, j = Ts.axis.remap("SS", iters)
                    A[i, j] = B[i, j]
                    out1[i, j] = B[i, j]

    with pytest.raises(ValueError):
        relax.analysis.check_well_formed(Module)


def test_if_inside_dataflow():
    with pytest.raises(ValueError):

        @R.function
        def foo(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            with R.dataflow():
                if cond:
                    w = R.add(x, x)
                    y = R.multiply(w, w)
                else:
                    w = R.multiply(x, x)
                    y = R.add(w, w)
                R.output(y)
            return y


def test_var_if_scoping_fail():
    with pytest.raises(NameError):

        @R.function
        def f(cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")):
            if cond:
                w = R.add(x, x)
                y = R.multiply(w, w)
            else:
                w = R.multiply(x, x)
                y = R.add(w, w)
            return w  # error: The w is not defined in the outer scope


def test_function_with_non_void_return_type_must_be_assigned():
    """Non-void results must be assigned to a variable"""

    with pytest.raises(ValueError):

        @R.function(pure=False)
        def func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.add(x, x)
            return x


@pytest.mark.parametrize("value", [(1,), ((),), [()]])
def test_nonempty_host_tuple_statement_is_rejected(value):
    with pytest.raises(TypeError, match="Unsupported expression statement"):

        @R.function
        def func():
            value
            return ()


def test_private_function_with_global_symbol_fail():
    with pytest.raises(ValueError):

        @I.ir_module
        class Addition:
            @R.function(private=True)
            def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                # it is an error to simultaneously mark a function private
                # and give it a global symbol manually
                R.func_attr({"global_symbol": "main"})
                y = R.add(x, x)
                return y

        # should not execute
        _check(Addition)


def test_private_function_with_global_symbol_no_module_fail():
    with pytest.raises(ValueError):

        @R.function(private=True)
        def func(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"global_symbol": "main"})
            y = R.add(x, x)
            return y

        # should not execute
        _check(func)


if __name__ == "__main__":
    tvm_tirx.testing.main()
