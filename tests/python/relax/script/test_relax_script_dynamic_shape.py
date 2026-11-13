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
"""Relax script dynamic shape."""

from __future__ import annotations

import sys

import pytest

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx import IRModule, tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

M = T.dynamic("M", "int64")


UNUSED_GENERIC = T.dynamic("UNUSED_GENERIC", "int64")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_type_vars_roundtrip():
    @R.function(private=True)
    def func(
        x: R.Tensor((M, M * 2), "float32"),
    ) -> R.Tensor((M, M * 2), "float32"):
        return x

    script = func.script()
    assert script.startswith("from __future__ import annotations\n\n")
    assert "def main[M: T.int64](" in script
    assert 'R.Tensor((M, M * 2), dtype="float32")' in script
    assert "M = T.int64()" not in script
    typed = tvm_tirx.script.from_source(
        """
@R.function(private=True)
def func[M: T.int64](x: R.Tensor((M, M * 2), "float32")):
    return x
""",
        extra_vars={"I": tvm_tirx.script.ir, "R": tvm.relax.script, "T": tvm_tirx.script.tirx},
    )
    tvm_tirx.ir.assert_structural_equal(func, typed)
    assert 'M = T.dynamic("M", dtype="int64")' not in script

    repeated = func.script()
    assert "from __future__ import annotations" in repeated
    assert "def main[M: T.int64](" in repeated
    assert 'R.Tensor((M, M * 2), dtype="float32")' in repeated
    assert "M = T.int64()" not in repeated
    assert "UNUSED_GENERIC" not in script
    assert [param.name for param in func.params] == ["x"]
    assert not hasattr(func, "type_params")
    assert func.attrs.get("relax.type_vars") is None
    tvm_tirx.ir.assert_structural_equal(
        func,
        tvm_tirx.script.from_source(
            script,
            extra_vars={"I": tvm_tirx.script.ir, "R": tvm.relax.script, "T": tvm_tirx.script.tirx},
        ),
    )
    tvm_tirx.ir.assert_structural_equal(
        func,
        tvm_tirx.script.from_source(
            repeated,
            extra_vars={"I": tvm_tirx.script.ir, "R": tvm.relax.script, "T": tvm_tirx.script.tirx},
        ),
    )


def test_dynamic_module_symbol_identity():
    shared = T.dynamic("n", "int64")
    independent = T.dynamic("n", "int64")

    @R.function(private=True)
    def first(x: R.Tensor((shared,), "float32")):
        return x

    @R.function(private=True)
    def second(x: R.Tensor((shared, independent), "float32")):
        return x

    mod = tvm_tirx.IRModule({"first": first, "second": second})
    source = mod.script()
    assert source.count('T.dynamic("n", dtype="int64")') == 2
    restored = tvm_tirx.script.from_source(
        source,
        check_well_formed=False,
        extra_vars={"I": tvm_tirx.script.ir, "R": tvm.relax.script, "T": tvm_tirx.script.tirx},
    )
    first_n = restored["first"].params[0].ty.shape.values[0]
    second_shape = restored["second"].params[0].ty.shape.values
    assert first_n.same_as(second_shape[0])
    assert not first_n.same_as(second_shape[1])
    assert str(second_shape[1].ty.dtype) == "int64"
    tvm_tirx.ir.assert_structural_equal(mod, restored)


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


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_symbolic_shape():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
        gv0 = R.call_dps_packed(
            _ExternFunc("extern_func"), x, ty_args=[R.Tensor((m, n), dtype="float32")]
        )
        return gv0

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def bar(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
        gv0 = R.call_dps_packed(
            _ExternFunc("extern_func"), x, ty_args=[R.Tensor((m, n), dtype="float32")]
        )
        return gv0

    with pytest.raises(RuntimeError):
        m = T.dynamic("m", "int64")
        n = T.dynamic("n", "int32")

        @R.function
        def mismatch_dtype(x: R.Tensor((m, n), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv0 = R.call_dps_packed(
                _ExternFunc("extern_func"), x, ty_args=[R.Tensor((m, n), dtype="float32")]
            )
            return gv0

    def _expected(name: str):
        n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
        x = relax.Var("x", R.Tensor([m, n], "float32"))
        bb = relax.BlockBuilder()
        with bb.function(name, (x,)):
            out = bb.emit(
                relax.call_dps_packed(
                    _ExternFunc("extern_func"), x, ty_args=[R.Tensor((m, n), dtype="float32")]
                )
            )
            bb.emit_func_output(out)
        return bb.get()[name]

    _check(foo, _expected("foo"))
    _check(bar, _expected("bar"))


def test_match_cast():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor("float32"), y: R.Tensor("float32")):
        x0 = R.match_cast(x, R.Tensor([m], "float32"))
        with R.dataflow():
            y0 = R.match_cast(y, R.Tensor([n], "float32"))
            gv = y0
            R.output(gv)
        return (x0, R.shape([m, n * 2]))

    x = relax.Var("x", R.Tensor("float32"))
    y = relax.Var("y", R.Tensor("float32"))
    m = tirx.Var("m", ty="int64")
    n = tirx.Var("n", ty="int64")
    y2 = relax.Var("y", R.Tensor([n], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        x0 = bb.match_cast(x, R.Tensor([m], "float32"))
        with bb.dataflow():
            y0 = bb.match_cast(y, R.Tensor([n], "float32"))
            bb.emit_output(y0)
        bb.emit_func_output(relax.Tuple([x0, relax.ShapeExpr([m, n * 2])]))

    _check(foo, bb.get()["foo"])


def test_call_tir_with_tir_var():
    n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(
            dumb_param: R.Tensor((n,), "float32"), x: R.Tensor((n * 2,), "float32")
        ) -> R.Tensor((n * 2,), "float32"):
            cls = Module
            y = R.call_tir(cls.copy, (x, n), ty_args=[R.Tensor((n * 2,), dtype="float32")])
            return y

        copy_n = T.int64()

        @Ts.function
        def copy(
            X: T.Tensor((copy_n * 2,), dtype="float32"),
            n: copy_n,
            Y: T.Tensor((copy_n * 2,), dtype="float32"),
        ):
            for i in T.grid(n * 2):
                with Ts.sblock("block"):
                    vi = Ts.axis.remap("S", [i])
                    Y[vi] = X[vi]

    _check(Module)
    repeated = Module.script(show_meta=True)
    tvm_tirx.ir.assert_structural_equal(
        Module,
        tvm_tirx.script.from_source(
            repeated,
            extra_vars={
                "I": tvm_tirx.script.ir,
                "R": tvm.relax.script,
                "T": tvm_tirx.script.tirx,
                "Ts": tvm.s_tir.script,
            },
        ),
    )


def test_if_branch_with_match_cast():
    """The last branch of a relax::IfExpr node may be a MatchCast

    This is a regression test.  In previous implementations, using
    R.match_cast as the last binding would cause a segfault while
    parsing.
    """

    @R.function
    def func(A: R.Tensor([16, 16]), is_bfloat16: T.bool):
        if is_bfloat16:
            matched = R.match_cast(A, R.Tensor([16, 16], "bfloat16"))
            B = matched.astype("float16")
        else:
            B = R.match_cast(A, R.Tensor([16, 16], "float16"))
        return B

    A, is_bfloat16 = func.params
    (block,) = func.body.blocks
    (B_binding,) = block.bindings

    B_var = B_binding.var
    assert isinstance(B_var, relax.Var)
    assert B_var.name == "B"

    if_then_else = B_binding.value
    assert isinstance(if_then_else, relax.IfExpr)
    assert isinstance(if_then_else.true_branch, relax.SeqExpr)
    assert isinstance(if_then_else.false_branch, relax.SeqExpr)

    else_branch = if_then_else.false_branch
    (else_block,) = else_branch.blocks

    assert isinstance(else_block.bindings[-1], relax.MatchCast)

    # If the `R.match_cast` were removed, the function would infer the
    # return value as `R.Tensor([16,16])`, with an unknown dtype.
    # With the `R.match_cast` retained, the output dtype is known.
    tvm_tirx.ir.assert_structural_equal(func.ret_ty, R.Tensor([16, 16], "float16"))


def test_erase_to_well_defined_removes_internal_vars():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor):
        q = x
        z = R.match_cast(q, R.Tensor((m, n)))
        w = z
        return w

    tvm_tirx.ir.assert_structural_equal(foo.ret_ty, R.Tensor(ndim=2))
    assert foo.ret_ty.shape is None
    _check(foo)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_erase_to_well_defined_keeps_variables_exposed_by_tensor_shape():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor([m, n])):
        q = x
        z = R.match_cast(q, R.Tensor((m, n)))
        w = z
        return w

    assert foo.ret_ty.shape is not None
    _check(foo)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_erase_to_well_defined_keeps_variants_exposed_by_shape_expr():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor, _: R.Shape([m, n])):
        q = x
        z = R.match_cast(q, R.Tensor((m, n)))
        w = z
        return w

    assert foo.ret_ty.shape is not None
    _check(foo)


def test_erase_to_well_defined_infers_from_shape_expr():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    main_m = T.dynamic("m", "int64")
    main_n = T.dynamic("n", "int64")

    @I.ir_module
    class Module:
        # The subroutine's symbolic variables are only in-scope for the subroutine.
        @R.function
        def subroutine(x: R.Tensor, _: R.Shape([m, n])) -> R.Tensor([m, n]):
            q = x
            z = R.match_cast(q, R.Tensor((m, n)))
            w = z
            return w

        # However, struct inference can make the symbolic variables in
        # the main function to the symbolic variables in the
        # subroutine.  Therefore, the shape of the tensor returned
        # from main can have a well-defined shape.
        @R.function
        def main(x: R.Tensor, shape: R.Shape([main_m, main_n])):
            output = Module.subroutine(x, shape)
            return output

    assert Module["main"].ret_ty.shape is not None
    _check(Module)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_symbolic_vars_in_tensor_shape_with_usage_first():
    """A captured symbol can first appear inside a compound dimension."""

    m = T.dynamic("m", "int64")

    @R.function
    def foo(x: R.Tensor((m + 1,), "float32"), y: R.Tensor((m, 1), "float32")):
        z = R.add(x, y)
        return z

    m = tirx.Var("m", "int64")
    x = relax.Var("x", relax.TensorType([m + 1], "float32"))
    y = relax.Var("y", relax.TensorType([m, 1], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("foo", (x, y)):
        z = bb.emit(relax.op.add(x, y))
        bb.emit_func_output(z)

    _check(foo, bb.get()["foo"])


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_symbolic_vars_in_tensor_shape_with_definition_first():
    """A captured symbol is shared across direct and compound dimensions."""

    m = T.dynamic("m", "int64")

    @R.function
    def bar(x: R.Tensor((m,), "float32"), y: R.Tensor((T.max(m, 20),), "float32")) -> R.Tensor(
        (T.max(m, 20) + 1,), "float32"
    ):
        z = R.call_dps_packed(
            _ExternFunc("test_intrin"),
            (x, y),
            ty_args=[R.Tensor((T.max(m, 20) + 1,), dtype="float32")],
        )
        return z

    m = tirx.Var("m", "int64")
    x = relax.Var("x", relax.TensorType([m], "float32"))
    y = relax.Var("y", relax.TensorType([tirx.max(m, 20)], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("bar", (x, y)):
        z = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("test_intrin"),
                (x, y),
                ty_args=[R.Tensor((tirx.max(m, 20) + 1,), dtype="float32")],
            )
        )
        bb.emit_func_output(z)

    _check(bar, bb.get()["bar"])


def test_bound_prim_param_reused_in_dependent_annotations():
    @R.function
    def func(
        n: T.int64,
        direct: R.Tensor([n], "float32"),
        repeated: R.Tensor([n], "float32"),
        shape: R.Shape([n]),
        compound: R.Tensor([n + 1], "float32"),
    ) -> R.Tensor([n + 1], "float32"):
        return compound

    n, direct, repeated, shape, compound = func.params
    assert direct.ty.shape[0].same_as(n)
    assert repeated.ty.shape[0].same_as(n)
    assert shape.ty.values[0].same_as(n)
    assert compound.ty.shape[0].a.same_as(n)
    assert func.ret_ty.shape[0].a.same_as(n)
    _check(func)


def test_bound_prim_param_reused_in_declared_function_signature():
    @I.ir_module
    class Module:
        @R.function
        def main(n: T.int64, x: R.Tensor([n + 1], "float32")) -> R.Tensor([n + 1], "float32"):
            return x

    func = Module["main"]
    n, x = func.params
    assert x.ty.shape[0].a.same_as(n)
    assert func.ret_ty.shape[0].a.same_as(n)
    _check(Module)


def test_non_int64_prim_param_rejected_in_shape_annotation():
    with pytest.raises(RuntimeError):

        @R.function
        def main(n: T.int32, x: R.Tensor([n], "float32")):  # noqa: F821
            return x


def test_recursive_local_function_reuses_earlier_prim_param_in_signature():
    @R.function
    def func(n: T.int64, x: R.Tensor([n], "float32")):  # noqa: F821
        @R.function
        def recurse(current: T.int64, value: R.Tensor([current], "float32")) -> R.Tensor(
            [current], "float32"
        ):
            return recurse(current, value)

        return recurse(n, x)

    recursive_binding = next(
        binding
        for block in func.body.blocks
        for binding in block.bindings
        if isinstance(binding.value, relax.Function)
    )
    recursive_func = recursive_binding.value
    current, value = recursive_func.params
    assert value.ty.shape[0].same_as(current)
    assert recursive_func.ret_ty.shape[0].same_as(current)
    relax.analysis.check_well_formed(func)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_symbolic_vars_in_shape():
    """Symbolic variable may be defined in R.Shape"""

    m = T.dynamic("m", "int64")

    @R.function
    def baz(x: R.Shape((m,)), y: R.Tensor((m * 2,), "float32")):
        z = R.call_dps_packed(
            _ExternFunc("test_intrin"), y, ty_args=[R.Tensor((m * 2,), dtype="float32")]
        )
        return z

    m = tirx.Var("m", "int64")
    x = relax.Var("x", relax.ShapeType([m]))
    y = relax.Var("y", relax.TensorType([m * 2], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("baz", (x, y)):
        z = bb.emit(
            relax.call_dps_packed(
                _ExternFunc("test_intrin"), (y), ty_args=[R.Tensor((m * 2,), dtype="float32")]
            )
        )
        bb.emit_func_output(z)

    _check(baz, bb.get()["baz"])


def test_string_shape_expression_is_not_resolved():
    """A quoted expression is ordinary data, not a symbol declaration."""
    with pytest.raises(TypeError, match=r"Array<ir\.PrimExpr>"):

        @R.function
        def foo(x: R.Tensor(("m + 1", "m * 2"), "float32")):
            return x


@pytest.mark.xfail(reason="Bug: Implicit bounds not provided when parsing")
def test_function_symbolic_variables_are_annotated():
    """Symbolic variables must be exposed for struct inference

    Because Relax struct inference is performed while the function is
    being built, all constraints on symbolic variables that are used
    for simplifications must be provided to the analyzer.
    """

    extent = T.dynamic("extent", "int64")

    @R.function(private=True)
    def inferred_ty(A: R.Tensor([extent])):
        output = R.strided_slice(A, [0], [0], [extent - 1])
        return output

    extent = T.dynamic("extent", "int64")

    @R.function(private=True)
    def expected(A: R.Tensor([extent])) -> R.Tensor([extent - 1]):
        output: R.Tensor([extent - 1]) = R.strided_slice(A, [0], [0], [extent - 1])
        return output

    tvm_tirx.ir.assert_structural_equal(inferred_ty, expected)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_non_declaration_prim_expr_emits_binding():
    """Dtype casts emit ordinary bindings without replacing shape symbols."""

    symbol = T.dynamic("extent", "int64")

    @R.function(private=True)
    def func(A: R.Tensor([symbol], "float32")):
        extent = T.int64(4)
        output = A
        return output

    symbolic_extent = func.params[0].ty.shape[0]
    extent_binding = func.body.blocks[0].bindings[0]
    assert isinstance(extent_binding, relax.VarBinding)
    assert tvm_tirx.ir.is_prim_var(symbolic_extent)
    assert tvm_tirx.ir.is_prim_var(extent_binding.var)
    assert not symbolic_extent.same_as(extent_binding.var)
    tvm_tirx.ir.assert_structural_equal(extent_binding.value, T.int64(4))
    _check(func)


def test_conditional_may_use_symbolic_variables_from_function_scope():
    """Symbolic variables from function scope may be used in branch

    This is a regression test.  In earlier implementations, the
    branches of `relax::IfExpr` were normalized with
    `EraseToWellDefinedInScope`, using a fresh variable scope.  While
    this had the intended behavior of preventing variables defined in
    a single branch from being usable outside of the conditional, it
    also caused the conditional's branches to treat function-scope
    symbolic variables as if they were undefined.

    """

    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def explicit_ty(
        A: R.Tensor([N], "float32"),
        B: R.Tensor([N], "float32"),
        cond: T.bool,
    ) -> R.Tensor([N], "float32"):
        if cond:
            out: R.Tensor([N], "float32") = A + B
        else:
            out: R.Tensor([N], "float32") = A * B

        return out

    N = T.dynamic("N", "int64")

    @R.function(private=True)
    def inferred_ty(
        A: R.Tensor([N], "float32"),
        B: R.Tensor([N], "float32"),
        cond: T.bool,
    ):
        if cond:
            out = A + B
        else:
            out = A * B

        return out

    tvm_tirx.ir.assert_structural_equal(explicit_ty, inferred_ty)


def relax_symbolic_var():
    """Relax tensors may use symbolic variables."""
    N = T.dynamic("N", "int64")

    @R.function
    def func(A: R.Tensor([N], "float16")):
        B: R.Tensor([N], "float16") = A
        return B

    return func


@pytest.mark.parametrize("ir_generator", [relax_symbolic_var], ids=lambda factory: factory.__name__)
@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires Python 3.12")
def test_roundtrip_dynamic_shape(ir_generator):
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


def test_later_prim_param_reuses_shape_symbol():
    @R.function
    def function(x: R.Tensor([n], "float32"), n: T.int64):
        return x

    x, n = function.params
    assert x.ty.shape[0].same_as(n)
    assert str(n.ty.dtype) == "int64"


if __name__ == "__main__":
    tvm_tirx.testing.main()
