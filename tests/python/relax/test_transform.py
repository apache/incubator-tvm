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

import pytest

import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def _make_dataflow_block_with_dual_role_var():
    prim_ty = tvm_tirx.ir.PrimType("int64")
    source = tvm_tirx.ir.Var("source", prim_ty)
    data = tvm_tirx.ir.Var("data", relax.TensorType(dtype="float32", ndim=1))
    extent = tvm_tirx.ir.Var("extent", prim_ty)
    matched_ty = relax.TensorType([extent], "float32")
    matched = relax.DataflowVar("matched", matched_ty)

    block = relax.DataflowBlock(
        [
            relax.VarBinding(extent, source),
            relax.MatchCast(matched, data, matched_ty),
        ]
    )
    func = relax.Function([source, data], relax.SeqExpr([block], extent), prim_ty)
    return relax.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))


def test_dataflowblock_pass_rejects_deleting_binding_role_of_symbolic_var():
    mod = _make_dataflow_block_with_dual_role_var()

    @relax.transform.dataflowblock_pass(opt_level=0)
    def delete_ordinary_binding(block, _mod, _ctx):
        return relax.DataflowBlock([block.bindings[1]])

    with pytest.raises(RuntimeError, match="global-scope Var"):
        delete_ordinary_binding(mod)


def test_dataflowblock_pass_rejects_rewriting_match_cast_role_of_binding_var():
    mod = _make_dataflow_block_with_dual_role_var()

    @relax.transform.dataflowblock_pass(opt_level=0)
    def rewrite_match_cast_symbol(block, _mod, _ctx):
        old_match_cast = block.bindings[1]
        replacement = tvm_tirx.ir.Var("replacement", "int64")
        replacement_ty = relax.TensorType([replacement], "float32")
        new_match_cast = relax.MatchCast(
            relax.DataflowVar(old_match_cast.var.name, replacement_ty),
            old_match_cast.value,
            replacement_ty,
        )
        return relax.DataflowBlock([block.bindings[0], new_match_cast])

    with pytest.raises(RuntimeError, match="symbolic Var declared by a MatchCast"):
        rewrite_match_cast_symbol(mod)


def test_to_non_dataflow():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class TestToNonDataflow:
        @R.function
        def foo(x: R.Tensor((m, n), "float32")):
            with R.dataflow():
                lv0 = R.call_dps_packed(
                    _ExternFunc("test.op.identity"),
                    (x,),
                    ty_args=[
                        R.Tensor(
                            (m, n),
                            dtype="float32",
                        )
                    ],
                )
                gv0 = R.call_dps_packed(
                    _ExternFunc("test.op.identity"),
                    (lv0,),
                    ty_args=[
                        R.Tensor(
                            (m, n),
                            dtype="float32",
                        )
                    ],
                )
                R.output(gv0)
            return gv0

    mod = TestToNonDataflow

    old_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var) and not tvm_tirx.ir.is_prim_expr(e):
            nonlocal old_vars
            old_vars.append(e)

    relax.analysis.post_order_visit(mod["foo"], fvisit)
    x, lv0, gv0 = old_vars

    new_mod = relax.transform.ToNonDataflow()(mod)

    new_vars = []

    def fvisit(e):
        if isinstance(e, relax.Var) and not tvm_tirx.ir.is_prim_expr(e):
            nonlocal new_vars
            new_vars.append(e)

    relax.analysis.post_order_visit(new_mod["foo"], fvisit)

    assert x == new_vars[0]
    assert lv0 != new_vars[1]
    assert isinstance(lv0, relax.DataflowVar)
    assert not isinstance(new_vars[1], relax.DataflowVar)

    assert isinstance(gv0, relax.Var)
    assert isinstance(new_vars[2], relax.Var)
    assert gv0 == new_vars[2]


def test_call_tir_rewrite():
    m_exp = T.dynamic("m", "int64")
    n_exp = T.dynamic("n", "int64")
    m_foo = T.dynamic("m", "int64")
    n_foo = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class TestCallTIRRewrite:
        @Ts.function
        def exp(A: T.Tensor((m_exp, n_exp), "float32"), B: T.Tensor((m_exp, n_exp), "float32")):
            T.evaluate(0)

        @R.function
        def foo(x: R.Tensor((m_foo, n_foo), "float32")):
            # we expect RemovePurityChecking to have been used before this point
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir(
                TestCallTIRRewrite.exp, (x,), ty_args=[R.Tensor((m_foo, n_foo), dtype="float32")]
            )
            return gv0

    mod = TestCallTIRRewrite

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, relax.Call)
    assert s0.op.name == "relax.call_tir"

    # after rewrite
    new_mod = relax.transform.CallTIRRewrite()(mod)
    func = new_mod["foo"]

    block = func.body.blocks[0]
    assert not isinstance(block, relax.DataflowBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, relax.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], relax.ShapeExpr)
    tvm_tirx.ir.assert_structural_equal(s1.args[0], s0.ty_args[0].shape)
    s2 = block.bindings[1].value
    assert s2.op.name == "relax.call_tir_packed"
    assert s2.args[0].name_hint == "exp"
    assert not s2.op.get_attr("FPurity")


def test_call_tir_rewrite_with_interspersed_primitive_argument():
    @I.ir_module
    class Module:
        @Ts.function
        def scale_add(
            A: T.Tensor((16,), "float32"),
            scale: T.float32,
            C: T.Tensor((16,), "float32"),
            B: T.Tensor((16,), "float32"),
        ):
            for i in range(16):
                B[i] = A[i] + scale * C[i]

        @R.function
        def main(
            A: R.Tensor((16,), "float32"),
            scale: T.float32,
            C: R.Tensor((16,), "float32"),
        ) -> R.Tensor((16,), "float32"):
            R.func_attr({"relax.force_pure": True})
            B = R.call_tir(Module.scale_add, (A, scale, C), ty_args=[R.Tensor((16,), "float32")])
            return B

    after = relax.transform.CallTIRRewrite()(Module)
    func = after["main"]
    bindings = func.body.blocks[0].bindings
    output_buffer = bindings[0].var
    call = bindings[1].value

    assert call.op.name == "relax.call_tir_packed"
    assert call.args[0].name_hint == "scale_add"
    assert not call.op.get_attr("FPurity")
    tvm_tirx.ir.assert_structural_equal(
        call.args[1].fields,
        [func.params[0], func.params[1], func.params[2], output_buffer],
    )


def test_transform_remove_purity_checking():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def base(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            z = R.add(x, y)
            return z

        @R.function
        def use_call_pure_packed(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            z = R.call_pure_packed("vm.builtin.copy", y, ty_args=(R.Tensor((), dtype="int32")))
            return z

        @R.function
        def use_invoke_pure_closure(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            closure = R.make_closure(Before.base, ())
            res = R.invoke_pure_closure(closure, (x,), ty_args=R.Tensor((), "int32"))
            return res

        @R.function(pure=False)
        def impure_func() -> R.Any:
            y = R.print("I am impure!")
            return y

        @R.function
        def nested_pure_func() -> R.Tensor((), "int32"):
            @R.function
            def nested(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                y = R.add(x, x)
                q = R.call_pure_packed("vm.builtin.copy", y, ty_args=(R.Tensor((), dtype="int32")))
                return q

            z = R.const(1, dtype="int32")
            w = nested(z)
            return w

        @R.function(pure=False)
        def nested_impure_func() -> R.Tensor((), "int32"):
            @R.function(pure=False)
            def nested() -> R.Any:
                x = R.print("Oops!")
                return x

            y = R.const(1, dtype="int32")
            z = nested()
            return y

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def base(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})
            y = R.add(x, x)
            z = R.add(x, y)
            return z

        @R.function
        def use_call_pure_packed(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})
            y = R.add(x, x)
            z = R.call_packed("vm.builtin.copy", y, ty_args=(R.Tensor((), dtype="int32")))
            return z

        @R.function
        def use_invoke_pure_closure(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})
            closure = R.make_closure(Expected.base, ())
            res = R.invoke_closure(closure, (x,), ty_args=R.Tensor((), "int32"))
            return res

        @R.function(pure=False)
        def impure_func() -> R.Any:
            y = R.print("I am impure!")
            return y

        @R.function
        def nested_pure_func() -> R.Tensor((), "int32"):
            R.func_attr({"relax.force_pure": True})

            @R.function
            def nested(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
                R.func_attr({"relax.force_pure": True})
                y = R.add(x, x)
                q = R.call_packed("vm.builtin.copy", y, ty_args=(R.Tensor((), dtype="int32")))
                return q

            z = R.const(1, dtype="int32")
            w = nested(z)
            return w

        @R.function(pure=False)
        def nested_impure_func() -> R.Tensor((), "int32"):
            @R.function(pure=False)
            def nested() -> R.Any:
                x = R.print("Oops!")
                return x

            y = R.const(1, dtype="int32")
            z = nested()
            return y

    new_mod = relax.transform.RemovePurityChecking()(Before)
    tvm_tirx.ir.assert_structural_equal(new_mod, Expected)


def test_call_dps_packed_rewrite():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class TestCallDPSPackedRewrite:
        @R.function
        def foo(x: R.Tensor((m, n), "float32")):
            # we expect RemovePurityChecking to have been used before this point
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_dps_packed(
                _ExternFunc("test.op.identity"), (x,), ty_args=[R.Tensor((m, n), dtype="float32")]
            )
            return gv0

    mod = TestCallDPSPackedRewrite

    # before rewrite
    v0 = mod["foo"].body.blocks[0].bindings[0].var
    s0 = mod["foo"].body.blocks[0].bindings[0].value
    assert isinstance(s0, relax.Call)
    assert s0.op.name == "relax.call_dps_packed"

    # CallTIRRewrite also works for call_dps_packed
    new_mod = relax.transform.CallTIRRewrite()(mod)
    func = new_mod["foo"]

    block = func.body.blocks[0]
    assert not isinstance(block, relax.DataflowBlock)

    s1 = block.bindings[0].value
    assert isinstance(s1, relax.Call)
    assert s1.op.name == "relax.builtin.alloc_tensor"
    assert isinstance(s1.args[0], relax.ShapeExpr)
    tvm_tirx.ir.assert_structural_equal(s1.args[0], s0.ty_args[0].shape)
    s2 = block.bindings[1].value
    assert s2.op.global_symbol == "test.op.identity"


def test_call_dps_packed_rewrite_nested_tuple_output():
    """Flatten nested outputs for the packed ABI, then rebuild their Relax structure."""
    input_ty = relax.TensorType((2, 3), "float32")
    flat_output_types = [
        relax.TensorType((2, 3), "float32"),
        relax.TensorType((4,), "int32"),
        relax.TensorType((5, 6), "float16"),
    ]
    output_ty = tvm_tirx.ir.TupleType(
        [flat_output_types[0], tvm_tirx.ir.TupleType(flat_output_types[1:])]
    )

    x = relax.Var("x", input_ty)
    call = relax.Call(
        tvm_tirx.ir.Op.get("relax.call_dps_packed"),
        [relax.ExternFunc("test.op.nested_outputs"), relax.Tuple([x])],
        ty_args=[output_ty],
    )
    builder = relax.BlockBuilder()
    with builder.function("main", [x], attrs={"relax.force_pure": True}):
        out = builder.emit(call)
        builder.emit_func_output(out)

    after = relax.transform.CallTIRRewrite()(builder.get())
    relax.analysis.check_well_formed(after)
    func = after["main"]
    block = func.body.blocks[0]

    alloc_bindings = block.bindings[:3]
    for binding, expected_ty in zip(alloc_bindings, flat_output_types):
        assert binding.value.op.name == "relax.builtin.alloc_tensor"
        tvm_tirx.ir.assert_structural_equal(binding.var.ty, expected_ty)

    packed_call = block.bindings[3].value
    assert packed_call.op.global_symbol == "test.op.nested_outputs"
    assert packed_call.args[0].same_as(func.params[0])
    assert all(
        arg.same_as(binding.var) for arg, binding in zip(packed_call.args[1:], alloc_bindings)
    )

    rebuilt = block.bindings[4].value
    assert rebuilt.fields[0].same_as(alloc_bindings[0].var)
    assert rebuilt.fields[1].fields[0].same_as(alloc_bindings[1].var)
    assert rebuilt.fields[1].fields[1].same_as(alloc_bindings[2].var)


def test_call_tir_inplace_simple():
    # simple case: one inplace argument
    @tvm_tirx.script.ir_module
    class Input:
        @Ts.function
        def zeros(A: T.Tensor((2, 3), "int32")):
            # just overwrites A with 0s
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.writes(A[ax0, ax1])
                    A[ax0, ax1] = T.int32(0)

        @R.function
        def foo(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
            # we expect RemovePurityChecking to have been used before this point
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir_inplace(
                Input.zeros, x, inplace_indices=0, ty_args=[R.Tensor((2, 3), dtype="int32")]
            )
            return gv0

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def zeros(A: T.Tensor((2, 3), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.writes(A[ax0, ax1])
                    A[ax0, ax1] = T.int32(0)

        @R.function
        def foo(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
            R.func_attr({"relax.force_pure": True})
            _ = R.call_tir_packed(Expected.zeros, (x,))
            gv0 = x
            return gv0

    new_mod = relax.transform.CallTIRRewrite()(Input)
    tvm_tirx.ir.assert_structural_equal(Expected["foo"], new_mod["foo"], map_free_vars=True)


def test_call_tir_inplace_multiple_args():
    @tvm_tirx.script.ir_module
    class Input:
        @Ts.function
        def copy(
            A: T.Tensor((2, 3), "int32"), B: T.Tensor((2, 3), "int32"), C: T.Tensor((2, 3), "int32")
        ):
            # copies the contents of C into A and B
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(C[ax0, ax1])
                    Ts.writes(A[ax0, ax1], B[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    B[ax0, ax1] = C[ax0, ax1]

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")):
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir_inplace(
                Input.copy,
                (x, y, z),
                inplace_indices=[0, 1],
                ty_args=[
                    _TupleType([R.Tensor((2, 3), dtype="int32"), R.Tensor((2, 3), dtype="int32")])
                ],
            )
            return gv0

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def copy(
            A: T.Tensor((2, 3), "int32"), B: T.Tensor((2, 3), "int32"), C: T.Tensor((2, 3), "int32")
        ):
            # copies the contents of C into A and B
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(C[ax0, ax1])
                    Ts.writes(A[ax0, ax1], B[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    B[ax0, ax1] = C[ax0, ax1]

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")):
            R.func_attr({"relax.force_pure": True})
            _ = R.call_tir_packed(Expected.copy, (x, y, z))
            gv0 = (x, y)
            return gv0

    new_mod = relax.transform.CallTIRRewrite()(Input)
    tvm_tirx.ir.assert_structural_equal(Expected["foo"], new_mod["foo"], map_free_vars=True)


def test_call_tir_inplace_some_new():
    @tvm_tirx.script.ir_module
    class Input:
        @Ts.function
        def copy(
            A: T.Tensor((2, 3), "int32"),
            B: T.Tensor((2, 3), "int32"),
            C: T.Tensor((2, 3), "int32"),
            out1: T.Tensor((2, 3), "int32"),
            out2: T.Tensor((2, 3), "int32"),
        ):
            # copies the contents of C into A, out1, and out2
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(C[ax0, ax1])
                    Ts.writes(A[ax0, ax1], out1[ax0, ax1], out2[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    out1[ax0, ax1] = C[ax0, ax1]
                    out2[ax0, ax1] = C[ax0, ax1]

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(
            R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), dtype="int32")
        ):
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir_inplace(
                Input.copy,
                (x, y, z),
                inplace_indices=[0, -1, -1],
                ty_args=[
                    _TupleType(
                        [
                            R.Tensor((2, 3), dtype="int32"),
                            R.Tensor((2, 3), dtype="int32"),
                            R.Tensor((2, 3), dtype="int32"),
                        ]
                    )
                ],
            )
            return gv0

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function
        def copy(
            A: T.Tensor((2, 3), "int32"),
            B: T.Tensor((2, 3), "int32"),
            C: T.Tensor((2, 3), "int32"),
            out1: T.Tensor((2, 3), "int32"),
            out2: T.Tensor((2, 3), "int32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(C[ax0, ax1])
                    Ts.writes(A[ax0, ax1], out1[ax0, ax1], out2[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    out1[ax0, ax1] = C[ax0, ax1]
                    out2[ax0, ax1] = C[ax0, ax1]

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(
            R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), dtype="int32")
        ):
            R.func_attr({"relax.force_pure": True})
            gv0: R.Tensor((2, 3), dtype="int32") = R.emit_with_type(
                "relax.builtin.alloc_tensor",
                (R.shape([2, 3]), R.dtype("int32"), R.prim_value(0), R.str("global")),
                (R.Tensor((2, 3), dtype="int32"),),
            )
            gv1: R.Tensor((2, 3), dtype="int32") = R.emit_with_type(
                "relax.builtin.alloc_tensor",
                (R.shape([2, 3]), R.dtype("int32"), R.prim_value(0), R.str("global")),
                (R.Tensor((2, 3), dtype="int32"),),
            )
            _ = R.call_tir_packed(Expected.copy, (x, y, z, gv0, gv1))
            gv2 = (x, gv0, gv1)
            return gv2

    new_mod = relax.transform.CallTIRRewrite()(Input)
    tvm_tirx.ir.assert_structural_equal(Expected["foo"], new_mod["foo"], map_free_vars=True)


def test_call_tir_inplace_repeated_input():
    @tvm_tirx.script.ir_module
    class Input:
        @Ts.function
        def func(
            A: T.Tensor((2, 3), "int32"),
            B: T.Tensor((2, 3), "int32"),
            C: T.Tensor((2, 3), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def foo(
            x: R.Tensor((2, 3), "int32"),
            y: R.Tensor((2, 3), "int32"),
            z: R.Tensor((2, 3), "int32"),
        ) -> R.Tuple(R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")):
            R.func_attr({"relax.force_pure": True})
            gv0 = R.call_tir_inplace(
                Input.func,
                (x, y, z),
                # repeated 0 -> that's an error
                inplace_indices=[0, 0],
                ty_args=[
                    _TupleType([R.Tensor((2, 3), dtype="int32"), R.Tensor((2, 3), dtype="int32")])
                ],
            )
            return gv0

    with pytest.raises(ValueError, match="All in-place indices must be unique"):
        relax.analysis.check_well_formed(Input)


def test_call_tir_inplace_all_new():
    @tvm_tirx.script.ir_module
    class Input:
        @Ts.function
        def func(A: T.Tensor((2, 3), "int32")):
            T.evaluate(0)

        @R.function
        def foo(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
            R.func_attr({"relax.force_pure": True})
            # cannot make the only output a fresh one
            gv0 = R.call_tir_inplace(
                Input.func, x, inplace_indices=-1, ty_args=[R.Tensor((2, 3), dtype="int32")]
            )
            return gv0

    with pytest.raises(ValueError, match="out_ty is incompatible with the native function"):
        relax.analysis.check_well_formed(Input)


def test_inplace_mutation_with_tuple_argument_raises_error():
    """TIR Functions do not support Tuple arguments

    The `R.call_tir_inplace` operator must receive an in-line tuple of
    arguments, where each argument in the tuple may be expressed in
    TIR.  Here, `[[A]]` specifies a tuple of arguments, where the
    first argument is itself a tuple.  Since Functions do not support
    Tuple arguments, this is invalid.

    This is a regression test.  In previous implementations, this
    triggered a segfault rather than raising an exception.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            cls = Module
            gv1 = R.call_tir_inplace(
                cls.multiply_by_two,
                [[A]],
                ty_args=[R.Tensor((16,), dtype="float32")],
                inplace_indices=[0],
            )
            return gv1

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor((16,), "float32")):
            for i in range(16):
                A[i] = A[i] * T.float32(2)

    with pytest.raises(
        ValueError, match="Argument 0 type mismatch: expected R.Tensor.*given T.Tuple"
    ):
        relax.analysis.check_well_formed(Module)


def test_inplace_mutation_with_non_tensor_argument_raises_error():
    """In-place argument must be a tensor

    The `R.call_tir_inplace` operator must receive an in-line tuple of
    arguments, where each argument in the tuple may be expressed in
    TIR.  Here, the argument `A` is not a tensor.

    This is a regression test.  In previous implementations, this
    triggered a segfault rather than raising an exception.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Any):
            gv1 = R.call_tir_inplace(
                Module.multiply_by_two,
                [A],
                ty_args=[R.Tensor((16,), dtype="float32")],
                inplace_indices=[0],
            )
            return gv1

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor((16,), "float32")):
            for i in range(16):
                A[i] = A[i] * T.float32(2)

    with pytest.raises(
        ValueError, match="Argument 0 type mismatch: expected R.Tensor.*given T.Any"
    ):
        relax.analysis.check_well_formed(Module)


def test_inplace_mutation_with_incompatible_tensor_shape_raises_error():
    """In-place argument must have compatible shape

    The `R.call_tir_inplace` operator must receive an in-line tuple of
    arguments, where the shape of each in-place argument is compatible
    with the corresponding output.  Here, the shape of argument `A` is
    different than the output's shape (`[32]` as opposed to `[16]`).

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([32], dtype="float32")):
            gv1 = R.call_tir_inplace(
                Module.multiply_by_two,
                [A],
                ty_args=[R.Tensor((16,), dtype="float32")],
                inplace_indices=[0],
            )
            return gv1

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor((16,), "float32")):
            for i in range(16):
                A[i] = A[i] * T.float32(2)

    with pytest.raises(
        ValueError, match=r"Argument 0 type mismatch: expected R.Tensor.*given R.Tensor\(\(32,\)"
    ):
        relax.analysis.check_well_formed(Module)


def test_inplace_mutation_with_incompatible_tensor_dtype_raises_error():
    """In-place argument must have compatible dtype

    The `R.call_tir_inplace` operator must receive an in-line tuple of
    arguments, where the shape of each in-place argument is compatible
    with the corresponding output.  Here, the dtype of argument `A` is
    different than the output's dtype (`int32` as opposed to `float32`).

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], dtype="int32")):
            gv1 = R.call_tir_inplace(
                Module.multiply_by_two,
                [A],
                ty_args=[R.Tensor((16,), dtype="float32")],
                inplace_indices=[0],
            )
            return gv1

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor((16,), "float32")):
            for i in range(16):
                A[i] = A[i] * T.float32(2)

    with pytest.raises(
        ValueError, match="Argument 0 type mismatch: expected R.Tensor.*given R.Tensor.*int32"
    ):
        relax.analysis.check_well_formed(Module)


if __name__ == "__main__":
    tvm_tirx.testing.main()
