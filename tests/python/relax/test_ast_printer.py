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
# ruff: noqa: F811, F841
import re
from functools import partial

import numpy as np
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.relax.testing import dump_ast
from tvm.relax.testing.ast_printer import ASTPrinter
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T

# Overload dump_ast to test both type and type annotations
dump_ast = partial(dump_ast, include_ty_annotations=True)


def strip_whitespace(text: str) -> str:
    """
    Remove all whitespace to avoid reasoning about newlines and indents
    """
    return re.sub(r"\s", "", text)


def normalize(func: rx.Function) -> rx.Function:
    """
    Normalize the expr to fill in the ty fields everywhere
    """

    # using a default mutator to use the BlockBuilder's normalizer,
    # which oddly differs from the Normalize pass
    @rx.expr_functor.mutator
    class DefaultMutator(rx.PyExprMutator):
        pass

    mod = tvm_tirx.IRModule()
    mod["main"] = func
    mut = DefaultMutator(mod)
    mod["main"] = mut.visit_expr(func)
    return mod["main"]


def assert_fields(nodename: str, fields: dict[str, str], target: str) -> None:
    """
    Given a target string, ensure that the string defines the specified node
    and that the given mappings of fields to values are present in the string.
    Strips all whitespace in the target and fields.
    Does not assume any particular ordering for the fields.
    """
    stripped_target = strip_whitespace(target)
    assert stripped_target.startswith(f"{nodename}(")
    for field, value in fields.items():
        assert f"{field}={strip_whitespace(value)}" in stripped_target


# test cases are mostly adapted from text_expr, only testing very basic properties


def test_var() -> None:
    v0 = rx.Var("v0")
    v0_str = dump_ast(v0)
    assert v0_str == 'Var(name="v0")'

    v1 = rx.Var("v1", R.Tensor([54, 96], "float32"))
    v1_no_annos = dump_ast(v1, include_ty_annotations=False)
    assert v1_no_annos == 'Var(name="v1")'
    v1_annos = dump_ast(v1)
    assert v1_annos != v1_no_annos
    assert "Expr" in v1_annos
    assert "ty" in v1_annos


def test_dataflow_var() -> None:
    v0 = rx.DataflowVar("v0")
    v0_str = dump_ast(v0)
    assert v0_str == 'DataflowVar(name="v0")'

    v1 = rx.DataflowVar("v1", R.Tensor([54, 96], "float16"))
    v1_no_annos = dump_ast(v1, include_ty_annotations=False)
    assert v1_no_annos == 'DataflowVar(name="v1")'
    v1_annos = dump_ast(v1)
    assert v1_annos != v1_no_annos
    assert "Expr" in v1_annos
    assert "ty" in v1_annos


def test_match_cast() -> None:
    # match_cast([16, 8], [m, n])
    m = T.dynamic("m", dtype="int64")
    n = T.dynamic("n", dtype="int64")
    shape = rx.const([16, 8], "int32")
    var = rx.Var("v0", R.Shape())
    b0 = rx.MatchCast(var, shape, R.Tensor([m, n], "int32"))
    b0_str = dump_ast(b0)
    assert b0_str.startswith("MatchCast(")
    assert "GenericConst" in b0_str
    assert "Expr(value=`m" in b0_str
    assert "Expr(value=`n" in b0_str
    assert "16" in b0_str
    assert "8" in b0_str

    # var1: Tensor((m, n), "float32") =
    #   match_cast(var0: R.Tensor("float32"), [m, n])
    value = rx.Var("value", R.Tensor("float32"))
    var = rx.Var("v1", R.Tensor([m, n], "float32"))
    b1 = rx.MatchCast(var, value, R.Tensor([m, n], "float32"))
    b1_str = dump_ast(b1)
    assert b1_str.startswith("MatchCast(")
    assert "Expr(value=`m" in b1_str
    assert "Expr(value=`n" in b1_str
    assert b1_str != dump_ast(b1, include_ty_annotations=False)


def test_var_binding() -> None:
    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b0 = rx.VarBinding(v0, val)
    b0_str = dump_ast(b0, include_ty_annotations=False)
    assert b0_str.startswith("VarBinding(")
    assert 'var=Var(name="v0")' in b0_str
    assert "value=" in b0_str
    assert "GenericConst(" in b0_str


def test_binding_block() -> None:
    m = T.dynamic("m", dtype="int64")
    n = T.dynamic("n", dtype="int64")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchCast(rx.Var("v0"), shape, R.Tensor([m, n], "int32"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.BindingBlock([b0, b1])
    block0_str = dump_ast(block0)
    assert block0_str.startswith("BindingBlock(")
    assert "bindings=" in block0_str
    assert "VarBinding(" in block0_str
    assert "MatchCast(" in block0_str
    assert '"v0"' in block0_str


def test_dataflow_block() -> None:
    m = T.dynamic("m", dtype="int64")
    n = T.dynamic("n", dtype="int64")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchCast(rx.Var("v0"), shape, R.Tensor([m, n], "int32"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.DataflowBlock([b0, b1])
    block0_str = dump_ast(block0)
    assert block0_str.startswith("DataflowBlock(")
    assert "bindings=" in block0_str
    assert "VarBinding(" in block0_str
    assert "MatchCast(" in block0_str
    assert '"v0"' in block0_str


def test_seq_expr() -> None:
    x = rx.Var("foo")
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    seqe_str = dump_ast(seqe)
    assert seqe_str.startswith("SeqExpr(")
    assert "blocks=" in seqe_str
    assert "BindingBlock(" in seqe_str
    assert "VarBinding(" in seqe_str
    assert "GenericConst(" in seqe_str
    assert 'var=Var(name="foo")' in seqe_str
    assert "value=GenericConst(value" in strip_whitespace(seqe_str)
    assert "body=" in seqe_str


def test_shape_expr() -> None:
    m = T.dynamic("m", dtype="int32")
    n = T.dynamic("n", dtype="int32")
    s = rx.ShapeExpr([m, n])
    s_str = dump_ast(s)
    assert s_str.startswith("ShapeExpr(")
    assert "values=" in s_str
    assert "Expr(value=`m: int32`)" in s_str
    assert "Expr(value=`n: int32`)" in s_str


def test_func():
    x = rx.Var("foo", R.Tensor("float32", ndim=2))
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    func = rx.Function([x], seqe, R.Tensor("float32"))
    func = func.with_attr("global_symbol", "func")

    func_str = dump_ast(func)
    assert func_str.startswith("Function(")
    assert "params=" in func_str
    assert "body=" in func_str
    assert "ret_ty=" in func_str
    assert "is_pure=" in func_str
    assert "attrs=" in func_str
    assert '"global_symbol": "func"' in func_str
    assert "SeqExpr(" in func_str
    assert "blocks=" in func_str
    assert "VarBinding(" in func_str


def test_shape_of():
    v0 = rx.Var("v0", R.Tensor(ndim=2))
    s0 = rx.get_shape_of(v0)
    s0_str = dump_ast(s0)
    assert s0_str.startswith("Call(")
    assert 'op=Op(name="relax.shape_of")' in s0_str
    assert "args=" in s0_str
    assert 'name="v0"' in s0_str

    v1 = rx.Var("v1", R.Tensor([96, 54]))
    s1 = rx.get_shape_of(v1)
    s1_str = dump_ast(s1)
    assert s1_str.startswith("ShapeExpr("), s1_str
    assert "values=" in s1_str
    assert "Expr(value=`T.int64(96)`)" in s1_str
    assert "Expr(value=`T.int64(54)`)" in s1_str


def test_shape_expr():
    shape_expr = rx.ShapeExpr([10, 20])
    shape_expr_str = dump_ast(shape_expr)
    assert shape_expr_str.startswith("ShapeExpr(")
    assert "values" in shape_expr_str
    assert "Expr(value=`T.int64(10)`)" in shape_expr_str
    assert "Expr(value=`T.int64(20)`)" in shape_expr_str


def test_types():
    printer = ASTPrinter()
    assert strip_whitespace(printer.visit_type_(rx.ShapeType(ndim=-1))) == "ShapeType(ndim=-1)"
    assert strip_whitespace(printer.visit_type_(rx.ShapeType(ndim=1))) == "ShapeType(ndim=1)"
    object_type = rx.AnyType()
    assert strip_whitespace(printer.visit_type_(object_type)) == "AnyType()"
    packed_type = rx.PackedFuncType()
    assert strip_whitespace(printer.visit_type_(packed_type)) == "PackedFuncType()"
    tensor_type = rx.TensorType(ndim=2, dtype="int32")
    assert strip_whitespace(printer.visit_type_(tensor_type)) == "TensorType(ndim=2,dtype=int32)"
    unit_type = rx.TupleType([])
    assert strip_whitespace(printer.visit_type_(unit_type)) == "TupleType(fields=[])"
    tuple_type = rx.TupleType([rx.ShapeType(ndim=-1), object_type])
    assert_fields(
        "TupleType",
        {"fields": "[ShapeType(ndim=-1),AnyType()]"},
        strip_whitespace(printer.visit_type_(tuple_type)),
    )

    func_type = rx.FuncType([tensor_type], unit_type)
    assert_fields(
        "FuncType",
        {
            "params": "[TensorType(ndim=2, dtype=int32)]",
            "ret": "TupleType(fields=[])",
            "purity": "True",
        },
        printer.visit_type_(func_type),
    )


def test_ty():
    printer = ASTPrinter()

    assert printer.visit_ty_(rx.AnyType()) == "AnyType()"

    assert printer.visit_ty_(tvm_tirx.ir.PrimType("int32")) == "PrimType(dtype=int32)"

    # empty shape
    empty_ssi = rx.ShapeType()
    assert printer.visit_ty_(empty_ssi) == "ShapeType(ndim=-1)"

    # include some dimensions
    shape_info = rx.ShapeType([tvm_tirx.ir.IntImm("int64", 1), tvm_tirx.ir.IntImm("int64", 2)])
    assert strip_whitespace(printer.visit_ty_(shape_info)) == strip_whitespace(
        """
        ShapeType(
            ndim=2,
            values=[
                Expr(value=`T.int64(1)`),
                Expr(value=`T.int64(2)`)
            ]
        )
        """
    )

    # tensor type
    default_tsi = rx.TensorType()
    assert strip_whitespace(printer.visit_ty_(default_tsi)) == "TensorType(dtype=float32,ndim=-1)"

    # use a var as the shape
    x = rx.Var("x", ty=rx.ShapeType(values=[]))
    var_tsi = rx.TensorType(shape=x, dtype="int32")
    assert strip_whitespace(printer.visit_ty_(var_tsi)) == strip_whitespace(
        """
        TensorType(
            dtype=int32,
            shape=Var(
                name="x",
                ty=ShapeType(ndim=0, values=[])
            )
        )
        """
    )

    empty_tuple = rx.TupleType([])
    assert printer.visit_ty_(empty_tuple) == "TupleType(fields=[])"

    tuple_of_shape = rx.TupleType([empty_ssi])
    assert strip_whitespace(printer.visit_ty_(tuple_of_shape)) == strip_whitespace(
        """
        TupleType(fields=[
            ShapeType(ndim=-1)
        ])
        """
    )

    simple_func = rx.FuncType([], rx.AnyType())
    assert (
        strip_whitespace(printer.visit_ty_(simple_func))
        == "FuncType(params=[],ret=AnyType(),purity=True)"
    )


def test_call_packed_dictionary_attrs():
    call = R.call_packed("test.packed_attrs", label="preserved", axis=2)
    assert call.attrs["label"] == "preserved"
    assert call.attrs["axis"] == 2
    explicit = R.call_packed("test.packed_attrs", attrs_type_key="ir.DictAttrs")
    assert isinstance(explicit.attrs, tvm_tirx.ir.DictAttrs)
    assert len(explicit.attrs) == 0


def test_call_packed():
    # test case from test_parser
    m = T.dynamic("m", "int64")

    @R.function(pure=False)
    def f(
        x: R.Tensor((32, m), "float32"),
        y: R.Tensor((m,), "float32"),
        r: R.Tensor(dtype="int64"),
    ) -> R.Any:
        z: R.Tensor((32, m), "float32") = R.multiply(x, y)
        w: R.Tensor(ndim=2) = R.multiply(z, z)
        q: R.Tensor = R.add(w, w)
        t = R.add(w, z)
        sh: R.Shape = R.shape_of(t)
        o: R.Any = R.call_packed(
            "contrib.tensor_array_stack", x, y, ty_args=R.Any(), test_attr=True
        )
        return o

    # checking that the call_packed call is turned into a call to an extern func
    f_str = strip_whitespace(
        dump_ast(
            f,
            include_ty_annotations=False,
            include_call_attrs=True,
        )
    )

    # the function has an annotated return type
    assert "ret_ty=AnyType()" in f_str
    # the purity attribute is set to false
    assert "is_pure=False"

    assert isinstance(f.body, rx.SeqExpr)
    extern_call = f.body.blocks[0].bindings[-1].value
    extern_call_text = dump_ast(
        extern_call,
        include_ty_annotations=False,
        include_call_attrs=True,
    )
    assert strip_whitespace(extern_call_text) in f_str
    assert_fields(
        "Call",
        {
            "op": 'ExternFunc(global_symbol="contrib.tensor_array_stack")',
            "args": '[Var(name="x"), Var(name="y")]',
            "ty_args": "[AnyType()]",
            "attrs": '{"test_attr": True}',
        },
        extern_call_text,
    )

    # check that the op call is there too
    op_call = f.body.blocks[0].bindings[0].value
    op_call_text = dump_ast(
        op_call,
        include_ty_annotations=False,
        include_call_attrs=True,
    )
    assert strip_whitespace(op_call_text) in f_str
    assert_fields(
        "Call",
        {
            "op": 'Op(name="relax.multiply")',
            "args": '[Var(name="x"), Var(name="y")]',
        },
        op_call_text,
    )


def test_op_attrs():
    x = rx.Var("x", R.Tensor((10,), "float32"))
    # Manually create a Call with attributes to test printer support for Op attributes
    op = tvm_tirx.ir.Op.get("relax.add")
    attrs = tvm_tirx.ir.make_node("ir.DictAttrs", my_attr="my_value")
    call_node = rx.Call(op, [x, x], attrs=attrs)

    call_str = dump_ast(call_node, include_call_attrs=True)
    assert_fields(
        "Call",
        {
            "op": 'Op(name="relax.add")',
            "attrs": '{"my_attr": "my_value"}',
        },
        call_str,
    )


def test_call_tir():
    # also from test_parser
    m_addone = T.dynamic("m", "int64")
    n_addone = T.dynamic("n", "int64")
    m_foo = T.dynamic("m", "int64")
    n_foo = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class TestCallTIR:
        @Ts.function
        def addone(
            A: T.Tensor((m_addone, n_addone), "float32"),
            B: T.Tensor((m_addone, n_addone), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "addone"})
            for i, j in T.grid(m_addone, n_addone):
                with Ts.sblock("addone"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi, vj] + T.int32(1)

        @R.function
        def foo(x: R.Tensor((m_foo, n_foo), "float32")):
            gv0 = R.call_tir(
                TestCallTIR.addone, (x,), ty_args=[R.Tensor((m_foo, n_foo), dtype="float32")]
            )
            return gv0

    mod = TestCallTIR
    foo = mod["foo"]

    foo_str = strip_whitespace(
        dump_ast(
            foo,
            include_ty_annotations=False,
            include_call_attrs=False,
        )
    )
    assert foo_str.startswith('Function(params=[Var(name="x")]')

    # call_tir is an op in Relax and it takes an extern func as an argument
    assert isinstance(foo.body, rx.SeqExpr)
    tir_call = foo.body.blocks[0].bindings[0].value
    tir_call_text = dump_ast(
        tir_call,
        include_ty_annotations=False,
        include_call_attrs=False,
    )
    assert_fields(
        "Call",
        {
            "op": 'Op(name="relax.call_tir")',
            "args": """[
                GlobalVar(name_hint="addone"),
                Tuple(fields=[Var(name="x")])
            ]""",
            "ty_args": """[
                TensorType(
                    dtype=float32,
                    shape=ShapeExpr(
                        values=[
                            Expr(value=`m`),
                            Expr(value=`n`)
                        ]
                    )
                )
            ]""",
        },
        tir_call_text,
    )
    assert strip_whitespace(tir_call_text) in foo_str


def test_call_dps_packed():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @R.function
    def foo(x: R.Tensor((m, n), "float32")):
        gv0 = R.call_dps_packed(
            _ExternFunc("test.op.identity"), (x,), ty_args=[R.Tensor((m, n), dtype="float32")]
        )
        return gv0

    foo_str = strip_whitespace(
        dump_ast(
            foo,
            include_ty_annotations=False,
            include_call_attrs=False,
        )
    )
    assert foo_str.startswith('Function(params=[Var(name="x")]')

    # call_dps_packed is an op in Relax and it takes an extern func as an argument
    assert isinstance(foo.body, rx.SeqExpr)
    tir_call = foo.body.blocks[0].bindings[0].value
    tir_call_text = dump_ast(
        tir_call,
        include_ty_annotations=False,
        include_call_attrs=False,
    )
    assert_fields(
        "Call",
        {
            "op": 'Op(name="relax.call_dps_packed")',
            "args": """[
                ExternFunc(global_symbol="test.op.identity"),
                Tuple(fields=[Var(name="x")])
            ]""",
            "ty_args": """[
                TensorType(
                    dtype=float32,
                    shape=ShapeExpr(
                        values=[
                            Expr(value=`m`),
                            Expr(value=`n`)
                        ]
                    )
                )
            ]""",
        },
        tir_call_text,
    )
    assert strip_whitespace(tir_call_text) in foo_str


def test_operators():
    @R.function
    def foo(x: R.Tensor):
        return R.unique(x, sorted=True, axis=-1)

    foo_str = strip_whitespace(
        dump_ast(
            foo,
            include_ty_annotations=False,
        )
    )
    assert 'Op(name="relax.unique")' in foo_str
    # the sorted argument is true, so it will be a boolean Expr
    assert "Expr(value=`T.bool(True)`)" in foo_str
    # axis is -1
    assert "Expr(value=`T.int64(-1)`)" in foo_str

    @R.function(pure=False)
    def bar(x: R.Tensor):
        return R.print("{}", x)

    bar_str = strip_whitespace(
        dump_ast(
            bar,
            include_ty_annotations=False,
        )
    )
    # the format string is a StringImm argument
    assert 'StringImm(value="{}")' in bar_str


def test_print_ty_annotation_non_var():
    @R.function
    def f() -> R.Tensor:
        return R.const([1, 2])

    body = normalize(f).body
    body_str = strip_whitespace(dump_ast(body))
    # the constant has a shape of (2,)
    ty = strip_whitespace(
        """
        ty=TensorType(
            dtype=int32,
            shape=ShapeExpr(
                values=[Expr(value=`T.int64(2)`)],
                ty=ShapeType(
                    ndim=1,
                    values=[Expr(value=`T.int64(2)`)]
                )
            )
        )
        """
    )
    assert ty in body_str


def test_print_type_annotation_non_var():
    @R.function
    def f() -> R.Shape:
        return R.shape_of(R.const(1))

    body = normalize(f).body
    assert isinstance(body, rx.SeqExpr)
    call = body.blocks[-1].bindings[-1].value
    assert isinstance(call, rx.Call)


def test_if():
    @R.function
    def f(cond: R.Tensor((), dtype="bool")) -> R.Tensor((), dtype="int32"):
        if cond:
            x = R.const(1)
        else:
            x = R.const(2)
        return x

    body = normalize(f).body
    assert isinstance(body, rx.SeqExpr)
    body_str = strip_whitespace(dump_ast(body))
    # we expect both branches to be seq exprs
    assert "IfExpr" in body_str
    assert "true_branch=SeqExpr(" in body_str
    assert "false_branch=SeqExpr(" in body_str


def test_tuple_get_item():
    @R.function
    def f(x: R.Tuple(R.Tensor((), dtype="int32"))) -> R.Tensor((), dtype="int32"):
        return x[0]

    body = normalize(f).body
    assert isinstance(body, rx.SeqExpr)
    body_str = strip_whitespace(dump_ast(body))

    assert "TupleGetItem" in body_str
    assert 'tuple_value=Var(name="x"' in body_str
    assert "index=0" in body_str


def test_prim_value():
    prim_value = tvm_tirx.ir.IntImm("int64", 1)
    prim_str = strip_whitespace(dump_ast(prim_value))
    assert prim_str == strip_whitespace(
        """
        Expr(value=`T.int64(1)`)
    """
    )


def test_string_imm():
    string_imm = tvm_tirx.ir.StringImm("test")
    str_str = strip_whitespace(dump_ast(string_imm))
    assert str_str == strip_whitespace(
        """
        StringImm(
            value="test",
            ty=StringType()
        )
    """
    )


def test_datatype_imm():
    data_type_imm = tvm_tirx.ir.DataTypeImm(tvm_ffi.dtype("int32"))
    data_type_str = strip_whitespace(dump_ast(data_type_imm))
    assert data_type_str == strip_whitespace(
        """
        DataTypeImm(
            value=int32,
            ty=AnyType()
        )
    """
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
