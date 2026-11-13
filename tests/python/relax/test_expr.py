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
# ruff: noqa: F811
import json

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
from tvm import relax as rx
from tvm.relax import script as R
from tvm.relax.expr import make_shape
from tvm_tirx import tirx


def _check_equal(x, y, map_free_vars=False):
    tvm_tirx.ir.assert_structural_equal(x, y, map_free_vars)
    tvm_tirx.ir.assert_structural_equal(y, x, map_free_vars)

    xhash = tvm_ffi.structural_hash(x, map_free_vars)
    yhash = tvm_ffi.structural_hash(y, map_free_vars)

    assert xhash == yhash


def _check_json_roundtrip(x):
    xret = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(x))
    _check_equal(x, xret, map_free_vars=True)
    return xret


def _check_type_missing(ty):
    assert isinstance(ty, tvm_tirx.ir.Type)
    assert isinstance(ty, tvm_tirx.ir.MissingType)


def test_var() -> None:
    v0 = rx.Var("v0")
    assert v0.name == "v0"
    _check_type_missing(v0.ty)
    shape = [54, 96]
    v1 = rx.Var("v1", R.Tensor(shape, "float32"))
    assert v1.name == "v1"
    for s0, s1 in zip(v1.ty.shape, shape):
        assert s0 == s1
    tvm_tirx.ir.assert_structural_equal(v1.ty, rx.TensorType(shape, "float32"))


def test_var_name_keyword_compatibility() -> None:
    primary = tvm_tirx.ir.Var(name="primary")
    legacy = tvm_tirx.ir.Var(name_hint="legacy")
    assert primary.name == "primary"
    assert legacy.name == "legacy"
    assert not hasattr(primary, "name_hint")
    assert not hasattr(legacy, "name_hint")
    with pytest.raises(TypeError, match="Specify either name or name_hint, not both"):
        tvm_tirx.ir.Var(name="primary", name_hint="legacy")


def test_tensor_type_empty_dtype_is_unknown() -> None:
    tvm_tirx.ir.assert_structural_equal(
        rx.TensorType([1, 2], dtype=""), rx.TensorType([1, 2], None)
    )


def test_relax_expr_ty_running_example() -> None:
    m = tirx.Var("m", "int64")
    x = rx.Var("x", R.Tensor([m, 16], "float32"))

    assert isinstance(x.ty, tvm_tirx.ir.Type)
    assert x.ty.dtype == "float32"
    assert x.ty.ndim == 2

    call = rx.op.add(x, x)
    _check_type_missing(call.ty)

    bb = rx.BlockBuilder()
    normalized = bb.normalize(call)

    assert isinstance(normalized.ty, tvm_tirx.ir.Type)
    tvm_tirx.ir.assert_structural_equal(normalized.ty, x.ty)


def test_dataflow_var() -> None:
    v0 = rx.DataflowVar("v0")
    assert v0.name == "v0"
    _check_type_missing(v0.ty)

    shape = [54, 96]
    v1 = rx.DataflowVar("v1", R.Tensor(shape, "float16"))
    assert v1.name == "v1"

    assert isinstance(v1, rx.DataflowVar)
    tvm_tirx.ir.assert_structural_equal(v1.ty, rx.TensorType(shape, "float16"))


def test_dataflow_var_name_keyword_compatibility() -> None:
    primary = rx.DataflowVar(name="primary")
    legacy = rx.DataflowVar(name_hint="legacy")
    assert primary.name == "primary"
    assert legacy.name == "legacy"
    assert not hasattr(primary, "name_hint")
    assert not hasattr(legacy, "name_hint")
    with pytest.raises(TypeError, match="Specify either name or name_hint, not both"):
        rx.DataflowVar(name="primary", name_hint="legacy")


def test_tuple() -> None:
    v0 = rx.Var("v0")
    v1 = rx.Var("v1")
    t = rx.Tuple((v0, v1))

    assert t.fields[0] == v0
    assert t.fields[1] == v1
    assert t[0] == v0
    assert t[1] == v1
    assert t[-1] == v1
    assert t[-2] == v0

    with pytest.raises(IndexError, match="Tuple index out of range"):
        t[2]

    with pytest.raises(IndexError, match="Tuple index out of range"):
        t[-3]


def test_tuple_ty_inferred_on_construction():
    v0 = rx.Var("v0", rx.AnyType())
    v1 = rx.Var("v1", rx.AnyType())
    tup = rx.Tuple((v0, v1))

    assert tup.ty is not None
    tvm_tirx.ir.assert_structural_equal(tup.ty, rx.TupleType([rx.AnyType(), rx.AnyType()]))


def test_tuple_ty_requires_fields_with_known_ty():
    v0 = rx.Var("v0", rx.AnyType())
    v1 = rx.Var("v1")
    tup = rx.Tuple((v0, v1))

    _check_type_missing(tup.ty)


def test_match_cast() -> None:
    # match_cast([16, 8], [m, n])
    m = tirx.Var("m", ty="int64")
    n = tirx.Var("n", ty="int64")
    shape = rx.const([16, 8], "int32")
    var = rx.Var("v0", R.Shape())
    b0 = rx.MatchCast(var, shape, R.Tensor([m, n], "int32"))
    assert b0.value == shape
    assert b0.pattern[0] == m
    assert b0.pattern[1] == n
    assert b0.var is not None

    # var1: R.Tensor((m, n), "float32") =
    #   match_cast(var0: R.Tensor("float32", ndim=-1), R.Tensor((m, n), "float32"))
    value = rx.Var("value", R.Tensor("float32", ndim=-1))

    var = rx.Var("v1", R.Tensor([m, n], "float32"))
    b1 = rx.MatchCast(var, value, R.Tensor([m, n], "float32"))
    assert b1.value == value
    assert b1.pattern[0] == m
    assert b1.pattern[1] == n
    assert b1.var is not None


def test_match_cast() -> None:
    m = tirx.Var("m", ty="int64")
    n = tirx.Var("n", ty="int64")
    ivalue = rx.Var("input_value")
    ty = rx.TensorType([n, m], "float32")
    b0 = rx.MatchCast(rx.Var("v"), ivalue, ty)
    assert b0.value.same_as(ivalue)
    assert b0.ty == ty
    _check_json_roundtrip(b0)


def test_var_binding() -> None:
    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b0 = rx.VarBinding(v0, val)
    assert b0.var.name == "v0"
    assert b0.value == val


def test_binding_block() -> None:
    m = tirx.Var("m", ty="int64")
    n = tirx.Var("n", ty="int64")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchCast(rx.Var("v0"), shape, R.Tensor([m, n], "int32"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.BindingBlock([b0, b1])
    assert block0.bindings[0] == b0
    assert block0.bindings[1] == b1


def test_dataflow_block() -> None:
    m = tirx.Var("m", ty="int64")
    n = tirx.Var("n", ty="int64")
    shape = rx.const([16, 8], "int32")
    b0 = rx.MatchCast(rx.Var("v0"), shape, R.Tensor([m, n], "int32"))

    v0 = rx.Var("v0")
    val = rx.const(np.random.rand(24, 56))
    b1 = rx.VarBinding(v0, val)

    block0 = rx.DataflowBlock([b0, b1])
    assert block0.bindings[0] == b0
    assert block0.bindings[1] == b1
    assert isinstance(block0, rx.DataflowBlock)


def test_seq_expr() -> None:
    x = rx.Var("foo")
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]
    seqe = rx.SeqExpr(blocks, x)
    assert seqe.blocks[0] == blocks[0]
    assert seqe.body == x


def test_func():
    x = rx.Var("foo", R.Tensor(dtype="float32", ndim=2))
    bindings = [rx.VarBinding(x, rx.const(1))]
    blocks = [rx.BindingBlock(bindings)]

    seqe = rx.SeqExpr(blocks, x)
    ret_ty = R.Tensor(dtype="float32", ndim=-1)
    func = rx.Function([x], seqe, ret_ty)
    with_attr = func.with_attr("global_symbol", "func")
    with_attrs = func.with_attr({"global_symbol": "func", "num_input": 1})

    assert "global_symbol" not in func.attrs
    assert with_attr.params[0] == x
    assert with_attr.body == seqe
    assert with_attr.ret_ty == ret_ty
    assert with_attr.attrs["global_symbol"] == "func"
    assert with_attrs.attrs["global_symbol"] == "func"
    assert with_attrs.attrs["num_input"] == 1


def test_shape_of():
    shape = [96, 54]
    v1 = rx.Var("v1", R.Tensor(shape))
    s1 = rx.get_shape_of(v1)
    for x, y in zip(shape, s1):
        assert x == y


def test_shape_expr():
    m = tirx.Var("m", ty="int64")
    n = tirx.Var("n", ty="int64")
    s = rx.ShapeExpr([m, n])
    assert s.values[0] == m
    assert s.values[1] == n
    assert s[0] == m
    assert s[1] == n
    assert s[-1] == n
    assert s[-2] == m
    assert isinstance(s.ty, rx.ShapeType)

    with pytest.raises(IndexError, match="ShapeExpr index out of range"):
        s[2]

    with pytest.raises(IndexError, match="ShapeExpr index out of range"):
        s[-3]

    shape_expr = rx.ShapeExpr([10, 20])
    assert shape_expr.values[0] == 10
    assert shape_expr.values[1] == 20
    tvm_tirx.ir.assert_structural_equal(shape_expr.ty, R.Shape((10, 20)))

    x = rx.Var("v0", R.Tensor((10, 20), "float32"))
    assert x.ty.shape[0] == 10
    assert x.ty.shape[1] == 20
    tvm_tirx.ir.assert_structural_equal(x.ty.shape.ty, R.Shape((10, 20)))

    m = tirx.Var("m", "int32")
    with pytest.raises(RuntimeError, match="the value in ShapeType can only have dtype of int64"):
        rx.ShapeExpr([m, 3])


def test_prim_value_helper_from_python_scalar():
    int_value = rx.prim_value(1)
    assert isinstance(int_value, tvm_tirx.ir.IntImm)
    tvm_tirx.ir.assert_structural_equal(int_value.ty, tvm_tirx.ir.PrimType("int64"))
    assert int_value.value == 1

    np_int_value = rx.prim_value(np.int32(2))
    assert isinstance(np_int_value, tvm_tirx.ir.IntImm)
    tvm_tirx.ir.assert_structural_equal(np_int_value.ty, tvm_tirx.ir.PrimType("int64"))
    assert np_int_value.value == 2

    float_value = rx.prim_value(1.0)
    assert isinstance(float_value, tvm_tirx.ir.FloatImm)
    tvm_tirx.ir.assert_structural_equal(float_value.ty, tvm_tirx.ir.PrimType("float64"))
    assert float_value.value == 1.0

    np_float_value = rx.prim_value(np.float32(1.5))
    assert isinstance(np_float_value, tvm_tirx.ir.FloatImm)
    tvm_tirx.ir.assert_structural_equal(np_float_value.ty, tvm_tirx.ir.PrimType("float64"))
    assert np_float_value.value == 1.5


def test_prim_value_helper_preserves_prim_expr():
    float_imm = tvm_tirx.ir.FloatImm("float32", 1.0)
    assert rx.prim_value(float_imm).same_as(float_imm)
    assert R.prim_value(float_imm).same_as(float_imm)

    n = tirx.Var("n", "int64")
    expr = n + 1
    assert rx.prim_value(n).same_as(n)
    assert rx.prim_value(expr).same_as(expr)


def test_prim_value_helper_rejects_relax_expr():
    with pytest.raises(TypeError, match="Cannot convert"):
        rx.prim_value(rx.Var("x"))


def test_string_imm():
    s0 = tvm_tirx.ir.StringImm("hello")
    s1 = tvm_tirx.ir.StringImm("hello")
    assert s0.value == "hello"
    _check_equal(s0, s1)
    _check_json_roundtrip(s0)


def test_datatype_imm():
    d0 = tvm_tirx.ir.DataTypeImm(tvm_ffi.dtype("int32"))
    d1 = tvm_tirx.ir.DataTypeImm(tvm_ffi.dtype("int32"))
    assert d0.value == "int32"
    _check_equal(d0, d1)
    _check_json_roundtrip(d0)


def test_call():
    dtype = tvm_tirx.ir.PrimType("int32")
    func = rx.Var("func", rx.FuncType([dtype], dtype))
    arg = rx.Var("arg", dtype)
    call = rx.Call(func, [arg])
    assert call.op.same_as(func)
    assert len(call.args) == 1
    assert call.args[0].same_as(arg)


def test_call_accepts_core_expr_operator():
    """relax.Call aliases the core ir.Call constructor."""
    dtype = tvm_tirx.ir.PrimType("int32")
    func = rx.Var("func", dtype)
    arg = rx.Var("arg", dtype)

    call = rx.Call(func, [arg])
    assert call.op.same_as(func)
    assert len(call.args) == 1
    assert call.args[0].same_as(arg)
    _check_type_missing(call.ty)


def test_call_raises_error_for_missing_operator():
    """relax::Call requires a defined operator."""
    with pytest.raises(TypeError, match=r"Expected `ir.Expr` but got `None`"):
        rx.Call(None, [])


def test_call_reinfer_type_from_current_inputs():
    original_ty = rx.TensorType([2], "float32")
    changed_ty = rx.TensorType([4], "float16")
    x = rx.Var("x", original_ty)
    y = rx.Var("y", changed_ty)

    initial = rx.Call("relax.abs", [x])
    _check_equal(tvm_tirx.ir.reinfer_type(initial), original_ty)
    assert tvm_tirx.ir.reinfer_type(initial).same_as(original_ty)
    _check_equal(initial.ty, original_ty)

    unchanged = rx.Call(
        "relax.abs",
        [x],
        ty=original_ty,
    )
    assert tvm_tirx.ir.reinfer_type(unchanged).same_as(unchanged.ty)

    equivalent_ty = rx.TensorType(original_ty.shape, original_ty.dtype)
    equivalent = rx.Call(
        "relax.abs",
        [x],
        ty=equivalent_ty,
    )
    assert tvm_tirx.ir.reinfer_type(equivalent).same_as(equivalent_ty)

    stale = rx.Call(
        "relax.abs",
        [y],
        ty=original_ty,
    )
    _check_equal(tvm_tirx.ir.reinfer_type(stale), changed_ty)
    assert tvm_tirx.ir.reinfer_type(stale).same_as(changed_ty)
    _check_equal(stale.ty, original_ty)

    untyped = rx.Call("relax.abs", [rx.Var("untyped")])
    _check_type_missing(tvm_tirx.ir.reinfer_type(untyped))
    _check_type_missing(untyped.ty)
    untyped_explicit = rx.Call(
        "relax.abs",
        [rx.Var("untyped")],
        ty=original_ty,
    )
    _check_type_missing(tvm_tirx.ir.reinfer_type(untyped_explicit))
    assert untyped_explicit.ty.same_as(original_ty)
    _check_type_missing(tvm_tirx.ir.reinfer_type(rx.Call("relax.matmul", [x, x])))
    func = rx.Var("func", rx.FuncType([original_ty], original_ty))
    _check_type_missing(tvm_tirx.ir.reinfer_type(rx.Call(func, [x])))

    invalid = rx.Call(
        "relax.abs",
        [rx.Var("scalar", tvm_tirx.ir.PrimType("int32"))],
        ty=original_ty,
    )
    with pytest.raises(TypeError):
        tvm_tirx.ir.reinfer_type(invalid)

    out_ty = rx.TensorType([8], "float32")
    closure = rx.Var("closure", rx.AnyType())
    explicit = rx.Call("relax.invoke_closure", [closure, rx.Tuple([])], ty_args=[out_ty])
    _check_equal(tvm_tirx.ir.reinfer_type(explicit), out_ty)
    changed_explicit = rx.Call(
        "relax.invoke_closure", [closure, rx.Tuple([])], ty_args=[changed_ty]
    )
    _check_equal(tvm_tirx.ir.reinfer_type(changed_explicit), changed_ty)

    _check_equal(
        tvm_tirx.ir.reinfer_type(rx.op.astype(x, "float16")), rx.TensorType([2], "float16")
    )
    _check_equal(tvm_tirx.ir.reinfer_type(rx.op.astype(x, "int32")), rx.TensorType([2], "int32"))


def test_shared_operator_surface_preserves_relax_semantics():
    tensor_ty = rx.TensorType([2], "float32")
    x = rx.Var("x", tensor_ty)
    y = rx.Var("y", tensor_ty)

    assert (x == x) is True
    assert (x != x) is False
    assert (x == y) is False
    assert (x != y) is True
    assert isinstance(hash(x), int)

    assert isinstance(x + y, tvm_tirx.ir.Call)
    assert isinstance(-x, tvm_tirx.ir.Call)
    assert isinstance(x.astype("float16"), tvm_tirx.ir.Call)

    with pytest.raises(ValueError, match="Cannot use and"):
        bool(x)


def test_shared_operator_surface_rejects_non_tensor_relax_values():
    shape = rx.ShapeExpr([1, 2])
    message = "Operator overloading is not supported for expression type"

    assert (shape == shape) is True
    assert (shape != shape) is False
    assert bool(shape)

    with pytest.raises(TypeError, match="unsupported operand type"):
        shape + shape

    for operation in (
        lambda: -shape,
        lambda: shape.equal(shape),
        lambda: shape.astype("float32"),
    ):
        with pytest.raises(TypeError, match=message):
            operation()


def test_shared_operator_surface_calls_function_typed_values():
    tensor_ty = rx.TensorType([2], "float32")
    arg = rx.Var("arg", tensor_ty)
    func_ty = rx.FuncType([tensor_ty], tensor_ty)

    for func in (rx.Var("func", func_ty), rx.ExternFunc("extern_func", func_ty)):
        call = func(arg)
        assert isinstance(call, tvm_tirx.ir.Call)
        assert call.op.same_as(func)
        assert len(call.args) == 1
        assert call.args[0].same_as(arg)


def test_op_set_signature_preserves_custom_validator():
    op = tvm_tirx.ir.Op.get("relax.call_pure_packed")
    original_args = [(info.name, info.doc) for info in op.args_info]
    original_ty_args = [(info.name, info.doc) for info in op.ty_args_info]
    original_var_args = (op.var_args_info.name, op.var_args_info.doc)
    original_var_ty_args = (op.var_ty_args_info.name, op.var_ty_args_info.doc)
    try:
        op.set_signature()
        with pytest.raises(TypeError, match="call_pure_packed expects a function argument"):
            tvm_tirx.ir.Call(op, []).validate()
    finally:
        op.set_signature(
            original_args,
            ty_args=original_ty_args,
            var_args=original_var_args,
            var_ty_args=original_var_ty_args,
        )


_LEGACY_TIRX_VAR_JSON = """{
  "root_index": 6,
  "nodes": [
    {"type": "ir.SourceName", "data": "legacy_tirx.py"},
    {"type": "ir.SourceLoc", "data": {"source_name": 0, "start_line": 7, "start_column": 2,
      "end_line": 9, "end_column": 14}},
    {"type": "ir.UnknownLoc", "data": null},
    {"type": "ir.PrimType", "data": {"loc": 2, "dtype": "int64"}},
    {"type": "ffi.String", "data": "legacy"},
    {"type": "tirx.Var", "data": {"loc": 1, "ty": 3, "name": 4}},
    {"type": "tirx.Add", "data": {"loc": 1, "ty": 3, "a": 5, "b": 5}}
  ],
  "metadata": {"tvm_version": "0.26.dev0"}
}"""


def _make_pre_name_field_json(type_key):
    graph = json.loads(_LEGACY_TIRX_VAR_JSON)
    graph["root_index"] = 5
    var = graph["nodes"][5]
    var["type"] = type_key
    var["data"]["name_hint"] = var["data"].pop("name")
    return json.dumps(graph)


def _check_legacy_var(
    var, source_name, line, end_line, column, end_column, expected_type=tvm_tirx.ir.Var
):
    assert type(var) is expected_type
    assert var.name == "legacy"
    assert var.ty == tvm_tirx.ir.PrimType("int64")
    assert var.loc.source_name.name == source_name
    assert var.loc.start_line == line
    assert var.loc.end_line == end_line
    assert var.loc.start_column == column
    assert var.loc.end_column == end_column


_PRE_NAME_FIELD_DATAFLOW_VAR_JSON = _make_pre_name_field_json("relax.expr.DataflowVar")


@pytest.mark.parametrize(
    ("legacy_json", "expected_type"),
    [
        (_PRE_NAME_FIELD_DATAFLOW_VAR_JSON, tvm.relax.DataflowVar),
    ],
)
def test_var_name_legacy_json_load(legacy_json, expected_type):
    restored = tvm_tirx.ir.load_json(legacy_json)
    _check_legacy_var(restored, "legacy_tirx.py", 7, 9, 2, 14, expected_type)
    graph = json.loads(tvm_tirx.ir.save_json(restored))
    fields = graph["nodes"][graph["root_index"]]["data"]
    assert "name" in fields
    assert "name_hint" not in fields


def test_dataflow_var_json_is_not_migrated_to_canonical_var():
    dataflow_var = tvm.relax.DataflowVar("value", tvm_tirx.ir.PrimType("int64"))
    graph = json.loads(tvm_tirx.ir.save_json(dataflow_var))
    root = graph["nodes"][graph["root_index"]]
    assert root["type"] == "relax.expr.DataflowVar"
    restored = tvm_tirx.ir.load_json(json.dumps(graph))
    assert type(restored) is tvm.relax.DataflowVar


def _empty_relax_func():
    return rx.Function([], rx.Tuple([]))


def test_ir_module_from_expr_freshens_main_collision():
    main_gv = tvm_tirx.ir.GlobalVar("main")
    mod = tvm_tirx.IRModule.from_expr(_empty_relax_func(), {main_gv: _empty_relax_func()})

    assert sorted(gvar.name_hint for gvar in mod.get_global_vars()) == ["main", "main_1"]


def test_ir_module_from_expr_reuses_existing_global_symbol():
    foo_gv = tvm_tirx.ir.GlobalVar("foo")
    func = _empty_relax_func().with_attr("global_symbol", "foo")
    mod = tvm_tirx.IRModule.from_expr(func, {foo_gv: _empty_relax_func()})

    assert mod.get_global_var("foo").same_as(foo_gv)
    assert [gvar.name_hint for gvar in mod.get_global_vars()] == ["foo"]


if __name__ == "__main__":
    tvm_tirx.testing.main()


def test_make_shape_invalid_type():
    with pytest.raises(TypeError):
        make_shape(123)


def test_make_shape_valid_list():
    shape = make_shape([1, 2, 3])
    assert len(shape) == 3


def test_make_shape_valid_tuple():
    shape = make_shape((4, 5))
    assert len(shape) == 2


if __name__ == "__main__":
    tvm_tirx.testing.main()
