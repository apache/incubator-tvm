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
# ruff: noqa: F811, RUF005

from __future__ import annotations

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

m = T.dynamic("m", "int64")
n = T.dynamic("n", "int64")
x = rx.Var("x", R.Tensor([m, n], "float32"))
cond = rx.Var("cond", R.Tensor([], "bool"))


def build_function(blocks, params=[]):
    """Returns relax.function with given blocks"""
    seq_expr = rx.SeqExpr(blocks, blocks[-1].bindings[-1].var)
    func = rx.Function([x, cond] + params, seq_expr, R.Tensor("float32")).with_attr(
        "global_symbol", "foo"
    )
    return func


def test_var():
    # Error: Var gv0 is not defined
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    gv1 = rx.Var("gv1", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, gv0)
    bindings = [rx.VarBinding(gv1, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    # Error: Var gv0 is defined more than once
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, x)
    call_node2 = rx.op.multiply(x, x)
    bindings = [rx.VarBinding(gv0, call_node), rx.VarBinding(gv0, call_node2)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_dataflow_var():
    # Error: DataflowVar lv0 is not defined
    lv0 = rx.DataflowVar("lv0", R.Tensor([m, n], "float32"))
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, lv0)
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.DataflowBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    # Error: DataflowVar gv0 is defined more than once
    lv0 = rx.DataflowVar("lv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, x)
    call_node2 = rx.op.multiply(x, x)
    bindings = [rx.VarBinding(lv0, call_node), rx.VarBinding(lv0, call_node2)]
    blocks = [rx.DataflowBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    # Error: DataflowVar lv0 is defined outside DataflowBlock
    lv0 = rx.DataflowVar("lv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, x)
    bindings = [rx.VarBinding(lv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    # Error: DataflowVar lv0 is used outside DataflowBlock
    lv0 = rx.DataflowVar("lv0", R.Tensor([m, n], "float32"))
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(lv0, x)
    bindings = [rx.VarBinding(lv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_nested_tir_call_in_dependent_type_is_well_formed():
    p = rx.Var("p", tvm_tirx.ir.PrimType("int64"))
    inner = tirx.call_intrin("tirx.gpu_thread_invariant", p, ty="int64")
    outer = tirx.call_intrin("tirx.gpu_thread_invariant", inner, ty="int64")
    x = rx.Var("x", rx.TensorType([outer], "float32"))
    bb = rx.BlockBuilder()

    with bb.function("main", [p, x]):
        bb.emit_func_output(x)

    assert rx.analysis.check_well_formed(bb.get()) is None


def test_nested_tir_call_in_shape_expr_is_well_formed():
    p = rx.Var("p", tvm_tirx.ir.PrimType("int64"))
    inner = tirx.call_intrin("tirx.gpu_thread_invariant", p, ty="int64")
    outer = tirx.call_intrin("tirx.gpu_thread_invariant", inner, ty="int64")
    bb = rx.BlockBuilder()

    with bb.function("main", [p]):
        shape = bb.emit(rx.ShapeExpr([outer]))
        bb.emit_func_output(shape)

    rx.analysis.check_well_formed(bb.get())


def test_param_var():
    v0 = rx.Var("v0", R.Tensor([m, n], "float32"))
    v1 = rx.Var("v1", R.Tensor([m, n], "float32"))
    v2 = rx.Var("v2", R.Tensor([m, n], "float32"))
    bb = rx.BlockBuilder()
    with bb.function("func1", [v0, v1]):
        gv0 = bb.emit(rx.op.add(v0, v1))
        bb.emit_func_output(gv0)
    with bb.function("func2", [v0, v2]):
        gv0 = bb.emit(rx.op.add(v2, v1))
        bb.emit_func_output(gv0)
    mod = bb.get()
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_global_var():
    # Error: GlobalVar GlobalVar0 is not defined
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    globalvar = rx.GlobalVar("GlobalVar0")
    call_node = rx.Call(
        op=tvm_tirx.ir.Op.get("relax.call_tir"),
        args=[globalvar, rx.Tuple([x]), rx.ShapeExpr([m, n])],
        ty=tvm_tirx.ir.Type.missing(),
    )
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_provisional_call_constructor():
    op = tvm_tirx.ir.Op.get("relax.add")
    with pytest.raises(Exception, match="Call.args expected 2 arguments, got 1"):
        tvm_tirx.ir.Call(op, [x]).validate()

    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("provisional.py"), 1, 0, 1, 1)
    call = tvm_tirx.ir.Call("relax.add", [x], attrs={"key": 1}, loc=loc)
    assert isinstance(call, tvm_tirx.ir.Call)
    assert call.op.same_as(op)
    assert isinstance(call.ty, tvm_tirx.ir.MissingType)
    assert call.loc.same_as(loc)
    assert isinstance(call.attrs, tvm_tirx.ir.DictAttrs)
    assert len(call.ty_args) == 0
    assert isinstance(
        tvm_tirx.ir.Call(
            op,
            [x],
            ty="handle",
        ).ty,
        tvm_tirx.ir.PtrType,
    )
    with pytest.raises(TypeError, match="skip_validate"):
        tvm_tirx.ir.Call(op, [x], skip_validate=True)


def test_symbolic_var():
    # Error: Symbolic Var new_s is not defined
    new_s = tirx.Var("new_s", "int64")
    gv0 = rx.Var("gv0", R.Tensor([m, new_s], "int64"))
    call_node = rx.op.add(x, x)
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_symbolic_var_across_functions():
    # Error: Symbolic Var s presents across different functions
    s = tirx.Var("s", "int64")
    v0 = rx.Var("v0", R.Tensor([5, s], "float32"))
    v1 = rx.Var("v1", R.Tensor([s, 7], "float32"))
    bb = rx.BlockBuilder()
    with bb.function("func1", [v0]):
        bb.emit_func_output(v0)
    with bb.function("func2", [v1]):
        bb.emit_func_output(v1)
    mod = bb.get()
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_symbolic_var_invalid_type():
    with pytest.raises(RuntimeError, match="the value in ShapeType can only have dtype of int64"):
        dim = tirx.Var("dim", "float32")
        y = rx.Var("y", R.Tensor([dim], "float32"))
        gv0 = rx.Var("gv0", R.Tensor([dim], "float32"))
        call_node = rx.op.add(y, y)
        bindings = [rx.VarBinding(gv0, call_node)]
        blocks = [rx.BindingBlock(bindings)]
        func = build_function(blocks, [y])
        mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
        with pytest.raises((ValueError, TypeError)):
            rx.analysis.check_well_formed(mod, check_ty=False)


def test_seq_expr():
    # Error: SeqExpr in VarBinding
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    # build a SeqExpr
    gv1 = rx.Var("gv1", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, gv0)
    _bindings = [rx.VarBinding(gv1, call_node)]
    _blocks = [rx.BindingBlock(_bindings)]
    _seq_expr = rx.SeqExpr(_blocks, gv1)
    # build a Binding with the SeqExpr as value
    bindings = [rx.VarBinding(gv0, _seq_expr)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_recursive():
    scalar_ty = rx.TensorType(shape=[], dtype="int32")
    gv0 = rx.Var("gv0", scalar_ty)
    f = rx.Var("f", rx.FuncType([scalar_ty], scalar_ty))
    ipt = rx.Var("ipt", scalar_ty)
    x0 = rx.Var("x0", scalar_ty)
    x1 = rx.Var("x1", scalar_ty)
    x2 = rx.Var("x2", scalar_ty)
    y = rx.Var("y", scalar_ty)
    inner_block = rx.BindingBlock(
        [rx.VarBinding(x0, rx.const(2, "int32")), rx.VarBinding(y, rx.Call(f, [x0]))]
    )
    inner_func = rx.Function([ipt], rx.SeqExpr([inner_block], y), scalar_ty)
    outer_block = rx.BindingBlock(
        [
            rx.VarBinding(f, inner_func),
            rx.VarBinding(x1, rx.const(1, "int32")),
            rx.VarBinding(x2, rx.op.add(x1, rx.Call(f, [x1]))),
            rx.VarBinding(gv0, x2),
        ]
    )
    func = rx.Function([], rx.SeqExpr([outer_block], gv0), scalar_ty)
    mod = tvm_tirx.IRModule.from_expr(func)
    normalized = rx.transform.Normalize()(mod)
    rx.analysis.check_well_formed(normalized)


def test_if():
    # Error: Var defined in true/false branch is invisible in the outer scope
    # except the return Var, i.e the var in the last stmt
    # v_in_if is invisible in the outer scope
    v_in_if = rx.Var("v_in_if", R.Tensor([m, n], "float32"))
    # gv0 is visible in the outer scope
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    # build true branch
    true_bindings = [
        rx.VarBinding(v_in_if, rx.op.add(x, x)),
        rx.VarBinding(gv0, rx.op.multiply(x, x)),
    ]
    true_blocks = [rx.BindingBlock(true_bindings)]
    true_seq_expr = rx.SeqExpr(true_blocks, true_blocks[-1].bindings[-1].var)
    # build false branch
    false_bindings = [
        rx.VarBinding(v_in_if, rx.op.multiply(x, x)),
        rx.VarBinding(gv0, rx.op.add(x, x)),
    ]
    false_blocks = [rx.BindingBlock(false_bindings)]
    false_seq_expr = rx.SeqExpr(false_blocks, false_blocks[-1].bindings[-1].var)
    # build If node
    if_node = rx.IfExpr(cond=cond, true_branch=true_seq_expr, false_branch=false_seq_expr)
    gv1 = rx.Var("gv1", R.Tensor([m, n], "float32"))
    # try to call v_in_if defined in the true/false branch
    bindings = [rx.VarBinding(gv0, if_node), rx.VarBinding(gv1, v_in_if)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=True)


def test_if_non_seq_body():
    # Error: If node has a body that is not a seq node
    if_node = rx.IfExpr(cond=cond, true_branch=x, false_branch=x)
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", R.Tensor([m, n], "float32")),
                    if_node,
                )
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule.from_expr(func)
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    # on the other hand, if they're wrapped in a seq node, it's fine
    seq = rx.SeqExpr([], x)
    new_if_node = rx.IfExpr(cond=cond, true_branch=seq, false_branch=seq)
    new_blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", R.Tensor([m, n], "float32")),
                    new_if_node,
                )
            ]
        )
    ]
    new_func = build_function(new_blocks)
    new_mod = tvm_tirx.IRModule.from_expr(new_func)
    # apply normalization to fill in ty
    normalized = rx.transform.Normalize()(new_mod)
    rx.analysis.check_well_formed(normalized, check_ty=True)


def test_if_complex_condition():
    # Error: If condition must be a leaf expression
    cond_tuple = rx.Tuple([cond])
    cond_idx = rx.TupleGetItem(cond_tuple, 0)
    if_node = rx.IfExpr(cond_idx, rx.SeqExpr([], x), rx.SeqExpr([], x))
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(
                    rx.Var("gv1", R.Tensor([m, n], "float32")),
                    if_node,
                )
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule.from_expr(func)
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    cond_var = rx.Var("q", R.Tensor([], "bool"))
    new_if = rx.IfExpr(cond_var, rx.SeqExpr([], x), rx.SeqExpr([], x))
    blocks = [
        rx.BindingBlock(
            [
                rx.VarBinding(cond_var, cond_idx),
                rx.VarBinding(
                    rx.Var("gv1", R.Tensor([m, n], "float32")),
                    new_if,
                ),
            ]
        )
    ]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule.from_expr(func)
    # apply normalization to fill in ty
    normalized = rx.transform.Normalize()(mod)
    rx.analysis.check_well_formed(normalized, check_ty=True)


def test_tuple_get_item_nested():
    # Error: The tuple value in tuple get item must be a leaf expression
    nested_tup = rx.Var("t", rx.TupleType([rx.TupleType([rx.TensorType([], "int32")])]))
    double_idx = rx.TupleGetItem(rx.TupleGetItem(nested_tup, 0), 0)
    ret_var = rx.Var("r", R.Tensor([], "int32"))
    f = rx.Function(
        [nested_tup],
        rx.SeqExpr([rx.BindingBlock([rx.VarBinding(ret_var, double_idx)])], ret_var),
        ret_ty=R.Tensor(ndim=0, dtype="int32"),
    )
    f = f.with_attr("global_symbol", "f")
    mod = tvm_tirx.IRModule.from_expr(f)
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    # okay with an intermediate binding
    first_idx = rx.TupleGetItem(nested_tup, 0)
    idx_var = rx.Var("v", rx.TupleType([rx.TensorType([], "int32")]))
    second_idx = rx.TupleGetItem(idx_var, 0)
    new_f = rx.Function(
        [nested_tup],
        rx.SeqExpr(
            [
                rx.BindingBlock(
                    [rx.VarBinding(idx_var, first_idx), rx.VarBinding(ret_var, second_idx)]
                )
            ],
            ret_var,
        ),
        ret_ty=R.Tensor(ndim=0, dtype="int32"),
    )
    new_f = new_f.with_attr("global_symbol", "new_f")
    mod = tvm_tirx.IRModule.from_expr(new_f)
    # normalize in order to fill in checked type
    normalized = rx.transform.Normalize()(mod)
    rx.analysis.check_well_formed(normalized, check_ty=True)


def test_complex_seq_body():
    # Error: seq expr with a body that is not a leaf expression is not permitted
    x = rx.Var("x", R.Tensor([], "int32"))
    y = rx.Var("y", R.Tensor([], "int32"))
    func = rx.Function(
        [x, y],
        rx.SeqExpr([], rx.op.add(x, y)),
        R.Tensor(ndim=0, dtype="int32"),
    ).with_attr("global_symbol", "foo")
    mod = tvm_tirx.IRModule.from_expr(func)
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    # but if the result is bound, then it's okay
    z = rx.Var("z", R.Tensor([], "int32"))
    new_func = rx.Function(
        [x, y],
        rx.SeqExpr(
            [
                rx.BindingBlock(
                    [
                        rx.VarBinding(
                            var=z,
                            value=rx.op.add(x, y),
                        )
                    ]
                )
            ],
            z,
        ),
        R.Tensor(ndim=0, dtype="int32"),
    ).with_attr("global_symbol", "foo")
    new_mod = tvm_tirx.IRModule.from_expr(new_func)
    # normalize in order to fill in checked type
    normalized = rx.transform.Normalize()(new_mod)
    rx.analysis.check_well_formed(normalized, check_ty=True)


def test_inline_function():
    # Error: inline function is disallowed in Relax IR
    x = rx.Var("x", R.Tensor([], "int32"))
    y = rx.Var("y", R.Tensor([], "int32"))
    new_func = rx.Function(
        [],
        rx.SeqExpr(
            [
                rx.BindingBlock(
                    [
                        rx.VarBinding(
                            var=x,
                            value=tirx.Function([], tvm_tirx.ir.Evaluate(0)),
                        ),
                        rx.VarBinding(
                            var=y,
                            value=rx.Call(
                                op=tvm_tirx.ir.Op.get("relax.call_tir"),
                                args=[
                                    rx.GlobalVar("GlobalVar0"),
                                    rx.Tuple([x, tirx.Function([], tvm_tirx.ir.Evaluate(0))]),
                                    rx.ShapeExpr([]),
                                ],
                                ty=tvm_tirx.ir.Type.missing(),
                            ),
                        ),
                    ]
                )
            ],
            y,
        ),
        R.Tensor(ndim=0, dtype="int32"),
    ).with_attr("global_symbol", "foo")
    new_mod = tvm_tirx.IRModule.from_expr(new_func)
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(new_mod, check_ty=False)


def test_ANF():
    # Error: Nested Call
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    call_node = rx.op.add(x, rx.op.add(x, x))
    bindings = [rx.VarBinding(gv0, call_node)]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)

    # Error: Call Node in Tuple
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    bindings = [rx.VarBinding(gv0, rx.Tuple((x, rx.op.add(x, x))))]
    blocks = [rx.BindingBlock(bindings)]
    func = build_function(blocks)
    mod = tvm_tirx.IRModule({rx.GlobalVar("foo"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_global_var_vs_gsymbol():
    # Error: gsymbol "main1" not equals to the name in global var "main"
    gv0 = rx.Var("gv0", R.Tensor([m, n], "float32"))
    bindings = [rx.VarBinding(gv0, x)]
    blocks = [rx.DataflowBlock(bindings)]
    func = rx.Function(
        [x],
        rx.SeqExpr(blocks, gv0),
        R.Tensor(ndim=2, dtype="float32"),
    ).with_attr("global_symbol", "main1")
    mod = tvm_tirx.IRModule({rx.GlobalVar("main"): func})
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_nested_dataflow():
    scalar_ty = rx.TensorType(shape=[], dtype="int32")
    gv0 = rx.Var("gv0", scalar_ty)
    f = rx.DataflowVar("f", rx.FuncType([], scalar_ty))
    x0 = rx.DataflowVar("x0", scalar_ty)
    x1 = rx.DataflowVar("x1", scalar_ty)
    x2 = rx.DataflowVar("x2", scalar_ty)
    y = rx.Var("y", scalar_ty)
    inner_block = rx.DataflowBlock([rx.VarBinding(x0, rx.const(2, "int32")), rx.VarBinding(y, x0)])
    inner_func = rx.Function([], rx.SeqExpr([inner_block], y), scalar_ty)
    outer_block = rx.DataflowBlock(
        [
            rx.VarBinding(x1, rx.const(1, "int32")),
            rx.VarBinding(f, inner_func),
            rx.VarBinding(x2, rx.op.add(x1, rx.Call(f, []))),
            rx.VarBinding(gv0, x2),
        ]
    )
    func = rx.Function([], rx.SeqExpr([outer_block], gv0), scalar_ty)
    mod = tvm_tirx.IRModule.from_expr(func)
    normalized = rx.transform.Normalize()(mod)
    rx.analysis.check_well_formed(normalized)


def test_ty_args_tir_var_used_before_define_call_packed():
    # Error: Symbolic Var m1, n1 are not defined
    m1 = tirx.Var("m1", "int64")
    n1 = tirx.Var("n1", "int64")
    call = R.call_packed("my_func", x, ty_args=R.Tensor((m1, n1), "float32"))
    func = build_function([rx.BindingBlock([rx.VarBinding(rx.Var("gv"), call)])])
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_ty_args_tir_var_used_before_define_call_tir():
    # Error: Symbolic Var m1, n1 are not defined
    m1 = tirx.Var("m1", "int64")
    n1 = tirx.Var("n1", "int64")
    call = R.call_dps_packed(_ExternFunc("my_func"), x, ty_args=[R.Tensor((m1, n1), "float32")])
    func = build_function([rx.BindingBlock([rx.VarBinding(rx.Var("gv"), call)])])
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod, check_ty=False)


def test_ty_erase_to_well_formed():
    # Error: The return ty contains undefined symbolic vars
    """
    m, n = T.dynamic("m", "int64"), T.dynamic("n", "int64")
    m1, n1 = T.dynamic("m1", "int64"), T.dynamic("n1", "int64")

    @R.function
    def foo(x: R.Tensor((m, n), dtype="float32")) -> R.Tensor((m1, n1), dtype="float32"):
        gv = R.call_dps_packed(
            R.ExternFunc("my_func"), (x,), ty_args=[R.Tensor((m, n), dtype="float32")]
        )
        return gv
    """
    m1 = tirx.Var("m1", "int64")
    n1 = tirx.Var("n1", "int64")
    call = R.call_dps_packed(_ExternFunc("my_func"), x, ty_args=[R.Tensor((m, n), "float32")])
    blocks = [rx.BindingBlock([rx.VarBinding(rx.Var("gv"), call)])]
    seq_expr = rx.SeqExpr(blocks, blocks[-1].bindings[-1].var)
    func = rx.Function([x], seq_expr, R.Tensor((m1, n1), "float32")).with_attr(
        "global_symbol", "foo"
    )
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod)


def test_func_ty_well_formed():
    @R.function
    def foo():
        @R.function
        def local(x: R.Tensor([m, n], "float32")):
            return x

        return local

    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(foo))
    rx.analysis.check_well_formed(mod)


def test_conditional_in_dataflow_block():
    # error: not allowed to have a conditional inside a dataflow block
    x = rx.Var("x", rx.TensorType([], dtype="int32"))
    y = rx.Var("y", rx.TensorType([], dtype="int32"))
    block = rx.DataflowBlock([rx.VarBinding(y, rx.IfExpr(rx.const(True, dtype="bool"), x, x))])
    func = rx.Function([x], rx.SeqExpr([block], y), R.Tensor((), dtype="int32")).with_attr(
        "global_symbol", "foo"
    )
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod)


def test_unlabeled_impure():
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.Var("y")
    block = rx.BindingBlock([rx.VarBinding(y, rx.op.print("{}", x))])
    # print is impure, but the function is not labeled as impure
    func = rx.Function([x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32")).with_attr(
        "global_symbol", "foo"
    )
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod)


def test_labeled_impure():
    # the function is labeled impure so the impure operation is permitted
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.Var("y")
    block = rx.BindingBlock([rx.VarBinding(y, rx.op.print("{}", x))])
    # print is impure, but the function is not labeled as impure
    func = rx.Function(
        [x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32"), is_pure=False
    ).with_attrs({"global_symbol": "foo"})
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    rx.analysis.check_well_formed(mod)


def test_force_pure():
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.Var("y")
    block = rx.BindingBlock([rx.VarBinding(y, rx.op.print("{}", x))])
    # print is impure, but force_pure overrides the judgment
    func = rx.Function([x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32")).with_attrs(
        {"global_symbol": "foo", "relax.force_pure": True}
    )
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    rx.analysis.check_well_formed(mod)


def test_force_pure_improper():
    # we require both the is_pure and force_pure flags to be set together
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    # otherwise inoffensive, but the flags are wrong
    func = rx.Function(
        [x], rx.SeqExpr([], x), R.Tensor((), dtype="int32"), is_pure=False
    ).with_attrs({"global_symbol": "foo", "relax.force_pure": True})
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod)


def test_impure_in_dataflow_block():
    # even if force_pure is set, an impure operation cannot appear in a dataflow block
    x = rx.Var("x", R.Tensor((), dtype="int32"))
    y = rx.DataflowVar("y")
    block = rx.DataflowBlock([rx.VarBinding(y, rx.op.print("{}", x))])
    func = rx.Function([x], rx.SeqExpr([block], x), R.Tensor((), dtype="int32")).with_attrs(
        {"global_symbol": "foo", "relax.force_pure": True}
    )
    mod = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(mod)

    # The throwing form surfaces the offending impure call in its message.
    with pytest.raises(Exception) as excinfo:
        rx.analysis.check_well_formed(mod)
    assert 'R.print("{}", x)' in str(excinfo.value)


def test_well_formed_function():
    """Relax's well-formed check can be applied on a function"""

    @R.function
    def func(A: R.Tensor([16, 32], "float32"), B: R.Tensor([32, 64], "float32")):
        return R.matmul(A, B)

    rx.analysis.check_well_formed(func)


def test_well_formed_function_referencing_global_var():
    """GlobalVar may refer to other functions in the module

    If validating that a IRModule is well-formed, the GlobalVar must
    have a definition.  If validating that a relax.Function is
    well-formed, no GlobalVar definitions are available.
    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 32], "float32"), B: R.Tensor([32, 64], "float32")):
            return Module.subroutine(A, B)

        @R.function(private=True)
        def subroutine(A: R.Tensor([16, 32], "float32"), B: R.Tensor([32, 64], "float32")):
            return R.matmul(A, B)

    rx.analysis.check_well_formed(Module)
    rx.analysis.check_well_formed(Module["main"])
    rx.analysis.check_well_formed(Module["subroutine"])


def test_pass_dltensor_arg_to_tir():
    """Relax may pass R.Tensor as DLTensor

    In TIR, a `DLTensor*` argument with unknown shape and dtype is
    represented as a `tirx.Var` with
    `tvm::PtrType::VoidPtrType()`, rather than a buffer-typed
    parameter.  In Relax, this is represented as `R.Tensor`.  Calls
    from Relax to TIR that pass a tensor of unknown rank/shape are
    well-formed.

    In the test case below, a TIR function accepts an arbitrary
    `R.Tensor`, and returns a boolean value based on inspection of the
    runtime datatype.
    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor) -> T.bool:
            return R.call_pure_packed(
                tvm_tirx.ir.Op.get("relax.call_tir_packed"), Module.is_bfloat16_dtype, (A,)
            )

        @Ts.function(private=True)
        def is_bfloat16_dtype(tensor: T.handle) -> T.bool:
            T.func_attr({"tirx.is_scheduled": True, "tirx.is_host_func": True})

            # From #include <tvm/ir/prim/op.h>
            kDLTensorTypeCode = T.meta_var(5)
            kDLTensorTypeBits = T.meta_var(6)
            kDLTensorTypeLanes = T.meta_var(7)

            # From #include <dlpack/dlpack.h>
            kDLBfloat = T.meta_var(4)

            type_code = T.abi_field_get(tensor, 0, kDLTensorTypeCode, ty="uint8")
            type_bits = T.abi_field_get(tensor, 0, kDLTensorTypeBits, ty="uint8")
            type_lanes = T.abi_field_get(tensor, 0, kDLTensorTypeLanes, ty="uint16")

            is_bfloat16: T.bool = (
                (type_code == kDLBfloat) and (type_bits == 16) and (type_lanes == 1)
            )
            return is_bfloat16

    rx.analysis.check_well_formed(Module)


def test_call_tir_with_matching_arguments():
    """R.call_tir is well-formed when called with matching arguments"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.add_one, A, ty_args=[R.Tensor([16], "float16")])
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16"), B: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    rx.analysis.check_well_formed(Module)


def test_call_tir_with_interspersed_primitive_argument():
    """Primitive values are positional arguments in the call_tir tuple."""

    @I.ir_module
    class Module:
        @R.function
        def main(
            A: R.Tensor([16], "float16"),
            scale: T.float32,
            C: R.Tensor([16], "float16"),
        ):
            B = R.call_tir(Module.add_scaled, (A, scale, C), ty_args=[R.Tensor([16], "float16")])
            return B

        @Ts.function
        def add_scaled(
            A: T.Tensor([T.int64(16)], "float16"),
            scale: T.float32,
            C: T.Tensor([T.int64(16)], "float16"),
            B: T.Tensor([T.int64(16)], "float16"),
        ):
            for i in range(T.int64(16)):
                B[i] = A[i] + T.Cast("float16", scale) * C[i]

    rx.analysis.check_well_formed(Module)


def test_call_tir_with_incorrect_primitive_argument_dtype():
    """Primitive call_tir arguments must match the Function parameter dtype."""

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16"), scale: T.int64):
            B = R.call_tir(Module.scale, (A, scale), ty_args=[R.Tensor([16], "float16")])
            return B

        @Ts.function
        def scale(
            A: T.Tensor([T.int64(16)], "float16"),
            scale: T.float32,
            B: T.Tensor([T.int64(16)], "float16"),
        ):
            for i in range(T.int64(16)):
                B[i] = A[i] * T.Cast("float16", scale)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)
    with pytest.raises(ValueError, match="Argument 1 type mismatch"):
        rx.transform.Normalize()(Module)


def test_call_tir_shape_expr_is_not_a_primitive_argument():
    """ShapeExpr groups must be unpacked into positional primitive values."""

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main():
            B = R.call_tir(
                Module.make_tensor,
                (R.shape([1, 2]),),
                ty_args=[R.Tensor([1], "float32")],
            )
            return B

        @Ts.function
        def make_tensor(m: T.int64, n: T.int64, B: T.Tensor([T.int64(1)], "float32")):
            B[0] = T.Cast("float32", m + n)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_input_ndim():
    """Arguments to R.call_tir must have the correct dimensionality

    Here, the `add_one` function expects a 1-d input tensor, but is
    called with a 2-d tensor.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([4, 4], "float16")):
            B = R.call_tir(Module.add_one, A, ty_args=[R.Tensor([16], "float16")])
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16"), B: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_output_ndim():
    """Output shape R.call_tir must have the correct dimensionality

    Here, the `add_one` function requires a 1-d output tensor, but is
    provided with a 2-d tensor.
    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.add_one, A, ty_args=[R.Tensor([4, 4], "float16")])
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16"), B: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_input_shape():
    """Arguments to R.call_tir must have the correct shape

    Here, the `add_one` function expects an input tensor with 16
    elements, but is called with an input tensor with 32 elements.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([32], "float16")):
            B = R.call_tir(Module.add_one, A, ty_args=[R.Tensor([16], "float16")])
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16"), B: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_output_shape():
    """Output shape R.call_tir must have the correct shape

    Here, the `add_one` function requires an output tensor with 16
    elements, but is provided an output tensor with 32 elements.
    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.add_one, A, ty_args=[R.Tensor([32], "float16")])
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16"), B: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_input_dtype():
    """Arguments to R.call_tir must have the correct dtype

    Here, the `add_one` function expects an input tensor containing
    float16 value, but is called with an input tensor containing
    float32 values.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            B = R.call_tir(Module.add_one, A, ty_args=[R.Tensor([16], "float16")])
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16"), B: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_output_dtype():
    """Output shape R.call_tir must have the correct shape

    Here, the `add_one` function requires an output tensor that may be
    populated with float16 values, but is provided an output tensor
    that may be populated with float32 elements.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.add_one, A, ty_args=[R.Tensor([16], "float32")])
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16"), B: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi] + T.float16(1.0)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_with_correct_dynamic_output_shape():
    """Output shape R.call_tir may not be verifiable

    Here, the input arguments to the `reshape` function are not
    sufficient to infer the shape of the outputs.  This is legal,
    since the output shape is determined by the `out_ty` parameter.

    Inability to verify the output shape does not mean that the output
    shape is invalid.

    """

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.reshape, A, ty_args=[R.Tensor([2, 8], "float16")])
            return B

        @Ts.function
        def reshape(A: T.Tensor(16, "float16"), B: T.Tensor([M, N], dtype="float16")):
            for i, j in T.grid(M, N):
                with Ts.sblock("compute"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi * N + vj]

    rx.analysis.check_well_formed(Module)


@pytest.mark.xfail(reason="Not supported")
def test_call_tir_with_incorrect_dynamic_output_shape():
    """Output shape R.call_tir may not be verifiable

    Here, the input arguments to the `reshape` function are not
    sufficient to infer the shape of the outputs.  Even though the
    IRModule will not provide well-defined output due to the
    out-of-bounds read from buffer A, catching this error is beyond
    the current scope of the Relax well-formed checker.

    """

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.reshape, A, ty_args=[R.Tensor([16, 16], "float16")])
            return B

        @Ts.function
        def reshape(A: T.Tensor(16, "float16"), B: T.Tensor([M, N], dtype="float16")):
            for i, j in T.grid(M, N):
                with Ts.sblock("compute"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi * N + vj]

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_incorrect_dimensionality_of_output_shape():
    """Dimensionality may be verified

    Here, the input arguments to the `reshape` function are not
    sufficient to infer the shape of the outputs.

    Even though the output shape may not be inferred from the input
    arguments, the output dimensionality can still be inferred from
    the Function signature.  The IRModule below is ill-formed, because
    the Function requires a 2-d output argument, but is provided with
    a 3-d output argument.

    """

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir(Module.reshape, A, ty_args=[R.Tensor([2, 4, 2], "float16")])
            return B

        @Ts.function
        def reshape(A: T.Tensor(16, "float16"), B: T.Tensor([M, N], dtype="float16")):
            for i, j in T.grid(M, N):
                with Ts.sblock("compute"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vi * N + vj]

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


@pytest.mark.xfail(reason="Not yet supported")
def test_call_tir_output_shape_with_mixed_static_and_dynamic():
    """Some dimensions of the R.call_tir output shape may be verifiable

    Here, the input arguments to the `reshape` function are not
    sufficient to infer the shape of the outputs.  This is legal,
    since the output shape is taken from the `out_ty` parameter.

    Identifying this failure mode is not yet supported in the current
    implementation.  This is because the output is inferred as
    `R.Tensor(ndim=3, dtype="float16")`, and the explicit `out_ty`
    is a 3-d tensor.  The mismatch in the first dimension is not yet
    counted, because the entire tensor shape is removed by
    `EraseToWellDefined`.

    """

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([256], "float16")):
            B = R.call_tir(Module.reshape, A, ty_args=[R.Tensor([8, 16, 2], "float16")])
            return B

        @Ts.function
        def reshape(A: T.Tensor(256, "float16"), B: T.Tensor([16, M, N], dtype="float16")):
            for i, j, k in T.grid(16, M, N):
                with Ts.sblock("compute"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    B[vi, vj, vk] = A[vi * N * M + vj * N + vk]

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_with_correct_inferred_dynamic_output_shape():
    """Some dynamic output shapes of R.call_tir may be inferred

    Here, the `flatten` function is dynamic, and will flatten any 2-d
    TIR buffer.  Even though it is dynamic, the input shapes are
    sufficient to infer that `M==8` and `N==4`.  As a result, the
    output shape of `[M*N]` can be inferred to be `[32]`, and the
    shape specified in `out_ty` can be validated.

    """

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([8, 4], "float16")):
            B = R.call_tir(Module.flatten, A, ty_args=[R.Tensor([32], "float16")])
            return B

        @Ts.function
        def flatten(A: T.Tensor([M, N], dtype="float16"), B: T.Tensor([M * N], dtype="float16")):
            for i in T.grid(M * N):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi // N, vi % N]

    rx.analysis.check_well_formed(Module)


def test_call_tir_with_incorrect_inferred_dynamic_output_shape():
    """Some dynamic output shapes of R.call_tir may be inferred

    Here, the `flatten` function is dynamic, and will flatten any 2-d
    TIR buffer.  Even though it is dynamic, the input shapes are
    sufficient to infer that `M==8` and `N==4`.  As a result, the
    output shape of `[M*N]` can be inferred to be `[32]`, and the
    shape specified in `out_ty` can be validated.

    This unit test is identical to the above test
    `test_call_tir_with_correct_inferred_dynamic_output_shape`, except
    that the output shape is explicitly specified as `[64]`, which is
    caught as a mismatch from the expected output shape.

    """

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([8, 4], "float16")):
            B = R.call_tir(Module.flatten, A, ty_args=[R.Tensor([64], "float16")])
            return B

        @Ts.function
        def flatten(A: T.Tensor([M, N], dtype="float16"), B: T.Tensor([M * N], dtype="float16")):
            for i in T.grid(M * N):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi // N, vi % N]

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_with_dtensor_arguments():
    """R.call_tir and R.dist.call_tir share the same operation

    Both `R.call_tir` and `R.dist.call_tir` produce the same
    "relax.call_tir" operation, differing only in the Type of
    their arguments.  Normalization of "relax.call_tir" must handle
    `R.DTensor` arguments.

    """

    # from tvm.script.parser import relax as R

    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @I.ir_module
    class Module:
        I.module_attrs({"device_num": 4})
        I.module_global_infos({"mesh": [R.dist.device_mesh([4], I.Range(0, 4))]})

        @R.function
        def main(A: R.dist.DTensor([8, 4], "float16", "mesh[0]", "S[0]")):
            B = R.dist.call_tir(
                Module.flatten, A, ty_args=[R.dist.DTensor([64], "float16", "mesh[0]", "S[0]")]
            )
            return B

        @Ts.function
        def flatten(A: T.Tensor([M, N], dtype="float16"), B: T.Tensor([M * N], dtype="float16")):
            for i in T.grid(M * N):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = A[vi // N, vi % N]

    rx.analysis.check_well_formed(Module)


def test_call_tir_inplace_with_correct_shapes():
    """R.call_tir_inplace is well-formed when called with matching arguments"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir_inplace(
                Module.add_one,
                A,
                inplace_indices=[0],
                ty_args=[R.Tensor([16], "float16")],
            )
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi] = A[vi] + T.float16(1.0)

    rx.analysis.check_well_formed(Module)


def test_call_tir_inplace_with_incorrect_shapes():
    """R.call_tir_inplace is ill-formed when output shape does not match input"""

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16")):
            B = R.call_tir_inplace(
                Module.add_one,
                A,
                inplace_indices=[0],
                ty_args=[R.Tensor([32], "float16")],
            )
            return B

        @Ts.function
        def add_one(A: T.Tensor(16, "float16")):
            for i in range(16):
                with Ts.sblock("compute"):
                    vi = Ts.axis.remap("S", [i])
                    A[vi] = A[vi] + T.float16(1.0)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_call_tir_inplace_with_some_allocated_outputs():
    """R.call_tir_inplace may contain some non-inplace outputs"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16], "float16"), B: R.Tensor([32], "float16")):
            out = R.call_tir_inplace(
                Module.add_one,
                (A, B),
                inplace_indices=[-1, 1],
                ty_args=[_TupleType([R.Tensor([16], "float16"), R.Tensor([32], "float16")])],
            )
            return out

        @Ts.function
        def add_one(
            A: T.Tensor(16, "float16"),
            B: T.Tensor(32, "float16"),
            C: T.Tensor(16, "float16"),
        ):
            for i in range(32):
                with Ts.sblock("inplace_B"):
                    vi = Ts.axis.remap("S", [i])
                    B[vi] = B[vi] + T.float16(1.0)

            for i in range(16):
                with Ts.sblock("output_C"):
                    vi = Ts.axis.remap("S", [i])
                    C[vi] = A[vi] + T.float16(1.0)

    rx.analysis.check_well_formed(Module)


def test_var_binding_must_have_compatible_ty():
    """Variables must accurately describe their contents

    To be well-formed, the inferred type must not conflict with
    the Type annotations.

    """

    # The function is equivalent to the TVMScript below.  However,
    # TVMScript applies additional checks that would catch this error
    # while parsing.  In order to validate the well-formed checker
    # itself, this test directly constructs the function withoutusing
    # TVMScript, skipping the TVMScript-specific checks.
    #
    # @R.function
    # def main(
    #     A: R.Tensor(shape=[128, 32], dtype="float32"),
    # ):
    #     B: R.Tensor(shape=[128, 32], dtype="int32") = A
    #     return B

    param = tvm.relax.Var("A", R.Tensor(shape=[128, 32], dtype="float32"))
    var = tvm.relax.Var("B", R.Tensor(shape=[128, 32], dtype="int32"))
    binding = tvm.relax.VarBinding(var, param)
    body = tvm.relax.SeqExpr([tvm.relax.BindingBlock([binding])], var)
    tvm.relax.expr._update_type(body, var.ty)
    main = tvm.relax.Function([param], body)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(main)


def test_var_binding_may_have_less_constrained_ty():
    """Type of variable may be less specific than expression

    The Type annotation of a variable is not required to be an
    exact match to the expression's Type, and may provide less
    specific information than the inference would provide.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            B: R.Any = R.add(A, A)
            return B

    assert isinstance(Module["main"].body.blocks[0].bindings[0].var.ty, tvm.relax.AnyType), (
        "Validity of this test requires a variable with R.Any type"
    )

    rx.analysis.check_well_formed(Module)


def test_var_binding_with_incomplete_ty_must_be_consistent():
    """Type of variable must be accurate

    Even though Type annotation may be less specific, the
    information that they do contain must be correct.

    """

    # The function is equivalent to the TVMScript below.  However,
    # TVMScript applies additional checks that would catch this error
    # while parsing.  In order to validate the well-formed checker
    # itself, this test directly constructs the function withoutusing
    # TVMScript, skipping the TVMScript-specific checks.
    #
    #   @R.function
    #   def main(
    #       A: R.Tensor(shape=[128, 32], dtype="float32"),
    #   ):
    #       B: R.Tensor(ndim=3) = A
    #       return B

    param = tvm.relax.Var("A", R.Tensor(shape=[128, 32], dtype="float32"))
    var = tvm.relax.Var("B", R.Tensor(ndim=3, dtype="int32"))
    binding = tvm.relax.VarBinding(var, param)
    body = tvm.relax.SeqExpr([tvm.relax.BindingBlock([binding])], var)
    tvm.relax.expr._update_type(body, var.ty)
    main = tvm.relax.Function([param], body)

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(main)


def test_incomplete_ty_must_be_consistent():
    """Type annotations must be accurate

    Even though Type annotation may be less specific, the
    information that they do contain must be correct.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
            B: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            C: R.Tensor(ndim=3) = R.add(A, B)
            return C

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_ty_annotations_must_be_correct():
    """Type annotations must be correct

    To be well-formed, the inferred type must not conflict with
    the Type annotations.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
            B: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            C: R.Tensor(shape=[128, 32], dtype="int32") = R.add(A, B)
            return C

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_ty_may_be_incomplete():
    """Type annotations may be less specific

    The Type annotations are not required to be an exact match
    to the inferred Type, and may provide less specific
    information than the inference would provide.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
            B: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            C: R.Any = R.add(A, B)
            return C

    rx.analysis.check_well_formed(Module)


def test_incomplete_ty_must_be_consistent():
    """Type annotations must be accurate

    Even though Type annotation may be less specific, the
    information that they do contain must be correct.

    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(
            A: R.Tensor(shape=[128, 32], dtype="float32"),
            B: R.Tensor(shape=[128, 32], dtype="float32"),
        ):
            C: R.Tensor(ndim=3) = R.add(A, B)
            return C

    with pytest.raises((ValueError, TypeError)):
        rx.analysis.check_well_formed(Module)


def test_stop_lift_params_optional_type_arg():
    x = rx.Var("x", rx.TensorType((4,), "float32"))
    out_ty = rx.TensorType((4,), "float32", rx.VDevice("llvm"))
    for ty_args in ([], [out_ty]):
        call = tvm_tirx.ir.Call("relax.builtin.stop_lift_params", [x], ty_args=ty_args)
        func = rx.Function([x], call, ty_args[0] if ty_args else x.ty)
        normalized = rx.transform.Normalize()(tvm_tirx.IRModule.from_expr(func))
        rx.analysis.check_well_formed(normalized)


if __name__ == "__main__":
    tvm_tirx.testing.main()
