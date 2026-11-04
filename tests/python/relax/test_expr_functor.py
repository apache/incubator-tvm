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
# ruff: noqa: E501, F841
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax, te
from tvm.relax import PyExprMutator, PyExprVisitor
from tvm.relax import script as R
from tvm.relax.expr import (
    BindingBlock,
    DataflowBlock,
    DataflowVar,
    Expr,
    ExternFunc,
    Function,
    GlobalVar,
    IfExpr,
    MatchCast,
    SeqExpr,
    ShapeExpr,
    Tuple,
    TupleGetItem,
    Var,
    VarBinding,
)
from tvm_tirx import tirx
from tvm_tirx.ir import Call, GenericConst, Op, StringImm
from tvm_tirx.ir.base import assert_structural_equal

m, n = tirx.Var("m", "int64"), tirx.Var("n", "int64")
x = relax.Var("x", R.Tensor([n], "float32"))
y = relax.Var("y", R.Tensor([m, n], "float32"))
bb = relax.BlockBuilder()


@relax.expr_functor.visitor
class BasicVisitor(PyExprVisitor):
    """Default ExprVisitor"""


class ASTLog:
    """Helper class to log AST"""

    def __init__(self) -> None:
        self.log = []
        self.indent = "\t"
        self.level = 0

    def push_scope(self):
        self.level += 1

    def pop_scope(self):
        self.level -= 1

    def add(self, s: str):
        self.log.append(self.indent * self.level + s)

    def __str__(self) -> str:
        return "\n".join(self.log)


@relax.expr_functor.visitor
class ASTPrinter(PyExprVisitor):
    """Print relax AST in structured format. The shape of Node is ignored."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_generic_const_(self, op: GenericConst) -> None:
        self.log.add("GenericConst")

    def visit_global_var_(self, op: GlobalVar) -> None:
        self.log.add("GlobalVar")

    def visit_tuple_(self, op: Tuple) -> None:
        self.log.add("Tuple")
        self.log.push_scope()
        for field in op.fields:
            self.visit_expr(field)
        self.log.pop_scope()

    def visit_var_(self, op: Var) -> None:
        self.log.add("Var")

    def visit_dataflow_var_(self, op: DataflowVar) -> None:
        self.log.add("DataflowVar")

    def visit_function_(self, op: Function) -> None:
        self.log.add("Function")
        self.log.push_scope()
        for param in op.params:
            self.visit_var_def(param)

        self.visit_expr(op.body)
        self.log.pop_scope()

    def visit_call_(self, op: Call) -> None:
        self.log.add("Call")
        self.log.push_scope()
        self.visit_expr(op.op)

        for arg in op.args:
            self.visit_expr(arg)
        self.log.pop_scope()

    def visit_if_expr_(self, op: IfExpr) -> None:
        self.log.add("IfExpr")
        self.log.push_scope()
        self.visit_expr(op.cond)
        self.visit_expr(op.true_branch)
        self.visit_expr(op.false_branch)
        self.log.pop_scope()

    def visit_op_(self, op: Op) -> None:
        self.log.add("Op")

    def visit_tuple_getitem_(self, op: TupleGetItem) -> None:
        self.log.add("TupleGetItem")
        self.log.push_scope()
        self.visit_expr(op.tuple_value)
        self.log.pop_scope()

    def visit_expr_fallback_(self, op: Expr) -> None:
        self.log.add("ExprFallback")

    def visit_string_imm_(self, op: StringImm) -> None:
        self.log.add("StringImm")

    def visit_shape_expr_(self, op: ShapeExpr) -> None:
        self.log.add("ShapeExpr")

    def visit_extern_func_(self, op: ExternFunc) -> None:
        self.log.add("ExternFunc")

    def visit_seq_expr_(self, op: SeqExpr) -> None:
        self.log.add("SeqExpr")
        self.log.push_scope()
        for block in op.blocks:
            self.visit_binding_block(block)
        self.visit_expr(op.body)
        self.log.pop_scope()

    def visit_var_binding_(self, binding: VarBinding) -> None:
        self.log.add("VarBinding")
        self.log.push_scope()
        self.visit_expr(binding.value)
        self.visit_var_def(binding.var)
        self.log.pop_scope()

    def visit_match_cast_(self, binding: MatchCast) -> None:
        self.log.add("MatchCast")
        self.log.push_scope()
        self.visit_var_def(binding.var)
        self.visit_expr(binding.value)
        self.log.pop_scope()

    def visit_binding_block_(self, block: BindingBlock) -> None:
        self.log.add("BindingBlock")
        self.log.push_scope()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.pop_scope()

    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        self.log.add("DataflowBlock")
        self.log.push_scope()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.pop_scope()

    def visit_var_def_(self, var: Var) -> None:
        self.log.add("VarDef")

    def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
        self.log.add("DataflowVarDef")


@relax.expr_functor.mutator
class BasicMutator(PyExprMutator):
    """Default ExprMutator"""


@relax.expr_functor.mutator
class ASTPostPrinterMutator(PyExprMutator):
    """Print relax AST in the post order format."""

    def __init__(self) -> None:
        super().__init__()
        self.log = ASTLog()

    def visit_generic_const_(self, op: GenericConst) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("GenericConst")
        return op

    def visit_global_var_(self, op: GlobalVar) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("GlobalVar")
        return op

    def visit_tuple_(self, op: Tuple) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Tuple")
        return op

    def visit_var_(self, op: Var) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Var")
        return op

    def visit_dataflow_var_(self, op: DataflowVar) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("DataflowVar")
        return op

    def visit_function_(self, op: Function) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Function")
        return op

    def visit_call_(self, op: Call) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Call")
        return op

    def visit_if_expr_(self, op: IfExpr) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("IfExpr")
        return op

    def visit_op_(self, op: Op) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("Op")
        return op

    def visit_tuple_getitem_(self, op: TupleGetItem) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("TupleGetItem")
        return op

    def visit_expr_fallback_(self, op: Expr) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("ExprFallback")
        return op

    def visit_string_imm_(self, op: StringImm) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("StringImm")
        return op

    def visit_shape_expr_(self, op: ShapeExpr) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("ShapeExpr")
        return op

    def visit_extern_func_(self, op: ExternFunc) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("ExternFunc")
        return op

    def visit_seq_expr_(self, op: SeqExpr) -> Expr:
        op = self.visit_expr_post_order(op)
        self.log.add("SeqExpr")
        return op

    def visit_var_binding_(self, binding: VarBinding) -> None:
        """Identical with ExprMutator::VisitBinding_(const VarBindingObj* binding) on the C++ side."""
        new_value = self.visit_expr(binding.value)
        new_var = self.visit_var_def(binding.var)

        self.log.add("VarBinding")
        if binding.var.same_as(new_var) and binding.value.same_as(new_value):
            self.builder_.emit_normalized(binding)
            return

        temp = self.with_type(new_var, new_value.ty)
        if not temp.same_as(new_var):
            new_var = temp
            self.set_var_remap(binding.var, new_var)

        self.builder_.emit_normalized(VarBinding(new_var, new_value))

    def visit_match_cast_(self, binding: MatchCast) -> None:
        """Identical with ExprMutator::VisitBinding_(const MatchCastObj* binding) on the C++ side."""
        new_var = self.visit_var_def(binding.var)
        new_value = self.visit_expr(binding.value)

        temp = self.with_type(new_var, binding.ty)
        if not temp.same_as(new_var):
            new_var = temp
            self.set_var_remap(binding.var, new_var)

        self.log.add("MatchCast")
        self.builder_.emit_normalized(MatchCast(new_var, new_value, binding.ty))

    def visit_binding_block_(self, block: BindingBlock) -> BindingBlock:
        """Identical with ExprMutator::VisitBindingBlock_(const BindingBlockObj* block) on the C++ side."""
        self.builder_._begin_binding_block()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.add("BindingBlock")
        return self.builder_._end_block()

    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        """Identical with ExprMutator::VisitBindingBlock_(const DataflowBlockObj* block) on the C++ side."""
        self.builder_._begin_dataflow_block()
        for binding in block.bindings:
            self.visit_binding(binding)
        self.log.add("DataflowBlock")
        return self.builder_._end_block()

    def visit_var_def_(self, var: Var) -> None:
        """Identical with ExprMutator::VisitVarDef_(const VarObj* var) on the C++ side."""
        self.log.add("VarDef")
        return var

    def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
        """Identical with ExprMutator::VisitVarDef_(const DataflowVarObj* var) on the C++ side."""
        self.log.add("DataflowVarDef")
        return var


def basic_check(expr, visitor_str, mutator_str):
    def visit(f, expr):
        if isinstance(expr, relax.Expr):
            return f.visit_expr(expr)
        elif isinstance(expr, relax.BindingBlock):
            return f.visit_binding_block(expr)

    # check no overloading case
    basic_visitor = BasicVisitor()
    visit(basic_visitor, expr)

    # check the output log
    log_visitor = ASTPrinter()
    visit(log_visitor, expr)
    assert str(log_visitor.log) == visitor_str

    # check no overloading case
    basic_mutator = BasicMutator()
    # skip normalize GlobalVar since it requires context IRModule to get ty
    if isinstance(expr, relax.Expr) and not isinstance(expr, relax.GlobalVar):
        expr = bb.normalize(expr)
        assert_structural_equal(visit(basic_mutator, expr), expr)

    # check the output log and return value
    post_log_mutator = ASTPostPrinterMutator()
    if isinstance(expr, relax.Expr) and not isinstance(expr, relax.GlobalVar):
        expr = bb.normalize(expr)
        assert_structural_equal(visit(post_log_mutator, expr), expr)
        assert str(post_log_mutator.log) == mutator_str


def test_constant():
    basic_check(relax.const(1.0), "GenericConst", "GenericConst")


def test_var():
    basic_check(x, "Var", "Var")


def test_dataflow_var():
    lv = relax.DataflowVar("lv", R.Tensor([n], "float32"))
    basic_check(lv, "DataflowVar", "DataflowVar")


def test_tuple():
    t = relax.Tuple([x, y])
    basic_check(t, "\n".join(["Tuple", "\tVar", "\tVar"]), "\n".join(["Var", "Var", "Tuple"]))


def test_global_var():
    gv = relax.GlobalVar("gv")
    basic_check(gv, "GlobalVar", "GlobalVar")


def test_seq_expr():
    bindings = [relax.VarBinding(x, relax.const(1))]
    blocks = [relax.BindingBlock(bindings)]
    seq_expr = relax.SeqExpr(blocks, x)
    basic_check(
        seq_expr,
        "\n".join(
            [
                "SeqExpr",
                "\tBindingBlock",
                "\t\tVarBinding",
                "\t\t\tGenericConst",
                "\t\t\tVarDef",
                "\tVar",
            ]
        ),
        "\n".join(["GenericConst", "VarDef", "VarBinding", "BindingBlock", "Var", "SeqExpr"]),
    )


def test_shape_expr():
    x = relax.ShapeExpr([m, n])
    basic_check(x, "ShapeExpr", "\n".join(["Var", "Var", "ShapeExpr"]))


def test_prim_expr():
    basic_check(tvm_tirx.ir.IntImm("int64", 1), "", "")
    basic_check(tvm_tirx.ir.FloatImm("float32", 1.0), "", "")
    basic_check(m + n, "\n".join(["Var", "Var"]), "\n".join(["Var", "Var"]))
    basic_check(
        tvm_tirx.tirx.call_extern("test", ty="int32"),
        "\n".join(["Call", "\tOp", "\tStringImm"]),
        "\n".join(["Op", "StringImm", "Call"]),
    )


def test_call():
    call_node = relax.op.add(x, y)
    basic_check(
        call_node,
        "\n".join(["Call", "\tOp", "\tVar", "\tVar"]),
        "\n".join(["Op", "Var", "Var", "Var", "Var", "ShapeExpr", "Call"]),
    )

    tensor_load = te.placeholder((1,), name="A")(0)
    BasicVisitor().visit_expr(tensor_load)
    assert BasicMutator().visit_expr(tensor_load).same_as(tensor_load)

    visitor = ASTPrinter()
    visitor.visit_expr(tensor_load)
    assert str(visitor.log) == "\n".join(["Call", "\tOp", "\tExprFallback"])

    mutator = ASTPostPrinterMutator()
    assert mutator.visit_expr(tensor_load).same_as(tensor_load)
    assert str(mutator.log) == "\n".join(["Op", "ExprFallback", "Call"])


def test_if():
    if_node = relax.IfExpr(x, x, x)
    basic_check(
        if_node,
        "\n".join(["IfExpr", "\tVar", "\tSeqExpr", "\t\tVar", "\tSeqExpr", "\t\tVar"]),
        "\n".join(["Var", "Var", "SeqExpr", "Var", "SeqExpr", "IfExpr"]),
    )


def test_tuple_getitem():
    tuple_getitem_node = relax.TupleGetItem(relax.Tuple([x, y]), 0)
    basic_check(
        tuple_getitem_node,
        "\n".join(["TupleGetItem", "\tTuple", "\t\tVar", "\t\tVar"]),
        "\n".join(["Var", "Var", "Tuple", "TupleGetItem"]),
    )


def test_binding_block():
    bb._begin_binding_block()
    gv0 = bb.emit(relax.op.add(x, y))
    gv1 = bb.match_cast(y, R.Tensor([m, n], "float32"))
    b0 = bb._end_block()
    basic_check(
        b0,
        "\n".join(
            [
                "BindingBlock",
                "\tVarBinding",
                "\t\tCall",
                "\t\t\tOp",
                "\t\t\tVar",
                "\t\t\tVar",
                "\t\tVarDef",
                "\tMatchCast",
                "\t\tVarDef",
                "\t\tVar",
            ]
        ),
        "\n".join(
            [
                "Op",
                "Var",
                "Var",
                "Call",
                "ShapeExpr",
                "VarDef",
                "VarBinding",
                "Var",
                "ShapeExpr",
                "ShapeExpr",
                "VarDef",
                "MatchCast",
                "BindingBlock",
            ]
        ),
    )


def test_dataflow_block():
    bb._begin_dataflow_block()
    lv0 = bb.emit(relax.op.add(x, y))
    gv1 = bb.match_cast(y, R.Tensor([m, n], "float32"))
    b0 = bb._end_block()
    basic_check(
        b0,
        "\n".join(
            [
                "DataflowBlock",
                "\tVarBinding",
                "\t\tCall",
                "\t\t\tOp",
                "\t\t\tVar",
                "\t\t\tVar",
                "\t\tDataflowVarDef",
                "\tMatchCast",
                "\t\tDataflowVarDef",
                "\t\tVar",
            ]
        ),
        "\n".join(
            [
                "Op",
                "Var",
                "Var",
                "Call",
                "ShapeExpr",
                "DataflowVarDef",
                "VarBinding",
                "Var",
                "ShapeExpr",
                "ShapeExpr",
                "DataflowVarDef",
                "MatchCast",
                "DataflowBlock",
            ]
        ),
    )


def test_function():
    bindings = [relax.VarBinding(x, relax.const(1))]
    blocks = [relax.BindingBlock(bindings)]
    seq_expr = relax.SeqExpr(blocks, x)
    func = relax.Function([x], seq_expr, R.Tensor([n], "float32"))
    basic_check(
        func,
        "\n".join(
            [
                "Function",
                "\tVarDef",
                "\tSeqExpr",
                "\t\tBindingBlock",
                "\t\t\tVarBinding",
                "\t\t\t\tGenericConst",
                "\t\t\t\tVarDef",
                "\t\tVar",
            ]
        ),
        "\n".join(
            [
                "VarDef",
                "GenericConst",
                "VarDef",
                "VarBinding",
                "BindingBlock",
                "Var",
                "SeqExpr",
                "Function",
            ]
        ),
    )


def test_extern_func():
    func = relax.ExternFunc("f")
    basic_check(func, "ExternFunc", "ExternFunc")


def test_inherit():
    # The internal class is not instantiated.
    class InternalVisitor(PyExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_call_(self, op: Call) -> None:
            self.log.add("InternalCall")
            self.log.push_scope()
            self.visit_expr(op.op)

            for arg in op.args:
                self.visit_expr(arg)
            self.log.pop_scope()

        def visit_var_(self, op: Var) -> None:
            self.log.add("Var")

        def visit_op_(self, op: Op) -> None:
            self.log.add("Op")

    @relax.expr_functor.visitor
    class LeafVisitor(InternalVisitor):
        def visit_call_(self, op: Call) -> None:
            self.log.add("LeafCall")
            self.log.push_scope()
            self.visit_expr(op.op)

            for arg in op.args:
                self.visit_expr(arg)
            self.log.pop_scope()

    call_node = relax.op.add(x, y)
    lv = LeafVisitor()
    lv.visit_expr(call_node)
    assert str(lv.log) == "\n".join(["LeafCall", "\tOp", "\tVar", "\tVar"])


def test_inherit_with_cls():
    # The decorator converts `InternalVisitor` to a wrapper class.
    @relax.expr_functor.visitor
    class InternalVisitor(PyExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_call_(self, op: Call) -> None:
            self.log.add("InternalCall")
            self.log.push_scope()
            self.visit_expr(op.op)

            for arg in op.args:
                self.visit_expr(arg)
            self.log.pop_scope()

        def visit_var_(self, op: Var) -> None:
            self.log.add("Var")

        def visit_op_(self, op: Op) -> None:
            self.log.add("Op")

    # `InternalVisitor._cls` refers to the original `InternalVisitor` users defined.
    @relax.expr_functor.visitor
    class LeafVisitor(InternalVisitor._cls):
        def visit_call_(self, op: Call) -> None:
            self.log.add("LeafCall")
            self.log.push_scope()
            self.visit_expr(op.op)

            for arg in op.args:
                self.visit_expr(arg)
            self.log.pop_scope()

    call_node = relax.op.add(x, y)
    iv = InternalVisitor()
    iv.visit_expr(call_node)
    assert str(iv.log) == "\n".join(["InternalCall", "\tOp", "\tVar", "\tVar"])

    lv = LeafVisitor()
    lv.visit_expr(call_node)
    assert str(lv.log) == "\n".join(["LeafCall", "\tOp", "\tVar", "\tVar"])


def test_wrong_inherit():
    @relax.expr_functor.visitor
    class InternalVisitor(PyExprVisitor):
        def visit_call_(self, op: Call) -> None:
            pass

    with pytest.raises(
        TypeError,
        match="Inheritance from a decorated object `LeafVisitor` is not allowed. Please inherit from `LeafVisitor._cls`.",
    ):

        @relax.expr_functor.visitor
        class LeafVisitor(InternalVisitor):
            def visit_call_(self, op: Call) -> None:
                pass


@R.function
def dummy(x: R.Tensor((10, 10))):
    lv = R.add(x, R.const(1))
    with R.dataflow():
        gv = lv
        R.output(gv)
    return gv


def test_call_visitor_super():
    @relax.expr_functor.visitor
    class InternalVisitor(PyExprVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_binding_block_(self, block: relax.BindingBlock) -> None:
            self.log.add("BindingBlock")
            super().visit_binding_block_(block)

        def visit_dataflow_block_(self, block: DataflowBlock) -> None:
            self.log.add("DataflowBlock")
            super().visit_dataflow_block_(block)

        def visit_var_binding_(self, binding: relax.VarBinding) -> None:
            self.log.add("VarBinding")
            super().visit_var_binding_(binding)

        def visit_call_(self, op: Call) -> None:
            self.log.add("InternalCall")
            super().visit_call_(op)  # call PyExprVisitor.visit_call_

        def visit_var_def_(self, var: Var) -> None:
            self.log.add("VarDef")
            super().visit_var_def_(var)

        def visit_dataflow_var_def_(self, var: Var) -> None:
            self.log.add("DataflowVarDef")
            super().visit_dataflow_var_def_(var)

        def visit_var_(self, op: Var) -> None:
            self.log.add("Var")

        def visit_op_(self, op: Op) -> None:
            self.log.add("Op")

    @relax.expr_functor.visitor
    class LeafVisitor(InternalVisitor._cls):
        def visit_call_(self, op: Call) -> None:
            self.log.add("LeafCall")
            super().visit_call_(op)  # call InternalVisit.visit_call_

    call_node = relax.op.add(x, y)
    iv = InternalVisitor()
    iv.visit_expr(call_node)
    assert str(iv.log) == "\n".join(["InternalCall", "Op", "Var", "Var"])

    lv = LeafVisitor()
    lv.visit_expr(call_node)
    assert str(lv.log) == "\n".join(["LeafCall", "InternalCall", "Op", "Var", "Var"])

    lv = LeafVisitor()
    lv.visit_expr(dummy)
    assert str(lv.log) == "\n".join(
        [
            "VarDef",
            "BindingBlock",
            "VarBinding",
            "LeafCall",
            "InternalCall",
            "Op",
            "Var",
            "VarDef",
            "DataflowBlock",
            "VarBinding",
            "Var",
            "VarDef",
            "Var",
        ]
    )


def test_call_mutator_super():
    @relax.expr_functor.mutator
    class InternalMutator(PyExprMutator):
        def __init__(self) -> None:
            super().__init__()
            self.log = ASTLog()

        def visit_binding_block_(self, block: relax.BindingBlock) -> None:
            self.log.add("BindingBlock")
            return super().visit_binding_block_(block)

        def visit_dataflow_block_(self, block: DataflowBlock) -> None:
            self.log.add("DataflowBlock")
            return super().visit_dataflow_block_(block)

        def visit_var_binding_(self, binding: relax.VarBinding) -> None:
            self.log.add("VarBinding")
            return super().visit_var_binding_(binding)

        def visit_call_(self, op: Call) -> None:
            self.log.add("InternalCall")
            return super().visit_call_(op)  # call PyExprMutator.visit_call_

        def visit_var_def_(self, var: Var) -> None:
            self.log.add("VarDef")
            return super().visit_var_def_(var)

        def visit_dataflow_var_def_(self, var: Var) -> None:
            self.log.add("DataflowVarDef")
            return super().visit_dataflow_var_def_(var)

        def visit_var_(self, op: Var) -> None:
            self.log.add("Var")
            return super().visit_var_(op)  # call PyExprMutator.visit_var_

        def visit_op_(self, op: Op) -> None:
            self.log.add("Op")
            return super().visit_op_(op)  # call PyExprMutator.visit_op_

    @relax.expr_functor.mutator
    class LeafMutator(InternalMutator._cls):
        def visit_call_(self, op: Call) -> None:
            self.log.add("LeafCall")
            return super().visit_call_(op)  # call InternalMutator.visit_call_

    call_node = relax.op.add(x, y)
    im = InternalMutator()
    im.visit_expr(call_node)
    assert str(im.log) == "\n".join(["InternalCall", "Op", "Var", "Var"])

    lm = LeafMutator()
    lm.visit_expr(call_node)
    assert str(lm.log) == "\n".join(["LeafCall", "InternalCall", "Op", "Var", "Var", "Var", "Var"])

    lm = LeafMutator()
    lm.visit_expr(dummy)
    assert str(lm.log) == "\n".join(
        [
            "VarDef",
            "BindingBlock",
            "VarBinding",
            "LeafCall",
            "InternalCall",
            "Op",
            "Var",
            "VarDef",
            "DataflowBlock",
            "VarBinding",
            "Var",
            "VarDef",
            "Var",
        ]
    )


def test_function_parameter_mutation():
    @relax.expr_functor.mutator
    class ParamMutator(PyExprMutator):
        def __init__(self, shape_replacements):
            super().__init__()
            self.shape_replacements = shape_replacements

        def visit_var_def_(self, var):
            if var.name in self.shape_replacements:
                new_shape = self.shape_replacements[var.name]
                new_ty = relax.TensorType(new_shape, dtype=var.ty.dtype)
                return relax.Var(f"{var.name}_with_new_shape", new_ty)
            else:
                return var

    @R.function(private=True)
    def before(A: R.Tensor((16, 32), "float32"), B: R.Tensor((32, 64), "float32")) -> R.Tensor(
        (16, 64), "float32"
    ):
        return R.matmul(A, B)

    @R.function(private=True)
    def expected(A: R.Tensor((1, 32), "float32"), B: R.Tensor((32, 64), "float32")) -> R.Tensor(
        (1, 64), "float32"
    ):
        return R.matmul(A, B)

    after = ParamMutator({"A": (1, 32)}).visit_expr(before)
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_unmapped_python_mutator_var_is_absent():
    @tvm.relax.expr_functor.mutator
    class Mutator(tvm.relax.PyExprMutator):
        pass

    mutator = Mutator()
    old = tvm.relax.Var("old", tvm_tirx.ir.PrimType("int32"))
    new = tvm.relax.Var("new", tvm_tirx.ir.PrimType("int32"))
    assert mutator.get_var_remap(old) is None
    mutator.set_var_remap(old, new)
    assert mutator.get_var_remap(old).same_as(new)


if __name__ == "__main__":
    tvm_tirx.testing.main()
