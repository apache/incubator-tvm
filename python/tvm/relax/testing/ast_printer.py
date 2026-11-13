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
# pylint: disable=redefined-builtin, abstract-method, arguments-differ
"""
Utility script for printing Relax modules as AST diagrams,
only intended to show how the AST is put together.
It is not a pretty-printer and, in fact, is more of an ugly-printer,
but it can be useful for tutorials and debugging.
"""

from collections.abc import Iterable

import tvm_tirx
from tvm import relax
from tvm.relax import ExprFunctor
from tvm_tirx.ir import Expr


def wrap_quotes(text: str) -> str:
    """
    Wraps the text in quotes.
    """
    return f'"{text}"'


class ASTPrinter(ExprFunctor):
    """
    Class for recursing down ASTs and printing them in a very simple format,
    mainly for instructive purposes and, perhaps, debugging.
    """

    def __init__(
        self,
        indent_str="    ",
        include_ty_annotations=True,
        include_call_attrs=True,
    ):
        self.indent_str = indent_str
        self.include_ty_annotations = include_ty_annotations
        self.include_call_attrs = include_call_attrs

    def visit_expr(self, expr: relax.Expr) -> str:
        # extend so we also dispatch to bindings and binding blocks,
        # a little silly but IRFunctor hasn't been ported to Python
        if isinstance(expr, relax.DataflowBlock):
            return self.visit_dataflow_block_(expr)
        if isinstance(expr, relax.BindingBlock):
            return self.visit_binding_block_(expr)
        if isinstance(expr, relax.Binding):
            return self.visit_binding_(expr)
        return super().visit_expr(expr)

    def indent(self, text: str) -> str:
        """
        Indent all lines of the input.
        """
        if text == "":
            return ""
        lines = text.split("\n")
        return self.indent_str + f"\n{self.indent_str}".join(lines)

    def build_ast_node(self, nodename: str, force_newline=False, **kwargs: str) -> str:
        """
        Returns 'nodename(..., fields[i][0]=fields[i][1], ...)'
        with appropriate indentation
        """
        return self.build_list(
            map(lambda field: f"{field[0]}={field[1]}", kwargs.items()),
            open_tok=f"{nodename}(",
            close_tok=")",
            force_newline=force_newline,
        )

    def build_expr(self, node: relax.Expr, nodename: str, force_newline=False, **kwargs: str):
        """
        Renders a Relax expression as a string using `build_ast_node`.
        Handles whether to include the ty fields.
        """
        fields = kwargs.copy()
        if not isinstance(node.ty, tvm_tirx.ir.MissingType) and self.include_ty_annotations:
            fields["ty"] = self.visit_ty_(node.ty)
        return self.build_ast_node(nodename, force_newline=force_newline, **fields)

    def build_list(
        self, members: Iterable[str], open_tok="[", close_tok="]", force_newline=False
    ) -> str:
        """
        Builds a list of the members given, appropriately indented,
        with each field on a line.
        (special case: if there is only one field, then we do not put it on a new line
        unless that field contains a newline or `force_newline` is set to true).
        `open_tok` and `close_tok` are used to open and close the list, respectively.
        """
        mem_list = list(members)
        if not mem_list:
            return f"{open_tok}{close_tok}"
        if len(mem_list) == 1 and not force_newline and "\n" not in mem_list[0]:
            return f"{open_tok}{mem_list[0]}{close_tok}"
        member_lines = ",\n".join(map(self.indent, mem_list))
        return f"{open_tok}\n{member_lines}\n{close_tok}"

    def visit_generic_const_(self, op: tvm_tirx.ir.GenericConst) -> str:
        # simple rule of thumb: keep scalars inline, but anything larger goes on a new one
        force_newline = isinstance(op.value, tvm_tirx.runtime.Tensor) and len(op.value.shape) > 0
        return self.build_expr(op, "GenericConst", force_newline=force_newline, value=str(op.value))

    def visit_tuple_(self, op: relax.Tuple) -> str:
        return self.build_expr(op, "Tuple", fields=self.build_list(map(self.visit_expr, op.fields)))

    def visit_dataflow_var_(self, op: relax.DataflowVar) -> str:
        return self.build_expr(op, "DataflowVar", name=wrap_quotes(op.name))

    def visit_var_(self, op: relax.Var) -> str:
        return self.build_expr(op, "Var", name=wrap_quotes(op.name))

    def visit_shape_expr_(self, op: relax.ShapeExpr) -> str:
        return self.build_expr(
            op, "ShapeExpr", values=self.build_list(map(self.visit_prim_expr_field_, op.values))
        )

    def visit_extern_func_(self, op: relax.ExternFunc) -> str:
        # ExternFunc does not inherit from relax.Expr either,
        # so it doesn't have ty fields and we don't use build_expr
        return self.build_ast_node("ExternFunc", global_symbol=wrap_quotes(op.global_symbol))

    def visit_global_var_(self, op: relax.GlobalVar) -> str:
        return self.build_expr(op, "GlobalVar", name_hint=wrap_quotes(op.name_hint))

    def visit_function_(self, op: relax.Function) -> str:
        fields = {
            "params": self.build_list(map(self.visit_expr, op.params)),
            "body": self.visit_expr(op.body),
            "ret_ty": self.visit_ty_(op.ret_ty),
            "is_pure": op.is_pure,
        }
        if op.attrs:
            fields["attrs"] = self.build_list(
                map(
                    lambda kv: f"{wrap_quotes(str(kv[0]))}: {wrap_quotes(str(kv[1]))}",
                    op.attrs.items(),
                ),
                open_tok="{",
                close_tok="}",
            )
        return self.build_expr(op, "Function", **fields)

    def visit_call_(self, op: relax.Call) -> str:
        fields = {
            "op": self.visit_expr(op.op),
            "args": self.build_list(map(self.visit_expr, op.args)),
        }
        if op.ty_args:
            fields["ty_args"] = self.build_list(map(self.visit_ty_, op.ty_args))
        if op.attrs and self.include_call_attrs:

            def display_attrs(attr_key):
                attr_val = op.attrs[attr_key]

                if isinstance(attr_val, str):
                    # attrs can be strings but also other types;
                    # we want to wrap strings in quotes
                    # (__repr__ would work but it uses single quotes)
                    attr_val = wrap_quotes(attr_val)
                elif isinstance(attr_val, tvm_tirx.ir.IntImm):
                    if attr_val.dtype == "bool":
                        attr_val = bool(attr_val.value)
                    else:
                        attr_val = int(attr_val.value)

                return f"{wrap_quotes(attr_key)}: {attr_val}"

            fields["attrs"] = self.build_list(
                map(display_attrs, op.attrs.keys()),
                open_tok="{",
                close_tok="}",
            )
        return self.build_expr(op, "Call", **fields)

    def visit_seq_expr_(self, op: relax.SeqExpr) -> str:
        return self.build_expr(
            op,
            "SeqExpr",
            blocks=self.build_list(map(self.visit_binding_block_, op.blocks)),
            body=self.visit_expr(op.body),
        )

    def visit_if_expr_(self, op: relax.IfExpr) -> str:
        return self.build_expr(
            op,
            "IfExpr",
            cond=self.visit_expr(op.cond),
            true_branch=self.visit_expr(op.true_branch),
            false_branch=self.visit_expr(op.false_branch),
        )

    def visit_string_imm_(self, op: tvm_tirx.ir.StringImm) -> str:
        return self.build_expr(op, "StringImm", value=wrap_quotes(op.value))

    def visit_data_type_imm_(self, op: tvm_tirx.ir.DataTypeImm) -> str:
        return self.build_expr(op, "DataTypeImm", value=str(op.value))

    def visit_op_(self, op: tvm_tirx.ir.Op) -> str:
        # TODO: List other attributes?
        # op is not actually a Relax expr and does not have
        # ty fields, so we don't use build_expr here
        return self.build_ast_node("Op", name=wrap_quotes(op.name))

    def visit_prim_expr_field_(self, prim_expr: Expr) -> str:
        # TODO: We may want to print Expr ASTs, but this is a simplification for now
        value = prim_expr.name if isinstance(prim_expr, tvm_tirx.ir.Var) else str(prim_expr)
        return self.build_ast_node("Expr", value=f"`{value}`")

    def visit_expr_fallback_(self, op: Expr) -> str:
        if not tvm_tirx.ir.is_prim_expr(op):
            raise ValueError(f"Invalid Relax expression {op} ({type(op)})")
        return self.visit_prim_expr_field_(op)

    def visit_tuple_getitem_(self, op: relax.TupleGetItem) -> str:
        return self.build_expr(
            op,
            "TupleGetItem",
            tuple_value=self.visit_expr(op.tuple_value),
            index=str(op.index),
        )

    def visit_type_(self, type_node: relax.Type) -> str:
        """
        Recurse down types and print their ASTs too
        """
        if isinstance(type_node, relax.ShapeType):
            return self.build_ast_node("ShapeType", ndim=str(type_node.ndim))
        if isinstance(type_node, relax.AnyType):
            return self.build_ast_node("AnyType")
        if isinstance(type_node, relax.PackedFuncType):
            return self.build_ast_node("PackedFuncType")
        if isinstance(type_node, tvm_tirx.ir.StringType):
            return self.build_ast_node("StringType")
        if isinstance(type_node, tvm_tirx.ir.PrimType):
            return self.build_ast_node("PrimType", dtype=type_node.dtype)
        if isinstance(type_node, relax.TensorType):
            fields = {}
            if type_node.ndim is not None:
                fields["ndim"] = str(type_node.ndim)
            if type_node.dtype != "":
                fields["dtype"] = type_node.dtype
            return self.build_ast_node("TensorType", **fields)
        if isinstance(type_node, relax.TupleType):
            return self.build_ast_node(
                "TupleType", fields=self.build_list(map(self.visit_type_, type_node.fields))
            )
        if isinstance(type_node, relax.FuncType):
            fields = {}
            if type_node.params is not None:
                fields["params"] = self.build_list(map(self.visit_type_, type_node.params))
            fields["ret"] = self.visit_type_(type_node.ret)
            fields["purity"] = bool(type_node.purity)
            return self.build_ast_node(
                "FuncType",
                **fields,
            )
        raise ValueError(f"Invalid Relax Type {type_node} ({type(type_node)})")

    def visit_ty_(self, ty_node: relax.Type) -> str:
        """
        Recurse down type and print their ASTs too
        """
        if isinstance(ty_node, relax.ShapeType):
            fields = {}
            fields["ndim"] = str(ty_node.ndim)
            if ty_node.values is not None:
                fields["values"] = self.build_list(map(self.visit_prim_expr_field_, ty_node.values))
            return self.build_ast_node("ShapeType", **fields)
        elif isinstance(ty_node, relax.AnyType):
            return self.build_ast_node("AnyType")
        elif isinstance(ty_node, tvm_tirx.ir.StringType):
            return self.build_ast_node("StringType")
        elif isinstance(ty_node, tvm_tirx.ir.PrimType):
            return self.build_ast_node("PrimType", dtype=ty_node.dtype)
        elif isinstance(ty_node, relax.TensorType):
            fields = {}
            fields["dtype"] = ty_node.dtype
            if ty_node.shape is not None:
                fields["shape"] = self.visit_expr(ty_node.shape)
            else:
                fields["ndim"] = str(ty_node.ndim)
            return self.build_ast_node("TensorType", **fields)
        elif isinstance(ty_node, relax.TupleType):
            return self.build_ast_node(
                "TupleType",
                fields=self.build_list(map(self.visit_ty_, ty_node.fields)),
            )
        elif isinstance(ty_node, relax.FuncType):
            fields = {}
            if ty_node.params is not None:
                fields["params"] = self.build_list(map(self.visit_ty_, ty_node.params))
            fields["ret"] = self.visit_ty_(ty_node.ret)
            fields["purity"] = bool(ty_node.purity)
            return self.build_ast_node("FuncType", **fields)
        else:
            raise ValueError(f"Invalid Relax Type {ty_node} ({type(ty_node)})")

    def visit_binding_block_(self, block: relax.BindingBlock) -> str:
        """
        Recurse down binding blocks
        """
        return self.build_ast_node(
            "BindingBlock",
            bindings=self.build_list(map(self.visit_binding_, block.bindings), force_newline=True),
        )

    def visit_dataflow_block_(self, block: relax.DataflowBlock) -> str:
        """
        Recurse down a dataflow block
        """
        return self.build_ast_node(
            "DataflowBlock",
            bindings=self.build_list(map(self.visit_binding_, block.bindings), force_newline=True),
        )

    def visit_binding_(self, binding: relax.Binding) -> str:
        """
        Distinguish between binding types
        """
        if isinstance(binding, relax.MatchCast):
            return self.visit_match_cast_(binding)
        if isinstance(binding, relax.VarBinding):
            return self.visit_var_binding_(binding)
        raise ValueError(f"Invalid binding type in {binding}: {type(binding)}")

    def visit_match_cast_(self, match_cast: relax.MatchCast) -> str:
        """
        Handle match shape
        """
        fields = {
            "var": self.visit_expr(match_cast.var),
            "value": self.visit_expr(match_cast.value),
            "ty": self.visit_ty_(match_cast.ty),
        }
        return self.build_ast_node("MatchCast", **fields)

    def visit_var_binding_(self, var_binding: relax.VarBinding) -> str:
        """
        Handle ordinary var bindings
        """
        return self.build_ast_node(
            "VarBinding",
            var=self.visit_expr(var_binding.var),
            value=self.visit_expr(var_binding.value),
        )


def dump_ast(
    exp: relax.Expr,
    indent_str="    ",
    include_ty_annotations=True,
    include_call_attrs=True,
) -> str:
    """
    Dump an AST in a text format.
    Can vary the indentation string and choose whether to include
    type and shape annotations or call attributes.
    """
    printer = ASTPrinter(
        indent_str=indent_str,
        include_ty_annotations=include_ty_annotations,
        include_call_attrs=include_call_attrs,
    )
    return printer.visit_expr(exp)
