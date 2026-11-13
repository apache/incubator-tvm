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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, arguments-differ
# ruff: noqa: RUF012
"""The expression functor of Relax."""

from collections.abc import Callable

import tvm_ffi

import tvm_tirx
from tvm import te as _te
from tvm.s_tir.utils import derived_object
from tvm_tirx import tirx as _tirx
from tvm_tirx.ir import Call, DataTypeImm, GenericConst, Op, StringImm, TensorLoad, is_prim_expr
from tvm_tirx.ir.module import IRModule

from . import _ffi_api
from .block_builder import BlockBuilder
from .expr import (
    Binding,
    BindingBlock,
    DataflowBlock,
    DataflowVar,
    Expr,
    ExternFunc,
    Function,
    GlobalVar,
    IfExpr,
    Location,
    MatchCast,
    SeqExpr,
    ShapeExpr,
    Tuple,
    TupleGetItem,
    Var,
    VarBinding,
)
from .type import Type

visitor = derived_object
"""
A decorator to wrap user-customized PyExprVisitor as TVM object _PyExprVisitor.

Parameters
----------
visitor_cls : PyExprVisitor
    The user-customized PyExprVisitor.

Returns
-------
cls : _PyExprVisitor
    The decorated TVM object _PyExprVisitor(ExprVisitor on the C++ side).

Example
-------
.. code-block:: python

    @relax.expr_functor.visitor
    class MyExprVisitor(PyExprVisitor):
        # customize visit function
        def visit_call_(self, op: Call) -> None:
            # just for demo purposes
            ...
    # myvisitor is now a special visitor that visit every Call with
    # user-customized visit_call_
    myvisitor = MyExprVisitor()
    # apply myvisitor to Expr/Binding/BindingBlock/VarDef
    myvisitor.visit_expr(expr)
    myvisitor.visit_binding(binding)
    myvisitor.visit_binding_block(bindingblock)
    myvisitor.visit_var_def(var)
"""

mutator = derived_object
"""
A decorator to wrap user-customized PyExprMutator as TVM object _PyExprMutator.
Note:  Cannot override visit function and post-order rewrite at the same time.

Parameters
----------
mutator_cls : PyExprMutator
    The user-customized PyExprMutator.

Returns
-------
cls : _PyExprMutator
    The decorated TVM object _PyExprMutator(ExprMutator on the C++ side).

Example
-------
.. code-block:: python

    @relax.expr_functor.mutator
    class MyExprMutator(PyExprMutator):
        # customize rewrite function
        def visit_tuple_(self, op: Tuple) -> Expr:
            # just for demo purposes
            ...

    # mymutator is now a special mutator that rewrite every Tuple with
    # user-customized visit_tuple_
    mymutator = MyExprMutator()
    # apply mymutator to Expr/Binding/BindingBlock/VarDef
    mymutator.visit_expr(expr)
    mymutator.visit_binding(binding)
    mymutator.visit_binding_block(bindingblock)
    mymutator.visit_var_def(var)
"""


class ExprFunctor:
    """
    An abstract visitor defined over Expr.
    Defines the default dispatch over expressions, and
    implements memoization.
    """

    def visit_expr(self, expr: Expr) -> Expr:
        """Apply the visitor to an expression."""
        if isinstance(expr, GenericConst):  # type: ignore
            ret = self.visit_generic_const_(expr)
        elif isinstance(expr, Tuple):
            ret = self.visit_tuple_(expr)
        elif isinstance(expr, DataflowVar):
            ret = self.visit_dataflow_var_(expr)
        elif isinstance(expr, Var):
            ret = self.visit_var_(expr)
        elif isinstance(expr, ShapeExpr):
            ret = self.visit_shape_expr_(expr)
        elif isinstance(expr, ExternFunc):
            ret = self.visit_extern_func_(expr)
        elif isinstance(expr, GlobalVar):  # type: ignore
            ret = self.visit_global_var_(expr)
        elif isinstance(expr, Function):
            ret = self.visit_function_(expr)
        elif isinstance(expr, Call):  # type: ignore
            ret = self.visit_call_(expr)
        elif isinstance(expr, TensorLoad):
            ret = self.visit_buffer_load_(expr)
        elif isinstance(expr, _tirx.Add):
            ret = self.visit_add_(expr)
        elif isinstance(expr, _tirx.Sub):
            ret = self.visit_sub_(expr)
        elif isinstance(expr, _tirx.Mul):
            ret = self.visit_mul_(expr)
        elif isinstance(expr, _tirx.Div):
            ret = self.visit_div_(expr)
        elif isinstance(expr, _tirx.Mod):
            ret = self.visit_mod_(expr)
        elif isinstance(expr, _tirx.FloorDiv):
            ret = self.visit_floor_div_(expr)
        elif isinstance(expr, _tirx.FloorMod):
            ret = self.visit_floor_mod_(expr)
        elif isinstance(expr, _tirx.LShift):
            ret = self.visit_lshift_(expr)
        elif isinstance(expr, _tirx.RShift):
            ret = self.visit_rshift_(expr)
        elif isinstance(expr, _tirx.BitwiseAnd):
            ret = self.visit_bitwise_and_(expr)
        elif isinstance(expr, _tirx.BitwiseOr):
            ret = self.visit_bitwise_or_(expr)
        elif isinstance(expr, _tirx.BitwiseXor):
            ret = self.visit_bitwise_xor_(expr)
        elif isinstance(expr, _tirx.BitwiseNot):
            ret = self.visit_bitwise_not_(expr)
        elif isinstance(expr, _tirx.Min):
            ret = self.visit_min_(expr)
        elif isinstance(expr, _tirx.Max):
            ret = self.visit_max_(expr)
        elif isinstance(expr, _tirx.EQ):
            ret = self.visit_eq_(expr)
        elif isinstance(expr, _tirx.NE):
            ret = self.visit_ne_(expr)
        elif isinstance(expr, _tirx.LT):
            ret = self.visit_lt_(expr)
        elif isinstance(expr, _tirx.LE):
            ret = self.visit_le_(expr)
        elif isinstance(expr, _tirx.GT):
            ret = self.visit_gt_(expr)
        elif isinstance(expr, _tirx.GE):
            ret = self.visit_ge_(expr)
        elif isinstance(expr, _tirx.And):
            ret = self.visit_and_(expr)
        elif isinstance(expr, _tirx.Or):
            ret = self.visit_or_(expr)
        elif isinstance(expr, _tirx.Cast):
            ret = self.visit_cast_(expr)
        elif isinstance(expr, _tirx.Not):
            ret = self.visit_not_(expr)
        elif isinstance(expr, _tirx.Select):
            ret = self.visit_select_(expr)
        elif isinstance(expr, tvm_tirx.ir.prim.vector_expr.Ramp):
            ret = self.visit_ramp_(expr)
        elif isinstance(expr, tvm_tirx.ir.prim.vector_expr.Broadcast):
            ret = self.visit_broadcast_(expr)
        elif isinstance(expr, tvm_tirx.ir.prim.vector_expr.Shuffle):
            ret = self.visit_shuffle_(expr)
        elif isinstance(expr, tvm_tirx.ir.IntImm):
            ret = self.visit_int_imm_(expr)
        elif isinstance(expr, tvm_tirx.ir.FloatImm):
            ret = self.visit_float_imm_(expr)
        elif isinstance(expr, SeqExpr):
            ret = self.visit_seq_expr_(expr)
        elif isinstance(expr, IfExpr):  # type: ignore
            ret = self.visit_if_expr_(expr)
        elif isinstance(expr, Op):
            ret = self.visit_op_(expr)
        elif isinstance(expr, TupleGetItem):
            ret = self.visit_tuple_getitem_(expr)
        elif isinstance(expr, DataTypeImm):
            ret = self.visit_data_type_imm_(expr)
        elif isinstance(expr, StringImm):
            ret = self.visit_string_imm_(expr)
        elif isinstance(expr, _tirx.Let | _te.Reduce):
            raise TypeError(f"Relax does not support {type(expr).__name__} expressions")
        elif isinstance(expr, Expr):
            if is_prim_expr(expr):
                raise TypeError(f"Invalid primitive expression type: {type(expr)}")
            ret = self.visit_expr_fallback_(expr)
        else:
            raise TypeError(f"Invalid type: {type(expr)}")

        return ret

    def visit_generic_const_(self, op: GenericConst):
        raise NotImplementedError()

    def visit_tuple_(self, op: Tuple):
        raise NotImplementedError()

    def visit_dataflow_var_(self, op: DataflowVar):
        raise NotImplementedError()

    def visit_var_(self, op: Var):
        raise NotImplementedError()

    def visit_shape_expr_(self, op: ShapeExpr):
        raise NotImplementedError()

    def visit_extern_func_(self, op: ExternFunc):
        raise NotImplementedError()

    def visit_global_var_(self, op: GlobalVar):
        raise NotImplementedError()

    def visit_function_(self, op: Function):
        raise NotImplementedError()

    def visit_call_(self, op: Call):
        raise NotImplementedError()

    def visit_buffer_load_(self, op: TensorLoad):
        return self.visit_expr_fallback_(op)

    def visit_add_(self, op: _tirx.Add):
        return self.visit_expr_fallback_(op)

    def visit_sub_(self, op: _tirx.Sub):
        return self.visit_expr_fallback_(op)

    def visit_mul_(self, op: _tirx.Mul):
        return self.visit_expr_fallback_(op)

    def visit_div_(self, op: _tirx.Div):
        return self.visit_expr_fallback_(op)

    def visit_mod_(self, op: _tirx.Mod):
        return self.visit_expr_fallback_(op)

    def visit_floor_div_(self, op: _tirx.FloorDiv):
        return self.visit_expr_fallback_(op)

    def visit_floor_mod_(self, op: _tirx.FloorMod):
        return self.visit_expr_fallback_(op)

    def visit_lshift_(self, op: _tirx.LShift):
        return self.visit_expr_fallback_(op)

    def visit_rshift_(self, op: _tirx.RShift):
        return self.visit_expr_fallback_(op)

    def visit_bitwise_and_(self, op: _tirx.BitwiseAnd):
        return self.visit_expr_fallback_(op)

    def visit_bitwise_or_(self, op: _tirx.BitwiseOr):
        return self.visit_expr_fallback_(op)

    def visit_bitwise_xor_(self, op: _tirx.BitwiseXor):
        return self.visit_expr_fallback_(op)

    def visit_bitwise_not_(self, op: _tirx.BitwiseNot):
        return self.visit_expr_fallback_(op)

    def visit_min_(self, op: _tirx.Min):
        return self.visit_expr_fallback_(op)

    def visit_max_(self, op: _tirx.Max):
        return self.visit_expr_fallback_(op)

    def visit_eq_(self, op: _tirx.EQ):
        return self.visit_expr_fallback_(op)

    def visit_ne_(self, op: _tirx.NE):
        return self.visit_expr_fallback_(op)

    def visit_lt_(self, op: _tirx.LT):
        return self.visit_expr_fallback_(op)

    def visit_le_(self, op: _tirx.LE):
        return self.visit_expr_fallback_(op)

    def visit_gt_(self, op: _tirx.GT):
        return self.visit_expr_fallback_(op)

    def visit_ge_(self, op: _tirx.GE):
        return self.visit_expr_fallback_(op)

    def visit_and_(self, op: _tirx.And):
        return self.visit_expr_fallback_(op)

    def visit_or_(self, op: _tirx.Or):
        return self.visit_expr_fallback_(op)

    def visit_cast_(self, op: _tirx.Cast):
        return self.visit_expr_fallback_(op)

    def visit_not_(self, op: _tirx.Not):
        return self.visit_expr_fallback_(op)

    def visit_select_(self, op: _tirx.Select):
        return self.visit_expr_fallback_(op)

    def visit_ramp_(self, op: tvm_tirx.ir.prim.vector_expr.Ramp):
        return self.visit_expr_fallback_(op)

    def visit_broadcast_(self, op: tvm_tirx.ir.prim.vector_expr.Broadcast):
        return self.visit_expr_fallback_(op)

    def visit_shuffle_(self, op: tvm_tirx.ir.prim.vector_expr.Shuffle):
        return self.visit_expr_fallback_(op)

    def visit_int_imm_(self, op: tvm_tirx.ir.IntImm):
        return self.visit_expr_fallback_(op)

    def visit_float_imm_(self, op: tvm_tirx.ir.FloatImm):
        return self.visit_expr_fallback_(op)

    def visit_seq_expr_(self, op: SeqExpr):
        raise NotImplementedError()

    def visit_if_expr_(self, op: IfExpr):
        raise NotImplementedError()

    def visit_op_(self, op: Op):
        raise NotImplementedError()

    def visit_tuple_getitem_(self, op: TupleGetItem):
        raise NotImplementedError()

    def visit_expr_fallback_(self, op: Expr):
        raise NotImplementedError()

    def visit_string_imm_(self, op: StringImm):
        raise NotImplementedError()

    def visit_data_type_imm_(self, op: DataTypeImm):
        raise NotImplementedError()

    def visit_var_binding_(self, binding: VarBinding):
        raise NotImplementedError()

    def visit_match_cast_(self, binding: MatchCast):
        raise NotImplementedError()

    def visit_binding_block_(self, block: BindingBlock):
        raise NotImplementedError()

    def visit_dataflow_block_(self, block: DataflowBlock):
        raise NotImplementedError()

    def visit_var_def_(self, var: Var):
        raise NotImplementedError()

    def visit_dataflow_var_def_(self, var: DataflowVar):
        raise NotImplementedError()

    def visit_binding(self, binding: Binding):
        if isinstance(binding, MatchCast):
            self.visit_match_cast_(binding)
        elif isinstance(binding, VarBinding):
            self.visit_var_binding_(binding)
        else:
            raise TypeError(f"Invalid type: {type(binding)}")

    def visit_binding_block(self, block: BindingBlock):
        if isinstance(block, DataflowBlock):
            self.visit_dataflow_block_(block)
        elif isinstance(block, BindingBlock):
            self.visit_binding_block_(block)
        else:
            raise TypeError(f"Invalid type: {type(block)}")

    def visit_var_def(self, var: Var):
        if isinstance(var, DataflowVar):
            self.visit_dataflow_var_def_(var)
        elif isinstance(var, Var):
            self.visit_var_def_(var)
        else:
            raise TypeError(f"Invalid type: {type(var)}")


@tvm_ffi.register_object("expr_functor.PyExprVisitor")
class _PyExprVisitor(tvm_ffi.core.Object):
    """
    A TVM object to support customization of ExprVisitor on the python side.
    This is the decorated result returned from visitor decorator.

    WARNING: This is NOT the user facing class for method overwriting inheritance.

    See also: visitor, PyExprVisitor
    """

    def __init__(
        self,
        f_visit_expr: Callable | None = None,
        f_visit_generic_const_: Callable | None = None,
        f_visit_tuple_: Callable | None = None,
        f_visit_var_: Callable | None = None,
        f_visit_dataflow_var_: Callable | None = None,
        f_visit_shape_expr_: Callable | None = None,
        f_visit_extern_func_: Callable | None = None,
        f_visit_global_var_: Callable | None = None,
        f_visit_function_: Callable | None = None,
        f_visit_call_: Callable | None = None,
        f_visit_seq_expr_: Callable | None = None,
        f_visit_if_expr_: Callable | None = None,
        f_visit_op_: Callable | None = None,
        f_visit_tuple_getitem_: Callable | None = None,
        f_visit_expr_fallback_: Callable | None = None,
        f_visit_string_imm_: Callable | None = None,
        f_visit_data_type_imm_: Callable | None = None,
        f_visit_binding: Callable | None = None,
        f_visit_var_binding_: Callable | None = None,
        f_visit_match_cast_: Callable | None = None,
        f_visit_binding_block: Callable | None = None,
        f_visit_binding_block_: Callable | None = None,
        f_visit_dataflow_block_: Callable | None = None,
        f_visit_var_def: Callable | None = None,
        f_visit_var_def_: Callable | None = None,
        f_visit_dataflow_var_def_: Callable | None = None,
        f_visit_loc: Callable | None = None,
    ) -> None:
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.MakePyExprVisitor,  # type: ignore
            f_visit_expr,
            f_visit_generic_const_,
            f_visit_tuple_,
            f_visit_var_,
            f_visit_dataflow_var_,
            f_visit_shape_expr_,
            f_visit_extern_func_,
            f_visit_global_var_,
            f_visit_function_,
            f_visit_call_,
            f_visit_seq_expr_,
            f_visit_if_expr_,
            f_visit_op_,
            f_visit_tuple_getitem_,
            f_visit_expr_fallback_,
            f_visit_string_imm_,
            f_visit_data_type_imm_,
            f_visit_binding,
            f_visit_var_binding_,
            f_visit_match_cast_,
            f_visit_binding_block,
            f_visit_binding_block_,
            f_visit_dataflow_block_,
            f_visit_var_def,
            f_visit_var_def_,
            f_visit_dataflow_var_def_,
            f_visit_loc,
        )

    def visit_expr(self, expr: Expr) -> None:
        """Generic dispatcher for Expr.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.
        """
        return _ffi_api.PyExprVisitorVisitExpr(self, expr)  # type: ignore

    def visit_binding(self, binding: Binding) -> None:
        """Generic dispatcher for Binding.

        Parameters
        ----------
        binding : Binding
            The binding to be visited.
        """
        return _ffi_api.PyExprVisitorVisitBinding(self, binding)  # type: ignore

    def visit_binding_block(self, block: BindingBlock) -> None:
        """Generic dispatcher for BindingBlock.

        Parameters
        ----------
        block : BindingBlock
            The block to be visited.
        """
        return _ffi_api.PyExprVisitorVisitBindingBlock(self, block)  # type: ignore

    def visit_var_def(self, var: Var) -> None:
        """Generic dispatcher for visiting the var definition site.
        Note that visit_var_() will only visit the usage site of an Var.

        Parameters
        ----------
        var : Var
            The var to be visited.
        """
        return _ffi_api.PyExprVisitorVisitVarDef(self, var)  # type: ignore


class PyExprVisitor:
    """
    An abstract ExprVisitor with customized methods on the python-side.
    This is the user facing class for method overwriting inheritance.
    _tvm_metadata discribes the class to inherit("cls"), the methods
    that users can overwrite("methods").

    Note: @relax.expr_functor.visitor is required for proper usage of any inherited class.

    See also: visitor, _PyExprVisitor

    Example:

    .. code-block:: python

        @relax.expr_functor.visitor
        def MyExprVisitor(PyExprVisitor):
            ...

    """

    _tvm_metadata = {
        "cls": _PyExprVisitor,
        "methods": [
            "visit_expr",
            "visit_generic_const_",
            "visit_tuple_",
            "visit_var_",
            "visit_dataflow_var_",
            "visit_shape_expr_",
            "visit_extern_func_",
            "visit_global_var_",
            "visit_function_",
            "visit_call_",
            "visit_seq_expr_",
            "visit_if_expr_",
            "visit_op_",
            "visit_tuple_getitem_",
            "visit_expr_fallback_",
            "visit_string_imm_",
            "visit_data_type_imm_",
            "visit_binding",
            "visit_var_binding_",
            "visit_match_cast_",
            "visit_binding_block",
            "visit_binding_block_",
            "visit_dataflow_block_",
            "visit_var_def",
            "visit_var_def_",
            "visit_dataflow_var_def_",
            "visit_loc",
        ],
    }

    def visit_expr(self, expr: Expr) -> None:
        """Generic dispatcher for Expr.
        Users can customized this function to overwrite VisitExpr(const Expr& expr) on the C++ side.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.PyExprVisitorVisitExpr(self._outer(), expr)  # type: ignore

    def visit_binding(self, binding: Binding) -> None:
        """Generic dispatcher for Binding.
        Users can customized this function to overwrite VisitBinding(const Binding& binding)
        on the C++ side.

        Parameters
        ----------
        binding : Binding
            The binding to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.PyExprVisitorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_binding_block(self, block: BindingBlock) -> None:
        """Generic dispatcher for BindingBlock.
        Users can customized this function to overwrite VisitBindingBlock(const BindingBlock& block)
        on the C++ side.

        Parameters
        ----------
        block : BindingBlock
            The block to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.PyExprVisitorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_var_def(self, var: Var) -> None:
        """Generic dispatcher for visiting the var definition site.
        Users can customized this function to overwrite VisitVarDef(const Var& var) on the C++ side.
        Note that visit_var_() will only visit the usage site of an Var.

        Parameters
        ----------
        var : Var
            The var to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.PyExprVisitorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_generic_const_(self, op: GenericConst) -> None:
        """Visit GenericConst.
        Users can customized this function to overwrite VisitExpr_(const GenericConstObj* op)
        on the C++ side.

        Parameters
        ----------
        op : GenericConst
            The GenericConst to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_tuple_(self, op: Tuple) -> None:
        """Visit Tuple.
        Users can customized this function to overwrite VisitExpr_(const TupleObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Tuple
            The Tuple to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_var_(self, op: Var) -> None:
        """Visit Var.
        Users can customized this function to overwrite VisitExpr_(const VarObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Var
            The Var to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_dataflow_var_(self, op: DataflowVar) -> None:
        """Visit DataflowVar.
        Users can customized this function to overwrite VisitExpr_(const DataflowVarObj* op)
        on the C++ side.

        Parameters
        ----------
        op : DataflowVar
            The DataflowVar to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_shape_expr_(self, op: ShapeExpr) -> None:
        """Visit ShapeExpr.
        Users can customized this function to overwrite VisitExpr_(const ShapeExprObj* op)
        on the C++ side.

        Parameters
        ----------
        op : ShapeExpr
            The ShapeExpr to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_extern_func_(self, op: ExternFunc) -> None:
        """Visit ExternFunc.
        Users can customized this function to overwrite VisitExpr_(const ExternFuncObj* op)
        on the C++ side.

        Parameters
        ----------
        op : ExternFunc
            The ExternFunc to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_global_var_(self, op: GlobalVar) -> None:
        """Visit GlobalVar.
        Users can customized this function to overwrite VisitExpr_(const GlobalVarObj* op)
        on the C++ side.

        Parameters
        ----------
        op : GlobalVar
            The GlobalVar to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_function_(self, op: Function) -> None:
        """Visit Function.
        Users can customized this function to overwrite VisitExpr_(const FunctionObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Function
            The Function to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_call_(self, op: Call) -> None:
        """Visit Call.
        Users can customized this function to overwrite VisitExpr_(const CallObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Call
            The Call to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_seq_expr_(self, op: SeqExpr) -> None:
        """Visit SeqExpr.
        Users can customized this function to overwrite VisitExpr_(const SeqExprObj* op)
        on the C++ side.

        Parameters
        ----------
        op : SeqExpr
            The SeqExpr to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_if_expr_(self, op: IfExpr) -> None:
        """Visit IfExpr.
        Users can customized this function to overwrite VisitExpr_(const IfExprObj* op)
        on the C++ side.

        Parameters
        ----------
        op : IfExpr
            The IfExpr to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_op_(self, op: Op) -> None:
        """Visit Op.
        Users can customized this function to overwrite VisitExpr_(const OpObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Op
            The Op to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_tuple_getitem_(self, op: TupleGetItem) -> None:
        """Visit TupleGetItem.
        Users can customized this function to overwrite VisitExpr_(const TupleGetItemObj* op)
        on the C++ side.

        Parameters
        ----------
        op : TupleGetItem
            The TupleGetItem to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_expr_fallback_(self, op: Expr) -> None:
        """Visit an expression handled by the C++ fallback.

        Parameters
        ----------
        op : Expr
            The expression to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExprFallback(self._outer(), op)  # type: ignore

    def visit_string_imm_(self, op: StringImm) -> None:
        """Visit StringImm.
        Users can customized this function to overwrite VisitExpr_(const StringImmObj* op)
        on the C++ side.

        Parameters
        ----------
        op : StringImm
            The StringImm to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_data_type_imm_(self, op: DataTypeImm) -> None:
        """Visit DataTypeImm.
        Users can customized this function to overwrite VisitExpr_(const DataTypeImmObj* op)
        on the C++ side.

        Parameters
        ----------
        op : DataTypeImm
            The DataTypeImm to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitExpr(self._outer(), op)  # type: ignore

    def visit_var_binding_(self, binding: VarBinding) -> None:
        """Visit VarBinding.
        Users can customized this function to overwrite VisitBinding_(const VarBindingObj* binding)
        on the C++ side.

        Parameters
        ----------
        binding : VarBinding
            The VarBinding to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_match_cast_(self, binding: MatchCast) -> None:
        """Visit MatchCast.
        Users can customized this function to overwrite VisitBinding_(const MatchCastObj* binding)
        on the C++ side.

        Parameters
        ----------
        binding : MatchCast
            The MatchCast to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_binding_block_(self, block: BindingBlock) -> None:
        """Visit BindingBlock.
        Users can customized this function to overwrite VisitBindingBlock_(const BindingBlockObj*
        block) on the C++ side.

        Parameters
        ----------
        block : BindingBlock
            The BindingBlock to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_dataflow_block_(self, block: DataflowBlock) -> None:
        """Visit DataflowBlock.
        Users can customized this function to overwrite VisitBindingBlock_(const DataflowBlockObj*
        block) on the C++ side.

        Parameters
        ----------
        block : DataflowBlock
            The DataflowBlock to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_var_def_(self, var: Var) -> None:
        """Visit the Var definition site.
        Users can customized this function to overwrite VisitVarDef_(const VarObj* var)
        on the C++ side.

        Parameters
        ----------
        var : Var
            The Var to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_dataflow_var_def_(self, var: DataflowVar) -> None:
        """Visit the DataflowVar definition site.
        Users can customized this function to overwrite VisitVarDef_(const DataflowVarObj* var)
        on the C++ side.

        Parameters
        ----------
        var : DataflowVar
            The DataflowVar to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_loc(self, loc: Location) -> None:
        """Visit Location.
        Override this function to customize VisitLoc(const Location& loc) on the C++ side.

        Parameters
        ----------
        loc : Location
            The Location to be visited.
        """
        # Using self._outer() to ref _PyExprVisitor
        return _ffi_api.ExprVisitorVisitLoc(self._outer(), loc)  # type: ignore


@tvm_ffi.register_object("expr_functor.PyExprMutator")
class _PyExprMutator(tvm_ffi.Object):
    """
    A TVM object to support customization of ExprMutator on the python side.
    This is the decorated result returned from mutator decorator.

    WARNING: This is NOT the user facing class for method overwriting inheritance.

    See also: mutator, PyExprmutator
    """

    def __init__(
        self,
        builder: BlockBuilder = None,
        f_visit_expr: Callable | None = None,
        f_visit_generic_const_: Callable | None = None,
        f_visit_tuple_: Callable | None = None,
        f_visit_var_: Callable | None = None,
        f_visit_dataflow_var_: Callable | None = None,
        f_visit_shape_expr_: Callable | None = None,
        f_visit_extern_func_: Callable | None = None,
        f_visit_global_var_: Callable | None = None,
        f_visit_function_: Callable | None = None,
        f_visit_call_: Callable | None = None,
        f_visit_seq_expr_: Callable | None = None,
        f_visit_if_expr_: Callable | None = None,
        f_visit_op_: Callable | None = None,
        f_visit_tuple_getitem_: Callable | None = None,
        f_visit_expr_fallback_: Callable | None = None,
        f_visit_string_imm_: Callable | None = None,
        f_visit_data_type_imm_: Callable | None = None,
        f_visit_binding: Callable | None = None,
        f_visit_var_binding_: Callable | None = None,
        f_visit_match_cast_: Callable | None = None,
        f_visit_binding_block: Callable | None = None,
        f_visit_binding_block_: Callable | None = None,
        f_visit_dataflow_block_: Callable | None = None,
        f_visit_var_def: Callable | None = None,
        f_visit_var_def_: Callable | None = None,
        f_visit_dataflow_var_def_: Callable | None = None,
        f_visit_loc: Callable | None = None,
    ) -> None:
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.MakePyExprMutator,  # type: ignore
            builder,
            f_visit_expr,
            f_visit_generic_const_,
            f_visit_tuple_,
            f_visit_var_,
            f_visit_dataflow_var_,
            f_visit_shape_expr_,
            f_visit_extern_func_,
            f_visit_global_var_,
            f_visit_function_,
            f_visit_call_,
            f_visit_seq_expr_,
            f_visit_if_expr_,
            f_visit_op_,
            f_visit_tuple_getitem_,
            f_visit_expr_fallback_,
            f_visit_string_imm_,
            f_visit_data_type_imm_,
            f_visit_binding,
            f_visit_var_binding_,
            f_visit_match_cast_,
            f_visit_binding_block,
            f_visit_binding_block_,
            f_visit_dataflow_block_,
            f_visit_var_def,
            f_visit_var_def_,
            f_visit_dataflow_var_def_,
            f_visit_loc,
        )

    def visit_expr(self, expr: Expr) -> Expr:
        """Generic dispatcher for Expr.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation.
        """
        return _ffi_api.PyExprMutatorVisitExpr(self, expr)  # type: ignore

    def visit_binding(self, binding: Binding) -> None:
        """Generic dispatcher for Binding.

        Parameters
        ----------
        binding : Binding
            The binding to be visited.
        """
        return _ffi_api.PyExprMutatorVisitBinding(self, binding)  # type: ignore

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        """Generic dispatcher for BindingBlock.

        Parameters
        ----------
        block : BindingBlock
            The block to be visited.

        Returns
        -------
        result : BindingBlock
            The binding block after transformation.
        """
        return _ffi_api.PyExprMutatorVisitBindingBlock(self, block)  # type: ignore

    def visit_var_def(self, var: Var) -> Var:
        """Generic dispatcher for visiting the var definition site.
        Note that visit_var_() will only visit the usage site of an Var.

        Parameters
        ----------
        var : Var
            The var to be visited.

        Returns
        -------
        result : Var
            The var after post-order rewritten.
        """
        return _ffi_api.PyExprMutatorVisitVarDef(self, var)  # type: ignore


class PyExprMutator:
    """
    An abstract ExprMutator with customized methods on the python-side.
    This is the user facing class for method overwriting inheritance.
    _tvm_metadata discribes the class to inherit("cls"), the methods that users can
    overwrite("methods"), the constructor's parameters("fields")

    Note: @relax.expr_functor.mutator is required for proper usage of any inherited class.

    See also: visitor, _PyExprVisitor

    Example:

    .. code-block:: python

        @relax.expr_functor.mutator
        def MyExprMutator(PyExprMutator):
            ...

    """

    _tvm_metadata = {
        "cls": _PyExprMutator,
        "fields": ["builder_"],
        "methods": [
            "visit_expr",
            "visit_generic_const_",
            "visit_tuple_",
            "visit_var_",
            "visit_dataflow_var_",
            "visit_shape_expr_",
            "visit_extern_func_",
            "visit_global_var_",
            "visit_function_",
            "visit_call_",
            "visit_seq_expr_",
            "visit_if_expr_",
            "visit_op_",
            "visit_tuple_getitem_",
            "visit_expr_fallback_",
            "visit_string_imm_",
            "visit_data_type_imm_",
            "visit_binding",
            "visit_var_binding_",
            "visit_match_cast_",
            "visit_binding_block",
            "visit_binding_block_",
            "visit_dataflow_block_",
            "visit_var_def",
            "visit_var_def_",
            "visit_dataflow_var_def_",
            "visit_loc",
        ],
    }

    def __init__(self, mod: IRModule | None = None) -> None:
        """Constructor"""
        self.builder_ = BlockBuilder(mod)

    def visit_expr(self, expr: Expr) -> Expr:
        """Generic dispatcher for Expr.
        Users can customized this function to overwrite VisitExpr(const Expr& expr) on the C++ side.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitExpr(self._outer(), expr)  # type: ignore

    def visit_binding(self, binding: Binding) -> None:
        """Generic dispatcher for Binding.
        Users can customized this function to overwrite VisitBinding(const Binding& binding)
        on the C++ side.

        Parameters
        ----------
        binding : Binding
            The binding to be visited.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_binding_block(self, block: BindingBlock) -> BindingBlock:
        """Generic dispatcher for BindingBlock.
        Users can customized this function to overwrite VisitBindingBlock(const BindingBlock& block)
        on the C++ side.

        Parameters
        ----------
        block : BindingBlock
            The block to be visited.

        Returns
        -------
        result : BindingBlock
            The binding block after transformation.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_var_def(self, var: Var) -> Var:
        """Generic dispatcher for visiting the var definition site.
        Users can customized this function to overwrite VisitVarDef(const Var& var) on the C++ side.
        Note that visit_var_() will only visit the usage site of an Var.

        Parameters
        ----------
        var : Var
            The var to be visited.

        Returns
        -------
        result: Var
            The var after post-order rewritten.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_generic_const_(self, op: GenericConst) -> Expr:
        """Visit GenericConst.
        Users can customized this function to overwrite VisitExpr_(const GenericConstObj* op)
        on the C++ side.

        Parameters
        ----------
        op : GenericConst
            The GenericConst to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_tuple_(self, op: Tuple) -> Expr:
        """Visit Tuple.
        Users can customized this function to overwrite VisitExpr_(const TupleObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Tuple
            The Tuple to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_var_(self, op: Var) -> Expr:
        """Visit Var.
        Users can customized this function to overwrite VisitExpr_(const VarObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Var
            The Var to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_dataflow_var_(self, op: DataflowVar) -> Expr:
        """Visit DataflowVar.
        Users can customized this function to overwrite VisitExpr_(const DataflowVarObj* op)
        on the C++ side.

        Parameters
        ----------
        op : DataflowVar
            The DataflowVar to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_shape_expr_(self, op: ShapeExpr) -> Expr:
        """Visit ShapeExpr.
        Users can customized this function to overwrite VisitExpr_(const ShapeExprObj* op)
        on the C++ side.

        Parameters
        ----------
        op : ShapeExpr
            The ShapeExpr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_extern_func_(self, op: ExternFunc) -> Expr:
        """Visit ExternFunc.
        Users can customized this function to overwrite VisitExpr_(const ExternFuncObj* op)
        on the C++ side.

        Parameters
        ----------
        op : ExternFunc
            The ExternFunc to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_global_var_(self, op: GlobalVar) -> Expr:
        """Visit GlobalVar.
        Users can customized this function to overwrite VisitExpr_(const GlobalVarObj* op)
        on the C++ side.

        Parameters
        ----------
        op : GlobalVar
            The GlobalVar to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_function_(self, op: Function) -> Expr:
        """Visit Function.
        Users can customized this function to overwrite VisitExpr_(const FunctionObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Function
            The Function to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_call_(self, op: Call) -> Expr:
        """Visit Call.
        Users can customized this function to overwrite VisitExpr_(const CallObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Call
            The Call to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_seq_expr_(self, op: SeqExpr) -> Expr:
        """Visit SeqExpr.
        Users can customized this function to overwrite VisitExpr_(const SeqExprObj* op)
        on the C++ side.

        Parameters
        ----------
        op : SeqExpr
            The SeqExpr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_if_expr_(self, op: IfExpr) -> Expr:
        """Visit IfExpr.
        Users can customized this function to overwrite VisitExpr_(const IfExprObj* op)
        on the C++ side.

        Parameters
        ----------
        op : IfExpr
            The IfExpr to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_op_(self, op: Op) -> Expr:
        """Visit Op.
        Users can customized this function to overwrite VisitExpr_(const OpObj* op)
        on the C++ side.

        Parameters
        ----------
        op : Op
            The Op to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_tuple_getitem_(self, op: TupleGetItem) -> Expr:
        """Visit TupleGetItem.
        Users can customized this function to overwrite VisitExpr_(const TupleGetItemObj* op)
        on the C++ side.

        Parameters
        ----------
        op : TupleGetItem
            The TupleGetItem to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_expr_fallback_(self, op: Expr) -> Expr:
        """Visit an expression handled by the C++ fallback.

        Parameters
        ----------
        op : Expr
            The expression to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExprFallback(self._outer(), op)  # type: ignore

    def visit_string_imm_(self, op: StringImm) -> Expr:
        """Visit StringImm.
        Users can customized this function to overwrite VisitExpr_(const StringImmObj* op)
        on the C++ side.

        Parameters
        ----------
        op : StringImm
            The StringImm to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_data_type_imm_(self, op: DataTypeImm) -> Expr:
        """Visit DataTypeImm.
        Users can customized this function to overwrite VisitExpr_(const DataTypeImmObj* op)
        on the C++ side.

        Parameters
        ----------
        op : DataTypeImm
            The DataTypeImm to be visited.

        Returns
        -------
        result : Expr
            The Expr after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitExpr(self._outer(), op)  # type: ignore

    def visit_var_binding_(self, binding: VarBinding) -> None:
        """Visit VarBinding.
        Users can customized this function to overwrite VisitBinding_(const VarBindingObj* binding)
        on the C++ side.

        Parameters
        ----------
        binding : VarBinding
            The VarBinding to be visited.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_match_cast_(self, binding: MatchCast) -> None:
        """Visit MatchCast.
        Users can customized this function to overwrite VisitBinding_(const MatchCastObj* binding)
        on the C++ side.

        Parameters
        ----------
        binding : MatchCast
            The MatchCast to be visited.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitBinding(self._outer(), binding)  # type: ignore

    def visit_binding_block_(self, block: BindingBlock) -> BindingBlock:
        """Visit BindingBlock.
        Users can customized this function to overwrite VisitBindingBlock_(const BindingBlockObj*
        block) on the C++ side.

        Parameters
        ----------
        block : BindingBlock
            The BindingBlock to be visited.

        Returns
        -------
        result : BindingBlock
            The binding block after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_dataflow_block_(self, block: DataflowBlock) -> BindingBlock:
        """Visit DataflowBlock.
        Users can customized this function to overwrite VisitBindingBlock_(const DataflowBlockObj*
        block) on the C++ side.

        Parameters
        ----------
        block : DataflowBlock
            The DataflowBlock to be visited.

        Returns
        -------
        result : BindingBlock
            The binding block after transformation
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitBindingBlock(self._outer(), block)  # type: ignore

    def visit_var_def_(self, var: Var) -> Var:
        """Visit the Var definition site.
        Users can customized this function to overwrite VisitVarDef_(const VarObj* var)
        on the C++ side.

        Parameters
        ----------
        var : Var
            The Var to be visited.

        Returns
        -------
        result : Var
            The var after post-order rewritten.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_dataflow_var_def_(self, var: DataflowVar) -> Var:
        """Visit the DataflowVar definition site.
        Users can customized this function to overwrite VisitVarDef_(const DataflowVarObj* var)
        on the C++ side.

        Parameters
        ----------
        var : DataflowVar
            The DataflowVar to be visited.

        Returns
        -------
        result : Var
            The var after post-order rewritten.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.ExprMutatorVisitVarDef(self._outer(), var)  # type: ignore

    def visit_loc(self, loc: Location) -> Location:
        """Visit Location.
        Override this function to customize VisitLoc(const Location& loc) on the C++ side.

        Parameters
        ----------
        loc : Location
            The Location to be visited.

        Returns
        -------
        result : Location
            The loc after transformation.
        """
        raise NotImplementedError

    def visit_expr_post_order(self, expr: Expr) -> Expr:
        """Post-order rewrite an Expr and normalize.

        Parameters
        ----------
        expr : Expr
            The Expr to be rewritten.

        Returns
        -------
        result : Expr
            The Expr after post-order rewritten.
        """
        return _ffi_api.PyExprMutatorVisitExprPostOrder(self._outer(), expr)  # type: ignore

    def set_var_remap(self, old_var: Var, new_var: Var) -> None:
        """Remap a var to a new var in use-site.

        Parameters
        ----------
        old_var : Var
            The old var.
        new_var : Var
            The new var.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorSetVarRemap(  # type: ignore
            self._outer(), old_var, new_var
        )

    def get_var_remap(self, var: Var) -> Var | None:
        """Remap a var to a new var in use-site.

        Parameters
        ----------
        var : Var
            The old var.

        Returns
        -------
        var : Var
            The remapped var, or None if no mapping exists.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorGetVarRemap(self._outer(), var)  # type: ignore

    def visit_with_new_scope(self, expr: Expr) -> Expr:
        """Rewrite the expr with a new scope, used in a Function's body and the branches of IfExpr.

        Parameters
        ----------
        expr : Expr
            The expr to be visited.

        Returns
        -------
        var : Var
            The expr after visiting.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorVisitWithNewScope(self._outer(), expr)  # type: ignore

    def lookup_binding(self, var: Var) -> Expr | None:
        """Look up the value bound to a variable.
        Note: For function parameters, this function returns std::nullopt.

        Parameters
        ----------
        var : Var
            The var to be looked up.

        Returns
        -------
        var : Var
            The value bound to the input var.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorLookupBinding(self._outer(), var)  # type: ignore

    def with_type(self, var: Var, ty: Type) -> Var:
        """Create a new var with specified shape and type if the original var's shape or type does
        not match with the specified ones.

        Parameters
        ----------
        var : Var
            The var to be updated.
        ty : Type
            The type.

        Returns
        -------
        var : Var
            The var filled with shape and type.
        """
        # Using self._outer() to ref _PyExprMutator
        return _ffi_api.PyExprMutatorWithType(self._outer(), var, ty)  # type: ignore
