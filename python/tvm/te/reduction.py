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
# pylint: disable=redefined-builtin
"""Tensor expression reductions and their construction helpers."""

from typing import TYPE_CHECKING

import tvm_ffi
from tvm_ffi import Array

import tvm_tirx
from tvm_tirx import ir
from tvm_tirx import tirx as tir
from tvm_tirx.ir import Expr, Scriptable, Var, const
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.ir.prim import _ffi_api as _prim_ffi_api
from tvm_tirx.ir.prim import max_value, min_value
from tvm_tirx.tirx.op import _primexpr_dtype

from . import _ffi_api

if TYPE_CHECKING:
    from tvm.s_tir import IterVar


def _iter_var_type():
    from tvm.s_tir.iter_var import IterVar

    return IterVar


@tvm_ffi.register_object("te.CommReducer")
class CommReducer(tvm_ffi.Object, Scriptable):
    """Commutative reduce operator

    Parameters
    ----------
    lhs : List[Var]
       The left arguments of the reducer.

    rhs : List[Var]
       The right arguments of the reducer.

    result : List[Expr]
       The reduction results.

    identity_element : List[Expr]
       The identity elements.

    loc : Location, optional
        The location of this expression in the source code.
    """

    lhs: list[Var]
    rhs: list[Var]
    result: list[Expr]
    identity_element: list[Expr]

    def __init__(
        self,
        lhs: list[Var],
        rhs: list[Var],
        result: list[Expr],
        identity_element: list[Expr],
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.CommReducer,
            lhs,
            rhs,
            result,
            identity_element,
            loc,  # type: ignore
        )


@tvm_ffi.register_object("te.Reduce")
class Reduce(ir.ExprWithOp):
    """Reduce node.

    Parameters
    ----------
    combiner : CommReducer
        The combiner.

    src : list of Expr
        The source expression.

    rdom : list of IterVar
        The iteration domain

    condition : Expr
        The reduce condition.

    value_index : int
        The value index.

    init : list of Expr
        The initial value for output. This can be an int, float, or TE tensor-load Call.

    loc : Location, optional
        The location of this expression in the source code.
    """

    combiner: CommReducer
    source: list[Expr]
    init: list[Expr]
    axis: "list[IterVar]"
    condition: Expr
    value_index: int

    def __init__(
        self,
        combiner: CommReducer,
        src: list[Expr],
        rdom: "list[IterVar]",
        condition: Expr,
        value_index: int,
        init: list[Expr] | None = None,
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        init = [] if init is None else init
        self.__init_handle_by_constructor__(
            _ffi_api.Reduce,
            combiner,
            src,
            rdom,
            condition,
            value_index,
            init,
            loc,  # type: ignore
        )


def comm_reducer(fcombine, fidentity, name="reduce"):
    """Create a commutative reducer for reduction.

    Parameters
    ----------
    fcombine : function(Expr -> Expr -> Expr)
        A binary function which takes two Expr as input to return a Expr.

    fidentity : function(str -> Expr)
        A function which takes a type string as input to return a const Expr.

    Returns
    -------
    reducer : function
        A function which creates a reduce expression over axis.
        There are two ways to use it:

        1. accept (expr, axis, where) to produce an Reduce Expr on
           specified axis;
        2. simply use it with multiple Exprs.

    Example
    -------
    .. code-block:: python

        n = te.var("n")
        m = te.var("m")
        mysum = te.comm_reducer(lambda x, y: x+y,
            lambda t: tvm_tirx.tirx.const(0, dtype=t), name="mysum")
        A = te.placeholder((n, m), name="A")
        k = te.reduce_axis((0, m), name="k")
        B = te.compute((n,), lambda i: mysum(A[i, k], axis=k), name="B")
    """

    def _reduce_directly(*args):
        num = len(args)
        # process `where` is None
        if num == 3 and args[2] is None:
            num = 2
        res = args[0]
        for i in range(num - 1):
            res = fcombine(res, args[i + 1])
        return res

    def _make_reduce(expr, axis, where=None, init=None):
        code = fcombine.__code__
        assert fcombine.__code__.co_argcount == 2
        expr = tir.convert(expr)
        if init is not None:
            init = tir.convert(init)
        if isinstance(expr, Array):
            size = len(expr)
            lhs = []
            rhs = []
            dtypes = []
            for i in range(size):
                dtype = _primexpr_dtype(expr[i])
                dtypes.append(dtype)
                lname = code.co_varnames[0] + "_" + str(i)
                lhs.append(Var(lname, dtype))
                rname = code.co_varnames[1] + "_" + str(i)
                rhs.append(Var(rname, dtype))
            if init is None:
                init = []
            result = fcombine(lhs, rhs)
            id_elem = fidentity(*dtypes)
        else:
            assert tvm_tirx.ir.is_prim_expr(expr)
            size = 1
            dtype = _primexpr_dtype(expr)
            lvar = Var(code.co_varnames[0], dtype)
            rvar = Var(code.co_varnames[1], dtype)
            result = [fcombine(lvar, rvar)]
            id_elem = [fidentity(dtype)]
            lhs = [lvar]
            rhs = [rvar]
            expr = [expr]
            if init is not None:
                init = [init]
        combiner = CommReducer(lhs, rhs, result, id_elem)
        if not isinstance(axis, list | tuple | tvm_ffi.Array):
            axis = [axis]
        if where is None:
            where = tir.convert(True)
        if init is None:
            outputs = tuple(Reduce(combiner, expr, axis, where, i, []) for i in range(size))
        else:
            outputs = tuple(Reduce(combiner, expr, axis, where, i, init) for i in range(size))
        return outputs[0] if size == 1 else outputs

    # pylint: disable=keyword-arg-before-vararg
    def reducer(expr, axis, where=None, init=None, *args):
        if isinstance(axis, _iter_var_type() | list | tuple | Array):
            assert not args
            return _make_reduce(expr, axis, where, init)

        if where is None:
            assert not args
            assert init is None
            return _reduce_directly(expr, axis)
        elif init is None:
            assert not args
            return _reduce_directly(expr, axis, where)
        else:
            return _reduce_directly(expr, axis, where, init, *args)

    doc_str = """Create a {0} expression over axis.

              Parameters
              ----------
              expr : Expr
                  The source expression.
              axis : IterVar
                  The reduction IterVar axis
              where : optional, Expr
                  Filtering predicate of the reduction.
              Returns
              -------
              value : Expr
                  The result value.

              Example
              -------
              .. code-block:: python

                m = te.var("m")
                n = te.var("n")
                A = te.placeholder((m, n), name="A")
                k = te.reduce_axis((0, n), name="k")

                # there are two way to use this {0} reducer:
                # mode 1, accept (expr, axis, where) to produce an Reduce Expr
                B = te.compute((m,), lambda i: te.{0}(A[i, k], axis=k), name="B")

                # mode 2, simply use it with multiple Exprs:
                {0}_res = te.{0}(m, n)
              """
    reducer.__doc__ = doc_str.format(name)
    return reducer


sum = comm_reducer(lambda x, y: x + y, lambda t: const(0, dtype=t), name="sum")
min = comm_reducer(
    lambda x, y: _prim_ffi_api._OpMin(x, y, UNKNOWN_LOC),
    lambda dtype: max_value(ty=dtype),
    name="min",
)  # type: ignore
max = comm_reducer(
    lambda x, y: _prim_ffi_api._OpMax(x, y, UNKNOWN_LOC),
    lambda dtype: min_value(ty=dtype),
    name="max",
)  # type: ignore
