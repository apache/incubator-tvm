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
# ruff: noqa: F821

"""Pattern types in Relax Dataflow Pattern Language"""
# pylint: disable=no-member
# pylint: disable=pointless-statement

from typing import Union

import tvm_ffi
from tvm_ffi import Array

import tvm_tirx
from tvm_tirx.ir import make_node
from tvm_tirx.ir.base import Node
from tvm_tirx.ir.op import Op

from ..expr import Expr, Var
from . import _ffi as ffi


def register_df_node(type_key=None):
    """
    Register a Relax node type

    Parameters
    ----------
    type_key : str or cls
        The type key of the node
    """
    if not isinstance(type_key, str):
        return tvm_ffi.register_object("relax.dpl." + type_key.__name__)(type_key)
    return tvm_ffi.register_object(type_key)


class DFPattern(Node):
    """Base class of all Patterns."""

    def __call__(self, *args, varg_default_wildcard=False, add_constraint=True) -> "CallPattern":
        """
        Syntax sugar for creating a CallPattern with argument patterns

        Returns
        -------
        result: CallPattern
            The resulting CallPattern
        """
        return CallPattern(self, args, varg_default_wildcard, add_constraint)

    def __or__(self, other: "DFPattern") -> "OrPattern":
        """
        Syntax sugar for creating an OrPattern

        Parameters
        ----------
        other: DFPattern
            Alternative pattern

        Returns
        -------
        result: OrPattern
            The resulting OrPattern
        """
        return OrPattern(self, other)

    def __and__(self, other: "DFPattern") -> "AndPattern":
        """
        Syntax sugar for creating an AndPattern

        Parameters
        ----------
        other: DFPattern
            Additional pattern to satisfy

        Returns
        -------
        result: AndPattern
            The resulting AndPattern
        """
        return AndPattern(self, other)

    def __invert__(self) -> "NotPattern":
        """
        Syntax sugar for creating a DFPattern to reject

        Returns
        -------
        result: NotPattern
            The resulting NotPattern
        """
        return reject(self)

    def has_attr(self, attrs: dict[str, tvm_ffi.Object]) -> "AttrPattern":
        """
        Add an attribute constraint to this pattern

        Parameters
        ----------
        attrs: Dict[str, tvm_ffi.Object]

        Returns
        -------
        result: AttrPattern
            The resulting AttrPattern
        """
        attrs = make_node("ir.DictAttrs", **attrs)
        return AttrPattern(self, attrs)

    def has_ty(self, ty: "Type") -> "TypePattern":
        return TypePattern(self, ty)

    def has_dtype(self, dtype: str) -> "DataTypePattern":
        """
        Add a type constraint to this pattern

        Parameters
        ----------
        dtype: str
            The dtype to match

        Returns
        -------
        result: DataTypePattern
            The resulting DataTypePattern
        """
        return has_dtype(dtype, self)

    def has_shape(self, shape: list[Expr]) -> "ShapePattern":
        """
        Add a shape constraint to this pattern

        Parameters
        ----------
        shape: List[Expr]
            Expected shape list

        Returns
        -------
        result: ShapePattern
            The resulting ShapePattern

        Note
        ----
        has_shape assumes that the matched relax.Expr only has one
        output tensor. Use is_tuple for those with multiple outputs.
        """
        if not isinstance(shape, list | tuple) and not tvm_tirx.ir.is_prim_expr(shape):
            raise ValueError("has_shape takes a list or tuple as input.")
        return ShapePattern(pattern=self, shape=shape)

    def match(self, expr, var2val: dict[Var, Expr] | None = None) -> bool:
        """
        Match a relax.Expr syntactically

        Parameters
        ----------
        expr : tvm.relax.Expr
            The expression to match
        var2val : Optional[Dict[tvm.relax.Var, tvm.relax.Expr]]
            A mapping from relax.Var to relax.Expr for autojump.

        Returns
        -------
        result: bool
            Whether or not the expression matches the pattern

        Note
        ----
        Functions in Relax consist of blocks of bindings that are not syntactically connected.
        We use a mapping (i.e., var2val) to mitigate the gap. For example, when matching
        "relax.add(lv0, lv1)", given var2val, we match lv0's bound expression
        when the recursive pattern matching goes to check lv0. The var2val mapping
        can be computed through the tvm.relax.analysis.get_var2val function.
        """
        return ffi.match_expr(self, expr, var2val)  # type: ignore

    def extract_matched_expr(
        self, expr, var2val: dict[Var, Expr] | None = None
    ) -> dict["DFPattern", Expr] | None:
        """
        Match a relax.Expr and return a map from matching patterns to matched expressions.

        Parameters
        ----------
        expr : tvm.relax.Expr
            The expression to match
        var2val : Optional[Dict[tvm.relax.Var, tvm.relax.Expr]]
            A mapping from relax.Var to relax.Expr for autojump.

        Returns
        -------
        result: Optional[Dict[DFPattern, Expr]]
            Map from matching patterns to matched expressions.
            Return None if the pattern does not match expr.

        Note
        ----
        Check the note of `match` for the meaning of var2val.
        """
        return ffi.extract_matched_expr(self, expr, var2val)

    def used_by(self, other: Union["DFPattern", "PatternSeq"], index=-1) -> "PatternSeq":
        """
        The current pattern being used by another pattern (sequence)

        Parameters
        ----------
        other : Union[DFPattern, DFPattern]
            The consumer pattern (sequence)
        index : int, optional
            The argument index called by the consumer pattern, by default -1

        Returns
        -------
        result: PatternSeq
            A chained pattern sequence
        """
        return _used_by(self, other, index)

    def __xor__(self, other: Union["DFPattern", "PatternSeq"]) -> "PatternSeq":
        """Syntax sugar of DFPattern.used_by"""
        return self.used_by(other, -1)

    def only_used_by(self, other: Union["DFPattern", "PatternSeq"], index=-1) -> "PatternSeq":
        """
        The current pattern being **ONLY** used by another pattern (sequence)

        Parameters
        ----------
        other : Union[DFPattern, DFPattern]
            The consumer pattern (sequence)
        index : int, optional
            The argument index called by the consumer pattern, by default -1

        Returns
        -------
        result: PatternSeq
            A chained pattern sequence
        """
        return _only_used_by(self, other, index)

    def __rshift__(self, other: Union["DFPattern", "PatternSeq"]) -> "PatternSeq":
        """Syntax sugar of DFPattern.only_used_by"""
        return self.only_used_by(other, -1)

    def dup(self) -> "DFPattern":
        """
        Duplicate the current pattern (new object under different address)

        Returns
        -------
        DFPattern
            A duplicated pattern
        """
        return ffi.dup_pattern(self)  # type: ignore

    def fork_to(self, *args) -> None:
        """Fork the current pattern to multiple pattern branches"""
        for v in args:
            self ^ v

    def same_shape_as(self, *args: list["DFPattern"]) -> "SameShapeConstraint":
        """
        The current pattern with the same shape as another pattern (sequence)

        Parameters
        ----------
        other : List[DFPattern]
            The other pattern (sequence)

        Returns
        -------
        result: PatternSeq
            A chained pattern sequence
        """
        return SameShapeConstraint(self, *args)


class DFConstraint(Node):
    """Base class of all constraints."""


@register_df_node
class ExprPattern(DFPattern):
    """A pattern which matches an expression.

    Parameters
    ----------
    expr : tvm.relax.Expr
        The expression to match.
    """

    def __init__(self, expr: Expr):
        self.__init_handle_by_constructor__(ffi.ExprPattern, expr)  # type: ignore


@register_df_node
class VarPattern(DFPattern):
    """A pattern for Var.

    Parameters
    ----------
    name_hint: str
        The name of the variable. Optional, if not provided,
        the pattern will match any VarObj.
    """

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(ffi.VarPattern, name_hint)  # type: ignore


@register_df_node
class DataflowVarPattern(VarPattern):
    """A pattern for DataflowVar.

    Parameters
    ----------
    name_hint: str
        The name of the variable. Optional, if not provided,
        the pattern will match any VarObj.
    """

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(ffi.DataflowVarPattern, name_hint)  # type: ignore


@register_df_node
class GlobalVarPattern(DFPattern):
    """A pattern for GlobalVar.

    Parameters
    ----------
    name_hint: str
        The name of the variable. Optional, if not provided,
        the pattern will match any GlobalVarObj.
    """

    def __init__(self, name_hint: str = ""):
        self.__init_handle_by_constructor__(ffi.GlobalVarPattern, name_hint)  # type: ignore


@register_df_node
class ExternFuncPattern(DFPattern):
    """A external function pattern.

    Parameters
    ----------
    global_symbol: str
        The name of the function. Optional, if not provided,
        the pattern will match any ExternFuncObj.
    """

    def __init__(self, global_symbol: str = ""):
        self.__init_handle_by_constructor__(ffi.ExternFuncPattern, global_symbol)  # type: ignore


@register_df_node
class ConstantPattern(DFPattern):
    """A pattern matching a Relax Constant."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.ConstantPattern)  # type: ignore


@register_df_node
class CallPattern(DFPattern):
    """A pattern matching a function call node.

    Parameters
    ----------
    op: tvm.relax.dpl.DFPattern
        The operation to be called.

    args: List[tvm.relax.dpl.DFPattern]
        The arguments to the call or None to match any arguments.

    varg_default_wildcard: bool
        If True, args can be fewer than actual provided arguments.

    add_constraint: bool
        If True, automatically add "used-by" constraints between caller and callee expressions.

    Note
    ----
    By setting varg_default_wildcard to True, we can only focus on the argument
    patterns we specified. For example, CallPattern(Op, [A, B]) can match
    a call of Op(A, B) or Op(A, B, C, ...) that has more arguments. However,
    the specified argument patterns must be matched (i.e., A and B).
    """

    def __init__(
        self,
        op: "DFPattern",
        args: list["DFPattern"] | tuple["DFPattern", ...],
        varg_default_wildcard: bool = False,
        add_constraint=True,
    ):
        self.__init_handle_by_constructor__(
            ffi.CallPattern,
            op,
            args,
            varg_default_wildcard,  # type: ignore
        )

        if add_constraint:
            for i, arg in enumerate(args):
                arg.used_by(self, i)


@register_df_node
class FunctionPattern(DFPattern):
    """A pattern matching a function node in Relax.

    Parameters
    ----------
    params: List[tvm.relax.dpl.DFPattern]
        The parameters to the Function or None to match any parameters.

    body: tvm.relax.dpl.DFPattern
        The body fo the Function

    """

    def __init__(
        self,
        params: list["DFPattern"],
        body: "DFPattern",
    ):
        self.__init_handle_by_constructor__(ffi.FunctionPattern, params, body)  # type: ignore


@register_df_node
class TuplePattern(DFPattern):
    """A patern matching a Relax Tuple.

    Parameters
    ----------
    fields : Array[tvm.relax.dpl.DFPattern]
        The fields in the tuple.
    """

    def __init__(self, fields: list):
        self.__init_handle_by_constructor__(ffi.TuplePattern, fields)  # type: ignore

    def __getitem__(self, index: int | None) -> "TupleGetItemPattern":
        if index is not None:
            # support negative index for being pythonic
            if index < 0:
                index += len(self)
            if index >= len(self):
                raise IndexError("TuplePattern index out of range")
        else:
            index = -1  # -1 means matching any index
        return TupleGetItemPattern(self, index)

    def __len__(self):
        return len(self.fields)


@register_df_node
class UnorderedTuplePattern(DFPattern):
    """A patern matching a Relax Tuple unorderedly.

    Parameters
    ----------
    fields : Array[tvm.relax.dpl.DFPattern]
        The fields in the tuple.
    """

    def __init__(self, fields: Array):
        self.__init_handle_by_constructor__(ffi.UnorderedTuplePattern, fields)  # type: ignore

    def __len__(self):
        return len(self.fields)


@register_df_node
class TupleGetItemPattern(DFPattern):
    """Get index-th item from a TuplePattern.

    Parameters
    ----------
    tuple_value: tvm.relax.dpl.DFPattern
        The input tuple expression.

    index: Optional[int]
        The index to match; Default (None) to match a TupleGetItem with any index.
    """

    def __init__(self, tuple_value: "DFPattern", index: int | None = None):
        match_index = index if index is not None else -1
        self.__init_handle_by_constructor__(
            ffi.TupleGetItemPattern,
            tuple_value,
            match_index,  # type: ignore
        )


@register_df_node
class OrPattern(DFPattern):
    """Create a Pattern that can match one of two conditions

    Parameters
    ----------
    left: tvm.relax.dpl.DFPattern
        One possible matching pattern.
    right: tvm.relax.dpl.DFPattern
        One possible matching pattern.
    """

    def __init__(self, left: "DFPattern", right: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.OrPattern, left, right)  # type: ignore


@register_df_node
class AndPattern(DFPattern):
    """Create a Pattern that must match two conditions

    Parameters
    ----------
    left: tvm.relax.dpl.DFPattern
        One must-matching pattern.
    right: tvm.relax.dpl.DFPattern
        One must-matching pattern.
    """

    def __init__(self, left: "DFPattern", right: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.AndPattern, left, right)  # type: ignore


@register_df_node
class NotPattern(DFPattern):
    """Create a Pattern that matches the negation of a condition.

     Parameters
    ----------
    to_reject: tvm.relax.dpl.DFPattern
        The pattern to deny.
    """

    def __init__(self, to_reject: "DFPattern"):
        self.__init_handle_by_constructor__(ffi.NotPattern, to_reject)  # type: ignore


@register_df_node
class WildcardPattern(DFPattern):
    """A pattern which matches anything."""

    def __init__(self):
        self.__init_handle_by_constructor__(ffi.WildcardPattern)  # type: ignore


@register_df_node
class TypePattern(DFPattern):
    """A pattern that matches another pattern with a certain Type

    Parameters
    ----------
    pattern: tvm.relax.dpl.DFPattern
        The input pattern that needs type annotation.

    ty: tvm.relax.Type
        The type to match against
    """

    def __init__(self, pattern: "DFPattern", ty: "Type"):
        self.__init_handle_by_constructor__(
            ffi.TypePattern,
            pattern,
            ty,
        )  # type: ignore


@register_df_node
class DataTypePattern(DFPattern):
    """A pattern that matches another pattern with certain data type

    Parameters
    ----------
    pattern: tvm.relax.dpl.DFPattern
        The input pattern that needs type annotation.

    dtype: str
        The dtype to match.
    """

    def __init__(self, pattern: "DFPattern", dtype: str):
        self.__init_handle_by_constructor__(ffi.DataTypePattern, pattern, dtype)  # type: ignore


@register_df_node
class ShapePattern(DFPattern):
    """A pattern that matches another pattern with a certain tensor shape

    Parameters
    ----------
    pattern: tvm.relax.dpl.DFPattern
        The input pattern that needs type annotation.

    shape: List[tvm_tirx.ir.Expr]
        The shape to match.
    """

    def __init__(self, pattern: "DFPattern", shape: list[tvm_tirx.ir.Expr]):
        self.__init_handle_by_constructor__(ffi.ShapePattern, pattern, shape)  # type: ignore


@register_df_node
class SameShapeConstraint(DFConstraint):
    """A pattern that requires a set of patterns to have the same shape

    Parameters
    ----------
    args: List[DFPattern]
        A set of patterns which must all provide the same shape.
    """

    def __init__(self, *args: list[DFPattern]):
        self.__init_handle_by_constructor__(ffi.SameShapeConstraint, args)  # type: ignore


@register_df_node
class PrimArrPattern(DFPattern):
    """
    A pattern to match an array of Expr

    Parameters
    ----------
    shape : List[tvm_tirx.ir.Expr]
        The shape to match.
    """

    def __init__(self, shape: list[tvm_tirx.ir.Expr]):
        self.__init_handle_by_constructor__(ffi.PrimArrPattern, shape)  # type: ignore

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError("PrimArrPattern index out of range")
        return self.fields[index]

    def __len__(self):
        return len(self.fields)


@register_df_node
class AttrPattern(DFPattern):
    """Match an expression with certain attributes.
    Supports Op attributes, Call attributes, and Function attributes.

    Parameters
    ----------
    pattern: tvm.relax.dpl.DFPattern
        The input pattern.

    attrs: tvm_tirx.ir.attrs.Attrs
        The attributes to match.
    """

    def __init__(self, pattern: "DFPattern", attrs: tvm_tirx.ir.attrs.Attrs):
        self.__init_handle_by_constructor__(ffi.AttrPattern, pattern, attrs)  # type: ignore


def is_var(name: str = "") -> VarPattern:
    """
    Syntatic sugar for creating an optionally named VarPattern.

    Parameters
    ----------
    name: str
        The name of the input pattern to match.

    Returns
    -------
    result: tvm.relax.dpl.VarPattern
        The resulting pattern.
    """
    return VarPattern(name)


def is_gv(name: str = "") -> GlobalVarPattern:
    """Syntax sugar for creating an optionally (if name is empty) named GlobalVarPattern."""
    return GlobalVarPattern(name)


def is_dfv(name: str = "") -> DataflowVarPattern:
    """Syntax sugar for creating an optionally (if name is empty) named DataflowVarPattern."""
    return DataflowVarPattern(name)


def is_const() -> ConstantPattern:
    """
    Syntatic sugar for creating a ConstantPattern.

    Parameters
    ----------
    name: str
        The name of the input pattern to match.

    Returns
    -------
    result: tvm.relax.dpl.ConstantPattern
        The resulting pattern.
    """
    return ConstantPattern()


def is_expr(expr: Expr) -> ExprPattern:
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    expr: Expr
        The Relax expression to match.

    Returns
    -------
    result: tvm.relax.dpl.ExprPattern
        The resulting pattern.
    """
    return ExprPattern(expr)


def is_op(op_name: str) -> ExprPattern:
    """
    Syntatic sugar for creating an operator ExprPattern.

    Parameters
    ----------
    op_name: String
        The name of the tvm_tirx.ir.op.Op object

    Returns
    -------
    result: tvm.relax.dpl.ExprPattern
        The resulting ExprPattern
    """
    op = Op.get(op_name)
    return ExprPattern(op)


def is_tuple(fields: Array | list | tuple, unordered=False) -> TuplePattern | UnorderedTuplePattern:
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    fields : Array[tvm.relax.dpl.DFPattern]
        The fields in the tuple.

    Returns
    -------
    result: tvm.relax.dpl.DFPattern
        The resulting pattern.
    """
    if not isinstance(fields, list | tuple | Array):
        raise ValueError("fields must be a list, tuple, or Array")
    if unordered:
        return UnorderedTuplePattern(fields)
    return TuplePattern(fields)


def is_tuple_get_item(tuple_value: DFPattern, index: int | None = None) -> TupleGetItemPattern:
    """
    Syntatic sugar for creating an ExprPattern.

    Parameters
    ----------
    tuple_value: tvm.relax.dpl.DFPattern
        The input tuple expression.

    index: Optional[int]
        The index to match; Default (None) to match a TupleGetItem with any index.

    Returns
    -------
    result: tvm.relax.dpl.TupleGetItemPattern
        The resulting pattern.
    """
    return TupleGetItemPattern(tuple_value, index)


def wildcard() -> WildcardPattern:
    """
    Syntatic sugar for creating a WildcardPattern.

    Returns
    -------
    result: tvm.relax.dpl.WildcardPattern
        The resulting pattern.
    """
    return WildcardPattern()


def has_dtype(dtype: str, pattern: DFPattern = None) -> DataTypePattern:
    """
    Syntatic sugar for creating a DataTypePattern

    Parameters
    ----------
    dtype: str
        The dtype to match

    pattern: tvm.relax.dpl.DFPattern
        The pattern that needs type annotation

    Returns
    -------
    result: tvm.relax.dpl.DataTypePattern
        The resulting DataTypePattern
    """
    if pattern is None:
        pattern = wildcard()
    return DataTypePattern(pattern, dtype)


def is_shape(shape: list[tvm_tirx.ir.Expr]) -> "PrimArrPattern":
    """
    Directly matches a shape which is an array of Expr

    Parameters
    ----------
    shape : List[tvm_tirx.ir.Expr]
        The expected shape

    Returns
    -------
    PrimArrPattern
        The resulting PrimArrPattern pattern

    Raises
    ------
    ValueError
        If the argument shape is not a list/tuple/tvm_ffi.Array

    Note
    ----
    The difference between p.has_shape(s) and is_shape(s) is that: has_shape
    puts assumptions on the shape of the tensor matched by pattern p. While
    is_shape directly matches the shape (an array of Expr).
    """
    if not isinstance(shape, list | tuple | Array):
        raise ValueError("is_shape takes a list or tuple as input.")
    return PrimArrPattern(shape)


# Todo(relax-team): Dataflow pattern for Type, and match out_ty
def _is_call_tir(
    func_pattern: DFPattern,
    args: list | tuple | TuplePattern = None,
) -> CallPattern:
    if args is None:
        args = wildcard()
    elif isinstance(args, list | tuple):
        args = TuplePattern(args)

    return is_op("relax.call_tir")(func_pattern, args, add_constraint=False)


# Todo(relax-team): Dataflow pattern for Type, and match out_ty
def is_call_tir(
    func_name: str,
    args: list | tuple | TuplePattern = None,
) -> CallPattern:
    """
    Syntax sugar for creating a CallPattern for call_tir that calls an function through global var.

    Parameters
    ----------
    func_name : str
        Name of the CPS function to call.
    args : Union[List[DFPattern], Tuple[DFPattern]], optional
        Arguments in expected call_packed, by default None meaning arbitrary (number of) arguments
    Returns
    -------
    CallPattern
        The resulting CallPattern
    """
    func_pattern = GlobalVarPattern(func_name)
    return _is_call_tir(func_pattern, args)


def _is_call_dps_packed(
    func_pattern: DFPattern,
    args: list | tuple | TuplePattern = None,
) -> CallPattern:
    if args is None:
        args = wildcard()
    elif isinstance(args, list | tuple):
        args = TuplePattern(args)

    return is_op("relax.call_dps_packed")(func_pattern, args, add_constraint=False)


def is_call_dps_packed(
    func_name: str,
    args: list | tuple | TuplePattern = None,
) -> CallPattern:
    """Syntax sugar for creating a CallPattern for call_dps_packed

    Parameters
    ----------
    func_name : str
        Name of the CPS function to call.
    args : Union[List[DFPattern], Tuple[DFPattern]], optional
        Arguments in expected call_packed, by default None meaning arbitrary (number of) arguments

    Returns
    -------
    CallPattern
        The resulting CallPattern
    """
    func_pattern = ExternFuncPattern(func_name)
    return _is_call_dps_packed(func_pattern, args)


def is_call_packed(
    func_name: str, args: list[DFPattern] | tuple[DFPattern] | None = None
) -> CallPattern:
    """
    Syntax sugar for creating a CallPattern for call_packed

    Parameters
    ----------
    func_name : str
        Name of the external function to call
    args : Union[List[DFPattern], Tuple[DFPattern]], optional
        Arguments in expected call_packed, by default None meaning arbitrary (number of) arguments

    Returns
    -------
    CallPattern
        The resulting CallPattern
    """
    if args is None:
        return ExternFuncPattern(func_name)(varg_default_wildcard=True, add_constraint=False)
    return ExternFuncPattern(func_name)(*args)


def reject(pattern: DFPattern) -> NotPattern:
    """
    Syntax sugar for creating a DFPattern to reject

    Parameters
    ----------
    pattern : DFPattern
        The pattern to deny

    Returns
    -------
    result: NotPattern
        The resulting NotPattern
    """
    return NotPattern(pattern)


def has_attr(attrs, pattern=None) -> AttrPattern:
    """
    Syntatic sugar for creating an AttrPattern

    Parameters
    ----------
    attrs: Dict[str, tvm_ffi.Object]
        The attributes to match

    pattern: Optional[tvm.relax.dpl.DFPattern]
        The input pattern.

    Returns
    -------
    result: tvm.relax.dpl.DFPattern
        The resulting AttrPattern
    """
    if pattern is None:
        pattern = wildcard()
    return pattern.has_attr(attrs)


@register_df_node
class PatternSeq(Node):
    """A sequence of patterns with consecutive constraints"""

    def __init__(self, patterns: list[DFPattern], only_use=False):
        """
        Initializer to PatternSeq

        Parameters
        ----------
        patterns : List[DFPattern]
            A chain of patterns
        only_use : bool, optional
            Whether the patterns follows only-used-by relations consecutively, by default False
        """
        self.__init_handle_by_constructor__(ffi.PatternSeq, patterns, only_use)  # type: ignore

    def used_by(self, other: Union[DFPattern, "PatternSeq"], index=-1) -> "PatternSeq":
        """
        Assuming the right-most pattern must be used by the `other` pattern as a producer

        Parameters
        ----------
        other : Union[DFPattern, PatternSeq]
            The consumer pattern (sequence)
        index : int, optional
            The argument index called by the consumer pattern, by default -1

        Returns
        -------
        PatternSeq
            A chained pattern sequence

        Note
        ----
        If other is PatternSeq, it means the right-most pattern must be used by the left-most
        pattern of the other sequence.
        """
        return _used_by(self, other, index)

    def only_used_by(self, other: Union[DFPattern, "PatternSeq"], index=-1) -> "PatternSeq":
        """
        Assuming the right-most pattern must be **ONLY** used by the `other` pattern as a producer

        Parameters
        ----------
        other : Union[DFPattern, PatternSeq]
            The consumer pattern (sequence)
        index : int, optional
            The argument index called by the consumer pattern, by default -1

        Returns
        -------
        PatternSeq
            A chained pattern sequence

        Note
        ----
        If other is PatternSeq, it means the right-most pattern must be **ONLY** used by the
        left-most pattern of the other sequence.
        """
        return _only_used_by(self, other, index)

    def __getitem__(self, index: int) -> DFPattern:
        """
        Access the pattern at the given index

        Parameters
        ----------
        index : int
            Index of the accessed pattern

        Returns
        -------
        DFPattern
            The accessed pattern
        """
        return self.patterns[index]

    def __xor__(self, other) -> "PatternSeq":
        """Syntax sugar of PatternSeq.used_by"""
        return self.used_by(other, -1)

    def __rshift__(self, other) -> "PatternSeq":
        """Syntax sugar of PatternSeq.only_used_by"""
        return self.only_used_by(other, -1)

    def dup(self) -> "PatternSeq":
        """
        Duplicate the pattern sequence (new object under different address)

        Returns
        -------
        PatternSeq
            A duplicated chain
        """
        return ffi.dup_seq(self)  # type: ignore


### Private functions


def _used_by(
    lhs: DFPattern | PatternSeq,
    rhs: DFPattern | PatternSeq,
    index=-1,
) -> PatternSeq:
    if isinstance(lhs, DFPattern):
        lhs = PatternSeq([lhs])
    if isinstance(rhs, DFPattern):
        rhs = PatternSeq([rhs])
    return ffi.used_by(lhs, rhs, index)  # type: ignore


def _only_used_by(lhs: DFPattern | PatternSeq, rhs: DFPattern | PatternSeq, index=-1) -> PatternSeq:
    if isinstance(lhs, DFPattern):
        lhs = PatternSeq([lhs])
    if isinstance(rhs, DFPattern):
        rhs = PatternSeq([rhs])
    return ffi.only_used_by(lhs, rhs, index)  # type: ignore


def make_fused_bias_activation_pattern(
    op_name, with_bias=False, activation=None, allow_reshape=False
):
    """
    A simple utility to create patterns for an operation fused with bias addition and activation.

    Parameters
    ----------
    op_name: str
        The name of a Relax op, such as "relax.nn.conv2d"

    with_bias: bool
        Whether or not to include bias addition

    activation: str
        The name of an activation Relax op, such as "relax.nn.relu"

    allow_reshape: bool
        Whether to allow reshape operation before bias addition (for PyTorch frontend)

    Returns
    -------
    pattern: DFPattern
        The resulting pattern describing a fused operation
    """
    lhs = wildcard()
    rhs = wildcard()
    out = is_op(op_name)(lhs, rhs)

    if with_bias:
        bias = wildcard()
        if allow_reshape:
            reshaped_bias = is_op("relax.reshape")(bias, wildcard(), varg_default_wildcard=True)
            out = is_op("relax.add")(out, reshaped_bias, varg_default_wildcard=True)
        else:
            out = is_op("relax.add")(out, bias)

    if activation:
        return is_op(activation)(out)

    return out
