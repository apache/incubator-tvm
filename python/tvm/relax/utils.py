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
# ruff: noqa: F401

# pylint: disable=invalid-name,too-many-locals

"""Utility functions for Relax"""

import itertools
import string
from collections.abc import Callable
from typing import Any, Optional

import tvm_ffi
from tvm_ffi import Array, Map

import tvm
import tvm_tirx
from tvm.relax.global_info import VDevice
from tvm_tirx import tirx
from tvm_tirx.ir import Attrs, StringImm, Type
from tvm_tirx.target import Target

from ..te import Tensor as te_Tensor
from ..te import create_function
from . import _ffi_api
from .expr import Expr, Function, ShapeExpr, te_tensor
from .expr import Tuple as rx_Tuple
from .type import ShapeType, TensorType


def metadata_partitioner(rx_txt: str) -> list[str]:
    """Extract Relax program and metadata section.

    Parameters
    ----------
    rx_txt : str
        The input relax text.

    Returns
    -------
    output : List[str]
        The result list of partitioned text, the first element
        is the relax program, and the second is metadata section.
    """
    partitions = []
    left_curly = 0
    meta_start = 0
    meta_end = 0
    for i, char in enumerate(rx_txt):
        if i < 0:
            raise ValueError("The program is invalid.")
        if char == "{":
            if meta_start == 0:
                meta_start = i
            left_curly += 1
        elif char == "}":
            left_curly -= 1
            if left_curly == 0:
                meta_end = i + 1
                break

    if meta_end == 0:
        raise ValueError("The metadata section was not found.")
    metadata = rx_txt[meta_start:meta_end]
    rx_program = rx_txt[meta_end:-1]

    partitions.append(rx_program)
    partitions.append(metadata)

    return partitions


def convert_to_expr(value: Any) -> Expr:
    """Helper function to convert the input to Expr, which follows the rules:
    1. Return the input itself if it's already a `relax.Expr`;
    2. Return `Expr` if the input is a primitive scalar;
    3. Return `tvm_tirx.ir.StringImm` if the input is `tvm.String` or `str`;
    4. Return `relax.Tuple` if the input is a tuple/list of `Expr`.
    """
    if isinstance(value, int):
        return tvm_tirx.ir.IntImm("int64", value)

    if isinstance(value, float):
        return tvm_tirx.ir.FloatImm("float64", value)

    tvm_value = tvm_ffi.convert(value)
    # Case 1
    if tvm_tirx.ir.is_prim_expr(tvm_value):
        return tvm_value
    # Case 2
    if isinstance(tvm_value, Expr):
        return tvm_value
    # Case 3
    if isinstance(tvm_value, str):
        return StringImm(value)
    # Case 4
    if isinstance(value, tuple | list):
        # `convert_to_expr` ensures that all elements are `Expr` if no exception raises
        return rx_Tuple([convert_to_expr(v) for v in value])
    raise TypeError(f"Cannot convert {value} with type {type(value)} to `relax.Expr`")


def copy_with_new_vars(func: Function) -> Function:
    """Copy the given function. All variables that are bound inside the original function
    would be copied to satisfy the restriction in the well-formed check: Variables in
    Relax must be bound exactly once. This also ensures that both the function and its copy
    can be inserted into the same IRModule, and be asserted on the structural equality
    agaisnt IRModule created by TVMScript.

    Parameters
    ----------
    func : Function
        The relax function to copy.

    Returns
    -------
    ret : Function
        The copied function.
    """
    return _ffi_api.CopyWithNewVars(func)  # type: ignore


def gen_call_tir_inputs(
    func: Callable, *args: Any, **kwargs: Any
) -> tuple[tirx.Function, Expr, tvm_tirx.ir.Type]:
    """Generate the inputs for call_tir according to the te function.
    This function converts arguments from relax expression to te tensor,
    The callback func should return a te tensor or a list of te tensors.

    Parameters
    ----------
    func : Callable
        A function that returns a te tensor or a list of te tensors.

    args : Any, optional
        arguments passed to the function.

    kwargs : Any, optional
        The keyword arguments passed to the function.
        Note that the keyword args 'function_attrs' is reserved for passing func
        attributes to be added to the Function that gets created.

    Returns
    -------
    ret : Tuple[tirx.Function, Expr, Type]
        ret contains the inputs for call_tir, including a tirx function, args,
        and the single result type argument.
    """

    tir_var_map: dict[tvm_tirx.ir.Var, tirx.Var] = {}

    call_tir_args = []
    create_function_args = []
    # extra list of tirx expression arguments
    # that are not covered by Tensor
    extra_tir_args_list = []

    def _copy_undefined_var(expr: tirx.Expr):
        def _visit_expr(e: tvm_tirx.ir.Var):
            if e not in tir_var_map:
                tir_var_map[e] = tvm_tirx.ir.Var(e.name, e.ty)

        tvm_ffi.structural_walk(expr, (tvm_tirx.ir.Var, _visit_expr), order="post")

    def _convert_te_arg(te_args: Any) -> Any:
        """Helper function used to convert Relax expressions to TE tensor.

        In the common case, the type of te_args is a Relax expression and is converted
        into a TE tensor.
        If te_args is a nested or recursive datatype (i.e list, dict, tvm_ffi.Map, tvm_ffi.Array),
        we recursive and convert any value of type Relax expression into a TE tensor.
        Common values of type int, float, and str are preserved.

        In dynamic shape cases, the passed in arguments may contain TIR variable.
        For example, the argument can be a Relax Var with TensorType, which
        has symbolic shape, or the argument can be a ShapeExpr with symbolic variables.
        To make the Function generated has independent variables with
        the caller Relax function, we will substitute the TIR variables in the input
        arguments with fresh ones, which is done by maintaining a TIR variable mapping.

        Parameters
        ----------
        te_args : Any
            Argument to convert to TE

        tir_var_map : Dict[tvm_tirx.ir.Var, tirx.Var]
            The variable mapping from caller-side canonical Vars to exact ordinary
            Vars used on the Function side.

        Returns
        -------
        ret : (Any, [tvm.te.Tensor])
            A tuple of the converted te_args, and a list of te tensors for each converted
            Relax expression
        """

        def _convert_te_arg_helper(arg):
            if (
                isinstance(arg, tvm_tirx.ir.Var)
                and tvm_tirx.ir.is_prim_expr(arg)
                and not (tvm_tirx.ir.is_prim_var(arg) and arg.ty.dtype == "int64")
            ):
                name = arg.name or f"scalar_input_{len(create_function_args)}"
                tir_param = tirx.Var(name, arg.ty.dtype)
                call_tir_args.append(arg)
                create_function_args.append(tir_param)
                return tir_param

            if tvm_tirx.ir.is_prim_expr(arg):
                _copy_undefined_var(arg)
                new_arg = tvm_ffi.structural_map(
                    arg, (tvm_tirx.ir.Var, lambda var: tir_var_map.get(var, var)), order="post"
                )
                extra_tir_args_list.append(new_arg)
                return new_arg

            if isinstance(arg, Expr):  # type: ignore
                if isinstance(arg.ty, TensorType):
                    assert isinstance(arg.ty.shape, ShapeExpr), (
                        "emit_te now only supports Tensor that has ShapeExpr shape"
                    )
                    for shape_value in arg.ty.shape.values:
                        _copy_undefined_var(shape_value)

                    n_args = len(create_function_args)
                    if isinstance(arg, tvm.relax.Var):
                        name = arg.name
                    elif n_args < len(string.ascii_uppercase):
                        name = string.ascii_uppercase[n_args]
                    else:
                        name = f"tensor_input_{n_args}"

                    te_arg = te_tensor(arg, tir_var_map, name)

                    call_tir_args.append(arg)
                    create_function_args.append(te_arg)

                    return te_arg

                if isinstance(arg.ty, ShapeType):
                    assert isinstance(arg, ShapeExpr), (
                        "For Expr having ShapeType, emit_te now only supports ShapeExpr"
                    )
                    return [_convert_te_arg_helper(val) for val in arg.values]

            elif isinstance(arg, list | Array):
                return [_convert_te_arg_helper(x) for x in arg]
            elif isinstance(arg, tuple):
                return tuple(_convert_te_arg_helper(x) for x in arg)
            elif isinstance(arg, dict | Map):
                for key in arg:
                    assert isinstance(key, str), (
                        "emit_te only supports dict with string as the key currently"
                    )
                return {k: _convert_te_arg_helper(arg[k]) for k in arg}
            elif isinstance(arg, int | float | str | Type | Attrs | Target) or arg is None:
                return arg
            raise TypeError(f"not supported type in emit_te: {type(arg)}")

        new_arg = _convert_te_arg_helper(te_args)
        return new_arg

    def _get_unbound_tir_vars(args: list[te_Tensor], extra_tir_args: list[Expr]) -> list[tirx.Var]:
        """get unbound TIR vars (i.e TIR vars used in the shape but is not
        itself a dimension of a shape)"""

        bound_vars = set()
        used_vars = set()

        def _populate_bound_vars(expr):
            if isinstance(expr, te_Tensor):
                for dim in expr.shape:
                    _populate_bound_vars(dim)
            elif isinstance(expr, tvm_tirx.ir.Var):
                bound_vars.add(expr)

        def _populate_used_vars(expr):
            if isinstance(expr, te_Tensor):
                for dim in expr.shape:
                    _populate_used_vars(dim)
            elif tvm_tirx.ir.is_prim_expr(expr):
                used_vars.update(tirx.analysis.undefined_vars(expr))

        for arg in itertools.chain(args, extra_tir_args):
            _populate_used_vars(arg)

        for arg in args:
            _populate_bound_vars(arg)

        diff = used_vars - bound_vars
        return list(diff)

    def _get_vdevice(arg: Any) -> VDevice | None:
        """get the virtual device from arguments."""
        vdevice = None
        if isinstance(arg, Expr):  # type: ignore
            if isinstance(arg.ty, TensorType):
                vdevice = arg.ty.vdevice
        elif isinstance(arg, list | Array | tuple):
            for x in arg:
                vdevice = _get_vdevice(x)
                if vdevice is not None:
                    return vdevice
        elif isinstance(arg, dict | Map):
            for k in arg:
                vdevice = _get_vdevice(arg[k])
                if vdevice is not None:
                    return vdevice
        return vdevice

    def _shape_with_old_tir_var(
        shape_values: list[tirx.Expr], tir_var_inverse_map: dict[tirx.Var, tirx.Expr]
    ):
        return ShapeExpr(
            [
                tvm_ffi.structural_map(
                    value,
                    (tvm_tirx.ir.Var, lambda var: tir_var_inverse_map.get(var, var)),
                    order="post",
                )
                for value in shape_values
            ]
        )

    function_attrs = kwargs.pop("function_attrs", None)
    custom_out_ty = kwargs.pop("ty_args", [])

    te_args = _convert_te_arg(args)
    te_kwargs = _convert_te_arg(kwargs)

    te_out = func(*te_args, **te_kwargs)
    assert isinstance(te_out, te_Tensor) or (
        isinstance(te_out, tuple | list | Array) and all(isinstance(t, te_Tensor) for t in te_out)
    ), "only support te.tensor or tuple/list/Array of te.tensor as function output"

    outs = [te_out] if isinstance(te_out, te_Tensor) else list(te_out)
    unbound_tir_vars = _get_unbound_tir_vars([*create_function_args, *outs], extra_tir_args_list)

    inputs = [*create_function_args, *unbound_tir_vars, *outs]
    tir_func = create_function(inputs, "int64")

    if function_attrs:
        tir_func = tir_func.with_attrs(function_attrs)

    tir_func = tir_func.without_attr("global_symbol")

    # Invert the TIR variable mapping, to convert the output shape back
    # with old set of variables.
    tir_var_inverse_map = {v: k for k, v in tir_var_map.items()}

    if len(custom_out_ty) == 1:
        output_ty = custom_out_ty[0]
    else:
        output_ty = [
            TensorType(
                _shape_with_old_tir_var(out.shape, tir_var_inverse_map),
                out.dtype,
                _get_vdevice(args),
            )
            for out in outs
        ]

    if isinstance(output_ty, list):
        output_ty = output_ty[0] if len(output_ty) == 1 else tvm_tirx.ir.TupleType(output_ty)

    call_tir_args.extend(
        tvm_ffi.structural_map(
            value, (tvm_tirx.ir.Var, lambda var: tir_var_inverse_map.get(var, var)), order="post"
        )
        for value in unbound_tir_vars
    )

    return (tir_func, call_tir_args, output_ty)
