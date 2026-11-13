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
# pylint: disable=no-else-return, invalid-name
# pylint: disable=unidiomatic-typecheck
"""
This file contains the set of passes for Relax, which exposes an interface for
configuring the passes and scripting them in Python.
"""

from collections.abc import Callable
from enum import IntEnum

import tvm
import tvm_tirx
from tvm.relax.expr import Binding, DataflowBlock, Expr, Function, GlobalVar
from tvm.relax.type import FuncType
from tvm.s_tir import SBlock
from tvm_tirx import IRModule, tirx
from tvm_tirx.ir import Call, Type
from tvm_tirx.tirx import IndexMap, Var

from . import _ffi_api


def get_static_type(ty: Type) -> Type:
    """Get the corresponding static type from a Type.

    Parameters
    ----------
    ty : Type
        The input type.

    Returns
    -------
    ret : Type
        The corresponding static type.
    """
    return _ffi_api.GetStaticType(ty)  # type: ignore


def erase_to_well_defined(
    ty: Type,
    shape_var_map: dict[tirx.Var, tirx.Expr] | None = None,
    var_map: dict[Var, Expr] | None = None,
) -> Type:
    """Erase ty into a well defined form.

    This function removes the Type's dependencies on shape and vars that
    are not defined in given maps.

    Parameters
    ----------
    ty : Type
        The input type.

    shape_var_map : Dict[tirx.Var, tirx.Expr]
        Specifies the defined shape vars and the values they should map to.

    var_map : Dict[Var, Expr]
        Specifies the defined vars and the values they should map to.

    Returns
    -------
    ret : Type
        The corresponding erased type.
    """
    shape_var_map = {} if shape_var_map is None else shape_var_map
    var_map = {} if var_map is None else var_map

    return _ffi_api.EraseToWellDefined(ty, shape_var_map, var_map)  # type: ignore


class BaseCheckResult(IntEnum):
    """Return result of fine-grained base check.

    Note
    ----
    Base check comes with fine-grained fail levels.

    - FAIL_L0: The lhs and rhs have no intersection at all.
    - FAIL_L1: We get the failure by looking at static information.
    - FAIL_L2: We get the failure due to unknown symbolic variable relations.
    """

    FAIL_L0 = 0
    FAIL_L1 = 1
    FAIL_L2 = 2
    PASS = 3


def type_base_check(base: Type, derived: Type) -> BaseCheckResult:
    """Run a base check to see if base subsumes derived.

    Parameters
    ----------
    base: Type
        The base type.

    derived: Type
        The derived type.

    Returns
    -------
    ret : Type
        The derived return value type.
    """
    return _ffi_api.TypeBaseCheck(base, derived)  # type: ignore


def derive_call_ret_type(func_ty: FuncType, call: Call, ctx: "tvm.relax.BlockBuilder") -> Type:
    """Derive the call's ret value type from inputs.

    Parameters
    ----------
    func_ty: FuncType
        The call's function signature.

    call: Call
        The call expression

    ctx: tvm.relax.BlockBuilder
        The context block builder.

    Returns
    -------
    ret : Type
        The derived return value type.

    Note
    ----
    This is an internal derivation function, call.op field is
    ignored in this case and the derivation only depends on func_ty.
    """
    return _ffi_api.DeriveCallRetType(func_ty, call, ctx)  # type: ignore


def type_lca(lhs: Type, rhs: Type) -> Type:
    """Unify the two type to their least common ancestor.

    Parameters
    ----------
    lhs: Type
        The left operand.

    rhs: Type
        The right operand.

    Returns
    -------
    ret : Type
        The corresponding lca result.
    """
    return _ffi_api.TypeLCA(lhs, rhs)  # type: ignore


def tir_vars_in_type(ty: Type) -> list[tirx.Var]:
    """Get the TIR variables that appear in the input type.
    The returned list is deduplicated - each TIR variable will appear at most once.

    Parameters
    ----------
    ty : Type
        The type object to be analyzed.

    Returns
    -------
    ret : List[tirx.Var]
        The list of TIR variables that appear in the input type.
    """
    return _ffi_api.TIRVarsInType(ty)  # type: ignore


def definable_tir_vars_in_type(ty: Type) -> list[tirx.Var]:
    """Get the TIR variables that may be defined from input type.
    The returned list is deduplicated - each TIR variable will appear at most once.

    Parameters
    ----------
    ty : Type
        The type object to be analyzed.

    Returns
    -------
    ret : List[tirx.Var]

        The list of TIR variables that can be defined from the Type
    """
    return _ffi_api.DefinableTIRVarsInType(ty)  # type: ignore


def collect_non_negative_expressions(ty: Type) -> list[tirx.Expr]:
    """Collect TIR expressions used in non-negative contexts

    Get TIR variables that are non-negative within the context where
    the type is used.  For example, any expression used as a
    tensor shape.

    The returned list is deduplicated - each TIR expression will
    appear at most once.  The order of the list is in the order of
    occurrence within the type.

    Parameters
    ----------
    ty : Type
        The type object to be analyzed.

    Returns
    -------
    ret : List[tirx.Var]

        The list of TIR variables that can be defined from the Type

    """

    return _ffi_api.CollectNonNegativeExpressions(ty)  # type: ignore


def defined_symbolic_vars(func: Function) -> list[Var]:
    """Get the TIR variables that defined in the input function.
    The returned list is deduplicated - each TIR variable will appear at most once.

    Parameters
    ----------
    func : Function
        The function object to be analyzed.

    Returns
    -------
    ret : List[Var]
        The list of symbolic variables that are defined in the input function.
    """
    return _ffi_api.DefinedSymbolicVars(func)  # type: ignore


def free_symbolic_vars(func: Function) -> list[Var]:
    """Get the TIR variables that are used but not defined in the input function.
    The returned list is deduplicated - each TIR variable will appear at most once.

    Parameters
    ----------
    func : Function
        The function object to be analyzed.

    Returns
    -------
    ret : List[Var]
        The list of symbolic variables that are used but not defined in the input function.
    """
    return _ffi_api.FreeSymbolicVars(func)  # type: ignore


def bound_vars(expr: Expr) -> list[Var]:
    """
    Return all bound variables from expression expr.
    Bound variables are all variables that are declared in the expr.
    They only have meaning inside that expr, and can only be used in it.
    Parameters
    ----------
    expr: Expr
        The expression.
    Returns
    -------
    ret: List[Var]
        List of bound vars in expr, in post-DFS order
    """
    return _ffi_api.bound_vars(expr)


def free_vars(expr: Expr) -> list[Var]:
    """
    Return all free variables from expression expr.
    Free variables are variables that are not bound by a
    VarBinding or a function parameter in the expression.
    Parameters
    ----------
    expr: Expr
        The expression.
    Returns
    -------
    ret: List[Var]
        List of free vars in expr, in post-DFS order
    """
    return _ffi_api.free_vars(expr)


def all_vars(expr: Expr) -> list[Var]:
    """
    Return all (local) variables from expression expr.
    Parameters
    ----------
    expr: Expr
        The expression.
    Returns
    -------
    ret: List[Var]
        List of vars in expr, in post-DFS order
    """
    return _ffi_api.all_vars(expr)


def used_vars(expr: Expr) -> list[Var]:
    """
    Return all variables used in an expression.

    This function collects all variable references within the given expression,
    which is useful for analyzing variable dependencies.

    Parameters
    ----------
    expr: Expr
        The expression to analyze.

    Returns
    -------
    ret: List[Var]
        List of variables used in the expression.
    """
    return _ffi_api.used_vars(expr)  # type: ignore


def all_global_vars(expr: Expr) -> list[GlobalVar]:
    """
    Return all global variables from expression expr.
    Parameters
    ----------
    expr: Expr
        The expression.
    Returns
    -------
    ret: List[GlobalVar]
        List of global vars in expr, in post-DFS order
    """
    return _ffi_api.all_global_vars(expr)


def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.

    Parameters
    ----------
    expr : tvm.relax.Expr
        The input expression.

    fvisit : function
        The visitor function to be applied.
    """
    return _ffi_api.post_order_visit(expr, fvisit)  # type: ignore


def has_reshape_pattern(func: tirx.Function) -> bool:
    """Check if the given Function is essentially doing a reshape operation.
    The reshape operation also includes expand_dims, squeeze, flatten, etc.

    Here the allowed reshape pattern is: for example, assume the operation is
    `B[l_0, l_1, ..., l_b] = A[r_0, r_1, ..., r_a]`, we check if we can prove
    that the flattened index of l_0, ..., l_b under tensor B equals to the
    flattened index of r_0, ..., r_a under tensor A.

    Parameters
    ----------
    func : tirx.Function
        The function to be examined.

    Returns
    -------
    ret : bool
        A boolean indicating if the given Function is doing a reshape.

    Notes
    -----
    According to the description above, the returned result can only be
    false-negative and cannot be false-positive, since whenever we cannot
    prove the equality, we return false. This property guarantees the safety
    of this function.
    """
    return _ffi_api.has_reshape_pattern(func)  # type: ignore


def contains_impure_call(expr: Expr, own_name: Var | GlobalVar | None = None) -> bool:
    """
    Check if the given expression (likely a function body) contains any impure calls.

    Parameters
    ----------
    expr : Expr
        The expression to be examined. If expr is a function, we check the body.

    own_name : Var or GlobalVar (optional)
        For a recursive function, the analysis can ignore the self-calls
        for checking purity.

    Returns
    -------
    ret : bool
        True if there is an impure call
        (call to a function that may have visible side effects).

    Notes
    -----
    Relies on Type annotations, so ensure that the module has been normalized first.
    Also, an impure call in a *nested* function does *not* mean that the outer expression contains
    an impure call--it only does if the nested function is *later called*.
    """
    return _ffi_api.contains_impure_call(expr, own_name)


def get_var2val(func: Function) -> dict[Var, Expr]:
    """
    Get a mapping from Var to Expr for each variable in the function.

    Parameters
    ----------
    func : Function
        The input function to be analyzed.

    Returns
    -------
    Dict[Var, Expr]
        A mapping from Var to Expr.
    """
    return _ffi_api.get_var2val(func)  # type: ignore


def udchain(dfb: DataflowBlock) -> dict[Var, list[Var]]:
    """
    Analyze the variable use-def chain in a dataflow block.

    Parameters
    ----------
    dfb : DataflowBlock
        The dataflow block to analyze

    Returns
    -------
    Dict[Var, List[Var]]
        A mapping from variable definition to its uses.
    """
    return _ffi_api.udchain(dfb)  # type: ignore


def name_to_binding(func: Function) -> dict[str, list[Binding]]:
    """Return a map from variable name to its bindings."""
    return _ffi_api.name_to_binding(func)  # type: ignore


def remove_all_unused(func: Function) -> Function:
    """It removes:
    1. Unused local VarBindings in a DataflowBlock.
    2. Unused DataflowBlocks in a function.

    Parameters
    ----------
    func : Function
        The input function to be analyzed.

    Notes
    -----
    For IRModule-wise DCE, use py:func:`tvm.relax.transform.DeadCodeElimination`.

    Returns
    -------
    Function
        The function with unused variables removed.
    """
    return _ffi_api.remove_all_unused(func)  # type: ignore


def check_well_formed(obj: IRModule | Function, check_ty: bool = True) -> None:
    """Check if the IRModule is well formed, raising on the first violation.

    Raises an error (seeded with the offending node so a pass runner can report a
    precise access path) on the first well-formedness violation.

    Parameters
    ----------
    obj : Union[tvm_tirx.IRModule, Function]
        The input IRModule or relax.Function.

    check_ty : bool
        A boolean flag indicating if the property "every Expr must
        have defined type information" will be checked.

    Raises
    ------
    ValueError, TypeError
        If the function or module is not well formed.

    Note
    ----
    By default the type information is always checked. It is only in test cases
    where `check_ty` might be false, so that other well-formed requirements
    will be well tested and will not be blocked by not having type information.
    """
    _ffi_api.check_well_formed(obj, check_ty)  # type: ignore


def _get_function_default_dtype(func: tirx.Function):
    """Detect default index dtype from TensorType-annotated parameters."""
    for param in func.params:
        if tirx.is_tensor_var(param):
            for value in param.shape:
                return value.ty
    return "int64"


def suggest_layout_transforms(
    func: tirx.Function, write_tensor_transforms: list[IndexMap | Callable]
) -> dict[SBlock, dict[SBlock | Var, IndexMap]]:
    """Suggest Layout transformations of blocks and tensors in a Function.

    Parameters
    ----------
    func: Function
        Function on which analysis will be performed and transformations suggested.

    write_tensor_transforms: List[Union[IndexMap, Callable]
        List of layout transformations on the output tensors. The number of layout
        transformations must match the number of outputs of the Function.

    Returns
    -------
    ret: Dict[SBlock, Dict[Union[SBlock, Var], IndexMap]]
         Suggested transforms per block in `func`. For each block the returned value is a map
         from the object (block or tensor) to it's index map transformation.
    """
    write_tensor_index_maps = []
    default_index_dtype = _get_function_default_dtype(func)
    for transform in write_tensor_transforms:
        if callable(transform):
            transform = IndexMap.from_func(transform, index_dtype=default_index_dtype)
        assert isinstance(transform, IndexMap)
        write_tensor_index_maps.append(transform)
    return _ffi_api.suggest_layout_transforms(func, write_tensor_index_maps)  # type: ignore


def detect_recursion(mod: tvm_tirx.IRModule) -> list[list[GlobalVar]]:
    """
    Find all sets of recursive or mutually recursive functions in the module.

    Two or more functions are mutually recursive if there is some cycle of references
    among them. For example, if there are two functions A and B, they are
    mutually recursive if A calls B and B calls A. Another case would be with
    three functions A, B, and C, where A calls B, B calls C, and C calls A.

    (Note that functions do not have to call each other to reference each other.
    For example, if a function returns another function, that is still a reference
    that could potentially be recursive, even without a call.)


    If a function is simply recursive and not mutually recursive with any other,
    it will be reported as a group by itself.

    Parameters
    ----------
    mod: The module

    Returns
    -------
    ret: List[List[GlobalVar]]
        Each member of the list is a list of global functions
        that references each other mutually recursively.
        If a function is simply recursive and not mutually recursive
        with any other, it will be a singleton in this list.
    """
    return _ffi_api.detect_recursion(mod)  # type: ignore


def computable_at_compile_time(func: Function) -> list[Var]:
    """Collect variables whose value can be computed at compile-time

    If a function has the `kNumInput` attribute, then the first
    `kNumInput` parameters are provided at run-time, while all
    remaining parameters may be known at compile-time.  This utility
    collects all variable bindings that only depend, directly or
    indirectly, on the parameters known at compile-time.

    Parameters
    ----------
    func: Function

        The `relax.Function` to analyze

    Returns
    -------
    ret: List[Var]

        The set of variables that can be computed at compile-time, in
        order of their occurrence within the function.
    """
    return _ffi_api.computable_at_compile_time(func)  # type: ignore
