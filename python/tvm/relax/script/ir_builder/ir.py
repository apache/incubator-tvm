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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name
"""Concrete Relax types, values and construction metadata."""

from __future__ import annotations

import builtins as _python
import inspect
import numbers as _numbers
import re as _re

from tvm import relax
from tvm import relax as _relax
from tvm.relax import Expr, ExternFunc, ShapeExpr, TupleGetItem, const
from tvm.relax.distributed import DeviceMesh as _DeviceMesh
from tvm.relax.distributed import DTensorType as _DTensorType
from tvm.relax.distributed import Placement as _Placement
from tvm.relax.distributed import device_mesh
from tvm.relax.dpl import PatternMatchingRewriter
from tvm.relax.global_info import DummyGlobalInfo, VDevice
from tvm.relax.type import FuncType as Func
from tvm_tirx import ir as _ir
from tvm_tirx.ir import IRModule
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.ir.type import MissingType as Missing
from tvm_tirx.ir.type import NoneType as NoneType
from tvm_tirx.ir.type import OpaqueType as Opaque
from tvm_tirx.ir.type import PrimType as Prim
from tvm_tirx.ir.type import PtrType as Ptr
from tvm_tirx.ir.type import StringType as String
from tvm_tirx.ir.type import TensorRegionType as TensorRegion
from tvm_tirx.ir.type import _normalize_type
from tvm_tirx.runtime._tensor import (
    cpu,
    cuda,
    ext_dev,
    hexagon,
    metal,
    opencl,
    rocm,
    vpi,
    vulkan,
    webgpu,
)
from tvm_tirx.script.ir_builder.base import LocationEntry as _LocationEntry
from tvm_tirx.script.ir_builder.base import resolve_global_info_args as _resolve_global_info_args
from tvm_tirx.script.ir_builder.ir import _global_infos
from tvm_tirx.script.ir_builder.ir import dtype as dtype
from tvm_tirx.script.ir_builder.parser_protocol import resolve_global_info_ as _resolve_global_info

from . import _ffi_api

py_tuple = _python.tuple
py_str = _python.str


def resolve_global_info_(content: py_str) -> _ir.GlobalInfo:
    """Resolve a module-owned named-list or virtual-device selector.

    Parameters
    ----------
    content : str
        Original global-info selector. "mesh[0]" indexes a named list;
        "cuda:1" selects the second CUDA vdevice; "vdevice:0" selects by absolute index.
        A trailing memory-scope suffix is accepted without changing device selection.

    Returns
    -------
    GlobalInfo
        The exact registered global-info object.

    Notes
    -----
    Lookup requires the nearest active native module frame and creates no metadata.
    Missing context, malformed selectors or unmatched devices raise ValueError; missing map
    entries or out-of-range indices propagate KeyError/IndexError. Non-string inputs
    raise TypeError; constructor argument handling preserves concrete objects before
    calling this hook. No source location is attached to an existing metadata object.

    .. code:: python

        # The constructor decorator calls this resolver for string selectors.
        R.Tensor((n,), "float32", vdevice="cuda:0")
        # Direct resolution uses the same module metadata.
        device = R.resolve_global_info_("cuda:0")
    """
    if not isinstance(content, _python.str):
        raise TypeError("Global-info selectors must be strings")
    if "[" in content or "]" in content:
        return _resolve_global_info(content)
    infos = _global_infos()
    selector = _re.fullmatch(r"([^:\[\]]+)(?::(\d+)(?::([^:]+))?)?", content)
    if selector is None:
        raise ValueError(f"Invalid global-info reference: {content!r}")
    target, index, _scope = selector.groups()
    ordinal = int(index) if index is not None else 0
    devices = infos.get("vdevice", ())
    if target == "vdevice":
        return devices[ordinal]
    for vdevice in devices:
        if vdevice.target.kind.name == target:
            if ordinal == 0:
                return vdevice
            ordinal -= 1
    raise ValueError(f"Global-info device reference was not found: {content!r}")


def dummy_global_info() -> DummyGlobalInfo:
    """Create a dummy global info expression.

    Returns
    -------
    res : DummyGlobalInfo
        The result dummy global info.
    """
    return DummyGlobalInfo()  # type: ignore[attr-defined] # pylint: disable=no-member


def vdevice(target=None, vdevice_id: int = 0, memory_scope: py_str = "global") -> VDevice:
    """Create a virtual device global info.
    Parameters
    ----------
    target
        The target.
    vdevice_id: int
        The virtual device index.
    memory_scope: py_str
        The memory scope, default is "global"

    Returns
    -------
    res : VDevice
        The result virtual device.
    """
    return VDevice(target, vdevice_id, memory_scope)  # type: ignore[attr-defined] # pylint: disable=no-member


def lookup_vdevice(target_kind: py_str | None = None, device_index: int = -1) -> VDevice:
    """Retrieve a virtual device from the active module's global-info list.

    Parameters
    ----------
    target_kind: str
        The target device kind, for example 'llvm' or 'cuda'. Use 'vdevice'
        to index the complete virtual-device list.
    device_index: int
        The zero-based index among devices of the selected target kind, or
        among all devices when target_kind is 'vdevice'.

    Returns
    -------
    res : VDevice
        The result virtual device.
    """
    _global_infos()  # Native lookup otherwise permits a default device without a module.
    return _ffi_api.LookupVDevice(target_kind, device_index)


def rewriter(rewriter_mod: IRModule | type) -> PatternMatchingRewriter:
    """Define a pattern-rewrite rule

    The IRModule must have two publicly-exposed functions, `pattern`
    and `replacement`, where `pattern` and `replacement` have the same
    function signature.

    .. code-block:: python

        @R.rewriter
        class RewriteAddIntoMultiply:
            @R.function
            def pattern(A: R.Tensor):
                B = A + A
                return B

            @R.function
            def replacement(A: R.Tensor):
                B = A * 2
                return B

    Parameters
    ----------
    rewriter_mod: Union[IRModule, Type]

        Either an IRModule that defines a rewrite pattern, or a
        TVMScript class that can be parsed into an IRModule.

    Returns
    -------
    rewriter: PatternMatchingRewriter

        A rewriter object, which can be applied either to a Relax
        function or to an entire IRModule.

    Notes
    -----
    Class members are parsed together after the class body completes. Their
    annotations use the decorator's original definition scope, which is released
    after parsing. An existing IRModule is used directly.

    """
    if not isinstance(rewriter_mod, IRModule):
        from tvm_tirx.script.parser.entry import parse
        from tvm_tirx.script.parser.inspect_source import capture_definition_scope

        if not inspect.isclass(rewriter_mod):
            raise TypeError(f"Expect a class, but got: {rewriter_mod}")
        frame = inspect.currentframe().f_back
        try:
            definition_scope = capture_definition_scope(frame)
            definition_source = (frame.f_code.co_filename, frame.f_lineno)
        finally:
            del frame
        try:
            module = parse(
                rewriter_mod,
                definition_scope=definition_scope,
                _definition_source=definition_source,
            )
        finally:
            del definition_scope
        module.__name__ = rewriter_mod.__name__
        rewriter_mod = module

    return PatternMatchingRewriter.from_module(rewriter_mod)


def tuple(*fields: Expr) -> Expr:
    """Create a tuple expression from already constructed expression fields.

    In script bindings and returns, prefer ordinary Python tuple syntax,
    including ``()`` and ``(value,)``, when fields are already expressions.
    This helper materializes an IR tuple outside parser conversion contexts.
    Numeric fields use core constructor conversion: for example, ``tuple(1)``
    has an int32 field, whereas a script tuple literal ``(1,)`` has an int64
    field. Use explicitly typed expressions to make either spelling equivalent.

    Parameters
    ----------
    *fields : Expr
        The fields of the tuple.
    Returns
    -------
    res : Expr
        The result tuple.
    """
    if len(fields) == 0:
        fields = py_tuple()

    return relax.Tuple(fields)  # type: ignore[attr-defined] # pylint: disable=no-member


shape = ShapeExpr
prim_value = relax.prim_value
str = _ir.StringImm


@_resolve_global_info_args("vdevice", resolver=resolve_global_info_)
def Tensor(
    shape=None, dtype=None, vdevice=None, ndim=-1, *, loc: _LocationEntry | Location = UNKNOWN_LOC
):
    """Construct a Relax tensor type.

    Parameters
    ----------
    shape : Expr or sequence of Expr, optional
        Tensor shape, or None when unknown. A string supplied without dtype
        is shorthand for the dtype. Symbolic dimensions are expressions over
        explicit variables, such as those created with I.dynamic.
    dtype : str or PrimType, optional
        Element type; None leaves the element type unknown.
    vdevice : VDevice or str, optional
        Concrete virtual device or a module metadata selector, such as "cuda:0".
        None leaves the virtual device unspecified. Strings use metadata from an
        active module builder. Use postponed annotations to defer resolution
        until function construction.
    ndim : int, optional
        Rank when shape is unknown; -1 means unknown rank. Do not supply
        an explicit rank together with a known shape.
    loc : LocationEntry or Location, optional
        Source location attached to the constructed IR; UnknownLoc leaves it unspecified.

    Returns
    -------
    result : TensorType
        The constructed tensor type.
        String selectors outside an active module raise ValueError.
    """
    if isinstance(shape, _python.str) and dtype is None:
        dtype, shape = shape, None
    return _relax.TensorType(
        shape, dtype, vdevice, ndim, loc.loc if isinstance(loc, _LocationEntry) else loc
    )


@_resolve_global_info_args("device_mesh", resolver=resolve_global_info_)
def DTensor(
    shape=None,
    dtype=None,
    device_mesh=None,
    placement="",
    *,
    ndim=-1,
    loc: _LocationEntry | Location = UNKNOWN_LOC,
):
    """Construct a Relax distributed tensor type.

    Parameters
    ----------
    shape : Expr or sequence of Expr, optional
        Global tensor shape, or None when unknown. Symbolic dimensions are
        expressions over explicit variables, such as those created with I.dynamic.
    dtype : str or PrimType, optional
        Element type; None leaves the element type unknown.
    device_mesh : DeviceMesh or str, optional
        Concrete mesh or module metadata selector. None creates an empty mesh
        placeholder. Strings require an active module builder. Use postponed
        annotations to defer resolution until function construction.
    placement : Placement or str, optional
        Distribution placement. Text, including the default empty string, is
        parsed with Placement.from_text.
    ndim : int, optional
        Global rank when shape is unknown; -1 means unknown rank.
    loc : LocationEntry or Location, optional
        Source location attached to the constructed IR; UnknownLoc leaves it unspecified.

    Returns
    -------
    result : DTensorType
        The constructed distributed type.
        String selectors outside an active module raise ValueError.
    """
    if device_mesh is None:
        device_mesh = _DeviceMesh([], _ir.Range(0, 1))
    if isinstance(placement, _python.str):
        placement = _Placement.from_text(placement)
    return _DTensorType(
        Tensor(shape, dtype, ndim=ndim),
        device_mesh,
        placement,
        loc.loc if isinstance(loc, _LocationEntry) else loc,
    )


def Shape(values=None, ndim=-1, *, loc: _LocationEntry | Location = UNKNOWN_LOC):
    """Construct a Relax shape type.

    Parameters
    ----------
    values : sequence of Expr, optional
        Known dimensions, or None for an unknown shape value. Symbolic dimensions
        are expressions over explicit variables, such as those created with I.dynamic.
    ndim : int, optional
        Number of dimensions when values is None; -1 leaves it unknown.
        Do not supply an explicit count together with known values.
    loc : LocationEntry or Location, optional
        Source location attached to the constructed IR; UnknownLoc leaves it unspecified.

    Returns
    -------
    result : ShapeType
        The constructed shape type.
    """
    return _relax.ShapeType(values, ndim, loc.loc if isinstance(loc, _LocationEntry) else loc)


def _type(value):
    if value is None:
        return _ir.TupleType([])
    if isinstance(value, _ir.Type) or (isinstance(value, type) and issubclass(value, _ir.Type)):
        return _normalize_type(value)
    if callable(value):
        value = value()
    if _ir.is_prim_expr(value):
        value = value.ty
    return _normalize_type(value)


def Callable(
    params=None,
    ret=None,
    purity=None,
    derive_func=None,
    *,
    loc: _LocationEntry | Location = UNKNOWN_LOC,
):
    """Construct a concrete or opaque Relax function type.

    Parameters
    ----------
    params : Type, callable, or sequence of annotations, optional
        Parameter annotations. A single annotation is accepted; None creates
        an opaque callable with an unspecified parameter list.
    ret : Type or callable, optional
        Return annotation. None means an empty tuple for a concrete callable
        and an unspecified result for an opaque callable.
    purity : bool, optional
        Whether the callable is pure. None selects True for a concrete
        parameter list and False for an opaque callable.
    derive_func : str or EnvFunc, optional
        Custom result-type derivation for an opaque callable. It is not
        accepted when params supplies a concrete parameter list.
    loc : LocationEntry or Location, optional
        Source location attached to the constructed IR; UnknownLoc leaves it unspecified.

    Returns
    -------
    result : FuncType
        The constructed function type.

    Notes
    -----
    Annotations may be types, primitive expressions supplying their types, or
    zero-argument factories returning either. Opaque result and derivation rules
    follow :meth:`tvm.relax.FuncType.opaque_func`.
    """
    if purity is None:
        purity = params is not None
    if params is None:
        return _relax.FuncType.opaque_func(
            ret=None if ret is None else _type(ret),
            derive_func=derive_func,
            purity=purity,
            loc=loc.loc if isinstance(loc, _LocationEntry) else loc,
        )
    if derive_func is not None:
        raise ValueError("A derivation function requires an opaque callable")
    if not isinstance(params, list | _python.tuple):
        params = [params]
    return _relax.FuncType(
        [_type(param) for param in params],
        _type(ret),
        purity,
        loc.loc if isinstance(loc, _LocationEntry) else loc,
    )


def Tuple(*fields, loc: _LocationEntry | Location = UNKNOWN_LOC):
    """Construct a Relax tuple type.

    Parameters
    ----------
    fields : Type or callable
        Field annotations as positional arguments, or one list or tuple.
        Each annotation is normalized to a type; None denotes an empty tuple.
    loc : LocationEntry or Location, optional
        Source location attached to the constructed IR; UnknownLoc leaves it unspecified.

    Returns
    -------
    result : TupleType
        The tuple type with fields in the supplied order.
    """
    if len(fields) == 1 and isinstance(fields[0], list | _python.tuple):
        fields = fields[0]
    return _ir.TupleType(
        [_type(field) for field in fields], loc.loc if isinstance(loc, _LocationEntry) else loc
    )


def Object(*, loc: _LocationEntry | Location = UNKNOWN_LOC):
    """Construct the unconstrained Relax value type.

    Parameters
    ----------
    loc : LocationEntry or Location, optional
        Source location attached to the constructed IR; UnknownLoc leaves it unspecified.

    Returns
    -------
    result : AnyType
        A type accepting any Relax value.
    """
    return _relax.AnyType(loc.loc if isinstance(loc, _LocationEntry) else loc)


def type_var(name, *, dtype=None, loc: _LocationEntry | Location = UNKNOWN_LOC):
    """Construct a fresh standalone primitive symbol.

    Parameters
    ----------
    name : str
        Name of the symbol.
    dtype : str or PrimType, optional
        Primitive type of the symbol; None selects "int64".
    loc : LocationEntry or Location, optional
        Source location attached to the constructed IR; UnknownLoc leaves it unspecified.

    Returns
    -------
    result : Var
        The newly constructed primitive variable.

    Notes
    -----
    This constructor creates a new symbol on each call. Reuse the returned
    variable to share its identity across annotations and function parameters.
    """
    return _ir.Var(
        name,
        "int64" if dtype is None else dtype,
        loc.loc if isinstance(loc, _LocationEntry) else loc,
    )


def _value(value, ty=None):
    if isinstance(value, _python.tuple):
        return _relax.utils.convert_to_expr(value)
    if isinstance(value, _numbers.Number):
        if isinstance(ty, _ir.PrimType):
            return _relax.prim_value(value, dtype=ty.dtype)
        return _relax.const(value)
    return value


def match_cast(value, ty, *, loc: _LocationEntry | Location = UNKNOWN_LOC):
    """Construct a match-cast descriptor for the binding hook.

    Parameters
    ----------
    value : Expr or Python value
        Value to match against the asserted type. Numbers and Python tuples
        are converted to Relax expressions; None is not accepted.
    ty : Type or callable
        Asserted type, or a zero-argument factory producing its annotation.
    loc : LocationEntry or Location, optional
        Source location attached to the constructed IR; UnknownLoc leaves it unspecified.

    Returns
    -------
    result : MatchCast
        An unbound match-cast descriptor, consumed by the language variant
        binding hook to emit and name the result.
    """
    if value is None:
        raise ValueError("The match-cast value cannot be None")
    ty = _type(ty)
    return _relax.MatchCast(
        _ir.Var("", ty), _value(value), ty, loc.loc if isinstance(loc, _LocationEntry) else loc
    )


Any = Object
PackedFunc = _relax.PackedFuncType
Range = _ir.Range

__all__ = [
    "Any",
    "Callable",
    "DTensor",
    "ExternFunc",
    "Func",
    "Missing",
    "NoneType",
    "Object",
    "Opaque",
    "PackedFunc",
    "Prim",
    "Ptr",
    "Range",
    "Shape",
    "ShapeExpr",
    "String",
    "Tensor",
    "TensorRegion",
    "Tuple",
    "TupleGetItem",
    "const",
    "cpu",
    "cuda",
    "device_mesh",
    "dtype",
    "dummy_global_info",
    "ext_dev",
    "hexagon",
    "lookup_vdevice",
    "match_cast",
    "metal",
    "opencl",
    "prim_value",
    "rewriter",
    "rocm",
    "shape",
    "str",
    "tuple",
    "type_var",
    "vdevice",
    "vpi",
    "vulkan",
    "webgpu",
]
