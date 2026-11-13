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
"""S-TIR block construction over shared primitive operations."""

import contextlib
import threading
from numbers import Integral
from typing import Any

import tvm_tirx
from tvm.s_tir.iter_var import IterVar
from tvm.s_tir.op import comm_reducer
from tvm.te import CommReducer, Reduce
from tvm_tirx import ir
from tvm_tirx import tirx as tir
from tvm_tirx.ir import IntImm, TensorLoad, TensorRegion, is_prim_expr
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.ir.type import _normalize_type
from tvm_tirx.runtime import MemoryScope
from tvm_tirx.script.ir_builder.base import MISSING, auto_emit_void_call
from tvm_tirx.script.parser.protocol_registry import register_mutable_decl as _register_mutable_decl
from tvm_tirx.tirx import Expr, Var
from tvm_tirx.tirx.layout import Layout
from tvm_tirx.tirx.script.ir_builder.ir import _get_layout, _record_meta_resource
from tvm_tirx.tirx.script.ir_builder.parser_protocol import region

from . import _ffi_api


def iter_var(v: Var | str, dom: ir.Range, iter_type: str, thread_tag: str) -> IterVar:
    """The iteration variable.

    Parameters
    ----------
    v : Union[Var, str]
        The internal variable that is used for iteration.

    dom : Range
        The domain of the iteration.

    iter_type : str
        The iteration type.

    thread_tag : str
        The thread type tag.

    Returns
    -------
    res : IterVar
        The iteration variable.
    """
    iter_type = getattr(IterVar, iter_type)
    return IterVar(dom, v, iter_type, thread_tag)


def async_copy_scope():
    """Mark eligible memory copies for asynchronous-copy lowering."""
    return region("s_tir.async_copy_scope", [])


@auto_emit_void_call
def async_commit(queue_id, *, ty=None, loc: Location = UNKNOWN_LOC):
    """Commit the preceding asynchronous copies to a queue."""
    return ir.Call("s_tir.async_commit", [queue_id], loc=loc, ty=ty)


@auto_emit_void_call
def async_wait(queue_id, inflight_count, *, ty=None, loc: Location = UNKNOWN_LOC):
    """Wait until at most inflight_count committed groups remain in flight."""
    return ir.Call("s_tir.async_wait", [queue_id, inflight_count], loc=loc, ty=ty)


def manual_sync():
    """Use explicitly authored synchronization within this region."""
    return region("s_tir.manual_sync", [])


@_register_mutable_decl("s_tir.match_tensor")
def match_tensor(
    param: TensorLoad | TensorRegion,
    shape: list[Expr] | tuple[Expr] | Expr | Integral = None,
    dtype: str = "float32",
    data: Var = None,
    scope: str | MemoryScope = MemoryScope("global"),
    align: int = -1,
    layout: str | Layout | None = MISSING,
    allocated_addr: Expr | int | tuple[Expr | int, ...] | None = None,
) -> Var:
    """Bind a tensor subregion inside an S-TIR block.

    Function inputs use Tensor annotations in the function signature.
    Shape and dtype are inferred when the source is a TensorRegion.
    """
    if not isinstance(param, TensorLoad | TensorRegion):
        raise TypeError("match_tensor requires a tensor subregion; use Tensor for function inputs")
    if isinstance(param, TensorRegion) and not tir.is_tensor_var(param.source):
        raise TypeError("match_tensor requires a TensorRegion with a TensorVar source")
    if shape is None:
        if isinstance(param, TensorRegion):
            dtype = param.source.ty.dtype
            shape = [region.extent for region in param.region]
        else:
            raise ValueError("Shape must be specified when binding a scalar tensor access")
    shape = (shape,) if is_prim_expr(shape) or isinstance(shape, Integral) else shape
    if allocated_addr is None:
        allocated_addr = []
    if not isinstance(allocated_addr, list | tuple):
        allocated_addr = [allocated_addr]
    result = _ffi_api.MatchTensor(  # type: ignore[attr-defined] # pylint: disable=no-member
        param,
        shape,
        _normalize_type(dtype),
        data,
        scope,
        align,
        _get_layout(layout, shape, scope),
        allocated_addr,
    )
    return result


_block_name_suffix = threading.local()


def _get_sblock_name_suffix() -> str:
    """Get the current block name suffix for macro expansion."""
    return getattr(_block_name_suffix, "value", "")


@contextlib.contextmanager
def block_name_suffix_context(block_suffix: str):
    """Context manager to set block name suffix during macro expansion.

    Parameters
    ----------
    block_suffix : str
        The suffix to append to block names (e.g., "_1", "_2").

    Yields
    ------
    None
    """
    old_suffix = getattr(_block_name_suffix, "value", "")
    _block_name_suffix.value = block_suffix
    try:
        yield
    finally:
        _block_name_suffix.value = old_suffix


def where(predicate: Expr | int) -> None:
    """The block predicate statement.

    Parameters
    ----------
    predicate : Union[Expr, Literal[0, 1]]
        The predicate condition.
    """
    if isinstance(predicate, bool):
        predicate = IntImm("bool", predicate)
    if isinstance(predicate, int):
        if predicate in [0, 1]:
            predicate = IntImm("bool", predicate)
        else:
            raise ValueError(f"Invalid value for predicate: {predicate}")
    _ffi_api.Where(predicate)  # type: ignore[attr-defined] # pylint: disable=no-member


def reads(*tensor_slices: list[TensorRegion | TensorLoad]) -> None:
    """The block tensor region reading statement.

    Parameters
    ----------
    tensor_slices : List[Union[TensorRegion, TensorLoad]]
        The array of tensor regions to read.
    """
    if len(tensor_slices) == 1:
        if isinstance(tensor_slices[0], tuple):
            tensor_slices = list(tensor_slices[0])
        elif isinstance(tensor_slices[0], list):
            tensor_slices = tensor_slices[0]  # type: ignore[assignment]
        else:
            tensor_slices = [tensor_slices[0]]
    else:
        tensor_slices = list(tensor_slices)  # type: ignore[assignment]
    _ffi_api.Reads(tensor_slices)  # type: ignore[attr-defined] # pylint: disable=no-member


def writes(*tensor_slices: list[TensorRegion | TensorLoad]) -> None:
    """The block tensor region writing statement.

    Parameters
    ----------
    tensor_slices : List[Union[TensorRegion, TensorLoad]]
        The array of tensor regions to write.
    """
    if len(tensor_slices) == 1:
        if isinstance(tensor_slices[0], tuple):
            tensor_slices = list(tensor_slices[0])
        elif isinstance(tensor_slices[0], list):
            tensor_slices = tensor_slices[0]  # type: ignore[assignment]
        else:
            tensor_slices = [tensor_slices[0]]
    else:
        tensor_slices = list(tensor_slices)  # type: ignore[assignment]
    _ffi_api.Writes(tensor_slices)  # type: ignore[attr-defined] # pylint: disable=no-member


def sblock_attr(attrs: dict[str, Any]) -> None:
    """The block annotation statement (for non-tirx SBlock usage).

    Parameters
    ----------
    attrs : Dict[str, Any]
        The annotation of the block.
    """
    return _ffi_api.BlockAttrs(attrs)  # type: ignore[attr-defined] # pylint: disable=no-member


def sblock_alloc_tensor(
    shape: list[Expr] | tuple[Expr] | Expr | Integral,
    dtype: str = "float32",
    data: Var = None,
    scope: str | MemoryScope = MemoryScope("global"),
    align: int = -1,
    layout: str | Layout | None = MISSING,
    allocated_addr: int | tuple[int, ...] | None = None,
) -> Var:
    """SBlock-level tensor allocation function.

    Parameters
    ----------
    shape : Union[List[Expr], Tuple[Expr], Expr, Integral]
        The type of the tensor prior to flattening.
    dtype : str
        The data type in the content of the tensor.
    data : Var
        The pointer to the head of the data.
    scope : str
        The optional storage scope of tensor data pointer.
    align : int
        The alignment requirement of data pointer in bytes.
    layout: Optional[Union[str, Layout]]
        The layout of the tensor.

    allocated_addr: Optional[Union[int, Tuple[int]]]
        The address of the allocated tensor. Might be multi-dimensional.
        There can be pooled storage scopes on some devices. For example,
        the Trainium device has a pooled storage scope for the SRAN tensors. ("trn.sbuf")
        CUDA has a pooled storage scope for the shared memory ("shared.dyn")

    Returns
    -------
    res : Var
        The allocated tensor.
    """
    shape = (shape,) if is_prim_expr(shape) or isinstance(shape, Integral) else shape
    if allocated_addr is None:
        allocated_addr = []
    if not isinstance(allocated_addr, list | tuple):
        allocated_addr = [allocated_addr]
    tensor = _ffi_api.SBlockAllocTensor(  # type: ignore[attr-defined] # pylint: disable=no-member
        shape,
        _normalize_type(dtype),
        data,
        scope,
        align,
        _get_layout(layout, shape, scope),
        allocated_addr,
    )
    _record_meta_resource(tensor, skip_frames=2)
    return tensor


def _as_range(dom: ir.Range | list[Expr]) -> ir.Range:
    """The range constructor.

    Parameters
    ----------
    dom : Union[Range, List[Expr]]
        The domain.

    Returns
    -------
    res : Range
        The Range.
    """
    if isinstance(dom, ir.Range):
        return dom
    if isinstance(dom, list | tuple):
        from tvm_tirx.sym import Analyzer  # pylint: disable=import-outside-toplevel

        extent = Analyzer().simplify(dom[1] - dom[0])
        if isinstance(extent, tvm_tirx.ir.IntImm):
            return ir.Range.from_min_extent(dom[0], extent)
        return ir.Range(dom[0], dom[1])
    if is_prim_expr(dom):
        return ir.Range(IntImm(dom.ty, 0), dom)
    return ir.Range(0, dom)


class axis:  # pylint: disable=invalid-name
    """The axis class"""

    @staticmethod
    def spatial(
        dom: ir.Range | list[Expr] | tuple[Expr],
        binding: Expr | None = None,
        dtype: str = "int32",
    ) -> Var:
        """The spatial block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[Expr], Tuple[Expr]]
            The domain of the iteration variable.

        binding : Expr, optional
            The binding value of the iteration variable. Omit only inside a
            block with no_realize=True.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisSpatial(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def reduce(
        dom: ir.Range | list[Expr] | tuple[Expr],
        binding: Expr | None = None,
        dtype: str = "int32",
    ) -> Var:
        """The reduced block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[Expr], Tuple[Expr]]
            The domain of the iteration variable.

        binding : Expr, optional
            The binding value of the iteration variable. Omit only inside a
            block with no_realize=True.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisReduce(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def scan(
        dom: ir.Range | list[Expr] | tuple[Expr],
        binding: Expr | None = None,
        dtype: str = "int32",
    ) -> Var:
        """The scanning block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[Expr], Tuple[Expr]]
            The domain of the iteration variable.

        binding : Expr, optional
            The binding value of the iteration variable. Omit only inside a
            block with no_realize=True.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisScan(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def opaque(
        dom: ir.Range | list[Expr] | tuple[Expr],
        binding: Expr | None = None,
        dtype: str = "int32",
    ) -> Var:
        """The opaque block axis defining function.

        Parameters
        ----------
        dom : Union[Range, List[Expr], Tuple[Expr]]
            The domain of the iteration variable.

        binding : Expr, optional
            The binding value of the iteration variable. Omit only inside a
            block with no_realize=True.

        dtype : str
            The data type of the iteration variable.

        Returns
        -------
        res : Var
            The iteration variable.
        """
        return _ffi_api.AxisOpaque(  # type: ignore[attr-defined] # pylint: disable=no-member
            _as_range(dom), binding, dtype
        )

    @staticmethod
    def remap(kinds: str, bindings: list[Expr], dtype: str = "int32") -> list[Var] | Var:
        """The block axis remapping function.

        Parameters
        ----------
        kinds : str
            The types of the iteration variables.

        bindings : List[Expr]
            The binding values of the iteration variables.

        dtype : str
            The data types of the iteration variables.

        Returns
        -------
        res : Var
            The iteration variables.
        """
        iter_vars = _ffi_api.AxisRemap(  # type: ignore[attr-defined] # pylint: disable=no-member
            kinds, bindings, dtype
        )
        return iter_vars[0] if len(iter_vars) == 1 else iter_vars

    S = spatial  # pylint: disable=invalid-name
    R = reduce  # pylint: disable=invalid-name


__all__ = [
    "CommReducer",
    "IterVar",
    "Reduce",
    "async_commit",
    "async_copy_scope",
    "async_wait",
    "axis",
    "block_name_suffix_context",
    "comm_reducer",
    "iter_var",
    "manual_sync",
    "match_tensor",
    "reads",
    "sblock_alloc_tensor",
    "sblock_attr",
    "where",
    "writes",
]
