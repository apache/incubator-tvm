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
"""Manipulation operators."""

from collections.abc import Callable

import tvm_ffi

from tvm_tirx.ir import Attrs, FloatImm, IntImm, is_prim_expr
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.attrs import make_node as _make_attrs
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.runtime import DataTypeCode
from tvm_tirx.tirx import IndexMap

from ..expr import Expr, ShapeExpr, prim_value
from ..expr import Tuple as RxTuple
from . import _ffi_api

PrimExprLike = int | Expr


def broadcast_to(
    x: Expr, shape: tuple[PrimExprLike] | Expr, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Broadcasts a tensor to a specified shape.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    shape : Union[Tuple[PrimExprLike], Expr]
        The target shape.

    Returns
    -------
    result : relax.Expr
        The broadcasted tensor.
    """
    if isinstance(shape, tuple | list):
        shape = ShapeExpr(shape)
    return _Call("relax.broadcast_to", [x, shape], loc=loc, ty=ty)  # type: ignore


@tvm_ffi.register_object("relax.attrs.ConcatAttrs")
class ConcatAttrs(Attrs):
    """Attributes for concat operator"""


def concat(
    tensors: Expr | list[Expr],
    axis: int | None = 0,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Concatenate the input tensors along the given axis.

    Parameters
    ----------
    tensors : Union[relax.Expr, List[relax.Expr]]
        An Expr in Tuple type, containing the tensors to be concatenated,
        or a list of Tensors.

    axis : Optional[int]
        The axis along which the tensors are concatenated.
        If `axis` is `None`, the input tensor is required to be flattened before concatenation.

    Returns
    -------
    result: relax.Expr
        The concatenated tensor.
    """
    if isinstance(tensors, list | tuple):
        tensors = RxTuple(tensors)
    return _Call(
        "relax.concat",
        [tensors],
        attrs=_make_attrs("relax.attrs.ConcatAttrs", axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.ExpandDimsAttrs")
class ExpandDimsAttrs(Attrs):
    """Attributes for expand_dims operator"""


def expand_dims(x: Expr, axis: int | list[int], *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Insert new axes at the positions given by `axis`.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    axis : Union[int, List[int]]
        The axes at which the input array are expanded.
        All values are required to lie in range `[-data.ndim - 1, data.ndim]`, with the convention
        of negative indexing.

    Returns
    -------
    result : relax.Expr
        The transformed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.expand_dims",
        [x],
        attrs=_make_attrs("relax.attrs.ExpandDimsAttrs", axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore


def flatten(x: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Flatten all the tensor dimensions into one.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    Returns
    -------
    result : relax.Expr
        The flattened result.
    """
    return _Call("relax.flatten", [x], loc=loc, ty=ty)  # type: ignore


@tvm_ffi.register_object("relax.attrs.LayoutTransformAttrs")
class LayoutTransformAttrs(Attrs):
    """Attributes used in layout_transform operator"""


def layout_transform(
    x: Expr,
    index_map: Callable | IndexMap,
    pad_value: int | float | Expr | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    """Modifies the layout of a tensor.

    Parameters
    ----------
    x : relax.Expr
        The input tensor to the operator.

    index_map : Callable | IndexMap
        The transformation to apply.

    pad_value : Optional[int | float | Expr]
        The value used for padding if the transformation results in implicit padding.
        If not specified, any value can be used.

    Returns
    -------
    result : relax.Expr
        The transformed tensor.
    """
    default_index_dtype = "int64"

    if callable(index_map):
        index_map = IndexMap.from_func(index_map, index_dtype=default_index_dtype)
    x_dtype = x.ty.dtype

    # Explicitly convert python int/float pad_value to the x's type.  If the default behavior
    # is applied, it would be converted to int32/float32, which may not match the x's type.
    if pad_value is None:
        pass
    elif not is_prim_expr(pad_value):
        if x_dtype.matches_code(DataTypeCode.INT, DataTypeCode.UINT) and isinstance(pad_value, int):
            pad_value = IntImm(x_dtype.dtype, pad_value)
        elif x_dtype.matches_code(DataTypeCode.FLOAT, DataTypeCode.BFLOAT) and (
            isinstance(pad_value, int | float)
        ):
            pad_value = FloatImm(x_dtype.dtype, float(pad_value))
        pad_value = prim_value(pad_value)

    return _Call(
        "relax.layout_transform",
        [x],
        attrs=_make_attrs(
            "relax.attrs.LayoutTransformAttrs", index_map=index_map, pad_value=pad_value
        ),
        loc=loc,
        ty=ty,
    )


@tvm_ffi.register_object("relax.attrs.PermuteDimsAttrs")
class PermuteDimsAttrs(Attrs):
    """Attributes for permute_dims operator"""


def permute_dims(
    x: Expr, axes: list[int] | None = None, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Permutes the dimensions of an array.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    axes : Optional[List[int]]
        The target axes order. If not specified, permute_dims will reverse the order of all axes.

    Returns
    -------
    result : relax.Expr
        The transposed result.
    """
    return _Call(
        "relax.permute_dims",
        [x],
        attrs=_make_attrs("relax.attrs.PermuteDimsAttrs", axes=axes),
        loc=loc,
        ty=ty,
    )  # type: ignore


def reshape(
    x: Expr, shape: tuple[PrimExprLike] | Expr, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Reshape the input array.

    ``-1`` infers the dimension of the output shape by using the remainder of
    the input dimensions keeping the size of the new array same as that of the input array.
    At most one dimension of shape can be -1.

        .. code-block:: python

            x.shape = (2, 3, 4), shape = (6, 1, -1), result.shape = (6, 1, 4)
            x.shape = (2, 3, 4), shape = (3, -1, 8), result.shape = (3, 1, 8)
            x.shape = (2, 3, 4), shape = (-1,), result.shape = (24,)

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    shape : Union[Tuple[PrimExprLike], Expr]
        The new shape. Should be compatible with the original shape.

    Returns
    -------
    result : relax.Expr
        The reshaped result.

    Note
    ----
    The ``-1`` inference is only performed at compile-time.
    That is to say, in any case the dimension length of ``-1`` cannot be inferred in
    compile-time, an error will be thrown.
    """
    if not isinstance(shape, tuple | list | Expr) or is_prim_expr(shape):
        raise TypeError("shape must be a tuple/list or a Relax shape expression")
    shape = _ffi_api._NormalizeReshapeShape(x, shape)
    return _Call("relax.reshape", [x, shape], loc=loc, ty=ty)  # type: ignore


@tvm_ffi.register_object("relax.attrs.SplitAttrs")
class SplitAttrs(Attrs):
    """Attributes used in split operator"""


def split(
    x: Expr,
    indices_or_sections: int | list[PrimExprLike],
    axis: int = 0,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Split input tensor along axis by sections or indices.

    If indices_or_sections is an integer, the input will be divided equally
    along given axis (if possible). Last section will be smaller if the tensor
    size along the given dimension is not divisible by the integer.

    If indices_or_sections is a tuple of mixture of int or Expr,
    the entries indicate the indices where along axis the array is split.

    Parameters
    ----------
    x : relax.Expr
        The tensor to be split.

    indices_or_sections : Union[int, List[PrimExprLike]]
        Indices or sections to split into. Accepts an int or a list.

    axis : int
        The axis over which to split.

    Returns
    -------
    ret : relax.Expr
        The computed result.
    """
    if isinstance(indices_or_sections, int):
        indices_or_sections = IntImm("int64", indices_or_sections)
    elif isinstance(indices_or_sections, list | tuple):
        indices_or_sections = [IntImm("int64", int(index)) for index in indices_or_sections]
    if isinstance(indices_or_sections, IntImm):
        if int(indices_or_sections) <= 0:
            raise RuntimeError("The number of split sections must be positive")
    elif not isinstance(indices_or_sections, list | tuple):
        raise TypeError("indices_or_sections must be an integer or a list of integers")
    return _Call(
        "relax.split",
        [x],
        attrs=_make_attrs(
            "relax.attrs.SplitAttrs", indices_or_sections=indices_or_sections, axis=axis
        ),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.SqueezeAttrs")
class SqueezeAttrs(Attrs):
    """Attributes for squeeze operator"""


def squeeze(
    x: Expr, axis: int | list[int] | None = None, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Squeeze axes in the array.

    Parameters
    ----------
    x : relax.Expr
        The input data to the operator.

    axis : Optional[Union[int, List[int]]
        The set of axes to remove.
        If axis = None, remove all axis of dimensions 1.
        If any specified axis has dimension that does not equal 1, it is an error.

    Returns
    -------
    result : relax.Expr
        The squeezed result.
    """
    if isinstance(axis, int):
        axis = [axis]
    return _Call(
        "relax.squeeze",
        [x],
        attrs=_make_attrs("relax.attrs.SqueezeAttrs", axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.StackAttrs")
class StackAttrs(Attrs):
    """Attributes for concat operator"""


def stack(
    tensors: Expr | list[Expr], axis: int = 0, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Stack the input tensors along a new axis.

    Parameters
    ----------
    tensors : Union[relax.Expr, List[relax.Expr]]
        An Expr in Tuple type, containing the tensors to be stacked,
        or a list of Tensors. All input tensors must have the same shape.

    axis : int
        The axis in the resulting tensor along which the input tensors will be stacked.
        Negative values wrap around. Default is 0.

    Returns
    -------
    result: relax.Expr
        The stacked tensor with an additional dimension compared to the input tensors.

    """
    if isinstance(tensors, list | tuple):
        tensors = RxTuple(tensors)
    return _Call(
        "relax.stack",
        [tensors],
        attrs=_make_attrs("relax.attrs.StackAttrs", axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore


def collapse_sum_like(
    data: Expr, collapse_target: Expr, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Return a summation of data to the shape of collapse_target.

    For details, please see relax.op.collapse_sum_to.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    collapse_target : relax.Expr
        The tensor whose shape is the shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor after summation.
    """
    return _Call(
        "relax.collapse_sum_like",
        [data, collapse_target],
        loc=loc,
        ty=ty,
    )  # type: ignore


def collapse_sum_to(
    data: Expr, shape: tuple[PrimExprLike] | Expr, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Return a summation of data to the given shape.

    collapse_sum_to is intended as the backward operator of tvm.relax.op.broadcast_to and
    other broadcast operators in the automatic differentiation process.

    We expect that data is the result of broadcasting some tensor of the given shape in some
    broadcast operation. Thus the given `shape` and `data.shape` must follow broadcast rules.

    During computation, all axes of `data.shape` and `shape` are checked from right to left.
    For an axis, if it follows these rules, `data` will be summed over this axis:
    - the axis exists in `data.shape` but not in `shape`, or
    - the axis exists in `data.shape` and equals to 1 in `shape`.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    shape : Union[Tuple[PrimExprLike], relax.Expr]
        The shape to collapse to.

    Returns
    -------
    result : relax.Expr
        The result tensor of the given shape after summation.
    """
    if isinstance(shape, tuple | list):
        shape = ShapeExpr(shape)
    return _Call(
        "relax.collapse_sum_to",
        [data, shape],
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.RepeatAttrs")
class RepeatAttrs(Attrs):
    """Attributes for repeat operator"""


def repeat(
    data: Expr,
    repeats: int,
    axis: int | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Repeats elements of an array.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    repeats : int
        The number of repetitions.

    axis: Optional[int]
        The axis along which to repeat values. The negative numbers are interpreted
        counting from the backward. By default, use the flattened input array, and
        return a flat output array.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = R.const([[1, 2], [3, 4]])
        lv1 = R.repeat(x, repeats=2) # lv1 == [1, 1, 2, 2, 3, 3, 4, 4]
        lv2 = R.repeat(x, repeats=2, axis=1) # lv2 == [[1., 1., 2., 2.],
                                             #         [3., 3., 4., 4.]]
    """
    return _Call(
        "relax.repeat",
        [data],
        attrs=_make_attrs("relax.attrs.RepeatAttrs", repeats=repeats, axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.TileAttrs")
class TileAttrs(Attrs):
    """Attributes for tile operator"""


def tile(
    data: Expr,
    repeats: int | tuple[int] | list[int],
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Construct an array by repeating data the number of times given by repeats.

    If repeats has length l, and data has dimension d, the result will have dimension of max(l, d).

    If d < l, data is promoted to be l-dimensional by prepending new axes. So a shape (3,) Tensor is
    promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication. If this is not
    the desired behavior, promote data to d-dimensions manually before calling this function.

    If d > l, reps is promoted to length d by pre-pending 1's to it. Thus for a data of shape
    (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2).

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    repeats : Union[int, Tuple[int], List[int]]
        The number of repetitions of data along each axis.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = R.const([[1, 2], [3, 4]])
        lv1 = R.tile(x, reps=(2, 3)) # lv1 = [[1., 2., 1., 2., 1., 2.],
                                     #        [3., 4., 3., 4., 3., 4.],
                                     #        [1., 2., 1., 2., 1., 2.],
                                     #        [3., 4., 3., 4., 3., 4.]]
        lv2 = R.tile(x, reps=2) # lv2 = [[1., 2., 1., 2.],
                                #        [3., 4., 3., 4.]]
    """
    if isinstance(repeats, int):
        repeats = [repeats]
    return _Call(
        "relax.tile",
        [data],
        attrs=_make_attrs("relax.attrs.TileAttrs", repeats=repeats),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.FlipAttrs")
class FlipAttrs(Attrs):
    """Attributes for flip operator"""


def flip(data, axis, *, ty=None, loc: Location = UNKNOWN_LOC):
    """Reverses the order of elements along given axis while preserving array shape.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    axis: int
        The axis along which to flip over.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        x = [[1., 2.], [3., 4.]]
        relax.flip(x, axis=0) = [[3., 4.], [1., 2.]]

        relax.flip(x, axis=1) = [[2., 1.], [4., 3.]]
    """
    return _Call(
        "relax.flip",
        [data],
        attrs=_make_attrs("relax.attrs.FlipAttrs", axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.ReverseSequenceAttrs")
class ReverseSequenceAttrs(Attrs):
    """Attributes for reverse_sequence operator"""


def reverse_sequence(
    data: Expr,
    seq_lengths: Expr,
    seq_axis: int = 1,
    batch_axis: int = 0,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Reverses variable length slices.

    Parameters
    ----------
    data : relax.Expr
        The input tensor.

    seq_lengths : relax.Expr
        A 1-D tensor containing sequence lengths for each batch.

    seq_axis : int
        The axis along which to reverse variable length slices.

    batch_axis : int
        The axis that indexes the batch.

    Returns
    -------
    ret : relax.Expr
        The computed result.
    """
    return _Call(
        "relax.reverse_sequence",
        [data, seq_lengths],
        attrs=_make_attrs(
            "relax.attrs.ReverseSequenceAttrs", seq_axis=seq_axis, batch_axis=batch_axis
        ),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.GatherElementsAttrs")
class GatherElementsAttrs(Attrs):
    """Attributes for gather_elements operator"""


def gather_elements(
    data: Expr, indices: Expr, axis: int = 0, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Gather elements from data according to indices along the specified axis.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    indices : relax.Expr
        The indices tensor, must have integer type.

    axis : int
        The axis along which to index. Default is 0.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        data = [[1, 2], [3, 4]]
        indices = [[0, 0], [1, 0]]
        axis = 1
        output = [[1, 1], [4, 3]]

        data = [[1, 2, 3], [4, 5, 6]]
        indices = [[1, 1, 1]]
        axis = 0
        output = [[4, 5, 6]]
    """
    return _Call(
        "relax.gather_elements",
        [data, indices],
        attrs=_make_attrs("relax.attrs.GatherElementsAttrs", axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.GatherNDAttrs")
class GatherNDAttrs(Attrs):
    """Attributes for gather_nd operator"""


def gather_nd(
    data: Expr, indices: Expr, batch_dims: int = 0, *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Update data at positions defined by indices with values in updates.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    indices : relax.Expr
        The indices tensor, must have integer type.

    batch_dims : int
        The number of batch dimensions. Default is 0.

    Returns
    -------
    ret : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        batch_dims = 0
        data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
        indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
        output  = [0,3]           # output_shape  = [2]

        batch_dims = 1
        data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
        indices = [[1],[0]]                     # indices_shape = [2, 1]
        output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]

    """
    return _Call(
        "relax.gather_nd",
        [data, indices],
        attrs=_make_attrs("relax.attrs.GatherNDAttrs", batch_dims=batch_dims),
        loc=loc,
        ty=ty,
    )  # type: ignore


def index_tensor(
    data: Expr, indices: Expr | list[Expr], *, ty=None, loc: Location = UNKNOWN_LOC
) -> Expr:
    """Advanced-tensor indexing (NumPy/PyTorch-style).

    Given k index tensors ``indices = (I0, I1, …, Ik-1)`` this
    operator selects elements from ``data`` as if one had written
    ``data[I0, I1, …, Ik-1]`` in NumPy/PyTorch:

    All index tensors must have an integer dtype.

    Their shapes are broadcast together to a common shape ``B`` in
    the usual NumPy way.

    The result shape is ``B + data.shape[k:]`` (i.e. the broadcast
    shape followed by the remaining axes of ``data`` that are *not*
    indexed).

    At compile-time Relax checks that the number of index tensors
    ``k`` does not exceed ``data.ndim``, that the dtypes are integer,
    and that the shapes are consitent (broadcast-compatible).

    Parameters
    ----------
    data : relax.Expr
        The input tensor to be indexed.

    indices : Union[relax.Expr, List[relax.Expr]]
        A Tuple expression containing the index tensors,
        or a Python ``list`` / ``tuple`` that will be promoted to a
        tuple expression automatically. Each tensor must have an
        integer dtype.

    Returns
    -------
    result : relax.Expr
        The tensor obtained after advanced indexing.  Its dtype equals
        ``data.dtype``

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import tvm.relax as R

        x   = R.const(np.arange(9).reshape(3, 3).astype("float32"))
        row = R.const(np.array([0, 2]))        # shape (2,)
        col = R.const(np.array([1, 0]))        # shape (2,)

        y = R.index_tensor(x, [row, col])
        # y.shape == (2,) ;  y == [1., 6.]

        # Broadcasting: row : (2,1), col : (1,3)  →  B = (2,3)
        row = R.const(np.array([[0],[1]]))
        col = R.const(np.array([[0,1,2]]))
        z = R.index_tensor(x, [row, col])
        # z.shape == (2,3)

    """
    if isinstance(indices, list | tuple):
        indices = RxTuple(indices)
    return _Call(
        "relax.index_tensor",
        [data, indices],
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.IndexPutAttrs")
class IndexPutAttrs(Attrs):
    """Attributes for index_put operator"""


def index_put(
    data: Expr,
    indices: Expr | tuple[Expr],
    values: Expr,
    accumulate: bool = False,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """This operation updates values in `data` at positions
    specified by `indices` with corresponding values from `values`. The `indices` is a tuple
    of tensors where each tensor corresponds to a dimension in `data`.
    When `accumulate` is True, the operation performs accumulation (addition) rather than
    replacement. The `reduction` parameter allows specifying different reduction operations.
    Parameters
    ----------
    data : relax.Expr
        The input tensor to be modified
    indices : Union[Expr, Tuple[Expr]]
        Tuple of index tensors (one for each dimension) specifying positions to update
    values : relax.Expr
        Values to place at the specified indices
    accumulate : bool
        Whether to accumulate (add) values rather than replace (default: False)

    Returns
    -------
    result : relax.Expr
        A new tensor with the same shape as data but with specified positions updated
    Examples
    --------
    .. code-block:: python

        # inputs
        data = torch.zeros(3, 3)
        indices = (torch.tensor([0, 2]), torch.tensor([1, 1]))
        values = torch.tensor([1.0, 2.0])
        # output
        output = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
        # with accumulate=True
        output = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    """
    if isinstance(indices, list | tuple):
        indices = RxTuple(indices)
    return _Call(
        "relax.index_put",
        [data, indices, values],
        attrs=_make_attrs("relax.attrs.IndexPutAttrs", accumulate=accumulate),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.MeshgridAttrs")
class MeshgridAttrs(Attrs):
    """Attributes for meshgrid operator"""


def meshgrid(
    tensors: Expr | list[Expr],
    indexing: str | None = "ij",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Generate coordinate grids from input tensors.

    Parameters
    ----------
    tensors : Union[relax.Expr, List[relax.Expr]]
        An Expr in Tuple type, containing 1D tensors (or scalars promoted to 1D)
        to generate coordinate grids from, or a list of such tensors.

    indexing : Optional[str]
        The indexing mode, either "ij" (matrix indexing) or "xy" (Cartesian indexing).
        Defaults to "ij".

    Returns
    -------
    result : relax.Expr
        A Tuple of tensors representing the coordinate grids.
    """
    if isinstance(tensors, list | tuple):
        tensors = RxTuple(tensors)
    return _Call(
        "relax.meshgrid",
        [tensors],
        attrs=_make_attrs("relax.attrs.MeshgridAttrs", indexing=indexing),
        loc=loc,
        ty=ty,
    )


@tvm_ffi.register_object("relax.attrs.ScatterElementsAttrs")
class ScatterElementsAttrs(Attrs):
    """Attributes for scatter_elements operator"""


def scatter_elements(
    data: Expr,
    indices: Expr,
    updates: Expr,
    axis: int = 0,
    reduction: str = "update",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    """ONNX style scatter elements. This operation updates its value in `data` to values
    specified by `updates` at specific index positions specified by `indices`.
    For example, in 2D tensor, the update corresponding to the [i][j] entry is performed
    as below:

    .. code-block::

        output[indices[i][j]][j] = updates[i][j] if axis = 0
        output[i][indices[i][j]] = updates[i][j] if axis = 1

    When the `reduction` is set to some reduction function `f`, the update corresponding to
    [i][j] entry is performed as below:

    .. code-block::

        output[indices[i][j]][j] += f(output[indices[i][j]][j], updates[i][j]) if axis = 0
        output[i][indices[i][j]] += f(output[i][indices[i][j]], updates[i][j]) if axis = 1

    Where `f` is update, add, mul, mean, max, min.

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    indices: relax.Expr
        The index positions to update in `data`.

    updates: relax.Expr
        Values to replace to.

    axis: int
        Axis to scatter on.

    reduction: str
        Type of reduction to apply: update, add, mul, mean, max, min.
        It is "update" by default.

    Returns
    -------
    result : relax.Expr
        The result has the same size as data, and the same shape as data

    Examples
    --------
    .. code-block:: python

       # inputs
       data = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        indices = [
            [1, 0, 2],
            [0, 2, 1],
        ]
        updates = [
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 2.2],
        ]
        axis = 0
        reduction = "update"

        # output P
        output = [
            [2.0, 1.1, 0.0]
            [1.0, 0.0, 2.2]
            [0.0, 2.1, 1.2]
        ]

    """
    return _Call(
        "relax.scatter_elements",
        [data, indices, updates],
        attrs=_make_attrs("relax.attrs.ScatterElementsAttrs", axis=axis, reduction=reduction),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.ScatterNDAttrs")
class ScatterNDAttrs(Attrs):
    """Attributes for scatter_nd operator"""


def scatter_nd(
    data: Expr,
    indices: Expr,
    updates: Expr,
    reduction: str = "update",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Scatter updates into an array according to indices.

    Parameters
    ----------
    data: relax.Expr
        The input data to be updated.

    indices: relax.Expr
        The index positions to update in `data`.

    updates: relax.Expr
        Values to replace to.

    reduction: str
        Type of reduction to apply: update, add, mul, max, min.
        It is "update" by default.

    Returns
    -------
    result : relax.Expr
        The result has the same shape as data.

    Examples
    --------
    .. code-block:: python

       # inputs
       data = [1, 2, 3, 4, 5, 6, 7, 8]
       indices = [[4], [3], [1], [7]]
       updates = [9, 10, 11, 12]

       # output
       output = [1, 11, 3, 10, 9, 6, 7, 12]

    """
    return _Call(
        "relax.scatter_nd",
        [data, indices, updates],
        attrs=_make_attrs("relax.attrs.ScatterNDAttrs", reduction=reduction),
        loc=loc,
        ty=ty,
    )  # type: ignore


@tvm_ffi.register_object("relax.attrs.SliceScatterAttrs")
class SliceScatterAttrs(Attrs):
    """Attributes for slice_scatter operator"""


def slice_scatter(
    input_tensor: Expr,
    src: Expr,
    start,
    end,
    step,
    axis=0,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
):
    """Embeds the values of the src tensor into input at the given dimension.

    Parameters
    ----------
    input_tensor: relax.Expr
        The input tensor to be updated.

    src: relax.Expr
        The tensor to embed into input.

    axis: int
        The dimension to insert the slice into.

    start:
        The start index of where to insert the slice.

    end:
        The end index of where to insert the slice.

    step:
        The how many elements to skip in.

    Returns
    -------
    result : relax.Expr
        The computed result tensor with the same shape as `data`.

    """
    if not is_prim_expr(start):
        start = prim_value(start)
    if not is_prim_expr(end):
        end = prim_value(end)
    if not is_prim_expr(step):
        step = prim_value(step)
    return _Call(
        "relax.slice_scatter",
        [input_tensor, src, start, end, step],
        attrs=_make_attrs("relax.attrs.SliceScatterAttrs", axis=axis),
        loc=loc,
        ty=ty,
    )


@tvm_ffi.register_object("relax.attrs.OneHotAttrs")
class OneHotAttrs(Attrs):
    """Attributes for one_hot operator"""


def one_hot(
    indices: Expr,
    on_value: int | float | Expr,
    off_value: int | float | Expr,
    depth: int,
    axis: int = -1,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Returns a one-hot tensor.

    Parameters
    ----------
    indices : relax.Expr
        The indices to set to `on_value`.

    on_value : int | float | Expr
        The value to fill at `indices`.

    off_value : int | float | Expr
        The value to fill at other locations.

    depth : int
        The depth of the one-hot dimension.

    axis : int, optional
        The axis to fill. Default is -1 which adds a new dimension at the end.

    Returns
    -------
    result : relax.Expr
        The computed result.

    Examples
    --------
    .. code-block:: python

        indices = [0, 1, 2]
        depth = 3
        on_value = 1
        off_value = 0

        one_hot(indices, on_value, off_value, depth) =
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
    """
    on_value = prim_value(on_value)
    off_value = prim_value(off_value)
    if depth <= 0:
        raise RuntimeError("one_hot: depth must be positive")
    return _Call(
        "relax.one_hot",
        [indices, on_value, off_value],
        attrs=_make_attrs("relax.attrs.OneHotAttrs", depth=depth, axis=axis),
        loc=loc,
        ty=ty,
    )  # type: ignore
