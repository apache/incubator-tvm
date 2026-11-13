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
# pylint: disable=import-outside-toplevel, redefined-builtin, unused-argument
"""Set operators."""

import numpy as np  # type: ignore
import tvm_ffi

import tvm_tirx
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.location import UNKNOWN_LOC, Location

from ..expr import Expr, prim_value


def unique(
    x: Expr,
    sorted: bool | Expr = True,
    return_index: bool | Expr = False,
    return_inverse: bool | Expr = False,
    return_counts: bool | Expr = False,
    axis: int | Expr | None = None,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Expr:
    """Find the unique elements in a given tensor.
    In addition, it optionally returns
    - the indices of the input tensor that give the unique values;
    - the indices of the unique tensor that reconstruct the input tensor;
    - the number of times each unique value comes up in the input tensor.

    Parameters
    ----------
    x : relax.Expr
        The input tensor.

    sorted : Union[bool, Expr]
        Whether to sort the unique elements in ascending order before
        returning as output.

    return_index : Union[bool, Expr]
        Whether to return an additional tensor with indices for where elements in
        the unique tensor come from the original input.

    return_inverse : Union[bool, Expr]
        Whether to return an additional tensor with indices for where elements in
        the original input ended up in the returned unique list.

    return_counts : Union[bool, Expr]
        Whether to return an additional tensor with counts of each unique elements.

    axis : Optional
        The dimension to apply unique.
        If not specified, the unique values of the flattened input are returned.

    Returns
    -------
    ret : relax.Expr
        The created relax call with
    """

    if isinstance(sorted, bool):
        sorted = prim_value(sorted)
    if isinstance(return_index, bool):
        return_index = prim_value(return_index)
    if isinstance(return_inverse, bool):
        return_inverse = prim_value(return_inverse)
    if isinstance(return_counts, bool):
        return_counts = prim_value(return_counts)
    if axis is not None and isinstance(axis, int):
        axis = prim_value(axis)
    return _Call(
        "relax.unique",
        [x, sorted, return_index, return_inverse, return_counts, *([] if axis is None else [axis])],
        loc=loc,
        ty=ty,
    )


@tvm_ffi.register_global_func("relax.run.unique")
def numpy_unique(
    x: tvm_tirx.runtime.tensor,
    sorted: int,
    return_index: int,
    return_inverse: int,
    return_counts: int,
    axis: int | None = None,
) -> tvm_tirx.runtime.tensor:
    """Returns the unique elements of the input tensor.

    Uses numpy.unique to compute unique elements.
    """
    import builtins

    x_numpy = x.numpy()

    # Call numpy.unique with all the requested return flags
    result = np.unique(
        x_numpy,
        return_index=bool(return_index),
        return_inverse=bool(return_inverse),
        return_counts=bool(return_counts),
        axis=axis,
    )

    # If no optional outputs requested, result is just the unique values
    if not bool(return_index) and not bool(return_inverse) and not bool(return_counts):
        unique_values = result
        if not sorted:
            indices = np.unique(x_numpy, return_index=True, axis=axis)[1]
            unique_values = np.take(x_numpy, builtins.sorted(indices), axis=axis)
        return tvm_tirx.runtime.tensor(unique_values)

    # Otherwise, numpy returns a tuple
    unique_values = result[0]
    output_list = []
    result_idx = 1

    # Handle sorting for unique values
    if not sorted and bool(return_index):
        # Get the indices from numpy result
        indices = result[result_idx]
        result_idx += 1
        # Sort indices to get original order
        sort_order = np.argsort(indices)
        unique_values = np.take(unique_values, sort_order, axis=axis)
        indices = np.sort(indices)
        output_list.append(tvm_tirx.runtime.tensor(unique_values))
        output_list.append(tvm_tirx.runtime.tensor(indices))
    elif not sorted:
        # Need to get indices to reorder
        _, indices = np.unique(x_numpy, return_index=True, axis=axis)
        sort_order = np.argsort(indices)
        unique_values = np.take(unique_values, sort_order, axis=axis)
        output_list.append(tvm_tirx.runtime.tensor(unique_values))
        if bool(return_index):
            indices_from_result = result[result_idx]
            result_idx += 1
            output_list.append(tvm_tirx.runtime.tensor(np.sort(indices_from_result)))
    else:
        # Sorted case
        output_list.append(tvm_tirx.runtime.tensor(unique_values))
        if bool(return_index):
            output_list.append(tvm_tirx.runtime.tensor(result[result_idx]))
            result_idx += 1

    if bool(return_inverse):
        inverse_indices = result[result_idx]
        if not sorted:
            # Need to remap inverse indices to match reordered unique values
            _, orig_indices = np.unique(x_numpy, return_index=True, axis=axis)
            sort_order = np.argsort(orig_indices)
            inverse_mapping = np.empty_like(sort_order)
            inverse_mapping[sort_order] = np.arange(len(sort_order))
            inverse_indices = inverse_mapping[inverse_indices]
        # ONNX spec: inverse_indices is always 1D
        # When axis is None, it has length X.size (flattened)
        # When axis is specified, it has length X.shape[axis]
        # numpy.unique already returns 1D inverse_indices, so no reshaping needed
        output_list.append(tvm_tirx.runtime.tensor(inverse_indices))
        result_idx += 1

    if bool(return_counts):
        counts = result[result_idx]
        if not sorted:
            # Reorder counts to match reordered unique values
            _, orig_indices = np.unique(x_numpy, return_index=True, axis=axis)
            sort_order = np.argsort(orig_indices)
            counts = counts[sort_order]
        output_list.append(tvm_tirx.runtime.tensor(counts))

    return tuple(output_list)


def nonzero(x: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """Find the indices of elements of a tensor that are non-zero.

    Parameters
    ----------
    x : relax.Expr
        The input data tensor.

    Returns
    -------
    result : relax.Expr
        A 2-D tensor containing indices of non-zero elements.

    Note
    ----
    This function is equivalent to `onnx.nonzero`.

    Examples
    --------

    .. code-block:: python

        x = [[0, 1],
             [2, 0]]
        nonzero(x) = [[0, 1],
                      [1, 0]]

    """
    return _Call("relax.nonzero", [x], loc=loc, ty=ty)  # type: ignore


@tvm_ffi.register_global_func("relax.run.nonzero")
def numpy_nonzero(x: tvm_tirx.runtime.tensor) -> tvm_tirx.runtime.tensor:
    np_result = np.atleast_1d(x.numpy()).nonzero()
    return tvm_tirx.runtime.tensor(np.stack(np_result, axis=0))
