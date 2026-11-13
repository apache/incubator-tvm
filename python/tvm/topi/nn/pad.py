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
"""Pad the data by constant value"""

import tvm
import tvm_tirx
from tvm import te

from .. import tag
from ..utils import equal_const_int


def get_padded_shape(data, pad_before, pad_after=None):
    """
    Calculates the output shape of a tensor after applying padding.

    Args:
        data (tvm.te.Tensor): The input tensor to which padding is applied.
        pad_before : list / tuple of n ints
            Pad width on each dimension to pad the before the axis begin.
        pad_after : list / tuple of n ints, optional
            Pad width each dimension to pad the after the axis end.

    Raises:
        ValueError: If `pad_before` or `pad_after` lengths mismatch with `data` dimensions.

    Returns:
        tuple: A tuple representing the padded shape of the tensor.
    """
    n = data.ndim
    pad_after = pad_after if pad_after else pad_before

    if len(pad_before) != n:
        raise ValueError(f"pad_before length {len(pad_before)} != input dims {n}")
    if len(pad_after) != n:
        raise ValueError(f"pad_after length {len(pad_after)} != input dims {n}")

    ana = tvm_tirx.sym.Analyzer()
    out_shape = tuple(ana.simplify(data.shape[i] + pad_before[i] + pad_after[i]) for i in range(n))

    return out_shape


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",pad")
def pad(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput", attrs=None):
    """Pad Input with using pad values.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D input, can be any layout.

    pad_before : list / tuple of n ints
        Pad width on each dimension to pad the before the axis begin.

    pad_after : list / tuple of n ints, optional
        Pad width each dimension to pad the after the axis end.

    pad_value : float, optional
        The value to be padded.

    name : str, optional
        The name prefix operators generated

    Returns
    -------
    Output : tvm.te.Tensor
        n-D, the same layout as Input.
    """
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(f"Input dimension and pad_before dismatch : {n} vs {len(pad_before)}")
    if len(pad_after) != n:
        raise ValueError(f"Input dimension and pad_after dismatch : {n} vs {len(pad_after)}")
    ana = tvm_tirx.sym.Analyzer()
    dshape = []
    for dim in data.shape:
        dshape.append(dim)
    out_shape = tuple(ana.simplify(dshape[i] + pad_before[i] + pad_after[i]) for i in range(n))
    pad_value = (
        pad_value
        if tvm_tirx.ir.is_prim_expr(pad_value)
        else tvm_tirx.tirx.const(pad_value, data.dtype)
    )

    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
            else:
                index_tuple.append(indices[i] - pad_before[i])
                not_zero.append(indices[i] >= pad_before[i])
                not_zero.append(indices[i] < data.shape[i] + pad_before[i])
        if not_zero:
            not_zero = tvm_tirx.tirx.all(*not_zero)
            return tvm_tirx.tirx.if_then_else(not_zero, data(*index_tuple), pad_value)
        return data(*index_tuple)

    return te.compute(out_shape, _pad, name=name, attrs=attrs)


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",pad")
def mirror_pad(data, pad_before, pad_after=None, mode="SYMMETRIC", name="MirrorPadInput"):
    """Pad Input with mirroring either symmetric or reflected.

    Parameters
    ----------
    data : tvm.te.Tensor
        n-D input, can be any layout.

    pad_before : list / tuple of n ints
        Pad width on each dimension to pad the before the axis begin.

    pad_after : list / tuple of n ints, optional
        Pad width each dimension to pad the after the axis end.

    mode: str, optional
        Type of mirror padding to apply. Must be SYMMETRIC or REFLECT

    name : str, optional
        The name prefix operators generated

    Returns
    -------
    Output : tvm.te.Tensor
        n-D, the same layout as Input.
    """
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(f"Input dimension and pad_before dismatch : {n} vs {len(pad_before)}")
    if len(pad_after) != n:
        raise ValueError(f"Input dimension and pad_after dismatch : {n} vs {len(pad_after)}")
    ana = tvm_tirx.sym.Analyzer()
    out_shape = tuple(ana.simplify(data.shape[i] + pad_before[i] + pad_after[i]) for i in range(n))
    assert mode in ("SYMMETRIC", "REFLECT")
    mode = int(mode == "SYMMETRIC")

    def _pad(*indices):
        index_tuple = []
        above = []
        below = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                index_tuple.append(indices[i])
                above.append(False)
                below.append(False)
            else:
                index_tuple.append(indices[i] - pad_before[i])
                above.append(indices[i] >= data.shape[i] + pad_before[i])
                below.append(indices[i] < pad_before[i])
        mapped_tuple = []
        for i, axis in enumerate(index_tuple):
            mapped_axis = tvm_tirx.tirx.if_then_else(below[i], -axis - mode, axis)
            mapped_axis = tvm_tirx.tirx.if_then_else(
                above[i], (2 * (data.shape[i] - 1)) - axis + mode, mapped_axis
            )
            mapped_tuple.append(mapped_axis)
        return data(*mapped_tuple)

    return te.compute(out_shape, _pad, name=name)


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",pad")
def reflect_pad(data, pad_before, pad_after=None, name="ReflectPadInput"):
    """
    Apply reflect padding to the input tensor.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input tensor.

    pad_before : List[int]
        Amount to pad before each dimension.

    pad_after : List[int], optional
        Amount to pad after each dimension. If None, defaults to pad_before.

    name : str
        Name of the resulting tensor.

    Returns
    -------
    out : tvm.te.Tensor
        Reflect-padded tensor.
    """
    out_shape = get_padded_shape(data, pad_before, pad_after)

    def _pad(*indices):
        index_tuple = []
        for i in range(data.ndim):
            idx = indices[i]
            size = data.shape[i]
            before = pad_before[i]

            orig_idx = idx - before

            # Branchless reflect-101 boundary index. This is bit-identical to the
            # nested if_then_else form (-orig_idx below 0, (2*size-2)-orig_idx at or
            # above size, orig_idx otherwise) over the valid reflect-pad domain, but
            # lowers to plain integer arithmetic instead of per-element branches.
            m = size - 1
            reflected_idx = m - tvm_tirx.tirx.abs(m - tvm_tirx.tirx.abs(orig_idx))
            index_tuple.append(reflected_idx)
        return data(*index_tuple)

    return te.compute(out_shape, _pad, name=name)


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",pad")
def replicate_pad(data, pad_before, pad_after=None, name="ReplicatePadInput"):
    """
    Apply replicate padding (edge padding) to the input tensor.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input tensor.

    pad_before : List[int]
        Amount to pad before each dimension.

    pad_after : List[int], optional
        Amount to pad after each dimension. If None, defaults to pad_before.

    name : str
        Name of the resulting tensor.

    Returns
    -------
    out : tvm.te.Tensor
        Replicate-padded tensor.
    """
    out_shape = get_padded_shape(data, pad_before, pad_after)

    def _pad(*indices):
        index_tuple = []
        for i in range(data.ndim):
            idx = indices[i]
            size = data.shape[i]
            before = pad_before[i]

            orig_idx = idx - before
            # Branchless edge clamp. This is bit-identical to the nested
            # if_then_else form (0 below 0, size-1 at or above size, orig_idx
            # otherwise) but lowers to min/max instead of per-element branches.
            clamped_idx = tvm_tirx.tirx.max(
                tvm_tirx.tirx.const(0, "int32"), tvm_tirx.tirx.min(size - 1, orig_idx)
            )
            index_tuple.append(clamped_idx)
        return data(*index_tuple)

    return te.compute(out_shape, _pad, name=name)


@tvm.te.tag_scope(tag=tag.INJECTIVE + ",pad")
def circular_pad(data, pad_before, pad_after=None, name="CircularPadInput"):
    """
    Apply circular padding (wrap around) to the input tensor.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input tensor.

    pad_before : List[int]
        Amount to pad before each dimension.

    pad_after : List[int], optional
        Amount to pad after each dimension. If None, defaults to pad_before.

    name : str
        Name of the resulting tensor.

    Returns
    -------
    out : tvm.te.Tensor
        Circular-padded tensor.
    """
    out_shape = get_padded_shape(data, pad_before, pad_after)

    def _pad(*indices):
        index_tuple = []
        for i in range(data.ndim):
            idx = indices[i]
            size = data.shape[i]
            before = pad_before[i]

            orig_idx = idx - before
            wrapped_idx = tvm_tirx.tirx.indexmod(orig_idx + size, size)
            index_tuple.append(wrapped_idx)
        return data(*index_tuple)

    return te.compute(out_shape, _pad, name=name)
