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
# pylint: disable=invalid-name
"""Legalization functions for DLTensor inspection."""

import enum

from tvm.s_tir import script as Ts
from tvm_tirx.ir import Call
from tvm_tirx.script import tirx as T

from ... import op
from ...block_builder import BlockBuilder
from ...expr import Expr
from .common import register_legalize


class TVMStructFieldKind(enum.IntEnum):
    """Equivalent to tvm::tirx::TVMStructFieldKind

    This does not use `enum.auto()` to define the values, because
    `enum.auto()` starts from 1, and this must match the C++
    definition which starts from 0.
    """

    kDLTensorAddr = 0
    kDLTensorData = 1
    kDLTensorShape = 2
    kDLTensorStrides = 3
    kDLTensorNDim = 4
    kDLTensorTypeCode = 5
    kDLTensorTypeBits = 6
    kDLTensorTypeLanes = 7
    kDLTensorByteOffset = 8
    kDLTensorDeviceId = 9
    kDLTensorDeviceType = 10
    kDLTensorKindBound_ = 11
    kTVMValueContent = 12
    kTVMValueKindBound_ = 13


@register_legalize("relax.inspect.tensor_stride_i")
def _tensor_stride_i(bb: BlockBuilder, call: Call) -> Expr:
    @Ts.function(private=True)
    def _get_tensor_stride_i(dlpack_handle: T.handle, axis: T.int64) -> T.int64:
        T.func_attr({"tirx.is_host_func": True, "tirx.is_scheduled": True})
        assert T.int64(0) <= axis, "Specified axis may not be negative"
        ndim: T.let[T.int32] = T.abi_field_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kDLTensorNDim), ty="int32"
        )
        assert axis < T.Cast("int64", ndim), (
            "Specified axis may not be larger than the tensor's dimensionality"
        )
        stride_ptr: T.let[T.handle("int64")] = T.abi_field_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kDLTensorStrides), ty=T.handle("int64").ty
        )

        if T.isnullptr(stride_ptr):
            shape_ptr: T.let[T.handle("int64")] = T.abi_field_get(
                dlpack_handle, 0, int(TVMStructFieldKind.kDLTensorShape), ty=T.handle("int64").ty
            )
            shape = T.decl_tensor(shape_ptr, ty_args=[T.Tensor(ndim, "int64", data=shape_ptr)])

            product = T.alloc_tensor(ty_args=[T.Tensor([], "int64")])
            product[()] = 1

            # TODO(Lunderberg): Add a TIR lowering pass to allow
            # ranges to start somewhere other than zero.  This loop
            # could then iterate on `range(axis+1, ndim)`.
            for dim_offset in range(ndim - (axis + 1)):
                dim: T.let[T.int64] = dim_offset + (axis + 1)
                product[()] = product[()] * shape[dim]

            return product[()]
        else:
            strides = T.decl_tensor(stride_ptr, ty_args=[T.Tensor(ndim, "int64", data=stride_ptr)])
            stride: T.let[T.int64] = strides[axis]
            return stride

    gvar = bb.add_func(_get_tensor_stride_i, "_get_tensor_stride_i")
    return op.call_tir_packed(gvar, tuple(call.args))


@register_legalize("relax.inspect.tensor_byte_offset")
def _tensor_byte_offset(bb: BlockBuilder, call: Call) -> Expr:
    @Ts.function(private=True)
    def _get_tensor_byte_offset(dlpack_handle: T.handle) -> T.uint64:
        T.func_attr({"tirx.is_host_func": True, "tirx.is_scheduled": True})
        byte_offset: T.let[T.uint64] = T.abi_field_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kDLTensorByteOffset), ty="uint64"
        )
        return byte_offset

    gvar = bb.add_func(_get_tensor_byte_offset, "_get_tensor_byte_offset")
    return op.call_tir_packed(gvar, tuple(call.args))


@register_legalize("relax.inspect.tensor_elem_offset")
def _tensor_elem_offset(bb: BlockBuilder, call: Call) -> Expr:
    @Ts.function(private=True)
    def _get_tensor_elem_offset(dlpack_handle: T.handle) -> T.uint64:
        T.func_attr({"tirx.is_host_func": True, "tirx.is_scheduled": True})
        byte_offset: T.let[T.uint64] = T.abi_field_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kDLTensorByteOffset), ty="uint64"
        )
        scalar_bits: T.let[T.uint8] = T.abi_field_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kDLTensorTypeBits), ty="uint8"
        )
        lanes: T.let[T.uint16] = T.abi_field_get(
            dlpack_handle, 0, int(TVMStructFieldKind.kDLTensorTypeLanes), ty="uint16"
        )
        bytes_per_element: T.let[T.uint64] = T.ceildiv(
            scalar_bits.astype("uint64") * lanes.astype("uint64"), 8
        )
        elem_offset: T.let[T.uint64] = byte_offset // bytes_per_element
        return elem_offset

    gvar = bb.add_func(_get_tensor_elem_offset, "_get_tensor_elem_offset")
    return op.call_tir_packed(gvar, tuple(call.args))


@register_legalize("relax.size")
def _size(_bb: BlockBuilder, call: Call) -> Expr:
    return op.prod(op.shape_to_tensor(op.shape_of(call.args[0])))
