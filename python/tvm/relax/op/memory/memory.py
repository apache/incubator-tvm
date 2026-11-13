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
"""Relax memory primitives."""

from tvm_tirx.ir import Call, DataTypeImm, GenericConst, StringImm
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.runtime import MemoryScope

from ...expr import Expr, prim_value
from ...utils import convert_to_expr


def alloc_storage(
    size: Expr,
    virtual_device_index: int | Expr,
    storage_scope: str | MemoryScope | Expr,
    dtype: str | Expr,
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Call:
    """Construct a Call to allocate a storage with specific size, virtual_device_index,
    storage_scope and dtype.

    Parameters
    ----------
    size : Expr
        The size of the storage to be allocated.

    virtual_device_index : Union[int, Expr]
        The virtual device index indicating on which device the storage is to be allocated.
        Index -1 is reserved for the host device.

    storage_scope : Union[str, MemoryScope, Expr]
        The storage scope to allocate the storage to.

    dtype : Union[str, Expr]
        The datatype of the storage to be allocated.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated storage.
    """
    size = convert_to_expr(size)
    if isinstance(dtype, str | GenericConst):
        dtype = DataTypeImm(dtype.value if isinstance(dtype, GenericConst) else dtype)
    if isinstance(storage_scope, MemoryScope):
        storage_scope = storage_scope.name
    if isinstance(storage_scope, str):
        storage_scope = StringImm(storage_scope)
    if isinstance(virtual_device_index, int):
        virtual_device_index = prim_value(virtual_device_index)
    return _Call(
        "relax.memory.alloc_storage",
        [size, virtual_device_index, storage_scope, dtype],
        loc=loc,
        ty=ty,
    )  # type: ignore


def alloc_tensor(
    storage: Expr,
    offset: int | Expr,
    shape: Expr,
    dtype: str | Expr,
    runtime_device_ind: int | Expr = prim_value(0),
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Call:
    """Construct a Call to allocate a tensor on a certain storage starting from the given offset.

    Parameters
    ----------
    storage : Expr
        The storage to allocate the tensor to.

    offset : Union[int, Expr]
        The storage offset to allocate the tensor.

    shape : Expr
        The shape of the tensor to be allocated.

    dtype : Union[str, Expr]
        The datatype of the tensor to be allocated.

    runtime_device_ind: Union[int, Expr]
        The device index indicating on which device the tensor is to be
        allocated at runtime. Index -1 is reserved for the host device.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated tensor.
    """
    if isinstance(offset, int):
        offset = prim_value(offset)
    shape = convert_to_expr(shape)
    if isinstance(dtype, str | GenericConst):
        dtype = DataTypeImm(dtype.value if isinstance(dtype, GenericConst) else dtype)
    if isinstance(runtime_device_ind, int):
        runtime_device_ind = prim_value(runtime_device_ind)
    return _Call(
        "relax.memory.alloc_tensor",
        [storage, offset, shape, dtype, runtime_device_ind],
        loc=loc,
        ty=ty,
    )  # type: ignore


def kill_storage(storage: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Call:
    """Construct a Call to kill a storage.

    Parameters
    ----------
    storage : Expr
        The storage to be killed.

    Returns
    -------
    result : Call
        A relax Call to kill a storage.
    """
    return _Call(
        "relax.memory.kill_storage",
        [storage],
        loc=loc,
        ty=ty,
    )  # type: ignore


def kill_tensor(tensor: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Call:
    """Construct a Call to kill a tensor.

    Parameters
    ----------
    tensor : Expr
        The tensor to be killed.

    Returns
    -------
    result : Call
        A relax Call to kill a tensor.
    """
    return _Call(
        "relax.memory.kill_tensor",
        [tensor],
        loc=loc,
        ty=ty,
    )  # type: ignore
