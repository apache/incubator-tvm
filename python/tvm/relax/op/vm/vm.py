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
"""Relax vm primitives."""

from tvm_tirx.ir import Call, DataTypeImm, GenericConst, StringImm
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.runtime import MemoryScope

from ...expr import Expr, Tuple, prim_value
from ...utils import convert_to_expr


def alloc_storage(
    shape: Expr,
    runtime_device_index: int | Expr,
    dtype: str | Expr,
    storage_scope: str | MemoryScope | StringImm = "global",
    *,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Call:
    """Construct a Call to allocate a storage with specific size,
    runtime_device_index, and dtype.

    Parameters
    ----------
    shape : Expr
        The shape of the storage to be allocated.

    runtime_device_index : Union[int, Expr]
        The device index indicating on which device the tensor is to
        be allocated at runtime. Index -1 is reserved for the host device.

    dtype : Union[str, Expr]
        The datatype of the storage to be allocated.

    storage_scope : Union[str, MemoryScope, StringImm]
        The storage scope of the storage to allocate. Default is global.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated storage.
    """
    shape = convert_to_expr(shape)
    if isinstance(dtype, str | GenericConst):
        dtype = DataTypeImm(dtype.value if isinstance(dtype, GenericConst) else dtype)
    if isinstance(storage_scope, MemoryScope):
        storage_scope = storage_scope.name
    if isinstance(storage_scope, str):
        storage_scope = StringImm(storage_scope)
    if isinstance(runtime_device_index, int):
        runtime_device_index = prim_value(runtime_device_index)
    return _Call(
        "relax.vm.alloc_storage",
        [shape, runtime_device_index, dtype, storage_scope],
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
        "relax.vm.alloc_tensor",
        [storage, offset, shape, dtype, runtime_device_ind],
        loc=loc,
        ty=ty,
    )  # type: ignore


def kill_object(obj: Expr, *, ty=None, loc: Location = UNKNOWN_LOC) -> Call:
    """Construct a Call to set the register corresponding to the input object to
    null at runtime, in order to kill the input object.

    Parameters
    ----------
    obj : Expr
        The object to be killed.

    Returns
    -------
    result : Call
        CallObj that kills the input object.
    """
    return _Call("relax.vm.kill_object", [obj], loc=loc, ty=ty)  # type: ignore


def call_tir_dyn(func: Expr, args: Tuple, *, ty=None, loc: Location = UNKNOWN_LOC) -> Call:
    """Construct a Call to call_tir_dyn (invoke the given TIR Function)
    consisting of the input tensors and the shape of the result.

    Parameters
    ----------
    func : Expr
        An expression evaluating to a TIR Function.

    args : Tuple
        The input args, includes a list of tensors, and a ShapeExpr.

    Returns
    -------
    result : Call
        A relax Call to call_tir_dyn.
    """
    func = convert_to_expr(func)
    if isinstance(args, list | tuple):
        args = Tuple(args)

    return _Call(
        "relax.vm.call_tir_dyn",
        [func, args],
        loc=loc,
        ty=ty,
    )  # type: ignore
