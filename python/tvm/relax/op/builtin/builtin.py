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
"""The builtin Relax operators."""

from tvm_tirx.ir import Call, DataTypeImm, GenericConst, StringImm
from tvm_tirx.ir import Call as _Call
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.runtime import MemoryScope

from ...expr import Expr, prim_value
from ...utils import convert_to_expr


def alloc_tensor(
    shape: Expr,
    dtype: str | Expr,
    runtime_device_index: int | Expr,
    storage_scope: str | MemoryScope | Expr = "global",
    *,
    ty_args=None,
    ty=None,
    loc: Location = UNKNOWN_LOC,
) -> Call:
    """Construct a Call to allocate a tensor with specific shape, dtype, runtime_device_index.

    Parameters
    ----------
    shape : Expr
        The shape of the tensor to be allocated.

    dtype : Union[str, Expr]
        The datatype of the tensor to be allocated.

    runtime_device_index : Union[int, Expr]
        The device index indicating on which device the tensor is to be allocated at runtime.
        Index -1 is reserved for the host device.

    storage_scope : Union[str, MemoryScope, Expr]
        The storage scope to allocate the storage to.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated tensor.
    """
    if not isinstance(shape, Expr):
        shape = convert_to_expr(shape)
    if isinstance(dtype, str | GenericConst):
        dtype = DataTypeImm(dtype.value if isinstance(dtype, GenericConst) else dtype)
    if isinstance(runtime_device_index, int):
        runtime_device_index = prim_value(runtime_device_index)
    if isinstance(storage_scope, MemoryScope):
        storage_scope = storage_scope.name
    if isinstance(storage_scope, str):
        storage_scope = StringImm(storage_scope)
    if not isinstance(storage_scope, StringImm):
        raise ValueError(
            "relax.builtin.alloc_tensor expects string as the storage scope, "
            f"but {storage_scope} is got."
        )

    return _Call(
        "relax.builtin.alloc_tensor",
        [shape, dtype, runtime_device_index, storage_scope],
        ty_args=ty_args,
        loc=loc,
        ty=ty,
    )  # type: ignore


def stop_lift_params(x: Expr, *, ty_args=None, ty=None, loc: Location = UNKNOWN_LOC) -> Expr:
    """
    An indicator that the consumers of input tensor should not be
    lifted to transform_params function

    Parameters
    ----------
    x: relax.Expr
        The input data

    Returns
    -------
    result : relax.Expr
        The result tensor that is the same as input tensor
    """
    return _Call(
        "relax.builtin.stop_lift_params",
        [x],
        ty_args=ty_args,
        loc=loc,
        ty=ty,
    )  # type: ignore
