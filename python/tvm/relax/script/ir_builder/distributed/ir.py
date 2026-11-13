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
# pylint: disable=redefined-builtin, wrong-import-order, no-member, invalid-name, unused-import

"""IRBuilder for distributed Relax dialect"""

from numbers import Number

import numpy as _np  # type: ignore
import tvm_ffi

import tvm_tirx
from tvm.relax.distributed import DeviceMesh, DTensorType
from tvm_tirx.ir import GenericConst
from tvm_tirx.runtime import _tensor


def const(
    value: bool | int | float | _np.ndarray | tvm_tirx.runtime.Tensor,
    ty: DTensorType,
) -> GenericConst:
    """Create a distributed constant value with the specified tensor type.

    Parameters
    ----------
    value : bool, int, float, numpy.ndarray or tvm_tirx.runtime.Tensor
        The constant value.

    ty : DTensorType
        The distributed tensor type, including its tensor dtype, device mesh
        and placement.

    Returns
    -------
    res : GenericConst
        The constant carrying the supplied distributed tensor type.

    Notes
    -----
    Python scalars and NumPy values are converted to the dtype specified by
    ``ty.tensor_ty``. The dtype is not inferred from the Python value.
    """
    ty = tvm_ffi.convert(ty)
    if not isinstance(ty, DTensorType):
        raise TypeError("ty needs to be an instance of DTensorType. ")
    dtype = str(ty.tensor_ty.dtype)
    if isinstance(value, Number | (bool | list)):
        value = _np.array(value, dtype=dtype)

    if isinstance(value, _np.ndarray | _np.generic):
        if dtype is not None:
            value = value.astype(dtype)
        value = _tensor.tensor(value)

    if not isinstance(value, _tensor.Tensor):
        raise ValueError("value has to be scalar or Tensor")

    return GenericConst(ty, value)


def _lookup_device_mesh(device_mesh_str: str) -> DeviceMesh:
    from ..parser_protocol import resolve_global_info_

    device_mesh = resolve_global_info_(device_mesh_str)
    if not isinstance(device_mesh, DeviceMesh):
        raise TypeError("The device_mesh global info must be a DeviceMesh.")
    return device_mesh


__all__ = ["const"]
