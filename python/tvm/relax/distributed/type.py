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
# pylint: disable=redefined-builtin, invalid-name
"""Types for distributed tensor."""

import enum

import tvm_ffi

from tvm.relax.type import TensorType, Type
from tvm_tirx.ir import Location
from tvm_tirx.ir.location import UNKNOWN_LOC

from . import _ffi_api
from .global_info import DeviceMesh


class PlacementSpecKind(enum.IntEnum):
    kSharding = 0
    kReplica = 1


@tvm_ffi.register_object("relax.distributed.PlacementSpec")
class PlacementSpec(tvm_ffi.Object):
    """Describes how data is distributed in one dimension of the device mesh

    Parameters
    ----------
    axis: int
        If the kind is sharding, this value represents the tensor dimension to shard.
        otherwise, axis is -1
    kind: PlacementSpecKind
        The kind of placement spec. Possible values: kSharding and kReplica.
    """

    axis: int
    kind: PlacementSpecKind

    def __init__(self, *args, **kwargs):
        raise RuntimeError("PlacementSpec is not intended to be constructed directly, ")

    @staticmethod
    def sharding(axis: int) -> "PlacementSpec":
        """Create a sharding placement spec

        Parameters
        ----------
        axis: int
            The tensor dimension to shard.

        Returns
        -------
        placement_spec: PlacementSpec
            The placement spec.
        """
        return _ffi_api.Sharding(axis)

    @staticmethod
    def replica() -> "PlacementSpec":
        """Create a replica placement spec

        Returns
        -------
        placement_spec: PlacementSpec
            The placement spec.
        """
        return _ffi_api.Replica()


@tvm_ffi.register_object("relax.distributed.Placement")
class Placement(tvm_ffi.Object):
    """Describes how data is distributed in each dimension of the device mesh

    Parameters
    ----------
    dim_specs: List[PlacementSpec]
        The placement spec for each dimension of the device mesh.
    """

    def __init__(self, dim_specs: list[PlacementSpec]):
        self.__init_handle_by_constructor__(_ffi_api.Placement, dim_specs)  # type: ignore

    @staticmethod
    def from_text(text: str) -> "Placement":
        """Create a placement from a text string.

        Parameters
        ----------
        text: str
            The text string.

        Returns
        -------
        placement: Placement
            The placement.
        """
        return _ffi_api.PlacementFromText(text)


@tvm_ffi.register_object("relax.DTensorType")
class DTensorType(Type):
    """Type of a Distributed Tensor value.

    Parameters
    ----------
    tensor_ty: TensorType
        The tensor type carried by the distributed tensor.
    device_mesh: DeviceMesh
        The device mesh of the tensor.
    placement: Placement
        The placement of the tensor among the device mesh

    """

    tensor_ty: TensorType
    device_mesh: DeviceMesh
    placement: Placement

    def __init__(
        self,
        tensor_ty: TensorType,
        device_mesh: DeviceMesh,
        placement: Placement,
        loc: Location = UNKNOWN_LOC,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DTensorType,
            tensor_ty,
            device_mesh,
            placement,
            loc,  # type: ignore
        )
