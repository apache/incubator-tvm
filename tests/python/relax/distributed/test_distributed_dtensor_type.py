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

import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm_tirx import tirx
from tvm_tirx.ir import Range


def _check_equal(x, y, map_free_vars=False):
    tvm_tirx.ir.assert_structural_equal(x, y, map_free_vars)
    tvm_tirx.ir.assert_structural_equal(y, x, map_free_vars)

    xhash = tvm_ffi.structural_hash(x, map_free_vars)
    yhash = tvm_ffi.structural_hash(y, map_free_vars)

    assert xhash == yhash


def _check_json_roundtrip(x):
    xret = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(x))
    _check_equal(x, xret, map_free_vars=True)
    return xret


def test_dtensor_type():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")

    tensor_ty0 = rx.TensorType([1, n + 1, m], "float32")
    tensor_ty1 = rx.TensorType([1, n + 1, m], "float32")
    assert tensor_ty0 == tensor_ty1

    device_mesh0 = rx.distributed.DeviceMesh((2, 2), Range(0, 4))
    device_mesh1 = rx.distributed.DeviceMesh((2, 2), Range(0, 4))
    tvm_tirx.ir.assert_structural_equal(device_mesh0, device_mesh1)

    shard0 = rx.distributed.PlacementSpec.sharding(0)
    replica = rx.distributed.PlacementSpec.replica()

    placement0 = rx.distributed.Placement([shard0, replica])
    placement1 = rx.distributed.Placement([shard0, replica])
    tvm_tirx.ir.assert_structural_equal(placement0, placement1)

    ty0 = rx.distributed.DTensorType(tensor_ty0, device_mesh0, placement0)
    ty1 = rx.distributed.DTensorType(tensor_ty1, device_mesh1, placement1)
    _check_equal(ty0, ty1)
    _check_json_roundtrip(ty0)
    _check_json_roundtrip(ty1)

    assert ty0 == ty1
    tvm_tirx.ir.assert_structural_equal(ty0.device_mesh, device_mesh0)
    assert ty0.device_mesh.shape == (2, 2)
    tvm_tirx.ir.assert_structural_equal(ty0.device_mesh.device_range, Range(0, 4))
    tvm_tirx.ir.assert_structural_equal(ty0.placement, placement0)
    assert len(ty0.placement.dim_specs) == 2
    assert ty0.placement.dim_specs[0] == shard0
    assert ty0.placement.dim_specs[1] == replica
    assert ty0.tensor_ty == tensor_ty0

    # can turn into str
    # str(ty0)

    # dimension of device mesh and placement should be the same
    shard1 = rx.distributed.PlacementSpec.sharding(1)
    placement2 = rx.distributed.Placement([shard0, replica, shard1])
    with pytest.raises(ValueError):
        rx.distributed.DTensorType(tensor_ty0, device_mesh0, placement2)

    # Sharding dimension should be smaller than tensor ndim
    shard3 = rx.distributed.PlacementSpec.sharding(3)
    placement3 = rx.distributed.Placement([shard3, replica])
    with pytest.raises(ValueError):
        rx.distributed.DTensorType(tensor_ty0, device_mesh0, placement3)


if __name__ == "__main__":
    tvm_tirx.testing.main()
