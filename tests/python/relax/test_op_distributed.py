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

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm_tirx.script.parser import relax as R


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_redistribute_R_to_S():
    bb = relax.BlockBuilder()
    mesh = R.device_mesh((4,), list(range(4)))
    x = relax.Var("x", R.DTensor((3, 4), "float32", device_mesh=mesh, placement="R"))

    # Distributed placement requires the contextual builder inference rule.
    distributed_abs = relax.op.abs(x)
    tvm_tirx.ir.assert_structural_equal(
        tvm_tirx.ir.reinfer_type(distributed_abs), tvm_tirx.ir.Type.missing()
    )
    _check_inference(bb, distributed_abs, x.ty)

    _check_inference(
        bb,
        R.distributed.redistribute_replica_to_shard(x, num_workers=4, axis=1),
        R.DTensor((3, 4), "float32", device_mesh=mesh, placement="S[1]"),
    )

    # wrong: indivisible
    with pytest.raises(ValueError):
        bb.normalize(R.distributed.redistribute_replica_to_shard(x, num_workers=4, axis=0))

    y = relax.Var("y", R.Tensor((3, 4), "float32"))
    _check_inference(
        bb,
        R.distributed.redistribute_replica_to_shard(y, num_workers=4, axis=1),
        R.Tensor((3, 1), "float32"),
    )

    # wrong: indivisible
    with pytest.raises(ValueError):
        bb.normalize(R.distributed.redistribute_replica_to_shard(y, num_workers=4, axis=0))


if __name__ == "__main__":
    tvm_tirx.testing.main()
