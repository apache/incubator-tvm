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

#  type: ignore
from __future__ import annotations

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R


def test_simple():
    @I.ir_module
    class Before:
        I.module_attrs({"device_num": 2})
        I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2))]})

        @R.function
        def foo(
            x1: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            x2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        ):
            R.func_attr({"num_input": 1})
            # scatter
            lv0 = R.dist.redistribute(x1, "mesh[0]", "S[1]")
            # do nothing
            lv1 = R.dist.redistribute(x2, "mesh[0]", "S[0]")
            return (lv0, lv1)

    @I.ir_module
    class Expected:
        I.module_attrs({"device_num": 2})
        I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2))]})

        @R.function
        def foo(
            x1: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
            x2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        ) -> R.Tuple(
            R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
            R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        ):
            R.func_attr({"num_input": 1})
            lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = (
                R.dist.redistribute_replica_to_shard(x1, num_workers=2, axis=1)
            )
            lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]") = x2
            return (lv0, lv1)

    after = relax.distributed.transform.LegalizeRedistribute()(Before)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
