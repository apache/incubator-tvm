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
# ruff: noqa: E501, F401, F841

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_redistribute_replica_to_shard():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((10, 10), "float32"))  -> R.Tensor((10, 5), "float32"):
            gv0 = R.dist.redistribute_replica_to_shard(x, num_workers=2, axis=1)
            return gv0

    worker_id = T.dynamic("worker_id", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def strided_slice(A: T.Tensor((T.int64(10), T.int64(10)), "float32"), worker_id: T.int64, redistribute_replica_to_shard: T.Tensor((T.int64(10), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(10), T.int64(5)):
                with Ts.sblock("redistribute_replica_to_shard"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, worker_id * T.int64(5) + v_i1])
                    Ts.writes(redistribute_replica_to_shard[v_i0, v_i1])
                    redistribute_replica_to_shard[v_i0, v_i1] = A[v_i0, worker_id * T.int64(5) + v_i1]

        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 5), dtype="float32"):
            cls = Expected
            gv: R.Shape(ndim=-1) = R.call_pure_packed("runtime.disco.worker_id", ty_args=(R.Shape(ndim=-1),))
            gv1: R.Shape([worker_id]) = R.match_cast(gv, R.Shape([worker_id]))
            gv0 = R.call_tir(cls.strided_slice, (x, worker_id), ty_args=[R.Tensor((10, 5), dtype="float32")])
            return gv0
    # fmt: on

    mod = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
