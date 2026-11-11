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
# ruff: noqa: F401

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

target = tvm_tirx.target.Target({"kind": "llvm", "num-cores": 16})


def test_apply_to_func_with_different_block_name():
    @I.ir_module
    class RecordModule:
        @Ts.function
        def main(A: T.Tensor((2,), "float32"), B: T.Tensor((2,), "float32")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            for i in T.serial(2):
                with Ts.sblock("block"):
                    vi = Ts.axis.spatial(2, i)
                    B[vi] = A[vi]

    @I.ir_module
    class BlockRenamedModule:
        @Ts.function
        def main(A: T.Tensor((2,), "float32"), B: T.Tensor((2,), "float32")):
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            for i in T.serial(2):
                with Ts.sblock("renamed_block"):
                    vi = Ts.axis.spatial(2, i)
                    B[vi] = A[vi]

    @I.ir_module
    class Expected:
        @Ts.function
        def main(A: T.Tensor((2,), "float32"), B: T.Tensor((2,), "float32")):
            T.func_attr(
                {
                    "tirx.is_scheduled": True,
                    "global_symbol": "main",
                    "tirx.noalias": True,
                }
            )
            for i in T.serial(2):
                with Ts.sblock("renamed_block"):
                    vi = Ts.axis.spatial(2, i)
                    B[vi] = A[vi]

    def create_trace(mod: tvm_tirx.IRModule):
        sch = tvm.s_tir.Schedule(mod)
        _ = sch.get_sblock("block")
        return sch.trace

    db = ms.database.create(kind="memory")
    db.commit_workload(RecordModule)
    db.commit_tuning_record(
        ms.database.TuningRecord(
            create_trace(RecordModule), ms.database.Workload(RecordModule), [0.0], target
        )
    )

    with db:
        mod = relax.transform.MetaScheduleApplyDatabase(target=target)(BlockRenamedModule)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
