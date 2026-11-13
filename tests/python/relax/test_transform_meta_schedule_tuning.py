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

import tempfile

import tvm_ffi

import tvm
import tvm.s_tir.meta_schedule as ms
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.s_tir import script as Ts
from tvm_tirx.ir import transform
from tvm_tirx.ir.module import IRModule
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.script import tirx as T

target = tvm_tirx.target.Target({"kind": "llvm", "num-cores": 16})


@tvm_tirx.script.ir_module
class InputModule:
    @Ts.function
    def tir_matmul(
        A: T.Tensor((32, 32), "float32"),
        B: T.Tensor((32, 32), "float32"),
        C: T.Tensor((32, 32), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})

        for i0, j0, k0 in T.grid(32, 32, 32):
            with Ts.sblock():
                i, j, k = Ts.axis.remap("SSR", [i0, j0, k0])
                with Ts.init():
                    C[i, j] = 0.0
                C[i, j] += A[i, k] * B[j, k]

    @Ts.function
    def tir_relu(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32")):
        T.func_attr({"global_symbol": "tir_relu"})

        for i, j in T.grid(32, 32):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        cls = InputModule
        with R.dataflow():
            lv0 = R.call_tir(cls.tir_matmul, (x, w), ty_args=[R.Tensor((32, 32), dtype="float32")])
            lv1 = R.call_tir(cls.tir_relu, (lv0), ty_args=[R.Tensor((32, 32), dtype="float32")])
            R.output(lv1)
        return lv1


# TODO(@sunggg): determine how to pass MS database object across different passes.
# PassContext might be an option, but we already have TuningAPI database.
# (MS database and TuningAPI database will be unified in the future)
# For now, we only support default JSON database config.
def test_ms_tuning_irmodule():
    mod = InputModule
    assert isinstance(mod, IRModule)

    with tempfile.TemporaryDirectory() as work_dir:
        """
        # TODO(@sunggg): revisit when ready
        with PassContext(trace=Trace(mod), opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneIRMod(
                params={}, work_dir=work_dir, max_trials_global=4, target=target
            )
            out_mod = tuning_pass(mod)
            assert PassContext.current().get_trace_stack_size() == 1
            assert PassContext.current().get_current_trace().size == 1
            tvm_tirx.ir.assert_structural_equal(mod, out_mod)
        """

        with PassContext(opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneIRMod(
                params={}, work_dir=work_dir, max_trials_global=4, target=target
            )
            out_mod = tuning_pass(mod)

            application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir, target=target)

            out_mod = application_pass(mod)
            assert not tvm_ffi.structural_equal(mod, out_mod)


def test_ms_tuning_function():
    mod = InputModule
    assert isinstance(mod, IRModule)
    with tempfile.TemporaryDirectory() as work_dir:
        """
        # TODO(@sunggg): revisit when ready
        with PassContext(trace=Trace(mod), opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneTIR(
                work_dir=work_dir, max_trials_global=4
            )
            out_mod = tuning_pass(mod)
            assert PassContext.current().get_trace_stack_size() == 1
            # TODO (@sunggg): Need to determine how to track subgraph-level tuning traces.
            # Currently, we don't track this so the trace size. Revisit this later.
            tvm_tirx.ir.assert_structural_equal(mod, out_mod)
        """
        with PassContext(opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneIRMod(
                params={}, work_dir=work_dir, max_trials_global=4, target=target
            )
            out_mod = tuning_pass(mod)

            application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir, target=target)
            out_mod = application_pass(mod)
            assert not tvm_ffi.structural_equal(mod, out_mod)

    with tempfile.TemporaryDirectory() as work_dir:
        with PassContext(opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneIRMod(
                params={},
                work_dir=work_dir,
                max_trials_global=4,
                max_trials_per_task=2,
                op_names=["matmul"],
                target=target,
            )
            tuning_pass(mod)

            db = ms.database.JSONDatabase(
                work_dir + "/database_workload.json", work_dir + "/database_tuning_record.json"
            )

            assert len(db.get_all_tuning_records()) == 2

            for rec in db.get_all_tuning_records():
                assert rec.workload.mod["main"].attrs["global_symbol"] == "tir_matmul"


@tvm_tirx.script.ir_module
class DefaultScheduledModule:
    @Ts.function
    def tir_matmul(
        A: T.Tensor((32, 32), "float32"),
        B: T.Tensor((32, 32), "float32"),
        C: T.Tensor((32, 32), "float32"),
    ):
        T.func_attr({"global_symbol": "tir_matmul", "tirx.is_scheduled": True})
        # with Ts.sblock("root"):
        for i0_j0_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for i0_j0_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                for k0 in range(32):
                    with Ts.sblock(""):
                        i = Ts.axis.spatial(32, (i0_j0_fused_0 * 1024 + i0_j0_fused_1) // 32)
                        j = Ts.axis.spatial(32, (i0_j0_fused_0 * 1024 + i0_j0_fused_1) % 32)
                        k = Ts.axis.reduce(32, k0)
                        Ts.reads(A[i, k], B[j, k])
                        Ts.writes(C[i, j])
                        with Ts.init():
                            C[i, j] = T.float32(0)
                        C[i, j] = C[i, j] + A[i, k] * B[j, k]

    @Ts.function
    def tir_relu(A: T.Tensor((32, 32), "float32"), B: T.Tensor((32, 32), "float32")):
        T.func_attr({"global_symbol": "tir_relu", "tirx.is_scheduled": True})
        # with Ts.sblock("root"):
        for i_j_fused_0 in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for i_j_fused_1 in T.parallel(1024, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock(""):
                    vi = Ts.axis.spatial(32, (i_j_fused_0 * 1024 + i_j_fused_1) // 32)
                    vj = Ts.axis.spatial(32, (i_j_fused_0 * 1024 + i_j_fused_1) % 32)
                    Ts.reads(A[vi, vj])
                    Ts.writes(B[vi, vj])
                    B[vi, vj] = T.max(A[vi, vj], T.float32(0))

    @R.function
    def main(
        x: R.Tensor((32, 32), dtype="float32"), w: R.Tensor((32, 32), dtype="float32")
    ) -> R.Tensor((32, 32), dtype="float32"):
        with R.dataflow():
            lv0 = R.call_tir(
                DefaultScheduledModule.tir_matmul,
                (x, w),
                ty_args=[R.Tensor((32, 32), dtype="float32")],
            )
            lv1 = R.call_tir(
                DefaultScheduledModule.tir_relu,
                (lv0,),
                ty_args=[R.Tensor((32, 32), dtype="float32")],
            )
            R.output(lv1)
        return lv1


def test_ms_database_apply_fallback():
    mod = InputModule
    target_cuda = tvm_tirx.target.Target("nvidia/geforce-rtx-3090-ti")
    assert isinstance(mod, IRModule)
    with tempfile.TemporaryDirectory() as work_dir:
        """
        with PassContext(trace=Trace(mod), opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneTIR(
                work_dir=work_dir, max_trials_global=0, target=target_cuda
            )
            out_mod = tuning_pass(mod)
            tvm_tirx.ir.assert_structural_equal(mod, out_mod)
        """
        with PassContext(opt_level=0):
            tuning_pass = relax.transform.MetaScheduleTuneTIR(
                work_dir=work_dir, max_trials_global=0, target=target_cuda
            )
            out_mod = tuning_pass(mod)
            default_pass = tvm.s_tir.transform.DefaultGPUSchedule(target=target_cuda)
            out_mod = default_pass(mod)
            tvm_tirx.ir.assert_structural_equal(out_mod, DefaultScheduledModule)


if __name__ == "__main__":
    tvm_tirx.testing.main()
