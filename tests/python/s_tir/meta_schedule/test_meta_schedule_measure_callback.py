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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import tempfile

import pytest

import tvm_tirx
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule import Schedule
from tvm.s_tir.utils import derived_object
from tvm_tirx.script import tirx as T

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,
# fmt: off

@tvm_tirx.script.ir_module
class Matmul:
    @Ts.function
    def main(
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        C: T.Tensor((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main"})

        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def test_meta_schedule_measure_callback():
    @derived_object
    class FancyMeasureCallback(ms.measure_callback.PyMeasureCallback):
        def apply(
            self,
            task_scheduler: ms.task_scheduler.TaskScheduler,
            task_id: int,
            measure_candidates: list[ms.MeasureCandidate],
            builder_results: list[ms.builder.BuilderResult],
            runner_results: list[ms.runner.RunnerResult],
        ) -> None:
            assert len(measure_candidates) == 1
            tvm_tirx.ir.assert_structural_equal(measure_candidates[0].sch.mod, Matmul)
            assert (
                len(builder_results) == 1
                and builder_results[0].error_msg is None
                and builder_results[0].artifact_path == "test_build"
            )
            assert (
                len(runner_results) == 1
                and runner_results[0].error_msg is None
                and len(runner_results[0].run_secs) == 2
            )

    measure_callback = FancyMeasureCallback()
    measure_callback.apply(
        ms.task_scheduler.RoundRobin(),
        0,
        [ms.MeasureCandidate(Schedule(Matmul), None)],
        [ms.builder.BuilderResult("test_build", None)],
        [ms.runner.RunnerResult([1.0, 2.1], None)],
    )


def test_meta_schedule_measure_callback_fail():
    @derived_object
    class FailingMeasureCallback(ms.measure_callback.PyMeasureCallback):
        def apply(
            self,
            task_scheduler: ms.task_scheduler.TaskScheduler,
            task_id: int,
            measure_candidates: list[ms.MeasureCandidate],
            builder_results: list[ms.builder.BuilderResult],
            runner_results: list[ms.runner.RunnerResult],
        ) -> None:
            raise ValueError("test")

    measure_callback = FailingMeasureCallback()
    with pytest.raises(ValueError, match="test"):
        measure_callback.apply(
            ms.task_scheduler.RoundRobin(),
            0,
            [ms.MeasureCandidate(Schedule(Matmul), None)],
            [ms.builder.BuilderResult("test_build", None)],
            [ms.runner.RunnerResult([1.0, 2.1], None)],
        )


@pytest.mark.skip("Tuning test - launches runner")
def test_meta_schedule_measure_callback_update_cost_model_with_zero():
    @derived_object
    class AllZeroRunnerFuture(ms.runner.PyRunnerFuture):
        def done(self) -> bool:
            return True

        def result(self) -> ms.runner.RunnerResult:
            return ms.runner.RunnerResult([0.0, 0.0], None)

    @derived_object
    class AllZeroRunner(ms.runner.PyRunner):
        def run(self, runner_inputs: list[ms.runner.RunnerInput]) -> list[ms.runner.RunnerResult]:
            return [AllZeroRunnerFuture() for _ in runner_inputs]

    with tempfile.TemporaryDirectory() as work_dir:
        ms.tune_tir(
            mod=Matmul,
            target={"kind": "llvm", "num-cores": 1},
            work_dir=work_dir,
            max_trials_global=10,
            runner=AllZeroRunner(),
            measure_callbacks=[ms.measure_callback.UpdateCostModel()],
        )


@pytest.mark.skip("Tuning test - launches runner")
def test_meta_schedule_measure_callback_update_cost_model_with_runtime_error():
    @derived_object
    class EmptyRunnerFuture(ms.runner.PyRunnerFuture):
        def done(self) -> bool:
            return True

        def result(self) -> ms.runner.RunnerResult:
            return ms.runner.RunnerResult(None, "error")

    @derived_object
    class EmptyRunner(ms.runner.PyRunner):
        def run(self, runner_inputs: list[ms.runner.RunnerInput]) -> list[ms.runner.RunnerResult]:
            return [EmptyRunnerFuture() for _ in runner_inputs]

    with tempfile.TemporaryDirectory() as work_dir:
        ms.tune_tir(
            mod=Matmul,
            target={"kind": "llvm", "num-cores": 1},
            work_dir=work_dir,
            max_trials_global=10,
            runner=EmptyRunner(),
            measure_callbacks=[ms.measure_callback.UpdateCostModel()],
        )


if __name__ == "__main__":
    test_meta_schedule_measure_callback()
    test_meta_schedule_measure_callback_fail()
    test_meta_schedule_measure_callback_update_cost_model_with_zero()
    test_meta_schedule_measure_callback_update_cost_model_with_runtime_error()
