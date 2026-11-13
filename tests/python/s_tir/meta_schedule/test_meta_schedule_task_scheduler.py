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
# ruff: noqa: F821, RUF012
"""Test Meta Schedule Task Scheduler"""

import random
import weakref

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import Schedule
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.testing.dummy_object import DummyBuilder, DummyRunner
from tvm.s_tir.utils import derived_object
from tvm_tirx.script import tirx as T

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@tvm_tirx.script.ir_module
class MatmulModule:
    @Ts.function
    def main(  # type: ignore
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        C: T.Tensor((1024, 1024), "float32"),
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm_tirx.script.ir_module
class MatmulReluModule:
    @Ts.function
    def main(  # type: ignore
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        D: T.Tensor((1024, 1024), "float32"),
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        C = Ts.sblock_alloc_tensor((1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(1024, 1024):
            with Ts.sblock("relu"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)  # type: ignore


@tvm_tirx.script.ir_module
class BatchMatmulModule:
    @Ts.function
    def main(  # type: ignore
        A: T.Tensor([16, 128, 128], "float32"),
        B: T.Tensor([16, 128, 128], "float32"),
        C: T.Tensor([16, 128, 128], "float32"),
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for n, i, j, k in T.grid(16, 128, 128, 128):
            with Ts.sblock("matmul"):
                vn, vi, vj, vk = Ts.axis.remap("SSSR", [n, i, j, k])
                with Ts.init():
                    C[vn, vi, vj] = 0.0  # type: ignore
                C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def _schedule_matmul(sch: Schedule):
    block = sch.get_sblock("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _schedule_batch_matmul(sch: Schedule):
    block = sch.get_sblock("matmul")
    i, j, k, t = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 2, 2, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[2, 4, 64, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    t_0, t_1 = sch.split(loop=t, factors=[2, 512])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3, t_0, t_1)


@derived_object
class MyTaskScheduler(ms.task_scheduler.PyTaskScheduler):
    done: set = set()

    def next_task_id(self) -> int:
        tasks = self._outer().tasks_
        while len(self.done) != len(tasks):
            x = random.randint(0, len(tasks) - 1)
            task = tasks[x]
            if not task.is_terminated:
                """Calling base func via following route:
                Python side:
                    PyTaskScheduler does not have `_touch_task`
                    Call TaskScheduler's `touch_task`, which calls ffi
                C++ side:
                    The ffi calls TaskScheduler's `touch_task`
                    But it is overridden in PyTaskScheduler
                    PyTaskScheduler checks if the function is overridden in python
                    If not, it returns the TaskScheduler's vtable, calling
                        TaskScheduler::TouchTask
                """
                if task.runner_futures is not None:
                    self.join_running_task(x)
                return x
            self.done.add(x)
        return -1


def test_meta_schedule_task_scheduler_single():
    device_types: list[str] = []

    @derived_object
    class RecordingRunner(ms.runner.PyRunner):
        def run(self, runner_inputs: list[ms.runner.RunnerInput]) -> list[ms.runner.RunnerFuture]:
            device_types.extend(str(runner_input.device_type) for runner_input in runner_inputs)
            return DummyRunner().run(runner_inputs)

    num_trials_per_iter = 3
    max_trials_per_task = 10
    database = ms.database.MemoryDatabase()
    round_robin = ms.task_scheduler.RoundRobin()
    round_robin.tune(
        [
            ms.TuneContext(
                MatmulModule,
                num_threads=1,
                target=tvm_tirx.target.Target("llvm"),
                space_generator=_schedule_matmul,
                search_strategy=ms.search_strategy.ReplayTrace(),
                task_name="Test",
                rand_state=42,
            )
        ],
        [1.0],
        max_trials_global=num_trials_per_iter,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=64,
        builder=DummyBuilder(),
        runner=RecordingRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        cost_model=None,
    )
    assert len(database) == max_trials_per_task
    assert device_types == ["cpu"] * max_trials_per_task


def test_meta_schedule_task_scheduler_multiple():
    num_trials_per_iter = 6
    max_trials_per_task = 101
    tasks = [
        ms.TuneContext(
            MatmulModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="Matmul",
            rand_state=42,
        ),
        ms.TuneContext(
            MatmulReluModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        ms.TuneContext(
            BatchMatmulModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_batch_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = ms.database.MemoryDatabase()
    round_robin = ms.task_scheduler.RoundRobin()
    round_robin.tune(
        tasks,
        [1.0, 1.0, 1.0],
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        max_trials_global=max_trials_per_task * len(tasks),
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        cost_model=None,
    )
    assert len(database) == max_trials_per_task * len(tasks)
    for task in tasks:
        assert (
            len(
                database.get_top_k(
                    database.commit_workload(task.mod),
                    100000,
                )
            )
            == max_trials_per_task
        )


def test_meta_schedule_task_scheduler_NIE():  # pylint: disable=invalid-name
    @derived_object
    class NIETaskScheduler(ms.task_scheduler.PyTaskScheduler):
        pass

    with pytest.raises(ValueError, match="next_task_id is not defined"):
        scheduler = NIETaskScheduler()
        scheduler.next_task_id()


def test_meta_schedule_task_scheduler_avoid_cyclic():  # pylint: disable=invalid-name
    scheduler = MyTaskScheduler()
    test = weakref.ref(scheduler)  # test if it can be destructed successfully
    del scheduler
    assert test() is None


def test_meta_schedule_task_scheduler_override_next_task_id_only():  # pylint: disable=invalid-name
    max_trials_per_task = 101
    tasks = [
        ms.TuneContext(
            MatmulModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="Matmul",
            rand_state=42,
        ),
        ms.TuneContext(
            MatmulReluModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        ms.TuneContext(
            BatchMatmulModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_batch_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = ms.database.MemoryDatabase()
    scheduler = MyTaskScheduler()
    scheduler.tune(
        tasks,
        task_weights=[1.0] * len(tasks),
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        max_trials_global=max_trials_per_task * len(tasks),
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=6,
        cost_model=None,
    )
    assert len(database) == max_trials_per_task * len(tasks)
    for task in tasks:
        assert (
            len(
                database.get_top_k(
                    database.commit_workload(task.mod),
                    100000,
                )
            )
            == max_trials_per_task
        )


def test_meta_schedule_task_scheduler_multiple_gradient_based():
    max_trials_per_task = 101
    tasks = [
        ms.TuneContext(
            MatmulModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="Matmul",
            rand_state=42,
        ),
        ms.TuneContext(
            MatmulReluModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        ms.TuneContext(
            BatchMatmulModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_batch_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = ms.database.MemoryDatabase()
    gradient_based = ms.task_scheduler.GradientBased()
    gradient_based.tune(
        tasks,
        task_weights=[1.0, 1.0, 1.0],
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        max_trials_global=max_trials_per_task * len(tasks),
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=6,
        cost_model=None,
    )
    assert len(database) == max_trials_per_task * len(tasks)
    for task in tasks:
        assert (
            len(database.get_top_k(database.commit_workload(task.mod), 10000))
            == max_trials_per_task
        )


def test_meta_schedule_task_scheduler_gradient_based_with_null_search_strategy():
    """
    When search strategy of one task returns empty list of candidates or None,
    the scheduler should continue working as normal for other tasks
    """

    @derived_object
    class NullSearchStrategy(ms.search_strategy.PySearchStrategy):
        def __init__(self, rounds_with_empty_candidates):
            self.rounds_with_empty_candidates = rounds_with_empty_candidates

        def _initialize_with_tune_context(self, context: "TuneContext") -> None:
            pass

        def pre_tuning(self, *args, **kwargs):
            pass

        def post_tuning(self):
            pass

        def generate_measure_candidates(self):
            """
            Returns empty list to indicate there is no result from search, while
            the search isn't ended.
            """
            if self.rounds_with_empty_candidates:
                self.rounds_with_empty_candidates -= 1
                return []
            return None

        def notify_runner_results(self, *args, **kwargs):
            pass

        def clone(self):
            return NullSearchStrategy(n=self.n)

    tasks = [
        ms.TuneContext(
            MatmulModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=NullSearchStrategy(rounds_with_empty_candidates=5),
            task_name="Matmul",
            rand_state=42,
        ),
        ms.TuneContext(
            BatchMatmulModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_batch_matmul,
            search_strategy=NullSearchStrategy(rounds_with_empty_candidates=0),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
        ms.TuneContext(
            MatmulReluModule,
            num_threads=1,
            target=tvm_tirx.target.Target("llvm"),
            space_generator=_schedule_matmul,
            search_strategy=ms.search_strategy.ReplayTrace(),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
    ]
    database = ms.database.MemoryDatabase()
    gradient_based = ms.task_scheduler.GradientBased()
    gradient_based.tune(
        tasks,
        task_weights=[1.0, 1.0, 1.0],
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        max_trials_global=30,
        max_trials_per_task=10,
        num_trials_per_iter=6,
        cost_model=None,
    )

    assert len(database) == 10
    assert len(database.get_top_k(database.commit_workload(MatmulModule), 100)) == 0
    assert len(database.get_top_k(database.commit_workload(BatchMatmulModule), 100)) == 0
    assert len(database.get_top_k(database.commit_workload(MatmulReluModule), 100)) == 10


if __name__ == "__main__":
    test_meta_schedule_task_scheduler_single()
    test_meta_schedule_task_scheduler_multiple()
    test_meta_schedule_task_scheduler_NIE()
    test_meta_schedule_task_scheduler_avoid_cyclic()
    test_meta_schedule_task_scheduler_override_next_task_id_only()
    test_meta_schedule_task_scheduler_multiple_gradient_based()
    test_meta_schedule_task_scheduler_gradient_based_with_null_search_strategy()
