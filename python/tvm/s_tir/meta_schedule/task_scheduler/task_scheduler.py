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
# ruff: noqa: RUF012
"""Auto-tuning Task Scheduler"""

from collections.abc import Callable
from typing import Union

import tvm_ffi

# isort: off
from typing import Literal

# isort: on

from tvm_ffi import register_object

from .. import _ffi_api
from ..builder import Builder, BuilderResult
from ..cost_model import CostModel
from ..database import Database
from ..logging import get_logger, get_logging_func
from ..measure_callback import MeasureCallback
from ..runner import Runner, RunnerResult
from ..search_strategy import MeasureCandidate
from ..tune_context import TuneContext

logger = get_logger(__name__)  # pylint: disable=invalid-name


@register_object("s_tir.meta_schedule.TaskRecord")
class TaskRecord(tvm_ffi.Object):
    """The running record of a task."""

    ctx: TuneContext
    task_weight: float
    flop: float
    is_terminated: bool
    build_error_count: int
    run_error_count: int
    measure_candidates: list[MeasureCandidate]
    builder_results: list[BuilderResult]
    runner_results: list[RunnerResult]


@register_object("s_tir.meta_schedule.TaskScheduler")
class TaskScheduler(tvm_ffi.Object):
    """The abstract task scheduler interface."""

    tasks_: list[TaskRecord]
    measure_callbacks_: list[MeasureCallback]
    database_: Database | None
    cost_model_: CostModel | None
    remaining_tasks_: int

    TaskSchedulerType = Union["TaskScheduler", Literal["gradient", "round-robin"]]

    def next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        next_task_id : int
            The next task id.
        """
        return _ffi_api.TaskSchedulerNextTaskId(self)  # type: ignore # pylint: disable=no-member

    def join_running_task(self, task_id: int) -> list[RunnerResult]:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.

        Returns
        -------
        results : List[RunnerResult]
            The list of results.
        """
        return _ffi_api.TaskSchedulerJoinRunningTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def tune(
        self,
        tasks: list[TuneContext],
        task_weights: list[float],
        max_trials_global: int,
        max_trials_per_task: int,
        num_trials_per_iter: int,
        builder: Builder,
        runner: Runner,
        measure_callbacks: list[MeasureCallback],
        database: Database | None,
        cost_model: CostModel | None,
    ) -> None:
        """Auto-tuning.

        Parameters
        ----------
        tasks : List[TuneContext]
            The list of tuning contexts as tasks.
        task_weights : List[float]
            The list of task weights.
        max_trials_global : int
            The maximum number of trials globally.
        max_trials_per_task : int
            The maximum number of trials per task.
        num_trials_per_iter : int
            The number of trials per iteration.
        builder : Builder
            The builder.
        runner : Runner
            The runner.
        measure_callbacks : List[MeasureCallback]
            The list of measure callbacks.
        database : Optional[Database]
            The database.
        cost_model : Optional[CostModel]
            The cost model.
        """
        task_weights = [float(w) for w in task_weights]
        _ffi_api.TaskSchedulerTune(  # type: ignore # pylint: disable=no-member
            self,
            tasks,
            task_weights,
            max_trials_global,
            max_trials_per_task,
            num_trials_per_iter,
            builder,
            runner,
            measure_callbacks,
            database,
            cost_model,
        )

    def terminate_task(self, task_id: int) -> None:
        """Terminate the task

        Parameters
        ----------
        task_id : int
            The task id to be terminated.
        """
        _ffi_api.TaskSchedulerTerminateTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def touch_task(self, task_id: int) -> None:
        """Touch the task and update its status

        Parameters
        ----------
        task_id : int
            The task id to be checked.
        """
        _ffi_api.TaskSchedulerTouchTask(self, task_id)  # type: ignore # pylint: disable=no-member

    def print_tuning_statistics(self) -> None:
        """Print out a human-readable format of the tuning statistics."""
        return _ffi_api.TaskSchedulerPrintTuningStatistics(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(  # pylint: disable=keyword-arg-before-vararg
        kind: Literal["round-robin", "gradient"] = "gradient",
        *args,
        **kwargs,
    ) -> "TaskScheduler":
        """Create a task scheduler."""
        from . import (  # pylint: disable=import-outside-toplevel
            GradientBased,
            RoundRobin,
        )

        if kind == "round-robin":
            return RoundRobin(*args, **kwargs)  # type: ignore
        if kind == "gradient":
            return GradientBased(*args, **kwargs)
        raise ValueError(f"Unknown TaskScheduler name: {kind}")


create = TaskScheduler.create  # pylint: disable=invalid-name


@register_object("s_tir.meta_schedule.PyTaskScheduler")
class _PyTaskScheduler(TaskScheduler):
    """
    A TVM object task scheduler to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyTaskScheduler
    """

    def __init__(
        self,
        f_next_task_id: Callable,
        f_join_running_task: Callable,
        f_tune: Callable,
    ):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.TaskSchedulerPyTaskScheduler,  # type: ignore # pylint: disable=no-member
            get_logging_func(logger),
            f_next_task_id,
            f_join_running_task,
            f_tune,
        )


class PyTaskScheduler:
    """
    An abstract task scheduler with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyTaskScheduler,
        "fields": [],
        "methods": ["next_task_id", "join_running_task", "tune"],
    }

    def __init__(self): ...

    def tune(
        self,
        tasks: list[TuneContext],
        task_weights: list[float],
        max_trials_global: int,
        max_trials_per_task: int,
        builder: Builder,
        runner: Runner,
        measure_callbacks: list[MeasureCallback],
        database: Database | None,
        cost_model: CostModel | None,
    ) -> None:
        """Auto-tuning."""
        # Using self._outer to replace the self pointer
        _ffi_api.TaskSchedulerTune(  # type: ignore # pylint: disable=no-member
            self._outer(),  # type: ignore # pylint: disable=no-member
            tasks,
            task_weights,
            max_trials_global,
            max_trials_per_task,
            builder,
            runner,
            measure_callbacks,
            database,
            cost_model,
        )

    def next_task_id(self) -> int:
        """Fetch the next task id.

        Returns
        -------
        next_task_id : int
            The next task id.
        """
        raise NotImplementedError

    def join_running_task(self, task_id: int) -> list[RunnerResult]:
        """Wait until the task is finished.

        Parameters
        ----------
        task_id : int
            The task id to be joined.
        """
        # Using self._outer to replace the self pointer
        return _ffi_api.TaskSchedulerJoinRunningTask(self._outer(), task_id)  # type: ignore # pylint: disable=no-member
