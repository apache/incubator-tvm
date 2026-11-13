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
"""MetaSchedule-TIR integration"""

from collections.abc import Mapping

# isort: off
from typing import Literal
from tvm_ffi import register_global_func

# isort: on
from tvm.s_tir.schedule import Schedule as _Schedule
from tvm_tirx import ir, tirx
from tvm_tirx.ir import IntImm
from tvm_tirx.target import Target

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .logging import get_loggers_from_work_dir
from .measure_callback import MeasureCallback
from .runner import Runner
from .search_strategy import SearchStrategy
from .space_generator import SpaceGenerator
from .task_scheduler import TaskScheduler
from .tune import tune_tasks
from .tune_context import TuneContext, _normalize_mod
from .utils import fork_seed


def tune_tir(  # pylint: disable=too-many-locals
    mod: ir.IRModule | tirx.Function,
    target: str | Target,
    work_dir: str,
    max_trials_global: int,
    *,
    max_trials_per_task: int | None = None,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    num_tuning_cores: Literal["physical", "logical"] | int = "physical",
    seed: int | None = None,
    module_equality: str = "structural",
    special_space: Mapping[str, SpaceGenerator.SpaceGeneratorType] | None = None,
    post_optimization: bool | None = False,
) -> Database:
    """Tune a TIR function or an IRModule of TIR functions.

    Parameters
    ----------
    mod : Union[ir.IRModule, tirx.Function]
        The TIR IRModule to tune.
    target : Union[str, Target]
        The target to tune for.
    work_dir : str
        The working directory.
    max_trials_global : int
        The maximum number of trials to run globally.
    max_trials_per_task : Optional[int]
        The maximum number of trials to run per task.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : Builder.BuilderType
        The builder.
    runner : Runner.RunnerType
        The runner.
    database : Database.DatabaseType
        The database.
    cost_model : CostModel.CostModelType
        The cost model.
    measure_callbacks : MeasureCallback.CallbackListType
        The measure callbacks.
    task_scheduler : TaskScheduler.TaskSchedulerType
        The task scheduler.
    space : SpaceGenerator.SpaceGeneratorType
        The space generator.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy.
    num_tuning_cores : Union[Literal["physical", "logical"], int]
        The number of CPU cores to use during tuning.
    seed : Optional[int]
        The seed for the random number generator.
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
    special_space : Optional[Mapping[str, SpaceGenerator.SpaceGeneratorType]]
        A mapping from task name to a special space generator for that task.

    Returns
    -------
    database : Database
        The database with all tuning records
    """
    if isinstance(mod, tirx.Function):
        mod = _normalize_mod(mod)

    named_tasks: list[tuple[str, tirx.Function]] = []
    for gv, func in mod.functions_items():  # pylint: disable=invalid-name
        if isinstance(func, tirx.Function):
            named_tasks.append((gv.name_hint, func))
    named_tasks.sort(key=lambda x: x[0])

    task_names = [x for x, _ in named_tasks]
    tasks: list[TuneContext] = []
    for task_name, task_func, logger, rand_state in zip(
        task_names,
        [x for _, x in named_tasks],
        get_loggers_from_work_dir(work_dir, task_names),
        fork_seed(seed, n=len(named_tasks)),
    ):
        if special_space and task_name in special_space:
            task_space = special_space[task_name]
        else:
            task_space = space
        if task_space is None:
            continue
        tasks.append(
            TuneContext(
                mod=task_func,
                target=target,
                space_generator=task_space,
                search_strategy=strategy,
                task_name=task_name,
                rand_state=rand_state,
                num_threads=num_tuning_cores,
                logger=logger,
            ).clone()
        )
    return tune_tasks(
        tasks=tasks,
        task_weights=[1.0] * len(tasks),
        work_dir=work_dir,
        max_trials_global=max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
        module_equality=module_equality,
        post_optimization=post_optimization,
    )


@register_global_func("tvm.s_tir.meta_schedule.tune_tir")
def _tune_tir(
    mod: ir.IRModule | tirx.Function,
    target: str | Target,
    work_dir: str,
    max_trials_global: int,
    *,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "round-robin",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    num_tuning_cores: Literal["physical", "logical"] | int = "physical",
    seed: int | None = None,
) -> Database:
    """Interface with tuning api to tune a TIR program.

    Parameters
    ----------
    mod : Union[ir.IRModule, tirx.Function]
        The TIR function to tune.
    target : Union[str, Target]
        The target to tune for.
    work_dir : str
        The working directory.
    max_trials_global : int
        The maximum number of trials to run globally.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : Builder.BuilderType
        The builder.
    runner : Runner.RunnerType
        The runner.
    database : Database.DatabaseType
        The database.
    cost_model : CostModel.CostModelType
        The cost model.
    measure_callbacks : MeasureCallback.CallbackListType
        The measure callbacks.
    task_scheduler : TaskScheduler.TaskSchedulerType
        The task scheduler.
    space : SpaceGenerator.SpaceGeneratorType
        The space generator.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy.
    num_tuning_cores : Union[Literal["physical", "logical"], int]
        The number of CPU cores to use during tuning.
    seed : Optional[int]
        The seed for the random number generator.

    Returns
    -------
    ret_mod : IRModule
        IRModule
    """
    if isinstance(max_trials_global, IntImm):
        max_trials_global = int(max_trials_global)
    tune_tir(
        mod,
        target,
        work_dir,
        max_trials_global,
        num_trials_per_iter=num_trials_per_iter,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
        space=space,
        strategy=strategy,
        num_tuning_cores=num_tuning_cores,
        seed=seed,
    )
    # Return original IRModule
    # This pass only makes optimization decision
    return mod


def compile_tir(
    database: Database,
    mod: ir.IRModule | tirx.Function,
    target: Target | str,
) -> _Schedule:
    """Compile a TIR to s_tir.Schedule, according to the records in the database.

    Parameters
    ----------
    database : Database
        The database of tuning records.
    mod : Union[ir.IRModule, tirx.Function]
        The TIR function to tune.
    target : Union[str, Target]
        The target to tune for.

    Returns
    -------
    sch : s_tir.Schedule
        The best schedule found in the database.
    """
    mod = _normalize_mod(mod)
    if not isinstance(target, Target):
        target = Target(target)
    return database.query_schedule(mod, target, workload_name="main")
