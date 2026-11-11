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
"""Meta schedule integration with high-level IR"""

import warnings
from typing import TYPE_CHECKING, Union

import tvm_tirx

# isort: off
from typing import Literal

# isort: on

from tvm_ffi import get_global_func, register_global_func

from tvm_tirx.ir import IntImm, IRModule
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.runtime import Tensor
from tvm_tirx.target import Target

from .builder import Builder
from .cost_model import CostModel
from .database import Database
from .extracted_task import ExtractedTask
from .logging import get_loggers_from_work_dir
from .measure_callback import MeasureCallback
from .runner import Runner
from .search_strategy import SearchStrategy
from .space_generator import SpaceGenerator
from .task_scheduler import TaskScheduler
from .tune import tune_tasks
from .tune_context import TuneContext
from .utils import fork_seed

if TYPE_CHECKING:
    from tvm import relax

_extract_task_func = get_global_func(  # pylint: disable=invalid-name
    "relax.backend.MetaScheduleExtractTask",
    allow_missing=True,
)


def extract_tasks(
    mod: Union[IRModule, "relax.Function"],
    target: Target,
    params: dict[str, Tensor] | None = None,
    module_equality: str = "structural",
) -> list[ExtractedTask]:
    """Extract tuning tasks from a relax program.

    Parameters
    ----------
    mod : Union[IRModule, relax.Function]
        The module or function to tune
    target : tvm_tirx.target.Target
        The compilation target
    params : Optional[Dict[str, tvm_tirx.runtime.Tensor]]
        The associated parameters of the program
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:

          - "structural": Use StructuralEqual/Hash
          - "ignore-tensor": Same as "structural", but ignore tensor raw data during
                              equality testing and hashing.
          - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
                            given module. The "ignore-tensor" varint is used for the extracted
                            blocks or in case no anchor block is found.
                            For the definition of the anchor block, see tirx/analysis/analysis.py.

    Returns
    -------
    tasks: List[ExtractedTask]
        The tasks extracted from this module
    """
    # pylint: disable=import-outside-toplevel
    from tvm.relax.expr import Function as RelaxFunc
    from tvm.relax.transform import BindParams

    # pylint: enable=import-outside-toplevel
    if isinstance(mod, RelaxFunc):
        mod = IRModule({"main": mod})
    if not isinstance(target, Target):
        target = Target(target)
    if params:
        mod = BindParams("main", params)(mod)
    return list(_extract_task_func(mod, target, module_equality))


def extracted_tasks_to_tune_contexts(
    extracted_tasks: list[ExtractedTask],
    work_dir: str,
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    num_threads: Literal["physical", "logical"] | int = "physical",
    seed: int | None = None,
) -> tuple[list[TuneContext], list[float]]:
    """Convert ExtractedTask to TuneContext.

    Parameters
    ----------
    tasks : List[ExtractedTask]
        The tasks to be converted
    work_dir : str
        The working directory to store logs and databases
    space : SpaceGenerator.SpaceGeneratorType
        The space generator to use.
    strategy : SearchStrategy.SearchStrategyType
        The search strategy to use.
    num_threads : Union[Literal["physical", "logical"], int]
        The number of threads to use in multi-threaded search algorithm.
    seed : Optional[int]
        The random seed to use.

    Returns
    -------
    tasks : List[TuneContext]
        The converted tasks
    task_weights : List[float]
        The weights of the tasks
    """
    tasks: list[TuneContext] = []
    task_weights: list[float] = []
    for task, logger, rand_state in zip(
        extracted_tasks,
        get_loggers_from_work_dir(work_dir, [t.task_name for t in extracted_tasks]),
        fork_seed(seed, n=len(extracted_tasks)),
    ):
        if task.mod.attrs.get("tirx.is_scheduled", False):
            warnings.warn("The task {task.task_name} is already scheduled, skipping it.")
            continue
        tasks.append(
            TuneContext(
                mod=task.dispatched[0],
                target=task.target,
                space_generator=space,
                search_strategy=strategy,
                task_name=task.task_name,
                logger=logger,
                rand_state=rand_state,
                num_threads=num_threads,
            ).clone()
        )
        task_weights.append(task.weight)
    return tasks, task_weights


def tune_relax(
    mod: Union[IRModule, "relax.Function"],
    params: dict[str, Tensor],
    target: str | Target,
    work_dir: str,
    max_trials_global: int,
    max_trials_per_task: int | None = None,
    op_names: list[str] | None = None,
    *,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    seed: int | None = None,
    module_equality: str = "structural",
) -> Database:
    """Tune a Relax program.

    Parameters
    ----------
    mod : Union[IRModule, relax.Function]
        The module or function to tune
    params : Optional[Dict[str, tvm_tirx.runtime.Tensor]]
        The associated parameters of the program
    target : Union[Target, str]
        The compilation target
    work_dir : str
        The working directory to store the tuning records
    max_trials_global : int
        The maximum number of trials to run
    max_trials_per_task : Optional[int]
        The maximum number of trials to run for each task
    op_names: Optional[List[str]]
        A list of operator names to specify which op to tune. When it is None, all operators
        are tuned.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : BuilderType
        The builder to use
    runner : RunnerType
        The runner to use
    database : DatabaseType
        The database to use
    cost_model : CostModelType
        The cost model to use
    measure_callbacks : CallbackListType
        The measure callbacks to use
    task_scheduler : TaskSchedulerType
        The task scheduler to use
    space : SpaceGeneratorType
        The space generator to use
    strategy : SearchStrategyType
        The search strategy to use
    seed : Optional[int]
        The random seed
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:

          - "structural": Use StructuralEqual/Hash
          - "ignore-tensor": Same as "structural", but ignore tensor raw data during
                              equality testing and hashing.
          - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
                            given module. The "ignore-tensor" variant is used for the extracted
                            blocks or in case no anchor block is found.
                            For the definition of the anchor block, see tirx/analysis/analysis.py.

    Returns
    -------
    database : Database
        The database that contains the tuning records
    """
    all_tasks = extract_tasks(mod, target, params, module_equality=module_equality)

    if not op_names:
        selected_tasks = all_tasks
    else:
        selected_tasks = []

        for task in all_tasks:
            for op_name in op_names:
                if op_name in task.task_name:
                    selected_tasks.append(task)

    tasks, task_weights = extracted_tasks_to_tune_contexts(
        extracted_tasks=selected_tasks,
        work_dir=work_dir,
        space=space,
        strategy=strategy,
        seed=seed,
    )
    return tune_tasks(
        tasks=tasks,
        task_weights=task_weights,
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
    )


@register_global_func("tvm.s_tir.meta_schedule.tune_relax")
def _tune_relax(
    mod: Union[IRModule, "relax.Function"],
    params: dict[str, Tensor],
    target: str | Target,
    work_dir: str,
    max_trials_global: int,
    max_trials_per_task: int | None = None,
    op_names: list[str] | None = None,
    *,
    num_trials_per_iter: int = 64,
    builder: Builder.BuilderType = "local",
    runner: Runner.RunnerType = "local",
    database: Database.DatabaseType = "json",
    cost_model: CostModel.CostModelType = "xgb",
    measure_callbacks: MeasureCallback.CallbackListType = "default",
    task_scheduler: TaskScheduler.TaskSchedulerType = "gradient",
    space: SpaceGenerator.SpaceGeneratorType = "post-order-apply",
    strategy: SearchStrategy.SearchStrategyType = "evolutionary",
    seed: int | None = None,
    module_equality: str = "structural",
) -> Database:
    """Interface with tuning api to tune a Relax program.

    Parameters
    ----------
    mod : Union[IRModule, relax.Function]
        The module or function to tune
    params : Optional[Dict[str, tvm_tirx.runtime.Tensor]]
        The associated parameters of the program
    target : Union[Target, str]
        The compilation target
    work_dir : str
        The working directory to store the tuning records
    max_trials_global : int
        The maximum number of trials to run
    max_trials_per_task : Optional[int]
        The maximum number of trials to run for each task
    op_names: Optional[List[str]]
        A list of operator names to specify which op to tune. When it is None, all operators
        are tuned.
    num_trials_per_iter : int
        The number of trials to run per iteration
    builder : BuilderType
        The builder to use
    runner : RunnerType
        The runner to use
    database : DatabaseType
        The database to use
    cost_model : CostModelType
        The cost model to use
    measure_callbacks : CallbackListType
        The measure callbacks to use
    task_scheduler : TaskSchedulerType
        The task scheduler to use
    space : SpaceGeneratorType
        The space generator to use
    strategy : SearchStrategyType
        The search strategy to use
    seed : Optional[int]
        The random seed
    module_equality : Optional[str]
        A string to specify the module equality testing and hashing method.
        It must be one of the followings:

          - "structural": Use StructuralEqual/Hash
          - "ignore-tensor": Same as "structural", but ignore tensor raw data during
                              equality testing and hashing.
          - "anchor-block": Apply equality testing and hashing on the anchor block extracted from a
                            given module. The "ignore-tensor" varint is used for the extracted
                            blocks or in case no anchor block is found.
                            For the definition of the anchor block, see tirx/analysis/analysis.py.

    Returns
    -------
    ret_mod : IRModule
        IRModule
    """
    if isinstance(max_trials_global, IntImm):
        max_trials_global = int(max_trials_global)
    if isinstance(max_trials_per_task, IntImm):
        max_trials_per_task = int(max_trials_per_task)

    tune_relax(
        mod,
        params,
        target,
        work_dir,
        max_trials_global,
        max_trials_per_task=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        op_names=op_names,
        builder=builder,
        runner=runner,
        database=database,
        cost_model=cost_model,
        measure_callbacks=measure_callbacks,
        task_scheduler=task_scheduler,
        space=space,
        strategy=strategy,
        seed=seed,
        module_equality=module_equality,
    )
    # Return original IRModule
    # This pass only makes optimization decision
    return mod


def compile_relax(
    database: Database,
    mod: IRModule,
    target: Target | str,
    params: dict[str, Tensor] | None,
    enable_warning: bool = False,
) -> "relax.VMExecutable":
    """Compile a relax program with a MetaSchedule database.

    Parameters
    ----------
    database : Database
        The database to use
    mod : IRModule
        The Relax program to be compiled
    target : tvm_tirx.target.Target
        The compilation target
    params : Optional[Dict[str, tvm_tirx.runtime.Tensor]]
        The associated parameters of the program
    enable_warning : bool
        A boolean value indicating if to print warnings for TIR functions not
        showing up in the database. By default we don't print warning.

    Returns
    -------
    lib : relax.VMExecutable
        The built runtime module or vm VMExecutable for the given relax workload.
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relax
    from tvm.relax import build as relax_build
    from tvm.relax import pipeline as relax_pipeline_mod
    from tvm.relax.transform import BindParams, MetaScheduleApplyDatabase
    from tvm.s_tir import dlight as dl

    # pylint: enable=import-outside-toplevel
    if not isinstance(target, Target):
        target = Target(target)
    if params:
        mod = BindParams("main", params)(mod)

    # Build a pipeline with the correct ordering:
    #   1. library_dispatch + LegalizeOps + FuseOps + FuseTIR
    #      (same preparation as extract_tasks, so database keys match)
    #   2. MetaScheduleApplyDatabase — replaces tuned fused-TIR functions
    #   3. DLight fallback — schedules remaining untuned functions
    #   4. dataflow_lower + finalize passes
    #
    # Applying MetaScheduleApplyDatabase BEFORE FuseOps (the original bug)
    # caused DLight.Matmul to fail on cache-write stages embedded in fused TIR.
    #
    # All pass lists are obtained from relax.pipeline.*_passes(target) so that
    # target-specific helpers (dispatch, finalize) are shared with the default
    # pipeline rather than duplicated here.
    try:
        dispatch_passes = relax_pipeline_mod.library_dispatch_passes(target)
    except (ValueError, AttributeError):
        dispatch_passes = []

    try:
        lower_passes = relax_pipeline_mod.dataflow_lower_passes(target)
        finalize_passes = relax_pipeline_mod.finalize_passes(target)
    except (ValueError, AttributeError):
        # Fallback for targets not yet registered in the pipeline dispatcher
        lower_passes = [
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
        ]
        finalize_passes = [
            relax.transform.StaticPlanBlockMemory(),
            relax.transform.LowerAllocTensor(),
            relax.transform.KillAfterLastUse(),
            relax.transform.LowerRuntimeBuiltin(),
            relax.transform.ComputePrimValue(),
            relax.transform.VMShapeLower(),
            relax.transform.AttachGlobalSymbol(),
        ]

    is_gpu_target = relax_pipeline_mod.BackendDispatcher.is_gpu_target(target)

    @tvm_tirx.ir.transform.module_pass(opt_level=3)
    def _ms_pipeline(
        mod: tvm_tirx.ir.IRModule, _ctx: tvm_tirx.ir.transform.PassContext
    ) -> tvm_tirx.ir.IRModule:
        fuse_seq = [
            *dispatch_passes,
            relax.transform.LegalizeOps(enable_warning=enable_warning, target=target),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FoldConstant(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ]
        mod = tvm_tirx.ir.transform.Sequential(fuse_seq)(mod)
        mod = MetaScheduleApplyDatabase(enable_warning=enable_warning, target=target)(mod)
        # DLight handles functions not covered by the database.
        # GPU rules apply only for GPU targets.
        if is_gpu_target:
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
                target=target,
            )(mod)
        mod = tvm_tirx.ir.transform.Sequential(lower_passes + finalize_passes)(mod)
        return mod

    with database, PassContext(opt_level=3):
        relax_ex = relax_build(mod, target=target, relax_pipeline=_ms_pipeline)
    return relax_ex
