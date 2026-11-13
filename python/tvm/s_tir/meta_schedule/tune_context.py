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
"""Meta Schedule tuning context."""

from typing import TYPE_CHECKING, Optional, Union

import tvm_ffi

# isort: off
from typing import Literal

from tvm_ffi import register_object, register_global_func

# isort: on

from tvm.s_tir import Schedule
from tvm_tirx import IRModule
from tvm_tirx.target import Target
from tvm_tirx.tirx import Function

from . import _ffi_api
from .logging import Logger, get_logger, get_logging_func
from .utils import cpu_count

if TYPE_CHECKING:
    from .cost_model import CostModel
    from .database import Database
    from .runner import RunnerResult
    from .search_strategy import MeasureCandidate, SearchStrategy
    from .space_generator import SpaceGenerator


@register_global_func("tvm.s_tir.meta_schedule.normalize_mod")
def _normalize_mod(mod: Function | IRModule) -> IRModule:
    """Normalize the input to an IRModule"""
    if isinstance(mod, Function):
        if not (mod.attrs and "global_symbol" in mod.attrs):
            mod = mod.with_attr("global_symbol", "main")
        mod = mod.with_attr("tirx.noalias", True)
        mod = IRModule({"main": mod})
    if not isinstance(mod, IRModule):
        raise TypeError(f"Expected `mod` to be Function or IRModule, but gets: {mod}")
    func_names = mod.get_global_vars()
    if len(func_names) == 1 and func_names[0].name_hint != "main":
        mod = IRModule({"main": mod[func_names[0]]})
    return mod


@register_object("s_tir.meta_schedule.TuneContext")
class TuneContext(tvm_ffi.Object):
    """The tune context class is designed to contain all resources for a tuning task.

    Parameters
    ----------
    mod : Optional[IRModule] = None
        The workload to be optimized.
    target : Optional[Target] = None
        The target to be optimized for.
    space_generator : Union[None, ScheduleFnType, SpaceGenerator] = None
        The design space generator.
    search_strategy : Union[None, SearchStrategy] = None
        The search strategy.
        if None, the strategy is left blank.
    task_name : Optional[str] = None
        The name of the tuning task.
    logger : logging.Logger
        The logger for the tuning task.
    rand_state : int = -1
        The random state.
        Need to be in integer in [1, 2^31-1], -1 means using random number.
    num_threads : int = None
        The number of threads to be used, None means using the logical cpu count.
    """

    mod: IRModule | None
    target: Target | None
    space_generator: Optional["SpaceGenerator"]
    search_strategy: Optional["SearchStrategy"]
    task_name: str
    logger: Logger | None
    rand_state: int
    num_threads: int

    def __init__(
        self,
        mod: IRModule | None = None,
        *,
        target: Target | str | None = None,
        space_generator: Union["SpaceGenerator.SpaceGeneratorType", None] = None,
        search_strategy: Union["SearchStrategy.SearchStrategyType", None] = None,
        task_name: str = "main",
        rand_state: int = -1,
        num_threads: int | Literal["physical", "logical"] = "physical",
        logger: Logger | None = None,
    ):
        # pylint: disable=import-outside-toplevel
        import tvm.s_tir.tensor_intrin  # pylint: disable=unused-import

        from .search_strategy import SearchStrategy
        from .space_generator import SpaceGenerator

        # pylint: enable=import-outside-toplevel
        if isinstance(mod, Function):
            mod = _normalize_mod(mod)
        if target is not None:
            if not isinstance(target, Target):
                target = Target(target)
        if space_generator is not None:
            if not isinstance(space_generator, SpaceGenerator):
                space_generator = SpaceGenerator.create(space_generator)
        if search_strategy is not None:
            if not isinstance(search_strategy, SearchStrategy):
                search_strategy = SearchStrategy.create(search_strategy)
            # Additional check: ensure it's not the abstract SearchStrategy class itself
            # Use type() for exact type check (not isinstance which would match subclasses)
            elif type(search_strategy) is SearchStrategy:  # pylint: disable=unidiomatic-typecheck
                raise TypeError(
                    "Cannot use abstract SearchStrategy class directly. "
                    "Use SearchStrategy.create() with a valid strategy type "
                    "(e.g., 'evolutionary', 'replay-trace', 'replay-func') "
                    "or use a concrete subclass instead."
                )
        if logger is None:
            logger = get_logger(__name__)
        if not isinstance(num_threads, int):
            if num_threads == "physical":
                num_threads = cpu_count(logical=False)
            elif num_threads == "logical":
                num_threads = cpu_count(logical=True)
            else:
                raise ValueError(
                    f"Invalid num_threads: {num_threads}, "
                    "should be either an integer, 'physical', or 'logical'"
                )
        self.__init_handle_by_constructor__(
            _ffi_api.TuneContext,  # type: ignore # pylint: disable=no-member
            mod,
            target,
            space_generator,
            search_strategy,
            task_name,
            num_threads,
            rand_state,
            get_logging_func(logger),
        )
        _ffi_api.TuneContextInitialize(self)  # type: ignore # pylint: disable=no-member

    def generate_design_space(self) -> list[Schedule]:
        """Generate design spaces given a module.

        Delegated to self.space_generator.generate_design_space with self.mod

        Returns
        -------
        design_spaces : List[tvm.s_tir.Schedule]
            The generated design spaces, i.e., schedules.
        """
        if self.mod is None:
            raise ValueError("`mod` is not provided. Please construct TuneContext with `mod`")
        if self.space_generator is None:
            raise ValueError(
                "space_generator is not provided.Please construct TuneContext with space_generator"
            )
        return self.space_generator.generate_design_space(self.mod)

    def pre_tuning(
        self,
        max_trials: int,
        num_trials_per_iter: int = 64,
        design_spaces: list[Schedule] | None = None,
        database: Optional["Database"] = None,
        cost_model: Optional["CostModel"] = None,
    ) -> None:
        """A method to be called for SearchStrategy to do necessary preparation before tuning.

        Delegated to self.search_strategy.pre_tuning.

        Parameters
        ----------
        max_trials : int
            The maximum number of trials to be executed.
        num_trials_per_iter : int = 64
            The number of trials to be executed per iteration.
        design_spaces : Optional[List[tvm.s_tir.Schedule]]
            The design spaces used during tuning process.
            If None, use the outcome of `self.generate_design_space()`.
        database : Optional[Database] = None
            The database used during tuning process.
            If None, and the search strategy is `EvolutionarySearch`,
            then use `tvm.s_tir.meta_schedule.database.MemoryDatabase`.
        cost_model : Optional[CostModel] = None
            The cost model used during tuning process.
            If None, and the search strategy is `EvolutionarySearch`,
            then use `tvm.s_tir.meta_schedule.cost_model.RandomModel`.
        """
        # pylint: disable=import-outside-toplevel
        from .cost_model import RandomModel
        from .database import MemoryDatabase
        from .search_strategy import EvolutionarySearch

        # pylint: enable=import-outside-toplevel

        if self.search_strategy is None:
            raise ValueError(
                "search_strategy is not provided.Please construct TuneContext with search_strategy"
            )
        if design_spaces is None:
            design_spaces = self.generate_design_space()
        if database is None:
            if isinstance(self.search_strategy, EvolutionarySearch):
                database = MemoryDatabase()  # type: ignore
        if cost_model is None:
            if isinstance(self.search_strategy, EvolutionarySearch):
                cost_model = RandomModel()  # type: ignore
        return self.search_strategy.pre_tuning(
            max_trials,
            num_trials_per_iter,
            design_spaces,
            database,
            cost_model,
        )

    def post_tuning(self) -> None:
        """A method to be called for SearchStrategy to do necessary cleanup after tuning.

        Delegated to self.search_strategy.post_tuning.
        """
        if self.search_strategy is None:
            raise ValueError(
                "search_strategy is not provided.Please construct TuneContext with search_strategy"
            )
        return self.search_strategy.post_tuning()

    def generate_measure_candidates(self) -> list["MeasureCandidate"] | None:
        """Generate a batch of measure candidates from design spaces for measurement.

        Delegated to self.search_strategy.generate_measure_candidates.

        Returns
        -------
        measure_candidates : Optional[List[IRModule]]
            The measure candidates generated, None if search is finished.
        """
        if self.search_strategy is None:
            raise ValueError(
                "search_strategy is not provided.Please construct TuneContext with search_strategy"
            )
        return self.search_strategy.generate_measure_candidates()

    def notify_runner_results(
        self,
        measure_candidates: list["MeasureCandidate"],
        results: list["RunnerResult"],
    ) -> None:
        """Update the state in SearchStrategy with profiling results.

        Delegated to self.search_strategy.notify_runner_results.

        Parameters
        ----------
        measure_candidates : List[MeasureCandidate]
            The measure candidates for update.
        results : List[RunnerResult]
            The profiling results from the runner.
        """
        if self.search_strategy is None:
            raise ValueError(
                "search_strategy is not provided.Please construct TuneContext with search_strategy"
            )
        return self.search_strategy.notify_runner_results(measure_candidates, results)

    def clone(self) -> "TuneContext":
        """Clone the TuneContext.

        Returns
        -------
        cloned_context : TuneContext
            The cloned TuneContext.
        """
        return _ffi_api.TuneContextClone(self)  # type: ignore # pylint: disable=no-member
