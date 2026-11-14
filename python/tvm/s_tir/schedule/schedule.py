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
"""The TensorIR schedule class"""

import inspect
from collections.abc import Callable
from typing import Literal

import tvm_ffi
from tvm_ffi import register_error
from tvm_ffi import register_object as _register_object

from tvm.s_tir import SBlock
from tvm_tirx.ir import Expr, FloatImm, For, GlobalVar, IntImm, IRModule, is_prim_expr
from tvm_tirx.runtime import DataTypeCode, MemoryScope
from tvm_tirx.tirx import Function, Var, is_tensor_var
from tvm_tirx.tirx.index_map import IndexMap

from . import _ffi_api
from ._type_checker import type_checked
from .state import ScheduleState, StmtSRef, _parse_debug_mask, _parse_mod
from .trace import Trace


@register_error
class ScheduleError(RuntimeError):
    """Error that happens during TensorIR scheduling."""


@_register_object("s_tir.LoopRV")
class LoopRV(tvm_ffi.Object):
    """A random variable that refers to a loop"""

    def __init__(self) -> None:
        """Construct a new LoopRV."""
        self.__init_handle_by_constructor__(
            _ffi_api.LoopRV  # type: ignore # pylint: disable=no-member
        )


@_register_object("s_tir.SBlockRV")
class SBlockRV(tvm_ffi.Object):
    """A random variable that refers to a block"""

    def __init__(self) -> None:
        """Construct a new SBlockRV."""
        self.__init_handle_by_constructor__(
            _ffi_api.SBlockRV  # type: ignore # pylint: disable=no-member
        )


# It is a workaround for mypy: https://github.com/python/mypy/issues/7866#issuecomment-549454370
# This feature is not supported until python 3.10:
# https://docs.python.org/3.10/whatsnew/3.10.html#pep-613-typealias
# A random variable that evaluates to an integer
ExprRV = Expr  # pylint: disable=invalid-name

RAND_VAR_TYPE = ExprRV | SBlockRV | LoopRV  # pylint: disable=invalid-name

_ERROR_RENDER_LEVEL: dict[Literal["detail", "fast", "none"], int] = {
    "detail": 0,
    "fast": 1,
    "none": 2,
}


def _parse_error_render_level(error_render_level: str) -> int:
    if error_render_level not in _ERROR_RENDER_LEVEL:
        raise ValueError(
            'error_render_level can be "detail", "fast", or "none", but got: '
            + f"{error_render_level}"
        )
    return _ERROR_RENDER_LEVEL.get(error_render_level)


def _parse_enable_checks(enable_checks: bool) -> bool:
    if not isinstance(enable_checks, bool):
        raise TypeError(f"enable_checks only accepts bool value, got {type(enable_checks)} instead")
    return enable_checks


def _parse_seed(seed: int | None) -> int:
    if seed is None:
        return -1
    if not isinstance(seed, int):
        raise TypeError(f"Expected `seed` to be int or None, but gets: {seed}")
    if seed < 1 or seed > 2147483647:
        raise ValueError(f"seed must be in the range [1, 2147483647], but gets: {seed}")
    return seed


def _get_sblock_default_dtype(block: SBlock) -> str:
    for i in block.iter_vars:
        return str(i.var.ty)
    for tensor_region in list(block.reads) + list(block.writes):
        for dom in tensor_region.region:
            return str(dom.begin.ty)
    return "int64"


@_register_object("s_tir.Schedule")
class Schedule(tvm_ffi.Object):
    """The user-facing schedule class

    A schedule is a set of transformations that change the order of computation but
    preserve the semantics of computation. Some example of schedules:
    1) Split a loop into two;
    2) Reorder two loops;
    3) Inline the computation of a specific tensor into its consumer

    The schedule class stores auxiliary information to schedule correctly and efficiently.

    Link to tutorial: https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html
    """

    @type_checked
    def __init__(
        self,
        mod: Function | IRModule,
        *,
        seed: int | None = None,
        debug_mask: str | int = "none",
        error_render_level: str = "detail",
        enable_check: bool = True,
    ) -> None:
        """Construct a TensorIR schedule class from an IRModule

        Parameters
        ----------
        mod : Function | IRModule
            The IRModule or Function to be scheduled
        seed: Optional[int]
            The seed value for schedule's random state
            Note that None and -1 means use device random, otherwise only integer between 1 and
            2147483647 is allowed.
        debug_mask : str | int
            Do extra correctness checking after the class creation and each time
            after calling the Replace method.
            Possible choices of `debug_mask`:
            1) "all" - Turn on all the checks
            2) "none" - Turn off all the checks
            3) An integer - Turn on checks according to the bitmasks provided in ScheduleDebugMask
        error_render_level : str = "detail"
            The level of error rendering. Choices: "detail", "fast", "none".
            - "detail": Render a detailed error message, with the TIR and error locations printed
            - "fast: Show a simple error message without rendering or string manipulation
            - "none": Do not show any error message.
        enable_check : bool = True
            The default schedule checks are too strict and might prevent us performing some valid
            schedules. `enable_check` is an argument to control whether we enable prerequisite
            checks for some schedule primitives or not:
            - true: perform prerequisite check before applying some schedules.
            - false: do not perform some check before applying schedules, but still raise error
            if schedule fails.

            It's user duty to guarantee schedule correctness if `enable_check` is set to `False`.

        Note
        ----
        The checks performed includes:
        1) VerifySRefTree
        2) VerifyCachedFlags
        """
        # call the constructor
        self.__init_handle_by_constructor__(
            _ffi_api.TracedSchedule,  # type: ignore # pylint: disable=no-member
            _parse_mod(mod),
            _parse_seed(seed),
            _parse_debug_mask(debug_mask),
            _parse_error_render_level(error_render_level),
            _parse_enable_checks(enable_check),
        )

    @staticmethod
    def _create_non_traced(
        mod: Function | IRModule,
        *,
        seed: int | None = None,
        debug_mask: str | int = "none",
        error_render_level: str = "detail",
        enable_check: bool = True,
    ) -> "Schedule":
        """Construct a non-traced TensorIR schedule class from an IRModule."""
        return _ffi_api.ConcreteSchedule(  # type: ignore # pylint: disable=no-member
            _parse_mod(mod),
            _parse_seed(seed),
            _parse_debug_mask(debug_mask),
            _parse_error_render_level(error_render_level),
            _parse_enable_checks(enable_check),
        )

    ########## Utilities ##########

    @property
    def mod(self) -> IRModule:
        """Returns the AST of the module being scheduled"""
        return _ffi_api.ScheduleGetMod(self)  # type: ignore # pylint: disable=no-member

    @property
    def state(self) -> ScheduleState:
        """Returns the ScheduleState in the current schedule class"""
        return _ffi_api.ScheduleGetState(self)  # type: ignore # pylint: disable=no-member

    @property
    def trace(self) -> Trace | None:
        """Returns the internally maintained trace of scheduling program execution"""
        return _ffi_api.ScheduleGetTrace(self)  # type: ignore # pylint: disable=no-member

    @property
    def func_working_on(self) -> GlobalVar | None:
        """Returns the GlobalVar of the func that the schedule is currently working on"""
        return _ffi_api.ScheduleGetFuncWorkingOn(self)  # type: ignore # pylint: disable=no-member

    def work_on(self, func_name: str) -> None:
        """Instruct the schedule to work on a function in the IRModule.

        By default, the schedule works on the function with the name "main", or the only function in
        the IRModule if there is only one. If there is multiple functions in the IRModule, and none
        of their names are "main", users will have to call this method to explicitly specify which
        function to work on.

        This sugar function will guide the `GetSBlock` method if its `func_name` is not specified.

        Parameters
        ----------
        func_name : str
            The name of the function to work on.
        """
        _ffi_api.ScheduleWorkOn(self, func_name)  # type: ignore # pylint: disable=no-member

    def copy(self) -> "Schedule":
        """Returns a copy of the schedule, including both the state and the symbol table,
        * guaranteeing that
        * 1) SRef tree is completely reconstructed;
        * 2) The IRModule being scheduled is untouched;
        * 3) All the random variables are valid in the copy, pointing to the corresponding sref
        * reconstructed

        Returns
        -------
        copy : Schedule
            A new copy of the schedule
        """
        return _ffi_api.ScheduleCopy(self)  # type: ignore # pylint: disable=no-member

    @type_checked
    def seed(self, seed: int) -> None:
        """Seed the randomness

        Parameters
        ----------
        seed : int
            The new random seed, -1 if use device random, otherwise non-negative
        """
        return _ffi_api.ScheduleSeed(self, seed)  # type: ignore # pylint: disable=no-member

    def fork_seed(self) -> int:
        """Returns a forked random state as seed for new schedules

        Returns
        -------
        seed : int
            The forked random state, not the same as the current random state
        """
        return _ffi_api.ScheduleForkSeed(self)  # type: ignore # pylint: disable=no-member

    def show(self, *args, **kwargs) -> None:
        """A sugar for print highlighted TVM script.

        All parameters are forwarded to the underlying `Module.show`
        and `Trace.show` methods.
        """
        mod = self.mod
        if mod is not None:
            mod.show(*args, **kwargs)

        trace = self.trace
        if trace is not None:
            # Trace.show only supports the style argument
            param_binding = inspect.signature(mod.show).bind(*args, **kwargs)
            param_binding.apply_defaults()
            bound_args = param_binding.arguments

            trace.show(style=bound_args["style"])

    ########## Lookup ##########

    @type_checked
    def get(self, rand_var_or_sref: RAND_VAR_TYPE | StmtSRef) -> int | SBlock | For | None:
        """Returns:
        - the corresponding SBlock that a SBlockRV evaluates to;
        - the corresponding For that a LoopRV evaluates to;
        - the corresponding integer that a ExprRV evaluates to;
        - the corresponding SBlock that a SBlock sref points to;
        - the corresponding For that a loop sref points to;

        Parameters
        ----------
        rand_var_or_sref : ExprRV | SBlockRV | LoopRV | StmtSRef
            The random variable / sref to be evaluated

        Returns
        -------
        result : Optional[int | SBlock | For]
            The corresponding result
        """
        if isinstance(rand_var_or_sref, StmtSRef):
            return rand_var_or_sref.stmt
        # pylint: disable-next=no-member
        result = _ffi_api.ScheduleGet(self, rand_var_or_sref)  # type: ignore
        if isinstance(result, IntImm):
            result = result.value
        return result

    @type_checked
    def get_sref(self, rand_var_or_stmt: SBlockRV | LoopRV | SBlock | For) -> StmtSRef | None:
        """Returns the corresponding sref to the given
        1) LoopRV
        2) SBlockRV
        3) Block
        4) For

        Parameters
        ----------
        rand_var_or_stmt : SBlockRV | LoopRV | SBlock | For
            The random variable / sref to be evaluated

        Returns
        -------
        result : Optional[StmtSRef]
            The corresponding result
        """
        return _ffi_api.ScheduleGetSRef(  # type: ignore # pylint: disable=no-member
            self, rand_var_or_stmt
        )

    @type_checked
    def remove_rv(self, rand_var: RAND_VAR_TYPE) -> None:
        """Remove a random variable from the symbol table

        Parameters
        ----------
        rand_var : SBlockRV | LoopRV | ExprRV
            The random variable to be removed
        """
        return _ffi_api.ScheduleRemoveRV(self, rand_var)  # type: ignore # pylint: disable=no-member

    ########## Schedule: Sampling ##########

    @type_checked
    def sample_categorical(
        self, candidates: list[int], probs: list[float], decision: int | None = None
    ) -> ExprRV:
        """Sample an integer given the probability distribution

        Parameters
        ----------
        candidates : List[int]
            The candidates to be sampled from
        probs : List[float]
            The probability of each candidate
        decision : Optional[int]
            The sampling decision, if any

        Returns
        -------
        result : ExprRV
            The random variable sampled from candidates
        """
        return _ffi_api.ScheduleSampleCategorical(  # type: ignore # pylint: disable=no-member
            self, candidates, probs, decision
        )

    @type_checked
    def sample_perfect_tile(
        self,
        loop: LoopRV,
        n: int,
        max_innermost_factor: int = 16,
        decision: list[int] | None = None,
    ) -> list[ExprRV | None]:
        """Sample the factors to perfect tile a specific loop

        Parameters
        ----------
        loop : LoopRV
            The loop to be tiled
        n : int
            The number of tiles to be sampled
        max_innermost_factor : int
            The maximum tile size allowed to be sampled in the innermost loop
        decision: Optional[List[int]]
            The sampling decision, if any

        Returns
        -------
        result : List[Optional[ExprRV]]
            A list of length `n`, the random perfect tile sizes sampled.
            A dynamic factor is represented by None.
        """
        return list(
            _ffi_api.ScheduleSamplePerfectTile(  # type: ignore  # pylint: disable=no-member
                self, loop, n, max_innermost_factor, decision
            )
        )

    @type_checked
    def sample_partitioned_tile(
        self,
        loop: LoopRV,
        n: int,
        partition_pos: int = 0,
        innerpart_factor: int = 1,
        decision: list[int] | None = None,
    ) -> list[ExprRV]:
        """Sample the factors to a partitioned tile for a specific loop

        Parameters
        ----------
        loop : LoopRV
            The loop to be tiled
        n : int
            The number of tiles to be sampled
        partition_pos : int
            The position to partition tiles to two parts
        innerpart_factor : int
            The factor of the second part
        decision: Optional[List[int]]
            The sampling decision, if any

        Returns
        -------
        result : List[ExprRV]
            A list of length `n`, the random partitioned tile sizes sampled
        """
        return list(
            _ffi_api.ScheduleSamplePartitionedTile(  # type: ignore  # pylint: disable=no-member
                self,
                loop,
                n,
                partition_pos,
                innerpart_factor,
                decision,
            )
        )

    @type_checked
    def sample_compute_location(self, block: SBlockRV | str, decision: int | None = None) -> LoopRV:
        """Sample a compute-at location of the given block

        Parameters
        ----------
        block : SBlockRV | str
            The block whose compute-at location is to be sampled
        decision : Optional[int]
            The sampling decision

        Returns
        -------
        result : LoopRV
            The sampled loop where the input block is to be computed at
        """
        block = self._normalize_block_arg(block)

        return _ffi_api.ScheduleSampleComputeLocation(  # type: ignore  # pylint: disable=no-member
            self, block, decision
        )

    ########## Schedule: Get blocks & loops ##########
    @type_checked
    def get_sblock(self, name: str, func_name: str | None = None) -> SBlockRV:
        """Retrieve a block in a specific function with its name

        By default, if `func_name` is not specified, the schedule will search for the block in the
        function that is currently being "worked on". To switch the function to be worked on, use
        `work_on` before calling this method.

        Parameters
        ----------
        name : str
            The name of the block
        func_name : Optional[str] = None
            The name of the function

        Returns
        -------
        block : SBlockRV
            The block retrieved
            IndexError is raised if 0 or multiple blocks exist with the specific name.
        """
        return _ffi_api.ScheduleGetSBlock(  # type: ignore # pylint: disable=no-member
            self, name, func_name
        )

    @type_checked
    def get_loops(self, block: SBlockRV | str) -> list[LoopRV]:
        """Get the parent loops of the block in its scope, from outer to inner

        Parameters
        ----------
        block : SBlockRV | str
            The query block

        Returns
        -------
        loops : List[LoopRV]
            A list of loops above the given block in its scope, from outer to inner
        """
        block = self._normalize_block_arg(block)
        # pylint: disable-next=no-member
        return list(_ffi_api.ScheduleGetLoops(self, block))  # type: ignore

    @type_checked
    def get_child_blocks(self, block_or_loop: SBlockRV | LoopRV) -> list[SBlockRV]:
        """Get the leaf blocks of a specific block/loop

        Parameters
        ----------
        block_or_loop : SBlockRV | LoopRV
            The query block/loop

        Returns
        -------
        blocks : List[LoopRV]
            A list of leaf blocks inside a specific block/loop
        """
        # pylint: disable-next=no-member
        return list(_ffi_api.ScheduleGetChildBlocks(self, block_or_loop))  # type: ignore

    @type_checked
    def get_producers(self, block: SBlockRV | str) -> list[SBlockRV]:
        """Get the producers of a specific block

        Parameters
        ----------
        block : SBlockRV | str
            The block in the query

        Returns
        -------
        producers : List[SBlockRV]
            A list of producers of the given block
        """
        block = self._normalize_block_arg(block)
        # pylint: disable-next=no-member
        return list(_ffi_api.ScheduleGetProducers(self, block))  # type: ignore

    @type_checked
    def get_consumers(self, block: SBlockRV | str) -> list[SBlockRV]:
        """Get the consumers of a specific block

        Parameters
        ----------
        block : SBlockRV | str
            The block in the query

        Returns
        -------
        consumers : List[SBlockRV]
            A list of consumers of the given block
        """
        block = self._normalize_block_arg(block)
        # pylint: disable-next=no-member
        return list(_ffi_api.ScheduleGetConsumers(self, block))  # type: ignore

    @type_checked
    def get_output_blocks(self, scope_block: SBlockRV | str) -> list[SBlockRV]:
        """Get the list of output blocks within the given scope
        An output block is a block which has atleast one tensor being written
        to, but is not allocated within the Function

        Parameters
        ----------
        scope_block : SBlockRV | str,
            The scope block from which output blocks are collected

        Returns
        -------
        output_blocks : List[SBlockRV]
            A list of all blocks that write to some output tensor

        """
        scope_block = self._normalize_block_arg(scope_block)
        # pylint: disable-next=no-member
        return list(_ffi_api.ScheduleGetOutputBlocks(self, scope_block))  # type: ignore

    ########## Schedule: Transform loops ##########
    @type_checked
    def merge(self, *loops: list[LoopRV]) -> LoopRV:
        """Merge a list of loops into one. The loops under their LCA requires:
        1) Under the same scope.
        2) Can't have annotations or thread bindings.
        3) Start with 0 and have same extent and same nesting depth.
        4) From target loop to their LCA, The inner loop must be the only child of the outer loop.

        Parameters
        ----------
        *loops : List[LoopRV]
            The loops to be merged

        Returns
        -------
        fused_loop : LoopRV
            The new loop after merge

        Examples
        --------

        Before applying merge, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_merge(
                A: T.Tensor((128, 128)), B: T.Tensor((128, 128)), C: T.Tensor((128, 128))
            ) -> None:

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do fuse:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_fuse)
            i1, _ = sch.get_loops(sch.get_sblock("B"))
            i2, _ = sch.get_loops(sch.get_sblock("C"))
            sch.merge(i1, i2)
            print(sch.mod["main"].script())

        After applying fuse, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_fuse(
                A: T.Tensor((128, 128)), B: T.Tensor((128, 128)), C: T.Tensor((128, 128))
            ) -> None:

                # the 2 loops are merged into 1
                for i_m in range(128):
                    for j in range(128):
                        with Ts.sblock("B"):
                            vi, vj = Ts.axis.remap("SS", [i_m, j])
                            Ts.reads(A[vi, vj])
                            Ts.writes(B[vi, vj])
                            B[vi, vj] = A[vi, vj] * T.float32(2)
                    for j in range(128):
                        with Ts.sblock("C"):
                            vi, vj = Ts.axis.remap("SS", [i_m, j])
                            Ts.reads(A[vi, vj])
                            Ts.writes(C[vi, vj])
                            C[vi, vj] = A[vi, vj] * T.float32(2)

        """
        return _ffi_api.ScheduleMerge(self, loops)  # type: ignore # pylint: disable=no-member

    @type_checked
    def fuse(self, *loops: list[LoopRV], preserve_unit_iters: bool = True) -> LoopRV:
        """Fuse a list of consecutive loops into one. It requires:

        1) The loops can't have annotations or thread bindings.
        2) The (i+1)-th loop must be the only child of the i-th loop.
        3) All loops must start with 0.
        4) The domain of a loop to be fused cannot depend on another loop to be fused.

        Parameters
        ----------
        *loops : List[LoopRV]
            The loops to be fused

        Returns
        -------
        fused_loop : LoopRV
            The new loop after fusion

        Examples
        --------

        Before applying fuse, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_fuse(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do fuse:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_fuse)
            i, j = sch.get_loops(sch.get_sblock("B"))
            sch.fuse(i, j)
            print(sch.mod["main"].script())

        After applying fuse, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_fuse(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                # the 2 loops are fused into 1
                for i_j_fused in T.serial(0, 16384):
                    with Ts.sblock("B"):
                        vi = Ts.axis.S(128, T.floordiv(i_j_fused, 128))
                        vj = Ts.axis.S(128, T.floormod(i_j_fused, 128))
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        # pylint: disable-next=no-member
        return _ffi_api.ScheduleFuse(self, loops, preserve_unit_iters)  # type: ignore

    @type_checked
    def split(
        self,
        loop: LoopRV,
        factors: list[int | ExprRV | None],
        preserve_unit_iters: bool = True,
        disable_predication: bool = False,
    ) -> list[LoopRV]:
        """Split a loop into a list of consecutive loops. It requires:

        - The loop can't have annotation or thread binding.
        - The loop must start with 0.

        Predicates may be added to ensure the total loop numbers keeps unchanged.
        In `factors`, at most one of the factors can be None,
        which will be automatically inferred.

        Parameters
        ----------
        loop : LoopRV
            The loop to be split

        factors: List[int | ExprRV | None]
            The splitting factors
            Potential inputs are:

            - None
            - ExprRV
            - Positive constant integers

        preserve_unit_iters : bool
            Whether or not to preserve unit iterators in block bindings

        disable_predication : bool
            If enabled, don't create a predicate for guarding the loop. This can
            be useful when splitting with scalable factors that the schedule writer
            knows are divisible by the loop bound.

            Warning: enabling this feature may result in incorrect code generation
            if not used carefully.

        Returns
        -------
        split_loops : List[LoopRV]
            The new loops after split

        Examples
        --------

        Before split, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_split(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do split:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_split)
            i, j = sch.get_loops(sch.get_sblock("B"))
            sch.split(i, factors=[2, 64])
            print(sch.mod["main"].script())

        After applying split, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_split(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                # the original loop is split into 2 loops
                for i0, i1, j in T.grid(2, 64, 128):
                    with Ts.sblock("B"):
                        vi = Ts.axis.S(128, i0 * 64 + i1)
                        vj = Ts.axis.S(128, j)
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        # it will be checked later in C++ implementation
        # that there is at most one None in `factors`
        return list(
            _ffi_api.ScheduleSplit(  # type: ignore # pylint: disable=no-member
                self,
                loop,
                factors,
                preserve_unit_iters,
                disable_predication,
            )
        )

    @type_checked
    def loop_partition(
        self,
        loop: LoopRV,
        factors: list[int | ExprRV | None],
        preserve_unit_iters: bool = True,
    ) -> list[LoopRV]:
        """Partition a loop into a list of consecutive loops. It requires:

        1) The loop can't have annotation or thread binding.
        2) The loop can't be a reduction loop of a block that still has an init statement,
           because every partition would re-run the init and discard the earlier partial results.
           Use `decompose_reduction` first to separate the init from the update block.

        Predicates may be added to ensure the total loop numbers keeps unchanged.
        In `factors`, at most one of the factors can be None,
        which will be automatically inferred.

        Parameters
        ----------
        loop : LoopRV
            The loop to be partition

        factors: List[int | ExprRV | None]
            The partitioning factors
            Potential inputs are:
            - None
            - ExprRV
            - Positive constant integers

        preserve_unit_iters : bool
            Whether or not to preserve unit iterators in block bindings

        Returns
        -------
        partition_loops : List[LoopRV]
            The new loops after partition

        Examples
        --------

        Before partition, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_partition(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do partition:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_partition)
            i, j = sch.get_loops(sch.get_sblock("B"))
            sch.partition(i, factors=[2, 64])
            print(sch.mod["main"].script())

        After applying partition, the IR becomes:

        .. code-block:: python

            def after_partition(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                # the original loop is partition into 3 loops
                with Ts.sblock("root"):
                    Ts.reads()
                    Ts.writes()
                    with Ts.sblock("B_i_common"):
                        Ts.reads()
                        Ts.writes()
                        with Ts.sblock("B_i0_partition"):
                            Ts.reads()
                            Ts.writes()
                            for i0, j in T.grid(2, 128):
                                with Ts.sblock("B_i0"):
                                    vi, vj = Ts.axis.remap("SS", [i0, j])
                                    Ts.reads(A[0:2, 0:128])
                                    Ts.writes(B[0:2, 0:128])
                                    B[vi, vj] = A[vi, vj] * T.float32(2)
                        with Ts.sblock("B_i1_partition"):
                            Ts.reads()
                            Ts.writes()
                            for i1 in range(2, 66):
                                for j in range(128):
                                    with Ts.sblock("B_i1"):
                                        vi, vj = Ts.axis.remap("SS", [i1, j])
                                        Ts.reads(A[2:66, 0:128])
                                        Ts.writes(B[2:66, 0:128])
                                        B[vi, vj] = A[vi, vj] * T.float32(2)
                        with Ts.sblock("B_partition_2"):
                            Ts.reads()
                            Ts.writes()
                            for i2 in range(66, 128):
                                for j in range(128):
                                    with Ts.sblock("B_i2"):
                                        vi, vj = Ts.axis.remap("SS", [i2, j])
                                        Ts.reads(A[66:128, 0:128])
                                        Ts.writes(B[66:128, 0:128])
                                        B[vi, vj] = A[vi, vj] * T.float32(2)
        """
        return list(
            _ffi_api.ScheduleLoopPartition(  # type: ignore # pylint: disable=no-member
                self, loop, factors, preserve_unit_iters
            )
        )

    @type_checked
    def reorder(self, *ordered_loops: list[LoopRV]) -> None:
        """
        Reorder a list of loops. It doesn't require the loops to be consecutive.
        It requires:

        1) The loops are in the same chain. That means: the loops can be ordered to [l_1, l_2, ... ,
        l_n] where l_i is an ancestor of l_{i+1} and there are only single-branch loops between
        l_1 and l_n (which also indicates they are under the same scope).
        2) After reordering, the domain of an outer loop cannot depend on any of the inner loops.
        3) For every block under the loop nests, its block binding must be affine, and the block
        variables must be either data parallel or reduction.
        4) No duplicated loops are allowed in the arguments.

        Parameters
        ----------
        *ordered_loops : List[LoopRV]
            The loops in the new order

        Examples
        --------

        Before reorder, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_reorder(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do reorder:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_reorder)
            i, j = sch.get_loops(sch.get_sblock("B"))
            sch.reorder(j, i)
            print(sch.mod["main"].script())

        After applying reorder, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_reorder(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                # Here j and i are reordered
                for j, i in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleReorder(self, ordered_loops)  # type: ignore # pylint: disable=no-member

    @type_checked
    def reorder_block_iter_var(self, block: SBlockRV, new_order: list[int]) -> None:
        """Reorder the itervars inside a given block.

        Parameters
        ----------
        block : SBlockRV
            The block to be transformed.
        new_order : List[int]
            The new block itervar order.

        Examples
        --------

        Before reorder_block_iter_var, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def matmul(
                A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32"),
                C: T.Tensor((128, 128), "float32"),
            ) -> None:
                for i, j, k in T.grid(128, 128, 128):
                    with Ts.sblock("C"):
                        vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                        with Ts.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        Create the schedule and do reorder_block_iter_var:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(matmul)
            C = sch.get_sblock("C")
            sch.reorder_block_iter_var(C, [2, 1, 0])

        After applying reorder_block_iter_var, the IR becomes:

        .. code-block:: python

            @Ts.function
            def matmul_after_reorder_block_iter_var(
                A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32"),
                C: T.Tensor((128, 128), "float32"),
            ):
                for i, j, k in T.grid(128, 128, 128):
                    with Ts.sblock("C"):
                        vk, vj, vi = Ts.axis.remap("RSS", [k, j, i])
                        Ts.reads(A[vi, vk], B[vj, vk])
                        Ts.writes(C[vi, vj])
                        with Ts.init():
                            C[vi, vj] = T.float32(0)
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        See Also
        --------
        reorder
        """
        # pylint: disable-next=no-member
        _ffi_api.ScheduleReorderBlockIterVar(self, block, new_order)  # type: ignore

    @type_checked
    def add_unit_loop(self, block_or_loop: LoopRV | SBlockRV) -> LoopRV:
        """Create a new unit loop on top of the specific block or loop.

        Parameters
        ----------
        block_or_loop : LoopRV | SBlockRV
            The block above which the new loop is created

        Returns
        -------
        new_loop : LoopRV
            The new unit loop

        Examples
        --------

        Before add_unit_loop, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_add_unit_loop(
                A: T.Tensor((), "int32"),
                B: T.Tensor((), "int32"),
                C: T.Tensor((), "int32"),
            ) -> None:
                with Ts.sblock("C"):
                    vi = Ts.axis.spatial(1, 0)
                    C[()] = A[()] + B[()]

        Create the schedule and do add-unit-loop:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_add_unit_loop)
            sch.add_unit_loop(sch.get_sblock("C"))
            print(sch.mod["main"].script())

        After applying add-unit-loop, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_add_unit_loop(
                A: T.Tensor((), "int32"),
                B: T.Tensor((), "int32"),
                C: T.Tensor((), "int32"),
            ) -> None:
                for u in T.serial(1):
                    with Ts.sblock("C"):
                        vi = Ts.axis.spatial(1, 0)
                        C[()] = A[()] + B[()]
        """
        # pylint: disable-next=no-member
        return _ffi_api.ScheduleAddUnitLoop(self, block_or_loop)  # type: ignore

    ########## Schedule: Manipulate ForKind ##########

    @type_checked
    def parallel(self, loop: LoopRV) -> None:
        """Parallelize the input loop. It requires:

        - The scope block that the loop is in should have stage-pipeline property.
        - All the blocks under the loop are complete blocks or reduction blocks, and have affine
          bindings.
        - For each block under the loop, the loop can only be contained in data-parallel block
          iters' bindings.

        Parameters
        ----------
        loop : LoopRV
            The loop to be parallelized

        Examples
        --------

        Before parallel, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_parallel(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do parallel:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_parallel)
            i, j = sch.get_loops(sch.get_sblock("B"))
            sch.parallel(i)

        After applying parallel, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_parallel(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i in T.parallel(0, 128):
                    for j in T.serial(0, 128):
                        with Ts.sblock("B"):
                            vi, vj = Ts.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleParallel(self, loop)  # type: ignore # pylint: disable=no-member

    @type_checked
    def vectorize(self, loop: LoopRV) -> None:
        """Vectorize the input loop. It requires:

        - The scope block that the loop is in should have stage-pipeline property.
        - All the blocks under the loop are complete blocks or reduction blocks, and have affine
          bindings.
        - For each block under the loop, the loop can only be contained in data-parallel block
          iters' bindings.

        Parameters
        ----------
        loop : LoopRV
            The loop to be vectorized

        Examples
        --------

        Before vectorize, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_vectorize(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do vectorize:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_vectorize)
            i, j = sch.get_loops(sch.get_sblock("B"))
            sch.vectorize(j)

        After applying vectorize, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_vectorize(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i in T.serial(0, 128):
                    for j in T.vectorized(0, 128):
                        with Ts.sblock("B"):
                            vi, vj = Ts.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleVectorize(self, loop)  # type: ignore # pylint: disable=no-member

    @type_checked
    def bind(self, loop: LoopRV, thread_axis: str) -> None:
        """Bind the input loop to the given thread axis. It requires:

        - The scope block that the loop is in should have stage-pipeline property.
        - All the blocks under the loop are complete blocks or reduction blocks, and have affine
          bindings.
        - For each block under the loop, if the thread axis starts with ``threadIdx``, the loop can
          only be contained in data-parallel block iter and reduction block iters' bindings.
          Otherwise the loop can only be contained in data-parallel block iters' bindings.

        Parameters
        ----------
        loop : LoopRV
            The loop to be bound to the thread axis
        thread_axis : str
            The thread axis to be bound to the loop. Possible candidates are ``blockIdx.x/y/z``,
            ``threadIdx.x/y/z``, ``vthread.x/y/z``, and ``vthread``. The ``vthread`` value is a
            legacy behavior that will be deprecated. Please use ``vthread.x/y/z`` instead.

        Examples
        --------

        Before bind, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_bind(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do bind:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_bind)
            i, j = sch.get_loops(sch.get_sblock("B"))
            sch.bind(i, "blockIdx.x")
            sch.bind(j, "threadIdx.x")

        After applying bind, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_bind(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i in T.parallel(0, 128, annotations={"thread_binding": "blockIdx.x"}):
                    for j in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
                        with Ts.sblock("B"):
                            vi, vj = Ts.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleBind(self, loop, thread_axis)  # type: ignore # pylint: disable=no-member

    @type_checked
    def unroll(self, loop: LoopRV) -> None:
        """Unroll the input loop. It requires nothing

        Parameters
        ----------
        loop : LoopRV
            The loop to be unrolled

        Examples
        --------

        Before unroll, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_unroll(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do unroll:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_unroll)
            i, j = sch.get_loops(sch.get_sblock("B"))
            sch.unroll(i)

        After applying unroll, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_unroll(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i in T.unroll(0, 128):
                    for j in T.serial(0, 128):
                        with Ts.sblock("B"):
                            vi, vj = Ts.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleUnroll(self, loop)  # type: ignore # pylint: disable=no-member

    ########## Schedule: Insert cache stages ##########

    @type_checked
    def cache_read(
        self,
        block: SBlockRV | str,
        read_tensor_index: int | str | Var,
        storage_scope: str | MemoryScope,
        consumer_blocks: list[SBlockRV | str] | None = None,
    ) -> SBlockRV:
        """Create a block that reads a tensor region into a read cache. It requires:


        1) There is at most one block who write the tensor in the scope.

        2) The scope block have stage-pipeline property.

        Parameters
        ----------
        block : SBlockRV | str
            The consumer block of the target tensor.

        tensor: int | str | Var
            The index of the tensor in block's read region, the unique
            name of a read tensor in the block, or a Var object
            that is within the blocks read region.

        storage_scope: str | MemoryScope
            The target storage scope.

        consumer_blocks: Optional[List[SBlockRV | str]]
            An optional list of consumers that should read from the cache. If not specified,
            all consumers will use the cache.

        Returns
        -------
        cached_block : SBlockRV
            The block of the cache stage

        Examples
        --------
        Before cache_read, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_cache_read(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and cache_read:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_cache_read)
            block_b = sch.get_sblock("B")
            sch.cache_read(block_b, 0, "local")
            print(sch.mod["main"].script())

        After applying cache_read, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_cache_read(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                A_local = Ts.sblock_alloc_tensor((128, 128), scope="local")
                for i, j in T.grid(128, 128):
                    with Ts.sblock("A_local"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        A_local[vi, vj] = A[vi, vj]
                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A_local[vi, vj] * 2.0

        """
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        if consumer_blocks is None:
            consumer_blocks = []

        # Convert any string SBlock names into SBlock RVs.
        consumer_blocks = [self._normalize_block_arg(b) for b in consumer_blocks]
        block = self._normalize_block_arg(block)

        if not isinstance(read_tensor_index, int):
            _, read_tensor_index, _ = self._normalize_tensor_arg(
                block, read_tensor_index, required_tensor_type="read"
            )
        return _ffi_api.ScheduleCacheRead(  # type: ignore # pylint: disable=no-member
            self, block, read_tensor_index, storage_scope, consumer_blocks
        )

    @type_checked
    def cache_write(
        self,
        block: SBlockRV | str,
        write_tensor_index: int | str | Var,
        storage_scope: str | MemoryScope,
        consumer_blocks: list[SBlockRV | str] | None = None,
    ) -> SBlockRV:
        """Create a block that reads a tensor region into a write cache. It requires:


        1) There is only one block who write the tensor in the scope.

        2) The scope block have stage-pipeline property.

        Parameters
        ----------
        block : SBlockRV | str
            The producer block of the target tensor.

        write_tensor_index: int
            The index of the tensor in block's write region, the unique
            name of a write tensor in the block, or a Var object
            that is within the blocks write region.

        storage_scope: str | MemoryScope
            The target storage scope.

        consumer_blocks: Optional[List[SBlockRV | str]]
            An optional list of consumers that should read directly from the cache.
            If not specified, all consumers will read from the original tensor.

        Returns
        -------
        cached_block : SBlockRV
            The block of the cache stage

        Examples
        --------
        Before cache_write, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_cache_write(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and cache_write:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_cache_write)
            block_b = sch.get_sblock("B")
            sch.cache_write(block_b, 0, "local")
            print(sch.mod["main"].script())

        After applying cache_write, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_cache_write(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                B_local = Ts.sblock_alloc_tensor((128, 128), scope="local")
                for i, j in T.grid(128, 128):
                    with Ts.sblock("A_local"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B_local[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = B_local[vi, vj]

        """
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        if consumer_blocks is None:
            consumer_blocks = []

        # Convert any string SBlock names into SBlock RVs.
        consumer_blocks = [self._normalize_block_arg(b) for b in consumer_blocks]
        block = self._normalize_block_arg(block)

        if not isinstance(write_tensor_index, int):
            _, write_tensor_index, _ = self._normalize_tensor_arg(
                block, write_tensor_index, required_tensor_type="write"
            )
        return _ffi_api.ScheduleCacheWrite(  # type: ignore # pylint: disable=no-member
            self, block, write_tensor_index, storage_scope, consumer_blocks
        )

    @type_checked
    def reindex_cache_read(
        self,
        block: SBlockRV | str,
        read_tensor_index: int,
        storage_scope: str | MemoryScope,
        index_map: IndexMap | Callable,
    ) -> SBlockRV:
        """Create a block that reads a tensor region into a read cache using customized
        indices specified by index map. The read region of the tensor must be a single point.

        The cache stage block follows the original order of loops and block itervars in the block.
        If a block itervar does not appear in the tensor access region, it and its corresponding
        loop variables will be omitted. User can then use `transform_block_layout` primitive to
        reorder the block itervars and surrounding loops of the cache read/write block.

        Unlike `cache_read`, `reindex_cache_read` only supports single consumer, please use
        `cache_read` when there are multiple consumers.

        Parameters
        ----------
        block : SBlockRV
            The consumer block of the target tensor.
        read_tensor_index: int
            The index of the tensor in block's read region.
        storage_scope: str | MemoryScope
            The target storage scope.
        index_map: IndexMap | Callable
            User defined indices to access allocated cache tensor, maps from block iter vars.

        Returns
        -------
        cached_block : SBlockRV
            The block of the cache stage

        Examples
        --------
        Before reindex_cache_read, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_reindex_cache_read(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and reindex_cache_read:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_cache_read)
            block_b = sch.get_sblock("B")
            sch.reindex_cache_read(block_b, 0, "local", lambda vi, vj: (vj, vi))
            print(sch.mod["main"].script())

        After applying reindex_cache_read, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_reindex_cache_read(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                A_local = Ts.sblock_alloc_tensor((128, 128), scope="local")
                for i, j in T.grid(128, 128):
                    with Ts.sblock("A_local"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        A_local[vj, vi] = A[vi, vj]
                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A_local[vj, vi] * 2.0

        See Also
        --------
        reindex_cache_write
        transform_block_layout
        transform_layout
        cache_read
        reindex
        """
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        # Convert any string SBlock names into SBlock RVs.
        block = self._normalize_block_arg(block)

        if callable(index_map):
            index_map = IndexMap.from_func(
                index_map,
                index_dtype=_get_sblock_default_dtype(self.get(block)),
            )
        return _ffi_api.ScheduleReindexCacheRead(  # type: ignore # pylint: disable=no-member
            self, block, read_tensor_index, storage_scope, index_map
        )

    @type_checked
    def reindex_cache_write(
        self,
        block: SBlockRV | str,
        write_tensor_index: int,
        storage_scope: str | MemoryScope,
        index_map: Callable | IndexMap,
    ) -> SBlockRV:
        r"""Create a block that reads a tensor region into a write cache using customized
        indices specified by index map. The write region of the tensor must be a single point.

        The cache stage block follows the original order of loops and block itervars in the block.
        If a block itervar does not appear in the tensor access region, it and its corresponding
        loop variables will be omitted. User can then use `transform_block_layout` primitive to
        reorder the block itervars and surrounding loops of the cache read/write block.

        Unlike `cache_write`, `reindex_cache_write` only supports single consumer, please use
        `cache_write` when there are multiple consumers.

        Parameters
        ----------
        block : SBlockRV | str
            The consumer block of the target tensor.
        write_tensor_index: int
            The index of the tensor in block's write region.
        storage_scope: str | MemoryScope
            The target storage scope.
        index_map: Callable | IndexMap
            User defined indices to access allocated cache tensor, maps from block iter vars.
        consumer_blocks: Optional[List[SBlockRV | str]]
            An optional list of consumers that should read directly from the cache.
            If not specified, all consumers will read from the original tensor.

        Returns
        -------
        cached_block : SBlockRV
            The block of the cache stage

        Examples
        --------
        Before reindex_cache_write, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_reindex_cache_write(
                A: T.Tensor((128, 128)), B: T.Tensor((128, 128))
            ) -> None:

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and reindex_cache_write:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_cache_write)
            block_b = sch.get_sblock("B")
            sch.reindex_cache_write(block_b, 0, "local", lambda vi, vj: (vi // 2, vi % 2, vj))
            print(sch.mod["main"].script())

        After applying reindex_cache_write, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_cache_write(A: T.Tensor((128, 128)), B: T.Tensor((64, 2, 128))) -> None:


                B_local = Ts.sblock_alloc_tensor((128, 128), scope="local")
                for i, j in T.grid(128, 128):
                    with Ts.sblock("A_local"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B_local[vi % 2, vi // 2, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = B_local[vi % 2, vi // 2, vj]

        See Also
        --------
        reindex_cache_read
        transform_block_layout
        transform_layout
        cache_write
        reindex
        """
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        # Convert any string SBlock names into SBlock RVs.
        block = self._normalize_block_arg(block)

        if callable(index_map):
            index_map = IndexMap.from_func(
                index_map,
                index_dtype=_get_sblock_default_dtype(self.get(block)),
            )
        return _ffi_api.ScheduleReindexCacheWrite(  # type: ignore # pylint: disable=no-member
            self, block, write_tensor_index, storage_scope, index_map
        )

    @type_checked
    def cache_inplace(
        self,
        block: SBlockRV | str,
        read_tensor_index: int | str | Var,
        storage_scope: str | MemoryScope,
    ) -> list[SBlockRV]:
        """Create blocks that reads & write a tensor region into a cache block.
        It requires the target block both read & write the target tensor.
        Mainly for inplace operation.

        Parameters
        ----------
        block : SBlockRV | str
            The target block operates on the target tensor.

        read_tensor_index: int
            The index of the tensor in block's read region, the unique
            name of a read tensor in the block, or a Var object
            that is within the blocks read region.

        storage_scope: str | MemoryScope
            The target storage scope.


        Returns
        -------
        cached_blocks : List[SBlockRV]
            The blocks of the cache stage, read cache first, write cache second

        Examples
        --------
        Before cache_inplace, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_cache_inplace(data_io: T.Tensor((64), "int32")):
                for i0 in T.serial(1):
                    with Ts.sblock("A"):
                        Ts.reads(data_io[:64])
                        Ts.writes(data_io[:64])
                        T.evaluate(T.call_extern("call_impl", data_io.data, ty="void"))

        Create the schedule and cache_inplace:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_cache_inplace)
            block_a = sch.get_sblock("A")
            sch.cache_inplace(block_a, 0, "local")
            print(sch.mod["main"].script())

        After applying cache_inplace, the IR becomes:

        .. code-block:: python

            @Ts.function
            def cache_inplace(data_io: T.Tensor(64, "int32")) -> None:
                data_io_local = Ts.sblock_alloc_tensor([64], dtype="int32", scope="local")
                for i0 in T.serial(1):
                    for ax0 in T.serial(64):
                        with Ts.sblock("data_io_local"):
                            v0 = Ts.axis.spatial(64, ax0)
                            Ts.reads(data_io[v0])
                            Ts.writes(data_io_local[v0])
                            data_io_local[v0] = data_io[v0]
                    with Ts.sblock("A"):
                        Ts.reads(data_io_local[0 : 64])
                        Ts.writes(data_io_local[0 : 64])
                        T.evaluate(T.call_extern("call_impl", data_io_local.data, ty="void"))
                    for ax0 in T.serial(64):
                        with Ts.sblock("data_io_local"):
                            v0 = Ts.axis.spatial(64, ax0)
                            Ts.reads(data_io_local[v0])
                            Ts.writes(data_io[v0])
                            data_io[v0] = data_io_local[v0]

        """
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        block = self._normalize_block_arg(block)

        if not isinstance(read_tensor_index, int):
            _, read_tensor_index, _ = self._normalize_tensor_arg(
                block, read_tensor_index, required_tensor_type="read"
            )
        return _ffi_api.ScheduleCacheInplace(  # type: ignore # pylint: disable=no-member
            self, block, read_tensor_index, storage_scope
        )

    @type_checked
    def cache_index(
        self, block: SBlockRV | str, storage_scope: str | MemoryScope, cse_thresh: int = 0
    ) -> list[SBlockRV]:
        """Create a block to cache precomputed index for later use.
        if there is no index computation, keep unchanged.

        Parameters
        ----------
        block : SBlockRV | str
            The target block operates on the target tensor.

        storage_scope: str | MemoryScope
            The storage scope of cached block.

        cse_thresh: int
            The repeat threshold that determines a common sub expr,
            default 0 means cache all index computation.


        Returns
        -------
        cached_blocks : List[SBlockRV]
            The blocks of the stage writing the cache tensors

        Examples
        --------
        Before cache_inplace, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def resize(A: T.Tensor((1, 3, 40, 40)), B: T.Tensor((1, 3, 80, 80))) -> None:


                for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
                    with Ts.sblock("A"):
                        n, c, vi, vj = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                        B[n, c, vi, vj] = A[n, c, vi//4 + vj//4, vj//2]

        Create the schedule and cache_index:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(resize)
            block_a = sch.get_sblock("A")
            sch.cache_index(block_a, "global", 1)
            print(sch.mod["main"].script())

        After applying cache_index, the IR becomes:

        .. code-block:: python

            @Ts.function
            def resize_cache_index(
                A: T.Tensor((1, 3, 40, 40), "float32"), B: T.Tensor((1, 3, 80, 80), "float32")
            ) -> None:
                index_var_0 = Ts.sblock_alloc_tensor([80, 80], dtype="int32")
                index_var_1 = Ts.sblock_alloc_tensor([80], dtype="int32")
                for ax0, ax1 in T.grid(80, 80):
                    with Ts.sblock("index_0"):
                        v0 = Ts.axis.spatial(80, ax0)
                        v1 = Ts.axis.spatial(80, ax1)
                        Ts.reads()
                        Ts.writes(index_var_0[v0, v1])
                        index_var_0[v0, v1] = v0 // 4 + v1 // 4
                for ax0 in T.serial(80):
                    with Ts.sblock("index_1"):
                        v0 = Ts.axis.spatial(80, ax0)
                        Ts.reads()
                        Ts.writes(index_var_1[v0])
                        index_var_1[v0] = v0 // 2
                for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
                    with Ts.sblock("A"):
                        n, c, vi, vj = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                        Ts.reads(A[n, c, vi // 4 + vj // 4, vj // 2])
                        Ts.writes(B[n, c, vi, vj])
                        B[n, c, vi, vj] = A[n, c, index_var_0[vi, vj], index_var_1[vj]]

        """
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        block = self._normalize_block_arg(block)

        return _ffi_api.ScheduleCacheIndex(  # type: ignore # pylint: disable=no-member
            self, block, storage_scope, cse_thresh
        )

    @type_checked
    def reindex(self, block: SBlockRV | str, tensor: tuple[str, int] | str | Var) -> SBlockRV:
        """Create a block that read/write a tensor region into a read/write cache with reindexing.
        The layout of the cache will be the same as by the iterators of the block that reads/writes
        the tensor. It requires:

        1) There is only one block who reads/writes the target tensor
        2) There is only one tensor load/store of this tensor in the block

        Parameters
        ----------
        block : SBlockRV | str

            The block that accesses the target tensor.  If a string,
            this must uniquely identify a block.

        tensor: Union[Tuple[str,int], Var, str]

            The tensor to be transformed, or a specification of how to
            identify the tensor to be transformed.

            If `tensor` if a tuple of ``(str,int)``, the first item
            should be either "read" or "write", and the second item is
            an index into the block's read or write regions.

            If `tensor` is a string, it is the name of the tensor,
            which must exist within the reads/writes of the block.  In
            addition, the reads/writes of the block may not contain
            more than one tensor with this name.

            If `tensor` is a Var object, it must exist within the
            reads/writes of the block.

        Returns
        -------
        reindex_block : SBlockRV
            The block of the reindex stage

        Examples
        --------

        Before reindex, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_reindex(
                A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32")
            ) -> None:
                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vj, vi] * 2.0

        Create the schedule and do reindex:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_reindex)
            block = sch.get_sblock("B")
            sch.reindex(block, ("read", 0))

        After applying reindex, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_reindex(
                A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32")
            ) -> None:
                A_reindex = Ts.sblock_alloc_tensor((128, 128), "float32")
                for i, j in T.grid(128, 128):
                    with Ts.sblock("A_reindex"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        A_reindex[vi, vj] = A[vj, vi]
                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A_reindex[vi, vj] * 2.0

        """
        block = self._normalize_block_arg(block)
        tensor_index_type, tensor_index, _ = self._normalize_tensor_arg(block, tensor)
        assert tensor_index_type in ["read", "write"], "Invalid tensor_index_type"
        tensor_index_type_enum = 0 if tensor_index_type == "read" else 1
        return _ffi_api.ScheduleReIndex(  # type: ignore # pylint: disable=no-member
            self, block, tensor_index, tensor_index_type_enum
        )

    ########## Schedule: Data movement ##########

    def read_at(
        self,
        loop: LoopRV,
        block: SBlockRV,
        read_tensor_index: int,
        storage_scope: str | MemoryScope,
    ) -> SBlockRV:
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        return _ffi_api.ScheduleReadAt(  # type: ignore # pylint: disable=no-member
            self, loop, block, read_tensor_index, storage_scope
        )

    def write_at(
        self,
        loop: LoopRV,
        block: SBlockRV,
        write_tensor_index: int,
        storage_scope: str | MemoryScope,
    ) -> SBlockRV:
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        return _ffi_api.ScheduleWriteAt(  # type: ignore # pylint: disable=no-member
            self, loop, block, write_tensor_index, storage_scope
        )

    ########## Schedule: Compute location ##########

    @type_checked
    def compute_at(
        self,
        block: SBlockRV | str,
        loop: LoopRV,
        preserve_unit_loops: bool = False,
        index: int = -1,
    ) -> None:
        """Compute-At. Move a producer block under the specific loop, and regenerate the
        loops induced by the block so that the tensor region produced by the producer block could
        cover those regions consumed by its consumer blocks under the given loop. It requires:


        1) `block` and `loop` are under the same scope, `loop` is not the ancestor of `block`

        2) The scope block has stage-pipeline property

        3) The subtree of the scope block, where the given block is in, satisfies the compact
        dataflow condition. i.e. all the blocks in the scope block's subtree must be either
        complete block or reduction block

        4) The block is not an output block with regard to the scope block, i.e. the tensors written
        by the block are allocated under the scope block

        5) All the consumers of the block are under the given loop

        Parameters
        ----------
        block : SBlockRV | str
            The block to be moved

        loop: LoopRV
            The loop where the block to be moved under

        preserve_unit_loops: bool
            Whether to keep the trivial loops whose extents are 1

        index: int
            The block index of the loop body subtree blocks:
            - `index = -1` means inserted into the last possible insertion point;
            - `index = -2` means inserted into the first possible insertion point;
            - Otherwise, `index` is a nonnegative number that indicates the insertion point

        Examples
        --------

        Before compute-at, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_compute_at(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:

                B = Ts.sblock_alloc_tensor((128, 128), "float32")

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do compute-at:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_compute_at)
            block = sch.get_sblock("B")
            loop, _ = sch.get_loops(sch.get_sblock("C"))
            sch.compute_at(block, loop, preserve_unit_loops=False)
            print(sch.mod["main"].script())

        After applying compute-at, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_compute_at(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:

                B = Ts.sblock_alloc_tensor((128, 128), "float32")

                for i in T.serial(0, 128):
                    for j in T.serial(0, 128):
                        with Ts.sblock("B"):
                            vi, vj = Ts.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0
                    for j in T.serial(0, 128):
                        with Ts.sblock("C"):
                            vi, vj = Ts.axis.remap("SS", [i, j])
                            C[vi, vj] = B[vi, vj] + 1.0

        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleComputeAt(  # type: ignore # pylint: disable=no-member
            self, block, loop, preserve_unit_loops, index
        )

    @type_checked
    def reverse_compute_at(
        self,
        block: SBlockRV | str,
        loop: LoopRV,
        preserve_unit_loops: bool = False,
        index: int = -1,
    ) -> None:
        """Reverse-Compute-At. Move a consumer block under the specific loop, and regenerate the
        loops induced by the block so that the tensor region consumed by the consumer block could
        cover those regions produced by its producer blocks under the given loop. It requires:


        1) `block` and `loop` are under the same scope, `loop` is not the ancestor of `block`

        2) The scope block has stage-pipeline property

        3) The subtree of the scope block, where the given block is in, satisfies the compact
        dataflow condition. i.e. all the blocks in the scope block's subtree must be either
        complete block or reduction block

        4) All the producers of the block are under the given loop

        Parameters
        ----------
        block : SBlockRV | str
            The block to be moved

        loop: LoopRV
            The loop where the block to be moved under

        preserve_unit_loops: bool
            Whether to keep the trivial loops whose extents are 1

        index: int
            The block index of the loop body subtree blocks:
            - `index = -1` means inserted into the last possible insertion point;
            - `index = -2` means inserted into the first possible insertion point;
            - Otherwise, `index` is a nonnegative number that indicates the insertion point

        Examples
        --------

        Before reverse-compute-at, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_reverse_compute_at(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:

                B = Ts.sblock_alloc_tensor((128, 128), "float32")

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do reverse-compute-at:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_reverse_compute_at)
            block = sch.get_sblock("C")
            loop, _ = sch.get_loops(sch.get_sblock("B"))
            sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
            print(sch.mod["main"].script())

        After applying reverse-compute-at, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_reverse_compute_at(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:

                B = Ts.sblock_alloc_tensor((128, 128), "float32")

                for i in T.serial(0, 128):
                    for j in T.serial(0, 128):
                        with Ts.sblock("B"):
                            vi, vj = Ts.axis.remap("SS", [i, j])
                            B[vi, vj] = A[vi, vj] * 2.0
                    for j in T.serial(0, 128):
                        with Ts.sblock("C"):
                            vi, vj = Ts.axis.remap("SS", [i, j])
                            C[vi, vj] = B[vi, vj] + 1.0

        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleReverseComputeAt(  # type: ignore # pylint: disable=no-member
            self, block, loop, preserve_unit_loops, index
        )

    @type_checked
    def compute_inline(self, block: SBlockRV | str) -> None:
        """Inline a block into its consumer(s). It requires:


        1) The block is a complete non-root block, which only produces one tensor

        2) The block must not be the only leaf in the scope.

        3) The body of the block must be a TensorStore statement in
           the form of, ``A[i, j, k, ...] = ...`` where the indices of
           the LHS are all distinct atomic variables, and no variables
           other than those indexing variables are allowed in the
           statement.

        Parameters
        ----------
        block : SBlockRV | str
            The block to be inlined to its consumer(s)

        Examples
        --------

        Before compute-inline, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_inline(A: T.Tensor((128, 128)), C: T.Tensor((128, 128))) -> None:

                B = Ts.sblock_alloc_tensor((128, 128))

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do compute-inline:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_inline)
            sch.compute_inline(sch.get_sblock("B"))
            print(sch.mod["main"].script())

        After applying compute-inline, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_inline(A: T.Tensor((128, 128)), C: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = A[vi, vj] * 2.0 + 1.0

        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleComputeInline(self, block)  # type: ignore # pylint: disable=no-member

    @type_checked
    def reverse_compute_inline(self, block: SBlockRV | str) -> None:
        """Inline a block into its only producer. It requires:


        1) The block is a complete non-root block, which only produces and consumes one tensor

        2) The block must not be the only leaf in the scope.

        3) The only producer of the block is a read-after-write producer and a
           complete non-root block

        4) The body of the block must be a TensorStore statement in the form of,
           ``B[f(i, j, k, ...)] = g(i, j, k, A[i, j, k, ...] ...)`` where the
           indices of each `TensorLoad` on the RHS are all distinct atomic
           variables, and no variables other than those indexing variables are
           allowed in the statement.

        Parameters
        ----------
        block : SBlockRV | str
            The block to be inlined to its producer

        Examples
        --------

        Before reverse-compute-inline, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_inline(A: T.Tensor((128, 128)), C: T.Tensor((128, 128))) -> None:

                B = Ts.sblock_alloc_tensor((128, 128))

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do reverse-compute-inline:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_inline)
            sch.reverse_compute_inline(sch.get_sblock("C"))
            print(sch.mod["main"].script())

        After applying reverse-compute-inline, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_inline(A: T.Tensor((128, 128)), C: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = A[vi, vj] * 2.0 + 1.0

        """
        block = self._normalize_block_arg(block)
        # pylint: disable-next=no-member
        _ffi_api.ScheduleReverseComputeInline(self, block)  # type: ignore

    @type_checked
    def fuse_reduction_epilogue(
        self,
        reduction_block: SBlockRV | str,
        epilogue_block: SBlockRV | str,
    ) -> None:
        """Fuse an epilogue block into a reduction block.

        It requires:


        1) The reduction block is a complete reduction block
        2) The epilogue block only reads from the reduction block's output
        3) The epilogue matches one of the supported patterns:

           - Bias: ``output = reduction_result + bias``
           - BiasReLU: ``output = max(reduction_result + bias, 0)``
           - Clipping: ``output = min(max(reduction_result, lower), upper)``
             or their commutative variants

        .. warning::

            **Semantic Change for Non-Linear Epilogues (BiasReLU, Clipping):**

            For non-linear epilogues (BiasReLU and Clipping), fusion changes the
            computation semantics from post-reduction application to per-iteration
            application. This can lead to different numerical results.

            **Example with Clipping to [-5, 5] and inputs [6, -2]:**

            - **Post-reduction clipping** (original): ``clip(sum([6, -2])) = clip(4) = 4``
            - **Per-iteration clipping** (fused): ``acc=0 → clip(0+6)=5 → clip(5+(-2))=3``

            The fused version applies clipping at each reduction iteration, which
            may be an intended optimization for some models but can cause unexpected
            correctness issues if users are not aware of this behavior.

            For linear epilogues (Bias), fusion preserves exact numerical equivalence.

        Parameters
        ----------
        reduction_block : SBlockRV | str
            The reduction block (e.g., matmul)
        epilogue_block : SBlockRV | str
            The epilogue block to be fused (e.g., bias add, ReLU, clipping)

        Examples
        --------
        See :py:func:`test_tir_schedule_fuse_reduction_epilogue` for examples.
        """
        reduction_block = self._normalize_block_arg(reduction_block)
        epilogue_block = self._normalize_block_arg(epilogue_block)
        # pylint: disable-next=no-member
        _ffi_api.ScheduleFuseReductionEpilogue(self, reduction_block, epilogue_block)  # type: ignore

    ########## Schedule: Reduction ##########

    @type_checked
    def decompose_reduction(self, block: SBlockRV | str, loop: LoopRV) -> SBlockRV:
        """Decompose a reduction block into two separate blocks.

        a) The init block, which is translated from the init statement of the reduction block;

        b) The update block, which is the original block without init statement.

        The init block is inserted right before the given loop.

        The schedule primitive requires:

        1) The input block is a reduction block.

        2) The input loop is the ancestor of the block.

        3) The input loop is not lower than all the loops related to reduce block var.

        Parameters
        ----------
        block : SBlockRV | str
            The reduction block to be decomposed
        loop : LoopRV
            The loop above which the init block is inserted before.

        Returns
        -------
        init_block : SBlockRV
            The init block

        Examples
        --------
        Before decompose-reduction, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_decompose(
                A: T.Tensor([128, 128]), B: T.Tensor([128, 128]), C: T.Tensor([128, 128])
            ) -> None:

                for i, j, k in T.grid(128, 128, 128):
                    with Ts.sblock("C"):
                        vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                        with Ts.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        Create the schedule and do decompose-reduction with specified loop:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_decompose)
            C = sch.get_sblock("C")
            i, j, k = sch.get_loops(C)
            sch.decompose_reduction(C, i)
            print(sch.mod["main"].script())

        After applying decompose-reduction, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_decompose(
                A: Ts.Tensor([128, 128]), B: Ts.Tensor([128, 128]), C: Ts.Tensor([128, 128])
            ) -> None:

                for i in tirx.serial(128):
                    for j in tirx.serial(128):
                        with tirx.block([128, 128]) as [vi, vj]:
                            C[vi, vj] = 0.0
                for i, j, k in tirx.grid(128, 128, 128):
                    with tirx.block([128, 128, tirx.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        """
        block = self._normalize_block_arg(block)
        # pylint: disable-next=no-member
        return _ffi_api.ScheduleDecomposeReduction(self, block, loop)  # type: ignore

    @type_checked
    def rfactor(self, loop: LoopRV, factor_axis: int) -> SBlockRV:
        """Factorize an associative reduction block by the specified loop.

        An associative reduction cannot be parallelized directly,
        because it leads to potential race condition during accumulation.
        Alternatively, the reduction could be factorized on a loop with the following steps:
        - Step 1: evenly slice the reduction into `n` separate chunks, where `n` is the loop extent
        - Step 2: compute the chunks separately and write the result into `n` intermediate tensors;
        - Step 3: accumulate the `n` separate tensor into the result tensor.
        Note that the Step 2 above introduces opportunities for parallelization.

        RFactor is a schedule primitive that implements the transformation described above:
        Given a block that writes to tensor `B`, it factorizes a loop of extent `n`.

        For example, the pseudocode below accumulates `B[i] = sum(A[i, : , : ])`:

        .. code-block:: python

            for i in range(128):                    # loop i is a data parallel loop
                for j in range(128):                # loop j is a reduction loop
                    for k in range(128):            # loop k is a reduction loop
                        B[i] = B[i] + A[i, j, k]

        Suppose RFactor is applied on the innermost loop `k` and `factor_axis = 1`.
        RFactor then creates an intermediate tensor and two blocks.

        1. The intermediate tensor, or "rf-tensor" is a tensor of rank `ndim(B) + 1` and
        size `size(B) * n`, whose shape expands from `shape(B)` by adding an axis of `n`
        at the position specified by `factor_axis`. For example,

            * shape(B) = [1, 2, 3], factor_axis = 0  => shape(B_rf) = [n, 1, 2, 3]
            * shape(B) = [1, 2, 3], factor_axis = 1  => shape(B_rf) = [1, n, 2, 3]
            * shape(B) = [1, 2, 3], factor_axis = 2  => shape(B_rf) = [1, 2, n, 3]
            * shape(B) = [1, 2, 3], factor_axis = 3  => shape(B_rf) = [1, 2, 3, n]

        2. The rfactor block, or "rf-block", is a block that writes to the `rf-tensor` without
        accumulating over the loop `k`, i.e. the loop `k` is converted from a reduction loop
        to a data parallel loop. In our example, the rf-block is:

        .. code-block:: python

            B_rf = np.zeros((128, 128))     # the rf-tensor
            for k in range(128):            # loop k is converted to a data parallel loop
                for i in range(128):        # loop i is a data parallel loop (unchanged)
                    for j in range(128):    # loop j is a reduction loop (unchanged)
                        B_rf[i, k] = B_rf[i, k] + A[i, j, k]


        3. The write-back block, or `wb-block`, is a block that accumulates the rf-tensor into
        the result tensor. All the reduction loops are removed except the loop `k` for accumulation.
        In our example, the wb-block is:

        .. code-block:: python

            for i in range(128):            # loop i is a data parallel loop (unchanged)
                                            # loop j is removed because it is a reduction loop
                for k in range(128):        # loop k is a reduction loop (unchanged)
                    B[i] = B[i] + B_rf[i, k]


        Parameters
        ----------
        loop : LoopRV
            The loop outside block for which we want to do rfactor
        factor_axis : int
            The position where the new dimension is placed in the new introduced rfactor tensor

        Returns
        -------
        rf_block : SBlockRV
            The block which computes partial results over each slices (i.e., the first block
            as described in the above illustration)

        Examples
        --------

        Before rfactor, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_rfactor(A: T.Tensor((128, 128, 128)), B: T.Tensor((128,))) -> None:
                for ii, i, j in T.grid(128, 128, 128):
                    with Ts.sblock("B"):
                        vii, vi, vj = Ts.axis.remap("SRR", [ii, i, j])
                        with Ts.init():
                            B[vii] = 0.0
                        B[vii] = B[vii] + A[vii, vi, vj]

        Create the schedule and do rfactor:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_rfactor)
            _, _, k = sch.get_loops(sch.get_sblock("B"))
            sch.rfactor(k, 0)
            print(sch.mod["main"].script())

        After applying rfactor, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_rfactor(A: T.Tensor([128, 128, 128]), B: T.Tensor([128])) -> None:


                B_rf = Ts.sblock_alloc_tensor([128, 128])
                for i2, ii, i in T.grid(128, 128, 128):
                    with Ts.sblock("B_rf"):
                        vi2, vii, vi = Ts.axis.remap("SSR", [i2, ii, i])
                        with Ts.init():
                            B_rf[vi2, vii] = 0.0
                        B_rf[vi2, vii] = (B_rf[vi2, vii] + A[vii, vi, vi2])
                for ii, i2 in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vii, vi2 = Ts.axis.remap("SR", [ii, i2])
                        with Ts.init():
                            B[vii] = 0.0
                        B[vii] = B[vii] + B_rf[vi2, vii]


        Note
        ----

        Rfactor requires:
        1) `loop` has only one child block, and it is a reduction block;
        2) `loop` is a reduction loop, i.e. the loop variable is bound to only reduction variables
        in the block binding;
        3) `loop` is not parallelized, vectorized, unrolled or bound to any thread axis;
        4) The block scope that `loop` is in is a staged-pipeline;
        5) The outermost loop outside the reduction block should has the reduction block as its
        first child block;
        6) The outermost reduction loop should have only one child block;
        7) An unary extent loop that is not bound to any reduction or data parallel variables in
        the block binding should not appear under some reduction loop;
        8) The reduction block should write to only one tensor, and its init and body are both
        simple `TensorStore`s, and the pattern is registered as an associative reducer.
        The pre-defined patterns include: plus, multiplication, min and max;
        9) Each of the loops on top of the block cannot be bound to a data parallel and a
        reduction block binding at the same time;
        10) `factor_axis` should be in range `[-ndim(B) - 1, ndim(B)]`,
        where `B` is the tensor that the reduction block writes to.
        Negative indexing is normalized according to numpy convention.
        """
        # pylint: disable-next=no-member
        return _ffi_api.ScheduleRFactor(self, loop, factor_axis)  # type: ignore

    ######## Schedule: SBlock annotation ########

    @type_checked
    def storage_align(  # pylint: disable=too-many-arguments
        self, block: SBlockRV | str, tensor_index: int, axis: int, factor: int, offset: int
    ) -> None:
        """Set alignment requirement for specific dimension such that
        stride[axis] == k * factor + offset for some k. This is useful to set memory layout for more
        friendly memory access pattern. For example, we can set alignment to be factor=2, offset=1
        to avoid bank conflict for thread access on higher dimension in GPU shared memory.

        Parameters
        ----------
        block : SBlockRV | str
            The producer block of the tensor.
        tensor_index : int
            The index of the tensor in block's write region.
        axis : int
            The dimension to be specified for alignment.
        factor : int
            The factor multiple of alignment.
        offset : int
            The required offset factor.

        Examples
        --------

        Before storage_align, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_storage_align(A: T.Tensor((128, 128)), C: T.Tensor((128, 128))) -> None:

                B = Ts.sblock_alloc_tensor((128, 128))

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do storage_align:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_storage_align)
            sch.storage_align(sch.get_sblock("B"), tensor_index=0, axis=0, factor=128, offset=1)
            print(sch.mod["main"].script())

        After applying storage_align, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_storage_align(A: T.Tensor((128, 128)), C: T.Tensor((128, 128))) -> None:

                B = Ts.sblock_alloc_tensor((128, 128))

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        Ts.sblock_attr({"tensor_dim_align": [[[0, 128, 1]]]})
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        After lowering passes, tensor B will have strides as [129, 1].

        Note
        ----
        Storage_align requires the tensor to be an intermediate tensor defined via `alloc_tensor`.
        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleStorageAlign(  # type: ignore # pylint: disable=no-member
            self, block, tensor_index, axis, factor, offset
        )

    @type_checked
    def set_scope(
        self, block: SBlockRV | str, tensor_index: int | str | Var, storage_scope: str | MemoryScope
    ) -> None:
        """Set the storage scope of a tensor, where the tensor is
        specified by the a block and a write-index.

        Parameters
        ----------
        block : SBlockRV | str
            The producer block of the tensor
        tensor_index : int
            The index of the tensor in block's write region
        storage_scope : str or MemoryScope
            The storage scope to be set

        Examples
        --------

        Before set_scope, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_set_scope(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:
                B = Ts.sblock_alloc_tensor((128, 128), dtype="float32")

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do set_scope:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_set_scope)
            sch.set_scope(sch.get_sblock("B"), tensor_index=0, storage_scope="shared")
            print(sch.mod["main"].script())

        After applying set_scope, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_set_scope(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:
                B_shared = Ts.sblock_alloc_tensor([128, 128], dtype="float32", scope="shared")

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B_shared[vi, vj] = A[vi, vj] * T.float32(2)
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B_shared[vi, vj] + T.float32(1)

        Note
        ----
        `set_scope` requires the tensor to be an intermediate tensor defined via `alloc_tensor`.
        """
        if isinstance(storage_scope, MemoryScope):
            storage_scope = storage_scope.name
        block = self._normalize_block_arg(block)
        if not isinstance(tensor_index, int):
            _, tensor_index, _ = self._normalize_tensor_arg(
                block, tensor_index, required_tensor_type="write"
            )
        _ffi_api.ScheduleSetScope(  # type: ignore # pylint: disable=no-member
            self, block, tensor_index, storage_scope
        )

    @type_checked
    def unsafe_set_dtype(self, block: SBlockRV | str, tensor_index: int, dtype: str) -> None:
        """Set the data type of a tensor, where the tensor is
        specified by the a block and write-index.

        This schedule primitive is unsafe and may change the correctness of program because of
        type conversion, please use with caution.

        Parameters
        ----------
        block : SBlockRV | str
            The producer block of the tensor
        tensor_index : int
            The index of the tensor in block's write region
        dtype : str
            The data type to be set

        Examples
        --------

        Before unsafe_set_dtype, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_set_dtype(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:
                B = Ts.sblock_alloc_tensor((128, 128), dtype="float32")

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j]
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do unsafe_set_dtype:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_set_dtype)
            sch.unsafe_set_dtype("B", tensor_index=0, dtype="float16")
            print(sch.mod["main"].script())

        After applying set_dtype, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_set_dtype(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:
                B = Ts.sblock_alloc_tensor((128, 128), dtype="float16")

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = T.cast(A[vi, vj] * 2.0, ty="float16")
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j]
                        C[vi, vj] = T.cast(B[vi, vj], ty="float32") + 1.0

        Note
        ----
        `unsafe_set_dtype` requires the tensor to be an intermediate tensor defined via
        `alloc_tensor`.
        """
        block = self._normalize_block_arg(block)
        _ffi_api.ScheduleUnsafeSetDType(  # type: ignore # pylint: disable=no-member
            self, block, tensor_index, dtype
        )

    ########## Schedule: Blockize & Tensorize ##########

    @type_checked
    def blockize(
        self, target: LoopRV | list[SBlockRV], preserve_unit_iters: bool = True
    ) -> SBlockRV:
        """Convert multiple blocks or the subtree rooted at a specific loop into a block.

        Parameters
        ----------
        target : LoopRV or List[SBlockRV]
            The root of the subtree or the specified blocks.
        preserve_unit_iters : bool
            Whether or not to preserve unit iterators in block bindings

        Returns
        -------
        result : SBlockRV
            The new block.

        Examples
        --------

        Before blockize, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_blockize(
                A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32")
            ) -> None:
                for i_0, j_0, i_1, j_1 in T.grid(8, 8, 16, 16):
                    with Ts.sblock("B"):
                        vi = Ts.axis.spatial(128, i_0 * 16 + i_1)
                        vj = Ts.axis.spatial(128, j_0 * 16 + j_1)
                        Ts.reads(A[vi, vj])
                        Ts.writes(B[vi, vj])
                        B[vi, vj] = A[vi, vj] * T.float32(2)

        Create the schedule and do set_scope:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_blockize)
            B = sch.get_sblock("B")
            _, _, i1, _ = sch.get_loops(B)
            sch.blockize(i1)
            print(sch.mod["main"].script())

        After applying blockize, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_blockize(
                A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32")
            )-> None:
                for i_0, j_0 in T.grid(8, 8):
                    with Ts.sblock("B_o"):
                        vio, vjo = Ts.axis.remap("SS", [i_0, j_0])
                        Ts.reads(A[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
                        Ts.writes(B[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
                        for i_1, j_1 in T.grid(16, 16):
                            with Ts.sblock("B"):
                                vi, vj = Ts.axis.remap("SS", [i_1, j_1])
                                Ts.reads(A[vio * 16 + vi, vjo * 16 + vj])
                                Ts.writes(B[vio * 16 + vi, vjo * 16 + vj])
                                B[vio * 16 + vi, vjo * 16 + vj] = A[vio * 16 + vi, vjo * 16 + vj] \
                                                                  * T.float32(2)

        Note
        ----
        blockize requires there is exactly one block under the given loop and the bindings of the
        block are divisible by the subspace represented by the loops starting at the given loop.
        """

        # pylint: disable-next=no-member
        return _ffi_api.ScheduleBlockize(self, target, preserve_unit_iters)  # type: ignore

    @type_checked
    def tensorize(
        self,
        block_or_loop: SBlockRV | LoopRV,
        tensor_intrin: str,
        preserve_unit_iters: bool = True,
    ) -> None:
        """Tensorize the computation enclosed by loop with the tensor intrinsic.

        Parameters
        ----------
        block_or_loop : SBlockRV | LoopRV
            The loop to be tensorized.
        tensor_intrin : str
            The tensor intrin or the name of the tensor intrin.
        preserve_unit_iters : bool
            Whether or not to preserve unit iterators in block bindings

        Examples
        --------

        Before tensorize, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_tensorize(
                A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32"),
                C: T.Tensor((128, 128), "float32"),
            ) -> None:
                # body
                # with Ts.sblock("root")
                for i_0, j_0, k_0, i_1, j_1, k_1 in T.grid(8, 8, 8, 16, 16, 16):
                    with Ts.sblock("update"):
                        vi = Ts.axis.spatial(128, i_0 * 16 + i_1)
                        vj = Ts.axis.spatial(128, j_0 * 16 + j_1)
                        vk = Ts.axis.reduce(128, k_0 * 16 + k_1)
                        Ts.reads(C[vi, vj], A[vi, vk], B[vj, vk])
                        Ts.writes(C[vi, vj])
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        Declare and register the tensor intrinsic:

        .. code-block:: python

            @Ts.function
            def mma_desc(
                A: T.Tensor((16, 16), align=128),
                B: T.Tensor((16, 16), align=128),
                C: T.Tensor((16, 16), align=128),
            ) -> None:

                with Ts.sblock("root"):
                    Ts.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
                    Ts.writes(C[0:16, 0:16])
                    for i, j, k in T.grid(16, 16, 16):
                        with Ts.sblock("update"):
                            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

            @Ts.function
            def mma_intrin(
                A: T.Tensor((16, 16), align=128),
                B: T.Tensor((16, 16), align=128),
                C: T.Tensor((16, 16), align=128),
            ) -> None:

                with Ts.sblock("root"):
                    Ts.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
                    Ts.writes(C[0:16, 0:16])
                    T.evaluate(
                        T.gpu_mma_sync(
                            C.data,
                            C.layout.apply(0, 0)["m"] // 256,
                            A.data,
                            A.layout.apply(0, 0)["m"] // 256,
                            B.data,
                            B.layout.apply(0, 0)["m"] // 256,
                            C.data,
                            C.layout.apply(0, 0)["m"] // 256,
                            dtype="void",
                        )
                    )

            tvm.s_tir.TensorIntrin.register("test_mma_intrin", mma_desc, mma_intrin)

        Create the schedule and do tensorize:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_tensorize)
            update = sch.get_sblock("update")
            _, _, _, i1, _, _ = sch.get_loops(update)
            sch.tensorize(i1, "test_mma_intrin")
            print(sch.mod["main"].script())

        After applying tensorize, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_tensorize(
                A: T.Tensor((128, 128), "float32"),
                B: T.Tensor((128, 128), "float32"),
                C: T.Tensor((128, 128), "float32"),
            ) -> None:
                # body
                # with Ts.sblock("root")
                for i_0, j_0, k_0 in T.grid(8, 8, 8):
                    with Ts.sblock("update_o"):
                        vio, vjo, vko = Ts.axis.remap("SSR", [i_0, j_0, k_0])
                        Ts.reads(
                            C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                            A[vio * 16 : vio * 16 + 16, vko * 16 : vko * 16 + 16],
                            B[vjo * 16 : vjo * 16 + 16, vko * 16 : vko * 16 + 16],
                        )
                        Ts.writes(C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16])
                        A_1 = Ts.match_tensor(
                            A[vio * 16 : vio * 16 + 16, vko * 16 : vko * 16 + 16],
                            [16, 16],
                            dtype="float32",

                        )
                        B_1 = Ts.match_tensor(
                            B[vjo * 16 : vjo * 16 + 16, vko * 16 : vko * 16 + 16],
                            [16, 16],
                            dtype="float32",

                        )
                        C_1 = Ts.match_tensor(
                            C[vio * 16 : vio * 16 + 16, vjo * 16 : vjo * 16 + 16],
                            [16, 16],
                            dtype="float32",

                        )
                        T.evaluate(
                            T.gpu_mma_sync(
                                C_1.data,
                                C_1.layout.apply(0, 0)["m"] // 256,
                                A_1.data,
                                A_1.layout.apply(0, 0)["m"] // 256,
                                B_1.data,
                                B_1.layout.apply(0, 0)["m"] // 256,
                                C_1.data,
                                C_1.layout.apply(0, 0)["m"] // 256,
                                dtype="void",
                            )
                        )
        """
        _ffi_api.ScheduleTensorize(  # type: ignore # pylint: disable=no-member
            self, block_or_loop, tensor_intrin, preserve_unit_iters
        )

    ########## Schedule: Annotation ##########

    PrimAnnotationValueT = str | int | float | ExprRV
    AnnotationValueT = (
        PrimAnnotationValueT
        | list[PrimAnnotationValueT]
        | dict[str, PrimAnnotationValueT | list[PrimAnnotationValueT]]
    )

    @type_checked
    def annotate(
        self, block_or_loop: SBlockRV | LoopRV, ann_key: str, ann_val: AnnotationValueT
    ) -> None:
        """Annotate a block/loop with a key value pair

        Parameters
        ----------
        block_or_loop: SBlockRV | LoopRV
            The block/loop to be annotated
        ann_key : str
            The annotation key
        ann_val : AnnotationValueT
            The annotation value

        Examples
        --------

        Before annotate, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_annotate(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do annotate:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_annotate)
            sch.annotate(sch.get_sblock("B"), "ann_key", "ann_value")
            print(sch.mod["main"].script())

        After applying annotate, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_annotate(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        Ts.sblock_attr({"ann_key", "ann_value"})
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleAnnotate(  # type: ignore # pylint: disable=no-member
            self, block_or_loop, ann_key, ann_val
        )

    @type_checked
    def unannotate(self, block_or_loop: SBlockRV | LoopRV, ann_key: str) -> None:
        """Unannotate a block/loop's annotation with key ann_key

        Parameters
        ----------
        block_or_loop: SBlockRV | LoopRV
            The block/loop to be unannotated
        ann_key : str
            The annotation key

        Examples
        --------

        Before unannotate, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_unannotate(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        Ts.sblock_attr({"ann_key", "ann_value"})
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do annotate:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_unannotate)
            sch.unannotate(sch.get_sblock("B"), "ann_key")
            print(sch.mod["main"].script())

        After applying unannotate, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_unannotate(A: T.Tensor((128, 128)), B: T.Tensor((128, 128))) -> None:


                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        """
        _ffi_api.ScheduleUnannotate(  # type: ignore # pylint: disable=no-member
            self, block_or_loop, ann_key
        )

    ########## Schedule: Layout transformation ##########

    def _normalize_block_arg(self, block: SBlockRV | str) -> SBlockRV:
        if isinstance(block, str):
            return self.get_sblock(block)

        return block

    def _normalize_tensor_arg(
        self,
        block: SBlockRV,
        tensor: tuple[str, int] | int | str | Var,
        required_tensor_type=None,
    ) -> tuple[str, int, Var]:
        block_obj: SBlock = self.get(block)
        block_name = block_obj.name_hint

        def iter_tensors():
            for i, read in enumerate(block_obj.reads):
                yield "read", i, read.source
            for i, write in enumerate(block_obj.writes):
                yield "write", i, write.source

        if isinstance(tensor, int):
            tensor = (required_tensor_type, tensor)

        if isinstance(tensor, str):
            possible_tensors = {}
            # String lookup requires ensuring that the name is unique
            for tensor_index_type, tensor_index, candidate_tensor in iter_tensors():
                if candidate_tensor.name == tensor:
                    possible_tensors[candidate_tensor] = (tensor_index_type, tensor_index)

            assert possible_tensors, f"Could not find tensor '{tensor}' in block '{block_name}'"
            assert len(possible_tensors) == 1, (
                f"Multiple tensors named '{tensor}' in block '{block_name}'"
            )
            tensor_obj, (tensor_index_type, tensor_index) = next(iter(possible_tensors.items()))

        elif is_tensor_var(tensor):
            # Var lookup has unique id, can break out early
            found = False
            for tensor_index_type, tensor_index, tensor_obj in iter_tensors():
                if tensor_obj.same_as(tensor):
                    found = True
                    break

            assert found, f"Could not find tensor '{tensor.name}' in block '{block_name}'"

        elif isinstance(tensor, tuple):
            tensor_index_type, tensor_index = tensor
            assert tensor_index_type in ["read", "write"], (
                f"Invalid tensor_index_type.  "
                f"Expected 'read' or 'write', "
                f"but received {tensor_index_type}"
            )
            tensor_list = block_obj.reads if tensor_index_type == "read" else block_obj.writes
            assert 0 <= tensor_index < len(tensor_list), (
                f"Invalid tensor_index {tensor_index}.  "
                f"Block {block_name} has only "
                f"{len(tensor_list)} {tensor_index_type} tensors."
            )
            tensor_obj = tensor_list[tensor_index].source

        else:
            raise TypeError(f"Invalid type for argument 'tensor': {type(tensor)}")

        if required_tensor_type is not None:
            assert tensor_index_type == required_tensor_type, (
                f"Expected tensor to be read tensor, "
                f"but {tensor_obj.name} was a {tensor_index_type} tensor "
                f"in the specified block"
            )

        return (tensor_index_type, tensor_index, tensor_obj)

    @type_checked
    def transform_layout(
        self,
        block: SBlockRV | str,
        tensor: tuple[str, int] | str | Var,
        index_map: IndexMap | Callable,
        pad_value: int | float | Expr | IndexMap | Callable | None = None,
        *,
        assume_injective_transform: bool = False,
    ) -> None:
        """Apply a transformation represented by IndexMap to tensor

        Parameters
        ----------
        block : SBlockRV | str

            The block that accesses the target tensor.  If a string,
            this must uniquely identify a block.

        tensor: Union[Tuple[str,int], Var, str]

            The tensor to be transformed, or a specification of how to
            identify the tensor to be transformed.

            If `tensor` if a tuple of ``(str,int)``, the first item
            should be either "read" or "write", and the second item is
            an index into the block's read or write regions.

            If `tensor` is a string, it is the name of the tensor,
            which must exist within the reads/writes of the block.  In
            addition, the reads/writes of the block may not contain
            more than one tensor with this name.

            If `tensor` is a Var object, it must exist within the
            reads/writes of the block.

        index_map : IndexMap | Callable

            The transformation to apply.

        pad_value: Optional[int | float | Expr | IndexMap | Callable]

            The value to be used for any padding introduced by the
            transformation.  If the schedule contains a producer block
            for the specified tensor, the pad value will be written as
            part of the producer block if possible, or after the producer
            block otherwise.  Otherwise, if the tensor is an input, will
            insert an annotation block to state that the padding contains
            the known value.

            The pad value may not contain instances of TensorLoad,
            except where it loads a value from the tensor being
            transformed (e.g. to create a circular tensor with
            padding that consists of repeated elements).

            Note: If applied to an input tensor, the calling scope is
            responsible for ensuring that the pad_value is present.
            Algebraic symplifications, branch elimination, and other
            optimizations may assume that this precondition is met, and
            may result in incorrect results being returned.

            If None, the transformation may not introduce padding.

            If an int, float or Expr, the transformation is the
            specific value to be present in the padding.

            If an IndexMap or Callable, the transformation is the
            value to be present in the padding in terms of the
            transformed index.

        assume_injective_transform : bool

            If set to true, the schedule  primitive will assume the index_map is injective and skip
            checking overlapping of the mapped indices. This can be useful for complicated index_map
            that the analysis does not cover. It is the callers' responsibility to ensure the
            index map is injective, otherwise, the correctness of the schedule is not guaranteed.

        Examples
        --------
        Before transform_layout, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_transform_layout(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:

                B = Ts.sblock_alloc_tensor((128, 128), "float32")

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do transform_layout:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_storage_align)
            sch.transform_layout(sch.get_sblock("B"), tensor=("write",0),
                                 index_map=lambda m, n: (m // 16, n // 16, m % 16, n % 16))
            print(sch.mod["main"].script())

        After applying transform_layout, the IR becomes:

        .. code-block:: python

            @Ts.function
            def two_elementwise_transformed_intermediate_tensor(
                A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
            ) -> None:

                B = Ts.sblock_alloc_tensor((8, 8, 16, 16), "float32")

                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi // 16, vj // 16, vi % 16, vj % 16] + 1.0

        """
        block = self._normalize_block_arg(block)
        tensor_index_type, tensor_index, tensor_obj = self._normalize_tensor_arg(block, tensor)

        ndim = len(tensor_obj.ty.shape)
        if callable(index_map):
            index_map = IndexMap.from_func(
                index_map,
                ndim=ndim,
                index_dtype=_get_sblock_default_dtype(self.get(block)),
            )

        if pad_value is None:
            pass
        elif callable(pad_value):
            pad_value = IndexMap.from_func(
                pad_value,
                ndim=len(index_map.final_indices),
                index_dtype=_get_sblock_default_dtype(self.get(block)),
            )
        elif not isinstance(pad_value, IndexMap):
            # Explicitly convert python int/float arguments to the
            # tensor's type.  If the default `tvm_ffi.convert`
            # behavior is applied, these would be converted to
            # int32/float32, which may not match the tensor's type.
            if tensor_obj.ty.dtype.matches_code(DataTypeCode.INT, DataTypeCode.UINT) and isinstance(
                pad_value, int
            ):
                pad_value = IntImm(tensor_obj.ty.dtype.dtype, pad_value)
            elif tensor_obj.ty.dtype.matches_code(
                DataTypeCode.FLOAT, DataTypeCode.BFLOAT
            ) and isinstance(pad_value, float):
                pad_value = FloatImm(tensor_obj.ty.dtype.dtype, pad_value)
            pad_value = IndexMap.from_func(
                lambda *indices: pad_value,
                ndim=len(index_map.final_indices),
                index_dtype=_get_sblock_default_dtype(self.get(block)),
            )

        tensor_index_type_enum = 0 if tensor_index_type == "read" else 1
        _ffi_api.ScheduleTransformLayout(  # type: ignore # pylint: disable=no-member
            self,
            block,
            tensor_index,
            tensor_index_type_enum,
            index_map,
            pad_value,
            assume_injective_transform,
        )

    @type_checked
    def transform_block_layout(self, block: SBlockRV | str, index_map: IndexMap | Callable) -> None:
        """Apply a transformation represented by IndexMap to block

        Parameters
        ----------
        block : SBlockRV | str
            The block to be transformed

        index_map : IndexMap | Callable
            The transformation to apply.

        Examples
        --------

        Before transform_block_layout, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_transform_block_layout(
                A: T.Tensor((16, 16), "float32"),
                B: T.Tensor((16, 16), "float32")
            ) -> None:
                for i, j in T.grid(16, 16):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0

        Create the schedule and do transform_block_layout:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_transform_block_layout)
            sch.transform_block_layout(sch.get_sblock("B"), lambda i, j: (i * 16 + j,))
            print(sch.mod["main"].script())

        After applying transform_block_layout, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_transform_block_layout(
                A: T.Tensor((16, 16), "float32"),
                B: T.Tensor((16, 16), "float32")
            ) -> None:
                for i in range(256):
                    with Ts.sblock("B"):
                        vi, = Ts.axis.remap("S", [i])
                        B[vi // 16, vi % 16] = A[vi // 16, vi % 16] * 2.0
        """
        block = self._normalize_block_arg(block)
        if callable(index_map):
            index_map = IndexMap.from_func(
                index_map,
                index_dtype=_get_sblock_default_dtype(self.get(block)),
            )
        _ffi_api.ScheduleTransformBlockLayout(  # type: ignore # pylint: disable=no-member
            self, block, index_map
        )

    ########## Schedule: Padding decomposition #########
    @type_checked
    def decompose_padding(self, block: SBlockRV | str, loop: LoopRV) -> SBlockRV:
        """Decompose a block of padding computation pattern into two separate blocks.

        a) The block which fill const pad values into full write region;

        b) The block which fill in-bound values into region where pad predicate is true.

        The pad value filling block is inserted right before the given loop.

        The schedule primitive requires:

        1) The input block is a complete block.

        2) The input loop is the ancestor of the block.

        3) The input block is a block which match padding pattern.

        Parameters
        ----------
        block : SBlockRV | str
            The padding block to be decomposed.
        loop : LoopRV
            The loop above which the pad value filling block is inserted before.

        Returns
        -------
        pad_value_block : SBlockRV
            The block filling const pad values.

        Examples
        --------
        Before decompose-padding, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_decompose(x: T.Tensor(128, "int32"), y: T.Tensor(140, "int32")):
                for i in range(140):
                    with Ts.sblock("block"):
                        vi = Ts.axis.remap("S", [i])
                        y[vi] = T.if_then_else(vi >= 6 and vi < 134, x[vi - 6], 0, dtype="int32")

        Create the schedule and do decompose-padding with specified loop:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_decompose, debug_mask="all")
            block = sch.get_sblock("block")
            sch.decompose_padding(block, sch.get_loops(block)[0])
            print(sch.mod["main].script())

        After applying decompose-padding, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_decompose(x: T.Tensor(128, "int32"), y: T.Tensor(140, "int32")):
                for i in T.serial(140):
                    with Ts.sblock("block_pad_const"):
                        vi = Ts.axis.spatial(140, i)
                        y[vi] = 0
                for i in T.serial(128):
                    with Ts.sblock("block"):
                        vi = Ts.axis.spatial(128, i)
                        y[vi + 6] = x[vi]
        """
        block = self._normalize_block_arg(block)
        return _ffi_api.ScheduleDecomposePadding(  # type: ignore # pylint: disable=no-member
            self, block, loop
        )

    @type_checked
    def can_decompose_padding(self, block: SBlockRV | str, loop: LoopRV) -> bool:
        """Check whether the block match padding pattern and can be decomposed."""
        # pylint: disable-next=no-member
        return _ffi_api.CanDecomposePadding(self, block, loop)  # type: ignore

    @type_checked
    def pad_einsum(self, block: SBlockRV | str, padding: list[int]) -> None:
        """Pad the computation of Einsum.

        On a block with trivial binding, this primitive pads the iteration domain of the block by
        the given padding factors, for example, 127 -> 128, 132 -> 144 when padding factor is 16.
        Extra producer and consumer padding blocks will be generated to avoid out-of-bound tensor
        access.

        Einsum pattern means all the indices on the tensor access are either by constants
        (e.g. B[0]) or by variables (e.g. B[i]), but not by composite expressions (e.g. B[i + 1]).

        Parameters
        ----------
        block : SBlockRV | str
            The block that matches the Einsum pattern.

        padding : List[int]
            The padding for each block iter.

        Examples
        --------

        Before applying pad-einsum, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_pad_einsum(
                A: T.Tensor((127, 127), "float32"),
                B: T.Tensor((127, 127), "float32"),
                C: T.Tensor((127, 127), "float32"),
            ) -> None:
                for i0, i1, i2 in T.grid(127, 127, 127):
                    with Ts.sblock("C_shared"):
                        i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
                        with Ts.init():
                            C[i, j] = T.float32(0)
                        C[i, j] = C[i, j] + A[i, k] * B[k, j]

        Create the schedule and do pad-einsum with specified block:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_pad_einsum, debug_mask="all")
            block = sch.get_sblock("C_shared")
            sch.pad_einsum(block, [32, 32, 32])
            print(sch.mod["main"].script())

        After applying decompose-padding, the IR becomes:

        .. code-block:: python

            @Ts.function
            def main(
                A: T.Tensor((127, 127), "float32"),
                B: T.Tensor((127, 127), "float32"),
                C: T.Tensor((127, 127), "float32"),
            ):
                # with Ts.sblock("root"):
                A_pad = Ts.sblock_alloc_tensor((128, 128))
                B_pad = Ts.sblock_alloc_tensor((128, 128))
                C_pad = Ts.sblock_alloc_tensor((128, 128))
                for i0, i1 in T.grid(128, 128):
                    with Ts.sblock("A_pad"):
                        v0, v1 = Ts.axis.remap("SS", [i0, i1])
                        A_pad[v0, v1] = T.if_then_else(
                            v0 < 127 and v1 < 127,
                            A[v0, v1],
                            T.float32(0),
                        )
                for i0, i1 in T.grid(128, 128):
                    with Ts.sblock("B_pad"):
                        v0, v1 = Ts.axis.remap("SS", [i0, i1])
                        B_pad[v0, v1] = T.if_then_else(
                            v0 < 127 and v1 < 127,
                            B[v0, v1],
                            T.float32(0),
                        )
                for i0, i1, i2 in T.grid(128, 128, 128):
                    with Ts.sblock("C_shared"):
                        i, j, k = Ts.axis.remap("SSR", [i0, i1, i2])
                        with Ts.init():
                            C_pad[i, j] = T.float32(0)
                        C_pad[i, j] = C_pad[i, j] + A_pad[i, k] * B_pad[k, j]
                for i0, i1 in T.grid(127, 127):
                    with Ts.sblock("C_pad"):
                        v0, v1 = Ts.axis.remap("SS", [i0, i1])
                        C[v0, v1] = C_pad[v0, v1]

        """
        block = self._normalize_block_arg(block)
        return _ffi_api.SchedulePadEinsum(  # type: ignore # pylint: disable=no-member
            self, block, padding
        )

    ######## Schedule: Var transformation ########

    @type_checked
    def rolling_buffer(self, block: SBlockRV | str, write_tensor_index: int) -> None:
        """Compute the target tensor via rolling buffering, select the outermost rollable
        axis with a positive bound overlap that appears in the block's ancestor loops
        as `rolling axis`, fold and circularize the tensor along the rolling dimension,
        append block predicate to avoid recomputing overlapping elements. It requires:


        1) The block is not an output block and has only RAW dependencies.

        2) The tensor to be an intermediate tensor defined via `alloc_tensor`.

        3) The LCA of the producer and consumer of the tensor is a for loop, typically,
        the producer and consumer of the tensor are cascaded through compute_at.

        4) The access region of the tensor has at least one dimension that contains
        a positive bound overlap.

        Parameters
        ----------
        block : SBlockRV | str
            The producer block of the tensor.
        write_tensor_index : int
            The index of the tensor in block's write region.

        Examples
        --------

        Before rolling_buffer, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_rolling_buffer(
                A: T.Tensor((12, 12), "int8"), C: T.Tensor((8, 8), "int8")
            ) -> None:
                # body
                # with Ts.sblock("root")
                B = Ts.sblock_alloc_tensor([10, 10], dtype="int8")
                for i0, i1 in T.grid(2, 2):
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 3, 3):
                        with Ts.sblock("B"):
                            ax0_1 = Ts.axis.spatial(10, i0 * 4 + ax0)
                            ax1_1 = Ts.axis.spatial(10, i1 * 4 + ax1)
                            rv0, rv1 = Ts.axis.remap("RR", [ax2, ax3])
                            B[ax0_1, ax1_1] = T.max(
                                B[ax0_1, ax1_1], A[ax0_1 + rv0, ax1_1 + rv1]
                            )
                    for ax0, ax1, ax2, ax3 in T.grid(4, 4, 3, 3):
                        with Ts.sblock("C"):
                            ax0_1 = Ts.axis.spatial(8, i0 * 4 + ax0)
                            ax1_1 = Ts.axis.spatial(8, i1 * 4 + ax1)
                            rv0, rv1 = Ts.axis.remap("RR", [ax2, ax3])
                            C[ax0_1, ax1_1] = T.max(
                                C[ax0_1, ax1_1], B[ax0_1 + rv0, ax1_1 + rv1]
                            )

        Create the schedule and do rolling_buffer:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_rolling_buffer)
            sch.rolling_buffer(sch.get_sblock("B"), write_tensor_index=0)
            print(sch.mod["main"].script())

        After applying rolling_buffer, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_rolling_buffer(
                A: T.Tensor((12, 12), "int8"),
                C: T.Tensor((8, 8), "int8")
            ) -> None:
                # body
                # with Ts.sblock("root")
                B = Ts.sblock_alloc_tensor([6, 10], dtype="int8")
                for i0, i1 in T.grid(2, 2):
                    for ax0, ax1, ax2, ax3 in T.grid(6, 6, 3, 3):
                        with Ts.sblock("B"):
                            Ts.where((i0 < 1 or 2 <= ax0) and (i1 < 1 or 2 <= ax1))
                            ax0_1 = Ts.axis.spatial(10, i0 * 4 + ax0)
                            ax1_1 = Ts.axis.spatial(10, i1 * 4 + ax1)
                            rv0, rv1 = Ts.axis.remap("RR", [ax2, ax3])
                            B[ax0_1 % 6, ax1_1] = T.max(
                                B[ax0_1 % 6, ax1_1], A[ax0_1 + rv0, ax1_1 + rv1]
                            )
                    for ax0, ax1, ax2, ax3 in T.grid(4, 4, 3, 3):
                        with Ts.sblock("C"):
                            ax0_1 = Ts.axis.spatial(8, i0 * 4 + ax0)
                            ax1_1 = Ts.axis.spatial(8, i1 * 4 + ax1)
                            rv0, rv1 = Ts.axis.remap("RR", [ax2, ax3])
                            C[ax0_1, ax1_1] = T.max(
                                C[ax0_1, ax1_1], B[ax0_1 % 6 + rv0, ax1_1 + rv1]
                            )

        Note
        ----
        The region_cover property of the consumer block of the target tensor will become false.
        """
        block = self._normalize_block_arg(block)
        # pylint: disable-next=no-member
        return _ffi_api.ScheduleRollingBuffer(self, block, write_tensor_index)  # type: ignore

    ########## Schedule: Misc ##########

    @type_checked
    def enter_postproc(self) -> None:
        """A no-op that marks the start of postprocessing phase of scheduling"""
        _ffi_api.ScheduleEnterPostproc(self)  # type: ignore # pylint: disable=no-member

    @type_checked
    def unsafe_hide_tensor_access(
        self, block: SBlockRV, tensor_type: str, tensor_index_array: list[int]
    ) -> None:
        """Hide some tensor access in a given block. This is an unsafe schedule primitive.

        Parameters
        ----------
        block : SBlockRV
            The block where we hide read access.
        tensor_type : str
            The tensor type: "read"/"write".
        tensor_index_array : List[int]
            The array of tensor indices we hide access.

        Note
        ----
        This schedule primitive is unsafe, and may fail dependency analysis.
        One use case of `unsafe_hide_tensor_access` is to hide the tensor access
        to indices tensors (e.g. in sparse computation) so that we can further tensorize
        the block (the indices tensors appeared in read/write regions may fail the pattern
        matching in `tensorize` primitive, and hide the access to these tensors could address
        the issue).
        """
        _ffi_api.ScheduleUnsafeHideTensorAccess(  # type: ignore # pylint: disable=no-member
            self,
            block,
            tensor_type,
            tensor_index_array,
        )

    @type_checked
    def annotate_tensor_access(
        self, block: SBlockRV, tensor_index: int, tensor_type: str, gen_new_ranges: Callable
    ) -> None:
        """Annotate the read or write region of a block

        Parameters
        ----------
        block : SBlockRV
            The block to be annotated
        tensor_index : int
            The index of the tensor in block's read or write region
        tensor_type : str
            The tensor type: "read" or "write"
        gen_new_ranges : Callable
            A function that takes the block's iter_vars and returns a
            Tuple[Union[Expr, Tuple[Expr, Expr]], ...]
            which defines the new read or write region for the tensor.
            Each element in the tuple can be:
            - A single Expr representing the iter_var itself
            - A tuple of two PrimExprs representing the range (begin, end)

        Examples
        --------
        Annotate a 2D read region for a tensor.
        Before annotate_tensor_access, in TensorIR, the IR is:

        .. code-block:: python

            @Ts.function
            def before_annotate_tensor_access(
                A: T.Tensor((128, 128), "float32"),
                C: T.Tensor((128, 128), "float32")
            ) -> None:
                B = Ts.sblock_alloc_tensor((128, 128), "float32")
                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        Create the schedule and do annotate_tensor_access:

        .. code-block:: python

            sch = tvm.s_tir.Schedule(before_annotate_tensor_access)
            block = sch.get_sblock("B")
            sch.annotate_tensor_access(block, 0, "read",
            lambda vi, vj: ((vi - 1, vi + 1), (vj - 1, vj + 1)))
            print(sch.mod["main"].script())

        After applying annotate_tensor_access, the IR becomes:

        .. code-block:: python

            @Ts.function
            def after_annotate_tensor_access(
                A: T.Tensor((128, 128), "float32"),
                C: T.Tensor((128, 128), "float32")
            ) -> None:
                B = Ts.sblock_alloc_tensor((128, 128), "float32")
                for i, j in T.grid(128, 128):
                    with Ts.sblock("B"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        Ts.reads(A[vi - 1:vi + 1, vj - 1:vj + 1])
                        Ts.writes(B[vi, vj])
                        Ts.sblock_attr({"explicit_read_region": 0})
                        B[vi, vj] = A[vi, vj] * 2.0
                for i, j in T.grid(128, 128):
                    with Ts.sblock("C"):
                        vi, vj = Ts.axis.remap("SS", [i, j])
                        C[vi, vj] = B[vi, vj] + 1.0

        This annotates the read region for tensor A (index 0) in block "B" to be
        [vi-1:vi+1, vj-1:vj+1] for each (vi, vj) in the block's iteration domain.

        Note
        ----
        This function allows manual specification of read or write regions, which
        can be useful in cases where the compiler cannot accurately infer the
        access pattern, such as complex data-dependent accesses.
        It overrides the automatically inferred region for the specified tensor.
        The function adds an annotation to the block, indicating that an explicit
        region has been provided for the tensor at the given index. This annotation
        is used in the CompactTensorAllocation pass to respect the manually specified
        region instead of relying on automatic inference.

        Caution should be exercised when using this function, as incorrect annotations
        may lead to incorrect code generation or runtime errors. It's crucial to
        ensure that the specified region covers all actual reads or writes performed
        by the block for the given tensor.

        """
        block_obj = self.get(block)
        iter_vars = [x.var for x in block_obj.iter_vars]
        new_ranges_spec = gen_new_ranges(*iter_vars)
        if len(iter_vars) != len(new_ranges_spec):
            raise ValueError(
                f"Number of iter_vars ({len(iter_vars)}) must match "
                f"number of new_ranges_spec ({len(new_ranges_spec)})"
            )

        result = []
        for rng in new_ranges_spec:
            if isinstance(rng, tuple | list):
                if len(rng) != 2:
                    raise ValueError(
                        "Tuple must have exactly 2 elements to represent (begin, end)."
                    )
                result.extend(rng)
            elif is_prim_expr(rng):
                result.extend([rng, rng + 1])  # Single point represented as (rng, rng + 1)
            else:
                raise TypeError(f"Expected Expr or tuple of Expr, got {type(rng)}")

        # Create index_map using IndexMap constructor
        index_map = IndexMap(
            initial_indices=iter_vars,
            final_indices=result,
            inverse_index_map=None,
        )

        if tensor_type == "read":
            tensor_index_type = 0
        elif tensor_type == "write":
            tensor_index_type = 1
        else:
            raise ValueError(f"Invalid tensor_type: {tensor_type}. Expected 'read' or 'write'.")

        return _ffi_api.ScheduleAnnotateTensorAccess(
            self, block, tensor_index, tensor_index_type, index_map
        )
