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
"""Runners"""

from collections.abc import Callable
from typing import Union

import tvm_ffi

# isort: off
from typing import Literal

# isort: on

from tvm_ffi import register_object

from .. import _ffi_api
from ..arg_info import ArgInfo


@register_object("s_tir.meta_schedule.RunnerInput")
class RunnerInput(tvm_ffi.Object):
    """The runner's input

    Parameters
    ----------
    artifact_path : str
        The path to the built artifact.
    device_type : str
        The device type.
    args_info : List[ArgInfo]
        The argument information.
    """

    artifact_path: str
    device_type: str
    args_info: list[ArgInfo]

    def __init__(
        self,
        artifact_path: str,
        device_type: str,
        args_info: list[ArgInfo],
    ) -> None:
        """Constructor

        Parameters
        ----------
        artifact_path : str
            The path to the built artifact.
        device_type : str
            The device type.
        args_info : List[ArgInfo]
            The argument information.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.RunnerInput,  # type: ignore # pylint: disable=no-member
            artifact_path,
            device_type,
            args_info,
        )


@register_object("s_tir.meta_schedule.RunnerResult")
class RunnerResult(tvm_ffi.Object):
    """The runner's result

    Parameters
    ----------
    run_secs : Optional[List[float]]
        The run time in seconds.
    error_msg : Optional[str]
        The error message, if any.
    """

    run_secs: list[float] | None
    error_msg: str | None

    def __init__(
        self,
        run_secs: list[float] | None,
        error_msg: str | None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        run_secs : Optional[List[float]]
            The run time in seconds.
        error_msg : Optional[str]
            The error message, if any.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.RunnerResult,  # type: ignore # pylint: disable=no-member
            run_secs,
            error_msg,
        )


@register_object("s_tir.meta_schedule.RunnerFuture")
class RunnerFuture(tvm_ffi.Object):
    """
    A class to fetch asynchronous runner's output.
    This is NOT the user facing class for function overloading inheritance.
    Can be used for general return type of runner.

    See also: PyRunnerFuture
    """

    def __init__(self, f_done: Callable, f_result: Callable | None = None) -> None:
        """Constructor"""

        self.__init_handle_by_constructor__(
            _ffi_api.RunnerFuture,  # type: ignore # pylint: disable=no-member
            f_done,
            f_result,
        )

    def done(self) -> bool:
        """Check whether the runner has finished."""
        return _ffi_api.RunnerFutureDone(self)  # type: ignore # pylint: disable=no-member

    def result(self) -> RunnerResult:
        """Fetch the runner's output if it is ready."""
        return _ffi_api.RunnerFutureResult(self)  # type: ignore # pylint: disable=no-member


class PyRunnerFuture:
    """
    A class to fetch asynchronous runner's output with customizable function on the python side.
    This is the user facing class for function overloading inheritance.
    Can NOT be used for general return type of runner.

    Note: @derived_object is required for proper usage of any inherited class.
    Example::

        @derived_object
        def LocalRunnerFuture(PyRunnerFuture):
            ...
    """

    _tvm_metadata = {
        "cls": RunnerFuture,
        "methods": ["done", "result"],
    }

    def done(self) -> bool:
        """Check whether the runner has finished."""
        raise NotImplementedError

    def result(self) -> RunnerResult:
        """Fetch the runner's output if it is ready."""
        raise NotImplementedError


@register_object("s_tir.meta_schedule.Runner")
class Runner(tvm_ffi.Object):
    """The abstract runner interface"""

    RunnerType = Union["Runner", Literal["local", "rpc"]]

    def run(self, runner_inputs: list[RunnerInput]) -> list[RunnerFuture]:
        """Run the built artifact and get runner futures.

        Parameters
        ----------
        runner_inputs : List[RunnerInput]
            The inputs to the runner.

        Returns
        -------
        runner_futures: List[RunnerFuture]
            The runner futures.
        """
        return _ffi_api.RunnerRun(self, runner_inputs)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(  # pylint: disable=keyword-arg-before-vararg
        kind: Literal["local", "rpc"] = "local",
        *args,
        **kwargs,
    ) -> "Runner":
        """Create a Runner."""
        from . import LocalRunner, RPCRunner  # pylint: disable=import-outside-toplevel

        if kind == "local":
            if "max_workers" in kwargs:
                kwargs.pop("max_workers")
            return LocalRunner(*args, **kwargs)  # type: ignore
        elif kind == "rpc":
            return RPCRunner(*args, **kwargs)  # type: ignore
        raise ValueError(f"Unknown Runner: {kind}")


create = Runner.create  # pylint: disable=invalid-name


@register_object("s_tir.meta_schedule.PyRunner")
class _PyRunner(Runner):
    """
    A TVM object runner to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyRunner
    """

    def __init__(self, f_run: Callable | None = None) -> None:
        """Constructor"""

        self.__init_handle_by_constructor__(
            _ffi_api.RunnerPyRunner,  # type: ignore # pylint: disable=no-member
            f_run,
        )


class PyRunner:
    """
    An abstract runner with customized run method on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyRunner,
        "methods": ["run"],
    }

    def run(self, runner_inputs: list[RunnerInput]) -> list[RunnerFuture]:
        """Run the built artifact and get runner futures.

        Parameters
        ----------
        runner_inputs : List[RunnerInput]
            The inputs to the runner.

        Returns
        -------
        runner_futures: List[RunnerFuture]
            The runner futures.
        """
        raise NotImplementedError
