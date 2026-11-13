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
"""RPC Runner"""

import concurrent.futures
import os.path as osp
from collections.abc import Callable
from contextlib import contextmanager

import tvm_ffi

from tvm.rpc import RPCSession
from tvm.s_tir.utils import derived_object
from tvm_tirx.runtime import Device
from tvm_tirx.support.popen_pool import PopenPoolExecutor

from ..logging import get_logger
from ..profiler import Profiler
from ..utils import (
    get_global_func_on_rpc_session,
    get_global_func_with_default_on_worker,
)
from .config import EvaluatorConfig, RPCConfig
from .runner import PyRunner, PyRunnerFuture, RunnerFuture, RunnerInput, RunnerResult
from .utils import (
    T_ARG_INFO_JSON_OBJ_LIST,
    T_ARGUMENT_LIST,
    alloc_argument_common,
    run_evaluator_common,
)

logger = get_logger(__name__)  # pylint: disable=invalid-name


T_CREATE_SESSION = Callable[  # pylint: disable=invalid-name
    [RPCConfig],  # The RPC configuration
    RPCSession,  # The RPC Session
]
T_UPLOAD_MODULE = Callable[  # pylint: disable=invalid-name
    [
        RPCSession,  # The RPC Session
        str,  # local path to the artifact
        str,  # remote path to the artifact
    ],
    tvm_ffi.Module,  # the Module opened on the remote
]
T_ALLOC_ARGUMENT = Callable[  # pylint: disable=invalid-name
    [
        RPCSession,  # The RPC Session
        Device,  # The device on the remote
        T_ARG_INFO_JSON_OBJ_LIST,  # The metadata information of the arguments to be allocated
        int,  # The number of repeated allocations to be done
    ],
    list[T_ARGUMENT_LIST],  # A list of argument lists
]
T_RUN_EVALUATOR = Callable[  # pylint: disable=invalid-name
    [
        RPCSession,  # The RPC Session
        tvm_ffi.Module,  # The Module opened on the remote
        Device,  # The device on the remote
        EvaluatorConfig,  # The evaluator configuration
        list[T_ARGUMENT_LIST],  # A list of argument lists
    ],
    list[float],  # A list of running time
]
T_CLEANUP = Callable[  # pylint: disable=invalid-name
    [
        RPCSession | None,  # The RPC Session to be cleaned up
        str | None,  # remote path to the artifact
    ],
    None,
]


@derived_object
class RPCRunnerFuture(PyRunnerFuture):
    """RPC based runner future

    Parameters
    ----------
    future: concurrent.futures.Future
        The concurrent function to check when the function is done and to return the result.
    timeout_sec: float
        The timeout in seconds.
    """

    future: concurrent.futures.Future
    timeout_sec: float

    def __init__(self, future: concurrent.futures.Future, timeout_sec: float) -> None:
        """Constructor

        Parameters
        ----------
        future: concurrent.futures.Future
            The concurrent function to check when the function is done and to return the result.
        timeout_sec: float
            The timeout in seconds.
        """
        super().__init__()
        self.future = future
        self.timeout_sec = timeout_sec

    def done(self) -> bool:
        return self.future.done()

    def result(self) -> RunnerResult:
        try:
            run_secs: list[float] = self.future.result()
        except TimeoutError:
            return RunnerResult(
                None,
                error_msg=f"RPCRunner: Timeout, killed after {self.timeout_sec} seconds",
            )
        except Exception as exception:  # pylint: disable=broad-except
            return RunnerResult(
                None,
                error_msg="RPCRunner: An exception occurred\n" + str(exception),
            )
        return RunnerResult(run_secs, None)


@derived_object
class RPCRunner(PyRunner):
    """RPC based runner

    Parameters
    ----------
    rpc_config: RPCConfig
        The rpc configuration.
    evaluator_config: EvaluatorConfig
        The evaluator configuration.
    cooldown_sec: float
        The cooldown in seconds. TODO(@junrushao1994,@zxybazh): This is not used yet.
    alloc_repeat: int
        The number of times to repeat the allocation.
    f_create_session: Optional[str, Callable]
        The function name to create the session or the function itself.
    f_upload_module: Optional[str, Callable]
        The function name to upload the module or the function itself.
    f_alloc_argument: Optional[str, Callable]
        The function name to allocate the arguments or the function itself.
    f_run_evaluator: Optional[str, Callable]
        The function name to run the evaluator or the function itself.
    f_cleanup: Optional[str, Callable]
        The function name to cleanup the session or the function itself.
    pool: PopenPoolExecutor
        The popen pool executor.

    Attributes
    ----------
    T_CREATE_SESSION : typing._GenericAlias
        The signature of the function `f_create_session`, which is:

        .. code-block:: python

        def default_create_session(rpc_config: RPCConfig) -> RPCSession:
            ...

    T_UPLOAD_MODULE : typing._GenericAlias
        The signature of the function `f_upload_module`, which is:

        .. code-block:: python

        def default_upload_module(
            session: RPCSession,
            local_path: str,
            remote_path: str,
        ) -> tvm_ffi.Module:
            ...

    T_ALLOC_ARGUMENT : typing._GenericAlias
        The signature of the function `f_alloc_argument`, which is:

        .. code-block:: python

        def default_alloc_argument(
            session: RPCSession,
            device: Device,
            args_info: T_ARG_INFO_JSON_OBJ_LIST,
            alloc_repeat: int,
        ) -> List[T_ARGUMENT_LIST]:
            ...

    T_RUN_EVALUATOR : typing._GenericAlias
        The signature of the function `f_run_evaluator`, which is:

        .. code-block:: python

        def default_run_evaluator(
            session: RPCSession,
            rt_mod: tvm_ffi.Module,
            device: Device,
            evaluator_config: EvaluatorConfig,
            repeated_args: List[T_ARGUMENT_LIST],
        ) -> List[float]:
            ...

    T_CLEANUP : typing._GenericAlias
        The signature of the function `f_cleanup`, which is:

        .. code-block:: python

        def default_cleanup(
            session: Optional[RPCSession],
            remote_path: Optional[str],
        ) -> None:
            ...
    """

    rpc_config: RPCConfig
    evaluator_config: EvaluatorConfig
    cooldown_sec: float
    alloc_repeat: int

    f_create_session: T_CREATE_SESSION | str | None
    f_upload_module: T_UPLOAD_MODULE | str | None
    f_alloc_argument: T_ALLOC_ARGUMENT | str | None
    f_run_evaluator: T_RUN_EVALUATOR | str | None
    f_cleanup: T_CLEANUP | str | None

    pool: PopenPoolExecutor

    def __init__(
        self,
        rpc_config: RPCConfig | None = None,
        evaluator_config: EvaluatorConfig | None = None,
        cooldown_sec: float = 0.0,
        alloc_repeat: int = 1,
        f_create_session: T_CREATE_SESSION | str | None = None,
        f_upload_module: T_UPLOAD_MODULE | str | None = None,
        f_alloc_argument: T_ALLOC_ARGUMENT | str | None = None,
        f_run_evaluator: T_RUN_EVALUATOR | str | None = None,
        f_cleanup: T_CLEANUP | str | None = None,
        max_workers: int | None = None,
        initializer: Callable[[], None] | None = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        rpc_config: RPCConfig
            The rpc configuration.
        evaluator_config: EvaluatorConfig
            The evaluator configuration.
        cooldown_sec: float
            The cooldown in seconds.
        alloc_repeat: int
            The number of times to random fill the allocation.
        f_create_session: Union[T_CREATE_SESSION, str, None]
            The function name to create the session or the function itself.
        f_upload_module: Union[T_UPLOAD_MODULE, str, None]
            The function name to upload the module or the function itself.
        f_alloc_argument: Union[T_ALLOC_ARGUMENT, str, None]
            The function name to allocate the arguments or the function itself.
        f_run_evaluator: Union[T_RUN_EVALUATOR, str, None]
            The function name to run the evaluator or the function itself.
        f_cleanup: Union[T_CLEANUP, str, None]
            The function name to cleanup the session or the function itself.
        max_workers: Optional[int] = None
            The maximum number of connections. Defaults to 1.
        initializer: Optional[Callable[[], None]]
            The initializer function.
        """
        super().__init__()
        self.rpc_config = RPCConfig._normalized(rpc_config)
        self.evaluator_config = EvaluatorConfig._normalized(evaluator_config)
        self.cooldown_sec = cooldown_sec
        self.alloc_repeat = alloc_repeat
        self.f_create_session = f_create_session
        self.f_upload_module = f_upload_module
        self.f_alloc_argument = f_alloc_argument
        self.f_run_evaluator = f_run_evaluator
        self.f_cleanup = f_cleanup
        if max_workers is None:
            max_workers = 1
        logger.info("RPCRunner: max_workers = %d", max_workers)
        self.pool = PopenPoolExecutor(
            max_workers=max_workers,
            initializer=initializer,
        )
        self._sanity_check()

    def run(self, runner_inputs: list[RunnerInput]) -> list[RunnerFuture]:
        results: list[RunnerFuture] = []
        for runner_input in runner_inputs:
            future = RPCRunnerFuture(
                future=self.pool.submit(
                    _worker_func,
                    self.f_create_session,
                    self.f_upload_module,
                    self.f_alloc_argument,
                    self.f_run_evaluator,
                    self.f_cleanup,
                    self.rpc_config,
                    self.evaluator_config,
                    self.alloc_repeat,
                    str(runner_input.artifact_path),
                    str(runner_input.device_type),
                    tuple(arg_info.as_json() for arg_info in runner_input.args_info),
                ),
                timeout_sec=self.rpc_config.session_timeout_sec,
            )
            results.append(future)  # type: ignore
        return results

    def _sanity_check(self) -> None:
        def _check(
            f_create_session,
            f_upload_module,
            f_alloc_argument,
            f_run_evaluator,
            f_cleanup,
        ) -> None:
            get_global_func_with_default_on_worker(name=f_create_session, default=None)
            get_global_func_with_default_on_worker(name=f_upload_module, default=None)
            get_global_func_with_default_on_worker(name=f_alloc_argument, default=None)
            get_global_func_with_default_on_worker(name=f_run_evaluator, default=None)
            get_global_func_with_default_on_worker(name=f_cleanup, default=None)

        value = self.pool.submit(
            _check,
            self.f_create_session,
            self.f_upload_module,
            self.f_alloc_argument,
            self.f_run_evaluator,
            self.f_cleanup,
        )
        value.result()


def _worker_func(
    _f_create_session: T_CREATE_SESSION | str | None,
    _f_upload_module: T_UPLOAD_MODULE | str | None,
    _f_alloc_argument: T_ALLOC_ARGUMENT | str | None,
    _f_run_evaluator: T_RUN_EVALUATOR | str | None,
    _f_cleanup: T_CLEANUP | str | None,
    rpc_config: RPCConfig,
    evaluator_config: EvaluatorConfig,
    alloc_repeat: int,
    artifact_path: str,
    device_type: str,
    args_info: T_ARG_INFO_JSON_OBJ_LIST,
) -> list[float]:
    # Step 0. Get the registered functions
    f_create_session: T_CREATE_SESSION = get_global_func_with_default_on_worker(
        _f_create_session, default_create_session
    )
    f_upload_module: T_UPLOAD_MODULE = get_global_func_with_default_on_worker(
        _f_upload_module, default_upload_module
    )
    f_alloc_argument: T_ALLOC_ARGUMENT = get_global_func_with_default_on_worker(
        _f_alloc_argument, default_alloc_argument
    )
    f_run_evaluator: T_RUN_EVALUATOR = get_global_func_with_default_on_worker(
        _f_run_evaluator, default_run_evaluator
    )
    f_cleanup: T_CLEANUP = get_global_func_with_default_on_worker(_f_cleanup, default_cleanup)
    # Managed resources
    session: RPCSession | None = None
    remote_path: str | None = None

    @contextmanager
    def resource_handler():
        try:
            yield
        finally:
            # Final step. Always clean up
            with Profiler.timeit("RPCRunner/cleanup"):
                f_cleanup(session, remote_path)

    with resource_handler():
        # Step 1. Create session
        with Profiler.timeit("RPCRunner/create_session"):
            session = f_create_session(rpc_config)
            device = session.device(device_type, 0)
        # Step 2. Upload the module
        with Profiler.timeit("RPCRunner/upload_module"):
            _, remote_path = osp.split(artifact_path)
            local_path: str = artifact_path
            rt_mod: tvm_ffi.Module = f_upload_module(session, local_path, remote_path)
        # Step 3: Allocate input arguments
        with Profiler.timeit("RPCRunner/alloc_argument"):
            repeated_args: list[T_ARGUMENT_LIST] = f_alloc_argument(
                session,
                device,
                args_info,
                alloc_repeat,
            )
        # Step 4: Run time_evaluator
        with Profiler.timeit("LocalRunner/run_evaluator"):
            costs: list[float] = f_run_evaluator(
                session,
                rt_mod,
                device,
                evaluator_config,
                repeated_args,
            )
    return costs


def default_create_session(rpc_config: RPCConfig) -> RPCSession:
    """Default function to create the session

    Parameters
    ----------
    rpc_config : RPCConfig
        The configuration of the RPC session

    Returns
    -------
    session : RPCSession
        The created rpc session
    """
    return rpc_config.connect_server()


def default_upload_module(
    session: RPCSession,
    local_path: str,
    remote_path: str,
) -> tvm_ffi.Module:
    """Default function to upload the module

    Parameters
    ----------
    session: RPCSession
        The session to upload the module
    local_path: str
        The local path of the module
    remote_path: str
        The remote path to place the module

    Returns
    -------
    rt_mod : tvm_ffi.Module
        The runtime module
    """
    session.upload(local_path, remote_path)
    rt_mod: tvm_ffi.Module = session.load_module(remote_path)
    return rt_mod


def default_alloc_argument(
    session: RPCSession,
    device: Device,
    args_info: T_ARG_INFO_JSON_OBJ_LIST,
    alloc_repeat: int,
) -> list[T_ARGUMENT_LIST]:
    """Default function to allocate the arguments

    Parameters
    ----------
    session: RPCSession
        The session to allocate the arguments
    device: Device
        The device to allocate the arguments
    args_info: T_ARG_INFO_JSON_OBJ_LIST
        The arguments info
    alloc_repeat: int
        The number of times to repeat the allocation

    Returns
    -------
    repeated_args: List[Args]
        The allocation args
    """
    f_random_fill = get_global_func_on_rpc_session(
        session,
        "tvm.contrib.random.random_fill_for_measure",
        "Please make sure '-DTVM_USE_RANDOM=ON' is set when configuring CMake on the RPC server.",
    )

    return alloc_argument_common(f_random_fill, device, args_info, alloc_repeat, device.empty)


def default_run_evaluator(
    session: RPCSession,  # pylint: disable=unused-argument
    rt_mod: tvm_ffi.Module,
    device: Device,
    evaluator_config: EvaluatorConfig,
    repeated_args: list[T_ARGUMENT_LIST],
) -> list[float]:
    """Default function to run the evaluator

    Parameters
    ----------
    session: RPCSession
        The session to run the evaluator
    rt_mod: tvm_ffi.Module
        The runtime module
    device: Device
        The device to run the evaluator
    evaluator_config: EvaluatorConfig
        The evaluator config
    repeated_args: List[T_ARGUMENT_LIST]
        The repeated arguments

    Returns
    -------
    costs: List[float]
        The evaluator results
    """
    return run_evaluator_common(rt_mod, device, evaluator_config, repeated_args)


def default_cleanup(
    session: RPCSession | None,
    remote_path: str | None,
) -> None:
    """Default function to clean up the session

    Parameters
    ----------
    session: RPCSession
        The session to clean up
    remote_path: str
        The remote path to clean up
    """
    if session is not None and remote_path is not None:
        session.remove(remote_path)
        session.remove(remote_path + ".so")
        session.remove("")
