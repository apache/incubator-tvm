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
"""Local builder that compile on the local host"""

import os
import tempfile
from collections.abc import Callable
from typing import Optional, Union

import tvm_ffi
from tvm_ffi import register_global_func

from tvm.runtime.vm import load_param_dict, save_param_dict
from tvm.s_tir.utils import derived_object
from tvm_tirx.codegen import Executable
from tvm_tirx.ir import IRModule
from tvm_tirx.runtime import Tensor
from tvm_tirx.support.popen_pool import MapResult, PopenPoolExecutor, StatusKind
from tvm_tirx.target import Target

from ..logging import get_logger
from ..utils import cpu_count, get_global_func_with_default_on_worker
from .builder import BuilderInput, BuilderResult, PyBuilder

logger = get_logger(__name__)  # pylint: disable=invalid-name


T_BUILD = Callable[  # pylint: disable=invalid-name
    [IRModule, Target, dict[str, Tensor] | None], tvm_ffi.Module
]
T_EXPORT = Callable[[tvm_ffi.Module], str]  # pylint: disable=invalid-name


def _serialize_params(params: dict[str, Tensor] | None) -> bytearray | None:
    if params is None:
        return None
    return save_param_dict(params)


def _deserialize_params(params: bytearray | None) -> dict[str, Tensor] | None:
    if params is None:
        return None
    return load_param_dict(params)


@derived_object
class LocalBuilder(PyBuilder):
    """A builder that builds the given input on local host.

    Parameters
    ----------
    pool : PopenPoolExecutor
        The process pool to run the build.
    max_workers: int
        The max number of Popen workers.
    timeout_sec : float
        The timeout in seconds for the build.
    initializer: Optional[Callable[[], None]]
        The initializer function for each popen worker.
    f_build : Union[None, str, T_BUILD]
        Name of the build function to be used.
        Defaults to `meta_schedule.builder.default_build`.
    f_export : Union[None, str, T_EXPORT]
        Name of the export function to be used.
        Defaults to `meta_schedule.builder.default_export`.

    Attributes
    ----------
    T_BUILD : typing._GenericAlias
        The signature of the function `f_build`, which is

        .. code-block:: python

        def default_build(
            mod: IRModule,
            target: Target,
            params: Optional[Dict[str, Tensor]]
        ) -> tvm_ffi.Module:
            ...

    T_EXPORT : typing._GenericAlias
        The signature of the function `f_export`, which is

        .. code-block:: python

        def default_export(mod: tvm_ffi.Module) -> str:
            ...

    Note
    ----
    The build function and export function should be registered in the worker process.
    The worker process is only aware of functions registered in TVM package,
    if there are extra functions to be registered,
    please send the registration logic via initializer.
    """

    max_workers: int
    timeout_sec: float
    initializer: Callable[[], None] | None
    f_build: None | str | T_BUILD
    f_export: None | str | T_EXPORT

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        timeout_sec: float = 30.0,
        f_build: None | str | T_BUILD = None,
        f_export: None | str | T_EXPORT = None,
        initializer: Callable[[], None] | None = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        max_workers : Optional[int]
            The maximum number of worker processes to be used.
            Defaults to number of CPUs.
        timeout_sec : float
            The timeout in seconds for the build.
        f_build : T_BUILD
            Name of the build function to be used.
            Defaults to `meta_schedule.builder.default_build`.
        f_export : T_EXPORT
            Name of the export function to be used.
            Defaults to `meta_schedule.builder.default_export`.
        initializer : Optional[Callable[[], None]]
            The initializer to be used for the worker processes.
        """
        super().__init__()

        if max_workers is None:
            max_workers = cpu_count(logical=True)
        logger.info("LocalBuilder: max_workers = %d", max_workers)

        self.max_workers = max_workers
        self.timeout_sec = timeout_sec
        self.initializer = initializer
        self.f_build = f_build
        self.f_export = f_export
        self._sanity_check()

    def build(self, build_inputs: list[BuilderInput]) -> list[BuilderResult]:
        results: list[BuilderResult] = []
        map_result: MapResult

        # Here we restart the PopenPool everytime because of a known memory leak issue with the
        # PopenPool workers after a couple times of usage. We don't apply the same to runners to
        # avoid potential problem caused by async behaviour.
        pool = PopenPoolExecutor(
            max_workers=self.max_workers,
            timeout=self.timeout_sec,
            initializer=self.initializer,
        )

        # Dispatch the build inputs to the worker processes.
        for map_result in pool.map_with_error_catching(
            lambda x: _worker_func(*x),
            [
                (
                    self.f_build,
                    self.f_export,
                    build_input.mod,
                    build_input.target,
                    _serialize_params(build_input.params),
                )
                for build_input in build_inputs
            ],
        ):
            if map_result.status == StatusKind.COMPLETE:
                results.append(BuilderResult(map_result.value, None))
            elif map_result.status == StatusKind.TIMEOUT:
                results.append(
                    BuilderResult(
                        None,
                        f"LocalBuilder: Timeout, killed after {self.timeout_sec} seconds",
                    )
                )
            elif map_result.status == StatusKind.EXCEPTION:
                results.append(
                    BuilderResult(
                        None,
                        "LocalBuilder: An exception occurred\n" + str(map_result.value),
                    )
                )
            else:
                raise ValueError("Unreachable: unexpected result: {map_result}")
        pool.shutdown()
        return results

    def _sanity_check(self) -> None:
        def _check(f_build, f_export) -> None:
            get_global_func_with_default_on_worker(name=f_build, default=None)
            get_global_func_with_default_on_worker(name=f_export, default=None)

        # Same reason for the single use PopenPool as mentioned above
        pool = PopenPoolExecutor(
            max_workers=self.max_workers,
            timeout=self.timeout_sec,
            initializer=self.initializer,
        )
        value = pool.submit(_check, self.f_build, self.f_export)
        value.result()
        pool.shutdown()


def _worker_func(
    _f_build: None | str | T_BUILD,
    _f_export: None | str | T_EXPORT,
    mod: IRModule,
    target: Target,
    params: bytearray | None,
) -> str:
    # Step 0. Get the registered functions
    f_build: T_BUILD = get_global_func_with_default_on_worker(
        _f_build,
        default_build,
    )
    f_export: T_EXPORT = get_global_func_with_default_on_worker(
        _f_export,
        default_export,
    )
    # Step 1. Build the IRModule
    rt_mod: tvm_ffi.Module = f_build(mod, target, _deserialize_params(params))
    # Step 2. Export the Module
    artifact_path: str = f_export(rt_mod)
    return artifact_path


@register_global_func("s_tir.meta_schedule.builder.default_build")
def default_build(
    mod: IRModule, target: Target, _params: dict[str, Tensor] | None
) -> tvm_ffi.Module:
    """Default build function.

    Parameters
    ----------
    mod : IRModule
        The IRModule to be built.
    target : Target
        The target to be built.
    _params : Optional[Dict[str, Tensor]]
        The parameters to be used for the build. Must be None.

    Returns
    -------
    rt_mod : tvm_ffi.Module
        The built Module.
    """
    # pylint: disable=import-outside-toplevel
    import tvm.s_tir.tensor_intrin  # pylint: disable=unused-import
    from tvm.s_tir.transform import RemoveWeightLayoutRewriteBlock
    from tvm_tirx.tirx import compile as tirx_compile

    # pylint: enable=import-outside-toplevel
    mod = RemoveWeightLayoutRewriteBlock(skip_tensor_rewrite=True)(mod)
    return tirx_compile(mod, target=target).mod


@register_global_func("s_tir.meta_schedule.builder.default_export")
def default_export(mod: tvm_ffi.Module) -> str:
    """Default export function.

    Parameters
    ----------
    mod : tvm_ffi.Module
        The Module to be exported.

    Returns
    -------
    artifact_path : str
        The path to the exported Module.
    """
    from tvm_tirx.support.tar import tar  # pylint: disable=import-outside-toplevel

    artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod." + tar.output_format)
    Executable(mod).export_library(artifact_path, fcompile=tar)
    return artifact_path


@register_global_func("s_tir.meta_schedule.builder.get_local_builder")
def get_local_builder() -> LocalBuilder:
    """Get the local builder.

    Returns
    -------
    builder : LocalBuilder
        The local builder.
    """
    return LocalBuilder()
