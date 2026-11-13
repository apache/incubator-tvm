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
"""Runner utility functions"""

import itertools
from collections.abc import Callable
from typing import Any

import tvm_ffi

import tvm_tirx.runtime
from tvm.runtime import time_evaluator
from tvm_tirx.runtime import Device

from .config import EvaluatorConfig

T_ARG_INFO_JSON_OBJ = list[Any]  # pylint: disable=invalid-name
T_ARG_INFO_JSON_OBJ_LIST = list[T_ARG_INFO_JSON_OBJ]  # pylint: disable=invalid-name
T_ARGUMENT = Any  # pylint: disable=invalid-name
T_ARGUMENT_LIST = list[T_ARGUMENT]  # pylint: disable=invalid-name


def alloc_argument_common(
    f_random_fill: Callable,
    device: Device,
    args_info: T_ARG_INFO_JSON_OBJ_LIST,
    alloc_repeat: int,
    f_alloc: Callable | None = None,
) -> list[T_ARGUMENT_LIST]:
    """Common function to allocate the arguments

    Parameters
    ----------
    f_random_fill: Callable
        The callable function for random fill
    device: Device
        The device to allocate the arguments
    args_info: T_ARG_INFO_JSON_OBJ_LIST
        The arguments info
    alloc_repeat: int
        The number of times to repeat the allocation

    Returns
    -------
    repeated_args: List[T_ARGUMENT_LIST]
        The allocation args
    """

    def alloc_tensor(_, dtype, shape) -> tvm_tirx.runtime.Tensor:
        arg = (
            f_alloc(shape, dtype)
            if f_alloc is not None
            else tvm_tirx.runtime.empty(shape=shape, dtype=dtype, device=device)
        )
        f_random_fill(arg)
        return arg

    def alloc_fail(*arg_info) -> None:
        raise NotImplementedError(arg_info)

    dispatcher: dict[Any, Callable] = {
        "TENSOR": alloc_tensor,
        None: alloc_fail,
    }

    repeated_args: list[T_ARGUMENT_LIST] = []
    for _ in range(alloc_repeat):
        args: T_ARGUMENT_LIST = []
        arg_info: T_ARG_INFO_JSON_OBJ
        for arg_info in args_info:
            arg_type = arg_info[0]
            arg: Any = dispatcher.get(arg_type, None)(*arg_info)
            args.append(arg)
        repeated_args.append(args)
    return repeated_args


def run_evaluator_common(
    rt_mod: tvm_ffi.Module,
    device: Device,
    evaluator_config: EvaluatorConfig,
    repeated_args: list[T_ARGUMENT_LIST],
) -> list[float]:
    """Common function to run the evaluator

    Parameters
    ----------
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
    evaluator = time_evaluator(
        rt_mod,
        func_name=rt_mod.entry_name,
        device=device,
        number=evaluator_config.number,
        repeat=evaluator_config.repeat,
        min_repeat_ms=evaluator_config.min_repeat_ms,
        f_preproc="runtime.CacheFlushCPUArguments"
        if evaluator_config.enable_cpu_cache_flush
        else "",
    )
    repeated_costs: list[list[float]] = []
    for args in repeated_args:
        device.sync()
        profile_result = evaluator(*args)
        repeated_costs.append(profile_result.results)
    costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
    return costs
