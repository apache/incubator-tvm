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
"""Utilities for meta schedule"""

import os
import shutil
from collections.abc import Callable
from typing import Any

import numpy as np  # type: ignore
from tvm_ffi import Array, Function, Map, get_global_func, register_global_func

from tvm.rpc import RPCSession
from tvm_tirx.ir import FloatImm, IntImm, IRModule


@register_global_func("s_tir.meta_schedule.cpu_count")
def _cpu_count_impl(logical: bool = True) -> int:
    """Return the number of logical or physical CPUs in the system

    Parameters
    ----------
    logical : bool = True
        If True, return the number of logical CPUs, otherwise return the number of physical CPUs

    Returns
    -------
    cpu_count : int
        The number of logical or physical CPUs in the system

    Note
    ----
    The meta schedule search infra intentionally does not adopt the following convention in TVM:
    - C++ API `tvm::runtime::threading::MaxConcurrency()`
    - Environment variable `TVM_NUM_THREADS` or
    - Environment variable `OMP_NUM_THREADS`
    This is because these variables are dedicated to controlling
    the runtime behavior of generated kernels, instead of the host-side search.
    Setting these variables may interfere the host-side search with profiling of generated kernels
    when measuring locally.
    """
    try:
        import psutil  # type: ignore  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "psutil is required by the meta schedule search. Install it with: pip install psutil"
        ) from err
    return psutil.cpu_count(logical=logical) or 1


def cpu_count(logical: bool = True) -> int:
    """Return the number of logical or physical CPUs in the system

    Parameters
    ----------
    logical : bool = True
        If True, return the number of logical CPUs, otherwise return the number of physical CPUs

    Returns
    -------
    cpu_count : int
        The number of logical or physical CPUs in the system

    Note
    ----
    The meta schedule search infra intentionally does not adopt the following convention in TVM:
    - C++ API `tvm::runtime::threading::MaxConcurrency()`
    - Environment variable `TVM_NUM_THREADS` or
    - Environment variable `OMP_NUM_THREADS`

    This is because these variables are dedicated to controlling
    the runtime behavior of generated kernels, instead of the host-side search.
    Setting these variables may interfere the host-side search with profiling of generated kernels
    when measuring locally.
    """
    return _cpu_count_impl(logical)


@register_global_func("s_tir.meta_schedule.using_ipython")
def _using_ipython() -> bool:
    """Return whether the current process is running in an IPython shell.

    Returns
    -------
    result : bool
        Whether the current process is running in an IPython shell.
    """
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore
    except NameError:
        return False


@register_global_func("s_tir.meta_schedule.print_interactive_table")
def print_interactive_table(data: str) -> None:
    """Print the dataframe interactive table in notebook.

    Parameters
    ----------
    data : str
        The serialized performance table from MetaSchedule table printer.
    """
    import pandas as pd  # type: ignore # pylint: disable=import-outside-toplevel
    from IPython.display import display  # type: ignore # pylint: disable=import-outside-toplevel

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    parsed = [
        x.split("|")[1:] for x in list(filter(lambda x: set(x) != {"-"}, data.strip().split("\n")))
    ]
    display(
        pd.DataFrame(
            parsed[1:],
            columns=parsed[0],
        )
    )


def get_global_func_with_default_on_worker(
    name: None | str | Callable,
    default: Callable,
) -> Callable:
    """Get the registered global function on the worker process.

    Parameters
    ----------
    name : Union[None, str, Callable]
        If given a string, retrieve the function in TVM's global registry;
        If given a python function, return it as it is;
        Otherwise, return `default`.

    default : Callable
        The function to be returned if `name` is None.

    Returns
    -------
    result : Callable
        The retrieved global function or `default` if `name` is None
    """
    if name is None:
        return default
    if callable(name):
        return name
    try:
        return get_global_func(name)
    except (ValueError, RuntimeError) as error:
        raise ValueError(
            "Function '{name}' is not registered on the worker process. "
            "The build function and export function should be registered in the worker process. "
            "Note that the worker process is only aware of functions registered in TVM package, "
            "if there are extra functions to be registered, "
            "please send the registration logic via initializer."
        ) from error


def get_global_func_on_rpc_session(
    session: RPCSession,
    name: str,
    extra_error_msg: str | None = None,
) -> Function:
    """Get a Function from the global registry from an RPCSession.

    Parameters
    ----------
    session : RPCSession
        The RPCSession to be retrieved from
    name : str
        The name of the Function
    extra_error_msg : Optional[str]
        Extra information to provide in the error message

    Returns
    -------
    result : Function
        The result
    """
    try:
        result = session.get_function(name)
    except AttributeError as error:
        error_msg = f'Unable to find function "{name}" on the remote RPC server.'
        if extra_error_msg:
            error_msg = f"{error_msg} {extra_error_msg}"
        raise AttributeError(error_msg) from error
    return result


@register_global_func("s_tir.meta_schedule.remove_build_dir")
def remove_build_dir(artifact_path: str) -> None:
    """Clean up the build directory"""
    shutil.rmtree(os.path.dirname(artifact_path))


def _json_de_tvm(obj: Any) -> Any:
    """Unpack a TVM nested container to a JSON object in python.

    Parameters
    ----------
    obj : Any
        The TVM nested container to be unpacked.

    Returns
    -------
    result : Any
        The unpacked json object.
    """
    if obj is None:
        return None
    if isinstance(obj, int | float):
        return obj
    if isinstance(obj, IntImm | FloatImm):
        return obj.value
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, Array):
        return [_json_de_tvm(i) for i in obj]
    if isinstance(obj, Map):
        return {_json_de_tvm(k): _json_de_tvm(v) for k, v in obj.items()}
    raise TypeError("Not supported type: " + str(type(obj)))


def shash2hex(mod: IRModule) -> str:
    """Get the structural hash of a module.

    Parameters
    ----------
    mod : IRModule
        The module to be hashed.

    Returns
    -------
    result : str
        The structural hash of the module.
    """
    func = get_global_func("s_tir.meta_schedule._SHash2Hex")
    return str(func(mod))


def fork_seed(seed: int | None, n: int) -> list[int]:
    # fmt: off
    return np.random.RandomState(seed=seed).randint(1, 2 ** 30, size=n).tolist()
    # fmt: on
