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
"""This module defines a Session in Disco. Session is the primary interface that users interact
with the distributed runtime.
"""

import logging
import os
import pickle
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import numpy as np
from tvm_ffi import Object, Shape, get_global_func, register_global_func, register_object

from tvm_tirx.runtime import Device
from tvm_tirx.runtime._tensor import Tensor
from tvm_tirx.runtime._tensor import tensor as _as_Tensor

from . import _ffi_api, process_pool  # pylint: disable=unused-import


@register_object("runtime.disco.DRef")
class DRef(Object):
    """An object that exists on all workers. The controller process assigns a unique "register id"
    to each object, and the worker process uses this id to refer to the object residing on itself.
    """

    def debug_get_from_remote(self, worker_id: int) -> Any:
        """Get the value of a DRef from a remote worker. It is only used for debugging purposes.

        Parameters
        ----------
        worker_id : int
            The id of the worker to be fetched from.

        Returns
        -------
        value : object
            The value of the register.
        """
        return _ffi_api.DRefDebugGetFromRemote(self, worker_id)  # type: ignore # pylint: disable=no-member

    def debug_copy_from(
        self,
        worker_id: int,
        value: np.ndarray | Tensor,
    ) -> None:
        """Copy an Tensor value to remote for debugging purposes.

        Parameters
        ----------
        worker_id : int
            The id of the worker to be copied to.

        value : Union[numpy.ndarray, Tensor]
            The value to be copied.
        """
        if not isinstance(value, Tensor):
            value = _as_Tensor(value)
        return _ffi_api.DRefDebugCopyFrom(self, worker_id, value)  # type: ignore # pylint: disable=no-member


class DPackedFunc(DRef):
    """A PackedFunc in a Disco session."""

    # tvm_ffi Object subclasses cannot store Python attributes by default
    # (the metaclass sets `__slots__ = ()`); list the field(s) we store here.
    __slots__ = ("session",)

    def __init__(self, dref: DRef, session: "Session") -> None:
        self.__move_handle_from__(dref)
        self.session = session

    def __call__(self, *args) -> DRef:
        return self.session.call_packed(self, *args)


class DModule(DRef):
    """A Module in a Disco session."""

    # tvm_ffi Object subclasses cannot store Python attributes by default
    # (the metaclass sets `__slots__ = ()`); list the field(s) we store here.
    __slots__ = ("session",)

    def __init__(self, dref: DRef, session: "Session") -> None:
        self.__move_handle_from__(dref)
        self.session = session

    def __getitem__(self, name: str) -> DPackedFunc:
        func = self.session._get_cached_method("ffi.ModuleGetFunction")
        return DPackedFunc(func(self, name, False), self.session)


@register_object("runtime.disco.Session")
class Session(Object):
    """A Disco interactive session. It allows users to interact with the Disco command queue with
    various PackedFunc calling convention."""

    # tvm_ffi Object subclasses cannot store Python attributes by default
    # (the metaclass sets `__slots__ = ()`); list the fields we store here:
    # the method-lookup cache and the lazily bound import helper.
    __slots__ = ("_cache", "_import_python_module")

    def _get_cached_method(self, name: str) -> Callable:
        if not hasattr(self, "_cache"):
            cache = self._cache = {}  # pylint: disable=attribute-defined-outside-init
        else:
            cache = self._cache
        if name not in cache:
            func = cache[name] = self.get_global_func(name)
        else:
            func = cache[name]
        return func

    def empty(
        self,
        shape: Sequence[int],
        dtype: str,
        device: Device | None = None,
        worker0_only: bool = False,
        in_group: bool = True,
    ) -> DRef:
        """Create an empty Tensor on all workers and attach them to a DRef.

        Parameters
        ----------
        shape : tuple of int
            The shape of the Tensor.

        dtype : str
            The data type of the Tensor.

        device : Optional[Device] = None
            The device of the Tensor.

        worker0_only: bool
            If False (default), allocate an array on each worker.  If
            True, only allocate an array on worker0.

        in_group: bool
            Take effective when `worker0_only` is True. If True (default),
            allocate an array on each first worker in each group. If
            False, only allocate an array on worker0 globally.

        Returns
        -------
        array : DRef
            The created Tensor.

        """
        func = self._get_cached_method("runtime.disco.empty")
        return func(Shape(shape), dtype, device, worker0_only, in_group)

    def shutdown(self):
        """Shut down the Disco session"""
        _ffi_api.SessionShutdown(self)  # type: ignore # pylint: disable=no-member

    @property
    def num_workers(self) -> int:
        """Return the number of workers in the session"""
        return _ffi_api.SessionGetNumWorkers(self)  # type: ignore # pylint: disable=no-member

    def get_global_func(self, name: str) -> DRef:
        """Get a global function on workers.

        Parameters
        ----------
        name : str
            The name of the global function.

        Returns
        -------
        func : DRef
            The global packed function
        """
        return DPackedFunc(_ffi_api.SessionGetGlobalFunc(self, name), self)  # type: ignore # pylint: disable=no-member

    def import_python_module(self, module_name: str) -> None:
        """Import a python module in each worker

        This may be required before call

        Parameters
        ----------
        module_name: str

            The python module name, as it would be used in a python
            `import` statement.
        """
        if not hasattr(self, "_import_python_module"):
            self._import_python_module = self.get_global_func("runtime.disco._import_python_module")

        self._import_python_module(module_name)

    def call_packed(self, func: DRef, *args) -> DRef:
        """Call a PackedFunc on workers providing variadic arguments.

        Parameters
        ----------
        func : PackedFunc
            The function to be called.
        *args : various types
            In the variadic arguments, the supported types include:
            - integers and floating point numbers;
            - DLDataType;
            - DLDevice;
            - str (std::string in C++);
            - DRef.

        Returns
        -------
        return_value : various types
            The return value of the function call.

        Notes
        -----
        Examples of unsupported types:
        - Tensor, DLTensor,;
        - TVM Objects, including PackedFunc, Module and String.
        """
        return _ffi_api.SessionCallPacked(self, 0, 0, func, *args)  # type: ignore # pylint: disable=no-member

    def _sync_worker(self, worker_id: int) -> None:
        """Synchronize the controller with a worker, and it will wait until the worker finishes
        executing all the existing instructions. This function is usually used for worker-0, because
        it is the only worker that is assumed to collocate with the controller. Syncing with other
        workers may not be supported and should only be used for debugging purposes.

        Parameters
        ----------
        worker_id : int
            The id of the worker to be synced with.
        """
        return _ffi_api.SessionSyncWorker(self, worker_id)  # type: ignore # pylint: disable=no-member

    def _sync_all(self) -> None:
        """Synchronize the controller with all workers in the current session, and it will
        wait until all workers finish executing all the existing instructions."""
        for i in range(self.num_workers):
            self._sync_worker(i)

    def sync_worker_0(self) -> None:
        """Synchronize the controller with worker-0, and it will wait until the worker-0 finishes
        executing all the existing instructions."""
        return self._sync_worker(0)

    def copy_from_worker_0(self, host_array: Tensor, remote_array: DRef) -> None:
        """Copy an Tensor from worker-0 to the controller-side Tensor.

        Parameters
        ----------
        host_array : numpy.ndarray
            The array to be copied to worker-0.

        remote_array : Tensor
            The Tensor on worker-0.
        """
        return _ffi_api.SessionCopyFromWorker0(self, host_array, remote_array)  # type: ignore # pylint: disable=no-member

    def copy_to_worker_0(self, host_array: Tensor, remote_array: DRef | None = None) -> DRef:
        """Copy the controller-side Tensor to worker-0.

        Parameters
        ----------
        host_array : Tensor
            The array to be copied to worker-0.

        remote_array : Optiona[DRef]
            The destination Tensor on worker-0.

        Returns
        -------
        output_array: DRef

            The DRef containing the copied data on worker0, and
            std::nullopt on all other workers.  If `remote_array` was
            provided, this return value is the same as `remote_array`.
            Otherwise, it is the newly allocated space.

        """
        if remote_array is None:
            remote_array = self.empty(host_array.shape, host_array.dtype, worker0_only=True)

        _ffi_api.SessionCopyToWorker0(self, host_array, remote_array)  # type: ignore # pylint: disable=no-member
        return remote_array

    def load_vm_module(
        self,
        path: str,
        device: Device | None = None,
    ) -> DModule:
        """Load a VM module from a file.

        Parameters
        ----------
        path : str
            The path to the VM module file.

        device : Optional[Device] = None
            The device to load the VM module to. Default to the default device of each worker.

        Returns
        -------
        module : DModule
            The loaded VM module.
        """
        func = self._get_cached_method("runtime.disco.load_vm_module")
        return DModule(func(path, device), self)

    def init_ccl(self, ccl: str, *device_ids):
        """Initialize the underlying communication collective library.

        Parameters
        ----------
        ccl : str
            The name of the communication collective library. Currently supported libraries are:
            - nccl
            - rccl
            - mpi

        *device_ids : int
            The device IDs to be used by the underlying communication library.
        """
        assert ccl in ("nccl", "rccl"), f"Unsupported CCL backend: {ccl}"
        _ffi_api.SessionInitCCL(self, ccl, Shape(device_ids))  # type: ignore # pylint: disable=no-member
        self._clear_ipc_memory_pool()

    def broadcast(
        self,
        src: np.ndarray | Tensor,
        dst: DRef | None = None,
        in_group: bool = True,
    ) -> DRef:
        """Broadcast an array to all workers

        Parameters
        ----------
        src: Union[np.ndarray, Tensor]
            The array to be broadcasted.

        dst: Optional[DRef]
            The output array.  If None, an array matching the shape
            and dtype of `src` will be allocated on each worker.

        in_group: bool
            Whether the broadcast operation performs globally or in group as default.

        Returns
        -------
        output_array: DRef

            The DRef containing the broadcasted data on all workers.
            If `dst` was provided, this return value is the same as
            `dst`.  Otherwise, it is the newly allocated space.

        """
        if not isinstance(src, Tensor):
            src = _as_Tensor(src)

        if dst is None:
            dst = self.empty(src.shape, src.dtype)

        src_dref = self.copy_to_worker_0(src)
        self.broadcast_from_worker0(src_dref, dst, in_group)

        return dst

    def broadcast_from_worker0(self, src: DRef, dst: DRef, in_group: bool = True) -> DRef:
        """Broadcast an array from worker-0 to all other workers.

        Parameters
        ----------
        src: Union[np.ndarray, Tensor]
            The array to be broadcasted.

        dst: Optional[DRef]
            The output array.  If None, an array matching the shape
            and dtype of `src` will be allocated on each worker.

        in_group: bool
            Whether the broadcast operation performs globally or in group as default.
        """
        func = self._get_cached_method("runtime.disco.broadcast_from_worker0")
        func(src, in_group, dst)

    def scatter(
        self,
        src: np.ndarray | Tensor,
        dst: DRef | None = None,
        in_group: bool = True,
    ) -> DRef:
        """Scatter an array across all workers

        Parameters
        ----------
        src: Union[np.ndarray, Tensor]
            The array to be scattered.  The first dimension of this
            array, `src.shape[0]`, must be equal to the number of
            workers.

        dst: Optional[DRef]
            The output array.  If None, an array with compatible shape
            and the same dtype as `src` will be allocated on each
            worker.

        in_group: bool
            Whether the scatter operation performs globally or in group as default.

        Returns
        -------
        output_array: DRef

            The DRef containing the scattered data on all workers.
            If `dst` was provided, this return value is the same as
            `dst`.  Otherwise, it is the newly allocated space.

        """
        assert src.shape[0] == self.num_workers

        if not isinstance(src, Tensor):
            src = _as_Tensor(src)

        if dst is None:
            dst = self.empty(src.shape[1:], src.dtype)

        src_dref = self.copy_to_worker_0(src)
        self.scatter_from_worker0(src_dref, dst, in_group)

        return dst

    def scatter_from_worker0(self, from_array: DRef, to_array: DRef, in_group: bool = True) -> None:
        """Scatter an array from worker-0 to all other workers.

        Parameters
        ----------
        src: Union[np.ndarray, Tensor]
            The array to be scattered.  The first dimension of this
            array, `src.shape[0]`, must be equal to the number of
            workers.

        dst: Optional[DRef]
            The output array.  If None, an array with compatible shape
            and the same dtype as `src` will be allocated on each
            worker.

        in_group: bool
            Whether the scatter operation performs globally or in group as default.
        """
        func = self._get_cached_method("runtime.disco.scatter_from_worker0")
        func(from_array, in_group, to_array)

    def gather_to_worker0(self, from_array: DRef, to_array: DRef, in_group: bool = True) -> None:
        """Gather an array from all other workers to worker-0.

        Parameters
        ----------
        from_array : DRef
            The array to be gathered from.

        to_array : DRef
            The array to be gathered to.

        in_group: bool
            Whether the gather operation performs globally or in group as default.
        """
        func = self._get_cached_method("runtime.disco.gather_to_worker0")
        func(from_array, in_group, to_array)

    def allreduce(
        self,
        src: DRef,
        dst: DRef,
        op: str = "sum",  # pylint: disable=invalid-name
        in_group: bool = True,
    ) -> DRef:
        """Perform an allreduce operation on an array.

        Parameters
        ----------
        array : DRef
            The array to be reduced.

        op : str = "sum"
            The reduce operation to be performed. Available options are:
            - "sum"
            - "prod"
            - "min"
            - "max"
            - "avg"

        in_group : bool
            Whether the reduce operation performs globally or in group as default.
        """
        if op not in REDUCE_OPS:
            raise ValueError(f"Unsupported reduce op: {op}. Available ops are: {REDUCE_OPS.keys()}")
        op = Shape([REDUCE_OPS[op]])
        func = self._get_cached_method("runtime.disco.allreduce")
        func(src, op, in_group, dst)

    def allgather(
        self,
        src: DRef,
        dst: DRef,
        in_group: bool = True,
    ) -> DRef:
        """Perform an allgather operation on an array.

        Parameters
        ----------
        src : DRef
            The array to be gathered from.

        dst : DRef
            The array to be gathered to.

        in_group : bool
            Whether the reduce operation performs globally or in group as default.
        """
        func = self._get_cached_method("runtime.disco.allgather")
        func(src, in_group, dst)

    def _clear_ipc_memory_pool(self):
        # Clear the IPC memory allocator when the allocator exists.
        name = "runtime.disco.cuda_ipc.cuda_ipc_memory_allocator_clear"
        if get_global_func(name, allow_missing=True) is not None:
            self.call_packed(self.get_global_func(name))


@register_object("runtime.disco.ThreadedSession")
class ThreadedSession(Session):
    """A Disco session backed by multi-threading."""

    def __init__(self, num_workers: int, num_groups: int = 1) -> None:
        """Create a disco session backed by multiple threads in the same process."""
        self.__init_handle_by_constructor__(
            _ffi_api.SessionThreaded,  # type: ignore # pylint: disable=no-member
            num_workers,
            num_groups,
        )


@register_object("runtime.disco.ProcessSession")
class ProcessSession(Session):
    """A Disco session backed by pipe-based multi-processing."""

    def __init__(
        self,
        num_workers: int,
        num_groups: int = 1,
        entrypoint: str = "tvm.exec.disco_worker",
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.SessionProcess,  # type: ignore # pylint: disable=no-member
            num_workers,
            num_groups,
            "runtime.disco.create_process_pool",
            entrypoint,
        )
        self._configure_structlog()

    def _configure_structlog(self) -> None:
        try:
            import structlog  # pylint: disable=import-outside-toplevel
        except ImportError:
            return

        root_logger = logging.getLogger()
        if len(root_logger.handlers) == 1 and isinstance(
            root_logger.handlers[0].formatter, structlog.stdlib.ProcessorFormatter
        ):
            stdlib_formatter = root_logger.handlers[0].formatter
        else:
            stdlib_formatter = None

        stdlib_level = root_logger.level

        full_config = (structlog.get_config(), stdlib_formatter, stdlib_level)

        config = pickle.dumps(full_config)
        func = self.get_global_func("runtime.disco._configure_structlog")
        func(config, os.getpid())


@register_global_func("runtime.disco.create_socket_session_local_workers")
def _create_socket_session_local_workers(num_workers) -> Session:
    """Create the local session for each distributed node over socket session."""
    return ProcessSession(num_workers)


@register_object("runtime.disco.SocketSession")
class SocketSession(Session):
    """A Disco session backed by socket-based multi-node communication."""

    def __init__(
        self,
        num_nodes: int,
        num_workers_per_node: int,
        num_groups: int,
        host: str,
        port: int,
    ) -> None:
        # Connecting remote nodes can block. A regular FFI call releases the GIL
        # so another Python thread can launch them; constructor calls retain it.
        self.__move_handle_from__(
            _ffi_api.SocketSession(  # type: ignore # pylint: disable=no-member
                num_nodes, num_workers_per_node, num_groups, host, port
            )
        )


@register_global_func("runtime.disco._configure_structlog")
def _configure_structlog(pickled_config: bytes, parent_pid: int) -> None:
    """Configure structlog for all disco workers

    The child processes

    Parameters
    ----------
    pickled_config: bytes

        The pickled configuration for structlog

    parent_pid: int

        The PID of the main process.  This is used to restrict the
    """
    if os.getpid() == parent_pid:
        return

    import structlog  # pylint: disable=import-outside-toplevel

    full_config = pickle.loads(pickled_config)
    structlog_config, stdlib_formatter, stdlib_level = full_config

    root_logger = logging.getLogger()

    root_logger.setLevel(stdlib_level)
    if stdlib_formatter is not None:
        handler = logging.StreamHandler()
        handler.setFormatter(stdlib_formatter)
        root_logger.addHandler(handler)

    structlog.configure(**structlog_config)


@register_global_func("runtime.disco._import_python_module")
def _import_python_module(module_name: str) -> None:
    __import__(module_name)


REDUCE_OPS = {
    "sum": 0,
    "prod": 1,
    "min": 2,
    "max": 3,
    "avg": 4,
}
