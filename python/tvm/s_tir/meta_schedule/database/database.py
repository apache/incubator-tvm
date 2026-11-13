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
"""TuningRecord database"""

from collections.abc import Callable
from typing import Any, Optional, Union

import tvm_ffi

# isort: off
from typing import Literal

# isort: on

from tvm_ffi import register_object

from tvm.s_tir.schedule import Schedule, Trace
from tvm_tirx.ir.module import IRModule
from tvm_tirx.target import Target

from .. import _ffi_api
from ..arg_info import ArgInfo
from ..utils import _json_de_tvm


@register_object("s_tir.meta_schedule.Workload")
class Workload(tvm_ffi.Object):
    """A workload, i.e. an IRModule and its structural hash.

    Parameters
    ----------
    mod : IRModule
        The workload's IRModule
    """

    mod: IRModule

    def __init__(self, mod: IRModule) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Workload,  # type: ignore # pylint: disable=no-member
            mod,
        )

    def as_json(self) -> Any:
        """Export the workload to JSON as a python object.

        Returns
        -------
        json : Any
            The JSON serialized as a python object (e.g. a Dict or List).
            Use json.dumps() to get the associated json string.
        """
        return _json_de_tvm(_ffi_api.WorkloadAsJSON(self))  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: Any) -> "Workload":
        """Create a workload from a json object.

        Parameters
        ----------
        json_obj : Any
            The json object to parse.

        Returns
        -------
        tuning_record : TuningRecord
            The parsed tuning record.
        """
        return _ffi_api.WorkloadFromJSON(json_obj)  # type: ignore # pylint: disable=no-member


@register_object("s_tir.meta_schedule.TuningRecord")
class TuningRecord(tvm_ffi.Object):
    """The class of tuning records.

    Parameters
    ----------
    trace : tvm_tirx.ir.Trace
        The trace of the tuning record.
    workload : Workload
        The workload of the tuning record.
    run_secs : Optional[List[float]]
        The run time of the tuning record.
    target : Optional[Target]
        The target of the tuning record.
    args_info : Optional[List[ArgInfo]]
        The argument information of the tuning record.
    """

    trace: Trace
    workload: Workload
    run_secs: list[float] | None
    target: Target | None
    args_info: list[ArgInfo] | None

    def __init__(  # type: ignore # pylint: disable=too-many-arguments
        self,
        trace: Trace,
        workload: Workload,
        run_secs: list[float] | None = None,
        target: Target | None = None,
        args_info: list[ArgInfo] | None = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TuningRecord,  # type: ignore # pylint: disable=no-member
            trace,
            workload,
            run_secs,
            target,
            args_info,
        )

    def as_measure_candidate(self) -> Any:
        """Generate a measure candidate given an initial IR module and a trace
        stored in the tuning record.

        Returns
        -------
        candidate : MeasureCandidate
            A generated candidate.
        """
        return _ffi_api.TuningRecordAsMeasureCandidate(self)  # type: ignore # pylint: disable=no-member

    def as_json(self) -> Any:
        """Export the tuning record to a JSON string.

        Returns
        -------
        json_str : str
            The JSON string exported.
        """
        return _json_de_tvm(_ffi_api.TuningRecordAsJSON(self))  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: Any, workload: Workload) -> "TuningRecord":
        """Create a tuning record from a json object.

        Parameters
        ----------
        json_obj : Any
            The json object to parse.
        workload : Workload
            The workload.

        Returns
        -------
        tuning_record : TuningRecord
            The parsed tuning record.
        """
        return _ffi_api.TuningRecordFromJSON(json_obj, workload)  # type: ignore # pylint: disable=no-member


@register_object("s_tir.meta_schedule.Database")
class Database(tvm_ffi.Object):
    """The abstract database interface."""

    DatabaseType = Union["Database", Literal["json", "memory"]]

    def has_workload(self, mod: IRModule) -> bool:
        """Check if the database has the given workload.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.

        Returns
        -------
        result : bool
            Whether the database has the given workload.
        """
        return _ffi_api.DatabaseHasWorkload(self, mod)  # type: ignore # pylint: disable=no-member

    def commit_workload(self, mod: IRModule) -> Workload:
        """Commit a workload to the database if missing.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for or added.

        Returns
        -------
        workload : Workload
            The workload corresponding to the given IRModule.
        """
        return _ffi_api.DatabaseCommitWorkload(self, mod)  # type: ignore # pylint: disable=no-member

    def commit_tuning_record(self, record: TuningRecord) -> None:
        """Commit a tuning record to the database.

        Parameters
        ----------
        record : TuningRecord
            The tuning record to add.
        """
        _ffi_api.DatabaseCommitTuningRecord(self, record)  # type: ignore # pylint: disable=no-member

    def get_top_k(self, workload: Workload, top_k: int) -> list[TuningRecord]:
        """Get the top K valid tuning records of given workload from the database.

        Parameters
        ----------
        workload : Workload
            The workload to be searched for.
        top_k : int
            The number of top records to get.

        Returns
        -------
        top_k_records : List[TuningRecord]
            The top K records.
        """
        return _ffi_api.DatabaseGetTopK(self, workload, top_k)  # type: ignore # pylint: disable=no-member

    def get_all_tuning_records(self) -> list[TuningRecord]:
        """Get all the tuning records from the database.

        Returns
        -------
        tuning_records : List[TuningRecord]
            All tuning records from the database.
        """
        return _ffi_api.DatabaseGetAllTuningRecords(self)  # type: ignore # pylint: disable=no-member

    def __len__(self) -> int:
        """Get the number of records in the database.

        Returns
        -------
        num_records : int
            The number of records in the database
        """
        return _ffi_api.DatabaseSize(self)  # type: ignore # pylint: disable=no-member

    def query_tuning_record(
        self,
        mod: IRModule,
        target: Target,
        workload_name: str,
    ) -> TuningRecord | None:
        """Query the best record of the given workload from the database.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.
        target : Target
            The target to be searched for.
        workload_name : str
            The name of the workload to be searched for.

        Returns
        -------
        tuning_record : Optional[TuningRecord]
            The best record of the given workload; None if not found.
        """
        return _ffi_api.DatabaseQueryTuningRecord(self, mod, target, workload_name)  # type: ignore # pylint: disable=no-member

    def query_schedule(
        self,
        mod: IRModule,
        target: Target,
        workload_name: str,
    ) -> Schedule | None:
        """Query the best schedule of the given workload from the database.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.
        target : Target
            The target to be searched for.
        workload_name : str
            The name of the workload to be searched for.

        Returns
        -------
        schedule : Optional[tvm.s_tir.Schedule]
            The best schedule of the given workload; None if not found.
        """
        return _ffi_api.DatabaseQuerySchedule(self, mod, target, workload_name)  # type: ignore # pylint: disable=no-member

    def query_ir_module(
        self,
        mod: IRModule,
        target: Target,
        workload_name: str,
    ) -> IRModule | None:
        """Query the best IRModule of the given workload from the database.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.
        target : Target
            The target to be searched for.
        workload_name : str
            The name of the workload to be searched for.

        Returns
        -------
        ir_module : Optional[IRModule]
            The best IRModule of the given workload; None if not found.
        """
        return _ffi_api.DatabaseQueryIRModule(self, mod, target, workload_name)  # type: ignore # pylint: disable=no-member

    def dump_pruned(self, destination: "Database") -> None:
        """Dump the pruned database to files of JSONDatabase format.

        Parameters
        ----------
        destination : Database
            The destination database to be dumped to.
        """
        return _ffi_api.DatabaseDumpPruned(  # type: ignore # pylint: disable=no-member
            self, destination
        )

    def query(
        self,
        mod: IRModule,
        target: Target,
        *,
        workload_name: str = "main",
        kind: Literal["schedule"] | Literal["record"] | Literal["ir_module"] = "schedule",
    ) -> Schedule | IRModule | TuningRecord:
        """Query the database to retrieve the best optimization outcome of the given workload.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.
        target : Target
            The target to be searched for.
        kind : str = "schedule" | "record" | "ir_module"
            The kind of the optimization outcome to be returned.

        Returns
        -------
        result : Union[tvm.s_tir.Schedule, IRModule, TuningRecord]
            The best optimization outcome of the given workload.
        """
        if kind == "schedule":
            return self.query_schedule(mod, target, workload_name)
        if kind == "record":
            return self.query_tuning_record(mod, target, workload_name)
        if kind == "ir_module":
            return self.query_ir_module(mod, target, workload_name)
        raise ValueError(f'Unknown kind: {kind}. Candidates are: "schedule", "record", "ir_module"')

    def __enter__(self) -> "Database":
        """Entering the scope of the context manager"""
        _ffi_api.DatabaseEnterWithScope(self)  # type: ignore # pylint: disable=no-member
        return self

    def __exit__(self, ptype, value, trace) -> None:
        """Exiting the scope of the context manager"""
        _ffi_api.DatabaseExitWithScope(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def current() -> Optional["Database"]:
        """Get the current database under scope."""
        return _ffi_api.DatabaseCurrent()  # type: ignore # pylint: disable=no-member

    @staticmethod
    def create(  # pylint: disable=keyword-arg-before-vararg
        kind: (
            Literal["json", "memory", "union", "ordered_union"] | Callable[[Schedule], bool]
        ) = "json",
        *args,
        **kwargs,
    ) -> "Database":
        """Create a Database.

        Parameters
        ----------
        kind : str = "json" | "memory" | "union" | "ordered_union" | Callable[[tvm.s_tir.Schedule],
        bool]
            The kind of the database to be created. The following kinds are supported:
            "json", "memory", "union", "ordered_union", and a custom schedule function.

        Returns
        -------
        database : Database
            The created database.
        """
        from . import (  # pylint: disable=import-outside-toplevel
            JSONDatabase,
            MemoryDatabase,
            OrderedUnionDatabase,
            ScheduleFnDatabase,
            UnionDatabase,
        )

        if callable(kind):
            return ScheduleFnDatabase(kind, *args, **kwargs)  # type: ignore
        if kind == "json":
            return JSONDatabase(*args, **kwargs)
        if kind == "memory":
            return MemoryDatabase(*args, **kwargs)  # type: ignore
        if kind == "union":
            return UnionDatabase(*args, **kwargs)  # type: ignore
        if kind == "ordered_union":
            return OrderedUnionDatabase(*args, **kwargs)  # type: ignore
        raise ValueError(f"Unknown Database: {kind}")


create = Database.create  # pylint: disable=invalid-name


@register_object("s_tir.meta_schedule.PyDatabase")
class _PyDatabase(Database):
    """
    A TVM object database to support customization on the python side.
    This is NOT the user facing class for function overloading inheritance.

    See also: PyDatabase
    """

    def __init__(
        self,
        f_has_workload: Callable | None = None,
        f_commit_workload: Callable | None = None,
        f_commit_tuning_record: Callable | None = None,
        f_get_top_k: Callable | None = None,
        f_get_all_tuning_records: Callable | None = None,
        f_query_tuning_record: Callable | None = None,
        f_query_schedule: Callable | None = None,
        f_query_ir_module: Callable | None = None,
        f_size: Callable | None = None,
        module_equality: str = "structural",
    ):
        """Constructor."""

        self.__init_handle_by_constructor__(
            _ffi_api.DatabasePyDatabase,  # type: ignore  # pylint: disable=no-member
            f_has_workload,
            f_commit_workload,
            f_commit_tuning_record,
            f_get_top_k,
            f_get_all_tuning_records,
            f_query_tuning_record,
            f_query_schedule,
            f_query_ir_module,
            f_size,
            module_equality,
        )


class PyDatabase:
    """
    An abstract database with customized methods on the python-side.
    This is the user facing class for function overloading inheritance.

    Note: @derived_object is required for proper usage of any inherited class.
    """

    _tvm_metadata = {
        "cls": _PyDatabase,
        "methods": [
            "has_workload",
            "commit_workload",
            "commit_tuning_record",
            "get_top_k",
            "get_all_tuning_records",
            "query_tuning_record",
            "query_schedule",
            "query_ir_module",
            "__len__",
        ],
    }

    def has_workload(self, mod: IRModule) -> bool:
        """Check if the database has the given workload.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.

        Returns
        -------
        result : bool
            Whether the database has the given workload.
        """
        raise NotImplementedError

    def commit_workload(self, mod: IRModule) -> Workload:
        """Commit a workload to the database if missing.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for or added.

        Returns
        -------
        workload : Workload
            The workload corresponding to the given IRModule.
        """
        raise NotImplementedError

    def commit_tuning_record(self, record: TuningRecord) -> None:
        """Commit a tuning record to the database.

        Parameters
        ----------
        record : TuningRecord
            The tuning record to add.
        """
        raise NotImplementedError

    def get_top_k(self, workload: Workload, top_k: int) -> list[TuningRecord]:
        """Get the top K tuning records of given workload from the database.

        Parameters
        ----------
        workload : Workload
            The workload to be searched for.
        top_k : int
            The number of top records to get.

        Returns
        -------
        top_k_records : List[TuningRecord]
            The top K records.
        """
        raise NotImplementedError

    def get_all_tuning_records(self) -> list[TuningRecord]:
        """Get all the tuning records from the database.

        Returns
        -------
        tuning_records : List[TuningRecord]
            All tuning records from the database.
        """
        raise NotImplementedError

    def query_tuning_record(
        self, mod: IRModule, target: Target, workload_name: str | None = None
    ) -> TuningRecord | None:
        """Query a tuning record from the database.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.
        target : Target
            The target to be searched for.
        workload_name : Optional[str]
            The workload name to be searched for.

        Returns
        -------
        record : Optional[TuningRecord]
            The tuning record corresponding to the given workload.
        """
        # Using self._outer to replace the self pointer
        return _ffi_api.DatabaseQueryTuningRecord(  # type: ignore # pylint: disable=no-member
            self._outer(),
            mod,
            target,
            workload_name,  # type: ignore # pylint: disable=no-member
        )

    def query_schedule(
        self, mod: IRModule, target: Target, workload_name: str | None = None
    ) -> Schedule | None:
        """Query a schedule from the database.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.
        target : Target
            The target to be searched for.
        workload_name : Optional[str]
            The workload name to be searched for.

        Returns
        -------
        schedule : Optional[Schedule]
            The schedule corresponding to the given workload.
        """
        # Using self._outer to replace the self pointer
        return _ffi_api.DatabaseQuerySchedule(  # type: ignore # pylint: disable=no-member
            self._outer(),
            mod,
            target,
            workload_name,  # type: ignore # pylint: disable=no-member
        )

    def query_ir_module(
        self, mod: IRModule, target: Target, workload_name: str | None = None
    ) -> IRModule | None:
        """Query an IRModule from the database.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.
        target : Target
            The target to be searched for.
        workload_name : Optional[str]
            The workload name to be searched for.

        Returns
        -------
        mod : Optional[IRModule]
            The IRModule corresponding to the given workload.
        """
        # Using self._outer to replace the self pointer
        return _ffi_api.DatabaseQueryIRModule(  # type: ignore # pylint: disable=no-member
            self._outer(),
            mod,
            target,
            workload_name,  # type: ignore # pylint: disable=no-member
        )

    def __len__(self) -> int:
        """Get the number of records in the database.

        Returns
        -------
        num_records : int
            The number of records in the database
        """
        raise NotImplementedError
