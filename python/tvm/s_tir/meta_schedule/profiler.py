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
# pylint: disable=used-before-assignment
"""A context manager that profiles tuning time cost for different parts."""

from contextlib import contextmanager
from typing import Optional

import tvm_ffi
from tvm_ffi import register_object

from . import _ffi_api


@register_object("s_tir.meta_schedule.Profiler")
class Profiler(tvm_ffi.Object):
    """Tuning time profiler."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Profiler,  # type: ignore # pylint: disable=no-member
        )

    def get(self) -> dict[str, float]:
        """Get the profiling results in seconds"""
        results = _ffi_api.ProfilerGet(self)  # type: ignore # pylint: disable=no-member
        return {str(name): float(value.value) for name, value in results.items()}

    def table(self) -> str:
        """Get the profiling results in a table format"""
        return _ffi_api.ProfilerTable(self)  # type: ignore # pylint: disable=no-member

    def __enter__(self) -> "Profiler":
        """Entering the scope of the context manager"""
        _ffi_api.ProfilerEnterWithScope(self)  # type: ignore # pylint: disable=no-member
        return self

    def __exit__(self, ptype, value, trace) -> None:
        """Exiting the scope of the context manager"""
        _ffi_api.ProfilerExitWithScope(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def current() -> Optional["Profiler"]:
        """Get the current profiler."""
        return _ffi_api.ProfilerCurrent()  # type: ignore # pylint: disable=no-member

    @staticmethod
    def timeit(name: str):
        """Timeit a block of code"""

        @contextmanager
        def _timeit():
            try:
                f = _ffi_api.ProfilerTimedScope(name)  # type: ignore # pylint: disable=no-member
                yield
            finally:
                if f:
                    f()

        return _timeit()
