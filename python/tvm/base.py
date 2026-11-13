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
# coding: utf-8
# pylint: disable=invalid-name, import-outside-toplevel
"""Load parent TVM extensions after their standalone TIRx dependency."""

from pathlib import Path

from tvm_ffi import libinfo as tvm_ffi_libinfo
from tvm_ffi.libinfo import load_lib_ctypes

from tvm_tirx.base import _RUNTIME_ONLY as _CORE_RUNTIME_ONLY

from . import libinfo

_RUNTIME_ONLY = _CORE_RUNTIME_ONLY
_LOADED_LIBS = {
    "tvm_runtime_extra": load_lib_ctypes(
        "tvm", "tvm_runtime_extra", "RTLD_GLOBAL", extra_lib_paths=libinfo.package_lib_paths()
    )
}
if not _RUNTIME_ONLY:
    _lib_paths = [Path(_LOADED_LIBS["tvm_runtime_extra"]._name).resolve().parent]
    try:
        tvm_ffi_libinfo._find_library_by_basename("tvm", "tvm_compiler", extra_lib_paths=_lib_paths)
    except RuntimeError:
        _RUNTIME_ONLY = True
    else:
        _LOADED_LIBS["tvm_compiler"] = load_lib_ctypes(
            "tvm", "tvm_compiler", "RTLD_LOCAL", extra_lib_paths=_lib_paths
        )
