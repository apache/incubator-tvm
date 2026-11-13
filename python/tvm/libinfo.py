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
"""Library and include path information for TVM."""

from __future__ import annotations

import os
from pathlib import Path

from tvm_ffi import libinfo as tvm_ffi_libinfo


def _rel_top_directory() -> Path:
    """Top directory in the installed (wheel) layout: ``python/tvm/``."""
    return Path(__file__).parent


def _dev_top_directory() -> Path:
    """Top directory in the source/dev layout: the worktree root."""
    return _rel_top_directory() / ".." / ".."


def package_lib_paths() -> list[Path]:
    """Return search directories for TVM's shared libraries.

    Anchored on this file's location (``python/tvm/libinfo.py``), the list
    covers the wheel-install layout (``python/tvm/lib/``) and the in-tree dev
    build layouts (``<worktree>/build/lib/`` and ``<worktree>/lib/``).
    ``TVM_LIBRARY_PATH`` is prepended when set so it takes priority. Callers
    pick the basenames they want (e.g. ``libtvm_runtime_extra.so``) and the load
    mode; this function only returns the search path.
    """
    pkg = _rel_top_directory()  # python/tvm/
    paths: list[Path] = []
    if os.environ.get("TVM_LIBRARY_PATH"):
        paths.append(Path(os.environ["TVM_LIBRARY_PATH"]))
    paths += [
        pkg / "lib",  # wheel layout
        _dev_top_directory() / "build" / "lib",  # dev: <worktree>/build/lib
        _dev_top_directory() / "lib",  # dev: <worktree>/lib
    ]
    return paths


def find_libtvm_runtime_extra() -> str:
    """Return the path to the ``libtvm_runtime_extra`` shared library.

    Returns
    -------
    path : str
        The resolved path to the TVM runtime shared library.

    Raises
    ------
    RuntimeError
        If the runtime library cannot be found.
    """
    candidate = tvm_ffi_libinfo._find_library_by_basename(
        "tvm", "tvm_runtime_extra", extra_lib_paths=package_lib_paths()
    )
    if ret := tvm_ffi_libinfo._resolve_and_validate([candidate], cond=lambda _: True):
        return ret
    raise RuntimeError("Cannot find libtvm_runtime_extra")


def find_include_path() -> str:
    """Return the path to TVM's own ``include/`` directory."""
    if ret := tvm_ffi_libinfo._resolve_and_validate(
        paths=[
            _rel_top_directory() / "include",
            _dev_top_directory() / "include",
        ],
        cond=lambda p: (p / "tvm" / "runtime" / "vm").is_dir(),
    ):
        return ret
    raise RuntimeError("Cannot find TVM include path.")


# The version is written by setuptools_scm into _version.py at build time
# (see [tool.setuptools_scm] in pyproject.toml). The fallback keeps a source
# checkout with no build run importable.
try:
    from ._version import version as __version__
except ImportError:  # pragma: no cover - source tree without a build
    __version__ = "0.28.dev0"
