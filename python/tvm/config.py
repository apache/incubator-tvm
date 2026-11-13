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
"""Lightweight compiler and linker configuration for the TVM runtime."""

import argparse
import os
import sys
from pathlib import Path


def _runtime_root() -> Path:
    """Locate this package without importing its native code."""
    return Path(__file__).resolve().parent


def _find_include() -> str:
    package = _runtime_root()
    for path in (package / "include", package.parent.parent / "include"):
        if (path / "tvm" / "runtime").is_dir():
            return str(path.resolve())
    raise RuntimeError("Cannot find TVM runtime headers")


def _library_paths() -> list[Path]:
    package = _runtime_root()
    source = package.parent.parent
    paths = [package / "lib", source / "build" / "lib", source / "lib"]
    if os.environ.get("TVM_LIBRARY_PATH"):
        paths.insert(0, Path(os.environ["TVM_LIBRARY_PATH"]))
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"):
        paths.extend(Path(value) for value in os.environ.get(name, "").split(os.pathsep) if value)
    return paths


def _find_runtime() -> str:
    filename = {"win32": "tvm_runtime_extra.dll", "darwin": "libtvm_runtime_extra.dylib"}.get(
        sys.platform, "libtvm_runtime_extra.so"
    )
    for directory in _library_paths():
        path = directory / filename
        if path.is_file():
            return str(path.resolve())
    raise RuntimeError("Cannot find libtvm_runtime_extra")


def __main__() -> None:
    """Print paths and flags using the same conventions as tvm_ffi.config."""
    parser = argparse.ArgumentParser(description="Get TVM runtime build configuration")
    for name, help_text in (
        ("includedir", "Print include directory"),
        ("libdir", "Print library directory"),
        ("libfiles", "Fully qualified library filenames"),
        ("libs", "Libraries to be linked"),
        ("cflags", "Print C flags"),
        ("cxxflags", "Print C++ flags"),
        ("ldflags", "Print linker flags"),
    ):
        parser.add_argument(f"--{name}", action="store_true", help=help_text)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    if args.includedir:
        print(_find_include())
    if args.libdir or args.libfiles or args.libs or args.ldflags:
        library = Path(_find_runtime())
        link_file = library.with_suffix(".lib") if sys.platform == "win32" else library
    if args.libdir:
        print(library.parent)
    if args.libfiles:
        print(link_file)
    if args.cxxflags:
        print(f"-I{_find_include()} -std=c++20")
    if args.cflags:
        print(f"-I{_find_include()}")
    if args.libs:
        print(link_file if sys.platform == "win32" else "-ltvm_runtime_extra")
    if args.ldflags and sys.platform != "win32":
        print(f"-L{library.parent}")


if __name__ == "__main__":
    __main__()
