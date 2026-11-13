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
"""Utils to path."""

import os

import tvm_ffi

from tvm import libinfo
from tvm_tirx import libinfo as core_libinfo
from tvm_tirx.support import cc


def find_minrpc_server_libpath(server="posix_popen_server"):
    """Get the path of minrpc server libary.

    Parameters
    ----------
    server : str
        The kind of built in minrpc server.

    Returns
    -------
    path : str
        The path to the min server library.
    """
    curr_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.abspath(os.path.join(curr_dir, "..", "..", ".."))
    roots = [os.path.join(curr_dir, ".."), source_dir]
    if os.environ.get("TVM_HOME"):
        roots.append(os.environ["TVM_HOME"])
    candidates = []
    for root in roots:
        minrpc_dir = os.path.join(root, "src", "runtime", "rpc", "minrpc")
        path = os.path.join(minrpc_dir, server, f"{server}.cc")
        candidates.append(path)
        if os.path.isfile(path):
            return minrpc_dir, path
    raise RuntimeError(f"Cannot find minserver {server}, in candidates {candidates}")


def with_minrpc(compile_func, server="posix_popen_server"):
    """Attach the compiler function with minrpc related options.

    Parameters
    ----------
    compile_func : Union[str, Callable[[str, str, Optional[str]], None]]
        The compilation function to decorate.

    server : str
        The server type.

    Returns
    -------
    fcompile : function
        The return compilation.
    """
    minrpc_dir, server_path = find_minrpc_server_libpath(server)
    runtime_path = libinfo.find_libtvm_runtime_extra()
    core_runtime_path = core_libinfo.find_libtvm_runtime()
    tvm_ffi_path = tvm_ffi.libinfo.find_libtvm_ffi()

    runtime_dir = os.path.abspath(os.path.dirname(runtime_path))
    tvm_ffi_dir = os.path.abspath(os.path.dirname(tvm_ffi_path))
    options = ["-std=c++17"]
    # Make sure the rpath to the libtvm_runtime is set so we can do local tests.
    # Note that however, this approach won't work on remote.
    # Always recommend to link statically.
    options += ["-Wl,-rpath=" + runtime_dir]
    options += ["-Wl,-rpath=" + os.path.dirname(core_runtime_path)]
    options += ["-Wl,-rpath=" + tvm_ffi_dir]
    # Force the linker to retain the runtime shared lib dependency even if no
    # symbol from it is directly referenced. The minrpc server binary only
    # touches tvm_ffi APIs at link time but relies on global functions (e.g.
    # ``rpc.CreateEventDrivenServer``) registered via static initializers
    # inside libtvm_runtime_extra; with the default ``--as-needed`` those
    # registrations would never run.
    options += ["-Wl,--no-as-needed"]
    default_include_paths = [
        libinfo.find_include_path(),
        core_libinfo.find_include_path(),
        tvm_ffi.libinfo.find_include_path(),
        tvm_ffi.libinfo.find_dlpack_include_path(),
    ]
    options += ["-I" + path for path in default_include_paths]
    options += ["-I" + minrpc_dir]
    fcompile = cc.cross_compiler(
        compile_func,
        options=options,
        add_files=[server_path, runtime_path, core_runtime_path, tvm_ffi_path],
    )
    fcompile.__name__ = "with_minrpc"
    fcompile.need_system_lib = True
    return fcompile
