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
"""Parent TVM build and environment information."""

import json
import platform
import sys
import textwrap

import tvm_ffi

import tvm


def libinfo():
    """Return native compile-time information when the library provides it."""
    get_libinfo = tvm_ffi.get_global_func("support.GetLibInfo", allow_missing=True)
    return dict(get_libinfo().items()) if get_libinfo is not None else {}


def describe():
    """
    Print out information about TVM and the current Python environment
    """
    info = list((k, v) for k, v in libinfo().items())
    info = dict(sorted(info, key=lambda x: x[0]))
    print("Python Environment")
    sys_version = sys.version.replace("\n", " ")
    uname = platform.uname()
    uname = f"{uname.system} {uname.release} {uname.version} {uname.machine}"
    lines = [
        f"TVM version    = {tvm.__version__}",
        f"Python version = {sys_version} ({sys.maxsize.bit_length() + 1} bit)",
        f"uname          = {uname}",
    ]
    print(textwrap.indent("\n".join(lines), prefix="  "))
    print("CMake Options:")
    print(textwrap.indent(json.dumps(info, indent=2), prefix="  "))
