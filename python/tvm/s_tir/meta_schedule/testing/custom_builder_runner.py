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
"""Customized builder and runner methods"""

# pylint: disable=import-outside-toplevel
from collections.abc import Callable

import numpy as np  # type: ignore
import tvm_ffi

from tvm.s_tir.meta_schedule.runner import RPCConfig
from tvm_tirx.codegen import Executable


def run_module_via_rpc(
    rpc_config: RPCConfig,
    lib: tvm_ffi.Module | Executable,
    dev_type: str,
    args: dict[int, np.ndarray] | dict[str, np.ndarray],
    continuation: Callable,
):
    """Execute a tvm_ffi.Module on RPC remote"""
    # pylint: disable=import-outside-toplevel
    import os
    import tempfile

    from tvm_tirx.support.tar import tar

    # pylint: enable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = os.path.join(tmp_dir, "tvm_tmp_mod." + tar.output_format)
        executable = lib if isinstance(lib, Executable) else Executable(lib)
        executable.export_library(filename, fcompile=tar)
        session = rpc_config.connect_server()
        session.upload(filename)
        _, filename = os.path.split(filename)
        rt_mod = session.load_module(filename)
        dev = session.device(dev_type, 0)
        nd_args = {k: dev.tensor(v) for k, v in args.items()}
        return continuation(rt_mod, dev, nd_args)
