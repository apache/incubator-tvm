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
"""Find checkout-only minrpc sources while importing an installed TVM wheel."""

from pathlib import Path

import pytest

from tvm import libinfo
from tvm.rpc import minrpc


@pytest.mark.parametrize("layout", ["source", "symlink", "wheel"])
def test_minrpc_source_location(tmp_path, monkeypatch, layout):
    checkout = tmp_path / "checkout"
    minrpc_dir = checkout / "src/runtime/rpc/minrpc"
    server = minrpc_dir / "posix_popen_server/posix_popen_server.cc"
    server.parent.mkdir(parents=True)
    server.write_text("// source fixture\n")
    # A wheel's package location is unrelated to the source checkout.
    package = tmp_path / "site-packages/tvm" if layout == "wheel" else checkout / "python/tvm"
    if layout == "symlink":
        package.mkdir(parents=True)
        alias = tmp_path / "python-alias"
        alias.symlink_to(checkout / "python", target_is_directory=True)
        package = alias / "tvm"
    monkeypatch.setattr(libinfo, "__file__", str(package / "libinfo.py"))
    monkeypatch.setattr(minrpc, "__file__", str(package / "rpc/minrpc.py"))
    monkeypatch.delenv("TVM_HOME", raising=False)
    if layout == "wheel":
        monkeypatch.setenv("TVM_HOME", str(checkout))
    directory, path = minrpc.find_minrpc_server_libpath()
    assert Path(directory) == minrpc_dir
    assert Path(path) == server
    server.unlink()
    with pytest.raises(RuntimeError, match="Cannot find minserver"):
        minrpc.find_minrpc_server_libpath()
