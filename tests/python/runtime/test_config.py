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
"""The config command must work without importing native runtime dependencies."""

import os
import subprocess
import sys
from pathlib import Path

import tvm
from tvm import config


def test_config_without_native_imports(tmp_path):
    # A startup hook rejects native loads and dependencies even when they are installed.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class BlockNative:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('tvm_ffi', 'torch', 'numpy') or "
        "fullname in ('tvm.base', 'tvm.runtime', 'tvm.backend', 'tvm.driver', 'tvm_tirx.base'):\n"
        "            raise AssertionError(fullname)\n"
        "sys.meta_path.insert(0, BlockNative())\n"
        "def audit(event, args):\n"
        "    if event == 'ctypes.dlopen': raise AssertionError(args)\n"
        "sys.addaudithook(audit)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(tmp_path),
            str(Path(tvm.__file__).parent.parent),
        )
    )
    command = [sys.executable, "-m", "tvm.config"]
    include = subprocess.check_output([*command, "--includedir"], env=env, text=True).strip()
    assert (Path(include) / "tvm" / "runtime").is_dir()
    library = subprocess.check_output([*command, "--libfiles"], env=env, text=True).strip()
    assert Path(library).is_file()
    flags = subprocess.check_output(
        [*command, "--cflags", "--cxxflags", "--libs", "--ldflags"], env=env, text=True
    )
    assert f"-I{include}" in flags
    assert "-std=c++20" in flags


def test_runtime_library_path_fallback(tmp_path, monkeypatch):
    library = tmp_path / Path(config._find_runtime()).name
    library.touch()
    monkeypatch.setattr(config, "_runtime_root", lambda: tmp_path / "tvm")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert config._find_runtime() == str(library)


if __name__ == "__main__":
    import tvm_tirx.testing

    tvm_tirx.testing.main()
