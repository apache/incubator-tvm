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

"""Tests for tvm.contrib.pickle_memoize"""

import os
import pathlib
import subprocess
import sys
import tempfile

import tvm_tirx.testing

TEST_SCRIPT_FILE = pathlib.Path(__file__).with_name("pickle_memoize_script.py").resolve()


def test_cache_dir_not_in_current_working_dir():
    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        temp_dir = pathlib.Path(temp_dir)
        subprocess.check_call([sys.executable, str(TEST_SCRIPT_FILE), "1", "1"], cwd=temp_dir)

        new_files = list(temp_dir.iterdir())
        assert not new_files, (
            "Use of tvm.contrib.pickle_memorize may not write to current directory."
        )


def test_current_directory_is_not_required_to_be_writable():
    """TVM may be imported without directory permissions

    This is a regression test.  In previous implementations, the
    `tvm.contrib.pickle_memoize.memoize` function would write to the
    current directory when importing TVM.  Import of a Python module
    should not write to any directory.

    """

    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        # User may read/cd into the temp dir, nobody may write to temp
        # dir.
        temp_dir.chmod(0o500)
        subprocess.check_call([sys.executable, "-c", "import tvm, tvm_tirx"], cwd=temp_dir)


def test_cache_dir_defaults_to_home_config_cache():
    with tempfile.TemporaryDirectory(prefix="tvm_") as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        subprocess.check_call([sys.executable, str(TEST_SCRIPT_FILE), "1", "0"], cwd=temp_dir)

        new_files = list(temp_dir.iterdir())
        assert not new_files, (
            "Use of tvm.contrib.pickle_memorize may not write to current directory."
        )

        cache_dir = pathlib.Path.home().joinpath(".cache", "tvm", "pkl_memoize_py3")
        assert cache_dir.exists()
        cache_files = list(cache_dir.iterdir())
        assert len(cache_files) >= 1


def test_cache_dir_respects_xdg_cache_home():
    with (
        tempfile.TemporaryDirectory(prefix="tvm_") as temp_working_dir,
        tempfile.TemporaryDirectory(prefix="tvm_") as temp_cache_dir,
    ):
        temp_cache_dir = pathlib.Path(temp_cache_dir)
        temp_working_dir = pathlib.Path(temp_working_dir)

        subprocess.check_call(
            [sys.executable, str(TEST_SCRIPT_FILE), "1", "0"],
            cwd=temp_working_dir,
            env={
                **os.environ,
                "XDG_CACHE_HOME": temp_cache_dir.as_posix(),
            },
        )

        new_files = list(temp_working_dir.iterdir())
        assert not new_files, (
            "Use of tvm.contrib.pickle_memorize may not write to current directory."
        )

        cache_dir = temp_cache_dir.joinpath("tvm", "pkl_memoize_py3")
        assert cache_dir.exists()
        cache_files = list(cache_dir.iterdir())
        assert len(cache_files) == 1


def test_cache_dir_only_created_when_used():
    with (
        tempfile.TemporaryDirectory(prefix="tvm_") as temp_working_dir,
        tempfile.TemporaryDirectory(prefix="tvm_") as temp_cache_dir,
    ):
        temp_cache_dir = pathlib.Path(temp_cache_dir)
        temp_working_dir = pathlib.Path(temp_working_dir)

        subprocess.check_call(
            [sys.executable, str(TEST_SCRIPT_FILE), "0", "1"],
            cwd=temp_working_dir,
            env={
                **os.environ,
                "XDG_CACHE_HOME": temp_cache_dir.as_posix(),
            },
        )

        cache_dir = temp_cache_dir.joinpath("tvm", "pkl_memoize_py3")
        assert not cache_dir.exists()


if __name__ == "__main__":
    tvm_tirx.testing.main()
