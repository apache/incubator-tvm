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
# ruff: noqa: E722
"""Configure pytest"""

# pylint: disable=invalid-name
import threading

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import rpc, te
from tvm.contrib import random
from tvm.testing import device_enabled


def test_randint():
    """Tests randint function"""
    m = 10240
    n = 10240
    A = random.randint(-127, 128, size=(m, n), dtype="int32")

    def verify(target="llvm"):
        if not device_enabled(target):
            print(f"skip because {target} is not enabled...")
            return
        if not tvm_ffi.get_global_func("tvm.contrib.random.randint", True):
            print("skip because extern function is not available")
            return
        dev = tvm_tirx.cpu(0)
        f = tvm.compile(te.create_function([A]), target=target)
        a = tvm_tirx.runtime.tensor(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na)) < 0.3
        assert np.min(na) == -127
        assert np.max(na) == 127

    verify()


def test_uniform():
    """Tests uniform function"""
    m = 10240
    n = 10240
    A = random.uniform(0, 1, size=(m, n))

    def verify(target="llvm"):
        if not device_enabled(target):
            print(f"skip because {target} is not enabled...")
            return
        if not tvm_ffi.get_global_func("tvm.contrib.random.uniform", True):
            print("skip because extern function is not available")
            return
        dev = tvm_tirx.cpu(0)
        f = tvm.compile(te.create_function([A]), target=target)
        a = tvm_tirx.runtime.tensor(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na) - 0.5) < 1e-1
        assert abs(np.min(na) - 0.0) < 1e-3
        assert abs(np.max(na) - 1.0) < 1e-3

    verify()


def test_normal():
    """Tests normal function"""
    m = 10240
    n = 10240
    A = random.normal(3, 4, size=(m, n))

    def verify(target="llvm"):
        if not device_enabled(target):
            print(f"skip because {target} is not enabled...")
            return
        if not tvm_ffi.get_global_func("tvm.contrib.random.normal", True):
            print("skip because extern function is not available")
            return
        dev = tvm_tirx.cpu(0)
        f = tvm.compile(te.create_function([A]), target=target)
        a = tvm_tirx.runtime.tensor(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na) - 3) < 1e-1
        assert abs(np.std(na) - 4) < 1e-2

    verify()


@pytest.mark.gpu
@pytest.mark.parametrize(
    "target",
    [
        "llvm",
        "cuda",
        "nvptx",
        {"kind": "vulkan", "from_device": 0},
        "opencl",
        {"kind": "opencl", "device": "mali"},
        {"kind": "opencl", "device": "intel_graphics"},
        "metal",
        "rocm",
        "hexagon",
    ],
)
def test_random_fill(target):
    """Tests random_fill function"""
    if not device_enabled(target):
        pytest.skip("target is unavailable")
    if not tvm_ffi.get_global_func("tvm.contrib.random.random_fill", True):
        pytest.skip("random_fill is not built")
    dev = tvm_tirx.device_from_target(target)

    def test_local(dev, dtype):
        if not tvm_ffi.get_global_func("tvm.contrib.random.random_fill", True):
            print("skip because extern function is not available")
            return
        value = tvm_tirx.runtime.empty((512, 512), dtype, dev)
        random_fill = tvm_ffi.get_global_func("tvm.contrib.random.random_fill")
        random_fill(value)

        assert np.count_nonzero(value.numpy()) == 512 * 512

        # make sure arithmentic doesn't overflow too
        np_values = value.numpy()
        assert np.isfinite(np_values * np_values + np_values).any()

    def test_rpc(dtype):
        if not tvm_ffi.get_global_func("tvm.contrib.random.random_fill", True):
            print("skip because extern function is not available")
            return
        if not device_enabled("rpc") or not tvm_tirx.runtime.enabled("llvm"):
            return

        def check_remote(server):
            remote = rpc.connect(server.host, server.port)
            value = remote.cpu().empty((512, 512), dtype)
            random_fill = remote.get_function("tvm.contrib.random.random_fill")
            random_fill(value)

            assert np.count_nonzero(value.numpy()) == 512 * 512

            # make sure arithmentic doesn't overflow too
            np_values = value.numpy()
            assert np.isfinite(np_values * np_values + np_values).any()

        check_remote(rpc.Server("127.0.0.1"))

    # Packed sub-byte dtypes (e.g. int4) are intentionally unsupported by
    # random_fill since #19714 and raise an error instead.
    for dtype in [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "int32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]:
        if tvm_tirx.target.Target(target).kind.name == "llvm":
            test_local(dev, dtype)
        else:
            tvm_tirx.testing.run_with_gpu_lock(test_local, dev, dtype)
        test_rpc(dtype)


def test_random_fill_mt():
    """Check random filler applicability in case of nontrivial thread pool configuration.
    Particularly when MaxConcurrency != num_workers_used_ which is actual for big-little systems.
    """
    no_exception_happened = True

    def test_body():
        try:
            num_thread_used = 1
            configure_threads = tvm_ffi.get_global_func("runtime.config_threadpool")
            configure_threads(1, num_thread_used)

            test_input = tvm_tirx.runtime.empty((10, 10))
            random_fill = tvm_ffi.get_global_func("tvm.contrib.random.random_fill_for_measure")
            random_fill(test_input)
        except:  # pylint: disable=bare-except
            nonlocal no_exception_happened
            no_exception_happened = False

    # ThreadPool object is thread local. To eliminate effect on other test cases put it into thread
    x = threading.Thread(target=test_body)
    x.start()
    x.join()
    assert no_exception_happened


if __name__ == "__main__":
    tvm_tirx.testing.main()
