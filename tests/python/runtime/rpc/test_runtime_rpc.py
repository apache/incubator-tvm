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
# ruff: noqa: E712, F841

import gc
import multiprocessing
import os
import platform
import stat
import sys
import time

import numpy as np
import pytest
import tvm_ffi

pytest.importorskip("tornado")  # tvm.rpc.proxy and tvm.rpc.tracker require tornado

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import rpc
from tvm.rpc.proxy import Proxy
from tvm.rpc.tracker import Tracker
from tvm_tirx.support import cc, utils

if __name__ == "__main__":
    # NOTE: must live here to avoid registering PackedFunc with libtvm_compiler.so twice.
    tvm_tirx.testing.main()


# tkonolige: The issue as I understand it is this: multiprocessing's spawn
# method launches a new process and then imports the relevant modules. This
# means that all registered functions must exist at the top level scope. In
# this file they are, so all is well when we run this file directly.
# However, when run under pytest, the functions aren't registered on the
# server. I believe this is because pytest is also using multiprocessing to
# run individual functions. Somewhere along the way, the imports are being
# lost, so the server ends up not registering the functions.
pytestmark = pytest.mark.skipif(
    # Windows does not support fork so we can enable Windows for testing
    sys.platform.startswith("win") == False and multiprocessing.get_start_method() != "fork",
    reason=(
        "pytest + multiprocessing spawn method causes tvm_ffi.register_global_func to "
        "not work on the rpc.Server."
    ),
)


@pytest.fixture(scope="module")
def native_source():
    return r"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
void add_one(tvm::ffi::TensorView a, tvm::ffi::TensorView b) {
  for (int64_t i = 0; i < a.size(0); ++i) {
    static_cast<float*>(b.data_ptr())[i] = static_cast<float*>(a.data_ptr())[i] + 1;
  }
}
int64_t answer() { return 42; }
int64_t with_arg(int64_t) { return 42; }
void callback(tvm::ffi::Function func) { func(); }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(main, add_one);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(answer, answer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(with_arg, with_arg);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(callback, callback);
"""


@pytest.fixture(scope="module")
def native_compile_options():
    return [
        "-std=c++17",
        "-I" + tvm_ffi.libinfo.find_include_path(),
        "-I" + tvm_ffi.libinfo.find_dlpack_include_path(),
    ]


@pytest.fixture(scope="module")
def native_object(tmp_path_factory, native_source, native_compile_options):
    directory = tmp_path_factory.mktemp("runtime_module")
    source = directory / "module.cc"
    source.write_text(native_source)
    path = directory / "module.o"
    cc.create_shared(str(path), [str(source)], options=["-c", *native_compile_options])
    return path


@pytest.fixture(scope="module")
def native_library(native_object):
    path = native_object.with_suffix(".so")
    cc.create_shared(str(path), [str(native_object)])
    return path


@pytest.fixture(scope="module")
def system_library_source(native_source):
    return (
        native_source
        + r"""
#include <tvm/ffi/extra/c_env_api.h>
struct RegisterSystemLibrary {
  RegisterSystemLibrary() {
    TVMFFIEnvModRegisterSystemLibSymbol("__tvm_ffi_main", (void*)__tvm_ffi_main);
    TVMFFIEnvModRegisterSystemLibSymbol("__tvm_ffi_answer", (void*)__tvm_ffi_answer);
  }
};
static RegisterSystemLibrary register_system_library;
"""
    )


# NOTE: When writing tests, wrap remote related checking in a sub-function
# to ensure all the remote resources destructs before the server terminates


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_simple():
    server = rpc.Server(key="x1")
    client = rpc.connect("127.0.0.1", server.port, key="x1")

    def check_remote():
        f1 = client.get_function("rpc.test.addone")
        assert f1(10) == 11
        f3 = client.get_function("rpc.test.except")

        with pytest.raises(RuntimeError):
            f3("abc")

        f2 = client.get_function("rpc.test.strcat")
        assert f2("abc", 11) == "abc:11"

    check_remote()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_runtime_string():
    server = rpc.Server(key="x1")
    client = rpc.connect("127.0.0.1", server.port, key="x1")

    def check_remote():
        func = client.get_function("rpc.test.runtime_str_concat")
        x = tvm_ffi.core.String("abc")
        y = tvm_ffi.core.String("def")
        assert str(func(x, y)) == "abcdef"

    check_remote()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_array():
    server = rpc.Server()
    remote = rpc.connect("127.0.0.1", server.port)

    def check_remote():
        x = np.ones((3, 4))
        r_cpu = remote.cpu(0).tensor(x)
        assert isinstance(r_cpu.device, rpc.RemoteDevice)
        np.testing.assert_equal(r_cpu.numpy(), x)
        fremote = remote.get_function("rpc.test.remote_tensor_func")
        fremote(r_cpu)

    check_remote()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_large_array():
    # testcase of large array creation
    server = rpc.Server()
    remote = rpc.connect("127.0.0.1", server.port)

    def check_remote():
        dev = remote.cpu(0)
        a_np = np.ones((5041, 720)).astype("float32")
        b_np = np.ones((720, 192)).astype("float32")
        a = dev.tensor(a_np)
        b = dev.tensor(b_np)
        np.testing.assert_equal(a.numpy(), a_np)
        np.testing.assert_equal(b.numpy(), b_np)

    check_remote()


@pytest.mark.skipif("32bit" in platform.architecture()[0], reason="skipping test for i386.")
@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_echo():
    def check(remote, local_session):
        fecho = remote.get_function("testing.echo")
        assert fecho(1, 2, 3) == 1
        assert fecho(100, 2, 3) == 100
        assert fecho("xyz") == "xyz"
        assert bytes(fecho(bytearray(b"123"))) == b"123"

        with pytest.raises(RuntimeError):
            raise_err = remote.get_function("testing.test_raise_error")
            raise_err("RuntimeError", "msg")

        remote.cpu().sync()
        # tests around system lib are not threadsafe by design
        # and do not work well with multithread pytest
        # skip local session as they are being tested elsewhere
        if not local_session:
            with pytest.raises(AttributeError):
                f3 = remote.system_lib()["notexist"]

    temp = rpc.server._server_env([])
    server = rpc.Server()
    client = rpc.connect("127.0.0.1", server.port)
    check(rpc.LocalSession(), True)

    check(client, False)


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_file_exchange():
    server = rpc.Server()
    remote = rpc.connect("127.0.0.1", server.port)

    def check_remote():
        blob = bytearray(np.random.randint(0, 10, size=(10)))
        remote.upload(blob, "dat.bin")
        rev = remote.download("dat.bin")
        assert rev == blob

    check_remote()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_remote_module(native_object, system_library_source, tmp_path):
    module = tvm_tirx.codegen.load_static_library(
        str(native_object), ["main", "answer", "with_arg"]
    )
    path_dso = tmp_path / "dev_lib.so"
    path_tar = tmp_path / "dev_lib.tar"
    tvm_tirx.codegen.Executable(module).export_library(str(path_dso))
    tvm_tirx.codegen.Executable(module).export_library(str(path_tar))

    def check_remote(remote, name):
        dev = remote.cpu(0)
        for artifact in (path_dso, path_tar, native_object):
            remote.upload(str(artifact))
            loaded = remote.load_module(artifact.name)
            a = dev.tensor(np.random.uniform(size=102).astype("float32"))
            b = dev.tensor(np.zeros(102, dtype="float32"))
            rpc.time_evaluator(loaded, "main", dev, number=10)(a, b)
            np.testing.assert_equal(b.numpy(), a.numpy() + 1)

        for artifact in (path_tar, native_object):
            path = tmp_path / f"download-{name}-{artifact.name}.so"
            path.write_bytes(remote.download_linked_module(artifact.name))
            downloaded = tvm_tirx.runtime.load_module(str(path))
            a = tvm_tirx.runtime.tensor(np.random.uniform(size=102).astype("float32"))
            b = tvm_tirx.runtime.tensor(np.zeros(102, dtype="float32"))
            downloaded(a, b)
            np.testing.assert_equal(b.numpy(), a.numpy() + 1)

    server0 = rpc.Server(key="x0")
    server1 = rpc.Server(key="x1")
    try:
        client = rpc.connect(
            "127.0.0.1",
            server0.port,
            key="x0",
            session_constructor_args=["rpc.Connect", "127.0.0.1", server1.port, "x1", False],
        )
        check_remote(rpc.LocalSession(), "local")
        check_remote(client, "tcp")
        del client
    finally:
        server0.terminate()
        server1.terminate()

    if tvm_ffi.get_global_func("rpc.CreatePipeClient", allow_missing=True) is None:
        return
    source = tmp_path / "system.cc"
    source.write_text(system_library_source)
    path_minrpc = tmp_path / "dev_lib.minrpc"
    rpc.with_minrpc(cc.create_executable)(str(path_minrpc), [str(source)])
    with pytest.raises(RuntimeError):
        rpc.PopenSession("filenotexist")
    remote = rpc.PopenSession(str(path_minrpc))
    dev = remote.cpu(0)
    loaded = remote.system_lib()
    a = dev.tensor(np.random.uniform(size=102).astype("float32"))
    b = dev.tensor(np.zeros(102, dtype="float32"))
    rpc.time_evaluator(loaded, "main", dev, number=1)(a, b)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)
    del a, b, loaded, dev, remote
    os.chmod(path_minrpc, stat.S_IRUSR)
    with pytest.raises(RuntimeError):
        rpc.PopenSession(str(path_minrpc))


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_return_func():
    server = rpc.Server(key="x1")
    client = rpc.connect("127.0.0.1", server.port, key="x1")

    def check_remote():
        f1 = client.get_function("rpc.test.add_to_lhs")
        fadd = f1(10)
        assert fadd(12) == 22

    check_remote()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_session_constructor_args():
    # start server
    server0 = rpc.Server(key="x0")
    server1 = rpc.Server(key="x1")

    def check_multi_hop():
        # use server0 as proxy to connect to server1
        client = rpc.connect(
            "127.0.0.1",
            server0.port,
            key="x0",
            session_constructor_args=["rpc.Connect", "127.0.0.1", server1.port, "x1", False],
        )

        fecho = client.get_function("testing.echo")
        assert fecho(1, 2, 3) == 1
        assert fecho(100, 2, 3) == 100
        assert fecho("xyz") == "xyz"
        assert bytes(fecho(bytearray(b"123"))) == b"123"

        nd = client.cpu(0).tensor([1, 2, 3])
        assert nd.numpy()[1] == 2

    def check_error_handling():
        with pytest.raises(RuntimeError):
            client = rpc.connect(
                "127.0.0.1",
                server0.port,
                key="x0",
                session_constructor_args=["rpc.NonExistingConstructor"],
            )

    check_multi_hop()
    check_error_handling()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_return_tensor():
    def run_arr_test():
        server = rpc.Server(key="x1")
        client = rpc.connect("127.0.0.1", server.port, key="x1")
        m = client.get_function("rpc.test.remote_return_nd")
        get_arr = m("get_arr")
        get_elem = m("get_elem")
        get_arr_elem = m("get_arr_elem")

        arr = get_arr()
        assert get_elem(0) == 0.0
        assert get_arr_elem(arr, 0) == 0.0

        del arr
        gc.collect()
        assert get_elem(0) == 0.0

    run_arr_test()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_rpc_return_remote_object():
    def check(client, is_local):
        make_shape = client.get_function("ffi.Shape")
        get_elem = client.get_function("rpc.testing.GetShapeElem")
        get_size = client.get_function("rpc.testing.GetShapeSize")
        shape = make_shape(2, 3)
        assert get_elem(shape, 0) == 2
        assert get_elem(shape, 1) == 3
        assert get_size(shape) == 2
        # Test free object by assigning to the same variable
        shape = make_shape(0)
        assert get_size(shape) == 1
        assert get_elem(shape, 0) == 0

    # start server

    check(rpc.LocalSession(), True)

    def check_remote():
        server = rpc.Server(key="x1")
        client = rpc.connect("127.0.0.1", server.port, key="x1")
        check(client, False)

    check_remote()

    def check_minrpc():
        if tvm_ffi.get_global_func("rpc.CreatePipeClient", allow_missing=True) is None:
            return
        # Test minrpc server.
        temp = utils.tempdir()
        minrpc_exec = temp.relpath("minrpc")
        tvm.rpc.with_minrpc(cc.create_executable)(minrpc_exec, [])
        check(rpc.PopenSession(minrpc_exec), False)
        # minrpc on the remote
        server = rpc.Server()
        client = rpc.connect(
            "127.0.0.1",
            server.port,
            session_constructor_args=["rpc.PopenSession", open(minrpc_exec, "rb").read()],
        )
        check(client, False)

    check_minrpc()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
def test_local_func():
    client = rpc.LocalSession()

    def check_remote():
        f1 = client.get_function("rpc.test.add_to_lhs")
        fadd = f1(10)
        assert fadd(12) == 22

        blob = bytearray(np.random.randint(0, 10, size=(10)))
        client.upload(blob, "dat.bin")
        rev = client.download("dat.bin")
        assert rev == blob

    check_remote()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
@pytest.mark.parametrize("device_key", ["test_device", "127.0.0.1:5555"])
def test_rpc_tracker_register(device_key):
    # test registration
    tracker = Tracker(port=9000, port_end=10000)
    server1 = rpc.Server(
        host="127.0.0.1",
        port=9000,
        port_end=10000,
        key=device_key,
        tracker_addr=("127.0.0.1", tracker.port),
    )
    server2 = rpc.Server(
        host="127.0.0.1",
        port=9000,
        port_end=10000,
        key=device_key,
        tracker_addr=("127.0.0.1", tracker.port),
        custom_addr="test_addr",  # this is a test address, which is unable to connect
    )
    time.sleep(1)
    client = rpc.connect_tracker("127.0.0.1", tracker.port)

    def exist_address(summary, key, host, port):
        server_info = summary["server_info"]
        for device in server_info:
            if device["key"] == f"server:{key}":
                addr = device["addr"]
                if (host is None or host == addr[0]) and port == addr[1]:
                    return True
        return False

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 2
    assert exist_address(summary, device_key, "127.0.0.1", server1.port)
    assert exist_address(summary, device_key, "test_addr", server2.port)

    remote = client.request(device_key)
    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 1

    del remote
    time.sleep(1)

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 2

    server1.terminate()
    time.sleep(1)

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 1
    assert not exist_address(summary, device_key, "127.0.0.1", server1.port)
    assert exist_address(summary, device_key, "test_addr", server2.port)

    server2.terminate()
    time.sleep(1)

    summary = client.summary()
    assert summary["queue_info"][device_key]["free"] == 0
    assert not exist_address(summary, device_key, "test_addr", server2.port)

    tracker.terminate()


def _target(host, port, device_key, timeout, acquired):
    client = rpc.connect_tracker(host, port)
    remote = client.request(device_key, session_timeout=timeout)
    acquired.set()
    multiprocessing.Event().wait()
    remote.cpu()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
@pytest.mark.parametrize("device_key", ["test_device", "127.0.0.1:5555"])
def test_rpc_tracker_request(device_key):
    tracker = Tracker(port=9000, port_end=10000)
    server = rpc.Server(
        port=9000,
        port_end=10000,
        key=device_key,
        tracker_addr=("127.0.0.1", tracker.port),
    )
    client = rpc.connect_tracker("127.0.0.1", tracker.port)
    acquired1 = multiprocessing.Event()
    acquired2 = multiprocessing.Event()
    proc1 = multiprocessing.Process(
        target=_target, args=("127.0.0.1", tracker.port, device_key, 4, acquired1)
    )
    proc2 = multiprocessing.Process(
        target=_target, args=("127.0.0.1", tracker.port, device_key, 200, acquired2)
    )
    try:
        proc1.start()
        assert acquired1.wait(20), "first RPC session was not acquired"
        proc2.start()
        deadline = time.monotonic() + 10
        while True:
            queue = client.summary()["queue_info"][device_key]
            if queue["pending"] == 1 or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        assert queue["free"] == 0
        assert queue["pending"] == 1
        proc1.terminate()
        proc1.join()
        assert acquired2.wait(20), "pending RPC request did not acquire the released session"
        queue = client.summary()["queue_info"][device_key]
        assert queue["free"] == 0
        assert queue["pending"] == 0
    finally:
        for process in (proc1, proc2):
            if process.pid is not None:
                process.terminate()
                process.join()
        server.terminate()
        tracker.terminate()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
@pytest.mark.parametrize("device_key", ["test_device", "127.0.0.1:5555"])
def test_rpc_tracker_via_proxy(device_key):
    """
         tracker
         /     \
    Host   --   Proxy -- RPC server
    """

    tracker_server = Tracker(port=9000, port_end=9100)
    proxy_server = Proxy(
        host=tracker_server.host,
        port=8888,
        port_end=8988,
        tracker_addr=(tracker_server.host, tracker_server.port),
    )

    server1 = rpc.Server(
        host=proxy_server.host,
        port=proxy_server.port,
        key=device_key,
        tracker_addr=(tracker_server.host, tracker_server.port),
        is_proxy=True,
    )
    server2 = rpc.Server(
        host=proxy_server.host,
        port=proxy_server.port,
        key=device_key,
        tracker_addr=(tracker_server.host, tracker_server.port),
        is_proxy=True,
    )

    client = rpc.connect_tracker(tracker_server.host, tracker_server.port)
    remote1 = client.request(device_key, session_timeout=30)  # pylint: disable=unused-variable
    remote2 = client.request(device_key, session_timeout=30)  # pylint: disable=unused-variable

    server2.terminate()
    server1.terminate()
    proxy_server.terminate()
    tracker_server.terminate()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
@pytest.mark.parametrize("with_proxy", (True, False))
def test_rpc_session_timeout_error(with_proxy):
    port = 9000
    port_end = 10000

    tracker = Tracker(port=port, port_end=port_end)
    time.sleep(0.5)
    tracker_addr = (tracker.host, tracker.port)

    if with_proxy:
        proxy = Proxy(host="0.0.0.0", port=port, port_end=port_end, tracker_addr=tracker_addr)
        time.sleep(0.5)
        server = rpc.Server(host=proxy.host, port=proxy.port, is_proxy=True, key="x1")
    else:
        server = rpc.Server(port=port, port_end=port_end, tracker_addr=tracker_addr, key="x1")
    time.sleep(0.5)

    try:
        # Leave room for server startup before testing expiration of an established session.
        rpc_sess = rpc.connect_tracker(*tracker_addr).request(key="x1", session_timeout=5)
        f1 = rpc_sess.get_function("rpc.test.addone")
        time.sleep(6)
        with pytest.raises(RuntimeError):
            f1(10)
    finally:
        server.terminate()
        if with_proxy:
            proxy.terminate()
        tracker.terminate()


@pytest.mark.skipif(
    tvm_ffi.get_global_func("rpc.LocalSession", allow_missing=True) is None, reason="need rpc"
)
@pytest.mark.parametrize("call_with_unused_argument", [True, False])
def test_remote_function_with_zero_arguments(native_library, call_with_unused_argument):
    server = rpc.Server(key="x1")
    try:
        client = rpc.connect("127.0.0.1", server.port, key="x1")
        client.upload(str(native_library))
        module = client.load_module(native_library.name)
        if call_with_unused_argument:
            result = module["with_arg"](0)
        else:
            result = module["answer"]()
        assert result == 42
        del module, client
    finally:
        server.terminate()


def test_min_repeat_ms(native_library):
    calls = []

    def callback():
        time.sleep(0.1)
        calls.append(1)

    module = tvm_tirx.runtime.load_module(str(native_library))
    tvm.runtime.time_evaluator(module, "callback", tvm_tirx.cpu(), number=1, repeat=1)(callback)
    assert len(calls) == 2
    tvm.runtime.time_evaluator(
        module, "callback", tvm_tirx.cpu(), number=1, repeat=1, min_repeat_ms=1000
    )(callback)
    assert len(calls) > 12


def test_benchmark_result():
    result = tvm.runtime.BenchmarkResult([1, 2, 2, 5])
    assert result.mean == 2.5
    assert result.median == 2.0
    assert result.min == 1
    assert result.max == 5
    assert result.std == 1.5
