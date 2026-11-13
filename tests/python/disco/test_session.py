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
"""Basic tests for a Disco session"""

# pylint: disable=missing-docstring
import socket
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pytest
from tvm_ffi import Shape
from tvm_ffi.core import String

import tvm
import tvm_tirx
import tvm_tirx.testing

# Imported for the side effect of registering the tests.disco.* worker functions.
from tvm.exec import disco_worker as _  # noqa: F401  # pylint: disable=unused-import
from tvm.relax import script as R
from tvm.runtime import disco as di
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

if di is None:
    pytest.skip("disco runtime is not available", allow_module_level=True)


_SOCKET_SESSION_TESTER = None


def _get_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class SocketSessionTester:
    """Run a disco SocketSession with one local node and remote nodes.

    Each remote node is a `tvm.exec.disco_remote_socket_session` subprocess
    launched with the current Python interpreter.
    """

    def __init__(self, num_workers, num_nodes=2, num_groups=1):
        # Initialize the attributes used by __del__ first, so that teardown is
        # safe even when __init__ raises below.
        self.sess = None
        self.remote_nodes = []
        assert num_workers % num_nodes == 0
        num_workers_per_node = num_workers // num_nodes
        server_host = "localhost"
        server_port = _get_free_port()
        server_exc = []

        def start_server():
            try:
                self.sess = di.SocketSession(
                    num_nodes, num_workers_per_node, num_groups, server_host, server_port
                )
            except Exception as exc:  # pylint: disable=broad-except
                server_exc.append(exc)

        thread = threading.Thread(target=start_server)
        thread.start()

        cmd = "tvm.exec.disco_remote_socket_session"
        for _i in range(num_nodes - 1):
            self.remote_nodes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        cmd,
                        server_host,
                        str(server_port),
                        str(num_workers_per_node),
                    ],
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                )
            )

        thread.join()
        if server_exc:
            raise server_exc[0]

    # Bound at class creation: module globals may already be cleared when
    # __del__ runs during interpreter shutdown.
    _TIMEOUT_EXPIRED = subprocess.TimeoutExpired

    def __del__(self):
        try:
            # Shut down the session first so remote nodes can exit gracefully.
            if self.sess is not None:
                self.sess.shutdown()
        finally:
            for node in self.remote_nodes:
                try:
                    node.wait(timeout=10)
                except self._TIMEOUT_EXPIRED:
                    node.kill()
                    node.wait()


def create_socket_session(num_workers):
    """Create a socket session backed by one local and one remote node.

    The tester is kept alive in a module-level global so that the session
    survives until the next call (or interpreter exit) replaces it.
    """
    global _SOCKET_SESSION_TESTER
    # Rebind (not `del`) so the global stays defined if the constructor raises.
    _SOCKET_SESSION_TESTER = None
    _SOCKET_SESSION_TESTER = SocketSessionTester(num_workers)
    assert _SOCKET_SESSION_TESTER.sess is not None
    return _SOCKET_SESSION_TESTER.sess


def _numpy_to_worker_0(sess: di.Session, np_array: np.array, device):
    x_array = sess.empty(np_array.shape, "float32", device=device)
    host_array = tvm_tirx.runtime.tensor(np_array, device=device)
    sess.copy_to_worker_0(host_array, x_array)
    return x_array


def _numpy_from_worker_0(sess: di.Session, remote_array, shape, dtype):
    host_array = tvm_tirx.runtime.empty(shape, dtype, device=tvm_tirx.cpu())
    sess.copy_from_worker_0(host_array, remote_array)
    sess.sync_worker_0()
    return host_array.numpy()


_all_session_kinds = [di.ThreadedSession, di.ProcessSession, create_socket_session]


@pytest.mark.parametrize("session_kind", _all_session_kinds)
def test_int(session_kind):  # pylint: disable=invalid-name
    num_workers = 4
    sess = session_kind(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.add_one")
    result: di.DRef = func(1)
    for i in range(num_workers):
        assert result.debug_get_from_remote(i) == 2


@pytest.mark.parametrize("session_kind", _all_session_kinds)
def test_float(session_kind):
    num_workers = 4
    sess = session_kind(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.add_one_float")
    result: di.DRef = func(1.5)

    for i in range(num_workers):
        assert result.debug_get_from_remote(i) == 2.0


@pytest.mark.parametrize("session_kind", _all_session_kinds)
def test_tensor(session_kind):
    num_workers = 4
    sess = session_kind(num_workers=num_workers)
    device = tvm_tirx.cpu(0)
    x_np = np.arange(6).astype("float32").reshape([2, 3])
    y_np = np.arange(6).astype("float32").reshape([2, 3]) + 1
    x_disc = _numpy_to_worker_0(sess, x_np, device=device)
    y_disc = sess.get_global_func("tests.disco.add_one_tensor")(x_disc)
    y_nd = _numpy_from_worker_0(sess, y_disc, shape=y_np.shape, dtype=y_np.dtype)
    np.testing.assert_equal(y_nd, y_np)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
def test_string(session_kind):
    num_workers = 4
    sess = session_kind(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.str")
    result: di.DRef = func("hello")

    for i in range(num_workers):
        assert result.debug_get_from_remote(i) == "hello_suffix"


@pytest.mark.parametrize("session_kind", _all_session_kinds)
def test_string_obj(session_kind):
    num_workers = 4
    sess = session_kind(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.str_obj")
    result: di.DRef = func(String("hello"))

    for i in range(num_workers):
        value = result.debug_get_from_remote(i)
        assert isinstance(value, str)
        assert value == "hello_suffix"


@pytest.mark.parametrize("session_kind", _all_session_kinds)
def test_shape_tuple(session_kind):
    num_workers = 4
    sess = session_kind(num_workers=num_workers)
    func: di.DPackedFunc = sess.get_global_func("tests.disco.shape_tuple")
    result: di.DRef = func(Shape([1, 2, 3]))
    for i in range(num_workers):
        value = result.debug_get_from_remote(i)
        assert isinstance(value, Shape)
        assert list(value) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("session_kind", _all_session_kinds)
def test_vm_module(session_kind):
    num_workers = 4
    sess = session_kind(num_workers=num_workers)

    # pylint: disable=invalid-name
    @I.ir_module
    class TestMod:
        @Ts.function
        def transpose(A: T.Tensor((8, 16), "float32"), B: T.Tensor((16, 8), "float32")):
            for i, j in T.grid(16, 8):
                with Ts.sblock("transpose"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @R.function
        def main(A: R.Tensor((8, 16), dtype="float32")) -> R.Tensor((16, 8), dtype="float32"):
            cls = TestMod
            with R.dataflow():
                B = R.call_tir(cls.transpose, (A,), ty_args=[R.Tensor((16, 8), dtype="float32")])
                R.output(B)
            return B

    # pylint: enable=invalid-name
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        device = tvm_tirx.cpu()
        x_np = np.arange(8 * 16).astype("float32").reshape([8, 16])
        y_np = x_np.transpose()

        tvm.compile(TestMod, target="llvm").export_library(path)
        mod = sess.load_vm_module(path, device=device)

        x_disc = _numpy_to_worker_0(sess, x_np, device=device)
        y_disc = mod["main"](x_disc)
        y_nd = _numpy_from_worker_0(sess, y_disc, shape=y_np.shape, dtype=y_np.dtype)
        np.testing.assert_equal(y_nd, y_np)

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        for i in range(num_workers):
            sess._sync_worker(i)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
def test_vm_multi_func(session_kind):
    num_workers = 4
    sess = session_kind(num_workers=num_workers)

    # pylint: disable=invalid-name
    @I.ir_module
    class TestMod:
        @Ts.function
        def t1(A: T.Tensor((8, 16), "float32"), B: T.Tensor((16, 8), "float32")):
            for i, j in T.grid(16, 8):
                with Ts.sblock("t1"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @Ts.function
        def t2(A: T.Tensor((16, 8), "float32"), B: T.Tensor((8, 16), "float32")):
            for i, j in T.grid(8, 16):
                with Ts.sblock("t2"):
                    vi, vj = Ts.axis.remap("SS", [i, j])
                    B[vi, vj] = A[vj, vi]

        @R.function
        def transpose_1(A: R.Tensor((8, 16), dtype="float32")) -> R.Tensor(
            (16, 8), dtype="float32"
        ):
            R.func_attr({"global_symbol": "transpose_1"})
            cls = TestMod
            with R.dataflow():
                B = R.call_tir(cls.t1, (A,), ty_args=[R.Tensor((16, 8), dtype="float32")])
                R.output(B)
            return B

        @R.function
        def transpose_2(A: R.Tensor((16, 8), dtype="float32")) -> R.Tensor(
            (8, 16), dtype="float32"
        ):
            R.func_attr({"global_symbol": "transpose_2"})
            cls = TestMod
            with R.dataflow():
                B = R.call_tir(cls.t2, (A,), ty_args=[R.Tensor((8, 16), dtype="float32")])
                R.output(B)
            return B

    # pylint: enable=invalid-name
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir + "/test.so"
        device = tvm_tirx.cpu()
        x_np = np.arange(8 * 16).astype("float32").reshape([8, 16])
        y_np = x_np.transpose()

        tvm.compile(TestMod, target="llvm").export_library(path)
        mod = sess.load_vm_module(path, device=device)

        x_disc = _numpy_to_worker_0(sess, x_np, device=device)
        y_disc = mod["transpose_1"](x_disc)
        z_disc = mod["transpose_2"](y_disc)
        y_nd = _numpy_from_worker_0(sess, y_disc, shape=y_np.shape, dtype=y_np.dtype)
        z_nd = _numpy_from_worker_0(sess, z_disc, shape=x_np.shape, dtype=x_np.dtype)
        np.testing.assert_equal(y_nd, y_np)
        np.testing.assert_equal(z_nd, x_np)

        # sync all workers to make sure the temporary files are cleaned up after all workers
        # finish the execution
        for i in range(num_workers):
            sess._sync_worker(i)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_num_workers(session_kind, num_workers):
    if session_kind == create_socket_session and num_workers < 2:
        return
    sess = session_kind(num_workers=num_workers)
    assert sess.num_workers == num_workers


if __name__ == "__main__":
    tvm_tirx.testing.main()
