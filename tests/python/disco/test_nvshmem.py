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
"""Disco NVSHMEM session initialization and symmetric allocation."""

# pylint: disable=missing-docstring
import multiprocessing
import socket
import subprocess
import sys
import threading

import pytest
import tvm_ffi
from tvm_ffi import Shape

import tvm_tirx
import tvm_tirx.testing
from tvm.runtime import disco as di
from tvm.testing import env as tvm_env

if di is None:
    pytest.skip("disco runtime is not available", allow_module_level=True)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not tvm_env.has_nvshmem(), reason="need nvshmem"),
]


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


_all_session_kinds = [di.ProcessSession, create_socket_session]
_all_num_workers = [2, 4]

_SUBPROCESS_TIMEOUT_SEC = 600


def _run_in_fresh_process(target, *args):
    """Run a test body in a freshly spawned process.

    After the first call to `nvshmem_init`, a subsequent call to `nvshmem_init`
    or `nvshmem_init_thread` in the same program results in undefined behavior,
    and worker-0 of a Disco session lives in the calling process. So each test
    body must run in its own process. The 'spawn' start method avoids
    inheriting CUDA state from this process.
    """

    def run_and_check():
        proc = multiprocessing.get_context("spawn").Process(target=target, args=args)
        proc.start()
        proc.join(timeout=_SUBPROCESS_TIMEOUT_SEC)
        if proc.is_alive():
            proc.kill()
            proc.join()
            pytest.fail(
                f"{target.__name__}{args} timed out after {_SUBPROCESS_TIMEOUT_SEC} seconds"
            )
        assert proc.exitcode == 0, f"{target.__name__}{args} failed with exit code {proc.exitcode}"

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def _require_cuda_devices(num_workers):
    # Each nvshmem worker binds its own CUDA device (cudaSetDevice(worker_id)).
    if not all(tvm_tirx.cuda(i).exist for i in range(num_workers)):
        pytest.skip(f"Requires {num_workers} CUDA devices")


def _init_finalize(session_kind, num_workers):
    sess = session_kind(num_workers=num_workers)
    f_init_nvshmem_uid = tvm_ffi.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()
    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()


def _empty(session_kind, num_workers):
    device = tvm_tirx.cuda()
    sess = session_kind(num_workers=num_workers)
    f_init_nvshmem_uid = tvm_ffi.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    uid = f_init_nvshmem_uid()
    init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_dfunc(uid, num_workers, 0)
    sess.sync_worker_0()
    empty_dfunc = sess.get_global_func("runtime.disco.nvshmem.empty")
    _a = empty_dfunc(Shape((32, 64)), "float32", device)
    _b = empty_dfunc(Shape((64, 32)), "float32", device)
    sess.sync_worker_0()
    finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
    finalize_dfunc()
    sess.sync_worker_0()


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("num_workers", _all_num_workers)
def test_nvshmem_init_finalize(session_kind, num_workers: int):
    _require_cuda_devices(num_workers)
    _run_in_fresh_process(_init_finalize, session_kind, num_workers)


@pytest.mark.parametrize("session_kind", _all_session_kinds)
@pytest.mark.parametrize("num_workers", _all_num_workers)
def test_nvshmem_empty(session_kind, num_workers: int):
    _require_cuda_devices(num_workers)
    _run_in_fresh_process(_empty, session_kind, num_workers)


if __name__ == "__main__":
    tvm_tirx.testing.main()
