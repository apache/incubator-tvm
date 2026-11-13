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
# ruff: noqa: F401
"""RPC server implementation.

Note
----
Server is TCP based with the following protocol:
- Initial handshake to the peer
  - [RPC_MAGIC, keysize(int32), key-bytes]
- The key is in format
   - {server|client}:device-type[:random-key] [-timeout=timeout]
"""

# pylint: disable=invalid-name
import ctypes
import errno
import logging
import multiprocessing
import os
import select
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import tvm_ffi

from tvm_tirx.support import utils
from tvm_tirx.support.popen_pool import PopenWorker

# pylint: disable=unused-import
from . import _ffi_api, base, testing
from .base import TrackerCode

logger = logging.getLogger("RPCServer")
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _link_module(path):
    """Link uploaded object files or archives before loading or downloading them."""
    if path.endswith((".o", ".tar")):
        # Compilation belongs to the RPC server, not the FFI module loader.
        from tvm_tirx.support import cc, tar  # pylint: disable=import-outside-toplevel

        with utils.tempdir() as temp:
            files = tar.normalize_file_list_by_unpacking_tars(temp, [path])
            cc.create_shared(path + ".so", files)
        return path + ".so"
    return path


def _server_env(load_library, work_path=None):
    """Server environment function return temp dir"""
    if work_path:
        temp = work_path
    else:
        temp = utils.tempdir()

    # pylint: disable=unused-variable
    @tvm_ffi.register_global_func("tvm.rpc.server.workpath", override=True)
    def get_workpath(path):
        return temp.relpath(path)

    @tvm_ffi.register_global_func("tvm.rpc.server.load_module", override=True)
    def load_module(file_name):
        """Load module from remote side."""
        path = temp.relpath(file_name)
        m = tvm_ffi.load_module(_link_module(path))
        logger.info("load_module %s", path)
        return m

    @tvm_ffi.register_global_func("tvm.rpc.server.download_linked_module", override=True)
    def download_linked_module(file_name):
        """Load module from remote side."""
        path = temp.relpath(file_name)
        if not path.endswith((".o", ".tar", ".dylib", ".so")):
            raise RuntimeError(f"Do not know how to link {file_name}")
        path = _link_module(path)
        logger.info("Send linked module %s to client", path)
        return bytearray(open(path, "rb").read())

    libs = []
    load_library = load_library.split(":") if load_library else []
    for file_name in load_library:
        # Resolve the literal library file name against the current working
        # directory, preserving the exact filename the caller requested.
        resolved = tvm_ffi.libinfo._resolve_and_validate(
            [Path.cwd() / file_name], cond=lambda p: p.is_file()
        )
        if resolved is None:
            raise RuntimeError(f"Cannot find library {file_name}")
        file_name = resolved
        libs.append(ctypes.CDLL(file_name, ctypes.RTLD_GLOBAL))
        logger.info("Load additional library %s", file_name)
    temp.libs = libs
    return temp


def _serve_loop(sock, load_library, work_path):
    _server_env(load_library, work_path)
    _ffi_api.ServerLoop(sock.fileno())


def _parse_server_opt(opts):
    # parse client options
    ret = {}
    for kv in opts:
        if kv.startswith("-timeout="):
            ret["timeout"] = float(kv[9:])
    return ret


def _serving(sock, addr, opts, load_library):
    logger.info(f"connected from {addr}")
    work_path = utils.tempdir()
    old_cwd = os.getcwd()
    os.chdir(work_path.path)  # Avoiding file name conflict between sessions.
    logger.info(f"start serving at {work_path.path}")

    server_proc = multiprocessing.Process(target=_serve_loop, args=(sock, load_library, work_path))
    server_proc.start()
    server_proc.join(opts.get("timeout", None))  # Wait until finish or timeout.

    if server_proc.is_alive():
        logger.info("timeout in RPC session, kill..")
        _ffi_api.ReturnException(
            sock.fileno(),
            f"RPCSessionTimeoutError: Your {opts['timeout']}s session has expired, "
            f'try to increase the "session_timeout" value.',
        )

        try:
            import psutil  # pylint: disable=import-outside-toplevel

            # Terminate worker children firstly.
            for child in psutil.Process(server_proc.pid).children(recursive=True):
                child.terminate()
        except ImportError:
            # Don't dependent `psutil` hardly, because it isn't a pure Python
            # package and maybe hard to be installed on some platforms.
            pass
        server_proc.terminate()
    elif server_proc.exitcode != 0:
        raise RuntimeError(
            f"Child process {server_proc.pid} exited unsuccessfully "
            f"with error code {server_proc.exitcode}"
        )

    logger.info(f"finish serving {addr}")
    os.chdir(old_cwd)
    work_path.remove()
    sock.close()


def _listen_loop(sock, port, rpc_key, tracker_addr, load_library, custom_addr):
    """Listening loop of the server."""

    def _accept_conn(listen_sock, tracker_conn, ping_period=2):
        """Accept connection from the other places.

        Parameters
        ----------
        listen_sock: Socket
            The socket used by listening process.

        tracker_conn : connection to tracker
            Tracker connection

        ping_period : float, optional
            ping tracker every k seconds if no connection is accepted.
        """
        old_keyset = set()
        # Report resource to tracker
        if tracker_conn:
            matchkey = base.random_key(rpc_key)
            base.sendjson(tracker_conn, [TrackerCode.PUT, rpc_key, (port, matchkey), custom_addr])
            assert base.recvjson(tracker_conn) == TrackerCode.SUCCESS
        else:
            matchkey = rpc_key

        unmatch_period_count = 0
        unmatch_timeout = 4
        # Wait until we get a valid connection
        while True:
            if tracker_conn:
                trigger = select.select([listen_sock], [], [], ping_period)
                if listen_sock not in trigger[0]:
                    base.sendjson(tracker_conn, [TrackerCode.GET_PENDING_MATCHKEYS])
                    pending_keys = base.recvjson(tracker_conn)
                    old_keyset.add(matchkey)
                    # if match key not in pending key set
                    # it means the key is acquired by a client but not used.
                    if matchkey not in pending_keys:
                        unmatch_period_count += 1
                    else:
                        unmatch_period_count = 0
                    # regenerate match key if key is acquired but not used for a while
                    if unmatch_period_count * ping_period > unmatch_timeout + ping_period:
                        logger.info("no incoming connections, regenerate key ...")
                        matchkey = base.random_key(rpc_key, cmap=old_keyset)
                        base.sendjson(
                            tracker_conn, [TrackerCode.PUT, rpc_key, (port, matchkey), custom_addr]
                        )
                        assert base.recvjson(tracker_conn) == TrackerCode.SUCCESS
                        unmatch_period_count = 0
                    continue
            conn, addr = listen_sock.accept()
            magic = struct.unpack("<i", base.recvall(conn, 4))[0]
            if magic != base.RPC_MAGIC:
                conn.close()
                continue
            keylen = struct.unpack("<i", base.recvall(conn, 4))[0]
            key = (base.recvall(conn, keylen)).decode("utf-8")
            arr = key.split()
            expect_header = "client:" + matchkey
            server_key = "server:" + rpc_key
            if arr[0] != expect_header:
                conn.sendall(struct.pack("<i", base.RPC_CODE_MISMATCH))
                conn.close()
                logger.warning("mismatch key from %s", addr)
                continue
            conn.sendall(struct.pack("<i", base.RPC_CODE_SUCCESS))
            conn.sendall(struct.pack("<i", len(server_key)))
            conn.sendall(server_key.encode("utf-8"))
            return conn, addr, _parse_server_opt(arr[1:])

    # Server logic
    tracker_conn = None
    while True:
        try:
            # step 1: setup tracker and report to tracker
            if tracker_addr and tracker_conn is None:
                tracker_conn = base.connect_with_retry(tracker_addr)
                tracker_conn.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
                magic = struct.unpack("<i", base.recvall(tracker_conn, 4))[0]
                if magic != base.RPC_TRACKER_MAGIC:
                    raise RuntimeError(f"{tracker_addr!s} is not RPC Tracker")
                # report status of current queue
                cinfo = {"key": "server:" + rpc_key, "addr": (custom_addr, port)}
                base.sendjson(tracker_conn, [TrackerCode.UPDATE_INFO, cinfo])
                assert base.recvjson(tracker_conn) == TrackerCode.SUCCESS

            # step 2: wait for in-coming connections
            conn, addr, opts = _accept_conn(sock, tracker_conn)
        except OSError:
            # retry when tracker is dropped
            if tracker_conn:
                tracker_conn.close()
                tracker_conn = None
            continue
        except RuntimeError as exc:
            raise exc

        # step 3: serving
        _serving(conn, addr, opts, load_library)


def _connect_proxy_loop(addr, key, load_library):
    key = "server:" + key
    retry_count = 0
    max_retry = 5
    retry_period = 5
    while True:
        try:
            sock = socket.socket(base.get_addr_family(addr), socket.SOCK_STREAM)
            sock.connect(addr)
            sock.sendall(struct.pack("<i", base.RPC_MAGIC))
            sock.sendall(struct.pack("<i", len(key)))
            sock.sendall(key.encode("utf-8"))
            magic = struct.unpack("<i", base.recvall(sock, 4))[0]
            if magic == base.RPC_CODE_DUPLICATE:
                raise RuntimeError(f"key: {key} has already been used in proxy")

            if magic == base.RPC_CODE_MISMATCH:
                logger.warning("RPCProxy do not have matching client key %s", key)
            elif magic != base.RPC_CODE_SUCCESS:
                raise RuntimeError(f"{addr!s} is not RPC Proxy")
            keylen = struct.unpack("<i", base.recvall(sock, 4))[0]
            remote_key = (base.recvall(sock, keylen)).decode("utf-8")

            _serving(sock, addr, _parse_server_opt(remote_key.split()[1:]), load_library)
            retry_count = 0
        except OSError as err:
            retry_count += 1
            logger.warning("Error encountered %s, retry in %g sec", str(err), retry_period)
            if retry_count > max_retry:
                raise RuntimeError(f"Maximum retry error: last error: {err!s}")
            time.sleep(retry_period)


class PopenRPCServerState:
    """Internal PopenRPCServer State"""

    current = None

    def __init__(
        self,
        host,
        port=9091,
        port_end=9199,
        is_proxy=False,
        tracker_addr=None,
        key="",
        load_library=None,
        custom_addr=None,
        silent=False,
        reuse_addr=True,
        timeout=None,
    ):
        # start update
        self.host = host
        self.port = port
        self.libs = []
        self.custom_addr = custom_addr

        if silent:
            logger.setLevel(logging.ERROR)

        if not is_proxy:
            sock = socket.socket(base.get_addr_family((host, port)), socket.SOCK_STREAM)

            # Never set socket SO_REUSEADDR on Windows. The SO_REUSEADDR flag allow reusing the
            # inactivate TIME_WATI state sockets on POSIX, but on Windows it will allow two or more
            # activate sockets to bind on the same address and port if they all set SO_REUSEADDR,
            # and result in indeterminate behavior.
            if reuse_addr and sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if timeout is not None:
                sock.settimeout(timeout)
            self.port = None
            for my_port in range(port, port_end):
                try:
                    sock.bind((host, my_port))
                    self.port = my_port
                    break
                except OSError as sock_err:
                    if sock_err.errno in [errno.EADDRINUSE]:
                        continue
                    raise sock_err
            if not self.port:
                raise ValueError(f"cannot bind to any port in [{port}, {port_end})")
            logger.info("bind to %s:%d", host, self.port)
            sock.listen(1)
            self.sock = sock
            self.thread = threading.Thread(
                target=_listen_loop,
                args=(self.sock, self.port, key, tracker_addr, load_library, self.custom_addr),
            )
            self.thread.start()
        else:
            self.thread = threading.Thread(
                target=_connect_proxy_loop, args=((host, port), key, load_library)
            )
            self.thread.start()


def _popen_start_rpc_server(
    host,
    port=9091,
    port_end=9199,
    is_proxy=False,
    tracker_addr=None,
    key="",
    load_library=None,
    custom_addr=None,
    silent=False,
    no_fork=False,
    server_init_callback=None,
    reuse_addr=True,
    timeout=None,
):
    if no_fork:
        multiprocessing.set_start_method("spawn")

    if server_init_callback:
        server_init_callback()

    # This is a function that will be sent to the
    # Popen worker to run on a separate process.
    # Create and start the server in a different thread
    state = PopenRPCServerState(
        host,
        port,
        port_end,
        is_proxy,
        tracker_addr,
        key,
        load_library,
        custom_addr,
        silent,
        reuse_addr,
        timeout,
    )
    PopenRPCServerState.current = state
    # returns the port so that the main can get the port number.
    return state.port


class Server:
    """Start RPC server on a separate process.

    This is a simple python implementation based on multi-processing.
    It is also possible to implement a similar C based server with
    TVM runtime which does not depend on the python.

    Parameters
    ----------
    host : str
        The host url of the server.

    port : int
        The port to be bind to

    port_end : int, optional
        The end port to search

    is_proxy : bool, optional
        Whether the address specified is a proxy.
        If this is true, the host and port actually corresponds to the
        address of the proxy server.

    tracker_addr: Tuple (str, int) , optional
        The address of RPC Tracker in tuple(host, ip) format.
        If is not None, the server will register itself to the tracker.

    key : str, optional
        The key used to identify the device type in tracker.

    load_library : str, optional
        List of additional libraries to be loaded during execution.

    custom_addr: str, optional
        Custom IP Address to Report to RPC Tracker

    silent: bool, optional
        Whether run this server in silent mode.

    no_fork: bool, optional
        Whether forbid fork in multiprocessing.

    server_init_callback: Callable, optional
        Additional initialization function when starting the server.

    reuse_addr: bool, optional
        Allows the kernel to reuse a local socket in TIME_WAIT state.

    timeout: float, optional
         set a timeout for all operations on the socket

    Note
    ----
    TVM RPC server assumes that the user is trusted and needs to be
    used in a trusted network environment and encrypted channels.
    It allows writings of arbitrary files into the server and provide
    full remote code execution capabilities to anyone who can access this API.

    The RPC server only sees functions in the tvm namespace.
    To bring additional custom functions to the server env, you can use server_init_callback.

    .. code:: python

        def server_init_callback():
            import tvm, tvm_tirx
            # must import mypackage here
            import mypackage

            tvm_ffi.register_global_func("function", mypackage.func)

        server = rpc.Server(host, server_init_callback=server_init_callback)
    """

    def __init__(
        self,
        host="0.0.0.0",
        port=9091,
        port_end=9199,
        is_proxy=False,
        tracker_addr=None,
        key="",
        load_library=None,
        custom_addr=None,
        silent=False,
        no_fork=False,
        server_init_callback=None,
        reuse_addr=True,
        timeout=None,
    ):
        try:
            if _ffi_api.ServerLoop is None:
                raise RuntimeError("Please compile with TVM_USE_RPC=1")
        except NameError:
            raise RuntimeError("Please compile with TVM_USE_RPC=1")
        self.proc = PopenWorker()
        # send the function
        self.proc.send(
            _popen_start_rpc_server,
            [
                host,
                port,
                port_end,
                is_proxy,
                tracker_addr,
                key,
                load_library,
                custom_addr,
                silent,
                no_fork,
                server_init_callback,
                reuse_addr,
                timeout,
            ],
        )
        # receive the port
        self.port = self.proc.recv()
        self.host = host

    def terminate(self):
        """Terminate the server process"""
        if self.proc:
            self.proc.kill()
            self.proc = None

    def __del__(self):
        try:
            self.terminate()
        except ImportError:
            pass
