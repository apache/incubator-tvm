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
# pylint: disable=used-before-assignment,broad-exception-caught
# ruff: noqa: F401
"""RPC client tools"""

import os
import socket
import stat
import struct
import time

import tvm_ffi
from tvm_ffi import DLDeviceType

import tvm
import tvm_tirx.runtime
from tvm_tirx.support import utils

from . import _ffi_api, base, server


class RPCSession:
    """RPC Client session module

    Do not directly create the object, call connect
    """

    # pylint: disable=invalid-name
    def __init__(self, sess):
        self._sess = sess
        self._remote_funcs = {}

    def system_lib(self):
        """Get system-wide library module.

        Returns
        -------
        module : tvm_ffi.Module
            The system-wide library module.

        See Also
        --------
        tvm_ffi.system_lib
        """
        return self.get_function("ffi.SystemLib")()

    def get_function(self, name):
        """Get function from the session.

        Parameters
        ----------
        name : str
            The name of the function

        Returns
        -------
        f : Function
            The result function.
        """
        return self._sess.get_function(name)

    def device(self, dev_type, dev_id=0):
        """Construct a remote device.

        Parameters
        ----------
        dev_type: int or str

        dev_id: int, optional

        Returns
        -------
        dev: RemoteDevice
            The session-owned device.
        """
        return _ffi_api.RemoteDevice(self._sess, tvm_ffi.device(dev_type, dev_id))

    def close(self):
        """Close this session; existing remote values reject further operations."""
        self._sess.get_function("CloseRPCConnection")()

    def upload(self, data, target=None):
        """Upload file to remote runtime temp folder

        Parameters
        ----------
        data : str or bytearray
            The file name or binary in local to upload.

        target : str, optional
            The path in remote
        """
        if isinstance(data, bytearray):
            if not target:
                raise ValueError("target must present when file is a bytearray")
            blob = data
        else:
            blob = bytearray(open(data, "rb").read())
            if not target:
                target = os.path.basename(data)

        if "upload" not in self._remote_funcs:
            self._remote_funcs["upload"] = self.get_function("tvm.rpc.server.upload")
        self._remote_funcs["upload"](target, blob)

    def download(self, path):
        """Download file from remote temp folder.

        Parameters
        ----------
        path : str
            The relative location to remote temp folder.

        Returns
        -------
        blob : bytearray
            The result blob from the file.
        """
        if "download" not in self._remote_funcs:
            self._remote_funcs["download"] = self.get_function("tvm.rpc.server.download")
        return self._remote_funcs["download"](path)

    def remove(self, path):
        """Remove file from remote temp folder.

        Parameters
        ----------
        path: str
            The relative location to remote temp folder.
        """
        if "remove" not in self._remote_funcs:
            self._remote_funcs["remove"] = self.get_function("tvm.rpc.server.remove")
        self._remote_funcs["remove"](path)

    def listdir(self, path):
        """ls files from remote temp folder.

        Parameters
        ----------
        path: str
            The relative location to remote temp folder.

        Returns
        -------
        dirs: str
            The files in the given directory with split token ','.
        """
        if "listdir" not in self._remote_funcs:
            self._remote_funcs["listdir"] = self.get_function("tvm.rpc.server.listdir")
        return self._remote_funcs["listdir"](path)

    def load_module(self, path):
        """Load a remote module, the file need to be uploaded first.

        Parameters
        ----------
        path : str
            The relative location to remote temp folder.

        Returns
        -------
        m : Module
            The remote module containing remote function.
        """
        return _ffi_api.LoadRemoteModule(self._sess, path)

    def download_linked_module(self, path):
        """Link a module in the remote and download it.

        Parameters
        ----------
        path : str
            The relative location to remote temp folder.

        Returns
        -------
        blob : bytearray
            The result blob from the file.

        Note
        ----
        This function can be helpful when a linker
        is not available on the local client.

        Examples
        --------
        .. code-block:: python

            from tvm_tirx.codegen import Executable

            mod = build_module_with_cross_compilation()
            # export the module as tar because a local linker is not available
            Executable(mod).export_library("lib.tar")
            remote.upload("lib.tar")
            # invoke the linker on the remote to link the module as a library
            # note that the library can only run on the same env as the remote
            with open("lib.so", "wb") as file:
                file.write(remote.download_linked_module("lib.tar"))
        """
        if "download_linked_module" not in self._remote_funcs:
            self._remote_funcs["download_linked_module"] = self.get_function(
                "tvm.rpc.server.download_linked_module"
            )
        return self._remote_funcs["download_linked_module"](path)

    def cpu(self, dev_id=0):
        """Construct CPU device."""
        return self.device(DLDeviceType.kDLCPU, dev_id)

    def cuda(self, dev_id=0):
        """Construct CUDA GPU device."""
        return self.device(DLDeviceType.kDLCUDA, dev_id)

    def cl(self, dev_id=0):
        """Construct OpenCL device."""
        return self.device(DLDeviceType.kDLOpenCL, dev_id)

    def vulkan(self, dev_id=0):
        """Construct Vulkan device."""
        return self.device(DLDeviceType.kDLVulkan, dev_id)

    def metal(self, dev_id=0):
        """Construct Metal device."""
        return self.device(DLDeviceType.kDLMetal, dev_id)

    def rocm(self, dev_id=0):
        """Construct ROCm device."""
        return self.device(DLDeviceType.kDLROCM, dev_id)

    def ext_dev(self, dev_id=0):
        """Construct extension device."""
        return self.device(DLDeviceType.kDLExtDev, dev_id)

    def hexagon(self, dev_id=0):
        """Construct Hexagon device."""
        return self.device(DLDeviceType.kDLHexagon, dev_id)

    def webgpu(self, dev_id=0):
        """Construct WebGPU device."""
        return self.device(DLDeviceType.kDLWebGPU, dev_id)


class LocalSession(RPCSession):
    """RPCSession interface backed by local environment.

    This class can be used to implement functions that
    need to be ran both locally and remotely.
    """

    def __init__(self):
        self._temp = server._server_env([])
        RPCSession.__init__(self, _ffi_api.LocalSession())


@tvm_ffi.register_global_func("rpc.PopenSession")
def _popen_session(binary):
    temp = utils.tempdir()

    if isinstance(binary, bytes | bytearray):
        path_exec = temp.relpath("server.minrpc")
        with open(path_exec, "wb") as outfile:
            outfile.write(binary)
        os.chmod(path_exec, stat.S_IXUSR | stat.S_IRUSR)
        path_exec = os.path.abspath(path_exec)
    else:
        path_exec = os.path.abspath(binary)
        if not os.path.isfile(path_exec):
            raise RuntimeError(f"{path_exec} does not exist.")
        if not os.access(path_exec, os.X_OK):
            raise RuntimeError(f"{path_exec} is not executable.")

    sess = _ffi_api.CreatePipeClient(path_exec)
    return sess


class PopenSession(RPCSession):
    """RPCSession interface backed by popen.

    Parameters
    ----------
    binary : List[Union[str, bytes]]
        The binary to be executed.
    """

    def __init__(self, binary):
        RPCSession.__init__(self, _popen_session(binary))


class TrackerSession:
    """Tracker client session.

    Parameters
    ----------
    addr : tuple
        The address tuple
    """

    def __init__(self, addr):
        self._addr = addr
        self._sock = None
        self._connect()

    def __del__(self):
        self.close()

    def _connect(self):
        self._sock = base.connect_with_retry(self._addr)
        self._sock.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
        magic = struct.unpack("<i", base.recvall(self._sock, 4))[0]
        if magic != base.RPC_TRACKER_MAGIC:
            raise RuntimeError(f"{self._addr!s} is not RPC Tracker")

    def close(self):
        """Close the tracker connection."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def summary(self):
        """Get the summary dict of the tracker."""
        base.sendjson(self._sock, [base.TrackerCode.SUMMARY])
        value = base.recvjson(self._sock)
        if value[0] != base.TrackerCode.SUCCESS:
            raise RuntimeError(f"Invalid return value {value!s}")
        return value[1]

    def text_summary(self):
        """Get a text summary of the tracker."""
        data = self.summary()

        total_ct = {}

        res = ""
        res += "Server List\n"
        res += "------------------------------\n"
        res += "server-address           key\n"
        res += "------------------------------\n"
        sorted_server = sorted(data["server_info"], key=lambda x: x["key"])
        for item in sorted_server:
            addr = item["addr"]
            res += f"{':'.join(map(str, addr)):21s}    "
            res += item["key"] + "\n"
            key = item["key"].split(":")[1]  # 'server:rasp3b` -> 'rasp3b'
            if key not in total_ct:
                total_ct[key] = 0
            total_ct[key] += 1
        res += "------------------------------\n"
        res += "\n"

        # compute max length of device key
        queue_info = data["queue_info"]
        keys = list(queue_info.keys())
        if keys:
            keys.sort()
            max_key_len = max([len(k) for k in keys])
        else:
            max_key_len = 0

        res += "Queue Status\n"
        title = f"{'key':<{max_key_len}s}   total  free  pending\n"
        separate_line = "-" * len(title) + "\n"
        res += separate_line + title + separate_line
        for k in keys:
            total = total_ct.get(k, 0)
            free, pending = queue_info[k]["free"], queue_info[k]["pending"]
            if total or pending:
                res += f"{k:<{max_key_len}}   {total:<5d}  {free:<4d}  {pending:<7d}\n"
        res += separate_line
        return res

    def request(
        self,
        key,
        priority=1,
        session_timeout=0,
        max_retry=5,
        session_constructor_args=None,
    ):
        """Request a new connection from the tracker.

        Parameters
        ----------
        key : str
            The type key of the device.

        priority : int, optional
            The priority of the request.

        session_timeout : float, optional
            The duration of the session, allows server to kill
            the connection when duration is longer than this value.
            When duration is zero, it means the request must always be kept alive.

        max_retry : int, optional
            Maximum number of times to retry before give up.

        session_constructor_args : list, optional
            List of additional arguments to passed as the remote session constructor.
            The first element of the list is always a string specifying the name of
            the session constructor, the following args are the positional args to that function.
        """
        last_err = None
        for _ in range(max_retry):
            try:
                if self._sock is None:
                    self._connect()
                base.sendjson(self._sock, [base.TrackerCode.REQUEST, key, "", priority])
                value = base.recvjson(self._sock)
                if value[0] != base.TrackerCode.SUCCESS:
                    raise RuntimeError(f"Invalid return value {value!s}")
                url, port, matchkey = value[1]
                return connect(
                    url,
                    port,
                    matchkey,
                    session_timeout,
                    session_constructor_args=session_constructor_args,
                )
            except OSError as err:
                self.close()
                last_err = err
            except RuntimeError as err:
                last_err = err
        raise RuntimeError(f"Cannot request {key} after {max_retry} retry, last_error:{last_err!s}")

    def request_and_run(self, key, func, priority=1, session_timeout=0, max_retry=2):
        """Request a resource from tracker and run the func.

        This function safe-guard rare server node dropout during execution.
        In such case, a new resource will be requested and func will be ran again.

        Parameters
        ----------
        key : str
            The type key of the device.

        func : function of session -> value
            A stateless function

        priority : int, optional
            The priority of the request.

        session_timeout : float, optional
            The duration of the session, allows server to kill
            the connection when duration is longer than this value.
            When duration is zero, it means the request must always be kept alive.

        max_retry : int, optional
            Maximum number of times to retry the function before give up.
        """
        last_err = None
        for _ in range(max_retry):
            try:
                sess = self.request(key, priority=priority, session_timeout=session_timeout)
                tstart = time.time()
                return func(sess)
            except RuntimeError as err:
                duration = time.time() - tstart
                # roughly estimate if the error is due to timeout termination
                if session_timeout and duration >= session_timeout * 0.95:
                    raise RuntimeError(f"Session timeout when running {func.__name__}")
                last_err = err
        raise RuntimeError(
            f"Failed to run on {key} after {max_retry} retry, last_error:{last_err!s}"
        )


def connect(
    url,
    port,
    key="",
    session_timeout=0,
    session_constructor_args=None,
    enable_logging=False,
):
    """Connect to RPC Server

    Parameters
    ----------
    url : str
        The url of the host

    port : int
        The port to connect to

    key : str, optional
        Additional key to match server

    session_timeout : float, optional
        The duration of the session in seconds, allows server to kill
        the connection when duration is longer than this value.
        When duration is zero, it means the request must always be kept alive.

    session_constructor_args: List
        List of additional arguments to passed as the remote session constructor.
        The first element of the list is always a string specifying the name of
        the session constructor, the following args are the positional args to that function.

    enable_logging: boolean
        flag to enable/disable logging. Logging is disabled by default.

    Returns
    -------
    sess : RPCSession
        The connected session.

    Examples
    --------
    Normal usage
    .. code-block:: python

        client = rpc.connect(server_url, server_port, server_key)

    Session_constructor can be used to customize the session in the remote
    The following code connects to a remote internal server via a proxy
    by constructing another RPCClientSession on the proxy machine and use that
    as the serving session of the proxy endpoint.

    .. code-block:: python

        client_via_proxy = rpc.connect(
            proxy_server_url, proxy_server_port, proxy_server_key, enable_logging
            session_constructor_args=[
                "rpc.Connect", internal_url, internal_port, internal_key, internal_logging])

    """
    try:
        if session_timeout:
            key += f" -timeout={session_timeout}"
        session_constructor_args = session_constructor_args if session_constructor_args else []
        if not isinstance(session_constructor_args, list | tuple):
            raise TypeError("Expect the session constructor to be a list or tuple")
        sess = _ffi_api.Connect(url, port, key, enable_logging, *session_constructor_args)
    except NameError:
        raise RuntimeError("Please compile with TVM_USE_RPC=1")
    return RPCSession(sess)


def connect_tracker(url, port):
    """Connect to a RPC tracker

    Parameters
    ----------
    url : str
        The url of the host

    port : int
        The port to connect to

    Returns
    -------
    sess : TrackerSession
        The connected tracker session.
    """
    return TrackerSession((url, port))
