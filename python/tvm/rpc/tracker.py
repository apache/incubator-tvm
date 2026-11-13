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
"""RPC Tracker, tracks and distributes the TVM RPC resources.

This folder implements the tracker server logic.

Note
----
Tracker is a TCP based rest api with the following protocol:
- Initial handshake to the peer
  - RPC_TRACKER_MAGIC
- Normal message: [size(int32), json-data]
- Each message is initiated by the client, and the tracker replies with a json.

List of available APIs:

- PING: check if tracker is alive
  - input: [TrackerCode.PING]
  - return: TrackerCode.SUCCESS
- PUT: report resource to tracker
  - input: [TrackerCode.PUT, [port, match-key]]
  - return: TrackerCode.SUCCESS
  - note: match-key is a randomly generated identify the resource during connection.
- REQUEST: request a new resource from tracker
  - input: [TrackerCode.REQUEST, [key, user, priority]]
  - return: [TrackerCode.SUCCESS, [url, port, match-key]]
"""
# pylint: disable=invalid-name

import asyncio
import errno
import heapq
import json
import logging
import socket
import struct
import sys
import threading

from tvm_tirx.support.popen_pool import PopenWorker

try:
    from tornado import ioloop

    from . import tornado_util
except ImportError as error_msg:
    raise ImportError(
        f"RPCTracker module requires tornado package {error_msg}. Try 'pip install tornado'."
    )

from . import base
from .base import RPC_TRACKER_MAGIC, TrackerCode

logger = logging.getLogger("RPCTracker")
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Maximum size in bytes for a single tracker message. Tracker frames carry
# small JSON command tuples; 1 MiB is well above any legitimate payload and
# bounds memory growth when a peer sends an oversized or malformed size
# header on the wire.
MAX_TRACKER_MSG_BYTES = 1 << 20


class Scheduler:
    """Abstract interface of scheduler."""

    def put(self, value):
        """Push a resource into the scheduler.

        This function can trigger callbacks in the scheduler.

        Parameters
        ----------
        value : object
            The resource to be put in the scheduler.
        """
        raise NotImplementedError()

    def request(self, user, priority, callback):
        """Request a resource.

        Parameters
        ----------
        user : str
            The user who is requesting the resource.

        priority : int
            The job priority

        callback : function: value->bool
            Callback function to receive an resource when ready
            returns True if the resource is consumed.
        """
        raise NotImplementedError()

    def remove(self, value):
        """Remove a resource in the scheduler

        Parameters
        ----------
        value: object
            The resource to remove
        """

    def summary(self):
        """Get summary information of the scheduler."""
        raise NotImplementedError()


class PriorityScheduler(Scheduler):
    """Priority based scheduler, FIFO based on request order"""

    def __init__(self, key):
        self._key = key
        self._request_cnt = 0
        self._lock = threading.Lock()
        self._values = []
        self._requests = []

    def _schedule(self):
        while self._requests and self._values:
            value = self._values.pop(0)
            item = heapq.heappop(self._requests)
            callback = item[-1]
            if callback(value[1:]):
                value[0].pending_matchkeys.remove(value[-1])
            else:
                self._values.append(value)

    def put(self, value):
        self._values.append(value)
        self._schedule()

    def request(self, user, priority, callback):
        with self._lock:
            heapq.heappush(self._requests, (-priority, self._request_cnt, callback))
            self._request_cnt += 1
        self._schedule()

    def remove(self, value):
        if value in self._values:
            self._values.remove(value)
            self._schedule()

    def summary(self):
        """Get summary information of the scheduler."""
        return {"free": len(self._values), "pending": len(self._requests)}


class TCPEventHandler(tornado_util.TCPHandler):
    """Base asynchronize message handler.

    The tracker and client follows a simple message protocol.
    The message is in form [nbytes(int32)] [json-str].
    All the information is packed in json-str
    """

    def __init__(self, tracker, sock, addr):
        super().__init__(sock)
        self._data = bytearray()
        self._tracker = tracker
        self._msg_size = 0
        self._addr = addr
        self._init_req_nbytes = 4
        self._info = {}
        # list of pending match keys that has not been used.
        self.pending_matchkeys = set()
        self._tracker._connections.add(self)
        self.put_values = []

    def name(self):
        """name of connection"""
        return f"TCPSocket: {self._addr!s}"

    def summary(self):
        """Summary of this connection"""
        return self._info

    def _init_conn(self, message):
        """Initialize the connection"""
        if len(message) != 4:
            logger.warning("Invalid connection from %s", self.name())
            self.close()
        magic = struct.unpack("<i", message)[0]
        if magic != RPC_TRACKER_MAGIC:
            logger.warning("Invalid magic from %s", self.name())
            self.close()
        self.write_message(struct.pack("<i", RPC_TRACKER_MAGIC), binary=True)
        self._init_req_nbytes = 0

    def on_message(self, message):
        """Callback when a message is received.

        Parameters
        ----------
        message : bytearray
            The bytes received
        """
        assert isinstance(message, bytes)
        if self._init_req_nbytes:
            self._init_conn(message)
            return

        self._data += message

        while True:
            if self._msg_size == 0:
                if len(self._data) >= 4:
                    self._msg_size = struct.unpack("<i", self._data[:4])[0]
                    if self._msg_size <= 0 or self._msg_size > MAX_TRACKER_MSG_BYTES:
                        logger.warning(
                            "Invalid msg_size %d from %s; closing connection",
                            self._msg_size,
                            self.name(),
                        )
                        self.close()
                        return
                    del self._data[:4]
                else:
                    return
            if self._msg_size != 0 and len(self._data) >= self._msg_size:
                msg = (bytes(self._data[: self._msg_size])).decode("utf-8")
                del self._data[: self._msg_size]
                self._msg_size = 0
                try:
                    self.call_handler(json.loads(msg))
                except Exception:  # pylint: disable=broad-except
                    logger.warning("Error handling message from %s", self.name(), exc_info=True)
                    self.close()
                    return
            else:
                return

    def ret_value(self, data):
        """return value to the output"""
        data = json.dumps(data)
        self.write_message(struct.pack("<i", len(data)), binary=True)
        self.write_message(data.encode("utf-8"), binary=True)

    def call_handler(self, args):
        """Event handler when json request arrives."""
        code = args[0]
        if code == TrackerCode.PUT:
            key = args[1]
            port, matchkey = args[2]
            self.pending_matchkeys.add(matchkey)
            # got custom address (from rpc server)
            if len(args) >= 4 and args[3] is not None:
                value = (self, args[3], port, matchkey)
            else:
                value = (self, self._addr[0], port, matchkey)
            self._tracker.put(key, value)
            self.put_values.append(value)
            self.ret_value(TrackerCode.SUCCESS)
        elif code == TrackerCode.REQUEST:
            key = args[1]
            user = args[2]
            priority = args[3]

            def _cb(value):
                # if the connection is already closed
                if not self._sock:
                    return False
                try:
                    self.ret_value([TrackerCode.SUCCESS, value])
                except OSError:
                    return False
                return True

            self._tracker.request(key, user, priority, _cb)
        elif code == TrackerCode.PING:
            self.ret_value(TrackerCode.SUCCESS)
        elif code == TrackerCode.GET_PENDING_MATCHKEYS:
            self.ret_value(list(self.pending_matchkeys))
        elif code == TrackerCode.STOP:
            # safe stop tracker
            if self._tracker._stop_key == args[1]:
                self.ret_value(TrackerCode.SUCCESS)
                self._tracker.stop()
            else:
                self.ret_value(TrackerCode.FAIL)
        elif code == TrackerCode.UPDATE_INFO:
            info = args[1]
            assert isinstance(info, dict)
            if info["addr"][0] is None:
                info["addr"][0] = self._addr[0]
            self._info.update(info)
            self.ret_value(TrackerCode.SUCCESS)
        elif code == TrackerCode.SUMMARY:
            status = self._tracker.summary()
            self.ret_value([TrackerCode.SUCCESS, status])
        else:
            logger.warning("Unknown tracker code %d", code)
            self.close()

    def on_close(self):
        self._tracker.close(self)

    def on_error(self, err):
        logger.warning("%s: Error in RPC Tracker: %s", self.name(), err)
        self.close()


class TrackerServerHandler:
    """Tracker that tracks the resources."""

    def __init__(self, sock, stop_key):
        self._scheduler_map = {}
        self._sock = sock
        self._sock.setblocking(0)
        self._ioloop = ioloop.IOLoop.current()
        self._stop_key = stop_key
        self._connections = set()

        def _event_handler(_, events):
            self._on_event(events)

        self._ioloop.add_handler(self._sock.fileno(), _event_handler, self._ioloop.READ)

    def _on_event(self, _):
        while True:
            try:
                conn, addr = self._sock.accept()
                TCPEventHandler(self, conn, addr)
            except OSError as err:
                if err.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break

    def create_scheduler(self, key):
        """Create a new scheduler."""
        return PriorityScheduler(key)

    def put(self, key, value):
        """Report a new resource to the tracker."""
        if key not in self._scheduler_map:
            self._scheduler_map[key] = self.create_scheduler(key)
        self._scheduler_map[key].put(value)

    def request(self, key, user, priority, callback):
        """Request a new resource."""
        if key not in self._scheduler_map:
            self._scheduler_map[key] = self.create_scheduler(key)
        self._scheduler_map[key].request(user, priority, callback)

    def close(self, conn):
        self._connections.remove(conn)
        if "key" in conn._info:
            for value in conn.put_values:
                _, _, _, key = value
                rpc_key, _ = base.split_random_key(key)
                self._scheduler_map[rpc_key].remove(value)

    def stop(self):
        """Safely stop tracker."""
        for conn in list(self._connections):
            conn.close()
        self._sock.close()
        self._ioloop.stop()

    def summary(self):
        """Return a dict summarizing current status."""
        qinfo = {}
        for k, v in self._scheduler_map.items():
            qinfo[k] = v.summary()
        cinfo = []
        # ignore client connections without key
        for conn in self._connections:
            res = conn.summary()
            if res.get("key", "").startswith("server"):
                cinfo.append(res)
        return {"queue_info": qinfo, "server_info": cinfo}

    def run(self):
        """Run the tracker server"""
        self._ioloop.start()


def _tracker_server(listen_sock, stop_key):
    asyncio.set_event_loop(asyncio.new_event_loop())
    handler = TrackerServerHandler(listen_sock, stop_key)
    handler.run()


class PopenTrackerServerState:
    """Internal PopenTrackerServer State"""

    current = None

    def __init__(self, host, port=9190, port_end=9199, silent=False, reuse_addr=True, timeout=None):
        if silent:
            logger.setLevel(logging.WARN)

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
        self.stop_key = base.random_key("tracker")
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
        self.thread = threading.Thread(target=_tracker_server, args=(sock, self.stop_key))
        self.thread.start()
        self.host = host


def _popen_start_tracker_server(
    host, port=9190, port_end=9199, silent=False, reuse_addr=True, timeout=None
):
    # This is a function that will be sent to the
    # Popen worker to run on a separate process.
    # Create and start the server in a different thread
    state = PopenTrackerServerState(host, port, port_end, silent, reuse_addr, timeout)
    PopenTrackerServerState.current = state
    # returns the port so that the main can get the port number.
    return (state.port, state.stop_key)


class Tracker:
    """Start RPC tracker on a separate process.

    Python implementation based on PopenWorker.

    Parameters
    ----------
    host : str
        The host url of the server.

    port : int
        The TCP port to be bind to

    port_end : int, optional
        The end TCP port to search

    silent: bool, optional
        Whether run in silent mode

    reuse_addr: bool, optional
        Allows the kernel to reuse a local socket in TIME_WAIT state.

    timeout: float, optional
         set a timeout for all operations on the socket

    """

    def __init__(
        self, host="0.0.0.0", port=9190, port_end=9199, silent=False, reuse_addr=True, timeout=None
    ):
        if silent:
            logger.setLevel(logging.WARN)
        self.proc = PopenWorker()
        # send the function
        self.proc.send(
            _popen_start_tracker_server, [host, port, port_end, silent, reuse_addr, timeout]
        )
        # receive the port
        self.port, self.stop_key = self.proc.recv()
        self.host = host

    def _stop_tracker(self):
        sock = socket.socket(base.get_addr_family((self.host, self.port)), socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", self.port))
        sock.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
        magic = struct.unpack("<i", base.recvall(sock, 4))[0]
        assert magic == base.RPC_TRACKER_MAGIC
        base.sendjson(sock, [TrackerCode.STOP, self.stop_key])
        assert base.recvjson(sock) == TrackerCode.SUCCESS
        sock.close()

    def terminate(self):
        """Terminate the server process"""
        if self.proc:
            if self.proc.is_alive():
                self._stop_tracker()
            self.proc.join(0.1)
            if self.proc.is_alive():
                logger.info("Terminating Tracker Server...")
                self.proc.kill()
            self.proc = None

    def __del__(self):
        try:
            self.terminate()
        except TypeError:
            pass
