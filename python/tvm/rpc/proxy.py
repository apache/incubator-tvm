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
# ruff: noqa: F841
"""RPC proxy, allows both client/server to connect and match connection.

In normal RPC, client directly connect to server's IP address.
Sometimes this cannot be done when server do not have a static address.
RPCProxy allows both client and server connect to the proxy server,
the proxy server will forward the message between the client and server.
"""

# pylint: disable=unused-variable, unused-argument
import asyncio
import errno
import logging
import os
import socket
import struct
import threading
import time

try:
    import tornado
    from tornado import gen, ioloop, websocket

    from . import tornado_util
except ImportError as error_msg:
    raise ImportError(
        f"RPCProxy module requires tornado package {error_msg}. Try 'pip install tornado'."
    )

from tvm_tirx.support.popen_pool import PopenWorker

from . import _ffi_api, base
from .base import TrackerCode
from .server import _server_env


class ForwardHandler:
    """Forward handler to forward the message."""

    def _init_handler(self):
        """Initialize handler."""
        self._init_message = b""
        self._init_req_nbytes = 4
        self._magic = None
        self.timeout = None
        self._rpc_key_length = None
        self._done = False
        self._proxy = ProxyServerHandler.current
        assert self._proxy
        self.rpc_key = None
        self.match_key = None
        self.forward_proxy = None
        self.alloc_time = None

    def __del__(self):
        logging.info("Delete %s...", self.name())

    def name(self):
        """Name of this connection."""
        return "RPCConnection"

    def _init_step(self, message):
        if self._magic is None:
            assert len(message) == 4
            self._magic = struct.unpack("<i", message)[0]
            if self._magic != base.RPC_MAGIC:
                logging.info("Invalid RPC magic from %s", self.name())
                self.close()
            self._init_req_nbytes = 4
        elif self._rpc_key_length is None:
            assert len(message) == 4
            self._rpc_key_length = struct.unpack("<i", message)[0]
            self._init_req_nbytes = self._rpc_key_length
        elif self.rpc_key is None:
            assert len(message) == self._rpc_key_length
            self.rpc_key = message.decode("utf-8")
            # match key is used to do the matching
            self.match_key = self.rpc_key[7:].split()[0]
            self.on_start()
        else:
            assert False

    def on_start(self):
        """Event when the initialization is completed"""
        self._proxy.handler_ready(self)

    def on_data(self, message):
        """on data"""
        assert isinstance(message, bytes)
        if self.forward_proxy:
            self.forward_proxy.send_data(message)
        else:
            while message and self._init_req_nbytes > len(self._init_message):
                nbytes = self._init_req_nbytes - len(self._init_message)
                self._init_message += message[:nbytes]
                message = message[nbytes:]
                if self._init_req_nbytes == len(self._init_message):
                    temp = self._init_message
                    self._init_req_nbytes = 0
                    self._init_message = b""
                    self._init_step(temp)
            if message:
                logging.info("Invalid RPC protocol, too many bytes %s", self.name())
                self.close()

    def on_error(self, err):
        logging.info("%s: Error in RPC %s", self.name(), err)
        self.close_pair()

    def close_pair(self):
        if self.forward_proxy:
            self.forward_proxy.signal_close()
            self.forward_proxy = None
        self.close()

    def on_close_event(self):
        """on close event"""
        assert not self._done
        logging.info("RPCProxy:on_close_event %s ...", self.name())
        if self.match_key:
            key = self.match_key
            if self._proxy._client_pool.get(key, None) == self:
                self._proxy._client_pool.pop(key)
            if self._proxy._server_pool.get(key, None) == self:
                self._proxy._server_pool.pop(key)
        self._done = True
        self.forward_proxy = None


class TCPHandler(tornado_util.TCPHandler, ForwardHandler):
    """Event driven TCP handler."""

    def __init__(self, sock, addr):
        super().__init__(sock)
        self._init_handler()
        self.addr = addr

    def name(self):
        return f"TCPSocketProxy:{self.addr[0]!s}:{self.rpc_key}"

    def send_data(self, message, binary=True):
        self.write_message(message, True)

    def on_message(self, message):
        self.on_data(message)

    def on_close(self):
        logging.info("RPCProxy: on_close %s ...", self.name())
        self._close_process = True

        if self.forward_proxy:
            self.forward_proxy.signal_close()
            self.forward_proxy = None
        self.on_close_event()


class WebSocketHandler(websocket.WebSocketHandler, ForwardHandler):
    """Handler for websockets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_handler()

    def name(self):
        return f"WebSocketProxy:{self.rpc_key}"

    def on_message(self, message):
        self.on_data(message)

    def data_received(self, _):
        raise NotImplementedError()

    def send_data(self, message):
        try:
            self.write_message(message, True)
        except websocket.WebSocketClosedError as err:
            self.on_error(err)

    def on_close(self):
        logging.info("RPCProxy: on_close %s ...", self.name())
        if self.forward_proxy:
            self.forward_proxy.signal_close()
            self.forward_proxy = None
        self.on_close_event()

    def signal_close(self):
        self.close()


MIME_MAP = {
    "js": "text/javascript",
    "css": "text/css",
    "wasm": "application/wasm",
    "json": "application/json",
}


class RequestHandler(tornado.web.RequestHandler):
    """Handles html request."""

    def __init__(self, *args, **kwargs):
        file_path = kwargs.pop("file_path")
        self.format = file_path.split(".")[-1]

        if file_path.endswith("html"):
            self.page = open(file_path).read()
            web_port = kwargs.pop("rpc_web_port", None)
            if web_port:
                self.page = self.page.replace(
                    "ws://localhost:9190/ws", f"ws://localhost:{web_port}/ws"
                )
        else:
            self.page = open(file_path, "rb").read()

        super().__init__(*args, **kwargs)

    def data_received(self, _):
        pass

    def get(self, *args, **kwargs):
        if self.format in MIME_MAP:
            self.set_header("Content-Type", MIME_MAP[self.format])
        self.write(self.page)


class ProxyServerHandler:
    """Internal proxy server handler class."""

    current = None

    def __init__(
        self,
        sock,
        listen_port,
        web_port,
        timeout_client,
        timeout_server,
        tracker_addr,
        index_page=None,
        resource_files=None,
    ):
        assert ProxyServerHandler.current is None
        ProxyServerHandler.current = self
        if web_port:
            handlers = [(r"/ws", WebSocketHandler)]
            if index_page:
                handlers.append(
                    (r"/", RequestHandler, {"file_path": index_page, "rpc_web_port": web_port})
                )
                logging.info("Serving RPC index html page at http://localhost:%d", web_port)
            resource_files = resource_files if resource_files else []
            for item in resource_files:
                prefix, fname = item
                if not prefix.endswith("/"):
                    prefix += "/"
                if not prefix.startswith("/"):
                    prefix = "/" + prefix
                basename = os.path.basename(fname)
                pair = (rf"{prefix}{basename}", RequestHandler, {"file_path": fname})
                handlers.append(pair)
                logging.info(pair)
            self.app = tornado.web.Application(handlers)
            self.app.listen(web_port)

        self.sock = sock
        self.sock.setblocking(0)
        self.loop = ioloop.IOLoop.current()

        def event_handler(_, events):
            self._on_event(events)

        self.loop.add_handler(self.sock.fileno(), event_handler, self.loop.READ)
        self._client_pool = {}
        self._server_pool = {}
        self.timeout_alloc = 5
        self.timeout_client = timeout_client
        self.timeout_server = timeout_server
        # tracker information
        self._listen_port = listen_port
        self._tracker_addr = tracker_addr
        self._tracker_conn = None
        self._tracker_pending_puts = []
        self._key_set = set()
        self.update_tracker_period = 2
        if tracker_addr:
            logging.info("Tracker address:%s", str(tracker_addr))

            def _callback():
                self._update_tracker(True)

            self.loop.call_later(self.update_tracker_period, _callback)
        logging.info("RPCProxy: Websock port bind to %d", web_port)

    def _on_event(self, _):
        while True:
            try:
                conn, addr = self.sock.accept()
                TCPHandler(conn, addr)
            except OSError as err:
                if err.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break

    def _pair_up(self, lhs, rhs):
        lhs.forward_proxy = rhs
        rhs.forward_proxy = lhs

        lhs.send_data(struct.pack("<i", base.RPC_CODE_SUCCESS))
        lhs.send_data(struct.pack("<i", len(rhs.rpc_key)))
        lhs.send_data(rhs.rpc_key.encode("utf-8"))

        rhs.send_data(struct.pack("<i", base.RPC_CODE_SUCCESS))
        rhs.send_data(struct.pack("<i", len(lhs.rpc_key)))
        rhs.send_data(lhs.rpc_key.encode("utf-8"))
        logging.info("Pairup connect %s  and %s", lhs.name(), rhs.name())

    def _regenerate_server_keys(self, keys):
        """Regenerate keys for server pool"""
        keyset = set(self._server_pool.keys())
        new_keys = []
        # re-generate the server match key, so old information is invalidated.
        for key in tuple(keys):
            rpc_key, _ = base.split_random_key(key)
            handle = self._server_pool[key]
            del self._server_pool[key]
            new_key = base.random_key(rpc_key, keyset)
            self._server_pool[new_key] = handle
            keyset.add(new_key)
            new_keys.append(new_key)
        return new_keys

    def _update_tracker(self, period_update=False):
        """Update information on tracker."""
        try:
            if self._tracker_conn is None:
                self._tracker_conn = socket.socket(
                    base.get_addr_family(self._tracker_addr), socket.SOCK_STREAM
                )
                self._tracker_conn.connect(self._tracker_addr)
                self._tracker_conn.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
                magic = struct.unpack("<i", base.recvall(self._tracker_conn, 4))[0]
                if magic != base.RPC_TRACKER_MAGIC:
                    self.loop.stop()
                    raise RuntimeError(f"{self._tracker_addr} is not RPC Tracker")
                # just connect to tracker, need to update all keys
                self._tracker_pending_puts = self._server_pool.keys()

            if self._tracker_conn and period_update:
                # periodically update tracker information
                # regenerate key if the key is not in tracker anymore
                # and there is no in-coming connection after timeout_alloc
                base.sendjson(self._tracker_conn, [TrackerCode.GET_PENDING_MATCHKEYS])
                pending_keys = set(base.recvjson(self._tracker_conn))
                update_keys = []
                for k, v in self._server_pool.items():
                    if k not in pending_keys:
                        if v.alloc_time is None:
                            v.alloc_time = time.time()
                        elif time.time() - v.alloc_time > self.timeout_alloc:
                            update_keys.append(k)
                            v.alloc_time = None
                if update_keys:
                    logging.info(
                        "RPCProxy: No incoming conn on %s, regenerate keys...", str(update_keys)
                    )
                    new_keys = self._regenerate_server_keys(update_keys)
                    self._tracker_pending_puts += new_keys

            need_update_info = False
            # report new connections
            for key in self._tracker_pending_puts:
                rpc_key, _ = base.split_random_key(key)
                base.sendjson(
                    self._tracker_conn, [TrackerCode.PUT, rpc_key, (self._listen_port, key), None]
                )
                assert base.recvjson(self._tracker_conn) == TrackerCode.SUCCESS
                if rpc_key not in self._key_set:
                    self._key_set.add(rpc_key)
                    need_update_info = True

            if need_update_info:
                keylist = "[" + ",".join(self._key_set) + "]"
                cinfo = {"key": "server:proxy" + keylist, "addr": [None, self._listen_port]}
                base.sendjson(self._tracker_conn, [TrackerCode.UPDATE_INFO, cinfo])
                assert base.recvjson(self._tracker_conn) == TrackerCode.SUCCESS
            self._tracker_pending_puts = []
        except OSError as err:
            logging.info(
                "Lost tracker connection: %s, try reconnect in %g sec",
                str(err),
                self.update_tracker_period,
            )
            self._tracker_conn.close()
            self._tracker_conn = None
            self._regenerate_server_keys(self._server_pool.keys())

        if period_update:

            def _callback():
                self._update_tracker(True)

            self.loop.call_later(self.update_tracker_period, _callback)

    def _handler_ready_tracker_mode(self, handler):
        """tracker mode to handle handler ready."""
        if handler.rpc_key.startswith("server:"):
            key = base.random_key(handler.match_key, cmap=self._server_pool)
            handler.match_key = key
            self._server_pool[key] = handler
            self._tracker_pending_puts.append(key)
            self._update_tracker()
        else:
            if handler.match_key in self._server_pool:
                self._pair_up(self._server_pool.pop(handler.match_key), handler)
            else:
                handler.send_data(struct.pack("<i", base.RPC_CODE_MISMATCH))
                handler.signal_close()

    def _handler_ready_proxy_mode(self, handler):
        """Normal proxy mode when handler is ready."""
        if handler.rpc_key.startswith("server:"):
            pool_src, pool_dst = self._client_pool, self._server_pool
            timeout = self.timeout_server
        else:
            pool_src, pool_dst = self._server_pool, self._client_pool
            timeout = self.timeout_client

        key = handler.match_key
        if key in pool_src:
            self._pair_up(pool_src.pop(key), handler)
            return
        if key not in pool_dst:
            pool_dst[key] = handler

            def cleanup():
                """Cleanup client connection if timeout"""
                if pool_dst.get(key, None) == handler:
                    logging.info(
                        "Timeout client connection %s, cannot find match key=%s",
                        handler.name(),
                        key,
                    )
                    pool_dst.pop(key)
                    handler.send_data(struct.pack("<i", base.RPC_CODE_MISMATCH))
                    handler.signal_close()

            self.loop.call_later(timeout, cleanup)
        else:
            logging.info("Duplicate connection with same key=%s", key)
            handler.send_data(struct.pack("<i", base.RPC_CODE_DUPLICATE))
            handler.signal_close()

    def handler_ready(self, handler):
        """Report handler to be ready."""
        logging.info("Handler ready %s", handler.name())
        if self._tracker_addr:
            self._handler_ready_tracker_mode(handler)
        else:
            self._handler_ready_proxy_mode(handler)

    def run(self):
        """Run the proxy server"""
        ioloop.IOLoop.current().start()


def _proxy_server(
    listen_sock,
    listen_port,
    web_port,
    timeout_client,
    timeout_server,
    tracker_addr,
    index_page,
    resource_files,
):
    asyncio.set_event_loop(asyncio.new_event_loop())
    handler = ProxyServerHandler(
        listen_sock,
        listen_port,
        web_port,
        timeout_client,
        timeout_server,
        tracker_addr,
        index_page,
        resource_files,
    )
    handler.run()


class PopenProxyServerState:
    """Internal PopenProxy State for Popen"""

    current = None

    def __init__(
        self,
        host,
        port=9091,
        port_end=9199,
        web_port=0,
        timeout_client=600,
        timeout_server=600,
        tracker_addr=None,
        index_page=None,
        resource_files=None,
    ):
        sock = socket.socket(base.get_addr_family((host, port)), socket.SOCK_STREAM)
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
        logging.info("RPCProxy: client port bind to %s:%d", host, self.port)
        sock.listen(1)
        self.thread = threading.Thread(
            target=_proxy_server,
            args=(
                sock,
                self.port,
                web_port,
                timeout_client,
                timeout_server,
                tracker_addr,
                index_page,
                resource_files,
            ),
        )
        # start the server in a different thread
        # so we can return the port directly
        self.thread.start()


def _popen_start_proxy_server(
    host,
    port=9091,
    port_end=9199,
    web_port=0,
    timeout_client=600,
    timeout_server=600,
    tracker_addr=None,
    index_page=None,
    resource_files=None,
):
    # This is a function that will be sent to the
    # Popen worker to run on a separate process.
    # Create and start the server in a different thread
    state = PopenProxyServerState(
        host,
        port,
        port_end,
        web_port,
        timeout_client,
        timeout_server,
        tracker_addr,
        index_page,
        resource_files,
    )
    PopenProxyServerState.current = state
    # returns the port so that the main can get the port number.
    return state.port


class Proxy:
    """Start RPC proxy server on a separate process.

    Python implementation based on PopenWorker.

    Parameters
    ----------
    host : str
        The host url of the server.

    port : int
        The TCP port to be bind to

    port_end : int, optional
        The end TCP port to search

    web_port : int, optional
        The http/websocket port of the server.

    timeout_client : float, optional
        Timeout of client until it sees a matching connection.

    timeout_server : float, optional
        Timeout of server until it sees a matching connection.

    tracker_addr: Tuple (str, int) , optional
        The address of RPC Tracker in tuple (host, ip) format.
        If is not None, the server will register itself to the tracker.

    index_page : str, optional
        Path to an index page that can be used to display at proxy index.

    resource_files : str, optional
        Path to local resources that can be included in the http request
    """

    def __init__(
        self,
        host,
        port=9091,
        port_end=9199,
        web_port=0,
        timeout_client=600,
        timeout_server=600,
        tracker_addr=None,
        index_page=None,
        resource_files=None,
    ):
        self.proc = PopenWorker()
        # send the function
        self.proc.send(
            _popen_start_proxy_server,
            [
                host,
                port,
                port_end,
                web_port,
                timeout_client,
                timeout_server,
                tracker_addr,
                index_page,
                resource_files,
            ],
        )
        # receive the port
        self.port = self.proc.recv()
        self.host = host

    def terminate(self):
        """Terminate the server process"""
        if self.proc:
            logging.info("Terminating Proxy Server...")
            self.proc.kill()
            self.proc = None

    def __del__(self):
        try:
            self.terminate()
        except ImportError:
            pass


def websocket_proxy_server(url, key=""):
    """Create a RPC server that uses an websocket that connects to a proxy.

    Parameters
    ----------
    url : str
        The url to be connected.

    key : str
        The key to identify the server.
    """

    def create_on_message(conn):
        def _fsend(data):
            data = bytes(data)
            conn.write_message(data, binary=True)
            return len(data)

        on_message = _ffi_api.CreateEventDrivenServer(_fsend, "WebSocketProxyServer", "%toinit")
        return on_message

    @gen.coroutine
    def _connect(key):
        conn = yield websocket.websocket_connect(url)
        on_message = create_on_message(conn)
        temp = _server_env(None)
        # Start connecton
        conn.write_message(struct.pack("<i", base.RPC_MAGIC), binary=True)
        key = "server:" + key
        conn.write_message(struct.pack("<i", len(key)), binary=True)
        conn.write_message(key.encode("utf-8"), binary=True)
        msg = yield conn.read_message()
        assert len(msg) >= 4
        magic = struct.unpack("<i", msg[:4])[0]
        if magic == base.RPC_CODE_DUPLICATE:
            raise RuntimeError(f"key: {key} has already been used in proxy")
        if magic == base.RPC_CODE_MISMATCH:
            logging.info("RPCProxy do not have matching client key %s", key)
        elif magic != base.RPC_CODE_SUCCESS:
            raise RuntimeError(f"{url} is not RPC Proxy")
        msg = msg[4:]

        logging.info("Connection established with remote")

        if msg:
            on_message(bytearray(msg), 3)

        while True:
            try:
                msg = yield conn.read_message()
                if msg is None:
                    break
                on_message(bytearray(msg), 3)
            except websocket.WebSocketClosedError as err:
                break
        logging.info("WebSocketProxyServer closed...")
        temp.remove()
        ioloop.IOLoop.current().stop()

    ioloop.IOLoop.current().spawn_callback(_connect, key)
    ioloop.IOLoop.current().start()
