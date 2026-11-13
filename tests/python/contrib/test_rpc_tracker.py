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
"""Configure pytest"""

# pylint: disable=invalid-name
import logging
import time

import tvm_ffi

from tvm import rpc


def check_server_drop():
    """test when server drops"""
    try:
        # pylint: disable=import-outside-toplevel
        from tvm.rpc import base, proxy, tracker

        # pylint: disable=import-outside-toplevel
        from tvm.rpc.base import TrackerCode

        @tvm_ffi.register_global_func("rpc.test2.addone")
        def addone(x):
            return x + 1

        def _put(tclient, value):
            base.sendjson(tclient._sock, value)
            base.recvjson(tclient._sock)

        tserver = tracker.Tracker("127.0.0.1", 8888)
        tproxy = proxy.Proxy("127.0.0.1", 8881, tracker_addr=("127.0.0.1", tserver.port))
        tclient = rpc.connect_tracker("127.0.0.1", tserver.port)

        server0 = rpc.Server(
            "127.0.0.1", port=9099, tracker_addr=("127.0.0.1", tserver.port), key="abc"
        )
        server1 = rpc.Server(
            "127.0.0.1", port=9099, tracker_addr=("127.0.0.1", tserver.port), key="xyz"
        )
        server2 = rpc.Server("127.0.0.1", tproxy.port, is_proxy=True, key="xyz")
        server3 = rpc.Server("127.0.0.1", tproxy.port, is_proxy=True, key="xyz1")

        # Fault tolerence to un-handled requested value
        _put(tclient, [TrackerCode.REQUEST, "abc", "", 1])
        _put(tclient, [TrackerCode.REQUEST, "xyz1", "", 1])

        # Fault tolerence to stale worker value
        _put(tclient, [TrackerCode.PUT, "xyz", (server1.port, "abc")])
        _put(tclient, [TrackerCode.PUT, "xyz", (server1.port, "abcxxx")])
        _put(tclient, [TrackerCode.PUT, "xyz", (tproxy.port, "abcxxx11")])

        # Fault tolerence server timeout
        def check_timeout(timeout, sleeptime):
            def myfunc(remote):
                time.sleep(sleeptime)
                f1 = remote.get_function("rpc.test2.addone")
                assert f1(10) == 11

            try:
                tclient.request_and_run("xyz", myfunc, session_timeout=timeout)
            except RuntimeError:
                pass
            print(tclient.text_summary())
            try:
                remote = tclient.request("xyz", priority=0, session_timeout=timeout)
                remote2 = tclient.request("xyz", session_timeout=timeout)
                time.sleep(sleeptime)
                f1 = remote.get_function("rpc.test2.addone")
                assert f1(10) == 11
                f1 = remote2.get_function("rpc.test2.addone")
                assert f1(10) == 11

            except RuntimeError:
                pass
            remote3 = tclient.request("abc")
            f1 = remote3.get_function("rpc.test2.addone")
            assert f1(10) == 11
            remote3 = tclient.request("xyz1")
            f1 = remote3.get_function("rpc.test2.addone")
            assert f1(10) == 11

        check_timeout(0.01, 0.1)
        check_timeout(2, 0)
        tserver.terminate()
        server0.terminate()
        server1.terminate()
        server2.terminate()
        server3.terminate()
        tproxy.terminate()
    except ImportError:
        print("Skip because tornado is not available")


def check_tracker_rejects_oversized_msg_size():
    """Tracker must reject an oversized msg_size header and close the connection
    instead of buffering an unbounded amount of data on a single TCP connection.

    Regression test for the unbounded buffer growth defect in
    TCPEventHandler.on_message. See MAX_TRACKER_MSG_BYTES in tracker.py.
    """
    try:
        # pylint: disable=import-outside-toplevel
        import socket
        import struct

        from tvm.rpc import base, tracker

        tserver = tracker.Tracker(port=9180, port_end=9290, silent=True)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(("127.0.0.1", tserver.port))
            # complete the 4-byte magic handshake
            sock.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
            magic_reply = sock.recv(4)
            assert struct.unpack("<i", magic_reply)[0] == base.RPC_TRACKER_MAGIC

            # send an oversized msg_size header (2 GiB)
            sock.sendall(struct.pack("<i", 0x7FFFFFFF))

            # server must close the connection (no payload buffering)
            for _ in range(20):
                chunk = sock.recv(4096)
                if chunk == b"":
                    break
                time.sleep(0.05)
            else:
                raise AssertionError("tracker did not close connection after oversized msg_size")
        finally:
            tserver.terminate()
    except ImportError:
        print("Skip because tornado is not available")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_server_drop()
    check_tracker_rejects_oversized_msg_size()
