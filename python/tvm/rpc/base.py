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
"""Base definitions for RPC."""
# pylint: disable=invalid-name

import errno
import json
import logging
import random
import socket
import struct
import time

# Magic header for RPC data plane
RPC_MAGIC = 0xFF271
# magic header for RPC tracker(control plane)
RPC_TRACKER_MAGIC = 0x2F271
# sucess response
RPC_CODE_SUCCESS = RPC_MAGIC + 0
# duplicate key in proxy
RPC_CODE_DUPLICATE = RPC_MAGIC + 1
# cannot found matched key in server
RPC_CODE_MISMATCH = RPC_MAGIC + 2

logger = logging.getLogger("RPCServer")


class TrackerCode:
    """Enumeration code for the RPC tracker"""

    FAIL = -1
    SUCCESS = 0
    PING = 1
    STOP = 2
    PUT = 3
    REQUEST = 4
    UPDATE_INFO = 5
    SUMMARY = 6
    GET_PENDING_MATCHKEYS = 7


# Use "127.0.0.1" or "::1" if there is a need to force ip4 or ip6
# connection for "localhost".
def get_addr_family(addr):
    res = socket.getaddrinfo(addr[0], addr[1], 0, 0, socket.IPPROTO_TCP)
    return res[0][0]


def recvall(sock, nbytes):
    """Receive all nbytes from socket.

    Parameters
    ----------
    sock: Socket
       The socket

    nbytes : int
       Number of bytes to be received.
    """
    res = []
    nread = 0
    while nread < nbytes:
        chunk = sock.recv(min(nbytes - nread, 1024))
        if not chunk:
            raise OSError("connection reset")
        nread += len(chunk)
        res.append(chunk)
    return b"".join(res)


def sendjson(sock, data):
    """send a python value to remote via json

    Parameters
    ----------
    sock : Socket
        The socket

    data : object
        Python value to be sent.
    """
    data = json.dumps(data)
    sock.sendall(struct.pack("<i", len(data)))
    sock.sendall(data.encode("utf-8"))


def recvjson(sock):
    """receive python value from remote via json

    Parameters
    ----------
    sock : Socket
        The socket

    Returns
    -------
    value : object
        The value received.
    """
    size = struct.unpack("<i", recvall(sock, 4))[0]
    data = json.loads((recvall(sock, size)).decode("utf-8"))
    return data


def random_key(prefix, delimiter=":", cmap=None):
    """Generate a random key

    Parameters
    ----------
    prefix : str
        The string prefix

    delimiter : str
        The delimiter

    cmap : dict
        Conflict map

    Returns
    -------
    key : str
        The generated random key
    """
    while True:
        key = f"{prefix}{delimiter}{random.random()}"
        if not cmap or key not in cmap:
            break
    return key


def split_random_key(key, delimiter=":"):
    """Split a random key by delimiter into prefix and random part

    Parameters
    ----------
    key : str
        The generated random key

    Returns
    -------
    prefix : str
        The string prefix

    random_part : str
        The generated random
    """
    return key.rsplit(delimiter, 1)


def connect_with_retry(addr, timeout=60, retry_period=5):
    """Connect to a TPC address with retry

    This function is only reliable to short period of server restart.

    Parameters
    ----------
    addr : tuple
        address tuple

    timeout : float
         Timeout during retry

    retry_period : float
         Number of seconds before we retry again.
    """
    tstart = time.time()
    while True:
        try:
            sock = socket.socket(get_addr_family(addr), socket.SOCK_STREAM)
            sock.connect(addr)
            return sock
        except OSError as sock_err:
            if sock_err.args[0] not in (errno.ECONNREFUSED,):
                raise sock_err
            period = time.time() - tstart
            if period > timeout:
                raise RuntimeError(f"Failed to connect to server {addr!s}")
            logger.warning(f"Cannot connect to tracker {addr!s}, retry in {retry_period:g} secs...")
            time.sleep(retry_period)
