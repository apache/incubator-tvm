# isort: skip_file
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
"""Lightweight TVM RPC module.

RPC enables connect to a remote server, upload and launch functions.
This is useful to for cross-compile and remote testing,
The compiler stack runs on local server, while we use RPC server
to run on remote runtime which don't have a compiler available.

The test program compiles the program on local server,
upload and run remote RPC server, get the result back to verify correctness.

TVM RPC server assumes that the user is trusted and needs to be
used in a trusted network environment and encrypted channels.
It allows writings of arbitrary files into the server and provide
full remote code execution capabilities to anyone who can access this API.
"""

from .server import Server
from .client import connect, connect_tracker
from .client import RPCSession, LocalSession, PopenSession, TrackerSession
from .minrpc import with_minrpc

from .remote import RemoteDevice, RemoteTensor, RemoteStream, time_evaluator, virtual_machine
