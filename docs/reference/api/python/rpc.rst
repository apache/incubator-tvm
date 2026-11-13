..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

tvm.rpc
-------
RPC devices and tensors belong to a session. They are distinct from local
``Device`` and ``Tensor`` values::

    remote = tvm.rpc.connect(host, port)
    device = remote.cpu()
    x = device.tensor(numpy_data)
    y = device.empty(x.shape, x.dtype)
    module = remote.load_module("kernel.so")
    module(x, y)
    result = y.numpy()

Use ``device.empty`` and ``device.tensor`` for remote allocations. Normal
``tvm.runtime.empty`` and ``tvm.runtime.tensor`` allocate locally. Remote
functions reject tensors, devices, modules, and objects from other sessions.
``RemoteTensor.copyto(tvm.cpu())`` explicitly materializes a local tensor;
remote tensors cannot export a local DLPack pointer. Views retain their remote
backing storage, and returned tensors retain the remote tensor object.

For a remote VM, use ``tvm.rpc.virtual_machine(module, device)``. This helper
validates session ownership and passes ordinary endpoint device descriptors
to VM initialization. ``tvm.runtime.time_evaluator`` similarly validates the
remote device through its RPC wrapper.

Streams remain owned by the session until ``device.free_stream(stream)`` or
session teardown. Selecting a temporary stream is safe, and aliases observe
explicit release. ``remote.close()`` closes the session; existing remote
values reject further operations while cleanup remains safe.

.. automodule:: tvm.rpc
   :members:
   :imported-members:
   :autosummary:
