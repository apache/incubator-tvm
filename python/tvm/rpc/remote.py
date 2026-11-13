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
"""Explicit RPC-owned devices, tensors, and streams.

A remote tensor is not a local tensor or a DLPack provider. Use ``numpy`` or
``copyto(tvm_tirx.cpu())`` to materialize its contents in the client process.
"""

import numpy as np
import tvm_ffi

import tvm_tirx.runtime

from . import _ffi_api


def _register_remote_object(type_key):
    # Keep RPC imports available for feature detection in builds without RPC.
    if tvm_ffi.get_global_func("rpc.RemoteDevice", allow_missing=True) is None:
        return lambda cls: cls
    return tvm_ffi.register_object(type_key)


class _RemoteDeviceAttributeQuery:
    """Bind Device's shared property definitions to an RPC-owned query hook.

    This is only a property receiver, never a local Device or tensor factory
    argument. Keeping conversions in Device avoids a second attribute catalog.
    """

    def __init__(self, remote):
        self.remote = remote

    def get_attr(self, attr):
        return self.remote.get_attr(attr)


@_register_remote_object("rpc.RemoteDevice")
class RemoteDevice(tvm_ffi.Object):
    """A device descriptor interpreted in one RPC session's execution endpoint.

    ``endpoint_device`` is a read-only ordinary Device containing only the device
    type and index. The native object retains its session; the descriptor alone
    does not identify that session and must not be used for local allocation.
    Device capability attributes use the same definitions and conversions as
    :class:`tvm_tirx.runtime.Device`, with queries dispatched by the owning session.
    Explicit session closure invalidates allocation, queries and stream calls.
    """

    @property
    def index(self):
        return self.endpoint_device.index

    @property
    def type(self):
        return self.endpoint_device.type

    def __repr__(self):
        return f"RemoteDevice({self.endpoint_device})"

    def get_attr(self, attr):
        """Query named device attributes through the owning session.

        Results and backend errors are those of DeviceAPI.GetAttr; named Python
        capabilities additionally use Device's existing result conversions.
        """
        return _ffi_api.DeviceGetAttr(self, attr)

    def __getattr__(self, name):
        # FFI reflection owns fields such as endpoint_device. Only missing fields
        # fall through to the ordinary Device's shared attribute definitions.
        try:
            return super().__getattr__(name)
        except AttributeError:
            attribute = getattr(tvm_tirx.runtime.Device, name, None)
            if not isinstance(attribute, property):
                raise AttributeError(f"RemoteDevice has no attribute {name!r}") from None
            return attribute.__get__(_RemoteDeviceAttributeQuery(self))

    def empty(self, shape, dtype="float32", mem_scope=None):
        """Allocate a remote tensor through this device's session."""
        return _ffi_api.Empty(tvm_ffi.Shape(shape), tvm_ffi.dtype(dtype), self, mem_scope)

    def tensor(self, data, dtype=None):
        """Allocate a remote tensor and copy data into it synchronously."""
        if isinstance(data, RemoteTensor):
            if dtype is not None and tvm_ffi.dtype(dtype) != data.dtype:
                raise ValueError("Remote tensor copy cannot convert dtype; materialize it first")
            result = self.empty(data.shape, data.dtype)
            return result.copyfrom(data)
        local = (
            data
            if isinstance(data, tvm_ffi.Tensor) and dtype is None
            else tvm_tirx.runtime.tensor(np.asarray(data, dtype=dtype))
        )
        return self.empty(local.shape, local.dtype).copyfrom(local)

    def set_device(self):
        _ffi_api.DeviceSetDevice(self)

    def create_stream(self):
        return _ffi_api.DeviceCreateStream(self)

    def free_stream(self, stream):
        _ffi_api.DeviceFreeStream(self, stream)

    def current_stream(self):
        return _ffi_api.DeviceCurrentStream(self)

    def set_stream(self, stream=None):
        _ffi_api.DeviceSetStream(self, stream)

    def sync(self, stream=None):
        _ffi_api.DeviceSync(self, stream)


@_register_remote_object("rpc.RemoteStream")
class RemoteStream(tvm_ffi.Object):
    """An opaque stream tied to one remote device and its session.

    The session retains created streams until ``device.free_stream`` or session
    teardown, including streams selected using a temporary Python wrapper.
    Current-stream aliases share closure state; releasing one invalidates them
    all. Stream operations reject a different device or session before use.
    """


@_register_remote_object("rpc.RemoteTensor")
class RemoteTensor(tvm_ffi.Object):
    """Contiguous remote storage with explicit metadata and transfer operations.

    Read-only ``device`` identifies the owning RemoteDevice; ``shape`` and
    ``dtype`` describe the logical tensor independently of remote metadata.
    Views retain shared backing storage, and returned tensors retain the remote
    object handle until the last local owner disappears. The backing owner also
    retains the session. No data pointer is exposed for local use or DLPack;
    ``numpy`` and ``copyto`` explicitly materialize data in client memory.
    """

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return f"RemoteTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    def __dlpack__(self, *args, **kwargs):
        raise TypeError("RemoteTensor cannot export local DLPack; copyto(tvm_tirx.cpu()) first")

    def __dlpack_device__(self):
        raise TypeError("RemoteTensor has no local DLPack device; copyto(tvm_tirx.cpu()) first")

    def copyfrom(self, source):
        """Copy a local value or a tensor from the same remote session."""
        if isinstance(source, RemoteTensor):
            _ffi_api.CopyFromTo(source, self, None)
            source.device.sync()
            self.device.sync()
        else:
            if not isinstance(source, tvm_ffi.Tensor):
                source = tvm_tirx.runtime.tensor(np.asarray(source, dtype=str(self.dtype)))
            if tuple(source.shape) != tuple(self.shape):
                raise ValueError("Tensor copy shape mismatch")
            if str(source.dtype) != str(self.dtype):
                raise ValueError("Tensor copy dtype mismatch")
            _ffi_api.CopyFromLocal(source, self)
        return self

    def copyto(self, target):
        """Synchronously copy into storage or allocate on an explicit device."""
        if isinstance(target, RemoteDevice):
            target = target.empty(self.shape, self.dtype)
        if isinstance(target, RemoteTensor):
            return target.copyfrom(self)
        if isinstance(target, tvm_tirx.runtime.Device):
            is_cpu = target.dlpack_device_type() == tvm_tirx.runtime.cpu().dlpack_device_type()
            local = tvm_tirx.runtime.empty(
                self.shape, self.dtype, target if is_cpu else tvm_tirx.runtime.cpu()
            )
            _ffi_api.CopyToLocal(self, local)
            if is_cpu:
                return local
            return local.copyto(target)
        _ffi_api.CopyToLocal(self, target)
        return target

    def numpy(self):
        """Materialize in client memory and convert to NumPy."""
        return self.copyto(tvm_tirx.runtime.cpu()).numpy()

    def _create_view(self, shape, dtype=None, relative_byte_offset=0):
        return _ffi_api.CreateView(
            self, tvm_ffi.Shape(shape), tvm_ffi.dtype(dtype or self.dtype), relative_byte_offset
        )

    def __setitem__(self, key, value):
        if (
            not isinstance(key, slice)
            or key.start is not None
            or key.stop is not None
            or key.step not in (None, 1)
        ):
            raise ValueError("RemoteTensor only supports full-slice assignment")
        self.copyfrom(value)


def time_evaluator(
    module,
    func_name,
    device,
    number=10,
    repeat=1,
    min_repeat_ms=0,
    limit_zero_time_iterations=100,
    cooldown_interval_ms=0,
    repeats_to_cooldown=1,
    cache_flush_bytes=0,
    f_preproc="",
):
    """Create an evaluator after validating the module and remote device session."""
    from tvm.runtime import time_evaluator as runtime_time_evaluator

    return runtime_time_evaluator(
        module,
        func_name,
        device,
        number,
        repeat,
        min_repeat_ms,
        limit_zero_time_iterations,
        cooldown_interval_ms,
        repeats_to_cooldown,
        cache_flush_bytes,
        f_preproc,
    )


def virtual_machine(module, device):
    """Initialize a remote VM with session-checked endpoint device descriptors."""
    from tvm.runtime.vm import VirtualMachine

    devices = list(device) if isinstance(device, list | tuple) else [device]
    local = [_ffi_api.ValidateDevice(module, dev) for dev in devices]
    return VirtualMachine(module, local)
