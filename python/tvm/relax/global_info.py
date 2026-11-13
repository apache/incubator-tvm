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
"""Concrete global information used by Relax."""

import tvm_ffi

from tvm_tirx.ir import GlobalInfo
from tvm_tirx.runtime import Device, MemoryScope
from tvm_tirx.target import Target

from . import _ffi_api


@tvm_ffi.register_object("relax.DummyGlobalInfo")
class DummyGlobalInfo(GlobalInfo):
    """DummyGlobalInfo"""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.DummyGlobalInfo,
        )


@tvm_ffi.register_object("relax.VDevice")
class VDevice(GlobalInfo):
    """VDevice"""

    def __init__(
        self,
        target=None,
        vdevice_id: int = 0,
        memory_scope: str | MemoryScope = MemoryScope("global"),
    ) -> None:
        if isinstance(target, dict | str):
            target = Target(tvm_ffi.convert(target))
        if isinstance(target, Device):
            target = Target.from_device(target)
        self.__init_handle_by_constructor__(_ffi_api.VDevice, target, vdevice_id, memory_scope)
