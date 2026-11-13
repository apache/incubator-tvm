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
# pylint: disable=invalid-name
"""Adreno Relax transformation passes."""

import tvm_tirx.ir
from tvm_tirx.target import Target

from . import _ffi_api


def AnnotateCustomMemoryScope(target: Target | None = None) -> tvm_tirx.ir.transform.Pass:
    """Allocate the memory scope information. This is Adreno specific pass to annotate
    The memory scope information and realize the same with RealizeVDevice pass followed by
    updating the Functiontion var_buffer mapping using SpecializeFunctionBasedOnCallSite.

    Returns
    -------
    ret: tvm_tirx.ir.transform.Pass
        The registered pass for allocating workspace.
    """
    return _ffi_api.AnnotateCustomMemoryScope(target)  # type: ignore


def FoldVDeviceScopeChange() -> tvm_tirx.ir.transform.Pass:
    """This pass is a texture specific pass that can optimize unnecessary to_device copies.
    Like texture_scope -> ToVDevice -> global scope. In this case the producer can directly
    store into global scope avoiding unnecessary device copy.

    Returns
    -------
    ret: tvm_tirx.ir.transform.Pass
        The registered pass for allocating workspace.
    """
    return _ffi_api.FoldVDeviceScopeChange()  # type: ignore
