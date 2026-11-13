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
"""Extracted tasks from high-level IR."""

import tvm_ffi
from tvm_ffi import register_object

from tvm_tirx.ir import IRModule
from tvm_tirx.target import Target

from . import _ffi_api


@register_object("s_tir.meta_schedule.ExtractedTask")
class ExtractedTask(tvm_ffi.Object):
    """A tuning task extracted from the high-level IR

    Parameters
    ----------
    task_name : str
        The name of the task extracted
    mod : IRModule
        The high-level IR
    target: Target
        Target information
    dispatched : List[IRModule]
        A list of low-level IRs that the high-level IR could potentially dispatch to
    weight : int
        The weight of the task
    """

    task_name: str
    mod: IRModule
    dispatched: list[IRModule]
    weight: int

    def __init__(
        self,
        task_name: str,
        mod: IRModule,
        target: Target,
        dispatched: list[IRModule],
        weight: int,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ExtractedTask,  # type: ignore # pylint: disable=no-member
            task_name,
            mod,
            target,
            dispatched,
            weight,
        )
