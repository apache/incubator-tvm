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
"""Specialized applications of trace"""

from tvm.s_tir import Schedule, Trace
from tvm_tirx.target import Target

from . import _ffi_api


def schedule_using_anchor_trace(sch: Schedule, anchor_trace: Trace, target: Target) -> None:
    """Apply the trace from a TIR module whose anchor block is the same but fused elemewise op
    blocks differ. This function can be used for transferring a trace tuned on a conv2d -> add
    subgraph to other subgraphs having the same conv2d workload, for example. We call such trace
    an "anchor trace". Those blocks that are not scheduled by the given anchor trace will be either
    inlined or parallelized.

    Parameters
    ----------
    sch : Schedule
        The target schedule
    anchor_trace: Trace
        The trace generated for other TIR module having the same anchor block
    target : tvm_tirx.target.Target
        The compilation target
    """
    _ffi_api.ScheduleUsingAnchorTrace(sch, anchor_trace, target)  # type: ignore
