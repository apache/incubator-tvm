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
"""Relax distributed-related transformation passes."""

import tvm_tirx.ir

from . import _ffi_api


def PropagateSharding() -> tvm_tirx.ir.transform.Pass:
    """Propagate sharding information.

    Returns
    -------
    ret : tvm_tirx.ir.transform.Pass
        The registered pass
    """
    return _ffi_api.PropagateSharding()  # type: ignore


def LowerGlobalViewToLocalView() -> tvm_tirx.ir.transform.Pass:
    """Lower global view TIR to local view

    Returns
    -------
    ret : tvm_tirx.ir.transform.Pass
        The registered pass
    """
    return _ffi_api.LowerGlobalViewToLocalView()  # type: ignore


def LegalizeRedistribute() -> tvm_tirx.ir.transform.Pass:
    """Legalize redistribute op to ccl op.
    S->R: R.ccl.allgather
    R->S: R.dist.redistribute_replica_to_shard

    Returns
    -------
    ret : tvm_tirx.ir.transform.Pass
        The registered pass
    """
    return _ffi_api.LegalizeRedistribute()  # type: ignore


def LowerDistIR() -> tvm_tirx.ir.transform.Pass:
    """Lower DistIR to Relax

    Returns
    -------
    ret : tvm_tirx.ir.transform.Pass
        The registered pass
    """
    return _ffi_api.LowerDistIR()  # type: ignore
