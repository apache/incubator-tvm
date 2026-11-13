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
"""Availability of parent TVM integrations and remote test devices."""

import functools
import json
import os

import tvm_ffi

import tvm_tirx
from tvm import support
from tvm_tirx.testing import env


def device_enabled(target):
    """Check parent test target selection, device availability, and cuDNN requirements."""
    if isinstance(target, dict):
        kind = target["kind"]
    elif hasattr(target, "kind"):
        kind = target.kind.name
    else:
        kind = target.split()[0]
    selected = os.environ.get("TVM_TEST_TARGETS", "").strip()
    candidates = [target]
    if selected:
        candidates = [
            json.loads(entry) if entry.startswith("{") else entry
            for value in selected.split(";")
            if (entry := value.strip())
        ]
    for candidate in candidates:
        candidate_kind = (
            candidate["kind"]
            if isinstance(candidate, dict)
            else (candidate.kind.name if hasattr(candidate, "kind") else candidate.split()[0])
        )
        if candidate_kind != kind:
            continue
        if kind == "hexagon":
            return has_hexagon()
        if not tvm_tirx.testing.device_enabled(candidate):
            continue
        libraries = set(tvm_tirx.target.Target(candidate).attrs.get("libs", []))
        libraries.update(tvm_tirx.target.Target(target).attrs.get("libs", []))
        if kind == "cuda" and "cudnn" in libraries:
            exists = tvm_ffi.get_global_func("tvm.contrib.cudnn.exists", allow_missing=True)
            if not (has_cudnn() and exists is not None and exists()):
                continue
        return True
    return False


@functools.cache
def build_flag_enabled(flag: str) -> bool:
    """Return whether an optional build flag (e.g. ``USE_CUTLASS``) is on.

    A flag counts as enabled unless it is explicitly disabled, so library
    flags carrying a path (rather than a boolean) still register as present.
    """
    try:
        value = support.libinfo().get(flag, "OFF")
        return str(value).lower() not in ("off", "false", "0")
    except Exception:  # pylint: disable=broad-except
        return False


def has_cudnn() -> bool:
    """True if cuDNN was built in and a CUDA device is present."""
    return env.has_cuda() and build_flag_enabled("USE_CUDNN")


def has_cublas() -> bool:
    """True if cuBLAS was built in and a CUDA device is present."""
    return env.has_cuda() and build_flag_enabled("USE_CUBLAS")


def has_nccl() -> bool:
    """True if NCCL was built in and a CUDA device is present."""
    try:
        return env.has_cuda() and (
            tvm_ffi.get_global_func("runtime.disco.nccl.init_ccl", allow_missing=True) is not None
        )
    except Exception:  # pylint: disable=broad-except
        return False


def has_hipblas() -> bool:
    """True if hipBLAS was built in and a ROCm device is present."""
    return env.has_rocm() and build_flag_enabled("USE_HIPBLAS")


@functools.cache
def has_nvshmem() -> bool:
    """True if the disco NVSHMEM runtime is available (requires CUDA).

    Probes the runtime global function rather than the ``USE_NVSHMEM`` build
    flag, since the flag can be set in builds that do not ship the runtime.
    """
    try:
        return env.has_cuda() and (
            tvm_ffi.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid", allow_missing=True)
            is not None
        )
    except Exception:  # pylint: disable=broad-except
        return False


@functools.cache
def has_hexagon() -> bool:
    """True if Hexagon can both compile and run (toolchain + attached device)."""
    from tvm_tirx.backend.hexagon.linker import has_hexagon_toolchain

    try:
        return (
            env._runtime_enabled("hexagon")
            and has_hexagon_toolchain()
            and "ANDROID_SERIAL_NUMBER" in os.environ
        )
    except Exception:  # pylint: disable=broad-except
        return False


@functools.cache
def has_adreno_opencl() -> bool:
    """True if remote Adreno OpenCL testing is configured (RPC_TARGET set)."""
    return build_flag_enabled("USE_OPENCL") and os.environ.get("RPC_TARGET") is not None
