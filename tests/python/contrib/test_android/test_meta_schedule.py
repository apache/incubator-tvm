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
# ruff: noqa: F401

"""Test rpc based launcher for Android"""

import tempfile

import numpy as np
import pytest

pytest.importorskip("scipy")  # tvm.topi.testing imports scipy

import tvm
import tvm.topi.testing
import tvm_tirx.testing
from tvm.s_tir import meta_schedule as ms
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule.builder import LocalBuilder
from tvm_tirx.script import tirx as T

from .infrastructure import get_android_gpu_target, get_rpc_runner


@Ts.function
def matmul(
    A: T.Tensor([128, 128], "float32"),
    B: T.Tensor([128, 128], "float32"),
    C: T.Tensor([128, 128], "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@pytest.mark.skip("Integration test")
def test_tune_tir_on_android():
    """Test tune_tir on Android through RPC."""
    max_workers = 4
    builder = LocalBuilder(
        f_export="s_tir.meta_schedule.builder.export_ndk", max_workers=max_workers
    )
    runner = get_rpc_runner()
    target = get_android_gpu_target()
    with tempfile.TemporaryDirectory() as work_dir:
        database = ms.tir_integration.tune_tir(
            mod=matmul,
            target=target,
            work_dir=work_dir,
            max_trials_global=32,
            num_trials_per_iter=16,
            builder=builder,
            runner=runner,
        )
        sch = ms.tir_integration.compile_tir(database, matmul, target)
        if sch is None:
            print("No valid schedule found!")
        else:
            sch.mod.show()
            sch.trace.show()


if __name__ == "__main__":
    tvm_tirx.testing.main()
