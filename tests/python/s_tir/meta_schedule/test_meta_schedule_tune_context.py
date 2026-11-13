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
"""Test the tune context of meta schedule."""

import sys

import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.meta_schedule import TuneContext
from tvm_tirx.script import tirx as T
from tvm_tirx.target import Target

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@tvm_tirx.script.ir_module
class Matmul:
    @Ts.function
    def main(
        A: T.Tensor((1024, 1024), "float32"),
        B: T.Tensor((1024, 1024), "float32"),
        C: T.Tensor((1024, 1024), "float32"),
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        for i, j, k in T.grid(1024, 1024, 1024):
            with Ts.sblock("matmul"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


def test_tune_context_create():
    mod = Matmul
    context = TuneContext(mod=mod, target=Target("llvm"), task_name="Test Task")
    assert context.num_threads > 0
    assert context.rand_state != -1
    assert context.task_name == "Test Task"
    assert context.mod == mod or tvm_ffi.structural_equal(context.mod, mod)


if __name__ == "__main__":
    tvm_tirx.testing.main()
