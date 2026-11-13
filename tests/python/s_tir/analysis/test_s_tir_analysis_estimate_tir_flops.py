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

import sys

import pytest

import tvm
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.analysis import estimate_tir_flops
from tvm.s_tir.meta_schedule.testing.te_workload import create_te_workload
from tvm_tirx.ir import IRModule
from tvm_tirx.script import tirx as T


@pytest.mark.parametrize(
    "workload, flops",
    [
        ("C1D", 6291456),
        ("C2D", 236027904),
        ("C3D", 13217562624),
        ("CAP", 75497472),
        ("DEP", 7225344),
        ("DIL", 223552896),
        ("GMM", 4194304),
        ("GRP", 28901376),
        ("T2D", 268435456),
        ("CBR", 239239168),
        ("TBG", 25165824),
        ("NRM", 131072),
        ("SFM", 262144),
    ],
)
def test_te_workload(workload, flops):
    te_workload = create_te_workload(workload, 0)
    mod = IRModule({"main": te_workload})
    assert float(flops) == estimate_tir_flops(mod)


@Ts.function
def flops_with_let(a: T.Tensor(16, "float32")):
    for i in range(8):
        j = i + 8
        a[j] = a[i]


def test_flops_with_let():
    flops = estimate_tir_flops(IRModule({"main": flops_with_let}))
    assert flops == 8


@Ts.function
def flops_with_if(a: T.Tensor(16, "float32"), b: T.Tensor(16, "float32")):
    for i in range(16):
        if i % 2 == 0:
            a[i] = b[i]
        else:
            if i % 3 == 0:
                a[i] = b[i - 1] + b[i - 2]


def test_flops_with_if():
    flops = estimate_tir_flops(IRModule({"main": flops_with_if}))
    assert flops == 16


@Ts.function
def flops_with_forloop_as_expression(A: T.Tensor(1, "float32")):
    for i in T.serial(0, 16):
        for k in T.serial(0, i):
            A[0] = A[0] + 1


@Ts.function
def flops_override(A: T.Tensor(16, "float32")):
    T.func_attr({"estimated_flops": 32})
    for i in range(16):
        A[0] = A[0] + 1


def test_estimate_flops_forloop_as_expression():
    flops = estimate_tir_flops(
        IRModule({"main": flops_with_forloop_as_expression.with_attr("estimated_flops", 32)})
    )
    assert flops == 32

    # test whether the user estimated flop would over ride
    flops = estimate_tir_flops(IRModule({"main": flops_override}))
    assert flops == 32


def test_estimate_flops_with_decl_tensor():
    @Ts.function
    def func(A_data: T.handle("float32")):
        A = T.decl_tensor(A_data, ty_args=[T.Tensor(16, "float32", data=A_data)])
        for i in range(16):
            A[0] = A[0] + 1

    flops = estimate_tir_flops(IRModule.from_expr(func))
    assert flops == 16


@Ts.function
def flops_with_nonint_extent(a: T.Tensor(16, "float32")):
    for i in range(4 + 4):
        a[i] = 2 * a[i]


def test_flops_with_nonint_extent():
    assert estimate_tir_flops(IRModule({"main": flops_with_nonint_extent})) == 8


@Ts.function
def flops_with_variable_extent(a: T.Tensor(16, "float32")):
    for i in range(4 + 4):
        for j in range(i + 8):
            a[j] = 2 * a[i]


def test_flops_with_variable_extent():
    assert estimate_tir_flops(IRModule({"main": flops_with_variable_extent})) == 120


if __name__ == "__main__":
    tvm_tirx.testing.main()
