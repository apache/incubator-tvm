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
from collections import defaultdict

import numpy
import pytest

import tvm
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm.s_tir.schedule.testing import verify_trace_roundtrip
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T

# pylint: disable=no-member,invalid-name,unused-variable


@Ts.function
def elementwise(
    A: T.Tensor((128, 257, 1470), "float32"), B: T.Tensor((128, 257, 1470), "float32")
) -> None:
    for i, j, k in T.grid(128, 257, 1470):
        with Ts.sblock("B"):
            vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@Ts.function
def tiled_conv2d_with_padding(
    inputs: T.Tensor((1, 224, 224, 3), "float32"),
    weight: T.Tensor((7, 7, 3, 64), "float32"),
    conv2d_nhwc: T.Tensor((1, 112, 112, 64), "float32"),
) -> None:
    PadInput = Ts.sblock_alloc_tensor([1, 230, 230, 3], dtype="float32")
    for i0, i1, i2, i3 in T.grid(1, 230, 230, 3):
        with Ts.sblock("PadInput"):
            i0_1, i1_1, i2_1, i3_1 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            Ts.reads(inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1])
            Ts.writes(PadInput[i0_1, i1_1, i2_1, i3_1])
            PadInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(
                3 <= i1_1 and i1_1 < 227 and 3 <= i2_1 and i2_1 < 227,
                inputs[i0_1, i1_1 - 3, i2_1 - 3, i3_1],
                T.float32(0),
            )
    for (
        i0_0,
        i1_0,
        i2_0,
        i3_0,
        i0_1_1,
        i1_1_1,
        i2_1_1,
        i3_1_1,
        i4_0,
        i5_0,
        i6_0,
        i0_2,
        i1_2,
        i2_2,
        i3_2,
        i4_1,
        i5_1,
        i6_1,
        i0_3,
        i1_3,
        i2_3,
        i3_3,
    ) in T.grid(1, 1, 4, 1, 1, 2, 4, 1, 7, 7, 1, 1, 1, 1, 1, 1, 1, 3, 1, 56, 7, 64):
        with Ts.sblock("conv2d_nhwc"):
            n = Ts.axis.spatial(1, 0)
            h = Ts.axis.spatial(112, i1_1_1 * 56 + i1_3)
            w = Ts.axis.spatial(112, i2_0 * 28 + i2_1_1 * 7 + i2_3)
            co, rh, rw, rc = Ts.axis.remap("SRRR", [i3_3, i4_0, i5_0, i6_1])
            Ts.reads(
                conv2d_nhwc[n, h, w, co],
                PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc],
                weight[rh, rw, rc, co],
            )
            Ts.writes(conv2d_nhwc[n, h, w, co])
            with Ts.init():
                conv2d_nhwc[n, h, w, co] = T.float32(0)
            conv2d_nhwc[n, h, w, co] = (
                conv2d_nhwc[n, h, w, co]
                + PadInput[n, h * 2 + rh, w * 2 + rw, co // 64 * 3 + rc] * weight[rh, rw, rc, co]
            )


# pylint: enable=no-member,invalid-name,unused-variable


def test_sample_categorical():
    """Test sample categorical sampling function"""
    n = 1000
    sch = tvm.s_tir.Schedule(elementwise, seed=42, debug_mask="all")
    counter = defaultdict(int)
    candidates = [5, 2, 7, 1]
    probs = [0.15, 0.55, 0.05, 0.25]
    for _ in range(n):
        v = sch.get(sch.sample_categorical(candidates, probs))
        counter[v] += 1
    for i, prob in enumerate(probs):
        assert (prob - 0.07) * n <= counter[candidates[i]] <= (prob + 0.07) * n
    verify_trace_roundtrip(sch, mod=elementwise)


def test_sample_categorical_copy():
    """Check the random variable sampling results after schedule copy"""
    n = 100
    sch = tvm.s_tir.Schedule(elementwise, seed=42, debug_mask="all")
    candidates = [1, 2, 3, 4]
    probs = [0.1, 0.2, 0.3, 0.4]
    rv_decisions = []
    for _ in range(n):
        rv = sch.sample_categorical(candidates, probs)  # pylint: disable=invalid-name
        rv_decisions.append((rv, sch.get(rv)))
    sch_copy = sch.copy()
    for rv, decision in rv_decisions:  # pylint: disable=invalid-name
        decision_copy = sch_copy.get(rv)
        assert int(decision) == int(decision_copy)


def test_sample_categorical_serialize():
    """Check the random variable sampling results after schedule serialization"""
    n = 100
    sch = tvm.s_tir.Schedule(elementwise, seed=42, debug_mask="all")
    candidates = [5, 6, 7, 8]
    probs = [0.23, 0.19, 0.37, 0.21]
    decisions = []
    for _ in range(n):
        rv = sch.get(sch.sample_categorical(candidates, probs))  # pylint: disable=invalid-name
        decisions.append(rv)
    new_sch = verify_trace_roundtrip(sch, mod=elementwise)
    for i, new_inst in enumerate(new_sch.trace.insts):
        assert decisions[i] == candidates[new_sch.trace.decisions[new_inst]]


def test_sample_perfect_tile_power_of_two():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    i, _, _ = sch.get_loops(sch.get_sblock("B"))
    factors = sch.sample_perfect_tile(i, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 128
    verify_trace_roundtrip(sch, mod=elementwise)


def test_sample_perfect_tile_prime():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    _, i, _ = sch.get_loops(sch.get_sblock("B"))
    factors = sch.sample_perfect_tile(i, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 257
    verify_trace_roundtrip(sch, mod=elementwise)


def test_sample_perfect_tile_composite():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    _, _, i = sch.get_loops(sch.get_sblock("B"))
    factors = sch.sample_perfect_tile(i, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 1470
    verify_trace_roundtrip(sch, mod=elementwise)


@pytest.fixture(
    params=[pytest.param(False, id="block_obj"), pytest.param(True, id="block_name")],
    scope="session",
)
def use_sugared_block(request):
    return request.param


def test_sample_compute_location(use_sugared_block):
    n = 100
    sch = tvm.s_tir.Schedule(tiled_conv2d_with_padding, seed=42, debug_mask="all")
    if use_sugared_block:
        pad_input = "PadInput"
    else:
        pad_input = sch.get_sblock("PadInput")
    decision_dict = dict()
    for _ in range(n):
        _ = sch.sample_compute_location(pad_input)  # pylint: disable=invalid-name
        decision = sch.trace.decisions[sch.trace.insts[-1]]
        decision_dict[decision] = decision_dict[decision] + 1 if decision in decision_dict else 1

    n_candidates = 8
    expected_rate = 1.0 / n_candidates
    for _, cnt in decision_dict.items():
        numpy.testing.assert_allclose(expected_rate, cnt / n, atol=0.04)


def test_sample_perfect_tile_after_copy():
    sch = tvm.s_tir.Schedule(elementwise, debug_mask="all")
    sch_copy = sch.copy()
    _, _, i = sch.get_loops(sch.get_sblock("B"))
    sch.sample_perfect_tile(i, n=4)

    _, _, i = sch_copy.get_loops(sch_copy.get_sblock("B"))
    # Hangs if ForkSeed is not invoked when copying a schedule
    sch_copy.sample_perfect_tile(i, n=4)


def test_sample_perfect_tile_on_dynamic_loops():
    """Currently dynamic loop is trivially tiled"""

    n = T.dynamic("n", "int32")

    @Ts.function
    def workload(A: T.Tensor((n, 1024), "float32")) -> None:
        for i, j in T.grid(n, 1024):
            with Ts.sblock("B"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                A[vi, vj] = 1.0

    sch = tvm.s_tir.Schedule(workload, debug_mask="all")
    di, si = sch.get_loops(sch.get_sblock("B"))

    factors = sch.sample_perfect_tile(si, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 1024

    factors = sch.sample_perfect_tile(di, n=4)
    assert factors[0] is None
    factors = [sch.get(i) for i in factors[1:]]
    prod = factors[0] * factors[1] * factors[2]
    assert prod == 1
    verify_trace_roundtrip(sch, mod=workload)


if __name__ == "__main__":
    tvm_tirx.testing.main()
