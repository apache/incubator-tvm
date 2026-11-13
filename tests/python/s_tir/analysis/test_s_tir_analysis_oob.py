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
import pytest

import tvm
import tvm_tirx
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T


@Ts.function
def bad_load(A: T.Tensor((2, 3), "float32"), B: T.Tensor((3, 2), "float32")):
    B[0, 0] = A[2, 2]


@Ts.function
def bad_load_loop(A: T.Tensor((2, 3), "float32"), B: T.Tensor((3, 2), "float32")):
    for i in range(3):
        B[i, 0] = A[i, 2]


@Ts.function
def bad_store(A: T.Tensor((2, 3), "float32"), B: T.Tensor((3, 2), "float32")):
    B[0, 3] = A[1, 2]


@Ts.function
def bad_store_loop(A: T.Tensor((2, 3), "float32"), B: T.Tensor((3, 2), "float32")):
    for i in range(3):
        B[0, i] = A[1, i]


@Ts.function
def unknown_bounds(A: T.Tensor((2, 3), "float32"), B: T.Tensor((3, 2), "float32"), N: T.int32):
    for i in range(3):
        B[0, N] = A[1, i]


def test_oob_load():
    with pytest.raises(tvm.s_tir.ScheduleError) as err:
        tvm.s_tir.analysis.OOBChecker()(tvm_tirx.IRModule.from_expr(bad_load))
    assert "tensor A" in err.value.args[0]

    with pytest.raises(tvm.s_tir.ScheduleError) as err:
        tvm.s_tir.analysis.OOBChecker()(tvm_tirx.IRModule.from_expr(bad_load_loop))
    assert "tensor A" in err.value.args[0]


def test_oob_store():
    with pytest.raises(tvm.s_tir.ScheduleError) as err:
        tvm.s_tir.analysis.OOBChecker()(tvm_tirx.IRModule.from_expr(bad_store))
    assert "tensor B" in err.value.args[0]

    with pytest.raises(tvm.s_tir.ScheduleError) as err:
        tvm.s_tir.analysis.OOBChecker()(tvm_tirx.IRModule.from_expr(bad_store_loop))
    assert "tensor B" in err.value.args[0]


def test_unknown_bounds():
    # This should not return an error as we can't probe that N goes out of bounds
    tvm.s_tir.analysis.OOBChecker()(tvm_tirx.IRModule.from_expr(unknown_bounds))


if __name__ == "__main__":
    tvm_tirx.testing.main()
