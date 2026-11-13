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
# pylint: disable=missing-docstring
# ruff: noqa: F841

import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.s_tir import script as Ts
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


@Ts.function
def main(p0: T.Tensor((), "int32"), T_stack: T.Tensor((T.int64(3),), "int32")):
    T.func_attr({"tirx.noalias": True})
    # with Ts.sblock("root"):
    compile_engine_const = Ts.sblock_alloc_tensor((), "int32")
    compile_engine_const_1 = Ts.sblock_alloc_tensor((), "int32")
    with Ts.sblock("compile_engine_const"):
        vi = Ts.axis.spatial(1, T.int64(0))
        Ts.reads()
        Ts.writes(compile_engine_const[()])
        compile_engine_const[()] = 16
    with Ts.sblock("compile_engine_const_1"):
        vi = Ts.axis.spatial(1, T.int64(0))
        Ts.reads()
        Ts.writes(compile_engine_const_1[()])
        compile_engine_const_1[()] = 20
    for ax0 in range(T.int64(3)):
        with Ts.sblock("T_stack"):
            v_ax0 = Ts.axis.spatial(T.int64(3), ax0)
            Ts.reads(compile_engine_const[()], p0[()], compile_engine_const_1[()])
            Ts.writes(T_stack[v_ax0])
            T_stack[v_ax0] = T.if_then_else(
                v_ax0 == T.int64(2),
                compile_engine_const[()],
                T.if_then_else(v_ax0 == T.int64(1), p0[()], compile_engine_const_1[()]),
            )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_normalize_function_with_scalar():
    sch = tvm.s_tir.Schedule(main)
    f_normalize_function = tvm_ffi.get_global_func("s_tir.schedule.NormalizeFunction")
    assert f_normalize_function(sch)


if __name__ == "__main__":
    tvm_tirx.testing.main()
