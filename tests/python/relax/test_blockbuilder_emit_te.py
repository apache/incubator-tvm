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
"""This file tests advanced emit_te features with help of TVMScript assertion"""

# The tests here depend on tvmscript

import tvm
import tvm_tirx
from tvm import relax as rx
from tvm import te
from tvm.s_tir import script as Ts
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


def test_emit_te_with_symbolic_arg():
    bb = rx.BlockBuilder()
    m = T.dynamic("m", "int64")
    x = rx.Var("x", R.Tensor([10], "float32"))
    y = rx.Var("y", R.Shape([m]))

    def te_func(A, offset):
        return te.compute(A.shape, lambda i: A[i + offset], name="B")

    with bb.function("main", [x, y]):
        out = bb.emit_te(te_func, x, m)
        bb.emit_func_output(out)

    after = bb.get()

    m = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def te_func(
            A: T.Tensor((T.int64(10),), "float32"),
            m: T.int64,
            B: T.Tensor((T.int64(10),), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i in range(T.int64(10)):
                with Ts.sblock("B"):
                    v_i = Ts.axis.spatial(T.int64(10), i)
                    Ts.writes(B[v_i])
                    B[v_i] = A[v_i + m]

        @R.function
        def main(x: R.Tensor((10,), dtype="float32"), y: R.Shape([m])) -> R.Tensor(
            (10,), dtype="float32"
        ):
            cls = Expected
            gv = R.call_tir(
                cls.te_func,
                (x, m),
                ty_args=[R.Tensor((10,), dtype="float32")],
            )
            return gv

    assert_structural_equal(after, Expected)


def test_symbolic_shape_in_prim_value():
    """Scalar primitive Vars become ordinary call_tir arguments."""

    def te_slice(tensor, i):
        return tvm.te.compute([tensor.shape[1]], lambda j: tensor[i, j], name="slice")

    def from_builder():
        bb = rx.BlockBuilder()
        A = rx.Var("A", R.Tensor([16, 16], "float32"))
        relax_i = rx.Var("relax_i", tvm_tirx.ir.PrimType("int64"))

        with bb.function("main", params=[A, relax_i]):
            A_sliced = bb.emit_te(te_slice, A, relax_i)
            bb.emit_func_output(A_sliced)

        return bb.get()

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def te_slice(
            A: T.Tensor([T.int64(16), T.int64(16)], "float32"),
            row_index: T.int64,
            Output: T.Tensor(T.int64(16), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for i in T.serial(T.int64(0), A.shape[1]):
                with Ts.sblock("slice"):
                    vi = Ts.axis.remap("S", [i])
                    Output[vi] = A[row_index, vi]

        @R.function
        def main(
            A: R.Tensor([16, 16], "float32"),
            arg_row_index: T.int64,
        ):
            cls = Expected

            gv = R.call_tir(
                cls.te_slice,
                (A, arg_row_index),
                ty_args=[R.Tensor([16], "float32")],
            )
            return gv

    tvm_tirx.ir.assert_structural_equal(from_builder(), Expected)
