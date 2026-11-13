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
# ruff: noqa: E501, F841

import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm.s_tir import script as Ts
from tvm_tirx.ir import Op
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

##################### Indexing #####################


def test_take():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Take:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32"), indices: R.Tensor((4,), "int64")) -> R.Tensor((2, 4, 4), "float32"):
            gv: R.Tensor((2, 4, 4), "float32") = R.take(x, indices, axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32"), indices: R.Tensor((4,), "int64")) -> R.Tensor((2, 4, 4), "float32"):
            gv = R.call_tir(Expected.take, (x, indices), ty_args=[R.Tensor((2, 4, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def take(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), rxplaceholder_1: T.Tensor(T.int64(4), "int64"), T_take: T.Tensor((T.int64(2), T.int64(4), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
                with Ts.sblock("T_take"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, rxplaceholder_1[ax1], ax2], rxplaceholder_1[ax1])
                    Ts.writes(T_take[ax0, ax1, ax2])
                    T_take[ax0, ax1, ax2] = rxplaceholder[ax0, rxplaceholder_1[ax1], ax2]
    # fmt: on

    mod = LegalizeOps()(Take)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_take_prim_value():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Take:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32"), index: T.int64) -> R.Tensor((2, 4), "float32"):
            gv: R.Tensor((2, 4), "float32") = R.take(x, index, axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32"), index: T.int64) -> R.Tensor((2, 4), "float32"):
            gv = R.call_tir(Expected.take, (x, index), ty_args=[R.Tensor((2, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def take(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), index: T.int64, T_take: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i2 in T.grid(T.int64(2), T.int64(4)):
                with Ts.sblock("T_take"):
                    ax0, ax2 = Ts.axis.remap("SS", [i0, i2])
                    Ts.reads(rxplaceholder[ax0, index, ax2])
                    Ts.writes(T_take[ax0, ax2])
                    T_take[ax0, ax2] = rxplaceholder[ax0, index, ax2]
    # fmt: on

    mod = LegalizeOps()(Take)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_take_const_prim_value():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Take:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 4), "float32"):
            gv: R.Tensor((2, 4), "float32") = R.take(x, R.prim_value(0), axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 4), "float32"):
            gv = R.call_tir(Expected.take, (x,), ty_args=[R.Tensor((2, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def take(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), T_take: T.Tensor((T.int64(2), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i2 in T.grid(T.int64(2), T.int64(4)):
                with Ts.sblock("T_take"):
                    ax0, ax2 = Ts.axis.remap("SS", [i0, i2])
                    Ts.reads(rxplaceholder[ax0, T.int64(0), ax2])
                    Ts.writes(T_take[ax0, ax2])
                    T_take[ax0, ax2] = rxplaceholder[ax0, T.int64(0), ax2]
    # fmt: on

    mod = LegalizeOps()(Take)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_take_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    i = T.dynamic("i", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Take:
        @R.function
        def main(x: R.Tensor((m, n), "float32"), indices: R.Tensor((i,), "int64")) -> R.Tensor((m, i), "float32"):
            gv: R.Tensor((m, i), "float32") = R.take(x, indices, axis=1)
            return gv

    m_main = T.dynamic("m", "int64")
    i_main = T.dynamic("i", "int64")
    n_main = T.dynamic("n", "int64")
    i_take = T.dynamic("i", "int64")
    m_take = T.dynamic("m", "int64")
    n_take = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32"), indices: R.Tensor((i_main,), "int64")) -> R.Tensor((m_main, i_main), "float32"):
            gv = R.call_tir(Expected.take, (x, indices), ty_args=[R.Tensor((m_main, i_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def take(rxplaceholder: T.Tensor([m_take, n_take], dtype='float32'), rxplaceholder_1: T.Tensor([i_take], dtype='int64'), T_take: T.Tensor([m_take, i_take], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_take, i_take):
                with Ts.sblock("T_take"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, rxplaceholder_1[ax1]], rxplaceholder_1[ax1])
                    Ts.writes(T_take[ax0, ax1])
                    T_take[ax0, ax1] = rxplaceholder[ax0, rxplaceholder_1[ax1]]
    # fmt: on

    mod = LegalizeOps()(Take)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_take_symbolic_prim_value():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Take:
        @R.function
        def main(x: R.Tensor((2, n, 4), "float32")) -> R.Tensor((2, 4), "float32"):
            gv: R.Tensor((2, 4), "float32") = R.take(x, R.prim_value(n-1), axis=1)
            return gv

    n_main = T.dynamic("n", "int64")
    n_take = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, n_main, 4), "float32")) -> R.Tensor((2, 4), "float32"):
            gv = R.call_tir(Expected.take, (x,), ty_args=[R.Tensor((2, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def take(rxplaceholder: T.Tensor((T.int64(2), n_take, T.int64(4)), 'float32'), T_take: T.Tensor((T.int64(2), T.int64(4)), "float32")):

            T.func_attr({"tirx.noalias": True})
            for i0, i2 in T.grid(T.int64(2), T.int64(4)):
                with Ts.sblock("T_take"):
                    ax0, ax2 = Ts.axis.remap("SS", [i0, i2])
                    Ts.reads(rxplaceholder[ax0, n_take-1, ax2])
                    Ts.writes(T_take[ax0, ax2])
                    T_take[ax0, ax2] = rxplaceholder[ax0, n_take-1, ax2]
    # fmt: on

    mod = LegalizeOps()(Take)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_strided_slice():
    # fmt: off
    @tvm_tirx.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32")) -> R.Tensor((4, 9, 10, 3), "float32"):
            gv: R.Tensor((4, 9, 10, 3), "float32") = R.strided_slice(x, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), dtype="float32")) -> R.Tensor((4, 9, 10, 3), dtype="float32"):
            gv = R.call_tir(Expected.strided_slice, (x,), ty_args=[R.Tensor((4, 9, 10, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def strided_slice(rxplaceholder: T.Tensor((T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"), T_strided_slice_with_axes: T.Tensor((T.int64(4), T.int64(9), T.int64(10), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(9), T.int64(10), T.int64(3)):
                with Ts.sblock("T_strided_slice_with_axes"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax0 * T.int64(2) + T.int64(1), ax1, ax2, T.int64(8) - ax3 * T.int64(3)])
                    Ts.writes(T_strided_slice_with_axes[ax0, ax1, ax2, ax3])
                    T_strided_slice_with_axes[ax0, ax1, ax2, ax3] = rxplaceholder[ax0 * T.int64(2) + T.int64(1), ax1, ax2, T.int64(8) - ax3 * T.int64(3)]
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_no_strides():
    # fmt: off
    @tvm_tirx.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32")) :
            gv: R.Tensor((7, 9, 10, 2), "float32") = R.strided_slice(x, axes=[0, 1, 3], begin=[1, 0, 2], end=[8, 9, 4])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), dtype="float32")):
            gv = R.call_tir(Expected.strided_slice, (x,), ty_args=[R.Tensor((7, 9, 10, 2), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def strided_slice(rxplaceholder: T.Tensor((T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"), T_strided_slice_with_axes: T.Tensor((T.int64(7), T.int64(9), T.int64(10), T.int64(2)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(7), T.int64(9), T.int64(10), T.int64(2)):
                with Ts.sblock("T_strided_slice_with_axes"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax0 + T.int64(1), v_ax1, v_ax2, v_ax3 + T.int64(2)])
                    Ts.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0 + T.int64(1), v_ax1, v_ax2, v_ax3 + T.int64(2)]
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_negative_axes():
    # fmt: off
    @tvm_tirx.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10), "float32")) -> R.Tensor((8, 9, 3), "float32"):
            gv: R.Tensor((8, 9, 3), "float32") = R.strided_slice(x, axes=[-1], begin=[2], end=[5])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 9, 10), dtype="float32")) -> R.Tensor((8, 9, 3), dtype="float32"):
            gv = R.call_tir(Expected.strided_slice, (x,), ty_args=[R.Tensor((8, 9, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def strided_slice(rxplaceholder: T.Tensor((T.int64(8), T.int64(9), T.int64(10)), "float32"), T_strided_slice_with_axes: T.Tensor((T.int64(8), T.int64(9), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1, ax2 in T.grid(T.int64(8), T.int64(9), T.int64(3)):
                with Ts.sblock("T_strided_slice_with_axes"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1, v_ax2 + T.int64(2)])
                    Ts.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                    T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] = rxplaceholder[v_ax0, v_ax1, v_ax2 + T.int64(2)]
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_symbolic_sliced_axis():
    # fmt: off
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((2, n), "float32"):
            gv: R.Tensor((3, n), "float32") = R.strided_slice(x, axes=[0], begin=[1], end=[8], strides=[3], assume_inbound=True)
            return gv

    m_strided_slice = T.dynamic("m", "int64")
    n_strided_slice = T.dynamic("n", "int64")
    n_main = T.dynamic("n", "int64")
    m_main = T.dynamic("m", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def strided_slice(A: T.Tensor((m_strided_slice, n_strided_slice), "float32"), T_dynamic_strided_slice_with_axes: T.Tensor((T.int64(3), n_strided_slice), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(3), n_strided_slice):
                with Ts.sblock("T_dynamic_strided_slice_with_axes"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0 * T.int64(3) + T.int64(1), v_ax1])
                    Ts.writes(T_dynamic_strided_slice_with_axes[v_ax0, v_ax1])
                    T_dynamic_strided_slice_with_axes[v_ax0, v_ax1] = A[v_ax0 * T.int64(3) + T.int64(1), v_ax1]

        @R.function
        def main(x: R.Tensor((m_main, n_main), dtype="float32")) -> R.Tensor((3, n_main), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.strided_slice, (x,), ty_args=[R.Tensor((3, n_main), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((10, n), "float32")) -> R.Tensor((3, n), "float32"):
            gv: R.Tensor((3, n), "float32") = R.strided_slice(x, axes=[0], begin=[1], end=[8], strides=[3])
            return gv

    n_main = T.dynamic("n", "int64")
    n_strided_slice = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, n_main), dtype="float32")) -> R.Tensor((3, n_main), dtype="float32"):
            gv = R.call_tir(Expected.strided_slice, (x,), ty_args=[R.Tensor((3, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def strided_slice(rxplaceholder: T.Tensor([T.int64(10), n_strided_slice], dtype='float32'), T_strided_slice_with_axes: T.Tensor([T.int64(3), n_strided_slice], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(T.int64(3), n_strided_slice):
                with Ts.sblock("T_strided_slice_with_axes"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1])
                    Ts.writes(T_strided_slice_with_axes[ax0, ax1])
                    T_strided_slice_with_axes[ax0, ax1] = rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1]
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_symbolic_bound():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((10, n), "float32")) -> R.Tensor((3, n), "float32"):
            gv: R.Tensor((3, n), "float32") = R.strided_slice(x, axes=[0, 1], begin=[1, 0], end=[8, n], strides=[3, 1])
            return gv

    n_main = T.dynamic("n", "int64")
    n_strided_slice = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, n_main), dtype="float32")) -> R.Tensor((3, n_main), dtype="float32"):
            gv = R.call_tir(Expected.strided_slice, (x,), ty_args=[R.Tensor((3, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def strided_slice(rxplaceholder: T.Tensor([T.int64(10), n_strided_slice], dtype='float32'), T_strided_slice_with_axes: T.Tensor([T.int64(3), n_strided_slice], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(T.int64(3), n_strided_slice):
                with Ts.sblock("T_strided_slice_with_axes"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1])
                    Ts.writes(T_strided_slice_with_axes[ax0, ax1])
                    T_strided_slice_with_axes[ax0, ax1] = rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1]


def test_strided_slice_non_unit_stride():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((10, n), "float32")) -> R.Tensor((3, n), "float32"):
            gv: R.Tensor((3, n), "float32") = R.strided_slice(x, axes=[0, 1], begin=[1, 0], end=[8, n], strides=[3, 1])
            return gv

    n_main = T.dynamic("n", "int64")
    n_strided_slice = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, n_main), dtype="float32")) -> R.Tensor((3, n_main), dtype="float32"):
            gv = R.call_tir(Expected.strided_slice, (x,), ty_args=[R.Tensor((3, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def strided_slice(rxplaceholder: T.Tensor([T.int64(10), n_strided_slice], dtype='float32'), T_strided_slice_with_axes: T.Tensor([T.int64(3), n_strided_slice], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(T.int64(3), n_strided_slice):
                with Ts.sblock("T_strided_slice_with_axes"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1])
                    Ts.writes(T_strided_slice_with_axes[ax0, ax1])
                    T_strided_slice_with_axes[ax0, ax1] = rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1]


def test_dynamic_strided_slice():
    # fmt: off
    @tvm_tirx.script.ir_module
    class DynamicStridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32"), begin: R.Tensor((4,),"int64"), end: R.Tensor((4,),"int64"), strides: R.Tensor((4,),"int64")) -> R.Tensor("float32", ndim=4):
            gv: R.Tensor("float32", ndim=4) = R.dynamic_strided_slice(x, begin, end, strides)
            return gv
    s_dynamic_strided_slice = T.dynamic("s", "int64")
    s_1_dynamic_strided_slice = T.dynamic("s_1", "int64")
    s_2_dynamic_strided_slice = T.dynamic("s_2", "int64")
    s_3_dynamic_strided_slice = T.dynamic("s_3", "int64")
    s_main = T.dynamic("s", "int64")
    s_1_main = T.dynamic("s_1", "int64")
    s_2_main = T.dynamic("s_2", "int64")
    s_3_main = T.dynamic("s_3", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def dynamic_strided_slice(
            rxplaceholder: T.Tensor(
                (T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"
            ),
            rxplaceholder_1: T.Tensor((T.int64(4),), "int64"),
            rxplaceholder_2: T.Tensor((T.int64(4),), "int64"),
            rxplaceholder_3: T.Tensor((T.int64(4),), "int64"),
            T_strided_slice_dynamic: T.Tensor((s_dynamic_strided_slice, s_1_dynamic_strided_slice, s_2_dynamic_strided_slice, s_3_dynamic_strided_slice), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(s_dynamic_strided_slice, s_1_dynamic_strided_slice, s_2_dynamic_strided_slice, s_3_dynamic_strided_slice):
                with Ts.sblock("T_strided_slice_dynamic"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(
                        rxplaceholder[
                            T.int64(0) : T.int64(8),
                            T.int64(0) : T.int64(9),
                            T.int64(0) : T.int64(10),
                            T.int64(0) : T.int64(10),
                        ],
                        rxplaceholder_1[T.int64(0) : T.int64(4)],
                        rxplaceholder_3[T.int64(0) : T.int64(4)],
                    )
                    Ts.writes(T_strided_slice_dynamic[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_strided_slice_dynamic[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[
                        T.min(
                            T.max(
                                T.if_then_else(
                                    rxplaceholder_1[T.int64(0)] < T.int64(0),
                                    rxplaceholder_1[T.int64(0)] + T.int64(8),
                                    rxplaceholder_1[T.int64(0)],
                                ),
                                T.if_then_else(
                                    rxplaceholder_3[T.int64(0)] < T.int64(0), T.int64(-1), T.int64(0)
                                ),
                            ),
                            T.if_then_else(
                                rxplaceholder_3[T.int64(0)] < T.int64(0), T.int64(7), T.int64(8)
                            ),
                        )
                        + v_ax0 * rxplaceholder_3[T.int64(0)],
                        T.min(
                            T.max(
                                T.if_then_else(
                                    rxplaceholder_1[T.int64(1)] < T.int64(0),
                                    rxplaceholder_1[T.int64(1)] + T.int64(9),
                                    rxplaceholder_1[T.int64(1)],
                                ),
                                T.if_then_else(
                                    rxplaceholder_3[T.int64(1)] < T.int64(0), T.int64(-1), T.int64(0)
                                ),
                            ),
                            T.if_then_else(
                                rxplaceholder_3[T.int64(1)] < T.int64(0), T.int64(8), T.int64(9)
                            ),
                        )
                        + v_ax1 * rxplaceholder_3[T.int64(1)],
                        T.min(
                            T.max(
                                T.if_then_else(
                                    rxplaceholder_1[T.int64(2)] < T.int64(0),
                                    rxplaceholder_1[T.int64(2)] + T.int64(10),
                                    rxplaceholder_1[T.int64(2)],
                                ),
                                T.if_then_else(
                                    rxplaceholder_3[T.int64(2)] < T.int64(0), T.int64(-1), T.int64(0)
                                ),
                            ),
                            T.if_then_else(
                                rxplaceholder_3[T.int64(2)] < T.int64(0), T.int64(9), T.int64(10)
                            ),
                        )
                        + v_ax2 * rxplaceholder_3[T.int64(2)],
                        T.min(
                            T.max(
                                T.if_then_else(
                                    rxplaceholder_1[T.int64(3)] < T.int64(0),
                                    rxplaceholder_1[T.int64(3)] + T.int64(10),
                                    rxplaceholder_1[T.int64(3)],
                                ),
                                T.if_then_else(
                                    rxplaceholder_3[T.int64(3)] < T.int64(0), T.int64(-1), T.int64(0)
                                ),
                            ),
                            T.if_then_else(
                                rxplaceholder_3[T.int64(3)] < T.int64(0), T.int64(9), T.int64(10)
                            ),
                        )
                        + v_ax3 * rxplaceholder_3[T.int64(3)],
                    ]

        @Ts.function(private=True)
        def shape_func(
            rxplaceholder: T.Tensor(
                (T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"
            ),
            rxplaceholder_1: T.Tensor((T.int64(4),), "int64"),
            rxplaceholder_2: T.Tensor((T.int64(4),), "int64"),
            rxplaceholder_3: T.Tensor((T.int64(4),), "int64"),
            T_shape_func_strided_slice_dynamic: T.Tensor((T.int64(4),), "int64"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i_index in range(T.int64(4)):
                with Ts.sblock("T_shape_func_strided_slice_dynamic"):
                    v_i = Ts.axis.spatial(T.int64(4), i_index)
                    Ts.reads(
                        rxplaceholder_3[v_i], rxplaceholder_1[v_i], rxplaceholder_2[v_i]
                    )
                    Ts.writes(T_shape_func_strided_slice_dynamic[v_i])
                    T_shape_func_strided_slice_dynamic[v_i] = T.Select(
                        rxplaceholder_3[v_i] < T.int64(0),
                        (
                            T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_1[v_i] < T.int64(0),
                                        rxplaceholder_1[v_i]
                                        + T.Select(
                                            v_i == T.int64(3),
                                            T.int64(10),
                                            T.Select(
                                                v_i == T.int64(2),
                                                T.int64(10),
                                                T.Select(
                                                    v_i == T.int64(1),
                                                    T.int64(9),
                                                    T.Select(
                                                        v_i == T.int64(0),
                                                        T.int64(8),
                                                        T.int64(-1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rxplaceholder_1[v_i],
                                    ),
                                    T.int64(-1),
                                ),
                                T.Select(
                                    v_i == T.int64(3),
                                    T.int64(10),
                                    T.Select(
                                        v_i == T.int64(2),
                                        T.int64(10),
                                        T.Select(
                                            v_i == T.int64(1),
                                            T.int64(9),
                                            T.Select(
                                                v_i == T.int64(0), T.int64(8), T.int64(-1)
                                            ),
                                        ),
                                    ),
                                )
                                - T.int64(1),
                            )
                            - T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_2[v_i] < T.int64(0),
                                        rxplaceholder_2[v_i]
                                        + T.Select(
                                            v_i == T.int64(3),
                                            T.int64(10),
                                            T.Select(
                                                v_i == T.int64(2),
                                                T.int64(10),
                                                T.Select(
                                                    v_i == T.int64(1),
                                                    T.int64(9),
                                                    T.Select(
                                                        v_i == T.int64(0),
                                                        T.int64(8),
                                                        T.int64(-1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rxplaceholder_2[v_i],
                                    ),
                                    T.int64(-1),
                                ),
                                T.Select(
                                    v_i == T.int64(3),
                                    T.int64(10),
                                    T.Select(
                                        v_i == T.int64(2),
                                        T.int64(10),
                                        T.Select(
                                            v_i == T.int64(1),
                                            T.int64(9),
                                            T.Select(
                                                v_i == T.int64(0), T.int64(8), T.int64(-1)
                                            ),
                                        ),
                                    ),
                                )
                                - T.int64(1),
                            )
                            - rxplaceholder_3[v_i]
                            - T.int64(1)
                        )
                        // (rxplaceholder_3[v_i] * T.int64(-1)),
                        (
                            T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_2[v_i] < T.int64(0),
                                        rxplaceholder_2[v_i]
                                        + T.Select(
                                            v_i == T.int64(3),
                                            T.int64(10),
                                            T.Select(
                                                v_i == T.int64(2),
                                                T.int64(10),
                                                T.Select(
                                                    v_i == T.int64(1),
                                                    T.int64(9),
                                                    T.Select(
                                                        v_i == T.int64(0),
                                                        T.int64(8),
                                                        T.int64(-1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rxplaceholder_2[v_i],
                                    ),
                                    T.int64(0),
                                ),
                                T.Select(
                                    v_i == T.int64(3),
                                    T.int64(10),
                                    T.Select(
                                        v_i == T.int64(2),
                                        T.int64(10),
                                        T.Select(
                                            v_i == T.int64(1),
                                            T.int64(9),
                                            T.Select(
                                                v_i == T.int64(0), T.int64(8), T.int64(-1)
                                            ),
                                        ),
                                    ),
                                ),
                            )
                            + rxplaceholder_3[v_i]
                            - T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_1[v_i] < T.int64(0),
                                        rxplaceholder_1[v_i]
                                        + T.Select(
                                            v_i == T.int64(3),
                                            T.int64(10),
                                            T.Select(
                                                v_i == T.int64(2),
                                                T.int64(10),
                                                T.Select(
                                                    v_i == T.int64(1),
                                                    T.int64(9),
                                                    T.Select(
                                                        v_i == T.int64(0),
                                                        T.int64(8),
                                                        T.int64(-1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rxplaceholder_1[v_i],
                                    ),
                                    T.int64(0),
                                ),
                                T.Select(
                                    v_i == T.int64(3),
                                    T.int64(10),
                                    T.Select(
                                        v_i == T.int64(2),
                                        T.int64(10),
                                        T.Select(
                                            v_i == T.int64(1),
                                            T.int64(9),
                                            T.Select(
                                                v_i == T.int64(0), T.int64(8), T.int64(-1)
                                            ),
                                        ),
                                    ),
                                ),
                            )
                            - T.int64(1)
                        )
                        // rxplaceholder_3[v_i],
                    )

        @R.function
        def main(
            x: R.Tensor((8, 9, 10, 10), dtype="float32"),
            begin: R.Tensor((4,), dtype="int64"),
            end: R.Tensor((4,), dtype="int64"),
            strides: R.Tensor((4,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=4):
            gv = R.call_tir(
                Expected.shape_func,
                (x, begin, end, strides),
                ty_args=[R.Tensor((4,), dtype="int64")],
            )
            gv1: R.Shape(ndim=4) = R.tensor_to_shape(gv)
            gv2: R.Shape([s_main, s_1_main, s_2_main, s_3_main]) = R.match_cast(
                gv1, R.Shape([s_main, s_1_main, s_2_main, s_3_main])
            )
            gv_1 = R.call_tir(
                Expected.dynamic_strided_slice,
                (x, begin, end, strides),
                ty_args=[R.Tensor((s_main, s_1_main, s_2_main, s_3_main), dtype="float32")],
            )
            return gv_1
    # fmt: on
    mod = LegalizeOps()(DynamicStridedSlice)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_dynamic_strided_slice_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class DynamicStridedSlice:
        @R.function
        def main(x: R.Tensor((10, n), "float32"), begin:R.Tensor((2,), "int64"), end:R.Tensor((2,), "int64"), strides:R.Tensor((2,), "int64")) -> R.Tensor("float32", ndim=2):
            gv: R.Tensor("float32", ndim=2) = R.dynamic_strided_slice(x, begin, end, strides)
            return gv
    n_dynamic_strided_slice = T.dynamic("n", "int64")
    s_dynamic_strided_slice = T.dynamic("s", "int64")
    s_1_dynamic_strided_slice = T.dynamic("s_1", "int64")
    n_shape_func = T.dynamic("n", "int64")
    n_main = T.dynamic("n", "int64")
    s_main = T.dynamic("s", "int64")
    s_1_main = T.dynamic("s_1", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def dynamic_strided_slice(
            rxplaceholder_3: T.Tensor((T.int64(10), n_dynamic_strided_slice), "float32"),
            rxplaceholder: T.Tensor((T.int64(2),), "int64"),
            rxplaceholder_1: T.Tensor((T.int64(2),), "int64"),
            rxplaceholder_2: T.Tensor((T.int64(2),), "int64"),
            T_strided_slice_dynamic: T.Tensor((s_dynamic_strided_slice, s_1_dynamic_strided_slice), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(s_dynamic_strided_slice, s_1_dynamic_strided_slice):
                with Ts.sblock("T_strided_slice_dynamic"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(
                        rxplaceholder_3[
                            T.int64(0) : T.int64(10),
                            T.int64(0) : n_dynamic_strided_slice,
                        ],
                        rxplaceholder[T.int64(0) : T.int64(2)],
                        rxplaceholder_2[T.int64(0) : T.int64(2)],
                    )
                    Ts.writes(T_strided_slice_dynamic[v_ax0, v_ax1])
                    T_strided_slice_dynamic[v_ax0, v_ax1] = rxplaceholder_3[
                        T.min(
                            T.max(
                                T.if_then_else(
                                    rxplaceholder[T.int64(0)] < T.int64(0),
                                    rxplaceholder[T.int64(0)] + T.int64(10),
                                    rxplaceholder[T.int64(0)],
                                ),
                                T.if_then_else(
                                    rxplaceholder_2[T.int64(0)] < T.int64(0), T.int64(-1), T.int64(0)
                                ),
                            ),
                            T.if_then_else(
                                rxplaceholder_2[T.int64(0)] < T.int64(0), T.int64(9), T.int64(10)
                            ),
                        )
                        + v_ax0 * rxplaceholder_2[T.int64(0)],
                        T.min(
                            T.max(
                                T.if_then_else(
                                    rxplaceholder[T.int64(1)] < T.int64(0),
                                    rxplaceholder[T.int64(1)] + n_dynamic_strided_slice,
                                    rxplaceholder[T.int64(1)],
                                ),
                                T.if_then_else(
                                    rxplaceholder_2[T.int64(1)] < T.int64(0), T.int64(-1), T.int64(0)
                                ),
                            ),
                            T.if_then_else(
                                rxplaceholder_2[T.int64(1)] < T.int64(0), n_dynamic_strided_slice - T.int64(1), n_dynamic_strided_slice
                            ),
                        )
                        + v_ax1 * rxplaceholder_2[T.int64(1)],
                    ]

        @Ts.function(private=True)
        def shape_func(
            rxplaceholder_3: T.Tensor((T.int64(10), n_shape_func), "float32"),
            rxplaceholder: T.Tensor((T.int64(2),), "int64"),
            rxplaceholder_1: T.Tensor((T.int64(2),), "int64"),
            rxplaceholder_2: T.Tensor((T.int64(2),), "int64"),
            T_shape_func_strided_slice_dynamic: T.Tensor((T.int64(2),), "int64"),
        ):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for i_index in range(T.int64(2)):
                with Ts.sblock("T_shape_func_strided_slice_dynamic"):
                    v_i = Ts.axis.spatial(T.int64(2), i_index)
                    Ts.reads(rxplaceholder_2[v_i], rxplaceholder[v_i], rxplaceholder_1[v_i])
                    Ts.writes(T_shape_func_strided_slice_dynamic[v_i])
                    T_shape_func_strided_slice_dynamic[v_i] = T.Select(
                        rxplaceholder_2[v_i] < T.int64(0),
                        (
                            T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder[v_i] < T.int64(0),
                                        rxplaceholder[v_i]
                                        + T.Select(
                                            v_i == T.int64(1),
                                            n_shape_func,
                                            T.Select(
                                                v_i == T.int64(0), T.int64(10), T.int64(-1)
                                            ),
                                        ),
                                        rxplaceholder[v_i],
                                    ),
                                    T.int64(-1),
                                ),
                                T.Select(
                                    v_i == T.int64(1),
                                    n_shape_func,
                                    T.Select(v_i == T.int64(0), T.int64(10), T.int64(-1)),
                                )
                                - T.int64(1),
                            )
                            - T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_1[v_i] < T.int64(0),
                                        rxplaceholder_1[v_i]
                                        + T.Select(
                                            v_i == T.int64(1),
                                            n_shape_func,
                                            T.Select(
                                                v_i == T.int64(0), T.int64(10), T.int64(-1)
                                            ),
                                        ),
                                        rxplaceholder_1[v_i],
                                    ),
                                    T.int64(-1),
                                ),
                                T.Select(
                                    v_i == T.int64(1),
                                    n_shape_func,
                                    T.Select(v_i == T.int64(0), T.int64(10), T.int64(-1)),
                                )
                                - T.int64(1),
                            )
                            - rxplaceholder_2[v_i]
                            - T.int64(1)
                        )
                        // (rxplaceholder_2[v_i] * T.int64(-1)),
                        (
                            T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_1[v_i] < T.int64(0),
                                        rxplaceholder_1[v_i]
                                        + T.Select(
                                            v_i == T.int64(1),
                                            n_shape_func,
                                            T.Select(
                                                v_i == T.int64(0), T.int64(10), T.int64(-1)
                                            ),
                                        ),
                                        rxplaceholder_1[v_i],
                                    ),
                                    T.int64(0),
                                ),
                                T.Select(
                                    v_i == T.int64(1),
                                    n_shape_func,
                                    T.Select(v_i == T.int64(0), T.int64(10), T.int64(-1)),
                                ),
                            )
                            + rxplaceholder_2[v_i]
                            - T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder[v_i] < T.int64(0),
                                        rxplaceholder[v_i]
                                        + T.Select(
                                            v_i == T.int64(1),
                                            n_shape_func,
                                            T.Select(
                                                v_i == T.int64(0), T.int64(10), T.int64(-1)
                                            ),
                                        ),
                                        rxplaceholder[v_i],
                                    ),
                                    T.int64(0),
                                ),
                                T.Select(
                                    v_i == T.int64(1),
                                    n_shape_func,
                                    T.Select(v_i == T.int64(0), T.int64(10), T.int64(-1)),
                                ),
                            )
                            - T.int64(1)
                        )
                        // rxplaceholder_2[v_i],
                    )

        @R.function
        def main(
            x: R.Tensor((10, n_main), dtype="float32"),
            begin: R.Tensor((2,), dtype="int64"),
            end: R.Tensor((2,), dtype="int64"),
            strides: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=2):
            gv = R.call_tir(
                Expected.shape_func,
                (x, begin, end, strides),
                ty_args=[R.Tensor((2,), dtype="int64")],
            )
            gv1: R.Shape(ndim=2) = R.tensor_to_shape(gv)
            gv2: R.Shape([s_main, s_1_main]) = R.match_cast(gv1, R.Shape([s_main, s_1_main]))
            gv_1 = R.call_tir(
                Expected.dynamic_strided_slice,
                (x, begin, end, strides),
                ty_args=[R.Tensor((s_main, s_1_main), dtype="float32")],
            )
            return gv_1
    # fmt: on

    mod = LegalizeOps()(DynamicStridedSlice)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


##################### Linear algebra #####################


def test_matmul_1_4():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((4,), "float32"), y: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 3, 5), "float32"):
            gv: R.Tensor((2, 3, 5), "float32") = R.matmul(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4,), "float32"), y: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 3, 5), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), ty_args=[R.Tensor((2, 3, 5), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def matmul(rxplaceholder: T.Tensor(T.int64(4), "float32"), rxplaceholder_1: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), matmul: T.Tensor((T.int64(2), T.int64(3), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(5), T.int64(4)):
                with Ts.sblock("matmul"):
                    i0_1, i1_1, i2_1, k = Ts.axis.remap("SSSR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[k], rxplaceholder_1[i0_1, i1_1, k, i2_1])
                    Ts.writes(matmul[i0_1, i1_1, i2_1])
                    with Ts.init():
                        matmul[i0_1, i1_1, i2_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1] = matmul[i0_1, i1_1, i2_1] + rxplaceholder[k] * rxplaceholder_1[i0_1, i1_1, k, i2_1]
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_matmul_4_1():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv: R.Tensor((2, 3, 4), "float32") = R.matmul(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), ty_args=[R.Tensor((2, 3, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def matmul(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_1: T.Tensor(T.int64(5), "float32"), matmul: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("matmul"):
                    i0_1, i1_1, i2_1, k = Ts.axis.remap("SSSR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1, k], rxplaceholder_1[k])
                    Ts.writes(matmul[i0_1, i1_1, i2_1])
                    with Ts.init():
                        matmul[i0_1, i1_1, i2_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1] = matmul[i0_1, i1_1, i2_1] + rxplaceholder[i0_1, i1_1, i2_1, k] * rxplaceholder_1[k]
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_matmul_1_1():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.matmul(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")) -> R.Tensor((), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def matmul(rxplaceholder: T.Tensor(T.int64(4), "float32"), rxplaceholder_1: T.Tensor(T.int64(4), "float32"), matmul: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0 in T.serial(T.int64(4)):
                with Ts.sblock("matmul"):
                    k = Ts.axis.reduce(T.int64(4), i0)
                    Ts.reads(rxplaceholder[k], rxplaceholder_1[k])
                    Ts.writes(matmul[()])
                    with Ts.init():
                        matmul[()] = T.float32(0)
                    matmul[()] = matmul[()] + rxplaceholder[k] * rxplaceholder_1[k]
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_matmul_4_5():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float16"), y: R.Tensor((6, 2, 3, 5, 7), "float16")) -> R.Tensor((6, 2, 3, 4, 7), "float32"):
            gv: R.Tensor((6, 2, 3, 4, 7), "float32") = R.matmul(x, y, out_dtype="float32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float16"), y: R.Tensor((6, 2, 3, 5, 7), "float16")) -> R.Tensor((6, 2, 3, 4, 7), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), ty_args=[R.Tensor((6, 2, 3, 4, 7), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def matmul(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16"), rxplaceholder_1: T.Tensor((T.int64(6), T.int64(2), T.int64(3), T.int64(5), T.int64(7)), "float16"), matmul: T.Tensor((T.int64(6), T.int64(2), T.int64(3), T.int64(4), T.int64(7)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(6), T.int64(2), T.int64(3), T.int64(4), T.int64(7), T.int64(5)):
                with Ts.sblock("matmul"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, k = Ts.axis.remap("SSSSSR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[i1_1, i2_1, i3_1, k], rxplaceholder_1[i0_1, i1_1, i2_1, k, i4_1])
                    Ts.writes(matmul[i0_1, i1_1, i2_1, i3_1, i4_1])
                    with Ts.init():
                        matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = matmul[i0_1, i1_1, i2_1, i3_1, i4_1] + T.Cast("float32", rxplaceholder[i1_1, i2_1, i3_1, k]) * T.Cast("float32", rxplaceholder_1[i0_1, i1_1, i2_1, k, i4_1])
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_matmul_4_5_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    k = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((b, 1, m, k), "float32"), y: R.Tensor((a, 1, c, k, n), "float32")) -> R.Tensor((a, b, c, m, n), "float32"):
            gv: R.Tensor((a, b, c, m, n), "float32") = R.matmul(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    k_main = T.dynamic("k", "int64")
    a_matmul = T.dynamic("a", "int64")
    b_matmul = T.dynamic("b", "int64")
    c_matmul = T.dynamic("c", "int64")
    k_matmul = T.dynamic("k", "int64")
    m_matmul = T.dynamic("m", "int64")
    n_matmul = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((b_main, 1, m_main, k_main), "float32"), y: R.Tensor((a_main, 1, c_main, k_main, n_main), "float32")) -> R.Tensor((a_main, b_main, c_main, m_main, n_main), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def matmul(rxplaceholder: T.Tensor([b_matmul, T.int64(1), m_matmul, k_matmul], dtype='float32'), rxplaceholder_1: T.Tensor([a_matmul, T.int64(1), c_matmul, k_matmul, n_matmul], dtype='float32'), matmul: T.Tensor([a_matmul, b_matmul, c_matmul, m_matmul, n_matmul], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3, i4, i5 in T.grid(a_matmul, b_matmul, c_matmul, m_matmul, n_matmul, k_matmul):
                with Ts.sblock("matmul"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, k_1 = Ts.axis.remap("SSSSSR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[i1_1, T.int64(0), i3_1, k_1], rxplaceholder_1[i0_1, T.int64(0), i2_1, k_1, i4_1])
                    Ts.writes(matmul[i0_1, i1_1, i2_1, i3_1, i4_1])
                    with Ts.init():
                        matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = matmul[i0_1, i1_1, i2_1, i3_1, i4_1] + rxplaceholder[i1_1, T.int64(0), i3_1, k_1] * rxplaceholder_1[i0_1, T.int64(0), i2_1, k_1, i4_1]
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_matmul_batching_dim_1():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((1, 1, 4, 5), "float32"), y: R.Tensor((1, 1, 5, 7), "float32")) -> R.Tensor((1, 1, 4, 7), "float32"):
            gv: R.Tensor((1, 1, 4, 7), "float32") = R.matmul(x, y, out_dtype="float32")
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def matmul(A: T.Tensor((T.int64(1), T.int64(1), T.int64(4), T.int64(5)), "float32"), B: T.Tensor((T.int64(1), T.int64(1), T.int64(5), T.int64(7)), "float32"), matmul_1: T.Tensor((T.int64(1), T.int64(1), T.int64(4), T.int64(7)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1, i2, i3, k_index in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(7), T.int64(5)):
                with Ts.sblock("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = Ts.axis.remap("SSSSR", [i0, i1, i2, i3, k_index])
                    Ts.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    Ts.writes(matmul_1[v_i0, v_i1, v_i2, v_i3])
                    with Ts.init():
                        matmul_1[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    matmul_1[v_i0, v_i1, v_i2, v_i3] = matmul_1[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

        @R.function
        def main(x: R.Tensor((1, 1, 4, 5), dtype="float32"), y: R.Tensor((1, 1, 5, 7), dtype="float32")) -> R.Tensor((1, 1, 4, 7), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.matmul, (x, y), ty_args=[R.Tensor((1, 1, 4, 7), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_matmul_zero_k_no_reduction():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((2, 0), "float32"), y: R.Tensor((0, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.matmul(x, y)
            return gv
    # fmt: on

    mod = LegalizeOps()(Matmul)
    script = mod.script()
    assert "Ts.axis.reduce" not in script
    assert "T.float32(0)" in script or "T.float32(0.0)" in script


def test_einsum():
    # fmt: off
    @I.ir_module
    class Einsum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((3, 4), "float32")):
            gv = R.einsum((x, y), subscripts="ij,jk->ik")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((3, 4), dtype="float32")
        ) -> R.Tensor((2, 4), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.einsum, (x, y), ty_args=[R.Tensor((2, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def einsum(
            rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"),
            rxplaceholder_1: T.Tensor((T.int64(3), T.int64(4)), "float32"),
            T_einsum: T.Tensor((T.int64(2), T.int64(4)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1, j in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with Ts.sblock("T_einsum"):
                    v_ax0, v_ax1, v_j = Ts.axis.remap("SSR", [ax0, ax1, j])
                    Ts.reads(rxplaceholder[v_ax0, v_j], rxplaceholder_1[v_j, v_ax1])
                    Ts.writes(T_einsum[v_ax0, v_ax1])
                    with Ts.init():
                        T_einsum[v_ax0, v_ax1] = T.float32(0)
                    T_einsum[v_ax0, v_ax1] = (
                        T_einsum[v_ax0, v_ax1]
                        + rxplaceholder[v_ax0, v_j] * rxplaceholder_1[v_j, v_ax1]
                    )
    # fmt: on

    mod = LegalizeOps()(Einsum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_einsum_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @I.ir_module
    class Einsum:
        @R.function
        def main(x: R.Tensor((a, b), "float32"), y: R.Tensor((b, c), "float32")):
            gv = R.einsum((x, y), subscripts="ij,jk->ik")
            return gv

    a_main = T.dynamic("a", "int64")
    c_main = T.dynamic("c", "int64")
    b_main = T.dynamic("b", "int64")
    a_einsum = T.dynamic("a", "int64")
    b_einsum = T.dynamic("b", "int64")
    c_einsum = T.dynamic("c", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((a_main, b_main), dtype="float32"),
            y: R.Tensor((b_main, c_main), dtype="float32"),
        ) -> R.Tensor((a_main, c_main), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.einsum, (x, y), ty_args=[R.Tensor((a_main, c_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def einsum(
            rxplaceholder: T.Tensor((a_einsum, b_einsum), "float32"),
            rxplaceholder_1: T.Tensor((b_einsum, c_einsum), "float32"),
            T_einsum: T.Tensor((a_einsum, c_einsum), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})

            for ax0, ax1, j in T.grid(a_einsum, c_einsum, b_einsum):
                with Ts.sblock("T_einsum"):
                    v_ax0, v_ax1, v_j = Ts.axis.remap("SSR", [ax0, ax1, j])
                    Ts.reads(rxplaceholder[v_ax0, v_j], rxplaceholder_1[v_j, v_ax1])
                    Ts.writes(T_einsum[v_ax0, v_ax1])
                    with Ts.init():
                        T_einsum[v_ax0, v_ax1] = T.float32(0)
                    T_einsum[v_ax0, v_ax1] = (
                        T_einsum[v_ax0, v_ax1]
                        + rxplaceholder[v_ax0, v_j] * rxplaceholder_1[v_j, v_ax1]
                    )
    # fmt: on

    mod = LegalizeOps()(Einsum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_data_dependent_attribute():
    dynamic_strided_slice_op = Op.get("relax.dynamic_strided_slice")
    assert dynamic_strided_slice_op.get_attr("FDataDependent")

    strided_slice_op = Op.get("relax.strided_slice")
    assert strided_slice_op.get_attr("FDataDependent") is None


if __name__ == "__main__":
    tvm_tirx.testing.main()
