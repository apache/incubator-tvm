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
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

##################### Search #####################


def test_where():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Where:
        @R.function
        def main(condition: R.Tensor((3, 2, 1), "bool"), x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")) -> R.Tensor((3, 2, 3), "float32"):
            gv: R.Tensor((3, 2, 3), "float32") = R.where(condition, x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(condition: R.Tensor((3, 2, 1), "bool"), x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 1), "float32")) -> R.Tensor((3, 2, 3), "float32"):
            gv = R.call_tir(Expected.where, (condition, x, y), ty_args=[R.Tensor((3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def where(rxplaceholder: T.Tensor((T.int64(3), T.int64(2), T.int64(1)), "bool"), rxplaceholder_1: T.Tensor((T.int64(2), T.int64(3)), "float32"), rxplaceholder_2: T.Tensor((T.int64(2), T.int64(1)), "float32"), T_where: T.Tensor((T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_where"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1, T.int64(0)], rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, T.int64(0)])
                    Ts.writes(T_where[ax0, ax1, ax2])
                    T_where[ax0, ax1, ax2] = T.Select(T.int64(0) < T.Cast("int64", rxplaceholder[ax0, ax1, T.int64(0)]), rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Where)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_where_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Where:
        @R.function
        def main(condition: R.Tensor((a, b, 1), "bool"), x: R.Tensor((b, c), "float32"), y: R.Tensor((b, 1), "float32")) -> R.Tensor((a, b, c), "float32"):
            gv: R.Tensor((a, b, c), "float32") = R.where(condition, x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_where = T.dynamic("a", "int64")
    b_where = T.dynamic("b", "int64")
    c_where = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(condition: R.Tensor((a_main, b_main, 1), "bool"), x: R.Tensor((b_main, c_main), "float32"), y: R.Tensor((b_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main), "float32"):
            gv = R.call_tir(Expected.where, (condition, x, y), ty_args=[R.Tensor((a_main, b_main, c_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def where(rxplaceholder: T.Tensor([a_where, b_where, T.int64(1)], dtype='bool'), rxplaceholder_1: T.Tensor([b_where, c_where], dtype='float32'), rxplaceholder_2: T.Tensor([b_where, T.int64(1)], dtype='float32'), T_where: T.Tensor([a_where, b_where, c_where], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2 in T.grid(a_where, b_where, c_where):
                with Ts.sblock("T_where"):
                    ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[ax0, ax1, T.int64(0)], rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, T.int64(0)])
                    Ts.writes(T_where[ax0, ax1, ax2])
                    T_where[ax0, ax1, ax2] = T.Select(T.int64(0) < T.Cast("int64", rxplaceholder[ax0, ax1, T.int64(0)]), rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Where)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_argmax():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Argmax:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 4, 5), "int64"):
            gv: R.Tensor((2, 4, 5), "int64") = R.argmax(x, axis=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((2, 4, 5), dtype="int64"):
            gv = R.call_tir(Expected.argmax, (x,), ty_args=[R.Tensor((2, 4, 5), dtype="int64")])
            return gv

        @Ts.function(private=True)
        def argmax(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_red: T.Tensor((T.int64(2), T.int64(4), T.int64(5)), "int64")):
            T.func_attr({"tirx.noalias": True})
            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(4), T.int64(5)), "int64")
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor((T.int64(2), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, k1 in T.grid(T.int64(2), T.int64(4), T.int64(5), T.int64(3)):
                with Ts.sblock("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_ax2, v_k1 = Ts.axis.remap("SSSR", [ax0, ax1, ax2, k1])
                    Ts.reads(rxplaceholder[v_ax0, v_k1, v_ax1, v_ax2])
                    Ts.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2], rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2] = T.int64(-1)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2] = T.min_value(ty="float32")
                    v_rxplaceholder_red_temp_v0: T.let[T.int64] = T.Select(rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2] > rxplaceholder[v_ax0, v_k1, v_ax1, v_ax2] or (rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2] == rxplaceholder[v_ax0, v_k1, v_ax1, v_ax2] and rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2] < v_k1), rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2], v_k1)
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = T.Select(rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2] > rxplaceholder[v_ax0, v_k1, v_ax1, v_ax2], rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2], rxplaceholder[v_ax0, v_k1, v_ax1, v_ax2])
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2] = v_rxplaceholder_red_temp_v1
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with Ts.sblock("rxplaceholder_red"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2])
                    Ts.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2])
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2] = rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2]
    # fmt: on

    mod = LegalizeOps()(Argmax)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_argmax_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")
    b = T.dynamic("b", "int64")

    @tvm_tirx.script.ir_module
    class Argmax:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((a, 1, c, d), "int64"):
            gv: R.Tensor((a, 1, c, d), "int64") = R.argmax(x, axis=1, keepdims=True)
            return gv

    a_main = T.dynamic("a", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    b_main = T.dynamic("b", "int64")
    a_argmax = T.dynamic("a", "int64")
    b_argmax = T.dynamic("b", "int64")
    c_argmax = T.dynamic("c", "int64")
    d_argmax = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")) -> R.Tensor((a_main, 1, c_main, d_main), dtype="int64"):
            gv = R.call_tir(Expected.argmax, (x,), ty_args=[R.Tensor((a_main, 1, c_main, d_main), dtype="int64")])
            return gv

        @Ts.function(private=True)
        def argmax(rxplaceholder: T.Tensor((a_argmax, b_argmax, c_argmax, d_argmax), "float32"), rxplaceholder_red: T.Tensor((a_argmax, T.int64(1), c_argmax, d_argmax), 'int64')):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor((a_argmax, T.int64(1), c_argmax, d_argmax), "int64")
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor((a_argmax, T.int64(1), c_argmax, d_argmax))
            for ax0, ax1, ax2, ax3, k1 in T.grid(a_argmax, T.int64(1), c_argmax, d_argmax, b_argmax):
                with Ts.sblock("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_k1 = Ts.axis.remap("SSSSR", [ax0, ax1, ax2, ax3, k1])
                    Ts.reads(rxplaceholder[v_ax0, v_k1, v_ax2, v_ax3])
                    Ts.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3], rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3] = T.int64(-1)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] = T.min_value(ty="float32")
                    v_rxplaceholder_red_temp_v0: T.let[T.int64] = T.Select(rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] > rxplaceholder[v_ax0, v_k1, v_ax2, v_ax3] or (rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] == rxplaceholder[v_ax0, v_k1, v_ax2, v_ax3] and rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3] < v_k1), rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3], v_k1)
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = T.Select(rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] > rxplaceholder[v_ax0, v_k1, v_ax2, v_ax3], rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3], rxplaceholder[v_ax0, v_k1, v_ax2, v_ax3])
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] = v_rxplaceholder_red_temp_v1
            for ax0, ax1, ax2, ax3 in T.grid(a_argmax, T.int64(1), c_argmax, d_argmax):
                with Ts.sblock("rxplaceholder_red"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3])
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3]
    # fmt: on

    mod = LegalizeOps()(Argmax)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_argmin():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Argmin:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((), "int64"):
            gv: R.Tensor((), "int64") = R.argmin(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def argmin(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_red: T.Tensor((), "int64")):
            T.func_attr({"tirx.noalias": True})
            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor((), "int64")
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor(())
            for k0, k1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("rxplaceholder_red_temp"):
                    v_k0, v_k1, v_k2, v_k3 = Ts.axis.remap("RRRR", [k0, k1, k2, k3])
                    Ts.reads(rxplaceholder[v_k0, v_k1, v_k2, v_k3])
                    Ts.writes(rxplaceholder_red_temp_v0[()], rxplaceholder_red_temp_v1[()])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[()] = T.int64(-1)
                        rxplaceholder_red_temp_v1[()] = T.max_value(ty="float32")
                    v_rxplaceholder_red_temp_v0: T.let[T.int64] = T.Select(rxplaceholder_red_temp_v1[()] < rxplaceholder[v_k0, v_k1, v_k2, v_k3] or (rxplaceholder_red_temp_v1[()] == rxplaceholder[v_k0, v_k1, v_k2, v_k3] and rxplaceholder_red_temp_v0[()] < v_k0 * T.int64(60) + v_k1 * T.int64(20) + v_k2 * T.int64(5) + v_k3), rxplaceholder_red_temp_v0[()], v_k0 * T.int64(60) + v_k1 * T.int64(20) + v_k2 * T.int64(5) + v_k3)
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = T.Select(rxplaceholder_red_temp_v1[()] < rxplaceholder[v_k0, v_k1, v_k2, v_k3], rxplaceholder_red_temp_v1[()], rxplaceholder[v_k0, v_k1, v_k2, v_k3])
                    rxplaceholder_red_temp_v0[()] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[()] = v_rxplaceholder_red_temp_v1
            with Ts.sblock("rxplaceholder_red"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(rxplaceholder_red_temp_v0[()])
                Ts.writes(rxplaceholder_red[()])
                rxplaceholder_red[()] = rxplaceholder_red_temp_v0[()]

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((), dtype="int64"):
            gv = R.call_tir(Expected.argmin, (x,), ty_args=[R.Tensor((), dtype="int64")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Argmin)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_argmin_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Argmin:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((1, 1, 1, 1), "int64"):
            gv: R.Tensor((1, 1, 1, 1), "int64") = R.argmin(x, keepdims=True)
            return gv

    a_argmin = T.dynamic("a", "int64")
    b_argmin = T.dynamic("b", "int64")
    c_argmin = T.dynamic("c", "int64")
    d_argmin = T.dynamic("d", "int64")
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def argmin(rxplaceholder: T.Tensor((a_argmin, b_argmin, c_argmin, d_argmin), "float32"), rxplaceholder_red: T.Tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)), "int64")):
            T.func_attr({"tirx.noalias": True})

            rxplaceholder_red_temp_v0 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)), "int64")
            rxplaceholder_red_temp_v1 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)))
            for ax0, ax1, ax2, ax3, k0, k1, k2, k3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), a_argmin, b_argmin, c_argmin, d_argmin):
                with Ts.sblock("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_k0, v_k1, v_k2, v_k3 = Ts.axis.remap("SSSSRRRR", [ax0, ax1, ax2, ax3, k0, k1, k2, k3])
                    Ts.reads(rxplaceholder[v_k0, v_k1, v_k2, v_k3])
                    Ts.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3], rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3] = T.int64(-1)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] = T.max_value(ty="float32")
                    v_rxplaceholder_red_temp_v0: T.let[T.int64] = T.Select(rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] < rxplaceholder[v_k0, v_k1, v_k2, v_k3] or (rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] == rxplaceholder[v_k0, v_k1, v_k2, v_k3] and rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3] < ((v_k0 * b_argmin + v_k1) * c_argmin + v_k2) * d_argmin + v_k3), rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3], ((v_k0 * b_argmin + v_k1) * c_argmin + v_k2) * d_argmin + v_k3)
                    v_rxplaceholder_red_temp_v1: T.let[T.float32] = T.Select(rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] < rxplaceholder[v_k0, v_k1, v_k2, v_k3], rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3], rxplaceholder[v_k0, v_k1, v_k2, v_k3])
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1, v_ax2, v_ax3] = v_rxplaceholder_red_temp_v1
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                with Ts.sblock("rxplaceholder_red"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3])
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_red_temp_v0[v_ax0, v_ax1, v_ax2, v_ax3]

        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")) -> R.Tensor((1, 1, 1, 1), dtype="int64"):
            gv = R.call_tir(Expected.argmin, (x,), ty_args=[R.Tensor((1, 1, 1, 1), dtype="int64")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Argmin)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


##################### Statistical #####################


def test_max():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Max:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 5), "float32"):
            gv: R.Tensor((2, 5), "float32") = R.max(x, axis=[1, 2])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 5), "float32"):
            gv = R.call_tir(Expected.max, (x,), ty_args=[R.Tensor((2, 5), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def max(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_red: T.Tensor((T.int64(2), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(5), T.int64(3), T.int64(4)):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, k1, k2 = Ts.axis.remap("SSRR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax0, k1, k2, ax1])
                    Ts.writes(rxplaceholder_red[ax0, ax1])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1] = T.min_value(ty="float32")
                    rxplaceholder_red[ax0, ax1] = T.max(rxplaceholder_red[ax0, ax1], rxplaceholder[ax0, k1, k2, ax1])
    # fmt: on

    mod = LegalizeOps()(Max)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_max_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    d = T.dynamic("d", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Max:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((a, d), "float32"):
            gv: R.Tensor((a, d), "float32") = R.max(x, axis=[1, 2])
            return gv

    a_main = T.dynamic("a", "int64")
    d_main = T.dynamic("d", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_max = T.dynamic("a", "int64")
    b_max = T.dynamic("b", "int64")
    c_max = T.dynamic("c", "int64")
    d_max = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), "float32")) -> R.Tensor((a_main, d_main), "float32"):
            gv = R.call_tir(Expected.max, (x,), ty_args=[R.Tensor((a_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def max(rxplaceholder: T.Tensor([a_max, b_max, c_max, d_max], dtype='float32'), rxplaceholder_red: T.Tensor([a_max, d_max], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_max, d_max, b_max, c_max):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, k1, k2 = Ts.axis.remap("SSRR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax0, k1, k2, ax1])
                    Ts.writes(rxplaceholder_red[ax0, ax1])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1] = T.min_value(ty="float32")
                    rxplaceholder_red[ax0, ax1] = T.max(rxplaceholder_red[ax0, ax1], rxplaceholder[ax0, k1, k2, ax1])
    # fmt: on

    mod = LegalizeOps()(Max)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_min():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Min:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 1, 1, 5), "float32"):
            gv: R.Tensor((2, 1, 1, 5), "float32") = R.min(x, axis=[1, 2], keepdims=True)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 1, 1, 5), "float32"):
            gv = R.call_tir(Expected.min, (x,), ty_args=[R.Tensor((2, 1, 1, 5), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def min(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_red: T.Tensor((T.int64(2), T.int64(1), T.int64(1), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(5), T.int64(3), T.int64(4)):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k1, k2 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[ax0, k1, k2, ax3])
                    Ts.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.max_value(ty="float32")
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = T.min(rxplaceholder_red[ax0, ax1, ax2, ax3], rxplaceholder[ax0, k1, k2, ax3])
    # fmt: on

    mod = LegalizeOps()(Min)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_min_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    d = T.dynamic("d", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")

    @tvm_tirx.script.ir_module
    class Min:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((a, 1, 1, d), "float32"):
            gv: R.Tensor((a, 1, 1, d), "float32") = R.min(x, axis=[1, 2], keepdims=True)
            return gv

    a_main = T.dynamic("a", "int64")
    d_main = T.dynamic("d", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_min = T.dynamic("a", "int64")
    b_min = T.dynamic("b", "int64")
    c_min = T.dynamic("c", "int64")
    d_min = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), "float32")) -> R.Tensor((a_main, 1, 1, d_main), "float32"):
            gv = R.call_tir(Expected.min, (x,), ty_args=[R.Tensor((a_main, 1, 1, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def min(rxplaceholder: T.Tensor([a_min, b_min, c_min, d_min], dtype='float32'), rxplaceholder_red: T.Tensor([a_min, T.int64(1), T.int64(1), d_min], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3, i4, i5 in T.grid(a_min, T.int64(1), T.int64(1), d_min, b_min, c_min):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k1, k2 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[ax0, k1, k2, ax3])
                    Ts.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.max_value(ty="float32")
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = T.min(rxplaceholder_red[ax0, ax1, ax2, ax3], rxplaceholder[ax0, k1, k2, ax3])
    # fmt: on

    mod = LegalizeOps()(Min)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_sum():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Sum:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.sum(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((), "float32"):
            gv = R.call_tir(Expected.sum, (x,), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def sum(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_red: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("rxplaceholder_red"):
                    k0, k1, k2, k3 = Ts.axis.remap("RRRR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[k0, k1, k2, k3])
                    Ts.writes(rxplaceholder_red[()])
                    with Ts.init():
                        rxplaceholder_red[()] = T.float32(0)
                    rxplaceholder_red[()] = rxplaceholder_red[()] + rxplaceholder[k0, k1, k2, k3]
    # fmt: on

    mod = LegalizeOps()(Sum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_sum_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Sum:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.sum(x)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_sum = T.dynamic("a", "int64")
    b_sum = T.dynamic("b", "int64")
    c_sum = T.dynamic("c", "int64")
    d_sum = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), "float32")) -> R.Tensor((), "float32"):
            gv = R.call_tir(Expected.sum, (x,), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def sum(rxplaceholder: T.Tensor([a_sum, b_sum, c_sum, d_sum], dtype='float32'), rxplaceholder_red: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_sum, b_sum, c_sum, d_sum):
                with Ts.sblock("rxplaceholder_red"):
                    k0, k1, k2, k3 = Ts.axis.remap("RRRR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[k0, k1, k2, k3])
                    Ts.writes(rxplaceholder_red[()])
                    with Ts.init():
                        rxplaceholder_red[()] = T.float32(0)
                    rxplaceholder_red[()] = rxplaceholder_red[()] + rxplaceholder[k0, k1, k2, k3]
    # fmt: on

    mod = LegalizeOps()(Sum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_prod():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Prod:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv: R.Tensor((1, 1, 1, 1), "float32") = R.prod(x, keepdims=True)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv = R.call_tir(Expected.prod, (x,), ty_args=[R.Tensor((1, 1, 1, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def prod(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_red: T.Tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k1, k2, k3 = Ts.axis.remap("SSSSRRRR", [i0, i1, i2, i3, i4, i5, i6, i7])
                    Ts.reads(rxplaceholder[k0, k1, k2, k3])
                    Ts.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(1)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] * rxplaceholder[k0, k1, k2, k3]
    # fmt: on

    mod = LegalizeOps()(Prod)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_prod_bool():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Prod:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "bool")) -> R.Tensor((1, 1, 1, 1), "bool"):
            gv: R.Tensor((1, 1, 1, 1), "bool") = R.prod(x, keepdims=True)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "bool")) -> R.Tensor((1, 1, 1, 1), "bool"):
            gv = R.call_tir(Expected.prod, (x,), ty_args=[R.Tensor((1, 1, 1, 1), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def prod(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "bool"), rxplaceholder_red: T.Tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k1, k2, k3 = Ts.axis.remap("SSSSRRRR", [i0, i1, i2, i3, i4, i5, i6, i7])
                    Ts.reads(rxplaceholder[k0, k1, k2, k3])
                    Ts.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.bool(1)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] and rxplaceholder[k0, k1, k2, k3]
    # fmt: on

    mod = LegalizeOps()(Prod)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_prod_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Prod:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv: R.Tensor((1, 1, 1, 1), "float32") = R.prod(x, keepdims=True)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_prod = T.dynamic("a", "int64")
    b_prod = T.dynamic("b", "int64")
    c_prod = T.dynamic("c", "int64")
    d_prod = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), "float32")) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv = R.call_tir(Expected.prod, (x,), ty_args=[R.Tensor((1, 1, 1, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def prod(rxplaceholder: T.Tensor([a_prod, b_prod, c_prod, d_prod], dtype='float32'), rxplaceholder_red: T.Tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)), "float32")):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), a_prod, b_prod, c_prod, d_prod):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k1, k2, k3 = Ts.axis.remap("SSSSRRRR", [i0, i1, i2, i3, i4, i5, i6, i7])
                    Ts.reads(rxplaceholder[k0, k1, k2, k3])
                    Ts.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(1)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] * rxplaceholder[k0, k1, k2, k3]
    # fmt: on

    mod = LegalizeOps()(Prod)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_sum_zero_dim_axis_identity():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Sum:
        @R.function
        def main(x: R.Tensor((2, 0, 4), "float32")) -> R.Tensor((2, 4), "float32"):
            gv: R.Tensor((2, 4), "float32") = R.sum(x, axis=[1], keepdims=False)
            return gv
    # fmt: on

    mod = LegalizeOps()(Sum)
    script = mod.script()
    assert "Ts.axis.reduce" not in script
    assert "T.float32(0)" in script or "T.float32(0.0)" in script


def test_sum_zero_dim_negative_axis_identity():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Sum:
        @R.function
        def main(x: R.Tensor((2, 3, 0), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.sum(x, axis=[-1], keepdims=False)
            return gv
    # fmt: on

    mod = LegalizeOps()(Sum)
    script = mod.script()
    assert "Ts.axis.reduce" not in script
    assert "T.float32(0)" in script or "T.float32(0.0)" in script


def test_prod_zero_dim_axis_identity():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Prod:
        @R.function
        def main(x: R.Tensor((2, 0, 4), "float32")) -> R.Tensor((2, 4), "float32"):
            gv: R.Tensor((2, 4), "float32") = R.prod(x, axis=[1], keepdims=False)
            return gv
    # fmt: on

    mod = LegalizeOps()(Prod)
    script = mod.script()
    assert "Ts.axis.reduce" not in script
    assert "T.float32(1)" in script or "T.float32(1.0)" in script


def test_prod_bool_zero_dim_axis_identity():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Prod:
        @R.function
        def main(x: R.Tensor((2, 0, 4), "bool")) -> R.Tensor((2, 4), "bool"):
            gv: R.Tensor((2, 4), "bool") = R.prod(x, axis=[1], keepdims=False)
            return gv
    # fmt: on

    mod = LegalizeOps()(Prod)
    script = mod.script()
    assert "Ts.axis.reduce" not in script
    assert "T.bool(1)" in script or "T.bool(True)" in script


def test_mean():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Mean:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((3, 4), "float32"):
            gv: R.Tensor((3, 4), "float32") = R.mean(x, [0, 3])
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((3, 4), "float32"):
            gv = R.call_tir(Expected.mean, (x,), ty_args=[R.Tensor((3, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def mean(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), T_divide: T.Tensor((T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            rxplaceholder_red = Ts.sblock_alloc_tensor([T.int64(3), T.int64(4)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(3), T.int64(4), T.int64(2), T.int64(5)):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, k0, k3 = Ts.axis.remap("SSRR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[k0, ax0, ax1, k3])
                    Ts.writes(rxplaceholder_red[ax0, ax1])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = rxplaceholder_red[ax0, ax1] + rxplaceholder[k0, ax0, ax1, k3]
            for i0, i1 in T.grid(T.int64(3), T.int64(4)):
                with Ts.sblock("T_divide"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder_red[ax0, ax1])
                    Ts.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder_red[ax0, ax1] / T.float32(10)
    # fmt: on

    mod = LegalizeOps()(Mean)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_mean_symbolic():
    # fmt: off
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    a = T.dynamic("a", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Mean:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((b, c), "float32"):
            gv: R.Tensor((b, c), "float32") = R.mean(x, [0, 3])
            return gv

    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_main = T.dynamic("a", "int64")
    d_main = T.dynamic("d", "int64")
    a_mean = T.dynamic("a", "int64")
    b_mean = T.dynamic("b", "int64")
    c_mean = T.dynamic("c", "int64")
    d_mean = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")) -> R.Tensor((b_main, c_main), dtype="float32"):
            gv = R.call_tir(Expected.mean, (x,), ty_args=[R.Tensor((b_main, c_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def mean(rxplaceholder: T.Tensor([a_mean, b_mean, c_mean, d_mean], dtype='float32'), T_divide: T.Tensor([b_mean, c_mean], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            rxplaceholder_red = Ts.sblock_alloc_tensor([b_mean, c_mean], dtype="float32")
            for i0, i1, i2, i3 in T.grid(b_mean, c_mean, a_mean, d_mean):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, k0, k3 = Ts.axis.remap("SSRR", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[k0, ax0, ax1, k3])
                    Ts.writes(rxplaceholder_red[ax0, ax1])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = rxplaceholder_red[ax0, ax1] + rxplaceholder[k0, ax0, ax1, k3]
            for i0, i1 in T.grid(b_mean, c_mean):
                with Ts.sblock("T_divide"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder_red[ax0, ax1])
                    Ts.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder_red[ax0, ax1] / T.Cast("float32", a_mean * d_mean)
    # fmt: on

    mod = LegalizeOps()(Mean)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_median():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Median:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tuple(R.Tensor((3, 4, 5), dtype="float32"), R.Tensor((3, 4, 5), dtype="int64")):
            gv: R.Tuple(R.Tensor((3, 4, 5), dtype="float32"), R.Tensor((3, 4, 5), dtype="int64")) = R.median(x, axis=[0], keepdims=False)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tuple(R.Tensor((3, 4, 5), dtype="float32"), R.Tensor((3, 4, 5), dtype="int64")):
            gv = R.call_tir(Expected.median, (x,), ty_args=[_TupleType([R.Tensor((3, 4, 5), dtype="float32"), R.Tensor((3, 4, 5), dtype="int64")])])
            return gv

        @Ts.function(private=True)
        def median(data_buf: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32", align=8), T_squeeze: T.Tensor((T.int64(3), T.int64(4), T.int64(5)), "float32"), T_squeeze_1: T.Tensor((T.int64(3), T.int64(4), T.int64(5)), "int64")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            T_full = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(4), T.int64(5)), "int64")
            out_buf = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "int64", align=8)
            T_gather = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_gather_1 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(4), T.int64(5)))
            T_gather_2 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(4), T.int64(5)), "int64")
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_full"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads()
                    Ts.writes(T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_full[v_ax0, v_ax1, v_ax2, v_ax3] = 0
            with Ts.sblock("argsort_cpu"):
                Ts.reads()
                Ts.writes()
                T.call_packed("tvm.contrib.sort.argsort", T.stack_make_dltensor(data_buf.data,
                                                                                 T.stack_make_shape(T.int64(2), T.int64(3), T.int64(4), T.int64(5)),
                                                                                 0, 4, T.float32(0.0), T.int64(0)),
                                                          T.stack_make_dltensor(out_buf.data,
                                                                                 T.stack_make_shape(T.int64(2), T.int64(3), T.int64(4), T.int64(5)),
                                                                                 0, 4, T.int64(0), T.int64(0)),
                                                          0, T.bool(True))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_gather"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(data_buf[out_buf[v_ax0, v_ax1, v_ax2, v_ax3], v_ax1, v_ax2, v_ax3], out_buf[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_gather[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_gather[v_ax0, v_ax1, v_ax2, v_ax3] = data_buf[out_buf[v_ax0, v_ax1, v_ax2, v_ax3], v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_gather_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_gather[T_full[v_ax0, v_ax1, v_ax2, v_ax3], v_ax1, v_ax2, v_ax3], T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_gather_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_gather_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_gather[T_full[v_ax0, v_ax1, v_ax2, v_ax3], v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_squeeze"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_gather_1[T.int64(0), v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
                    T_squeeze[v_ax0, v_ax1, v_ax2] = T_gather_1[T.int64(0), v_ax0, v_ax1, v_ax2]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_gather_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(out_buf[T_full[v_ax0, v_ax1, v_ax2, v_ax3], v_ax1, v_ax2, v_ax3], T_full[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_gather_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_gather_2[v_ax0, v_ax1, v_ax2, v_ax3] = out_buf[T_full[v_ax0, v_ax1, v_ax2, v_ax3], v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_squeeze_1"):
                    v_ax0, v_ax1, v_ax2 = Ts.axis.remap("SSS", [ax0, ax1, ax2])
                    Ts.reads(T_gather_2[T.int64(0), v_ax0, v_ax1, v_ax2])
                    Ts.writes(T_squeeze_1[v_ax0, v_ax1, v_ax2])
                    T_squeeze_1[v_ax0, v_ax1, v_ax2] = T_gather_2[T.int64(0), v_ax0, v_ax1, v_ax2]
    # fmt: on

    mod = LegalizeOps()(Median)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_std():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Std:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.std(x)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def std(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), compute: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            rxplaceholder_red = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)))
            T_divide = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)))
            T_subtract = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply_red = Ts.sblock_alloc_tensor(())
            T_divide_1 = Ts.sblock_alloc_tensor(())
            for ax0, ax1, ax2, ax3, k0, k1, k2, k3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("rxplaceholder_red"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_k0, v_k1, v_k2, v_k3 = Ts.axis.remap("SSSSRRRR", [ax0, ax1, ax2, ax3, k0, k1, k2, k3])
                    Ts.reads(rxplaceholder[v_k0, v_k1, v_k2, v_k3])
                    Ts.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] + rxplaceholder[v_k0, v_k1, v_k2, v_k3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                with Ts.sblock("T_divide"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] / T.float32(120.0)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_subtract"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_divide[T.int64(0), T.int64(0), T.int64(0), T.int64(0)])
                    Ts.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_divide[T.int64(0), T.int64(0), T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract[v_ax0, v_ax1, v_ax2, v_ax3]
            for k0, k1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply_red"):
                    v_k0, v_k1, v_k2, v_k3 = Ts.axis.remap("RRRR", [k0, k1, k2, k3])
                    Ts.reads(T_multiply[v_k0, v_k1, v_k2, v_k3])
                    Ts.writes(T_multiply_red[()])
                    with Ts.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply[v_k0, v_k1, v_k2, v_k3]
            with Ts.sblock("T_divide_1"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_multiply_red[()])
                Ts.writes(T_divide_1[()])
                T_divide_1[()] = T_multiply_red[()] / T.float32(120.0)
            with Ts.sblock("compute"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_divide_1[()])
                Ts.writes(compute[()])
                compute[()] = T.sqrt(T_divide_1[()])

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.std, (x,), ty_args=[R.Tensor((), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Std)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_std_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Std:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.std(x)
            return gv

    a_std = T.dynamic("a", "int64")
    b_std = T.dynamic("b", "int64")
    c_std = T.dynamic("c", "int64")
    d_std = T.dynamic("d", "int64")
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def std(rxplaceholder: T.Tensor((a_std, b_std, c_std, d_std), "float32"), compute: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            rxplaceholder_red = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)))
            T_divide = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1), T.int64(1), T.int64(1)))
            T_subtract = Ts.sblock_alloc_tensor((a_std, b_std, c_std, d_std))
            T_multiply = Ts.sblock_alloc_tensor((a_std, b_std, c_std, d_std))
            T_multiply_red = Ts.sblock_alloc_tensor(())
            T_divide_1 = Ts.sblock_alloc_tensor(())
            for ax0, ax1, ax2, ax3, k0, k1, k2, k3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1), a_std, b_std, c_std, d_std):
                with Ts.sblock("rxplaceholder_red"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_k0, v_k1, v_k2, v_k3 = Ts.axis.remap("SSSSRRRR", [ax0, ax1, ax2, ax3, k0, k1, k2, k3])
                    Ts.reads(rxplaceholder[v_k0, v_k1, v_k2, v_k3])
                    Ts.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] + rxplaceholder[v_k0, v_k1, v_k2, v_k3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), T.int64(1)):
                with Ts.sblock("T_divide"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] / T.Cast("float32", a_std * b_std * c_std * d_std)
            for ax0, ax1, ax2, ax3 in T.grid(a_std, b_std, c_std, d_std):
                with Ts.sblock("T_subtract"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_divide[T.int64(0), T.int64(0), T.int64(0), T.int64(0)])
                    Ts.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_divide[T.int64(0), T.int64(0), T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(a_std, b_std, c_std, d_std):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract[v_ax0, v_ax1, v_ax2, v_ax3]
            for k0, k1, k2, k3 in T.grid(a_std, b_std, c_std, d_std):
                with Ts.sblock("T_multiply_red"):
                    v_k0, v_k1, v_k2, v_k3 = Ts.axis.remap("RRRR", [k0, k1, k2, k3])
                    Ts.reads(T_multiply[v_k0, v_k1, v_k2, v_k3])
                    Ts.writes(T_multiply_red[()])
                    with Ts.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply[v_k0, v_k1, v_k2, v_k3]
            with Ts.sblock("T_divide_1"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_multiply_red[()])
                Ts.writes(T_divide_1[()])
                T_divide_1[()] = T_multiply_red[()] / T.Cast("float32", a_std * b_std * c_std * d_std)
            with Ts.sblock("compute"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(T_divide_1[()])
                Ts.writes(compute[()])
                compute[()] = T.sqrt(T_divide_1[()])

        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.std, (x,), ty_args=[R.Tensor((), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Std)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_variance():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Variance:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((1, 3, 4, 1), "float32"):
            gv: R.Tensor((1, 3, 4, 1), "float32") = R.variance(x, [0, 3], keepdims=True)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((1, 3, 4, 1), dtype="float32"):
            gv = R.call_tir(Expected.variance, (x,), ty_args=[R.Tensor((1, 3, 4, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def variance(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), T_divide: T.Tensor((T.int64(1), T.int64(3), T.int64(4), T.int64(1)), "float32")):
            T.func_attr({"tirx.noalias": True})
            rxplaceholder_red = Ts.sblock_alloc_tensor([T.int64(1), T.int64(3), T.int64(4), T.int64(1)], dtype="float32")
            T_divide_1 = Ts.sblock_alloc_tensor([T.int64(1), T.int64(3), T.int64(4), T.int64(1)], dtype="float32")
            T_subtract = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32")
            T_multiply = Ts.sblock_alloc_tensor([T.int64(2), T.int64(3), T.int64(4), T.int64(5)], dtype="float32")
            T_multiply_red = Ts.sblock_alloc_tensor([T.int64(1), T.int64(3), T.int64(4), T.int64(1)], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1), T.int64(2), T.int64(5)):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k3 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[k0, ax1, ax2, k3])
                    Ts.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] + rxplaceholder[k0, ax1, ax2, k3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1)):
                with Ts.sblock("T_divide"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    Ts.writes(T_divide_1[ax0, ax1, ax2, ax3])
                    T_divide_1[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] / T.float32(10.0)
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_subtract"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide_1[T.int64(0), ax1, ax2, T.int64(0)])
                    Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[ax0, ax1, ax2, ax3] - T_divide_1[T.int64(0), ax1, ax2, T.int64(0)]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(T_subtract[ax0, ax1, ax2, ax3])
                    Ts.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] * T_subtract[ax0, ax1, ax2, ax3]
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1), T.int64(2), T.int64(5)):
                with Ts.sblock("T_multiply_red"):
                    ax0, ax1, ax2, ax3, k0, k3 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(T_multiply[k0, ax1, ax2, k3])
                    Ts.writes(T_multiply_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        T_multiply_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    T_multiply_red[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] + T_multiply[k0, ax1, ax2, k3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1)):
                with Ts.sblock("T_divide_1"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(T_multiply_red[ax0, ax1, ax2, ax3])
                    Ts.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] / T.float32(10)
    # fmt: on

    mod = LegalizeOps()(Variance)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_variance_symbolic():
    # fmt: off
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    a = T.dynamic("a", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Variance:
        @R.function
        def main(x: R.Tensor((a, b, c, d), "float32")) -> R.Tensor((1, b, c, 1), "float32"):
            gv: R.Tensor((1, b, c, 1), "float32") = R.variance(x, [0, 3], keepdims=True)
            return gv

    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    a_main = T.dynamic("a", "int64")
    d_main = T.dynamic("d", "int64")
    a_variance = T.dynamic("a", "int64")
    b_variance = T.dynamic("b", "int64")
    c_variance = T.dynamic("c", "int64")
    d_variance = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((a_main, b_main, c_main, d_main), "float32")) -> R.Tensor((1, b_main, c_main, 1), "float32"):
            gv = R.call_tir(Expected.variance, (x,), ty_args=[R.Tensor((1, b_main, c_main, 1), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def variance(rxplaceholder: T.Tensor([a_variance, b_variance, c_variance, d_variance], dtype='float32'), T_divide: T.Tensor([T.int64(1), b_variance, c_variance, T.int64(1)], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            rxplaceholder_red = Ts.sblock_alloc_tensor([T.int64(1), b_variance, c_variance, T.int64(1)], dtype="float32")
            T_divide_1 = Ts.sblock_alloc_tensor([T.int64(1), b_variance, c_variance, T.int64(1)], dtype="float32")
            T_subtract = Ts.sblock_alloc_tensor([a_variance, b_variance, c_variance, d_variance], dtype="float32")
            T_multiply = Ts.sblock_alloc_tensor([a_variance, b_variance, c_variance, d_variance], dtype="float32")
            T_multiply_red = Ts.sblock_alloc_tensor([T.int64(1), b_variance, c_variance, T.int64(1)], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), b_variance, c_variance, T.int64(1), a_variance, d_variance):
                with Ts.sblock("rxplaceholder_red"):
                    ax0, ax1, ax2, ax3, k0, k3 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(rxplaceholder[k0, ax1, ax2, k3])
                    Ts.writes(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        rxplaceholder_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    rxplaceholder_red[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] + rxplaceholder[k0, ax1, ax2, k3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), b_variance, c_variance, T.int64(1)):
                with Ts.sblock("T_divide"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_red[ax0, ax1, ax2, ax3])
                    Ts.writes(T_divide_1[ax0, ax1, ax2, ax3])
                    T_divide_1[ax0, ax1, ax2, ax3] = rxplaceholder_red[ax0, ax1, ax2, ax3] / T.Cast("float32", a_variance * d_variance)
            for i0, i1, i2, i3 in T.grid(a_variance, b_variance, c_variance, d_variance):
                with Ts.sblock("T_subtract"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[ax0, ax1, ax2, ax3], T_divide_1[T.int64(0), ax1, ax2, T.int64(0)])
                    Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[ax0, ax1, ax2, ax3] - T_divide_1[T.int64(0), ax1, ax2, T.int64(0)]
            for i0, i1, i2, i3 in T.grid(a_variance, b_variance, c_variance, d_variance):
                with Ts.sblock("T_multiply"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(T_subtract[ax0, ax1, ax2, ax3])
                    Ts.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = T_subtract[ax0, ax1, ax2, ax3] * T_subtract[ax0, ax1, ax2, ax3]
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(1), b_variance, c_variance, T.int64(1), a_variance, d_variance):
                with Ts.sblock("T_multiply_red"):
                    ax0, ax1, ax2, ax3, k0, k3 = Ts.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    Ts.reads(T_multiply[k0, ax1, ax2, k3])
                    Ts.writes(T_multiply_red[ax0, ax1, ax2, ax3])
                    with Ts.init():
                        T_multiply_red[ax0, ax1, ax2, ax3] = T.float32(0)
                    T_multiply_red[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] + T_multiply[k0, ax1, ax2, k3]
            for i0, i1, i2, i3 in T.grid(T.int64(1), b_variance, c_variance, T.int64(1)):
                with Ts.sblock("T_divide_1"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(T_multiply_red[ax0, ax1, ax2, ax3])
                    Ts.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = T_multiply_red[ax0, ax1, ax2, ax3] / T.Cast("float32", a_variance * d_variance)
    # fmt: on

    mod = LegalizeOps()(Variance)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_variance_no_keepdims():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Variance:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((3, 4), "float32"):
            gv: R.Tensor((3, 4), "float32") = R.variance(x, [0, 3], keepdims=False)
            return gv

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def variance(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), T_divide: T.Tensor((T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            rxplaceholder_red = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(4), T.int64(1)))
            T_divide_1 = Ts.sblock_alloc_tensor((T.int64(1), T.int64(3), T.int64(4), T.int64(1)))
            T_subtract = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply = Ts.sblock_alloc_tensor((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply_red = Ts.sblock_alloc_tensor((T.int64(3), T.int64(4)))
            for ax0, ax1, ax2, ax3, k0, k3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1), T.int64(2), T.int64(5)):
                with Ts.sblock("rxplaceholder_red"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_k0, v_k3 = Ts.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, k0, k3])
                    Ts.reads(rxplaceholder[v_k0, v_ax1, v_ax2, v_k3])
                    Ts.writes(rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3])
                    with Ts.init():
                        rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] + rxplaceholder[v_k0, v_ax1, v_ax2, v_k3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(4), T.int64(1)):
                with Ts.sblock("T_divide"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide_1[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_red[v_ax0, v_ax1, v_ax2, v_ax3] / T.float32(10)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_subtract"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_divide_1[T.int64(0), v_ax1, v_ax2, T.int64(0)])
                    Ts.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_divide_1[T.int64(0), v_ax1, v_ax2, T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    Ts.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, k0, k3 in T.grid(T.int64(3), T.int64(4), T.int64(2), T.int64(5)):
                with Ts.sblock("T_multiply_red"):
                    v_ax0, v_ax1, v_k0, v_k3 = Ts.axis.remap("SSRR", [ax0, ax1, k0, k3])
                    Ts.reads(T_multiply[v_k0, v_ax0, v_ax1, v_k3])
                    Ts.writes(T_multiply_red[v_ax0, v_ax1])
                    with Ts.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_k0, v_ax0, v_ax1, v_k3]
            for ax0, ax1 in T.grid(T.int64(3), T.int64(4)):
                with Ts.sblock("T_divide_1"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(T_multiply_red[v_ax0, v_ax1])
                    Ts.writes(T_divide[v_ax0, v_ax1])
                    T_divide[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] / T.float32(10)

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32")) -> R.Tensor((3, 4), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.variance, (x,), ty_args=[R.Tensor((3, 4), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Variance)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_max_zero_dim():
    # Reducing a 0-D (scalar) tensor is the identity; it must legalize, not crash.
    # Regression test for https://github.com/apache/tvm/issues/19676
    # fmt: off
    @tvm_tirx.script.ir_module
    class Max:
        @R.function
        def main(x: R.Tensor((), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.max(x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((), dtype="float32"):
            gv = R.call_tir(Expected.max, (x,), ty_args=[R.Tensor((), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def max(x: T.Tensor((), "float32"), x_red: T.Tensor((), "float32")):
            T.func_attr({"tirx.noalias": True})
            with Ts.sblock("x_red"):
                vi = Ts.axis.spatial(T.int64(1), T.int64(0))
                Ts.reads(x[()])
                Ts.writes(x_red[()])
                x_red[()] = x[()]
    # fmt: on

    mod = LegalizeOps()(Max)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
