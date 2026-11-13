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
# ruff: noqa: E501

import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

##################### Binary arithmetic #####################


def test_add():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.add(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.add, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def add(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_add: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_add_with_arg0_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.add(x, R.const(1, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.add, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def add(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_add: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_add[ax0, ax1])
                    T_add[ax0, ax1] = rxplaceholder[ax0, ax1] + T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_add_with_arg1_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.add(R.const(1, "float32"), x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.add, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def add(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_add: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_add[ax0, ax1])
                    T_add[ax0, ax1] = T.float32(1) + rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_add_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.add(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_add = T.dynamic("a", "int64")
    b_add = T.dynamic("b", "int64")
    c_add = T.dynamic("c", "int64")
    d_add = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "float32"):
            gv = R.call_tir(Expected.add, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def add(rxplaceholder: T.Tensor([T.int64(1), c_add, d_add], dtype='float32'), rxplaceholder_1: T.Tensor([a_add, b_add, c_add, T.int64(1)], dtype='float32'), T_add: T.Tensor([a_add, b_add, c_add, d_add], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_add, b_add, c_add, d_add):
                with Ts.sblock("T_add"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] + rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_add_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.add(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.add, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")])
            return gv

        @Ts.function(private=True)
        def add(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = lhs[vi, vj, vk] + rhs

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_divide():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Divide:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.divide(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.divide, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def divide(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_divide: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_divide"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] / rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Divide)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_divide_with_arg0_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Divide:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.divide(x, R.const(1, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.divide, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def divide(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_divide: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_divide"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = rxplaceholder[ax0, ax1] / T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Divide)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_divide_with_arg1_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Divide:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.divide(R.const(1, "float32"), x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.divide, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def divide(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_divide: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_divide"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = T.float32(1) / rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Divide)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_divide_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Divide:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.divide(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_divide = T.dynamic("a", "int64")
    b_divide = T.dynamic("b", "int64")
    c_divide = T.dynamic("c", "int64")
    d_divide = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "float32"):
            gv = R.call_tir(Expected.divide, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def divide(rxplaceholder: T.Tensor([T.int64(1), c_divide, d_divide], dtype='float32'), rxplaceholder_1: T.Tensor([a_divide, b_divide, c_divide, T.int64(1)], dtype='float32'), T_divide: T.Tensor([a_divide, b_divide, c_divide, d_divide], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_divide, b_divide, c_divide, d_divide):
                with Ts.sblock("T_divide"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_divide[ax0, ax1, ax2, ax3])
                    T_divide[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] / rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Divide)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_divide_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.divide(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.divide, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")])
            return gv

        @Ts.function(private=True)
        def divide(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = lhs[vi, vj, vk] / rhs

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_floor_divide():
    # fmt: off
    @tvm_tirx.script.ir_module
    class FloorDivide:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.floor_divide(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.floor_divide, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def floor_divide(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_floor_divide: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_floor_divide"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_floor_divide[ax0, ax1, ax2, ax3])
                    T_floor_divide[ax0, ax1, ax2, ax3] = T.floor(rxplaceholder[T.int64(0), ax2, ax3] / rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(FloorDivide)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_floor_divide_with_arg0_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class FloorDivide:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.floor_divide(x, R.const(1, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.floor_divide, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def floor_divide(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_floor_divide: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_floor_divide"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_floor_divide[ax0, ax1])
                    T_floor_divide[ax0, ax1] = T.floor(rxplaceholder[ax0, ax1] / T.float32(1))
    # fmt: on

    mod = LegalizeOps()(FloorDivide)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_floor_divide_with_arg1_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class FloorDivide:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.floor_divide(R.const(1, "float32"), x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.floor_divide, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def floor_divide(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_floor_divide: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_floor_divide"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_floor_divide[ax0, ax1])
                    T_floor_divide[ax0, ax1] = T.floor(T.float32(1) / rxplaceholder[ax0, ax1])
    # fmt: on

    mod = LegalizeOps()(FloorDivide)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_floor_divide_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class FloorDivide:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.floor_divide(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_floor_divide = T.dynamic("a", "int64")
    b_floor_divide = T.dynamic("b", "int64")
    c_floor_divide = T.dynamic("c", "int64")
    d_floor_divide = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "float32"):
            gv = R.call_tir(Expected.floor_divide, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def floor_divide(rxplaceholder: T.Tensor([T.int64(1), c_floor_divide, d_floor_divide], dtype='float32'), rxplaceholder_1: T.Tensor([a_floor_divide, b_floor_divide, c_floor_divide, T.int64(1)], dtype='float32'), T_floor_divide: T.Tensor([a_floor_divide, b_floor_divide, c_floor_divide, d_floor_divide], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_floor_divide, b_floor_divide, c_floor_divide, d_floor_divide):
                with Ts.sblock("T_floor_divide"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_floor_divide[ax0, ax1, ax2, ax3])
                    T_floor_divide[ax0, ax1, ax2, ax3] = T.floor(rxplaceholder[T.int64(0), ax2, ax3] / rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(FloorDivide)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_floordiv_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.floor_divide(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(
                cls.floor_divide, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")]
            )
            return gv

        @Ts.function(private=True)
        def floor_divide(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_floordiv"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = T.floor(lhs[vi, vj, vk] / rhs)

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_multiply():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Multiply:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.multiply(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.multiply, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def multiply(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_multiply: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] * rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Multiply)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_multiply_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Multiply:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.multiply(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_multiply = T.dynamic("a", "int64")
    b_multiply = T.dynamic("b", "int64")
    c_multiply = T.dynamic("c", "int64")
    d_multiply = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "float32"):
            gv = R.call_tir(Expected.multiply, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def multiply(rxplaceholder: T.Tensor([T.int64(1), c_multiply, d_multiply], dtype='float32'), rxplaceholder_1: T.Tensor([a_multiply, b_multiply, c_multiply, T.int64(1)], dtype='float32'), T_multiply: T.Tensor([a_multiply, b_multiply, c_multiply, d_multiply], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_multiply, b_multiply, c_multiply, d_multiply):
                with Ts.sblock("T_multiply"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_multiply[ax0, ax1, ax2, ax3])
                    T_multiply[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] * rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Multiply)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_multiply_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.multiply(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.multiply, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")])
            return gv

        @Ts.function(private=True)
        def multiply(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = lhs[vi, vj, vk] * rhs

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_power():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Power:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.power(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def power(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_power: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_power"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])
                    Ts.writes(T_power[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_power[v_ax0, v_ax1, v_ax2, v_ax3] = T.pow(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])

        @R.function
        def main(x: R.Tensor((1, 2, 3), dtype="float32"), y: R.Tensor((4, 3, 2, 1), dtype="float32")) -> R.Tensor((4, 3, 2, 3), dtype="float32"):
            gv = R.call_tir(Expected.power, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

    # fmt: on

    mod = LegalizeOps()(Power)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_power_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Power:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.power(x, y)
            return gv

    c_power = T.dynamic("c", "int64")
    d_power = T.dynamic("d", "int64")
    a_power = T.dynamic("a", "int64")
    b_power = T.dynamic("b", "int64")
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def power(rxplaceholder: T.Tensor((T.int64(1), c_power, d_power), "float32"), rxplaceholder_1: T.Tensor((a_power, b_power, c_power, T.int64(1)), "float32"), T_power: T.Tensor((a_power, b_power, c_power, d_power), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(a_power, b_power, c_power, d_power):
                with Ts.sblock("T_power"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])
                    Ts.writes(T_power[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_power[v_ax0, v_ax1, v_ax2, v_ax3] = T.pow(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])

        @R.function
        def main(x: R.Tensor((1, c_main, d_main), dtype="float32"), y: R.Tensor((a_main, b_main, c_main, 1), dtype="float32")) -> R.Tensor((a_main, b_main, c_main, d_main), dtype="float32"):
            gv = R.call_tir(Expected.power, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Expected)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_power_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.power(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.power, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")])
            return gv

        @Ts.function(private=True)
        def power(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_power"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = T.pow(lhs[vi, vj, vk], rhs)

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_atan2():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Atan2:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.atan2(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def atan2(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_atan2: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_atan2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])
                    Ts.writes(T_atan2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_atan2[v_ax0, v_ax1, v_ax2, v_ax3] = T.atan2(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])

        @R.function
        def main(x: R.Tensor((1, 2, 3), dtype="float32"), y: R.Tensor((4, 3, 2, 1), dtype="float32")) -> R.Tensor((4, 3, 2, 3), dtype="float32"):
            gv = R.call_tir(Expected.atan2, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

    # fmt: on

    mod = LegalizeOps()(Atan2)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_atan2_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Atan2:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.atan2(x, y)
            return gv

    c_atan2 = T.dynamic("c", "int64")
    d_atan2 = T.dynamic("d", "int64")
    a_atan2 = T.dynamic("a", "int64")
    b_atan2 = T.dynamic("b", "int64")
    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @Ts.function(private=True)
        def atan2(rxplaceholder: T.Tensor((T.int64(1), c_atan2, d_atan2), "float32"), rxplaceholder_1: T.Tensor((a_atan2, b_atan2, c_atan2, T.int64(1)), "float32"), T_atan2: T.Tensor((a_atan2, b_atan2, c_atan2, d_atan2), "float32")):
            T.func_attr({"tirx.noalias": True})

            for ax0, ax1, ax2, ax3 in T.grid(a_atan2, b_atan2, c_atan2, d_atan2):
                with Ts.sblock("T_atan2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    Ts.reads(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])
                    Ts.writes(T_atan2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_atan2[v_ax0, v_ax1, v_ax2, v_ax3] = T.atan2(rxplaceholder[T.int64(0), v_ax2, v_ax3], rxplaceholder_1[v_ax0, v_ax1, v_ax2, T.int64(0)])

        @R.function
        def main(x: R.Tensor((1, c_main, d_main), dtype="float32"), y: R.Tensor((a_main, b_main, c_main, 1), dtype="float32")) -> R.Tensor((a_main, b_main, c_main, d_main), dtype="float32"):
            gv = R.call_tir(Expected.atan2, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(Expected)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_atan2_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.atan2(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.atan2, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")])
            return gv

        @Ts.function(private=True)
        def atan2(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_atan2"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = T.atan2(lhs[vi, vj, vk], rhs)

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_subtract():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Subtract:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.subtract(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.subtract, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def subtract(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_subtract: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_subtract"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] - rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Subtract)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_subtract_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Subtract:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.subtract(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_subtract = T.dynamic("a", "int64")
    b_subtract = T.dynamic("b", "int64")
    c_subtract = T.dynamic("c", "int64")
    d_subtract = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "float32"):
            gv = R.call_tir(Expected.subtract, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def subtract(rxplaceholder: T.Tensor([T.int64(1), c_subtract, d_subtract], dtype='float32'), rxplaceholder_1: T.Tensor([a_subtract, b_subtract, c_subtract, T.int64(1)], dtype='float32'), T_subtract: T.Tensor([a_subtract, b_subtract, c_subtract, d_subtract], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_subtract, b_subtract, c_subtract, d_subtract):
                with Ts.sblock("T_subtract"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_subtract[ax0, ax1, ax2, ax3])
                    T_subtract[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] - rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Subtract)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_subtract_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.subtract(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.subtract, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")])
            return gv

        @Ts.function(private=True)
        def subtract(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = lhs[vi, vj, vk] - rhs

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


##################### Binary comparison #####################


def test_equal():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Equal:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.equal(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.equal, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def equal(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_equal: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_equal"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_equal[ax0, ax1, ax2, ax3])
                    T_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] == rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Equal)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_equal_with_arg0_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.equal(x, R.const(1, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.equal, (x,), ty_args=[R.Tensor((2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def equal(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_equal: T.Tensor((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_equal"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_equal[ax0, ax1])
                    T_equal[ax0, ax1] = rxplaceholder[ax0, ax1] == T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_equal_with_arg1_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.equal(R.const(1, "float32"), x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.equal, (x,), ty_args=[R.Tensor((2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def equal(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_equal: T.Tensor((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_equal"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_equal[ax0, ax1])
                    T_equal[ax0, ax1] = T.float32(1) == rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_equal_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Equal:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "bool"):
            gv: R.Tensor((a, b, c, d), "bool") = R.equal(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_equal = T.dynamic("a", "int64")
    b_equal = T.dynamic("b", "int64")
    c_equal = T.dynamic("c", "int64")
    d_equal = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "bool"):
            gv = R.call_tir(Expected.equal, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def equal(rxplaceholder: T.Tensor([T.int64(1), c_equal, d_equal], dtype='float32'), rxplaceholder_1: T.Tensor([a_equal, b_equal, c_equal, T.int64(1)], dtype='float32'), T_equal: T.Tensor([a_equal, b_equal, c_equal, d_equal], dtype='bool')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_equal, b_equal, c_equal, d_equal):
                with Ts.sblock("T_equal"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_equal[ax0, ax1, ax2, ax3])
                    T_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] == rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Equal)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_equal_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.equal(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.equal, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="bool")])
            return gv

        @Ts.function(private=True)
        def equal(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "bool"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = lhs[vi, vj, vk] == rhs

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_greater():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Greater:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.greater(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.greater, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def greater(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_greater: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_greater"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    Ts.writes(T_greater[ax0, ax1, ax2, ax3])
                    T_greater[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] < rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(Greater)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_greater_with_arg0_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.greater(x, R.const(1, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.greater, (x,), ty_args=[R.Tensor((2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def greater(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_greater: T.Tensor((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_greater"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_greater[ax0, ax1])
                    T_greater[ax0, ax1] = T.float32(1) < rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_greater_with_arg1_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.greater(R.const(1, "float32"), x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.greater, (x,), ty_args=[R.Tensor((2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def greater(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_greater: T.Tensor((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_greater"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_greater[ax0, ax1])
                    T_greater[ax0, ax1] = rxplaceholder[ax0, ax1] < T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_greater_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Greater:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "bool"):
            gv: R.Tensor((a, b, c, d), "bool") = R.greater(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_greater = T.dynamic("a", "int64")
    b_greater = T.dynamic("b", "int64")
    c_greater = T.dynamic("c", "int64")
    d_greater = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "bool"):
            gv = R.call_tir(Expected.greater, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def greater(rxplaceholder: T.Tensor([T.int64(1), c_greater, d_greater], dtype='float32'), rxplaceholder_1: T.Tensor([a_greater, b_greater, c_greater, T.int64(1)], dtype='float32'), T_greater: T.Tensor([a_greater, b_greater, c_greater, d_greater], dtype='bool')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_greater, b_greater, c_greater, d_greater):
                with Ts.sblock("T_greater"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    Ts.writes(T_greater[ax0, ax1, ax2, ax3])
                    T_greater[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] < rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(Greater)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_greater_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.greater(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.greater, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="bool")])
            return gv

        @Ts.function(private=True)
        def greater(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "bool"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = rhs < lhs[vi, vj, vk]

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_greater_equal():
    # fmt: off
    @tvm_tirx.script.ir_module
    class GreaterEqual:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.greater_equal(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.greater_equal, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def greater_equal(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_greater_equal: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_greater_equal"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    Ts.writes(T_greater_equal[ax0, ax1, ax2, ax3])
                    T_greater_equal[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] <= rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(GreaterEqual)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_greater_equal_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class GreaterEqual:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "bool"):
            gv: R.Tensor((a, b, c, d), "bool") = R.greater_equal(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_greater_equal = T.dynamic("a", "int64")
    b_greater_equal = T.dynamic("b", "int64")
    c_greater_equal = T.dynamic("c", "int64")
    d_greater_equal = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "bool"):
            gv = R.call_tir(Expected.greater_equal, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def greater_equal(rxplaceholder: T.Tensor([T.int64(1), c_greater_equal, d_greater_equal], dtype='float32'), rxplaceholder_1: T.Tensor([a_greater_equal, b_greater_equal, c_greater_equal, T.int64(1)], dtype='float32'), T_greater_equal: T.Tensor([a_greater_equal, b_greater_equal, c_greater_equal, d_greater_equal], dtype='bool')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_greater_equal, b_greater_equal, c_greater_equal, d_greater_equal):
                with Ts.sblock("T_greater_equal"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    Ts.writes(T_greater_equal[ax0, ax1, ax2, ax3])
                    T_greater_equal[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] <= rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(GreaterEqual)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_greater_equal_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.greater_equal(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(
                cls.greater_equal, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="bool")]
            )
            return gv

        @Ts.function(private=True)
        def greater_equal(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "bool"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = rhs <= lhs[vi, vj, vk]

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_less():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Less:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.less(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.less, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def less(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_less: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_less"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_less[ax0, ax1, ax2, ax3])
                    T_less[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] < rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Less)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_less_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Less:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "bool"):
            gv: R.Tensor((a, b, c, d), "bool") = R.less(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_less = T.dynamic("a", "int64")
    b_less = T.dynamic("b", "int64")
    c_less = T.dynamic("c", "int64")
    d_less = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "bool"):
            gv = R.call_tir(Expected.less, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def less(rxplaceholder: T.Tensor([T.int64(1), c_less, d_less], dtype='float32'), rxplaceholder_1: T.Tensor([a_less, b_less, c_less, T.int64(1)], dtype='float32'), T_less: T.Tensor([a_less, b_less, c_less, d_less], dtype='bool')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_less, b_less, c_less, d_less):
                with Ts.sblock("T_less"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_less[ax0, ax1, ax2, ax3])
                    T_less[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] < rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(Less)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_less_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.less(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.less, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="bool")])
            return gv

        @Ts.function(private=True)
        def less(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "bool"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = lhs[vi, vj, vk] < rhs

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_less_equal():
    # fmt: off
    @tvm_tirx.script.ir_module
    class LessEqual:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.less_equal(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.less_equal, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def less_equal(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_less_equal: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_less_equal"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_less_equal[ax0, ax1, ax2, ax3])
                    T_less_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] <= rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(LessEqual)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_less_equal_with_arg0_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.less_equal(x, R.const(1, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.less_equal, (x,), ty_args=[R.Tensor((2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def less_equal(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_less_equal: T.Tensor((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_less_equal"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_less_equal[ax0, ax1])
                    T_less_equal[ax0, ax1] = rxplaceholder[ax0, ax1] <= T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_less_equal_with_arg1_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Add:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv: R.Tensor((2, 3), dtype="bool") = R.less_equal(R.const(1, "float32"), x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "bool"):
            gv = R.call_tir(Expected.less_equal, (x,), ty_args=[R.Tensor((2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def less_equal(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_less_equal: T.Tensor((T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_less_equal"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_less_equal[ax0, ax1])
                    T_less_equal[ax0, ax1] = T.float32(1) <= rxplaceholder[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_less_equal_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class LessEqual:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "bool"):
            gv: R.Tensor((a, b, c, d), "bool") = R.less_equal(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_less_equal = T.dynamic("a", "int64")
    b_less_equal = T.dynamic("b", "int64")
    c_less_equal = T.dynamic("c", "int64")
    d_less_equal = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "bool"):
            gv = R.call_tir(Expected.less_equal, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def less_equal(rxplaceholder: T.Tensor([T.int64(1), c_less_equal, d_less_equal], dtype='float32'), rxplaceholder_1: T.Tensor([a_less_equal, b_less_equal, c_less_equal, T.int64(1)], dtype='float32'), T_less_equal: T.Tensor([a_less_equal, b_less_equal, c_less_equal, d_less_equal], dtype='bool')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_less_equal, b_less_equal, c_less_equal, d_less_equal):
                with Ts.sblock("T_less_equal"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_less_equal[ax0, ax1, ax2, ax3])
                    T_less_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] <= rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(LessEqual)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_less_equal_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.less_equal(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.less_equal, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="bool")])
            return gv

        @Ts.function(private=True)
        def less_equal(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "bool"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = lhs[vi, vj, vk] <= rhs

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_not_equal():
    # fmt: off
    @tvm_tirx.script.ir_module
    class NotEqual:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv: R.Tensor((4, 3, 2, 3), "bool") = R.not_equal(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "bool"):
            gv = R.call_tir(Expected.not_equal, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def not_equal(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_not_equal: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "bool")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_not_equal"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_not_equal[ax0, ax1, ax2, ax3])
                    T_not_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] != rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(NotEqual)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_not_equal_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class NotEqual:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "bool"):
            gv: R.Tensor((a, b, c, d), "bool") = R.not_equal(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_not_equal = T.dynamic("a", "int64")
    b_not_equal = T.dynamic("b", "int64")
    c_not_equal = T.dynamic("c", "int64")
    d_not_equal = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "bool"):
            gv = R.call_tir(Expected.not_equal, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="bool")])
            return gv

        @Ts.function(private=True)
        def not_equal(rxplaceholder: T.Tensor([T.int64(1), c_not_equal, d_not_equal], dtype='float32'), rxplaceholder_1: T.Tensor([a_not_equal, b_not_equal, c_not_equal, T.int64(1)], dtype='float32'), T_not_equal: T.Tensor([a_not_equal, b_not_equal, c_not_equal, d_not_equal], dtype='bool')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_not_equal, b_not_equal, c_not_equal, d_not_equal):
                with Ts.sblock("T_not_equal"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_not_equal[ax0, ax1, ax2, ax3])
                    T_not_equal[ax0, ax1, ax2, ax3] = rxplaceholder[T.int64(0), ax2, ax3] != rxplaceholder_1[ax0, ax1, ax2, T.int64(0)]
    # fmt: on

    mod = LegalizeOps()(NotEqual)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_not_equal_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.not_equal(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.not_equal, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="bool")])
            return gv

        @Ts.function(private=True)
        def not_equal(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "bool"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = lhs[vi, vj, vk] != rhs

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_maximum():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Maximum:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.maximum(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.maximum, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def maximum(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_maximum: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_maximum"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_maximum[ax0, ax1, ax2, ax3])
                    T_maximum[ax0, ax1, ax2, ax3] = T.max(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Maximum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_maximum_with_arg0_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Maximum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.maximum(x, R.const(1, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.maximum, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def maximum(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_maximum: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_maximum"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_maximum[ax0, ax1])
                    T_maximum[ax0, ax1] = T.max(rxplaceholder[ax0, ax1], T.float32(1))
    # fmt: on

    mod = LegalizeOps()(Maximum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_maximum_with_arg1_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Maximum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.maximum(R.const(1, "float32"), x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.maximum, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def maximum(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_maximum: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_maximum"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_maximum[ax0, ax1])
                    T_maximum[ax0, ax1] = T.max(T.float32(1), rxplaceholder[ax0, ax1])
    # fmt: on

    mod = LegalizeOps()(Maximum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_maximum_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Maximum:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.maximum(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_maximum = T.dynamic("a", "int64")
    b_maximum = T.dynamic("b", "int64")
    c_maximum = T.dynamic("c", "int64")
    d_maximum = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "float32"):
            gv = R.call_tir(Expected.maximum, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def maximum(rxplaceholder: T.Tensor([T.int64(1), c_maximum, d_maximum], dtype='float32'), rxplaceholder_1: T.Tensor([a_maximum, b_maximum, c_maximum, T.int64(1)], dtype='float32'), T_maximum: T.Tensor([a_maximum, b_maximum, c_maximum, d_maximum], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_maximum, b_maximum, c_maximum, d_maximum):
                with Ts.sblock("T_maximum"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_maximum[ax0, ax1, ax2, ax3])
                    T_maximum[ax0, ax1, ax2, ax3] = T.max(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Maximum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_max_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.maximum(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.maximum, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")])
            return gv

        @Ts.function(private=True)
        def maximum(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = T.max(lhs[vi, vj, vk], rhs)

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_minimum():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Minimum:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv: R.Tensor((4, 3, 2, 3), "float32") = R.minimum(x, y)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3), "float32"), y: R.Tensor((4, 3, 2, 1), "float32")) -> R.Tensor((4, 3, 2, 3), "float32"):
            gv = R.call_tir(Expected.minimum, (x, y), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def minimum(rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), T_minimum: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_minimum"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_minimum[ax0, ax1, ax2, ax3])
                    T_minimum[ax0, ax1, ax2, ax3] = T.min(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Minimum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_minimum_with_arg0_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Minimum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.minimum(x, R.const(1, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.minimum, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def minimum(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_minimum: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_minimum"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_minimum[ax0, ax1])
                    T_minimum[ax0, ax1] = T.min(rxplaceholder[ax0, ax1], T.float32(1))
    # fmt: on

    mod = LegalizeOps()(Minimum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_minimum_with_arg1_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Minimum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), dtype="float32") = R.minimum(R.const(1, "float32"), x)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.minimum, (x,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def minimum(rxplaceholder: T.Tensor((T.int64(2), T.int64(3)), "float32"), T_minimum: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_minimum"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[ax0, ax1])
                    Ts.writes(T_minimum[ax0, ax1])
                    T_minimum[ax0, ax1] = T.min(T.float32(1), rxplaceholder[ax0, ax1])
    # fmt: on

    mod = LegalizeOps()(Minimum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_minimum_symbolic():
    # fmt: off
    a = T.dynamic("a", "int64")
    b = T.dynamic("b", "int64")
    c = T.dynamic("c", "int64")
    d = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Minimum:
        @R.function
        def main(x: R.Tensor((1, c, d), "float32"), y: R.Tensor((a, b, c, 1), "float32")) -> R.Tensor((a, b, c, d), "float32"):
            gv: R.Tensor((a, b, c, d), "float32") = R.minimum(x, y)
            return gv

    a_main = T.dynamic("a", "int64")
    b_main = T.dynamic("b", "int64")
    c_main = T.dynamic("c", "int64")
    d_main = T.dynamic("d", "int64")
    a_minimum = T.dynamic("a", "int64")
    b_minimum = T.dynamic("b", "int64")
    c_minimum = T.dynamic("c", "int64")
    d_minimum = T.dynamic("d", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, c_main, d_main), "float32"), y: R.Tensor((a_main, b_main, c_main, 1), "float32")) -> R.Tensor((a_main, b_main, c_main, d_main), "float32"):
            gv = R.call_tir(Expected.minimum, (x, y), ty_args=[R.Tensor((a_main, b_main, c_main, d_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def minimum(rxplaceholder: T.Tensor([T.int64(1), c_minimum, d_minimum], dtype='float32'), rxplaceholder_1: T.Tensor([a_minimum, b_minimum, c_minimum, T.int64(1)], dtype='float32'), T_minimum: T.Tensor([a_minimum, b_minimum, c_minimum, d_minimum], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2, i3 in T.grid(a_minimum, b_minimum, c_minimum, d_minimum):
                with Ts.sblock("T_minimum"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
                    Ts.writes(T_minimum[ax0, ax1, ax2, ax3])
                    T_minimum[ax0, ax1, ax2, ax3] = T.min(rxplaceholder[T.int64(0), ax2, ax3], rxplaceholder_1[ax0, ax1, ax2, T.int64(0)])
    # fmt: on

    mod = LegalizeOps()(Minimum)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_min_primvalue():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            gv = R.minimum(x, y)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([64, 32, 16], "float32"),
            y: T.float32,
        ):
            cls = Expected
            gv = R.call_tir(cls.minimum, (x, y), ty_args=[R.Tensor([64, 32, 16], dtype="float32")])
            return gv

        @Ts.function(private=True)
        def minimum(
            lhs: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
            rhs: T.float32,
            output: T.Tensor([T.int64(64), T.int64(32), T.int64(16)], "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j, k in T.grid(*lhs.shape):
                with Ts.sblock("T_add"):
                    vi, vj, vk = Ts.axis.remap("SSS", [i, j, k])
                    output[vi, vj, vk] = T.min(lhs[vi, vj, vk], rhs)

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
