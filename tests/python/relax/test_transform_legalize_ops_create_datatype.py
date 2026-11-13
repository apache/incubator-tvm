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
from tvm_tirx.script import tirx as T

##################### Creation #####################


def test_full():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Full:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.full((2, 3), v, dtype="int32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(Expected.full, (v,), ty_args=[R.Tensor((2, 3), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def full(rxplaceholder: T.Tensor((), "int32"), T_full: T.Tensor((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(Full)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_full_constant_scalar_fill_value():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Full:
        @R.function
        def main() -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.full((2, 3), R.const(3.5, "float32"), dtype="int32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(Expected.full, R.tuple(), ty_args=[R.Tensor((2, 3), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def full(T_full: T.Tensor((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = 3
    # fmt: on

    mod = LegalizeOps()(Full)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_full_different_dtype():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Full:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.full((2, 3), v, dtype="float32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(v: R.Tensor((), "int32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.full, (v,), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def full(rxplaceholder: T.Tensor((), "int32"), T_full: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.Cast("float32", rxplaceholder[()])
    # fmt: on

    mod = LegalizeOps()(Full)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_full_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Full:
        @R.function
        def main(dumb_param: R.Tensor((m, n)), v: R.Tensor((), "int32")) -> R.Tensor((m, n), "int32"):
            gv: R.Tensor((m, n), "int32") = R.full((m, n), v, dtype="int32")
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_full = T.dynamic("m", "int64")
    n_full = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor((m_main, n_main)), v: R.Tensor((), "int32")) -> R.Tensor((m_main, n_main), "int32"):
            gv = R.call_tir(Expected.full, (v,), ty_args=[R.Tensor((m_main, n_main), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def full(rxplaceholder: T.Tensor((), "int32"), T_full: T.Tensor([m_full, n_full], dtype='int32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_full, n_full):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(Full)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_full_like():
    # fmt: off
    @tvm_tirx.script.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.full_like(x, v)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(Expected.full, (v,), ty_args=[R.Tensor((2, 3), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def full(rxplaceholder: T.Tensor((), "float32"), T_full: T.Tensor((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(FullLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_full_like_constant_scalar_fill_value():
    # fmt: off
    @tvm_tirx.script.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.full_like(x, R.const(-5, "float32"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "int32")) -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(Expected.full, R.tuple(), ty_args=[R.Tensor((2, 3), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def full(T_full: T.Tensor((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.int32(-5)
    # fmt: on

    mod = LegalizeOps()(FullLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_full_like_different_dtype():
    # fmt: off
    @tvm_tirx.script.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((2, 3), "float64"):
            gv: R.Tensor((2, 3), "float64") = R.full_like(x, v, dtype="float64")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((2, 3), "float64"):
            gv = R.call_tir(Expected.full, (v,), ty_args=[R.Tensor((2, 3), dtype="float64")])
            return gv

        @Ts.function(private=True)
        def full(rxplaceholder: T.Tensor((), "float32"), T_full: T.Tensor((T.int64(2), T.int64(3)), "float64")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.Cast("float64", rxplaceholder[()])
    # fmt: on

    mod = LegalizeOps()(FullLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_full_like_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class FullLike:
        @R.function
        def main(x: R.Tensor((m, n), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((m, n), "int32"):
            gv: R.Tensor((m, n), "int32") = R.full_like(x, v)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_full = T.dynamic("m", "int64")
    n_full = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "int32"), v: R.Tensor((), "float32")) -> R.Tensor((m_main, n_main), "int32"):
            gv = R.call_tir(Expected.full, (v,), ty_args=[R.Tensor((m_main, n_main), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def full(rxplaceholder: T.Tensor((), "float32"), T_full: T.Tensor([m_full, n_full], dtype='int32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_full, n_full):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[()])
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.int32(rxplaceholder[()])
    # fmt: on

    mod = LegalizeOps()(FullLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_ones():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Ones:
        @R.function
        def main() -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.ones((2, 3), "float32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.ones, R.tuple(), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def ones(T_full: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Ones)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_ones_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Ones:
        @R.function
        def main(dumb_param: R.Tensor((m, n))) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.ones((m, n), "float32")
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_ones = T.dynamic("m", "int64")
    n_ones = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor((m_main, n_main))) -> R.Tensor((m_main, n_main), "float32"):
            gv = R.call_tir(Expected.ones, R.tuple(), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def ones(T_full: T.Tensor([m_ones, n_ones], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_ones, n_ones):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(1)
    # fmt: on

    mod = LegalizeOps()(Ones)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_ones_like():
    # fmt: off
    @tvm_tirx.script.ir_module
    class OnesLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.ones_like(x, "int32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(Expected.ones, R.tuple(), ty_args=[R.Tensor((2, 3), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def ones(T_full: T.Tensor((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = 1
    # fmt: on

    mod = LegalizeOps()(OnesLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_ones_like_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class OnesLike:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.ones_like(x)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_ones = T.dynamic("m", "int64")
    n_ones = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32")) -> R.Tensor((m_main, n_main), "float32"):
            gv = R.call_tir(Expected.ones, R.tuple(), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def ones(T_full: T.Tensor([m_ones, n_ones], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_ones, n_ones):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(1)
    # fmt: on

    mod = LegalizeOps()(OnesLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_zeros():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Zeros:
        @R.function
        def main() -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.zeros((2, 3), "float32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.zeros, R.tuple(), ty_args=[R.Tensor((2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def zeros(T_full: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(0)
    # fmt: on

    mod = LegalizeOps()(Zeros)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_zeros_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Zeros:
        @R.function
        def main(dumb_param: R.Tensor((m, n))) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.zeros((m, n), "float32")
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_zeros = T.dynamic("m", "int64")
    n_zeros = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor((m_main, n_main))) -> R.Tensor((m_main, n_main), "float32"):
            gv = R.call_tir(Expected.zeros, R.tuple(), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def zeros(T_full: T.Tensor([m_zeros, n_zeros], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_zeros, n_zeros):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(0)
    # fmt: on

    mod = LegalizeOps()(Zeros)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_zeros_like():
    # fmt: off
    @tvm_tirx.script.ir_module
    class ZerosLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "int32"):
            gv: R.Tensor((2, 3), "int32") = R.zeros_like(x, "int32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "int32"):
            gv = R.call_tir(Expected.zeros, R.tuple(), ty_args=[R.Tensor((2, 3), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def zeros(T_full: T.Tensor((T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = 0
    # fmt: on

    mod = LegalizeOps()(ZerosLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_zeros_like_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class ZerosLike:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "float32"):
            gv: R.Tensor((m, n), "float32") = R.zeros_like(x)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_zeros = T.dynamic("m", "int64")
    n_zeros = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32")) -> R.Tensor((m_main, n_main), "float32"):
            gv = R.call_tir(Expected.zeros, R.tuple(), ty_args=[R.Tensor((m_main, n_main), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def zeros(T_full: T.Tensor([m_zeros, n_zeros], dtype='float32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_zeros, n_zeros):
                with Ts.sblock("T_full"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads()
                    Ts.writes(T_full[ax0, ax1])
                    T_full[ax0, ax1] = T.float32(0)
    # fmt: on

    mod = LegalizeOps()(ZerosLike)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_arange_const():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Arange:
        @R.function
        def main():
            gv = R.arange(1, 10, 2)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main():
            gv = R.const([1, 3, 5, 7, 9], dtype="int64")
            return gv
    # fmt: on

    mod = LegalizeOps()(Arange)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_arange_symbolic():
    # fmt: off
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Arange:
        @R.function
        def main(x: R.Tensor([n], "float32")):
            gv = R.arange(1, R.prim_value(n), 2)
            return gv

    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor([n], "float32")):
            cls = Expected
            gv = R.call_tir(cls.arange, (n,), ty_args=[R.Tensor((n // 2,), dtype="int64")])
            return gv

        arange_n = T.int64()

        @Ts.function(private=True)
        def arange(n: arange_n, T_arange: T.Tensor((arange_n // T.int64(2),), 'int64')):
            T.func_attr({"tirx.noalias": True})

            for ax0 in range(n // T.int64(2)):
                with Ts.sblock("T_arange"):
                    v_ax0 = Ts.axis.spatial(n // T.int64(2), ax0)
                    T_arange[v_ax0] = v_ax0 * T.int64(2) + T.int64(1)
    # fmt: on

    mod = LegalizeOps()(Arange)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_shape_to_tensor():
    # fmt: off
    @tvm_tirx.script.ir_module
    class ShapeToTensor:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")):
            gv = R.shape_to_tensor(R.shape_of(x))
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((3,), "int64"):
            cls = Expected
            gv: R.Shape([2, 3, 4]) = R.shape_of(x)
            gv_1 = R.call_tir(cls.shape_to_tensor, R.tuple(), ty_args=[R.Tensor((3,), dtype="int64")])
            return gv_1

        @Ts.function(private=True)
        def shape_to_tensor(shape_to_tensor: T.Tensor((T.int64(3),), "int64")):
            T.func_attr({"tirx.noalias": True})
            for i in range(T.int64(3)):
                with Ts.sblock("shape_to_tensor"):
                    v_i = Ts.axis.spatial(T.int64(3), i)
                    shape_to_tensor[v_i] = T.if_then_else(v_i == T.int64(0), T.int64(2), T.if_then_else(v_i == T.int64(1), T.int64(3), T.if_then_else(v_i == T.int64(2), T.int64(4), T.int64(0))))
    # fmt: on

    mod = LegalizeOps()(ShapeToTensor)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_shape_to_tensor_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class ShapeToTensor:
        @R.function
        def main(x: R.Tensor((m, n), "float32")):
            gv = R.shape_to_tensor(R.shape_of(x))
            return gv

    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((2,), "int64"):
            cls = Expected
            gv: R.Shape([m, n]) = R.shape_of(x)
            gv_1 = R.call_tir(cls.shape_to_tensor, (m, n), ty_args=[R.Tensor((2,), dtype="int64")])
            return gv_1

        @Ts.function(private=True)
        def shape_to_tensor(m: T.int64, n: T.int64, shape_to_tensor: T.Tensor((T.int64(2),), "int64")):
            T.func_attr({"tirx.noalias": True})
            for i in range(T.int64(2)):
                with Ts.sblock("shape_to_tensor"):
                    v_i = Ts.axis.spatial(T.int64(2), i)
                    shape_to_tensor[v_i] = T.if_then_else(v_i == T.int64(0), m, T.if_then_else(v_i == T.int64(1), n, T.int64(0)))
    # fmt: on

    mod = LegalizeOps()(ShapeToTensor)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_shape_to_tensor_mixed():
    # fmt: off
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class ShapeToTensor:
        @R.function
        def main(x: R.Tensor((m, 3), "float32")):
            gv = R.shape_to_tensor(R.shape_of(x))
            return gv

    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m, 3), "float32")) -> R.Tensor((2,), "int64"):
            cls = Expected
            gv: R.Shape([m, 3]) = R.shape_of(x)
            gv_1 = R.call_tir(cls.shape_to_tensor, (m,), ty_args=[R.Tensor((2,), dtype="int64")])
            return gv_1

        @Ts.function(private=True)
        def shape_to_tensor(m: T.int64, shape_to_tensor: T.Tensor((T.int64(2),), "int64")):
            T.func_attr({"tirx.noalias": True})
            for i in range(T.int64(2)):
                with Ts.sblock("shape_to_tensor"):
                    v_i = Ts.axis.spatial(T.int64(2), i)
                    shape_to_tensor[v_i] = T.if_then_else(v_i == T.int64(0), m, T.if_then_else(v_i == T.int64(1), T.int64(3), T.int64(0)))
    # fmt: on

    mod = LegalizeOps()(ShapeToTensor)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_shape_to_tensor_unknown_values():
    @tvm_tirx.script.ir_module
    class ShapeToTensor:
        @R.function
        def main(s: R.Shape(ndim=2)):
            gv = R.shape_to_tensor(s)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(s: R.Shape(ndim=2)) -> R.Tensor((2,), "int64"):
            gv: R.Tensor((2,), dtype="int64") = R.call_pure_packed(
                "relax.run.shape_to_tensor", s, ty_args=(R.Tensor((2,), dtype="int64"),)
            )
            return gv

    mod = LegalizeOps()(ShapeToTensor)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_tril():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Tril:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv: R.Tensor((2, 3, 4), "float32") = R.tril(x, k=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv = R.call_tir(Expected.tril, (x,), ty_args=[R.Tensor((2, 3, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def tril(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), trilu: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with Ts.sblock("trilu"):
                    i0_1, i1_1, i2_1 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    Ts.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(i2_1 <= i1_1 + T.int64(1), rxplaceholder[i0_1, i1_1, i2_1], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(Tril)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_tril_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    k = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class Tril:
        @R.function
        def main(x: R.Tensor((m, n, k), "int8")) -> R.Tensor((m, n, k), "int8"):
            gv: R.Tensor((m, n, k), "int8") = R.tril(x, k=-2)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    k_main = T.dynamic("k", "int64")
    k_tril = T.dynamic("k", "int64")
    m_tril = T.dynamic("m", "int64")
    n_tril = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main, k_main), "int8")) -> R.Tensor((m_main, n_main, k_main), "int8"):
            gv = R.call_tir(Expected.tril, (x,), ty_args=[R.Tensor((m_main, n_main, k_main), dtype="int8")])
            return gv

        @Ts.function(private=True)
        def tril(rxplaceholder: T.Tensor([m_tril, n_tril, k_tril], dtype='int8'), trilu: T.Tensor([m_tril, n_tril, k_tril], dtype='int8')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2 in T.grid(m_tril, n_tril, k_tril):
                with Ts.sblock("trilu"):
                    i0_1, i1_1, i2_1 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    Ts.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(i2_1 + T.int64(2) <= i1_1, rxplaceholder[i0_1, i1_1, i2_1], T.int8(0))
    # fmt: on

    mod = LegalizeOps()(Tril)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_triu():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Triu:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv: R.Tensor((2, 3, 4), "float32") = R.triu(x, k=1)
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv = R.call_tir(Expected.triu, (x,), ty_args=[R.Tensor((2, 3, 4), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def triu(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), trilu: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with Ts.sblock("trilu"):
                    i0_1, i1_1, i2_1 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    Ts.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(i1_1 < i2_1, rxplaceholder[i0_1, i1_1, i2_1], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(Triu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_triu_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")
    k = T.dynamic("k", "int64")

    @tvm_tirx.script.ir_module
    class Triu:
        @R.function
        def main(x: R.Tensor((m, n, k), "int8")) -> R.Tensor((m, n, k), "int8"):
            gv: R.Tensor((m, n, k), "int8") = R.triu(x, k=-2)
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    k_main = T.dynamic("k", "int64")
    k_triu = T.dynamic("k", "int64")
    m_triu = T.dynamic("m", "int64")
    n_triu = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main, k_main), "int8")) -> R.Tensor((m_main, n_main, k_main), "int8"):
            gv = R.call_tir(Expected.triu, (x,), ty_args=[R.Tensor((m_main, n_main, k_main), dtype="int8")])
            return gv

        @Ts.function(private=True)
        def triu(rxplaceholder: T.Tensor([m_triu, n_triu, k_triu], dtype='int8'), trilu: T.Tensor([m_triu, n_triu, k_triu], dtype='int8')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1, i2 in T.grid(m_triu, n_triu, k_triu):
                with Ts.sblock("trilu"):
                    i0_1, i1_1, i2_1 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    Ts.writes(trilu[i0_1, i1_1, i2_1])
                    trilu[i0_1, i1_1, i2_1] = T.Select(i1_1 <= i2_1 + T.int64(2), rxplaceholder[i0_1, i1_1, i2_1], T.int8(0))
    # fmt: on

    mod = LegalizeOps()(Triu)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


##################### Datatype #####################


def test_astype():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Astype:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "int32"):
            gv: R.Tensor((2, 3, 4), "int32") = R.astype(x, "int32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 3, 4), "int32"):
            gv = R.call_tir(Expected.cast, (x,), ty_args=[R.Tensor((2, 3, 4), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def cast(rxplaceholder: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "float32"), compute: T.Tensor((T.int64(2), T.int64(3), T.int64(4)), "int32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with Ts.sblock("compute"):
                    i0_1, i1_1, i2_1 = Ts.axis.remap("SSS", [i0, i1, i2])
                    Ts.reads(rxplaceholder[i0_1, i1_1, i2_1])
                    Ts.writes(compute[i0_1, i1_1, i2_1])
                    compute[i0_1, i1_1, i2_1] = T.Cast("int32", rxplaceholder[i0_1, i1_1, i2_1])
    # fmt: on

    mod = LegalizeOps()(Astype)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_astype_input_constant_scalar():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Astype:
        @R.function
        def main() -> R.Tensor((), "int32"):
            gv: R.Tensor((), "int32") = R.astype(R.const(1.5, "float32"), "int32")
            return gv

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), "int32"):
            gv: R.Tensor((), "int32") = R.const(1, "int32")
            return gv
    # fmt: on

    mod = LegalizeOps()(Astype)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_astype_symbolic():
    # fmt: off
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Astype:
        @R.function
        def main(x: R.Tensor((m, n), "float32")) -> R.Tensor((m, n), "int32"):
            gv: R.Tensor((m, n), "int32") = R.astype(x, "int32")
            return gv

    m_main = T.dynamic("m", "int64")
    n_main = T.dynamic("n", "int64")
    m_cast = T.dynamic("m", "int64")
    n_cast = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((m_main, n_main), "float32")) -> R.Tensor((m_main, n_main), "int32"):
            gv = R.call_tir(Expected.cast, (x,), ty_args=[R.Tensor((m_main, n_main), dtype="int32")])
            return gv

        @Ts.function(private=True)
        def cast(rxplaceholder: T.Tensor([m_cast, n_cast], dtype='float32'), compute: T.Tensor([m_cast, n_cast], dtype='int32')):
            T.func_attr({"tirx.noalias": True})

            for i0, i1 in T.grid(m_cast, n_cast):
                with Ts.sblock("compute"):
                    i0_1, i1_1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(rxplaceholder[i0_1, i1_1])
                    Ts.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.Cast("int32", rxplaceholder[i0_1, i1_1])
    # fmt: on

    mod = LegalizeOps()(Astype)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
