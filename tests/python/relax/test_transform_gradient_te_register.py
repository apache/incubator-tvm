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
"""Unit tests for registering tirx gradient functions in the gradient pass."""

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.training.utils import register_te_gradient
from tvm.relax.transform import Gradient
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.ir.base import assert_structural_equal
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


# Only run once in the whole test session
@pytest.fixture(scope="module")
def register_te_grads():
    # register the gradient function
    @register_te_gradient("f_mul_grad")
    def f_mul_grad(output_grad, src1, src2):
        def mul_grad_1(*idx):
            return src2[idx] * output_grad[idx]

        def mul_grad_2(*idx):
            return src1[idx] * output_grad[idx]

        return [
            tvm.te.compute(src1.shape, mul_grad_1, name="f_mul_grad_1"),
            tvm.te.compute(src2.shape, mul_grad_2, name="f_mul_grad_2"),
        ]

    # register the gradient function
    @register_te_gradient("f_mulk_grad")
    def f_mulk_grad(output_grad, src1, k):
        def mulk_grad(*idx):
            return output_grad[idx] * k

        return [
            tvm.te.compute(src1.shape, mulk_grad, name="f_mulk_grad"),
        ]


def get_expected_1():
    # fmt: off
    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def f_mul(A: T.Tensor((T.int64(5), T.int64(5)), "float32"), B: T.Tensor((T.int64(5), T.int64(5)), "float32"), f_mul_1: T.Tensor((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with Ts.sblock("f_mul"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1], B[v_i0, v_i1])
                    Ts.writes(f_mul_1[v_i0, v_i1])
                    f_mul_1[v_i0, v_i1] = A[v_i0, v_i1] * B[v_i0, v_i1]

        @Ts.function(private=True)
        def f_mul_grad(A: T.Tensor((T.int64(5), T.int64(5)), "float32"), B: T.Tensor((T.int64(5), T.int64(5)), "float32"), C: T.Tensor((T.int64(5), T.int64(5)), "float32"), f_mul_grad_1: T.Tensor((T.int64(5), T.int64(5)), "float32"), f_mul_grad_2: T.Tensor((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with Ts.sblock("f_mul_grad_1"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(C[v_i0, v_i1], A[v_i0, v_i1])
                    Ts.writes(f_mul_grad_1[v_i0, v_i1])
                    f_mul_grad_1[v_i0, v_i1] = C[v_i0, v_i1] * A[v_i0, v_i1]
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with Ts.sblock("f_mul_grad_2"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(B[v_i0, v_i1], A[v_i0, v_i1])
                    Ts.writes(f_mul_grad_2[v_i0, v_i1])
                    f_mul_grad_2[v_i0, v_i1] = B[v_i0, v_i1] * A[v_i0, v_i1]

        @R.function
        def main_adjoint(a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((5, 5), dtype="float32"), R.Tensor((5, 5), dtype="float32"))):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.f_mul, (a, b), ty_args=[R.Tensor((5, 5), dtype="float32")])
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv_adjoint: R.Tensor((5, 5), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([5, 5]))
                lv_1 = R.call_tir(cls.f_mul_grad, (lv_adjoint, a, b), ty_args=[_TupleType([R.Tensor((5, 5), dtype="float32"), R.Tensor((5, 5), dtype="float32")])])
                a_adjoint: R.Tensor((5, 5), dtype="float32") = lv_1[0]
                b_adjoint: R.Tensor((5, 5), dtype="float32") = lv_1[1]
                a_adjoint_out: R.Tensor((5, 5), dtype="float32") = a_adjoint
                b_adjoint_out: R.Tensor((5, 5), dtype="float32") = b_adjoint
                R.output(gv, a_adjoint_out, b_adjoint_out)
            return (gv, (a_adjoint_out, b_adjoint_out))

        @R.function
        def main(a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a, b), ty_args=[R.Tensor((5, 5), dtype="float32")], te_grad_name="f_mul_grad")
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on
    return Expected


def test_emit_te(register_te_grads):
    # Build the target module using emit_te
    def f_mul(src1, src2):
        def mul(*idx):
            return src1[idx] * src2[idx]

        return tvm.te.compute(src1.shape, mul, name="f_mul")

    a = relax.Var("a", relax.TensorType([5, 5], "float32"))
    b = relax.Var("b", relax.TensorType([5, 5], "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [a, b]):
        with bb.dataflow():
            d = bb.emit(
                bb.call_te_with_grad(
                    f_mul, a, b, function_name_hint="f_mul", te_grad_name="f_mul_grad"
                )
            )
            out = bb.emit_output(R.sum(d))
        bb.emit_func_output(out)

    Before = bb.get()
    After = Gradient("main")(Before)
    assert_structural_equal(After, get_expected_1())


def test_call_tir(register_te_grads):
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def f_mul(A: T.Tensor((T.int64(5), T.int64(5)), "float32"), B: T.Tensor((T.int64(5), T.int64(5)), "float32"), f_mul_1: T.Tensor((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with Ts.sblock("f_mul"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1], B[v_i0, v_i1])
                    Ts.writes(f_mul_1[v_i0, v_i1])
                    f_mul_1[v_i0, v_i1] = A[v_i0, v_i1] * B[v_i0, v_i1]

        @R.function
        def main(a: R.Tensor((5, 5), dtype="float32"), b: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a, b), ty_args=[R.Tensor((5, 5), dtype="float32")], te_grad_name="f_mul_grad")
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: off

    After = Gradient("main")(Before)
    assert_structural_equal(After, get_expected_1())


def get_expected_2():
    # fmt: off
    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def f_mul(A: T.Tensor((T.int64(5), T.int64(5)), "float32"), f_mul2: T.Tensor((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with Ts.sblock("f_mul2"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1])
                    Ts.writes(f_mul2[v_i0, v_i1])
                    f_mul2[v_i0, v_i1] = A[v_i0, v_i1] * T.float32(2)

        @Ts.function(private=True)
        def f_mulk_grad(A: T.Tensor((T.int64(5), T.int64(5)), "float32"), B: T.Tensor((T.int64(5), T.int64(5)), "float32"), f_mulk_grad_1: T.Tensor((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with Ts.sblock("f_mulk_grad"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1])
                    Ts.writes(f_mulk_grad_1[v_i0, v_i1])
                    f_mulk_grad_1[v_i0, v_i1] = A[v_i0, v_i1] * T.float32(2)

        @R.function
        def main_adjoint(a: R.Tensor((5, 5), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((5, 5), dtype="float32"))):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.f_mul, (a,), ty_args=[R.Tensor((5, 5), dtype="float32")])
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv_adjoint: R.Tensor((5, 5), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([5, 5]))
                lv_1 = R.call_tir(cls.f_mulk_grad, (lv_adjoint, a), ty_args=[R.Tensor((5, 5), dtype="float32")])
                a_adjoint: R.Tensor((5, 5), dtype="float32") = lv_1
                a_adjoint_out: R.Tensor((5, 5), dtype="float32") = a_adjoint
                R.output(gv, a_adjoint_out)
            return (gv, (a_adjoint_out,))

        @R.function
        def main(a: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a,), ty_args=[R.Tensor((5, 5), dtype="float32")], te_grad_name="f_mulk_grad", te_grad_kwargs={"k": T.float32(2)})
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on
    return Expected


def test_emit_te_kwargs(register_te_grads):
    # Build the target module using emit_te
    def f_mul2(src):
        return tvm.te.compute(src.shape, lambda *idx: src[idx] * T.float32(2), name="f_mul2")

    a = relax.Var("a", relax.TensorType([5, 5], "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [a]):
        with bb.dataflow():
            d = bb.emit(
                bb.call_te_with_grad(
                    f_mul2,
                    a,
                    function_name_hint="f_mul",
                    te_grad_name="f_mulk_grad",
                    te_grad_kwargs={"k": T.float32(2)},
                )
            )
            out = bb.emit_output(R.sum(d))
        bb.emit_func_output(out)

    Before = bb.get()
    After = Gradient("main")(Before)

    assert_structural_equal(After, get_expected_2())


def test_call_tir_kwargs(register_te_grads):
    # fmt: off
    @I.ir_module
    class Before:
        @Ts.function(private=True)
        def f_mul(A: T.Tensor((T.int64(5), T.int64(5)), "float32"), f_mul2: T.Tensor((T.int64(5), T.int64(5)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i0, i1 in T.grid(T.int64(5), T.int64(5)):
                with Ts.sblock("f_mul2"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1])
                    Ts.writes(f_mul2[v_i0, v_i1])
                    f_mul2[v_i0, v_i1] = A[v_i0, v_i1] * T.float32(2)

        @R.function
        def main(a: R.Tensor((5, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Before
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a,), ty_args=[R.Tensor((5, 5), dtype="float32")], te_grad_name="f_mulk_grad", te_grad_kwargs={"k": T.float32(2)})
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on

    After = Gradient("main")(Before)
    assert_structural_equal(After, get_expected_2())


def get_expected_3():
    # fmt: off
    n_f_mul = T.dynamic("n", "int64")
    n_f_mul_grad = T.dynamic("n", "int64")
    n_main_adjoint = T.dynamic("n", "int64")
    n_main = T.dynamic("n", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def f_mul(A: T.Tensor((n_f_mul, n_f_mul), "float32"), B: T.Tensor((n_f_mul, n_f_mul), "float32"), f_mul_1: T.Tensor((n_f_mul, n_f_mul), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for i0, i1 in T.grid(n_f_mul, n_f_mul):
                with Ts.sblock("f_mul"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[v_i0, v_i1], B[v_i0, v_i1])
                    Ts.writes(f_mul_1[v_i0, v_i1])
                    f_mul_1[v_i0, v_i1] = A[v_i0, v_i1] * B[v_i0, v_i1]

        @Ts.function(private=True)
        def f_mul_grad(A: T.Tensor((n_f_mul_grad, n_f_mul_grad), "float32"), B: T.Tensor((n_f_mul_grad, n_f_mul_grad), "float32"), C: T.Tensor((n_f_mul_grad, n_f_mul_grad), "float32"), f_mul_grad_1: T.Tensor((n_f_mul_grad, n_f_mul_grad), "float32"), f_mul_grad_2: T.Tensor((n_f_mul_grad, n_f_mul_grad), "float32")):
            T.func_attr({"tirx.noalias": True})

            # with Ts.sblock("root"):
            for i0, i1 in T.grid(n_f_mul_grad, n_f_mul_grad):
                with Ts.sblock("f_mul_grad_1"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(C[v_i0, v_i1], A[v_i0, v_i1])
                    Ts.writes(f_mul_grad_1[v_i0, v_i1])
                    f_mul_grad_1[v_i0, v_i1] = C[v_i0, v_i1] * A[v_i0, v_i1]
            for i0, i1 in T.grid(n_f_mul_grad, n_f_mul_grad):
                with Ts.sblock("f_mul_grad_2"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(B[v_i0, v_i1], A[v_i0, v_i1])
                    Ts.writes(f_mul_grad_2[v_i0, v_i1])
                    f_mul_grad_2[v_i0, v_i1] = B[v_i0, v_i1] * A[v_i0, v_i1]

        @R.function
        def main_adjoint(a: R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32"), b: R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32")) -> R.Tuple(R.Tensor((), dtype="float32"), R.Tuple(R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32"), R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32"))):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.f_mul, (a, b), ty_args=[R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32")])
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                gv_adjoint: R.Tensor((), dtype="float32") = R.ones(R.shape([]), dtype="float32")
                lv_adjoint: R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32") = R.broadcast_to(gv_adjoint, R.shape([n_main_adjoint, n_main_adjoint]))
                lv_1 = R.call_tir(cls.f_mul_grad, (lv_adjoint, a, b), ty_args=[_TupleType([R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32"), R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32")])])
                a_adjoint: R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32") = lv_1[0]
                b_adjoint: R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32") = lv_1[1]
                a_adjoint_out: R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32") = a_adjoint
                b_adjoint_out: R.Tensor((n_main_adjoint, n_main_adjoint), dtype="float32") = b_adjoint
                R.output(gv, a_adjoint_out, b_adjoint_out)
            return (gv, (a_adjoint_out, b_adjoint_out))

        @R.function
        def main(a: R.Tensor((n_main, n_main), dtype="float32"), b: R.Tensor((n_main, n_main), dtype="float32")) -> R.Tensor((), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv = R.call_tir_with_grad(cls.f_mul, (a, b), ty_args=[R.Tensor((n_main, n_main), dtype="float32")], te_grad_name="f_mul_grad")
                gv: R.Tensor((), dtype="float32") = R.sum(lv, axis=None, keepdims=False)
                R.output(gv)
            return gv
    # fmt: on
    return Expected


def test_tir_var(register_te_grads):
    def f_mul(src1, src2):
        def mul(*idx):
            return src1[idx] * src2[idx]

        return tvm.te.compute(src1.shape, mul, name="f_mul")

    n = tirx.Var("n", "int64")
    a = relax.Var("a", relax.TensorType([n, n], "float32"))
    b = relax.Var("b", relax.TensorType([n, n], "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [a, b]):
        with bb.dataflow():
            d = bb.emit(
                bb.call_te_with_grad(
                    f_mul, a, b, function_name_hint="f_mul", te_grad_name="f_mul_grad"
                )
            )
            out = bb.emit_output(R.sum(d))
        bb.emit_func_output(out)

    Before = bb.get()
    After = Gradient("main")(Before)
    assert_structural_equal(After, get_expected_3())
    relax.analysis.check_well_formed(After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
