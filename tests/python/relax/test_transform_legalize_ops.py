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

from __future__ import annotations

import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.transform import LegalizeOps
from tvm.relax.transform.legalize_ops.common import register_legalize
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_customize_legalize():
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
            cls = Expected
            gv = R.call_tir(cls.add, (y, x), ty_args=[R.Tensor((4, 3, 2, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def add(rxplaceholder_1: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(1)), "float32"), rxplaceholder: T.Tensor((T.int64(1), T.int64(2), T.int64(3)), "float32"), T_add: T.Tensor((T.int64(4), T.int64(3), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(3), T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    ax0, ax1, ax2, ax3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
                    Ts.reads(rxplaceholder_1[ax0, ax1, ax2, T.int64(0)], rxplaceholder[T.int64(0), ax2, ax3])
                    Ts.writes(T_add[ax0, ax1, ax2, ax3])
                    T_add[ax0, ax1, ax2, ax3] = rxplaceholder_1[ax0, ax1, ax2, T.int64(0)] + rxplaceholder[T.int64(0), ax2, ax3]
    # fmt: on

    def customize_legalize_add(bb: relax.BlockBuilder, call: relax.Call):
        from tvm import topi  # pylint: disable=import-outside-toplevel

        return bb.call_te(topi.add, call.args[1], call.args[0])

    mod = LegalizeOps({"relax.add": customize_legalize_add})(Add)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def test_legalize_multiple_types_of_call():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def mul2(x: R.Tensor((3, 3), "float32")):
            gv = R.multiply(x, R.const(2.0, "float32"))
            return gv

        @Ts.function(private=True)
        def identity(rxplaceholder: T.Tensor((T.int64(3), T.int64(3)), "float32"), T_id: T.Tensor((T.int64(3), T.int64(3)), "float32")):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1])
                    Ts.writes(T_id[v_ax0, v_ax1])
                    T_id[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1]

        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            cls = Before
            gv: R.Tensor((3, 3), "float32") = cls.mul2(x)
            gv1 = R.call_tir(cls.identity, gv, ty_args=[R.Tensor((3, 3), dtype="float32")])
            gv2 = R.multiply(gv1, R.const(2.0, "float32"))
            return gv2

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def mul2(x: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.multiply, (x,), ty_args=[R.Tensor((3, 3), dtype="float32")])
            return gv

        @Ts.function(private=True)
        def identity(rxplaceholder: T.Tensor((T.int64(3), T.int64(3)), "float32"), T_id: T.Tensor((T.int64(3), T.int64(3)), "float32")):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1])
                    Ts.writes(T_id[v_ax0, v_ax1])
                    T_id[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1]

        @Ts.function(private=True)
        def multiply(rxplaceholder: T.Tensor((T.int64(3), T.int64(3)), "float32"), T_multiply: T.Tensor((T.int64(3), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] * T.float32(2)

        @R.function
        def main(x1: R.Tensor((3, 3), dtype="float32")) -> R.Tensor((3, 3), dtype="float32"):
            cls = Expected
            gv1: R.Tensor((3, 3), dtype="float32") = cls.mul2(x1)
            gv11 = R.call_tir(cls.identity, gv1, ty_args=[R.Tensor((3, 3), dtype="float32")])
            gv2 = R.call_tir(cls.multiply, (gv11,), ty_args=[R.Tensor((3, 3), dtype="float32")])
            return gv2
    # fmt: on

    After = LegalizeOps()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


def test_can_not_legalize():
    # case 1: does't have legalization
    add_legalize = tvm_tirx.ir.Op.get("relax.add").get_attr("FLegalize")
    # reset it for test
    tvm_tirx.ir.Op.get("relax.add").reset_attr("FLegalize")

    # fmt: off
    @tvm_tirx.script.ir_module
    class Before0:
        @R.function
        def main(x: R.Tensor((3, 3), "float32")):
            gv: R.Tensor((3, 3), "float32") = R.add(x, x)
            return gv
    # fmt: on
    After0 = LegalizeOps()(Before0)
    tvm_tirx.ir.assert_structural_equal(After0, Before0)

    register_legalize("relax.add", add_legalize)

    # case 2: don't know all shape
    s = relax.Var("s", relax.ShapeType((3, 3)))
    x = relax.Var("x", relax.TensorType((3, 3), "float32"))
    y = relax.Var("y", relax.TensorType(s, "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            gv = bb.emit_output(R.add(x, y))
        bb.emit_func_output(gv)
    Before1 = bb.get()
    After1 = LegalizeOps()(Before1)
    tvm_tirx.ir.assert_structural_equal(After1, Before1)


def test_legalize_scalar_data_type_preserve():
    # fmt: off
    @tvm_tirx.script.ir_module
    class Before0:
        @R.function
        def main(x: R.Tensor((3, 3), "float16")):
            gv: R.Tensor((3, 3), "float16") = R.multiply(x, R.const(1.14514, "float16"))
            return gv

    @tvm_tirx.script.ir_module
    class Before1:
        @R.function
        def main(x: R.Tensor((3, 3), "uint8")):
            gv: R.Tensor((3, 3), "uint8") = R.multiply(x, R.const(2, "uint8"))
            return gv

    @tvm_tirx.script.ir_module
    class Before2:
        @R.function
        def main(x: R.Tensor((3, 3), "bool")):
            gv: R.Tensor((3, 3), "bool") = R.equal(x, R.const(True, "bool"))
            return gv

    @tvm_tirx.script.ir_module
    class Expected0:
        @Ts.function(private=True)
        def multiply(
            rxplaceholder: T.Tensor((T.int64(3), T.int64(3)), "float16"),
            T_multiply: T.Tensor((T.int64(3), T.int64(3)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] * T.float16(
                        1.1455078125
                    )

        @R.function
        def main(x: R.Tensor((3, 3), dtype="float16")) -> R.Tensor((3, 3), dtype="float16"):
            cls = Expected0
            gv = R.call_tir(cls.multiply, (x,), ty_args=[R.Tensor((3, 3), dtype="float16")])
            return gv

    @tvm_tirx.script.ir_module
    class Expected1:
        @Ts.function(private=True)
        def multiply(
            rxplaceholder: T.Tensor((T.int64(3), T.int64(3)), "uint8"),
            T_multiply: T.Tensor((T.int64(3), T.int64(3)), "uint8"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with Ts.sblock("T_multiply"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1])
                    Ts.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] * T.uint8(2)

        @R.function
        def main(x: R.Tensor((3, 3), dtype="uint8")) -> R.Tensor((3, 3), dtype="uint8"):
            cls = Expected1
            gv = R.call_tir(cls.multiply, (x,), ty_args=[R.Tensor((3, 3), dtype="uint8")])
            return gv

    @tvm_tirx.script.ir_module
    class Expected2:
        @Ts.function(private=True)
        def equal(
            rxplaceholder: T.Tensor((T.int64(3), T.int64(3)), "bool"),
            T_equal: T.Tensor((T.int64(3), T.int64(3)), "bool"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(3), T.int64(3)):
                with Ts.sblock("T_equal"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(rxplaceholder[v_ax0, v_ax1])
                    Ts.writes(T_equal[v_ax0, v_ax1])
                    T_equal[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] == tvm_tirx.tirx.const(True, "bool")

        @R.function
        def main(x: R.Tensor((3, 3), dtype="bool")) -> R.Tensor((3, 3), dtype="bool"):
            cls = Expected2
            gv = R.call_tir(cls.equal, (x,), ty_args=[R.Tensor((3, 3), dtype="bool")])
            return gv
    # fmt: on

    After0 = LegalizeOps()(Before0)
    tvm_tirx.ir.assert_structural_equal(After0, Expected0)
    After1 = LegalizeOps()(Before1)
    tvm_tirx.ir.assert_structural_equal(After1, Expected1)
    After2 = LegalizeOps()(Before2)
    tvm_tirx.ir.assert_structural_equal(After2, Expected2)


def test_matmul_legalization_requires_known_dtype():
    @I.ir_module
    class ArbitraryDtype:
        @R.function
        def main(A: R.Tensor([16, 32]), B: R.Tensor([32, 8])) -> R.Tensor([16, 8]):
            return R.matmul(A, B)

    with pytest.raises(AssertionError) as err:
        LegalizeOps()(ArbitraryDtype)

    # This error should be caught while attempting to legalize the
    # R.matmul, where we can present a user-friendly error.
    # Otherwise, the error isn't caught until the implementation of
    # `BlockBuilder.call_te`, when attempting to create a numeric
    # constant of type kHandle, which produces a much less
    # user-friendly error.
    err_message = err.value.args[0]
    assert err_message.startswith("To legalize R.matmul")


@pytest.fixture(
    params=[pytest.param(False, id="return_relax_expr"), pytest.param(True, id="return_relax_var")],
    scope="session",
)
def emit_legalization_through_builder(request):
    return request.param


@pytest.fixture
def custom_op(emit_legalization_through_builder):
    op_name = "custom_op.matmul_bias_add"

    def infer_ty(call: relax.Call, context):
        activations, weight, bias = call.args

        matmul_call = relax.op.matmul(activations, weight)
        matmul_ty = tvm_tirx.ir.Op.get("relax.matmul").get_attr("relax.FInferTypeWithBuilder")(
            matmul_call, context
        )

        matmul_var = relax.Var("dummy_var", matmul_ty)
        add_call = matmul_var + bias
        add_ty = tvm_tirx.ir.Op.get("relax.add").get_attr("relax.FInferTypeWithBuilder")(
            add_call, context
        )

        return add_ty

    def legalize(bb: relax.BlockBuilder, call: relax.Call):
        activations, weight, bias = call.args
        legalized = relax.op.matmul(activations, weight) + bias
        if emit_legalization_through_builder:
            legalized = bb.emit(legalized)
        return legalized

    op_attrs = {
        "relax.FInferTypeWithBuilder": infer_ty,
        "FLegalize": legalize,
        "FPurity": True,
    }

    if op_name not in tvm_tirx.ir.Op.list_op_names():
        tvm_tirx.ir._ffi_api.RegisterOp(op_name, "Custom Relax test operator")
    for key, value in op_attrs.items():
        tvm_tirx.ir.register_op_attr(op_name, key, value)

    op = tvm_tirx.ir.Op.get(op_name)
    yield op

    for key in op_attrs:
        op.reset_attr(key)


def test_recursive_legalization(custom_op):
    """Legalization of an operator may produce new operators requiring legalization"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor([16, 32, 64], "float32"),
            Weight: R.Tensor([64, 128], "float32"),
            Bias: R.Tensor([16, 32, 128], "float32"),
        ):
            return relax.Call(custom_op, [A, Weight, Bias])

    AfterFirstIter = LegalizeOps()(Before)
    AfterSecondIter = LegalizeOps()(AfterFirstIter)

    # After LegalizeOps, the custom operation should be replaced by
    # `R.matmul` and `R.add`, which should in turn be replaced with
    # TIR implementations.  Therefore, the second application of
    # LegalizeOps() should be a no-op.
    tvm_tirx.ir.assert_structural_equal(AfterFirstIter, AfterSecondIter)


def test_legalize_with_vdevice():
    """Legalization may generate kernels for multiple targets

    This is a regression test.  In previous implementations, Relax
    expressions whose argument types differed only by their `vdevice`
    would be legalized to use the same `Function`.

    """

    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [R.vdevice("llvm")]})

        @R.function
        def func_cuda(A: R.Tensor([32, 32], "float32"), B: R.Tensor([32, 32], "float32")):
            C = R.add(A, B)
            return C

        @R.function
        def func_llvm(
            A: R.Tensor([32, 32], "float32", "llvm"), B: R.Tensor([32, 32], "float32", "llvm")
        ):
            C = R.add(A, B)
            return C

    @I.ir_module
    class Expected:
        I.module_global_infos({"vdevice": [R.vdevice("llvm")]})

        @R.function
        def func_cuda(
            A: R.Tensor((32, 32), dtype="float32"),
            B: R.Tensor((32, 32), dtype="float32"),
        ):
            cls = Expected
            C = R.call_tir(cls.add, (A, B), ty_args=[R.Tensor((32, 32), dtype="float32")])
            return C

        @Ts.function(private=True)
        def add(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for (*iters,) in T.grid(T.int64(32), T.int64(32)):
                with Ts.sblock("T_add"):
                    ax0, ax1 = Ts.axis.remap("SS", iters)
                    C[ax0, ax1] = A[ax0, ax1] + B[ax0, ax1]

        @R.function
        def func_llvm(
            A: R.Tensor((32, 32), dtype="float32", vdevice="llvm"),
            B: R.Tensor((32, 32), dtype="float32", vdevice="llvm"),
        ):
            cls = Expected
            C = R.call_tir(
                cls.add_llvm,
                (A, B),
                ty_args=[R.Tensor((32, 32), dtype="float32", vdevice="llvm")],
            )
            return C

        @Ts.function(private=True)
        def add_llvm(
            A: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            B: T.Tensor((T.int64(32), T.int64(32)), "float32"),
            C: T.Tensor((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"target": T.target("llvm"), "tirx.noalias": True})
            for (*iters,) in T.grid(T.int64(32), T.int64(32)):
                with Ts.sblock("T_add"):
                    ax0, ax1 = Ts.axis.remap("SS", iters)
                    C[ax0, ax1] = A[ax0, ax1] + B[ax0, ax1]

    After = tvm.relax.transform.LegalizeOps(target=tvm_tirx.target.Target("cuda"))(Before)

    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_explicit_legalize_target_and_vdevice_precedence():
    @I.ir_module
    class Module:
        I.module_global_infos({"vdevice": [R.vdevice("llvm")]})

        @R.function
        def default(x: R.Tensor((4,), "float32")):
            return R.add(x, x)

        @R.function
        def annotated(x: R.Tensor((4,), "float32", "llvm")):
            return R.add(x, x)

    targets = []

    def record_target(bb, call):
        targets.append(bb.target.kind.name)
        return call

    cuda_pass = tvm.relax.transform.LegalizeOps({"relax.add": record_target}, target="cuda")
    llvm_pass = tvm.relax.transform.LegalizeOps({"relax.add": record_target}, target="llvm")
    cuda_pass(Module)
    assert sorted(targets) == ["cuda", "llvm"]
    targets.clear()
    llvm_pass(Module)
    assert targets == ["llvm", "llvm"]


if __name__ == "__main__":
    tvm_tirx.testing.main()
