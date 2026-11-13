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
# ruff: noqa: F841

"""Test FNormalize usage"""

import pytest
import tvm_ffi

import tvm
import tvm.relax.testing.transform
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.s_tir import script as Ts
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.script.parser import tirx as T


@pytest.fixture(params=[True], scope="session")
def define_normalization(request):
    return request.param


@pytest.fixture
def custom_op(define_normalization):
    """A custom operator for testing purposes

    The custom operator ignores its second argument.  If there isn't a
    second argument which can be ignored, FNormalize appends an
    additional argument so that it can be properly ignored.
    """

    op_name = "custom_op.ignore_second_argument"

    def infer_ty(call: relax.Call, context: relax.BlockBuilder):
        return call.args[0].ty

    def normalize(context: relax.BlockBuilder, call: relax.Call):
        if len(call.args) == 1:
            return relax.Call(call.op, [call.args[0], relax.Tuple([])])
        else:
            return call

    def legalize(context: relax.BlockBuilder, call: relax.Call):
        return call.args[0]

    op_attrs = {
        "relax.FInferTypeWithBuilder": infer_ty,
        "FLegalize": legalize,
        "FPurity": True,
    }
    if define_normalization:
        op_attrs["FNormalize"] = normalize

    if op_name not in tvm_tirx.ir.Op.list_op_names():
        tvm_tirx.ir._ffi_api.RegisterOp(op_name, "Custom Relax test operator")
    for key, value in op_attrs.items():
        tvm_tirx.ir.register_op_attr(op_name, key, value)

    op = tvm_tirx.ir.Op.get(op_name)
    yield op

    for key in op_attrs:
        op.reset_attr(key)


def test_normalization_suppressed_for_tvmscript(custom_op):
    """FNormalize isn't applied when parsing TVMScript

    TVMScript should be able to produce un-normalized Relax IR for
    specifying test cases if the well-formed check is disabled,
    and to ensure that no changes occur when performing a round-trip
    through TVMScript.
    """

    @R.function(check_well_formed=False)
    def func(A: R.Tensor):
        return relax.Call(custom_op, [A])

    call_expr = func.body.blocks[0].bindings[0].value
    assert isinstance(call_expr, relax.Call), (
        "Test implementation error, didn't extract the correct expression"
    )
    assert len(call_expr.args) == 1, (
        "Expected TVMScript to suppress use of FNormalize, produce arguments as written"
    )


@pytest.mark.skip_well_formed_check_before_transform
def test_normalization_applied_during_cpp_mutator(custom_op):
    """FNormalize is applied by relax::ExprMutator subclasses"""

    @I.ir_module(check_well_formed=False)
    class Before:
        @R.function
        def main(A: R.Tensor):
            return relax.Call(custom_op, [A])

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor):
            return relax.Call(custom_op, [A, R.tuple()])

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm_ffi.structural_equal(Before, After)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


def test_normalization_applied_during_python_mutator(custom_op):
    """FNormalize is applied by relax.ExprMutator subclasses"""

    @R.function(private=True, check_well_formed=False)
    def before(A: R.Tensor):
        return relax.Call(custom_op, [A])

    @R.function(private=True)
    def expected(A: R.Tensor):
        return relax.Call(custom_op, [A, R.tuple()])

    @relax.expr_functor.mutator
    class EmptyPyExprMutator(relax.PyExprMutator):
        """Default ExprMutator"""

    after = EmptyPyExprMutator().visit_expr(before)

    assert not tvm_ffi.structural_equal(before, after)
    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_normalized_call_node_is_well_formed(custom_op):
    """If FNormalize wouldn't apply a change, the IR is well-formed"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor):
            return relax.Call(custom_op, [A, A])

    relax.analysis.check_well_formed(Module)


@pytest.mark.skip_well_formed_check_before_transform
@pytest.mark.parametrize("define_normalization", [True, False])
def test_un_normalized_call_node_is_ill_formed(custom_op, define_normalization):
    """If FNormalize would apply a change, the IR is ill-formed

    This only applies if FNormalize exists.  An operator without
    FNormalize has no corresponding check applied.
    """

    @I.ir_module(check_well_formed=False)
    class Module:
        @R.function
        def main(A: R.Tensor):
            return relax.Call(custom_op, [A])

    if define_normalization:
        with pytest.raises((ValueError, TypeError)):
            relax.analysis.check_well_formed(Module)
    else:
        relax.analysis.check_well_formed(Module)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_to_inline_tuple_for_call_tir(custom_op):
    """FNormalize in-lines the argument tuple for R.call_tir"""

    @I.ir_module(check_well_formed=False)
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Before
            args = (A,)
            return relax.Call(
                tvm_tirx.ir.Op.get("relax.call_tir"),
                [cls.multiply_by_two, args],
                ty_args=[A.ty],
            )

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Expected
            args = (A,)
            return relax.Call(
                tvm_tirx.ir.Op.get("relax.call_tir"),
                [cls.multiply_by_two, relax.Tuple([A])],
                ty_args=[A.ty],
            )

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm_ffi.structural_equal(Before, After)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_argument_to_inline_tuple_for_call_tir(custom_op):
    """FNormalize in-lines the argument tuple for R.call_tir

    Like `test_normalize_to_inline_tuple_for_call_tir`, but the
    argument tuple is provided as a relax function argument.
    """

    @I.ir_module(check_well_formed=False)
    class Before:
        @R.function
        def main(args: R.Tuple([R.Tensor([16], "float32")])):
            cls = Before
            return relax.Call(
                tvm_tirx.ir.Op.get("relax.call_tir"),
                [cls.multiply_by_two, args],
                ty_args=[args[0].ty],
            )

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

    @I.ir_module
    class Expected:
        @R.function
        def main(args: R.Tuple([R.Tensor([16], "float32")])):
            cls = Expected
            return relax.Call(
                tvm_tirx.ir.Op.get("relax.call_tir"),
                [cls.multiply_by_two, relax.Tuple([args[0]])],
                ty_args=[args[0].ty],
            )

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm_ffi.structural_equal(Before, After)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_to_inline_tuple_for_call_tir_inplace(custom_op):
    """FNormalize in-lines the argument tuple for R.call_tir_inplace"""

    # The CallTIRInplaceAttrs is difficult to construct in the Python
    # API, so it is more convenient to declare the expected one first
    # and reuse its attributes
    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Expected
            args = (A,)
            return R.call_tir_inplace(
                cls.multiply_by_two,
                A,
                inplace_indices=[0],
                ty_args=[A.ty],
            )

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor(16, "float32")):
            for i in range(16):
                A[i] = A[i] * 2.0

    inplace_attrs = Expected["main"].body.blocks[0].bindings[1].value.attrs

    @I.ir_module(check_well_formed=False)
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Before
            args = (A,)
            return relax.Call(
                tvm_tirx.ir.Op.get("relax.call_tir_inplace"),
                [cls.multiply_by_two, args],
                attrs=inplace_attrs,
                ty_args=[A.ty],
            )

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor(16, "float32")):
            for i in range(16):
                A[i] = A[i] * 2.0

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm_ffi.structural_equal(Before, After)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


@pytest.mark.skip_well_formed_check_before_transform
def test_normalize_to_inline_tuple_for_call_tir_with_grad(custom_op):
    """FNormalize in-lines the argument tuple for R.call_tir_with_grad"""

    # The CallTIRWithGradAttrs is difficult to construct in the Python
    # API, so it is more convenient to declare the expected one first
    # and reuse its attributes
    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Expected
            args = (A,)
            return R.call_tir_with_grad(
                cls.multiply_by_two,
                A,
                ty_args=[A.ty],
                te_grad_name="f_grad",
            )

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

        @Ts.function(private=True)
        def f_grad(
            A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32"), Grad: T.Tensor(16, "float32")
        ):
            for i in range(16):
                Grad[i] = 2.0

    with_grad_attrs = Expected["main"].body.blocks[0].bindings[1].value.attrs

    @I.ir_module(check_well_formed=False)
    class Before:
        @R.function
        def main(A: R.Tensor([16], "float32")):
            cls = Before
            args = (A,)
            return relax.Call(
                tvm_tirx.ir.Op.get("relax.call_tir_with_grad"),
                [cls.multiply_by_two, args],
                attrs=with_grad_attrs,
                ty_args=[A.ty],
            )

        @Ts.function(private=True)
        def multiply_by_two(A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32")):
            for i in range(16):
                B[i] = A[i] * 2.0

        @Ts.function(private=True)
        def f_grad(
            A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32"), Grad: T.Tensor(16, "float32")
        ):
            for i in range(16):
                Grad[i] = 2.0

    After = tvm.relax.testing.transform.ApplyEmptyCppMutator()(Before)

    assert not tvm_ffi.structural_equal(Before, After)
    tvm_tirx.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm_tirx.testing.main()
