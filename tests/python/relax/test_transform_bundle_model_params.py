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
# ruff: noqa: F401

from __future__ import annotations

import pytest

pytest.importorskip("scipy")  # tvm.topi.testing imports scipy

import tvm
import tvm.topi.testing
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T


def test_basic():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            expr = a
            expr = R.add(expr, b)
            expr = R.add(expr, c)
            return expr

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            params: R.Tuple(R.Tensor([16], "float32"), R.Tensor([16], "float32")),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            expr = a
            b = params[0]
            expr = R.add(expr, b)
            c = params[1]
            expr = R.add(expr, c)
            return expr

    mod = Before
    after = relax.transform.BundleModelParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_no_model_params():
    """If all parameters are inputs, model params should be an empty tuple

    This ensures that a caller does not need to check whether the
    model has compile-time inputs, and can instead provide the output
    of a lifted parameter transformation in all cases, even if that
    transformation returns an empty tuple.
    """

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 3})
            expr = a
            expr = R.add(expr, b)
            expr = R.add(expr, c)
            return expr

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
            params: R.Tuple(),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 3})
            expr = a
            expr = R.add(expr, b)
            expr = R.add(expr, c)
            return expr

    mod = Before
    after = relax.transform.BundleModelParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_dataflow():
    """Parameters can be substituted into a dataflow block"""

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                expr = a
                expr = R.add(expr, b)
                expr = R.add(expr, c)
                R.output(expr)
            return expr

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            params: R.Tuple(R.Tensor([16], "float32"), R.Tensor([16], "float32")),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                expr = a
                b = params[0]
                expr = R.add(expr, b)
                c = params[1]
                expr = R.add(expr, c)
                R.output(expr)
            return expr

    mod = Before
    after = relax.transform.BundleModelParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_variable_names():
    """Parameters retain their names within the updated function

    For readability, the parameter names should be used to generate
    the new variable names.

    Like `test_basic`, but explicitly checks the names of bound
    variables.
    """

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            expr = a
            expr = R.add(expr, b)
            expr = R.add(expr, c)
            return expr

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            params: R.Tuple(R.Tensor([16], "float32"), R.Tensor([16], "float32")),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            expr = a
            b = params[0]
            expr = R.add(expr, b)
            c = params[1]
            expr = R.add(expr, c)
            return expr

    mod = Before
    after = relax.transform.BundleModelParams()(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)

    for binding, expected_binding in zip(
        after["main"].body.blocks[0].bindings,
        Expected["main"].body.blocks[0].bindings,
    ):
        assert binding.var.name == expected_binding.var.name


def test_bundled_param_name():
    """The tuple parameter can have an explicit name"""

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            b: R.Tensor([16], "float32"),
            c: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            expr = a
            expr = R.add(expr, b)
            expr = R.add(expr, c)
            return expr

    @tvm_tirx.script.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor([16], "float32"),
            custom_tuple_name: R.Tuple(R.Tensor([16], "float32"), R.Tensor([16], "float32")),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})
            expr = a
            b = custom_tuple_name[0]
            expr = R.add(expr, b)
            c = custom_tuple_name[1]
            expr = R.add(expr, c)
            return expr

    mod = Before
    after = relax.transform.BundleModelParams("custom_tuple_name")(mod)
    tvm_tirx.ir.assert_structural_equal(after, Expected)

    for param, expected_param in zip(after["main"].params, Expected["main"].params):
        assert param.name == expected_param.name


def test_nested_function_preserves_outer_rewrite_context():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            weight: R.Tensor([16], "float32"),
        ) -> R.Tensor([16], "float32"):
            R.func_attr({"num_input": 1})

            @R.function
            def inner(
                inner_x: R.Tensor([16], "float32"),
                inner_weight: R.Tensor([16], "float32"),
            ) -> R.Tensor([16], "float32"):
                R.func_attr({"num_input": 1})
                return R.add(inner_x, inner_weight)

            return R.add(x, weight)

    after = relax.transform.BundleModelParams()(Before)
    relax.analysis.check_well_formed(after)

    func = after["main"]
    assert len(func.params) == 2
    assert isinstance(func.params[1].ty, relax.TupleType)


def test_bundled_tensor_preserves_retained_shape_dependency():
    x = relax.Var("x", relax.TensorType(dtype="float32", ndim=1))
    shape = relax.Var("shape", relax.ShapeType(ndim=1))
    weight = relax.Var("weight", relax.TensorType(shape, "float32"))

    bb = relax.BlockBuilder()
    with bb.function("main", [x, shape, weight]):
        with bb.dataflow():
            out = bb.emit(relax.op.add(x, weight))
            output = bb.emit_output(out)
        bb.emit_func_output(output)
    before = bb.get()
    global_var = before.get_global_var("main")
    before.update_func(global_var, before[global_var].with_attr("num_input", 2))
    relax.analysis.check_well_formed(before)

    after = relax.transform.BundleModelParams()(before)
    relax.analysis.check_well_formed(after)

    func = after[global_var]
    bundled_ty = func.params[2].ty
    assert isinstance(bundled_ty, relax.TupleType)
    assert isinstance(bundled_ty.fields[0], relax.TensorType)
    assert bundled_ty.fields[0].shape.same_as(func.params[1])


def test_primitive_model_param_is_materialized_outside_if_branches():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            cond: R.Tensor((), "bool"),
            extent: T.int64,
            weight: R.Tensor([extent], "float32"),  # noqa: F821
        ) -> R.Tensor([extent], "float32"):  # noqa: F821
            R.func_attr({"num_input": 1})
            if cond:
                out = R.add(weight, weight)
            else:
                out = R.multiply(weight, weight)
            return out

    after = relax.transform.BundleModelParams()(Before)
    relax.analysis.check_well_formed(after)

    func = after["main"]
    extent_binding = func.body.blocks[0].bindings[0]
    assert isinstance(extent_binding, relax.VarBinding)
    assert tvm_tirx.ir.is_prim_var(extent_binding.var)
    if_expr = next(
        binding.value
        for block in func.body.blocks
        for binding in block.bindings
        if isinstance(binding, relax.VarBinding) and isinstance(binding.value, relax.IfExpr)
    )
    for branch in [if_expr.true_branch, if_expr.false_branch]:
        match_cast = next(
            binding
            for block in branch.blocks
            for binding in block.bindings
            if isinstance(binding, relax.MatchCast)
        )
        assert match_cast.ty.shape[0].same_as(extent_binding.var)


def test_primitive_model_param_remains_linked_to_dependent_tensor():
    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(dtype="float32", ndim=1),
            extent: T.int64,
            weight: R.Tensor([extent], "float32"),
        ):
            R.func_attr({"num_input": 1})
            out = R.add(x, weight)
            return out

    old_extent = Before["main"].params[1]
    after = relax.transform.BundleModelParams()(Before)
    relax.analysis.check_well_formed(after)

    func = after["main"]
    bundled_ty = func.params[1].ty
    assert isinstance(bundled_ty, relax.TupleType)
    assert isinstance(bundled_ty.fields[0], tvm_tirx.ir.PrimType)
    assert isinstance(bundled_ty.fields[1], relax.TensorType)
    assert bundled_ty.fields[1].shape is None
    assert bundled_ty.fields[1].ndim == 1

    bindings = list(func.body.blocks[0].bindings)
    extent_index = next(
        i
        for i, binding in enumerate(bindings)
        if isinstance(binding, relax.VarBinding) and tvm_tirx.ir.is_prim_var(binding.var)
    )
    match_cast_index = next(
        i for i, binding in enumerate(bindings) if isinstance(binding, relax.MatchCast)
    )
    extent = bindings[extent_index].var
    weight = bindings[match_cast_index]
    assert extent_index < match_cast_index
    assert weight.ty.shape[0].same_as(extent)

    signature_types = [param.ty for param in func.params] + [func.ret_ty]
    assert all(
        all(not var.same_as(old_extent) for var in relax.analysis.tir_vars_in_type(ty))
        for ty in signature_types
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
