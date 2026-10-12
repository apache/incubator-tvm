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

import gc
import sys
import tempfile
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

m = T.dynamic("m", "int64")
n = T.dynamic("n", "int64")


@tvm_tirx.script.ir_module
class InputModule:
    @R.function
    def foo(x: R.Tensor((m, n), "int64")):
        y = R.unique(x, sorted=False)
        y_sorted = R.unique(x)
        return y, y_sorted


def run_cpu(mod, func_name, *args):
    if isinstance(mod, relax.Function):
        func = mod
        args = [func_name, *args]
        func_name = func.attrs["global_symbol"]
        mod = tvm_tirx.IRModule.from_expr(func)

    target = tvm_tirx.target.Target("llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    return vm[func_name](*args)


def test_unique():
    # TODO(prakalp): also add test for compiling and running on CUDA device.
    data_numpy = np.random.randint(0, 16, (16, 16))
    data = tvm_tirx.runtime.tensor(data_numpy)
    result, result_sorted = run_cpu(InputModule, "foo", data)

    expected_output_sorted, indices = np.unique(data_numpy, return_index=True)
    expected_output = [data_numpy.flatten()[index] for index in sorted(indices)]

    np.testing.assert_array_equal(expected_output_sorted, result_sorted.numpy())
    np.testing.assert_array_equal(expected_output, result.numpy())


@tvm_tirx.script.ir_module
class PrintTest:
    @R.function(pure=False)
    def foo(x: R.Tensor((), "int32")):
        # results have to be bound, but we don't use them
        # TODO: We should allow calls whose results are not bound for side effects;
        #       it would be easy syntactic sugar to add.
        p1 = R.print("", x)
        p2 = R.print("Number: {}", x)
        t = (x, x)
        p3 = R.print("Tuple: {}", t)
        p4 = R.print("", x, t)
        p5 = R.print("Custom print: {} {}", x, x)
        p6 = R.print("Another print: {} {}", x, t)
        return x


def test_print():
    try:
        stdout = sys.stdout
        with tempfile.TemporaryFile(mode="w+") as test_out:
            sys.stdout = test_out
            run_cpu(
                PrintTest,
                "foo",
                tvm_tirx.runtime.tensor(np.array(1).astype("int32")),
            )
            test_out.seek(0)
            printed_text = str(test_out.read())
            expected = "1\nNumber: 1\nTuple: (1, 1)\n1 (1, 1)\nCustom print: 1 1\nAnother print: 1 (1, 1)\n"
            assert printed_text in expected, ("printed_text is ", printed_text)
    finally:
        sys.stdout = stdout


def test_assert_passes():
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(True))
        return x

    run_cpu(func, tvm_tirx.runtime.tensor(np.array(1).astype("int32")))


def test_assert_passes_with_format_args():
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(True), "You won't see me", x)
        return x

    run_cpu(func, tvm_tirx.runtime.tensor(np.array(1).astype("int32")))


def test_assert_fails():
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(False))
        return x

    with pytest.raises(AssertionError, match="Assertion Failed"):
        run_cpu(func, tvm_tirx.runtime.tensor(np.array(1).astype("int32")))


def test_assert_fails_with_message():
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(False), "I failed...")
        return x

    with pytest.raises(AssertionError, match="I failed..."):
        run_cpu(func, tvm_tirx.runtime.tensor(np.array(1).astype("int32")))


def test_assert_fails_with_args():
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(False), "", x, x)
        return x

    with pytest.raises(AssertionError, match="5, 5"):
        run_cpu(func, tvm_tirx.runtime.tensor(np.array(5).astype("int32")))


def test_assert_fails_with_formatted_args():
    @R.function(pure=False)
    def func(x: R.Tensor((), "int32")):
        _ = R.assert_op(relax.const(False), "Number: {}", x)
        return x

    with pytest.raises(AssertionError, match="Number: 6"):
        run_cpu(func, tvm_tirx.runtime.tensor(np.array(6).astype("int32")))


def test_assert_on_argument_passes():
    @R.function(pure=False)
    def func(condition: R.Tensor((), "bool"), x: R.Tensor((), "int32")):
        _ = R.assert_op(condition)
        return x

    condition = tvm_tirx.runtime.tensor(np.array(True))
    x = tvm_tirx.runtime.tensor(np.array(5).astype("int32"))
    run_cpu(func, condition, x)


def test_assert_on_argument_fails():
    @R.function(pure=False)
    def func(condition: R.Tensor((), "bool"), x: R.Tensor((), "int32")):
        _ = R.assert_op(condition)
        return x

    condition = tvm_tirx.runtime.tensor(np.array(False))
    x = tvm_tirx.runtime.tensor(np.array(5).astype("int32"))
    with pytest.raises(AssertionError):
        run_cpu(func, condition, x)


def test_assert_on_symbolic_var_passes():
    N = T.dynamic("N", "int64")

    @R.function(pure=False)
    def func(x: R.Tensor([N], "int32")):
        _ = R.assert_op(R.prim_value(N % 8 == 0))
        return x

    x = tvm_tirx.runtime.tensor(np.arange(8, dtype="int32"))
    run_cpu(func, x)


def test_assert_on_symbolic_var_fails():
    N = T.dynamic("N", "int64")

    @R.function(pure=False)
    def func(x: R.Tensor([N], "int32")):
        _ = R.assert_op(R.prim_value(N % 8 == 0))
        return x

    x = tvm_tirx.runtime.tensor(np.arange(10, dtype="int32"))
    with pytest.raises(AssertionError):
        run_cpu(func, x)


@tvm_tirx.script.ir_module
class ShapeOfTest:
    @R.function
    def get_shape(t: R.Tensor(ndim=-1, dtype="int32")) -> R.Shape(ndim=-1):
        return R.shape_of(t)

    @R.function
    def get_constrained_shape(t: R.Tensor(ndim=1, dtype="int32")) -> R.Shape(ndim=1):
        # require the input tensor to have rank 1
        return R.shape_of(t)

    @R.function
    def get_scalar_shape() -> R.Shape(()):
        x: R.Tensor((), "int32") = R.const(1, dtype="int32")
        return R.shape_of(x)

    @R.function
    def get_constant_shape() -> R.Shape((2, 2)):
        x: R.Tensor((2, 2), "int32") = R.const(
            np.array([[1, 2], [3, 4]], dtype="int32"), dtype="int32"
        )
        return R.shape_of(x)


def test_op_shape_of():
    unit_shape = run_cpu(ShapeOfTest, "get_scalar_shape")
    assert unit_shape == tvm_ffi.Shape([])

    const_shape = run_cpu(ShapeOfTest, "get_constant_shape")
    assert const_shape == tvm_ffi.Shape([2, 2])

    scalar_shape = run_cpu(
        ShapeOfTest,
        "get_shape",
        tvm_tirx.runtime.tensor(np.array(1, dtype="int32")),
    )
    assert scalar_shape == tvm_ffi.Shape([])

    tensor_shape = run_cpu(
        ShapeOfTest,
        "get_shape",
        tvm_tirx.runtime.tensor(np.zeros((1, 2, 3)).astype("int32")),
    )
    assert tensor_shape == tvm_ffi.Shape([1, 2, 3])

    constrained_shape = run_cpu(
        ShapeOfTest,
        "get_constrained_shape",
        tvm_tirx.runtime.tensor(np.zeros((1,)).astype("int32")),
    )
    assert constrained_shape == tvm_ffi.Shape([1])


m = T.dynamic("m", "int64")
n = T.dynamic("n", "int64")


@tvm_tirx.script.ir_module
class ShapeToTensorTest:
    @R.function
    def const_shape(shape: R.Shape(ndim=-1)) -> R.Tensor(ndim=-1):
        return R.shape_to_tensor(shape)

    @R.function
    def symbolic_shape(shape: R.Shape((m, n))) -> R.Tensor(ndim=-1):
        return R.shape_to_tensor(shape)


def test_op_shape_to_tensor():
    # Check type
    isinstance(ShapeToTensorTest["const_shape"].body.ty, tvm.relax.TensorType)
    assert ShapeToTensorTest["const_shape"].body.ty.ndim == 1
    isinstance(ShapeToTensorTest["symbolic_shape"].body.ty, tvm.relax.TensorType)
    assert ShapeToTensorTest["symbolic_shape"].body.ty.ndim == 1

    # Check its functionality
    out2d = run_cpu(ShapeToTensorTest, "const_shape", tvm_ffi.Shape([3, 2]))
    assert isinstance(out2d, tvm_tirx.runtime.Tensor)
    assert np.array_equal(out2d.numpy(), np.array([3, 2]))

    out3d = run_cpu(ShapeToTensorTest, "const_shape", tvm_ffi.Shape([3, 3, 2]))
    assert isinstance(out3d, tvm_tirx.runtime.Tensor)
    assert np.array_equal(out3d.numpy(), np.array([3, 3, 2]))

    out4d = run_cpu(ShapeToTensorTest, "const_shape", tvm_ffi.Shape([3, 3, 2, 2]))
    assert isinstance(out4d, tvm_tirx.runtime.Tensor)
    assert np.array_equal(out4d.numpy(), np.array([3, 3, 2, 2]))

    outs = run_cpu(ShapeToTensorTest, "symbolic_shape", tvm_ffi.Shape([3, 2]))
    assert isinstance(outs, tvm_tirx.runtime.Tensor)
    assert np.array_equal(outs.numpy(), np.array([3, 2]))


def test_op_call_pure_packed():
    @tvm_tirx.script.ir_module
    class CallPureTest:
        @R.function
        def pure_copy(x: R.Tensor((3, 4), "float32")):
            z = R.call_pure_packed(
                "vm.builtin.copy", x, ty_args=(R.Tensor((3, 4), dtype="float32"))
            )
            return z

    np.random.seed(0)  # to avoid flakiness
    arr = np.random.rand(3, 4).astype("float32")
    copy_found = run_cpu(CallPureTest, "pure_copy", tvm_tirx.runtime.tensor(arr))
    assert (copy_found.numpy() == arr).all()


def test_op_call_inplace_packed():
    # in this case we can use the same test as above
    @tvm_tirx.script.ir_module
    class CallInplaceTest:
        @R.function
        def pure_copy(x: R.Tensor((3, 4), "float32")):
            z = R.call_inplace_packed(
                "vm.builtin.copy",
                x,
                inplace_indices=0,
                ty_args=(R.Tensor((3, 4), dtype="float32")),
            )
            return z

    @tvm_ffi.register_global_func("test.inplace.add", override=True)
    def inplace_add(a, b):
        arr_a = a.numpy()
        arr_b = b.numpy()
        for i in range(len(arr_a)):
            for j in range(len(arr_a[i])):
                arr_a[i][j] = arr_a[i][j] + arr_b[i][j]
        a.copyfrom(arr_a)
        return a

    @tvm_tirx.script.ir_module
    class CallInplaceAddTest:
        @R.function
        def inplace_add(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            z = R.call_inplace_packed(
                "test.inplace.add",
                x,
                y,
                inplace_indices=0,
                ty_args=(R.Tensor((3, 4), dtype="float32")),
            )
            return z

    np.random.seed(1)  # to avoid flakiness
    arr_a = np.random.rand(3, 4).astype("float32")
    arr_b = np.random.rand(3, 4).astype("float32")
    sum = arr_a + arr_b
    tvm_arr_a = tvm_tirx.runtime.tensor(arr_a)
    result = run_cpu(CallInplaceAddTest, "inplace_add", tvm_arr_a, tvm_tirx.runtime.tensor(arr_b))
    assert result == tvm_arr_a
    assert (result.numpy() == sum).all()

    @tvm_ffi.register_global_func("test.inplace.tuple_add", override=True)
    def inplace_tuple_add(a, b):
        arr_a = a.numpy()
        arr_b = b.numpy()
        c = tvm_tirx.runtime.tensor(arr_a + arr_b)
        for i in range(len(arr_a)):
            for j in range(len(arr_a[i])):
                arr_a[i][j] = arr_a[i][j] + arr_b[i][j]
        a.copyfrom(arr_a)
        return tvm_ffi.convert([a, c])

    @tvm_tirx.script.ir_module
    class CallInplaceTuple:
        @R.function
        def inplace_tuple(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            z = R.call_inplace_packed(
                "test.inplace.tuple_add",
                x,
                y,
                inplace_indices=[0, -1],
                ty_args=(R.Tensor((3, 4), dtype="float32"), R.Tensor((3, 4), dtype="float32")),
            )
            return z

    np.random.seed(2)  # to avoid flakiness
    arr_a = np.random.rand(3, 4).astype("float32")
    arr_b = np.random.rand(3, 4).astype("float32")
    sum = arr_a + arr_b
    tvm_arr_a = tvm_tirx.runtime.tensor(arr_a)
    tvm_arr_b = tvm_tirx.runtime.tensor(arr_b)
    result = run_cpu(CallInplaceTuple, "inplace_tuple", tvm_arr_a, tvm_arr_b)
    assert result[0] == tvm_arr_a
    assert (result[0].numpy() == sum).all()
    assert result[1] != tvm_arr_a and result[1] != tvm_arr_b
    assert (result[1].numpy() == sum).all()


@pytest.mark.parametrize("name", ["call_tir", "call_tir_with_grad", "call_tir_inplace"])
@pytest.mark.parametrize("tuple_result", [False, True])
@pytest.mark.parametrize("result_kind", ["inferred", "missing", "explicit"])
@pytest.mark.parametrize("namespace", [relax.op, R])
def test_call_tir_canonical_roundtrip(name, tuple_result, result_kind, namespace):
    tensor_type = R.Tensor((3,), "float32")
    output_type = tvm_tirx.ir.TupleType([tensor_type, tensor_type]) if tuple_result else tensor_type
    result_type = {
        "inferred": None,
        "missing": tvm_tirx.ir.Type.missing(),
        "explicit": output_type,
    }[result_kind]
    output_count = 2 if tuple_result else 1
    inplace = name == "call_tir_inplace"
    params = [
        tvm_tirx.tirx.Var(f"p{i}", T.Tensor((3,), "float32"))
        for i in range(1 + output_count - int(inplace))
    ]
    func = tvm_tirx.tirx.Function(
        params, [tvm_tirx.ir.Evaluate(0)], ret_type=tvm_tirx.ir.TupleType([])
    )
    callee = relax.BlockBuilder().add_func(func, "callee")
    x = relax.Var("x", tensor_type)
    kwargs = {}
    if name == "call_tir_with_grad":
        kwargs = {"te_grad_name": "gradient", "te_grad_kwargs": {"scale": 2}}
    elif inplace:
        kwargs = {"inplace_indices": [0, -1] if tuple_result else [0]}
    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("native_call"), 1, 3, 2, 4)
    builder = getattr(namespace, name)
    call = builder(callee, (x,), ty_args=[output_type], ty=result_type, loc=loc, **kwargs)
    call.validate()
    assert call.loc.same_as(loc)
    assert call.args[0].same_as(callee)
    assert call.args[1].fields[0].same_as(x)
    tvm_tirx.ir.assert_structural_equal(call.ty_args[0], output_type)
    tvm_tirx.ir.assert_structural_equal(
        call.ty, output_type if result_type is None else result_type
    )
    # An attrs object and semantic keywords construct the same canonical Call.
    with_attrs = builder(
        callee, call.args[1], ty_args=[output_type], attrs=call.attrs, ty=result_type, loc=loc
    )
    tvm_tirx.ir.assert_structural_equal(call, with_attrs)
    # Roundtrip printing requires a type-consistent Call after constructor checks.
    call = tvm_tirx.ir.Call(
        call.op, call.args, attrs=call.attrs, ty_args=call.ty_args, loc=call.loc
    )
    printed = call.script()
    assert printed.startswith(f"R.{name}(")
    assert "ty_args=" in printed and "out_ty=" not in printed
    reconstructed = eval(
        printed, {"I": I, "R": R, "T": T, "Module": SimpleNamespace(callee=callee), "x": x}
    )
    tvm_tirx.ir.assert_structural_equal(call, reconstructed)


@pytest.mark.parametrize("tuple_size", [None, 0, 1, 2])
@pytest.mark.parametrize("callee_kind", ["string", "extern", "string_imm", "var"])
@pytest.mark.parametrize("result_kind", ["inferred", "missing", "explicit"])
def test_call_dps_packed_canonical_roundtrip(tuple_size, callee_kind, result_kind):
    tensor_type = R.Tensor((3,), "float32")
    output_type = (
        tensor_type if tuple_size is None else tvm_tirx.ir.TupleType([tensor_type] * tuple_size)
    )
    result_type = {
        "inferred": None,
        "missing": tvm_tirx.ir.Type.missing(),
        "explicit": R.Tensor((3,), "float64"),
    }[result_kind]
    callee = {
        "string": "packed",
        "extern": relax.ExternFunc("packed"),
        "string_imm": tvm_tirx.ir.StringImm("packed"),
        "var": relax.Var("callee", R.Object()),
    }[callee_kind]
    expected_callee = relax.ExternFunc("packed") if callee_kind == "string" else callee
    x = relax.Var("x", tensor_type)
    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("packed_call"), 1, 3, 2, 4)
    call = relax.op.call_dps_packed(callee, (x,), ty_args=[output_type], ty=result_type, loc=loc)
    expected = tvm_tirx.ir.Call(
        "relax.call_dps_packed",
        [expected_callee, (x,)],
        ty_args=[output_type],
        loc=loc,
        ty=result_type,
    )
    tvm_tirx.ir.assert_structural_equal(call, expected)
    assert call.loc.same_as(loc)
    if callee_kind != "string":
        assert call.args[0].same_as(callee)
    else:
        assert isinstance(call.args[0], relax.ExternFunc)
    tvm_tirx.ir.assert_structural_equal(
        call.ty, output_type if result_type is None else result_type
    )
    # Roundtrip printing requires a type-consistent Call after constructor checks.
    call = tvm_tirx.ir.Call(
        call.op, call.args, attrs=call.attrs, ty_args=call.ty_args, loc=call.loc
    )
    printed = call.script()
    assert printed.startswith("R.call_dps_packed(")
    if callee_kind in ("string", "extern"):
        assert printed.startswith('R.call_dps_packed("packed",')
    reconstructed = eval(printed, {"I": I, "R": R, "T": T, "x": x, "callee": callee})
    tvm_tirx.ir.assert_structural_equal(call, reconstructed)


@pytest.mark.parametrize("exception", ["attrs", "typed_callee", "tuple_var"])
def test_call_dps_packed_raw_roundtrip(exception):
    tensor_type = R.Tensor((3,), "float32")
    x = relax.Var("x", tensor_type)
    callee = relax.ExternFunc("packed")
    args = (x,)
    attrs = None
    if exception == "attrs":
        attrs = tvm_tirx.ir.make_node("ir.DictAttrs", label="preserved")
    elif exception == "typed_callee":
        callee = relax.ExternFunc("packed", ty=relax.FuncType([tensor_type], tensor_type))
    else:
        args = relax.Var("args", tvm_tirx.ir.TupleType([tensor_type]))
    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("packed_raw"), 1, 3, 2, 4)
    call = R.call_dps_packed(
        callee, args, ty_args=[tensor_type], attrs=attrs, ty=tvm_tirx.ir.Type.missing(), loc=loc
    )
    assert call.args[0].same_as(callee)
    assert call.loc.same_as(loc)
    if attrs is not None:
        assert call.attrs.same_as(attrs)
    printed = call.script()
    assert printed.startswith('T.Call("relax.call_dps_packed",')
    reconstructed = eval(printed, {"I": I, "R": R, "T": T, "x": x, "args": args})
    tvm_tirx.ir.assert_structural_equal(call, reconstructed)


@pytest.mark.parametrize("tuple_size", [None, 0, 1, 2])
@pytest.mark.parametrize("result_kind", ["inferred", "missing", "explicit"])
@pytest.mark.parametrize("builder", [relax.op.call_py_func, R.call_py_func])
def test_call_py_func_canonical_roundtrip(tuple_size, result_kind, builder):
    tensor_type = R.Tensor((3,), "float32")
    output_type = (
        tensor_type if tuple_size is None else tvm_tirx.ir.TupleType([tensor_type] * tuple_size)
    )
    result_type = {
        "inferred": None,
        "missing": tvm_tirx.ir.Type.missing(),
        "explicit": R.Tensor((3,), "float64"),
    }[result_kind]
    loc = tvm_tirx.ir.SourceLoc(tvm_tirx.ir.SourceName("callback"), 1, 3, 2, 4)
    x = relax.Var("x", tensor_type)
    call = builder("callback", (x,), ty_args=[output_type], ty=result_type, loc=loc)
    expected = tvm_tirx.ir.Call(
        "relax.call_py_func",
        ["callback", (x,)],
        ty_args=[output_type],
        loc=loc,
        ty=result_type,
    )
    tvm_tirx.ir.assert_structural_equal(call, expected)
    assert call.loc.same_as(loc)
    assert call.args[0].value == "callback"
    assert len(call.args[1].fields) == 1 and call.args[1].fields[0].same_as(x)
    tvm_tirx.ir.assert_structural_equal(call.ty_args[0], output_type)
    tvm_tirx.ir.assert_structural_equal(
        call.ty, output_type if result_type is None else result_type
    )
    # Roundtrip printing requires a type-consistent Call after constructor checks.
    call = tvm_tirx.ir.Call(
        call.op, call.args, attrs=call.attrs, ty_args=call.ty_args, loc=call.loc
    )
    printed = call.script()
    assert printed.startswith("R.call_py_func(")
    assert "ty_args=" in printed and "out_ty=" not in printed
    reconstructed = eval(printed, {"I": I, "R": R, "T": T, "x": x})
    tvm_tirx.ir.assert_structural_equal(call, reconstructed)


@pytest.mark.parametrize("tuple_size", [None, 0, 1, 2])
def test_call_py_func_function_roundtrip(tuple_size):
    tensor_type = R.Tensor((3,), "float32")
    output_type = (
        tensor_type if tuple_size is None else tvm_tirx.ir.TupleType([tensor_type] * tuple_size)
    )

    @R.function
    def func(x: R.Tensor((3,), "float32")):
        value = R.call_py_func("callback", (x,), ty_args=[output_type])
        return value

    reconstructed = tvm_tirx.script.from_source(func.script(), extra_vars={"I": I, "R": R, "T": T})
    tvm_tirx.ir.assert_structural_equal(func, reconstructed)


def test_call_py_func_numpy_execution():
    register = tvm_ffi.get_global_func("vm.builtin.register_py_func")
    unregister = tvm_ffi.get_global_func("vm.builtin.unregister_py_func")

    def numpy_pair(x):
        data = x.numpy()
        return tvm_tirx.runtime.tensor(np.maximum(data, 0)), tvm_tirx.runtime.tensor(-data)

    register("canonical_numpy_pair", numpy_pair)
    try:

        @tvm_tirx.script.ir_module
        class Module:
            @R.function
            def main(x: R.Tensor((3,), "float32")):
                pair = R.call_py_func(
                    "canonical_numpy_pair",
                    (x,),
                    ty_args=[R.Tuple(R.Tensor((3,), "float32"), R.Tensor((3,), "float32"))],
                )
                return pair

        data = np.array([-1.0, 0.0, 1.0], dtype="float32")
        result = run_cpu(Module, "main", tvm_tirx.runtime.tensor(data))
        np.testing.assert_array_equal(result[0].numpy(), np.maximum(data, 0))
        np.testing.assert_array_equal(result[1].numpy(), -data)
    finally:
        unregister("canonical_numpy_pair")


def test_op_call_py_func():
    """Test R.call_py_func operator functionality."""
    import torch

    def torch_relu(x):
        if isinstance(x, tvm_tirx.runtime.Tensor):
            x_torch = torch.from_numpy(x.numpy())
        elif hasattr(x, "asnumpy"):
            x_torch = torch.from_numpy(x.asnumpy())
        else:
            x_np = np.array(x)
            if isinstance(x_np, tvm_tirx.runtime.Tensor):
                x_torch = torch.from_numpy(x_np.numpy())
            elif len(x_np) > 0 and isinstance(x_np[0], tvm_tirx.runtime.Tensor):
                x_torch = torch.from_numpy(np.array([t.numpy() for t in x_np]))
                if x_torch.ndim > 1:
                    x_torch = x_torch.flatten()
            else:
                x_torch = torch.from_numpy(x_np)
        result = torch.relu(x_torch)
        return tvm_tirx.runtime.tensor(result.numpy())

    def torch_sigmoid(x):
        if isinstance(x, tvm_tirx.runtime.Tensor):
            x_torch = torch.from_numpy(x.numpy())
        elif hasattr(x, "asnumpy"):
            x_torch = torch.from_numpy(x.asnumpy())
        else:
            x_np = np.array(x)
            if isinstance(x_np, tvm_tirx.runtime.Tensor):
                x_torch = torch.from_numpy(x_np.numpy())
            elif len(x_np) > 0 and isinstance(x_np[0], tvm_tirx.runtime.Tensor):
                x_torch = torch.from_numpy(np.array([t.numpy() for t in x_np]))
                if x_torch.ndim > 1:
                    x_torch = x_torch.flatten()
            else:
                x_torch = torch.from_numpy(x_np)
        result = torch.sigmoid(x_torch)
        return tvm_tirx.runtime.tensor(result.numpy())

    register_func = tvm_ffi.get_global_func("vm.builtin.register_py_func")
    register_func("torch_relu", torch_relu)
    register_func("torch_sigmoid", torch_sigmoid)

    @tvm_tirx.script.ir_module
    class CallPyFuncTest:
        @R.function
        def simple_call(x: R.Tensor((3,), "float32")):
            result = R.call_py_func(R.str("torch_relu"), (x,), ty_args=[R.Tensor((3,), "float32")])
            return result

        @R.function
        def multiple_calls(x: R.Tensor((2,), "float32")):
            y = R.call_py_func(R.str("torch_relu"), (x,), ty_args=[R.Tensor((2,), "float32")])
            z = R.call_py_func(R.str("torch_sigmoid"), (y,), ty_args=[R.Tensor((2,), "float32")])
            return z

    np.random.seed(0)
    x_data = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    x_tvm = tvm_tirx.runtime.tensor(x_data)

    result = run_cpu(CallPyFuncTest, "simple_call", x_tvm)
    expected = np.maximum(x_data, 0.0)
    assert (result.numpy() == expected).all()

    y_data = np.array([-0.5, 0.5], dtype=np.float32)
    y_tvm = tvm_tirx.runtime.tensor(y_data)

    result2 = run_cpu(CallPyFuncTest, "multiple_calls", y_tvm)
    expected2 = 1.0 / (1.0 + np.exp(-np.maximum(y_data, 0.0)))
    assert (result2.numpy() == expected2).all()

    unregister_func = tvm_ffi.get_global_func("vm.builtin.unregister_py_func")
    unregister_func("torch_relu")
    unregister_func("torch_sigmoid")


def test_py_func_registry_is_scoped_to_its_module():
    """A module's finalizer must drop its own registrations and nothing else."""
    from tvm.relax.base_py_module import BasePyModule

    get_func = tvm_ffi.get_global_func("vm.builtin.get_py_func")
    tvm_ffi.get_global_func("vm.builtin.register_py_func")("registry_probe", lambda x: x)

    # __new__ skips __init__'s JIT compilation; only the registration matters here.
    module = BasePyModule.__new__(BasePyModule)
    module.ir_mod = SimpleNamespace(__pyfuncs__={"registry_owned": lambda self, x: x})
    module._register_python_functions()
    assert get_func("registry_owned") is not None
    module_ref = weakref.ref(module)

    del module
    gc.collect()

    assert module_ref() is None, "the registry must not keep the module alive"
    assert get_func("registry_probe") is not None, "another owner's function was dropped"
    with pytest.raises(RuntimeError, match="not found in registry"):
        get_func("registry_owned")
    tvm_ffi.get_global_func("vm.builtin.unregister_py_func")("registry_probe")


def test_py_func_registry_exit_with_registered_function():
    """Exiting with a Python function still registered must not crash the process."""
    import subprocess

    code = (
        "import tvm\n"
        "tvm.get_global_func('vm.builtin.register_py_func')('py_identity', lambda x: x)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_op_to_device():
    @tvm_tirx.script.ir_module
    class CallToDevice:
        @R.function
        def to_dev(x: R.Tensor((3, 4), "float32")):
            z = R.call_pure_packed(
                "vm.builtin.to_device",
                x,
                1,
                0,
                ty_args=(R.Tensor((3, 4), dtype="float32")),
            )
            return z

    np.random.seed(0)  # to avoid flakiness
    arr = np.random.rand(3, 4).astype("float32")
    copy_found = run_cpu(CallToDevice, "to_dev", tvm_tirx.runtime.tensor(arr))
    assert (copy_found.numpy() == arr).all()


def test_op_to_vdevice():
    @tvm_tirx.script.ir_module
    class ToVDevice:
        I.module_global_infos({"vdevice": [R.vdevice("llvm")]})

        @R.function
        def to_vdev(x: R.Tensor((3, 4), "float32")):
            dst_vdev = tvm.relax.VDevice("llvm", 0, "global")
            ret = R.to_vdevice(x, "llvm")
            return ret

    np.random.seed(0)
    arr = np.random.rand(3, 4).astype("float32")
    copy_found = run_cpu(ToVDevice, "to_vdev", tvm_tirx.runtime.tensor(arr))
    assert (copy_found.numpy() == arr).all()


def test_scalar_tensor_as_branch_condition():
    """The condition of a branch may be a scalar tensor"""

    @R.function
    def func(condition: R.Tensor((), "bool")):
        if condition:
            out = R.prim_value(5)
        else:
            out = R.prim_value(10)
        return out

    res = run_cpu(func, tvm_tirx.runtime.tensor(np.array(True)))
    assert res == 5

    res = run_cpu(func, tvm_tirx.runtime.tensor(np.array(False)))
    assert res == 10


def test_prim_value_as_branch_condition():
    """The condition may be a Expr"""

    @R.function
    def func(condition: T.bool):
        if condition:
            out = R.prim_value(5)
        else:
            out = R.prim_value(10)
        return out

    res = run_cpu(func, True)
    assert res == 5

    res = run_cpu(func, False)
    assert res == 10


def test_computed_prim_value_as_branch_condition():
    """The primitive scalar condition may be computed within the function"""

    N = T.dynamic("N", "int64")

    @R.function
    def func(x: R.Tensor([N], "int64")):
        if R.prim_value(N % 16 == 0):
            out = R.prim_value(5)
        else:
            out = R.prim_value(10)
        return out

    res = run_cpu(func, tvm_tirx.runtime.tensor(np.arange(16)))
    assert res == 5

    res = run_cpu(func, tvm_tirx.runtime.tensor(np.arange(20)))
    assert res == 10


if __name__ == "__main__":
    tvm_tirx.testing.main()
