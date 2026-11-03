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
# pylint: disable=unused-argument
# ruff: noqa: E501, F841
"""
ONNX testcases
================
This file is a test script to test Relax ONNX frontend coverage.
"""

# Allow TVMScript expected IR to capture shape/dtype names used only in annotations.
from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx
import onnxruntime
import tvm_ffi
from onnx import ModelProto, TensorProto, helper, numpy_helper

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.frontend.onnx import from_onnx
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T

bg = np.random.MT19937(0)
rg = np.random.Generator(bg)


def collect_relax_call_ops(func):
    call_ops = set()

    def _visit(expr):
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm_tirx.ir.Op):
            call_ops.add(expr.op.name)

    relax.analysis.post_order_visit(func.body, _visit)
    return call_ops


def generate_random_inputs(
    model: ModelProto, inputs: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    input_values = {}
    # Iterate through model inputs and extract their shape.
    for i in model.graph.input:
        if inputs is not None and i.name in inputs and inputs[i.name] is not None:
            input_values[i.name] = inputs[i.name]
            continue
        shape = []
        for dim in i.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)

        input_values[i.name] = generate_random_value(shape, i.type.tensor_type.elem_type)

    return input_values


def generate_random_value(shape, elem_type) -> np.ndarray:
    # Extract datatype for the input.
    if elem_type:
        dtype = str(helper.tensor_dtype_to_np_dtype(elem_type))
    else:
        dtype = "float32"

    # Generate random inputs for each input.
    if dtype == "bool":
        # random_value = np.random.choice(a=[False, True], size=shape)
        random_value = rg.choice(a=[False, True], size=shape)
    elif dtype.startswith("int"):
        # Keep non-zero values
        random_value = rg.integers(low=-63, high=63, size=shape).astype(dtype)
        random_value[random_value <= 0] -= 1
    else:
        random_value = rg.standard_normal(size=shape).astype(dtype)

    return random_value


def check_correctness(
    model: ModelProto,
    inputs: dict[str, np.ndarray] | None = None,
    ir_version: int = 8,
    opset: int = 14,
    rtol: float = 1e-7,
    atol: float = 1e-5,
    check_dtypes: bool = False,
) -> None:
    """Run an onnx model in both onnxruntime and TVM through our importer
       confirm that the results match. Otherwise, an exception will be raised.

    Parameters
    ----------
    model: ModelProto
        The input onnx model that should be tested.
    inputs: Optional[Dict[str, np.ndarray]]
        An optional dictionary containing values for each input in the onnx model.
    ir_version: int
        Which version of the onnx IR to use.
    opset: int
        The opset version to use for the onnx importer.
    atol: float
        Set the tolerance of correctness checking. Some ops may be show more
        arithmetic variance than others.
    check_dtypes: bool
        Check if data types are the same.
    """
    # Configure model format.
    if ir_version is not None:
        model.ir_version = ir_version
    if opset is not None:
        model.opset_import[0].version = opset

    # If inputs are not provided, extract them from the onnx graph and produce random
    # values that we'll use for testing.
    inputs = generate_random_inputs(model, inputs)

    # Run the model through onnx to get the expected result.
    ort_session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_output = ort_session.run([], inputs)

    # Convert the onnx model into relax through the onnx importer.
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    # Convert operators for inference mode.
    tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
    # Legalize any relax ops into tensorir.
    tvm_model = relax.transform.LegalizeOps()(tvm_model)

    # Separate model from parameters.
    tvm_model, params = relax.frontend.detach_params(tvm_model)
    # Compile the relax graph into a VM then run.
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        ex = tvm.compile(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    # Prepare inputs.
    input_list = [inputs[key.name] for key in tvm_model["main"].params if key.name in inputs]
    if params:
        input_list += params["main"]

    # Run model and check outputs.
    vm.set_input("main", *input_list)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    # Wrap as a list if there is only one output.
    if len(ort_output) == 1:
        # Do not check the output number for TVM
        # As for sequence output, the TVM output is a Tuple
        # while the ONNX output number is one, which is a list
        tvm_output = [tvm_output]

    def _get_numpy_subdtype(narray):
        if np.issubdtype(narray.dtype, np.integer):
            return "integer"
        elif np.issubdtype(narray.dtype, np.floating):
            return "floating"
        elif np.issubdtype(narray.dtype, np.bool_):
            return "bool"
        elif np.issubdtype(narray.dtype, np.complexfloating):
            return "complexfloating"
        else:
            return "other"

    def _check_output(tvm_out, ort_out):
        if isinstance(tvm_out, tuple) and isinstance(ort_out, tvm_ffi.Shape | list):
            assert len(tvm_out) == len(ort_out), "Unequal number of outputs"
            for tvm_out_i, ort_out_i in zip(tvm_out, ort_out):
                _check_output(tvm_out_i, ort_out_i)
        elif isinstance(tvm_out, tvm_tirx.runtime.Tensor) and isinstance(ort_out, np.ndarray):
            if check_dtypes:
                assert tvm_out.numpy().dtype == ort_out.dtype
            tvm_tirx.testing.assert_allclose(tvm_out.numpy(), ort_out, rtol=rtol, atol=atol)
        elif isinstance(tvm_out, tvm_ffi.Shape) and isinstance(ort_out, np.ndarray):
            shape_out = tvm_tirx.runtime.tensor([int(i) for i in tvm_out])
            if check_dtypes:
                assert _get_numpy_subdtype(shape_out.numpy()) == _get_numpy_subdtype(ort_out)
            tvm_tirx.testing.assert_allclose(shape_out.numpy(), ort_out, rtol=rtol, atol=atol)
        elif isinstance(tvm_out, int | float | bool) and isinstance(ort_out, np.ndarray):
            if check_dtypes:
                assert _get_numpy_subdtype(np.array(tvm_out)) == _get_numpy_subdtype(ort_out)
            tvm_tirx.testing.assert_allclose(np.array(tvm_out), ort_out, rtol=rtol, atol=atol)
        else:
            raise ValueError(f"Unsupported types: {type(tvm_out)}, {type(ort_out)}")

    # Check that number of outputs match.
    assert len(tvm_output) == len(ort_output), "Unequal number of outputs"
    for tvm_out, ort_out in zip(tvm_output, ort_output):
        # TODO Allow configurable tolerance.
        if ort_out is not None:
            _check_output(tvm_out, ort_out)


def run_in_tvm(
    model: ModelProto,
    inputs: dict[str, np.ndarray] | None = None,
    ir_version: int = 8,
    opset: int = 14,
):
    if ir_version is not None:
        model.ir_version = ir_version
    if opset is not None:
        for opset_import in model.opset_import:
            if opset_import.domain in ["", "ai.onnx"]:
                opset_import.version = opset
                break

    inputs = generate_random_inputs(model, inputs)
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    tvm_model, params = relax.frontend.detach_params(tvm_model)

    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        ex = tvm.compile(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    input_list = [inputs[key.name] for key in tvm_model["main"].params if key.name in inputs]
    if params:
        input_list += params["main"]

    vm.set_input("main", *input_list)
    vm.invoke_stateful("main")
    return vm.get_outputs("main")


@pytest.mark.parametrize(
    "input_names, expected_names",
    [
        ([".", "123"], ["_", "input_123"]),
        ([".", "_"], ["_", "__1"]),
        (["123", "input_123"], ["input_123", "input_123_1"]),
    ],
)
def test_sanitize(input_names, expected_names):
    node = helper.make_node("Add", inputs=input_names, outputs=["output"])
    graph = helper.make_graph(
        [node],
        "test",
        inputs=[
            helper.make_tensor_value_info(str(var), TensorProto.FLOAT, [32, 32])
            for var in input_names
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32]),
        ],
    )
    model = helper.make_model(graph, producer_name="test_sanitizer")

    tvm_model = from_onnx(model)

    for i, param in enumerate(tvm_model["main"].params):
        assert param.name == expected_names[i]


def verify_unary(
    op_name,
    shape,
    attrs={},
    domain=None,
    input_dtype=TensorProto.FLOAT,
    output_dtype=TensorProto.FLOAT,
    opset=14,
):
    test_node = helper.make_node(op_name, ["x"], ["y"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "elemwise_test",
        inputs=[
            helper.make_tensor_value_info("x", input_dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", output_dtype, shape)],
    )

    model = helper.make_model(graph, producer_name="elemwise_test")
    check_correctness(model, opset=opset)


def make_unary_model(
    op_name,
    shape,
    attrs=None,
    domain=None,
    input_dtype=TensorProto.FLOAT,
    output_dtype=TensorProto.FLOAT,
):
    attrs = attrs or {}
    test_node = helper.make_node(op_name, ["x"], ["y"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "elemwise_structural_test",
        inputs=[
            helper.make_tensor_value_info("x", input_dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", output_dtype, shape)],
    )
    return helper.make_model(graph, producer_name="elemwise_structural_test")


def verify_binary(
    op_name, shape_a, shape_b, shape_c, attrs={}, domain=None, dtype=TensorProto.FLOAT, opset=14
):
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "binary_test",
        inputs=[
            helper.make_tensor_value_info("a", dtype, shape_a),
            helper.make_tensor_value_info("b", dtype, shape_b),
        ],
        outputs=[helper.make_tensor_value_info("c", dtype, shape_c)],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    check_correctness(model, opset=opset, check_dtypes=True)


def verify_binary_scalar(op_name, attrs={}, domain=None, dtype=TensorProto.INT32, opset=14):
    a = make_constant_node("a", dtype, [], [4])
    b = make_constant_node("b", dtype, [], [8])
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [a, b, test_node],
        "binary_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("c", dtype, ())],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    model.opset_import[0].version = opset
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)

    dtype_str = str(helper.tensor_dtype_to_np_dtype(dtype))
    lhs = np.array(4, dtype=dtype_str)
    rhs = np.array(8, dtype=dtype_str)
    op = {
        "Add": np.add,
        "Sub": np.subtract,
        "Mul": np.multiply,
        "Div": np.divide,
        "Pow": np.power,
        "Mod": np.fmod if attrs.get("fmod", 0) else np.mod,
    }[op_name]
    expected_value = op(lhs, rhs).astype(dtype_str)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), dtype=dtype_str):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((), dtype=dtype_str) = R.const(expected_value.item(), dtype_str)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("dynamic", [True, False])
def test_matmul(dynamic):
    matmul_node = helper.make_node("MatMul", ["a", "b"], ["c"])

    a_shape = [32, 48]
    b_shape = [48, 64]
    output_shape = [32, 64]

    if dynamic:
        a_shape = ["?", "?"]

    graph = helper.make_graph(
        [matmul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor(
                "b", TensorProto.FLOAT, b_shape, np.random.normal(size=b_shape).astype("float32")
            )
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, output_shape)],
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    inputs = None
    if dynamic:
        inputs = {
            "a": np.random.normal(size=[32, 48]).astype("float32"),
        }
    check_correctness(model, inputs)


def test_matmulinteger16():
    def verify_matmulinteger16(a_dtype, b_dtype, a_shape, b_shape, expected):
        out_dtype = np.uint32 if a_dtype == np.uint16 and b_dtype == np.uint16 else np.int32
        output_shape = [
            *np.broadcast_shapes(tuple(a_shape[:-2]), tuple(b_shape[:-2])),
            a_shape[-2],
            b_shape[-1],
        ]

        node = helper.make_node("MatMulInteger16", ["a", "b"], ["y"], domain="com.microsoft")
        graph = helper.make_graph(
            [node],
            "matmulinteger16_test",
            inputs=[
                helper.make_tensor_value_info(
                    "a", helper.np_dtype_to_tensor_dtype(np.dtype(a_dtype)), a_shape
                ),
                helper.make_tensor_value_info(
                    "b", helper.np_dtype_to_tensor_dtype(np.dtype(b_dtype)), b_shape
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "y",
                    helper.np_dtype_to_tensor_dtype(np.dtype(out_dtype)),
                    output_shape,
                )
            ],
        )
        model = helper.make_model(
            graph,
            producer_name="matmulinteger16_test",
            opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 11

        tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedInt16:
        @R.function
        def main(
            a: R.Tensor((2, 3), dtype="int16"),
            b: R.Tensor((3, 4), dtype="int16"),
        ) -> R.Tensor((2, 4), dtype="int32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="int32") = R.astype(a, dtype="int32")
                lv1: R.Tensor((3, 4), dtype="int32") = R.astype(b, dtype="int32")
                gv: R.Tensor((2, 4), dtype="int32") = R.matmul(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedUInt16:
        @R.function
        def main(
            a: R.Tensor((2, 3), dtype="uint16"),
            b: R.Tensor((3, 4), dtype="uint16"),
        ) -> R.Tensor((2, 4), dtype="uint32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="uint32") = R.astype(a, dtype="uint32")
                lv1: R.Tensor((3, 4), dtype="uint32") = R.astype(b, dtype="uint32")
                gv: R.Tensor((2, 4), dtype="uint32") = R.matmul(lv, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMixedBatched:
        @R.function
        def main(
            a: R.Tensor((2, 1, 3, 5), dtype="int16"),
            b: R.Tensor((1, 2, 5, 4), dtype="uint16"),
        ) -> R.Tensor((2, 2, 3, 4), dtype="int32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 1, 3, 5), dtype="int32") = R.astype(a, dtype="int32")
                lv1: R.Tensor((1, 2, 5, 4), dtype="int32") = R.astype(b, dtype="int32")
                gv: R.Tensor((2, 2, 3, 4), dtype="int32") = R.matmul(lv, lv1)
                R.output(gv)
            return gv

    verify_matmulinteger16(np.int16, np.int16, [2, 3], [3, 4], ExpectedInt16)
    verify_matmulinteger16(np.uint16, np.uint16, [2, 3], [3, 4], ExpectedUInt16)
    verify_matmulinteger16(
        np.int16,
        np.uint16,
        [2, 1, 3, 5],
        [1, 2, 5, 4],
        ExpectedMixedBatched,
    )


def test_matmulinteger16_invalid_dtype_raises():
    node = helper.make_node("MatMulInteger16", ["a", "b"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "matmulinteger16_invalid_dtype_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.INT8, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.UINT16, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT32, [2, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="matmulinteger16_invalid_dtype_test",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11

    with pytest.raises(ValueError, match="input A"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_concat_with_param_shape_value():
    """Concat must handle a 1D-int64 initializer mixed with a ShapeExpr when
    keep_params_in_input=True. Standard pattern in PyTorch-exported ONNX
    models for dynamic-batch Reshape: Reshape(x, Concat(Shape(x)[:1], [12]))."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 4])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 12])
    twelve = numpy_helper.from_array(np.array([12], dtype=np.int64), "twelve")
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), "ends")
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Slice", ["x_shape", "starts", "ends"], ["dyn_n"]),
        helper.make_node("Concat", ["dyn_n", "twelve"], ["new_shape"], axis=0),
        helper.make_node("Reshape", ["x", "new_shape"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "concat_param_shape",
        [inp],
        [out],
        initializer=[twelve, starts, ends],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    # Both modes should succeed; previously True crashed with
    # "Op(relax.concat) expects the input to be a Tuple of Tensors".
    from_onnx(model, keep_params_in_input=False)
    from_onnx(model, keep_params_in_input=True)


def test_concat_with_param_tensor_keeps_runtime_param():
    """Concat(input, weight) under keep_params_in_input=True must keep `weight`
    as a runtime param, not fold it into a constant."""
    weight_np = np.arange(8, dtype=np.float32).reshape(2, 4)
    graph = helper.make_graph(
        [helper.make_node("Concat", ["x", "w"], ["y"], axis=0)],
        "concat_param_tensor",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])],
        initializer=[numpy_helper.from_array(weight_np, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    mod, params = relax.frontend.detach_params(from_onnx(model, keep_params_in_input=True))
    assert "w" in [p.name for p in mod["main"].params]
    assert len(params["main"]) == 1
    np.testing.assert_array_equal(params["main"][0].numpy(), weight_np)


@pytest.mark.parametrize(
    "op_name,np_op",
    [
        ("Less", np.less),
        ("LessOrEqual", np.less_equal),
        ("Greater", np.greater),
        ("GreaterOrEqual", np.greater_equal),
    ],
)
@pytest.mark.parametrize("np_dtype", ["int32", "float32"])
def test_constant_comparison_outputs_bool(op_name, np_op, np_dtype):
    a_np = np.array([[1], [5]], dtype=np_dtype)
    b_np = np.array([[3]], dtype=np_dtype)
    rhs_np = np.array([[3]], dtype=np_dtype)
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["d"], ["dummy"]),
            helper.make_node("Concat", ["a", "b"], ["lhs"], axis=0),
            helper.make_node(op_name, ["lhs", "rhs"], ["y"]),
        ],
        "constant_comparison",
        [helper.make_tensor_value_info("d", TensorProto.INT32, [1])],
        [
            helper.make_tensor_value_info("y", TensorProto.BOOL, [3, 1]),
            helper.make_tensor_value_info("dummy", TensorProto.INT32, [1]),
        ],
        initializer=[
            numpy_helper.from_array(a_np, "a"),
            numpy_helper.from_array(b_np, "b"),
            numpy_helper.from_array(rhs_np, "rhs"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
    onnx.checker.check_model(model)

    mod = from_onnx(model, opset=18, shape_dict={"d": [1]}, keep_params_in_input=False)
    constants = []

    def collect_constants(expr):
        if isinstance(expr, tvm_tirx.ir.GenericConst):
            constants.append(expr.value.numpy())

    relax.analysis.post_order_visit(mod["main"].body, collect_constants)
    folded_outputs = [arr for arr in constants if arr.shape == (3, 1)]
    assert len(folded_outputs) == 1
    expected = np_op(np.concatenate([a_np, b_np], axis=0), rhs_np)
    np.testing.assert_array_equal(folded_outputs[0], expected)
    assert folded_outputs[0].dtype == np.dtype("bool")


@pytest.mark.parametrize("op_name", ["Add", "Sub", "Mul", "Div", "Pow"])
def test_binary(op_name: str):
    verify_binary(op_name, [1, 32], [1, 32], [1, 32])
    verify_binary_scalar(op_name)


def test_div_integer_constant_zero_divisor_raises_valueerror():
    b_init = numpy_helper.from_array(np.array([3, 0, -2, 1], dtype=np.int32), name="b")
    node = helper.make_node("Div", ["a", "b"], ["y"])
    graph = helper.make_graph(
        [node],
        "div_const_zero",
        [helper.make_tensor_value_info("a", TensorProto.INT32, [4])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [4])],
        initializer=[b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9

    with pytest.raises(
        ValueError, match="ONNX Div with integer inputs encountered divisor value 0"
    ):
        from_onnx(model, opset=18, keep_params_in_input=False)


def test_div_integer_constant_folding_truncates_toward_zero():
    a = make_constant_node("a", TensorProto.INT64, [2], [-5, 5])
    b = make_constant_node("b", TensorProto.INT64, [2], [2, 2])
    node = helper.make_node("Div", ["a", "b"], ["y"])
    graph = helper.make_graph(
        [a, b, node],
        "div_integer_constant",
        [],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    tvm_model = from_onnx(model, opset=13)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2,), dtype="int64"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="int64") = R.const([-2, 2], "int64")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_div_integer_constant_folding_preserves_int64_precision():
    dividend_values = np.array([2**53 + 1, 2**53 + 3, -(2**53 + 3), -5, 5], dtype=np.int64)
    divisor_values = np.array([1, 1, 1, 2, -2], dtype=np.int64)
    expected = np.array([2**53 + 1, 2**53 + 3, -(2**53 + 3), -2, -2], dtype=np.int64)

    a = numpy_helper.from_array(dividend_values, name="a")
    b = numpy_helper.from_array(divisor_values, name="b")
    node = helper.make_node("Div", ["a", "b"], ["y"])
    graph = helper.make_graph(
        [node],
        "div_integer_constant_precision",
        [],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [5])],
        initializer=[a, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9

    tvm_model = from_onnx(model, opset=18, keep_params_in_input=False)
    folded_outputs = []

    def collect_constants(expr):
        if isinstance(expr, tvm_tirx.ir.GenericConst):
            folded_outputs.append(expr.value.numpy())

    relax.analysis.post_order_visit(tvm_model["main"].body, collect_constants)
    assert len(folded_outputs) == 1
    np.testing.assert_array_equal(folded_outputs[0], expected)


@pytest.mark.parametrize(
    ("input_size", "divisor_shape", "offset"),
    [(386, [], None), (384, [1], 2)],
)
def test_div_integer_primexpr_folding_truncates_toward_zero(input_size, divisor_shape, offset):
    shape = helper.make_node("Shape", ["x"], ["x_shape"])
    axis = make_constant_node("axis", TensorProto.INT64, [], [0])
    dim = helper.make_node("Gather", ["x_shape", "axis"], ["dim"])
    nodes = [shape, axis, dim]
    dividend = "dim"
    if offset is not None:
        offset_node = make_constant_node("offset", TensorProto.INT64, [], [offset])
        shifted_dim = helper.make_node("Add", ["dim", "offset"], ["shifted_dim"])
        nodes.extend([offset_node, shifted_dim])
        dividend = "shifted_dim"

    divisor = make_constant_node("divisor", TensorProto.INT64, divisor_shape, [3])
    end = helper.make_node("Div", [dividend, "divisor"], ["end"])
    starts = make_constant_node("starts", TensorProto.INT64, [1], [0])
    axes = make_constant_node("axes", TensorProto.INT64, [1], [0])
    steps = make_constant_node("steps", TensorProto.INT64, [1], [1])
    slice_node = helper.make_node("Slice", ["x", "starts", "end", "axes", "steps"], ["y"])
    nodes.extend([divisor, end, starts, axes, steps, slice_node])

    graph = helper.make_graph(
        nodes,
        "div_integer_primexpr",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [input_size])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [128])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    tvm_model = from_onnx(model, opset=13)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((input_size,), dtype="float32"),
        ) -> R.Tensor((128,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((128,), dtype="float32") = R.strided_slice(
                    x, axes=[0], begin=[0], end=[128], strides=[1], assume_inbound=False
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("int_mode", [True, False])
def test_mod(int_mode: bool):
    if int_mode:
        dtype, fmod = TensorProto.INT32, 0
    else:
        dtype, fmod = TensorProto.FLOAT, 1
    verify_binary("Mod", [1, 32], [1, 32], [1, 32], attrs={"fmod": fmod}, dtype=dtype)
    verify_binary_scalar("Mod", attrs={"fmod": fmod}, dtype=dtype)


@pytest.mark.parametrize(
    "fmod, dtype, a_vals, b_vals",
    [
        (0, TensorProto.INT32, [-5, 5, -5, 5], [3, 3, -3, -3]),
        (1, TensorProto.INT32, [-5, 5, -5, 5], [3, 3, -3, -3]),
        (1, TensorProto.FLOAT, [-5.5, 5.5, -5.5, 5.5], [3.0, 3.0, -3.0, -3.0]),
    ],
)
def test_mod_constant_fold_negative_operands(fmod, dtype, a_vals, b_vals):
    """Mod over two constants is folded at import time. The folded value must
    match onnxruntime for negative operands, where integer mod (sign follows
    divisor) and fmod (sign follows dividend) disagree."""
    a = make_constant_node("a", dtype, [4], a_vals)
    b = make_constant_node("b", dtype, [4], b_vals)
    mod_node = helper.make_node("Mod", ["a", "b"], ["c"], fmod=fmod)
    graph = helper.make_graph(
        [a, b, mod_node],
        "mod_constant_fold_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("c", dtype, [4])],
    )
    model = helper.make_model(graph, producer_name="mod_constant_fold_test")
    check_correctness(model)


SHAPE_PARAMS = [
    ([[32, 32], [32, 32]], [32, 32]),
    ([[32, 1], [1, 2]], [32, 2]),
    (
        [
            [
                32,
            ],
            [
                1,
            ],
        ],
        [
            32,
        ],
    ),
    ([[32, 32, 1, 1], [1, 32, 32]], [32, 32, 32, 32]),
    (
        [
            [32, 32, 1, 1],
            [1, 32, 1],
            [
                32,
            ],
        ],
        [32, 32, 32, 32],
    ),
]


def test_multi_input_broadcasting():
    """Multi-input reductions should import broadcast + stack + reduce."""

    def verify_multi_input_broadcasting(op_name, input_shapes, expected_output_shape, expected):
        num_inputs = len(input_shapes)
        input_names = [f"i{i}" for i in range(num_inputs)]

        input_values_info = []
        for name, shape in zip(input_names, input_shapes):
            input_values_info.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        test_node = helper.make_node(op_name, input_names, ["output"])
        output_info = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, expected_output_shape
        )
        graph = helper.make_graph(
            [test_node],
            f"multi_input_{op_name}_test",
            inputs=input_values_info,
            outputs=[output_info],
        )
        model = helper.make_model(graph, producer_name="multi_input_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    def make_expected(op_name, input_shapes, output_shape):
        input_shapes = [tuple(shape) for shape in input_shapes]
        output_shape = tuple(output_shape)
        reduce_op = {
            "Min": R.min,
            "Max": R.max,
            "Sum": R.sum,
            "Mean": R.mean,
        }[op_name]
        input_shape_0 = input_shapes[0]
        input_shape_1 = input_shapes[1]

        if len(input_shapes) == 2:

            @I.ir_module
            class ExpectedMultiInputReduction2:
                @R.function
                def main(
                    i0: R.Tensor(input_shape_0, dtype="float32"),
                    i1: R.Tensor(input_shape_1, dtype="float32"),
                ) -> R.Tensor(output_shape, dtype="float32"):
                    R.func_attr({"num_input": 2})
                    with R.dataflow():
                        lv = R.broadcast_to(i0, R.shape(output_shape))
                        lv1 = R.broadcast_to(i1, R.shape(output_shape))
                        lv2 = R.stack((lv, lv1), axis=0)
                        gv = reduce_op(lv2, axis=[0], keepdims=False)
                        R.output(gv)
                    return gv

            return ExpectedMultiInputReduction2

        input_shape_2 = input_shapes[2]

        @I.ir_module
        class ExpectedMultiInputReduction3:
            @R.function
            def main(
                i0: R.Tensor(input_shape_0, dtype="float32"),
                i1: R.Tensor(input_shape_1, dtype="float32"),
                i2: R.Tensor(input_shape_2, dtype="float32"),
            ) -> R.Tensor(output_shape, dtype="float32"):
                R.func_attr({"num_input": 3})
                with R.dataflow():
                    lv = R.broadcast_to(i0, R.shape(output_shape))
                    lv1 = R.broadcast_to(i1, R.shape(output_shape))
                    lv2 = R.broadcast_to(i2, R.shape(output_shape))
                    lv3 = R.stack((lv, lv1, lv2), axis=0)
                    gv = reduce_op(lv3, axis=[0], keepdims=False)
                    R.output(gv)
                return gv

        return ExpectedMultiInputReduction3

    for input_shapes, output_shape in SHAPE_PARAMS:
        for op_name in ["Min", "Max", "Sum", "Mean"]:
            verify_multi_input_broadcasting(
                op_name,
                input_shapes,
                output_shape,
                make_expected(op_name, input_shapes, output_shape),
            )


@pytest.mark.parametrize(
    "op_name, dtype, shapes, values",
    [
        # Two rank-1 constants, the shape reported in apache/tvm#20117.
        ("Min", TensorProto.FLOAT, [[2], [2]], [[1.0, 2.0], [3.0, 4.0]]),
        ("Max", TensorProto.FLOAT, [[2], [2]], [[1.0, 2.0], [3.0, 4.0]]),
        ("Sum", TensorProto.FLOAT, [[2], [2]], [[1.0, 2.0], [3.0, 4.0]]),
        ("Mean", TensorProto.FLOAT, [[2], [2]], [[1.0, 2.0], [3.0, 4.0]]),
        # Sum and Mean accept only floating point operands in ONNX, so the
        # integer cases cover Min and Max. A rank-0 operand holding a valid
        # axis index is the input that returned a wrong answer rather than
        # raising, so it needs 0 or 1 here and not an out-of-range value.
        ("Min", TensorProto.INT64, [[3, 2], []], [[1, 2, 3, 4, 5, 6], [0]]),
        ("Max", TensorProto.INT64, [[3, 2], []], [[1, 2, 3, 4, 5, 6], [1]]),
    ],
)
def test_multi_input_all_constant_inputs(op_name, dtype, shapes, values):
    """Folding constant operands must match the broadcast + stack + reduce path."""
    nodes, names = [], []
    for i, (shape, value) in enumerate(zip(shapes, values)):
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[f"c{i}"],
                value=helper.make_tensor(f"c{i}_v", dtype, shape, value),
            )
        )
        names.append(f"c{i}")

    nodes.append(helper.make_node(op_name, names, ["output"]))
    output_shape = list(np.broadcast_shapes(*[tuple(shape) for shape in shapes]))
    graph = helper.make_graph(
        nodes,
        f"all_constant_{op_name}",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", dtype, output_shape)],
    )
    check_correctness(helper.make_model(graph), opset=13)


@pytest.mark.parametrize("op_name", ["Min", "Max", "Sum", "Mean"])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_multi_input_constant_rank_axis_bounds(op_name, rank):
    """All-constant inputs reduce over the new leading stack axis.

    Stacking the inputs along ``axis=0`` yields a tensor of rank ``rank+1``,
    so the reduction axis must lie in ``[-(rank+1), rank]``. ``axis=0`` equals
    the lower bound ``-(rank+1)`` and is therefore valid for every rank, while
    a positive ``axis=rank+1`` is out of bounds and must raise. The importer
    relies on this in ``MultiInputBase._impl_v1``, so both the numpy invariant
    and the end-to-end folding result are asserted here.
    """
    shape = (2,) * rank
    stacked = np.stack([np.ones(shape, np.float32), np.ones(shape, np.float32)], axis=0)
    reduce = {"Min": np.min, "Max": np.max, "Sum": np.sum, "Mean": np.mean}[op_name]
    assert stacked.ndim == rank + 1
    for axis in (0, -(rank + 1), rank, -rank):
        reduce(stacked, axis=axis)  # must not raise
    for axis in (rank + 1, -(rank + 2)):
        with pytest.raises((IndexError, ValueError)):
            reduce(stacked, axis=axis)

    # End-to-end: two identical rank-``rank`` constants fold to the identity.
    const_nodes = []
    for name in ("c0", "c1"):
        const_nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[name],
                value=helper.make_tensor(
                    f"{name}_v",
                    TensorProto.FLOAT,
                    list(shape),
                    np.ones(shape, np.float32).flatten().tolist(),
                ),
            )
        )
    graph = helper.make_graph(
        [*const_nodes, helper.make_node(op_name, ["c0", "c1"], ["output"])],
        f"const_rank_{op_name}",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(shape))],
    )
    check_correctness(helper.make_model(graph), opset=13)


def _make_onnx_min_max_model(
    x_shape: list,
    y_shape: list,
    out_shape: list,
    op_name: str = "Min",
    graph_name: str = "test_graph",
) -> onnx.ModelProto:
    """Construct a Min/Max ONNX model.

    Args:
        x_shape: Shape of the first input tensor
        y_shape: Shape of the second input tensor
        out_shape: Shape of the output tensor
        op_name: The operation name, either "Min" or "Max"
        graph_name: The name of the ONNX graph

    Returns:
        An ONNX ModelProto object representing the Min/Max operation.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)

    graph = helper.make_graph(
        [helper.make_node(op_name, ["x", "y"], ["out"])],
        graph_name,
        [x, y],
        [out],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])


def _make_expected_broadcast_ir_min(
    x_shape: tuple,
    y_shape: tuple,
):
    """Generate expected broadcast IR module for Min operation.

    Args:
        x_shape: Shape of the first input tensor
        y_shape: Shape of the second input tensor

    Returns:
        Expected IR module for the Min operation.
    """

    n = T.dynamic("n", "int64")

    @I.ir_module
    class ExpectedMin:
        @R.function
        def main(
            x: R.Tensor((n, x_shape[1]), dtype="float32"),
            y: R.Tensor((n, y_shape[1]), dtype="float32"),
        ) -> R.Tensor((n, 4), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.broadcast_to(x, R.shape((n, 4)))
                lv1 = R.broadcast_to(y, R.shape((n, 4)))
                lv2 = R.stack((lv, lv1), axis=0)
                gv = R.min(lv2, axis=[0], keepdims=False)
                R.output(gv)
            return gv

    return ExpectedMin


def _make_expected_broadcast_ir_max(
    x_shape: tuple,
    y_shape: tuple,
):
    """Generate expected broadcast IR module for Max operation.

    Args:
        x_shape: Shape of the first input tensor
        y_shape: Shape of the second input tensor

    Returns:
        Expected IR module for the Max operation.
    """

    n = T.dynamic("n", "int64")

    @I.ir_module
    class ExpectedMax:
        @R.function
        def main(
            x: R.Tensor((n, x_shape[1]), dtype="float32"),
            y: R.Tensor((n, y_shape[1]), dtype="float32"),
        ) -> R.Tensor((n, 4), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.broadcast_to(x, R.shape((n, 4)))
                lv1 = R.broadcast_to(y, R.shape((n, 4)))
                lv2 = R.stack((lv, lv1), axis=0)
                gv = R.max(lv2, axis=[0], keepdims=False)
                R.output(gv)
            return gv

    return ExpectedMax


def _test_symbolic_broadcast_case(
    x_shape: list,
    y_shape: list,
    out_shape: list,
    op_name: str,
    graph_name: str,
    should_pass: bool = True,
    error_pattern: str | None = None,
):
    """Execute a single symbolic broadcast test case.

    Args:
        x_shape: Shape of the first input tensor
        y_shape: Shape of the second input tensor
        out_shape: Shape of the output tensor
        op_name: The operation name, either "Min" or "Max"
        graph_name: The name of the ONNX graph
        should_pass: Whether the test is expected to pass or raise an error
        error_pattern: The expected error message pattern if should_pass is False
    """
    model = _make_onnx_min_max_model(x_shape, y_shape, out_shape, op_name, graph_name)
    onnx.checker.check_model(model)

    if should_pass:
        tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)
        if op_name == "Min":
            expected = _make_expected_broadcast_ir_min(tuple(x_shape), tuple(y_shape))
        else:
            expected = _make_expected_broadcast_ir_max(tuple(x_shape), tuple(y_shape))
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)
    else:
        with pytest.raises(ValueError, match=error_pattern):
            from_onnx(model, opset=18, keep_params_in_input=True)


def test_multi_input_broadcasting_symbolic_shapes():
    """Comprehensive test for symbolic shape broadcasting in Min/Max operations.

    Tests four levels:
    1. LEVEL 1 - Compile-time: Import and symbolic dimension handling (4 cases)
    2. LEVEL 2 - Shape lowering: Pass through LegalizeOps pipeline
    3. LEVEL 3 - IR verification: Check broadcast_to operations
    4. LEVEL 4 - Conservative rejection: Verify symbolic vs non-1 constant is rejected

    When dimensions are symbolic (not constant), broadcasting should handle them correctly
    without raising "Cannot use and / or / not operator to Expr".
    Conservative rejection prevents silent miscompilation where TOPI cannot correctly handle
    symbolic source dimensions in broadcast operations.
    """
    # LEVEL 1 - Compile-time: Import and symbolic dimension handling (4 cases)
    _test_symbolic_broadcast_case(
        x_shape=["n", 4],
        y_shape=["n", 4],
        out_shape=["n", 4],
        op_name="Min",
        graph_name="min_symbolic_broadcast",
    )

    _test_symbolic_broadcast_case(
        x_shape=["n", 4],
        y_shape=["n", 4],
        out_shape=["n", 4],
        op_name="Max",
        graph_name="max_symbolic_broadcast",
    )

    _test_symbolic_broadcast_case(
        x_shape=["n", 1],
        y_shape=["n", 4],
        out_shape=["n", 4],
        op_name="Min",
        graph_name="min_mixed_symbolic_broadcast",
    )

    _test_symbolic_broadcast_case(
        x_shape=["n", 4],
        y_shape=["m", 4],
        out_shape=["n", 4],
        op_name="Min",
        graph_name="min_different_symbolic",
        should_pass=False,
        error_pattern="Cannot broadcast symbolic dimensions",
    )

    # LEVEL 2 - Shape lowering: Pass through LegalizeOps pipeline
    model_lower = _make_onnx_min_max_model(
        ["n", 4], ["n", 4], ["n", 4], "Min", "min_shape_lower_test"
    )
    onnx.checker.check_model(model_lower)

    tvm_model = from_onnx(model_lower, opset=18, keep_params_in_input=True)

    tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
    tvm_model = relax.transform.LegalizeOps()(tvm_model)

    tvm_model, _ = relax.frontend.detach_params(tvm_model)

    # LEVEL 3 - IR verification: Check broadcast_to operations
    ir_str = str(tvm_model)
    assert "broadcast" in ir_str, "IR should contain broadcast operation"

    # LEVEL 4 - Conservative rejection: Verify symbolic vs non-1 constant is rejected
    _test_symbolic_broadcast_case(
        x_shape=["n", 4],
        y_shape=[2, 4],
        out_shape=[2, 4],
        op_name="Min",
        graph_name="min_symbolic_vs_constant",
        should_pass=False,
        error_pattern="runtime value of symbolic dimension is unknown",
    )


@pytest.mark.parametrize("constant_inputs", [False, True])
def test_multi_input_broadcasting_zero_extent(constant_inputs):
    # Both orders exercise zero/unit, zero/zero and rank-padding in the shared
    # MultiInputBase shape calculation used by Min, Max, Sum and Mean.
    inputs = {
        name: np.zeros(shape, dtype="float32")
        for name, shape in [("empty", (0, 3)), ("unit", (1, 3)), ("vector", (3,))]
    }
    graph = helper.make_graph(
        [
            helper.make_node("Sum", ["empty", "unit", "vector", "empty"], ["forward"]),
            helper.make_node("Sum", ["vector", "unit", "empty", "empty"], ["reverse"]),
        ],
        "multi_input_broadcasting_zero_extent",
        []
        if constant_inputs
        else [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape))
            for name, value in inputs.items()
        ],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [0, 3])
            for name in ["forward", "reverse"]
        ],
        initializer=[numpy_helper.from_array(value, name) for name, value in inputs.items()]
        if constant_inputs
        else [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    actual_inputs = {} if constant_inputs else inputs
    check_correctness(model, actual_inputs, opset=18, check_dtypes=True)
    if constant_inputs:
        # Exercise constant folding as well as initializer parameters kept in the VM ABI.
        mod = from_onnx(model, keep_params_in_input=False)
        executable = tvm.compile(mod, target="llvm")
        outputs = relax.VirtualMachine(executable, tvm.cpu())["main"]()
        for output in outputs:
            assert tuple(output.shape) == (0, 3)
            assert output.numpy().dtype == np.dtype("float32")


@pytest.mark.parametrize("op_name", ["Min", "Max", "Sum", "Mean"])
@pytest.mark.parametrize("shape", [[], [5], [2, 3]])
def test_multi_input_constant_single_input(op_name, shape):
    """A single constant input is returned unchanged (shape preserved)."""
    if shape:
        vals = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    else:
        vals = np.array(1.0, np.float32)
    node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c0"],
        value=helper.make_tensor("c0_v", TensorProto.FLOAT, shape, vals.flatten().tolist()),
    )
    graph = helper.make_graph(
        [node, helper.make_node(op_name, ["c0"], ["output"])],
        f"const_single_{op_name}",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    check_correctness(helper.make_model(graph), opset=13)


@pytest.mark.parametrize("op_name", ["Min", "Max", "Sum", "Mean"])
@pytest.mark.parametrize("num_inputs", [1, 2, 3])
def test_multi_input_unknown_static_shape(op_name, num_inputs):
    """An input with no static shape imports instead of raising len(None).

    A Slice with runtime starts/ends lowers to R.dynamic_strided_slice, whose
    struct info is R.Tensor(dtype=..., ndim=k) with shape None. That reaches
    MultiInputBase, where compute_broadcast_shape used to call len() on it.
    The model is valid ONNX and onnxruntime executes it.
    """
    slice_node = helper.make_node(
        "Slice", ["x", "starts", "ends", "axes"], ["sliced"], name="slice0"
    )
    other_names = [f"y{i}" for i in range(num_inputs - 1)]
    op_node = helper.make_node(op_name, ["sliced", *other_names], ["output"], name="op0")

    graph = helper.make_graph(
        [slice_node, op_node],
        f"slice_then_{op_name.lower()}",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 3]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1]),
        ]
        + [helper.make_tensor_value_info(name, TensorProto.FLOAT, [2, 3]) for name in other_names],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model, full_check=True)

    tvm_model = from_onnx(model, keep_params_in_input=True)
    assert "dynamic_strided_slice" in str(tvm_model)


@pytest.mark.parametrize("op_name", ["And", "Or", "Xor"])
def test_binary_bool(op_name: str):
    verify_binary(op_name, [32, 32], [32, 32], [32, 32], dtype=TensorProto.BOOL)


def test_bitwise_not():
    verify_unary(
        "BitwiseNot",
        [32, 32],
        input_dtype=TensorProto.UINT64,
        output_dtype=TensorProto.UINT64,
        opset=18,
    )


@pytest.mark.parametrize(
    "op_name",
    [
        "Sin",
        "Cos",
        "Tan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Asin",
        "Acos",
        "Atan",
        "Asinh",
        "Acosh",
        "Atanh",
        "Neg",
        "Abs",
        "Log",
        "Exp",
        "Not",
        "Floor",
        "Ceil",
        "Round",
        "IsInf",
        "IsNaN",
        "Sqrt",
        "Relu",
        "Sign",
        "Softplus",
        "Erf",
        "Sigmoid",
        "Softmax",
        "LogSoftmax",
    ],
)
def test_unary(op_name: str):
    input_dtype = TensorProto.FLOAT
    if op_name in [
        "IsNaN",
        "IsInf",
    ]:
        pytest.skip(f"Skipping test {op_name} because current LegalizeOps does not support it.")
    elif op_name == "Not":
        input_dtype = TensorProto.BOOL
        output_dtype = TensorProto.BOOL
    else:
        output_dtype = TensorProto.FLOAT
    verify_unary(op_name, [8, 8, 8], input_dtype=input_dtype, output_dtype=output_dtype)


def test_reciprocal_ir():
    model = make_unary_model("Reciprocal", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.divide(R.const(1.0, "float32"), x)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_identity_ir():
    model = make_unary_model("Identity", [8, 8, 8])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 8, 8), dtype="float32")) -> R.Tensor((8, 8, 8), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((8, 8, 8), dtype="float32") = x
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_elu_ir():
    model = make_unary_model("Elu", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(R.const(-1.0, "float32"), lv2)
                lv4: R.Tensor((2, 3), dtype="float32") = R.nn.relu(x)
                gv: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv4)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_hardswish_ir():
    model = make_unary_model("HardSwish", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.add(x, R.const(3.0, "float32"))
                lv1: R.Tensor((2, 3), dtype="float32") = R.clip(
                    lv, R.prim_value(0), R.prim_value(6)
                )
                lv2: R.Tensor((2, 3), dtype="float32") = R.divide(lv1, R.const(6.0, "float32"))
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(x, lv2)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_softsign_ir():
    model = make_unary_model("Softsign", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.abs(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.add(lv, R.const(1.0, "float32"))
                gv: R.Tensor((2, 3), dtype="float32") = R.divide(x, lv1)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_hardmax_ir():
    model = make_unary_model("Hardmax", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="int64") = R.argmax(x, axis=1, keepdims=False)
                gv: R.Tensor((2, 3), dtype="float32") = R.one_hot(
                    lv,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=3,
                    axis=1,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("opset", [11, 13])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("symbolic", [False, True], ids=["static", "symbolic"])
def test_hardmax(opset, axis, symbolic):
    shape = [2, 3, 4]
    model = make_unary_model(
        "Hardmax", ["N", "C", "W"] if symbolic else shape, attrs={"axis": axis}
    )
    # Few distinct values, so rows tie and the first maximum must win.
    x = rg.integers(0, 3, size=shape).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=opset)


def test_legacy_softmax_family_opset11_axis_semantics():
    def verify_legacy_softmax_family_axis_ir(op_name: str, expected, axis_attr: int | None = None):
        attrs = {} if axis_attr is None else {"axis": axis_attr}
        node = helper.make_node(op_name, ["x"], ["y"], **attrs)
        graph = helper.make_graph(
            [node],
            "legacy_softmax_family_axis_ir_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
        )
        model = helper.make_model(
            graph,
            producer_name="legacy_softmax_family_axis_ir_test",
            opset_imports=[helper.make_opsetid("", 11)],
        )
        tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedSoftmaxAxis0:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 24), dtype="float32") = R.reshape(x, R.shape([1, 24]))
                lv1: R.Tensor((1, 24), dtype="float32") = R.nn.softmax(lv, axis=-1)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSoftmaxAxis1:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                lv1: R.Tensor((2, 12), dtype="float32") = R.nn.softmax(lv, axis=-1)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSoftmaxAxisRank:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((24, 1), dtype="float32") = R.reshape(x, R.shape([24, 1]))
                lv1: R.Tensor((24, 1), dtype="float32") = R.nn.softmax(lv, axis=-1)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedLogSoftmaxAxis0:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 24), dtype="float32") = R.reshape(x, R.shape([1, 24]))
                lv1: R.Tensor((1, 24), dtype="float32") = R.nn.log_softmax(lv, axis=-1)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedLogSoftmaxAxis1:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                lv1: R.Tensor((2, 12), dtype="float32") = R.nn.log_softmax(lv, axis=-1)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedLogSoftmaxAxisRank:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((24, 1), dtype="float32") = R.reshape(x, R.shape([24, 1]))
                lv1: R.Tensor((24, 1), dtype="float32") = R.nn.log_softmax(lv, axis=-1)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedHardmaxAxis0:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 24), dtype="float32") = R.reshape(x, R.shape([1, 24]))
                lv1: R.Tensor((1,), dtype="int64") = R.argmax(lv, axis=1, keepdims=False)
                lv2: R.Tensor((1, 24), dtype="float32") = R.one_hot(
                    lv1,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=24,
                    axis=1,
                )
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv2, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedHardmaxAxis1:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                lv1: R.Tensor((2,), dtype="int64") = R.argmax(lv, axis=1, keepdims=False)
                lv2: R.Tensor((2, 12), dtype="float32") = R.one_hot(
                    lv1,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=12,
                    axis=1,
                )
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv2, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedHardmaxAxisRank:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((24, 1), dtype="float32") = R.reshape(x, R.shape([24, 1]))
                lv1: R.Tensor((24,), dtype="int64") = R.argmax(lv, axis=1, keepdims=False)
                lv2: R.Tensor((24, 1), dtype="float32") = R.one_hot(
                    lv1,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=1,
                    axis=1,
                )
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv2, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    # Default axis and equivalent negative axis both flatten from axis 1.
    verify_legacy_softmax_family_axis_ir("Softmax", ExpectedSoftmaxAxis1)
    verify_legacy_softmax_family_axis_ir("LogSoftmax", ExpectedLogSoftmaxAxis1)
    verify_legacy_softmax_family_axis_ir("Hardmax", ExpectedHardmaxAxis1)
    verify_legacy_softmax_family_axis_ir("Softmax", ExpectedSoftmaxAxis1, axis_attr=-2)
    verify_legacy_softmax_family_axis_ir("LogSoftmax", ExpectedLogSoftmaxAxis1, axis_attr=-2)
    verify_legacy_softmax_family_axis_ir("Hardmax", ExpectedHardmaxAxis1, axis_attr=-2)

    # Positive axis 0 flattens the whole input as one row.
    verify_legacy_softmax_family_axis_ir("Softmax", ExpectedSoftmaxAxis0, axis_attr=0)
    verify_legacy_softmax_family_axis_ir("LogSoftmax", ExpectedLogSoftmaxAxis0, axis_attr=0)
    verify_legacy_softmax_family_axis_ir("Hardmax", ExpectedHardmaxAxis0, axis_attr=0)

    # Axis equal to rank produces a trailing singleton reduction dimension.
    verify_legacy_softmax_family_axis_ir("Softmax", ExpectedSoftmaxAxisRank, axis_attr=3)
    verify_legacy_softmax_family_axis_ir("LogSoftmax", ExpectedLogSoftmaxAxisRank, axis_attr=3)
    verify_legacy_softmax_family_axis_ir("Hardmax", ExpectedHardmaxAxisRank, axis_attr=3)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax"])
def test_softmax_family_opset13_default_axis_semantics(op_name: str):
    verify_unary(op_name, [2, 3, 4], opset=13)


def test_hardmax_opset13_default_axis_ir():
    model = make_unary_model("Hardmax", [2, 3, 4])
    model.opset_import[0].version = 13
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="int64") = R.argmax(x, axis=2, keepdims=False)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.one_hot(
                    lv,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=4,
                    axis=2,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_legacy_softmax_family_opset1_ir_semantics():
    def verify_legacy_softmax_family_opset1_ir(op_name: str, expected):
        node = helper.make_node(op_name, ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "legacy_softmax_family_opset1_ir_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
        )
        model = helper.make_model(
            graph,
            producer_name="legacy_softmax_family_opset1_ir_test",
            opset_imports=[helper.make_opsetid("", 1)],
        )
        tvm_model = from_onnx(model, opset=1, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedSoftmax:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                lv1: R.Tensor((2, 12), dtype="float32") = R.nn.softmax(lv, axis=-1)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedLogSoftmax:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                lv1: R.Tensor((2, 12), dtype="float32") = R.nn.log_softmax(lv, axis=-1)
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv1, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedHardmax:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((2, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 12), dtype="float32") = R.reshape(x, R.shape([2, 12]))
                lv1: R.Tensor((2,), dtype="int64") = R.argmax(lv, axis=1, keepdims=False)
                lv2: R.Tensor((2, 12), dtype="float32") = R.one_hot(
                    lv1,
                    R.prim_value(T.float32(1.0)),
                    R.prim_value(T.float32(0.0)),
                    depth=12,
                    axis=1,
                )
                gv: R.Tensor((2, 3, 4), dtype="float32") = R.reshape(lv2, R.shape([2, 3, 4]))
                R.output(gv)
            return gv

    verify_legacy_softmax_family_opset1_ir("Softmax", ExpectedSoftmax)
    verify_legacy_softmax_family_opset1_ir("LogSoftmax", ExpectedLogSoftmax)
    verify_legacy_softmax_family_opset1_ir("Hardmax", ExpectedHardmax)


@pytest.mark.parametrize("from_type", [TensorProto.INT32, TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("to_type", [TensorProto.INT32, TensorProto.FLOAT, TensorProto.FLOAT16])
def test_cast(from_type, to_type):
    cast_node = helper.make_node("Cast", ["a"], ["a_float"], to=to_type)

    graph = helper.make_graph(
        [cast_node],
        "cast_test",
        inputs=[
            helper.make_tensor_value_info("a", from_type, [1, 32]),
        ],
        outputs=[helper.make_tensor_value_info("a_float", to_type, [1, 32])],
    )

    model = helper.make_model(graph, producer_name="cast_test")
    check_correctness(model, opset=13)


@pytest.mark.parametrize("to_type", [TensorProto.INT64, TensorProto.UINT64])
def test_cast_float_to_64bit_int_dynamic(to_type):
    cast_node = helper.make_node("Cast", ["a"], ["b"], to=to_type)
    graph = helper.make_graph(
        [cast_node],
        "cast_float_to_64bit_int_dynamic_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 8])],
        outputs=[helper.make_tensor_value_info("b", to_type, [1, 8])],
    )
    model = helper.make_model(graph, producer_name="cast_float_to_64bit_int_dynamic_test")
    inputs = {"a": np.array([[0.0, 1.2, 2.8, 7.9, 15.1, 31.7, 63.4, 127.9]], dtype=np.float32)}
    check_correctness(model, inputs=inputs, opset=13, check_dtypes=True)


def test_cast_nan_inf_to_int8():
    vals = np.array([300.0, np.nan, np.inf, -np.inf, 50.0, -50.0], dtype=np.float32)
    node = helper.make_node("Cast", inputs=["a"], outputs=["b"], to=TensorProto.INT8)
    graph = helper.make_graph(
        [node],
        "cast_nan_inf_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, list(vals.shape))],
        outputs=[helper.make_tensor_value_info("b", TensorProto.INT8, list(vals.shape))],
    )
    model = helper.make_model(graph, producer_name="cast_nan_inf_test")
    tvm_output = run_in_tvm(model, inputs={"a": vals}, opset=13)
    out_np = tvm_output.numpy()
    expected = np.array([44, 0, 0, 0, 50, -50], dtype=np.int8)
    assert out_np.dtype == np.int8
    np.testing.assert_array_equal(out_np, expected)


def test_castlike_ir():
    castlike_node = helper.make_node("CastLike", ["a", "b"], ["c"])
    graph = helper.make_graph(
        [castlike_node],
        "castlike_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.INT32, [1, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 32])],
    )
    model = helper.make_model(graph, producer_name="castlike_test")
    tvm_model = from_onnx(model, opset=15, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((1, 32), dtype="int32"),
            b: R.Tensor((1,), dtype="float32"),
        ) -> R.Tensor((1, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((1, 32), dtype="float32") = R.astype(a, "float32")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_castlike_nan_inf_to_int8():
    vals = np.array([np.nan, np.inf, -np.inf, 1.9, -1.9], dtype=np.float32)
    castlike_node = helper.make_node("CastLike", ["data", "like"], ["output"])
    graph = helper.make_graph(
        [castlike_node],
        "castlike_nan_inf_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(vals.shape)),
            helper.make_tensor_value_info("like", TensorProto.INT8, [1]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT8, list(vals.shape))],
    )
    model = helper.make_model(graph, producer_name="castlike_nan_inf_test")
    inputs = {"data": vals, "like": np.array([0], dtype=np.int8)}
    check_correctness(model, inputs=inputs, opset=15, check_dtypes=True)


@pytest.mark.parametrize("symbolic", [False, True], ids=["static", "symbolic"])
def test_castlike_shape_expr_data(symbolic: bool):
    shape_node = helper.make_node("Shape", ["data"], ["data_shape"])
    castlike_node = helper.make_node("CastLike", ["data_shape", "like"], ["output"])
    data_shape = ["n", 3] if symbolic else [2, 3]
    graph = helper.make_graph(
        [shape_node, castlike_node],
        "castlike_shape_data_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("like", TensorProto.FLOAT, [1]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, producer_name="castlike_shape_data_test")
    inputs = None
    if symbolic:
        inputs = {
            "data": np.ones((2, 3), dtype="float32"),
            "like": np.zeros((1,), dtype="float32"),
        }
    check_correctness(model, inputs=inputs, opset=15, check_dtypes=True)


def test_castlike_shape_expr_target():
    shape_node = helper.make_node("Shape", ["like"], ["like_shape"])
    castlike_node = helper.make_node("CastLike", ["data", "like_shape"], ["output"])
    graph = helper.make_graph(
        [shape_node, castlike_node],
        "castlike_shape_target_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("like", TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [3])],
    )
    model = helper.make_model(graph, producer_name="castlike_shape_target_test")
    check_correctness(model, opset=15, check_dtypes=True)


def test_gather():
    def _verify_gather(data_shape, indices, out_shape, expected, axis=0):
        gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=axis)

        if isinstance(indices, list | tuple):
            indices_shape = np.asarray(indices).shape
        else:
            indices_shape = []

        graph = helper.make_graph(
            [gather_node],
            "gather_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(
            graph, producer_name="gather_test", opset_imports=[helper.make_opsetid("", 14)]
        )
        tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedRank4Axis0:
        @R.function
        def main(
            data: R.Tensor((5, 4, 3, 2), dtype="float32"),
            indices: R.Tensor((3,), dtype="int64"),
        ) -> R.Tensor((3, 4, 3, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Shape([5, 4, 3, 2]) = R.shape_of(data)
                lv1: R.Tensor((4,), dtype="int64") = R.shape_to_tensor(lv)
                lv2: R.Tensor((3,), dtype="bool") = R.less(indices, R.const(0, "int64"))
                lv3: R.Tensor((), dtype="int64") = R.take(
                    lv1, R.const(0, "int64"), axis=0, mode="wrap"
                )
                lv4: R.Tensor((3,), dtype="int64") = R.add(indices, lv3)
                lv5: R.Tensor((3,), dtype="int64") = R.where(lv2, lv4, indices)
                gv: R.Tensor((3, 4, 3, 2), dtype="float32") = R.take(data, lv5, axis=0, mode="fast")
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedScalarIndex:
        @R.function
        def main(
            data: R.Tensor((3,), dtype="float32"),
            indices: R.Tensor((), dtype="int64"),
        ) -> R.Tensor((), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Shape([3]) = R.shape_of(data)
                lv1: R.Tensor((1,), dtype="int64") = R.shape_to_tensor(lv)
                lv2: R.Tensor((), dtype="bool") = R.less(indices, R.const(0, "int64"))
                lv3: R.Tensor((), dtype="int64") = R.take(
                    lv1, R.const(0, "int64"), axis=0, mode="wrap"
                )
                lv4: R.Tensor((), dtype="int64") = R.add(indices, lv3)
                lv5: R.Tensor((), dtype="int64") = R.where(lv2, lv4, indices)
                gv: R.Tensor((), dtype="float32") = R.take(data, lv5, axis=0, mode="fast")
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedRank2Axis1:
        @R.function
        def main(
            data: R.Tensor((3, 3), dtype="float32"),
            indices: R.Tensor((1, 2), dtype="int64"),
        ) -> R.Tensor((3, 1, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Shape([3, 3]) = R.shape_of(data)
                lv1: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                lv2: R.Tensor((1, 2), dtype="bool") = R.less(indices, R.const(0, "int64"))
                lv3: R.Tensor((), dtype="int64") = R.take(
                    lv1, R.const(1, "int64"), axis=0, mode="wrap"
                )
                lv4: R.Tensor((1, 2), dtype="int64") = R.add(indices, lv3)
                lv5: R.Tensor((1, 2), dtype="int64") = R.where(lv2, lv4, indices)
                gv: R.Tensor((3, 1, 2), dtype="float32") = R.take(data, lv5, axis=1, mode="fast")
                R.output(gv)
            return gv

    _verify_gather([5, 4, 3, 2], [0, 1, 3], [3, 4, 3, 2], ExpectedRank4Axis0)
    _verify_gather([3], 0, [], ExpectedScalarIndex)
    _verify_gather([3, 3], [[0, 2]], [3, 1, 2], ExpectedRank2Axis1, 1)


def test_gather_indices_from_shape():
    """Gather from a tensor using the dimensions of another tensor as indices."""
    shape_node = helper.make_node("Shape", ["shape_source"], ["indices"])
    gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=0)

    graph = helper.make_graph(
        [shape_node, gather_node],
        "gather_indices_from_shape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("shape_source", TensorProto.FLOAT, [2, 3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )

    model = helper.make_model(
        graph,
        producer_name="gather_indices_from_shape_test",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_values = {
        "data": np.random.randn(4).astype("float32"),
        "shape_source": np.random.randn(2, 3).astype("float32"),
    }
    check_correctness(model, inputs=input_values, opset=18)


@pytest.mark.parametrize("index", [0, 2, 3, -1, -4])
def test_gather_shape_dynamic_index(index):
    """Gather a dimension out of a Shape result using a non-constant index.

    Detection post-processing graphs (e.g. FasterRCNN) feed a runtime-computed
    index into a Gather whose data is a Shape output. The index is not a
    constant, so the importer must materialize the shape as a tensor and gather
    from it at runtime rather than resolving the dimension at compile time.
    """
    data_shape = [3, 4, 5, 6]
    shape_node = helper.make_node("Shape", ["data"], ["shape"])
    gather_node = helper.make_node("Gather", ["shape", "index"], ["y"], axis=0)

    graph = helper.make_graph(
        [shape_node, gather_node],
        "gather_shape_dynamic_index_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("index", TensorProto.INT64, []),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT64, [])],
    )

    model = helper.make_model(graph, producer_name="gather_shape_dynamic_index_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "index": np.array(index).astype("int64"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize(
    "indices",
    [
        np.array(2, dtype="int64"),  # 0-D scalar -> PrimValue fast path
        np.array([2], dtype="int64"),  # (1,) -> must stay rank 1
        np.array([[2]], dtype="int64"),  # (1, 1) -> must stay rank 2
        np.array([1, 3], dtype="int64"),  # (2,) -> multiple dims
        np.array([-1], dtype="int64"),  # (1,) negative index
    ],
)
def test_gather_shape_constant_index(indices):
    """Gather from a Shape result using a constant index of varying rank.

    ONNX Gather defines the output rank as q + r - 1 where q is the rank of the
    indices. Since a Shape output is rank 1, only a true 0-D scalar index should
    collapse to a scalar; a (1,) index must produce a rank-1 result rather than
    being folded into a PrimValue.
    """
    data_shape = [3, 4, 5, 6]
    shape_node = helper.make_node("Shape", ["data"], ["shape"])
    # Emit the indices through a Constant node: an initializer would become a
    # function parameter under keep_params_in_input=True and bypass the
    # constant fast path in the Gather converter.
    const_node = helper.make_node(
        "Constant",
        [],
        ["indices"],
        value=helper.make_tensor(
            "value",
            TensorProto.INT64,
            indices.shape,
            indices.flatten().tolist(),
        ),
    )
    gather_node = helper.make_node("Gather", ["shape", "indices"], ["y"], axis=0)

    graph = helper.make_graph(
        [shape_node, const_node, gather_node],
        "gather_shape_constant_index_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT64, list(indices.shape))],
    )

    model = helper.make_model(graph, producer_name="gather_shape_constant_index_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
    }
    check_correctness(model, inputs=input_values)

    # check_correctness broadcasts a scalar against a one-element tensor, so
    # assert the rank explicitly: only a 0-D index may collapse to a scalar.
    tvm_out = run_in_tvm(model, inputs=input_values)
    if isinstance(tvm_out, tvm_tirx.runtime.Tensor):
        out_shape = tuple(tvm_out.numpy().shape)
    else:
        # PrimValue fast-path outputs come back as plain Python scalars.
        out_shape = ()
    assert out_shape == indices.shape


def _make_gather_negative_indices_expected(axis: int, indices_shape, indices_type):
    indices_shape = tuple(indices_shape)
    indices_dtype = "int64" if indices_type == TensorProto.INT64 else "int32"

    if indices_type == TensorProto.INT64:

        @I.ir_module
        class ExpectedGatherNegativeInt64:
            @R.function
            def main(
                data: R.Tensor((3, 4), dtype="float32"),
                indices: R.Tensor(indices_shape, dtype=indices_dtype),
            ):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Shape([3, 4]) = R.shape_of(data)
                    lv1: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                    lv2 = R.less(indices, R.const(0, "int64"))
                    lv3: R.Tensor((), dtype="int64") = R.take(
                        lv1, R.const(axis, "int64"), axis=0, mode="wrap"
                    )
                    lv4 = R.add(indices, lv3)
                    lv5 = R.where(lv2, lv4, indices)
                    gv = R.take(data, lv5, axis=axis, mode="fast")
                    R.output(gv)
                return gv

        return ExpectedGatherNegativeInt64

    if indices_type == TensorProto.INT32:

        @I.ir_module
        class ExpectedGatherNegativeInt32:
            @R.function
            def main(
                data: R.Tensor((3, 4), dtype="float32"),
                indices: R.Tensor(indices_shape, dtype=indices_dtype),
            ):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Shape([3, 4]) = R.shape_of(data)
                    lv1: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                    lv2: R.Tensor((), dtype="int64") = R.take(
                        lv1, R.const(axis, "int64"), axis=0, mode="wrap"
                    )
                    lv3 = R.less(indices, R.const(0, "int32"))
                    lv4: R.Tensor((), dtype="int32") = R.astype(lv2, dtype="int32")
                    lv5 = R.add(indices, lv4)
                    lv6 = R.where(lv3, lv5, indices)
                    gv = R.take(data, lv6, axis=axis, mode="fast")
                    R.output(gv)
                return gv

        return ExpectedGatherNegativeInt32

    raise AssertionError(
        f"Unexpected Gather negative-index case: axis={axis}, "
        f"indices_shape={indices_shape}, indices_type={indices_type}"
    )


def test_gather_negative_indices():
    def verify_gather_negative_indices(axis, indices, out_shape, indices_type, expected):
        gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=axis)
        indices_shape = np.asarray(indices).shape

        graph = helper.make_graph(
            [gather_node],
            "gather_negative_indices_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4]),
                helper.make_tensor_value_info("indices", indices_type, indices_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(
            graph,
            producer_name="gather_negative_indices_test",
            opset_imports=[helper.make_opsetid("", 14)],
        )
        tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    for axis, indices, out_shape, indices_type in [
        (0, [-1, 0], [2, 4], TensorProto.INT64),
        (1, [-1, 0], [3, 2], TensorProto.INT64),
        (1, [[-1, 0], [1, -2]], [3, 2, 2], TensorProto.INT64),
        (0, [-1, 0], [2, 4], TensorProto.INT32),
        (1, [-1, 0], [3, 2], TensorProto.INT32),
        (1, [[-1, 0], [1, -2]], [3, 2, 2], TensorProto.INT32),
    ]:
        verify_gather_negative_indices(
            axis,
            indices,
            out_shape,
            indices_type,
            _make_gather_negative_indices_expected(axis, np.asarray(indices).shape, indices_type),
        )


@pytest.mark.parametrize(
    "data_shape, indices_shape, axis",
    [
        ([3, 4, 5], [1, 4, 5], 0),
        ([3, 4, 5], [3, 2, 5], 1),
        ([3, 4, 5], [3, 4, 2], 2),
    ],
)
def test_gather_elements(data_shape, indices_shape, axis):
    gather_elements_node = helper.make_node("GatherElements", ["data", "indices"], ["y"], axis=axis)

    graph = helper.make_graph(
        [gather_elements_node],
        "gather_elements_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, indices_shape)],
    )

    model = helper.make_model(graph, producer_name="gather_elements_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "indices": np.random.randint(0, data_shape[axis], indices_shape).astype("int64"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize(
    "data_shape, indices_shape, axis",
    [
        ([3, 4, 5], [1, 4, 5], 0),
        ([3, 4, 5], [3, 2, 5], 1),
        ([3, 4, 5], [3, 4, 2], 2),
    ],
)
def test_gather_elements_negative_indices(data_shape, indices_shape, axis):
    gather_elements_node = helper.make_node("GatherElements", ["data", "indices"], ["y"], axis=axis)

    graph = helper.make_graph(
        [gather_elements_node],
        "gather_elements_negative_indices_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, indices_shape)],
    )

    model = helper.make_model(graph, producer_name="gather_elements_negative_indices_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "indices": np.random.randint(-data_shape[axis], data_shape[axis], indices_shape).astype(
            "int64"
        ),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize(
    "data_shape, indices_shape, batch_dims",
    [
        ([2, 2], [2, 2], 0),
        ([2, 2], [2, 1], 0),
        ([2, 2, 2], [1], 0),
        ([2, 2, 2], [2, 2], 0),
        ([2, 2, 2], [2, 1, 2], 0),
        ([2, 2, 2], [2, 2], 1),
        ([2, 2, 2], [2, 1], 1),
    ],
)
def test_gather_nd(data_shape, indices_shape, batch_dims):
    gather_nd_node = helper.make_node("GatherND", ["data", "indices"], ["y"], batch_dims=batch_dims)

    graph = helper.make_graph(
        [gather_nd_node],
        "gather_nd_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )

    model = helper.make_model(graph, producer_name="gather_nd_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "indices": np.random.randint(0, 2, indices_shape).astype("int64"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize(
    "data_shape, indices_shape, batch_dims",
    [
        ([2, 2], [2, 2], 0),
        ([2, 2], [2, 1], 0),
        ([2, 2, 2], [1], 0),
        ([2, 2, 2], [2, 2], 0),
        ([2, 2, 2], [2, 1, 2], 0),
        ([2, 2, 2], [2, 2], 1),
        ([2, 2, 2], [2, 1], 1),
    ],
)
def test_gather_nd_negative_indices(data_shape, indices_shape, batch_dims):
    gather_nd_node = helper.make_node("GatherND", ["data", "indices"], ["y"], batch_dims=batch_dims)

    graph = helper.make_graph(
        [gather_nd_node],
        "gather_nd_negative_indices_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )

    model = helper.make_model(graph, producer_name="gather_nd_negative_indices_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "indices": np.random.randint(-2, 2, indices_shape).astype("int64"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize(("name", "opset"), [("Scatter", 10), ("ScatterElements", 11)])
def test_scatter(axis: int, name: str, opset: int):
    if name == "ScatterElements" and axis != 1:
        pytest.skip(
            "ScatterElements with indices smaller than data is lowered via scatter_elements, "
            "which only works for axis=1"
        )
    input_shape = [16, 16, 16]
    indices_shape = [8, 8, 8]
    updates_shape = [8, 8, 8]
    output_shape = [16, 16, 16]
    node = helper.make_node(name, ["data", "indices", "updates"], ["output"], axis=axis)
    graph = helper.make_graph(
        [node],
        "scatter_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="scatter_test")
    indices = np.random.randint(0, 16, indices_shape)
    check_correctness(model, inputs={"indices": indices}, opset=opset)


@pytest.mark.parametrize("indices_dtype", ["int32", "int64"])
def test_scatter_broadcast(indices_dtype):
    """Scatter (opset 9/10) whose indices/updates are smaller than data (size-1
    broadcast dims) must follow the per-entry semantics of the ONNX spec:
    output[idx[:axis] + (indices[idx],) + idx[axis+1:]] = updates[idx] for each
    entry idx in indices' own shape. The previous lowering passed such indices
    directly to scatter_elements, which silently produced wrong values. Both
    ONNX-permitted indices dtypes (int32/int64) are covered, since the lowered
    coordinate grid must match the indices dtype."""
    data_shape = (2, 3, 4)
    indices_proto = TensorProto.INT32 if indices_dtype == "int32" else TensorProto.INT64
    rng = np.random.RandomState(0)
    for axis in [0, 1, 2, -1]:
        for indices_shape in [(1, 3, 4), (2, 1, 4), (1, 3, 1), (1, 1, 1)]:
            graph = helper.make_graph(
                [
                    helper.make_node(
                        "Scatter", ["data", "indices", "updates"], ["output"], axis=axis
                    )
                ],
                "scatter_broadcast_test",
                inputs=[
                    helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                    helper.make_tensor_value_info("indices", indices_proto, list(indices_shape)),
                    helper.make_tensor_value_info(
                        "updates", TensorProto.FLOAT, list(indices_shape)
                    ),
                ],
                outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
            )
            model = helper.make_model(
                graph,
                producer_name="scatter_broadcast_test",
                opset_imports=[helper.make_opsetid("", 9)],
            )
            inputs = {
                "data": rng.randn(*data_shape).astype("float32"),
                "indices": rng.randint(0, data_shape[axis % len(data_shape)], indices_shape).astype(
                    indices_dtype
                ),
                "updates": rng.randn(*indices_shape).astype("float32"),
            }
            check_correctness(model, inputs=inputs, opset=9, check_dtypes=True)


def test_scatter_dynamic_shape():
    """Dynamic-shape Scatter: when indices is structurally provably equal to data
    (shared symbolic dims) it still lowers via scatter_elements and is correct;
    a dynamic indices that is *not* provably equal (e.g. a broadcast size-1 dim)
    must raise instead of silently emitting wrong values."""
    n = T.dynamic("N", "int64")
    rng = np.random.RandomState(1)
    batch = 5

    # data/indices/updates all share the symbolic batch dim N -> provably equal,
    # lowered via scatter_elements, correct per ONNX semantics.
    shape_dict = {"data": [n, 3, 4], "indices": [n, 3, 4], "updates": [n, 3, 4]}
    graph = helper.make_graph(
        [helper.make_node("Scatter", ["data", "indices", "updates"], ["output"], axis=0)],
        "scatter_dynamic_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, ["N", 3, 4]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, ["N", 3, 4]),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, ["N", 3, 4]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, 4])],
    )
    model = helper.make_model(
        graph, producer_name="scatter_dynamic_test", opset_imports=[helper.make_opsetid("", 9)]
    )
    model.ir_version = 8
    data = rng.randn(batch, 3, 4).astype("float32")
    indices = rng.randint(0, batch, size=(batch, 3, 4)).astype("int64")
    updates = rng.randn(batch, 3, 4).astype("float32")
    ort_output = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"data": data, "indices": indices, "updates": updates})[0]
    tvm_model = from_onnx(model, shape_dict=shape_dict, opset=9)
    tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        ex = tvm.compile(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm.set_input("main", data, indices, updates)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    np.testing.assert_allclose(ort_output, tvm_output.numpy(), rtol=1e-6, atol=1e-6)

    # A dynamic indices that is not provably equal to data (size-1 broadcast dim)
    # must raise rather than silently produce wrong values.
    shape_dict_bad = {"data": [n, 3, 4], "indices": [n, 1, 4], "updates": [n, 1, 4]}
    graph_bad = helper.make_graph(
        [helper.make_node("Scatter", ["data", "indices", "updates"], ["output"], axis=0)],
        "scatter_dynamic_bad_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, ["N", 3, 4]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, ["N", 1, 4]),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, ["N", 1, 4]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, 4])],
    )
    model_bad = helper.make_model(
        graph_bad,
        producer_name="scatter_dynamic_bad_test",
        opset_imports=[helper.make_opsetid("", 9)],
    )
    with pytest.raises(ValueError, match="dynamic"):
        from_onnx(model_bad, shape_dict=shape_dict_bad, opset=9)


@pytest.mark.parametrize(
    "reduction, opset, data, indices, updates",
    [
        (
            None,
            11,
            np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"),
            np.array([[2, 0, 1], [1, 2, 0]], dtype="int64"),
            np.array([[30, 10, 20], [50, 60, 40]], dtype="float32"),
        ),
        (
            "none",
            18,
            np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"),
            np.array([[2, 0, 1], [1, 2, 0]], dtype="int64"),
            np.array([[30, 10, 20], [50, 60, 40]], dtype="float32"),
        ),
        (
            "add",
            16,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "mul",
            16,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "min",
            18,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "max",
            18,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
    ],
)
def test_scatter_elements_reduction(reduction, opset, data, indices, updates):
    attrs = {"axis": 1}
    if reduction is not None:
        attrs["reduction"] = reduction
    scatter_elements_node = helper.make_node(
        "ScatterElements", ["data", "indices", "updates"], ["output"], **attrs
    )

    graph = helper.make_graph(
        [scatter_elements_node],
        "scatter_elements_reduction_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, list(updates.shape)),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(data.shape))],
    )
    model = helper.make_model(graph, producer_name="scatter_elements_reduction_test")

    check_correctness(
        model,
        inputs={"data": data, "indices": indices, "updates": updates},
        opset=opset,
    )


def test_scatter_elements_invalid_reduction():
    data_shape = [2, 3]
    scatter_elements_node = helper.make_node(
        "ScatterElements",
        ["data", "indices", "updates"],
        ["output"],
        axis=1,
        reduction="unsupported",
    )

    graph = helper.make_graph(
        [scatter_elements_node],
        "scatter_elements_invalid_reduction_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, data_shape),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, data_shape),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
    )
    model = helper.make_model(graph, producer_name="scatter_elements_invalid_reduction_test")

    with pytest.raises(ValueError, match="Only .* reductions are supported, but got unsupported"):
        from_onnx(model, opset=18, keep_params_in_input=True)


@pytest.mark.parametrize("reduction", ["none", "add", "mul"])
def test_scatter_nd(reduction):
    def verify_scatter_nd(data_shape, indices_shape, updates_shape):
        scatter_nd_node = helper.make_node(
            "ScatterND",
            ["data", "indices", "updates"],
            ["output"],
            reduction=reduction,
        )

        graph = helper.make_graph(
            [scatter_nd_node],
            "scatter_nd_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
            ],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
        )

        model = helper.make_model(graph, producer_name="scatter_nd_test")

        indices = np.random.choice(data_shape[0], indices_shape)
        check_correctness(model, inputs={"indices": indices}, opset=16)

    verify_scatter_nd([8], [4, 1], [4])
    verify_scatter_nd([4, 4, 4], [2, 1], [2, 4, 4])
    verify_scatter_nd([4, 5, 6], [2, 3, 2], [2, 3, 6])
    verify_scatter_nd([10], [5, 1], [5])


@pytest.mark.parametrize("reduction", ["none", "add", "mul"])
def test_scatter_nd_negative_indices(reduction):
    def verify_scatter_nd_negative_indices(data_shape, indices_shape, updates_shape):
        scatter_nd_node = helper.make_node(
            "ScatterND",
            ["data", "indices", "updates"],
            ["output"],
            reduction=reduction,
        )

        graph = helper.make_graph(
            [scatter_nd_node],
            "scatter_nd_negative_indices_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
            ],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
        )

        model = helper.make_model(graph, producer_name="scatter_nd_negative_indices_test")

        indices = np.random.randint(-data_shape[0], data_shape[0], indices_shape)
        check_correctness(model, inputs={"indices": indices}, opset=16)

    verify_scatter_nd_negative_indices([8], [4, 1], [4])
    verify_scatter_nd_negative_indices([4, 4, 4], [2, 1], [2, 4, 4])
    verify_scatter_nd_negative_indices([4, 5, 6], [2, 3, 2], [2, 3, 6])
    verify_scatter_nd_negative_indices([10], [5, 1], [5])


def test_compress():
    def verify_compress(
        tensor_shape: list[int],
        condition_shape: list[int] | None,
        axis: int | None,
        expected,
    ):
        if condition_shape is None:
            condition_shape = [tensor_shape[axis]]
        compress_node = helper.make_node("Compress", ["tensor", "condition"], ["output"], axis=axis)
        graph = helper.make_graph(
            [compress_node],
            "compress_test",
            inputs=[
                helper.make_tensor_value_info("tensor", TensorProto.FLOAT, tensor_shape),
                helper.make_tensor_value_info("condition", TensorProto.BOOL, condition_shape),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [])
            ],  # shape is unknown
        )
        model = helper.make_model(graph, producer_name="compress_test")
        tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    def make_expected(tensor_shape: list[int], condition_shape: list[int] | None, axis: int | None):
        if condition_shape is None:
            condition_shape = [tensor_shape[axis]]
        tensor_shape = tuple(tensor_shape)
        condition_shape = tuple(condition_shape)

        if axis is None:
            flat_shape = (int(np.prod(tensor_shape)),)

            num_nonzero = T.dynamic("num_nonzero", "int64")

            @I.ir_module
            class ExpectedCompressFlat:
                @R.function
                def main(
                    tensor: R.Tensor(tensor_shape, dtype="float32"),
                    condition: R.Tensor(condition_shape, dtype="bool"),
                ):
                    R.func_attr({"num_input": 2})
                    with R.dataflow():
                        lv: R.Tensor((1, num_nonzero), dtype="int64") = R.match_cast(
                            R.nonzero(condition), R.Tensor((1, num_nonzero), dtype="int64")
                        )
                        lv1 = R.reshape(tensor, R.shape(flat_shape))
                        lv2: R.Tensor((num_nonzero,), dtype="int64") = R.reshape(
                            lv, R.shape([num_nonzero])
                        )
                        gv = R.take(lv1, lv2, axis=0, mode="fast")
                        R.output(gv)
                    return gv

            return ExpectedCompressFlat

        num_nonzero = T.dynamic("num_nonzero", "int64")

        @I.ir_module
        class ExpectedCompressAxis:
            @R.function
            def main(
                tensor: R.Tensor(tensor_shape, dtype="float32"),
                condition: R.Tensor(condition_shape, dtype="bool"),
            ):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Tensor((1, num_nonzero), dtype="int64") = R.match_cast(
                        R.nonzero(condition), R.Tensor((1, num_nonzero), dtype="int64")
                    )
                    lv1: R.Tensor((num_nonzero,), dtype="int64") = R.reshape(
                        lv, R.shape([num_nonzero])
                    )
                    gv = R.take(tensor, lv1, axis=axis, mode="fast")
                    R.output(gv)
                return gv

        return ExpectedCompressAxis

    for tensor_shape, condition_shape, axis in [
        ([32, 32], [8], None),
        ([32, 32], [16], None),
        ([32, 32], [8], 0),
        ([32, 32], [16], 0),
        ([32, 32], None, 0),
        ([32, 32], [8], 1),
        ([32, 32], [16], 1),
        ([32, 32], None, 1),
    ]:
        verify_compress(
            tensor_shape, condition_shape, axis, make_expected(tensor_shape, condition_shape, axis)
        )


def test_size():
    test_node = helper.make_node("Size", ["x"], ["y"])
    input_shape = [3, 3, 3]
    graph = helper.make_graph(
        [test_node],
        "size_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT64, [3])],
    )

    model = helper.make_model(graph, producer_name="size_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 3, 3), dtype="float32")) -> R.Tensor((), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((), dtype="int64") = R.size(x)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_eye_like(k: int):
    node = helper.make_node("EyeLike", ["x"], ["y"], k=k)
    graph = helper.make_graph(
        [node],
        "eye_like_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="eye_like_test")
    check_correctness(model)


def test_gemm():
    def verify_gemm(alpha, beta, useC, expected):
        if useC:
            gemm_node = helper.make_node(
                "Gemm", ["a", "b", "c"], ["y"], alpha=alpha, beta=beta, transA=1, transB=1
            )
        else:
            gemm_node = helper.make_node(
                "Gemm", ["a", "b"], ["y"], alpha=alpha, beta=beta, transA=1, transB=1
            )

        inputs = [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [4, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [5, 4]),
        ]
        if useC:
            inputs.append(helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 5]))

        graph = helper.make_graph(
            [gemm_node],
            "gemm_test",
            inputs=inputs,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 5])],
        )

        model = helper.make_model(graph, producer_name="gemm_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    def make_expected(alpha, beta, useC):
        alpha = 1.0 if alpha is None else alpha
        beta = 1.0 if beta is None else beta
        alpha = float(np.float32(alpha))
        beta = float(np.float32(beta))

        if not useC and alpha != 1.0:

            @I.ir_module
            class ExpectedScaledA:
                @R.function
                def main(
                    a: R.Tensor((4, 3), dtype="float32"),
                    b: R.Tensor((5, 4), dtype="float32"),
                ) -> R.Tensor((3, 5), dtype="float32"):
                    R.func_attr({"num_input": 2})
                    with R.dataflow():
                        lv: R.Tensor((4, 3), dtype="float32") = R.multiply(
                            a, R.const(alpha, "float32")
                        )
                        lv1: R.Tensor((3, 4), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                        lv2: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=[1, 0])
                        gv: R.Tensor((3, 5), dtype="float32") = R.matmul(lv1, lv2)
                        R.output(gv)
                    return gv

            return ExpectedScaledA

        if not useC:

            @I.ir_module
            class ExpectedMatmulOnly:
                @R.function
                def main(
                    a: R.Tensor((4, 3), dtype="float32"),
                    b: R.Tensor((5, 4), dtype="float32"),
                ) -> R.Tensor((3, 5), dtype="float32"):
                    R.func_attr({"num_input": 2})
                    with R.dataflow():
                        lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=[1, 0])
                        lv1: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=[1, 0])
                        gv: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, lv1)
                        R.output(gv)
                    return gv

            return ExpectedMatmulOnly

        if alpha != 1.0 and beta != 1.0:

            @I.ir_module
            class ExpectedScaledAAndC:
                @R.function
                def main(
                    a: R.Tensor((4, 3), dtype="float32"),
                    b: R.Tensor((5, 4), dtype="float32"),
                    c: R.Tensor((1, 5), dtype="float32"),
                ) -> R.Tensor((3, 5), dtype="float32"):
                    R.func_attr({"num_input": 3})
                    with R.dataflow():
                        lv: R.Tensor((4, 3), dtype="float32") = R.multiply(
                            a, R.const(alpha, "float32")
                        )
                        lv1: R.Tensor((3, 4), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                        lv2: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=[1, 0])
                        lv3: R.Tensor((3, 5), dtype="float32") = R.matmul(lv1, lv2)
                        lv4: R.Tensor((1, 5), dtype="float32") = R.multiply(
                            c, R.const(beta, "float32")
                        )
                        gv: R.Tensor((3, 5), dtype="float32") = R.add(lv3, lv4)
                        R.output(gv)
                    return gv

            return ExpectedScaledAAndC

        if alpha != 1.0:

            @I.ir_module
            class ExpectedScaledAWithC:
                @R.function
                def main(
                    a: R.Tensor((4, 3), dtype="float32"),
                    b: R.Tensor((5, 4), dtype="float32"),
                    c: R.Tensor((1, 5), dtype="float32"),
                ) -> R.Tensor((3, 5), dtype="float32"):
                    R.func_attr({"num_input": 3})
                    with R.dataflow():
                        lv: R.Tensor((4, 3), dtype="float32") = R.multiply(
                            a, R.const(alpha, "float32")
                        )
                        lv1: R.Tensor((3, 4), dtype="float32") = R.permute_dims(lv, axes=[1, 0])
                        lv2: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=[1, 0])
                        lv3: R.Tensor((3, 5), dtype="float32") = R.matmul(lv1, lv2)
                        gv: R.Tensor((3, 5), dtype="float32") = R.add(lv3, c)
                        R.output(gv)
                    return gv

            return ExpectedScaledAWithC

        if beta != 1.0:

            @I.ir_module
            class ExpectedScaledC:
                @R.function
                def main(
                    a: R.Tensor((4, 3), dtype="float32"),
                    b: R.Tensor((5, 4), dtype="float32"),
                    c: R.Tensor((1, 5), dtype="float32"),
                ) -> R.Tensor((3, 5), dtype="float32"):
                    R.func_attr({"num_input": 3})
                    with R.dataflow():
                        lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=[1, 0])
                        lv1: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=[1, 0])
                        lv2: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, lv1)
                        lv3: R.Tensor((1, 5), dtype="float32") = R.multiply(
                            c, R.const(beta, "float32")
                        )
                        gv: R.Tensor((3, 5), dtype="float32") = R.add(lv2, lv3)
                        R.output(gv)
                    return gv

            return ExpectedScaledC

        @I.ir_module
        class ExpectedMatmulAddC:
            @R.function
            def main(
                a: R.Tensor((4, 3), dtype="float32"),
                b: R.Tensor((5, 4), dtype="float32"),
                c: R.Tensor((1, 5), dtype="float32"),
            ) -> R.Tensor((3, 5), dtype="float32"):
                R.func_attr({"num_input": 3})
                with R.dataflow():
                    lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=[1, 0])
                    lv1: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=[1, 0])
                    lv2: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, lv1)
                    gv: R.Tensor((3, 5), dtype="float32") = R.add(lv2, c)
                    R.output(gv)
                return gv

        return ExpectedMatmulAddC

    for alpha, beta, useC in [
        (None, None, False),
        (0.25, None, False),
        (1.0, None, False),
        (None, 0.35, False),
        (0.25, 0.35, False),
        (1.0, 0.35, False),
        (None, 1.0, False),
        (0.25, 1.0, False),
        (1.0, 1.0, False),
        (None, None, True),
        (None, 0.35, True),
        (None, 1.0, True),
        (1.0, None, True),
        (1.0, 0.35, True),
        (1.0, 1.0, True),
        (0.25, None, True),
        (0.25, 0.35, True),
        (0.25, 1.0, True),
    ]:
        verify_gemm(alpha, beta, useC, make_expected(alpha, beta, useC))


def test_reshape():
    def verify_reshape(in_shape, shape, out_shape, expected):
        reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"])

        graph = helper.make_graph(
            [reshape_node],
            "reshape_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, in_shape),
            ],
            initializer=[helper.make_tensor("shape", TensorProto.INT64, [len(shape)], shape)],
            outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, out_shape)],
        )
        model = helper.make_model(graph, producer_name="reshape_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedStaticShape:
        @R.function
        def main(
            data: R.Tensor((7, 32, 32, 8), dtype="float32"),
            shape: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((224, 256), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((224, 256), dtype="float32") = R.reshape(data, R.shape([224, 256]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedInferDim:
        @R.function
        def main(
            data: R.Tensor((7, 32, 32, 8), dtype="float32"),
            shape: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((7, 8192), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((7, 8192), dtype="float32") = R.reshape(data, R.shape([7, 8192]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedCopyInputDim:
        @R.function
        def main(
            data: R.Tensor((7, 32, 32, 8), dtype="float32"),
            shape: R.Tensor((4,), dtype="int64"),
        ) -> R.Tensor((7, 32, 32, 8), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((7, 32, 32, 8), dtype="float32") = R.reshape(
                    data, R.shape([7, 32, 32, 8])
                )
                R.output(gv)
            return gv

    verify_reshape([7, 32, 32, 8], [224, 256], [224, 256], ExpectedStaticShape)
    verify_reshape([7, 32, 32, 8], [-1, 8192], [7, 8192], ExpectedInferDim)
    verify_reshape([7, 32, 32, 8], [0, 32, 32, 8], [7, 32, 32, 8], ExpectedCopyInputDim)


def test_reshape_constant_zero_copy_dimension():
    data = np.arange(6, dtype="float32").reshape(2, 3)
    reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"])
    graph = helper.make_graph(
        [reshape_node],
        "reshape_constant_zero_copy_test",
        inputs=[],
        initializer=[
            numpy_helper.from_array(data, "data"),
            helper.make_tensor("shape", TensorProto.INT64, [2], [0, 3]),
        ],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(
        graph,
        producer_name="reshape_constant_zero_copy_test",
        opset_imports=[helper.make_opsetid("", 14)],
    )

    output = run_in_tvm(model, opset=14)

    tvm_tirx.testing.assert_allclose(output.numpy(), data)


def test_reshape_allowzero_literal_zero_dimension():
    reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"], allowzero=1)
    graph = helper.make_graph(
        [reshape_node],
        "reshape_allowzero_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 0])],
        initializer=[helper.make_tensor("shape", TensorProto.INT64, [2], [0, 2])],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [0, 2])],
    )
    model = helper.make_model(
        graph,
        producer_name="reshape_allowzero_test",
        opset_imports=[helper.make_opsetid("", 14)],
    )

    output = run_in_tvm(model, {"data": np.zeros((2, 0), dtype="float32")}, opset=14)

    assert tuple(output.shape) == (0, 2)


def test_reshape_allowzero_infers_dimension_without_literal_zero():
    reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"], allowzero=1)
    graph = helper.make_graph(
        [reshape_node],
        "reshape_allowzero_infer_dimension_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4])],
        initializer=[helper.make_tensor("shape", TensorProto.INT64, [2], [-1, 2])],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [6, 2])],
    )
    model = helper.make_model(
        graph,
        producer_name="reshape_allowzero_infer_dimension_test",
        opset_imports=[helper.make_opsetid("", 14)],
    )
    data = np.arange(12, dtype="float32").reshape(3, 4)

    output = run_in_tvm(model, {"data": data}, opset=14)

    tvm_tirx.testing.assert_allclose(output.numpy(), data.reshape(6, 2))


def test_reshape_shape_output():
    def verify_reshape_shape_output(target_shape, output_shape, expected):
        shape_node = helper.make_node("Shape", ["data"], ["shape_out"])
        reshape_node = helper.make_node("Reshape", ["shape_out", "target_shape"], ["reshaped"])

        data_shape = [2, 3, 4]

        graph = helper.make_graph(
            [shape_node, reshape_node],
            "reshape_shape_output",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            ],
            initializer=[
                helper.make_tensor(
                    "target_shape", TensorProto.INT64, [len(target_shape)], target_shape
                )
            ],
            outputs=[helper.make_tensor_value_info("reshaped", TensorProto.INT64, output_shape)],
        )
        model = helper.make_model(graph, producer_name="reshape_shape_output")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        assert len(tvm_model["main"].attrs["params"]) == 1
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedFlattenShape:
        @R.function
        def main(
            data: R.Tensor((2, 3, 4), dtype="float32"),
            target_shape: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape([2, 3, 4]):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([2, 3, 4]) = R.shape([2, 3, 4])
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedRank2Shape:
        @R.function
        def main(
            data: R.Tensor((2, 3, 4), dtype="float32"),
            target_shape: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((1, 3), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int64") = R.shape_to_tensor(R.shape([2, 3, 4]))
                gv: R.Tensor((1, 3), dtype="int64") = R.reshape(lv, R.shape([1, 3]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedRank2ColumnShape:
        @R.function
        def main(
            data: R.Tensor((2, 3, 4), dtype="float32"),
            target_shape: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((3, 1), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3,), dtype="int64") = R.shape_to_tensor(R.shape([2, 3, 4]))
                gv: R.Tensor((3, 1), dtype="int64") = R.reshape(lv, R.shape([3, 1]))
                R.output(gv)
            return gv

    verify_reshape_shape_output([-1], [3], ExpectedFlattenShape)
    verify_reshape_shape_output([1, 3], [1, 3], ExpectedRank2Shape)
    verify_reshape_shape_output([3, 1], [3, 1], ExpectedRank2ColumnShape)


def test_transpose_scalar():
    """Test Transpose with scalar inputs - should return scalar unchanged."""
    scalar_node = helper.make_node("Transpose", ["x"], ["y"])
    graph = helper.make_graph(
        [scalar_node],
        "transpose_scalar_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="transpose_scalar_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedScalar:
        @R.function
        def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = x
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedScalar)

    scalar_constant = helper.make_node(
        "Constant",
        [],
        ["scalar"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [], [5.0]),
    )

    transpose_node = helper.make_node("Transpose", ["scalar"], ["y"])
    graph = helper.make_graph(
        [scalar_constant, transpose_node],
        "transpose_scalar_constant_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="transpose_scalar_constant_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedConstant:
        @R.function
        def main() -> R.Tensor((), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((), dtype="float32") = R.const(5.0, "float32")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedConstant)


def test_transpose_axes_validation():
    """Test Transpose validation - perm axes count must match tensor dimensions"""

    def assert_transpose_ir(input_shape, axes, output_shape, name, expected):
        transpose_node = helper.make_node("Transpose", ["x"], ["y"], perm=axes)
        graph = helper.make_graph(
            [transpose_node],
            name,
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(graph, producer_name=name)
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedTranspose1D:
        @R.function
        def main(
            x: R.Tensor((10,), dtype="float32"),
        ) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((10,), dtype="float32") = R.permute_dims(x, axes=[0])
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedTranspose2D:
        @R.function
        def main(
            x: R.Tensor((3, 4), dtype="float32"),
        ) -> R.Tensor((4, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 3), dtype="float32") = R.permute_dims(x, axes=[1, 0])
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedTranspose3D:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tensor((4, 2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 2, 3), dtype="float32") = R.permute_dims(x, axes=[2, 0, 1])
                R.output(gv)
            return gv

    assert_transpose_ir([10], [0], [10], "transpose_1d_valid_test", ExpectedTranspose1D)
    assert_transpose_ir([3, 4], [1, 0], [4, 3], "transpose_2d_valid_test", ExpectedTranspose2D)
    assert_transpose_ir(
        [2, 3, 4], [2, 0, 1], [4, 2, 3], "transpose_3d_valid_test", ExpectedTranspose3D
    )


def assert_static_unsqueeze_ir(
    model: ModelProto,
    *,
    opset: int,
    axes_as_param: bool,
    expected,
):
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    if axes_as_param:
        tvm_model["main"] = tvm_model["main"].without_attr("params")
    tvm_tirx.ir.assert_structural_equal(tvm_model, expected)


def test_unsqueeze():
    axes = [0, 2, 3]
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])
    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [3], vals=axes)],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 1, 1, 32])],
    )

    model = helper.make_model(
        graph, producer_name="unsqueeze_test", opset_imports=[helper.make_opsetid("", 13)]
    )

    @I.ir_module
    class ExpectedAxesParam:
        @R.function
        def main(
            a: R.Tensor((32, 32), dtype="float32"),
            axes_param: R.Tensor((3,), dtype="int64"),
        ) -> R.Tensor((1, 32, 1, 1, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv0: R.Tensor((1, 32, 32), dtype="float32") = R.expand_dims(a, axis=0)
                lv1: R.Tensor((1, 32, 1, 32), dtype="float32") = R.expand_dims(lv0, axis=2)
                gv: R.Tensor((1, 32, 1, 1, 32), dtype="float32") = R.expand_dims(lv1, axis=3)
                R.output(gv)
            return gv

    assert_static_unsqueeze_ir(
        model,
        opset=13,
        axes_as_param=True,
        expected=ExpectedAxesParam,
    )


def test_unsqueeze_scalar_input():
    axes = [0, 1]
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_scalar_input",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [2], vals=axes)],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1])],
    )

    model = helper.make_model(
        graph,
        producer_name="unsqueeze_scalar_input_test",
        opset_imports=[helper.make_opsetid("", 13)],
    )

    @I.ir_module
    class ExpectedScalar:
        @R.function
        def main(
            a: R.Tensor((), dtype="float32"),
            axes_param: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((1, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv0: R.Tensor((1,), dtype="float32") = R.expand_dims(a, axis=0)
                gv: R.Tensor((1, 1), dtype="float32") = R.expand_dims(lv0, axis=1)
                R.output(gv)
            return gv

    assert_static_unsqueeze_ir(
        model,
        opset=13,
        axes_as_param=True,
        expected=ExpectedScalar,
    )


def test_unsqueeze_dynamic_axes_ir():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_dynamic_axes_ir",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 32, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_dynamic_axes_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    unsqueeze_dim_0 = T.dynamic("unsqueeze_dim_0", "int64")
    unsqueeze_dim_1 = T.dynamic("unsqueeze_dim_1", "int64")
    unsqueeze_dim_2 = T.dynamic("unsqueeze_dim_2", "int64")
    unsqueeze_dim_3 = T.dynamic("unsqueeze_dim_3", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((32, 32), dtype="float32"),
            axes: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=4):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Shape([32, 32]) = R.shape_of(a)
                lv1: R.Tensor((2,), dtype="bool") = R.less(axes, R.const(0, "int64"))
                lv2: R.Tensor((2,), dtype="int64") = R.add(axes, R.const(4, "int64"))
                lv3: R.Tensor((4,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(4), R.prim_value(1), dtype="int64"
                )
                lv4: R.Tensor((2,), dtype="int64") = R.where(lv1, lv2, axes)
                lv5: R.Tensor((4, 1), dtype="int64") = R.expand_dims(lv3, axis=[1])
                lv6: R.Tensor((1, 2), dtype="int64") = R.expand_dims(lv4, axis=[0])
                lv7: R.Tensor((4, 2), dtype="bool") = R.equal(lv5, lv6)
                lv8: R.Tensor((4, 2), dtype="int64") = R.astype(lv7, dtype="int64")
                lv9: R.Tensor((4,), dtype="int64") = R.sum(lv8, axis=[1], keepdims=False)
                lv10: R.Tensor((4,), dtype="int64") = R.subtract(R.const(1, "int64"), lv9)
                lv11: R.Tensor((4,), dtype="int64") = R.cumsum(lv10, axis=0, exclusive=False)
                lv12: R.Tensor((4,), dtype="int64") = R.subtract(lv11, R.const(1, "int64"))
                lv13: R.Tensor((4,), dtype="bool") = R.less(lv12, R.const(0, "int64"))
                lv14: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                lv15: R.Tensor((4,), dtype="int64") = R.where(lv13, R.const(0, "int64"), lv12)
                lv16: R.Tensor((4,), dtype="bool") = R.greater(lv9, R.const(0, "int64"))
                lv17: R.Tensor((4,), dtype="int64") = R.take(lv14, lv15, axis=0, mode="fast")
                lv18: R.Tensor((4,), dtype="int64") = R.match_cast(
                    R.where(lv16, R.const(1, "int64"), lv17), R.Tensor((4,), dtype="int64")
                )
                lv19: R.Shape(ndim=4) = R.tensor_to_shape(lv18)
                lv20: R.Shape(
                    [unsqueeze_dim_0, unsqueeze_dim_1, unsqueeze_dim_2, unsqueeze_dim_3]
                ) = R.match_cast(
                    lv19,
                    R.Shape([unsqueeze_dim_0, unsqueeze_dim_1, unsqueeze_dim_2, unsqueeze_dim_3]),
                )
                gv: R.Tensor(
                    (unsqueeze_dim_0, unsqueeze_dim_1, unsqueeze_dim_2, unsqueeze_dim_3),
                    dtype="float32",
                ) = R.reshape(
                    a,
                    R.shape([unsqueeze_dim_0, unsqueeze_dim_1, unsqueeze_dim_2, unsqueeze_dim_3]),
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_unsqueeze_dynamic_axes_rank_validation():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_dynamic_axes_rank_validation",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1, 2]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 32, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_dynamic_axes_rank_validation_test")
    with pytest.raises(ValueError, match="Expected a 1-D tensor"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_unsqueeze_duplicate_axes_validation():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_duplicate_axes_validation",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [2], vals=[0, 0])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 32, 32])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_duplicate_axes_validation_test")
    with pytest.raises(ValueError, match="axes must be unique"):
        from_onnx(model, opset=13)


def test_unsqueeze_v1():
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Unsqueeze-1
    axes = [0, 2, 3]
    unsqueeze_node = helper.make_node("Unsqueeze", ["a"], ["b"], axes=axes)
    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_v1",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 1, 1, 32])],
    )

    model = helper.make_model(
        graph, producer_name="unsqueeze_v1_test", opset_imports=[helper.make_opsetid("", 6)]
    )

    @I.ir_module
    class ExpectedAxesAttr:
        @R.function
        def main(
            a: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((1, 32, 1, 1, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv0: R.Tensor((1, 32, 32), dtype="float32") = R.expand_dims(a, axis=0)
                lv1: R.Tensor((1, 32, 1, 32), dtype="float32") = R.expand_dims(lv0, axis=2)
                gv: R.Tensor((1, 32, 1, 1, 32), dtype="float32") = R.expand_dims(lv1, axis=3)
                R.output(gv)
            return gv

    assert_static_unsqueeze_ir(
        model,
        opset=10,
        axes_as_param=False,
        expected=ExpectedAxesAttr,
    )


def test_gelu():
    verify_unary("Gelu", [32, 32], domain="com.microsoft")


def test_gelu_approximate():
    """Test Gelu with approximate attribute from ONNX Opset 20."""
    # Test Gelu with approximate="tanh"
    verify_unary("Gelu", [32, 32], attrs={"approximate": "tanh"}, opset=20)
    # Test Gelu with approximate="none" (default, same as standard Gelu)
    verify_unary("Gelu", [32, 32], attrs={"approximate": "none"}, opset=20)


def test_bias_gelu():
    bias_gelu_node = helper.make_node("BiasGelu", ["a", "b"], ["c"], domain="com.microsoft")
    graph = helper.make_graph(
        [bias_gelu_node],
        "bias_gelu_structural_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, producer_name="bias_gelu_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((2, 3), dtype="float32"),
            b: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.add(a, b)
                gv: R.Tensor((2, 3), dtype="float32") = R.nn.gelu(lv)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_fast_gelu():
    """Test FastGelu with and without bias"""
    fast_gelu_node = helper.make_node("FastGelu", ["x"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [fast_gelu_node],
        "fast_gelu_structural_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, producer_name="fast_gelu_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.multiply(R.const(0.5, "float32"), x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.79788458347320557, "float32"), x
                )
                lv2: R.Tensor((2, 3), dtype="float32") = R.multiply(x, x)
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(lv2, x)
                lv4: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.035677406936883926, "float32"), lv3
                )
                lv5: R.Tensor((2, 3), dtype="float32") = R.add(lv1, lv4)
                lv6: R.Tensor((2, 3), dtype="float32") = R.tanh(lv5)
                lv7: R.Tensor((2, 3), dtype="float32") = R.add(R.const(1.0, "float32"), lv6)
                lv8: R.Tensor((2, 3), dtype="float32") = R.multiply(lv, lv7)
                gv: R.Tensor((2, 3), dtype="float32") = lv8
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)

    fast_gelu_with_bias_node = helper.make_node(
        "FastGelu", ["x", "bias"], ["y"], domain="com.microsoft"
    )
    graph_with_bias = helper.make_graph(
        [fast_gelu_with_bias_node],
        "fast_gelu_with_bias_structural_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model_with_bias = helper.make_model(
        graph_with_bias, producer_name="fast_gelu_with_bias_structural_test"
    )
    tvm_model_with_bias = from_onnx(model_with_bias, keep_params_in_input=True)

    @I.ir_module
    class ExpectedWithBias:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            bias: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.add(x, bias)
                lv1: R.Tensor((2, 3), dtype="float32") = R.multiply(R.const(0.5, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.79788458347320557, "float32"), lv
                )
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(lv, lv)
                lv4: R.Tensor((2, 3), dtype="float32") = R.multiply(lv3, lv)
                lv5: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.035677406936883926, "float32"), lv4
                )
                lv6: R.Tensor((2, 3), dtype="float32") = R.add(lv2, lv5)
                lv7: R.Tensor((2, 3), dtype="float32") = R.tanh(lv6)
                lv8: R.Tensor((2, 3), dtype="float32") = R.add(R.const(1.0, "float32"), lv7)
                lv9: R.Tensor((2, 3), dtype="float32") = R.multiply(lv1, lv8)
                gv: R.Tensor((2, 3), dtype="float32") = lv9
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model_with_bias, ExpectedWithBias)


def test_clip():
    def verify_clip(input_names, extra_inputs, expected, tir_func_names=()):
        clip_node = helper.make_node("Clip", input_names, ["output"])
        inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 64])]
        inputs.extend(extra_inputs)
        graph = helper.make_graph(
            [clip_node],
            "clip_test",
            inputs=inputs,
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 64])],
        )

        model = helper.make_model(graph, producer_name="clip_test")
        model.opset_import[0].version = 14
        tvm_model = from_onnx(model, keep_params_in_input=True)
        if tir_func_names:
            expected = tvm_tirx.IRModule(expected.functions)
            for name in tir_func_names:
                expected.update_func(expected.get_global_var(name), tvm_model[name])
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedClipMinMax:
        @Ts.function(private=True)
        def maximum(
            var_input: T.Tensor((32, 64), "float32"),
            var_min: T.Tensor((), "float32"),
            var_output: T.Tensor((32, 64), "float32"),
        ):
            T.evaluate(0)

        @Ts.function(private=True)
        def minimum(
            var_input: T.Tensor((32, 64), "float32"),
            var_max: T.Tensor((), "float32"),
            var_output: T.Tensor((32, 64), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            input: R.Tensor((32, 64), dtype="float32"),
            min: R.Tensor((), dtype="float32"),
            max: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((32, 64), dtype="float32"):
            R.func_attr({"num_input": 3})
            cls = ExpectedClipMinMax
            with R.dataflow():
                lv: R.Tensor((), dtype="bool") = R.isnan(min)
                lv1: R.Tensor((), dtype="float32") = R.where(
                    lv, R.const(float("-inf"), "float32"), min
                )
                lv2 = R.call_tir(
                    cls.maximum,
                    (input, lv1),
                    ty_args=[R.Tensor((32, 64), dtype="float32")],
                )
                lv3: R.Tensor((), dtype="bool") = R.isnan(max)
                lv4: R.Tensor((), dtype="float32") = R.where(
                    lv3, R.const(float("inf"), "float32"), max
                )
                lv5 = R.call_tir(
                    cls.minimum,
                    (lv2, lv4),
                    ty_args=[R.Tensor((32, 64), dtype="float32")],
                )
                gv: R.Tensor((32, 64), dtype="float32") = lv5
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedClipMin:
        @Ts.function(private=True)
        def maximum(
            var_input: T.Tensor((32, 64), "float32"),
            var_min: T.Tensor((), "float32"),
            var_output: T.Tensor((32, 64), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            input: R.Tensor((32, 64), dtype="float32"),
            min: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((32, 64), dtype="float32"):
            R.func_attr({"num_input": 2})
            cls = ExpectedClipMin
            with R.dataflow():
                lv: R.Tensor((), dtype="bool") = R.isnan(min)
                lv1: R.Tensor((), dtype="float32") = R.where(
                    lv, R.const(float("-inf"), "float32"), min
                )
                lv2 = R.call_tir(
                    cls.maximum,
                    (input, lv1),
                    ty_args=[R.Tensor((32, 64), dtype="float32")],
                )
                gv: R.Tensor((32, 64), dtype="float32") = lv2
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedClipMaxOnlyInput:
        @Ts.function(private=True)
        def maximum(
            var_input: T.Tensor((32, 64), "float32"),
            var_min: T.Tensor((), "float32"),
            var_output: T.Tensor((32, 64), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            input: R.Tensor((32, 64), dtype="float32"),
            max: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((32, 64), dtype="float32"):
            R.func_attr({"num_input": 2})
            cls = ExpectedClipMaxOnlyInput
            with R.dataflow():
                lv: R.Tensor((), dtype="bool") = R.isnan(max)
                lv1: R.Tensor((), dtype="float32") = R.where(
                    lv, R.const(float("-inf"), "float32"), max
                )
                lv2 = R.call_tir(
                    cls.maximum,
                    (input, lv1),
                    ty_args=[R.Tensor((32, 64), dtype="float32")],
                )
                gv: R.Tensor((32, 64), dtype="float32") = lv2
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedClipIdentity:
        @R.function
        def main(
            input: R.Tensor((32, 64), dtype="float32"),
        ) -> R.Tensor((32, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((32, 64), dtype="float32") = input
                R.output(gv)
            return gv

    min_info = helper.make_tensor_value_info("min", TensorProto.FLOAT, ())
    max_info = helper.make_tensor_value_info("max", TensorProto.FLOAT, ())
    verify_clip(
        ["input", "min", "max"], [min_info, max_info], ExpectedClipMinMax, ("maximum", "minimum")
    )
    verify_clip(["input", "min"], [min_info], ExpectedClipMin, ("maximum",))
    verify_clip(["input", "max"], [max_info], ExpectedClipMaxOnlyInput, ("maximum",))
    verify_clip(["input"], [], ExpectedClipIdentity)


@pytest.mark.parametrize("min", [-6.0, 0.0])
@pytest.mark.parametrize("max", [6.0])
def test_clip_v6(max, min):
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-6
    clip_node = helper.make_node("Clip", ["input"], ["output"], max=max, min=min)
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 64])]
    graph = helper.make_graph(
        [clip_node],
        "clip_v6_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 64])],
    )
    model = helper.make_model(
        graph, producer_name="clip_v6_test", opset_imports=[helper.make_opsetid("", 6)]
    )
    tvm_model = from_onnx(model, opset=10, keep_params_in_input=True)

    @I.ir_module
    class ExpectedClipV6:
        @Ts.function(private=True)
        def maximum(
            var_input: T.Tensor((32, 64), "float32"), var_output: T.Tensor((32, 64), "float32")
        ):
            T.evaluate(0)

        @Ts.function(private=True)
        def minimum(
            var_input: T.Tensor((32, 64), "float32"), var_output: T.Tensor((32, 64), "float32")
        ):
            T.evaluate(0)

        @R.function
        def main(input: R.Tensor((32, 64), dtype="float32")) -> R.Tensor((32, 64), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = ExpectedClipV6
            with R.dataflow():
                lv = R.call_tir(
                    cls.maximum,
                    (input,),
                    ty_args=[R.Tensor((32, 64), dtype="float32")],
                )
                lv1 = R.call_tir(
                    cls.minimum,
                    (lv,),
                    ty_args=[R.Tensor((32, 64), dtype="float32")],
                )
                gv: R.Tensor((32, 64), dtype="float32") = lv1
                R.output(gv)
            return gv

    expected = tvm_tirx.IRModule(ExpectedClipV6.functions)
    expected.update_func(expected.get_global_var("maximum"), tvm_model["maximum"])
    expected.update_func(expected.get_global_var("minimum"), tvm_model["minimum"])
    tvm_tirx.ir.assert_structural_equal(tvm_model, expected)


@pytest.mark.parametrize(
    "min,max",
    [
        pytest.param(
            np.array(0.0, dtype=np.float32),
            np.array(6.0, dtype=np.float32),
        ),
        pytest.param(
            np.array(0.0, dtype=np.float32),
            np.array(np.nan, dtype=np.float32),
        ),
        pytest.param(
            np.array(np.nan, dtype=np.float32),
            np.array(6.0, dtype=np.float32),
        ),
        pytest.param(
            np.array(np.nan, dtype=np.float32),
            np.array(np.nan, dtype=np.float32),
        ),
    ],
)
@pytest.mark.parametrize(
    "input",
    [
        np.array([0.5, -3.0, 4.5, 11.0, 7.0], dtype=np.float32),
    ],
)
def test_clip_v13(input, min, max):
    # Opset 13: tensor min/max. NaN bound => unbounded on that side (ORT).
    clip_node = helper.make_node("Clip", ["input", "min", "max"], ["output"])
    graph = helper.make_graph(
        [clip_node],
        "clip_v13_nan_max",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [5]),
            helper.make_tensor_value_info("min", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("max", TensorProto.FLOAT, []),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [5])],
    )
    model = helper.make_model(graph, producer_name="clip_v13_nan_max")
    check_correctness(
        model,
        inputs={"input": input, "min": min, "max": max},
        opset=13,
    )


def test_equal():
    equal_node = helper.make_node("Equal", ["a", "b"], ["output"])

    graph = helper.make_graph(
        [equal_node],
        "equal_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="equal_test")
    check_correctness(
        model, {"a": np.zeros([32, 32], dtype="float32"), "b": np.zeros([32, 32], dtype="float32")}
    )
    check_correctness(
        model, {"a": np.ones([32, 32], dtype="float32"), "b": np.zeros([32, 32], dtype="float32")}
    )
    check_correctness(model)


def _make_where_shape_expr_model(t1_shape, cond_vals, y_vals, cond_is_input=False):
    """Build an ONNX model whose Where input X comes from a Shape op.

    X = Shape(t1) yields the int64 shape tensor of t1 (e.g. [2, 3]); cond and Y
    are length-1 or length-rank initializers (or cond is a graph input). Where
    must follow NumPy-style broadcasting over the element lists.
    """
    inits, gin, nodes = [], [], []
    gin.append(helper.make_tensor_value_info("__t", TensorProto.FLOAT, list(t1_shape)))
    nodes.append(helper.make_node("Shape", ["__t"], ["X"]))
    if cond_is_input:
        gin.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, [len(cond_vals)]))
    else:
        inits.append(
            helper.make_tensor(
                "cond",
                TensorProto.BOOL,
                [len(cond_vals)],
                list(np.array(cond_vals, dtype=np.bool_)),
            )
        )
    inits.append(
        helper.make_tensor(
            "Y", TensorProto.INT64, [len(y_vals)], list(np.array(y_vals, dtype=np.int64))
        )
    )
    nodes.append(helper.make_node("Where", ["cond", "X", "Y"], ["Yout"]))
    graph = helper.make_graph(
        nodes,
        "where_shape_expr_test",
        gin,
        [helper.make_tensor_value_info("Yout", TensorProto.INT64, [len(t1_shape)])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    return model


def test_where_shape_expr():
    """Where with a shape tensor (Shape op output) must import and run.

    Regression: a shape expression mixed with a runtime tensor was passed
    straight to relax.op.where, which raised 'Op(relax.where) requires argument
    1 (x1) to be a tensor'. The shape expression must be materialized to an
    int64 tensor first.
    """
    # cond is an initializer (1,) -> broadcasts to (2,); x = Shape((2, 3)) = [2, 3]
    model = _make_where_shape_expr_model((2, 3), [True], [4, 5])
    check_correctness(model, inputs={"__t": np.ones((2, 3), dtype="float32")}, opset=16)

    # cond is a graph input (2,); x = Shape((2, 3)); y initializer (2,)
    model = _make_where_shape_expr_model((2, 3), [True, False], [4, 5], cond_is_input=True)
    check_correctness(
        model,
        inputs={
            "__t": np.ones((2, 3), dtype="float32"),
            "cond": np.array([True, False], dtype=np.bool_),
        },
        opset=16,
    )


@pytest.mark.parametrize(
    "cond_vals, y_vals, expected",
    [
        ([True], [4, 5], [2, 3]),  # cond (1,) broadcasts to (2,)
        ([True, False], [4], [2, 4]),  # y (1,) broadcasts to (2,)
        ([True], [4], [2, 3]),  # both (1,) broadcast to (2,)
        ([True, False], [4, 5], [2, 5]),  # equal lengths (regression)
    ],
)
def test_where_shape_expr_broadcast(cond_vals, y_vals, expected):
    """The all-constant/shape-expression fast path must broadcast size-1 dims.

    Regression: length-1 element lists were rejected with 'Cannot broadcast
    condition to x and y' instead of being broadcast to the common length, so
    e.g. cond=[True] with x=[2, 3] failed to import.
    """
    t1_shape = (2, 3)
    model = _make_where_shape_expr_model(t1_shape, cond_vals, y_vals)
    feeds = {"__t": np.ones(t1_shape, dtype="float32")}
    mod = from_onnx(model, shape_dict={"__t": list(t1_shape)})
    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        ex = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    out = vm["main"](feeds["__t"])
    np.testing.assert_array_equal(
        np.array([int(i) for i in out], dtype=np.int64), np.array(expected, dtype=np.int64)
    )


def test_shape():
    shape_node = helper.make_node("Shape", ["data"], ["output"])

    graph = helper.make_graph(
        [shape_node],
        "shape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4, 5, 6]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [4])],
    )

    model = helper.make_model(graph, producer_name="shape_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((3, 4, 5, 6), dtype="float32")) -> R.Shape([3, 4, 5, 6]):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([3, 4, 5, 6]) = R.shape([3, 4, 5, 6])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_zero_extent_input():
    # Zero must remain concrete alongside named/unknown dimensions, and must
    # not bind another input's unknown dimension to zero.
    annotations = {"empty": [0], "data": [None], "mixed": [2, 0, "N", None]}
    inputs = {
        "empty": np.empty((0,), dtype="float32"),
        "data": np.zeros((5,), dtype="float32"),
        "mixed": np.empty((2, 0, 3, 5), dtype="float32"),
    }
    graph = helper.make_graph(
        [helper.make_node("Shape", [name], [f"{name}_shape"]) for name in annotations],
        "shape_zero_extent_input",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in annotations.items()
        ],
        [
            helper.make_tensor_value_info(f"{name}_shape", TensorProto.INT64, [len(shape)])
            for name, shape in annotations.items()
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)

    mod = from_onnx(model)
    for param in mod["main"].params:
        for index, extent in enumerate(annotations[param.name]):
            if extent == 0:
                assert isinstance(param.ty.shape[index], tirx.IntImm)
                assert int(param.ty.shape[index]) == 0
    check_correctness(model, inputs, opset=18, check_dtypes=True)


def test_shape_scalar_input():
    # A rank-0 input has a known, empty static shape. It used to be imported as
    # a runtime R.shape_of because an empty ShapeExpr is falsy, which made the
    # result opaque to every converter that matches on relax.ShapeExpr.
    shape_node = helper.make_node("Shape", ["data"], ["output"])

    graph = helper.make_graph(
        [shape_node],
        "shape_scalar_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, []),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [0])],
    )

    model = helper.make_model(graph, producer_name="shape_scalar_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((), dtype="float32")) -> R.Shape([]):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([]) = R.shape([])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_unknown_rank_input():
    # An input whose ValueInfoProto carries no shape field has unknown rank, which
    # must stay distinct from a rank-0 tensor. It has no static shape to fold, so
    # Shape has to keep the runtime path rather than reporting R.shape([]).
    shape_node = helper.make_node("Shape", ["data"], ["output"])

    data_vi = helper.make_tensor_value_info("data", TensorProto.FLOAT, None)
    assert not data_vi.type.tensor_type.HasField("shape"), "test needs an absent shape field"

    graph = helper.make_graph(
        [shape_node],
        "shape_unknown_rank_test",
        inputs=[data_vi],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, None)],
    )

    model = helper.make_model(graph, producer_name="shape_unknown_rank_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    # The input keeps an unknown shape rather than collapsing to R.Tensor(()).
    data_ty = tvm_model["main"].params[0].ty
    assert data_ty.shape is None
    assert data_ty.ndim == -1

    # And Shape falls back to computing it at runtime.
    op_names = []

    def collect_ops(expr):
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm_tirx.ir.Op):
            op_names.append(expr.op.name)

    relax.analysis.post_order_visit(tvm_model["main"], collect_ops)
    assert "relax.shape_of" in op_names


def test_slice_of_scalar_shape():
    # Slice consuming Shape of a rank-0 input used to raise "Slice requires a
    # statically known input rank", because Shape handed it an opaque value
    # instead of a ShapeExpr. ONNX Runtime returns an empty int64 tensor here.
    nodes = [
        helper.make_node("Shape", ["data"], ["shape"]),
        helper.make_node("Slice", ["shape", "starts", "ends"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "slice_of_scalar_shape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, []),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [0])],
        initializer=[
            helper.make_tensor("starts", TensorProto.INT64, [1], [0]),
            helper.make_tensor("ends", TensorProto.INT64, [1], [1]),
        ],
    )

    model = helper.make_model(graph, producer_name="slice_of_scalar_shape_test")
    tvm_model = from_onnx(model)

    output_ty = tvm_model["main"].ret_ty
    assert isinstance(output_ty, relax.TensorType)
    assert [int(dim) for dim in output_ty.shape] == [0]
    assert output_ty.dtype == "int64"


@pytest.mark.parametrize(
    "attrs,expected_shape",
    [
        ({"start": 1}, (4, 5, 6)),
        ({"end": -1}, (3, 4, 5)),
        ({"start": -2}, (5, 6)),
        ({"start": 1, "end": 3}, (4, 5)),
        ({"start": -10, "end": 10}, (3, 4, 5, 6)),
        ({"start": 3, "end": 2}, ()),
    ],
)
def test_shape_start_end(attrs, expected_shape):
    expected_shape = list(expected_shape)
    shape_node = helper.make_node("Shape", ["data"], ["output"], **attrs)

    graph = helper.make_graph(
        [shape_node],
        "shape_start_end_test",
        inputs=[
            helper.make_tensor_value_info(
                "data",
                TensorProto.FLOAT,
                [3, 4, 5, 6],
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output",
                TensorProto.INT64,
                [len(expected_shape)],
            )
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="shape_start_end_test",
        opset_imports=[helper.make_opsetid("", 15)],
    )
    tvm_model = from_onnx(
        model,
        opset=15,
        keep_params_in_input=True,
    )

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((3, 4, 5, 6), dtype="float32"),
        ) -> R.Shape(expected_shape):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape(expected_shape) = R.shape(expected_shape)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_start_end_symbolic():
    shape_node = helper.make_node(
        "Shape",
        ["data"],
        ["output"],
        start=1,
        end=3,
    )
    graph = helper.make_graph(
        [shape_node],
        "shape_start_end_symbolic_test",
        inputs=[
            helper.make_tensor_value_info(
                "data",
                TensorProto.FLOAT,
                [3, "B", 5, 6],
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output",
                TensorProto.INT64,
                [2],
            )
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="shape_start_end_symbolic_test",
        opset_imports=[helper.make_opsetid("", 15)],
    )
    tvm_model = from_onnx(
        model,
        opset=15,
        keep_params_in_input=True,
    )

    B = T.dynamic("B", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((3, B, 5, 6), dtype="float32"),
        ) -> R.Shape(ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([B, 5]) = R.shape([B, 5])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_start_end_scalar():
    shape_node = helper.make_node(
        "Shape",
        ["data"],
        ["output"],
        start=1,
    )

    graph = helper.make_graph(
        [shape_node],
        "shape_start_end_scalar_test",
        inputs=[
            helper.make_tensor_value_info(
                "data",
                TensorProto.FLOAT,
                [],
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output",
                TensorProto.INT64,
                [0],
            )
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="shape_start_end_scalar_test",
        opset_imports=[helper.make_opsetid("", 15)],
    )
    tvm_model = from_onnx(
        model,
        opset=15,
        keep_params_in_input=True,
    )

    relax.analysis.check_well_formed(tvm_model)

    # A rank-0 input has a known, empty static shape, so start=1 slices an empty
    # ShapeExpr and folds at import time. This used to fall back to a runtime
    # shape_of / shape_to_tensor / strided_slice / tensor_to_shape chain, because
    # the empty ShapeExpr tested as falsy in Shape._impl_v13.
    op_names = []

    def collect_ops(expr):
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm_tirx.ir.Op):
            op_names.append(expr.op.name)

    relax.analysis.post_order_visit(tvm_model["main"], collect_ops)

    assert op_names == []

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((), dtype="float32")) -> R.Shape([]):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([]) = R.shape([])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_trilu():
    def verify_trilu(upper: bool):
        node = helper.make_node("Trilu", ["x", ""], ["y"], upper=upper)
        graph = helper.make_graph(
            [node],
            "trilu_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 5, 5])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 5, 5])],
        )
        model = helper.make_model(graph, producer_name="trilu_test")
        check_correctness(model)

    verify_trilu(True)
    verify_trilu(False)


@pytest.mark.parametrize("k_value", [-1, 0, 1])
def test_trilu_with_const_k(k_value: int):
    """test_trilu_with_const_k"""

    input_shape = [2, 3, 3]

    graph = helper.make_graph(
        [
            make_constant_node("k", onnx.TensorProto.INT64, [1], [k_value]),
            helper.make_node("Trilu", inputs=["x", "k"], outputs=["y"]),
        ],
        "trilu_graph",
        inputs=[
            helper.make_tensor_value_info("x", onnx.TensorProto.DOUBLE, input_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="trilu_graph")
    check_correctness(model)


def test_trilu_initializer_with_params_in_input():
    k_value = np.array(1, dtype="int64")
    graph = helper.make_graph(
        [helper.make_node("Trilu", inputs=["x", "k"], outputs=["y"])],
        "trilu_initializer_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        initializer=[numpy_helper.from_array(k_value, name="k")],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(
        graph,
        producer_name="trilu_initializer_test",
        opset_imports=[helper.make_opsetid("", 14)],
    )

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    np.testing.assert_array_equal(tvm_model["main"].attrs["params"][0].numpy(), k_value)
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            k: R.Tensor((), dtype="int64"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.triu(x, 1)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("upper", [True, False])
def test_trilu_dynamic_k_ir(upper: bool):
    if upper:
        nodes = [helper.make_node("Trilu", inputs=["x", "k"], outputs=["y"], upper=True)]
        inputs = [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("k", TensorProto.INT64, []),
        ]
        outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])]
    else:
        index = numpy_helper.from_array(np.array(0, dtype="int64"), name="index")
        nodes = [
            helper.make_node("Shape", ["x"], ["x_shape"]),
            helper.make_node("Constant", [], ["index"], value=index),
            helper.make_node("Gather", ["x_shape", "index"], ["k"]),
            helper.make_node("Trilu", ["x", "k"], ["y"], upper=False),
        ]
        inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["m", 3])]
        outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["m", 3])]

    graph = helper.make_graph(nodes, "trilu_dynamic_k_graph", inputs, outputs)
    model = helper.make_model(graph, producer_name="trilu_dynamic_k_graph")
    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    if upper:

        @I.ir_module
        class ExpectedTriu:
            @R.function
            def main(
                x: R.Tensor((2, 3), dtype="float32"),
                k: R.Tensor((), dtype="int64"),
            ) -> R.Tensor((2, 3), dtype="float32"):
                R.func_attr({"num_input": 2})
                with R.dataflow():
                    lv: R.Tensor((3,), dtype="int64") = R.arange(0, 3, 1, dtype="int64")
                    lv1: R.Tensor((1, 3), dtype="int64") = R.reshape(lv, R.shape([1, 3]))
                    lv2: R.Tensor((2,), dtype="int64") = R.arange(0, 2, 1, dtype="int64")
                    lv3: R.Tensor((2, 1), dtype="int64") = R.reshape(lv2, R.shape([2, 1]))
                    lv4: R.Tensor((2, 3), dtype="int64") = R.subtract(lv1, lv3)
                    lv5: R.Tensor((), dtype="int64") = R.astype(k, dtype="int64")
                    lv6: R.Tensor((2, 3), dtype="bool") = R.greater_equal(lv4, lv5)
                    lv7: R.Tensor((2, 3), dtype="bool") = R.broadcast_to(lv6, R.shape([2, 3]))
                    gv: R.Tensor((2, 3), dtype="float32") = R.where(lv7, x, R.const(0.0, "float32"))
                    R.output(gv)
                return gv

        expected = ExpectedTriu
    else:
        m = T.dynamic("m", "int64")

        @I.ir_module
        class ExpectedTril:
            @R.function
            def main(x: R.Tensor((m, 3), dtype="float32")) -> R.Tensor((m, 3), dtype="float32"):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((1,), dtype="int64") = R.shape_to_tensor(R.shape([m]))
                    lv1: R.Tensor((3,), dtype="int64") = R.arange(0, 3, 1, dtype="int64")
                    lv2: R.Tensor((1, 3), dtype="int64") = R.reshape(lv1, R.shape([1, 3]))
                    lv3: R.Tensor((m,), dtype="int64") = R.arange(0, m, 1, dtype="int64")
                    lv4: R.Tensor((m, 1), dtype="int64") = R.reshape(lv3, R.shape([m, 1]))
                    lv5: R.Tensor((m, 3), dtype="int64") = R.subtract(lv2, lv4)
                    lv6: R.Tensor((), dtype="int64") = R.squeeze(lv, axis=[0])
                    lv7: R.Tensor((m, 3), dtype="bool") = R.less_equal(lv5, lv6)
                    lv8: R.Tensor((m, 3), dtype="bool") = R.broadcast_to(lv7, R.shape([m, 3]))
                    gv: R.Tensor((m, 3), dtype="float32") = R.where(lv8, x, R.const(0.0, "float32"))
                    R.output(gv)
                return gv

        expected = ExpectedTril

    tvm_tirx.ir.assert_structural_equal(tvm_model, expected)


def test_selu():
    model = make_unary_model("Selu", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(-1.6732631921768188, "float32"), lv2
                )
                lv4: R.Tensor((2, 3), dtype="float32") = R.nn.relu(x)
                lv5: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv4)
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(1.0507010221481323, "float32"), lv5
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)

    model = make_unary_model("Selu", [2, 3], attrs={"alpha": 0.25, "gamma": 0.3})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.nn.relu(lv1)
                lv3: R.Tensor((2, 3), dtype="float32") = R.multiply(R.const(-0.25, "float32"), lv2)
                lv4: R.Tensor((2, 3), dtype="float32") = R.nn.relu(x)
                lv5: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv4)
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.30000001192092896, "float32"), lv5
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedCustom)


def test_mish():
    model = make_unary_model("Mish", [2, 3])
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.exp(x)
                lv1: R.Tensor((2, 3), dtype="float32") = R.add(R.const(1.0, "float32"), lv)
                lv2: R.Tensor((2, 3), dtype="float32") = R.log(lv1)
                lv3: R.Tensor((2, 3), dtype="float32") = R.tanh(lv2)
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(x, lv3)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_prelu():
    def _assert_prelu_ir(slope_shape, expected, input_shape=(3, 32, 32)):
        prelu_node = helper.make_node("PRelu", ["a", "b"], ["c"])
        graph = helper.make_graph(
            [prelu_node],
            "prelu_structural_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, slope_shape),
            ],
            outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, input_shape)],
        )
        model = helper.make_model(graph, producer_name="prelu_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedRankZeroSlope:
        @R.function
        def main(
            a: R.Tensor((3, 32, 32), dtype="float32"),
            b: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1,), dtype="float32") = R.reshape(b, R.shape([1]))
                gv: R.Tensor((3, 32, 32), dtype="float32") = R.nn.prelu(a, lv, axis=2)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedScalarSlope:
        @R.function
        def main(
            a: R.Tensor((3, 32, 32), dtype="float32"),
            b: R.Tensor((1,), dtype="float32"),
        ) -> R.Tensor((3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1,), dtype="float32") = R.reshape(b, R.shape([1]))
                gv: R.Tensor((3, 32, 32), dtype="float32") = R.nn.prelu(a, lv, axis=2)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedTwoDimScalarSlope:
        @R.function
        def main(
            a: R.Tensor((3, 32, 32), dtype="float32"),
            b: R.Tensor((1, 1), dtype="float32"),
        ) -> R.Tensor((3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1,), dtype="float32") = R.reshape(b, R.shape([1]))
                gv: R.Tensor((3, 32, 32), dtype="float32") = R.nn.prelu(a, lv, axis=2)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedChannelSlope:
        @R.function
        def main(
            a: R.Tensor((3, 32, 32), dtype="float32"),
            b: R.Tensor((32,), dtype="float32"),
        ) -> R.Tensor((3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((32,), dtype="float32") = R.reshape(b, R.shape([32]))
                gv: R.Tensor((3, 32, 32), dtype="float32") = R.nn.prelu(a, lv, axis=2)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedBatchSlope:
        @R.function
        def main(
            a: R.Tensor((3, 32, 32), dtype="float32"),
            b: R.Tensor((3, 1, 1), dtype="float32"),
        ) -> R.Tensor((3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((3,), dtype="float32") = R.reshape(b, R.shape([3]))
                gv: R.Tensor((3, 32, 32), dtype="float32") = R.nn.prelu(a, lv, axis=0)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedLowerRankChannelSlope:
        @R.function
        def main(
            a: R.Tensor((1, 32, 16, 16), dtype="float32"),
            b: R.Tensor((32, 1, 1), dtype="float32"),
        ) -> R.Tensor((1, 32, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((32,), dtype="float32") = R.reshape(b, R.shape([32]))
                gv: R.Tensor((1, 32, 16, 16), dtype="float32") = R.nn.prelu(a, lv, axis=1)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMultiAxisSlope:
        @R.function
        def main(
            a: R.Tensor((2, 3, 4, 5), dtype="float32"),
            b: R.Tensor((4, 5), dtype="float32"),
        ) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((2, 3, 4, 5), dtype="bool") = R.less(a, R.const(0.0, "float32"))
                lv1: R.Tensor((2, 3, 4, 5), dtype="float32") = R.multiply(a, b)
                gv: R.Tensor((2, 3, 4, 5), dtype="float32") = R.where(lv, lv1, a)
                R.output(gv)
            return gv

    _assert_prelu_ir([], ExpectedRankZeroSlope)
    _assert_prelu_ir([1], ExpectedScalarSlope)
    _assert_prelu_ir([1, 1], ExpectedTwoDimScalarSlope)
    _assert_prelu_ir([32], ExpectedChannelSlope)
    _assert_prelu_ir([3, 1, 1], ExpectedBatchSlope)
    _assert_prelu_ir([32, 1, 1], ExpectedLowerRankChannelSlope, input_shape=(1, 32, 16, 16))
    _assert_prelu_ir([4, 5], ExpectedMultiAxisSlope, input_shape=(2, 3, 4, 5))


def test_prelu_lower_rank_slope():
    input_shape = (1, 4, 3, 3)
    slope_shape = (4, 1, 1)
    graph = helper.make_graph(
        [helper.make_node("PRelu", ["x", "slope"], ["y"])],
        "prelu_lower_rank_slope_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("slope", TensorProto.FLOAT, slope_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape)],
    )
    model = helper.make_model(
        graph,
        producer_name="prelu_lower_rank_slope_test",
        opset_imports=[helper.make_opsetid("", 16)],
    )
    inputs = {
        "x": np.linspace(-2.0, 2.0, np.prod(input_shape), dtype="float32").reshape(input_shape),
        "slope": np.array([0.1, 0.2, 0.3, 0.4], dtype="float32").reshape(slope_shape),
    }
    check_correctness(model, inputs=inputs, opset=16, check_dtypes=True)


def test_prelu_multi_axis_slope():
    """A slope broadcastable across multiple axes (incl. a slope shaped like x) is
    lowered elementwise to PRelu(x, s) = where(x < 0, s * x, x) since nn.prelu can
    only express a single per-axis slope."""
    input_shape = (2, 3, 4, 5)
    for slope_shape in [(4, 5), (1, 3, 4, 5), (2, 3, 4, 5)]:
        graph = helper.make_graph(
            [helper.make_node("PRelu", ["x", "slope"], ["y"])],
            "prelu_multi_axis_slope_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("slope", TensorProto.FLOAT, list(slope_shape)),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape)],
        )
        model = helper.make_model(
            graph,
            producer_name="prelu_multi_axis_slope_test",
            opset_imports=[helper.make_opsetid("", 16)],
        )
        inputs = {
            "x": np.linspace(-2.0, 2.0, np.prod(input_shape), dtype="float32").reshape(input_shape),
            # negative slopes exercise the s * x path on both sides of the sign.
            "slope": np.linspace(-0.5, 0.8, np.prod(slope_shape), dtype="float32").reshape(
                slope_shape
            ),
        }
        check_correctness(model, inputs=inputs, opset=16, check_dtypes=True)


def test_softplus_large_values():
    """ONNX Softplus is y = log(exp(x) + 1) for every finite input. The frontend
    must not clamp the output to the linear function x beyond a hardcoded threshold
    (it used to pass threshold=20 to relax.op.nn.softplus). Large float32 inputs,
    including values beyond that threshold, must still match onnxruntime."""
    for shape in [[3, 32, 32], [5]]:
        graph = helper.make_graph(
            [helper.make_node("Softplus", ["x"], ["y"])],
            "softplus_large_values",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
        )
        model = helper.make_model(graph, producer_name="softplus_large_values")
        inputs = {
            "x": np.linspace(-10.0, 80.0, np.prod(shape), dtype="float32").reshape(shape),
        }
        check_correctness(model, inputs=inputs, opset=14, rtol=1e-6, atol=1e-6)


def test_softplus_float64_accuracy():
    """float64 Softplus must equal log(exp(x) + 1) across the threshold region.
    Regression: the frontend used to clamp to the linear function x for x > 20,
    losing ~1.9e-9 at x = 20.1. onnxruntime's CPU EP has no float64 Softplus, so
    the expected value is computed with the numerically stable form
    max(x, 0) + log1p(exp(-|x|)), which equals log(exp(x) + 1) mathematically."""
    xs = np.array([19.0, 20.0, 20.1, 21.0, 25.0, 30.0, 40.0], dtype=np.float64)
    expected = np.maximum(xs, 0.0) + np.log1p(np.exp(-np.abs(xs)))

    graph = helper.make_graph(
        [helper.make_node("Softplus", ["x"], ["y"])],
        "softplus_float64",
        inputs=[helper.make_tensor_value_info("x", TensorProto.DOUBLE, [len(xs)])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.DOUBLE, [len(xs)])],
    )
    model = helper.make_model(graph, producer_name="softplus_float64")
    model.opset_import[0].version = 14

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
    tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    tvm_model, params = relax.frontend.detach_params(tvm_model)

    with tvm_tirx.ir.transform.PassContext(opt_level=3):
        ex = tvm.compile(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    vm.set_input("main", xs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main").numpy()

    np.testing.assert_allclose(tvm_output, expected, rtol=1e-12, atol=1e-12)


def test_thresholded_relu():
    model = make_unary_model("ThresholdedRelu", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="bool") = R.greater(x, R.const(1.0, "float32"))
                lv1: R.Tensor((2, 3), dtype="float32") = R.astype(lv, dtype="float32")
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(lv1, x)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)

    model = make_unary_model("ThresholdedRelu", [2, 3], attrs={"alpha": -0.01})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="bool") = R.greater(
                    x, R.const(-0.0099999997764825821, "float32")
                )
                lv1: R.Tensor((2, 3), dtype="float32") = R.astype(lv, dtype="float32")
                gv: R.Tensor((2, 3), dtype="float32") = R.multiply(lv1, x)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedCustom)


def test_leakyrelu():
    verify_unary("LeakyRelu", [32, 32])
    verify_unary("LeakyRelu", [32, 32], attrs={"alpha": 0.2})


def test_hardsigmoid():
    model = make_unary_model("HardSigmoid", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.20000000298023224, "float32"), x
                )
                lv1: R.Tensor((2, 3), dtype="float32") = R.add(lv, R.const(0.5, "float32"))
                gv: R.Tensor((2, 3), dtype="float32") = R.clip(
                    lv1, R.prim_value(0), R.prim_value(1)
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)

    model = make_unary_model("HardSigmoid", [2, 3], attrs={"alpha": 0.3, "beta": 0.4})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="float32") = R.multiply(
                    R.const(0.30000001192092896, "float32"), x
                )
                lv1: R.Tensor((2, 3), dtype="float32") = R.add(
                    lv, R.const(0.40000000596046448, "float32")
                )
                gv: R.Tensor((2, 3), dtype="float32") = R.clip(
                    lv1, R.prim_value(0), R.prim_value(1)
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedCustom)

    model = make_unary_model("HardSigmoid", [1, 3, 20, 20], attrs={"alpha": 0.5, "beta": 0.6})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom4D:
        @R.function
        def main(
            x: R.Tensor((1, 3, 20, 20), dtype="float32"),
        ) -> R.Tensor((1, 3, 20, 20), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 20, 20), dtype="float32") = R.multiply(
                    R.const(0.5, "float32"), x
                )
                lv1: R.Tensor((1, 3, 20, 20), dtype="float32") = R.add(
                    lv, R.const(0.60000002384185791, "float32")
                )
                gv: R.Tensor((1, 3, 20, 20), dtype="float32") = R.clip(
                    lv1, R.prim_value(0), R.prim_value(1)
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedCustom4D)


def test_shrink():
    model = make_unary_model("Shrink", [2, 3])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="bool") = R.greater(x, R.const(0.5, "float32"))
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(x, R.const(0.0, "float32"))
                lv2: R.Tensor((2, 3), dtype="float32") = R.zeros_like(x)
                lv3: R.Tensor((2, 3), dtype="float32") = R.where(lv, lv1, lv2)
                lv4: R.Tensor((), dtype="float32") = R.negative(R.const(0.5, "float32"))
                lv5: R.Tensor((2, 3), dtype="bool") = R.less(x, lv4)
                lv6: R.Tensor((2, 3), dtype="float32") = R.add(x, R.const(0.0, "float32"))
                lv7: R.Tensor((2, 3), dtype="float32") = R.where(lv5, lv6, lv2)
                gv: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv7)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)

    model = make_unary_model("Shrink", [2, 3], attrs={"lambd": 0.2, "bias": 0.1})
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class ExpectedCustom:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3), dtype="bool") = R.greater(
                    x, R.const(0.20000000298023224, "float32")
                )
                lv1: R.Tensor((2, 3), dtype="float32") = R.subtract(
                    x, R.const(0.10000000149011612, "float32")
                )
                lv2: R.Tensor((2, 3), dtype="float32") = R.zeros_like(x)
                lv3: R.Tensor((2, 3), dtype="float32") = R.where(lv, lv1, lv2)
                lv4: R.Tensor((), dtype="float32") = R.negative(
                    R.const(0.20000000298023224, "float32")
                )
                lv5: R.Tensor((2, 3), dtype="bool") = R.less(x, lv4)
                lv6: R.Tensor((2, 3), dtype="float32") = R.add(
                    x, R.const(0.10000000149011612, "float32")
                )
                lv7: R.Tensor((2, 3), dtype="float32") = R.where(lv5, lv6, lv2)
                gv: R.Tensor((2, 3), dtype="float32") = R.add(lv3, lv7)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedCustom)


def _make_conv_model(input_shape, weight_shape, stride, dilation, pad, bias, auto_pad):
    nd = len(weight_shape) - 2
    groups = input_shape[1] // weight_shape[1]
    node_attrs = {
        "strides": [stride] * nd,
        "dilations": [dilation] * nd,
        "group": groups,
    }
    if auto_pad == "VALID":
        output_shape = [input_shape[0], weight_shape[0]] + [
            (input_shape[i] - dilation * (weight_shape[i] - 1) - 1) // stride + 1
            for i in range(2, len(input_shape))
        ]
        node_attrs["auto_pad"] = auto_pad
    elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        output_shape = [input_shape[0], weight_shape[0]] + [
            (input_shape[i] + stride - 1) // stride for i in range(2, len(input_shape))
        ]
        node_attrs["auto_pad"] = auto_pad
    else:
        output_shape = [input_shape[0], weight_shape[0]] + [
            (input_shape[i] + 2 * pad - dilation * (weight_shape[i] - 1) - 1) // stride + 1
            for i in range(2, len(input_shape))
        ]
        node_attrs["pads"] = [pad] * nd * 2

    conv_node = helper.make_node(
        "Conv",
        inputs=["x", "w"] + (["b"] if bias else []),
        outputs=["y"],
        **node_attrs,
    )
    graph = helper.make_graph(
        [conv_node],
        "conv_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
        ]
        + (
            [helper.make_tensor_value_info("b", TensorProto.FLOAT, [output_shape[1]])]
            if bias
            else []
        ),
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(
        graph,
        producer_name="conv_test",
        opset_imports=[helper.make_opsetid("", 14)],
    )
    return model, output_shape, groups


CONV_IMPORT_CONFIGS = [
    *(("VALID", stride, dilation, 0) for stride in [1, 2] for dilation in [1, 2]),
    *((auto_pad, stride, 1, 0) for auto_pad in ["SAME_UPPER", "SAME_LOWER"] for stride in [1, 2]),
    *(
        ("NOTSET", stride, dilation, pad)
        for stride in [1, 2]
        for dilation in [1, 2]
        for pad in [0, 2]
    ),
]


def _verify_conv_import(auto_pad, stride, dilation, pad, bias, nd, groups):
    input_shape = [1, 4] + [8] * nd
    weight_shape = [4, 4 // groups] + [3] * nd
    model, output_shape, expected_groups = _make_conv_model(
        input_shape, weight_shape, stride, dilation, pad, bias, auto_pad
    )
    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
    func = tvm_model["main"]

    conv_op_name = f"relax.nn.conv{nd}d"
    conv_calls = []

    def visit(expr):
        if (
            isinstance(expr, relax.Call)
            and isinstance(expr.op, tvm_tirx.ir.Op)
            and expr.op.name == conv_op_name
        ):
            conv_calls.append(expr)

    relax.analysis.post_order_visit(func.body, visit)
    assert len(conv_calls) == 1
    conv_call = conv_calls[0]
    assert tuple(int(value) for value in func.ret_ty.shape.values) == tuple(output_shape)
    assert tuple(int(value) for value in conv_call.attrs.strides) == (stride,) * nd
    assert tuple(int(value) for value in conv_call.attrs.dilation) == (dilation,) * nd
    assert int(conv_call.attrs.groups) == expected_groups
    assert ("relax.add" in collect_relax_call_ops(func)) == bias

    expected_padding = (pad,) * (nd * 2) if auto_pad == "NOTSET" else (0,) * (nd * 2)
    assert tuple(int(value) for value in conv_call.attrs.padding) == expected_padding


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("nd", [1, 2, 3])
@pytest.mark.parametrize("groups", [1, 2])
def test_conv_import(bias, nd, groups):
    for auto_pad, stride, dilation, pad in CONV_IMPORT_CONFIGS:
        _verify_conv_import(auto_pad, stride, dilation, pad, bias, nd, groups)


@pytest.mark.parametrize(
    "nd, groups, auto_pad, stride, dilation, pad, bias",
    [
        (1, 1, "VALID", 1, 2, 0, False),
        (1, 2, "NOTSET", 2, 1, 2, True),
        (2, 1, "SAME_UPPER", 2, 1, 0, True),
        (2, 2, "SAME_LOWER", 2, 1, 0, False),
        (3, 2, "VALID", 2, 1, 0, True),
        (3, 1, "NOTSET", 1, 2, 2, False),
    ],
)
def test_conv_numerical(nd, groups, auto_pad, stride, dilation, pad, bias):
    input_shape = [1, 4] + [8] * nd
    weight_shape = [4, 4 // groups] + [3] * nd
    model, _, _ = _make_conv_model(input_shape, weight_shape, stride, dilation, pad, bias, auto_pad)
    check_correctness(model, opset=14, atol=1e-4)


def _make_conv_transpose_model(
    nd, groups, auto_pad, stride, dilation, pad, bias, output_pad, spatial_extent=8
):
    input_shape = [1, 4] + [spatial_extent] * nd
    weight_shape = [4, 4 // groups] + [3] * nd
    output_channels = weight_shape[1] * groups
    effective_kernel = dilation * (weight_shape[2] - 1) + 1

    node_attrs = {
        "strides": [stride] * nd,
        "dilations": [dilation] * nd,
        "output_padding": [output_pad] * nd,
        "group": groups,
    }
    if auto_pad == "NOTSET":
        node_attrs["pads"] = [pad] * nd * 2
        output_spatial = [
            (spatial_extent - 1) * stride - 2 * pad + effective_kernel + output_pad
        ] * nd
    elif auto_pad == "VALID":
        node_attrs["auto_pad"] = auto_pad
        node_attrs["kernel_shape"] = weight_shape[2:]
        output_spatial = [(spatial_extent - 1) * stride + effective_kernel + output_pad] * nd
    else:
        node_attrs["auto_pad"] = auto_pad
        node_attrs["kernel_shape"] = weight_shape[2:]
        output_spatial = [spatial_extent * stride] * nd

    output_shape = [input_shape[0], output_channels, *output_spatial]
    conv_node = helper.make_node(
        "ConvTranspose",
        inputs=["x", "w"] + (["b"] if bias else []),
        outputs=["y"],
        **node_attrs,
    )
    graph = helper.make_graph(
        [conv_node],
        "conv_transpose_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
        ]
        + (
            [helper.make_tensor_value_info("b", TensorProto.FLOAT, [output_channels])]
            if bias
            else []
        ),
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(
        graph,
        producer_name="conv_transpose_test",
        opset_imports=[helper.make_opsetid("", 14)],
    )
    return model, output_shape


CONV_TRANSPOSE_IMPORT_CONFIGS = [
    ("NOTSET", 1, 1, 0, 0),
    ("NOTSET", 2, 2, 2, 1),
    ("SAME_UPPER", 1, 1, 0, 0),
    ("SAME_UPPER", 2, 1, 0, 0),
    ("SAME_LOWER", 2, 1, 0, 0),
    ("VALID", 1, 2, 0, 0),
    ("VALID", 2, 1, 0, 1),
]


def _verify_conv_transpose_import(nd, groups, auto_pad, stride, dilation, pad, bias, output_pad):
    model, output_shape = _make_conv_transpose_model(
        nd, groups, auto_pad, stride, dilation, pad, bias, output_pad
    )
    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)
    func = tvm_model["main"]

    conv_op_name = f"relax.nn.conv{nd}d_transpose"
    conv_calls = []

    def visit(expr):
        if (
            isinstance(expr, relax.Call)
            and isinstance(expr.op, tvm_tirx.ir.Op)
            and expr.op.name == conv_op_name
        ):
            conv_calls.append(expr)

    relax.analysis.post_order_visit(func.body, visit)
    assert len(conv_calls) == 1
    conv_call = conv_calls[0]
    assert tuple(int(value) for value in func.ret_ty.shape.values) == tuple(output_shape)
    assert tuple(int(value) for value in conv_call.attrs.strides) == (stride,) * nd
    assert tuple(int(value) for value in conv_call.attrs.dilation) == (dilation,) * nd
    assert tuple(int(value) for value in conv_call.attrs.output_padding) == (output_pad,) * nd
    assert int(conv_call.attrs.groups) == groups
    assert ("relax.add" in collect_relax_call_ops(func)) == bias

    if auto_pad == "NOTSET":
        expected_padding = (pad,) * (nd * 2)
    elif auto_pad == "VALID":
        expected_padding = (0,) * (nd * 2)
    else:
        total_pad = max((3 - 1) * dilation + 1 + output_pad - stride, 0)
        pad_begin = total_pad // 2 if auto_pad == "SAME_UPPER" else total_pad - total_pad // 2
        expected_padding = (pad_begin,) * nd + (total_pad - pad_begin,) * nd
    assert tuple(int(value) for value in conv_call.attrs.padding) == expected_padding


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("nd", [1, 2, 3])
@pytest.mark.parametrize("groups", [1, 2])
def test_conv_transpose_import(bias, nd, groups):
    for auto_pad, stride, dilation, pad, output_pad in CONV_TRANSPOSE_IMPORT_CONFIGS:
        _verify_conv_transpose_import(nd, groups, auto_pad, stride, dilation, pad, bias, output_pad)


@pytest.mark.parametrize(
    "nd, groups, auto_pad, stride, dilation, pad, bias, output_pad",
    [
        # Broad attribute coverage, including dilation and output padding.
        (1, 1, "NOTSET", 2, 2, 2, False, 1),
        (2, 2, "VALID", 1, 2, 0, True, 0),
        (3, 1, "NOTSET", 1, 1, 0, True, 0),
        (3, 2, "VALID", 2, 1, 0, False, 1),
        # Each rank uses a distinct Relax op and legalizer.  Exercise both
        # directions of asymmetric SAME padding numerically for every rank.
        (1, 2, "SAME_UPPER", 2, 1, 0, True, 0),
        (1, 1, "SAME_LOWER", 2, 1, 0, False, 0),
        (2, 2, "SAME_UPPER", 2, 1, 0, True, 0),
        (2, 1, "SAME_LOWER", 2, 1, 0, False, 0),
        (3, 1, "SAME_UPPER", 2, 1, 0, False, 0),
        (3, 2, "SAME_LOWER", 2, 1, 0, True, 0),
        # Preserve the 2-D output_padding regression through LLVM execution.
        (2, 1, "NOTSET", 2, 1, 2, True, 1),
    ],
)
def test_conv_transpose_numerical(nd, groups, auto_pad, stride, dilation, pad, bias, output_pad):
    spatial_extent = 4 if nd == 3 else 8
    model, _ = _make_conv_transpose_model(
        nd,
        groups,
        auto_pad,
        stride,
        dilation,
        pad,
        bias,
        output_pad,
        spatial_extent,
    )
    check_correctness(model, opset=14, atol=1e-4)


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("exclusive", [True, False])
def test_cumsum(reverse, exclusive):
    cumsum_node = helper.make_node(
        "CumSum", ["x", "axis"], ["y"], reverse=reverse, exclusive=exclusive
    )
    shape = [32, 32]
    graph = helper.make_graph(
        [cumsum_node],
        "cumsum_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=[helper.make_tensor("axis", TensorProto.INT64, (), [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(
        graph, producer_name="cumsum_test", opset_imports=[helper.make_opsetid("", 14)]
    )
    check_correctness(model)


def test_cumsum_int32_1d_axis_initializer():
    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
        ],
        initializer=[helper.make_tensor("axis", onnx.TensorProto.INT32, [1], [0])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_graph")
    check_correctness(model)


def test_cumsum_dynamic_axis_not_supported():
    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_dynamic_axis_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
            helper.make_tensor_value_info("axis", onnx.TensorProto.INT32, [1], "axis"),
        ],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_dynamic_axis_graph")
    with pytest.raises(ValueError, match="non-constant axis input is not supported"):
        from_onnx(model, opset=14, keep_params_in_input=True)


def test_cumsum_axis_shape_validation():
    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_invalid_axis_shape_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
        ],
        initializer=[helper.make_tensor("axis", onnx.TensorProto.INT64, [2], [0, 1])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_invalid_axis_shape_graph")
    with pytest.raises(
        ValueError,
        match=r"axis input must be a scalar \(0-D\) or a single-element 1-D tensor",
    ):
        from_onnx(model, opset=14, keep_params_in_input=True)


def test_squeeze():
    def verify_squeeze(axis, expected):
        if axis:
            squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
        else:
            squeeze_node = helper.make_node("Squeeze", ["x"], ["y"])
        shape = [1, 32, 1, 32]

        initializer = (
            [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
        )

        graph = helper.make_graph(
            [squeeze_node],
            "squeeze_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            ],
            initializer=initializer,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
        )

        model = helper.make_model(
            graph, producer_name="squeeze_test", opset_imports=[helper.make_opsetid("", 13)]
        )
        tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
        if axis:
            tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedSqueezeAxes:
        @R.function
        def main(
            x: R.Tensor((1, 32, 1, 32), dtype="float32"),
            axes: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = R.squeeze(x, axis=[0, 2])
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSqueezeAll:
        @R.function
        def main(x: R.Tensor((1, 32, 1, 32), dtype="float32")) -> R.Tensor(
            (32, 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = R.squeeze(x, axis=None)
                R.output(gv)
            return gv

    verify_squeeze([0, 2], ExpectedSqueezeAxes)
    verify_squeeze(None, ExpectedSqueezeAll)


def test_squeeze_axes_attribute():
    # Prior to opset 13, ONNX Squeeze takes `axes` as an attribute rather than an input.
    squeeze_node = helper.make_node("Squeeze", ["x"], ["y"], axes=[0, 2])
    shape = [1, 32, 1, 32]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_axes_attribute_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(
        graph,
        producer_name="squeeze_axes_attribute_test",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 32, 1, 32), dtype="float32")) -> R.Tensor(
            (32, 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = R.squeeze(x, axis=[0, 2])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_squeeze_non_unit_axis_raises():
    # Per the ONNX spec, squeezing an axis whose length is not 1 must raise an error
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [2, 3]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_non_unit_axis_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [1], [0])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
    )

    model = helper.make_model(
        graph,
        producer_name="squeeze_non_unit_axis_test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    with pytest.raises(ValueError, match="has size 2"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_squeeze_symbolic_axis_raises():
    # Squeezing an axis whose extent is symbolic can't be proven to be 1 at import time
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = ["N", 3]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_symbolic_axis_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [1], [0])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
    )

    model = helper.make_model(
        graph,
        producer_name="squeeze_symbolic_axis_test",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    with pytest.raises(ValueError, match="symbolic extent"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_squeeze_constant():
    def verify_squeeze_constant(axis, expected):
        shape = [1, 2, 1, 3]
        data = np.arange(6, dtype="float32").reshape(shape)
        constant = make_constant_node("x", onnx.TensorProto.FLOAT, shape, data.flatten().tolist())
        if axis:
            squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
        else:
            squeeze_node = helper.make_node("Squeeze", ["x"], ["y"])

        initializer = (
            [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
        )

        graph = helper.make_graph(
            [constant, squeeze_node],
            "squeeze_test",
            inputs=[],
            initializer=initializer,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
        )

        model = helper.make_model(
            graph, producer_name="squeeze_test", opset_imports=[helper.make_opsetid("", 13)]
        )
        tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
        if axis:
            tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedSqueezeConstantAxes:
        @R.function
        def main(axes: R.Tensor((2,), dtype="int64")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.const(
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], "float32"
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSqueezeConstantAll:
        @R.function
        def main() -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.const(
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], "float32"
                )
                R.output(gv)
            return gv

    verify_squeeze_constant([0, 2], ExpectedSqueezeConstantAxes)
    verify_squeeze_constant(None, ExpectedSqueezeConstantAll)


def test_dynamic_squeeze():
    axis = [0]
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, "A", "B"]

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["A", "B"])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, A, B), dtype="float32"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((A, B), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B), dtype="float32") = R.squeeze(x, axis=[0])
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_squeeze_dynamic_axes_ir():
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, 32, 1, 32]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_dynamic_axes_ir",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_dynamic_axes_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    squeeze_num_keep_dims = T.dynamic("squeeze_num_keep_dims", "int64")
    squeeze_dim_0 = T.dynamic("squeeze_dim_0", "int64")
    squeeze_dim_1 = T.dynamic("squeeze_dim_1", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 32, 1, 32), dtype="float32"),
            axes: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=2):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Shape([1, 32, 1, 32]) = R.shape_of(x)
                lv1: R.Tensor((2,), dtype="bool") = R.less(axes, R.const(0, "int64"))
                lv2: R.Tensor((2,), dtype="int64") = R.add(axes, R.const(4, "int64"))
                lv3: R.Tensor((4,), dtype="int64") = R.arange(
                    R.prim_value(0), R.prim_value(4), R.prim_value(1), dtype="int64"
                )
                lv4: R.Tensor((2,), dtype="int64") = R.where(lv1, lv2, axes)
                lv5: R.Tensor((4, 1), dtype="int64") = R.expand_dims(lv3, axis=[1])
                lv6: R.Tensor((1, 2), dtype="int64") = R.expand_dims(lv4, axis=[0])
                lv7: R.Tensor((4, 2), dtype="bool") = R.equal(lv5, lv6)
                lv8: R.Tensor((4, 2), dtype="int64") = R.astype(lv7, dtype="int64")
                lv9: R.Tensor((4,), dtype="int64") = R.sum(lv8, axis=[1], keepdims=False)
                lv10: R.Tensor((4,), dtype="bool") = R.equal(lv9, R.const(0, "int64"))
                lv11: R.Tensor((1, squeeze_num_keep_dims), dtype="int64") = R.match_cast(
                    R.nonzero(lv10), R.Tensor((1, squeeze_num_keep_dims), dtype="int64")
                )
                lv12: R.Tensor((4,), dtype="int64") = R.shape_to_tensor(lv)
                lv13: R.Tensor((squeeze_num_keep_dims,), dtype="int64") = R.reshape(
                    lv11, R.shape([squeeze_num_keep_dims])
                )
                lv14: R.Tensor((2,), dtype="int64") = R.match_cast(
                    R.take(lv12, lv13, axis=0, mode="fast"), R.Tensor((2,), dtype="int64")
                )
                lv15: R.Shape(ndim=2) = R.tensor_to_shape(lv14)
                lv16: R.Shape([squeeze_dim_0, squeeze_dim_1]) = R.match_cast(
                    lv15, R.Shape([squeeze_dim_0, squeeze_dim_1])
                )
                gv: R.Tensor((squeeze_dim_0, squeeze_dim_1), dtype="float32") = R.reshape(
                    x, R.shape([squeeze_dim_0, squeeze_dim_1])
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_squeeze_dynamic_axes_rank_validation():
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, 32, 1, 32]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_dynamic_axes_rank_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1, 2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_dynamic_axes_rank_validation_test")
    with pytest.raises(ValueError, match="Expected a 1-D tensor"):
        from_onnx(model, opset=13, keep_params_in_input=True)


@pytest.mark.parametrize("axis", [[0]])
def test_dynamic_shape_squeeze(axis):
    shape_node = helper.make_node("Shape", ["x"], ["y"])
    squeeze_node = helper.make_node("Squeeze", ["y", "axes"], ["z"])
    shape = ["A"]

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [shape_node, squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, [])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    # Use an ordinary symbolic Var for the dynamic shape binding.
    a = T.dynamic("A", "int64")
    x = relax.Var("x", relax.TensorType([a], "float32"))
    axes = relax.Var("axes", relax.TensorType([1], "int64"))
    gv = relax.Var("gv", tvm_tirx.ir.PrimType("int64"))
    body = relax.SeqExpr([relax.DataflowBlock([relax.VarBinding(gv, a)])], gv)
    # Match the importer boundary, where BlockBuilder populates the SeqExpr result type.
    body = relax.BlockBuilder().normalize(body)
    expected_func = relax.Function([x, axes], body, tvm_tirx.ir.PrimType("int64")).with_attrs(
        {"num_input": 1, "global_symbol": "main"}
    )
    tvm_tirx.ir.assert_structural_equal(tvm_model, tvm_tirx.IRModule({"main": expected_func}))


def test_const():
    shape = [32, 32]
    const_value = np.random.rand(*shape).astype(np.float32)
    const_node = helper.make_node(
        "Constant",
        [],
        ["y"],
        value=helper.make_tensor("value", TensorProto.FLOAT, shape, const_value.flatten()),
    )
    graph = helper.make_graph(
        [const_node],
        "const_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="const_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((32, 32), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = R.const(const_value, "float32")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_instance_norm():
    def verify_instance_norm(input_shape, scale_shape, bias_shape, expected):
        node = helper.make_node("InstanceNormalization", ["a", "b", "c"], ["d"], epsilon=1e-12)
        graph = helper.make_graph(
            [node],
            "instance_norm_test",
            inputs=[
                helper.make_tensor_value_info("a", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("b", TensorProto.FLOAT, scale_shape),
                helper.make_tensor_value_info("c", TensorProto.FLOAT, bias_shape),
            ],
            outputs=[helper.make_tensor_value_info("d", TensorProto.FLOAT, input_shape)],
        )

        model = helper.make_model(graph, producer_name="instance_norm_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class Expected4D:
        @R.function
        def main(
            a: R.Tensor((1, 3, 32, 32), dtype="float32"),
            b: R.Tensor((3,), dtype="float32"),
            c: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((1, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.mean(a, axis=[2, 3], keepdims=True)
                lv1: R.Tensor((1, 3, 32, 32), dtype="float32") = R.subtract(a, lv)
                lv2: R.Tensor((1, 3, 1, 1), dtype="float32") = R.variance(
                    a, axis=[2, 3], keepdims=True
                )
                lv3: R.Tensor((1, 3, 1, 1), dtype="float32") = R.add(lv2, R.const(1e-12, "float32"))
                lv4: R.Tensor((1, 3, 1, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((1, 3, 32, 32), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((3, 1, 1), dtype="float32") = R.reshape(b, R.shape([3, 1, 1]))
                lv7: R.Tensor((1, 3, 32, 32), dtype="float32") = R.multiply(lv5, lv6)
                lv8: R.Tensor((3, 1, 1), dtype="float32") = R.reshape(c, R.shape([3, 1, 1]))
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.add(lv7, lv8)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3D:
        @R.function
        def main(
            a: R.Tensor((1, 32, 32), dtype="float32"),
            b: R.Tensor((32,), dtype="float32"),
            c: R.Tensor((32,), dtype="float32"),
        ) -> R.Tensor((1, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 32, 1), dtype="float32") = R.mean(a, axis=[2], keepdims=True)
                lv1: R.Tensor((1, 32, 32), dtype="float32") = R.subtract(a, lv)
                lv2: R.Tensor((1, 32, 1), dtype="float32") = R.variance(a, axis=[2], keepdims=True)
                lv3: R.Tensor((1, 32, 1), dtype="float32") = R.add(lv2, R.const(1e-12, "float32"))
                lv4: R.Tensor((1, 32, 1), dtype="float32") = R.sqrt(lv3)
                lv5: R.Tensor((1, 32, 32), dtype="float32") = R.divide(lv1, lv4)
                lv6: R.Tensor((32, 1), dtype="float32") = R.reshape(b, R.shape([32, 1]))
                lv7: R.Tensor((1, 32, 32), dtype="float32") = R.multiply(lv5, lv6)
                lv8: R.Tensor((32, 1), dtype="float32") = R.reshape(c, R.shape([32, 1]))
                gv: R.Tensor((1, 32, 32), dtype="float32") = R.add(lv7, lv8)
                R.output(gv)
            return gv

    verify_instance_norm([1, 3, 32, 32], [3], [3], Expected4D)
    verify_instance_norm([1, 32, 32], [32], [32], Expected3D)


def test_mean_variance_norm():
    def verify_mean_variance_norm(axes, expected):
        node = helper.make_node("MeanVarianceNormalization", ["x"], ["y"], axes=axes)
        graph = helper.make_graph(
            [node],
            "mean_variance_norm_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 32, 32])],
        )

        model = helper.make_model(graph, producer_name="mean_variance_norm_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedDefaultAxes:
        @R.function
        def main(
            x: R.Tensor((1, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.mean(
                    x, axis=[0, 2, 3], keepdims=True
                )
                lv1: R.Tensor((1, 3, 32, 32), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((1, 3, 32, 32), dtype="float32") = R.power(x, R.const(2.0, "float32"))
                lv3: R.Tensor((1, 3, 1, 1), dtype="float32") = R.mean(
                    lv2, axis=[0, 2, 3], keepdims=True
                )
                lv4: R.Tensor((1, 3, 1, 1), dtype="float32") = R.power(lv, R.const(2.0, "float32"))
                lv5: R.Tensor((1, 3, 1, 1), dtype="float32") = R.subtract(lv3, lv4)
                lv6: R.Tensor((1, 3, 1, 1), dtype="float32") = R.sqrt(lv5)
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.divide(lv1, lv6)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedChannelAxes:
        @R.function
        def main(
            x: R.Tensor((1, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 1, 1, 1), dtype="float32") = R.mean(
                    x, axis=[1, 2, 3], keepdims=True
                )
                lv1: R.Tensor((1, 3, 32, 32), dtype="float32") = R.subtract(x, lv)
                lv2: R.Tensor((1, 3, 32, 32), dtype="float32") = R.power(x, R.const(2.0, "float32"))
                lv3: R.Tensor((1, 1, 1, 1), dtype="float32") = R.mean(
                    lv2, axis=[1, 2, 3], keepdims=True
                )
                lv4: R.Tensor((1, 1, 1, 1), dtype="float32") = R.power(lv, R.const(2.0, "float32"))
                lv5: R.Tensor((1, 1, 1, 1), dtype="float32") = R.subtract(lv3, lv4)
                lv6: R.Tensor((1, 1, 1, 1), dtype="float32") = R.sqrt(lv5)
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.divide(lv1, lv6)
                R.output(gv)
            return gv

    verify_mean_variance_norm((0, 2, 3), ExpectedDefaultAxes)
    verify_mean_variance_norm((1, 2, 3), ExpectedChannelAxes)


def test_layer_norm():
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model)

    # Test case with no bias that is an optional input
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model)

    # No bias with a non-square input where data.shape[1] differs from the scale
    # shape, see https://github.com/apache/tvm/issues/19691.
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model)

    # No bias with a non-square fp16 input. The synthesized zero bias must match
    # the scale dtype, otherwise layer_norm rejects the float32 bias, see
    # https://github.com/apache/tvm/issues/19691.
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model, opset=17, atol=1e-2, rtol=1e-2)

    # Same no-bias path for bf16. ONNX Runtime's CPU provider has no bf16
    # LayerNormalization kernel, so this only checks the importer builds the
    # graph with a bf16 zero bias (the dtype the fix derives from the scale).
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.BFLOAT16, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    model.opset_import[0].version = 17
    from_onnx(model, opset=17, keep_params_in_input=True)


def test_layer_norm_with_nd_gamma_beta():
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], axis=1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_with_nd_gamma_beta_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3, 4, 4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3, 4, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_with_nd_gamma_beta_test")
    check_correctness(model)

    # Test case with no bias that is an optional input
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_with_nd_gamma_beta_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_with_nd_gamma_beta_test")
    check_correctness(model)


def test_layer_norm_numerical_stability():
    """Numerical stability test for https://github.com/apache/tvm/issues/19592."""
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], axis=-1, epsilon=1e-5
    )
    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_numerical_stability",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4]),
        ],
    )
    model = helper.make_model(graph, producer_name="layer_norm_numerical_stability")

    input_array = np.array([[80000.0, 80001.0, 80002.0, 80003.0]], dtype=np.float32)
    scale_array = np.ones(4, dtype=np.float32)
    bias_array = np.zeros(4, dtype=np.float32)
    inputs = {"input": input_array, "scale": scale_array, "bias": bias_array}

    # ONNXRuntime also returns NaN for Large-value, small-variance inputs, so we here
    # compare against a two-pass reference instead of ORT.
    mean = input_array.mean(axis=-1, keepdims=True)
    var = ((input_array - mean) ** 2).mean(axis=-1, keepdims=True)
    expected = ((input_array - mean) / np.sqrt(var + 1e-5) * scale_array + bias_array).astype(
        np.float32
    )

    tvm_output = run_in_tvm(model, inputs=inputs, ir_version=9, opset=17)

    assert np.isfinite(tvm_output.numpy()).all()
    tvm_tirx.testing.assert_allclose(tvm_output.numpy(), expected)


def test_rms_norm():
    # Basic test: default axis=-1
    rms_norm_node = helper.make_node("RMSNormalization", ["input", "scale"], ["Y"], epsilon=1e-05)

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 8, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_test")
    check_correctness(model, opset=23)

    # Test with explicit axis=1 (normalize over last 2 dims)
    rms_norm_node = helper.make_node(
        "RMSNormalization", ["input", "scale"], ["Y"], axis=1, epsilon=1e-06
    )

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_axis_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8, 16]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [8, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8, 16]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_axis_test")
    check_correctness(model, opset=23)

    # Test with float16 input (stash_type=1 means compute in float32)
    rms_norm_node = helper.make_node(
        "RMSNormalization", ["input", "scale"], ["Y"], epsilon=1e-05, stash_type=1
    )

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_fp16_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 8, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 8, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_fp16_test")
    check_correctness(model, opset=23, rtol=1e-2, atol=1e-2)


def _make_group_norm_expected_ir(
    input_shape: list[int],
    scale_shape: list[int],
    bias_shape: list[int],
    num_groups: int,
    opset: int = 21,
    dtype: str = "float32",
    stash_type: int = 1,
):
    input_shape = tuple(input_shape)
    scale_shape = tuple(scale_shape)
    bias_shape = tuple(bias_shape)
    axes = list(range(2, len(input_shape)))
    epsilon = float(np.float32(1e-5))
    affine_shape = (input_shape[1],) + (1,) * (len(input_shape) - 2)

    if opset == 18:
        channels = input_shape[1]
        channels_per_group = channels // num_groups

        @I.ir_module
        class ExpectedGroupNormOpset18:
            @R.function
            def main(
                input: R.Tensor(input_shape, dtype=dtype),
                scale: R.Tensor(scale_shape, dtype=dtype),
                bias: R.Tensor(bias_shape, dtype=dtype),
            ) -> R.Tensor(input_shape, dtype=dtype):
                R.func_attr({"num_input": 3})
                with R.dataflow():
                    lv: R.Tensor((num_groups, 1), dtype=dtype) = R.reshape(
                        scale, R.shape([num_groups, 1])
                    )
                    lv1: R.Tensor((num_groups, channels_per_group), dtype=dtype) = R.broadcast_to(
                        lv, R.shape([num_groups, channels_per_group])
                    )
                    lv2: R.Tensor((channels,), dtype=dtype) = R.reshape(lv1, R.shape([channels]))
                    lv3: R.Tensor((num_groups, 1), dtype=dtype) = R.reshape(
                        bias, R.shape([num_groups, 1])
                    )
                    lv4: R.Tensor((num_groups, channels_per_group), dtype=dtype) = R.broadcast_to(
                        lv3, R.shape([num_groups, channels_per_group])
                    )
                    lv5: R.Tensor((channels,), dtype=dtype) = R.reshape(lv4, R.shape([channels]))
                    gv: R.Tensor(input_shape, dtype=dtype) = R.nn.group_norm(
                        input,
                        lv2,
                        lv5,
                        num_groups=num_groups,
                        channel_axis=1,
                        axes=axes,
                        epsilon=epsilon,
                    )
                    R.output(gv)
                return gv

        return ExpectedGroupNormOpset18

    if opset == 21 and stash_type == 1 and dtype != "float32":

        @I.ir_module
        class ExpectedGroupNormOpset21Stash:
            @R.function
            def main(
                input: R.Tensor(input_shape, dtype=dtype),
                scale: R.Tensor(scale_shape, dtype=dtype),
                bias: R.Tensor(bias_shape, dtype=dtype),
            ) -> R.Tensor(input_shape, dtype=dtype):
                R.func_attr({"num_input": 3})
                with R.dataflow():
                    lv: R.Tensor(input_shape, dtype="float32") = R.astype(input, dtype="float32")
                    lv1: R.Tensor(scale_shape, dtype="float32") = R.astype(scale, dtype="float32")
                    lv2: R.Tensor(scale_shape, dtype="float32") = R.ones_like(lv1)
                    lv3: R.Tensor(bias_shape, dtype="float32") = R.astype(bias, dtype="float32")
                    lv4: R.Tensor(bias_shape, dtype="float32") = R.zeros_like(lv3)
                    lv5: R.Tensor(input_shape, dtype="float32") = R.nn.group_norm(
                        lv,
                        lv2,
                        lv4,
                        num_groups=num_groups,
                        channel_axis=1,
                        axes=axes,
                        epsilon=epsilon,
                        center=False,
                        scale=False,
                    )
                    lv6: R.Tensor(input_shape, dtype=dtype) = R.astype(lv5, dtype=dtype)
                    lv7: R.Tensor(affine_shape, dtype=dtype) = R.reshape(
                        scale, R.shape(affine_shape)
                    )
                    lv8: R.Tensor(input_shape, dtype=dtype) = R.multiply(lv6, lv7)
                    lv9: R.Tensor(affine_shape, dtype=dtype) = R.reshape(
                        bias, R.shape(affine_shape)
                    )
                    gv: R.Tensor(input_shape, dtype=dtype) = R.add(lv8, lv9)
                    R.output(gv)
                return gv

        return ExpectedGroupNormOpset21Stash

    if opset == 21:

        @I.ir_module
        class ExpectedGroupNormOpset21:
            @R.function
            def main(
                input: R.Tensor(input_shape, dtype=dtype),
                scale: R.Tensor(scale_shape, dtype=dtype),
                bias: R.Tensor(bias_shape, dtype=dtype),
            ) -> R.Tensor(input_shape, dtype=dtype):
                R.func_attr({"num_input": 3})
                with R.dataflow():
                    lv: R.Tensor(scale_shape, dtype=dtype) = R.ones_like(scale)
                    lv1: R.Tensor(bias_shape, dtype=dtype) = R.zeros_like(bias)
                    lv2: R.Tensor(input_shape, dtype=dtype) = R.nn.group_norm(
                        input,
                        lv,
                        lv1,
                        num_groups=num_groups,
                        channel_axis=1,
                        axes=axes,
                        epsilon=epsilon,
                        center=False,
                        scale=False,
                    )
                    lv3: R.Tensor(affine_shape, dtype=dtype) = R.reshape(
                        scale, R.shape(affine_shape)
                    )
                    lv4: R.Tensor(input_shape, dtype=dtype) = R.multiply(lv2, lv3)
                    lv5: R.Tensor(affine_shape, dtype=dtype) = R.reshape(
                        bias, R.shape(affine_shape)
                    )
                    gv: R.Tensor(input_shape, dtype=dtype) = R.add(lv4, lv5)
                    R.output(gv)
                return gv

        return ExpectedGroupNormOpset21

    raise AssertionError(f"No GroupNormalization expected IR for opset={opset}")


def test_group_norm():
    def verify_group_norm(
        input_shape: list[int],
        scale_shape: list[int],
        bias_shape: list[int],
        num_groups: int,
        expected,
        opset: int = 21,
        dtype: int = TensorProto.FLOAT,
        stash_type: int = 1,
    ):
        attrs = {"num_groups": num_groups, "epsilon": 1e-5}
        if opset == 21:
            attrs["stash_type"] = stash_type

        node = helper.make_node(
            "GroupNormalization", ["input", "scale", "bias"], ["output"], **attrs
        )
        graph = helper.make_graph(
            [node],
            "group_norm_test",
            inputs=[
                helper.make_tensor_value_info("input", dtype, list(input_shape)),
                helper.make_tensor_value_info("scale", dtype, list(scale_shape)),
                helper.make_tensor_value_info("bias", dtype, list(bias_shape)),
            ],
            outputs=[helper.make_tensor_value_info("output", dtype, list(input_shape))],
        )

        model = helper.make_model(
            graph,
            producer_name="group_norm_test",
            opset_imports=[helper.make_opsetid("", opset)],
        )
        tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        expected = tvm_tirx.IRModule(expected.functions)
        for gv in expected.get_global_vars():
            if gv.name_hint != "main":
                expected.update_func(gv, tvm_model[gv.name_hint])
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    for input_shape, scale_shape, bias_shape, num_groups, opset, dtype, dtype_str, stash_type in [
        ([1, 4, 2, 2], [2], [2], 2, 18, TensorProto.FLOAT, "float32", 1),
        ([1, 4, 2, 2], [4], [4], 2, 21, TensorProto.FLOAT, "float32", 1),
        ([1, 4, 8], [4], [4], 2, 21, TensorProto.FLOAT, "float32", 1),
        ([1, 4, 2, 2], [4], [4], 2, 21, TensorProto.FLOAT16, "float16", 1),
    ]:
        verify_group_norm(
            input_shape,
            scale_shape,
            bias_shape,
            num_groups,
            _make_group_norm_expected_ir(
                input_shape,
                scale_shape,
                bias_shape,
                num_groups,
                opset=opset,
                dtype=dtype_str,
                stash_type=stash_type,
            ),
            opset=opset,
            dtype=dtype,
            stash_type=stash_type,
        )

    for bad_stash_type in [0, 10, 11, 16]:
        with pytest.raises(ValueError, match="stash_type=1"):
            verify_group_norm(
                [1, 4, 2, 2],
                [4],
                [4],
                2,
                _make_group_norm_expected_ir(
                    [1, 4, 2, 2],
                    [4],
                    [4],
                    2,
                    opset=21,
                    dtype="float16",
                    stash_type=1,
                ),
                opset=21,
                dtype=TensorProto.FLOAT16,
                stash_type=bad_stash_type,
            )

    with pytest.raises(ValueError, match="currently only supports float32"):
        verify_group_norm(
            [1, 4, 2, 2],
            [2],
            [2],
            2,
            _make_group_norm_expected_ir(
                [1, 4, 2, 2],
                [2],
                [2],
                2,
                opset=18,
                dtype="float16",
            ),
            opset=18,
            dtype=TensorProto.FLOAT16,
        )


# TODO Enable dynamism
@pytest.mark.parametrize("dynamic", [False])
def test_skiplayernormalization(dynamic):
    def verify_skiplayernormalization(input_, skip, gamma, beta, bias):
        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["input", "skip", "gamma", "beta", "bias"],
            outputs=["output", "mean", "std_dev"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        input_shape = list(input_.shape)
        skip_shape = list(skip.shape)
        gamma_shape = list(gamma.shape)
        beta_shape = list(beta.shape)
        bias_shape = list(bias.shape)
        output_shape = list(input_.shape)
        mean_shape = list([1])
        std_dev_shape = list([1])
        if dynamic:
            input_shape = ["?" for _ in range(len(input_.shape))]
            skip_shape = ["?" for _ in range(len(skip.shape))]
            gamma_shape = ["?" for _ in range(len(gamma.shape))]
            beta_shape = ["?" for _ in range(len(beta.shape))]
            bias_shape = ["?" for _ in range(len(bias.shape))]
            output_shape = ["?" for _ in range(len(input_.shape))]

        graph = helper.make_graph(
            [node],
            "skiplayernormalization_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT, skip_shape),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, gamma_shape),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, beta_shape),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias_shape),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean_shape),
                helper.make_tensor_value_info("std_dev", TensorProto.FLOAT, std_dev_shape),
            ],
        )

        model = helper.make_model(graph, producer_name="skiplayernormalization_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)

        @I.ir_module
        class Expected:
            @R.function
            def main(
                input: R.Tensor((4, 4, 384), dtype="float32"),
                skip: R.Tensor((4, 4, 384), dtype="float32"),
                gamma: R.Tensor((384,), dtype="float32"),
                beta: R.Tensor((384,), dtype="float32"),
                bias: R.Tensor((384,), dtype="float32"),
            ) -> R.Tuple(
                R.Tensor((4, 4, 384), dtype="float32"),
                R.Tensor((), dtype="float32"),
                R.Tensor((), dtype="float32"),
            ):
                R.func_attr({"num_input": 5})
                with R.dataflow():
                    lv: R.Tensor((4, 4, 384), dtype="float32") = R.add(input, skip)
                    lv1: R.Tensor((4, 4, 384), dtype="float32") = R.add(lv, bias)
                    lv2: R.Tensor((4, 4, 384), dtype="float32") = R.nn.layer_norm(
                        lv1, gamma, beta, axes=-1, epsilon=9.999999747378752e-05
                    )
                    gv: R.Tuple(
                        R.Tensor((4, 4, 384), dtype="float32"),
                        R.Tensor((), dtype="float32"),
                        R.Tensor((), dtype="float32"),
                    ) = (lv2, R.const(0, "float32"), R.const(0, "float32"))
                    R.output(gv)
                return gv

        tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)

    hidden_size = 384
    batch_size = 4
    sequence_length = 4

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    skip = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype(dtype)
    beta = np.random.randn(hidden_size).astype(dtype) * 0.1
    bias = np.random.randn(hidden_size).astype(dtype)

    verify_skiplayernormalization(input_array, skip, gamma, beta, bias)


def test_embedlayernormalization():
    def verify_embedlayernormalization(
        input_ids,
        segment_ids,
        word_embedding,
        position_embedding,
        segment_embedding,
        gamma,
        beta,
        expected,
    ):
        node = onnx.helper.make_node(
            "EmbedLayerNormalization",
            inputs=[
                "input_ids",
                "" if segment_ids is None else "segment_ids",
                "word_embedding",
                "position_embedding",
                "" if segment_embedding is None else "segment_embedding",
                "gamma",
                "beta",
            ],
            outputs=["output", "mask_index"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        segment_ids_shape = [] if segment_ids is None else segment_ids.shape
        segment_embedding_shape = [] if segment_embedding is None else segment_embedding.shape

        graph = helper.make_graph(
            [node],
            "embedlayernormalization_test",
            inputs=[
                helper.make_tensor_value_info(
                    "input_ids", TensorProto.INT32, list(input_ids.shape)
                ),
                helper.make_tensor_value_info("segment_ids", TensorProto.INT32, segment_ids_shape),
                helper.make_tensor_value_info(
                    "word_embedding", TensorProto.FLOAT, list(word_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "position_embedding", TensorProto.FLOAT, list(position_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "segment_embedding", TensorProto.FLOAT, segment_embedding_shape
                ),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, list(gamma.shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, list(beta.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, list((batch_size, sequence_length, hidden_size))
                ),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, [batch_size]),
            ],
        )

        model = helper.make_model(graph, producer_name="embedlayernormalization_test")

        tvm_model = from_onnx(model, keep_params_in_input=True)

        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

        # TODO(@anwang2009): onnxruntime v1.9.0 requires empty list for optional argument,
        # but v1.10.0+ requires None instead.
        # verify_with_ort_with_inputs(
        #     model,
        #     [
        #         input_ids,
        #         np.empty(0, dtype="int32") if segment_ids is None else segment_ids,
        #         word_embedding,
        #         position_embedding,
        #         np.empty(0, dtype="float32") if segment_embedding is None else segment_embedding,
        #         gamma,
        #         beta,
        #     ],
        #     [
        #         (batch_size, sequence_length, hidden_size),
        #         batch_size,
        #     ],
        #     target=target,
        #     dev=dev,
        #     rtol=1e-4,
        #     atol=1e-4,
        # )

    hidden_size = 384
    batch_size = 4
    sequence_length = 3
    vocab_size = 5

    input_ids = np.full((batch_size, sequence_length), 3).astype("int32")
    segment_ids = np.zeros((batch_size, sequence_length)).astype("int32")
    word_embedding = np.full((vocab_size, hidden_size), 1).astype("float32")
    position_embedding = np.full((sequence_length, hidden_size), 2).astype("float32")
    segment_embedding = np.full((vocab_size, hidden_size), 3).astype("float32")

    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype("float32")
    beta = np.random.randn(hidden_size).astype("float32") * 0.1

    @I.ir_module
    class ExpectedNoSegment:
        @R.function
        def main(
            input_ids: R.Tensor((4, 3), dtype="int32"),
            segment_ids: R.Tensor((), dtype="int32"),
            word_embedding: R.Tensor((5, 384), dtype="float32"),
            position_embedding: R.Tensor((3, 384), dtype="float32"),
            segment_embedding: R.Tensor((), dtype="float32"),
            gamma: R.Tensor((384,), dtype="float32"),
            beta: R.Tensor((384,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((4, 3, 384), dtype="float32"),
            R.Tensor((4,), dtype="int32"),
        ):
            R.func_attr({"num_input": 7})
            with R.dataflow():
                lv: R.Tensor((4, 3, 384), dtype="float32") = R.take(
                    word_embedding, input_ids, axis=0, mode="fast"
                )
                lv1: R.Tensor((4, 3, 384), dtype="float32") = R.take(
                    position_embedding,
                    R.const([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], "int64"),
                    axis=0,
                    mode="fast",
                )
                lv2: R.Tensor((4, 3, 384), dtype="float32") = R.add(lv, lv1)
                lv3: R.Tensor((4, 3, 384), dtype="float32") = R.nn.layer_norm(
                    lv2, gamma, beta, axes=-1, epsilon=9.999999747378752e-05
                )
                gv: R.Tuple(
                    R.Tensor((4, 3, 384), dtype="float32"),
                    R.Tensor((4,), dtype="int32"),
                ) = (lv3, R.const([0, 0, 0, 0], "int32"))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedWithSegment:
        @R.function
        def main(
            input_ids: R.Tensor((4, 3), dtype="int32"),
            segment_ids: R.Tensor((4, 3), dtype="int32"),
            word_embedding: R.Tensor((5, 384), dtype="float32"),
            position_embedding: R.Tensor((3, 384), dtype="float32"),
            segment_embedding: R.Tensor((5, 384), dtype="float32"),
            gamma: R.Tensor((384,), dtype="float32"),
            beta: R.Tensor((384,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((4, 3, 384), dtype="float32"),
            R.Tensor((4,), dtype="int32"),
        ):
            R.func_attr({"num_input": 7})
            with R.dataflow():
                lv: R.Tensor((4, 3, 384), dtype="float32") = R.take(
                    word_embedding, input_ids, axis=0, mode="fast"
                )
                lv1: R.Tensor((4, 3, 384), dtype="float32") = R.take(
                    position_embedding,
                    R.const([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], "int64"),
                    axis=0,
                    mode="fast",
                )
                lv2: R.Tensor((4, 3, 384), dtype="float32") = R.add(lv, lv1)
                lv3: R.Tensor((4, 3, 384), dtype="float32") = R.take(
                    segment_embedding, segment_ids, axis=0, mode="fast"
                )
                lv4: R.Tensor((4, 3, 384), dtype="float32") = R.add(lv2, lv3)
                lv5: R.Tensor((4, 3, 384), dtype="float32") = R.nn.layer_norm(
                    lv4, gamma, beta, axes=-1, epsilon=9.999999747378752e-05
                )
                gv: R.Tuple(
                    R.Tensor((4, 3, 384), dtype="float32"),
                    R.Tensor((4,), dtype="int32"),
                ) = (lv5, R.const([0, 0, 0, 0], "int32"))
                R.output(gv)
            return gv

    verify_embedlayernormalization(
        input_ids,
        segment_ids,
        word_embedding,
        position_embedding,
        segment_embedding,
        gamma,
        beta,
        ExpectedWithSegment,
    )

    # Test with undefined segment embedding
    verify_embedlayernormalization(
        input_ids,
        None,
        word_embedding,
        position_embedding,
        None,
        gamma,
        beta,
        ExpectedNoSegment,
    )


def test_local_response_norm():
    lrn_node = helper.make_node(
        op_type="LRN",
        inputs=["input"],
        outputs=["output"],
        name="LRN_Node",
        alpha=0.0001,
        beta=0.75,
        bias=1.0,
        size=3,
    )

    graph = helper.make_graph(
        [lrn_node],
        "local_response_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="local_response_norm_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            input: R.Tensor((1, 3, 32, 32), dtype="float32"),
        ) -> R.Tensor((1, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.multiply(input, input)
                lv1: R.Tensor((1, 1, 3, 32, 32), dtype="float32") = R.expand_dims(lv, axis=[1])
                lv2: R.Tensor((1, 1, 3, 32, 32), dtype="float32") = R.nn.avg_pool3d(
                    lv1,
                    pool_size=[3, 1, 1],
                    strides=[1, 1, 1],
                    dilation=[1, 1, 1],
                    padding=[1, 0, 0, 1, 0, 0],
                    ceil_mode=False,
                    count_include_pad=True,
                    layout="NCDHW",
                    out_layout="NCDHW",
                )
                lv3: R.Tensor((1, 3, 32, 32), dtype="float32") = R.squeeze(lv2, axis=[1])
                lv4: R.Tensor((1, 3, 32, 32), dtype="float32") = R.multiply(
                    lv3, R.const(9.9999997473787516e-05, "float32")
                )
                lv5: R.Tensor((1, 3, 32, 32), dtype="float32") = R.add(lv4, R.const(1.0, "float32"))
                lv6: R.Tensor((1, 3, 32, 32), dtype="float32") = R.power(
                    lv5, R.const(0.75, "float32")
                )
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = R.divide(input, lv6)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


COMPOSITE_REDUCE_FUNCS = [
    "ReduceSumSquare",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceL1",
    "ReduceL2",
]

REDUCE_AXES_ATTR_TEST_CASES = [
    ([3, 2, 2], None),
    ([3, 2, 3], None),
    ([3, 3, 3], (1,)),
    ([3, 3, 3, 1], (1, 2)),
    ([3, 3, 3, 1], (1,)),
    ([1, 3, 4, 1], (1,)),
]

REDUCE_AXES_INPUT_TEST_CASES = [
    ([3, 2, 2], [], False),
    ([3, 2, 2], None, False),
    ([4, 3], [], True),
    ([3, 3, 3, 1], (1, 2), False),
]


def _reduce_output_shape(input_shape: list[int], axes, keepdims: bool, noop_with_empty_axes=False):
    if noop_with_empty_axes and not axes:
        return list(input_shape)
    axis = None if not axes else axes
    return list(np.sum(np.empty(input_shape), axis=axis, keepdims=keepdims).shape)


def verify_composite_reduce_axes_attr_ir(
    func: str,
    input_shape: list[int],
    axes,
    keepdims: bool,
    dynamic: bool,
    opset: int,
    expected,
):
    attrs = {"keepdims": keepdims}
    if axes:
        attrs["axes"] = axes
    node = onnx.helper.make_node(func, inputs=["x"], outputs=["y"], **attrs)
    output_shape = _reduce_output_shape(input_shape, axes, keepdims)
    graph = helper.make_graph(
        [node],
        "composite_reduce_axes_attr_ir_test",
        inputs=[
            helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, ["?"] * len(input_shape) if dynamic else input_shape
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, ["?"] * len(output_shape) if dynamic else output_shape
            )
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="composite_reduce_axes_attr_ir_test",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    tvm_tirx.ir.assert_structural_equal(tvm_model, expected, map_free_vars=dynamic)


def create_reduce_test_parameters_axes_attr():
    output = []
    for value in [True, False]:
        output.append(("ReduceMax", value, 11))
        output.append(("ReduceMean", value, 13))
        output.append(("ReduceMin", value, 11))
        output.append(("ReduceProd", value, 13))
        output.append(("ReduceSum", value, 11))
        # Opset 11-12 axes-as-attr: verifies get_converter does not
        # underflow to the v18 (axes-as-input) implementation.
        output.append(("ReduceMean", value, 11))
        output.append(("ReduceProd", value, 11))
    return output


def _verify_reduce_numerical(
    func: str,
    opset: int,
    *,
    axes_as_input: bool,
    axes,
    noop_with_empty_axes: bool = False,
    dynamic: bool = False,
    keepdims: bool = False,
):
    input_shape = [3, 3, 3]
    node_inputs = ["x"]
    initializers = []
    node_attrs = {"keepdims": keepdims}

    if axes_as_input:
        node_attrs["noop_with_empty_axes"] = noop_with_empty_axes
        if axes is not None:
            axes_np = np.asarray(axes, dtype=np.int64)
            initializers.append(
                helper.make_tensor(
                    name="reduce_axes",
                    data_type=TensorProto.INT64,
                    dims=axes_np.shape,
                    vals=axes_np,
                )
            )
            node_inputs.append("reduce_axes")
    elif axes:
        node_attrs["axes"] = axes

    if axes_as_input and noop_with_empty_axes and not axes:
        output_shape = input_shape
    else:
        axis = None if not axes else tuple(axes)
        output_shape = list(np.sum(np.empty(input_shape), axis=axis, keepdims=keepdims).shape)

    graph_input_shape = ["?"] * len(input_shape) if dynamic else input_shape
    graph_output_shape = ["?"] * len(output_shape) if dynamic else output_shape

    node = helper.make_node(func, inputs=node_inputs, outputs=["y"], **node_attrs)
    graph = helper.make_graph(
        [node],
        "reduce_numerical_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, graph_input_shape)],
        initializer=initializers,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, graph_output_shape)],
    )
    model = helper.make_model(
        graph,
        producer_name="reduce_numerical_test",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    inputs = {"x": np.random.randn(*input_shape).astype(np.float32)}
    check_correctness(model, inputs, opset=opset, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("func, dynamic, opset", create_reduce_test_parameters_axes_attr())
def test_all_reduce_funcs_axes_attr(func, dynamic, opset):
    for keepdims in [True, False]:
        for input_shape, axes in REDUCE_AXES_ATTR_TEST_CASES:
            expected = _make_reduce_expected_ir(func, input_shape, axes, False, keepdims, dynamic)
            verify_composite_reduce_axes_attr_ir(
                func, input_shape, axes, keepdims, dynamic, opset, expected
            )


@pytest.mark.parametrize(
    "func, opset, dynamic, keepdims",
    [
        ("ReduceMax", 11, False, False),
        ("ReduceMean", 11, False, False),
        ("ReduceMean", 13, True, True),
        ("ReduceMin", 11, False, False),
        ("ReduceProd", 11, False, False),
        ("ReduceProd", 13, False, False),
        ("ReduceSum", 11, False, False),
    ],
)
def test_reduce_funcs_axes_attr_numerical(func, opset, dynamic, keepdims):
    _verify_reduce_numerical(
        func,
        opset,
        axes_as_input=False,
        axes=[1],
        dynamic=dynamic,
        keepdims=keepdims,
    )


def _make_reduce_expected_ir(
    func: str,
    input_shape: list[int],
    axes,
    noop_with_empty_axes: bool,
    keepdims: bool,
    dynamic: bool,
    axes_as_input: bool = False,
):
    def expected_input_shape(shape):
        if not dynamic:
            return tuple(shape)
        return tuple(tirx.Var(f"reduce_dim_{i}", "int64") for i in range(len(shape)))

    axis = None if not axes else tuple(axes)
    parser_vars = {
        "I": I,
        "R": R,
        "input_shape": expected_input_shape(input_shape),
        "axis": axis,
        "keepdims": keepdims,
    }
    params = ['        x: R.Tensor(input_shape, dtype="float32")']
    if axes_as_input and axes is not None:
        axes_shape = tuple(np.asarray(axes, dtype=np.int64).shape)
        parser_vars["axes_shape"] = axes_shape
        params.append('        reduce_axes: R.Tensor(axes_shape, dtype="int64")')

    basic_reduce_op = {
        "ReduceMax": R.max,
        "ReduceMean": R.mean,
        "ReduceMin": R.min,
        "ReduceProd": R.prod,
        "ReduceSum": R.sum,
    }.get(func)

    if noop_with_empty_axes and not axes:
        body = ["            gv = x"]
    elif basic_reduce_op is not None:
        parser_vars["reduce_op"] = basic_reduce_op
        body = ["            gv = reduce_op(x, axis=axis, keepdims=keepdims)"]
    elif func == "ReduceSumSquare":
        body = [
            "            lv = R.multiply(x, x)",
            "            gv = R.sum(lv, axis=axis, keepdims=keepdims)",
        ]
    elif func == "ReduceLogSum":
        body = [
            "            lv = R.sum(x, axis=axis, keepdims=keepdims)",
            "            gv = R.log(lv)",
        ]
    elif func == "ReduceLogSumExp":
        parser_vars["logsumexp_keepdims"] = True
        body = [
            "            lv = R.max(x, axis=axis, keepdims=logsumexp_keepdims)",
            "            lv1 = R.subtract(x, lv)",
            "            lv2 = R.exp(lv1)",
            "            lv3 = R.sum(lv2, axis=axis, keepdims=logsumexp_keepdims)",
            "            lv4 = R.log(lv3)",
        ]
        if keepdims:
            body.append("            gv = R.add(lv4, lv)")
        else:
            parser_vars["squeeze_axis"] = None if axis is None else list(axis)
            body += [
                "            lv5 = R.add(lv4, lv)",
                "            gv = R.squeeze(lv5, axis=squeeze_axis)",
            ]
    elif func == "ReduceL1":
        body = [
            "            lv = R.abs(x)",
            "            gv = R.sum(lv, axis=axis, keepdims=keepdims)",
        ]
    elif func == "ReduceL2":
        body = [
            "            lv = R.multiply(x, x)",
            "            lv1 = R.sum(lv, axis=axis, keepdims=keepdims)",
            "            gv = R.sqrt(lv1)",
        ]
    else:
        raise AssertionError(f"No composite reduce expected IR for {func}")

    source = "\n".join(
        [
            "@I.ir_module",
            "class Expected:",
            "    @R.function",
            "    def main(",
            ",\n".join(params),
            "    ):",
            '        R.func_attr({"num_input": 1})',
            "        with R.dataflow():",
            *body,
            "            R.output(gv)",
            "        return gv",
            "",
        ]
    )
    return tvm_tirx.script.from_source(source, extra_vars=parser_vars)


def test_composite_reduce_funcs_axes_attr_ir():
    for func in COMPOSITE_REDUCE_FUNCS:
        for keepdims in [True, False]:
            for dynamic in [True, False]:
                for input_shape, axes in REDUCE_AXES_ATTR_TEST_CASES:
                    expected = _make_reduce_expected_ir(
                        func, input_shape, axes, False, keepdims, dynamic
                    )
                    for opset in [13, 11]:
                        verify_composite_reduce_axes_attr_ir(
                            func, input_shape, axes, keepdims, dynamic, opset, expected
                        )


def create_reduce_test_parameters_axes_input():
    output = []
    for dynamic in [True, False]:
        output.append(("ReduceMax", dynamic, 18))
        output.append(("ReduceMean", dynamic, 18))
        output.append(("ReduceMin", dynamic, 18))
        output.append(("ReduceProd", dynamic, 18))
        output.append(("ReduceSum", dynamic, 13))
    return output


def verify_composite_reduce_axes_input_ir(
    func: str,
    input_shape: list[int],
    axes,
    noop_with_empty_axes: bool,
    keepdims: bool,
    dynamic: bool,
    opset: int,
    expected,
):
    node_inputs = ["x"]
    initializers = []
    axes_input_shape = None
    if axes is not None:
        axes_np = np.asarray(axes, dtype=np.int64)
        axes_input_shape = list(axes_np.shape)
        initializers.append(
            helper.make_tensor(
                name="reduce_axes",
                data_type=TensorProto.INT64,
                dims=axes_input_shape,
                vals=axes_np,
            )
        )
        node_inputs.append("reduce_axes")

    effective_axes = None if not axes and not noop_with_empty_axes else axes
    output_shape = _reduce_output_shape(
        input_shape, effective_axes, keepdims, noop_with_empty_axes=noop_with_empty_axes
    )
    node = onnx.helper.make_node(
        func,
        inputs=node_inputs,
        outputs=["y"],
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
    )
    graph = helper.make_graph(
        [node],
        "composite_reduce_axes_input_ir_test",
        inputs=[
            helper.make_tensor_value_info(
                "x", TensorProto.FLOAT, ["?"] * len(input_shape) if dynamic else input_shape
            )
        ],
        initializer=initializers,
        outputs=[
            helper.make_tensor_value_info(
                "y", TensorProto.FLOAT, ["?"] * len(output_shape) if dynamic else output_shape
            )
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="composite_reduce_axes_input_ir_test",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    if axes_input_shape is not None:
        assert len(tvm_model["main"].attrs["params"]) == 1
        tvm_model["main"] = tvm_model["main"].without_attr("params")

    tvm_tirx.ir.assert_structural_equal(tvm_model, expected, map_free_vars=dynamic)


@pytest.mark.parametrize("func, dynamic, opset", create_reduce_test_parameters_axes_input())
def test_all_reduce_funcs_axes_input(func, dynamic, opset):
    for keepdims in [True, False]:
        for input_shape, axes, noop_with_empty_axes in REDUCE_AXES_INPUT_TEST_CASES:
            expected = _make_reduce_expected_ir(
                func,
                input_shape,
                axes,
                noop_with_empty_axes,
                keepdims,
                dynamic,
                axes_as_input=True,
            )
            verify_composite_reduce_axes_input_ir(
                func,
                input_shape,
                axes,
                noop_with_empty_axes,
                keepdims,
                dynamic,
                opset,
                expected,
            )


@pytest.mark.parametrize(
    "func, opset, axes, noop_with_empty_axes, dynamic, keepdims",
    [
        ("ReduceMax", 18, [1], False, False, False),
        ("ReduceMean", 18, [1], False, True, True),
        ("ReduceMin", 18, [1], False, False, False),
        ("ReduceProd", 18, [1], False, False, False),
        ("ReduceSum", 13, [1], False, False, False),
        ("ReduceSum", 13, [], True, False, False),
    ],
)
def test_reduce_funcs_axes_input_numerical(
    func, opset, axes, noop_with_empty_axes, dynamic, keepdims
):
    _verify_reduce_numerical(
        func,
        opset,
        axes_as_input=True,
        axes=axes,
        noop_with_empty_axes=noop_with_empty_axes,
        dynamic=dynamic,
        keepdims=keepdims,
    )


def test_composite_reduce_funcs_axes_input_ir():
    for func in COMPOSITE_REDUCE_FUNCS:
        for keepdims in [True, False]:
            for dynamic in [True, False]:
                for input_shape, axes, noop_with_empty_axes in REDUCE_AXES_INPUT_TEST_CASES:
                    expected = _make_reduce_expected_ir(
                        func,
                        input_shape,
                        axes,
                        noop_with_empty_axes,
                        keepdims,
                        dynamic,
                        axes_as_input=True,
                    )
                    verify_composite_reduce_axes_input_ir(
                        func,
                        input_shape,
                        axes,
                        noop_with_empty_axes,
                        keepdims,
                        dynamic,
                        18,
                        expected,
                    )


@pytest.mark.parametrize("axis", [None, 0, 1, 2])
@pytest.mark.parametrize("keepdims", [None, True, False])
def test_arg_min_max(axis, keepdims):
    def verify_arg_min_max(input_shape, op_name="ArgMax", axis=None, keepdims=None):
        out_shape = list(input_shape)
        def_axis = axis if axis is not None else 0
        if keepdims == 1 or keepdims is None:
            out_shape[def_axis] = 1
        else:
            out_shape.pop(def_axis)

        node = helper.make_node(op_name, inputs=["a_np1"], outputs=["out"])

        if keepdims is not None:
            keepdims_attr = helper.make_attribute("keepdims", keepdims)
            node.attribute.append(keepdims_attr)
        if axis is not None:
            axis_attr = helper.make_attribute("axis", axis)
            node.attribute.append(axis_attr)

        graph = helper.make_graph(
            [node],
            "argreduce_test",
            inputs=[helper.make_tensor_value_info("a_np1", TensorProto.INT32, input_shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, out_shape)],
        )

        model = helper.make_model(graph, producer_name="arg_min_max_test")
        check_correctness(model)

    verify_arg_min_max([3, 4, 4], "ArgMax", axis, keepdims)
    verify_arg_min_max([3, 4, 4], "ArgMin", axis, keepdims)


@pytest.mark.parametrize("axis", [-1, 0, 1])
@pytest.mark.parametrize("largest", [True, False])
def test_topk(axis: int, largest: int):
    in_shape = [32, 32, 32]
    k_value = 4
    out_shape = in_shape
    out_shape[axis] = k_value
    k = make_constant_node("k", TensorProto.INT64, [1], [k_value])
    node = onnx.helper.make_node(
        "TopK",
        inputs=["data", "k"],
        outputs=["values", "indices"],
        axis=axis,
        largest=largest,
    )
    graph = helper.make_graph(
        [k, node],
        "topk_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, in_shape)],
        outputs=[
            helper.make_tensor_value_info("values", TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, out_shape),
        ],
    )
    model = helper.make_model(graph, producer_name="topk_test")

    check_correctness(model, check_dtypes=True)


def test_expand():
    def _assert_expand_ir(name, input_shape, target_shape, output_shape, expected):
        shape_array = np.array(target_shape)
        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten().astype("int64"),
            ),
        )
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_teint64st",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name=name)
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    def _assert_expand_dynamic_shapeexpr_ir(name, input_shape, shape_input_shape, expected):
        shape_node = onnx.helper.make_node("Shape", inputs=["in_2"], outputs=["shape"])
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])
        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_test",
            inputs=[
                helper.make_tensor_value_info("in", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("in_2", TensorProto.FLOAT, shape_input_shape),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, shape_input_shape)],
        )

        model = helper.make_model(graph, producer_name=name)
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedSameRank:
        @R.function
        def main(in_: R.Tensor((3, 1), dtype="float32")) -> R.Tensor((3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 4), dtype="float32") = R.broadcast_to(in_, R.shape([3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedHigherRank:
        @R.function
        def main(in_: R.Tensor((3, 1), dtype="float32")) -> R.Tensor((1, 3, 4), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 4), dtype="float32") = R.broadcast_to(in_, R.shape([1, 3, 4]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSameSuffix:
        @R.function
        def main(in_: R.Tensor((3, 1), dtype="float32")) -> R.Tensor((1, 1, 3, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 1, 3, 1), dtype="float32") = R.broadcast_to(
                    in_, R.shape([1, 1, 3, 1])
                )
                R.output(gv)
            return gv

    batch = T.dynamic("batch", "int64")

    @I.ir_module
    class ExpectedDynamicShape:
        @R.function
        def main(
            in_: R.Tensor((1, 32, 32), dtype="float32"),
            in_2: R.Tensor((batch, 32, 32), dtype="float32"),
        ) -> R.Tensor((batch, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((batch, 32, 32), dtype="float32") = R.broadcast_to(
                    in_, R.shape([batch, 32, 32])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedHigherRankSamePaddedShape:
        @R.function
        def main(
            in_: R.Tensor((1,), dtype="float32"),
            in_2: R.Tensor((1, 1), dtype="float32"),
        ) -> R.Tensor((1, 1), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((1, 1), dtype="float32") = R.broadcast_to(in_, R.shape([1, 1]))
                R.output(gv)
            return gv

    _assert_expand_ir("expand_with_dim_unchanged_test", [3, 1], [3, 4], [3, 4], ExpectedSameRank)
    _assert_expand_ir("expand_with_diff_dim", [3, 1], [1, 3, 4], [1, 3, 4], ExpectedHigherRank)
    _assert_expand_ir(
        "expand_with_the_same_suffix_dims", [3, 1], [1, 1, 3, 1], [1, 1, 3, 1], ExpectedSameSuffix
    )
    _assert_expand_dynamic_shapeexpr_ir(
        "expand_with_dynamic_dim", [1, 32, 32], ["batch", 32, 32], ExpectedDynamicShape
    )
    _assert_expand_dynamic_shapeexpr_ir(
        "expand_with_higher_rank_same_padded_shape",
        [1],
        [1, 1],
        ExpectedHigherRankSamePaddedShape,
    )


def test_expand_incompatible_broadcasting():
    """
    This test case reproduces the error where input tensor shape at dim 1 is 25
    and target shape at dim 3 is 56, which violates ONNX broadcasting rules
    """

    def _test_expand_error_case(name, data_shape, target_shape_vals):
        data = np.random.uniform(size=data_shape).astype(np.float32)

        shape_array = np.array(target_shape_vals, dtype=np.int64)
        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten(),
            ),
        )

        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_error_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(data.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, target_shape_vals)],
        )

        model = helper.make_model(graph, producer_name=name)

        with pytest.raises(ValueError) as exc_info:
            from_onnx(model, keep_params_in_input=True)

        error_msg = str(exc_info.value)
        assert "broadcast" in error_msg.lower() or "incompatible" in error_msg.lower(), (
            f"Expected broadcasting error, but got: {error_msg}"
        )

    # Test case 1: Reproduce the exact error from the issue-17769
    # Input shape: (25,), target shape: (1, 1, 1, 56)
    # This should faill because input dim 1 (25) != target dim 3 (56) and neither is 1
    _test_expand_error_case(
        "expand_incompatible_25_to_56",
        data_shape=(25,),
        target_shape_vals=(1, 1, 1, 56),
    )

    # Test case 2: Another incompatible case
    # Input shape: (1, 25), target shape: (1, 1, 1, 56)
    # After right-alignment, input (1, 1, 1, 25) vs. target (1, 1, 1, 56)
    # This should fail because 25 != 56 and neither is 1
    _test_expand_error_case(
        "expand_incompatible_aligned_25_to_56",
        data_shape=(1, 25),
        target_shape_vals=(1, 1, 1, 56),
    )

    # Test case 3: Valid case for comparison - should not raise error
    def _test_expand_valid_case():
        """Test a valid expand case to ensure our fix doesn't break valid operations"""
        data_shape = (1, 25)
        target_shape_vals = [2, 25]  # Valid: input (1, 25) can broadcast to (2, 25)

        data = np.random.uniform(size=data_shape).astype(np.float32)
        shape_array = np.array(target_shape_vals, dtype=np.int64)

        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten(),
            ),
        )

        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_valid_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(data.shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, target_shape_vals)],
        )

        model = helper.make_model(graph, producer_name="expand_valid_test_case")

        try:
            tvm_model = from_onnx(model, keep_params_in_input=True)
        except Exception as e:
            pytest.fail(f"Valid expand case should not fail, but got error: {e}")

    _test_expand_valid_case()


# TODO(jwfromm) Current approach to dynamic expand is technically not well formed. Reenable once fixed.
@pytest.mark.skip("Produces ill-formed IR")
def test_constantofshape():
    def verify_constantofshape(input_dim, value, dtype):
        fill_node = helper.make_node(
            "ConstantOfShape",
            ["input"],
            ["output"],
            value=helper.make_tensor(
                "value", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), (1,), (value,)
            ),
        )

        inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [len(input_dim)])]

        graph = helper.make_graph(
            [fill_node],
            "fill_test",
            inputs,
            initializer=[
                helper.make_tensor(
                    "input",
                    TensorProto.INT64,
                    [len(input_dim)],
                    np.asarray(input_dim).astype("int64"),
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), input_dim
                )
            ],
        )

        model = helper.make_model(graph, producer_name="fill_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        assert tuple(dim.value for dim in tvm_model["main"].ret_ty.shape.values) == input_dim

    verify_constantofshape((2, 3, 4, 5), 10, "float32")
    verify_constantofshape((3, 3), 0, "int32")
    verify_constantofshape((1, 2, 3), -1, "float32")


def test_constantofshape_default_value():
    """ConstantOfShape value attribute should default to float32 zero."""
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3])
    node = helper.make_node("ConstantOfShape", ["shape"], ["y"])
    graph = helper.make_graph(
        [node],
        "constantofshape_default_value_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, producer_name="constantofshape_default_value_test")

    tvm_model = from_onnx(model)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.broadcast_to(
                    R.const(0.0, "float32"), R.shape([2, 3])
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_constantofshape_initializer_shape_with_keep_params_in_input():
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [1], [3])
    node = helper.make_node(
        "ConstantOfShape",
        ["shape"],
        ["y"],
        value=helper.make_tensor("value", TensorProto.INT64, [1], [1]),
    )
    graph = helper.make_graph(
        [node],
        "constantofshape_initializer_shape_test",
        inputs=[helper.make_tensor_value_info("shape", TensorProto.INT64, [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT64, [3])],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, producer_name="constantofshape_initializer_shape_test")

    tvm_model = from_onnx(model, keep_params_in_input=True)

    assert tuple(dim.value for dim in tvm_model["main"].ret_ty.shape.values) == (3,)
    assert tvm_model["main"].ret_ty.dtype == "int64"


def test_slice():
    def verify_slice(data_shape, output_shape, starts, ends, expected, axes=None, steps=None):
        if isinstance(starts, list):
            starts = np.array(starts, "int64")
        if isinstance(ends, list):
            ends = np.array(ends, "int64")
        if isinstance(axes, list):
            axes = np.array(axes, "int64")
        if isinstance(steps, list):
            steps = np.array(steps, "int64")

        slice_inputs = ["x", "starts", "ends"]
        initializer = [
            helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
            helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
        ]

        if axes is not None:
            initializer.append(helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes))
            slice_inputs.append("axes")
        if steps is not None:
            initializer.append(helper.make_tensor("steps", TensorProto.INT64, steps.shape, steps))
            slice_inputs.append("steps")

        slice_node = helper.make_node("Slice", inputs=slice_inputs, outputs=["y"])

        graph = helper.make_graph(
            [slice_node],
            "slice_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializer=initializer,
        )

        model = helper.make_model(graph, producer_name="slice_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedSliceAxesAndSteps:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((2,), dtype="int64"),
            ends: R.Tensor((2,), dtype="int64"),
            axes: R.Tensor((2,), dtype="int64"),
            steps: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((3, 10, 5), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 10, 5), dtype="float32") = R.strided_slice(
                    x,
                    axes=[0, 1],
                    begin=[0, 0],
                    end=[3, 10],
                    strides=[1, 1],
                    assume_inbound=False,
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSliceDefaultAxesAndSteps:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((2,), dtype="int64"),
            ends: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((3, 10, 5), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 10, 5), dtype="float32") = R.strided_slice(
                    x,
                    axes=[0, 1],
                    begin=[0, 0],
                    end=[3, 10],
                    strides=[1, 1],
                    assume_inbound=False,
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSliceNegativeSteps:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((3,), dtype="int64"),
            ends: R.Tensor((3,), dtype="int64"),
            axes: R.Tensor((3,), dtype="int64"),
            steps: R.Tensor((3,), dtype="int64"),
        ) -> R.Tensor((19, 3, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((19, 3, 2), dtype="float32") = R.strided_slice(
                    x,
                    axes=[0, 1, 2],
                    begin=[20, 10, 4],
                    end=[0, 0, 1],
                    strides=[-1, -3, -2],
                    assume_inbound=False,
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSliceAxesOnly:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((2,), dtype="int64"),
            ends: R.Tensor((2,), dtype="int64"),
            axes: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((20, 3, 5), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((20, 3, 5), dtype="float32") = R.strided_slice(
                    x,
                    axes=[1, 2],
                    begin=[0, 0],
                    end=[3, 10],
                    strides=[1, 1],
                    assume_inbound=False,
                )
                R.output(gv)
            return gv

    # Test with all parameters set.
    verify_slice(
        [20, 10, 5],
        [3, 10, 5],
        starts=[0, 0],
        ends=[3, 10],
        axes=[0, 1],
        steps=[1, 1],
        expected=ExpectedSliceAxesAndSteps,
    )
    # Test with default axes and steps.
    verify_slice(
        [20, 10, 5],
        [3, 10, 5],
        starts=[0, 0],
        ends=[3, 10],
        expected=ExpectedSliceDefaultAxesAndSteps,
    )
    # Test with negative steps.
    verify_slice(
        [20, 10, 5],
        [19, 3, 2],
        starts=[20, 10, 4],  # NOTE: the start is out of bounds
        ends=[0, 0, 1],
        steps=[-1, -3, -2],
        axes=[0, 1, 2],
        expected=ExpectedSliceNegativeSteps,
    )
    verify_slice(
        [20, 10, 5],
        [10, 5],
        starts=[0, 0],
        ends=[3, 10],
        axes=[1, 2],
        expected=ExpectedSliceAxesOnly,
    )
    verify_slice(
        [20, 10, 5],
        [10, 5],
        starts=[0, 0],
        ends=[3, 10],
        axes=[1, 2],
        expected=ExpectedSliceAxesOnly,
    )

    # TODO (gigiblender): Enable this test when we have a way to pass the steps but not axes.
    # verify_slice(
    #     [20, 10, 5],
    #     [19, 3, 2],
    #     starts=[20, 10, 4],
    #     ends=[0, 0, 1],
    #     steps=[-1, -3, -2],
    # )


def test_slice_dynamic_inputs_ir():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_dynamic_inputs_ir",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_inputs_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((2,), dtype="int64"),
            ends: R.Tensor((2,), dtype="int64"),
            axes: R.Tensor((2,), dtype="int64"),
            steps: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=3):
            R.func_attr({"num_input": 5})
            with R.dataflow():
                lv: R.Tensor((2,), dtype="bool") = R.less(axes, R.const(0, "int64"))
                lv1: R.Tensor((2,), dtype="int64") = R.add(axes, R.const(3, "int64"))
                lv2: R.Shape([20, 10, 5]) = R.shape_of(x)
                lv3: R.Tensor((2,), dtype="int64") = R.where(lv, lv1, axes)
                lv4: R.Tensor((3,), dtype="int64") = R.shape_to_tensor(lv2)
                lv5: R.Tensor((3,), dtype="int64") = R.scatter_elements(
                    R.const([0, 0, 0], "int64"), lv3, starts, axis=0, reduction="update"
                )
                lv6: R.Tensor((3,), dtype="int64") = R.scatter_elements(
                    lv4, lv3, ends, axis=0, reduction="update"
                )
                lv7: R.Tensor((3,), dtype="int64") = R.scatter_elements(
                    R.const([1, 1, 1], "int64"), lv3, steps, axis=0, reduction="update"
                )
                gv: R.Tensor(dtype="float32", ndim=3) = R.dynamic_strided_slice(x, lv5, lv6, lv7)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_slice_dynamic_inputs_length_validation():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_dynamic_inputs_length_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_inputs_length_validation_test")
    with pytest.raises(ValueError, match="starts and ends to have the same length"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_slice_dynamic_shape_expr_input_validation():
    shape_node = helper.make_node("Shape", ["x"], ["y"])
    slice_node = helper.make_node("Slice", ["y", "starts", "ends", "axes", "steps"], ["z"])

    graph = helper.make_graph(
        [shape_node, slice_node],
        "slice_dynamic_shape_expr_input_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [1]),
        ],
        outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, [1])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_shape_expr_input_validation_test")
    with pytest.raises(ValueError, match="does not support ShapeExpr input"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_slice_zero_step_validation():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_zero_step_validation",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5])],
        initializer=[
            helper.make_tensor("starts", TensorProto.INT64, [2], vals=[0, 0]),
            helper.make_tensor("ends", TensorProto.INT64, [2], vals=[3, 10]),
            helper.make_tensor("axes", TensorProto.INT64, [2], vals=[0, 1]),
            helper.make_tensor("steps", TensorProto.INT64, [2], vals=[1, 0]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_zero_step_validation_test")
    with pytest.raises(ValueError, match="step values must be non-zero"):
        from_onnx(model, opset=13)


def test_slice_dynamic_shape():
    def verify_slice(data_shape, output_shape, starts, ends, axes, expected):
        if isinstance(starts, list):
            starts = np.array(starts, "int64")
        if isinstance(ends, list):
            ends = np.array(ends, "int64")
        if isinstance(axes, list):
            axes = np.array(axes, "int64")

        slice_inputs = ["y", "starts", "ends"]
        initializer = [
            helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
            helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
            helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes),
        ]
        slice_inputs.append("axes")

        shape_node = helper.make_node("Shape", inputs=["x"], outputs=["y"])
        slice_node = helper.make_node("Slice", inputs=slice_inputs, outputs=["z"])

        graph = helper.make_graph(
            [shape_node, slice_node],
            "slice_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape),
            ],
            outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, output_shape)],
            initializer=initializer,
        )

        model = helper.make_model(graph, producer_name="slice_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        assert len(tvm_model["main"].attrs["params"]) == 3
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedShapeSlice0:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((2,), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="int64") = R.const([20, 10], "int64")
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")

    @I.ir_module
    class ExpectedShapeSlice1:
        @R.function
        def main(
            x: R.Tensor((A, 10, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape(ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([A, 10]) = R.shape([A, 10])
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class ExpectedShapeSlice2:
        @R.function
        def main(
            x: R.Tensor((A, B, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape(ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([A, B]) = R.shape([A, B])
                R.output(gv)
            return gv

    C = T.dynamic("C", "int64")

    @I.ir_module
    class ExpectedShapeSlice3:
        @R.function
        def main(
            x: R.Tensor((20, 10, C), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((2,), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="int64") = R.const([20, 10], "int64")
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")
    C = T.dynamic("C", "int64")

    @I.ir_module
    class ExpectedShapeSlice4:
        @R.function
        def main(
            x: R.Tensor((A, B, C), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape(ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([A, B]) = R.shape([A, B])
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedShapeSlice5:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((1,), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1,), dtype="int64") = R.const([10], "int64")
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")

    @I.ir_module
    class ExpectedShapeSlice6:
        @R.function
        def main(
            x: R.Tensor((A, 10, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((1,), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1,), dtype="int64") = R.const([10], "int64")
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class ExpectedShapeSlice7:
        @R.function
        def main(
            x: R.Tensor((A, B, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape(ndim=1):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([B]) = R.shape([B])
                R.output(gv)
            return gv

    C = T.dynamic("C", "int64")

    @I.ir_module
    class ExpectedShapeSlice8:
        @R.function
        def main(
            x: R.Tensor((20, 10, C), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((1,), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1,), dtype="int64") = R.const([10], "int64")
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")
    C = T.dynamic("C", "int64")

    @I.ir_module
    class ExpectedShapeSlice9:
        @R.function
        def main(
            x: R.Tensor((A, B, C), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape(ndim=1):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([B]) = R.shape([B])
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedShapeSlice10:
        @R.function
        def main(
            x: R.Tensor((20, 10, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((2,), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="int64") = R.const([10, 5], "int64")
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")

    @I.ir_module
    class ExpectedShapeSlice11:
        @R.function
        def main(
            x: R.Tensor((A, 10, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((2,), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="int64") = R.const([10, 5], "int64")
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class ExpectedShapeSlice12:
        @R.function
        def main(
            x: R.Tensor((A, B, 5), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape(ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([B, 5]) = R.shape([B, 5])
                R.output(gv)
            return gv

    C = T.dynamic("C", "int64")

    @I.ir_module
    class ExpectedShapeSlice13:
        @R.function
        def main(
            x: R.Tensor((20, 10, C), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape(ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([10, C]) = R.shape([10, C])
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")
    C = T.dynamic("C", "int64")

    @I.ir_module
    class ExpectedShapeSlice14:
        @R.function
        def main(
            x: R.Tensor((A, B, C), dtype="float32"),
            starts: R.Tensor((1,), dtype="int64"),
            ends: R.Tensor((1,), dtype="int64"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Shape(ndim=2):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Shape([B, C]) = R.shape([B, C])
                R.output(gv)
            return gv

    verify_slice([20, 10, 5], [2], starts=[0], ends=[2], axes=[0], expected=ExpectedShapeSlice0)
    verify_slice(["A", 10, 5], [2], starts=[0], ends=[2], axes=[0], expected=ExpectedShapeSlice1)
    verify_slice(["A", "B", 5], [2], starts=[0], ends=[2], axes=[0], expected=ExpectedShapeSlice2)
    verify_slice([20, 10, "C"], [2], starts=[0], ends=[2], axes=[0], expected=ExpectedShapeSlice3)
    verify_slice(["A", "B", "C"], [2], starts=[0], ends=[2], axes=[0], expected=ExpectedShapeSlice4)
    verify_slice([20, 10, 5], [1], starts=[1], ends=[2], axes=[0], expected=ExpectedShapeSlice5)
    verify_slice(["A", 10, 5], [1], starts=[1], ends=[2], axes=[0], expected=ExpectedShapeSlice6)
    verify_slice(["A", "B", 5], [1], starts=[1], ends=[2], axes=[0], expected=ExpectedShapeSlice7)
    verify_slice([20, 10, "C"], [1], starts=[1], ends=[2], axes=[0], expected=ExpectedShapeSlice8)
    verify_slice(["A", "B", "C"], [1], starts=[1], ends=[2], axes=[0], expected=ExpectedShapeSlice9)
    verify_slice([20, 10, 5], [2], starts=[1], ends=[3], axes=[0], expected=ExpectedShapeSlice10)
    verify_slice(["A", 10, 5], [2], starts=[1], ends=[3], axes=[0], expected=ExpectedShapeSlice11)
    verify_slice(["A", "B", 5], [2], starts=[1], ends=[3], axes=[0], expected=ExpectedShapeSlice12)
    verify_slice([20, 10, "C"], [2], starts=[1], ends=[3], axes=[0], expected=ExpectedShapeSlice13)
    verify_slice(
        ["A", "B", "C"], [2], starts=[1], ends=[3], axes=[0], expected=ExpectedShapeSlice14
    )


# TODO Enable dynamism
@pytest.mark.parametrize("dynamic", [False])
def test_attention(dynamic):
    def verify_attention(
        input_,
        weight,
        bias,
        mask_index,
        num_heads,
        mask_filter_value,
        qkv_hidden_sizes,
        relative_position_bias,
    ):
        node = onnx.helper.make_node(
            "Attention",
            inputs=["input", "weight", "bias", "mask_index", "", "relative_position_bias"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
            # TODO(jwfromm) OnnxRT doesnt work with this attribute, figure out why not.
            # mask_filter_value=mask_filter_value,
            qkv_hidden_sizes=qkv_hidden_sizes,
        )

        input_shape = list(input_.shape)
        weight_shape = list(weight.shape)
        bias_shape = list(bias.shape)
        mask_shape = list(mask_index.shape)
        relative_position_bias_shape = list(relative_position_bias.shape)
        output_shape = list(input_.shape)
        if dynamic:
            input_shape = ["?" for _ in range(len(input_.shape))]
            weight_shape = ["?" for _ in range(len(weight.shape))]
            bias_shape = ["?" for _ in range(len(bias.shape))]
            mask_shape = ["?" for _ in range(len(mask_index.shape))]
            output_shape = ["?" for _ in range(len(input_.shape))]

        graph = helper.make_graph(
            [node],
            "attention_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("weight", TensorProto.FLOAT, weight_shape),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias_shape),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, mask_shape),
                helper.make_tensor_value_info(
                    "relative_position_bias", TensorProto.FLOAT, relative_position_bias_shape
                ),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
            ],
        )

        model = helper.make_model(graph, producer_name="attention_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)

        @I.ir_module
        class ExpectedAttention:
            @R.function
            def main(
                input: R.Tensor((4, 4, 128), dtype="float32"),
                weight: R.Tensor((128, 480), dtype="float32"),
                bias: R.Tensor((480,), dtype="float32"),
                mask_index: R.Tensor((4, 4), dtype="int32"),
                relative_position_bias: R.Tensor((4, 12, 4, 4), dtype="float32"),
            ) -> R.Tensor((4, 4, 96), dtype="float32"):
                R.func_attr({"num_input": 5})
                with R.dataflow():
                    lv: R.Tensor((4, 4), dtype="int32") = R.subtract(
                        R.const(1, "int32"), mask_index
                    )
                    lv1: R.Tensor((4, 4), dtype="float32") = R.astype(lv, dtype="float32")
                    lv2: R.Tensor((4, 4), dtype="float32") = R.multiply(
                        lv1, R.const(-10000.0, "float32")
                    )
                    lv3: R.Tensor((4, 1, 1, 4), dtype="float32") = R.reshape(
                        lv2, R.shape([4, 1, 1, 4])
                    )
                    lv4: R.Tensor((4, 4, 480), dtype="float32") = R.matmul(input, weight)
                    lv5: R.Tensor((4, 4, 480), dtype="float32") = R.add(lv4, bias)
                    lv6: R.Tuple(
                        R.Tensor((4, 4, 192), dtype="float32"),
                        R.Tensor((4, 4, 192), dtype="float32"),
                        R.Tensor((4, 4, 96), dtype="float32"),
                    ) = R.split(lv5, indices_or_sections=[192, 384], axis=2)
                    lv7: R.Tensor((4, 4, 192), dtype="float32") = lv6[0]
                    lv8: R.Tensor((4, 4, 192), dtype="float32") = lv6[1]
                    lv9: R.Tensor((4, 4, 96), dtype="float32") = lv6[2]
                    lv10: R.Tensor((4, 4, 12, 16), dtype="float32") = R.reshape(
                        lv7, R.shape([4, 4, 12, 16])
                    )
                    lv11: R.Tensor((4, 4, 12, 16), dtype="float32") = R.reshape(
                        lv8, R.shape([4, 4, 12, 16])
                    )
                    lv12: R.Tensor((4, 4, 12, 8), dtype="float32") = R.reshape(
                        lv9, R.shape([4, 4, 12, 8])
                    )
                    lv13: R.Tensor((4, 12, 4, 4), dtype="float32") = R.add(
                        relative_position_bias, lv3
                    )
                    lv14: R.Tensor((4, 4, 12, 8), dtype="float32") = R.nn.attention(
                        lv10, lv11, lv12, lv13
                    )
                    lv15: R.Tensor((4, 4, 96), dtype="float32") = R.reshape(
                        lv14, R.shape([4, 4, 96])
                    )
                    gv: R.Tensor((4, 4, 96), dtype="float32") = lv15
                    R.output(gv)
                return gv

        tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedAttention)
        # "present" output should be nullptr when the "past" input isn't included,
        # but ort requires an output shape to be specified?
        # verify_with_ort_with_inputs(
        #     model,
        #     [input_, weight, bias, mask_index],
        #     [input_.shape, present_output_shape],
        #     target=target,
        #     dev=dev,
        #     rtol=1e-4,
        #     atol=1e-4,
        # )

    input_hidden_size = 128
    batch_size = 4
    sequence_length = 4
    num_heads = 12
    qkv_hidden_sizes = [192, 192, 96]
    mask_filter_value = -512.0

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, input_hidden_size)).astype(dtype)
    weight = np.random.normal(size=(input_hidden_size, sum(qkv_hidden_sizes))).astype(dtype) * 0.1
    bias = np.random.randn(sum(qkv_hidden_sizes)).astype(dtype)
    mask_index = np.random.randint(2, size=(batch_size, sequence_length)).astype("int32")
    relative_position_bias = np.random.randn(
        batch_size, num_heads, sequence_length, sequence_length
    ).astype(dtype)

    verify_attention(
        input_array,
        weight,
        bias,
        mask_index,
        num_heads,
        mask_filter_value,
        qkv_hidden_sizes,
        relative_position_bias,
    )


def _make_pad_expected_ir(input_shape, pads, mode="constant", value=0.0, opset=14, axes=None):
    len_dim = len(pads) // 2
    np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]

    if axes is not None:
        rank = len(input_shape)
        full_pads = [(0, 0)] * rank
        for i, axis in enumerate(axes):
            axis = axis if axis >= 0 else axis + rank
            full_pads[axis] = np_pads[i]
        np_pads = full_pads

    if mode == "constant":
        out_shape = np.pad(
            np.empty(input_shape, dtype=np.float32),
            pad_width=np_pads,
            mode="constant",
            constant_values=value,
        ).shape
    else:
        out_shape = np.pad(
            np.empty(input_shape, dtype=np.float32), pad_width=np_pads, mode=mode
        ).shape
    input_shape = tuple(input_shape)
    out_shape = tuple(out_shape)
    pads_shape = (len(pads),)
    axes_shape = None if axes is None else (len(axes),)

    if mode == "constant" and opset >= 11:

        @I.ir_module
        class ExpectedPadConstantWithInputs:
            @Ts.function(private=True)
            def pad(
                input: T.Tensor(input_shape, "float32"), PadInput: T.Tensor(out_shape, "float32")
            ):
                T.evaluate(0)

            @R.function
            def main(
                input: R.Tensor(input_shape, dtype="float32"),
                pads: R.Tensor(pads_shape, dtype="int64"),
                constant_value: R.Tensor((1,), dtype="float32"),
            ) -> R.Tensor(out_shape, dtype="float32"):
                R.func_attr({"num_input": 1})
                cls = ExpectedPadConstantWithInputs
                with R.dataflow():
                    lv = R.call_tir(
                        cls.pad,
                        (input,),
                        ty_args=[R.Tensor(out_shape, dtype="float32")],
                    )
                    gv: R.Tensor(out_shape, dtype="float32") = lv
                    R.output(gv)
                return gv

        return ExpectedPadConstantWithInputs

    if mode == "constant":

        @I.ir_module
        class ExpectedPadConstantAttrs:
            @Ts.function(private=True)
            def pad(
                input: T.Tensor(input_shape, "float32"), PadInput: T.Tensor(out_shape, "float32")
            ):
                T.evaluate(0)

            @R.function
            def main(input: R.Tensor(input_shape, dtype="float32")) -> R.Tensor(
                out_shape, dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                cls = ExpectedPadConstantAttrs
                with R.dataflow():
                    lv = R.call_tir(
                        cls.pad,
                        (input,),
                        ty_args=[R.Tensor(out_shape, dtype="float32")],
                    )
                    gv: R.Tensor(out_shape, dtype="float32") = lv
                    R.output(gv)
                return gv

        return ExpectedPadConstantAttrs

    if mode == "reflect" and opset >= 11:

        @I.ir_module
        class ExpectedPadReflectWithInputs:
            @Ts.function(private=True)
            def mirror_pad(
                input: T.Tensor(input_shape, "float32"),
                MirrorPadInput: T.Tensor(out_shape, "float32"),
            ):
                T.evaluate(0)

            @R.function
            def main(
                input: R.Tensor(input_shape, dtype="float32"),
                pads: R.Tensor(pads_shape, dtype="int64"),
            ) -> R.Tensor(out_shape, dtype="float32"):
                R.func_attr({"num_input": 1})
                cls = ExpectedPadReflectWithInputs
                with R.dataflow():
                    lv = R.call_tir(
                        cls.mirror_pad,
                        (input,),
                        ty_args=[R.Tensor(out_shape, dtype="float32")],
                    )
                    gv: R.Tensor(out_shape, dtype="float32") = lv
                    R.output(gv)
                return gv

        return ExpectedPadReflectWithInputs

    if mode == "reflect":

        @I.ir_module
        class ExpectedPadReflectAttrs:
            @Ts.function(private=True)
            def mirror_pad(
                input: T.Tensor(input_shape, "float32"),
                MirrorPadInput: T.Tensor(out_shape, "float32"),
            ):
                T.evaluate(0)

            @R.function
            def main(input: R.Tensor(input_shape, dtype="float32")) -> R.Tensor(
                out_shape, dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                cls = ExpectedPadReflectAttrs
                with R.dataflow():
                    lv = R.call_tir(
                        cls.mirror_pad,
                        (input,),
                        ty_args=[R.Tensor(out_shape, dtype="float32")],
                    )
                    gv: R.Tensor(out_shape, dtype="float32") = lv
                    R.output(gv)
                return gv

        return ExpectedPadReflectAttrs

    if mode == "edge" and opset >= 11:

        @I.ir_module
        class ExpectedPadEdgeWithInputs:
            @Ts.function(private=True)
            def replicate_pad(
                input: T.Tensor(input_shape, "float32"),
                ReplicatePadInput: T.Tensor(out_shape, "float32"),
            ):
                T.evaluate(0)

            @R.function
            def main(
                input: R.Tensor(input_shape, dtype="float32"),
                pads: R.Tensor(pads_shape, dtype="int64"),
            ) -> R.Tensor(out_shape, dtype="float32"):
                R.func_attr({"num_input": 1})
                cls = ExpectedPadEdgeWithInputs
                with R.dataflow():
                    lv = R.call_tir(
                        cls.replicate_pad,
                        (input,),
                        ty_args=[R.Tensor(out_shape, dtype="float32")],
                    )
                    gv: R.Tensor(out_shape, dtype="float32") = lv
                    R.output(gv)
                return gv

        return ExpectedPadEdgeWithInputs

    if mode == "edge":

        @I.ir_module
        class ExpectedPadEdgeAttrs:
            @Ts.function(private=True)
            def replicate_pad(
                input: T.Tensor(input_shape, "float32"),
                ReplicatePadInput: T.Tensor(out_shape, "float32"),
            ):
                T.evaluate(0)

            @R.function
            def main(input: R.Tensor(input_shape, dtype="float32")) -> R.Tensor(
                out_shape, dtype="float32"
            ):
                R.func_attr({"num_input": 1})
                cls = ExpectedPadEdgeAttrs
                with R.dataflow():
                    lv = R.call_tir(
                        cls.replicate_pad,
                        (input,),
                        ty_args=[R.Tensor(out_shape, dtype="float32")],
                    )
                    gv: R.Tensor(out_shape, dtype="float32") = lv
                    R.output(gv)
                return gv

        return ExpectedPadEdgeAttrs

    if mode == "wrap" and opset >= 19:
        if axes is None:

            @I.ir_module
            class ExpectedPadWrapWithInputs:
                @Ts.function(private=True)
                def circular_pad(
                    input: T.Tensor(input_shape, "float32"),
                    CircularPadInput: T.Tensor(out_shape, "float32"),
                ):
                    T.evaluate(0)

                @R.function
                def main(
                    input: R.Tensor(input_shape, dtype="float32"),
                    pads: R.Tensor(pads_shape, dtype="int64"),
                ) -> R.Tensor(out_shape, dtype="float32"):
                    R.func_attr({"num_input": 1})
                    cls = ExpectedPadWrapWithInputs
                    with R.dataflow():
                        lv = R.call_tir(
                            cls.circular_pad,
                            (input,),
                            ty_args=[R.Tensor(out_shape, dtype="float32")],
                        )
                        gv: R.Tensor(out_shape, dtype="float32") = lv
                        R.output(gv)
                    return gv

            return ExpectedPadWrapWithInputs

        @I.ir_module
        class ExpectedPadWrapWithAxes:
            @Ts.function(private=True)
            def circular_pad(
                input: T.Tensor(input_shape, "float32"),
                CircularPadInput: T.Tensor(out_shape, "float32"),
            ):
                T.evaluate(0)

            @R.function
            def main(
                input: R.Tensor(input_shape, dtype="float32"),
                pads: R.Tensor(pads_shape, dtype="int64"),
                axes: R.Tensor(axes_shape, dtype="int64"),
            ) -> R.Tensor(out_shape, dtype="float32"):
                R.func_attr({"num_input": 1})
                cls = ExpectedPadWrapWithAxes
                with R.dataflow():
                    lv = R.call_tir(
                        cls.circular_pad,
                        (input,),
                        ty_args=[R.Tensor(out_shape, dtype="float32")],
                    )
                    gv: R.Tensor(out_shape, dtype="float32") = lv
                    R.output(gv)
                return gv

        return ExpectedPadWrapWithAxes

    raise AssertionError(f"No Pad expected IR for mode={mode}, opset={opset}")


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad(dynamic):
    if dynamic:
        pytest.skip("Dynamic pad not supported")

    def verify_pad(input_shape, pads, expected, mode="constant", value=0.0, opset=14, axes=None):
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]

        if axes is not None:
            rank = len(input_shape)
            full_pads = [(0, 0)] * rank
            for i, axis in enumerate(axes):
                axis = axis if axis >= 0 else axis + rank
                full_pads[axis] = np_pads[i]
            np_pads = full_pads

        pads = np.array(pads, dtype=np.int64)
        #  onnx graph
        if mode in ["edge", "reflect", "wrap"]:
            outdata = np.pad(np.empty(input_shape, dtype=np.float32), pad_width=np_pads, mode=mode)

            node_inputs = ["input", "pads"]
            initializer = [helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)]

            if axes is not None:
                axes = np.array(axes, dtype=np.int64)
                node_inputs = ["input", "pads", "", "axes"]
                initializer.append(
                    helper.make_tensor("axes", TensorProto.INT64, (len(axes),), axes)
                )

            node = helper.make_node("Pad", inputs=node_inputs, outputs=["output"], mode=mode)
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                initializer=initializer,
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        else:
            outdata = np.pad(
                np.empty(input_shape, dtype=np.float32),
                pad_width=np_pads,
                mode="constant",
                constant_values=value,
            )
            node = helper.make_node(
                "Pad",
                inputs=["input", "pads", "constant_value"],
                outputs=["output"],
                mode="constant",
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                initializer=[
                    helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
                    helper.make_tensor("constant_value", TensorProto.FLOAT, (1,), [value]),
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        model.opset_import[0].version = opset
        tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        expected = tvm_tirx.IRModule(expected.functions)
        for gv in expected.get_global_vars():
            if gv.name_hint != "main":
                expected.update_func(gv, tvm_model[gv.name_hint])
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    for input_shape, pads, mode, value, opset, axes in [
        ((2, 2), [0, 1, 0, 0], "constant", 0.0, 14, None),
        ((2, 3), [1, 0, 0, 1], "constant", 0.0, 14, None),
        ((3, 2), [0, 0, 1, 0], "constant", 5.0, 14, None),
        ((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "reflect", 0.0, 14, None),
        ((2, 3), [1, 1, 1, 1], "edge", 0.0, 14, None),
        ((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "edge", 0.0, 14, None),
        ((1, 3, 4), [0, 0, 2, 0, 0, 2], "wrap", 0.0, 19, None),
        ((1, 3, 4), [2, 2], "wrap", 0.0, 19, [2]),
        ((1, 3, 4), [1, 2, 1, 2], "wrap", 0.0, 19, [1, 2]),
    ]:
        verify_pad(
            input_shape,
            pads,
            _make_pad_expected_ir(
                input_shape, pads, mode=mode, value=value, opset=opset, axes=axes
            ),
            mode,
            value,
            opset,
            axes,
        )


def test_pad_opset18_axes():
    """Pad-18 introduced the optional axes input but not mode="wrap" (Pad-19 only).
    opset-18 axes must lower correctly for constant/reflect/edge modes (matching
    onnxruntime), and mode="wrap" must be rejected for opset 18."""

    def make_model(input_shape, pads, axes, mode, opset=18):
        node_inputs = ["x", "pads"]
        initializer = [
            helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
        ]
        if axes is not None:
            node_inputs += ["", "axes"]
            initializer.append(helper.make_tensor("axes", TensorProto.INT64, (len(axes),), axes))
        node = helper.make_node("Pad", inputs=node_inputs, outputs=["y"], mode=mode)
        graph = helper.make_graph(
            [node],
            "pad_opset18_axes",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(input_shape))],
            initializer=initializer,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    # opset-18 axes work for constant/reflect/edge modes.
    for shape, pads, axes, mode in [
        ((1, 3, 4), [1, 2], [1], "constant"),
        ((1, 3, 4), [1, 2], [1], "reflect"),
        ((1, 3, 4), [2, 1], [2], "edge"),
        ((2, 3, 4), [0, 1, 0, 1], [0, 2], "reflect"),
        ((1, 3, 4), [1, 2], [-1], "reflect"),
    ]:
        model = make_model(shape, pads, axes, mode, opset=18)
        inputs = {"x": np.arange(np.prod(shape), dtype="float32").reshape(shape) + 1.0}
        check_correctness(model, inputs=inputs, opset=18)

    # mode="wrap" was only added in Pad-19 and must be rejected for opset 18.
    model = make_model((1, 3, 4), [2, 2], [2], "wrap", opset=18)
    with pytest.raises(AttributeError):
        from_onnx(model, opset=18, keep_params_in_input=True)


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad_v2(dynamic):
    if dynamic:
        pytest.skip("Dynamic pad not supported")

    def verify_pad(input_shape, pads, expected, mode="constant", value=0.0):
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        pads = np.array(pads)
        #  onnx graph
        if mode in ["edge", "reflect"]:
            outdata = np.pad(np.empty(input_shape, dtype=np.float32), pad_width=np_pads, mode=mode)
            node = helper.make_node(
                "Pad", inputs=["input"], outputs=["output"], mode=mode, pads=pads
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        else:
            outdata = np.pad(
                np.empty(input_shape, dtype=np.float32),
                pad_width=np_pads,
                mode="constant",
                constant_values=value,
            )
            node = helper.make_node(
                "Pad",
                inputs=["input"],
                outputs=["output"],
                mode="constant",
                pads=pads,
                value=value,
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        model.opset_import[0].version = 10
        tvm_model = from_onnx(model, opset=10, keep_params_in_input=True)
        expected = tvm_tirx.IRModule(expected.functions)
        for gv in expected.get_global_vars():
            if gv.name_hint != "main":
                expected.update_func(gv, tvm_model[gv.name_hint])
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    for input_shape, pads, mode, value in [
        ((2, 2), [0, 1, 0, 0], "constant", 0.0),
        ((2, 3), [1, 0, 0, 1], "constant", 0.0),
        ((3, 2), [0, 0, 1, 0], "constant", 5.0),
        ((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "reflect", 0.0),
        ((2, 3), [1, 1, 1, 1], "edge", 0.0),
        ((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "edge", 0.0),
    ]:
        verify_pad(
            input_shape,
            pads,
            _make_pad_expected_ir(input_shape, pads, mode=mode, value=value, opset=10),
            mode,
            value,
        )


def test_split():
    def verify_split(
        fp_arith,
        dynamic,
        indata_shape,
        outdata_shapes,
        split,
        expected,
        axis=0,
        pass_split=True,
        opset=11,
    ):
        indata = np.random.normal(size=indata_shape).astype(fp_arith)
        input_names = ["input"]
        initializer = []

        if split:
            split_index = range(len(split))
        else:
            split_index = range(len(outdata_shapes))

        indata_shape = list(indata.shape)
        if dynamic:
            indata_shape = ["?" for _ in range(len(indata.shape))]
            outdata_shapes = [["?" for _ in range(len(o))] for o in outdata_shapes]

        inputs = [
            helper.make_tensor_value_info(
                "input", helper.np_dtype_to_tensor_dtype(indata.dtype), indata_shape
            )
        ]

        split_constant = None
        if pass_split:
            if opset >= 13:
                np_split = np.array(split).astype(np.int64)
                split_constant = make_constant_node(
                    "split", onnx.TensorProto.INT64, list(np_split.shape), np_split
                )
                input_names.append("split")

        node = helper.make_node(
            "Split",
            inputs=input_names,
            outputs=[f"output_{i}" for i in range(len(split_index))],
            axis=axis,
        )

        if pass_split and opset < 13:
            split_attr = helper.make_attribute("split", split)
            node.attribute.append(split_attr)

        nodes = [split_constant, node] if split_constant else [node]

        graph = helper.make_graph(
            nodes,
            "split_test",
            inputs=inputs,
            initializer=initializer,
            outputs=[
                helper.make_tensor_value_info(
                    f"output_{i}",
                    helper.np_dtype_to_tensor_dtype(indata.dtype),
                    list(outdata_shapes[i]),
                )
                for i in range(len(split_index))
            ],
        )
        model = helper.make_model(graph, producer_name="split_test")
        tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    def make_expected(fp_arith, dynamic, indata_shape, outdata_shapes, split, axis, pass_split):
        def shape_tuple(shape):
            if isinstance(shape, int):
                shape = (shape,)
            return tuple(shape)

        def expected_input_shape(shape):
            shape = shape_tuple(shape)
            if not dynamic:
                return shape
            return tuple(tirx.Var(f"split_input_dim_{i}", "int64") for i in range(len(shape)))

        dtype = np.dtype(fp_arith).name
        input_shape = expected_input_shape(indata_shape)
        if not pass_split:
            indices_or_sections = len(outdata_shapes)
        elif len(outdata_shapes) == 1:
            indices_or_sections = 1
        else:
            indices_or_sections = list(np.cumsum(split)[:-1])

        if len(outdata_shapes) == 1:

            @I.ir_module
            class ExpectedSplitSingle:
                @R.function
                def main(input: R.Tensor(input_shape, dtype=dtype)):
                    R.func_attr({"num_input": 1})
                    with R.dataflow():
                        gv = R.split(input, indices_or_sections=indices_or_sections, axis=axis)
                        R.output(gv)
                    return gv

            return ExpectedSplitSingle

        if len(outdata_shapes) == 2:

            @I.ir_module
            class ExpectedSplitPair:
                @R.function
                def main(input: R.Tensor(input_shape, dtype=dtype)):
                    R.func_attr({"num_input": 1})
                    with R.dataflow():
                        lv = R.split(input, indices_or_sections=indices_or_sections, axis=axis)
                        lv1 = lv[0]
                        lv2 = lv[1]
                        gv = (lv1, lv2)
                        R.output(gv)
                    return gv

            return ExpectedSplitPair

        assert len(outdata_shapes) == 3

        @I.ir_module
        class ExpectedSplitTriple:
            @R.function
            def main(input: R.Tensor(input_shape, dtype=dtype)):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv = R.split(input, indices_or_sections=indices_or_sections, axis=axis)
                    lv1 = lv[0]
                    lv2 = lv[1]
                    lv3 = lv[2]
                    gv = (lv1, lv2, lv3)
                    R.output(gv)
                return gv

        return ExpectedSplitTriple

    split_cases = [
        (6, [[2], [2], [2]], [2, 2, 2], 0, True, 11),
        (6, [[2], [2], [2]], [2, 2, 2], 0, False, 11),
        (6, [[2], [1], [3]], [2, 1, 3], 0, True, 11),
        (6, [[2], [1], [3]], [2, 1, 3], 0, True, 13),
        ((4, 4), [[2, 2], [2, 2]], [2, 2], 1, True, 11),
        ((4, 4), [[2, 2], [2, 2]], [2, 2], 1, True, 13),
        (3, [[1], [1], [1]], False, 0, False, 11),
        (1, [[1]], [1], 0, True, 11),
        ((1, 2), [[2]], [2], 1, True, 11),
        ((1, 2), [[2]], [1], 0, True, 11),
    ]

    for fp_arith in [np.float16, np.float32]:
        for dynamic in [True, False]:
            for indata_shape, outdata_shapes, split, axis, pass_split, opset in split_cases:
                verify_split(
                    fp_arith,
                    dynamic,
                    indata_shape,
                    outdata_shapes,
                    split,
                    make_expected(
                        fp_arith, dynamic, indata_shape, outdata_shapes, split, axis, pass_split
                    ),
                    axis=axis,
                    pass_split=pass_split,
                    opset=opset,
                )


def test_split_initializer_with_params_in_input():
    split_sizes = np.array([2, 4], dtype="int64")
    split_node = helper.make_node(
        "Split",
        ["data", "split_sizes"],
        ["left", "right"],
        axis=0,
    )
    graph = helper.make_graph(
        [split_node],
        "split_initializer_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, [6])],
        initializer=[numpy_helper.from_array(split_sizes, name="split_sizes")],
        outputs=[
            helper.make_tensor_value_info("left", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("right", TensorProto.FLOAT, [4]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="split_initializer_test",
        opset_imports=[helper.make_opsetid("", 13)],
    )

    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    np.testing.assert_array_equal(tvm_model["main"].attrs["params"][0].numpy(), split_sizes)
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((6,), dtype="float32"),
            split_sizes: R.Tensor((2,), dtype="int64"),
        ) -> R.Tuple(
            R.Tensor((2,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv = R.split(data, indices_or_sections=[2], axis=0)
                lv1 = lv[0]
                lv2 = lv[1]
                gv = (lv1, lv2)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_tile():
    def verify_tile(dynamic, in_shape, repeats, out_shape, expected):
        node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])

        model_in_shape = list(in_shape)
        model_out_shape = list(out_shape)
        if dynamic:
            model_in_shape = ["?" for _ in range(len(in_shape))]
            model_out_shape = ["?" for _ in range(len(out_shape))]

        graph = helper.make_graph(
            [node],
            "tile_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, model_in_shape),
            ],
            initializer=[
                helper.make_tensor("repeats", TensorProto.INT64, list(repeats.shape), repeats)
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, model_out_shape)],
        )

        model = helper.make_model(
            graph, producer_name="tile_test", opset_imports=[helper.make_opsetid("", 14)]
        )
        tvm_model = from_onnx(model, keep_params_in_input=True)
        assert len(tvm_model["main"].attrs["params"]) == 1
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        expected = tvm_tirx.IRModule(expected.functions)
        expected.update_func(expected.get_global_var("tile"), tvm_model["tile"])
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    tile_input_dim_0 = T.dynamic("tile_input_dim_0", "int64")
    tile_input_dim_1 = T.dynamic("tile_input_dim_1", "int64")
    tile_input_dim_2 = T.dynamic("tile_input_dim_2", "int64")
    tile_input_dim_3 = T.dynamic("tile_input_dim_3", "int64")

    @I.ir_module
    class ExpectedTileDynamicInput:
        @Ts.function(private=True)
        def tile(
            input: T.Tensor(
                (tile_input_dim_0, tile_input_dim_1, tile_input_dim_2, tile_input_dim_3), "float32"
            ),
            T_tile: T.Tensor(
                (
                    tile_input_dim_0 * 2,
                    tile_input_dim_1,
                    tile_input_dim_2 * 3,
                    tile_input_dim_3 * 2,
                ),
                "float32",
            ),
        ):
            T.evaluate(0)

        @R.function
        def main(
            input: R.Tensor(
                (
                    tile_input_dim_0,
                    tile_input_dim_1,
                    tile_input_dim_2,
                    tile_input_dim_3,
                ),
                dtype="float32",
            ),
            repeats: R.Tensor((4,), dtype="int64"),
        ) -> R.Tensor(
            (
                tile_input_dim_0 * 2,
                tile_input_dim_1,
                tile_input_dim_2 * 3,
                tile_input_dim_3 * 2,
            ),
            dtype="float32",
        ):
            R.func_attr({"num_input": 1})
            cls = ExpectedTileDynamicInput
            with R.dataflow():
                lv = R.call_tir(
                    cls.tile,
                    (input,),
                    ty_args=[
                        R.Tensor(
                            (
                                tile_input_dim_0 * 2,
                                tile_input_dim_1,
                                tile_input_dim_2 * 3,
                                tile_input_dim_3 * 2,
                            ),
                            dtype="float32",
                        )
                    ],
                )
                gv: R.Tensor(
                    (
                        tile_input_dim_0 * 2,
                        tile_input_dim_1,
                        tile_input_dim_2 * 3,
                        tile_input_dim_3 * 2,
                    ),
                    dtype="float32",
                ) = lv
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedTileStaticInput:
        @Ts.function(private=True)
        def tile(
            input: T.Tensor((2, 3, 4, 5), "float32"), T_tile: T.Tensor((4, 3, 12, 10), "float32")
        ):
            T.evaluate(0)

        @R.function
        def main(
            input: R.Tensor((2, 3, 4, 5), dtype="float32"),
            repeats: R.Tensor((4,), dtype="int64"),
        ) -> R.Tensor((4, 3, 12, 10), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = ExpectedTileStaticInput
            with R.dataflow():
                lv = R.call_tir(
                    cls.tile,
                    (input,),
                    ty_args=[R.Tensor((4, 3, 12, 10), dtype="float32")],
                )
                gv: R.Tensor((4, 3, 12, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.array([2, 1, 3, 2], dtype=np.int64)
    z_array = np.tile(x, repeats)
    verify_tile(True, x.shape, repeats, z_array.shape, ExpectedTileDynamicInput)
    verify_tile(False, x.shape, repeats, z_array.shape, ExpectedTileStaticInput)


def test_tile_dynamic_repeats():
    def verify_tile_dynamic_repeats(dynamic_input, in_shape, repeats, expected):
        out_shape = np.tile(np.empty(in_shape, dtype=np.float32), repeats).shape

        input_shape = ["?" for _ in in_shape] if dynamic_input else list(in_shape)
        output_shape = ["?" for _ in out_shape] if dynamic_input else list(out_shape)

        node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])
        graph = helper.make_graph(
            [node],
            "tile_dynamic_repeats_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("repeats", TensorProto.INT64, [len(repeats)]),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(
            graph,
            producer_name="tile_dynamic_repeats_test",
            opset_imports=[helper.make_opsetid("", 13)],
        )

        tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
        expected = tvm_tirx.IRModule(expected.functions)
        expected.update_func(expected.get_global_var("dyn_tile"), tvm_model["dyn_tile"])
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    def make_expected(dynamic_input, in_shape):
        rank = len(in_shape)
        input_shape = (
            tuple(tirx.Var(f"tile_data_dim_{i}", "int64") for i in range(rank))
            if dynamic_input
            else tuple(in_shape)
        )

        if rank == 2:
            tile_dim_0 = T.dynamic("tile_dim_0", "int64")
            tile_dim_1 = T.dynamic("tile_dim_1", "int64")

            @I.ir_module
            class ExpectedTileRank2:
                @Ts.function(private=True)
                def dyn_tile(
                    input: T.Tensor(input_shape, "float32"),
                    var_T_tile: T.Tensor((tile_dim_0, tile_dim_1), "float32"),
                ):
                    T.evaluate(0)

                @R.function
                def main(
                    input: R.Tensor(input_shape, dtype="float32"),
                    repeats: R.Tensor((2,), dtype="int64"),
                ) -> R.Tensor(dtype="float32", ndim=2):
                    R.func_attr({"num_input": 2})
                    cls = ExpectedTileRank2
                    with R.dataflow():
                        lv = R.shape_of(input)
                        lv1: R.Tensor((2,), dtype="int64") = R.shape_to_tensor(lv)
                        lv2: R.Tensor((2,), dtype="int64") = R.multiply(repeats, lv1)
                        lv3: R.Shape([tile_dim_0, tile_dim_1]) = R.match_cast(
                            R.tensor_to_shape(lv2), R.Shape([tile_dim_0, tile_dim_1])
                        )
                        lv4 = R.call_tir(
                            cls.dyn_tile,
                            (input,),
                            ty_args=[R.Tensor((tile_dim_0, tile_dim_1), dtype="float32")],
                        )
                        gv: R.Tensor((tile_dim_0, tile_dim_1), dtype="float32") = lv4
                        R.output(gv)
                    return gv

            return ExpectedTileRank2

        if rank == 3:
            tile_dim_0 = T.dynamic("tile_dim_0", "int64")
            tile_dim_1 = T.dynamic("tile_dim_1", "int64")
            tile_dim_2 = T.dynamic("tile_dim_2", "int64")

            @I.ir_module
            class ExpectedTileRank3:
                @Ts.function(private=True)
                def dyn_tile(
                    input: T.Tensor(input_shape, "float32"),
                    var_T_tile: T.Tensor((tile_dim_0, tile_dim_1, tile_dim_2), "float32"),
                ):
                    T.evaluate(0)

                @R.function
                def main(
                    input: R.Tensor(input_shape, dtype="float32"),
                    repeats: R.Tensor((3,), dtype="int64"),
                ) -> R.Tensor(dtype="float32", ndim=3):
                    R.func_attr({"num_input": 2})
                    cls = ExpectedTileRank3
                    with R.dataflow():
                        lv = R.shape_of(input)
                        lv1: R.Tensor((3,), dtype="int64") = R.shape_to_tensor(lv)
                        lv2: R.Tensor((3,), dtype="int64") = R.multiply(repeats, lv1)
                        lv3: R.Shape([tile_dim_0, tile_dim_1, tile_dim_2]) = R.match_cast(
                            R.tensor_to_shape(lv2),
                            R.Shape([tile_dim_0, tile_dim_1, tile_dim_2]),
                        )
                        lv4 = R.call_tir(
                            cls.dyn_tile,
                            (input,),
                            ty_args=[
                                R.Tensor((tile_dim_0, tile_dim_1, tile_dim_2), dtype="float32")
                            ],
                        )
                        gv: R.Tensor((tile_dim_0, tile_dim_1, tile_dim_2), dtype="float32") = lv4
                        R.output(gv)
                    return gv

            return ExpectedTileRank3

        if rank == 4:
            tile_dim_0 = T.dynamic("tile_dim_0", "int64")
            tile_dim_1 = T.dynamic("tile_dim_1", "int64")
            tile_dim_2 = T.dynamic("tile_dim_2", "int64")
            tile_dim_3 = T.dynamic("tile_dim_3", "int64")

            @I.ir_module
            class ExpectedTileRank4:
                @Ts.function(private=True)
                def dyn_tile(
                    input: T.Tensor(input_shape, "float32"),
                    var_T_tile: T.Tensor(
                        (tile_dim_0, tile_dim_1, tile_dim_2, tile_dim_3), "float32"
                    ),
                ):
                    T.evaluate(0)

                @R.function
                def main(
                    input: R.Tensor(input_shape, dtype="float32"),
                    repeats: R.Tensor((4,), dtype="int64"),
                ) -> R.Tensor(dtype="float32", ndim=4):
                    R.func_attr({"num_input": 2})
                    cls = ExpectedTileRank4
                    with R.dataflow():
                        lv = R.shape_of(input)
                        lv1: R.Tensor((4,), dtype="int64") = R.shape_to_tensor(lv)
                        lv2: R.Tensor((4,), dtype="int64") = R.multiply(repeats, lv1)
                        lv3: R.Shape([tile_dim_0, tile_dim_1, tile_dim_2, tile_dim_3]) = (
                            R.match_cast(
                                R.tensor_to_shape(lv2),
                                R.Shape([tile_dim_0, tile_dim_1, tile_dim_2, tile_dim_3]),
                            )
                        )
                        lv4 = R.call_tir(
                            cls.dyn_tile,
                            (input,),
                            ty_args=[
                                R.Tensor(
                                    (tile_dim_0, tile_dim_1, tile_dim_2, tile_dim_3),
                                    dtype="float32",
                                )
                            ],
                        )
                        gv: R.Tensor(
                            (tile_dim_0, tile_dim_1, tile_dim_2, tile_dim_3), dtype="float32"
                        ) = lv4
                        R.output(gv)
                    return gv

            return ExpectedTileRank4

        raise AssertionError(f"No dynamic Tile expected IR for rank {rank}")

    tile_cases = [
        (True, (2, 3), np.array([2, 2], dtype=np.int64)),
        (True, (2, 3, 4), np.array([2, 2, 1], dtype=np.int64)),
        (True, (2, 3, 4, 5), np.array([1, 2, 1, 2], dtype=np.int64)),
        (False, (2, 3), np.array([2, 2], dtype=np.int64)),
        (False, (2, 3, 4), np.array([2, 2, 1], dtype=np.int64)),
        (False, (2, 3, 4, 5), np.array([1, 2, 1, 2], dtype=np.int64)),
    ]
    for dynamic_input, in_shape, repeats in tile_cases:
        verify_tile_dynamic_repeats(
            dynamic_input, in_shape, repeats, make_expected(dynamic_input, in_shape)
        )


def _generate_roi_cases():
    return [
        pytest.param(False, None, False, id="no_roi"),
        pytest.param(True, [], True, id="empty_roi_constant"),
        pytest.param(True, [], False, id="empty_roi_initializer"),
        pytest.param(True, [0.1, 0.2, 0.9, 0.8], True, id="spatial_roi_constant"),
        pytest.param(True, [0.1, 0.2, 0.9, 0.8], False, id="spatial_roi_initializer"),
    ]


@pytest.mark.parametrize("with_roi, roi_list, with_constant", _generate_roi_cases())
def test_resize(with_roi, roi_list, with_constant):
    nodes = []
    resize_node = helper.make_node(
        "Resize", ["X", "roi" if with_roi else "", "scales"], ["Y"], mode="cubic"
    )

    if with_roi and with_constant:
        roi_tensor = helper.make_tensor(
            name="roi",
            data_type=TensorProto.FLOAT,
            dims=[len(roi_list)],
            vals=roi_list,
        )

        roi_const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["roi"],
            value=roi_tensor,
        )
        nodes.append(roi_const_node)

    nodes.append(resize_node)

    initializers = [
        helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
    ]

    if with_roi and not with_constant:
        initializers.append(helper.make_tensor("roi", TensorProto.FLOAT, [len(roi_list)], roi_list))

    graph = helper.make_graph(
        nodes,
        "resize_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
        initializer=initializers,
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64]),
        ],
    )

    model = helper.make_model(graph, producer_name="resize_test")
    check_correctness(model)


def test_resize_dynamic_roi_tf_crop_and_resize():
    """ROI is a graph input (not initializer), lowered through TOPI dynamic-ROI path."""
    resize_node = helper.make_node(
        "Resize",
        ["X", "roi", "scales"],
        ["Y"],
        mode="linear",
        coordinate_transformation_mode="tf_crop_and_resize",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize_dynamic_roi",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32]),
            helper.make_tensor_value_info("roi", TensorProto.FLOAT, [8]),
        ],
        initializer=[
            helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64]),
        ],
    )
    model = helper.make_model(graph, producer_name="resize_dynamic_roi")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    seen_call_tir = False

    def _visit(expr):
        nonlocal seen_call_tir
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm_tirx.ir.Op):
            if expr.op.name == "relax.call_tir":
                seen_call_tir = True

    relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
    assert seen_call_tir


def test_resize_dynamic_roi_3d_tf_crop_and_resize():
    """5-D NCDHW: ROI is a graph input; covers dynamic-ROI TOPI resize3d path."""
    resize_node = helper.make_node(
        "Resize",
        ["X", "roi", "scales"],
        ["Y"],
        mode="linear",
        coordinate_transformation_mode="tf_crop_and_resize",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize_dynamic_roi_3d",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 4, 5]),
            helper.make_tensor_value_info("roi", TensorProto.FLOAT, [10]),
        ],
        initializer=[
            helper.make_tensor("scales", TensorProto.FLOAT, [5], [1.0, 1.0, 2.0, 2.0, 2.0]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 6, 8, 10]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="resize_dynamic_roi_3d",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)
    seen_call_tir = False

    def _visit(expr):
        nonlocal seen_call_tir
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm_tirx.ir.Op):
            if expr.op.name == "relax.call_tir":
                seen_call_tir = True

    relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
    assert seen_call_tir


def test_resize_nd_sizes():
    cases = [
        ("resize1d", [1, 1, 4], [1, 1, 7]),
        ("resize2d", [1, 1, 4, 5], [1, 1, 6, 7]),
        ("resize3d", [1, 1, 3, 4, 5], [1, 1, 4, 6, 7]),
    ]

    for name, input_shape, sizes in cases:
        resize_node = helper.make_node(
            "Resize",
            ["X", "", "", "sizes"],
            ["Y"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )

        graph = helper.make_graph(
            [resize_node],
            name,
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
            ],
            initializer=[
                helper.make_tensor("sizes", TensorProto.INT64, [len(sizes)], sizes),
            ],
            outputs=[
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, sizes),
            ],
        )

        model = helper.make_model(
            graph, producer_name=name, opset_imports=[helper.make_opsetid("", 18)]
        )
        if name != "resize1d":
            check_correctness(model, opset=18)
            continue

        tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)
        seen_call_tir = False

        def _visit(expr):
            nonlocal seen_call_tir
            if isinstance(expr, relax.Call) and isinstance(expr.op, tvm_tirx.ir.Op):
                if expr.op.name == "relax.call_tir":
                    seen_call_tir = True

        relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
        assert seen_call_tir


def test_resize_5d_emits_relax_resize3d():
    resize_node = helper.make_node(
        "Resize",
        ["X", "", "", "sizes"],
        ["Y"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize3d_ir_check",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 4, 5])],
        initializer=[helper.make_tensor("sizes", TensorProto.INT64, [5], [1, 1, 4, 6, 7])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 6, 7])],
    )
    model = helper.make_model(graph, producer_name="resize3d_ir_check")
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    seen_resize3d = False

    def _visit(expr):
        nonlocal seen_resize3d
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm_tirx.ir.Op):
            if expr.op.name == "relax.image.resize3d":
                seen_resize3d = True

    relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
    assert seen_resize3d


def test_einsum():
    eqn = "ij->i"
    einsum_node = helper.make_node("Einsum", ["x"], ["y"], equation=eqn)

    graph = helper.make_graph(
        [einsum_node],
        "einsum_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [3]),
        ],
    )

    model = helper.make_model(graph, producer_name="einsum_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def einsum(x: T.Tensor((3, 4), "float32"), T_einsum: T.Tensor((3,), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.einsum, (x,), ty_args=[R.Tensor((3,), dtype="float32")])
                gv: R.Tensor((3,), dtype="float32") = lv
                R.output(gv)
            return gv

    expected = tvm_tirx.IRModule(Expected.functions)
    expected.update_func(expected.get_global_var("einsum"), tvm_model["einsum"])
    tvm_tirx.ir.assert_structural_equal(tvm_model, expected)


def test_range():
    range_node = helper.make_node(
        "Range",
        ["start", "limit", "delta"],
        ["output"],
    )

    graph = helper.make_graph(
        [range_node],
        "range_test",
        inputs=[],
        initializer=[
            helper.make_tensor("start", TensorProto.INT64, [], [1]),
            helper.make_tensor("limit", TensorProto.INT64, [], [5]),
            helper.make_tensor("delta", TensorProto.INT64, [], [2]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.INT64, [2]),
        ],
    )

    model = helper.make_model(graph, producer_name="range_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            start: R.Tensor((), dtype="int64"),
            limit: R.Tensor((), dtype="int64"),
            delta: R.Tensor((), dtype="int64"),
        ) -> R.Tensor((2,), dtype="int64"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((2,), dtype="int64") = R.const(
                    np.array([1, 3], dtype=np.int64), "int64"
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_range_constant_float():
    range_node = helper.make_node(
        "Range",
        ["start", "limit", "delta"],
        ["output"],
    )

    graph = helper.make_graph(
        [range_node],
        "range_constant_float_test",
        inputs=[],
        initializer=[
            helper.make_tensor("start", TensorProto.FLOAT, [], [-7.8]),
            helper.make_tensor("limit", TensorProto.FLOAT, [], [-1.4]),
            helper.make_tensor("delta", TensorProto.FLOAT, [], [0.2]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, ["range_len"]),
        ],
    )

    model = helper.make_model(graph, producer_name="range_constant_float_test")
    check_correctness(model, opset=12, check_dtypes=True)


@pytest.mark.parametrize(
    "start, limit, delta, tensor_dtype, np_dtype",
    [
        (0, 6, 2, TensorProto.INT64, np.int64),
        (8, 0, -2, TensorProto.INT64, np.int64),
        (5, 1, 1, TensorProto.INT64, np.int64),
        (0, 7, 2, TensorProto.INT32, np.int32),
        (0.0, 1.0, 0.25, TensorProto.FLOAT, np.float32),
        (1.0, -1.0, -0.5, TensorProto.FLOAT, np.float32),
        (0.0, 0.3, 0.1, TensorProto.FLOAT, np.float32),
        (-7.8, -1.4, 0.2, TensorProto.FLOAT, np.float32),
    ],
)
def test_range_dynamic_scalar_inputs(start, limit, delta, tensor_dtype, np_dtype):
    range_node = helper.make_node(
        "Range",
        ["start", "limit", "delta"],
        ["output"],
    )

    graph = helper.make_graph(
        [range_node],
        "range_dynamic_scalar_inputs_test",
        inputs=[
            helper.make_tensor_value_info("start", tensor_dtype, []),
            helper.make_tensor_value_info("limit", tensor_dtype, []),
            helper.make_tensor_value_info("delta", tensor_dtype, []),
        ],
        outputs=[
            helper.make_tensor_value_info("output", tensor_dtype, ["range_len"]),
        ],
    )

    model = helper.make_model(graph, producer_name="range_dynamic_scalar_inputs_test")
    check_correctness(
        model,
        inputs={
            "start": np.array(start, dtype=np_dtype),
            "limit": np.array(limit, dtype=np_dtype),
            "delta": np.array(delta, dtype=np_dtype),
        },
        opset=12,
        check_dtypes=True,
    )


def test_range_mixed_tensor_and_primexpr_limit():
    shape = helper.make_node("Shape", ["x"], ["x_shape"])
    axis = make_constant_node("axis", TensorProto.INT64, [], [1])
    gather = helper.make_node("Gather", ["x_shape", "axis"], ["limit_int"])
    cast = helper.make_node("Cast", ["limit_int"], ["limit"], to=TensorProto.FLOAT)
    delta = make_constant_node("delta", TensorProto.FLOAT, [], [1.0])
    range_node = helper.make_node(
        "Range",
        ["start", "limit", "delta"],
        ["output"],
    )

    graph = helper.make_graph(
        [shape, axis, gather, cast, delta, range_node],
        "range_mixed_tensor_and_primexpr_limit_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "range_len"]),
            helper.make_tensor_value_info("start", TensorProto.FLOAT, []),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, ["range_len"]),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="range_mixed_tensor_and_primexpr_limit_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    check_correctness(
        model,
        inputs={
            "x": np.ones((1, 4), dtype=np.float32),
            "start": np.array(0.0, dtype=np.float32),
        },
        opset=17,
        check_dtypes=True,
    )


@pytest.mark.parametrize(
    "start_from_dim, limit_from_dim, delta",
    [
        (True, False, -1),
        (False, True, -1),
    ],
)
def test_range_primexpr_negative_and_empty(start_from_dim, limit_from_dim, delta):
    shape = helper.make_node("Shape", ["x"], ["x_shape"])
    axis = make_constant_node("axis", TensorProto.INT64, [], [1])
    gather = helper.make_node("Gather", ["x_shape", "axis"], ["dim"])
    start = make_constant_node("start", TensorProto.INT64, [], [0])
    limit = make_constant_node("limit", TensorProto.INT64, [], [0])
    delta_node = make_constant_node("delta", TensorProto.INT64, [], [delta])

    range_inputs = [
        "dim" if start_from_dim else "start",
        "dim" if limit_from_dim else "limit",
        "delta",
    ]
    range_node = helper.make_node("Range", range_inputs, ["output"])

    graph = helper.make_graph(
        [shape, axis, gather, start, limit, delta_node, range_node],
        "range_primexpr_negative_and_empty_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "range_len"]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.INT64, ["output_len"]),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="range_primexpr_negative_and_empty_test",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 8
    check_correctness(
        model,
        inputs={
            "x": np.ones((1, 4), dtype=np.float32),
        },
        opset=17,
        check_dtypes=True,
    )


def test_batch_norm():
    batch_norm_node = helper.make_node(
        "BatchNormalization", ["x", "s", "bias", "mean", "var"], ["y"], epsilon=1e-2
    )
    graph = helper.make_graph(
        [batch_norm_node],
        "batch_norm_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4, 5])],
    )

    model = helper.make_model(graph, producer_name="batch_norm_test")
    check_correctness(model, opset=15)


def test_batch_norm_defaults_to_inference_mode():
    batch_norm_node = helper.make_node(
        "BatchNormalization", ["x", "s", "bias", "mean", "var"], ["y"], epsilon=1e-2
    )
    graph = helper.make_graph(
        [batch_norm_node],
        "batch_norm_inference_attr_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4, 5])],
    )
    model = helper.make_model(graph, producer_name="batch_norm_inference_attr_test")
    model.opset_import[0].version = 15

    tvm_model = from_onnx(model, opset=15, keep_params_in_input=True)
    batch_norm_attrs = []

    def visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm_tirx.ir.Op.get("relax.nn.batch_norm"):
            batch_norm_attrs.append(expr.attrs)

    relax.analysis.post_order_visit(tvm_model["main"], visit)

    assert len(batch_norm_attrs) == 1
    assert batch_norm_attrs[0].training is False


def test_batch_norm_mixed_dtype_params():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT16, [1, 3, 2, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, 2, 2])
    params = [
        numpy_helper.from_array(np.array([1.0, 1.5, 2.0], dtype=np.float32), name="gamma"),
        numpy_helper.from_array(np.array([0.0, 0.1, -0.1], dtype=np.float32), name="beta"),
        numpy_helper.from_array(np.array([0.2, -0.3, 0.4], dtype=np.float32), name="mean"),
        numpy_helper.from_array(np.array([1.0, 1.5, 2.0], dtype=np.float32), name="var"),
    ]
    batch_norm_node = helper.make_node(
        "BatchNormalization",
        ["data", "gamma", "beta", "mean", "var"],
        ["output"],
        epsilon=1e-5,
        momentum=0.9,
        training_mode=0,
    )
    graph = helper.make_graph(
        [batch_norm_node],
        "mixed_dtype_batchnorm",
        [data],
        [output],
        initializer=params,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])

    tvm_model = from_onnx(model, keep_params_in_input=False)

    assert tuple(dim.value for dim in tvm_model["main"].ret_ty.shape.values) == (1, 3, 2, 2)
    assert tvm_model["main"].ret_ty.dtype == "float16"

    batch_norm_calls = []

    def visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm_tirx.ir.Op.get("relax.nn.batch_norm"):
            batch_norm_calls.append(expr)

    relax.analysis.post_order_visit(tvm_model["main"], visit)

    assert len(batch_norm_calls) == 1
    arg_dtypes = [
        str(getattr(arg, "struct_info", getattr(arg, "ty", None)).dtype)
        for arg in batch_norm_calls[0].args
    ]
    assert arg_dtypes == ["float32"] * 5


def test_batch_norm_training_preserves_output_dtypes():
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT16, [1, 3, 2, 2])
    outputs = [
        helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, 2, 2]),
        helper.make_tensor_value_info("running_mean", TensorProto.FLOAT16, [3]),
        helper.make_tensor_value_info("running_var", TensorProto.FLOAT16, [3]),
    ]
    inputs = [
        data,
        helper.make_tensor_value_info("gamma", TensorProto.FLOAT16, [3]),
        helper.make_tensor_value_info("beta", TensorProto.FLOAT16, [3]),
        helper.make_tensor_value_info("mean", TensorProto.FLOAT16, [3]),
        helper.make_tensor_value_info("var", TensorProto.FLOAT16, [3]),
    ]
    batch_norm_node = helper.make_node(
        "BatchNormalization",
        [value.name for value in inputs],
        [value.name for value in outputs],
        training_mode=1,
    )
    graph = helper.make_graph(
        [batch_norm_node],
        "mixed_dtype_training_batchnorm",
        inputs,
        outputs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])

    tvm_model = from_onnx(model, keep_params_in_input=True)

    assert [str(field.dtype) for field in tvm_model["main"].ret_ty.fields] == [
        "float16",
        "float16",
        "float16",
    ]

    batch_norm_calls = []

    def visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm_tirx.ir.Op.get("relax.nn.batch_norm"):
            batch_norm_calls.append(expr)

    relax.analysis.post_order_visit(tvm_model["main"], visit)

    assert len(batch_norm_calls) == 1
    assert [str(arg.ty.dtype) for arg in batch_norm_calls[0].args] == ["float32"] * 5


def get_pool_padding(shape, auto_pad, kernel_shape, strides, pads):
    def get_pad_pair(input1d, kernel1d, stride1d, mode):
        if input1d % stride1d == 0:
            pad = max(kernel1d - stride1d, 0)
        else:
            pad = max(kernel1d - (input1d % stride1d), 0)
        pad_before = pad // 2
        pad_after = pad - pad_before
        if "LOWER" in mode:
            return [pad_after, pad_before]
        return [pad_before, pad_after]

    strides = strides or [1] * (len(shape) - 2)
    padding = pads if pads is not None else 0

    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        pad_pairs = [
            get_pad_pair(int(shape[2 + axis]), kernel_shape[axis], strides[axis], auto_pad)
            for axis in range(len(shape) - 2)
        ]
        padding = tuple(val for pair in zip(*pad_pairs) for val in pair)

    return padding


def verify_pool_ir(pool_name, shape, auto_pad, kernel_shape, strides, pads, expected):
    attrs = {
        "kernel_shape": kernel_shape,
        "strides": strides,
        "auto_pad": auto_pad,
    }
    if pads is not None:
        attrs["pads"] = pads

    node = helper.make_node(pool_name, ["x"], ["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "pool_structural_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, producer_name="pool_structural_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    tvm_tirx.ir.assert_structural_equal(tvm_model, expected)


def test_pool():
    def make_expected(pool_name, shape, auto_pad, kernel_shape, strides, pads):
        rank = len(shape) - 2
        layout = {1: "NCW", 2: "NCHW", 3: "NCDHW"}[rank]
        padding = get_pool_padding(shape, auto_pad, kernel_shape, strides, pads)
        pool_op = {
            ("MaxPool", 1): R.nn.max_pool1d,
            ("MaxPool", 2): R.nn.max_pool2d,
            ("MaxPool", 3): R.nn.max_pool3d,
            ("AveragePool", 1): R.nn.avg_pool1d,
            ("AveragePool", 2): R.nn.avg_pool2d,
            ("AveragePool", 3): R.nn.avg_pool3d,
            ("LpPool", 1): R.nn.avg_pool1d,
            ("LpPool", 2): R.nn.avg_pool2d,
            ("LpPool", 3): R.nn.avg_pool3d,
        }[(pool_name, rank)]
        input_shape = tuple(shape)
        pool_size = kernel_shape
        dilation = [1] * rank

        if pool_name == "MaxPool":

            @I.ir_module
            class ExpectedMaxPool:
                @R.function
                def main(x: R.Tensor(input_shape, dtype="float32")):
                    R.func_attr({"num_input": 1})
                    with R.dataflow():
                        gv = pool_op(
                            x,
                            pool_size=pool_size,
                            strides=strides,
                            dilation=dilation,
                            padding=padding,
                            ceil_mode=False,
                            layout=layout,
                            out_layout=layout,
                        )
                        R.output(gv)
                    return gv

            return ExpectedMaxPool

        if pool_name == "AveragePool":

            @I.ir_module
            class ExpectedAveragePool:
                @R.function
                def main(x: R.Tensor(input_shape, dtype="float32")):
                    R.func_attr({"num_input": 1})
                    with R.dataflow():
                        gv = pool_op(
                            x,
                            pool_size=pool_size,
                            strides=strides,
                            dilation=dilation,
                            padding=padding,
                            ceil_mode=False,
                            count_include_pad=False,
                            layout=layout,
                            out_layout=layout,
                        )
                        R.output(gv)
                    return gv

            return ExpectedAveragePool

        kernel_elements = float(np.prod(kernel_shape))

        @I.ir_module
        class ExpectedLpPool:
            @R.function
            def main(x: R.Tensor(input_shape, dtype="float32")):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv = R.abs(x)
                    lv1 = R.power(lv, R.const(2.0, "float32"))
                    lv2 = pool_op(
                        lv1,
                        pool_size=pool_size,
                        strides=strides,
                        dilation=dilation,
                        padding=padding,
                        ceil_mode=False,
                        count_include_pad=True,
                        layout=layout,
                        out_layout=layout,
                    )
                    lv3 = R.multiply(lv2, R.const(kernel_elements, "float32"))
                    gv = R.power(lv3, R.const(0.5, "float32"))
                    R.output(gv)
                return gv

        return ExpectedLpPool

    pool_cases = [
        ([1, 1, 32], "NOTSET", [3], [1], [1, 1]),
        ([1, 1, 32], "NOTSET", [3], [2], [1, 1]),
        ([1, 1, 32], "SAME_UPPER", [7], [2], None),
        ([1, 1, 32], "SAME_LOWER", [4], [4], None),
        ([1, 1, 32], "VALID", [5], [5], None),
        ([1, 1, 32], "SAME_UPPER", [3], [1], None),
        ([1, 1, 32, 32], "NOTSET", [3, 3], [1, 1], [1, 1, 1, 1]),
        ([1, 1, 32, 32], "NOTSET", [3, 3], [2, 2], [1, 1, 1, 1]),
        ([1, 1, 32, 32], "SAME_UPPER", [3, 7], [3, 2], None),
        ([1, 1, 32, 32], "SAME_LOWER", [3, 3], [2, 2], None),
        ([1, 1, 32, 32], "VALID", [3, 3], [2, 2], None),
        ([1, 1, 32, 32], "SAME_UPPER", [3, 3], [1, 1], None),
        ([1, 1, 32, 32, 32], "NOTSET", [3, 3, 4], [1, 1, 1], [1, 2, 1, 1, 2, 2]),
        ([1, 1, 32, 32, 32], "NOTSET", [3, 4, 3], [2, 2, 3], [1, 1, 1, 1, 1, 2]),
        ([1, 1, 32, 32, 32], "SAME_UPPER", [4, 3, 3], [3, 2, 2], None),
        ([1, 1, 32, 32, 32], "SAME_LOWER", [3, 3, 4], [2, 2, 2], None),
        ([1, 1, 32, 32, 32], "VALID", [3, 3, 5], [2, 2, 3], None),
        ([1, 1, 32, 32, 32], "SAME_UPPER", [3, 3, 5], [1, 1, 1], None),
    ]

    for pool_name in ["MaxPool", "AveragePool", "LpPool"]:
        for shape, auto_pad, kernel_shape, strides, pads in pool_cases:
            verify_pool_ir(
                pool_name,
                shape,
                auto_pad,
                kernel_shape,
                strides,
                pads,
                make_expected(pool_name, shape, auto_pad, kernel_shape, strides, pads),
            )


@pytest.mark.parametrize("p", [1, 3])
def test_lppool_negative_input(p: int):
    input_data = np.array([[[-1.0, 2.0, -3.0, 4.0]]], dtype="float32")

    node = helper.make_node(
        "LpPool",
        ["x"],
        ["y"],
        kernel_shape=[2],
        strides=[1],
        p=p,
    )

    graph = helper.make_graph(
        [node],
        "lppool_negative_input_test",
        inputs=[
            helper.make_tensor_value_info(
                "x",
                TensorProto.FLOAT,
                [1, 1, 4],
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "y",
                TensorProto.FLOAT,
                [1, 1, 3],
            )
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="lppool_negative_input_test",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    check_correctness(
        model,
        inputs={"x": input_data},
        opset=18,
    )


def test_global_average_pool():
    def verify_global_average_pool_ir(input_shape, expected):
        output_shape = input_shape[:2] + [1] * (len(input_shape) - 2)
        node = helper.make_node("GlobalAveragePool", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "global_average_pool_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(
            graph,
            producer_name="global_average_pool_test",
            opset_imports=[helper.make_opsetid("", 14)],
        )
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class Expected1D:
        @R.function
        def main(x: R.Tensor((1, 3, 32), dtype="float32")) -> R.Tensor((1, 3, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1), dtype="float32") = R.mean(x, axis=[2], keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2D:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 1, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.mean(x, axis=[2, 3], keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3D:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 1, 1, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.mean(
                    x, axis=[2, 3, 4], keepdims=True
                )
                R.output(gv)
            return gv

    verify_global_average_pool_ir([1, 3, 32], Expected1D)
    verify_global_average_pool_ir([1, 3, 32, 32], Expected2D)
    verify_global_average_pool_ir([1, 3, 32, 32, 32], Expected3D)


def test_global_max_pool():
    def verify_global_max_pool_ir(input_shape, expected):
        output_shape = input_shape[:2] + [1] * (len(input_shape) - 2)
        node = helper.make_node("GlobalMaxPool", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "global_max_pool_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(
            graph,
            producer_name="global_max_pool_test",
            opset_imports=[helper.make_opsetid("", 14)],
        )
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class Expected1D:
        @R.function
        def main(x: R.Tensor((1, 3, 32), dtype="float32")) -> R.Tensor((1, 3, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1), dtype="float32") = R.max(x, axis=[2], keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected2D:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 1, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.max(x, axis=[2, 3], keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected3D:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 1, 1, 1), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.max(
                    x, axis=[2, 3, 4], keepdims=True
                )
                R.output(gv)
            return gv

    verify_global_max_pool_ir([1, 3, 32], Expected1D)
    verify_global_max_pool_ir([1, 3, 32, 32], Expected2D)
    verify_global_max_pool_ir([1, 3, 32, 32, 32], Expected3D)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_global_lp_pool(p: int):
    p_value = float(p)
    inv_p_value = float(1 / p)

    def verify_global_lp_pool(input_shape, expected):
        output_shape = input_shape[:2] + [1] * (len(input_shape) - 2)
        node = helper.make_node("GlobalLpPool", ["x"], ["y"], p=p)
        graph = helper.make_graph(
            [node],
            "global_lp_pool_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(graph, producer_name="global_lp_pool_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedGlobalLpPool1D:
        @R.function
        def main(
            x: R.Tensor((1, 3, 4), dtype="float32"),
        ) -> R.Tensor((1, 3, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 4), dtype="float32") = R.abs(x)
                lv1: R.Tensor((1, 3, 4), dtype="float32") = R.power(lv, R.const(p_value, "float32"))
                lv2: R.Tensor((1, 3, 1), dtype="float32") = R.sum(lv1, axis=[2], keepdims=True)
                gv: R.Tensor((1, 3, 1), dtype="float32") = R.power(
                    lv2, R.const(inv_p_value, "float32")
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedGlobalLpPool2D:
        @R.function
        def main(
            x: R.Tensor((1, 3, 4, 4), dtype="float32"),
        ) -> R.Tensor((1, 3, 1, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4), dtype="float32") = R.abs(x)
                lv1: R.Tensor((1, 3, 4, 4), dtype="float32") = R.power(
                    lv, R.const(p_value, "float32")
                )
                lv2: R.Tensor((1, 3, 1, 1), dtype="float32") = R.sum(
                    lv1, axis=[2, 3], keepdims=True
                )
                gv: R.Tensor((1, 3, 1, 1), dtype="float32") = R.power(
                    lv2, R.const(inv_p_value, "float32")
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedGlobalLpPool3D:
        @R.function
        def main(
            x: R.Tensor((1, 3, 4, 4, 4), dtype="float32"),
        ) -> R.Tensor((1, 3, 1, 1, 1), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = R.abs(x)
                lv1: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = R.power(
                    lv, R.const(p_value, "float32")
                )
                lv2: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.sum(
                    lv1, axis=[2, 3, 4], keepdims=True
                )
                gv: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.power(
                    lv2, R.const(inv_p_value, "float32")
                )
                R.output(gv)
            return gv

    verify_global_lp_pool([1, 3, 4], ExpectedGlobalLpPool1D)
    verify_global_lp_pool([1, 3, 4, 4], ExpectedGlobalLpPool2D)
    verify_global_lp_pool([1, 3, 4, 4, 4], ExpectedGlobalLpPool3D)


def test_maxunpool():
    input_shape = [16, 3, 16, 16]

    def verify_maxunpool(kernel_shape, pads, strides, expected):
        input_names = ["X", "I"]
        input_info = [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("I", TensorProto.INT64, input_shape),
        ]

        attrs = {"kernel_shape": kernel_shape}
        if pads is not None:
            attrs["pads"] = pads
        if strides is not None:
            attrs["strides"] = strides

        node = helper.make_node("MaxUnpool", inputs=input_names, outputs=["y"], **attrs)

        graph = helper.make_graph(
            [node],
            "maxunpool_test",
            inputs=input_info,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )

        model = helper.make_model(graph, producer_name="maxunpool_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedMaxUnpool0:
        @R.function
        def main(
            X: R.Tensor((16, 3, 16, 16), dtype="float32"),
            I_1: R.Tensor((16, 3, 16, 16), dtype="int64"),
        ) -> R.Tensor((16, 3, 17, 17), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 3, 17, 17), dtype="float32") = R.zeros(
                    R.shape([16, 3, 17, 17]), dtype="float32"
                )
                lv1: R.Tensor((13872,), dtype="float32") = R.reshape(lv, R.shape([13872]))
                lv2: R.Tensor((12288,), dtype="int64") = R.reshape(I_1, R.shape([12288]))
                lv3: R.Tensor((12288,), dtype="float32") = R.reshape(X, R.shape([12288]))
                lv4: R.Tensor((13872,), dtype="float32") = R.scatter_elements(
                    lv1, lv2, lv3, axis=0, reduction="update"
                )
                gv: R.Tensor((16, 3, 17, 17), dtype="float32") = R.reshape(
                    lv4, R.shape([16, 3, 17, 17])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMaxUnpool1:
        @R.function
        def main(
            X: R.Tensor((16, 3, 16, 16), dtype="float32"),
            I_1: R.Tensor((16, 3, 16, 16), dtype="int64"),
        ) -> R.Tensor((16, 3, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 3, 32, 32), dtype="float32") = R.zeros(
                    R.shape([16, 3, 32, 32]), dtype="float32"
                )
                lv1: R.Tensor((49152,), dtype="float32") = R.reshape(lv, R.shape([49152]))
                lv2: R.Tensor((12288,), dtype="int64") = R.reshape(I_1, R.shape([12288]))
                lv3: R.Tensor((12288,), dtype="float32") = R.reshape(X, R.shape([12288]))
                lv4: R.Tensor((49152,), dtype="float32") = R.scatter_elements(
                    lv1, lv2, lv3, axis=0, reduction="update"
                )
                gv: R.Tensor((16, 3, 32, 32), dtype="float32") = R.reshape(
                    lv4, R.shape([16, 3, 32, 32])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMaxUnpool2:
        @R.function
        def main(
            X: R.Tensor((16, 3, 16, 16), dtype="float32"),
            I_1: R.Tensor((16, 3, 16, 16), dtype="int64"),
        ) -> R.Tensor((16, 3, 15, 15), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 3, 15, 15), dtype="float32") = R.zeros(
                    R.shape([16, 3, 15, 15]), dtype="float32"
                )
                lv1: R.Tensor((10800,), dtype="float32") = R.reshape(lv, R.shape([10800]))
                lv2: R.Tensor((12288,), dtype="int64") = R.reshape(I_1, R.shape([12288]))
                lv3: R.Tensor((12288,), dtype="float32") = R.reshape(X, R.shape([12288]))
                lv4: R.Tensor((10800,), dtype="float32") = R.scatter_elements(
                    lv1, lv2, lv3, axis=0, reduction="update"
                )
                gv: R.Tensor((16, 3, 15, 15), dtype="float32") = R.reshape(
                    lv4, R.shape([16, 3, 15, 15])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMaxUnpool3:
        @R.function
        def main(
            X: R.Tensor((16, 3, 16, 16), dtype="float32"),
            I_1: R.Tensor((16, 3, 16, 16), dtype="int64"),
        ) -> R.Tensor((16, 3, 30, 30), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 3, 30, 30), dtype="float32") = R.zeros(
                    R.shape([16, 3, 30, 30]), dtype="float32"
                )
                lv1: R.Tensor((43200,), dtype="float32") = R.reshape(lv, R.shape([43200]))
                lv2: R.Tensor((12288,), dtype="int64") = R.reshape(I_1, R.shape([12288]))
                lv3: R.Tensor((12288,), dtype="float32") = R.reshape(X, R.shape([12288]))
                lv4: R.Tensor((43200,), dtype="float32") = R.scatter_elements(
                    lv1, lv2, lv3, axis=0, reduction="update"
                )
                gv: R.Tensor((16, 3, 30, 30), dtype="float32") = R.reshape(
                    lv4, R.shape([16, 3, 30, 30])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMaxUnpool4:
        @R.function
        def main(
            X: R.Tensor((16, 3, 16, 16), dtype="float32"),
            I_1: R.Tensor((16, 3, 16, 16), dtype="int64"),
        ) -> R.Tensor((16, 3, 18, 18), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 3, 18, 18), dtype="float32") = R.zeros(
                    R.shape([16, 3, 18, 18]), dtype="float32"
                )
                lv1: R.Tensor((15552,), dtype="float32") = R.reshape(lv, R.shape([15552]))
                lv2: R.Tensor((12288,), dtype="int64") = R.reshape(I_1, R.shape([12288]))
                lv3: R.Tensor((12288,), dtype="float32") = R.reshape(X, R.shape([12288]))
                lv4: R.Tensor((15552,), dtype="float32") = R.scatter_elements(
                    lv1, lv2, lv3, axis=0, reduction="update"
                )
                gv: R.Tensor((16, 3, 18, 18), dtype="float32") = R.reshape(
                    lv4, R.shape([16, 3, 18, 18])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMaxUnpool5:
        @R.function
        def main(
            X: R.Tensor((16, 3, 16, 16), dtype="float32"),
            I_1: R.Tensor((16, 3, 16, 16), dtype="int64"),
        ) -> R.Tensor((16, 3, 33, 33), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 3, 33, 33), dtype="float32") = R.zeros(
                    R.shape([16, 3, 33, 33]), dtype="float32"
                )
                lv1: R.Tensor((52272,), dtype="float32") = R.reshape(lv, R.shape([52272]))
                lv2: R.Tensor((12288,), dtype="int64") = R.reshape(I_1, R.shape([12288]))
                lv3: R.Tensor((12288,), dtype="float32") = R.reshape(X, R.shape([12288]))
                lv4: R.Tensor((52272,), dtype="float32") = R.scatter_elements(
                    lv1, lv2, lv3, axis=0, reduction="update"
                )
                gv: R.Tensor((16, 3, 33, 33), dtype="float32") = R.reshape(
                    lv4, R.shape([16, 3, 33, 33])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMaxUnpool6:
        @R.function
        def main(
            X: R.Tensor((16, 3, 16, 16), dtype="float32"),
            I_1: R.Tensor((16, 3, 16, 16), dtype="int64"),
        ) -> R.Tensor((16, 3, 16, 16), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 3, 16, 16), dtype="float32") = R.zeros(
                    R.shape([16, 3, 16, 16]), dtype="float32"
                )
                lv1: R.Tensor((12288,), dtype="float32") = R.reshape(lv, R.shape([12288]))
                lv2: R.Tensor((12288,), dtype="int64") = R.reshape(I_1, R.shape([12288]))
                lv3: R.Tensor((12288,), dtype="float32") = R.reshape(X, R.shape([12288]))
                lv4: R.Tensor((12288,), dtype="float32") = R.scatter_elements(
                    lv1, lv2, lv3, axis=0, reduction="update"
                )
                gv: R.Tensor((16, 3, 16, 16), dtype="float32") = R.reshape(
                    lv4, R.shape([16, 3, 16, 16])
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedMaxUnpool7:
        @R.function
        def main(
            X: R.Tensor((16, 3, 16, 16), dtype="float32"),
            I_1: R.Tensor((16, 3, 16, 16), dtype="int64"),
        ) -> R.Tensor((16, 3, 31, 31), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((16, 3, 31, 31), dtype="float32") = R.zeros(
                    R.shape([16, 3, 31, 31]), dtype="float32"
                )
                lv1: R.Tensor((46128,), dtype="float32") = R.reshape(lv, R.shape([46128]))
                lv2: R.Tensor((12288,), dtype="int64") = R.reshape(I_1, R.shape([12288]))
                lv3: R.Tensor((12288,), dtype="float32") = R.reshape(X, R.shape([12288]))
                lv4: R.Tensor((46128,), dtype="float32") = R.scatter_elements(
                    lv1, lv2, lv3, axis=0, reduction="update"
                )
                gv: R.Tensor((16, 3, 31, 31), dtype="float32") = R.reshape(
                    lv4, R.shape([16, 3, 31, 31])
                )
                R.output(gv)
            return gv

    verify_maxunpool([2, 2], None, None, ExpectedMaxUnpool0)
    verify_maxunpool([2, 2], None, [2, 2], ExpectedMaxUnpool1)
    verify_maxunpool([2, 2], [1, 1, 1, 1], None, ExpectedMaxUnpool2)
    verify_maxunpool([2, 2], [1, 1, 1, 1], [2, 2], ExpectedMaxUnpool3)
    verify_maxunpool([3, 3], None, None, ExpectedMaxUnpool4)
    verify_maxunpool([3, 3], None, [2, 2], ExpectedMaxUnpool5)
    verify_maxunpool([3, 3], [1, 1, 1, 1], None, ExpectedMaxUnpool6)
    verify_maxunpool([3, 3], [1, 1, 1, 1], [2, 2], ExpectedMaxUnpool7)


def test_dropout():
    def verify_dropout_ir(opset, attrs, expected):
        node = helper.make_node("Dropout", ["x"], ["y"], **attrs)
        graph = helper.make_graph(
            [node],
            "dropout_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 32, 32])],
        )
        model = helper.make_model(
            graph,
            producer_name="dropout_structural_test",
            opset_imports=[helper.make_opsetid("", opset)],
        )
        tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedDropoutRateHalf:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3, 32, 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 32, 32), dtype="float32"),
                    R.Tensor((1, 3, 32, 32), dtype="float32"),
                ) = R.nn.dropout(x, rate=0.5)
                lv1: R.Tensor((1, 3, 32, 32), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 3, 32, 32), dtype="float32") = lv[1]
                gv: R.Tensor((1, 3, 32, 32), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_dropout_ir(14, {}, ExpectedDropoutRateHalf)
    verify_dropout_ir(11, {"ratio": 0.5}, ExpectedDropoutRateHalf)

    # Opset 12+ passes ratio as an optional input; check it is captured into the relax op.
    node = helper.make_node("Dropout", ["x", "ratio"], ["y"])
    graph = helper.make_graph(
        [node],
        "dropout_ratio_input",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        initializer=[helper.make_tensor("ratio", TensorProto.FLOAT, [], [0.3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    model = helper.make_model(graph, producer_name="dropout_ratio_input")
    model.opset_import[0].version = 13
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=False)

    @I.ir_module
    class ExpectedDropoutRatioInput:
        @R.function
        def main(x: R.Tensor((1, 3, 4, 4), dtype="float32")) -> R.Tensor(
            (1, 3, 4, 4), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 4, 4), dtype="float32"),
                    R.Tensor((1, 3, 4, 4), dtype="float32"),
                ) = R.nn.dropout(x, rate=0.30000001192092896)
                lv1: R.Tensor((1, 3, 4, 4), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 3, 4, 4), dtype="float32") = lv[1]
                gv: R.Tensor((1, 3, 4, 4), dtype="float32") = lv1
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedDropoutRatioInput)


@pytest.mark.parametrize(
    "annotation, input_shape, axis, expected_shape",
    [
        ([0, 0], (0, 0), 0, (1, 0)),
        (["N", 0, 3], (5, 0, 3), 3, (0, 1)),
        ([2, 3, 0], (2, 3, 0), 2, (6, 0)),
        ([0, "N", 3], (0, 5, 3), 1, (0, 15)),
        (["N", 0, 3], (5, 0, 3), -2, (5, 0)),
    ],
)
def test_flatten_zero_extent(annotation, input_shape, axis, expected_shape):
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["data"], ["output"], axis=axis)],
        "flatten_zero_extent",
        [helper.make_tensor_value_info("data", TensorProto.FLOAT, annotation)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(expected_shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    mod = from_onnx(model)
    for candidate in [mod, relax.transform.FoldConstant()(mod)]:
        restored = tvm.script.from_source(candidate.script(), extra_vars={"I": I, "R": R, "T": T})
        tvm.ir.assert_structural_equal(candidate, restored)
    inputs = {"data": np.empty(input_shape, dtype="float32")}
    check_correctness(model, inputs, opset=18, check_dtypes=True)


def test_flatten():
    def verify_flatten_ir(axis, output_shape, expected):
        node = helper.make_node("Flatten", ["x"], ["y"], axis=axis)
        graph = helper.make_graph(
            [node],
            "flatten_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )
        model = helper.make_model(graph, producer_name="flatten_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedAxis0:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (1, 3072), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 3072), dtype="float32") = R.reshape(x, R.shape([1, 3072]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedAxisNegative1:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (96, 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((96, 32), dtype="float32") = R.reshape(x, R.shape([96, 32]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedAxis2:
        @R.function
        def main(x: R.Tensor((1, 3, 32, 32), dtype="float32")) -> R.Tensor(
            (3, 1024), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((3, 1024), dtype="float32") = R.reshape(x, R.shape([3, 1024]))
                R.output(gv)
            return gv

    verify_flatten_ir(0, [1, 3072], ExpectedAxis0)
    verify_flatten_ir(-1, [96, 32], ExpectedAxisNegative1)
    verify_flatten_ir(2, [3, 1024], ExpectedAxis2)


def test_flatten_dynamic():
    def verify_flatten_dynamic_ir(axis, expected):
        node = helper.make_node("Flatten", ["x"], ["y"], axis=axis)
        graph = helper.make_graph(
            [node],
            "flatten_dynamic_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "A", "B", 32])],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, None])],
        )
        model = helper.make_model(graph, producer_name="flatten_dynamic_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class ExpectedDynamicAxis0:
        @R.function
        def main(x: R.Tensor((1, A, B, 32), dtype="float32")) -> R.Tensor(
            (1, A * B * 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, A * B * 32), dtype="float32") = R.reshape(
                    x, R.shape([1, A * B * 32])
                )
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class ExpectedDynamicAxisNegative1:
        @R.function
        def main(x: R.Tensor((1, A, B, 32), dtype="float32")) -> R.Tensor(
            (A * B, 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A * B, 32), dtype="float32") = R.reshape(x, R.shape([A * B, 32]))
                R.output(gv)
            return gv

    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class ExpectedDynamicAxis2:
        @R.function
        def main(x: R.Tensor((1, A, B, 32), dtype="float32")) -> R.Tensor(
            (A, B * 32), dtype="float32"
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B * 32), dtype="float32") = R.reshape(x, R.shape([A, B * 32]))
                R.output(gv)
            return gv

    verify_flatten_dynamic_ir(0, ExpectedDynamicAxis0)
    verify_flatten_dynamic_ir(-1, ExpectedDynamicAxisNegative1)
    verify_flatten_dynamic_ir(2, ExpectedDynamicAxis2)


def test_onehot():
    one_hot_node = helper.make_node("OneHot", ["indices", "depth", "values"], ["y"], axis=1)
    graph = helper.make_graph(
        [one_hot_node],
        "one_hot_test",
        inputs=[
            helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2]),
        ],
        initializer=[
            helper.make_tensor("depth", TensorProto.INT64, [], [10]),
            helper.make_tensor("values", TensorProto.FLOAT, [2], [3, 1]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 10, 2])],
    )

    model = helper.make_model(graph, producer_name="one_hot_test")
    values = {
        "indices": np.array([[1, 9], [2, 4]], dtype="int64"),
    }
    check_correctness(model, inputs=values)


def test_onehot_negative_indices():
    one_hot_node = helper.make_node("OneHot", ["indices", "depth", "values"], ["y"], axis=1)
    graph = helper.make_graph(
        [one_hot_node],
        "one_hot_negative_indices_test",
        inputs=[
            helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2]),
        ],
        initializer=[
            helper.make_tensor("depth", TensorProto.INT64, [], [10]),
            helper.make_tensor("values", TensorProto.FLOAT, [2], [3, 1]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 10, 2])],
    )

    model = helper.make_model(graph, producer_name="one_hot_negative_indices_test")
    values = {
        "indices": np.array([[-1, -10], [-2, 4]], dtype="int64"),
    }
    check_correctness(model, inputs=values)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("sorted", [0, 1])
@pytest.mark.parametrize("num_outputs", [1, 2, 3, 4])
def test_unique(axis: int | None, sorted: int, num_outputs: int):
    if num_outputs in [3, 4] and axis is None:
        pytest.xfail("RuntimeError: Check failed: input_shape.size() == size (2 vs. 1)")

    input_shape = [8, 8]
    if axis is None:
        output_shape = [-1]
    else:
        output_shape = [8, 8]
        output_shape[axis] = -1

    output_names = ["y", "indices", "inverse_indices", "counts"][:num_outputs]
    unique_node = helper.make_node("Unique", ["x"], output_names, axis=axis, sorted=sorted)

    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)]
    if num_outputs > 1:
        outputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, [-1]))
    if num_outputs > 2:
        # ONNX spec: inverse_indices is always 1D
        outputs.append(helper.make_tensor_value_info("inverse_indices", TensorProto.INT64, [-1]))
    if num_outputs > 3:
        outputs.append(helper.make_tensor_value_info("counts", TensorProto.INT64, [-1]))

    graph = helper.make_graph(
        [unique_node],
        "unique_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=outputs,
    )
    model = helper.make_model(graph, producer_name="unique_test")
    check_correctness(model)


def test_nonzero():
    def verify_nonzero(shape, expected):
        ndim = max(len(shape), 1)
        node = helper.make_node("NonZero", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "nonzero_structural_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.BOOL, shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.INT64, [ndim, None])],
        )
        model = helper.make_model(graph, producer_name="nonzero_structural_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    nonzero_numbers = T.dynamic("nonzero_numbers", "int64")

    @I.ir_module
    class ExpectedScalar:
        @R.function
        def main(x: R.Tensor((), dtype="bool")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, nonzero_numbers), dtype="int64") = R.match_cast(
                    R.nonzero(x), R.Tensor((1, nonzero_numbers), dtype="int64")
                )
                gv: R.Tensor((1, nonzero_numbers), dtype="int64") = lv
                R.output(gv)
            return gv

    nonzero_numbers = T.dynamic("nonzero_numbers", "int64")

    @I.ir_module
    class ExpectedRank1:
        @R.function
        def main(x: R.Tensor((1,), dtype="bool")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, nonzero_numbers), dtype="int64") = R.match_cast(
                    R.nonzero(x), R.Tensor((1, nonzero_numbers), dtype="int64")
                )
                gv: R.Tensor((1, nonzero_numbers), dtype="int64") = lv
                R.output(gv)
            return gv

    nonzero_numbers = T.dynamic("nonzero_numbers", "int64")

    @I.ir_module
    class ExpectedRank2:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="bool")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, nonzero_numbers), dtype="int64") = R.match_cast(
                    R.nonzero(x), R.Tensor((2, nonzero_numbers), dtype="int64")
                )
                gv: R.Tensor((2, nonzero_numbers), dtype="int64") = lv
                R.output(gv)
            return gv

    nonzero_numbers = T.dynamic("nonzero_numbers", "int64")

    @I.ir_module
    class ExpectedRank3:
        @R.function
        def main(x: R.Tensor((4, 5, 6), dtype="bool")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((3, nonzero_numbers), dtype="int64") = R.match_cast(
                    R.nonzero(x), R.Tensor((3, nonzero_numbers), dtype="int64")
                )
                gv: R.Tensor((3, nonzero_numbers), dtype="int64") = lv
                R.output(gv)
            return gv

    nonzero_numbers = T.dynamic("nonzero_numbers", "int64")

    @I.ir_module
    class ExpectedRank4:
        @R.function
        def main(x: R.Tensor((7, 8, 9, 10), dtype="bool")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((4, nonzero_numbers), dtype="int64") = R.match_cast(
                    R.nonzero(x), R.Tensor((4, nonzero_numbers), dtype="int64")
                )
                gv: R.Tensor((4, nonzero_numbers), dtype="int64") = lv
                R.output(gv)
            return gv

    verify_nonzero((), ExpectedScalar)
    verify_nonzero((1,), ExpectedRank1)
    verify_nonzero((2, 3), ExpectedRank2)
    verify_nonzero((4, 5, 6), ExpectedRank3)
    verify_nonzero((7, 8, 9, 10), ExpectedRank4)


def test_depth_to_space():
    def verify_depth_to_space(mode: Literal["DCR", "CRD"], expected):
        in_shape = [1, 8, 2, 3]
        out_shape = [1, 2, 4, 6]
        blocksize = 2
        node = onnx.helper.make_node(
            "DepthToSpace", inputs=["x"], outputs=["y"], blocksize=blocksize, mode=mode
        )
        graph = helper.make_graph(
            [node],
            "depth_to_space_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        )
        model = helper.make_model(graph, producer_name="depth_to_space_test")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedDCR:
        @R.function
        def main(
            x: R.Tensor((1, 8, 2, 3), dtype="float32"),
        ) -> R.Tensor((1, 2, 4, 6), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2, 2, 3), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 2, 2, 2, 3])
                )
                lv1: R.Tensor((1, 2, 2, 2, 3, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 3, 4, 1, 5, 2]
                )
                gv: R.Tensor((1, 2, 4, 6), dtype="float32") = R.reshape(lv1, R.shape([1, 2, 4, 6]))
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedCRD:
        @R.function
        def main(
            x: R.Tensor((1, 8, 2, 3), dtype="float32"),
        ) -> R.Tensor((1, 2, 4, 6), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2, 2, 3), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 2, 2, 2, 3])
                )
                lv1: R.Tensor((1, 2, 2, 2, 3, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 1, 4, 2, 5, 3]
                )
                gv: R.Tensor((1, 2, 4, 6), dtype="float32") = R.reshape(lv1, R.shape([1, 2, 4, 6]))
                R.output(gv)
            return gv

    verify_depth_to_space("DCR", ExpectedDCR)
    verify_depth_to_space("CRD", ExpectedCRD)


def test_space_to_depth():
    in_shape = [1, 2, 4, 6]
    out_shape = [1, 8, 2, 3]
    blocksize = 2
    node = onnx.helper.make_node("SpaceToDepth", inputs=["x"], outputs=["y"], blocksize=blocksize)
    graph = helper.make_graph(
        [node],
        "space_to_depth_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="space_to_depth_test")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 2, 4, 6), dtype="float32"),
        ) -> R.Tensor((1, 8, 2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2, 3, 2), dtype="float32") = R.reshape(
                    x, R.shape([1, 2, 2, 2, 3, 2])
                )
                lv1: R.Tensor((1, 2, 2, 2, 2, 3), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 3, 5, 1, 2, 4]
                )
                gv: R.Tensor((1, 8, 2, 3), dtype="float32") = R.reshape(lv1, R.shape([1, 8, 2, 3]))
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def construct_sequence(input_shape: list[int], num_tensors: int, name: str = "sequence"):
    inputs = [f"data{i}" for i in range(num_tensors)]
    sequence_construct_node = helper.make_node("SequenceConstruct", inputs, [name])
    graph_inputs = [
        helper.make_tensor_value_info(f"data{i}", TensorProto.FLOAT, input_shape)
        for i in range(num_tensors)
    ]
    return sequence_construct_node, graph_inputs


def make_constant_node(name: str, data_type: int, dims: list[int], vals: list[int]):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=vals),
    )


def make_optional_tensor_value_info(name: str, elem_type: int, shape: list[int]):
    return helper.make_value_info(
        name, helper.make_optional_type_proto(helper.make_tensor_type_proto(elem_type, shape))
    )


def make_optional_sequence_value_info(name: str, elem_type: int, shape: list[int]):
    return helper.make_value_info(
        name,
        helper.make_optional_type_proto(
            helper.make_sequence_type_proto(helper.make_tensor_type_proto(elem_type, shape))
        ),
    )


def test_sequence_construct():
    node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=2)
    graph = helper.make_graph(
        [node],
        "test_sequence_construct",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_construct")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tuple(R.Tensor((32, 32), dtype="float32"), R.Tensor((32, 32), dtype="float32")):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                ) = data0, data1
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_sequence_empty():
    sequence_empty_node = helper.make_node("SequenceEmpty", [], ["sequence"])
    graph = helper.make_graph(
        [sequence_empty_node],
        "test_sequence_empty",
        inputs=[],
        outputs=[helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_empty")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tuple:
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tuple = ()
                R.output(gv)
            return ()

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_sequence_erase():
    def verify_sequence_erase(explicit_position: bool, expected):
        seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
        index = make_constant_node("index", TensorProto.INT64, (), [1])
        node_input = ["sequence", "index"] if explicit_position else ["sequence"]
        sequence_erase_node = helper.make_node("SequenceErase", node_input, ["output"])
        graph = helper.make_graph(
            [index, seq_node, sequence_erase_node],
            "test_sequence_erase",
            inputs=graph_inputs,
            outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [32, 32])],
        )
        model = helper.make_model(graph, producer_name="test_sequence_erase")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedEraseExplicit:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
            data2: R.Tensor((32, 32), dtype="float32"),
            data3: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
        ):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                ) = data0, data2, data3
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedEraseDefault:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
            data2: R.Tensor((32, 32), dtype="float32"),
            data3: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
        ):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                ) = data0, data1, data2
                R.output(gv)
            return gv

    verify_sequence_erase(True, ExpectedEraseExplicit)
    verify_sequence_erase(False, ExpectedEraseDefault)


def test_sequence_insert():
    def verify_sequence_insert(explicit_position: bool, expected):
        seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
        index = make_constant_node("index", TensorProto.INT64, (), [0])
        node_input = ["sequence", "value", "index"] if explicit_position else ["sequence", "value"]
        sequence_insert_node = helper.make_node("SequenceInsert", node_input, ["output"])
        graph = helper.make_graph(
            [index, seq_node, sequence_insert_node],
            "test_sequence_insert",
            inputs=[
                *graph_inputs,
                helper.make_tensor_value_info("value", TensorProto.FLOAT, [32, 32]),
            ],
            outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [32, 32])],
        )
        model = helper.make_model(graph, producer_name="test_sequence_insert")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedInsertExplicit:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
            data2: R.Tensor((32, 32), dtype="float32"),
            data3: R.Tensor((32, 32), dtype="float32"),
            value: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
        ):
            R.func_attr({"num_input": 5})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                ) = value, data0, data1, data2, data3
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedInsertDefault:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
            data2: R.Tensor((32, 32), dtype="float32"),
            data3: R.Tensor((32, 32), dtype="float32"),
            value: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
            R.Tensor((32, 32), dtype="float32"),
        ):
            R.func_attr({"num_input": 5})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                    R.Tensor((32, 32), dtype="float32"),
                ) = data0, data1, data2, data3, value
                R.output(gv)
            return gv

    verify_sequence_insert(True, ExpectedInsertExplicit)
    verify_sequence_insert(False, ExpectedInsertDefault)


def test_concat_from_sequence():
    def verify_concat_from_sequence(new_axis: int, axis: int, expected_shape: list[int], expected):
        seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=2)
        concat_from_sequence_node = helper.make_node(
            "ConcatFromSequence", ["sequence"], ["output"], axis=axis, new_axis=new_axis
        )
        graph = helper.make_graph(
            [seq_node, concat_from_sequence_node],
            "test_concat_from_sequence",
            inputs=graph_inputs,
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, expected_shape)],
        )
        model = helper.make_model(graph, producer_name="test_concat_from_sequence")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedConcatAxis0:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((64, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((64, 32), dtype="float32") = R.concat((data0, data1), axis=0)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedConcatAxis1:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((32, 64), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((32, 64), dtype="float32") = R.concat((data0, data1), axis=1)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedStackAxis0:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((2, 32, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 32, 32), dtype="float32") = R.expand_dims(data0, axis=[0])
                lv1: R.Tensor((1, 32, 32), dtype="float32") = R.expand_dims(data1, axis=[0])
                gv: R.Tensor((2, 32, 32), dtype="float32") = R.concat((lv, lv1), axis=0)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedStackAxis1:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((32, 2, 32), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((32, 1, 32), dtype="float32") = R.expand_dims(data0, axis=[1])
                lv1: R.Tensor((32, 1, 32), dtype="float32") = R.expand_dims(data1, axis=[1])
                gv: R.Tensor((32, 2, 32), dtype="float32") = R.concat((lv, lv1), axis=1)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedStackAxisMinusOne:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((32, 32, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((32, 32, 1), dtype="float32") = R.expand_dims(data0, axis=[-1])
                lv1: R.Tensor((32, 32, 1), dtype="float32") = R.expand_dims(data1, axis=[-1])
                gv: R.Tensor((32, 32, 2), dtype="float32") = R.concat((lv, lv1), axis=-1)
                R.output(gv)
            return gv

    verify_concat_from_sequence(0, 0, [64, 32], ExpectedConcatAxis0)
    verify_concat_from_sequence(0, 1, [32, 64], ExpectedConcatAxis1)
    verify_concat_from_sequence(1, 0, [2, 32, 32], ExpectedStackAxis0)
    verify_concat_from_sequence(1, 1, [32, 2, 32], ExpectedStackAxis1)
    verify_concat_from_sequence(1, -1, [32, 32, 2], ExpectedStackAxisMinusOne)


def test_concat_from_sequence_new_axis_three_tensors():
    """new_axis=1 with three sequence elements (stack then concat along axis)."""
    seq_node, graph_inputs = construct_sequence(input_shape=[16, 8], num_tensors=3)
    concat_node = helper.make_node(
        "ConcatFromSequence", ["sequence"], ["output"], axis=0, new_axis=1
    )
    graph = helper.make_graph(
        [seq_node, concat_node],
        "test_concat_from_sequence_new_axis_three",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 16, 8])],
    )
    model = helper.make_model(graph, producer_name="test_concat_from_sequence_new_axis_three")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data0: R.Tensor((16, 8), dtype="float32"),
            data1: R.Tensor((16, 8), dtype="float32"),
            data2: R.Tensor((16, 8), dtype="float32"),
        ) -> R.Tensor((3, 16, 8), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((1, 16, 8), dtype="float32") = R.expand_dims(data0, axis=[0])
                lv1: R.Tensor((1, 16, 8), dtype="float32") = R.expand_dims(data1, axis=[0])
                lv2: R.Tensor((1, 16, 8), dtype="float32") = R.expand_dims(data2, axis=[0])
                gv: R.Tensor((3, 16, 8), dtype="float32") = R.concat((lv, lv1, lv2), axis=0)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_concat_from_sequence_invalid_new_axis():
    """Verify that new_axis values other than 0 or 1 raise a ValueError."""
    seq_node, graph_inputs = construct_sequence(input_shape=[16, 8], num_tensors=2)
    concat_node = helper.make_node(
        "ConcatFromSequence", ["sequence"], ["output"], axis=0, new_axis=2
    )
    graph = helper.make_graph(
        [seq_node, concat_node],
        "test_concat_from_sequence_invalid_new_axis",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 8])],
    )
    model = helper.make_model(graph, producer_name="test_concat_from_sequence_invalid_new_axis")

    with pytest.raises(ValueError, match="ConcatFromSequence only supports new_axis in"):
        from_onnx(model, opset=11)


def test_split_to_sequence():
    def verify_split_to_sequence(split, data_shape: list[int], output_shape: list[int], expected):
        split_to_sequence_node = helper.make_node(
            "SplitToSequence",
            ["data", "split"],
            ["output"],
            axis=0,
        )
        split_shape = [len(split)] if isinstance(split, list) else ()
        split_node = make_constant_node(
            "split", TensorProto.INT64, split_shape, [split] if isinstance(split, int) else split
        )
        graph = helper.make_graph(
            [split_node, split_to_sequence_node],
            "test_split_to_sequence",
            inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape)],
            outputs=[
                helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, output_shape)
            ],
        )
        model = helper.make_model(graph, producer_name="test_split_to_sequence")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedScalarSplit:
        @R.function
        def main(
            data: R.Tensor((6, 32), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 32), dtype="float32"),
            R.Tensor((2, 32), dtype="float32"),
            R.Tensor((2, 32), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((2, 32), dtype="float32"),
                    R.Tensor((2, 32), dtype="float32"),
                    R.Tensor((2, 32), dtype="float32"),
                ) = R.split(data, indices_or_sections=3, axis=0)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedSectionsSplit:
        @R.function
        def main(
            data: R.Tensor((64, 32), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((16, 32), dtype="float32"),
            R.Tensor((48, 32), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((16, 32), dtype="float32"),
                    R.Tensor((48, 32), dtype="float32"),
                ) = R.split(data, indices_or_sections=[16], axis=0)
                R.output(gv)
            return gv

    verify_split_to_sequence(2, [6, 32], [2, 32], ExpectedScalarSplit)
    verify_split_to_sequence([16, 48], [64, 32], [32, 32], ExpectedSectionsSplit)


def test_sequence_at():
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [1])
    node_input = ["sequence", "index"]
    sequence_at_node = helper.make_node("SequenceAt", node_input, ["output"])
    graph = helper.make_graph(
        [index, seq_node, sequence_at_node],
        "test_sequence_at",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_at")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
            data2: R.Tensor((32, 32), dtype="float32"),
            data3: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((32, 32), dtype="float32"):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = data1
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_get_element_tensor():
    x_shape = [2, 3]
    optional_node = helper.make_node("Optional", ["x"], ["optional"])
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, get_element_node],
        "test_optional_get_element_tensor",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)],
        value_info=[make_optional_tensor_value_info("optional", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_tensor")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = x
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_has_element_tensor():
    x_shape = [2, 3]
    optional_node = helper.make_node("Optional", ["x"], ["optional"])
    has_element_node = helper.make_node("OptionalHasElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, has_element_node],
        "test_optional_has_element_tensor",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
        value_info=[make_optional_tensor_value_info("optional", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_tensor")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((), dtype="bool"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.const(True, "bool")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_has_element_empty():
    x_shape = [2, 3]
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, x_shape)
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    has_element_node = helper.make_node("OptionalHasElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, has_element_node],
        "test_optional_has_element_empty",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
        value_info=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_empty")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), dtype="bool"):
            R.func_attr({"num_input": 0})
            with R.dataflow():
                gv: R.Tensor((), dtype="bool") = R.const(False, "bool")
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_get_element_sequence():
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [1])
    optional_node = helper.make_node("Optional", ["sequence"], ["optional"])
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["unwrapped"])
    sequence_at_node = helper.make_node("SequenceAt", ["unwrapped", "index"], ["output"])
    graph = helper.make_graph(
        [index, seq_node, optional_node, get_element_node, sequence_at_node],
        "test_optional_get_element_sequence",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])],
        value_info=[make_optional_sequence_value_info("optional", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_sequence")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data0: R.Tensor((32, 32), dtype="float32"),
            data1: R.Tensor((32, 32), dtype="float32"),
            data2: R.Tensor((32, 32), dtype="float32"),
            data3: R.Tensor((32, 32), dtype="float32"),
        ) -> R.Tensor((32, 32), dtype="float32"):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                gv: R.Tensor((32, 32), dtype="float32") = data1
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_optional_without_input_requires_type_attr():
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, [2, 3])
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"])
    graph = helper.make_graph(
        [optional_node],
        "test_optional_without_input_requires_type_attr",
        inputs=[],
        outputs=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_without_input_requires_type_attr")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="type attribute"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_empty_optional_graph_output_raises():
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, [2, 3])
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    graph = helper.make_graph(
        [optional_node],
        "test_empty_optional_graph_output_raises",
        inputs=[],
        outputs=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_empty_optional_graph_output_raises")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="Empty optional graph outputs are not supported"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_optional_has_element_requires_one_input():
    has_element_node = helper.make_node("OptionalHasElement", [], ["output"])
    graph = helper.make_graph(
        [has_element_node],
        "test_optional_has_element_requires_one_input",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_requires_one_input")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="expects one input"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_optional_get_element_empty_raises():
    x_shape = [2, 3]
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, x_shape)
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, get_element_node],
        "test_optional_get_element_empty_raises",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)],
        value_info=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_empty_raises")
    model.opset_import[0].version = 18
    with pytest.raises(ValueError, match="empty optional"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_symbolic_shape_deduction():
    def verify_symbolic_shape_deduction(with_reshape_flatten, expected):
        index_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["indices"],
            value=helper.make_tensor("indices", TensorProto.INT64, [], [0]),
        )
        shape_node = helper.make_node("Shape", ["data"], ["shape_output"])
        nodes = [index_node, shape_node]
        gather_input = "shape_output"

        if with_reshape_flatten:
            reshape_node = helper.make_node(
                "Reshape", ["shape_output", "target_shape"], ["reshaped_shape"]
            )
            nodes.append(reshape_node)
            gather_input = "reshaped_shape"

        gather_node = helper.make_node("Gather", [gather_input, "indices"], ["gather_output"])
        unsqueeze_node = helper.make_node(
            "Unsqueeze", ["gather_output", "axes"], ["unsqueeze_output"]
        )
        constant_of_shape_node = helper.make_node(
            "ConstantOfShape",
            ["unsqueeze_output"],
            ["output"],
            value=helper.make_tensor("value", TensorProto.FLOAT, [], [1]),
        )
        nodes.extend([gather_node, unsqueeze_node, constant_of_shape_node])

        initializers = [helper.make_tensor("axes", TensorProto.INT64, [1], vals=[0])]
        if with_reshape_flatten:
            initializers.append(
                helper.make_tensor("target_shape", TensorProto.INT64, [1], vals=[-1])
            )

        graph = helper.make_graph(
            nodes,
            "test_shape_deduction",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, ["batch", "seq"]),
            ],
            initializer=initializers,
            outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [1])],
        )
        model = helper.make_model(graph, producer_name="test_shape_deduction")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(
            tvm_model["main"].without_attr("params"), expected["main"]
        )

    batch = T.dynamic("batch", "int64")
    seq = T.dynamic("seq", "int64")

    @I.ir_module
    class ExpectedWithReshapeFlatten:
        @R.function
        def main(
            data: R.Tensor((batch, seq), dtype="float32"),
            axes: R.Tensor((1,), dtype="int64"),
            target_shape: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((batch,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((batch,), dtype="float32") = R.broadcast_to(
                    R.const(1, "float32"), R.shape([batch])
                )
                R.output(gv)
            return gv

    batch = T.dynamic("batch", "int64")
    seq = T.dynamic("seq", "int64")

    @I.ir_module
    class ExpectedWithoutReshapeFlatten:
        @R.function
        def main(
            data: R.Tensor((batch, seq), dtype="float32"),
            axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((batch,), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((batch,), dtype="float32") = R.broadcast_to(
                    R.const(1, "float32"), R.shape([batch])
                )
                R.output(gv)
            return gv

    verify_symbolic_shape_deduction(False, ExpectedWithoutReshapeFlatten)
    verify_symbolic_shape_deduction(True, ExpectedWithReshapeFlatten)


def test_multi_inputs_with_same_symbolic_shape():
    concat_node = helper.make_node("Concat", ["data1", "data2"], ["output"], axis=1)

    graph = helper.make_graph(
        [concat_node],
        "test_multi_symbolic_shape_input",
        inputs=[
            helper.make_tensor_value_info("data1", TensorProto.FLOAT, ["batch", 1]),
            helper.make_tensor_value_info("data2", TensorProto.FLOAT, ["batch", 1]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
    )
    model = helper.make_model(graph, producer_name="test_multi_symbolic_shape_input")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    batch = T.dynamic("batch", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data1: R.Tensor((batch, 1), dtype="float32"),
            data2: R.Tensor((batch, 1), dtype="float32"),
        ) -> R.Tensor((batch, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((batch, 2), dtype="float32") = R.concat((data1, data2), axis=1)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_multi_ops_with_same_params():
    reshape_node_1 = helper.make_node("Reshape", ["a", "x"], ["b"])
    reshape_node_2 = helper.make_node("Reshape", ["b", "x"], ["c"])

    a_shape = [16]
    output_shape = [1, 16]

    graph = helper.make_graph(
        [reshape_node_1, reshape_node_2],
        "test_multi_ops_with_same_params",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor("x", TensorProto.INT64, [2], output_shape),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="test_multi_ops_with_same_params")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((16,), dtype="float32"),
            v: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((1, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((1, 16), dtype="float32") = R.reshape(a, R.shape([1, 16]))
                gv: R.Tensor((1, 16), dtype="float32") = R.reshape(lv, R.shape([1, 16]))
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_params_names_start_with_onnx():
    reshape_node = helper.make_node("Reshape", ["a", "onnx::x"], ["b"])

    a_shape = [16]
    output_shape = [1, 16]

    graph = helper.make_graph(
        [reshape_node],
        "test_params_names_start_with_onnx",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor("onnx::x", TensorProto.INT64, [2], output_shape),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="test_params_names_start_with_onnx")
    tvm_model = from_onnx(model, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            a: R.Tensor((16,), dtype="float32"),
            v: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((1, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((1, 16), dtype="float32") = R.reshape(a, R.shape([1, 16]))
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize(
    ("initializer_name", "expected_name"),
    [
        ("onnx::weight", "weight"),
        (
            "neck.lateral_convs.2.conv2.weight_quantized",
            "neck.lateral_convs.2.conv2.weight_quantized",
        ),
    ],
)
def test_initializer_name_only_removes_onnx_prefix(initializer_name, expected_name):
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", initializer_name], ["output"])],
        "test_initializer_name_only_removes_onnx_prefix",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        initializer=[numpy_helper.from_array(np.ones([1], dtype="float32"), initializer_name)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8

    tvm_model = from_onnx(
        model,
        keep_params_in_input=True,
        sanitize_input_names=False,
    )

    assert tvm_model["main"].params[-1].name == expected_name


def test_shape_dim_string_expression_graph_add():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A + B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((A, B, A + B), dtype="float32")) -> R.Tensor((A, B, A + B), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A + B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_subtract():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A - B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((A, B, A - B), dtype="float32")) -> R.Tensor((A, B, A - B), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A - B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_mul():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A * B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((A, B, A * B), dtype="float32")) -> R.Tensor((A, B, A * B), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A * B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_div_1():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    # this will result in a floordiv despite not using // since the operands are always int
    x_shape = ["A", "B", "A / B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((A, B, A // B), dtype="float32")) -> R.Tensor((A, B, A // B), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A // B), dtype="float32") = x
                R.output(gv)
            return gv

    # fmt: on

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_div_2():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A // B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    A = T.dynamic("A", "int64")
    B = T.dynamic("B", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((A, B, A // B), dtype="float32")) -> R.Tensor((A, B, A // B), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A // B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@I.ir_module
class ExpectedNMSFiveBoxes:
    @R.function
    def main(
        boxes: R.Tensor((1, 5, 4), dtype="float32"),
        scores: R.Tensor((1, 2, 5), dtype="float32"),
        max_output_boxes_per_class: R.Tensor((1,), dtype="int64"),
        iou_threshold: R.Tensor((1,), dtype="float32"),
        score_threshold: R.Tensor((1,), dtype="float32"),
    ):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv = R.vision.all_class_non_max_suppression(
                boxes,
                scores,
                R.const(3, "int64"),
                R.const(0.5, "float32"),
                R.const(0.10000000149011612, "float32"),
                "onnx",
            )
            lv1 = lv[0]
            gv = lv1
            R.output(gv)
        return gv


@I.ir_module
class ExpectedNMSFourBoxesDefaultParams:
    @R.function
    def main(
        boxes: R.Tensor((1, 4, 4), dtype="float32"),
        scores: R.Tensor((1, 1, 4), dtype="float32"),
    ):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv = R.vision.all_class_non_max_suppression(
                boxes,
                scores,
                R.const(0, "int64"),
                R.const(0.5, "float32"),
                R.const(0.0, "float32"),
                "onnx",
            )
            lv1 = lv[0]
            gv = lv1
            R.output(gv)
        return gv


@I.ir_module
class ExpectedNMSFourBoxesWithMaxParam:
    @R.function
    def main(
        boxes: R.Tensor((1, 4, 4), dtype="float32"),
        scores: R.Tensor((1, 1, 4), dtype="float32"),
        max_output_boxes_per_class: R.Tensor((1,), dtype="int64"),
    ):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv = R.vision.all_class_non_max_suppression(
                boxes,
                scores,
                R.const(0, "int64"),
                R.const(0.5, "float32"),
                R.const(0.0, "float32"),
                "onnx",
            )
            lv1 = lv[0]
            gv = lv1
            R.output(gv)
        return gv


def _assert_nms_import(model, expected, num_params=0):
    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)
    if num_params:
        assert len(tvm_model["main"].attrs["params"]) == num_params
        tvm_model["main"] = tvm_model["main"].without_attr("params")

    tvm_tirx.ir.assert_structural_equal(tvm_model, expected)


def test_nms():
    """NonMaxSuppression should import as all_class_non_max_suppression."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 5, 4]  # batch_size, num_boxes, 4
    scores_shape = [1, 2, 5]  # batch_size, num_classes, num_boxes

    graph = helper.make_graph(
        [nms_node],
        "nms_test",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test")
    model.ir_version = 8
    model.opset_import[0].version = 11

    _assert_nms_import(
        model,
        ExpectedNMSFiveBoxes,
        num_params=3,
    )


def test_nms_scalar_shape1_constants():
    """Scalar params given as 1-D single-element constants must import (NumPy 2.x cast)."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
    )
    graph = helper.make_graph(
        [nms_node],
        "nms_scalar_shape1",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 5, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 5]),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.0]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    # Default import folds initializers to tvm.ir.GenericConst, exercising the scalar-cast path.
    from_onnx(model)


def test_nms_max_output_boxes_per_class_zero():
    """ONNX default for max_output_boxes_per_class should import as 0."""

    def verify(with_explicit_max, expected):
        node_inputs = ["boxes", "scores"]
        initializer = []
        if with_explicit_max:
            node_inputs.append("max_output_boxes_per_class")
            initializer.append(
                helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [0])
            )

        nms_node = helper.make_node(
            "NonMaxSuppression",
            node_inputs,
            ["selected_indices"],
            center_point_box=0,
        )

        boxes_shape = [1, 4, 4]
        scores_shape = [1, 1, 4]
        graph = helper.make_graph(
            [nms_node],
            "nms_max_output_boxes_per_class_zero",
            inputs=[
                helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
                helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
            ],
            initializer=initializer,
            outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
        )

        model = helper.make_model(graph, producer_name="nms_max_output_boxes_per_class_zero")
        model.ir_version = 8
        model.opset_import[0].version = 11

        _assert_nms_import(
            model,
            expected,
            num_params=int(with_explicit_max),
        )

    verify(False, ExpectedNMSFourBoxesDefaultParams)
    verify(True, ExpectedNMSFourBoxesWithMaxParam)


# align_corners=None omits the attribute, exercising the ONNX default of 0.
def test_affine_grid():
    def verify_affine_grid(align_corners, expected):
        attrs = {} if align_corners is None else {"align_corners": align_corners}
        affine_grid_node = helper.make_node("AffineGrid", ["theta", "size"], ["grid"], **attrs)

        graph = helper.make_graph(
            [affine_grid_node],
            "affine_grid_test",
            inputs=[
                helper.make_tensor_value_info("theta", TensorProto.FLOAT, [2, 2, 3]),
            ],
            initializer=[
                helper.make_tensor("size", TensorProto.INT64, [4], [2, 3, 16, 16]),
            ],
            outputs=[
                helper.make_tensor_value_info("grid", TensorProto.FLOAT, [2, 16, 16, 2]),
            ],
        )

        model = helper.make_model(
            graph, producer_name="affine_grid_test", opset_imports=[helper.make_opsetid("", 20)]
        )
        tvm_model = from_onnx(model, opset=20, keep_params_in_input=True)
        assert len(tvm_model["main"].attrs["params"]) == 1
        tvm_model["main"] = tvm_model["main"].without_attr("params")
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedAlignCorners:
        @R.function
        def main(
            theta: R.Tensor((2, 2, 3), dtype="float32"),
            size: R.Tensor((4,), dtype="int64"),
        ) -> R.Tensor((2, 16, 16, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 2, 16, 16), dtype="float32") = R.image.affine_grid(
                    theta, size=(16, 16), align_corners=True
                )
                lv1: R.Tensor((2, 16, 16, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 2, 3, 1]
                )
                gv: R.Tensor((2, 16, 16, 2), dtype="float32") = lv1
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedDefaultAlignCorners:
        @R.function
        def main(
            theta: R.Tensor((2, 2, 3), dtype="float32"),
            size: R.Tensor((4,), dtype="int64"),
        ) -> R.Tensor((2, 16, 16, 2), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 2, 16, 16), dtype="float32") = R.image.affine_grid(
                    theta, size=(16, 16), align_corners=False
                )
                lv1: R.Tensor((2, 16, 16, 2), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 2, 3, 1]
                )
                gv: R.Tensor((2, 16, 16, 2), dtype="float32") = lv1
                R.output(gv)
            return gv

    verify_affine_grid(None, ExpectedDefaultAlignCorners)
    verify_affine_grid(0, ExpectedDefaultAlignCorners)
    verify_affine_grid(1, ExpectedAlignCorners)


def test_affine_grid_3d():
    affine_grid_node = helper.make_node(
        "AffineGrid",
        ["theta", "size"],
        ["grid"],
        align_corners=1,
    )

    graph = helper.make_graph(
        [affine_grid_node],
        "affine_grid_3d_test",
        inputs=[
            helper.make_tensor_value_info("theta", TensorProto.FLOAT, [2, 3, 4]),
        ],
        initializer=[
            helper.make_tensor("size", TensorProto.INT64, [5], [2, 3, 8, 16, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, [2, 8, 16, 16, 3]),
        ],
    )

    model = helper.make_model(graph, producer_name="affine_grid_3d_test")

    tvm_model = from_onnx(model, opset=20, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            theta: R.Tensor((2, 3, 4), dtype="float32"),
            size: R.Tensor((5,), dtype="int64"),
        ) -> R.Tensor((2, 8, 16, 16, 3), dtype="float32"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((2, 3, 8, 16, 16), dtype="float32") = R.image.affine_grid(
                    theta, size=(8, 16, 16), align_corners=True
                )
                lv1: R.Tensor((2, 8, 16, 16, 3), dtype="float32") = R.permute_dims(
                    lv, axes=[0, 2, 3, 4, 1]
                )
                gv: R.Tensor((2, 8, 16, 16, 3), dtype="float32") = lv1
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample(mode, padding_mode, align_corners):
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(
        graph, producer_name="grid_sample_test", opset_imports=[helper.make_opsetid("", 16)]
    )
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    @I.ir_module
    class ExpectedGridSample4D:
        @R.function
        def main(
            X: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 2, 2, 2), dtype="float32"),
        ) -> R.Tensor((1, 3, 2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2), dtype="float32") = R.permute_dims(
                    grid, axes=[0, 3, 1, 2]
                )
                gv: R.Tensor((1, 3, 2, 2), dtype="float32") = R.image.grid_sample(
                    X,
                    lv,
                    method=mode,
                    layout="NCHW",
                    padding_mode=padding_mode,
                    align_corners=bool(align_corners),
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedGridSample4D)


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample_5d(mode, padding_mode, align_corners):
    x_shape = [1, 1, 4, 4, 4]
    grid_shape = [1, 4, 4, 4, 3]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_5d_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(
        graph, producer_name="grid_sample_5d_test", opset_imports=[helper.make_opsetid("", 16)]
    )
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    @I.ir_module
    class ExpectedGridSample5D:
        @R.function
        def main(
            X: R.Tensor((1, 1, 4, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 4, 4, 4, 3), dtype="float32"),
        ) -> R.Tensor((1, 1, 4, 4, 4), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4, 4), dtype="float32") = R.permute_dims(
                    grid, axes=[0, 4, 1, 2, 3]
                )
                gv: R.Tensor((1, 1, 4, 4, 4), dtype="float32") = R.image.grid_sample(
                    X,
                    lv,
                    method=mode,
                    layout="NCDHW",
                    padding_mode=padding_mode,
                    align_corners=bool(align_corners),
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, ExpectedGridSample5D)


def test_grid_sample_5d_cubic_unsupported():
    x_shape = [1, 1, 4, 4, 4]
    grid_shape = [1, 2, 3, 5, 3]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="cubic",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_5d_cubic_unsupported_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_5d_cubic_unsupported_test")
    with pytest.raises(
        NotImplementedError,
        match="5D .*GridSample with mode='cubic' is not supported",
    ):
        from_onnx(model, opset=16, keep_params_in_input=True)


def test_grid_sample_4d_non_square_output_shape():
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 3, 5, 2]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="bilinear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_4d_non_square_output_shape_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_4d_non_square_output_shape_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            X: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 3, 5, 2), dtype="float32"),
        ) -> R.Tensor((1, 3, 3, 5), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 3, 5), dtype="float32") = R.permute_dims(
                    grid, axes=[0, 3, 1, 2]
                )
                gv: R.Tensor((1, 3, 3, 5), dtype="float32") = R.image.grid_sample(
                    X,
                    lv,
                    method="bilinear",
                    layout="NCHW",
                    padding_mode="zeros",
                    align_corners=False,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_grid_sample_unsupported_rank():
    x_shape = [1, 3, 4]
    grid_shape = [1, 4, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="bilinear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_unsupported_rank_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, x_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_unsupported_rank_test")
    with pytest.raises(NotImplementedError, match="GridSample only supports 4D or 5D input"):
        from_onnx(model, opset=16, keep_params_in_input=True)


def test_grid_sample_linear_mode_translation():
    """Test that ONNX mode='linear' is correctly translated to 'bilinear'.

    The ONNX spec defines 'linear' as a valid mode for GridSample, but
    onnxruntime rejects it in practice. Real ONNX models exported from
    frameworks like PyTorch may still use 'linear'. We verify the translation
    by inspecting the Relax IR directly rather than running check_correctness.
    """
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="linear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_linear_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]
            ),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_linear_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            X: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 2, 2, 2), dtype="float32"),
        ) -> R.Tensor((1, 3, 2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2), dtype="float32") = R.permute_dims(
                    grid, axes=[0, 3, 1, 2]
                )
                gv: R.Tensor((1, 3, 2, 2), dtype="float32") = R.image.grid_sample(
                    X,
                    lv,
                    method="bilinear",
                    layout="NCHW",
                    padding_mode="zeros",
                    align_corners=False,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_grid_sample_cubic_mode_translation():
    """Test that ONNX mode='cubic' is correctly translated to 'bicubic'.

    The ONNX spec defines 'cubic' as a valid mode for GridSample, but
    TVM uses 'bicubic'. We verify the translation by inspecting the
    Relax IR directly rather than running check_correctness.
    """
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="cubic",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_cubic_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]
            ),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_cubic_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            X: R.Tensor((1, 3, 4, 4), dtype="float32"),
            grid: R.Tensor((1, 2, 2, 2), dtype="float32"),
        ) -> R.Tensor((1, 3, 2, 2), dtype="float32"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv: R.Tensor((1, 2, 2, 2), dtype="float32") = R.permute_dims(
                    grid, axes=[0, 3, 1, 2]
                )
                gv: R.Tensor((1, 3, 2, 2), dtype="float32") = R.image.grid_sample(
                    X,
                    lv,
                    method="bicubic",
                    layout="NCHW",
                    padding_mode="zeros",
                    align_corners=False,
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_roi_align():
    def verify_roi_align(coordinate_transformation_mode, rois, expected):
        x_shape = [1, 4, 8, 8]
        rois_shape = list(rois.shape)
        batch_indices_shape = [2]
        out_shape = [2, 4, 3, 3]

        node = helper.make_node(
            "RoiAlign",
            inputs=["X", "rois", "batch_indices"],
            outputs=["Y"],
            output_height=3,
            output_width=3,
            sampling_ratio=2,
            spatial_scale=1.0,
            mode="avg",
            coordinate_transformation_mode=coordinate_transformation_mode,
        )

        graph = helper.make_graph(
            [node],
            "roi_align_test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
                helper.make_tensor_value_info("rois", TensorProto.FLOAT, rois_shape),
                helper.make_tensor_value_info(
                    "batch_indices", TensorProto.INT64, batch_indices_shape
                ),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name="roi_align_test")
        tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedRoiAlignHalfPixel:
        @R.function
        def main(
            X: R.Tensor((1, 4, 8, 8), dtype="float32"),
            rois: R.Tensor((2, 4), dtype="float32"),
            batch_indices: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((2, 4, 3, 3), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((2, 1), dtype="int64") = R.expand_dims(batch_indices, axis=1)
                lv1: R.Tensor((2, 1), dtype="float32") = R.astype(lv, dtype="float32")
                lv2: R.Tensor((2, 4), dtype="float32") = R.add(
                    rois, R.const([-0.5, -0.5, -0.5, -0.5], "float32")
                )
                lv3: R.Tensor((2, 5), dtype="float32") = R.concat((lv1, lv2), axis=1)
                gv: R.Tensor((2, 4, 3, 3), dtype="float32") = R.vision.roi_align(
                    X,
                    lv3,
                    pooled_size=(3, 3),
                    spatial_scale=1.0,
                    sample_ratio=2,
                    aligned=True,
                    layout="NCHW",
                    mode="avg",
                )
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedRoiAlignOutputHalfPixel:
        @R.function
        def main(
            X: R.Tensor((1, 4, 8, 8), dtype="float32"),
            rois: R.Tensor((2, 4), dtype="float32"),
            batch_indices: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor((2, 4, 3, 3), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv: R.Tensor((2, 1), dtype="int64") = R.expand_dims(batch_indices, axis=1)
                lv1: R.Tensor((2, 1), dtype="float32") = R.astype(lv, dtype="float32")
                lv2: R.Tensor((2, 5), dtype="float32") = R.concat((lv1, rois), axis=1)
                gv: R.Tensor((2, 4, 3, 3), dtype="float32") = R.vision.roi_align(
                    X,
                    lv2,
                    pooled_size=(3, 3),
                    spatial_scale=1.0,
                    sample_ratio=2,
                    aligned=False,
                    layout="NCHW",
                    mode="avg",
                )
                R.output(gv)
            return gv

    verify_roi_align(
        "output_half_pixel",
        np.array([[1.0, 1.0, 6.0, 6.0], [2.0, 0.5, 7.0, 7.0]], dtype="float32"),
        ExpectedRoiAlignOutputHalfPixel,
    )
    verify_roi_align(
        "half_pixel",
        np.array([[1.0, 1.0, 1.2, 1.2], [2.0, 0.5, 1.1, 1.1]], dtype="float32"),
        ExpectedRoiAlignHalfPixel,
    )


def test_if():
    """Test ONNX If operator with scalar and tensor bool conditions."""

    def verify_if(cond_info, expected):
        x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
        result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

        two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
        then_mul = helper.make_node("Mul", ["x", "two"], ["then_out"])
        then_out_info = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [3])
        then_graph = helper.make_graph(
            [then_mul], "then_graph", [], [then_out_info], initializer=[two]
        )

        three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
        else_mul = helper.make_node("Mul", ["x", "three"], ["else_out"])
        else_out_info = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [3])
        else_graph = helper.make_graph(
            [else_mul], "else_graph", [], [else_out_info], initializer=[three]
        )

        if_node = helper.make_node(
            "If",
            inputs=["cond"],
            outputs=["result"],
            then_branch=then_graph,
            else_branch=else_graph,
        )
        main_graph = helper.make_graph([if_node], "if_test", [cond_info, x_info], [result_info])
        model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedScalarCondition:
        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"),
            x: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 2})
            if cond:
                gv: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                gv2: R.Tensor((3,), dtype="float32") = gv
            else:
                gv1: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                gv2: R.Tensor((3,), dtype="float32") = gv1
            return gv2

    @I.ir_module
    class ExpectedTensorCondition:
        @R.function
        def main(
            cond: R.Tensor((1,), dtype="bool"),
            x: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 2})
            if cond:
                gv: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                gv2: R.Tensor((3,), dtype="float32") = gv
            else:
                gv1: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                gv2: R.Tensor((3,), dtype="float32") = gv1
            return gv2

    verify_if(helper.make_tensor_value_info("cond", TensorProto.BOOL, []), ExpectedScalarCondition)
    verify_if(helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]), ExpectedTensorCondition)


def test_if_computed_condition():
    """Test If where condition is computed from another op in the main graph."""
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    reduce_node = helper.make_node(
        "ReduceSum", ["x"], ["x_sum"], keepdims=0, noop_with_empty_axes=0
    )
    greater_node = helper.make_node("Greater", ["x_sum", "zero"], ["cond"])

    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    then_mul = helper.make_node("Mul", ["x", "two"], ["then_out"])
    then_out_info = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [3])
    then_graph = helper.make_graph([then_mul], "then_graph", [], [then_out_info], initializer=[two])

    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
    else_mul = helper.make_node("Mul", ["x", "three"], ["else_out"])
    else_out_info = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [3])
    else_graph = helper.make_graph(
        [else_mul], "else_graph", [], [else_out_info], initializer=[three]
    )

    if_node = helper.make_node(
        "If", inputs=["cond"], outputs=["result"], then_branch=then_graph, else_branch=else_graph
    )

    main_graph = helper.make_graph(
        [reduce_node, greater_node, if_node],
        "if_computed_cond",
        [x_info],
        [result_info],
        initializer=[zero],
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])

    tvm_model = from_onnx(model, keep_params_in_input=True)
    assert len(tvm_model["main"].attrs["params"]) == 1
    tvm_model["main"] = tvm_model["main"].without_attr("params")

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3,), dtype="float32"),
            zer: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 1})
            gv: R.Tensor((), dtype="float32") = R.sum(x, axis=None, keepdims=False)
            gv1: R.Tensor((), dtype="bool") = R.greater(gv, zer)
            if gv1:
                gv2: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                gv4: R.Tensor((3,), dtype="float32") = gv2
            else:
                gv3: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                gv4: R.Tensor((3,), dtype="float32") = gv3
            return gv4

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_if_multiple_outputs():
    """Test If operator where branches return multiple outputs."""
    cond_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    out1_info = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [3])
    out2_info = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [3])

    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])

    then_mul1 = helper.make_node("Mul", ["x", "two"], ["then_out1"])
    then_mul2 = helper.make_node("Mul", ["x", "three"], ["then_out2"])
    then_o1 = helper.make_tensor_value_info("then_out1", TensorProto.FLOAT, [3])
    then_o2 = helper.make_tensor_value_info("then_out2", TensorProto.FLOAT, [3])
    then_graph = helper.make_graph(
        [then_mul1, then_mul2], "then_graph", [], [then_o1, then_o2], initializer=[two, three]
    )

    four = helper.make_tensor("four", TensorProto.FLOAT, [1], [4.0])
    five = helper.make_tensor("five", TensorProto.FLOAT, [1], [5.0])
    else_mul1 = helper.make_node("Mul", ["x", "four"], ["else_out1"])
    else_mul2 = helper.make_node("Mul", ["x", "five"], ["else_out2"])
    else_o1 = helper.make_tensor_value_info("else_out1", TensorProto.FLOAT, [3])
    else_o2 = helper.make_tensor_value_info("else_out2", TensorProto.FLOAT, [3])
    else_graph = helper.make_graph(
        [else_mul1, else_mul2], "else_graph", [], [else_o1, else_o2], initializer=[four, five]
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["out1", "out2"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    main_graph = helper.make_graph(
        [if_node], "if_multi_out", [cond_info, x_info], [out1_info, out2_info]
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])

    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"),
            x: R.Tensor((3,), dtype="float32"),
        ) -> R.Tuple(R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")):
            R.func_attr({"num_input": 2})
            if cond:
                gv: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                gv1: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                gv4: R.Tuple(
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = gv, gv1
            else:
                gv2: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([4.0], "float32"))
                gv3: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([5.0], "float32"))
                gv4: R.Tuple(
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = gv2, gv3
            gv5: R.Tensor((3,), dtype="float32") = gv4[0]
            gv6: R.Tensor((3,), dtype="float32") = gv4[1]
            return (gv5, gv6)

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_if_nested():
    """Test nested If operator inside a branch."""
    cond1_info = helper.make_tensor_value_info("cond1", TensorProto.BOOL, [])
    cond2_info = helper.make_tensor_value_info("cond2", TensorProto.BOOL, [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

    # Inner then: x * 2
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    inner_then_mul = helper.make_node("Mul", ["x", "two"], ["inner_then_out"])
    inner_then_out_info = helper.make_tensor_value_info("inner_then_out", TensorProto.FLOAT, [3])
    inner_then_graph = helper.make_graph(
        [inner_then_mul], "inner_then", [], [inner_then_out_info], initializer=[two]
    )

    # Inner else: x * 3
    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
    inner_else_mul = helper.make_node("Mul", ["x", "three"], ["inner_else_out"])
    inner_else_out_info = helper.make_tensor_value_info("inner_else_out", TensorProto.FLOAT, [3])
    inner_else_graph = helper.make_graph(
        [inner_else_mul], "inner_else", [], [inner_else_out_info], initializer=[three]
    )

    # Outer then: nested If(cond2, x*2, x*3)
    inner_if = helper.make_node(
        "If",
        inputs=["cond2"],
        outputs=["outer_then_out"],
        then_branch=inner_then_graph,
        else_branch=inner_else_graph,
    )
    outer_then_out_info = helper.make_tensor_value_info("outer_then_out", TensorProto.FLOAT, [3])
    outer_then_graph = helper.make_graph([inner_if], "outer_then", [], [outer_then_out_info])

    # Outer else: x * 4
    four = helper.make_tensor("four", TensorProto.FLOAT, [1], [4.0])
    outer_else_mul = helper.make_node("Mul", ["x", "four"], ["outer_else_out"])
    outer_else_out_info = helper.make_tensor_value_info("outer_else_out", TensorProto.FLOAT, [3])
    outer_else_graph = helper.make_graph(
        [outer_else_mul], "outer_else", [], [outer_else_out_info], initializer=[four]
    )

    outer_if = helper.make_node(
        "If",
        inputs=["cond1"],
        outputs=["result"],
        then_branch=outer_then_graph,
        else_branch=outer_else_graph,
    )
    main_graph = helper.make_graph(
        [outer_if], "nested_if", [cond1_info, cond2_info, x_info], [result_info]
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            cond1: R.Tensor((), dtype="bool"),
            cond2: R.Tensor((), dtype="bool"),
            x: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((3,), dtype="float32"):
            R.func_attr({"num_input": 3})
            if cond2:
                gv: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([2.0], "float32"))
                gv2: R.Tensor((3,), dtype="float32") = gv
            else:
                gv1: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([3.0], "float32"))
                gv2: R.Tensor((3,), dtype="float32") = gv1
            if cond1:
                gv4: R.Tensor((3,), dtype="float32") = gv2
            else:
                gv3: R.Tensor((3,), dtype="float32") = R.multiply(x, R.const([4.0], "float32"))
                gv4: R.Tensor((3,), dtype="float32") = gv3
            return gv4

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_if_subgraph():
    """Test If subgraph."""
    input_tensor_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 3])
    cond_tensor_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    y_tensor_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 3, 3])
    b_tensor_info = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 1, 3, 3])
    c_tensor_info = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 1, 3, 3])
    unsqueeze_axes_tensor = helper.make_tensor(
        name="unsqueeze_axes", data_type=TensorProto.INT64, dims=[1], vals=[0]
    )
    unsqueeze_then_node = helper.make_node(
        "Unsqueeze", inputs=["input", "unsqueeze_axes"], outputs=["input_unsqueezed_then"]
    )
    then_out_info = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [1, 1, 3, 3])
    then_node = helper.make_node(
        "Conv",
        inputs=["input_unsqueezed_then", "B"],
        outputs=["then_out"],
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    then_graph = helper.make_graph(
        nodes=[unsqueeze_then_node, then_node],
        name="then_branch_graph",
        inputs=[],
        outputs=[then_out_info],
    )
    unsqueeze_else_node = helper.make_node(
        "Unsqueeze", inputs=["input", "unsqueeze_axes"], outputs=["input_unsqueezed_else"]
    )
    else_out_info = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [1, 1, 3, 3])
    else_node = helper.make_node(
        "Conv",
        inputs=["input_unsqueezed_else", "C"],
        outputs=["else_out"],
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    else_graph = helper.make_graph(
        nodes=[unsqueeze_else_node, else_node],
        name="else_branch_graph",
        inputs=[],
        outputs=[else_out_info],
    )

    if_node = helper.make_node(
        "If", inputs=["cond"], outputs=["Y"], then_branch=then_graph, else_branch=else_graph
    )
    outer_graph = helper.make_graph(
        nodes=[if_node],
        name="CondSubgraph",
        inputs=[cond_tensor_info, input_tensor_info, b_tensor_info, c_tensor_info],
        outputs=[y_tensor_info],
        initializer=[unsqueeze_axes_tensor],
    )
    opset_imports = [helper.make_operatorsetid("", 15)]
    model = helper.make_model(
        outer_graph, producer_name="condsubgraph", opset_imports=opset_imports
    )

    tvm_model = from_onnx(model, keep_params_in_input=True)
    tvm_model, _ = tvm.relax.frontend.detach_params(tvm_model)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"),
            input: R.Tensor((1, 3, 3), dtype="float32"),
            B: R.Tensor((1, 1, 3, 3), dtype="float32"),
            C: R.Tensor((1, 1, 3, 3), dtype="float32"),
            unsqueeze_axes: R.Tensor((1,), dtype="int64"),
        ) -> R.Tensor((1, 1, 3, 3), dtype="float32"):
            R.func_attr({"num_input": 4})
            gv: R.Tensor((1, 1, 3, 3), dtype="float32") = R.expand_dims(input, axis=[0])
            gv1: R.Tensor((1, 1, 3, 3), dtype="float32") = R.expand_dims(input, axis=[0])
            if cond:
                gv2: R.Tensor((1, 1, 3, 3), dtype="float32") = R.nn.conv2d(
                    gv,
                    B,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype=None,
                )
                gv4: R.Tensor((1, 1, 3, 3), dtype="float32") = gv2
            else:
                gv3: R.Tensor((1, 1, 3, 3), dtype="float32") = R.nn.conv2d(
                    gv1,
                    C,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype=None,
                )
                gv4: R.Tensor((1, 1, 3, 3), dtype="float32") = gv3
            return gv4

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


# Helper that builds the ONNX graph for MatMulInteger so the tests don't repeat boilerplate code every time
def _make_matmulinteger_model(A_shape, B_shape, A_dtype, B_dtype, a_zp_array=None, b_zp_array=None):
    """Build a minimal single-node ONNX graph for MatMulInteger."""

    def np_dtype_to_onnx(dt):
        return {np.int8: TensorProto.INT8, np.uint8: TensorProto.UINT8}[dt]

    A_info = helper.make_tensor_value_info("A", np_dtype_to_onnx(A_dtype), A_shape)
    B_info = helper.make_tensor_value_info("B", np_dtype_to_onnx(B_dtype), B_shape)
    graph_inputs = [A_info, B_info]
    node_inputs = ["A", "B"]
    initializers = []

    def _add_zp(name, arr, dtype):
        onnx_dtype = np_dtype_to_onnx(dtype)
        shape = list(arr.shape)
        initializers.append(helper.make_tensor(name, onnx_dtype, shape, arr.flatten().tolist()))
        node_inputs.append(name)

    if a_zp_array is not None:
        _add_zp("a_zero_point", a_zp_array, A_dtype)
    elif b_zp_array is not None:
        node_inputs.append("")  # placeholder only needed if b_zp is present

    if b_zp_array is not None:
        _add_zp("b_zero_point", b_zp_array, B_dtype)

    out_info = helper.make_tensor_value_info("output", TensorProto.INT32, None)
    node = helper.make_node("MatMulInteger", inputs=node_inputs, outputs=["output"])
    graph = helper.make_graph(
        [node], "matmulinteger", graph_inputs, [out_info], initializer=initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 8
    return model


def verify_matmulinteger_ir(A_shape, B_shape, A_dtype, B_dtype, expected, a_zp=None, b_zp=None):
    model = _make_matmulinteger_model(
        A_shape,
        B_shape,
        A_dtype,
        B_dtype,
        a_zp_array=np.array(a_zp, dtype=A_dtype) if a_zp is not None else None,
        b_zp_array=np.array(b_zp, dtype=B_dtype) if b_zp is not None else None,
    )
    tvm_model = from_onnx(model, opset=10, keep_params_in_input=True)
    if a_zp is not None or b_zp is not None:
        assert len(tvm_model["main"].attrs["params"]) == 2
        tvm_model["main"] = tvm_model["main"].without_attr("params")
    tvm_tirx.ir.assert_structural_equal(tvm_model, expected)


def test_matmulinteger():
    """2-D MatMulInteger should import dtype casts and zero-point subtraction."""

    @I.ir_module
    class ExpectedInt8:
        @R.function
        def main(
            A: R.Tensor((4, 8), dtype="int8"),
            B: R.Tensor((8, 6), dtype="int8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(B, dtype="int32")
                gv = R.matmul(lv, lv1, out_dtype="int32")
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedUInt8:
        @R.function
        def main(
            A: R.Tensor((4, 8), dtype="uint8"),
            B: R.Tensor((8, 6), dtype="uint8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(B, dtype="int32")
                gv = R.matmul(lv, lv1, out_dtype="int32")
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedUInt8Int8:
        @R.function
        def main(
            A: R.Tensor((4, 8), dtype="uint8"),
            B: R.Tensor((8, 6), dtype="int8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(B, dtype="int32")
                gv = R.matmul(lv, lv1, out_dtype="int32")
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedInt8UInt8:
        @R.function
        def main(
            A: R.Tensor((4, 8), dtype="int8"),
            B: R.Tensor((8, 6), dtype="uint8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(B, dtype="int32")
                gv = R.matmul(lv, lv1, out_dtype="int32")
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedUInt8ScalarZeroPoints:
        @R.function
        def main(
            A: R.Tensor((4, 8), dtype="uint8"),
            B: R.Tensor((8, 6), dtype="uint8"),
            a_zero_point: R.Tensor((), dtype="uint8"),
            b_zero_point: R.Tensor((), dtype="uint8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(a_zero_point, dtype="int32")
                lv2 = R.subtract(lv, lv1)
                lv3 = R.astype(B, dtype="int32")
                lv4 = R.astype(b_zero_point, dtype="int32")
                lv5 = R.subtract(lv3, lv4)
                gv = R.matmul(lv2, lv5, out_dtype="int32")
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedInt8ScalarZeroPoints:
        @R.function
        def main(
            A: R.Tensor((4, 8), dtype="int8"),
            B: R.Tensor((8, 6), dtype="int8"),
            a_zero_point: R.Tensor((), dtype="int8"),
            b_zero_point: R.Tensor((), dtype="int8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(a_zero_point, dtype="int32")
                lv2 = R.subtract(lv, lv1)
                lv3 = R.astype(B, dtype="int32")
                lv4 = R.astype(b_zero_point, dtype="int32")
                lv5 = R.subtract(lv3, lv4)
                gv = R.matmul(lv2, lv5, out_dtype="int32")
                R.output(gv)
            return gv

    verify_matmulinteger_ir([4, 8], [8, 6], np.int8, np.int8, ExpectedInt8)
    verify_matmulinteger_ir([4, 8], [8, 6], np.uint8, np.uint8, ExpectedUInt8)
    verify_matmulinteger_ir([4, 8], [8, 6], np.uint8, np.int8, ExpectedUInt8Int8)
    verify_matmulinteger_ir([4, 8], [8, 6], np.int8, np.uint8, ExpectedInt8UInt8)
    verify_matmulinteger_ir(
        [4, 8],
        [8, 6],
        np.uint8,
        np.uint8,
        ExpectedUInt8ScalarZeroPoints,
        a_zp=np.uint8(128),
        b_zp=np.uint8(128),
    )
    verify_matmulinteger_ir(
        [4, 8],
        [8, 6],
        np.int8,
        np.int8,
        ExpectedInt8ScalarZeroPoints,
        a_zp=np.int8(1),
        b_zp=np.int8(2),
    )


def test_matmulinteger_batched():
    """Batched MatMulInteger should import as batched Relax matmul."""

    @I.ir_module
    class ExpectedBatched3D:
        @R.function
        def main(
            A: R.Tensor((2, 4, 8), dtype="int8"),
            B: R.Tensor((2, 8, 6), dtype="int8"),
            a_zero_point: R.Tensor((), dtype="int8"),
            b_zero_point: R.Tensor((), dtype="int8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(a_zero_point, dtype="int32")
                lv2 = R.subtract(lv, lv1)
                lv3 = R.astype(B, dtype="int32")
                lv4 = R.astype(b_zero_point, dtype="int32")
                lv5 = R.subtract(lv3, lv4)
                gv = R.matmul(lv2, lv5, out_dtype="int32")
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedBatched4D:
        @R.function
        def main(
            A: R.Tensor((2, 3, 4, 8), dtype="int8"),
            B: R.Tensor((2, 3, 8, 6), dtype="int8"),
            a_zero_point: R.Tensor((), dtype="int8"),
            b_zero_point: R.Tensor((), dtype="int8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(a_zero_point, dtype="int32")
                lv2 = R.subtract(lv, lv1)
                lv3 = R.astype(B, dtype="int32")
                lv4 = R.astype(b_zero_point, dtype="int32")
                lv5 = R.subtract(lv3, lv4)
                gv = R.matmul(lv2, lv5, out_dtype="int32")
                R.output(gv)
            return gv

    verify_matmulinteger_ir(
        [2, 4, 8],
        [2, 8, 6],
        np.int8,
        np.int8,
        ExpectedBatched3D,
        a_zp=np.int8(1),
        b_zp=np.int8(2),
    )
    verify_matmulinteger_ir(
        [2, 3, 4, 8],
        [2, 3, 8, 6],
        np.int8,
        np.int8,
        ExpectedBatched4D,
        a_zp=np.int8(1),
        b_zp=np.int8(2),
    )


def test_matmulinteger_per_channel_zp():
    """1-D zero points should expand for per-row/per-column MatMulInteger."""

    @I.ir_module
    class ExpectedPerChannelZeroPoints:
        @R.function
        def main(
            A: R.Tensor((4, 8), dtype="int8"),
            B: R.Tensor((8, 6), dtype="int8"),
            a_zero_point: R.Tensor((4,), dtype="int8"),
            b_zero_point: R.Tensor((6,), dtype="int8"),
        ):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv = R.astype(A, dtype="int32")
                lv1 = R.astype(a_zero_point, dtype="int32")
                lv2 = R.expand_dims(lv1, axis=-1)
                lv3 = R.subtract(lv, lv2)
                lv4 = R.astype(B, dtype="int32")
                lv5 = R.astype(b_zero_point, dtype="int32")
                lv6 = R.expand_dims(lv5, axis=0)
                lv7 = R.subtract(lv4, lv6)
                gv = R.matmul(lv3, lv7, out_dtype="int32")
                R.output(gv)
            return gv

    verify_matmulinteger_ir(
        [4, 8],
        [8, 6],
        np.int8,
        np.int8,
        ExpectedPerChannelZeroPoints,
        a_zp=np.arange(4, dtype=np.int8),
        b_zp=np.arange(6, dtype=np.int8),
    )


@pytest.mark.parametrize(
    ("pooled_shape", "rois"),
    [
        ((1, 1), np.array([[0.0, 1.0, 1.0, 6.0, 6.0], [0.0, 0.0, 0.0, 7.0, 7.0]], dtype="float32")),
        (
            (2, 3),
            np.array([[0.0, 1.2, 0.5, 6.8, 7.0], [0.0, -1.0, 2.0, 3.5, 5.2]], dtype="float32"),
        ),
        (
            (2, 2),
            np.array(
                [[0.0, 100.0, 100.0, 110.0, 110.0], [0.0, 1.0, 1.0, 6.0, 6.0]], dtype="float32"
            ),
        ),
    ],
)
def test_max_roi_pool(pooled_shape, rois):
    x_shape = [1, 4, 8, 8]
    out_shape = [2, 4, pooled_shape[0], pooled_shape[1]]

    node = helper.make_node(
        "MaxRoiPool",
        inputs=["X", "rois"],
        outputs=["Y"],
        pooled_shape=pooled_shape,
        spatial_scale=1.0,
    )

    graph = helper.make_graph(
        [node],
        "max_roi_pool_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, [2, 5]),
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="max_roi_pool_test")
    inputs = {
        "X": rg.standard_normal(size=x_shape).astype("float32"),
        "rois": rois,
    }
    check_correctness(model, inputs=inputs, opset=16, rtol=1e-5, atol=1e-5)


def test_arg_min_max_select_last_index():
    """select_last_index=1 should lower to flip + argreduce + index remap."""

    def verify_select_last_index(op_name, axis, keepdims, expected):
        shape = [3, 4, 5]
        node = helper.make_node(
            op_name,
            inputs=["data"],
            outputs=["out"],
            axis=axis,
            keepdims=int(keepdims),
            select_last_index=1,
        )

        out_shape = list(shape)
        if keepdims:
            out_shape[axis] = 1
        else:
            out_shape.pop(axis)

        graph = helper.make_graph(
            [node],
            "arg_select_last_index_test",
            inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, out_shape)],
        )
        model = helper.make_model(graph, producer_name="arg_select_last_index_test")
        tvm_model = from_onnx(model, opset=12, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    def make_expected(op_name, axis, keepdims):
        axis_extent = [3, 4, 5][axis] - 1
        reduce_op = R.argmax if op_name == "ArgMax" else R.argmin

        @I.ir_module
        class ExpectedArgReduceSelectLast:
            @R.function
            def main(
                data: R.Tensor((3, 4, 5), dtype="float32"),
            ):
                R.func_attr({"num_input": 1})
                with R.dataflow():
                    lv: R.Tensor((3, 4, 5), dtype="float32") = R.flip(data, axis=axis)
                    lv1 = reduce_op(lv, axis=axis, keepdims=keepdims)
                    gv = R.subtract(R.const(axis_extent, "int64"), lv1)
                    R.output(gv)
                return gv

        return ExpectedArgReduceSelectLast

    for op_name in ["ArgMax", "ArgMin"]:
        for axis in [0, 1, 2]:
            for keepdims in [True, False]:
                verify_select_last_index(
                    op_name, axis, keepdims, make_expected(op_name, axis, keepdims)
                )


def test_arg_min_max_select_last_index_no_tie():
    """select_last_index=0 should keep direct argreduce lowering."""

    def verify_no_tie(op_name, expected):
        shape = [4, 5]
        node = helper.make_node(
            op_name,
            inputs=["data"],
            outputs=["out"],
            axis=1,
            keepdims=1,
            select_last_index=0,
        )
        graph = helper.make_graph(
            [node],
            "arg_no_tie_test",
            inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, [4, 1])],
        )
        model = helper.make_model(graph, producer_name="arg_no_tie_test")
        tvm_model = from_onnx(model, opset=12, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedArgMax:
        @R.function
        def main(
            data: R.Tensor((4, 5), dtype="float32"),
        ) -> R.Tensor((4, 1), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 1), dtype="int64") = R.argmax(data, axis=1, keepdims=True)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedArgMin:
        @R.function
        def main(
            data: R.Tensor((4, 5), dtype="float32"),
        ) -> R.Tensor((4, 1), dtype="int64"):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((4, 1), dtype="int64") = R.argmin(data, axis=1, keepdims=True)
                R.output(gv)
            return gv

    verify_no_tie("ArgMax", ExpectedArgMax)
    verify_no_tie("ArgMin", ExpectedArgMin)


def test_split_to_sequence_keepdims_0():
    """keepdims=0, no split input: each chunk of size 1 has the split axis squeezed out."""

    def verify_split_to_sequence_keepdims_0(axis: int, expected):
        shape = [3, 4, 5]
        out_shape = [s for i, s in enumerate(shape) if i != axis]

        split_to_seq_node = helper.make_node(
            "SplitToSequence",
            ["data"],
            ["output"],
            axis=axis,
            keepdims=0,
        )
        graph = helper.make_graph(
            [split_to_seq_node],
            f"test_split_to_sequence_keepdims_0_axis{axis}",
            inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
            outputs=[
                helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, out_shape)
            ],
        )
        model = helper.make_model(graph, producer_name="test_split_to_sequence_keepdims_0")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedKeepdims0Axis0:
        @R.function
        def main(
            data: R.Tensor((3, 4, 5), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((4, 5), dtype="float32"),
            R.Tensor((4, 5), dtype="float32"),
            R.Tensor((4, 5), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 4, 5), dtype="float32"),
                    R.Tensor((1, 4, 5), dtype="float32"),
                    R.Tensor((1, 4, 5), dtype="float32"),
                ) = R.split(data, indices_or_sections=3, axis=0)
                lv1: R.Tensor((1, 4, 5), dtype="float32") = lv[0]
                lv2: R.Tensor((1, 4, 5), dtype="float32") = lv[1]
                lv3: R.Tensor((1, 4, 5), dtype="float32") = lv[2]
                lv4: R.Tensor((4, 5), dtype="float32") = R.squeeze(lv1, axis=[0])
                lv5: R.Tensor((4, 5), dtype="float32") = R.squeeze(lv2, axis=[0])
                lv6: R.Tensor((4, 5), dtype="float32") = R.squeeze(lv3, axis=[0])
                gv: R.Tuple(
                    R.Tensor((4, 5), dtype="float32"),
                    R.Tensor((4, 5), dtype="float32"),
                    R.Tensor((4, 5), dtype="float32"),
                ) = lv4, lv5, lv6
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedKeepdims0Axis1:
        @R.function
        def main(
            data: R.Tensor((3, 4, 5), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((3, 5), dtype="float32"),
            R.Tensor((3, 5), dtype="float32"),
            R.Tensor((3, 5), dtype="float32"),
            R.Tensor((3, 5), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((3, 1, 5), dtype="float32"),
                    R.Tensor((3, 1, 5), dtype="float32"),
                    R.Tensor((3, 1, 5), dtype="float32"),
                    R.Tensor((3, 1, 5), dtype="float32"),
                ) = R.split(data, indices_or_sections=4, axis=1)
                lv1: R.Tensor((3, 1, 5), dtype="float32") = lv[0]
                lv2: R.Tensor((3, 1, 5), dtype="float32") = lv[1]
                lv3: R.Tensor((3, 1, 5), dtype="float32") = lv[2]
                lv4: R.Tensor((3, 1, 5), dtype="float32") = lv[3]
                lv5: R.Tensor((3, 5), dtype="float32") = R.squeeze(lv1, axis=[1])
                lv6: R.Tensor((3, 5), dtype="float32") = R.squeeze(lv2, axis=[1])
                lv7: R.Tensor((3, 5), dtype="float32") = R.squeeze(lv3, axis=[1])
                lv8: R.Tensor((3, 5), dtype="float32") = R.squeeze(lv4, axis=[1])
                gv: R.Tuple(
                    R.Tensor((3, 5), dtype="float32"),
                    R.Tensor((3, 5), dtype="float32"),
                    R.Tensor((3, 5), dtype="float32"),
                    R.Tensor((3, 5), dtype="float32"),
                ) = lv5, lv6, lv7, lv8
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedKeepdims0Axis2:
        @R.function
        def main(
            data: R.Tensor((3, 4, 5), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((3, 4), dtype="float32"),
            R.Tensor((3, 4), dtype="float32"),
            R.Tensor((3, 4), dtype="float32"),
            R.Tensor((3, 4), dtype="float32"),
            R.Tensor((3, 4), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((3, 4, 1), dtype="float32"),
                    R.Tensor((3, 4, 1), dtype="float32"),
                    R.Tensor((3, 4, 1), dtype="float32"),
                    R.Tensor((3, 4, 1), dtype="float32"),
                    R.Tensor((3, 4, 1), dtype="float32"),
                ) = R.split(data, indices_or_sections=5, axis=2)
                lv1: R.Tensor((3, 4, 1), dtype="float32") = lv[0]
                lv2: R.Tensor((3, 4, 1), dtype="float32") = lv[1]
                lv3: R.Tensor((3, 4, 1), dtype="float32") = lv[2]
                lv4: R.Tensor((3, 4, 1), dtype="float32") = lv[3]
                lv5: R.Tensor((3, 4, 1), dtype="float32") = lv[4]
                lv6: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv1, axis=[2])
                lv7: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv2, axis=[2])
                lv8: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv3, axis=[2])
                lv9: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv4, axis=[2])
                lv10: R.Tensor((3, 4), dtype="float32") = R.squeeze(lv5, axis=[2])
                gv: R.Tuple(
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                    R.Tensor((3, 4), dtype="float32"),
                ) = lv6, lv7, lv8, lv9, lv10
                R.output(gv)
            return gv

    verify_split_to_sequence_keepdims_0(0, ExpectedKeepdims0Axis0)
    verify_split_to_sequence_keepdims_0(1, ExpectedKeepdims0Axis1)
    verify_split_to_sequence_keepdims_0(2, ExpectedKeepdims0Axis2)


def test_split_to_sequence_keepdims_ignored_when_split_provided():
    """Per spec: keepdims is ignored when split input is provided.
    TVM follows the spec — output keeps the split axis even with keepdims=0."""
    split_node = make_constant_node("split", TensorProto.INT64, (), [1])
    split_to_seq_node = helper.make_node(
        "SplitToSequence",
        ["data", "split"],
        ["output"],
        axis=0,
        keepdims=0,
    )
    graph = helper.make_graph(
        [split_node, split_to_seq_node],
        "test_split_to_sequence_keepdims_ignored",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 5])],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [1, 5])],
    )
    model = helper.make_model(
        graph,
        producer_name="test_split_to_sequence_keepdims_ignored",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 8
    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((4, 5), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((1, 5), dtype="float32"),
            R.Tensor((1, 5), dtype="float32"),
            R.Tensor((1, 5), dtype="float32"),
            R.Tensor((1, 5), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((1, 5), dtype="float32"),
                    R.Tensor((1, 5), dtype="float32"),
                    R.Tensor((1, 5), dtype="float32"),
                    R.Tensor((1, 5), dtype="float32"),
                ) = R.split(data, indices_or_sections=4, axis=0)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_split_to_sequence_uneven_last_chunk():
    """Spec: last chunk may be smaller if dim is not divisible by scalar split."""

    def verify_split_to_sequence_uneven_last_chunk(axis: int, shape: list[int], expected):
        split_node = make_constant_node("split", TensorProto.INT64, (), [2])
        split_to_seq_node = helper.make_node(
            "SplitToSequence", ["data", "split"], ["output"], axis=axis, keepdims=1
        )
        graph = helper.make_graph(
            [split_node, split_to_seq_node],
            f"test_split_to_sequence_uneven_axis{axis}",
            inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
            outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, producer_name="test_split_to_sequence_uneven")
        tvm_model = from_onnx(model, keep_params_in_input=True)
        tvm_tirx.ir.assert_structural_equal(tvm_model, expected)

    @I.ir_module
    class ExpectedUnevenAxis0:
        @R.function
        def main(
            data: R.Tensor((5, 4), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 4), dtype="float32"),
            R.Tensor((2, 4), dtype="float32"),
            R.Tensor((1, 4), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((2, 4), dtype="float32"),
                    R.Tensor((2, 4), dtype="float32"),
                    R.Tensor((1, 4), dtype="float32"),
                ) = R.split(data, indices_or_sections=3, axis=0)
                R.output(gv)
            return gv

    @I.ir_module
    class ExpectedUnevenAxis1:
        @R.function
        def main(
            data: R.Tensor((3, 5), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((3, 2), dtype="float32"),
            R.Tensor((3, 2), dtype="float32"),
            R.Tensor((3, 1), dtype="float32"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((3, 2), dtype="float32"),
                    R.Tensor((3, 2), dtype="float32"),
                    R.Tensor((3, 1), dtype="float32"),
                ) = R.split(data, indices_or_sections=3, axis=1)
                R.output(gv)
            return gv

    verify_split_to_sequence_uneven_last_chunk(0, [5, 4], ExpectedUnevenAxis0)
    verify_split_to_sequence_uneven_last_chunk(1, [3, 5], ExpectedUnevenAxis1)


def test_quantizelinear_singleton_qparams_opset10():
    """QuantizeLinear must treat shape-[1] scale/zp as scalar in opset10."""
    node = helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_singleton_qparams_opset10",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 3, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [4, 3, 2, 2])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.03125]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [1], [127]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.standard_normal((4, 3, 2, 2)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


def test_dequantizelinear_singleton_qparams_opset10():
    """DequantizeLinear must treat shape-[1] scale/zp as scalar in opset10."""
    node = helper.make_node("DequantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "dequantizelinear_singleton_qparams_opset10",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [64])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.125]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [1], [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.integers(low=0, high=255, size=(64,), dtype=np.uint8)
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


@pytest.mark.parametrize(
    ("op_type", "input_dtype", "output_dtype", "input_value"),
    [
        ("QuantizeLinear", TensorProto.FLOAT, TensorProto.UINT8, np.array(1.25, "float32")),
        ("DequantizeLinear", TensorProto.UINT8, TensorProto.FLOAT, np.array(7, "uint8")),
    ],
)
def test_qdqlinear_scalar_input(op_type, input_dtype, output_dtype, input_value):
    node = helper.make_node(op_type, ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        f"{op_type.lower()}_scalar_input",
        [helper.make_tensor_value_info("x", input_dtype, [])],
        [helper.make_tensor_value_info("y", output_dtype, [])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [], [0.25]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [], [2]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    check_correctness(model, inputs={"x": input_value}, opset=18, check_dtypes=True)


def test_quantizelinear_optional_zero_point_opset13():
    """ONNX allows missing zero_point input; importer should default it to 0 (uint8)."""
    node = helper.make_node("QuantizeLinear", ["x", "scale"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_optional_zero_point_opset13",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 5])],
        initializer=[helper.make_tensor("scale", TensorProto.FLOAT, [], [0.2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    x = rg.standard_normal((2, 5)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=13, check_dtypes=True)


def test_dynamicquantizelinear_opset11():
    """DynamicQuantizeLinear should import as quantization helper ops."""
    node = helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "y_scale", "y_zero_point"])
    graph = helper.make_graph(
        [node],
        "dynamicquantizelinear_opset11",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [
            helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3, 4]),
            helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("y_zero_point", TensorProto.UINT8, []),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    tvm_model = from_onnx(model, opset=11, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 4), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 3, 4), dtype="uint8"),
            R.Tensor((), dtype="float32"),
            R.Tensor((), dtype="uint8"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv: R.Tensor((), dtype="float32") = R.max(x, axis=None, keepdims=False)
                lv1: R.Tensor((), dtype="float32") = R.maximum(R.const(0.0, "float32"), lv)
                lv2: R.Tensor((), dtype="float32") = R.min(x, axis=None, keepdims=False)
                lv3: R.Tensor((), dtype="float32") = R.minimum(R.const(0.0, "float32"), lv2)
                lv4: R.Tensor((), dtype="float32") = R.subtract(lv1, lv3)
                lv5: R.Tensor((), dtype="float32") = R.divide(lv4, R.const(255.0, "float32"))
                lv6: R.Tensor((), dtype="float32") = R.divide(lv3, lv5)
                lv7: R.Tensor((), dtype="float32") = R.subtract(R.const(0.0, "float32"), lv6)
                lv8: R.Tensor((), dtype="float32") = R.clip(lv7, R.prim_value(0), R.prim_value(255))
                lv9: R.Tensor((), dtype="float32") = R.round(lv8)
                lv10: R.Tensor((), dtype="uint8") = R.astype(lv9, dtype="uint8")
                lv11: R.Tensor((2, 3, 4), dtype="uint8") = R.quantize(
                    x, lv5, lv10, out_dtype="uint8", axis=0
                )
                gv: R.Tuple(
                    R.Tensor((2, 3, 4), dtype="uint8"),
                    R.Tensor((), dtype="float32"),
                    R.Tensor((), dtype="uint8"),
                ) = (lv11, lv5, lv10)
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


def test_quantizelinear_default_axis_opset10():
    """opset10 QuantizeLinear should honor default axis=1 (not hardcode axis=0)."""
    node = helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_axis_opset10",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3, 4])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [3], [0.05, 0.1, 0.2]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [3], [1, 127, 250]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.standard_normal((2, 3, 4)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


def test_dequantizelinear_default_axis_opset10():
    """opset10 DequantizeLinear should honor default axis=1 (not hardcode axis=0)."""
    node = helper.make_node("DequantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "dequantizelinear_axis_opset10",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [3], [0.05, 0.1, 0.2]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [3], [1, 127, 250]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.integers(low=0, high=255, size=(2, 3, 4), dtype=np.uint8)
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


@pytest.mark.parametrize("opset", [21, 23, 24, 25])
def test_quantizelinear_output_dtype(opset):
    node = helper.make_node("QuantizeLinear", ["x", "scale"], ["y"], output_dtype=TensorProto.INT16)
    graph = helper.make_graph(
        [node],
        "quantizelinear_output_dtype",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.INT16, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            scale: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((2, 3), dtype="int16"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="int16") = R.quantize(
                    x, scale, R.const(0, "int16"), out_dtype="int16", axis=1
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("opset", [19, 21, 23, 24, 25])
def test_dequantizelinear_scale_dtype(opset):
    node = helper.make_node("DequantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "dequantizelinear_scale_dtype",
        [
            helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, []),
            helper.make_tensor_value_info("zero_point", TensorProto.INT8, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="int8"),
            scale: R.Tensor((), dtype="float16"),
            zero_point: R.Tensor((), dtype="int8"),
        ) -> R.Tensor((2, 3), dtype="float16"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float16") = R.dequantize(
                    x, scale, zero_point, out_dtype="float16", axis=1
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("opset", [23, 24, 25])
def test_dequantizelinear_output_dtype(opset):
    node = helper.make_node(
        "DequantizeLinear",
        ["x", "scale", "zero_point"],
        ["y"],
        output_dtype=TensorProto.FLOAT,
    )
    graph = helper.make_graph(
        [node],
        "dequantizelinear_output_dtype",
        [
            helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, []),
            helper.make_tensor_value_info("zero_point", TensorProto.INT8, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="int8"),
            scale: R.Tensor((), dtype="float16"),
            zero_point: R.Tensor((), dtype="int8"),
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="float32") = R.dequantize(
                    x, scale, zero_point, out_dtype="float32", axis=1
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("opset", [19, 21, 23, 24, 25])
def test_quantizelinear_integer_saturate(opset):
    node = helper.make_node("QuantizeLinear", ["x", "scale"], ["y"], saturate=0)
    graph = helper.make_graph(
        [node],
        "quantizelinear_integer_saturate",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"),
            scale: R.Tensor((), dtype="float32"),
        ) -> R.Tensor((2, 3), dtype="uint8"):
            R.func_attr({"num_input": 2})
            with R.dataflow():
                gv: R.Tensor((2, 3), dtype="uint8") = R.quantize(
                    x, scale, R.const(0, "uint8"), out_dtype="uint8", axis=1
                )
                R.output(gv)
            return gv

    tvm_tirx.ir.assert_structural_equal(tvm_model, Expected)


@pytest.mark.parametrize("opset", [19, 21, 23, 24, 25])
def test_quantizelinear_float8_saturate_rejected(opset):
    node = helper.make_node(
        "QuantizeLinear",
        ["x", "scale", "zero_point"],
        ["y"],
        saturate=0,
    )
    graph = helper.make_graph(
        [node],
        "quantizelinear_float8_saturate",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("zero_point", TensorProto.FLOAT8E4M3FN, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT8E4M3FN, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    with pytest.raises(ValueError, match="saturate=0"):
        from_onnx(model, opset=opset, keep_params_in_input=True)


@pytest.mark.parametrize("opset", [21, 23, 24, 25])
@pytest.mark.parametrize(
    "op_name,input_dtype,output_dtype",
    [
        ("QuantizeLinear", TensorProto.FLOAT, TensorProto.INT8),
        ("DequantizeLinear", TensorProto.INT8, TensorProto.FLOAT),
    ],
)
def test_qdq_blocked_quantization_rejected(opset, op_name, input_dtype, output_dtype):
    node = helper.make_node(
        op_name,
        ["x", "scale", "zero_point"],
        ["y"],
        axis=1,
        block_size=2,
    )
    graph = helper.make_graph(
        [node],
        "qdq_blocked_quantization",
        [
            helper.make_tensor_value_info("x", input_dtype, [1, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("zero_point", TensorProto.INT8, [1, 2]),
        ],
        [helper.make_tensor_value_info("y", output_dtype, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    with pytest.raises(ValueError, match="blocked quantization"):
        from_onnx(model, opset=opset, keep_params_in_input=True)


@pytest.mark.parametrize("opset", [23, 24, 25])
def test_quantizelinear_precision_rejected(opset):
    node = helper.make_node(
        "QuantizeLinear",
        ["x", "scale"],
        ["y"],
        output_dtype=TensorProto.INT8,
        precision=TensorProto.FLOAT16,
    )
    graph = helper.make_graph(
        [node],
        "quantizelinear_precision",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.INT8, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    with pytest.raises(ValueError, match="precision attribute"):
        from_onnx(model, opset=opset, keep_params_in_input=True)


@pytest.mark.parametrize("opset", [21, 23, 24, 25])
def test_quantizelinear_output_dtype_mismatch(opset):
    node = helper.make_node(
        "QuantizeLinear",
        ["x", "scale", "zero_point"],
        ["y"],
        output_dtype=TensorProto.UINT8,
    )
    graph = helper.make_graph(
        [node],
        "quantizelinear_output_dtype_mismatch",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("zero_point", TensorProto.INT8, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    with pytest.raises(ValueError, match="must match the zero-point dtype"):
        from_onnx(model, opset=opset, keep_params_in_input=True)


if __name__ == "__main__":
    tvm_tirx.testing.main()
