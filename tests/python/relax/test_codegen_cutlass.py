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
# ruff: noqa: E501, F841, RUF005
import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import ExternFunc as _ExternFunc
from tvm.testing import env as tvm_env
from tvm_tirx.ir import TupleType as _TupleType

pytest.importorskip("scipy")  # tvm.topi.testing imports scipy

import tvm.topi.testing
from tvm import relax
from tvm.contrib.cutlass.build import is_shape_valid_for_cutlass_matmul
from tvm.contrib.pickle_memoize import memoize
from tvm.relax import script as R
from tvm.relax.backend.cuda.cutlass import partition_for_cutlass
from tvm.relax.script import ir_builder as relax_builder
from tvm.relax.testing import (
    get_relax_attention_module,
    get_relax_matmul_module,
    get_relax_stacked_attention_module,
)
from tvm.s_tir import script as Ts
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.script.ir_builder import IRBuilder


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


@tvm_tirx.script.ir_module
class Conv2dBiasReLU:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), "float16"),
        weight: R.Tensor((32, 3, 3, 16), "float16"),
        bias: R.Tensor((1, 1, 1, 32), "float16"),
    ):
        with R.dataflow():
            conv1 = R.nn.relu(
                R.nn.conv2d(data, weight, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI")
                + bias,
            )
            R.output(conv1)

        return conv1


@tvm_tirx.script.ir_module
class Conv2dx2:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 8), "float16"),
        weight1: R.Tensor((8, 3, 3, 8), "float16"),
        weight2: R.Tensor((8, 3, 3, 8), "float16"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.conv2d(
                data, weight1, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
            )
            conv2 = relax.op.nn.conv2d(
                conv1, weight2, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
            )
            R.output(conv2)

        return conv2


pytestmark = [
    pytest.mark.skipif(not tvm_env.build_flag_enabled("USE_CUTLASS"), reason="need cutlass"),
]


def build_and_run(mod, inputs_np, target, legalize=True, cuda_graph=False):
    with tvm_tirx.ir.transform.PassContext(
        config={
            "relax.backend.use_cuda_graph": cuda_graph,
            "relax.transform.apply_legalize_ops": legalize,
        }
    ):
        ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target, 0)
        vm = relax.VirtualMachine(ex, dev)
        f = vm["main"]
        inputs = [tvm_tirx.runtime.tensor(inp, dev) for inp in inputs_np]

        # For cuda graph, run the compiled function twice to make sure that we can launch the
        # cached graph on the second run.
        if cuda_graph:
            f(*inputs)

        return f(*inputs).numpy()

    return tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def build_cutlass(mod, assert_all_bindings_fused=True, num_final_bindings=1):
    mod = partition_for_cutlass(mod)

    if assert_all_bindings_fused:
        assert len(mod["main"].body.blocks[0].bindings) == num_final_bindings, (
            "Not all bindings are fused. " + str(mod["main"])
        )

    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    mod = codegen_pass(mod)
    return mod


def get_result_with_relax_cutlass_offload(
    mod, *args, assert_all_bindings_fused=True, num_final_bindings=1
):
    mod = build_cutlass(mod, assert_all_bindings_fused, num_final_bindings)
    return build_and_run(mod, args, "cuda")


def test_kernel_sharing():
    low, high = -1, 1
    data_np = np.random.randint(low, high, size=(16, 32, 32, 8)).astype("float16")
    weight1_np = np.random.randint(low, high, size=(8, 3, 3, 8)).astype("float16")
    weight2_np = np.random.randint(low, high, size=(8, 3, 3, 8)).astype("float16")

    out = get_result_with_relax_cutlass_offload(
        Conv2dx2, data_np, weight1_np, weight2_np, assert_all_bindings_fused=False
    )
    ref = build_and_run(Conv2dx2, [data_np, weight1_np, weight2_np], "llvm")

    np.testing.assert_equal(out, ref)


def get_relax_conv2d_module(
    data_shape,
    weight_shape,
    dtype,
    with_bias=False,
    activation=None,
    residual_bin_op=None,
    residual_activation=None,
):
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            data = R.arg_("data", R.Tensor(data_shape, dtype))
            weight = R.arg_("weight", R.Tensor(weight_shape, dtype))
            if with_bias:
                bias = R.arg_("bias", R.Tensor((1, 1, 1, weight_shape[0]), dtype))

            with R.dataflow() as frame:
                output = R.emit(
                    R.nn.conv2d(
                        data,
                        weight,
                        out_dtype=dtype,
                        padding=(1, 1),
                        data_layout="NHWC",
                        kernel_layout="OHWI",
                    )
                )
                if with_bias:
                    output = R.emit(output + bias)
                if activation is not None:
                    output = R.emit(activation(output))
                if residual_bin_op is not None:
                    output = R.emit(residual_bin_op(output, data))
                    if residual_activation is not None:
                        output = R.emit(residual_activation(output))
                R.output(output)

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def _to_concrete_shape(symbolic_shape, var_table=None):
    if var_table is None:
        var_table = {}

    result = []
    for dim in symbolic_shape:
        if isinstance(dim, tuple):
            result.append(_to_concrete_shape(dim, var_table))
            continue

        if not isinstance(dim, tvm_tirx.tirx.expr.Var):
            result.append(dim)
            continue

        if dim not in var_table:
            var_table[dim] = np.random.randint(10, 50)
        result.append(var_table[dim])

    return tuple(result)


_vars = {
    "a": tvm_tirx.tirx.expr.Var("a", "int64"),
    "b": tvm_tirx.tirx.expr.Var("b", "int64"),
}


_epilogue_table = {
    "none": (False, None),
    "bias": (True, None),
    "relu": (True, R.nn.relu),
    "gelu": (True, R.nn.gelu),
    "silu": (True, R.nn.silu),
}


_residual_block_table = {
    "none": (None, None),
    "add_relu": (R.add, R.nn.relu),
    "mul_relu": (R.multiply, R.nn.relu),
    "add": (R.add, None),
    "mul": (R.multiply, None),
}


@pytest.mark.parametrize(
    "data_shape, weight_shape, dtype, epilogue, residual_block",
    [
        # Regular
        ((16, 32, 32, 16), (32, 3, 3, 16), "float16", "none", "none"),
        ((40, 128, 50, 16), (16, 2, 2, 16), "float16", "bias", "none"),
        ((3, 64, 64, 128), (32, 1, 1, 128), "float16", "relu", "none"),
        ((12, 32, 32, 16), (45, 5, 5, 16), "float16", "silu", "none"),
        # residual block
        ((3, 64, 64, 16), (16, 3, 3, 16), "float16", "relu", "add"),
        ((16, 32, 32, 16), (16, 3, 3, 16), "float16", "relu", "mul_relu"),
        ((40, 128, 50, 16), (16, 3, 3, 16), "float16", "bias", "add_relu"),
        ((128, 32, 32, 16), (16, 3, 3, 16), "float16", "silu", "mul"),
    ],
)
def test_conv2d_offload(data_shape, weight_shape, dtype, epilogue, residual_block):
    low, high = -1, 1
    data = np.random.randint(low, high, size=data_shape).astype(dtype)
    weight = np.random.randint(low, high, size=weight_shape).astype(dtype)
    bias = np.random.randint(low, high, size=(1, 1, 1, weight_shape[0])).astype(dtype)

    with_bias, activation = _epilogue_table[epilogue]
    residual_bin_op, residual_activation = _residual_block_table[residual_block]

    if with_bias:
        args = (data, weight, bias)
    else:
        args = (data, weight)

    mod = get_relax_conv2d_module(
        data_shape,
        weight_shape,
        dtype,
        with_bias=with_bias,
        activation=activation,
        residual_bin_op=residual_bin_op,
        residual_activation=residual_activation,
    )
    out = get_result_with_relax_cutlass_offload(mod, *args)

    ref = build_and_run(mod, args, "llvm")

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "data_shape, weight_shape, dtype",
    [
        # batch dynamism
        ((T.Var("n", "int64"), 32, 32, 16), (32, 3, 3, 16), "float16"),
        # channel dynamism
        ((16, 32, 32, T.Var("c", "int64")), (32, 3, 3, T.Var("c", "int64")), "float16"),
    ],
)
def test_conv2d_dynamic(data_shape, weight_shape, dtype):
    # Create dynamic conv2d module.
    mod = get_relax_conv2d_module(
        data_shape,
        weight_shape,
        dtype,
    )
    # Attempt to offload to cutlass, should run without an error
    # but not offload due to incompatibility.
    mod = build_cutlass(mod)
    # Check that no cutlass call is introduced (until we support dynamism).
    assert "call_dps" not in str(mod.__repr__())


def test_cutlass_partition_conv2d_residual_blocked():
    @tvm_tirx.script.ir_module
    class Conv2dReLU:
        """
        This conv2d should not be fused as conv2d residual block, because both lhs and rhs of
        the last R.add depends on the result of conv2d.
        """

        @R.function
        def main(
            data: R.Tensor((32, 3, 3, 16), "float32"),
            weight: R.Tensor((16, 3, 3, 16), "float32"),
            bias: R.Tensor((1, 1, 1, 16), "float32"),
        ):
            with R.dataflow():
                conv1 = R.nn.conv2d(
                    data,
                    weight,
                    padding=(1, 1),
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                )
                out = R.nn.relu(conv1 + bias)
                # residual depends on conv result, which cannot be handled in cutlass
                result = out + out
                R.output(result)

            return result

    mod = partition_for_cutlass(Conv2dReLU, annotate_codegen=False)
    for f_var in mod.functions:
        func = mod[f_var]
        if "Composite" in func.attrs:
            # verify that the function is not fused as residual block
            assert func.attrs["Composite"] == "cutlass.conv2d_bias_relu"


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, epilogue, residual_block",
    [
        # Regular
        ((32, 6), (6, 16), False, "none", "none"),
        ((_vars["a"], 6), (6, 16), False, "bias", "none"),
        # Transposed
        ((4, 16), (16, 128), True, "relu", "none"),
        ((35, 8), (8, 8), True, "gelu", "none"),
        # 3D x 3D
        ((6, 32, 8), (6, 8, 10), False, "bias", "none"),
        ((6, 32, 8), (6, 8, 10), True, "none", "none"),
        ((_vars["a"], 32, 8), (_vars["a"], 8, 10), True, "gelu", "none"),
        # 3D x 2D
        ((6, 32, 8), (8, 10), False, "none", "none"),
        ((_vars["a"], 32, 8), (8, 10), False, "bias", "none"),
        ((10, 16, 8), (8, 10), True, "relu", "none"),
        # 2D x 3D
        ((32, 8), (10, 8, 10), False, "relu", "none"),
        ((32, 8), (_vars["a"], 8, 10), True, "gelu", "none"),
        # ND x 2D
        ((3, 6, 32, 8), (8, 10), False, "bias", "none"),
        ((_vars["a"], _vars["b"], 6, 32, 8), (8, 10), False, "none", "none"),
        # 2D x ND
        ((32, 8), (5, 3, 8, 10), False, "gelu", "none"),
        # ND x ND
        ((5, 3, 32, 8), (5, 3, 8, 10), True, "relu", "none"),
        ((3, 2, 4, 16, 15), (1, 1, 15, 2), True, "gelu", "none"),
        ((1, 1, 16, 15), (3, 2, _vars["a"], 15, 2), False, "none", "none"),
        # Residual
        ((32, 8), (8, 8), False, "bias", "add"),
        ((4, 16), (16, 16), True, "relu", "add_relu"),
        ((8, 32, 8), (8, 8, 8), False, "bias", "add"),
        ((5, 3, 32, 8), (8, 8), True, "relu", "add"),
        # Residual fusion without bias - this is supported via the matmul + bias pattern
        # where bias == residual input
        ((4, 16), (16, 16), False, "none", "add"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "float16",
    ],
)
def test_matmul_offload(
    x_shape,
    y_shape,
    transpose_y,
    epilogue,
    residual_block,
    dtype,
):
    with_bias, activation = _epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = _to_concrete_shape(x_shape, var_table)
    concrete_y_shape = _to_concrete_shape(y_shape, var_table)
    x = np.random.randn(*concrete_x_shape).astype(dtype)
    y = np.random.randn(*concrete_y_shape).astype(dtype)

    if transpose_y:
        y = np.swapaxes(y, -2, -1)
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    if with_bias:
        bias = np.random.randn(concrete_y_shape[-1]).astype(dtype)
        args = (x, y, bias)
    else:
        bias = None
        args = (x, y)

    residual_bin_op, residual_activation = _residual_block_table[residual_block]

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        dtype,
        bias_shape=bias.shape if with_bias else None,
        transposed_y=transpose_y,
        activation=activation,
        residual_bin_op=residual_bin_op,
        residual_activation=residual_activation,
    )
    out = get_result_with_relax_cutlass_offload(mod, *args)
    ref = build_and_run(mod, args, "llvm")

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_matmul_with_3d_bias_offload():
    x_shape = (1, 4, 8)
    y_shape = (1, 8, 16)
    dtype = "float16"

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)
    bias = np.random.randn(1, x_shape[-2], y_shape[-1]).astype(dtype)
    args = (x, y, bias)

    @tvm_tirx.script.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((1, 4, 8), "float16"),
            y: R.Tensor((1, 8, 16), "float16"),
            bias: R.Tensor((1, 4, 16), "float16"),
        ):
            with R.dataflow():
                lv1 = R.matmul(x, y)
                gv1 = lv1 + bias
                R.output(gv1)

            return gv1

    out = get_result_with_relax_cutlass_offload(Mod, *args)
    ref = build_and_run(Mod, args, "llvm", legalize=True)

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "x_shape, y_shape, expected",
    [
        # Regular matmul
        ((3, 4), (4, 5), True),
        # Batch matmul without stretching
        ((3, 16, 15), (3, 15, 2), True),
        ((_vars["a"], 16, 15), (_vars["a"], 15, 2), True),
        # Broadcast 2D to 3D
        ((3, 16, 15), (15, 2), True),
        ((_vars["a"], 16, 15), (15, 2), True),
        ((16, 15), (3, 15, 2), True),
        # Broadcast one-length dimension
        ((1, 16, 15), (3, 15, 2), True),
        ((3, 16, 15), (1, 15, 2), True),
        ((1, 1, 16, 15), (3, 2, 4, 15, 2), True),
        ((1, 1, 16, 15), (3, _vars["a"], 4, 15, 2), True),
        # ND x ND
        ((3, 2, 4, 16, 15), (3, 2, 4, 15, 2), True),
        ((_vars["a"], 2, 4, 16, 15), (_vars["a"], 2, 4, 15, 2), True),
        (
            (_vars["a"], _vars["b"], 4, 16, 15),
            (_vars["a"], _vars["b"], 4, 15, 2),
            True,
        ),
        # ND x ND with one-length dimension
        ((1, 2, 4, 16, 15), (1, 2, 4, 15, 2), True),
        ((3, 2, 1, 16, 15), (3, 2, 1, 15, 2), True),
        # Extra one-length dimension doesn't block broadcasting
        ((3, 2, 1, 16, 15), (1, 1, 3, 2, 1, 15, 2), True),
        # Not broadcasting all dims. Cannot be computed by stride-based batch gemm
        ((3, 1, 1, 16, 15), (3, 2, 4, 15, 2), False),
        ((3, 2, 4, 16, 15), (2, 4, 15, 2), False),
        # Different shape
        ((3, 4, 16, 15), (3, 2, 15, 2), False),
        ((3, _vars["a"], 16, 15), (3, _vars["b"], 15, 2), False),
        # Cannot prove that broadcast dimensions are equal
        ((_vars["a"], 16, 15), (3, 15, 2), False),
        ((3, _vars["a"], 1, 16, 15), (1, 1, 3, 2, 1, 15, 2), False),
        # Reduction axis must be constant
        ((3, _vars["a"]), (_vars["a"], 5), False),
    ],
)
def test_is_shape_valid_for_cutlass_matmul(x_shape, y_shape, expected):
    assert is_shape_valid_for_cutlass_matmul(x_shape, y_shape) == expected


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, dtype",
    [
        # Not broadcasting all dims. Cannot be computed by stride-based batch gemm
        ((3, 1, 1, 16, 15), (3, 2, 4, 15, 2), False, "float16"),
        ((3, 2, _vars["a"], 16, 15), (3, 2, 4, 15, 2), False, "float16"),
        ((1, 2, 1, 16, 15), (2, 1, 4, 15, 2), False, "float16"),
        ((3, 2, 4, 16, 15), (2, 4, 15, 2), True, "float16"),
        ((3, 16, 15), (2, 1, 3, 15, 2), True, "float16"),
        ((3, 16, 15), (_vars["a"], 1, 3, 15, 2), True, "float16"),
        ((_vars["a"], 1, 3, 16, 15), (_vars["b"], 1, 3, 15, 2), True, "float16"),
        ((_vars["a"], _vars["b"], 3, 16, 15), (_vars["a"], 1, 3, 15, 2), True, "float16"),
    ],
)
def test_cutlass_partition_matmul_blocked(x_shape, y_shape, transpose_y, dtype):
    if transpose_y:
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    mod = get_relax_matmul_module(x_shape, y_shape, dtype, transposed_y=transpose_y)
    mod = partition_for_cutlass(mod)

    assert len(mod.functions) == 1


def test_cutlass_partition_matmul_tuple_return_blocked():
    @tvm_tirx.script.ir_module
    class TransposedMatmul:
        @R.function
        def main(
            x: R.Tensor((4, 4), "float32"),
            y: R.Tensor((4, 4), "float32"),
        ):
            with R.dataflow():
                lv1 = R.permute_dims(y)
                # Because lv1 is used by both lv2 and out, it should stay out of
                # the fused function. Otherwise the fused function will return
                # tuple output, which isn't possible in cutlass, e.g.
                # @R.function
                # def fused_relax_permute_dims_relax_matmul(...):
                #     R.func_attr({"Composite": "cutlass.matmul_transposed", "Primitive": True})
                #     with R.dataflow():
                #         gv: R.Tensor((4, 4), dtype="float32") = R.permute_dims(y, axes=None)
                #         gv1: R.Tensor((4, 4), dtype="float32") = R.matmul(x, gv)
                #         R.output(gv, gv1)
                #     return (gv, gv1)  # Cannot get `gv` if dispatch to cutlass kernel.
                lv2 = R.matmul(x, lv1)
                out = R.matmul(lv1, lv2)
                R.output(out)

            return out

    mod = partition_for_cutlass(TransposedMatmul, annotate_codegen=False)
    for f_var in mod.functions:
        func = mod[f_var]
        if "Composite" in func.attrs:
            # verify that the function is not fused as transposed matmul
            assert func.attrs["Composite"] == "cutlass.matmul"


def test_cutlass_partition_matmul_cyclic_dependency_blocked():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((128, 128), "float16"), w: R.Tensor((128, 128), "float16")):
            with R.dataflow():
                # Because lv1 depends on lv, this block should be fused as matmul instead of matmul_bias.
                lv = R.matmul(x, w)
                lv1 = R.power(lv, R.const(2.0, "float16"))
                lv2 = R.add(lv, lv1)
                R.output(lv2)
            return lv2

    mod = partition_for_cutlass(Module, annotate_codegen=False)
    for f_var in mod.functions:
        func = mod[f_var]
        if "Composite" in func.attrs:
            assert func.attrs["Composite"] == "cutlass.matmul"


@pytest.fixture(params=["float16", "float32"])
def attention_dtype(request):
    return request.param


@pytest.fixture(
    params=[
        # B, S, N, H
        (32, (_vars["a"], 8), 16, (8, 8)),
        (32, (8, 8), 16, (8, 8)),
        (4, (16, 8), 32, (8, 8)),  # s != s_kv
        (4, (16, 8), 32, (8, 16)),  # h != h_v
        (32, (8, 8), 16, (4, 4)),  # h is not aligned
        (2, (8, 8), 8, (256, 256)),  # needs output accumulator buffer
    ]
)
def attention_size(request):
    return request.param


def get_numpy_attention_ref(
    b,
    s,
    s_kv,
    n,
    h,
    h_v,
    bias_shape,
    qk_scale,
    causal,
    dtype,
    window_size=None,
    num_kv_head=None,
):
    num_kv_head = num_kv_head or n
    q = np.random.randn(b, s, n, h).astype(dtype)
    k = np.random.randn(b, s_kv, num_kv_head, h).astype(dtype)
    v = np.random.randn(b, s_kv, num_kv_head, h_v).astype(dtype)
    if bias_shape == "none":
        bias = None
    else:
        bias = np.random.randn(*bias_shape).astype(dtype)

    ref = tvm.topi.testing.attention_python(
        q, k, v, bias, qk_scale, causal=causal, window_size=window_size, layout="BSNH"
    )

    return q, k, v, bias, ref


def test_attention_offload(attention_size, attention_dtype):
    b, (s, s_kv), n, (h, h_v) = attention_size
    concrete_s, concrete_s_kv = _to_concrete_shape((s, s_kv))
    q, k, v, _, ref = get_numpy_attention_ref(
        b, concrete_s, concrete_s_kv, n, h, h_v, "none", "none", "none", attention_dtype
    )

    q_shape = (b, s, n, h)
    k_shape = (b, s_kv, n, h)
    v_shape = (b, s_kv, n, h_v)

    mod = get_relax_attention_module(q_shape, k_shape, v_shape, dtype=attention_dtype)
    out = get_result_with_relax_cutlass_offload(mod, q, k, v, num_final_bindings=2)

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.fixture(
    params=[
        # B, S, N, H, bias_shape
        (4, (16, 8), 32, (8, 16), (4, 32, 16, 8)),
        (4, (16, 8), 32, (8, 16), (4, 1, 16, 8)),
        (4, (16, 8), 32, (8, 16), (4, 32, 1, 8)),
        (4, (16, 8), 32, (8, 16), (4, 1, 1, 8)),
        (4, (16, 8), 32, (8, 16), (1, 32, 16, 8)),
        (4, (16, 8), 32, (8, 16), (1, 1, 16, 8)),
        (4, (16, 8), 32, (8, 16), (1, 32, 1, 8)),
        (4, (16, 8), 32, (8, 16), (1, 1, 1, 8)),
    ]
)
def attention_bias_size(request):
    return request.param


def test_attention_bias_offload(attention_bias_size):
    b, (s, s_kv), n, (h, h_v), bias_shape = attention_bias_size
    concrete_s, concrete_s_kv, concrete_bias_shape = _to_concrete_shape((s, s_kv, bias_shape))

    q, k, v, bias, ref = get_numpy_attention_ref(
        b, concrete_s, concrete_s_kv, n, h, h_v, concrete_bias_shape, "none", "none", "float32"
    )

    q_shape = (b, s, n, h)
    k_shape = (b, s_kv, n, h)
    v_shape = (b, s_kv, n, h_v)

    mod = get_relax_attention_module(
        q_shape, k_shape, v_shape, bias_shape=bias_shape, dtype="float32"
    )
    out = get_result_with_relax_cutlass_offload(mod, q, k, v, bias, num_final_bindings=2)

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.fixture(
    params=[
        # B, S, N, H, bias_shape
        (4, (16, 8), 32, (8, 16), (4, 32, 16, 8)),
        (4, (16, 8), 32, (8, 16), "none"),
    ]
)
def attention_scale_size(request):
    return request.param


@pytest.fixture(params=[0.01, 1e-8, -0.5, 1.23])
def attention_scale(request):
    return request.param


def test_attention_scale_offload(attention_scale_size, attention_scale):
    b, (s, s_kv), n, (h, h_v), bias_shape = attention_scale_size
    q, k, v, bias, ref = get_numpy_attention_ref(
        b, s, s_kv, n, h, h_v, bias_shape, attention_scale, "none", "float32"
    )

    q_shape = (b, s, n, h)
    k_shape = (b, s_kv, n, h)
    v_shape = (b, s_kv, n, h_v)

    mod = get_relax_attention_module(
        q_shape, k_shape, v_shape, dtype="float32", bias_shape=bias_shape, qk_scale=attention_scale
    )
    if bias is None:
        out = get_result_with_relax_cutlass_offload(mod, q, k, v, num_final_bindings=2)
    else:
        out = get_result_with_relax_cutlass_offload(mod, q, k, v, bias, num_final_bindings=2)
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.fixture(
    params=[
        # B, S, N, H, bias_shape
        (2, (16, 8), 4, (8, 16), "none"),
        (2, (8, 16), 4, (8, 16), "none"),
        (2, (16, 8), 4, (8, 16), (2, 4, 16, 8)),
    ]
)
def attention_causal_size(request):
    return request.param


@pytest.fixture(params=["TopLeft", "BottomRight"])
def attention_causal(request):
    return request.param


def test_attention_causal_offload(attention_causal_size, attention_causal):
    b, (s, s_kv), n, (h, h_v), bias_shape = attention_causal_size
    q, k, v, bias, ref = get_numpy_attention_ref(
        b, s, s_kv, n, h, h_v, bias_shape, "none", attention_causal, "float16"
    )

    q_shape = (b, s, n, h)
    k_shape = (b, s_kv, n, h)
    v_shape = (b, s_kv, n, h_v)

    mod = get_relax_attention_module(
        q_shape,
        k_shape,
        v_shape,
        dtype="float16",
        bias_shape=bias_shape,
        causal_mask=attention_causal,
    )

    if bias is None:
        out = get_result_with_relax_cutlass_offload(mod, q, k, v, num_final_bindings=2)
    else:
        out = get_result_with_relax_cutlass_offload(mod, q, k, v, bias, num_final_bindings=2)
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@memoize("topi.tests.test_codegen_cutlass.test_stacked_attention_offload")
def get_numpy_stacked_attention_ref(b, s, n, h, h_v, bias_shape, qk_scale, dtype):
    qkv = np.random.randn(b, s, n * h + n * h + n * h_v).astype(dtype)
    split_qkv = np.split(qkv, [n * h, n * h * 2], axis=2)
    q = np.reshape(split_qkv[0], (b, s, n, h))
    k = np.reshape(split_qkv[1], (b, s, n, h))
    v = np.reshape(split_qkv[2], (b, s, n, h_v))
    if not bias_shape == "none":
        bias = np.random.randn(*bias_shape).astype(dtype)
    else:
        bias = None
    ref = tvm.topi.testing.attention_python(
        q, k, v, bias, qk_scale, causal="none", window_size=None, layout="BSNH"
    )
    return qkv, bias, ref


@pytest.fixture(
    params=[
        # B, S, N, H, bias_shape, scale, single_shape
        (4, 8, 32, (64, 32), "none", "none", False),
        (4, 8, 32, (64, 32), (4, 32, 8, 8), 0.5, False),
        (4, 8, 32, (64, 64), "none", "none", True),
    ]
)
def stacked_attention_size(request):
    return request.param


def test_stacked_attention_split_offload(stacked_attention_size):
    b, s, n, (h, h_v), bias_shape, scale, single_shape = stacked_attention_size
    qkv, bias, ref = get_numpy_stacked_attention_ref(b, s, n, h, h_v, bias_shape, scale, "float16")
    if scale == "none":
        mod = get_relax_stacked_attention_module(
            qkv,
            b,
            s,
            n,
            h,
            h_v,
            "split",
            bias,
            single_shape=single_shape,
            layout="BS3NH",
        )
    else:
        mod = get_relax_stacked_attention_module(
            qkv,
            b,
            s,
            n,
            h,
            h_v,
            "split",
            bias,
            scale,
            single_shape=single_shape,
            layout="BS3NH",
        )

    if bias is None:
        out = get_result_with_relax_cutlass_offload(mod, qkv, num_final_bindings=2)
    else:
        out = get_result_with_relax_cutlass_offload(mod, qkv, bias, num_final_bindings=2)
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_stacked_attention_strided_slice_offload(stacked_attention_size):
    b, s, n, (h, h_v), bias_shape, scale, single_shape = stacked_attention_size
    qkv, bias, ref = get_numpy_stacked_attention_ref(b, s, n, h, h_v, bias_shape, scale, "float32")
    if scale == "none":
        mod = get_relax_stacked_attention_module(
            qkv,
            b,
            s,
            n,
            h,
            h_v,
            "strided_slice",
            bias,
            single_shape=single_shape,
            layout="BS3NH",
        )
    else:
        mod = get_relax_stacked_attention_module(
            qkv,
            b,
            s,
            n,
            h,
            h_v,
            "strided_slice",
            bias,
            scale,
            single_shape=single_shape,
            layout="BS3NH",
        )
    if bias is None:
        out = get_result_with_relax_cutlass_offload(mod, qkv, num_final_bindings=2)
    else:
        out = get_result_with_relax_cutlass_offload(mod, qkv, bias, num_final_bindings=2)
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.fixture(
    params=[
        # B, S, N, H, bias_shape, scale
        (4, (16, 8), 32, (8, 16), "none", 0.5),
        (4, (16, 8), 32, (8, 16), (4, 32, 16, 8), 0.5),
        (4, (16, 8), "none", (8, 16), "none", 0.5),
        (4, (16, 8), "none", (8, 16), (4, 32, 16, 8), 0.5),
    ]
)
def attention_rewrite_size(request):
    return request.param


def get_relax_attention_rewrite_module(
    q_shape, k_shape, v_shape, out_shape, dtype, bias_shape=None, scale=None
):
    from tvm.relax.script import ir_builder as relax_builder
    from tvm_tirx.script.ir_builder import IRBuilder
    from tvm_tirx.tirx.script import ir_builder as T

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            q = R.arg_("q", R.Tensor(q_shape, dtype))
            k = R.arg_("k", R.Tensor(k_shape, dtype))
            v = R.arg_("v", R.Tensor(v_shape, dtype))
            if bias_shape is not None:
                bias = R.arg_("bias", R.Tensor(bias_shape, dtype))
            with R.dataflow() as frame:
                if len(q_shape) == 4:
                    q = R.emit(R.permute_dims(q, axes=[0, 2, 1, 3]))
                    q = R.emit(R.reshape(q, [q_shape[0] * q_shape[2], q_shape[1], q_shape[3]]))

                if len(k_shape) == 4:
                    k = R.emit(R.permute_dims(k, axes=[0, 2, 1, 3]))
                    k = R.emit(R.reshape(k, [k_shape[0] * k_shape[2], k_shape[1], k_shape[3]]))

                if len(v_shape) == 4:
                    v = R.emit(R.permute_dims(v, axes=[0, 2, 1, 3]))
                    v = R.emit(R.reshape(v, [v_shape[0] * v_shape[2], v_shape[1], v_shape[3]]))

                k = R.emit(R.permute_dims(k, axes=[0, 2, 1]))
                qk = R.emit(R.matmul(q, k))
                qk_scaled = R.emit(R.multiply(qk, R.const(scale, "float32")))
                if bias_shape is not None:
                    if len(bias_shape) == 4:
                        bias = R.emit(
                            R.reshape(bias, [bias_shape[0] * bias_shape[1], *bias_shape[2:]])
                        )
                    qk_added = R.emit(R.add(qk_scaled, bias))
                    softmax = R.emit(R.nn.softmax(qk_added, axis=-1))
                else:
                    softmax = R.emit(R.nn.softmax(qk_scaled, axis=-1))
                out = R.emit(R.matmul(softmax, v))

                if len(out_shape) == 4:
                    out = R.emit(
                        R.reshape(
                            out,
                            [out_shape[0], out_shape[2], out_shape[1], out_shape[3]],
                        )
                    )
                    out = R.emit(R.permute_dims(out, axes=[0, 2, 1, 3]))
                R.output(out)

            R.func_ret_value(frame.output_vars[0])

    original_func = builder.get()

    if scale is not None:
        scale = T.FloatImm("float32", scale)

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            q = R.arg_("q", R.Tensor(q_shape, dtype))
            k = R.arg_("k", R.Tensor(k_shape, dtype))
            v = R.arg_("v", R.Tensor(v_shape, dtype))
            if bias_shape is not None:
                bias = R.arg_("bias", R.Tensor(bias_shape, dtype))
            with R.dataflow() as frame:
                if len(q_shape) == 3:
                    q = R.emit(R.reshape(q, [q_shape[0], q_shape[1], 1, q_shape[2]]))

                if len(k_shape) == 3:
                    k = R.emit(R.reshape(k, [k_shape[0], k_shape[1], 1, k_shape[2]]))

                if len(v_shape) == 3:
                    v = R.emit(R.reshape(v, [v_shape[0], v_shape[1], 1, v_shape[2]]))

                if bias_shape is not None:
                    if len(bias_shape) == 4:
                        bias = R.emit(
                            R.reshape(
                                bias,
                                [
                                    bias_shape[0] * bias_shape[1],
                                    bias_shape[2],
                                    bias_shape[3],
                                ],
                            )
                        )
                        bias = R.emit(
                            R.reshape(
                                bias,
                                [
                                    bias_shape[0],
                                    bias_shape[1],
                                    bias_shape[2],
                                    bias_shape[3],
                                ],
                            )
                        )
                    elif len(bias_shape) == 3:
                        bias = R.emit(
                            R.reshape(bias, [bias_shape[0], 1, bias_shape[1], bias_shape[2]])
                        )
                else:
                    bias = None
                out = R.emit(R.nn.attention(q, k, v, bias, scale))

                if len(out_shape) == 3:
                    out = R.emit(R.reshape(out, [out_shape[0], out_shape[1], out_shape[2]]))
                R.output(out)

            R.func_ret_value(frame.output_vars[0])

    expected_func = builder.get()
    return tvm_tirx.IRModule({"main": original_func}), tvm_tirx.IRModule({"main": expected_func})


def get_numpy_attention_input(q_shape, k_shape, v_shape, bias_shape, dtype):
    q = np.random.randn(*q_shape).astype(dtype)
    k = np.random.randn(*k_shape).astype(dtype)
    v = np.random.randn(*v_shape).astype(dtype)
    if not bias_shape == "none":
        bias = np.random.randn(*bias_shape).astype(dtype)
    else:
        bias = None
    return q, k, v, bias


def test_attention_rewrite_offload(attention_rewrite_size):
    b, (s, s_kv), n, (h, h_v), bias_shape, scale = attention_rewrite_size
    q_shape = [b, s, n, h] if n != "none" else [b, s, h]
    k_shape = [b, s_kv, n, h] if n != "none" else [b, s_kv, h]
    v_shape = [b, s_kv, n, h_v] if n != "none" else [b, s_kv, h_v]
    out_shape = [b, s, n, h_v] if n != "none" else [b, s, h_v]
    bias_shape = [b, n, s, s_kv] if n != "none" else [b, s, s_kv]
    q, k, v, bias = get_numpy_attention_input(q_shape, k_shape, v_shape, bias_shape, "float32")
    original_mod, expected_mod = get_relax_attention_rewrite_module(
        q_shape, k_shape, v_shape, out_shape, "float32", bias_shape, scale
    )
    original_mod = partition_for_cutlass(original_mod, True)
    expected_mod = partition_for_cutlass(expected_mod, True)
    tvm_tirx.ir.assert_structural_equal(original_mod, expected_mod, True)

    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})
    original_mod = codegen_pass(original_mod)
    expected_mod = codegen_pass(expected_mod)
    if bias is None:
        original_out = build_and_run(original_mod, [q, k, v], "cuda")
        expected_out = build_and_run(expected_mod, [q, k, v], "cuda")
        tvm_tirx.testing.assert_allclose(original_out, expected_out, rtol=1e-5, atol=1e-5)
    else:
        original_out = build_and_run(original_mod, [q, k, v, bias], "cuda", legalize=False)
        expected_out = build_and_run(expected_mod, [q, k, v, bias], "cuda", legalize=False)
        tvm_tirx.testing.assert_allclose(original_out, expected_out, rtol=1e-5, atol=1e-5)


def test_conv2d_residual_broadcast():
    data_shape = (2, 64, 64, 8)
    weight_shape = (8, 3, 3, 8)
    dtype = "float16"

    def get_mod(residual_batch):
        with IRBuilder() as builder:
            with relax_builder.function():
                R.func_name_("main")
                data = R.arg_("data", R.Tensor(data_shape, dtype))
                weight = R.arg_("weight", R.Tensor(weight_shape, dtype))
                bias = R.arg_("bias", R.Tensor((1, 1, weight_shape[0]), dtype))
                residual = R.arg_(
                    "residual", R.Tensor((residual_batch, 1, 1, weight_shape[0]), dtype)
                )

                with R.dataflow() as frame:
                    output = R.emit(
                        R.nn.conv2d(
                            data,
                            weight,
                            out_dtype=dtype,
                            padding=(1, 1),
                            data_layout="NHWC",
                            kernel_layout="OHWI",
                        )
                    )
                    output = R.emit(output + bias)
                    output = R.emit(R.nn.relu(output))
                    output = R.emit(R.add(output, residual))
                    R.output(output)

                R.func_ret_value(frame.output_vars[0])

        func = builder.get()
        return tvm_tirx.IRModule({"main": func})

    low = -1
    high = 1

    residual_batch = 1
    mod = get_mod(residual_batch)
    data = np.random.randint(low, high, size=data_shape).astype(dtype)
    weight = np.random.randint(low, high, size=weight_shape).astype(dtype)
    bias = np.random.randint(low, high, size=(1, 1, weight_shape[0])).astype(dtype)
    bias2 = np.random.randint(low, high, size=(residual_batch, 1, 1, weight_shape[0])).astype(dtype)

    args = [data, weight, bias, bias2]
    out = get_result_with_relax_cutlass_offload(mod, *args)
    ref = build_and_run(mod, args, "llvm")
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "data_shape, dtype, axes",
    [
        ((2, 128, 64), "float16", [-1]),
        ((128, 30), "float32", [-1]),
        ((2, 128, 64), "float32", [1]),
        ((2, 128, 64), "float32", [1, 2]),
    ],
)
def test_layer_norm(data_shape, dtype, axes):
    def get_mod(data_shape, dtype, axes):
        reduced_shape = [data_shape[axis] for axis in axes]
        with IRBuilder() as builder:
            with relax_builder.function():
                R.func_name_("main")
                inp = R.arg_("input", R.Tensor(data_shape, dtype))
                gamma = R.arg_("gamma", R.Tensor(reduced_shape, dtype))
                beta = R.arg_("beta", R.Tensor(reduced_shape, dtype))

                with R.dataflow() as frame:
                    output = R.emit(R.nn.layer_norm(inp, gamma, beta, axes))
                    R.output(output)

                R.func_ret_value(frame.output_vars[0])

        func = builder.get()
        return tvm_tirx.IRModule({"main": func})

    Module = get_mod(data_shape, dtype, axes)
    mod = partition_for_cutlass(Module)

    if len(axes) != 1 or (axes[0] != -1 and axes[0] != len(data_shape) - 1):
        tvm_tirx.ir.assert_structural_equal(mod, Module)
        return

    mod = relax.transform.RunCodegen()(mod)

    inp = np.random.randn(*data_shape).astype(dtype)
    gamma = np.random.randn(data_shape[-1]).astype(dtype)
    beta = np.random.randn(data_shape[-1]).astype(dtype)
    out = build_and_run(mod, [inp, gamma, beta], "cuda")
    ref = build_and_run(Module, [inp, gamma, beta], "llvm")

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_attention_rewrite_fp16():
    @I.ir_module
    class Module:
        @R.function
        def main(
            q: R.Tensor((4, 16, 32, 8), dtype="float16"),
            k: R.Tensor((4, 8, 32, 8), dtype="float16"),
            v: R.Tensor((4, 8, 32, 16), dtype="float16"),
            bias: R.Tensor((4, 32, 16, 8), dtype="float16"),
        ) -> R.Tensor((4, 16, 32, 16), dtype="float16"):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                lv = R.permute_dims(q, axes=[0, 2, 1, 3])
                lv1 = R.reshape(lv, R.shape([128, 16, 8]))
                lv2 = R.permute_dims(k, axes=[0, 2, 1, 3])
                lv3 = R.reshape(lv2, R.shape([128, 8, 8]))
                lv4 = R.permute_dims(v, axes=[0, 2, 1, 3])
                lv5 = R.reshape(lv4, R.shape([128, 8, 16]))
                lv6 = R.permute_dims(lv3, axes=[0, 2, 1])
                lv7 = R.matmul(lv1, lv6, out_dtype="float16")
                lv3_1 = R.astype(R.const(0.5, "float32"), dtype="float16")
                lv8 = R.multiply(lv7, lv3_1)
                lv9 = R.reshape(bias, R.shape([128, 16, 8]))
                lv10 = R.add(lv8, lv9)
                lv10_fp16 = R.astype(lv10, dtype="float16")
                lv11 = R.nn.softmax(lv10_fp16, axis=2)
                lv5_1 = R.astype(lv11, dtype="float16")
                lv12 = R.matmul(lv5_1, lv5, out_dtype="float16")
                lv13 = R.reshape(lv12, R.shape([4, 32, 16, 16]))
                lv6_1 = R.permute_dims(lv13, axes=[0, 2, 1, 3])
                lv14 = R.astype(lv6_1, dtype="float32")
                R.output(lv14)
            return lv14

    @I.ir_module
    class Expected:
        @R.function
        def fused_relax_nn_attention_bias_cutlass1(
            q: R.Tensor((4, 16, 32, 8), dtype="float16"),
            k: R.Tensor((4, 8, 32, 8), dtype="float16"),
            v: R.Tensor((4, 8, 32, 16), dtype="float16"),
            lv1: R.Tensor((4, 32, 16, 8), dtype="float16"),
            workspace: R.Tensor((65536,), dtype="uint8"),
        ) -> R.Tensor((4, 16, 32, 16), dtype="float16"):
            R.func_attr(
                {
                    "Codegen": "cutlass",
                    "WorkspaceSize": T.int64(65536),
                    "global_symbol": "fused_relax_nn_attention_bias_cutlass1",
                }
            )

            @R.function
            def gv_1(
                q_1: R.Tensor((4, 16, 32, 8), dtype="float16"),
                k_1: R.Tensor((4, 8, 32, 8), dtype="float16"),
                v_1: R.Tensor((4, 8, 32, 16), dtype="float16"),
                lv1_1: R.Tensor((4, 32, 16, 8), dtype="float16"),
                workspace_1: R.Tensor((65536,), dtype="uint8"),
            ) -> R.Tensor((4, 16, 32, 16), dtype="float16"):
                R.func_attr(
                    {
                        "Composite": "cutlass.attention_bias",
                        "WorkspaceSize": T.int64(65536),
                    }
                )
                with R.dataflow():
                    gv_2 = R.nn.attention(
                        q_1, k_1, v_1, lv1_1, scale=T.float32(0.5), causal_mask=None
                    )
                    R.output(gv_2)
                return gv_2

            gv1: R.Tensor((4, 16, 32, 16), dtype="float16") = gv_1(q, k, v, lv1, workspace)
            return gv1

        @R.function
        def main(
            q: R.Tensor((4, 16, 32, 8), dtype="float16"),
            k: R.Tensor((4, 8, 32, 8), dtype="float16"),
            v: R.Tensor((4, 8, 32, 16), dtype="float16"),
            bias: R.Tensor((4, 32, 16, 8), dtype="float16"),
        ) -> R.Tensor((4, 16, 32, 16), dtype="float32"):
            R.func_attr({"num_input": 4})
            cls = Expected
            with R.dataflow():
                workspace_main = R.builtin.alloc_tensor(
                    R.shape([65536]), R.dtype("uint8"), R.prim_value(0)
                )
                lv_1 = R.reshape(bias, R.shape([128, 16, 8]))
                lv1 = R.reshape(lv_1, R.shape([4, 32, 16, 8]))
                lv_2 = cls.fused_relax_nn_attention_bias_cutlass1(q, k, v, lv1, workspace_main)
                lv14 = R.astype(lv_2, dtype="float32")
                R.output(lv14)
            return lv14

    mod = partition_for_cutlass(Module)
    tvm_tirx.ir.assert_structural_equal(mod, Expected)


def split_transform_deploy_mod(mod):
    mod_transform = tvm_tirx.IRModule()
    mod_deploy = tvm_tirx.IRModule().with_attrs(mod.attrs)

    transform_func_name = None

    for gv, func in mod.functions.items():
        if "transform_params" in gv.name_hint:
            transform_func_name = gv.name_hint
            mod_transform[gv] = func
        elif isinstance(func, tvm_tirx.tirx.Function):
            mod_transform[gv] = func
        else:
            mod_deploy[gv] = func

    assert transform_func_name is not None
    return mod_transform, mod_deploy, transform_func_name


def test_fp16A_int4B_gemm():
    @I.ir_module
    class Module:
        @Ts.function
        def decode(
            A: T.Tensor((T.int64(64), T.int64(64)), "int8"),
            B: T.Tensor((T.int64(128),), "float16"),
            decode_1: T.Tensor((T.int64(64), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i, j in T.grid(T.int64(64), T.int64(128)):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A[v_i, v_j // T.int64(2)], B[v_j])
                    Ts.writes(decode_1[v_i, v_j])
                    decode_1[v_i, v_j] = (
                        T.Cast(
                            "float16",
                            T.shift_right(
                                T.shift_left(
                                    T.bitwise_and(
                                        T.shift_right(
                                            T.Cast("int32", A[v_i, v_j // T.int64(2)]),
                                            T.Cast("int32", v_j % T.int64(2)) * 4,
                                        ),
                                        15,
                                    ),
                                    28,
                                ),
                                28,
                            ),
                        )
                        * B[v_j]
                    )

        @Ts.function
        def encode(
            A: T.Tensor((T.int64(128), T.int64(64)), "float16"),
            w_gathered: T.Tensor((T.int64(64), T.int64(64)), "int8"),
            compute: T.Tensor((T.int64(128),), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            max_abs_value = Ts.sblock_alloc_tensor((T.int64(128),), "float16")
            scale = Ts.sblock_alloc_tensor((T.int64(128),))
            for i, k in T.grid(T.int64(128), T.int64(64)):
                with Ts.sblock("max_abs_value"):
                    v_i, v_k = Ts.axis.remap("SR", [i, k])
                    Ts.reads(A[v_i, v_k])
                    Ts.writes(max_abs_value[v_i])
                    with Ts.init():
                        max_abs_value[v_i] = T.float16(-65504)
                    max_abs_value[v_i] = T.max(max_abs_value[v_i], T.fabs(A[v_i, v_k]))
            for i in range(T.int64(128)):
                with Ts.sblock("scale"):
                    v_i = Ts.axis.spatial(T.int64(128), i)
                    Ts.reads(max_abs_value[v_i])
                    Ts.writes(scale[v_i])
                    scale[v_i] = T.max(
                        T.Cast("float32", max_abs_value[v_i]), T.float32(0.0001)
                    ) * T.float32(0.125)
            for j, i, k in T.grid(T.int64(64), T.int64(64), T.int64(2)):
                with Ts.sblock("w_gathered"):
                    v_j, v_i, v_k = Ts.axis.remap("SSR", [j, i, k])
                    Ts.reads(A[v_i * T.int64(2) + v_k, v_j], scale[v_i * T.int64(2) + v_k])
                    Ts.writes(w_gathered[v_j, v_i])
                    with Ts.init():
                        w_gathered[v_j, v_i] = T.int8(0)
                    w_gathered[v_j, v_i] = T.bitwise_or(
                        w_gathered[v_j, v_i],
                        T.if_then_else(
                            v_i * T.int64(2) + v_k < T.int64(128),
                            T.shift_left(
                                T.bitwise_and(
                                    T.Cast(
                                        "int8",
                                        T.min(
                                            T.max(
                                                T.round(
                                                    T.Cast(
                                                        "float32", A[v_i * T.int64(2) + v_k, v_j]
                                                    )
                                                    / scale[v_i * T.int64(2) + v_k]
                                                ),
                                                T.float32(-8),
                                            ),
                                            T.float32(7),
                                        ),
                                    ),
                                    T.int8(15),
                                ),
                                T.Cast("int8", v_k) * T.int8(4),
                            ),
                            T.int8(0),
                        ),
                    )
            for i0 in range(T.int64(128)):
                with Ts.sblock("compute"):
                    v_i0 = Ts.axis.spatial(T.int64(128), i0)
                    Ts.reads(scale[v_i0])
                    Ts.writes(compute[v_i0])
                    compute[v_i0] = T.Cast("float16", scale[v_i0])

        @R.function
        def main_bias(
            x: R.Tensor((64, 64), dtype="float16"),
            y: R.Tensor((128, 64), dtype="float16"),
            bias: R.Tensor((1, 128), dtype="float16"),
        ) -> R.Tensor((64, 128), dtype="float16"):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (y,),
                    ty_args=[
                        _TupleType(
                            [R.Tensor((64, 64), dtype="int8"), R.Tensor((128,), dtype="float16")]
                        )
                    ],
                )
                lv1 = lv[0]
                lv2 = R.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    lv1,
                    80,
                    True,
                    ty_args=(R.Tensor((64, 64), dtype="int8"),),
                )
                lv3: R.Tensor((128,), dtype="float16") = lv[1]
                lv6 = R.call_tir(
                    cls.decode, (lv2, lv3), ty_args=[R.Tensor((64, 128), dtype="float16")]
                )
                lv1_1: R.Tensor((64, 128), dtype="float16") = R.matmul(x, lv6, out_dtype="float16")
                lv2_1: R.Tensor((64, 128), dtype="float16") = R.add(lv1_1, bias)
                R.output(lv2_1)
            return lv2_1

        @R.function
        def main_cast_bias(
            x: R.Tensor((64, 64), dtype="float16"),
            y: R.Tensor((128, 64), dtype="float16"),
            bias: R.Tensor((1, 128), dtype="float16"),
        ) -> R.Tensor((64, 128), dtype="float16"):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (y,),
                    ty_args=[
                        _TupleType(
                            [R.Tensor((64, 64), dtype="int8"), R.Tensor((128,), dtype="float16")]
                        )
                    ],
                )
                lv1 = lv[0]
                lv2 = R.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    lv1,
                    80,
                    True,
                    ty_args=(R.Tensor((64, 64), dtype="int8"),),
                )
                lv3: R.Tensor((128,), dtype="float16") = lv[1]
                lv6 = R.call_tir(
                    cls.decode, (lv2, lv3), ty_args=[R.Tensor((64, 128), dtype="float16")]
                )
                lv1_1: R.Tensor((64, 128), dtype="float32") = R.matmul(x, lv6, out_dtype="float32")
                cast: R.Tensor((64, 128), dtype="float16") = R.astype(lv1_1, dtype="float16")
                lv2_1: R.Tensor((64, 128), dtype="float16") = R.add(cast, bias)
                R.output(lv2_1)
            return lv2_1

        @R.function
        def main_residual(
            x: R.Tensor((64, 64), dtype="float16"),
            residual: R.Tensor((64, 128), dtype="float16"),
            y: R.Tensor((128, 64), dtype="float16"),
            bias: R.Tensor((1, 128), dtype="float16"),
        ) -> R.Tensor((64, 128), dtype="float16"):
            R.func_attr({"num_input": 2})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (y,),
                    ty_args=[
                        _TupleType(
                            [R.Tensor((64, 64), dtype="int8"), R.Tensor((128,), dtype="float16")]
                        )
                    ],
                )
                lv1 = lv[0]
                lv2 = R.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    lv1,
                    80,
                    True,
                    ty_args=(R.Tensor((64, 64), dtype="int8"),),
                )
                lv3: R.Tensor((128,), dtype="float16") = lv[1]
                lv6 = R.call_tir(
                    cls.decode, (lv2, lv3), ty_args=[R.Tensor((64, 128), dtype="float16")]
                )
                lv1_1: R.Tensor((64, 128), dtype="float16") = R.matmul(x, lv6, out_dtype="float16")
                lv2_1: R.Tensor((64, 128), dtype="float16") = R.add(lv1_1, bias)
                lv3_1: R.Tensor((64, 128), dtype="float16") = R.add(lv2_1, residual)
                R.output(lv3_1)
            return lv3_1

    x_shape = (64, 64)
    y_shape = (128, 64)

    mod = partition_for_cutlass(Module)
    func_names = [name.name_hint for (name, _) in mod.functions.items()]
    assert "fused_decode_relax_matmul_relax_add_cutlass" in func_names
    assert "fused_decode_relax_matmul_relax_add_relax_add_cutlass" in func_names
    assert "fused_decode_relax_matmul_relax_astype_relax_add_cutlass" in func_names

    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": 80, "find_first_valid": False}},
        entry_functions=["main_bias", "main_residual", "main_cast_bias"],
    )(mod)

    x = np.random.randn(*x_shape).astype("float16")
    y = np.random.normal(0, 0.002, size=y_shape).astype("float16")
    bias = np.random.randn(1, y_shape[0]).astype("float16")
    residual = np.random.randn(x_shape[0], y_shape[0]).astype("float16")

    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.LiftTransformParams()(mod)

    mod_transform, mod_deploy, transform_func_name = split_transform_deploy_mod(mod)

    ex = tvm.compile(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm_tirx.cpu(0))

    packed_weight, scales, bias_trans = vm[transform_func_name](
        (tvm_tirx.runtime.tensor(y), tvm_tirx.runtime.tensor(bias))
    )

    ex_cuda = tvm.compile(mod_deploy, target="cuda")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        vm = relax.vm.VirtualMachine(ex_cuda, dev)
        x_nd = tvm_tirx.runtime.tensor(x, dev)
        residual_nd = tvm_tirx.runtime.tensor(residual, dev)
        params = [packed_weight.copyto(dev), scales.copyto(dev), bias_trans.copyto(dev)]

        for f_name in ["main_bias", "main_cast_bias", "main_residual"]:
            with_residual = "residual" in f_name

            if with_residual:
                inp = [x_nd, residual_nd] + params
            else:
                inp = [x_nd] + params

            out = vm[f_name](*inp).numpy()
            ref = np.dot(x, y.transpose()) + bias
            if with_residual:
                ref += residual
            tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_fp16A_int8B_gemm():
    @I.ir_module
    class Module:
        @Ts.function
        def decode(
            A: T.Tensor((T.int64(64), T.int64(64)), "int8"),
            B: T.Tensor((T.int64(64),), "float16"),
            decode_1: T.Tensor((T.int64(64), T.int64(64)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i, j in T.grid(T.int64(64), T.int64(64)):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A[v_i, v_j], B[v_j])
                    Ts.writes(decode_1[v_i, v_j])
                    decode_1[v_i, v_j] = T.Cast("float16", A[v_i, v_j]) * B[v_j]

        @Ts.function
        def encode(
            A: T.Tensor((T.int64(64), T.int64(64)), "float16"),
            w_gathered: T.Tensor((T.int64(64), T.int64(64)), "int8"),
            compute: T.Tensor((T.int64(64),), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            max_abs_value = Ts.sblock_alloc_tensor((T.int64(64),), "float16")
            scale = Ts.sblock_alloc_tensor((T.int64(64),))
            for i, k in T.grid(T.int64(64), T.int64(64)):
                with Ts.sblock("max_abs_value"):
                    v_i, v_k = Ts.axis.remap("SR", [i, k])
                    Ts.reads(A[v_i, v_k])
                    Ts.writes(max_abs_value[v_i])
                    with Ts.init():
                        max_abs_value[v_i] = T.float16(-65504)
                    max_abs_value[v_i] = T.max(max_abs_value[v_i], T.fabs(A[v_i, v_k]))
            for i in range(T.int64(64)):
                with Ts.sblock("scale"):
                    v_i = Ts.axis.spatial(T.int64(64), i)
                    Ts.reads(max_abs_value[v_i])
                    Ts.writes(scale[v_i])
                    scale[v_i] = T.max(
                        T.Cast("float32", max_abs_value[v_i]), T.float32(0.0001)
                    ) * T.float32(0.0078125)
            for j, i in T.grid(T.int64(64), T.int64(64)):
                with Ts.sblock("w_gathered"):
                    v_j, v_i = Ts.axis.remap("SS", [j, i])
                    Ts.reads(A[v_i, v_j], scale[v_i])
                    Ts.writes(w_gathered[v_j, v_i])
                    w_gathered[v_j, v_i] = T.Cast(
                        "int8",
                        T.min(
                            T.max(
                                T.round(T.Cast("float32", A[v_i, v_j]) / scale[v_i]),
                                T.float32(-128),
                            ),
                            T.float32(127),
                        ),
                    )
            for i0 in range(T.int64(64)):
                with Ts.sblock("compute"):
                    v_i0 = Ts.axis.spatial(T.int64(64), i0)
                    Ts.reads(scale[v_i0])
                    Ts.writes(compute[v_i0])
                    compute[v_i0] = T.Cast("float16", scale[v_i0])

        @R.function
        def main(
            x: R.Tensor((64, 64), dtype="float16"),
            y: R.Tensor((64, 64), dtype="float16"),
            bias: R.Tensor((64, 64), dtype="float16"),
        ) -> R.Tensor((64, 64), dtype="float16"):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (y,),
                    ty_args=[
                        _TupleType(
                            [R.Tensor((64, 64), dtype="int8"), R.Tensor((64,), dtype="float16")]
                        )
                    ],
                )
                lv1: R.Tensor((64, 64), dtype="int8") = lv[0]
                lv2: R.Tensor((64, 64), dtype="int8") = R.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    lv1,
                    R.prim_value(80),
                    R.prim_value(0),
                    ty_args=(R.Tensor((64, 64), dtype="int8"),),
                )
                lv3: R.Tensor((64,), dtype="float16") = lv[1]
                lv4: R.Tensor((64, 64), dtype="int8") = R.builtin.stop_lift_params(lv2)
                lv5: R.Tensor((64,), dtype="float16") = R.builtin.stop_lift_params(lv3)
                lv6 = R.call_tir(
                    cls.decode, (lv4, lv5), ty_args=[R.Tensor((64, 64), dtype="float16")]
                )
                lv1_1: R.Tensor((64, 64), dtype="float16") = R.matmul(x, lv6, out_dtype="float16")
                lv2_1: R.Tensor((64, 128), dtype="float16") = R.add(lv1_1, bias)
                lv2_2: R.Tensor((64, 128), dtype="float16") = R.nn.gelu(lv2_1)
                R.output(lv2_2)
            return lv2_2

    x_shape = (64, 64)
    y_shape = (64, 64)

    mod = partition_for_cutlass(Module)
    func_names = [name.name_hint for (name, _) in mod.functions.items()]
    assert "fused_decode_relax_matmul_relax_add_relax_nn_gelu_cutlass" in func_names

    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": 80, "find_first_valid": False}},
    )(mod)

    x = np.random.randn(*x_shape).astype("float16")
    y = np.random.normal(0, 0.002, size=y_shape).astype("float16")
    bias = np.random.randn(x_shape[0], y_shape[0]).astype("float16")

    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.LiftTransformParams()(mod)

    mod_transform, mod_deploy, transform_func_name = split_transform_deploy_mod(mod)

    ex = tvm.compile(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm_tirx.cpu(0))

    packed_weight, scales, bias_trans = vm[transform_func_name](
        (tvm_tirx.runtime.tensor(y), tvm_tirx.runtime.tensor(bias))
    )

    ex_cuda = tvm.compile(mod_deploy, target="cuda")

    def gelu_fp16(x):
        erf_inp = x * (0.5**0.5)
        from scipy.special import erf

        erf_out = erf(erf_inp.astype("float32")).astype("float16")
        return x * 0.5 * (1.0 + erf_out)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        vm = relax.vm.VirtualMachine(ex_cuda, dev)
        x_nd = tvm_tirx.runtime.tensor(x, dev)
        inp = [x_nd, packed_weight.copyto(dev), scales.copyto(dev), bias_trans.copyto(dev)]
        out = vm["main"](*inp).numpy()
        ref = gelu_fp16(np.dot(x, y.transpose()) + bias)
        tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_rms_norm():
    @I.ir_module
    class Module:
        @Ts.function
        def rms_norm(
            A: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
            B: T.Tensor((T.int64(4096),), "float16"),
            rms_norm: T.Tensor((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            Ared_temp = Ts.sblock_alloc_tensor((T.int64(1), T.int64(1)))
            for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
                with Ts.sblock("Ared_temp"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSR", [bsz, i, k])
                    Ts.reads(A[v_bsz, v_i, v_k])
                    Ts.writes(Ared_temp[v_bsz, v_i])
                    with Ts.init():
                        Ared_temp[v_bsz, v_i] = T.float32(0)
                    Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                        "float32", A[v_bsz, v_i, v_k]
                    ) * T.Cast("float32", A[v_bsz, v_i, v_k])
            for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
                with Ts.sblock("rms_norm"):
                    v_bsz, v_i, v_k = Ts.axis.remap("SSS", [bsz, i, k])
                    Ts.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                    Ts.writes(rms_norm[v_bsz, v_i, v_k])
                    rms_norm[v_bsz, v_i, v_k] = T.Cast(
                        "float16",
                        T.Cast("float32", B[v_k])
                        * (
                            T.Cast("float32", A[v_bsz, v_i, v_k])
                            / T.sqrt(
                                Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                                + T.float32(9.9999999999999995e-07)
                            )
                        ),
                    )

        @R.function
        def main(
            input: R.Tensor((1, 1, 4096), dtype="float16"),
            weight: R.Tensor((4096,), dtype="float16"),
        ) -> R.Tensor((1, 1, 4096), dtype="float16"):
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.rms_norm, (input, weight), ty_args=[R.Tensor((1, 1, 4096), dtype="float16")]
                )
                R.output(lv)
            return lv

    data_shape = (1, 1, 4096)
    dtype = "float16"
    mod = partition_for_cutlass(Module)

    # TODO(@tvm-team): This is temporary patch.Currently, the remaining packed function triggers error since it is not scheduled.
    # This is because RunCodegen does not support Function well yet.
    # i.e., it does remove the global symbol of Function, which would be no longer used,
    # and thus, the following DCE cannot remove this. Revisit when resolved.
    mod = tvm.s_tir.transform.DefaultGPUSchedule(target=tvm_tirx.target.Target("cuda"))(mod)

    mod = relax.transform.RunCodegen(
        {"cutlass": {"rms_eps": 1e-6}},
    )(mod)

    inp = np.random.randn(*data_shape).astype(dtype)
    weight = np.random.randn(data_shape[-1]).astype(dtype)
    out = build_and_run(mod, [inp, weight], "cuda")
    ref = build_and_run(Module, [inp, weight], "llvm", legalize=True)

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_conv2d_cuda_graph():
    @tvm_tirx.script.ir_module
    class Conv2d:
        @R.function
        def main(
            data: R.Tensor((16, 32, 32, 16), "float16"),
            weight1: R.Tensor((16, 3, 3, 16), "float16"),
            weight2: R.Tensor((16, 3, 3, 16), "float16"),
            weight3: R.Tensor((16, 3, 3, 16), "float16"),
            gamma: R.Tensor((16,), "float16"),
            beta: R.Tensor((16,), "float16"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                conv1 = R.nn.relu(
                    R.nn.conv2d(
                        data, weight1, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    )
                )
                conv2 = R.nn.relu(
                    R.nn.conv2d(
                        conv1, weight2, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    )
                )
                ln = R.nn.layer_norm(conv2, gamma, beta, axes=[-1])
                conv3 = R.nn.relu(
                    R.nn.conv2d(
                        ln, weight3, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
                    )
                )
                R.output(conv3)

            return conv3

    low, high = -1, 1
    data_shape = (16, 32, 32, 16)
    weight_shape = (16, 3, 3, 16)
    dtype = "float16"
    data = np.random.randint(low, high, size=data_shape).astype(dtype)
    weight1 = np.random.randint(low, high, size=weight_shape).astype(dtype)
    weight2 = np.random.randint(low, high, size=weight_shape).astype(dtype)
    weight3 = np.random.randint(low, high, size=weight_shape).astype(dtype)
    gamma = np.random.randint(low, high, size=(weight_shape[0],)).astype(dtype)
    beta = np.random.randint(low, high, size=(weight_shape[0],)).astype(dtype)
    inputs = [data, weight1, weight2, weight3, gamma, beta]

    mod = partition_for_cutlass(Conv2d)
    mod = relax.transform.RunCodegen({"cutlass": {"sm": 80, "find_first_valid": True}})(mod)
    mod = relax.pipeline.get_pipeline()(mod)  # pylint: disable=no-value-for-parameter

    mod = tvm.s_tir.transform.DefaultGPUSchedule(target=tvm_tirx.target.Target("cuda"))(mod)

    out = build_and_run(mod, inputs, "cuda", cuda_graph=True)
    ref = build_and_run(Conv2d, inputs, "llvm", legalize=True)
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_fp16A_int8B_gemm_batched():
    b = T.dynamic("b", "int64")

    @I.ir_module
    class Module:
        @Ts.function
        def decode(
            A: T.Tensor((T.int64(64), T.int64(64)), "int8"),
            B: T.Tensor((T.int64(64),), "float16"),
            decode_1: T.Tensor((T.int64(64), T.int64(64)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i, j in T.grid(T.int64(64), T.int64(64)):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A[v_i, v_j], B[v_j])
                    Ts.writes(decode_1[v_i, v_j])
                    decode_1[v_i, v_j] = T.Cast("float16", A[v_i, v_j]) * B[v_j]

        @Ts.function
        def encode(
            A: T.Tensor((T.int64(64), T.int64(64)), "float16"),
            w_gathered: T.Tensor((T.int64(64), T.int64(64)), "int8"),
            compute: T.Tensor((T.int64(64),), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            max_abs_value = Ts.sblock_alloc_tensor((T.int64(64),), "float16")
            scale = Ts.sblock_alloc_tensor((T.int64(64),))
            for i, k in T.grid(T.int64(64), T.int64(64)):
                with Ts.sblock("max_abs_value"):
                    v_i, v_k = Ts.axis.remap("SR", [i, k])
                    Ts.reads(A[v_i, v_k])
                    Ts.writes(max_abs_value[v_i])
                    with Ts.init():
                        max_abs_value[v_i] = T.float16(-65504)
                    max_abs_value[v_i] = T.max(max_abs_value[v_i], T.fabs(A[v_i, v_k]))
            for i in range(T.int64(64)):
                with Ts.sblock("scale"):
                    v_i = Ts.axis.spatial(T.int64(64), i)
                    Ts.reads(max_abs_value[v_i])
                    Ts.writes(scale[v_i])
                    scale[v_i] = T.max(
                        T.Cast("float32", max_abs_value[v_i]), T.float32(0.0001)
                    ) * T.float32(0.0078125)
            for j, i in T.grid(T.int64(64), T.int64(64)):
                with Ts.sblock("w_gathered"):
                    v_j, v_i = Ts.axis.remap("SS", [j, i])
                    Ts.reads(A[v_i, v_j], scale[v_i])
                    Ts.writes(w_gathered[v_j, v_i])
                    w_gathered[v_j, v_i] = T.Cast(
                        "int8",
                        T.min(
                            T.max(
                                T.round(T.Cast("float32", A[v_i, v_j]) / scale[v_i]),
                                T.float32(-128),
                            ),
                            T.float32(127),
                        ),
                    )
            for i0 in range(T.int64(64)):
                with Ts.sblock("compute"):
                    v_i0 = Ts.axis.spatial(T.int64(64), i0)
                    Ts.reads(scale[v_i0])
                    Ts.writes(compute[v_i0])
                    compute[v_i0] = T.Cast("float16", scale[v_i0])

        @R.function
        def main(
            x: R.Tensor((b, 64, 64), dtype="float16"),
            y: R.Tensor((64, 64), dtype="float16"),
        ) -> R.Tensor((b, 64, 64), dtype="float16"):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (y,),
                    ty_args=[
                        _TupleType(
                            [R.Tensor((64, 64), dtype="int8"), R.Tensor((64,), dtype="float16")]
                        )
                    ],
                )
                lv1: R.Tensor((64, 64), dtype="int8") = lv[0]
                lv2: R.Tensor((64, 64), dtype="int8") = R.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    lv1,
                    R.prim_value(80),
                    R.prim_value(0),
                    ty_args=(R.Tensor((64, 64), dtype="int8"),),
                )
                lv3: R.Tensor((64,), dtype="float16") = lv[1]
                lv4: R.Tensor((64, 64), dtype="int8") = R.builtin.stop_lift_params(lv2)
                lv5: R.Tensor((64,), dtype="float16") = R.builtin.stop_lift_params(lv3)
                lv6 = R.call_tir(
                    cls.decode, (lv4, lv5), ty_args=[R.Tensor((64, 64), dtype="float16")]
                )
                lv1_1: R.Tensor((b, 64, 64), dtype="float16") = R.matmul(
                    x, lv6, out_dtype="float16"
                )
                R.output(lv1_1)
            return lv1_1

    x_shape = (4, 64, 64)
    y_shape = (64, 64)

    mod = partition_for_cutlass(Module)

    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": 80, "find_first_valid": False}},
    )(mod)

    x = np.random.randn(*x_shape).astype("float16")
    y = np.random.normal(0, 0.002, size=y_shape).astype("float16")

    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.LiftTransformParams()(mod)

    mod_transform, mod_deploy, transform_func_name = split_transform_deploy_mod(mod)

    ex = tvm.compile(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm_tirx.cpu(0))

    packed_weight, scales = vm[transform_func_name]((tvm_tirx.runtime.tensor(y),))

    ex_cuda = tvm.compile(mod_deploy, target="cuda")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        vm = relax.vm.VirtualMachine(ex_cuda, dev)
        x_nd = tvm_tirx.runtime.tensor(x, dev)
        inp = [x_nd, packed_weight.copyto(dev), scales.copyto(dev)]
        out = vm["main"](*inp).numpy()
        ref = np.dot(x, y.transpose())
        tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_fp16A_int8B_gemm_batched_finegrained():
    b = T.dynamic("b", "int64")

    @I.ir_module
    class Module:
        @Ts.function
        def decode(
            A: T.Tensor((T.int64(128), T.int64(128)), "int8"),
            B: T.Tensor((T.int64(2), T.int64(128)), "float16"),
            decode_1: T.Tensor((T.int64(128), T.int64(128)), "float16"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i, j in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock("decode"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(A[v_i, v_j], B[v_i // T.int64(64), v_j])
                    Ts.writes(decode_1[v_i, v_j])
                    decode_1[v_i, v_j] = T.Cast("float16", A[v_i, v_j]) * B[v_i // T.int64(64), v_j]

        @Ts.function
        def encode(
            A: T.Tensor((T.int64(128), T.int64(128)), "float16"),
            w_gathered: T.Tensor((T.int64(128), T.int64(128)), "int8"),
            compute: T.Tensor(
                (
                    T.int64(2),
                    T.int64(128),
                ),
                "float16",
            ),
        ):
            T.func_attr({"tirx.noalias": True})
            max_abs_value = Ts.sblock_alloc_tensor(
                (
                    T.int64(2),
                    T.int64(128),
                ),
                "float16",
            )
            scale = Ts.sblock_alloc_tensor(
                (
                    T.int64(2),
                    T.int64(128),
                )
            )
            for i, j, k in T.grid(T.int64(2), T.int64(128), T.int64(64)):
                with Ts.sblock("max_abs_value"):
                    v_i, v_j, v_k = Ts.axis.remap("SSR", [i, j, k])
                    Ts.reads(A[v_j, v_i * T.int64(64) + v_k])
                    Ts.writes(max_abs_value[v_i, v_j])
                    with Ts.init():
                        max_abs_value[v_i, v_j] = T.float16(-65504)
                    max_abs_value[v_i, v_j] = T.max(
                        max_abs_value[v_i, v_j], T.fabs(A[v_j, v_i * T.int64(64) + v_k])
                    )
            for i, j in T.grid(T.int64(2), T.int64(128)):
                with Ts.sblock("scale"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(max_abs_value[v_i, v_j])
                    Ts.writes(scale[v_i, v_j])
                    scale[v_i, v_j] = T.max(
                        T.Cast("float32", max_abs_value[v_i, v_j]), T.float32(0.0001)
                    ) * T.float32(0.0078125)
            for j, i in T.grid(T.int64(128), T.int64(128)):
                with Ts.sblock("w_gathered"):
                    v_j, v_i = Ts.axis.remap("SS", [j, i])
                    Ts.reads(A[v_i, v_j], scale[v_j // T.int64(64), v_i])
                    Ts.writes(w_gathered[v_j, v_i])
                    w_gathered[v_j, v_i] = T.Cast(
                        "int8",
                        T.min(
                            T.max(
                                T.round(
                                    T.Cast("float32", A[v_i, v_j]) / scale[v_j // T.int64(64), v_i]
                                ),
                                T.float32(-128),
                            ),
                            T.float32(127),
                        ),
                    )
            for i0, i1 in T.grid(T.int64(2), T.int64(128)):
                with Ts.sblock("compute"):
                    v_i0, v_i1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(scale[v_i0, v_i1])
                    Ts.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.Cast("float16", scale[v_i0, v_i1])

        @R.function
        def main(
            x: R.Tensor((b, 128, 128), dtype="float16"),
            y: R.Tensor((128, 128), dtype="float16"),
        ) -> R.Tensor((b, 128, 128), dtype="float16"):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (y,),
                    ty_args=[
                        _TupleType(
                            [
                                R.Tensor((128, 128), dtype="int8"),
                                R.Tensor((2, 128), dtype="float16"),
                            ]
                        )
                    ],
                )
                lv1: R.Tensor((128, 128), dtype="int8") = lv[0]
                lv2: R.Tensor((128, 128), dtype="int8") = R.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    lv1,
                    R.prim_value(80),
                    R.prim_value(0),
                    ty_args=(R.Tensor((128, 128), dtype="int8"),),
                )
                lv3: R.Tensor((2, 128), dtype="float16") = lv[1]
                lv4: R.Tensor((128, 128), dtype="int8") = R.builtin.stop_lift_params(lv2)
                lv5: R.Tensor((2, 128), dtype="float16") = R.builtin.stop_lift_params(lv3)
                lv6 = R.call_tir(
                    cls.decode, (lv4, lv5), ty_args=[R.Tensor((128, 128), dtype="float16")]
                )
                lv1_1: R.Tensor((b, 128, 128), dtype="float16") = R.matmul(
                    x, lv6, out_dtype="float16"
                )
                R.output(lv1_1)
            return lv1_1

    x_shape = (4, 128, 128)
    y_shape = (128, 128)

    mod = partition_for_cutlass(Module)

    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": 80, "find_first_valid": False}},
    )(mod)

    x = np.random.randn(*x_shape).astype("float16")
    y = np.random.normal(0, 0.002, size=y_shape).astype("float16")

    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.LiftTransformParams()(mod)

    mod_transform, mod_deploy, transform_func_name = split_transform_deploy_mod(mod)

    ex = tvm.compile(mod_transform, target="llvm")
    vm = relax.vm.VirtualMachine(ex, tvm_tirx.cpu(0))

    packed_weight, scales = vm[transform_func_name]((tvm_tirx.runtime.tensor(y),))

    ex_cuda = tvm.compile(mod_deploy, target="cuda")

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        vm = relax.vm.VirtualMachine(ex_cuda, dev)
        x_nd = tvm_tirx.runtime.tensor(x, dev)
        inp = [x_nd, packed_weight.copyto(dev), scales.copyto(dev)]
        out = vm["main"](*inp).numpy()
        ref = np.dot(x, y.transpose())
        tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_attention_rewrite_multi_query():
    @I.ir_module
    class Module:
        @R.function
        def main(
            q: R.Tensor((4, 16, 32, 16), dtype="float16"),
            k_single: R.Tensor((4, 16, 1, 16), dtype="float16"),
            v_single: R.Tensor((4, 16, 1, 16), dtype="float16"),
        ) -> R.Tensor((4, 16, 32, 8), dtype="float16"):
            with R.dataflow():
                k = R.repeat(k_single, 32, axis=2)
                v = R.repeat(v_single, 32, axis=2)

                lv = R.permute_dims(q, axes=[0, 2, 1, 3])
                lv1 = R.reshape(lv, R.shape([128, 16, 16]))
                lv2 = R.permute_dims(k, axes=[0, 2, 1, 3])
                lv3 = R.reshape(lv2, R.shape([128, 16, 16]))
                lv4 = R.permute_dims(v, axes=[0, 2, 1, 3])
                lv5 = R.reshape(lv4, R.shape([128, 16, 16]))

                lv6 = R.permute_dims(lv3, axes=[0, 2, 1])
                lv7 = R.matmul(lv1, lv6, out_dtype="float16")
                lv3_1 = R.astype(R.const(0.25, "float32"), "float16")
                lv8 = R.multiply(lv7, lv3_1)
                lv11 = R.astype(R.nn.softmax(R.astype(lv8, "float32"), axis=2), "float16")
                lv12 = R.matmul(lv11, lv5, out_dtype="float16")
                lv13 = R.reshape(lv12, R.shape([4, 32, 16, 16]))
                lv6_1 = R.permute_dims(lv13, axes=[0, 2, 1, 3])
                R.output(lv6_1)
            return lv6_1

    q_np = np.random.randn(4, 16, 32, 16).astype("float16")
    k_np = np.random.randn(4, 16, 1, 16).astype("float16")
    v_np = np.random.randn(4, 16, 1, 16).astype("float16")
    args = [q_np, k_np, v_np]
    ref = build_and_run(Module, args, "llvm", legalize=True)

    mod = partition_for_cutlass(Module, use_flash_mqa=True)
    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80}})
    mod = codegen_pass(mod)

    out = build_and_run(mod, args, "cuda")

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def _test_batched_var_len_attention(
    mod, seq_lens, num_head, num_kv_head, head_size, window_size=None
):
    if not tvm_ffi.get_global_func("tvm.contrib.thrust.sum_scan", True):
        return

    hidden_size = num_head * head_size

    batched_queries = []
    batched_keys = []
    batched_values = []
    batched_refs = []

    for s in seq_lens:
        q, k, v, _, ref = get_numpy_attention_ref(
            1,
            s,
            s,
            num_head,
            head_size,
            head_size,
            "none",
            "none",
            "BottomRight",
            "float16",
            num_kv_head=num_kv_head,
            window_size=window_size,
        )
        batched_queries.append(np.reshape(q, [-1, hidden_size]))
        batched_keys.append(np.reshape(k, [-1, num_kv_head * head_size]))
        batched_values.append(np.reshape(v, [-1, num_kv_head * head_size]))
        batched_refs.append(np.reshape(ref, [-1, hidden_size]))

    batched_queries = np.vstack(batched_queries)
    batched_keys = np.vstack(batched_keys)
    batched_values = np.vstack(batched_values)
    ref = np.vstack(batched_refs)

    mod = partition_for_cutlass(mod)
    codegen_pass = relax.transform.RunCodegen({"cutlass": {"sm": 80}})
    mod = codegen_pass(mod)

    mod = relax.transform.LegalizeOps(target=tvm_tirx.target.Target("cuda"))(mod)
    mod = tvm.s_tir.transform.DefaultGPUSchedule(target=tvm_tirx.target.Target("cuda"))(mod)

    out = build_and_run(
        mod,
        [
            batched_queries,
            batched_keys,
            batched_values,
            np.array(seq_lens, dtype="int32"),
        ],
        "cuda",
    )

    ############# xformer reference for verification #############

    # attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_lens)

    # queries = torch.from_numpy(np.reshape(batched_queries, [1, -1, num_head, head_size])).to("cuda")
    # keys = torch.from_numpy(np.reshape(batched_keys, [1, -1, num_head, head_size])).to("cuda")
    # values = torch.from_numpy(np.reshape(batched_values, [1, -1, num_head, head_size])).to("cuda")

    # out = xops.memory_efficient_attention_forward(
    #     queries, keys, values,
    #     attn_bias=attn_bias,
    # ).cpu().numpy()[0]
    # out = np.reshape(out, [-1, hidden_size])

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_batched_var_len_attention():
    num_tokens = T.dynamic("num_tokens", "int64")
    num_seq = T.dynamic("num_seq", "int64")

    @I.ir_module
    class Module:
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def main(
            queries: R.Tensor((num_tokens, 4096), dtype="float16"),
            keys: R.Tensor((num_tokens, 4096), dtype="float16"),
            values: R.Tensor((num_tokens, 4096), dtype="float16"),
            seq_lens: R.Tensor((num_seq,), dtype="int32"),
        ) -> R.Tensor((num_tokens, 4096), dtype="float16"):
            R.func_attr({"num_input": 4})
            cls = Module

            with R.dataflow():
                # TODO(masahi): Workaround for the broken Relax cumsum op on GPU.
                # https://github.com/apache/tvm/issues/15851
                cumsum = R.call_dps_packed(
                    _ExternFunc("tvm.contrib.thrust.sum_scan"), seq_lens, ty_args=[seq_lens.ty]
                )
                max_seqlen_q = R.to_vdevice(R.max(seq_lens), "llvm:0")
                seqstart_q = R.concat([R.zeros((1,), "int32"), cumsum])
                q = R.reshape(queries, R.shape([1, num_tokens, 128, 32]))
                k = R.reshape(keys, R.shape([1, num_tokens, 128, 32]))
                v = R.reshape(values, R.shape([1, num_tokens, 128, 32]))
                attn_out = R.nn.attention_var_len(
                    q,
                    k,
                    v,
                    seqstart_q,
                    max_seqlen_q,
                    causal_mask="BottomRight",
                )
                out = R.reshape(attn_out, R.shape([num_tokens, 4096]))
                R.output(out)
            return out

    seq_lens = [5, 3, 8]
    num_head = 128
    head_size = 32

    _test_batched_var_len_attention(Module, seq_lens, num_head, num_head, head_size)


def test_batched_var_len_multi_query_attention():
    num_tokens = T.dynamic("num_tokens", "int64")
    num_seq = T.dynamic("num_seq", "int64")

    @I.ir_module
    class Module:
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def main(
            queries: R.Tensor((num_tokens, 4096), dtype="float16"),
            keys: R.Tensor((num_tokens, 512), dtype="float16"),
            values: R.Tensor((num_tokens, 512), dtype="float16"),
            seq_lens: R.Tensor((num_seq,), dtype="int32"),
        ) -> R.Tensor((num_tokens, 4096), dtype="float16"):
            R.func_attr({"num_input": 4})
            cls = Module

            with R.dataflow():
                # TODO(masahi): Workaround for the broken Relax cumsum op on GPU.
                # https://github.com/apache/tvm/issues/15851
                cumsum = R.call_dps_packed(
                    _ExternFunc("tvm.contrib.thrust.sum_scan"), seq_lens, ty_args=[seq_lens.ty]
                )
                max_seqlen_q = R.to_vdevice(R.max(seq_lens), "llvm:0")
                seqstart_q = R.concat([R.zeros((1,), "int32"), cumsum])
                q = R.reshape(queries, R.shape([1, num_tokens, 128, 32]))
                k = R.reshape(keys, R.shape([1, num_tokens, 16, 32]))
                v = R.reshape(values, R.shape([1, num_tokens, 16, 32]))
                attn_out = R.nn.attention_var_len(
                    q,
                    k,
                    v,
                    seqstart_q,
                    max_seqlen_q,
                    causal_mask="BottomRight",
                )
                out = R.reshape(attn_out, R.shape([num_tokens, 4096]))
                R.output(out)
            return out

    seq_lens = [5, 3, 8]
    num_head = 128
    num_kv_head = 16
    head_size = 32

    _test_batched_var_len_attention(Module, seq_lens, num_head, num_kv_head, head_size)


def test_sliding_window():
    q_shape = (1, 64, 16, 8)
    k_shape = v_shape = q_shape
    window_size = 8
    causal = "BottomRight"

    mod = get_relax_attention_module(
        q_shape,
        k_shape,
        v_shape,
        dtype="float16",
        causal_mask=causal,
        window_size=window_size,
    )

    q, k, v, _, ref = get_numpy_attention_ref(
        1, 64, 64, 16, 8, 8, "none", "none", causal, "float16", window_size=window_size
    )

    out = get_result_with_relax_cutlass_offload(mod, q, k, v, num_final_bindings=2)

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)

    ############# xformer reference for verification #############

    # attn_bias = BlockDiagonalCausalMask.from_seqlens([64])

    # if window_size > 0:
    #     attn_bias = attn_bias.make_local_attention(window_size)

    # query = torch.from_numpy(q).to("cuda")
    # key = torch.from_numpy(k).to("cuda")
    # value = torch.from_numpy(v).to("cuda")

    # ref = xops.memory_efficient_attention_forward(
    #     query, key, value, attn_bias=attn_bias,
    # ).cpu().numpy()

    # tvm.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_batched_var_len_sliding_window():
    num_tokens = T.dynamic("num_tokens", "int64")
    num_seq = T.dynamic("num_seq", "int64")

    @I.ir_module
    class Module:
        I.module_global_infos(
            {
                "vdevice": [
                    R.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def main(
            queries: R.Tensor((num_tokens, 4096), dtype="float16"),
            keys: R.Tensor((num_tokens, 4096), dtype="float16"),
            values: R.Tensor((num_tokens, 4096), dtype="float16"),
            seq_lens: R.Tensor((num_seq,), dtype="int32"),
        ) -> R.Tensor((num_tokens, 4096), dtype="float16"):
            R.func_attr({"num_input": 4})
            cls = Module

            with R.dataflow():
                # TODO(masahi): Workaround for the broken Relax cumsum op on GPU.
                # https://github.com/apache/tvm/issues/15851
                cumsum = R.call_dps_packed(
                    _ExternFunc("tvm.contrib.thrust.sum_scan"), seq_lens, ty_args=[seq_lens.ty]
                )
                max_seqlen_q = R.to_vdevice(R.max(seq_lens), "llvm:0")
                seqstart_q = R.concat([R.zeros((1,), "int32"), cumsum])
                q = R.reshape(queries, R.shape([1, num_tokens, 128, 32]))
                k = R.reshape(keys, R.shape([1, num_tokens, 128, 32]))
                v = R.reshape(values, R.shape([1, num_tokens, 128, 32]))
                attn_out = R.nn.attention_var_len(
                    q,
                    k,
                    v,
                    seqstart_q,
                    max_seqlen_q,
                    causal_mask="BottomRight",
                    window_size=T.IntImm("int32", 8),
                )
                out = R.reshape(attn_out, R.shape([num_tokens, 4096]))
                R.output(out)
            return out

    seq_lens = [64, 64, 64]
    num_head = 128
    num_kv_head = 128
    head_size = 32
    window_size = 8

    _test_batched_var_len_attention(Module, seq_lens, num_head, num_kv_head, head_size, window_size)


if __name__ == "__main__":
    tvm_tirx.testing.main()
