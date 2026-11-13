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
# ruff: noqa: RUF005
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.testing import env as tvm_env
from tvm_tirx.testing import env

pytest.importorskip("scipy")  # tvm.topi.testing imports scipy

import tvm.topi.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.backend.cuda.cublas import partition_for_cublas
from tvm.relax.script import ir_builder as relax_builder
from tvm.relax.testing import get_relax_matmul_module
from tvm_tirx.script.ir_builder import IRBuilder

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not tvm_env.has_cublas(), reason="need cublas"),
]


def build_and_run(mod, inputs_np, target, legalize=False, cuda_graph=False):
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

    if tvm_tirx.target.Target(target).kind.name == "cuda":
        return tvm_tirx.testing.run_with_gpu_lock(run_and_check)
    return run_and_check()


def get_result_with_relax_cublas_offload(mod, np_inputs, cuda_graph=False, bind_constants=False):
    mod = partition_for_cublas(mod, bind_constants=bind_constants)
    mod = relax.transform.RunCodegen()(mod)

    return build_and_run(mod, np_inputs, "cuda", cuda_graph)


def _to_concrete_shape(symbolic_shape, var_table):
    result = []
    for dim in symbolic_shape:
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
}


def get_relax_matmul_dequantize_module(
    x_shape,
    y_shape,
    in_dtype,
    out_dtype,
    transposed_y=False,
    scale_const=1.0,
    zero_point_const=0.0,
):
    """Create a matmul op followd by dequantize operations."""
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            x = R.arg_("x", R.Tensor(x_shape, in_dtype))
            y = R.arg_("y", R.Tensor(y_shape, in_dtype))

            with R.dataflow() as frame:
                if transposed_y:
                    axes = list(range(len(y_shape) - 2)) + [-1, -2]
                    y = R.emit(R.permute_dims(y, axes=axes))
                result = R.emit(R.matmul(x, y, out_dtype="float32"))
                result = R.emit(
                    R.dequantize(
                        result,
                        scale=R.const(scale_const, "float16"),
                        zero_point=R.const(zero_point_const, "float16"),
                        axis=-1,
                        out_dtype=out_dtype,
                    )
                )
                R.output(result)
            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


def get_relax_matmul_multiply_module(
    x_shape,
    y_shape,
    z_shape,
    in_dtype,
    acc_dtype,
    out_dtype,
    transposed_y=False,
):
    """Create a matmul op followd by multiply operations."""
    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name_("main")
            x = R.arg_("x", R.Tensor(x_shape, in_dtype))
            y = R.arg_("y", R.Tensor(y_shape, in_dtype))
            scaleA = R.arg_("scaleA", R.Tensor(z_shape, acc_dtype))
            scaleB = R.arg_("scaleB", R.Tensor(z_shape, acc_dtype))

            with R.dataflow() as frame:
                if transposed_y:
                    axes = list(range(len(y_shape) - 2)) + [-1, -2]
                    y = R.emit(R.permute_dims(y, axes=axes))
                result = R.emit(R.matmul(x, y, out_dtype=acc_dtype))
                z = R.emit(R.multiply(scaleA, scaleB))
                result = R.emit(R.multiply(result, z))
                if acc_dtype != out_dtype:
                    result = R.emit(R.astype(result, out_dtype))
                R.output(result)
            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm_tirx.IRModule({"main": func})


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, epilogue",
    [
        # Regular
        ((8, 8), (8, 8), False, "none"),
        ((_vars["a"], 6), (6, 16), False, "bias"),
        # Transposed
        ((4, 16), (16, 128), True, "relu"),
        ((35, 8), (8, 8), True, "gelu"),
        # # 3D x 3D
        ((6, 32, 8), (6, 8, 10), False, "bias"),
        ((6, 32, 8), (6, 8, 10), True, "none"),
        ((_vars["a"], 32, 8), (_vars["a"], 8, 10), True, "gelu"),
        # ND x ND
        ((5, 3, 32, 8), (5, 3, 8, 10), True, "relu"),
        ((_vars["a"], 3, 32, 8), (_vars["a"], 3, 8, 10), True, "relu"),
        ((_vars["a"], _vars["b"], 32, 8), (_vars["a"], _vars["b"], 8, 10), True, "relu"),
        # ND x 2D
        ((5, 3, 32, 8), (8, 10), False, "none"),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        ("float16", "float16"),
        ("float32", "float32"),
    ],
)
def test_matmul_offload(
    x_shape,
    y_shape,
    transpose_y,
    epilogue,
    in_dtype,
    out_dtype,
):
    with_bias, activation = _epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = _to_concrete_shape(x_shape, var_table)
    concrete_y_shape = _to_concrete_shape(y_shape, var_table)
    x = np.random.randn(*concrete_x_shape).astype(in_dtype)
    y = np.random.randn(*concrete_y_shape).astype(in_dtype)

    if transpose_y:
        y = np.swapaxes(y, -2, -1)
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    if with_bias:
        bias = np.random.randn(concrete_y_shape[-1]).astype(out_dtype)
        args = (x, y, bias)
    else:
        bias = None
        args = (x, y)

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        in_dtype,
        out_dtype,
        bias_shape=bias.shape if with_bias else None,
        transposed_y=transpose_y,
        activation=activation,
    )

    out = get_result_with_relax_cublas_offload(mod, args)
    ref = build_and_run(mod, args, "llvm", legalize=True)

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, epilogue",
    [
        # Regular
        ((8, 8), (8, 8), False, "none"),
        ((_vars["a"], 8), (8, 16), False, "none"),
        # Transposed
        ((4, 16), (16, 128), True, "none"),
        ((35, 16), (16, 128), False, "none"),
        # # 3D x 3D
        ((6, 32, 8), (6, 8, 12), False, "none"),
        ((6, 32, 8), (6, 8, 10), True, "none"),
        ((_vars["a"], 32, 8), (_vars["a"], 8, 10), True, "none"),
        # ND x ND
        ((5, 3, 32, 8), (5, 3, 8, 12), False, "none"),
        # ND x 2D
        ((5, 3, 32, 8), (8, 12), False, "none"),
    ],
)
def test_matmul_igemm_offload(
    x_shape,
    y_shape,
    transpose_y,
    epilogue,
):
    in_dtype = "int8"
    out_dtype = "int32"
    with_bias, activation = _epilogue_table[epilogue]
    var_table = {}
    concrete_x_shape = _to_concrete_shape(x_shape, var_table)
    concrete_y_shape = _to_concrete_shape(y_shape, var_table)
    x = np.random.randn(*concrete_x_shape).astype(in_dtype)
    y = np.random.randn(*concrete_y_shape).astype(in_dtype)

    if transpose_y:
        y = np.swapaxes(y, -2, -1)
        y_shape = (*y_shape[:-2], y_shape[-1], y_shape[-2])

    if with_bias:
        bias = np.random.randn(concrete_y_shape[-1]).astype(out_dtype)
        args = (x, y, bias)
    else:
        bias = None
        args = (x, y)

    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        in_dtype,
        out_dtype,
        bias_shape=bias.shape if with_bias else None,
        transposed_y=transpose_y,
        activation=activation,
    )

    out = get_result_with_relax_cublas_offload(mod, args)
    ref = build_and_run(mod, args, "llvm", legalize=True)

    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.skipif(ml_dtypes is None, reason="requires ml_dtypes to be installed")
@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, out_dtype",
    [
        ((10, 32), (64, 32), True, "float32"),
        ((32, 16), (32, 16), True, "float16"),
        ((2, 10, 32), (2, 64, 32), True, "float32"),
    ],
)
def test_matmul_fp8_offload(
    x_shape,
    y_shape,
    transpose_y,
    out_dtype,
):
    in_dtype = "float8_e4m3fn"
    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        in_dtype,
        out_dtype,
        bias_shape=None,
        transposed_y=transpose_y,
        activation=None,
    )
    numpytype = "float8_e4m3fn"
    x = np.random.uniform(low=0, high=5, size=x_shape).astype(numpytype)
    y = np.random.uniform(low=0, high=5, size=y_shape).astype(numpytype)
    z = np.swapaxes(y, -2, -1) if transpose_y else y
    args = (x, y)

    out = get_result_with_relax_cublas_offload(mod, args)
    ref_out = np.matmul(x, z).astype(out_dtype)

    tvm_tirx.testing.assert_allclose(out, ref_out, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.skipif(ml_dtypes is None, reason="requires ml_dtypes to be installed")
def test_matmul_fp8_dequantize_offload():
    x_shape = (10, 32)
    y_shape = (64, 32)
    in_dtype = "float8_e4m3fn"
    mod = get_relax_matmul_dequantize_module(
        x_shape,
        y_shape,
        in_dtype,
        "float16",
        transposed_y=True,
        scale_const=0.34786,
        zero_point_const=0.0,
    )

    numpytype = "float8_e4m3fn"
    x = np.random.uniform(low=0, high=5, size=x_shape).astype(numpytype)
    y = np.random.uniform(low=0, high=5, size=y_shape).astype(numpytype)
    args = (x, y)

    out = get_result_with_relax_cublas_offload(mod, args, bind_constants=True)
    ref = build_and_run(mod, args, "llvm", legalize=True)
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.skipif(ml_dtypes is None, reason="requires ml_dtypes to be installed")
def test_matmul_fp8_multiply_offload():
    x_shape = (10, 32)
    y_shape = (64, 32)
    z_shape = (1,)
    in_dtype, acc_dtype = ("float8_e4m3fn", "float32")

    mod = get_relax_matmul_multiply_module(
        x_shape,
        y_shape,
        z_shape,
        in_dtype,
        acc_dtype,
        "float16",
        transposed_y=True,
    )

    numpytype = "float8_e4m3fn"
    x = np.random.uniform(low=0, high=5, size=x_shape).astype(numpytype)
    y = np.random.uniform(low=0, high=5, size=y_shape).astype(numpytype)
    scaleA = np.random.uniform(low=0, high=5, size=z_shape).astype(acc_dtype)
    scaleB = np.random.uniform(low=0, high=5, size=z_shape).astype(acc_dtype)
    args = (x, y, scaleA, scaleB)

    out = get_result_with_relax_cublas_offload(mod, args)
    ref = build_and_run(mod, args, "llvm", legalize=True)
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(ml_dtypes is None, reason="requires ml_dtypes to be installed")
@pytest.mark.parametrize(
    "x_shape, y_shape, transpose_y, out_dtype",
    [
        ((10, 32), (64, 32), True, "float32"),
        ((32, 16), (32, 16), True, "float32"),
        ((2, 10, 32), (2, 64, 32), True, "float32"),
    ],
)
def test_matmul_bfloat16_offload(
    x_shape,
    y_shape,
    transpose_y,
    out_dtype,
):
    in_dtype = "bfloat16"
    mod = get_relax_matmul_module(
        x_shape,
        y_shape,
        in_dtype,
        out_dtype,
        bias_shape=None,
        transposed_y=transpose_y,
        activation=None,
    )
    # Generate input data in float32 and then convert to bfloat16 using ml_dtypes.
    x_float32 = np.random.uniform(low=0, high=5, size=x_shape).astype("float32")
    y_float32 = np.random.uniform(low=0, high=5, size=y_shape).astype("float32")
    x_bf16 = ml_dtypes.bfloat16(x_float32)
    y_bf16 = ml_dtypes.bfloat16(y_float32)

    # For the reference result, adjust y (if needed) in float32.
    z = np.swapaxes(y_float32, -2, -1) if transpose_y else y_float32
    args = (x_bf16, y_bf16)

    out = get_result_with_relax_cublas_offload(mod, args)
    ref_out = np.matmul(x_float32, z).astype(out_dtype)

    tvm_tirx.testing.assert_allclose(out, ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M, N, K, out_dtype, transposed_y, partition_done",
    [
        (15, 64, 32, "float32", True, True),
        (15, 64, 32, "float8_e4m3fn", True, True),
        (15, 64, 32, "float8_e5m2", True, False),
        (16, 32, 60, "float32", True, False),
        (16, 30, 64, "float32", True, False),
        (16, 8, 16, "float16", True, True),
        (16, 16, 16, "float16", False, False),
    ],
)
def test_cublas_partition_fp8_matmul(M, N, K, out_dtype, transposed_y, partition_done):
    mod = get_relax_matmul_module(
        (M, K), (N, K), "float8_e4m3fn", out_dtype, transposed_y=transposed_y
    )
    mod = partition_for_cublas(mod)
    func_name = "relax_matmul_cublas" if partition_done else "R.matmul"
    assert func_name in mod["main"].script()


@pytest.mark.parametrize(
    "M, N, K, scale, zp, num_bindings",
    [
        (16, 64, 32, 2.0, 0.0, 1),
        (16, 64, 32, 2.0, 1.0, 2),
        (16, 64, 32, [2.0] * 64, [2.0] * 64, 2),
    ],
)
def test_cublas_partition_fp8_matmul_dequantize(M, N, K, scale, zp, num_bindings):
    mod = get_relax_matmul_dequantize_module(
        (M, K),
        (N, K),
        "float8_e4m3fn",
        "float16",
        transposed_y=True,
        scale_const=scale,
        zero_point_const=zp,
    )
    mod = partition_for_cublas(mod)
    # Check whether R.dequantize is still in main function or not
    assert len(mod["main"].body.blocks[0].bindings) == num_bindings


def test_cublas_partition_fp8_matmul_multiply():
    M, N, K = (32, 64, 128)
    mod = get_relax_matmul_multiply_module(
        (M, K),
        (N, K),
        (1,),
        "float8_e4m3fn",
        "float32",
        "float16",
        transposed_y=True,
    )
    mod = partition_for_cublas(mod)
    assert len(mod["main"].body.blocks[0].bindings) == 1


def test_cublas_partition_matmul_without_bias():
    # cuBLAS does not handle 2D bias (residual input)
    mod = get_relax_matmul_module((16, 32), (32, 32), "float16", "float16", bias_shape=(16, 32))
    mod = partition_for_cublas(mod)

    # R.add is still in the main function
    assert len(mod["main"].body.blocks[0].bindings) == 2


@pytest.mark.parametrize(
    "M, N, K, was_partitioned", [(16, 8, 32, True), (16, 8, 33, False), (16, 9, 32, False)]
)
def test_cublas_partition_igemm(M, N, K, was_partitioned):
    mod = get_relax_matmul_module((M, K), (K, N), "int8", "int32")
    mod = partition_for_cublas(mod)
    func_name = "fused_relax_matmul_cublas" if was_partitioned else "R.matmul"
    assert func_name in mod["main"].script()


def test_cublas_partition_igemm_with_bias():
    mod = get_relax_matmul_module((16, 32), (32, 8), "int8", "int32", bias_shape=(8,))
    mod = partition_for_cublas(mod)
    func = mod["main"].script()
    assert "fused_relax_matmul_cublas" in func and "R.add" in func


def test_cublas_matmul_cuda_graph():
    @tvm_tirx.script.ir.ir_module
    class Mod:
        @R.function
        def main(
            x: R.Tensor((16, 16), "float16"),
            w0: R.Tensor((16, 16), "float16"),
            w1: R.Tensor((16, 16), "float16"),
            w2: R.Tensor((16, 16), "float16"),
        ):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.matmul(lv0, w1)
                lv2 = R.matmul(lv1, w2)
                R.output(lv2)
            return lv2

    mod = Mod
    shape = [16, 16]
    data = np.random.rand(*shape).astype(np.float16)
    w0 = np.random.rand(*shape).astype(np.float16)
    w1 = np.random.rand(*shape).astype(np.float16)
    w2 = np.random.rand(*shape).astype(np.float16)
    inputs = (data, w0, w1, w2)

    out = get_result_with_relax_cublas_offload(Mod, inputs, cuda_graph=True)

    mod = tvm.s_tir.transform.DefaultGPUSchedule(target=tvm_tirx.target.Target("cuda"))(mod)
    ref = build_and_run(mod, inputs, "llvm", legalize=True)
    tvm_tirx.testing.assert_allclose(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tvm_tirx.testing.main()
