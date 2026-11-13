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
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.testing import env as tvm_env

pytest.importorskip("scipy")  # tvm.topi.testing imports scipy

import tvm.topi.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.backend.rocm.hipblas import partition_for_hipblas
from tvm.relax.testing import get_relax_matmul_module

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


@pytest.fixture(autouse=True)
def reset_seed():
    np.random.seed(0)


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not tvm_env.has_hipblas(), reason="need hipblas"),
]


def build_and_run(mod, inputs_np, target, legalize=False):
    with tvm_tirx.ir.transform.PassContext(config={"relax.transform.apply_legalize_ops": legalize}):
        ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm_tirx.device_from_target(target, 0)
        vm = relax.VirtualMachine(ex, dev)
        f = vm["main"]
        inputs = [tvm_tirx.runtime.tensor(inp, dev) for inp in inputs_np]
        return f(*inputs).numpy()

    return tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def get_result_with_relax_cublas_offload(mod, np_inputs):
    mod = partition_for_hipblas(mod)
    mod = relax.transform.RunCodegen()(mod)

    return build_and_run(mod, np_inputs, "rocm")


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


def test_hipblas_partition_matmul_without_bias():
    # hipBLAS does not handle 2D bias (residual input)
    mod = get_relax_matmul_module((16, 32), (32, 32), "float16", "float16", bias_shape=(16, 32))
    mod = partition_for_hipblas(mod)

    # R.add is still in the main function
    assert len(mod["main"].body.blocks[0].bindings) == 2


if __name__ == "__main__":
    tvm_tirx.testing.main()
