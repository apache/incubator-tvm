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

import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx import tirx


@pytest.fixture(params=["by_string", "by_var"], scope="session")
def param_specification(request):
    return request.param


@pytest.fixture(params=["static_shape", "dynamic_shape", "ndim", "arbitrary"], scope="session")
def param_shape(request):
    return request.param


@pytest.fixture(params=["float32", None], scope="session")
def tensor_param_dtype(request):
    return request.param


def test_bind_tensor_param(param_specification, param_shape, tensor_param_dtype):
    if param_shape == "static_shape":
        shape = [16]
        ndim = -1
    elif param_shape == "dynamic_shape":
        shape = [tirx.Var("N", "int64")]
        ndim = -1
    elif param_shape == "ndim":
        shape = None
        ndim = 1
    elif param_shape == "arbitrary":
        shape = None
        ndim = -1
    else:
        raise ValueError(f"Unknown param_shape: {param_shape}")

    @R.function
    def before(A: R.Tensor(shape, ndim=ndim, dtype=tensor_param_dtype)):
        R.func_attr({"global_symbol": "main"})
        B: R.Tensor(shape=shape, ndim=ndim, dtype=tensor_param_dtype) = A
        out = R.add(B, B)
        return out

    np_data = np.arange(16).astype("float32")
    inlined_relax_const = relax.const(np_data)

    @R.function
    def expected() -> R.Tensor([16], "float32"):
        R.func_attr({"global_symbol": "main"})
        B = inlined_relax_const
        out = R.add(B, B)
        return out

    if param_specification == "by_string":
        var = "A"
    elif param_specification == "by_var":
        var = before.params[0]
    else:
        raise ValueError("Unknown param_specification: {param_specification}")

    after = before.bind_params({var: np.arange(16).astype("float32")})

    tvm_tirx.ir.assert_structural_equal(expected, after)


def test_bind_shape_param(param_shape):
    if param_shape == "static_shape":
        shape = [16]
        ndim = -1
    elif param_shape == "dynamic_shape":
        shape = [tirx.Var("N", "int64")]
        ndim = -1
    elif param_shape == "ndim":
        shape = None
        ndim = 1
    elif param_shape == "arbitrary":
        shape = None
        ndim = -1
    else:
        raise ValueError(f"Unknown param_shape: {param_shape}")

    @R.function
    def before(A: R.Shape(shape, ndim=ndim)):
        R.func_attr({"global_symbol": "main"})
        B: R.Shape(shape, ndim=ndim) = A
        return B

    @R.function
    def expected() -> R.Shape([16]):
        R.func_attr({"global_symbol": "main"})
        B = R.ShapeExpr([16])
        return B

    after = before.bind_params({"A": relax.ShapeExpr([16])})

    tvm_tirx.ir.assert_structural_equal(expected, after)


@pytest.fixture(params=["int64", "int32", "float32"], scope="session")
def prim_value_dtype(request):
    return request.param


def test_bind_prim_value(prim_value_dtype):
    prim_type = tvm_tirx.ir.PrimType(prim_value_dtype)
    param = relax.Var("A", prim_type)
    before = relax.Function([param], param, prim_type).with_attr("global_symbol", "main")
    value = tirx.const(16, prim_value_dtype)

    after = before.bind_params({"A": value})

    assert not after.params
    tvm_tirx.ir.assert_structural_equal(after.ret_ty, prim_type)
    tvm_tirx.ir.assert_structural_equal(after.body.body, value)


def test_error_on_unknown_var():
    @R.function
    def before(A: R.Tensor([16], dtype="float32")):
        R.func_attr({"global_symbol": "main"})
        return A

    unknown_var = relax.Var("unknown_var")

    with pytest.raises(RuntimeError):
        before.bind_params({unknown_var: np.arange(16).astype("float32")})


def test_error_on_unknown_var_name():
    @R.function
    def before(A: R.Tensor([16], dtype="float32")):
        R.func_attr({"global_symbol": "main"})
        return A

    with pytest.raises(RuntimeError):
        before.bind_params({"unknown_var_name": np.arange(16).astype("float32")})


if __name__ == "__main__":
    tvm_tirx.testing.main()
