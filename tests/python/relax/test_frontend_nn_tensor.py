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
# ruff: noqa: E501, F401
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm.relax.frontend.nn import Module, Tensor, spec


def test_tensor_from_numpy():
    x = np.random.rand(1, 10)
    tensor_x = Tensor.from_const(x)
    assert tensor_x.shape == [1, 10]
    assert tensor_x.ndim == 2
    assert isinstance(tensor_x.dtype, str)
    assert tensor_x.dtype == "float32"
    assert repr(tensor_x) == 'Tensor([1, 10], "float32")'


def test_tensor_from_scalar():
    x = 123.321
    tensor_x = Tensor.from_scalar(x, dtype="float16")
    assert tensor_x.shape == []
    assert tensor_x.ndim == 0
    assert tensor_x.dtype == "float16"
    assert repr(tensor_x) == 'Tensor([], "float16")'


def test_tensor_op_binary_tensor_tensor():
    class Model(Module):
        def test(self, x: Tensor, y: Tensor):
            z0 = x + y
            z1 = x * y
            z2 = x / y
            z3 = x.maximum(y)
            z4 = x.minimum(y)
            return (z0, z1, z2, z3, z4)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 10), dtype="float32"), y: R.Tensor((2, 1), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tuple(R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32")), R.Tuple(R.Any)):
        R.func_attr({"num_input": 3})
        with R.dataflow():
            add: R.Tensor((2, 10), dtype="float32") = R.add(x, y)
            mul: R.Tensor((2, 10), dtype="float32") = R.multiply(x, y)
            divide: R.Tensor((2, 10), dtype="float32") = R.divide(x, y)
            maximum: R.Tensor((2, 10), dtype="float32") = R.maximum(x, y)
            minimum: R.Tensor((2, 10), dtype="float32") = R.minimum(x, y)
            gv1: R.Tuple(R.Tuple(R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32"), R.Tensor((2, 10), dtype="float32")), R.Tuple(R.Any)) = (add, mul, divide, maximum, minimum), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": spec.Tensor([1, 10], "float32"), "y": spec.Tensor([2, 1], "float32")}},
        debug=True,
    )

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_tensor_op_binary_tensor_scalar():
    class Model(Module):
        def test(self, x: Tensor):
            y = 10
            z0 = x + y
            z1 = y + x
            z2 = x * y
            z3 = x / y
            z4 = x.maximum(y)
            z5 = x.minimum(y)
            return (z0, z1, z2, z3, z4, z5)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 10), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tuple(R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32")), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            add: R.Tensor((1, 10), dtype="float32") = R.add(x, R.const(10, "float32"))
            add1: R.Tensor((1, 10), dtype="float32") = R.add(x, R.const(10, "float32"))
            mul: R.Tensor((1, 10), dtype="float32") = R.multiply(x, R.const(10, "float32"))
            divide: R.Tensor((1, 10), dtype="float32") = R.divide(x, R.const(10, "float32"))
            maximum: R.Tensor((1, 10), dtype="float32") = R.maximum(x, R.const(10, "float32"))
            minimum: R.Tensor((1, 10), dtype="float32") = R.minimum(x, R.const(10, "float32"))
            gv1: R.Tuple(R.Tuple(R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10), dtype="float32")), R.Tuple(R.Any)) = (add, add1, mul, divide, maximum, minimum), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([1, 10], "float32")}}, debug=True)

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_tensor_op_datatype():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = x.astype(dtype="float16")
            return z0

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 10), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tensor((1, 10), dtype="float16"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            astype: R.Tensor((1, 10), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tuple(R.Tensor((1, 10), dtype="float16"), R.Tuple(R.Any)) = astype, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([1, 10], "float32")}}, debug=True)

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_tensor_op_manipulate():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = x.reshape(2, 5, 2)
            z1 = x.permute_dims(2, 1, 0)
            z2 = x.repeat(2, axis=1)
            return (z0, z1, z2)

    # fmt: off
    @R.function
    def test(x: R.Tensor((2, 1, 10), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tuple(R.Tensor((2, 5, 2), dtype="float32"), R.Tensor((10, 1, 2), dtype="float32"), R.Tensor((2, 2, 10), dtype="float32")), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            reshape: R.Tensor((2, 5, 2), dtype="float32") = R.reshape(x, R.shape([2, 5, 2]))
            permute_dims: R.Tensor((10, 1, 2), dtype="float32") = R.permute_dims(x, axes=[2, 1, 0])
            repeat: R.Tensor((2, 2, 10), dtype="float32") = R.repeat(x, repeats=2, axis=1)
            gv1: R.Tuple(R.Tuple(R.Tensor((2, 5, 2), dtype="float32"), R.Tensor((10, 1, 2), dtype="float32"), R.Tensor((2, 2, 10), dtype="float32")), R.Tuple(R.Any)) = (reshape, permute_dims, repeat), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([2, 1, 10], "float32")}}, debug=True)

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


if __name__ == "__main__":
    tvm_tirx.testing.main()
