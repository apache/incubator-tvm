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
# pylint: disable=missing-docstring, invalid-name
# ruff: noqa: E501, F841
import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax, s_tir
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.relax.frontend.nn import Module, Tensor, op, spec
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

# mypy: disable-error-code="attr-defined,valid-type,name-defined"


def test_unary():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = op.square(x)
            z1 = op.sqrt(x)
            return (z0, z1)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 10), dtype="float32"), _io: R.Any):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            square: R.Tensor((1, 10), dtype="float32") = R.square(x)
            sqrt: R.Tensor((1, 10), dtype="float32") = R.sqrt(x)
            gv1 = (square, sqrt), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": spec.Tensor([1, 10], "float32")}},
        debug=True,
    )
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_binary():
    class Model(Module):
        def test(self, x: Tensor, y: Tensor):
            z0 = op.add(x, y)
            z1 = op.multiply(x, y)
            z2 = op.divide(x, y)
            z3 = op.matmul(x, y)
            z4 = op.maximum(x, y)
            z5 = op.minimum(x, y)
            z6 = op.subtract(x, y)
            z7 = op.greater(x, y)
            z8 = op.greater_equal(x, y)
            z9 = op.less(x, y)
            z10 = op.less_equal(x, y)
            z11 = op.equal(x, y)
            z12 = op.not_equal(x, y)

            return (z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 10), dtype="float32"), y: R.Tensor((10, 1), dtype="float32"), _io: R.Any):
        R.func_attr({"num_input": 3})
        with R.dataflow():
            add: R.Tensor((10, 10), dtype="float32") = R.add(x, y)
            mul: R.Tensor((10, 10), dtype="float32") = R.multiply(x, y)
            divide: R.Tensor((10, 10), dtype="float32") = R.divide(x, y)
            matmul: R.Tensor((1, 1), dtype="float32") = R.matmul(x, y, out_dtype=None)
            maximum: R.Tensor((10, 10), dtype="float32") = R.maximum(x, y)
            minimum: R.Tensor((10, 10), dtype="float32") = R.minimum(x, y)
            subtract: R.Tensor((10, 10), dtype="float32") = R.subtract(x, y)
            greater: R.Tensor((10, 10), dtype="bool") = x > y
            greater_equal: R.Tensor((10, 10), dtype="bool") = x >= y
            less: R.Tensor((10, 10), dtype="bool") = x < y
            less_equal: R.Tensor((10, 10), dtype="bool") = x <= y
            equal: R.Tensor((10, 10), dtype="bool") = R.equal(x, y)
            not_equal: R.Tensor((10, 10), dtype="bool") = R.not_equal(x, y)
            gv1 = (add, mul, divide, matmul, maximum, minimum, subtract, greater, greater_equal, less, less_equal, equal, not_equal), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": spec.Tensor([1, 10], "float32"), "y": spec.Tensor([10, 1], "float32")}},
        debug=True,
    )
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_sum():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = op.sum(x, axis=[1, 2], keepdims=True)
            return z0

    # fmt: off
    @R.function
    def test(x: R.Tensor((3, 5, 2, 4), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tensor((3, 1, 1, 4), dtype="float32"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            sum: R.Tensor((3, 1, 1, 4), dtype="float32") = R.sum(x, axis=[1, 2], keepdims=True)
            gv1: R.Tuple(R.Tensor((3, 1, 1, 4), dtype="float32"), R.Tuple(R.Any)) = sum, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": spec.Tensor([3, 5, 2, 4], "float32")}}, debug=True
    )
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_max():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = op.max(x, axis=[1, 2], keepdims=True)
            return z0

    # fmt: off
    @R.function
    def test(x: R.Tensor((3, 5, 2, 4), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tensor((3, 1, 1, 4), dtype="float32"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            max: R.Tensor((3, 1, 1, 4), dtype="float32") = R.max(x, axis=[1, 2], keepdims=True)
            gv1: R.Tuple(R.Tensor((3, 1, 1, 4), dtype="float32"), R.Tuple(R.Any)) = max, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": spec.Tensor([3, 5, 2, 4], "float32")}}, debug=True
    )
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_min():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = op.min(x, axis=[1, 2], keepdims=True)
            return z0

    # fmt: off
    @R.function
    def test(x: R.Tensor((3, 5, 2, 4), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tensor((3, 1, 1, 4), dtype="float32"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            min: R.Tensor((3, 1, 1, 4), dtype="float32") = R.min(x, axis=[1, 2], keepdims=True)
            gv1: R.Tuple(R.Tensor((3, 1, 1, 4), dtype="float32"), R.Tuple(R.Any)) = min, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={"test": {"x": spec.Tensor([3, 5, 2, 4], "float32")}}, debug=True
    )
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_manipulate():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = op.broadcast_to(x, [2, 5, 2])
            z1 = op.permute_dims(x, [2, 1, 0])
            z2 = op.reshape(x, [1, 10])
            z3 = op.repeat(x, repeats=2, axis=1)
            z4 = op.squeeze(x, 0)
            z5 = op.unsqueeze(x, 0)
            z6 = op.concat([x, x], dim=0)
            return (z0, z1, z2, z3, z4, z5, z6)

    # fmt: off
    @R.function
    def test(x: R.Tensor((1, 5, 2), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tuple(R.Tensor((2, 5, 2), dtype="float32"), R.Tensor((2, 5, 1), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10, 2), dtype="float32"), R.Tensor((5, 2), dtype="float32"), R.Tensor((1, 1, 5, 2), dtype="float32"), R.Tensor((2, 5, 2), dtype="float32")), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            broadcast_to: R.Tensor((2, 5, 2), dtype="float32") = R.broadcast_to(x, R.shape([2, 5, 2]))
            permute_dims: R.Tensor((2, 5, 1), dtype="float32") = R.permute_dims(x, axes=[2, 1, 0])
            reshape: R.Tensor((1, 10), dtype="float32") = R.reshape(x, R.shape([1, 10]))
            repeat: R.Tensor((1, 10, 2), dtype="float32") = R.repeat(x, repeats=2, axis=1)
            squeeze: R.Tensor((5, 2), dtype="float32") = R.squeeze(x, axis=[0])
            unsqueeze: R.Tensor((1, 1, 5, 2), dtype="float32") = R.expand_dims(x, axis=0)
            concat: R.Tensor((2, 5, 2), dtype="float32") = R.concat([x, x], axis=0)
            gv1: R.Tuple(R.Tuple(R.Tensor((2, 5, 2), dtype="float32"), R.Tensor((2, 5, 1), dtype="float32"), R.Tensor((1, 10), dtype="float32"), R.Tensor((1, 10, 2), dtype="float32"), R.Tensor((5, 2), dtype="float32"), R.Tensor((1, 1, 5, 2), dtype="float32"), R.Tensor((2, 5, 2), dtype="float32")), R.Tuple(R.Any)) = (broadcast_to, permute_dims, reshape, repeat, squeeze, unsqueeze, concat), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([1, 5, 2], "float32")}}, debug=True)

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_index():
    class Model(Module):
        def test(self, x: Tensor, y: Tensor):
            z0 = op.take(x, y, axis=2)
            return z0

    # fmt: off
    @R.function
    def test(x: R.Tensor((2, 1, 10), dtype="float32"), y: R.Tensor((5,), dtype="int32"), _io: R.Any) -> R.Tuple(R.Tensor((2, 1, 5), dtype="float32"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 3})
        with R.dataflow():
            take: R.Tensor((2, 1, 5), dtype="float32") = R.take(x, y, axis=2)
            gv1: R.Tuple(R.Tensor((2, 1, 5), dtype="float32"), R.Tuple(R.Any)) = take, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(
        spec={"test": {"x": spec.Tensor([2, 1, 10], "float32"), "y": spec.Tensor([5], "int32")}},
        debug=True,
    )

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_datatype():
    class Model(Module):
        def test(self, x: Tensor):
            z0 = op.astype(x, "float16")
            return z0

    # fmt: off
    @R.function
    def test(x: R.Tensor((2, 1, 10), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tensor((2, 1, 10), dtype="float16"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            astype: R.Tensor((2, 1, 10), dtype="float16") = R.astype(x, dtype="float16")
            gv1: R.Tuple(R.Tensor((2, 1, 10), dtype="float16"), R.Tuple(R.Any)) = astype, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([2, 1, 10], "float32")}}, debug=True)

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_image():
    class Model(Module):
        def test(self, x: Tensor, weight: Tensor, bias: Tensor):
            padded = op.pad(x, [0, 0, 0, 0, 1, 1, 1, 1])
            conv2d = op.conv2d(padded, weight, bias)
            interpolate = op.interpolate(x, size=[40, 40])  # type: ignore
            return (conv2d, interpolate)

    @R.function
    def test(
        x: R.Tensor((1, 3, 32, 32), dtype="float32"),
        weight: R.Tensor((32, 3, 3, 3), dtype="float32"),
        bias: R.Tensor((32,), dtype="float32"),
        _io: R.Any,
    ) -> R.Tuple(
        R.Tuple(
            R.Tensor((1, 32, 32, 32), dtype="float32"), R.Tensor((1, 3, 40, 40), dtype="float32")
        ),
        R.Tuple(R.Any),
    ):
        R.func_attr({"num_input": 4})
        with R.dataflow():
            lv0: R.Tensor((1, 3, 34, 34), dtype="float32") = R.nn.pad(x, (0, 0, 0, 0, 1, 1, 1, 1))
            lv1: R.Tensor((1, 32, 32, 32), dtype="float32") = R.nn.conv2d(
                lv0,
                weight,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype=None,
            )
            lv2: R.Tensor((1, 32, 1, 1), dtype="float32") = R.reshape(bias, R.shape([1, 32, 1, 1]))
            conv2d: R.Tensor((1, 32, 32, 32), dtype="float32") = R.add(lv1, lv2)
            interpolate: R.Tensor((1, 3, 40, 40), dtype="float32") = R.image.resize2d(
                x,
                R.shape([40, 40]),
                roi=[T.float32(0), T.float32(0), T.float32(0), T.float32(0)],
                layout="NCHW",
                method="nearest_neighbor",
                coordinate_transformation_mode="asymmetric",
                rounding_method="round",
                cubic_alpha=-0.75,
                cubic_exclude=0,
                extrapolation_value=0,
                out_dtype=None,
            )
            gv1: R.Tuple(
                R.Tuple(
                    R.Tensor((1, 32, 32, 32), dtype="float32"),
                    R.Tensor((1, 3, 40, 40), dtype="float32"),
                ),
                R.Tuple(R.Any),
            ) = (conv2d, interpolate), (_io,)
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={
            "test": {
                "x": spec.Tensor([1, 3, 32, 32], "float32"),
                "weight": spec.Tensor([32, 3, 3, 3], "float32"),
                "bias": spec.Tensor([32], "float32"),
            }
        },
        debug=True,
    )
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_chunk():
    class Model(Module):
        def test(self, x: Tensor):
            chunk = op.chunk(x, chunks=4)
            return chunk

    @R.function
    def test(x: R.Tensor((8,), dtype="float32"), _io: R.Any) -> R.Tuple(
        R.Tuple(
            R.Tensor((2,), dtype="float32"),
            R.Tensor((2,), dtype="float32"),
            R.Tensor((2,), dtype="float32"),
            R.Tensor((2,), dtype="float32"),
        ),
        R.Tuple(R.Any),
    ):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            chunk: R.Tuple(
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
                R.Tensor((2,), dtype="float32"),
            ) = R.split(x, indices_or_sections=4, axis=0)
            chunk_0: R.Tensor((2,), dtype="float32") = chunk[0]
            chunk_1: R.Tensor((2,), dtype="float32") = chunk[1]
            chunk_2: R.Tensor((2,), dtype="float32") = chunk[2]
            chunk_3: R.Tensor((2,), dtype="float32") = chunk[3]
            gv1: R.Tuple(
                R.Tuple(
                    R.Tensor((2,), dtype="float32"),
                    R.Tensor((2,), dtype="float32"),
                    R.Tensor((2,), dtype="float32"),
                    R.Tensor((2,), dtype="float32"),
                ),
                R.Tuple(R.Any),
            ) = (chunk_0, chunk_1, chunk_2, chunk_3), (_io,)
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([8], "float32")}}, debug=True)
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_nn():
    class Model(Module):
        def test(self, x: Tensor, weight: Tensor, bias: Tensor):
            log_out = op.log(x)
            floor_out = op.floor(x)
            relu_out = op.relu(x)
            relu6_out = op.relu6(x)
            silu_out = op.silu(x)
            gelu_out = op.gelu(x)
            sigmoid_out = op.sigmoid(x)
            tanh_out = op.tanh(x)
            exp_out = op.exp(x)
            negative_out = op.negative(x)
            softplus_out = op.softplus(x, beta=1.0, threshold=20.0)
            softmax_out = op.softmax(x, axis=2)
            prelu_out = op.prelu(x, alpha=bias)
            rms_norm_out = op.rms_norm(x, weight, axes=[-2, -1])
            rms_norm_with_bias_out = op.rms_norm(x, weight, axes=[-2, -1])
            group_norm_out = op.group_norm(x, num_groups=1, weight=bias, bias=bias)
            return x

    @R.function
    def test(
        x: R.Tensor((2, 3, 4, 5), dtype="float32"),
        weight: R.Tensor((4, 5), dtype="float32"),
        bias: R.Tensor((3,), dtype="float32"),
        _io: R.Any,
    ) -> R.Tuple(R.Tensor((2, 3, 4, 5), dtype="float32"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 4})
        with R.dataflow():
            log: R.Tensor((2, 3, 4, 5), dtype="float32") = R.log(x)
            floor: R.Tensor((2, 3, 4, 5), dtype="float32") = R.floor(x)
            relu: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.relu(x)
            relu6: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.relu6(x)
            silu: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.silu(x)
            gelu: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.gelu(x)
            sigmoid: R.Tensor((2, 3, 4, 5), dtype="float32") = R.sigmoid(x)
            tanh: R.Tensor((2, 3, 4, 5), dtype="float32") = R.tanh(x)
            exp: R.Tensor((2, 3, 4, 5), dtype="float32") = R.exp(x)
            negative: R.Tensor((2, 3, 4, 5), dtype="float32") = R.negative(x)
            softplus: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.softplus(
                x, beta=1.0, threshold=20.0
            )
            softmax: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.softmax(x, axis=2)
            prelu: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.prelu(x, bias)
            rms_norm: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.rms_norm(
                x, weight, axes=[-2, -1], epsilon=1.0000000000000001e-05
            )
            rms_norm1: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.rms_norm(
                x, weight, axes=[-2, -1], epsilon=1.0000000000000001e-05
            )
            group_norm: R.Tensor((2, 3, 4, 5), dtype="float32") = R.nn.group_norm(
                x, bias, bias, num_groups=1, channel_axis=1, axes=[2, 3]
            )
            gv1: R.Tuple(R.Tensor((2, 3, 4, 5), dtype="float32"), R.Tuple(R.Any)) = x, (_io,)
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, params = m.export_tvm(
        spec={
            "test": {
                "x": spec.Tensor([2, 3, 4, 5], "float32"),
                "weight": spec.Tensor([4, 5], "float32"),
                "bias": spec.Tensor([3], "float32"),
            }
        },
        debug=True,
    )

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_create():
    class Model(Module):
        def test(self, x: Tensor):
            triu_out = op.triu(x)
            full_with_scalar_out = op.full([10, 10], fill_value=10)  # type: ignore
            full_with_FloatImm_out = op.full(
                [10, 10], fill_value=tvm_tirx.ir.FloatImm(ty="float32", value=10)
            )
            full_with_Tensor_out = op.full(
                [10, 10], fill_value=Tensor.from_scalar(10, dtype="float32")
            )
            zeros_out = op.zeros([10, 10])
            zeros_fp16_out = op.zeros([10, 10], dtype="float16")

            arange_out = op.arange(0, 10, 1, "float32")
            return x

    # fmt: off
    @R.function
    def test(x: R.Tensor((10, 10), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            triu: R.Tensor((10, 10), dtype="float32") = R.triu(x, k=0)
            full: R.Tensor((10, 10), dtype="float32") = R.full(R.shape([10, 10]), R.const(10, "float32"), dtype="float32")
            full1: R.Tensor((10, 10), dtype="float32") = R.full(R.shape([10, 10]), R.const(10, "float32"), dtype="float32")
            full2: R.Tensor((10, 10), dtype="float32") = R.full(R.shape([10, 10]), R.const(10, "float32"), dtype="float32")
            zeros: R.Tensor((10, 10), dtype="float32") = R.zeros(R.shape([10, 10]), dtype="float32")
            zeros1: R.Tensor((10, 10), dtype="float16") = R.zeros(R.shape([10, 10]), dtype="float16")
            arange: R.Tensor((10,), dtype="float32") = R.arange(T.int64(0), T.int64(10), T.int64(1), dtype="float32")
            gv1: R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Any)) = x, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    m = Model()
    irmodule, params = m.export_tvm(
        spec={"test": {"x": spec.Tensor([10, 10], "float32")}}, debug=True
    )

    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_timestep_embedding():
    class Model(Module):
        def test(self, x: Tensor):
            get_timestep_out = op.get_timestep_embedding(x, 10)
            return get_timestep_out

    @R.function
    def test(x: R.Tensor((3,), dtype="float32"), _io: R.Any) -> R.Tuple(
        R.Tensor((3, 10), dtype="float32"), R.Tuple(R.Any)
    ):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((3,), dtype="float32") = R.astype(x, dtype="float32")
            lv2: R.Tensor((3, 1), dtype="float32") = R.expand_dims(lv1, axis=[1])
            lv3: R.Tensor((5,), dtype="float32") = R.arange(
                R.prim_value(T.int64(0)),
                R.prim_value(T.int64(5)),
                R.prim_value(T.int64(1)),
                dtype="float32",
            )
            lv4: R.Tensor((5,), dtype="float32") = R.multiply(
                R.const(-9.2103404998779297, "float32"), lv3
            )
            lv5: R.Tensor((5,), dtype="float32") = R.divide(lv4, R.const(4, "float32"))
            lv6: R.Tensor((5,), dtype="float32") = R.exp(lv5)
            lv7: R.Tensor((1, 5), dtype="float32") = R.expand_dims(lv6, axis=[0])
            lv8: R.Tensor((3, 5), dtype="float32") = R.multiply(lv2, lv7)
            lv9: R.Tensor((3, 5), dtype="float32") = R.sin(lv8)
            lv10: R.Tensor((3, 5), dtype="float32") = R.cos(lv8)
            lv11: R.Tensor((3, 10), dtype="float32") = R.concat((lv9, lv10), axis=-1)
            get_timestep_embedding: R.Tensor((3, 10), dtype="float32") = R.astype(
                lv11, dtype="float32"
            )
            gv1: R.Tuple(R.Tensor((3, 10), dtype="float32"), R.Tuple(R.Any)) = (
                get_timestep_embedding,
                (_io,),
            )
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([3], "float32")}}, debug=True)
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_scaled_dot_product_attention():
    class Model(Module):
        def test(self, query: Tensor, key: Tensor, value: Tensor):
            scaled_dot_product_attention = op.scaled_dot_product_attention(query, key, value)
            return scaled_dot_product_attention

    @R.function
    def test(
        query: R.Tensor((1, 32, 32, 32), dtype="float32"),
        key: R.Tensor((1, 32, 32, 32), dtype="float32"),
        value: R.Tensor((1, 32, 32, 32), dtype="float32"),
        _io: R.Any,
    ) -> R.Tuple(R.Tensor((1, 32, 32, 32), dtype="float32"), R.Tuple(R.Any)):
        R.func_attr({"num_input": 4})
        with R.dataflow():
            scaled_dot_product_attention: R.Tensor((1, 32, 32, 32), dtype="float32") = (
                R.nn.attention(query, key, value, scale=None, causal_mask=None)
            )
            gv1: R.Tuple(R.Tensor((1, 32, 32, 32), dtype="float32"), R.Tuple(R.Any)) = (
                scaled_dot_product_attention,
                (_io,),
            )
            R.output(gv1)
        return gv1

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={
            "test": {
                "query": spec.Tensor([1, 32, 32, 32], "float32"),
                "key": spec.Tensor([1, 32, 32, 32], "float32"),
                "value": spec.Tensor([1, 32, 32, 32], "float32"),
            }
        },
        debug=True,
    )
    tvm_tirx.ir.assert_structural_equal(irmodule["test"], test)


def test_tensor_expr_op():
    class Model(Module):
        def test(self, x: Tensor):
            tensor_expr_op_out = op.tensor_expr_op(
                tensor_expr_func=lambda x: x + 1, name_hint="add_one", args=[x]
            )
            return tensor_expr_op_out

    # fmt: off
    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def add_one(A: T.Tensor((T.int64(10), T.int64(10)), "float32"), T_add: T.Tensor((T.int64(10), T.int64(10)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(T.int64(10), T.int64(10)):
                with Ts.sblock("T_add"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(A[v_ax0, v_ax1])
                    Ts.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = A[v_ax0, v_ax1] + T.float32(1)

        @R.function
        def _initialize_effect() -> R.Tuple(R.Any):
            with R.dataflow():
                _io: R.Any = R.null_value()
                lv: R.Tuple(R.Any) = (_io,)
                gv: R.Tuple(R.Any) = lv
                R.output(gv)
            return gv

        @R.function
        def test(x: R.Tensor((10, 10), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Any)):
            cls = Expected
            R.func_attr({"num_input": 2})
            with R.dataflow():
                lv1 = R.call_tir(cls.add_one, (x,), ty_args=[R.Tensor((10, 10), dtype="float32")])
                gv1: R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tuple(R.Any)) = lv1, (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"x": spec.Tensor([10, 10], "float32")}}, debug=True)

    tvm_tirx.ir.assert_structural_equal(irmodule, Expected)


def test_tensor_ir_op():
    num_q_heads, num_kv_heads, head_dim = 8, 8, 16
    fused_heads = num_q_heads + num_kv_heads * 2
    dtype = "float16"

    batch_size = T.dynamic("batch_size", "int64")
    seq_len = T.dynamic("seq_len", "int64")

    @Ts.function(private=True)
    def fused_rope(  # pylint: disable=too-many-locals
        qkv: T.Tensor((batch_size, seq_len, fused_heads, head_dim), dtype),
        offset: T.int64,
        q: T.Tensor((batch_size, seq_len, num_q_heads, head_dim), dtype),
        k: T.Tensor((batch_size, seq_len, num_kv_heads, head_dim), dtype),
        v: T.Tensor((batch_size, seq_len, num_kv_heads, head_dim), dtype),
    ):
        T.evaluate(offset)

    class Model(Module):
        def test(self, qkv: Tensor, offset: tirx.Var):
            tensor_expr_op_out = op.tensor_ir_op(
                fused_rope,
                "llama_fused_rope",
                args=[qkv, offset],
                out=[
                    Tensor.placeholder((1, 1, num_q_heads, head_dim), dtype),
                    Tensor.placeholder((1, 1, num_kv_heads, head_dim), dtype),
                    Tensor.placeholder((1, 1, num_kv_heads, head_dim), dtype),
                ],
            )
            return tensor_expr_op_out

    # fmt: off
    batch_size = T.dynamic("batch_size", "int64")
    seq_len = T.dynamic("seq_len", "int64")
    offset_1 = T.dynamic("offset_1", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def llama_fused_rope(qkv: T.Tensor((batch_size, seq_len, 24, 16), 'float16'), offset: T.int64, q: T.Tensor((batch_size, seq_len, 8, 16), 'float16'), k: T.Tensor((batch_size, seq_len, 8, 16), 'float16'), v: T.Tensor((batch_size, seq_len, 8, 16), 'float16')):

            T.evaluate(offset)

        @R.function
        def _initialize_effect() -> R.Tuple(R.Any):
            with R.dataflow():
                _io: R.Any = R.null_value()
                lv: R.Tuple(R.Any) = (_io,)
                gv: R.Tuple(R.Any) = lv
                R.output(gv)
            return gv

        @R.function
        def test(qkv: R.Tensor((1, 1, 24, 16), dtype="float16"), offset: R.Shape([offset_1]), _io: R.Any) -> R.Tuple(R.Tuple(R.Tensor((1, 1, 8, 16), dtype="float16"), R.Tensor((1, 1, 8, 16), dtype="float16"), R.Tensor((1, 1, 8, 16), dtype="float16")), R.Tuple(R.Any)):
            R.func_attr({"num_input": 3})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(cls.llama_fused_rope, (qkv, offset_1), ty_args=[_TupleType([R.Tensor((1, 1, 8, 16), dtype="float16"), R.Tensor((1, 1, 8, 16), dtype="float16"), R.Tensor((1, 1, 8, 16), dtype="float16")])])
                llama_fused_rope_0: R.Tensor((1, 1, 8, 16), dtype="float16") = lv1[0]
                llama_fused_rope_1: R.Tensor((1, 1, 8, 16), dtype="float16") = lv1[1]
                llama_fused_rope_2: R.Tensor((1, 1, 8, 16), dtype="float16") = lv1[2]
                gv1: R.Tuple(R.Tuple(R.Tensor((1, 1, 8, 16), dtype="float16"), R.Tensor((1, 1, 8, 16), dtype="float16"), R.Tensor((1, 1, 8, 16), dtype="float16")), R.Tuple(R.Any)) = (llama_fused_rope_0, llama_fused_rope_1, llama_fused_rope_2), (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={
            "test": {"qkv": spec.Tensor([1, 1, fused_heads, head_dim], "float16"), "offset": int}
        },
        debug=True,
    )
    tvm_tirx.ir.assert_structural_equal(irmodule, Expected)


def test_tensor_ir_inplace_op():
    hidden_size = 4096
    dtype = "float16"

    vocab_size = T.dynamic("vocab_size", "int64")
    seq_len = T.dynamic("seq_len", "int64")
    total_seq_len = T.dynamic("total_seq_len", "int64")

    @Ts.function
    def inplace_take(
        weight: T.Tensor((vocab_size, hidden_size), dtype),
        pos: T.Tensor((seq_len,), "int32"),
        embeddings: T.Tensor((total_seq_len, hidden_size), dtype),
        offset: T.int64,
    ):
        T.func_attr({"tirx.noalias": True})

        for ax0, ax1 in T.grid(seq_len, hidden_size):
            with Ts.sblock("T_take"):
                v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                Ts.reads(weight[pos[v0], v1], pos[v0])
                Ts.writes(embeddings[v0, v1])
                embeddings[v0 + offset, v1] = weight[pos[v0], v1]

    class Model(Module):
        def test(
            self, embedding_table: Tensor, input_ids: Tensor, embedding_dst: Tensor, offset: int
        ):
            tensor_expr_op_out = op.tensor_ir_inplace_op(
                inplace_take,
                "inplace_take",
                args=[embedding_table, input_ids, embedding_dst, offset],
                inplace_indices=[2],
                out=Tensor.placeholder(embedding_dst.shape, embedding_dst.dtype),
            )
            return tensor_expr_op_out

    vocab_size_inplace_take = T.dynamic("vocab_size", "int64")
    seq_len_inplace_take = T.dynamic("seq_len", "int64")
    total_seq_len_inplace_take = T.dynamic("total_seq_len", "int64")
    total_seq_len_test = T.dynamic("total_seq_len", "int64")
    offset_1 = T.dynamic("offset_1", "int64")
    vocab_size_test = T.dynamic("vocab_size", "int64")
    seq_len_test = T.dynamic("seq_len", "int64")

    @I.ir_module
    class Expected:
        @Ts.function
        def inplace_take(
            weight: T.Tensor((vocab_size_inplace_take, hidden_size), dtype),
            pos: T.Tensor((seq_len_inplace_take,), "int32"),
            embeddings: T.Tensor((total_seq_len_inplace_take, hidden_size), dtype),
            offset: T.int64,
        ):
            T.func_attr({"tirx.noalias": True})

            for ax0, ax1 in T.grid(seq_len_inplace_take, hidden_size):
                with Ts.sblock("T_take"):
                    v0, v1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(weight[pos[v0], v1], pos[v0])
                    Ts.writes(embeddings[v0, v1])
                    embeddings[v0 + offset, v1] = weight[pos[v0], v1]

        @R.function
        def _initialize_effect() -> R.Tuple(R.Any):
            with R.dataflow():
                _io: R.Any = R.null_value()
                lv: R.Tuple(R.Any) = (_io,)
                gv: R.Tuple(R.Any) = lv
                R.output(gv)
            return gv

        @R.function
        def test(
            embedding_table: R.Tensor((vocab_size_test, hidden_size), dtype),
            input_ids: R.Tensor((seq_len_test,), "int32"),
            embedding_dst: R.Tensor((total_seq_len_test, hidden_size), dtype),
            offset: R.Shape([offset_1]),
            packed_params: R.Tuple,
        ) -> R.Tensor((total_seq_len_test, hidden_size), dtype):
            R.func_attr({"num_input": 4})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir_inplace(
                    cls.inplace_take,
                    (embedding_table, input_ids, embedding_dst, offset_1),
                    ty_args=[R.Tensor((total_seq_len_test, hidden_size), dtype)],
                    inplace_indices=[2],
                )
                gv1: R.Tensor((total_seq_len_test, hidden_size), dtype) = lv1
                R.output(gv1)
            return gv1

    m = Model()
    irmodule, _ = m.export_tvm(
        spec={
            "test": {
                "embedding_table": spec.Tensor(["vocab_size", hidden_size], dtype),
                "input_ids": spec.Tensor(["seq_len"], "int32"),
                "embedding_dst": spec.Tensor(["total_seq_len", hidden_size], dtype),
                "offset": int,
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
        },
        debug=True,
    )
    tvm_tirx.ir.assert_structural_equal(irmodule, Expected)


def test_tensor_ir_op_no_tir_var():
    @Ts.function(private=True)
    def tir_func(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")):
        T.evaluate(0)

    class Model(Module):
        def test(self, A: Tensor):
            tensor_expr_op_out = op.tensor_ir_op(
                tir_func,
                "tir_func",
                args=[A],
                out=[Tensor.placeholder((16, 16), "float32")],
            )
            return tensor_expr_op_out

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def tir_func(A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")):
            T.evaluate(0)

        @R.function
        def test(A: R.Tensor((16, 16), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv = R.call_tir(cls.tir_func, (A,), ty_args=[R.Tensor((16, 16), dtype="float32")])
                gv: R.Tensor((16, 16), dtype="float32") = lv
                R.output(gv)
            return gv

    m = Model()
    irmodule, _ = m.export_tvm(spec={"test": {"A": spec.Tensor([16, 16], "float32")}})
    tvm_tirx.ir.assert_structural_equal(irmodule, Expected)


def test_extern():
    class Model(Module):
        def test(self, q: Tensor, k: Tensor, v: Tensor):
            b, s, h_q, d = q.shape
            tensor_expr_op_out = op.extern(
                name="flashinfer.single_decode",
                args=[q, k, v, 0, 0, 1.0, 10000.0],
                out=Tensor.placeholder((b, s, h_q * d), dtype="float16"),
            )
            return tensor_expr_op_out

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def _initialize_effect() -> R.Tuple(R.Any):
            with R.dataflow():
                _io: R.Any = R.null_value()
                lv: R.Tuple(R.Any) = (_io,)
                gv: R.Tuple(R.Any) = lv
                R.output(gv)
            return gv

        @R.function
        def test(q: R.Tensor((1, 1, 16, 8), dtype="float32"), k: R.Tensor((64, 16, 8), dtype="float32"), v: R.Tensor((64, 16, 8), dtype="float32"), _io: R.Any) -> R.Tuple(R.Tensor((1, 1, 128), dtype="float16"), R.Tuple(R.Any)):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                flashinfer_single_decode = R.call_dps_packed(_ExternFunc("flashinfer.single_decode"), (q, k, v, R.prim_value(0), R.prim_value(0), R.prim_value(T.float64(1)), R.prim_value(T.float64(10000))), ty_args=[R.Tensor((1, 1, 128), dtype="float16")])
                gv1: R.Tuple(R.Tensor((1, 1, 128), dtype="float16"), R.Tuple(R.Any)) = flashinfer_single_decode, (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    batch, seq, t, d, h_q, h_kv = 1, 1, 64, 8, 16, 16
    m = Model()
    irmodule, _ = m.export_tvm(
        spec={
            "test": {
                "q": spec.Tensor([batch, seq, h_q, d], "float32"),
                "k": spec.Tensor([t, h_kv, d], "float32"),
                "v": spec.Tensor([t, h_kv, d], "float32"),
            }
        },
        debug=True,
    )
    tvm_tirx.ir.assert_structural_equal(irmodule, Expected)


def test_empty():
    @tvm_ffi.register_global_func("test_empty_assert", override=True)
    def test_empty_assert(_lineo, x):
        assert x.shape == (10, 10)
        assert x.dtype == "float32"

    class Model(Module):
        def test(self):
            result = op.empty([10, 10], dtype="float32")
            op.debug_func("test_empty_assert", result)
            return result

    irmodule, _ = Model().export_tvm(spec={"test": {}}, debug=True)
    ex = tvm.compile(irmodule, "llvm")
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    effects = vm["_initialize_effect"]()
    vm["test"](*effects)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_multinomial_from_uniform():
    prob_shape = (3, 5)
    sample_shape = (6, 1)

    class Model(Module):
        def foo(self, prob: Tensor, uniform_sample: Tensor, sample_indices: Tensor):
            z0 = op.multinomial_from_uniform(prob, uniform_sample, sample_indices)
            return z0

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def _initialize_effect() -> R.Tuple(R.Any):
            with R.dataflow():
                _io: R.Any = R.null_value()
                lv: R.Tuple(R.Any) = (_io,)
                gv: R.Tuple(R.Any) = lv
                R.output(gv)
            return gv

        @R.function
        def foo(prob: R.Tensor((3, 5), dtype="float32"), uniform_sample: R.Tensor((6, 1), dtype="float32"), sample_indices: R.Tensor((6, 1), dtype="int64"), _io: R.Any) -> R.Tuple(R.Tensor((6, 1), dtype="int64"), R.Tuple(R.Any)):
            R.func_attr({"num_input": 4})
            with R.dataflow():
                multinomial_from_uniform: R.Tensor((6, 1), dtype="int64") = R.multinomial_from_uniform(prob, uniform_sample, sample_indices, dtype="int64")
                gv1: R.Tuple(R.Tensor((6, 1), dtype="int64"), R.Tuple(R.Any)) = multinomial_from_uniform, (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    m = Model()
    mod, _ = m.export_tvm(
        spec={
            "foo": {
                "prob": spec.Tensor(prob_shape, "float32"),
                "uniform_sample": spec.Tensor(sample_shape, "float32"),
                "sample_indices": spec.Tensor(sample_shape, "int64"),
            }
        },
        debug=True,
    )

    tvm_tirx.ir.assert_structural_equal(mod, Expected)

    target = tvm_tirx.target.Target("cuda", host="llvm")
    mod = relax.backend.DispatchSampling(target=target)(mod)
    mod = s_tir.transform.DefaultGPUSchedule(target=target)(mod)
    ex = tvm.compile(mod, target)
    dev = tvm_tirx.cuda(0)
    vm = relax.VirtualMachine(ex, dev)

    effects = vm["_initialize_effect"]()

    np_rand = np.random.rand(*prob_shape).astype(np.float32)
    # normalize it to get the random prob
    np_prob = np_rand / np_rand.sum(axis=1, keepdims=True)
    nd_prob = tvm_tirx.runtime.tensor(np_prob, dev)
    # special sample to get deterministic results
    nd_sample = tvm_tirx.runtime.tensor(
        np.array([[1], [0], [1], [1], [0], [1]]).astype(np.float32), dev
    )
    nd_sample_indices = tvm_tirx.runtime.tensor(
        np.array([[0], [1], [1], [2], [2], [2]]).astype(np.int64), dev
    )
    inputs = [nd_prob, nd_sample, nd_sample_indices, effects]
    res = vm["foo"](*inputs)
    tvm_tirx.testing.assert_allclose(
        res[0].numpy(), np.array([[4], [0], [4], [4], [0], [4]]).astype(np.int64)
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_sample_top_p_top_k_from_sorted_prob():
    prob_shape = (2, 3)
    sample_shape = (3, 1)

    class Model(Module):
        def foo(
            self,
            prob: Tensor,
            index: Tensor,
            top_p: Tensor,
            top_k: Tensor,
            uniform_sample: Tensor,
            sample_indices: Tensor,
        ):
            z0 = op.sample_top_p_top_k_from_sorted_prob(
                prob, index, top_p, top_k, uniform_sample, sample_indices
            )
            return z0

    # fmt: off
    batch_get_index_from_sorted = T.dynamic("batch", "int64")
    vocab_size_get_index_from_sorted = T.dynamic("vocab_size", "int64")
    out_batch = T.dynamic("out_batch", "int64")
    batch_get_renorm_prob = T.dynamic("batch", "int64")
    vocab_size_get_renorm_prob = T.dynamic("vocab_size", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def get_index_from_sorted(cumsum_sorted: T.Tensor((batch_get_index_from_sorted, vocab_size_get_index_from_sorted), "float32"), indices: T.Tensor((batch_get_index_from_sorted, vocab_size_get_index_from_sorted), 'int64'), renorm_prob: T.Tensor((batch_get_index_from_sorted, 1), "float32"), usample: T.Tensor((out_batch, 1), "float32"), sample_indices: T.Tensor((out_batch, 1), 'int64'), output_index: T.Tensor((out_batch, 1), 'int64')):

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(out_batch, vocab_size_get_index_from_sorted):
                with Ts.sblock("T_get_index_from_sorted"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(usample[v_ax0, T.int64(0)], cumsum_sorted[sample_indices[v_ax0, T.int64(0)], v_ax1 - T.int64(1):v_ax1 - T.int64(1) + T.int64(2)], sample_indices[v_ax0, T.int64(0)], renorm_prob[sample_indices[v_ax0, T.int64(0)], 0], indices[sample_indices[v_ax0, T.int64(0)], T.min(T.int64(0), v_ax1):T.min(T.int64(0), v_ax1) + (T.max(T.int64(0), v_ax1) + T.int64(1) - T.min(T.int64(0), v_ax1))])
                    Ts.writes(output_index[v_ax0, 0])
                    if usample[v_ax0, T.int64(0)] < cumsum_sorted[sample_indices[v_ax0, T.int64(0)], v_ax1] / renorm_prob[sample_indices[v_ax0, T.int64(0)], 0] or v_ax1 + T.int64(1) == vocab_size_get_index_from_sorted:
                        if v_ax1 == T.int64(0):
                            output_index[v_ax0, 0] = indices[sample_indices[v_ax0, T.int64(0)], 0]
                        else:
                            if usample[v_ax0, T.int64(0)] >= cumsum_sorted[sample_indices[v_ax0, T.int64(0)], v_ax1 - T.int64(1)] / renorm_prob[sample_indices[v_ax0, T.int64(0)], 0]:
                                output_index[v_ax0, 0] = indices[sample_indices[v_ax0, T.int64(0)], v_ax1]

        @Ts.function(private=True)
        def get_renorm_prob(cumsum_sorted: T.Tensor((batch_get_renorm_prob, vocab_size_get_renorm_prob), "float32"), top_p: T.Tensor((batch_get_renorm_prob, 1), "float32"), top_k: T.Tensor((batch_get_renorm_prob, 1), 'int64'), renorm_prob: T.Tensor((batch_get_renorm_prob, 1), "float32")):

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(batch_get_renorm_prob, vocab_size_get_renorm_prob):
                with Ts.sblock("T_get_renorm_prob"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(cumsum_sorted[v_ax0, T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)):T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)) + (T.max(T.max(T.int64(0), v_ax1), v_ax1 + T.int64(1)) + T.int64(1) - T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)))], top_p[v_ax0, 0], top_k[v_ax0, 0])
                    Ts.writes(renorm_prob[v_ax0, 0])
                    if not (cumsum_sorted[v_ax0, 0] < top_p[v_ax0, 0] and top_k[v_ax0, 0] > T.int64(1)):
                        renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, 0]
                    else:
                        if cumsum_sorted[v_ax0, v_ax1] < top_p[v_ax0, 0] and v_ax1 + T.int64(1) < top_k[v_ax0, 0]:
                            if v_ax1 + T.int64(1) == vocab_size_get_renorm_prob:
                                renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, v_ax1]
                            else:
                                if not (cumsum_sorted[v_ax0, v_ax1 + T.int64(1)] < top_p[v_ax0, 0] and v_ax1 + T.int64(1) + T.int64(1) < top_k[v_ax0, 0]):
                                    renorm_prob[v_ax0, 0] = cumsum_sorted[v_ax0, v_ax1 + T.int64(1)]

        @R.function
        def _initialize_effect() -> R.Tuple(R.Any):
            with R.dataflow():
                _io: R.Any = R.null_value()
                lv: R.Tuple(R.Any) = (_io,)
                gv: R.Tuple(R.Any) = lv
                R.output(gv)
            return gv

        @R.function
        def foo(prob: R.Tensor((2, 3), dtype="float32"), index: R.Tensor((2, 3), dtype="int64"), top_p: R.Tensor((2, 1), dtype="float32"), top_k: R.Tensor((2, 1), dtype="int64"), uniform_sample: R.Tensor((3, 1), dtype="float32"), sample_indices: R.Tensor((3, 1), dtype="int64"), _io: R.Any,) -> R.Tuple(R.Tensor((3, 1), dtype="int64"), R.Tuple(R.Any)):
            R.func_attr({"num_input": 7})
            cls = Expected
            with R.dataflow():
                cumsum: R.Tensor((2, 3), dtype="float32") = R.cumsum(prob, axis=1, dtype=None, exclusive=None)
                lv1 = R.call_tir(cls.get_renorm_prob, (cumsum, top_p, top_k), ty_args=[R.Tensor((2, 1), dtype="float32")])
                lv2 = R.call_tir(cls.get_index_from_sorted, (cumsum, index, lv1, uniform_sample, sample_indices), ty_args=[R.Tensor((3, 1), dtype="int64")])
                gv1: R.Tuple(R.Tensor((3, 1), dtype="int64"), R.Tuple(R.Any)) = lv2, (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    m = Model()
    mod, _ = m.export_tvm(
        spec={
            "foo": {
                "prob": spec.Tensor(prob_shape, "float32"),
                "index": spec.Tensor(prob_shape, "int64"),
                "top_p": spec.Tensor((prob_shape[0], 1), "float32"),
                "top_k": spec.Tensor((prob_shape[0], 1), "int64"),
                "uniform_sample": spec.Tensor(sample_shape, "float32"),
                "sample_indices": spec.Tensor(sample_shape, "int64"),
            }
        },
        debug=True,
    )

    tvm_tirx.ir.assert_structural_equal(mod, Expected)

    target = tvm_tirx.target.Target({"kind": "cuda", "libs": ["thrust"]}, host="llvm")
    mod = s_tir.transform.DefaultGPUSchedule(target=target)(mod)

    ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        vm = relax.VirtualMachine(ex, dev)
        effects = vm["_initialize_effect"]()
        sorted_prob = tvm_tirx.runtime.tensor(
            np.array([[0.5, 0.4, 0.1], [0.4, 0.3, 0.3]]).astype(np.float32), dev
        )
        indices = tvm_tirx.runtime.tensor(np.array([[2, 1, 0], [2, 0, 1]]).astype(np.int64), dev)
        top_p = tvm_tirx.runtime.tensor(np.array([[0.6], [0.9]]).astype(np.float32), dev)
        top_k = tvm_tirx.runtime.tensor(np.array([[3], [2]]).astype(np.int64), dev)
        usample = tvm_tirx.runtime.tensor(np.array([[0.5], [0.6], [0.7]]).astype(np.float32), dev)
        sample_indices = tvm_tirx.runtime.tensor(np.array([[0], [1], [1]]).astype(np.int64), dev)
        inputs = [sorted_prob, indices, top_p, top_k, usample, sample_indices, effects]
        res = vm["foo"](*inputs)
        tvm_tirx.testing.assert_allclose(res[0].numpy(), np.array([[2], [0], [0]]).astype(np.int64))

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_renormalize_top_p_top_k_prob():
    prob_shape = (2, 3)
    sample_shape = (2, 1)

    class Model(Module):
        def foo(
            self,
            prob: Tensor,
            sorted_prob: Tensor,
            top_p: Tensor,
            top_k: Tensor,
        ):
            z0 = op.renormalize_top_p_top_k_prob(prob, sorted_prob, top_p, top_k)
            return z0

    # fmt: off
    batch = T.dynamic("batch", "int64")
    vocab_size = T.dynamic("vocab_size", "int64")

    @I.ir_module
    class Expected:
        @Ts.function(private=True)
        def filter_with_top_p_top_k(A: T.Tensor((T.int64(2), T.int64(3)), "float32"), B: T.Tensor((T.int64(2), T.int64(1)), "float32"), filter_with_top_p_top_k: T.Tensor((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tirx.noalias": True})
            # with Ts.sblock("root"):
            for i, j in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("filter_with_top_p_top_k"):
                    v_i, v_j = Ts.axis.remap("SS", [i, j])
                    Ts.reads(B[v_i, T.int64(0)], A[v_i, v_j])
                    Ts.writes(filter_with_top_p_top_k[v_i, v_j])
                    filter_with_top_p_top_k[v_i, v_j] = T.Select(B[v_i, T.int64(0)] <= A[v_i, v_j], A[v_i, v_j], T.float32(0))

        @Ts.function(private=True)
        def get_renorm_cutoff(sorted_prob: T.Tensor((batch, vocab_size), "float32"), cumsum_sorted: T.Tensor((batch, vocab_size), "float32"), top_p: T.Tensor((batch, 1), "float32"), top_k: T.Tensor((batch, 1), 'int64'), cutoff: T.Tensor((batch, 1), "float32")):

            # with Ts.sblock("root"):
            for ax0, ax1 in T.grid(batch, vocab_size):
                with Ts.sblock("T_get_renorm_prob"):
                    v_ax0, v_ax1 = Ts.axis.remap("SS", [ax0, ax1])
                    Ts.reads(cumsum_sorted[v_ax0, T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)):T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)) + (T.max(T.max(T.int64(0), v_ax1), v_ax1 + T.int64(1)) + T.int64(1) - T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)))], top_p[v_ax0, 0], top_k[v_ax0, 0], sorted_prob[v_ax0, T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)):T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)) + (T.max(T.max(T.int64(0), v_ax1), v_ax1 + T.int64(1)) + T.int64(1) - T.min(T.min(T.int64(0), v_ax1), v_ax1 + T.int64(1)))])
                    Ts.writes(cutoff[v_ax0, 0])
                    if T.Cast("int32", cumsum_sorted[v_ax0, 0] < top_p[v_ax0, 0] and top_k[v_ax0, 0] > T.int64(1)) == 0:
                        cutoff[v_ax0, 0] = sorted_prob[v_ax0, 0]
                    else:
                        if T.Cast("int32", cumsum_sorted[v_ax0, v_ax1] < top_p[v_ax0, 0] and v_ax1 + T.int64(1) < top_k[v_ax0, 0]) == 1:
                            if v_ax1 + T.int64(1) == vocab_size:
                                cutoff[v_ax0, 0] = sorted_prob[v_ax0, v_ax1]
                            else:
                                if T.Cast("int32", cumsum_sorted[v_ax0, v_ax1 + T.int64(1)] < top_p[v_ax0, 0] and v_ax1 + T.int64(1) + T.int64(1) < top_k[v_ax0, 0]) == 0:
                                    cutoff[v_ax0, 0] = sorted_prob[v_ax0, v_ax1 + T.int64(1)]

        @R.function
        def _initialize_effect() -> R.Tuple(R.Any):
            with R.dataflow():
                _io: R.Any = R.null_value()
                lv: R.Tuple(R.Any) = (_io,)
                gv: R.Tuple(R.Any) = lv
                R.output(gv)
            return gv

        @R.function
        def foo(prob: R.Tensor((2, 3), dtype="float32"), sorted_prob: R.Tensor((2, 3), dtype="float32"), top_p: R.Tensor((2, 1), dtype="float32"), top_k: R.Tensor((2, 1), dtype="int64"), _io: R.Any) -> R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tuple(R.Any)):
            R.func_attr({"num_input": 5})
            cls = Expected
            with R.dataflow():
                cumsum: R.Tensor((2, 3), dtype="float32") = R.cumsum(sorted_prob, axis=1, dtype=None, exclusive=None)
                lv1 = R.call_tir(cls.get_renorm_cutoff, (sorted_prob, cumsum, top_p, top_k), ty_args=[R.Tensor((2, 1), dtype="float32")])
                lv2 = R.call_tir(cls.filter_with_top_p_top_k, (prob, lv1), ty_args=[R.Tensor((2, 3), dtype="float32")])
                sum: R.Tensor((2, 1), dtype="float32") = R.sum(lv2, axis=[1], keepdims=True)
                divide: R.Tensor((2, 3), dtype="float32") = R.divide(lv2, sum)
                gv1: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tuple(R.Any)) = divide, (_io,)
                R.output(gv1)
            return gv1
    # fmt: on

    m = Model()
    mod, _ = m.export_tvm(
        spec={
            "foo": {
                "prob": spec.Tensor(prob_shape, "float32"),
                "sorted_prob": spec.Tensor(prob_shape, "float32"),
                "top_p": spec.Tensor(sample_shape, "float32"),
                "top_k": spec.Tensor(sample_shape, "int64"),
            }
        },
        debug=True,
    )

    tvm_tirx.ir.assert_structural_equal(mod, Expected)

    target = tvm_tirx.target.Target({"kind": "cuda", "libs": ["thrust"]}, host="llvm")
    mod = relax.transform.LegalizeOps(target=target)(mod)
    mod = s_tir.transform.DefaultGPUSchedule(target=target)(mod)

    ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm_tirx.cuda(0)
        vm = relax.VirtualMachine(ex, dev)
        effects = vm["_initialize_effect"]()
        prob = tvm_tirx.runtime.tensor(
            np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]]).astype(np.float32), dev
        )
        sorted_prob = tvm_tirx.runtime.tensor(
            np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]]).astype(np.float32), dev
        )
        top_p = tvm_tirx.runtime.tensor(np.array([[0.6], [0.9]]).astype(np.float32), dev)
        top_k = tvm_tirx.runtime.tensor(np.array([[3], [2]]).astype(np.int64), dev)
        inputs = [prob, sorted_prob, top_p, top_k, effects]
        res = vm["foo"](*inputs)
        tvm_tirx.testing.assert_allclose(
            res[0].numpy(),
            np.array([[0, 0.375, 0.625], [0.3, 0.3, 0.4]]).astype(np.float32),
        )

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_sort_argsort_topk():
    class Model(Module):
        def foo(self, x: Tensor):
            z0 = op.sort(x, axis=-1, descending=True)
            z1 = op.argsort(x, axis=-1, descending=False)
            z2 = op.topk(x, k=2, axis=-1)
            return z0, z1, z2

    seq_len = T.dynamic("seq_len", "int64")

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((seq_len, 64), dtype="float16")):
            R.func_attr({"num_input": 1})
            with R.dataflow():
                sort = R.sort(x, axis=-1, descending=True)
                argsort = R.argsort(x, axis=-1, descending=False, dtype="int32")
                topk = R.topk(x, k=2, axis=-1, ret_type="both", largest=True, dtype="int32")
                topk_0 = topk[0]
                topk_1 = topk[1]
                gv = sort, argsort, (topk_0, topk_1)
                R.output(gv)
            return gv

    m = Model()
    mod, _ = m.export_tvm({"foo": {"x": spec.Tensor(("seq_len", 64), "float16")}})

    tvm_tirx.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
