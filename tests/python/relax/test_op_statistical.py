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
from collections.abc import Callable

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import VDevice
from tvm.relax import script as R
from tvm_tirx import tirx
from tvm_tirx.ir import Op


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    assert relax.op.max(x).op == Op.get("relax.max")
    assert relax.op.mean(x).op == Op.get("relax.mean")
    assert relax.op.min(x).op == Op.get("relax.min")
    assert relax.op.prod(x).op == Op.get("relax.prod")
    assert relax.op.std(x).op == Op.get("relax.std")
    assert relax.op.sum(x).op == Op.get("relax.sum")
    assert relax.op.variance(x).op == Op.get("relax.variance")
    assert relax.op.median(x).op == Op.get("relax.median")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_statistical_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))
    x4 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32", vdev0))

    _check_inference(bb, relax.op.sum(x0, axis=[1, 2]), relax.TensorType((2, 5), "float32"))
    _check_inference(bb, relax.op.sum(x4, axis=[1, 2]), relax.TensorType((2, 5), "float32", vdev0))
    _check_inference(
        bb,
        relax.op.sum(x0, axis=[1, 2], keepdims=True),
        relax.TensorType((2, 1, 1, 5), "float32"),
    )
    _check_inference(bb, relax.op.sum(x0, axis=None), relax.TensorType((), "float32"))
    _check_inference(
        bb,
        relax.op.sum(x0, axis=None, keepdims=True),
        relax.TensorType((1, 1, 1, 1), "float32"),
    )
    _check_inference(bb, relax.op.mean(x1, axis=[1, 2]), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(
        bb,
        relax.op.mean(x1, axis=[1, 2], keepdims=True),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(bb, relax.op.mean(x1, axis=None), relax.TensorType((), "float32"))
    _check_inference(
        bb,
        relax.op.mean(x1, axis=None, keepdims=True),
        relax.TensorType((1, 1, 1, 1), "float32"),
    )
    _check_inference(bb, relax.op.variance(x2, axis=[1, 2]), relax.TensorType(dtype="float32"))
    _check_inference(
        bb,
        relax.op.variance(x2, axis=[1, 2], keepdims=True),
        relax.TensorType(dtype="float32"),
    )
    _check_inference(bb, relax.op.variance(x2, axis=None), relax.TensorType((), "float32"))
    _check_inference(
        bb,
        relax.op.variance(x2, axis=None, keepdims=True),
        relax.TensorType(dtype="float32"),
    )
    _check_inference(bb, relax.op.max(x3, axis=[1, 2]), relax.TensorType((2, 5), dtype=""))
    _check_inference(
        bb,
        relax.op.max(x3, axis=[1, 2], keepdims=True),
        relax.TensorType((2, 1, 1, 5), dtype=""),
    )
    _check_inference(bb, relax.op.max(x3, axis=None), relax.TensorType((), dtype=""))
    _check_inference(
        bb,
        relax.op.max(x3, axis=None, keepdims=True),
        relax.TensorType((1, 1, 1, 1), dtype=""),
    )
    _check_inference(bb, relax.op.prod(x0, axis=[1, 2]), relax.TensorType((2, 5), "float32"))
    _check_inference(
        bb,
        relax.op.prod(x0, axis=[1, 2], keepdims=True),
        relax.TensorType((2, 1, 1, 5), "float32"),
    )
    _check_inference(bb, relax.op.std(x0, axis=[1, 2]), relax.TensorType((2, 5), "float32"))
    _check_inference(
        bb,
        relax.op.std(x0, axis=[1, 2], keepdims=True),
        relax.TensorType((2, 1, 1, 5), "float32"),
    )
    _check_inference(bb, relax.op.sum(x0, axis=[-1, -4]), relax.TensorType((3, 4), "float32"))
    _check_inference(bb, relax.op.sum(x0, axis=[]), relax.TensorType((2, 3, 4, 5), "float32"))


def test_statistical_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    d = tirx.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "float32"))

    _check_inference(bb, relax.op.min(x, axis=[1, 2]), relax.TensorType((a, d), "float32"))
    _check_inference(
        bb,
        relax.op.min(x, axis=[1, 2], keepdims=True),
        relax.TensorType((a, 1, 1, d), "float32"),
    )
    _check_inference(bb, relax.op.min(x, axis=None), relax.TensorType((), "float32"))
    _check_inference(
        bb,
        relax.op.min(x, axis=None, keepdims=True),
        relax.TensorType((1, 1, 1, 1), "float32"),
    )


def test_statistical_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=4))
    s1 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))

    _check_inference(bb, relax.op.max(x0), relax.TensorType((), dtype="float32"))
    _check_inference(
        bb, relax.op.max(x0, keepdims=True), relax.TensorType((1, 1, 1, 1), dtype="float32")
    )
    _check_inference(bb, relax.op.max(x0, axis=[2, 3]), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(
        bb,
        relax.op.max(x0, axis=[2, 3], keepdims=True),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(bb, relax.op.max(x1), relax.TensorType((), dtype="float32"))
    _check_inference(bb, relax.op.max(x1, keepdims=True), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.max(x1, axis=[2, 3]), relax.TensorType(dtype="float32"))
    _check_inference(
        bb, relax.op.max(x1, axis=[2, 3], keepdims=True), relax.TensorType(dtype="float32")
    )


def test_statistical_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))

    _check_inference(bb, relax.op.sum(x0), relax.TensorType((), "float16"))
    _check_inference(bb, relax.op.sum(x1), relax.TensorType((), "int8"))


def test_statistical_infer_ty_axis_out_of_range_repetitive():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.mean(x0, axis=[4]))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.mean(x1, axis=[3, 3]))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.mean(x0, axis=[-1, 3]))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.mean(x1, axis=[-4, -4]))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.mean(x0, axis=[-5]))


def test_statistical_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 4, 5), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.variance(x0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.variance(x1))


scan_ops = [
    relax.op.cumprod,
    relax.op.cumsum,
]


@pytest.mark.parametrize("scan_op", scan_ops)
def test_scan_op_infer_ty(scan_op: Callable):
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(bb, scan_op(x0, axis=1), relax.TensorType((2, 10, 4), "float32"))
    _check_inference(bb, scan_op(x6, axis=1), relax.TensorType((2, 10, 4), "float32", vdev0))
    _check_inference(bb, scan_op(x1, axis=1), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, scan_op(x2, axis=1), relax.TensorType(dtype="float32"))
    _check_inference(bb, scan_op(x3, axis=1), relax.TensorType((2, 10, 4), dtype=""))
    _check_inference(bb, scan_op(x4, axis=1), relax.TensorType(dtype="", ndim=3))
    _check_inference(bb, scan_op(x5, axis=1), relax.TensorType(dtype=""))
    _check_inference(bb, scan_op(x0), relax.TensorType((80,), "float32"))
    _check_inference(bb, scan_op(x0, axis=1, dtype="int32"), relax.TensorType((2, 10, 4), "int32"))


@pytest.mark.parametrize("scan_op", scan_ops)
def test_scan_op_infer_ty_shape_symbolic(scan_op: Callable):
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(bb, scan_op(x, axis=1), relax.TensorType((a, b, c), "float32"))
    _check_inference(bb, scan_op(x), relax.TensorType((a * b * c,), "float32"))


@pytest.mark.parametrize("scan_op", scan_ops)
def test_scan_op_infer_ty_more_input_dtype(scan_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))

    _check_inference(bb, scan_op(x0, axis=1), relax.TensorType((2, 3, 4), "float16"))
    _check_inference(bb, scan_op(x1, axis=1), relax.TensorType((2, 3, 4), "int8"))


@pytest.mark.parametrize("scan_op", scan_ops)
def test_scan_op_wrong_input_number(scan_op: Callable):
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    y = relax.Var("y", R.Tensor((2, 3, 4), "float32"))

    with pytest.raises(TypeError):
        scan_op(x, y)


@pytest.mark.parametrize("scan_op", scan_ops)
def test_scan_opinfer_ty_wrong_input_type(scan_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 4, 5), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(scan_op(x0, axis=1))
    with pytest.raises(TypeError):
        bb.normalize(scan_op(x1, axis=1))


def test_statistical_ext_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))
    x4 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.median(x0, axis=[1]),
        relax.TupleType(
            [
                relax.TensorType((2, 4, 5), "float32"),
                relax.TensorType((2, 4, 5), "int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x0, axis=[1], keepdims=True),
        relax.TupleType(
            [
                relax.TensorType((2, 1, 4, 5), "float32"),
                relax.TensorType((2, 1, 4, 5), "int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x1, axis=[1]),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x1, axis=[1], keepdims=True),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=4),
                relax.TensorType(dtype="int64", ndim=4),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x1, axis=None, keepdims=True),
        relax.TensorType((1, 1, 1, 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.median(x2, axis=[1]),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x2, axis=[1], keepdims=True),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64"),
            ]
        ),
    )
    _check_inference(bb, relax.op.median(x2, axis=None), relax.TensorType((), "float32"))
    _check_inference(
        bb,
        relax.op.median(x3, axis=[1], keepdims=True),
        relax.TupleType(
            [
                relax.TensorType((2, 1, 4, 5), dtype=""),
                relax.TensorType((2, 1, 4, 5), dtype="int64"),
            ]
        ),
    )
    _check_inference(bb, relax.op.median(x3, axis=None), relax.TensorType((), dtype=""))
    _check_inference(
        bb,
        relax.op.median(x3, axis=None, keepdims=True),
        relax.TensorType((1, 1, 1, 1), dtype=""),
    )
    _check_inference(
        bb,
        relax.op.median(x4, axis=[1]),
        relax.TupleType(
            [
                relax.TensorType((2, 4, 5), "float32", vdev0),
                relax.TensorType((2, 4, 5), "int64", vdev0),
            ]
        ),
    )


def test_statistical_ext_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    d = tirx.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "float32"))

    _check_inference(
        bb,
        relax.op.median(x, axis=[1]),
        relax.TupleType(
            [
                relax.TensorType((a, c, d), "float32"),
                relax.TensorType((a, c, d), "int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x, axis=[1], keepdims=True),
        relax.TupleType(
            [
                relax.TensorType((a, 1, c, d), "float32"),
                relax.TensorType((a, 1, c, d), "int64"),
            ]
        ),
    )
    _check_inference(bb, relax.op.median(x, axis=None), relax.TensorType((), "float32"))
    _check_inference(
        bb,
        relax.op.median(x, axis=None, keepdims=True),
        relax.TensorType((1, 1, 1, 1), "float32"),
    )


def test_statistical_ext_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=4))
    s1 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))

    _check_inference(bb, relax.op.median(x0), relax.TensorType((), dtype="float32"))
    _check_inference(
        bb,
        relax.op.median(x0, keepdims=True),
        relax.TensorType((1, 1, 1, 1), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.median(x0, axis=[2]),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x0, axis=[2], keepdims=True),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=4),
                relax.TensorType(dtype="int64", ndim=4),
            ]
        ),
    )
    _check_inference(bb, relax.op.median(x1), relax.TensorType((), dtype="float32"))
    _check_inference(
        bb,
        relax.op.median(x1, keepdims=True),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x1, axis=[2]),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.median(x1, axis=[2], keepdims=True),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64"),
            ]
        ),
    )


def test_statistical_ext_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))

    _check_inference(bb, relax.op.median(x0), relax.TensorType((), "float16"))
    _check_inference(bb, relax.op.median(x1), relax.TensorType((), "int8"))


def test_statistical_ext_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 4, 5), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.median(x0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.median(x1))


if __name__ == "__main__":
    tvm_tirx.testing.main()
