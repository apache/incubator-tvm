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
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import VDevice
from tvm.relax import script as R
from tvm_tirx import tirx
from tvm_tirx.ir import Op


def test_op_correctness():
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    assert relax.op.sort(x, axis=1).op == Op.get("relax.sort")
    assert relax.op.argsort(x, axis=1).op == Op.get("relax.argsort")
    assert relax.op.topk(x, k=1, axis=1).op == Op.get("relax.topk")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_sort_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(bb, relax.op.sort(x0, axis=1), relax.TensorType((2, 10, 4), "float32"))
    _check_inference(bb, relax.op.sort(x6, axis=1), relax.TensorType((2, 10, 4), "float32", vdev0))
    _check_inference(bb, relax.op.sort(x1, axis=1), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.sort(x2, axis=1), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.sort(x3, axis=1), relax.TensorType((2, 10, 4), dtype=""))
    _check_inference(bb, relax.op.sort(x4, axis=1), relax.TensorType(dtype="", ndim=3))
    _check_inference(bb, relax.op.sort(x5, axis=1), relax.TensorType(dtype=""))
    _check_inference(bb, relax.op.sort(x0), relax.TensorType((2, 10, 4), "float32"))
    _check_inference(
        bb,
        relax.op.sort(x0, axis=1, descending=False),
        relax.TensorType((2, 10, 4), "float32"),
    )


def test_sort_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(bb, relax.op.sort(x, axis=1), relax.TensorType((a, b, c), "float32"))
    _check_inference(bb, relax.op.sort(x), relax.TensorType((a, b, c), "float32"))


def test_sort_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))

    _check_inference(bb, relax.op.sort(x0, axis=1), relax.TensorType((2, 3, 4), "float16"))
    _check_inference(bb, relax.op.sort(x1, axis=1), relax.TensorType((2, 3, 4), "int8"))


def test_sort_wrong_input():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 4, 5), "float32")))
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    y = relax.Var("y", R.Tensor((2, 3, 4), "float32"))

    with pytest.raises(TypeError):
        relax.op.sort(x, y)

    with pytest.raises(TypeError):
        bb.normalize(relax.op.sort(x0, axis=1))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.sort(x1, axis=1))


def test_argsort_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.argsort(x0, axis=1, descending=False, dtype="int64"),
        relax.TensorType((2, 10, 4), "int64"),
    )
    _check_inference(bb, relax.op.argsort(x6, axis=1), relax.TensorType((2, 10, 4), "int32", vdev0))
    _check_inference(bb, relax.op.argsort(x1, axis=1), relax.TensorType(dtype="int32", ndim=3))
    _check_inference(
        bb, relax.op.argsort(x2, axis=1, dtype="float16"), relax.TensorType(dtype="float16")
    )
    _check_inference(bb, relax.op.argsort(x3, axis=1), relax.TensorType((2, 10, 4), dtype="int32"))
    _check_inference(bb, relax.op.argsort(x4, axis=1), relax.TensorType(dtype="int32", ndim=3))
    _check_inference(bb, relax.op.argsort(x5, axis=1), relax.TensorType(dtype="int32"))
    _check_inference(bb, relax.op.argsort(x0), relax.TensorType((2, 10, 4), "int32"))
    _check_inference(
        bb,
        relax.op.argsort(x0, axis=1, descending=False),
        relax.TensorType((2, 10, 4), "int32"),
    )


def test_argsort_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(bb, relax.op.argsort(x, axis=1), relax.TensorType((a, b, c), "int32"))
    _check_inference(bb, relax.op.argsort(x), relax.TensorType((a, b, c), "int32"))


def test_topk_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.topk(x0, k=5, axis=1, ret_type="both", largest=False, dtype="int64"),
        relax.TupleType(
            [
                relax.TensorType((2, 5, 4), "float32"),
                relax.TensorType((2, 5, 4), "int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x6),
        relax.TupleType(
            [
                relax.TensorType((2, 10, 1), "float32", vdev0),
                relax.TensorType((2, 10, 1), "int32", vdev0),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x1, k=3, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x2),
        relax.TupleType([relax.TensorType(dtype="float32"), relax.TensorType(dtype="int32")]),
    )
    _check_inference(
        bb,
        relax.op.topk(x3, axis=0),
        relax.TupleType(
            [
                relax.TensorType((1, 10, 4), None),
                relax.TensorType((1, 10, 4), dtype="int32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x4, axis=1),
        relax.TupleType(
            [
                relax.TensorType(ndim=3, dtype=None),
                relax.TensorType(dtype="int32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x5, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype=None),
                relax.TensorType(dtype="int32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x0),
        relax.TupleType(
            [
                relax.TensorType((2, 10, 1), "float32"),
                relax.TensorType((2, 10, 1), "int32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x0, k=-1),
        relax.TupleType(
            [
                relax.TensorType((2, 10, 4), "float32"),
                relax.TensorType((2, 10, 4), "int32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x0, k=6),
        relax.TupleType(
            [
                relax.TensorType((2, 10, 4), "float32"),
                relax.TensorType((2, 10, 4), "int32"),
            ]
        ),
    )


def test_topk_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(
        bb,
        relax.op.topk(x, axis=1),
        relax.TupleType(
            [
                relax.TensorType((a, 1, c), "float32"),
                relax.TensorType((a, 1, c), "int32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.topk(x, k=3),
        relax.TupleType(
            [
                relax.TensorType((a, b, 3), "float32"),
                relax.TensorType((a, b, 3), "int32"),
            ]
        ),
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
