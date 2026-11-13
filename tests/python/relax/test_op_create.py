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
from tvm_tirx.script import tirx as T


def test_op_correctness():
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    fill_value = relax.Var("fill_value", R.Tensor((), "float32"))
    assert relax.op.full((2, 3), fill_value).op == Op.get("relax.full")
    assert relax.op.full_like(x, fill_value).op == Op.get("relax.full_like")
    assert relax.op.ones((2, 3), "float32").op == Op.get("relax.ones")
    assert relax.op.ones_like(x).op == Op.get("relax.ones_like")
    assert relax.op.zeros((2, 3), "float32").op == Op.get("relax.zeros")
    assert relax.op.zeros_like(x).op == Op.get("relax.zeros_like")
    assert relax.op.arange(3, 4, 1, "float32").op == Op.get("relax.arange")
    assert relax.op.tril(x).op == Op.get("relax.tril")
    assert relax.op.triu(x).op == Op.get("relax.triu")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_full_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    v0 = relax.Var("v", R.Tensor((), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=0))
    v2 = relax.Var("v", R.Tensor(()))
    v3 = relax.Var("v", R.Tensor(ndim=0))
    v4 = relax.Var("v", R.Tensor((), "float32", vdev0))
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeType((2, 3)))
    s2 = relax.Var("s", relax.ShapeType(ndim=2))
    s3 = relax.Var("s", relax.ShapeType())

    _check_inference(bb, relax.op.full((2, 3), v0, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full((2, 3), v0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full(s0, v0, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full(s0, v0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full(s0, v4), relax.TensorType((2, 3), "float32", vdev0))
    _check_inference(bb, relax.op.full(s1, v0, "float16"), relax.TensorType(s1, "float16"))
    _check_inference(bb, relax.op.full(s1, v0), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.full(s2, v0, "float16"), relax.TensorType(s2, "float16"))
    _check_inference(bb, relax.op.full(s2, v0), relax.TensorType(s2, "float32"))
    _check_inference(bb, relax.op.full(s3, v0, "float16"), relax.TensorType(s3, "float16"))
    _check_inference(bb, relax.op.full(s3, v0), relax.TensorType(s3, "float32"))
    _check_inference(bb, relax.op.full((2, 3), v1, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full((2, 3), v1), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full(s0, v1, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full(s0, v1), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full(s1, v1, "float16"), relax.TensorType(s1, "float16"))
    _check_inference(bb, relax.op.full(s1, v1), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.full(s2, v1, "float16"), relax.TensorType(s2, "float16"))
    _check_inference(bb, relax.op.full(s2, v1), relax.TensorType(s2, "float32"))
    _check_inference(bb, relax.op.full(s3, v1, "float16"), relax.TensorType(s3, "float16"))
    _check_inference(bb, relax.op.full(s3, v1), relax.TensorType(s3, "float32"))
    _check_inference(bb, relax.op.full((2, 3), v2, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full((2, 3), v2), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.full(s0, v2, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full(s0, v2), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.full(s1, v2, "float16"), relax.TensorType(s1, "float16"))
    _check_inference(bb, relax.op.full(s1, v2), relax.TensorType(s1, dtype=None))
    _check_inference(bb, relax.op.full(s2, v2, "float16"), relax.TensorType(s2, "float16"))
    _check_inference(bb, relax.op.full(s2, v2), relax.TensorType(s2, dtype=None))
    _check_inference(bb, relax.op.full(s3, v2, "float16"), relax.TensorType(s3, "float16"))
    _check_inference(bb, relax.op.full(s3, v2), relax.TensorType(s3, dtype=None))
    _check_inference(bb, relax.op.full((2, 3), v3, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full((2, 3), v3), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.full(s0, v3, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full(s0, v3), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.full(s1, v3, "float16"), relax.TensorType(s1, "float16"))
    _check_inference(
        bb,
        relax.op.full(
            s1,
            v3,
        ),
        relax.TensorType(s1, dtype=None),
    )
    _check_inference(bb, relax.op.full(s2, v3, "float16"), relax.TensorType(s2, "float16"))
    _check_inference(
        bb,
        relax.op.full(
            s2,
            v3,
        ),
        relax.TensorType(s2, dtype=None),
    )
    _check_inference(bb, relax.op.full(s3, v3, "float16"), relax.TensorType(s3, "float16"))
    _check_inference(bb, relax.op.full(s3, v3), relax.TensorType(s3, dtype=None))


def test_full_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    v = relax.Var("v", R.Tensor((), "float32"))
    s0 = relax.ShapeExpr((a, 3))
    s1 = relax.Var("s", relax.ShapeType((a, 3)))

    _check_inference(bb, relax.op.full((a, 3), v, "float16"), relax.TensorType((a, 3), "float16"))
    _check_inference(bb, relax.op.full((a, 3), v), relax.TensorType((a, 3), "float32"))
    _check_inference(bb, relax.op.full(s0, v, "float16"), relax.TensorType((a, 3), "float16"))
    _check_inference(bb, relax.op.full(s0, v), relax.TensorType((a, 3), "float32"))
    _check_inference(bb, relax.op.full(s1, v, "float16"), relax.TensorType(s1, "float16"))
    _check_inference(bb, relax.op.full(s1, v), relax.TensorType(s1, "float32"))


def test_full_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(()))
    s1 = relax.Var("s", relax.ShapeType(ndim=0))
    v0 = relax.Var("v", relax.TensorType(s0, "float32"))
    v1 = relax.Var("v", relax.TensorType(s1, "float32"))

    _check_inference(bb, relax.op.full((2, 3), v0, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full((2, 3), v1, "float16"), relax.TensorType((2, 3), "float16"))


def test_full_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    v0 = relax.Var("v", R.Tensor((), "float16"))
    v1 = relax.Var("v", R.Tensor((), "int8"))
    v2 = relax.Var("v", R.Tensor((), "int32"))

    _check_inference(bb, relax.op.full((2, 3), v0, "float32"), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full((2, 3), v0), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full((2, 3), v1, "int32"), relax.TensorType((2, 3), "int32"))
    _check_inference(bb, relax.op.full((2, 3), v1), relax.TensorType((2, 3), "int8"))
    _check_inference(bb, relax.op.full((2, 3), v2, "int8"), relax.TensorType((2, 3), "int8"))
    _check_inference(bb, relax.op.full((2, 3), v2), relax.TensorType((2, 3), "int32"))


def test_full_infer_ty_fill_value_not_scalar_tensor():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType((1,)))
    s1 = relax.Var("s", relax.ShapeType(ndim=1))
    s2 = relax.Var("s", relax.ShapeType())
    v0 = relax.Var("v", R.Tensor((1,), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=1))
    v2 = relax.Var("v", R.Tensor("float32"))
    v3 = relax.Var("v", relax.TensorType(s0, "float32"))
    v4 = relax.Var("v", relax.TensorType(s1, "float32"))
    v5 = relax.Var("v", relax.TensorType(s2, "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.full((2, 3), v0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full((2, 3), v1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full((2, 3), v2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full((2, 3), v3))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full((2, 3), v4))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full((2, 3), v5))


def test_full_shape_not_tuple():
    m = tirx.Var("m", "int64")
    v = relax.Var("v", R.Tensor((), "float32"))

    with pytest.raises(TypeError):
        relax.op.full(4, v)
    with pytest.raises(TypeError):
        relax.op.full(m, v)


def test_full_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    v0 = relax.Var("v", R.Tensor((), "float32"))
    v1 = relax.Var("v", relax.ShapeType(()))
    v2 = relax.Var("v", relax.FuncType([], R.Tensor((), "float32")))
    s = relax.Var("s", R.Tensor((2, 3)))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.full(s, v0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full((2, 3), v1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full((2, 3), v2))


def test_full_like_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())
    v0 = relax.Var("v", R.Tensor((), "float16"))
    v1 = relax.Var("v", R.Tensor("float16", ndim=0))
    v2 = relax.Var("v", R.Tensor(()))
    v3 = relax.Var("v", R.Tensor(ndim=0))

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v2), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v3), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x1, v0), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.full_like(x1, v1), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.full_like(x1, v2), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.full_like(x1, v3), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.full_like(x2, v0), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v1), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v2), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v3), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x3, v0), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.full_like(x3, v1), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.full_like(x3, v2), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.full_like(x3, v3), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.full_like(x4, v0), relax.TensorType(dtype=None, ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v1), relax.TensorType(dtype=None, ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v2), relax.TensorType(dtype=None, ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v3), relax.TensorType(dtype=None, ndim=2))
    _check_inference(bb, relax.op.full_like(x5, v0), relax.TensorType(dtype=None))
    _check_inference(bb, relax.op.full_like(x5, v1), relax.TensorType(dtype=None))
    _check_inference(bb, relax.op.full_like(x5, v2), relax.TensorType(dtype=None))
    _check_inference(bb, relax.op.full_like(x5, v3), relax.TensorType(dtype=None))
    _check_inference(
        bb, relax.op.full_like(x0, v0, dtype="float16"), relax.TensorType((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.full_like(x0, v2, dtype="float16"), relax.TensorType((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.full_like(x3, v0, dtype="float16"), relax.TensorType((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.full_like(x3, v2, dtype="float16"), relax.TensorType((2, 3), "float16")
    )


def test_full_like_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))
    v = relax.Var("v", R.Tensor((), "float16"))

    _check_inference(bb, relax.op.full_like(x0, v), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.full_like(x1, v), relax.TensorType((m, n), dtype=None))


def test_full_like_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    s0 = relax.Var("s", relax.ShapeType((2, 3)))
    s1 = relax.Var("s", relax.ShapeType(ndim=2))
    s2 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s2, "float32"))
    x3 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x4 = relax.Var("x", R.Tensor((2, 3), "float32", vdev0))
    sv0 = relax.Var("sv", relax.ShapeType(()))
    sv1 = relax.Var("sv", relax.ShapeType(ndim=0))
    v0 = relax.Var("v", relax.TensorType(sv0, "float16"))
    v1 = relax.Var("v", relax.TensorType(sv1, "float16"))
    v2 = relax.Var("v", R.Tensor((), "float16"))
    v3 = relax.Var("v", relax.TensorType(sv1, "float16", vdev0))

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x0, v2), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v0), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v1), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v2), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v0), relax.TensorType(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v1), relax.TensorType(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v2), relax.TensorType(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x3, v0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x3, v1), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x4, v3), relax.TensorType((2, 3), "float32", vdev0))


def test_full_like_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    v0 = relax.Var("v", R.Tensor((), "int32"))
    v1 = relax.Var("v", R.Tensor((), "float64"))

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.full_like(x1, v0), relax.TensorType((2, 3), "int8"))
    _check_inference(bb, relax.op.full_like(x1, v1), relax.TensorType((2, 3), "int8"))


def test_full_like_infer_ty_fill_value_not_scalar_tensor():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    s0 = relax.Var("s", relax.ShapeType((1,)))
    s1 = relax.Var("s", relax.ShapeType(ndim=1))
    s2 = relax.Var("s", relax.ShapeType())
    v0 = relax.Var("v", R.Tensor((1,), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=1))
    v2 = relax.Var("v", R.Tensor("float32"))
    v3 = relax.Var("v", relax.TensorType(s0, "float32"))
    v4 = relax.Var("v", relax.TensorType(s1, "float32"))
    v5 = relax.Var("v", relax.TensorType(s2, "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.full_like(x, v0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full_like(x, v1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full_like(x, v2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full_like(x, v3))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full_like(x, v4))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.full_like(x, v5))


def test_full_like_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3)))
    v0 = relax.Var("v", R.Tensor(()))
    v1 = relax.Var("v", relax.ShapeType(()))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.full_like(x0, v0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.full_like(x1, v0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.full_like(x2, v1))


def test_ones_zeros_infer_ty():
    bb = relax.BlockBuilder()
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeType((2, 3)))
    s2 = relax.Var("s", relax.ShapeType(ndim=2))
    s3 = relax.Var("s", relax.ShapeType())

    _check_inference(bb, relax.op.ones((2, 3), "float32"), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.ones(s0, "float32"), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.ones(s1, "float32"), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.ones(s2, "float32"), relax.TensorType(s2, "float32"))
    _check_inference(bb, relax.op.ones(s3, "float32"), relax.TensorType(s3, "float32"))
    _check_inference(bb, relax.op.zeros((2, 3), "float32"), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.zeros(s0, "float32"), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.zeros(s1, "float32"), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.zeros(s2, "float32"), relax.TensorType(s2, "float32"))
    _check_inference(bb, relax.op.zeros(s3, "float32"), relax.TensorType(s3, "float32"))


def test_ones_zeros_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    s0 = relax.ShapeExpr((m, n))
    s1 = relax.Var("s", relax.ShapeType((m, n)))

    _check_inference(bb, relax.op.ones((m, n), "float32"), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.ones(s0, "float32"), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.ones(s1, "float32"), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.zeros((m, n), "float32"), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.zeros(s0, "float32"), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.zeros(s1, "float32"), relax.TensorType(s1, "float32"))


def test_ones_zeros_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeType((2, 3)))
    s2 = relax.Var("s", relax.ShapeType(ndim=2))
    s3 = relax.Var("s", relax.ShapeType())

    _check_inference(bb, relax.op.ones(s0, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.ones(s1, "int8"), relax.TensorType(s1, "int8"))
    _check_inference(bb, relax.op.zeros(s2, "int32"), relax.TensorType(s2, "int32"))
    _check_inference(bb, relax.op.zeros(s3, "float64"), relax.TensorType(s3, "float64"))


def test_ones_zeros_shape_not_tuple():
    m = tirx.Var("m", "int64")

    with pytest.raises(TypeError):
        relax.op.ones(10, "float32")
    with pytest.raises(TypeError):
        relax.op.zeros(m, "float32")


def test_ones_zeros_wrong_dtype():
    with pytest.raises(TypeError):
        relax.op.ones((2, 3))
    with pytest.raises(TypeError):
        relax.op.zeros((2, 3))


def test_ones_zeros_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", R.Tensor((2, 3)))
    s1 = relax.Var("s", relax.FuncType([], R.Tensor((2, 3))))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.ones(s0, "float32"))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.zeros(s1, "float32"))


def test_ones_like_zeros_like_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.ones_like(x2), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.zeros_like(x3), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.ones_like(x4), relax.TensorType(dtype=None, ndim=2))
    _check_inference(bb, relax.op.zeros_like(x5), relax.TensorType(dtype=None))
    _check_inference(
        bb, relax.op.ones_like(x0, dtype="float16"), relax.TensorType((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.zeros_like(x3, dtype="float16"), relax.TensorType((2, 3), "float16")
    )


def test_ones_like_zeros_like_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorType((m, n), dtype=None))


def test_ones_like_zeros_like_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType((2, 3)))
    s1 = relax.Var("s", relax.ShapeType(ndim=2))
    s2 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s2, "float32"))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.zeros_like(x2), relax.TensorType(s2, "float32"))


def test_ones_like_zeros_like_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorType((2, 3), "float64"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorType((2, 3), "int8"))


def test_ones_like_zeros_like_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.ones_like(x0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.zeros_like(x1))


def test_eye_infer_ty():
    bb = relax.BlockBuilder()

    _check_inference(bb, relax.op.eye(3), relax.TensorType((3, 3), "float32"))
    _check_inference(bb, relax.op.eye(2, 4), relax.TensorType((2, 4), "float32"))
    _check_inference(bb, relax.op.eye(3, dtype="int64"), relax.TensorType((3, 3), "int64"))
    _check_inference(bb, relax.op.eye(3, 5, k=1), relax.TensorType((3, 5), "float32"))
    _check_inference(bb, relax.op.eye(3, 5, k=-2), relax.TensorType((3, 5), "float32"))


def test_eye_infer_ty_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    m = tirx.Var("m", "int64")
    k = tirx.Var("k", "int64")

    _check_inference(bb, relax.op.eye(n), relax.TensorType((n, n), "float32"))
    _check_inference(bb, relax.op.eye(n, m), relax.TensorType((n, m), "float32"))
    _check_inference(bb, relax.op.eye(n, k=k), relax.TensorType((n, n), "float32"))


def test_eye_like_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 5), "int64"))
    x2 = relax.Var("x", R.Tensor((3, 3)))

    _check_inference(bb, relax.op.eye_like(x0), relax.TensorType((3, 4), "float32"))
    _check_inference(bb, relax.op.eye_like(x1), relax.TensorType((2, 5), "int64"))
    _check_inference(bb, relax.op.eye_like(x2), relax.TensorType((3, 3), dtype=None))
    _check_inference(bb, relax.op.eye_like(x0, k=1), relax.TensorType((3, 4), "float32"))
    _check_inference(
        bb, relax.op.eye_like(x1, dtype="float32"), relax.TensorType((2, 5), "float32")
    )


def test_eye_like_infer_ty_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    m = tirx.Var("m", "int64")
    x = relax.Var("x", R.Tensor((n, m), "float32"))
    k = tirx.Var("k", "int64")

    _check_inference(bb, relax.op.eye_like(x), relax.TensorType((n, m), "float32"))
    _check_inference(bb, relax.op.eye_like(x, k=k), relax.TensorType((n, m), "float32"))


def test_eye_like_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.eye_like(x0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.eye_like(x1))


def test_arange_infer_ty():
    bb = relax.BlockBuilder()

    _check_inference(bb, relax.op.arange(10), relax.TensorType((10,), "int64"))
    _check_inference(bb, relax.op.arange(1, 10), relax.TensorType((9,), "int64"))
    _check_inference(bb, relax.op.arange(0, 10, 2), relax.TensorType((5,), "int64"))
    _check_inference(bb, relax.op.arange(1, 10, 2), relax.TensorType((5,), "int64"))

    _check_inference(bb, relax.op.arange(10.0), relax.TensorType((10,), "float32"))
    _check_inference(bb, relax.op.arange(1.0, 10), relax.TensorType((9,), "float32"))
    _check_inference(bb, relax.op.arange(0, 20, 2.5), relax.TensorType((8,), "float32"))
    _check_inference(bb, relax.op.arange(1, 10, 2.3), relax.TensorType((4,), "float32"))
    _check_inference(
        bb,
        relax.op.arange(0, 10, 1, tvm_tirx.ir.PrimType("float32")),
        relax.TensorType((10,), "float32"),
    )


def test_arange_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    start = tirx.Var("start", "int64")
    stop = tirx.Var("stop", "int64")
    step = tirx.Var("step", "int64")

    _check_inference(bb, relax.op.arange(stop), relax.TensorType((stop,), "int64"))
    _check_inference(bb, relax.op.arange(1, stop), relax.TensorType((stop - 1,), "int64"))
    _check_inference(bb, relax.op.arange(start, stop), relax.TensorType((stop - start,), "int64"))
    _check_inference(
        bb,
        relax.op.arange(start, stop, 2),
        relax.TensorType(((stop + 1 - start) // 2,), "int64"),
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop, step),
        relax.TensorType(((stop + step - start - 1) // step,), "int64"),
    )

    start = tirx.Var("start", "float32")
    stop = tirx.Var("stop", "float32")
    step = tirx.Var("step", "float32")

    _check_inference(
        bb,
        relax.op.arange(stop),
        relax.TensorType((T.cast(T.ceil(stop), ty="int64"),), "float32"),
    )
    _check_inference(
        bb,
        relax.op.arange(1, stop),
        relax.TensorType((T.cast(T.ceil(stop - 1.0), ty="int64"),), "float32"),
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop),
        relax.TensorType((T.cast(T.ceil(stop - start), ty="int64"),), "float32"),
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop, 2),
        relax.TensorType((T.cast(T.ceil((stop - start) / 2), ty="int64"),), "float32"),
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop, step),
        relax.TensorType((T.cast(T.ceil((stop - start) / step), ty="int64"),), "float32"),
    )


def test_tril_triu_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))

    _check_inference(bb, relax.op.tril(x0, k=1), relax.TensorType((2, 3, 4), "float32"))
    _check_inference(bb, relax.op.triu(x0, k=0), relax.TensorType((2, 3, 4), "float32"))
    _check_inference(bb, relax.op.tril(x0), relax.TensorType((2, 3, 4), "float32"))
    _check_inference(bb, relax.op.triu(x1), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.tril(x2), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.triu(x3), relax.TensorType((2, 3, 4), dtype=None))
    _check_inference(bb, relax.op.tril(x4), relax.TensorType(dtype=None, ndim=3))
    _check_inference(bb, relax.op.triu(x5), relax.TensorType(dtype=None))
    _check_inference(bb, relax.op.tril(x6), relax.TensorType((2, 3, 4), "float32", vdev0))


def test_tril_triu_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((a, b, c), "float32"))
    x1 = relax.Var("x", R.Tensor((a, b, c)))
    x2 = relax.Var("x", R.Tensor((a, b, c), "float32", vdev0))
    x3 = relax.Var("x", R.Tensor((16, 32, 64)))

    # Dynamic tensor, static offset
    _check_inference(bb, relax.op.tril(x0), relax.TensorType((a, b, c), "float32"))
    _check_inference(bb, relax.op.triu(x1), relax.TensorType((a, b, c), dtype=None))
    _check_inference(bb, relax.op.tril(x2), relax.TensorType((a, b, c), "float32", vdev0))

    # Static tensor, dynamic offset
    _check_inference(bb, relax.op.tril(x3, a), relax.TensorType((16, 32, 64), dtype=None))

    # Dynamic tensor, dynamic offset
    _check_inference(bb, relax.op.tril(x0, a), relax.TensorType((a, b, c), "float32"))


def test_tril_triu_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeType(ndim=3))
    s2 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s2, "float32"))

    _check_inference(bb, relax.op.tril(x0), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.triu(x1), relax.TensorType(s1, "float32"))
    _check_inference(bb, relax.op.tril(x2), relax.TensorType(s2, "float32"))


def test_tril_triu_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 4), "int32"))

    _check_inference(bb, relax.op.triu(x0), relax.TensorType((2, 3, 4), "float16"))
    _check_inference(bb, relax.op.tril(x1), relax.TensorType((2, 3, 4), "int8"))
    _check_inference(bb, relax.op.triu(x2), relax.TensorType((2, 3, 4), "int32"))


def test_tril_triu_infer_ty_less_than_two_ndim():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType((2,)))
    s1 = relax.Var("s", relax.ShapeType(()))
    s2 = relax.Var("s", relax.ShapeType(ndim=1))
    s3 = relax.Var("s", relax.ShapeType(ndim=0))
    x0 = relax.Var("x", R.Tensor((2,), "float32"))
    x1 = relax.Var("x", R.Tensor((), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=1))
    x3 = relax.Var("x", R.Tensor("float32", ndim=0))
    x4 = relax.Var("x", relax.TensorType(s0, "float32"))
    x5 = relax.Var("x", relax.TensorType(s1, "float32"))
    x6 = relax.Var("x", relax.TensorType(s2, "float32"))
    x7 = relax.Var("x", relax.TensorType(s3, "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.tril(x0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.triu(x1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.tril(x2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.triu(x3))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.tril(x4))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.triu(x5))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.tril(x6))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.triu(x7))


def test_tril_triu_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.tril(x0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.triu(x1))


if __name__ == "__main__":
    tvm_tirx.testing.main()
