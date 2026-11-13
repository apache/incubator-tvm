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
# ruff: noqa: F841
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import VDevice
from tvm.relax import script as R
from tvm_tirx import tirx
from tvm_tirx.ir import Op


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((3, 4), "float32"))
    assert relax.op.matmul(x, y).op == Op.get("relax.matmul")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_matmul_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((4,), "float32"))
    x2 = relax.Var("x", R.Tensor((2, 3, 5, 4), "float32"))
    x3 = relax.Var("x", R.Tensor((2, 1, 4, 5), "float32"))
    x4 = relax.Var("x", R.Tensor((2, 1, 4, 5)))
    x5 = relax.Var("x", R.Tensor("float32"))
    x6 = relax.Var("x", R.Tensor((2, 1, 4, 5), "float16"))
    x7 = relax.Var("x", R.Tensor((3, 4), "float32", vdev0))
    y0 = relax.Var("y", R.Tensor((4, 5), "float32"))
    y1 = relax.Var("y", R.Tensor((4,), "float32"))
    y2 = relax.Var("y", R.Tensor((2, 3, 4, 5), "float32"))
    y3 = relax.Var("y", R.Tensor((6, 1, 3, 5, 7), "float32"))
    y4 = relax.Var("y", R.Tensor("float32", ndim=5))
    y5 = relax.Var("y", R.Tensor())
    y6 = relax.Var("y", R.Tensor((4, 5), "float32", vdev0))

    _check_inference(bb, relax.op.matmul(x0, y0), relax.TensorType((3, 5), "float32"))
    _check_inference(bb, relax.op.matmul(x7, y6), relax.TensorType((3, 5), "float32", vdev0))
    _check_inference(bb, relax.op.matmul(x1, y1), relax.TensorType((), "float32"))
    _check_inference(bb, relax.op.matmul(x1, y2), relax.TensorType((2, 3, 5), "float32"))
    _check_inference(bb, relax.op.matmul(x2, y1), relax.TensorType((2, 3, 5), "float32"))
    _check_inference(bb, relax.op.matmul(x3, y3), relax.TensorType((6, 2, 3, 4, 7), "float32"))
    _check_inference(bb, relax.op.matmul(x4, y3), relax.TensorType((6, 2, 3, 4, 7), ""))
    _check_inference(bb, relax.op.matmul(x3, y4), relax.TensorType(dtype="float32", ndim=5))
    _check_inference(bb, relax.op.matmul(x5, y3), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.matmul(x3, y5), relax.TensorType(dtype=""))
    _check_inference(
        bb,
        relax.op.matmul(x3, y3, out_dtype="float16"),
        relax.TensorType((6, 2, 3, 4, 7), "float16"),
    )
    _check_inference(
        bb,
        relax.op.matmul(x6, y3, out_dtype="float16"),
        relax.TensorType((6, 2, 3, 4, 7), "float16"),
    )


def test_matmul_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    k0 = tirx.Var("k0", "int64")
    k1 = tirx.Var("k1", "int64")
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    b1 = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((m, k0), "float32"))
    x1 = relax.Var("x", R.Tensor((k0,), "float32"))
    x2 = relax.Var("x", R.Tensor((a, b, m, k0), "float32"))
    x3 = relax.Var("x", R.Tensor((b, 1, m, k0), "float32"))
    x4 = relax.Var("x", R.Tensor((b, 1, m, k1), "float32"))
    y0 = relax.Var("y", R.Tensor((k0, n), "float32"))
    y1 = relax.Var("y", R.Tensor((k0,), "float32"))
    y2 = relax.Var("y", R.Tensor((a, b, k0, n), "float32"))
    y3 = relax.Var("y", R.Tensor((a, 1, c, k0, n), "float32"))
    y4 = relax.Var("y", R.Tensor((a, b1, c, k0, n), "float32"))

    _check_inference(bb, relax.op.matmul(x0, y0), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.matmul(x1, y1), relax.TensorType((), "float32"))
    _check_inference(bb, relax.op.matmul(x1, y2), relax.TensorType((a, b, n), "float32"))
    _check_inference(bb, relax.op.matmul(x2, y1), relax.TensorType((a, b, m), "float32"))
    _check_inference(bb, relax.op.matmul(x3, y3), relax.TensorType((a, b, c, m, n), "float32"))
    _check_inference(bb, relax.op.matmul(x4, y3), relax.TensorType((a, b, c, m, n), "float32"))
    _check_inference(bb, relax.op.matmul(x3, y4), relax.TensorType(dtype="float32", ndim=5))


def test_matmul_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s0", relax.ShapeType(ndim=4))
    s1 = relax.Var("s1", relax.ShapeType(ndim=3))
    s2 = relax.Var("s3", relax.ShapeType(ndim=1))
    s3 = relax.Var("s4", relax.ShapeType(ndim=1))
    s5 = relax.Var("s5", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s2, "float32"))
    x2 = relax.Var("x", relax.TensorType(s5, "float32"))
    y0 = relax.Var("y", relax.TensorType(s1, "float32"))
    y1 = relax.Var("y", relax.TensorType(s2, "float32"))
    y2 = relax.Var("y", relax.TensorType(s3, "float32"))

    _check_inference(bb, relax.op.matmul(x0, y0), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, relax.op.matmul(x1, y0), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.matmul(x2, y0), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.matmul(x0, y1), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.matmul(x1, y1), relax.TensorType(dtype="float32", ndim=0))
    _check_inference(bb, relax.op.matmul(x2, y1), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.matmul(x1, y2), relax.TensorType(dtype="float32", ndim=0))


def test_matmul_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4), "float16"))
    y0 = relax.Var("y", R.Tensor((4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((3, 4), "int8"))
    y1 = relax.Var("y", R.Tensor((4, 5), "int8"))
    x2 = relax.Var("x", R.Tensor((3, 4), "int64"))
    y2 = relax.Var("y", R.Tensor((4, 5), "int64"))

    _check_inference(bb, relax.op.matmul(x0, y0), relax.TensorType((3, 5), "float16"))
    _check_inference(bb, relax.op.matmul(x1, y1), relax.TensorType((3, 5), "int8"))
    _check_inference(bb, relax.op.matmul(x2, y2), relax.TensorType((3, 5), "int64"))


def test_matmul_infer_ty_mixed_precision():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4), "float16"))
    y0 = relax.Var("y", R.Tensor((4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((3, 4), "int8"))
    y1 = relax.Var("y", R.Tensor((4, 5), "int8"))
    x2 = relax.Var("x", R.Tensor((3, 4)))
    y2 = relax.Var("y", R.Tensor((4, 5)))

    _check_inference(
        bb,
        relax.op.matmul(x0, y0, out_dtype="float32"),
        relax.TensorType((3, 5), "float32"),
    )
    _check_inference(
        bb, relax.op.matmul(x1, y1, out_dtype="int32"), relax.TensorType((3, 5), "int32")
    )
    _check_inference(
        bb,
        relax.op.matmul(x2, y2, out_dtype="float32"),
        relax.TensorType((3, 5), "float32"),
    )


def test_matmul_infer_ty_zero_rank_input():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((), "float32"))
    y0 = relax.Var("y", R.Tensor((4, 5), "float32"))
    y1 = relax.Var("y", R.Tensor((), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.matmul(x0, y1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.matmul(x1, y0))


def test_matmul_infer_ty_not_broadcastable():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    y = relax.Var("y", R.Tensor((2, 8, 3, 5, 6), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.matmul(x, y))


def test_matmul_infer_ty_unequal_reduction_length():
    bb = relax.BlockBuilder()
    k = tirx.Var("k", "int64")
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((3, k), "float32"))
    y0 = relax.Var("y", R.Tensor((6, 5), "float32"))
    y1 = relax.Var("y", R.Tensor((k + 1, 5), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.matmul(x0, y0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.matmul(x1, y1))


def test_linear():
    # Since linear is only a sugar for transpose + matmul + add,
    # we only have brief tests here.
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))
    w1 = relax.Var("w", R.Tensor((5, 4), "float32"))
    w2 = relax.Var("w", R.Tensor((4,), "float32"))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((5, 4), "float32", vdev0))
    b1 = relax.Var("b", R.Tensor((5,), "float32"))
    b2 = relax.Var("b", R.Tensor((), "float32"))
    b3 = relax.Var("b", R.Tensor((5,), "float32", vdev0))

    # Need a scope to normalize non-leaf nodes
    with bb.function("func", [x1]):
        _check_inference(bb, relax.op.linear(x1, w1, b1), relax.TensorType((2, 3, 5), "float32"))
        _check_inference(
            bb, relax.op.linear(x3, w4, b3), relax.TensorType((2, 3, 5), "float32", vdev0)
        )
        _check_inference(bb, relax.op.linear(x1, w1, b2), relax.TensorType((2, 3, 5), "float32"))
        with pytest.raises(ValueError):
            bb.normalize(relax.op.linear(x1, w2, b1))  # error on Add with shape (2, 3, 5) and (4,)
        _check_inference(bb, relax.op.linear(x1, w2, b2), relax.TensorType((2, 3), "float32"))
        _check_inference(bb, relax.op.linear(x1, w3, b1), relax.TensorType(dtype="float32"))
        _check_inference(bb, relax.op.linear(x1, w3, b2), relax.TensorType(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w1, b1), relax.TensorType(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w1, b2), relax.TensorType(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w2, b1), relax.TensorType(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w2, b2), relax.TensorType(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w3, b1), relax.TensorType(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w3, b2), relax.TensorType(dtype="float32"))

        # Fake output
        gv = bb.emit_func_output(relax.Tuple([]))


def test_einsum_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x0", R.Tensor((), "float32"))
    x1 = relax.Var("x1", R.Tensor((5,), "int32"))
    x2 = relax.Var("x2", R.Tensor((5, 5), "int32"))
    x3 = relax.Var("x3", R.Tensor((1, 4), "float32"))
    x4 = relax.Var("x4", R.Tensor((2, 4), "float32"))
    x5 = relax.Var("x5", R.Tensor((2, 3), "float32"))
    x6 = relax.Var("x6", R.Tensor((3, 4), "float32"))
    x7 = relax.Var("x7", R.Tensor((4, 2), "float32"))
    x8 = relax.Var("x8", R.Tensor((4, 5), "float32"))
    x9 = relax.Var("x9", R.Tensor((3, 4, 5), "float32"))
    x10 = relax.Var("x10", R.Tensor((4, 3, 2), "float32"))
    x11 = relax.Var("x11", R.Tensor((3, 4, 4), "float32"))
    x12 = relax.Var("x12", R.Tensor((1, 1, 1, 4), "float16"))
    x13 = relax.Var("x13", R.Tensor((1, 1, 1, 3), "float16"))
    x14 = relax.Var("x14", R.Tensor((1, 5, 3, 8, 4), "float32"))
    x15 = relax.Var("x15", R.Tensor((2, 5, 3, 6, 4), "float32"))
    x16 = relax.Var("x16", R.Tensor((5, 5), "int32", vdev0))

    _check_inference(bb, relax.op.einsum((x2,), "ii"), relax.TensorType((), "int32"))
    _check_inference(bb, relax.op.einsum((x16,), "ii"), relax.TensorType((), "int32", vdev0))
    _check_inference(bb, relax.op.einsum((x2,), "ii->i"), relax.TensorType((5,), "int32"))
    _check_inference(bb, relax.op.einsum([x2], "...j->..."), relax.TensorType((5,), "int32"))
    _check_inference(bb, relax.op.einsum((x2, x1), "...j, j"), relax.TensorType((5,), "int32"))
    _check_inference(bb, relax.op.einsum((x0, x5), "..., ..."), relax.TensorType((2, 3), "float32"))
    _check_inference(
        bb, relax.op.einsum((x5, x6), "ij,jk->ik"), relax.TensorType((2, 4), "float32")
    )
    _check_inference(
        bb, relax.op.einsum((x5, x6, x8), "ij,jk,km->im"), relax.TensorType((2, 5), "float32")
    )
    _check_inference(
        bb, relax.op.einsum((x9, x10), "ijk, jil->kl"), relax.TensorType((5, 2), "float32")
    )
    _check_inference(
        bb, relax.op.einsum((x3, x4), "ij, ij -> i"), relax.TensorType((2,), "float32")
    )
    _check_inference(
        bb,
        relax.op.einsum((x3, x7), "...ij, ...jk -> ...ik"),
        relax.TensorType((1, 2), "float32"),
    )
    _check_inference(
        bb,
        relax.op.einsum((x12, x13), "...ij, ...ik -> ...jk"),
        relax.TensorType((1, 1, 4, 3), "float16"),
    )
    _check_inference(
        bb,
        relax.op.einsum((x11, x14, x15), "...ik, ...jk, ...hk -> i...jh"),
        relax.TensorType((4, 2, 5, 3, 8, 6), "float32"),
    )


def test_einsum_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b), "float32"))
    y = relax.Var("y", R.Tensor((b, c), "float32"))
    z = relax.Var("z", R.Tensor((a, a), "float32"))

    _check_inference(bb, relax.op.einsum((z,), "ii->i"), relax.TensorType((a,), "float32"))
    _check_inference(bb, relax.op.einsum((x, y), "ij,jk->ik"), relax.TensorType((a, c), "float32"))


def test_einsum_infer_ty_wrong_inputs():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x0", relax.ShapeType((2, 3, 4, 5)))
    x1 = relax.Var("x1", R.Tensor((5, 5), "int32"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.einsum(x0, subscripts="ii"))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.einsum(x1, subscripts="ijk"))


if __name__ == "__main__":
    tvm_tirx.testing.main()
