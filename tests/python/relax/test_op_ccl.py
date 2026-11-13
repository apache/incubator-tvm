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
# ruff: noqa: F401
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx import tirx
from tvm_tirx.ir import Op


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    assert relax.op.ccl.allreduce(x).op == Op.get("relax.ccl.allreduce")
    assert relax.op.ccl.broadcast_from_worker0(x).op == Op.get("relax.ccl.broadcast_from_worker0")
    assert relax.op.ccl.allgather(x, 2).op == Op.get("relax.ccl.allgather")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_allreduce_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((3, 4)))

    _check_inference(bb, relax.op.ccl.allreduce(x0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.ccl.allreduce(x1), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.ccl.allreduce(x2), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.ccl.allreduce(x3), relax.TensorType((2, 3), dtype=None))
    _check_inference(bb, relax.op.ccl.allreduce(x4), relax.TensorType(dtype=None))
    _check_inference(bb, relax.op.ccl.allreduce(x5), relax.TensorType((3, 4), dtype=None))


def test_allreduce_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(bb, relax.op.ccl.allreduce(x0), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.ccl.allreduce(x1), relax.TensorType((4, n), "float32"))


def test_allreduce_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=2))
    s1 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))

    _check_inference(bb, relax.op.ccl.allreduce(x0), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.ccl.allreduce(x1), relax.TensorType(s1, "float32"))


def test_allreduce_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(bb, relax.op.ccl.allreduce(x0), relax.TensorType((2, 3), "float64"))
    _check_inference(bb, relax.op.ccl.allreduce(x1), relax.TensorType((2, 3), "int8"))
    _check_inference(bb, relax.op.ccl.allreduce(x2), relax.TensorType((2, 3), "int64"))


def test_allgather_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((3, 4)))

    _check_inference(bb, relax.op.ccl.allgather(x0, 2), relax.TensorType((4, 3), "float32"))
    _check_inference(bb, relax.op.ccl.allgather(x1, 2), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.ccl.allgather(x2, 2), relax.TensorType(dtype="float32"))
    _check_inference(bb, relax.op.ccl.allgather(x3, 2), relax.TensorType((4, 3), dtype=None))
    _check_inference(bb, relax.op.ccl.allgather(x4, 2), relax.TensorType(dtype=None))
    _check_inference(bb, relax.op.ccl.allgather(x5, 2), relax.TensorType((6, 4), dtype=None))


def test_allgather_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(bb, relax.op.ccl.allgather(x0, 2), relax.TensorType((m * 2, n), "float32"))
    _check_inference(bb, relax.op.ccl.allgather(x1, 2), relax.TensorType((8, n), "float32"))


def test_allgather_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=2))
    s1 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))

    _check_inference(bb, relax.op.ccl.allgather(x0, 2), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.ccl.allgather(x1, 2), relax.TensorType(s1, "float32"))


def test_allgather_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(bb, relax.op.ccl.allgather(x0, 2), relax.TensorType((4, 3), "float64"))
    _check_inference(bb, relax.op.ccl.allgather(x1, 2), relax.TensorType((4, 3), "int8"))
    _check_inference(bb, relax.op.ccl.allgather(x2, 2), relax.TensorType((4, 3), "int64"))


def test_broadcast_from_worker0_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((3, 4)))

    _check_inference(
        bb, relax.op.ccl.broadcast_from_worker0(x0), relax.TensorType((2, 3), "float32")
    )
    _check_inference(
        bb, relax.op.ccl.broadcast_from_worker0(x1), relax.TensorType(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.ccl.broadcast_from_worker0(x2), relax.TensorType(dtype="float32"))
    _check_inference(
        bb, relax.op.ccl.broadcast_from_worker0(x3), relax.TensorType((2, 3), dtype=None)
    )
    _check_inference(bb, relax.op.ccl.broadcast_from_worker0(x4), relax.TensorType(dtype=None))
    _check_inference(
        bb, relax.op.ccl.broadcast_from_worker0(x5), relax.TensorType((3, 4), dtype=None)
    )


def test_broadcast_from_worker0_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(
        bb, relax.op.ccl.broadcast_from_worker0(x0), relax.TensorType((m, n), "float32")
    )
    _check_inference(
        bb, relax.op.ccl.broadcast_from_worker0(x1), relax.TensorType((4, n), "float32")
    )


def test_broadcast_from_worker0_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=2))
    s1 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))

    _check_inference(bb, relax.op.ccl.broadcast_from_worker0(x0), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.ccl.broadcast_from_worker0(x1), relax.TensorType(s1, "float32"))


def test_broadcast_from_worker0_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(
        bb, relax.op.ccl.broadcast_from_worker0(x0), relax.TensorType((2, 3), "float64")
    )
    _check_inference(bb, relax.op.ccl.broadcast_from_worker0(x1), relax.TensorType((2, 3), "int8"))
    _check_inference(bb, relax.op.ccl.broadcast_from_worker0(x2), relax.TensorType((2, 3), "int64"))


def test_scatter_from_worker0_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((3, 4, 5)))

    _check_inference(
        bb, relax.op.ccl.scatter_from_worker0(x0, 2), relax.TensorType((1, 3), "float32")
    )
    _check_inference(
        bb, relax.op.ccl.scatter_from_worker0(x1, 3), relax.TensorType((1, 4, 5), dtype=None)
    )


def test_scatter_from_worker0_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(
        bb,
        relax.op.ccl.scatter_from_worker0(x0, 2),
        relax.TensorType((tirx.div(m, 2), n), "float32"),
    )
    _check_inference(
        bb, relax.op.ccl.scatter_from_worker0(x1, 2), relax.TensorType((2, n), "float32")
    )


def test_scatter_from_worker0_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType((2, 4, 8)))
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))

    _check_inference(
        bb, relax.op.ccl.scatter_from_worker0(x0, 2), relax.TensorType((1, 4, 8), "float32")
    )


def test_scatter_from_worker0_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(
        bb, relax.op.ccl.scatter_from_worker0(x0, 2), relax.TensorType((1, 3), "float64")
    )
    _check_inference(bb, relax.op.ccl.scatter_from_worker0(x1, 2), relax.TensorType((1, 3), "int8"))
    _check_inference(
        bb, relax.op.ccl.scatter_from_worker0(x2, 2), relax.TensorType((1, 3), "int64")
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
