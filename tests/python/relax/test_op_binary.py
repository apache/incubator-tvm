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
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    assert relax.op.add(x, y).op == Op.get("relax.add")
    assert relax.op.divide(x, y).op == Op.get("relax.divide")
    assert relax.op.floor_divide(x, y).op == Op.get("relax.floor_divide")
    assert relax.op.multiply(x, y).op == Op.get("relax.multiply")
    assert relax.op.power(x, y).op == Op.get("relax.power")
    assert relax.op.atan2(x, y).op == Op.get("relax.atan2")
    assert relax.op.subtract(x, y).op == Op.get("relax.subtract")
    assert relax.op.mod(x, y).op == Op.get("relax.mod")
    assert relax.op.floor_mod(x, y).op == Op.get("relax.floor_mod")

    assert relax.op.equal(x, y).op == Op.get("relax.equal")
    assert relax.op.greater(x, y).op == Op.get("relax.greater")
    assert relax.op.greater_equal(x, y).op == Op.get("relax.greater_equal")
    assert relax.op.less(x, y).op == Op.get("relax.less")
    assert relax.op.less_equal(x, y).op == Op.get("relax.less_equal")
    assert relax.op.not_equal(x, y).op == Op.get("relax.not_equal")

    x = relax.Var("x", R.Tensor((2, 3), "int32"))
    y = relax.Var("y", R.Tensor((2, 3), "int32"))
    assert relax.op.bitwise_and(x, y).op == Op.get("relax.bitwise_and")
    assert relax.op.bitwise_or(x, y).op == Op.get("relax.bitwise_or")
    assert relax.op.bitwise_xor(x, y).op == Op.get("relax.bitwise_xor")
    assert relax.op.left_shift(x, y).op == Op.get("relax.left_shift")
    assert relax.op.right_shift(x, y).op == Op.get("relax.right_shift")

    x = relax.Var("x", R.Tensor((2, 3), "bool"))
    y = relax.Var("y", R.Tensor((2, 3), "bool"))
    assert relax.op.logical_and(x, y).op == Op.get("relax.logical_and")
    assert relax.op.logical_or(x, y).op == Op.get("relax.logical_or")
    assert relax.op.logical_xor(x, y).op == Op.get("relax.logical_xor")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


binary_arith_ops = [
    (relax.op.add, tirx.Add),
    (relax.op.divide, tirx.Div),
    (relax.op.floor_divide, tirx.FloorDiv),
    (relax.op.multiply, tirx.Mul),
    (relax.op.power, tirx.pow),
    (relax.op.atan2, tirx.atan2),
    (relax.op.subtract, tirx.Sub),
    (relax.op.maximum, tirx.Max),
    (relax.op.minimum, tirx.Min),
    (relax.op.mod, tirx.Mod),
    (relax.op.floor_mod, tirx.FloorMod),
]


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_arith_infer_ty(binary_arith_op: Callable):
    bb = relax.BlockBuilder()
    vdevice0 = VDevice("llvm")
    vdevice1 = VDevice("cuda", 0)
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((1, 3), "float32"))
    x2 = relax.Var("x", R.Tensor((3, 2, 3), "float32"))
    x3 = relax.Var("x", R.Tensor((3, 1, 3), "float32"))
    x4 = relax.Var("x", R.Tensor("float32", ndim=2))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor("float32", ndim=2, vdevice=vdevice0))
    x7 = relax.Var("x", R.Tensor((2, 3), "float32", vdevice0))
    y0 = relax.Var("y", R.Tensor((2, 3), "float32"))
    y1 = relax.Var("y", R.Tensor((4, 3, 2, 1), "float32"))
    y2 = relax.Var("y", R.Tensor("float32", ndim=2))
    y3 = relax.Var("y", R.Tensor("float32", ndim=-1))
    y4 = relax.Var("y", R.Tensor((2, 3), "float32", vdevice0))
    y5 = relax.Var("y", R.Tensor("float32", ndim=2, vdevice=vdevice0))

    _check_inference(bb, binary_arith_op(x0, y0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, binary_arith_op(x1, y0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, binary_arith_op(x1, y1), relax.TensorType((4, 3, 2, 3), "float32"))
    _check_inference(bb, binary_arith_op(x2, y2), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, binary_arith_op(x3, y2), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, binary_arith_op(x4, y0), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, binary_arith_op(x4, y1), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, binary_arith_op(x4, y2), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, binary_arith_op(x4, y3), relax.TensorType(dtype="float32", ndim=-1))
    _check_inference(bb, binary_arith_op(x5, y0), relax.TensorType(dtype="", ndim=-1))
    _check_inference(
        bb,
        binary_arith_op(x6, y5),
        relax.TensorType(dtype="float32", ndim=2, vdevice=vdevice0),
    )
    _check_inference(
        bb,
        binary_arith_op(x6, y2),
        relax.TensorType(dtype="float32", ndim=2, vdevice=vdevice0),
    )
    _check_inference(bb, binary_arith_op(x7, y4), relax.TensorType((2, 3), "float32", vdevice0))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_infer_ty_binary_arith_prim_value_with_tensor(binary_arith_op: Callable):
    bb = relax.BlockBuilder()

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", tvm_tirx.ir.PrimType("float32"))

    _check_inference(bb, binary_arith_op(x, y), relax.TensorType((2, 3), "float32"))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_infer_ty_binary_arith_prim_value_with_prim_value(binary_arith_op: Callable):
    bb = relax.BlockBuilder()

    x = relax.Var("x", tvm_tirx.ir.PrimType("float32"))
    y = relax.Var("y", tvm_tirx.ir.PrimType("float32"))

    _check_inference(bb, binary_arith_op(x, y), tvm_tirx.ir.PrimType("float32"))


binary_cmp_ops = [
    (relax.op.equal, tirx.EQ),
    (relax.op.greater, tirx.GT),
    (relax.op.greater_equal, tirx.GE),
    (relax.op.less, tirx.LT),
    (relax.op.less_equal, tirx.LE),
    (relax.op.not_equal, tirx.NE),
]


@pytest.mark.parametrize("binary_cmp_op", [row[0] for row in binary_cmp_ops])
def test_binary_cmp_infer_ty(binary_cmp_op: Callable):
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", R.Tensor((2, 3), "float32"))
    y1 = relax.Var("y", R.Tensor((2, 3), "int32"))
    y2 = relax.Var("y", R.Tensor((2, 3), "float32", vdev0))
    _check_inference(bb, binary_cmp_op(x, y0), relax.TensorType((2, 3), "bool"))
    _check_inference(bb, binary_cmp_op(x, y1), relax.TensorType((2, 3), "bool"))
    _check_inference(bb, binary_cmp_op(x, y0), relax.TensorType((2, 3), "bool"))
    _check_inference(bb, binary_cmp_op(x, y1), relax.TensorType((2, 3), "bool"))
    _check_inference(bb, binary_cmp_op(x, y0), relax.TensorType((2, 3), "bool"))
    _check_inference(bb, binary_cmp_op(x, y1), relax.TensorType((2, 3), "bool"))
    _check_inference(bb, binary_cmp_op(x, y2), relax.TensorType((2, 3), "bool", vdev0))


@pytest.mark.parametrize("binary_cmp_op", [row[0] for row in binary_cmp_ops])
def test_infer_ty_binary_cmp_prim_value_to_tensor(binary_cmp_op: Callable):
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", tvm_tirx.ir.PrimType("float32"))
    _check_inference(bb, binary_cmp_op(x, y), relax.TensorType((2, 3), "bool"))
    _check_inference(bb, binary_cmp_op(y, x), relax.TensorType((2, 3), "bool"))


@pytest.mark.parametrize("binary_cmp_op", [row[0] for row in binary_cmp_ops])
def test_infer_ty_binary_cmp_prim_value_to_prim_value(binary_cmp_op: Callable):
    bb = relax.BlockBuilder()
    x = relax.Var("x", tvm_tirx.ir.PrimType("float32"))
    y = relax.Var("y", tvm_tirx.ir.PrimType("float32"))
    _check_inference(bb, binary_cmp_op(x, y), tvm_tirx.ir.PrimType("bool"))
    _check_inference(bb, binary_cmp_op(y, x), tvm_tirx.ir.PrimType("bool"))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_infer_ty_shape_symbolic(binary_arith_op: Callable):
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    k = tirx.Var("k", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((1, n), "float32"))
    x2 = relax.Var("x", R.Tensor((k, n, m), "float32"))
    x3 = relax.Var("x", R.Tensor((3, 1, n), "float32"))
    x4 = relax.Var("x", R.Tensor("float32", ndim=2))
    y0 = relax.Var("y", R.Tensor((m, n), "float32"))
    y1 = relax.Var("y", R.Tensor((m, n + 2), "float32"))
    y2 = relax.Var("y", R.Tensor((4, k, m, 1), "float32"))
    y3 = relax.Var("y", R.Tensor("float32", ndim=2))
    y4 = relax.Var("y", R.Tensor("float32", ndim=-1))
    _check_inference(bb, binary_arith_op(x0, y0), relax.TensorType((m, n), "float32"))
    _check_inference(bb, binary_arith_op(x0, y1), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, binary_arith_op(x1, y0), relax.TensorType((m, n), "float32"))
    _check_inference(bb, binary_arith_op(x1, y2), relax.TensorType((4, k, m, n), "float32"))
    _check_inference(bb, binary_arith_op(x2, y2), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, binary_arith_op(x2, y3), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, binary_arith_op(x3, y3), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, binary_arith_op(x4, y0), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, binary_arith_op(x4, y2), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, binary_arith_op(x4, y3), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, binary_arith_op(x4, y4), relax.TensorType(dtype="float32", ndim=-1))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_infer_ty_shape_var(binary_arith_op: Callable):
    bb = relax.BlockBuilder()
    s0 = relax.Var("s0", relax.ShapeType(ndim=2))
    s1 = relax.Var("s1", relax.ShapeType(ndim=2))
    s2 = relax.Var("s2", relax.ShapeType(ndim=4))
    s3 = relax.Var("s3", relax.ShapeType(ndim=1))
    s4 = relax.Var("s4", relax.ShapeType())
    x = relax.Var("x", relax.TensorType(s0, "float32"))
    y0 = relax.Var("y", relax.TensorType(s0, "float32"))
    y1 = relax.Var("y", relax.TensorType(s1, "float32"))
    y2 = relax.Var("y", relax.TensorType(s2, "float32"))
    y3 = relax.Var("y", relax.TensorType(s3, "float32"))
    y4 = relax.Var("y", relax.TensorType(s4, "float32"))

    _check_inference(bb, binary_arith_op(x, y0), relax.TensorType(s0, "float32"))
    _check_inference(bb, binary_arith_op(x, y1), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, binary_arith_op(x, y2), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, binary_arith_op(x, y3), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, binary_arith_op(x, y4), relax.TensorType(dtype="float32"))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_arith_infer_ty_more_input_dtype(binary_arith_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    y0 = relax.Var("y", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    y1 = relax.Var("y", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))
    y2 = relax.Var("y", R.Tensor((2, 3), "int64"))

    _check_inference(bb, binary_arith_op(x0, y0), relax.TensorType((2, 3), "float64"))
    _check_inference(bb, binary_arith_op(x1, y1), relax.TensorType((2, 3), "int8"))
    _check_inference(bb, binary_arith_op(x2, y2), relax.TensorType((2, 3), "int64"))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_infer_ty_shape_unequal_const_int(binary_arith_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", R.Tensor((2, 4), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(binary_arith_op(x0, y0))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_arith_infer_ty_dtype_mismatch(binary_arith_op: Callable):
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 3), "int32"))
    with pytest.raises(TypeError):
        bb.normalize(binary_arith_op(x, y))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_arith_infer_ty_vdevice_mismatch(binary_arith_op: Callable):
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32", VDevice("llvm")))
    y = relax.Var("y", R.Tensor((2, 3), "int32", VDevice("cuda")))
    with pytest.raises(TypeError):
        bb.normalize(binary_arith_op(x, y))


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_wrong_input_number(binary_arith_op: Callable):
    x = relax.Var("x", R.Tensor((2, 3), "float32"))

    with pytest.raises(TypeError):
        binary_arith_op(x, x, x)
    with pytest.raises(TypeError):
        binary_arith_op(x)
    with pytest.raises(TypeError):
        binary_arith_op(x, x, x, x)


@pytest.mark.parametrize("binary_arith_op", [row[0] for row in binary_arith_ops])
def test_binary_infer_ty_wrong_input_type(binary_arith_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3), "float32")))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))

    with pytest.raises(TypeError):
        bb.normalize(binary_arith_op(x0, y))
    with pytest.raises(TypeError):
        bb.normalize(binary_arith_op(x1, y))


if __name__ == "__main__":
    tvm_tirx.testing.main()
