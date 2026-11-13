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
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    z = relax.Var("z", R.Tensor((2, 3), "float32"))
    assert relax.op.ewise_fma(x, y, z).op == Op.get("relax.ewise_fma")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_ewise_fma_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3)))
    x2 = relax.Var("x", R.Tensor((2, 3), "float32", vdev0))
    y0 = relax.Var("y", R.Tensor((2, 3), "float32"))
    y1 = relax.Var("y", R.Tensor(dtype="float32", ndim=2))
    y2 = relax.Var("y", R.Tensor((2, 3), "float32", vdev0))
    z0 = relax.Var("z", R.Tensor((2, 3), "float32"))
    z1 = relax.Var("z", R.Tensor("float32"))
    z2 = relax.Var("z", R.Tensor((2, 3), "float32", vdev0))

    _check_inference(bb, relax.op.ewise_fma(x0, y0, z0), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.ewise_fma(x2, y2, z2), relax.TensorType((2, 3), "float32", vdev0))
    _check_inference(bb, relax.op.ewise_fma(x0, y1, z0), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.ewise_fma(x0, y1, z1), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.ewise_fma(x1, y0, z0), relax.TensorType((2, 3), dtype=""))


def test_ewise_fma_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    y0 = relax.Var("y", R.Tensor((m, n), "float32"))
    y1 = relax.Var("y", R.Tensor(dtype="float32", ndim=2))
    z0 = relax.Var("z", R.Tensor((m, n), "float32"))

    _check_inference(bb, relax.op.ewise_fma(x0, y0, z0), relax.TensorType((m, n), "float32"))
    _check_inference(bb, relax.op.ewise_fma(x0, y1, z0), relax.TensorType(dtype="float32", ndim=2))


def test_ewise_fma_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=2))
    s1 = relax.Var("s", relax.ShapeType(ndim=2))
    s2 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s2, "float32"))
    y = relax.Var("y", relax.TensorType(s0, "float32"))
    z = relax.Var("z", relax.TensorType(s0, "float32"))

    _check_inference(bb, relax.op.ewise_fma(x0, y, z), relax.TensorType(s0, "float32"))
    _check_inference(bb, relax.op.ewise_fma(x1, y, z), relax.TensorType(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.ewise_fma(x2, y, z), relax.TensorType(dtype="float32", ndim=2))


def test_ewise_fma_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    y0 = relax.Var("y", R.Tensor((2, 3), "float64"))
    z0 = relax.Var("z", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    y1 = relax.Var("y", R.Tensor((2, 3), "int8"))
    z1 = relax.Var("z", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))
    y2 = relax.Var("y", R.Tensor((2, 3), "int64"))
    z2 = relax.Var("z", R.Tensor((2, 3), "int64"))

    _check_inference(bb, relax.op.ewise_fma(x0, y0, z0), relax.TensorType((2, 3), "float64"))
    _check_inference(bb, relax.op.ewise_fma(x1, y1, z1), relax.TensorType((2, 3), "int8"))
    _check_inference(bb, relax.op.ewise_fma(x2, y2, z2), relax.TensorType((2, 3), "int64"))


def test_ewise_fma_infer_ty_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", R.Tensor((2, 3), "int32"))
    y1 = relax.Var("y", R.Tensor((2, 3), "float32"))
    z0 = relax.Var("z", R.Tensor((2, 3), "float32"))
    z1 = relax.Var("z", R.Tensor((2, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.ewise_fma(x, y0, z0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.ewise_fma(x, y1, z1))


def test_ewise_fma_infer_ty_ndim_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", R.Tensor((2, 3), "float32"))
    y1 = relax.Var("y", R.Tensor((2, 3, 4), "float32"))
    z0 = relax.Var("z", R.Tensor((2, 3), "float32"))
    z1 = relax.Var("z", R.Tensor(dtype="float32", ndim=4))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.ewise_fma(x, y1, z0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.ewise_fma(x, y0, z1))


def test_ewise_fma_wrong_input_number():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))

    with pytest.raises(TypeError):
        relax.op.ewise_fma(x)
    with pytest.raises(TypeError):
        relax.op.ewise_fma(x, x)
    with pytest.raises(TypeError):
        relax.op.ewise_fma(x, x, x, x)


def test_ewise_fma_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", relax.ShapeType((2, 3)))
    y1 = relax.Var("y", relax.FuncType([], R.Tensor((2, 3), "float32")))
    z = relax.Var("z", R.Tensor((2, 3), "float32"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.ewise_fma(x, y0, z))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.ewise_fma(x, y1, z))


if __name__ == "__main__":
    tvm_tirx.testing.main()
