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
import numpy as np  # type: ignore
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax import script as R
from tvm_tirx import tirx
from tvm_tirx.ir import Op


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    c = relax.const(tvm_tirx.runtime.tensor(np.array([1, 2, 3], dtype="float16")))
    assert relax.op.astype(x, "float16").op == Op.get("relax.astype")
    assert relax.op.wrap_param(c, "float32").op == Op.get("relax.wrap_param")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_astype_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.astype(x0, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.astype(x1, "float16"), relax.TensorType(dtype="float16", ndim=2))
    _check_inference(bb, relax.op.astype(x2, "float16"), relax.TensorType(dtype="float16"))
    _check_inference(bb, relax.op.astype(x3, "float16"), relax.TensorType((2, 3), "float16"))
    _check_inference(bb, relax.op.astype(x4, "float16"), relax.TensorType(dtype="float16", ndim=2))
    _check_inference(bb, relax.op.astype(x5, "float16"), relax.TensorType(dtype="float16"))


def test_astype_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tirx.Var("m", "int64")
    n = tirx.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))

    _check_inference(bb, relax.op.astype(x0, "float16"), relax.TensorType((m, n), "float16"))
    _check_inference(bb, relax.op.astype(x1, "float16"), relax.TensorType((m, n), "float16"))


def test_astype_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType((2, 3)))
    s1 = relax.Var("s", relax.ShapeType(ndim=2))
    s2 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s2, "float32"))

    _check_inference(bb, relax.op.astype(x0, "float16"), relax.TensorType(s0, "float16"))
    _check_inference(bb, relax.op.astype(x1, "float16"), relax.TensorType(s1, "float16"))
    _check_inference(bb, relax.op.astype(x2, "float16"), relax.TensorType(s2, "float16"))


def test_astype_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int32"))

    _check_inference(bb, relax.op.astype(x0, "float32"), relax.TensorType((2, 3), "float32"))
    _check_inference(bb, relax.op.astype(x1, "int32"), relax.TensorType((2, 3), "int32"))
    _check_inference(bb, relax.op.astype(x2, "int8"), relax.TensorType((2, 3), "int8"))


def test_astype_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.astype(x0, "float16"))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.astype(x1, "float16"))


def test_wrap_param_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.const(tvm_tirx.runtime.tensor(np.zeros([1, 2, 3], dtype="float16")))
    x1 = relax.const(tvm_tirx.runtime.tensor(np.zeros([1, 2, 3], dtype="int8")))
    _check_inference(bb, relax.op.wrap_param(x0, "float32"), relax.TensorType((1, 2, 3), "float32"))
    _check_inference(bb, relax.op.wrap_param(x1, "int32"), relax.TensorType((1, 2, 3), "int32"))


if __name__ == "__main__":
    tvm_tirx.testing.main()
