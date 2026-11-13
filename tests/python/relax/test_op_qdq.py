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
from tvm.relax import script as R
from tvm_tirx import tirx
from tvm_tirx.ir import Op


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    dx = relax.Var("dx", R.Tensor((2, 3), "uint8"))
    s = relax.Var("s", R.Tensor([3], "float32"))
    zp = relax.Var("zp", R.Tensor([3], "int8"))
    assert relax.op.quantize(x, s, zp, 1, "int8").op == Op.get("relax.quantize")
    assert relax.op.dequantize(dx, s, zp, 1, "float32").op == Op.get("relax.dequantize")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_qdq_op_infer_ty():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    dx = relax.Var("dx", R.Tensor((2, 3), "uint8"))
    s = relax.Var("s", R.Tensor([3], "float32"))
    zp = relax.Var("zp", R.Tensor([3], "int8"))
    _check_inference(bb, relax.op.quantize(x, s, zp, 1, "int8"), relax.TensorType((2, 3), "int8"))
    _check_inference(
        bb,
        relax.op.dequantize(dx, s, zp, 1, "float32"),
        relax.TensorType((2, 3), "float32"),
    )


def test_qdq_op_infer_ty_scalar():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((), "float32"))
    dx = relax.Var("dx", R.Tensor((), "uint8"))
    s = relax.Var("s", R.Tensor((), "float32"))
    zp = relax.Var("zp", R.Tensor((), "uint8"))
    _check_inference(bb, relax.op.quantize(x, s, zp, 0, "uint8"), relax.TensorType((), "uint8"))
    _check_inference(
        bb,
        relax.op.dequantize(dx, s, zp, 0, "float32"),
        relax.TensorType((), "float32"),
    )


@pytest.mark.parametrize("op_name", ["quantize", "dequantize"])
def test_qdq_op_rejects_per_axis_params_for_scalar_input(op_name):
    bb = relax.BlockBuilder()
    s = relax.Var("s", R.Tensor((2,), "float32"))
    zp = relax.Var("zp", R.Tensor((2,), "uint8"))
    if op_name == "quantize":
        x = relax.Var("x", R.Tensor((), "float32"))
        call = relax.op.quantize(x, s, zp, 0, "uint8")
    else:
        x = relax.Var("x", R.Tensor((), "uint8"))
        call = relax.op.dequantize(x, s, zp, 0, "float32")

    with pytest.raises(ValueError, match="scalar or singleton tensors"):
        bb.normalize(call)


def test_qdq_op_infer_ty_unknown_dtype():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), dtype=None))
    dx = relax.Var("dx", R.Tensor((2, 3), dtype=None))
    s = relax.Var("s", R.Tensor([3], "float32"))
    s_unknown = relax.Var("s_unknown", R.Tensor([3], dtype=None))
    zp = relax.Var("zp", R.Tensor([3], "int8"))
    zp_unknown = relax.Var("zp_unknown", R.Tensor([3], dtype=None))
    _check_inference(
        bb, relax.op.quantize(x, s, zp, 1, "int8"), relax.TensorType((2, 3), dtype=None)
    )
    _check_inference(
        bb,
        relax.op.quantize(dx, s_unknown, zp, 1, "int8"),
        relax.TensorType((2, 3), dtype=None),
    )
    _check_inference(
        bb,
        relax.op.quantize(dx, s, zp_unknown, 1, "int8"),
        relax.TensorType((2, 3), dtype=None),
    )
    _check_inference(
        bb,
        relax.op.dequantize(dx, s, zp, 1, "float32"),
        relax.TensorType((2, 3), dtype=None),
    )


def test_qdq_op_infer_ty_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    x = relax.Var("x", R.Tensor((n, 3), "float32"))
    dx = relax.Var("dx", R.Tensor((n, 3), "int8"))
    s = relax.Var("s", R.Tensor([3], "float32"))
    zp = relax.Var("zp", R.Tensor([3], "int8"))
    _check_inference(bb, relax.op.quantize(x, s, zp, 1, "int8"), relax.TensorType((n, 3), "int8"))
    _check_inference(
        bb,
        relax.op.dequantize(dx, s, zp, 1, "float32"),
        relax.TensorType((n, 3), "float32"),
    )


def test_qdq_float8_e4m3fn_op_infer_ty_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    x = relax.Var("x", R.Tensor((n, 3), "float32"))
    dx = relax.Var("dx", R.Tensor((n, 3), "float8_e4m3fn"))
    s = relax.Var("s", R.Tensor([3], "float32"))
    zp = relax.Var("zp", R.Tensor([3], "float16"))
    _check_inference(
        bb,
        relax.op.quantize(x, s, zp, 1, "float8_e4m3fn"),
        relax.TensorType((n, 3), "float8_e4m3fn"),
    )
    _check_inference(
        bb,
        relax.op.dequantize(dx, s, zp, 1, "float32"),
        relax.TensorType((n, 3), "float32"),
    )


def test_qdq_float8_e5m2_op_infer_ty_symbolic():
    dtype = "float8_e5m2"
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    x = relax.Var("x", R.Tensor((n, 3), "float32"))
    dx = relax.Var("dx", R.Tensor((n, 3), dtype))
    s = relax.Var("s", R.Tensor([3], "float32"))
    zp = relax.Var("zp", R.Tensor([3], "float16"))
    _check_inference(bb, relax.op.quantize(x, s, zp, 1, dtype), relax.TensorType((n, 3), dtype))
    _check_inference(
        bb,
        relax.op.dequantize(dx, s, zp, 1, "float32"),
        relax.TensorType((n, 3), "float32"),
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
