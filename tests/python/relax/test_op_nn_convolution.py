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


def test_conv1d_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    assert relax.op.nn.conv1d(x, w).op == Op.get("relax.nn.conv1d")
    assert relax.op.nn.conv1d_transpose(x, w).op == Op.get("relax.nn.conv1d_transpose")


def test_conv2d_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    assert relax.op.nn.conv2d(x, w).op == Op.get("relax.nn.conv2d")
    assert relax.op.nn.conv2d_transpose(x, w).op == Op.get("relax.nn.conv2d_transpose")


def test_conv3d_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3, 3), "float32"))
    assert relax.op.nn.conv3d(x, w).op == Op.get("relax.nn.conv3d")
    wt = relax.Var("wt", R.Tensor((3, 4, 3, 3, 3), "float32"))
    assert relax.op.nn.conv3d_transpose(x, wt).op == Op.get("relax.nn.conv3d_transpose")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_conv1d_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=3))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((48, 4, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((4, 3, 3), "float32", vdev0))

    _check_inference(bb, relax.op.nn.conv1d(x0, w0), relax.TensorType((2, 4, 26), "float32"))
    _check_inference(bb, relax.op.nn.conv1d(x6, w5), relax.TensorType((2, 4, 26), "float32", vdev0))
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, out_dtype="float16"),
        relax.TensorType((2, 4, 26), "float16"),
    )
    _check_inference(
        bb, relax.op.nn.conv1d(x0, w0, padding=1), relax.TensorType((2, 4, 28), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, padding=[1, 3]),
        relax.TensorType((2, 4, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, strides=2),
        relax.TensorType((2, 4, 13), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, strides=(2,)),
        relax.TensorType((2, 4, 13), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, dilation=2),
        relax.TensorType((2, 4, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, dilation=(2,)),
        relax.TensorType((2, 4, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w0, data_layout="NWC"),
        relax.TensorType((2, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, out_layout="NWC"),
        relax.TensorType((2, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w1, kernel_layout="IOW"),
        relax.TensorType((2, 4, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(
            x5, w4, data_layout="NCW16c", kernel_layout="OIW16i", out_layout="NWC16c"
        ),
        relax.TensorType((2, 26, 3, 16), "float32"),
    )
    _check_inference(bb, relax.op.nn.conv1d(x2, w0), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.nn.conv1d(x3, w0), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.nn.conv1d(x0, w2), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.nn.conv1d(x0, w3), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.nn.conv1d(x4, w0), relax.TensorType(dtype="", ndim=3))


def test_conv1d_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    c = tirx.Var("c", "int64")
    c16 = tirx.Var("c16", "int64")
    iw = tirx.Var("iw", "int64")
    ki = tirx.Var("ki", "int64")
    ko = tirx.Var("ko", "int64")
    kw = tirx.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ko, ki, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((ko, c, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((ko, c, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0),
        relax.TensorType((n, ko, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w1),
        relax.TensorType((n, ko, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w2, data_layout="NCW16c", kernel_layout="OIW16i", out_layout="NCW"),
        relax.TensorType((n, ko, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, strides=2, padding=1, dilation=2),
        relax.TensorType(
            (n, ko, tvm_tirx.tirx.floordiv(iw + 3, 2) + 1 - kw),
            "float32",
        ),
    )


def test_conv1d_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=3))
    s1 = relax.Var("s", relax.ShapeType(ndim=4))
    s2 = relax.Var("s", relax.ShapeType(ndim=3))
    s3 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s3, "float32"))
    w = relax.Var("w", relax.TensorType(s2, "float32"))

    _check_inference(bb, relax.op.nn.conv1d(x0, w), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w, data_layout="NCW16c"),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w, out_layout="NCW16c"),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x2, w),
        relax.TensorType(dtype="float32", ndim=3),
    )


def test_conv1d_infer_ty_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 28, 16), "float32"))
    w0 = relax.Var("w", R.Tensor((48, 16, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((48, 2, 3, 8), "float32"))

    _check_inference(
        bb, relax.op.nn.conv1d(x0, w0, groups=8), relax.TensorType((2, 48, 26), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w1, kernel_layout="OIW8i", groups=8),
        relax.TensorType((2, 48, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w0, data_layout="NCW16c", groups=8),
        relax.TensorType((2, 3, 26, 16), "float32"),
    )


def test_conv1d_infer_ty_symbolic_groups():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x = relax.Var("x", R.Tensor((n, ic * 4, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((oc * 4, ic, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((oc, ic, 3), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d(x, w0, groups=4),
        relax.TensorType((n, oc * 4, 26), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv1d(x, w1, groups=4), relax.TensorType((n, oc, 26), "float32")
    )


def test_conv1d_infer_ty_input_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((48, 20, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic * 6, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((oc, ic - 1, 3), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x0, w0, groups=6))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x1, w1, groups=6))


def test_conv1d_infer_ty_output_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 120, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 20, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic * 6, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((oc * 6 + 4, ic * 6, 3), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x0, w0, groups=6))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x1, w1, groups=6))


def test_conv1d_non_positive_group():
    x = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    w = relax.Var("w", R.Tensor((48, 16, 3), "float32"))

    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d(x, w, groups=0)
    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d(x, w, groups=-2)


def test_conv1d_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28), "float64"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3), "float64"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28), "int8"))
    w2 = relax.Var("w", R.Tensor((4, 3, 3), "int8"))
    x3 = relax.Var("x", R.Tensor((2, 3, 28), "int32"))
    w3 = relax.Var("w", R.Tensor((4, 3, 3), "int32"))

    _check_inference(bb, relax.op.nn.conv1d(x0, w0), relax.TensorType((2, 4, 26), "float16"))
    _check_inference(bb, relax.op.nn.conv1d(x1, w1), relax.TensorType((2, 4, 26), "float64"))
    _check_inference(bb, relax.op.nn.conv1d(x2, w2), relax.TensorType((2, 4, 26), "int8"))
    _check_inference(bb, relax.op.nn.conv1d(x3, w3), relax.TensorType((2, 4, 26), "int32"))


def test_conv1d_infer_ty_mixed_precision():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28), "int8"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28)))
    w2 = relax.Var("w", R.Tensor((4, 3, 3)))

    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w0, out_dtype="float32"),
        relax.TensorType((2, 4, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w1, out_dtype="int32"),
        relax.TensorType((2, 4, 26), "int32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x2, w2, out_dtype="float32"),
        relax.TensorType((2, 4, 26), "float32"),
    )


def test_conv1d_unequal_input_channel():
    bb = relax.BlockBuilder()
    ic = tirx.Var("ic", "int64")
    x0 = relax.Var("x", R.Tensor([2, 3, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([3, 4, 3], "float32"))
    x1 = relax.Var("x", R.Tensor([2, ic, 28], "float32"))
    w1 = relax.Var("w", R.Tensor([4, ic + 2, 3], "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x0, w0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x1, w1))


def test_conv1d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    conv1d = relax.op.nn.conv1d(x, w, strides=(1,), padding=(1, 1), dilation=(1,))

    assert isinstance(conv1d.attrs.strides[0], int)
    assert isinstance(conv1d.attrs.padding[0], int)
    assert isinstance(conv1d.attrs.padding[1], int)
    assert isinstance(conv1d.attrs.dilation[0], int)


def test_conv1d_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d(x, w, strides=(1, 2))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d(x, w, padding=(1, 2, 3))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d(x, w, dilation=(1, 2))


def test_conv1d_infer_ty_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x, w, data_layout="OIW"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x, w, kernel_layout="NWC"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x, w, out_layout="OWI"))


def test_conv1d_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv1d(x, w))


def test_conv1d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=2))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((4, 3, 6, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=5))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x0, w1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x0, w1, data_layout="NCW16c"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x0, w2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x1, w0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d(x2, w0))


def test_conv1d_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeType((2, 3, 28)))
    w0 = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    w1 = relax.Var("w", relax.FuncType([], R.Tensor((4, 3, 3), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv1d(x0, w1))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv1d(x1, w0))


def test_conv1d_transpose_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=3))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((4, 48, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((3, 4, 3), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x0, w0), relax.TensorType((2, 4, 30), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x6, w5),
        relax.TensorType((2, 4, 30), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, out_dtype="float16"),
        relax.TensorType((2, 4, 30), "float16"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, padding=1),
        relax.TensorType((2, 4, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, padding=[1, 3]),
        relax.TensorType((2, 4, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, strides=3, output_padding=1),
        relax.TensorType((2, 4, 85), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, strides=2),
        relax.TensorType((2, 4, 57), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, dilation=2),
        relax.TensorType((2, 4, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, dilation=(2,)),
        relax.TensorType((2, 4, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x1, w0, data_layout="NWC"),
        relax.TensorType((2, 30, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, out_layout="NWC"),
        relax.TensorType((2, 30, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w1, kernel_layout="OIW"),
        relax.TensorType((2, 4, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(
            x5, w4, data_layout="NCW16c", kernel_layout="IOW16i", out_layout="NWC16c"
        ),
        relax.TensorType((2, 30, 3, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x2, w0), relax.TensorType(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x3, w0), relax.TensorType(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x0, w2), relax.TensorType(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x0, w3), relax.TensorType(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.nn.conv1d_transpose(x4, w0), relax.TensorType(dtype="", ndim=3))


def test_conv1d_transpose_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    c = tirx.Var("c", "int64")
    c16 = tirx.Var("c16", "int64")
    iw = tirx.Var("iw", "int64")
    ki = tirx.Var("ki", "int64")
    ko = tirx.Var("ko", "int64")
    kw = tirx.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ki, ko, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((c, ko, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((c, ko, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0),
        relax.TensorType((n, ko, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w1),
        relax.TensorType((n, ko, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(
            x1, w2, data_layout="NCW16c", kernel_layout="IOW16i", out_layout="NCW"
        ),
        relax.TensorType((n, ko, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, strides=2, padding=1, dilation=2, output_padding=1),
        relax.TensorType(
            (n, ko, iw * 2 + kw * 2 - 4),
            "float32",
        ),
    )


def test_conv1d_transpose_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=3))
    s1 = relax.Var("s", relax.ShapeType(ndim=4))
    s2 = relax.Var("s", relax.ShapeType(ndim=3))
    s3 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s3, "float32"))
    w = relax.Var("w", relax.TensorType(s2, "float32"))

    _check_inference(bb, relax.op.nn.conv1d(x0, w), relax.TensorType(dtype="float32", ndim=3))
    _check_inference(
        bb,
        relax.op.nn.conv1d(x1, w, data_layout="NCW16c"),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x0, w, out_layout="NCW16c"),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d(x2, w),
        relax.TensorType(dtype="float32", ndim=3),
    )


def test_conv1d_transpose_infer_ty_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 28, 16), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 6, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((16, 6, 3, 8), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, groups=8),
        relax.TensorType((2, 48, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w1, kernel_layout="IOW8i", groups=8),
        relax.TensorType((2, 48, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x1, w0, data_layout="NCW16c", groups=8),
        relax.TensorType((2, 3, 30, 16), "float32"),
    )


def test_conv1d_transpose_infer_ty_symbolic_groups():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x = relax.Var("x", R.Tensor((n, ic * 4, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((ic, oc, 3), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x, w0, groups=4),
        relax.TensorType((n, oc * 4, 30), "float32"),
    )


def test_conv1d_transpose_infer_ty_input_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 20, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((ic - 1, oc, 3), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w0, groups=6))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x1, w1, groups=6))


def test_conv1d_transpose_non_positive_group():
    x = relax.Var("x", R.Tensor((2, 128, 28), "float32"))
    w = relax.Var("w", R.Tensor((128, 16, 3), "float32"))

    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d_transpose(x, w, groups=0)
    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d_transpose(x, w, groups=-2)


def test_conv1d_transpose_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28), "float64"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3), "float64"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28), "int8"))
    w2 = relax.Var("w", R.Tensor((3, 4, 3), "int8"))
    x3 = relax.Var("x", R.Tensor((2, 3, 28), "int32"))
    w3 = relax.Var("w", R.Tensor((3, 4, 3), "int32"))

    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x0, w0), relax.TensorType((2, 4, 30), "float16")
    )
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x1, w1), relax.TensorType((2, 4, 30), "float64")
    )
    _check_inference(bb, relax.op.nn.conv1d_transpose(x2, w2), relax.TensorType((2, 4, 30), "int8"))
    _check_inference(
        bb, relax.op.nn.conv1d_transpose(x3, w3), relax.TensorType((2, 4, 30), "int32")
    )


def test_conv1d_transpose_unequal_input_channel():
    bb = relax.BlockBuilder()
    ic = tirx.Var("ic", "int64")
    x0 = relax.Var("x", R.Tensor([2, 3, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([4, 3, 3], "float32"))
    x1 = relax.Var("x", R.Tensor([2, ic, 28], "float32"))
    w1 = relax.Var("w", R.Tensor([ic + 2, 4, 3], "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x1, w1))


def test_conv1d_transpose_wrong_output_padding():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor([2, 3, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([3, 4, 3], "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w0, strides=2, output_padding=2))


def test_conv1d_transpose_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    conv1d = relax.op.nn.conv1d_transpose(x, w, strides=1, padding=1, dilation=1)

    assert isinstance(conv1d.attrs.strides[0], int)
    assert isinstance(conv1d.attrs.padding[0], int)
    assert isinstance(conv1d.attrs.dilation[0], int)


def test_conv1d_transpose_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d_transpose(x, w, strides=(1, 2))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d_transpose(x, w, padding=(1, 2, 3))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv1d_transpose(x, w, dilation=(1, 2))


def test_conv1d_transpose_infer_ty_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x, w, data_layout="IOW"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x, w, kernel_layout="NWC"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x, w, out_layout="OWI"))


def test_conv1d_transpose_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv1d_transpose(x, w))


def test_conv1d_transpose_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=2))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 6, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=5))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w1, data_layout="NCW16c"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x1, w0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv1d_transpose(x2, w0))


def test_conv1d_transpose_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeType((2, 3, 28)))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float32"))
    w1 = relax.Var("w", relax.FuncType([], R.Tensor((3, 4, 3), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv1d_transpose(x0, w1))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv1d_transpose(x1, w0))


def test_conv1d_transpose_infer_ty_mixed_precision():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28), "int8"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3), "int8"))

    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x0, w0, out_dtype="float32"),
        relax.TensorType((2, 4, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv1d_transpose(x1, w1, out_dtype="int32"),
        relax.TensorType((2, 4, 30), "int32"),
    )


def test_conv2d_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=4))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((48, 4, 3, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32", vdev0))

    _check_inference(bb, relax.op.nn.conv2d(x0, w0), relax.TensorType((2, 4, 26, 26), "float32"))
    _check_inference(
        bb, relax.op.nn.conv2d(x6, w5), relax.TensorType((2, 4, 26, 26), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, out_dtype="float16"),
        relax.TensorType((2, 4, 26, 26), "float16"),
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x0, w0, padding=1), relax.TensorType((2, 4, 28, 28), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, padding=[1, 2]),
        relax.TensorType((2, 4, 28, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, padding=[1, 2, 3, 4]),
        relax.TensorType((2, 4, 30, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, strides=2),
        relax.TensorType((2, 4, 13, 13), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, strides=(2, 3)),
        relax.TensorType((2, 4, 13, 9), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, dilation=2),
        relax.TensorType((2, 4, 24, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, dilation=(2, 1)),
        relax.TensorType((2, 4, 24, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x1, w0, data_layout="NHWC"),
        relax.TensorType((2, 26, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, out_layout="NHWC"),
        relax.TensorType((2, 26, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w1, kernel_layout="IOHW"),
        relax.TensorType((2, 4, 26, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(
            x5, w4, data_layout="NCHW16c", kernel_layout="OIHW16i", out_layout="NHWC16c"
        ),
        relax.TensorType((2, 26, 26, 3, 16), "float32"),
    )
    _check_inference(bb, relax.op.nn.conv2d(x2, w0), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, relax.op.nn.conv2d(x3, w0), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, relax.op.nn.conv2d(x0, w2), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, relax.op.nn.conv2d(x0, w3), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(bb, relax.op.nn.conv2d(x4, w0), relax.TensorType(dtype="", ndim=4))


def test_conv2d_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    c = tirx.Var("c", "int64")
    c16 = tirx.Var("c16", "int64")
    ih = tirx.Var("ih", "int64")
    iw = tirx.Var("iw", "int64")
    ki = tirx.Var("ki", "int64")
    ko = tirx.Var("ko", "int64")
    kh = tirx.Var("kh", "int64")
    kw = tirx.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ko, ki, kh, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((ko, c, kh, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((ko, c, kh, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0),
        relax.TensorType((n, ko, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w1),
        relax.TensorType((n, ko, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(
            x1, w2, data_layout="NCHW16c", kernel_layout="OIHW16i", out_layout="NCHW"
        ),
        relax.TensorType((n, ko, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, strides=(2, 2), padding=(1, 1), dilation=(2, 2)),
        relax.TensorType(
            (
                n,
                ko,
                tvm_tirx.tirx.floordiv(ih + 3, 2) + 1 - kh,
                tvm_tirx.tirx.floordiv(iw + 3, 2) + 1 - kw,
            ),
            "float32",
        ),
    )


def test_conv2d_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=4))
    s1 = relax.Var("s", relax.ShapeType(ndim=5))
    s2 = relax.Var("s", relax.ShapeType(ndim=4))
    s3 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s3, "float32"))
    w = relax.Var("w", relax.TensorType(s2, "float32"))

    _check_inference(bb, relax.op.nn.conv2d(x0, w), relax.TensorType(dtype="float32", ndim=4))
    _check_inference(
        bb,
        relax.op.nn.conv2d(x1, w, data_layout="NCHW16c"),
        relax.TensorType(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w, out_layout="NCHW16c"),
        relax.TensorType(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x2, w),
        relax.TensorType(dtype="float32", ndim=4),
    )


def test_conv2d_infer_ty_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 28, 28, 16), "float32"))
    w0 = relax.Var("w", R.Tensor((48, 16, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((48, 2, 3, 3, 8), "float32"))

    _check_inference(
        bb, relax.op.nn.conv2d(x0, w0, groups=8), relax.TensorType((2, 48, 26, 26), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w1, kernel_layout="OIHW8i", groups=8),
        relax.TensorType((2, 48, 26, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x1, w0, data_layout="NCHW16c", groups=8),
        relax.TensorType((2, 3, 26, 26, 16), "float32"),
    )


def test_conv2d_infer_ty_symbolic_groups():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x = relax.Var("x", R.Tensor((n, ic * 4, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((oc * 4, ic, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((oc, ic, 3, 3), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d(x, w0, groups=4),
        relax.TensorType((n, oc * 4, 26, 26), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv2d(x, w1, groups=4), relax.TensorType((n, oc, 26, 26), "float32")
    )


def test_conv2d_infer_ty_input_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((48, 20, 3, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic * 6, 28, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((oc, ic - 1, 3, 3), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x0, w0, groups=6))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x1, w1, groups=6))


def test_conv2d_infer_ty_output_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 120, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 20, 3, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic * 6, 28, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((oc * 6 + 4, ic * 6, 3, 3), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x0, w0, groups=6))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x1, w1, groups=6))


def test_conv2d_non_positive_group():
    x = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((48, 16, 3, 3), "float32"))

    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d(x, w, groups=0)
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d(x, w, groups=-2)


def test_conv2d_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float64"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float64"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    w2 = relax.Var("w", R.Tensor((4, 3, 3, 3), "int8"))
    x3 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int32"))
    w3 = relax.Var("w", R.Tensor((4, 3, 3, 3), "int32"))

    _check_inference(bb, relax.op.nn.conv2d(x0, w0), relax.TensorType((2, 4, 26, 26), "float16"))
    _check_inference(bb, relax.op.nn.conv2d(x1, w1), relax.TensorType((2, 4, 26, 26), "float64"))
    _check_inference(bb, relax.op.nn.conv2d(x2, w2), relax.TensorType((2, 4, 26, 26), "int8"))
    _check_inference(bb, relax.op.nn.conv2d(x3, w3), relax.TensorType((2, 4, 26, 26), "int32"))


def test_conv2d_infer_ty_mixed_precision():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28, 28)))
    w2 = relax.Var("w", R.Tensor((4, 3, 3, 3)))

    _check_inference(
        bb,
        relax.op.nn.conv2d(x0, w0, out_dtype="float32"),
        relax.TensorType((2, 4, 26, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x1, w1, out_dtype="int32"),
        relax.TensorType((2, 4, 26, 26), "int32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d(x2, w2, out_dtype="float32"),
        relax.TensorType((2, 4, 26, 26), "float32"),
    )


def test_conv2d_unequal_input_channel():
    bb = relax.BlockBuilder()
    ic = tirx.Var("ic", "int64")
    x0 = relax.Var("x", R.Tensor([2, 3, 28, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([3, 4, 3, 3], "float32"))
    x1 = relax.Var("x", R.Tensor([2, ic, 28, 28], "float32"))
    w1 = relax.Var("w", R.Tensor([4, ic + 2, 3, 3], "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x0, w0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x1, w1))


def test_conv2d_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    conv2d = relax.op.nn.conv2d(x, w, strides=(1, 1), padding=(1, 1), dilation=(1, 1))

    assert isinstance(conv2d.attrs.strides[0], int)
    assert isinstance(conv2d.attrs.strides[1], int)
    assert isinstance(conv2d.attrs.padding[0], int)
    assert isinstance(conv2d.attrs.padding[1], int)
    assert isinstance(conv2d.attrs.padding[2], int)
    assert isinstance(conv2d.attrs.padding[3], int)
    assert isinstance(conv2d.attrs.dilation[0], int)
    assert isinstance(conv2d.attrs.dilation[1], int)


def test_conv2d_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d(x, w, strides=(1, 2, 3))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d(x, w, padding=(1, 2, 3))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d(x, w, dilation=(1, 2, 3))


def test_conv2d_infer_ty_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x, w, data_layout="OIHW"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x, w, kernel_layout="NHWC"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x, w, out_layout="OHWI"))


def test_conv2d_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((4, 3, 3, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv2d(x, w))


def test_conv2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((4, 3, 6, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=6))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x0, w1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x0, w1, data_layout="NCHW16c"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x0, w2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x1, w0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d(x2, w0))


def test_conv2d_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeType((2, 3, 28, 28)))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    w1 = relax.Var("w", relax.FuncType([], R.Tensor((4, 3, 3, 3), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv2d(x0, w1))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv2d(x1, w0))


def test_conv2d_transpose_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((4, 3, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=4))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((4, 48, 3, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w0), relax.TensorType((2, 4, 30, 30), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x6, w5),
        relax.TensorType((2, 4, 30, 30), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, out_dtype="float16"),
        relax.TensorType((2, 4, 30, 30), "float16"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, padding=1),
        relax.TensorType((2, 4, 28, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, padding=[1, 2]),
        relax.TensorType((2, 4, 28, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, padding=[1, 2, 3, 4]),
        relax.TensorType((2, 4, 26, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, strides=3, output_padding=1),
        relax.TensorType((2, 4, 85, 85), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, strides=3, output_padding=[2, 1]),
        relax.TensorType((2, 4, 86, 85), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, strides=2),
        relax.TensorType((2, 4, 57, 57), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, strides=(2, 3)),
        relax.TensorType((2, 4, 57, 84), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, dilation=2),
        relax.TensorType((2, 4, 32, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, dilation=(2, 1)),
        relax.TensorType((2, 4, 32, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x1, w0, data_layout="NHWC"),
        relax.TensorType((2, 30, 30, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, out_layout="NHWC"),
        relax.TensorType((2, 30, 30, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w1, kernel_layout="OIHW"),
        relax.TensorType((2, 4, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(
            x5, w4, data_layout="NCHW16c", kernel_layout="IOHW16i", out_layout="NHWC16c"
        ),
        relax.TensorType((2, 30, 30, 3, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x2, w0), relax.TensorType(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x3, w0), relax.TensorType(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w2), relax.TensorType(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w3), relax.TensorType(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.nn.conv2d_transpose(x4, w0), relax.TensorType(dtype="", ndim=4))


def test_conv2d_transpose_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    c = tirx.Var("c", "int64")
    c16 = tirx.Var("c16", "int64")
    ih = tirx.Var("ih", "int64")
    iw = tirx.Var("iw", "int64")
    ki = tirx.Var("ki", "int64")
    ko = tirx.Var("ko", "int64")
    kh = tirx.Var("kh", "int64")
    kw = tirx.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ki, ko, kh, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((c, ko, kh, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((c, ko, kh, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0),
        relax.TensorType((n, ko, ih + kh - 1, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w1),
        relax.TensorType((n, ko, ih + kh - 1, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(
            x1, w2, data_layout="NCHW16c", kernel_layout="IOHW16i", out_layout="NCHW"
        ),
        relax.TensorType((n, ko, ih + kh - 1, iw + kw - 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(
            x0, w0, strides=(2, 2), padding=(1, 1), output_padding=(1, 0), dilation=(2, 2)
        ),
        relax.TensorType(
            (n, ko, ih * 2 + kh * 2 - 4, iw * 2 + kw * 2 - 5),
            "float32",
        ),
    )


def test_conv2d_transpose_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=4))
    s1 = relax.Var("s", relax.ShapeType(ndim=5))
    s2 = relax.Var("s", relax.ShapeType(ndim=4))
    s3 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s3, "float32"))
    w = relax.Var("w", relax.TensorType(s2, "float32"))

    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w), relax.TensorType(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x1, w, data_layout="NCHW16c"),
        relax.TensorType(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w, out_layout="NCHW16c"),
        relax.TensorType(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x2, w),
        relax.TensorType(dtype="float32", ndim=4),
    )


def test_conv2d_transpose_infer_ty_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 28, 28, 16), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 6, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((16, 6, 3, 3, 8), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, groups=8),
        relax.TensorType((2, 48, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w1, kernel_layout="IOHW8i", groups=8),
        relax.TensorType((2, 48, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x1, w0, data_layout="NCHW16c", groups=8),
        relax.TensorType((2, 3, 30, 30, 16), "float32"),
    )


def test_conv2d_transpose_infer_ty_symbolic_groups():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x = relax.Var("x", R.Tensor((n, ic * 4, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((ic, oc, 3, 3), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x, w0, groups=4),
        relax.TensorType((n, oc * 4, 30, 30), "float32"),
    )


def test_conv2d_transpose_infer_ty_input_channel_group_incompatible():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    ic = tirx.Var("c", "int64")
    oc = tirx.Var("oc", "int64")
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 20, 3, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((n, ic, 28, 28), "float32"))
    w1 = relax.Var("w", R.Tensor((ic - 1, oc, 3, 3), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w0, groups=6))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x1, w1, groups=6))


def test_conv2d_transpose_non_positive_group():
    x = relax.Var("x", R.Tensor((2, 128, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((128, 16, 3, 3), "float32"))

    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d_transpose(x, w, groups=0)
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d_transpose(x, w, groups=-2)


def test_conv2d_transpose_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float64"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float64"))
    x2 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    w2 = relax.Var("w", R.Tensor((3, 4, 3, 3), "int8"))
    x3 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int32"))
    w3 = relax.Var("w", R.Tensor((3, 4, 3, 3), "int32"))

    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x0, w0), relax.TensorType((2, 4, 30, 30), "float16")
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x1, w1), relax.TensorType((2, 4, 30, 30), "float64")
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x2, w2), relax.TensorType((2, 4, 30, 30), "int8")
    )
    _check_inference(
        bb, relax.op.nn.conv2d_transpose(x3, w3), relax.TensorType((2, 4, 30, 30), "int32")
    )


def test_conv2d_transpose_unequal_input_channel():
    bb = relax.BlockBuilder()
    ic = tirx.Var("ic", "int64")
    x0 = relax.Var("x", R.Tensor([2, 3, 28, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([4, 3, 3, 3], "float32"))
    x1 = relax.Var("x", R.Tensor([2, ic, 28, 28], "float32"))
    w1 = relax.Var("w", R.Tensor([ic + 2, 4, 3, 3], "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x1, w1))


def test_conv2d_transpose_wrong_output_padding():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor([2, 3, 28, 28], "float32"))
    w0 = relax.Var("w", R.Tensor([3, 4, 3, 3], "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w0, strides=2, output_padding=2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w0, strides=(2, 2), output_padding=(2, 2)))


def test_conv2d_transpose_stride_padding_dilation_int64():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    conv2d_transpose = relax.op.nn.conv2d_transpose(
        x, w, strides=(1, 1), padding=(1, 1), output_padding=(1, 2), dilation=(1, 1)
    )

    assert isinstance(conv2d_transpose.attrs.strides[0], int)
    assert isinstance(conv2d_transpose.attrs.strides[1], int)
    assert isinstance(conv2d_transpose.attrs.padding[0], int)
    assert isinstance(conv2d_transpose.attrs.padding[1], int)
    assert isinstance(conv2d_transpose.attrs.padding[2], int)
    assert isinstance(conv2d_transpose.attrs.padding[3], int)
    assert isinstance(conv2d_transpose.attrs.output_padding[0], int)
    assert isinstance(conv2d_transpose.attrs.output_padding[1], int)
    assert isinstance(conv2d_transpose.attrs.dilation[0], int)
    assert isinstance(conv2d_transpose.attrs.dilation[1], int)


def test_conv2d_transpose_wrong_strides_padding_dilation_length():
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d_transpose(x, w, strides=(1, 2, 3))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d_transpose(x, w, padding=(1, 2, 3))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d_transpose(x, w, output_padding=(1, 2, 3))
    with pytest.raises(RuntimeError):
        relax.op.nn.conv2d_transpose(x, w, dilation=(1, 2, 3))


def test_conv2d_transpose_infer_ty_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x, w, data_layout="IOHW"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x, w, kernel_layout="NHWC"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x, w, out_layout="OHWI"))


def test_conv2d_transpose_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    w = relax.Var("w", R.Tensor((3, 4, 3, 3), "int8"))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv2d_transpose(x, w))


def test_conv2d_transpose_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 6, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=6))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w1, data_layout="NCHW16c"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x1, w0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv2d_transpose(x2, w0))


def test_conv2d_transpose_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeType((2, 3, 28, 28)))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float32"))
    w1 = relax.Var("w", relax.FuncType([], R.Tensor((3, 4, 3, 3), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv2d_transpose(x0, w1))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.nn.conv2d_transpose(x1, w0))


def test_conv2d_transpose_infer_ty_mixed_precision():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float16"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3, 3), "int8"))

    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x0, w0, out_dtype="float32"),
        relax.TensorType((2, 4, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv2d_transpose(x1, w1, out_dtype="int32"),
        relax.TensorType((2, 4, 30, 30), "int32"),
    )


def test_conv3d_transpose_infer_ty():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3, 3), "float32"))
    _check_inference(
        bb,
        relax.op.nn.conv3d_transpose(x0, w0),
        relax.TensorType((2, 4, 30, 30, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d_transpose(x0, w0, padding=1),
        relax.TensorType((2, 4, 28, 28, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d_transpose(x0, w0, strides=2, output_padding=1),
        relax.TensorType((2, 4, 58, 58, 58), "float32"),
    )


def test_conv3d_transpose_infer_ty_ndhwc_out_layout():
    bb = relax.BlockBuilder()
    x_ndhwc = relax.Var("x_nd", R.Tensor((2, 28, 28, 28, 3), "float32"))
    x_ncdhw = relax.Var("x_nc", R.Tensor((2, 3, 28, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3, 3), "float32"))
    _check_inference(
        bb,
        relax.op.nn.conv3d_transpose(x_ndhwc, w0, data_layout="NDHWC"),
        relax.TensorType((2, 30, 30, 30, 4), "float32"),
    )
    # Default data_layout is NCDHW; use NCDHW-shaped input when only out_layout is NDHWC.
    _check_inference(
        bb,
        relax.op.nn.conv3d_transpose(x_ncdhw, w0, out_layout="NDHWC"),
        relax.TensorType((2, 30, 30, 30, 4), "float32"),
    )


def test_conv3d_transpose_infer_ty_groups():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 128, 28, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((128, 16, 3, 3, 3), "float32"))
    _check_inference(
        bb,
        relax.op.nn.conv3d_transpose(x0, w0, groups=8),
        relax.TensorType((2, 128, 30, 30, 30), "float32"),
    )


def test_conv3d_transpose_wrong_output_padding():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((3, 4, 3, 3, 3), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv3d_transpose(x0, w0, strides=2, output_padding=2))
    with pytest.raises(ValueError):
        bb.normalize(
            relax.op.nn.conv3d_transpose(x0, w0, strides=(2, 2, 2), output_padding=(2, 2, 2))
        )


def test_conv3d_transpose_unequal_input_channel():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    w0 = relax.Var("w", R.Tensor((4, 4, 3, 3, 3), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.nn.conv3d_transpose(x0, w0))


def test_conv3d_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 28, 28, 28, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=5))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 4, 28, 28, 28, 16), "float32"))
    x6 = relax.Var("x", R.Tensor((2, 3, 28, 28, 28), "float32", vdev0))
    w0 = relax.Var("w", R.Tensor((4, 3, 3, 3, 3), "float32"))
    w1 = relax.Var("w", R.Tensor((3, 4, 3, 3, 3), "float32"))
    w2 = relax.Var("w", R.Tensor("float32", ndim=5))
    w3 = relax.Var("w", R.Tensor("float32"))
    w4 = relax.Var("w", R.Tensor((48, 4, 3, 3, 3, 16), "float32"))
    w5 = relax.Var("w", R.Tensor((4, 3, 3, 3, 3), "float32", vdev0))

    _check_inference(
        bb, relax.op.nn.conv3d(x0, w0), relax.TensorType((2, 4, 26, 26, 26), "float32")
    )
    _check_inference(
        bb, relax.op.nn.conv3d(x6, w5), relax.TensorType((2, 4, 26, 26, 26), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, out_dtype="float16"),
        relax.TensorType((2, 4, 26, 26, 26), "float16"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, padding=1),
        relax.TensorType((2, 4, 28, 28, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, padding=[1, 2, 3]),
        relax.TensorType((2, 4, 28, 30, 32), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, padding=[1, 2, 3, 4, 5, 6]),
        relax.TensorType((2, 4, 31, 33, 35), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, strides=2),
        relax.TensorType((2, 4, 13, 13, 13), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, strides=(2, 3, 4)),
        relax.TensorType((2, 4, 13, 9, 7), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, dilation=2),
        relax.TensorType((2, 4, 24, 24, 24), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, dilation=(3, 2, 1)),
        relax.TensorType((2, 4, 22, 24, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x1, w0, data_layout="NDHWC"),
        relax.TensorType((2, 26, 26, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, out_layout="NDHWC"),
        relax.TensorType((2, 26, 26, 26, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w1, kernel_layout="IODHW"),
        relax.TensorType((2, 4, 26, 26, 26), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(
            x5, w4, data_layout="NCDHW16c", kernel_layout="OIDHW16i", out_layout="NDHWC16c"
        ),
        relax.TensorType((2, 26, 26, 26, 3, 16), "float32"),
    )
    _check_inference(bb, relax.op.nn.conv3d(x2, w0), relax.TensorType(dtype="float32", ndim=5))
    _check_inference(bb, relax.op.nn.conv3d(x3, w0), relax.TensorType(dtype="float32", ndim=5))
    _check_inference(bb, relax.op.nn.conv3d(x0, w2), relax.TensorType(dtype="float32", ndim=5))
    _check_inference(bb, relax.op.nn.conv3d(x0, w3), relax.TensorType(dtype="float32", ndim=5))
    _check_inference(bb, relax.op.nn.conv3d(x4, w0), relax.TensorType(dtype="", ndim=5))


def test_conv3d_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    c = tirx.Var("c", "int64")
    c16 = tirx.Var("c16", "int64")
    id = tirx.Var("id", "int64")
    ih = tirx.Var("ih", "int64")
    iw = tirx.Var("iw", "int64")
    ki = tirx.Var("ki", "int64")
    ko = tirx.Var("ko", "int64")
    kd = tirx.Var("kd", "int64")
    kh = tirx.Var("kh", "int64")
    kw = tirx.Var("kw", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, id, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, id, ih, iw, c16), "float32"))
    w0 = relax.Var("w", R.Tensor((ko, ki, kd, kh, kw), "float32"))
    w1 = relax.Var("w", R.Tensor((ko, c, kd, kh, kw), "float32"))
    w2 = relax.Var("w", R.Tensor((ko, c, kd, kh, kw, c16), "float32"))

    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0),
        relax.TensorType((n, ko, id + 1 - kd, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w1),
        relax.TensorType((n, ko, id + 1 - kd, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(
            x1, w2, data_layout="NCDHW16c", kernel_layout="OIDHW16i", out_layout="NCDHW"
        ),
        relax.TensorType((n, ko, id + 1 - kd, ih + 1 - kh, iw + 1 - kw), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w0, strides=(2, 2, 2), padding=(1, 1, 1), dilation=(2, 2, 2)),
        relax.TensorType(
            (
                n,
                ko,
                tvm_tirx.tirx.floordiv(id + 3, 2) + 1 - kd,
                tvm_tirx.tirx.floordiv(ih + 3, 2) + 1 - kh,
                tvm_tirx.tirx.floordiv(iw + 3, 2) + 1 - kw,
            ),
            "float32",
        ),
    )


def test_conv3d_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=5))
    s1 = relax.Var("s", relax.ShapeType(ndim=6))
    s2 = relax.Var("s", relax.ShapeType(ndim=5))
    s3 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s3, "float32"))
    w = relax.Var("w", relax.TensorType(s2, "float32"))

    _check_inference(bb, relax.op.nn.conv3d(x0, w), relax.TensorType(dtype="float32", ndim=5))
    _check_inference(
        bb,
        relax.op.nn.conv3d(x1, w, data_layout="NCDHW16c"),
        relax.TensorType(dtype="float32", ndim=6),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x0, w, out_layout="NCDHW16c"),
        relax.TensorType(dtype="float32", ndim=6),
    )
    _check_inference(
        bb,
        relax.op.nn.conv3d(x2, w),
        relax.TensorType(dtype="float32", ndim=5),
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
