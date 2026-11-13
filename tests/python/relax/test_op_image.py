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
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing

pytest.importorskip("scipy")  # tvm.topi.testing imports scipy

import tvm.topi.testing
from tvm import relax
from tvm.relax import VDevice
from tvm.relax import script as R
from tvm_tirx import tirx
from tvm_tirx.ir import Op


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    assert relax.op.image.resize2d(x, (28, 28)).op == Op.get("relax.image.resize2d")
    theta = relax.Var("theta", R.Tensor((2, 2, 3), "float32"))
    assert relax.op.image.affine_grid(theta, (16, 16)).op == Op.get("relax.image.affine_grid")
    y = relax.Var("y", R.Tensor((2, 3, 8, 16, 32), "float32"))
    assert relax.op.image.resize3d(y, (4, 8, 12)).op == Op.get("relax.image.resize3d")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_resize2d_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor((2, 4, 32, 32, 16), "float32"))
    x3 = relax.Var("x", R.Tensor("float32", ndim=4))
    x4 = relax.Var("x", R.Tensor("float32", ndim=5))
    x5 = relax.Var("x", R.Tensor("float32"))
    x6 = relax.Var("x", R.Tensor(ndim=4))
    x7 = relax.Var("x", R.Tensor())
    x8 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32", vdev0))

    _check_inference(
        bb, relax.op.image.resize2d(x0, (28, 28)), relax.TensorType((2, 3, 28, 28), "float32")
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x8, (28, 28)),
        relax.TensorType((2, 3, 28, 28), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x0, size=28),
        relax.TensorType((2, 3, 28, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x0, size=(28, 30)),
        relax.TensorType((2, 3, 28, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x1, size=28, layout="NHWC"),
        relax.TensorType((2, 28, 28, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x0, size=28, out_dtype="float16"),
        relax.TensorType((2, 3, 28, 28), "float16"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x2, size=28, layout="NCHW16c"),
        relax.TensorType((2, 4, 28, 28, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.image.resize2d(x3, size=28), relax.TensorType(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x4, size=28, layout="NCHW16c"),
        relax.TensorType(dtype="float32", ndim=5),
    )
    _check_inference(
        bb, relax.op.image.resize2d(x5, size=28), relax.TensorType(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.image.resize2d(x6, size=28), relax.TensorType(dtype="", ndim=4))
    _check_inference(
        bb,
        relax.op.image.resize2d(x6, size=28, out_dtype="float32"),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(bb, relax.op.image.resize2d(x7, size=28), relax.TensorType(dtype="", ndim=4))


def test_resize2d_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    c = tirx.Var("c", "int64")
    ih = tirx.Var("ih", "int64")
    iw = tirx.Var("iw", "int64")
    oh = tirx.Var("oh", "int64")
    ow = tirx.Var("ow", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, 16), "float32"))

    _check_inference(
        bb, relax.op.image.resize2d(x0, size=oh), relax.TensorType((n, c, oh, oh), "float32")
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x0, size=(oh, ow)),
        relax.TensorType((n, c, oh, ow), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x1, size=(oh, ow), layout="NCHW16c"),
        relax.TensorType((n, c, oh, ow, 16), "float32"),
    )


def test_resize2d_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(ndim=4))
    s1 = relax.Var("s", relax.ShapeType(ndim=5))
    s2 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))
    x2 = relax.Var("x", relax.TensorType(s2, "float32"))

    _check_inference(
        bb, relax.op.image.resize2d(x0, size=32), relax.TensorType(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x1, size=32, layout="NCHW16c"),
        relax.TensorType(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x2, size=32, layout="NCHW16c"),
        relax.TensorType(dtype="float32", ndim=5),
    )


def test_resize2d_infer_ty_pool_symbolic_shape():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    s0 = relax.Var("s", relax.ShapeType((30, 30)))
    s1 = relax.Var("s", relax.ShapeType(ndim=2))

    _check_inference(
        bb,
        relax.op.image.resize2d(x0, s0),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(bb, relax.op.image.resize2d(x0, s1), relax.TensorType(dtype="float32", ndim=4))


def test_resize2d_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int64"))
    _check_inference(
        bb, relax.op.image.resize2d(x0, size=28), relax.TensorType((2, 3, 28, 28), "float16")
    )
    _check_inference(
        bb, relax.op.image.resize2d(x1, size=28), relax.TensorType((2, 3, 28, 28), "int8")
    )
    _check_inference(
        bb, relax.op.image.resize2d(x2, size=28), relax.TensorType((2, 3, 28, 28), "int64")
    )


def test_resize3d_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 8, 16, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 8, 16, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor((2, 4, 8, 16, 32, 8), "float32"))
    x3 = relax.Var("x", R.Tensor("float32", ndim=5))
    x4 = relax.Var("x", R.Tensor((2, 3, 8, 16, 32), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.image.resize3d(x0, (4, 8, 12)),
        relax.TensorType((2, 3, 4, 8, 12), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize3d(x4, (4, 8, 12)),
        relax.TensorType((2, 3, 4, 8, 12), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.image.resize3d(x0, 7),
        relax.TensorType((2, 3, 7, 7, 7), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize3d(x1, (4, 8, 12), layout="NDHWC"),
        relax.TensorType((2, 4, 8, 12, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize3d(x2, (4, 8, 12), layout="NCDHW8c"),
        relax.TensorType((2, 4, 4, 8, 12, 8), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize3d(x0, (4, 8, 12), out_dtype="float16"),
        relax.TensorType((2, 3, 4, 8, 12), "float16"),
    )
    _check_inference(
        bb, relax.op.image.resize3d(x3, (4, 8, 12)), relax.TensorType(dtype="float32", ndim=5)
    )


def test_resize3d_infer_ty_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 8, 16, 32), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x, size=(4, 8, 12), layout="OIHW"))


def test_resize3d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 8, 16, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 8, 16, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, size=(4, 8, 12), layout="NCDHW8c"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x1, size=(4, 8, 12), layout="NCDHW"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x2, size=(4, 8, 12)))


def test_resize3d_wrong_size_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 8, 16, 32), "float16"))
    s0 = relax.ShapeExpr((3, 3))
    s1 = relax.Var("s", relax.ShapeType((30, 30, 30, 30)))
    s2 = relax.Var("s", relax.ShapeType(ndim=4))
    s3 = relax.Var("s", relax.ShapeType(ndim=2))
    s4 = relax.Var("s", relax.ShapeType(ndim=1))
    s5 = relax.Var("s", relax.ShapeType(ndim=0))
    s6 = relax.Var("s", relax.ShapeType())

    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, (3, 3)))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, s0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, s1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, s2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, s3))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, s4))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, s5))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize3d(x0, s6))


def test_resize3d_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 8, 16, 32)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 8, 16, 32), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3, 8, 16, 32), "float32"))
    s0 = relax.Var("s", R.Tensor((3, 3, 3)))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.image.resize3d(x0, size=(4, 8, 12)))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.image.resize3d(x1, size=(4, 8, 12)))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.image.resize3d(x2, s0))


def test_resize2d_infer_ty_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x, size=28, layout="OIHW"))


def test_resize2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x0, size=28, layout="NCHW16c"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x1, size=28, layout="NCHW"))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x2, size=28))


def test_resize2d_wrong_pool_size_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    s0 = relax.ShapeExpr((3,))
    s1 = relax.Var("s", relax.ShapeType((30, 30, 30)))
    s2 = relax.Var("s", relax.ShapeType(ndim=3))
    s3 = relax.Var("s", relax.ShapeType(ndim=1))
    s4 = relax.Var("s", relax.ShapeType(ndim=0))
    s5 = relax.Var("s", relax.ShapeType())

    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x0, (3, 3, 3)))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x0, s0))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x0, s1))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x0, s2))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x0, s3))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x0, s4))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.resize2d(x0, s5))


def test_resize2d_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 28, 28)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 28, 28), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    s0 = relax.Var("s", R.Tensor((3, 3)))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.image.resize2d(x0, size=32))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.image.resize2d(x1, size=32))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.image.resize2d(x2, s0))


def test_affine_grid_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 2, 3), "float32", vdev0))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    x3 = relax.Var("x", R.Tensor("float32"))
    x4 = relax.Var("x", R.Tensor(ndim=3))

    _check_inference(
        bb,
        relax.op.image.affine_grid(x0, (16, 16)),
        relax.TensorType((2, 2, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.affine_grid(x1, (16, 16)),
        relax.TensorType((2, 2, 16, 16), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.image.affine_grid(x0, size=16),
        relax.TensorType((2, 2, 16, 16), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.affine_grid(x0, size=(16, 20)),
        relax.TensorType((2, 2, 16, 20), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.affine_grid(x2, size=(16, 16)),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.image.affine_grid(x3, size=(16, 16)),
        relax.TensorType(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.image.affine_grid(x4, size=(16, 16)),
        relax.TensorType(dtype="", ndim=4),
    )


def test_affine_grid_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    oh = tirx.Var("oh", "int64")
    ow = tirx.Var("ow", "int64")
    x0 = relax.Var("x", R.Tensor((n, 2, 3), "float32"))

    _check_inference(
        bb,
        relax.op.image.affine_grid(x0, size=(oh, ow)),
        relax.TensorType((n, 2, oh, ow), "float32"),
    )


def test_affine_grid_infer_ty_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 2, 3)))
    x1 = relax.Var("x", R.Tensor((2, 2, 3), "float32"))
    s0 = relax.Var("s", R.Tensor((3, 3)))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.image.affine_grid(x0, size=(16, 16)))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.image.affine_grid(x1, s0))


def test_affine_grid_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.affine_grid(x0, size=(16, 16)))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.affine_grid(x1, size=(16, 16)))


def test_affine_grid_wrong_size_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 2, 3), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.affine_grid(x0, (16, 16, 16)))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.image.affine_grid(x0, (16,)))


@pytest.mark.parametrize(
    "batch, target_h, target_w",
    [
        (1, 16, 16),
        (2, 8, 12),
        (4, 32, 32),
    ],
)
def test_affine_grid_e2e(batch, target_h, target_w):
    """End-to-end numerical correctness test: build, run, compare with numpy reference."""

    @tvm_tirx.script.ir_module
    class AffineGridModule:
        @R.function
        def main(theta: R.Tensor((batch, 2, 3), "float32")) -> R.Tensor("float32", ndim=4):
            gv = R.image.affine_grid(theta, size=(target_h, target_w))
            return gv

    target = "llvm"
    dev = tvm_tirx.cpu()
    exe = tvm.compile(AffineGridModule, target=target)
    vm = relax.VirtualMachine(exe, dev)

    theta_np = np.random.uniform(-1, 1, size=(batch, 2, 3)).astype("float32")
    theta_nd = tvm_tirx.runtime.tensor(theta_np, dev)

    out_nd = vm["main"](theta_nd)
    out_np = out_nd.numpy()

    ref_np = tvm.topi.testing.affine_grid_python(theta_np, (target_h, target_w))

    tvm_tirx.testing.assert_allclose(out_np, ref_np, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm_tirx.testing.main()
