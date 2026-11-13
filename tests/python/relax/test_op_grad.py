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
from tvm_tirx.ir import Op


def test_op_correctness():
    g = relax.Var("g", R.Tensor((3, 10, 10), "float32"))
    x = relax.Var("x", R.Tensor((3, 5, 10, 10), "float32"))
    y = relax.Var("y", R.Tensor((3, 10, 10), "int64"))
    w = relax.Var("w", R.Tensor((5,), "float32"))
    assert relax.op.grad.nll_loss_backward(g, x, y, w).op == Op.get("relax.grad.nll_loss_backward")

    g = relax.Var("g", R.Tensor((3, 3, 8, 8), "float32"))
    x = relax.Var("x", R.Tensor((3, 2, 10, 10), "float32"))
    assert relax.op.grad.max_pool2d_backward(g, x, (3, 3)).op == Op.get(
        "relax.grad.max_pool2d_backward"
    )
    assert relax.op.grad.avg_pool2d_backward(g, x, (3, 3)).op == Op.get(
        "relax.grad.avg_pool2d_backward"
    )
    g = relax.Var("g", R.Tensor((3, 2, 5), "float32"))
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    indices = relax.Var("indices", R.Tensor((2,), "float32"))
    assert relax.op.grad.take_backward(g, x, indices, axis=1).op == Op.get(
        "relax.grad.take_backward"
    )
    assert relax.op.grad.no_grad(x).op == Op.get("relax.grad.no_grad")
    assert relax.op.grad.no_grad(x).args[0] == x
    assert relax.op.grad.start_checkpoint(x).op == Op.get("relax.grad.start_checkpoint")
    assert relax.op.grad.start_checkpoint(x).args[0] == x
    assert relax.op.grad.end_checkpoint(x).op == Op.get("relax.grad.end_checkpoint")
    assert relax.op.grad.end_checkpoint(x).args[0] == x


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_start_checkpoint_input_not_var():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((3, 4), "float32"))
    y = relax.Var("y", R.Tensor((3, 4), "float32"))

    # ok because x + y will be normalized into a relax Var
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.grad.start_checkpoint(x + y))
        bb.emit_func_output(gv)

    # wrong: tuple will not be normalized
    with pytest.raises(TypeError):
        bb.normalize(relax.op.grad.start_checkpoint((x, y)))

    # wrong: const will not be normalized
    with pytest.raises(TypeError):
        bb.normalize(relax.op.grad.start_checkpoint(relax.const(1, "float32")))


def test_end_checkpoint_input_not_var():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((3, 4), "float32"))
    y = relax.Var("y", R.Tensor((3, 4), "float32"))

    # ok because x + y will be normalized into a relax Var
    with bb.function("main", [x, y]):
        gv = bb.emit(relax.op.grad.end_checkpoint(x + y))
        bb.emit_func_output(gv)

    # wrong: tuple will not be normalized
    with pytest.raises(TypeError):
        bb.normalize(relax.op.grad.end_checkpoint((x, y)))

    # wrong: const will not be normalized
    with pytest.raises(TypeError):
        bb.normalize(relax.op.grad.end_checkpoint(relax.const(1, "float32")))


def test_nll_loss_backward_infer_ty():
    bb = relax.BlockBuilder()

    g = relax.Var("g", R.Tensor((3, 10, 10)))
    x = relax.Var("x", R.Tensor((3, 5, 10, 10), "float32"))
    y = relax.Var("y", R.Tensor((3, 10, 10), "int64"))
    w = relax.Var("w", R.Tensor((5,), "float32"))

    _check_inference(bb, relax.op.grad.nll_loss_backward(g, x, y), x.ty)
    _check_inference(bb, relax.op.grad.nll_loss_backward(g, x, y, w), x.ty)


def test_max_pool2d_backward_infer_ty():
    bb = relax.BlockBuilder()

    g = relax.Var("g", R.Tensor((3, 3, 8, 8), "float32"))
    x = relax.Var("x", R.Tensor((3, 2, 10, 10), "float32"))

    _check_inference(bb, relax.op.grad.max_pool2d_backward(g, x, (2, 2)), x.ty)
    _check_inference(bb, relax.op.grad.max_pool2d_backward(g, x, (3, 3)), x.ty)


def test_avg_pool2d_backward_infer_ty():
    bb = relax.BlockBuilder()

    g = relax.Var("g", R.Tensor((3, 3, 8, 8), "float32"))
    x = relax.Var("x", R.Tensor((3, 2, 10, 10), "float32"))

    _check_inference(bb, relax.op.grad.avg_pool2d_backward(g, x, (2, 2)), x.ty)
    _check_inference(bb, relax.op.grad.avg_pool2d_backward(g, x, (3, 3)), x.ty)


def test_take_backward_infer_ty():
    bb = relax.BlockBuilder()

    g = relax.Var("g", R.Tensor((3, 2, 5), "float32"))
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    indices = relax.Var("indices", R.Tensor((2,), "float32"))

    _check_inference(bb, relax.op.grad.take_backward(g, x, indices, axis=1), x.ty)


if __name__ == "__main__":
    tvm_tirx.testing.main()
