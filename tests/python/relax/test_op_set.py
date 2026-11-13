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
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    assert relax.op.unique(x).op == Op.get("relax.unique")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_ty: relax.Type):
    ret = bb.normalize(call)
    tvm_tirx.ir.assert_structural_equal(ret.ty, expected_ty)


def test_unique_infer_ty():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x4, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType(dtype="float32", ndim=1, vdevice=vdev0),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorType(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=False, return_counts=True, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=True, return_counts=False, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=False, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=True, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=False, return_counts=False, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=False, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=False, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=False, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=False, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=-2),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, sorted=True, return_index=True, return_inverse=True, return_counts=True, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, sorted=True, return_index=True, return_inverse=True, return_counts=True, axis=1
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x1, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorType(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x1, return_index=False, return_inverse=True, return_counts=False, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=True, return_counts=False, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=False, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=False, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x2, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorType(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x2, return_index=True, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=False, return_counts=False, axis=1),
        relax.TupleType(
            [relax.TensorType(dtype="float32"), relax.TensorType(dtype="int64", ndim=1)]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=False, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=False, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x3, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType(dtype="", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorType(dtype="", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x3, return_index=False, return_inverse=False, return_counts=True, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )


def test_unique_infer_ty_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tirx.Var("a", "int64")
    b = tirx.Var("b", "int64")
    c = tirx.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(
        bb,
        relax.op.unique(
            x, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorType(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=False, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )


def test_unique_infer_ty_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeType())
    x0 = relax.Var("x", relax.TensorType(s0, "float32"))
    x1 = relax.Var("x", relax.TensorType(s1, "float32"))

    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorType(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=False, return_counts=True, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=3),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x1, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorType(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x1, return_index=False, return_inverse=False, return_counts=True, axis=None
        ),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleType(
            [relax.TensorType(dtype="float32"), relax.TensorType(dtype="int64", ndim=1)]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleType(
            [
                relax.TensorType(dtype="float32"),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )


def test_unique_infer_ty_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 4), "int32"))

    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="float16", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="int8", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType(dtype="int32", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
                relax.TensorType(dtype="int64", ndim=1),
            ]
        ),
    )


def test_unique_infer_ty_input_zero_rank():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeType(()))
    s1 = relax.Var("s", relax.ShapeType(ndim=0))
    x0 = relax.Var("x", R.Tensor((), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=0))
    x2 = relax.Var("x", relax.TensorType(s0, "float32"))
    x3 = relax.Var("x", relax.TensorType(s1, "float32"))

    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleType(
            [
                relax.TensorType((1,), "float32"),
                relax.TensorType((1,), "int64"),
                relax.TensorType((1,), "int64"),
                relax.TensorType((1,), "int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=False, axis=None),
        relax.TupleType(
            [
                relax.TensorType((1,), "float32"),
                relax.TensorType((1,), "int64"),
                relax.TensorType((1,), "int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x2, return_index=True, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TupleType([relax.TensorType((1,), "float32"), relax.TensorType((1,), "int64")]),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x3, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorType((1,), "float32"),
    )


def test_unique_infer_ty_axis_out_of_range():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((), "float32"))

    with pytest.raises(ValueError):
        bb.normalize(relax.op.unique(x0, axis=3))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.unique(x0, axis=-4))
    with pytest.raises(ValueError):
        bb.normalize(relax.op.unique(x1, axis=0))


def test_unique_infer_ty_wrong_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.unique(x0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.unique(x1))


@pytest.mark.parametrize("shape", [(1,), (2, 3), (4, 5, 6)])
def test_nonzero_infer_ty(shape):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor(shape, "bool"))

    _check_inference(
        bb,
        relax.op.nonzero(x0),
        relax.TensorType(ndim=2, dtype="int64"),
    )


def test_nonzero_infer_ty_ndim_zero():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((), "bool"))

    _check_inference(
        bb,
        relax.op.nonzero(x),
        relax.TensorType(ndim=2, dtype="int64"),
    )


def test_nonzero_infer_ty_wrong_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeType((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncType([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TypeError):
        bb.normalize(relax.op.nonzero(x0))
    with pytest.raises(TypeError):
        bb.normalize(relax.op.nonzero(x1))


if __name__ == "__main__":
    tvm_tirx.testing.main()
