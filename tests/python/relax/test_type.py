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
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm_tirx import tirx


def _check_equal(x, y, map_free_vars=False):
    tvm_tirx.ir.assert_structural_equal(x, y, map_free_vars)
    tvm_tirx.ir.assert_structural_equal(y, x, map_free_vars)

    xhash = tvm_ffi.structural_hash(x, map_free_vars)
    yhash = tvm_ffi.structural_hash(y, map_free_vars)

    assert xhash == yhash


def _check_json_roundtrip(x):
    xret = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(x))
    _check_equal(x, xret, map_free_vars=True)
    return xret


def test_any_ty():
    s0 = rx.AnyType()
    s1 = rx.AnyType()

    # can turn into str
    str(s0)
    _check_equal(s0, s1)

    assert isinstance(s0, rx.AnyType)
    _check_json_roundtrip(s0)


def test_object_ty_compat_alias():
    s0 = rx.ObjectType()

    assert isinstance(s0, rx.AnyType)
    assert rx.ObjectType is rx.AnyType
    _check_equal(s0, rx.AnyType())


def test_shape_type():
    t0 = rx.ShapeType(ndim=-1)
    t1 = rx.ShapeType(ndim=-1)
    assert t0 == t1


def test_dyn_tensor_type():
    t0 = rx.TensorType()
    assert t0.ndim == -1
    t1 = rx.TensorType(ndim=3, dtype="int32")
    assert t1.ndim == 3
    assert t1.dtype == "int32"


def test_prim_ty():
    s0 = tvm_tirx.ir.PrimType("float32")
    s1 = tvm_tirx.ir.PrimType("float32")
    s2 = tvm_tirx.ir.PrimType("int32")

    _check_equal(s0, s1)

    # can turn into str
    str(s0)

    assert s0 == s1
    assert s0 != s2

    assert isinstance(s0, tvm_tirx.ir.PrimType)
    _check_json_roundtrip(s0)
    _check_json_roundtrip(s1)

    assert s1.dtype == "float32"
    assert s2.dtype == "int32"

    # wrong API constructors
    with pytest.raises((RuntimeError, TypeError)):
        tvm_tirx.ir.PrimType([1])


def test_shape_ty():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")

    s0 = rx.ShapeType([1, n + 1, m])
    s1 = rx.ShapeType([1, n + 1, m])

    _check_equal(s0, s1)

    assert s0 == s1
    assert s0.ndim == 3
    assert s1.ndim == 3

    assert s0.values[2] == m

    assert isinstance(s0, rx.ShapeType)
    _check_json_roundtrip(s0)
    _check_json_roundtrip(s1)

    s2 = rx.ShapeType(ndim=2)

    assert s2.ndim == 2
    assert s2.values is None
    _check_json_roundtrip(s2)
    assert s0 != s2

    # can turn into str
    str(s0)

    # wrong argument type
    with pytest.raises((RuntimeError, TypeError)):
        rx.ShapeType(1)

    # cannot pass both ndim and values
    with pytest.raises(ValueError):
        rx.ShapeType([1, 2], ndim=3)

    # cannot pass both ndim and values even if they are consistent
    with pytest.raises(ValueError):
        rx.ShapeType([1, 2], ndim=2)


def test_tensor_ty():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")

    s0 = rx.TensorType([1, n + 1, m], "float32")
    s1 = rx.TensorType(rx.ShapeExpr([1, n + 1, m]), "float32")

    _check_equal(s0, s1)

    assert s0 == s1
    assert s0.ndim == 3
    assert s1.ndim == 3

    assert isinstance(s0, rx.TensorType)
    _check_json_roundtrip(s0)
    _check_json_roundtrip(s1)

    s2 = rx.TensorType(ndim=2, dtype="int32")

    assert s2.ndim == 2
    assert s2.dtype == "int32"
    assert s2.shape is None
    _check_json_roundtrip(s2)
    assert s0 != s2

    # take in opaque var
    rshape = rx.Var("shape", rx.ShapeType(ndim=2))

    s3 = rx.TensorType(rshape, dtype="int32")
    assert s3.dtype == "int32"
    assert s3.shape == rshape
    assert s3.ndim == 2
    _check_json_roundtrip(s3)

    # can turn into str
    str(s0)

    # cannot pass both ndim and values
    with pytest.raises(ValueError):
        rx.TensorType([1, 2], ndim=3)

    # cannot pass both ndim and values even if they are consistent
    with pytest.raises(ValueError):
        rx.TensorType([1, 2], ndim=2)


def test_tuple_ty():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")

    s0 = rx.TensorType([1, 2, m + n], "float32")
    s1 = rx.AnyType()

    t0 = rx.TupleType([s0, s1])
    t1 = rx.TupleType([s0, rx.AnyType()])
    t2 = rx.TupleType([s0, s0])

    _check_equal(t0, t1)

    assert t0 == t1

    assert rx.TupleType is tvm_tirx.ir.TupleType
    assert isinstance(t0, tvm_tirx.ir.TupleType)
    assert t0.__class__ is tvm_tirx.ir.TupleType
    t0 = _check_json_roundtrip(t0)
    t1 = _check_json_roundtrip(t1)
    t2 = _check_json_roundtrip(t2)

    # can turn into str
    str(t0)

    # wrong argument type
    with pytest.raises(TypeError):
        rx.TupleType(1)


def test_func_ty():
    def fn_info(c):
        n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
        x = rx.TensorType([c, n, m], "float32")
        y = rx.TensorType([c, n, 1], "float32")
        z = rx.TensorType([c, n, m], "float32")
        return rx.FuncType([x, y], z)

    f0 = fn_info(1)
    f1 = fn_info(1)
    f2 = fn_info(2)
    f3 = rx.FuncType.opaque_func()

    _check_equal(f0, f1)

    assert f0 == f1
    assert f0 != f2

    assert len(f0.params) == 2
    assert isinstance(f0.ret, rx.TensorType)
    assert f2.derive_func is None
    assert f3.params is None
    assert f3.derive_func is None
    _check_equal(f3.ret, rx.AnyType())

    assert isinstance(f0, rx.FuncType)
    f0 = _check_json_roundtrip(f0)
    f1 = _check_json_roundtrip(f1)
    f2 = _check_json_roundtrip(f2)
    f3 = _check_json_roundtrip(f3)

    # can turn into str
    str(f3)


if __name__ == "__main__":
    tvm_tirx.testing.main()
