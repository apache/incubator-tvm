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
# ruff: noqa: E731, F401, F841

"""Tests Relax dependent type analysis functions."""

import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax as rx
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm_tirx import ir, tirx
from tvm_tirx.script import tirx as T


def test_get_static_type_basic():
    # object
    s0 = rx.AnyType()
    tvm_tirx.ir.assert_structural_equal(rx.analysis.get_static_type(s0), rx.AnyType())

    # prim
    s1 = tvm_tirx.ir.PrimType("float32")
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.get_static_type(s1), tvm_tirx.ir.PrimType("float32")
    )


def test_get_static_type_shape():
    # shape
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")

    s2 = rx.ShapeType([1, n + 1, m])
    s3 = rx.ShapeType(ndim=2)

    tvm_tirx.ir.assert_structural_equal(rx.analysis.get_static_type(s2), rx.ShapeType(ndim=3))

    tvm_tirx.ir.assert_structural_equal(rx.analysis.get_static_type(s3), rx.ShapeType(ndim=2))


def test_get_static_type_tensor():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    s4 = rx.TensorType([1, n + 1, m], "int64")

    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.get_static_type(s4), rx.TensorType(ndim=3, dtype="int64")
    )


def test_get_static_type_tuple():
    # tuple
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    s0 = rx.AnyType()
    s2 = rx.ShapeType([1, n + 1, m])
    s4 = rx.TensorType([1, n + 1, m], "int64")
    t0 = rx.TupleType([s4, s0])
    t1 = rx.TupleType([t0, s2])

    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.get_static_type(t1),
        rx.TupleType(
            [
                rx.TupleType([rx.TensorType(ndim=3, dtype="int64"), rx.AnyType()]),
                rx.ShapeType(ndim=3),
            ]
        ),
    )


def test_get_static_type_func():
    # tuple
    def fn_info(c):
        n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
        x = rx.TensorType([c, n, m], "float32")
        y = rx.TensorType([c, n, 1], "float32")
        z = rx.TensorType([c, n], "float32")
        return rx.FuncType([x, y], z)

    def fn_type():
        x = rx.TensorType(ndim=3, dtype="float32")
        y = rx.TensorType(ndim=3, dtype="float32")
        z = rx.TensorType(ndim=2, dtype="float32")
        return rx.FuncType([x, y], z)

    f0 = fn_info(1)

    tvm_tirx.ir.assert_structural_equal(rx.analysis.get_static_type(fn_info(1)), fn_type())


def test_erase_to_well_defined_basic():
    s0 = rx.AnyType()
    tvm_tirx.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s0), s0)

    # prim
    s1 = tvm_tirx.ir.PrimType("float32")
    tvm_tirx.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s1), s1)


def test_erase_to_well_defined_shape():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")

    s2 = rx.ShapeType([1, n + 1, m])
    s3 = rx.ShapeType(ndim=2)
    # have undefined
    tvm_tirx.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s2), rx.ShapeType(ndim=3))
    # all defined
    tvm_tirx.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s2, {n: n, m: m}), s2)

    # replacement
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s2, {n: 2, m: m + 1}), rx.ShapeType([1, 3, m + 1])
    )

    # partial defined
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s2, {n: n}), rx.ShapeType(ndim=3)
    )


def test_erase_to_well_defined_tensor():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    rshape = rx.Var("shape", rx.ShapeType(ndim=2))
    s0 = rx.TensorType(rshape, dtype="int32")

    # undefined
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s0, None, None),
        rx.TensorType(ndim=2, dtype="int32"),
    )

    # defined
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s0, None, {rshape: rshape}), s0
    )

    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s0, None, {rshape: rx.ShapeExpr([1, 2])}),
        rx.TensorType([1, 2], dtype="int32"),
    )

    s1 = rx.TensorType([m + 1, n], dtype="float32")

    tvm_tirx.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s1, {n: n, m: m}), s1)

    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s1, {n: 2, m: 3}),
        rx.TensorType([4, 2], dtype="float32"),
    )

    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s1, {m: m}, {rshape: rshape}),
        rx.TensorType(ndim=2, dtype="float32"),
    )

    s2 = rx.TensorType([1, 2], dtype="float32")

    tvm_tirx.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s2), s2)


def test_erase_to_well_defined_tuple():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    s0 = rx.AnyType()
    s2 = rx.ShapeType([1, m])
    s4 = rx.TensorType([1, n + 1, m], "int64")
    t0 = rx.TupleType([s4, s0])
    t1 = rx.TupleType([t0, s2])

    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(t1, {m: m + 1}),
        rx.TupleType(
            [
                rx.TupleType([rx.TensorType(ndim=3, dtype="int64"), rx.AnyType()]),
                rx.ShapeType([1, m + 1]),
            ]
        ),
    )


def test_erase_to_well_defined_func():
    def fn_info(c):
        n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
        x = rx.TensorType([c, n, m], "float32")
        y = rx.TensorType([c, n, 1], "float32")
        z = rx.TensorType([c, n], "float32")
        return rx.FuncType([x, y], z)

    f0 = fn_info(1)

    tvm_tirx.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(f0), f0)


def test_base_check():
    BR = rx.analysis.BaseCheckResult
    bcheck = rx.analysis.type_base_check

    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    obj0 = rx.AnyType()
    prim0 = tvm_tirx.ir.PrimType("int32")
    prim1 = tvm_tirx.ir.PrimType("float32")

    shape0 = rx.ShapeType(ndim=-1)
    shape1 = rx.ShapeType(ndim=2)
    shape2 = rx.ShapeType(ndim=3)
    shape3 = rx.ShapeType([1, 2, 3])
    shape4 = rx.ShapeType([1, n, 3])

    vdevice0 = rx.VDevice()
    vdevice1 = rx.VDevice("llvm")
    vdevice2 = rx.VDevice("cuda", 0)
    vdevice3 = rx.VDevice("cuda", 2)
    vdevice4 = rx.VDevice("cuda", 0, "global")

    tensor0 = rx.TensorType(ndim=-1, dtype="int32")
    tensor1 = rx.TensorType(ndim=-1, dtype="float32")
    tensor2 = rx.TensorType(ndim=2, dtype="int32")
    tensor3 = rx.TensorType(ndim=2, dtype="float32")
    tensor4 = rx.TensorType([n, m], "int32")
    tensor5 = rx.TensorType([n, m, 1], "int32")
    tensor6 = rx.TensorType([n, m, 2], "int32")
    tensor7 = rx.TensorType(ndim=2, dtype="float32", vdevice=vdevice0)
    tensor8 = rx.TensorType(ndim=2, dtype="float32", vdevice=vdevice1)
    tensor9 = rx.TensorType(ndim=2, dtype="float32", vdevice=vdevice2)
    tensor10 = rx.TensorType(ndim=2, dtype="float32", vdevice=vdevice3)
    tensor11 = rx.TensorType(ndim=2, dtype="float32", vdevice=vdevice4)
    tensor12 = rx.TensorType([n, m, 2], "int32", vdevice0)
    tensor13 = rx.TensorType([n, m, 2], "int32", vdevice1)
    tensor14 = rx.TensorType([n, m, 2], "int32", vdevice2)
    tensor15 = rx.TensorType([n, m, 2], "int32", vdevice3)
    tensor16 = rx.TensorType([n, m, 2], "int32", vdevice4)

    # obj
    assert bcheck(obj0, prim0) == BR.PASS
    assert bcheck(obj0, shape1) == BR.PASS
    assert bcheck(obj0, tensor2) == BR.PASS
    assert obj0.is_base_of(tensor2)

    # prim
    assert prim0.is_base_of(prim0)
    assert not prim0.is_base_of(prim1)
    assert bcheck(prim0, obj0) == BR.FAIL_L1
    assert bcheck(prim0, prim0) == BR.PASS
    assert bcheck(prim0, prim1) == BR.FAIL_L0

    # shape
    assert bcheck(shape0, obj0) == BR.FAIL_L1
    assert bcheck(shape0, prim0) == BR.FAIL_L0

    # unknown dim
    assert bcheck(shape0, shape1) == BR.PASS
    assert bcheck(shape1, shape0) == BR.FAIL_L1

    # ndim mismatch
    assert bcheck(shape1, shape2) == BR.FAIL_L0

    # lhs do not have symbolic value but ndim match
    assert bcheck(shape2, shape3) == BR.PASS

    # rhs do not symbolic but lhs do
    assert bcheck(shape3, shape2) == BR.FAIL_L2

    # shape mismatch
    assert bcheck(shape3, shape4) == BR.FAIL_L2
    assert shape4.is_base_of(rx.ShapeType([1, n, 3]))

    # tensor
    assert bcheck(tensor0, obj0) == BR.FAIL_L1
    assert bcheck(tensor0, prim0) == BR.FAIL_L0
    assert bcheck(tensor0, shape0) == BR.FAIL_L0

    # dtype mismatch
    assert bcheck(tensor0, tensor1) == BR.FAIL_L0
    assert bcheck(tensor0, tensor3) == BR.FAIL_L0
    assert bcheck(tensor3, tensor4) == BR.FAIL_L0
    assert bcheck(tensor1, tensor2) == BR.FAIL_L0

    # vdevice mismatch
    assert bcheck(tensor8, tensor9) == BR.FAIL_L0
    assert bcheck(tensor9, tensor10) == BR.FAIL_L0
    assert bcheck(tensor10, tensor11) == BR.FAIL_L0
    assert bcheck(tensor13, tensor14) == BR.FAIL_L0
    assert bcheck(tensor14, tensor15) == BR.FAIL_L0
    assert bcheck(tensor15, tensor16) == BR.FAIL_L0

    # ndim mismatch
    assert bcheck(tensor2, tensor5) == BR.FAIL_L0

    # static shape mismatch
    assert bcheck(tensor5, tensor6) == BR.FAIL_L0

    # match
    assert tensor0.is_base_of(rx.TensorType(ndim=-1, dtype="int32"))
    assert tensor0.is_base_of(tensor2)
    assert tensor0.is_base_of(tensor4)
    assert tensor0.is_base_of(tensor5)
    assert tensor0.is_base_of(tensor6)
    assert tensor2.is_base_of(tensor4)
    assert tensor3.is_base_of(tensor7)
    assert tensor3.is_base_of(tensor8)
    assert tensor6.is_base_of(tensor12)
    assert tensor6.is_base_of(tensor13)
    assert tensor4.is_base_of(rx.TensorType([n, m], dtype="int32"))

    # tuple
    t0 = rx.TupleType([obj0, tensor0])
    t1 = rx.TupleType([prim0, tensor4])
    t2 = rx.TupleType([obj0, tensor0, obj0])
    t3 = rx.TupleType([tensor0, obj0])

    assert t0.is_base_of(t1)

    assert bcheck(t0, t2) == BR.FAIL_L0
    assert bcheck(t0, t3) == BR.FAIL_L1

    assert rx.TupleType([t0, t1]).is_base_of(rx.TupleType([t1, t1]))
    assert bcheck(rx.TupleType([t0, t1]), rx.TupleType([t1, t0])) == BR.FAIL_L1

    def fn_info_shape(c):
        n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
        x = rx.TensorType([c, n, m], "float32")
        y = rx.TensorType([c, n, 1], "float32")
        z = rx.TensorType([c, n], "float32")
        return rx.FuncType([x, y], z)

    def fn_info_erased():
        x = rx.TensorType(ndim=3, dtype="float32")
        y = rx.TensorType(ndim=3, dtype="float32")
        z = rx.TensorType(ndim=2, dtype="float32")
        return rx.FuncType([x, y], z)

    assert fn_info_shape(1).is_base_of(fn_info_shape(1))
    assert fn_info_erased().is_base_of(fn_info_shape(1))
    assert bcheck(fn_info_shape(1), fn_info_erased()) == BR.FAIL_L2

    fopaque = rx.FuncType.opaque_func()
    assert fopaque.is_base_of(fn_info_shape(1))


def _check_derive(ctx, finfo, args_ty, ret):
    gv = rx.GlobalVar("test")
    rx.expr._update_type(gv, finfo)
    args = []
    for i, ty in enumerate(args_ty):
        arg = rx.Var(f"arg{i}", ty)
        args.append(arg)
    call = rx.Call(gv, args)
    derived_ret = rx.analysis.derive_call_ret_type(finfo, call, ctx)
    tvm_tirx.ir.assert_structural_equal(ret, derived_ret)


def test_derive_call_ret_type():
    obj0 = rx.AnyType()
    prim0 = tvm_tirx.ir.PrimType("float32")

    n, m = tirx.Var("n0", "int64"), tirx.Var("m0", "int64")
    bb = rx.BlockBuilder()
    # derivation cases
    with bb.testing_scope(def_vars=[n, m]):

        def func0(c):
            n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
            x = rx.TensorType([n, m], "float32")
            z = rx.TensorType([m + c, n], "float32")
            return rx.FuncType([x], z)

        # Tensor => Tensor
        _check_derive(
            bb,
            func0(1),
            [rx.TensorType([10, 11], "float32")],
            rx.TensorType([12, 10], "float32"),
        )

        _check_derive(
            bb,
            func0(2),
            [rx.TensorType([n, m], "float32")],
            rx.TensorType([m + 2, n], "float32"),
        )

        # passing in information that cannot deduce n, m
        # it is still OK as type still matches, return an
        # eriased output
        _check_derive(
            bb,
            func0(2),
            [rx.TensorType(ndim=2, dtype="float32")],
            rx.TensorType(ndim=2, dtype="float32"),
        )

        # Error: wrong number of arguments
        with pytest.raises(ValueError):
            _check_derive(
                bb,
                func0(2),
                [rx.TensorType(ndim=2, dtype="float32"), obj0],
                rx.TensorType(ndim=2, dtype="float32"),
            )

        # Error:type mismatch
        with pytest.raises(ValueError):
            _check_derive(bb, func0(2), [obj0], obj0)

        # Tensor with vdevice
        vdev = rx.VDevice("llvm")

        def func1(c):
            n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
            x = rx.TensorType([n, m], "float32", vdev)
            z = rx.TensorType([m + c, n], "float32", vdev)
            return rx.FuncType([x], z)

        _check_derive(
            bb,
            func1(1),
            [rx.TensorType([10, 11], "float32", vdev)],
            rx.TensorType([12, 10], "float32", vdev),
        )

        # opaque derivation
        fopaque0 = lambda: rx.FuncType.opaque_func()
        fopaque1 = lambda: rx.FuncType.opaque_func(ret=prim0)
        _check_derive(bb, fopaque0(), [obj0, prim0], obj0)
        _check_derive(bb, fopaque1(), [obj0, prim0], prim0)

        # recursive tuple derivation
        def func_tuple0(c):
            n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
            x0 = rx.TensorType([n, c], "float32")
            x1 = rx.TensorType([n + c, m], "float32")
            z = rx.TupleType([rx.TensorType([m, n], "float32")])
            return rx.FuncType([rx.TupleType([x0, x1])], z)

        _check_derive(
            bb,
            func_tuple0(2),
            [
                rx.TupleType(
                    [
                        rx.TensorType([n, 2], "float32"),
                        rx.TensorType([n + 2, 10], "float32"),
                    ]
                )
            ],
            rx.TupleType([rx.TensorType([10, n], "float32")]),
        )

        def func_tuple1(c):
            n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
            x0 = rx.TensorType([n, m], "float32")
            x1 = rx.TensorType([n + c, c], "float32")
            z = rx.TupleType([rx.TensorType([m, n], "float32")])
            return rx.FuncType([rx.TupleType([x0, x1])], z)

        # Still OK, to pass erased tensor into n+2, n is captured by other argument.
        _check_derive(
            bb,
            func_tuple1(4),
            [
                rx.TupleType(
                    [
                        rx.TensorType([n, 4], "float32"),
                        rx.TensorType(ndim=2, dtype="float32"),
                    ]
                )
            ],
            rx.TupleType([rx.TensorType([4, n], "float32")]),
        )

        # tuple length mismatch is not causes an error
        with pytest.raises(ValueError):
            _check_derive(
                bb,
                func_tuple0(4),
                [rx.TupleType([rx.TensorType([n, 4], "float32")])],
                rx.TupleType([rx.TensorType([10, n], "float32")]),
            )

        # mixed shape types
        def func_shape_mixed(c):
            n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
            x0 = rx.ShapeType([n, m])
            f0 = func_tuple0(c)
            z = rx.ShapeType([m + n, c])
            return rx.FuncType([x0, f0], z)

        _check_derive(
            bb,
            func_shape_mixed(3),
            [
                rx.ShapeType([10, 20]),
                # have to specify purity because an impure function cannot be passed
                # where a pure one is expected
                rx.FuncType.opaque_func(ret=rx.ShapeType(ndim=2), purity=True),
            ],
            rx.ShapeType([30, 3]),
        )


def _check_lca(lhs, rhs, target):
    tvm_tirx.ir.assert_structural_equal(rx.analysis.type_lca(lhs, rhs), target)
    tvm_tirx.ir.assert_structural_equal(rx.analysis.type_lca(rhs, lhs), target)


def test_type_lca():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    obj0 = rx.AnyType()
    prim0 = tvm_tirx.ir.PrimType("int32")
    prim1 = tvm_tirx.ir.PrimType("float32")

    vdevice0 = rx.VDevice("llvm")
    vdevice1 = rx.VDevice("cuda", 0)

    shape0 = rx.ShapeType(ndim=-1)
    shape1 = rx.ShapeType(ndim=2)
    shape2 = rx.ShapeType(ndim=3)
    shape3 = rx.ShapeType([1, 2, 3])
    shape4 = rx.ShapeType([1, n, 3])

    tensor0 = rx.TensorType(ndim=-1, dtype="int32")
    tensor1 = rx.TensorType(ndim=-1, dtype="float32")
    tensor2 = rx.TensorType(ndim=2, dtype="int32")
    tensor3 = rx.TensorType(ndim=2, dtype="float32")
    tensor4 = rx.TensorType([n, m], "int32")
    tensor5 = rx.TensorType([n, m, 1], "int32")
    tensor6 = rx.TensorType([n, m, 2], "int32")
    tensor7 = rx.TensorType(ndim=2, dtype="float32", vdevice=vdevice0)
    tensor8 = rx.TensorType(ndim=2, dtype="float32", vdevice=vdevice1)
    tensor9 = rx.TensorType([n, m, 2], "int32", vdevice0)
    tensor10 = rx.TensorType([n, m, 2], "int32", vdevice1)

    # obj
    _check_lca(obj0, prim0, obj0)
    _check_lca(obj0, prim1, obj0)

    # shape
    _check_lca(shape0, tensor0, obj0)
    _check_lca(shape0, shape1, shape0)
    _check_lca(shape1, shape2, shape0)
    _check_lca(shape1, shape3, shape0)

    _check_lca(shape2, shape3, shape2)
    _check_lca(shape3, shape4, shape2)
    _check_lca(shape4, rx.ShapeType([1, n, 3]), shape4)

    # tensor
    _check_lca(tensor0, prim0, obj0)
    _check_lca(tensor0, tensor1, rx.TensorType(ndim=-1, dtype=None))
    _check_lca(tensor0, tensor2, tensor0)
    _check_lca(tensor0, tensor4, tensor0)
    _check_lca(tensor0, tensor4, tensor0)
    _check_lca(tensor1, tensor3, tensor1)
    _check_lca(tensor3, tensor7, tensor3)
    _check_lca(tensor3, tensor8, tensor3)
    _check_lca(tensor1, tensor8, tensor1)
    _check_lca(tensor6, tensor9, tensor6)
    _check_lca(tensor6, tensor10, tensor6)

    _check_lca(tensor2, tensor4, tensor2)
    _check_lca(tensor5, tensor6, rx.TensorType(ndim=3, dtype="int32"))
    _check_lca(tensor4, tensor5, rx.TensorType(ndim=-1, dtype="int32"))
    _check_lca(tensor4, rx.TensorType([n, m], dtype="int32"), tensor4)

    # tuple
    t0 = rx.TupleType([obj0, tensor0])
    t1 = rx.TupleType([prim0, tensor4])
    t2 = rx.TupleType([obj0, tensor0, obj0])
    t3 = rx.TupleType([tensor0, obj0])

    _check_lca(t0, t1, t0)
    _check_lca(t0, t2, obj0)
    _check_lca(t0, t3, rx.TupleType([obj0, obj0]))

    t5 = rx.TupleType([t0, t1])
    t6 = rx.TupleType([t1, t2])

    _check_lca(t5, t6, rx.TupleType([t0, obj0]))

    t7 = rx.TupleType([])
    _check_lca(t7, rx.TupleType([]), t7)

    def fn_info_shape(c):
        n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
        x = rx.TensorType([c, n, m], "float32")
        y = rx.TensorType([c, n, 1], "float32")
        z = rx.TensorType([c, n], "float32")
        return rx.FuncType([x, y], z)

    def fn_info_erased():
        x = rx.TensorType(ndim=3, dtype="float32")
        y = rx.TensorType(ndim=3, dtype="float32")
        z = rx.TensorType(ndim=2, dtype="float32")
        return rx.FuncType([x, y], z)

    fopaque0 = lambda: rx.FuncType.opaque_func()
    fopaque1 = lambda: rx.FuncType.opaque_func(ret=prim0)
    fopaque2 = lambda: rx.FuncType.opaque_func(ret=rx.TensorType(ndim=2, dtype="float32"))

    _check_lca(fn_info_shape(1), fn_info_shape(2), fn_info_erased())
    _check_lca(fn_info_shape(2), fn_info_shape(2), fn_info_shape(2))

    _check_lca(fopaque0(), fopaque1(), fopaque0())
    _check_lca(fopaque0(), fn_info_shape(1), fopaque0())
    _check_lca(fopaque2(), fn_info_shape(1), fopaque2())


def _generate_prim_test_cases():
    dtypes = [
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]

    for dtype in dtypes:
        # LCA of a PrimType with itself yields itself
        yield (
            tvm_tirx.ir.PrimType(dtype),
            tvm_tirx.ir.PrimType(dtype),
            tvm_tirx.ir.PrimType(dtype),
        )

    for dtype_a in dtypes:
        for dtype_b in dtypes:
            if dtype_a != dtype_b:
                # If the dtype differs between the two annotations,
                # the next wider category is R.Any.
                yield (tvm_tirx.ir.PrimType(dtype_a), tvm_tirx.ir.PrimType(dtype_b), R.Any)


@pytest.mark.parametrize("test_case", list(_generate_prim_test_cases()))
def test_prim_type_lca(test_case):
    def _normalize_ty(ty):
        if isinstance(ty, tvm.relax.Type):
            return ty
        elif callable(ty):
            return ty()
        else:
            raise TypeError(f"Cannot normalize {type(ty)} to Type")

    lhs, rhs, expected = map(_normalize_ty, test_case)

    lca = rx.analysis.type_lca(lhs, rhs)
    assert tvm_ffi.structural_equal(lca, expected), (
        f"Expected {lhs} and {rhs} to have LCA of {expected}, but instead found {lca}"
    )


def _generate_tir_var_test_cases():
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    shape0 = rx.ShapeType([1, n, 3])
    shape1 = rx.ShapeType([1, 2 * n, n, m])
    shape2 = rx.ShapeType([1, 2 * n, m])
    tensor0 = rx.TensorType([1, n, 3], "int32")
    tensor1 = rx.TensorType([1, 2 * n, n, m], "int32")
    tensor2 = rx.TensorType([1, 2 * n, m], "int32")
    func = rx.FuncType(
        [rx.TensorType([1, 2 * n, n, m], "int32")], rx.TensorType([1, n, 3], "int32")
    )

    yield shape0, [n], [n]
    yield shape1, [n, m], [n, m]
    yield shape2, [m], [n, m]
    yield tensor0, [n], [n]
    yield tensor1, [n, m], [n, m]
    yield tensor2, [m], [n, m]
    yield func, [n, m], [n, m]


@pytest.fixture(params=[*_generate_tir_var_test_cases()], scope="session")
def tir_var_test_case(request):
    return request.param


def test_tir_vars_in_type(tir_var_test_case):
    ty, _vars_definable, vars_used = tir_var_test_case
    tvm_tirx.ir.assert_structural_equal(rx.analysis.tir_vars_in_type(ty), vars_used)


def test_definable_tir_vars_in_type(tir_var_test_case):
    ty, vars_definable, _vars_used = tir_var_test_case
    tvm_tirx.ir.assert_structural_equal(rx.analysis.definable_tir_vars_in_type(ty), vars_definable)


def test_collect_symbolic_var_from_tensor_shape():
    n, m, k, q, p = (
        tirx.Var("n", "int64"),
        tirx.Var("m", "int64"),
        tirx.Var("k", "int64"),
        tirx.Var("q", "int64"),
        tirx.Var("p", "int64"),
    )
    bb = rx.BlockBuilder()
    x = rx.Var("x", rx.TensorType([m, m + n], "float32"))
    with bb.function("main", [x]):
        v0 = bb.match_cast(x, rx.TensorType([m, k], "float32"))
        v1 = bb.emit(
            rx.call_dps_packed(_ExternFunc("test"), x, ty_args=[rx.TensorType([p, q], "float32")])
        )
        bb.emit_func_output(rx.const(1))
    func = bb.get()["main"]

    defined_vars = set(rx.analysis.defined_symbolic_vars(func))
    free_vars = set(rx.analysis.free_symbolic_vars(func))
    assert defined_vars == {m, k}
    assert free_vars == {n, p, q}


@pytest.fixture(params=["shape_expr"], scope="session")
def param_type(request):
    return request.param


@pytest.fixture(params=["definition_first", "usage_first"], scope="session")
def param_order(request):
    return request.param


def test_collect_symbolic_var_from_non_tensor_params(param_type, param_order):
    tir_n = tirx.Var("n", "int64")
    tir_m = tirx.Var("m", "int64")

    bb = rx.BlockBuilder()
    arg = rx.Var("arg", rx.TensorType([tir_n * tir_m]))

    if param_type == "shape_expr":
        extra_params = [
            rx.Var("shape_expr", rx.ShapeType([tir_n, tir_m])),
        ]
    else:
        raise ValueError(f"Unknown param_type: {param_type}")

    if param_order == "definition_first":
        params = [*extra_params, arg]
    elif param_order == "usage_first":
        params = [arg, *extra_params]
    else:
        raise ValueError(f"Unknown param_order: {param_order}")

    with bb.function("main", params=params):
        out = rx.op.reshape(arg, [tir_n, tir_m])
        bb.emit_func_output(out)
    func = bb.get()["main"]

    defined_vars = set(rx.analysis.defined_symbolic_vars(func))
    free_vars = set(rx.analysis.free_symbolic_vars(func))
    assert defined_vars == {tir_n, tir_m}
    assert free_vars == set()


def test_collect_nonnegative_expressions():
    M = T.dynamic("M", "int64")
    N = T.dynamic("N", "int64")

    @R.function
    def func(
        A: R.Tensor([1024, M, N - 2]),
        B: R.Tensor([128, N, M + 2]),
        C: R.Shape([M, N]),
        D: T.int64,
    ):
        return ()

    M, N = list(func.params[2].ty.values)

    # Expressions are de-duplicated, in order of their first appearance
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.collect_non_negative_expressions(func.ty),
        [M, N - 2, N, M + 2],
    )

    # Tensor shapes can imply that their shapes are non-negative
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.collect_non_negative_expressions(func.params[0].ty),
        [M, N - 2],
    )
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.collect_non_negative_expressions(func.params[1].ty),
        [N, M + 2],
    )

    # ShapeExpr values can imply that their contents are non-negative
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.collect_non_negative_expressions(func.params[2].ty),
        [M, N],
    )

    # Expr instances may contain negative values, and do not
    # imply that their contents are non-negative.
    tvm_tirx.ir.assert_structural_equal(
        rx.analysis.collect_non_negative_expressions(func.params[3].ty),
        [],
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
