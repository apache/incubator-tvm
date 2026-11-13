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
# pylint: disable=missing-function-docstring,missing-module-docstring
# ruff: noqa: E501, F841
from __future__ import annotations

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import s_tir, te, topi
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import Range
from tvm_tirx.script import tirx as T


def test_unique_name_complete_block():
    A = te.placeholder((16, 16), name="A")
    B = te.compute((16, 16), lambda x, y: A[x, y] * 2, name="main")
    C = te.compute((16, 16), lambda x, y: B[x, y] + 1, name="main")
    func = te.create_function([A, C])
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    assert isinstance(s.get_sref(s.get_sblock("main")), s_tir.schedule.StmtSRef)
    assert isinstance(s.get_sref(s.get_sblock("main_1")), s_tir.schedule.StmtSRef)


def test_unique_name_reduction_block():
    k1 = te.reduce_axis((0, 16), "k1")
    k2 = te.reduce_axis((0, 16), "k2")
    A = te.placeholder((16, 16), name="A")
    B = te.compute((16,), lambda i: te.sum(A[i, k1], axis=k1), name="sum")
    C = te.compute((), lambda: te.sum(B[k2], axis=k2), name="sum")
    func = te.create_function([A, C])
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    assert isinstance(s.get_sref(s.get_sblock("sum")), s_tir.schedule.StmtSRef)
    assert isinstance(s.get_sref(s.get_sblock("sum_1")), s_tir.schedule.StmtSRef)


def _check_workload(te_workload, tir_workload, index_dtype_override=None, do_simplify=False):
    func = te.create_function(te_workload(), index_dtype_override)
    if do_simplify:
        simplify = tirx.transform.StmtSimplify()
        func = simplify(tvm_tirx.IRModule.from_expr(func))["main"]
        tir_workload = simplify(tvm_tirx.IRModule.from_expr(tir_workload))["main"]
    tvm_tirx.ir.assert_structural_equal(func, tir_workload)
    # make sure that we can create schedule from the func
    s = tvm.s_tir.Schedule(func, debug_mask="all")
    assert s


def te_matmul():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    return [A, B, C]


@Ts.function
def tir_matmul(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    for i0, j0, k0 in T.grid(128, 128, 128):
        with Ts.sblock():
            i, j, k = Ts.axis.remap("SSR", [i0, j0, k0])
            with Ts.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


@Ts.function
def tir_matmul_int64(
    A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    C: T.Tensor((T.int64(128), T.int64(128)), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    for i0, j0, k0 in T.grid(T.int64(128), T.int64(128), T.int64(128)):
        with Ts.sblock():
            i, j, k = Ts.axis.remap("SSR", [i0, j0, k0])
            with Ts.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


def test_matmul():
    _check_workload(te_matmul, tir_matmul)


def test_matmul_int64():
    _check_workload(te_matmul, tir_matmul_int64, index_dtype_override="int64")


def te_element_wise():
    A = te.placeholder((128, 128), name="A")
    B = te.compute((128, 128), lambda x, y: A[x, y] * 2, name="B")
    C = te.compute((128, 128), lambda x, y: B[x, y] + 1, name="C")
    return [A, C]


@Ts.function
def tir_element_wise(
    A: T.Tensor((128, 128), "float32"), C: T.Tensor((128, 128), "float32")
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    B = Ts.sblock_alloc_tensor((128, 128))

    for i0, j0 in T.grid(128, 128):
        with Ts.sblock():
            i, j = Ts.axis.remap("SS", [i0, j0])
            B[i, j] = A[i, j] * 2.0
    for i0, j0 in T.grid(128, 128):
        with Ts.sblock():
            i, j = Ts.axis.remap("SS", [i0, j0])
            C[i, j] = B[i, j] + 1.0


def test_element_wise():
    _check_workload(te_element_wise, tir_element_wise)


def te_conv2d():
    batch = 16
    in_channel = 16
    out_channel = 32
    size = 14
    kernel = 3

    A = te.placeholder((batch, in_channel, size, size), name="A")
    W = te.placeholder((in_channel, kernel, kernel, out_channel), name="W")
    Apad = te.compute(
        (batch, in_channel, size + 2, size + 2),
        lambda nn, cc, yy, xx: tvm_tirx.tirx.if_then_else(
            tvm_tirx.tirx.all(yy >= 1, yy - 1 < size, xx >= 1, xx - 1 < size),
            A[nn, cc, yy - 1, xx - 1],
            0.0,
        ),
        name="Apad",
    )
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel), name="ry")
    rx = te.reduce_axis((0, kernel), name="rx")
    B = te.compute(
        (batch, out_channel, size, size),
        lambda nn, ff, yy, xx: te.sum(
            Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff], axis=[rc, ry, rx]
        ),
        name="B",
    )
    return [A, W, B]


@Ts.function
def tir_conv2d(
    A: T.Tensor([16, 16, 14, 14], "float32"),
    W: T.Tensor([16, 3, 3, 32], "float32"),
    B: T.Tensor([16, 32, 14, 14], "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    Apad = Ts.sblock_alloc_tensor([16, 16, 16, 16])

    for n, c, y, x in T.grid(16, 16, 16, 16):
        with Ts.sblock("Apad"):
            nn, cc, yy, xx = Ts.axis.remap("SSSS", [n, c, y, x])
            Apad[nn, cc, yy, xx] = T.if_then_else(
                1 <= yy and yy < 15 and 1 <= xx and xx < 15,
                A[nn, cc, yy - 1, xx - 1],
                0.0,
            )
    for n, f, y, x, kc, ky, kx in T.grid(16, 32, 14, 14, 16, 3, 3):
        with Ts.sblock("B"):
            nn, ff, yy, xx, rc, ry, rx = Ts.axis.remap("SSSSRRR", [n, f, y, x, kc, ky, kx])
            with Ts.init():
                B[nn, ff, yy, xx] = 0.0
            B[nn, ff, yy, xx] += Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff]


def test_conv2d():
    _check_workload(te_conv2d, tir_conv2d)


def te_multi_output():
    n = te.var("n")
    m = te.var("m")
    A0 = te.placeholder((m, n), name="A0")
    A1 = te.placeholder((m, n), name="A1")
    B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] + 2, A1[i, j] * 3), name="B")
    return [A0, A1, B0, B1]


m = T.dynamic("m", "int32")
n = T.dynamic("n", "int32")


@Ts.function
def tir_multi_output(
    A0: T.Tensor((m, n), "float32"),
    A1: T.Tensor((m, n), "float32"),
    B0: T.Tensor((m, n), "float32"),
    B1: T.Tensor((m, n), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    for i0, i1 in T.grid(m, n):
        with Ts.sblock("B.v0"):
            i, j = Ts.axis.remap("SS", [i0, i1])
            B0[i, j] = A0[i, j] + 2.0
        with Ts.sblock("B.v1"):
            i, j = Ts.axis.remap("SS", [i0, i1])
            B1[i, j] = A1[i, j] * 3.0


def test_multi_output():
    _check_workload(te_multi_output, tir_multi_output)


def te_extern():
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.extern(
        (128, 128),
        [A, B],
        lambda ins, outs: tvm_tirx.tirx.call_packed(
            "tvm.contrib.cblas.matmul", ins[0], ins[1], outs[0], 0, 0
        ),
        name="C",
    )
    return [A, B, C]


off1 = T.dynamic("off1", "int32")
off2 = T.dynamic("off2", "int32")
off3 = T.dynamic("off3", "int32")


@Ts.function
def tir_extern(
    A: T.Tensor((128, 128), "float32", layout=T.TileLayout(T.S[(128, 128)] + (off1))),
    B: T.Tensor((128, 128), "float32", layout=T.TileLayout(T.S[(128, 128)] + (off2))),
    C: T.Tensor((128, 128), "float32", layout=T.TileLayout(T.S[(128, 128)] + (off3))),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    # body
    with Ts.sblock("C"):
        Ts.reads()
        Ts.writes()
        T.evaluate(
            T.call_packed(
                "tvm.contrib.cblas.matmul",
                T.stack_make_dltensor(
                    A.data,
                    T.stack_make_shape(128, 128),
                    0,
                    2,
                    0.0,
                    off1,
                ),
                T.stack_make_dltensor(
                    B.data,
                    T.stack_make_shape(128, 128),
                    0,
                    2,
                    0.0,
                    off2,
                ),
                T.stack_make_dltensor(
                    C.data,
                    T.stack_make_shape(128, 128),
                    0,
                    2,
                    0.0,
                    off3,
                ),
                0,
                0,
            )
        )


def test_extern():
    _check_workload(te_extern, tir_extern)


def te_extern_epilogue():
    A = te.placeholder((4, 3), name="A")
    B = te.placeholder((3, 2), name="B")
    C = te.extern(
        (4, 2),
        [A, B],
        lambda ins, outs: tvm_tirx.tirx.call_packed("testing.echo", ins[0], ins[1], outs[0]),
        name="C",
    )
    D = te.compute(C.shape, lambda i, j: C[i, j] + 1.0, name="D")
    return [A, B, D]


tir_extern_epilogue_a_offset = T.dynamic("tir_extern_epilogue_a_offset", "int32")
tir_extern_epilogue_b_offset = T.dynamic("tir_extern_epilogue_b_offset", "int32")


@Ts.function
def tir_extern_epilogue(
    A: T.Tensor(
        (4, 3), "float32", layout=T.TileLayout(T.S[(4, 3)] + (tir_extern_epilogue_a_offset))
    ),
    B: T.Tensor(
        (3, 2), "float32", layout=T.TileLayout(T.S[(3, 2)] + (tir_extern_epilogue_b_offset))
    ),
    D: T.Tensor((4, 2), "float32"),
):
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    C = Ts.sblock_alloc_tensor((4, 2))
    with Ts.sblock("C"):
        Ts.reads()
        Ts.writes()
        T.call_packed("testing.echo", A, B, C)
    for i, j in T.grid(4, 2):
        with Ts.sblock("D"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            Ts.reads(C[vi, vj])
            Ts.writes(D[vi, vj])
            D[vi, vj] = C[vi, vj] + T.float32(1)


def test_extern_epilogue():
    _check_workload(te_extern_epilogue, tir_extern_epilogue)
    func = te.create_function(te_extern_epilogue()).with_attr("global_symbol", "extern_epilogue")
    tvm.compile(func, target="llvm")


def te_reordered_matmul():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    return [C, A, B]


@Ts.function
def tir_reordered_matmul(
    C: T.Tensor((128, 128), "float32"),
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    for i0, j0, k0 in T.grid(128, 128, 128):
        with Ts.sblock():
            i, j, k = Ts.axis.remap("SSR", [i0, j0, k0])
            with Ts.init():
                C[i, j] = 0.0
            C[i, j] += A[i, k] * B[j, k]


def test_arg_order():
    _check_workload(te_reordered_matmul, tir_reordered_matmul)


def te_scan():
    m = te.var("m")
    n = te.var("n")
    X = te.placeholder((m, n), name="X")
    s_state = te.placeholder((m, n))
    s_init = te.compute((1, n), lambda _, i: X[0, i])
    s_update = te.compute((m, n), lambda t, i: s_state[t - 1, i] + X[t, i])
    s_scan = tvm.te.scan(s_init, s_update, s_state, inputs=[X])
    return [X, s_scan]


def test_error_reporting():
    try:
        te.create_function(te_scan())
        assert False
    except (TypeError, RuntimeError) as e:
        error_message = str(e)
        assert error_message.find("Unsupported Operation: te.ScanOp.") != -1
        return
    assert False


def test_constant():
    M = 11
    A = te.placeholder((M,), name="A")
    B = te.compute(tuple(), lambda: 2, name="B")
    C = te.compute((M,), lambda x: A[x] + B(), name="C", tag="broadcast")

    func = te.create_function([C, A])
    func = tvm.compile(func)
    a_np = np.random.uniform(size=(M,)).astype(A.dtype.dtype)
    c = tvm_tirx.runtime.tensor(np.zeros(M, dtype=C.dtype.dtype))
    x = func(c, tvm_tirx.runtime.tensor(a_np))
    tvm_tirx.testing.assert_allclose(a_np + 2, c.numpy())


@pytest.mark.parametrize("op_name", ["acos", "acosh", "asin", "asinh", "atanh"])
def test_topi_float_unary_rejects_integer_input(op_name):
    x = te.placeholder((1, 8), dtype="int16", name="x")
    op = getattr(topi, op_name)

    with pytest.raises(
        TypeError,
        match=rf"topi\.{op_name} only supports floating-point inputs, but got int16",
    ):
        op(x)


@pytest.mark.parametrize("op_name", ["acos", "acosh", "asin", "asinh", "atanh"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_topi_float_unary_accepts_float_input(op_name, dtype):
    x = te.placeholder((1, 8), dtype=dtype, name="x")
    op = getattr(topi, op_name)
    out = op(x)

    func = te.create_function([x, out]).with_attr("target", tvm_tirx.target.Target("llvm"))
    mod = tvm_tirx.IRModule({"main": func})
    compiled = tvm_tirx.tirx.compile(mod, target="llvm").mod

    assert compiled is not None


def test_data_dependent_access():
    A = te.placeholder((10,), name="A")
    B = te.placeholder((10,), name="B", dtype="int32")
    C = te.compute((10,), lambda i: A[B[i]])

    func = te.create_function([C, A, B])
    func = tvm.compile(func)

    a_np = np.random.uniform(size=(10,)).astype(A.dtype.dtype)
    b_np = np.arange(10, dtype=B.dtype.dtype)
    c = tvm_tirx.runtime.tensor(np.zeros(10, dtype=C.dtype.dtype))
    func(c, tvm_tirx.runtime.tensor(a_np), tvm_tirx.runtime.tensor(b_np))
    tvm_tirx.testing.assert_allclose(a_np[b_np], c.numpy())


def test_select_simplify():
    placeholder = te.placeholder([1, 128, 10, 10, 4], dtype="float32")
    tensor = topi.nn.adaptive_pool(placeholder, [1, 1], "avg", "NCHW4c")
    result = te.create_function([placeholder, tensor])
    script_func = result.script()
    # There should be no Select
    assert script_func.find("Select") == -1
    # There should be no undefined vars
    assert script_func.find("Var") == -1


def test_tensor_attr():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute(
        (128, 128),
        lambda x, y: te.sum(A[x, k] * B[y, k], axis=k),
        name="C",
        attrs={"layout_free_placeholders": [B]},
    )
    func = te.create_function([A, B, C])
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "Ts": tvm.s_tir.script},
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)


@Ts.function
def expected_layout_attr(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    D: T.Tensor((128, 128), "float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
    C = Ts.sblock_alloc_tensor([128, 128], dtype="float32")
    for i0, i1, i2 in T.grid(128, 128, 128):
        with Ts.sblock("C"):
            x, y, k = Ts.axis.remap("SSR", [i0, i1, i2])
            with Ts.init():
                C[x, y] = T.float32(0)
            C[x, y] = C[x, y] + A[x, k] * B[y, k]
    for i0, i1 in T.grid(128, 128):
        with Ts.sblock("D"):
            Ts.sblock_attr({"layout_free_placeholders": [C]})
            x, y = Ts.axis.remap("SS", [i0, i1])
            D[x, y] = C[x, y] + T.float32(1)


@Ts.function
def expected_layout_attr_int64(
    A: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    B: T.Tensor((T.int64(128), T.int64(128)), "float32"),
    D: T.Tensor((T.int64(128), T.int64(128)), "float32"),
):
    T.func_attr({"global_symbol": "main", "tirx.noalias": True, "layout_free_tensors": [1]})
    C = Ts.sblock_alloc_tensor([T.int64(128), T.int64(128)], dtype="float32")
    for x, y, k in T.grid(T.int64(128), T.int64(128), T.int64(128)):
        with Ts.sblock("C"):
            v_x, v_y, v_k = Ts.axis.remap("SSR", [x, y, k])
            Ts.reads(A[v_x, v_k], B[v_y, v_k])
            Ts.writes(C[v_x, v_y])
            with Ts.init():
                C[v_x, v_y] = T.float32(0)
            C[v_x, v_y] = C[v_x, v_y] + A[v_x, v_k] * B[v_y, v_k]
    for x, y in T.grid(T.int64(128), T.int64(128)):
        with Ts.sblock("D"):
            Ts.sblock_attr({"layout_free_placeholders": [C]})
            v_x, v_y = Ts.axis.remap("SS", [x, y])
            Ts.reads(C[v_x, v_y])
            Ts.writes(D[v_x, v_y])
            D[v_x, v_y] = C[v_x, v_y] + T.float32(1)


@pytest.mark.parametrize(
    "index_dtype_override, expected",
    [(None, expected_layout_attr), ("int64", expected_layout_attr_int64)],
)
def test_tensor_layout_attr(index_dtype_override, expected):
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute(
        (128, 128),
        lambda x, y: te.sum(A[x, k] * B[y, k], axis=k),
        name="C",
        attrs={"layout_free_placeholders": [B]},
    )
    D = te.compute(
        (128, 128),
        lambda x, y: C[x, y] + 1,
        name="D",
        attrs={"layout_free_placeholders": [C]},
    )
    func = te.create_function([A, B, D], index_dtype_override=index_dtype_override)
    tvm_tirx.ir.assert_structural_equal(func, expected)


def te_argmax_idx_val():
    def f_combine(x, y):
        lhs = tvm_tirx.tirx.Select((x[1] >= y[1]), x[0], y[0])
        rhs = tvm_tirx.tirx.Select((x[1] >= y[1]), x[1], y[1])
        return lhs, rhs

    def f_identity(dtype0: tvm_ffi.dtype, dtype1: tvm_ffi.dtype):
        return tvm_tirx.tirx.const(-1, dtype0), tvm.te.min_value(ty=dtype1)

    argmax = te.comm_reducer(f_combine, f_identity, name="argmax")

    m = te.var("m")
    n = te.var("n")
    idx = te.placeholder((m, n), name="idx", dtype="int32")
    val = te.placeholder((m, n), name="val", dtype="float32")
    k = te.reduce_axis((0, n), "k")
    max_idx, max_val = te.compute(
        (m,), lambda i: argmax((idx[i, k], val[i, k]), axis=k), name="argmax"
    )
    return [idx, val, max_idx, max_val]


m = T.dynamic("m", "int32")
n = T.dynamic("n", "int32")


@Ts.function
def tir_argmax_idx_val(
    idx: T.Tensor([m, n], dtype="int32"),
    val: T.Tensor([m, n], dtype="float32"),
    argmax_v0: T.Tensor([m], dtype="int32"),
    argmax_v1: T.Tensor([m], dtype="float32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    for i0, i1 in T.grid(m, n):
        with Ts.sblock("argmax"):
            i, k = Ts.axis.remap("SR", [i0, i1])
            Ts.reads(val[i, k], idx[i, k])
            Ts.writes(argmax_v0[i], argmax_v1[i])
            with Ts.init():
                argmax_v0[i] = T.int32(-1)
                argmax_v1[i] = T.min_value(ty="float32")
            v_argmax_v0: T.let[T.int32] = T.Select(
                argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k]
            )
            v_argmax_v1: T.let[T.float32] = T.Select(
                argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k]
            )
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


def te_argmax_val_idx():
    def f_combine(x, y):
        lhs = tvm_tirx.tirx.Select((x[0] >= y[0]), x[0], y[0])
        rhs = tvm_tirx.tirx.Select((x[0] >= y[0]), x[1], y[1])
        return lhs, rhs

    def f_identity(dtype0: tvm_ffi.dtype, dtype1: tvm_ffi.dtype):
        return tvm.te.min_value(ty=dtype0), tvm_tirx.tirx.const(-1, dtype1)

    argmax = te.comm_reducer(f_combine, f_identity, name="argmax")

    m = te.var("m")
    n = te.var("n")
    val = te.placeholder((m, n), name="val", dtype="float32")
    idx = te.placeholder((m, n), name="idx", dtype="int32")
    k = te.reduce_axis((0, n), "k")
    max_val, max_idx = te.compute(
        (m,), lambda i: argmax((val[i, k], idx[i, k]), axis=k), name="argmax"
    )
    return [val, idx, max_val, max_idx]


m = T.dynamic("m", "int32")
n = T.dynamic("n", "int32")


@Ts.function
def tir_argmax_val_idx(
    val: T.Tensor([m, n], dtype="float32"),
    idx: T.Tensor([m, n], dtype="int32"),
    argmax_v0: T.Tensor([m], dtype="float32"),
    argmax_v1: T.Tensor([m], dtype="int32"),
) -> None:
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    for i0, i1 in T.grid(m, n):
        with Ts.sblock("argmax"):
            i, k = Ts.axis.remap("SR", [i0, i1])
            Ts.reads(val[i, k], idx[i, k])
            Ts.writes(argmax_v0[i], argmax_v1[i])
            with Ts.init():
                argmax_v0[i] = T.min_value(ty="float32")
                argmax_v1[i] = T.int32(-1)
            v_argmax_v0: T.let[T.float32] = T.Select(
                argmax_v0[i] >= val[i, k], argmax_v0[i], val[i, k]
            )
            v_argmax_v1: T.let[T.int32] = T.Select(
                argmax_v0[i] >= val[i, k], argmax_v1[i], idx[i, k]
            )
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


def test_argmax_idx_val():
    _check_workload(te_argmax_idx_val, tir_argmax_idx_val)


def test_argmax_val_idx():
    _check_workload(te_argmax_val_idx, tir_argmax_val_idx)


def test_int64_indices():
    n = te.var("n", "int64")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1, name="B")
    function = te.create_function([A, B])
    loop = function.body[0].block.body[0]
    assert loop.var.ty.dtype == "int64"
    assert loop.begin.ty.dtype == "int64"
    assert Range.get_extent_from_begin_end(loop.begin, loop.end).ty.dtype == "int64"


def test_zero_dim_add():
    def te_func():
        a = te.placeholder((), name="a", dtype="int32")
        b = te.placeholder((), name="b", dtype="int32")
        c = te.compute(a.shape, lambda *i: a(*i) + b(*i), name="c")
        return [a, b, c]

    @Ts.function
    def expected(
        a: T.Tensor((), "int32"),
        b: T.Tensor((), "int32"),
        c: T.Tensor((), "int32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        with Ts.sblock("root"):
            Ts.reads()
            Ts.writes()
            with Ts.sblock("c"):
                vi = Ts.axis.spatial(1, 0)
                Ts.reads(a[()], b[()])
                Ts.writes(c[()])
                c[()] = a[()] + b[()]

    _check_workload(te_func, expected)


def te_reshape():
    # The following is possible to be generated by TOPI. So we test this case.
    A = te.placeholder((tvm_tirx.ir.IntImm("int64", 2), tvm_tirx.ir.IntImm("int64", 4)), name="A")
    B = topi.reshape(A, (4, 2))
    return [A, B]


@Ts.function
def tir_reshape(
    A: T.Tensor((T.int64(2), T.int64(4)), "float32"),
    T_reshape: T.Tensor((T.int64(4), T.int64(2)), "float32"),
):
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})
    for i0, i1 in T.grid(T.int64(4), T.int64(2)):
        with Ts.sblock("T_reshape"):
            ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
            Ts.reads(
                A[
                    (ax0 * T.int64(2) + ax1) % T.int64(8) // T.int64(4),
                    (ax0 * T.int64(2) + ax1) % T.int64(4),
                ]
            )
            Ts.writes(T_reshape[ax0, ax1])
            T_reshape[ax0, ax1] = A[
                (ax0 * T.int64(2) + ax1) % T.int64(8) // T.int64(4),
                (ax0 * T.int64(2) + ax1) % T.int64(4),
            ]


def test_reshape():
    _check_workload(te_reshape, tir_reshape, index_dtype_override="int64")


def te_resize2d_symbolic():
    oh = tirx.Var("oh", "int64")
    ow = tirx.Var("ow", "int64")
    roi = (0.0, 0.0, 0.0, 0.0)
    A = te.placeholder((2, 3, 128, 128), "float32", name="A")
    B = topi.image.resize2d(
        A,
        roi,
        size=(oh, ow),
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="round",
    )
    return [A, B]


oh = T.dynamic("oh", "int64")
ow = T.dynamic("ow", "int64")


@Ts.function
def tir_resize2d_symbolic(
    A: T.Tensor((T.int64(2), T.int64(3), T.int64(128), T.int64(128)), "float32"),
    resize: T.Tensor([T.int64(2), T.int64(3), oh, ow], dtype="float32"),
):
    T.func_attr({"global_symbol": "main", "tirx.noalias": True})

    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), oh, ow):
        with Ts.sblock("resize"):
            v_i0, v_i1, v_i2, v_i3 = Ts.axis.remap("SSSS", [i0, i1, i2, i3])
            Ts.reads(A[v_i0, v_i1, T.int64(0) : T.int64(128), T.int64(0) : T.int64(128)])
            Ts.writes(resize[v_i0, v_i1, v_i2, v_i3])
            resize[v_i0, v_i1, v_i2, v_i3] = A[
                v_i0,
                v_i1,
                T.max(
                    T.min(
                        T.Cast(
                            "int64",
                            T.round(
                                T.float32(128) / T.Cast("float32", oh) * T.Cast("float32", v_i2),
                            ),
                        ),
                        T.int64(127),
                    ),
                    T.int64(0),
                ),
                T.max(
                    T.min(
                        T.Cast(
                            "int64",
                            T.round(
                                T.float32(128) / T.Cast("float32", ow) * T.Cast("float32", v_i3),
                            ),
                        ),
                        T.int64(127),
                    ),
                    T.int64(0),
                ),
            ]


def test_resize2d_symbolic():
    _check_workload(te_resize2d_symbolic, tir_resize2d_symbolic, index_dtype_override="int64")


def test_extern_with_explicit_buffer_access():
    def te_extern():
        A = te.placeholder((128, 128), name="A")
        B = te.placeholder((128, 128), name="B")
        P = te.placeholder((1,), name="P")
        C = te.extern(
            (128, 128),
            [A, B, P],
            lambda ins, outs: tvm_tirx.tirx.call_extern(
                "myfunc", ins[0].data, ins[1].data, outs[0].data, ins[2][0], ty=""
            ),
            name="C",
        )
        return [A, B, P, C]

    tir_extern_a_offset = T.dynamic("tir_extern_a_offset", "int32")
    tir_extern_b_offset = T.dynamic("tir_extern_b_offset", "int32")
    tir_extern_p_offset = T.dynamic("tir_extern_p_offset", "int32")
    tir_extern_c_offset = T.dynamic("tir_extern_c_offset", "int32")

    @Ts.function
    def tir_extern(
        A: T.Tensor(
            [128, 128],
            dtype="float32",
            layout=T.TileLayout(T.S[[128, 128]] + (tir_extern_a_offset)),
        ),
        B: T.Tensor(
            [128, 128],
            dtype="float32",
            layout=T.TileLayout(T.S[[128, 128]] + (tir_extern_b_offset)),
        ),
        P: T.Tensor([1], dtype="float32", layout=T.TileLayout(T.S[[1]] + (tir_extern_p_offset))),
        C: T.Tensor(
            [128, 128],
            dtype="float32",
            layout=T.TileLayout(T.S[[128, 128]] + (tir_extern_c_offset)),
        ),
    ):
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})

        with Ts.sblock("C"):
            Ts.reads()
            Ts.writes()
            T.call_extern("myfunc", A.data, B.data, C.data, P[0], ty="")

    _check_workload(te_extern, tir_extern)


def te_slice_with_var_input():
    idx = te.var("idx", dtype="int64")
    m = te.var("m", dtype="int64")
    n = te.var("n", dtype="int64")
    tensor = te.placeholder((m, n), name="tensor")
    slice0 = te.compute((idx, n), lambda i, j: tensor[i, j], name="slice")
    return [tensor, idx, slice0]


m = T.dynamic("m", "int64")
n = T.dynamic("n", "int64")


@Ts.function
def tir_slice_with_var_input(
    tensor: T.Tensor((m, n), "float32"),
    idx: T.int64,
    slice: T.Tensor((idx, n), "float32"),  # noqa: F821
):
    T.func_attr({"tirx.noalias": True, "global_symbol": "main"})

    # with Ts.sblock("root"):
    for i, j in T.grid(idx, n):
        with Ts.sblock("slice"):
            v_i = Ts.axis.spatial(idx, i)
            v_j = Ts.axis.spatial(n, j)
            Ts.reads(tensor[v_i, v_j])
            Ts.writes(slice[v_i, v_j])
            slice[v_i, v_j] = tensor[v_i, v_j]


def test_with_var_input():
    _check_workload(te_slice_with_var_input, tir_slice_with_var_input, index_dtype_override="int64")


def test_loop_aware_initial_value():
    """Test initial value aware of spatial iter position"""

    @Ts.function
    def tir_workload(
        a: T.Tensor((5, 5), "float32"),
        b: T.Tensor((5,), "float32"),
        sum_red: T.Tensor((5,), "float32"),
    ):
        T.func_attr({"tirx.noalias": True, "global_symbol": "main"})

        for i, ax in T.grid(5, 5):
            with Ts.sblock("sum_red"):
                v_i, v_ax = Ts.axis.remap("SR", [i, ax])
                Ts.reads(b[v_i], a[v_i, v_ax])
                Ts.writes(sum_red[v_i])
                with Ts.init():
                    sum_red[v_i] = b[v_i]
                sum_red[v_i] = sum_red[v_i] + a[v_i, v_ax]

    def te_workload():
        data = te.placeholder((5, 5), "float32", "a")
        init = te.placeholder((5,), "float32", "b")
        ax = te.reduce_axis((0, 5), "ax")
        sum_red = te.compute(
            (5,),
            lambda i: te.comm_reducer(
                lambda x, y: x + y,
                lambda t: init[i],
            )(data[i, ax], axis=[ax]),
            name="sum_red",
        )
        return [data, init, sum_red]

    _check_workload(te_workload, tir_workload)


def test_loop_aware_reducer_combiner():
    """Test combiner aware of spatial iter position"""

    @Ts.function
    def tir_workload(
        a: T.Tensor((5, 5), "float32"),
        b: T.Tensor((5,), "float32"),
        sum_red: T.Tensor((5,), "float32"),
    ):
        T.func_attr({"tirx.noalias": True, "global_symbol": "main"})

        for i, ax in T.grid(5, 5):
            with Ts.sblock("sum_red"):
                v_i = Ts.axis.spatial(5, i)
                v_ax = Ts.axis.reduce(5, ax)
                Ts.reads(a[v_i, 0:5])
                Ts.writes(sum_red[v_i])
                with Ts.init():
                    sum_red[v_i] = T.float32(0.0)
                sum_red[v_i] = T.if_then_else(
                    a[v_i, sum_red[v_i]] < a[v_i, v_ax], sum_red[v_i], T.Cast("float32", v_ax)
                )

    def te_workload():
        data = te.placeholder((5, 5), "float32", "a")
        init = te.placeholder((5,), "float32", "b")
        ax = te.reduce_axis((0, 5), "ax")
        sum_red = te.compute(
            (5,),
            lambda i: te.comm_reducer(
                lambda x, y: te.if_then_else(data[i, x] < y, x, ax),
                lambda _: te.const(0, "float32"),
            )(data[i, ax], axis=[ax]),
            name="sum_red",
        )
        return [data, init, sum_red]

    _check_workload(te_workload, tir_workload)


def test_adaptive_pooling_window():
    @Ts.function
    def tir_workload(
        x: T.Tensor((1, 1024, 16, 40), "float32"),
        adaptive_pool_avg: T.Tensor((1, 1024, 12, 30), "float32"),
    ):
        T.func_attr({"tirx.noalias": True, "global_symbol": "main"})
        # fmt: off
        adaptive_pool_sum = Ts.sblock_alloc_tensor((1, 1024, 12, 30))
        for ax0, ax1, ax2, ax3 in T.grid(1, 1024, 12, 30):
            with Ts.sblock("adaptive_pool_sum_l1"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(x[v_ax0, v_ax1, v_ax2 * 16 // 12:v_ax2 * 16 // 12 + ((v_ax2 % 3 * 4 + 16) // 12 + 1), v_ax3 * 40 // 30:v_ax3 * 40 // 30 + ((v_ax3 % 3 * 10 + 40) // 30 + 1)])
                Ts.writes(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                for rv0, rv1 in T.grid(T.Select((v_ax2 * 16 + 4) % 12 == 0, (v_ax2 * 16 + 16) // 12, (v_ax2 * 16 + 16) // 12 + 1) - v_ax2 * 16 // 12, T.Select((v_ax3 * 40 + 10) % 30 == 0, (v_ax3 * 40 + 40) // 30, (v_ax3 * 40 + 40) // 30 + 1) - v_ax3 * 40 // 30):
                    with Ts.sblock("adaptive_pool_sum"):
                        v_ax0_1 = Ts.axis.spatial((v_ax0, v_ax0 + 1), v_ax0)
                        v_ax1_1 = Ts.axis.spatial((v_ax1, v_ax1 + 1), v_ax1)
                        v_ax2_1 = Ts.axis.spatial((v_ax2, v_ax2 + 1), v_ax2)
                        v_ax3_1 = Ts.axis.spatial((v_ax3, v_ax3 + 1), v_ax3)
                        v_rv0, v_rv1 = Ts.axis.remap("RR", [rv0, rv1])
                        Ts.reads(x[v_ax0_1, v_ax1_1, v_ax2_1 * 16 // 12 + v_rv0, v_ax3_1 * 40 // 30 + v_rv1])
                        Ts.writes(adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1])
                        with Ts.init():
                            adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1] = T.float32(0.0)
                        adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1] = adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1] + x[v_ax0_1, v_ax1_1, v_ax2_1 * 16 // 12 + v_rv0, v_ax3_1 * 40 // 30 + v_rv1]
        for ax0, ax1, ax2, ax3 in T.grid(1, 1024, 12, 30):
            with Ts.sblock("adaptive_pool_avg"):
                v_ax0, v_ax1, v_ax2, v_ax3 = Ts.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                Ts.reads(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                Ts.writes(adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                Ts.sblock_attr({"schedule_rule": "meta_schedule.adaptive_pool_avg"})
                adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] / (T.Cast("float32", T.Select((v_ax2 * 16 + 4) % 12 == 0, (v_ax2 * 16 + 16) // 12, (v_ax2 * 16 + 16) // 12 + 1) - v_ax2 * 16 // 12) * T.Cast("float32", T.Select((v_ax3 * 40 + 10) % 30 == 0, (v_ax3 * 40 + 40) // 30, (v_ax3 * 40 + 40) // 30 + 1) - v_ax3 * 40 // 30))
        # fmt: on

    def te_workload():
        x = te.placeholder([1, 1024, 16, 40], "float32", "x")
        y = topi.nn.adaptive_pool(x, [12, 30], pool_type="avg")
        f = te.create_function([x, y])
        return [x, y]

    _check_workload(te_workload, tir_workload)


@pytest.mark.parametrize(
    ("input_shape", "expected"),
    [
        ((3, 4), [[12.5, 14.5], [16.5, 18.5]]),
        ((4, 3), [[12.0, 13.0], [18.0, 19.0]]),
    ],
)
def test_adaptive_pooling_mixed_reduction_levels(input_shape, expected):
    data = te.placeholder((1, 1, *input_shape), "float32", "data")
    output = topi.nn.adaptive_pool(data, [2, 2], pool_type="avg")
    function = te.create_function([data, output])
    compiled = tvm.compile(function)

    input_data = np.arange(10, 10 + np.prod(input_shape), dtype="float32").reshape(
        1, 1, *input_shape
    )
    actual = tvm_tirx.runtime.tensor(np.empty((1, 1, 2, 2), dtype="float32"))
    compiled(tvm_tirx.runtime.tensor(input_data), actual)

    tvm_tirx.testing.assert_allclose(actual.numpy()[0, 0], np.array(expected, dtype="float32"))


def test_global_pool():
    # fix the issue-17938
    data = te.placeholder((1, 1, 32, 32), dtype="int8", name="data")
    op_output = topi.nn.global_pool(data=data, pool_type="avg", layout="NCHW")
    f = te.create_function([data, op_output])
    assert f


def test_nested_reduce_domain_dependency():
    @Ts.function
    def tir_workload(
        x: T.Tensor((8, 8, 8, 8, 8), "float32"), compute: T.Tensor((8, 8, 8), "float32")
    ):
        T.func_attr({"tirx.noalias": True, "global_symbol": "main"})
        for i0, i1, i2 in T.grid(8, 8, 8):
            with Ts.sblock("compute_2"):
                v_i0, v_i1, v_i2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(x[v_i0, v_i1, v_i2, 0:v_i1, 0 : v_i1 - 1])
                Ts.writes(compute[v_i0, v_i1, v_i2])
                for rv in range(v_i1):
                    with Ts.sblock("compute_1"):
                        v_i0_1 = Ts.axis.spatial((v_i0, v_i0 + 1), v_i0)
                        v_i1_1 = Ts.axis.spatial((v_i1, v_i1 + 1), v_i1)
                        v_i2_1 = Ts.axis.spatial((v_i2, v_i2 + 1), v_i2)
                        v_rv = Ts.axis.reduce(v_i1, rv)
                        Ts.reads(x[v_i0_1, v_i1_1, v_i2_1, v_rv, 0:v_rv])
                        Ts.writes(compute[v_i0_1, v_i1_1, v_i2_1])
                        with Ts.init():
                            compute[v_i0_1, v_i1_1, v_i2_1] = T.float32(0.0)
                        for rv_1 in range(v_rv):
                            with Ts.sblock("compute"):
                                v_i0_2 = Ts.axis.spatial((v_i0_1, v_i0_1 + 1), v_i0_1)
                                v_i1_2 = Ts.axis.spatial((v_i1_1, v_i1_1 + 1), v_i1_1)
                                v_i2_2 = Ts.axis.spatial((v_i2_1, v_i2_1 + 1), v_i2_1)
                                v_rv_1 = Ts.axis.reduce((v_rv, v_rv + 1), v_rv)
                                v_rv_2 = Ts.axis.reduce(v_rv, rv_1)
                                Ts.reads(
                                    compute[v_i0_2, v_i1_2, v_i2_2],
                                    x[v_i0_2, v_i1_2, v_i2_2, v_rv_1, v_rv_2],
                                )
                                Ts.writes(compute[v_i0_2, v_i1_2, v_i2_2])
                                compute[v_i0_2, v_i1_2, v_i2_2] = (
                                    compute[v_i0_2, v_i1_2, v_i2_2]
                                    + x[v_i0_2, v_i1_2, v_i2_2, v_rv_1, v_rv_2]
                                )

    def te_workload():
        x = te.placeholder([8, 8, 8, 8, 8], "float32", "x")

        def fcompute(*axes):
            r1 = te.reduce_axis(tvm_tirx.ir.Range.from_min_extent(0, axes[1]))
            r2 = te.reduce_axis(tvm_tirx.ir.Range.from_min_extent(0, r1))
            all_axes = [*axes, r1, r2]
            return te.sum(x(*all_axes), [r1, r2])

        y = te.compute([8, 8, 8], fcompute)
        f = te.create_function([x, y])
        return [x, y]

    _check_workload(te_workload, tir_workload)


if __name__ == "__main__":
    tvm_tirx.testing.main()
