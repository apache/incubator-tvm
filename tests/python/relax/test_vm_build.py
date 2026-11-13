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
# ruff: noqa: F841

from __future__ import annotations

import ctypes
from collections.abc import Callable

import numpy as np
import pytest
import tvm_ffi
from tvm_ffi import Shape

import tvm
import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm import relax, rpc, te, topi
from tvm.relax import ExternFunc as _ExternFunc
from tvm.relax import script as R
from tvm.relax.testing import nn
from tvm.relax.testing.vm import check_saved_func
from tvm.s_tir import script as Ts
from tvm_tirx import tirx
from tvm_tirx.ir import TupleType as _TupleType
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import cc, popen_pool, utils
from tvm_tirx.testing import env


def test_vm_compile_simple():
    @tvm_tirx.script.ir_module
    class TestVMCompileStage0:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            z = R.call_pure_packed(
                "test.vm.identity", x, y, ty_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            return y

    mod = TestVMCompileStage0
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    inp1 = tvm_tirx.runtime.tensor(np.random.rand(3, 4).astype(np.float32))
    inp2 = tvm_tirx.runtime.tensor(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm["foo"](inp1, inp2)
    tvm_tirx.testing.assert_allclose(inp2.numpy(), inp1.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_compile_without_target_arg():
    """Like test_vm_compile_simple, but with a default target"""

    @tvm_tirx.script.ir_module
    class mod:
        @R.function
        def foo(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            z = R.call_pure_packed(
                "test.vm.identity", x, y, ty_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            return y

    ex = relax.build(mod)
    inp1 = tvm_tirx.runtime.tensor(np.random.rand(3, 4).astype(np.float32))
    inp2 = tvm_tirx.runtime.tensor(np.random.rand(3, 4).astype(np.float32))
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    vm["foo"](inp1, inp2)
    tvm_tirx.testing.assert_allclose(inp2.numpy(), inp1.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_compile_unlowered_operator_error():
    @tvm_tirx.script.ir_module
    class Conv2dTranspose:
        @R.function
        def main(
            x: R.Tensor((1, 4, 4, 1), "float32"),
            w: R.Tensor((1, 2, 3, 3), "float32"),
        ):
            gv = R.nn.conv2d_transpose(
                x,
                w,
                data_layout="NHWC",
                kernel_layout="IOHW",
            )
            return gv

    mod = relax.transform.LegalizeOps()(Conv2dTranspose)
    with pytest.raises(
        RuntimeError,
        match=r"(?s)Offending call:.*(?:R|relax)\.nn\.conv2d_transpose.*data_layout.*NHWC",
    ):
        relax.build(mod, target="llvm")


def test_match_check():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class TestMatchCheck:
        @R.function
        def foo(x: R.Tensor([n, m], "int32"), y: R.Any) -> R.Tensor([m, n], dtype=None):
            return y

    mod = TestMatchCheck
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    x0 = tvm_tirx.runtime.tensor(np.zeros((1, 2)).astype("int32"))
    y0 = tvm_tirx.runtime.tensor(np.zeros((2, 1)).astype("float32"))
    y1 = tvm_tirx.runtime.tensor(np.zeros((1, 2)).astype("float32"))
    y2 = tvm_tirx.runtime.tensor(np.zeros((2, 1, 1)).astype("float32"))

    vm["foo"](x0, y0)

    with pytest.raises(RuntimeError, match=".*return.*"):
        vm["foo"](x0, y1)

    with pytest.raises(ValueError, match=".*return.*"):
        vm["foo"](x0, y2)


def test_vm_compile_stage2():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class TestVMCompileStage2:
        @R.function
        def foo(x: R.Tensor(dtype="float32")) -> R.Shape:
            _ = R.match_cast(x, R.Tensor((n, m), "float32"))
            return R.shape([n * 2, m * 3])

    mod = TestVMCompileStage2
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    shape = (32, 16)
    arr = tvm_tirx.runtime.tensor(np.random.rand(*shape).astype("float32"))
    res = vm["foo"](arr)
    assert res[0] == shape[0] * 2
    assert res[1] == shape[1] * 3

    # dtype mismatch
    with pytest.raises(ValueError, match=".*dtype.*"):
        vm["foo"](tvm_tirx.runtime.tensor(np.zeros((1, 2)).astype("int32")))

    # ndim mismatch
    with pytest.raises(ValueError, match=".*match_cast.*ndim.*"):
        vm["foo"](tvm_tirx.runtime.tensor(np.zeros((1,)).astype("float32")))

    # type mismach
    with pytest.raises(TypeError):
        vm["foo"]([])


def test_vm_compile_stage3():
    @tvm_tirx.script.ir_module
    class TestVMCompileStage3:
        @R.function
        def foo(x: R.Tensor((32, 16), "float32")) -> R.Tensor:
            with R.dataflow():
                y = R.call_dps_packed(
                    _ExternFunc("test.vm.identity"),
                    (x),
                    ty_args=[R.Tensor((32, 16), dtype="float32")],
                )
                R.output(y)
            return y

    mod = TestVMCompileStage3
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    shape = (32, 16)
    inp = tvm_tirx.runtime.tensor(np.random.rand(*shape).astype(np.float32))
    res = vm["foo"](inp)
    tvm_tirx.testing.assert_allclose(res.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_compile_e2e():
    n = T.dynamic("n", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class TestVMCompileE2E:
        @R.function
        def foo(x: R.Tensor(dtype="float32")) -> R.Tensor:
            with R.dataflow():
                _ = R.match_cast(x, R.Tensor((n, m), "float32"))
                y = R.call_dps_packed(
                    _ExternFunc("test.vm.tile"),
                    (x),
                    ty_args=[R.Tensor((n, m * 2), dtype="float32")],
                )
                R.output(y)
            return y

    mod = TestVMCompileE2E

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    shape = (32, 16)
    inp = tvm_tirx.runtime.tensor(np.random.rand(*shape).astype(np.float32))
    res = check_saved_func(vm, "foo", inp)
    tvm_tirx.testing.assert_allclose(
        res.numpy(), np.tile(inp.numpy(), (1, 2)), rtol=1e-7, atol=1e-7
    )


def test_vm_compile_e2e_func_param_with_shape():
    m_tir_matmul = T.dynamic("m", "int32")
    n_tir_matmul = T.dynamic("n", "int32")
    k_tir_matmul = T.dynamic("k", "int32")
    m_func = T.dynamic("m", "int64")
    k_func = T.dynamic("k", "int64")
    n_func = T.dynamic("n", "int64")

    @tvm_tirx.script.ir_module
    class TestVMCompileE2E2:
        @Ts.function
        def tir_matmul(
            A: T.Tensor((m_tir_matmul, n_tir_matmul), "float32"),
            B: T.Tensor((n_tir_matmul, k_tir_matmul), "float32"),
            C: T.Tensor((m_tir_matmul, k_tir_matmul), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})

            for i, j, k_tir_matmul_index in T.grid(m_tir_matmul, k_tir_matmul, n_tir_matmul):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k_tir_matmul_index])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def func(
            x: R.Tensor((m_func, n_func), "float32"), w: R.Tensor((n_func, k_func), "float32")
        ) -> R.Tensor:
            cls = TestVMCompileE2E2
            gv0 = R.call_tir(
                cls.tir_matmul, (x, w), ty_args=[R.Tensor((m_func, k_func), dtype="float32")]
            )
            return gv0

    mod = TestVMCompileE2E2

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    data = tvm_tirx.runtime.tensor(np.random.rand(32, 16).astype(np.float32))
    weight = tvm_tirx.runtime.tensor(np.random.rand(16, 32).astype(np.float32))
    res = check_saved_func(vm, "func", data, weight)
    expected = np.dot(data.numpy(), weight.numpy())
    tvm_tirx.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_call_tir_inplace_e2e_simple():
    @tvm_tirx.script.ir_module
    class TestCallTIRInplaceE2ESimple:
        @Ts.function
        def copy(
            A: T.Tensor((2, 3), "int32"),
            B: T.Tensor((2, 3), "int32"),
            C: T.Tensor((2, 3), "int32"),
            out1: T.Tensor((2, 3), "int32"),
        ):
            # copies the contents of C into A, B, and out1
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_zeros"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(C[ax0, ax1])
                    Ts.writes(A[ax0, ax1], B[ax0, ax1], out1[ax0, ax1])
                    A[ax0, ax1] = C[ax0, ax1]
                    B[ax0, ax1] = C[ax0, ax1]
                    out1[ax0, ax1] = C[ax0, ax1]

        @R.function
        def main(
            x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32"), z: R.Tensor((2, 3), "int32")
        ) -> R.Tuple(
            R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32"), R.Tensor((2, 3), "int32")
        ):
            res = R.call_tir_inplace(
                TestCallTIRInplaceE2ESimple.copy,
                (x, y, z),
                inplace_indices=[0, 1, -1],
                ty_args=[
                    _TupleType(
                        [
                            R.Tensor((2, 3), "int32"),
                            R.Tensor((2, 3), "int32"),
                            R.Tensor((2, 3), "int32"),
                        ]
                    )
                ],
            )
            return res

    mod = TestCallTIRInplaceE2ESimple

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    x = tvm_tirx.runtime.tensor(np.zeros((2, 3)).astype(np.int32))
    y = tvm_tirx.runtime.tensor(np.zeros((2, 3)).astype(np.int32))
    z = tvm_tirx.runtime.tensor(np.ones((2, 3)).astype(np.int32))
    vm.set_input("main", x, y, z)
    vm.invoke_stateful("main")
    outs = vm.get_outputs("main")
    # check the expected aliasing (the last result is newly allocated)
    assert x == outs[0]
    assert y == outs[1]
    assert x != y
    assert x != outs[2]
    assert y != outs[2]
    tvm_tirx.testing.assert_allclose(x.numpy(), z.numpy(), rtol=1e-7, atol=1e-7)
    tvm_tirx.testing.assert_allclose(y.numpy(), z.numpy(), rtol=1e-7, atol=1e-7)
    tvm_tirx.testing.assert_allclose(outs[2].numpy(), z.numpy(), rtol=1e-7, atol=1e-7)


def test_call_tir_inplace_e2e_rw():
    # read and write from the same tensor
    @tvm_tirx.script.ir_module
    class TestCallTIRInplaceE2ERW:
        @Ts.function
        def inplace_add(A: T.Tensor((2, 3), "int32"), B: T.Tensor((2, 3), "int32")):
            # sums A and B, storing the result in A
            T.func_attr({"tirx.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with Ts.sblock("T_add"):
                    ax0, ax1 = Ts.axis.remap("SS", [i0, i1])
                    Ts.reads(A[ax0, ax1], B[ax0, ax1])
                    Ts.writes(A[ax0, ax1])
                    A[ax0, ax1] = A[ax0, ax1] + B[ax0, ax1]

        @R.function
        def main(x: R.Tensor((2, 3), "int32"), y: R.Tensor((2, 3), "int32")) -> R.Tensor(
            (2, 3), "int32"
        ):
            res = R.call_tir_inplace(
                TestCallTIRInplaceE2ERW.inplace_add,
                (x, y),
                inplace_indices=[0],
                ty_args=[R.Tensor((2, 3), "int32")],
            )
            return res

    mod = TestCallTIRInplaceE2ERW

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    x = tvm_tirx.runtime.tensor(np.ones((2, 3)).astype(np.int32))
    y = tvm_tirx.runtime.tensor(np.ones((2, 3)).astype(np.int32))
    vm.set_input("main", x, y)
    vm.invoke_stateful("main")
    out = vm.get_outputs("main")
    expected = tvm_tirx.runtime.tensor(np.full((2, 3), 2).astype(np.int32))

    assert x == out
    tvm_tirx.testing.assert_allclose(out.numpy(), expected.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_emit_te_extern():
    if not tvm_ffi.get_global_func("tvm.contrib.cblas.matmul", True):
        print("skip because extern function is not available")
        return
    bb = relax.BlockBuilder()
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = relax.Var("x", R.Tensor([n, m], "float32"))
    y = relax.Var("y", R.Tensor([m, n], "float32"))

    with bb.function("rx_cblas_matmul", [x, y]):
        out = bb.emit_te(tvm.contrib.cblas.matmul, x, y, transa=False, transb=False)
        bb.emit_func_output(out)

    mod = bb.get()

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    data = tvm_tirx.runtime.tensor(np.random.rand(16, 32).astype(np.float32))
    weight = tvm_tirx.runtime.tensor(np.random.rand(32, 16).astype(np.float32))
    res = check_saved_func(vm, "rx_cblas_matmul", data, weight)
    expected = np.dot(data.numpy(), weight.numpy())
    tvm_tirx.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_vm_emit_te_concat():
    # concatenate of two vectors of size (n,) and (m,)
    bb = relax.BlockBuilder()
    n, m = tirx.Var("n", "int64"), tirx.Var("m", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))
    y = relax.Var("y", R.Tensor([m], "float32"))

    def te_func(A, B):
        C = te.compute((n + m), lambda i: tvm_tirx.tirx.if_then_else(i < n, A[i], B[i - n]))
        return C

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x, y)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    inp = tvm_tirx.runtime.tensor(
        np.random.rand(
            1,
        ).astype(np.float32)
    )
    inp2 = tvm_tirx.runtime.tensor(
        np.random.rand(
            2,
        ).astype(np.float32)
    )
    res = check_saved_func(vm, "rx_func", inp, inp2)
    tvm_tirx.testing.assert_allclose(
        res.numpy(), np.append(inp.numpy(), inp2.numpy()), rtol=1e-7, atol=1e-7
    )


def test_vm_emit_te_dtype_change():
    bb = relax.BlockBuilder()
    n = T.dynamic("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))

    # convert a tensor with dtype of float32 to int16
    def te_func(A):
        B = te.compute((n,), lambda i: A[i].astype("int16"))
        return B

    with bb.function("rx_func", [x]):
        y = bb.emit_te(te_func, x)
        bb.emit_func_output(y)

    mod = bb.get()

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    inp = tvm_tirx.runtime.tensor(
        np.random.rand(
            1,
        ).astype(np.float32)
    )
    res = check_saved_func(vm, "rx_func", inp)
    tvm_tirx.testing.assert_allclose(res.numpy(), inp.numpy().astype("int16"))


def test_vm_emit_te_floor_symbolic_shape():
    bb = relax.BlockBuilder()
    n = T.dynamic("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))

    def te_func(A):
        C = te.compute((tirx.floordiv(n, 2),), lambda i: A[i] + 1)
        return C

    with bb.function("rx_func", [x]):
        x1 = bb.emit_te(te_func, x)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    shape = (9,)
    inp = tvm_tirx.runtime.tensor(np.random.rand(*shape).astype(np.float32))
    res = check_saved_func(vm, "rx_func", inp)

    def expected_output():
        output_shape = (shape[0] // 2,)
        return inp.numpy()[: output_shape[0]] + 1

    tvm_tirx.testing.assert_allclose(res.numpy(), expected_output(), rtol=1e-7, atol=1e-7)


def test_vm_emit_te_constant_param_cpu():
    x_np = np.random.rand(2, 2).astype("float32")
    c_np = np.random.rand(2, 2).astype("float32")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 2), "float32"))
    c = relax.const(c_np, "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.add, x, c)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    mod = bb.get()
    exec = relax.build(mod, "llvm")
    dev = tvm_tirx.cpu()
    vm = relax.VirtualMachine(exec, dev)

    add_res = check_saved_func(vm, "main", tvm_tirx.runtime.tensor(x_np, dev))
    tvm_tirx.testing.assert_allclose(add_res.numpy(), x_np + c_np, rtol=1e-7, atol=1e-7)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_vm_emit_te_constant_param_gpu():
    x_np = np.random.rand(2, 2).astype("float32")
    c_np = np.random.rand(2, 2).astype("float32")

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 2), "float32"))
    c = relax.const(c_np, "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.add, x, c)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    mod = bb.get()
    sch = tvm.s_tir.Schedule(mod, debug_mask="all")
    loops = sch.get_loops(sch.get_sblock(name="T_add", func_name="add"))
    sch.bind(loops[0], "threadIdx.x")

    exec = relax.build(sch.mod, "cuda")

    def run_and_check():
        dev = tvm_tirx.cuda()
        vm = relax.VirtualMachine(exec, dev)
        add_res = check_saved_func(vm, "main", tvm_tirx.runtime.tensor(x_np, dev))
        tvm_tirx.testing.assert_allclose(add_res.numpy(), x_np + c_np, rtol=1e-7, atol=1e-7)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_vm_relax_symbolic_shape():
    bb = relax.BlockBuilder()
    n = T.dynamic("n", "int64")
    x = relax.Var("x", R.Tensor([n], "float32"))
    y = relax.Var("y", R.Tensor([(n // 2) + 1], "float32"))

    def te_func(A, B):
        C = te.compute((n,), lambda i: A[i] + B[i // 2])
        return C

    with bb.function("rx_func", [x, y]):
        x1 = bb.emit_te(te_func, x, y)
        bb.emit_func_output(x1)

    mod = bb.get()

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    shape1 = (5,)
    shape2 = (3,)
    inp = tvm_tirx.runtime.tensor(np.random.rand(*shape1).astype(np.float32))
    inp2 = tvm_tirx.runtime.tensor(np.random.rand(*shape2).astype(np.float32))
    res = check_saved_func(vm, "rx_func", inp, inp2)

    def expected_output():
        return inp.numpy() + np.repeat(inp2.numpy(), 2)[:5]

    tvm_tirx.testing.assert_allclose(res.numpy(), expected_output(), rtol=1e-7, atol=1e-7)


def test_vm_relax_symbolic_shape_tuple():
    m = T.dynamic("m", "int64")
    n = T.dynamic("n", "int64")

    @I.ir_module
    class mod:
        @R.function
        def main(shape: R.Shape([m, n])):
            return R.shape([2 * m, 3 * n])

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    func = vm["main"]

    assert func(Shape([2, 3])) == (4, 9)

    with pytest.raises(ValueError):
        func(Shape([2, 3, 4]))

    with pytest.raises(TypeError):
        func(R.prim_value(2))


def test_vm_relax_dyn_tir_shape():
    # case where TIR variables are unbound in generated Function
    bb = relax.BlockBuilder()
    n = T.dynamic("n", "int64")

    def te_func(A):
        C = te.compute((n + 1), lambda i: A[i])
        return C

    with bb.function("rx_func"):
        x = nn.Placeholder((n,), dtype="float32", name="x")
        y = nn.Placeholder((n + 1,), dtype="float32", name="y")

        x1 = bb.emit_te(te_func, y)
        bb.emit_func_output(x1, params=[x, y])

    mod = bb.get()

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)

    with utils.tempdir() as temp:
        ex.export_library(temp.relpath("exec.so"))
        vm = relax.VirtualMachine(
            tvm_tirx.runtime.load_module(temp.relpath("exec.so")), tvm_tirx.cpu()
        )

    inp = tvm_tirx.runtime.tensor(np.random.rand(2).astype(np.float32))
    inp2 = tvm_tirx.runtime.tensor(np.random.rand(3).astype(np.float32))

    res = check_saved_func(vm, "rx_func", inp, inp2)

    tvm_tirx.testing.assert_allclose(res.numpy(), inp2.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_tuple():
    bb = relax.BlockBuilder()
    n = T.dynamic("n", "int64")

    with bb.function("rx_func"):
        x = nn.Placeholder((n,), dtype="float32", name="x")
        y = nn.Placeholder((n,), dtype="float32", name="y")
        tup = relax.Tuple([x, y])
        item = tup[0]
        bb.emit_func_output([tup, item], params=[x, y])

    mod = bb.get()

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)

    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    shape = (5,)
    inp = tvm_tirx.runtime.tensor(np.random.rand(*shape).astype(np.float32))
    inp2 = tvm_tirx.runtime.tensor(np.random.rand(*shape).astype(np.float32))
    (res1, res2), res3 = vm["rx_func"](inp, inp2)

    tvm_tirx.testing.assert_allclose(res1.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)
    tvm_tirx.testing.assert_allclose(res2.numpy(), inp2.numpy(), rtol=1e-7, atol=1e-7)
    tvm_tirx.testing.assert_allclose(res3.numpy(), inp.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_tuplegetitem():
    @tvm_tirx.script.ir_module
    class TestVMTupleGetItem:
        @R.function
        def tuple_get_item(
            x: R.Tensor(ndim=2, dtype="float32"),
            y: R.Tensor(ndim=2, dtype="float32"),
        ):
            t = (x, y)
            a = t[0]
            b = t[1]
            c = R.call_pure_packed("test.vm.add", a, b, ty_args=(R.Tensor(ndim=2, dtype="float32")))
            return c

    mod = TestVMTupleGetItem
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    x_inp = tvm_tirx.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    y_inp = tvm_tirx.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    res = check_saved_func(vm, "tuple_get_item", x_inp, y_inp)
    tvm_tirx.testing.assert_allclose(
        res.numpy(), x_inp.numpy() + y_inp.numpy(), rtol=1e-7, atol=1e-7
    )


def test_lower_memory_alloc_storage_tensor():
    @tvm_tirx.script.ir_module
    class TestMemoryAllocStorageTensor:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = TestMemoryAllocStorageTensor
            storage = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            y = R.memory.alloc_tensor(storage, 0, R.shape([2, 3]), dtype="float32")
            # this is an impure operation, but the overall function is pure so we force purity
            _ = R.call_tir_packed(cls.copy, (x, y))
            return y

        @Ts.function
        def copy(A: T.Tensor((2, 3), "float32"), B: T.Tensor((2, 3), "float32")):
            for i0, i1 in T.grid(2, 3):
                with Ts.sblock("block"):
                    vi0, vi1 = Ts.axis.remap("SS", [i0, i1])
                    B[vi0, vi1] = A[vi0, vi1]

    mod = TestMemoryAllocStorageTensor
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    x = tvm_tirx.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    y = vm["main"](x)
    tvm_tirx.testing.assert_allclose(y.numpy(), x.numpy(), rtol=1e-7, atol=1e-7)


def test_sub_func_call():
    m = T.dynamic("m", "int32")
    n = T.dynamic("n", "int32")
    k = T.dynamic("k", "int32")

    @tvm_tirx.script.ir_module
    class TestVMSubFunction:
        @Ts.function
        def tir_matmul(
            A: T.Tensor((m, n), "float32"),
            B: T.Tensor((n, k), "float32"),
            C: T.Tensor((m, k), "float32"),
        ) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})

            for i, j, k_index in T.grid(m, k, n):
                with Ts.sblock("matmul"):
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k_index])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def relax_matmul_tir(
            x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")
        ) -> R.Tensor((32, 32), dtype="float32"):
            cls = TestVMSubFunction
            with R.dataflow():
                gv0 = R.call_tir(
                    cls.tir_matmul, (x, w), ty_args=[R.Tensor((32, 32), dtype="float32")]
                )
                R.output(gv0)
            return gv0

        @R.function
        def relax_matmul_packed(
            x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")
        ) -> R.Any:
            gv0 = R.call_pure_packed(
                "test.vm.mul", x, w, ty_args=(R.Tensor(ndim=2, dtype="float32"))
            )
            return gv0

        @R.function
        def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Any:
            cls = TestVMSubFunction
            gv0 = cls.relax_matmul_tir(x, w)
            gv1 = cls.relax_matmul_packed(gv0, gv0)
            return gv1

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(TestVMSubFunction, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    x_inp = tvm_tirx.runtime.tensor(np.random.rand(32, 32).astype(np.float32))
    y_inp = tvm_tirx.runtime.tensor(np.random.rand(32, 32).astype(np.float32))
    res = check_saved_func(vm, "main", x_inp, y_inp)
    product = np.dot(x_inp.numpy(), y_inp.numpy())
    expected = product * product
    tvm_tirx.testing.assert_allclose(res.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_recursion():
    @tvm_tirx.script.ir_module
    class TestVMRecursion:
        @R.function
        def recursion(n: R.Tensor((1,), "float32")) -> R.Tensor:
            cond = R.call_pure_packed(
                "test.vm.equal_zero", n, ty_args=(R.Tensor(ndim=1, dtype="float32"))
            )
            if cond:
                res = R.const(1.0)
            else:
                gv0 = R.call_pure_packed(
                    "test.vm.subtract_one", n, ty_args=(R.Tensor(ndim=1, dtype="float32"))
                )
                tmp = TestVMRecursion.recursion(gv0)
                res = R.call_pure_packed(
                    "test.vm.add", tmp, tmp, ty_args=(R.Tensor(ndim=1, dtype="float32"))
                )
            return res

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(TestVMRecursion, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    inp = np.empty(1).astype("float32")
    recursion_runs = np.random.randint(1, 10)
    inp.fill(recursion_runs)
    inp = tvm_tirx.runtime.tensor(inp)
    res = check_saved_func(vm, "recursion", inp)
    tvm_tirx.testing.assert_allclose(
        res.numpy(), np.power(2.0, recursion_runs), rtol=1e-7, atol=1e-7
    )


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_vm_to_device():
    @tvm_tirx.script.ir_module
    class TestToVDevice:
        @R.function
        def foo1(
            x: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32"):
            copied = R.to_vdevice(x, tvm.relax.VDevice("cuda", 0, "global"))
            return copied

        @R.function
        def foo2(
            x: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32"):
            copied = R.to_vdevice(x, tvm.relax.VDevice("llvm", 0, "global"))
            return copied

    mod = TestToVDevice
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)

    def run_and_check():
        vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
        x_inp = tvm_tirx.runtime.tensor(np.random.rand(2, 3).astype("float32"))
        res_1 = check_saved_func(vm, "foo1", x_inp)
        res_2 = check_saved_func(vm, "foo2", x_inp)

        # check the copied tensor's device
        assert res_1.device == tvm_tirx.cuda(0)
        assert res_2.device == tvm_tirx.cpu(0)

        tvm_tirx.testing.assert_allclose(res_1.numpy(), x_inp.numpy())
        tvm_tirx.testing.assert_allclose(res_2.numpy(), x_inp.numpy())

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


def test_vm_closure():
    @tvm_tirx.script.ir_module
    class TestClosure:
        @R.function
        def lifted_func_1(x: R.Tensor((2, 3), "float32"), env: R.Tensor((2, 3), "float32")):
            return R.call_pure_packed("test.vm.add", x, env, ty_args=(R.Tensor()))

        @R.function
        def main(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((2, 3), "float32"),
        ):
            cls = TestClosure
            clo = R.make_closure(cls.lifted_func_1, (x,))
            res = R.invoke_pure_closure(clo, (y,), ty_args=(R.Tensor()))
            return res

    mod = TestClosure
    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    x_inp = tvm_tirx.runtime.tensor(np.random.rand(2, 3).astype("float32"))
    y_inp = tvm_tirx.runtime.tensor(np.array([[3.1, 4.0, 5.0], [6.0, 7.1, 9.0]], dtype="float32"))
    res = check_saved_func(vm, "main", x_inp, y_inp)
    tvm_tirx.testing.assert_allclose(res.numpy(), x_inp.numpy() + y_inp.numpy())


def test_time_evaluator():
    @tvm_tirx.script.ir_module
    class TestTimeEvaluator:
        @R.function
        def main(x: R.Tensor((1,), "float32"), y: R.Tensor((1,), "float32")):
            return R.call_pure_packed(
                "test.vm.add", x, y, ty_args=(R.Tensor(ndim=1, dtype="float32"))
            )

    target = tvm_tirx.target.Target("llvm", host="llvm")
    ex = relax.build(TestTimeEvaluator, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    x = tvm_tirx.runtime.tensor(np.random.rand(1).astype("float32"))
    y = tvm_tirx.runtime.tensor(np.random.rand(1).astype("float32"))

    # ensure we can use time_evaluator with the stateful API
    vm.set_input("main", x, y)
    timing_res = vm.time_evaluator("invoke_stateful", tvm_tirx.cpu())("main")
    # just checking that it has some results at all
    assert timing_res.results

    # ensure we can use it with a closure
    vm.save_function("main", "saved_main", x, y)
    timing_res = vm.time_evaluator("saved_main", tvm_tirx.cpu())()
    assert timing_res.results


m = T.dynamic("m", "int32")
n = T.dynamic("n", "int32")


@tvm_tirx.script.ir_module
class TestVMSetInput:
    @Ts.function
    def test_vm_mul(
        A: T.Tensor((m, n), "float32"),
        B: T.Tensor((m, n), "float32"),
        C: T.Tensor((m, n), "float32"),
    ):
        T.func_attr({"global_symbol": "test_vm_mul"})

        for i, j in T.grid(m, n):
            with Ts.sblock("mul"):
                vi = Ts.axis.spatial(m, i)
                vj = Ts.axis.spatial(n, j)
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = A[vi, vj] * B[vi, vj]

    # test returning a tuple
    @R.function
    def test_vm_tuple(
        x: R.Tensor((), "int32"),
    ) -> R.Tuple(R.Tensor((), "int32"), R.Tensor((), "int32")):
        return (x, x)

    # nested tuple too
    @R.function
    def test_vm_nested_tuple(x: R.Tensor((), "int32")) -> R.Tuple(
        R.Tuple(
            R.Tensor((), "int32"),
            R.Tuple(
                R.Tensor((), "int32"),
            ),
        ),
        R.Tensor((), "int32"),
    ):
        return ((x, (x,)), x)

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        cls = TestVMSetInput
        gv0 = R.call_tir(cls.test_vm_mul, (x, w), ty_args=[R.Tensor((32, 32), dtype="float32")])
        return gv0


def test_multi_systemlib():
    pytest.importorskip("cloudpickle")  # needed by popen_pool.PopenWorker

    N = T.dynamic("N", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class ModA:
        I.module_attrs({"system_lib_prefix": "libA_"})

        @Ts.function
        def tir_init(x: T.Tensor([N], "float32")):
            for i in range(N):
                x[i] = T.float32(0)

        @R.function
        def main(s: R.Shape([m])) -> R.Tensor:
            gv0 = R.call_tir(ModA.tir_init, (), ty_args=[R.Tensor((m + 1,), dtype="float32")])
            return gv0

    N = T.dynamic("N", "int64")
    m = T.dynamic("m", "int64")

    @tvm_tirx.script.ir_module
    class ModB:
        I.module_attrs({"system_lib_prefix": "libB_"})

        @Ts.function
        def tir_init(x: T.Tensor([N], "float32")):
            for i in range(N):
                x[i] = T.float32(1)

        @R.function
        def main(s: R.Shape([m])) -> R.Tensor:
            gv0 = R.call_tir(ModB.tir_init, (), ty_args=[R.Tensor((m,), dtype="float32")])
            return gv0

    target = tvm_tirx.target.Target("llvm", host="llvm")
    libA = relax.build(ModA, target)
    libB = relax.build(ModB, target)

    temp = utils.tempdir()
    pathA = temp.relpath("libA.a")
    pathB = temp.relpath("libB.a")
    path_dso = temp.relpath("mylibAll.so")
    libA.export_library(pathA, fcompile=cc.create_staticlib)
    libB.export_library(pathB, fcompile=cc.create_staticlib)

    # package two static libs together
    # check that they do not interfere with each other
    # even though they have shared global var names
    # intentionally craft same gvar function with different behaviors
    cc.create_shared(path_dso, ["-Wl,--whole-archive", pathA, pathB, "-Wl,--no-whole-archive"])

    def popen_check():
        # Load dll, will trigger system library registration
        ctypes.CDLL(path_dso)
        # Load the system wide library
        vmA = relax.VirtualMachine(tvm_ffi.system_lib("libA_"), tvm_tirx.cpu())
        vmB = relax.VirtualMachine(tvm_ffi.system_lib("libB_"), tvm_tirx.cpu())

        retA = vmA["main"](tvm_ffi.Shape([1]))
        retB = vmB["main"](tvm_ffi.Shape([2]))
        np.testing.assert_equal(retA.numpy(), np.array([0, 0]).astype("float32"))
        np.testing.assert_equal(retB.numpy(), np.array([1, 1]).astype("float32"))

    # system lib should be loaded in different process
    worker = popen_pool.PopenWorker()
    worker.send(popen_check)


def _trial_tensor(data, device):
    if isinstance(device, rpc.RemoteDevice):
        return device.tensor(data)
    return tvm_tirx.runtime.tensor(data, device)


def set_input_trial(vm: relax.VirtualMachine, device: tvm_tirx.runtime.Device) -> None:
    a = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    b = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    vm.invoke_stateful("main")
    res0 = vm.get_outputs("main")

    data_dict = {"x": a, "w": b}
    vm.set_input("main", **data_dict)
    vm.invoke_stateful("main")
    res1 = vm.get_outputs("main")
    tvm_tirx.testing.assert_allclose(res0.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)
    tvm_tirx.testing.assert_allclose(res0.numpy(), res1.numpy(), rtol=1e-7, atol=1e-7)

    # bug! If you don't bind the Tensor to a var, the memory will get corrupted.
    # Possibly due to object lifecycles and other FFI issues
    a = _trial_tensor(np.array(2).astype("int32"), device)
    vm.set_input("test_vm_tuple", a)
    vm.invoke_stateful("test_vm_tuple")
    res2 = vm.get_outputs("test_vm_tuple")
    # the results are Tensors wrapped around scalars,
    # so we have to get the scalar out of the Tensor
    assert tuple(map(lambda a: int(a.numpy()), res2)) == (2, 2)

    b = _trial_tensor(np.array(1).astype("int32"), device)
    vm.set_input("test_vm_nested_tuple", b)
    vm.invoke_stateful("test_vm_nested_tuple")
    res3 = vm.get_outputs("test_vm_nested_tuple")
    assert len(res3) == 2 and len(res3[0]) == 2 and len(res3[0][1]) == 1
    result_cast = ((int(res3[0][0].numpy()), (int(res3[0][1][0].numpy()),)), int(res3[1].numpy()))
    assert result_cast == ((1, (1,)), 1)


def set_input_attempt_stateless(vm: relax.VirtualMachine, device: tvm_tirx.runtime.Device) -> None:
    # this should fail: once you set inputs, you cannot run statelessly
    a = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    b = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    # must use invoke stateful!
    vm["main"]()


def set_input_attempt_invoke(vm: relax.VirtualMachine, device: tvm_tirx.runtime.Device) -> None:
    # this should fail: if the function needs inputs, you can't invoke directly
    vm.invoke_stateful("main")


def set_input_attempt_get(vm: relax.VirtualMachine, device: tvm_tirx.runtime.Device) -> None:
    # this should fail: you can't get outputs without invoking the function first
    a = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    b = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    vm.set_input("main", a, b)
    _ = vm.get_outputs("main")


def make_vm(mod, temp) -> tuple[relax.VirtualMachine, tvm_tirx.runtime.Device]:
    """Returns a local VM for the given mod and the device"""
    target = tvm_tirx.target.Target("llvm", host="llvm")
    exec = relax.build(mod, target)
    libname = temp.relpath("exec.so")
    exec.export_library(libname)
    exec_loaded = tvm_tirx.runtime.load_module(libname)
    device = tvm_tirx.cpu()
    return relax.VirtualMachine(exec_loaded, device), device


def run_on_rpc(
    mod: tvm_tirx.IRModule,
    trial_func: Callable[[relax.VirtualMachine, tvm_tirx.runtime.Device], None],
):
    """
    Sets up a VM over localhost using the given mod and runs the given trial function.
    The trial function should take a VM and a device
    """
    target = tvm_tirx.target.Target("llvm", host="llvm")
    exec = relax.build(mod, target)
    temp = utils.tempdir()
    path = temp.relpath("vm_library.so")
    exec.export_library(path)

    # Use local rpc server for testing.
    # Server must use popen so it doesn't inherit the current process state. It
    # will crash otherwise.
    def check_remote(server):
        remote = rpc.connect(server.host, server.port, session_timeout=10)

        # Upload the serialized Executable.
        remote.upload(path)
        # Get a handle to remote Executable.
        rexec = remote.load_module("vm_library.so")

        device = remote.cpu()
        # Build a VM out of the executable and context.
        vm = rpc.virtual_machine(rexec, device)
        trial_func(vm, device)

    check_remote(rpc.Server("127.0.0.1"))


def test_set_input():
    temp = utils.tempdir()
    set_input_trial(*make_vm(TestVMSetInput, temp))


def test_set_input_tuple():
    @tvm_tirx.script.ir_module
    class MyMod:
        @R.function
        def main(x: R.Tuple([R.Tensor((32,), "float32"), R.Tensor((32,), "float32")])) -> R.Tensor:
            y = x[0]
            return y

    temp = utils.tempdir()
    vm, device = make_vm(MyMod, temp)
    device = tvm_tirx.cpu(0)
    a = tvm_tirx.runtime.empty((32,), "float32", device=device)
    b = tvm_tirx.runtime.empty((32,), "float32", device=device)
    vm.set_input("main", (a, b))
    vm.invoke_stateful("main")


def save_function_kwargs_trial(vm: relax.VirtualMachine, device: tvm_tirx.runtime.Device) -> None:
    # just checking that we can use kwargs for the args when saving a function
    a = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    b = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    vm.save_function("main", "saved_main", x=a, w=b)
    res0 = vm["saved_main"]()
    tvm_tirx.testing.assert_allclose(res0.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)


def test_save_function_kwargs():
    temp = utils.tempdir()
    save_function_kwargs_trial(*make_vm(TestVMSetInput, temp))


def test_save_function_kwargs_rpc():
    pytest.importorskip("cloudpickle")  # needed by the popen RPC server
    run_on_rpc(TestVMSetInput, save_function_kwargs_trial)


def save_function_time_evaluator_trial(
    vm: relax.VirtualMachine, device: tvm_tirx.runtime.Device
) -> None:
    # just checking that the saved function can be called in the time evaluator
    a = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    b = _trial_tensor(np.random.rand(32, 32).astype("float32"), device)
    vm.save_function("main", "saved_main", a, b)
    vm.time_evaluator("saved_main", device)()


def test_save_function_time_evaluator():
    temp = utils.tempdir()
    save_function_time_evaluator_trial(*make_vm(TestVMSetInput, temp))


def test_save_function_time_evaluator_rpc():
    pytest.importorskip("cloudpickle")  # needed by the popen RPC server
    run_on_rpc(TestVMSetInput, save_function_time_evaluator_trial)


# if you set an input, you should not be able to call statelessly


def test_set_input_stateless_failure():
    temp = utils.tempdir()
    args = make_vm(TestVMSetInput, temp)
    with pytest.raises(RuntimeError):
        set_input_attempt_stateless(*args)


def test_set_input_stateless_failure_rpc():
    pytest.importorskip("cloudpickle")  # needed by the popen RPC server
    with pytest.raises(RuntimeError):
        run_on_rpc(TestVMSetInput, set_input_attempt_stateless)


def test_set_input_invoke_failure():
    temp = utils.tempdir()
    args = make_vm(TestVMSetInput, temp)
    with pytest.raises(ValueError):
        set_input_attempt_invoke(*args)


def test_set_input_invoke_failure_rpc():
    pytest.importorskip("cloudpickle")  # needed by the popen RPC server
    with pytest.raises(RuntimeError):
        run_on_rpc(TestVMSetInput, set_input_attempt_invoke)


def test_set_input_get_failure():
    temp = utils.tempdir()
    args = make_vm(TestVMSetInput, temp)
    with pytest.raises(ValueError):
        set_input_attempt_get(*args)


def test_set_input_get_failure_rpc():
    pytest.importorskip("cloudpickle")  # needed by the popen RPC server
    with pytest.raises(RuntimeError):
        run_on_rpc(TestVMSetInput, set_input_attempt_get)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_gpu(), reason="need gpu")
def test_relax_module_with_multiple_targets():
    """Relax functions may contain kernels for multiple targets

    In this example, the module contains one function to execute on
    LLVM, and one function to execute on CUDA.

    """

    @I.ir_module
    class Module:
        I.module_global_infos({"vdevice": [R.vdevice("llvm")]})

        @R.function
        def func_cuda(A: R.Tensor([32, 32], "float32"), B: R.Tensor([32, 32], "float32")):
            C = R.add(A, B)
            return C

        @R.function
        def func_llvm(
            A: R.Tensor([32, 32], "float32", "llvm"), B: R.Tensor([32, 32], "float32", "llvm")
        ):
            C = R.add(A, B)
            return C

    seq = tvm_tirx.ir.transform.Sequential(
        [
            tvm.relax.transform.LegalizeOps(),
            tvm.s_tir.dlight.ApplyDefaultSchedule(tvm.s_tir.dlight.gpu.Fallback(), target="cuda"),
        ],
        name="LegalizeAndSchedule",
    )
    built = tvm.relax.build(seq(Module), target="cuda")

    np_A = np.random.random([32, 32]).astype("float32")
    np_B = np.random.random([32, 32]).astype("float32")

    dev_llvm = tvm_tirx.cpu()
    vm_llvm = tvm.relax.VirtualMachine(built, device=dev_llvm)
    llvm_output = vm_llvm["func_llvm"](
        tvm_tirx.runtime.tensor(np_A, dev_llvm),
        tvm_tirx.runtime.tensor(np_B, dev_llvm),
    )

    np_C = np_A + np_B

    tvm_tirx.testing.assert_allclose(llvm_output.numpy(), np_C)

    def run_and_check():
        dev_cuda = tvm_tirx.cuda()
        vm_cuda = tvm.relax.VirtualMachine(built, device=dev_cuda)
        cuda_output = vm_cuda["func_cuda"](
            tvm_tirx.runtime.tensor(np_A, dev_cuda),
            tvm_tirx.runtime.tensor(np_B, dev_cuda),
        )
        tvm_tirx.testing.assert_allclose(cuda_output.numpy(), np_C)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
