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
"""Lowest level testing VM. Test execbuilder and execution."""

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
from tvm import relax
from tvm.relax import script as R
from tvm.relax.testing.vm import check_saved_func


def test_vm_execute():
    ib = relax.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    a = tvm_tirx.runtime.tensor(
        np.random.rand(
            4,
        )
    )
    b = tvm_tirx.runtime.tensor(
        np.random.rand(
            4,
        )
    )

    add_res = check_saved_func(vm, "func0", a, b)
    tvm_tirx.testing.assert_allclose(add_res.numpy(), a.numpy() + b.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_multiple_func():
    ib = relax.ExecBuilder()
    with ib.function("func0", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    with ib.function("func1", num_inputs=2):
        ib.emit_call("test.vm.mul", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    a = tvm_tirx.runtime.tensor(
        np.random.rand(
            4,
        )
    )
    b = tvm_tirx.runtime.tensor(
        np.random.rand(
            4,
        )
    )
    mul_res = check_saved_func(vm, "func1", a, b)
    add_res = check_saved_func(vm, "func0", a, b)
    tvm_tirx.testing.assert_allclose(add_res.numpy(), a.numpy() + b.numpy(), rtol=1e-7, atol=1e-7)
    tvm_tirx.testing.assert_allclose(mul_res.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_checker():
    ib = relax.ExecBuilder()
    with pytest.raises(RuntimeError):
        with ib.function("func0", num_inputs=2):
            ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(2)], dst=ib.r(2))
            ib.emit_ret(ib.r(2))
        ib.get()


def test_neg_imm():
    ib = relax.ExecBuilder()

    with ib.function("func0", num_inputs=1):
        ib.emit_call("test.vm.add_scalar", args=[ib.imm(-3), ib.r(0)], dst=ib.r(1))
        ib.emit_ret(ib.r(1))
    ib.get()

    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    assert vm["func0"](1) == -2
    assert vm["func0"](-3) == -6


def test_emit_cache():
    ib = relax.ExecBuilder()

    with ib.function("func0", num_inputs=1):
        x0 = ib.convert_constant("str0")
        x1 = ib.convert_constant("str0")
        # cache constant str
        assert x0 == x1
        s0 = ib.convert_constant(tvm_ffi.Shape([1, 2]))
        s1 = ib.convert_constant(tvm_ffi.Shape([1, 2]))
        s2 = ib.convert_constant(tvm_ffi.Shape([1, 3]))
        assert s0 == s1
        assert s1 != s2
        y0 = ib.convert_constant(tvm_tirx.runtime.tensor(np.array([1, 2, 3]).astype("int32")))
        y1 = ib.convert_constant(tvm_tirx.runtime.tensor(np.array([1, 2, 3]).astype("int32")))
        assert y0 == y1
        ib.emit_ret(ib.r(0))


def test_vm_formalize():
    ib0 = relax.ExecBuilder()
    ib1 = relax.ExecBuilder()
    with ib0.function("func0", num_inputs=2):
        ib0.emit_call("test.vm.add", args=[ib0.r(0), ib0.r(1)], dst=ib0.r(100))
        ib0.emit_call("test.vm.mul", args=[ib0.r(1), ib0.r(100)], dst=ib0.r(50))
        ib0.emit_ret(ib0.r(50))
    with ib1.function("func0", num_inputs=2):
        ib1.emit_call("test.vm.add", args=[ib1.r(0), ib1.r(1)], dst=ib1.r(2))
        ib1.emit_call("test.vm.mul", args=[ib1.r(1), ib1.r(2)], dst=ib1.r(3))
        ib1.emit_ret(ib1.r(3))
    exec0 = ib0.get()
    exec1 = ib1.get()
    assert exec0.as_text() == exec1.as_text()


def test_vm_operand():
    ib0 = relax.ExecBuilder()
    with ib0.function("func0", num_inputs=2):
        ib0.emit_call("test.vm.add_scalar", args=[ib0.r(0), ib0.r(1)], dst=ib0.r(2))
        ib0.emit_ret(ib0.r(2))
    exec0 = ib0.get()
    vm = relax.VirtualMachine(exec0, tvm_tirx.cpu())
    res = vm["func0"](2, 3)
    assert res == 5

    ib1 = relax.ExecBuilder()
    with ib1.function("func1", num_inputs=1):
        ib1.emit_call("test.vm.get_device_id", args=[ib1.r(0)], dst=ib1.r(1))
        ib1.emit_ret(ib1.r(1))
    exec1 = ib1.get()
    vm = relax.VirtualMachine(exec1, tvm_tirx.cpu())
    res = vm["func1"](tvm_tirx.cpu(3))
    assert res == 3


def test_vm_shapeof():
    ib = relax.ExecBuilder()
    shape = (32, 16)
    arr = tvm_tirx.runtime.tensor(np.random.rand(*shape))
    with ib.function("main", num_inputs=0):
        ib.emit_call("vm.builtin.shape_of", args=[arr], dst=ib.r(0))
        ib.emit_ret(ib.r(0))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    res = vm["main"]()
    for i, s in enumerate(res):
        assert s == shape[i]


def test_vm_storage():
    dtype = tvm_ffi.dtype("float32")
    shape = (4, 6)
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=0):
        ib.emit_call(
            "vm.builtin.alloc_storage",
            args=[
                ib.vm_state(),
                (24,),
                ib.convert_constant(0),
                dtype,
                ib.convert_constant("global"),
            ],
            dst=ib.r(1),
        )
        ib.emit_call(
            "vm.builtin.alloc_tensor",
            args=[ib.vm_state(), ib.r(1), ib.imm(0), shape, dtype],
            dst=ib.r(2),
        )
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    res = vm["main"]()
    assert res.device == tvm_tirx.cpu()
    assert res.shape == shape


def test_vm_goto():
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=2):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(2))
        ib.emit_goto(2)
        ib.emit_call("test.vm.mul", args=[ib.r(2), ib.r(1)], dst=ib.r(2))
        ib.emit_ret(ib.r(2))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    a = tvm_tirx.runtime.tensor(
        np.random.rand(
            4,
        )
    )
    b = tvm_tirx.runtime.tensor(
        np.random.rand(
            4,
        )
    )
    res = check_saved_func(vm, "main", a, b)
    tvm_tirx.testing.assert_allclose(res.numpy(), a.numpy() + b.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_if():
    ib = relax.ExecBuilder()
    with ib.function("main", num_inputs=3):
        ib.emit_if(ib.r(0), 3)
        ib.emit_call("test.vm.add", args=[ib.r(1), ib.r(2)], dst=ib.r(3))
        ib.emit_goto(2)
        ib.emit_call("test.vm.mul", args=[ib.r(1), ib.r(2)], dst=ib.r(3))
        ib.emit_ret(ib.r(3))
    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    a = tvm_tirx.runtime.tensor(
        np.random.rand(
            4,
        )
    )
    b = tvm_tirx.runtime.tensor(
        np.random.rand(
            4,
        )
    )
    res = vm["main"](0, a, b)
    tvm_tirx.testing.assert_allclose(res.numpy(), a.numpy() * b.numpy(), rtol=1e-7, atol=1e-7)
    res = vm["main"](1, a, b)
    tvm_tirx.testing.assert_allclose(res.numpy(), a.numpy() + b.numpy(), rtol=1e-7, atol=1e-7)


def test_vm_invoke_closure():
    ib = relax.ExecBuilder()
    with ib.function("lifted_func_1", num_inputs=4):
        ib.emit_call("test.vm.add", args=[ib.r(0), ib.r(1)], dst=ib.r(4))
        ib.emit_call("test.vm.add", args=[ib.r(2), ib.r(4)], dst=ib.r(5))
        ib.emit_call("test.vm.add", args=[ib.r(3), ib.r(5)], dst=ib.r(6))
        ib.emit_ret(ib.r(6))
    with ib.function("main", num_inputs=2):
        ib.emit_call(
            "vm.builtin.make_closure", args=[ib.f("lifted_func_1"), ib.r(0), ib.r(1)], dst=ib.r(2)
        )
        ib.emit_ret(ib.r(2))

    ex = ib.get()
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())
    w_inp = tvm_tirx.runtime.tensor(np.random.rand(2, 3))
    x_inp = tvm_tirx.runtime.tensor(np.random.rand(2, 3))
    y_inp = tvm_tirx.runtime.tensor([[3.1, 4.0, 5.0], [6.0, 7.1, 9.0]])
    z_inp = tvm_tirx.runtime.tensor(np.random.rand(2, 3))
    clo = vm["main"](w_inp, x_inp)
    res = vm.invoke_closure(clo, y_inp, z_inp)
    tvm_tirx.testing.assert_allclose(
        res.numpy(), w_inp.numpy() + x_inp.numpy() + y_inp.numpy() + z_inp.numpy()
    )


def test_vm_stack_restore_after_failure():
    @tvm_tirx.script.ir_module
    class Module:
        @R.function
        def main(inp: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.multiply(inp, R.const(2, "float32"))
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    ex = tvm.compile(Module, "llvm")
    vm = relax.VirtualMachine(ex, tvm_tirx.cpu())

    correct_input = tvm_tirx.runtime.tensor(np.random.normal(size=(10, 10)).astype("float32"))
    incorrect_input = tvm_tirx.runtime.tensor(np.random.normal(size=(12, 10)).astype("float32"))

    try:
        vm["main"](incorrect_input)
    except RuntimeError:
        pass

    # VM should executes correctly after encountered incorrect shape in previous invocation
    vm["main"](correct_input)


if __name__ == "__main__":
    tvm_tirx.testing.main()
