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
# ruff: noqa: E741
import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm import relax
from tvm.relax.testing.gradient import check_numerical_grads
from tvm.testing import device_enabled
from tvm_tirx.script.parser import ir as I
from tvm_tirx.script.parser import relax as R
from tvm_tirx.testing import assert_allclose


def rand(dtype, *shape):
    return tvm_tirx.runtime.tensor(np.random.rand(*shape).astype(dtype))


def _legalize_and_build(mod, target):
    ex = tvm.compile(mod, target)
    vm = relax.VirtualMachine(ex, tvm_tirx.device_from_target(target))
    return vm


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_manual_gradient():
    target = "llvm"

    # The expression computed is sum((2x - 2y) * (y + z))
    # the gradient of x is broadcast_to(2y + 2z, x.shape)
    # the gradient of y is collapse_sum_to((2x - 4y - 2z), y.shape)
    # the gradient of z is collapse_sum_to((2x - 2y), z.shape)
    # the gradient of u is 0
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((3, 5), "float32"),
            y: R.Tensor((5,), "float32"),
            z: R.Tensor((5,), "float32"),
            u: R.Tensor((5,), "float32"),
        ):
            with R.dataflow():
                lv1 = R.add(x, x)
                lv2 = R.subtract(lv1, y)
                lv3 = R.subtract(lv2, y)
                lv4 = R.add(y, z)
                lv5 = R.multiply(lv3, lv4)
                lv6 = R.sum(lv5)
                R.output(lv6)
            return lv6

    After = relax.transform.Gradient("main")(Before)

    args = [rand("float32", 3, 5), rand("float32", 5), rand("float32", 5), rand("float32", 5)]
    args_np = [x.numpy() for x in args]

    vm = _legalize_and_build(After, target)
    output, grads = vm["main_adjoint"](*args)
    output_np = np.sum((2 * args_np[0] - 2 * args_np[1]) * (args_np[1] + args_np[2]))
    assert_allclose(output.numpy(), output_np, atol=1e-4)

    expected_grads_nd = [
        (2 * args_np[1] + 2 * args_np[2]) * np.ones_like(args_np[0]),
        np.sum((2 * args_np[0] - 4 * args_np[1] - 2 * args_np[2]), axis=0),
        np.sum((2 * args_np[0] - 2 * args_np[1]), axis=0),
        np.zeros_like(args_np[3]),
    ]
    for i, j in zip(grads, expected_grads_nd):
        assert_allclose(i.numpy(), j, atol=1e-4)


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_mlp_blockbuilder():
    target = "llvm"
    layers, in_size, out_size, hidden_size, batch_size = 3, 5, 5, 5, 4

    input_list = [relax.Var("x", R.Tensor((batch_size, in_size), "float32"))]
    w_list = (
        [relax.Var("w_0", R.Tensor((in_size, hidden_size), "float32"))]
        + [
            relax.Var("w_" + str(i + 1), R.Tensor((hidden_size, hidden_size), "float32"))
            for i in range(layers - 2)
        ]
        + [relax.Var("w_" + str(layers - 1), R.Tensor((hidden_size, out_size), "float32"))]
    )
    b_list = [
        relax.Var("b_" + str(i), R.Tensor((hidden_size,), "float32")) for i in range(layers - 1)
    ] + [relax.Var("b_" + str(layers - 1), R.Tensor((out_size,), "float32"))]
    label_list = [relax.Var("y", R.Tensor((batch_size,), "int64"))]
    args_list = input_list + w_list + b_list + label_list

    bb = relax.BlockBuilder()
    with bb.function("MLP", args_list):
        with bb.dataflow():
            current = input_list[0]
            for i in range(layers):
                lv0 = bb.emit(R.matmul(current, w_list[i]))
                lv1 = bb.emit(R.add(lv0, b_list[i]))
                current = bb.emit(R.nn.relu(lv1) if i < layers - 1 else lv1)
            logits = R.nn.log_softmax(current)
            loss = bb.emit(R.nn.nll_loss(logits, label_list[0]))
            gv0 = bb.emit_output(loss)
        bb.emit_func_output(gv0)

    Before = bb.get()
    After = relax.transform.Gradient("MLP", w_list + b_list)(Before)
    # Check numerical gradients equal
    args = []
    for arg in After["MLP_adjoint"].params:
        shape = [int(l) for l in arg.ty.shape]
        if arg.ty.dtype == "int64":
            args.append(
                tvm_tirx.runtime.tensor(np.random.randint(0, out_size, size=shape).astype(np.int64))
            )
        else:  # float32
            args.append(rand("float32", *shape))

    vm_before = _legalize_and_build(Before, target)
    vm_after = _legalize_and_build(After, target)
    _, grad = vm_after["MLP_adjoint"](*args)

    def func(*inputs):
        loss = vm_before["MLP"](args[0], *[tvm_tirx.runtime.tensor(i) for i in inputs], args[-1])
        return loss.numpy()

    check_numerical_grads(func, [i.numpy() for i in args[1:-1]], [i.numpy() for i in grad])


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_complex():
    target = "llvm"
    cst = relax.const(np.ones((6,)), dtype="float32")
    cst1 = relax.const(np.array(3), dtype="int64")

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((6,), "float32"), y: R.Tensor((6, 3, 4), "float32")):
            with R.dataflow():
                lv1 = R.split(x, 2)
                lv2 = lv1[0]
                lv3 = lv1[1]
                lv4 = lv2 + lv3
                lv5 = (lv4, lv3)
                lv6 = R.concat(lv5)
                lv7 = (x, x)
                lv8 = R.concat(lv7)
                lv9 = R.concat(lv7)
                lv10 = R.add(lv8, lv9)
                lv11 = R.split(lv10, 2)
                lv12 = R.add(lv6, lv11[0])
                lv13 = cst
                lv14 = R.add(lv12, lv13)
                lv15 = R.subtract(lv13, lv14)
                lv16 = R.multiply(lv14, lv15)
                lv17 = R.multiply(lv15, lv16)
                lv18 = R.tanh(lv17)
                lv19 = R.sigmoid(lv18)
                lv20 = R.permute_dims(y, axes=[0, 2, 1])
                lv21 = R.sigmoid(lv20)
                lv22 = R.matmul(y, lv21)
                lv23 = R.sum(lv22, axis=[1, 2])
                lv24 = R.add(lv19, lv23)
                lv25 = R.nn.log_softmax(lv24)
                gv = R.nn.nll_loss(lv25, cst1)
                R.output(gv)
            return gv

    After = relax.transform.Gradient("main")(Before)
    args = []
    for arg in After["main_adjoint"].params:
        shape = [int(l) for l in arg.ty.shape]
        args.append(rand("float32", *shape))

    vm_before = _legalize_and_build(Before, target)
    vm_after = _legalize_and_build(After, target)
    _, grad = vm_after["main_adjoint"](*args)

    def func(*inputs):
        loss = vm_before["main"](*[tvm_tirx.runtime.tensor(i) for i in inputs])
        return loss.numpy()

    check_numerical_grads(func, [i.numpy() for i in args], [i.numpy() for i in grad])


@pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")
def test_matmul():
    target = "llvm"

    @tvm_tirx.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((3, 3), "float32"), y: R.Tensor((3, 3), "float32")):
            with R.dataflow():
                lv1 = R.matmul(x, y)
                lv2 = R.permute_dims(x)
                lv3 = R.matmul(lv2, y)
                lv4 = R.permute_dims(y)
                lv5 = R.matmul(x, lv4)
                lv6 = R.permute_dims(x)
                lv7 = R.permute_dims(y)
                lv8 = R.matmul(lv6, lv7)
                lv9 = lv1 + lv3 + lv5 + lv8
                gv = R.sum(lv9)
                R.output(gv)
            return gv

    After = relax.transform.Gradient("main")(Before)
    args = []
    for arg in After["main_adjoint"].params:
        shape = [int(l) for l in arg.ty.shape]
        args.append(rand("float32", *shape))

    vm_before = _legalize_and_build(Before, target)
    vm_after = _legalize_and_build(After, target)
    _, grad = vm_after["main_adjoint"](*args)

    def func(*inputs):
        loss = vm_before["main"](*[tvm_tirx.runtime.tensor(i) for i in inputs])
        return loss.numpy()

    check_numerical_grads(func, [i.numpy() for i in args], [i.numpy() for i in grad])


if __name__ == "__main__":
    tvm_tirx.testing.main()
