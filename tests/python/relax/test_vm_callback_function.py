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

import numpy as np
import pytest

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax import script as R
from tvm.testing import device_enabled

pytestmark = pytest.mark.skipif(not device_enabled("llvm"), reason="llvm not enabled")


def test_pass_tensor_to_function():
    target = "llvm"
    dev = tvm_tirx.cpu()

    @R.function
    def relax_func(
        A: R.Tensor([16], "int32"),
        callback: R.Callable([R.Tensor([16], "int32")], R.Tuple([])),
    ):
        B = R.multiply(A, R.const(2))
        _ = callback(B)
        return ()

    ex = tvm.relax.build(
        tvm_tirx.IRModule.from_expr(relax_func),
        target=target,
    )
    vm = tvm.relax.VirtualMachine(ex, dev)

    from_callback = None

    def custom_callback(arr):
        nonlocal from_callback
        from_callback = arr

    np_A = np.arange(16, dtype="int32")
    tvm_A = tvm_tirx.runtime.tensor(np_A)

    vm["relax_func"](tvm_A, custom_callback)

    assert from_callback is not None
    np.testing.assert_array_equal(np_A * 2, from_callback.numpy())


def test_generate_tensor_in_function():
    target = "llvm"
    dev = tvm_tirx.cpu()

    @R.function
    def relax_func(
        callback: R.Callable([], R.Tensor([16], "int32")),
    ):
        A = callback()
        B = R.multiply(A, R.const(2))
        return B

    ex = tvm.relax.build(
        tvm_tirx.IRModule.from_expr(relax_func),
        target=target,
    )
    vm = tvm.relax.VirtualMachine(ex, dev)

    np_A = np.arange(16, dtype="int32")

    def custom_callback():
        return tvm_tirx.runtime.tensor(np_A)

    output = vm["relax_func"](custom_callback)

    np.testing.assert_array_equal(np_A * 2, output.numpy())


def test_catch_exception_with_full_stack_trace():
    target = "llvm"
    dev = tvm_tirx.cpu()

    @R.function
    def relax_func(
        callback: R.Callable([], R.Tensor([16], "int32")),
    ):
        A = callback()
        return A

    ex = tvm.relax.build(
        tvm_tirx.IRModule.from_expr(relax_func),
        target=target,
    )
    vm = tvm.relax.VirtualMachine(ex, dev)

    # custom callback that raises an error in python
    def custom_callback():
        local_var = 42
        raise RuntimeError("Error thrown from callback")

    try:
        vm["relax_func"](custom_callback)
    except RuntimeError as err:
        stack = err.__traceback__
        while stack.tb_next is not None:
            stack = stack.tb_next
        frame = stack.tb_frame
        assert frame.f_code.co_filename.find("test_vm_callback_function.py") != -1, (
            "Inner-most stack frame should be from Python callback"
        )

    else:
        raise RuntimeError("Exception thrown in callback was not propagated to calling scope")


if __name__ == "__main__":
    tvm_tirx.testing.main()
